import Mathlib

namespace necessary_and_sufficient_condition_l1243_124337

-- Sum of the first n terms of the sequence
noncomputable def S_n (n : ℕ) (c : ℤ) : ℤ := (n + 1) * (n + 1) + c

-- The nth term of the sequence
noncomputable def a_n (n : ℕ) (c : ℤ) : ℤ := S_n n c - (S_n (n - 1) c)

-- Define the sequence being arithmetic
noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n+1) - a n = d

theorem necessary_and_sufficient_condition (c : ℤ) :
  (∀ n ≥ 1, a_n n c - a_n (n-1) c = 2) ↔ (c = -1) :=
by
  sorry

end necessary_and_sufficient_condition_l1243_124337


namespace product_of_abc_l1243_124346

variable (a b c m : ℚ)

-- Conditions
def condition1 : Prop := a + b + c = 200
def condition2 : Prop := 8 * a = m
def condition3 : Prop := m = b - 10
def condition4 : Prop := m = c + 10

-- The theorem to prove
theorem product_of_abc :
  a + b + c = 200 ∧ 8 * a = m ∧ m = b - 10 ∧ m = c + 10 →
  a * b * c = 505860000 / 4913 :=
by
  sorry

end product_of_abc_l1243_124346


namespace exists_same_color_points_at_unit_distance_l1243_124385

theorem exists_same_color_points_at_unit_distance
  (color : ℝ × ℝ → ℕ)
  (coloring : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end exists_same_color_points_at_unit_distance_l1243_124385


namespace ratio_of_amount_lost_l1243_124389

noncomputable def amount_lost (initial_amount spent_motorcycle spent_concert after_loss : ℕ) : ℕ :=
  let remaining_after_motorcycle := initial_amount - spent_motorcycle
  let remaining_after_concert := remaining_after_motorcycle / 2
  remaining_after_concert - after_loss

noncomputable def ratio (a b : ℕ) : ℕ × ℕ :=
  let g := Nat.gcd a b
  (a / g, b / g)

theorem ratio_of_amount_lost 
  (initial_amount spent_motorcycle spent_concert after_loss : ℕ)
  (h1 : initial_amount = 5000)
  (h2 : spent_motorcycle = 2800)
  (h3 : spent_concert = (initial_amount - spent_motorcycle) / 2)
  (h4 : after_loss = 825) :
  ratio (amount_lost initial_amount spent_motorcycle spent_concert after_loss)
        spent_concert = (1, 4) := by
  sorry

end ratio_of_amount_lost_l1243_124389


namespace c_is_younger_l1243_124381

variables (a b c d : ℕ) -- assuming ages as natural numbers

-- Conditions
axiom cond1 : a + b = b + c + 12
axiom cond2 : b + d = c + d + 8
axiom cond3 : d = a + 5

-- Question
theorem c_is_younger : c = a - 12 :=
sorry

end c_is_younger_l1243_124381


namespace survived_more_than_died_l1243_124356

-- Define the given conditions
def total_trees : ℕ := 13
def trees_died : ℕ := 6
def trees_survived : ℕ := total_trees - trees_died

-- The proof statement
theorem survived_more_than_died :
  trees_survived - trees_died = 1 := 
by
  -- This is where the proof would go
  sorry

end survived_more_than_died_l1243_124356


namespace sphere_radius_eq_cylinder_radius_l1243_124345

theorem sphere_radius_eq_cylinder_radius
  (r h d : ℝ) (h_eq_d : h = 16) (d_eq_h : d = 16)
  (sphere_surface_area_eq_cylinder : 4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h) : 
  r = 8 :=
by
  sorry

end sphere_radius_eq_cylinder_radius_l1243_124345


namespace second_cat_weight_l1243_124335

theorem second_cat_weight :
  ∀ (w1 w2 w3 w_total : ℕ), 
    w1 = 2 ∧ w3 = 4 ∧ w_total = 13 → 
    w_total = w1 + w2 + w3 → 
    w2 = 7 :=
by
  sorry

end second_cat_weight_l1243_124335


namespace factorial_divides_exponential_difference_l1243_124306

theorem factorial_divides_exponential_difference (n : ℕ) : n! ∣ 2^(2 * n!) - 2^n! :=
by
  sorry

end factorial_divides_exponential_difference_l1243_124306


namespace range_independent_variable_l1243_124352

theorem range_independent_variable (x : ℝ) (h : x + 1 > 0) : x > -1 :=
sorry

end range_independent_variable_l1243_124352


namespace surveyDSuitableForComprehensiveSurvey_l1243_124342

inductive Survey where
| A : Survey
| B : Survey
| C : Survey
| D : Survey

def isComprehensiveSurvey (s : Survey) : Prop :=
  match s with
  | Survey.A => False
  | Survey.B => False
  | Survey.C => False
  | Survey.D => True

theorem surveyDSuitableForComprehensiveSurvey : isComprehensiveSurvey Survey.D :=
by
  sorry

end surveyDSuitableForComprehensiveSurvey_l1243_124342


namespace cosine_product_inequality_l1243_124328

theorem cosine_product_inequality (A B C : ℝ) (h : A + B + C = Real.pi) :
  8 * Real.cos A * Real.cos B * Real.cos C ≤ 1 := 
sorry

end cosine_product_inequality_l1243_124328


namespace two_bacteria_fill_time_l1243_124388

-- Define the conditions
def one_bacterium_fills_bottle_in (a : Nat) (t : Nat) : Prop :=
  (2^t = 2^a)

def two_bacteria_fill_bottle_in (a : Nat) (x : Nat) : Prop :=
  (2 * 2^x = 2^a)

-- State the theorem
theorem two_bacteria_fill_time (a : Nat) : ∃ x, two_bacteria_fill_bottle_in a x ∧ x = a - 1 :=
by
  -- Use the given conditions
  sorry

end two_bacteria_fill_time_l1243_124388


namespace fixed_point_line_passes_through_range_of_t_l1243_124369

-- Definition for first condition: Line with slope k (k ≠ 0)
variables {k : ℝ} (hk : k ≠ 0)

-- Definition for second condition: Ellipse C
def ellipse_C (x y : ℝ) : Prop := (x^2) / 4 + y^2 = 1

-- Third condition: Intersections M and N
variables (M N : ℝ × ℝ)
variables (intersection_M : ellipse_C M.1 M.2)
variables (intersection_N : ellipse_C N.1 N.2)

-- Fourth condition: Slopes are k1 and k2
variables {k1 k2 : ℝ}
variables (hk1 : k1 = M.2 / M.1)
variables (hk2 : k2 = N.2 / N.1)

-- Fifth condition: Given equation 3(k1 + k2) = 8k
variables (h_eq : 3 * (k1 + k2) = 8 * k)

-- Proof for question 1: Line passes through a fixed point
theorem fixed_point_line_passes_through 
    (h_eq : 3 * (k1 + k2) = 8 * k) : 
    ∃ n : ℝ, n = 1/2 ∨ n = -1/2 := sorry

-- Additional conditions for question 2
variables {D : ℝ × ℝ} (hD : D = (1, 0))
variables (t : ℝ)
variables (area_ratio : (M.2 / N.2) = t)
variables (h_ineq : k^2 < 5 / 12)

-- Proof for question 2: Range for t
theorem range_of_t
    (hD : D = (1, 0))
    (area_ratio : (M.2 / N.2) = t)
    (h_ineq : k^2 < 5 / 12) : 
    2 < t ∧ t < 3 ∨ 1 / 3 < t ∧ t < 1 / 2 := sorry

end fixed_point_line_passes_through_range_of_t_l1243_124369


namespace melissa_total_cost_l1243_124320

-- Definitions based on conditions
def daily_rental_rate : ℝ := 15
def mileage_rate : ℝ := 0.10
def number_of_days : ℕ := 3
def number_of_miles : ℕ := 300

-- Theorem statement to prove the total cost
theorem melissa_total_cost : daily_rental_rate * number_of_days + mileage_rate * number_of_miles = 75 := 
by 
  sorry

end melissa_total_cost_l1243_124320


namespace divisors_of_90_l1243_124323

def num_pos_divisors (n : ℕ) : ℕ :=
  let factors := if n = 90 then [(2, 1), (3, 2), (5, 1)] else []
  factors.foldl (fun acc (p, k) => acc * (k + 1)) 1

theorem divisors_of_90 : num_pos_divisors 90 = 12 := by
  sorry

end divisors_of_90_l1243_124323


namespace find_angle_sum_l1243_124319

theorem find_angle_sum (c d : ℝ) (hc : 0 < c ∧ c < π/2) (hd : 0 < d ∧ d < π/2)
    (h1 : 4 * (Real.cos c)^2 + 3 * (Real.sin d)^2 = 1)
    (h2 : 4 * Real.sin (2 * c) = 3 * Real.cos (2 * d)) :
    2 * c + 3 * d = π / 2 :=
by
  sorry

end find_angle_sum_l1243_124319


namespace project_completion_l1243_124343

theorem project_completion (a b c d e : ℕ) 
  (h₁ : 1 / (a : ℝ) + 1 / b + 1 / c + 1 / d = 1 / 6)
  (h₂ : 1 / (b : ℝ) + 1 / c + 1 / d + 1 / e = 1 / 8)
  (h₃ : 1 / (a : ℝ) + 1 / e = 1 / 12) : 
  e = 48 :=
sorry

end project_completion_l1243_124343


namespace coefficient_of_x8y2_l1243_124367

theorem coefficient_of_x8y2 :
  let term1 := (1 / x^2)
  let term2 := (3 / y)
  let expansion := (x^2 - y)^7
  let coeff1 := 21 * (x ^ 10) * (y ^ 2) * (-1)
  let coeff2 := 35 * (3 / y) * (x ^ 8) * (y ^ 3)
  let comb := coeff1 + coeff2
  comb = -84 * x ^ 8 * y ^ 2 := by
  sorry

end coefficient_of_x8y2_l1243_124367


namespace base_area_of_cuboid_l1243_124350

theorem base_area_of_cuboid (V h : ℝ) (hv : V = 144) (hh : h = 8) : ∃ A : ℝ, A = 18 := by
  sorry

end base_area_of_cuboid_l1243_124350


namespace sum_of_roots_l1243_124329

variable {h b : ℝ}
variable {x₁ x₂ : ℝ}

-- Definition of the distinct property
def distinct (x₁ x₂ : ℝ) : Prop := x₁ ≠ x₂

-- Definition of the original equations given the conditions
def satisfies_equation (x : ℝ) (h b : ℝ) : Prop := 3 * x^2 - h * x = b

-- Main theorem statement translating the given mathematical problem
theorem sum_of_roots (h b : ℝ) (x₁ x₂ : ℝ) (h₁ : satisfies_equation x₁ h b) 
  (h₂ : satisfies_equation x₂ h b) (h₃ : distinct x₁ x₂) : x₁ + x₂ = h / 3 :=
sorry

end sum_of_roots_l1243_124329


namespace number_of_solutions_abs_eq_l1243_124365

theorem number_of_solutions_abs_eq (f : ℝ → ℝ) (g : ℝ → ℝ) : 
  (∀ x : ℝ, f x = |3 * x| ∧ g x = |x - 2| ∧ (f x + g x = 4) → 
  ∃! x1 x2 : ℝ, 
    ((0 < x1 ∧ x1 < 2 ∧ f x1 + g x1 = 4 ) ∨ 
    (x2 < 0 ∧ f x2 + g x2 = 4) ∧ x1 ≠ x2)) :=
by
  sorry

end number_of_solutions_abs_eq_l1243_124365


namespace swimmer_speed_in_still_water_l1243_124338

variable (distance : ℝ) (time : ℝ) (current_speed : ℝ) (swimmer_speed_still_water : ℝ)

-- Define the given conditions
def conditions := 
  distance = 8 ∧
  time = 5 ∧
  current_speed = 1.4 ∧
  (distance / time = swimmer_speed_still_water - current_speed)

-- The theorem we want to prove
theorem swimmer_speed_in_still_water : 
  conditions distance time current_speed swimmer_speed_still_water → 
  swimmer_speed_still_water = 3 := 
by 
  -- Skipping the actual proof
  sorry

end swimmer_speed_in_still_water_l1243_124338


namespace sum_of_arith_geo_progression_l1243_124397

noncomputable def sum_two_numbers (a b : ℝ) : ℝ :=
  a + b

theorem sum_of_arith_geo_progression : 
  ∃ (a b : ℝ), (∃ d : ℝ, a = 4 + d ∧ b = 4 + 2 * d) ∧ 
  (∃ r : ℝ, a * r = b ∧ b * r = 16) ∧ 
  sum_two_numbers a b = 8 + 6 * Real.sqrt 3 :=
by
  sorry

end sum_of_arith_geo_progression_l1243_124397


namespace perpendicular_vectors_x_value_l1243_124321

theorem perpendicular_vectors_x_value 
  (x : ℝ) 
  (a : ℝ × ℝ := (1, 2)) 
  (b : ℝ × ℝ := (x, -1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : x = 2 :=
by
  sorry

end perpendicular_vectors_x_value_l1243_124321


namespace maximize_profit_l1243_124392

theorem maximize_profit (x : ℤ) (hx : 20 ≤ x ∧ x ≤ 30) :
  (∀ y, 20 ≤ y ∧ y ≤ 30 → ((y - 20) * (30 - y)) ≤ ((25 - 20) * (30 - 25))) := 
sorry

end maximize_profit_l1243_124392


namespace sin_sides_of_triangle_l1243_124333

theorem sin_sides_of_triangle {a b c : ℝ} 
  (habc: a + b > c) (hbac: a + c > b) (hcbc: b + c > a) (h_sum: a + b + c ≤ 2 * Real.pi) :
  a > 0 ∧ a < Real.pi ∧ b > 0 ∧ b < Real.pi ∧ c > 0 ∧ c < Real.pi ∧ 
  (Real.sin a + Real.sin b > Real.sin c) ∧ 
  (Real.sin a + Real.sin c > Real.sin b) ∧ 
  (Real.sin b + Real.sin c > Real.sin a) :=
by
  sorry

end sin_sides_of_triangle_l1243_124333


namespace sufficient_condition_l1243_124390

theorem sufficient_condition (a : ℝ) (h : a ≥ 10) : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 → x^2 - a ≤ 0 :=
by
  sorry

end sufficient_condition_l1243_124390


namespace cr_inequality_l1243_124370

theorem cr_inequality 
  (a b : ℝ) (r : ℝ)
  (cr : ℝ := if r < 1 then 1 else 2^(r - 1)) 
  (h0 : r ≥ 0) : 
  |a + b|^r ≤ cr * (|a|^r + |b|^r) :=
by 
  sorry

end cr_inequality_l1243_124370


namespace granger_buys_3_jars_of_peanut_butter_l1243_124349

theorem granger_buys_3_jars_of_peanut_butter :
  ∀ (spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count: ℕ),
    spam_cost = 3 → peanut_butter_cost = 5 → bread_cost = 2 →
    spam_count = 12 → loaf_count = 4 → total_cost = 59 →
    spam_cost * spam_count + bread_cost * loaf_count + peanut_butter_cost * peanut_butter_count = total_cost →
    peanut_butter_count = 3 :=
by
  intros spam_cost peanut_butter_cost bread_cost total_cost spam_count loaf_count peanut_butter_count
  intros hspam_cost hpeanut_butter_cost hbread_cost hspam_count hloaf_count htotal_cost htotal
  sorry  -- The proof step is omitted as requested.

end granger_buys_3_jars_of_peanut_butter_l1243_124349


namespace fraction_invariant_l1243_124310

variable {R : Type*} [Field R]
variables (x y : R)

theorem fraction_invariant : (2 * x) / (3 * x - y) = (6 * x) / (9 * x - 3 * y) :=
by
  sorry

end fraction_invariant_l1243_124310


namespace max_value_of_determinant_l1243_124380

noncomputable def determinant_of_matrix (θ : ℝ) : ℝ :=
  Matrix.det ![
    ![1, 1, 1],
    ![1, 1 + Real.sin (2 * θ), 1],
    ![1, 1, 1 + Real.cos (2 * θ)]
  ]

theorem max_value_of_determinant : 
  ∃ θ : ℝ, (∀ θ : ℝ, determinant_of_matrix θ ≤ (1 / 2)) ∧ determinant_of_matrix (θ_at_maximum) = (1 / 2) :=
sorry

end max_value_of_determinant_l1243_124380


namespace georgie_ghost_ways_l1243_124316

-- Define the total number of windows and locked windows
def total_windows : ℕ := 8
def locked_windows : ℕ := 2

-- Define the number of usable windows
def usable_windows : ℕ := total_windows - locked_windows

-- Define the theorem to prove the number of ways Georgie the Ghost can enter and exit
theorem georgie_ghost_ways :
  usable_windows * (usable_windows - 1) = 30 := by
  sorry

end georgie_ghost_ways_l1243_124316


namespace permutation_and_combination_results_l1243_124334

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def A (n k : ℕ) : ℕ := factorial n / factorial (n - k)

def C (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem permutation_and_combination_results :
  A 5 2 = 20 ∧ C 6 3 + C 6 4 = 35 := by
  sorry

end permutation_and_combination_results_l1243_124334


namespace sqrt_81_eq_pm_9_l1243_124330

theorem sqrt_81_eq_pm_9 (x : ℤ) (hx : x^2 = 81) : x = 9 ∨ x = -9 :=
by
  sorry

end sqrt_81_eq_pm_9_l1243_124330


namespace total_supermarkets_FGH_chain_l1243_124354

variable (US_supermarkets : ℕ) (Canada_supermarkets : ℕ)
variable (total_supermarkets : ℕ)

-- Conditions
def condition1 := US_supermarkets = 37
def condition2 := US_supermarkets = Canada_supermarkets + 14

-- Goal
theorem total_supermarkets_FGH_chain
    (h1 : condition1 US_supermarkets)
    (h2 : condition2 US_supermarkets Canada_supermarkets) :
    total_supermarkets = US_supermarkets + Canada_supermarkets :=
sorry

end total_supermarkets_FGH_chain_l1243_124354


namespace consecutive_integers_sum_l1243_124355

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l1243_124355


namespace problem1_problem2_l1243_124301

-- Problem 1: Prove that \(\sqrt{27}+3\sqrt{\frac{1}{3}}-\sqrt{24} \times \sqrt{2} = 0\)
theorem problem1 : Real.sqrt 27 + 3 * Real.sqrt (1 / 3) - Real.sqrt 24 * Real.sqrt 2 = 0 := 
by sorry

-- Problem 2: Prove that \((\sqrt{5}-2)(2+\sqrt{5})-{(\sqrt{3}-1)}^{2} = -3 + 2\sqrt{3}\)
theorem problem2 : (Real.sqrt 5 - 2) * (2 + Real.sqrt 5) - (Real.sqrt 3 - 1) ^ 2 = -3 + 2 * Real.sqrt 3 := 
by sorry

end problem1_problem2_l1243_124301


namespace find_f3_value_l1243_124360

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := a * Real.tan x - b * x^5 + c * x - 3

theorem find_f3_value (a b c : ℝ) (h : f (-3) a b c = 7) : f 3 a b c = -13 := 
by 
  sorry

end find_f3_value_l1243_124360


namespace scissor_count_l1243_124302

theorem scissor_count :
  let initial_scissors := 54 
  let added_scissors := 22
  let removed_scissors := 15
  initial_scissors + added_scissors - removed_scissors = 61 := by
  sorry

end scissor_count_l1243_124302


namespace least_five_digit_congruent_l1243_124351

theorem least_five_digit_congruent (x : ℕ) (h1 : x ≥ 10000) (h2 : x < 100000) (h3 : x % 17 = 8) : x = 10004 :=
by {
  sorry
}

end least_five_digit_congruent_l1243_124351


namespace sum_third_column_l1243_124312

variable (a b c d e f g h i : ℕ)

theorem sum_third_column :
  (a + b + c = 24) →
  (d + e + f = 26) →
  (g + h + i = 40) →
  (a + d + g = 27) →
  (b + e + h = 20) →
  (c + f + i = 43) :=
by
  intros
  sorry

end sum_third_column_l1243_124312


namespace vitamin_supplement_problem_l1243_124344

theorem vitamin_supplement_problem :
  let packA := 7
  let packD := 17
  (∀ n : ℕ, n ≠ 0 → (packA * n = packD * n)) → n = 119 :=
by
  sorry

end vitamin_supplement_problem_l1243_124344


namespace harkamal_payment_l1243_124322

theorem harkamal_payment :
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  total_payment = 1125 :=
by
  let grapes_kg := 9
  let grape_rate_per_kg := 70
  let mangoes_kg := 9
  let mango_rate_per_kg := 55
  let cost_of_grapes := grapes_kg * grape_rate_per_kg
  let cost_of_mangoes := mangoes_kg * mango_rate_per_kg
  let total_payment := cost_of_grapes + cost_of_mangoes
  sorry

end harkamal_payment_l1243_124322


namespace frank_change_l1243_124318

theorem frank_change (n_c n_b money_given c_c c_b : ℕ) 
  (h1 : n_c = 5) 
  (h2 : n_b = 2) 
  (h3 : money_given = 20) 
  (h4 : c_c = 2) 
  (h5 : c_b = 3) : 
  money_given - (n_c * c_c + n_b * c_b) = 4 := 
by
  sorry

end frank_change_l1243_124318


namespace height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l1243_124379

-- Definitions
def Height (x : ℕ) : Prop := x = 140
def Weight (x : ℕ) : Prop := x = 23
def BookLength (x : ℕ) : Prop := x = 20
def BookThickness (x : ℕ) : Prop := x = 7
def CargoCapacity (x : ℕ) : Prop := x = 4
def SleepTime (x : ℕ) : Prop := x = 9
def TreeHeight (x : ℕ) : Prop := x = 12

-- Propositions
def XiaohongHeightUnit := "centimeters"
def XiaohongWeightUnit := "kilograms"
def MathBookLengthUnit := "centimeters"
def MathBookThicknessUnit := "millimeters"
def TruckCargoCapacityUnit := "tons"
def ChildrenSleepTimeUnit := "hours"
def BigTreeHeightUnit := "meters"

theorem height_is_centimeters (x : ℕ) (h : Height x) : XiaohongHeightUnit = "centimeters" := sorry
theorem weight_is_kilograms (x : ℕ) (w : Weight x) : XiaohongWeightUnit = "kilograms" := sorry
theorem book_length_is_centimeters (x : ℕ) (l : BookLength x) : MathBookLengthUnit = "centimeters" := sorry
theorem book_thickness_is_millimeters (x : ℕ) (t : BookThickness x) : MathBookThicknessUnit = "millimeters" := sorry
theorem cargo_capacity_is_tons (x : ℕ) (c : CargoCapacity x) : TruckCargoCapacityUnit = "tons" := sorry
theorem sleep_time_is_hours (x : ℕ) (s : SleepTime x) : ChildrenSleepTimeUnit = "hours" := sorry
theorem tree_height_is_meters (x : ℕ) (th : TreeHeight x) : BigTreeHeightUnit = "meters" := sorry

end height_is_centimeters_weight_is_kilograms_book_length_is_centimeters_book_thickness_is_millimeters_cargo_capacity_is_tons_sleep_time_is_hours_tree_height_is_meters_l1243_124379


namespace total_steps_correct_l1243_124393

/-- Definition of the initial number of steps on the first day --/
def steps_first_day : Nat := 200 + 300

/-- Definition of the number of steps on the second day --/
def steps_second_day : Nat := (3 / 2) * steps_first_day -- 1.5 is expressed as 3/2

/-- Definition of the number of steps on the third day --/
def steps_third_day : Nat := 2 * steps_second_day

/-- The total number of steps Eliana walked during the three days --/
def total_steps : Nat := steps_first_day + steps_second_day + steps_third_day

theorem total_steps_correct : total_steps = 2750 :=
  by
  -- provide the proof here
  sorry

end total_steps_correct_l1243_124393


namespace derivative_of_y_l1243_124378

variable (x : ℝ)

def y := x^3 + 3 * x^2 + 6 * x - 10

theorem derivative_of_y : (deriv y) x = 3 * x^2 + 6 * x + 6 :=
sorry

end derivative_of_y_l1243_124378


namespace four_m_plus_one_2013_eq_neg_one_l1243_124359

theorem four_m_plus_one_2013_eq_neg_one (m : ℝ) (h : |m| = m + 1) : (4 * m + 1) ^ 2013 = -1 := 
sorry

end four_m_plus_one_2013_eq_neg_one_l1243_124359


namespace value_of_X_l1243_124325

theorem value_of_X (X : ℝ) (h : ((X + 0.064)^2 - (X - 0.064)^2) / (X * 0.064) = 4.000000000000002) : X ≠ 0 :=
sorry

end value_of_X_l1243_124325


namespace find_n_l1243_124395

theorem find_n (n : ℕ) (M N : ℕ) (hM : M = 4 ^ n) (hN : N = 2 ^ n) (h : M - N = 992) : n = 5 :=
sorry

end find_n_l1243_124395


namespace books_combination_l1243_124371

theorem books_combination : (Nat.choose 15 3) = 455 := by
  sorry

end books_combination_l1243_124371


namespace virus_affected_computers_l1243_124364

theorem virus_affected_computers (m n : ℕ) (h1 : 5 * m + 2 * n = 52) : m = 8 :=
by
  sorry

end virus_affected_computers_l1243_124364


namespace expected_value_coin_flip_l1243_124387

-- Define the conditions
def probability_heads := 2 / 3
def probability_tails := 1 / 3
def gain_heads := 5
def loss_tails := -10

-- Define the expected value calculation
def expected_value := (probability_heads * gain_heads) + (probability_tails * loss_tails)

-- Prove that the expected value is 0.00
theorem expected_value_coin_flip : expected_value = 0 := 
by sorry

end expected_value_coin_flip_l1243_124387


namespace collinear_points_l1243_124339

-- Define collinear points function
def collinear (x1 y1 z1 x2 y2 z2 x3 y3 z3: ℝ) : Prop :=
  ∀ (a b c : ℝ), a * (y2 - y1) * (z3 - z1) + b * (z2 - z1) * (x3 - x1) + c * (x2 - x1) * (y3 - y1) = 0

-- Problem statement
theorem collinear_points (a b : ℝ)
  (h : collinear 2 a b a 3 b a b 4) :
  a + b = -2 :=
sorry

end collinear_points_l1243_124339


namespace Mary_younger_by_14_l1243_124391

variable (Betty_age : ℕ) (Albert_age : ℕ) (Mary_age : ℕ)

theorem Mary_younger_by_14 :
  (Betty_age = 7) →
  (Albert_age = 4 * Betty_age) →
  (Albert_age = 2 * Mary_age) →
  (Albert_age - Mary_age = 14) :=
by
  intros
  sorry

end Mary_younger_by_14_l1243_124391


namespace solution_set_inequality_l1243_124374

theorem solution_set_inequality (x : ℝ) : 4 * x^2 - 3 * x > 5 ↔ x < -5/4 ∨ x > 1 :=
by
  sorry

end solution_set_inequality_l1243_124374


namespace problem_S_equal_102_l1243_124373

-- Define the values in Lean
def S : ℕ := 1 * 3^1 + 2 * 3^2 + 3 * 3^3

-- Theorem to prove that S is equal to 102
theorem problem_S_equal_102 : S = 102 :=
by
  sorry

end problem_S_equal_102_l1243_124373


namespace cooking_time_eq_80_l1243_124304

-- Define the conditions
def hushpuppies_per_guest : Nat := 5
def number_of_guests : Nat := 20
def hushpuppies_per_batch : Nat := 10
def time_per_batch : Nat := 8

-- Calculate total number of hushpuppies needed
def total_hushpuppies : Nat := hushpuppies_per_guest * number_of_guests

-- Calculate number of batches needed
def number_of_batches : Nat := total_hushpuppies / hushpuppies_per_batch

-- Calculate total time needed
def total_time_needed : Nat := number_of_batches * time_per_batch

-- Statement to prove the correctness
theorem cooking_time_eq_80 : total_time_needed = 80 := by
  sorry

end cooking_time_eq_80_l1243_124304


namespace T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l1243_124303

def T (r n : ℕ) : ℕ :=
  sorry -- Define the function T_r(n) according to the problem's condition

-- Specific cases given in the problem statement
noncomputable def T_0_2006 : ℕ := T 0 2006
noncomputable def T_1_2006 : ℕ := T 1 2006
noncomputable def T_2_2006 : ℕ := T 2 2006

-- Theorems stating the result
theorem T_0_2006_correct : T_0_2006 = 1764 := sorry
theorem T_1_2006_correct : T_1_2006 = 122 := sorry
theorem T_2_2006_correct : T_2_2006 = 121 := sorry

end T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l1243_124303


namespace jessica_marbles_62_l1243_124332

-- Definitions based on conditions
def marbles_kurt (marbles_dennis : ℕ) : ℕ := marbles_dennis - 45
def marbles_laurie (marbles_kurt : ℕ) : ℕ := marbles_kurt + 12
def marbles_jessica (marbles_laurie : ℕ) : ℕ := marbles_laurie + 25

-- Given marbles for Dennis
def marbles_dennis : ℕ := 70

-- Proof statement: Prove that Jessica has 62 marbles given the conditions
theorem jessica_marbles_62 : marbles_jessica (marbles_laurie (marbles_kurt marbles_dennis)) = 62 := 
by
  sorry

end jessica_marbles_62_l1243_124332


namespace eighteenth_entry_of_sequence_l1243_124341

def r_7 (n : ℕ) : ℕ := n % 7

theorem eighteenth_entry_of_sequence : ∃ n : ℕ, (r_7 (4 * n) ≤ 3) ∧ (∀ m : ℕ, m < 18 → (r_7 (4 * m) ≤ 3) → m ≠ n) ∧ n = 30 := 
by 
  sorry

end eighteenth_entry_of_sequence_l1243_124341


namespace rent_increase_percentage_l1243_124399

theorem rent_increase_percentage (a x: ℝ) (h1: a ≠ 0) (h2: (9 / 10) * a = (4 / 5) * a * (1 + x / 100)) : x = 12.5 :=
sorry

end rent_increase_percentage_l1243_124399


namespace division_of_mixed_numbers_l1243_124326

noncomputable def mixed_to_improper (n : ℕ) (a b : ℕ) : ℚ :=
  n + (a / b)

theorem division_of_mixed_numbers : 
  (mixed_to_improper 7 1 3) / (mixed_to_improper 2 1 2) = 44 / 15 :=
by
  sorry

end division_of_mixed_numbers_l1243_124326


namespace tangent_line_to_ellipse_l1243_124311

theorem tangent_line_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 1 → x^2 + 4 * y^2 = 1 → (x^2 + 4 * (m * x + 1)^2 = 1)) →
  m^2 = 3 / 4 :=
by
  sorry

end tangent_line_to_ellipse_l1243_124311


namespace neg_p_iff_forall_l1243_124308

-- Define the proposition p
def p : Prop := ∃ (x : ℝ), x > 1 ∧ x^2 - 1 > 0

-- State the negation of p as a theorem
theorem neg_p_iff_forall : ¬ p ↔ ∀ (x : ℝ), x > 1 → x^2 - 1 ≤ 0 :=
by sorry

end neg_p_iff_forall_l1243_124308


namespace problem_statement_l1243_124383

def complex_number (m : ℂ) : ℂ :=
  (m^2 - 3*m - 4) + (m^2 - 5*m - 6) * Complex.I

theorem problem_statement (m : ℂ) :
  (complex_number m).im = m^2 - 5*m - 6 →
  (complex_number m).re = 0 →
  m ≠ -1 ∧ m ≠ 6 :=
by
  sorry

end problem_statement_l1243_124383


namespace parabola_value_l1243_124366

theorem parabola_value (b c : ℝ) (h : 3 = -(-2) ^ 2 + b * -2 + c) : 2 * c - 4 * b - 9 = 5 := by
  sorry

end parabola_value_l1243_124366


namespace ones_digit_of_8_pow_47_l1243_124327

theorem ones_digit_of_8_pow_47 : (8^47) % 10 = 2 := 
  sorry

end ones_digit_of_8_pow_47_l1243_124327


namespace p_iff_q_l1243_124362

variable (a b : ℝ)

def p := a > 2 ∧ b > 3

def q := a + b > 5 ∧ (a - 2) * (b - 3) > 0

theorem p_iff_q : p a b ↔ q a b := by
  sorry

end p_iff_q_l1243_124362


namespace arcsin_one_half_l1243_124348

theorem arcsin_one_half : Real.arcsin (1 / 2) = Real.pi / 6 :=
by
  sorry

end arcsin_one_half_l1243_124348


namespace calculate_expression_l1243_124361

theorem calculate_expression (a b c : ℕ) (h1 : a = 2011) (h2 : b = 2012) (h3 : c = 2013) :
  a^2 + b^2 + c^2 - a * b - b * c - c * a = 3 :=
by
  sorry

end calculate_expression_l1243_124361


namespace sum_of_ages_l1243_124372

variables (M A : ℕ)

def Maria_age_relation : Prop :=
  M = A + 8

def future_age_relation : Prop :=
  M + 10 = 3 * (A - 6)

theorem sum_of_ages (h₁ : Maria_age_relation M A) (h₂ : future_age_relation M A) : M + A = 44 :=
by
  sorry

end sum_of_ages_l1243_124372


namespace red_pairs_count_l1243_124324

theorem red_pairs_count (students_green : ℕ) (students_red : ℕ) (total_students : ℕ) (total_pairs : ℕ)
(pairs_green_green : ℕ) : 
students_green = 63 →
students_red = 69 →
total_students = 132 →
total_pairs = 66 →
pairs_green_green = 21 →
∃ (pairs_red_red : ℕ), pairs_red_red = 24 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end red_pairs_count_l1243_124324


namespace Sahil_transportation_charges_l1243_124314

theorem Sahil_transportation_charges
  (cost_machine : ℝ)
  (cost_repair : ℝ)
  (actual_selling_price : ℝ)
  (profit_percentage : ℝ)
  (transportation_charges : ℝ)
  (h1 : cost_machine = 12000)
  (h2 : cost_repair = 5000)
  (h3 : profit_percentage = 0.50)
  (h4 : actual_selling_price = 27000)
  (h5 : transportation_charges + (cost_machine + cost_repair) * (1 + profit_percentage) = actual_selling_price) :
  transportation_charges = 1500 :=
by
  sorry

end Sahil_transportation_charges_l1243_124314


namespace find_f_2005_1000_l1243_124347

-- Define the real-valued function and its properties
def f (x y : ℝ) : ℝ := sorry

-- The condition given in the problem
axiom condition :
  ∀ x y z : ℝ, f x y = f x z - 2 * f y z - 2 * z

-- The target we need to prove
theorem find_f_2005_1000 : f 2005 1000 = 5 := 
by 
  -- all necessary logical steps (detailed in solution) would go here
  sorry

end find_f_2005_1000_l1243_124347


namespace value_of_expression_is_one_l1243_124315

theorem value_of_expression_is_one : 
  ∃ (a b c d : ℚ), (a = 1) ∧ (b = -1) ∧ (c = 0) ∧ (d = 1 ∨ d = -1) ∧ (a - b + c^2 - |d| = 1) :=
by
  sorry

end value_of_expression_is_one_l1243_124315


namespace victor_total_money_l1243_124394

def initial_amount : ℕ := 10
def allowance : ℕ := 8
def total_amount : ℕ := initial_amount + allowance

theorem victor_total_money : total_amount = 18 := by
  -- This is where the proof steps would go
  sorry

end victor_total_money_l1243_124394


namespace tangent_point_value_l1243_124357

noncomputable def circle_tangent_problem :=
  let r1 := 3 -- radius of the first circle
  let r2 := 5 -- radius of the second circle
  let d := 12 -- distance between the centers of the circles
  ∃ (x : ℚ), (x / (d - x) = r1 / r2) → x = 9 / 2
  
theorem tangent_point_value : 
  circle_tangent_problem
:= sorry

end tangent_point_value_l1243_124357


namespace coeff_x20_greater_in_Q_l1243_124375

noncomputable def coeff (f : ℕ → ℕ → ℤ) (p x : ℤ) : ℤ :=
(x ^ 20) * p

noncomputable def P (x : ℤ) := (1 - x^2 + x^3) ^ 1000
noncomputable def Q (x : ℤ) := (1 + x^2 - x^3) ^ 1000

theorem coeff_x20_greater_in_Q :
  coeff 20 (Q x) x > coeff 20 (P x) x :=
  sorry

end coeff_x20_greater_in_Q_l1243_124375


namespace hyperbola_min_value_l1243_124353

def hyperbola_condition : Prop :=
  ∀ (m : ℝ), ∀ (x y : ℝ), (4 * x + 3 * y + m = 0 → (x^2 / 9 - y^2 / 16 = 1) → false)

noncomputable def minimum_value : ℝ :=
  2 * Real.sqrt 37 - 6

theorem hyperbola_min_value :
  hyperbola_condition → minimum_value =  2 * Real.sqrt 37 - 6 :=
by
  intro h
  sorry

end hyperbola_min_value_l1243_124353


namespace find_x_value_l1243_124307

open Real

theorem find_x_value (a : ℝ) (x : ℝ) (h : a > 0) (h_eq : 10^x = log (10 * a) + log (a⁻¹)) : x = 0 :=
by
  sorry

end find_x_value_l1243_124307


namespace parabola_x_intercepts_incorrect_l1243_124309

-- Define the given quadratic function
noncomputable def f (x : ℝ) : ℝ := -1 / 2 * (x - 1)^2 + 2

-- The Lean statement for the problem
theorem parabola_x_intercepts_incorrect :
  ¬ ((f 3 = 0) ∧ (f (-3) = 0)) :=
by
  sorry

end parabola_x_intercepts_incorrect_l1243_124309


namespace find_length_of_second_train_l1243_124398

def length_of_second_train (L : ℝ) : Prop :=
  let speed_first_train := 33.33 -- Speed in m/s
  let speed_second_train := 22.22 -- Speed in m/s
  let relative_speed := speed_first_train + speed_second_train -- Relative speed in m/s
  let time_to_cross := 9 -- time in seconds
  let length_first_train := 260 -- Length in meters
  length_first_train + L = relative_speed * time_to_cross

theorem find_length_of_second_train : length_of_second_train 239.95 :=
by
  admit -- To be completed (proof)

end find_length_of_second_train_l1243_124398


namespace correct_option_l1243_124336

-- Definitions representing the conditions
variable (a b c : Line) -- Define the lines a, b, and c

-- Conditions for the problem
def is_parallel (x y : Line) : Prop := -- Define parallel property
  sorry

def is_perpendicular (x y : Line) : Prop := -- Define perpendicular property
  sorry

noncomputable def proof_statement : Prop :=
  is_parallel a b → is_perpendicular a c → is_perpendicular b c

-- Lean statement of the proof problem
theorem correct_option (h1 : is_parallel a b) (h2 : is_perpendicular a c) : is_perpendicular b c :=
  sorry

end correct_option_l1243_124336


namespace cost_of_camel_l1243_124382

-- Define the cost of each animal as variables
variables (C H O E : ℝ)

-- Assume the given relationships as hypotheses
def ten_camels_eq_twentyfour_horses := (10 * C = 24 * H)
def sixteens_horses_eq_four_oxen := (16 * H = 4 * O)
def six_oxen_eq_four_elephants := (6 * O = 4 * E)
def ten_elephants_eq_140000 := (10 * E = 140000)

-- The theorem that we want to prove
theorem cost_of_camel (h1 : ten_camels_eq_twentyfour_horses C H)
                      (h2 : sixteens_horses_eq_four_oxen H O)
                      (h3 : six_oxen_eq_four_elephants O E)
                      (h4 : ten_elephants_eq_140000 E) :
  C = 5600 := sorry

end cost_of_camel_l1243_124382


namespace ethanol_percentage_fuel_B_l1243_124368

noncomputable def percentage_ethanol_in_fuel_B : ℝ :=
  let tank_capacity := 208
  let ethanol_in_fuelA := 0.12
  let total_ethanol := 30
  let volume_fuelA := 82
  let ethanol_from_fuelA := volume_fuelA * ethanol_in_fuelA
  let ethanol_from_fuelB := total_ethanol - ethanol_from_fuelA
  let volume_fuelB := tank_capacity - volume_fuelA
  (ethanol_from_fuelB / volume_fuelB) * 100

theorem ethanol_percentage_fuel_B :
  percentage_ethanol_in_fuel_B = 16 :=
by
  sorry

end ethanol_percentage_fuel_B_l1243_124368


namespace non_black_cows_l1243_124317

-- Define the main problem conditions
def total_cows : ℕ := 18
def black_cows : ℕ := (total_cows / 2) + 5

-- Statement to prove the number of non-black cows
theorem non_black_cows :
  total_cows - black_cows = 4 :=
by
  sorry

end non_black_cows_l1243_124317


namespace modulus_of_z_l1243_124384

open Complex

theorem modulus_of_z (z : ℂ) (h : z * ⟨0, 1⟩ = ⟨2, 1⟩) : abs z = Real.sqrt 5 :=
by
  sorry

end modulus_of_z_l1243_124384


namespace find_percentage_l1243_124386

theorem find_percentage (P N : ℝ) (h1 : (P / 100) * N = 60) (h2 : 0.80 * N = 240) : P = 20 :=
sorry

end find_percentage_l1243_124386


namespace bus_trip_distance_l1243_124377

-- Defining the problem variables
variables (x D : ℝ) -- x: speed in mph, D: total distance in miles

-- Main theorem stating the problem
theorem bus_trip_distance
  (h1 : 0 < x) -- speed of the bus is positive
  (h2 : (2 * x + 3 * (D - 2 * x) / (2 / 3 * x) / 2 + 0.75) - 2 - 4 = 0)
  -- The first scenario summarising the travel and delays
  (h3 : ((2 * x + 120) / x + 3 * (D - (2 * x + 120)) / (2 / 3 * x) / 2 + 0.75) - 3 = 0)
  -- The second scenario summarising the travel and delays; accident 120 miles further down
  : D = 720 := sorry

end bus_trip_distance_l1243_124377


namespace factorization_of_M_l1243_124340

theorem factorization_of_M :
  ∀ (x y z : ℝ), x^3 * (y - z) + y^3 * (z - x) + z^3 * (x - y) = 
  (x + y + z) * (x - y) * (y - z) * (z - x) := by
  sorry

end factorization_of_M_l1243_124340


namespace find_a_l1243_124305

-- Define the polynomial expansion term conditions
def binomial_coefficient (n k : ℕ) := Nat.choose n k

def fourth_term_coefficient (x a : ℝ) : ℝ :=
  binomial_coefficient 9 3 * x^6 * a^3

theorem find_a (a : ℝ) (x : ℝ) (h : fourth_term_coefficient x a = 84) : a = 1 :=
by
  unfold fourth_term_coefficient at h
  sorry

end find_a_l1243_124305


namespace gallons_in_pond_after_50_days_l1243_124331

def initial_amount : ℕ := 500
def evaporation_rate : ℕ := 1
def days_passed : ℕ := 50
def total_evaporation : ℕ := days_passed * evaporation_rate
def final_amount : ℕ := initial_amount - total_evaporation

theorem gallons_in_pond_after_50_days : final_amount = 450 := by
  sorry

end gallons_in_pond_after_50_days_l1243_124331


namespace mike_ride_equals_42_l1243_124300

-- Define the costs as per the conditions
def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie : ℝ := 2.50 + 5.00 + 0.25 * 22

-- State the theorem that needs to be proved
theorem mike_ride_equals_42 : ∃ M : ℕ, cost_mike M = cost_annie ∧ M = 42 :=
by
  sorry

end mike_ride_equals_42_l1243_124300


namespace compound_interest_two_years_l1243_124376

/-- Given the initial amount, and year-wise interest rates, 
     we want to find the amount in 2 years and prove it equals to a specific value. -/
theorem compound_interest_two_years 
  (P : ℝ) (R1 : ℝ) (R2 : ℝ) (T1 : ℝ) (T2 : ℝ) 
  (initial_amount : P = 7644) 
  (interest_rate_first_year : R1 = 0.04) 
  (interest_rate_second_year : R2 = 0.05) 
  (time_first_year : T1 = 1) 
  (time_second_year : T2 = 1) : 
  (P + (P * R1 * T1) + ((P + (P * R1 * T1)) * R2 * T2) = 8347.248) := 
by 
  sorry

end compound_interest_two_years_l1243_124376


namespace students_play_both_sports_l1243_124396

theorem students_play_both_sports 
  (total_students : ℕ) (students_play_football : ℕ) 
  (students_play_cricket : ℕ) (students_play_neither : ℕ) :
  total_students = 470 → students_play_football = 325 → 
  students_play_cricket = 175 → students_play_neither = 50 → 
  (students_play_football + students_play_cricket - 
    (total_students - students_play_neither)) = 80 :=
by
  intros h_total h_football h_cricket h_neither
  sorry

end students_play_both_sports_l1243_124396


namespace largest_constant_C_l1243_124313

theorem largest_constant_C :
  ∃ C : ℝ, 
    (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ C * (x + y + z - 1)) 
      ∧ (∀ D : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 2 ≥ D * (x + y + z - 1)) → C ≥ D)
    ∧ C = (2 + 2 * Real.sqrt 7) / 3 :=
sorry

end largest_constant_C_l1243_124313


namespace california_more_license_plates_l1243_124363

theorem california_more_license_plates :
  let CA_format := 26^4 * 10^2
  let NY_format := 26^3 * 10^3
  CA_format - NY_format = 28121600 := by
  let CA_format : Nat := 26^4 * 10^2
  let NY_format : Nat := 26^3 * 10^3
  have CA_plates : CA_format = 45697600 := by sorry
  have NY_plates : NY_format = 17576000 := by sorry
  calc
    CA_format - NY_format = 45697600 - 17576000 := by rw [CA_plates, NY_plates]
                    _ = 28121600 := by norm_num

end california_more_license_plates_l1243_124363


namespace find_q_l1243_124358

theorem find_q (p : ℝ) (q : ℝ) (h1 : p ≠ 0) (h2 : p = 4) (h3 : q ≠ 0) (avg_speed_eq : (2 * p * 3) / (p + 3) = 24 / q) : q = 7 := 
 by
  sorry

end find_q_l1243_124358
