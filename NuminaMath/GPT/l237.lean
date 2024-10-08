import Mathlib

namespace sum_last_two_digits_l237_237984

theorem sum_last_two_digits (h1 : 9 ^ 23 ≡ a [MOD 100]) (h2 : 11 ^ 23 ≡ b [MOD 100]) :
  (a + b) % 100 = 60 := 
  sorry

end sum_last_two_digits_l237_237984


namespace length_of_room_l237_237196

def area_of_room : ℝ := 10
def width_of_room : ℝ := 2

theorem length_of_room : width_of_room * 5 = area_of_room :=
by
  sorry

end length_of_room_l237_237196


namespace combined_eel_length_l237_237230

def Lengths : Type := { j : ℕ // j = 16 }

def jenna_eel_length : Lengths := ⟨16, rfl⟩

def bill_eel_length (j : Lengths) : ℕ := 3 * j.val

#check bill_eel_length

theorem combined_eel_length (j : Lengths) :
  j.val + bill_eel_length j = 64 :=
by
  -- The proof would go here
  sorry

end combined_eel_length_l237_237230


namespace balance_balls_l237_237831

variable (G B Y W P : ℝ)

-- Given conditions
def cond1 : 4 * G = 9 * B := sorry
def cond2 : 3 * Y = 8 * B := sorry
def cond3 : 7 * B = 5 * W := sorry
def cond4 : 4 * P = 10 * B := sorry

-- Theorem we need to prove
theorem balance_balls : 5 * G + 3 * Y + 3 * W + P = 26 * B :=
by
  -- skipping the proof
  sorry

end balance_balls_l237_237831


namespace avg_bc_eq_70_l237_237033

-- Definitions of the given conditions
variables (a b c : ℝ)

def avg_ab (a b : ℝ) : Prop := (a + b) / 2 = 45
def diff_ca (a c : ℝ) : Prop := c - a = 50

-- The main theorem statement
theorem avg_bc_eq_70 (h1 : avg_ab a b) (h2 : diff_ca a c) : (b + c) / 2 = 70 :=
by
  sorry

end avg_bc_eq_70_l237_237033


namespace maximum_acute_triangles_from_four_points_l237_237588

-- Define a point in a plane
structure Point (α : Type) := (x : α) (y : α)

-- Definition of an acute triangle is intrinsic to the problem
def is_acute_triangle {α : Type} [LinearOrderedField α] (A B C : Point α) : Prop :=
  sorry -- Assume implementation for determining if a triangle is acute angles based

def maximum_number_acute_triangles {α : Type} [LinearOrderedField α] (A B C D : Point α) : ℕ :=
  sorry -- Assume implementation for verifying maximum number of acute triangles from four points

theorem maximum_acute_triangles_from_four_points {α : Type} [LinearOrderedField α] (A B C D : Point α) :
  maximum_number_acute_triangles A B C D = 4 :=
  sorry

end maximum_acute_triangles_from_four_points_l237_237588


namespace roger_candies_left_l237_237438

theorem roger_candies_left (initial_candies : ℕ) (to_stephanie : ℕ) (to_john : ℕ) (to_emily : ℕ) : 
  initial_candies = 350 ∧ to_stephanie = 45 ∧ to_john = 25 ∧ to_emily = 18 → 
  initial_candies - (to_stephanie + to_john + to_emily) = 262 :=
by
  sorry

end roger_candies_left_l237_237438


namespace simplify_expression_l237_237642

theorem simplify_expression : ((3 + 4 + 5 + 6) / 3) + ((3 * 6 + 9) / 4) = 12.75 := by
  sorry

end simplify_expression_l237_237642


namespace solution_set_inequality_l237_237223

def custom_op (a b : ℝ) : ℝ := a * b + 2 * a + b

theorem solution_set_inequality : {x : ℝ | custom_op x (x - 2) < 0} = {x : ℝ | -2 < x ∧ x < 1} :=
by
  sorry

end solution_set_inequality_l237_237223


namespace initial_investment_l237_237992

theorem initial_investment
  (P r : ℝ)
  (h1 : P + (P * r * 2) / 100 = 600)
  (h2 : P + (P * r * 7) / 100 = 850) :
  P = 500 :=
sorry

end initial_investment_l237_237992


namespace cube_of_product_of_ab_l237_237982

theorem cube_of_product_of_ab (a b c : ℕ) (h1 : a * b * c = 180) (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) : (a * b) ^ 3 = 216 := 
sorry

end cube_of_product_of_ab_l237_237982


namespace solution_l237_237581

-- Define the equation
def equation (x : ℝ) := x^2 + 4*x + 3 + (x + 3)*(x + 5) = 0

-- State that x = -3 is a solution to the equation
theorem solution : equation (-3) :=
by
  unfold equation
  simp
  sorry

end solution_l237_237581


namespace find_borrowed_interest_rate_l237_237812

theorem find_borrowed_interest_rate :
  ∀ (principal : ℝ) (time : ℝ) (lend_rate : ℝ) (gain_per_year : ℝ) (borrow_rate : ℝ),
  principal = 5000 →
  time = 1 → -- Considering per year
  lend_rate = 0.06 →
  gain_per_year = 100 →
  (principal * lend_rate - gain_per_year = principal * borrow_rate * time) →
  borrow_rate * 100 = 4 :=
by
  intros principal time lend_rate gain_per_year borrow_rate h_principal h_time h_lend_rate h_gain h_equation
  rw [h_principal, h_time, h_lend_rate] at h_equation
  have h_borrow_rate := h_equation
  sorry

end find_borrowed_interest_rate_l237_237812


namespace element_in_set_l237_237491

theorem element_in_set (A : Set ℕ) (h : A = {1, 2}) : 1 ∈ A := 
by 
  rw[h]
  simp

end element_in_set_l237_237491


namespace largest_n_l237_237813

def canBeFactored (A B : ℤ) : Bool :=
  A * B = 54

theorem largest_n (n : ℤ) (h : ∃ (A B : ℤ), canBeFactored A B ∧ 3 * B + A = n) :
  n = 163 :=
by
  sorry

end largest_n_l237_237813


namespace fraction_equality_l237_237081

theorem fraction_equality (x : ℝ) (h : (4 + x) / (6 + x) = (2 + x) / (3 + x)) : x = 0 := 
sorry

end fraction_equality_l237_237081


namespace sin_arithmetic_sequence_l237_237641

theorem sin_arithmetic_sequence (a : ℝ) (h : 0 < a ∧ a < 2 * Real.pi) : 
  (Real.sin a + Real.sin (3 * a) = 2 * Real.sin (2 * a)) ↔ (a = Real.pi) :=
sorry

end sin_arithmetic_sequence_l237_237641


namespace trigonometric_inequality_l237_237198

theorem trigonometric_inequality (a b : ℝ) (ha : 0 < a ∧ a < Real.pi / 2) (hb : 0 < b ∧ b < Real.pi / 2) :
  5 / Real.cos a ^ 2 + 5 / (Real.sin a ^ 2 * Real.sin b ^ 2 * Real.cos b ^ 2) ≥ 27 * Real.cos a + 36 * Real.sin a :=
sorry

end trigonometric_inequality_l237_237198


namespace length_of_AE_l237_237235

/-- Given the conditions on the pentagon ABCDE:
1. AB = 2, BC = 2, CD = 5, DE = 7
2. AC is the largest side in triangle ABC
3. CE is the smallest side in triangle ECD
4. In triangle ACE all sides are integers and have distinct lengths,
prove that the length of side AE is 5. -/
theorem length_of_AE
  (AB BC CD DE : ℕ)
  (hAB : AB = 2)
  (hBC : BC = 2)
  (hCD : CD = 5)
  (hDE : DE = 7)
  (AC : ℕ) 
  (hAC_large : AB < AC ∧ BC < AC)
  (CE : ℕ)
  (hCE_small : CE < CD ∧ CE < DE)
  (AE : ℕ)
  (distinct_sides : ∀ x y z : ℕ, x ≠ y → x ≠ z → y ≠ z → (AC = x ∨ CE = x ∨ AE = x) → (AC = y ∨ CE = y ∨ AE = y) → (AC = z ∨ CE = z ∨ AE = z)) :
  AE = 5 :=
sorry

end length_of_AE_l237_237235


namespace symmetric_scanning_codes_count_l237_237036

-- Definition of a symmetric 8x8 scanning code grid under given conditions
def is_symmetric_code (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∀ i j : Fin 8, grid i j = grid (7 - i) (7 - j) ∧ grid i j = grid j i

def at_least_one_each_color (grid : Fin 8 → Fin 8 → Bool) : Prop :=
  ∃ i j k l : Fin 8, grid i j = true ∧ grid k l = false

def total_symmetric_scanning_codes : Nat :=
  1022

theorem symmetric_scanning_codes_count :
  ∀ (grid : Fin 8 → Fin 8 → Bool), is_symmetric_code grid ∧ at_least_one_each_color grid → 
  1022 = total_symmetric_scanning_codes :=
by
  sorry

end symmetric_scanning_codes_count_l237_237036


namespace cos_240_eq_neg_half_l237_237950

theorem cos_240_eq_neg_half : Real.cos (4 * Real.pi / 3) = -1/2 :=
by
  sorry

end cos_240_eq_neg_half_l237_237950


namespace alice_age_multiple_sum_l237_237268

theorem alice_age_multiple_sum (B : ℕ) (C : ℕ := 3) (A : ℕ := B + 2) (next_multiple_age : ℕ := A + (3 - (A % 3))) :
  B % C = 0 ∧ A = B + 2 ∧ C = 3 → 
  (next_multiple_age % 3 = 0 ∧
   (next_multiple_age / 10) + (next_multiple_age % 10) = 6) := 
by
  intros h
  sorry

end alice_age_multiple_sum_l237_237268


namespace possible_n_values_l237_237724

theorem possible_n_values (x y n : ℕ) (h1 : x > 0) (h2 : y > 0) (h3 : n > 0)
  (top_box_eq : x * y * n^2 = 720) :
  ∃ k : ℕ,  k = 6 :=
by 
  sorry

end possible_n_values_l237_237724


namespace number_of_teams_l237_237991

-- Define the statement representing the problem and conditions
theorem number_of_teams (n : ℕ) (h : 2 * n * (n - 1) = 9800) : n = 50 :=
sorry

end number_of_teams_l237_237991


namespace radius_of_circle_with_chords_l237_237241

theorem radius_of_circle_with_chords 
  (chord1_length : ℝ) (chord2_length : ℝ) (distance_between_midpoints : ℝ) 
  (h1 : chord1_length = 9) (h2 : chord2_length = 17) (h3 : distance_between_midpoints = 5) : 
  ∃ r : ℝ, r = 85 / 8 :=
by
  sorry

end radius_of_circle_with_chords_l237_237241


namespace find_max_n_l237_237099

variables {α : Type*} [LinearOrderedField α]

-- Define the sum S_n of the first n terms of an arithmetic sequence
noncomputable def S_n (a d : α) (n : ℕ) : α := 
  (n : α) / 2 * (2 * a + (n - 1) * d)

-- Given conditions
variable {a d : α}
axiom S11_pos : S_n a d 11 > 0
axiom S12_neg : S_n a d 12 < 0

theorem find_max_n : ∃ (n : ℕ), ∀ k < n, S_n a d k ≤ S_n a d n ∧ (k ≠ n → S_n a d k < S_n a d n) :=
sorry

end find_max_n_l237_237099


namespace part1_l237_237675

   noncomputable def sin_20_deg_sq : ℝ := (Real.sin (20 * Real.pi / 180))^2
   noncomputable def cos_80_deg_sq : ℝ := (Real.sin (10 * Real.pi / 180))^2
   noncomputable def sqrt3_sin20_cos80 : ℝ := Real.sqrt 3 * Real.sin (20 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)
   noncomputable def value : ℝ := sin_20_deg_sq + cos_80_deg_sq + sqrt3_sin20_cos80

   theorem part1 : value = 1 / 4 := by
     sorry
   
end part1_l237_237675


namespace friend_spent_more_l237_237168

theorem friend_spent_more (total_spent friend_spent: ℝ) (h_total: total_spent = 15) (h_friend: friend_spent = 10) :
  friend_spent - (total_spent - friend_spent) = 5 :=
by
  sorry

end friend_spent_more_l237_237168


namespace value_of_a_l237_237603

theorem value_of_a
  (a : ℝ)
  (h1 : ∀ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1)
  (h2 : ∀ (ρ : ℝ), ρ = a)
  (h3 : ∃ (ρ θ : ℝ), ρ * (Real.cos θ + Real.sin θ) = 1 ∧ ρ = a ∧ θ = 0)  :
  a = Real.sqrt 2 / 2 := 
sorry

end value_of_a_l237_237603


namespace stock_yield_percentage_l237_237953

noncomputable def FaceValue : ℝ := 100
noncomputable def AnnualYield : ℝ := 0.20 * FaceValue
noncomputable def MarketPrice : ℝ := 166.66666666666669
noncomputable def ExpectedYieldPercentage : ℝ := 12

theorem stock_yield_percentage :
  (AnnualYield / MarketPrice) * 100 = ExpectedYieldPercentage :=
by
  -- given conditions directly from the problem
  have h1 : FaceValue = 100 := rfl
  have h2 : AnnualYield = 0.20 * FaceValue := rfl
  have h3 : MarketPrice = 166.66666666666669 := rfl
  
  -- we are proving that the yield percentage is 12%
  sorry

end stock_yield_percentage_l237_237953


namespace distance_probability_at_least_sqrt2_over_2_l237_237484

noncomputable def prob_dist_at_least : ℝ := 
  let T := ((0,0), (1,0), (0,1))
  -- Assumes conditions incorporated through identifying two random points within the triangle T.
  let area_T : ℝ := 0.5
  let valid_area : ℝ := 0.5 - (Real.pi * (Real.sqrt 2 / 2)^2 / 8 + ((Real.sqrt 2 / 2)^2 / 2) / 2)
  valid_area / area_T

theorem distance_probability_at_least_sqrt2_over_2 :
  prob_dist_at_least = (4 - π) / 8 :=
by
  sorry

end distance_probability_at_least_sqrt2_over_2_l237_237484


namespace sale_in_third_month_l237_237371

theorem sale_in_third_month (s_1 s_2 s_4 s_5 s_6 : ℝ) (avg_sale : ℝ) (h1 : s_1 = 6435) (h2 : s_2 = 6927) (h4 : s_4 = 7230) (h5 : s_5 = 6562) (h6 : s_6 = 6191) (h_avg : avg_sale = 6700) :
  ∃ s_3 : ℝ, s_1 + s_2 + s_3 + s_4 + s_5 + s_6 = 6 * avg_sale ∧ s_3 = 6855 :=
by 
  sorry

end sale_in_third_month_l237_237371


namespace length_of_BA_is_sqrt_557_l237_237628

-- Define the given conditions
def AD : ℝ := 6
def DC : ℝ := 11
def CB : ℝ := 6
def AC : ℝ := 14

-- Define the theorem statement
theorem length_of_BA_is_sqrt_557 (x : ℝ) (H1 : AD = 6) (H2 : DC = 11) (H3 : CB = 6) (H4 : AC = 14) :
  x = Real.sqrt 557 :=
  sorry

end length_of_BA_is_sqrt_557_l237_237628


namespace tangent_line_through_points_of_tangency_l237_237942

noncomputable def equation_of_tangent_line (x1 y1 x y : ℝ) : Prop :=
x1 * x + (y1 - 2) * (y - 2) = 4

theorem tangent_line_through_points_of_tangency
  (x1 y1 x2 y2 : ℝ)
  (h1 : equation_of_tangent_line x1 y1 2 (-2))
  (h2 : equation_of_tangent_line x2 y2 2 (-2)) :
  (2 * x1 - 4 * (y1 - 2) = 4) ∧ (2 * x2 - 4 * (y2 - 2) = 4) →
  ∃ a b c, (a = 1) ∧ (b = -2) ∧ (c = 2) ∧ (a * x + b * y + c = 0) :=
by
  sorry

end tangent_line_through_points_of_tangency_l237_237942


namespace speed_of_policeman_l237_237231

theorem speed_of_policeman 
  (d_initial : ℝ) 
  (v_thief : ℝ) 
  (d_thief : ℝ)
  (d_policeman : ℝ)
  (h_initial : d_initial = 100) 
  (h_v_thief : v_thief = 8) 
  (h_d_thief : d_thief = 400) 
  (h_d_policeman : d_policeman = 500) 
  : ∃ (v_p : ℝ), v_p = 10 :=
by
  -- Use the provided conditions
  sorry

end speed_of_policeman_l237_237231


namespace point_in_first_quadrant_l237_237045

-- Define the imaginary unit i
def i := Complex.I

-- Define the complex number z
def z := i * (2 - i)

-- Define a predicate that checks if a complex number is in the first quadrant
def isFirstQuadrant (x : ℂ) : Prop := x.re > 0 ∧ x.im > 0

-- State the theorem
theorem point_in_first_quadrant : isFirstQuadrant z := sorry

end point_in_first_quadrant_l237_237045


namespace same_solution_eq_l237_237531

theorem same_solution_eq (a b : ℤ) (x y : ℤ) 
  (h₁ : 4 * x + 3 * y = 11)
  (h₂ : a * x + b * y = -2)
  (h₃ : 3 * x - 5 * y = 1)
  (h₄ : b * x - a * y = 6) :
  (a + b) ^ 2023 = 0 := by
  sorry

end same_solution_eq_l237_237531


namespace line_length_limit_l237_237702

theorem line_length_limit : 
  ∑' n : ℕ, 1 / ((3 : ℝ) ^ n) + (1 / (3 ^ (n + 1))) * (Real.sqrt 3) = (3 + Real.sqrt 3) / 2 :=
sorry

end line_length_limit_l237_237702


namespace domain_of_function_l237_237093

theorem domain_of_function :
  ∀ x : ℝ, (2 - x > 0) ∧ (2 * x + 1 > 0) ↔ (-1 / 2 < x) ∧ (x < 2) :=
sorry

end domain_of_function_l237_237093


namespace find_a2015_l237_237934

def seq (a : ℕ → ℕ) :=
  (a 1 = 1) ∧
  (a 2 = 4) ∧
  (a 3 = 9) ∧
  (∀ n, 4 ≤ n → a n = a (n-1) + a (n-2) - a (n-3))

theorem find_a2015 (a : ℕ → ℕ) (h_seq : seq a) : a 2015 = 8057 :=
sorry

end find_a2015_l237_237934


namespace problem_solution_set_l237_237244

noncomputable def f (x : ℝ) : ℝ :=
if x ≥ 0 then x^3 - 8 else (-x)^3 - 8

theorem problem_solution_set : 
  { x : ℝ | f (x-2) > 0 } = {x : ℝ | x < 0} ∪ {x : ℝ | x > 4} :=
by sorry

end problem_solution_set_l237_237244


namespace brenda_age_l237_237394

variables (A B J : ℝ)

-- Conditions
def condition1 : Prop := A = 4 * B
def condition2 : Prop := J = B + 7
def condition3 : Prop := A = J

-- Target to prove
theorem brenda_age (h1 : condition1 A B) (h2 : condition2 B J) (h3 : condition3 A J) : B = 7 / 3 :=
by
  sorry

end brenda_age_l237_237394


namespace correct_calculation_l237_237116

variable (a : ℝ) -- assuming a ∈ ℝ

theorem correct_calculation : (a ^ 3) ^ 2 = a ^ 6 :=
by {
  sorry
}

end correct_calculation_l237_237116


namespace polynomial_expansion_l237_237546

-- Definitions of the polynomials
def p (w : ℝ) : ℝ := 3 * w^3 + 4 * w^2 - 7
def q (w : ℝ) : ℝ := 2 * w^3 - 3 * w^2 + 1

-- Statement of the theorem
theorem polynomial_expansion (w : ℝ) : 
  (p w) * (q w) = 6 * w^6 - 6 * w^5 + 9 * w^3 + 12 * w^2 - 3 :=
by
  sorry

end polynomial_expansion_l237_237546


namespace laptop_weight_difference_is_3_67_l237_237732

noncomputable def karen_tote_weight : ℝ := 8
noncomputable def kevin_empty_briefcase_weight : ℝ := karen_tote_weight / 2
noncomputable def umbrella_weight : ℝ := kevin_empty_briefcase_weight / 2
noncomputable def briefcase_full_weight_rainy_day : ℝ := 2 * karen_tote_weight
noncomputable def work_papers_weight : ℝ := (briefcase_full_weight_rainy_day - umbrella_weight) / 6
noncomputable def laptop_weight : ℝ := briefcase_full_weight_rainy_day - umbrella_weight - work_papers_weight
noncomputable def weight_difference : ℝ := laptop_weight - karen_tote_weight

theorem laptop_weight_difference_is_3_67 : weight_difference = 3.67 := by
  sorry

end laptop_weight_difference_is_3_67_l237_237732


namespace division_addition_rational_eq_l237_237656

theorem division_addition_rational_eq :
  (3 / 7 / 4) + (1 / 2) = 17 / 28 :=
by
  sorry

end division_addition_rational_eq_l237_237656


namespace problem_l237_237979

-- Definition for condition 1
def condition1 (uniform_band : Prop) (appropriate_model : Prop) := 
  uniform_band → appropriate_model

-- Definition for condition 2
def condition2 (smaller_residual : Prop) (better_fit : Prop) :=
  smaller_residual → better_fit

-- Formal statement of the problem
theorem problem (uniform_band appropriate_model smaller_residual better_fit : Prop)
  (h1 : condition1 uniform_band appropriate_model)
  (h2 : condition2 smaller_residual better_fit)
  (h3 : uniform_band ∧ smaller_residual) :
  appropriate_model ∧ better_fit :=
  sorry

end problem_l237_237979


namespace pastries_calculation_l237_237842

theorem pastries_calculation 
    (G : ℕ) (C : ℕ) (P : ℕ) (F : ℕ)
    (hG : G = 30) 
    (hC : C = G - 5)
    (hP : P = G - 5)
    (htotal : C + P + F + G = 97) :
    C - F = 8 ∧ P - F = 8 :=
by
  sorry

end pastries_calculation_l237_237842


namespace behavior_of_g_l237_237905

def g (x : ℝ) : ℝ := -3 * x ^ 3 + 4 * x ^ 2 + 5

theorem behavior_of_g :
  (∀ x, (∃ M, x ≥ M → g x < 0)) ∧ (∀ x, (∃ N, x ≤ N → g x > 0)) :=
by
  sorry

end behavior_of_g_l237_237905


namespace all_are_multiples_of_3_l237_237893

theorem all_are_multiples_of_3 :
  (123 % 3 = 0) ∧
  (234 % 3 = 0) ∧
  (345 % 3 = 0) ∧
  (456 % 3 = 0) ∧
  (567 % 3 = 0) :=
by
  sorry

end all_are_multiples_of_3_l237_237893


namespace real_roots_of_quadratic_l237_237060

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

theorem real_roots_of_quadratic (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) ↔ m ≥ -1/4 := by
  sorry

end real_roots_of_quadratic_l237_237060


namespace mod_remainder_l237_237375

theorem mod_remainder (n : ℤ) (h : n % 5 = 3) : (4 * n - 5) % 5 = 2 := by
  sorry

end mod_remainder_l237_237375


namespace total_fencing_cost_is_5300_l237_237097

-- Define the conditions
def length_more_than_breadth_condition (l b : ℕ) := l = b + 40
def fencing_cost_per_meter : ℝ := 26.50
def given_length : ℕ := 70

-- Define the perimeter calculation
def perimeter (l b : ℕ) := 2 * l + 2 * b

-- Define the total cost calculation
def total_cost (P : ℕ) (cost_per_meter : ℝ) := P * cost_per_meter

-- State the theorem to be proven
theorem total_fencing_cost_is_5300 (b : ℕ) (l := given_length) :
  length_more_than_breadth_condition l b →
  total_cost (perimeter l b) fencing_cost_per_meter = 5300 :=
by
  sorry

end total_fencing_cost_is_5300_l237_237097


namespace average_of_distinct_s_values_l237_237420

theorem average_of_distinct_s_values : 
  (1 + 5 + 2 + 4 + 3 + 3 + 4 + 2 + 5 + 1) / 3 = 7.33 :=
by
  sorry

end average_of_distinct_s_values_l237_237420


namespace polynomial_value_at_2_l237_237671

def f (x : ℤ) : ℤ := 7 * x^7 + 6 * x^6 + 5 * x^5 + 4 * x^4 + 3 * x^3 + 2 * x^2 + x

theorem polynomial_value_at_2:
  f 2 = 1538 := by
  sorry

end polynomial_value_at_2_l237_237671


namespace number_of_young_fish_l237_237124

-- Define the conditions
def tanks : ℕ := 3
def pregnantFishPerTank : ℕ := 4
def youngPerFish : ℕ := 20

-- Define the proof problem
theorem number_of_young_fish : (tanks * pregnantFishPerTank * youngPerFish) = 240 := by
  sorry

end number_of_young_fish_l237_237124


namespace polynomial_abs_value_at_neg_one_l237_237630

theorem polynomial_abs_value_at_neg_one:
  ∃ g : Polynomial ℝ, 
  (∀ x ∈ ({0, 1, 2, 4, 5, 6} : Set ℝ), |g.eval x| = 15) → 
  |g.eval (-1)| = 75 :=
by
  sorry

end polynomial_abs_value_at_neg_one_l237_237630


namespace find_a1_general_term_sum_of_terms_l237_237734

-- Given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom h_condition : ∀ n, S n = (3 / 2) * a n - (1 / 2)

-- Specific condition for finding a1
axiom h_S1_eq_1 : S 1 = 1

-- Prove statements
theorem find_a1 : a 1 = 1 :=
by
  sorry

theorem general_term (n : ℕ) : n ≥ 1 → a n = 3 ^ (n - 1) :=
by
  sorry

theorem sum_of_terms (n : ℕ) : n ≥ 1 → S n = (3 ^ n - 1) / 2 :=
by
  sorry

end find_a1_general_term_sum_of_terms_l237_237734


namespace tadd_2019th_number_l237_237662

def next_start_point (n : ℕ) : ℕ := 
    1 + (n * (2 * 3 + (n - 1) * 9)) / 2

def block_size (n : ℕ) : ℕ := 
    1 + 3 * (n - 1)

def nth_number_said_by_tadd (n : ℕ) (k : ℕ) : ℕ :=
    let block_n := next_start_point n
    block_n + k - 1

theorem tadd_2019th_number :
    nth_number_said_by_tadd 37 2019 = 5979 := 
sorry

end tadd_2019th_number_l237_237662


namespace correct_conclusions_count_l237_237209

theorem correct_conclusions_count :
  (¬ (¬ p → (q ∨ r)) ↔ (¬ p → ¬ q ∧ ¬ r)) = false ∧
  ((¬ p → q) ↔ (p → ¬ q)) = false ∧
  (¬ ∃ n : ℕ, n > 0 ∧ (n ^ 2 + 3 * n) % 10 = 0 ∧ (∀ n : ℕ, n > 0 → (n ^ 2 + 3 * n) % 10 ≠ 0)) = true ∧
  (¬ ∀ x, x ^ 2 - 2 * x + 3 > 0 ∧ (∃ x, x ^ 2 - 2 * x + 3 < 0)) = false :=
by
  sorry

end correct_conclusions_count_l237_237209


namespace pictures_at_museum_l237_237619

-- Define the given conditions
def z : ℕ := 24
def k : ℕ := 14
def p : ℕ := 22

-- Define the number of pictures taken at the museum
def M : ℕ := 12

-- The theorem to be proven
theorem pictures_at_museum :
  z + M - k = p ↔ M = 12 :=
by
  sorry

end pictures_at_museum_l237_237619


namespace trajectory_equation_find_m_l237_237820

-- Define points A and B.
def A : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (-1, 0)

-- Define the condition for P:
def P_condition (P : ℝ × ℝ) : Prop :=
  let PA_len := Real.sqrt ((P.1 - 1)^2 + P.2^2)
  let AB_len := Real.sqrt ((1 - (-1))^2 + (0 - 0)^2)
  let PB_dot_AB := (P.1 + 1) * (-2)
  PA_len * AB_len = PB_dot_AB

-- Problem (1): The trajectory equation
theorem trajectory_equation (P : ℝ × ℝ) (hP : P_condition P) : P.2^2 = 4 * P.1 :=
sorry

-- Define orthogonality condition
def orthogonal (M N : ℝ × ℝ) : Prop := 
  let OM := M
  let ON := N
  OM.1 * ON.1 + OM.2 * ON.2 = 0

-- Problem (2): Finding the value of m
theorem find_m (m : ℝ) (hm1 : m ≠ 0) (hm2 : m < 1) 
  (P : ℝ × ℝ) (hP : P.2^2 = 4 * P.1)
  (M N : ℝ × ℝ) (hM : M.2 = M.1 + m) (hN : N.2 = N.1 + m)
  (hMN : orthogonal M N) : m = -4 :=
sorry

end trajectory_equation_find_m_l237_237820


namespace probability_of_3_black_2_white_l237_237249

def total_balls := 15
def black_balls := 10
def white_balls := 5
def drawn_balls := 5
def drawn_black_balls := 3
def drawn_white_balls := 2

noncomputable def probability_black_white_draw : ℝ :=
  (Nat.choose black_balls drawn_black_balls * Nat.choose white_balls drawn_white_balls : ℝ) /
  (Nat.choose total_balls drawn_balls : ℝ)

theorem probability_of_3_black_2_white :
  probability_black_white_draw = 400 / 1001 := by
  sorry

end probability_of_3_black_2_white_l237_237249


namespace simplify_expression_l237_237277

theorem simplify_expression (y : ℝ) :
  (18 * y^3) * (9 * y^2) * (1 / (6 * y)^2) = (9 / 2) * y^3 :=
by sorry

end simplify_expression_l237_237277


namespace gcd_computation_l237_237948

theorem gcd_computation (a b : ℕ) (h₁ : a = 7260) (h₂ : b = 540) : 
  Nat.gcd a b - 12 + 5 = 53 :=
by
  rw [h₁, h₂]
  sorry

end gcd_computation_l237_237948


namespace chicken_burger_cost_l237_237716

namespace BurgerCost

variables (C B : ℕ)

theorem chicken_burger_cost (h1 : B = C + 300) 
                            (h2 : 3 * B + 3 * C = 21000) : 
                            C = 3350 := 
sorry

end BurgerCost

end chicken_burger_cost_l237_237716


namespace quadrant_of_point_C_l237_237590

theorem quadrant_of_point_C
  (a b : ℝ)
  (h1 : -(a-2) = -1)
  (h2 : b+5 = 3) :
  a = 3 ∧ b = -2 ∧ 0 < a ∧ b < 0 :=
by {
  sorry
}

end quadrant_of_point_C_l237_237590


namespace average_first_8_matches_l237_237541

/--
Assume we have the following conditions:
1. The average score for 12 matches is 48 runs.
2. The average score for the last 4 matches is 64 runs.
Prove that the average score for the first 8 matches is 40 runs.
-/
theorem average_first_8_matches (A1 A2 : ℕ) :
  (A1 / 12 = 48) → 
  (A2 / 4 = 64) →
  ((A1 - A2) / 8 = 40) :=
by
  sorry

end average_first_8_matches_l237_237541


namespace cube_sum_identity_l237_237847

theorem cube_sum_identity (r : ℝ) (h : (r + 1/r)^2 = 5) : r^3 + 1/r^3 = 2 * Real.sqrt 5 ∨ r^3 + 1/r^3 = -2 * Real.sqrt 5 := by
  sorry

end cube_sum_identity_l237_237847


namespace yoongi_initial_books_l237_237232

theorem yoongi_initial_books 
  (Y E U : ℕ)
  (h1 : Y - 5 + 15 = 45)
  (h2 : E + 5 - 10 = 45)
  (h3 : U - 15 + 10 = 45) : 
  Y = 35 := 
by 
  -- To be completed with proof
  sorry

end yoongi_initial_books_l237_237232


namespace remainder_division_of_product_l237_237112

theorem remainder_division_of_product
  (h1 : 1225 % 12 = 1)
  (h2 : 1227 % 12 = 3) :
  ((1225 * 1227 * 1) % 12) = 3 :=
by
  sorry

end remainder_division_of_product_l237_237112


namespace distinct_solution_condition_l237_237912

theorem distinct_solution_condition (a : ℝ) : (∀ x1 x2 : ℝ, x1 ≠ x2 → ( x1^2 + 2 * x1 + 2 * |x1 + 1| = a ∧ x2^2 + 2 * x2 + 2 * |x2 + 1| = a )) ↔  a > -1 := 
by
  sorry

end distinct_solution_condition_l237_237912


namespace how_many_oranges_put_back_l237_237836

variables (A O x : ℕ)

-- Conditions: prices and initial selection.
def price_apple (A : ℕ) : ℕ := 40 * A
def price_orange (O : ℕ) : ℕ := 60 * O
def total_fruit := 20
def average_price_initial : ℕ := 56 -- Average price in cents

-- Conditions: equation from initial average price.
def total_initial_cost := total_fruit * average_price_initial
axiom initial_cost_eq : price_apple A + price_orange O = total_initial_cost
axiom total_fruit_eq : A + O = total_fruit

-- New conditions: desired average price and number of fruits
def average_price_new : ℕ := 52 -- Average price in cents
axiom new_cost_eq : price_apple A + price_orange (O - x) = (total_fruit - x) * average_price_new

-- The statement to be proven
theorem how_many_oranges_put_back : 40 * A + 60 * (O - 10) = (total_fruit - 10) * 52 → x = 10 :=
sorry

end how_many_oranges_put_back_l237_237836


namespace baron_munchausen_is_telling_truth_l237_237594

def digit_sum (n : ℕ) : ℕ :=
  (n.digits 10).sum

def is_10_digit (n : ℕ) : Prop :=
  10^9 ≤ n ∧ n < 10^10

def not_divisible_by_10 (n : ℕ) : Prop :=
  ¬(n % 10 = 0)

theorem baron_munchausen_is_telling_truth :
  ∃ a b : ℕ, a ≠ b ∧ is_10_digit a ∧ is_10_digit b ∧ not_divisible_by_10 a ∧ not_divisible_by_10 b ∧
  (a - digit_sum (a^2) = b - digit_sum (b^2)) := sorry

end baron_munchausen_is_telling_truth_l237_237594


namespace more_oranges_than_apples_l237_237835

def apples : ℕ := 14
def oranges : ℕ := 2 * 12

theorem more_oranges_than_apples : oranges - apples = 10 :=
by
  sorry

end more_oranges_than_apples_l237_237835


namespace max_sum_of_squares_eq_l237_237105

theorem max_sum_of_squares_eq (x y : ℝ) (h1 : x^2 + y^2 = 100) (h2 : x * y = 40) : 
  x + y ≤ 6 * Real.sqrt 5 := 
by
  sorry

end max_sum_of_squares_eq_l237_237105


namespace remainder_calculation_l237_237068

theorem remainder_calculation 
  (dividend divisor quotient : ℕ)
  (h1 : dividend = 140)
  (h2 : divisor = 15)
  (h3 : quotient = 9) :
  dividend = (divisor * quotient) + (dividend - (divisor * quotient)) := by
sorry

end remainder_calculation_l237_237068


namespace periodic_function_with_period_sqrt2_l237_237690

-- Definition of an even function
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- Definition of symmetry about x = sqrt(2)/2
def is_symmetric_about_line (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c - x) = f (c + x)

-- Main theorem to prove
theorem periodic_function_with_period_sqrt2 (f : ℝ → ℝ) :
  is_even_function f → is_symmetric_about_line f (Real.sqrt 2 / 2) → ∃ T, T = Real.sqrt 2 ∧ ∀ x, f (x + T) = f x :=
by
  sorry

end periodic_function_with_period_sqrt2_l237_237690


namespace largest_distance_between_spheres_l237_237940

theorem largest_distance_between_spheres :
  let O1 := (3, -14, 8)
  let O2 := (-9, 5, -12)
  let d := Real.sqrt ((3 + 9)^2 + (-14 - 5)^2 + (8 + 12)^2)
  let r1 := 24
  let r2 := 50
  r1 + d + r2 = Real.sqrt 905 + 74 :=
by
  intro O1 O2 d r1 r2
  sorry

end largest_distance_between_spheres_l237_237940


namespace solve_for_k_l237_237035

theorem solve_for_k :
  (∀ x : ℤ, (2 * x + 4 = 4 * (x - 2)) ↔ ( -x + 17 = 2 * x - 1 )) :=
by
  sorry

end solve_for_k_l237_237035


namespace ella_stamps_value_l237_237023

theorem ella_stamps_value :
  let total_stamps := 18
  let value_of_6_stamps := 18
  let consistent_value_per_stamp := value_of_6_stamps / 6
  total_stamps * consistent_value_per_stamp = 54 := by
  sorry

end ella_stamps_value_l237_237023


namespace arithmetic_mean_of_fractions_l237_237192

theorem arithmetic_mean_of_fractions :
  let a := (5 : ℚ) / 8
  let b := (9 : ℚ) / 16
  let c := (11 : ℚ) / 16
  a = (b + c) / 2 := by
  sorry

end arithmetic_mean_of_fractions_l237_237192


namespace sally_weekly_bread_l237_237874

-- Define the conditions
def monday_bread : Nat := 3
def tuesday_bread : Nat := 2
def wednesday_bread : Nat := 4
def thursday_bread : Nat := 2
def friday_bread : Nat := 1
def saturday_bread : Nat := 2 * 2  -- 2 sandwiches, 2 pieces each
def sunday_bread : Nat := 2

-- Define the total bread count
def total_bread : Nat := 
  monday_bread + 
  tuesday_bread + 
  wednesday_bread + 
  thursday_bread + 
  friday_bread + 
  saturday_bread + 
  sunday_bread

-- The proof statement
theorem sally_weekly_bread : total_bread = 18 := by
  sorry

end sally_weekly_bread_l237_237874


namespace find_y_from_x_squared_l237_237669

theorem find_y_from_x_squared (x y : ℤ) (h1 : x^2 = y + 7) (h2 : x = 6) : y = 29 :=
by
  sorry

end find_y_from_x_squared_l237_237669


namespace find_integer_l237_237526

theorem find_integer (n : ℤ) (h1 : n + 10 > 11) (h2 : -4 * n > -12) : 
  n = 2 :=
sorry

end find_integer_l237_237526


namespace power_mod_eq_one_l237_237939

theorem power_mod_eq_one (n : ℕ) (h₁ : 444 ≡ 3 [MOD 13]) (h₂ : 3^12 ≡ 1 [MOD 13]) :
  444^444 ≡ 1 [MOD 13] :=
by
  sorry

end power_mod_eq_one_l237_237939


namespace hyperbola_eccentricity_l237_237376

theorem hyperbola_eccentricity (a : ℝ) (h1 : a > 0) (h2 : 2 = (Real.sqrt (a^2 + 3)) / a) : a = 1 := 
by
  sorry

end hyperbola_eccentricity_l237_237376


namespace rita_bought_4_pounds_l237_237202

-- Define the conditions
def card_amount : ℝ := 70
def cost_per_pound : ℝ := 8.58
def amount_left : ℝ := 35.68

-- Define the theorem to prove the number of pounds of coffee bought is 4
theorem rita_bought_4_pounds :
  (card_amount - amount_left) / cost_per_pound = 4 := by sorry

end rita_bought_4_pounds_l237_237202


namespace maximize_profit_l237_237354

noncomputable def profit (x a : ℝ) : ℝ :=
  19 - 24 / (x + 2) - (3 / 2) * x

theorem maximize_profit (a : ℝ) (ha : 0 < a) :
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ a) ∧ 
  (if a ≥ 2 then x = 2 else x = a) :=
by
  sorry

end maximize_profit_l237_237354


namespace calculate_probability_two_cards_sum_to_15_l237_237851

-- Define the probability calculation as per the problem statement
noncomputable def probability_two_cards_sum_to_15 : ℚ :=
  let total_cards := 52
  let number_cards := 36 -- 9 values (2 through 10) each with 4 cards
  let card_combinations := (number_cards * (number_cards - 1)) / 2 -- Total pairs to choose from
  let favourable_combinations := 144 -- Manually calculated from cases in the solution
  favourable_combinations / card_combinations

theorem calculate_probability_two_cards_sum_to_15 :
  probability_two_cards_sum_to_15 = 8 / 221 :=
by
  -- Here we ignore the proof steps and directly state it assuming the provided assumption
  admit

end calculate_probability_two_cards_sum_to_15_l237_237851


namespace budget_spent_on_salaries_l237_237071

theorem budget_spent_on_salaries :
  ∀ (B R U E S T : ℕ),
  R = 9 ∧
  U = 5 ∧
  E = 4 ∧
  S = 2 ∧
  T = (72 * 100) / 360 → 
  B = 100 →
  (B - (R + U + E + S + T)) = 60 :=
by sorry

end budget_spent_on_salaries_l237_237071


namespace max_sum_of_distances_l237_237998

theorem max_sum_of_distances (x1 x2 y1 y2 : ℝ)
  (h1 : x1^2 + y1^2 = 1)
  (h2 : x2^2 + y2^2 = 1)
  (h3 : x1 * x2 + y1 * y2 = 1 / 2) :
  (|x1 + y1 - 1| / Real.sqrt 2 + |x2 + y2 - 1| / Real.sqrt 2) ≤ Real.sqrt 2 + Real.sqrt 3 :=
sorry

end max_sum_of_distances_l237_237998


namespace seeds_in_bucket_A_l237_237569

theorem seeds_in_bucket_A (A B C : ℕ) (h_total : A + B + C = 100) (h_B : B = 30) (h_C : C = 30) : A = 40 :=
by
  sorry

end seeds_in_bucket_A_l237_237569


namespace evaluate_fraction_eq_10_pow_10_l237_237764

noncomputable def evaluate_fraction (a b c : ℕ) : ℕ :=
  (a ^ 20) / ((a * b) ^ 10)

theorem evaluate_fraction_eq_10_pow_10 :
  evaluate_fraction 30 3 10 = 10 ^ 10 :=
by
  -- We define what is given and manipulate it directly to form a proof outline.
  sorry

end evaluate_fraction_eq_10_pow_10_l237_237764


namespace average_pages_correct_l237_237372

noncomputable def total_pages : ℝ := 50 + 75 + 80 + 120 + 100 + 90 + 110 + 130
def num_books : ℝ := 8
noncomputable def average_pages : ℝ := total_pages / num_books

theorem average_pages_correct : average_pages = 94.375 :=
by
  sorry

end average_pages_correct_l237_237372


namespace price_per_ticket_is_six_l237_237030

-- Definition of the conditions
def total_tickets (friends_tickets extra_tickets : ℕ) : ℕ :=
  friends_tickets + extra_tickets

def total_cost (tickets price_per_ticket : ℕ) : ℕ :=
  tickets * price_per_ticket

-- Given conditions
def friends_tickets : ℕ := 8
def extra_tickets : ℕ := 2
def total_spent : ℕ := 60

-- Formulate the problem to prove the price per ticket
theorem price_per_ticket_is_six :
  ∃ (price_per_ticket : ℕ), price_per_ticket = 6 ∧ 
  total_cost (total_tickets friends_tickets extra_tickets) price_per_ticket = total_spent :=
by
  -- The proof is not required; we assume its correctness here.
  sorry

end price_per_ticket_is_six_l237_237030


namespace rectangle_area_l237_237838

theorem rectangle_area (P : ℕ) (w : ℕ) (h : ℕ) (A : ℕ) 
  (hP : P = 28) 
  (hw : w = 6)
  (hW : P = 2 * (h + w)) 
  (hA : A = h * w) : 
  A = 48 :=
by
  sorry

end rectangle_area_l237_237838


namespace polygon_interior_angle_sum_l237_237158

theorem polygon_interior_angle_sum (n : ℕ) (h : (n - 2) * 180 = 1800) : n = 12 :=
by sorry

end polygon_interior_angle_sum_l237_237158


namespace arithmetic_sequence_second_term_l237_237161

theorem arithmetic_sequence_second_term (a1 a5 : ℝ) (h1 : a1 = 2020) (h5 : a5 = 4040) : 
  ∃ d a2 : ℝ, a2 = a1 + d ∧ d = (a5 - a1) / 4 ∧ a2 = 2525 :=
by
  sorry

end arithmetic_sequence_second_term_l237_237161


namespace english_score_l237_237250

theorem english_score (s1 s2 s3 e : ℕ) :
  (s1 + s2 + s3) = 276 → (s1 + s2 + s3 + e) = 376 → e = 100 :=
by
  intros h1 h2
  sorry

end english_score_l237_237250


namespace find_different_weighted_coins_l237_237118

-- Define the conditions and the theorem
def num_coins : Nat := 128
def weight_types : Nat := 2
def coins_of_each_weight : Nat := 64

theorem find_different_weighted_coins (weighings_at_most : Nat := 7) :
  ∃ (w1 w2 : Nat) (coins : Fin num_coins → Nat), w1 ≠ w2 ∧ 
  (∃ (pair : Fin num_coins × Fin num_coins), pair.fst ≠ pair.snd ∧ coins pair.fst ≠ coins pair.snd) :=
sorry

end find_different_weighted_coins_l237_237118


namespace toys_in_row_l237_237179

theorem toys_in_row (n_left n_right : ℕ) (hy : 10 = n_left + 1) (hy' : 7 = n_right + 1) :
  n_left + n_right + 1 = 16 :=
by
  -- Fill in the proof here
  sorry

end toys_in_row_l237_237179


namespace matt_homework_time_l237_237516

variable (T : ℝ)
variable (h_math : 0.30 * T = math_time)
variable (h_science : 0.40 * T = science_time)
variable (h_others : math_time + science_time + 45 = T)

theorem matt_homework_time (h_math : 0.30 * T = math_time)
                             (h_science : 0.40 * T = science_time)
                             (h_others : math_time + science_time + 45 = T) :
  T = 150 := by
  sorry

end matt_homework_time_l237_237516


namespace work_together_l237_237122

theorem work_together (A B : ℝ) (hA : A = 1/3) (hB : B = 1/6) : (1 / (A + B)) = 2 := by
  sorry

end work_together_l237_237122


namespace mrs_hilt_water_fountain_trips_l237_237255

theorem mrs_hilt_water_fountain_trips (d : ℕ) (t : ℕ) (n : ℕ) 
  (h1 : d = 30) 
  (h2 : t = 120) 
  (h3 : 2 * d * n = t) : 
  n = 2 :=
by
  -- Proof omitted
  sorry

end mrs_hilt_water_fountain_trips_l237_237255


namespace spencer_total_distance_l237_237300

-- Define the individual segments of Spencer's travel
def walk1 : ℝ := 1.2
def bike1 : ℝ := 1.8
def bus1 : ℝ := 3
def walk2 : ℝ := 0.4
def walk3 : ℝ := 0.6
def bike2 : ℝ := 2
def walk4 : ℝ := 1.5

-- Define the conversion factors
def bike_to_walk_conversion : ℝ := 0.5
def bus_to_walk_conversion : ℝ := 0.8

-- Calculate the total walking distance
def total_walking_distance : ℝ := walk1 + walk2 + walk3 + walk4

-- Calculate the total biking distance as walking equivalent
def total_biking_distance_as_walking : ℝ := (bike1 + bike2) * bike_to_walk_conversion

-- Calculate the total bus distance as walking equivalent
def total_bus_distance_as_walking : ℝ := bus1 * bus_to_walk_conversion

-- Define the total walking equivalent distance
def total_distance : ℝ := total_walking_distance + total_biking_distance_as_walking + total_bus_distance_as_walking

-- Theorem stating the total distance covered is 8 miles
theorem spencer_total_distance : total_distance = 8 := by
  unfold total_distance
  unfold total_walking_distance
  unfold total_biking_distance_as_walking
  unfold total_bus_distance_as_walking
  norm_num
  sorry

end spencer_total_distance_l237_237300


namespace zero_integers_in_range_such_that_expr_is_perfect_square_l237_237980

theorem zero_integers_in_range_such_that_expr_is_perfect_square :
  (∃ n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n ^ 2 + n + 2 = m ^ 2) → False :=
by sorry

end zero_integers_in_range_such_that_expr_is_perfect_square_l237_237980


namespace ali_spending_ratio_l237_237477

theorem ali_spending_ratio
  (initial_amount : ℝ := 480)
  (remaining_amount : ℝ := 160)
  (F : ℝ)
  (H1 : (initial_amount - F - (1/3) * (initial_amount - F) = remaining_amount))
  : (F / initial_amount) = 1 / 2 :=
by
  sorry

end ali_spending_ratio_l237_237477


namespace neither_sufficient_nor_necessary_l237_237520

theorem neither_sufficient_nor_necessary (a b : ℝ) (h1 : a ≠ 5) (h2 : b ≠ -5) : ¬((a + b ≠ 0) ↔ (a ≠ 5 ∧ b ≠ -5)) :=
by sorry

end neither_sufficient_nor_necessary_l237_237520


namespace kite_diagonals_sum_l237_237521

theorem kite_diagonals_sum (a b e f : ℝ) (h₁ : a ≥ b) 
    (h₂ : e < 2 * a) (h₃ : f < a + b) : 
    e + f < 2 * a + b := by 
    sorry

end kite_diagonals_sum_l237_237521


namespace ad_equals_two_l237_237088

noncomputable def geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

theorem ad_equals_two (a b c d : ℝ) 
  (h1 : geometric_sequence a b c d) 
  (h2 : ∃ (b c : ℝ), (1, 2) = (b, c) ∧ b = 1 ∧ c = 2) :
  a * d = 2 :=
by
  sorry

end ad_equals_two_l237_237088


namespace angles_identity_l237_237595
open Real

theorem angles_identity (α β : ℝ) (hα : 0 < α ∧ α < (π / 2)) (hβ : 0 < β ∧ β < (π / 2))
  (h1 : 3 * (sin α)^2 + 2 * (sin β)^2 = 1)
  (h2 : 3 * sin (2 * α) - 2 * sin (2 * β) = 0) :
  α + 2 * β = π / 2 :=
sorry

end angles_identity_l237_237595


namespace parity_expression_l237_237587

theorem parity_expression
  (a b c : ℕ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_odd : a % 2 = 1)
  (h_b_odd : b % 2 = 1) :
  (5^a + (b + 1)^2 * c) % 2 = 1 :=
by
  sorry

end parity_expression_l237_237587


namespace find_a_l237_237100

variable (f g : ℝ → ℝ) (a : ℝ)

-- Conditions
axiom h1 : ∀ x, f x = a^x * g x
axiom h2 : ∀ x, g x ≠ 0
axiom h3 : ∀ x, f x * (deriv g x) > (deriv f x) * g x

-- Question and target proof
theorem find_a (h4 : (f 1) / (g 1) + (f (-1)) / (g (-1)) = 5 / 2) : a = 1 / 2 :=
by sorry

end find_a_l237_237100


namespace plains_total_square_miles_l237_237157

theorem plains_total_square_miles (RegionB : ℝ) (h1 : RegionB = 200) (RegionA : ℝ) (h2 : RegionA = RegionB - 50) : 
  RegionA + RegionB = 350 := 
by 
  sorry

end plains_total_square_miles_l237_237157


namespace not_symmetric_about_point_l237_237623

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) + Real.log (4 - x)

theorem not_symmetric_about_point : ¬ (∀ h : ℝ, f (1 + h) = f (1 - h)) :=
by
  sorry

end not_symmetric_about_point_l237_237623


namespace simplify_expression_l237_237591

variable (x : ℝ)

theorem simplify_expression (x : ℝ) : 
  2 * (1 - (2 * (1 - (1 + (2 * (1 - x)))))) = 8 * x - 10 := 
by sorry

end simplify_expression_l237_237591


namespace range_of_a_l237_237775

variable (a : ℝ)

def p := ∀ x : ℝ, x^2 + a ≥ 0
def q := ∃ x : ℝ, x^2 + (2 + a) * x + 1 = 0

theorem range_of_a (h : p a ∧ q a) : 0 ≤ a :=
by
  sorry

end range_of_a_l237_237775


namespace no_solution_for_k_eq_2_l237_237213

theorem no_solution_for_k_eq_2 :
  ∀ m n : ℕ, m ≠ n → ¬ (lcm m n - gcd m n = 2 * (m - n)) :=
by
  sorry

end no_solution_for_k_eq_2_l237_237213


namespace larger_number_is_1891_l237_237843

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem larger_number_is_1891 :
  ∃ L S : ℕ, (L - S = 1355) ∧ (L = 6 * S + 15) ∧ is_prime (sum_of_digits L) ∧ sum_of_digits L ≠ 12
  :=
sorry

end larger_number_is_1891_l237_237843


namespace no_ratio_p_squared_l237_237955

theorem no_ratio_p_squared {p : ℕ} (hp : Nat.Prime p) :
  ∀ l n m : ℕ, 1 ≤ l → (∃ k : ℕ, k = p^l) → ((2 * (n*(n+1)) = (m*(m+1))*p^(2*l)) → false) := 
sorry

end no_ratio_p_squared_l237_237955


namespace valentines_proof_l237_237791

-- Definitions of the conditions in the problem
def original_valentines : ℝ := 58.5
def remaining_valentines : ℝ := 16.25
def valentines_given : ℝ := 42.25

-- The statement that we need to prove
theorem valentines_proof : original_valentines - remaining_valentines = valentines_given := by
  sorry

end valentines_proof_l237_237791


namespace jinho_remaining_money_l237_237586

def jinho_initial_money : ℕ := 2500
def cost_per_eraser : ℕ := 120
def erasers_bought : ℕ := 5
def cost_per_pencil : ℕ := 350
def pencils_bought : ℕ := 3

theorem jinho_remaining_money :
  jinho_initial_money - (erasers_bought * cost_per_eraser + pencils_bought * cost_per_pencil) = 850 :=
by
  sorry

end jinho_remaining_money_l237_237586


namespace no_real_solution_l237_237782

theorem no_real_solution (x : ℝ) : ¬ ∃ x : ℝ, (x - 5*x + 12)^2 + 1 = -|x| := 
sorry

end no_real_solution_l237_237782


namespace washer_cost_difference_l237_237193

theorem washer_cost_difference (W D : ℝ) 
  (h1 : W + D = 1200) (h2 : D = 490) : W - D = 220 :=
sorry

end washer_cost_difference_l237_237193


namespace part_b_part_c_l237_237303

-- Statement for part b: In how many ways can the figure be properly filled with the numbers from 1 to 5?
def proper_fill_count_1_to_5 : Nat :=
  8

-- Statement for part c: In how many ways can the figure be properly filled with the numbers from 1 to 7?
def proper_fill_count_1_to_7 : Nat :=
  48

theorem part_b :
  proper_fill_count_1_to_5 = 8 :=
sorry

theorem part_c :
  proper_fill_count_1_to_7 = 48 :=
sorry

end part_b_part_c_l237_237303


namespace trucks_initial_count_l237_237718

theorem trucks_initial_count (x : ℕ) (h : x - 13 = 38) : x = 51 :=
by sorry

end trucks_initial_count_l237_237718


namespace accounting_major_students_count_l237_237430

theorem accounting_major_students_count (p q r s: ℕ) (h1: p * q * r * s = 1365) (h2: 1 < p) (h3: p < q) (h4: q < r) (h5: r < s):
  p = 3 :=
sorry

end accounting_major_students_count_l237_237430


namespace polynomial_root_triples_l237_237846

theorem polynomial_root_triples (a b c : ℝ) :
  (∀ x : ℝ, x > 0 → (x^4 + a * x^3 + b * x^2 + c * x + b = 0)) ↔ (a, b, c) = (-21, 112, -204) ∨ (a, b, c) = (-12, 48, -80) :=
by
  sorry

end polynomial_root_triples_l237_237846


namespace kira_memory_space_is_140_l237_237949

def kira_songs_memory_space 
  (n_m : ℕ) -- number of songs downloaded in the morning
  (n_d : ℕ) -- number of songs downloaded later that day
  (n_n : ℕ) -- number of songs downloaded at night
  (s : ℕ) -- size of each song in MB
  : ℕ := (n_m + n_d + n_n) * s

theorem kira_memory_space_is_140 :
  kira_songs_memory_space 10 15 3 5 = 140 := 
by
  sorry

end kira_memory_space_is_140_l237_237949


namespace number_of_possible_flags_l237_237977

-- Define the number of colors available
def num_colors : ℕ := 3

-- Define the number of stripes on the flag
def num_stripes : ℕ := 3

-- Define the total number of possible flags
def total_flags : ℕ := num_colors ^ num_stripes

-- The statement we need to prove
theorem number_of_possible_flags : total_flags = 27 := by
  sorry

end number_of_possible_flags_l237_237977


namespace geom_sixth_term_is_31104_l237_237965

theorem geom_sixth_term_is_31104 :
  ∃ (r : ℝ), 4 * r^8 = 39366 ∧ 4 * r^(6-1) = 31104 :=
by
  sorry

end geom_sixth_term_is_31104_l237_237965


namespace sin_alpha_through_point_l237_237252

theorem sin_alpha_through_point (α : ℝ) (P : ℝ × ℝ) (hP : P = (-3, -Real.sqrt 3)) :
    Real.sin α = -1 / 2 :=
by
  sorry

end sin_alpha_through_point_l237_237252


namespace sandy_correct_sums_l237_237189

theorem sandy_correct_sums (c i : ℕ) (h1 : c + i = 30) (h2 : 3 * c - 2 * i = 45) : c = 21 :=
  sorry

end sandy_correct_sums_l237_237189


namespace trigonometric_identity_l237_237012

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
  (Real.sin (Real.pi - α) - Real.sin (Real.pi / 2 + α)) / 
  (Real.cos (3 * Real.pi / 2 - α) + 2 * Real.cos (-Real.pi + α)) = -2 / 5 := 
by
  sorry

end trigonometric_identity_l237_237012


namespace students_got_on_second_stop_l237_237972

-- Given conditions translated into definitions and hypotheses
def students_after_first_stop := 39
def students_after_second_stop := 68

-- The proof statement we aim to prove
theorem students_got_on_second_stop : (students_after_second_stop - students_after_first_stop) = 29 := by
  -- Proof goes here
  sorry

end students_got_on_second_stop_l237_237972


namespace wrong_conclusion_l237_237239

def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem wrong_conclusion {a b c : ℝ} (h₀ : a ≠ 0) (h₁ : 2 * a + b = 0) (h₂ : a + b + c = 3) (h₃ : 4 * a + 2 * b + c = 8) :
  quadratic a b c (-1) ≠ 0 :=
sorry

end wrong_conclusion_l237_237239


namespace factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l237_237464

-- Statement for question 1
theorem factorize_m4_minus_5m_plus_4 (m : ℤ) : 
  (m ^ 4 - 5 * m + 4) = (m ^ 4 - 5 * m + 4) := sorry

-- Statement for question 2
theorem factorize_x3_plus_2x2_plus_4x_plus_3 (x : ℝ) :
  (x ^ 3 + 2 * x ^ 2 + 4 * x + 3) = (x + 1) * (x ^ 2 + x + 3) := sorry

-- Statement for question 3
theorem factorize_x5_minus_1 (x : ℝ) :
  (x ^ 5 - 1) = (x - 1) * (x ^ 4 + x ^ 3 + x ^ 2 + x + 1) := sorry

end factorize_m4_minus_5m_plus_4_factorize_x3_plus_2x2_plus_4x_plus_3_factorize_x5_minus_1_l237_237464


namespace f_at_2_lt_e6_l237_237701

variable (f : ℝ → ℝ)

-- Specify the conditions
axiom derivable_f : Differentiable ℝ f
axiom condition_3f_gt_fpp : ∀ x : ℝ, 3 * f x > (deriv (deriv f)) x
axiom f_at_1 : f 1 = Real.exp 3

-- Conclusion to prove
theorem f_at_2_lt_e6 : f 2 < Real.exp 6 :=
sorry

end f_at_2_lt_e6_l237_237701


namespace min_acute_triangles_for_isosceles_l237_237981

noncomputable def isosceles_triangle_acute_division : ℕ :=
  sorry

theorem min_acute_triangles_for_isosceles {α : ℝ} (hα : α = 108) (isosceles : ∀ β γ : ℝ, β = γ) :
  isosceles_triangle_acute_division = 7 :=
sorry

end min_acute_triangles_for_isosceles_l237_237981


namespace negation_proposition_l237_237264

open Classical

variable (x : ℝ)

theorem negation_proposition :
  (¬ ∀ x : ℝ, x ≥ 0) ↔ (∃ x : ℝ, x < 0) :=
by
  sorry

end negation_proposition_l237_237264


namespace repeating_decimal_sum_l237_237563

theorem repeating_decimal_sum :
  (0.6666666666 : ℝ) + (0.7777777777 : ℝ) = (13 : ℚ) / 9 := by
  sorry

end repeating_decimal_sum_l237_237563


namespace polynomial_division_quotient_l237_237749

theorem polynomial_division_quotient :
  ∀ (x : ℝ), (x^5 - 21*x^3 + 8*x^2 - 17*x + 12) / (x - 3) = (x^4 + 3*x^3 - 12*x^2 - 28*x - 101) :=
by
  sorry

end polynomial_division_quotient_l237_237749


namespace total_staff_correct_l237_237452

noncomputable def total_staff_weekdays_weekends : ℕ := 84

theorem total_staff_correct :
  let chefs_weekdays := 16
  let waiters_weekdays := 16
  let busboys_weekdays := 10
  let hostesses_weekdays := 5
  let additional_chefs_weekends := 5
  let additional_hostesses_weekends := 2
  
  let chefs_leave := chefs_weekdays * 25 / 100
  let waiters_leave := waiters_weekdays * 20 / 100
  let busboys_leave := busboys_weekdays * 30 / 100
  let hostesses_leave := hostesses_weekdays * 15 / 100
  
  let chefs_left_weekdays := chefs_weekdays - chefs_leave
  let waiters_left_weekdays := waiters_weekdays - Nat.floor waiters_leave
  let busboys_left_weekdays := busboys_weekdays - busboys_leave
  let hostesses_left_weekdays := hostesses_weekdays - Nat.ceil hostesses_leave

  let total_staff_weekdays := chefs_left_weekdays + waiters_left_weekdays + busboys_left_weekdays + hostesses_left_weekdays

  let chefs_weekends := chefs_weekdays + additional_chefs_weekends
  let waiters_weekends := waiters_left_weekdays
  let busboys_weekends := busboys_left_weekdays
  let hostesses_weekends := hostesses_weekdays + additional_hostesses_weekends
  
  let total_staff_weekends := chefs_weekends + waiters_weekends + busboys_weekends + hostesses_weekends

  total_staff_weekdays + total_staff_weekends = total_staff_weekdays_weekends
:= by
  sorry

end total_staff_correct_l237_237452


namespace second_set_parallel_lines_l237_237487

theorem second_set_parallel_lines (n : ℕ) :
  (5 * (n - 1)) = 280 → n = 71 :=
by
  intros h
  sorry

end second_set_parallel_lines_l237_237487


namespace initial_pennies_l237_237767

theorem initial_pennies (P : ℕ)
  (h1 : P - (P / 2 + 1) = P / 2 - 1)
  (h2 : (P / 2 - 1) - (P / 4 + 1 / 2) = P / 4 - 3 / 2)
  (h3 : (P / 4 - 3 / 2) - (P / 8 + 3 / 4) = P / 8 - 9 / 4)
  (h4 : P / 8 - 9 / 4 = 1)
  : P = 26 := 
by
  sorry

end initial_pennies_l237_237767


namespace smallest_part_division_l237_237474

theorem smallest_part_division (S : ℚ) (P1 P2 P3 : ℚ) (total : ℚ) :
  (P1, P2, P3) = (1, 2, 3) →
  total = 64 →
  S = total / (P1 + P2 + P3) →
  S = 10 + 2/3 :=
by
  sorry

end smallest_part_division_l237_237474


namespace initial_lives_l237_237109

theorem initial_lives (x : ℕ) (h1 : x - 23 + 46 = 70) : x = 47 := 
by 
  sorry

end initial_lives_l237_237109


namespace AM_GM_HY_order_l237_237755

noncomputable def AM (a b c : ℝ) : ℝ := (a + b + c) / 3
noncomputable def GM (a b c : ℝ) : ℝ := (a * b * c)^(1/3)
noncomputable def HY (a b c : ℝ) : ℝ := 2 * a * b * c / (a * b + b * c + c * a)

theorem AM_GM_HY_order (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  AM a b c > GM a b c ∧ GM a b c > HY a b c := by
  sorry

end AM_GM_HY_order_l237_237755


namespace supplement_of_complement_of_65_l237_237609

def complement (angle : ℝ) : ℝ := 90 - angle
def supplement (angle : ℝ) : ℝ := 180 - angle

theorem supplement_of_complement_of_65 : supplement (complement 65) = 155 :=
by
  -- provide the proof steps here
  sorry

end supplement_of_complement_of_65_l237_237609


namespace num_students_third_class_num_students_second_class_l237_237070

-- Definition of conditions for both problems
def class_student_bounds (n : ℕ) : Prop := 40 < n ∧ n ≤ 50
def option_one_cost (n : ℕ) : ℕ := 40 * n * 7 / 10
def option_two_cost (n : ℕ) : ℕ := 40 * (n - 6) * 8 / 10

-- Problem Part 1
theorem num_students_third_class (x : ℕ) (h1 : class_student_bounds x) (h2 : option_one_cost x = option_two_cost x) : x = 48 := 
sorry

-- Problem Part 2
theorem num_students_second_class (y : ℕ) (h1 : class_student_bounds y) (h2 : option_one_cost y < option_two_cost y) : y = 49 ∨ y = 50 := 
sorry

end num_students_third_class_num_students_second_class_l237_237070


namespace larger_tent_fabric_amount_l237_237248

-- Define the fabric used for the small tent
def small_tent_fabric : ℝ := 4

-- Define the fabric computation for the larger tent
def larger_tent_fabric (small_tent_fabric : ℝ) : ℝ :=
  2 * small_tent_fabric

-- Theorem stating the amount of fabric needed for the larger tent
theorem larger_tent_fabric_amount : larger_tent_fabric small_tent_fabric = 8 :=
by
  -- Skip the actual proof
  sorry

end larger_tent_fabric_amount_l237_237248


namespace seashells_total_l237_237631

def seashells :=
  let sam_seashells := 18
  let mary_seashells := 47
  sam_seashells + mary_seashells

theorem seashells_total : seashells = 65 := by
  sorry

end seashells_total_l237_237631


namespace find_fraction_increase_l237_237460

noncomputable def present_value : ℝ := 64000
noncomputable def value_after_two_years : ℝ := 87111.11111111112

theorem find_fraction_increase (f : ℝ) :
  64000 * (1 + f) ^ 2 = 87111.11111111112 → f = 0.1666666666666667 := 
by
  intro h
  -- proof steps here
  sorry

end find_fraction_increase_l237_237460


namespace neighbors_have_even_total_bells_not_always_divisible_by_3_l237_237680

def num_bushes : ℕ := 19

def is_neighbor (circ : ℕ → ℕ) (i j : ℕ) : Prop := 
  if i = num_bushes - 1 then j = 0
  else j = i + 1

-- Part (a)
theorem neighbors_have_even_total_bells (bells : Fin num_bushes → ℕ) :
  ∃ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 2 = 0 := sorry

-- Part (b)
theorem not_always_divisible_by_3 (bells : Fin num_bushes → ℕ) :
  ¬ (∀ i : Fin num_bushes, (bells i + bells (⟨(i + 1) % num_bushes, sorry⟩ : Fin num_bushes)) % 3 = 0) := sorry

end neighbors_have_even_total_bells_not_always_divisible_by_3_l237_237680


namespace area_triangle_BQW_l237_237750

theorem area_triangle_BQW (ABCD : Rectangle) (AZ WC : ℝ) (AB : ℝ)
    (area_trapezoid_ZWCD : ℝ) :
    AZ = WC ∧ AZ = 6 ∧ AB = 12 ∧ area_trapezoid_ZWCD = 120 →
    (1/2) * ((120) - (1/2) * 6 * 12) = 42 :=
by
  intros
  sorry

end area_triangle_BQW_l237_237750


namespace triangle_is_isosceles_l237_237391

theorem triangle_is_isosceles (a b c : ℝ) (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC_sum : A + B + C = π) 
  (cos_rule : a * Real.cos B + b * Real.cos A = a) :
  a = c :=
by
  sorry

end triangle_is_isosceles_l237_237391


namespace solve_for_x2_minus_y2_minus_z2_l237_237556

theorem solve_for_x2_minus_y2_minus_z2
  (x y z : ℝ)
  (h1 : x + y + z = 12)
  (h2 : x - y = 4)
  (h3 : y + z = 7) :
  x^2 - y^2 - z^2 = -12 :=
by
  sorry

end solve_for_x2_minus_y2_minus_z2_l237_237556


namespace min_dist_l237_237322

open Real

theorem min_dist (a b : ℝ) :
  let A := (0, -1)
  let B := (1, 3)
  let C := (2, 6)
  let D := (0, b)
  let E := (1, a + b)
  let F := (2, 2 * a + b)
  let AD_sq := (b + 1) ^ 2
  let BE_sq := (a + b - 3) ^ 2
  let CF_sq := (2 * a + b - 6) ^ 2
  AD_sq + BE_sq + CF_sq = (b + 1) ^ 2 + (a + b - 3) ^ 2 + (2 * a + b - 6) ^ 2 → 
  a = 7 / 2 ∧ b = -5 / 6 :=
sorry

end min_dist_l237_237322


namespace first_group_number_l237_237777

variable (x : ℕ)

def number_of_first_group :=
  x = 6

theorem first_group_number (H1 : ∀ k : ℕ, k = 8 * 15 + x)
                          (H2 : k = 126) : 
                          number_of_first_group x :=
by
  sorry

end first_group_number_l237_237777


namespace jim_taxi_total_charge_l237_237434

noncomputable def total_charge (initial_fee : ℝ) (per_mile_fee : ℝ) (mile_chunk : ℝ) (distance : ℝ) : ℝ :=
  initial_fee + (distance / mile_chunk) * per_mile_fee

theorem jim_taxi_total_charge :
  total_charge 2.35 0.35 (2/5) 3.6 = 5.50 :=
by
  sorry

end jim_taxi_total_charge_l237_237434


namespace people_believing_mostly_purple_l237_237511

theorem people_believing_mostly_purple :
  ∀ (total : ℕ) (mostly_pink : ℕ) (both_mostly_pink_purple : ℕ) (neither : ℕ),
  total = 150 →
  mostly_pink = 80 →
  both_mostly_pink_purple = 40 →
  neither = 25 →
  (total - neither + both_mostly_pink_purple - mostly_pink) = 85 :=
by
  intros total mostly_pink both_mostly_pink_purple neither h_total h_mostly_pink h_both h_neither
  have people_identified_without_mostly_purple : ℕ := mostly_pink + both_mostly_pink_purple - mostly_pink + neither
  have leftover_people : ℕ := total - people_identified_without_mostly_purple
  have people_mostly_purple := both_mostly_pink_purple + leftover_people
  suffices people_mostly_purple = 85 by sorry
  sorry

end people_believing_mostly_purple_l237_237511


namespace probability_of_same_type_is_correct_l237_237044

noncomputable def total_socks : ℕ := 12 + 10 + 6
noncomputable def ways_to_pick_any_3_socks : ℕ := Nat.choose total_socks 3
noncomputable def ways_to_pick_3_black_socks : ℕ := Nat.choose 12 3
noncomputable def ways_to_pick_3_white_socks : ℕ := Nat.choose 10 3
noncomputable def ways_to_pick_3_striped_socks : ℕ := Nat.choose 6 3
noncomputable def ways_to_pick_3_same_type : ℕ := ways_to_pick_3_black_socks + ways_to_pick_3_white_socks + ways_to_pick_3_striped_socks
noncomputable def probability_same_type : ℚ := ways_to_pick_3_same_type / ways_to_pick_any_3_socks

theorem probability_of_same_type_is_correct :
  probability_same_type = 60 / 546 :=
by
  sorry

end probability_of_same_type_is_correct_l237_237044


namespace find_x_l237_237090

noncomputable def area_of_figure (x : ℝ) : ℝ :=
  let A_rectangle := 3 * x * 2 * x
  let A_square1 := x ^ 2
  let A_square2 := (4 * x) ^ 2
  let A_triangle := (3 * x * 2 * x) / 2
  A_rectangle + A_square1 + A_square2 + A_triangle

theorem find_x (x : ℝ) : area_of_figure x = 1250 → x = 6.93 :=
  sorry

end find_x_l237_237090


namespace find_original_number_l237_237503

theorem find_original_number (x : ℤ) (h : (x + 19) % 25 = 0) : x = 6 :=
sorry

end find_original_number_l237_237503


namespace probability_red_in_both_jars_l237_237966

def original_red_buttons : ℕ := 6
def original_blue_buttons : ℕ := 10
def total_original_buttons : ℕ := original_red_buttons + original_blue_buttons
def remaining_buttons : ℕ := (2 * total_original_buttons) / 3
def moved_buttons : ℕ := total_original_buttons - remaining_buttons
def moved_red_buttons : ℕ := 2
def moved_blue_buttons : ℕ := 3

theorem probability_red_in_both_jars :
  moved_red_buttons = moved_blue_buttons →
  remaining_buttons = 11 →
  (∃ m n : ℚ, m / remaining_buttons = 4 / 11 ∧ n / (moved_red_buttons + moved_blue_buttons) = 2 / 5 ∧ (m / remaining_buttons) * (n / (moved_red_buttons + moved_blue_buttons)) = 8 / 55) :=
by sorry

end probability_red_in_both_jars_l237_237966


namespace triangle_side_length_l237_237451

-- Definitions based on problem conditions
variables {A B C D : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

variables (AC BC AD AB CD : ℝ)

-- Conditions from the problem
axiom h1 : BC = 2 * AC
axiom h2 : AD = (1 / 3) * AB

-- Theorem statement to be proved
theorem triangle_side_length (h1 : BC = 2 * AC) (h2 : AD = (1 / 3) * AB) : CD = 2 * AD :=
sorry

end triangle_side_length_l237_237451


namespace calculate_wheel_radii_l237_237009

theorem calculate_wheel_radii (rpmA rpmB : ℕ) (length : ℝ) (r R : ℝ) :
  rpmA = 1200 →
  rpmB = 1500 →
  length = 9 →
  (4 : ℝ) / 5 * r = R →
  2 * (R + r) = 9 →
  r = 2 ∧ R = 2.5 :=
by
  intros
  sorry

end calculate_wheel_radii_l237_237009


namespace clownfish_ratio_l237_237829

theorem clownfish_ratio (C B : ℕ) (h₁ : C = B) (h₂ : C + B = 100) (h₃ : C = B) : 
  (let B := 50; 
  let initially_clownfish := B - 26; -- Number of clownfish that initially joined display tank
  let swam_back := (B - 26) - 16; -- Number of clownfish that swam back
  initially_clownfish > 0 → 
  swam_back > 0 → 
  (swam_back : ℚ) / (initially_clownfish : ℚ) = 1 / 3) :=
by 
  sorry

end clownfish_ratio_l237_237829


namespace dice_sum_probability_l237_237405

def four_dice_probability_sum_to_remain_die : ℚ :=
  let total_outcomes : ℚ := 6^4
  let favorable_outcomes : ℚ := 4 * 120
  favorable_outcomes / total_outcomes

theorem dice_sum_probability : four_dice_probability_sum_to_remain_die = 10 / 27 :=
  sorry

end dice_sum_probability_l237_237405


namespace time_spent_per_piece_l237_237871

-- Conditions
def number_of_chairs : ℕ := 7
def number_of_tables : ℕ := 3
def total_furniture : ℕ := number_of_chairs + number_of_tables
def total_time_spent : ℕ := 40

-- Proof statement
theorem time_spent_per_piece : total_time_spent / total_furniture = 4 :=
by
  -- Proof goes here
  sorry

end time_spent_per_piece_l237_237871


namespace sequence_solution_l237_237672

theorem sequence_solution (n : ℕ) (S : ℕ → ℕ) (a : ℕ → ℕ)
  (h : ∀ n, S n = 2 * a n - 2^n + 1) : a n = n * 2^(n-1) :=
sorry

end sequence_solution_l237_237672


namespace product_of_distinct_nonzero_real_numbers_l237_237098

variable {x y : ℝ}

theorem product_of_distinct_nonzero_real_numbers (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) (h : x + 4 / x = y + 4 / y) : x * y = 4 := 
sorry

end product_of_distinct_nonzero_real_numbers_l237_237098


namespace evaluate_propositions_l237_237148

variable (x y : ℝ)

def p : Prop := (x > y) → (-x < -y)
def q : Prop := (x < y) → (x^2 > y^2)

theorem evaluate_propositions : (p x y ∨ q x y) ∧ (p x y ∧ ¬q x y) := by
  -- Correct answer: \( \boxed{\text{C}} \)
  sorry

end evaluate_propositions_l237_237148


namespace linda_coats_l237_237826

variable (wall_area : ℝ) (cover_per_gallon : ℝ) (gallons_bought : ℝ)

theorem linda_coats (h1 : wall_area = 600)
                    (h2 : cover_per_gallon = 400)
                    (h3 : gallons_bought = 3) :
  (gallons_bought / (wall_area / cover_per_gallon)) = 2 :=
by
  sorry

end linda_coats_l237_237826


namespace find_k_and_other_root_l237_237627

def quadratic_eq (a b c x : ℝ) : Prop := a * x^2 + b * x + c = 0

theorem find_k_and_other_root (k β : ℝ) (h1 : quadratic_eq 4 k 2 (-0.5)) (h2 : 4 * (-0.5) ^ 2 + k * (-0.5) + 2 = 0) : 
  k = 6 ∧ β = -1 ∧ quadratic_eq 4 k 2 β := 
by 
  sorry

end find_k_and_other_root_l237_237627


namespace factorization_difference_of_squares_l237_237698

theorem factorization_difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) :=
by
  -- The proof will go here.
  sorry

end factorization_difference_of_squares_l237_237698


namespace remainder_of_x_mod_10_l237_237470

def x : ℕ := 2007 ^ 2008

theorem remainder_of_x_mod_10 : x % 10 = 1 := by
  sorry

end remainder_of_x_mod_10_l237_237470


namespace books_finished_correct_l237_237518

def miles_traveled : ℕ := 6760
def miles_per_book : ℕ := 450
def books_finished (miles_traveled miles_per_book : ℕ) : ℕ :=
  miles_traveled / miles_per_book

theorem books_finished_correct :
  books_finished miles_traveled miles_per_book = 15 :=
by
  -- The steps of the proof would go here
  sorry

end books_finished_correct_l237_237518


namespace find_d_not_unique_solution_l237_237427

variable {x y k d : ℝ}

-- Definitions of the conditions
def eq1 (d : ℝ) (x y : ℝ) := 4 * (3 * x + 4 * y) = d
def eq2 (k : ℝ) (x y : ℝ) := k * x + 12 * y = 30

-- The theorem we need to prove
theorem find_d_not_unique_solution (h1: eq1 d x y) (h2: eq2 k x y) (h3 : ¬ ∃! (x y : ℝ), eq1 d x y ∧ eq2 k x y) : d = 40 := 
by
  sorry

end find_d_not_unique_solution_l237_237427


namespace find_fixed_monthly_fee_l237_237817

noncomputable def fixed_monthly_fee (f h : ℝ) (february_bill march_bill : ℝ) : Prop :=
  (f + h = february_bill) ∧ (f + 3 * h = march_bill)

theorem find_fixed_monthly_fee (h : ℝ):
  fixed_monthly_fee 13.44 h 20.72 35.28 :=
by 
  sorry

end find_fixed_monthly_fee_l237_237817


namespace time_to_school_l237_237395

theorem time_to_school (total_distance walk_speed run_speed distance_ran : ℕ) (h_total : total_distance = 1800)
    (h_walk_speed : walk_speed = 70) (h_run_speed : run_speed = 210) (h_distance_ran : distance_ran = 600) :
    total_distance / walk_speed + distance_ran / run_speed = 20 := by
  sorry

end time_to_school_l237_237395


namespace length_of_one_side_of_regular_octagon_l237_237120

theorem length_of_one_side_of_regular_octagon
  (a b : ℕ)
  (h_pentagon : a = 16)   -- Side length of regular pentagon
  (h_total_yarn_pentagon : b = 80)  -- Total yarn for pentagon
  (hpentagon_yarn_length : 5 * a = b)  -- Total yarn condition
  (hoctagon_total_sides : 8 = 8)   -- Number of sides of octagon
  (hoctagon_side_length : 10 = b / 8)  -- Side length condition for octagon
  : 10 = 10 :=
by
  sorry

end length_of_one_side_of_regular_octagon_l237_237120


namespace boats_equation_correct_l237_237174

theorem boats_equation_correct (x : ℕ) (h1 : x ≤ 8) (h2 : 4 * x + 6 * (8 - x) = 38) : 
    4 * x + 6 * (8 - x) = 38 :=
by
  sorry

end boats_equation_correct_l237_237174


namespace percentage_of_40_eq_140_l237_237162

theorem percentage_of_40_eq_140 (p : ℝ) (h : (p / 100) * 40 = 140) : p = 350 :=
sorry

end percentage_of_40_eq_140_l237_237162


namespace find_m_value_l237_237960

theorem find_m_value
  (m : ℝ)
  (h1 : 10 - m > 0)
  (h2 : m - 2 > 0)
  (h3 : 2 * Real.sqrt (10 - m - (m - 2)) = 4) :
  m = 4 := by
sorry

end find_m_value_l237_237960


namespace staircase_steps_l237_237859

theorem staircase_steps (x : ℕ) (h1 : x + 2 * x + (2 * x - 10) = 2 * 45) : x = 20 :=
by 
  -- The proof is skipped
  sorry

end staircase_steps_l237_237859


namespace circle_equation_l237_237944

theorem circle_equation (x y : ℝ) : (x^2 = 16 * y) → (y = 4) → (x, -4) = (x, 4) → x^2 + (y-4)^2 = 64 :=
by
  sorry

end circle_equation_l237_237944


namespace intersection_A_B_l237_237251

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {0, 2, 4, 6}

theorem intersection_A_B : A ∩ B = {0} :=
by
  sorry

end intersection_A_B_l237_237251


namespace equal_share_payments_l237_237417

theorem equal_share_payments (j n : ℝ) 
  (jack_payment : ℝ := 80) 
  (emma_payment : ℝ := 150) 
  (noah_payment : ℝ := 120)
  (liam_payment : ℝ := 200) 
  (total_cost := jack_payment + emma_payment + noah_payment + liam_payment) 
  (individual_share := total_cost / 4) 
  (jack_due := individual_share - jack_payment) 
  (emma_due := emma_payment - individual_share) 
  (noah_due := individual_share - noah_payment) 
  (liam_due := liam_payment - individual_share) 
  (j := jack_due) 
  (n := noah_due) : 
  j - n = 40 := 
by 
  sorry

end equal_share_payments_l237_237417


namespace brother_catch_up_in_3_minutes_l237_237204

variables (v_s v_b : ℝ) (t t_new : ℝ)

-- Conditions
def brother_speed_later_leaves_catch (v_b : ℝ) (v_s : ℝ) : Prop :=
18 * v_s = 12 * v_b

def new_speed_of_brother (v_b v_s : ℝ) : ℝ :=
2 * v_b

def time_to_catch_up (v_s : ℝ) (t_new : ℝ) : Prop :=
6 + t_new = 3 * t_new

-- Goal: prove that t_new = 3
theorem brother_catch_up_in_3_minutes (v_s v_b : ℝ) (t_new : ℝ) :
  (brother_speed_later_leaves_catch v_b v_s) → 
  (new_speed_of_brother v_b v_s) = 3 * v_s → 
  time_to_catch_up v_s t_new → 
  t_new = 3 :=
by sorry

end brother_catch_up_in_3_minutes_l237_237204


namespace apple_tree_total_apples_l237_237805

def firstYear : ℕ := 40
def secondYear : ℕ := 8 + 2 * firstYear
def thirdYear : ℕ := secondYear - (secondYear / 4)

theorem apple_tree_total_apples (FirstYear := firstYear) (SecondYear := secondYear) (ThirdYear := thirdYear) :
  FirstYear + SecondYear + ThirdYear = 194 :=
by 
  sorry

end apple_tree_total_apples_l237_237805


namespace pencils_left_l237_237747

-- Define initial count of pencils
def initial_pencils : ℕ := 20

-- Define pencils misplaced
def misplaced_pencils : ℕ := 7

-- Define pencils broken and thrown away
def broken_pencils : ℕ := 3

-- Define pencils found
def found_pencils : ℕ := 4

-- Define pencils bought
def bought_pencils : ℕ := 2

-- Define the final number of pencils
def final_pencils: ℕ := initial_pencils - misplaced_pencils - broken_pencils + found_pencils + bought_pencils

-- Prove that the final number of pencils is 16
theorem pencils_left : final_pencils = 16 :=
by
  -- The proof steps are omitted here
  sorry

end pencils_left_l237_237747


namespace money_left_over_l237_237138

def initial_amount : ℕ := 120
def sandwich_fraction : ℚ := 1 / 5
def museum_ticket_fraction : ℚ := 1 / 6
def book_fraction : ℚ := 1 / 2

theorem money_left_over :
  let sandwich_cost := initial_amount * sandwich_fraction
  let museum_ticket_cost := initial_amount * museum_ticket_fraction
  let book_cost := initial_amount * book_fraction
  let total_spent := sandwich_cost + museum_ticket_cost + book_cost
  initial_amount - total_spent = 16 :=
by
  sorry

end money_left_over_l237_237138


namespace solution_set_inequality_l237_237989

theorem solution_set_inequality (x : ℝ) :
  ((x^2 - 4) * (x - 6)^2 ≤ 0) ↔ (-2 ≤ x ∧ x ≤ 2 ∨ x = 6) :=
  sorry

end solution_set_inequality_l237_237989


namespace lcm_of_36_and_45_l237_237029

theorem lcm_of_36_and_45 : Nat.lcm 36 45 = 180 := by
  sorry

end lcm_of_36_and_45_l237_237029


namespace integer_solution_exists_l237_237667

theorem integer_solution_exists (a : ℤ) : 
  (∃ k : ℤ, 2 * a^2 = 7 * k + 2) ↔ (a % 7 = 1 ∨ a % 7 = 6) :=
by sorry

end integer_solution_exists_l237_237667


namespace cos_seventh_eq_sum_of_cos_l237_237808

theorem cos_seventh_eq_sum_of_cos:
  ∃ (b₁ b₂ b₃ b₄ b₅ b₆ b₇ : ℝ),
  (∀ θ : ℝ, (Real.cos θ) ^ 7 = b₁ * Real.cos θ + b₂ * Real.cos (2 * θ) + b₃ * Real.cos (3 * θ) + b₄ * Real.cos (4 * θ) + b₅ * Real.cos (5 * θ) + b₆ * Real.cos (6 * θ) + b₇ * Real.cos (7 * θ)) ∧
  (b₁ ^ 2 + b₂ ^ 2 + b₃ ^ 2 + b₄ ^ 2 + b₅ ^ 2 + b₆ ^ 2 + b₇ ^ 2 = 1555 / 4096) :=
sorry

end cos_seventh_eq_sum_of_cos_l237_237808


namespace evaluate_f_at_7_l237_237916

theorem evaluate_f_at_7 :
  (∃ f : ℕ → ℕ, (∀ x, f (2 * x + 1) = x ^ 2 - 2 * x) ∧ f 7 = 3) :=
by 
  sorry

end evaluate_f_at_7_l237_237916


namespace number_of_integer_pairs_l237_237005

theorem number_of_integer_pairs (n : ℕ) : 
  (∀ x y : ℤ, 5 * x^2 - 6 * x * y + y^2 = 6^100) → n = 19594 :=
sorry

end number_of_integer_pairs_l237_237005


namespace solve_for_r_l237_237558

theorem solve_for_r (r : ℤ) : 24 - 5 = 3 * r + 7 → r = 4 :=
by
  intro h
  sorry

end solve_for_r_l237_237558


namespace negative_solution_iff_sum_zero_l237_237440

theorem negative_solution_iff_sum_zero (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔
  a + b + c = 0 :=
by
  sorry

end negative_solution_iff_sum_zero_l237_237440


namespace range_of_a_l237_237509

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) : 
  (∀ x, f x = Real.exp x + a * x) ∧ (∃ x, 0 < x ∧ (DifferentiableAt ℝ f x) ∧ (deriv f x = 0)) → a < -1 :=
by
  sorry

end range_of_a_l237_237509


namespace find_b_l237_237316

def h (x : ℝ) : ℝ := 4 * x - 5

theorem find_b (b : ℝ) (h_b : h b = 1) : b = 3 / 2 :=
by
  sorry

end find_b_l237_237316


namespace carrie_strawberry_harvest_l237_237827

/-- Carrie has a rectangular garden that measures 10 feet by 7 feet.
    She plants the entire garden with strawberry plants. Carrie is able to
    plant 5 strawberry plants per square foot, and she harvests an average of
    12 strawberries per plant. How many strawberries can she expect to harvest?
-/
theorem carrie_strawberry_harvest :
  let width := 10
  let length := 7
  let plants_per_sqft := 5
  let strawberries_per_plant := 12
  let area := width * length
  let total_plants := plants_per_sqft * area
  let total_strawberries := strawberries_per_plant * total_plants
  total_strawberries = 4200 :=
by
  sorry

end carrie_strawberry_harvest_l237_237827


namespace no_square_number_divisible_by_six_in_range_l237_237187

theorem no_square_number_divisible_by_six_in_range :
  ¬ ∃ x : ℕ, (∃ n : ℕ, x = n^2) ∧ (6 ∣ x) ∧ (50 < x) ∧ (x < 120) :=
by
  sorry

end no_square_number_divisible_by_six_in_range_l237_237187


namespace total_songs_bought_l237_237924

def country_albums : ℕ := 2
def pop_albums : ℕ := 8
def songs_per_album : ℕ := 7

theorem total_songs_bought :
  (country_albums + pop_albums) * songs_per_album = 70 := by
  sorry

end total_songs_bought_l237_237924


namespace separation_of_homologous_chromosomes_only_in_meiosis_l237_237406

-- We start by defining the conditions extracted from the problem.
def chromosome_replication (phase: String) : Prop :=  
  phase = "S phase"

def separation_of_homologous_chromosomes (process: String) : Prop := 
  process = "meiosis I"

def separation_of_chromatids (process: String) : Prop := 
  process = "mitosis anaphase" ∨ process = "meiosis II anaphase II"

def cytokinesis (end_phase: String) : Prop := 
  end_phase = "end mitosis" ∨ end_phase = "end meiosis"

-- Now, we state that the separation of homologous chromosomes does not occur during mitosis.
theorem separation_of_homologous_chromosomes_only_in_meiosis :
  ∀ (process: String), ¬ separation_of_homologous_chromosomes "mitosis" := 
sorry

end separation_of_homologous_chromosomes_only_in_meiosis_l237_237406


namespace inequality_proof_l237_237598

theorem inequality_proof (a b c d : ℝ) : (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := 
by
  sorry

end inequality_proof_l237_237598


namespace cos_value_l237_237302

theorem cos_value (α : ℝ) (h : Real.sin (π / 5 - α) = 1 / 3) : 
  Real.cos (2 * α + 3 * π / 5) = -7 / 9 := by
  sorry

end cos_value_l237_237302


namespace positive_integer_solution_l237_237292

theorem positive_integer_solution (x y z t : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (ht : 0 < t) :
  (1 / (x * x : ℝ) + 1 / (y * y : ℝ) + 1 / (z * z : ℝ) + 1 / (t * t : ℝ) = 1) ↔ (x = 2 ∧ y = 2 ∧ z = 2 ∧ t = 2) :=
by
  sorry

end positive_integer_solution_l237_237292


namespace exists_C_a_n1_minus_a_n_l237_237579

noncomputable def a : ℕ → ℚ
| 0 => 0
| 1 => 1
| 2 => 8
| (n+1) => a (n - 1) + (4 / n) * a n

theorem exists_C (C : ℕ) (hC : C = 2) : ∃ C > 0, ∀ n > 0, a n ≤ C * n^2 := by
  use 2
  sorry

theorem a_n1_minus_a_n (n : ℕ) (h : n > 0) : a (n + 1) - a n ≤ 4 * n + 3 := by
  sorry

end exists_C_a_n1_minus_a_n_l237_237579


namespace prime_factor_of_difference_l237_237510

theorem prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A) (hA9 : A ≤ 9) (hC : 1 ≤ C) (hC9 : C ≤ 9) (hA_ne_C : A ≠ C) :
  ∃ p : ℕ, Prime p ∧ p = 3 ∧ p ∣ 3 * (100 * A + 10 * B + C - (100 * C + 10 * B + A)) := by
  sorry

end prime_factor_of_difference_l237_237510


namespace calculate_expression_l237_237861

theorem calculate_expression : 61 + 5 * 12 / (180 / 3) = 62 := by
  sorry

end calculate_expression_l237_237861


namespace limit_sum_perimeters_areas_of_isosceles_triangles_l237_237315

theorem limit_sum_perimeters_areas_of_isosceles_triangles (b s h : ℝ) : 
  ∃ P A : ℝ, 
    (P = 2*(b + 2*s)) ∧ 
    (A = (2/3)*b*h) :=
  sorry

end limit_sum_perimeters_areas_of_isosceles_triangles_l237_237315


namespace fraction_habitable_earth_l237_237446

theorem fraction_habitable_earth (one_fifth_land: ℝ) (one_third_inhabitable: ℝ)
  (h_land_fraction : one_fifth_land = 1 / 5)
  (h_inhabitable_fraction : one_third_inhabitable = 1 / 3) :
  (one_fifth_land * one_third_inhabitable) = 1 / 15 :=
by
  sorry

end fraction_habitable_earth_l237_237446


namespace compute_fraction_pow_mul_l237_237592

theorem compute_fraction_pow_mul :
  8 * (2 / 3)^4 = 128 / 81 :=
by 
  sorry

end compute_fraction_pow_mul_l237_237592


namespace circle_x_intercept_l237_237936

theorem circle_x_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 3) (k1 : y1 = 2) (h2 : x2 = 11) (k2 : y2 = 8) :
  ∃ x : ℝ, (x ≠ 3) ∧ ((x - 7) ^ 2 + (0 - 5) ^ 2 = 25) ∧ (x = 7) :=
by
  sorry

end circle_x_intercept_l237_237936


namespace minimum_value_inequality_l237_237601

theorem minimum_value_inequality
  (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 2 * x + y - 3 = 0) :
  ∃ t : ℝ, (∀ (x y : ℝ), (2 * x + y = 3) → (0 < x) → (0 < y) → (t = (4 * y - x + 6) / (x * y)) → 9 ≤ t) ∧
          (∃ (x_ y_: ℝ), 2 * x_ + y_ = 3 ∧ 0 < x_ ∧ 0 < y_ ∧ (4 * y_ - x_ + 6) / (x_ * y_) = 9) :=
sorry

end minimum_value_inequality_l237_237601


namespace hash_of_hash_of_hash_of_70_l237_237664

def hash (N : ℝ) : ℝ := 0.4 * N + 2

theorem hash_of_hash_of_hash_of_70 : hash (hash (hash 70)) = 8 := by
  sorry

end hash_of_hash_of_hash_of_70_l237_237664


namespace investor_difference_l237_237819

/-
Scheme A yields 30% of the capital within a year.
Scheme B yields 50% of the capital within a year.
Investor invested $300 in scheme A.
Investor invested $200 in scheme B.
We need to prove that the difference in total money between scheme A and scheme B after a year is $90.
-/

def schemeA_yield_rate : ℝ := 0.30
def schemeB_yield_rate : ℝ := 0.50
def schemeA_investment : ℝ := 300
def schemeB_investment : ℝ := 200

def total_after_year (investment : ℝ) (yield_rate : ℝ) : ℝ :=
  investment * (1 + yield_rate)

theorem investor_difference :
  total_after_year schemeA_investment schemeA_yield_rate - total_after_year schemeB_investment schemeB_yield_rate = 90 := by
  sorry

end investor_difference_l237_237819


namespace factorize_xy_l237_237986

theorem factorize_xy (x y : ℕ): xy - x + y - 1 = (x + 1) * (y - 1) :=
by
  sorry

end factorize_xy_l237_237986


namespace scientific_notation_flu_virus_diameter_l237_237170

theorem scientific_notation_flu_virus_diameter :
  0.000000823 = 8.23 * 10^(-7) :=
sorry

end scientific_notation_flu_virus_diameter_l237_237170


namespace find_abc_l237_237616

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) : 
  abc_value a b c = 762 :=
sorry

end find_abc_l237_237616


namespace prove_correct_operation_l237_237844

def correct_operation (a b : ℕ) : Prop :=
  (a^3 * a^2 ≠ a^6) ∧
  ((a * b^2)^2 = a^2 * b^4) ∧
  (a^10 / a^5 ≠ a^2) ∧
  (a^2 + a ≠ a^3)

theorem prove_correct_operation (a b : ℕ) : correct_operation a b :=
by {
  sorry
}

end prove_correct_operation_l237_237844


namespace inequality_solution_l237_237932

theorem inequality_solution (y : ℝ) : 
  (3 ≤ |y - 4| ∧ |y - 4| ≤ 7) ↔ (7 ≤ y ∧ y ≤ 11 ∨ -3 ≤ y ∧ y ≤ 1) :=
by
  sorry

end inequality_solution_l237_237932


namespace benny_leftover_money_l237_237327

-- Define the conditions
def initial_money : ℕ := 67
def spent_money : ℕ := 34

-- Define the leftover money calculation
def leftover_money : ℕ := initial_money - spent_money

-- Prove that Benny had 33 dollars left over
theorem benny_leftover_money : leftover_money = 33 :=
by 
  -- Proof
  sorry

end benny_leftover_money_l237_237327


namespace sqrt_of_4_eq_2_l237_237208

theorem sqrt_of_4_eq_2 : Real.sqrt 4 = 2 := by
  sorry

end sqrt_of_4_eq_2_l237_237208


namespace expected_value_of_coins_is_95_5_l237_237384

-- Define the individual coin values in cents
def penny_value : ℕ := 1
def nickel_value : ℕ := 5
def dime_value : ℕ := 10
def quarter_value : ℕ := 25
def fifty_cent_value : ℕ := 50
def dollar_value : ℕ := 100

-- Expected value function with 1/2 probability 
def expected_value (coin_value : ℕ) : ℚ := (coin_value : ℚ) / 2

-- Calculate the total expected value of all coins flipped
noncomputable def total_expected_value : ℚ :=
  expected_value penny_value +
  expected_value nickel_value +
  expected_value dime_value +
  expected_value quarter_value +
  expected_value fifty_cent_value +
  expected_value dollar_value

-- Prove that the expected total value is 95.5
theorem expected_value_of_coins_is_95_5 :
  total_expected_value = 95.5 := by
  sorry

end expected_value_of_coins_is_95_5_l237_237384


namespace driver_days_off_l237_237597

theorem driver_days_off 
  (drivers : ℕ) 
  (cars : ℕ) 
  (maintenance_rate : ℚ) 
  (days_in_month : ℕ)
  (needed_driver_days : ℕ)
  (x : ℚ) :
  drivers = 54 →
  cars = 60 →
  maintenance_rate = 0.25 →
  days_in_month = 30 →
  needed_driver_days = 45 * days_in_month →
  54 * (30 - x) = needed_driver_days →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end driver_days_off_l237_237597


namespace crafts_sold_l237_237573

theorem crafts_sold (x : ℕ) 
  (h1 : ∃ (n : ℕ), 12 * n = x * 12)
  (h2 : x * 12 + 7 - 18 = 25):
  x = 3 :=
by
  sorry

end crafts_sold_l237_237573


namespace eunice_pots_l237_237825

theorem eunice_pots (total_seeds pots_with_3_seeds last_pot_seeds : ℕ)
  (h1 : total_seeds = 10)
  (h2 : pots_with_3_seeds * 3 + last_pot_seeds = total_seeds)
  (h3 : last_pot_seeds = 1) : pots_with_3_seeds + 1 = 4 :=
by
  -- Proof omitted
  sorry

end eunice_pots_l237_237825


namespace sufficient_but_not_necessary_l237_237665

theorem sufficient_but_not_necessary (x : ℝ) : (x > 0 → x * (x + 1) > 0) ∧ ¬ (x * (x + 1) > 0 → x > 0) := 
by 
sorry

end sufficient_but_not_necessary_l237_237665


namespace determine_xyz_l237_237220

variables {x y z : ℝ}

theorem determine_xyz (h : (x - y - 3)^2 + (y - z)^2 + (x - z)^2 = 3) : 
  x = z + 1 ∧ y = z - 1 := 
sorry

end determine_xyz_l237_237220


namespace max_value_of_expressions_l237_237385

theorem max_value_of_expressions (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : a + b = 1) :
  b > 1/2 ∧ b > 2 * a * b ∧ b > a^2 + b^2 :=
by
  sorry

end max_value_of_expressions_l237_237385


namespace find_interest_rate_l237_237872

theorem find_interest_rate
  (P : ℝ) (A : ℝ) (n t : ℕ) (hP : P = 3000) (hA : A = 3307.5) (hn : n = 2) (ht : t = 1) :
  ∃ r : ℝ, r = 10 :=
by
  sorry

end find_interest_rate_l237_237872


namespace find_x2_plus_y2_l237_237821

theorem find_x2_plus_y2 
  (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x * y + x + y = 83)
  (h2 : x^2 * y + x * y^2 = 1056) :
  x^2 + y^2 = 458 :=
by
  sorry

end find_x2_plus_y2_l237_237821


namespace not_inequality_l237_237896

theorem not_inequality (x : ℝ) : ¬ (x^2 + 2*x - 3 < 0) :=
sorry

end not_inequality_l237_237896


namespace angle_line_plane_l237_237498

theorem angle_line_plane {l α : Type} (θ : ℝ) (h : θ = 150) : 
  ∃ φ : ℝ, φ = 60 := 
by
  -- This part would require the actual proof.
  sorry

end angle_line_plane_l237_237498


namespace percent_owning_only_cats_l237_237144

theorem percent_owning_only_cats (total_students dogs cats both : ℕ) (h1 : total_students = 500)
  (h2 : dogs = 150) (h3 : cats = 80) (h4 : both = 25) : (cats - both) / total_students * 100 = 11 :=
by
  sorry

end percent_owning_only_cats_l237_237144


namespace problem1_problem2_l237_237396

-- Problem 1:
theorem problem1 (α : ℝ) (h : Real.tan α = 2) : 
  (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 3 :=
sorry

-- Problem 2:
theorem problem2 (α : ℝ) : 
  (Real.tan (2 * Real.pi - α) * Real.cos (2 * Real.pi - α) * Real.sin (-α + 3 * Real.pi / 2)) /
  (Real.cos (-α + Real.pi) * Real.sin (-Real.pi + α)) = 1 :=
sorry

end problem1_problem2_l237_237396


namespace trip_time_difference_l237_237762

-- Definitions of the given conditions
def speed_AB := 160 -- speed from A to B in km/h
def speed_BA := 120 -- speed from B to A in km/h
def distance_AB := 480 -- distance between A and B in km

-- Calculation of the time for each trip
def time_AB := distance_AB / speed_AB
def time_BA := distance_AB / speed_BA

-- The statement we need to prove
theorem trip_time_difference :
  (time_BA - time_AB) = 1 :=
by
  sorry

end trip_time_difference_l237_237762


namespace watch_cost_price_l237_237344

theorem watch_cost_price 
  (C : ℝ)
  (h1 : 0.9 * C + 180 = 1.05 * C) :
  C = 1200 :=
sorry

end watch_cost_price_l237_237344


namespace polygon_interior_angles_540_implies_5_sides_l237_237423

theorem polygon_interior_angles_540_implies_5_sides (n : ℕ) :
  (n - 2) * 180 = 540 → n = 5 :=
by
  sorry

end polygon_interior_angles_540_implies_5_sides_l237_237423


namespace part2_l237_237774

noncomputable def f (x : ℝ) : ℝ := |2 * x - 1|

theorem part2 (x y : ℝ) (h₁ : |x - y - 1| ≤ 1 / 3) (h₂ : |2 * y + 1| ≤ 1 / 6) :
  f x < 1 := 
by
  sorry

end part2_l237_237774


namespace solve_system_of_equations_l237_237790

theorem solve_system_of_equations :
    ∃ x y : ℚ, 4 * x - 3 * y = 2 ∧ 6 * x + 5 * y = 1 ∧ x = 13 / 38 ∧ y = -4 / 19 :=
by
  sorry

end solve_system_of_equations_l237_237790


namespace senior_year_allowance_more_than_twice_l237_237515

noncomputable def middle_school_allowance : ℝ :=
  8 + 2

noncomputable def twice_middle_school_allowance : ℝ :=
  2 * middle_school_allowance

noncomputable def senior_year_increase : ℝ :=
  1.5 * middle_school_allowance

noncomputable def senior_year_allowance : ℝ :=
  middle_school_allowance + senior_year_increase

theorem senior_year_allowance_more_than_twice : 
  senior_year_allowance = twice_middle_school_allowance + 5 :=
by
  sorry

end senior_year_allowance_more_than_twice_l237_237515


namespace price_per_glass_on_second_day_l237_237700

 -- Definitions based on the conditions
def orangeade_first_day (O: ℝ) : ℝ := 2 * O -- Total volume on first day, O + O
def orangeade_second_day (O: ℝ) : ℝ := 3 * O -- Total volume on second day, O + 2O
def revenue_first_day (O: ℝ) (price_first_day: ℝ) : ℝ := 2 * O * price_first_day -- Revenue on first day
def revenue_second_day (O: ℝ) (P: ℝ) : ℝ := 3 * O * P -- Revenue on second day
def price_first_day: ℝ := 0.90 -- Given price per glass on the first day

 -- Statement to be proved
theorem price_per_glass_on_second_day (O: ℝ) (P: ℝ) (h: revenue_first_day O price_first_day = revenue_second_day O P) :
  P = 0.60 :=
by
  sorry

end price_per_glass_on_second_day_l237_237700


namespace camel_steps_divisibility_l237_237703

variables (A B : Type) (p q : ℕ)

-- Description of the conditions
-- let A, B be vertices
-- p and q be the steps to travel from A to B in different paths

theorem camel_steps_divisibility (h1: ∃ r : ℕ, p + r ≡ 0 [MOD 3])
                                  (h2: ∃ r : ℕ, q + r ≡ 0 [MOD 3]) : (p - q) % 3 = 0 := by
  sorry

end camel_steps_divisibility_l237_237703


namespace not_partitionable_1_to_15_l237_237441

theorem not_partitionable_1_to_15 :
  ∀ (A B : Finset ℕ), (∀ x ∈ A, x ∈ Finset.range 16) →
    (∀ x ∈ B, x ∈ Finset.range 16) →
    A.card = 2 → B.card = 13 →
    A ∪ B = Finset.range 16 →
    ¬(A.sum id = B.prod id) :=
by
  -- To be proved
  sorry

end not_partitionable_1_to_15_l237_237441


namespace tyler_age_l237_237200

theorem tyler_age (T B : ℕ) (h1 : T = B - 3) (h2 : T + B = 11) : T = 4 :=
  sorry

end tyler_age_l237_237200


namespace pieces_after_cuts_l237_237867

theorem pieces_after_cuts (n : ℕ) (h : n = 10) : (n + 1) = 11 := by
  sorry

end pieces_after_cuts_l237_237867


namespace tan_add_pi_over_3_l237_237648

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = 3) : 
  Real.tan (x + Real.pi / 3) = (3 + Real.sqrt 3) / (1 - 3 * Real.sqrt 3) :=
by
  sorry

end tan_add_pi_over_3_l237_237648


namespace sufficient_not_necessary_l237_237325

theorem sufficient_not_necessary (x y : ℝ) :
  (x ≥ 2 ∧ y ≥ 2) ↔ (x + y ≥ 4) :=
by sorry

end sufficient_not_necessary_l237_237325


namespace factorization_a_minus_b_l237_237719

-- Define the problem in Lean 4
theorem factorization_a_minus_b (a b : ℤ) : 
  (∀ y : ℝ, 2 * y^2 - 5 * y - 12 = (2 * y + a) * (y + b))
  → a - b = 7 := 
by 
  sorry

end factorization_a_minus_b_l237_237719


namespace ratio_mark_to_jenna_l237_237352

-- Definitions based on the given conditions
def total_problems : ℕ := 20

def problems_angela : ℕ := 9
def problems_martha : ℕ := 2
def problems_jenna : ℕ := 4 * problems_martha - 2

def problems_completed : ℕ := problems_angela + problems_martha + problems_jenna
def problems_mark : ℕ := total_problems - problems_completed

-- The proof statement based on the question and conditions
theorem ratio_mark_to_jenna :
  (problems_mark : ℚ) / problems_jenna = 1 / 2 :=
by
  sorry

end ratio_mark_to_jenna_l237_237352


namespace down_payment_calculation_l237_237210

theorem down_payment_calculation 
  (purchase_price : ℝ)
  (monthly_payment : ℝ)
  (n : ℕ)
  (interest_rate : ℝ)
  (down_payment : ℝ) :
  purchase_price = 127 ∧ 
  monthly_payment = 10 ∧ 
  n = 12 ∧ 
  interest_rate = 0.2126 ∧
  down_payment + (n * monthly_payment) = purchase_price * (1 + interest_rate) 
  → down_payment = 34 := 
sorry

end down_payment_calculation_l237_237210


namespace arithmetic_sequence_sum_l237_237135

theorem arithmetic_sequence_sum (b : ℕ → ℝ) (h_arith : ∀ n, b (n+1) - b n = b 2 - b 1) (h_b5 : b 5 = 2) :
  b 1 + b 2 + b 3 + b 4 + b 5 + b 6 + b 7 + b 8 + b 9 = 18 := 
sorry

end arithmetic_sequence_sum_l237_237135


namespace fraction_to_decimal_l237_237214

theorem fraction_to_decimal : (7 : ℚ) / 16 = 0.4375 := by
  sorry

end fraction_to_decimal_l237_237214


namespace arithmetic_sequence_problem_l237_237089

variable (n : ℕ) (a S : ℕ → ℕ)

theorem arithmetic_sequence_problem
  (h1 : a 2 + a 8 = 82)
  (h2 : S 41 = S 9)
  (hSn : ∀ n, S n = n * (a 1 + a n) / 2) :
  (∀ n, a n = 51 - 2 * n) ∧ (∀ n, S n ≤ 625) := sorry

end arithmetic_sequence_problem_l237_237089


namespace find_expression_value_l237_237424

theorem find_expression_value 
  (x y : ℝ) 
  (h1 : 4 * x + y = 10) 
  (h2 : x + 4 * y = 18) : 
  16 * x^2 + 24 * x * y + 16 * y^2 = 424 := 
by 
  sorry

end find_expression_value_l237_237424


namespace part1_part2_l237_237927

-- Part (1)
theorem part1 (x : ℝ) (m : ℝ) (h : x = 2) : 
  (x / (x - 3) + m / (3 - x) = 3) → m = 5 :=
sorry

-- Part (2)
theorem part2 (x : ℝ) (m : ℝ) :
  (x / (x - 3) + m / (3 - x) = 3) → (x > 0) → (m < 9) ∧ (m ≠ 3) :=
sorry

end part1_part2_l237_237927


namespace intersecting_circle_radius_l237_237650

-- Definitions representing the conditions
def non_intersecting_circles (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) : Prop :=
  ∀ i j, i ≠ j → dist (O_i i) (O_i j) ≥ r_i i + r_i j

def min_radius_one (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) := 
  ∀ i, r_i i ≥ 1

-- The main theorem stating the proof goal
theorem intersecting_circle_radius 
  (O_i : Fin 6 → ℕ) (r_i : Fin 6 → ℝ) (O : ℕ) (r : ℝ)
  (h_non_intersecting : non_intersecting_circles O_i r_i)
  (h_min_radius : min_radius_one O_i r_i)
  (h_intersecting : ∀ i, dist O (O_i i) ≤ r + r_i i) :
  r ≥ 1 := 
sorry

end intersecting_circle_radius_l237_237650


namespace arun_age_proof_l237_237379

theorem arun_age_proof {A G M : ℕ} 
  (h1 : (A - 6) / 18 = G)
  (h2 : G = M - 2)
  (h3 : M = 5) :
  A = 60 :=
by
  sorry

end arun_age_proof_l237_237379


namespace solve_equation_l237_237203

theorem solve_equation (x : ℝ) (hx : (x + 1) ≠ 0) :
  (x = -3 / 4) ∨ (x = -1) ↔ (x^3 + x^2 + x + 1) / (x + 1) = x^2 + 4 * x + 4 :=
by
  sorry

end solve_equation_l237_237203


namespace freda_flag_dimensions_l237_237952

/--  
Given the area of the dove is 192 cm², and the perimeter of the dove consists of quarter-circles or straight lines,
prove that the dimensions of the flag are 24 cm by 16 cm.
-/
theorem freda_flag_dimensions (area_dove : ℝ) (h1 : area_dove = 192) : 
∃ (length width : ℝ), length = 24 ∧ width = 16 := 
sorry

end freda_flag_dimensions_l237_237952


namespace octagon_reflected_arcs_area_l237_237742

theorem octagon_reflected_arcs_area :
  let s := 2
  let θ := 45
  let r := 2 / Real.sqrt (2 - Real.sqrt (2))
  let sector_area := θ / 360 * Real.pi * r^2
  let total_arc_area := 8 * sector_area
  let circle_area := Real.pi * r^2
  let bounded_region_area := 8 * (circle_area - 2 * Real.sqrt (2) * 1 / 2)
  bounded_region_area = (16 * Real.sqrt 2 / 3 - Real.pi)
:= sorry

end octagon_reflected_arcs_area_l237_237742


namespace Connie_total_markers_l237_237096

/--
Connie has 41 red markers and 64 blue markers. 
We want to prove that the total number of markers Connie has is 105.
-/
theorem Connie_total_markers : 
  let red_markers := 41
  let blue_markers := 64
  let total_markers := red_markers + blue_markers
  total_markers = 105 :=
by
  sorry

end Connie_total_markers_l237_237096


namespace ellipse_major_minor_axis_condition_l237_237467

theorem ellipse_major_minor_axis_condition (h1 : ∀ x y : ℝ, x^2 + m * y^2 = 1) 
                                          (h2 : ∀ a b : ℝ, a = 2 * b) :
  m = 1 / 4 :=
sorry

end ellipse_major_minor_axis_condition_l237_237467


namespace Crimson_Valley_skirts_l237_237413

theorem Crimson_Valley_skirts
  (Azure_Valley_skirts : ℕ)
  (Seafoam_Valley_skirts : ℕ)
  (Purple_Valley_skirts : ℕ)
  (Crimson_Valley_skirts : ℕ)
  (h1 : Azure_Valley_skirts = 90)
  (h2 : Seafoam_Valley_skirts = (2/3 : ℚ) * Azure_Valley_skirts)
  (h3 : Purple_Valley_skirts = (1/4 : ℚ) * Seafoam_Valley_skirts)
  (h4 : Crimson_Valley_skirts = (1/3 : ℚ) * Purple_Valley_skirts)
  : Crimson_Valley_skirts = 5 := 
sorry

end Crimson_Valley_skirts_l237_237413


namespace no_integer_roots_l237_237685

theorem no_integer_roots (a b c : ℤ) (h1 : a ≠ 0) (h2 : a % 2 = 1) (h3 : b % 2 = 1) (h4 : c % 2 = 1) :
  ∀ x : ℤ, a * x^2 + b * x + c ≠ 0 :=
by
  sorry

end no_integer_roots_l237_237685


namespace final_apples_count_l237_237617

-- Definitions from the problem conditions
def initialApples : ℕ := 150
def soldToJill (initial : ℕ) : ℕ := initial * 30 / 100
def remainingAfterJill (initial : ℕ) := initial - soldToJill initial
def soldToJune (remaining : ℕ) : ℕ := remaining * 20 / 100
def remainingAfterJune (remaining : ℕ) := remaining - soldToJune remaining
def givenToFriend (current : ℕ) : ℕ := current - 2
def soldAfterFriend (current : ℕ) : ℕ := current * 10 / 100
def remainingAfterAll (current : ℕ) := current - soldAfterFriend current

theorem final_apples_count : remainingAfterAll (givenToFriend (remainingAfterJune (remainingAfterJill initialApples))) = 74 :=
by
  sorry

end final_apples_count_l237_237617


namespace susan_change_sum_susan_possible_sums_l237_237501

theorem susan_change_sum
  (change : ℕ)
  (h_lt_100 : change < 100)
  (h_nickels : ∃ k : ℕ, change = 5 * k + 2)
  (h_quarters : ∃ m : ℕ, change = 25 * m + 5) :
  change = 30 ∨ change = 55 ∨ change = 80 :=
sorry

theorem susan_possible_sums :
  30 + 55 + 80 = 165 :=
by norm_num

end susan_change_sum_susan_possible_sums_l237_237501


namespace farm_distance_l237_237999

theorem farm_distance (a x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (triangle_ineq1 : x + z = 85)
  (triangle_ineq2 : x + y = 4 * z)
  (triangle_ineq3 : z + y = x + a) :
  0 < a ∧ a < 85 ∧
  x = (340 - a) / 6 ∧
  y = (2 * a + 85) / 3 ∧
  z = (170 + a) / 6 :=
sorry

end farm_distance_l237_237999


namespace fred_seashells_l237_237288

-- Definitions based on conditions
def tom_seashells : Nat := 15
def total_seashells : Nat := 58

-- The theorem we want to prove
theorem fred_seashells : (15 + F = 58) → F = 43 := 
by
  intro h
  have h1 : F = 58 - 15 := by linarith
  exact h1

end fred_seashells_l237_237288


namespace parabola_standard_eq_l237_237279

theorem parabola_standard_eq (h : ∃ (x y : ℝ), x - 2 * y - 4 = 0 ∧ (
                         (y = 0 ∧ x = 4 ∧ y^2 = 16 * x) ∨ 
                         (x = 0 ∧ y = -2 ∧ x^2 = -8 * y))
                         ) :
                         (y^2 = 16 * x) ∨ (x^2 = -8 * y) :=
by 
  sorry

end parabola_standard_eq_l237_237279


namespace number_of_zeros_l237_237967

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
noncomputable def f'' : ℝ → ℝ := sorry

def odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

def conditions (f : ℝ → ℝ) (f'' : ℝ → ℝ) :=
  odd_function f ∧ ∀ x : ℝ, x < 0 → (2 * f x + x * f'' x < x * f x)

theorem number_of_zeros (f : ℝ → ℝ) (f'' : ℝ → ℝ) (h : conditions f f'') :
  ∃! x : ℝ, f x = 0 :=
sorry

end number_of_zeros_l237_237967


namespace second_piece_cost_l237_237864

theorem second_piece_cost
  (total_spent : ℕ)
  (num_pieces : ℕ)
  (single_piece1 : ℕ)
  (single_piece2 : ℕ)
  (remaining_piece_count : ℕ)
  (remaining_piece_cost : ℕ)
  (total_cost : total_spent = 610)
  (number_of_items : num_pieces = 7)
  (first_item_cost : single_piece1 = 49)
  (remaining_piece_item_cost : remaining_piece_cost = 96)
  (first_item_total_cost : remaining_piece_count = 5)
  (sum_equation : single_piece1 + single_piece2 + (remaining_piece_count * remaining_piece_cost) = total_spent) :
  single_piece2 = 81 := 
  sorry

end second_piece_cost_l237_237864


namespace determine_constants_and_sum_l237_237173

theorem determine_constants_and_sum (A B C x : ℝ) (h₁ : A = 3) (h₂ : B = 5) (h₃ : C = 40 / 3)
  (h₄ : (x + B) * (A * x + 40) / ((x + C) * (x + 5)) = 3) :
  ∀ x : ℝ, x ≠ -5 → x ≠ -40 / 3 → (-(5 : ℝ) + -40 / 3 = -55 / 3) :=
sorry

end determine_constants_and_sum_l237_237173


namespace sub_seq_arithmetic_l237_237629

variable (a : ℕ → ℝ) (d : ℝ)

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def sub_seq (a : ℕ → ℝ) (k : ℕ) : ℝ :=
  a (3 * k - 1)

theorem sub_seq_arithmetic (h : is_arithmetic_sequence a d) : is_arithmetic_sequence (sub_seq a) (3 * d) := 
sorry


end sub_seq_arithmetic_l237_237629


namespace perfect_square_factors_count_450_l237_237165

theorem perfect_square_factors_count_450 : 
  (∃ n : ℕ, n = 4 ∧ (∀ d : ℕ, d ∣ 450 → ∃ k : ℕ, d = k * k)) := sorry

end perfect_square_factors_count_450_l237_237165


namespace jason_initial_speed_correct_l237_237545

noncomputable def jason_initial_speed (d : ℝ) (t1 t2 : ℝ) (v2 : ℝ) : ℝ :=
  let t_total := t1 + t2
  let d2 := v2 * t2
  let d1 := d - d2
  let v1 := d1 / t1
  v1

theorem jason_initial_speed_correct :
  jason_initial_speed 120 0.5 1 90 = 60 := 
by 
  sorry

end jason_initial_speed_correct_l237_237545


namespace find_positive_integral_solution_l237_237403

theorem find_positive_integral_solution :
  ∃ n : ℕ, n > 0 ∧ (n - 1) * 101 = (n + 1) * 100 := by
sorry

end find_positive_integral_solution_l237_237403


namespace total_stuffed_animals_l237_237613

theorem total_stuffed_animals (M K T : ℕ) 
  (hM : M = 34) 
  (hK : K = 2 * M) 
  (hT : T = K + 5) : 
  M + K + T = 175 :=
by
  -- Adding sorry to complete the placeholder
  sorry

end total_stuffed_animals_l237_237613


namespace more_bags_found_l237_237785

def bags_Monday : ℕ := 7
def bags_nextDay : ℕ := 12

theorem more_bags_found : bags_nextDay - bags_Monday = 5 := by
  -- Proof Skipped
  sorry

end more_bags_found_l237_237785


namespace larger_of_two_numbers_l237_237807

theorem larger_of_two_numbers (hcf : ℕ) (f1 : ℕ) (f2 : ℕ) 
(h_hcf : hcf = 10) 
(h_f1 : f1 = 11) 
(h_f2 : f2 = 15) 
: max (hcf * f1) (hcf * f2) = 150 :=
by
  have lcm := hcf * f1 * f2
  have num1 := hcf * f1
  have num2 := hcf * f2
  sorry

end larger_of_two_numbers_l237_237807


namespace option_B_proof_option_C_proof_l237_237494

-- Definitions and sequences
variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}

-- Statement of the problem

theorem option_B_proof (A B : ℝ) :
  (∀ n : ℕ, S n = A * (n : ℝ)^2 + B * n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d := 
sorry

theorem option_C_proof :
  (∀ n : ℕ, S n = 1 - (-1)^n) →
  (∀ n : ℕ, a n = S n - S (n - 1)) →
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end option_B_proof_option_C_proof_l237_237494


namespace smallest_prime_factor_of_difference_l237_237038

theorem smallest_prime_factor_of_difference (A B C : ℕ) (hA : 1 ≤ A ∧ A ≤ 9) (hB : 0 ≤ B ∧ B ≤ 9) (hC : 1 ≤ C ∧ C ≤ 9) (h_diff : A ≠ C) : 
  ∃ p : ℕ, Prime p ∧ p ∣ (100 * A + 10 * B + C - (100 * C + 10 * B + A)) ∧ p = 3 :=
by
  sorry

end smallest_prime_factor_of_difference_l237_237038


namespace wooden_parallelepiped_length_l237_237095

theorem wooden_parallelepiped_length (n : ℕ) (h1 : n ≥ 7)
    (h2 : ∀ total_cubes unpainted_cubes : ℕ,
      total_cubes = n * (n - 2) * (n - 4) ∧
      unpainted_cubes = (n - 2) * (n - 4) * (n - 6) ∧
      unpainted_cubes = 2 / 3 * total_cubes) :
  n = 18 := 
sorry

end wooden_parallelepiped_length_l237_237095


namespace find_shirt_numbers_calculate_profit_l237_237110

def total_shirts_condition (x y : ℕ) : Prop := x + y = 200
def total_cost_condition (x y : ℕ) : Prop := 25 * x + 15 * y = 3500
def profit_calculation (x y : ℕ) : ℕ := (50 - 25) * x + (35 - 15) * y

theorem find_shirt_numbers (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  x = 50 ∧ y = 150 :=
sorry

theorem calculate_profit (x y : ℕ) (h1 : total_shirts_condition x y) (h2 : total_cost_condition x y) :
  profit_calculation x y = 4250 :=
sorry

end find_shirt_numbers_calculate_profit_l237_237110


namespace students_participated_in_both_l237_237756

theorem students_participated_in_both (total_students volleyball track field no_participation both: ℕ) 
  (h1 : total_students = 45) 
  (h2 : volleyball = 12) 
  (h3 : track = 20) 
  (h4 : no_participation = 19) 
  (h5 : both = volleyball + track - (total_students - no_participation)) 
  : both = 6 :=
by
  sorry

end students_participated_in_both_l237_237756


namespace Benny_and_Tim_have_47_books_together_l237_237043

/-
  Definitions and conditions:
  1. Benny_has_24_books : Benny has 24 books.
  2. Benny_gave_10_books_to_Sandy : Benny gave Sandy 10 books.
  3. Tim_has_33_books : Tim has 33 books.
  
  Goal:
  Prove that together Benny and Tim have 47 books.
-/

def Benny_has_24_books : ℕ := 24
def Benny_gave_10_books_to_Sandy : ℕ := 10
def Tim_has_33_books : ℕ := 33

def Benny_remaining_books : ℕ := Benny_has_24_books - Benny_gave_10_books_to_Sandy

def Benny_and_Tim_together : ℕ := Benny_remaining_books + Tim_has_33_books

theorem Benny_and_Tim_have_47_books_together :
  Benny_and_Tim_together = 47 := by
  sorry

end Benny_and_Tim_have_47_books_together_l237_237043


namespace cos_double_alpha_proof_l237_237559

theorem cos_double_alpha_proof (α : ℝ) (h1 : Real.sin (π / 3 - α) = 1 / 3) : 
  Real.cos (π / 3 + 2 * α) = - 7 / 9 :=
by
  sorry

end cos_double_alpha_proof_l237_237559


namespace sum_of_abc_l237_237276

theorem sum_of_abc (a b c : ℕ) (h : a + b + c = 12) 
  (area_ratio : ℝ) (side_length_ratio : ℝ) 
  (ha : area_ratio = 50 / 98) 
  (hb : side_length_ratio = (Real.sqrt 50) / (Real.sqrt 98))
  (hc : side_length_ratio = (a * (Real.sqrt b)) / c) :
  a + b + c = 12 :=
by
  sorry

end sum_of_abc_l237_237276


namespace gcd_35_x_eq_7_in_range_80_90_l237_237052

theorem gcd_35_x_eq_7_in_range_80_90 {n : ℕ} (h₁ : Nat.gcd 35 n = 7) (h₂ : 80 < n) (h₃ : n < 90) : n = 84 :=
by
  sorry

end gcd_35_x_eq_7_in_range_80_90_l237_237052


namespace parallel_lines_slope_l237_237006

theorem parallel_lines_slope (k : ℝ) :
  (∀ x : ℝ, 5 * x - 3 = (3 * k) * x + 7 -> ((3 * k) = 5)) -> (k = 5 / 3) :=
by
  -- Posing the conditions on parallel lines
  intro h_eq_slopes
  -- We know 3k = 5, hence k = 5 / 3
  have slope_eq : 3 * k = 5 := by sorry
  -- Therefore k = 5 / 3 follows from the fact 3k = 5
  have k_val : k = 5 / 3 := by sorry
  exact k_val

end parallel_lines_slope_l237_237006


namespace P_plus_Q_l237_237351

theorem P_plus_Q (P Q : ℝ) (h : (P / (x - 3) + Q * (x - 2)) = (-5 * x^2 + 18 * x + 27) / (x - 3)) : P + Q = 31 := 
by {
  sorry
}

end P_plus_Q_l237_237351


namespace cheryl_needed_first_material_l237_237018

noncomputable def cheryl_material (x : ℚ) : ℚ :=
  x + 1 / 3 - 3 / 8

theorem cheryl_needed_first_material
  (h_total_used : 0.33333333333333326 = 1 / 3) :
  cheryl_material x = 1 / 3 → x = 3 / 8 :=
by
  intros
  rw [h_total_used] at *
  sorry

end cheryl_needed_first_material_l237_237018


namespace minimum_value_of_f_l237_237901

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

theorem minimum_value_of_f : ∃ y, (∀ x, f x ≥ y) ∧ y = 3 := 
by
  sorry

end minimum_value_of_f_l237_237901


namespace part1_part2_l237_237761

-- Define the operation * on integers
def op (a b : ℤ) : ℤ := a^2 - b + a * b

-- Prove that 2 * 3 = 7 given the defined operation
theorem part1 : op 2 3 = 7 := 
sorry

-- Prove that (-2) * (op 2 (-3)) = 1 given the defined operation
theorem part2 : op (-2) (op 2 (-3)) = 1 := 
sorry

end part1_part2_l237_237761


namespace range_of_m_length_of_chord_l237_237295

-- Definition of Circle C
def CircleC (x y m : ℝ) := x^2 + y^2 - 2*x - 4*y + m = 0

-- Definition of Circle D
def CircleD (x y : ℝ) := (x + 3)^2 + (y + 1)^2 = 16

-- Definition of Line l
def LineL (x y : ℝ) := x + 2*y - 4 = 0

-- Problem 1: Prove range of values for m
theorem range_of_m (m : ℝ) : (∀ x y, CircleC x y m) → m < 5 := by
  sorry

-- Problem 2: Prove length of chord MN
theorem length_of_chord (x y : ℝ) :
  CircleC x y 4 ∧ CircleD x y ∧ LineL x y →
  (∃ MN, MN = (4*Real.sqrt 5) / 5) := by
    sorry

end range_of_m_length_of_chord_l237_237295


namespace andrey_travel_distance_l237_237731

theorem andrey_travel_distance:
  ∃ s t: ℝ, 
    (s = 60 * (t + 4/3) + 20  ∧ s = 90 * (t - 1/3) + 60) ∧ s = 180 :=
by
  sorry

end andrey_travel_distance_l237_237731


namespace solve_inequality_l237_237163

theorem solve_inequality : {x : ℝ | (2 * x - 7) * (x - 3) / x ≥ 0} = {x | (0 < x ∧ x ≤ 3) ∨ (x ≥ 7 / 2)} :=
by
  sorry

end solve_inequality_l237_237163


namespace browser_usage_information_is_false_l237_237856

def num_people_using_A : ℕ := 316
def num_people_using_B : ℕ := 478
def num_people_using_both_A_and_B : ℕ := 104
def num_people_only_using_one_browser : ℕ := 567

theorem browser_usage_information_is_false :
  num_people_only_using_one_browser ≠ (num_people_using_A - num_people_using_both_A_and_B) + (num_people_using_B - num_people_using_both_A_and_B) :=
by
  sorry

end browser_usage_information_is_false_l237_237856


namespace distinct_elements_triangle_not_isosceles_l237_237956

theorem distinct_elements_triangle_not_isosceles
  {a b c : ℝ} (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a + b > c) (h5 : a + c > b) (h6 : b + c > a) :
  ¬(a = b ∨ b = c ∨ a = c) := by
  sorry

end distinct_elements_triangle_not_isosceles_l237_237956


namespace quadratic_inequality_solution_l237_237533

theorem quadratic_inequality_solution (a b : ℝ)
  (h1 : ∀ x, (x > -1 ∧ x < 2) ↔ ax^2 + x + b > 0) :
  a + b = 1 :=
sorry

end quadratic_inequality_solution_l237_237533


namespace percentage_customers_not_pay_tax_l237_237678

theorem percentage_customers_not_pay_tax
  (daily_shoppers : ℕ)
  (weekly_tax_payers : ℕ)
  (h1 : daily_shoppers = 1000)
  (h2 : weekly_tax_payers = 6580)
  : ((7000 - weekly_tax_payers) / 7000) * 100 = 6 := 
by sorry

end percentage_customers_not_pay_tax_l237_237678


namespace gcd_of_360_and_150_is_30_l237_237282

theorem gcd_of_360_and_150_is_30 : Nat.gcd 360 150 = 30 :=
by
  sorry

end gcd_of_360_and_150_is_30_l237_237282


namespace bus_ride_cost_l237_237121

noncomputable def bus_cost : ℝ := 1.75

theorem bus_ride_cost (B T : ℝ) (h1 : T = B + 6.35) (h2 : T + B = 9.85) : B = bus_cost :=
by
  sorry

end bus_ride_cost_l237_237121


namespace current_speed_l237_237003

-- Define the constants based on conditions
def rowing_speed_kmph : Float := 24
def distance_meters : Float := 40
def time_seconds : Float := 4.499640028797696

-- Intermediate calculation: Convert rowing speed from km/h to m/s
def rowing_speed_mps : Float := rowing_speed_kmph * 1000 / 3600

-- Calculate downstream speed
def downstream_speed_mps : Float := distance_meters / time_seconds

-- Define the expected speed of the current
def expected_current_speed : Float := 2.22311111

-- The theorem to prove
theorem current_speed : 
  (downstream_speed_mps - rowing_speed_mps) = expected_current_speed :=
by 
  -- skipping the proof steps, as instructed
  sorry

end current_speed_l237_237003


namespace solution_set_of_inequality_l237_237788

theorem solution_set_of_inequality (x : ℝ) : 
  (3*x^2 - 4*x + 7 > 0) → (1 - 2*x) / (3*x^2 - 4*x + 7) ≥ 0 ↔ x ≤ 1 / 2 :=
by
  intros
  sorry

end solution_set_of_inequality_l237_237788


namespace left_handed_classical_music_lovers_l237_237285

-- Define the conditions
variables (total_people left_handed classical_music right_handed_dislike : ℕ)
variables (x : ℕ) -- x will represent the number of left-handed classical music lovers

-- State the assumptions based on conditions
axiom h1 : total_people = 30
axiom h2 : left_handed = 12
axiom h3 : classical_music = 20
axiom h4 : right_handed_dislike = 3
axiom h5 : 30 = x + (12 - x) + (20 - x) + 3

-- State the theorem to prove
theorem left_handed_classical_music_lovers : x = 5 :=
by {
  -- Skip the proof using sorry
  sorry
}

end left_handed_classical_music_lovers_l237_237285


namespace probability_not_black_l237_237092

theorem probability_not_black (white_balls black_balls red_balls : ℕ) (total_balls : ℕ) (non_black_balls : ℕ) :
  white_balls = 7 → black_balls = 6 → red_balls = 4 →
  total_balls = white_balls + black_balls + red_balls →
  non_black_balls = white_balls + red_balls →
  (non_black_balls / total_balls : ℚ) = 11 / 17 :=
by
  sorry

end probability_not_black_l237_237092


namespace total_amount_l237_237059

variable (Brad Josh Doug : ℝ)

axiom h1 : Josh = 2 * Brad
axiom h2 : Josh = (3 / 4) * Doug
axiom h3 : Doug = 32

theorem total_amount : Brad + Josh + Doug = 68 := by
  sorry

end total_amount_l237_237059


namespace subset_M_union_N_l237_237428

theorem subset_M_union_N (M N P : Set ℝ) (f g : ℝ → ℝ)
  (hM : M = {x | f x = 0} ∧ M ≠ ∅)
  (hN : N = {x | g x = 0} ∧ N ≠ ∅)
  (hP : P = {x | f x * g x = 0} ∧ P ≠ ∅) :
  P ⊆ (M ∪ N) := 
sorry

end subset_M_union_N_l237_237428


namespace part_1_part_2_part_3_l237_237534

variable {f : ℝ → ℝ}

axiom C1 : ∀ x y : ℝ, f (x + y) = f x + f y
axiom C2 : ∀ x : ℝ, x > 0 → f x < 0
axiom C3 : f 3 = -4

theorem part_1 : f 0 = 0 :=
by
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x :=
by
  sorry

theorem part_3 : ∀ x : ℝ, -9 ≤ x ∧ x ≤ 9 → f x ≤ 12 ∧ f x ≥ -12 :=
by
  sorry

end part_1_part_2_part_3_l237_237534


namespace susan_ate_6_candies_l237_237094

-- Definitions based on the problem conditions
def candies_tuesday := 3
def candies_thursday := 5
def candies_friday := 2
def candies_left := 4

-- The total number of candies bought
def total_candies_bought := candies_tuesday + candies_thursday + candies_friday

-- The number of candies eaten
def candies_eaten := total_candies_bought - candies_left

-- Theorem statement to prove that Susan ate 6 candies
theorem susan_ate_6_candies : candies_eaten = 6 :=
by
  -- Proof will be provided here
  sorry

end susan_ate_6_candies_l237_237094


namespace probability_of_both_l237_237868

variable (A B : Prop)

-- Assumptions
def p_A : ℝ := 0.55
def p_B : ℝ := 0.60

-- Probability of both A and B telling the truth at the same time
theorem probability_of_both : p_A * p_B = 0.33 := by
  sorry

end probability_of_both_l237_237868


namespace fallen_tree_trunk_length_l237_237964

noncomputable def tiger_speed (tiger_length : ℕ) (time_pass_grass : ℕ) : ℕ := tiger_length / time_pass_grass

theorem fallen_tree_trunk_length
  (tiger_length : ℕ)
  (time_pass_grass : ℕ)
  (time_pass_tree : ℕ)
  (speed := tiger_speed tiger_length time_pass_grass) :
  tiger_length = 5 →
  time_pass_grass = 1 →
  time_pass_tree = 5 →
  (speed * time_pass_tree) = 25 :=
by
  intros h_tiger_length h_time_pass_grass h_time_pass_tree
  sorry

end fallen_tree_trunk_length_l237_237964


namespace servings_of_honey_l237_237585

theorem servings_of_honey :
  let total_ounces := 37 + 1/3
  let serving_size := 1 + 1/2
  total_ounces / serving_size = 24 + 8/9 :=
by
  sorry

end servings_of_honey_l237_237585


namespace sugar_ratio_l237_237010

theorem sugar_ratio (total_sugar : ℕ)  (bags : ℕ) (remaining_sugar : ℕ) (sugar_each_bag : ℕ) (sugar_fell : ℕ)
  (h1 : total_sugar = 24) (h2 : bags = 4) (h3 : total_sugar - remaining_sugar = sugar_fell) 
  (h4 : total_sugar / bags = sugar_each_bag) (h5 : remaining_sugar = 21) : 
  2 * sugar_fell = sugar_each_bag := by
  -- proof goes here
  sorry

end sugar_ratio_l237_237010


namespace complex_number_identity_l237_237046

theorem complex_number_identity : |-i| + i^2018 = 0 := by
  sorry

end complex_number_identity_l237_237046


namespace price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l237_237479

def cost_price : ℝ := 40
def initial_price : ℝ := 60
def initial_sales_volume : ℕ := 300
def sales_decrease_rate (x : ℕ) : ℕ := 10 * x
def sales_increase_rate (a : ℕ) : ℕ := 20 * a

noncomputable def price_increase_proft_relation (x : ℕ) : ℝ :=
  -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000

theorem price_increase_profit_relation_proof (x : ℕ) (h : 0 ≤ x ∧ x ≤ 30) :
  price_increase_proft_relation x = -10 * (x : ℝ)^2 + 100 * (x : ℝ) + 6000 := sorry

noncomputable def price_decrease_profit_relation (a : ℕ) : ℝ :=
  -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000

theorem price_decrease_profit_relation_proof (a : ℕ) :
  price_decrease_profit_relation a = -20 * (a : ℝ)^2 + 100 * (a : ℝ) + 6000 := sorry

theorem max_profit_price_increase :
  ∃ x : ℕ, 0 ≤ x ∧ x ≤ 30 ∧ price_increase_proft_relation x = 6250 := sorry

end price_increase_profit_relation_proof_price_decrease_profit_relation_proof_max_profit_price_increase_l237_237479


namespace evaluate_sqrt_log_expression_l237_237355

noncomputable def evaluate_log_expression : ℝ :=
  let log3 (x : ℝ) := Real.log x / Real.log 3
  let log4 (x : ℝ) := Real.log x / Real.log 4
  Real.sqrt (log3 8 + log4 8)

theorem evaluate_sqrt_log_expression : evaluate_log_expression = Real.sqrt 3 := 
by
  sorry

end evaluate_sqrt_log_expression_l237_237355


namespace beef_original_weight_l237_237024

theorem beef_original_weight (W : ℝ) (h : 0.65 * W = 546): W = 840 :=
sorry

end beef_original_weight_l237_237024


namespace find_N_l237_237975

theorem find_N :
  ∃ N : ℕ, N > 1 ∧
    (1743 % N = 2019 % N) ∧ (2019 % N = 3008 % N) ∧ N = 23 :=
by
  sorry

end find_N_l237_237975


namespace number_of_boxes_of_nectarines_l237_237858

namespace ProofProblem

/-- Define the given conditions: -/
def crates : Nat := 12
def oranges_per_crate : Nat := 150
def nectarines_per_box : Nat := 30
def total_fruit : Nat := 2280

/-- Define the number of oranges: -/
def total_oranges : Nat := crates * oranges_per_crate

/-- Calculate the number of nectarines: -/
def total_nectarines : Nat := total_fruit - total_oranges

/-- Calculate the number of boxes of nectarines: -/
def boxes_of_nectarines : Nat := total_nectarines / nectarines_per_box

-- Theorem stating that given the conditions, the number of boxes of nectarines is 16.
theorem number_of_boxes_of_nectarines :
  boxes_of_nectarines = 16 := by
  sorry

end ProofProblem

end number_of_boxes_of_nectarines_l237_237858


namespace probability_of_y_gt_2x_l237_237539

noncomputable def probability_y_gt_2x : ℝ := 
  (∫ x in (0:ℝ)..(1000:ℝ), ∫ y in (2*x)..(2000:ℝ), (1 / (1000 * 2000) : ℝ)) * (1000 * 2000)

theorem probability_of_y_gt_2x : probability_y_gt_2x = 0.5 := sorry

end probability_of_y_gt_2x_l237_237539


namespace plate_arrangement_l237_237722

def arrangements_without_restriction : Nat :=
  Nat.factorial 10 / (Nat.factorial 4 * Nat.factorial 3 * Nat.factorial 3)

def arrangements_adjacent_green : Nat :=
  (Nat.factorial 8 / (Nat.factorial 4 * Nat.factorial 3)) * Nat.factorial 3

def allowed_arrangements : Nat :=
  arrangements_without_restriction - arrangements_adjacent_green

theorem plate_arrangement : 
  allowed_arrangements = 2520 := 
by
  sorry

end plate_arrangement_l237_237722


namespace find_cheese_calories_l237_237677

noncomputable def lettuce_calories := 50
noncomputable def carrots_calories := 2 * lettuce_calories
noncomputable def dressing_calories := 210

noncomputable def crust_calories := 600
noncomputable def pepperoni_calories := crust_calories / 3

noncomputable def total_salad_calories := lettuce_calories + carrots_calories + dressing_calories
noncomputable def total_pizza_calories (cheese_calories : ℕ) := crust_calories + pepperoni_calories + cheese_calories

theorem find_cheese_calories (consumed_calories : ℕ) (cheese_calories : ℕ) :
  consumed_calories = 330 →
  1/4 * total_salad_calories + 1/5 * total_pizza_calories cheese_calories = consumed_calories →
  cheese_calories = 400 := by
  sorry

end find_cheese_calories_l237_237677


namespace number_of_black_squares_in_58th_row_l237_237269

theorem number_of_black_squares_in_58th_row :
  let pattern := [1, 0, 0] -- pattern where 1 represents a black square
  let n := 58
  let total_squares := 2 * n - 1 -- total squares in the 58th row
  let black_count := total_squares / 3 -- number of black squares in the repeating pattern
  black_count = 38 :=
by
  let pattern := [1, 0, 0]
  let n := 58
  let total_squares := 2 * n - 1
  let black_count := total_squares / 3
  have black_count_eq_38 : 38 = (115 / 3) := by sorry
  exact black_count_eq_38.symm

end number_of_black_squares_in_58th_row_l237_237269


namespace sixth_ninth_grader_buddy_fraction_l237_237426

theorem sixth_ninth_grader_buddy_fraction
  (s n : ℕ)
  (h_fraction_pairs : n / 4 = s / 3)
  (h_buddy_pairing : (∀ i, i < n -> ∃ j, j < s) 
     ∧ (∀ j, j < s -> ∃ i, i < n) -- each sixth grader paired with one ninth grader and vice versa
  ) :
  (n / 4 + s / 3) / (n + s) = 2 / 7 :=
by 
  sorry

end sixth_ninth_grader_buddy_fraction_l237_237426


namespace Clinton_belts_l237_237444

variable {Shoes Belts Hats : ℕ}

theorem Clinton_belts :
  (Shoes = 14) → (Shoes = 2 * Belts) → Belts = 7 :=
by
  sorry

end Clinton_belts_l237_237444


namespace double_bed_heavier_than_single_bed_l237_237638

theorem double_bed_heavier_than_single_bed 
  (S D : ℝ) 
  (h1 : 5 * S = 50) 
  (h2 : 2 * S + 4 * D = 100) 
  : D - S = 10 :=
sorry

end double_bed_heavier_than_single_bed_l237_237638


namespace max_airlines_in_country_l237_237795

-- Definition of the problem parameters
variable (N k : ℕ) 

-- Definition of the problem conditions
variable (hN_pos : 0 < N)
variable (hk_pos : 0 < k)
variable (hN_ge_k : k ≤ N)

-- Definition of the function calculating the maximum number of air routes
def max_air_routes (N k : ℕ) : ℕ :=
  Nat.choose N 2 - Nat.choose k 2

-- Theorem stating the maximum number of airlines given the conditions
theorem max_airlines_in_country (N k : ℕ) (hN_pos : 0 < N) (hk_pos : 0 < k) (hN_ge_k : k ≤ N) :
  max_air_routes N k = Nat.choose N 2 - Nat.choose k 2 :=
by sorry

end max_airlines_in_country_l237_237795


namespace find_m_l237_237620

def f (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + m
def g (x : ℝ) (m : ℝ) : ℝ := x^2 - 3*x + 5*m

theorem find_m (m : ℝ) : 3 * f 4 m = g 4 m → m = 4 :=
by 
  sorry

end find_m_l237_237620


namespace number_of_female_officers_l237_237439

theorem number_of_female_officers (total_on_duty : ℕ) (female_on_duty : ℕ) (percentage_on_duty : ℚ) : 
  total_on_duty = 500 → 
  female_on_duty = 250 → 
  percentage_on_duty = 1/4 → 
  (female_on_duty : ℚ) = percentage_on_duty * (total_on_duty / 2 : ℚ) →
  (total_on_duty : ℚ) = 4 * female_on_duty →
  total_on_duty = 1000 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end number_of_female_officers_l237_237439


namespace solution_set_l237_237904

theorem solution_set {x : ℝ} :
  abs ((7 - x) / 4) < 3 ∧ 0 ≤ x ↔ 0 ≤ x ∧ x < 19 :=
by
  sorry

end solution_set_l237_237904


namespace min_value_of_sum_of_squares_l237_237903

theorem min_value_of_sum_of_squares (x y z : ℝ) (h : x - 2 * y - 3 * z = 4) : 
  (x^2 + y^2 + z^2) ≥ 8 / 7 :=
sorry

end min_value_of_sum_of_squares_l237_237903


namespace union_A_B_eq_real_subset_A_B_l237_237600

def A (a : ℝ) : Set ℝ := {x : ℝ | a < x ∧ x < 3 + a}
def B : Set ℝ := {x : ℝ | x ≤ -1 ∨ x ≥ 1}

theorem union_A_B_eq_real (a : ℝ) : (A a ∪ B) = Set.univ ↔ -2 ≤ a ∧ a ≤ -1 :=
by
  sorry

theorem subset_A_B (a : ℝ) : A a ⊆ B ↔ (a ≤ -4 ∨ a ≥ 1) :=
by
  sorry

end union_A_B_eq_real_subset_A_B_l237_237600


namespace max_path_length_CQ_D_l237_237704

noncomputable def maxCQDPathLength (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) : ℝ :=
  let r := dAB / 2
  let dCD := dAB - dAC - dBD
  2 * Real.sqrt (r^2 - (dCD / 2)^2)

theorem max_path_length_CQ_D 
  (dAB : ℝ) (dAC : ℝ) (dBD : ℝ) (r := dAB / 2) (dCD := dAB - dAC - dBD) :
  dAB = 16 ∧ dAC = 3 ∧ dBD = 5 ∧ r = 8 ∧ dCD = 8
  → maxCQDPathLength 16 3 5 = 8 * Real.sqrt 3 :=
by
  intros h
  cases h
  sorry

end max_path_length_CQ_D_l237_237704


namespace regular_polygon_sides_l237_237976

theorem regular_polygon_sides (n : ℕ) (h1 : ∀ i, 0 < i → i < n → ∃ θ, θ = 18) :  n = 20 :=
by
    sorry

end regular_polygon_sides_l237_237976


namespace even_function_on_neg_interval_l237_237938

theorem even_function_on_neg_interval
  (f : ℝ → ℝ)
  (h_even : ∀ x : ℝ, f (-x) = f x)
  (h_incr : ∀ x₁ x₂ : ℝ, 1 ≤ x₁ → x₁ < x₂ → x₂ ≤ 3 → f x₁ ≤ f x₂)
  (h_min : ∀ x : ℝ, 1 ≤ x → x ≤ 3 → 0 ≤ f x) :
  (∀ x : ℝ, -3 ≤ x → x ≤ -1 → 0 ≤ f x) ∧ (∀ x₁ x₂ : ℝ, -3 ≤ x₁ → x₁ < x₂ → x₂ ≤ -1 → f x₁ ≥ f x₂) :=
sorry

end even_function_on_neg_interval_l237_237938


namespace tom_total_trip_cost_is_correct_l237_237726

noncomputable def Tom_total_cost : ℝ :=
  let cost_vaccines := 10 * 45
  let cost_doctor := 250
  let total_medical := cost_vaccines + cost_doctor
  
  let insurance_coverage := 0.8 * total_medical
  let out_of_pocket_medical := total_medical - insurance_coverage
  
  let cost_flight := 1200

  let cost_lodging := 7 * 150
  let cost_transportation := 200
  let cost_food := 7 * 60
  let total_local_usd := cost_lodging + cost_transportation + cost_food
  let total_local_bbd := total_local_usd * 2

  let conversion_fee_bbd := 0.03 * total_local_bbd
  let conversion_fee_usd := conversion_fee_bbd / 2

  out_of_pocket_medical + cost_flight + total_local_usd + conversion_fee_usd

theorem tom_total_trip_cost_is_correct : Tom_total_cost = 3060.10 :=
  by
    -- Proof skipped
    sorry

end tom_total_trip_cost_is_correct_l237_237726


namespace lemonade_calories_l237_237500

theorem lemonade_calories 
    (lime_juice_weight : ℕ)
    (lime_juice_calories_per_grams : ℕ)
    (sugar_weight : ℕ)
    (sugar_calories_per_grams : ℕ)
    (water_weight : ℕ)
    (water_calories_per_grams : ℕ)
    (mint_weight : ℕ)
    (mint_calories_per_grams : ℕ)
    :
    lime_juice_weight = 150 →
    lime_juice_calories_per_grams = 30 →
    sugar_weight = 200 →
    sugar_calories_per_grams = 390 →
    water_weight = 500 →
    water_calories_per_grams = 0 →
    mint_weight = 50 →
    mint_calories_per_grams = 7 →
    (300 * ((150 * 30 + 200 * 390 + 500 * 0 + 50 * 7) / 900) = 276) :=
by
  sorry

end lemonade_calories_l237_237500


namespace sphere_center_ratio_l237_237828

/-
Let O be the origin and let (a, b, c) be a fixed point.
A plane with the equation x + 2y + 3z = 6 passes through (a, b, c)
and intersects the x-axis, y-axis, and z-axis at A, B, and C, respectively, all distinct from O.
Let (p, q, r) be the center of the sphere passing through A, B, C, and O.
Prove: a / p + b / q + c / r = 2
-/
theorem sphere_center_ratio (a b c : ℝ) (p q r : ℝ)
  (h_plane : a + 2 * b + 3 * c = 6) 
  (h_p : p = 3)
  (h_q : q = 1.5)
  (h_r : r = 1) :
  a / p + b / q + c / r = 2 :=
by
  sorry

end sphere_center_ratio_l237_237828


namespace rob_has_12_pennies_l237_237141

def total_value_in_dollars (quarters dimes nickels pennies : ℕ) : ℚ :=
  (quarters * 25 + dimes * 10 + nickels * 5 + pennies) / 100

theorem rob_has_12_pennies
  (quarters : ℕ) (dimes : ℕ) (nickels : ℕ) (pennies : ℕ)
  (h1 : quarters = 7) (h2 : dimes = 3) (h3 : nickels = 5) 
  (h4 : total_value_in_dollars quarters dimes nickels pennies = 2.42) :
  pennies = 12 :=
by
  sorry

end rob_has_12_pennies_l237_237141


namespace area_of_win_sector_l237_237768

theorem area_of_win_sector (r : ℝ) (p : ℝ) (A : ℝ) (h_1 : r = 10) (h_2 : p = 1 / 4) (h_3 : A = π * r^2) : 
  (p * A) = 25 * π := 
by
  sorry

end area_of_win_sector_l237_237768


namespace estimate_2_sqrt_5_l237_237107

theorem estimate_2_sqrt_5: 4 < 2 * Real.sqrt 5 ∧ 2 * Real.sqrt 5 < 5 :=
by
  sorry

end estimate_2_sqrt_5_l237_237107


namespace odd_multiple_of_9_is_multiple_of_3_l237_237320

theorem odd_multiple_of_9_is_multiple_of_3 (n : ℕ) (h1 : n % 2 = 1) (h2 : n % 9 = 0) : n % 3 = 0 := 
by sorry

end odd_multiple_of_9_is_multiple_of_3_l237_237320


namespace regular_polygon_sides_l237_237056

theorem regular_polygon_sides (n : ℕ) (h : 2 ≤ n) (h_angle : 180 * (n - 2) / n = 150) : n = 12 :=
sorry

end regular_polygon_sides_l237_237056


namespace func_has_extrema_l237_237744

theorem func_has_extrema (a b c : ℝ) (h_a_nonzero : a ≠ 0) (h_discriminant_positive : b^2 + 8 * a * c > 0) 
    (h_pos_sum_roots : b / a > 0) (h_pos_product_roots : -2 * c / a > 0) : 
    (a * b > 0) ∧ (a * c < 0) :=
by 
  -- Proof skipped.
  sorry

end func_has_extrema_l237_237744


namespace pump_fill_time_without_leak_l237_237015

variable (T : ℕ)

def rate_pump (T : ℕ) : ℚ := 1 / T
def rate_leak : ℚ := 1 / 20

theorem pump_fill_time_without_leak : rate_pump T - rate_leak = rate_leak → T = 10 := by 
  intro h
  sorry

end pump_fill_time_without_leak_l237_237015


namespace rectangle_MQ_l237_237888

theorem rectangle_MQ :
  ∀ (PQ QR PM MQ : ℝ),
    PQ = 4 →
    QR = 10 →
    PM = MQ →
    MQ = 2 * Real.sqrt 10 → 
    0 < MQ
:= by
  intros PQ QR PM MQ h1 h2 h3 h4
  sorry

end rectangle_MQ_l237_237888


namespace claire_sleep_hours_l237_237074

def hours_in_day := 24
def cleaning_hours := 4
def cooking_hours := 2
def crafting_hours := 5
def tailoring_hours := crafting_hours

theorem claire_sleep_hours :
  hours_in_day - (cleaning_hours + cooking_hours + crafting_hours + tailoring_hours) = 8 := by
  sorry

end claire_sleep_hours_l237_237074


namespace negation_of_exists_l237_237130

theorem negation_of_exists (x : ℝ) (h : ∃ x : ℝ, x^2 - x + 1 ≤ 0) : 
  (∀ x : ℝ, x^2 - x + 1 > 0) :=
sorry

end negation_of_exists_l237_237130


namespace sin_cos_values_trigonometric_expression_value_l237_237917

-- Define the conditions
variables (α : ℝ)
def point_on_terminal_side (x y : ℝ) (r : ℝ) : Prop :=
  (x = 3) ∧ (y = 4) ∧ (r = 5)

-- Define the problem statements
theorem sin_cos_values (x y r : ℝ) (h: point_on_terminal_side x y r) : 
  (Real.sin α = 4 / 5) ∧ (Real.cos α = 3 / 5) :=
sorry

theorem trigonometric_expression_value (h1: Real.sin α = 4 / 5) (h2: Real.cos α = 3 / 5) :
  (2 * Real.cos (π / 2 - α) - Real.cos (π + α)) / (2 * Real.sin (π - α)) = 11 / 8 :=
sorry

end sin_cos_values_trigonometric_expression_value_l237_237917


namespace find_a1_over_1_minus_q_l237_237191

noncomputable def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem find_a1_over_1_minus_q 
  (a : ℕ → ℝ) (q : ℝ)
  (h_geom : geometric_sequence a q)
  (h1 : a 1 + a 2 + a 3 + a 4 = 3)
  (h2 : a 5 + a 6 + a 7 + a 8 = 48) :
  (a 1) / (1 - q) = -1 / 5 :=
sorry

end find_a1_over_1_minus_q_l237_237191


namespace decreasing_function_condition_l237_237313

theorem decreasing_function_condition (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, x ≤ 3 → deriv f x ≤ 0) ↔ (m ≥ 1) :=
by 
  sorry

end decreasing_function_condition_l237_237313


namespace num_distinct_ordered_pairs_l237_237178

theorem num_distinct_ordered_pairs (a b c : ℕ) (h₀ : a + b + c = 50) (h₁ : c = 10) (h₂ : 0 < a ∧ 0 < b) :
  ∃ n : ℕ, n = 39 := 
sorry

end num_distinct_ordered_pairs_l237_237178


namespace john_can_see_jane_for_45_minutes_l237_237404

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

end john_can_see_jane_for_45_minutes_l237_237404


namespace find_angle_degree_l237_237897

theorem find_angle_degree (x : ℝ) (h : 90 - x = 0.4 * (180 - x)) : x = 30 := by
  sorry

end find_angle_degree_l237_237897


namespace element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l237_237549

-- defining element, nuclide, and valence based on protons, neutrons, and electrons
def Element (protons : ℕ) := protons
def Nuclide (protons neutrons : ℕ) := (protons, neutrons)
def ChemicalProperties (outermostElectrons : ℕ) := outermostElectrons
def HighestPositiveValence (mainGroupNum : ℕ) := mainGroupNum

-- The proof problems as Lean theorems
theorem element_type_determined_by_protons (protons : ℕ) :
  Element protons = protons := sorry

theorem nuclide_type_determined_by_protons_neutrons (protons neutrons : ℕ) :
  Nuclide protons neutrons = (protons, neutrons) := sorry

theorem chemical_properties_determined_by_outermost_electrons (outermostElectrons : ℕ) :
  ChemicalProperties outermostElectrons = outermostElectrons := sorry
  
theorem highest_positive_valence_determined_by_main_group_num (mainGroupNum : ℕ) :
  HighestPositiveValence mainGroupNum = mainGroupNum := sorry

end element_type_determined_by_protons_nuclide_type_determined_by_protons_neutrons_chemical_properties_determined_by_outermost_electrons_highest_positive_valence_determined_by_main_group_num_l237_237549


namespace symmetry_x_axis_l237_237211

theorem symmetry_x_axis (a b : ℝ) (h1 : a - 3 = 2) (h2 : 1 = -(b + 1)) : a + b = 3 :=
by
  sorry

end symmetry_x_axis_l237_237211


namespace watermelon_slices_l237_237111

theorem watermelon_slices (total_seeds slices_black seeds_white seeds_per_slice num_slices : ℕ)
  (h1 : seeds_black = 20)
  (h2 : seeds_white = 20)
  (h3 : seeds_per_slice = seeds_black + seeds_white)
  (h4 : total_seeds = 1600)
  (h5 : num_slices = total_seeds / seeds_per_slice) :
  num_slices = 40 :=
by
  sorry

end watermelon_slices_l237_237111


namespace isosceles_triangle_l237_237245

variable (a b c : ℝ)
variable (α β γ : ℝ)
variable (h1 : a + b = Real.tan (γ / 2) * (a * Real.tan α + b * Real.tan β))
variable (triangle_angles : γ = π - (α + β))

theorem isosceles_triangle : α = β :=
by
  sorry

end isosceles_triangle_l237_237245


namespace evaluate_ceiling_sum_l237_237576

theorem evaluate_ceiling_sum :
  (⌈Real.sqrt (16 / 9)⌉ : ℤ) + (⌈(16 / 9: ℝ)⌉ : ℤ) + (⌈(16 / 9: ℝ)^2⌉ : ℤ) = 8 := 
by
  -- Placeholder for proof
  sorry

end evaluate_ceiling_sum_l237_237576


namespace problem_1_problem_2_l237_237069

def set_A := { y : ℝ | 2 < y ∧ y < 3 }
def set_B := { x : ℝ | x > 1 ∨ x < -1 }

theorem problem_1 : { x : ℝ | x ∈ set_A ∧ x ∈ set_B } = { x : ℝ | 2 < x ∧ x < 3 } :=
by
  sorry

def set_C := { x : ℝ | x ∈ set_B ∧ ¬(x ∈ set_A) }

theorem problem_2 : set_C = { x : ℝ | x < -1 ∨ (1 < x ∧ x ≤ 2) ∨ x ≥ 3 } :=
by
  sorry

end problem_1_problem_2_l237_237069


namespace minimum_boxes_l237_237679

theorem minimum_boxes (x y z : ℕ) (h1 : 50 * x = 40 * y) (h2 : 50 * x = 25 * z) :
  x + y + z = 17 :=
by
  -- Prove that given these equations, the minimum total number of boxes (x + y + z) is 17
  sorry

end minimum_boxes_l237_237679


namespace complex_expr_simplify_l237_237400

noncomputable def complex_demo : Prop :=
  let i := Complex.I
  7 * (4 + 2 * i) - 2 * i * (7 + 3 * i) = (34 : ℂ)

theorem complex_expr_simplify : 
  complex_demo :=
by
  -- proof skipped
  sorry

end complex_expr_simplify_l237_237400


namespace max_area_rectangular_playground_l237_237482

theorem max_area_rectangular_playground (l w : ℝ) 
  (h_perimeter : 2 * l + 2 * w = 360) 
  (h_length : l ≥ 90) 
  (h_width : w ≥ 50) : 
  (l * w) ≤ 8100 :=
by
  sorry

end max_area_rectangular_playground_l237_237482


namespace kyro_percentage_paid_l237_237321

theorem kyro_percentage_paid
    (aryan_debt : ℕ) -- Aryan owes Fernanda $1200
    (kyro_debt : ℕ) -- Kyro owes Fernanda
    (aryan_debt_twice_kyro_debt : aryan_debt = 2 * kyro_debt) -- Aryan's debt is twice what Kyro owes
    (aryan_payment : ℕ) -- Aryan's payment
    (aryan_payment_percentage : aryan_payment = 60 * aryan_debt / 100) -- Aryan pays 60% of her debt
    (initial_savings : ℕ) -- Initial savings in Fernanda's account
    (final_savings : ℕ) -- Final savings in Fernanda's account
    (initial_savings_cond : initial_savings = 300) -- Fernanda's initial savings is $300
    (final_savings_cond : final_savings = 1500) -- Fernanda's final savings is $1500
    : kyro_payment = 80 * kyro_debt / 100 := -- Kyro paid 80% of her debt
by {
    sorry
}

end kyro_percentage_paid_l237_237321


namespace problem_statement_false_adjacent_complementary_l237_237461

-- Definition of straight angle, supplementary angles, and complementary angles.
def is_straight_angle (θ : ℝ) : Prop := θ = 180
def are_supplementary (θ ψ : ℝ) : Prop := θ + ψ = 180
def are_complementary (θ ψ : ℝ) : Prop := θ + ψ = 90

-- Definition of adjacent angles (for completeness, though we don't use adjacency differently right now)
def are_adjacent (θ ψ : ℝ) : Prop := ∀ x, θ + x + ψ + x = θ + ψ -- Simplified

-- Additional conditions that could be true or false -- we need one of them to be false.
def false_statement_D (θ ψ : ℝ) : Prop :=
  are_complementary θ ψ → are_adjacent θ ψ

theorem problem_statement_false_adjacent_complementary :
  ∃ (θ ψ : ℝ), ¬ false_statement_D θ ψ :=
by
  sorry

end problem_statement_false_adjacent_complementary_l237_237461


namespace parallelogram_area_l237_237421

theorem parallelogram_area (b : ℝ) (h : ℝ) (A : ℝ) (base_condition : b = 8) (altitude_condition : h = 2 * b) : 
  A = 128 :=
by 
  sorry

end parallelogram_area_l237_237421


namespace f_eq_for_neg_l237_237502

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Given conditions
noncomputable def f (x : ℝ) : ℝ :=
  if h : x ≥ 0 then x * (2^(-x) + 1) else x * (2^x + 1)

-- Theorem to prove
theorem f_eq_for_neg (f : ℝ → ℝ) (h1 : is_odd f) (h2 : ∀ x : ℝ, 0 ≤ x → f x = x * (2^(-x) + 1)) :
  ∀ x : ℝ, x < 0 → f x = x * (2^x + 1) :=
by
  intro x hx
  sorry

end f_eq_for_neg_l237_237502


namespace find_x_l237_237153

theorem find_x (x : ℕ) (hv1 : x % 6 = 0) (hv2 : x^2 > 144) (hv3 : x < 30) : x = 18 ∨ x = 24 :=
  sorry

end find_x_l237_237153


namespace probability_journalist_A_to_group_A_l237_237974

open Nat

theorem probability_journalist_A_to_group_A :
  let group_A := 0
  let group_B := 1
  let group_C := 2
  let journalists := [0, 1, 2, 3]  -- four journalists

  -- total number of ways to distribute 4 journalists into 3 groups such that each group has at least one journalist
  let total_ways := 36

  -- number of ways to assign journalist 0 to group A specifically
  let favorable_ways := 12

  -- probability calculation
  ∃ (prob : ℚ), prob = favorable_ways / total_ways ∧ prob = 1 / 3 :=
sorry

end probability_journalist_A_to_group_A_l237_237974


namespace problem_statement_l237_237804

variable (U M N : Set ℕ)

theorem problem_statement (hU : U = {1, 2, 3, 4, 5})
                         (hM : M = {1, 4})
                         (hN : N = {2, 5}) :
                         N ∪ (U \ M) = {2, 3, 5} :=
by sorry

end problem_statement_l237_237804


namespace shopkeeper_loss_percent_l237_237152

theorem shopkeeper_loss_percent (cost_price goods_lost_percent profit_percent : ℝ)
    (h_cost_price : cost_price = 100)
    (h_goods_lost_percent : goods_lost_percent = 0.4)
    (h_profit_percent : profit_percent = 0.1) :
    let initial_revenue := cost_price * (1 + profit_percent)
    let goods_lost_value := cost_price * goods_lost_percent
    let remaining_goods_value := cost_price - goods_lost_value
    let remaining_revenue := remaining_goods_value * (1 + profit_percent)
    let loss_in_revenue := initial_revenue - remaining_revenue
    let loss_percent := (loss_in_revenue / initial_revenue) * 100
    loss_percent = 40 := sorry

end shopkeeper_loss_percent_l237_237152


namespace solve_log_eq_l237_237411

noncomputable def log3 (x : ℝ) := Real.log x / Real.log 3

theorem solve_log_eq :
  (∃ x : ℝ, log3 ((5 * x + 15) / (7 * x - 5)) + log3 ((7 * x - 5) / (2 * x - 3)) = 3 ∧ x = 96 / 49) :=
by
  sorry

end solve_log_eq_l237_237411


namespace caleb_double_burgers_count_l237_237449

theorem caleb_double_burgers_count
    (S D : ℕ)
    (cost_single cost_double total_hamburgers total_cost : ℝ)
    (h1 : cost_single = 1.00)
    (h2 : cost_double = 1.50)
    (h3 : total_hamburgers = 50)
    (h4 : total_cost = 66.50)
    (h5 : S + D = total_hamburgers)
    (h6 : cost_single * S + cost_double * D = total_cost) :
    D = 33 := 
sorry

end caleb_double_burgers_count_l237_237449


namespace money_total_l237_237626

theorem money_total (A B C : ℕ) (h1 : A + C = 200) (h2 : B + C = 350) (h3 : C = 100) : A + B + C = 450 :=
by {
  sorry
}

end money_total_l237_237626


namespace arrows_from_530_to_533_l237_237639

-- Define what it means for the pattern to be cyclic with period 5
def cycle_period (n m : Nat) : Prop := n % m = 0

-- Define the equivalent points on the circular track
def equiv_point (n : Nat) (m : Nat) : Nat := n % m

-- Given conditions
def arrow_pattern : Prop :=
  ∀ n : Nat, cycle_period n 5 ∧
  (equiv_point 530 5 = 0) ∧ (equiv_point 533 5 = 3)

-- The theorem to be proved
theorem arrows_from_530_to_533 :
  (∃ seq : List (Nat × Nat),
    seq = [(0, 1), (1, 2), (2, 3)]) :=
sorry

end arrows_from_530_to_533_l237_237639


namespace taxi_ride_cost_l237_237935

theorem taxi_ride_cost (base_fare : ℚ) (cost_per_mile : ℚ) (distance : ℕ) :
  base_fare = 2 ∧ cost_per_mile = 0.30 ∧ distance = 10 →
  base_fare + cost_per_mile * distance = 5 :=
by
  sorry

end taxi_ride_cost_l237_237935


namespace parabola_translation_correct_l237_237418

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Given vertex translation
def translated_vertex : ℝ × ℝ := (-2, -2)

-- Define the translated parabola equation
def translated_parabola (x : ℝ) : ℝ := 3 * (x + 2)^2 - 2

-- The proof statement
theorem parabola_translation_correct :
  ∀ x, translated_parabola x = 3 * (x + 2)^2 - 2 := by
  sorry

end parabola_translation_correct_l237_237418


namespace distance_between_A_and_B_l237_237505

theorem distance_between_A_and_B
  (vA vB D : ℝ)
  (hvB : vB = (3/2) * vA)
  (second_meeting_distance : 20 = D * 2 / 5) : 
  D = 50 := 
by
  sorry

end distance_between_A_and_B_l237_237505


namespace youngest_child_age_l237_237084

theorem youngest_child_age (x : ℕ) (h : x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 55) : x = 7 := 
by
  sorry

end youngest_child_age_l237_237084


namespace cubic_polynomial_evaluation_l237_237801

theorem cubic_polynomial_evaluation (Q : ℚ → ℚ) (m : ℚ)
  (hQ0 : Q 0 = 2 * m) 
  (hQ1 : Q 1 = 5 * m) 
  (hQm1 : Q (-1) = 0) : 
  Q 2 + Q (-2) = 8 * m := 
by
  sorry

end cubic_polynomial_evaluation_l237_237801


namespace circle_center_coordinates_l237_237283

theorem circle_center_coordinates :
  ∃ (h k : ℝ), (∀ x y : ℝ, x^2 + y^2 - 4 * x + 6 * y = 0 ↔ (x - h)^2 + (y - k)^2 = 13) ∧ h = 2 ∧ k = -3 :=
sorry

end circle_center_coordinates_l237_237283


namespace max_value_of_E_l237_237602

theorem max_value_of_E (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a ^ 5 + b ^ 5 = a ^ 3 + b ^ 3) : 
  a^2 - a*b + b^2 ≤ 1 :=
sorry

end max_value_of_E_l237_237602


namespace shaded_area_is_correct_l237_237263

-- Definitions based on the conditions
def is_square (s : ℝ) (area : ℝ) : Prop := s * s = area
def rect_area (l w : ℝ) : ℝ := l * w

variables (s : ℝ) (area_s : ℝ) (rect1_l rect1_w rect2_l rect2_w : ℝ)

-- Given conditions
def square := is_square s area_s
def rect1 := rect_area rect1_l rect1_w
def rect2 := rect_area rect2_l rect2_w

-- Problem statement: Prove the area of the shaded region
theorem shaded_area_is_correct
  (s: ℝ)
  (rect1_l rect1_w rect2_l rect2_w : ℝ)
  (h_square: is_square s 16)
  (h_rect1: rect_area rect1_l rect1_w = 6)
  (h_rect2: rect_area rect2_l rect2_w = 2) :
  (16 - (6 + 2) = 8) := 
  sorry

end shaded_area_is_correct_l237_237263


namespace download_time_ratio_l237_237891

-- Define the conditions of the problem
def mac_download_time : ℕ := 10
def audio_glitches : ℕ := 2 * 4
def video_glitches : ℕ := 6
def time_with_glitches : ℕ := audio_glitches + video_glitches
def time_without_glitches : ℕ := 2 * time_with_glitches
def total_time : ℕ := 82

-- Define the Windows download time as a variable
def windows_download_time : ℕ := total_time - (mac_download_time + time_with_glitches + time_without_glitches)

-- Prove the required ratio
theorem download_time_ratio : 
  (windows_download_time / mac_download_time = 3) :=
by
  -- Perform a straightforward calculation as defined in the conditions and solution steps
  sorry

end download_time_ratio_l237_237891


namespace subtraction_of_largest_three_digit_from_smallest_five_digit_l237_237990

def largest_three_digit_number : ℕ := 999
def smallest_five_digit_number : ℕ := 10000

theorem subtraction_of_largest_three_digit_from_smallest_five_digit :
  smallest_five_digit_number - largest_three_digit_number = 9001 :=
by
  sorry

end subtraction_of_largest_three_digit_from_smallest_five_digit_l237_237990


namespace bucket_weight_one_third_l237_237946

theorem bucket_weight_one_third 
    (x y c b : ℝ) 
    (h1 : x + 3/4 * y = c)
    (h2 : x + 1/2 * y = b) :
    x + 1/3 * y = 5/3 * b - 2/3 * c :=
by
  sorry

end bucket_weight_one_third_l237_237946


namespace mother_picked_38_carrots_l237_237792

theorem mother_picked_38_carrots
  (haley_carrots : ℕ)
  (good_carrots : ℕ)
  (bad_carrots : ℕ)
  (total_carrots_picked : ℕ)
  (mother_carrots : ℕ)
  (h1 : haley_carrots = 39)
  (h2 : good_carrots = 64)
  (h3 : bad_carrots = 13)
  (h4 : total_carrots_picked = good_carrots + bad_carrots)
  (h5 : total_carrots_picked = haley_carrots + mother_carrots) :
  mother_carrots = 38 :=
by
  sorry

end mother_picked_38_carrots_l237_237792


namespace total_amount_paid_l237_237850

-- Definitions based on the conditions.
def cost_per_pizza : ℝ := 12
def delivery_charge : ℝ := 2
def distance_threshold : ℝ := 1000 -- distance in meters
def park_distance : ℝ := 100
def building_distance : ℝ := 2000

def pizzas_at_park : ℕ := 3
def pizzas_at_building : ℕ := 2

-- The proof problem stating the total amount paid to Jimmy.
theorem total_amount_paid :
  let total_pizzas := pizzas_at_park + pizzas_at_building
  let cost_without_delivery := total_pizzas * cost_per_pizza
  let park_charge := if park_distance > distance_threshold then pizzas_at_park * delivery_charge else 0
  let building_charge := if building_distance > distance_threshold then pizzas_at_building * delivery_charge else 0
  let total_cost := cost_without_delivery + park_charge + building_charge
  total_cost = 64 :=
by
  sorry

end total_amount_paid_l237_237850


namespace value_of_f_f_3_l237_237108

def f (x : ℝ) := 3 * x^2 + 3 * x - 2

theorem value_of_f_f_3 : f (f 3) = 3568 :=
by {
  -- Definition of f is already given in the conditions
  sorry
}

end value_of_f_f_3_l237_237108


namespace distance_big_rock_correct_l237_237783

noncomputable def rower_in_still_water := 7 -- km/h
noncomputable def river_flow := 2 -- km/h
noncomputable def total_trip_time := 1 -- hour

def distance_to_big_rock (D : ℝ) :=
  (D / (rower_in_still_water - river_flow)) + (D / (rower_in_still_water + river_flow)) = total_trip_time

theorem distance_big_rock_correct {D : ℝ} (h : distance_to_big_rock D) : D = 45 / 14 :=
sorry

end distance_big_rock_correct_l237_237783


namespace A_can_finish_remaining_work_in_4_days_l237_237517

theorem A_can_finish_remaining_work_in_4_days
  (A_days : ℕ) (B_days : ℕ) (B_worked_days : ℕ) : 
  A_days = 12 → B_days = 15 → B_worked_days = 10 → 
  (4 * (1 / A_days) = 1 / 3 - B_worked_days * (1 / B_days)) :=
by
  intros hA hB hBwork
  sorry

end A_can_finish_remaining_work_in_4_days_l237_237517


namespace linear_term_zero_implies_sum_zero_l237_237881

-- Define the condition that the product does not have a linear term
def no_linear_term (x a b : ℝ) : Prop :=
  (x + a) * (x + b) = x^2 + (a + b) * x + a * b

-- Given the condition, we need to prove that a + b = 0
theorem linear_term_zero_implies_sum_zero {a b : ℝ} (h : ∀ x : ℝ, no_linear_term x a b) : a + b = 0 :=
by 
  sorry

end linear_term_zero_implies_sum_zero_l237_237881


namespace line_tangent_through_A_l237_237855

theorem line_tangent_through_A {A : ℝ × ℝ} (hA : A = (1, 2)) : 
  ∃ m b : ℝ, (b = 2) ∧ (∀ x : ℝ, y = m * x + b) ∧ (∀ y x : ℝ, y^2 = 4*x → y = 2) :=
by
  sorry

end line_tangent_through_A_l237_237855


namespace domain_of_f_l237_237493

noncomputable def f (x : ℝ) : ℝ := (x^3 - 125) / (x + 5)

theorem domain_of_f :
  {x : ℝ | ∃ y : ℝ, f x = y} = {x : ℝ | x ≠ -5} := 
by
  sorry

end domain_of_f_l237_237493


namespace cubic_roots_quadratic_l237_237243

theorem cubic_roots_quadratic (A B C p : ℚ)
  (hA : A ≠ 0)
  (h1 : (∀ x : ℚ, A * x^2 + B * x + C = 0 ↔ x = (root1) ∨ x = (root2)))
  (h2 : root1 + root2 = - B / A)
  (h3 : root1 * root2 = C / A)
  (new_eq : ∀ x : ℚ, x^2 + p*x + q = 0 ↔ x = root1^3 ∨ x = root2^3) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by
  sorry

end cubic_roots_quadratic_l237_237243


namespace cost_of_orange_juice_l237_237253

theorem cost_of_orange_juice (O : ℝ) (H1 : ∀ (apple_juice_cost : ℝ), apple_juice_cost = 0.60 ):
  let total_bottles := 70
  let total_cost := 46.20
  let orange_juice_bottles := 42
  let apple_juice_bottles := total_bottles - orange_juice_bottles
  let equation := (orange_juice_bottles * O + apple_juice_bottles * 0.60 = total_cost)
  equation -> O = 0.70 := by
  sorry

end cost_of_orange_juice_l237_237253


namespace plane_equidistant_from_B_and_C_l237_237606

-- Define points B and C
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def B : Point3D := { x := 4, y := 1, z := 0 }
def C : Point3D := { x := 2, y := 0, z := 3 }

-- Define the predicate for a plane equation
def plane_eq (a b c d : ℝ) (P : Point3D) : Prop :=
  a * P.x + b * P.y + c * P.z + d = 0

-- The problem statement
theorem plane_equidistant_from_B_and_C :
  ∃ D : ℝ, plane_eq (-2) (-1) 3 D { x := B.x, y := B.y, z := B.z } ∧
            plane_eq (-2) (-1) 3 D { x := C.x, y := C.y, z := C.z } :=
sorry

end plane_equidistant_from_B_and_C_l237_237606


namespace equilateral_triangle_side_length_l237_237499

noncomputable def side_length_of_triangle (PQ PR PS : ℕ) : ℝ := 
  let s := 8 * Real.sqrt 3
  s

theorem equilateral_triangle_side_length (PQ PR PS : ℕ) (P_inside_triangle : true) 
  (Q_foot : true) (R_foot : true) (S_foot : true)
  (hPQ : PQ = 2) (hPR : PR = 4) (hPS : PS = 6) : 
  side_length_of_triangle PQ PR PS = 8 * Real.sqrt 3 := 
sorry

end equilateral_triangle_side_length_l237_237499


namespace total_games_l237_237067

theorem total_games (teams : ℕ) (games_per_pair : ℕ) (h_teams : teams = 12) (h_games_per_pair : games_per_pair = 4) : 
  (teams * (teams - 1) / 2) * games_per_pair = 264 :=
by
  sorry

end total_games_l237_237067


namespace tan_subtraction_formula_l237_237186

theorem tan_subtraction_formula 
  (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) :
  Real.tan (α - β) = 1 / 7 := 
by
  sorry

end tan_subtraction_formula_l237_237186


namespace necessary_but_not_sufficient_l237_237568

def simple_prop (p q : Prop) :=
  (¬ (p ∧ q)) → (¬ (p ∨ q))

theorem necessary_but_not_sufficient (p q : Prop) (h : simple_prop p q) :
  ((¬ (p ∧ q)) → (¬ (p ∨ q))) ∧ ¬ ((¬ (p ∨ q)) → (¬ (p ∧ q))) := by
sorry

end necessary_but_not_sufficient_l237_237568


namespace sticker_height_enlarged_l237_237437

theorem sticker_height_enlarged (orig_width orig_height new_width : ℝ)
    (h1 : orig_width = 3) (h2 : orig_height = 2) (h3 : new_width = 12) :
    new_width / orig_width * orig_height = 8 :=
by
  rw [h1, h2, h3]
  norm_num

end sticker_height_enlarged_l237_237437


namespace chess_tournament_games_l237_237450

theorem chess_tournament_games (P : ℕ) (TotalGames : ℕ) (hP : P = 21) (hTotalGames : TotalGames = 210) : 
  ∃ G : ℕ, G = 20 ∧ TotalGames = (P * (P - 1)) / 2 :=
by
  sorry

end chess_tournament_games_l237_237450


namespace students_not_enrolled_in_course_l237_237225

def total_students : ℕ := 150
def french_students : ℕ := 61
def german_students : ℕ := 32
def spanish_students : ℕ := 45
def french_and_german : ℕ := 15
def french_and_spanish : ℕ := 12
def german_and_spanish : ℕ := 10
def all_three_courses : ℕ := 5

theorem students_not_enrolled_in_course : total_students - 
    (french_students + german_students + spanish_students - 
     french_and_german - french_and_spanish - german_and_spanish + 
     all_three_courses) = 44 := by
  sorry

end students_not_enrolled_in_course_l237_237225


namespace toys_calculation_l237_237080

-- Define the number of toys each person has as variables
variables (Jason John Rachel : ℕ)

-- State the conditions
variables (h1 : Jason = 3 * John)
variables (h2 : John = Rachel + 6)
variables (h3 : Jason = 21)

-- Define the theorem to prove the number of toys Rachel has
theorem toys_calculation : Rachel = 1 :=
by {
  sorry
}

end toys_calculation_l237_237080


namespace determine_initial_fund_l237_237050

def initial_amount_fund (n : ℕ) := 60 * n + 30 - 10

theorem determine_initial_fund (n : ℕ) (h : 50 * n + 110 = 60 * n - 10) : initial_amount_fund n = 740 :=
by
  -- we skip the proof steps here
  sorry

end determine_initial_fund_l237_237050


namespace solution_set_inequality_l237_237155

theorem solution_set_inequality (x : ℝ) :
  (3 * x + 2 ≥ 1 ∧ (5 - x) / 2 < 0) ↔ (-1 / 3 ≤ x ∧ x < 5) :=
by
  sorry

end solution_set_inequality_l237_237155


namespace ratio_of_width_to_perimeter_l237_237957

-- Condition definitions
def length := 22
def width := 13
def perimeter := 2 * (length + width)

-- Statement of the problem in Lean 4
theorem ratio_of_width_to_perimeter : width = 13 ∧ length = 22 → width * 70 = 13 * perimeter :=
by
  sorry

end ratio_of_width_to_perimeter_l237_237957


namespace reducedRatesFraction_l237_237140

variable (total_hours_per_week : ℕ := 168)
variable (reduced_rate_hours_weekdays : ℕ := 12 * 5)
variable (reduced_rate_hours_weekends : ℕ := 24 * 2)

theorem reducedRatesFraction
  (h1 : total_hours_per_week = 7 * 24)
  (h2 : reduced_rate_hours_weekdays = 12 * 5)
  (h3 : reduced_rate_hours_weekends = 24 * 2) :
  (reduced_rate_hours_weekdays + reduced_rate_hours_weekends) / total_hours_per_week = 9 / 14 := 
  sorry

end reducedRatesFraction_l237_237140


namespace sixth_term_sequence_l237_237643

theorem sixth_term_sequence (a b c d : ℚ)
  (h1 : a = 1/4 * (3 + b))
  (h2 : b = 1/4 * (a + c))
  (h3 : c = 1/4 * (b + 48))
  (h4 : 48 = 1/4 * (c + d)) :
  d = 2001 / 14 :=
sorry

end sixth_term_sequence_l237_237643


namespace parabola_unique_solution_l237_237041

theorem parabola_unique_solution (a : ℝ) :
  (∃ x : ℝ, (0 ≤ x^2 + a * x + 5) ∧ (x^2 + a * x + 5 ≤ 4)) → (a = 2 ∨ a = -2) :=
by
  sorry

end parabola_unique_solution_l237_237041


namespace imaginary_part_of_z_l237_237618

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 1 - 3 * I) : z.im = -2 := by
  sorry

end imaginary_part_of_z_l237_237618


namespace technicians_in_workshop_l237_237442

theorem technicians_in_workshop :
  (∃ T R: ℕ, T + R = 42 ∧ 8000 * 42 = 18000 * T + 6000 * R) → ∃ T: ℕ, T = 7 :=
by
  sorry

end technicians_in_workshop_l237_237442


namespace side_length_of_square_l237_237390

theorem side_length_of_square (P : ℝ) (h1 : P = 12 / 25) : 
  P / 4 = 0.12 := 
by
  sorry

end side_length_of_square_l237_237390


namespace simplify_expression_l237_237065

theorem simplify_expression (m : ℝ) (h1 : m ≠ 3) :
  (m / (m - 3) + 2 / (3 - m)) / ((m - 2) / (m^2 - 6 * m + 9)) = m - 3 := 
by
  sorry

end simplify_expression_l237_237065


namespace find_a_l237_237906

theorem find_a (x y : ℝ) (h1 : y > x) (h2 : x > 0) (h3 : (x / y) + (y / x) = 8) :
  (x + y) / (x - y) = Real.sqrt (5 / 3) :=
sorry

end find_a_l237_237906


namespace take_home_pay_is_correct_l237_237224

-- Definitions and Conditions
def pay : ℤ := 650
def tax_rate : ℤ := 10

-- Calculations
def tax_amount := pay * tax_rate / 100
def take_home_pay := pay - tax_amount

-- The Proof Statement
theorem take_home_pay_is_correct : take_home_pay = 585 := by
  sorry

end take_home_pay_is_correct_l237_237224


namespace probability_exactly_five_shots_expected_shots_to_hit_all_l237_237803

-- Part (a)
theorem probability_exactly_five_shots
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∃ t₁ t₂ t₃ : ℕ, t₁ ≠ t₂ ∧ t₁ ≠ t₃ ∧ t₂ ≠ t₃ ∧ t₁ + t₂ + t₃ = 5) →
  6 * p ^ 3 * (1 - p) ^ 2 = 6 * p ^ 3 * (1 - p) ^ 2 :=
by sorry

-- Part (b)
theorem expected_shots_to_hit_all
  (p : ℝ) (hp : 0 < p ∧ p ≤ 1) :
  (∀ t: ℕ, (t * p * (1 - p)^(t-1)) = 1/p) →
  3 * (1/p) = 3 / p :=
by sorry

end probability_exactly_five_shots_expected_shots_to_hit_all_l237_237803


namespace volume_of_sphere_l237_237522

theorem volume_of_sphere
    (area1 : ℝ) (area2 : ℝ) (distance : ℝ)
    (h1 : area1 = 9 * π)
    (h2 : area2 = 16 * π)
    (h3 : distance = 1) :
    ∃ R : ℝ, (4 / 3) * π * R ^ 3 = 500 * π / 3 :=
by
  sorry

end volume_of_sphere_l237_237522


namespace holiday_price_correct_l237_237419

-- Define the problem parameters
def original_price : ℝ := 250
def first_discount_rate : ℝ := 0.40
def second_discount_rate : ℝ := 0.10

-- Define the calculation for the first discount
def price_after_first_discount (original: ℝ) (rate: ℝ) : ℝ :=
  original * (1 - rate)

-- Define the calculation for the second discount
def price_after_second_discount (intermediate: ℝ) (rate: ℝ) : ℝ :=
  intermediate * (1 - rate)

-- The final Lean statement to prove
theorem holiday_price_correct : 
  price_after_second_discount (price_after_first_discount original_price first_discount_rate) second_discount_rate = 135 :=
by
  sorry

end holiday_price_correct_l237_237419


namespace codger_feet_l237_237794

theorem codger_feet (F : ℕ) (h1 : 6 = 2 * (5 - 1) * F) : F = 3 := by
  sorry

end codger_feet_l237_237794


namespace smallest_x_y_sum_l237_237713

theorem smallest_x_y_sum :
  ∃ x y : ℕ,
    0 < x ∧ 0 < y ∧ x ≠ y ∧ (1 / (x : ℝ) + 1 / (y : ℝ) = 1 / 15) ∧ (x + y = 64) := 
by
  sorry

end smallest_x_y_sum_l237_237713


namespace intersection_points_in_plane_l237_237886

-- Define the cones with parallel axes and equal angles
def cone1 (a1 b1 c1 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a1)^2 + (y - b1)^2 = k^2 * (z - c1)^2

def cone2 (a2 b2 c2 k : ℝ) (x y z : ℝ) : Prop :=
  (x - a2)^2 + (y - b2)^2 = k^2 * (z - c2)^2

-- Given conditions
variable (a1 b1 c1 a2 b2 c2 k : ℝ)

-- The theorem to be proven
theorem intersection_points_in_plane (x y z : ℝ) 
  (h1 : cone1 a1 b1 c1 k x y z) (h2 : cone2 a2 b2 c2 k x y z) : 
  ∃ (A B C D : ℝ), A * x + B * y + C * z + D = 0 :=
by
  sorry

end intersection_points_in_plane_l237_237886


namespace Denise_age_l237_237759

-- Define the ages of Amanda, Carlos, Beth, and Denise
variables (A C B D : ℕ)

-- State the given conditions
def condition1 := A = C - 4
def condition2 := C = B + 5
def condition3 := D = B + 2
def condition4 := A = 16

-- The theorem to prove
theorem Denise_age (A C B D : ℕ) (h1 : condition1 A C) (h2 : condition2 C B) (h3 : condition3 D B) (h4 : condition4 A) : D = 17 :=
by
  sorry

end Denise_age_l237_237759


namespace rope_lengths_l237_237799

theorem rope_lengths (joey_len chad_len mandy_len : ℝ) (h1 : joey_len = 56) 
  (h2 : 8 / 3 = joey_len / chad_len) (h3 : 5 / 2 = chad_len / mandy_len) : 
  chad_len = 21 ∧ mandy_len = 8.4 :=
by
  sorry

end rope_lengths_l237_237799


namespace find_quotient_l237_237234

def dividend : ℕ := 55053
def divisor : ℕ := 456
def remainder : ℕ := 333

theorem find_quotient (Q : ℕ) (h : dividend = (divisor * Q) + remainder) : Q = 120 := by
  sorry

end find_quotient_l237_237234


namespace find_minimum_value_l237_237959

theorem find_minimum_value (c : ℝ) : 
  (∀ c : ℝ, (c = -12) ↔ (∀ d : ℝ, (1 / 3) * d^2 + 8 * d - 7 ≥ (1 / 3) * (-12)^2 + 8 * (-12) - 7)) :=
sorry

end find_minimum_value_l237_237959


namespace find_number_l237_237696

theorem find_number (x : ℝ) (h : 0.9 * x = 0.0063) : x = 0.007 := 
by {
  sorry
}

end find_number_l237_237696


namespace digit_B_condition_l237_237102

theorem digit_B_condition {B : ℕ} (h10 : ∃ d : ℕ, 58709310 = 10 * d)
  (h5 : ∃ e : ℕ, 58709310 = 5 * e)
  (h6 : ∃ f : ℕ, 58709310 = 6 * f)
  (h4 : ∃ g : ℕ, 58709310 = 4 * g)
  (h3 : ∃ h : ℕ, 58709310 = 3 * h)
  (h2 : ∃ i : ℕ, 58709310 = 2 * i) :
  B = 0 := by
  sorry

end digit_B_condition_l237_237102


namespace servings_left_proof_l237_237708

-- Define the number of servings prepared
def total_servings : ℕ := 61

-- Define the number of guests
def total_guests : ℕ := 8

-- Define the fraction of servings the first 3 guests shared
def first_three_fraction : ℚ := 2 / 5

-- Define the fraction of servings the next 4 guests shared
def next_four_fraction : ℚ := 1 / 4

-- Define the number of servings consumed by the 8th guest
def eighth_guest_servings : ℕ := 5

-- Total consumed servings by the first three guests (rounded down)
def first_three_consumed := (first_three_fraction * total_servings).floor

-- Total consumed servings by the next four guests (rounded down)
def next_four_consumed := (next_four_fraction * total_servings).floor

-- Total consumed servings in total
def total_consumed := first_three_consumed + next_four_consumed + eighth_guest_servings

-- The number of servings left unconsumed
def servings_left_unconsumed := total_servings - total_consumed

-- The theorem stating there are 17 servings left unconsumed
theorem servings_left_proof : servings_left_unconsumed = 17 := by
  sorry

end servings_left_proof_l237_237708


namespace exists_triangle_cut_into_2005_congruent_l237_237542

theorem exists_triangle_cut_into_2005_congruent :
  ∃ (Δ : Type) (a b c : Δ → ℝ )
  (h₁ : a^2 + b^2 = c^2) (h₂ : a * b / 2 = 2005 / 2),
  true :=
sorry

end exists_triangle_cut_into_2005_congruent_l237_237542


namespace value_of_expression_l237_237397

theorem value_of_expression (m n : ℝ) (h : m + n = 4) : 2 * m^2 + 4 * m * n + 2 * n^2 - 5 = 27 :=
  sorry

end value_of_expression_l237_237397


namespace min_rectangle_area_l237_237291

theorem min_rectangle_area : 
  ∃ (x y : ℕ), 2 * (x + y) = 80 ∧ x * y = 39 :=
by
  sorry

end min_rectangle_area_l237_237291


namespace voting_proposal_l237_237465

theorem voting_proposal :
  ∀ (T Votes_against Votes_in_favor More_votes_in_favor : ℕ),
    T = 290 →
    Votes_against = (40 * T) / 100 →
    Votes_in_favor = T - Votes_against →
    More_votes_in_favor = Votes_in_favor - Votes_against →
    More_votes_in_favor = 58 :=
by sorry

end voting_proposal_l237_237465


namespace cost_per_foot_of_fence_l237_237185

theorem cost_per_foot_of_fence 
  (area : ℝ) 
  (total_cost : ℝ) 
  (h_area : area = 289) 
  (h_total_cost : total_cost = 4080) 
  : total_cost / (4 * (Real.sqrt area)) = 60 := 
by
  sorry

end cost_per_foot_of_fence_l237_237185


namespace find_digit_l237_237661

theorem find_digit:
  ∃ d: ℕ, d < 1000 ∧ 1995 * d = 610470 :=
  sorry

end find_digit_l237_237661


namespace abs_m_minus_n_l237_237649

theorem abs_m_minus_n (m n : ℝ) (h_avg : (m + n + 9 + 8 + 10) / 5 = 9) (h_var : (1 / 5 * (m^2 + n^2 + 81 + 64 + 100) - 81) = 2) : |m - n| = 4 :=
  sorry

end abs_m_minus_n_l237_237649


namespace symmetric_point_l237_237880

theorem symmetric_point (a b : ℝ) (h1 : a = 2) (h2 : 3 = -b) : (a + b) ^ 2023 = -1 := 
by
  sorry

end symmetric_point_l237_237880


namespace delta_k_f_l237_237709

open Nat

-- Define the function
def f (n : ℕ) : ℕ := 3^n

-- Define the discrete difference operator
def Δ (g : ℕ → ℕ) (n : ℕ) : ℕ := g (n + 1) - g n

-- Define the k-th discrete difference
def Δk (g : ℕ → ℕ) (k : ℕ) (n : ℕ) : ℕ :=
  if k = 0 then g n else Δk (Δ g) (k - 1) n

-- State the theorem
theorem delta_k_f (k : ℕ) (n : ℕ) (h : k ≥ 1) : Δk f k n = 2^k * 3^n := by
  sorry

end delta_k_f_l237_237709


namespace minimize_sum_of_squares_l237_237463

theorem minimize_sum_of_squares (a b c : ℕ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) (h₄ : a + b + c = 16) :
  a^2 + b^2 + c^2 ≥ 86 :=
sorry

end minimize_sum_of_squares_l237_237463


namespace initial_population_is_3162_l237_237715

noncomputable def initial_population (P : ℕ) : Prop :=
  let after_bombardment := 0.95 * (P : ℝ)
  let after_fear := 0.85 * after_bombardment
  after_fear = 2553

theorem initial_population_is_3162 : initial_population 3162 :=
  by
    -- By our condition setup, we need to prove:
    -- let after_bombardment := 0.95 * 3162
    -- let after_fear := 0.85 * after_bombardment
    -- after_fear = 2553

    -- This can be directly stated and verified through concrete calculations as in the problem steps.
    sorry

end initial_population_is_3162_l237_237715


namespace pattern_proof_l237_237329

theorem pattern_proof (h1 : 1 = 6) (h2 : 2 = 36) (h3 : 3 = 363) (h4 : 4 = 364) (h5 : 5 = 365) : 36 = 3636 := by
  sorry

end pattern_proof_l237_237329


namespace remainder_when_dividing_928927_by_6_l237_237885

theorem remainder_when_dividing_928927_by_6 :
  928927 % 6 = 1 :=
by
  sorry

end remainder_when_dividing_928927_by_6_l237_237885


namespace tile_covering_possible_l237_237876

theorem tile_covering_possible (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  ((m % 6 = 0) ∨ (n % 6 = 0)) := 
sorry

end tile_covering_possible_l237_237876


namespace contrapositive_example_l237_237786

theorem contrapositive_example (α : ℝ) : (α = Real.pi / 3 → Real.cos α = 1 / 2) → (Real.cos α ≠ 1 / 2 → α ≠ Real.pi / 3) :=
by
  sorry

end contrapositive_example_l237_237786


namespace desks_per_row_calc_l237_237266

theorem desks_per_row_calc :
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  (total_desks / 4 = 6) :=
by
  let restroom_students := 2
  let absent_students := 3 * restroom_students - 1
  let total_students := 23
  let classroom_students := total_students - restroom_students - absent_students
  let total_desks := classroom_students * 3 / 2
  show total_desks / 4 = 6
  sorry

end desks_per_row_calc_l237_237266


namespace sector_area_is_4_l237_237845

/-- Given a sector of a circle with perimeter 8 and central angle 2 radians,
    the area of the sector is 4. -/
theorem sector_area_is_4 (r l : ℝ) (h1 : l + 2 * r = 8) (h2 : l / r = 2) : 
    (1 / 2) * l * r = 4 :=
sorry

end sector_area_is_4_l237_237845


namespace appropriate_chart_for_milk_powder_l237_237492

-- Define the chart requirements and the correctness condition
def ChartType := String
def pie : ChartType := "pie"
def line : ChartType := "line"
def bar : ChartType := "bar"

-- The condition we need for our proof
def representsPercentagesWell (chart: ChartType) : Prop :=
  chart = pie

-- The main theorem statement
theorem appropriate_chart_for_milk_powder : representsPercentagesWell pie :=
by
  sorry

end appropriate_chart_for_milk_powder_l237_237492


namespace coins_left_l237_237456

-- Define the initial number of coins from each source
def piggy_bank_coins : ℕ := 15
def brother_coins : ℕ := 13
def father_coins : ℕ := 8

-- Define the number of coins given to Laura
def given_to_laura_coins : ℕ := 21

-- Define the total initial coins collected by Kylie
def total_initial_coins : ℕ := piggy_bank_coins + brother_coins + father_coins

-- Lean statement to prove
theorem coins_left : total_initial_coins - given_to_laura_coins = 15 :=
by
  sorry

end coins_left_l237_237456


namespace calculate_expression_l237_237176

theorem calculate_expression :
  ((650^2 - 350^2) * 3 = 900000) := by
  sorry

end calculate_expression_l237_237176


namespace bottles_from_shop_c_correct_l237_237335

-- Definitions for the given conditions
def total_bottles := 550
def bottles_from_shop_a := 150
def bottles_from_shop_b := 180

-- Definition for the bottles from Shop C
def bottles_from_shop_c := total_bottles - (bottles_from_shop_a + bottles_from_shop_b)

-- The statement to prove
theorem bottles_from_shop_c_correct : bottles_from_shop_c = 220 :=
by
  -- proof will be filled later
  sorry

end bottles_from_shop_c_correct_l237_237335


namespace max_val_a_l237_237887

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x + a * (x^2 - 3 * x + 2)

theorem max_val_a (a : ℝ) (h1 : a > 0) (h2 : ∀ x > 1, f a x ≥ 0) : a ≤ 1 := sorry

end max_val_a_l237_237887


namespace largest_sum_of_three_largest_angles_l237_237635

-- Definitions and main theorem statement
theorem largest_sum_of_three_largest_angles (EFGH : Type*)
    (a b c d : ℝ) 
    (h1 : a + b + c + d = 360)
    (h2 : b = 3 * c)
    (h3 : ∃ (common_diff : ℝ), (c - a = common_diff) ∧ (b - c = common_diff) ∧ (d - b = common_diff))
    (h4 : ∀ (x y z : ℝ), (x = y + z) ↔ (∃ (progression_diff : ℝ), x - y = y - z ∧ y - z = z - x)) :
    (∃ (A B C D : ℝ), A = a ∧ B = b ∧ C = c ∧ D = d ∧ A + B + C + D = 360 ∧ A = max a (max b (max c d)) ∧ B = 2 * D ∧ A + B + C = 330) :=
sorry

end largest_sum_of_three_largest_angles_l237_237635


namespace sum_of_roots_l237_237951

-- Define the polynomial equation
def poly (x : ℝ) := (3 * x + 4) * (x - 5) + (3 * x + 4) * (x - 7)

-- The theorem claiming the sum of the roots
theorem sum_of_roots : 
  (∀ x : ℝ, poly x = 0 → (x = -4/3 ∨ x = 6)) → 
  (∀ s : ℝ, s = -4 / 3 + 6) → s = 14 / 3 :=
by
  sorry

end sum_of_roots_l237_237951


namespace pythagorean_theorem_l237_237171

theorem pythagorean_theorem {a b c p q : ℝ} 
  (h₁ : p * c = a ^ 2) 
  (h₂ : q * c = b ^ 2)
  (h₃ : p + q = c) : 
  c ^ 2 = a ^ 2 + b ^ 2 := 
by 
  sorry

end pythagorean_theorem_l237_237171


namespace number_of_valid_pairs_l237_237272

theorem number_of_valid_pairs (a b : ℝ) :
  (∃ x y : ℤ, a * (x : ℝ) + b * (y : ℝ) = 2 ∧ (x : ℝ)^2 + (y : ℝ)^2 = 65) →
  ∃! pairs_count : ℕ, pairs_count = 72 :=
by
  sorry

end number_of_valid_pairs_l237_237272


namespace num_pos_divisors_36_l237_237780

theorem num_pos_divisors_36 : ∃ (n : ℕ), n = 9 ∧ (∀ d : ℕ, d > 0 → d ∣ 36 → d ∣ 9) :=
by
  sorry

end num_pos_divisors_36_l237_237780


namespace L_shaped_region_area_l237_237862

-- Define the conditions
def square_area (side_length : ℕ) : ℕ := side_length * side_length

def WXYZ_side_length : ℕ := 6
def XUVW_side_length : ℕ := 2
def TYXZ_side_length : ℕ := 3

-- Define the areas of the squares
def WXYZ_area : ℕ := square_area WXYZ_side_length
def XUVW_area : ℕ := square_area XUVW_side_length
def TYXZ_area : ℕ := square_area TYXZ_side_length

-- Lean statement to prove the area of the L-shaped region
theorem L_shaped_region_area : WXYZ_area - XUVW_area - TYXZ_area = 23 := by
  sorry

end L_shaped_region_area_l237_237862


namespace ed_more_marbles_l237_237301

-- Define variables for initial number of marbles
variables {E D : ℕ}

-- Ed had some more marbles than Doug initially.
-- Doug lost 8 of his marbles at the playground.
-- Now Ed has 30 more marbles than Doug.
theorem ed_more_marbles (h : E = (D - 8) + 30) : E - D = 22 :=
by
  sorry

end ed_more_marbles_l237_237301


namespace initial_riding_time_l237_237000

theorem initial_riding_time (t : ℝ) (h1 : t * 60 + 90 + 30 + 120 = 270) : t * 60 = 30 :=
by sorry

end initial_riding_time_l237_237000


namespace roots_cubic_sum_l237_237890

theorem roots_cubic_sum :
  (∃ x1 x2 x3 x4 : ℂ, (x1^4 + 5*x1^3 + 6*x1^2 + 5*x1 + 1 = 0) ∧
                       (x2^4 + 5*x2^3 + 6*x2^2 + 5*x2 + 1 = 0) ∧
                       (x3^4 + 5*x3^3 + 6*x3^2 + 5*x3 + 1 = 0) ∧
                       (x4^4 + 5*x4^3 + 6*x4^2 + 5*x4 + 1 = 0)) →
  (x1^3 + x2^3 + x3^3 + x4^3 = -54) :=
sorry

end roots_cubic_sum_l237_237890


namespace david_marks_in_english_l237_237077

theorem david_marks_in_english 
  (math : ℤ) (phys : ℤ) (chem : ℤ) (bio : ℤ) (avg : ℤ) 
  (marks_per_math : math = 85) 
  (marks_per_phys : phys = 92) 
  (marks_per_chem : chem = 87) 
  (marks_per_bio : bio = 95) 
  (avg_marks : avg = 89) 
  (num_subjects : ℤ := 5) :
  ∃ (eng : ℤ), eng + 85 + 92 + 87 + 95 = 89 * 5 ∧ eng = 86 :=
by
  sorry

end david_marks_in_english_l237_237077


namespace part1_part2_part3_l237_237784

-- Define conditions
variables (n : ℕ) (h₁ : 5 ≤ n)

-- Problem part (1): Define p_n and prove its value
def p_n (n : ℕ) := (10 * n) / ((n + 5) * (n + 4))

-- Problem part (2): Define EX and prove its value for n = 5
def EX : ℚ := 5 / 3

-- Problem part (3): Prove n = 20 maximizes P
def P (n : ℕ) := 3 * ((p_n n) ^ 3 - 2 * (p_n n) ^ 2 + (p_n n))
def n_max := 20

-- Making the proof skeletons for clarity, filling in later
theorem part1 : p_n n = 10 * n / ((n + 5) * (n + 4)) :=
sorry

theorem part2 (h₂ : n = 5) : EX = 5 / 3 :=
sorry

theorem part3 : n_max = 20 :=
sorry

end part1_part2_part3_l237_237784


namespace abc_sum_l237_237941

theorem abc_sum (f : ℝ → ℝ) (a b c : ℝ) :
  f (x - 2) = 2 * x^2 - 5 * x + 3 → f x = a * x^2 + b * x + c → a + b + c = 6 :=
by
  intros h₁ h₂
  sorry

end abc_sum_l237_237941


namespace range_of_x_l237_237695

theorem range_of_x (x : ℝ) : (∃ y : ℝ, y = 1 / (Real.sqrt (x - 2))) ↔ x > 2 :=
by
  sorry

end range_of_x_l237_237695


namespace range_of_a_l237_237741

theorem range_of_a (a : ℝ) : 
  (∀ P Q : ℝ × ℝ, P ≠ Q ∧ P.snd = a * P.fst ^ 2 - 1 ∧ Q.snd = a * Q.fst ^ 2 - 1 ∧ 
  P.fst + P.snd = -(Q.fst + Q.snd)) →
  a > 3 / 4 :=
by
  sorry

end range_of_a_l237_237741


namespace find_decimal_decrease_l237_237299

noncomputable def tax_diminished_percentage (T C : ℝ) (X : ℝ) : Prop :=
  let new_tax := T * (1 - X / 100)
  let new_consumption := C * 1.15
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  new_revenue = original_revenue * 0.943

theorem find_decimal_decrease (T C : ℝ) (X : ℝ) :
  tax_diminished_percentage T C X → X = 18 := sorry

end find_decimal_decrease_l237_237299


namespace expression_value_l237_237652

theorem expression_value (x y : ℝ) (h : x - 2 * y = 3) : 1 - 2 * x + 4 * y = -5 :=
by
  sorry

end expression_value_l237_237652


namespace intersection_M_N_l237_237016

open Set

def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def N := {x : ℝ | x > 1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 1 < x ∧ x ≤ 2} :=
by
  sorry

end intersection_M_N_l237_237016


namespace region_area_proof_l237_237681

noncomputable def region_area := 
  let region := {p : ℝ × ℝ | abs (p.1 - p.2^2 / 2) + p.1 + p.2^2 / 2 ≤ 2 - p.2}
  2 * (0.5 * (3 * (2 + 0.5)))

theorem region_area_proof : region_area = 15 / 2 :=
by
  sorry

end region_area_proof_l237_237681


namespace sufficient_but_not_necessary_to_increasing_l237_237066

theorem sufficient_but_not_necessary_to_increasing (a : ℝ) :
  (∀ x y : ℝ, 1 ≤ x → x ≤ y → (x^2 - 2*a*x) ≤ (y^2 - 2*a*y)) ↔ (a ≤ 1) := sorry

end sufficient_but_not_necessary_to_increasing_l237_237066


namespace imaginary_part_of_f_i_div_i_is_one_l237_237139

def f (x : ℂ) : ℂ := x^3 - 1

theorem imaginary_part_of_f_i_div_i_is_one 
    (i : ℂ) (h : i^2 = -1) :
    ( (f i) / i ).im = 1 := 
sorry

end imaginary_part_of_f_i_div_i_is_one_l237_237139


namespace difference_divisible_by_10_l237_237331

theorem difference_divisible_by_10 : (43 ^ 43 - 17 ^ 17) % 10 = 0 := by
  sorry

end difference_divisible_by_10_l237_237331


namespace function_positive_for_x_gt_neg1_l237_237133

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (3*x^2 + 6*x + 9)

theorem function_positive_for_x_gt_neg1 : ∀ (x : ℝ), x > -1 → f x > 0.5 :=
by
  sorry

end function_positive_for_x_gt_neg1_l237_237133


namespace sum_of_circle_areas_l237_237305

theorem sum_of_circle_areas (a b c: ℝ)
  (h1: a + b = 6)
  (h2: b + c = 8)
  (h3: a + c = 10) :
  π * a^2 + π * b^2 + π * c^2 = 56 * π := 
by
  sorry

end sum_of_circle_areas_l237_237305


namespace no_perfect_square_l237_237996

theorem no_perfect_square (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) 
  (h : ∃ (a : ℕ), p + q^2 = a^2) : ∀ (n : ℕ), n > 0 → ¬ (∃ (b : ℕ), p^2 + q^n = b^2) := 
by
  sorry

end no_perfect_square_l237_237996


namespace algebraic_expression_value_l237_237670

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y^2 = 1) : -2 * x + 4 * y^2 + 1 = -1 :=
by
  sorry

end algebraic_expression_value_l237_237670


namespace min_value_of_mn_squared_l237_237737

theorem min_value_of_mn_squared 
  (a b c : ℝ) 
  (h : a^2 + b^2 = c^2) 
  (m n : ℝ) 
  (h_point : a * m + b * n + 2 * c = 0) : 
  m^2 + n^2 = 4 :=
sorry

end min_value_of_mn_squared_l237_237737


namespace eccentricity_hyperbola_l237_237054

theorem eccentricity_hyperbola : 
  let a2 := 4
  let b2 := 5
  let e := Real.sqrt (1 + (b2 / a2))
  e = 3 / 2 := by
    apply sorry

end eccentricity_hyperbola_l237_237054


namespace simplify_expression_l237_237834

theorem simplify_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 9^(3/2)) = -7 :=
by sorry

end simplify_expression_l237_237834


namespace calculate_series_l237_237884

theorem calculate_series : 20^2 - 18^2 + 16^2 - 14^2 + 12^2 - 10^2 + 8^2 - 6^2 + 4^2 - 2^2 = 200 := 
by
  sorry

end calculate_series_l237_237884


namespace passing_probability_l237_237721

theorem passing_probability :
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  probability = 44 / 45 :=
by
  let num_students := 6
  let probability :=
    1 - (2/6) * (2/5) * (2/4) * (2/3) * (2/2)
  have p_eq : probability = 44 / 45 := sorry
  exact p_eq

end passing_probability_l237_237721


namespace compound_interest_eq_440_l237_237408

-- Define the conditions
variables (P R T SI CI : ℝ)
variables (H_SI : SI = P * R * T / 100)
variables (H_R : R = 20)
variables (H_T : T = 2)
variables (H_given : SI = 400)
variables (H_question : CI = P * (1 + R / 100)^T - P)

-- Define the goal to prove
theorem compound_interest_eq_440 : CI = 440 :=
by
  -- Conditions and the result should be proved here, but we'll use sorry to skip the proof step.
  sorry

end compound_interest_eq_440_l237_237408


namespace solution_system_l237_237907

theorem solution_system (x y : ℝ) (h1 : x * y = 8) (h2 : x^2 * y + x * y^2 + x + y = 80) :
  x^2 + y^2 = 5104 / 81 := by
  sorry

end solution_system_l237_237907


namespace bus_stop_time_l237_237691

-- Usual time to walk to the bus stop
def usual_time (T : ℕ) := T

-- Usual speed
def usual_speed (S : ℕ) := S

-- New speed when walking at 4/5 of usual speed
def new_speed (S : ℕ) := (4 * S) / 5

-- Time relationship when walking at new speed
def time_relationship (T : ℕ) (S : ℕ) := (S / ((4 * S) / 5)) = (T + 10) / T

-- Prove the usual time T is 40 minutes
theorem bus_stop_time (T S : ℕ) (h1 : time_relationship T S) : T = 40 :=
by
  sorry

end bus_stop_time_l237_237691


namespace eval_expression_l237_237748

theorem eval_expression :
  16^3 + 3 * (16^2) * 2 + 3 * 16 * (2^2) + 2^3 = 5832 :=
by
  sorry

end eval_expression_l237_237748


namespace total_sections_l237_237687

theorem total_sections (boys girls gcd sections_boys sections_girls : ℕ) 
  (h_boys : boys = 408) 
  (h_girls : girls = 264) 
  (h_gcd: gcd = Nat.gcd boys girls)
  (h_sections_boys : sections_boys = boys / gcd)
  (h_sections_girls : sections_girls = girls / gcd)
  (h_total_sections : sections_boys + sections_girls = 28)
: sections_boys + sections_girls = 28 := by
  sorry

end total_sections_l237_237687


namespace intersection_equals_l237_237915

def A : Set ℝ := {x | x < 1}

def B : Set ℝ := {x | x^2 + x ≤ 6}

theorem intersection_equals : A ∩ B = {x : ℝ | -3 ≤ x ∧ x < 1} :=
by
  sorry

end intersection_equals_l237_237915


namespace pentagon_area_l237_237483

theorem pentagon_area (a b c d e : ℝ)
  (ht_base ht_height : ℝ)
  (trap_base1 trap_base2 trap_height : ℝ)
  (side_a : a = 17)
  (side_b : b = 22)
  (side_c : c = 30)
  (side_d : d = 26)
  (side_e : e = 22)
  (rt_height : ht_height = 17)
  (rt_base : ht_base = 22)
  (trap_base1_eq : trap_base1 = 26)
  (trap_base2_eq : trap_base2 = 30)
  (trap_height_eq : trap_height = 22)
  : 1/2 * ht_base * ht_height + 1/2 * (trap_base1 + trap_base2) * trap_height = 803 :=
by sorry

end pentagon_area_l237_237483


namespace percentage_of_work_day_in_meetings_is_25_l237_237412

-- Define the conditions
def workDayHours : ℕ := 9
def firstMeetingMinutes : ℕ := 45
def secondMeetingMinutes : ℕ := 2 * firstMeetingMinutes
def totalMeetingMinutes : ℕ := firstMeetingMinutes + secondMeetingMinutes
def workDayMinutes : ℕ := workDayHours * 60

-- Define the percentage calculation
def percentageOfWorkdaySpentInMeetings : ℕ := (totalMeetingMinutes * 100) / workDayMinutes

-- The theorem to be proven
theorem percentage_of_work_day_in_meetings_is_25 :
  percentageOfWorkdaySpentInMeetings = 25 :=
sorry

end percentage_of_work_day_in_meetings_is_25_l237_237412


namespace total_meals_per_week_l237_237802

-- Definitions of the conditions
def meals_per_day_r1 : ℕ := 20
def meals_per_day_r2 : ℕ := 40
def meals_per_day_r3 : ℕ := 50
def days_per_week : ℕ := 7

-- The proof goal
theorem total_meals_per_week : 
  (meals_per_day_r1 * days_per_week) + 
  (meals_per_day_r2 * days_per_week) + 
  (meals_per_day_r3 * days_per_week) = 770 :=
by
  sorry

end total_meals_per_week_l237_237802


namespace calc_expression_solve_system_inequalities_l237_237530

-- Proof Problem 1: Calculation
theorem calc_expression : 
  |1 - Real.sqrt 3| - Real.sqrt 2 * Real.sqrt 6 + 1 / (2 - Real.sqrt 3) - (2 / 3) ^ (-2 : ℤ) = -5 / 4 := 
by 
  sorry

-- Proof Problem 2: System of Inequalities Solution
variable (m : ℝ)
variable (x : ℝ)
  
theorem solve_system_inequalities (h : m < 0) : 
  (4 * x - 1 > x - 7) ∧ (-1 / 4 * x < 3 / 2 * m - 1) → x > 4 - 6 * m := 
by 
  sorry

end calc_expression_solve_system_inequalities_l237_237530


namespace tom_bought_6_hardcover_l237_237008

-- Given conditions and statements
def toms_books_condition_1 (h p : ℕ) : Prop :=
  h + p = 10

def toms_books_condition_2 (h p : ℕ) : Prop :=
  28 * h + 18 * p = 240

-- The theorem to prove
theorem tom_bought_6_hardcover (h p : ℕ) 
  (h_condition : toms_books_condition_1 h p)
  (c_condition : toms_books_condition_2 h p) : 
  h = 6 :=
sorry

end tom_bought_6_hardcover_l237_237008


namespace exists_same_color_points_one_meter_apart_l237_237188

-- Declare the colors as an enumeration
inductive Color
| red : Color
| black : Color

-- Define the function that assigns a color to each point in the plane
def color (point : ℝ × ℝ) : Color := sorry

-- The theorem to be proven
theorem exists_same_color_points_one_meter_apart :
  ∃ x y : ℝ × ℝ, x ≠ y ∧ dist x y = 1 ∧ color x = color y :=
sorry

end exists_same_color_points_one_meter_apart_l237_237188


namespace smallest_four_digit_equivalent_6_mod_7_l237_237636

theorem smallest_four_digit_equivalent_6_mod_7 :
  (∃ (n : ℕ), n >= 1000 ∧ n < 10000 ∧ n % 7 = 6 ∧ (∀ (m : ℕ), m >= 1000 ∧ m < 10000 ∧ m % 7 = 6 → m >= n)) ∧ ∃ (n : ℕ), n = 1000 :=
sorry

end smallest_four_digit_equivalent_6_mod_7_l237_237636


namespace price_of_basic_computer_l237_237142

variable (C P : ℝ)

theorem price_of_basic_computer 
    (h1 : C + P = 2500)
    (h2 : P = (1 / 6) * (C + 500 + P)) : 
  C = 2000 :=
by
  sorry

end price_of_basic_computer_l237_237142


namespace rabbits_to_hamsters_l237_237227

theorem rabbits_to_hamsters (rabbits hamsters : ℕ) (h_ratio : 3 * hamsters = 4 * rabbits) (h_rabbits : rabbits = 18) : hamsters = 24 :=
by
  sorry

end rabbits_to_hamsters_l237_237227


namespace anne_clean_house_in_12_hours_l237_237205

theorem anne_clean_house_in_12_hours (B A : ℝ) (h1 : 4 * (B + A) = 1) (h2 : 3 * (B + 2 * A) = 1) : A = 1 / 12 ∧ (1 / A) = 12 :=
by
  -- We will leave the proof as a placeholder
  sorry

end anne_clean_house_in_12_hours_l237_237205


namespace problem_integer_square_l237_237676

theorem problem_integer_square 
  (a b c d A : ℤ) 
  (H1 : a^2 + A = b^2) 
  (H2 : c^2 + A = d^2) : 
  ∃ (k : ℕ), 2 * (a + b) * (c + d) * (a * c + b * d - A) = k^2 :=
by
  sorry

end problem_integer_square_l237_237676


namespace soda_preference_respondents_l237_237508

noncomputable def fraction_of_soda (angle_soda : ℝ) (total_angle : ℝ) : ℝ :=
  angle_soda / total_angle

noncomputable def number_of_soda_preference (total_people : ℕ) (fraction : ℝ) : ℝ :=
  total_people * fraction

theorem soda_preference_respondents (total_people : ℕ) (angle_soda : ℝ) (total_angle : ℝ) : 
  total_people = 520 → angle_soda = 298 → total_angle = 360 → 
  number_of_soda_preference total_people (fraction_of_soda angle_soda total_angle) = 429 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold fraction_of_soda number_of_soda_preference
  -- further calculation steps
  sorry

end soda_preference_respondents_l237_237508


namespace number_of_multiples_of_six_ending_in_four_and_less_than_800_l237_237468

-- Definitions from conditions
def is_multiple_of_six (n : ℕ) : Prop := n % 6 = 0
def ends_with_four (n : ℕ) : Prop := n % 10 = 4
def less_than_800 (n : ℕ) : Prop := n < 800

-- Theorem to prove
theorem number_of_multiples_of_six_ending_in_four_and_less_than_800 :
  ∃ k : ℕ, k = 26 ∧ ∀ n : ℕ, (is_multiple_of_six n ∧ ends_with_four n ∧ less_than_800 n) → n = 24 + 60 * k ∨ n = 54 + 60 * k :=
sorry

end number_of_multiples_of_six_ending_in_four_and_less_than_800_l237_237468


namespace geometric_sequence_ratio_l237_237296

theorem geometric_sequence_ratio (a b c q : ℝ) :
  (∃ x : ℝ, x ≠ 0 ∧ b + c - a = x * q ∧ c + a - b = x * q^2 ∧ a + b - c = x * q^3 ∧ a + b + c = x) →
  q^3 + q^2 + q = 1 :=
by
  sorry

end geometric_sequence_ratio_l237_237296


namespace find_b_if_even_function_l237_237087

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem find_b_if_even_function (h : ∀ x : ℝ, f (-x) = f (x)) : b = 0 := by
  sorry

end find_b_if_even_function_l237_237087


namespace polar_to_cartesian_l237_237688

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * Real.cos θ) :
  ∃ x y : ℝ, (x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ ∧
  (x - 2)^2 + y^2 = 4) :=
sorry

end polar_to_cartesian_l237_237688


namespace number_of_correct_statements_l237_237699

def is_opposite (a b : ℤ) : Prop := a + b = 0

def statement1 : Prop := ∀ a b : ℤ, (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0) → is_opposite a b
def statement2 : Prop := ∀ n : ℤ, n = -n → n < 0
def statement3 : Prop := ∀ a b : ℤ, is_opposite a b → a + b = 0
def statement4 : Prop := ∀ a b : ℤ, is_opposite a b → (a > 0 ∧ b < 0 ∨ a < 0 ∧ b > 0)

theorem number_of_correct_statements : (¬statement1 ∧ ¬statement2 ∧ statement3 ∧ ¬statement4) ↔ (∃n : ℕ, n = 1) :=
by
  sorry

end number_of_correct_statements_l237_237699


namespace merchant_articles_l237_237697

theorem merchant_articles (N CP SP : ℝ) 
  (h1 : N * CP = 16 * SP)
  (h2 : SP = CP * 1.375) : 
  N = 22 :=
by
  sorry

end merchant_articles_l237_237697


namespace digits_base8_sum_l237_237349

open Nat

theorem digits_base8_sum (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z) 
  (h_distinct : X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z) (h_base8 : X < 8 ∧ Y < 8 ∧ Z < 8) 
  (h_eq : (8^2 * X + 8 * Y + Z) + (8^2 * Y + 8 * Z + X) + (8^2 * Z + 8 * X + Y) = 8^3 * X + 8^2 * X + 8 * X) : 
  Y + Z = 7 :=
by
  sorry

end digits_base8_sum_l237_237349


namespace sum_first_and_third_angle_l237_237287

-- Define the conditions
variable (A : ℕ)
axiom C1 : A + 2 * A + (A - 40) = 180

-- State the theorem to be proven
theorem sum_first_and_third_angle : A + (A - 40) = 70 :=
by
  sorry

end sum_first_and_third_angle_l237_237287


namespace length_BC_l237_237789

noncomputable def center (O : Type) : Prop := sorry   -- Center of the circle.

noncomputable def diameter (AD : Type) : Prop := sorry   -- AD is a diameter.

noncomputable def chord (ABC : Type) : Prop := sorry   -- ABC is a chord.

noncomputable def radius_equal (BO : ℝ) : Prop := BO = 8   -- BO = 8.

noncomputable def angle_ABO (α : ℝ) : Prop := α = 45   -- ∠ABO = 45°.

noncomputable def arc_CD (β : ℝ) : Prop := β = 90   -- Arc CD subtended by ∠AOD = 90°.

theorem length_BC (O AD ABC : Type) (BO : ℝ) (α β γ : ℝ)
  (h1 : center O)
  (h2 : diameter AD)
  (h3 : chord ABC)
  (h4 : radius_equal BO)
  (h5 : angle_ABO α)
  (h6 : arc_CD β)
  : γ = 8 := 
sorry

end length_BC_l237_237789


namespace cos_arcsin_l237_237578

theorem cos_arcsin (x : ℝ) (hx : x = 3 / 5) : Real.cos (Real.arcsin x) = 4 / 5 := by
  sorry

end cos_arcsin_l237_237578


namespace find_common_difference_find_minimum_sum_minimum_sum_value_l237_237281

-- Defining the arithmetic sequence and its properties
def a (n : ℕ) (d : ℚ) := (-3 : ℚ) + n * d

-- Given conditions
def condition_1 : ℚ := -3
def condition_2 (d : ℚ) := 11 * a 4 d = 5 * a 7 d - 13
def common_difference : ℚ := 31 / 9

-- Sum of the first n terms of an arithmetic sequence
def S (n : ℕ) (d : ℚ) := n * (-3 + (n - 1) * d / 2)

-- Defining the necessary theorems
theorem find_common_difference (d : ℚ) : condition_2 d → d = common_difference := by
  sorry

theorem find_minimum_sum (n : ℕ) : S n common_difference ≥ S 2 common_difference := by
  sorry

theorem minimum_sum_value : S 2 common_difference = -23 / 9 := by
  sorry

end find_common_difference_find_minimum_sum_minimum_sum_value_l237_237281


namespace find_k_l237_237011

noncomputable def g (a b c : ℤ) (x : ℤ) := a * x^2 + b * x + c

theorem find_k (a b c k : ℤ) 
  (h1 : g a b c (-1) = 0) 
  (h2 : 30 < g a b c 5) (h3 : g a b c 5 < 40)
  (h4 : 120 < g a b c 7) (h5 : g a b c 7 < 130)
  (h6 : 2000 * k < g a b c 50) (h7 : g a b c 50 < 2000 * (k + 1)) : 
  k = 5 := 
sorry

end find_k_l237_237011


namespace Taehyung_age_l237_237284

variable (T U : Nat)

-- Condition 1: Taehyung is 17 years younger than his uncle
def condition1 : Prop := U = T + 17

-- Condition 2: Four years later, the sum of their ages is 43
def condition2 : Prop := (T + 4) + (U + 4) = 43

-- The goal is to prove that Taehyung's current age is 9, given the conditions above
theorem Taehyung_age : condition1 T U ∧ condition2 T U → T = 9 := by
  sorry

end Taehyung_age_l237_237284


namespace evaluate_expression_l237_237480

theorem evaluate_expression (x c : ℕ) (h1 : x = 3) (h2 : c = 2) : 
  ((x^2 + c)^2 - (x^2 - c)^2) = 72 :=
by
  rw [h1, h2]
  sorry

end evaluate_expression_l237_237480


namespace david_marks_in_english_l237_237063

theorem david_marks_in_english : 
  ∀ (E : ℕ), 
  let math_marks := 85 
  let physics_marks := 82 
  let chemistry_marks := 87 
  let biology_marks := 85 
  let avg_marks := 85 
  let total_subjects := 5 
  let total_marks := avg_marks * total_subjects 
  let total_known_subject_marks := math_marks + physics_marks + chemistry_marks + biology_marks 
  total_marks = total_known_subject_marks + E → 
  E = 86 :=
by 
  intros
  sorry

end david_marks_in_english_l237_237063


namespace length_ab_l237_237037

section geometry

variables {A B C D E : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

-- Define the lengths and perimeters as needed
variables (AB AC BC CD DE CE : ℝ)

-- Isosceles Triangle properties
axiom isosceles_abc : AC = BC
axiom isosceles_cde : CD = DE

-- Conditons given in the problem
axiom perimeter_cde : CE + CD + DE = 22
axiom perimeter_abc : AB + BC + AC = 24
axiom length_ce : CE = 8

-- Goal: To prove the length of AB
theorem length_ab : AB = 10 :=
by 
  sorry

end geometry

end length_ab_l237_237037


namespace pattern_equation_l237_237824

theorem pattern_equation (n : ℕ) (h : n ≥ 1) : 
  (Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2))) :=
by
  sorry

end pattern_equation_l237_237824


namespace probability_of_suitcase_at_60th_position_expected_waiting_time_l237_237778

/-- Part (a):
    Prove that the probability that the businesspeople's 10th suitcase 
    appears exactly at the 60th position is equal to 
    (binom 59 9) / (binom 200 10) given 200 suitcases and 10 business people's suitcases,
    and a suitcase placed on the belt every 2 seconds. -/
theorem probability_of_suitcase_at_60th_position : 
  ∃ (P : ℚ), P = (Nat.choose 59 9 : ℚ) / (Nat.choose 200 10 : ℚ) :=
sorry

/-- Part (b):
    Prove that the expected waiting time for the businesspeople to get 
    their last suitcase is equal to 4020 / 11 seconds given 200 suitcases and 
    10 business people's suitcases, and a suitcase placed on the belt 
    every 2 seconds. -/
theorem expected_waiting_time : 
  ∃ (E : ℚ), E = 4020 / 11 :=
sorry

end probability_of_suitcase_at_60th_position_expected_waiting_time_l237_237778


namespace side_length_of_square_l237_237604

theorem side_length_of_square (d s : ℝ) (h1: d = 2 * Real.sqrt 2) (h2: d = s * Real.sqrt 2) : s = 2 :=
by
  sorry

end side_length_of_square_l237_237604


namespace system_solution_l237_237772

theorem system_solution (x : Fin 1995 → ℤ) :
  (∀ i : (Fin 1995),
    x (i + 1) ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) →
  (∀ n : (Fin 1995),
    (x n = 0 ∧ x (n + 1) = 1 ∧ x (n + 2) = -1) ∨
    (x n = 0 ∧ x (n + 1) = -1 ∧ x (n + 2) = 1)) :=
by sorry

end system_solution_l237_237772


namespace line_does_not_pass_through_third_quadrant_l237_237032

variable {a b c : ℝ}

theorem line_does_not_pass_through_third_quadrant
  (hac : a * c < 0) (hbc : b * c < 0) : ¬ ∃ x y, x < 0 ∧ y < 0 ∧ a * x + b * y + c = 0 :=
sorry

end line_does_not_pass_through_third_quadrant_l237_237032


namespace find_interest_rate_l237_237857

noncomputable def interest_rate_solution : ℝ :=
  let P := 800
  let A := 1760
  let t := 4
  let n := 1
  (A / P) ^ (1 / (n * t)) - 1

theorem find_interest_rate : interest_rate_solution = 0.1892 := 
by
  sorry

end find_interest_rate_l237_237857


namespace words_on_each_page_l237_237341

theorem words_on_each_page (p : ℕ) (h : 150 * p ≡ 198 [MOD 221]) : p = 93 :=
sorry

end words_on_each_page_l237_237341


namespace inequality_solution_l237_237007

open Set

theorem inequality_solution (x : ℝ) : (1 - 7 / (2 * x - 1) < 0) ↔ (1 / 2 < x ∧ x < 4) := 
by
  sorry

end inequality_solution_l237_237007


namespace average_speed_with_stoppages_l237_237381

theorem average_speed_with_stoppages
    (D : ℝ) -- distance the train travels
    (T_no_stop : ℝ := D / 250) -- time taken to cover the distance without stoppages
    (T_with_stop : ℝ := 2 * T_no_stop) -- total time with stoppages
    : (D / T_with_stop) = 125 := 
by sorry

end average_speed_with_stoppages_l237_237381


namespace problem_statement_l237_237156

def f (x : ℝ) : ℝ := x^2 - 3 * x + 4
def g (x : ℝ) : ℝ := x - 2

theorem problem_statement : f (g 5) - g (f 5) = -8 := by sorry

end problem_statement_l237_237156


namespace no_valid_a_exists_l237_237497

theorem no_valid_a_exists 
  (a : ℝ)
  (h1: ∀ x : ℝ, x^2 + 2*(a+1)*x - (a-1) = 0 → (1 < x ∨ x < 1)) :
  false := by
  sorry

end no_valid_a_exists_l237_237497


namespace inequality_sin_values_l237_237920

theorem inequality_sin_values :
  let a := Real.sin (-5)
  let b := Real.sin 3
  let c := Real.sin 5
  a > b ∧ b > c :=
by
  sorry

end inequality_sin_values_l237_237920


namespace sum_of_2x2_table_is_zero_l237_237308

theorem sum_of_2x2_table_is_zero {a b c d : ℤ} 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_eq : a + b = c + d)
  (prod_eq : a * c = b * d) :
  a + b + c + d = 0 :=
by sorry

end sum_of_2x2_table_is_zero_l237_237308


namespace invested_sum_l237_237926

theorem invested_sum (P r : ℝ) 
  (peter_total : P + 3 * P * r = 815) 
  (david_total : P + 4 * P * r = 870) 
  : P = 650 := 
by
  sorry

end invested_sum_l237_237926


namespace sufficient_not_necessary_l237_237309

theorem sufficient_not_necessary (p q : Prop) (h : p ∧ q) : (p ∧ q → p ∨ q) ∧ ¬(p ∨ q → p ∧ q) :=
by
  sorry

end sufficient_not_necessary_l237_237309


namespace calculate_total_cost_l237_237543

def sandwich_cost : ℕ := 4
def soda_cost : ℕ := 3
def discount_threshold : ℕ := 10
def discount_rate : ℝ := 0.10
def num_sandwiches : ℕ := 7
def num_sodas : ℕ := 5

theorem calculate_total_cost :
  let total_items := num_sandwiches + num_sodas
  let cost_before_discount := num_sandwiches * sandwich_cost + num_sodas * soda_cost
  let discount := if total_items > discount_threshold then cost_before_discount * discount_rate else 0
  let final_cost := cost_before_discount - discount
  final_cost = 38.7 :=
by
  sorry

end calculate_total_cost_l237_237543


namespace class_mean_calculation_correct_l237_237735

variable (s1 s2 : ℕ) (mean1 mean2 : ℕ)
variable (n : ℕ) (mean_total : ℕ)

def overall_class_mean (s1 s2 mean1 mean2 : ℕ) : ℕ :=
  let total_score := (s1 * mean1) + (s2 * mean2)
  total_score / (s1 + s2)

theorem class_mean_calculation_correct
  (h1 : s1 = 40)
  (h2 : s2 = 10)
  (h3 : mean1 = 80)
  (h4 : mean2 = 90)
  (h5 : n = 50)
  (h6 : mean_total = 82) :
  overall_class_mean s1 s2 mean1 mean2 = mean_total :=
  sorry

end class_mean_calculation_correct_l237_237735


namespace sum_first_2009_terms_arith_seq_l237_237143

variable {a : ℕ → ℝ}

-- Given condition a_1004 + a_1005 + a_1006 = 3
axiom H : a 1004 + a 1005 + a 1006 = 3

-- Arithmetic sequence definition
def is_arith_seq (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

noncomputable def sum_arith_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem sum_first_2009_terms_arith_seq
  (d : ℝ) (h_arith_seq : is_arith_seq a d)
  : sum_arith_seq a 2009 = 2009 := 
by
  sorry

end sum_first_2009_terms_arith_seq_l237_237143


namespace total_earnings_l237_237307

variable (phone_cost : ℕ) (laptop_cost : ℕ) (computer_cost : ℕ)
variable (num_phone_repairs : ℕ) (num_laptop_repairs : ℕ) (num_computer_repairs : ℕ)

theorem total_earnings (h1 : phone_cost = 11) (h2 : laptop_cost = 15) 
                       (h3 : computer_cost = 18) (h4 : num_phone_repairs = 5) 
                       (h5 : num_laptop_repairs = 2) (h6 : num_computer_repairs = 2) :
                       (num_phone_repairs * phone_cost + num_laptop_repairs * laptop_cost + num_computer_repairs * computer_cost) = 121 := 
by
  sorry

end total_earnings_l237_237307


namespace expand_product_l237_237199

theorem expand_product (x : ℝ) : (x + 2) * (x + 5) = x^2 + 7 * x + 10 := 
by 
  sorry

end expand_product_l237_237199


namespace ellipse_equation_l237_237711

-- Definitions from conditions
def ecc (e : ℝ) := e = Real.sqrt 3 / 2
def parabola_focus (c : ℝ) (a : ℝ) := c = Real.sqrt 3 ∧ a = 2
def b_val (b a c : ℝ) := b = Real.sqrt (a^2 - c^2)

-- Main problem statement
theorem ellipse_equation (e a b c : ℝ) (x y : ℝ) :
  ecc e → parabola_focus c a → b_val b a c → (x^2 + y^2 / 4 = 1) := 
by
  intros h1 h2 h3
  sorry

end ellipse_equation_l237_237711


namespace angle_bisector_slope_l237_237146

theorem angle_bisector_slope :
  ∀ m1 m2 : ℝ, m1 = 2 → m2 = 4 → (∃ k : ℝ, k = (6 - Real.sqrt 21) / (-7) → k = (-6 + Real.sqrt 21) / 7) :=
by
  sorry

end angle_bisector_slope_l237_237146


namespace cistern_depth_l237_237342

theorem cistern_depth (h : ℝ) :
  (6 * 4 + 2 * (h * 6) + 2 * (h * 4) = 49) → (h = 1.25) :=
by
  sorry

end cistern_depth_l237_237342


namespace krishan_money_l237_237914

variable {R G K : ℕ}

theorem krishan_money 
  (h1 : R / G = 7 / 17)
  (h2 : G / K = 7 / 17)
  (hR : R = 588)
  : K = 3468 :=
by
  sorry

end krishan_money_l237_237914


namespace find_m_minus_n_l237_237271

theorem find_m_minus_n (m n : ℤ) (h1 : |m| = 14) (h2 : |n| = 23) (h3 : m + n > 0) : m - n = -9 ∨ m - n = -37 := 
sorry

end find_m_minus_n_l237_237271


namespace velocity_volleyball_league_members_l237_237365

theorem velocity_volleyball_league_members (total_cost : ℕ) (socks_cost t_shirt_cost cost_per_member members : ℕ)
  (h_socks_cost : socks_cost = 6)
  (h_t_shirt_cost : t_shirt_cost = socks_cost + 7)
  (h_cost_per_member : cost_per_member = 2 * (socks_cost + t_shirt_cost))
  (h_total_cost : total_cost = 3510)
  (h_total_cost_eq : total_cost = cost_per_member * members) :
  members = 92 :=
by
  sorry

end velocity_volleyball_league_members_l237_237365


namespace ratio_sphere_locus_l237_237348

noncomputable def sphere_locus_ratio (r : ℝ) : ℝ :=
  let F1 := 2 * Real.pi * r^2 * (1 - Real.sqrt (2 / 3))
  let F2 := Real.pi * r^2 * (2 * Real.sqrt 3 / 3)
  F1 / F2

theorem ratio_sphere_locus (r : ℝ) (h : r > 0) : sphere_locus_ratio r = Real.sqrt 3 - 1 :=
by
  sorry

end ratio_sphere_locus_l237_237348


namespace earnings_from_roosters_l237_237057

-- Definitions from the conditions
def price_per_kg : Float := 0.50
def weight_of_rooster1 : Float := 30.0
def weight_of_rooster2 : Float := 40.0

-- The theorem we need to prove (mathematically equivalent proof problem)
theorem earnings_from_roosters (p : Float := price_per_kg)
                               (w1 : Float := weight_of_rooster1)
                               (w2 : Float := weight_of_rooster2) :
  p * w1 + p * w2 = 35.0 := 
by {
  sorry
}

end earnings_from_roosters_l237_237057


namespace origin_in_circle_m_gt_5_l237_237338

theorem origin_in_circle_m_gt_5 (m : ℝ) : ((0 - 1)^2 + (0 + 2)^2 < m) → (m > 5) :=
by
  intros h
  sorry

end origin_in_circle_m_gt_5_l237_237338


namespace exponential_function_condition_l237_237739

theorem exponential_function_condition (a : ℝ) (x : ℝ) 
  (h1 : a^2 - 5 * a + 5 = 1) 
  (h2 : a > 0) 
  (h3 : a ≠ 1) : 
  a = 4 := 
sorry

end exponential_function_condition_l237_237739


namespace first_year_exceeds_two_million_l237_237083

-- Definition of the initial R&D investment in 2015
def initial_investment : ℝ := 1.3

-- Definition of the annual growth rate
def growth_rate : ℝ := 1.12

-- Definition of the investment function for year n
def investment (n : ℕ) : ℝ := initial_investment * growth_rate ^ (n - 2015)

-- The problem statement to be proven
theorem first_year_exceeds_two_million : ∃ n : ℕ, n > 2015 ∧ investment n > 2 ∧ ∀ m : ℕ, (m < n ∧ m > 2015) → investment m ≤ 2 := by
  sorry

end first_year_exceeds_two_million_l237_237083


namespace symmetric_point_condition_l237_237822

theorem symmetric_point_condition (a b : ℝ) (l : ℝ → ℝ → Prop) 
  (H_line: ∀ x y, l x y ↔ x + y + 1 = 0)
  (H_symmetric: l a b ∧ l (2*(-a-1) + a) (2*(-b-1) + b))
  : a + b = -1 :=
by 
  sorry

end symmetric_point_condition_l237_237822


namespace lisa_total_spoons_l237_237599

def number_of_baby_spoons (num_children num_spoons_per_child : Nat) : Nat :=
  num_children * num_spoons_per_child

def number_of_decorative_spoons : Nat := 2

def number_of_old_spoons (baby_spoons decorative_spoons : Nat) : Nat :=
  baby_spoons + decorative_spoons
  
def number_of_new_spoons (large_spoons teaspoons : Nat) : Nat :=
  large_spoons + teaspoons

def total_number_of_spoons (old_spoons new_spoons : Nat) : Nat :=
  old_spoons + new_spoons

theorem lisa_total_spoons
  (children : Nat)
  (spoons_per_child : Nat)
  (large_spoons : Nat)
  (teaspoons : Nat)
  (children_eq : children = 4)
  (spoons_per_child_eq : spoons_per_child = 3)
  (large_spoons_eq : large_spoons = 10)
  (teaspoons_eq : teaspoons = 15)
  : total_number_of_spoons (number_of_old_spoons (number_of_baby_spoons children spoons_per_child) number_of_decorative_spoons) (number_of_new_spoons large_spoons teaspoons) = 39 :=
by
  sorry

end lisa_total_spoons_l237_237599


namespace directrix_of_parabola_l237_237429

theorem directrix_of_parabola (x y : ℝ) : (y^2 = 8*x) → (x = -2) :=
by
  sorry

end directrix_of_parabola_l237_237429


namespace sum_of_fraction_equiv_l237_237389

theorem sum_of_fraction_equiv : 
  let x := 3.714714714
  let num := 3711
  let denom := 999
  3711 + 999 = 4710 :=
by 
  sorry

end sum_of_fraction_equiv_l237_237389


namespace area_of_triangle_CDE_l237_237022

theorem area_of_triangle_CDE
  (DE : ℝ) (h : ℝ)
  (hDE : DE = 12) (hh : h = 15) :
  1/2 * DE * h = 90 := by
  sorry

end area_of_triangle_CDE_l237_237022


namespace coefficient_x2y3_in_expansion_l237_237061

-- Define the binomial coefficient function
def binomial (n k : ℕ) : ℕ := Nat.choose n k

-- State the theorem
theorem coefficient_x2y3_in_expansion (x y : ℝ) : 
  binomial 5 3 * (2 : ℝ) ^ 2 * (-1 : ℝ) ^ 3 = -40 := by
sorry

end coefficient_x2y3_in_expansion_l237_237061


namespace max_value_f_on_interval_l237_237180

open Real

def f (x : ℝ) : ℝ := 2 * x^2 + 4 * x - 1

theorem max_value_f_on_interval :
  ∃ x ∈ Set.Icc (-2 : ℝ) (2 : ℝ), ∀ y ∈ Set.Icc (-2 : ℝ) (2 : ℝ), f y ≤ f x ∧ f x = 23 := by
  sorry

end max_value_f_on_interval_l237_237180


namespace ratio_of_shoppers_l237_237634

theorem ratio_of_shoppers (boxes ordered_of_yams: ℕ) (packages_per_box shoppers total_shoppers: ℕ)
  (h1 : packages_per_box = 25)
  (h2 : ordered_of_yams = 5)
  (h3 : total_shoppers = 375)
  (h4 : shoppers = ordered_of_yams * packages_per_box):
  (shoppers : ℕ) / total_shoppers = 1 / 3 := 
sorry

end ratio_of_shoppers_l237_237634


namespace trig_identity_l237_237334

theorem trig_identity : 4 * Real.sin (20 * Real.pi / 180) + Real.tan (20 * Real.pi / 180) = Real.sqrt 3 := 
by sorry

end trig_identity_l237_237334


namespace square_of_1008_l237_237911

theorem square_of_1008 : 1008^2 = 1016064 := 
by sorry

end square_of_1008_l237_237911


namespace john_umbrella_in_car_l237_237985

variable (UmbrellasInHouse : Nat)
variable (CostPerUmbrella : Nat)
variable (TotalAmountPaid : Nat)

theorem john_umbrella_in_car
  (h1 : UmbrellasInHouse = 2)
  (h2 : CostPerUmbrella = 8)
  (h3 : TotalAmountPaid = 24) :
  (TotalAmountPaid / CostPerUmbrella) - UmbrellasInHouse = 1 := by
  sorry

end john_umbrella_in_car_l237_237985


namespace sufficient_condition_l237_237425

theorem sufficient_condition (a b : ℝ) (h : b > a ∧ a > 0) : (a + 2) / (b + 2) > a / b :=
by sorry

end sufficient_condition_l237_237425


namespace remaining_storage_space_l237_237779

/-- Given that 1 GB = 1024 MB, a hard drive with 300 GB of total storage,
and 300000 MB of used storage, prove that the remaining storage space is 7200 MB. -/
theorem remaining_storage_space (total_gb : ℕ) (mb_per_gb : ℕ) (used_mb : ℕ) :
  total_gb = 300 → mb_per_gb = 1024 → used_mb = 300000 →
  (total_gb * mb_per_gb - used_mb) = 7200 :=
by
  intros h1 h2 h3
  sorry

end remaining_storage_space_l237_237779


namespace project_work_time_ratio_l237_237760

theorem project_work_time_ratio (A B C : ℕ) (h_ratio : A = x ∧ B = 2 * x ∧ C = 3 * x) (h_total : A + B + C = 120) : 
  (C - A = 40) :=
by
  sorry

end project_work_time_ratio_l237_237760


namespace geometric_series_six_terms_l237_237233

theorem geometric_series_six_terms :
  (1/4 - 1/16 + 1/64 - 1/256 + 1/1024 - 1/4096 : ℚ) = 4095 / 20480 :=
by
  sorry

end geometric_series_six_terms_l237_237233


namespace min_value_of_sum_l237_237025

theorem min_value_of_sum (a b : ℤ) (h : a * b = 150) : a + b = -151 :=
  sorry

end min_value_of_sum_l237_237025


namespace remainder_when_7n_divided_by_5_l237_237900

theorem remainder_when_7n_divided_by_5 (n : ℕ) (h : n % 4 = 3) : (7 * n) % 5 = 1 := 
  sorry

end remainder_when_7n_divided_by_5_l237_237900


namespace original_quadrilateral_area_l237_237432

theorem original_quadrilateral_area :
  let deg45 := (Real.pi / 4)
  let h := 1 * Real.sin deg45
  let base_bottom := 1 + 2 * h
  let area_perspective := 0.5 * (1 + base_bottom) * h
  let area_original := area_perspective * (2 * Real.sqrt 2)
  area_original = 2 + Real.sqrt 2 := by
  sorry

end original_quadrilateral_area_l237_237432


namespace sequence_eventually_periodic_modulo_l237_237815

noncomputable def a_n (n : ℕ) : ℕ :=
  n ^ n + (n - 1) ^ (n + 1)

theorem sequence_eventually_periodic_modulo (m : ℕ) (hm : m > 0) : ∃ K s : ℕ, ∀ k : ℕ, (K ≤ k → a_n (k) % m = a_n (k + s) % m) :=
sorry

end sequence_eventually_periodic_modulo_l237_237815


namespace bat_pattern_area_l237_237552

-- Define the areas of the individual components
def area_large_square : ℕ := 8
def num_large_squares : ℕ := 2

def area_medium_square : ℕ := 4
def num_medium_squares : ℕ := 2

def area_triangle : ℕ := 1
def num_triangles : ℕ := 3

-- Define the total area calculation
def total_area : ℕ :=
  (num_large_squares * area_large_square) +
  (num_medium_squares * area_medium_square) +
  (num_triangles * area_triangle)

-- The theorem statement
theorem bat_pattern_area : total_area = 27 := by
  sorry

end bat_pattern_area_l237_237552


namespace total_dots_not_visible_l237_237294

theorem total_dots_not_visible :
  let total_dots := 4 * 21
  let visible_sum := 1 + 2 + 3 + 3 + 4 + 5 + 5 + 6
  total_dots - visible_sum = 55 :=
by
  sorry

end total_dots_not_visible_l237_237294


namespace incorrect_statement_C_l237_237317

theorem incorrect_statement_C : 
  (∀ x : ℝ, |x| = x → x = 0 ∨ x = 1) ↔ False :=
by
  -- Proof goes here
  sorry

end incorrect_statement_C_l237_237317


namespace digit_b_divisible_by_7_l237_237273

theorem digit_b_divisible_by_7 (B : ℕ) (h : 0 ≤ B ∧ B ≤ 9) 
  (hdiv : (4000 + 110 * B + 3) % 7 = 0) : B = 0 :=
by
  sorry

end digit_b_divisible_by_7_l237_237273


namespace solve_f_inv_zero_l237_237954

noncomputable def f (a b x : ℝ) : ℝ := 1 / (a * x + b)
noncomputable def f_inv (a b x : ℝ) : ℝ := sorry -- this is where the inverse function definition would go

theorem solve_f_inv_zero (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : f_inv a b 0 = (1 / b) :=
by sorry

end solve_f_inv_zero_l237_237954


namespace cost_price_of_book_l237_237345

theorem cost_price_of_book (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 90) 
  (h2 : rate_of_profit = 0.8) 
  (h3 : rate_of_profit = (SP - CP) / CP) : 
  CP = 50 :=
sorry

end cost_price_of_book_l237_237345


namespace sum_even_probability_l237_237433

def probability_even_sum_of_wheels : ℚ :=
  let prob_wheel1_odd := 3 / 5
  let prob_wheel1_even := 2 / 5
  let prob_wheel2_odd := 2 / 3
  let prob_wheel2_even := 1 / 3
  (prob_wheel1_odd * prob_wheel2_odd) + (prob_wheel1_even * prob_wheel2_even)

theorem sum_even_probability :
  probability_even_sum_of_wheels = 8 / 15 :=
by
  -- Goal statement with calculations showed in the equivalent problem
  sorry

end sum_even_probability_l237_237433


namespace polar_to_cartesian_coordinates_l237_237280

noncomputable def polar_to_cartesian (r θ : ℝ) : ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ)

theorem polar_to_cartesian_coordinates :
  polar_to_cartesian 2 (2 / 3 * Real.pi) = (-1, Real.sqrt 3) :=
by
  sorry

end polar_to_cartesian_coordinates_l237_237280


namespace factor_polynomial_l237_237729

theorem factor_polynomial (x : ℝ) : 66 * x^6 - 231 * x^12 = 33 * x^6 * (2 - 7 * x^6) := by
  sorry

end factor_polynomial_l237_237729


namespace sam_bikes_speed_l237_237047

noncomputable def EugeneSpeed : ℝ := 5
noncomputable def ClaraSpeed : ℝ := (3/4) * EugeneSpeed
noncomputable def SamSpeed : ℝ := (4/3) * ClaraSpeed

theorem sam_bikes_speed :
  SamSpeed = 5 :=
by
  -- Proof will be filled here.
  sorry

end sam_bikes_speed_l237_237047


namespace election_votes_total_l237_237668

theorem election_votes_total 
  (winner_votes : ℕ) (opponent1_votes opponent2_votes opponent3_votes : ℕ)
  (excess1 excess2 excess3 : ℕ)
  (h1 : winner_votes = opponent1_votes + excess1)
  (h2 : winner_votes = opponent2_votes + excess2)
  (h3 : winner_votes = opponent3_votes + excess3)
  (votes_winner : winner_votes = 195)
  (votes_opponent1 : opponent1_votes = 142)
  (votes_opponent2 : opponent2_votes = 116)
  (votes_opponent3 : opponent3_votes = 90)
  (he1 : excess1 = 53)
  (he2 : excess2 = 79)
  (he3 : excess3 = 105) :
  winner_votes + opponent1_votes + opponent2_votes + opponent3_votes = 543 :=
by sorry

end election_votes_total_l237_237668


namespace find_possible_sets_C_l237_237328

open Set

def A : Set ℕ := {3, 4}
def B : Set ℕ := {0, 1, 2, 3, 4}
def possible_C_sets : Set (Set ℕ) :=
  { {3, 4}, {3, 4, 0}, {3, 4, 1}, {3, 4, 2}, {3, 4, 0, 1},
    {3, 4, 0, 2}, {3, 4, 1, 2}, {0, 1, 2, 3, 4} }

theorem find_possible_sets_C :
  {C : Set ℕ | A ⊆ C ∧ C ⊆ B} = possible_C_sets :=
by
  sorry

end find_possible_sets_C_l237_237328


namespace min_value_x_3y_6z_l237_237175

theorem min_value_x_3y_6z (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0 ∧ xyz = 27) : x + 3 * y + 6 * z ≥ 27 :=
sorry

end min_value_x_3y_6z_l237_237175


namespace Jean_average_speed_correct_l237_237529

noncomputable def Jean_avg_speed_until_meet
    (total_distance : ℕ)
    (chantal_flat_distance : ℕ)
    (chantal_flat_speed : ℕ)
    (chantal_steep_distance : ℕ)
    (chantal_steep_ascend_speed : ℕ)
    (chantal_steep_descend_distance : ℕ)
    (chantal_steep_descend_speed : ℕ)
    (jean_meet_position_ratio : ℚ) : ℚ :=
  let chantal_flat_time := (chantal_flat_distance : ℚ) / chantal_flat_speed
  let chantal_steep_ascend_time := (chantal_steep_distance : ℚ) / chantal_steep_ascend_speed
  let chantal_steep_descend_time := (chantal_steep_descend_distance : ℚ) / chantal_steep_descend_speed
  let total_time_until_meet := chantal_flat_time + chantal_steep_ascend_time + chantal_steep_descend_time
  let jean_distance_until_meet := (jean_meet_position_ratio * chantal_steep_distance : ℚ) + chantal_flat_distance
  jean_distance_until_meet / total_time_until_meet

theorem Jean_average_speed_correct :
  Jean_avg_speed_until_meet 6 3 5 3 3 1 4 (1 / 3) = 80 / 37 :=
by
  sorry

end Jean_average_speed_correct_l237_237529


namespace infinitely_many_lovely_no_lovely_square_gt_1_l237_237458

def lovely (n : ℕ) : Prop :=
  ∃ (k : ℕ) (d : Fin k → ℕ),
    n = (List.ofFn d).prod ∧
    ∀ i, (d i)^2 ∣ n + (d i)

theorem infinitely_many_lovely : ∀ N : ℕ, ∃ n > N, lovely n :=
  sorry

theorem no_lovely_square_gt_1 : ∀ n : ℕ, n > 1 → lovely n → ¬∃ m, n = m^2 :=
  sorry

end infinitely_many_lovely_no_lovely_square_gt_1_l237_237458


namespace each_piece_of_paper_weight_l237_237961

noncomputable def paper_weight : ℚ :=
 sorry

theorem each_piece_of_paper_weight (w : ℚ) (n : ℚ) (envelope_weight : ℚ) (stamps_needed : ℚ) (paper_pieces : ℚ) :
  paper_pieces = 8 →
  envelope_weight = 2/5 →
  stamps_needed = 2 →
  n = paper_pieces * w + envelope_weight →
  n ≤ stamps_needed →
  w = 1/5 :=
by sorry

end each_piece_of_paper_weight_l237_237961


namespace random_event_proof_l237_237898

def statement_A := "Strong youth leads to a strong country"
def statement_B := "Scooping the moon in the water"
def statement_C := "Waiting by the stump for a hare"
def statement_D := "Green waters and lush mountains are mountains of gold and silver"

def is_random_event (statement : String) : Prop :=
statement = statement_C

theorem random_event_proof : is_random_event statement_C :=
by
  -- Based on the analysis in the problem, Statement C is determined to be random.
  sorry

end random_event_proof_l237_237898


namespace smallest_integer_inequality_l237_237612

theorem smallest_integer_inequality (x y z : ℝ) : 
  (x^3 + y^3 + z^3)^2 ≤ 3 * (x^6 + y^6 + z^6) ∧ 
  (∃ n : ℤ, (0 < n ∧ n < 3) → ∀ x y z : ℝ, ¬(x^3 + y^3 + z^3)^2 ≤ n * (x^6 + y^6 + z^6)) :=
by
  sorry

end smallest_integer_inequality_l237_237612


namespace average_score_of_all_matches_is_36_l237_237455

noncomputable def average_score_of_all_matches
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) : ℝ :=
  (x + y + a + b + c) / 5

theorem average_score_of_all_matches_is_36
  (x y a b c : ℝ)
  (h1 : (x + y) / 2 = 30)
  (h2 : (a + b + c) / 3 = 40)
  (h3x : x ≤ 60)
  (h3y : y ≤ 60)
  (h3a : a ≤ 60)
  (h3b : b ≤ 60)
  (h3c : c ≤ 60)
  (h4 : x + y ≥ 100 ∨ a + b + c ≥ 100) :
  average_score_of_all_matches x y a b c h1 h2 h3x h3y h3a h3b h3c h4 = 36 := 
  by 
  sorry

end average_score_of_all_matches_is_36_l237_237455


namespace union_P_Q_l237_237246

noncomputable def P : Set ℝ := {x : ℝ | abs x ≥ 3}
noncomputable def Q : Set ℝ := {y : ℝ | ∃ x : ℝ, y = 2^x - 1}

theorem union_P_Q :
  (P ∪ Q) = Set.Iic (-3) ∪ Set.Ici (-1) :=
by {
  sorry
}

end union_P_Q_l237_237246


namespace number_of_students_at_table_l237_237921

theorem number_of_students_at_table :
  ∃ (n : ℕ), n ∣ 119 ∧ (n = 7 ∨ n = 17) :=
sorry

end number_of_students_at_table_l237_237921


namespace slope_parallel_to_line_l237_237184

theorem slope_parallel_to_line (x y : ℝ) (h : 3 * x - 6 * y = 15) :
  (∃ m, (∀ b, y = m * x + b) ∧ (∀ k, k ≠ m → ¬ 3 * x - 6 * (k * x + b) = 15)) →
  ∃ p, p = 1/2 :=
sorry

end slope_parallel_to_line_l237_237184


namespace chi_square_test_l237_237436

-- Conditions
def n : ℕ := 100
def a : ℕ := 5
def b : ℕ := 55
def c : ℕ := 15
def d : ℕ := 25

-- Critical chi-square value for alpha = 0.001
def chi_square_critical : ℝ := 10.828

-- Calculated chi-square value
noncomputable def chi_square_value : ℝ :=
  (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))

-- Statement to prove
theorem chi_square_test : chi_square_value > chi_square_critical :=
by sorry

end chi_square_test_l237_237436


namespace right_triangle_acute_angle_30_l237_237364

theorem right_triangle_acute_angle_30 (α β : ℝ) (h1 : α = 60) (h2 : α + β + 90 = 180) : β = 30 :=
by
  sorry

end right_triangle_acute_angle_30_l237_237364


namespace sum_of_coefficients_l237_237889

theorem sum_of_coefficients (a : ℕ → ℝ) :
  (∀ x : ℝ, (2 - x) ^ 10 = a 0 + a 1 * x + a 2 * x ^ 2 + a 3 * x ^ 3 + a 4 * x ^ 4 + a 5 * x ^ 5 + a 6 * x ^ 6 + a 7 * x ^ 7 + a 8 * x ^ 8 + a 9 * x ^ 9 + a 10 * x ^ 10) →
  a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 1 →
  a 0 = 1024 →
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = -1023 :=  
by
  intro h1 h2 h3
  sorry

end sum_of_coefficients_l237_237889


namespace fraction_irreducible_l237_237644

theorem fraction_irreducible (a b c d : ℤ) (h : a * d - b * c = 1) : ∀ m : ℤ, m > 1 → ¬ (m ∣ (a^2 + b^2) ∧ m ∣ (a * c + b * d)) :=
by sorry

end fraction_irreducible_l237_237644


namespace multiply_res_l237_237414

theorem multiply_res (
  h : 213 * 16 = 3408
) : 1.6 * 213 = 340.8 :=
sorry

end multiply_res_l237_237414


namespace ranking_possibilities_l237_237771

theorem ranking_possibilities (A B C D E : Type) : 
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1 → n ≠ last)) →
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 5 → (n ≠ 1)) →
  ∃ (positions : Finset (List ℕ)),
    positions.card = 54 :=
by
  sorry

end ranking_possibilities_l237_237771


namespace cost_of_purchase_l237_237763

theorem cost_of_purchase :
  (5 * 3) + (8 * 2) = 31 :=
by
  sorry

end cost_of_purchase_l237_237763


namespace boat_speed_in_still_water_l237_237823

theorem boat_speed_in_still_water (V_b : ℝ) : 
  (∀ t : ℝ, t = 26 / (V_b + 6) → t = 14 / (V_b - 6)) → V_b = 20 :=
by
  sorry

end boat_speed_in_still_water_l237_237823


namespace fixed_point_of_transformed_exponential_l237_237435

variable (a : ℝ)
variable (h_pos : 0 < a)
variable (h_ne_one : a ≠ 1)

theorem fixed_point_of_transformed_exponential :
    (∃ x y : ℝ, (y = a^(x-2) + 2) ∧ (y = x) ∧ (x = 2) ∧ (y = 3)) :=
by {
    sorry -- Proof goes here
}

end fixed_point_of_transformed_exponential_l237_237435


namespace ellipse_constants_sum_l237_237853

/-- Given the center of the ellipse at (h, k) = (3, -5),
    the semi-major axis a = 7,
    and the semi-minor axis b = 4,
    prove that h + k + a + b = 9. -/
theorem ellipse_constants_sum :
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  h + k + a + b = 9 :=
by
  let h := 3
  let k := -5
  let a := 7
  let b := 4
  sorry

end ellipse_constants_sum_l237_237853


namespace solve_for_r_l237_237358

theorem solve_for_r : ∃ r : ℝ, r ≠ 4 ∧ r ≠ 5 ∧ 
  (r^2 - 6*r + 8) / (r^2 - 9*r + 20) = (r^2 - 3*r - 10) / (r^2 - 2*r - 15) ↔ 
  r = 2*Real.sqrt 2 ∨ r = -2*Real.sqrt 2 := 
by {
  sorry
}

end solve_for_r_l237_237358


namespace incorrect_C_l237_237346

variable (D : ℝ → ℝ)

-- Definitions to encapsulate conditions
def range_D : Set ℝ := {0, 1}
def is_even := ∀ x, D x = D (-x)
def is_periodic := ∀ T > 0, ∃ p, ∀ x, D (x + p) = D x
def is_monotonic := ∀ x y, x < y → D x ≤ D y

-- The proof statement
theorem incorrect_C : ¬ is_periodic D :=
sorry

end incorrect_C_l237_237346


namespace eve_walked_distance_l237_237378

-- Defining the distances Eve ran and walked
def distance_ran : ℝ := 0.7
def distance_walked : ℝ := distance_ran - 0.1

-- Proving that the distance Eve walked is 0.6 mile
theorem eve_walked_distance : distance_walked = 0.6 := by
  -- The proof is omitted.
  sorry

end eve_walked_distance_l237_237378


namespace price_without_and_with_coupon_l237_237832

theorem price_without_and_with_coupon
  (commission_rate sale_tax_rate discount_rate : ℝ)
  (cost producer_price shipping_fee: ℝ)
  (S: ℝ)
  (h_commission: commission_rate = 0.20)
  (h_sale_tax: sale_tax_rate = 0.08)
  (h_discount: discount_rate = 0.10)
  (h_producer_price: producer_price = 20)
  (h_shipping_fee: shipping_fee = 5)
  (h_total_cost: cost = producer_price + shipping_fee)
  (h_profit: 0.20 * cost = 5)
  (h_total_earn: cost + sale_tax_rate * S + 5 = 0.80 * S)
  (h_S: S = 41.67):
  S = 41.67 ∧ 0.90 * S = 37.50 :=
by
  sorry

end price_without_and_with_coupon_l237_237832


namespace proof_of_problem_l237_237216

noncomputable def problem : Prop :=
  (1 + Real.cos (20 * Real.pi / 180)) / (2 * Real.sin (20 * Real.pi / 180)) -
  (Real.sin (10 * Real.pi / 180) * 
  (1 / Real.tan (5 * Real.pi / 180) - Real.tan (5 * Real.pi / 180))) =
  (Real.sqrt 3) / 2

theorem proof_of_problem : problem :=
by
  sorry

end proof_of_problem_l237_237216


namespace x_intercept_of_line_l237_237448

theorem x_intercept_of_line (x y : ℚ) (h : 4 * x + 6 * y = 24) (hy : y = 0) : (x, y) = (6, 0) :=
by
  sorry

end x_intercept_of_line_l237_237448


namespace equilateral_division_l237_237261

theorem equilateral_division (k : ℕ) :
  (k = 1 ∨ k = 3 ∨ k = 4 ∨ k = 9 ∨ k = 12 ∨ k = 36) ↔
  (k ∣ 36 ∧ ¬ (k = 2 ∨ k = 6 ∨ k = 18)) := by
  sorry

end equilateral_division_l237_237261


namespace negation_of_universal_statement_l237_237765

theorem negation_of_universal_statement :
  ¬(∀ x : ℝ, Real.sin x ≤ 1) ↔ ∃ x : ℝ, Real.sin x > 1 :=
by 
  -- Proof steps would be added here
  sorry

end negation_of_universal_statement_l237_237765


namespace inequality_product_geq_two_power_n_equality_condition_l237_237806

open Real BigOperators

noncomputable def is_solution (a : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i ∧ a i = 1

theorem inequality_product_geq_two_power_n (a : ℕ → ℝ) (n : ℕ)
  (h1 : ( ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i))
  (h2 : ∑ i in Finset.range n, a (i + 1) = n) :
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) ≥ 2 ^ n :=
sorry

theorem equality_condition (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ n → 0 < a i)
  (h2 : ∑ i in Finset.range n, a (i + 1) = n):
  (∏ i in Finset.range n, (1 + 1 / a (i + 1))) = 2 ^ n ↔ is_solution a n :=
sorry

end inequality_product_geq_two_power_n_equality_condition_l237_237806


namespace find_n_and_d_l237_237988

theorem find_n_and_d (n d : ℕ) (hn_pos : 0 < n) (hd_digit : d < 10)
    (h1 : 3 * n^2 + 2 * n + d = 263)
    (h2 : 3 * n^2 + 2 * n + 4 = 1 * 8^3 + 1 * 8^2 + d * 8 + 1) :
    n + d = 12 := 
sorry

end find_n_and_d_l237_237988


namespace polynomial_identity_l237_237712

open Function

-- Define the polynomial terms
def f1 (x : ℝ) := 2*x^5 + 4*x^3 + 3*x + 4
def f2 (x : ℝ) := x^4 - 2*x^3 + 3
def g (x : ℝ) := -2*x^5 + x^4 - 6*x^3 - 3*x - 1

-- Lean theorem statement
theorem polynomial_identity :
  ∀ x : ℝ, f1 x + g x = f2 x :=
by
  intros x
  sorry

end polynomial_identity_l237_237712


namespace find_remainder_l237_237645

-- Given conditions
def dividend : ℕ := 144
def divisor : ℕ := 11
def quotient : ℕ := 13

-- Theorem statement
theorem find_remainder (dividend divisor quotient : ℕ) (h1 : dividend = divisor * quotient + 1):
  ∃ r, r = dividend % divisor := 
by 
  exists 1
  sorry

end find_remainder_l237_237645


namespace base6_base5_subtraction_in_base10_l237_237878

def base6_to_nat (n : ℕ) : ℕ :=
  3 * 6^2 + 2 * 6^1 + 5 * 6^0

def base5_to_nat (n : ℕ) : ℕ :=
  2 * 5^2 + 3 * 5^1 + 1 * 5^0

theorem base6_base5_subtraction_in_base10 : base6_to_nat 325 - base5_to_nat 231 = 59 := by
  sorry

end base6_base5_subtraction_in_base10_l237_237878


namespace smallest_n_gcd_l237_237312

theorem smallest_n_gcd (n : ℕ) :
  (∃ n > 0, gcd (11 * n - 3) (8 * n + 2) > 1) ∧ (∀ m > 0, gcd (11 * m - 3) (8 * m + 2) > 1 → m ≥ n) ↔ n = 19 :=
by
  sorry

end smallest_n_gcd_l237_237312


namespace find_number_l237_237443

theorem find_number (x : ℝ) : 
  ( ((x - 1.9) * 1.5 + 32) / 2.5 = 20 ) → x = 13.9 :=
by
  sorry

end find_number_l237_237443


namespace problem_l237_237306

def count_numbers_with_more_ones_than_zeros (n : ℕ) : ℕ :=
  -- function that counts numbers less than or equal to 'n'
  -- whose binary representation has more '1's than '0's
  sorry

theorem problem (M := count_numbers_with_more_ones_than_zeros 1500) : 
  M % 1000 = 884 :=
sorry

end problem_l237_237306


namespace angle_R_values_l237_237380

theorem angle_R_values (P Q : ℝ) (h1: 5 * Real.sin P + 2 * Real.cos Q = 5) (h2: 2 * Real.sin Q + 5 * Real.cos P = 3) : 
  ∃ R : ℝ, R = Real.arcsin (1/20) ∨ R = 180 - Real.arcsin (1/20) :=
by
  sorry

end angle_R_values_l237_237380


namespace correctness_of_statements_l237_237781

theorem correctness_of_statements :
  (statement1 ∧ statement4 ∧ statement5) :=
by sorry

end correctness_of_statements_l237_237781


namespace f_2010_eq_0_l237_237723

theorem f_2010_eq_0 (f : ℝ → ℝ) (h1 : ∀ x : ℝ, f (-x) = -f x) (h2 : ∀ x : ℝ, f (x + 2) = f x) : 
  f 2010 = 0 :=
by sorry

end f_2010_eq_0_l237_237723


namespace mass_percentage_O_in_C6H8O6_l237_237657

theorem mass_percentage_O_in_C6H8O6 :
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  mass_percentage_O = 72.67 :=
by
  -- Definitions
  let atomic_mass_C := 12.01
  let atomic_mass_H := 1.01
  let atomic_mass_O := 16.00
  let molar_mass_C6H8O6 := (6 * atomic_mass_C) + (8 * atomic_mass_H) + (6 * atomic_mass_O)
  let mass_of_oxygen := 8 * atomic_mass_O
  let mass_percentage_O := (mass_of_oxygen / molar_mass_C6H8O6) * 100
  -- Proof
  sorry

end mass_percentage_O_in_C6H8O6_l237_237657


namespace quadratic_roots_l237_237787

theorem quadratic_roots:
  ∀ x : ℝ, x^2 - 1 = 0 ↔ (x = -1 ∨ x = 1) :=
by
  sorry

end quadratic_roots_l237_237787


namespace lcm_two_numbers_l237_237115

theorem lcm_two_numbers
  (a b : ℕ)
  (hcf_ab : Nat.gcd a b = 20)
  (product_ab : a * b = 2560) :
  Nat.lcm a b = 128 :=
by
  sorry

end lcm_two_numbers_l237_237115


namespace range_of_y_coordinate_of_C_l237_237968

-- Define the given parabola equation
def on_parabola (x y : ℝ) : Prop := y^2 = x + 4

-- Define the coordinates for point A
def A : (ℝ × ℝ) := (0, 2)

-- Determine if points B and C lies on the parabola
def point_on_parabola (B C : ℝ × ℝ) : Prop :=
  on_parabola B.1 B.2 ∧ on_parabola C.1 C.2

-- Determine if lines AB and BC are perpendicular
def perpendicular_slopes (B C : ℝ × ℝ) : Prop :=
  let k_AB := (B.2 - A.2) / (B.1 - A.1)
  let k_BC := (C.2 - B.2) / (C.1 - B.1)
  k_AB * k_BC = -1

-- Prove the range for y-coordinate of C
theorem range_of_y_coordinate_of_C (B C : ℝ × ℝ) (h1 : point_on_parabola B C) (h2 : perpendicular_slopes B C) :
  C.2 ≤ 0 ∨ C.2 ≥ 4 := sorry

end range_of_y_coordinate_of_C_l237_237968


namespace number_of_hens_l237_237873

-- Conditions as Lean definitions
def total_heads (H C : ℕ) : Prop := H + C = 48
def total_feet (H C : ℕ) : Prop := 2 * H + 4 * C = 136

-- Mathematically equivalent proof problem
theorem number_of_hens (H C : ℕ) (h1 : total_heads H C) (h2 : total_feet H C) : H = 28 :=
by
  sorry

end number_of_hens_l237_237873


namespace simplify_expression_l237_237659

theorem simplify_expression (x : ℝ) : 7 * x + 8 - 3 * x + 14 = 4 * x + 22 :=
by
  sorry

end simplify_expression_l237_237659


namespace cost_per_charge_l237_237727

theorem cost_per_charge
  (charges : ℕ) (budget left : ℝ) (cost_per_charge : ℝ)
  (charges_eq : charges = 4)
  (budget_eq : budget = 20)
  (left_eq : left = 6) :
  cost_per_charge = (budget - left) / charges :=
by
  apply sorry

end cost_per_charge_l237_237727


namespace max_two_digit_times_max_one_digit_is_three_digit_l237_237937

def max_two_digit : ℕ := 99
def max_one_digit : ℕ := 9
def product := max_two_digit * max_one_digit

theorem max_two_digit_times_max_one_digit_is_three_digit :
  100 ≤ product ∧ product < 1000 :=
by
  -- Prove that the product is a three-digit number
  sorry

end max_two_digit_times_max_one_digit_is_three_digit_l237_237937


namespace complement_of_M_l237_237567

def M : Set ℝ := {x | x^2 - 2 * x > 0}

def U : Set ℝ := Set.univ

theorem complement_of_M :
  (U \ M) = (Set.Icc 0 2) :=
by
  sorry

end complement_of_M_l237_237567


namespace parallel_vectors_l237_237663

def a : ℝ × ℝ := (1, 2)
def b (m : ℝ) : ℝ × ℝ := (-1, m)

theorem parallel_vectors (m : ℝ) (h : (1 : ℝ) / (-1 : ℝ) = (2 : ℝ) / m) : m = -2 :=
sorry

end parallel_vectors_l237_237663


namespace stratified_sampling_third_year_students_l237_237743

theorem stratified_sampling_third_year_students 
  (total_students : ℕ)
  (sample_size : ℕ)
  (ratio_1st : ℕ)
  (ratio_2nd : ℕ)
  (ratio_3rd : ℕ)
  (ratio_4th : ℕ)
  (h1 : total_students = 1000)
  (h2 : sample_size = 200)
  (h3 : ratio_1st = 4)
  (h4 : ratio_2nd = 3)
  (h5 : ratio_3rd = 2)
  (h6 : ratio_4th = 1) :
  (ratio_3rd : ℚ) / (ratio_1st + ratio_2nd + ratio_3rd + ratio_4th : ℚ) * sample_size = 40 :=
by
  sorry

end stratified_sampling_third_year_students_l237_237743


namespace parabola_midpoint_locus_minimum_slope_difference_exists_l237_237129

open Real

def parabola_locus (x y : ℝ) : Prop :=
  x^2 = 4 * y

def slope_difference_condition (x1 x2 k1 k2 : ℝ) : Prop :=
  |k1 - k2| = 1

theorem parabola_midpoint_locus :
  ∀ (x y : ℝ), parabola_locus x y :=
by
  intros x y
  apply sorry

theorem minimum_slope_difference_exists :
  ∀ {x1 y1 x2 y2 k1 k2 : ℝ},
  slope_difference_condition x1 x2 k1 k2 :=
by
  intros x1 y1 x2 y2 k1 k2
  apply sorry

end parabola_midpoint_locus_minimum_slope_difference_exists_l237_237129


namespace magician_decks_l237_237201

theorem magician_decks :
  ∀ (initial_decks price_per_deck earnings decks_sold decks_left_unsold : ℕ),
  initial_decks = 5 →
  price_per_deck = 2 →
  earnings = 4 →
  decks_sold = earnings / price_per_deck →
  decks_left_unsold = initial_decks - decks_sold →
  decks_left_unsold = 3 :=
by
  intros initial_decks price_per_deck earnings decks_sold decks_left_unsold
  intros h_initial h_price h_earnings h_sold h_left
  rw [h_initial, h_price, h_earnings] at *
  sorry

end magician_decks_l237_237201


namespace reflected_coordinates_l237_237228

-- Define the coordinates of point P
def point_P : ℝ × ℝ := (-2, -3)

-- Define the function for reflection across the origin
def reflect_origin (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.1, -p.2)

-- State the theorem to prove
theorem reflected_coordinates :
  reflect_origin point_P = (2, 3) := by
  sorry

end reflected_coordinates_l237_237228


namespace inequality_proof_l237_237610

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^3 + 3 * b^3) / (5 * a + b) + (b^3 + 3 * c^3) / (5 * b + c) + (c^3 + 3 * a^3) / (5 * c + a) >= (2 / 3) * (a^2 + b^2 + c^2) := 
sorry

end inequality_proof_l237_237610


namespace javier_average_hits_per_game_l237_237319

theorem javier_average_hits_per_game (total_games_first_part : ℕ) (average_hits_first_part : ℕ) 
  (remaining_games : ℕ) (average_hits_remaining : ℕ) : 
  total_games_first_part = 20 → average_hits_first_part = 2 → 
  remaining_games = 10 → average_hits_remaining = 5 →
  (total_games_first_part * average_hits_first_part + 
  remaining_games * average_hits_remaining) /
  (total_games_first_part + remaining_games) = 3 := 
by intros h1 h2 h3 h4;
   sorry

end javier_average_hits_per_game_l237_237319


namespace number_of_teams_l237_237145

-- Define the conditions
def math_club_girls : ℕ := 4
def math_club_boys : ℕ := 7
def team_girls : ℕ := 3
def team_boys : ℕ := 3

-- Compute the number of ways to choose 3 girls from 4 girls
def choose_comb_girls : ℕ := Nat.choose math_club_girls team_girls

-- Compute the number of ways to choose 3 boys from 7 boys
def choose_comb_boys : ℕ := Nat.choose math_club_boys team_boys

-- Formulate the goal statement
theorem number_of_teams : choose_comb_girls * choose_comb_boys = 140 := by
  sorry

end number_of_teams_l237_237145


namespace simplify_fraction_l237_237247

theorem simplify_fraction (a b m n : ℕ) (h : a ≠ 0 ∧ b ≠ 0 ∧ m ≠ 0 ∧ n ≠ 0) : 
  (a^2 * b) / (m * n^2) / ((a * b) / (3 * m * n)) = 3 * a / n :=
by
  sorry

end simplify_fraction_l237_237247


namespace intersection_of_lines_l237_237535

theorem intersection_of_lines : ∃ (x y : ℝ), (9 * x - 4 * y = 30) ∧ (7 * x + y = 11) ∧ (x = 2) ∧ (y = -3) := 
by
  sorry

end intersection_of_lines_l237_237535


namespace caleb_grandfather_age_l237_237398

theorem caleb_grandfather_age :
  let yellow_candles := 27
  let red_candles := 14
  let blue_candles := 38
  yellow_candles + red_candles + blue_candles = 79 :=
by
  sorry

end caleb_grandfather_age_l237_237398


namespace volume_ratio_of_spheres_l237_237265

theorem volume_ratio_of_spheres (r1 r2 r3 : ℝ) 
  (h : r1 / r2 = 1 / 2 ∧ r2 / r3 = 2 / 3) : 
  (4/3 * π * r3^3) = 3 * (4/3 * π * r1^3 + 4/3 * π * r2^3) :=
by
  sorry

end volume_ratio_of_spheres_l237_237265


namespace problem_l237_237637

theorem problem
  (a b : ℚ)
  (h1 : 3 * a + 5 * b = 47)
  (h2 : 7 * a + 2 * b = 52)
  : a + b = 35 / 3 :=
sorry

end problem_l237_237637


namespace one_cow_one_bag_in_34_days_l237_237682

-- Definitions: 34 cows eat 34 bags in 34 days, each cow eats one bag in those 34 days.
def cows : Nat := 34
def bags : Nat := 34
def days : Nat := 34

-- Hypothesis: each cow eats one bag in 34 days.
def one_bag_days (c : Nat) (b : Nat) : Nat := days

-- Theorem: One cow will eat one bag of husk in 34 days.
theorem one_cow_one_bag_in_34_days : one_bag_days 1 1 = 34 := sorry

end one_cow_one_bag_in_34_days_l237_237682


namespace each_baby_worms_per_day_l237_237393

variable (babies : Nat) (worms_papa : Nat) (worms_mama_caught : Nat) (worms_mama_stolen : Nat) (worms_needed : Nat)
variable (days : Nat)

theorem each_baby_worms_per_day 
  (h1 : babies = 6) 
  (h2 : worms_papa = 9) 
  (h3 : worms_mama_caught = 13) 
  (h4 : worms_mama_stolen = 2)
  (h5 : worms_needed = 34) 
  (h6 : days = 3) :
  (worms_papa + (worms_mama_caught - worms_mama_stolen) + worms_needed) / babies / days = 3 :=
by
  sorry

end each_baby_worms_per_day_l237_237393


namespace B_work_days_proof_l237_237324

-- Define the main variables
variables (W : ℝ) (x : ℝ) (daysA : ℝ) (daysBworked : ℝ) (daysAremaining : ℝ)

-- Given conditions from the problem
def A_work_days : ℝ := 6
def B_work_days : ℝ := x
def B_worked_days : ℝ := 10
def A_remaining_days : ℝ := 2

-- We are asked to prove this statement
theorem B_work_days_proof (h1 : daysA = A_work_days)
                           (h2 : daysBworked = B_worked_days)
                           (h3 : daysAremaining = A_remaining_days) 
                           (hx : (W/6 = (W - 10*W/x) / 2)) : x = 15 :=
by 
  -- Proof omitted
  sorry 

end B_work_days_proof_l237_237324


namespace monotonic_intervals_of_f_l237_237692

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.exp x - 1 - x

-- Define the derivative f'
noncomputable def f' (x : ℝ) : ℝ := Real.exp x - 1

-- Prove the monotonicity intervals of the function f
theorem monotonic_intervals_of_f :
  (∀ x : ℝ, x < 0 → f' x < 0) ∧ (∀ x : ℝ, 0 < x → f' x > 0) :=
by
  sorry

end monotonic_intervals_of_f_l237_237692


namespace complex_series_sum_l237_237258

theorem complex_series_sum (ω : ℂ) (h₁ : ω^7 = 1) (h₂ : ω ≠ 1) :
  (ω^16 + ω^18 + ω^20 + ω^22 + ω^24 + ω^26 + ω^28 + ω^30 + ω^32 + 
   ω^34 + ω^36 + ω^38 + ω^40 + ω^42 + ω^44 + ω^46 + ω^48 + ω^50 + 
   ω^52 + ω^54) = -1 :=
by
  sorry

end complex_series_sum_l237_237258


namespace find_subtracted_number_l237_237298

theorem find_subtracted_number (x y : ℝ) (h1 : x = 62.5) (h2 : (2 * (x + 5)) / 5 - y = 22) : y = 5 :=
sorry

end find_subtracted_number_l237_237298


namespace smallest_prime_after_six_nonprimes_l237_237970

-- Define the set of natural numbers and prime numbers
def is_natural (n : ℕ) : Prop := n ≥ 1
def is_prime (n : ℕ) : Prop := 1 < n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_nonprime (n : ℕ) : Prop := ¬ is_prime n

-- The condition of six consecutive nonprime numbers
def six_consecutive_nonprime (n : ℕ) : Prop := 
  is_nonprime n ∧ 
  is_nonprime (n + 1) ∧ 
  is_nonprime (n + 2) ∧ 
  is_nonprime (n + 3) ∧ 
  is_nonprime (n + 4) ∧ 
  is_nonprime (n + 5)

-- The main theorem stating that 37 is the smallest prime following six consecutive nonprime numbers
theorem smallest_prime_after_six_nonprimes : 
  ∃ (n : ℕ), six_consecutive_nonprime n ∧ is_prime (n + 6) ∧ (∀ m, m < (n + 6) → ¬ is_prime m) :=
sorry

end smallest_prime_after_six_nonprimes_l237_237970


namespace number_difference_l237_237454

theorem number_difference 
  (a1 a2 a3 : ℝ)
  (h1 : a1 = 2 * a2)
  (h2 : a1 = 3 * a3)
  (h3 : (a1 + a2 + a3) / 3 = 88) : 
  a1 - a3 = 96 :=
sorry

end number_difference_l237_237454


namespace military_unit_soldiers_l237_237459

theorem military_unit_soldiers:
  ∃ (x N : ℕ), 
      (N = x * (x + 5)) ∧
      (N = 5 * (x + 845)) ∧
      N = 4550 :=
by
  sorry

end military_unit_soldiers_l237_237459


namespace five_digit_odd_and_multiples_of_5_sum_l237_237212

theorem five_digit_odd_and_multiples_of_5_sum :
  let A := 9 * 10^3 * 5
  let B := 9 * 10^3 * 1
  A + B = 45000 := by
sorry

end five_digit_odd_and_multiples_of_5_sum_l237_237212


namespace max_value_of_n_l237_237532

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
variable (S_2015_pos : S 2015 > 0)
variable (S_2016_neg : S 2016 < 0)

theorem max_value_of_n (a : ℕ → ℝ) (S : ℕ → ℝ)
  (S_sum : ∀ n, S n = n * (a 1 + a n) / 2)
  (S_2015_pos : S 2015 > 0)
  (S_2016_neg : S 2016 < 0) : 
  ∃ n, n = 1008 ∧ ∀ m, S m < S n := 
sorry

end max_value_of_n_l237_237532


namespace find_second_quadrant_point_l237_237409

def is_second_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem find_second_quadrant_point :
  (is_second_quadrant (2, 3) = false) ∧
  (is_second_quadrant (2, -3) = false) ∧
  (is_second_quadrant (-2, -3) = false) ∧
  (is_second_quadrant (-2, 3) = true) := 
sorry

end find_second_quadrant_point_l237_237409


namespace find_triples_l237_237267

-- Define the conditions in Lean 4
def is_prime (p : ℕ) : Prop := Nat.Prime p

def is_positive_integer (n : ℕ) : Prop := n > 0

-- Define the math proof problem
theorem find_triples (m n p : ℕ) (hp : is_prime p) (hm : is_positive_integer m) (hn : is_positive_integer n) : 
  p^n + 3600 = m^2 ↔ (m = 61 ∧ n = 2 ∧ p = 11) ∨ (m = 65 ∧ n = 4 ∧ p = 5) ∨ (m = 68 ∧ n = 10 ∧ p = 2) :=
by
  sorry

end find_triples_l237_237267


namespace range_of_k_l237_237929

noncomputable def meets_hyperbola (k : ℝ) : Prop :=
  ∃ (x y : ℝ), x^2 - y^2 = 4 ∧ y = k * x - 1

theorem range_of_k : 
  { k : ℝ | meets_hyperbola k } = { k : ℝ | k = 1 ∨ k = -1 ∨ - (Real.sqrt 5) / 2 ≤ k ∧ k ≤ (Real.sqrt 5) / 2 } :=
by
  sorry

end range_of_k_l237_237929


namespace shorter_side_of_quilt_l237_237207

theorem shorter_side_of_quilt :
  ∀ (x : ℕ), (∃ y : ℕ, 24 * y = 144) -> x = 6 :=
by
  intros x h
  sorry

end shorter_side_of_quilt_l237_237207


namespace find_angle_A_l237_237058

theorem find_angle_A (A B C a b c : ℝ) 
  (h1 : 0 < A ∧ A < π)
  (h2 : 0 < B ∧ B < π)
  (h3 : 0 < C ∧ C < π)
  (h4 : a > 0)
  (h5 : b > 0)
  (h6 : c > 0)
  (sin_eq : Real.sin (C + π / 6) = b / (2 * a)) :
  A = π / 6 :=
sorry

end find_angle_A_l237_237058


namespace modular_inverse_sum_eq_14_l237_237913

theorem modular_inverse_sum_eq_14 : 
(9 + 13 + 15 + 16 + 12 + 3 + 14) % 17 = 14 := by
  sorry

end modular_inverse_sum_eq_14_l237_237913


namespace optimal_washing_effect_l237_237621

noncomputable def optimal_laundry_addition (x y : ℝ) : Prop :=
  (5 + 0.02 * 2 + x + y = 20) ∧
  (0.02 * 2 + x = (20 - 5) * 0.004)

theorem optimal_washing_effect :
  ∃ x y : ℝ, optimal_laundry_addition x y ∧ x = 0.02 ∧ y = 14.94 :=
by
  sorry

end optimal_washing_effect_l237_237621


namespace chantal_gain_l237_237666

variable (sweaters balls cost_selling cost_yarn total_gain : ℕ)

def chantal_knits_sweaters : Prop :=
  sweaters = 28 ∧
  balls = 4 ∧
  cost_yarn = 6 ∧
  cost_selling = 35 ∧
  total_gain = (sweaters * cost_selling) - (sweaters * balls * cost_yarn)

theorem chantal_gain : chantal_knits_sweaters sweaters balls cost_selling cost_yarn total_gain → total_gain = 308 :=
by sorry

end chantal_gain_l237_237666


namespace find_red_cards_l237_237445

-- We use noncomputable here as we are dealing with real numbers in a theoretical proof context.
noncomputable def red_cards (r b : ℕ) (_initial_prob : r / (r + b) = 1 / 5) 
                            (_added_prob : r / (r + b + 6) = 1 / 7) : ℕ := 
r

theorem find_red_cards 
  {r b : ℕ}
  (h1 : r / (r + b) = 1 / 5)
  (h2 : r / (r + b + 6) = 1 / 7) : 
  red_cards r b h1 h2 = 3 :=
sorry  -- Proof not required

end find_red_cards_l237_237445


namespace quadratic_has_two_distinct_roots_l237_237062

noncomputable def discriminant (a b c : ℝ) : ℝ :=
  b^2 - 4 * a * c

theorem quadratic_has_two_distinct_roots 
  (a b c : ℝ) (h : 2016 + a^2 + a * c < a * b) : 
  discriminant a b c > 0 :=
sorry

end quadratic_has_two_distinct_roots_l237_237062


namespace wendi_owns_rabbits_l237_237753

/-- Wendi's plot of land is 200 feet by 900 feet. -/
def area_land_in_feet : ℕ := 200 * 900

/-- One rabbit can eat enough grass to clear ten square yards of lawn area per day. -/
def rabbit_clear_per_day : ℕ := 10

/-- It would take 20 days for all of Wendi's rabbits to clear all the grass off of her grassland property. -/
def days_to_clear : ℕ := 20

/-- Convert feet to yards (3 feet in a yard). -/
def feet_to_yards (feet : ℕ) : ℕ := feet / 3

/-- Calculate the total area of the land in square yards. -/
def area_land_in_yards : ℕ := (feet_to_yards 200) * (feet_to_yards 900)

theorem wendi_owns_rabbits (total_area : ℕ := area_land_in_yards)
                            (clear_area_per_rabbit : ℕ := rabbit_clear_per_day * days_to_clear) :
  total_area / clear_area_per_rabbit = 100 := 
sorry

end wendi_owns_rabbits_l237_237753


namespace winnie_balloons_remainder_l237_237206

theorem winnie_balloons_remainder :
  let red_balloons := 20
  let white_balloons := 40
  let green_balloons := 70
  let chartreuse_balloons := 90
  let violet_balloons := 15
  let friends := 10
  let total_balloons := red_balloons + white_balloons + green_balloons + chartreuse_balloons + violet_balloons
  total_balloons % friends = 5 :=
by
  sorry

end winnie_balloons_remainder_l237_237206


namespace find_abc_l237_237055

open Real

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c) 
  (h1 : a * (b + c) = 154)
  (h2 : b * (c + a) = 164) 
  (h3 : c * (a + b) = 172) : 
  (a * b * c = Real.sqrt 538083) := 
by 
  sorry

end find_abc_l237_237055


namespace quadratic_int_roots_iff_n_eq_3_or_4_l237_237218

theorem quadratic_int_roots_iff_n_eq_3_or_4 (n : ℕ) (hn : 0 < n) :
    (∃ m k : ℤ, (m ≠ k) ∧ (m^2 - 4 * m + n = 0) ∧ (k^2 - 4 * k + n = 0)) ↔ (n = 3 ∨ n = 4) := sorry

end quadratic_int_roots_iff_n_eq_3_or_4_l237_237218


namespace third_square_area_difference_l237_237154

def side_length (p : ℕ) : ℕ :=
  p / 4

def area (s : ℕ) : ℕ :=
  s * s

theorem third_square_area_difference
  (p1 p2 p3 : ℕ)
  (h1 : p1 = 60)
  (h2 : p2 = 48)
  (h3 : p3 = 36)
  : area (side_length p3) = area (side_length p1) - area (side_length p2) :=
by
  sorry

end third_square_area_difference_l237_237154


namespace workers_work_5_days_a_week_l237_237640

def total_weekly_toys : ℕ := 5500
def daily_toys : ℕ := 1100
def days_worked : ℕ := total_weekly_toys / daily_toys

theorem workers_work_5_days_a_week : days_worked = 5 := 
by 
  sorry

end workers_work_5_days_a_week_l237_237640


namespace boxes_sold_l237_237962

def case_size : ℕ := 12
def remaining_boxes : ℕ := 7

theorem boxes_sold (sold_boxes : ℕ) : ∃ n : ℕ, sold_boxes = n * case_size + remaining_boxes :=
sorry

end boxes_sold_l237_237962


namespace simplify_subtracted_terms_l237_237401

theorem simplify_subtracted_terms (r : ℝ) : 180 * r - 88 * r = 92 * r := 
by 
  sorry

end simplify_subtracted_terms_l237_237401


namespace find_M_coordinate_l237_237167

-- Definitions of the given points
structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def A : Point3D := ⟨1, 0, 2⟩
def B : Point3D := ⟨1, -3, 1⟩
def M (y : ℝ) : Point3D := ⟨0, y, 0⟩

-- Definition for the squared distance between two points
def dist_sq (p1 p2 : Point3D) : ℝ :=
  (p2.x - p1.x)^2 + (p2.y - p1.y)^2 + (p2.z - p1.z)^2

-- Main theorem statement
theorem find_M_coordinate (y : ℝ) : 
  dist_sq (M y) A = dist_sq (M y) B → y = -1 :=
by
  simp [dist_sq, A, B, M]
  sorry

end find_M_coordinate_l237_237167


namespace xyz_divisible_by_55_l237_237646

-- Definitions and conditions from part (a)
variables (x y z a b c : ℤ)
variable (h1 : x^2 + y^2 = a^2)
variable (h2 : y^2 + z^2 = b^2)
variable (h3 : z^2 + x^2 = c^2)

-- The final statement to prove that xyz is divisible by 55
theorem xyz_divisible_by_55 : 55 ∣ x * y * z := 
by sorry

end xyz_divisible_by_55_l237_237646


namespace smallest_difference_l237_237738

noncomputable def triangle_lengths (DE EF FD : ℕ) : Prop :=
  DE < EF ∧ EF ≤ FD ∧ DE + EF + FD = 3010 ∧ DE + EF > FD ∧ EF + FD > DE ∧ FD + DE > EF

theorem smallest_difference :
  ∃ (DE EF FD : ℕ), triangle_lengths DE EF FD ∧ EF - DE = 1 :=
by
  sorry

end smallest_difference_l237_237738


namespace negation_of_p_l237_237137

-- Define the original proposition p
def p : Prop := ∀ x : ℝ, x ≥ 2

-- State the proof problem as a Lean theorem
theorem negation_of_p : (∀ x : ℝ, x ≥ 2) → ∃ x₀ : ℝ, x₀ < 2 :=
by
  intro h
  -- Define how the proof would generally proceed
  -- as the negation of a universal statement is an existential statement.
  sorry

end negation_of_p_l237_237137


namespace kevin_wings_record_l237_237031

-- Conditions
def alanWingsPerMinute : ℕ := 5
def additionalWingsNeeded : ℕ := 4
def kevinRecordDuration : ℕ := 8

-- Question and answer
theorem kevin_wings_record : 
  (alanWingsPerMinute + additionalWingsNeeded) * kevinRecordDuration = 72 :=
by
  sorry

end kevin_wings_record_l237_237031


namespace quadratic_inequality_l237_237310

variable (b c : ℝ)

def f (x : ℝ) : ℝ := x^2 + b * x + c

theorem quadratic_inequality (h : f b c (-1) = f b c 3) : f b c 1 < c ∧ c < f b c 3 :=
by
  sorry

end quadratic_inequality_l237_237310


namespace gcd_840_1764_l237_237796

theorem gcd_840_1764 : Nat.gcd 840 1764 = 84 :=
by
  sorry

end gcd_840_1764_l237_237796


namespace cyclist_downhill_speed_l237_237304

noncomputable def downhill_speed (d uphill_speed avg_speed : ℝ) : ℝ :=
  let downhill_speed := (2 * d * uphill_speed) / (avg_speed * d - uphill_speed * 2)
  -- We want to prove
  downhill_speed

theorem cyclist_downhill_speed :
  downhill_speed 150 25 35 = 58.33 :=
by
  -- Proof omitted
  sorry

end cyclist_downhill_speed_l237_237304


namespace negative_root_no_positive_l237_237973

theorem negative_root_no_positive (a : ℝ) : 
  (∃ x : ℝ, x < 0 ∧ |x| = ax + 1) ∧ (¬ ∃ x : ℝ, x > 0 ∧ |x| = ax + 1) → a > -1 :=
by
  sorry

end negative_root_no_positive_l237_237973


namespace calculate_expression_l237_237367

theorem calculate_expression (a : ℝ) : 3 * a * (2 * a^2 - 4 * a) - 2 * a^2 * (3 * a + 4) = -20 * a^2 :=
by
  sorry

end calculate_expression_l237_237367


namespace jame_annual_earnings_difference_l237_237447

-- Define conditions
def new_hourly_wage := 20
def new_hours_per_week := 40
def old_hourly_wage := 16
def old_hours_per_week := 25
def weeks_per_year := 52

-- Define annual earnings calculations
def annual_earnings_old (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

def annual_earnings_new (hourly_wage : ℕ) (hours_per_week : ℕ) (weeks_per_year : ℕ) : ℕ :=
  hourly_wage * hours_per_week * weeks_per_year

-- Problem statement to prove
theorem jame_annual_earnings_difference :
  annual_earnings_new new_hourly_wage new_hours_per_week weeks_per_year -
  annual_earnings_old old_hourly_wage old_hours_per_week weeks_per_year = 20800 := by
  sorry

end jame_annual_earnings_difference_l237_237447


namespace final_stamp_collection_l237_237758

section StampCollection

structure Collection :=
  (nature : ℕ)
  (architecture : ℕ)
  (animals : ℕ)
  (vehicles : ℕ)
  (famous_people : ℕ)

def initial_collections : Collection := {
  nature := 10, architecture := 15, animals := 12, vehicles := 6, famous_people := 4
}

-- define transactions as functions that take a collection and return a modified collection
def transaction1 (c : Collection) : Collection :=
  { c with nature := c.nature + 4, architecture := c.architecture + 5, animals := c.animals + 5, vehicles := c.vehicles + 2, famous_people := c.famous_people + 1 }

def transaction2 (c : Collection) : Collection := 
  { c with nature := c.nature + 2, animals := c.animals - 1 }

def transaction3 (c : Collection) : Collection := 
  { c with animals := c.animals - 5, architecture := c.architecture + 3 }

def transaction4 (c : Collection) : Collection :=
  { c with animals := c.animals - 4, nature := c.nature + 7 }

def transaction7 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles - 2, nature := c.nature + 5 }

def transaction8 (c : Collection) : Collection :=
  { c with vehicles := c.vehicles + 3, famous_people := c.famous_people - 3 }

def final_collection (c : Collection) : Collection :=
  transaction8 (transaction7 (transaction4 (transaction3 (transaction2 (transaction1 c)))))

theorem final_stamp_collection :
  final_collection initial_collections = { nature := 28, architecture := 23, animals := 7, vehicles := 9, famous_people := 2 } :=
by
  -- skip the proof
  sorry

end StampCollection

end final_stamp_collection_l237_237758


namespace associate_professor_charts_l237_237818

theorem associate_professor_charts (A B C : ℕ) : 
  A + B = 8 → 
  2 * A + B = 10 → 
  C * A + 2 * B = 14 → 
  C = 1 := 
by 
  intros h1 h2 h3 
  sorry

end associate_professor_charts_l237_237818


namespace contrapositive_of_sum_of_squares_l237_237101

theorem contrapositive_of_sum_of_squares
  (a b : ℝ)
  (h : a ≠ 0 ∨ b ≠ 0) :
  a^2 + b^2 ≠ 0 := 
sorry

end contrapositive_of_sum_of_squares_l237_237101


namespace proof_A_cap_complement_B_l237_237519

variable (A B U : Set ℕ) (h1 : A ⊆ U) (h2 : B ⊆ U)
variable (h3 : U = {1, 2, 3, 4})
variable (h4 : (U \ (A ∪ B)) = {4}) -- \ represents set difference, complement in the universal set
variable (h5 : B = {1, 2})

theorem proof_A_cap_complement_B : A ∩ (U \ B) = {3} := by
  sorry

end proof_A_cap_complement_B_l237_237519


namespace tennis_tournament_rounds_l237_237584

/-- Defining the constants and conditions stated in the problem -/
def first_round_games : ℕ := 8
def second_round_games : ℕ := 4
def third_round_games : ℕ := 2
def finals_games : ℕ := 1
def cans_per_game : ℕ := 5
def balls_per_can : ℕ := 3
def total_balls_used : ℕ := 225

/-- Theorem stating the number of rounds in the tennis tournament -/
theorem tennis_tournament_rounds : 
  first_round_games + second_round_games + third_round_games + finals_games = 15 ∧
  15 * cans_per_game = 75 ∧
  75 * balls_per_can = total_balls_used →
  4 = 4 :=
by sorry

end tennis_tournament_rounds_l237_237584


namespace sequence_satisfies_recurrence_l237_237580

theorem sequence_satisfies_recurrence (n : ℕ) (a : ℕ → ℕ) (h : ∀ k, 2 ≤ k → k ≤ n - 1 → a (k + 1) = (a k ^ 2 + 1) / (a (k - 1) + 1) - 1) :
  n = 3 ∨ n = 4 := by
  sorry

end sequence_satisfies_recurrence_l237_237580


namespace root_sum_abs_gt_6_l237_237190

variables (r1 r2 p : ℝ)

theorem root_sum_abs_gt_6 
  (h1 : r1 + r2 = -p)
  (h2 : r1 * r2 = 9)
  (h3 : p^2 > 36) :
  |r1 + r2| > 6 :=
by sorry

end root_sum_abs_gt_6_l237_237190


namespace negation_of_P_l237_237660

-- Defining the original proposition
def P : Prop := ∃ x₀ : ℝ, x₀^2 = 1

-- The problem is to prove the negation of the proposition
theorem negation_of_P : (¬P) ↔ (∀ x : ℝ, x^2 ≠ 1) :=
  by sorry

end negation_of_P_l237_237660


namespace cost_of_soap_per_year_l237_237506

-- Conditions:
def duration_of_soap (bar: Nat) : Nat := 2
def cost_per_bar (bar: Nat) : Real := 8.0
def months_in_year : Nat := 12

-- Derived quantity
def bars_needed (months: Nat) (duration: Nat): Nat := months / duration

-- Theorem statement:
theorem cost_of_soap_per_year : 
  let n := bars_needed months_in_year (duration_of_soap 1)
  n * (cost_per_bar 1) = 48.0 := 
  by 
    -- Skipping proof
    sorry

end cost_of_soap_per_year_l237_237506


namespace range_of_m_l237_237816

noncomputable def f (x : ℝ) := |x - 3| - 2
noncomputable def g (x : ℝ) := -|x + 1| + 4

theorem range_of_m (m : ℝ) : (∀ x, f x - g x ≥ m + 1) ↔ m ≤ -3 :=
by
  sorry

end range_of_m_l237_237816


namespace distance_from_point_to_plane_l237_237053

-- Definitions representing the conditions
def side_length_base := 6
def base_area := side_length_base * side_length_base
def volume_pyramid := 96

-- Proof statement
theorem distance_from_point_to_plane (h : ℝ) : 
  (1/3) * base_area * h = volume_pyramid → h = 8 := 
by 
  sorry

end distance_from_point_to_plane_l237_237053


namespace flower_total_l237_237524

theorem flower_total (H C D : ℕ) (h1 : H = 34) (h2 : H = C - 13) (h3 : C = D + 23) : 
  H + C + D = 105 :=
by 
  sorry  -- Placeholder for the proof

end flower_total_l237_237524


namespace ball_bounces_less_than_two_meters_l237_237564

theorem ball_bounces_less_than_two_meters : ∀ k : ℕ, 500 * (1/3 : ℝ)^k < 2 → k ≥ 6 := by
  sorry

end ball_bounces_less_than_two_meters_l237_237564


namespace solve_for_x_l237_237863

theorem solve_for_x (x : ℝ) (h : (3 * x - 17) / 4 = (x + 12) / 5) : x = 12.09 :=
by
  sorry

end solve_for_x_l237_237863


namespace max_percent_liquid_X_l237_237021

theorem max_percent_liquid_X (wA wB wC : ℝ) (XA XB XC YA YB YC : ℝ)
  (hXA : XA = 0.8 / 100) (hXB : XB = 1.8 / 100) (hXC : XC = 3.0 / 100)
  (hYA : YA = 2.0 / 100) (hYB : YB = 1.0 / 100) (hYC : YC = 0.5 / 100)
  (hwA : wA = 500) (hwB : wB = 700) (hwC : wC = 300)
  (H_combined_limit : XA * wA + XB * wB + XC * wC + YA * wA + YB * wB + YC * wC ≤ 0.025 * (wA + wB + wC)) :
  XA * wA + XB * wB + XC * wC ≤ 0.0171 * (wA + wB + wC) :=
sorry

end max_percent_liquid_X_l237_237021


namespace bristol_to_carlisle_routes_l237_237343

-- Given conditions
def r_bb := 6
def r_bs := 3
def r_sc := 2

-- The theorem we want to prove
theorem bristol_to_carlisle_routes :
  (r_bb * r_bs * r_sc) = 36 :=
by
  sorry

end bristol_to_carlisle_routes_l237_237343


namespace trigonometric_identity_l237_237963

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 3) : 
    (2 * Real.sin α - Real.cos α) / (Real.sin α + 3 * Real.cos α) = 5 / 6 :=
by
  sorry

end trigonometric_identity_l237_237963


namespace similar_triangle_perimeter_l237_237001

theorem similar_triangle_perimeter 
  (a b c : ℝ) (ha : a = 12) (hb : b = 12) (hc : c = 24) 
  (k : ℝ) (hk : k = 1.5) : 
  (1.5 * a) + (1.5 * b) + (1.5 * c) = 72 :=
by
  sorry

end similar_triangle_perimeter_l237_237001


namespace r_exceeds_s_l237_237431

theorem r_exceeds_s (x y : ℚ) (h1 : x + 2 * y = 16 / 3) (h2 : 5 * x + 3 * y = 26) :
  x - y = 106 / 21 :=
sorry

end r_exceeds_s_l237_237431


namespace dow_jones_morning_value_l237_237943

theorem dow_jones_morning_value 
  (end_of_day_value : ℝ) 
  (percentage_fall : ℝ)
  (expected_morning_value : ℝ) 
  (h1 : end_of_day_value = 8722) 
  (h2 : percentage_fall = 0.02) 
  (h3 : expected_morning_value = 8900) :
  expected_morning_value = end_of_day_value / (1 - percentage_fall) :=
sorry

end dow_jones_morning_value_l237_237943


namespace kim_monthly_revenue_l237_237693

-- Define the cost to open the store
def initial_cost : ℤ := 25000

-- Define the monthly expenses
def monthly_expenses : ℤ := 1500

-- Define the number of months
def months : ℕ := 10

-- Define the revenue per month
def revenue_per_month (total_revenue : ℤ) (months : ℕ) : ℤ := total_revenue / months

theorem kim_monthly_revenue :
  ∃ r, revenue_per_month r months = 4000 :=
by 
  let total_expenses := monthly_expenses * months
  let total_revenue := initial_cost + total_expenses
  use total_revenue
  unfold revenue_per_month
  sorry

end kim_monthly_revenue_l237_237693


namespace find_pairs_of_square_numbers_l237_237290

theorem find_pairs_of_square_numbers (a b k : ℕ) (hk : k ≥ 2) 
  (h_eq : (a * a + b * b) = k * k * (a * b + 1)) : 
  (a = k ∧ b = k * k * k) ∨ (b = k ∧ a = k * k * k) :=
by
  sorry

end find_pairs_of_square_numbers_l237_237290


namespace work_completion_time_l237_237877

theorem work_completion_time (work_per_day_A : ℚ) (work_per_day_B : ℚ) (work_per_day_C : ℚ) 
(days_A_worked: ℚ) (days_C_worked: ℚ) :
work_per_day_A = 1 / 20 ∧ work_per_day_B = 1 / 30 ∧ work_per_day_C = 1 / 10 ∧
days_A_worked = 2 ∧ days_C_worked = 4  → 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked) +
(1 - 
(work_per_day_A * days_A_worked + work_per_day_B * days_A_worked + work_per_day_C * days_A_worked +
work_per_day_B * (days_C_worked - days_A_worked) + work_per_day_C * (days_C_worked - days_A_worked)))
/ work_per_day_B + days_C_worked) 
= 15 := by
sorry

end work_completion_time_l237_237877


namespace initially_calculated_average_weight_l237_237357

-- Define the conditions
def num_boys : ℕ := 20
def correct_average_weight : ℝ := 58.7
def misread_weight : ℝ := 56
def correct_weight : ℝ := 62
def weight_difference : ℝ := correct_weight - misread_weight

-- State the goal
theorem initially_calculated_average_weight :
  let correct_total_weight := correct_average_weight * num_boys
  let initial_total_weight := correct_total_weight - weight_difference
  let initially_calculated_weight := initial_total_weight / num_boys
  initially_calculated_weight = 58.4 :=
by
  sorry

end initially_calculated_average_weight_l237_237357


namespace find_number_l237_237565

theorem find_number (x n : ℝ) (h1 : x > 0) (h2 : x / 50 + x / n = 0.06 * x) : n = 25 :=
by
  sorry

end find_number_l237_237565


namespace vec_expression_l237_237238

def vec_a : ℝ × ℝ := (1, -2)
def vec_b : ℝ × ℝ := (3, 5)

theorem vec_expression : 2 • vec_a + vec_b = (5, 1) := by
  sorry

end vec_expression_l237_237238


namespace remainder_23_to_2047_mod_17_l237_237337

theorem remainder_23_to_2047_mod_17 :
  23^2047 % 17 = 11 := 
by {
  sorry
}

end remainder_23_to_2047_mod_17_l237_237337


namespace find_tricias_age_l237_237811

variables {Tricia Amilia Yorick Eugene Khloe Rupert Vincent : ℕ}

theorem find_tricias_age 
  (h1 : Tricia = Amilia / 3)
  (h2 : Amilia = Yorick / 4)
  (h3 : Yorick = 2 * Eugene)
  (h4 : Khloe = Eugene / 3)
  (h5 : Rupert = Khloe + 10)
  (h6 : Rupert = Vincent - 2)
  (h7 : Vincent = 22) :
  Tricia = 5 :=
by
  -- skipping the proof using sorry
  sorry

end find_tricias_age_l237_237811


namespace inscribed_circle_radius_l237_237382

theorem inscribed_circle_radius
  (A p s : ℝ) (h1 : A = p) (h2 : s = p / 2) (r : ℝ) (h3 : A = r * s) :
  r = 2 :=
sorry

end inscribed_circle_radius_l237_237382


namespace final_price_is_correct_l237_237705

-- Define the original price and the discount rate
variable (a : ℝ)

-- The final price of the product after two 10% discounts
def final_price_after_discounts (a : ℝ) : ℝ :=
  a * (0.9 ^ 2)

-- Theorem stating the final price after two consecutive 10% discounts
theorem final_price_is_correct (a : ℝ) :
  final_price_after_discounts a = a * (0.9 ^ 2) :=
by sorry

end final_price_is_correct_l237_237705


namespace sum_arithmetic_sequence_max_l237_237548

theorem sum_arithmetic_sequence_max (d : ℝ) (a : ℕ → ℝ) 
  (h1 : d < 0) (h2 : (a 1)^2 = (a 13)^2) :
  ∃ n, n = 6 ∨ n = 7 :=
by
  sorry

end sum_arithmetic_sequence_max_l237_237548


namespace find_sum_abc_l237_237869

noncomputable def f (x a b c : ℝ) : ℝ :=
  x^3 + a * x^2 + b * x + c

theorem find_sum_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (habc_distinct : a ≠ b) (hfa : f a a b c = a^3) (hfb : f b a b c = b^3) : 
  a + b + c = 18 := 
sorry

end find_sum_abc_l237_237869


namespace axis_of_symmetry_sine_function_l237_237593

theorem axis_of_symmetry_sine_function :
  ∃ k : ℤ, x = k * (π / 2) := sorry

end axis_of_symmetry_sine_function_l237_237593


namespace parabola_coefficients_l237_237583

theorem parabola_coefficients (a b c : ℝ) :
  (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ (y = (x + 2)^2 + 5) ∧ y = 9 ↔ x = 0) →
  (a, b, c) = (1, 4, 9) :=
by
  intros h
  sorry

end parabola_coefficients_l237_237583


namespace incorrect_parallel_m_n_l237_237410

variables {l m n : Type} [LinearOrder m] [LinearOrder n] {α β : Type}

-- Assumptions for parallelism and orthogonality
def parallel (x y : Type) : Prop := sorry
def orthogonal (x y : Type) : Prop := sorry

-- Conditions
axiom parallel_m_l : parallel m l
axiom parallel_n_l : parallel n l
axiom orthogonal_m_α : orthogonal m α
axiom parallel_m_β : parallel m β
axiom parallel_m_α : parallel m α
axiom parallel_n_α : parallel n α
axiom orthogonal_m_β : orthogonal m β
axiom orthogonal_α_β : orthogonal α β

-- The theorem to prove
theorem incorrect_parallel_m_n : parallel m α ∧ parallel n α → ¬ parallel m n := sorry

end incorrect_parallel_m_n_l237_237410


namespace avg_ballpoint_pens_per_day_l237_237683

theorem avg_ballpoint_pens_per_day (bundles_sold : ℕ) (pens_per_bundle : ℕ) (days : ℕ) (total_pens : ℕ) (avg_per_day : ℕ) 
  (h1 : bundles_sold = 15)
  (h2 : pens_per_bundle = 40)
  (h3 : days = 5)
  (h4 : total_pens = bundles_sold * pens_per_bundle)
  (h5 : avg_per_day = total_pens / days) :
  avg_per_day = 120 :=
by
  -- placeholder proof
  sorry

end avg_ballpoint_pens_per_day_l237_237683


namespace nonrational_ab_l237_237048

theorem nonrational_ab {a b : ℝ} (ha : a ≠ 0) (hb : b ≠ 0) 
    (h : a^2 * b^2 * (a^2 * b^2 + 4) = 2 * (a^6 + b^6)) : 
    ¬(∃ (p q r s : ℤ), q ≠ 0 ∧ s ≠ 0 ∧ a = p / q ∧ b = r / s) := by
  sorry

end nonrational_ab_l237_237048


namespace calculate_max_marks_l237_237028

theorem calculate_max_marks (shortfall_math : ℕ) (shortfall_science : ℕ) 
                            (shortfall_literature : ℕ) (shortfall_social_studies : ℕ)
                            (required_math : ℕ) (required_science : ℕ)
                            (required_literature : ℕ) (required_social_studies : ℕ)
                            (max_math : ℕ) (max_science : ℕ)
                            (max_literature : ℕ) (max_social_studies : ℕ) :
                            shortfall_math = 40 ∧ required_math = 95 ∧ max_math = 800 ∧
                            shortfall_science = 35 ∧ required_science = 92 ∧ max_science = 438 ∧
                            shortfall_literature = 30 ∧ required_literature = 90 ∧ max_literature = 300 ∧
                            shortfall_social_studies = 25 ∧ required_social_studies = 88 ∧ max_social_studies = 209 :=
by
  sorry

end calculate_max_marks_l237_237028


namespace inequality_holds_l237_237536

theorem inequality_holds (a b c : ℝ) (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) :
  a / (b + c + 1) + b / (c + a + 1) + c / (a + b + 1) + (1 - a) * (1 - b) * (1 - c) ≤ 1 :=
sorry

end inequality_holds_l237_237536


namespace tan_ratio_l237_237004

theorem tan_ratio (α β : ℝ) 
  (h1 : Real.sin (α + β) = (Real.sqrt 3) / 2) 
  (h2 : Real.sin (α - β) = (Real.sqrt 2) / 2) : 
  (Real.tan α) / (Real.tan β) = (5 + 2 * Real.sqrt 6) / (5 - 2 * Real.sqrt 6) :=
by
  sorry

end tan_ratio_l237_237004


namespace sin_inequality_iff_angle_inequality_l237_237740

section
variables {A B : ℝ} {a b : ℝ} (R : ℝ) (hA : A = Real.sin a) (hB : B = Real.sin b)

theorem sin_inequality_iff_angle_inequality (A B : ℝ) :
  (A > B) ↔ (Real.sin A > Real.sin B) :=
sorry
end

end sin_inequality_iff_angle_inequality_l237_237740


namespace no_such_function_exists_l237_237257

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∃ M > 0, ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧
                    (f 1 = 1) ∧
                    (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end no_such_function_exists_l237_237257


namespace chloe_candies_l237_237469

-- Definitions for the conditions
def lindaCandies : ℕ := 34
def totalCandies : ℕ := 62

-- The statement to prove
theorem chloe_candies :
  (totalCandies - lindaCandies) = 28 :=
by
  -- Proof would go here
  sorry

end chloe_candies_l237_237469


namespace time_for_trains_to_clear_l237_237318

noncomputable def train_length_1 : ℕ := 120
noncomputable def train_length_2 : ℕ := 320
noncomputable def train_speed_1_kmph : ℚ := 42
noncomputable def train_speed_2_kmph : ℚ := 30

noncomputable def kmph_to_mps (speed: ℚ) : ℚ := (5/18) * speed

noncomputable def train_speed_1_mps : ℚ := kmph_to_mps train_speed_1_kmph
noncomputable def train_speed_2_mps : ℚ := kmph_to_mps train_speed_2_kmph

noncomputable def total_length : ℕ := train_length_1 + train_length_2
noncomputable def relative_speed : ℚ := train_speed_1_mps + train_speed_2_mps

noncomputable def collision_time : ℚ := total_length / relative_speed

theorem time_for_trains_to_clear : collision_time = 22 := by
  sorry

end time_for_trains_to_clear_l237_237318


namespace rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l237_237550

-- Define the digit constraints and the RD sum function
def is_digit (n : ℕ) : Prop := n < 10
def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

def rd_sum (A B C D : ℕ) : ℕ :=
  let abcd := 1000 * A + 100 * B + 10 * C + D
  let dcba := 1000 * D + 100 * C + 10 * B + A
  abcd + dcba

-- Problem (a)
theorem rd_sum_4281 : rd_sum 4 2 8 1 = 6105 := sorry

-- Problem (b)
theorem rd_sum_formula (A B C D : ℕ) (hA : is_nonzero_digit A) (hD : is_nonzero_digit D) :
  ∃ m n, m = 1001 ∧ n = 110 ∧ rd_sum A B C D = m * (A + D) + n * (B + C) :=
  sorry

-- Problem (c)
theorem rd_sum_count_3883 :
  ∃ n, n = 18 ∧ ∃ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D ∧ rd_sum A B C D = 3883 :=
  sorry

-- Problem (d)
theorem count_self_equal_rd_sum : 
  ∃ n, n = 143 ∧ ∀ (A B C D : ℕ), is_nonzero_digit A ∧ is_digit B ∧ is_digit C ∧ is_nonzero_digit D → (1001 * (A + D) + 110 * (B + C) ≤ 9999 → (1000 * A + 100 * B + 10 * C + D = rd_sum A B C D → 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ D ∧ D ≤ 9)) :=
  sorry

end rd_sum_4281_rd_sum_formula_rd_sum_count_3883_count_self_equal_rd_sum_l237_237550


namespace new_average_income_l237_237182

theorem new_average_income (old_avg_income : ℝ) (num_members : ℕ) (deceased_income : ℝ) 
  (old_avg_income_eq : old_avg_income = 735) (num_members_eq : num_members = 4) 
  (deceased_income_eq : deceased_income = 990) : 
  ((old_avg_income * num_members) - deceased_income) / (num_members - 1) = 650 := 
by sorry

end new_average_income_l237_237182


namespace slope_of_tangent_at_A_l237_237086

def f (x : ℝ) : ℝ := x^2 + 3 * x

def f' (x : ℝ) : ℝ := 2 * x + 3

theorem slope_of_tangent_at_A : f' 2 = 7 := by
  sorry

end slope_of_tangent_at_A_l237_237086


namespace rectangle_perimeter_is_22_l237_237476

-- Definition of sides of the triangle DEF
def side1 : ℕ := 5
def side2 : ℕ := 12
def hypotenuse : ℕ := 13

-- Helper function to compute the area of a right triangle
def triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Ensure the triangle is a right triangle and calculate its area
def area_of_triangle : ℕ :=
  if (side1 * side1 + side2 * side2 = hypotenuse * hypotenuse) then
    triangle_area side1 side2
  else
    0

-- Definition of rectangle's width and equation to find its perimeter
def width : ℕ := 5
def rectangle_length : ℕ := area_of_triangle / width
def perimeter_of_rectangle : ℕ := 2 * (width + rectangle_length)

theorem rectangle_perimeter_is_22 : perimeter_of_rectangle = 22 :=
by
  -- Proof content goes here
  sorry

end rectangle_perimeter_is_22_l237_237476


namespace num_true_propositions_l237_237362

theorem num_true_propositions : 
  (∀ (a b : ℝ), a = 0 → ab = 0) ∧
  (∀ (a b : ℝ), ab ≠ 0 → a ≠ 0) ∧
  ¬ (∀ (a b : ℝ), ab = 0 → a = 0) ∧
  ¬ (∀ (a b : ℝ), a ≠ 0 → ab ≠ 0) → 
  2 = 2 :=
by 
  sorry

end num_true_propositions_l237_237362


namespace sin_identity_l237_237833

theorem sin_identity (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) :
  Real.sin (2 * α + π / 6) = 7 / 8 := 
by
  sorry

end sin_identity_l237_237833


namespace find_alpha_l237_237466

theorem find_alpha (α β : ℝ) (h1 : Real.arctan α = 1/2) (h2 : Real.arctan (α - β) = 1/3)
  (h3 : 0 < α ∧ α < π/2) (h4 : 0 < β ∧ β < π/2) : α = π/4 := by
  sorry

end find_alpha_l237_237466


namespace pq_sum_equals_4_l237_237422

theorem pq_sum_equals_4 (p q : ℝ) (h : (Polynomial.C 1 + Polynomial.C q * Polynomial.X + Polynomial.C p * Polynomial.X^2 + Polynomial.X^4).eval (2 + I) = 0) :
  p + q = 4 :=
sorry

end pq_sum_equals_4_l237_237422


namespace determine_value_of_a_l237_237366

theorem determine_value_of_a (a : ℝ) (h : 1 < a) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ a → 
  1 ≤ (1 / 2 * x^2 - x + 3 / 2) ∧ (1 / 2 * x^2 - x + 3 / 2) ≤ a) →
  a = 3 :=
by
  sorry

end determine_value_of_a_l237_237366


namespace cole_drive_time_l237_237359

noncomputable def time_to_drive_to_work (D : ℝ) : ℝ :=
  D / 50

theorem cole_drive_time (D : ℝ) (h₁ : time_to_drive_to_work D + (D / 110) = 2) : time_to_drive_to_work D * 60 = 82.5 :=
by
  sorry

end cole_drive_time_l237_237359


namespace necessary_and_sufficient_condition_l237_237222

def line1 (a : ℝ) (x y : ℝ) := 2 * x - a * y + 1 = 0
def line2 (a : ℝ) (x y : ℝ) := (a - 1) * x - y + a = 0
def parallel (a : ℝ) : Prop := ∀ x y : ℝ, line1 a x y = line2 a x y

theorem necessary_and_sufficient_condition (a : ℝ) : 
  (a = 2 ↔ parallel a) :=
sorry

end necessary_and_sufficient_condition_l237_237222


namespace part_a_solution_part_b_solution_l237_237134

-- Part (a) Statement in Lean 4
theorem part_a_solution (N : ℕ) (a b : ℕ) (h : N = a * 10^n + b * 10^(n-1)) :
  ∃ (m : ℕ), (N / 10 = m) -> m * 10 = N := sorry

-- Part (b) Statement in Lean 4
theorem part_b_solution (N : ℕ) (a b c : ℕ) (h : N = a * 10^n + b * 10^(n-1) + c * 10^(n-2)) :
  ∃ (m : ℕ), (N / 10^(n-1) = m) -> m * 10^(n-1) = N := sorry

end part_a_solution_part_b_solution_l237_237134


namespace carpet_dimensions_l237_237387
open Real

theorem carpet_dimensions (x y : ℝ) 
  (h1 : x > 0)
  (h2 : y > 0)
  (h3 : ∃ k: ℝ, y = k * x)
  (h4 : ∃ α β: ℝ, α + k * β = 50 ∧ k * α + β = 55)
  (h5 : ∃ γ δ: ℝ, γ + k * δ = 38 ∧ k * γ + δ = 55) :
  x = 25 ∧ y = 50 :=
by sorry

end carpet_dimensions_l237_237387


namespace range_of_a_l237_237481

noncomputable def exists_unique_y (a : ℝ) (x : ℝ) : Prop :=
∃! (y : ℝ), y ∈ Set.Icc (-1) 1 ∧ x + y^2 * Real.exp y = a

theorem range_of_a (e : ℝ) (H_e : e = Real.exp 1) :
  (∀ x ∈ Set.Icc 0 1, exists_unique_y a x) →
  a ∈ Set.Ioc (1 + 1/e) e :=
by
  sorry

end range_of_a_l237_237481


namespace complex_div_eq_i_l237_237673

noncomputable def i := Complex.I

theorem complex_div_eq_i : (1 + i) / (1 - i) = i := 
by
  sorry

end complex_div_eq_i_l237_237673


namespace min_draw_to_ensure_one_red_l237_237278

theorem min_draw_to_ensure_one_red (b y r : ℕ) (h1 : b + y + r = 20) (h2 : b = y / 6) (h3 : r < y) : 
  ∃ n : ℕ, n = 15 ∧ ∀ d : ℕ, d < 15 → ∀ drawn : Finset (ℕ × ℕ × ℕ), drawn.card = d → ∃ card ∈ drawn, card.2 = r := 
sorry

end min_draw_to_ensure_one_red_l237_237278


namespace triple_composition_f_3_l237_237930

def f (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition_f_3 : f (f (f 3)) = 107 :=
by
  sorry

end triple_composition_f_3_l237_237930


namespace verify_mass_percentage_l237_237147

-- Define the elements in HBrO3
def hydrogen : String := "H"
def bromine : String := "Br"
def oxygen : String := "O"

-- Define the given molar masses
def molar_masses (e : String) : Float :=
  if e = hydrogen then 1.01
  else if e = bromine then 79.90
  else if e = oxygen then 16.00
  else 0.0

-- Define the molar mass of HBrO3
def molar_mass_HBrO3 : Float := 128.91

-- Function to calculate mass percentage of a given element in HBrO3
def mass_percentage (e : String) : Float :=
  if e = bromine then 79.90 / molar_mass_HBrO3 * 100
  else if e = hydrogen then 1.01 / molar_mass_HBrO3 * 100
  else if e = oxygen then 48.00 / molar_mass_HBrO3 * 100
  else 0.0

-- The proof problem statement
theorem verify_mass_percentage (e : String) (h : e ∈ [hydrogen, bromine, oxygen]) : mass_percentage e = 0.78 :=
sorry

end verify_mass_percentage_l237_237147


namespace complex_addition_l237_237237

def imag_unit_squared (i : ℂ) : Prop := i * i = -1

theorem complex_addition (a b : ℝ) (i : ℂ)
  (h1 : a + b * i = i * i)
  (h2 : imag_unit_squared i) : a + b = -1 := 
sorry

end complex_addition_l237_237237


namespace rhombus_diagonals_perpendicular_l237_237894

section circumscribed_quadrilateral

variables {a b c d : ℝ}

-- Definition of a tangential quadrilateral satisfying Pitot's theorem.
def tangential_quadrilateral (a b c d : ℝ) :=
  a + c = b + d

-- Defining a rhombus in terms of its sides
def rhombus (a b c d : ℝ) :=
  a = b ∧ b = c ∧ c = d

-- The theorem we want to prove
theorem rhombus_diagonals_perpendicular
  (h : tangential_quadrilateral a b c d)
  (hr : rhombus a b c d) : 
  true := sorry

end circumscribed_quadrilateral

end rhombus_diagonals_perpendicular_l237_237894


namespace find_a_minus_b_l237_237925

theorem find_a_minus_b (a b : ℝ)
  (h1 : 6 = a * 3 + b)
  (h2 : 26 = a * 7 + b) :
  a - b = 14 := 
sorry

end find_a_minus_b_l237_237925


namespace trains_cross_time_l237_237655

theorem trains_cross_time (length : ℝ) (time1 time2 : ℝ) (speed1 speed2 relative_speed : ℝ) 
  (H1 : length = 120) 
  (H2 : time1 = 12) 
  (H3 : time2 = 20) 
  (H4 : speed1 = length / time1) 
  (H5 : speed2 = length / time2) 
  (H6 : relative_speed = speed1 + speed2) 
  (total_distance : ℝ) (H7 : total_distance = length + length) 
  (T : ℝ) (H8 : T = total_distance / relative_speed) :
  T = 15 := 
sorry

end trains_cross_time_l237_237655


namespace sweeties_remainder_l237_237994

theorem sweeties_remainder (m k : ℤ) (h : m = 12 * k + 11) :
  (4 * m) % 12 = 8 :=
by
  -- The proof steps will go here
  sorry

end sweeties_remainder_l237_237994


namespace birch_count_is_87_l237_237274

def num_trees : ℕ := 130
def incorrect_signs (B L : ℕ) : Prop := B + L = num_trees ∧ L + 1 = num_trees - 1 ∧ B = 87

theorem birch_count_is_87 (B L : ℕ) (h1 : B + L = num_trees) (h2 : L + 1 = num_trees - 1) :
  B = 87 :=
sorry

end birch_count_is_87_l237_237274


namespace prime_quadratic_roots_l237_237560

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_integer_roots (a b c : ℤ) : Prop :=
  ∃ x y : ℤ, (a * x * x + b * x + c = 0) ∧ (a * y * y + b * y + c = 0)

theorem prime_quadratic_roots (p : ℕ) (h_prime : is_prime p)
  (h_roots : has_integer_roots 1 (p : ℤ) (-444 * (p : ℤ))) :
  31 < p ∧ p ≤ 41 :=
sorry

end prime_quadratic_roots_l237_237560


namespace num_possibilities_for_asima_integer_l237_237336

theorem num_possibilities_for_asima_integer (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 65) :
  ∃ (n : ℕ), n = 64 :=
by
  sorry

end num_possibilities_for_asima_integer_l237_237336


namespace fencing_required_l237_237625

theorem fencing_required (L W : ℕ) (hL : L = 40) (hA : 40 * W = 680) : 2 * W + L = 74 :=
by sorry

end fencing_required_l237_237625


namespace masha_comb_teeth_count_l237_237918

theorem masha_comb_teeth_count (katya_teeth : ℕ) (masha_to_katya_ratio : ℕ) 
  (katya_teeth_eq : katya_teeth = 11) 
  (masha_to_katya_ratio_eq : masha_to_katya_ratio = 5) : 
  ∃ masha_teeth : ℕ, masha_teeth = 53 :=
by
  have katya_segments := 2 * katya_teeth - 1
  have masha_segments := masha_to_katya_ratio * katya_segments
  let masha_teeth := (masha_segments + 1) / 2
  use masha_teeth
  have masha_teeth_eq := (2 * masha_teeth - 1 = 105)
  sorry

end masha_comb_teeth_count_l237_237918


namespace square_b_perimeter_l237_237899

/-- Square A has an area of 121 square centimeters. Square B has a certain perimeter.
  If square B is placed within square A and a random point is chosen within square A,
  the probability that the point is not within square B is 0.8677685950413223.
  Prove the perimeter of square B is 16 centimeters. -/
theorem square_b_perimeter (area_A : ℝ) (prob : ℝ) (perimeter_B : ℝ) 
  (h1 : area_A = 121)
  (h2 : prob = 0.8677685950413223)
  (h3 : ∃ (a b : ℝ), area_A = a * a ∧ a * a - b * b = prob * area_A) :
  perimeter_B = 16 :=
sorry

end square_b_perimeter_l237_237899


namespace sum_of_coefficients_eq_10_l237_237260

theorem sum_of_coefficients_eq_10 
  (s : ℕ → ℝ) 
  (a b c : ℝ) 
  (h0 : s 0 = 3) 
  (h1 : s 1 = 5) 
  (h2 : s 2 = 9)
  (h : ∀ k ≥ 2, s (k + 1) = a * s k + b * s (k - 1) + c * s (k - 2)) : 
  a + b + c = 10 :=
sorry

end sum_of_coefficients_eq_10_l237_237260


namespace thirteen_coins_value_l237_237286

theorem thirteen_coins_value :
  ∃ (p n d q : ℕ), p + n + d + q = 13 ∧ 
                   1 * p + 5 * n + 10 * d + 25 * q = 141 ∧ 
                   2 ≤ p ∧ 2 ≤ n ∧ 2 ≤ d ∧ 2 ≤ q ∧ 
                   d = 3 :=
  sorry

end thirteen_coins_value_l237_237286


namespace problem_complement_intersection_l237_237633

open Set

-- Define the universal set U
def U : Set ℕ := {0, 2, 4, 6, 8, 10}

-- Define set A
def A : Set ℕ := {0, 2, 4, 6}

-- Define set B based on A
def B : Set ℕ := {x | x ∈ A ∧ x < 4}

-- Define the complement of set A within U
def complement_A_U : Set ℕ := U \ A

-- Define the complement of set B within U
def complement_B_U : Set ℕ := U \ B

-- Prove the given equations
theorem problem_complement_intersection :
  (complement_A_U = {8, 10}) ∧ (A ∩ complement_B_U = {4, 6}) := 
by
  sorry

end problem_complement_intersection_l237_237633


namespace geo_seq_condition_l237_237830

-- Definitions based on conditions
variable (a b c : ℝ)

-- Condition of forming a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop :=
  ∃ r : ℝ, -1 * r = a ∧ a * r = b ∧ b * r = c ∧ c * r = -9

-- Proof problem statement
theorem geo_seq_condition (h : geometric_sequence a b c) : b = -3 ∧ a * c = 9 :=
sorry

end geo_seq_condition_l237_237830


namespace sequence_integers_l237_237330

theorem sequence_integers (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) 
  (h3 : ∀ n ≥ 3, a n = (a (n - 1))^2 + 2 / a (n - 2)) : ∀ n, ∃ k : ℤ, a n = k :=
sorry

end sequence_integers_l237_237330


namespace toys_produced_in_week_l237_237922

-- Define the number of working days in a week
def working_days_in_week : ℕ := 4

-- Define the number of toys produced per day
def toys_produced_per_day : ℕ := 1375

-- The statement to be proved
theorem toys_produced_in_week :
  working_days_in_week * toys_produced_per_day = 5500 :=
by
  sorry

end toys_produced_in_week_l237_237922


namespace incorrect_statement_A_l237_237473

-- We need to prove that statement (A) is incorrect given the provided conditions.

theorem incorrect_statement_A :
  ¬(∀ (a b : ℝ), a > b → ∀ (c : ℝ), c < 0 → a * c > b * c ∧ a / c > b / c) := 
sorry

end incorrect_statement_A_l237_237473


namespace expression1_expression2_expression3_expression4_l237_237647

theorem expression1 : 12 - (-10) + 7 = 29 := 
by
  sorry

theorem expression2 : 1 + (-2) * abs (-2 - 3) - 5 = -14 :=
by
  sorry

theorem expression3 : (-8 * (-1 / 6 + 3 / 4 - 1 / 12)) / (1 / 6) = -24 :=
by
  sorry

theorem expression4 : -1 ^ 2 - (2 - (-2) ^ 3) / (-2 / 5) * (5 / 2) = 123 / 2 := 
by
  sorry

end expression1_expression2_expression3_expression4_l237_237647


namespace sqrt_two_irrational_l237_237314

theorem sqrt_two_irrational :
  ¬ ∃ (p q : ℕ), p ≠ 0 ∧ q ≠ 0 ∧ gcd p q = 1 ∧ (↑q / ↑p) ^ 2 = (2:ℝ) :=
sorry

end sqrt_two_irrational_l237_237314


namespace complete_square_transform_l237_237507

theorem complete_square_transform (x : ℝ) :
  x^2 - 8 * x + 2 = 0 → (x - 4)^2 = 14 :=
by
  intro h
  sorry

end complete_square_transform_l237_237507


namespace simplify_expr_C_l237_237217

theorem simplify_expr_C (x y : ℝ) : 5 * x - (x - 2 * y) = 4 * x + 2 * y :=
by
  sorry

end simplify_expr_C_l237_237217


namespace Emily_candies_l237_237923

theorem Emily_candies (jennifer_candies emily_candies bob_candies : ℕ) 
    (h1: jennifer_candies = 2 * emily_candies)
    (h2: jennifer_candies = 3 * bob_candies)
    (h3: bob_candies = 4) : emily_candies = 6 :=
by
  -- Proof to be provided
  sorry

end Emily_candies_l237_237923


namespace secondTrain_speed_l237_237106

/-
Conditions:
1. Two trains start from A and B and travel towards each other.
2. The distance between them is 1100 km.
3. At the time of their meeting, one train has traveled 100 km more than the other.
4. The first train's speed is 50 kmph.
-/

-- Let v be the speed of the second train
def secondTrainSpeed (v : ℝ) : Prop :=
  ∃ d : ℝ, 
    d > 0 ∧
    v > 0 ∧
    (d + (d - 100) = 1100) ∧
    ((d / 50) = ((d - 100) / v))

-- Here is the main theorem translating the problem statement:
theorem secondTrain_speed :
  secondTrainSpeed (250 / 6) :=
by
  sorry

end secondTrain_speed_l237_237106


namespace right_triangle_5_12_13_l237_237289

theorem right_triangle_5_12_13 (a b c : ℕ) (h1 : a = 5) (h2 : b = 12) (h3 : c = 13) : a^2 + b^2 = c^2 := 
by 
   sorry

end right_triangle_5_12_13_l237_237289


namespace total_volume_of_quiche_l237_237525

def raw_spinach_volume : ℝ := 40
def cooked_volume_percentage : ℝ := 0.20
def cream_cheese_volume : ℝ := 6
def eggs_volume : ℝ := 4

theorem total_volume_of_quiche :
  raw_spinach_volume * cooked_volume_percentage + cream_cheese_volume + eggs_volume = 18 := by
  sorry

end total_volume_of_quiche_l237_237525


namespace total_splash_width_l237_237752

def pebbles : ℚ := 1/5
def rocks : ℚ := 2/5
def boulders : ℚ := 7/5
def mini_boulders : ℚ := 4/5
def large_pebbles : ℚ := 3/5

def num_pebbles : ℚ := 10
def num_rocks : ℚ := 5
def num_boulders : ℚ := 4
def num_mini_boulders : ℚ := 3
def num_large_pebbles : ℚ := 7

theorem total_splash_width : 
  num_pebbles * pebbles + 
  num_rocks * rocks + 
  num_boulders * boulders + 
  num_mini_boulders * mini_boulders + 
  num_large_pebbles * large_pebbles = 16.2 := by
  sorry

end total_splash_width_l237_237752


namespace correct_calculation_l237_237374

-- Definition of the conditions
def condition1 (a : ℕ) : Prop := a^2 * a^3 = a^6
def condition2 (a : ℕ) : Prop := (a^2)^10 = a^20
def condition3 (a : ℕ) : Prop := (2 * a) * (3 * a) = 6 * a
def condition4 (a : ℕ) : Prop := a^12 / a^2 = a^6

-- The main theorem to state that condition2 is the correct calculation
theorem correct_calculation (a : ℕ) : condition2 a :=
sorry

end correct_calculation_l237_237374


namespace one_twentieth_of_eighty_l237_237562

/--
Given the conditions, to prove that \(\frac{1}{20}\) of 80 is equal to 4.
-/
theorem one_twentieth_of_eighty : (80 : ℚ) * (1 / 20) = 4 :=
by
  sorry

end one_twentieth_of_eighty_l237_237562


namespace odd_function_neg_expression_l237_237577

theorem odd_function_neg_expression (f : ℝ → ℝ) (h₀ : ∀ x > 0, f x = x^3 + x + 1)
    (h₁ : ∀ x, f (-x) = -f x) : ∀ x < 0, f x = x^3 + x - 1 :=
by
  sorry

end odd_function_neg_expression_l237_237577


namespace no_nonzero_integer_solution_l237_237733

theorem no_nonzero_integer_solution (m n p : ℤ) :
  (m + n * Real.sqrt 2 + p * Real.sqrt 3 = 0) → (m = 0 ∧ n = 0 ∧ p = 0) :=
by sorry

end no_nonzero_integer_solution_l237_237733


namespace mean_home_runs_correct_l237_237326

-- Define the total home runs in April
def total_home_runs_April : ℕ := 5 * 4 + 6 * 4 + 8 * 2 + 10

-- Define the total home runs in May
def total_home_runs_May : ℕ := 5 * 2 + 6 * 2 + 8 * 3 + 10 * 2 + 11

-- Define the total number of top hitters/players
def total_players : ℕ := 12

-- Define the total home runs over two months
def total_home_runs : ℕ := total_home_runs_April + total_home_runs_May

-- Calculate the mean number of home runs
def mean_home_runs : ℚ := total_home_runs / total_players

-- Prove that the calculated mean is equal to the expected result
theorem mean_home_runs_correct : mean_home_runs = 12.08 := by
  sorry

end mean_home_runs_correct_l237_237326


namespace poly_eq_zero_or_one_l237_237183

noncomputable def k : ℝ := 2 -- You can replace 2 with any number greater than 1.

theorem poly_eq_zero_or_one (P : ℝ → ℝ) 
  (h1 : k > 1) 
  (h2 : ∀ x : ℝ, P (x ^ k) = (P x) ^ k) : 
  (∀ x, P x = 0) ∨ (∀ x, P x = 1) :=
sorry

end poly_eq_zero_or_one_l237_237183


namespace four_inv_mod_35_l237_237902

theorem four_inv_mod_35 : ∃ x : ℕ, 4 * x ≡ 1 [MOD 35] ∧ x = 9 := 
by 
  use 9
  sorry

end four_inv_mod_35_l237_237902


namespace tan_add_pi_over_3_l237_237195

theorem tan_add_pi_over_3 (x : ℝ) (h : Real.tan x = Real.sqrt 3) :
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end tan_add_pi_over_3_l237_237195


namespace tennis_tournament_total_rounds_l237_237368

theorem tennis_tournament_total_rounds
  (participants : ℕ)
  (points_win : ℕ)
  (points_loss : ℕ)
  (pairs_formation : ℕ → ℕ)
  (single_points_award : ℕ → ℕ)
  (elimination_condition : ℕ → Prop)
  (tournament_continues : ℕ → Prop)
  (progression_condition : ℕ → ℕ → ℕ)
  (group_split : Π (n : ℕ), Π (k : ℕ), (ℕ × ℕ))
  (rounds_needed : ℕ) :
  participants = 1152 →
  points_win = 1 →
  points_loss = 0 →
  pairs_formation participants ≥ 0 →
  single_points_award participants ≥ 0 →
  (∀ p, p > 1 → participants / p > 0 → tournament_continues participants) →
  (∀ m n, progression_condition m n = n - m) →
  (group_split 1152 1024 = (1024, 128)) →
  rounds_needed = 14 :=
by
  sorry

end tennis_tournament_total_rounds_l237_237368


namespace juice_drinks_costs_2_l237_237978

-- Define the conditions and the proof problem
theorem juice_drinks_costs_2 (given_amount : ℕ) (amount_returned : ℕ) 
                            (pizza_cost : ℕ) (number_of_pizzas : ℕ) 
                            (number_of_juice_packs : ℕ) 
                            (total_spent_on_juice : ℕ) (cost_per_pack : ℕ) 
                            (h1 : given_amount = 50) (h2 : amount_returned = 22)
                            (h3 : pizza_cost = 12) (h4 : number_of_pizzas = 2)
                            (h5 : number_of_juice_packs = 2) 
                            (h6 : given_amount - amount_returned - number_of_pizzas * pizza_cost = total_spent_on_juice) 
                            (h7 : total_spent_on_juice / number_of_juice_packs = cost_per_pack) : 
                            cost_per_pack = 2 := by
  sorry

end juice_drinks_costs_2_l237_237978


namespace find_ab_for_equation_l237_237909

theorem find_ab_for_equation (a b : ℝ) :
  (∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (∃ x, x = 12 - x1 - x2) ∧ (a * x1^2 - 24 * x1 + b) / (x1^2 - 1) = x1
  ∧ (a * x2^2 - 24 * x2 + b) / (x2^2 - 1) = x2) ∧ (a = 11 ∧ b = -35) ∨ (a = 35 ∧ b = -5819) := sorry

end find_ab_for_equation_l237_237909


namespace value_divided_by_3_l237_237540

-- Given condition
def given_condition (x : ℕ) : Prop := x - 39 = 54

-- Correct answer we need to prove
theorem value_divided_by_3 (x : ℕ) (h : given_condition x) : x / 3 = 31 := 
by
  sorry

end value_divided_by_3_l237_237540


namespace marbles_problem_a_marbles_problem_b_l237_237809

-- Define the problem as Lean statements.

-- Part (a): m = 2004, n = 2006
theorem marbles_problem_a (m n : ℕ) (h_m : m = 2004) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) := 
sorry

-- Part (b): m = 2005, n = 2006
theorem marbles_problem_b (m n : ℕ) (h_m : m = 2005) (h_n : n = 2006) :
  ∃ (marbles : ℕ → ℕ → ℕ), 
  (∀ i j, 1 ≤ i ∧ i ≤ m ∧ 1 ≤ j ∧ j ≤ n → marbles i j = 1) → false := 
sorry

end marbles_problem_a_marbles_problem_b_l237_237809


namespace inequality_solution_set_range_of_k_l237_237242

variable {k m x : ℝ}

theorem inequality_solution_set (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k)) 
  (sol_set_f_x_gt_m : ∀ x, f x > m ↔ (x < -3 ∨ x > -2)) :
  -1 < x ∧ x < 3 / 2 := 
sorry

theorem range_of_k (k_pos : k > 0) 
  (f : ℝ → ℝ) (hf : ∀ x, f x = k * x / (x^2 + 3 * k))
  (exists_f_x_gt_1 : ∃ x > 3, f x > 1) : 
  k > 12 :=
sorry

end inequality_solution_set_range_of_k_l237_237242


namespace parallel_lines_m_eq_one_l237_237575

theorem parallel_lines_m_eq_one (m : ℝ) :
  (∀ x y : ℝ, 2 * x + m * y + 8 = 0 ∧ (m + 1) * x + y + (m - 2) = 0 → m = 1) :=
by
  intro x y h
  let L1_slope := -2 / m
  let L2_slope := -(m + 1)
  have h_slope : L1_slope = L2_slope := sorry
  have m_positive : m = 1 := sorry
  exact m_positive

end parallel_lines_m_eq_one_l237_237575


namespace sectionB_seats_correct_l237_237653

-- Definitions for the number of seats in Section A
def seatsA_subsection1 : Nat := 60
def seatsA_subsection2 : Nat := 3 * 80
def totalSeatsA : Nat := seatsA_subsection1 + seatsA_subsection2

-- Condition for the number of seats in Section B
def seatsB : Nat := 3 * totalSeatsA + 20

-- Theorem statement to prove the number of seats in Section B
theorem sectionB_seats_correct : seatsB = 920 := by
  sorry

end sectionB_seats_correct_l237_237653


namespace remaining_budget_for_public_spaces_l237_237197

noncomputable def total_budget : ℝ := 32
noncomputable def policing_budget : ℝ := total_budget / 2
noncomputable def education_budget : ℝ := 12
noncomputable def remaining_budget : ℝ := total_budget - (policing_budget + education_budget)

theorem remaining_budget_for_public_spaces : remaining_budget = 4 :=
by
  -- Proof is skipped
  sorry

end remaining_budget_for_public_spaces_l237_237197


namespace fred_balloons_l237_237221

theorem fred_balloons (T S D F : ℕ) (hT : T = 72) (hS : S = 46) (hD : D = 16) (hTotal : T = F + S + D) : F = 10 := 
by
  sorry

end fred_balloons_l237_237221


namespace countDistinguishedDigitsTheorem_l237_237993

-- Define a function to count numbers with four distinct digits where leading zeros are allowed
def countDistinguishedDigits : Nat :=
  10 * 9 * 8 * 7

-- State the theorem we need to prove
theorem countDistinguishedDigitsTheorem :
  countDistinguishedDigits = 5040 := 
by
  sorry

end countDistinguishedDigitsTheorem_l237_237993


namespace complement_intersection_l237_237608

def A : Set ℝ := {x | x^2 - 5 * x - 6 ≤ 0}
def B : Set ℝ := {x | x > 7}

theorem complement_intersection :
  (Set.univ \ A) ∩ B = {x | x > 7} :=
by
  sorry

end complement_intersection_l237_237608


namespace hannah_remaining_money_l237_237931

-- Define the conditions of the problem
def initial_amount : Nat := 120
def rides_cost : Nat := initial_amount * 40 / 100
def games_cost : Nat := initial_amount * 15 / 100
def remaining_after_rides_games : Nat := initial_amount - rides_cost - games_cost

def dessert_cost : Nat := 8
def cotton_candy_cost : Nat := 5
def hotdog_cost : Nat := 6
def keychain_cost : Nat := 7
def poster_cost : Nat := 10
def additional_attraction_cost : Nat := 15
def total_food_souvenirs_cost : Nat := dessert_cost + cotton_candy_cost + hotdog_cost + keychain_cost + poster_cost + additional_attraction_cost

def final_remaining_amount : Nat := remaining_after_rides_games - total_food_souvenirs_cost

-- Formulate the theorem to prove
theorem hannah_remaining_money : final_remaining_amount = 3 := by
  sorry

end hannah_remaining_money_l237_237931


namespace salary_increase_l237_237323

variable (S : ℝ) (P : ℝ)

theorem salary_increase (h1 : 1.16 * S = 406) (h2 : 350 + 350 * P = 420) : P * 100 = 20 := 
by
  sorry

end salary_increase_l237_237323


namespace prob_score_3_points_l237_237123

-- Definitions for the probabilities
def probability_hit_A := 3/4
def score_hit_A := 1
def score_miss_A := -1

def probability_hit_B := 2/3
def score_hit_B := 2
def score_miss_B := 0

-- Conditional probabilities and their calculations
noncomputable def prob_scenario_1 : ℚ := 
  probability_hit_A * 2 * probability_hit_B * (1 - probability_hit_B)

noncomputable def prob_scenario_2 : ℚ := 
  (1 - probability_hit_A) * probability_hit_B^2

noncomputable def total_prob : ℚ := 
  prob_scenario_1 + prob_scenario_2

-- The final proof statement
theorem prob_score_3_points : total_prob = 4/9 := sorry

end prob_score_3_points_l237_237123


namespace product_of_all_n_satisfying_quadratic_l237_237402

theorem product_of_all_n_satisfying_quadratic :
  (∃ n : ℕ, n^2 - 40 * n + 399 = 3) ∧
  (∀ p : ℕ, Prime p → ((∃ n : ℕ, n^2 - 40 * n + 399 = p) → p = 3)) →
  ∃ n1 n2 : ℕ, (n1^2 - 40 * n1 + 399 = 3) ∧ (n2^2 - 40 * n2 + 399 = 3) ∧ n1 ≠ n2 ∧ (n1 * n2 = 396) :=
by
  sorry

end product_of_all_n_satisfying_quadratic_l237_237402


namespace reciprocal_neg_one_over_2023_l237_237553

theorem reciprocal_neg_one_over_2023 : 1 / (- (1 / 2023 : ℝ)) = -2023 :=
by
  sorry

end reciprocal_neg_one_over_2023_l237_237553


namespace percentage_calculation_l237_237166

theorem percentage_calculation (P : ℝ) : 
    (P / 100) * 24 + 0.10 * 40 = 5.92 ↔ P = 8 :=
by 
    sorry

end percentage_calculation_l237_237166


namespace length_other_diagonal_l237_237151

variables (d1 d2 : ℝ) (Area : ℝ)

theorem length_other_diagonal 
  (h1 : Area = 432)
  (h2 : d1 = 36) :
  d2 = 24 :=
by
  -- Insert proof here
  sorry

end length_other_diagonal_l237_237151


namespace find_a_l237_237504

theorem find_a (a : ℝ) (h_pos : 0 < a) 
(h : a + a^2 = 6) : a = 2 :=
sorry

end find_a_l237_237504


namespace floor_equality_iff_l237_237865

variable (x : ℝ)

theorem floor_equality_iff :
  (⌊3 * x + 4⌋ = ⌊5 * x - 1⌋) ↔
  (11 / 5 ≤ x ∧ x < 7 / 3) ∨
  (12 / 5 ≤ x ∧ x < 13 / 5) ∨
  (17 / 5 ≤ x ∧ x < 18 / 5) := by
  sorry

end floor_equality_iff_l237_237865


namespace rectangle_perimeter_is_28_l237_237769

-- Define the variables and conditions
variables (h w : ℝ)

-- Problem conditions
def rectangle_area (h w : ℝ) : Prop := h * w = 40
def width_greater_than_twice_height (h w : ℝ) : Prop := w > 2 * h
def parallelogram_area (h w : ℝ) : Prop := h * (w - h) = 24

-- The theorem stating the perimeter of the rectangle given the conditions
theorem rectangle_perimeter_is_28 (h w : ℝ) 
  (H1 : rectangle_area h w) 
  (H2 : width_greater_than_twice_height h w) 
  (H3 : parallelogram_area h w) :
  2 * h + 2 * w = 28 :=
sorry

end rectangle_perimeter_is_28_l237_237769


namespace bicycle_speed_l237_237908

theorem bicycle_speed (d1 d2 v1 v_avg : ℝ)
  (h1 : d1 = 300) 
  (h2 : d1 + d2 = 450) 
  (h3 : v1 = 20) 
  (h4 : v_avg = 18) : 
  (d2 / ((d1 / v1) + d2 / (d2 * v_avg / 450)) = 15) :=
by 
  sorry

end bicycle_speed_l237_237908


namespace radius_of_circle_is_ten_l237_237013

noncomputable def radius_of_circle (diameter : ℝ) : ℝ :=
  diameter / 2

theorem radius_of_circle_is_ten :
  radius_of_circle 20 = 10 :=
by
  unfold radius_of_circle
  sorry

end radius_of_circle_is_ten_l237_237013


namespace lucas_payment_l237_237383

noncomputable def payment (windows_per_floor : ℕ) (floors : ℕ) (days : ℕ) 
  (earn_per_window : ℝ) (delay_penalty : ℝ) (period : ℕ) : ℝ :=
  let total_windows := windows_per_floor * floors
  let earnings := total_windows * earn_per_window
  let penalty_periods := days / period
  let total_penalty := penalty_periods * delay_penalty
  earnings - total_penalty

theorem lucas_payment :
  payment 3 3 6 2 1 3 = 16 := by
  sorry

end lucas_payment_l237_237383


namespace find_perpendicular_line_through_intersection_l237_237983

theorem find_perpendicular_line_through_intersection : 
  (∃ (M : ℚ × ℚ), 
    (M.1 - 2 * M.2 + 3 = 0) ∧ 
    (2 * M.1 + 3 * M.2 - 8 = 0) ∧ 
    (∃ (c : ℚ), M.1 + 3 * M.2 + c = 0 ∧ 3 * M.1 - M.2 + 1 = 0)) → 
  ∃ (c : ℚ), x + 3 * y + c = 0 :=
sorry

end find_perpendicular_line_through_intersection_l237_237983


namespace correct_answers_unanswered_minimum_correct_answers_l237_237254

-- Definition of the conditions in the problem
def total_questions := 25
def unanswered_questions := 1
def correct_points := 4
def wrong_points := -1
def total_score_1 := 86
def total_score_2 := 90

-- Part 1: Define the conditions and prove that x = 22
theorem correct_answers_unanswered (x : ℕ) (h1 : total_questions - unanswered_questions = 24)
  (h2 : 4 * x + wrong_points * (total_questions - unanswered_questions - x) = total_score_1) : x = 22 :=
sorry

-- Part 2: Define the conditions and prove that at least 23 correct answers are needed
theorem minimum_correct_answers (a : ℕ)
  (h3 : correct_points * a + wrong_points * (total_questions - a) ≥ total_score_2) : a ≥ 23 :=
sorry

end correct_answers_unanswered_minimum_correct_answers_l237_237254


namespace pascal_row_with_ratio_456_exists_at_98_l237_237632

theorem pascal_row_with_ratio_456_exists_at_98 :
  ∃ n, ∃ r, 0 ≤ r ∧ r + 2 ≤ n ∧ 
  ((Nat.choose n r : ℚ) / Nat.choose n (r + 1) = 4 / 5) ∧
  ((Nat.choose n (r + 1) : ℚ) / Nat.choose n (r + 2) = 5 / 6) ∧ 
  n = 98 := by
  sorry

end pascal_row_with_ratio_456_exists_at_98_l237_237632


namespace fraction_reducible_l237_237514

theorem fraction_reducible (l : ℤ) : ∃ d : ℤ, d ≠ 1 ∧ d > 0 ∧ d = gcd (5 * l + 6) (8 * l + 7) := by 
  use 13
  sorry

end fraction_reducible_l237_237514


namespace workers_contribution_l237_237229

theorem workers_contribution (W C : ℕ) 
    (h1 : W * C = 300000) 
    (h2 : W * (C + 50) = 325000) : 
    W = 500 :=
by
    sorry

end workers_contribution_l237_237229


namespace derivative_at_1_l237_237589

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem derivative_at_1 : (deriv f 1) = 2 * Real.exp 1 := by
  sorry

end derivative_at_1_l237_237589


namespace grid_area_l237_237707

-- Definitions based on problem conditions
def num_lines : ℕ := 36
def perimeter : ℕ := 72
def side_length : ℕ := perimeter / num_lines

-- Problem statement
theorem grid_area (h : num_lines = 36) (p : perimeter = 72)
  (s : side_length = 2) :
  let n_squares := (8 - 1) * (4 - 1)
  let area_square := side_length ^ 2
  let total_area := n_squares * area_square
  total_area = 84 :=
by {
  -- Skipping proof
  sorry
}

end grid_area_l237_237707


namespace minimum_15_equal_differences_l237_237766

-- Definition of distinct integers a_i
def distinct_sequence (a : Fin 100 → ℕ) : Prop :=
  ∀ i j : Fin 100, i < j → a i < a j

-- Definition of the differences d_i
def differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) : Prop :=
  ∀ i : Fin 99, d i = a ⟨i + 1, Nat.lt_of_lt_of_le (Nat.succ_lt_succ i.2) (by norm_num)⟩ - a i

-- Main theorem statement
theorem minimum_15_equal_differences (a : Fin 100 → ℕ) (d : Fin 99 → ℕ) :
  (∀ i : Fin 100, 1 ≤ a i ∧ a i ≤ 400) →
  distinct_sequence a →
  differences a d →
  ∃ t : Finset ℕ, t.card ≥ 15 ∧ ∀ x : ℕ, x ∈ t → (∃ i j : Fin 99, i ≠ j ∧ d i = x ∧ d j = x) :=
sorry

end minimum_15_equal_differences_l237_237766


namespace log_xy_l237_237136

-- Definitions from conditions
def log (z : ℝ) : ℝ := sorry -- Assume a definition of log function
variables (x y : ℝ)
axiom h1 : log (x^2 * y^2) = 1
axiom h2 : log (x^3 * y) = 2

-- The proof goal
theorem log_xy (x y : ℝ) (h1 : log (x^2 * y^2) = 1) (h2 : log (x^3 * y) = 2) : log (x * y) = 1/2 :=
sorry

end log_xy_l237_237136


namespace actors_in_one_hour_l237_237369

theorem actors_in_one_hour (actors_per_set : ℕ) (minutes_per_set : ℕ) (total_minutes : ℕ) :
  actors_per_set = 5 → minutes_per_set = 15 → total_minutes = 60 →
  (total_minutes / minutes_per_set) * actors_per_set = 20 :=
by
  intros h1 h2 h3
  sorry

end actors_in_one_hour_l237_237369


namespace probability_at_least_one_six_l237_237849

theorem probability_at_least_one_six (h: ℚ) : h = 91 / 216 :=
by 
  sorry

end probability_at_least_one_six_l237_237849


namespace arithmetic_mean_of_4_and_16_l237_237686

-- Define the arithmetic mean condition
def is_arithmetic_mean (a b x : ℝ) : Prop :=
  x = (a + b) / 2

-- Theorem to prove that x = 10 if it is the mean of 4 and 16
theorem arithmetic_mean_of_4_and_16 (x : ℝ) (h : is_arithmetic_mean 4 16 x) : x = 10 :=
by
  sorry

end arithmetic_mean_of_4_and_16_l237_237686


namespace not_perfect_square_l237_237350

theorem not_perfect_square (n : ℤ) (hn : n > 4) : ¬ (∃ k : ℕ, n^2 - 3*n = k^2) :=
sorry

end not_perfect_square_l237_237350


namespace foldable_topless_cubical_box_count_l237_237073

def isFoldable (placement : Char) : Bool :=
  placement = 'C' ∨ placement = 'E' ∨ placement = 'G'

theorem foldable_topless_cubical_box_count :
  (List.filter isFoldable ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']).length = 3 :=
by
  sorry

end foldable_topless_cubical_box_count_l237_237073


namespace polynomial_equality_l237_237793

theorem polynomial_equality :
  (3 * x + 1) ^ 4 = a * x ^ 4 + b * x ^ 3 + c * x ^ 2 + d * x + e →
  a - b + c - d + e = 16 :=
by
  intro h
  sorry

end polynomial_equality_l237_237793


namespace original_price_of_wand_l237_237860

theorem original_price_of_wand (x : ℝ) (h : x / 8 = 12) : x = 96 :=
by
  sorry

end original_price_of_wand_l237_237860


namespace locus_centers_of_circles_l237_237544

theorem locus_centers_of_circles (P : ℝ × ℝ) (a : ℝ) (a_pos : 0 < a):
  {O : ℝ × ℝ | dist O P = a} = {O : ℝ × ℝ | dist O P = a} :=
by
  sorry

end locus_centers_of_circles_l237_237544


namespace triangle_angle_equality_l237_237373

theorem triangle_angle_equality (A B C : ℝ) (h : ∃ (x : ℝ), x^2 - x * (Real.cos A * Real.cos B) - Real.cos (C / 2)^2 = 0 ∧ x = 1) : A = B :=
by {
  sorry
}

end triangle_angle_equality_l237_237373


namespace sum_of_coefficients_is_neg40_l237_237020

noncomputable def p (x : ℝ) : ℝ := 3 * (x^8 - x^5 + 2 * x^3 - 6) - 5 * (x^4 + 3 * x^2) + 2 * (x^6 - 5)

theorem sum_of_coefficients_is_neg40 : p 1 = -40 := by
  sorry

end sum_of_coefficients_is_neg40_l237_237020


namespace clock_hand_swap_times_l237_237472

noncomputable def time_between_2_and_3 : ℚ := (2 * 143 + 370) / 143
noncomputable def time_between_6_and_7 : ℚ := (6 * 143 + 84) / 143

theorem clock_hand_swap_times :
  time_between_2_and_3 = 2 + 31 * 7 / 143 ∧
  time_between_6_and_7 = 6 + 12 * 84 / 143 :=
by
  -- Math proof will go here
  sorry

end clock_hand_swap_times_l237_237472


namespace quadratic_inequality_solution_set_l237_237571

theorem quadratic_inequality_solution_set (a b c : ℝ) (h1 : a < 0)
  (h2 : -1 + 2 = b / a) (h3 : -1 * 2 = c / a) :
  (b = a) ∧ (c = -2 * a) :=
by
  sorry

end quadratic_inequality_solution_set_l237_237571


namespace find_integers_with_sum_and_gcd_l237_237770

theorem find_integers_with_sum_and_gcd {a b : ℕ} (h_sum : a + b = 104055) (h_gcd : Nat.gcd a b = 6937) :
  (a = 6937 ∧ b = 79118) ∨ (a = 13874 ∧ b = 90181) ∨ (a = 27748 ∧ b = 76307) ∨ (a = 48559 ∧ b = 55496) :=
sorry

end find_integers_with_sum_and_gcd_l237_237770


namespace geometric_sequence_sum_terms_l237_237919

noncomputable def geometric_sequence (a_1 : ℕ) (q : ℕ) (n : ℕ) : ℕ :=
  a_1 * q ^ (n - 1)

theorem geometric_sequence_sum_terms :
  ∀ (a_1 q : ℕ), a_1 = 3 → 
  (geometric_sequence 3 q 1 + geometric_sequence 3 q 2 + geometric_sequence 3 q 3 = 21) →
  (q > 0) →
  (geometric_sequence 3 q 3 + geometric_sequence 3 q 4 + geometric_sequence 3 q 5 = 84) :=
by
  intros a_1 q h1 hsum hqpos
  sorry

end geometric_sequence_sum_terms_l237_237919


namespace sarah_initial_money_l237_237256

def initial_money 
  (cost_toy_car : ℕ)
  (cost_scarf : ℕ)
  (cost_beanie : ℕ)
  (remaining_money : ℕ)
  (number_of_toy_cars : ℕ) : ℕ :=
  remaining_money + cost_beanie + cost_scarf + number_of_toy_cars * cost_toy_car

theorem sarah_initial_money : 
  (initial_money 11 10 14 7 2) = 53 :=
by
  rfl 

end sarah_initial_money_l237_237256


namespace cost_per_spool_l237_237745

theorem cost_per_spool
  (p : ℕ) (f : ℕ) (y : ℕ) (t : ℕ) (n : ℕ)
  (hp : p = 15) (hf : f = 24) (hy : y = 5) (ht : t = 141) (hn : n = 2) :
  (t - (p + y * f)) / n = 3 :=
by sorry

end cost_per_spool_l237_237745


namespace range_of_m_l237_237275

def has_solution_in_interval (m : ℝ) : Prop :=
  ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), x^2 - 2 * x - 1 + m ≤ 0 

theorem range_of_m (m : ℝ) : has_solution_in_interval m ↔ m ≤ 2 := by 
  sorry

end range_of_m_l237_237275


namespace second_machine_finishes_in_10_minutes_l237_237736

-- Definitions for the conditions:
def time_to_clear_by_first_machine (t : ℝ) : Prop := t = 1
def time_to_clear_by_second_machine (t : ℝ) : Prop := t = 3 / 4
def time_first_machine_works (t : ℝ) : Prop := t = 1 / 3
def remaining_time (t : ℝ) : Prop := t = 1 / 6

-- Theorem statement:
theorem second_machine_finishes_in_10_minutes (t₁ t₂ t₃ t₄ : ℝ) 
  (h₁ : time_to_clear_by_first_machine t₁) 
  (h₂ : time_to_clear_by_second_machine t₂) 
  (h₃ : time_first_machine_works t₃) 
  (h₄ : remaining_time t₄) 
  : t₄ = 1 / 6 → t₄ * 60 = 10 := 
by
  -- here we can provide the proof steps, but the task does not require the proof
  sorry

end second_machine_finishes_in_10_minutes_l237_237736


namespace alex_hours_per_week_l237_237347

theorem alex_hours_per_week
  (summer_earnings : ℕ)
  (summer_weeks : ℕ)
  (summer_hours_per_week : ℕ)
  (academic_year_weeks : ℕ)
  (academic_year_earnings : ℕ)
  (same_hourly_rate : Prop) :
  summer_earnings = 4000 →
  summer_weeks = 8 →
  summer_hours_per_week = 40 →
  academic_year_weeks = 32 →
  academic_year_earnings = 8000 →
  same_hourly_rate →
  (academic_year_earnings / ((summer_earnings : ℚ) / (summer_weeks * summer_hours_per_week)) / academic_year_weeks) = 20 :=
by
  sorry

end alex_hours_per_week_l237_237347


namespace num_divisors_of_30_l237_237538

theorem num_divisors_of_30 : 
  (∀ n : ℕ, n > 0 → (30 = 2^1 * 3^1 * 5^1) → (∀ k : ℕ, 0 < k ∧ k ∣ 30 → ∃ m : ℕ, k = 2^m ∧ k ∣ 30)) → 
  ∃ num_divisors : ℕ, num_divisors = 8 := 
by 
  sorry

end num_divisors_of_30_l237_237538


namespace area_square_A_32_l237_237892

-- Define the areas of the squares in Figure B and Figure A and their relationship with the triangle areas
def identical_isosceles_triangles_with_squares (area_square_B : ℝ) (area_triangle_B : ℝ) (area_square_A : ℝ) (area_triangle_A : ℝ) :=
  area_triangle_B = (area_square_B / 2) * 4 ∧
  area_square_A / area_triangle_A = 4 / 9

theorem area_square_A_32 {area_square_B : ℝ} (h : area_square_B = 36) :
  identical_isosceles_triangles_with_squares area_square_B 72 32 72 :=
by
  sorry

end area_square_A_32_l237_237892


namespace double_theta_acute_l237_237416

theorem double_theta_acute (θ : ℝ) (h : 0 < θ ∧ θ < 90) : 0 < 2 * θ ∧ 2 * θ < 180 :=
by
  sorry

end double_theta_acute_l237_237416


namespace find_a_l237_237596

theorem find_a {a : ℝ} (h : ∀ x : ℝ, (x^2 - 4 * x + a) + |x - 3| ≤ 5 → x ≤ 3) : a = 8 :=
sorry

end find_a_l237_237596


namespace cyclist_wait_time_l237_237262

theorem cyclist_wait_time 
  (hiker_speed : ℝ) (cyclist_speed : ℝ) (wait_time : ℝ) (catch_up_time : ℝ) 
  (hiker_speed_eq : hiker_speed = 4) 
  (cyclist_speed_eq : cyclist_speed = 12) 
  (wait_time_eq : wait_time = 5 / 60) 
  (catch_up_time_eq : catch_up_time = (2 / 3) / (1 / 15)) 
  : catch_up_time * 60 = 10 := 
by 
  sorry

end cyclist_wait_time_l237_237262


namespace f_cos_x_l237_237079

theorem f_cos_x (f : ℝ → ℝ) (x : ℝ) (h : f (Real.sin x) = 2 - Real.cos x ^ 2) : f (Real.cos x) = 2 + Real.sin x ^ 2 := by
  sorry

end f_cos_x_l237_237079


namespace tangent_lengths_l237_237353

noncomputable def internal_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 + r2)^2)

noncomputable def external_tangent_length (r1 r2 d : ℝ) : ℝ :=
  Real.sqrt (d^2 - (r1 - r2)^2)

theorem tangent_lengths (r1 r2 d : ℝ) (h_r1 : r1 = 8) (h_r2 : r2 = 10) (h_d : d = 50) :
  internal_tangent_length r1 r2 d = 46.67 ∧ external_tangent_length r1 r2 d = 49.96 :=
by
  sorry

end tangent_lengths_l237_237353


namespace trees_died_due_to_typhoon_l237_237017

-- defining the initial number of trees
def initial_trees : ℕ := 9

-- defining the additional trees grown after the typhoon
def additional_trees : ℕ := 5

-- defining the final number of trees after all events
def final_trees : ℕ := 10

-- we introduce D as the number of trees that died due to the typhoon
def trees_died (D : ℕ) : Prop := initial_trees - D + additional_trees = final_trees

-- the theorem we need to prove is that 4 trees died
theorem trees_died_due_to_typhoon : trees_died 4 :=
by
  sorry

end trees_died_due_to_typhoon_l237_237017


namespace range_x_y_l237_237488

variable (x y : ℝ)

theorem range_x_y (hx : 60 < x ∧ x < 84) (hy : 28 < y ∧ y < 33) : 
  27 < x - y ∧ x - y < 56 :=
sorry

end range_x_y_l237_237488


namespace isosceles_triangle_perimeter_l237_237091

theorem isosceles_triangle_perimeter : 
  ∀ a b c : ℝ, a^2 - 6 * a + 5 = 0 → b^2 - 6 * b + 5 = 0 → 
    (a = b ∨ b = c ∨ a = c) →
    (a + b > c ∧ b + c > a ∧ a + c > b) →
    a + b + c = 11 := 
by
  intros a b c ha hb hiso htri
  sorry

end isosceles_triangle_perimeter_l237_237091


namespace valid_schedule_count_l237_237928

theorem valid_schedule_count :
  ∃ (valid_schedules : Finset (Fin 8 → Option (Fin 4))),
    valid_schedules.card = 488 ∧
    (∀ (schedule : Fin 8 → Option (Fin 4)), schedule ∈ valid_schedules →
      (∀ i : Fin 7, schedule i ≠ none ∧ schedule (i + 1) ≠ schedule i) ∧
      schedule 4 = none) :=
sorry

end valid_schedule_count_l237_237928


namespace treasure_distribution_l237_237866

noncomputable def calculate_share (investment total_investment total_value : ℝ) : ℝ :=
  (investment / total_investment) * total_value

theorem treasure_distribution 
  (investment_fonzie investment_aunt_bee investment_lapis investment_skylar investment_orion total_treasure : ℝ)
  (total_investment : ℝ)
  (h : total_investment = investment_fonzie + investment_aunt_bee + investment_lapis + investment_skylar + investment_orion) :
  calculate_share investment_fonzie total_investment total_treasure = 210000 ∧
  calculate_share investment_aunt_bee total_investment total_treasure = 255000 ∧
  calculate_share investment_lapis total_investment total_treasure = 270000 ∧
  calculate_share investment_skylar total_investment total_treasure = 225000 ∧
  calculate_share investment_orion total_investment total_treasure = 240000 :=
by
  sorry

end treasure_distribution_l237_237866


namespace sunzi_problem_solution_l237_237226

theorem sunzi_problem_solution (x y : ℝ) :
  (y = x + 4.5) ∧ (0.5 * y = x - 1) ↔ (y = x + 4.5 ∧ 0.5 * y = x - 1) :=
by 
  sorry

end sunzi_problem_solution_l237_237226


namespace day_53_days_from_friday_l237_237694

def day_of_week (n : ℕ) : String :=
  match n % 7 with
  | 0 => "Friday"
  | 1 => "Saturday"
  | 2 => "Sunday"
  | 3 => "Monday"
  | 4 => "Tuesday"
  | 5 => "Wednesday"
  | 6 => "Thursday"
  | _ => "Unknown"

theorem day_53_days_from_friday : day_of_week 53 = "Tuesday" := by
  sorry

end day_53_days_from_friday_l237_237694


namespace math_problem_l237_237132

theorem math_problem (a b c : ℝ) (h1 : a + b + c = 3) (h2 : a^2 + b^2 + c^2 = 3) : a^(2008 : ℕ) + b^(2008 : ℕ) + c^(2008 : ℕ) = 3 :=
by 
  let h1' : a + b + c = 3 := h1
  let h2' : a^2 + b^2 + c^2 = 3 := h2
  sorry

end math_problem_l237_237132


namespace find_cos_beta_l237_237574

variable {α β : ℝ}
variable (h_acute_α : 0 < α ∧ α < π / 2)
variable (h_acute_β : 0 < β ∧ β < π / 2)
variable (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
variable (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5)

theorem find_cos_beta 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_sin_α : Real.sin α = 2 / 5 * Real.sqrt 5)
  (h_sin_α_plus_β : Real.sin (α + β) = 3 / 5) :
  Real.cos β = Real.sqrt 5 / 5 := 
sorry

end find_cos_beta_l237_237574


namespace actual_height_of_boy_l237_237614

variable (wrong_height : ℕ) (boys : ℕ) (wrong_avg correct_avg : ℕ)
variable (x : ℕ)

-- Given conditions
def conditions 
:= boys = 35 ∧
   wrong_height = 166 ∧
   wrong_avg = 185 ∧
   correct_avg = 183

-- Question: Proving the actual height
theorem actual_height_of_boy (h : conditions boys wrong_height wrong_avg correct_avg) : 
  x = wrong_height + (boys * wrong_avg - boys * correct_avg) := 
  sorry

end actual_height_of_boy_l237_237614


namespace parrots_fraction_l237_237720

variable (P T : ℚ) -- P: fraction of parrots, T: fraction of toucans

def fraction_parrots (P T : ℚ) : Prop :=
  P + T = 1 ∧
  (2 / 3) * P + (1 / 4) * T = 0.5

theorem parrots_fraction (P T : ℚ) (h : fraction_parrots P T) : P = 3 / 5 :=
by
  sorry

end parrots_fraction_l237_237720


namespace concert_ticket_sales_l237_237072

theorem concert_ticket_sales (A C : ℕ) (total : ℕ) :
  (C = 3 * A) →
  (7 * A + 3 * C = 6000) →
  (total = A + C) →
  total = 1500 :=
by
  intros
  -- The proof is not required
  sorry

end concert_ticket_sales_l237_237072


namespace spheres_volume_ratio_l237_237240

theorem spheres_volume_ratio (S1 S2 V1 V2 : ℝ)
  (h1 : S1 / S2 = 1 / 9) 
  (h2a : S1 = 4 * π * r1^2) 
  (h2b : S2 = 4 * π * r2^2)
  (h3a : V1 = 4 / 3 * π * r1^3)
  (h3b : V2 = 4 / 3 * π * r2^3)
  : V1 / V2 = 1 / 27 :=
by
  sorry

end spheres_volume_ratio_l237_237240


namespace part_a_l237_237624

def A (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f (x * y) = x * f y

theorem part_a (f : ℝ → ℝ) (h : A f) : ∀ x y : ℝ, f (x + y) = f x + f y :=
sorry

end part_a_l237_237624


namespace price_of_sports_equipment_l237_237654

theorem price_of_sports_equipment (x y : ℕ) (a b : ℕ) :
  (2 * x + y = 330) → (5 * x + 2 * y = 780) → x = 120 ∧ y = 90 ∧
  (120 * a + 90 * b = 810) → a = 3 ∧ b = 5 :=
by
  intros h1 h2 h3
  sorry

end price_of_sports_equipment_l237_237654


namespace greatest_possible_remainder_l237_237797

theorem greatest_possible_remainder (x : ℕ) : ∃ r, r < 9 ∧ x % 9 = r ∧ r = 8 :=
by
  use 8
  sorry -- Proof to be filled in

end greatest_possible_remainder_l237_237797


namespace building_height_l237_237049

-- We start by defining the heights of the stories.
def first_story_height : ℕ := 12
def additional_height_per_story : ℕ := 3
def number_of_stories : ℕ := 20
def first_ten_stories : ℕ := 10
def remaining_stories : ℕ := number_of_stories - first_ten_stories

-- Now we define what it means for the total height of the building to be 270 feet.
theorem building_height :
  first_ten_stories * first_story_height + remaining_stories * (first_story_height + additional_height_per_story) = 270 := by
  sorry

end building_height_l237_237049


namespace ticket_cost_is_25_l237_237478

-- Define the given conditions
def num_tickets_first_show : ℕ := 200
def num_tickets_second_show : ℕ := 3 * num_tickets_first_show
def total_tickets : ℕ := num_tickets_first_show + num_tickets_second_show
def total_revenue_in_dollars : ℕ := 20000

-- Claim to prove
theorem ticket_cost_is_25 : ∃ x : ℕ, total_tickets * x = total_revenue_in_dollars ∧ x = 25 :=
by
  -- sorry is used here to skip the proof
  sorry

end ticket_cost_is_25_l237_237478


namespace area_of_quadrilateral_l237_237776

theorem area_of_quadrilateral (A B C : ℝ) (h1 : A + B = C) (h2 : A = 16) (h3 : B = 16) :
  (C - A - B) / 2 = 8 :=
by
  sorry

end area_of_quadrilateral_l237_237776


namespace eval_expression_l237_237717

variable {x : ℝ}

theorem eval_expression (x : ℝ) : x * (x * (x * (3 - x) - 5) + 8) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 8 * x + 2 :=
by
  sorry

end eval_expression_l237_237717


namespace problem_inequality_l237_237485

variable {a b : ℝ}

theorem problem_inequality 
  (h_a_nonzero : a ≠ 0) 
  (h_b_nonzero : b ≠ 0)
  (h_a_gt_b : a > b) : 
  1 / (a * b^2) > 1 / (a^2 * b) := 
by 
  sorry

end problem_inequality_l237_237485


namespace trig_identity_l237_237127

theorem trig_identity (θ : ℝ) (h : Real.tan θ = 2) : 
  (3 * Real.sin θ - 2 * Real.cos θ) / (Real.sin θ + 3 * Real.cos θ) = 4 / 5 :=
by 
  sorry

end trig_identity_l237_237127


namespace geometric_sequence_term_formula_l237_237333

theorem geometric_sequence_term_formula (a n : ℕ) (a_seq : ℕ → ℕ)
  (h1 : a_seq 0 = a - 1) (h2 : a_seq 1 = a + 1) (h3 : a_seq 2 = a + 4)
  (geometric_seq : ∀ n, a_seq (n + 1) = a_seq n * ((a_seq 1) / (a_seq 0))) :
  a = 5 ∧ a_seq n = 4 * (3 / 2) ^ (n - 1) :=
by
  sorry

end geometric_sequence_term_formula_l237_237333


namespace sum_of_numbers_equal_16_l237_237495

theorem sum_of_numbers_equal_16 
  (a b c : ℕ) 
  (h1 : a * b = a * c - 1 ∨ a * b = b * c - 1 ∨ a * c = b * c - 1) 
  (h2 : a * b = a * c + 49 ∨ a * b = b * c + 49 ∨ a * c = b * c + 49) :
  a + b + c = 16 :=
sorry

end sum_of_numbers_equal_16_l237_237495


namespace quadratic_function_has_specific_k_l237_237149

theorem quadratic_function_has_specific_k (k : ℤ) :
  (∀ x : ℝ, ∃ y : ℝ, y = (k-1)*x^(k^2-k+2) + k*x - 1) ↔ k = 0 :=
by
  sorry

end quadratic_function_has_specific_k_l237_237149


namespace inv_composition_l237_237270

theorem inv_composition (f g : ℝ → ℝ) (hf : Function.Bijective f) (hg : Function.Bijective g) (h : ∀ x, f⁻¹ (g x) = 2 * x - 4) : 
  g⁻¹ (f (-3)) = 1 / 2 :=
by
  sorry

end inv_composition_l237_237270


namespace range_of_f_l237_237361

noncomputable def f (x : ℝ) : ℝ :=
  (Real.arccos (x / 2))^2 + 
  Real.pi * Real.arcsin (x / 2) - 
  (Real.arcsin (x / 2))^2 + 
  (Real.pi^2 / 6) * (x^2 + 2 * x + 1)

theorem range_of_f (x : ℝ) (hx : -2 ≤ x ∧ x ≤ 2) :
  ∃ y : ℝ, (f y) = x ∧  (Real.pi^2 / 4) ≤ y ∧ y ≤ (39 * Real.pi^2 / 96) := 
sorry

end range_of_f_l237_237361


namespace package_weights_l237_237489

theorem package_weights (a b c : ℕ) 
  (h1 : a + b = 108) 
  (h2 : b + c = 132) 
  (h3 : c + a = 138) 
  (h4 : a ≥ 40) 
  (h5 : b ≥ 40) 
  (h6 : c ≥ 40) : 
  a + b + c = 189 :=
sorry

end package_weights_l237_237489


namespace product_of_b_product_of_values_l237_237104

/-- 
If the distance between the points (3b, b+2) and (6, 3) is 3√5 units,
then the product of all possible values of b is -0.8.
-/
theorem product_of_b (b : ℝ)
  (h : (6 - 3 * b)^2 + (3 - (b + 2))^2 = (3 * Real.sqrt 5)^2) :
  b = 4 ∨ b = -0.2 := sorry

/--
The product of the values satisfying the theorem product_of_b is -0.8.
-/
theorem product_of_values : (4 : ℝ) * (-0.2) = -0.8 := 
by norm_num -- using built-in arithmetic simplification

end product_of_b_product_of_values_l237_237104


namespace time_to_fill_pond_l237_237311

noncomputable def pond_capacity : ℝ := 200
noncomputable def normal_pump_rate : ℝ := 6
noncomputable def restriction_factor : ℝ := 2 / 3
noncomputable def restricted_pump_rate : ℝ := restriction_factor * normal_pump_rate

theorem time_to_fill_pond : pond_capacity / restricted_pump_rate = 50 := 
by 
  -- This is where the proof would go
  sorry

end time_to_fill_pond_l237_237311


namespace parabola_trajectory_l237_237078

theorem parabola_trajectory (P : ℝ × ℝ) : 
  (dist P (3, 0) = dist P (3 - 1, P.2 - 0)) → P.2^2 = 12 * P.1 := 
sorry

end parabola_trajectory_l237_237078


namespace solve_inequality_l237_237933

theorem solve_inequality (a x : ℝ) : 
  (a = 0 ∨ a = 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ False)) ∧
  (0 < a ∧ a < 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ a^2 < x ∧ x < a)) ∧
  (a < 0 ∨ a > 1 → (x^2 - (a^2 + a) * x + a^3 < 0 ↔ a < x ∧ x < a^2)) :=
  by
    sorry

end solve_inequality_l237_237933


namespace course_length_l237_237340

noncomputable def timeBicycling := 12 / 60 -- hours
noncomputable def avgRateBicycling := 30 -- miles per hour
noncomputable def timeRunning := (117 - 12) / 60 -- hours
noncomputable def avgRateRunning := 8 -- miles per hour

theorem course_length : avgRateBicycling * timeBicycling + avgRateRunning * timeRunning = 20 := 
by
  sorry

end course_length_l237_237340


namespace quadratic_has_two_distinct_real_roots_l237_237392

theorem quadratic_has_two_distinct_real_roots :
  ∀ (x : ℝ), ∃ (r1 r2 : ℝ), (x^2 - 2*x - 1 = 0) → r1 ≠ r2 :=
by
  sorry

end quadratic_has_two_distinct_real_roots_l237_237392


namespace age_difference_l237_237839

theorem age_difference (d : ℕ) (h1 : 18 + (18 - d) + (18 - 2 * d) + (18 - 3 * d) = 48) : d = 4 :=
sorry

end age_difference_l237_237839


namespace painter_total_cost_l237_237547

-- Define the arithmetic sequence for house addresses
def south_side_arith_seq (n : ℕ) : ℕ := 5 + (n - 1) * 7
def north_side_arith_seq (n : ℕ) : ℕ := 6 + (n - 1) * 8

-- Define the counting of digits
def digit_count (n : ℕ) : ℕ :=
  if n < 10 then 1
  else if n < 100 then 2
  else 3

-- Define the condition of painting cost for multiples of 10
def painting_cost (n : ℕ) : ℕ :=
  if n % 10 = 0 then 2 * digit_count n
  else digit_count n

-- Calculate total cost for side with given arithmetic sequence
def total_cost_for_side (side_arith_seq : ℕ → ℕ): ℕ :=
  List.range 25 |>.map (λ n => painting_cost (side_arith_seq (n + 1))) |>.sum

-- Main theorem to prove
theorem painter_total_cost : total_cost_for_side south_side_arith_seq + total_cost_for_side north_side_arith_seq = 147 := by
  sorry

end painter_total_cost_l237_237547


namespace min_sum_four_consecutive_nat_nums_l237_237126

theorem min_sum_four_consecutive_nat_nums (a : ℕ) (h1 : a % 11 = 0) (h2 : (a + 1) % 7 = 0)
    (h3 : (a + 2) % 5 = 0) (h4 : (a + 3) % 3 = 0) : a + (a + 1) + (a + 2) + (a + 3) = 1458 :=
  sorry

end min_sum_four_consecutive_nat_nums_l237_237126


namespace leo_total_travel_cost_l237_237356

-- Define the conditions as variables and assumptions in Lean
def cost_one_way : ℕ := 24
def working_days : ℕ := 20

-- Define the total travel cost as a function
def total_travel_cost (cost_one_way : ℕ) (working_days : ℕ) : ℕ :=
  cost_one_way * 2 * working_days

-- State the theorem to prove the total travel cost
theorem leo_total_travel_cost : total_travel_cost 24 20 = 960 :=
sorry

end leo_total_travel_cost_l237_237356


namespace original_number_l237_237757

theorem original_number (x : ℝ) (h1 : 74 * x = 19732) : x = 267 := by
  sorry

end original_number_l237_237757


namespace distance_between_X_and_Y_l237_237377

theorem distance_between_X_and_Y :
  ∀ (D : ℝ), 
  (10 : ℝ) * (D / (10 : ℝ) + D / (4 : ℝ)) / (10 + 4) = 142.85714285714286 → 
  D = 1000 :=
by
  intro D
  sorry

end distance_between_X_and_Y_l237_237377


namespace min_value_of_f_l237_237475

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * x - 3 / x

theorem min_value_of_f : ∃ x < 0, ∀ y : ℝ, y = f x → y ≥ 1 + 2 * Real.sqrt 6 :=
by
  -- Sorry is used to skip the actual proof.
  sorry

end min_value_of_f_l237_237475


namespace number_pairs_sum_diff_prod_quotient_l237_237064

theorem number_pairs_sum_diff_prod_quotient (x y : ℤ) (h : x ≥ y) :
  (x + y) + (x - y) + x * y + x / y = 800 ∨ (x + y) + (x - y) + x * y + x / y = 400 :=
sorry

-- Correct answers for A = 800
example : (38 + 19) + (38 - 19) + 38 * 19 + 38 / 19 = 800 := by norm_num
example : (-42 + -21) + (-42 - -21) + (-42 * -21) + (-42 / -21) = 800 := by norm_num
example : (72 + 9) + (72 - 9) + 72 * 9 + 72 / 9 = 800 := by norm_num
example : (-88 + -11) + (-88 - -11) + -(88 * -11) + (-88 / -11) = 800 := by norm_num
example : (128 + 4) + (128 - 4) + 128 * 4 + 128 / 4 = 800 := by norm_num
example : (-192 + -6) + (-192 - -6) + -192 * -6 + ( -192 / -6 ) = 800 := by norm_num
example : (150 + 3) + (150 - 3) + 150 * 3 + 150 / 3 = 800 := by norm_num
example : (-250 + -5) + (-250 - -5) + (-250 * -5) + (-250 / -5) = 800 := by norm_num
example : (200 + 1) + (200 - 1) + 200 * 1 + 200 / 1 = 800 := by norm_num
example : (-600 + -3) + (-600 - -3) + -600 * -3 + -600 / -3 = 800 := by norm_num

-- Correct answers for A = 400
example : (19 + 19) + (19 - 19) + 19 * 19 + 19 / 19 = 400 := by norm_num
example : (-21 + -21) + (-21 - -21) + (-21 * -21) + (-21 / -21) = 400 := by norm_num
example : (36 + 9) + (36 - 9) + 36 * 9 + 36 / 9 = 400 := by norm_num
example : (-44 + -11) + (-44 - -11) + (-44 * -11) + (-44 / -11) = 400 := by norm_num
example : (64 + 4) + (64 - 4) + 64 * 4 + 64 / 4 = 400 := by norm_num
example : (-96 + -6) + (-96 - -6) + (-96 * -6) + (-96 / -6) = 400 := by norm_num
example : (75 + 3) + (75 - 3) + 75 * 3 + 75 / 3 = 400 := by norm_num
example : (-125 + -5) + (-125 - -5) + (-125 * -5) + (-125 / -5) = 400 := by norm_num
example : (100 + 1) + (100 - 1) + 100 * 1 + 100 / 1 = 400 := by norm_num
example : (-300 + -3) + (-300 - -3) + (-300 * -3) + (-300 / -3) = 400 := by norm_num

end number_pairs_sum_diff_prod_quotient_l237_237064


namespace scooter_gain_percent_l237_237554

theorem scooter_gain_percent 
  (purchase_price : ℝ) (repair_costs : ℝ) (selling_price : ℝ) 
  (h1 : purchase_price = 800) (h2 : repair_costs = 200) (h3 : selling_price = 1200) : 
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100 = 20 :=
by
  sorry

end scooter_gain_percent_l237_237554


namespace find_speed_of_A_l237_237814

noncomputable def speed_of_A_is_7_5 (a : ℝ) : Prop :=
  -- Conditions
  ∃ (b : ℝ), b = a + 5 ∧ 
  (60 / a = 100 / b) → 
  -- Conclusion
  a = 7.5

-- Statement in Lean 4
theorem find_speed_of_A (a : ℝ) (h : speed_of_A_is_7_5 a) : a = 7.5 :=
  sorry

end find_speed_of_A_l237_237814


namespace rent_3600_rents_88_max_revenue_is_4050_l237_237034

def num_total_cars : ℕ := 100
def initial_rent : ℕ := 3000
def rent_increase_step : ℕ := 50
def maintenance_cost_rented : ℕ := 150
def maintenance_cost_unrented : ℕ := 50

def rented_cars (rent : ℕ) : ℕ :=
  if rent < initial_rent then num_total_cars
  else num_total_cars - ((rent - initial_rent) / rent_increase_step)

def monthly_revenue (rent : ℕ) : ℕ :=
  let rented := rented_cars rent
  rent * rented - (rented * maintenance_cost_rented + (num_total_cars - rented) * maintenance_cost_unrented)

theorem rent_3600_rents_88 :
  rented_cars 3600 = 88 := by 
  sorry

theorem max_revenue_is_4050 :
  ∃ (rent : ℕ), rent = 4050 ∧ monthly_revenue rent = 37050 := by
  sorry

end rent_3600_rents_88_max_revenue_is_4050_l237_237034


namespace distinct_real_roots_find_other_root_and_k_l237_237026

-- Definition of the quadratic equation
def quadratic_eq (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Part (1): Proving the discriminant condition
theorem distinct_real_roots (k : ℝ) : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_eq 2 k (-1) x1 = 0 ∧ quadratic_eq 2 k (-1) x2 = 0 := by
  sorry

-- Part (2): Finding the other root and the value of k
theorem find_other_root_and_k : 
  ∃ k : ℝ, ∃ x2 : ℝ,
    quadratic_eq 2 1 (-1) (-1) = 0 ∧ quadratic_eq 2 1 (-1) x2 = 0 ∧ k = 1 ∧ x2 = 1/2 := by
  sorry

end distinct_real_roots_find_other_root_and_k_l237_237026


namespace diamond_two_three_l237_237528

def diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem diamond_two_three : diamond 2 3 = 16 := by
  sorry

end diamond_two_three_l237_237528


namespace Carly_injured_week_miles_l237_237987

def week1_miles : ℕ := 2
def week2_miles : ℕ := week1_miles * 2 + 3
def week3_miles : ℕ := week2_miles * 9 / 7
def week4_miles : ℕ := week3_miles - 5

theorem Carly_injured_week_miles : week4_miles = 4 :=
  by
    sorry

end Carly_injured_week_miles_l237_237987


namespace find_first_term_of_geometric_series_l237_237995

theorem find_first_term_of_geometric_series 
  (r : ℚ) (S : ℚ) (a : ℚ) 
  (hr : r = -1/3) (hS : S = 9)
  (h_sum_formula : S = a / (1 - r)) : 
  a = 12 := 
by
  sorry

end find_first_term_of_geometric_series_l237_237995


namespace first_platform_length_is_150_l237_237177

-- Defining the conditions
def train_length : ℝ := 150
def first_platform_time : ℝ := 15
def second_platform_length : ℝ := 250
def second_platform_time : ℝ := 20

-- The distance covered when crossing the first platform is length of train + length of first platform
def distance_first_platform (L : ℝ) : ℝ := train_length + L

-- The distance covered when crossing the second platform is length of train + length of a known 250 m platform
def distance_second_platform : ℝ := train_length + second_platform_length

-- We are to prove that the length of the first platform, given the conditions, is 150 meters.
theorem first_platform_length_is_150 : ∃ L : ℝ, (distance_first_platform L / distance_second_platform) = (first_platform_time / second_platform_time) ∧ L = 150 :=
by
  let L := 150
  have h1 : distance_first_platform L = train_length + L := rfl
  have h2 : distance_second_platform = train_length + second_platform_length := rfl
  have h3 : distance_first_platform L / distance_second_platform = first_platform_time / second_platform_time :=
    by sorry
  use L
  exact ⟨h3, rfl⟩

end first_platform_length_is_150_l237_237177


namespace Milly_spends_135_minutes_studying_l237_237840

-- Definitions of homework times
def mathHomeworkTime := 60
def geographyHomeworkTime := mathHomeworkTime / 2
def scienceHomeworkTime := (mathHomeworkTime + geographyHomeworkTime) / 2

-- Definition of Milly's total study time
def totalStudyTime := mathHomeworkTime + geographyHomeworkTime + scienceHomeworkTime

-- Theorem stating that Milly spends 135 minutes studying
theorem Milly_spends_135_minutes_studying : totalStudyTime = 135 :=
by
  -- Proof omitted
  sorry

end Milly_spends_135_minutes_studying_l237_237840


namespace age_difference_ratio_l237_237870

theorem age_difference_ratio (h : ℕ) (f : ℕ) (m : ℕ) 
  (harry_age : h = 50) 
  (father_age : f = h + 24) 
  (mother_age : m = 22 + h) :
  (f - m) / h = 1 / 25 := 
by 
  sorry

end age_difference_ratio_l237_237870


namespace largest_possible_b_b_eq_4_of_largest_l237_237773

theorem largest_possible_b (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) : b ≤ 4 := by
  sorry

theorem b_eq_4_of_largest (b : ℚ) (h : (3*b + 4)*(b - 2) = 9*b) (hb : b = 4) : True := by
  sorry

end largest_possible_b_b_eq_4_of_largest_l237_237773


namespace div_by_20_l237_237039

theorem div_by_20 (n : ℕ) : 20 ∣ (9 ^ (8 * n + 4) - 7 ^ (8 * n + 4)) :=
  sorry

end div_by_20_l237_237039


namespace variance_is_4_l237_237462

variable {datapoints : List ℝ}

noncomputable def variance (datapoints : List ℝ) : ℝ :=
  let n := datapoints.length
  let mean := (datapoints.sum / n : ℝ)
  (1 / n : ℝ) * ((datapoints.map (λ x => x ^ 2)).sum - n * mean ^ 2)

theorem variance_is_4 :
  (datapoints.length = 20)
  → ((datapoints.map (λ x => x ^ 2)).sum = 800)
  → (datapoints.sum / 20 = 6)
  → variance datapoints = 4 := by
  intros length_cond sum_squares_cond mean_cond
  sorry

end variance_is_4_l237_237462


namespace find_garden_perimeter_l237_237042

noncomputable def garden_perimeter (a : ℝ) (P : ℝ) : Prop :=
  a = 2 * P + 14.25 ∧ a = 90.25

theorem find_garden_perimeter :
  ∃ P : ℝ, garden_perimeter 90.25 P ∧ P = 38 :=
by
  sorry

end find_garden_perimeter_l237_237042


namespace bert_ernie_ratio_l237_237399

theorem bert_ernie_ratio (berts_stamps ernies_stamps peggys_stamps : ℕ) 
  (h1 : peggys_stamps = 75) 
  (h2 : ernies_stamps = 3 * peggys_stamps) 
  (h3 : berts_stamps = peggys_stamps + 825) : 
  berts_stamps / ernies_stamps = 4 := 
by sorry

end bert_ernie_ratio_l237_237399


namespace nonnegative_difference_roots_eq_12_l237_237128

theorem nonnegative_difference_roots_eq_12 :
  ∀ (x : ℝ), (x^2 + 40 * x + 300 = -64) →
  ∃ (r₁ r₂ : ℝ), (x^2 + 40 * x + 364 = 0) ∧ 
  (r₁ = -26 ∧ r₂ = -14)
  ∧ (|r₁ - r₂| = 12) :=
by
  sorry

end nonnegative_difference_roots_eq_12_l237_237128


namespace attendance_rate_comparison_l237_237848

theorem attendance_rate_comparison (attendees_A total_A attendees_B total_B : ℕ) 
  (hA : (attendees_A / total_A: ℚ) > (attendees_B / total_B: ℚ)) : 
  (attendees_A > attendees_B) → false :=
by
  sorry

end attendance_rate_comparison_l237_237848


namespace necessary_and_sufficient_conditions_l237_237360

open Real

def cubic_has_arithmetic_sequence_roots (a b c : ℝ) : Prop :=
∃ x y : ℝ,
  (x - y) * (x) * (x + y) + a * (x^2 + x - y + x + y) + b * x + c = 0 ∧
  3 * x = -a

theorem necessary_and_sufficient_conditions
  (a b c : ℝ) (h : cubic_has_arithmetic_sequence_roots a b c) :
  2 * a^3 - 9 * a * b + 27 * c = 0 ∧ a^2 - 3 * b ≥ 0 :=
sorry

end necessary_and_sufficient_conditions_l237_237360


namespace sqrt_of_1024_l237_237159

theorem sqrt_of_1024 (x : ℝ) (h1 : x > 0) (h2 : x ^ 2 = 1024) : x = 32 :=
sorry

end sqrt_of_1024_l237_237159


namespace find_x_such_that_l237_237615

theorem find_x_such_that {x : ℝ} (h : ⌈x⌉ * x + 15 = 210) : x = 195 / 14 :=
by
  sorry

end find_x_such_that_l237_237615


namespace sum_a5_a6_a7_l237_237471

def S (n : ℕ) : ℕ :=
  n^2 + 2 * n + 5

theorem sum_a5_a6_a7 : S 7 - S 4 = 39 :=
  by sorry

end sum_a5_a6_a7_l237_237471


namespace value_of_x_l237_237841

theorem value_of_x (x y : ℕ) (h1 : x / y = 7 / 3) (h2 : y = 21) : x = 49 := sorry

end value_of_x_l237_237841


namespace pencils_purchased_l237_237293

variable (P : ℕ)

theorem pencils_purchased (misplaced broke found bought left : ℕ) (h1 : misplaced = 7) (h2 : broke = 3) (h3 : found = 4) (h4 : bought = 2) (h5 : left = 16) :
  P - misplaced - broke + found + bought = left → P = 22 :=
by
  intros h
  have h_eq : P - 7 - 3 + 4 + 2 = 16 := by
    rw [h1, h2, h3, h4, h5] at h; exact h
  sorry

end pencils_purchased_l237_237293


namespace trajectory_of_point_P_l237_237297

theorem trajectory_of_point_P :
  ∀ (x y : ℝ), 
  (∀ (m n : ℝ), n = 2 * m - 4 → (1 - m, -n) = (x - 1, y)) → 
  y = 2 * x :=
by
  sorry

end trajectory_of_point_P_l237_237297


namespace gnomes_in_fifth_house_l237_237997

-- Defining the problem conditions
def num_houses : Nat := 5
def gnomes_per_house : Nat := 3
def total_gnomes : Nat := 20

-- Defining the condition for the first four houses
def gnomes_in_first_four_houses : Nat := 4 * gnomes_per_house

-- Statement of the problem
theorem gnomes_in_fifth_house : 20 - (4 * 3) = 8 := by
  sorry

end gnomes_in_fifth_house_l237_237997


namespace probability_same_length_l237_237453

/-- Defining the set of all sides and diagonals of a regular hexagon. -/
def T : Finset ℚ := sorry

/-- There are exactly 6 sides in the set T. -/
def sides_count : ℕ := 6

/-- There are exactly 9 diagonals in the set T. -/
def diagonals_count : ℕ := 9

/-- The total number of segments in the set T. -/
def total_segments : ℕ := sides_count + diagonals_count

theorem probability_same_length :
  let prob_side := (6 : ℚ) / total_segments * (5 / (total_segments - 1))
  let prob_diagonal := (9 : ℚ) / total_segments * (4 / (total_segments - 1))
  prob_side + prob_diagonal = 17 / 35 := 
by
  admit

end probability_same_length_l237_237453


namespace cost_of_each_skirt_l237_237947

-- Problem definitions based on conditions
def cost_of_art_supplies : ℕ := 20
def total_expenditure : ℕ := 50
def number_of_skirts : ℕ := 2

-- Proving the cost of each skirt
theorem cost_of_each_skirt (cost_of_each_skirt : ℕ) : 
  number_of_skirts * cost_of_each_skirt + cost_of_art_supplies = total_expenditure → 
  cost_of_each_skirt = 15 := 
by 
  sorry

end cost_of_each_skirt_l237_237947


namespace tangent_line_equation_l237_237557

noncomputable def f (x : ℝ) : ℝ := (x^3 - 1) / x

theorem tangent_line_equation :
  let x₀ := 1
  let y₀ := f x₀
  let m := deriv f x₀
  y₀ = 0 →
  m = 3 →
  ∀ x y : ℝ, (y - y₀ = m * (x - x₀)) ↔ (y = 3 * x - 3) :=
by
  intros x₀ y₀ m h₀ hm x y
  sorry

end tangent_line_equation_l237_237557


namespace product_mod_10_l237_237895

theorem product_mod_10 (a b c : ℕ) (ha : a % 10 = 4) (hb : b % 10 = 5) (hc : c % 10 = 5) :
  (a * b * c) % 10 = 0 :=
sorry

end product_mod_10_l237_237895


namespace square_area_twice_triangle_perimeter_l237_237555

noncomputable def perimeter_of_triangle (a b c : ℕ) : ℕ :=
  a + b + c

noncomputable def side_length_of_square (perimeter : ℕ) : ℕ :=
  perimeter / 4

noncomputable def area_of_square (side_length : ℕ) : ℕ :=
  side_length * side_length

theorem square_area_twice_triangle_perimeter (a b c : ℕ) (h1 : perimeter_of_triangle a b c = 22) (h2 : a = 5) (h3 : b = 7) (h4 : c = 10) : area_of_square (side_length_of_square (2 * perimeter_of_triangle a b c)) = 121 :=
by
  sorry

end square_area_twice_triangle_perimeter_l237_237555


namespace triangle_area_l237_237194

theorem triangle_area :
  ∃ (a b c : ℕ), a + b + c = 12 ∧ a + b > c ∧ a + c > b ∧ b + c > a ∧ ∃ (A : ℝ), 
  A = Real.sqrt (6 * (6 - a) * (6 - b) * (6 - c)) ∧ A = 6 := by
  sorry

end triangle_area_l237_237194


namespace sum_of_coordinates_of_A_l237_237674

theorem sum_of_coordinates_of_A
  (A B C : ℝ × ℝ)
  (AC AB BC : ℝ)
  (h1 : AC / AB = 1 / 3)
  (h2 : BC / AB = 2 / 3)
  (hB : B = (2, 5))
  (hC : C = (5, 8)) :
  (A.1 + A.2) = 16 :=
sorry

end sum_of_coordinates_of_A_l237_237674


namespace air_conditioner_sale_price_l237_237117

theorem air_conditioner_sale_price (P : ℝ) (d1 d2 : ℝ) (hP : P = 500) (hd1 : d1 = 0.10) (hd2 : d2 = 0.20) :
  ((P * (1 - d1)) * (1 - d2)) / P * 100 = 72 :=
by
  sorry

end air_conditioner_sale_price_l237_237117


namespace intersection_is_empty_l237_237523

open Finset

namespace ComplementIntersection

-- Define the universal set U, sets M and N
def U : Finset ℕ := {1, 2, 3, 4, 5}
def M : Finset ℕ := {1, 3, 4}
def N : Finset ℕ := {2, 4, 5}

-- The complement of M with respect to U
def complement_U_M : Finset ℕ := U \ M

-- The complement of N with respect to U
def complement_U_N : Finset ℕ := U \ N

-- The intersection of the complements
def intersection_complements : Finset ℕ := complement_U_M ∩ complement_U_N

-- The proof statement
theorem intersection_is_empty : intersection_complements = ∅ :=
by sorry

end ComplementIntersection

end intersection_is_empty_l237_237523


namespace lcm_of_times_l237_237728

-- Define the times each athlete takes to complete one lap
def time_A : Nat := 4
def time_B : Nat := 5
def time_C : Nat := 6

-- Prove that the LCM of 4, 5, and 6 is 60
theorem lcm_of_times : Nat.lcm time_A (Nat.lcm time_B time_C) = 60 := by
  sorry

end lcm_of_times_l237_237728


namespace number_of_customers_trimmed_l237_237027

-- Definitions based on the conditions
def total_sounds : ℕ := 60
def sounds_per_person : ℕ := 20

-- Statement to prove
theorem number_of_customers_trimmed :
  ∃ n : ℕ, n * sounds_per_person = total_sounds ∧ n = 3 :=
sorry

end number_of_customers_trimmed_l237_237027


namespace corn_increase_factor_l237_237800

noncomputable def field_area : ℝ := 1

-- Let x be the remaining part of the field
variable (x : ℝ)

-- First condition: if the remaining part is fully planted with millet
-- Millet will occupy half of the field
axiom condition1 : (field_area - x) + x = field_area / 2

-- Second condition: if the remaining part x is equally divided between oats and corn
-- Oats will occupy half of the field
axiom condition2 : (field_area - x) + 0.5 * x = field_area / 2

-- Prove the factor by which the amount of corn increases
theorem corn_increase_factor : (0.5 * x + x) / (0.5 * x / 2) = 3 :=
by
  sorry

end corn_increase_factor_l237_237800


namespace willowbrook_team_combinations_l237_237002

theorem willowbrook_team_combinations :
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  team_count = 100 :=
by
  let girls := 5
  let boys := 5
  let choose_three (n : ℕ) := n.choose 3
  let team_count := choose_three girls * choose_three boys
  have h1 : choose_three girls = 10 := by sorry
  have h2 : choose_three boys = 10 := by sorry
  have h3 : team_count = 10 * 10 := by sorry
  exact h3

end willowbrook_team_combinations_l237_237002


namespace hiking_trip_time_l237_237512

noncomputable def R_up : ℝ := 7
noncomputable def R_down : ℝ := 1.5 * R_up
noncomputable def Distance_down : ℝ := 21
noncomputable def T_down : ℝ := Distance_down / R_down
noncomputable def T_up : ℝ := T_down

theorem hiking_trip_time :
  T_up = 2 := by
      sorry

end hiking_trip_time_l237_237512


namespace bathing_suits_per_model_l237_237751

def models : ℕ := 6
def evening_wear_sets_per_model : ℕ := 3
def time_per_trip_minutes : ℕ := 2
def total_show_time_minutes : ℕ := 60

theorem bathing_suits_per_model : (total_show_time_minutes - (models * evening_wear_sets_per_model * time_per_trip_minutes)) / (time_per_trip_minutes * models) = 2 :=
by
  sorry

end bathing_suits_per_model_l237_237751


namespace domain_g_l237_237332

noncomputable def f : ℝ → ℝ := sorry  -- f is a real-valued function

theorem domain_g:
  (∀ x, x ∈ [-2, 4] ↔ f x ∈ [-2, 4]) →  -- The domain of f(x) is [-2, 4]
  (∀ x, x ∈ [-2, 2] ↔ (f x + f (-x)) ∈ [-2, 2]) :=  -- The domain of g(x) = f(x) + f(-x) is [-2, 2]
by
  intros h
  sorry

end domain_g_l237_237332


namespace balance_two_diamonds_three_bullets_l237_237169

-- Define the variables
variables (a b c : ℝ)

-- Define the conditions as hypotheses
def condition1 : Prop := 3 * a + b = 9 * c
def condition2 : Prop := a = b + c

-- Goal is to prove two diamonds (2 * b) balance three bullets (3 * c)
theorem balance_two_diamonds_three_bullets (h1 : condition1 a b c) (h2 : condition2 a b c) : 
  2 * b = 3 * c := 
by 
  sorry

end balance_two_diamonds_three_bullets_l237_237169


namespace seashells_count_l237_237882

theorem seashells_count {s : ℕ} (h : s + 6 = 25) : s = 19 :=
by
  sorry

end seashells_count_l237_237882


namespace polynomial_remainder_l237_237837

noncomputable def divisionRemainder (f g : Polynomial ℝ) : Polynomial ℝ := Polynomial.modByMonic f g

theorem polynomial_remainder :
  divisionRemainder (Polynomial.X ^ 5 + 2) (Polynomial.X ^ 2 - 4 * Polynomial.X + 7) = -29 * Polynomial.X - 54 :=
by
  sorry

end polynomial_remainder_l237_237837


namespace original_digit_sum_six_and_product_is_1008_l237_237854

theorem original_digit_sum_six_and_product_is_1008 (x : ℕ) :
  (2 ∣ x / 10) → (4 ∣ x / 10) → 
  (x % 10 + (x / 10) = 6) →
  ((x % 10) * 10 + (x / 10)) * ((x / 10) * 10 + (x % 10)) = 1008 →
  x = 42 ∨ x = 24 :=
by
  intro h1 h2 h3 h4
  sorry


end original_digit_sum_six_and_product_is_1008_l237_237854


namespace hyperbola_eccentricity_l237_237215

theorem hyperbola_eccentricity (a b c e : ℝ) (ha : a > 0) (hb : b > 0)
  (h1 : ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1 → (x, y) ≠ (3, -4))
  (h2 : b / a = 4 / 3)
  (h3 : b^2 = c^2 - a^2)
  (h4 : c / a = e):
  e = 5 / 3 :=
by
  sorry

end hyperbola_eccentricity_l237_237215


namespace inheritance_problem_l237_237582

def wifeAmounts (K J M : ℝ) : Prop :=
  K + J + M = 396 ∧
  J = K + 10 ∧
  M = J + 10

def husbandAmounts (wifeAmount : ℝ) (husbandMultiplier : ℝ := 1) : ℝ :=
  husbandMultiplier * wifeAmount

theorem inheritance_problem (K J M : ℝ)
  (h1 : wifeAmounts K J M)
  : ∃ wifeOf : String → String,
    wifeOf "John Smith" = "Katherine" ∧
    wifeOf "Henry Snooks" = "Jane" ∧
    wifeOf "Tom Crow" = "Mary" ∧
    husbandAmounts K = K ∧
    husbandAmounts J 1.5 = 1.5 * J ∧
    husbandAmounts M 2 = 2 * M :=
by 
  sorry

end inheritance_problem_l237_237582


namespace evaluate_trig_expression_l237_237496

theorem evaluate_trig_expression (α : ℝ) (h : Real.tan α = -4/3) : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1 / 7 :=
by
  sorry

end evaluate_trig_expression_l237_237496


namespace similar_triangle_perimeter_l237_237051

theorem similar_triangle_perimeter 
  (a b c : ℝ) (a_sim : ℝ)
  (h1 : a = b) (h2 : b = c)
  (h3 : a = 15) (h4 : a_sim = 45)
  (h5 : a_sim / a = 3) :
  a_sim + a_sim + a_sim = 135 :=
by
  sorry

end similar_triangle_perimeter_l237_237051


namespace driver_net_hourly_rate_l237_237883

theorem driver_net_hourly_rate
  (hours : ℝ) (speed : ℝ) (efficiency : ℝ) (cost_per_gallon : ℝ) (compensation_rate : ℝ)
  (h1 : hours = 3)
  (h2 : speed = 50)
  (h3 : efficiency = 25)
  (h4 : cost_per_gallon = 2.50)
  (h5 : compensation_rate = 0.60)
  :
  ((compensation_rate * (speed * hours) - (cost_per_gallon * (speed * hours / efficiency))) / hours) = 25 :=
sorry

end driver_net_hourly_rate_l237_237883


namespace sum_of_coordinates_l237_237746

variable (f : ℝ → ℝ)

/-- Given that the point (2, 3) is on the graph of y = f(x) / 3,
    show that (9, 2/3) must be on the graph of y = f⁻¹(x) / 3 and the
    sum of its coordinates is 29/3. -/
theorem sum_of_coordinates (h : 3 = f 2 / 3) : (9 : ℝ) + (2 / 3 : ℝ) = 29 / 3 :=
by
  have h₁ : f 2 = 9 := by
    linarith
    
  have h₂ : f⁻¹ 9 = 2 := by
    -- We assume that f has an inverse and it is well-defined
    sorry

  have point_on_graph : (9, (2 / 3)) ∈ { p : ℝ × ℝ | p.2 = f⁻¹ p.1 / 3 } := by
    sorry

  show 9 + 2 / 3 = 29 / 3
  norm_num

end sum_of_coordinates_l237_237746


namespace y_paid_per_week_l237_237527

variable (x y z : ℝ)

-- Conditions
axiom h1 : x + y + z = 900
axiom h2 : x = 1.2 * y
axiom h3 : z = 0.8 * y

-- Theorem to prove
theorem y_paid_per_week : y = 300 := by
  sorry

end y_paid_per_week_l237_237527


namespace find_abcdef_l237_237040

def repeating_decimal_to_fraction_abcd (a b c d : ℕ) : ℚ :=
  (1000 * a + 100 * b + 10 * c + d) / 9999

def repeating_decimal_to_fraction_abcdef (a b c d e f : ℕ) : ℚ :=
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f) / 999999

theorem find_abcdef :
  ∀ a b c d e f : ℕ,
  0 ≤ a ∧ a ≤ 9 ∧
  0 ≤ b ∧ b ≤ 9 ∧
  0 ≤ c ∧ c ≤ 9 ∧
  0 ≤ d ∧ d ≤ 9 ∧
  0 ≤ e ∧ e ≤ 9 ∧
  0 ≤ f ∧ f ≤ 9 ∧
  (repeating_decimal_to_fraction_abcd a b c d + repeating_decimal_to_fraction_abcdef a b c d e f = 49 / 999) →
  (100000 * a + 10000 * b + 1000 * c + 100 * d + 10 * e + f = 490) :=
by
  repeat {sorry}

end find_abcdef_l237_237040


namespace sandy_money_taken_l237_237172

-- Condition: Let T be the total money Sandy took for shopping, and it is known that 70% * T = $224
variable (T : ℝ)
axiom h : 0.70 * T = 224

-- Theorem to prove: T is 320
theorem sandy_money_taken : T = 320 :=
by 
  sorry

end sandy_money_taken_l237_237172


namespace simplify_sqrt_neg2_squared_l237_237181

theorem simplify_sqrt_neg2_squared : 
  Real.sqrt ((-2 : ℝ)^2) = 2 := 
by
  sorry

end simplify_sqrt_neg2_squared_l237_237181


namespace ocean_depth_l237_237689

/-
  Problem:
  Determine the depth of the ocean at the current location of the ship.
  
  Given conditions:
  - The signal sent by the echo sounder was received after 5 seconds.
  - The speed of sound in water is 1.5 km/s.

  Correct answer to prove:
  - The depth of the ocean is 3750 meters.
-/

theorem ocean_depth
  (v : ℝ) (t : ℝ) (depth : ℝ) 
  (hv : v = 1500) 
  (ht : t = 5) 
  (hdepth : depth = 3750) :
  depth = (v * t) / 2 :=
sorry

end ocean_depth_l237_237689


namespace gcd_21_eq_7_count_l237_237605

theorem gcd_21_eq_7_count : Nat.card {n : Fin 200 // Nat.gcd 21 n = 7} = 19 := 
by
  sorry

end gcd_21_eq_7_count_l237_237605


namespace katie_clock_l237_237710

theorem katie_clock (t_clock t_actual : ℕ) :
  t_clock = 540 →
  t_actual = (540 * 60) / 37 →
  8 * 60 + 875 = 22 * 60 + 36 :=
by
  intros h1 h2
  have h3 : 875 = (540 * 60 / 37) := sorry
  have h4 : 8 * 60 + 875 = 480 + 875 := sorry
  have h5 : 480 + 875 = 22 * 60 + 36 := sorry
  exact h5

end katie_clock_l237_237710


namespace subtraction_result_l237_237363

theorem subtraction_result :
  let x := 567.89
  let y := 123.45
  (x - y) = 444.44 :=
by
  sorry

end subtraction_result_l237_237363


namespace total_apples_l237_237113

theorem total_apples (apples_per_person : ℕ) (num_people : ℕ) (h1 : apples_per_person = 25) (h2 : num_people = 6) : apples_per_person * num_people = 150 := by
  sorry

end total_apples_l237_237113


namespace greatest_int_value_not_satisfy_condition_l237_237570

/--
For the inequality 8 - 6x > 26, the greatest integer value 
of x that satisfies this is -4.
-/
theorem greatest_int_value (x : ℤ) : 8 - 6 * x > 26 → x ≤ -4 :=
by sorry

theorem not_satisfy_condition (x : ℤ) : x > -4 → ¬ (8 - 6 * x > 26) :=
by sorry

end greatest_int_value_not_satisfy_condition_l237_237570


namespace total_earnings_first_two_weeks_l237_237513

-- Conditions
variable (x : ℝ)  -- Xenia's hourly wage
variable (earnings_first_week : ℝ := 12 * x)  -- Earnings in the first week
variable (earnings_second_week : ℝ := 20 * x)  -- Earnings in the second week

-- Xenia earned $36 more in the second week than in the first
axiom h1 : earnings_second_week = earnings_first_week + 36

-- Proof statement
theorem total_earnings_first_two_weeks : earnings_first_week + earnings_second_week = 144 := by
  -- Proof is omitted
  sorry

end total_earnings_first_two_weeks_l237_237513


namespace find_t_l237_237076

variable (g V V0 c S t : ℝ)
variable (h1 : V = g * t + V0 + c)
variable (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2)

theorem find_t
  (h1 : V = g * t + V0 + c)
  (h2 : S = (1/2) * g * t^2 + V0 * t + c * t^2) :
  t = 2 * S / (V + V0 - c) :=
sorry

end find_t_l237_237076


namespace min_value_of_expression_l237_237131

noncomputable def expression (x : ℝ) : ℝ := (15 - x) * (12 - x) * (15 + x) * (12 + x)

theorem min_value_of_expression :
  ∃ x : ℝ, (expression x) = -1640.25 :=
sorry

end min_value_of_expression_l237_237131


namespace necessary_conditions_l237_237754

theorem necessary_conditions (a b c d e : ℝ) (h : (a + b + e) / (b + c) = (c + d + e) / (d + a)) :
  a = c ∨ a + b + c + d + e = 0 :=
by
  sorry

end necessary_conditions_l237_237754


namespace problem_proof_l237_237798

noncomputable def triangle_expression (a b c : ℝ) (A B C : ℝ) : ℝ :=
  b^2 * (Real.cos (C / 2))^2 + c^2 * (Real.cos (B / 2))^2 + 
  2 * b * c * Real.cos (B / 2) * Real.cos (C / 2) * Real.sin (A / 2)

theorem problem_proof (a b c A B C : ℝ) (h1 : a + b + c = 16) : 
  triangle_expression a b c A B C = 64 := 
sorry

end problem_proof_l237_237798


namespace inequality_reciprocal_of_negatives_l237_237651

theorem inequality_reciprocal_of_negatives (a b : ℝ) (ha : a < b) (hb : b < 0) : (1 / a) > (1 / b) :=
sorry

end inequality_reciprocal_of_negatives_l237_237651


namespace cab_time_l237_237415

theorem cab_time (d t : ℝ) (v : ℝ := d / t)
    (v1 : ℝ := (5 / 6) * v)
    (t1 : ℝ := d / v1)
    (v2 : ℝ := (2 / 3) * v)
    (t2 : ℝ := d / v2)
    (T : ℝ := t1 + t2)
    (delay : ℝ := 5) :
    let total_time := 2 * t + delay
    t * d ≠ 0 → T = total_time → t = 50 / 7 := by
    sorry

end cab_time_l237_237415


namespace bigger_part_of_sum_and_linear_combination_l237_237119

theorem bigger_part_of_sum_and_linear_combination (x y : ℕ) 
  (h1 : x + y = 24) 
  (h2 : 7 * x + 5 * y = 146) : x = 13 :=
by 
  sorry

end bigger_part_of_sum_and_linear_combination_l237_237119


namespace find_group_2018_l237_237407

theorem find_group_2018 :
  ∃ n : ℕ, 2 ≤ n ∧ 2018 ≤ 2 * n * (n + 1) ∧ 2018 > 2 * (n - 1) * n :=
by
  sorry

end find_group_2018_l237_237407


namespace min_omega_sin_two_max_l237_237082

theorem min_omega_sin_two_max (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → ∃ k : ℤ, (ω * x = (2 + 2 * k) * π)) →
  ∃ ω_min : ℝ, ω_min = 4 * π :=
by
  sorry

end min_omega_sin_two_max_l237_237082


namespace salary_increase_l237_237339

theorem salary_increase (new_salary increase : ℝ) (h_new : new_salary = 25000) (h_inc : increase = 5000) : 
  ((increase / (new_salary - increase)) * 100) = 25 :=
by
  -- We will write the proof to satisfy the requirement, but it is currently left out as per the instructions.
  sorry

end salary_increase_l237_237339


namespace candy_bag_division_l237_237910

theorem candy_bag_division (total_candy bags_candy : ℕ) (h1 : total_candy = 42) (h2 : bags_candy = 21) : 
  total_candy / bags_candy = 2 := 
by
  sorry

end candy_bag_division_l237_237910


namespace max_value_of_x_l237_237085

theorem max_value_of_x (x y : ℝ) (h : x^2 + y^2 = 18 * x + 20 * y) : x ≤ 9 + Real.sqrt 181 :=
by
  sorry

end max_value_of_x_l237_237085


namespace find_z_l237_237561

theorem find_z (x : ℕ) (z : ℚ) (h1 : x = 103)
               (h2 : x^3 * z - 3 * x^2 * z + 2 * x * z = 208170) 
               : z = 5 / 265 := 
by 
  sorry

end find_z_l237_237561


namespace tan_identity_at_30_degrees_l237_237684

theorem tan_identity_at_30_degrees :
  let A := 30
  let B := 30
  let deg_to_rad := pi / 180
  let tan := fun x : ℝ => Real.tan (x * deg_to_rad)
  (1 + tan A) * (1 + tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end tan_identity_at_30_degrees_l237_237684


namespace find_interval_for_inequality_l237_237114

open Set

theorem find_interval_for_inequality :
  {x : ℝ | (1 / (x^2 + 2) > 4 / x + 21 / 10)} = Ioo (-2 : ℝ) (0 : ℝ) := 
sorry

end find_interval_for_inequality_l237_237114


namespace remainder_when_divided_by_2_l237_237457

-- Define the main parameters
def n : ℕ := sorry  -- n is a positive integer
def k : ℤ := sorry  -- Provided for modular arithmetic context

-- Conditions
axiom h1 : n > 0  -- n is a positive integer
axiom h2 : (n + 1) % 6 = 4  -- When n + 1 is divided by 6, the remainder is 4

-- The theorem statement
theorem remainder_when_divided_by_2 : n % 2 = 1 :=
by
  sorry

end remainder_when_divided_by_2_l237_237457


namespace wade_total_spent_l237_237160

def sandwich_cost : ℕ := 6
def drink_cost : ℕ := 4
def num_sandwiches : ℕ := 3
def num_drinks : ℕ := 2

def total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_drinks * drink_cost)

theorem wade_total_spent : total_cost = 26 := by
  sorry

end wade_total_spent_l237_237160


namespace which_is_right_triangle_l237_237852

-- Definitions for each group of numbers
def sides_A := (1, 2, 3)
def sides_B := (3, 4, 5)
def sides_C := (4, 5, 6)
def sides_D := (7, 8, 9)

-- Definition of a condition for right triangle using the converse of the Pythagorean theorem
def is_right_triangle (a b c: ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem which_is_right_triangle :
    ¬is_right_triangle 1 2 3 ∧
    ¬is_right_triangle 4 5 6 ∧
    ¬is_right_triangle 7 8 9 ∧
    is_right_triangle 3 4 5 :=
by
  sorry

end which_is_right_triangle_l237_237852


namespace wage_percent_change_l237_237490

-- Definitions based on given conditions
def initial_wage (W : ℝ) := W
def first_decrease (W : ℝ) := 0.60 * W
def first_increase (W : ℝ) := 0.78 * W
def second_decrease (W : ℝ) := 0.624 * W
def second_increase (W : ℝ) := 0.6864 * W

-- Lean theorem statement to prove overall percent change
theorem wage_percent_change : ∀ (W : ℝ), 
  ((second_increase (second_decrease (first_increase (first_decrease W))) - initial_wage W) / initial_wage W) * 100 = -31.36 :=
by sorry

end wage_percent_change_l237_237490


namespace probability_of_three_heads_in_eight_tosses_l237_237875

theorem probability_of_three_heads_in_eight_tosses :
  (∃ (p : ℚ), p = 7 / 32) :=
by 
  sorry

end probability_of_three_heads_in_eight_tosses_l237_237875


namespace distance_to_fourth_side_l237_237386

theorem distance_to_fourth_side (s : ℕ) (d1 d2 d3 : ℕ) (x : ℕ) 
  (cond1 : d1 = 4) (cond2 : d2 = 7) (cond3 : d3 = 12)
  (h : d1 + d2 + d3 + x = s) : x = 9 ∨ x = 15 :=
  sorry

end distance_to_fourth_side_l237_237386


namespace range_of_a_l237_237259

/-- Given that the point (1, 1) is located inside the circle (x - a)^2 + (y + a)^2 = 4, 
    proving that the range of values for a is -1 < a < 1. -/
theorem range_of_a (a : ℝ) : 
  ((1 - a)^2 + (1 + a)^2 < 4) → 
  (-1 < a ∧ a < 1) :=
by
  intro h
  sorry

end range_of_a_l237_237259


namespace simplify_expression_l237_237219

theorem simplify_expression (y : ℝ) : (5 * y) ^ 3 + (4 * y) * (y ^ 2) = 129 * (y ^ 3) := by
  sorry

end simplify_expression_l237_237219


namespace luna_total_monthly_budget_l237_237706

theorem luna_total_monthly_budget
  (H F phone_bill : ℝ)
  (h1 : F = 0.60 * H)
  (h2 : H + F = 240)
  (h3 : phone_bill = 0.10 * F) :
  H + F + phone_bill = 249 :=
by sorry

end luna_total_monthly_budget_l237_237706


namespace find_S_l237_237611

theorem find_S (x y : ℝ) (h : x + y = 4) : 
  ∃ S, (∀ x y, x + y = 4 → 3*x^2 + y^2 = 12) → S = 6 := 
by 
  sorry

end find_S_l237_237611


namespace exp_fn_max_min_diff_l237_237164

theorem exp_fn_max_min_diff (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1) :
  (max (a^1) (a^0) - min (a^1) (a^0)) = 1 / 2 → (a = 1 / 2 ∨ a = 3 / 2) :=
by
  sorry

end exp_fn_max_min_diff_l237_237164


namespace find_missing_employee_l237_237945

-- Definitions based on the problem context
def employee_numbers : List Nat := List.range (52)
def sample_size := 4

-- The given conditions, stating that these employees are in the sample
def in_sample (x : Nat) : Prop := x = 6 ∨ x = 32 ∨ x = 45 ∨ x = 19

-- Define systematic sampling method condition
def systematic_sample (nums : List Nat) (size interval : Nat) : Prop :=
  nums = List.map (fun i => 6 + i * interval % 52) (List.range size)

-- The employees in the sample must include 6
def start_num := 6
def interval := 13
def expected_sample := [6, 19, 32, 45]

-- The Lean theorem we need to prove
theorem find_missing_employee :
  systematic_sample expected_sample sample_size interval ∧
  in_sample 6 ∧ in_sample 32 ∧ in_sample 45 →
  in_sample 19 :=
by
  sorry

end find_missing_employee_l237_237945


namespace conor_chop_eggplants_l237_237236

theorem conor_chop_eggplants (E : ℕ) 
  (condition1 : E + 9 + 8 = (E + 17))
  (condition2 : 4 * (E + 9 + 8) = 116) :
  E = 12 :=
by {
  sorry
}

end conor_chop_eggplants_l237_237236


namespace InfinitePairsExist_l237_237572

theorem InfinitePairsExist (a b : ℕ) : (∀ n : ℕ, ∃ a b : ℕ, a ∣ b^2 + 1 ∧ b ∣ a^2 + 1) :=
sorry

end InfinitePairsExist_l237_237572


namespace limit_log_div_x_alpha_l237_237971

open Real

theorem limit_log_div_x_alpha (α : ℝ) (hα : α > 0) :
  (Filter.Tendsto (fun x => (log x) / (x^α)) Filter.atTop (nhds 0)) :=
by
  sorry

end limit_log_div_x_alpha_l237_237971


namespace find_x_l237_237388

theorem find_x (x y : ℤ) (h1 : x > y) (h2 : y > 0) (h3 : x + y + x * y = 101) : x = 50 :=
sorry

end find_x_l237_237388


namespace vase_net_gain_l237_237622

theorem vase_net_gain 
  (selling_price : ℝ)
  (V1_cost : ℝ)
  (V2_cost : ℝ)
  (hyp1 : selling_price = 2.50)
  (hyp2 : 1.25 * V1_cost = selling_price)
  (hyp3 : 0.85 * V2_cost = selling_price) :
  (selling_price + selling_price) - (V1_cost + V2_cost) = 0.06 := 
by 
  sorry

end vase_net_gain_l237_237622


namespace average_cookies_per_package_l237_237879

def cookies_per_package : List ℕ := [9, 11, 14, 12, 0, 18, 15, 16, 19, 21]

theorem average_cookies_per_package :
  (cookies_per_package.sum : ℚ) / cookies_per_package.length = 13.5 := by
  sorry

end average_cookies_per_package_l237_237879


namespace find_angle_A_l237_237486

theorem find_angle_A (a b : ℝ) (sin_B : ℝ) (ha : a = 3) (hb : b = 4) (hsinB : sin_B = 2/3) :
  ∃ A : ℝ, A = π / 6 :=
by
  sorry

end find_angle_A_l237_237486


namespace square_of_volume_of_rect_box_l237_237725

theorem square_of_volume_of_rect_box (x y z : ℝ) 
  (h1 : x * y = 15) 
  (h2 : y * z = 18) 
  (h3 : z * x = 10) : (x * y * z) ^ 2 = 2700 :=
sorry

end square_of_volume_of_rect_box_l237_237725


namespace solve_abs_eq_zero_l237_237969

theorem solve_abs_eq_zero : ∃ x : ℝ, |5 * x - 3| = 0 ↔ x = 3 / 5 :=
by
  sorry

end solve_abs_eq_zero_l237_237969


namespace no_perfect_square_in_seq_l237_237958

noncomputable def seq : ℕ → ℕ
| 0       => 2
| 1       => 7
| (n + 2) => 4 * seq (n + 1) - seq n

theorem no_perfect_square_in_seq :
  ¬ ∃ (n : ℕ), ∃ (k : ℕ), (seq n) = k * k :=
sorry

end no_perfect_square_in_seq_l237_237958


namespace correct_assignment_statement_l237_237014

def is_assignment_statement (stmt : String) : Prop :=
  stmt = "a = 2a"

theorem correct_assignment_statement : is_assignment_statement "a = 2a" :=
by
  sorry

end correct_assignment_statement_l237_237014


namespace symmetry_axis_is_neg_pi_over_12_l237_237607

noncomputable def symmetry_axis_of_sine_function : Prop :=
  ∃ k : ℤ, ∀ x : ℝ, (3 * x + 3 * Real.pi / 4 = Real.pi / 2 + k * Real.pi) ↔ (x = - Real.pi / 12 + k * Real.pi / 3)

theorem symmetry_axis_is_neg_pi_over_12 : symmetry_axis_of_sine_function := sorry

end symmetry_axis_is_neg_pi_over_12_l237_237607


namespace find_S_l237_237810

theorem find_S (a b : ℝ) (R : ℝ) (S : ℝ)
  (h1 : a + b = R) 
  (h2 : a^2 + b^2 = 12)
  (h3 : R = 2)
  (h4 : S = a^3 + b^3) : S = 32 :=
by
  sorry

end find_S_l237_237810


namespace scooter_cost_l237_237103

variable (saved needed total_cost : ℕ)

-- The conditions given in the problem
def greg_saved_57 : saved = 57 := sorry
def greg_needs_33_more : needed = 33 := sorry

-- The proof goal
theorem scooter_cost (h1 : saved = 57) (h2 : needed = 33) :
  total_cost = saved + needed → total_cost = 90 := by
  sorry

end scooter_cost_l237_237103


namespace game_cost_l237_237019

theorem game_cost (initial_money : ℕ) (toys_count : ℕ) (toy_price : ℕ) (left_money : ℕ) : 
  initial_money = 63 ∧ toys_count = 5 ∧ toy_price = 3 ∧ left_money = 15 → 
  (initial_money - left_money = 48) :=
by
  sorry

end game_cost_l237_237019


namespace lesser_fraction_l237_237125

theorem lesser_fraction (x y : ℚ) (h₁ : x + y = 3/4) (h₂ : x * y = 1/8) : min x y = 1/4 :=
by
  -- The proof would go here
  sorry

end lesser_fraction_l237_237125


namespace largest_multiple_of_15_less_than_500_l237_237075

theorem largest_multiple_of_15_less_than_500 : ∃ (n : ℕ), 15 * n < 500 ∧ 15 * n ≥ 15 * (33 - 1) + 1 := by
  sorry

end largest_multiple_of_15_less_than_500_l237_237075


namespace greatest_common_divisor_sum_arithmetic_sequence_l237_237714

theorem greatest_common_divisor_sum_arithmetic_sequence (x c : ℕ) (hx : 0 < x) (hc : 0 < c) :
  ∃ d : ℕ, d = 15 ∧ ∀ (n : ℕ), n = 15 → ∀ k : ℕ, k = 15 ∧ 15 ∣ (15 * (x + 7 * c)) :=
by
  sorry

end greatest_common_divisor_sum_arithmetic_sequence_l237_237714


namespace diagonal_AC_possibilities_l237_237150

/-
In a quadrilateral with sides AB, BC, CD, and DA, the length of diagonal AC must 
satisfy the inequalities determined by the triangle inequalities for triangles 
ABC and CDA. Prove the number of different whole numbers that could be the 
length of diagonal AC is 13.
-/

def number_of_whole_numbers_AC (AB BC CD DA : ℕ) : ℕ :=
  if 6 < AB ∧ AB < 20 then 19 - 7 + 1 else sorry

theorem diagonal_AC_possibilities : number_of_whole_numbers_AC 7 13 15 10 = 13 :=
  by
    sorry

end diagonal_AC_possibilities_l237_237150


namespace trig_identity_l237_237566

noncomputable def sin_40 := Real.sin (40 * Real.pi / 180)
noncomputable def tan_10 := Real.tan (10 * Real.pi / 180)
noncomputable def sqrt_3 := Real.sqrt 3

theorem trig_identity : sin_40 * (tan_10 - sqrt_3) = -1 := by
  sorry

end trig_identity_l237_237566


namespace find_x_l237_237551

theorem find_x (x : ℝ) (h : (x * 74) / 30 = 1938.8) : x = 786 := by
  sorry

end find_x_l237_237551


namespace intersection_M_N_l237_237658

def M : Set ℕ := { y | y < 6 }
def N : Set ℕ := {2, 3, 6}

theorem intersection_M_N : M ∩ N = {2, 3} := by
  sorry

end intersection_M_N_l237_237658


namespace max_b_squared_l237_237370

theorem max_b_squared (a b : ℤ) (h : (a + b) * (a + b) + a * (a + b) + b = 0) : b^2 ≤ 81 :=
sorry

end max_b_squared_l237_237370


namespace percent_less_than_m_plus_d_l237_237730

-- Define the given conditions
variables (m d : ℝ) (distribution : ℝ → ℝ)

-- Assume the distribution is symmetric about the mean m
axiom symmetric_distribution :
  ∀ x, distribution (m + x) = distribution (m - x)

-- 84 percent of the distribution lies within one standard deviation d of the mean
axiom within_one_sd :
  ∫ x in -d..d, distribution (m + x) = 0.84

-- The goal is to prove that 42 percent of the distribution is less than m + d
theorem percent_less_than_m_plus_d : 
  ( ∫ x in -d..0, distribution (m + x) ) = 0.42 :=
by 
  sorry

end percent_less_than_m_plus_d_l237_237730


namespace distinct_arrangements_of_beads_l237_237537

noncomputable def factorial (n : Nat) : Nat := if h : n = 0 then 1 else n * factorial (n - 1)

theorem distinct_arrangements_of_beads : 
  ∃ (arrangements : Nat), arrangements = factorial 8 / (8 * 2) ∧ arrangements = 2520 := 
by
  -- Sorry to skip the proof, only requiring the statement.
  sorry

end distinct_arrangements_of_beads_l237_237537
