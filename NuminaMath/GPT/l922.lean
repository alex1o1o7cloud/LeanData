import Mathlib

namespace exists_prime_and_positive_integer_l922_92227

theorem exists_prime_and_positive_integer (a : ℕ) (h : a = 9) : 
  ∃ (p : ℕ) (hp : Nat.Prime p) (b : ℕ) (hb : b ≥ 2), (a^p - a) / p = b^2 := 
  by
  sorry

end exists_prime_and_positive_integer_l922_92227


namespace final_amount_l922_92226

-- Definitions for the initial amount, price per pound, and quantity purchased.
def initial_amount : ℕ := 20
def price_per_pound : ℕ := 2
def quantity_purchased : ℕ := 3

-- Formalizing the statement
theorem final_amount (A P Q : ℕ) (hA : A = initial_amount) (hP : P = price_per_pound) (hQ : Q = quantity_purchased) :
  A - P * Q = 14 :=
by
  sorry

end final_amount_l922_92226


namespace g_min_value_l922_92221

noncomputable def g (x : ℝ) : ℝ :=
  x + x / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem g_min_value (x : ℝ) (h : x > 0) : g x >= 6 :=
sorry

end g_min_value_l922_92221


namespace vehicle_capacity_rental_plans_l922_92272

variables (a b x y : ℕ)

/-- Conditions -/
axiom cond1 : 2*x + y = 11
axiom cond2 : x + 2*y = 13

/-- Resulting capacities for each vehicle type -/
theorem vehicle_capacity : 
  x = 3 ∧ y = 5 :=
by
  sorry

/-- Rental plans for transporting 33 tons of drugs -/
theorem rental_plans :
  3*a + 5*b = 33 ∧ ((a = 6 ∧ b = 3) ∨ (a = 1 ∧ b = 6)) :=
by
  sorry

end vehicle_capacity_rental_plans_l922_92272


namespace player_winning_strategy_l922_92206

-- Define the game conditions
def Sn (n : ℕ) : Type := Equiv.Perm (Fin n)

def game_condition (n : ℕ) : Prop :=
  n > 1 ∧ (∀ G : Set (Sn n), ∃ x : Sn n, x ∈ G → G ≠ (Set.univ : Set (Sn n)))

-- Statement of the proof problem
theorem player_winning_strategy (n : ℕ) (hn : n > 1) : 
  ((n = 2 ∨ n = 3) → (∃ strategyA : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyA x x)) ∧ 
  ((n ≥ 4 ∧ n % 2 = 1) → (∃ strategyB : Sn n → (Sn n → Prop), ∀ x : Sn n, strategyB x x)) :=
by
  sorry

end player_winning_strategy_l922_92206


namespace initial_marbles_l922_92273

-- Define the conditions as constants
def marbles_given_to_Juan : ℕ := 73
def marbles_left_with_Connie : ℕ := 70

-- Prove that Connie initially had 143 marbles
theorem initial_marbles (initial_marbles : ℕ) :
  initial_marbles = marbles_given_to_Juan + marbles_left_with_Connie → 
  initial_marbles = 143 :=
by
  intro h
  rw [h]
  rfl

end initial_marbles_l922_92273


namespace Jason_earned_60_dollars_l922_92234

-- Define initial and final amounts of money
variable (Jason_initial Jason_final : ℕ)

-- State the assumption about Jason's initial and final amounts of money
variable (h_initial : Jason_initial = 3) (h_final : Jason_final = 63)

-- Define the amount of money Jason earned
def Jason_earn := Jason_final - Jason_initial

-- Prove that Jason earned 60 dollars by delivering newspapers
theorem Jason_earned_60_dollars : Jason_earn Jason_initial Jason_final = 60 := by
  sorry

end Jason_earned_60_dollars_l922_92234


namespace number_of_valid_permutations_l922_92244

noncomputable def count_valid_permutations : Nat :=
  let multiples_of_77 := [154, 231, 308, 385, 462, 539, 616, 693, 770, 847, 924]
  let total_count := multiples_of_77.foldl (fun acc x =>
    if x == 770 then
      acc + 3
    else if x == 308 then
      acc + 6 - 2
    else
      acc + 6) 0
  total_count

theorem number_of_valid_permutations : count_valid_permutations = 61 :=
  sorry

end number_of_valid_permutations_l922_92244


namespace simplify_exponent_multiplication_l922_92205

theorem simplify_exponent_multiplication (x : ℝ) : x^5 * x^3 = x^8 :=
by sorry

end simplify_exponent_multiplication_l922_92205


namespace bird_families_to_Asia_l922_92258

theorem bird_families_to_Asia (total_families initial_families left_families went_to_Africa went_to_Asia: ℕ) 
  (h1 : total_families = 85) 
  (h2 : went_to_Africa = 23) 
  (h3 : left_families = 25) 
  (h4 : went_to_Asia = total_families - left_families - went_to_Africa) 
  : went_to_Asia = 37 := 
by 
  rw [h1, h2, h3] at h4 
  simp at h4 
  exact h4

end bird_families_to_Asia_l922_92258


namespace at_least_one_pass_l922_92293

variable (n : ℕ) (p : ℝ)

theorem at_least_one_pass (h_p_range : 0 < p ∧ p < 1) :
  (1 - (1 - p) ^ n) = 1 - (1 - p) ^ n :=
sorry

end at_least_one_pass_l922_92293


namespace base_salary_at_least_l922_92215

-- Definitions for the conditions.
def previous_salary : ℕ := 75000
def commission_rate : ℚ := 0.15
def sale_value : ℕ := 750
def min_sales_required : ℚ := 266.67

-- Calculate the commission per sale
def commission_per_sale : ℚ := commission_rate * sale_value

-- Calculate the total commission for the minimum sales required
def total_commission : ℚ := min_sales_required * commission_per_sale

-- The base salary S required to not lose money
theorem base_salary_at_least (S : ℚ) : S + total_commission ≥ previous_salary ↔ S ≥ 45000 := 
by
  -- Use sorry to skip the proof
  sorry

end base_salary_at_least_l922_92215


namespace problem_statement_l922_92202

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 4))

theorem problem_statement :
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), f x ∈ Set.Icc (-Real.sqrt 2 / 2) 1) ∧
  (f (Real.pi / 2) = -Real.sqrt 2 / 2) ∧
  (∀ k : ℤ, ∀ x ∈ Set.Icc (k * Real.pi - 3 * Real.pi / 8) (k * Real.pi + Real.pi / 8), 
    ∃ δ > 0, ∀ y ∈ Set.Ioc x (x + δ), f x < f y) :=
by {
  sorry
}

end problem_statement_l922_92202


namespace minimum_embrasure_length_l922_92216

theorem minimum_embrasure_length : ∀ (s : ℝ), 
  (∀ t : ℝ, (∃ k : ℤ, t = k / 2 ∧ k % 2 = 0) ∨ (∃ k : ℤ, t = (k + 1) / 2 ∧ k % 2 = 1)) → 
  (∃ z : ℝ, z = 2 / 3) := 
sorry

end minimum_embrasure_length_l922_92216


namespace abs_less_than_2_sufficient_but_not_necessary_l922_92290

theorem abs_less_than_2_sufficient_but_not_necessary (x : ℝ) : 
  (|x| < 2 → (x^2 - x - 6 < 0)) ∧ ¬(x^2 - x - 6 < 0 → |x| < 2) :=
by
  sorry

end abs_less_than_2_sufficient_but_not_necessary_l922_92290


namespace percent_increase_l922_92288

theorem percent_increase (P x : ℝ) (h1 : P + x/100 * P - 0.2 * (P + x/100 * P) = P) : x = 25 :=
by
  sorry

end percent_increase_l922_92288


namespace calculate_x_l922_92203

theorem calculate_x : 121 + 2 * 11 * 8 + 64 = 361 :=
by
  sorry

end calculate_x_l922_92203


namespace find_c_share_l922_92211

theorem find_c_share (a b c : ℕ) 
  (h1 : a + b + c = 1760)
  (h2 : ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x)
  (h3 : 6 * a = 8 * b ∧ 8 * b = 20 * c) : 
  c = 250 :=
by
  sorry

end find_c_share_l922_92211


namespace problem_equiv_proof_l922_92218

theorem problem_equiv_proof :
  2015 * (1 + 1999 / 2015) * (1 / 4) - (2011 / 2015) = 503 := 
by
  sorry

end problem_equiv_proof_l922_92218


namespace coefficient_of_8th_term_l922_92286

-- Define the general term of the binomial expansion
def binomial_expansion_term (n r : ℕ) (a b : ℕ) : ℕ := 
  Nat.choose n r * a^(n - r) * b^r

-- Define the specific scenario given in the problem
def specific_binomial_expansion_term : ℕ := 
  binomial_expansion_term 8 7 2 1  -- a = 2, b = x (consider b as 1 for coefficient calculation)

-- Problem statement to prove the coefficient of the 8th term is 16
theorem coefficient_of_8th_term : specific_binomial_expansion_term = 16 := by
  sorry

end coefficient_of_8th_term_l922_92286


namespace total_investment_sum_l922_92212

-- Definitions of the problem
variable (Raghu Trishul Vishal : ℕ)
variable (h1 : Raghu = 2000)
variable (h2 : Trishul = Nat.div (Raghu * 9) 10)
variable (h3 : Vishal = Nat.div (Trishul * 11) 10)

-- The theorem to prove
theorem total_investment_sum :
  Vishal + Trishul + Raghu = 5780 :=
by
  sorry

end total_investment_sum_l922_92212


namespace income_expenditure_ratio_l922_92269

theorem income_expenditure_ratio (I E S : ℕ) (hI : I = 19000) (hS : S = 11400) (hRel : S = I - E) :
  I / E = 95 / 38 :=
by
  sorry

end income_expenditure_ratio_l922_92269


namespace travel_time_l922_92296

def speed : ℝ := 60  -- Speed of the car in miles per hour
def distance : ℝ := 300  -- Distance to the campground in miles

theorem travel_time : distance / speed = 5 := by
  sorry

end travel_time_l922_92296


namespace max_value_of_f_l922_92240

noncomputable def f (x : ℝ) := 2 * Real.cos x + Real.sin x

theorem max_value_of_f : ∃ x, f x = Real.sqrt 5 := sorry

end max_value_of_f_l922_92240


namespace lcm_of_ratio_hcf_l922_92207

theorem lcm_of_ratio_hcf {a b : ℕ} (ratioCond : a = 14 * 28) (ratioCond2 : b = 21 * 28) (hcfCond : Nat.gcd a b = 28) : Nat.lcm a b = 1176 := by
  sorry

end lcm_of_ratio_hcf_l922_92207


namespace combined_yells_l922_92239

def yells_at_obedient : ℕ := 12
def yells_at_stubborn (y_obedient : ℕ) : ℕ := 4 * y_obedient
def total_yells (y_obedient : ℕ) (y_stubborn : ℕ) : ℕ := y_obedient + y_stubborn

theorem combined_yells : total_yells yells_at_obedient (yells_at_stubborn yells_at_obedient) = 60 := 
by
  sorry

end combined_yells_l922_92239


namespace relationship_between_m_and_n_l922_92270

theorem relationship_between_m_and_n
  (m n : ℝ)
  (circle_eq : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 2 * y - 4 = 0)
  (line_eq : ∀ (x y : ℝ), m * x + 2 * n * y - 4 = 0) :
  m - n - 2 = 0 := 
  sorry

end relationship_between_m_and_n_l922_92270


namespace Jorge_is_24_years_younger_l922_92271

-- Define the conditions
def Jorge_age_2005 := 16
def Simon_age_2010 := 45

-- Prove that Jorge is 24 years younger than Simon
theorem Jorge_is_24_years_younger :
  (Simon_age_2010 - (Jorge_age_2005 + 5) = 24) :=
by
  sorry

end Jorge_is_24_years_younger_l922_92271


namespace probability_one_head_one_tail_l922_92278

def toss_outcomes : List (String × String) := [("head", "head"), ("head", "tail"), ("tail", "head"), ("tail", "tail")]

def favorable_outcomes (outcomes : List (String × String)) : List (String × String) :=
  outcomes.filter (fun x => (x = ("head", "tail")) ∨ (x = ("tail", "head")))

theorem probability_one_head_one_tail :
  (favorable_outcomes toss_outcomes).length / toss_outcomes.length = 1 / 2 :=
by
  -- Proof will be filled in here
  sorry

end probability_one_head_one_tail_l922_92278


namespace employed_females_percentage_l922_92254

theorem employed_females_percentage (total_employed_percentage employed_males_percentage employed_females_percentage : ℝ) 
    (h1 : total_employed_percentage = 64) 
    (h2 : employed_males_percentage = 48) 
    (h3 : employed_females_percentage = total_employed_percentage - employed_males_percentage) :
    (employed_females_percentage / total_employed_percentage * 100) = 25 :=
by
  sorry

end employed_females_percentage_l922_92254


namespace number_of_pipes_l922_92264

theorem number_of_pipes (h_same_height : forall (height : ℝ), height > 0)
  (diam_large : ℝ) (hl : diam_large = 6)
  (diam_small : ℝ) (hs : diam_small = 1) :
  (π * (diam_large / 2)^2) / (π * (diam_small / 2)^2) = 36 :=
by
  sorry

end number_of_pipes_l922_92264


namespace smallest_n_for_factorization_l922_92210

theorem smallest_n_for_factorization :
  ∃ n : ℤ, (∀ A B : ℤ, A * B = 60 ↔ n = 5 * B + A) ∧ n = 56 :=
by
  sorry

end smallest_n_for_factorization_l922_92210


namespace number_of_free_ranging_chickens_l922_92231

-- Define the conditions as constants
def coop_chickens : ℕ := 14
def run_chickens : ℕ := 2 * coop_chickens
def barn_chickens : ℕ := coop_chickens / 2
def total_chickens_in_coop_and_run : ℕ := coop_chickens + run_chickens    
def free_ranging_chickens_condition : ℕ := 2 * run_chickens - 4
def ratio_condition : Prop := total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + free_ranging_chickens_condition)
def target_free_ranging_chickens : ℕ := 105

-- The proof statement
theorem number_of_free_ranging_chickens : 
  total_chickens_in_coop_and_run * 5 = 2 * (total_chickens_in_coop_and_run + target_free_ranging_chickens) →
  free_ranging_chickens_condition = target_free_ranging_chickens :=
by {
  sorry
}

end number_of_free_ranging_chickens_l922_92231


namespace inequality_solution_correct_l922_92245

variable (f : ℝ → ℝ)

def f_one : Prop := f 1 = 1

def f_prime_half : Prop := ∀ x : ℝ, (deriv f x) > (1 / 2)

def inequality_solution_set : Prop := ∀ x : ℝ, f (x^2) < (x^2 / 2 + 1 / 2) ↔ -1 < x ∧ x < 1

theorem inequality_solution_correct (h1 : f_one f) (h2 : f_prime_half f) : inequality_solution_set f := sorry

end inequality_solution_correct_l922_92245


namespace online_game_months_l922_92228

theorem online_game_months (m : ℕ) (initial_cost monthly_cost total_cost : ℕ) 
  (h1 : initial_cost = 5) (h2 : monthly_cost = 8) (h3 : total_cost = 21) 
  (h_equation : initial_cost + monthly_cost * m = total_cost) : m = 2 :=
by {
  -- Placeholder for the proof, as we don't need to include it
  sorry
}

end online_game_months_l922_92228


namespace magazines_sold_l922_92261

theorem magazines_sold (total_sold : Float) (newspapers_sold : Float) (magazines_sold : Float)
  (h1 : total_sold = 425.0)
  (h2 : newspapers_sold = 275.0) :
  magazines_sold = total_sold - newspapers_sold :=
by
  sorry

#check magazines_sold

end magazines_sold_l922_92261


namespace ellipse_area_l922_92281

theorem ellipse_area (P : ℝ) (b : ℝ) (a : ℝ) (A : ℝ) (h1 : P = 18)
  (h2 : a = b + 4)
  (h3 : A = π * a * b) :
  A = 5 * π :=
by
  sorry

end ellipse_area_l922_92281


namespace largest_expression_l922_92220

def U := 2 * 2004^2005
def V := 2004^2005
def W := 2003 * 2004^2004
def X := 2 * 2004^2004
def Y := 2004^2004
def Z := 2004^2003

theorem largest_expression :
  U - V > V - W ∧
  U - V > W - X ∧
  U - V > X - Y ∧
  U - V > Y - Z :=
by
  sorry

end largest_expression_l922_92220


namespace divisible_by_3_l922_92266

theorem divisible_by_3 :
  ∃ n : ℕ, (5 + 2 + n + 4 + 8) % 3 = 0 ∧ n = 2 := 
by
  sorry

end divisible_by_3_l922_92266


namespace simplify_expression_l922_92209

theorem simplify_expression : (8^(1/3) / 8^(1/6)) = 8^(1/6) :=
by
  sorry

end simplify_expression_l922_92209


namespace find_d_l922_92229

theorem find_d (d : ℝ) (h1 : ∃ (x y : ℝ), y = x + d ∧ x = -y + d ∧ x = d-1 ∧ y = d) : d = 1 :=
sorry

end find_d_l922_92229


namespace paperclips_volume_75_l922_92255

noncomputable def paperclips (v : ℝ) : ℝ := 60 / Real.sqrt 27 * Real.sqrt v

theorem paperclips_volume_75 :
  paperclips 75 = 100 :=
by
  sorry

end paperclips_volume_75_l922_92255


namespace sum_of_solutions_l922_92297

theorem sum_of_solutions (y : ℝ) (h : y^2 = 25) : ∃ (a b : ℝ), (a = 5 ∨ a = -5) ∧ (b = 5 ∨ b = -5) ∧ a + b = 0 :=
sorry

end sum_of_solutions_l922_92297


namespace average_computation_l922_92223

variable {a b c X Y Z : ℝ}

theorem average_computation 
  (h1 : a + b + c = 15)
  (h2 : X + Y + Z = 21) :
  ((2 * a + 3 * X) + (2 * b + 3 * Y) + (2 * c + 3 * Z)) / 3 = 31 :=
by
  sorry

end average_computation_l922_92223


namespace greatest_product_of_two_even_integers_whose_sum_is_300_l922_92238

theorem greatest_product_of_two_even_integers_whose_sum_is_300 :
  ∃ (x y : ℕ), (2 ∣ x) ∧ (2 ∣ y) ∧ (x + y = 300) ∧ (x * y = 22500) :=
by
  sorry

end greatest_product_of_two_even_integers_whose_sum_is_300_l922_92238


namespace binom_30_3_is_4060_l922_92230

theorem binom_30_3_is_4060 : Nat.choose 30 3 = 4060 := by
  sorry

end binom_30_3_is_4060_l922_92230


namespace max_combined_weight_l922_92285

theorem max_combined_weight (E A : ℕ) (h1 : A = 2 * E) (h2 : A + E = 90) (w_A : ℕ := 5) (w_E : ℕ := 2 * w_A) :
  E * w_E + A * w_A = 600 :=
by
  sorry

end max_combined_weight_l922_92285


namespace granger_bought_4_loaves_of_bread_l922_92275

-- Define the prices of items
def price_of_spam : Nat := 3
def price_of_pb : Nat := 5
def price_of_bread : Nat := 2

-- Define the quantities bought by Granger
def qty_spam : Nat := 12
def qty_pb : Nat := 3
def total_amount_paid : Nat := 59

-- The problem statement in Lean: Prove the number of loaves of bread bought
theorem granger_bought_4_loaves_of_bread :
  (qty_spam * price_of_spam) + (qty_pb * price_of_pb) + (4 * price_of_bread) = total_amount_paid :=
sorry

end granger_bought_4_loaves_of_bread_l922_92275


namespace find_f_l922_92291

theorem find_f (f : ℝ → ℝ)
  (h1 : ∀ x > 0, f x > 0)
  (h2 : f 1 = 1)
  (h3 : ∀ a b : ℝ, f (a + b) * (f a + f b) = 2 * f a * f b + a^2 + b^2) : ∀ x : ℝ, f x = x := by
  sorry

end find_f_l922_92291


namespace opposite_sides_of_line_l922_92233

theorem opposite_sides_of_line 
  (x₀ y₀ : ℝ) 
  (h : (3 * x₀ + 2 * y₀ - 8) * (3 * 1 + 2 * 2 - 8) < 0) :
  3 * x₀ + 2 * y₀ > 8 :=
by
  sorry

end opposite_sides_of_line_l922_92233


namespace players_count_l922_92287

def total_socks : ℕ := 22
def socks_per_player : ℕ := 2

theorem players_count : total_socks / socks_per_player = 11 :=
by
  sorry

end players_count_l922_92287


namespace quadruplets_sets_l922_92299

theorem quadruplets_sets (a b c babies: ℕ) (h1: 2 * a + 3 * b + 4 * c = 1200) (h2: b = 5 * c) (h3: a = 2 * b) :
  4 * c = 123 :=
by
  sorry

end quadruplets_sets_l922_92299


namespace Evelyn_bottle_caps_problem_l922_92201

theorem Evelyn_bottle_caps_problem (E : ℝ) (H1 : E - 18.0 = 45) : E = 63.0 := 
by
  sorry


end Evelyn_bottle_caps_problem_l922_92201


namespace endpoint_correctness_l922_92268

-- Define two points in 2D space
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define start point (2, 2)
def startPoint : Point := ⟨2, 2⟩

-- Define the endpoint's conditions
def endPoint (x y : ℝ) : Prop :=
  y = 2 * x + 1 ∧ (x > 0) ∧ (Real.sqrt ((x - startPoint.x) ^ 2 + (y - startPoint.y) ^ 2) = 6)

-- The solution to the problem proving (3.4213, 7.8426) satisfies the conditions
theorem endpoint_correctness : ∃ (x y : ℝ), endPoint x y ∧ x = 3.4213 ∧ y = 7.8426 := by
  use 3.4213
  use 7.8426
  sorry

end endpoint_correctness_l922_92268


namespace john_spent_at_candy_store_l922_92279

-- Conditions
def weekly_allowance : ℚ := 2.25
def spent_at_arcade : ℚ := (3 / 5) * weekly_allowance
def remaining_after_arcade : ℚ := weekly_allowance - spent_at_arcade
def spent_at_toy_store : ℚ := (1 / 3) * remaining_after_arcade
def remaining_after_toy_store : ℚ := remaining_after_arcade - spent_at_toy_store

-- Problem: Prove that John spent $0.60 at the candy store
theorem john_spent_at_candy_store : remaining_after_toy_store = 0.60 :=
by
  sorry

end john_spent_at_candy_store_l922_92279


namespace gimbap_total_cost_l922_92280

theorem gimbap_total_cost :
  let basic_gimbap_cost := 2000
  let tuna_gimbap_cost := 3500
  let red_pepper_gimbap_cost := 3000
  let beef_gimbap_cost := 4000
  let nude_gimbap_cost := 3500
  let cost_of_two gimbaps := (tuna_gimbap_cost * 2) + (beef_gimbap_cost * 2) + (nude_gimbap_cost * 2)
  cost_of_two gimbaps = 22000 := 
by 
  sorry

end gimbap_total_cost_l922_92280


namespace geometric_sequence_properties_l922_92241

/-- Given {a_n} is a geometric sequence, a_1 = 1 and a_4 = 1/8, 
the common ratio q of {a_n} is 1/2 and the sum of the first 5 terms of {1/a_n} is 31. -/
theorem geometric_sequence_properties (a : ℕ → ℝ) (h1 : a 1 = 1) (h4 : a 4 = 1 / 8) : 
  (∃ q : ℝ, (∀ n : ℕ, a n = a 1 * q ^ (n - 1)) ∧ q = 1 / 2) ∧ 
  (∃ S : ℝ, S = 31 ∧ S = (1 - 2^5) / (1 - 2)) :=
by
  -- Skipping the proof
  sorry

end geometric_sequence_properties_l922_92241


namespace probability_function_has_zero_point_l922_92242

noncomputable def probability_of_zero_point : ℚ :=
by
  let S := ({-1, 1, 2} : Finset ℤ).product ({-1, 1, 2} : Finset ℤ)
  let zero_point_pairs := S.filter (λ p => (p.1 * p.2 ≤ 1))
  let favorable_outcomes := zero_point_pairs.card
  let total_outcomes := S.card
  exact favorable_outcomes / total_outcomes

theorem probability_function_has_zero_point :
  probability_of_zero_point = (2 / 3 : ℚ) :=
  sorry

end probability_function_has_zero_point_l922_92242


namespace pencils_multiple_of_28_l922_92253

theorem pencils_multiple_of_28 (students pens pencils : ℕ) 
  (h1 : students = 28) 
  (h2 : pens = 1204) 
  (h3 : ∃ k, pens = students * k) 
  (h4 : ∃ n, pencils = students * n) : 
  ∃ m, pencils = 28 * m :=
by
  sorry

end pencils_multiple_of_28_l922_92253


namespace certain_number_l922_92294

theorem certain_number (x y a : ℤ) (h1 : 4 * x + y = a) (h2 : 2 * x - y = 20) 
  (h3 : y ^ 2 = 4) : a = 46 :=
sorry

end certain_number_l922_92294


namespace radish_patch_area_l922_92282

-- Definitions from the conditions
variables (R P : ℕ) -- R: area of radish patch, P: area of pea patch
variable (h1 : P = 2 * R) -- The pea patch is twice as large as the radish patch
variable (h2 : P / 6 = 5) -- One-sixth of the pea patch is 5 square feet

-- Goal statement
theorem radish_patch_area : R = 15 :=
by
  sorry

end radish_patch_area_l922_92282


namespace circle_chords_intersect_radius_square_l922_92213

theorem circle_chords_intersect_radius_square
  (r : ℝ) -- The radius of the circle
  (AB CD BP : ℝ) -- The lengths of chords AB, CD, and segment BP
  (angle_APD : ℝ) -- The angle ∠APD in degrees
  (AB_len : AB = 8)
  (CD_len : CD = 12)
  (BP_len : BP = 10)
  (angle_APD_val : angle_APD = 60) :
  r^2 = 91 := 
sorry

end circle_chords_intersect_radius_square_l922_92213


namespace sum_of_ages_l922_92237

-- Define the ages of Maggie, Juliet, and Ralph
def maggie_age : ℕ := by
  let juliet_age := 10
  let maggie_age := juliet_age - 3
  exact maggie_age

def ralph_age : ℕ := by
  let juliet_age := 10
  let ralph_age := juliet_age + 2
  exact ralph_age

-- The main theorem: The sum of Maggie's and Ralph's ages
theorem sum_of_ages : maggie_age + ralph_age = 19 := by
  sorry

end sum_of_ages_l922_92237


namespace min_third_side_triangle_l922_92251

theorem min_third_side_triangle (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
    (h_distinct_1 : 42 * a ≠ 72 * b) (h_distinct_2 : 42 * a ≠ c) (h_distinct_3 : 72 * b ≠ c) :
    (42 * a + 72 * b > c) ∧ (42 * a + c > 72 * b) ∧ (72 * b + c > 42 * a) → c ≥ 7 :=
sorry

end min_third_side_triangle_l922_92251


namespace gnomes_red_hats_small_noses_l922_92259

theorem gnomes_red_hats_small_noses :
  ∀ (total_gnomes red_hats blue_hats big_noses_blue_hats : ℕ),
  total_gnomes = 28 →
  red_hats = (3 * total_gnomes) / 4 →
  blue_hats = total_gnomes - red_hats →
  big_noses_blue_hats = 6 →
  (total_gnomes / 2) - big_noses_blue_hats = 8 →
  red_hats - 8 = 13 :=
by
  intros total_gnomes red_hats blue_hats big_noses_blue_hats
  intros h1 h2 h3 h4 h5
  sorry

end gnomes_red_hats_small_noses_l922_92259


namespace vikki_take_home_pay_l922_92243

-- Define the conditions
def hours_worked : ℕ := 42
def pay_rate : ℝ := 10
def tax_rate : ℝ := 0.20
def insurance_rate : ℝ := 0.05
def union_dues : ℝ := 5

-- Define the gross earnings function
def gross_earnings (hours_worked : ℕ) (pay_rate : ℝ) : ℝ := hours_worked * pay_rate

-- Define the deductions functions
def tax_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def insurance_deduction (gross : ℝ) (rate : ℝ) : ℝ := gross * rate
def total_deductions (tax : ℝ) (insurance : ℝ) (dues : ℝ) : ℝ := tax + insurance + dues

-- Define the take-home pay function
def take_home_pay (gross : ℝ) (deductions : ℝ) : ℝ := gross - deductions

theorem vikki_take_home_pay :
  take_home_pay (gross_earnings hours_worked pay_rate)
    (total_deductions (tax_deduction (gross_earnings hours_worked pay_rate) tax_rate)
                      (insurance_deduction (gross_earnings hours_worked pay_rate) insurance_rate)
                      union_dues) = 310 :=
by
  sorry

end vikki_take_home_pay_l922_92243


namespace find_a_value_l922_92225

theorem find_a_value 
  (A : Set ℤ := {-1, 0, 1})
  (a : ℤ) 
  (B : Set ℤ := {a, a^2}) 
  (h_union : A ∪ B = A) : 
  a = -1 :=
sorry

end find_a_value_l922_92225


namespace problem_2002_multiples_l922_92277

theorem problem_2002_multiples :
  ∃ (n : ℕ), 
    n = 1800 ∧
    (∀ (i j : ℕ), 0 ≤ i ∧ i < j ∧ j ≤ 149 →
      2002 ∣ (10^j - 10^i) ↔ j - i ≡ 0 [MOD 6]) :=
sorry

end problem_2002_multiples_l922_92277


namespace inclination_angle_of_line_l922_92248

noncomputable def angle_of_inclination (m : ℝ) : ℝ :=
  Real.arctan m

theorem inclination_angle_of_line (α : ℝ) :
  angle_of_inclination (-1) = 3 * Real.pi / 4 :=
by
  sorry

end inclination_angle_of_line_l922_92248


namespace rotated_line_l1_l922_92208

-- Define the original line equation and the point around which the line is rotated
def line_l (x y : ℝ) : Prop := x - y + 1 = 0
def point_A : ℝ × ℝ := (2, 3)

-- Define the line equation that needs to be proven
def line_l1 (x y : ℝ) : Prop := x + y - 5 = 0

-- The theorem stating that after a 90-degree rotation of line l around point A, the new line is equation l1
theorem rotated_line_l1 : 
  ∀ (x y : ℝ), 
  (∃ (k : ℝ), k = 1 ∧ ∀ (x y), line_l x y ∧ ∀ (x y), line_l1 x y) ∧ 
  ∀ (a b : ℝ), (a, b) = point_A → 
  x + y - 5 = 0 := 
by
  sorry

end rotated_line_l1_l922_92208


namespace division_remainder_l922_92217

noncomputable def remainder (p q : Polynomial ℝ) : Polynomial ℝ :=
  p % q

theorem division_remainder :
  remainder (Polynomial.X ^ 3) (Polynomial.X ^ 2 + 7 * Polynomial.X + 2) = 47 * Polynomial.X + 14 :=
by
  sorry

end division_remainder_l922_92217


namespace commission_percentage_l922_92236

-- Define the given conditions
def cost_of_item : ℝ := 17
def observed_price : ℝ := 25.50
def desired_profit_percentage : ℝ := 0.20

-- Calculate the desired profit in dollars
def desired_profit : ℝ := desired_profit_percentage * cost_of_item

-- Calculate the total desired price for the distributor
def total_desired_price : ℝ := cost_of_item + desired_profit

-- Calculate the commission in dollars
def commission_in_dollars : ℝ := observed_price - total_desired_price

-- Prove that commission percentage taken by the online store is 20%
theorem commission_percentage :
  (commission_in_dollars / observed_price) * 100 = 20 := 
by
  -- This is the placeholder for the proof
  sorry

end commission_percentage_l922_92236


namespace original_denominator_is_21_l922_92265

theorem original_denominator_is_21 (d : ℕ) : (3 + 6) / (d + 6) = 1 / 3 → d = 21 :=
by
  intros h
  sorry

end original_denominator_is_21_l922_92265


namespace average_speed_last_segment_l922_92267

theorem average_speed_last_segment
  (total_distance : ℕ)
  (total_time : ℕ)
  (speed1 speed2 speed3 : ℕ)
  (last_segment_time : ℕ)
  (average_speed_total : ℕ) :
  total_distance = 180 →
  total_time = 180 →
  speed1 = 40 →
  speed2 = 50 →
  speed3 = 60 →
  average_speed_total = 60 →
  last_segment_time = 45 →
  ∃ (speed4 : ℕ), speed4 = 90 :=
by sorry

end average_speed_last_segment_l922_92267


namespace minimum_possible_length_of_third_side_l922_92289

theorem minimum_possible_length_of_third_side (a b : ℝ) (h : a = 8 ∧ b = 15 ∨ a = 15 ∧ b = 8) : 
  ∃ c : ℝ, (c * c = a * a + b * b ∨ c * c = a * a - b * b ∨ c * c = b * b - a * a) ∧ c = Real.sqrt 161 :=
by
  sorry

end minimum_possible_length_of_third_side_l922_92289


namespace minimum_value_expression_l922_92260

theorem minimum_value_expression (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 := 
by
  sorry

end minimum_value_expression_l922_92260


namespace original_price_l922_92252

-- Definitions of conditions
def SalePrice : Float := 70
def DecreasePercentage : Float := 30

-- Statement to prove
theorem original_price (P : Float) (h : 0.70 * P = SalePrice) : P = 100 := by
  sorry

end original_price_l922_92252


namespace keychain_arrangement_l922_92256

open Function

theorem keychain_arrangement (keys : Finset ℕ) (h : keys.card = 7)
  (house_key car_key office_key : ℕ) (hmem : house_key ∈ keys)
  (cmem : car_key ∈ keys) (omem : office_key ∈ keys) : 
  ∃ n : ℕ, n = 72 :=
by
  sorry

end keychain_arrangement_l922_92256


namespace fractions_of_group_money_l922_92274

def moneyDistribution (m l n o : ℕ) (moeGave : ℕ) (lokiGave : ℕ) (nickGave : ℕ) : Prop :=
  moeGave = 1 / 5 * m ∧
  lokiGave = 1 / 4 * l ∧
  nickGave = 1 / 3 * n ∧
  moeGave = lokiGave ∧
  lokiGave = nickGave ∧
  o = moeGave + lokiGave + nickGave

theorem fractions_of_group_money (m l n o total : ℕ) :
  moneyDistribution m l n o 1 1 1 →
  total = m + l + n →
  (o : ℚ) / total = 1 / 4 :=
by sorry

end fractions_of_group_money_l922_92274


namespace absolute_value_equality_l922_92204

variables {a b c d : ℝ}

theorem absolute_value_equality (h1 : |a - b| + |c - d| = 99) (h2 : |a - c| + |b - d| = 1) : |a - d| + |b - c| = 99 :=
sorry

end absolute_value_equality_l922_92204


namespace ratio_of_sides_l922_92214

theorem ratio_of_sides (
  perimeter_triangle perimeter_square : ℕ)
  (h_triangle : perimeter_triangle = 48)
  (h_square : perimeter_square = 64) :
  (perimeter_triangle / 3) / (perimeter_square / 4) = 1 :=
by
  sorry

end ratio_of_sides_l922_92214


namespace fans_per_set_l922_92257

theorem fans_per_set (total_fans : ℕ) (sets_of_bleachers : ℕ) (fans_per_set : ℕ)
  (h1 : total_fans = 2436) (h2 : sets_of_bleachers = 3) : fans_per_set = 812 :=
by
  sorry

end fans_per_set_l922_92257


namespace trapezoid_EC_length_l922_92292

-- Define a trapezoid and its properties.
structure Trapezoid (A B C D : Type) :=
(base1 : ℝ) -- AB
(base2 : ℝ) -- CD
(diagonal_AC : ℝ) -- AC
(AB_eq_3CD : base1 = 3 * base2)
(AC_length : diagonal_AC = 15)
(E : Type) -- point of intersection of diagonals

-- Proof statement that length of EC is 15/4
theorem trapezoid_EC_length
  {A B C D E : Type}
  (t : Trapezoid A B C D)
  (E : Type)
  (intersection_E : E) :
  ∃ (EC : ℝ), EC = 15 / 4 :=
by
  have h1 : t.base1 = 3 * t.base2 := t.AB_eq_3CD
  have h2 : t.diagonal_AC = 15 := t.AC_length
  -- Use the given conditions to derive the length of EC
  sorry

end trapezoid_EC_length_l922_92292


namespace find_price_per_backpack_l922_92232

noncomputable def original_price_of_each_backpack
  (total_backpacks : ℕ)
  (monogram_cost : ℕ)
  (total_cost : ℕ)
  (backpacks_cost_before_discount : ℕ) : ℕ :=
total_cost - (total_backpacks * monogram_cost)

theorem find_price_per_backpack
  (total_backpacks : ℕ := 5)
  (monogram_cost : ℕ := 12)
  (total_cost : ℕ := 140)
  (expected_price_per_backpack : ℕ := 16) :
  original_price_of_each_backpack total_backpacks monogram_cost total_cost / total_backpacks = expected_price_per_backpack :=
by
  sorry

end find_price_per_backpack_l922_92232


namespace n_value_l922_92250

theorem n_value (n : ℕ) (h1 : ∃ a b : ℕ, a = (5 + 10 + 15 + 20 + 25 + 30 + 35) / 7 ∧ b = 2 * n ∧ a ^ 2 - b ^ 2 = 0) : n = 10 := 
  by sorry

end n_value_l922_92250


namespace mass_scientific_notation_l922_92249

def mass := 37e-6

theorem mass_scientific_notation : mass = 3.7 * 10^(-5) :=
by
  sorry

end mass_scientific_notation_l922_92249


namespace carol_meets_alice_in_30_minutes_l922_92222

def time_to_meet (alice_speed carol_speed initial_distance : ℕ) : ℕ :=
((initial_distance * 60) / (alice_speed + carol_speed))

theorem carol_meets_alice_in_30_minutes :
  time_to_meet 4 6 5 = 30 := 
by 
  sorry

end carol_meets_alice_in_30_minutes_l922_92222


namespace opposite_of_two_l922_92295

def opposite (n : ℤ) : ℤ := -n

theorem opposite_of_two : opposite 2 = -2 :=
by
  -- proof skipped
  sorry

end opposite_of_two_l922_92295


namespace determine_p_and_q_l922_92283

theorem determine_p_and_q (x p q : ℝ) : 
  (x + 4) * (x - 1) = x^2 + p * x + q → (p = 3 ∧ q = -4) := 
by 
  sorry

end determine_p_and_q_l922_92283


namespace problem1_problem2_l922_92247

def f (x : ℝ) : ℝ := |x - 3|

theorem problem1 :
  {x : ℝ | f x < 2 + |x + 1|} = {x : ℝ | 0 < x} := sorry

theorem problem2 (m n : ℝ) (h_mn : m > 0) (h_nn : n > 0) (h : (1 / m) + (1 / n) = 2 * m * n) :
  m * f n + n * f (-m) ≥ 6 := sorry

end problem1_problem2_l922_92247


namespace inequality_sqrt_a_b_c_l922_92262

noncomputable def sqrt (x : ℝ) := x ^ (1 / 2)

theorem inequality_sqrt_a_b_c (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_sum : a + b + c = 1) :
  sqrt (a ^ (1 - a) * b ^ (1 - b) * c ^ (1 - c)) ≤ 1 / 3 := 
sorry

end inequality_sqrt_a_b_c_l922_92262


namespace panic_percentage_l922_92263

theorem panic_percentage (original_population disappeared_after first_population second_population : ℝ) 
  (h₁ : original_population = 7200)
  (h₂ : disappeared_after = original_population * 0.10)
  (h₃ : first_population = original_population - disappeared_after)
  (h₄ : second_population = 4860)
  (h₅ : second_population = first_population - (first_population * 0.25)) : 
  second_population = first_population * (1 - 0.25) :=
by
  sorry

end panic_percentage_l922_92263


namespace ratio_of_black_to_white_after_border_l922_92246

def original_tiles (black white : ℕ) : Prop := black = 14 ∧ white = 21
def original_dimensions (length width : ℕ) : Prop := length = 5 ∧ width = 7

def border_added (length width l w : ℕ) : Prop := l = length + 2 ∧ w = width + 2

def total_white_tiles (initial_white new_white total_white : ℕ) : Prop :=
  total_white = initial_white + new_white

def black_white_ratio (black_tiles white_tiles : ℕ) (ratio : ℚ) : Prop :=
  ratio = black_tiles / white_tiles

theorem ratio_of_black_to_white_after_border 
  (black_white_tiles : ℕ → ℕ → Prop)
  (dimensions : ℕ → ℕ → Prop)
  (border : ℕ → ℕ → ℕ → ℕ → Prop)
  (total_white : ℕ → ℕ → ℕ → Prop)
  (ratio : ℕ → ℕ → ℚ → Prop)
  (black_tiles white_tiles initial_white total_white_new length width l w : ℕ)
  (rat : ℚ) :
  black_white_tiles black_tiles initial_white →
  dimensions length width →
  border length width l w →
  total_white initial_white (l * w - length * width) white_tiles →
  ratio black_tiles white_tiles rat →
  rat = 2 / 7 :=
by
  intros
  sorry

end ratio_of_black_to_white_after_border_l922_92246


namespace pencil_pen_cost_l922_92219

theorem pencil_pen_cost 
  (p q : ℝ) 
  (h1 : 6 * p + 3 * q = 3.90) 
  (h2 : 2 * p + 5 * q = 4.45) :
  3 * p + 4 * q = 3.92 :=
by
  sorry

end pencil_pen_cost_l922_92219


namespace num_intersecting_chords_on_circle_l922_92224

theorem num_intersecting_chords_on_circle (points : Fin 20 → Prop) : 
  ∃ num_chords : ℕ, num_chords = 156180 :=
by
  sorry

end num_intersecting_chords_on_circle_l922_92224


namespace hexagon_area_is_32_l922_92298

noncomputable def area_of_hexagon : ℝ := 
  let p0 : ℝ × ℝ := (0, 0)
  let p1 : ℝ × ℝ := (2, 4)
  let p2 : ℝ × ℝ := (5, 4)
  let p3 : ℝ × ℝ := (7, 0)
  let p4 : ℝ × ℝ := (5, -4)
  let p5 : ℝ × ℝ := (2, -4)
  -- Triangle 1: p0, p1, p2
  let area_tri1 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 2: p2, p3, p4
  let area_tri2 := 1 / 2 * (8 : ℝ) * (2 : ℝ)
  -- Triangle 3: p4, p5, p0
  let area_tri3 := 1 / 2 * (3 : ℝ) * (4 : ℝ)
  -- Triangle 4: p1, p2, p5
  let area_tri4 := 1 / 2 * (8 : ℝ) * (3 : ℝ)
  area_tri1 + area_tri2 + area_tri3 + area_tri4

theorem hexagon_area_is_32 : area_of_hexagon = 32 := 
by
  sorry

end hexagon_area_is_32_l922_92298


namespace train_speed_comparison_l922_92235

variables (V_A V_B : ℝ)

open Classical

theorem train_speed_comparison
  (distance_AB : ℝ)
  (h_distance : distance_AB = 360)
  (h_time_limit : V_A ≤ 72)
  (h_meeting_time : 3 * V_A + 2 * V_B > 360) :
  V_B > V_A :=
by {
  sorry
}

end train_speed_comparison_l922_92235


namespace initial_goats_l922_92284

theorem initial_goats (G : ℕ) (h1 : 2 + 3 + G + 3 + 5 + 2 = 21) : G = 4 :=
by
  sorry

end initial_goats_l922_92284


namespace problem1_problem2_l922_92200

-- Define sets A and B based on the conditions
def setA (a : ℝ) : Set ℝ := {x : ℝ | a ≤ x ∧ x ≤ a + 3}
def setB : Set ℝ := {x : ℝ | x < -2 ∨ x > 6}

-- Define the two proof problems as Lean statements
theorem problem1 (a : ℝ) : setA a ∩ setB = ∅ ↔ -2 ≤ a ∧ a ≤ 3 := by
  sorry

theorem problem2 (a : ℝ) : setA a ⊆ setB ↔ (a < -5 ∨ a > 6) := by
  sorry

end problem1_problem2_l922_92200


namespace pentagon_edges_and_vertices_sum_l922_92276

theorem pentagon_edges_and_vertices_sum :
  let edges := 5
  let vertices := 5
  edges + vertices = 10 := by
  sorry

end pentagon_edges_and_vertices_sum_l922_92276
