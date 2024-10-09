import Mathlib

namespace soldiers_in_groups_l1082_108249

theorem soldiers_in_groups (x : ℕ) (h1 : x % 2 = 1) (h2 : x % 3 = 2) (h3 : x % 5 = 3) : x % 30 = 23 :=
by
  sorry

end soldiers_in_groups_l1082_108249


namespace total_spending_l1082_108239

theorem total_spending :
  let price_per_pencil := 0.20
  let tolu_pencils := 3
  let robert_pencils := 5
  let melissa_pencils := 2
  let tolu_cost := tolu_pencils * price_per_pencil
  let robert_cost := robert_pencils * price_per_pencil
  let melissa_cost := melissa_pencils * price_per_pencil
  let total_cost := tolu_cost + robert_cost + melissa_cost
  total_cost = 2.00 := by
  sorry

end total_spending_l1082_108239


namespace angle_E_degree_l1082_108229

-- Given conditions
variables {E F G H : ℝ} -- degrees of the angles in quadrilateral EFGH

-- Condition 1: The angles satisfy a specific ratio
axiom angle_ratio : E = 3 * F ∧ E = 2 * G ∧ E = 6 * H

-- Condition 2: The sum of the angles in the quadrilateral is 360 degrees
axiom angle_sum : E + (E / 3) + (E / 2) + (E / 6) = 360

-- Prove the degree measure of angle E is 180 degrees
theorem angle_E_degree : E = 180 :=
by
  sorry

end angle_E_degree_l1082_108229


namespace n_n_plus_one_div_by_2_l1082_108251

theorem n_n_plus_one_div_by_2 (n : ℕ) (h₁ : 1 ≤ n) (h₂ : n ≤ 99) : 2 ∣ n * (n + 1) :=
by
  sorry

end n_n_plus_one_div_by_2_l1082_108251


namespace Harold_spending_l1082_108202

theorem Harold_spending
  (num_shirt_boxes : ℕ)
  (num_xl_boxes : ℕ)
  (wraps_shirt_boxes : ℕ)
  (wraps_xl_boxes : ℕ)
  (cost_per_roll : ℕ)
  (h1 : num_shirt_boxes = 20)
  (h2 : num_xl_boxes = 12)
  (h3 : wraps_shirt_boxes = 5)
  (h4 : wraps_xl_boxes = 3)
  (h5 : cost_per_roll = 4) :
  num_shirt_boxes / wraps_shirt_boxes + num_xl_boxes / wraps_xl_boxes * cost_per_roll = 32 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end Harold_spending_l1082_108202


namespace smallest_GCD_value_l1082_108266

theorem smallest_GCD_value (a b c d N : ℕ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : a ≠ d)
    (h4 : b ≠ c) (h5 : b ≠ d) (h6 : c ≠ d) (h7 : N > 5)
    (hc1 : Nat.gcd a b = 1 ∨ Nat.gcd a c = 1 ∨ Nat.gcd a d = 1 ∨ Nat.gcd b c = 1 ∨ Nat.gcd b d = 1 ∨ Nat.gcd c d = 1)
    (hc2 : Nat.gcd a b = 2 ∨ Nat.gcd a c = 2 ∨ Nat.gcd a d = 2 ∨ Nat.gcd b c = 2 ∨ Nat.gcd b d = 2 ∨ Nat.gcd c d = 2)
    (hc3 : Nat.gcd a b = 3 ∨ Nat.gcd a c = 3 ∨ Nat.gcd a d = 3 ∨ Nat.gcd b c = 3 ∨ Nat.gcd b d = 3 ∨ Nat.gcd c d = 3)
    (hc4 : Nat.gcd a b = 4 ∨ Nat.gcd a c = 4 ∨ Nat.gcd a d = 4 ∨ Nat.gcd b c = 4 ∨ Nat.gcd b d = 4 ∨ Nat.gcd c d = 4)
    (hc5 : Nat.gcd a b = 5 ∨ Nat.gcd a c = 5 ∨ Nat.gcd a d = 5 ∨ Nat.gcd b c = 5 ∨ Nat.gcd b d = 5 ∨ Nat.gcd c d = 5)
    (hcN : Nat.gcd a b = N ∨ Nat.gcd a c = N ∨ Nat.gcd a d = N ∨ Nat.gcd b c = N ∨ Nat.gcd b d = N ∨ Nat.gcd c d = N):
    N = 14 :=
sorry

end smallest_GCD_value_l1082_108266


namespace lower_limit_tip_percentage_l1082_108242

namespace meal_tip

def meal_cost : ℝ := 35.50
def total_paid : ℝ := 40.825
def tip_limit : ℝ := 15

-- Define the lower limit tip percentage as the solution to the given conditions.
theorem lower_limit_tip_percentage :
  ∃ x : ℝ, x > 0 ∧ x < 25 ∧ (meal_cost + (x / 100) * meal_cost = total_paid) → 
  x = tip_limit :=
sorry

end meal_tip

end lower_limit_tip_percentage_l1082_108242


namespace proof_problem_l1082_108243

noncomputable def problem_expression : ℝ :=
  50 * 39.96 * 3.996 * 500

theorem proof_problem : problem_expression = (3996 : ℝ)^2 :=
by
  sorry

end proof_problem_l1082_108243


namespace decrease_of_negative_distance_l1082_108267

theorem decrease_of_negative_distance (x : Int) (increase : Int → Int) (decrease : Int → Int) :
  (increase 30 = 30) → (decrease 5 = -5) → (decrease 5 = -5) :=
by
  intros
  sorry

end decrease_of_negative_distance_l1082_108267


namespace missing_digit_l1082_108246

theorem missing_digit (B : ℕ) (h : B < 10) : 
  (15 ∣ (200 + 10 * B)) ↔ B = 1 ∨ B = 4 :=
by sorry

end missing_digit_l1082_108246


namespace geometric_progression_identity_l1082_108250

theorem geometric_progression_identity (a b c : ℝ) (h : b^2 = a * c) : 
  (a + b + c) * (a - b + c) = a^2 + b^2 + c^2 := 
by
  sorry

end geometric_progression_identity_l1082_108250


namespace polynomial_degree_l1082_108200

variable {P : Polynomial ℝ}

theorem polynomial_degree (h1 : ∀ x : ℝ, (x - 4) * P.eval (2 * x) = 4 * (x - 1) * P.eval x) (h2 : P.eval 0 ≠ 0) : P.degree = 2 := 
sorry

end polynomial_degree_l1082_108200


namespace sin_593_l1082_108279

theorem sin_593 (h : Real.sin (37 * Real.pi / 180) = 3/5) : 
  Real.sin (593 * Real.pi / 180) = -3/5 :=
by
sorry

end sin_593_l1082_108279


namespace james_fish_weight_l1082_108297

theorem james_fish_weight :
  let trout := 200
  let salmon := trout + (trout * 0.5)
  let tuna := 2 * salmon
  trout + salmon + tuna = 1100 := 
by
  sorry

end james_fish_weight_l1082_108297


namespace light_distance_200_years_l1082_108224

-- Define the distance light travels in one year.
def distance_one_year := 5870000000000

-- Define the scientific notation representation for distance in one year
def distance_one_year_sci := 587 * 10^10

-- Define the distance light travels in 200 years.
def distance_200_years := distance_one_year * 200

-- Define the expected distance in scientific notation for 200 years.
def expected_distance := 1174 * 10^12

-- The theorem stating the given condition and the conclusion to prove
theorem light_distance_200_years : distance_200_years = expected_distance :=
by
  -- skipping the proof
  sorry

end light_distance_200_years_l1082_108224


namespace baseball_card_value_decrease_l1082_108288

theorem baseball_card_value_decrease (V0 : ℝ) (V1 V2 : ℝ) :
  V1 = V0 * 0.5 → V2 = V1 * 0.9 → (V0 - V2) / V0 * 100 = 55 :=
by 
  intros hV1 hV2
  sorry

end baseball_card_value_decrease_l1082_108288


namespace angle_A_is_30_degrees_l1082_108299

theorem angle_A_is_30_degrees {A : ℝ} (hA_acute : 0 < A ∧ A < π / 2) (hA_sin : Real.sin A = 1 / 2) : A = π / 6 :=
sorry

end angle_A_is_30_degrees_l1082_108299


namespace least_number_of_faces_l1082_108285

def faces_triangular_prism : ℕ := 5
def faces_quadrangular_prism : ℕ := 6
def faces_triangular_pyramid : ℕ := 4
def faces_quadrangular_pyramid : ℕ := 5
def faces_truncated_quadrangular_pyramid : ℕ := 6

theorem least_number_of_faces : faces_triangular_pyramid < faces_triangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_prism ∧
                                faces_triangular_pyramid < faces_quadrangular_pyramid ∧
                                faces_triangular_pyramid < faces_truncated_quadrangular_pyramid 
                                :=
by {
  sorry
}

end least_number_of_faces_l1082_108285


namespace function_symmetry_l1082_108233

noncomputable def f (ω : ℝ) (x : ℝ) := 2 * Real.sin (ω * x + (Real.pi / 6))

theorem function_symmetry (ω : ℝ) (hω : ω > 0) (hT : (2 * Real.pi / ω) = 4 * Real.pi) :
  ∃ (k : ℤ), f ω (2 * k * Real.pi - Real.pi / 3) = f ω 0 := by
  sorry

end function_symmetry_l1082_108233


namespace purple_shoes_count_l1082_108264

-- Define the conditions
def total_shoes : ℕ := 1250
def blue_shoes : ℕ := 540
def remaining_shoes : ℕ := total_shoes - blue_shoes
def green_shoes := remaining_shoes / 2
def purple_shoes := green_shoes

-- State the theorem to be proven
theorem purple_shoes_count : purple_shoes = 355 := 
by
-- Proof can be filled in here (not needed for the task)
sorry

end purple_shoes_count_l1082_108264


namespace triangle_area_proof_l1082_108294

noncomputable def segment_squared (a b : ℝ) : ℝ := a ^ 2 - b ^ 2

noncomputable def triangle_conditions (a b c : ℝ): Prop :=
  segment_squared b a = a ^ 2 - c ^ 2

noncomputable def area_triangle_OLK (r a b c : ℝ) (cond : triangle_conditions a b c): ℝ :=
  (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3))

theorem triangle_area_proof (r a b c : ℝ) (cond : triangle_conditions a b c) :
  area_triangle_OLK r a b c cond = (a / (2 * Real.sqrt 3)) * Real.sqrt (r^2 - (a^2 / 3)) :=
sorry

end triangle_area_proof_l1082_108294


namespace opposite_of_neg_five_halves_l1082_108231

theorem opposite_of_neg_five_halves : -(- (5 / 2: ℝ)) = 5 / 2 :=
by
    sorry

end opposite_of_neg_five_halves_l1082_108231


namespace cost_of_apples_l1082_108238

def cost_per_kilogram (m : ℝ) : ℝ := m
def number_of_kilograms : ℝ := 3

theorem cost_of_apples (m : ℝ) : cost_per_kilogram m * number_of_kilograms = 3 * m :=
by
  unfold cost_per_kilogram number_of_kilograms
  sorry

end cost_of_apples_l1082_108238


namespace exists_hamiltonian_path_l1082_108203

theorem exists_hamiltonian_path (n : ℕ) (cities : Fin n → Type) (roads : ∀ (i j : Fin n), cities i → cities j → Prop) 
(road_one_direction : ∀ i j (c1 : cities i) (c2 : cities j), roads i j c1 c2 → ¬ roads j i c2 c1) :
∃ start : Fin n, ∃ path : Fin n → Fin n, ∀ i j : Fin n, i ≠ j → path i ≠ path j :=
sorry

end exists_hamiltonian_path_l1082_108203


namespace sum_of_first_10_terms_of_arithmetic_sequence_l1082_108204

theorem sum_of_first_10_terms_of_arithmetic_sequence :
  ∀ (a n : ℕ) (a₁ : ℤ) (d : ℤ),
  (d = -2) →
  (a₇ : ℤ := a₁ + 6 * d) →
  (a₃ : ℤ := a₁ + 2 * d) →
  (a₁₀ : ℤ := a₁ + 9 * d) →
  (a₇ * a₇ = a₃ * a₁₀) →
  (S₁₀ : ℤ := 10 * a₁ + 45 * d) →
  S₁₀ = 270 :=
by
  intros a n a₁ d hd ha₇ ha₃ ha₁₀ hgm hS₁₀
  sorry

end sum_of_first_10_terms_of_arithmetic_sequence_l1082_108204


namespace actual_distance_traveled_l1082_108290

theorem actual_distance_traveled (D : ℕ) (h : D / 10 = (D + 20) / 15) : D = 40 := 
sorry

end actual_distance_traveled_l1082_108290


namespace income_is_12000_l1082_108272

theorem income_is_12000 (P : ℝ) : (P * 1.02 = 12240) → (P = 12000) :=
by
  intro h
  sorry

end income_is_12000_l1082_108272


namespace florist_first_picking_l1082_108232

theorem florist_first_picking (x : ℝ) (h1 : 37.0 + x + 19.0 = 72.0) : x = 16.0 :=
by
  sorry

end florist_first_picking_l1082_108232


namespace correct_factorization_l1082_108216

theorem correct_factorization : 
  (¬ (6 * x^2 * y^3 = 2 * x^2 * 3 * y^3)) ∧ 
  (¬ (x^2 + 2 * x + 1 = x * (x^2 + 2) + 1)) ∧ 
  (¬ ((x + 2) * (x - 3) = x^2 - x - 6)) ∧ 
  (x^2 - 9 = (x - 3) * (x + 3)) :=
by 
  sorry

end correct_factorization_l1082_108216


namespace dinosaur_book_cost_l1082_108228

-- Define the constants for costs and savings/needs
def dict_cost : ℕ := 11
def cookbook_cost : ℕ := 7
def savings : ℕ := 8
def needed : ℕ := 29
def total_cost : ℕ := savings + needed
def dino_cost : ℕ := 19

-- Mathematical statement to prove
theorem dinosaur_book_cost :
  dict_cost + dino_cost + cookbook_cost = total_cost :=
by
  -- The proof steps would go here
  sorry

end dinosaur_book_cost_l1082_108228


namespace TruckCapacities_RentalPlanExists_MinimumRentalCost_l1082_108276

-- Problem 1
theorem TruckCapacities (x y : ℕ) (h1: 2 * x + y = 10) (h2: x + 2 * y = 11) :
  x = 3 ∧ y = 4 :=
by
  sorry

-- Problem 2
theorem RentalPlanExists (a b : ℕ) (h: 3 * a + 4 * b = 31) :
  (a = 9 ∧ b = 1) ∨ (a = 5 ∧ b = 4) ∨ (a = 1 ∧ b = 7) :=
by
  sorry

-- Problem 3
theorem MinimumRentalCost (a b : ℕ) (h1: 3 * a + 4 * b = 31) 
  (h2: 100 * a + 120 * b = 940) :
  ∃ a b, a = 1 ∧ b = 7 :=
by
  sorry

end TruckCapacities_RentalPlanExists_MinimumRentalCost_l1082_108276


namespace food_initially_meant_to_last_22_days_l1082_108219

variable (D : ℕ)   -- Denoting the initial number of days the food was meant to last
variable (m : ℕ := 760)  -- Initial number of men
variable (total_men : ℕ := 1520)  -- Total number of men after 2 days

-- The first condition derived from the problem: total amount of food
def total_food := m * D

-- The second condition derived from the problem: Remaining food after 2 days
def remaining_food_after_2_days := total_food - m * 2

-- The third condition derived from the problem: Remaining food to last for 10 more days
def remaining_food_to_last_10_days := total_men * 10

-- Statement to prove
theorem food_initially_meant_to_last_22_days :
  D - 2 = 10 →
  D = 22 :=
by
  sorry

end food_initially_meant_to_last_22_days_l1082_108219


namespace water_usage_in_May_l1082_108258

theorem water_usage_in_May (x : ℝ) (h_cost : 45 = if x ≤ 12 then 2 * x 
                                                else if x ≤ 18 then 24 + 2.5 * (x - 12) 
                                                else 39 + 3 * (x - 18)) : x = 20 :=
sorry

end water_usage_in_May_l1082_108258


namespace pythagorean_triple_example_l1082_108244

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triple_example :
  is_pythagorean_triple 7 24 25 :=
sorry

end pythagorean_triple_example_l1082_108244


namespace graduating_class_total_students_l1082_108245

theorem graduating_class_total_students (boys girls students : ℕ) (h1 : girls = boys + 69) (h2 : boys = 208) :
  students = boys + girls → students = 485 :=
by
  sorry

end graduating_class_total_students_l1082_108245


namespace unique_two_digit_integer_solution_l1082_108259

variable {s : ℕ}

-- Conditions
def is_two_digit_positive_integer (s : ℕ) : Prop :=
  10 ≤ s ∧ s < 100

def last_two_digits_of_13s_are_52 (s : ℕ) : Prop :=
  13 * s % 100 = 52

-- Theorem statement
theorem unique_two_digit_integer_solution (h1 : is_two_digit_positive_integer s)
                                          (h2 : last_two_digits_of_13s_are_52 s) :
  s = 4 :=
sorry

end unique_two_digit_integer_solution_l1082_108259


namespace ordered_pairs_condition_l1082_108214

theorem ordered_pairs_condition (m n : ℕ) (hmn : m ≥ n) (hm_pos : 0 < m) (hn_pos : 0 < n) (h_eq : 3 * m * n = 8 * (m + n - 1)) :
    (m, n) = (16, 3) ∨ (m, n) = (6, 4) := by
  sorry

end ordered_pairs_condition_l1082_108214


namespace percentage_increase_in_sales_l1082_108212

theorem percentage_increase_in_sales (P S : ℝ) (hP : P > 0) (hS : S > 0) :
  (∃ X : ℝ, (0.8 * (1 + X / 100) = 1.44) ∧ X = 80) :=
sorry

end percentage_increase_in_sales_l1082_108212


namespace exists_root_in_interval_l1082_108207

noncomputable def f (x : ℝ) := 3^x + 3 * x - 8

theorem exists_root_in_interval :
  f 1 < 0 → f 1.5 > 0 → f 1.25 < 0 → ∃ x ∈ (Set.Ioo 1.25 1.5), f x = 0 :=
by
  intros h1 h2 h3
  sorry

end exists_root_in_interval_l1082_108207


namespace complement_M_l1082_108201

section ComplementSet

variable (x : ℝ)

def M : Set ℝ := {x | 1 / x < 1}

theorem complement_M : {x | 0 ≤ x ∧ x ≤ 1} = Mᶜ := sorry

end ComplementSet

end complement_M_l1082_108201


namespace pentagon_position_3010_l1082_108222

def rotate_72 (s : String) : String :=
match s with
| "ABCDE" => "EABCD"
| "EABCD" => "DCBAE"
| "DCBAE" => "EDABC"
| "EDABC" => "ABCDE"
| _ => s

def reflect_vertical (s : String) : String :=
match s with
| "EABCD" => "DCBAE"
| "DCBAE" => "EABCD"
| _ => s

def transform (s : String) (n : Nat) : String :=
match n % 5 with
| 0 => s
| 1 => reflect_vertical (rotate_72 s)
| 2 => rotate_72 (reflect_vertical (rotate_72 s))
| 3 => reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s)))
| 4 => rotate_72 (reflect_vertical (rotate_72 (reflect_vertical (rotate_72 s))))
| _ => s

theorem pentagon_position_3010 :
  transform "ABCDE" 3010 = "ABCDE" :=
by 
  sorry

end pentagon_position_3010_l1082_108222


namespace constant_term_of_expansion_l1082_108206

open BigOperators

noncomputable def binomialCoeff (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_of_expansion :
  ∑ r in Finset.range (6 + 1), binomialCoeff 6 r * (2^r * (x : ℚ)^r) / (x^3 : ℚ) = 160 :=
by
  sorry

end constant_term_of_expansion_l1082_108206


namespace mean_after_removal_l1082_108235

variable {n : ℕ}
variable {S : ℝ}
variable {S' : ℝ}
variable {mean_original : ℝ}
variable {size_original : ℕ}
variable {x1 : ℝ}
variable {x2 : ℝ}

theorem mean_after_removal (h_mean_original : mean_original = 42)
    (h_size_original : size_original = 60)
    (h_x1 : x1 = 50)
    (h_x2 : x2 = 60)
    (h_S : S = mean_original * size_original)
    (h_S' : S' = S - (x1 + x2)) :
    S' / (size_original - 2) = 41.55 :=
by
  sorry

end mean_after_removal_l1082_108235


namespace sum_six_consecutive_integers_l1082_108261

-- Statement of the problem
theorem sum_six_consecutive_integers (n : ℤ) :
  (n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5)) = 6 * n + 15 :=
by
  sorry

end sum_six_consecutive_integers_l1082_108261


namespace g_at_5_l1082_108263

variable (g : ℝ → ℝ)

-- Define the condition on g
def functional_condition : Prop :=
  ∀ x : ℝ, g x + 3 * g (1 - x) = 2 * x ^ 2 + 1

-- The statement proven should be g(5) = 8 given functional_condition
theorem g_at_5 (h : functional_condition g) : g 5 = 8 := by
  sorry

end g_at_5_l1082_108263


namespace probability_reach_correct_l1082_108262

noncomputable def probability_reach (n : ℕ) : ℚ :=
  (2/3) + (1/12) * (1 - (-1/3)^(n-1))

theorem probability_reach_correct (n : ℕ) (P_n : ℚ) :
  P_n = probability_reach n :=
by
  sorry

end probability_reach_correct_l1082_108262


namespace eval_expr_l1082_108256

theorem eval_expr : (3 : ℚ) / (2 - (5 / 4)) = 4 := by
  sorry

end eval_expr_l1082_108256


namespace combined_area_of_walls_l1082_108278

theorem combined_area_of_walls (A : ℕ) 
  (h1: ∃ (A : ℕ), A ≥ 0)
  (h2 : (A - 2 * 40 - 40 = 180)) :
  A = 300 := 
sorry

end combined_area_of_walls_l1082_108278


namespace solution_inequality_1_range_of_a_l1082_108257

noncomputable def f (x : ℝ) : ℝ := abs x + abs (x - 2)

theorem solution_inequality_1 :
  {x : ℝ | f x < 3} = {x : ℝ | - (1/2) < x ∧ x < (5/2)} :=
by
  sorry

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, f x < a) → a > 2 :=
by
  sorry

end solution_inequality_1_range_of_a_l1082_108257


namespace paula_candies_l1082_108255

def candies_per_friend (total_candies : ℕ) (number_of_friends : ℕ) : ℕ :=
  total_candies / number_of_friends

theorem paula_candies :
  let initial_candies := 20
  let additional_candies := 4
  let total_candies := initial_candies + additional_candies
  let number_of_friends := 6
  candies_per_friend total_candies number_of_friends = 4 :=
by
  sorry

end paula_candies_l1082_108255


namespace area_arccos_cos_eq_pi_sq_l1082_108215

noncomputable def area_bounded_by_arccos_cos : ℝ :=
  ∫ x in (0 : ℝ)..2 * Real.pi, Real.arccos (Real.cos x)

theorem area_arccos_cos_eq_pi_sq :
  area_bounded_by_arccos_cos = Real.pi ^ 2 :=
sorry

end area_arccos_cos_eq_pi_sq_l1082_108215


namespace remainder_8357_to_8361_div_9_l1082_108283

theorem remainder_8357_to_8361_div_9 :
  (8357 + 8358 + 8359 + 8360 + 8361) % 9 = 3 := 
by
  sorry

end remainder_8357_to_8361_div_9_l1082_108283


namespace range_of_a_l1082_108253

theorem range_of_a (a : ℝ) : 
  4 * a^2 - 12 * (a + 6) > 0 ↔ a < -3 ∨ a > 6 := 
by sorry

end range_of_a_l1082_108253


namespace range_of_m_l1082_108209

open Real

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ (x : ℝ), x > 0 → log x ≤ x * exp (m^2 - m - 1)

theorem range_of_m : 
  {m : ℝ | satisfies_inequality m} = {m : ℝ | m ≤ 0 ∨ m ≥ 1} :=
by 
  sorry

end range_of_m_l1082_108209


namespace find_y_eq_7_5_l1082_108213

theorem find_y_eq_7_5 (y : ℝ) (hy1 : 0 < y) (hy2 : ∃ z : ℤ, ((z : ℝ) ≤ y) ∧ (y < z + 1))
  (hy3 : (Int.floor y : ℝ) * y = 45) : y = 7.5 :=
sorry

end find_y_eq_7_5_l1082_108213


namespace prob_three_friends_same_group_l1082_108298

theorem prob_three_friends_same_group :
  let students := 800
  let groups := 4
  let group_size := students / groups
  let p_same_group := 1 / groups
  p_same_group * p_same_group = 1 / 16 := 
by
  sorry

end prob_three_friends_same_group_l1082_108298


namespace parabola_directrix_l1082_108270

theorem parabola_directrix (p : ℝ) (h_focus : ∃ x y : ℝ, y^2 = 2*p*x ∧ 2*x + 3*y - 4 = 0) : 
  ∀ x y : ℝ, y^2 = 2*p*x → x = -p/2 := 
sorry

end parabola_directrix_l1082_108270


namespace hyperbola_eccentricity_l1082_108230

noncomputable def eccentricity (a b : ℝ) : ℝ :=
  Real.sqrt (1 + (a^2) / (b^2))

theorem hyperbola_eccentricity {b : ℝ} (hb_pos : b > 0)
  (h_area : b = 1) :
  eccentricity 1 b = Real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l1082_108230


namespace total_laps_jogged_l1082_108289

-- Defining the conditions
def jogged_PE_class : ℝ := 1.12
def jogged_track_practice : ℝ := 2.12

-- Statement to prove
theorem total_laps_jogged : jogged_PE_class + jogged_track_practice = 3.24 := by
  -- Proof would go here
  sorry

end total_laps_jogged_l1082_108289


namespace trigonometric_identity_l1082_108210

theorem trigonometric_identity (t : ℝ) : 
  5.43 * Real.cos (22 * Real.pi / 180 - t) * Real.cos (82 * Real.pi / 180 - t) +
  Real.cos (112 * Real.pi / 180 - t) * Real.cos (172 * Real.pi / 180 - t) = 
  0.5 * (Real.sin t + Real.cos t) :=
sorry

end trigonometric_identity_l1082_108210


namespace find_gamma_l1082_108292

variable (γ δ : ℝ)

def directly_proportional (γ δ : ℝ) : Prop := ∃ c : ℝ, γ = c * δ

theorem find_gamma (h1 : directly_proportional γ δ) (h2 : γ = 5) (h3 : δ = -10) : δ = 25 → γ = -25 / 2 := by
  sorry

end find_gamma_l1082_108292


namespace total_chairs_calculation_l1082_108217

-- Definitions of the conditions
def numIndoorTables : Nat := 9
def numOutdoorTables : Nat := 11
def chairsPerIndoorTable : Nat := 10
def chairsPerOutdoorTable : Nat := 3

-- The proposition we want to prove
theorem total_chairs_calculation :
  numIndoorTables * chairsPerIndoorTable + numOutdoorTables * chairsPerOutdoorTable = 123 := by
sorry

end total_chairs_calculation_l1082_108217


namespace candy_boxes_system_l1082_108281

-- Given conditions and definitions
def sheets_total (x y : ℕ) : Prop := x + y = 35
def sheet_usage (x y : ℕ) : Prop := 20 * x = 30 * y / 2

-- Statement
theorem candy_boxes_system (x y : ℕ) (h1 : sheets_total x y) (h2 : sheet_usage x y) : 
  (x + y = 35) ∧ (20 * x = 30 * y / 2) := 
by
sorry

end candy_boxes_system_l1082_108281


namespace ninth_group_number_l1082_108293

-- Conditions
def num_workers : ℕ := 100
def sample_size : ℕ := 20
def group_size : ℕ := num_workers / sample_size
def fifth_group_number : ℕ := 23

-- Theorem stating the result for the 9th group number.
theorem ninth_group_number : ∃ n : ℕ, n = 43 :=
by
  -- We calculate the numbers step by step.
  have interval : ℕ := group_size
  have difference : ℕ := 9 - 5
  have increment : ℕ := difference * interval
  have ninth_group_num : ℕ := fifth_group_number + increment
  use ninth_group_num
  sorry

end ninth_group_number_l1082_108293


namespace division_quotient_l1082_108225

theorem division_quotient (dividend divisor remainder quotient : ℕ)
  (H1 : dividend = 190)
  (H2 : divisor = 21)
  (H3 : remainder = 1)
  (H4 : dividend = divisor * quotient + remainder) : quotient = 9 :=
by {
  sorry
}

end division_quotient_l1082_108225


namespace path_length_of_B_l1082_108205

noncomputable def lengthPathB (BC : ℝ) : ℝ :=
  let radius := BC
  let circumference := 2 * Real.pi * radius
  circumference

theorem path_length_of_B (BC : ℝ) (h : BC = 4 / Real.pi) : lengthPathB BC = 8 := by
  rw [lengthPathB, h]
  simp [Real.pi_ne_zero, div_mul_cancel]
  sorry

end path_length_of_B_l1082_108205


namespace find_white_daisies_l1082_108273

theorem find_white_daisies (W P R : ℕ) 
  (h1 : P = 9 * W) 
  (h2 : R = 4 * P - 3) 
  (h3 : W + P + R = 273) : 
  W = 6 :=
by
  sorry

end find_white_daisies_l1082_108273


namespace hypotenuse_length_l1082_108220

theorem hypotenuse_length (a b : ℝ) (c : ℝ) (h₁ : a = Real.sqrt 5) (h₂ : b = Real.sqrt 12) : c = Real.sqrt 17 :=
by
  -- Proof not required, hence skipped with 'sorry'
  sorry

end hypotenuse_length_l1082_108220


namespace square_side_length_l1082_108296

variable (s : ℝ)
variable (k : ℝ := 6)

theorem square_side_length :
  s^2 = k * 4 * s → s = 24 :=
by
  intro h
  sorry

end square_side_length_l1082_108296


namespace find_circle_equation_l1082_108241

noncomputable def center_of_parabola : ℝ × ℝ := (1, 0)

noncomputable def tangent_line (x y : ℝ) : Prop := 3 * x + 4 * y + 2 = 0

noncomputable def equation_of_circle (x y : ℝ) : Prop := (x - 1) ^ 2 + y ^ 2 = 1

theorem find_circle_equation 
  (center_c : ℝ × ℝ := center_of_parabola)
  (tangent : ∀ x y, tangent_line x y → (x - 1) ^ 2 + (y - 0) ^ 2 = 1) :
  equation_of_circle = (fun x y => sorry) :=
sorry

end find_circle_equation_l1082_108241


namespace intersect_at_0_intersect_at_180_intersect_at_90_l1082_108221

-- Define radii R and r, and the distance c
variables {R r c : ℝ}

-- Formalize the conditions and corresponding angles
theorem intersect_at_0 (h : c = R - r) : True := 
sorry

theorem intersect_at_180 (h : c = R + r) : True := 
sorry

theorem intersect_at_90 (h : c = Real.sqrt (R^2 + r^2)) : True := 
sorry

end intersect_at_0_intersect_at_180_intersect_at_90_l1082_108221


namespace intersection_of_M_and_N_l1082_108218

namespace ProofProblem

def M := { x : ℝ | x^2 < 4 }
def N := { x : ℝ | x < 1 }

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | -2 < x ∧ x < 1 } :=
sorry

end ProofProblem

end intersection_of_M_and_N_l1082_108218


namespace value_of_f_x_plus_5_l1082_108277

open Function

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x + 1

-- State the theorem
theorem value_of_f_x_plus_5 (x : ℝ) : f (x + 5) = 3 * x + 16 :=
by
  sorry

end value_of_f_x_plus_5_l1082_108277


namespace number_of_people_who_selected_dog_l1082_108234

theorem number_of_people_who_selected_dog 
  (total : ℕ) 
  (cat : ℕ) 
  (fish : ℕ) 
  (bird : ℕ) 
  (other : ℕ) 
  (h_total : total = 90) 
  (h_cat : cat = 25) 
  (h_fish : fish = 10) 
  (h_bird : bird = 15) 
  (h_other : other = 5) :
  (total - (cat + fish + bird + other) = 35) :=
by
  sorry

end number_of_people_who_selected_dog_l1082_108234


namespace unique_root_of_linear_equation_l1082_108282

theorem unique_root_of_linear_equation (a b : ℝ) (h : a ≠ 0) : ∃! x : ℝ, a * x = b :=
by
  sorry

end unique_root_of_linear_equation_l1082_108282


namespace range_of_a_l1082_108286

theorem range_of_a (a : ℝ) :
  (∀ x, (x < -1 ∨ x > 5) ∨ (a < x ∧ x < a + 8)) ↔ (-3 < a ∧ a < -1) :=
by
  sorry

end range_of_a_l1082_108286


namespace polar_to_rectangular_l1082_108208

noncomputable def curve_equation (θ : ℝ) : ℝ := 2 * Real.cos θ

theorem polar_to_rectangular (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) :
  ∃ (x y : ℝ), (x - 1) ^ 2 + y ^ 2 = 1 ∧ 0 ≤ y ∧ y ≤ 1 ∧
  (x = curve_equation θ * Real.cos θ ∧ y = curve_equation θ * Real.sin θ) :=
sorry

end polar_to_rectangular_l1082_108208


namespace annulus_divide_l1082_108295

theorem annulus_divide (r : ℝ) (h₁ : 2 < 14) (h₂ : 2 > 0) (h₃ : 14 > 0)
    (h₄ : π * 196 - π * r^2 = π * r^2 - π * 4) : r = 10 := 
sorry

end annulus_divide_l1082_108295


namespace find_n_from_binomial_variance_l1082_108236

variable (ξ : Type)
variable (n : ℕ)
variable (p : ℝ := 0.3)
variable (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p))

-- Given conditions
axiom binomial_distribution : p = 0.3 ∧ Var n p = 2.1

-- Prove n = 10
theorem find_n_from_binomial_variance (ξ : Type) (n : ℕ) (p : ℝ := 0.3) (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p)) :
  p = 0.3 ∧ Var n p = 2.1 → n = 10 :=
by
  sorry

end find_n_from_binomial_variance_l1082_108236


namespace larger_number_is_50_l1082_108287

theorem larger_number_is_50 (x y : ℤ) (h1 : 4 * y = 5 * x) (h2 : y - x = 10) : y = 50 :=
sorry

end larger_number_is_50_l1082_108287


namespace quadrilateral_property_indeterminate_l1082_108260

variable {α : Type*}
variable (Q A : α → Prop)

theorem quadrilateral_property_indeterminate :
  (¬ ∀ x, Q x → A x) → ¬ ((∃ x, Q x ∧ A x) ↔ False) :=
by
  intro h
  sorry

end quadrilateral_property_indeterminate_l1082_108260


namespace find_x_l1082_108211

-- Define the condition as a Lean equation
def equation (x : ℤ) : Prop :=
  45 - (28 - (37 - (x - 19))) = 58

-- The proof statement: if the equation holds, then x = 15
theorem find_x (x : ℤ) (h : equation x) : x = 15 := by
  sorry

end find_x_l1082_108211


namespace number_of_true_statements_l1082_108284

theorem number_of_true_statements 
  (a b c : ℝ) 
  (Hc : c ≠ 0) : 
  ((a > b → a * c^2 > b * c^2) ∧ (a * c^2 ≤ b * c^2 → a ≤ b)) ∧ 
  ¬((a * c^2 > b * c^2 → a > b) ∨ (a ≤ b → a * c^2 ≤ b * c^2)) :=
by
  sorry

end number_of_true_statements_l1082_108284


namespace non_zero_digits_fraction_l1082_108291

def count_non_zero_digits (n : ℚ) : ℕ :=
  -- A placeholder for the actual implementation.
  sorry

theorem non_zero_digits_fraction : count_non_zero_digits (120 / (2^4 * 5^9 : ℚ)) = 3 :=
  sorry

end non_zero_digits_fraction_l1082_108291


namespace max_f_and_sin_alpha_l1082_108268

noncomputable def f (x : ℝ) : ℝ := Real.sin x + 2 * Real.cos x

theorem max_f_and_sin_alpha :
  (∀ x : ℝ, f x ≤ Real.sqrt 5) ∧ (∃ α : ℝ, (α + Real.arccos (1 / Real.sqrt 5) = π / 2 + 2 * π * some_integer) ∧ (f α = Real.sqrt 5) ∧ (Real.sin α = 1 / Real.sqrt 5)) :=
by
  sorry

end max_f_and_sin_alpha_l1082_108268


namespace multiplication_result_l1082_108240

theorem multiplication_result :
  10 * 9.99 * 0.999 * 100 = (99.9)^2 := 
by
  sorry

end multiplication_result_l1082_108240


namespace sixteen_pow_five_eq_four_pow_p_l1082_108280

theorem sixteen_pow_five_eq_four_pow_p (p : ℕ) (h : 16^5 = 4^p) : p = 10 := 
  sorry

end sixteen_pow_five_eq_four_pow_p_l1082_108280


namespace coin_flip_sequences_l1082_108223

theorem coin_flip_sequences :
  let total_sequences := 2^10
  let sequences_starting_with_two_heads := 2^8
  total_sequences - sequences_starting_with_two_heads = 768 :=
by
  sorry

end coin_flip_sequences_l1082_108223


namespace number_machine_output_l1082_108252

def number_machine (n : ℕ) : ℕ :=
  let step1 := n * 3
  let step2 := step1 + 20
  let step3 := step2 / 2
  let step4 := step3 ^ 2
  let step5 := step4 - 45
  step5

theorem number_machine_output : number_machine 90 = 20980 := by
  sorry

end number_machine_output_l1082_108252


namespace triangle_sides_consecutive_and_angle_relationship_l1082_108274

theorem triangle_sides_consecutive_and_angle_relationship (a b c : ℕ) 
  (h1 : a < b) (h2 : b < c) (h3 : b = a + 1) (h4 : c = b + 1) 
  (angle_A angle_B angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_B + angle_C = π) 
  (h_angle_relation : angle_B = 2 * angle_A) : 
  (a, b, c) = (4, 5, 6) :=
sorry

end triangle_sides_consecutive_and_angle_relationship_l1082_108274


namespace value_to_add_l1082_108237

theorem value_to_add (a b c n m : ℕ) (h₁ : a = 510) (h₂ : b = 4590) (h₃ : c = 105) (h₄ : n = 627) (h₅ : m = Nat.lcm a (Nat.lcm b c)) :
  m - n = 31503 :=
by
  sorry

end value_to_add_l1082_108237


namespace hyperbola_focus_l1082_108227

theorem hyperbola_focus (m : ℝ) :
  (∃ (F : ℝ × ℝ), F = (0, 5) ∧ F ∈ {P : ℝ × ℝ | ∃ x y : ℝ, 
  x = P.1 ∧ y = P.2 ∧ (y^2 / m - x^2 / 9 = 1)}) → 
  m = 16 :=
by
  sorry

end hyperbola_focus_l1082_108227


namespace polynomial_factors_sum_l1082_108269

theorem polynomial_factors_sum (a b : ℝ) 
  (h : ∃ c : ℝ, (∀ x: ℝ, x^3 + a * x^2 + b * x + 8 = (x + 1) * (x + 2) * (x + c))) : 
  a + b = 21 :=
sorry

end polynomial_factors_sum_l1082_108269


namespace scale_division_l1082_108271

theorem scale_division (total_feet : ℕ) (inches_extra : ℕ) (part_length : ℕ) (total_parts : ℕ) :
  total_feet = 6 → inches_extra = 8 → part_length = 20 → 
  total_parts = (6 * 12 + 8) / 20 → total_parts = 4 :=
by
  intros
  sorry

end scale_division_l1082_108271


namespace students_correct_answers_l1082_108247

theorem students_correct_answers
  (total_questions : ℕ)
  (correct_score per_question : ℕ)
  (incorrect_penalty : ℤ)
  (xiao_ming_score xiao_hong_score xiao_hua_score : ℤ)
  (xm_correct_answers xh_correct_answers xh_correct_answers : ℕ)
  (total : ℕ)
  (h_1 : total_questions = 10)
  (h_2 : correct_score = 10)
  (h_3 : incorrect_penalty = -3)
  (h_4 : xiao_ming_score = 87)
  (h_5 : xiao_hong_score = 74)
  (h_6 : xiao_hua_score = 9)
  (h_xm : xm_correct_answers = total_questions - (xiao_ming_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hong_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (h_xh : xh_correct_answers = total_questions - (xiao_hua_score - total_questions * correct_score) / (correct_score - incorrect_penalty))
  (expected : total = 20) :
  xm_correct_answers + xh_correct_answers + xh_correct_answers = total := 
sorry

end students_correct_answers_l1082_108247


namespace find_n_l1082_108226

theorem find_n (x : ℝ) (h1 : x = 596.95) (h2 : ∃ n : ℝ, n + 11.95 - x = 3054) : ∃ n : ℝ, n = 3639 :=
by
  sorry

end find_n_l1082_108226


namespace intersection_equal_l1082_108275

noncomputable def M := { y : ℝ | ∃ x : ℝ, y = Real.log (x + 1) / Real.log (1 / 2) ∧ x ≥ 3 }
noncomputable def N := { x : ℝ | x^2 + 2 * x - 3 ≤ 0 }

theorem intersection_equal : M ∩ N = {a : ℝ | -3 ≤ a ∧ a ≤ -2} :=
by
  sorry

end intersection_equal_l1082_108275


namespace distribute_money_equation_l1082_108248

theorem distribute_money_equation (x : ℕ) (hx : x > 0) : 
  (10 : ℚ) / x = (40 : ℚ) / (x + 6) := 
sorry

end distribute_money_equation_l1082_108248


namespace fractional_part_of_students_who_walk_home_l1082_108254

def fraction_bus := 1 / 3
def fraction_automobile := 1 / 5
def fraction_bicycle := 1 / 8
def fraction_scooter := 1 / 10

theorem fractional_part_of_students_who_walk_home :
  (1 : ℚ) - (fraction_bus + fraction_automobile + fraction_bicycle + fraction_scooter) = 29 / 120 :=
by
  sorry

end fractional_part_of_students_who_walk_home_l1082_108254


namespace standard_concession_l1082_108265

theorem standard_concession (x : ℝ) : 
  (∀ (x : ℝ), (2000 - (x / 100) * 2000) - 0.2 * (2000 - (x / 100) * 2000) = 1120) → x = 30 := 
by 
  sorry

end standard_concession_l1082_108265
