import Mathlib

namespace abs_eq_case_solution_l391_39114

theorem abs_eq_case_solution :
  ∃! x : ℝ, |x - 2| = |x - 3| + |x - 4| + |x - 5| := sorry

end abs_eq_case_solution_l391_39114


namespace trajectory_equation_l391_39153

theorem trajectory_equation (x y a : ℝ) (h : x^2 + y^2 = a^2) :
  (x - y)^2 + 2*x*y = a^2 :=
by
  sorry

end trajectory_equation_l391_39153


namespace circle_equation_through_points_l391_39137

theorem circle_equation_through_points 
  (M N : ℝ × ℝ)
  (hM : M = (5, 2))
  (hN : N = (3, 2))
  (hk : ∃ k : ℝ, (M.1 + N.1) / 2 = k ∧ (M.2 + N.2) / 2 = (2 * k - 3))
  : (∃ h : ℝ, ∀ x y: ℝ, (x - 4) ^ 2 + (y - 5) ^ 2 = h) ∧ (∃ r : ℝ, r = 10) := 
sorry

end circle_equation_through_points_l391_39137


namespace attendees_received_all_items_l391_39107

theorem attendees_received_all_items {n : ℕ} (h1 : ∀ k, k ∣ 45 → n % k = 0) (h2 : ∀ k, k ∣ 75 → n % k = 0) (h3 : ∀ k, k ∣ 100 → n % k = 0) (h4 : n = 4500) :
  (4500 / Nat.lcm (Nat.lcm 45 75) 100) = 5 :=
by
  sorry

end attendees_received_all_items_l391_39107


namespace total_amount_shared_l391_39109

theorem total_amount_shared
  (A B C : ℕ)
  (h_ratio : A / 2 = B / 3 ∧ B / 3 = C / 8)
  (h_Ben_share : B = 30) : A + B + C = 130 :=
by
  -- Add placeholder for the proof.
  sorry

end total_amount_shared_l391_39109


namespace Bridget_skittles_after_giving_l391_39113

-- Given conditions
def Bridget_initial_skittles : ℕ := 4
def Henry_skittles : ℕ := 4
def Henry_gives_all_to_Bridget : Prop := True

-- Prove that Bridget will have 8 Skittles in total after Henry gives all of his Skittles to her.
theorem Bridget_skittles_after_giving (h : Henry_gives_all_to_Bridget) :
  Bridget_initial_skittles + Henry_skittles = 8 :=
by
  sorry

end Bridget_skittles_after_giving_l391_39113


namespace circle_equation_through_points_l391_39101

theorem circle_equation_through_points 
  (D E F : ℝ)
  (h_eq1 : 0^2 + 0^2 + D*0 + E*0 + F = 0)
  (h_eq2 : 4^2 + 0^2 + D*4 + E*0 + F = 0)
  (h_eq3 : (-1)^2 + 1^2 + D*(-1) + E*1 + F = 0) :
  ∃ D E : ℝ, F = 0 ∧ D = -4 ∧ E = -6 ∧ 
  x^2 + y^2 + D*x + E*y + F = (x^2 + y^2 - 4*x - 6*y) := 
sorry

end circle_equation_through_points_l391_39101


namespace infinite_primes_4k1_l391_39162

theorem infinite_primes_4k1 : ∀ (P : List ℕ), (∀ (p : ℕ), p ∈ P → Nat.Prime p ∧ ∃ k, p = 4 * k + 1) → 
  ∃ q, Nat.Prime q ∧ ∃ k, q = 4 * k + 1 ∧ q ∉ P :=
sorry

end infinite_primes_4k1_l391_39162


namespace graph_passes_through_0_1_l391_39150

theorem graph_passes_through_0_1 (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : (0, 1) ∈ { p : ℝ × ℝ | ∃ x : ℝ, p = (x, a^x) } :=
sorry

end graph_passes_through_0_1_l391_39150


namespace cadastral_value_of_land_l391_39188

theorem cadastral_value_of_land (tax_amount : ℝ) (tax_rate : ℝ) (V : ℝ)
    (h1 : tax_amount = 4500)
    (h2 : tax_rate = 0.003) :
    V = 1500000 :=
by
  sorry

end cadastral_value_of_land_l391_39188


namespace opposite_sqrt_4_l391_39172

theorem opposite_sqrt_4 : - (Real.sqrt 4) = -2 := sorry

end opposite_sqrt_4_l391_39172


namespace average_age_union_l391_39103

theorem average_age_union (students_A students_B students_C : ℕ)
  (sumA sumB sumC : ℕ) (avgA avgB avgC avgAB avgAC avgBC : ℚ)
  (hA : avgA = (sumA : ℚ) / students_A)
  (hB : avgB = (sumB : ℚ) / students_B)
  (hC : avgC = (sumC : ℚ) / students_C)
  (hAB : avgAB = (sumA + sumB) / (students_A + students_B))
  (hAC : avgAC = (sumA + sumC) / (students_A + students_C))
  (hBC : avgBC = (sumB + sumC) / (students_B + students_C))
  (h_avgA: avgA = 34)
  (h_avgB: avgB = 25)
  (h_avgC: avgC = 45)
  (h_avgAB: avgAB = 30)
  (h_avgAC: avgAC = 42)
  (h_avgBC: avgBC = 36) :
  (sumA + sumB + sumC : ℚ) / (students_A + students_B + students_C) = 33 := 
  sorry

end average_age_union_l391_39103


namespace find_principal_amount_l391_39182

def interest_rate_first_year : ℝ := 0.10
def compounding_periods_first_year : ℕ := 2
def interest_rate_second_year : ℝ := 0.12
def compounding_periods_second_year : ℕ := 4
def diff_interest : ℝ := 12

theorem find_principal_amount (P : ℝ)
  (h1_first : interest_rate_first_year / (compounding_periods_first_year : ℝ) = 0.05)
  (h1_second : interest_rate_second_year / (compounding_periods_second_year : ℝ) = 0.03)
  (compounded_amount : ℝ := P * (1 + 0.05)^(compounding_periods_first_year) * (1 + 0.03)^compounding_periods_second_year)
  (simple_interest : ℝ := P * (interest_rate_first_year + interest_rate_second_year) / 2 * 2)
  (h_diff : compounded_amount - P - simple_interest = diff_interest) : P = 597.01 :=
sorry

end find_principal_amount_l391_39182


namespace price_of_basic_computer_l391_39157

-- Definitions for the prices
variables (C_b P M K C_e : ℝ)

-- Conditions
axiom h1 : C_b + P + M + K = 2500
axiom h2 : C_e + P + M + K = 3100
axiom h3 : P = (3100 / 6)
axiom h4 : M = (3100 / 5)
axiom h5 : K = (3100 / 8)
axiom h6 : C_e = C_b + 600

-- Theorem stating the price of the basic computer
theorem price_of_basic_computer : C_b = 975.83 :=
by {
  sorry
}

end price_of_basic_computer_l391_39157


namespace percentage_increase_in_radius_l391_39175

theorem percentage_increase_in_radius (r R : ℝ) (h : π * R^2 = π * r^2 + 1.25 * (π * r^2)) :
  R = 1.5 * r :=
by
  -- Proof goes here
  sorry

end percentage_increase_in_radius_l391_39175


namespace brother_books_total_l391_39124

-- Define the conditions
def sarah_paperbacks : ℕ := 6
def sarah_hardbacks : ℕ := 4
def brother_paperbacks : ℕ := sarah_paperbacks / 3
def brother_hardbacks : ℕ := 2 * sarah_hardbacks

-- Define the statement to be proven
theorem brother_books_total : brother_paperbacks + brother_hardbacks = 10 :=
by
  -- Proof will be added here
  sorry

end brother_books_total_l391_39124


namespace ben_bonus_leftover_l391_39146

theorem ben_bonus_leftover (b : ℝ) (k h c : ℝ) (bk : k = 1/22 * b) (bh : h = 1/4 * b) (bc : c = 1/8 * b) :
  b - (k + h + c) = 867 :=
by
  sorry

end ben_bonus_leftover_l391_39146


namespace intersection_of_A_and_B_l391_39142

def A := {x : ℝ | |x - 2| ≤ 1}
def B := {x : ℝ | x^2 - 2 * x - 3 < 0}
def C := {x : ℝ | 1 ≤ x ∧ x < 3}

theorem intersection_of_A_and_B : A ∩ B = C := by
  sorry

end intersection_of_A_and_B_l391_39142


namespace number_of_walls_l391_39174

theorem number_of_walls (bricks_per_row rows_per_wall total_bricks : Nat) :
  bricks_per_row = 30 → 
  rows_per_wall = 50 → 
  total_bricks = 3000 → 
  total_bricks / (bricks_per_row * rows_per_wall) = 2 := 
by
  intros h1 h2 h3
  sorry

end number_of_walls_l391_39174


namespace percentage_no_job_diploma_l391_39149

def percentage_with_university_diploma {total_population : ℕ} (has_diploma : ℕ) : ℕ :=
  (has_diploma / total_population) * 100

variables {total_population : ℕ} (p_no_diploma_and_job : ℕ) (p_with_job : ℕ) (p_diploma : ℕ)

axiom percentage_no_diploma_job :
  p_no_diploma_and_job = 10

axiom percentage_with_job :
  p_with_job = 40

axiom percentage_diploma :
  p_diploma = 39

theorem percentage_no_job_diploma :
  ∃ p : ℕ, p = (9 / 60) * 100 := sorry

end percentage_no_job_diploma_l391_39149


namespace oxygen_atoms_in_compound_l391_39167

-- Define given conditions as parameters in the problem.
def number_of_oxygen_atoms (molecular_weight : ℕ) (weight_Al : ℕ) (weight_H : ℕ) (weight_O : ℕ) (atoms_Al : ℕ) (atoms_H : ℕ) (weight : ℕ) : ℕ := 
  (weight - (atoms_Al * weight_Al + atoms_H * weight_H)) / weight_O

-- Define the actual problem using the defined conditions.
theorem oxygen_atoms_in_compound
  (molecular_weight : ℕ := 78) 
  (weight_Al : ℕ := 27) 
  (weight_H : ℕ := 1) 
  (weight_O : ℕ := 16) 
  (atoms_Al : ℕ := 1) 
  (atoms_H : ℕ := 3) : 
  number_of_oxygen_atoms molecular_weight weight_Al weight_H weight_O atoms_Al atoms_H molecular_weight = 3 := 
sorry

end oxygen_atoms_in_compound_l391_39167


namespace inequality_of_products_l391_39126

theorem inequality_of_products
  (a b c d : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hc : 0 < c) 
  (hd : 0 < d)
  (h : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a * b + a * c + a * d + b * c + b * d + c * d ≥ 10 :=
by
  sorry

end inequality_of_products_l391_39126


namespace number_of_good_students_is_5_or_7_l391_39119

-- Definitions based on the conditions
def total_students : ℕ := 25
def number_of_good_students (G : ℕ) (T : ℕ) := G + T = total_students
def first_group_condition (T : ℕ) := T > 12
def second_group_condition (G : ℕ) (T : ℕ) := T = 3 * (G - 1)

-- Problem statement in Lean 4:
theorem number_of_good_students_is_5_or_7 (G T : ℕ) :
  number_of_good_students G T ∧ first_group_condition T ∧ second_group_condition G T → G = 5 ∨ G = 7 :=
by
  sorry

end number_of_good_students_is_5_or_7_l391_39119


namespace rate_of_interest_per_annum_l391_39156

theorem rate_of_interest_per_annum (SI P : ℝ) (T : ℕ) (hSI : SI = 4016.25) (hP : P = 10040.625) (hT : T = 5) :
  (SI * 100) / (P * T) = 8 :=
by 
  -- Given simple interest formula
  -- SI = P * R * T / 100, solving for R we get R = (SI * 100) / (P * T)
  -- Substitute SI = 4016.25, P = 10040.625, and T = 5
  -- (4016.25 * 100) / (10040.625 * 5) = 8
  sorry

end rate_of_interest_per_annum_l391_39156


namespace component_unqualified_l391_39112

theorem component_unqualified :
  ∀ (φ : ℝ), (19.98 ≤ φ ∧ φ ≤ 20.02) → ¬(φ = 19.9) → True :=
by
  intro φ
  intro h
  intro h'
  -- skip proof
  sorry

end component_unqualified_l391_39112


namespace paul_crayons_left_l391_39133

theorem paul_crayons_left (initial_crayons lost_crayons : ℕ) 
  (h_initial : initial_crayons = 253) 
  (h_lost : lost_crayons = 70) : (initial_crayons - lost_crayons) = 183 := 
by
  sorry

end paul_crayons_left_l391_39133


namespace trip_time_difference_l391_39140

theorem trip_time_difference
  (avg_speed : ℝ)
  (dist1 dist2 : ℝ)
  (h_avg_speed : avg_speed = 60)
  (h_dist1 : dist1 = 540)
  (h_dist2 : dist2 = 570) :
  ((dist2 - dist1) / avg_speed) * 60 = 30 := by
  sorry

end trip_time_difference_l391_39140


namespace coral_three_night_total_pages_l391_39168

-- Definitions based on conditions in the problem
def night1_pages : ℕ := 30
def night2_pages : ℕ := 2 * night1_pages - 2
def night3_pages : ℕ := night1_pages + night2_pages + 3
def total_pages : ℕ := night1_pages + night2_pages + night3_pages

-- The statement we want to prove
theorem coral_three_night_total_pages : total_pages = 179 := by
  sorry

end coral_three_night_total_pages_l391_39168


namespace correct_proposition_is_D_l391_39139

-- Define the propositions
def propositionA : Prop :=
  (∀ x : ℝ, x^2 = 4 → x = 2 ∨ x = -2) → (∀ x : ℝ, (x ≠ 2 ∨ x ≠ -2) → x^2 ≠ 4)

def propositionB (p : Prop) : Prop :=
  (p → (∀ x : ℝ, x^2 - 2*x + 3 > 0)) → (¬p → (∃ x : ℝ, x^2 - 2*x + 3 < 0))

def propositionC : Prop :=
  ∀ (a b : ℝ) (n : ℕ), a > b → n > 0 → a^n > b^n

def p : Prop := ∀ x : ℝ, x^3 ≥ 0
def q : Prop := ∀ e : ℝ, e > 0 → e < 1
def propositionD := p ∧ q

-- The proof problem
theorem correct_proposition_is_D : propositionD :=
  sorry

end correct_proposition_is_D_l391_39139


namespace probability_of_same_color_balls_l391_39105

-- Definitions of the problem
def total_balls_bag_A := 8 + 4
def total_balls_bag_B := 6 + 6
def white_balls_bag_A := 8
def red_balls_bag_A := 4
def white_balls_bag_B := 6
def red_balls_bag_B := 6

def P (event: Nat -> Bool) (total: Nat) : Nat :=
  let favorable := (List.range total).filter event |>.length
  favorable / total

-- Probability of drawing a white ball from bag A
def P_A := P (λ n => n < white_balls_bag_A) total_balls_bag_A

-- Probability of drawing a red ball from bag A
def P_not_A := P (λ n => n >= white_balls_bag_A && n < total_balls_bag_A) total_balls_bag_A

-- Probability of drawing a white ball from bag B
def P_B := P (λ n => n < white_balls_bag_B) total_balls_bag_B

-- Probability of drawing a red ball from bag B
def P_not_B := P (λ n => n >= white_balls_bag_B && n < total_balls_bag_B) total_balls_bag_B

-- Independence assumption (product rule for independent events)
noncomputable def P_same_color := P_A * P_B + P_not_A * P_not_B

-- Final theorem to prove
theorem probability_of_same_color_balls :
  P_same_color = 1 / 2 := by
    sorry

end probability_of_same_color_balls_l391_39105


namespace johns_trip_distance_is_160_l391_39189

noncomputable def total_distance (y : ℕ) : Prop :=
  y / 2 + 40 + y / 4 = y

theorem johns_trip_distance_is_160 : ∃ y : ℕ, total_distance y ∧ y = 160 :=
by
  use 160
  unfold total_distance
  sorry

end johns_trip_distance_is_160_l391_39189


namespace evaluate_expression_l391_39171

theorem evaluate_expression :
  (∃ (a b c : ℕ), a = 18 ∧ b = 3 ∧ c = 54 ∧ c = a * b ∧ (18^36 / 54^18) = (6^18)) :=
sorry

end evaluate_expression_l391_39171


namespace simplify_problem_l391_39151

noncomputable def simplify_expression : ℝ :=
  let numer := (Real.sqrt 3 - 1) ^ (1 - Real.sqrt 2)
  let denom := (Real.sqrt 3 + 1) ^ (1 + Real.sqrt 2)
  numer / denom

theorem simplify_problem :
  simplify_expression = 2 ^ (1 - Real.sqrt 2) * (4 - 2 * Real.sqrt 3) :=
by
  sorry

end simplify_problem_l391_39151


namespace find_number_l391_39161

-- Define the condition given in the problem
def condition (x : ℕ) : Prop :=
  x / 5 + 6 = 65

-- Prove that the solution satisfies the condition
theorem find_number : ∃ x : ℕ, condition x ∧ x = 295 :=
by
  -- Skip the actual proof steps
  sorry

end find_number_l391_39161


namespace directrix_parabola_l391_39193

-- Given the equation of the parabola and required transformations:
theorem directrix_parabola (d : ℚ) : 
  (∀ x : ℚ, y = -4 * x^2 + 4) → d = 65 / 16 :=
by sorry

end directrix_parabola_l391_39193


namespace sufficient_not_necessary_of_and_false_or_true_l391_39118

variables (p q : Prop)

theorem sufficient_not_necessary_of_and_false_or_true :
  (¬(p ∧ q) → (p ∨ q)) ∧ ((p ∨ q) → ¬(¬(p ∧ q))) :=
sorry

end sufficient_not_necessary_of_and_false_or_true_l391_39118


namespace trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l391_39190

open Real

theorem trig_cos2_minus_sin2_eq_neg_sqrt5_div3 (α : ℝ) (hα1 : 0 < α ∧ α < π) (hα2 : sin α + cos α = sqrt 3 / 3) :
  cos α ^ 2 - sin α ^ 2 = - sqrt 5 / 3 := 
  sorry

end trig_cos2_minus_sin2_eq_neg_sqrt5_div3_l391_39190


namespace two_digit_product_GCD_l391_39199

-- We define the condition for two-digit integer numbers
def two_digit_num (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

-- Lean statement capturing the conditions
theorem two_digit_product_GCD :
  ∃ (a b : ℕ), two_digit_num a ∧ two_digit_num b ∧ a * b = 1728 ∧ Nat.gcd a b = 12 := 
by {
  sorry -- The proof steps would go here
}

end two_digit_product_GCD_l391_39199


namespace percentage_basketball_l391_39108

theorem percentage_basketball (total_students : ℕ) (chess_percentage : ℝ) (students_like_chess_basketball : ℕ) 
  (percentage_conversion : ∀ p : ℝ, 0 ≤ p → p / 100 = p) 
  (h_total : total_students = 250) 
  (h_chess : chess_percentage = 10) 
  (h_chess_basketball : students_like_chess_basketball = 125) :
  ∃ (basketball_percentage : ℝ), basketball_percentage = 40 := by
  sorry

end percentage_basketball_l391_39108


namespace identical_functions_l391_39164

def f (x : ℝ) : ℝ := x^2 - 1
def g (x : ℝ) : ℝ := (x^2 - 1)^3^(1/3)

theorem identical_functions : ∀ x : ℝ, f x = g x :=
by
  intro x
  -- Proof to be completed
  sorry

end identical_functions_l391_39164


namespace no_positive_integers_solution_l391_39132

theorem no_positive_integers_solution (m n : ℕ) (hm : m > 0) (hn : n > 0) : 4 * m * (m + 1) ≠ n * (n + 1) := 
by
  sorry

end no_positive_integers_solution_l391_39132


namespace fraction_is_5_div_9_l391_39163

-- Define the conditions t = f * (k - 32), t = 35, and k = 95
theorem fraction_is_5_div_9 {f k t : ℚ} (h1 : t = f * (k - 32)) (h2 : t = 35) (h3 : k = 95) : f = 5 / 9 :=
by
  sorry

end fraction_is_5_div_9_l391_39163


namespace range_of_t_l391_39166

variable {f : ℝ → ℝ}

theorem range_of_t (h₁ : ∀ x y : ℝ, x < y → f x ≥ f y) (h₂ : ∀ t : ℝ, f (t^2) < f t) : 
  ∀ t : ℝ, f (t^2) < f t ↔ (t < 0 ∨ t > 1) := 
by 
  sorry

end range_of_t_l391_39166


namespace placement_proof_l391_39195

def claimed_first_place (p: String) : Prop := 
  p = "Olya" ∨ p = "Oleg" ∨ p = "Pasha"

def odd_places_boys (positions: ℕ → String) : Prop := 
  (positions 1 = "Oleg" ∨ positions 1 = "Pasha") ∧ (positions 3 = "Oleg" ∨ positions 3 = "Pasha")

def olya_wrong (positions : ℕ → String) : Prop := 
  ¬odd_places_boys positions

def always_truthful_or_lying (Olya_st: Prop) (Oleg_st: Prop) (Pasha_st: Prop) : Prop := 
  Olya_st = Oleg_st ∧ Oleg_st = Pasha_st

def competition_placement : Prop :=
  ∃ (positions: ℕ → String),
    claimed_first_place (positions 1) ∧
    claimed_first_place (positions 2) ∧
    claimed_first_place (positions 3) ∧
    (positions 1 = "Oleg") ∧
    (positions 2 = "Pasha") ∧
    (positions 3 = "Olya") ∧
    olya_wrong positions ∧
    always_truthful_or_lying
      ((claimed_first_place "Olya" ∧ odd_places_boys positions))
      ((claimed_first_place "Oleg" ∧ olya_wrong positions))
      (claimed_first_place "Pasha")

theorem placement_proof : competition_placement :=
  sorry

end placement_proof_l391_39195


namespace function_properties_l391_39155

noncomputable def f (x : ℝ) : ℝ := Real.sin (x * Real.cos x)

theorem function_properties :
  (f x = -f (-x)) ∧
  (∀ x, 0 < x ∧ x < Real.pi / 2 → 0 < f x) ∧
  ¬(∃ T, ∀ x, f (x + T) = f x) ∧
  (∀ n : ℤ, f (n * Real.pi) = 0) := 
by
  sorry

end function_properties_l391_39155


namespace system1_solution_system2_solution_l391_39177

theorem system1_solution (x y : ℝ) (h1 : x - 2 * y = 0) (h2 : 3 * x + 2 * y = 8) : 
  x = 2 ∧ y = 1 := sorry

theorem system2_solution (x y : ℝ) (h1 : 3 * x - 5 * y = 9) (h2 : 2 * x + 3 * y = -6) : 
  x = -3 / 19 ∧ y = -36 / 19 := sorry

end system1_solution_system2_solution_l391_39177


namespace range_of_f_l391_39147

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.arctan (2 * x)

theorem range_of_f :
  Set.range f = Set.Icc (Real.pi / 2 + Real.arctan (-2)) (Real.pi / 2 + Real.arctan 2) :=
sorry

end range_of_f_l391_39147


namespace maggie_goldfish_fraction_l391_39100

theorem maggie_goldfish_fraction :
  ∀ (x : ℕ), 3*x / 5 + 20 = x → (x / 100 : ℚ) = 1 / 2 :=
by
  sorry

end maggie_goldfish_fraction_l391_39100


namespace twentieth_number_l391_39170

-- Defining the conditions and goal
theorem twentieth_number :
  ∃ x : ℕ, x % 8 = 5 ∧ x % 3 = 2 ∧ (∃ n : ℕ, x = 5 + 24 * n) ∧ x = 461 := 
sorry

end twentieth_number_l391_39170


namespace stay_nights_l391_39154

theorem stay_nights (cost_per_night : ℕ) (num_people : ℕ) (total_cost : ℕ) (n : ℕ) 
    (h1 : cost_per_night = 40) (h2 : num_people = 3) (h3 : total_cost = 360) (h4 : cost_per_night * num_people * n = total_cost) :
    n = 3 :=
sorry

end stay_nights_l391_39154


namespace fraction_of_total_l391_39152

def total_amount : ℝ := 5000
def r_amount : ℝ := 2000.0000000000002

theorem fraction_of_total
  (h1 : r_amount = 2000.0000000000002)
  (h2 : total_amount = 5000) :
  r_amount / total_amount = 0.40000000000000004 :=
by
  -- The proof is skipped
  sorry

end fraction_of_total_l391_39152


namespace smallest_positive_m_l391_39194

theorem smallest_positive_m (m : ℕ) (h : ∀ (n : ℕ), n % 2 = 1 → (529^n + m * 132^n) % 262417 = 0) : m = 1 :=
sorry

end smallest_positive_m_l391_39194


namespace sum_a2012_a2013_l391_39160

-- Define the geometric sequence and its conditions
def geometric_sequence (a : ℕ → ℚ) (q : ℚ) : Prop := 
  ∀ n : ℕ, a (n + 1) = a n * q

-- Parameters for the problem
variable (a : ℕ → ℚ)
variable (q : ℚ)
variable (h_seq : geometric_sequence a q)
variable (h_q : 1 < q)
variable (h_eq : ∀ x : ℚ, 4 * x^2 - 8 * x + 3 = 0 → x = a 2010 ∨ x = a 2011)

-- Statement to prove
theorem sum_a2012_a2013 : a 2012 + a 2013 = 18 :=
by
  sorry

end sum_a2012_a2013_l391_39160


namespace runners_meet_again_l391_39122

theorem runners_meet_again 
  (v1 v2 v3 v4 v5 : ℕ)
  (h1 : v1 = 32) 
  (h2 : v2 = 40) 
  (h3 : v3 = 48) 
  (h4 : v4 = 56) 
  (h5 : v5 = 64) 
  (h6 : 400 % (v2 - v1) = 0)
  (h7 : 400 % (v3 - v2) = 0)
  (h8 : 400 % (v4 - v3) = 0)
  (h9 : 400 % (v5 - v4) = 0) :
  ∃ t : ℕ, t = 500 :=
by sorry

end runners_meet_again_l391_39122


namespace g_54_l391_39196

def g : ℕ → ℤ := sorry

axiom g_multiplicative (x y : ℕ) (hx : x > 0) (hy : y > 0) : g (x * y) = g x + g y
axiom g_6 : g 6 = 10
axiom g_18 : g 18 = 14

theorem g_54 : g 54 = 18 := by
  sorry

end g_54_l391_39196


namespace train_passing_pole_l391_39184

variables (v L t_platform D_platform t_pole : ℝ)
variables (H1 : L = 500)
variables (H2 : t_platform = 100)
variables (H3 : D_platform = L + 500)
variables (H4 : t_platform = D_platform / v)

theorem train_passing_pole :
  t_pole = L / v := 
sorry

end train_passing_pole_l391_39184


namespace no_integer_solution_for_150_l391_39136

theorem no_integer_solution_for_150 : ∀ (x : ℤ), x - Int.sqrt x ≠ 150 := 
sorry

end no_integer_solution_for_150_l391_39136


namespace max_area_of_sector_l391_39115

variable (r l S : ℝ)

theorem max_area_of_sector (h_circumference : 2 * r + l = 8) (h_area : S = (1 / 2) * l * r) : 
  S ≤ 4 :=
sorry

end max_area_of_sector_l391_39115


namespace usual_time_of_train_l391_39102

theorem usual_time_of_train (S T : ℝ) (h_speed : S ≠ 0) 
(h_speed_ratio : ∀ (T' : ℝ), T' = T + 3/4 → S * T = (4/5) * S * T' → T = 3) : Prop :=
  T = 3

end usual_time_of_train_l391_39102


namespace rod_cut_l391_39159

theorem rod_cut (x : ℕ) (h : 3 * x + 5 * x + 7 * x = 120) : 3 * x = 24 :=
by
  sorry

end rod_cut_l391_39159


namespace factorize_1_factorize_2_l391_39178

-- Proof problem 1: Prove x² - 6x + 9 = (x - 3)²
theorem factorize_1 (x : ℝ) : x^2 - 6 * x + 9 = (x - 3)^2 :=
by sorry

-- Proof problem 2: Prove x²(y - 2) - 4(y - 2) = (y - 2)(x + 2)(x - 2)
theorem factorize_2 (x y : ℝ) : x^2 * (y - 2) - 4 * (y - 2) = (y - 2) * (x + 2) * (x - 2) :=
by sorry

end factorize_1_factorize_2_l391_39178


namespace pie_shop_revenue_l391_39143

noncomputable def revenue_day1 := 5 * 6 * 12 + 6 * 6 * 8 + 7 * 6 * 10
noncomputable def revenue_day2 := 6 * 6 * 15 + 7 * 6 * 10 + 8 * 6 * 14
noncomputable def revenue_day3 := 4 * 6 * 18 + 7 * 6 * 7 + 9 * 6 * 13
noncomputable def total_revenue := revenue_day1 + revenue_day2 + revenue_day3

theorem pie_shop_revenue : total_revenue = 4128 := by
  sorry

end pie_shop_revenue_l391_39143


namespace solution_set_of_inequality_system_l391_39135

theorem solution_set_of_inequality_system (x : ℝ) : (x - 1 < 0 ∧ x + 1 > 0) ↔ (-1 < x ∧ x < 1) := by
  sorry

end solution_set_of_inequality_system_l391_39135


namespace critics_voted_same_actor_actress_l391_39191

theorem critics_voted_same_actor_actress :
  ∃ (critic1 critic2 : ℕ) 
  (actor_vote1 actor_vote2 actress_vote1 actress_vote2 : ℕ),
  1 ≤ critic1 ∧ critic1 ≤ 3366 ∧
  1 ≤ critic2 ∧ critic2 ≤ 3366 ∧
  (critic1 ≠ critic2) ∧
  ∃ (vote_count : Fin 100 → ℕ) 
  (actor actress : Fin 3366 → Fin 100),
  (∀ n : Fin 100, ∃ act : Fin 100, vote_count act = n + 1) ∧
  actor critic1 = actor_vote1 ∧ actress critic1 = actress_vote1 ∧
  actor critic2 = actor_vote2 ∧ actress critic2 = actress_vote2 ∧
  actor_vote1 = actor_vote2 ∧ actress_vote1 = actress_vote2 :=
by
  -- Proof omitted
  sorry

end critics_voted_same_actor_actress_l391_39191


namespace find_m_range_l391_39144

noncomputable def p (m : ℝ) : Prop :=
  m < 1 / 3

noncomputable def q (m : ℝ) : Prop :=
  0 < m ∧ m < 15

theorem find_m_range (m : ℝ) :
  (¬(p m ∧ q m) ∧ (p m ∨ q m)) ↔ (1 / 3 ≤ m ∧ m < 15) :=
by
  sorry

end find_m_range_l391_39144


namespace identity_1_identity_2_identity_3_l391_39185

-- Variables and assumptions
variables (a b c : ℝ)
variables (h_different : a ≠ b ∧ b ≠ c ∧ c ≠ a)
variables (h_pos : a > 0 ∧ b > 0 ∧ c > 0)

-- Part 1
theorem identity_1 : 
  (1 / ((a - b) * (a - c))) + (1 / ((b - c) * (b - a))) + (1 / ((c - a) * (c - b))) = 0 := 
by sorry

-- Part 2
theorem identity_2 :
  (a / ((a - b) * (a - c))) + (b / ((b - c) * (b - a))) + (c / ((c - a) * (c - b))) = 0 :=
by sorry

-- Part 3
theorem identity_3 :
  (a^2 / ((a - b) * (a - c))) + (b^2 / ((b - c) * (b - a))) + (c^2 / ((c - a) * (c - b))) = 1 :=
by sorry

end identity_1_identity_2_identity_3_l391_39185


namespace functionG_has_inverse_l391_39158

noncomputable def functionG : ℝ → ℝ := -- function G described in the problem.
sorry

-- Define the horizontal line test
def horizontal_line_test (f : ℝ → ℝ) : Prop :=
∀ y : ℝ, ∃! x : ℝ, f x = y

theorem functionG_has_inverse : horizontal_line_test functionG :=
sorry

end functionG_has_inverse_l391_39158


namespace moles_of_NaOH_l391_39127

-- Statement of the problem conditions and desired conclusion
theorem moles_of_NaOH (moles_H2SO4 moles_NaHSO4 : ℕ) (h : moles_H2SO4 = 3) (h_eq : moles_H2SO4 = moles_NaHSO4) : moles_NaHSO4 = 3 := by
  sorry

end moles_of_NaOH_l391_39127


namespace coral_third_week_pages_l391_39173

theorem coral_third_week_pages :
  let total_pages := 600
  let week1_read := total_pages / 2
  let remaining_after_week1 := total_pages - week1_read
  let week2_read := remaining_after_week1 * 0.30
  let remaining_after_week2 := remaining_after_week1 - week2_read
  remaining_after_week2 = 210 :=
by
  sorry

end coral_third_week_pages_l391_39173


namespace increase_by_percentage_proof_l391_39110

def initial_number : ℕ := 150
def percentage_increase : ℝ := 0.4
def final_number : ℕ := 210

theorem increase_by_percentage_proof :
  initial_number + (percentage_increase * initial_number) = final_number :=
by
  sorry

end increase_by_percentage_proof_l391_39110


namespace salary_decrease_increase_l391_39129

theorem salary_decrease_increase (S : ℝ) (x : ℝ) (h : (S * (1 - x / 100) * (1 + x / 100) = 0.51 * S)) : x = 70 := 
by sorry

end salary_decrease_increase_l391_39129


namespace solve_for_x_l391_39123

theorem solve_for_x :
  ∀ (x y : ℚ), (3 * x - 4 * y = 8) → (2 * x + 3 * y = 1) → x = 28 / 17 :=
by
  intros x y h1 h2
  sorry

end solve_for_x_l391_39123


namespace flour_vs_sugar_difference_l391_39131

-- Definitions based on the conditions
def flour_needed : ℕ := 10
def flour_added : ℕ := 7
def sugar_needed : ℕ := 2

-- Define the mathematical statement to prove
theorem flour_vs_sugar_difference :
  (flour_needed - flour_added) - sugar_needed = 1 :=
by
  sorry

end flour_vs_sugar_difference_l391_39131


namespace Carolyn_wants_to_embroider_l391_39141

theorem Carolyn_wants_to_embroider (s : ℕ) (f : ℕ) (u : ℕ) (g : ℕ) (n_f : ℕ) (t : ℕ) (number_of_unicorns : ℕ) :
  s = 4 ∧ f = 60 ∧ u = 180 ∧ g = 800 ∧ n_f = 50 ∧ t = 1085 ∧ 
  (t * s - (n_f * f) - g) / u = number_of_unicorns ↔ number_of_unicorns = 3 :=
by 
  sorry

end Carolyn_wants_to_embroider_l391_39141


namespace corporate_event_handshakes_l391_39116

def GroupHandshakes (A B C : Nat) (knows_all_A : Nat) (knows_none : Nat) (C_knows_none : Nat) : Nat :=
  -- Handshakes between Group A and Group B
  let handshakes_AB := knows_none * A
  -- Handshakes within Group B
  let handshakes_B := (knows_none * (knows_none - 1)) / 2
  -- Handshakes between Group B and Group C
  let handshakes_BC := B * C_knows_none
  -- Total handshakes
  handshakes_AB + handshakes_B + handshakes_BC

theorem corporate_event_handshakes : GroupHandshakes 15 20 5 5 15 = 430 :=
by
  sorry

end corporate_event_handshakes_l391_39116


namespace mike_total_cost_self_correct_l391_39187

-- Definition of the given conditions
def cost_per_rose_bush : ℕ := 75
def total_rose_bushes : ℕ := 6
def friend_rose_bushes : ℕ := 2
def cost_per_tiger_tooth_aloes : ℕ := 100
def total_tiger_tooth_aloes : ℕ := 2

-- Calculate the total cost for Mike's plants
def total_cost_mike_self: ℕ := 
  (total_rose_bushes - friend_rose_bushes) * cost_per_rose_bush + total_tiger_tooth_aloes * cost_per_tiger_tooth_aloes

-- The main proposition to be proved
theorem mike_total_cost_self_correct : total_cost_mike_self = 500 := by
  sorry

end mike_total_cost_self_correct_l391_39187


namespace divisibility_by_2k_l391_39180

-- Define the sequence according to the given conditions
def seq (a : ℕ → ℤ) : Prop :=
  a 0 = 0 ∧ a 1 = 1 ∧ ∀ n, 2 ≤ n → a n = 2 * a (n - 1) + a (n - 2)

-- The theorem to be proved
theorem divisibility_by_2k (a : ℕ → ℤ) (k : ℕ) (n : ℕ)
  (h : seq a) :
  2^k ∣ a n ↔ 2^k ∣ n :=
sorry

end divisibility_by_2k_l391_39180


namespace solve_for_n_l391_39104

theorem solve_for_n (n : ℚ) (h : n + (n + 1) + (n + 2) + (n + 3) = 20) : 
    n = 3.5 :=
  sorry

end solve_for_n_l391_39104


namespace second_solution_sugar_percentage_l391_39106

theorem second_solution_sugar_percentage
  (initial_solution_pct : ℝ)
  (second_solution_pct : ℝ)
  (initial_solution_amount : ℝ)
  (final_solution_pct : ℝ)
  (replaced_fraction : ℝ)
  (final_amount : ℝ) :
  initial_solution_pct = 0.1 →
  final_solution_pct = 0.17 →
  replaced_fraction = 1/4 →
  initial_solution_amount = 100 →
  final_amount = 100 →
  second_solution_pct = 0.38 :=
by
  intros h1 h2 h3 h4 h5
  -- Proof goes here
  sorry

end second_solution_sugar_percentage_l391_39106


namespace harmonic_mean_closest_to_six_l391_39138

def harmonic_mean (a b : ℕ) : ℚ := (2 * a * b) / (a + b)

theorem harmonic_mean_closest_to_six : 
     |harmonic_mean 3 2023 - 6| < 1 :=
sorry

end harmonic_mean_closest_to_six_l391_39138


namespace total_work_completed_in_18_days_l391_39197

theorem total_work_completed_in_18_days :
  let amit_work_rate := 1/10
  let ananthu_work_rate := 1/20
  let amit_days := 2
  let amit_work_done := amit_days * amit_work_rate
  let remaining_work := 1 - amit_work_done
  let ananthu_days := remaining_work / ananthu_work_rate
  amit_days + ananthu_days = 18 := 
by
  sorry

end total_work_completed_in_18_days_l391_39197


namespace sum_polynomials_l391_39125

def p (x : ℝ) : ℝ := 4 * x^2 - 2 * x + 1
def q (x : ℝ) : ℝ := -3 * x^2 + x - 5
def r (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3

theorem sum_polynomials (x : ℝ) : p x + q x + r x = 3 * x^2 - 5 * x - 1 :=
by
  sorry

end sum_polynomials_l391_39125


namespace zephyr_island_population_capacity_reach_l391_39198

-- Definitions for conditions
def acres := 30000
def acres_per_person := 2
def initial_year := 2023
def initial_population := 500
def population_growth_rate := 4
def growth_period := 20

-- Maximum population supported by the island
def max_population := acres / acres_per_person

-- Function to calculate population after a given number of years
def population (years : ℕ) : ℕ := initial_population * (population_growth_rate ^ (years / growth_period))

-- The Lean statement to prove that the population will reach or exceed max_capacity in 60 years
theorem zephyr_island_population_capacity_reach : ∃ t : ℕ, t ≤ 60 ∧ population t ≥ max_population :=
by
  sorry

end zephyr_island_population_capacity_reach_l391_39198


namespace find_k_intersecting_lines_l391_39120

theorem find_k_intersecting_lines : 
  ∃ (k : ℚ), (∃ (x y : ℚ), y = 6 * x + 4 ∧ y = -3 * x - 30 ∧ y = 4 * x + k) ∧ k = -32 / 9 :=
by
  sorry

end find_k_intersecting_lines_l391_39120


namespace problem1_problem2_l391_39148

theorem problem1 : (1 * (-5) - (-6) + (-7)) = -6 :=
by
  sorry

theorem problem2 : (-1)^2021 + (-18) * abs (-2 / 9) - 4 / (-2) = -3 :=
by
  sorry

end problem1_problem2_l391_39148


namespace value_of_expression_l391_39186

variables {a b c : ℝ}

theorem value_of_expression (h1 : a * b * c = 10) (h2 : a + b + c = 15) (h3 : a * b + b * c + c * a = 25) :
  (2 + a) * (2 + b) * (2 + c) = 128 := 
sorry

end value_of_expression_l391_39186


namespace puzzle_pieces_l391_39183

theorem puzzle_pieces (x : ℝ) (h : x + 2 * 1.5 * x = 4000) : x = 1000 :=
  sorry

end puzzle_pieces_l391_39183


namespace coeff_of_x_square_l391_39169

-- Define the binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Statement of the problem
theorem coeff_of_x_square :
  (binom 8 3 = 56) ∧ (8 - 2 * 3 = 2) :=
sorry

end coeff_of_x_square_l391_39169


namespace average_age_of_boys_l391_39145

def boys_age_proportions := (3, 5, 7)
def eldest_boy_age := 21

theorem average_age_of_boys : 
  ∃ (x : ℕ), 7 * x = eldest_boy_age ∧ (3 * x + 5 * x + 7 * x) / 3 = 15 :=
by
  sorry

end average_age_of_boys_l391_39145


namespace difference_in_balances_l391_39181

/-- Define the parameters for Angela's and Bob's accounts --/
def P_A : ℕ := 5000  -- Angela's principal
def r_A : ℚ := 0.05  -- Angela's annual interest rate
def n_A : ℕ := 2  -- Compounding frequency for Angela
def t : ℕ := 15  -- Time in years

def P_B : ℕ := 7000  -- Bob's principal
def r_B : ℚ := 0.04  -- Bob's annual interest rate

/-- Computing the final amounts for Angela and Bob after 15 years --/
noncomputable def A_A : ℚ := P_A * ((1 + (r_A / n_A)) ^ (n_A * t))  -- Angela's final amount
noncomputable def A_B : ℚ := P_B * (1 + r_B * t)  -- Bob's final amount

/-- Proof statement: The difference in account balances to the nearest dollar --/
theorem difference_in_balances : abs (A_A - A_B) = 726 := by
  sorry

end difference_in_balances_l391_39181


namespace find_a_subtract_two_l391_39192

theorem find_a_subtract_two (a b : ℤ) 
    (h1 : 2 + a = 5 - b) 
    (h2 : 5 + b = 8 + a) : 
    2 - a = 2 := 
by
  sorry

end find_a_subtract_two_l391_39192


namespace find_xyz_l391_39128

open Complex

theorem find_xyz (a b c x y z : ℂ)
(h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : x ≠ 0) (h5 : y ≠ 0) (h6 : z ≠ 0)
(h7 : a = (b + c) / (x - 3)) (h8 : b = (a + c) / (y - 3)) (h9 : c = (a + b) / (z - 3))
(h10 : x * y + x * z + y * z = 10) (h11 : x + y + z = 6) : 
(x * y * z = 15) :=
by
  sorry

end find_xyz_l391_39128


namespace complex_power_sum_l391_39111

open Complex

theorem complex_power_sum (z : ℂ) (h : z^2 - z + 1 = 0) : 
  z^99 + z^100 + z^101 + z^102 + z^103 = 2 + Complex.I * Real.sqrt 3 ∨ z^99 + z^100 + z^101 + z^102 + z^103 = 2 - Complex.I * Real.sqrt 3 :=
sorry

end complex_power_sum_l391_39111


namespace susan_vacation_pay_missed_l391_39121

noncomputable def susan_weekly_pay (hours_worked : ℕ) : ℕ :=
  let regular_hours := min 40 hours_worked
  let overtime_hours := max (hours_worked - 40) 0
  15 * regular_hours + 20 * overtime_hours

noncomputable def susan_sunday_pay (num_sundays : ℕ) (hours_per_sunday : ℕ) : ℕ :=
  25 * num_sundays * hours_per_sunday

noncomputable def pay_without_sundays : ℕ :=
  susan_weekly_pay 48
    
noncomputable def total_three_week_pay : ℕ :=
  let weeks_normal_pay := 3 * pay_without_sundays
  let sunday_hours_1 := 1 * 8
  let sunday_hours_2 := 2 * 8
  let sunday_hours_3 := 0 * 8
  let sundays_total_pay := susan_sunday_pay 1 8 + susan_sunday_pay 2 8 + susan_sunday_pay 0 8
  weeks_normal_pay + sundays_total_pay
  
noncomputable def paid_vacation_pay : ℕ :=
  let paid_days := 6
  let paid_weeks_pay := susan_weekly_pay 40 + susan_weekly_pay (paid_days % 5 * 8)
  paid_weeks_pay

theorem susan_vacation_pay_missed :
  let missed_pay := total_three_week_pay - paid_vacation_pay
  missed_pay = 2160 := sorry

end susan_vacation_pay_missed_l391_39121


namespace correct_statement_c_l391_39117

-- Definitions
variables {Point : Type*} {Line Plane : Type*}
variables (l m : Line) (α β : Plane)

-- Conditions
def parallel_planes (α β : Plane) : Prop := sorry  -- α ∥ β
def perpendicular_line_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊥ α
def line_in_plane (l : Line) (α : Plane) : Prop := sorry  -- l ⊂ α
def line_perpendicular (l m : Line) : Prop := sorry  -- l ⊥ m

-- Theorem to be proven
theorem correct_statement_c 
  (α β : Plane) (l : Line)
  (h_parallel : parallel_planes α β)
  (h_perpendicular : perpendicular_line_plane l α) :
  ∀ (m : Line), line_in_plane m β → line_perpendicular m l := 
sorry

end correct_statement_c_l391_39117


namespace problem1_problem2_l391_39179

-- Proof of Problem 1
theorem problem1 (x y : ℤ) (h1 : x = -2) (h2 : y = -3) : (6 * x - 5 * y + 3 * y - 2 * x) = -2 :=
by
  sorry

-- Proof of Problem 2
theorem problem2 (a : ℚ) (h : a = -1 / 2) : (1 / 4 * (-4 * a^2 + 2 * a - 8) - (1 / 2 * a - 2)) = -1 / 4 :=
by
  sorry

end problem1_problem2_l391_39179


namespace problem_1_problem_2_l391_39130

-- Define proposition p
def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, (1 ≤ x ∧ x ≤ 2) → x^2 - a ≥ 0

-- Define proposition q
def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the range of values for a in proposition p
def range_p (a : ℝ) : Prop :=
  a ≤ 1

-- Define set A and set B
def set_A (a : ℝ) : Prop := a ≤ 1
def set_B (a : ℝ) : Prop := a ≥ 1 ∨ a ≤ -2

theorem problem_1 (a : ℝ) (h : proposition_p a) : range_p a := 
sorry

theorem problem_2 (a : ℝ) : 
  (∃ h1 : proposition_p a, set_A a) ∧ (∃ h2 : proposition_q a, set_B a)
  ↔ ¬ ((∃ h1 : proposition_p a, set_B a) ∧ (∃ h2 : proposition_q a, set_A a)) :=
sorry

end problem_1_problem_2_l391_39130


namespace decimal_digits_of_fraction_l391_39176

noncomputable def fraction : ℚ := 987654321 / (2 ^ 30 * 5 ^ 2)

theorem decimal_digits_of_fraction :
  ∃ n ≥ 30, fraction = (987654321 / 10^2) / 2^28 := sorry

end decimal_digits_of_fraction_l391_39176


namespace average_of_remaining_two_numbers_l391_39165

theorem average_of_remaining_two_numbers (a b c d e f : ℝ) 
  (h1 : (a + b + c + d + e + f) / 6 = 8) 
  (h2 : (a + b + c + d) / 4 = 5) : 
  (e + f) / 2 = 14 := 
by  
  sorry

end average_of_remaining_two_numbers_l391_39165


namespace sandy_correct_sums_l391_39134

theorem sandy_correct_sums
  (c i : ℕ)
  (h1 : c + i = 30)
  (h2 : 3 * c - 2 * i = 45) :
  c = 21 :=
by
  sorry

end sandy_correct_sums_l391_39134
