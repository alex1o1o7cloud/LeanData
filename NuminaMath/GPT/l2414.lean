import Mathlib

namespace P_Ravi_is_02_l2414_241408

def P_Ram : ℚ := 6 / 7
def P_Ram_and_Ravi : ℚ := 0.17142857142857143

theorem P_Ravi_is_02 (P_Ravi : ℚ) : P_Ram_and_Ravi = P_Ram * P_Ravi → P_Ravi = 0.2 :=
by
  intro h
  sorry

end P_Ravi_is_02_l2414_241408


namespace smallest_r_for_B_in_C_l2414_241463

def A : Set ℝ := {t | 0 < t ∧ t < 2 * Real.pi}

def B : Set (ℝ × ℝ) := 
  {p | ∃ t ∈ A, p.1 = Real.sin t ∧ p.2 = 2 * Real.sin t * Real.cos t}

def C (r : ℝ) : Set (ℝ × ℝ) := 
  {p | p.1^2 + p.2^2 ≤ r^2 ∧ r > 0}

theorem smallest_r_for_B_in_C : ∃ r, (B ⊆ C r ∧ ∀ r', r' < r → ¬ (B ⊆ C r')) :=
  sorry

end smallest_r_for_B_in_C_l2414_241463


namespace miles_mike_l2414_241461

def cost_mike (M : ℕ) : ℝ := 2.50 + 0.25 * M
def cost_annie (A : ℕ) : ℝ := 2.50 + 5.00 + 0.25 * A

theorem miles_mike {M A : ℕ} (annie_ride_miles : A = 16) (same_cost : cost_mike M = cost_annie A) : M = 36 :=
by
  rw [cost_annie, annie_ride_miles] at same_cost
  simp [cost_mike] at same_cost
  sorry

end miles_mike_l2414_241461


namespace division_result_l2414_241428

-- Define n in terms of the given condition
def n : ℕ := 9^2023

theorem division_result : n / 3 = 3^4045 :=
by
  sorry

end division_result_l2414_241428


namespace find_other_root_l2414_241407

theorem find_other_root (m : ℝ) (h : 2^2 - 2 + m = 0) : 
  ∃ α : ℝ, α = -1 ∧ (α^2 - α + m = 0) :=
by
  -- Assuming x = 2 is a root, prove that the other root is -1.
  sorry

end find_other_root_l2414_241407


namespace sin_1200_eq_sqrt3_div_2_l2414_241456

theorem sin_1200_eq_sqrt3_div_2 : Real.sin (1200 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end sin_1200_eq_sqrt3_div_2_l2414_241456


namespace side_length_of_square_ground_l2414_241438

theorem side_length_of_square_ground
    (radius : ℝ)
    (Q_area : ℝ)
    (pi : ℝ)
    (quarter_circle_area : Q_area = (pi * (radius^2) / 4))
    (pi_approx : pi = 3.141592653589793)
    (Q_area_val : Q_area = 15393.804002589986)
    (radius_val : radius = 140) :
    ∃ (s : ℝ), s^2 = radius^2 :=
by
  sorry -- Proof not required per the instructions

end side_length_of_square_ground_l2414_241438


namespace find_a_l2414_241494

noncomputable def f (a : ℝ) (x : ℝ) := (1/2) * a * x^2 + Real.log x

theorem find_a (h_max : ∃ (x : Set.Icc (0 : ℝ) 1), f (-Real.exp 1) x = -1) : 
  ∀ a : ℝ, (∀ x : ℝ, 0 < x → x ≤ 1 → f a x ≤ -1) → a = -Real.exp 1 :=
sorry

end find_a_l2414_241494


namespace altered_prism_edges_l2414_241427

theorem altered_prism_edges :
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  total_edges = 42 :=
by
  let original_edges := 12
  let vertices := 8
  let edges_per_vertex := 3
  let faces := 6
  let edges_per_face := 1
  let total_edges := original_edges + edges_per_vertex * vertices + edges_per_face * faces
  show total_edges = 42
  sorry

end altered_prism_edges_l2414_241427


namespace smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l2414_241417

theorem smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5 :
  ∃ n : ℕ, n > 0 ∧ (7^n % 5 = n^7 % 5) ∧
  ∀ m : ℕ, m > 0 ∧ (7^m % 5 = m^7 % 5) → n ≤ m :=
by
  sorry

end smallest_n_satisfying_7_n_mod_5_eq_n_7_mod_5_l2414_241417


namespace cubic_root_expression_l2414_241498

theorem cubic_root_expression (p q r : ℝ)
  (h₁ : p + q + r = 8)
  (h₂ : p * q + p * r + q * r = 11)
  (h₃ : p * q * r = 3) :
  p / (q * r + 1) + q / (p * r + 1) + r / (p * q + 1) = 32 / 15 :=
by 
  sorry

end cubic_root_expression_l2414_241498


namespace find_a_plus_b_l2414_241467

noncomputable def f (a b x : ℝ) : ℝ := a * x ^ 2 + b * x + 3 * a + b

theorem find_a_plus_b (a b : ℝ) (h1 : ∀ x : ℝ, f a b x = f a b (-x)) (h2 : 2 * a = 3 - a) : a + b = 1 :=
by
  unfold f at h1
  sorry

end find_a_plus_b_l2414_241467


namespace opposite_of_one_over_2023_l2414_241482

def one_over_2023 : ℚ := 1 / 2023

theorem opposite_of_one_over_2023 : -one_over_2023 = -1 / 2023 :=
by
  sorry

end opposite_of_one_over_2023_l2414_241482


namespace countFibSequences_l2414_241412

-- Define what it means for a sequence to be Fibonacci-type
def isFibType (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

-- Define a Fibonacci-type sequence condition with given constraints
def fibSeqCondition (a : ℤ → ℤ) (N : ℤ) : Prop :=
  isFibType a ∧ ∃ n : ℤ, 0 < a n ∧ a n ≤ N ∧ 0 < a (n + 1) ∧ a (n + 1) ≤ N

-- Main theorem
theorem countFibSequences (N : ℤ) :
  ∃ count : ℤ,
    (N % 2 = 0 → count = (N / 2) * (N / 2 + 1)) ∧
    (N % 2 = 1 → count = ((N + 1) / 2) ^ 2) ∧
    (∀ a : ℤ → ℤ, fibSeqCondition a N → (∃ n : ℤ, a n = count)) :=
by
  sorry

end countFibSequences_l2414_241412


namespace value_of_m_l2414_241401

def p (m : ℝ) : Prop :=
  4 < m ∧ m < 10

def q (m : ℝ) : Prop :=
  8 < m ∧ m < 12

theorem value_of_m (m : ℝ) :
  (p m ∨ q m) ∧ ¬ (p m ∧ q m) ↔ (4 < m ∧ m ≤ 8) ∨ (10 ≤ m ∧ m < 12) :=
by
  sorry

end value_of_m_l2414_241401


namespace series_sum_equality_l2414_241444

noncomputable def sum_series : ℝ :=
  ∑' k : ℕ, 12^k / ((4^k - 3^k) * (4^(k+1) - 3^(k+1)))

theorem series_sum_equality : sum_series = 1 := 
by sorry

end series_sum_equality_l2414_241444


namespace length_of_second_platform_l2414_241439

-- Given conditions
def length_of_train : ℕ := 310
def length_of_first_platform : ℕ := 110
def time_to_cross_first_platform : ℕ := 15
def time_to_cross_second_platform : ℕ := 20

-- Calculated based on conditions
def total_distance_first_platform : ℕ :=
  length_of_train + length_of_first_platform

def speed_of_train : ℕ :=
  total_distance_first_platform / time_to_cross_first_platform

def total_distance_second_platform : ℕ :=
  speed_of_train * time_to_cross_second_platform

-- Statement to prove
theorem length_of_second_platform :
  total_distance_second_platform = length_of_train + 250 := sorry

end length_of_second_platform_l2414_241439


namespace z_real_iff_z_complex_iff_z_pure_imaginary_iff_l2414_241487

-- Definitions for the problem conditions
def z_real (m : ℝ) : Prop := (m^2 - 2 * m - 15 = 0)
def z_pure_imaginary (m : ℝ) : Prop := (m^2 - 9 * m - 36 = 0) ∧ (m^2 - 2 * m - 15 ≠ 0)

-- Question 1: Prove that z is a real number if and only if m = -3 or m = 5
theorem z_real_iff (m : ℝ) : z_real m ↔ m = -3 ∨ m = 5 := sorry

-- Question 2: Prove that z is a complex number with non-zero imaginary part if and only if m ≠ -3 and m ≠ 5
theorem z_complex_iff (m : ℝ) : ¬z_real m ↔ m ≠ -3 ∧ m ≠ 5 := sorry

-- Question 3: Prove that z is a pure imaginary number if and only if m = 12
theorem z_pure_imaginary_iff (m : ℝ) : z_pure_imaginary m ↔ m = 12 := sorry

end z_real_iff_z_complex_iff_z_pure_imaginary_iff_l2414_241487


namespace distance_from_origin_to_line_AB_is_sqrt6_div_3_l2414_241443

open Real

structure Point where
  x : ℝ
  y : ℝ

def ellipse (p : Point) : Prop :=
  p.x^2 / 2 + p.y^2 = 1

def left_focus : Point := ⟨-1, 0⟩

def line_through_focus (t : ℝ) (p : Point) : Prop :=
  p.x = t * p.y - 1

def origin : Point := ⟨0, 0⟩

def perpendicular (A B : Point) : Prop :=
  A.x * B.x + A.y * B.y = 0

noncomputable def distance (O : Point) (A B : Point) : ℝ :=
  let a := A.y - B.y
  let b := B.x - A.x
  let c := A.x * B.y - A.y * B.x
  abs (a * O.x + b * O.y + c) / sqrt (a^2 + b^2)

theorem distance_from_origin_to_line_AB_is_sqrt6_div_3 
  (A B : Point)
  (hA_on_ellipse : ellipse A)
  (hB_on_ellipse : ellipse B)
  (h_line_through_focus : ∃ t : ℝ, line_through_focus t A ∧ line_through_focus t B)
  (h_perpendicular : perpendicular A B) :
  distance origin A B = sqrt 6 / 3 := sorry

end distance_from_origin_to_line_AB_is_sqrt6_div_3_l2414_241443


namespace find_f2_l2414_241479

-- Define the function f and the condition it satisfies
variable (f : ℝ → ℝ)

-- Define the condition given in the problem
def condition : Prop := ∀ x, x ≠ 1 / 3 → f x + f ((x + 1) / (1 - 3 * x)) = x

-- State the theorem to prove the value of f(2)
theorem find_f2 (h : condition f) : f 2 = 48 / 35 := 
by
  sorry

end find_f2_l2414_241479


namespace find_DG_l2414_241406

theorem find_DG 
  (a b : ℕ) -- sides DE and EC
  (S : ℕ := 19 * (a + b)) -- area of each rectangle
  (k l : ℕ) -- sides DG and CH
  (h1 : S = a * k) 
  (h2 : S = b * l) 
  (h_bc : 19 * (a + b) = S)
  (h_div_a : S % a = 0)
  (h_div_b : S % b = 0)
  : DG = 380 :=
sorry

end find_DG_l2414_241406


namespace soccer_team_games_l2414_241455

theorem soccer_team_games :
  ∃ G : ℕ, G % 2 = 0 ∧ 
           45 / 100 * 36 = 16 ∧ 
           ∀ R, R = G - 36 → (16 + 75 / 100 * R) = 62 / 100 * G ∧
           G = 84 :=
sorry

end soccer_team_games_l2414_241455


namespace find_point_P_l2414_241477

structure Point3D :=
  (x : ℝ)
  (y : ℝ)
  (z : ℝ)

def isEquidistant (p1 p2 : Point3D) (q : Point3D) : Prop :=
  (q.x - p1.x)^2 + (q.y - p1.y)^2 + (q.z - p1.z)^2 = (q.x - p2.x)^2 + (q.y - p2.y)^2 + (q.z - p2.z)^2

theorem find_point_P (P : Point3D) :
  (∀ (Q : Point3D), isEquidistant ⟨2, 3, -4⟩ P Q → (8 * Q.x - 6 * Q.y + 18 * Q.z = 70)) →
  P = ⟨6, 0, 5⟩ :=
by 
  sorry

end find_point_P_l2414_241477


namespace exists_universal_accessible_city_l2414_241454

-- Define the basic structure for cities and flights
structure Country :=
  (City : Type)
  (accessible : City → City → Prop)

namespace Country

-- Define the properties of accessibility in the country
variables {C : Country}

-- Axiom: Each city is accessible from itself
axiom self_accessible (A : C.City) : C.accessible A A

-- Axiom: For any two cities, there exists a city from which both are accessible
axiom exists_intermediate (P Q : C.City) : ∃ R : C.City, C.accessible R P ∧ C.accessible R Q

-- Definition of the main theorem
theorem exists_universal_accessible_city :
  ∃ U : C.City, ∀ A : C.City, C.accessible U A :=
sorry

end Country

end exists_universal_accessible_city_l2414_241454


namespace negation_of_universal_proposition_l2414_241486

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 > 1) ↔ (∃ x : ℝ, x^2 ≤ 1) :=
by
  sorry

end negation_of_universal_proposition_l2414_241486


namespace magnitude_product_l2414_241472

-- Definitions based on conditions
def z1 : Complex := ⟨7, -4⟩
def z2 : Complex := ⟨3, 10⟩

-- Statement of the theorem to be proved
theorem magnitude_product :
  Complex.abs (z1 * z2) = Real.sqrt 7085 := by
  sorry

end magnitude_product_l2414_241472


namespace multiple_statements_l2414_241449

theorem multiple_statements (c d : ℤ)
  (hc4 : ∃ k : ℤ, c = 4 * k)
  (hd8 : ∃ k : ℤ, d = 8 * k) :
  (∃ k : ℤ, d = 4 * k) ∧
  (∃ k : ℤ, c + d = 4 * k) ∧
  (∃ k : ℤ, c + d = 2 * k) :=
by
  sorry

end multiple_statements_l2414_241449


namespace dadAgeWhenXiaoHongIs7_l2414_241418

variable {a : ℕ}

-- Condition: Dad's age is given as 'a'
-- Condition: Dad's age is 4 times plus 3 years more than Xiao Hong's age
def xiaoHongAge (a : ℕ) : ℕ := (a - 3) / 4

theorem dadAgeWhenXiaoHongIs7 : xiaoHongAge a = 7 → a = 31 := by
  intro h
  have h1 : a - 3 = 28 := by sorry   -- Algebraic manipulation needed
  have h2 : a = 31 := by sorry       -- Algebraic manipulation needed
  exact h2

end dadAgeWhenXiaoHongIs7_l2414_241418


namespace workers_in_workshop_l2414_241409

theorem workers_in_workshop :
  (∀ (W : ℕ), 8000 * W = 12000 * 7 + 6000 * (W - 7) → W = 21) :=
by
  intro W h
  sorry

end workers_in_workshop_l2414_241409


namespace find_n_l2414_241422

theorem find_n (n : ℕ) (x y a b : ℕ) (hx : x = 1) (hy : y = 1) (ha : a = 1) (hb : b = 1)
  (h : (x + 3 * y) ^ n = (7 * a + b) ^ 10) : n = 5 :=
by
  sorry

end find_n_l2414_241422


namespace fractional_equation_root_l2414_241475

theorem fractional_equation_root (k : ℚ) (x : ℚ) (h : (2 * k) / (x - 1) - 3 / (1 - x) = 1) : k = -3 / 2 :=
sorry

end fractional_equation_root_l2414_241475


namespace counting_error_l2414_241489

theorem counting_error
  (b g : ℕ)
  (initial_balloons := 5 * b + 4 * g)
  (popped_balloons := g + 2 * b)
  (remaining_balloons := initial_balloons - popped_balloons)
  (Dima_count := 100) :
  remaining_balloons ≠ Dima_count := by
  sorry

end counting_error_l2414_241489


namespace value_of_f_sin_7pi_over_6_l2414_241457

def f (x : ℝ) : ℝ := 4 * x^2 + 2 * x

theorem value_of_f_sin_7pi_over_6 :
  f (Real.sin (7 * Real.pi / 6)) = 0 :=
by
  sorry

end value_of_f_sin_7pi_over_6_l2414_241457


namespace Diamond_result_l2414_241430

-- Define the binary operation Diamond
def Diamond (a b : ℕ) : ℕ := a * b^2 - b + 1

theorem Diamond_result : Diamond (Diamond 3 4) 2 = 179 := 
by 
  sorry

end Diamond_result_l2414_241430


namespace samantha_more_posters_l2414_241421

theorem samantha_more_posters :
  ∃ S : ℕ, S > 18 ∧ 18 + S = 51 ∧ S - 18 = 15 :=
by
  sorry

end samantha_more_posters_l2414_241421


namespace tan_alpha_implication_l2414_241492

theorem tan_alpha_implication (α : ℝ) (h : Real.tan α = 2) :
    (2 * Real.sin α - Real.cos α) / (2 * Real.sin α + Real.cos α) = 3 / 5 := 
by 
  sorry

end tan_alpha_implication_l2414_241492


namespace find_value_of_expression_l2414_241429

noncomputable def roots_g : Set ℂ := { x | x^2 - 3*x - 2 = 0 }

theorem find_value_of_expression:
  ∀ γ δ : ℂ, γ ∈ roots_g → δ ∈ roots_g →
  (γ + δ = 3) → (7 * γ^4 + 10 * δ^3 = 1363) :=
by
  intros γ δ hγ hδ hsum
  -- Proof skipped
  sorry

end find_value_of_expression_l2414_241429


namespace simplify_expression_l2414_241473

noncomputable def x : ℝ := Real.sqrt 3 + 1

theorem simplify_expression (x : ℝ) (h : x = Real.sqrt 3 + 1) :
  ((1 - (x / (x + 1))) / ((x^2 - 1) / (x^2 + 2*x + 1))) = Real.sqrt 3 / 3 :=
by
  sorry

end simplify_expression_l2414_241473


namespace shopper_savings_percentage_l2414_241481

theorem shopper_savings_percentage
  (amount_saved : ℝ) (final_price : ℝ)
  (h_saved : amount_saved = 3)
  (h_final : final_price = 27) :
  (amount_saved / (final_price + amount_saved)) * 100 = 10 := 
by
  sorry

end shopper_savings_percentage_l2414_241481


namespace regina_final_earnings_l2414_241434

-- Define the number of animals Regina has
def cows := 20
def pigs := 4 * cows
def goats := pigs / 2
def chickens := 2 * cows
def rabbits := 30

-- Define sale prices for each animal
def cow_price := 800
def pig_price := 400
def goat_price := 600
def chicken_price := 50
def rabbit_price := 25

-- Define annual earnings from animal products
def cow_milk_income := 500
def rabbit_meat_income := 10

-- Define annual farm maintenance and animal feed costs
def maintenance_cost := 10000

-- Define a calculation for the final earnings
def final_earnings : ℕ :=
  let cow_income := cows * cow_price
  let pig_income := pigs * pig_price
  let goat_income := goats * goat_price
  let chicken_income := chickens * chicken_price
  let rabbit_income := rabbits * rabbit_price
  let total_animal_sale_income := cow_income + pig_income + goat_income + chicken_income + rabbit_income

  let cow_milk_earning := cows * cow_milk_income
  let rabbit_meat_earning := rabbits * rabbit_meat_income
  let total_annual_income := cow_milk_earning + rabbit_meat_earning

  let total_income := total_animal_sale_income + total_annual_income
  let final_income := total_income - maintenance_cost

  final_income

-- Prove that the final earnings is as calculated
theorem regina_final_earnings : final_earnings = 75050 := by
  sorry

end regina_final_earnings_l2414_241434


namespace a_plus_d_eq_zero_l2414_241432

noncomputable def f (a b c d x : ℝ) : ℝ := (2 * a * x + b) / (c * x + 2 * d)

theorem a_plus_d_eq_zero (a b c d : ℝ) (h : a * b * c * d ≠ 0) (hff : ∀ x, f a b c d (f a b c d x) = 3 * x - 4) : a + d = 0 :=
by
  sorry

end a_plus_d_eq_zero_l2414_241432


namespace hyperbola_through_C_l2414_241425

noncomputable def equation_of_hyperbola_passing_through_C : Prop :=
  let A := (-1/2, 1/4)
  let B := (2, 4)
  let C := (-1/2, 4)
  ∃ (k : ℝ), k = -2 ∧ (∀ x : ℝ, x ≠ 0 → x * (4) = k)

theorem hyperbola_through_C :
  equation_of_hyperbola_passing_through_C :=
by
  sorry

end hyperbola_through_C_l2414_241425


namespace cookies_remaining_percentage_l2414_241441

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

end cookies_remaining_percentage_l2414_241441


namespace percent_flowers_are_carnations_l2414_241496

-- Define the conditions
def one_third_pink_are_roses (total_flower pink_flower pink_roses : ℕ) : Prop :=
  pink_roses = (1/3) * pink_flower

def three_fourths_red_are_carnations (total_flower red_flower red_carnations : ℕ) : Prop :=
  red_carnations = (3/4) * red_flower

def six_tenths_are_pink (total_flower pink_flower : ℕ) : Prop :=
  pink_flower = (6/10) * total_flower

-- Define the proof problem statement
theorem percent_flowers_are_carnations (total_flower pink_flower pink_roses red_flower red_carnations : ℕ) :
  one_third_pink_are_roses total_flower pink_flower pink_roses →
  three_fourths_red_are_carnations total_flower red_flower red_carnations →
  six_tenths_are_pink total_flower pink_flower →
  (red_flower = total_flower - pink_flower) →
  (pink_flower - pink_roses + red_carnations = (4/10) * total_flower) →
  ((pink_flower - pink_roses) + red_carnations) * 100 / total_flower = 40 := 
sorry

end percent_flowers_are_carnations_l2414_241496


namespace intersection_of_A_and_B_l2414_241478

def A : Set ℝ := {x | x - 1 > 1}
def B : Set ℝ := {x | x < 3}

theorem intersection_of_A_and_B : (A ∩ B) = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_of_A_and_B_l2414_241478


namespace tobias_swimming_distance_l2414_241414

def swimming_time_per_100_meters : ℕ := 5
def pause_time : ℕ := 5
def swimming_period : ℕ := 25
def total_visit_hours : ℕ := 3

theorem tobias_swimming_distance :
  let total_visit_minutes := total_visit_hours * 60
  let sequence_time := swimming_period + pause_time
  let number_of_sequences := total_visit_minutes / sequence_time
  let total_pause_time := number_of_sequences * pause_time
  let total_swimming_time := total_visit_minutes - total_pause_time
  let number_of_100m_lengths := total_swimming_time / swimming_time_per_100_meters
  let total_distance := number_of_100m_lengths * 100
  total_distance = 3000 :=
by
  sorry

end tobias_swimming_distance_l2414_241414


namespace nissa_grooming_time_correct_l2414_241497

def clipping_time_per_claw : ℕ := 10
def cleaning_time_per_ear : ℕ := 90
def shampooing_time_minutes : ℕ := 5

def claws_per_foot : ℕ := 4
def feet_count : ℕ := 4
def ear_count : ℕ := 2

noncomputable def total_grooming_time_in_seconds : ℕ := 
  (clipping_time_per_claw * claws_per_foot * feet_count) + 
  (cleaning_time_per_ear * ear_count) + 
  (shampooing_time_minutes * 60) -- converting minutes to seconds

theorem nissa_grooming_time_correct :
  total_grooming_time_in_seconds = 640 := by
  sorry

end nissa_grooming_time_correct_l2414_241497


namespace dennis_teaching_years_l2414_241495

noncomputable def years_taught (V A D E N : ℕ) := V + A + D + E + N
noncomputable def sum_of_ages := 375
noncomputable def teaching_years : Prop :=
  ∃ (A V D E N : ℕ),
    V + A + D + E + N = 225 ∧
    V = A + 9 ∧
    V = D - 15 ∧
    E = A - 3 ∧
    E = 2 * N ∧
    D = 101

theorem dennis_teaching_years : teaching_years :=
by
  sorry

end dennis_teaching_years_l2414_241495


namespace circle_tangent_line_l2414_241445

noncomputable def line_eq (x : ℝ) : ℝ := 2 * x + 1
noncomputable def circle_eq (x y b : ℝ) : ℝ := x^2 + (y - b)^2

theorem circle_tangent_line 
  (b : ℝ) 
  (tangency : ∃ b, (1 - b) / (0 - 1) = -(1 / 2)) 
  (center_point : 1^2 + (3 - b)^2 = 5 / 4) : 
  circle_eq 1 3 b = circle_eq 0 b (7/2) :=
sorry

end circle_tangent_line_l2414_241445


namespace number_of_cloth_bags_l2414_241452

-- Definitions based on the conditions
def dozen := 12

def total_peaches : ℕ := 5 * dozen
def peaches_in_knapsack : ℕ := 12
def peaches_per_bag : ℕ := 2 * peaches_in_knapsack

-- The proof statement
theorem number_of_cloth_bags :
  (total_peaches - peaches_in_knapsack) / peaches_per_bag = 2 := by
  sorry

end number_of_cloth_bags_l2414_241452


namespace intersection_eq_singleton_l2414_241451

-- Defining the sets M and N
def M : Set ℤ := {-1, 1, -2, 2}
def N : Set ℤ := {1, 4}

-- Stating the intersection problem
theorem intersection_eq_singleton :
  M ∩ N = {1} := 
by 
  sorry

end intersection_eq_singleton_l2414_241451


namespace incorrect_conclusion_l2414_241462

-- Define the linear regression model
def model (x : ℝ) : ℝ := 0.85 * x - 85.71

-- Define the conditions
axiom linear_correlation : ∀ (x y : ℝ), ∃ (x_i y_i : ℝ) (i : ℕ), model x = y

-- The theorem to prove the statement for x = 170 is false
theorem incorrect_conclusion (x : ℝ) (h : x = 170) : ¬ (model x = 58.79) :=
  by sorry

end incorrect_conclusion_l2414_241462


namespace find_p_l2414_241433

theorem find_p
  (A B C r s p q : ℝ)
  (h1 : A ≠ 0)
  (h2 : r + s = -B / A)
  (h3 : r * s = C / A)
  (h4 : r^3 + s^3 = -p) :
  p = (B^3 - 3 * A * B * C) / A^3 :=
by {
  sorry
}

end find_p_l2414_241433


namespace find_correct_answer_l2414_241405

theorem find_correct_answer (x : ℕ) (h : 3 * x = 135) : x / 3 = 15 :=
sorry

end find_correct_answer_l2414_241405


namespace acid_solution_replacement_percentage_l2414_241491

theorem acid_solution_replacement_percentage 
  (original_concentration fraction_replaced final_concentration replaced_percentage : ℝ)
  (h₁ : original_concentration = 0.50)
  (h₂ : fraction_replaced = 0.5)
  (h₃ : final_concentration = 0.40)
  (h₄ : 0.25 + fraction_replaced * replaced_percentage = final_concentration) :
  replaced_percentage = 0.30 :=
by
  sorry

end acid_solution_replacement_percentage_l2414_241491


namespace total_sheets_folded_l2414_241400

theorem total_sheets_folded (initially_folded : ℕ) (additionally_folded : ℕ) (total_folded : ℕ) :
  initially_folded = 45 → additionally_folded = 18 → total_folded = initially_folded + additionally_folded → total_folded = 63 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3


end total_sheets_folded_l2414_241400


namespace find_a8_in_arithmetic_sequence_l2414_241484

variable {a : ℕ → ℕ} -- Define a as a function from natural numbers to natural numbers

-- Assume a is an arithmetic sequence
axiom arithmetic_sequence (a : ℕ → ℕ) : ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem find_a8_in_arithmetic_sequence (h : a 4 + a 6 + a 8 + a 10 + a 12 = 120) : a 8 = 24 :=
by
  sorry  -- Proof to be filled in separately

end find_a8_in_arithmetic_sequence_l2414_241484


namespace parabola_focus_coordinates_l2414_241468

theorem parabola_focus_coordinates (x y : ℝ) (h : y = -2 * x^2) : (0, -1 / 8) = (0, (-1 / 2) * (y: ℝ)) :=
sorry

end parabola_focus_coordinates_l2414_241468


namespace downstream_distance_80_l2414_241480

-- Conditions
variables (Speed_boat Speed_stream Distance_upstream : ℝ)

-- Assign given values
def speed_boat := 36 -- kmph
def speed_stream := 12 -- kmph
def distance_upstream := 40 -- km

-- Effective speeds
def speed_downstream := speed_boat + speed_stream -- kmph
def speed_upstream := speed_boat - speed_stream -- kmph

-- Downstream distance
noncomputable def distance_downstream : ℝ := 80 -- km

-- Theorem
theorem downstream_distance_80 :
  speed_boat = 36 → speed_stream = 12 → distance_upstream = 40 →
  (distance_upstream / speed_upstream = distance_downstream / speed_downstream) :=
by
  sorry

end downstream_distance_80_l2414_241480


namespace number_of_chickens_l2414_241424

def cost_per_chicken := 3
def total_cost := 15
def potato_cost := 6
def remaining_amount := total_cost - potato_cost

theorem number_of_chickens : (total_cost - potato_cost) / cost_per_chicken = 3 := by
  sorry

end number_of_chickens_l2414_241424


namespace bouquet_combinations_l2414_241493

theorem bouquet_combinations :
  ∃ n : ℕ, (∀ r c t : ℕ, 4 * r + 3 * c + 2 * t = 60 → true) ∧ n = 13 :=
sorry

end bouquet_combinations_l2414_241493


namespace money_left_is_correct_l2414_241416

noncomputable def total_income : ℝ := 800000
noncomputable def children_pct : ℝ := 0.2
noncomputable def num_children : ℝ := 3
noncomputable def wife_pct : ℝ := 0.3
noncomputable def donation_pct : ℝ := 0.05

noncomputable def remaining_income_after_donations : ℝ := 
  let distributed_to_children := total_income * children_pct * num_children
  let distributed_to_wife := total_income * wife_pct
  let total_distributed := distributed_to_children + distributed_to_wife
  let remaining_after_family := total_income - total_distributed
  let donation := remaining_after_family * donation_pct
  remaining_after_family - donation

theorem money_left_is_correct :
  remaining_income_after_donations = 76000 := 
by 
  sorry

end money_left_is_correct_l2414_241416


namespace matrix_det_eq_seven_l2414_241423

theorem matrix_det_eq_seven (p q r s : ℝ) (h : p * s - q * r = 7) : 
  (p - 2 * r) * s - (q - 2 * s) * r = 7 := 
sorry

end matrix_det_eq_seven_l2414_241423


namespace even_function_value_l2414_241450

theorem even_function_value (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = 2 ^ x) :
  f (Real.log 9 / Real.log 4) = 1 / 3 :=
by
  sorry

end even_function_value_l2414_241450


namespace max_third_altitude_l2414_241469

theorem max_third_altitude (h1 h2 : ℕ) (h1_eq : h1 = 6) (h2_eq : h2 = 18) (triangle_scalene : true)
: (exists h3 : ℕ, (∀ h3_alt > h3, h3_alt > 8)) := 
sorry

end max_third_altitude_l2414_241469


namespace ages_correct_in_2018_l2414_241490

-- Define the initial ages in the year 2000
def age_marianne_2000 : ℕ := 20
def age_bella_2000 : ℕ := 8
def age_carmen_2000 : ℕ := 15

-- Define the birth year of Elli
def birth_year_elli : ℕ := 2003

-- Define the target year when Bella turns 18
def year_bella_turns_18 : ℕ := 2000 + 18

-- Define the ages to be proven
def age_marianne_2018 : ℕ := 30
def age_carmen_2018 : ℕ := 33
def age_elli_2018 : ℕ := 15

theorem ages_correct_in_2018 :
  age_marianne_2018 = age_marianne_2000 + (year_bella_turns_18 - 2000) ∧
  age_carmen_2018 = age_carmen_2000 + (year_bella_turns_18 - 2000) ∧
  age_elli_2018 = year_bella_turns_18 - birth_year_elli :=
by 
  -- The proof would go here
  sorry

end ages_correct_in_2018_l2414_241490


namespace cylinder_height_percentage_l2414_241411

-- Lean 4 statement for the problem
theorem cylinder_height_percentage (h : ℝ) (r : ℝ) (H : ℝ) :
  (7 / 8) * h = (3 / 5) * (1.25 * r)^2 * H → H = 0.9333 * h :=
by 
  sorry

end cylinder_height_percentage_l2414_241411


namespace vector_dot_product_l2414_241485

-- Define the vectors a and b
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (1, -1)

-- Prove that the scalar product a · (a - 2b) equals 2
theorem vector_dot_product :
  let u := a
  let v := b
  u • (u - (2 • v)) = 2 :=
by 
  -- Placeholder for the proof
  sorry

end vector_dot_product_l2414_241485


namespace water_on_wednesday_l2414_241402

-- Define the total water intake for the week.
def total_water : ℕ := 60

-- Define the water intake amounts for specific days.
def water_on_mon_thu_sat : ℕ := 9
def water_on_tue_fri_sun : ℕ := 8

-- Define the number of days for each intake.
def days_mon_thu_sat : ℕ := 3
def days_tue_fri_sun : ℕ := 3

-- Define the water intake calculated for specific groups of days.
def total_water_mon_thu_sat := water_on_mon_thu_sat * days_mon_thu_sat
def total_water_tue_fri_sun := water_on_tue_fri_sun * days_tue_fri_sun

-- Define the total water intake for these days combined.
def total_water_other_days := total_water_mon_thu_sat + total_water_tue_fri_sun

-- Define the water intake for Wednesday, which we need to prove to be 9 liters.
theorem water_on_wednesday : total_water - total_water_other_days = 9 := by
  -- Proof omitted.
  sorry

end water_on_wednesday_l2414_241402


namespace students_at_year_end_l2414_241446

theorem students_at_year_end (initial_students left_students new_students end_students : ℕ)
  (h_initial : initial_students = 31)
  (h_left : left_students = 5)
  (h_new : new_students = 11)
  (h_end : end_students = initial_students - left_students + new_students) :
  end_students = 37 :=
by
  sorry

end students_at_year_end_l2414_241446


namespace expand_remains_same_l2414_241483

variable (m n : ℤ)

-- Define a function that represents expanding m and n by a factor of 3
def expand_by_factor_3 (m n : ℤ) : ℤ := 
  2 * (3 * m) / (3 * m - 3 * n)

-- Define the original fraction
def original_fraction (m n : ℤ) : ℤ :=
  2 * m / (m - n)

-- Theorem to prove that expanding m and n by a factor of 3 does not change the fraction
theorem expand_remains_same (m n : ℤ) : 
  expand_by_factor_3 m n = original_fraction m n := 
by sorry

end expand_remains_same_l2414_241483


namespace range_of_a_l2414_241447

theorem range_of_a :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ -1 ≤ a ∧ a ≤ 3 := 
by
  sorry

end range_of_a_l2414_241447


namespace smallest_positive_multiple_of_45_is_45_l2414_241448

-- Define that a positive multiple of 45 is parameterized by a positive integer
def is_positive_multiple (m : ℕ) : Prop := ∃ x : ℕ, x > 0 ∧ m = 45 * x

-- State the theorem
theorem smallest_positive_multiple_of_45_is_45 :
  (∀ m : ℕ, is_positive_multiple m → (45 ≤ m)) :=
by
  sorry

end smallest_positive_multiple_of_45_is_45_l2414_241448


namespace total_cans_to_collect_l2414_241460

def cans_for_project (marthas_cans : ℕ) (additional_cans_needed : ℕ) (total_cans_needed : ℕ) : Prop :=
  ∃ diegos_cans : ℕ, diegos_cans = (marthas_cans / 2) + 10 ∧ 
  total_cans_needed = marthas_cans + diegos_cans + additional_cans_needed

theorem total_cans_to_collect : 
  cans_for_project 90 5 150 :=
by
  -- Insert proof here in actual usage
  sorry

end total_cans_to_collect_l2414_241460


namespace parabola_equation_l2414_241420

theorem parabola_equation :
  (∃ h k : ℝ, h^2 = 3 ∧ k^2 = 6) →
  (∃ c : ℝ, c^2 = (3 + 6)) →
  (∃ x y : ℝ, x = 3 ∧ y = 0) →
  (y^2 = 12 * x) :=
sorry

end parabola_equation_l2414_241420


namespace sticks_per_pot_is_181_l2414_241470

/-- Define the problem conditions -/
def number_of_pots : ℕ := 466
def flowers_per_pot : ℕ := 53
def total_flowers_and_sticks : ℕ := 109044

/-- Define the function to calculate the number of sticks per pot -/
def sticks_per_pot (S : ℕ) : Prop :=
  (number_of_pots * flowers_per_pot + number_of_pots * S = total_flowers_and_sticks)

/-- State the theorem -/
theorem sticks_per_pot_is_181 : sticks_per_pot 181 :=
by
  sorry

end sticks_per_pot_is_181_l2414_241470


namespace commission_percentage_l2414_241442

def commission_rate (amount: ℕ) : ℚ :=
  if amount <= 500 then
    0.20 * amount
  else
    0.20 * 500 + 0.50 * (amount - 500)

theorem commission_percentage (total_sale : ℕ) (h : total_sale = 800) :
  (commission_rate total_sale) / total_sale * 100 = 31.25 :=
by
  sorry

end commission_percentage_l2414_241442


namespace new_average_after_increase_and_bonus_l2414_241403

theorem new_average_after_increase_and_bonus 
  (n : ℕ) (initial_avg : ℝ) (k : ℝ) (bonus : ℝ) 
  (h1: n = 37) 
  (h2: initial_avg = 73) 
  (h3: k = 1.65) 
  (h4: bonus = 15) 
  : (initial_avg * k) + bonus = 135.45 := 
sorry

end new_average_after_increase_and_bonus_l2414_241403


namespace fraction_given_away_is_three_fifths_l2414_241404

variable (initial_bunnies : ℕ) (final_bunnies : ℕ) (kittens_per_bunny : ℕ)

def fraction_given_away (given_away : ℕ) (initial_bunnies : ℕ) : ℚ :=
  given_away / initial_bunnies

theorem fraction_given_away_is_three_fifths 
  (initial_bunnies : ℕ := 30) (final_bunnies : ℕ := 54) (kittens_per_bunny : ℕ := 2)
  (h : final_bunnies = initial_bunnies + kittens_per_bunny * (initial_bunnies - 18)) : 
  fraction_given_away 18 initial_bunnies = 3 / 5 :=
by
  sorry

end fraction_given_away_is_three_fifths_l2414_241404


namespace total_chips_is_90_l2414_241499

theorem total_chips_is_90
  (viv_vanilla : ℕ)
  (sus_choco : ℕ)
  (viv_choco_more : ℕ)
  (sus_vanilla_ratio : ℚ)
  (viv_choco : ℕ)
  (sus_vanilla : ℕ)
  (total_choco : ℕ)
  (total_vanilla : ℕ)
  (total_chips : ℕ) :
  viv_vanilla = 20 →
  sus_choco = 25 →
  viv_choco_more = 5 →
  sus_vanilla_ratio = 3 / 4 →
  viv_choco = sus_choco + viv_choco_more →
  sus_vanilla = (sus_vanilla_ratio * viv_vanilla) →
  total_choco = viv_choco + sus_choco →
  total_vanilla = viv_vanilla + sus_vanilla →
  total_chips = total_choco + total_vanilla →
  total_chips = 90 :=
by
  intros
  sorry

end total_chips_is_90_l2414_241499


namespace tree_cost_calculation_l2414_241464

theorem tree_cost_calculation :
  let c := 1500 -- park circumference in meters
  let i := 30 -- interval distance in meters
  let p := 5000 -- price per tree in mill
  let n := c / i -- number of trees
  let cost := n * p -- total cost in mill
  cost = 250000 :=
by
  sorry

end tree_cost_calculation_l2414_241464


namespace line_through_point_area_T_l2414_241476

variable (a T : ℝ)

def triangle_line_equation (a T : ℝ) : Prop :=
  ∃ y x : ℝ, (a^2 * y + 2 * T * x - 2 * a * T = 0) ∧ (y = -((2 * T)/a^2) * x + (2 * T) / a) ∧ (x ≥ 0) ∧ (y ≥ 0)

theorem line_through_point_area_T (a T : ℝ) (h₁ : a > 0) (h₂ : T > 0) :
  triangle_line_equation a T :=
sorry

end line_through_point_area_T_l2414_241476


namespace product_sum_of_roots_l2414_241471

theorem product_sum_of_roots
  {p q r : ℝ}
  (h : (∀ x : ℝ, (4 * x^3 - 8 * x^2 + 16 * x - 12) = 0 → (x = p ∨ x = q ∨ x = r))) :
  p * q + q * r + r * p = 4 := 
sorry

end product_sum_of_roots_l2414_241471


namespace smallest_ninequality_l2414_241410

theorem smallest_ninequality 
  (n : ℕ) 
  (h : ∀ x : ℝ, (Real.sin x) ^ n + (Real.cos x) ^ n ≤ 2 ^ (1 - n)) : 
  n = 2 := 
by
  sorry

end smallest_ninequality_l2414_241410


namespace total_apples_picked_l2414_241435

-- Define the number of apples picked by Benny
def applesBenny : Nat := 2

-- Define the number of apples picked by Dan
def applesDan : Nat := 9

-- The theorem we want to prove
theorem total_apples_picked : applesBenny + applesDan = 11 := 
by 
  sorry

end total_apples_picked_l2414_241435


namespace average_of_remaining_two_numbers_l2414_241413

theorem average_of_remaining_two_numbers 
(A B C D E F G H : ℝ) 
(h_avg1 : (A + B + C + D + E + F + G + H) / 8 = 4.5) 
(h_avg2 : (A + B + C) / 3 = 5.2) 
(h_avg3 : (D + E + F) / 3 = 3.6) : 
  ((G + H) / 2 = 4.8) :=
sorry

end average_of_remaining_two_numbers_l2414_241413


namespace unknown_number_l2414_241474

theorem unknown_number (n : ℕ) (h1 : Nat.lcm 24 n = 168) (h2 : Nat.gcd 24 n = 4) : n = 28 :=
by
  sorry

end unknown_number_l2414_241474


namespace polygons_intersection_l2414_241419

/-- In a square with an area of 5, nine polygons, each with an area of 1, are placed. 
    Prove that some two of them must have an intersection area of at least 1 / 9. -/
theorem polygons_intersection 
  (S : ℝ) (hS : S = 5)
  (n : ℕ) (hn : n = 9)
  (polygons : Fin n → ℝ) (hpolys : ∀ i, polygons i = 1)
  (intersection : Fin n → Fin n → ℝ) : 
  ∃ i j : Fin n, i ≠ j ∧ intersection i j ≥ 1 / 9 := 
sorry

end polygons_intersection_l2414_241419


namespace solution_one_solution_two_l2414_241436

section

variables {a x : ℝ}

def f (x : ℝ) (a : ℝ) := |2 * x - a| - |x + 1|

-- (1) Prove the solution set for f(x) > 2 when a = 1 is (-∞, -2/3) ∪ (4, ∞)
theorem solution_one (x : ℝ) : f x 1 > 2 ↔ x < -2/3 ∨ x > 4 :=
by sorry

-- (2) Prove the range of a for which f(x) + |x + 1| + x > a² - 1/2 always holds for x ∈ ℝ is (-1/2, 1)
theorem solution_two (a : ℝ) : 
  (∀ x, f x a + |x + 1| + x > a^2 - 1/2) ↔ -1/2 < a ∧ a < 1 :=
by sorry

end

end solution_one_solution_two_l2414_241436


namespace factorial_expression_l2414_241415

theorem factorial_expression :
  (4 * (Nat.factorial 7) + 28 * (Nat.factorial 6)) / Nat.factorial 8 = 1 := by
  sorry

end factorial_expression_l2414_241415


namespace prob_black_yellow_l2414_241437

theorem prob_black_yellow:
  ∃ (x y : ℚ), 12 > 0 ∧
  (∃ (r b y' : ℚ), r = 1/3 ∧ b - y' = 1/6 ∧ b + y' = 2/3 ∧ r + b + y' = 1) ∧
  x = 5/12 ∧ y = 1/4 :=
by
  sorry

end prob_black_yellow_l2414_241437


namespace degrees_for_basic_astrophysics_correct_l2414_241465

-- Definitions for conditions
def percentage_allocations : List ℚ := [13, 24, 15, 29, 8]
def total_percentage : ℚ := percentage_allocations.sum
def remaining_percentage : ℚ := 100 - total_percentage

-- The question to answer
def total_degrees : ℚ := 360
def degrees_for_basic_astrophysics : ℚ := remaining_percentage / 100 * total_degrees

-- Prove that the degrees for basic astrophysics is 39.6
theorem degrees_for_basic_astrophysics_correct :
  degrees_for_basic_astrophysics = 39.6 :=
by
  sorry

end degrees_for_basic_astrophysics_correct_l2414_241465


namespace speed_second_hour_l2414_241426

noncomputable def speed_in_first_hour : ℝ := 95
noncomputable def average_speed : ℝ := 77.5
noncomputable def total_time : ℝ := 2
def speed_in_second_hour : ℝ := sorry -- to be deduced

theorem speed_second_hour :
  speed_in_second_hour = 60 :=
by
  sorry

end speed_second_hour_l2414_241426


namespace integral_exp_neg_l2414_241440

theorem integral_exp_neg : ∫ x in (Set.Ioi 0), Real.exp (-x) = 1 := sorry

end integral_exp_neg_l2414_241440


namespace candies_problem_l2414_241458

theorem candies_problem (emily jennifer bob : ℕ) (h1 : emily = 6) 
  (h2 : jennifer = 2 * emily) (h3 : jennifer = 3 * bob) : bob = 4 := by
  -- Lean code to skip the proof
  sorry

end candies_problem_l2414_241458


namespace product_mod_self_inverse_l2414_241431

theorem product_mod_self_inverse 
  {n : ℕ} (hn : 0 < n) (a b : ℤ) (ha : a * a % n = 1) (hb : b * b % n = 1) :
  (a * b) % n = 1 := 
sorry

end product_mod_self_inverse_l2414_241431


namespace alice_steps_l2414_241459

noncomputable def num_sticks (n : ℕ) : ℕ :=
  (n + 1 : ℕ) ^ 2

theorem alice_steps (n : ℕ) (h : num_sticks n = 169) : n = 13 :=
by sorry

end alice_steps_l2414_241459


namespace calculate_Al2O3_weight_and_H2_volume_l2414_241453

noncomputable def weight_of_Al2O3 (moles : ℕ) : ℝ :=
  moles * ((2 * 26.98) + (3 * 16.00))

noncomputable def volume_of_H2_at_STP (moles_of_Al2O3 : ℕ) : ℝ :=
  (moles_of_Al2O3 * 3) * 22.4

theorem calculate_Al2O3_weight_and_H2_volume :
  weight_of_Al2O3 6 = 611.76 ∧ volume_of_H2_at_STP 6 = 403.2 :=
by
  sorry

end calculate_Al2O3_weight_and_H2_volume_l2414_241453


namespace solve_abc_l2414_241466

theorem solve_abc (a b c : ℝ) (h1 : a ≤ b ∧ b ≤ c) (h2 : a + b + c = -1) (h3 : a * b + b * c + a * c = -4) (h4 : a * b * c = -2) :
  a = -1 - Real.sqrt 3 ∧ b = -1 + Real.sqrt 3 ∧ c = 1 :=
by
  -- Proof goes here
  sorry

end solve_abc_l2414_241466


namespace correct_equation_l2414_241488

variable (x : ℝ) (h1 : x > 0)

def length_pipeline : ℝ := 3000
def efficiency_increase : ℝ := 0.2
def days_ahead : ℝ := 10

theorem correct_equation :
  (length_pipeline / x) - (length_pipeline / ((1 + efficiency_increase) * x)) = days_ahead :=
by
  sorry

end correct_equation_l2414_241488
