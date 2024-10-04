import Mathlib

namespace evaluate_expression_l26_26780

theorem evaluate_expression : (528 * 528) - (527 * 529) = 1 := by
  sorry

end evaluate_expression_l26_26780


namespace pine_cone_weight_on_roof_l26_26332

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l26_26332


namespace correct_inequality_l26_26379

theorem correct_inequality (a b c d : ℝ)
    (hab : a > b) (hb0 : b > 0)
    (hcd : c > d) (hd0 : d > 0) :
    Real.sqrt (a / d) > Real.sqrt (b / c) :=
by
    sorry

end correct_inequality_l26_26379


namespace octal_addition_l26_26662

theorem octal_addition (x y : ℕ) (h1 : x = 1 * 8^3 + 4 * 8^2 + 6 * 8^1 + 3 * 8^0)
                     (h2 : y = 2 * 8^2 + 7 * 8^1 + 5 * 8^0) :
  x + y = 1 * 8^3 + 7 * 8^2 + 5 * 8^1 + 0 * 8^0 := sorry

end octal_addition_l26_26662


namespace matrix_sum_correct_l26_26343

def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![4, -3],
  ![2, 5]
]

def B : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-6, 8],
  ![-3, 7]
]

def C : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-2, 5],
  ![-1, 12]
]

theorem matrix_sum_correct : A + B = C := by
  sorry

end matrix_sum_correct_l26_26343


namespace Liked_Both_Proof_l26_26533

section DessertProblem

variable (Total_Students Liked_Apple_Pie Liked_Chocolate_Cake Did_Not_Like_Either Liked_Both : ℕ)
variable (h1 : Total_Students = 50)
variable (h2 : Liked_Apple_Pie = 25)
variable (h3 : Liked_Chocolate_Cake = 20)
variable (h4 : Did_Not_Like_Either = 10)

theorem Liked_Both_Proof :
  Liked_Both = (Liked_Apple_Pie + Liked_Chocolate_Cake) - (Total_Students - Did_Not_Like_Either) :=
by
  sorry

end DessertProblem

end Liked_Both_Proof_l26_26533


namespace problem_solution_l26_26976

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end problem_solution_l26_26976


namespace largest_frog_weight_l26_26582

theorem largest_frog_weight (S L : ℕ) (h1 : L = 10 * S) (h2 : L = S + 108): L = 120 := by
  sorry

end largest_frog_weight_l26_26582


namespace salary_proof_l26_26291

-- Defining the monthly salaries of the officials
def D_Dupon : ℕ := 6000
def D_Duran : ℕ := 8000
def D_Marten : ℕ := 5000

-- Defining the statements made by each official
def Dupon_statement1 : Prop := D_Dupon = 6000
def Dupon_statement2 : Prop := D_Duran = D_Dupon + 2000
def Dupon_statement3 : Prop := D_Marten = D_Dupon - 1000

def Duran_statement1 : Prop := D_Duran > D_Marten
def Duran_statement2 : Prop := D_Duran - D_Marten = 3000
def Duran_statement3 : Prop := D_Marten = 9000

def Marten_statement1 : Prop := D_Marten < D_Dupon
def Marten_statement2 : Prop := D_Dupon = 7000
def Marten_statement3 : Prop := D_Duran = D_Dupon + 3000

-- Defining the constraints about the number of truth and lies
def Told_the_truth_twice_and_lied_once : Prop :=
  (Dupon_statement1 ∧ Dupon_statement2 ∧ ¬Dupon_statement3) ∨
  (Dupon_statement1 ∧ ¬Dupon_statement2 ∧ Dupon_statement3) ∨
  (¬Dupon_statement1 ∧ Dupon_statement2 ∧ Dupon_statement3) ∨
  (Duran_statement1 ∧ Duran_statement2 ∧ ¬Duran_statement3) ∨
  (Duran_statement1 ∧ ¬Duran_statement2 ∧ Duran_statement3) ∨
  (¬Duran_statement1 ∧ Duran_statement2 ∧ Duran_statement3) ∨
  (Marten_statement1 ∧ Marten_statement2 ∧ ¬Marten_statement3) ∨
  (Marten_statement1 ∧ ¬Marten_statement2 ∧ Marten_statement3) ∨
  (¬Marten_statement1 ∧ Marten_statement2 ∧ Marten_statement3)

-- The final proof goal
theorem salary_proof : Told_the_truth_twice_and_lied_once →
  D_Dupon = 6000 ∧ D_Duran = 8000 ∧ D_Marten = 5000 := by 
  sorry

end salary_proof_l26_26291


namespace polygon_E_has_largest_area_l26_26501

-- Define the areas of square and right triangle
def area_square (side : ℕ): ℕ := side * side
def area_right_triangle (leg : ℕ): ℕ := (leg * leg) / 2

-- Define the areas of each polygon
def area_polygon_A : ℕ := 2 * (area_square 2) + (area_right_triangle 2)
def area_polygon_B : ℕ := 3 * (area_square 2)
def area_polygon_C : ℕ := (area_square 2) + 4 * (area_right_triangle 2)
def area_polygon_D : ℕ := 3 * (area_right_triangle 2)
def area_polygon_E : ℕ := 4 * (area_square 2)

-- The theorem assertion
theorem polygon_E_has_largest_area : 
  area_polygon_E = 16 ∧ 
  16 > area_polygon_A ∧
  16 > area_polygon_B ∧
  16 > area_polygon_C ∧
  16 > area_polygon_D := 
sorry

end polygon_E_has_largest_area_l26_26501


namespace only_solution_l26_26355

theorem only_solution (a b c : ℕ) (ha_pos : 0 < a) (hb_pos : 0 < b) (hc_pos : 0 < c)
    (h_le : a ≤ b ∧ b ≤ c) (h_gcd : Int.gcd (Int.gcd a b) c = 1) 
    (h_div_a2b : a^3 + b^3 + c^3 % (a^2 * b) = 0)
    (h_div_b2c : a^3 + b^3 + c^3 % (b^2 * c) = 0)
    (h_div_c2a : a^3 + b^3 + c^3 % (c^2 * a) = 0) : 
    a = 1 ∧ b = 1 ∧ c = 1 :=
  by
  sorry

end only_solution_l26_26355


namespace probability_sum_m_n_reaches_4_4_l26_26430

theorem probability_sum_m_n_reaches_4_4 :
  let p : ℚ := (1575 : ℚ) / 262144 in
  p.denom.coprime p.num ∧ (p.num + p.denom = 263719) := 
by
  sorry

end probability_sum_m_n_reaches_4_4_l26_26430


namespace integer_solutions_l26_26428

-- Define the equation to be solved
def equation (x y : ℤ) : Prop := x * y + 3 * x - 5 * y + 3 = 0

-- Define the solutions
def solution_set : List (ℤ × ℤ) := 
  [(-13,-2), (-4,-1), (-1,0), (2, 3), (3, 6), (4, 15), (6, -21),
   (7, -12), (8, -9), (11, -6), (14, -5), (23, -4)]

-- The theorem stating the solutions are correct
theorem integer_solutions : ∀ (x y : ℤ), (x, y) ∈ solution_set → equation x y :=
by
  sorry

end integer_solutions_l26_26428


namespace salon_customers_l26_26324

variables (n : ℕ) (c : ℕ)

theorem salon_customers :
  ∀ (n = 33) (extra_cans = 5) (cans_per_customer = 2),
  (n - extra_cans) / cans_per_customer = 14 :=
begin
  sorry
end

end salon_customers_l26_26324


namespace hyperbola_asymptotes_equation_l26_26006

noncomputable def hyperbola_asymptotes (x y : ℝ) : Prop :=
  (x^2 / 4 - y^2 / 9 = 1) → (y = (3 / 2) * x) ∨ (y = -(3 / 2) * x)

-- Now we assert the theorem that states this
theorem hyperbola_asymptotes_equation :
  ∀ (x y : ℝ), hyperbola_asymptotes x y :=
by
  intros x y
  unfold hyperbola_asymptotes
  -- proof here
  sorry

end hyperbola_asymptotes_equation_l26_26006


namespace river_flow_rate_l26_26637

theorem river_flow_rate
  (depth width volume_per_minute : ℝ)
  (h1 : depth = 2)
  (h2 : width = 45)
  (h3 : volume_per_minute = 6000) :
  (volume_per_minute / (depth * width)) * (1 / 1000) * 60 = 4.0002 :=
by
  -- Sorry is used to skip the proof.
  sorry

end river_flow_rate_l26_26637


namespace beetle_total_distance_l26_26179

theorem beetle_total_distance (r : ℝ) (r_eq : r = 75) : (2 * r + r + r) = 300 := 
by
  sorry

end beetle_total_distance_l26_26179


namespace equation_solutions_l26_26270

theorem equation_solutions : 
  ∀ x : ℝ, (2 * x - 1) - x * (1 - 2 * x) = 0 ↔ (x = 1 / 2 ∨ x = -1) :=
by
  intro x
  sorry

end equation_solutions_l26_26270


namespace number_exceeds_part_l26_26916

theorem number_exceeds_part (x : ℝ) (h : x = (5 / 9) * x + 150) : x = 337.5 := sorry

end number_exceeds_part_l26_26916


namespace marathon_times_total_l26_26129

theorem marathon_times_total 
  (runs_as_fast : ℝ → ℝ → Prop)
  (takes_more_time : ℝ → ℝ → ℝ → Prop)
  (dean_time : ℝ)
  (h_micah_speed : runs_as_fast 2/3 1)
  (h_jake_time : ∀ t, takes_more_time 1/3 t (t * 4/3))
  (h_dean_time : dean_time = 9) :
  let micah_time := dean_time * (2/3)
  let jake_time := micah_time + (1/3 * micah_time)
  dean_time + micah_time + jake_time = 23 :=
by
  sorry

end marathon_times_total_l26_26129


namespace polynomial_square_l26_26667

theorem polynomial_square (x : ℝ) : x^4 + 2*x^3 - 2*x^2 - 4*x - 5 = y^2 → x = 3 ∨ x = -3 := by
  sorry

end polynomial_square_l26_26667


namespace gcd_of_a_and_b_is_one_l26_26857

theorem gcd_of_a_and_b_is_one {a b : ℕ} (h1 : a > b) (h2 : Nat.gcd (a + b) (a - b) = 1) : Nat.gcd a b = 1 :=
by
  sorry

end gcd_of_a_and_b_is_one_l26_26857


namespace find_positive_integer_solutions_l26_26218

theorem find_positive_integer_solutions (x y z : ℕ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  2^x + 3^y = z^2 ↔ (x = 0 ∧ y = 1 ∧ z = 2) ∨ (x = 3 ∧ y = 0 ∧ z = 3) ∨ (x = 4 ∧ y = 2 ∧ z = 5) := 
sorry

end find_positive_integer_solutions_l26_26218


namespace all_are_multiples_of_3_l26_26517

theorem all_are_multiples_of_3 :
  (123 % 3 = 0) ∧
  (234 % 3 = 0) ∧
  (345 % 3 = 0) ∧
  (456 % 3 = 0) ∧
  (567 % 3 = 0) :=
by
  sorry

end all_are_multiples_of_3_l26_26517


namespace pencils_per_row_l26_26660

-- Define the conditions as parameters
variables (total_pencils : Int) (rows : Int) 

-- State the proof problem using the conditions and the correct answer
theorem pencils_per_row (h₁ : total_pencils = 12) (h₂ : rows = 3) : total_pencils / rows = 4 := 
by 
  sorry

end pencils_per_row_l26_26660


namespace probability_even_sum_l26_26567

theorem probability_even_sum (tiles : Finset ℕ) (players : Finset (Finset ℕ)) :
  tiles = {1, 2, 3, 4, 5, 6, 7, 8, 9} →
  players.card = 3 →
  (∀ p ∈ players, p.card = 3 ∧ (∑ t in p, t) % 2 = 0) →
  ∃ m n : ℕ, Nat.coprime m n ∧ (∀ m_o n_o : ℕ, m * n_o = m_o * n → n_o = n -> m + n = 8) :=
begin
  intro h_tiles,
  intro h_players_card,
  intro h_player_conditions,
  sorry
end

end probability_even_sum_l26_26567


namespace polygon_sides_l26_26635

theorem polygon_sides (n : ℕ) (h_sum : 180 * (n - 2) = 1980) : n = 13 :=
by {
  sorry
}

end polygon_sides_l26_26635


namespace employee_n_weekly_wage_l26_26002

theorem employee_n_weekly_wage (Rm Rn : ℝ) (Hm Hn : ℝ) 
    (h1 : (Rm * Hm) + (Rn * Hn) = 770) 
    (h2 : (Rm * Hm) = 1.3 * (Rn * Hn)) :
    Rn * Hn = 335 :=
by
  sorry

end employee_n_weekly_wage_l26_26002


namespace total_amount_l26_26640

theorem total_amount (W X Y Z : ℝ) (h1 : X = 0.8 * W) (h2 : Y = 0.65 * W) (h3 : Z = 0.45 * W) (h4 : Y = 78) : 
  W + X + Y + Z = 348 := by
  sorry

end total_amount_l26_26640


namespace find_f_prime_at_1_l26_26964

variable (f : ℝ → ℝ)

-- Initial condition
variable (h : ∀ x, f x = x^2 + deriv f 2 * (Real.log x - x))

-- The goal is to prove that f'(1) = 2
theorem find_f_prime_at_1 : deriv f 1 = 2 :=
by
  sorry

end find_f_prime_at_1_l26_26964


namespace minimum_value_l26_26248

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 20) :
  (∃ (m : ℝ), m = (1 / x ^ 2 + 1 / y ^ 2) ∧ m ≥ 2 / 25) :=
by
  sorry

end minimum_value_l26_26248


namespace width_of_field_l26_26886

theorem width_of_field (W L : ℝ) (h1 : L = (7 / 5) * W) (h2 : 2 * L + 2 * W = 360) : W = 75 :=
sorry

end width_of_field_l26_26886


namespace alice_bob_get_same_heads_mn_sum_correct_l26_26191

def prob_heads_eq (fair_prob : ℚ) (biased_prob : ℚ) (num_heads : ℚ) : ℚ :=
  let fair_gen := 1 + fair_prob in
  let biased_gen := 2 * biased_prob + 3 in
  (fair_gen ^ 2 * biased_gen).num ^ 2 - ((fair_prob ^ 2 + 8 * fair_prob + biased_prob ^ 3) ^ 2 + (9 + 64 + 49 + 4))
  
#check prob_heads_eq (1/2) (3/5) (63/200)

theorem alice_bob_get_same_heads : prob_heads_eq (1/2) (3/5) (63/200) = 63 / 200 :=
by sorry

theorem mn_sum_correct : 63 + 200 = 263 :=
by sorry

end alice_bob_get_same_heads_mn_sum_correct_l26_26191


namespace smallest_value_of_f4_l26_26995

def f (x : ℝ) : ℝ := (x + 3) ^ 2 - 2

theorem smallest_value_of_f4 : ∀ x : ℝ, f (f (f (f x))) ≥ 23 :=
by 
  sorry -- Proof goes here.

end smallest_value_of_f4_l26_26995


namespace top_card_is_joker_probability_l26_26326

theorem top_card_is_joker_probability :
  let totalCards := 54
  let jokerCards := 2
  let probability := (jokerCards : ℚ) / (totalCards : ℚ)
  probability = 1 / 27 :=
by
  sorry

end top_card_is_joker_probability_l26_26326


namespace find_angle_B_l26_26101

theorem find_angle_B (a b c : ℝ) (A B C : ℝ)
  (h1 : (sin B - sin A, sqrt 3 * a + c) = (sin C, a + b)) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l26_26101


namespace has_zero_in_intervals_l26_26561

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x
noncomputable def f' (x : ℝ) : ℝ := (1 / 3) - (1 / x)

theorem has_zero_in_intervals : 
  (∃ x : ℝ, 0 < x ∧ x < 3 ∧ f x = 0) ∧ (∃ x : ℝ, 3 < x ∧ f x = 0) :=
sorry

end has_zero_in_intervals_l26_26561


namespace exist_unique_xy_solution_l26_26668

theorem exist_unique_xy_solution :
  ∃! (x y : ℝ), x^2 + (1 - y)^2 + (x - y)^2 = 1 / 3 ∧ x = 1 / 3 ∧ y = 2 / 3 :=
by
  sorry

end exist_unique_xy_solution_l26_26668


namespace sum_of_integers_l26_26794

theorem sum_of_integers (a b c d : ℕ) (h1 : a > 1) (h2 : b > 1) (h3 : c > 1) (h4 : d > 1)
    (h_prod : a * b * c * d = 1000000)
    (h_gcd1 : Nat.gcd a b = 1) (h_gcd2 : Nat.gcd a c = 1) (h_gcd3 : Nat.gcd a d = 1)
    (h_gcd4 : Nat.gcd b c = 1) (h_gcd5 : Nat.gcd b d = 1) (h_gcd6 : Nat.gcd c d = 1) : 
    a + b + c + d = 15698 :=
sorry

end sum_of_integers_l26_26794


namespace ellipse_tangent_and_fixed_point_l26_26384

-- Definitions based on given conditions
def ellipse (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (x^2 / a^2 + y^2 = 1)

def is_tangent_to (l : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) :=
  ∀ x y, l x y → circle x y

-- Define the specific ellipse and circle based on conditions
def ellipseC : ℝ → ℝ → Prop := ellipse (sqrt 2)
def circle : ℝ → ℝ → Prop := λ x y, x^2 + y^2 = 2 / 3

-- Main theorem statement
theorem ellipse_tangent_and_fixed_point :
  (∀ x y, (x = sqrt 2 * (1 - y)) → circle x y) →
  ∃ p : ℝ × ℝ, ∀ A B : ℝ × ℝ, 
    (ellipseC A.1 A.2) →
    (ellipseC B.1 B.2) →
    (A.2 = 1 ∨ B.2 = 1 → A.1 + B.1 = -2) →
    (A.2 ≠ 1 ∧ B.2 ≠ 1) →
    (∃ k1 k2 : ℝ, k1 + k2 = 2) →
    (∃ (L : ℝ × ℝ → Prop), L (A.1, A.2) ∧ L (B.1, B.2) ∧ L p) :=
begin
  sorry
end

end ellipse_tangent_and_fixed_point_l26_26384


namespace triangle_altitude_from_rectangle_l26_26003

theorem triangle_altitude_from_rectangle (a b : ℕ) (A : ℕ) (h : ℕ) (H1 : a = 7) (H2 : b = 21) (H3 : A = 147) (H4 : a * b = A) (H5 : 2 * A = h * b) : h = 14 :=
sorry

end triangle_altitude_from_rectangle_l26_26003


namespace comparison_l26_26688

noncomputable def a := Real.log 3000 / Real.log 9
noncomputable def b := Real.log 2023 / Real.log 4
noncomputable def c := (11 * Real.exp (0.01 * Real.log 1.001)) / 2

theorem comparison : a < b ∧ b < c :=
by
  sorry

end comparison_l26_26688


namespace apples_handout_l26_26005

theorem apples_handout {total_apples pies_needed pies_count handed_out : ℕ}
  (h1 : total_apples = 51)
  (h2 : pies_needed = 5)
  (h3 : pies_count = 2)
  (han : handed_out = total_apples - (pies_needed * pies_count)) :
  handed_out = 41 :=
by {
  sorry
}

end apples_handout_l26_26005


namespace larger_number_is_400_l26_26171

def problem_statement : Prop :=
  ∃ (a b hcf lcm num1 num2 : ℕ),
  hcf = 25 ∧
  a = 14 ∧
  b = 16 ∧
  lcm = hcf * a * b ∧
  num1 = hcf * a ∧
  num2 = hcf * b ∧
  num1 < num2 ∧
  num2 = 400

theorem larger_number_is_400 : problem_statement :=
  sorry

end larger_number_is_400_l26_26171


namespace ab_range_l26_26521

theorem ab_range (a b : ℝ) (h : a * b = a + b + 3) : a * b ≤ 1 ∨ a * b ≥ 9 := by
  sorry

end ab_range_l26_26521


namespace final_price_after_discounts_l26_26926

noncomputable def initial_price : ℝ := 9795.3216374269
noncomputable def discount_20 (p : ℝ) : ℝ := p * 0.80
noncomputable def discount_10 (p : ℝ) : ℝ := p * 0.90
noncomputable def discount_5 (p : ℝ) : ℝ := p * 0.95

theorem final_price_after_discounts : discount_5 (discount_10 (discount_20 initial_price)) = 6700 := 
by
  sorry

end final_price_after_discounts_l26_26926


namespace cos_largest_angle_value_l26_26245

noncomputable def cos_largest_angle (a b c : ℝ) (h1 : a = 2) (h2 : b = 3) (h3 : c = 4) : ℝ :=
  (a * a + b * b - c * c) / (2 * a * b)

theorem cos_largest_angle_value : cos_largest_angle 2 3 4 (by rfl) (by rfl) (by rfl) = -1 / 4 := 
sorry

end cos_largest_angle_value_l26_26245


namespace cole_runs_7_miles_l26_26455

theorem cole_runs_7_miles
  (xavier_miles : ℕ)
  (katie_miles : ℕ)
  (cole_miles : ℕ)
  (h1 : xavier_miles = 3 * katie_miles)
  (h2 : katie_miles = 4 * cole_miles)
  (h3 : xavier_miles = 84)
  (h4 : katie_miles = 28) :
  cole_miles = 7 := 
sorry

end cole_runs_7_miles_l26_26455


namespace book_selection_l26_26971

def num_books_in_genre (mystery fantasy biography : ℕ) : ℕ :=
  mystery + fantasy + biography

def num_combinations_two_diff_genres (mystery fantasy biography : ℕ) : ℕ :=
  if mystery = 4 ∧ fantasy = 4 ∧ biography = 4 then 48 else 0

theorem book_selection : 
  ∀ (mystery fantasy biography : ℕ),
  num_books_in_genre mystery fantasy biography = 12 →
  num_combinations_two_diff_genres mystery fantasy biography = 48 :=
by
  intros mystery fantasy biography h
  sorry

end book_selection_l26_26971


namespace probability_zhang_watches_entire_news_l26_26145

noncomputable def broadcast_time_start := 12 * 60 -- 12:00 in minutes
noncomputable def broadcast_time_end := 12 * 60 + 30 -- 12:30 in minutes
noncomputable def news_report_duration := 5 -- 5 minutes
noncomputable def zhang_on_tv_time := 12 * 60 + 20 -- 12:20 in minutes
noncomputable def favorable_time_start := zhang_on_tv_time
noncomputable def favorable_time_end := zhang_on_tv_time + news_report_duration -- 12:20 to 12:25

theorem probability_zhang_watches_entire_news : 
  let total_broadcast_time := broadcast_time_end - broadcast_time_start
  let favorable_time_span := favorable_time_end - favorable_time_start
  favorable_time_span / total_broadcast_time = 1 / 6 :=
by
  sorry

end probability_zhang_watches_entire_news_l26_26145


namespace intercepts_of_line_l26_26007

theorem intercepts_of_line :
  (∀ x y : ℝ, (x = 4 ∨ y = -3) → (x / 4 - y / 3 = 1)) ∧ (∀ x y : ℝ, (x / 4 = 1 ∧ y = 0) ∧ (x = 0 ∧ y / 3 = -1)) :=
by
  sorry

end intercepts_of_line_l26_26007


namespace books_borrowed_l26_26896

theorem books_borrowed (initial_books : ℕ) (additional_books : ℕ) (remaining_books : ℕ) : 
  initial_books = 300 → 
  additional_books = 10 * 5 → 
  remaining_books = 210 → 
  initial_books + additional_books - remaining_books = 140 :=
by
  intros h1 h2 h3
  rw [h1, h2]
  sorry

end books_borrowed_l26_26896


namespace min_value_of_quadratic_expression_l26_26377

theorem min_value_of_quadratic_expression (a b c : ℝ) (h : a + 2 * b + 3 * c = 6) : a^2 + 4 * b^2 + 9 * c^2 ≥ 12 :=
by
  sorry

end min_value_of_quadratic_expression_l26_26377


namespace trigonometric_identity_l26_26376

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l26_26376


namespace gold_coins_distribution_l26_26336

theorem gold_coins_distribution (x y : ℝ) (h₁ : x + y = 25) (h₂ : x ≠ y)
  (h₃ : (x^2 - y^2) = k * (x - y)) : k = 25 :=
sorry

end gold_coins_distribution_l26_26336


namespace projectile_height_reach_l26_26580

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end projectile_height_reach_l26_26580


namespace sum_of_values_of_n_l26_26605

theorem sum_of_values_of_n (n : ℚ) (h : |3 * n - 4| = 6) : 
  (n = 10 / 3 ∨ n = -2 / 3) → (10 / 3 + -2 / 3 = 8 / 3) :=
sorry

end sum_of_values_of_n_l26_26605


namespace part1_part2_l26_26810

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l26_26810


namespace rectangle_enclosing_ways_l26_26784

/-- Given five horizontal lines and five vertical lines, the total number of ways to choose four lines (two horizontal, two vertical) such that they form a rectangle is 100 --/
theorem rectangle_enclosing_ways : 
  let horizontal_lines := [1, 2, 3, 4, 5]
  let vertical_lines := [1, 2, 3, 4, 5]
  let ways_horizontal := Nat.choose 5 2
  let ways_vertical := Nat.choose 5 2
  ways_horizontal * ways_vertical = 100 := 
by
  sorry

end rectangle_enclosing_ways_l26_26784


namespace johnson_family_seating_problem_l26_26877

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l26_26877


namespace tangent_line_parabola_l26_26221

theorem tangent_line_parabola (a : ℝ) :
  (∀ x y : ℝ, y^2 = 32 * x → 4 * x + 3 * y + a = 0) → a = 18 :=
by
  sorry

end tangent_line_parabola_l26_26221


namespace part1_part2_l26_26827

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l26_26827


namespace min_value_of_quadratic_l26_26749

theorem min_value_of_quadratic :
  ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 2000 ≤ 3 * y^2 - 18 * y + 2000) ∧ (3 * x^2 - 18 * x + 2000 = 1973) :=
by
  sorry

end min_value_of_quadratic_l26_26749


namespace average_marks_of_all_students_l26_26311

/-
Consider two classes:
- The first class has 12 students with an average mark of 40.
- The second class has 28 students with an average mark of 60.

We are to prove that the average marks of all students from both classes combined is 54.
-/

theorem average_marks_of_all_students (s1 s2 : ℕ) (m1 m2 : ℤ)
  (h1 : s1 = 12) (h2 : m1 = 40) (h3 : s2 = 28) (h4 : m2 = 60) :
  (s1 * m1 + s2 * m2) / (s1 + s2) = 54 :=
by
  rw [h1, h2, h3, h4]
  sorry

end average_marks_of_all_students_l26_26311


namespace units_digit_seven_pow_ten_l26_26300

theorem units_digit_seven_pow_ten : ∃ u : ℕ, (7^10) % 10 = u ∧ u = 9 :=
by
  use 9
  sorry

end units_digit_seven_pow_ten_l26_26300


namespace find_some_number_l26_26439

theorem find_some_number (d : ℝ) (x : ℝ) (h1 : d = (0.889 * x) / 9.97) (h2 : d = 4.9) :
  x = 54.9 := by
  sorry

end find_some_number_l26_26439


namespace paving_cost_l26_26312

def length : ℝ := 5.5
def width : ℝ := 3.75
def rate : ℝ := 1000
def area : ℝ := length * width
def cost : ℝ := area * rate

theorem paving_cost :
  cost = 20625 := by sorry

end paving_cost_l26_26312


namespace find_particular_number_l26_26894

def particular_number (x : ℕ) : Prop :=
  (2 * (67 - (x / 23))) = 102

theorem find_particular_number : particular_number 2714 :=
by {
  sorry
}

end find_particular_number_l26_26894


namespace primes_sum_product_composite_l26_26858

theorem primes_sum_product_composite {p q r : ℕ} (hp : Nat.Prime p) (hq : Nat.Prime q) (hr : Nat.Prime r) (hdistinct_pq : p ≠ q) (hdistinct_pr : p ≠ r) (hdistinct_qr : q ≠ r) :
  ¬ Nat.Prime (p + q + r + p * q * r) :=
by
  sorry

end primes_sum_product_composite_l26_26858


namespace no_correlation_pair_D_l26_26928

-- Define the pairs of variables and their relationships
def pair_A : Prop := ∃ (fertilizer_applied grain_yield : ℝ), (fertilizer_applied ≠ 0 → grain_yield ≠ 0)
def pair_B : Prop := ∃ (review_time scores : ℝ), (review_time ≠ 0 → scores ≠ 0)
def pair_C : Prop := ∃ (advertising_expenses sales : ℝ), (advertising_expenses ≠ 0 → sales ≠ 0)
def pair_D : Prop := ∃ (books_sold revenue : ℕ), (revenue = books_sold * 5)

/-- Prove that pair D does not have a correlation in the context of the problem. --/
theorem no_correlation_pair_D : ¬pair_D :=
by
  sorry

end no_correlation_pair_D_l26_26928


namespace gcd_2024_2048_l26_26451

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l26_26451


namespace distance_squared_l26_26573

noncomputable def circumcircle_radius (R : ℝ) : Prop := sorry
noncomputable def excircle_radius (p : ℝ) : Prop := sorry
noncomputable def distance_between_centers (d : ℝ) (R : ℝ) (p : ℝ) : Prop := sorry

theorem distance_squared (R p d : ℝ) (h1 : circumcircle_radius R) (h2 : excircle_radius p) (h3 : distance_between_centers d R p) :
  d^2 = R^2 + 2 * R * p := sorry

end distance_squared_l26_26573


namespace fill_box_with_L_blocks_l26_26748

theorem fill_box_with_L_blocks (m n k : ℕ) 
  (hm : m > 1) (hn : n > 1) (hk : k > 1) (hk_div3 : k % 3 = 0) : 
  ∃ (fill : ℕ → ℕ → ℕ → Prop), fill m n k → True := 
by
  sorry

end fill_box_with_L_blocks_l26_26748


namespace tan_identity_l26_26368

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l26_26368


namespace seulgi_second_round_score_l26_26389

theorem seulgi_second_round_score
    (h_score1 : Nat) (h_score2 : Nat)
    (hj_score1 : Nat) (hj_score2 : Nat)
    (s_score1 : Nat) (required_second_score : Nat) :
    h_score1 = 23 →
    h_score2 = 28 →
    hj_score1 = 32 →
    hj_score2 = 17 →
    s_score1 = 27 →
    required_second_score = 25 →
    s_score1 + required_second_score > h_score1 + h_score2 ∧ 
    s_score1 + required_second_score > hj_score1 + hj_score2 :=
by
  intros
  sorry

end seulgi_second_round_score_l26_26389


namespace bonifac_distance_l26_26744

/-- Given the conditions provided regarding the paths of Pankrác, Servác, and Bonifác,
prove that the total distance Bonifác walked is 625 meters. -/
theorem bonifac_distance
  (path_Pankrac : ℕ)  -- distance of Pankráč's path in segments
  (meters_Pankrac : ℕ)  -- distance Pankráč walked in meters
  (path_Bonifac : ℕ)  -- distance of Bonifác's path in segments
  (meters_per_segment : ℚ)  -- meters per segment walked
  (Hp : path_Pankrac = 40)  -- Pankráč's path in segments
  (Hm : meters_Pankrac = 500)  -- Pankráč walked 500 meters
  (Hms : meters_per_segment = 500 / 40)  -- meters per segment
  (Hb : path_Bonifac = 50)  -- Bonifác's path in segments
  : path_Bonifac * meters_per_segment = 625 := sorry

end bonifac_distance_l26_26744


namespace max_value_of_f_l26_26204

def f (x : ℝ) : ℝ := 12 * x - 4 * x^2

theorem max_value_of_f : ∀ x : ℝ, f x ≤ 9 :=
by
  have h₁ : ∀ x : ℝ, 12 * x - 4 * x^2 ≤ 9
  { sorry }
  exact h₁

end max_value_of_f_l26_26204


namespace mul_18396_9999_l26_26030

theorem mul_18396_9999 :
  18396 * 9999 = 183941604 :=
by
  sorry

end mul_18396_9999_l26_26030


namespace eliminate_denominators_l26_26340

variable {x : ℝ}

theorem eliminate_denominators (h : 3 / (2 * x) = 1 / (x - 1)) :
  3 * x - 3 = 2 * x := 
by
  sorry

end eliminate_denominators_l26_26340


namespace no_sum_of_consecutive_integers_to_420_l26_26701

noncomputable def perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k * k = n

def sum_sequence (n a : ℕ) : ℕ :=
n * a + n * (n - 1) / 2

theorem no_sum_of_consecutive_integers_to_420 
  (h1 : 420 > 0)
  (h2 : ∀ (n a : ℕ), n ≥ 2 → sum_sequence n a = 420 → perfect_square a)
  (h3 : ∃ n a, n ≥ 2 ∧ sum_sequence n a = 420 ∧ perfect_square a) :
  false :=
by
  sorry

end no_sum_of_consecutive_integers_to_420_l26_26701


namespace second_integer_is_66_l26_26586

-- Define the conditions
def are_two_units_apart (a b c : ℤ) : Prop :=
  b = a + 2 ∧ c = a + 4

def sum_of_first_and_third_is_132 (a b c : ℤ) : Prop :=
  a + c = 132

-- State the theorem
theorem second_integer_is_66 (a b c : ℤ) 
  (H1 : are_two_units_apart a b c) 
  (H2 : sum_of_first_and_third_is_132 a b c) : b = 66 :=
by
  sorry -- Proof omitted

end second_integer_is_66_l26_26586


namespace value_of_7_star_3_l26_26109

def star (a b : ℕ) : ℕ := 4 * a + 3 * b - a * b

theorem value_of_7_star_3 : star 7 3 = 16 :=
by
  -- Proof would go here
  sorry

end value_of_7_star_3_l26_26109


namespace green_ball_count_l26_26262

theorem green_ball_count 
  (total_balls : ℕ)
  (n_red n_blue n_green : ℕ)
  (h_total : n_red + n_blue + n_green = 50)
  (h_red : ∀ (A : Finset ℕ), A.card = 34 -> ∃ a ∈ A, a < n_red)
  (h_blue : ∀ (A : Finset ℕ), A.card = 35 -> ∃ a ∈ A, a < n_blue)
  (h_green : ∀ (A : Finset ℕ), A.card = 36 -> ∃ a ∈ A, a < n_green)
  : n_green = 15 ∨ n_green = 16 ∨ n_green = 17 :=
by
  sorry

end green_ball_count_l26_26262


namespace discriminant_eq_complete_square_form_l26_26689

theorem discriminant_eq_complete_square_form (a b c t : ℝ) (h : a ≠ 0) (ht : a * t^2 + b * t + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * t + b)^2 := 
sorry

end discriminant_eq_complete_square_form_l26_26689


namespace frog_jump_vertical_side_prob_l26_26320

-- Definitions of the conditions
def square_side_prob {x y : ℕ} (p : ℕ × ℕ → ℚ) := 
  p (0, y) + p (4, y)

-- Main statement
theorem frog_jump_vertical_side_prob :
  ∀ (p : ℕ × ℕ → ℚ), 
    let start: ℕ × ℕ := (1, 2) in
    (∀ y, 0 ≤ y ∧ y ≤ 4 → p (0, y) = 1) → 
    (∀ y, 0 ≤ y ∧ y ≤ 4 → p (4, y) = 1) → 
    (∀ x, 0 ≤ x ∧ x ≤ 4 → p (x, 0) = 0) → 
    (∀ x, 0 ≤ x ∧ x ≤ 4 → p (x, 4) = 0) → 
    square_side_prob p = 5 / 8 :=
by
  intros
  sorry

end frog_jump_vertical_side_prob_l26_26320


namespace maximize_magnitude_l26_26366

theorem maximize_magnitude (a x y : ℝ) 
(h1 : 4 * x^2 + 4 * y^2 = -a^2 + 16 * a - 32)
(h2 : 2 * x * y = a) : a = 8 := 
sorry

end maximize_magnitude_l26_26366


namespace sum_of_all_possible_values_of_g10_l26_26941

noncomputable def g : ℕ → ℝ := sorry

axiom h1 : g 1 = 2
axiom h2 : ∀ m n : ℕ, m ≥ n → g (m + n) + g (m - n) = 3 * (g m + g n)
axiom h3 : g 0 = 0

theorem sum_of_all_possible_values_of_g10 : g 10 = 59028 :=
by
  sorry

end sum_of_all_possible_values_of_g10_l26_26941


namespace pencil_fraction_white_part_l26_26321

theorem pencil_fraction_white_part
  (L : ℝ )
  (H1 : L = 9.333333333333332)
  (H2 : (1 / 8) * L + (7 / 12 * 7 / 8) * (7 / 8) * L + W * (7 / 8) * L = L) :
  W = 5 / 12 :=
by
  sorry

end pencil_fraction_white_part_l26_26321


namespace find_common_difference_l26_26014

def arithmetic_sequence (S_n : ℕ → ℝ) (d : ℝ) :=
  ∀ n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)

theorem find_common_difference (S_n : ℕ → ℝ) (d : ℝ) (h : ∀n, S_n n = (n / 2) * (2 * (S_n 1 / 1) + (n - 1) * d)) 
    (h_condition : S_n 3 / 3 - S_n 2 / 2 = 1) :
  d = 2 :=
sorry

end find_common_difference_l26_26014


namespace probability_of_log2N_is_integer_and_N_is_even_l26_26183

-- Defining the range of N as a four-digit number in base four
def is_base4_four_digit (N : ℕ) : Prop := 64 ≤ N ∧ N ≤ 255

-- Defining the condition that log_2 N is an integer
def is_power_of_two (N : ℕ) : Prop := ∃ k : ℕ, N = 2^k

-- Defining the condition that N is even
def is_even (N : ℕ) : Prop := N % 2 = 0

-- Combining all conditions
def meets_conditions (N : ℕ) : Prop := is_base4_four_digit N ∧ is_power_of_two N ∧ is_even N

-- Total number of four-digit numbers in base four
def total_base4_four_digits : ℕ := 192

-- Set of N values that meet the conditions
def valid_N_values : Finset ℕ := {64, 128}

-- The probability calculation
def calculated_probability : ℚ := valid_N_values.card / total_base4_four_digits

-- The final proof statement
theorem probability_of_log2N_is_integer_and_N_is_even : calculated_probability = 1 / 96 :=
by
  -- Prove the equality here (matching the solution given)
  sorry

end probability_of_log2N_is_integer_and_N_is_even_l26_26183


namespace rectangle_ratio_expression_value_l26_26775

theorem rectangle_ratio_expression_value (l w : ℝ) (S : ℝ) (h1 : l / w = (2 * (l + w)) / (2 * l)) (h2 : S = w / l) :
  S ^ (S ^ (S^2 + 1/S) + 1/S) + 1/S = Real.sqrt 5 :=
by
  sorry

end rectangle_ratio_expression_value_l26_26775


namespace gcd_6Tn_nplus1_l26_26664

theorem gcd_6Tn_nplus1 (n : ℕ) (h : 0 < n) : gcd (3 * n * n + 3 * n) (n + 1) = 1 := by
  sorry

end gcd_6Tn_nplus1_l26_26664


namespace average_speed_correct_l26_26622

-- Define the conditions
def part1_distance : ℚ := 10
def part1_speed : ℚ := 12
def part2_distance : ℚ := 12
def part2_speed : ℚ := 10

-- Total distance
def total_distance : ℚ := part1_distance + part2_distance

-- Time computations
def time1 : ℚ := part1_distance / part1_speed
def time2 : ℚ := part2_distance / part2_speed
def total_time : ℚ := time1 + time2

-- Average speed computation
def average_speed : ℚ := total_distance / total_time

theorem average_speed_correct :
  average_speed = 660 / 61 := sorry

end average_speed_correct_l26_26622


namespace problem_solution_l26_26237

theorem problem_solution (a b c d : ℝ) (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : c = 6 * d) :
  (a + b * c) / (c + d * b) = (3 * (5 + 6 * d)) / (1 + 3 * d) :=
by
  sorry

end problem_solution_l26_26237


namespace mean_proportional_of_segments_l26_26840

theorem mean_proportional_of_segments (a b c : ℝ) (a_val : a = 2) (b_val : b = 6) :
  c = 2 * Real.sqrt 3 ↔ c*c = a * b := by
  sorry

end mean_proportional_of_segments_l26_26840


namespace prob_five_fish_eaten_expected_fish_eaten_value_l26_26861

noncomputable def prob_eats_at_least_five (n : ℕ) : ℚ :=
  if n = 7 then 19 / 35 else 0

noncomputable def expected_fish_eaten (n : ℕ) : ℚ :=
  if n = 7 then 5 else 0

theorem prob_five_fish_eaten {n : ℕ} (h : n = 7) :
  prob_eats_at_least_five n = 19 / 35 :=
begin
  rw h,
  exact rfl,
end

theorem expected_fish_eaten_value {n : ℕ} (h : n = 7) :
  expected_fish_eaten n = 5 :=
begin
  rw h,
  exact rfl,
end

end prob_five_fish_eaten_expected_fish_eaten_value_l26_26861


namespace relationship_among_a_b_c_l26_26557

noncomputable def a : ℝ := Real.log 4 / Real.log 5
noncomputable def b : ℝ := (Real.log 3 / Real.log 5)^2
noncomputable def c : ℝ := Real.log 5 / Real.log 4

theorem relationship_among_a_b_c : b < a ∧ a < c := by
  sorry

end relationship_among_a_b_c_l26_26557


namespace find_fathers_age_l26_26787

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l26_26787


namespace lateral_surface_area_of_cone_l26_26955

open Real

theorem lateral_surface_area_of_cone
  (SA : ℝ) (SB : ℝ)
  (cos_angle_SA_SB : ℝ) (angle_SA_base : ℝ)
  (area_SAB : ℝ) :
  cos_angle_SA_SB = 7 / 8 →
  angle_SA_base = π / 4 →
  area_SAB = 5 * sqrt 15 →
  SA = 4 * sqrt 5 →
  SB = SA →
  (1/2) * (sqrt 2 / 2 * SA) * (2 * π * SA) = 40 * sqrt 2 * π :=
sorry

end lateral_surface_area_of_cone_l26_26955


namespace true_statements_count_l26_26234

variables {Ω : Type*} [MeasureTheory.ProbabilityMeasure Ω]
variables {A B : Event Ω}

theorem true_statements_count (h : P (A ∩ B) = P A * P B) :
  (P (¬A ∩ B) = P (¬A) * P B) ∧ (P (A ∩ ¬B) = P A * P (¬B)) ∧ (P (¬A ∩ ¬B) = P (¬A) * P (¬B)) :=
by
  sorry

end true_statements_count_l26_26234


namespace line_plane_intersection_l26_26358

theorem line_plane_intersection :
  ∃ (x y z : ℝ), (∃ t : ℝ, x = -3 + 2 * t ∧ y = 1 + 3 * t ∧ z = 1 + 5 * t) ∧ (2 * x + 3 * y + 7 * z - 52 = 0) ∧ (x = -1) ∧ (y = 4) ∧ (z = 6) :=
sorry

end line_plane_intersection_l26_26358


namespace prob_three_red_cards_l26_26699

noncomputable def probability_of_three_red_cards : ℚ :=
  let total_ways := 52 * 51 * 50
  let ways_to_choose_red_cards := 26 * 25 * 24
  ways_to_choose_red_cards / total_ways

theorem prob_three_red_cards : probability_of_three_red_cards = 4 / 17 := sorry

end prob_three_red_cards_l26_26699


namespace sum_of_factors_l26_26305

theorem sum_of_factors (W F c : ℕ) (hW_gt_20: W > 20) (hF_gt_20: F > 20) (product_eq : W * F = 770) (sum_eq : W + F = c) :
  c = 57 :=
by sorry

end sum_of_factors_l26_26305


namespace find_original_number_l26_26526

theorem find_original_number (x : ℝ) 
  (h1 : x * 16 = 3408) 
  (h2 : 1.6 * 21.3 = 34.080000000000005) : 
  x = 213 :=
sorry

end find_original_number_l26_26526


namespace B_days_to_complete_work_l26_26034

theorem B_days_to_complete_work (A_days : ℕ) (efficiency_less_percent : ℕ) 
  (hA : A_days = 12) (hB_efficiency : efficiency_less_percent = 20) :
  let A_work_rate := 1 / 12
  let B_work_rate := (1 - (20 / 100)) * A_work_rate
  let B_days := 1 / B_work_rate
  B_days = 15 :=
by
  sorry

end B_days_to_complete_work_l26_26034


namespace part1_part2_l26_26814

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l26_26814


namespace Uki_earnings_l26_26599

theorem Uki_earnings (cupcake_price cookie_price biscuit_price : ℝ) 
                     (cupcake_count cookie_count biscuit_count : ℕ)
                     (days : ℕ) :
  cupcake_price = 1.50 →
  cookie_price = 2 →
  biscuit_price = 1 →
  cupcake_count = 20 →
  cookie_count = 10 →
  biscuit_count = 20 →
  days = 5 →
  (days : ℝ) * (cupcake_price * (cupcake_count : ℝ) + cookie_price * (cookie_count : ℝ) + biscuit_price * (biscuit_count : ℝ)) = 350 := 
by
  sorry

end Uki_earnings_l26_26599


namespace mutually_exclusive_any_two_l26_26953

variables (A B C : Prop)
axiom all_not_defective : A
axiom all_defective : B
axiom not_all_defective : C

theorem mutually_exclusive_any_two :
  (¬(A ∧ B)) ∧ (¬(A ∧ C)) ∧ (¬(B ∧ C)) :=
sorry

end mutually_exclusive_any_two_l26_26953


namespace initial_paper_count_l26_26405

theorem initial_paper_count (used left initial : ℕ) (h_used : used = 156) (h_left : left = 744) :
  initial = used + left :=
sorry

end initial_paper_count_l26_26405


namespace number_times_quarter_squared_eq_four_cubed_l26_26033

theorem number_times_quarter_squared_eq_four_cubed : 
  ∃ (number : ℕ), number * (1 / 4 : ℚ) ^ 2 = (4 : ℚ) ^ 3 ∧ number = 1024 :=
by 
  use 1024
  sorry

end number_times_quarter_squared_eq_four_cubed_l26_26033


namespace emily_extra_distance_five_days_l26_26159

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end emily_extra_distance_five_days_l26_26159


namespace find_rope_costs_l26_26062

theorem find_rope_costs (x y : ℕ) (h1 : 10 * x + 5 * y = 175) (h2 : 15 * x + 10 * y = 300) : x = 10 ∧ y = 15 :=
    sorry

end find_rope_costs_l26_26062


namespace remainder_of_sum_div_11_is_9_l26_26361

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end remainder_of_sum_div_11_is_9_l26_26361


namespace largest_negative_integer_is_neg_one_l26_26027

def is_negative_integer (n : Int) : Prop := n < 0

def is_largest_negative_integer (n : Int) : Prop := 
  is_negative_integer n ∧ ∀ m : Int, is_negative_integer m → m ≤ n

theorem largest_negative_integer_is_neg_one : 
  is_largest_negative_integer (-1) := by
  sorry

end largest_negative_integer_is_neg_one_l26_26027


namespace function_machine_output_is_38_l26_26988

def function_machine (input : ℕ) : ℕ :=
  let multiplied := input * 3
  if multiplied > 40 then
    multiplied - 7
  else
    multiplied + 10

theorem function_machine_output_is_38 :
  function_machine 15 = 38 :=
by
   sorry

end function_machine_output_is_38_l26_26988


namespace james_total_distance_l26_26406

-- Define the conditions
def speed_part1 : ℝ := 30  -- mph
def time_part1 : ℝ := 0.5  -- hours
def speed_part2 : ℝ := 2 * speed_part1  -- 2 * 30 mph
def time_part2 : ℝ := 2 * time_part1  -- 2 * 0.5 hours

-- Compute distances
def distance_part1 : ℝ := speed_part1 * time_part1
def distance_part2 : ℝ := speed_part2 * time_part2

-- Total distance
def total_distance : ℝ := distance_part1 + distance_part2

-- The theorem to prove
theorem james_total_distance :
  total_distance = 75 := 
sorry

end james_total_distance_l26_26406


namespace star_example_l26_26651

def star (a b : ℤ) : ℤ := a * b^3 - 2 * b + 2

theorem star_example : star 2 3 = 50 := by
  sorry

end star_example_l26_26651


namespace last_child_loses_l26_26910

-- Definitions corresponding to conditions
def num_children := 11
def child_sequence := List.range' 1 num_children
def valid_two_digit_numbers := 90
def invalid_digit_sum_6 := 6
def invalid_digit_sum_9 := 9
def valid_numbers := valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9
def complete_cycles := valid_numbers / num_children
def remaining_numbers := valid_numbers % num_children

-- Statement to be proven
theorem last_child_loses (h1 : num_children = 11)
                         (h2 : valid_two_digit_numbers = 90)
                         (h3 : invalid_digit_sum_6 = 6)
                         (h4 : invalid_digit_sum_9 = 9)
                         (h5 : valid_numbers = valid_two_digit_numbers - invalid_digit_sum_6 - invalid_digit_sum_9)
                         (h6 : remaining_numbers = valid_numbers % num_children) :
  (remaining_numbers = 9) ∧ (num_children - remaining_numbers = 2) :=
by
  sorry

end last_child_loses_l26_26910


namespace arithmetic_sequence_product_l26_26434

theorem arithmetic_sequence_product 
  (a d : ℤ)
  (h1 : a + 6 * d = 20)
  (h2 : d = 2) : 
  a * (a + d) * (a + 2 * d) = 960 := 
by
  -- proof goes here
  sorry

end arithmetic_sequence_product_l26_26434


namespace initial_participants_l26_26119

theorem initial_participants (p : ℕ) (h1 : 0.6 * p = 0.6 * (p : ℝ)) (h2 : ∀ (n : ℕ), n = 4 * m → 30 = (2 / 5) * n * (1 / 4)) :
  p = 300 :=
by sorry

end initial_participants_l26_26119


namespace propositions_correct_l26_26251

-- Problem statement
theorem propositions_correct (S : ℕ → ℝ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h1 : ∀ n, S(n + 1) = S n + a(n + 1)) (h2 : ∀ n, S n = a n * a n + b n) :
  (¬ ∀ n, a (n + 1) = a n) ∧
  (¬ (∀ n, S n = 1 - (-1)^n → ∀ n, S(n + 1) - S n = 2 * (-1)^n → ∀ n, a(n + 1) / a n = -1)) ∧
  (∀ n, S n = 1 - (-1)^n → ∀ n, S(n + 1) - S n = 2 * (-1)^n → ∀ n, a(n + 1) / a n = -1) ∧
  (∀ n, ∃ d ∈ ℝ, S(n + 1) - S n = d ∧ S(2 * n) - S n = d ∧ S(3 * n) - S(2 * n) = d) :=
begin
  -- Proof not included as per instructions
  sorry
end

end propositions_correct_l26_26251


namespace part1_part2_l26_26830

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l26_26830


namespace ellipse_eccentricity_l26_26057

open Complex

-- Define the roots of the polynomial equation.
def roots : List Complex := [2, -1 / 2 + (sqrt 7 : ℂ) / 2 * I, -1 / 2 - (sqrt 7 : ℂ) / 2 * I, -5 / 2 + (1 / 2) * I, -5 / 2 - (1 / 2) * I]

-- Define the points in the complex plane derived from the roots.
def points : List (ℝ × ℝ) := [ (2, 0), (-0.5, (real.sqrt 7) / 2), (-0.5, -(real.sqrt 7) / 2), (-2.5, 0.5), (-2.5, -0.5) ]

-- Define the eccentricity of the ellipse passing through these points.
def eccentricity_of_ellipse : ℝ :=
  let h := 0 in  -- Assume the center is (h, 0)
  let a :=  // semi-major axis length
  let b :=  // semi-minor axis length
  real.sqrt ((a ^ 2 - b ^ 2) / a ^ 2)

theorem ellipse_eccentricity (e : ℝ) :
  ∀ (a b : ℝ) (c : ℝ), e = c / a → c^2 = a^2 - b^2 → list_all points_on_ellipse points a b h →
  e = 1 / real.sqrt 5 :=
sorry

end ellipse_eccentricity_l26_26057


namespace inequality_holds_l26_26088

-- Given conditions
variables {a b x y : ℝ}
variables (pos_a : 0 < a) (pos_b : 0 < b) (pos_x : 0 < x) (pos_y : 0 < y)
variable (h : a + b = 1)

-- Goal/Question
theorem inequality_holds : (a * x + b * y) * (b * x + a * y) ≥ x * y :=
by sorry

end inequality_holds_l26_26088


namespace purple_valley_skirts_l26_26721

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l26_26721


namespace solve_floor_equation_l26_26868

theorem solve_floor_equation (x : ℝ) (hx : (∃ (y : ℤ), (x^3 - 40 * (y : ℝ) - 78 = 0) ∧ (y : ℝ) ≤ x ∧ x < (y + 1 : ℝ))) :
  x = -5.45 ∨ x = -4.96 ∨ x = -1.26 ∨ x = 6.83 ∨ x = 7.10 :=
by sorry

end solve_floor_equation_l26_26868


namespace woman_year_of_birth_l26_26188

def year_of_birth (x : ℕ) : ℕ := x^2 - x

theorem woman_year_of_birth : ∃ (x : ℕ), 1850 ≤ year_of_birth x ∧ year_of_birth x < 1900 ∧ year_of_birth x = 1892 :=
by
  sorry

end woman_year_of_birth_l26_26188


namespace quadratic_inequality_range_a_l26_26095

theorem quadratic_inequality_range_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - a * x + 2 * a > 0) ↔ (0 < a ∧ a < 8) :=
by
  sorry

end quadratic_inequality_range_a_l26_26095


namespace prob1_prob2_prob3_prob4_l26_26484

theorem prob1 : (-20) + (-14) - (-18) - 13 = -29 := sorry

theorem prob2 : (-24) * (-1/2 + 3/4 - 1/3) = 2 := sorry

theorem prob3 : (- (49 + 24/25)) * 10 = -499.6 := sorry

theorem prob4 :
  -3^2 + ((-1/3) * (-3) - 8/5 / 2^2) = -8 - 2/5 := sorry

end prob1_prob2_prob3_prob4_l26_26484


namespace triangle_side_length_l26_26853

/-
  Given a triangle ABC with sides |AB| = c, |AC| = b, and centroid G, incenter I,
  if GI is perpendicular to BC, then we need to prove that |BC| = (b+c)/2.
-/
variable {A B C G I : Type}
variable {AB AC BC : ℝ} -- Lengths of the sides
variable {b c : ℝ} -- Given lengths
variable {G_centroid : IsCentroid A B C G} -- G is the centroid of triangle ABC
variable {I_incenter : IsIncenter A B C I} -- I is the incenter of triangle ABC
variable {G_perp_BC : IsPerpendicular G I BC} -- G I ⊥ BC

theorem triangle_side_length (h1 : |AB| = c) (h2 : |AC| = b) :
  |BC| = (b + c) / 2 := 
sorry

end triangle_side_length_l26_26853


namespace pure_imaginary_solutions_l26_26205

theorem pure_imaginary_solutions:
  ∀ (x : ℂ), (x.im ≠ 0 ∧ x.re = 0) → (x ^ 4 - 5 * x ^ 3 + 10 * x ^ 2 - 50 * x - 75 = 0)
         → (x = Complex.I * Real.sqrt 10 ∨ x = -Complex.I * Real.sqrt 10) :=
by
  sorry

end pure_imaginary_solutions_l26_26205


namespace vector_projection_condition_l26_26199

noncomputable def line_l (t : ℝ) : ℝ × ℝ := (2 + 3 * t, 3 + 2 * t)
noncomputable def line_m (s : ℝ) : ℝ × ℝ := (4 + 2 * s, 5 + 3 * s)

def is_perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

theorem vector_projection_condition 
  (t s : ℝ)
  (C : ℝ × ℝ := line_l t)
  (D : ℝ × ℝ := line_m s)
  (Q : ℝ × ℝ)
  (hQ : is_perpendicular (Q.1 - C.1, Q.2 - C.2) (2, 3))
  (v1 v2 : ℝ)
  (hv_sum : v1 + v2 = 3)
  (hv_def : ∃ k : ℝ, v1 = 3 * k ∧ v2 = -2 * k)
  : (v1, v2) = (9, -6) := 
sorry

end vector_projection_condition_l26_26199


namespace optimal_rental_decision_optimal_purchase_decision_l26_26625

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end optimal_rental_decision_optimal_purchase_decision_l26_26625


namespace complement_U_A_union_B_is_1_and_9_l26_26685

-- Define the universe set U
def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

-- Define set A according to the given condition
def is_elem_of_A (x : ℕ) : Prop := 2 < x ∧ x ≤ 6
def A : Set ℕ := {x | is_elem_of_A x}

-- Define set B explicitly
def B : Set ℕ := {0, 2, 4, 5, 7, 8}

-- Define the union A ∪ B
def A_union_B : Set ℕ := A ∪ B

-- Define the complement of A ∪ B in U
def complement_U_A_union_B : Set ℕ := {x ∈ U | x ∉ A_union_B}

-- State the theorem
theorem complement_U_A_union_B_is_1_and_9 :
  complement_U_A_union_B = {1, 9} :=
by
  sorry

end complement_U_A_union_B_is_1_and_9_l26_26685


namespace find_n_l26_26296

variable (x n : ℕ)
variable (y : ℕ) {h1 : y = 24}

theorem find_n
  (h1 : y = 24) 
  (h2 : x / y = 1 / 4) 
  (h3 : (x + n) / y = 1 / 2) : 
  n = 6 := 
sorry

end find_n_l26_26296


namespace johnson_family_seating_l26_26872

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l26_26872


namespace surface_area_of_solid_block_l26_26352

theorem surface_area_of_solid_block :
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  top_bottom_area + front_back_area + left_right_area = 66 :=
by
  let unit_cube_surface_area := 6
  let top_bottom_area := 2 * (3 * 5)
  let front_back_area := 2 * (3 * 5)
  let left_right_area := 2 * (3 * 1)
  sorry

end surface_area_of_solid_block_l26_26352


namespace olivia_wallet_l26_26591

theorem olivia_wallet (initial_amount spent_amount remaining_amount : ℕ)
  (h1 : initial_amount = 78)
  (h2 : spent_amount = 15):
  remaining_amount = initial_amount - spent_amount →
  remaining_amount = 63 :=
sorry

end olivia_wallet_l26_26591


namespace num_ways_to_select_officers_l26_26472

def ways_to_select_five_officers (n : ℕ) (k : ℕ) : ℕ :=
  (List.range' (n - k + 1) k).foldl (λ acc x => acc * x) 1

theorem num_ways_to_select_officers :
  ways_to_select_five_officers 12 5 = 95040 :=
by
  -- By definition of ways_to_select_five_officers, this is equivalent to 12 * 11 * 10 * 9 * 8.
  sorry

end num_ways_to_select_officers_l26_26472


namespace separation_of_homologous_chromosomes_only_in_meiosis_l26_26738

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

end separation_of_homologous_chromosomes_only_in_meiosis_l26_26738


namespace present_age_of_B_l26_26170

theorem present_age_of_B (A B : ℕ) (h1 : A + 20 = 2 * (B - 20)) (h2 : A = B + 10) : B = 70 :=
by
  sorry

end present_age_of_B_l26_26170


namespace binom_sum_mod_1000_l26_26051

theorem binom_sum_mod_1000 : 
  (∑ i in (finset.range 2012).filter (λ i, i % 4 = 0), nat.choose 2011 i) % 1000 = 15 :=
sorry

end binom_sum_mod_1000_l26_26051


namespace solve_for_x_l26_26269

theorem solve_for_x : ∀ (x : ℝ), (x ≠ 3) → ((x - 3) / (x + 2) + (3 * x - 6) / (x - 3) = 2) → x = 1 / 2 := 
by
  intros x hx h
  sorry

end solve_for_x_l26_26269


namespace arithmetic_sequence_sum_ratio_l26_26085

theorem arithmetic_sequence_sum_ratio 
  (a_n : ℕ → ℝ) 
  (S_n : ℕ → ℝ) 
  (a : ℝ) 
  (d : ℝ) 
  (n : ℕ) 
  (a_n_def : ∀ n, a_n n = a + (n - 1) * d) 
  (S_n_def : ∀ n, S_n n = n * (2 * a + (n - 1) * d) / 2) 
  (h : 3 * (a + 4 * d) = 5 * (a + 2 * d)) : 
  S_n 5 / S_n 3 = 5 / 2 := 
by 
  sorry

end arithmetic_sequence_sum_ratio_l26_26085


namespace burger_cost_l26_26706

theorem burger_cost {B : ℝ} (sandwich_cost : ℝ) (smoothies_cost : ℝ) (total_cost : ℝ)
  (H1 : sandwich_cost = 4)
  (H2 : smoothies_cost = 8)
  (H3 : total_cost = 17)
  (H4 : B + sandwich_cost + smoothies_cost = total_cost) :
  B = 5 :=
sorry

end burger_cost_l26_26706


namespace nine_fact_div_four_fact_eq_15120_l26_26802

theorem nine_fact_div_four_fact_eq_15120 :
  (362880 / 24) = 15120 :=
by
  sorry

end nine_fact_div_four_fact_eq_15120_l26_26802


namespace find_fathers_age_l26_26785

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l26_26785


namespace pull_ups_per_time_l26_26418

theorem pull_ups_per_time (pull_ups_week : ℕ) (times_day : ℕ) (days_week : ℕ)
  (h1 : pull_ups_week = 70) (h2 : times_day = 5) (h3 : days_week = 7) :
  pull_ups_week / (times_day * days_week) = 2 := by
  sorry

end pull_ups_per_time_l26_26418


namespace physics_marks_l26_26639

variables (P C M : ℕ)

theorem physics_marks (h1 : P + C + M = 195)
                      (h2 : P + M = 180)
                      (h3 : P + C = 140) : P = 125 :=
by
  sorry

end physics_marks_l26_26639


namespace rhombus_diagonal_l26_26147

theorem rhombus_diagonal (d1 d2 : ℝ) (area : ℝ) 
  (h_d1 : d1 = 70) 
  (h_area : area = 5600): 
  (area = (d1 * d2) / 2) → d2 = 160 :=
by
  sorry

end rhombus_diagonal_l26_26147


namespace fill_tank_time_l26_26306

theorem fill_tank_time :
  ∀ (rate_fill rate_empty : ℝ), 
    rate_fill = 1 / 25 → 
    rate_empty = 1 / 50 → 
    (1/2) / (rate_fill - rate_empty) = 25 :=
by
  intros rate_fill rate_empty h_fill h_empty
  sorry

end fill_tank_time_l26_26306


namespace mean_difference_l26_26672

variable (a1 a2 a3 a4 a5 a6 A : ℝ)

-- Arithmetic mean of six numbers is A
axiom mean_six_numbers : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

-- Arithmetic mean of the first four numbers is A + 10
axiom mean_first_four : (a1 + a2 + a3 + a4) / 4 = A + 10

-- Arithmetic mean of the last four numbers is A - 7
axiom mean_last_four : (a3 + a4 + a5 + a6) / 4 = A - 7

-- Prove the arithmetic mean of the first, second, fifth, and sixth numbers differs from A by 3
theorem mean_difference :
  (a1 + a2 + a5 + a6) / 4 = A - 3 := 
sorry

end mean_difference_l26_26672


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l26_26456

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l26_26456


namespace wendy_chocolates_l26_26024

theorem wendy_chocolates (h : ℕ) : 
  let chocolates_per_4_hours := 1152
  let chocolates_per_hour := chocolates_per_4_hours / 4
  (chocolates_per_hour * h) = 288 * h :=
by
  sorry

end wendy_chocolates_l26_26024


namespace total_population_after_births_l26_26624

theorem total_population_after_births:
  let initial_population := 300000
  let immigrants := 50000
  let emigrants := 30000
  let pregnancies_fraction := 1 / 8
  let twins_fraction := 1 / 4
  let net_population := initial_population + immigrants - emigrants
  let pregnancies := net_population * pregnancies_fraction
  let twin_pregnancies := pregnancies * twins_fraction
  let twin_children := twin_pregnancies * 2
  let single_births := pregnancies - twin_pregnancies
  net_population + single_births + twin_children = 370000 := by
  sorry

end total_population_after_births_l26_26624


namespace divides_n3_minus_7n_l26_26426

theorem divides_n3_minus_7n (n : ℕ) : 6 ∣ n^3 - 7 * n := 
sorry

end divides_n3_minus_7n_l26_26426


namespace binomial_expansion_limit_l26_26682

open Real BigOperators

theorem binomial_expansion_limit (a : ℕ → ℕ → ℤ) (T R : ℕ → ℤ) :
  (∀ n, (3 * (x : ℝ) - 1)^(2*n) = (∑ i in range (2*n + 1), a n i * x^i)) →
  (∀ n, T n = ∑ i in range (n + 1), a n (2 * i)) →
  (∀ n, R n = ∑ i in range (n + 1), a n (2 * i + 1)) →
  (∀ n, ∀ x, (3 * x - 1)^(2 * n) = 2^(2 * n)) →
  (∀ n, ∀ x, (3 * (-x) - 1)^(2 * n) = 4^(2 * n)) →
  tendsto (λ n, (T n)/(R n) : ℕ → ℝ) at_top (𝓝 (-1)) :=
  by
  sorry

end binomial_expansion_limit_l26_26682


namespace felicity_gasoline_usage_l26_26072

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

end felicity_gasoline_usage_l26_26072


namespace find_n_equiv_l26_26357

theorem find_n_equiv :
  ∃ (n : ℕ), 3 ≤ n ∧ n ≤ 9 ∧ n ≡ 12345 [MOD 6] ∧ (n = 3 ∨ n = 9) :=
by
  sorry

end find_n_equiv_l26_26357


namespace tank_fill_time_l26_26139

theorem tank_fill_time :
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  time_with_both_pipes + additional_time_A = 70 :=
by
  let fill_rate_A := 1 / 8
  let empty_rate_B := 1 / 24
  let combined_rate := fill_rate_A - empty_rate_B
  let time_with_both_pipes := 66
  let partial_fill := time_with_both_pipes * combined_rate
  let remaining_fill := 1 - (partial_fill % 1)
  let additional_time_A := remaining_fill / fill_rate_A
  have h : time_with_both_pipes + additional_time_A = 70 := sorry
  exact h

end tank_fill_time_l26_26139


namespace salon_customers_l26_26325

theorem salon_customers (C : ℕ) (H : C * 2 + 5 = 33) : C = 14 :=
by {
  sorry
}

end salon_customers_l26_26325


namespace contrapositive_even_contrapositive_not_even_l26_26275

theorem contrapositive_even (x y : ℤ) : 
  (∃ a b : ℤ, x = 2*a ∧ y = 2*b)  → (∃ c : ℤ, x + y = 2*c) :=
sorry

theorem contrapositive_not_even (x y : ℤ) :
  (¬ ∃ c : ℤ, x + y = 2*c) → (¬ ∃ a b : ℤ, x = 2*a ∧ y = 2*b) :=
sorry

end contrapositive_even_contrapositive_not_even_l26_26275


namespace find_k_l26_26529

theorem find_k (k : ℝ) : (∃ x : ℝ, x - 2 = 0 ∧ 1 - (x + k) / 3 = 0) → k = 1 :=
by
  sorry

end find_k_l26_26529


namespace swiss_slices_correct_l26_26862

-- Define the variables and conditions
variables (S : ℕ) (cheddar_slices : ℕ := 12) (total_cheddar_slices : ℕ := 84) (total_swiss_slices : ℕ := 84)

-- Define the statement to be proved
theorem swiss_slices_correct (H : total_cheddar_slices = total_swiss_slices) : S = 12 :=
sorry

end swiss_slices_correct_l26_26862


namespace total_cards_across_decks_l26_26409

-- Conditions
def DeckA_cards : ℕ := 52
def DeckB_cards : ℕ := 40
def DeckC_cards : ℕ := 50
def DeckD_cards : ℕ := 48

-- Question as a statement
theorem total_cards_across_decks : (DeckA_cards + DeckB_cards + DeckC_cards + DeckD_cards = 190) := by
  sorry

end total_cards_across_decks_l26_26409


namespace triangle_is_isosceles_l26_26799

variable {α β γ : ℝ} (quadrilateral_angles : List ℝ)

-- Conditions from the problem
axiom triangle_angle_sum : α + β + γ = 180
axiom quadrilateral_angle_sum : quadrilateral_angles.sum = 360
axiom quadrilateral_angle_conditions : ∀ (a b : ℝ), a ∈ [α, β, γ] → b ∈ [α, β, γ] → a ≠ b → (a + b ∈ quadrilateral_angles)

-- Proof statement
theorem triangle_is_isosceles : (α = β) ∨ (β = γ) ∨ (γ = α) := 
  sorry

end triangle_is_isosceles_l26_26799


namespace min_races_required_to_determine_top_3_horses_l26_26931

def maxHorsesPerRace := 6
def totalHorses := 30
def possibleConditions := "track conditions and layouts change for each race"

noncomputable def minRacesToDetermineTop3 : Nat :=
  7

-- Problem Statement: Prove that given the conditions on track and race layout changes,
-- the minimum number of races needed to confidently determine the top 3 fastest horses is 7.
theorem min_races_required_to_determine_top_3_horses 
  (maxHorsesPerRace : Nat := 6) 
  (totalHorses : Nat := 30)
  (possibleConditions : String := "track conditions and layouts change for each race") :
  minRacesToDetermineTop3 = 7 :=
  sorry

end min_races_required_to_determine_top_3_horses_l26_26931


namespace coin_flips_prob_l26_26458

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l26_26458


namespace cost_difference_zero_l26_26763

theorem cost_difference_zero
  (A O X : ℝ)
  (h1 : 3 * A + 7 * O = 4.56)
  (h2 : A + O = 0.26)
  (h3 : O = A + X) :
  X = 0 := 
sorry

end cost_difference_zero_l26_26763


namespace second_year_associates_l26_26534

theorem second_year_associates (total_associates : ℕ) (not_first_year : ℕ) (more_than_two_years : ℕ) 
  (h1 : not_first_year = 60 * total_associates / 100) 
  (h2 : more_than_two_years = 30 * total_associates / 100) :
  not_first_year - more_than_two_years = 30 * total_associates / 100 :=
by
  sorry

end second_year_associates_l26_26534


namespace sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l26_26367

open Real

noncomputable def problem_conditions (α β : ℝ) : Prop :=
  0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧
  cos α = 3/5 ∧ cos (β + α) = 5/13

theorem sin_beta_value 
  {α β : ℝ} (h : problem_conditions α β) : 
  sin β = 16 / 65 :=
sorry

theorem sin2alpha_over_cos2alpha_plus_cos2alpha_value
  {α β : ℝ} (h : problem_conditions α β) : 
  (sin (2 * α)) / (cos α^2 + cos (2 * α)) = 12 :=
sorry

end sin_beta_value_sin2alpha_over_cos2alpha_plus_cos2alpha_value_l26_26367


namespace cost_of_each_math_book_l26_26165

-- Define the given conditions
def total_books : ℕ := 90
def math_books : ℕ := 53
def history_books : ℕ := total_books - math_books
def history_book_cost : ℕ := 5
def total_price : ℕ := 397

-- The required theorem
theorem cost_of_each_math_book (M : ℕ) (H : 53 * M + history_books * history_book_cost = total_price) : M = 4 :=
by
  sorry

end cost_of_each_math_book_l26_26165


namespace ratio_of_areas_l26_26770

-- Definitions of conditions
def side_length (s : ℝ) : Prop := s > 0
def original_area (A s : ℝ) : Prop := A = s^2

-- Definition of the new area after folding
def new_area (B A s : ℝ) : Prop := B = (7/8) * s^2

-- The proof statement to show the ratio B/A is 7/8
theorem ratio_of_areas (s A B : ℝ) (h_side : side_length s) (h_area : original_area A s) (h_B : new_area B A s) : 
  B / A = 7 / 8 := 
by 
  sorry

end ratio_of_areas_l26_26770


namespace anna_plants_needed_l26_26195

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end anna_plants_needed_l26_26195


namespace geometric_sequence_increasing_condition_l26_26677

theorem geometric_sequence_increasing_condition (a₁ a₂ a₄ : ℝ) (q : ℝ) (n : ℕ) (a : ℕ → ℝ):
  (∀ n, a n = a₁ * q^n) →
  (a₁ < a₂ ∧ a₂ < a₄) → 
  ¬ (∀ n, a n < a (n + 1)) → 
  (a₁ < a₂ ∧ a₂ < a₄) ∧ ¬ (∀ n, a n < a (n + 1)) :=
sorry

end geometric_sequence_increasing_condition_l26_26677


namespace total_pens_left_l26_26631

def initial_blue_pens := 9
def removed_blue_pens := 4
def initial_black_pens := 21
def removed_black_pens := 7
def initial_red_pens := 6

def remaining_blue_pens := initial_blue_pens - removed_blue_pens
def remaining_black_pens := initial_black_pens - removed_black_pens
def remaining_red_pens := initial_red_pens

def total_remaining_pens := remaining_blue_pens + remaining_black_pens + remaining_red_pens

theorem total_pens_left : total_remaining_pens = 25 :=
by
  -- Proof will be provided here
  sorry

end total_pens_left_l26_26631


namespace point_inside_circle_range_of_a_l26_26767

/- 
  Define the circle and the point P. 
  We would show that ensuring the point lies inside the circle implies |a| < 1/13.
-/

theorem point_inside_circle_range_of_a (a : ℝ) : 
  ((5 * a + 1 - 1) ^ 2 + (12 * a) ^ 2 < 1) -> |a| < 1 / 13 := 
by 
  sorry

end point_inside_circle_range_of_a_l26_26767


namespace expr_1989_eval_expr_1990_eval_l26_26000

def nestedExpr : ℕ → ℤ
| 0     => 0
| (n+1) => -1 - (nestedExpr n)

-- Conditions translated into Lean definitions:
def expr_1989 := nestedExpr 1989
def expr_1990 := nestedExpr 1990

-- The proof statements:
theorem expr_1989_eval : expr_1989 = -1 := sorry
theorem expr_1990_eval : expr_1990 = 0 := sorry

end expr_1989_eval_expr_1990_eval_l26_26000


namespace sqrt_rational_rational_l26_26246

theorem sqrt_rational_rational 
  (a b : ℚ) 
  (h : ∃ r : ℚ, r = (a : ℝ).sqrt + (b : ℝ).sqrt) : 
  (∃ p : ℚ, p = (a : ℝ).sqrt) ∧ (∃ q : ℚ, q = (b : ℝ).sqrt) := 
sorry

end sqrt_rational_rational_l26_26246


namespace find_quadruple_l26_26353

theorem find_quadruple :
  ∃ (a b c d : ℕ), 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 
  a^3 + b^4 + c^5 = d^11 ∧ a * b * c < 10^5 :=
sorry

end find_quadruple_l26_26353


namespace words_in_power_form_l26_26035

theorem words_in_power_form 
  (X Y : List Char) 
  (h : simplified_equation X Y) : 
  ∃ (Z : Char) (k l : ℕ), X = List.repeat Z k ∧ Y = List.repeat Z l :=
sorry

end words_in_power_form_l26_26035


namespace smallest_three_digit_times_largest_single_digit_l26_26740

theorem smallest_three_digit_times_largest_single_digit :
  let x := 100
  let y := 9
  ∃ z : ℕ, z = x * y ∧ 100 ≤ z ∧ z < 1000 :=
by
  let x := 100
  let y := 9
  use x * y
  sorry

end smallest_three_digit_times_largest_single_digit_l26_26740


namespace max_f_on_interval_l26_26947

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 2 + (Real.sqrt 3) * Real.sin x * Real.cos x

theorem max_f_on_interval : 
  ∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (Real.pi / 2) ∧ ∀ y ∈ Set.Icc (Real.pi / 4) (Real.pi / 2), f y ≤ f x ∧ f x = 3 / 2 :=
  sorry

end max_f_on_interval_l26_26947


namespace tangent_lines_through_point_l26_26963

noncomputable def f (x : ℝ) : ℝ :=
  (x - 1) * (x^2 + 1) + 1

def f' (x : ℝ) : ℝ :=
  deriv f x

def tangent_line_eqn_1 (x y : ℝ) : Prop :=
  2 * x - y - 1 = 0

def tangent_line_eqn_2 (x y : ℝ) : Prop :=
  y = x

theorem tangent_lines_through_point :
  ∃ (line_eqn : ℝ → ℝ → Prop), line_eqn = tangent_line_eqn_1 ∨ line_eqn = tangent_line_eqn_2 := by
  sorry

end tangent_lines_through_point_l26_26963


namespace minimum_value_l26_26670

variable (m n x y : ℝ)

theorem minimum_value (h1 : m^2 + n^2 = 1) (h2 : x^2 + y^2 = 4) : 
  ∃ (min_val : ℝ), min_val = -2 ∧ ∀ (my_nx : ℝ), my_nx = my + nx → my_nx ≥ min_val :=
by
  sorry

end minimum_value_l26_26670


namespace no_natural_numbers_condition_l26_26867

theorem no_natural_numbers_condition :
  ¬ ∃ (a : Fin 2018 → ℕ), ∀ i : Fin 2018,
    ∃ k : ℕ, (a i) ^ 2018 + a ((i + 1) % 2018) = 5 ^ k :=
by sorry

end no_natural_numbers_condition_l26_26867


namespace factorization_l26_26659

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := 
by
  sorry

end factorization_l26_26659


namespace range_of_a_l26_26416

noncomputable def f (x : ℝ) : ℝ := (x + 1) * Real.log (x + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ≥ 0, f x ≥ a * x) ↔ (a ≤ 1) :=
by
  sorry

end range_of_a_l26_26416


namespace factorize_polynomial_value_of_x_cubed_l26_26912

-- Problem 1: Factorization
theorem factorize_polynomial (x : ℝ) : 42 * x^2 - 33 * x + 6 = 3 * (2 * x - 1) * (7 * x - 2) :=
sorry

-- Problem 2: Given condition and proof of x^3 + 1/x^3
theorem value_of_x_cubed (x : ℝ) (h : x^2 - 3 * x + 1 = 0) : x^3 + 1 / x^3 = 18 :=
sorry

end factorize_polynomial_value_of_x_cubed_l26_26912


namespace cos_alpha_second_quadrant_l26_26678

variable (α : Real)
variable (h₁ : α ∈ Set.Ioo (π / 2) π)
variable (h₂ : Real.sin α = 5 / 13)

theorem cos_alpha_second_quadrant : Real.cos α = -12 / 13 := by
  sorry

end cos_alpha_second_quadrant_l26_26678


namespace solution_unique_2014_l26_26354

theorem solution_unique_2014 (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (2 * x - 2 * y + 1 / z = 1 / 2014) ∧
  (2 * y - 2 * z + 1 / x = 1 / 2014) ∧
  (2 * z - 2 * x + 1 / y = 1 / 2014) →
  x = 2014 ∧ y = 2014 ∧ z = 2014 :=
by
  sorry

end solution_unique_2014_l26_26354


namespace bicycles_in_garage_l26_26589

theorem bicycles_in_garage 
  (B : ℕ) 
  (h1 : 4 * 3 = 12) 
  (h2 : 7 * 1 = 7) 
  (h3 : 2 * B + 12 + 7 = 25) : 
  B = 3 := 
by
  sorry

end bicycles_in_garage_l26_26589


namespace simplify_expression_l26_26223

def operation (a b : ℚ) : ℚ := 2 * a - b

theorem simplify_expression (x y : ℚ) : 
  operation (operation (x - y) (x + y)) (-3 * y) = 2 * x - 3 * y :=
by
  sorry

end simplify_expression_l26_26223


namespace walking_speed_of_A_l26_26477

-- Given conditions
def B_speed := 20 -- kmph
def start_delay := 10 -- hours
def distance_covered := 200 -- km

-- Prove A's walking speed
theorem walking_speed_of_A (v : ℝ) (time_A : ℝ) (time_B : ℝ) :
  distance_covered = v * time_A ∧ distance_covered = B_speed * time_B ∧ time_B = time_A - start_delay → v = 10 :=
by
  intro h
  sorry

end walking_speed_of_A_l26_26477


namespace infinite_3_stratum_numbers_l26_26446

-- Condition for 3-stratum number
def is_3_stratum_number (n : ℕ) : Prop :=
  ∃ (A B C : Finset ℕ), A ∪ B ∪ C = (Finset.range (n + 1)).filter (λ x => n % x = 0) ∧
  A ∩ B = ∅ ∧ B ∩ C = ∅ ∧ A ∩ C = ∅ ∧
  A.sum id = B.sum id ∧ B.sum id = C.sum id

-- Part (a): Find a 3-stratum number
example : is_3_stratum_number 120 := sorry

-- Part (b): Prove there are infinitely many 3-stratum numbers
theorem infinite_3_stratum_numbers : ∃ (f : ℕ → ℕ), ∀ n, is_3_stratum_number (f n) := sorry

end infinite_3_stratum_numbers_l26_26446


namespace sum_of_ages_l26_26578

variables (P M Mo : ℕ)

def age_ratio_PM := 3 * M = 5 * P
def age_ratio_MMo := 3 * Mo = 5 * M
def age_difference := Mo = P + 64

theorem sum_of_ages : age_ratio_PM P M → age_ratio_MMo M Mo → age_difference P Mo → P + M + Mo = 196 :=
by
  intros h1 h2 h3
  sorry

end sum_of_ages_l26_26578


namespace symmetric_point_x_axis_l26_26276

theorem symmetric_point_x_axis (P : ℝ × ℝ) (hx : P = (2, 3)) : P.1 = 2 ∧ P.2 = -3 :=
by
  -- The proof is omitted
  sorry

end symmetric_point_x_axis_l26_26276


namespace geometric_sequence_b_l26_26778

theorem geometric_sequence_b (b : ℝ) (r : ℝ) (hb : b > 0)
  (h1 : 10 * r = b)
  (h2 : b * r = 10 / 9)
  (h3 : (10 / 9) * r = 10 / 81) :
  b = 10 :=
sorry

end geometric_sequence_b_l26_26778


namespace Jenny_walked_distance_l26_26553

-- Given: Jenny ran 0.6 mile.
-- Given: Jenny ran 0.2 miles farther than she walked.
-- Prove: Jenny walked 0.4 miles.

variable (r w : ℝ)

theorem Jenny_walked_distance
  (h1 : r = 0.6) 
  (h2 : r = w + 0.2) : 
  w = 0.4 :=
sorry

end Jenny_walked_distance_l26_26553


namespace log_eqn_proof_l26_26528

theorem log_eqn_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h1 : Real.log a / Real.log 2 + Real.log b / Real.log 4 = 8)
  (h2 : Real.log a / Real.log 4 + Real.log b / Real.log 8 = 2) :
  Real.log a / Real.log 8 + Real.log b / Real.log 2 = -52 / 3 := 
by
  sorry

end log_eqn_proof_l26_26528


namespace correct_equation_l26_26566

-- Define the initial deposit
def initial_deposit : ℝ := 2500

-- Define the total amount after one year with interest tax deducted
def total_amount : ℝ := 2650

-- Define the annual interest rate
variable (x : ℝ)

-- Define the interest tax rate
def interest_tax_rate : ℝ := 0.20

-- Define the equation for the total amount after one year considering the tax
theorem correct_equation :
  initial_deposit * (1 + (1 - interest_tax_rate) * x) = total_amount :=
sorry

end correct_equation_l26_26566


namespace smallest_number_of_ten_consecutive_natural_numbers_l26_26620

theorem smallest_number_of_ten_consecutive_natural_numbers 
  (x : ℕ) 
  (h : 6 * x + 39 = 2 * (4 * x + 6) + 15) : 
  x = 6 := 
by 
  sorry

end smallest_number_of_ten_consecutive_natural_numbers_l26_26620


namespace MishaTotalMoney_l26_26254

-- Define Misha's initial amount of money
def initialMoney : ℕ := 34

-- Define the amount of money Misha earns
def earnedMoney : ℕ := 13

-- Define the total amount of money Misha will have
def totalMoney : ℕ := initialMoney + earnedMoney

-- Statement to prove
theorem MishaTotalMoney : totalMoney = 47 := by
  sorry

end MishaTotalMoney_l26_26254


namespace time_addition_correct_l26_26989

def start_time := (3, 0, 0) -- Representing 3:00:00 PM as (hours, minutes, seconds)
def additional_time := (315, 78, 30) -- Representing additional time as (hours, minutes, seconds)

noncomputable def resulting_time (start add : ℕ × ℕ × ℕ) : ℕ × ℕ × ℕ :=
  let (sh, sm, ss) := start -- start hours, minutes, seconds
  let (ah, am, as) := add -- additional hours, minutes, seconds
  let total_seconds := ss + as
  let extra_minutes := total_seconds / 60
  let remaining_seconds := total_seconds % 60
  let total_minutes := sm + am + extra_minutes
  let extra_hours := total_minutes / 60
  let remaining_minutes := total_minutes % 60
  let total_hours := sh + ah + extra_hours
  let resulting_hours := (total_hours % 12) -- Modulo 12 for wrap-around
  (resulting_hours, remaining_minutes, remaining_seconds)

theorem time_addition_correct :
  let (A, B, C) := resulting_time start_time additional_time
  A + B + C = 55 := by
  sorry

end time_addition_correct_l26_26989


namespace functions_of_same_family_count_l26_26241

theorem functions_of_same_family_count : 
  (∃ (y : ℝ → ℝ), ∀ x, y x = x^2) ∧ 
  (∃ (range_set : Set ℝ), range_set = {1, 2}) → 
  ∃ n, n = 9 :=
by
  sorry

end functions_of_same_family_count_l26_26241


namespace side_length_of_octagon_l26_26892

-- Define the conditions
def is_octagon (n : ℕ) := n = 8
def perimeter (p : ℕ) := p = 72

-- Define the problem statement
theorem side_length_of_octagon (n p l : ℕ) 
  (h1 : is_octagon n) 
  (h2 : perimeter p) 
  (h3 : p / n = l) :
  l = 9 := 
  sorry

end side_length_of_octagon_l26_26892


namespace max_value_expression_l26_26510

noncomputable def f : Real → Real := λ x => 3 * Real.sin x + 4 * Real.cos x

theorem max_value_expression (θ : Real) (h_max : ∀ x, f x ≤ 5) :
  (3 * Real.sin θ + 4 * Real.cos θ = 5) →
  (Real.sin (2 * θ) + Real.cos θ ^ 2 + 1) / Real.cos (2 * θ) = 65 / 7 := by
  sorry

end max_value_expression_l26_26510


namespace sum_of_cubes_l26_26864

theorem sum_of_cubes (k : ℤ) : 
  24 * k = (k + 2)^3 + (-k)^3 + (-k)^3 + (k - 2)^3 :=
by
  sorry

end sum_of_cubes_l26_26864


namespace ratio_of_typing_speeds_l26_26313

-- Defining Tim's and Tom's typing speeds
variables (T M : ℝ)

-- Conditions given in the problem
def condition1 : Prop := T + M = 15
def condition2 : Prop := T + 1.6 * M = 18

-- Conclusion to be proved: the ratio of M to T is 1:2
theorem ratio_of_typing_speeds (h1 : condition1 T M) (h2 : condition2 T M) :
  M / T = 1 / 2 :=
by
  -- skip the proof
  sorry

end ratio_of_typing_speeds_l26_26313


namespace father_current_age_is_85_l26_26790

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l26_26790


namespace how_many_leaves_l26_26743

def ladybugs_per_leaf : ℕ := 139
def total_ladybugs : ℕ := 11676

theorem how_many_leaves : total_ladybugs / ladybugs_per_leaf = 84 :=
by
  sorry

end how_many_leaves_l26_26743


namespace pine_cones_on_roof_l26_26330

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l26_26330


namespace diametrically_opposite_points_l26_26328

theorem diametrically_opposite_points (n : ℕ) (h : (35 - 7 = n / 2)) : n = 56 := by
  sorry

end diametrically_opposite_points_l26_26328


namespace uki_total_earnings_l26_26600

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end uki_total_earnings_l26_26600


namespace prove_ratio_chickens_pigs_horses_sheep_l26_26168

noncomputable def ratio_chickens_pigs_horses_sheep (c p h s : ℕ) : Prop :=
  (∃ k : ℕ, c = 26*k ∧ p = 5*k) ∧
  (∃ l : ℕ, s = 25*l ∧ h = 9*l) ∧
  (∃ m : ℕ, p = 10*m ∧ h = 3*m) ∧
  c = 156 ∧ p = 30 ∧ h = 9 ∧ s = 25

theorem prove_ratio_chickens_pigs_horses_sheep (c p h s : ℕ) :
  ratio_chickens_pigs_horses_sheep c p h s :=
sorry

end prove_ratio_chickens_pigs_horses_sheep_l26_26168


namespace points_on_inverse_proportion_l26_26570

theorem points_on_inverse_proportion (y_1 y_2 : ℝ) :
  (2:ℝ) = 5 / y_1 → (3:ℝ) = 5 / y_2 → y_1 > y_2 :=
by
  intros h1 h2
  sorry

end points_on_inverse_proportion_l26_26570


namespace problem1_problem2_problem3_l26_26649

-- Proof Problem 1
theorem problem1 : -12 - (-18) + (-7) = -1 := 
by {
  sorry
}

-- Proof Problem 2
theorem problem2 : ((4 / 7) - (1 / 9) + (2 / 21)) * (-63) = -35 := 
by {
  sorry
}

-- Proof Problem 3
theorem problem3 : ((-4) ^ 2) / 2 + 9 * (-1 / 3) - abs (3 - 4) = 4 := 
by {
  sorry
}

end problem1_problem2_problem3_l26_26649


namespace fifty_third_number_is_2_pow_53_l26_26554

theorem fifty_third_number_is_2_pow_53 :
  ∀ n : ℕ, (n = 53) → ∃ seq : ℕ → ℕ, (seq 1 = 2) ∧ (∀ k : ℕ, seq (k+1) = 2 * seq k) ∧ (seq n = 2 ^ 53) :=
  sorry

end fifty_third_number_is_2_pow_53_l26_26554


namespace jessie_weight_before_jogging_l26_26925

-- Definitions: conditions from the problem statement
variables (lost_weight current_weight : ℤ)
-- Conditions
def condition_lost_weight : Prop := lost_weight = 126
def condition_current_weight : Prop := current_weight = 66

-- Proposition to be proved
theorem jessie_weight_before_jogging (W_before_jogging : ℤ) :
  condition_lost_weight lost_weight → condition_current_weight current_weight →
  W_before_jogging = current_weight + lost_weight → W_before_jogging = 192 :=
by
  intros
  sorry

end jessie_weight_before_jogging_l26_26925


namespace parallelogram_base_length_l26_26143

theorem parallelogram_base_length (A h : ℕ) (hA : A = 32) (hh : h = 8) : (A / h) = 4 := by
  sorry

end parallelogram_base_length_l26_26143


namespace theta_half_quadrant_l26_26844

open Real

theorem theta_half_quadrant (θ : ℝ) (k : ℤ) 
  (h1 : 2 * k * π + 3 * π / 2 ≤ θ ∧ θ ≤ 2 * k * π + 2 * π) 
  (h2 : |cos (θ / 2)| = -cos (θ / 2)) : 
  k * π + 3 * π / 4 ≤ θ / 2 ∧ θ / 2 ≤ k * π + π ∧ cos (θ / 2) < 0 := 
sorry

end theta_half_quadrant_l26_26844


namespace find_r_for_f_of_3_eq_0_l26_26707

noncomputable def f (x r : ℝ) : ℝ := 3 * x^4 - 2 * x^3 + x^2 - 4 * x + r

theorem find_r_for_f_of_3_eq_0 : ∃ r : ℝ, f 3 r = 0 ∧ r = -186 := by
  sorry

end find_r_for_f_of_3_eq_0_l26_26707


namespace area_of_common_region_l26_26918

noncomputable def common_area (length : ℝ) (width : ℝ) (radius : ℝ) : ℝ :=
  let pi := Real.pi
  let sector_area := (pi * radius^2 / 4) * 4
  let triangle_area := (1 / 2) * (width / 2) * (length / 2) * 4
  sector_area - triangle_area

theorem area_of_common_region :
  common_area 10 (Real.sqrt 18) 3 = 9 * (Real.pi) - 9 :=
by
  sorry

end area_of_common_region_l26_26918


namespace gcd_2024_2048_l26_26448

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end gcd_2024_2048_l26_26448


namespace sum_of_subsets_is_power_of_two_l26_26078

theorem sum_of_subsets_is_power_of_two :
  let S := finset.Icc 1 1999 in
  let f (s: finset ℕ) : ℕ := s.sum id in
  let total_sum := f S in
  total_sum = 1999000 ∧
  (∑ E in S.powerset, (f E : ℚ) / (f S : ℚ)) = 2 ^ 1998 :=
by {
  let S := finset.Icc 1 1999,
  let f : finset ℕ → ℕ := fun s => s.sum id,
  let total_sum := f S,
  have h_total_sum : total_sum = 1999000 := sorry,
  have h_sum :
    ∑ E in S.powerset, (f E : ℚ) / (f S : ℚ) = 2 ^ 1998 := sorry,
  exact ⟨h_total_sum, h_sum⟩
}

end sum_of_subsets_is_power_of_two_l26_26078


namespace scissors_total_l26_26018

theorem scissors_total (initial_scissors : ℕ) (additional_scissors : ℕ) (h1 : initial_scissors = 54) (h2 : additional_scissors = 22) : 
  initial_scissors + additional_scissors = 76 :=
by
  sorry

end scissors_total_l26_26018


namespace faster_train_length_225_l26_26032

noncomputable def length_of_faster_train (speed_slower speed_faster : ℝ) (time : ℝ) : ℝ :=
  let relative_speed_kmph := speed_slower + speed_faster
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  relative_speed_mps * time

theorem faster_train_length_225 :
  length_of_faster_train 36 45 10 = 225 := by
  sorry

end faster_train_length_225_l26_26032


namespace spider_total_distance_l26_26922

-- Define points where spider starts and moves
def start_position : ℤ := 3
def first_move : ℤ := -4
def second_move : ℤ := 8
def final_move : ℤ := 2

-- Define the total distance the spider crawls
def total_distance : ℤ :=
  |first_move - start_position| +
  |second_move - first_move| +
  |final_move - second_move|

-- Theorem statement
theorem spider_total_distance : total_distance = 25 :=
sorry

end spider_total_distance_l26_26922


namespace set_of_positive_reals_l26_26961

theorem set_of_positive_reals (S : Set ℝ) (h1 : ∀ x, x ∈ S → 0 < x)
  (h2 : ∀ a b, a ∈ S → b ∈ S → a + b ∈ S)
  (h3 : ∀ (a b : ℝ), 0 < a → a ≤ b → ∃ c d, a ≤ c ∧ c ≤ d ∧ d ≤ b ∧ ∀ x, c ≤ x ∧ x ≤ d → x ∈ S) :
  S = {x : ℝ | 0 < x} :=
sorry

end set_of_positive_reals_l26_26961


namespace part1_part2_l26_26807

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l26_26807


namespace vinegar_mixture_concentration_l26_26611

theorem vinegar_mixture_concentration :
  let c1 := 5 / 100
  let c2 := 10 / 100
  let v1 := 10
  let v2 := 10
  (v1 * c1 + v2 * c2) / (v1 + v2) = 7.5 / 100 :=
by
  sorry

end vinegar_mixture_concentration_l26_26611


namespace positive_integer_solutions_of_inequality_system_l26_26271

theorem positive_integer_solutions_of_inequality_system :
  {x : ℤ | 2 * (x - 1) < x + 1 ∧ 1 - (2 * x + 5) / 3 ≤ x ∧ x > 0} = {1, 2} :=
by
  sorry

end positive_integer_solutions_of_inequality_system_l26_26271


namespace grasshopper_visit_all_points_min_jumps_l26_26286

noncomputable def grasshopper_min_jumps : ℕ := 18

theorem grasshopper_visit_all_points_min_jumps (n m : ℕ) (h₁ : n = 2014) (h₂ : m = 18) :
  ∃ k : ℕ, k ≤ m ∧ (∀ i : ℤ, 0 ≤ i → i < n → ∃ j : ℕ, j < k ∧ (j * 57 + i * 10) % n = i) :=
sorry

end grasshopper_visit_all_points_min_jumps_l26_26286


namespace t_shirt_cost_calculation_l26_26717

variables (initial_amount ticket_cost food_cost money_left t_shirt_cost : ℕ)

axiom h1 : initial_amount = 75
axiom h2 : ticket_cost = 30
axiom h3 : food_cost = 13
axiom h4 : money_left = 9

theorem t_shirt_cost_calculation : 
  t_shirt_cost = initial_amount - (ticket_cost + food_cost) - money_left :=
sorry

end t_shirt_cost_calculation_l26_26717


namespace square_of_binomial_b_value_l26_26110

theorem square_of_binomial_b_value (b : ℤ) (h : ∃ c : ℤ, 16 * (x : ℤ) * x + 40 * x + b = (4 * x + c) ^ 2) : b = 25 :=
sorry

end square_of_binomial_b_value_l26_26110


namespace popsicle_melting_faster_l26_26036

theorem popsicle_melting_faster (t : ℕ) :
  ∀ (n : ℕ), if n = 6 then (2 ^ (n - 1)) * t = 32 * t else true :=
by
  intro n
  cases n
  case zero => exact true.intro
  case succ n =>
    cases n
    case zero => exact true.intro
    case succ n =>
      cases n
      case zero => exact true.intro
      case succ n =>
        cases n
        case zero => exact true.intro
        case succ n =>
          cases n
          case zero => exact true.intro
          case succ n =>
            cases n
            case zero => exact true.intro
            case succ n =>
              case zero => exact true.intro
              sorry

end popsicle_melting_faster_l26_26036


namespace cube_root_of_unity_identity_l26_26226

theorem cube_root_of_unity_identity (ω : ℂ) (hω3: ω^3 = 1) (hω_ne_1 : ω ≠ 1) (hunit : ω^2 + ω + 1 = 0) :
  (1 - ω) * (1 - ω^2) * (1 - ω^4) * (1 - ω^8) = 9 :=
by
  sorry

end cube_root_of_unity_identity_l26_26226


namespace eduardo_frankie_classes_total_l26_26943

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end eduardo_frankie_classes_total_l26_26943


namespace right_triangle_side_lengths_l26_26739

theorem right_triangle_side_lengths (a b c : ℝ) (varrho r : ℝ) (h_varrho : varrho = 8) (h_r : r = 41) : 
  (a = 80 ∧ b = 18 ∧ c = 82) ∨ (a = 18 ∧ b = 80 ∧ c = 82) :=
by
  sorry

end right_triangle_side_lengths_l26_26739


namespace different_color_socks_l26_26394

def total_socks := 15
def white_socks := 6
def brown_socks := 5
def blue_socks := 4

theorem different_color_socks (total : ℕ) (white : ℕ) (brown : ℕ) (blue : ℕ) :
  total = white + brown + blue →
  white ≠ 0 → brown ≠ 0 → blue ≠ 0 →
  (white * brown + brown * blue + white * blue) = 74 :=
by
  intros
  -- proof goes here
  sorry

end different_color_socks_l26_26394


namespace gcd_2024_2048_l26_26449

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := 
by
  sorry

end gcd_2024_2048_l26_26449


namespace find_k_l26_26856

def vec2 := ℝ × ℝ

-- Definitions
def i : vec2 := (1, 0)
def j : vec2 := (0, 1)
def a : vec2 := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : vec2 := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Dot product definition for 2D vectors
def dot_product (u v : vec2) : ℝ := u.1 * v.1 + u.2 * v.2

-- Theorem
theorem find_k (k : ℝ) : dot_product a (b k) = 0 → k = 6 :=
by
  sorry

end find_k_l26_26856


namespace find_additional_student_number_l26_26444

def classSize : ℕ := 52
def sampleSize : ℕ := 4
def sampledNumbers : List ℕ := [5, 31, 44]
def additionalStudentNumber : ℕ := 18

theorem find_additional_student_number (classSize sampleSize : ℕ) 
    (sampledNumbers : List ℕ) : additionalStudentNumber ∈ (5 :: 31 :: 44 :: []) →
    (sampledNumbers = [5, 31, 44]) →
    (additionalStudentNumber = 18) := by
  sorry

end find_additional_student_number_l26_26444


namespace free_time_left_after_cleaning_l26_26687

-- Define the time it takes for each task
def vacuuming_time : ℤ := 45
def dusting_time : ℤ := 60
def mopping_time : ℤ := 30
def brushing_time_per_cat : ℤ := 5
def number_of_cats : ℤ := 3
def total_free_time_in_minutes : ℤ := 3 * 60 -- 3 hours converted to minutes

-- Define the total cleaning time
def total_cleaning_time : ℤ := vacuuming_time + dusting_time + mopping_time + (brushing_time_per_cat * number_of_cats)

-- Prove that the free time left after cleaning is 30 minutes
theorem free_time_left_after_cleaning : (total_free_time_in_minutes - total_cleaning_time) = 30 :=
by
  sorry

end free_time_left_after_cleaning_l26_26687


namespace mean_first_second_fifth_sixth_diff_l26_26675

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end mean_first_second_fifth_sixth_diff_l26_26675


namespace cos_neg_two_pi_over_three_eq_l26_26495

noncomputable def cos_neg_two_pi_over_three : ℝ := -2 * Real.pi / 3

theorem cos_neg_two_pi_over_three_eq :
  Real.cos cos_neg_two_pi_over_three = -1 / 2 :=
sorry

end cos_neg_two_pi_over_three_eq_l26_26495


namespace optionA_optionB_optionC_optionD_l26_26120

-- Given an acute triangle ABC with angles A, B, and C
variables {A B C : ℝ}
-- Assume the angles are between 0 and π/2
variable hacute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2
-- Assume the angles sum to π
variable hsum : A + B + C = π

-- Prove each statement
theorem optionA (hA_B : A > B) : sin A > sin B := sorry
theorem optionB (hA_eq : A = π / 3) : ¬(0 < B ∧ B < π / 2) := sorry
theorem optionC : sin A + sin B > cos A + cos B := sorry
theorem optionD : tan B * tan C > 1 := sorry

end optionA_optionB_optionC_optionD_l26_26120


namespace rabbit_is_hit_l26_26292

noncomputable def P_A : ℝ := 0.6
noncomputable def P_B : ℝ := 0.5
noncomputable def P_C : ℝ := 0.4

noncomputable def P_none_hit : ℝ := (1 - P_A) * (1 - P_B) * (1 - P_C)
noncomputable def P_rabbit_hit : ℝ := 1 - P_none_hit

theorem rabbit_is_hit :
  P_rabbit_hit = 0.88 :=
by
  -- Proof is omitted
  sorry

end rabbit_is_hit_l26_26292


namespace larger_angle_measure_l26_26897

theorem larger_angle_measure (x : ℝ) (hx : 7 * x = 90) : 4 * x = 360 / 7 := by
sorry

end larger_angle_measure_l26_26897


namespace part1_part2_l26_26836

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l26_26836


namespace circumcircle_eq_l26_26086

noncomputable def A : (ℝ × ℝ) := (0, 0)
noncomputable def B : (ℝ × ℝ) := (4, 0)
noncomputable def C : (ℝ × ℝ) := (0, 6)

theorem circumcircle_eq :
  ∃ h k r, h = 2 ∧ k = 3 ∧ r = 13 ∧ (∀ x y, ((x - h)^2 + (y - k)^2 = r) ↔ (x - 2)^2 + (y - 3)^2 = 13) := sorry

end circumcircle_eq_l26_26086


namespace inequality_part1_inequality_part2_l26_26818

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l26_26818


namespace inequality_part1_inequality_part2_l26_26819

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l26_26819


namespace cube_volume_l26_26588

theorem cube_volume (S : ℝ) (h : S = 150) : ∃ V, V = 125 := 
by
  sorry

end cube_volume_l26_26588


namespace arithmetic_seq_max_S_l26_26800

theorem arithmetic_seq_max_S {S : ℕ → ℝ} (h1 : S 2023 > 0) (h2 : S 2024 < 0) : S 1012 > S 1013 :=
sorry

end arithmetic_seq_max_S_l26_26800


namespace interval_length_l26_26940

theorem interval_length (c d : ℝ) (h : ∃ x : ℝ, c ≤ 3 * x + 4 ∧ 3 * x + 4 ≤ d)
  (length : (d - 4) / 3 - (c - 4) / 3 = 15) : d - c = 45 :=
by
  sorry

end interval_length_l26_26940


namespace probability_no_defective_pencils_l26_26532

theorem probability_no_defective_pencils : 
  let total_pencils := 9
  let defective_pencils := 2
  let chosen_pencils := 3
  let non_defective_pencils := total_pencils - defective_pencils
  let total_ways := Nat.choose total_pencils chosen_pencils
  let non_defective_ways := Nat.choose non_defective_pencils chosen_pencils
  let probability := non_defective_ways / total_ways
  probability = 5 / 12 := 
by
  sorry

end probability_no_defective_pencils_l26_26532


namespace transform_center_l26_26342

def point := (ℝ × ℝ)

def reflect_x_axis (p : point) : point :=
  (p.1, -p.2)

def translate_right (p : point) (d : ℝ) : point :=
  (p.1 + d, p.2)

theorem transform_center (C : point) (hx : C = (3, -4)) :
  translate_right (reflect_x_axis C) 3 = (6, 4) :=
by
  sorry

end transform_center_l26_26342


namespace f_eq_f_inv_l26_26055

noncomputable def f (x : ℝ) : ℝ := 3 * x - 7

noncomputable def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv (x : ℝ) : f x = f_inv x ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_l26_26055


namespace alternating_boys_girls_arrangements_l26_26890

theorem alternating_boys_girls_arrangements :
  let n := 4
  let arrangements := (nat.factorial n) * (nat.factorial n)
  2 * arrangements * arrangements = 2 * (nat.factorial n) ^ 2 * (nat.factorial n) ^ 2 :=
by
  sorry  -- Mathematically equivalent proof yet to be provided.

end alternating_boys_girls_arrangements_l26_26890


namespace original_ratio_l26_26163

theorem original_ratio (x y : ℤ) (h1 : y = 24) (h2 : (x + 6) / y = 1 / 2) : x / y = 1 / 4 := by
  sorry

end original_ratio_l26_26163


namespace pine_cone_weight_on_roof_l26_26333

theorem pine_cone_weight_on_roof
  (num_trees : ℕ) (cones_per_tree : ℕ) (percentage_on_roof : ℝ) (weight_per_cone : ℕ)
  (H1 : num_trees = 8)
  (H2 : cones_per_tree = 200)
  (H3 : percentage_on_roof = 0.30)
  (H4 : weight_per_cone = 4) :
  num_trees * cones_per_tree * percentage_on_roof * weight_per_cone = 1920 := by
  sorry

end pine_cone_weight_on_roof_l26_26333


namespace find_third_sum_l26_26535

def arithmetic_sequence_sum (a : ℕ → ℝ) (d : ℝ) : Prop :=
  (a 1) + (a 4) + (a 7) = 39 ∧ (a 2) + (a 5) + (a 8) = 33

theorem find_third_sum (a : ℕ → ℝ)
                       (d : ℝ)
                       (h_seq : arithmetic_sequence_sum a d)
                       (a_1 : ℝ) :
  a 1 = a_1 ∧ a 2 = a_1 + d ∧ a 3 = a_1 + 2 * d ∧
  a 4 = a_1 + 3 * d ∧ a 5 = a_1 + 4 * d ∧ a 6 = a_1 + 5 * d ∧
  a 7 = a_1 + 6 * d ∧ a 8 = a_1 + 7 * d ∧ a 9 = a_1 + 8 * d →
  a 3 + a 6 + a 9 = 27 :=
by
  sorry

end find_third_sum_l26_26535


namespace math_problem_l26_26937

theorem math_problem :
  ( ∏ i in [3, 4, 5, 6, 7], (i^3 - 1) / (i^3 + 1) ) = 57 / 84 := sorry

end math_problem_l26_26937


namespace product_of_sum_positive_and_quotient_negative_l26_26982

-- Definitions based on conditions in the problem
def sum_positive (a b : ℝ) : Prop := a + b > 0
def quotient_negative (a b : ℝ) : Prop := a / b < 0

-- Problem statement as a theorem
theorem product_of_sum_positive_and_quotient_negative (a b : ℝ)
  (h1 : sum_positive a b)
  (h2 : quotient_negative a b) :
  a * b < 0 := by
  sorry

end product_of_sum_positive_and_quotient_negative_l26_26982


namespace equilateral_triangle_coloring_l26_26447

theorem equilateral_triangle_coloring (color : Fin 3 → Prop) :
  (∀ i, color i = true ∨ color i = false) →
  ∃ i j : Fin 3, i ≠ j ∧ color i = color j :=
by
  sorry

end equilateral_triangle_coloring_l26_26447


namespace union_sets_l26_26671

def M (a : ℕ) : Set ℕ := {a, 0}
def N : Set ℕ := {1, 2}

theorem union_sets (a : ℕ) (h_inter : M a ∩ N = {2}) : M a ∪ N = {0, 1, 2} :=
by
  sorry

end union_sets_l26_26671


namespace delaney_travel_time_l26_26653

def bus_leaves_at := 8 * 60
def delaney_left_at := 7 * 60 + 50
def missed_by := 20

theorem delaney_travel_time
  (bus_leaves_at : ℕ) (delaney_left_at : ℕ) (missed_by : ℕ) :
  delaney_left_at + (bus_leaves_at + missed_by - bus_leaves_at) - delaney_left_at = 30 :=
by
  exact sorry

end delaney_travel_time_l26_26653


namespace administrative_staff_drawn_in_stratified_sampling_l26_26137

theorem administrative_staff_drawn_in_stratified_sampling
  (total_staff : ℕ)
  (full_time_teachers : ℕ)
  (administrative_staff : ℕ)
  (logistics_personnel : ℕ)
  (sample_size : ℕ)
  (h_total : total_staff = 320)
  (h_teachers : full_time_teachers = 248)
  (h_admin : administrative_staff = 48)
  (h_logistics : logistics_personnel = 24)
  (h_sample : sample_size = 40)
  : (administrative_staff * (sample_size / total_staff) = 6) :=
by
  -- mathematical proof goes here
  sorry

end administrative_staff_drawn_in_stratified_sampling_l26_26137


namespace n_sum_of_two_squares_l26_26845

theorem n_sum_of_two_squares (n : ℤ) (m : ℤ) (hn_gt_2 : n > 2) (hn2_eq_diff_cubes : n^2 = (m+1)^3 - m^3) : 
  ∃ a b : ℤ, n = a^2 + b^2 :=
sorry

end n_sum_of_two_squares_l26_26845


namespace years_to_earn_house_l26_26633

-- Defining the variables
variables (E S H : ℝ)

-- Defining the assumptions
def annual_expenses_savings_relation (E S : ℝ) : Prop :=
  8 * E = 12 * S

def annual_income_relation (H E S : ℝ) : Prop :=
  H / 24 = E + S

-- Theorem stating that it takes 60 years to earn the amount needed to buy the house
theorem years_to_earn_house (E S H : ℝ) 
  (h1 : annual_expenses_savings_relation E S) 
  (h2 : annual_income_relation H E S) : 
  H / S = 60 :=
by
  sorry

end years_to_earn_house_l26_26633


namespace correct_statements_l26_26222

variables (a : Nat → ℤ) (d : ℤ)

-- Suppose {a_n} is an arithmetic sequence with common difference d
def S (n : ℕ) : ℤ := (n * (2 * a 1 + (n - 1) * d)) / 2

-- Conditions: S_11 > 0 and S_12 < 0
axiom S11_pos : S a d 11 > 0
axiom S12_neg : S a d 12 < 0

-- The goal is to determine which statements are correct
theorem correct_statements : (d < 0) ∧ (∀ n, 1 ≤ n → n ≤ 12 → S a d 6 ≥ S a d n ∧ S a d 6 ≠ S a d 11 ) := 
sorry

end correct_statements_l26_26222


namespace trig_expression_zero_l26_26371

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l26_26371


namespace necessary_but_not_sufficient_l26_26619

theorem necessary_but_not_sufficient (x : ℝ) : 
  (0 < x ∧ x < 2) → (x^2 - x - 6 < 0) ∧ ¬ ((x^2 - x - 6 < 0) → (0 < x ∧ x < 2)) :=
by
  sorry

end necessary_but_not_sufficient_l26_26619


namespace sum_place_values_of_specified_digits_l26_26753

def numeral := 95378637153370261

def place_values_of_3s := [3 * 100000000000, 3 * 10]
def place_values_of_7s := [7 * 10000000000, 7 * 1000000, 7 * 100]
def place_values_of_5s := [5 * 10000000000000, 5 * 1000, 5 * 10000, 5 * 1]

def sum_place_values (lst : List ℕ) : ℕ :=
  lst.foldl (· + ·) 0

def sum_of_place_values := 
  sum_place_values place_values_of_3s + 
  sum_place_values place_values_of_7s + 
  sum_place_values place_values_of_5s

theorem sum_place_values_of_specified_digits :
  sum_of_place_values = 350077055735 :=
by
  sorry

end sum_place_values_of_specified_digits_l26_26753


namespace raghu_investment_l26_26617

theorem raghu_investment (R T V : ℝ) (h1 : T = 0.9 * R) (h2 : V = 1.1 * T) (h3 : R + T + V = 5780) : R = 2000 :=
by
  sorry

end raghu_investment_l26_26617


namespace first_grade_children_count_l26_26544

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l26_26544


namespace selling_price_per_machine_l26_26201

theorem selling_price_per_machine (parts_cost patent_cost : ℕ) (num_machines : ℕ) 
  (hc1 : parts_cost = 3600) (hc2 : patent_cost = 4500) (hc3 : num_machines = 45) :
  (parts_cost + patent_cost) / num_machines = 180 :=
by
  sorry

end selling_price_per_machine_l26_26201


namespace age_multiplier_l26_26634

theorem age_multiplier (S F M X : ℕ) (h1 : S = 27) (h2 : F = 48) (h3 : S + F = 75)
  (h4 : 27 - X = F - S) (h5 : F = M * X) : M = 8 :=
by
  -- Proof will be filled in here
  sorry

end age_multiplier_l26_26634


namespace tractor_planting_rate_l26_26071

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l26_26071


namespace total_chairs_in_canteen_l26_26329

theorem total_chairs_in_canteen (numRoundTables : ℕ) (numRectangularTables : ℕ) 
                                (chairsPerRoundTable : ℕ) (chairsPerRectangularTable : ℕ)
                                (h1 : numRoundTables = 2)
                                (h2 : numRectangularTables = 2)
                                (h3 : chairsPerRoundTable = 6)
                                (h4 : chairsPerRectangularTable = 7) : 
                                (numRoundTables * chairsPerRoundTable + numRectangularTables * chairsPerRectangularTable = 26) :=
by
  sorry

end total_chairs_in_canteen_l26_26329


namespace roots_cubic_eq_sum_fraction_l26_26414

theorem roots_cubic_eq_sum_fraction (p q r : ℝ)
  (h1 : p + q + r = 8)
  (h2 : p * q + p * r + q * r = 10)
  (h3 : p * q * r = 3) :
  p / (q * r + 2) + q / (p * r + 2) + r / (p * q + 2) = 8 / 69 := 
sorry

end roots_cubic_eq_sum_fraction_l26_26414


namespace michael_twenty_dollar_bills_l26_26715

/--
Michael has $280 dollars and each bill is $20 dollars.
We need to prove that the number of $20 dollar bills Michael has is 14.
-/
theorem michael_twenty_dollar_bills (total_money : ℕ) (bill_denomination : ℕ) (number_of_bills : ℕ) :
  total_money = 280 →
  bill_denomination = 20 →
  number_of_bills = total_money / bill_denomination →
  number_of_bills = 14 :=
by
  intros h1 h2 h3
  sorry

end michael_twenty_dollar_bills_l26_26715


namespace increasing_sequence_condition_l26_26227

theorem increasing_sequence_condition (a : ℕ → ℝ) (λ : ℝ) (h : ∀ n : ℕ, n > 0 → a n = n^2 + λ * n) :
  (∀ n : ℕ, n > 0 → a n < a (n + 1)) ↔ λ > -3 :=
by
  intros
  sorry

end increasing_sequence_condition_l26_26227


namespace age_in_1988_equals_sum_of_digits_l26_26401

def birth_year (x y : ℕ) : ℕ := 1900 + 10 * x + y

def age_in_1988 (birth_year : ℕ) : ℕ := 1988 - birth_year

def sum_of_digits (x y : ℕ) : ℕ := 1 + 9 + x + y

theorem age_in_1988_equals_sum_of_digits (x y : ℕ) (h0 : 0 ≤ x) (h1 : x ≤ 9) (h2 : 0 ≤ y) (h3 : y ≤ 9) 
  (h4 : age_in_1988 (birth_year x y) = sum_of_digits x y) :
  age_in_1988 (birth_year x y) = 22 :=
by {
  sorry
}

end age_in_1988_equals_sum_of_digits_l26_26401


namespace jenna_weight_lift_l26_26991

theorem jenna_weight_lift:
  ∀ (n : Nat), (2 * 10 * 25 = 500) ∧ (15 * n >= 500) ∧ (n = Nat.ceil (500 / 15 : ℝ))
  → n = 34 := 
by
  intros n h
  have h₀ : 2 * 10 * 25 = 500 := h.1
  have h₁ : 15 * n >= 500 := h.2.1
  have h₂ : n = Nat.ceil (500 / 15 : ℝ) := h.2.2
  sorry

end jenna_weight_lift_l26_26991


namespace pills_per_day_l26_26565

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end pills_per_day_l26_26565


namespace range_of_a_for_inequality_l26_26683

theorem range_of_a_for_inequality (a : ℝ) :
  (∀ x : ℝ, a * x^2 + 4 * x + a ≥ -2 * x^2 + 1) ↔ a ≥ 2 :=
by {
  sorry
}

end range_of_a_for_inequality_l26_26683


namespace henry_has_30_more_lollipops_than_alison_l26_26106

noncomputable def num_lollipops_alison : ℕ := 60
noncomputable def num_lollipops_diane : ℕ := 2 * num_lollipops_alison
noncomputable def total_num_days : ℕ := 6
noncomputable def num_lollipops_per_day : ℕ := 45
noncomputable def total_lollipops : ℕ := total_num_days * num_lollipops_per_day
noncomputable def num_lollipops_total_ad : ℕ := num_lollipops_alison + num_lollipops_diane
noncomputable def num_lollipops_henry : ℕ := total_lollipops - num_lollipops_total_ad
noncomputable def lollipops_diff_henry_alison : ℕ := num_lollipops_henry - num_lollipops_alison

theorem henry_has_30_more_lollipops_than_alison :
  lollipops_diff_henry_alison = 30 :=
by
  unfold lollipops_diff_henry_alison
  unfold num_lollipops_henry
  unfold num_lollipops_total_ad
  unfold total_lollipops
  sorry

end henry_has_30_more_lollipops_than_alison_l26_26106


namespace fraction_powers_sum_l26_26297

theorem fraction_powers_sum : 
  ( (5:ℚ) / (3:ℚ) )^6 + ( (2:ℚ) / (3:ℚ) )^6 = (15689:ℚ) / (729:ℚ) :=
by
  sorry

end fraction_powers_sum_l26_26297


namespace sum_of_solutions_of_quadratic_l26_26576

theorem sum_of_solutions_of_quadratic :
    let a := 1;
    let b := -8;
    let c := -40;
    let discriminant := b * b - 4 * a * c;
    let root_discriminant := Real.sqrt discriminant;
    let sol1 := (-b + root_discriminant) / (2 * a);
    let sol2 := (-b - root_discriminant) / (2 * a);
    sol1 + sol2 = 8 := by
{
  sorry
}

end sum_of_solutions_of_quadratic_l26_26576


namespace area_of_rectangle_l26_26442

theorem area_of_rectangle (width length : ℝ) (h_width : width = 5.4) (h_length : length = 2.5) : width * length = 13.5 :=
by
  -- We are given that the width is 5.4 and the length is 2.5
  -- We need to show that the area (width * length) is 13.5
  sorry

end area_of_rectangle_l26_26442


namespace janet_speed_l26_26705

def janet_sister_speed : ℝ := 12
def lake_width : ℝ := 60
def wait_time : ℝ := 3

theorem janet_speed :
  (lake_width / (lake_width / janet_sister_speed - wait_time)) = 30 := 
sorry

end janet_speed_l26_26705


namespace original_calculation_l26_26393

theorem original_calculation
  (x : ℝ)
  (h : ((x * 3) + 14) * 2 = 946) :
  ((x / 3) + 14) * 2 = 130 :=
sorry

end original_calculation_l26_26393


namespace measure_of_angle_l26_26146

theorem measure_of_angle (x : ℝ) (h1 : 90 = x + (3 * x + 10)) : x = 20 :=
by
  sorry

end measure_of_angle_l26_26146


namespace solve_for_x_l26_26496

theorem solve_for_x (x : ℝ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 6) : x = 3 :=
by
  sorry

end solve_for_x_l26_26496


namespace inequality_part1_inequality_part2_l26_26816

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l26_26816


namespace relative_error_approximation_l26_26445

theorem relative_error_approximation (y : ℝ) (h : |y| < 1) :
  (1 / (1 + y) - (1 - y)) / (1 / (1 + y)) = y^2 :=
by
  sorry

end relative_error_approximation_l26_26445


namespace range_of_m_l26_26522

variable {m x : ℝ}

theorem range_of_m (h : ∀ x, -1 < x ∧ x < 4 ↔ x > 2 * m ^ 2 - 3) : m ∈ [-1, 1] :=
sorry

end range_of_m_l26_26522


namespace scientific_notation_of_number_l26_26703

def number := 460000000
def scientific_notation (n : ℕ) (s : ℝ) := s * 10 ^ n

theorem scientific_notation_of_number :
  scientific_notation 8 4.6 = number :=
sorry

end scientific_notation_of_number_l26_26703


namespace max_k_no_real_roots_l26_26115

theorem max_k_no_real_roots : ∀ k : ℤ, (∀ x : ℝ, x^2 - 2 * x - (k : ℝ) ≠ 0) → k ≤ -2 :=
by
  sorry

end max_k_no_real_roots_l26_26115


namespace problem_statement_l26_26047

def S : ℤ := (-2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19)

theorem problem_statement (hS : S = -2^20 + 4) : 2 - 2^2 - 2^3 - 2^4 - 2^5 - 2^6 - 2^7 - 2^8 - 2^9 - 2^10 - 2^11 - 2^12 - 2^13 - 2^14 - 2^15 - 2^16 - 2^17 - 2^18 - 2^19 + 2^20 = 6 :=
by
  sorry

end problem_statement_l26_26047


namespace solve_for_x_l26_26497

theorem solve_for_x (x : ℝ) (h : 5 * (x - 9) = 6 * (3 - 3 * x) + 6) : x = 3 :=
by
  sorry

end solve_for_x_l26_26497


namespace binomial_coefficient_sum_mod_l26_26049

theorem binomial_coefficient_sum_mod : 
  let S := ((1 + Complex.exp (Complex.I * Real.pi / 2))^2011) + 
           ((1 + Complex.exp (3 * Complex.I * Real.pi / 2))^2011) + 
           ((1 + -1)^2011) + 
           ((1 + 1)^2011)
  in 
  let desired_sum := (range 503).sum (λ j, Nat.choose 2011 (4 * j)) / 4
  in 
  (S % 1000 = 137) :
  nat.Mod 1000 S = 137 := 
begin
  sorry
end

end binomial_coefficient_sum_mod_l26_26049


namespace trig_expression_zero_l26_26372

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l26_26372


namespace coin_flips_prob_l26_26459

open Probability

noncomputable def probability_Vanya_more_heads_than_Tanya (n : ℕ) : ℝ :=
  P(X Vanya > X Tanya)

theorem coin_flips_prob (n : ℕ) :
   P(Vanya gets more heads than Tanya | Vanya flips (n+1) times, Tanya flips n times) = 0.5 :=
sorry

end coin_flips_prob_l26_26459


namespace S7_value_l26_26437

def arithmetic_seq_sum (n : ℕ) (a_1 d : ℚ) : ℚ :=
  n * a_1 + (n * (n - 1) / 2) * d

def a_n (n : ℕ) (a_1 d : ℚ) : ℚ :=
  a_1 + (n - 1) * d

theorem S7_value (a_1 d : ℚ) (S_n : ℕ → ℚ)
  (hSn_def : ∀ n, S_n n = arithmetic_seq_sum n a_1 d)
  (h_sum_condition : S_n 7 + S_n 5 = 10)
  (h_a3_condition : a_n 3 a_1 d = 5) :
  S_n 7 = -15 :=
by
  sorry

end S7_value_l26_26437


namespace complement_union_l26_26252

open Set

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {4, 5}
def B : Set ℕ := {3, 4}

theorem complement_union : (U \ (A ∪ B)) = {1, 2, 6} :=
by simp only [U, A, B, Set.mem_union, Set.mem_compl, Set.mem_diff];
   sorry

end complement_union_l26_26252


namespace factorization_l26_26658

theorem factorization (a : ℝ) : 2 * a ^ 2 - 8 = 2 * (a + 2) * (a - 2) := 
by
  sorry

end factorization_l26_26658


namespace volleyball_team_selection_l26_26569

open Nat

-- Definitions based on the conditions
def numTotalPlayers := 14
def numTriplets := 3
def numStarters := 6
def triplets : Fin numTriplets → String := 
  λ i, ["Alicia", "Amanda", "Anna"].nth i sorry

-- Proof goal
theorem volleyball_team_selection :
  ∃ ways : ℕ,
    ways = (numTriplets.choose 2) * ((numTotalPlayers - numTriplets).choose (numStarters - 2)) ∧
    ways = 990 := 
by
  sorry

end volleyball_team_selection_l26_26569


namespace part1_part2_l26_26808

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l26_26808


namespace members_playing_both_l26_26462

theorem members_playing_both
  (N B T Neither : ℕ)
  (hN : N = 40)
  (hB : B = 20)
  (hT : T = 18)
  (hNeither : Neither = 5) :
  (B + T) - (N - Neither) = 3 := by
-- to complete the proof
sorry

end members_playing_both_l26_26462


namespace arithmetic_mean_of_scores_l26_26551

theorem arithmetic_mean_of_scores :
  let s1 := 85
  let s2 := 94
  let s3 := 87
  let s4 := 93
  let s5 := 95
  let s6 := 88
  let s7 := 90
  (s1 + s2 + s3 + s4 + s5 + s6 + s7) / 7 = 90.2857142857 :=
by
  sorry

end arithmetic_mean_of_scores_l26_26551


namespace range_of_a_l26_26098

noncomputable def setA : Set ℝ := {x | 3 + 2 * x - x^2 >= 0}
noncomputable def setB (a : ℝ) : Set ℝ := {x | x > a}

theorem range_of_a (a : ℝ) : (setA ∩ setB a).Nonempty → a < 3 :=
by
  sorry

end range_of_a_l26_26098


namespace polynomial_multiplication_identity_l26_26603

-- Statement of the problem
theorem polynomial_multiplication_identity (x : ℝ) : 
  (25 * x^3) * (12 * x^2) * (1 / (5 * x)^3) = (12 / 5) * x^2 :=
by
  sorry

end polynomial_multiplication_identity_l26_26603


namespace euler_totient_bound_l26_26720

theorem euler_totient_bound (n k : ℕ) (h1 : 0 < n) (h2 : 0 < k) (h3 : (Nat.totient^[k]) n = 1) :
  n ≤ 3^k :=
sorry

end euler_totient_bound_l26_26720


namespace maximum_elements_in_A_l26_26959

theorem maximum_elements_in_A (n : ℕ) (h : n > 0)
  (A : Finset (Finset (Fin n))) 
  (hA : ∀ a ∈ A, ∀ b ∈ A, a ≠ b → ¬ a ⊆ b) :  
  A.card ≤ Nat.choose n (n / 2) :=
sorry

end maximum_elements_in_A_l26_26959


namespace planting_rate_l26_26063

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l26_26063


namespace count_arithmetic_progressions_22_1000_l26_26516

def num_increasing_arithmetic_progressions (n k max_val : ℕ) : ℕ :=
  -- This is a stub for the arithmetic sequence counting function.
  sorry

theorem count_arithmetic_progressions_22_1000 :
  num_increasing_arithmetic_progressions 22 22 1000 = 23312 :=
sorry

end count_arithmetic_progressions_22_1000_l26_26516


namespace polynomial_square_solution_l26_26736

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end polynomial_square_solution_l26_26736


namespace pos_int_solutions_l26_26887

theorem pos_int_solutions (x : ℤ) : (3 * x - 4 < 2 * x) → (0 < x) → (x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro h1 h2
  have h3 : x - 4 < 0 := by sorry  -- Step derived from inequality simplification
  have h4 : x < 4 := by sorry     -- Adding 4 to both sides
  sorry                           -- Combine conditions to get the specific solutions

end pos_int_solutions_l26_26887


namespace belle_stickers_l26_26198

theorem belle_stickers (c_stickers : ℕ) (diff : ℕ) (b_stickers : ℕ) (h1 : c_stickers = 79) (h2 : diff = 18) (h3 : c_stickers = b_stickers - diff) : b_stickers = 97 := 
by
  sorry

end belle_stickers_l26_26198


namespace betty_oranges_l26_26339

-- Define the givens and result as Lean definitions and theorems
theorem betty_oranges (kg_apples : ℕ) (cost_apples_per_kg cost_oranges_per_kg total_cost_oranges num_oranges : ℕ) 
    (h1 : kg_apples = 3)
    (h2 : cost_apples_per_kg = 2)
    (h3 : cost_apples_per_kg * 2 = cost_oranges_per_kg)
    (h4 : 12 = total_cost_oranges)
    (h5 : total_cost_oranges / cost_oranges_per_kg = num_oranges) :
    num_oranges = 3 :=
sorry

end betty_oranges_l26_26339


namespace find_xy_l26_26465

theorem find_xy (x y : ℝ) :
  0.75 * x - 0.40 * y = 0.20 * 422.50 →
  0.30 * x + 0.50 * y = 0.35 * 530 →
  x = 52.816 ∧ y = -112.222 :=
by
  intro h1 h2
  sorry

end find_xy_l26_26465


namespace least_possible_integral_QR_l26_26746

theorem least_possible_integral_QR (PQ PR SR SQ QR : ℝ) (hPQ : PQ = 7) (hPR : PR = 10) (hSR : SR = 15) (hSQ : SQ = 24) :
  9 ≤ QR ∧ QR < 17 :=
by
  sorry

end least_possible_integral_QR_l26_26746


namespace remainder_g_x12_div_g_x_l26_26965

-- Define the polynomial g
noncomputable def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Proving the remainder when g(x^12) is divided by g(x) is 6
theorem remainder_g_x12_div_g_x : 
  (g (x^12) % g x) = 6 :=
sorry

end remainder_g_x12_div_g_x_l26_26965


namespace missy_tv_watching_time_l26_26255

def reality_show_count : Nat := 5
def reality_show_duration : Nat := 28
def cartoon_duration : Nat := 10

theorem missy_tv_watching_time :
  reality_show_count * reality_show_duration + cartoon_duration = 150 := by
  sorry

end missy_tv_watching_time_l26_26255


namespace newly_grown_uneaten_potatoes_l26_26253

variable (u : ℕ)

def initially_planted : ℕ := 8
def total_now : ℕ := 11

theorem newly_grown_uneaten_potatoes : u = total_now - initially_planted := by
  sorry

end newly_grown_uneaten_potatoes_l26_26253


namespace perimeter_trapezoid_l26_26126

theorem perimeter_trapezoid 
(E F G H : Point)
(EF GH : ℝ)
(HJ EI FG EH : ℝ)
(h_eq1 : EF = GH)
(h_FG : FG = 10)
(h_EH : EH = 20)
(h_EI : EI = 5)
(h_HJ : HJ = 5)
(h_EF_HG : EF = Real.sqrt (EI^2 + ((EH - FG) / 2)^2)) :
  2 * EF + FG + EH = 30 + 10 * Real.sqrt 2 :=
by
  sorry

end perimeter_trapezoid_l26_26126


namespace remainder_of_f_l26_26302

theorem remainder_of_f (f y : ℤ) 
  (hy : y % 5 = 4)
  (hfy : (f + y) % 5 = 2) : f % 5 = 3 :=
by
  sorry

end remainder_of_f_l26_26302


namespace debby_jogged_total_l26_26568

theorem debby_jogged_total :
  let monday_distance := 2
  let tuesday_distance := 5
  let wednesday_distance := 9
  monday_distance + tuesday_distance + wednesday_distance = 16 :=
by
  sorry

end debby_jogged_total_l26_26568


namespace Joann_lollipop_theorem_l26_26993

noncomputable def Joann_lollipops (a : ℝ) : ℝ := a + 9

theorem Joann_lollipop_theorem (a : ℝ) (total_lollipops : ℝ) 
  (h1 : a + (a + 3) + (a + 6) + (a + 9) + (a + 12) + (a + 15) = 150) 
  (h2 : total_lollipops = 150) : 
  Joann_lollipops a = 26.5 :=
by
  sorry

end Joann_lollipop_theorem_l26_26993


namespace probability_black_cubecube_approx_183_times_10_neg_37_l26_26467

theorem probability_black_cubecube_approx_183_times_10_neg_37 :
  let pos_prob := (factorial 8 * factorial 12 * factorial 6) / factorial 27
  let orient_prob := (1 / 8 ^ 8) * (1 / 12 ^ 12) * (1 / 6 ^ 6)
  let total_prob := pos_prob * orient_prob
  total_prob ≈ 1.83e-37 :=
by
  sorry

end probability_black_cubecube_approx_183_times_10_neg_37_l26_26467


namespace obtain_2020_from_20_and_21_l26_26011

theorem obtain_2020_from_20_and_21 :
  ∃ (a b : ℕ), 20 * a + 21 * b = 2020 :=
by
  -- We only need to construct the proof goal, leaving the proof itself out.
  sorry

end obtain_2020_from_20_and_21_l26_26011


namespace printers_ratio_l26_26172

theorem printers_ratio (Rate_X : ℝ := 1 / 16) (Rate_Y : ℝ := 1 / 10) (Rate_Z : ℝ := 1 / 20) :
  let Time_X := 1 / Rate_X
  let Time_YZ := 1 / (Rate_Y + Rate_Z)
  (Time_X / Time_YZ) = 12 / 5 := by
  sorry

end printers_ratio_l26_26172


namespace last_term_arithmetic_progression_eq_62_l26_26498

theorem last_term_arithmetic_progression_eq_62
  (a : ℕ) (d : ℕ) (n : ℕ) 
  (h_a : a = 2)
  (h_d : d = 2)
  (h_n : n = 31) : 
  a + (n - 1) * d = 62 :=
by
  sorry

end last_term_arithmetic_progression_eq_62_l26_26498


namespace part1_part2_l26_26812

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l26_26812


namespace pine_cones_on_roof_l26_26331

theorem pine_cones_on_roof 
  (num_trees : ℕ) 
  (pine_cones_per_tree : ℕ) 
  (percent_on_roof : ℝ) 
  (weight_per_pine_cone : ℝ) 
  (h1 : num_trees = 8)
  (h2 : pine_cones_per_tree = 200)
  (h3 : percent_on_roof = 0.30)
  (h4 : weight_per_pine_cone = 4) : 
  (num_trees * pine_cones_per_tree * percent_on_roof * weight_per_pine_cone = 1920) :=
by
  sorry

end pine_cones_on_roof_l26_26331


namespace four_b_is_222_22_percent_of_a_l26_26727

-- noncomputable is necessary because Lean does not handle decimal numbers directly
noncomputable def a (b : ℝ) : ℝ := 1.8 * b
noncomputable def four_b (b : ℝ) : ℝ := 4 * b

theorem four_b_is_222_22_percent_of_a (b : ℝ) : four_b b = 2.2222 * a b := 
by
  sorry

end four_b_is_222_22_percent_of_a_l26_26727


namespace major_axis_of_ellipse_l26_26646

structure Ellipse :=
(center : ℝ × ℝ)
(tangent_y_axis : Bool)
(tangent_y_eq_3 : Bool)
(focus_1 : ℝ × ℝ)
(focus_2 : ℝ × ℝ)

noncomputable def major_axis_length (e : Ellipse) : ℝ :=
  2 * (e.focus_1.2 - e.center.2)

theorem major_axis_of_ellipse : 
  ∀ (e : Ellipse), 
    e.center = (3, 0) ∧
    e.tangent_y_axis = true ∧
    e.tangent_y_eq_3 = true ∧
    e.focus_1 = (3, 2 + Real.sqrt 2) ∧
    e.focus_2 = (3, -2 - Real.sqrt 2) →
      major_axis_length e = 4 + 2 * Real.sqrt 2 :=
by
  intro e
  intro h
  sorry

end major_axis_of_ellipse_l26_26646


namespace canyon_trail_length_l26_26863

theorem canyon_trail_length
  (a b c d e : ℝ)
  (h1 : a + b + c = 36)
  (h2 : b + c + d = 42)
  (h3 : c + d + e = 45)
  (h4 : a + d = 29) :
  a + b + c + d + e = 71 :=
by sorry

end canyon_trail_length_l26_26863


namespace isosceles_triangle_base_vertex_trajectory_l26_26509

theorem isosceles_triangle_base_vertex_trajectory :
  ∀ (x y : ℝ), 
  (∀ (A : ℝ × ℝ) (B : ℝ × ℝ), 
    A = (2, 4) ∧ B = (2, 8) ∧ 
    ((x-2)^2 + (y-4)^2 = 16)) → 
  ((x ≠ 2) ∧ (y ≠ 8) → (x-2)^2 + (y-4)^2 = 16) :=
sorry

end isosceles_triangle_base_vertex_trajectory_l26_26509


namespace reciprocal_neg_six_l26_26888

-- Define the concept of reciprocal
def reciprocal (a : ℤ) (h : a ≠ 0) : ℚ := 1 / a

theorem reciprocal_neg_six : reciprocal (-6) (by norm_num) = -1 / 6 := 
by 
  sorry

end reciprocal_neg_six_l26_26888


namespace prob_remainder_mod_1000_l26_26053

-- Define the binomial coefficient function
def binom : ℕ → ℕ → ℕ 
| n 0 := 1
| 0 k := 0
| n k := binom (n-1) (k-1) * n / k

-- Define the sum we are interested in, only including indices that are multiples of 4
def sum_binom_2011_multiple_4 : ℕ :=
  (Finset.range (2012 / 4 + 1)).sum (λ i, binom 2011 (4 * i))

-- The statement we want to prove
theorem prob_remainder_mod_1000 : 
  sum_binom_2011_multiple_4 % 1000 = 12 := 
sorry

end prob_remainder_mod_1000_l26_26053


namespace eval_expr_l26_26046

theorem eval_expr : (900 ^ 2) / (262 ^ 2 - 258 ^ 2) = 389.4 := 
by
  sorry

end eval_expr_l26_26046


namespace part_a_l26_26902

-- Power tower with 100 twos
def power_tower_100_t2 : ℕ := sorry

theorem part_a : power_tower_100_t2 > 3 := sorry

end part_a_l26_26902


namespace probability_of_drawing_white_l26_26187

-- Definitions of the conditions
def initial_urn : Type := { a // 0 ≤ a ∧ a ≤ 2 }
def urn_after_adding_white (b : initial_urn) := b.val + 1
def possible_initial_states := [0, 1, 2]

-- Probabilities tied to initial hypotheses
def prob_b1 : ℝ := 1 / 3
def prob_b2 : ℝ := 1 / 3
def prob_b3 : ℝ := 1 / 3

-- Conditional probabilities given each hypothesis
def P_A_given_B1 : ℝ := 1 / 3
def P_A_given_B2 : ℝ := 2 / 3
def P_A_given_B3 : ℝ := 1

-- Law of Total Probability
def P_A : ℝ := 
  prob_b1 * P_A_given_B1 + 
  prob_b2 * P_A_given_B2 + 
  prob_b3 * P_A_given_B3

theorem probability_of_drawing_white :
  P_A = 2 / 3 :=
sorry

end probability_of_drawing_white_l26_26187


namespace zeros_at_end_of_quotient_factorial_l26_26970

def count_factors_of_five (n : ℕ) : ℕ :=
  n / 5 + n / 25 + n / 125 + n / 625

theorem zeros_at_end_of_quotient_factorial :
  count_factors_of_five 2018 - count_factors_of_five 30 - count_factors_of_five 11 = 493 :=
by
  sorry

end zeros_at_end_of_quotient_factorial_l26_26970


namespace B_in_fourth_quadrant_l26_26538

theorem B_in_fourth_quadrant (a b : ℝ) (h_a : a > 0) (h_b : -b > 0) : (a > 0 ∧ b < 0) := 
  begin
    have h_b_neg : b < 0 := by linarith,
    exact ⟨h_a, h_b_neg⟩,
  end

end B_in_fourth_quadrant_l26_26538


namespace digits_sum_18_to_21_sum_digits_0_to_99_l26_26908

open Nat List

-- Lean statement definition
def sum_of_digits (n : Nat) : Nat :=
  n.digits 10 |> List.sum

theorem digits_sum_18_to_21 : 
  sum_of_digits 18 + sum_of_digits 19 + sum_of_digits 20 + sum_of_digits 21 = 24 := by 
  sorry

noncomputable def q := 
  List.range 100 |> List.map sum_of_digits |> List.sum

theorem sum_digits_0_to_99 : q = 900 := by
  sorry

end digits_sum_18_to_21_sum_digits_0_to_99_l26_26908


namespace find_some_number_l26_26756

theorem find_some_number (some_number : ℝ) (h : (3.242 * some_number) / 100 = 0.045388) : some_number = 1.400 := 
sorry

end find_some_number_l26_26756


namespace percent_increase_from_first_to_second_quarter_l26_26904

theorem percent_increase_from_first_to_second_quarter 
  (P : ℝ) :
  ((1.60 * P - 1.20 * P) / (1.20 * P)) * 100 = 33.33 := by
  sorry

end percent_increase_from_first_to_second_quarter_l26_26904


namespace range_of_alpha_minus_beta_l26_26680

theorem range_of_alpha_minus_beta (α β : ℝ) (h1 : -180 < α) (h2 : α < β) (h3 : β < 180) :
  -360 < α - β ∧ α - β < 0 :=
by
  sorry

end range_of_alpha_minus_beta_l26_26680


namespace distance_between_points_l26_26213

open Real

theorem distance_between_points :
  ∀ (x1 y1 x2 y2 : ℝ),
  (x1, y1) = (-3, 1) →
  (x2, y2) = (5, -5) →
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) = 10 :=
by
  intros x1 y1 x2 y2 h1 h2
  sorry

end distance_between_points_l26_26213


namespace product_plus_one_eq_216_l26_26618

variable (a b c : ℝ)

theorem product_plus_one_eq_216 
  (h1 : a * b + a + b = 35)
  (h2 : b * c + b + c = 35)
  (h3 : c * a + c + a = 35)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c) :
  (a + 1) * (b + 1) * (c + 1) = 216 := 
sorry

end product_plus_one_eq_216_l26_26618


namespace proof_problem_l26_26847

-- Define the given condition as a constant
def condition : Prop := 213 * 16 = 3408

-- Define the statement we need to prove under the given condition
theorem proof_problem (h : condition) : 0.16 * 2.13 = 0.3408 := 
by 
  sorry

end proof_problem_l26_26847


namespace three_points_in_circle_of_radius_one_seventh_l26_26176

-- Define the problem
theorem three_points_in_circle_of_radius_one_seventh (P : Fin 51 → ℝ × ℝ) :
  (∀ i, 0 ≤ (P i).1 ∧ (P i).1 ≤ 1 ∧ 0 ≤ (P i).2 ∧ (P i).2 ≤ 1) →
  ∃ (i j k : Fin 51), i ≠ j ∧ j ≠ k ∧ i ≠ k ∧
    dist (P i) (P j) ≤ 2/7 ∧ dist (P j) (P k) ≤ 2/7 ∧ dist (P k) (P i) ≤ 2/7 :=
by
  sorry

end three_points_in_circle_of_radius_one_seventh_l26_26176


namespace triangle_height_relationship_l26_26293

theorem triangle_height_relationship
  (b : ℝ) (h1 h2 h3 : ℝ)
  (area1 area2 area3 : ℝ)
  (h_equal_angle : area1 / area2 = 16 / 25)
  (h_diff_angle : area1 / area3 = 4 / 9) :
  4 * h2 = 5 * h1 ∧ 6 * h2 = 5 * h3 := by
    sorry

end triangle_height_relationship_l26_26293


namespace hashtag_3_8_l26_26240

-- Define the hashtag operation
def hashtag (a b : ℤ) : ℤ := a * b - b + b ^ 2

-- Prove that 3 # 8 equals 80
theorem hashtag_3_8 : hashtag 3 8 = 80 := by
  sorry

end hashtag_3_8_l26_26240


namespace abs_fraction_eq_sqrt_seven_thirds_l26_26773

open Real

theorem abs_fraction_eq_sqrt_seven_thirds {a b : ℝ} 
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt (7 / 3) :=
by
  sorry

end abs_fraction_eq_sqrt_seven_thirds_l26_26773


namespace find_geo_prog_numbers_l26_26891

noncomputable def geo_prog_numbers (a1 a2 a3 : ℝ) : Prop :=
a1 * a2 * a3 = 27 ∧ a1 + a2 + a3 = 13

theorem find_geo_prog_numbers :
  geo_prog_numbers 1 3 9 ∨ geo_prog_numbers 9 3 1 :=
sorry

end find_geo_prog_numbers_l26_26891


namespace math_competition_l26_26531

theorem math_competition :
  let Sammy_score := 20
  let Gab_score := 2 * Sammy_score
  let Cher_score := 2 * Gab_score
  let Total_score := Sammy_score + Gab_score + Cher_score
  let Opponent_score := 85
  Total_score - Opponent_score = 55 :=
by
  sorry

end math_competition_l26_26531


namespace min_value_pt_qu_rv_sw_l26_26709

theorem min_value_pt_qu_rv_sw (p q r s t u v w : ℝ) (h1 : p * q * r * s = 8) (h2 : t * u * v * w = 27) :
  (p * t) ^ 2 + (q * u) ^ 2 + (r * v) ^ 2 + (s * w) ^ 2 ≥ 96 :=
by
  sorry

end min_value_pt_qu_rv_sw_l26_26709


namespace find_t_l26_26309

-- Given conditions 
variables (p j t : ℝ)

-- Condition 1: j is 25% less than p
def condition1 : Prop := j = 0.75 * p

-- Condition 2: j is 20% less than t
def condition2 : Prop := j = 0.80 * t

-- Condition 3: t is t% less than p
def condition3 : Prop := t = p * (1 - t / 100)

-- Final proof statement
theorem find_t (h1 : condition1 p j) (h2 : condition2 j t) (h3 : condition3 p t) : t = 6.25 :=
sorry

end find_t_l26_26309


namespace find_a2_and_sum_l26_26236

theorem find_a2_and_sum (a a1 a2 a3 a4 : ℝ) (x : ℝ) (h1 : (1 + 2 * x)^4 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4) :
  a2 = 24 ∧ a + a1 + a2 + a3 + a4 = 81 :=
by
  sorry

end find_a2_and_sum_l26_26236


namespace probability_A_in_swimming_pool_l26_26432
-- Ensure necessary libraries are imported

open ProbabilityTheory

-- Define the problem conditions in Lean 4
theorem probability_A_in_swimming_pool :
  let num_volunteers := 5
  let venues := ["gymnasium", "swimming pool", "comprehensive training hall"]
  let volunteers := ["A", "B", "C", "D", "E"]
  let venues_count := venues.length
  let volunteer_count := volunteers.length
  let same_venue_condition (a b : String) (assignment : String → String) : Prop := assignment a = assignment b
  let at_least_one_each_venue (assignment : String → String) : Prop :=
    ∀ v ∈ venues, ∃ vol ∈ volunteers, assignment vol = v

  -- considering all assignments where A and B are in the same venue
  let valid_assignment : Type := {f : String → String // same_venue_condition "A" "B" f ∧ at_least_one_each_venue f}

  -- The probability calculation
  let probability_A_swimming_pool :=
    let total := {f // at_least_one_each_venue f}.length
    let favorable :=
      (filter (λ f : valid_assignment, f.val "A" = "swimming pool") (finset.univ : finset valid_assignment)).card
    favorable / total
    
  -- Show the probability is 1/3
  probability_A_swimming_pool = 1 / 3 :=
sorry

end probability_A_in_swimming_pool_l26_26432


namespace sum_of_ages_in_5_years_l26_26656

noncomputable def age_will_three_years_ago := 4
noncomputable def years_elapsed := 3
noncomputable def age_will_now := age_will_three_years_ago + years_elapsed
noncomputable def age_diane_now := 2 * age_will_now
noncomputable def years_into_future := 5
noncomputable def age_will_in_future := age_will_now + years_into_future
noncomputable def age_diane_in_future := age_diane_now + years_into_future

theorem sum_of_ages_in_5_years :
  age_will_in_future + age_diane_in_future = 31 := by
  sorry

end sum_of_ages_in_5_years_l26_26656


namespace contemporaries_probability_l26_26898

open Real

noncomputable def probability_of_contemporaries
  (born_within : ℝ) (lifespan : ℝ) : ℝ :=
  let total_area := born_within * born_within
  let side := born_within - lifespan
  let non_overlap_area := 2 * (1/2 * side * side)
  let overlap_area := total_area - non_overlap_area
  overlap_area / total_area

theorem contemporaries_probability :
  probability_of_contemporaries 300 80 = 104 / 225 := 
by
  sorry

end contemporaries_probability_l26_26898


namespace unique_solution_for_4_circ_20_l26_26202

def operation (x y : ℝ) : ℝ := 3 * x - 2 * y + 2 * x * y

theorem unique_solution_for_4_circ_20 : ∃! y : ℝ, operation 4 y = 20 :=
by 
  sorry

end unique_solution_for_4_circ_20_l26_26202


namespace sum_of_elements_in_M_l26_26233

theorem sum_of_elements_in_M (m : ℝ) (h : ∃ x : ℝ, x^2 - 2 * x + m = 0) :
  (∀ x : ℝ, x ∈ {x | x^2 - 2 * x + m = 0} → x = 1) ∧ m = 1 ∨
  (∃ x1 x2 : ℝ, x1 ∈ {x | x^2 - 2 * x + m = 0} ∧ x2 ∈ {x | x^2 - 2 * x + m = 0} ∧ x1 ≠ x2 ∧
   x1 + x2 = 2 ∧ m < 1) :=
sorry

end sum_of_elements_in_M_l26_26233


namespace no_real_pairs_for_same_lines_l26_26777

theorem no_real_pairs_for_same_lines : ¬ ∃ (a b : ℝ), (∀ x y : ℝ, 2 * x + a * y + b = 0 ↔ b * x - 3 * y + 15 = 0) :=
by {
  sorry
}

end no_real_pairs_for_same_lines_l26_26777


namespace number_of_students_l26_26118

theorem number_of_students : 
    ∃ (n : ℕ), 
      (∃ (x : ℕ), 
        (∀ (k : ℕ), x = 4 * k ∧ 5 * x + 1 = n)
      ) ∧ 
      (∃ (y : ℕ), 
        (∀ (k : ℕ), y = 5 * k ∧ 4 * y + 1 = n)
      ) ∧
      n ≤ 30 ∧ 
      n = 21 :=
  sorry

end number_of_students_l26_26118


namespace frac_plus_a_ge_seven_l26_26571

theorem frac_plus_a_ge_seven (a : ℝ) (h : a > 3) : 4 / (a - 3) + a ≥ 7 := 
by
  sorry

end frac_plus_a_ge_seven_l26_26571


namespace rice_and_wheat_grains_division_l26_26124

-- Definitions for the conditions in the problem
def total_grains : ℕ := 1534
def sample_size : ℕ := 254
def wheat_in_sample : ℕ := 28

-- Proving the approximate amount of wheat grains in the batch  
theorem rice_and_wheat_grains_division : total_grains * (wheat_in_sample / sample_size) = 169 := by 
  sorry

end rice_and_wheat_grains_division_l26_26124


namespace solve_xyz_l26_26572

theorem solve_xyz (a b c : ℝ) (h1 : a = y + z) (h2 : b = x + z) (h3 : c = x + y) 
                   (h4 : 0 < y) (h5 : 0 < z) (h6 : 0 < x)
                   (hab : b + c > a) (hbc : a + c > b) (hca : a + b > c) :
  x = (b - a + c)/2 ∧ y = (a - b + c)/2 ∧ z = (a + b - c)/2 :=
by
  sorry

end solve_xyz_l26_26572


namespace polygon_divided_into_7_triangles_l26_26273

theorem polygon_divided_into_7_triangles (n : ℕ) (h : n - 2 = 7) : n = 9 :=
by
  sorry

end polygon_divided_into_7_triangles_l26_26273


namespace additional_discount_A_is_8_l26_26597

-- Define the problem conditions
def full_price_A : ℝ := 125
def full_price_B : ℝ := 130
def discount_B : ℝ := 0.10
def price_difference : ℝ := 2

-- Define the unknown additional discount of store A
def discount_A (x : ℝ) : Prop :=
  full_price_A - (full_price_A * (x / 100)) = (full_price_B - (full_price_B * discount_B)) - price_difference

-- Theorem stating that the additional discount offered by store A is 8%
theorem additional_discount_A_is_8 : discount_A 8 :=
by
  -- Proof can be filled in here
  sorry

end additional_discount_A_is_8_l26_26597


namespace exists_integers_A_B_C_l26_26489

theorem exists_integers_A_B_C (a b : ℚ) (N_star : Set ℕ) (Q : Set ℚ)
  (h : ∀ x ∈ N_star, (a * (x : ℚ) + b) / (x : ℚ) ∈ Q) : 
  ∃ A B C : ℤ, ∀ x ∈ N_star, 
    (a * (x : ℚ) + b) / (x : ℚ) = (A * (x : ℚ) + B) / (C * (x : ℚ)) := 
sorry

end exists_integers_A_B_C_l26_26489


namespace probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l26_26457

open ProbabilityTheory

noncomputable def Vanya_probability_more_heads_than_Tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  X_V 3 = binomial 3 (1/2) ∧
  X_T 2 = binomial 2 (1/2) ∧
  Pr(X_V > X_T) = 1/2

noncomputable def Vanya_probability_more_heads_than_Tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : Prop :=
  ∀ n : ℕ,
  X_V (n+1) = binomial (n+1) (1/2) ∧
  X_T n = binomial n (1/2) ∧
  Pr(X_V > X_T) = 1/2

theorem probability_heads_vanya_tanya_a (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_a X_V X_T := sorry

theorem probability_heads_vanya_tanya_b (X_V X_T : Nat → ProbabilityTheory ℕ) : 
  Vanya_probability_more_heads_than_Tanya_b X_V X_T := sorry

end probability_heads_vanya_tanya_a_probability_heads_vanya_tanya_b_l26_26457


namespace percent_diploma_thirty_l26_26536

-- Defining the conditions using Lean definitions

def percent_without_diploma_with_job := 0.10 -- 10%
def percent_with_job := 0.20 -- 20%
def percent_without_job_with_diploma :=
  (1 - percent_with_job) * 0.25 -- 25% of people without job is 25% of 80% which is 20%

def percent_with_diploma := percent_with_job - percent_without_diploma_with_job + percent_without_job_with_diploma

-- Theorem to prove that 30% of the people have a university diploma
theorem percent_diploma_thirty
  (H1 : percent_without_diploma_with_job = 0.10) -- condition 1
  (H2 : percent_with_job = 0.20) -- condition 3
  (H3 : percent_without_job_with_diploma = 0.20) -- evaluated from condition 2
  : percent_with_diploma = 0.30 := by
  -- prove that the percent with diploma is 30%
  sorry

end percent_diploma_thirty_l26_26536


namespace arithmetic_series_sum_l26_26950

theorem arithmetic_series_sum :
  let a1 := 5
  let an := 105
  let d := 1
  let n := (an - a1) / d + 1
  (n * (a1 + an) / 2) = 5555 := by
  sorry

end arithmetic_series_sum_l26_26950


namespace x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l26_26758

theorem x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5 :
  (∀ x : ℝ, x > 5 → x > 3) ∧ ¬(∀ x : ℝ, x > 3 → x > 5) :=
by 
  -- Prove implications with provided conditions
  sorry

end x_gt_3_is_necessary_but_not_sufficient_for_x_gt_5_l26_26758


namespace scientific_notation_l26_26132

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end scientific_notation_l26_26132


namespace equal_real_roots_eq_one_l26_26730

theorem equal_real_roots_eq_one (m : ℝ) : 
  (∀ x : ℝ, ∃ y : ℝ, y * y = x) ∧ (∀ x y : ℝ, x^2 - 2 * x + m = 0 ↔ (x = y) → b^2 - 4 * a * c = 0) → m = 1 := 
sorry

end equal_real_roots_eq_one_l26_26730


namespace greatest_gcd_of_6T_n_and_n_plus_1_l26_26663

theorem greatest_gcd_of_6T_n_and_n_plus_1 (n : ℕ) (h_pos : 0 < n) :
  let T_n := n * (n + 1) / 2 in
  gcd (6 * T_n) (n + 1) = 3 ↔ (n + 1) % 3 = 0 :=
by
  sorry

end greatest_gcd_of_6T_n_and_n_plus_1_l26_26663


namespace value_of_f_at_2_l26_26094

def f (x : ℝ) : ℝ := x^3 - x^2 - 1

theorem value_of_f_at_2 : f 2 = 3 := by
  sorry

end value_of_f_at_2_l26_26094


namespace cos_four_times_arccos_val_l26_26782

theorem cos_four_times_arccos_val : 
  ∀ x : ℝ, x = Real.arccos (1 / 4) → Real.cos (4 * x) = 17 / 32 :=
by
  intro x h
  sorry

end cos_four_times_arccos_val_l26_26782


namespace coin_flip_probability_l26_26762

open Classical BigOperators

/-- Given a fair coin that is flipped 8 times, the probability that exactly 6 flips result in heads is 7/64. -/
theorem coin_flip_probability :
  let total_outcomes := 2 ^ 8
  let favorable_outcomes := Nat.choose 8 6
  let probability := favorable_outcomes / (total_outcomes: ℚ)
  probability = 7 / 64 := by
  sorry

end coin_flip_probability_l26_26762


namespace polynomial_square_b_value_l26_26733

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end polynomial_square_b_value_l26_26733


namespace one_sofa_in_room_l26_26590

def num_sofas_in_room : ℕ :=
  let num_4_leg_tables := 4
  let num_4_leg_chairs := 2
  let num_3_leg_tables := 3
  let num_1_leg_table := 1
  let num_2_leg_rocking_chairs := 1
  let total_legs := 40

  let legs_of_4_leg_tables := num_4_leg_tables * 4
  let legs_of_4_leg_chairs := num_4_leg_chairs * 4
  let legs_of_3_leg_tables := num_3_leg_tables * 3
  let legs_of_1_leg_table := num_1_leg_table * 1
  let legs_of_2_leg_rocking_chairs := num_2_leg_rocking_chairs * 2

  let accounted_legs := legs_of_4_leg_tables + legs_of_4_leg_chairs + legs_of_3_leg_tables + legs_of_1_leg_table + legs_of_2_leg_rocking_chairs

  let remaining_legs := total_legs - accounted_legs

  let sofa_legs := 4
  remaining_legs / sofa_legs

theorem one_sofa_in_room : num_sofas_in_room = 1 :=
  by
    unfold num_sofas_in_room
    rfl

end one_sofa_in_room_l26_26590


namespace johnson_family_seating_l26_26880

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l26_26880


namespace five_minus_x_eight_l26_26519

theorem five_minus_x_eight (x y : ℤ) (h1 : 5 + x = 3 - y) (h2 : 2 + y = 6 + x) : 5 - x = 8 :=
by
  sorry

end five_minus_x_eight_l26_26519


namespace thomas_lost_pieces_l26_26647

theorem thomas_lost_pieces (audrey_lost : ℕ) (total_pieces_left : ℕ) (initial_pieces_each : ℕ) (total_pieces_initial : ℕ) (audrey_remaining_pieces : ℕ) (thomas_remaining_pieces : ℕ) : 
  audrey_lost = 6 → total_pieces_left = 21 → initial_pieces_each = 16 → total_pieces_initial = 32 → 
  audrey_remaining_pieces = initial_pieces_each - audrey_lost → 
  thomas_remaining_pieces = total_pieces_left - audrey_remaining_pieces → 
  initial_pieces_each - thomas_remaining_pieces = 5 :=
by
  sorry

end thomas_lost_pieces_l26_26647


namespace number_of_male_employees_l26_26308

theorem number_of_male_employees (num_female : ℕ) (x y : ℕ) 
  (h1 : 7 * x = y) 
  (h2 : 8 * x = num_female) 
  (h3 : 9 * (7 * x + 3) = 8 * num_female) :
  y = 189 := by
  sorry

end number_of_male_employees_l26_26308


namespace johnson_family_seating_l26_26879

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l26_26879


namespace hall_length_l26_26468

theorem hall_length (L : ℝ) (H : ℝ) 
  (h1 : 2 * (L * 15) = 2 * (L * H) + 2 * (15 * H)) 
  (h2 : L * 15 * H = 1687.5) : 
  L = 15 :=
by 
  sorry

end hall_length_l26_26468


namespace not_equal_factorial_l26_26772

noncomputable def permutations (n k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

theorem not_equal_factorial (n : ℕ) :
  permutations (n + 1) n ≠ (by apply Nat.factorial n) := by
  sorry

end not_equal_factorial_l26_26772


namespace linear_dependent_vectors_l26_26013

variable (m : ℝ) (a b : ℝ) 

theorem linear_dependent_vectors :
  (∃ (a b : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ 
    a • (⟨2, 3⟩ : ℝ × ℝ) + b • (⟨5, m⟩ : ℝ × ℝ) = (⟨0, 0⟩ : ℝ × ℝ)) ↔ m = 15 / 2 :=
sorry

end linear_dependent_vectors_l26_26013


namespace find_missing_number_l26_26602

theorem find_missing_number (n x : ℕ) (h : n * (n + 1) / 2 - x = 2012) : x = 4 := by
  sorry

end find_missing_number_l26_26602


namespace part1_part2_l26_26837

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l26_26837


namespace multiply_5915581_7907_l26_26774

theorem multiply_5915581_7907 : 5915581 * 7907 = 46757653387 := 
by
  -- sorry is used here to skip the proof
  sorry

end multiply_5915581_7907_l26_26774


namespace number_of_subsets_of_starOperation_l26_26652

open Set

def A : Set ℕ := {1, 3, 5, 7}
def B : Set ℕ := {2, 3, 5}

def starOperation (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem number_of_subsets_of_starOperation :
  starOperation A B = {1, 7} →
  (λ S, S ⊆ starOperation A B) '' (univ : Set (Set ℕ)).toFinset.card = 4 := by
  intro h
  rw h
  exact sorry

end number_of_subsets_of_starOperation_l26_26652


namespace amount_paid_to_z_l26_26757

-- Definitions based on conditions
def work_per_day (d : ℕ) : ℚ := 1 / d

def x_work_rate : ℚ := work_per_day 15
def y_work_rate : ℚ := work_per_day 10

-- Total payment and collective work rate definitions
def total_payment : ℚ := 720
def combined_work_rate : ℚ := work_per_day 5

-- Needed as part of the definition that will be used in the proof
def z_work_rate : ℚ := combined_work_rate - x_work_rate - y_work_rate

-- Verification statement to show the amount paid to z is Rs. 120
theorem amount_paid_to_z :
  z_work_rate * total_payment / (x_work_rate + y_work_rate + z_work_rate) = 120 := 
by
  sorry

end amount_paid_to_z_l26_26757


namespace frac_sum_equals_seven_eights_l26_26974

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end frac_sum_equals_seven_eights_l26_26974


namespace count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l26_26913

def is_progressive_number (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 d5 : ℕ), 1 ≤ d1 ∧ d1 < d2 ∧ d2 < d3 ∧ d3 < d4 ∧ d4 < d5 ∧ d5 ≤ 9 ∧
                          n = d1 * 10000 + d2 * 1000 + d3 * 100 + d4 * 10 + d5

theorem count_five_digit_progressive_numbers : ∃ n, n = 126 :=
by
  sorry

theorem find_110th_five_digit_progressive_number : ∃ n, n = 34579 :=
by
  sorry

end count_five_digit_progressive_numbers_find_110th_five_digit_progressive_number_l26_26913


namespace first_number_is_210_l26_26885

theorem first_number_is_210 (A B hcf lcm : ℕ) (h1 : lcm = 2310) (h2: hcf = 47) (h3 : B = 517) :
  A * B = lcm * hcf → A = 210 :=
by
  sorry

end first_number_is_210_l26_26885


namespace solve_stamps_l26_26081

noncomputable def stamps_problem : Prop :=
  ∃ (A B C D : ℝ), 
    A + B + C + D = 251 ∧
    A = 2 * B + 2 ∧
    A = 3 * C + 6 ∧
    A = 4 * D - 16 ∧
    D = 32

theorem solve_stamps : stamps_problem :=
sorry

end solve_stamps_l26_26081


namespace arithmetic_expression_l26_26341

theorem arithmetic_expression : 125 - 25 * 4 = 25 := 
by
  sorry

end arithmetic_expression_l26_26341


namespace number_of_sets_of_popcorn_l26_26470

theorem number_of_sets_of_popcorn (t p s : ℝ) (k : ℕ) 
  (h1 : t = 5)
  (h2 : p = 0.80 * t)
  (h3 : s = 0.50 * p)
  (h4 : 4 * t + 4 * s + k * p = 36) :
  k = 2 :=
by sorry

end number_of_sets_of_popcorn_l26_26470


namespace greatest_perimeter_isosceles_triangle_l26_26054

theorem greatest_perimeter_isosceles_triangle :
  let base := 12
  let height := 15
  let segments := 6
  let max_perimeter := 32.97
  -- Assuming division such that each of the 6 pieces is of equal area,
  -- the greatest perimeter among these pieces to the nearest hundredth is:
  (∀ (base height segments : ℝ), base = 12 ∧ height = 15 ∧ segments = 6 → 
   max_perimeter = 32.97) :=
by
  sorry

end greatest_perimeter_isosceles_triangle_l26_26054


namespace pairs_of_old_roller_skates_l26_26555

def cars := 2
def bikes := 2
def trash_can := 1
def tricycle := 1
def car_wheels := 4
def bike_wheels := 2
def trash_can_wheels := 2
def tricycle_wheels := 3
def total_wheels := 25

def roller_skates_wheels := 2
def skates_per_pair := 2

theorem pairs_of_old_roller_skates : (total_wheels - (cars * car_wheels + bikes * bike_wheels + trash_can * trash_can_wheels + tricycle * tricycle_wheels)) / roller_skates_wheels / skates_per_pair = 2 := by
  sorry

end pairs_of_old_roller_skates_l26_26555


namespace min_sum_abc_l26_26282

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end min_sum_abc_l26_26282


namespace frog_probability_l26_26319

-- Definition of the vertices of the square
def vertices : List (ℕ × ℕ) := [(0, 0), (0, 4), (4, 4), (4, 0)]

-- Definitions of points and their probabilities
def P (x y : ℕ) : ℚ := sorry

-- The starting point for our problem
def start_point : ℕ × ℕ := (1, 2)

-- Conditions extracted from the problem
def move_parallel_to_axis : bool := true
def jump_length : ℕ := 1
def directions_independent : bool := true
def within_square (x y : ℕ) : bool := x ≤ 4 ∧ y ≤ 4

-- The theorem that proves the probability 
theorem frog_probability : P 1 2 = 5/8 :=
  sorry

end frog_probability_l26_26319


namespace maximize_average_distance_l26_26482

open Classical

noncomputable def maximize_distance (s : ℕ) (n : ℕ) : Prop :=
  s = 4 ∧ (∀ i, if i = n then True else False) ∧ n = 4

theorem maximize_average_distance : 
  ∃ n, maximize_distance 4 n :=
by
  have hyp : maximize_distance 4 4,
  {
    split,
    {
      exact rfl,
    },
    split,
    {
      intro i,
      split_ifs,
      exact True.intro,
      contradiction,
    },
    {
      exact rfl,
    }
  },
  exact ⟨4, hyp⟩

end maximize_average_distance_l26_26482


namespace keith_turnips_l26_26128

theorem keith_turnips (a t k : ℕ) (h1 : a = 9) (h2 : t = 15) : k = t - a := by
  sorry

end keith_turnips_l26_26128


namespace jack_can_return_3900_dollars_l26_26851

/-- Jack's Initial Gift Card Values and Counts --/
def best_buy_card_value : ℕ := 500
def walmart_card_value : ℕ := 200
def initial_best_buy_cards : ℕ := 6
def initial_walmart_cards : ℕ := 9

/-- Jack's Sent Gift Card Counts --/
def sent_best_buy_cards : ℕ := 1
def sent_walmart_cards : ℕ := 2

/-- Calculate the remaining dollar value of Jack's gift cards. --/
def remaining_gift_cards_value : ℕ := 
  (initial_best_buy_cards * best_buy_card_value - sent_best_buy_cards * best_buy_card_value) +
  (initial_walmart_cards * walmart_card_value - sent_walmart_cards * walmart_card_value)

/-- Proving the remaining value of gift cards Jack can return is $3900. --/
theorem jack_can_return_3900_dollars : remaining_gift_cards_value = 3900 := by
  sorry

end jack_can_return_3900_dollars_l26_26851


namespace part1_part2_l26_26809

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l26_26809


namespace directrix_of_parabola_l26_26279

theorem directrix_of_parabola (p : ℝ) : by { assume h : y² = 4 * p * x, sorry } :=
assume h₁ : y² = 2 * x,
have hp : p = 1 / 2 , from sorry,
have directrix_eq : x = -p, from sorry,
show x = -1 / 2, from sorry

end directrix_of_parabola_l26_26279


namespace product_of_two_numbers_is_21_l26_26587

noncomputable def product_of_two_numbers (x y : ℝ) : ℝ :=
  x * y

theorem product_of_two_numbers_is_21 (x y : ℝ) (h₁ : x + y = 10) (h₂ : x^2 + y^2 = 58) :
  product_of_two_numbers x y = 21 :=
by sorry

end product_of_two_numbers_is_21_l26_26587


namespace part1_part2_l26_26824

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l26_26824


namespace log_identity_l26_26939

theorem log_identity :
  (Real.log 25 / Real.log 10) - 2 * (Real.log (1 / 2) / Real.log 10) = 2 :=
by
  sorry

end log_identity_l26_26939


namespace sector_area_l26_26511

theorem sector_area (r α l S : ℝ) (h1 : l + 2 * r = 8) (h2 : α = 2) (h3 : l = α * r) :
  S = 4 :=
by
  -- Let the radius be 2 as a condition derived from h1 and h2
  have r := 2
  -- Substitute and compute to find S
  have S_calculated := (1 / 2 * α * r * r)
  sorry

end sector_area_l26_26511


namespace percent_cities_less_than_50000_l26_26278

-- Definitions of the conditions
def percent_cities_50000_to_149999 := 40
def percent_cities_less_than_10000 := 35
def percent_cities_10000_to_49999 := 10
def percent_cities_150000_or_more := 15

-- Prove that the total percentage of cities with fewer than 50,000 residents is 45%
theorem percent_cities_less_than_50000 :
  percent_cities_less_than_10000 + percent_cities_10000_to_49999 = 45 :=
by
  sorry

end percent_cities_less_than_50000_l26_26278


namespace problem_l26_26690

theorem problem (a b : ℝ) (h1 : |a - 2| + (b + 1)^2 = 0) : a - b = 3 := by
  sorry

end problem_l26_26690


namespace total_seeds_planted_l26_26138

theorem total_seeds_planted 
    (seeds_per_bed : ℕ) 
    (seeds_grow_per_bed : ℕ) 
    (total_flowers : ℕ) 
    (h1 : seeds_per_bed = 15) 
    (h2 : seeds_grow_per_bed = 60) 
    (h3 : total_flowers = 220) : 
    ∃ (total_seeds : ℕ), total_seeds = 85 := 
by
    sorry

end total_seeds_planted_l26_26138


namespace compute_fraction_product_l26_26938

theorem compute_fraction_product :
  (∏ i in (finset.range 5).map (λ n, n + 3), (i ^ 3 - 1) / (i ^ 3 + 1)) = (57 / 168) := by
  sorry

end compute_fraction_product_l26_26938


namespace correct_system_equations_l26_26173

theorem correct_system_equations (x y : ℤ) : 
  (8 * x - y = 3) ∧ (y - 7 * x = 4) ↔ 
    (8 * x - y = 3) ∧ (y - 7 * x = 4) := by
  sorry

end correct_system_equations_l26_26173


namespace cistern_empty_time_without_tap_l26_26180

noncomputable def leak_rate (L : ℕ) : Prop :=
  let tap_rate := 4
  let cistern_volume := 480
  let empty_time_with_tap := 24
  let empty_rate_net := cistern_volume / empty_time_with_tap
  L - tap_rate = empty_rate_net

theorem cistern_empty_time_without_tap (L : ℕ) (h : leak_rate L) :
  480 / L = 20 := by
  -- placeholder for the proof
  sorry

end cistern_empty_time_without_tap_l26_26180


namespace part1_part2_l26_26820

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l26_26820


namespace pens_left_in_jar_l26_26630

theorem pens_left_in_jar : 
  ∀ (initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens : ℕ),
  initial_blue_pens = 9 →
  initial_black_pens = 21 →
  initial_red_pens = 6 →
  removed_blue_pens = 4 →
  removed_black_pens = 7 →
  (initial_blue_pens - removed_blue_pens) + (initial_black_pens - removed_black_pens) + initial_red_pens = 25 :=
begin
  intros initial_blue_pens initial_black_pens initial_red_pens removed_blue_pens removed_black_pens,
  intros h1 h2 h3 h4 h5,
  simp [h1, h2, h3, h4, h5],
  norm_num,
end

end pens_left_in_jar_l26_26630


namespace find_b_l26_26583

noncomputable def Q (x : ℝ) (a b c : ℝ) := 3 * x ^ 3 + a * x ^ 2 + b * x + c

theorem find_b (a b c : ℝ) (h₀ : c = 6) 
  (h₁ : ∃ (r₁ r₂ r₃ : ℝ), Q r₁ a b c = 0 ∧ Q r₂ a b c = 0 ∧ Q r₃ a b c = 0 ∧ (r₁ + r₂ + r₃) / 3 = -(c / 3) ∧ r₁ * r₂ * r₃ = -(c / 3))
  (h₂ : 3 + a + b + c = -(c / 3)): 
  b = -29 :=
sorry

end find_b_l26_26583


namespace angle_E_measure_l26_26127

theorem angle_E_measure {D E F : Type} (angle_D angle_E angle_F : ℝ) 
  (h1 : angle_E = angle_F)
  (h2 : angle_F = 3 * angle_D)
  (h3 : angle_D = (1/2) * angle_E) 
  (h_sum : angle_D + angle_E + angle_F = 180) :
  angle_E = 540 / 7 := 
by
  sorry

end angle_E_measure_l26_26127


namespace remainder_when_divided_by_eleven_l26_26363

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_eleven_l26_26363


namespace Loisa_saves_70_l26_26713

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end Loisa_saves_70_l26_26713


namespace quadrilateral_identity_l26_26986

theorem quadrilateral_identity 
  {A B C D : Type*} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D]
  (AB : ℝ) (BC : ℝ) (CD : ℝ) (DA : ℝ) (AC : ℝ) (BD : ℝ)
  (angle_A : ℝ) (angle_C : ℝ) 
  (h_angle_sum : angle_A + angle_C = 120)
  : (AC * BD)^2 = (AB * CD)^2 + (BC * AD)^2 + AB * BC * CD * DA := 
by {
  sorry
}

end quadrilateral_identity_l26_26986


namespace remainder_of_product_mod_12_l26_26751

-- Define the given constants
def a := 1125
def b := 1127
def c := 1129
def d := 12

-- State the conditions as Lean hypotheses
lemma mod_eq_1125 : a % d = 9 := by sorry
lemma mod_eq_1127 : b % d = 11 := by sorry
lemma mod_eq_1129 : c % d = 1 := by sorry

-- Define the theorem to prove
theorem remainder_of_product_mod_12 : (a * b * c) % d = 3 := by
  -- Use the conditions stated above to prove the theorem
  sorry

end remainder_of_product_mod_12_l26_26751


namespace sales_not_books_magazines_stationery_l26_26142

variable (books_sales : ℕ := 45)
variable (magazines_sales : ℕ := 30)
variable (stationery_sales : ℕ := 10)
variable (total_sales : ℕ := 100)

theorem sales_not_books_magazines_stationery : 
  books_sales + magazines_sales + stationery_sales < total_sales → 
  total_sales - (books_sales + magazines_sales + stationery_sales) = 15 :=
by
  sorry

end sales_not_books_magazines_stationery_l26_26142


namespace closest_perfect_square_to_325_is_324_l26_26453

theorem closest_perfect_square_to_325_is_324 :
  ∃ n : ℕ, n^2 = 324 ∧ (∀ m : ℕ, m * m ≠ 325) ∧
    (n = 18 ∧ (∀ k : ℕ, (k*k < 325 ∧ (325 - k*k) > 325 - 324) ∨ 
               (k*k > 325 ∧ (k*k - 325) > 361 - 325))) :=
by
  sorry

end closest_perfect_square_to_325_is_324_l26_26453


namespace sachin_borrowed_amount_l26_26268

variable (P : ℝ) (gain : ℝ)
variable (interest_rate_borrow : ℝ := 4 / 100)
variable (interest_rate_lend : ℝ := 25 / 4 / 100)
variable (time_period : ℝ := 2)
variable (gain_provided : ℝ := 112.5)

theorem sachin_borrowed_amount (h : gain = 0.0225 * P) : P = 5000 :=
by sorry

end sachin_borrowed_amount_l26_26268


namespace population_of_missing_village_eq_945_l26_26584

theorem population_of_missing_village_eq_945
  (pop1 pop2 pop3 pop4 pop5 pop6 : ℕ)
  (avg_pop total_population missing_population : ℕ)
  (h1 : pop1 = 803)
  (h2 : pop2 = 900)
  (h3 : pop3 = 1100)
  (h4 : pop4 = 1023)
  (h5 : pop5 = 980)
  (h6 : pop6 = 1249)
  (h_avg : avg_pop = 1000)
  (h_total_population : total_population = avg_pop * 7)
  (h_missing_population : missing_population = total_population - (pop1 + pop2 + pop3 + pop4 + pop5 + pop6)) :
  missing_population = 945 :=
by {
  -- Here would go the proof steps if needed
  sorry 
}

end population_of_missing_village_eq_945_l26_26584


namespace perpendicular_line_eq_l26_26475

theorem perpendicular_line_eq (a b : ℝ) (ha : 2 * a - 5 * b + 3 = 0) (hpt : a = 2 ∧ b = -1) : 
    ∃ c : ℝ, c = 5 * a + 2 * b - 8 := 
sorry

end perpendicular_line_eq_l26_26475


namespace point_B_in_first_quadrant_l26_26539

theorem point_B_in_first_quadrant 
  (a b : ℝ) 
  (h1 : a > 0) 
  (h2 : -b > 0) : 
  (a > 0) ∧ (b > 0) := 
by 
  sorry

end point_B_in_first_quadrant_l26_26539


namespace warehouse_rental_comparison_purchase_vs_rent_comparison_l26_26627

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end warehouse_rental_comparison_purchase_vs_rent_comparison_l26_26627


namespace fraction_of_coins_1800_to_1809_l26_26865

theorem fraction_of_coins_1800_to_1809
  (total_coins : ℕ)
  (coins_1800_1809 : ℕ)
  (h_total : total_coins = 22)
  (h_coins : coins_1800_1809 = 5) :
  (coins_1800_1809 : ℚ) / total_coins = 5 / 22 := by
  sorry

end fraction_of_coins_1800_to_1809_l26_26865


namespace part1_part2_l26_26832

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l26_26832


namespace find_values_of_a_and_b_l26_26084

theorem find_values_of_a_and_b
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (x : ℝ) (hx : x > 1)
  (h : 9 * (Real.log x / Real.log a)^2 + 5 * (Real.log x / Real.log b)^2 = 17)
  (h2 : (Real.log b / Real.log a) * (Real.log a / Real.log b) = 2) :
  a = 10 ^ Real.sqrt 2 ∧ b = 10 := by
sorry

end find_values_of_a_and_b_l26_26084


namespace birds_meeting_distance_l26_26148

theorem birds_meeting_distance :
  ∀ (d distance speed1 speed2: ℕ),
  distance = 20 →
  speed1 = 4 →
  speed2 = 1 →
  (d / speed1) = ((distance - d) / speed2) →
  d = 16 :=
by
  intros d distance speed1 speed2 hdist hspeed1 hspeed2 htime
  sorry

end birds_meeting_distance_l26_26148


namespace company_spends_less_l26_26317

noncomputable def total_spending_reduction_in_dollars : ℝ :=
  let magazine_initial_cost := 840.00
  let online_resources_initial_cost_gbp := 960.00
  let exchange_rate := 1.40
  let mag_cut_percentage := 0.30
  let online_cut_percentage := 0.20

  let magazine_cost_cut := magazine_initial_cost * mag_cut_percentage
  let online_resource_cost_cut_gbp := online_resources_initial_cost_gbp * online_cut_percentage
  
  let new_magazine_cost := magazine_initial_cost - magazine_cost_cut
  let new_online_resource_cost_gbp := online_resources_initial_cost_gbp - online_resource_cost_cut_gbp

  let online_resources_initial_cost := online_resources_initial_cost_gbp * exchange_rate
  let new_online_resource_cost := new_online_resource_cost_gbp * exchange_rate

  let mag_cut_amount := magazine_initial_cost - new_magazine_cost
  let online_cut_amount := online_resources_initial_cost - new_online_resource_cost
  
  mag_cut_amount + online_cut_amount

theorem company_spends_less : total_spending_reduction_in_dollars = 520.80 :=
by
  sorry

end company_spends_less_l26_26317


namespace Tim_total_money_l26_26443

theorem Tim_total_money :
  let nickels_amount := 3 * 0.05
  let dimes_amount_shoes := 13 * 0.10
  let shining_shoes := nickels_amount + dimes_amount_shoes
  let dimes_amount_tip_jar := 7 * 0.10
  let half_dollars_amount := 9 * 0.50
  let tip_jar := dimes_amount_tip_jar + half_dollars_amount
  let total := shining_shoes + tip_jar
  total = 6.65 :=
by
  sorry

end Tim_total_money_l26_26443


namespace baby_grasshoppers_l26_26552

-- Definition for the number of grasshoppers on the plant
def grasshoppers_on_plant : ℕ := 7

-- Definition for the total number of grasshoppers found
def total_grasshoppers : ℕ := 31

-- The theorem to prove the number of baby grasshoppers under the plant
theorem baby_grasshoppers : 
  (total_grasshoppers - grasshoppers_on_plant) = 24 := 
by
  sorry

end baby_grasshoppers_l26_26552


namespace dandelions_surviving_to_flower_l26_26207

/-- 
Each dandelion produces 300 seeds. 
1/3rd of the seeds land in water and die. 
1/6 of the starting number are eaten by insects. 
Half the remainder sprout and are immediately eaten.
-/
def starting_seeds : ℕ := 300

def seeds_lost_to_water : ℕ := starting_seeds / 3
def seeds_after_water : ℕ := starting_seeds - seeds_lost_to_water

def seeds_eaten_by_insects : ℕ := starting_seeds / 6
def seeds_after_insects : ℕ := seeds_after_water - seeds_eaten_by_insects

def seeds_eaten_after_sprouting : ℕ := seeds_after_insects / 2
def seeds_surviving : ℕ := seeds_after_insects - seeds_eaten_after_sprouting

theorem dandelions_surviving_to_flower 
  (starting_seeds = 300) 
  (seeds_lost_to_water = starting_seeds / 3) 
  (seeds_after_water = starting_seeds - seeds_lost_to_water) 
  (seeds_eaten_by_insects = starting_seeds / 6) 
  (seeds_after_insects = seeds_after_water - seeds_eaten_by_insects) 
  (seeds_eaten_after_sprouting = seeds_after_insects / 2) 
  (seeds_surviving = seeds_after_insects - seeds_eaten_after_sprouting) : 
  seeds_surviving = 75 := 
sorry

end dandelions_surviving_to_flower_l26_26207


namespace cheese_cookies_price_is_correct_l26_26474

-- Define the problem conditions and constants
def total_boxes_per_carton : ℕ := 15
def total_packs_per_box : ℕ := 12
def discount_15_percent : ℝ := 0.15
def total_number_of_cartons : ℕ := 13
def total_cost_paid : ℝ := 2058

-- Calculate the expected price per pack
noncomputable def price_per_pack : ℝ :=
  let total_packs := total_boxes_per_carton * total_packs_per_box * total_number_of_cartons
  let total_cost_without_discount := total_cost_paid / (1 - discount_15_percent)
  total_cost_without_discount / total_packs

theorem cheese_cookies_price_is_correct : 
  abs (price_per_pack - 1.0347) < 0.0001 :=
by sorry

end cheese_cookies_price_is_correct_l26_26474


namespace elderly_sample_correct_l26_26760

-- Conditions
def young_employees : ℕ := 300
def middle_aged_employees : ℕ := 150
def elderly_employees : ℕ := 100
def total_employees : ℕ := young_employees + middle_aged_employees + elderly_employees
def sample_size : ℕ := 33
def elderly_sample (total : ℕ) (elderly : ℕ) (sample : ℕ) : ℕ := (sample * elderly) / total

-- Statement to prove
theorem elderly_sample_correct :
  elderly_sample total_employees elderly_employees sample_size = 6 := 
by
  sorry

end elderly_sample_correct_l26_26760


namespace complement_of_intersection_l26_26966

-- Definitions of the sets M and N
def M : Set ℝ := { x | x ≥ 2 }
def N : Set ℝ := { x | x < 3 }

-- Definition of the intersection of M and N
def M_inter_N : Set ℝ := { x | 2 ≤ x ∧ x < 3 }

-- Definition of the complement of M ∩ N in ℝ
def complement_M_inter_N : Set ℝ := { x | x < 2 ∨ x ≥ 3 }

-- The theorem to be proved
theorem complement_of_intersection :
  (M_inter_Nᶜ) = complement_M_inter_N :=
by sorry

end complement_of_intersection_l26_26966


namespace marriage_year_proof_l26_26181

-- Definitions based on conditions
def marriage_year : ℕ := sorry
def child1_birth_year : ℕ := 1982
def child2_birth_year : ℕ := 1984
def reference_year : ℕ := 1986

-- Age calculations based on reference year
def age_in_1986 (birth_year : ℕ) : ℕ := reference_year - birth_year

-- Combined ages in the reference year
def combined_ages_in_1986 : ℕ := age_in_1986 child1_birth_year + age_in_1986 child2_birth_year

-- The main theorem to prove
theorem marriage_year_proof :
  combined_ages_in_1986 = reference_year - marriage_year →
  marriage_year = 1980 := by
  sorry

end marriage_year_proof_l26_26181


namespace simplify_expression_l26_26606

theorem simplify_expression (b : ℝ) (hb : b = -1) : 
  (3 * b⁻¹ + (2 * b⁻¹) / 3) / b = 11 / 3 :=
by
  sorry

end simplify_expression_l26_26606


namespace jenny_total_distance_seven_hops_l26_26992

noncomputable def sum_geometric_series (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem jenny_total_distance_seven_hops :
  let a := (1 / 4 : ℚ)
  let r := (3 / 4 : ℚ)
  let n := 7
  sum_geometric_series a r n = (14197 / 16384 : ℚ) :=
by
  sorry

end jenny_total_distance_seven_hops_l26_26992


namespace number_of_rabbits_l26_26759

-- Defining the problem conditions
variables (x y : ℕ)
axiom heads_condition : x + y = 40
axiom legs_condition : 4 * x = 10 * 2 * y - 8

--  Prove the number of rabbits is 33
theorem number_of_rabbits : x = 33 :=
by
  sorry

end number_of_rabbits_l26_26759


namespace loisa_saves_70_l26_26711

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end loisa_saves_70_l26_26711


namespace max_remainder_when_divided_by_7_l26_26388

theorem max_remainder_when_divided_by_7 (y : ℕ) (r : ℕ) (h : r = y % 7) : r ≤ 6 ∧ ∃ k, y = 7 * k + r :=
by
  sorry

end max_remainder_when_divided_by_7_l26_26388


namespace value_standard_deviations_from_mean_l26_26144

-- Define the mean (µ)
def μ : ℝ := 15.5

-- Define the standard deviation (σ)
def σ : ℝ := 1.5

-- Define the value X
def X : ℝ := 12.5

-- Prove that the Z-score is -2
theorem value_standard_deviations_from_mean : (X - μ) / σ = -2 := by
  sorry

end value_standard_deviations_from_mean_l26_26144


namespace part1_part2_l26_26825

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l26_26825


namespace tan_double_angle_cos_beta_l26_26795

theorem tan_double_angle (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.tan (2 * α) = -(8 * Real.sqrt 3) / 47 :=
  sorry

theorem cos_beta (α β : ℝ) (h1 : Real.sin α = 4 * Real.sqrt 3 / 7) 
  (h2 : Real.cos (β - α) = 13 / 14) (h3 : 0 < β ∧ β < α ∧ α < Real.pi / 2) : 
  Real.cos β = 1 / 2 :=
  sorry

end tan_double_angle_cos_beta_l26_26795


namespace inequality_part1_inequality_part2_l26_26817

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l26_26817


namespace no_discrepancy_l26_26157

-- Definitions based on the conditions
def t1_hours : ℝ := 1.5 -- time taken clockwise in hours
def t2_minutes : ℝ := 90 -- time taken counterclockwise in minutes

-- Lean statement to prove the equivalence
theorem no_discrepancy : t1_hours * 60 = t2_minutes :=
by sorry

end no_discrepancy_l26_26157


namespace number_of_puppies_l26_26016

theorem number_of_puppies (P K : ℕ) (h1 : K = 2 * P + 14) (h2 : K = 78) : P = 32 :=
by sorry

end number_of_puppies_l26_26016


namespace even_marked_squares_9x9_l26_26378

open Nat

theorem even_marked_squares_9x9 :
  let n := 9
  let total_squares := n * n
  let odd_rows_columns := [1, 3, 5, 7, 9]
  let odd_squares := odd_rows_columns.length * odd_rows_columns.length
  total_squares - odd_squares = 56 :=
by
  sorry

end even_marked_squares_9x9_l26_26378


namespace intersection_empty_l26_26249

open Set

-- Definition of set A
def A : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 2 * x + 3) }

-- Definition of set B
def B : Set (ℝ × ℝ) := { p | ∃ x : ℝ, p = (x, 4 * x + 1) }

-- The proof problem statement in Lean
theorem intersection_empty : A ∩ B = ∅ := sorry

end intersection_empty_l26_26249


namespace problem_1_problem_2_l26_26505

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

end problem_1_problem_2_l26_26505


namespace binomial_sum_mod_1000_l26_26052

open BigOperators

theorem binomial_sum_mod_1000 :
  ((∑ k in finset.range 503 \ finset.range 3, nat.choose 2011 (4 * k)) % 1000) = 49 := 
sorry

end binomial_sum_mod_1000_l26_26052


namespace point_D_eq_1_2_l26_26381

-- Definitions and conditions
def point : Type := ℝ × ℝ

def A : point := (-1, 4)
def B : point := (-4, -1)
def C : point := (4, 7)

-- Translate function
def translate (p : point) (dx dy : ℝ) := (p.1 + dx, p.2 + dy)

-- The translation distances found from A to C
def dx := C.1 - A.1
def dy := C.2 - A.2

-- The point D
def D : point := translate B dx dy

-- Proof objective
theorem point_D_eq_1_2 : D = (1, 2) := by
  sorry

end point_D_eq_1_2_l26_26381


namespace part1_part2_l26_26811

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l26_26811


namespace find_value_of_a_l26_26978

-- Definitions based on the conditions
def x (k : ℕ) : ℕ := 3 * k
def y (k : ℕ) : ℕ := 4 * k
def z (k : ℕ) : ℕ := 6 * k

-- Setting up the sum equation
def sum_eq_52 (k : ℕ) : Prop := x k + y k + z k = 52

-- Defining the y equation
def y_eq (a : ℚ) (k : ℕ) : Prop := y k = 15 * a + 5

-- Stating the main problem
theorem find_value_of_a (a : ℚ) (k : ℕ) : sum_eq_52 k → y_eq a k → a = 11 / 15 := by
  sorry

end find_value_of_a_l26_26978


namespace tractor_planting_rate_l26_26069

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l26_26069


namespace find_k_and_f_min_total_cost_l26_26021

-- Define the conditions
def construction_cost (x : ℝ) : ℝ := 60 * x
def energy_consumption_cost (x : ℝ) : ℝ := 40 - 4 * x
def total_cost (x : ℝ) : ℝ := construction_cost x + 20 * energy_consumption_cost x

theorem find_k_and_f :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → energy_consumption_cost 0 = 8 → energy_consumption_cost x = 40 - 4 * x) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 10 → total_cost x = 800 - 74 * x) :=
by
  sorry

theorem min_total_cost :
  (∀ x, 0 ≤ x ∧ x ≤ 10 → 800 - 74 * x ≥ 70) ∧
  total_cost 5 = 70 :=
by
  sorry

end find_k_and_f_min_total_cost_l26_26021


namespace total_students_is_46_l26_26244

-- Define the constants for the problem
def students_in_history : ℕ := 19
def students_in_math : ℕ := 14
def students_in_english : ℕ := 26
def students_in_all_three : ℕ := 3
def students_in_exactly_two : ℕ := 7

-- The total number of students as per the inclusion-exclusion principle
def total_students : ℕ :=
  students_in_history + students_in_math + students_in_english
  - students_in_exactly_two - 2 * students_in_all_three + students_in_all_three

theorem total_students_is_46 : total_students = 46 :=
  sorry

end total_students_is_46_l26_26244


namespace negation_of_implication_l26_26009

theorem negation_of_implication (x : ℝ) :
  ¬ (x > 1 → x^2 > 1) ↔ (x ≤ 1 → x^2 ≤ 1) :=
by
  sorry

end negation_of_implication_l26_26009


namespace JohnsonFamilySeating_l26_26882

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l26_26882


namespace determine_g_l26_26732

theorem determine_g (t : ℝ) : ∃ (g : ℝ → ℝ), (∀ x y, y = 2 * x - 40 ∧ y = 20 * t - 14 → g t = 10 * t + 13) :=
by
  sorry

end determine_g_l26_26732


namespace missy_total_watching_time_l26_26258

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end missy_total_watching_time_l26_26258


namespace find_b_l26_26741

noncomputable def f (x : ℝ) : ℝ := (x+1)^3 + (x / (x + 1))

theorem find_b (b : ℝ) (h_sum : ∃ x1 x2 : ℝ, f x1 = -x1 + b ∧ f x2 = -x2 + b ∧ x1 + x2 = -2) : b = 0 :=
by
  sorry

end find_b_l26_26741


namespace water_fraction_final_l26_26466

theorem water_fraction_final (initial_volume : ℚ) (removed_volume : ℚ) (replacements : ℕ) (water_initial_fraction : ℚ) :
  initial_volume = 20 ∧ removed_volume = 5 ∧ replacements = 5 ∧ water_initial_fraction = 1 ->
  let water_fraction := water_initial_fraction * (3 / 4)^replacements in
  water_fraction = 243 / 1024 :=
by
  sorry

end water_fraction_final_l26_26466


namespace number_of_children_l26_26546

-- Define the given conditions in Lean 4
variable {a : ℕ}
variable {R : ℕ}
variable {L : ℕ}
variable {k : ℕ}

-- Conditions given in the problem
def condition1 : 200 ≤ a ∧ a ≤ 300 := sorry
def condition2 : a = 25 * R + 10 := sorry
def condition3 : a = 30 * L - 15 := sorry 
def condition4 : a + 15 = 150 * k := sorry

-- The theorem to prove
theorem number_of_children : a = 285 :=
by
  assume a R L k // This assumption is for the variables needed.
  have h₁ : condition1 := sorry
  have h₂ : condition2 := sorry
  have h₃ : condition3 := sorry
  have h₄ : condition4 := sorry 
  exact sorry

end number_of_children_l26_26546


namespace perimeter_square_C_l26_26272

theorem perimeter_square_C 
  (a b c : ℝ) 
  (ha : 4 * a = 16) 
  (hb : 4 * b = 28) 
  (hc : c = |a - b|) : 
  4 * c = 12 := 
sorry

end perimeter_square_C_l26_26272


namespace parabola_properties_l26_26433

def parabola (a b x : ℝ) : ℝ :=
  a * x ^ 2 + b * x - 4

theorem parabola_properties :
  ∃ (a b : ℝ), (a = 2) ∧ (b = 2) ∧
  parabola a b (-2) = 0 ∧ 
  parabola a b (-1) = -4 ∧ 
  parabola a b 0 = -4 ∧ 
  parabola a b 1 = 0 ∧ 
  parabola a b 2 = 8 ∧ 
  parabola a b (-3) = 8 ∧ 
  (0, -4) ∈ {(x, y) | ∃ a b, y = parabola a b x} :=
sorry

end parabola_properties_l26_26433


namespace binom_coeff_divisibility_l26_26420

theorem binom_coeff_divisibility (p : ℕ) (hp : Prime p) : Nat.choose (2 * p) p - 2 ≡ 0 [MOD p^2] := 
sorry

end binom_coeff_divisibility_l26_26420


namespace perimeter_of_larger_triangle_is_65_l26_26335

noncomputable def similar_triangle_perimeter : ℝ :=
  let a := 7
  let b := 7
  let c := 12
  let longest_side_similar := 30
  let perimeter_small := a + b + c
  let ratio := longest_side_similar / c
  ratio * perimeter_small

theorem perimeter_of_larger_triangle_is_65 :
  similar_triangle_perimeter = 65 := by
  sorry

end perimeter_of_larger_triangle_is_65_l26_26335


namespace value_of_expression_l26_26235

theorem value_of_expression (m a b c d : ℚ) 
  (hm : |m + 1| = 4)
  (hab : a = -b) 
  (hcd : c * d = 1) :
  a + b + 3 * c * d - m = 0 ∨ a + b + 3 * c * d - m = 8 :=
by
  sorry

end value_of_expression_l26_26235


namespace soccer_ball_cost_l26_26136

theorem soccer_ball_cost :
  ∃ x y : ℝ, x + y = 100 ∧ 2 * x + 3 * y = 262 ∧ x = 38 :=
by
  sorry

end soccer_ball_cost_l26_26136


namespace quadratic_no_real_roots_l26_26999

theorem quadratic_no_real_roots (c : ℝ) (h : c > 1) : ∀ x : ℝ, x^2 + 2 * x + c ≠ 0 :=
by
  sorry

end quadratic_no_real_roots_l26_26999


namespace parallelepiped_volume_k_l26_26347

theorem parallelepiped_volume_k (k : ℝ) : 
    abs (3 * k^2 - 13 * k + 27) = 20 ↔ k = (13 + Real.sqrt 85) / 6 ∨ k = (13 - Real.sqrt 85) / 6 := 
by sorry

end parallelepiped_volume_k_l26_26347


namespace rectangle_area_l26_26471

theorem rectangle_area (x w : ℝ) (h₁ : 3 * w = 3 * w) (h₂ : x^2 = 9 * w^2 + w^2) : 
  (3 * w) * w = (3 / 10) * x^2 := 
by
  sorry

end rectangle_area_l26_26471


namespace sum_of_ages_l26_26994

theorem sum_of_ages (a b c : ℕ) (twin : a = b) (product : a * b * c = 256) : a + b + c = 20 := by
  sorry

end sum_of_ages_l26_26994


namespace sum_integers_neg50_to_60_l26_26752

theorem sum_integers_neg50_to_60 : 
  (Finset.sum (Finset.Icc (-50 : ℤ) 60) id) = 555 := 
by
  -- Placeholder for the actual proof
  sorry

end sum_integers_neg50_to_60_l26_26752


namespace solve_system_l26_26001

theorem solve_system :
    (∃ x y z : ℝ, 5 * x^2 + 3 * y^2 + 3 * x * y + 2 * x * z - y * z - 10 * y + 5 = 0 ∧
                49 * x^2 + 65 * y^2 + 49 * z^2 - 14 * x * y - 98 * x * z + 14 * y * z - 182 * x - 102 * y + 182 * z + 233 =0
                ∧ ((x = 0 ∧ y = 1 ∧ z = -2)
                   ∨ (x = 2/7 ∧ y = 1 ∧ z = -12/7))) :=
by
  sorry

end solve_system_l26_26001


namespace percentage_decrease_l26_26125

theorem percentage_decrease (x y : ℝ) : 
  (xy^2 - (0.7 * x) * (0.6 * y)^2) / xy^2 = 0.748 :=
by
  sorry

end percentage_decrease_l26_26125


namespace planting_rate_l26_26064

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l26_26064


namespace tractor_planting_rate_l26_26068

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l26_26068


namespace estimate_total_number_of_fish_l26_26182

-- Define the conditions
variables (totalMarked : ℕ) (secondSample : ℕ) (markedInSecondSample : ℕ) (N : ℕ)

-- Assume the conditions
axiom condition1 : totalMarked = 60
axiom condition2 : secondSample = 80
axiom condition3 : markedInSecondSample = 5

-- Lean theorem statement proving N = 960 given the conditions
theorem estimate_total_number_of_fish (totalMarked secondSample markedInSecondSample N : ℕ)
  (h1 : totalMarked = 60)
  (h2 : secondSample = 80)
  (h3 : markedInSecondSample = 5) :
  N = 960 :=
sorry

end estimate_total_number_of_fish_l26_26182


namespace isosceles_triangle_base_length_l26_26737

theorem isosceles_triangle_base_length (a b : ℝ) (h1 : a = 3 ∨ b = 3) (h2 : a + a + b = 15 ∨ a + b + b = 15) :
  b = 3 := 
sorry

end isosceles_triangle_base_length_l26_26737


namespace original_recipe_serves_7_l26_26488

theorem original_recipe_serves_7 (x : ℕ)
  (h1 : 2 / x = 10 / 35) :
  x = 7 := by
  sorry

end original_recipe_serves_7_l26_26488


namespace sector_area_l26_26981

theorem sector_area (s θ r : ℝ) (hs : s = 4) (hθ : θ = 2) (hr : r = s / θ) : (1/2) * r^2 * θ = 4 := by
  sorry

end sector_area_l26_26981


namespace find_f2_l26_26796

def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + b * x - 8

theorem find_f2 (a b : ℝ) (h : f (-2) a b = 10) : f 2 a b = -26 := by
  sorry

end find_f2_l26_26796


namespace parabola_directrix_l26_26280

theorem parabola_directrix (x y : ℝ) : (y^2 = 2 * x) → (x = -(1 / 2)) := by
  sorry

end parabola_directrix_l26_26280


namespace range_of_a_in_second_quadrant_l26_26114

theorem range_of_a_in_second_quadrant :
  (∀ (x y : ℝ), x^2 + y^2 + 6*x - 4*a*y + 3*a^2 + 9 = 0 → x < 0 ∧ y > 0) → (0 < a ∧ a < 3) :=
by
  sorry

end range_of_a_in_second_quadrant_l26_26114


namespace spherical_to_rectangular_example_l26_26776

noncomputable def spherical_to_rectangular (ρ θ ϕ : ℝ) : ℝ × ℝ × ℝ :=
  (ρ * Real.sin ϕ * Real.cos θ, ρ * Real.sin ϕ * Real.sin θ, ρ * Real.cos ϕ)

theorem spherical_to_rectangular_example :
  spherical_to_rectangular 4 (Real.pi / 4) (Real.pi / 6) = (Real.sqrt 2, Real.sqrt 2, 2 * Real.sqrt 3) :=
by
  sorry

end spherical_to_rectangular_example_l26_26776


namespace freshmen_sophomores_without_pets_l26_26287

theorem freshmen_sophomores_without_pets : 
  let total_students := 400
  let percentage_freshmen_sophomores := 0.50
  let percentage_with_pets := 1/5
  let freshmen_sophomores := percentage_freshmen_sophomores * total_students
  160 = (freshmen_sophomores - (percentage_with_pets * freshmen_sophomores)) :=
by
  sorry

end freshmen_sophomores_without_pets_l26_26287


namespace find_m_n_l26_26073

theorem find_m_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (hmn : m^n = n^(m - n)) : 
  (m = 9 ∧ n = 3) ∨ (m = 8 ∧ n = 2) :=
sorry

end find_m_n_l26_26073


namespace mod_add_5000_l26_26755

theorem mod_add_5000 (n : ℤ) (h : n % 6 = 4) : (n + 5000) % 6 = 0 :=
sorry

end mod_add_5000_l26_26755


namespace f_prime_neg1_l26_26399

def f (a b c x : ℝ) : ℝ := a * x^4 + b * x^2 + c

def f' (a b c x : ℝ) : ℝ := 4 * a * x^3 + 2 * b * x

theorem f_prime_neg1 (a b c : ℝ) (h : f' a b c 1 = 2) : f' a b c (-1) = -2 :=
by
  sorry

end f_prime_neg1_l26_26399


namespace fgf_one_l26_26413

/-- Define the function f(x) = 5x + 2 --/
def f (x : ℝ) := 5 * x + 2

/-- Define the function g(x) = 3x - 1 --/
def g (x : ℝ) := 3 * x - 1

/-- Prove that f(g(f(1))) = 102 given the definitions of f and g --/
theorem fgf_one : f (g (f 1)) = 102 := by
  sorry

end fgf_one_l26_26413


namespace trapezium_area_l26_26615

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 12) :
  (1 / 2 * (a + b) * h = 228) :=
by
  sorry

end trapezium_area_l26_26615


namespace father_l26_26791

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l26_26791


namespace range_of_a_l26_26801

variables (a : ℝ) (x : ℝ) (x0 : ℝ)

def proposition_P (a : ℝ) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def proposition_Q (a : ℝ) : Prop :=
  ∃ x0, x0^2 + 2 * a * x0 + 2 - a = 0

theorem range_of_a (a : ℝ) :
  (proposition_P a ∧ proposition_Q a) → a ∈ {a : ℝ | a ≤ -2} ∪ {a : ℝ | a = 1} :=
by {
  sorry -- Proof goes here.
}

end range_of_a_l26_26801


namespace coefficients_of_polynomial_l26_26508

theorem coefficients_of_polynomial (a_5 a_4 a_3 a_2 a_1 a_0 : ℝ) :
  (∀ x : ℝ, x^5 = a_5 * (2*x + 1)^5 + a_4 * (2*x + 1)^4 + a_3 * (2*x + 1)^3 + a_2 * (2*x + 1)^2 + a_1 * (2*x + 1) + a_0) →
  a_5 = 1/32 ∧ a_4 = -5/32 :=
by sorry

end coefficients_of_polynomial_l26_26508


namespace range_of_a_odd_not_even_l26_26093

noncomputable def f (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def A : Set ℝ := Set.Ioo (-1 : ℝ) 1

def B (a : ℝ) : Set ℝ := Set.Ioo a (a + 1)

theorem range_of_a (a : ℝ) (h1 : B a ⊆ A) : -1 ≤ a ∧ a ≤ 0 := by
  sorry

theorem odd_not_even : (∀ x ∈ A, f (-x) = - f x) ∧ ¬ (∀ x ∈ A, f x = f (-x)) := by
  sorry

end range_of_a_odd_not_even_l26_26093


namespace div_poly_odd_power_l26_26415

theorem div_poly_odd_power (a b : ℤ) (n : ℕ) : (a + b) ∣ (a^(2*n+1) + b^(2*n+1)) :=
sorry

end div_poly_odd_power_l26_26415


namespace function_bounds_l26_26243

theorem function_bounds {a : ℝ} :
  (∀ x : ℝ, x > 0 → 4 - x^2 + a * Real.log x ≤ 3) → a = 2 :=
by
  sorry

end function_bounds_l26_26243


namespace sequence_a_n_l26_26438

-- Given conditions from the problem
variable {a : ℕ → ℕ}
variable (S : ℕ → ℕ)
variable (n : ℕ)

-- The sum of the first n terms of the sequence is given by S_n
axiom sum_Sn : ∀ n : ℕ, n > 0 → S n = 2 * n * n

-- Definition of a_n, the nth term of the sequence
def a_n (n : ℕ) : ℕ :=
  if n = 1 then
    S 1
  else
    S n - S (n - 1)

-- Prove that a_n = 4n - 2 for all n > 0.
theorem sequence_a_n (n : ℕ) (h : n > 0) : a_n S n = 4 * n - 2 :=
by
  sorry

end sequence_a_n_l26_26438


namespace find_number_l26_26609

theorem find_number (x : ℤ) (h : 3 * x - 4 = 5) : x = 3 :=
sorry

end find_number_l26_26609


namespace seating_arrangements_l26_26874

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l26_26874


namespace find_constants_l26_26212

theorem find_constants (a b c d : ℚ) :
  (6 * x^3 - 4 * x + 2) * (a * x^3 + b * x^2 + c * x + d) =
  18 * x^6 - 2 * x^5 + 16 * x^4 - (28 / 3) * x^3 + (8 / 3) * x^2 - 4 * x + 2 →
  a = 3 ∧ b = -1 / 3 ∧ c = 14 / 9 :=
by
  sorry

end find_constants_l26_26212


namespace no_such_n_l26_26493

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end no_such_n_l26_26493


namespace first_grade_children_count_l26_26543

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l26_26543


namespace time_to_cut_mans_hair_l26_26850

theorem time_to_cut_mans_hair :
  ∃ (x : ℕ),
    (3 * 50) + (2 * x) + (3 * 25) = 255 ∧ x = 15 :=
by {
  sorry
}

end time_to_cut_mans_hair_l26_26850


namespace extreme_value_f_range_of_a_l26_26841

noncomputable def f (x : ℝ) : ℝ := x * Real.log x
noncomputable def g (x a : ℝ) : ℝ := -x^2 + a * x - 3
noncomputable def h (x : ℝ) : ℝ := 2 * Real.log x + x + 3 / x

theorem extreme_value_f : ∃ x, f x = -1 / Real.exp 1 :=
by sorry

theorem range_of_a (a : ℝ) : (∀ x > 0, 2 * f x ≥ g x a) → a ≤ 4 :=
by sorry

end extreme_value_f_range_of_a_l26_26841


namespace problem_statement_l26_26900

noncomputable def smallest_integer_exceeding := 
  let x : ℝ := (Real.sqrt 3 + Real.sqrt 2) ^ 8
  Int.ceil x

theorem problem_statement : smallest_integer_exceeding = 5360 :=
by 
  -- The proof is omitted
  sorry

end problem_statement_l26_26900


namespace Uki_earnings_l26_26598

theorem Uki_earnings (cupcake_price cookie_price biscuit_price : ℝ) 
                     (cupcake_count cookie_count biscuit_count : ℕ)
                     (days : ℕ) :
  cupcake_price = 1.50 →
  cookie_price = 2 →
  biscuit_price = 1 →
  cupcake_count = 20 →
  cookie_count = 10 →
  biscuit_count = 20 →
  days = 5 →
  (days : ℝ) * (cupcake_price * (cupcake_count : ℝ) + cookie_price * (cookie_count : ℝ) + biscuit_price * (biscuit_count : ℝ)) = 350 := 
by
  sorry

end Uki_earnings_l26_26598


namespace symmetric_point_x_correct_l26_26848

-- Define the Cartesian coordinate system
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define the symmetry with respect to the x-axis
def symmetricPointX (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

-- Given point (-2, 1, 4)
def givenPoint : Point3D := { x := -2, y := 1, z := 4 }

-- Define the expected symmetric point
def expectedSymmetricPoint : Point3D := { x := -2, y := -1, z := -4 }

-- State the theorem to prove the expected symmetric point
theorem symmetric_point_x_correct :
  symmetricPointX givenPoint = expectedSymmetricPoint := by
  -- here the proof would go, but we leave it as sorry
  sorry

end symmetric_point_x_correct_l26_26848


namespace exists_x_for_integer_conditions_l26_26502

-- Define the conditions as functions in Lean
def is_int_div (a b : Int) : Prop := ∃ k : Int, a = b * k

-- The target statement in Lean 4
theorem exists_x_for_integer_conditions :
  ∃ t_1 : Int, ∃ x : Int, (x = 105 * t_1 + 52) ∧ 
    (is_int_div (x - 3) 7) ∧ 
    (is_int_div (x - 2) 5) ∧ 
    (is_int_div (x - 4) 3) :=
by 
  sorry

end exists_x_for_integer_conditions_l26_26502


namespace mean_difference_l26_26673

variable (a1 a2 a3 a4 a5 a6 A : ℝ)

-- Arithmetic mean of six numbers is A
axiom mean_six_numbers : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

-- Arithmetic mean of the first four numbers is A + 10
axiom mean_first_four : (a1 + a2 + a3 + a4) / 4 = A + 10

-- Arithmetic mean of the last four numbers is A - 7
axiom mean_last_four : (a3 + a4 + a5 + a6) / 4 = A - 7

-- Prove the arithmetic mean of the first, second, fifth, and sixth numbers differs from A by 3
theorem mean_difference :
  (a1 + a2 + a5 + a6) / 4 = A - 3 := 
sorry

end mean_difference_l26_26673


namespace elvins_first_month_bill_l26_26350

variable (F C : ℕ)

def total_bill_first_month := F + C
def total_bill_second_month := F + 2 * C

theorem elvins_first_month_bill :
  total_bill_first_month F C = 46 ∧
  total_bill_second_month F C = 76 ∧
  total_bill_second_month F C - total_bill_first_month F C = 30 →
  total_bill_first_month F C = 46 :=
by
  intro h
  sorry

end elvins_first_month_bill_l26_26350


namespace prism_volume_l26_26593

theorem prism_volume (a b c : ℝ) (h1 : a * b = 45) (h2 : b * c = 49) (h3 : a * c = 56) : a * b * c = 1470 := by
  sorry

end prism_volume_l26_26593


namespace cost_per_serving_is_3_62_l26_26595

noncomputable def cost_per_serving : ℝ :=
  let beef_cost := 4 * 6
  let chicken_cost := (2.2 * 5) * 0.85
  let carrots_cost := 2 * 1.50
  let potatoes_cost := (1.5 * 1.80) * 0.85
  let onions_cost := 1 * 3
  let discounted_carrots := carrots_cost * 0.80
  let discounted_potatoes := potatoes_cost * 0.80
  let total_cost_before_tax := beef_cost + chicken_cost + discounted_carrots + discounted_potatoes + onions_cost
  let sales_tax := total_cost_before_tax * 0.07
  let total_cost_after_tax := total_cost_before_tax + sales_tax
  total_cost_after_tax / 12

theorem cost_per_serving_is_3_62 : cost_per_serving = 3.62 :=
by
  sorry

end cost_per_serving_is_3_62_l26_26595


namespace sqrt_expression_evaluation_l26_26779

theorem sqrt_expression_evaluation : 
  (Real.sqrt (4 + 2 * Real.sqrt 3) + Real.sqrt (4 - 2 * Real.sqrt 3) = 2) := 
by
  sorry

end sqrt_expression_evaluation_l26_26779


namespace eval_expression_l26_26351

theorem eval_expression : 4 * (8 - 3) - 7 = 13 := by
  sorry

end eval_expression_l26_26351


namespace solve_inequality_l26_26869

theorem solve_inequality (x : ℝ) : 
  (0 < (x - 3) * (x - 4) * (x - 5) / ((x - 2) * (x - 6))) ↔ 
  (x < 2) ∨ (4 < x ∧ x < 5) ∨ (6 < x) :=
by 
  sorry

end solve_inequality_l26_26869


namespace smallest_positive_period_max_min_values_l26_26092

noncomputable def f (x a : ℝ) : ℝ :=
  (Real.cos x) * (2 * Real.sqrt 3 * Real.sin x - Real.cos x) + a * Real.sin x ^ 2

theorem smallest_positive_period (a : ℝ) (h : f (Real.pi / 12) a = 0) : 
  ∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) a = f x a) ∧ (∀ ε > 0, ε < T → ∃ y, y < T ∧ f y a ≠ f 0 a) := 
sorry

theorem max_min_values (a : ℝ) (h : f (Real.pi / 12) a = 0) :
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), f x a ≤ Real.sqrt 3) ∧ 
  (∀ x ∈ Set.Icc (-Real.pi / 6) (Real.pi / 4), -2 ≤ f x a) := 
sorry

end smallest_positive_period_max_min_values_l26_26092


namespace arnold_danny_age_l26_26043

theorem arnold_danny_age (x : ℕ) : (x + 1) * (x + 1) = x * x + 17 → x = 8 :=
by
  sorry

end arnold_danny_age_l26_26043


namespace smallest_number_l26_26901

theorem smallest_number
  (A : ℕ := 2^3 + 2^2 + 2^1 + 2^0)
  (B : ℕ := 2 * 6^2 + 1 * 6)
  (C : ℕ := 1 * 4^3)
  (D : ℕ := 8 + 1) :
  A < B ∧ A < C ∧ A < D :=
by {
  sorry
}

end smallest_number_l26_26901


namespace adult_ticket_cost_l26_26594

theorem adult_ticket_cost 
  (child_ticket_cost : ℕ)
  (total_tickets : ℕ)
  (total_cost : ℕ)
  (adults_attended : ℕ)
  (children_tickets : ℕ)
  (adults_ticket_cost : ℕ)
  (h1 : child_ticket_cost = 6)
  (h2 : total_tickets = 225)
  (h3 : total_cost = 1875)
  (h4 : adults_attended = 175)
  (h5 : children_tickets = total_tickets - adults_attended)
  (h6 : total_cost = adults_attended * adults_ticket_cost + children_tickets * child_ticket_cost) :
  adults_ticket_cost = 9 :=
sorry

end adult_ticket_cost_l26_26594


namespace factorial_division_l26_26804

theorem factorial_division (h : 9.factorial = 362880) : 9.factorial / 4.factorial = 15120 := by
  sorry

end factorial_division_l26_26804


namespace tractor_planting_rate_l26_26066

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l26_26066


namespace right_triangle_hypotenuse_segment_ratio_l26_26636

theorem right_triangle_hypotenuse_segment_ratio
  (x : ℝ)
  (h₀ : 0 < x)
  (AB BC : ℝ)
  (h₁ : AB = 3 * x)
  (h₂ : BC = 4 * x) :
  ∃ AD DC : ℝ, AD / DC = 3 := 
by
  sorry

end right_triangle_hypotenuse_segment_ratio_l26_26636


namespace factorable_polynomial_l26_26281

theorem factorable_polynomial (d f e g b : ℤ) (h1 : d * f = 28) (h2 : e * g = 14)
  (h3 : d * g + e * f = b) : b = 42 :=
by sorry

end factorable_polynomial_l26_26281


namespace median_of_siblings_list_l26_26694

def siblings_list : List ℕ := [0, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 5, 6]

theorem median_of_siblings_list : List.median siblings_list = 3 := sorry

end median_of_siblings_list_l26_26694


namespace part1_part2_l26_26829

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l26_26829


namespace find_m_for_parallel_vectors_l26_26515

theorem find_m_for_parallel_vectors (m : ℝ) :
  let a := (1, m)
  let b := (2, -1)
  (2 * a.1 + b.1, 2 * a.2 + b.2) = (k * (a.1 - 2 * b.1), k * (a.2 - 2 * b.2)) → m = -1/2 :=
by
  sorry

end find_m_for_parallel_vectors_l26_26515


namespace exponentiation_of_squares_l26_26483

theorem exponentiation_of_squares :
  ((Real.sqrt 2 + 1)^2000 * (Real.sqrt 2 - 1)^2000 = 1) :=
by
  sorry

end exponentiation_of_squares_l26_26483


namespace tan_sixty_eq_sqrt_three_l26_26911

theorem tan_sixty_eq_sqrt_three : Real.tan (Real.pi / 3) = Real.sqrt 3 := 
by
  sorry

end tan_sixty_eq_sqrt_three_l26_26911


namespace diagonal_of_rectangular_prism_l26_26485

theorem diagonal_of_rectangular_prism
  (width height depth : ℕ)
  (h1 : width = 15)
  (h2 : height = 20)
  (h3 : depth = 25) : 
  (width ^ 2 + height ^ 2 + depth ^ 2).sqrt = 25 * (2 : ℕ).sqrt :=
by {
  sorry
}

end diagonal_of_rectangular_prism_l26_26485


namespace locus_of_points_l26_26464

-- Define points A and B
variable {A B : (ℝ × ℝ)}
-- Define constant d
variable {d : ℝ}

-- Definition of the distances
def distance_sq (p q : ℝ × ℝ) : ℝ :=
  (p.1 - q.1)^2 + (p.2 - q.2)^2

theorem locus_of_points (A B : (ℝ × ℝ)) (d : ℝ) :
  ∀ M : (ℝ × ℝ), distance_sq M A - distance_sq M B = d ↔ 
  ∃ x : ℝ, ∃ y : ℝ, (M.1, M.2) = (x, y) ∧ 
  x = ((B.1 - A.1)^2 + d) / (2 * (B.1 - A.1)) :=
by
  sorry

end locus_of_points_l26_26464


namespace janet_dresses_pockets_l26_26704

theorem janet_dresses_pockets :
  ∀ (x : ℕ), (∀ (dresses_with_pockets remaining_dresses total_pockets : ℕ),
  dresses_with_pockets = 24 / 2 →
  total_pockets = 32 →
  remaining_dresses = dresses_with_pockets - dresses_with_pockets / 3 →
  (dresses_with_pockets / 3) * x + remaining_dresses * 3 = total_pockets →
  x = 2) :=
by
  intros x dresses_with_pockets remaining_dresses total_pockets h1 h2 h3 h4
  sorry

end janet_dresses_pockets_l26_26704


namespace quadratic_increasing_l26_26842

noncomputable def quadratic (a b c x : ℝ) := a * x^2 + b * x + c

theorem quadratic_increasing (a b c : ℝ) 
  (h1 : quadratic a b c 0 = quadratic a b c 6)
  (h2 : quadratic a b c 0 < quadratic a b c 7) :
  ∀ x, x > 3 → ∀ y, y > 3 → x < y → quadratic a b c x < quadratic a b c y :=
sorry

end quadratic_increasing_l26_26842


namespace scientific_notation_l26_26134

theorem scientific_notation : (10374 * 10^9 : Real) = 1.037 * 10^13 :=
by
  sorry

end scientific_notation_l26_26134


namespace value_of_k_l26_26469

theorem value_of_k :
  ∀ (k : ℝ), (∃ m : ℝ, m = 4/5 ∧ (21 - (-5)) / (k - 3) = m) →
  k = 35.5 :=
by
  intros k hk
  -- Here hk is the proof that the line through (3, -5) and (k, 21) has the same slope as 4/5
  sorry

end value_of_k_l26_26469


namespace triangle_lattice_points_l26_26264

theorem triangle_lattice_points :
  let S := 300
  let L := 60
  let N := 271 in
  S = N + (1/2 : ℚ) * L - 1 :=
by
  -- Calculate the number of lattice points on OA; 31
  -- Calculate the number of lattice points on OB; 10
  -- Calculate the number of lattice points on AB; 19
  -- Sum these to get L: 31 + 10 + 19 = 60
  -- Calculate the area of triangle ABO: 300
  -- Use Pick's theorem to solve for N: 271
  sorry

end triangle_lattice_points_l26_26264


namespace sum_of_digits_of_n_l26_26020

theorem sum_of_digits_of_n (n : ℕ) (h1 : 0 < n) (h2 : (n+1)! + (n+3)! = n! * 1320) : n.digits.sum = 1 := by
  sorry

end sum_of_digits_of_n_l26_26020


namespace even_two_digit_numbers_count_l26_26518

/-- Even positive integers less than 1000 with at most two different digits -/
def count_even_two_digit_numbers : ℕ :=
  let one_digit := [2, 4, 6, 8].length
  let two_d_same := [22, 44, 66, 88].length
  let two_d_diff := [24, 42, 26, 62, 28, 82, 46, 64, 48, 84, 68, 86].length
  let three_d_same := [222, 444, 666, 888].length
  let three_d_diff := 16 + 12
  one_digit + two_d_same + two_d_diff + three_d_same + three_d_diff

theorem even_two_digit_numbers_count :
  count_even_two_digit_numbers = 52 :=
by sorry

end even_two_digit_numbers_count_l26_26518


namespace log_comparison_l26_26225

theorem log_comparison (a b c : ℝ) (h₁ : a = Real.log 6 / Real.log 4) (h₂ : b = Real.log 3 / Real.log 2) (h₃ : c = 3/2) : b > c ∧ c > a := 
by 
  sorry

end log_comparison_l26_26225


namespace melanie_trout_catch_l26_26423

def trout_caught_sara : ℕ := 5
def trout_caught_melanie (sara_trout : ℕ) : ℕ := 2 * sara_trout

theorem melanie_trout_catch :
  trout_caught_melanie trout_caught_sara = 10 :=
by
  sorry

end melanie_trout_catch_l26_26423


namespace price_of_mixture_l26_26702

theorem price_of_mixture :
  (1 * 64 + 1 * 74) / (1 + 1) = 69 :=
by
  sorry

end price_of_mixture_l26_26702


namespace sin_double_angle_l26_26082

theorem sin_double_angle (α : ℝ) 
  (h1 : Real.cos (α + Real.pi / 4) = 3 / 5)
  (h2 : Real.pi / 2 ≤ α ∧ α ≤ 3 * Real.pi / 2) : 
  Real.sin (2 * α) = 7 / 25 := 
by sorry

end sin_double_angle_l26_26082


namespace average_price_of_towels_l26_26189

-- Definitions based on the conditions
def cost_of_three_towels := 3 * 100
def cost_of_five_towels := 5 * 150
def cost_of_two_towels := 550
def total_cost := cost_of_three_towels + cost_of_five_towels + cost_of_two_towels
def total_number_of_towels := 3 + 5 + 2
def average_price := total_cost / total_number_of_towels

-- The theorem statement
theorem average_price_of_towels :
  average_price = 160 :=
by
  sorry

end average_price_of_towels_l26_26189


namespace tan_identity_l26_26369

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l26_26369


namespace polynomial_square_b_value_l26_26734

theorem polynomial_square_b_value
  (a b : ℚ)
  (h : ∃ (p q r : ℚ), (x^4 - x^3 + x^2 + a * x + b) = (p * x^2 + q * x + r)^2) :
  b = 9 / 64 :=
sorry

end polynomial_square_b_value_l26_26734


namespace triangle_side_length_l26_26404

   theorem triangle_side_length
   (A B C D E F : Type)
   (angle_bac angle_edf : Real)
   (AB AC DE DF : Real)
   (h1 : angle_bac = angle_edf)
   (h2 : AB = 5)
   (h3 : AC = 4)
   (h4 : DE = 2.5)
   (area_eq : (1 / 2) * AB * AC * Real.sin angle_bac = (1 / 2) * DE * DF * Real.sin angle_edf):
   DF = 8 :=
   by
   sorry
   
end triangle_side_length_l26_26404


namespace problem_x_value_l26_26524

theorem problem_x_value (x : ℝ) (h : 0.25 * x = 0.15 * 1500 - 15) : x = 840 :=
by
  sorry

end problem_x_value_l26_26524


namespace second_largest_is_D_l26_26155

noncomputable def A := 3 * 3
noncomputable def C := 4 * A
noncomputable def B := C - 15
noncomputable def D := A + 19

theorem second_largest_is_D : 
    ∀ (A B C D : ℕ), 
      A = 9 → 
      B = 21 →
      C = 36 →
      D = 28 →
      D = 28 :=
by
  intros A B C D hA hB hC hD
  have h1 : A = 9 := by assumption
  have h2 : B = 21 := by assumption
  have h3 : C = 36 := by assumption
  have h4 : D = 28 := by assumption
  exact h4

end second_largest_is_D_l26_26155


namespace father_l26_26793

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l26_26793


namespace first_tribe_term_is_longer_l26_26985

def years_to_days_first_tribe (years : ℕ) : ℕ := 
  years * 12 * 30

def months_to_days_first_tribe (months : ℕ) : ℕ :=
  months * 30

def total_days_first_tribe (years months days : ℕ) : ℕ :=
  (years_to_days_first_tribe years) + (months_to_days_first_tribe months) + days

def years_to_days_second_tribe (years : ℕ) : ℕ := 
  years * 13 * 4 * 7

def moons_to_days_second_tribe (moons : ℕ) : ℕ :=
  moons * 4 * 7

def weeks_to_days_second_tribe (weeks : ℕ) : ℕ :=
  weeks * 7

def total_days_second_tribe (years moons weeks days : ℕ) : ℕ :=
  (years_to_days_second_tribe years) + (moons_to_days_second_tribe moons) + (weeks_to_days_second_tribe weeks) + days

theorem first_tribe_term_is_longer :
  total_days_first_tribe 7 1 18 > total_days_second_tribe 6 12 1 3 :=
by
  sorry

end first_tribe_term_is_longer_l26_26985


namespace scientific_notation_l26_26133

def z := 10374 * 10^9

theorem scientific_notation (a : ℝ) (n : ℤ) (h₁ : 1 ≤ |a|) (h₂ : |a| < 10) (h₃ : a * 10^n = z) : a = 1.04 ∧ n = 13 := sorry

end scientific_notation_l26_26133


namespace part1_part2_l26_26805

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l26_26805


namespace find_function_range_of_a_l26_26843

variables (a b : ℝ) (f : ℝ → ℝ) 

-- Given: f(x) = ax + b where a ≠ 0 
--        f(2x + 1) = 4x + 1
-- Prove: f(x) = 2x - 1
theorem find_function (h1 : ∀ x, f (2 * x + 1) = 4 * x + 1) : 
  ∃ a b, a = 2 ∧ b = -1 ∧ ∀ x, f x = a * x + b :=
by sorry

-- Given: A = {x | a - 1 < x < 2a +1 }
--        B = {x | 1 < f(x) < 3 }
--        B ⊆ A
-- Prove: 1/2 ≤ a ≤ 2
theorem range_of_a (Hf : ∀ x, f x = 2 * x - 1) (Hsubset: ∀ x, 1 < f x ∧ f x < 3 → a - 1 < x ∧ x < 2 * a + 1) :
  1 / 2 ≤ a ∧ a ≤ 2 :=
by sorry

end find_function_range_of_a_l26_26843


namespace unique_number_encoding_l26_26149

-- Defining participants' score ranges 
def score_range := {x : ℕ // x ≤ 5}

-- Defining total score
def total_score (s1 s2 s3 s4 s5 s6 : score_range) : ℕ := 
  s1.val + s2.val + s3.val + s4.val + s5.val + s6.val

-- Main statement to encode participant's scores into a unique number
theorem unique_number_encoding (s1 s2 s3 s4 s5 s6 : score_range) :
  ∃ n : ℕ, ∃ s : ℕ, 
    s = total_score s1 s2 s3 s4 s5 s6 ∧ 
    n = s * 10^6 + s1.val * 10^5 + s2.val * 10^4 + s3.val * 10^3 + s4.val * 10^2 + s5.val * 10 + s6.val := 
sorry

end unique_number_encoding_l26_26149


namespace minimum_value_inequality_equality_condition_exists_l26_26708

theorem minimum_value_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) ≥ 12 := by
  sorry

theorem equality_condition_exists : 
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (6 * c / (3 * a + b) + 6 * a / (b + 3 * c) + 2 * b / (a + c) = 12) := by
  sorry

end minimum_value_inequality_equality_condition_exists_l26_26708


namespace engineer_last_name_is_smith_l26_26290

/-- Given these conditions:
 1. Businessman Robinson and a conductor live in Sheffield.
 2. Businessman Jones and a stoker live in Leeds.
 3. Businessman Smith and the railroad engineer live halfway between Leeds and Sheffield.
 4. The conductor’s namesake earns $10,000 a year.
 5. The engineer earns exactly 1/3 of what the businessman who lives closest to him earns.
 6. Railroad worker Smith beats the stoker at billiards.
 
We need to prove that the last name of the engineer is Smith. -/
theorem engineer_last_name_is_smith
  (lives_in_Sheffield_Robinson : Prop)
  (lives_in_Sheffield_conductor : Prop)
  (lives_in_Leeds_Jones : Prop)
  (lives_in_Leeds_stoker : Prop)
  (lives_in_halfway_Smith : Prop)
  (lives_in_halfway_engineer : Prop)
  (conductor_namesake_earns_10000 : Prop)
  (engineer_earns_one_third_closest_bizman : Prop)
  (railway_worker_Smith_beats_stoker_at_billiards : Prop) :
  (engineer_last_name = "Smith") :=
by
  -- Proof will go here
  sorry

end engineer_last_name_is_smith_l26_26290


namespace number_of_children_l26_26545

-- Define the given conditions in Lean 4
variable {a : ℕ}
variable {R : ℕ}
variable {L : ℕ}
variable {k : ℕ}

-- Conditions given in the problem
def condition1 : 200 ≤ a ∧ a ≤ 300 := sorry
def condition2 : a = 25 * R + 10 := sorry
def condition3 : a = 30 * L - 15 := sorry 
def condition4 : a + 15 = 150 * k := sorry

-- The theorem to prove
theorem number_of_children : a = 285 :=
by
  assume a R L k // This assumption is for the variables needed.
  have h₁ : condition1 := sorry
  have h₂ : condition2 := sorry
  have h₃ : condition3 := sorry
  have h₄ : condition4 := sorry 
  exact sorry

end number_of_children_l26_26545


namespace smallest_next_divisor_l26_26408

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_divisor (a b : ℕ) : Prop := b % a = 0

theorem smallest_next_divisor 
  (m : ℕ) 
  (h1 : 1000 ≤ m ∧ m < 10000) 
  (h2 : is_even m) 
  (h3 : is_divisor 171 m)
  : ∃ k, k > 171 ∧ k = 190 ∧ is_divisor k m := 
by
  sorry

end smallest_next_divisor_l26_26408


namespace permutations_of_3_3_3_7_7_l26_26390

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem permutations_of_3_3_3_7_7 : 
  (factorial 5) / (factorial 3 * factorial 2) = 10 :=
by
  sorry

end permutations_of_3_3_3_7_7_l26_26390


namespace prove_students_second_and_third_l26_26131

namespace MonicaClasses

def Monica := 
  let classes_per_day := 6
  let students_first_class := 20
  let students_fourth_class := students_first_class / 2
  let students_fifth_class := 28
  let students_sixth_class := 28
  let total_students := 136
  let known_students := students_first_class + students_fourth_class + students_fifth_class + students_sixth_class
  let students_second_and_third := total_students - known_students
  students_second_and_third = 50

theorem prove_students_second_and_third : Monica :=
  by
    sorry

end MonicaClasses

end prove_students_second_and_third_l26_26131


namespace part1_part2_l26_26838

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l26_26838


namespace johnson_family_seating_problem_l26_26876

theorem johnson_family_seating_problem : 
  ∃ n : ℕ, n = 9! - 5! * 4! ∧ n = 359760 :=
by
  have total_ways := (Nat.factorial 9)
  have no_adjacent_boys := (Nat.factorial 5) * (Nat.factorial 4)
  have result := total_ways - no_adjacent_boys
  use result
  split
  . exact eq.refl result
  . norm_num -- This will replace result with its evaluated form, 359760

end johnson_family_seating_problem_l26_26876


namespace binom_28_7_l26_26676

theorem binom_28_7 (h1 : Nat.choose 26 3 = 2600) (h2 : Nat.choose 26 4 = 14950) (h3 : Nat.choose 26 5 = 65780) : 
  Nat.choose 28 7 = 197340 :=
by
  sorry

end binom_28_7_l26_26676


namespace quadratic_roots_value_r_l26_26558

theorem quadratic_roots_value_r
  (a b m p r : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h_root1 : a^2 - m*a + 3 = 0)
  (h_root2 : b^2 - m*b + 3 = 0)
  (h_ab : a * b = 3)
  (h_root3 : (a + 1/b) * (b + 1/a) = r) :
  r = 16 / 3 :=
sorry

end quadratic_roots_value_r_l26_26558


namespace loss_percent_l26_26614

theorem loss_percent (CP SP Loss : ℝ) (h1 : CP = 600) (h2 : SP = 450) (h3 : Loss = CP - SP) : (Loss / CP) * 100 = 25 :=
by
  sorry

end loss_percent_l26_26614


namespace range_of_m_l26_26884

noncomputable def satisfies_inequality (m : ℝ) : Prop :=
  ∀ x : ℝ, |x + 3| - |x - 1| ≤ m^2 - 3 * m

theorem range_of_m (m : ℝ) : 
  satisfies_inequality m ↔ (m ≥ 4 ∨ m ≤ -1) :=
by
  sorry

end range_of_m_l26_26884


namespace nonnegative_integer_solution_count_l26_26392

theorem nonnegative_integer_solution_count :
  ∃ n : ℕ, (∀ x : ℕ, x^2 + 6 * x = 0 → x = 0) ∧ n = 1 :=
by
  sorry

end nonnegative_integer_solution_count_l26_26392


namespace jose_investment_l26_26895

theorem jose_investment 
  (T_investment : ℕ := 30000) -- Tom's investment in Rs.
  (J_months : ℕ := 10)        -- Jose's investment period in months
  (T_months : ℕ := 12)        -- Tom's investment period in months
  (total_profit : ℕ := 72000) -- Total profit in Rs.
  (jose_profit : ℕ := 40000)  -- Jose's share of profit in Rs.
  : ∃ X : ℕ, (jose_profit * (T_investment * T_months)) = ((total_profit - jose_profit) * (X * J_months)) ∧ X = 45000 :=
  sorry

end jose_investment_l26_26895


namespace satisfies_conditions_l26_26608

noncomputable def m := 29 / 3

def real_part (m : ℝ) : ℝ := m^2 - 8*m + 15
def imag_part (m : ℝ) : ℝ := m^2 - 5*m - 14

theorem satisfies_conditions (m : ℝ) 
  (real_cond : m < 3 ∨ m > 5) 
  (imag_cond : -2 < m ∧ m < 7)
  (line_cond : real_part m = imag_part m): 
  m = 29 / 3 :=
by {
  sorry
}

end satisfies_conditions_l26_26608


namespace missy_tv_watching_time_l26_26256

def reality_show_count : Nat := 5
def reality_show_duration : Nat := 28
def cartoon_duration : Nat := 10

theorem missy_tv_watching_time :
  reality_show_count * reality_show_duration + cartoon_duration = 150 := by
  sorry

end missy_tv_watching_time_l26_26256


namespace linear_function_common_quadrants_l26_26669

theorem linear_function_common_quadrants {k b : ℝ} (h : k * b < 0) :
  (exists (q1 q2 : ℕ), q1 = 1 ∧ q2 = 4) := 
sorry

end linear_function_common_quadrants_l26_26669


namespace cylinder_original_radius_inch_l26_26349

theorem cylinder_original_radius_inch (r : ℝ) :
  (∃ r : ℝ, (π * (r + 4)^2 * 3 = π * r^2 * 15) ∧ (r > 0)) →
  r = 1 + Real.sqrt 5 :=
by 
  sorry

end cylinder_original_radius_inch_l26_26349


namespace celine_change_l26_26920

theorem celine_change
  (price_laptop : ℕ)
  (price_smartphone : ℕ)
  (num_laptops : ℕ)
  (num_smartphones : ℕ)
  (total_money : ℕ)
  (h1 : price_laptop = 600)
  (h2 : price_smartphone = 400)
  (h3 : num_laptops = 2)
  (h4 : num_smartphones = 4)
  (h5 : total_money = 3000) :
  total_money - (num_laptops * price_laptop + num_smartphones * price_smartphone) = 200 :=
by
  sorry

end celine_change_l26_26920


namespace triangle_side_cube_l26_26909

theorem triangle_side_cube 
  (a b c : ℕ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_gcd : Nat.gcd a (Nat.gcd b c) = 1)
  (angle_condition : ∃ A B : ℝ, A = 3 * B) 
  : ∃ n m : ℕ, (a = n ^ 3 ∨ b = n ^ 3 ∨ c = n ^ 3) :=
sorry

end triangle_side_cube_l26_26909


namespace optimal_station_placement_l26_26987

def distance_between_buildings : ℕ := 50
def workers_in_building (n : ℕ) : ℕ := n

def total_walking_distance (x : ℝ) : ℝ :=
  |x| + 2 * |x - 50| + 3 * |x - 100| + 4 * |x - 150| + 5 * |x - 200|

theorem optimal_station_placement : ∃ x : ℝ, x = 150 ∧ (∀ y : ℝ, total_walking_distance x ≤ total_walking_distance y) :=
  sorry

end optimal_station_placement_l26_26987


namespace original_six_digit_number_is_105262_l26_26186

def is_valid_number (N : ℕ) : Prop :=
  ∃ A : ℕ, A < 100000 ∧ (N = 10 * A + 2) ∧ (200000 + A = 2 * N + 2)

theorem original_six_digit_number_is_105262 :
  ∃ N : ℕ, is_valid_number N ∧ N = 105262 :=
by
  sorry

end original_six_digit_number_is_105262_l26_26186


namespace f_g_evaluation_l26_26111

-- Definitions of the functions g and f
def g (x : ℤ) : ℤ := x^3
def f (x : ℤ) : ℤ := 3 * x - 2

-- Goal: Prove that f(g(2)) = 22
theorem f_g_evaluation : f (g 2) = 22 :=
by
  sorry

end f_g_evaluation_l26_26111


namespace no_unique_y_exists_l26_26980

theorem no_unique_y_exists (x y : ℕ) (k m : ℤ) 
  (h1 : x % 82 = 5)
  (h2 : (x + 7) % y = 12) :
  ¬ ∃! y, (∃ k m : ℤ, x = 82 * k + 5 ∧ (x + 7) = y * m + 12) :=
by
  sorry

end no_unique_y_exists_l26_26980


namespace vacation_days_proof_l26_26915

-- Define the conditions
def family_vacation (total_days rain_days clear_afternoons : ℕ) : Prop :=
  total_days = 18 ∧ rain_days = 13 ∧ clear_afternoons = 12

-- State the theorem to be proved
theorem vacation_days_proof : family_vacation 18 13 12 → 18 = 18 :=
by
  -- Skip the proof
  intro h
  sorry

end vacation_days_proof_l26_26915


namespace accurate_to_hundreds_place_l26_26666

def rounded_number : ℝ := 8.80 * 10^4

theorem accurate_to_hundreds_place
  (n : ℝ) (h : n = rounded_number) : 
  exists (d : ℤ), n = d * 100 ∧ |round n - n| < 50 :=
sorry

end accurate_to_hundreds_place_l26_26666


namespace janet_practiced_days_l26_26407

theorem janet_practiced_days (total_miles : ℕ) (miles_per_day : ℕ) (days_practiced : ℕ) :
  total_miles = 72 ∧ miles_per_day = 8 → days_practiced = total_miles / miles_per_day → days_practiced = 9 :=
by
  sorry

end janet_practiced_days_l26_26407


namespace combined_score_is_210_l26_26696

theorem combined_score_is_210 :
  ∀ (total_questions : ℕ) (marks_per_question : ℕ) (jose_wrong : ℕ) 
    (meghan_less_than_jose : ℕ) (jose_more_than_alisson : ℕ) (jose_total : ℕ),
  total_questions = 50 →
  marks_per_question = 2 →
  jose_wrong = 5 →
  meghan_less_than_jose = 20 →
  jose_more_than_alisson = 40 →
  jose_total = total_questions * marks_per_question - (jose_wrong * marks_per_question) →
  (jose_total - meghan_less_than_jose) + jose_total + (jose_total - jose_more_than_alisson) = 210 :=
by
  intros total_questions marks_per_question jose_wrong meghan_less_than_jose jose_more_than_alisson jose_total
  intros h1 h2 h3 h4 h5 h6
  sorry

end combined_score_is_210_l26_26696


namespace book_collection_example_l26_26031

theorem book_collection_example :
  ∃ (P C B : ℕ), 
    (P : ℚ) / C = 3 / 2 ∧ 
    (C : ℚ) / B = 4 / 3 ∧ 
    P + C + B = 3002 ∧ 
    P + C + B > 3000 :=
by
  sorry

end book_collection_example_l26_26031


namespace remainder_is_3_l26_26499

-- Define the polynomial p(x)
def p (x : ℝ) := x^3 - 3 * x + 5

-- Define the divisor d(x)
def d (x : ℝ) := x - 1

-- The theorem: remainder when p(x) is divided by d(x)
theorem remainder_is_3 : p 1 = 3 := by 
  sorry

end remainder_is_3_l26_26499


namespace molecular_weight_of_NH4I_correct_l26_26452

-- Define the atomic weights as given conditions
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_I : ℝ := 126.90

-- Define the calculation of the molecular weight of NH4I
def molecular_weight_NH4I : ℝ :=
  atomic_weight_N + 4 * atomic_weight_H + atomic_weight_I

-- Theorem stating the molecular weight of NH4I is 144.95 g/mol
theorem molecular_weight_of_NH4I_correct : molecular_weight_NH4I = 144.95 :=
by
  sorry

end molecular_weight_of_NH4I_correct_l26_26452


namespace Mina_additional_miles_l26_26716

theorem Mina_additional_miles:
  let distance1 := 20 -- distance in miles for the first part of the trip
  let speed1 := 40 -- speed in mph for the first part of the trip
  let speed2 := 60 -- speed in mph for the second part of the trip
  let avg_speed := 55 -- average speed needed for the entire trip in mph
  let distance2 := (distance1 / speed1 + (avg_speed * (distance1 / speed1)) / (speed1 - avg_speed * speed1 / speed2)) * speed2 -- formula to find the additional distance
  distance2 = 90 :=
by {
  sorry
}

end Mina_additional_miles_l26_26716


namespace final_surface_area_l26_26314

noncomputable def surface_area (total_cubes remaining_cubes cube_surface removed_internal_surface : ℕ) : ℕ :=
  (remaining_cubes * cube_surface) + (remaining_cubes * removed_internal_surface)

theorem final_surface_area :
  surface_area 64 55 54 6 = 3300 :=
by
  sorry

end final_surface_area_l26_26314


namespace viable_combinations_l26_26924

-- Given conditions
def totalHerbs : Nat := 4
def totalCrystals : Nat := 6
def incompatibleComb1 : Nat := 2
def incompatibleComb2 : Nat := 1

-- Theorem statement proving the number of viable combinations
theorem viable_combinations : totalHerbs * totalCrystals - (incompatibleComb1 + incompatibleComb2) = 21 := by
  sorry

end viable_combinations_l26_26924


namespace doughnuts_left_l26_26060

theorem doughnuts_left (total_doughnuts : ℕ) (total_staff : ℕ) (staff3 : ℕ) (staff2 : ℕ) :
  total_doughnuts = 120 → total_staff = 35 →
  staff3 = 15 → staff2 = 10 →
  (let staff4 := total_staff - (staff3 + staff2) in
   let eaten3 := staff3 * 3 in
   let eaten2 := staff2 * 2 in
   let eaten4 := staff4 * 4 in
   let total_eaten := eaten3 + eaten2 + eaten4 in
   total_doughnuts - total_eaten = 15) :=
by
  intros total_doughnuts_eq total_staff_eq staff3_eq staff2_eq
  have staff4 := total_staff - (staff3 + staff2)
  have eaten3 := staff3 * 3
  have eaten2 := staff2 * 2
  have eaten4 := staff4 * 4
  have total_eaten := eaten3 + eaten2 + eaten4
  exact (total_doughnuts - total_eaten = 15)
  sorry

end doughnuts_left_l26_26060


namespace three_circles_area_less_than_total_radius_squared_l26_26893

theorem three_circles_area_less_than_total_radius_squared
    (x y z R : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : R > 0)
    (descartes_theorem : ( (1/x + 1/y + 1/z - 1/R)^2 = 2 * ( (1/x)^2 + (1/y)^2 + (1/z)^2 + (1/R)^2 ) )) :
    x^2 + y^2 + z^2 < 4 * R^2 := 
sorry

end three_circles_area_less_than_total_radius_squared_l26_26893


namespace maximum_value_expression_l26_26559

theorem maximum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z)
  (h : x^2 - 3 * x * y + 4 * y^2 - z = 0) :
  ∃ x y z, x^2 - 3 * x * y + 4 * y^2 - z = 0 
  ∧ (x > 0 ∧ y > 0 ∧ z > 0) 
  ∧ x + 2 * y - z ≤ 2 :=
sorry

end maximum_value_expression_l26_26559


namespace smallest_n_l26_26141

theorem smallest_n (n : ℕ) : 
  (n % 6 = 4) ∧ (n % 7 = 2) ∧ (n > 20) → n = 58 :=
by
  sorry

end smallest_n_l26_26141


namespace Ed_more_marbles_than_Doug_l26_26348

-- Definitions based on conditions
def Ed_marbles_initial : ℕ := 45
def Doug_loss : ℕ := 11
def Doug_marbles_initial : ℕ := Ed_marbles_initial - 10
def Doug_marbles_after_loss : ℕ := Doug_marbles_initial - Doug_loss

-- Theorem statement
theorem Ed_more_marbles_than_Doug :
  Ed_marbles_initial - Doug_marbles_after_loss = 21 :=
by
  -- Proof would go here
  sorry

end Ed_more_marbles_than_Doug_l26_26348


namespace relationship_among_a_b_c_l26_26957

noncomputable def a := (1/2)^(2/3)
noncomputable def b := (1/5)^(2/3)
noncomputable def c := (1/2)^(1/3)

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l26_26957


namespace probability_one_each_item_l26_26108

theorem probability_one_each_item :
  let num_items := 32
  let total_ways := Nat.choose num_items 4
  let favorable_outcomes := 8 * 8 * 8 * 8
  total_ways = 35960 →
  let probability := favorable_outcomes / total_ways
  probability = (128 : ℚ) / 1125 :=
by
  sorry

end probability_one_each_item_l26_26108


namespace smallest_area_of_2020th_square_l26_26638

theorem smallest_area_of_2020th_square :
  ∃ (S : ℤ) (A : ℕ), 
    (S * S - 2019 = A) ∧ 
    (∃ k : ℕ, k * k = A) ∧ 
    (∀ (T : ℤ) (B : ℕ), ((T * T - 2019 = B) ∧ (∃ l : ℕ, l * l = B)) → (A ≤ B)) :=
sorry

end smallest_area_of_2020th_square_l26_26638


namespace triangle_third_side_l26_26642

theorem triangle_third_side (x : ℕ) : 
  (3 < x) ∧ (x < 17) → 
  (x = 11) :=
by
  sorry

end triangle_third_side_l26_26642


namespace connie_correct_answer_l26_26344

theorem connie_correct_answer (y : ℕ) (h1 : y - 8 = 32) : y + 8 = 48 := by
  sorry

end connie_correct_answer_l26_26344


namespace solve_for_y_l26_26396

theorem solve_for_y (x y : ℝ) (h1 : x - y = 8) (h2 : x + y = 10) : y = 1 :=
by
  sorry

end solve_for_y_l26_26396


namespace shakes_indeterminable_l26_26932

variable {B S C x : ℝ}

theorem shakes_indeterminable (h1 : 3 * B + x * S + C = 130) (h2 : 4 * B + 10 * S + C = 164.5) : 
  ¬ (∃ x, 3 * B + x * S + C = 130 ∧ 4 * B + 10 * S + C = 164.5) :=
by
  sorry

end shakes_indeterminable_l26_26932


namespace necessary_not_sufficient_condition_not_sufficient_condition_l26_26454

theorem necessary_not_sufficient_condition (x : ℝ) :
  (1 < x ∧ x < 4) → (|x - 2| < 1) := sorry

theorem not_sufficient_condition (x : ℝ) :
  (|x - 2| < 1) → (1 < x ∧ x < 4) := sorry

end necessary_not_sufficient_condition_not_sufficient_condition_l26_26454


namespace initial_flowers_per_bunch_l26_26648

theorem initial_flowers_per_bunch (x : ℕ) (h₁: 8 * x = 72) : x = 9 :=
  by
  sorry

end initial_flowers_per_bunch_l26_26648


namespace find_number_l26_26029

theorem find_number (x : ℝ) : 3 * (2 * x + 9) = 75 → x = 8 :=
by {
  sorry
}

end find_number_l26_26029


namespace find_X_l26_26364

-- Defining the given conditions and what we need to prove
theorem find_X (X : ℝ) (h : (X + 43 / 151) * 151 = 2912) : X = 19 :=
sorry

end find_X_l26_26364


namespace johnson_family_seating_l26_26873

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

theorem johnson_family_seating (sons daughters : ℕ) (total_seats : ℕ) 
  (condition1 : sons = 5) (condition2 : daughters = 4) (condition3 : total_seats = 9) :
  let total_arrangements := factorial total_seats,
      restricted_arrangements := factorial sons * factorial daughters,
      answer := total_arrangements - restricted_arrangements
  in answer = 360000 := 
by
  -- The proof would go here
  sorry

end johnson_family_seating_l26_26873


namespace trebled_principal_after_5_years_l26_26585

theorem trebled_principal_after_5_years 
(P R : ℝ) (T total_interest : ℝ) (n : ℝ) 
(h1 : T = 10) 
(h2 : total_interest = 800) 
(h3 : (P * R * 10) / 100 = 400) 
(h4 : (P * R * n) / 100 + (3 * P * R * (10 - n)) / 100 = 800) :
n = 5 :=
by
-- The Lean proof will go here
sorry

end trebled_principal_after_5_years_l26_26585


namespace suff_not_nec_for_abs_eq_one_l26_26507

variable (m : ℝ)

theorem suff_not_nec_for_abs_eq_one (hm : m = 1) : |m| = 1 ∧ (¬(|m| = 1 → m = 1)) := by
  sorry

end suff_not_nec_for_abs_eq_one_l26_26507


namespace mr_johnson_pill_intake_l26_26563

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end mr_johnson_pill_intake_l26_26563


namespace money_combination_l26_26745

variable (Raquel Tom Nataly Sam : ℝ)

-- Given Conditions 
def condition1 : Prop := Tom = (1 / 4) * Nataly
def condition2 : Prop := Nataly = 3 * Raquel
def condition3 : Prop := Sam = 2 * Nataly
def condition4 : Prop := Nataly = (5 / 3) * Sam
def condition5 : Prop := Raquel = 40

-- Proving this combined total
def combined_total : Prop := Tom + Raquel + Nataly + Sam = 262

theorem money_combination (h1: condition1 Tom Nataly) 
                          (h2: condition2 Nataly Raquel) 
                          (h3: condition3 Sam Nataly) 
                          (h4: condition4 Nataly Sam) 
                          (h5: condition5 Raquel) 
                          : combined_total Tom Raquel Nataly Sam :=
sorry

end money_combination_l26_26745


namespace parallel_lines_direction_vector_l26_26164

theorem parallel_lines_direction_vector (k : ℝ) :
  (∃ c : ℝ, (5, -3) = (c * -2, c * k)) ↔ k = 6 / 5 :=
by sorry

end parallel_lines_direction_vector_l26_26164


namespace negative_solution_range_l26_26091

theorem negative_solution_range (m : ℝ) : (∃ x : ℝ, 2 * x + 4 = m - x ∧ x < 0) → m < 4 := by
  sorry

end negative_solution_range_l26_26091


namespace choose_president_vice_president_and_committee_l26_26537

theorem choose_president_vice_president_and_committee :
  let num_ways : ℕ := 10 * 9 * (Nat.choose 8 2)
  num_ways = 2520 :=
by
  sorry

end choose_president_vice_president_and_committee_l26_26537


namespace prevent_four_digit_number_l26_26766

theorem prevent_four_digit_number (N : ℕ) (n : ℕ) :
  n = 123 + 102 * N ∧ ∀ x : ℕ, (3 + 2 * x) % 10 < 1000 → x < 1000 := 
sorry

end prevent_four_digit_number_l26_26766


namespace complex_number_equality_l26_26962

theorem complex_number_equality (a b : ℂ) : a - b = 0 ↔ a = b := sorry

end complex_number_equality_l26_26962


namespace verify_statements_l26_26665

noncomputable def f (x : ℝ) : ℝ := 10 ^ x

theorem verify_statements (x1 x2 : ℝ) (h : x1 ≠ x2) :
  (f (x1 + x2) = f x1 * f x2) ∧
  (f x1 - f x2) / (x1 - x2) > 0 :=
by
  sorry

end verify_statements_l26_26665


namespace telescoping_product_l26_26935

theorem telescoping_product :
  (∏ x in {3, 4, 5, 6, 7}, (x^3 - 1) / (x^3 + 1)) = 57 / 168 := by
  sorry

end telescoping_product_l26_26935


namespace second_percentage_increase_l26_26012

theorem second_percentage_increase :
  ∀ (P : ℝ) (x : ℝ), (P * 1.30 * (1 + x) = P * 1.5600000000000001) → x = 0.2 :=
by
  intros P x h
  sorry

end second_percentage_increase_l26_26012


namespace part1_part2_l26_26822

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l26_26822


namespace find_x_l26_26238

theorem find_x (x : ℚ) (h : (3 * x - 6 + 4) / 7 = 15) : x = 107 / 3 :=
by
  sorry

end find_x_l26_26238


namespace hilary_regular_toenails_in_jar_l26_26107

-- Conditions
def jar_capacity : Nat := 100
def big_toenail_size : Nat := 2
def num_big_toenails : Nat := 20
def remaining_regular_toenails_space : Nat := 20

-- Question & Answer
theorem hilary_regular_toenails_in_jar : 
  (jar_capacity - remaining_regular_toenails_space - (num_big_toenails * big_toenail_size)) = 40 :=
by
  sorry

end hilary_regular_toenails_in_jar_l26_26107


namespace unique_number_not_in_range_of_g_l26_26487

noncomputable def g (x : ℝ) (a b c d : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem unique_number_not_in_range_of_g 
  (a b c d : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) (h4 : d ≠ 0)
  (h5 : g 5 a b c d = 5) (h6 : g 25 a b c d = 25) 
  (h7 : ∀ x, x ≠ -d/c → g (g x a b c d) a b c d = x) :
  ∃ r, r = 15 ∧ ∀ y, g y a b c d ≠ r := 
by
  sorry

end unique_number_not_in_range_of_g_l26_26487


namespace shoe_length_increase_l26_26765

theorem shoe_length_increase
  (L : ℝ)
  (x : ℝ)
  (h1 : L + 9*x = L * 1.2)
  (h2 : L + 7*x = 10.4) :
  x = 0.2 :=
by
  sorry

end shoe_length_increase_l26_26765


namespace min_value_of_expression_ge_9_l26_26513

theorem min_value_of_expression_ge_9 
    (x : ℝ)
    (h1 : -2 < x ∧ x < -1)
    (m n : ℝ)
    (a b : ℝ)
    (ha : a = -2)
    (hb : b = -1)
    (h2 : mn > 0)
    (h3 : m * a + n * b + 1 = 0) :
    (2 / m) + (1 / n) ≥ 9 := by
  sorry

end min_value_of_expression_ge_9_l26_26513


namespace cost_equivalence_at_325_l26_26104

def cost_plan1 (x : ℕ) : ℝ := 65 + 0.40 * x
def cost_plan2 (x : ℕ) : ℝ := 0.60 * x

theorem cost_equivalence_at_325 : cost_plan1 325 = cost_plan2 325 :=
by sorry

end cost_equivalence_at_325_l26_26104


namespace min_value_m_plus_n_l26_26431

theorem min_value_m_plus_n (m n : ℕ) (hm : 0 < m) (hn : 0 < n) (h : 45 * m = n^3) : m + n = 90 :=
sorry

end min_value_m_plus_n_l26_26431


namespace cube_sum_gt_zero_l26_26556

variable {x y z : ℝ}

theorem cube_sum_gt_zero (h1 : x < y) (h2 : y < z) : 
  (x - y)^3 + (y - z)^3 + (z - x)^3 > 0 :=
sorry

end cube_sum_gt_zero_l26_26556


namespace S_30_value_l26_26417

noncomputable def geometric_sequence_sum (n : ℕ) : ℝ := sorry

axiom S_10 : geometric_sequence_sum 10 = 10
axiom S_20 : geometric_sequence_sum 20 = 30

theorem S_30_value : geometric_sequence_sum 30 = 70 :=
by
  sorry

end S_30_value_l26_26417


namespace math_problem_l26_26650

variable (x b : ℝ)
variable (h1 : x < b)
variable (h2 : b < 0)
variable (h3 : b = -2)

theorem math_problem : x^2 > b * x ∧ b * x > b^2 :=
by
  sorry

end math_problem_l26_26650


namespace units_digit_of_7_pow_y_plus_6_is_9_l26_26166

theorem units_digit_of_7_pow_y_plus_6_is_9 (y : ℕ) (hy : 0 < y) : 
  (7^y + 6) % 10 = 9 ↔ ∃ k : ℕ, y = 4 * k + 3 := by
  sorry

end units_digit_of_7_pow_y_plus_6_is_9_l26_26166


namespace no_such_n_l26_26492

theorem no_such_n (n : ℕ) (h_pos : 0 < n) :
  ¬ ∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧ A.prod id = B.prod id := 
sorry

end no_such_n_l26_26492


namespace tractor_planting_rate_l26_26070

theorem tractor_planting_rate
  (acres : ℕ) (days : ℕ) (first_crew_tractors : ℕ) (first_crew_days : ℕ) 
  (second_crew_tractors : ℕ) (second_crew_days : ℕ) 
  (total_acres : ℕ) (total_days : ℕ) 
  (first_crew_days_calculated : ℕ) 
  (second_crew_days_calculated : ℕ) 
  (total_tractor_days : ℕ) 
  (acres_per_tractor_day : ℕ) :
  total_acres = acres → 
  total_days = days → 
  first_crew_tractors * first_crew_days = first_crew_days_calculated → 
  second_crew_tractors * second_crew_days = second_crew_days_calculated → 
  first_crew_days_calculated + second_crew_days_calculated = total_tractor_days → 
  total_acres / total_tractor_days = acres_per_tractor_day → 
  acres_per_tractor_day = 68 :=
by
  intros
  sorry

end tractor_planting_rate_l26_26070


namespace eight_row_triangle_pieces_l26_26644

def unit_rods (n : ℕ) : ℕ := 3 * (n * (n + 1)) / 2

def connectors (n : ℕ) : ℕ := (n * (n + 1)) / 2

theorem eight_row_triangle_pieces : unit_rods 8 + connectors 9 = 153 :=
by
  sorry

end eight_row_triangle_pieces_l26_26644


namespace unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l26_26303

-- Definitions based on conditions
variable (unit_price quantity total_price : ℕ)
variable (map_distance actual_distance scale : ℕ)

-- Given conditions
def total_price_fixed := unit_price * quantity = total_price
def scale_fixed := map_distance * scale = actual_distance

-- Proof problem statements
theorem unit_price_quantity_inverse_proportion (h : total_price_fixed unit_price quantity total_price) :
  ∃ k : ℕ, unit_price = k / quantity := sorry

theorem map_distance_actual_distance_direct_proportion (h : scale_fixed map_distance actual_distance scale) :
  ∃ k : ℕ, map_distance * scale = k * actual_distance := sorry

end unit_price_quantity_inverse_proportion_map_distance_actual_distance_direct_proportion_l26_26303


namespace equilateral_triangle_side_length_l26_26930

variable (R : ℝ)

theorem equilateral_triangle_side_length (R : ℝ) :
  (∃ (s : ℝ), s = R * Real.sqrt 3) :=
sorry

end equilateral_triangle_side_length_l26_26930


namespace calculate_gallons_of_milk_l26_26860

-- Definitions of the given constants and conditions
def price_of_soup : Nat := 2
def price_of_bread : Nat := 5
def price_of_cereal : Nat := 3
def price_of_milk : Nat := 4
def total_amount_paid : Nat := 4 * 10

-- Calculation of total cost of non-milk items
def total_cost_non_milk : Nat :=
  (6 * price_of_soup) + (2 * price_of_bread) + (2 * price_of_cereal)

-- The function to calculate the remaining amount to be spent on milk
def remaining_amount : Nat := total_amount_paid - total_cost_non_milk

-- Statement to compute the number of gallons of milk
def gallons_of_milk (remaining : Nat) (price_per_gallon : Nat) : Nat :=
  remaining / price_per_gallon

-- Proof theorem statement (no implementation required, proof skipped)
theorem calculate_gallons_of_milk : 
  gallons_of_milk remaining_amount price_of_milk = 3 := 
by
  sorry

end calculate_gallons_of_milk_l26_26860


namespace maximize_profit_l26_26338

-- Define constants for purchase and selling prices
def priceA_purchase : ℝ := 16
def priceA_selling : ℝ := 20
def priceB_purchase : ℝ := 20
def priceB_selling : ℝ := 25

-- Define constant for total weight
def total_weight : ℝ := 200

-- Define profit function
def profit (weightA weightB : ℝ) : ℝ :=
  (priceA_selling - priceA_purchase) * weightA + (priceB_selling - priceB_purchase) * weightB

-- Define constraints
def constraint1 (weightA weightB : ℝ) : Prop :=
  weightA + weightB = total_weight

def constraint2 (weightA weightB : ℝ) : Prop :=
  weightA >= 3 * weightB

open Real

-- Define the maximum profit we aim to prove
def max_profit : ℝ := 850

-- The main theorem to prove
theorem maximize_profit : 
  ∃ weightA weightB : ℝ, constraint1 weightA weightB ∧ constraint2 weightA weightB ∧ profit weightA weightB = max_profit :=
by {
  sorry
}

end maximize_profit_l26_26338


namespace polygon_sides_count_l26_26044

theorem polygon_sides_count :
    ∀ (n1 n2 n3 n4 n5 n6 : ℕ),
    n1 = 3 ∧ n2 = 4 ∧ n3 = 5 ∧ n4 = 6 ∧ n5 = 7 ∧ n6 = 8 →
    (n1 - 2) + (n2 - 2) + (n3 - 2) + (n4 - 2) + (n5 - 2) + (n6 - 1) + 3 = 24 :=
by
  intros n1 n2 n3 n4 n5 n6 h
  sorry

end polygon_sides_count_l26_26044


namespace find_x_for_opposite_directions_l26_26103

-- Define the vectors and the opposite direction condition
def vector_a (x : ℝ) : ℝ × ℝ := (1, -x)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -16)

-- Define the condition that vectors are in opposite directions
def opp_directions (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ a = (-k) • b

-- The main theorem statement
theorem find_x_for_opposite_directions : ∃ x : ℝ, opp_directions (vector_a x) (vector_b x) ∧ x = -5 := 
sorry

end find_x_for_opposite_directions_l26_26103


namespace geometric_sequence_a7_l26_26697

noncomputable def a_n (a_1 q : ℝ) (n : ℕ) := a_1 * q^(n - 1)

theorem geometric_sequence_a7 
  (a1 q : ℝ)
  (a1_neq_zero : a1 ≠ 0)
  (a9_eq_256 : a_n a1 q 9 = 256)
  (a1_a3_eq_4 : a_n a1 q 1 * a_n a1 q 3 = 4) :
  a_n a1 q 7 = 64 := 
sorry

end geometric_sequence_a7_l26_26697


namespace resulting_polygon_sides_l26_26345

theorem resulting_polygon_sides :
  let square_sides := 4
  let pentagon_sides := 5
  let hexagon_sides := 6
  let heptagon_sides := 7
  let octagon_sides := 8
  let nonagon_sides := 9
  let decagon_sides := 10
  let shared_square_decagon := 2
  let shared_between_others := 2 * 5 -- 2 sides shared for pentagon to nonagon
  let total_shared_sides := shared_square_decagon + shared_between_others
  let total_unshared_sides := 
    square_sides + pentagon_sides + hexagon_sides + heptagon_sides + octagon_sides + nonagon_sides + decagon_sides
  total_unshared_sides - total_shared_sides = 37 := by
  sorry

end resulting_polygon_sides_l26_26345


namespace cut_rectangle_to_square_l26_26200

theorem cut_rectangle_to_square (a b : ℕ) (h₁ : a = 16) (h₂ : b = 9) :
  ∃ (s : ℕ), s * s = a * b ∧ s = 12 :=
by {
  sorry
}

end cut_rectangle_to_square_l26_26200


namespace mean_first_second_fifth_sixth_diff_l26_26674

def six_numbers_arithmetic_mean_condition (a1 a2 a3 a4 a5 a6 A : ℝ) :=
  (a1 + a2 + a3 + a4 + a5 + a6) / 6 = A

def mean_first_four_numbers (a1 a2 a3 a4 A : ℝ) :=
  (a1 + a2 + a3 + a4) / 4 = A + 10

def mean_last_four_numbers (a3 a4 a5 a6 A : ℝ) :=
  (a3 + a4 + a5 + a6) / 4 = A - 7

theorem mean_first_second_fifth_sixth_diff (a1 a2 a3 a4 a5 a6 A : ℝ) :
  six_numbers_arithmetic_mean_condition a1 a2 a3 a4 a5 a6 A →
  mean_first_four_numbers a1 a2 a3 a4 A →
  mean_last_four_numbers a3 a4 a5 a6 A →
  ((a1 + a2 + a5 + a6) / 4) = A - 3 :=
by
  intros h1 h2 h3
  sorry

end mean_first_second_fifth_sixth_diff_l26_26674


namespace length_of_bridge_l26_26907

noncomputable def train_length : ℝ := 155
noncomputable def train_speed_km_hr : ℝ := 45
noncomputable def crossing_time_seconds : ℝ := 30

noncomputable def train_speed_m_s : ℝ := train_speed_km_hr * 1000 / 3600

noncomputable def total_distance : ℝ := train_speed_m_s * crossing_time_seconds

noncomputable def bridge_length : ℝ := total_distance - train_length

theorem length_of_bridge : bridge_length = 220 := by
  sorry

end length_of_bridge_l26_26907


namespace apples_per_pie_l26_26728

-- Definitions of given conditions
def total_apples : ℕ := 75
def handed_out_apples : ℕ := 19
def remaining_apples : ℕ := total_apples - handed_out_apples
def pies_made : ℕ := 7

-- Statement of the problem to be proved
theorem apples_per_pie : remaining_apples / pies_made = 8 := by
  sorry

end apples_per_pie_l26_26728


namespace classification_of_square_and_cube_roots_l26_26026

-- Define the three cases: positive, zero, and negative
inductive NumberCase
| positive 
| zero 
| negative 

-- Define the concept of "classification and discussion thinking"
def is_classification_and_discussion_thinking (cases : List NumberCase) : Prop :=
  cases = [NumberCase.positive, NumberCase.zero, NumberCase.negative]

-- The main statement to be proven
theorem classification_of_square_and_cube_roots :
  is_classification_and_discussion_thinking [NumberCase.positive, NumberCase.zero, NumberCase.negative] :=
by
  sorry

end classification_of_square_and_cube_roots_l26_26026


namespace equal_cost_miles_l26_26105

   -- Conditions:
   def initial_fee_first_plan : ℝ := 65
   def cost_per_mile_first_plan : ℝ := 0.40
   def cost_per_mile_second_plan : ℝ := 0.60

   -- Proof problem:
   theorem equal_cost_miles : 
     let x := 325 in 
     initial_fee_first_plan + cost_per_mile_first_plan * x = cost_per_mile_second_plan * x :=
   by
     -- Placeholder for the proof
     sorry
   
end equal_cost_miles_l26_26105


namespace sales_last_year_l26_26923

theorem sales_last_year (x : ℝ) (h1 : 416 = (1 + 0.30) * x) : x = 320 :=
by
  sorry

end sales_last_year_l26_26923


namespace justin_reading_ratio_l26_26410

theorem justin_reading_ratio
  (pages_total : ℝ)
  (pages_first_day : ℝ)
  (pages_left : ℝ)
  (days_remaining : ℝ) :
  pages_total = 130 → 
  pages_first_day = 10 → 
  pages_left = pages_total - pages_first_day →
  days_remaining = 6 →
  (∃ R : ℝ, 60 * R = pages_left) → 
  ∃ R : ℝ, 60 * R = pages_left ∧ R = 2 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end justin_reading_ratio_l26_26410


namespace tractor_planting_rate_l26_26067

theorem tractor_planting_rate
  (A : ℕ) (D : ℕ)
  (T1_days : ℕ) (T1 : ℕ)
  (T2_days : ℕ) (T2 : ℕ)
  (total_acres : A = 1700)
  (total_days : D = 5)
  (crew1_tractors : T1 = 2)
  (crew1_days : T1_days = 2)
  (crew2_tractors : T2 = 7)
  (crew2_days : T2_days = 3)
  : (A / (T1 * T1_days + T2 * T2_days)) = 68 := 
sorry

end tractor_planting_rate_l26_26067


namespace p_or_q_not_necessarily_true_l26_26693

theorem p_or_q_not_necessarily_true (p q : Prop) (hnp : ¬p) (hpq : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) :=
by
  sorry

end p_or_q_not_necessarily_true_l26_26693


namespace calculate_value_l26_26112

theorem calculate_value :
  let X := (354 * 28) ^ 2
  let Y := (48 * 14) ^ 2
  (X * 9) / (Y * 2) = 2255688 :=
by
  sorry

end calculate_value_l26_26112


namespace number_of_unique_numbers_l26_26391

theorem number_of_unique_numbers :
  let digits := [3, 3, 3, 7, 7] in
  list.permutations digits |>.length = 10 :=
by
  have : multiset.card (multiset.pmap (λ _ _, 1) [3, 3, 3, 7, 7] _) = 10 :=
    sorry
  exact this

end number_of_unique_numbers_l26_26391


namespace samantha_marble_choices_l26_26574

open BigOperators

noncomputable def choose_five_with_at_least_one_red (total_marbles red_marbles marbles_needed : ℕ) : ℕ :=
choose total_marbles marbles_needed - choose (total_marbles - red_marbles) marbles_needed

theorem samantha_marble_choices :
  choose_five_with_at_least_one_red 10 1 5 = 126 :=
by
  sorry

end samantha_marble_choices_l26_26574


namespace largest_n_consecutive_product_l26_26359

theorem largest_n_consecutive_product (n : ℕ) : n = 0 ↔ (n! = (n+1) * (n+2) * (n+3) * (n+4) * (n+5)) := by
  sorry

end largest_n_consecutive_product_l26_26359


namespace squirrels_in_tree_l26_26441

-- Definitions based on the conditions
def nuts : Nat := 2
def squirrels : Nat := nuts + 2

-- Theorem stating the main proof problem
theorem squirrels_in_tree : squirrels = 4 := by
  -- Proof steps would go here, but we're adding sorry to skip them
  sorry

end squirrels_in_tree_l26_26441


namespace find_c_l26_26089

variable (x y c : ℝ)

def condition1 : Prop := 2 * x + 5 * y = 3
def condition2 : Prop := c = Real.sqrt (4^(x + 1/2) * 32^y)

theorem find_c (h1 : condition1 x y) (h2 : condition2 x y c) : c = 4 := by
  sorry

end find_c_l26_26089


namespace fraction_of_fritz_money_l26_26866

theorem fraction_of_fritz_money
  (Fritz_money : ℕ)
  (total_amount : ℕ)
  (fraction : ℚ)
  (Sean_money : ℚ)
  (Rick_money : ℚ)
  (h1 : Fritz_money = 40)
  (h2 : total_amount = 96)
  (h3 : Sean_money = fraction * Fritz_money + 4)
  (h4 : Rick_money = 3 * Sean_money)
  (h5 : Rick_money + Sean_money = total_amount) :
  fraction = 1 / 2 :=
by
  sorry

end fraction_of_fritz_money_l26_26866


namespace integer_solutions_inequality_system_l26_26870

theorem integer_solutions_inequality_system :
  {x : ℤ | 2 * (x - 1) ≤ x + 3 ∧ (x + 1) / 3 < x - 1} = {3, 4, 5} :=
by
  sorry

end integer_solutions_inequality_system_l26_26870


namespace acute_triangle_statements_l26_26121

variable (A B C : ℝ)

-- Conditions for acute triangle
def acute_triangle := A + B + C = π ∧ 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2

-- Statement A: If A > B, then sin A > sin B.
def statement_A := ∀ h : A > B, Real.sin A > Real.sin B

-- Statement B: If A = π / 3, then the range of values for B is (0, π / 2).
def statement_B := ∀ h : A = π / 3, 0 < B ∧ B < π / 2

-- Statement C: sin A + sin B > cos A + cos B
def statement_C := Real.sin A + Real.sin B > Real.cos A + Real.cos B

-- Statement D: tan B tan C > 1
def statement_D := Real.tan B * Real.tan C > 1

-- The theorem to prove
theorem acute_triangle_statements (h : acute_triangle A B C) :
  statement_A A B C ∧ ¬statement_B A B C ∧ statement_C A B C ∧ statement_D A B C :=
by sorry

end acute_triangle_statements_l26_26121


namespace part1_part2_l26_26826

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l26_26826


namespace find_k_eq_l26_26754

theorem find_k_eq (n : ℝ) (k m : ℤ) (h : ∀ n : ℝ, n * (n + 1) * (n + 2) * (n + 3) + m = (n^2 + k * n + 1)^2) : k = 3 := 
sorry

end find_k_eq_l26_26754


namespace mr_johnson_pill_intake_l26_26562

theorem mr_johnson_pill_intake (total_days : ℕ) (remaining_pills : ℕ) (fraction : ℚ) (dose : ℕ)
  (h1 : total_days = 30)
  (h2 : remaining_pills = 12)
  (h3 : fraction = 4 / 5) :
  dose = 2 :=
by
  sorry

end mr_johnson_pill_intake_l26_26562


namespace cube_mono_l26_26525

theorem cube_mono {a b : ℝ} (h : a > b) : a^3 > b^3 :=
sorry

end cube_mono_l26_26525


namespace calculate_remainder_l26_26050

open Nat

theorem calculate_remainder :
  let ω := complex.exp (2 * complex.pi * complex.I / 4) in
  ω^4 = 1 ∧ ω ≠ 1 ∧ ω^2 = -1 ∧ ω^3 = -ω ∧
  let S := (1 + ω)^2011 + (1 + ω^2)^2011 + (1 + ω^3)^2011 + (2:ℂ)^2011 in
  S = 4 * ∑ k in range (503), nat.choose 2011 (4 * k) →
  (1 + ω^2)^2011 = 0 ∧ (1 + ω)^2011 + (1 + ω^3)^2011 = 0 →
  S = (2:ℂ)^2011 →
  (2^2011 : ℕ) % 8 = 0 ∧ (2^2011 : ℕ) % 125 = 48 →
  (2^2011 : ℕ) % 1000 = 48 →
  (4 * ∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 48 →
  ((∑ k in range (503), nat.choose 2011 (4 * k)) % 1000 = 12) :=
by
  intros ω ω4 ω_ne ω2 ω3 S hS h1 h2 h3 h4 h5
  sorry

end calculate_remainder_l26_26050


namespace probability_product_less_than_30_l26_26260

-- Define the spinner outcomes
def PacoSpinner := {n : ℕ // 1 ≤ n ∧ n ≤ 5}
def ManuSpinner := {n : ℕ // 1 ≤ n ∧ n ≤ 12}

-- Define the probability space for Paco and Manu's spinners
noncomputable def uniform_prob {α : Type} [Fintype α] : ProbabilityMeasure α :=
Fintype.uniformProbability α

-- Event definition
def eventProductLessThanThirty (p : PacoSpinner) (m : ManuSpinner) : Prop :=
p.val * m.val < 30

-- Main theorem statement
theorem probability_product_less_than_30 :
  ProbabilityMeasure.toMeasure (uniform_prob : ProbabilityMeasure (PacoSpinner × ManuSpinner))
    (λ pm : PacoSpinner × ManuSpinner, eventProductLessThanThirty pm.1 pm.2) =
  51 / 60 := 
sorry

end probability_product_less_than_30_l26_26260


namespace a_11_is_12_l26_26229

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d
def a_2 (a : ℕ → ℝ) := a 2 = 3
def a_6 (a : ℕ → ℝ) := a 6 = 7

-- The statement to prove
theorem a_11_is_12 (a : ℕ → ℝ) (h_arith : arithmetic_sequence a) (h_a2 : a_2 a) (h_a6 : a_6 a) : a 11 = 12 :=
  sorry

end a_11_is_12_l26_26229


namespace apple_counting_l26_26192

theorem apple_counting
  (n m : ℕ)
  (vasya_trees_a_b petya_trees_a_b vasya_trees_b_c petya_trees_b_c vasya_trees_c_d petya_trees_c_d vasya_apples_a_b petya_apples_a_b vasya_apples_c_d petya_apples_c_d : ℕ)
  (h1 : petya_trees_a_b = 2 * vasya_trees_a_b)
  (h2 : petya_apples_a_b = 7 * vasya_apples_a_b)
  (h3 : petya_trees_b_c = 2 * vasya_trees_b_c)
  (h4 : petya_trees_c_d = 2 * vasya_trees_c_d)
  (h5 : n = vasya_trees_a_b + petya_trees_a_b)
  (h6 : m = vasya_apples_a_b + petya_apples_a_b)
  (h7 : vasya_trees_c_d = n / 3)
  (h8 : petya_trees_c_d = 2 * (n / 3))
  (h9 : vasya_apples_c_d = 3 * petya_apples_c_d)
  : vasya_apples_c_d = 3 * petya_apples_c_d :=
by 
  sorry

end apple_counting_l26_26192


namespace part1_part2_l26_26806

variable (a b c : ℝ)

-- Conditions
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0
axiom eq_sum_square : a^2 + b^2 + 4*c^2 = 3

-- Part 1
theorem part1 : a + b + 2 * c ≤ 3 := 
by
  sorry

-- Additional condition for Part 2
variable (hc : b = 2*c)

-- Part 2
theorem part2 : (1 / a) + (1 / c) ≥ 3 :=
by
  sorry

end part1_part2_l26_26806


namespace warehouse_rental_comparison_purchase_vs_rent_comparison_l26_26628

-- Define the necessary constants and conditions
def monthly_cost_first : ℕ := 50000
def monthly_cost_second : ℕ := 10000
def moving_cost : ℕ := 70000
def months_in_year : ℕ := 12
def purchase_cost : ℕ := 2000000
def duration_installments : ℕ := 3 * 12 -- 3 years in months
def worst_case_prob : ℕ := 50

-- Question (a)
theorem warehouse_rental_comparison
  (annual_cost_first : ℕ := monthly_cost_first * months_in_year)
  (cost_second_4months : ℕ := monthly_cost_second * 4)
  (cost_switching : ℕ := moving_cost)
  (cost_first_8months : ℕ := monthly_cost_first * 8)
  (worst_case_cost_second : ℕ := cost_second_4months + cost_first_8months + cost_switching) :
  annual_cost_first > worst_case_cost_second :=
by
  sorry

-- Question (b)
theorem purchase_vs_rent_comparison
  (total_rent_cost_4years : ℕ := 4 * annual_cost_first + worst_case_cost_second)
  (total_purchase_cost : ℕ := purchase_cost) :
  total_rent_cost_4years > total_purchase_cost :=
by
  sorry

end warehouse_rental_comparison_purchase_vs_rent_comparison_l26_26628


namespace hyperbola_eccentricity_l26_26783

theorem hyperbola_eccentricity : 
  (∃ (a b : ℝ), (a^2 = 1 ∧ b^2 = 2) ∧ ∀ e : ℝ, e = Real.sqrt (1 + b^2 / a^2) → e = Real.sqrt 3) :=
by 
  sorry

end hyperbola_eccentricity_l26_26783


namespace find_matrix_A_l26_26946

-- Define the condition that A v = 3 v for all v in R^3
def satisfiesCondition (A : Matrix (Fin 3) (Fin 3) ℝ) : Prop :=
  ∀ (v : Fin 3 → ℝ), A.mulVec v = 3 • v

theorem find_matrix_A (A : Matrix (Fin 3) (Fin 3) ℝ) :
  satisfiesCondition A → A = 3 • 1 :=
by
  intro h
  sorry

end find_matrix_A_l26_26946


namespace constant_term_in_expansion_l26_26654

theorem constant_term_in_expansion : 
  let a := (x : ℝ)
  let b := - (2 / Real.sqrt x)
  let n := 6
  let general_term (r : Nat) : ℝ := Nat.choose n r * a * (b ^ (n - r))
  (∀ x : ℝ, ∃ (r : Nat), r = 4 ∧ (1 - (n - r) / 2 = 0) →
  general_term 4 = 60) :=
by
  sorry

end constant_term_in_expansion_l26_26654


namespace min_area_of_rectangle_with_perimeter_100_l26_26768

theorem min_area_of_rectangle_with_perimeter_100 :
  ∃ (length width : ℕ), 
    (length + width = 50) ∧ 
    (length * width = 49) := 
by
  sorry

end min_area_of_rectangle_with_perimeter_100_l26_26768


namespace initial_average_weight_l26_26004

theorem initial_average_weight (a b c d e : ℝ) (A : ℝ) 
    (h1 : (a + b + c) / 3 = A) 
    (h2 : (a + b + c + d) / 4 = 80) 
    (h3 : e = d + 3) 
    (h4 : (b + c + d + e) / 4 = 79) 
    (h5 : a = 75) : A = 84 :=
sorry

end initial_average_weight_l26_26004


namespace intersection_A_B_l26_26087

open Set

-- Given definitions of sets A and B
def A : Set ℤ := {-1, 0, 1, 2}
def B : Set ℤ := {x : ℤ | x^2 - 2 * x ≥ 0}

-- Theorem statement
theorem intersection_A_B :
  A ∩ B = {-1, 0, 2} :=
sorry

end intersection_A_B_l26_26087


namespace intersection_A_B_l26_26398

def setA : Set ℝ := {x | x^2 - 1 < 0}
def setB : Set ℝ := {x | x > 0}

theorem intersection_A_B : setA ∩ setB = {x | 0 < x ∧ x < 1} := 
by 
  sorry

end intersection_A_B_l26_26398


namespace log_6_15_expression_l26_26503

theorem log_6_15_expression (a b : ℝ) (h1 : Real.log 2 = a) (h2 : Real.log 3 = b) :
  Real.log 15 / Real.log 6 = (b + 1 - a) / (a + b) :=
sorry

end log_6_15_expression_l26_26503


namespace parameter_values_l26_26211

def system_equation_1 (x y : ℝ) : Prop :=
  (|y + 9| + |x + 2| - 2) * (x^2 + y^2 - 3) = 0

def system_equation_2 (x y a : ℝ) : Prop :=
  (x + 2)^2 + (y + 4)^2 = a

theorem parameter_values (a : ℝ) :
  (∃ x y : ℝ, system_equation_1 x y ∧ system_equation_2 x y a ∧ 
    -- counting the number of solutions to the system of equations that total exactly three,
    -- meaning the system has exactly three solutions
    -- Placeholder for counting solutions
    sorry) ↔ (a = 9 ∨ a = 23 + 4 * Real.sqrt 15) := 
sorry

end parameter_values_l26_26211


namespace domain_of_function_l26_26214

theorem domain_of_function :
  {x : ℝ | x < -1 ∨ 4 ≤ x} = {x : ℝ | (x^2 - 7*x + 12) / (x^2 - 2*x - 3) ≥ 0} \ {3} :=
by
  sorry

end domain_of_function_l26_26214


namespace all_children_receive_candy_iff_power_of_two_l26_26059

theorem all_children_receive_candy_iff_power_of_two (n : ℕ) : 
  (∀ (k : ℕ), k < n → ∃ (m : ℕ), (m * (m + 1) / 2) % n = k) ↔ ∃ (k : ℕ), n = 2^k :=
by sorry

end all_children_receive_candy_iff_power_of_two_l26_26059


namespace cricket_players_count_l26_26984

theorem cricket_players_count (hockey: ℕ) (football: ℕ) (softball: ℕ) (total: ℕ) : 
  hockey = 15 ∧ football = 21 ∧ softball = 19 ∧ total = 77 → ∃ cricket, cricket = 22 := by
  sorry

end cricket_players_count_l26_26984


namespace daniel_gpa_probability_l26_26983

theorem daniel_gpa_probability :
  let A_points := 4
  let B_points := 3
  let C_points := 2
  let D_points := 1
  let total_classes := 4
  let gpa (points: ℚ) := points / total_classes
  let min_gpa := (13 : ℚ) / total_classes
  let prob_A_eng := (1 : ℚ) / 5
  let prob_B_eng := (1 : ℚ) / 3
  let prob_C_eng := 1 - prob_A_eng - prob_B_eng
  let prob_A_sci := (1 : ℚ) / 3
  let prob_B_sci := (1 : ℚ) / 2
  let prob_C_sci := (1 : ℚ) / 6
  let required_points := 13
  let points_math := A_points
  let points_hist := A_points
  let points_achieved := points_math + points_hist
  let needed_points := required_points - points_achieved
  let success_prob := prob_A_eng * prob_A_sci + 
                      prob_A_eng * prob_B_sci + 
                      prob_B_eng * prob_A_sci + 
                      prob_B_eng * prob_B_sci 
  in success_prob = (4 : ℚ) / 9 := sorry

end daniel_gpa_probability_l26_26983


namespace uki_total_earnings_l26_26601

-- Define the conditions
def price_cupcake : ℝ := 1.50
def price_cookie : ℝ := 2
def price_biscuit : ℝ := 1
def cupcakes_per_day : ℕ := 20
def cookies_per_day : ℕ := 10
def biscuits_per_day : ℕ := 20
def days : ℕ := 5

-- Prove the total earnings for five days
theorem uki_total_earnings : 
    (cupcakes_per_day * price_cupcake + 
     cookies_per_day * price_cookie + 
     biscuits_per_day * price_biscuit) * days = 350 := 
by
  -- The actual proof will go here, but is omitted for now.
  sorry

end uki_total_earnings_l26_26601


namespace children_tickets_l26_26621

theorem children_tickets (A C : ℝ) (h1 : A + C = 200) (h2 : 3 * A + 1.5 * C = 510) : C = 60 := by
  sorry

end children_tickets_l26_26621


namespace minimize_on_interval_l26_26948

def f (x a : ℝ) : ℝ := x^2 - 2*a*x - 2

theorem minimize_on_interval (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 2 → f x a ≥ if a < 0 then -2 else if 0 ≤ a ∧ a ≤ 2 then -a^2 - 2 else 2 - 4*a) :=
by 
  sorry

end minimize_on_interval_l26_26948


namespace max_viewers_per_week_l26_26315

theorem max_viewers_per_week :
  ∃ (x y : ℕ), 80 * x + 40 * y ≤ 320 ∧ x + y ≥ 6 ∧ 600000 * x + 200000 * y = 2000000 :=
by
  sorry

end max_viewers_per_week_l26_26315


namespace commodity_price_difference_l26_26700

theorem commodity_price_difference (r : ℝ) (t : ℕ) :
  let P_X (t : ℕ) := 4.20 * (1 + (2*r + 10)/100)^(t - 2001)
  let P_Y (t : ℕ) := 4.40 * (1 + (r + 15)/100)^(t - 2001)
  P_X t = P_Y t + 0.90  ->
  ∃ t : ℕ, true :=
by
  sorry

end commodity_price_difference_l26_26700


namespace simplest_quadratic_radical_l26_26167
  
theorem simplest_quadratic_radical (A B C D: ℝ) 
  (hA : A = Real.sqrt 0.1) 
  (hB : B = Real.sqrt (-2)) 
  (hC : C = 3 * Real.sqrt 2) 
  (hD : D = -Real.sqrt 20) : C = 3 * Real.sqrt 2 :=
by
  have h1 : ∀ (x : ℝ), Real.sqrt x = Real.sqrt x := sorry
  sorry

end simplest_quadratic_radical_l26_26167


namespace words_per_page_l26_26914

theorem words_per_page (p : ℕ) (h1 : p ≤ 150) (h2 : 120 * p ≡ 172 [MOD 221]) : p = 114 := by
  sorry

end words_per_page_l26_26914


namespace find_a_for_odd_function_l26_26710

noncomputable def f (a x : ℝ) : ℝ := ((x + 1) * (x + a)) / x

theorem find_a_for_odd_function (a : ℝ) :
  (∀ x : ℝ, f a x + f a (-x) = 0) ↔ a = -1 := sorry

end find_a_for_odd_function_l26_26710


namespace height_of_second_triangle_l26_26616

theorem height_of_second_triangle
  (base1 : ℝ) (height1 : ℝ) (base2 : ℝ) (height2 : ℝ)
  (h_base1 : base1 = 15)
  (h_height1 : height1 = 12)
  (h_base2 : base2 = 20)
  (h_area_relation : (base2 * height2) / 2 = 2 * (base1 * height1) / 2) :
  height2 = 18 :=
sorry

end height_of_second_triangle_l26_26616


namespace jelly_cost_l26_26944

theorem jelly_cost (N B J : ℕ) (hN_gt_1 : N > 1) (h_cost_eq : N * (3 * B + 7 * J) = 252) : 7 * N * J = 168 := by
  sorry

end jelly_cost_l26_26944


namespace value_of_expression_l26_26797

theorem value_of_expression (m : ℝ) (h : m^2 + m - 1 = 0) : m^3 + 2*m^2 + 2006 = 2007 :=
sorry

end value_of_expression_l26_26797


namespace volume_of_given_tetrahedron_l26_26960

noncomputable def volume_of_tetrahedron (radius : ℝ) (total_length : ℝ) : ℝ := 
  let R := radius
  let L := total_length
  let a := (2 * Real.sqrt 33) / 3
  let V := (a^3 * Real.sqrt 2) / 12
  V

theorem volume_of_given_tetrahedron :
  volume_of_tetrahedron (Real.sqrt 22 / 2) (8 * Real.pi) = 48 := 
  sorry

end volume_of_given_tetrahedron_l26_26960


namespace Emily_walks_more_distance_than_Troy_l26_26161

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end Emily_walks_more_distance_than_Troy_l26_26161


namespace value_range_f_at_4_l26_26798

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_range_f_at_4 (f : ℝ → ℝ)
  (h1 : 1 ≤ f (-1) ∧ f (-1) ≤ 2)
  (h2 : 1 ≤ f (1) ∧ f (1) ≤ 3)
  (h3 : 2 ≤ f (2) ∧ f (2) ≤ 4)
  (h4 : -1 ≤ f (3) ∧ f (3) ≤ 1) :
  -21.75 ≤ f 4 ∧ f 4 ≤ 1 :=
sorry

end value_range_f_at_4_l26_26798


namespace probability_first_heart_second_spades_or_clubs_l26_26295

-- Defining probabilities and conditions in the problem
variable (deck : Finset ℕ)
variable (n_cards : ℕ)
variable (hearts : Finset ℕ)
variable (spades : Finset ℕ)
variable (clubs : Finset ℕ)

-- Definitions of card sets
def is_standard_deck := deck.card = 52
def is_hearts_set := hearts.card = 13
def is_spades_set := spades.card = 13
def is_clubs_set := clubs.card = 13

-- Defining events
def first_card_heart : Prop := (13 : ℝ) / (52 : ℝ) = 1 / 4
def second_card_spades_or_clubs : Prop := (26 : ℝ) / (51 : ℝ) = 26 / 51

-- Proof goal
theorem probability_first_heart_second_spades_or_clubs :
  is_standard_deck deck →
  is_hearts_set hearts →
  is_spades_set spades →
  is_clubs_set clubs →
  first_card_heart →
  second_card_spades_or_clubs →
  (1 / 4 * 26 / 51 = 13 / 102) :=
by
  intro h1 h2 h3 h4 h5 h6
  sorry

end probability_first_heart_second_spades_or_clubs_l26_26295


namespace sheep_count_l26_26288

theorem sheep_count (cows sheep shepherds : ℕ) 
  (h_cows : cows = 12) 
  (h_ears : 2 * cows < sheep) 
  (h_legs : sheep < 4 * cows) 
  (h_shepherds : sheep = 12 * shepherds) :
  sheep = 36 :=
by {
  sorry
}

end sheep_count_l26_26288


namespace natural_pair_prime_ratio_l26_26210

noncomputable def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem natural_pair_prime_ratio :
  ∃ (x y : ℕ), (x = 14 ∧ y = 2) ∧ is_prime (x * y^3 / (x + y)) :=
by
  use 14
  use 2
  sorry

end natural_pair_prime_ratio_l26_26210


namespace father_current_age_is_85_l26_26788

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l26_26788


namespace find_angle_B_l26_26102

noncomputable def triangle_sides_and_angles 
(a b c : ℝ) (A B C : ℝ) : Prop :=
  a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C

noncomputable def vectors_parallel 
(A B C a b c : ℝ) : Prop :=
  (Real.sin B - Real.sin A) / Real.sin C = (Real.sqrt 3 * a + c) / (a + b)

theorem find_angle_B (A B C a b c : ℝ)
  (h_triangle : triangle_sides_and_angles a b c A B C)
  (h_parallel : vectors_parallel A B C a b c) :
  B = 5 * Real.pi / 6 :=
sorry

end find_angle_B_l26_26102


namespace sufficient_not_necessary_l26_26224

theorem sufficient_not_necessary (a b : ℝ) : (a^2 + b^2 ≤ 2) → (-1 ≤ a * b ∧ a * b ≤ 1) ∧ ¬((-1 ≤ a * b ∧ a * b ≤ 1) → a^2 + b^2 ≤ 2) := 
by
  sorry

end sufficient_not_necessary_l26_26224


namespace constant_term_of_expansion_l26_26298

noncomputable def constant_term := 
  (20: ℕ) * (216: ℕ) * (1/27: ℚ) = (160: ℕ)

theorem constant_term_of_expansion : constant_term :=
  by sorry

end constant_term_of_expansion_l26_26298


namespace C_gets_more_than_D_by_500_l26_26327

-- Definitions based on conditions
def proportionA := 5
def proportionB := 2
def proportionC := 4
def proportionD := 3

def totalProportion := proportionA + proportionB + proportionC + proportionD

def A_share := 2500
def totalMoney := A_share * (totalProportion / proportionA)

def C_share := (proportionC / totalProportion) * totalMoney
def D_share := (proportionD / totalProportion) * totalMoney

-- The theorem stating the final question
theorem C_gets_more_than_D_by_500 : C_share - D_share = 500 := by
  sorry

end C_gets_more_than_D_by_500_l26_26327


namespace sum_abs_a_l26_26383

def S (n : ℕ) : ℤ := n^2 - 4 * n + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem sum_abs_a :
  (|a 1| + |a 2| + |a 3| + |a 4| + |a 5| + 
   |a 6| + |a 7| + |a 8| + |a 9| + |a 10| = 67) :=
by
  sorry

end sum_abs_a_l26_26383


namespace JohnsonFamilySeating_l26_26883

theorem JohnsonFamilySeating : 
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 359000 := by
  let total_arrangements := Nat.factorial 9
  let restricted_arrangements := Nat.factorial 5 * Nat.factorial 4
  show total_arrangements - restricted_arrangements = 359000 from sorry

end JohnsonFamilySeating_l26_26883


namespace travel_cost_from_B_to_C_l26_26140

noncomputable def distance (x y : ℝ) : ℝ :=
  Real.sqrt (x^2 + y^2)

noncomputable def travel_cost_by_air (distance : ℝ) (booking_fee : ℝ) (per_km_cost : ℝ) : ℝ :=
  booking_fee + (distance * per_km_cost)

theorem travel_cost_from_B_to_C :
  let AC := 4000
  let AB := 4500
  let BC := Real.sqrt (AB^2 - AC^2)
  let booking_fee := 120
  let per_km_cost := 0.12
  travel_cost_by_air BC booking_fee per_km_cost = 367.39 := by
  sorry

end travel_cost_from_B_to_C_l26_26140


namespace pills_per_day_l26_26564

theorem pills_per_day (total_days : ℕ) (prescription_days_frac : ℚ) (remaining_pills : ℕ) (days_taken : ℕ) (remaining_days : ℕ) (pills_per_day : ℕ)
  (h1 : total_days = 30)
  (h2 : prescription_days_frac = 4/5)
  (h3 : remaining_pills = 12)
  (h4 : days_taken = prescription_days_frac * total_days)
  (h5 : remaining_days = total_days - days_taken)
  (h6 : pills_per_day = remaining_pills / remaining_days) :
  pills_per_day = 2 := by
  sorry

end pills_per_day_l26_26564


namespace max_3x_4y_eq_73_l26_26090

theorem max_3x_4y_eq_73 :
  (∀ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 → 3 * x + 4 * y ≤ 73) ∧
  (∃ x y : ℝ, x ^ 2 + y ^ 2 = 14 * x + 6 * y + 6 ∧ 3 * x + 4 * y = 73) :=
by sorry

end max_3x_4y_eq_73_l26_26090


namespace lowest_possible_price_l26_26612

theorem lowest_possible_price 
  (MSRP : ℝ)
  (regular_discount_percentage additional_discount_percentage : ℝ)
  (h1 : MSRP = 40)
  (h2 : regular_discount_percentage = 0.30)
  (h3 : additional_discount_percentage = 0.20) : 
  (MSRP * (1 - regular_discount_percentage) * (1 - additional_discount_percentage) = 22.40) := 
by
  sorry

end lowest_possible_price_l26_26612


namespace three_digit_numbers_with_repeats_l26_26899

theorem three_digit_numbers_with_repeats :
  (let total_numbers := 9 * 10 * 10
   let non_repeating_numbers := 9 * 9 * 8
   total_numbers - non_repeating_numbers = 252) :=
by
  sorry

end three_digit_numbers_with_repeats_l26_26899


namespace tank_width_problem_l26_26771

noncomputable def tank_width (cost_per_sq_meter : ℚ) (total_cost : ℚ) (length depth : ℚ) : ℚ :=
  let total_cost_in_paise := total_cost * 100
  let total_area := total_cost_in_paise / cost_per_sq_meter
  let w := (total_area - (2 * length * depth) - (2 * depth * 6)) / (length + 2 * depth)
  w

theorem tank_width_problem :
  tank_width 55 409.20 25 6 = 12 := 
by 
  sorry

end tank_width_problem_l26_26771


namespace part1_part2_l26_26834

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l26_26834


namespace minimum_value_of_f_l26_26846

open Real

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2) / x

theorem minimum_value_of_f (h : 1 < x) : ∃ y, f x = y ∧ (∀ z, (f z) ≥ 2*sqrt 2) :=
by
  sorry

end minimum_value_of_f_l26_26846


namespace father_l26_26792

noncomputable def father's_current_age : ℕ :=
  let S : ℕ := 40 -- Sebastian's current age
  let Si : ℕ := S - 10 -- Sebastian's sister's current age
  let sum_five_years_ago := (S - 5) + (Si - 5) -- Sum of their ages five years ago
  let father_age_five_years_ago := (4 * sum_five_years_ago) / 3 -- From the given condition
  father_age_five_years_ago + 5 -- Their father's current age

theorem father's_age_is_85 : father's_current_age = 85 :=
  sorry

end father_l26_26792


namespace vector_addition_l26_26954

-- Let vectors a and b be defined as
def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -3)

-- Theorem statement to prove
theorem vector_addition : a + 2 • b = (4, -5) :=
by
  sorry

end vector_addition_l26_26954


namespace variance_of_transformed_variable_l26_26250

-- Define the binomial random variable
variable (Ω : Type) [ProbabilitySpace Ω]

def X : Ω → ℕ := sorry 
axiom binom_X : Binomial 10 (0.8) X

-- Define the transformed random variable
def Y (ω : Ω) := 2 * X ω + 1

-- Prove the variance of 2X + 1
theorem variance_of_transformed_variable : (Var[Y]) = 6.4 :=
by {
  -- Expected proof goes here
  sorry
}

end variance_of_transformed_variable_l26_26250


namespace emily_extra_distance_five_days_l26_26158

-- Define the distances
def distance_troy : ℕ := 75
def distance_emily : ℕ := 98

-- Emily's extra walking distance in one-way
def extra_one_way : ℕ := distance_emily - distance_troy

-- Emily's extra walking distance in a round trip
def extra_round_trip : ℕ := extra_one_way * 2

-- The extra distance Emily walks in five days
def extra_five_days : ℕ := extra_round_trip * 5

-- Theorem to be proven
theorem emily_extra_distance_five_days : extra_five_days = 230 := by
  -- Proof will go here
  sorry

end emily_extra_distance_five_days_l26_26158


namespace ball_radius_l26_26178

noncomputable def radius_of_ball (d h : ℝ) : ℝ :=
  let r := d / 2
  (325 / 20 : ℝ)

theorem ball_radius (d h : ℝ) (hd : d = 30) (hh : h = 10) :
  radius_of_ball d h = 16.25 := by
  sorry

end ball_radius_l26_26178


namespace pow_1999_mod_26_l26_26750

theorem pow_1999_mod_26 (n : ℕ) (h1 : 17^1 % 26 = 17)
  (h2 : 17^2 % 26 = 17) (h3 : 17^3 % 26 = 17) : 17^1999 % 26 = 17 := by
  sorry

end pow_1999_mod_26_l26_26750


namespace factor_expression_l26_26945

variable (y : ℝ)

theorem factor_expression : 64 - 16 * y ^ 3 = 16 * (2 - y) * (4 + 2 * y + y ^ 2) := by
  sorry

end factor_expression_l26_26945


namespace range_of_a_l26_26512

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := if x ≤ 1 then (a - 3) * x - 3 else Real.log x / Real.log a

theorem range_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≤ f a y) ↔ 3 < a ∧ a ≤ 6 :=
by
  sorry

end range_of_a_l26_26512


namespace t_range_inequality_l26_26951

theorem t_range_inequality (t : ℝ) :
  (1/8) * (2 * t - t^2) ≤ -1/4 ∧ 3 - t^2 ≥ 2 ↔ -1 ≤ t ∧ t ≤ 1 - Real.sqrt 3 :=
by
  sorry

end t_range_inequality_l26_26951


namespace solution_for_x_l26_26080

theorem solution_for_x (t : ℤ) :
  ∃ x : ℤ, (∃ (k1 k2 k3 : ℤ), 
    (2 * x + 1 = 3 * k1) ∧ (3 * x + 1 = 4 * k2) ∧ (4 * x + 1 = 5 * k3)) :=
  sorry

end solution_for_x_l26_26080


namespace max_area_house_l26_26761

def price_colored := 450
def price_composite := 200
def cost_limit := 32000

def material_cost (x y : ℝ) : ℝ := 900 * x + 400 * y + 200 * x * y

theorem max_area_house : 
  ∃ (x y S : ℝ), 
    (S = x * y) ∧ 
    (material_cost x y ≤ cost_limit) ∧ 
    (0 < S ∧ S ≤ 100) ∧ 
    (S = 100 → x = 20 / 3) := 
by
  sorry

end max_area_house_l26_26761


namespace alice_speed_is_6_5_l26_26162

-- Definitions based on the conditions.
def a : ℝ := sorry -- Alice's speed
def b : ℝ := a + 3 -- Bob's speed

-- Alice cycles towards the park 80 miles away and Bob meets her 15 miles away from the park
def d_alice : ℝ := 65 -- Alice's distance traveled (80 - 15)
def d_bob : ℝ := 95 -- Bob's distance traveled (80 + 15)

-- Equating the times
def time_eqn := d_alice / a = d_bob / b

-- Alice's speed is 6.5 mph
theorem alice_speed_is_6_5 : a = 6.5 :=
by
  have h1 : b = a + 3 := sorry
  have h2 : a * 65 = (a + 3) * 95 := sorry
  have h3 : 30 * a = 195 := sorry
  have h4 : a = 6.5 := sorry
  exact h4

end alice_speed_is_6_5_l26_26162


namespace sin_product_eq_one_sixteenth_l26_26048

theorem sin_product_eq_one_sixteenth : 
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (48 * Real.pi / 180)) * 
  (Real.sin (54 * Real.pi / 180)) * 
  (Real.sin (78 * Real.pi / 180)) = 
  1 / 16 := 
sorry

end sin_product_eq_one_sixteenth_l26_26048


namespace initial_walking_speed_l26_26917

theorem initial_walking_speed
  (t : ℝ) -- Time in minutes for bus to reach the bus stand from when the person starts walking
  (h₁ : 5 = 5 * ((t - 5) / 60)) -- When walking at 5 km/h, person reaches 5 minutes early
  (h₂ : 5 = v * ((t + 10) / 60)) -- At initial speed v, person misses the bus by 10 minutes
  : v = 4 := 
by
  sorry

end initial_walking_speed_l26_26917


namespace find_certain_number_l26_26113

theorem find_certain_number (n : ℕ) (h : 9823 + n = 13200) : n = 3377 :=
by
  sorry

end find_certain_number_l26_26113


namespace misread_number_is_correct_l26_26274

-- Definitions for the given conditions
def avg_incorrect : ℕ := 19
def incorrect_number : ℕ := 26
def avg_correct : ℕ := 24

-- Statement to prove the actual number that was misread
theorem misread_number_is_correct (x : ℕ) (h : 10 * avg_correct - 10 * avg_incorrect = x - incorrect_number) : x = 76 :=
by {
  sorry
}

end misread_number_is_correct_l26_26274


namespace alice_has_ball_after_two_turns_l26_26040

def prob_alice_keeps_ball : ℚ := (2/3 * 1/2) + (1/3 * 1/3)

theorem alice_has_ball_after_two_turns :
  prob_alice_keeps_ball = 4 / 9 :=
by
  -- This line is just a placeholder for the actual proof
  sorry

end alice_has_ball_after_two_turns_l26_26040


namespace num_sets_satisfying_union_is_four_l26_26514

variable (M : Set ℕ) (N : Set ℕ)

def num_sets_satisfying_union : Prop :=
  M = {1, 2} ∧ (M ∪ N = {1, 2, 6} → (N = {6} ∨ N = {1, 6} ∨ N = {2, 6} ∨ N = {1, 2, 6}))

theorem num_sets_satisfying_union_is_four :
  (∃ M : Set ℕ, M = {1, 2}) →
  (∃ N : Set ℕ, M ∪ N = {1, 2, 6}) →
  (∃ (num_sets : ℕ), num_sets = 4) :=
by
  sorry

end num_sets_satisfying_union_is_four_l26_26514


namespace route_speeds_l26_26123

theorem route_speeds (x : ℝ) (hx : x > 0) :
  (25 / x) - (21 / (1.4 * x)) = (20 / 60) := by
  sorry

end route_speeds_l26_26123


namespace fruits_turned_yellow_on_friday_l26_26015

theorem fruits_turned_yellow_on_friday :
  ∃ (F : ℕ), F + 2*F = 6 ∧ 14 - F - 2*F = 8 :=
by
  existsi 2
  sorry

end fruits_turned_yellow_on_friday_l26_26015


namespace no_partition_equal_product_l26_26490

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end no_partition_equal_product_l26_26490


namespace frac_m_over_q_l26_26520

variable (m n p q : ℚ)

theorem frac_m_over_q (h1 : m / n = 10) (h2 : p / n = 2) (h3 : p / q = 1 / 5) : m / q = 1 :=
by
  sorry

end frac_m_over_q_l26_26520


namespace evaluate_expression_l26_26494

theorem evaluate_expression : 
  (5^1001 + 6^1002)^2 - (5^1001 - 6^1002)^2 = 24 * 30^1001 :=
by
  sorry

end evaluate_expression_l26_26494


namespace portion_of_money_given_to_Blake_l26_26045

theorem portion_of_money_given_to_Blake
  (initial_amount : ℝ)
  (tripled_amount : ℝ)
  (sale_amount : ℝ)
  (amount_given_to_Blake : ℝ)
  (h1 : initial_amount = 20000)
  (h2 : tripled_amount = 3 * initial_amount)
  (h3 : sale_amount = tripled_amount)
  (h4 : amount_given_to_Blake = 30000) :
  amount_given_to_Blake / sale_amount = 1 / 2 :=
sorry

end portion_of_money_given_to_Blake_l26_26045


namespace compound_interest_is_correct_l26_26419

noncomputable def compoundInterest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * (1 + R)^T - P

theorem compound_interest_is_correct
  (P : ℝ)
  (R : ℝ)
  (T : ℝ)
  (SI : ℝ) : SI = P * R * T / 100 ∧ R = 0.10 ∧ T = 2 ∧ SI = 600 → compoundInterest P R T = 630 :=
by
  sorry

end compound_interest_is_correct_l26_26419


namespace first_prime_year_with_digit_sum_8_l26_26530

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |> List.sum

def is_prime (n : ℕ) : Prop :=
  ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem first_prime_year_with_digit_sum_8 :
  ∃ y : ℕ, y > 2015 ∧ sum_of_digits y = 8 ∧ is_prime y ∧
  ∀ z : ℕ, z > 2015 ∧ sum_of_digits z = 8 ∧ is_prime z → y ≤ z :=
sorry

end first_prime_year_with_digit_sum_8_l26_26530


namespace airplane_seat_count_l26_26481

theorem airplane_seat_count (s : ℝ) 
  (h1 : 30 + 0.2 * s + 0.75 * s = s) : 
  s = 600 :=
sorry

end airplane_seat_count_l26_26481


namespace Jose_age_proof_l26_26548

-- Definitions based on the conditions
def Inez_age : ℕ := 15
def Zack_age : ℕ := Inez_age + 5
def Jose_age : ℕ := Zack_age - 7

theorem Jose_age_proof : Jose_age = 13 :=
by
  -- Proof omitted
  sorry

end Jose_age_proof_l26_26548


namespace snow_leopards_arrangement_l26_26998

theorem snow_leopards_arrangement : 
  ∃ (perm : Fin 9 → Fin 9), 
    (∀ i, perm i ≠ perm j → i ≠ j) ∧ 
    (perm 0 < perm 1 ∧ perm 8 < perm 1 ∧ perm 0 < perm 8) ∧ 
    (∃ count_ways, count_ways = 4320) :=
sorry

end snow_leopards_arrangement_l26_26998


namespace intersection_x_coordinate_l26_26731

theorem intersection_x_coordinate (k b : ℝ) (h : k ≠ b) :
  (∃ x y : ℝ, y = k * x + b ∧ y = b * x + k) → (∃ x : ℝ, x = 1) :=
by
  intro h_intersect
  cases h_intersect with x h_xy
  cases h_xy with y h_1
  use 1
  sorry

end intersection_x_coordinate_l26_26731


namespace triangle_ABC_properties_l26_26412

theorem triangle_ABC_properties 
  (a b c : ℝ) 
  (A B C : ℝ) 
  (h1 : 2 * Real.sin B * Real.sin C * Real.cos A + Real.cos A = 3 * Real.sin A ^ 2 - Real.cos (B - C)) : 
  (2 * a = b + c) ∧ 
  (b + c = 2) →
  (Real.cos A = 3/5) → 
  (1 / 2 * b * c * Real.sin A = 3 / 8) :=
by
  sorry

end triangle_ABC_properties_l26_26412


namespace polynomial_square_solution_l26_26735

variable (a b : ℝ)

theorem polynomial_square_solution (h : 
  ∃ g : Polynomial ℝ, g^2 = Polynomial.C (1 : ℝ) * Polynomial.X^4 -
  Polynomial.C (1 : ℝ) * Polynomial.X^3 +
  Polynomial.C (1 : ℝ) * Polynomial.X^2 +
  Polynomial.C a * Polynomial.X +
  Polynomial.C b) : b = 9 / 64 :=
by sorry

end polynomial_square_solution_l26_26735


namespace convert_base_7_to_base_10_l26_26929

theorem convert_base_7_to_base_10 : 
  ∀ n : ℕ, (n = 3 * 7^2 + 2 * 7^1 + 1 * 7^0) → n = 162 :=
by
  intros n h
  rw [pow_zero, pow_one, pow_two] at h
  norm_num at h
  exact h

end convert_base_7_to_base_10_l26_26929


namespace exists_six_digit_number_l26_26058

theorem exists_six_digit_number : ∃ (n : ℕ), 100000 ≤ n ∧ n < 1000000 ∧ (∃ (x y : ℕ), n = 1000 * x + y ∧ 0 ≤ x ∧ x < 1000 ∧ 0 ≤ y ∧ y < 1000 ∧ 6 * n = 1000 * y + x) :=
by
  sorry

end exists_six_digit_number_l26_26058


namespace problem_solution_l26_26977

theorem problem_solution
  (p q r u v w : ℝ)
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 :=
sorry

end problem_solution_l26_26977


namespace number_of_pupils_l26_26906

theorem number_of_pupils (n : ℕ) (M : ℕ)
  (avg_all : 39 * n = M)
  (pupil_marks : 25 + 12 + 15 + 19 = 71)
  (new_avg : (M - 71) / (n - 4) = 44) :
  n = 21 := sorry

end number_of_pupils_l26_26906


namespace smallest_n_condition_l26_26219

open Nat

-- Define the sum of squares formula
noncomputable def sum_of_squares (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

-- Define the condition for being a square number
def is_square (m : ℕ) : Prop :=
  ∃ k : ℕ, k * k = m

-- The proof problem statement
theorem smallest_n_condition : 
  ∃ n : ℕ, n > 1 ∧ is_square (sum_of_squares n / n) ∧ (∀ m : ℕ, m > 1 ∧ is_square (sum_of_squares m / m) → n ≤ m) :=
sorry

end smallest_n_condition_l26_26219


namespace truck_total_distance_l26_26643

noncomputable def truck_distance (b t : ℝ) : ℝ :=
  let acceleration := b / 3
  let time_seconds := 300 + t
  let distance_feet := (1 / 2) * (acceleration / t) * time_seconds^2
  distance_feet / 5280

theorem truck_total_distance (b t : ℝ) : 
  truck_distance b t = b * (90000 + 600 * t + t ^ 2) / (31680 * t) :=
by
  sorry

end truck_total_distance_l26_26643


namespace students_prob_red_light_l26_26038

noncomputable def probability_red_light_encountered (p1 p2 p3 : ℚ) : ℚ :=
  1 - ((1 - p1) * (1 - p2) * (1 - p3))

theorem students_prob_red_light :
  probability_red_light_encountered (1/2) (1/3) (1/4) = 3/4 :=
by
  sorry

end students_prob_red_light_l26_26038


namespace part1_part2_l26_26835

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l26_26835


namespace planting_rate_l26_26065

theorem planting_rate (total_acres : ℕ) (days : ℕ) (initial_tractors : ℕ) (initial_days : ℕ) (additional_tractors : ℕ) (additional_days : ℕ) :
  total_acres = 1700 →
  days = 5 →
  initial_tractors = 2 →
  initial_days = 2 →
  additional_tractors = 7 →
  additional_days = 3 →
  (total_acres / ((initial_tractors * initial_days) + (additional_tractors * additional_days))) = 68 :=
by
  sorry

end planting_rate_l26_26065


namespace trigonometric_identity_l26_26375

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l26_26375


namespace fertilizer_percentage_l26_26718

theorem fertilizer_percentage (total_volume : ℝ) (vol_74 : ℝ) (vol_53 : ℝ) (perc_74 : ℝ) (perc_53 : ℝ) (final_perc : ℝ) :
  total_volume = 42 ∧ vol_74 = 20 ∧ vol_53 = total_volume - vol_74 ∧ perc_74 = 0.74 ∧ perc_53 = 0.53 
  → final_perc = ((vol_74 * perc_74 + vol_53 * perc_53) / total_volume) * 100
  → final_perc = 63.0 :=
by
  intros
  sorry

end fertilizer_percentage_l26_26718


namespace arithmetic_sequence_100_l26_26230

variable {a : ℕ → ℝ}

def is_arithmetic_sequence (a : ℕ → ℝ) :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

variables (S₉ : ℝ) (a₁₀ : ℝ)

theorem arithmetic_sequence_100
  (h1: is_arithmetic_sequence a)
  (h2: S₉ = 27) 
  (h3: a₁₀ = 8): 
  a 100 = 98 := 
sorry

end arithmetic_sequence_100_l26_26230


namespace lila_stickers_correct_l26_26422

-- Defining the constants for number of stickers each has
def Kristoff_stickers : ℕ := 85
def Riku_stickers : ℕ := 25 * Kristoff_stickers
def Lila_stickers : ℕ := 2 * (Kristoff_stickers + Riku_stickers)

-- The theorem to prove
theorem lila_stickers_correct : Lila_stickers = 4420 := 
by {
  sorry
}

end lila_stickers_correct_l26_26422


namespace total_amount_earned_l26_26769

theorem total_amount_earned (avg_price_per_pair : ℝ) (number_of_pairs : ℕ) (price : avg_price_per_pair = 9.8 ) (pairs : number_of_pairs = 50 ) : 
avg_price_per_pair * number_of_pairs = 490 := by
  -- Given conditions
  sorry

end total_amount_earned_l26_26769


namespace part1_part2_l26_26831

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l26_26831


namespace value_range_of_f_l26_26742

noncomputable def f (x : ℝ) : ℝ := 2 + Real.logb 5 (x + 3)

theorem value_range_of_f :
  ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x ∈ Set.Icc (2 : ℝ) 3 := 
by
  sorry

end value_range_of_f_l26_26742


namespace hyperbola_asymptotes_l26_26231

def hyperbola (x y : ℝ) : Prop := (x^2 / 8) - (y^2 / 2) = 1

theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola x y → (y = (1/2) * x ∨ y = - (1/2) * x) :=
by
  sorry

end hyperbola_asymptotes_l26_26231


namespace perfect_squares_in_interval_l26_26403

theorem perfect_squares_in_interval (s : Set Int) (h1 : ∃ a : Nat, ∀ x ∈ s, a^4 ≤ x ∧ x ≤ (a+9)^4)
                                     (h2 : ∃ b : Nat, ∀ x ∈ s, b^3 ≤ x ∧ x ≤ (b+99)^3) :
  ∃ c : Nat, c ≥ 2000 ∧ ∀ x ∈ s, x = c^2 :=
sorry

end perfect_squares_in_interval_l26_26403


namespace steve_commute_l26_26729

theorem steve_commute :
  ∃ (D : ℝ), 
    (∃ (V : ℝ), 2 * V = 5 ∧ (D / V + D / (2 * V) = 6)) ∧ D = 10 :=
by
  sorry

end steve_commute_l26_26729


namespace range_of_a_l26_26228

variable (f : ℝ → ℝ)
variable (a : ℝ)

-- Conditions
def decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop := 
  ∀ ⦃x y⦄, x ∈ s → y ∈ s → x < y → f x > f y

def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def condition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f (1 - a) + f (1 - 2 * a) < 0

-- Theorem statement
theorem range_of_a (h_decreasing : decreasing_on f (Set.Ioo (-1) 1))
                   (h_odd : odd_function f)
                   (h_condition : condition f a) :
  0 < a ∧ a < 2 / 3 :=
sorry

end range_of_a_l26_26228


namespace smallest_number_divisible_by_6_with_perfect_square_product_l26_26323

-- Definition of a two-digit number
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

-- Definition of divisibility by 6
def is_divisible_by_6 (n : ℕ) : Prop :=
  n % 6 = 0

-- Definition of the product of digits being a perfect square
def digits_product (n : ℕ) : ℕ :=
  let d1 := n / 10
  let d2 := n % 10
  d1 * d2

def is_perfect_square (m : ℕ) : Prop :=
  ∃ (k : ℕ), k * k = m

-- The problem statement as a Lean theorem
theorem smallest_number_divisible_by_6_with_perfect_square_product : ∃ (n : ℕ), is_two_digit n ∧ is_divisible_by_6 n ∧ is_perfect_square (digits_product n) ∧ ∀ (m : ℕ), (is_two_digit m ∧ is_divisible_by_6 m ∧ is_perfect_square (digits_product m)) → n ≤ m :=
by
  sorry

end smallest_number_divisible_by_6_with_perfect_square_product_l26_26323


namespace sin_beta_value_l26_26691

theorem sin_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π/2) (hβ : 0 < β ∧ β < π/2)
  (h1 : Real.cos α = 4 / 5) (h2 : Real.cos (α + β) = 5 / 13) :
  Real.sin β = 33 / 65 :=
sorry

end sin_beta_value_l26_26691


namespace garden_area_increase_l26_26177

theorem garden_area_increase :
    let length := 60
    let width := 20
    let perimeter := 2 * (length + width)
    let side_of_square := perimeter / 4
    let area_rectangular := length * width
    let area_square := side_of_square * side_of_square
    area_square - area_rectangular = 400 :=
by
  sorry

end garden_area_increase_l26_26177


namespace base_6_addition_l26_26008

-- Definitions of base conversion and addition
def base_6_to_nat (n : ℕ) : ℕ :=
  n.div 100 * 36 + n.div 10 % 10 * 6 + n % 10

def nat_to_base_6 (n : ℕ) : ℕ :=
  let a := n.div 216
  let b := (n % 216).div 36
  let c := ((n % 216) % 36).div 6
  let d := n % 6
  a * 1000 + b * 100 + c * 10 + d

-- Conversion from base 6 to base 10 for the given numbers
def nat_256 := base_6_to_nat 256
def nat_130 := base_6_to_nat 130

-- The final theorem to prove
theorem base_6_addition : nat_to_base_6 (nat_256 + nat_130) = 1042 :=
by
  -- Proof omitted since it is not required
  sorry

end base_6_addition_l26_26008


namespace Loisa_saves_70_l26_26714

-- Define the conditions
def tablet_cost_cash := 450
def down_payment := 100
def payment_first_4_months := 40 * 4
def payment_next_4_months := 35 * 4
def payment_last_4_months := 30 * 4

-- Define the total installment payment
def total_installment_payment := down_payment + payment_first_4_months + payment_next_4_months + payment_last_4_months

-- Define the amount saved by paying cash instead of on installment
def amount_saved := total_installment_payment - tablet_cost_cash

-- The theorem to prove the savings amount
theorem Loisa_saves_70 : amount_saved = 70 := by
  -- Direct calculation or further proof steps here
  sorry

end Loisa_saves_70_l26_26714


namespace sin_2pi_minus_alpha_l26_26395

noncomputable def alpha_condition (α : ℝ) : Prop :=
  (3 * Real.pi / 2 < α) ∧ (α < 2 * Real.pi) ∧ (Real.cos (Real.pi + α) = -1 / 2)

theorem sin_2pi_minus_alpha (α : ℝ) (h : alpha_condition α) : Real.sin (2 * Real.pi - α) = Real.sqrt 3 / 2 :=
by
  sorry

end sin_2pi_minus_alpha_l26_26395


namespace max_value_sqrt_abc_expression_l26_26855

theorem max_value_sqrt_abc_expression (a b c : ℝ) (ha : 0 ≤ a) (ha1 : a ≤ 1)
                                       (hb : 0 ≤ b) (hb1 : b ≤ 1)
                                       (hc : 0 ≤ c) (hc1 : c ≤ 1) :
    (Real.sqrt (a * b * c) + Real.sqrt ((1 - a) * (1 - b) * (1 - c)) ≤ 1) :=
sorry

end max_value_sqrt_abc_expression_l26_26855


namespace purple_valley_skirts_l26_26723

theorem purple_valley_skirts (azure_valley_skirts : ℕ) (h1 : azure_valley_skirts = 60) :
    let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts in
    let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts in
    purple_valley_skirts = 10 :=
by
  let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts
  let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts
  have h2 : seafoam_valley_skirts = (2 / 3 : ℚ) * 60 := by
    rw [h1]
  have h3 : purple_valley_skirts = (1 / 4 : ℚ) * ((2 / 3 : ℚ) * 60) := by
    rw [h2]
  have h4 : purple_valley_skirts = (1 / 4 : ℚ) * 40 := by
    norm_num [h3]
  have h5 : purple_valley_skirts = 10 := by
    norm_num [h4]
  exact h5

end purple_valley_skirts_l26_26723


namespace total_time_correct_l26_26130

-- Definitions for the conditions
def dean_time : ℕ := 9
def micah_time : ℕ := (2 * dean_time) / 3
def jake_time : ℕ := micah_time + micah_time / 3

-- Proof statement for the total time
theorem total_time_correct : micah_time + dean_time + jake_time = 23 := by
  sorry

end total_time_correct_l26_26130


namespace Zlatoust_to_Miass_distance_l26_26294

theorem Zlatoust_to_Miass_distance
  (x g k m : ℝ)
  (H1 : (x + 18) / k = (x - 18) / m)
  (H2 : (x + 25) / k = (x - 25) / g)
  (H3 : (x + 8) / m = (x - 8) / g) :
  x = 60 :=
sorry

end Zlatoust_to_Miass_distance_l26_26294


namespace probability_even_sum_l26_26747

theorem probability_even_sum :
  let total_outcomes := 12 * 11,
      favorable_even_even := 6 * 5,
      favorable_odd_odd := 6 * 5,
      favorable_outcomes := favorable_even_even + favorable_odd_odd,
      probability := favorable_outcomes / total_outcomes in
  probability = (5 : ℚ) / 11 :=
by
  sorry

end probability_even_sum_l26_26747


namespace scientific_notation_l26_26135

theorem scientific_notation : (10374 * 10^9 : Real) = 1.037 * 10^13 :=
by
  sorry

end scientific_notation_l26_26135


namespace units_digit_of_n_l26_26679

theorem units_digit_of_n
  (m n : ℕ)
  (h1 : m * n = 23^7)
  (h2 : m % 10 = 9) : n % 10 = 3 :=
sorry

end units_digit_of_n_l26_26679


namespace anna_should_plant_8_lettuce_plants_l26_26193

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end anna_should_plant_8_lettuce_plants_l26_26193


namespace sin_value_l26_26506

theorem sin_value (α : ℝ) (h : Real.cos (α + π / 6) = - (Real.sqrt 2) / 10) : 
  Real.sin (2 * α - π / 6) = 24 / 25 :=
by
  sorry

end sin_value_l26_26506


namespace Harold_spending_l26_26968

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

end Harold_spending_l26_26968


namespace tetrahedron_edge_pairs_l26_26698

-- Problem statement
theorem tetrahedron_edge_pairs (tetrahedron_edges : ℕ) (tetrahedron_edges_eq_six : tetrahedron_edges = 6) :
  ∃ n, n = (Nat.choose tetrahedron_edges 2) ∧ n = 15 :=
by
  use Nat.choose 6 2
  constructor
  . rfl
  . exact Nat.choose_succ_self_right 5

end tetrahedron_edge_pairs_l26_26698


namespace squares_in_rectangle_l26_26695

theorem squares_in_rectangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a ≤ 1) (h5 : b ≤ 1) (h6 : c ≤ 1) (h7 : a + b + c = 2)  : 
  a + b + c ≤ 2 := sorry

end squares_in_rectangle_l26_26695


namespace geometric_progression_condition_l26_26356

noncomputable def condition_for_geometric_progression (a q : ℝ) (n p : ℤ) : Prop :=
  ∃ m : ℤ, a = q^m

theorem geometric_progression_condition (a q : ℝ) (n p k : ℤ) :
  condition_for_geometric_progression a q n p ↔ a * q^(n + p) = a * q^k :=
by
  sorry

end geometric_progression_condition_l26_26356


namespace percentage_increase_twice_l26_26151

theorem percentage_increase_twice (P : ℝ) (x : ℝ) :
  P * (1 + x)^2 = P * 1.3225 → x = 0.15 :=
by
  intro h
  have h1 : (1 + x)^2 = 1.3225 := by sorry
  have h2 : x^2 + 2 * x = 0.3225 := by sorry
  have h3 : x = (-2 + Real.sqrt 5.29) / 2 := by sorry
  have h4 : x = -2 / 2 + Real.sqrt 5.29 / 2 := by sorry
  have h5 : x = 0.15 := by sorry
  exact h5

end percentage_increase_twice_l26_26151


namespace inequality_example_l26_26575

theorem inequality_example (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a^2 + b^2 + c^2 = 3) : 
    1 / (1 + a * b) + 1 / (1 + b * c) + 1 / (1 + a * c) ≥ 3 / 2 :=
by
  sorry

end inequality_example_l26_26575


namespace part1_part2_l26_26823

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l26_26823


namespace cylinder_surface_area_l26_26380

theorem cylinder_surface_area (r : ℝ) (l : ℝ) (h1 : r = 2) (h2 : l = 2 * r) : 
  2 * Real.pi * r^2 + 2 * Real.pi * r * l = 24 * Real.pi :=
by
  subst h1
  subst h2
  sorry

end cylinder_surface_area_l26_26380


namespace square_difference_l26_26686

theorem square_difference (x : ℤ) (h : x^2 = 9801) : (x - 2) * (x + 2) = 9797 :=
by 
  have diff_squares : (x - 2) * (x + 2) = x^2 - 4 := by ring
  rw [diff_squares, h]
  norm_num

end square_difference_l26_26686


namespace derivative_at_2_l26_26232

def f (x : ℝ) : ℝ := x^3 + 2

theorem derivative_at_2 : deriv f 2 = 12 := by
  sorry

end derivative_at_2_l26_26232


namespace intersection_complement_eq_find_a_l26_26386

-- Proof Goal 1: A ∩ ¬B = {x : ℝ | x ∈ (-∞, -3] ∪ [14, ∞)}

def setA : Set ℝ := {x | (x + 3) * (x - 6) ≥ 0}
def setB : Set ℝ := {x | (x + 2) / (x - 14) < 0}
def negB : Set ℝ := {x | x ≤ -2 ∨ x ≥ 14}

theorem intersection_complement_eq :
  setA ∩ negB = {x : ℝ | x ≤ -3 ∨ x ≥ 14} :=
by
  sorry

-- Proof Goal 2: The range of a such that E ⊆ B

def E (a : ℝ) : Set ℝ := {x | 2 * a < x ∧ x < a + 1}

theorem find_a (a : ℝ) :
  (∀ x, E a x → setB x) → a ≥ -1 :=
by
  sorry

end intersection_complement_eq_find_a_l26_26386


namespace probability_at_least_one_prize_proof_l26_26440

noncomputable def probability_at_least_one_wins_prize
  (total_tickets : ℕ) (prize_tickets : ℕ) (people : ℕ) :
  ℚ :=
1 - ((@Nat.choose (total_tickets - prize_tickets) people) /
      (@Nat.choose total_tickets people))

theorem probability_at_least_one_prize_proof :
  probability_at_least_one_wins_prize 10 3 5 = 11 / 12 :=
by
  sorry

end probability_at_least_one_prize_proof_l26_26440


namespace min_k_value_l26_26079

noncomputable def minimum_k_condition (x y z k : ℝ) : Prop :=
  k * (x^2 - x + 1) * (y^2 - y + 1) * (z^2 - z + 1) ≥ (x * y * z)^2 - (x * y * z) + 1

theorem min_k_value :
  ∀ x y z : ℝ, x ≤ 0 → y ≤ 0 → z ≤ 0 → minimum_k_condition x y z (16 / 9) :=
by
  sorry

end min_k_value_l26_26079


namespace fraction_arithmetic_l26_26607

theorem fraction_arithmetic : ((3 / 5 : ℚ) + (4 / 15)) * (2 / 3) = 26 / 45 := 
by
  sorry

end fraction_arithmetic_l26_26607


namespace int_div_condition_l26_26692

theorem int_div_condition (n : ℕ) (hn₁ : ∃ m : ℤ, 2^n - 2 = m * n) :
  ∃ k : ℤ, 2^(2^n - 1) - 2 = k * (2^n - 1) :=
by sorry

end int_div_condition_l26_26692


namespace larger_of_two_numbers_l26_26463

noncomputable def larger_number (HCF LCM A B : ℕ) : ℕ :=
  if HCF = 23 ∧ LCM = 23 * 9 * 10 ∧ A * B = HCF * LCM ∧ (A = 10 ∧ B = 23 * 9 ∨ B = 10 ∧ A = 23 * 9)
  then max A B
  else 0

theorem larger_of_two_numbers : larger_number (23) (23 * 9 * 10) 230 207 = 230 := by
  sorry

end larger_of_two_numbers_l26_26463


namespace collinear_vectors_l26_26504

open Vector

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def not_collinear (a b : V) : Prop :=
¬(∃ k : ℝ, k ≠ 0 ∧ a = k • b)

theorem collinear_vectors
  {a b m n : V}
  (h1 : m = a + b)
  (h2 : n = 2 • a + 2 • b)
  (h3 : not_collinear a b) :
  ∃ k : ℝ, k ≠ 0 ∧ n = k • m :=
by
  sorry

end collinear_vectors_l26_26504


namespace monotonic_increasing_f_C_l26_26610

noncomputable def f_A (x : ℝ) : ℝ := -Real.log x
noncomputable def f_B (x : ℝ) : ℝ := 1 / (2^x)
noncomputable def f_C (x : ℝ) : ℝ := -(1 / x)
noncomputable def f_D (x : ℝ) : ℝ := 3^(abs (x - 1))

theorem monotonic_increasing_f_C : ∀ x y : ℝ, 0 < x → 0 < y → x < y → f_C x < f_C y :=
sorry

end monotonic_increasing_f_C_l26_26610


namespace zadam_win_probability_l26_26028

theorem zadam_win_probability :
  ∃ p : ℚ, p = 1 / 2 ∧
  (∃ (num_attempts success_attempts : ℕ), num_attempts ≥ 5 ∧ num_attempts ≤ 9 ∧ success_attempts = 5 ∧
    (∀ k : ℕ, k < num_attempts → 
      ∃ successful_rolls : finset (fin num_attempts), successful_rolls.card = success_attempts ∧
        (∀ i : fin num_attempts, i ∈ successful_rolls → 
          (1/2 : ℚ)) ∧
        (probability (λ outcome : finset (fin num_attempts), ∃ win_set, 
          win_set = successful_rolls ∧ win_set.card = success_attempts ∧ win_set.sum (λ x, (1/2 : ℚ)) = p
        ) = 1 / 2)
     )
  )

end zadam_win_probability_l26_26028


namespace greatest_award_correct_l26_26613

-- Definitions and constants
def total_prize : ℕ := 600
def num_winners : ℕ := 15
def min_award : ℕ := 15
def prize_fraction_num : ℕ := 2
def prize_fraction_den : ℕ := 5
def winners_fraction_num : ℕ := 3
def winners_fraction_den : ℕ := 5

-- Conditions (translated and simplified)
def num_specific_winners : ℕ := (winners_fraction_num * num_winners) / winners_fraction_den
def specific_prize : ℕ := (prize_fraction_num * total_prize) / prize_fraction_den
def remaining_winners : ℕ := num_winners - num_specific_winners
def min_total_award_remaining : ℕ := remaining_winners * min_award
def remaining_prize : ℕ := total_prize - min_total_award_remaining
def min_award_specific : ℕ := num_specific_winners - 1
def sum_min_awards_specific : ℕ := min_award_specific * min_award

-- Correct answer
def greatest_award : ℕ := remaining_prize - sum_min_awards_specific

-- Theorem statement (Proof skipped with sorry)
theorem greatest_award_correct :
  greatest_award = 390 := sorry

end greatest_award_correct_l26_26613


namespace anna_should_plant_8_lettuce_plants_l26_26194

/-- Anna wants to grow some lettuce in the garden and would like to grow enough to have at least
    12 large salads.
- Conditions:
  1. Half of the lettuce will be lost to insects and rabbits.
  2. Each lettuce plant is estimated to provide 3 large salads.
  
  Proof that Anna should plant 8 lettuce plants in the garden. --/
theorem anna_should_plant_8_lettuce_plants 
    (desired_salads: ℕ)
    (salads_per_plant: ℕ)
    (loss_fraction: ℚ) :
    desired_salads = 12 →
    salads_per_plant = 3 →
    loss_fraction = 1 / 2 →
    ∃ plants: ℕ, plants = 8 :=
by
  intros h1 h2 h3
  sorry

end anna_should_plant_8_lettuce_plants_l26_26194


namespace solution_pairs_correct_l26_26209

theorem solution_pairs_correct:
  { (n, m) : ℕ × ℕ | m^2 + 2 * 3^n = m * (2^(n+1) - 1) }
  = {(3, 6), (3, 9), (6, 54), (6, 27)} :=
by
  sorry -- no proof is required as per the instruction

end solution_pairs_correct_l26_26209


namespace quadratic_function_integer_values_not_imply_integer_coefficients_l26_26550

theorem quadratic_function_integer_values_not_imply_integer_coefficients :
  ∃ (a b c : ℚ), (∀ x : ℤ, ∃ y : ℤ, (a * (x : ℚ)^2 + b * (x : ℚ) + c = (y : ℚ))) ∧
    (¬ (∃ (a_int b_int c_int : ℤ), a = (a_int : ℚ) ∧ b = (b_int : ℚ) ∧ c = (c_int : ℚ))) :=
by
  sorry

end quadratic_function_integer_values_not_imply_integer_coefficients_l26_26550


namespace num_valid_N_l26_26365

theorem num_valid_N : 
  ∃ n : ℕ, n = 4 ∧ ∀ (N : ℕ), (N > 0) → (∃ k : ℕ, 60 = (N+3) * k ∧ k % 2 = 0) ↔ (N = 1 ∨ N = 9 ∨ N = 17 ∨ N = 57) :=
sorry

end num_valid_N_l26_26365


namespace part1_part2_l26_26813

variable {a b c : ℝ}

-- Condition: a, b, c > 0
axiom pos_a : a > 0
axiom pos_b : b > 0
axiom pos_c : c > 0

-- Condition: a^2 + b^2 + 4c^2 = 3
axiom condition : a^2 + b^2 + 4c^2 = 3

-- First proof statement: a + b + 2c ≤ 3
theorem part1 : a + b + 2 * c ≤ 3 := 
  sorry

-- Second proof statement: if b = 2c, then 1/a + 1/c ≥ 3
theorem part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 :=
  sorry

end part1_part2_l26_26813


namespace inequality_l26_26859

def domain (x : ℝ) : Prop := -2 < x ∧ x < 3

theorem inequality (a b : ℝ) (ha : domain a) (hb : domain b) :
  |a + b| < |3 + ab / 3| :=
by
  sorry

end inequality_l26_26859


namespace line_circle_intersection_l26_26150

theorem line_circle_intersection (a : ℝ) : 
  (∀ x y : ℝ, (4 * x + 3 * y + a = 0) → ((x - 1)^2 + (y - 2)^2 = 9)) ∧
  (∃ A B : ℝ, dist A B = 4 * Real.sqrt 2) →
  (a = -5 ∨ a = -15) :=
by 
  sorry

end line_circle_intersection_l26_26150


namespace value_of_b_over_a_l26_26152

def rectangle_ratio (a b : ℝ) : Prop :=
  let d := Real.sqrt (a^2 + b^2)
  let P := 2 * (a + b)
  (b / d) = (d / (a + b))

theorem value_of_b_over_a (a b : ℝ) (h : rectangle_ratio a b) : b / a = 1 :=
by sorry

end value_of_b_over_a_l26_26152


namespace estimation_correct_l26_26117

-- Definitions corresponding to conditions.
def total_population : ℕ := 10000
def surveyed_population : ℕ := 200
def aware_surveyed : ℕ := 125

-- The proportion step: 125/200 = x/10000
def proportion (aware surveyed total_pop : ℕ) : ℕ :=
  (aware * total_pop) / surveyed

-- Using this to define our main proof goal
def estimated_aware := proportion aware_surveyed surveyed_population total_population

-- Final proof statement
theorem estimation_correct :
  estimated_aware = 6250 :=
sorry

end estimation_correct_l26_26117


namespace find_fraction_l26_26849

-- Define the initial amount, the amount spent on pads, and the remaining amount
def initial_amount := 150
def spent_on_pads := 50
def remaining := 25

-- Define the fraction she spent on hockey skates
def fraction_spent_on_skates (f : ℚ) : Prop :=
  let spent_on_skates := initial_amount - remaining - spent_on_pads
  (spent_on_skates / initial_amount) = f

theorem find_fraction : fraction_spent_on_skates (1 / 2) :=
by
  -- Proof steps go here
  sorry

end find_fraction_l26_26849


namespace spring_excursion_participants_l26_26289

theorem spring_excursion_participants (water fruit neither both total : ℕ) 
  (h_water : water = 80) 
  (h_fruit : fruit = 70) 
  (h_neither : neither = 6) 
  (h_both : both = total / 2) 
  (h_total_eq : total = water + fruit - both + neither) : 
  total = 104 := 
  sorry

end spring_excursion_participants_l26_26289


namespace complement_inter_proof_l26_26099

open Set

variable (U : Set ℕ) (A B : Set ℕ)

def complement_inter (U A B : Set ℕ) : Set ℕ :=
  compl (A ∩ B)

theorem complement_inter_proof (hU : U = {1, 2, 3, 4, 5, 6, 7, 8} )
  (hA : A = {1, 2, 3}) (hB : B = {2, 3, 4, 5}) :
  complement_inter U A B = {1, 4, 5, 6, 7, 8} :=
by
  sorry

end complement_inter_proof_l26_26099


namespace first_grade_enrollment_l26_26541

theorem first_grade_enrollment (a : ℕ) (R : ℕ) (L : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
  (h3 : a = 25 * R + 10) (h4 : a = 30 * L - 15) : a = 285 :=
by
  sorry

end first_grade_enrollment_l26_26541


namespace tangent_line_equation_l26_26581

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 1)

theorem tangent_line_equation :
  let x1 : ℝ := 1
  let y1 : ℝ := f 1
  ∀ x y : ℝ, 
    (y - y1 = (1 / (x1 + 1)) * (x - x1)) ↔ 
    (x - 2 * y + 2 * Real.log 2 - 1 = 0) :=
by
  sorry

end tangent_line_equation_l26_26581


namespace find_fathers_age_l26_26786

noncomputable def sebastian_age : ℕ := 40
noncomputable def age_difference : ℕ := 10
noncomputable def sum_ages_five_years_ago_ratio : ℚ := (3 : ℚ) / 4

theorem find_fathers_age 
  (sebastian_age : ℕ) 
  (age_difference : ℕ) 
  (sum_ages_five_years_ago_ratio : ℚ) 
  (h1 : sebastian_age = 40) 
  (h2 : age_difference = 10) 
  (h3 : sum_ages_five_years_ago_ratio = 3 / 4) 
: ∃ father_age : ℕ, father_age = 85 :=
sorry

end find_fathers_age_l26_26786


namespace inequality_part1_inequality_part2_l26_26815

variable (a b c : ℝ)

-- Declaring the positivity conditions of a, b, and c
axiom pos_a : 0 < a
axiom pos_b : 0 < b
axiom pos_c : 0 < c

-- Declaring the equation condition
axiom eq_sum : a^2 + b^2 + 4 * c^2 = 3

-- Propositions to prove
theorem inequality_part1 : a + b + 2 * c ≤ 3 := sorry

theorem inequality_part2 (h : b = 2 * c) : 1/a + 1/c ≥ 3 := sorry

end inequality_part1_inequality_part2_l26_26815


namespace father_current_age_is_85_l26_26789

theorem father_current_age_is_85 (sebastian_age : ℕ) (sister_diff : ℕ) (age_sum_fraction : ℕ → ℕ → ℕ → Prop) :
  sebastian_age = 40 →
  sister_diff = 10 →
  (∀ (s s' f : ℕ), age_sum_fraction s s' f → f = 4 * (s + s') / 3) →
  age_sum_fraction (sebastian_age - 5) (sebastian_age - sister_diff - 5) (40 + 5) →
  ∃ father_age : ℕ, father_age = 85 :=
by
  intros
  sorry

end father_current_age_is_85_l26_26789


namespace number_of_shelves_l26_26479

/-- Adam could fit 11 action figures on each shelf -/
def action_figures_per_shelf : ℕ := 11

/-- Adam's shelves could hold a total of 44 action figures -/
def total_action_figures_on_shelves : ℕ := 44

/-- Prove the number of shelves in Adam's room -/
theorem number_of_shelves:
  total_action_figures_on_shelves / action_figures_per_shelf = 4 := 
by {
    sorry
}

end number_of_shelves_l26_26479


namespace alex_age_div_M_l26_26334

variable {A M : ℕ}

-- Definitions provided by the conditions
def alex_age_current : ℕ := A
def sum_children_age : ℕ := A
def alex_age_M_years_ago (A M : ℕ) : ℕ := A - M
def children_age_M_years_ago (A M : ℕ) : ℕ := A - 4 * M

-- Given condition as a hypothesis
def condition (A M : ℕ) := alex_age_M_years_ago A M = 3 * children_age_M_years_ago A M

-- The theorem to prove
theorem alex_age_div_M (A M : ℕ) (h : condition A M) : A / M = 11 / 2 := 
by
  -- This is a placeholder for the actual proof.
  sorry

end alex_age_div_M_l26_26334


namespace max_subset_size_l26_26074

theorem max_subset_size :
  ∃ S : Finset ℕ, (∀ (x y : ℕ), x ∈ S → y ∈ S → y ≠ 2 * x) →
  S.card = 1335 :=
sorry

end max_subset_size_l26_26074


namespace inequality_proof_l26_26267

theorem inequality_proof (a b c : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) :
  a^3 * b + b^3 * c + c^3 * a ≥ a^2 * b * c + b^2 * c * a + c^2 * a * b :=
by {
  sorry
}

end inequality_proof_l26_26267


namespace f_eq_f_inv_iff_x_eq_3_5_l26_26056

def f (x : ℝ) : ℝ := 3 * x - 7
def f_inv (x : ℝ) : ℝ := (x + 7) / 3

theorem f_eq_f_inv_iff_x_eq_3_5 (x : ℝ) : f(x) = f_inv(x) ↔ x = 3.5 := by
  sorry

end f_eq_f_inv_iff_x_eq_3_5_l26_26056


namespace perimeter_is_correct_l26_26429

def side_length : ℕ := 2
def original_horizontal_segments : ℕ := 16
def original_vertical_segments : ℕ := 10

def horizontal_length : ℕ := original_horizontal_segments * side_length
def vertical_length : ℕ := original_vertical_segments * side_length

def perimeter : ℕ := horizontal_length + vertical_length

theorem perimeter_is_correct : perimeter = 52 :=
by 
  -- Proof goes here.
  sorry

end perimeter_is_correct_l26_26429


namespace integer_solutions_ineq_system_l26_26871

theorem integer_solutions_ineq_system:
  ∀ (x : ℤ), 
  (2 * (x - 1) ≤ x + 3) ∧ ((x + 1) / 3 < x - 1) ↔ (x = 3 ∨ x = 4 ∨ x = 5) := 
by 
  intros x 
  split
  · intro h
    cases h with h1 h2
    sorry -- to be proved later
  · intro h
    sorry -- to be proved later

end integer_solutions_ineq_system_l26_26871


namespace factorize_polynomial_find_value_l26_26725

-- Problem 1: Factorize a^3 - 3a^2 - 4a + 12
theorem factorize_polynomial (a : ℝ) :
  a^3 - 3 * a^2 - 4 * a + 12 = (a - 3) * (a - 2) * (a + 2) :=
sorry

-- Problem 2: Given m + n = 5 and m - n = 1, prove m^2 - n^2 + 2m - 2n = 7
theorem find_value (m n : ℝ) (h1 : m + n = 5) (h2 : m - n = 1) :
  m^2 - n^2 + 2 * m - 2 * n = 7 :=
sorry

end factorize_polynomial_find_value_l26_26725


namespace garden_roller_length_l26_26277

/-- The length of a garden roller with diameter 1.4m,
covering 52.8m² in 6 revolutions, and using π = 22/7,
is 2 meters. -/
theorem garden_roller_length
  (diameter : ℝ)
  (total_area_covered : ℝ)
  (revolutions : ℕ)
  (approx_pi : ℝ)
  (circumference : ℝ := approx_pi * diameter)
  (area_per_revolution : ℝ := total_area_covered / (revolutions : ℝ))
  (length : ℝ := area_per_revolution / circumference) :
  diameter = 1.4 ∧ total_area_covered = 52.8 ∧ revolutions = 6 ∧ approx_pi = (22 / 7) → length = 2 :=
by
  sorry

end garden_roller_length_l26_26277


namespace optimal_rental_decision_optimal_purchase_decision_l26_26626

-- Definitions of conditions
def monthly_fee_first : ℕ := 50000
def monthly_fee_second : ℕ := 10000
def probability_seizure : ℚ := 0.5
def moving_cost : ℕ := 70000
def months_first_year : ℕ := 12
def months_seizure : ℕ := 4
def months_after_seizure : ℕ := months_first_year - months_seizure
def purchase_cost : ℕ := 2000000
def installment_period : ℕ := 36

-- Proving initial rental decision
theorem optimal_rental_decision :
  let annual_cost_first := monthly_fee_first * months_first_year
  let annual_cost_second := (monthly_fee_second * months_seizure) + (monthly_fee_first * months_after_seizure) + moving_cost
  annual_cost_second < annual_cost_first := 
by
  sorry

-- Proving purchasing decision
theorem optimal_purchase_decision :
  let total_rent_cost_after_seizure := (monthly_fee_second * months_seizure) + moving_cost + (monthly_fee_first * (4 * months_first_year - months_seizure))
  let total_purchase_cost := purchase_cost
  total_purchase_cost < total_rent_cost_after_seizure :=
by
  sorry

end optimal_rental_decision_optimal_purchase_decision_l26_26626


namespace machines_needed_l26_26523

theorem machines_needed (original_machines : ℕ) (original_days : ℕ) (additional_machines : ℕ) :
  original_machines = 12 → original_days = 40 → 
  additional_machines = ((original_machines * original_days) / (3 * original_days / 4)) - original_machines →
  additional_machines = 4 :=
by
  intros h_machines h_days h_additional
  rw [h_machines, h_days] at h_additional
  sorry

end machines_needed_l26_26523


namespace calculate_value_l26_26934

theorem calculate_value :
  ( (3^3 - 1) / (3^3 + 1) ) * ( (4^3 - 1) / (4^3 + 1) ) * ( (5^3 - 1) / (5^3 + 1) ) * ( (6^3 - 1) / (6^3 + 1) ) * ( (7^3 - 1) / (7^3 + 1) )
  = 57 / 84 := by
  sorry

end calculate_value_l26_26934


namespace team_formation_problem_l26_26952

def num_team_formation_schemes : Nat :=
  let comb (n k : Nat) : Nat := Nat.choose n k
  (comb 5 1 * comb 4 2) + (comb 5 2 * comb 4 1)

theorem team_formation_problem :
  num_team_formation_schemes = 70 :=
sorry

end team_formation_problem_l26_26952


namespace part1_part2_l26_26821

variables (a b c : ℝ)

noncomputable theory

-- Definitions of the conditions
def cond1 (a b c : ℝ) := a > 0 ∧ b > 0 ∧ c > 0
def cond2 (a b c : ℝ) := a^2 + b^2 + 4 * c^2 = 3
def cond3 (b c : ℝ) := b = 2 * c

-- Proof to show a + b + 2c <= 3
theorem part1
  (a b c : ℝ) 
  (h1 : cond1 a b c) 
  (h2 : cond2 a b c) : 
  a + b + 2 * c ≤ 3 :=
sorry

-- Proof to show 1/a + 1/c >= 3
theorem part2
  (a c : ℝ) 
  (h1 : cond1 a (2 * c) c) 
  (h2 : cond2 a (2 * c) c) 
  (h3 : cond3 (2 * c) c) : 
  1 / a + 1 / c ≥ 3 :=
sorry

end part1_part2_l26_26821


namespace card_draw_probability_l26_26592

theorem card_draw_probability:
  let hearts := 13
  let diamonds := 13
  let clubs := 13
  let total_cards := 52
  let first_draw_probability := hearts / (total_cards : ℝ)
  let second_draw_probability := diamonds / (total_cards - 1 : ℝ)
  let third_draw_probability := clubs / (total_cards - 2 : ℝ)
  first_draw_probability * second_draw_probability * third_draw_probability = 2197 / 132600 :=
by
  sorry

end card_draw_probability_l26_26592


namespace max_grapes_leftover_l26_26316

-- Define variables and conditions
def total_grapes (n : ℕ) : ℕ := n
def kids : ℕ := 5
def grapes_leftover (n : ℕ) : ℕ := n % kids

-- The proposition we need to prove
theorem max_grapes_leftover (n : ℕ) (h : n ≥ 5) : grapes_leftover n = 4 :=
sorry

end max_grapes_leftover_l26_26316


namespace triangle_angle_and_area_l26_26956

theorem triangle_angle_and_area (a b c A B C : ℝ)
  (h₁ : c * Real.tan C = Real.sqrt 3 * (a * Real.cos B + b * Real.cos A))
  (h₂ : 0 < C ∧ C < Real.pi)
  (h₃ : c = 2 * Real.sqrt 3) :
  C = Real.pi / 3 ∧ 0 ≤ (1 / 2) * a * b * Real.sin C ∧ (1 / 2) * a * b * Real.sin C ≤ 3 * Real.sqrt 3 :=
by
  sorry

end triangle_angle_and_area_l26_26956


namespace trigonometric_identity_l26_26374

theorem trigonometric_identity (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := 
by 
  sorry

end trigonometric_identity_l26_26374


namespace order_of_f_l26_26996

variable (f : ℝ → ℝ)

/-- Conditions:
1. f is an even function for all x ∈ ℝ
2. f is increasing on [0, +∞)
Question:
Prove that the order of f(-2), f(-π), f(3) is f(-2) < f(3) < f(-π) -/
theorem order_of_f (h_even : ∀ x : ℝ, f (-x) = f x)
                   (h_incr : ∀ {x y : ℝ}, 0 ≤ x → x ≤ y → f x ≤ f y) : 
                   f (-2) < f 3 ∧ f 3 < f (-π) :=
by
  sorry

end order_of_f_l26_26996


namespace minimize_sum_AP_BP_l26_26019

def point := (ℝ × ℝ)

def A : point := (-1, 0)
def B : point := (1, 0)
def center : point := (3, 4)
def radius : ℝ := 2

def on_circle (P : point) : Prop := (P.1 - center.1)^2 + (P.2 - center.2)^2 = radius^2

def AP_squared (P : point) : ℝ := (P.1 - A.1)^2 + (P.2 - A.2)^2
def BP_squared (P : point) : ℝ := (P.1 - B.1)^2 + (P.2 - B.2)^2
def sum_AP_BP_squared (P : point) : ℝ := AP_squared P + BP_squared P

theorem minimize_sum_AP_BP :
  ∀ P : point, on_circle P → sum_AP_BP_squared P = AP_squared (9/5, 12/5) + BP_squared (9/5, 12/5) → 
  P = (9/5, 12/5) :=
sorry

end minimize_sum_AP_BP_l26_26019


namespace find_interest_rate_l26_26042
noncomputable def annualInterestRate (P A : ℝ) (n t : ℕ) (r : ℝ) : Prop :=
  P * (1 + r / n)^(n * t) = A

theorem find_interest_rate :
  annualInterestRate 5000 6050.000000000001 1 2 0.1 :=
by
  -- The proof goes here
  sorry

end find_interest_rate_l26_26042


namespace part1_part2_l26_26828

variable {a b c : ℝ}

-- Condition that a, b, and c are all positive.
variable (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)

-- Condition a^2 + b^2 + 4c^2 = 3.
variable (h_eq : a^2 + b^2 + 4 * c^2 = 3)

-- The first statement to prove: a + b + 2c ≤ 3.
theorem part1 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) : 
  a + b + 2 * c ≤ 3 :=
by
  sorry

-- The second statement to prove: if b = 2c, then 1/a + 1/c ≥ 3.
theorem part2 (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_eq : a^2 + b^2 + 4 * c^2 = 3) (h_eq_bc : b = 2 * c) : 
  1 / a + 1 / c ≥ 3 :=
by
  sorry

end part1_part2_l26_26828


namespace restaurant_problem_l26_26337

theorem restaurant_problem (A K : ℕ) (h1 : A + K = 11) (h2 : 8 * A = 72) : K = 2 :=
by
  sorry

end restaurant_problem_l26_26337


namespace simplify_expression_l26_26427

theorem simplify_expression (q : ℤ) : 
  (((7 * q - 2) + 2 * q * 3) * 4 + (5 + 2 / 2) * (4 * q - 6)) = 76 * q - 44 := by
  sorry

end simplify_expression_l26_26427


namespace first_nonzero_digit_one_div_139_l26_26299

theorem first_nonzero_digit_one_div_139 :
  ∀ n : ℕ, (n > 0 → (∀ m : ℕ, (m > 0 → (m * 10^n) ∣ (10^n * 1 - 1) ∧ n ∣ (139 * 10 ^ (n + 1)) ∧ 10^(n+1 - 1) * 1 - 1 < 10^n))) :=
sorry

end first_nonzero_digit_one_div_139_l26_26299


namespace vanya_more_heads_probability_l26_26460

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l26_26460


namespace triangle_area_l26_26184

-- Defining the rectangle dimensions
def length : ℝ := 35
def width : ℝ := 48

-- Defining the area of the right triangle formed by the diagonal of the rectangle
theorem triangle_area : (1 / 2) * length * width = 840 := by
  sorry

end triangle_area_l26_26184


namespace find_unknown_number_l26_26310

theorem find_unknown_number (x : ℕ) (h₁ : (20 + 40 + 60) / 3 = 5 + (10 + 50 + x) / 3) : x = 45 :=
by sorry

end find_unknown_number_l26_26310


namespace product_remainder_mod_5_l26_26500

theorem product_remainder_mod_5 : (2024 * 1980 * 1848 * 1720) % 5 = 0 := by
  sorry

end product_remainder_mod_5_l26_26500


namespace group_C_questions_l26_26122

theorem group_C_questions (a b c : ℕ) (total_questions : ℕ) (h1 : a + b + c = 100)
  (h2 : b = 23)
  (h3 : a ≥ (6 * (a + 2 * b + 3 * c)) / 10)
  (h4 : 2 * b ≤ (25 * (a + 2 * b + 3 * c)) / 100)
  (h5 : 1 ≤ a ∧ 1 ≤ b ∧ 1 ≤ c) :
  c = 1 :=
sorry

end group_C_questions_l26_26122


namespace problem_statement_l26_26629

theorem problem_statement : ¬ (487.5 * 10^(-10) = 0.0000004875) :=
by
  sorry

end problem_statement_l26_26629


namespace no_partition_equal_product_l26_26491

theorem no_partition_equal_product (n : ℕ) (h_pos : 0 < n) :
  ¬∃ (A B : Finset ℕ), A ∪ B = {n, n+1, n+2, n+3, n+4, n+5} ∧ A ∩ B = ∅ ∧
  A.prod id = B.prod id := sorry

end no_partition_equal_product_l26_26491


namespace candle_height_half_after_9_hours_l26_26023

-- Define the initial heights and burn rates
def initial_height_first : ℝ := 12
def burn_rate_first : ℝ := 2
def initial_height_second : ℝ := 15
def burn_rate_second : ℝ := 3

-- Define the height functions after t hours
def height_first (t : ℝ) : ℝ := initial_height_first - burn_rate_first * t
def height_second (t : ℝ) : ℝ := initial_height_second - burn_rate_second * t

-- Prove that at t = 9, the height of the first candle is half the height of the second candle
theorem candle_height_half_after_9_hours : height_first 9 = 0.5 * height_second 9 := by
  sorry

end candle_height_half_after_9_hours_l26_26023


namespace expression_equivalence_l26_26486

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by sorry

end expression_equivalence_l26_26486


namespace trig_expression_zero_l26_26373

theorem trig_expression_zero (θ : ℝ) (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 :=
sorry

end trig_expression_zero_l26_26373


namespace min_sum_abc_l26_26283

theorem min_sum_abc (a b c : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_prod : a * b * c = 2450) : a + b + c ≥ 82 :=
sorry

end min_sum_abc_l26_26283


namespace parallel_line_slope_l26_26604

theorem parallel_line_slope {x y : ℝ} (h : 3 * x + 6 * y = -24) : 
  ∀ m b : ℝ, (y = m * x + b) → m = -1 / 2 :=
sorry

end parallel_line_slope_l26_26604


namespace cubes_sum_eq_ten_squared_l26_26301

theorem cubes_sum_eq_ten_squared : 1^3 + 2^3 + 3^3 + 4^3 = 10^2 := by
  sorry

end cubes_sum_eq_ten_squared_l26_26301


namespace andrew_subway_time_l26_26022

variable (S : ℝ) -- Let \( S \) be the time Andrew spends on the subway in hours

variable (total_time : ℝ)
variable (bike_time : ℝ)
variable (train_time : ℝ)

noncomputable def travel_conditions := 
  total_time = S + 2 * S + bike_time ∧ 
  total_time = 38 ∧ 
  bike_time = 8

theorem andrew_subway_time
  (S : ℝ)
  (total_time : ℝ)
  (bike_time : ℝ)
  (train_time : ℝ)
  (h : travel_conditions S total_time bike_time) : 
  S = 10 := 
sorry

end andrew_subway_time_l26_26022


namespace population_doubles_l26_26577

theorem population_doubles (initial_population: ℕ) (initial_year: ℕ) (doubling_period: ℕ) (target_population : ℕ) (target_year : ℕ) : 
  initial_population = 500 → 
  initial_year = 2023 → 
  doubling_period = 20 → 
  target_population = 8000 → 
  target_year = 2103 :=
by 
  sorry

end population_doubles_l26_26577


namespace domain_of_g_l26_26203

noncomputable def g (x : ℝ) : ℝ := (x + 2) / Real.sqrt (x^2 - 5 * x + 6)

theorem domain_of_g :
  {x : ℝ | x^2 - 5 * x + 6 ≥ 0} = {x : ℝ | x ≤ 2 ∨ x ≥ 3} :=
by {
  sorry
}

end domain_of_g_l26_26203


namespace seth_sold_candy_bars_l26_26425

theorem seth_sold_candy_bars (max_sold : ℕ) (seth_sold : ℕ) 
  (h1 : max_sold = 24) 
  (h2 : seth_sold = 3 * max_sold + 6) : 
  seth_sold = 78 := 
by sorry

end seth_sold_candy_bars_l26_26425


namespace pencils_in_drawer_l26_26017

/-- 
If there were originally 2 pencils in the drawer and there are now 5 pencils in total, 
then Tim must have placed 3 pencils in the drawer.
-/
theorem pencils_in_drawer (original_pencils tim_pencils total_pencils : ℕ) 
  (h1 : original_pencils = 2) 
  (h2 : total_pencils = 5) 
  (h3 : total_pencils = original_pencils + tim_pencils) : 
  tim_pencils = 3 := 
by
  rw [h1, h2] at h3
  linarith

end pencils_in_drawer_l26_26017


namespace parallelogram_sides_l26_26435

theorem parallelogram_sides (x y : ℝ) 
  (h1 : 5 * x - 7 = 14) 
  (h2 : 3 * y + 4 = 8 * y - 3) : 
  x + y = 5.6 :=
sorry

end parallelogram_sides_l26_26435


namespace smallest_n_for_rotation_matrix_l26_26949

open Real
open Matrix

-- Define the rotation matrix for 60 degrees
def rotationMatrix60 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![cos (π / 3), -sin (π / 3)],
    ![sin (π / 3), cos (π / 3)]
  ]

-- Identity matrix of size 2
def identityMatrix2 : Matrix (Fin 2) (Fin 2) ℝ :=
  1

theorem smallest_n_for_rotation_matrix :
  ∃ n : ℕ, n > 0 ∧ (rotationMatrix60 ^ n = identityMatrix2) ∧ ∀ m : ℕ, m > 0 → m < n → rotationMatrix60 ^ m ≠ identityMatrix2 :=
by 
  existsi 6
  split
  · exact Nat.succ_pos'
  · sorry
  · sorry

end smallest_n_for_rotation_matrix_l26_26949


namespace lattice_points_in_triangle_271_l26_26265

def Pick_theorem (N L S : ℝ) : Prop :=
  S = N + (1/2) * L - 1

noncomputable def lattice_triangle_inside_points (A B O : ℕ × ℕ) : ℕ :=
  271

theorem lattice_points_in_triangle_271 :
  ∃ N L : ℝ, 
  let S := 300 in
  let L := 60 in
  A = (0, 30) ∧ B = (20, 10) ∧
  O = (0, 0) ∧ Pick_theorem N L S :=
sorry

end lattice_points_in_triangle_271_l26_26265


namespace soda_cost_l26_26041

-- Definitions based on conditions of the problem
variable (b s : ℤ)
variable (h1 : 4 * b + 3 * s = 540)
variable (h2 : 3 * b + 2 * s = 390)

-- The theorem to prove the cost of a soda
theorem soda_cost : s = 60 := by
  sorry

end soda_cost_l26_26041


namespace factorial_division_l26_26803

theorem factorial_division :
  9! = 362880 → (9! / 4!) = 15120 := by
  sorry

end factorial_division_l26_26803


namespace pens_left_is_25_l26_26632

def total_pens_left (initial_blue initial_black initial_red removed_blue removed_black : Nat) : Nat :=
  let blue_left := initial_blue - removed_blue
  let black_left := initial_black - removed_black
  let red_left := initial_red
  blue_left + black_left + red_left

theorem pens_left_is_25 :
  total_pens_left 9 21 6 4 7 = 25 :=
by 
  rw [total_pens_left, show 9 - 4 = 5 from Nat.sub_eq_of_eq_add (rfl), show 21 - 7 = 14 from Nat.sub_eq_of_eq_add (rfl)]
  rfl

end pens_left_is_25_l26_26632


namespace johnson_family_seating_l26_26878

/-- The Johnson family has 5 sons and 4 daughters. We want to find the number of ways to seat them in a row of 9 chairs such that at least 2 boys are next to each other. -/
theorem johnson_family_seating : 
  let boys := 5 in
  let girls := 4 in
  let total_children := boys + girls in
  fact total_children - 
  2 * (fact boys * fact girls) = 357120 := 
by
  let boys := 5
  let girls := 4
  let total_children := boys + girls
  have total_arrangements : ℕ := fact total_children
  have no_two_boys_next_to_each_other : ℕ := 2 * (fact boys * fact girls)
  have at_least_two_boys_next_to_each_other : ℕ := total_arrangements - no_two_boys_next_to_each_other
  show at_least_two_boys_next_to_each_other = 357120
  sorry

end johnson_family_seating_l26_26878


namespace find_a_in_subset_l26_26997

theorem find_a_in_subset 
  (A : Set ℝ)
  (B : Set ℝ)
  (hA : A = { x | x^2 ≠ 1 })
  (hB : ∃ a : ℝ, B = { x | a * x = 1 })
  (h_subset : B ⊆ A) : 
  ∃ a : ℝ, a = 0 ∨ a = 1 ∨ a = -1 := 
by
  sorry

end find_a_in_subset_l26_26997


namespace highway_length_on_map_l26_26153

theorem highway_length_on_map (total_length_km : ℕ) (scale : ℚ) (length_on_map_cm : ℚ) 
  (h1 : total_length_km = 155) (h2 : scale = 1 / 500000) :
  length_on_map_cm = 31 :=
by
  sorry

end highway_length_on_map_l26_26153


namespace loisa_saves_70_l26_26712

def tablet_cash_price : ℕ := 450
def down_payment : ℕ := 100
def first_4_months_payment : ℕ := 40
def next_4_months_payment : ℕ := 35
def last_4_months_payment : ℕ := 30
def total_installment_payment : ℕ := down_payment + (4 * first_4_months_payment) + (4 * next_4_months_payment) + (4 * last_4_months_payment)
def savings : ℕ := total_installment_payment - tablet_cash_price

theorem loisa_saves_70 : savings = 70 := by
  sorry

end loisa_saves_70_l26_26712


namespace scientific_notation_to_standard_form_l26_26527

theorem scientific_notation_to_standard_form :
  - 3.96 * 10^5 = -396000 :=
sorry

end scientific_notation_to_standard_form_l26_26527


namespace determine_a_if_fx_odd_l26_26973

theorem determine_a_if_fx_odd (a : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f x = 2^x + a * 2^(-x)) (h2 : ∀ x, f (-x) = -f x) : a = -1 :=
by
  sorry

end determine_a_if_fx_odd_l26_26973


namespace books_borrowed_by_lunchtime_l26_26967

theorem books_borrowed_by_lunchtime (x : ℕ) :
  (∀ x : ℕ, 100 - x + 40 - 30 = 60) → (x = 50) :=
by
  intro h
  have eqn := h x
  sorry

end books_borrowed_by_lunchtime_l26_26967


namespace limit_of_R_l26_26247

noncomputable def R (m b : ℝ) : ℝ :=
  let x := ((-b) + Real.sqrt (b^2 + 4 * m)) / 2
  m * x + 3 

theorem limit_of_R (b : ℝ) (hb : b ≠ 0) : 
  (∀ m : ℝ, m < 3) → 
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 0) < δ → abs ((R x (-b) - R x b) / x - b) < ε) :=
by
  sorry

end limit_of_R_l26_26247


namespace find_interest_rate_of_second_part_l26_26039

-- Definitions for the problem
def total_sum : ℚ := 2678
def P2 : ℚ := 1648
def P1 : ℚ := total_sum - P2
def r1 : ℚ := 0.03  -- 3% per annum
def t1 : ℚ := 8     -- 8 years
def I1 : ℚ := P1 * r1 * t1
def t2 : ℚ := 3     -- 3 years

-- Statement to prove
theorem find_interest_rate_of_second_part : ∃ r2 : ℚ, I1 = P2 * r2 * t2 ∧ r2 * 100 = 5 := by
  sorry

end find_interest_rate_of_second_part_l26_26039


namespace minimum_value_of_f_l26_26096

noncomputable def f (x : ℝ) : ℝ := sorry

theorem minimum_value_of_f :
  (∀ x : ℝ, f (x + 1) + f (x - 1) = 2 * x^2 - 4 * x) →
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ m = -2 :=
by
  sorry

end minimum_value_of_f_l26_26096


namespace exists_li_ge_2018_l26_26411

noncomputable def polynomial_S (i : ℕ) (li : ℕ) : Polynomial ℝ :=
  Polynomial.C 1 * Polynomial.X ^ 2 - Polynomial.C 2018 * Polynomial.X + Polynomial.C li

theorem exists_li_ge_2018
  (n : ℕ) (h_n : 1 < n ∧ n < 2018)
  (l : Fin n → ℕ)
  (h_distinct : Function.Injective l)
  (h_pos : ∀ i, 0 < l i)
  (h_root : ∃ x : ℤ, ∑ i in Finset.univ, (polynomial_S i (l i)).eval x = 0) :
  ∃ i, 2018 ≤ l i := 
sorry

end exists_li_ge_2018_l26_26411


namespace trigonometric_identity_l26_26083

open Real

theorem trigonometric_identity (α : ℝ) (h1 : tan α = 4/3) (h2 : 0 < α ∧ α < π / 2) :
  sin (π + α) + cos (π - α) = -7/5 :=
by
  sorry

end trigonometric_identity_l26_26083


namespace tan_identity_l26_26370

variable {θ : ℝ} (h : Real.tan θ = 3)

theorem tan_identity (h : Real.tan θ = 3) :
  (1 - Real.cos θ) / Real.sin θ - Real.sin θ / (1 + Real.cos θ) = 0 := sorry

end tan_identity_l26_26370


namespace parallel_line_through_intersection_perpendicular_line_through_intersection_l26_26215

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and parallel to the line 2x - y - 1 = 0 
is 2x - y + 1 = 0 --/
theorem parallel_line_through_intersection :
  ∃ (c : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (2 * x - y + c = 0) ∧ c = 1 :=
by
  sorry

/-- Given two lines l1: x + y - 4 = 0 and l2: x - y + 2 = 0,
the line passing through their intersection point and perpendicular to the line 2x - y - 1 = 0
is x + 2y - 7 = 0 --/
theorem perpendicular_line_through_intersection :
  ∃ (d : ℝ), ∃ (x y : ℝ), (x + y - 4 = 0 ∧ x - y + 2 = 0) ∧ (x + 2 * y + d = 0) ∧ d = -7 :=
by
  sorry

end parallel_line_through_intersection_perpendicular_line_through_intersection_l26_26215


namespace part1_part2_l26_26839

variables (a b c : ℝ)

-- Ensure that a, b and c are all positive numbers
axiom (ha : a > 0)
axiom (hb : b > 0)
axiom (hc : c > 0)

-- Given condition
axiom (h_cond : a^2 + b^2 + 4 * c^2 = 3)

/- Part (1): Prove that a + b + 2c ≤ 3 -/
theorem part1 : a + b + 2 * c ≤ 3 := 
sorry

/- Part (2): Additional condition b = 2c and prove 1/a + 1/c ≥ 3 -/
axiom (h_b_eq_2c : b = 2 * c)

theorem part2 : 1 / a + 1 / c ≥ 3 := 
sorry

end part1_part2_l26_26839


namespace purple_valley_skirts_l26_26722

def AzureValley : ℕ := 60

def SeafoamValley (A : ℕ) : ℕ := (2 * A) / 3

def PurpleValley (S : ℕ) : ℕ := S / 4

theorem purple_valley_skirts :
  PurpleValley (SeafoamValley AzureValley) = 10 :=
by
  sorry

end purple_valley_skirts_l26_26722


namespace probability_three_correct_l26_26077

open Fintype

/-- 
The probability that exactly three out of five packages are delivered 
to the correct houses, given random delivery, is 1/6.
-/
theorem probability_three_correct : 
  (∃ (X : Finset (Fin 5)) (hX : X.card = 3) (Y : Finset (Fin 5)) (hY : Y.card = 2) (f : (Fin 5) → (Fin 5)),
    ∀ (x ∈ X, f x = x) ∧ (∀ y ∈ Y, f y ≠ y) ∧ HasDistribNeg.negOne Y y 
    ∧ ∑ y in Y, 1! = 2) 
  → (Real.ofRat (⅙)) := sorry

end probability_three_correct_l26_26077


namespace anna_plants_needed_l26_26196

def required_salads : ℕ := 12
def salads_per_plant : ℕ := 3
def loss_fraction : ℚ := 1 / 2

theorem anna_plants_needed : 
  ∀ (plants_needed : ℕ), 
  plants_needed = Nat.ceil (required_salads / salads_per_plant * (1 / (1 - (loss_fraction : ℚ)))) :=
by
  sorry

end anna_plants_needed_l26_26196


namespace math_problem_l26_26596

/-
Two mathematicians take a morning coffee break each day.
They arrive at the cafeteria independently, at random times between 9 a.m. and 10:30 a.m.,
and stay for exactly m minutes.
Given the probability that either one arrives while the other is in the cafeteria is 30%,
and m = a - b√c, where a, b, and c are positive integers, and c is not divisible by the square of any prime,
prove that a + b + c = 127.

-/

noncomputable def is_square_free (c : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p * p ∣ c → False

theorem math_problem
  (m a b c : ℕ)
  (h1 : 0 < m)
  (h2 : m = a - b * Real.sqrt c)
  (h3 : is_square_free c)
  (h4 : 30 * (90 * 90) / 100 = (90 - m) * (90 - m)) :
  a + b + c = 127 :=
sorry

end math_problem_l26_26596


namespace initial_mixture_volume_l26_26037

/--
Given:
1. A mixture initially contains 20% water.
2. When 13.333333333333334 liters of water is added, water becomes 25% of the new mixture.

Prove that the initial volume of the mixture is 200 liters.
-/
theorem initial_mixture_volume (V : ℝ) (h1 : V > 0) (h2 : 0.20 * V + 13.333333333333334 = 0.25 * (V + 13.333333333333334)) : V = 200 :=
sorry

end initial_mixture_volume_l26_26037


namespace rhombus_construction_possible_l26_26346

-- Definitions for points, lines, and distances
variables {Point : Type} {Line : Type}
def is_parallel (l1 l2 : Line) : Prop := sorry
def distance_between (l1 l2 : Line) : ℝ := sorry
def point_on_line (p : Point) (l : Line) : Prop := sorry

-- Given parallel lines l₁ and l₂ and their distance a
variable {l1 l2 : Line}
variable (a : ℝ)
axiom parallel_lines : is_parallel l1 l2
axiom distance_eq_a : distance_between l1 l2 = a

-- Given points A and B
variable (A B : Point)

-- Definition of a rhombus that meets the criteria
noncomputable def construct_rhombus (A B : Point) (l1 l2 : Line) (a : ℝ) : Prop :=
  ∃ C1 C2 D1 D2 : Point, 
    point_on_line C1 l1 ∧ 
    point_on_line D1 l2 ∧ 
    point_on_line C2 l1 ∧ 
    point_on_line D2 l2 ∧ 
    sorry -- additional conditions ensuring sides passing through A and B and forming a rhombus

theorem rhombus_construction_possible : 
  construct_rhombus A B l1 l2 a :=
sorry

end rhombus_construction_possible_l26_26346


namespace probability_three_correct_l26_26076

open Fintype

/-- 
The probability that exactly three out of five packages are delivered 
to the correct houses, given random delivery, is 1/6.
-/
theorem probability_three_correct : 
  (∃ (X : Finset (Fin 5)) (hX : X.card = 3) (Y : Finset (Fin 5)) (hY : Y.card = 2) (f : (Fin 5) → (Fin 5)),
    ∀ (x ∈ X, f x = x) ∧ (∀ y ∈ Y, f y ≠ y) ∧ HasDistribNeg.negOne Y y 
    ∧ ∑ y in Y, 1! = 2) 
  → (Real.ofRat (⅙)) := sorry

end probability_three_correct_l26_26076


namespace cube_volume_in_cubic_yards_l26_26318

def volume_in_cubic_feet := 64
def cubic_feet_per_cubic_yard := 27

theorem cube_volume_in_cubic_yards : 
  volume_in_cubic_feet / cubic_feet_per_cubic_yard = 64 / 27 :=
by
  sorry

end cube_volume_in_cubic_yards_l26_26318


namespace sufficient_not_necessary_condition_l26_26436

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x < 1 → x < 2) ∧ ¬ (x < 2 → x < 1) :=
by
  sorry

end sufficient_not_necessary_condition_l26_26436


namespace linda_spent_amount_l26_26905

theorem linda_spent_amount :
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  total_cost = 6.80 :=
by
  let cost_notebooks := 3 * 1.20
  let cost_pencils := 1.50
  let cost_pens := 1.70
  let total_cost := cost_notebooks + cost_pencils + cost_pens
  show total_cost = 6.80
  sorry

end linda_spent_amount_l26_26905


namespace remainder_when_divided_by_5_l26_26322

theorem remainder_when_divided_by_5 (k : ℕ) 
  (h1 : k % 6 = 5) 
  (h2 : k % 7 = 3)
  (h3 : k < 41) : k % 5 = 2 :=
sorry

end remainder_when_divided_by_5_l26_26322


namespace sum_of_cubes_l26_26220

theorem sum_of_cubes (x y : ℝ) (h₁ : x + y = -1) (h₂ : x * y = -1) : x^3 + y^3 = -4 := by
  sorry

end sum_of_cubes_l26_26220


namespace track_width_l26_26185

theorem track_width (r_1 r_2 : ℝ) (h1 : r_2 = 20) (h2 : 2 * Real.pi * r_1 - 2 * Real.pi * r_2 = 20 * Real.pi) : r_1 - r_2 = 10 :=
sorry

end track_width_l26_26185


namespace change_received_l26_26921

def laptop_price : ℕ := 600
def smartphone_price : ℕ := 400
def num_laptops : ℕ := 2
def num_smartphones : ℕ := 4
def initial_amount : ℕ := 3000

theorem change_received : (initial_amount -
  ((laptop_price * num_laptops) + (smartphone_price * num_smartphones))) = 200 :=
by
  calc
    initial_amount - ((laptop_price * num_laptops) + (smartphone_price * num_smartphones))
        = 3000 - ((600 * 2) + (400 * 4)) : by simp [initial_amount, laptop_price, num_laptops, smartphone_price, num_smartphones]
    ... = 3000 - (1200 + 1600) : by norm_num
    ... = 3000 - 2800 : by norm_num
    ... = 200 : by norm_num

end change_received_l26_26921


namespace m_range_satisfies_inequality_l26_26242

open Real

noncomputable def f (x : ℝ) : ℝ := -2 * x + sin x

theorem m_range_satisfies_inequality :
  ∀ (m : ℝ), f (2 * m ^ 2 - m + π - 1) ≥ -2 * π ↔ -1 / 2 ≤ m ∧ m ≤ 1 := 
by
  sorry

end m_range_satisfies_inequality_l26_26242


namespace factorable_polynomial_with_integer_coeffs_l26_26655

theorem factorable_polynomial_with_integer_coeffs (m : ℤ) : 
  ∃ A B C D E F : ℤ, 
  (A * D = 1) ∧ (B * E = 0) ∧ (A * E + B * D = 5) ∧ 
  (A * F + C * D = 1) ∧ (B * F + C * E = 2 * m) ∧ (C * F = -10) ↔ m = 5 := sorry

end factorable_polynomial_with_integer_coeffs_l26_26655


namespace total_increase_area_l26_26764

theorem total_increase_area (increase_broccoli increase_cauliflower increase_cabbage : ℕ)
    (area_broccoli area_cauliflower area_cabbage : ℝ)
    (h1 : increase_broccoli = 79)
    (h2 : increase_cauliflower = 25)
    (h3 : increase_cabbage = 50)
    (h4 : area_broccoli = 1)
    (h5 : area_cauliflower = 2)
    (h6 : area_cabbage = 1.5) :
    increase_broccoli * area_broccoli +
    increase_cauliflower * area_cauliflower +
    increase_cabbage * area_cabbage = 204 := 
by 
    sorry

end total_increase_area_l26_26764


namespace sum_volumes_spheres_l26_26473

theorem sum_volumes_spheres (l : ℝ) (h_l : l = 2) : 
  ∑' (n : ℕ), (4 / 3) * π * ((1 / (3 ^ (n + 1))) ^ 3) = (2 * π / 39) :=
by
  sorry

end sum_volumes_spheres_l26_26473


namespace kittens_more_than_twice_puppies_l26_26156

-- Define the number of puppies
def num_puppies : ℕ := 32

-- Define the number of kittens
def num_kittens : ℕ := 78

-- Define the problem statement
theorem kittens_more_than_twice_puppies :
  num_kittens = 2 * num_puppies + 14 :=
by sorry

end kittens_more_than_twice_puppies_l26_26156


namespace gcd_2024_2048_l26_26450

theorem gcd_2024_2048 : Nat.gcd 2024 2048 = 8 := by
  sorry

end gcd_2024_2048_l26_26450


namespace eduardo_frankie_classes_total_l26_26942

theorem eduardo_frankie_classes_total (eduardo_classes : ℕ) (h₁ : eduardo_classes = 3) 
                                       (h₂ : ∀ frankie_classes, frankie_classes = 2 * eduardo_classes) :
  ∃ total_classes : ℕ, total_classes = eduardo_classes + 2 * eduardo_classes := 
by
  use 3 + 2 * 3
  sorry

end eduardo_frankie_classes_total_l26_26942


namespace num_sets_B_l26_26560

open Set

def A : Set ℕ := {1, 3}

theorem num_sets_B :
  ∃ (B : ℕ → Set ℕ), (∀ b, B b ∪ A = {1, 3, 5}) ∧ (∃ s t u v, B s = {5} ∧
                                                   B t = {1, 5} ∧
                                                   B u = {3, 5} ∧
                                                   B v = {1, 3, 5} ∧ 
                                                   s ≠ t ∧ s ≠ u ∧ s ≠ v ∧
                                                   t ≠ u ∧ t ≠ v ∧
                                                   u ≠ v) :=
sorry

end num_sets_B_l26_26560


namespace train_length_l26_26641

theorem train_length (speed_kmph : ℤ) (time_sec : ℤ) (expected_length_m : ℤ) 
    (speed_kmph_eq : speed_kmph = 72)
    (time_sec_eq : time_sec = 7)
    (expected_length_eq : expected_length_m = 140) :
    expected_length_m = (speed_kmph * 1000 / 3600) * time_sec :=
by 
    sorry

end train_length_l26_26641


namespace tan_double_angle_l26_26116

theorem tan_double_angle (α : ℝ) (x y : ℝ) (hxy : y / x = -2) : 
  2 * y / (1 - (y / x)^2) = (4 : ℝ) / 3 :=
by sorry

end tan_double_angle_l26_26116


namespace Emily_walks_more_distance_than_Troy_l26_26160

theorem Emily_walks_more_distance_than_Troy (Troy_distance Emily_distance : ℕ) (days : ℕ) 
  (hTroy : Troy_distance = 75) (hEmily : Emily_distance = 98) (hDays : days = 5) : 
  ((Emily_distance * 2 - Troy_distance * 2) * days) = 230 :=
by
  sorry

end Emily_walks_more_distance_than_Troy_l26_26160


namespace A_investment_is_correct_l26_26190

-- Definitions based on the given conditions
def B_investment : ℝ := 8000
def C_investment : ℝ := 10000
def P_B : ℝ := 1000
def diff_P_A_P_C : ℝ := 500

-- Main statement we need to prove
theorem A_investment_is_correct (A_investment : ℝ) 
  (h1 : B_investment = 8000) 
  (h2 : C_investment = 10000)
  (h3 : P_B = 1000)
  (h4 : diff_P_A_P_C = 500)
  (h5 : A_investment = B_investment * (P_B / 1000) * 1.5) :
  A_investment = 12000 :=
sorry

end A_investment_is_correct_l26_26190


namespace problem1_monotonic_f_problem2a_monotonic_g_small_a_problem2b_monotonic_g_large_a_problem3_max_value_b_l26_26385

noncomputable def f (x b : ℝ) : ℝ := (1/2) * x^2 + b * x + real.log x
noncomputable def g (x b a : ℝ) : ℝ := f x b - b * x - (1 + a) / 2 * x^2

-- Problem (1)
theorem problem1_monotonic_f (x : ℝ) (b : ℝ) (hx0 : 0 < x) (hf : ∀ x, 0 < x → 0 ≤ x + b + 1 / x) : b ≥ -2 :=
sorry

-- Problem (2a)
theorem problem2a_monotonic_g_small_a (x a : ℝ) (hx0 : 0 < x) (ha : a ≤ 0) (hg : ∀ x, 0 < x → 0 ≤ 1 / x - a * x) : ∀ x, 0 < x → monotone_on (g x b) (set.Ioi 0) :=
sorry

-- Problem (2b)
theorem problem2b_monotonic_g_large_a (x a : ℝ) (hx0 : 0 < x) (ha : 0 < a) :
  (monotone_on (g x b a) (set.Icc (0 : ℝ) (real.sqrt a / a))) ∧ (antimono_on (g x b a) (set.Ioi (real.sqrt a / a))) :=
sorry

-- Problem (3)
theorem problem3_max_value_b (x : ℝ) (b : ℝ) (hx0 : 0 < x ∧ x ≤ 1)
  (hf_ineq : ∀ x, 0 < x ∧ x ≤ 1 → f x b ≤ x^2 + (1 / (2 * x^2)) - 3 * x + 1) (hmf : b ≥ -2) : -2 ≤ b ∧ b ≤ -1 :=
sorry

end problem1_monotonic_f_problem2a_monotonic_g_small_a_problem2b_monotonic_g_large_a_problem3_max_value_b_l26_26385


namespace interior_lattice_points_of_triangle_l26_26969

-- Define the vertices of the triangle
def A : (ℤ × ℤ) := (0, 99)
def B : (ℤ × ℤ) := (5, 100)
def C : (ℤ × ℤ) := (2003, 500)

-- The problem is to find the number of interior lattice points
-- according to Pick's Theorem (excluding boundary points).

theorem interior_lattice_points_of_triangle :
  let I : ℤ := 0 -- number of interior lattice points
  I = 0 :=
by
  sorry

end interior_lattice_points_of_triangle_l26_26969


namespace petya_green_balls_l26_26263

theorem petya_green_balls (total_balls : ℕ) (red_balls blue_balls green_balls : ℕ)
  (h1 : total_balls = 50)
  (h2 : ∀ s, s.card ≥ 34 → ∃ r, r ∈ s ∧ r = red_balls)
  (h3 : ∀ s, s.card ≥ 35 → ∃ b, b ∈ s ∧ b = blue_balls)
  (h4 : ∀ s, s.card ≥ 36 → ∃ g, g ∈ s ∧ g = green_balls) :
  green_balls = 15 ∨ green_balls = 16 ∨ green_balls = 17 :=
sorry

end petya_green_balls_l26_26263


namespace seq_proof_l26_26958
noncomputable def seq1_arithmetic (a1 a2 : ℝ) : Prop :=
  ∃ d : ℝ, a1 = -2 + d ∧ a2 = a1 + d ∧ -8 = a2 + d

noncomputable def seq2_geometric (b1 b2 b3 : ℝ) : Prop :=
  ∃ r : ℝ, b1 = -2 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -8 = b3 * r

theorem seq_proof (a1 a2 b1 b2 b3: ℝ) (h1 : seq1_arithmetic a1 a2) (h2 : seq2_geometric b1 b2 b3) :
  (a2 - a1) / b2 = 1 / 2 :=
sorry

end seq_proof_l26_26958


namespace find_S2_side_length_l26_26726

theorem find_S2_side_length 
    (x r : ℝ)
    (h1 : 2 * r + x = 2100)
    (h2 : 3 * x + 300 = 3500)
    : x = 1066.67 := 
sorry

end find_S2_side_length_l26_26726


namespace problem_statement_l26_26852

theorem problem_statement (n : ℕ) (a b c : ℕ → ℤ)
  (h1 : n > 0)
  (h2 : ∀ i j, i ≠ j → ¬ (a i - a j) % n = 0 ∧
                           ¬ ((b i + c i) - (b j + c j)) % n = 0 ∧
                           ¬ (b i - b j) % n = 0 ∧
                           ¬ ((c i + a i) - (c j + a i)) % n = 0 ∧
                           ¬ (c i - c j) % n = 0 ∧
                           ¬ ((a i + b i) - (a j + b i)) % n = 0 ∧
                           ¬ ((a i + b i + c i) - (a j + b i + c j)) % n = 0) :
  (Odd n) ∧ (¬ ∃ k, n = 3 * k) :=
by sorry

end problem_statement_l26_26852


namespace students_per_group_l26_26154

-- Define the conditions:
def total_students : ℕ := 120
def not_picked_students : ℕ := 22
def groups : ℕ := 14

-- Calculate the picked students:
def picked_students : ℕ := total_students - not_picked_students

-- Statement of the problem:
theorem students_per_group : picked_students / groups = 7 :=
  by sorry

end students_per_group_l26_26154


namespace middle_rectangle_frequency_l26_26547

theorem middle_rectangle_frequency (S A : ℝ) (h1 : S + A = 100) (h2 : A = S / 3) : A = 25 :=
by
  sorry

end middle_rectangle_frequency_l26_26547


namespace find_third_number_l26_26623

theorem find_third_number : ∃ (x : ℝ), 0.3 * 0.8 + x * 0.5 = 0.29 ∧ x = 0.1 :=
by
  use 0.1
  sorry

end find_third_number_l26_26623


namespace train_length_is_225_m_l26_26476

noncomputable def speed_kmph : ℝ := 90
noncomputable def time_s : ℝ := 9

noncomputable def speed_ms : ℝ := speed_kmph / 3.6
noncomputable def distance_m (speed : ℝ) (time : ℝ) : ℝ := speed * time

theorem train_length_is_225_m :
  distance_m speed_ms time_s = 225 := by
  sorry

end train_length_is_225_m_l26_26476


namespace min_sum_of_factors_l26_26285

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end min_sum_of_factors_l26_26285


namespace sum_of_remainders_mod_11_l26_26025

theorem sum_of_remainders_mod_11
    (a b c d : ℤ)
    (h₁ : a % 11 = 2)
    (h₂ : b % 11 = 4)
    (h₃ : c % 11 = 6)
    (h₄ : d % 11 = 8) :
    (a + b + c + d) % 11 = 9 :=
by
  sorry

end sum_of_remainders_mod_11_l26_26025


namespace frac_sum_equals_seven_eights_l26_26975

theorem frac_sum_equals_seven_eights (p q r u v w : ℝ) 
  (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (hu : 0 < u) (hv : 0 < v) (hw : 0 < w)
  (h1 : p^2 + q^2 + r^2 = 49)
  (h2 : u^2 + v^2 + w^2 = 64)
  (h3 : p * u + q * v + r * w = 56) :
  (p + q + r) / (u + v + w) = 7 / 8 := 
  sorry

end frac_sum_equals_seven_eights_l26_26975


namespace one_fourth_of_2_pow_30_eq_2_pow_x_l26_26397

theorem one_fourth_of_2_pow_30_eq_2_pow_x (x : ℕ) : (1 / 4 : ℝ) * (2:ℝ)^30 = (2:ℝ)^x → x = 28 := by
  sorry

end one_fourth_of_2_pow_30_eq_2_pow_x_l26_26397


namespace no_distinct_ordered_pairs_l26_26217

theorem no_distinct_ordered_pairs (x y : ℕ) (h₁ : 0 < x) (h₂ : 0 < y) :
  (x^2 * y^2)^2 - 14 * x^2 * y^2 + 49 ≠ 0 :=
by
  sorry

end no_distinct_ordered_pairs_l26_26217


namespace largest_inscribed_equilateral_triangle_area_l26_26933

noncomputable def inscribed_triangle_area (r : ℝ) : ℝ :=
  let s := r * (3 / Real.sqrt 3)
  let h := (Real.sqrt 3 / 2) * s
  (1 / 2) * s * h

theorem largest_inscribed_equilateral_triangle_area :
  inscribed_triangle_area 10 = 75 * Real.sqrt 3 :=
by
  simp [inscribed_triangle_area]
  sorry

end largest_inscribed_equilateral_triangle_area_l26_26933


namespace unit_vector_perpendicular_l26_26075

theorem unit_vector_perpendicular (x y : ℝ) (h : 3 * x + 4 * y = 0) (m : x^2 + y^2 = 1) : 
  (x = -4/5 ∧ y = 3/5) ∨ (x = 4/5 ∧ y = -3/5) :=
by
  sorry

end unit_vector_perpendicular_l26_26075


namespace number_of_winning_scores_l26_26402

theorem number_of_winning_scores : 
  ∃ (scores: ℕ), scores = 19 := by
  sorry

end number_of_winning_scores_l26_26402


namespace abs_lt_one_iff_sq_lt_one_l26_26979

variable {x : ℝ}

theorem abs_lt_one_iff_sq_lt_one : |x| < 1 ↔ x^2 < 1 := sorry

end abs_lt_one_iff_sq_lt_one_l26_26979


namespace min_sum_of_factors_l26_26284

theorem min_sum_of_factors 
  (a b c: ℕ)
  (h1: a > 0)
  (h2: b > 0)
  (h3: c > 0)
  (h4: a * b * c = 2450) :
  a + b + c ≥ 76 :=
sorry

end min_sum_of_factors_l26_26284


namespace max_value_of_x_plus_y_l26_26400

variable (x y : ℝ)

-- Define the condition
def condition : Prop := x^2 + y + 3 * x - 3 = 0

-- Define the proof statement
theorem max_value_of_x_plus_y (hx : condition x y) : x + y ≤ 4 :=
sorry

end max_value_of_x_plus_y_l26_26400


namespace complement_union_l26_26100

open Set

def U : Set ℕ := {x | x < 6}

def A : Set ℕ := {1, 3}

def B : Set ℕ := {3, 5}

theorem complement_union :
  (U \ (A ∪ B)) = {0, 2, 4} :=
by
  sorry

end complement_union_l26_26100


namespace initial_video_files_l26_26480

theorem initial_video_files (V : ℕ) (h1 : 26 + V - 48 = 14) : V = 36 := 
by
  sorry

end initial_video_files_l26_26480


namespace product_of_integers_with_cubes_sum_189_l26_26010

theorem product_of_integers_with_cubes_sum_189 :
  ∃ a b : ℤ, a^3 + b^3 = 189 ∧ a * b = 20 :=
by
  -- The proof is omitted for brevity.
  sorry

end product_of_integers_with_cubes_sum_189_l26_26010


namespace factor_expression_l26_26208

theorem factor_expression (b : ℤ) : 53 * b^2 + 159 * b = 53 * b * (b + 3) :=
by
  sorry

end factor_expression_l26_26208


namespace alyssa_bought_224_new_cards_l26_26990

theorem alyssa_bought_224_new_cards
  (initial_cards : ℕ)
  (after_purchase_cards : ℕ)
  (h1 : initial_cards = 676)
  (h2 : after_purchase_cards = 900) :
  after_purchase_cards - initial_cards = 224 :=
by
  -- Placeholder to avoid proof since it's explicitly not required 
  sorry

end alyssa_bought_224_new_cards_l26_26990


namespace my_cousin_reading_time_l26_26719

-- Define the conditions
def reading_time_me_hours : ℕ := 3
def reading_speed_ratio : ℕ := 5
def reading_time_me_min : ℕ := reading_time_me_hours * 60

-- Define the statement to be proved
theorem my_cousin_reading_time : (reading_time_me_min / reading_speed_ratio) = 36 := by
  sorry

end my_cousin_reading_time_l26_26719


namespace purple_valley_skirts_l26_26724

theorem purple_valley_skirts (azure_valley_skirts : ℕ) (h1 : azure_valley_skirts = 60) :
    let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts in
    let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts in
    purple_valley_skirts = 10 :=
by
  let seafoam_valley_skirts := (2 / 3 : ℚ) * azure_valley_skirts
  let purple_valley_skirts := (1 / 4 : ℚ) * seafoam_valley_skirts
  have h2 : seafoam_valley_skirts = (2 / 3 : ℚ) * 60 := by
    rw [h1]
  have h3 : purple_valley_skirts = (1 / 4 : ℚ) * ((2 / 3 : ℚ) * 60) := by
    rw [h2]
  have h4 : purple_valley_skirts = (1 / 4 : ℚ) * 40 := by
    norm_num [h3]
  have h5 : purple_valley_skirts = 10 := by
    norm_num [h4]
  exact h5

end purple_valley_skirts_l26_26724


namespace sample_var_interpretation_l26_26540

theorem sample_var_interpretation (squared_diffs : Fin 10 → ℝ) :
  (10 = 10) ∧ (∀ i, squared_diffs i = (i - 20)^2) →
  (∃ n: ℕ, n = 10 ∧ ∃ μ: ℝ, μ = 20) :=
by
  intro h
  sorry

end sample_var_interpretation_l26_26540


namespace xy_product_of_sample_l26_26684

/-- Given a sample {9, 10, 11, x, y} such that the average is 10 and the standard deviation is sqrt(2), 
    prove that the product of x and y is 96. -/
theorem xy_product_of_sample (x y : ℝ) 
  (h_avg : (9 + 10 + 11 + x + y) / 5 = 10)
  (h_stddev : ( (9 - 10)^2 + (10 - 10)^2 + (11 - 10)^2 + (x - 10)^2 + (y - 10)^2 ) / 5 = 2) :
  x * y = 96 :=
by
  -- Proof goes here
  sorry

end xy_product_of_sample_l26_26684


namespace remainder_when_divided_by_eleven_l26_26362

-- Definitions from the conditions
def two_pow_five_mod_eleven : ℕ := 10
def two_pow_ten_mod_eleven : ℕ := 1
def ten_mod_eleven : ℕ := 10
def ten_square_mod_eleven : ℕ := 1

-- Proposition we want to prove
theorem remainder_when_divided_by_eleven :
  (7 * 10^20 + 2^20) % 11 = 8 := 
by 
  -- Proof goes here
  sorry

end remainder_when_divided_by_eleven_l26_26362


namespace seth_sold_78_candy_bars_l26_26424

def num_candy_sold_by_seth (num_candy_max: Nat): Nat :=
  3 * num_candy_max + 6

theorem seth_sold_78_candy_bars :
  num_candy_sold_by_seth 24 = 78 :=
by
  unfold num_candy_sold_by_seth
  simp
  rfl

end seth_sold_78_candy_bars_l26_26424


namespace original_perimeter_of_rectangle_l26_26645

theorem original_perimeter_of_rectangle
  (a b : ℝ)
  (h : (a + 3) * (b + 3) - a * b = 90) :
  2 * (a + b) = 54 :=
sorry

end original_perimeter_of_rectangle_l26_26645


namespace min_x_plus_y_l26_26681

theorem min_x_plus_y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 1 / x + 4 / y = 1) : x + y ≥ 9 :=
sorry

end min_x_plus_y_l26_26681


namespace dandelions_survive_to_flower_l26_26206

def seeds_initial : ℕ := 300
def seeds_in_water : ℕ := seeds_initial / 3
def seeds_eaten_by_insects : ℕ := seeds_initial / 6
def seeds_remaining : ℕ := seeds_initial - seeds_in_water - seeds_eaten_by_insects
def seeds_to_flower : ℕ := seeds_remaining / 2

theorem dandelions_survive_to_flower : seeds_to_flower = 75 := by
  sorry

end dandelions_survive_to_flower_l26_26206


namespace solution_to_problem_l26_26657

def number_exists (n : ℝ) : Prop :=
  n / 0.25 = 400

theorem solution_to_problem : ∃ n : ℝ, number_exists n ∧ n = 100 := by
  sorry

end solution_to_problem_l26_26657


namespace missy_total_watching_time_l26_26257

def num_reality_shows := 5
def length_reality_show := 28
def num_cartoons := 1
def length_cartoon := 10

theorem missy_total_watching_time : 
  (num_reality_shows * length_reality_show + num_cartoons * length_cartoon) = 150 := 
by 
  sorry

end missy_total_watching_time_l26_26257


namespace cubics_product_l26_26936

theorem cubics_product :
  (∏ n in [3, 4, 5, 6, 7], (n^3 - 1) / (n^3 + 1)) = (57 / 168) := by
  sorry

end cubics_product_l26_26936


namespace paul_needs_score_to_achieve_mean_l26_26261

theorem paul_needs_score_to_achieve_mean (x : ℤ) :
  (78 + 84 + 76 + 82 + 88 + x) / 6 = 85 → x = 102 :=
by 
  sorry

end paul_needs_score_to_achieve_mean_l26_26261


namespace wire_length_before_cut_l26_26478

theorem wire_length_before_cut (S : ℝ) (L : ℝ) (h1 : S = 4) (h2 : S = (2/5) * L) : S + L = 14 :=
by 
  sorry

end wire_length_before_cut_l26_26478


namespace vanya_more_heads_probability_l26_26461

-- Define binomial distributions for Vanya and Tanya's coin flips
def binomial (n : ℕ) (p : ℝ) : distribution :=
  λ k : ℕ, choose n k * (p ^ k) * ((1 - p) ^ (n - k))

-- Define the random variables X_V and X_T
def X_V (n : ℕ) : distribution := binomial (n + 1) (1 / 2)
def X_T (n : ℕ) : distribution := binomial n (1 / 2)

-- Define the probability of Vanya getting more heads than Tanya
def probability_vanya_more_heads (n : ℕ) : ℝ :=
  ∑ (k : ℕ) in finset.range (n + 2), ∑ (j : ℕ) in finset.range (k), X_V n k * X_T n j

-- The main theorem to prove
theorem vanya_more_heads_probability (n : ℕ) : probability_vanya_more_heads n = 1 / 2 :=
begin
  sorry,
end

end vanya_more_heads_probability_l26_26461


namespace part1_part2_l26_26833

variables {a b c : ℝ}

-- Condition 1: a, b, and c are positive numbers
variables (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
-- Condition 2: a² + b² + 4c² = 3
variables (h4 : a^2 + b^2 + 4*c^2 = 3)

-- Part 1: Prove a + b + 2c ≤ 3
theorem part1 : a + b + 2*c ≤ 3 :=
by
  sorry

-- Condition for Part 2: b = 2c
variables (h5 : b = 2 * c)

-- Part 2: Prove 1/a + 1/c ≥ 3
theorem part2 : 1/a + 1/c ≥ 3 :=
by
  sorry

end part1_part2_l26_26833


namespace product_of_x_y_l26_26307

theorem product_of_x_y (x y : ℝ) (h1 : -3 * x + 4 * y = 28) (h2 : 3 * x - 2 * y = 8) : x * y = 264 :=
by
  sorry

end product_of_x_y_l26_26307


namespace find_x_l26_26549

theorem find_x (x y : ℤ) (hx : x > y) (hy : y > 0) (hxy : x + y + x * y = 80) : x = 26 :=
sorry

end find_x_l26_26549


namespace count_integer_values_l26_26889

theorem count_integer_values (x : ℤ) (h1 : 4 < Real.sqrt (3 * x + 1)) (h2 : Real.sqrt (3 * x + 1) < 5) : 
  (5 < x ∧ x < 8 ∧ ∃ (N : ℕ), N = 2) :=
by sorry

end count_integer_values_l26_26889


namespace actual_distance_traveled_l26_26169

theorem actual_distance_traveled (D : ℕ) (h : (D:ℚ) / 12 = (D + 20) / 16) : D = 60 :=
sorry

end actual_distance_traveled_l26_26169


namespace tangency_splits_segments_l26_26197

def pentagon_lengths (a b c d e : ℕ) (h₁ : a = 1) (h₃ : c = 1) (x1 x2 : ℝ) :=
x1 + x2 = b ∧ x1 = 1/2 ∧ x2 = 1/2

theorem tangency_splits_segments {a b c d e : ℕ} (h₁ : a = 1) (h₃ : c = 1) :
    ∃ x1 x2 : ℝ, pentagon_lengths a b c d e h₁ h₃ x1 x2 :=
    by 
    sorry

end tangency_splits_segments_l26_26197


namespace circle_center_count_l26_26387

noncomputable def num_circle_centers (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) : ℕ :=
  if (c = d) then 4 else 8

-- Here is the theorem statement
theorem circle_center_count (b c d : ℝ) (h₁ : b < c) (h₂ : c ≤ d) :
  num_circle_centers b c d h₁ h₂ = if (c = d) then 4 else 8 :=
sorry

end circle_center_count_l26_26387


namespace necklace_cost_l26_26259

theorem necklace_cost (N : ℕ) (h1 : N + (N + 5) = 73) : N = 34 := by
  sorry

end necklace_cost_l26_26259


namespace algebraic_expression_value_l26_26421

theorem algebraic_expression_value
  (x y : ℚ)
  (h : |2 * x - 3 * y + 1| + (x + 3 * y + 5)^2 = 0) :
  (-2 * x * y)^2 * (-y^2) * 6 * x * y^2 = 192 :=
  sorry

end algebraic_expression_value_l26_26421


namespace fraction_question_l26_26781

theorem fraction_question :
  ((3 / 8 + 5 / 6) / (5 / 12 + 1 / 4) = 29 / 16) :=
by
  -- This is where we will put the proof steps 
  sorry

end fraction_question_l26_26781


namespace projectile_height_reach_l26_26579

theorem projectile_height_reach (t : ℝ) (h : -16 * t^2 + 64 * t = 25) : t = 3.6 :=
by
  sorry

end projectile_height_reach_l26_26579


namespace base_b_representation_l26_26239

theorem base_b_representation (b : ℕ) (h₁ : 1 * b + 5 = n) (h₂ : n^2 = 4 * b^2 + 3 * b + 3) : b = 7 :=
by {
  sorry
}

end base_b_representation_l26_26239


namespace doughnuts_remaining_l26_26061

theorem doughnuts_remaining 
  (total_doughnuts : ℕ)
  (total_staff : ℕ)
  (staff_3_doughnuts : ℕ)
  (doughnuts_eaten_by_3 : ℕ)
  (staff_2_doughnuts : ℕ)
  (doughnuts_eaten_by_2 : ℕ)
  (staff_4_doughnuts : ℕ)
  (doughnuts_eaten_by_4 : ℕ) :
  total_doughnuts = 120 →
  total_staff = 35 →
  staff_3_doughnuts = 15 →
  staff_2_doughnuts = 10 →
  doughnuts_eaten_by_3 = staff_3_doughnuts * 3 →
  doughnuts_eaten_by_2 = staff_2_doughnuts * 2 →
  staff_4_doughnuts = total_staff - (staff_3_doughnuts + staff_2_doughnuts) →
  doughnuts_eaten_by_4 = staff_4_doughnuts * 4 →
  total_doughnuts - (doughnuts_eaten_by_3 + doughnuts_eaten_by_2 + doughnuts_eaten_by_4) = 15 :=
by
  intros
  -- Proof goes here
  sorry

end doughnuts_remaining_l26_26061


namespace fraction_of_repeating_decimal_l26_26216

theorem fraction_of_repeating_decimal :
  ∃ (f : ℚ), f = 0.73 ∧ f = 73 / 99 := by
  sorry

end fraction_of_repeating_decimal_l26_26216


namespace ellipse_equation_l26_26382

theorem ellipse_equation (a b c : ℝ) :
  (2 * a = 10) ∧ (c / a = 4 / 5) →
  ((x:ℝ)^2 / 25 + (y:ℝ)^2 / 9 = 1) ∨ ((x:ℝ)^2 / 9 + (y:ℝ)^2 / 25 = 1) :=
by
  sorry

end ellipse_equation_l26_26382


namespace minimize_cost_l26_26919

-- Define the unit prices of the soccer balls.
def price_A := 50
def price_B := 80

-- Define the condition for the total number of balls and cost function.
def total_balls := 80
def cost (a : ℕ) : ℕ := price_A * a + price_B * (total_balls - a)
def valid_a (a : ℕ) : Prop := 30 ≤ a ∧ a ≤ (3 * (total_balls - a))

-- Prove the number of brand A soccer balls to minimize the total cost.
theorem minimize_cost : ∃ a : ℕ, valid_a a ∧ ∀ b : ℕ, valid_a b → cost a ≤ cost b :=
sorry

end minimize_cost_l26_26919


namespace poly_coeff_sum_l26_26972

theorem poly_coeff_sum (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) (x : ℝ) :
  (2 * x + 3)^8 = a_0 + a_1 * (x + 1) + a_2 * (x + 1)^2 + 
                 a_3 * (x + 1)^3 + a_4 * (x + 1)^4 + 
                 a_5 * (x + 1)^5 + a_6 * (x + 1)^6 + 
                 a_7 * (x + 1)^7 + a_8 * (x + 1)^8 →
  a_0 + a_2 + a_4 + a_6 + a_8 = 3281 :=
by
  sorry

end poly_coeff_sum_l26_26972


namespace seating_arrangements_l26_26875

theorem seating_arrangements (sons daughters : ℕ) (totalSeats : ℕ) (h_sons : sons = 5) (h_daughters : daughters = 4) (h_seats : totalSeats = 9) :
  let total_arrangements := totalSeats.factorial
  let unwanted_arrangements := sons.factorial * daughters.factorial
  total_arrangements - unwanted_arrangements = 360000 :=
by
  rw [h_sons, h_daughters, h_seats]
  let total_arrangements := 9.factorial
  let unwanted_arrangements := 5.factorial * 4.factorial
  exact Nat.sub_eq_of_eq_add $ eq_comm.mpr (Nat.add_sub_eq_of_eq total_arrangements_units)
where
  total_arrangements_units : 9.factorial = 5.factorial * 4.factorial + 360000 := by
    rw [Nat.factorial, Nat.factorial, Nat.factorial, ←Nat.factorial_mul_factorial_eq 5 4]
    simp [tmp_rewriting]

end seating_arrangements_l26_26875


namespace number_of_lattice_points_in_triangle_l26_26266

theorem number_of_lattice_points_in_triangle (N L S : ℕ) (A B O : (ℕ × ℕ)) :
  (A = (0, 30)) →
  (B = (20, 10)) →
  (O = (0, 0)) →
  (S = 300) →
  (L = 60) →
  S = N + L / 2 - 1 →
  N = 271 :=
by
  intros hA hB hO hS hL hPick
  sorry

end number_of_lattice_points_in_triangle_l26_26266


namespace determine_parallel_planes_l26_26174

def Plane : Type := sorry
def Line : Type := sorry
def Parallel (x y : Line) : Prop := sorry
def Skew (x y : Line) : Prop := sorry
def PlaneParallel (α β : Plane) : Prop := sorry

variables (α β : Plane) (a b : Line)
variable (hSkew : Skew a b)
variable (hαa : Parallel a α) 
variable (hαb : Parallel b α)
variable (hβa : Parallel a β)
variable (hβb : Parallel b β)

theorem determine_parallel_planes : PlaneParallel α β := sorry

end determine_parallel_planes_l26_26174


namespace union_eq_l26_26097

def setA : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

theorem union_eq : setA ∪ setB = { x : ℝ | -1 ≤ x ∧ x ≤ 5 } :=
by
  sorry

end union_eq_l26_26097


namespace proof_problem_l26_26854

noncomputable def find_values (a b c x y z : ℝ) := 
  14 * x + b * y + c * z = 0 ∧ 
  a * x + 24 * y + c * z = 0 ∧ 
  a * x + b * y + 43 * z = 0 ∧ 
  a ≠ 14 ∧ b ≠ 24 ∧ c ≠ 43 ∧ x ≠ 0

theorem proof_problem (a b c x y z : ℝ) 
  (h : find_values a b c x y z):
  (a / (a - 14)) + (b / (b - 24)) + (c / (c - 43)) = 1 :=
by
  sorry

end proof_problem_l26_26854


namespace remainder_of_sum_div_11_is_9_l26_26360

def seven_times_ten_pow_twenty : ℕ := 7 * 10 ^ 20
def two_pow_twenty : ℕ := 2 ^ 20
def sum : ℕ := seven_times_ten_pow_twenty + two_pow_twenty

theorem remainder_of_sum_div_11_is_9 : sum % 11 = 9 := by
  sorry

end remainder_of_sum_div_11_is_9_l26_26360


namespace correct_statement_l26_26927

-- Definitions
def certain_event (P : ℝ → Prop) : Prop := P 1
def impossible_event (P : ℝ → Prop) : Prop := P 0
def uncertain_event (P : ℝ → Prop) : Prop := ∀ p, 0 < p ∧ p < 1 → P p

-- Theorem to prove
theorem correct_statement (P : ℝ → Prop) :
  (certain_event P ∧ impossible_event P ∧ uncertain_event P) →
  (∀ p, P p → p = 1)
:= by
  sorry

end correct_statement_l26_26927


namespace problem1_problem2_l26_26175

theorem problem1 :
  (27 : ℝ)^(2/3) - 2^(Real.log2 3) * Real.log2 (1/8) + Real.log2 3 * (Real.logb 3 4) = 20 :=
by
  sorry 

theorem problem2 (α : ℝ) :
  (sin (α - π/2) * cos (3*π/2 + α) * tan (π - α)) / (tan (-α - π) * sin (-α - π)) = -cos α :=
by
  sorry

end problem1_problem2_l26_26175


namespace true_proposition_B_l26_26903

theorem true_proposition_B : (3 > 4) ∨ (3 < 4) :=
sorry

end true_proposition_B_l26_26903


namespace johnson_family_seating_l26_26881

theorem johnson_family_seating (boys girls : Finset ℕ) (h_boys : boys.card = 5) (h_girls : girls.card = 4) :
  (∃ (arrangement : List ℕ), arrangement.length = 9 ∧ at_least_two_adjacent boys arrangement) :=
begin
  -- Given the total number of ways: 9! 
  -- subtract 5! * 4! from 9! to get the result 
  have total_arrangements := nat.factorial 9,
  have restrictive_arrangements := nat.factorial 5 * nat.factorial 4,
  exact (total_arrangements - restrictive_arrangements) = 360000,
end

end johnson_family_seating_l26_26881


namespace first_grade_enrollment_l26_26542

theorem first_grade_enrollment (a : ℕ) (R : ℕ) (L : ℕ) (h1 : 200 ≤ a) (h2 : a ≤ 300)
  (h3 : a = 25 * R + 10) (h4 : a = 30 * L - 15) : a = 285 :=
by
  sorry

end first_grade_enrollment_l26_26542


namespace max_value_of_f_l26_26661

noncomputable def f (x : ℝ) : ℝ := 3 * Real.cos (2 * x)

theorem max_value_of_f :
  ∃ x : ℝ, ∃ k : ℤ, f x = 3 ∧ x = k * Real.pi :=
by
  -- The proof is omitted
  sorry

end max_value_of_f_l26_26661


namespace lengths_C_can_form_triangle_l26_26304

-- Definition of sets of lengths
def lengths_A := (3, 6, 9)
def lengths_B := (3, 5, 9)
def lengths_C := (4, 6, 9)
def lengths_D := (2, 6, 4)

-- Triangle condition for a given set of lengths
def can_form_triangle (a b c : Nat) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Proof problem statement 
theorem lengths_C_can_form_triangle : can_form_triangle 4 6 9 :=
by
  sorry

end lengths_C_can_form_triangle_l26_26304
