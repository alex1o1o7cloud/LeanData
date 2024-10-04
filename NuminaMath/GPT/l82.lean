import Mathlib

namespace arithmetic_sequence_middle_term_l82_82127

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l82_82127


namespace smallest_four_digit_divisible_by_25_l82_82268

theorem smallest_four_digit_divisible_by_25 : ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 25 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 25 = 0 → n ≤ m := by
  -- Prove that the smallest four-digit number divisible by 25 is 1000
  sorry

end smallest_four_digit_divisible_by_25_l82_82268


namespace gcd_75_100_l82_82905

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l82_82905


namespace tan_15_degree_identity_l82_82142

theorem tan_15_degree_identity : (1 + Real.tan (15 * Real.pi / 180)) / (1 - Real.tan (15 * Real.pi / 180)) = Real.sqrt 3 :=
by sorry

end tan_15_degree_identity_l82_82142


namespace polygon_sides_l82_82046

-- Definitions of the conditions
def is_regular_polygon (n : ℕ) (int_angle ext_angle : ℝ) : Prop :=
  int_angle = 5 * ext_angle ∧ (int_angle + ext_angle = 180)

-- Main theorem statement
theorem polygon_sides (n : ℕ) (int_angle ext_angle : ℝ) :
  is_regular_polygon n int_angle ext_angle →
  (ext_angle = 360 / n) →
  n = 12 :=
sorry

end polygon_sides_l82_82046


namespace roy_older_than_julia_l82_82459

variable {R J K x : ℝ}

theorem roy_older_than_julia (h1 : R = J + x)
                            (h2 : R = K + x / 2)
                            (h3 : R + 2 = 2 * (J + 2))
                            (h4 : (R + 2) * (K + 2) = 192) :
                            x = 2 :=
by
  sorry

end roy_older_than_julia_l82_82459


namespace sum_geometric_sequence_l82_82101

theorem sum_geometric_sequence {n : ℕ} (S : ℕ → ℝ) (h1 : S n = 10) (h2 : S (2 * n) = 30) : 
  S (3 * n) = 70 := 
by 
  sorry

end sum_geometric_sequence_l82_82101


namespace common_chord_of_circles_l82_82836

theorem common_chord_of_circles : 
  ∀ (x y : ℝ), 
  (x^2 + y^2 + 2*x = 0 ∧ x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := 
by 
  sorry

end common_chord_of_circles_l82_82836


namespace remainder_product_l82_82865

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82865


namespace solve_system_of_equations_l82_82087

theorem solve_system_of_equations :
  ∃ (x y z : ℝ),
    (x^2 + y^2 + 8 * x - 6 * y = -20) ∧
    (x^2 + z^2 + 8 * x + 4 * z = -10) ∧
    (y^2 + z^2 - 6 * y + 4 * z = 0) ∧
    ((x = -3 ∧ y = 1 ∧ z = 1) ∨
     (x = -3 ∧ y = 1 ∧ z = -5) ∨
     (x = -3 ∧ y = 5 ∧ z = 1) ∨
     (x = -3 ∧ y = 5 ∧ z = -5) ∨
     (x = -5 ∧ y = 1 ∧ z = 1) ∨
     (x = -5 ∧ y = 1 ∧ z = -5) ∨
     (x = -5 ∧ y = 5 ∧ z = 1) ∨
     (x = -5 ∧ y = 5 ∧ z = -5)) :=
sorry

end solve_system_of_equations_l82_82087


namespace madeline_water_intake_l82_82239

-- Declare necessary data and conditions
def bottle_A : ℕ := 8
def bottle_B : ℕ := 12
def bottle_C : ℕ := 16

def goal_yoga : ℕ := 15
def goal_work : ℕ := 35
def goal_jog : ℕ := 20
def goal_evening : ℕ := 30

def intake_yoga : ℕ := 2 * bottle_A
def intake_work : ℕ := 3 * bottle_B
def intake_jog : ℕ := 2 * bottle_C
def intake_evening : ℕ := 2 * bottle_A + 2 * bottle_C

def total_intake : ℕ := intake_yoga + intake_work + intake_jog + intake_evening
def goal_total : ℕ := 100

-- Statement of the proof problem
theorem madeline_water_intake : total_intake = 132 ∧ total_intake - goal_total = 32 :=
by
  -- Calculation parts go here (not needed per instruction)
  sorry

end madeline_water_intake_l82_82239


namespace convert_speed_l82_82004

-- Definitions based on the given condition
def kmh_to_mps (kmh : ℝ) : ℝ := kmh * 0.277778

-- Theorem statement
theorem convert_speed : kmh_to_mps 84 = 23.33 :=
by
  -- Proof omitted
  sorry

end convert_speed_l82_82004


namespace ex1_simplified_ex2_simplified_l82_82651

-- Definitions and problem setup
def ex1 (a : ℝ) : ℝ := ((-a^3)^2 * a^3 - 4 * a^2 * a^7)
def ex2 (a : ℝ) : ℝ := (2 * a + 1) * (-2 * a + 1)

-- Proof goals
theorem ex1_simplified (a : ℝ) : ex1 a = -3 * a^9 :=
by sorry

theorem ex2_simplified (a : ℝ) : ex2 a = 4 * a^2 - 1 :=
by sorry

end ex1_simplified_ex2_simplified_l82_82651


namespace simplify_rationalize_expr_l82_82827

theorem simplify_rationalize_expr : 
  (1 / (2 + 1 / (Real.sqrt 5 - 2))) = (4 - Real.sqrt 5) / 11 := 
by 
  sorry

end simplify_rationalize_expr_l82_82827


namespace vartan_spent_on_recreation_last_week_l82_82232

variable (W P : ℝ)
variable (h1 : P = 0.20)
variable (h2 : W > 0)

theorem vartan_spent_on_recreation_last_week :
  (P * W) = 0.20 * W :=
by
  sorry

end vartan_spent_on_recreation_last_week_l82_82232


namespace sum_of_zeros_l82_82428

-- Defining the conditions and the result
theorem sum_of_zeros (f : ℝ → ℝ) (h : ∀ x, f (1 - x) = f (3 + x)) (a b c : ℝ)
  (h1 : f a = 0) (h2 : f b = 0) (h3 : f c = 0) : 
  a + b + c = 3 := 
by 
  sorry

end sum_of_zeros_l82_82428


namespace sum_of_coordinates_A_l82_82786

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l82_82786


namespace cos_330_eq_sqrt_3_div_2_l82_82655

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l82_82655


namespace max_A_min_A_l82_82637

-- Define the problem and its conditions and question

def A_max (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

def A_min (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) : ℕ :=
  let b := B % 10
  10^8 * b + (B - b) / 10

theorem max_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_max B h1 h2 h3 = 999999998 := sorry

theorem min_A (B : ℕ) (h1 : B > 22222222) (h2 : (B / 100000000 : ℕ) ≠ 0) (h3 : Nat.gcd B 18 = 1) :
  A_min B h1 h2 h3 = 122222224 := sorry

end max_A_min_A_l82_82637


namespace area_PQR_is_4_5_l82_82108

noncomputable def point := (ℝ × ℝ)

def P : point := (2, 1)
def Q : point := (1, 4)
def R_line (x: ℝ) : point := (x, 6 - x)

def area_triangle (A B C : point) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

theorem area_PQR_is_4_5 (x : ℝ) (h : R_line x ∈ {p : point | p.1 + p.2 = 6}) : 
  area_triangle P Q (R_line x) = 4.5 :=
    sorry

end area_PQR_is_4_5_l82_82108


namespace part1_part2_l82_82543

def set_A := {x : ℝ | x^2 + 2*x - 8 = 0}
def set_B (a : ℝ) := {x : ℝ | x^2 + 2*(a+1)*x + 2*a^2 - 2 = 0}

theorem part1 (a : ℝ) (h : a = 1) : 
  (set_A ∩ set_B a) = {-4} := by
  sorry

theorem part2 (a : ℝ) : 
  (set_A ∩ (set_B a) = set_B a) → (a < -1 ∨ a > 3) := by
  sorry

end part1_part2_l82_82543


namespace solve_equation_l82_82464

theorem solve_equation : ∀ x y : ℤ, x^2 + y^2 = 3 * x * y → x = 0 ∧ y = 0 := by
  intros x y h
  sorry

end solve_equation_l82_82464


namespace number_of_ways_to_select_courses_l82_82151

theorem number_of_ways_to_select_courses :
  let typeA := 3
  let typeB := 4
  ∃ k, (k = (nat.choose typeA 2) * (nat.choose typeB 1) + (nat.choose typeA 1) * (nat.choose typeB 2) + (nat.choose typeA 1) * (nat.choose typeB 1)) ∧ k = 42 :=
begin
  sorry
end

end number_of_ways_to_select_courses_l82_82151


namespace total_cookies_l82_82604

theorem total_cookies (num_people : ℕ) (cookies_per_person : ℕ) (total_cookies : ℕ) 
  (h1: num_people = 4) (h2: cookies_per_person = 22) : total_cookies = 88 :=
by
  sorry

end total_cookies_l82_82604


namespace eccentricity_of_ellipse_l82_82451

open Real

def ellipse_eq (a b x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def foci_dist_eq (a c : ℝ) : Prop :=
  2 * c / (2 * a) = sqrt 6 / 2

noncomputable def eccentricity (c a : ℝ) : ℝ :=
  c / a

theorem eccentricity_of_ellipse (a b x y c : ℝ)
  (h1 : ellipse_eq a b x y)
  (h2 : foci_dist_eq a c) :
  eccentricity c a = sqrt 6 / 3 :=
sorry

end eccentricity_of_ellipse_l82_82451


namespace base_256_6_digits_l82_82279

theorem base_256_6_digits (b : ℕ) (h1 : b ^ 5 ≤ 256) (h2 : 256 < b ^ 6) : b = 3 := 
sorry

end base_256_6_digits_l82_82279


namespace fewest_cookies_by_ben_l82_82979

noncomputable def cookie_problem : Prop :=
  let ana_area := 4 * Real.pi
  let ben_area := 9
  let carol_area := Real.sqrt (5 * (5 + 2 * Real.sqrt 5))
  let dave_area := 3.375 * Real.sqrt 3
  let dough := ana_area * 10
  let ana_cookies := dough / ana_area
  let ben_cookies := dough / ben_area
  let carol_cookies := dough / carol_area
  let dave_cookies := dough / dave_area
  ben_cookies < ana_cookies ∧ ben_cookies < carol_cookies ∧ ben_cookies < dave_cookies

theorem fewest_cookies_by_ben : cookie_problem := by
  sorry

end fewest_cookies_by_ben_l82_82979


namespace cos_330_eq_sqrt3_div_2_l82_82684

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l82_82684


namespace impossible_load_two_coins_l82_82928

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l82_82928


namespace part1_part2_part3_l82_82598

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

-- 1. Prove that B ⊆ A implies m ≤ 3
theorem part1 (m : ℝ) : B m ⊆ A → m ≤ 3 := sorry

-- 2. Prove that the number of non-empty proper subsets of A when x ∈ ℤ is 254
theorem part2 : ∀ x ∈ (Set.univ : Set ℤ), x ∈ {-2, -1, 0, 1, 2, 3, 4, 5} → ∃! n, n = 254 := sorry

-- 3. Prove that A ∩ B = ∅ implies m < 2 or m > 4
theorem part3 (m : ℝ) : A ∩ B m = ∅ → m < 2 ∨ m > 4 := sorry

end part1_part2_part3_l82_82598


namespace three_digit_integers_with_odd_factors_l82_82364

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l82_82364


namespace number_of_three_digit_integers_with_odd_factors_l82_82374

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l82_82374


namespace no_such_n_exists_l82_82968

theorem no_such_n_exists : ∀ (n : ℕ), n ≥ 1 → ¬ Prime (n^n - 4 * n + 3) :=
by
  intro n hn
  sorry

end no_such_n_exists_l82_82968


namespace repeating_decimal_calculation_l82_82168

theorem repeating_decimal_calculation :
  2 * (8 / 9 - 2 / 9 + 4 / 9) = 20 / 9 :=
by
  -- sorry proof will be inserted here.
  sorry

end repeating_decimal_calculation_l82_82168


namespace circumscribed_sphere_eqn_l82_82173

-- Define vertices of the tetrahedron
variables {A_1 A_2 A_3 A_4 : Point}

-- Define barycentric coordinates
variables {x_1 x_2 x_3 x_4 : ℝ}

-- Define edge lengths
variables {a_12 a_13 a_14 a_23 a_24 a_34: ℝ}

-- Define the equation of the circumscribed sphere in barycentric coordinates
theorem circumscribed_sphere_eqn (h1 : A_1 ≠ A_2) (h2 : A_1 ≠ A_3) (h3 : A_1 ≠ A_4)
                                 (h4 : A_2 ≠ A_3) (h5 : A_2 ≠ A_4) (h6 : A_3 ≠ A_4) :
    (x_1 * x_2 * a_12^2 + x_1 * x_3 * a_13^2 + x_1 * x_4 * a_14^2 +
     x_2 * x_3 * a_23^2 + x_2 * x_4 * a_24^2 + x_3 * x_4 * a_34^2) = 0 :=
 sorry

end circumscribed_sphere_eqn_l82_82173


namespace product_sets_not_identical_l82_82891

theorem product_sets_not_identical :
  ∀ (M : array (array nat 10) 10),
  (∀ i j, 101 ≤ M[i][j] ∧ M[i][j] ≤ 200) →
  let row_products := array.map (array.foldr (*) 1) M in
  let col_products := array.map (array.foldr (*) 1) (array.transpose M) in
  row_products ≠ col_products :=
by
  sorry

end product_sets_not_identical_l82_82891


namespace incorrect_statements_l82_82628

-- Define basic properties for lines and their equations.
def point_slope_form (y y1 x x1 k : ℝ) : Prop := (y - y1) = k * (x - x1)
def intercept_form (x y a b : ℝ) : Prop := x / a + y / b = 1
def distance_to_origin_on_y_axis (k b : ℝ) : ℝ := abs b
def slope_intercept_form (y m x c : ℝ) : Prop := y = m * x + c

-- The conditions specified in the problem.
variables (A B C D : Prop)
  (hA : A ↔ ∀ (y y1 x x1 k : ℝ), ¬point_slope_form y y1 x x1 k)
  (hB : B ↔ ∀ (x y a b : ℝ), intercept_form x y a b)
  (hC : C ↔ ∀ (k b : ℝ), distance_to_origin_on_y_axis k b = abs b)
  (hD : D ↔ ∀ (y m x c : ℝ), slope_intercept_form y m x c)

theorem incorrect_statements : ¬ B ∧ ¬ C ∧ ¬ D :=
by
  -- Intermediate steps would be to show each statement B, C, and D are false.
  sorry

end incorrect_statements_l82_82628


namespace five_lattice_points_l82_82330

theorem five_lattice_points (p : Fin 5 → ℤ × ℤ) :
  ∃ i j : Fin 5, i ≠ j ∧ ((p i).1 + (p j).1) % 2 = 0 ∧ ((p i).2 + (p j).2) % 2 = 0 := by
  sorry

end five_lattice_points_l82_82330


namespace michael_ratio_zero_l82_82951

theorem michael_ratio_zero (M : ℕ) (h1: M ≤ 60) (h2: 15 = (60 - M) / 2 - 15) : M = 0 := by
  sorry 

end michael_ratio_zero_l82_82951


namespace range_of_a_l82_82540

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 - 4*x + a

theorem range_of_a 
  (f : ℝ → ℝ → ℝ)
  (h : ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → f x a ≥ 0) : 
  3 ≤ a :=
sorry

end range_of_a_l82_82540


namespace intersection_point_of_lines_l82_82005

theorem intersection_point_of_lines :
  ∃ x y : ℚ, 
    (y = -3 * x + 4) ∧ 
    (y = (1 / 3) * x + 1) ∧ 
    x = 9 / 10 ∧ 
    y = 13 / 10 :=
by sorry

end intersection_point_of_lines_l82_82005


namespace avg_of_nine_numbers_l82_82212

theorem avg_of_nine_numbers (average : ℝ) (sum : ℝ) (h : average = (sum / 9)) (h_avg : average = 5.3) : sum = 47.7 := by
  sorry

end avg_of_nine_numbers_l82_82212


namespace system_solution_and_range_l82_82344

theorem system_solution_and_range (a x y : ℝ) (h1 : 2 * x + y = 5 * a) (h2 : x - 3 * y = -a + 7) :
  (x = 2 * a + 1 ∧ y = a - 2) ∧ (-1/2 ≤ a ∧ a < 2 → 2 * a + 1 ≥ 0 ∧ a - 2 < 0) :=
by
  sorry

end system_solution_and_range_l82_82344


namespace greatest_mean_YZ_l82_82107

noncomputable def X_mean := 60
noncomputable def Y_mean := 70
noncomputable def XY_mean := 64
noncomputable def XZ_mean := 66

theorem greatest_mean_YZ (Xn Yn Zn : ℕ) (m : ℕ) :
  (60 * Xn + 70 * Yn) / (Xn + Yn) = 64 →
  (60 * Xn + m) / (Xn + Zn) = 66 →
  ∃ (k : ℕ), k = 69 :=
by
  intro h1 h2
  -- Sorry is used to skip the proof
  sorry

end greatest_mean_YZ_l82_82107


namespace correct_equation_after_moving_digit_l82_82050

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end correct_equation_after_moving_digit_l82_82050


namespace opponent_choice_is_random_l82_82791

-- Define the possible outcomes in the game
inductive Outcome
| rock
| paper
| scissors

-- Defining the opponent's choice set
def opponent_choice := {outcome : Outcome | outcome = Outcome.rock ∨ outcome = Outcome.paper ∨ outcome = Outcome.scissors}

-- The event where the opponent chooses "scissors"
def event_opponent_chooses_scissors := Outcome.scissors ∈ opponent_choice

-- Proving that the event of opponent choosing "scissors" is a random event
theorem opponent_choice_is_random : ¬(∀outcome ∈ opponent_choice, outcome = Outcome.scissors) ∧ (∃ outcome ∈ opponent_choice, outcome = Outcome.scissors) → event_opponent_chooses_scissors := 
sorry

end opponent_choice_is_random_l82_82791


namespace correct_operation_l82_82275

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end correct_operation_l82_82275


namespace apples_handed_out_to_students_l82_82094

def initial_apples : ℕ := 47
def apples_per_pie : ℕ := 4
def number_of_pies : ℕ := 5
def apples_for_pies : ℕ := number_of_pies * apples_per_pie

theorem apples_handed_out_to_students : 
  initial_apples - apples_for_pies = 27 := 
by
  -- Since 20 apples are used for pies and there were initially 47 apples,
  -- it follows that 27 apples were handed out to students.
  sorry

end apples_handed_out_to_students_l82_82094


namespace three_digit_integers_with_odd_number_of_factors_count_l82_82398

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l82_82398


namespace total_pencils_is_54_l82_82818

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end total_pencils_is_54_l82_82818


namespace set_intersection_complement_l82_82032

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -1 ∨ x > 4}
def complement_B : Set ℝ := U \ B
def expected_set : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

theorem set_intersection_complement :
  A ∩ complement_B = expected_set :=
by
  sorry

end set_intersection_complement_l82_82032


namespace correct_equation_after_moving_digit_l82_82051

theorem correct_equation_after_moving_digit :
  (101 - 102 = 1) →
  101 - 10^2 = 1 :=
by
  intro h
  sorry

end correct_equation_after_moving_digit_l82_82051


namespace total_rainfall_2004_l82_82433

theorem total_rainfall_2004 (avg_2003 : ℝ) (increment : ℝ) (months : ℕ) (total_2004 : ℝ) 
  (h1 : avg_2003 = 41.5) 
  (h2 : increment = 2) 
  (h3 : months = 12) 
  (h4 : total_2004 = avg_2003 + increment * months) :
  total_2004 = 522 :=
by 
  sorry

end total_rainfall_2004_l82_82433


namespace required_hours_for_fifth_week_l82_82263

def typical_hours_needed (week1 week2 week3 week4 week5 add_hours total_weeks target_avg : ℕ) : ℕ :=
  if (week1 + week2 + week3 + week4 + week5 + add_hours) / total_weeks = target_avg then 
    week5 
  else 
    0

theorem required_hours_for_fifth_week :
  typical_hours_needed 10 14 11 9 x 1 5 12 = 15 :=
by
  sorry

end required_hours_for_fifth_week_l82_82263


namespace total_money_spent_l82_82230

/-- 
John buys a gaming PC for $1200.
He decides to replace the video card in it.
He sells the old card for $300 and buys a new one for $500.
Prove total money spent on the computer after counting the savings from selling the old card is $1400.
-/
theorem total_money_spent (initial_cost : ℕ) (sale_price_old_card : ℕ) (price_new_card : ℕ) : 
  (initial_cost = 1200) → (sale_price_old_card = 300) → (price_new_card = 500) → 
  (initial_cost + (price_new_card - sale_price_old_card) = 1400) :=
by 
  intros
  sorry

end total_money_spent_l82_82230


namespace gcd_75_100_l82_82906

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l82_82906


namespace find_point_B_l82_82333

structure Point where
  x : ℝ
  y : ℝ

def vec_scalar_mult (c : ℝ) (v : Point) : Point :=
  ⟨c * v.x, c * v.y⟩

def vec_add (p : Point) (v : Point) : Point :=
  ⟨p.x + v.x, p.y + v.y⟩

theorem find_point_B :
  let A := Point.mk 1 (-3)
  let a := Point.mk 3 4
  let B := vec_add A (vec_scalar_mult 2 a)
  B = Point.mk 7 5 :=
by {
  sorry
}

end find_point_B_l82_82333


namespace solve_for_x_l82_82424

theorem solve_for_x (x : ℝ) (h : 9 / x^2 = x / 81) : x = 9 := 
  sorry

end solve_for_x_l82_82424


namespace gcd_75_100_l82_82907

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l82_82907


namespace find_b_perpendicular_lines_l82_82471

theorem find_b_perpendicular_lines :
  ∀ (b : ℝ), (∀ x y : ℝ, 2 * x - 3 * y + 6 = 0 ∧ b * x - 3 * y + 6 = 0 →
      (2 / 3) * (b / 3) = -1) → b = -9 / 2 :=
sorry

end find_b_perpendicular_lines_l82_82471


namespace right_angle_vertex_trajectory_l82_82023

theorem right_angle_vertex_trajectory (x y : ℝ) :
  let M := (-2, 0)
  let N := (2, 0)
  let P := (x, y)
  (∃ (x y : ℝ), (x + 2)^2 + y^2 + (x - 2)^2 + y^2 = 16) →
  x ≠ 2 ∧ x ≠ -2 →
  x^2 + y^2 = 4 :=
by
  intro h₁ h₂
  sorry

end right_angle_vertex_trajectory_l82_82023


namespace element_in_set_l82_82799

open Set

theorem element_in_set : -7 ∈ ({1, -7} : Set ℤ) := by
  sorry

end element_in_set_l82_82799


namespace tree_count_in_yard_l82_82567

-- Definitions from conditions
def yard_length : ℕ := 350
def tree_distance : ℕ := 14

-- Statement of the theorem
theorem tree_count_in_yard : (yard_length / tree_distance) + 1 = 26 := by
  sorry

end tree_count_in_yard_l82_82567


namespace greatest_value_of_a_greatest_value_of_a_achieved_l82_82242

theorem greatest_value_of_a (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120) : a ≤ 20 :=
sorry

theorem greatest_value_of_a_achieved (a b : ℕ) (h1 : 5 * Nat.lcm a b + 2 * Nat.gcd a b = 120)
  (h2 : Nat.gcd a b = 10) (h3 : 10 ∣ a ∧ 10 ∣ b) (h4 : Nat.lcm a b = 20) : a = 20 :=
sorry

end greatest_value_of_a_greatest_value_of_a_achieved_l82_82242


namespace Luke_mowing_lawns_l82_82590

theorem Luke_mowing_lawns (L : ℕ) (h1 : 18 + L = 27) : L = 9 :=
by
  sorry

end Luke_mowing_lawns_l82_82590


namespace product_mod_7_l82_82846

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82846


namespace gcd_72_120_180_is_12_l82_82837

theorem gcd_72_120_180_is_12 : Int.gcd (Int.gcd 72 120) 180 = 12 := by
  sorry

end gcd_72_120_180_is_12_l82_82837


namespace four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l82_82969

noncomputable def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

noncomputable def is_divisible_by_11 (n : ℕ) : Prop := n % 11 = 0

noncomputable def sum_of_digits_is_11 (n : ℕ) : Prop := 
  let d1 := n / 1000
  let r1 := n % 1000
  let d2 := r1 / 100
  let r2 := r1 % 100
  let d3 := r2 / 10
  let d4 := r2 % 10
  d1 + d2 + d3 + d4 = 11

theorem four_digit_numbers_divisible_by_11_with_sum_of_digits_11
  (n : ℕ) 
  (h1 : is_four_digit n)
  (h2 : is_divisible_by_11 n)
  (h3 : sum_of_digits_is_11 n) : 
  n = 2090 ∨ n = 3080 ∨ n = 4070 ∨ n = 5060 ∨ n = 6050 ∨ n = 7040 ∨ n = 8030 ∨ n = 9020 :=
sorry

end four_digit_numbers_divisible_by_11_with_sum_of_digits_11_l82_82969


namespace train_length_calculation_l82_82510

theorem train_length_calculation (speed_kmph : ℝ) (time_seconds : ℝ) (platform_length_m : ℝ) (train_length_m: ℝ) : speed_kmph = 45 → time_seconds = 51.99999999999999 → platform_length_m = 290 → train_length_m = 360 :=
by
  sorry

end train_length_calculation_l82_82510


namespace positive_integers_mod_l82_82033

theorem positive_integers_mod (n : ℕ) (h : n > 0) :
  ∃! (x : ℕ), x < 10^n ∧ x^2 % 10^n = x % 10^n :=
sorry

end positive_integers_mod_l82_82033


namespace lottery_tickets_equal_chance_l82_82477

-- Definition: Each lottery ticket has a 0.1% (0.001) chance of winning.
-- Definition: Each lottery ticket's outcome is independent of the others.

theorem lottery_tickets_equal_chance :
  let p := 0.001 in
  ∀ (n : ℕ) (tickets : Fin n → Prop),
    (∀ i, Prob (tickets i) = p) →
    (∀ i j, i ≠ j → Indep (tickets i) (tickets j)) →
    ∀ i, Prob (tickets i) = p :=
by
  intros p n tickets hprob hindep i
  exact hprob i
  sorry

end lottery_tickets_equal_chance_l82_82477


namespace solve_equation_l82_82831

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end solve_equation_l82_82831


namespace train_distance_covered_l82_82640

-- Definitions based on the given conditions
def average_speed := 3   -- in meters per second
def total_time := 9      -- in seconds

-- Theorem statement: Given the average speed and total time, the total distance covered is 27 meters
theorem train_distance_covered : average_speed * total_time = 27 := 
by
  sorry

end train_distance_covered_l82_82640


namespace three_digit_cubes_divisible_by_16_l82_82558

theorem three_digit_cubes_divisible_by_16 :
  (count (λ n : ℕ, 4 * n = n ∧ (100 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 999)) {n | 1 ≤ n ∧ n ≤ 2}) = 1 :=
sorry

end three_digit_cubes_divisible_by_16_l82_82558


namespace sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l82_82458

noncomputable def sum_series_a : ℝ :=
∑' n, (1 / (n * (n + 1)))

noncomputable def sum_series_b : ℝ :=
∑' n, (1 / ((n + 1) * (n + 2)))

noncomputable def sum_series_c : ℝ :=
∑' n, (1 / ((n + 2) * (n + 3)))

theorem sum_series_a_eq_one : sum_series_a = 1 := sorry

theorem sum_series_b_eq_half : sum_series_b = 1 / 2 := sorry

theorem sum_series_c_eq_third : sum_series_c = 1 / 3 := sorry

end sum_series_a_eq_one_sum_series_b_eq_half_sum_series_c_eq_third_l82_82458


namespace common_chord_line_l82_82174

def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 8 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + 2 * x - 4 * y - 4 = 0

theorem common_chord_line : 
  ∀ x y : ℝ, (circle1 x y ∧ circle2 x y) ↔ (x - y + 1 = 0) := 
by sorry

end common_chord_line_l82_82174


namespace radius_of_circular_garden_l82_82283

theorem radius_of_circular_garden
  (r : ℝ) 
  (h₁ : 2 * real.pi * r = (1/8) * real.pi * r^2) : r = 16 :=
by 
  sorry

end radius_of_circular_garden_l82_82283


namespace percentage_increase_first_year_l82_82327

theorem percentage_increase_first_year (P : ℝ) (X : ℝ) 
  (h1 : P * (1 + X / 100) * 0.75 * 1.15 = P * 1.035) : 
  X = 20 :=
by
  sorry

end percentage_increase_first_year_l82_82327


namespace part1_part2_l82_82548

noncomputable def f (a c x : ℝ) : ℝ :=
  if x >= c then a * Real.log x + (x - c) ^ 2
  else a * Real.log x - (x - c) ^ 2

theorem part1 (a c : ℝ)
  (h_a : a = 2 * c - 2)
  (h_c_gt_0 : c > 0)
  (h_f_geq : ∀ x, x ∈ (Set.Ioi c) → f a c x >= 1 / 4) :
    a ∈ Set.Icc (-2 : ℝ) (-1 : ℝ) :=
  sorry

theorem part2 (a c x1 x2 : ℝ)
  (h_a_lt_0 : a < 0)
  (h_c_gt_0 : c > 0)
  (h_x1 : x1 = Real.sqrt (- a / 2))
  (h_x2 : x2 = c)
  (h_tangents_intersect : deriv (f a c) x1 * deriv (f a c) x2 = -1) :
    c >= 3 * Real.sqrt 3 / 2 :=
  sorry

end part1_part2_l82_82548


namespace number_of_three_digit_integers_with_odd_factors_l82_82376

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l82_82376


namespace no_term_un_eq_neg1_l82_82237

theorem no_term_un_eq_neg1 (p : ℕ) [hp_prime: Fact (Nat.Prime p)] (hp_odd: p % 2 = 1) (hp_not_five: p ≠ 5) :
  ∀ n : ℕ, ∀ u : ℕ → ℤ, ((u 0 = 0) ∧ (u 1 = 1) ∧ (∀ k, k ≥ 2 → u (k-2) = 2 * u (k-1) - p * u k)) → 
    (u n ≠ -1) :=
  sorry

end no_term_un_eq_neg1_l82_82237


namespace flat_fee_l82_82955

theorem flat_fee (f n : ℝ) 
  (h1 : f + 3 * n = 205) 
  (h2 : f + 6 * n = 350) : 
  f = 60 := 
by
  sorry

end flat_fee_l82_82955


namespace total_pencils_correct_l82_82816

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end total_pencils_correct_l82_82816


namespace box_depth_is_10_l82_82952

variable (depth : ℕ)

theorem box_depth_is_10 
  (length width : ℕ)
  (cubes : ℕ)
  (h1 : length = 35)
  (h2 : width = 20)
  (h3 : cubes = 56)
  (h4 : ∃ (cube_size : ℕ), ∀ (c : ℕ), c = cube_size → (length % cube_size = 0 ∧ width % cube_size = 0 ∧ 56 * cube_size^3 = length * width * depth)) :
  depth = 10 :=
by
  sorry

end box_depth_is_10_l82_82952


namespace probability_answered_within_first_four_rings_l82_82917

theorem probability_answered_within_first_four_rings 
  (P1 P2 P3 P4 : ℝ) (h1 : P1 = 0.1) (h2 : P2 = 0.3) (h3 : P3 = 0.4) (h4 : P4 = 0.1) :
  (1 - ((1 - P1) * (1 - P2) * (1 - P3) * (1 - P4))) = 0.9 := 
sorry

end probability_answered_within_first_four_rings_l82_82917


namespace perpendicular_lines_a_value_l82_82332

theorem perpendicular_lines_a_value (a : ℝ) : 
  (∃ x y : ℝ, ax + y + 1 = 0) ∧ (∃ x y : ℝ, x + y + 2 = 0) ∧ (∃ x y : ℝ, (y = -ax)) → a = -1 := by
  sorry

end perpendicular_lines_a_value_l82_82332


namespace deposit_time_l82_82251

theorem deposit_time (r t : ℕ) : 
  8000 + 8000 * r * t / 100 = 10200 → 
  8000 + 8000 * (r + 2) * t / 100 = 10680 → 
  t = 3 :=
by 
  sorry

end deposit_time_l82_82251


namespace find_other_number_l82_82228

theorem find_other_number (a b : ℤ) (h1 : 2 * a + 3 * b = 100) (h2 : a = 28 ∨ b = 28) : a = 8 ∨ b = 8 :=
sorry

end find_other_number_l82_82228


namespace casey_pumping_time_l82_82526

theorem casey_pumping_time :
  let pump_rate := 3 -- gallons per minute
  let corn_rows := 4
  let corn_per_row := 15
  let water_per_corn := 1 / 2
  let total_corn := corn_rows * corn_per_row
  let corn_water := total_corn * water_per_corn
  let num_pigs := 10
  let water_per_pig := 4
  let pig_water := num_pigs * water_per_pig
  let num_ducks := 20
  let water_per_duck := 1 / 4
  let duck_water := num_ducks * water_per_duck
  let total_water := corn_water + pig_water + duck_water
  let time_needed := total_water / pump_rate
  time_needed = 25 :=
by
  sorry

end casey_pumping_time_l82_82526


namespace enrique_commission_l82_82712

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l82_82712


namespace num_three_digit_integers_with_odd_factors_l82_82352

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l82_82352


namespace herd_total_cows_l82_82502

theorem herd_total_cows (n : ℕ) (h1 : (1 / 3 : ℚ) * n + (1 / 5 : ℚ) * n + (1 / 6 : ℚ) * n + 19 = n) : n = 63 :=
sorry

end herd_total_cows_l82_82502


namespace factor_x4_minus_81_l82_82311

variable (x : ℝ)

theorem factor_x4_minus_81 : 
  (x^4 - 81) = (x - 3) * (x + 3) * (x^2 + 9) :=
  by { -- proof steps would go here 
    sorry 
}

end factor_x4_minus_81_l82_82311


namespace evaluate_expression_l82_82001

theorem evaluate_expression : (20 * 3 + 10) / (5 + 3) = 9 := by
  sorry

end evaluate_expression_l82_82001


namespace find_z_to_8_l82_82336

noncomputable def complex_number_z (z : ℂ) : Prop :=
  z + z⁻¹ = 2 * Complex.cos (Real.pi / 4)

theorem find_z_to_8 (z : ℂ) (h : complex_number_z z) : (z ^ 8 + (z ^ 8)⁻¹ = 2) :=
by
  sorry

end find_z_to_8_l82_82336


namespace emma_investment_l82_82487

-- Define the basic problem parameters
def P : ℝ := 2500
def r : ℝ := 0.04
def n : ℕ := 21
def expected_amount : ℝ := 6101.50

-- Define the compound interest formula result
def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

-- State the theorem
theorem emma_investment : 
  compound_interest P r n = expected_amount := 
  sorry

end emma_investment_l82_82487


namespace function_domain_l82_82572

theorem function_domain (x : ℝ) : x ≠ 3 → ∃ y : ℝ, y = (1 / (x - 3)) :=
by
  sorry

end function_domain_l82_82572


namespace cos_330_is_sqrt3_over_2_l82_82679

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l82_82679


namespace three_digit_oddfactors_count_l82_82354

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l82_82354


namespace gcd_75_100_l82_82908

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l82_82908


namespace cos_330_eq_sqrt3_div_2_l82_82682

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l82_82682


namespace rectangle_side_length_along_hypotenuse_l82_82219

-- Define the right triangle with given sides
def triangle_PQR (PR PQ QR : ℝ) : Prop := 
  PR^2 + PQ^2 = QR^2

-- Condition: Right triangle PQR with PR = 9 and PQ = 12
def PQR : Prop := triangle_PQR 9 12 (Real.sqrt (9^2 + 12^2))

-- Define the property of the rectangle
def rectangle_condition (x : ℝ) (s : ℝ) : Prop := 
  (3 / (Real.sqrt (9^2 + 12^2))) = (x / 9) ∧ s = ((9 - x) * (Real.sqrt (9^2 + 12^2)) / 9)

-- Main theorem
theorem rectangle_side_length_along_hypotenuse : 
  PQR ∧ (∃ x, rectangle_condition x 12) → (∃ s, s = 12) :=
by
  intro h
  sorry

end rectangle_side_length_along_hypotenuse_l82_82219


namespace alpha_identity_l82_82987

theorem alpha_identity (α : ℝ) (hα : α ≠ 0) (h_tan : Real.tan α = -α) : 
    (α^2 + 1) * (1 + Real.cos (2 * α)) = 2 := 
by
  sorry

end alpha_identity_l82_82987


namespace find_value_of_m_and_n_l82_82340

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x^3 + 3*x^2 + m * x
noncomputable def g (x : ℝ) (n : ℝ) : ℝ := Real.log (x + 1) + n * x

theorem find_value_of_m_and_n (m n : ℝ) (h₀ : n > 0) 
  (h₁ : f (-1) m = -1) 
  (h₂ : ∀ x : ℝ, f x m = g x n → x = 0) :
  m + n = 5 := 
by 
  sorry

end find_value_of_m_and_n_l82_82340


namespace trig_identity_one_trig_identity_two_l82_82303

theorem trig_identity_one :
  2 * (Real.cos (45 * Real.pi / 180)) - (3 / 2) * (Real.tan (30 * Real.pi / 180)) * (Real.cos (30 * Real.pi / 180)) + (Real.sin (60 * Real.pi / 180))^2 = Real.sqrt 2 :=
sorry

theorem trig_identity_two :
  (Real.sin (30 * Real.pi / 180))⁻¹ * (Real.sin (60 * Real.pi / 180) - Real.cos (45 * Real.pi / 180)) - Real.sqrt ((1 - Real.tan (60 * Real.pi / 180))^2) = 1 - Real.sqrt 2 :=
sorry

end trig_identity_one_trig_identity_two_l82_82303


namespace pasta_cost_is_one_l82_82249

-- Define the conditions
def pasta_cost (p : ℝ) : ℝ := p -- The cost of the pasta per box
def sauce_cost : ℝ := 2.00 -- The cost of the sauce
def meatballs_cost : ℝ := 5.00 -- The cost of the meatballs
def servings : ℕ := 8 -- The number of servings
def cost_per_serving : ℝ := 1.00 -- The cost per serving

-- Calculate the total meal cost
def total_meal_cost : ℝ := servings * cost_per_serving

-- Calculate the combined cost of sauce and meatballs
def combined_cost_of_sauce_and_meatballs : ℝ := sauce_cost + meatballs_cost

-- Calculate the cost of the pasta
def pasta_cost_calculation : ℝ := total_meal_cost - combined_cost_of_sauce_and_meatballs

-- The theorem stating that the pasta cost should be $1
theorem pasta_cost_is_one (p : ℝ) (h : pasta_cost_calculation = p) : p = 1 := by
  sorry

end pasta_cost_is_one_l82_82249


namespace remainder_of_product_mod_7_l82_82857

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82857


namespace initial_assessed_value_l82_82044

theorem initial_assessed_value (V : ℝ) (tax_rate : ℝ) (new_value : ℝ) (tax_increase : ℝ) 
  (h1 : tax_rate = 0.10) 
  (h2 : new_value = 28000) 
  (h3 : tax_increase = 800) 
  (h4 : tax_rate * new_value = tax_rate * V + tax_increase) : 
  V = 20000 :=
by
  sorry

end initial_assessed_value_l82_82044


namespace maximize_take_home_pay_l82_82758

-- Define the tax system condition
def tax (y : ℝ) : ℝ := y^3

-- Define the take-home pay condition
def take_home_pay (y : ℝ) : ℝ := 100 * y^2 - tax y

-- The theorem to prove the maximum take-home pay is achieved at a specific income level
theorem maximize_take_home_pay : 
  ∃ y : ℝ, take_home_pay y = 100 * 50^2 - 50^3 := sorry

end maximize_take_home_pay_l82_82758


namespace smallest_n_not_divisible_by_10_l82_82010

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l82_82010


namespace gcd_75_100_l82_82903

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l82_82903


namespace range_of_f_l82_82258

noncomputable def f (x : ℝ) : ℝ :=
  (Real.exp (3 * x) - 2) / (Real.exp (3 * x) + 2)

theorem range_of_f (x : ℝ) : -1 < f x ∧ f x < 1 :=
by
  sorry

end range_of_f_l82_82258


namespace inequality_solution_set_l82_82030

variable {a b x : ℝ}

theorem inequality_solution_set (h : ∀ x : ℝ, ax - b > 0 ↔ x < -1) : 
  ∀ x : ℝ, (x-2) * (ax + b) < 0 ↔ x < 1 ∨ x > 2 :=
by sorry

end inequality_solution_set_l82_82030


namespace Marley_fruits_total_is_31_l82_82813

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l82_82813


namespace total_number_of_birds_l82_82481

def geese : ℕ := 58
def ducks : ℕ := 37
def swans : ℕ := 42

theorem total_number_of_birds : geese + ducks + swans = 137 := by
  sorry

end total_number_of_birds_l82_82481


namespace cos_330_eq_sqrt_3_div_2_l82_82673

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l82_82673


namespace impossible_to_load_two_coins_l82_82936

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l82_82936


namespace probability_two_or_fewer_distinct_digits_l82_82950

def digits : Set ℕ := {1, 2, 3}

def total_3_digit_numbers : ℕ := 27

def distinct_3_digit_numbers : ℕ := 6

def at_most_two_distinct_numbers : ℕ := total_3_digit_numbers - distinct_3_digit_numbers

theorem probability_two_or_fewer_distinct_digits :
  (at_most_two_distinct_numbers : ℚ) / total_3_digit_numbers = 7 / 9 := by
  sorry

end probability_two_or_fewer_distinct_digits_l82_82950


namespace range_of_m_l82_82334

variable (p q : Prop)
variable (m : ℝ)
variable (hp : (∀ x y : ℝ, (x^2 / (2 * m) + y^2 / (1 - m) = 1) → (0 < m ∧ m < 1/3)))
variable (hq : (m^2 - 15 * m < 0))

theorem range_of_m (h_not_p_and_q : ¬ (p ∧ q)) (h_p_or_q : p ∨ q) :
  (1/3 ≤ m ∧ m < 15) :=
sorry

end range_of_m_l82_82334


namespace range_of_a_l82_82586

variable (a : ℝ)
def A := Set.Ico (-2 : ℝ) 4
def B := {x : ℝ | x^2 - a * x - 4 ≤ 0 }

theorem range_of_a (h : B a ⊆ A) : 0 ≤ a ∧ a < 3 :=
by
  sorry

end range_of_a_l82_82586


namespace integer_pairs_satisfy_equation_l82_82554

theorem integer_pairs_satisfy_equation :
  ∃ (S : Finset (ℤ × ℤ)), S.card = 5 ∧ ∀ (m n : ℤ), (m, n) ∈ S ↔ m^2 + n = m * n + 1 :=
by
  sorry

end integer_pairs_satisfy_equation_l82_82554


namespace cos_330_eq_sqrt_3_div_2_l82_82675

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l82_82675


namespace lap_time_improvement_l82_82071

theorem lap_time_improvement (initial_laps : ℕ) (initial_time : ℕ) (current_laps : ℕ) (current_time : ℕ)
  (h1 : initial_laps = 15) (h2 : initial_time = 45) (h3 : current_laps = 18) (h4 : current_time = 42) :
  (45 / 15 - 42 / 18 : ℚ) = 2 / 3 :=
by
  sorry

end lap_time_improvement_l82_82071


namespace triangle_angle_contradiction_l82_82280

theorem triangle_angle_contradiction (α β γ : ℝ) (h1 : α + β + γ = 180)
(h2 : α > 60) (h3 : β > 60) (h4 : γ > 60) : false :=
sorry

end triangle_angle_contradiction_l82_82280


namespace min_value_f_l82_82180

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end min_value_f_l82_82180


namespace ascending_function_k_ge_2_l82_82069

open Real

def is_ascending (f : ℝ → ℝ) (k : ℝ) (M : Set ℝ) : Prop :=
  ∀ x ∈ M, f (x + k) ≥ f x

theorem ascending_function_k_ge_2 :
  ∀ (k : ℝ), (∀ x : ℝ, x ≥ -1 → (x + k) ^ 2 ≥ x ^ 2) → k ≥ 2 :=
by
  intros k h
  sorry

end ascending_function_k_ge_2_l82_82069


namespace smallest_magnitude_z_theorem_l82_82588

noncomputable def smallest_magnitude_z (z : ℂ) : ℝ :=
  Complex.abs z

theorem smallest_magnitude_z_theorem : 
  ∃ z : ℂ, (Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) ∧
  smallest_magnitude_z z = 36 / Real.sqrt 97 := 
sorry

end smallest_magnitude_z_theorem_l82_82588


namespace parking_spaces_remaining_l82_82883

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end parking_spaces_remaining_l82_82883


namespace value_of_y_in_arithmetic_sequence_l82_82130

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l82_82130


namespace cos_330_eq_sqrt_3_div_2_l82_82656

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l82_82656


namespace sums_of_coordinates_of_A_l82_82781

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l82_82781


namespace roots_seventh_sum_l82_82804

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end roots_seventh_sum_l82_82804


namespace num_three_digit_ints_with_odd_factors_l82_82358

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l82_82358


namespace find_f_2011_l82_82733

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f(x) = f(-x)
axiom symmetric_shift : ∀ x : ℝ, f(2 + x) = f(2 - x)
axiom given_condition : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → f(x) = Real.log 2 (1 - x)

theorem find_f_2011 : f 2011 = 1 :=
by
  sorry

end find_f_2011_l82_82733


namespace problem_statement_l82_82734

open Set

noncomputable def U := ℝ

def A : Set ℝ := { x | 0 < 2 * x + 4 ∧ 2 * x + 4 < 10 }
def B : Set ℝ := { x | x < -4 ∨ x > 2 }
def C (a : ℝ) (h : a < 0) : Set ℝ := { x | x^2 - 4 * a * x + 3 * a^2 < 0 }

theorem problem_statement (a : ℝ) (ha : a < 0) :
    A ∪ B = { x | x < -4 ∨ x > -2 } ∧
    compl (A ∪ B) ⊆ C a ha → -2 < a ∧ a < -4 / 3 :=
sorry

end problem_statement_l82_82734


namespace remainder_product_l82_82869

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82869


namespace find_n_l82_82014

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l82_82014


namespace coin_loading_impossible_l82_82922

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l82_82922


namespace find_n_l82_82027

theorem find_n (n : ℝ) (h1 : (n ≠ 0)) (h2 : ∃ (n' : ℝ), n = n' ∧ -n' = -9 / n') (h3 : ∀ x : ℝ, x > 0 → -n * x < 0) : n = 3 :=
sorry

end find_n_l82_82027


namespace edge_length_of_box_l82_82262

noncomputable def edge_length_cubical_box (num_cubes : ℕ) (edge_length_cube : ℝ) : ℝ :=
  if num_cubes = 8 ∧ edge_length_cube = 0.5 then -- 50 cm in meters
    1 -- The edge length of the cubical box in meters
  else
    0 -- Placeholder for other cases

theorem edge_length_of_box :
  edge_length_cubical_box 8 0.5 = 1 :=
sorry

end edge_length_of_box_l82_82262


namespace three_digit_integers_with_odd_factors_l82_82362

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l82_82362


namespace cubic_equation_roots_l82_82109

theorem cubic_equation_roots (a b c d : ℝ) (h_a : a ≠ 0) 
(h_root1 : a * 4^3 + b * 4^2 + c * 4 + d = 0)
(h_root2 : a * (-3)^3 + b * (-3)^2 - 3 * c + d = 0) :
 (b + c) / a = -13 :=
by sorry

end cubic_equation_roots_l82_82109


namespace cream_ratio_l82_82795

theorem cream_ratio (j : ℝ) (jo : ℝ) (jc : ℝ) (joc : ℝ) (jdrank : ℝ) (jodrank : ℝ) :
  j = 15 ∧ jo = 15 ∧ jc = 3 ∧ joc = 2.5 ∧ jdrank = 0 ∧ jodrank = 0.5 →
  j + jc - jdrank = jc ∧ jo + jc - jodrank = joc →
  (jc / joc) = (6 / 5) :=
  by
  sorry

end cream_ratio_l82_82795


namespace average_A_B_l82_82609

variables (A B C : ℝ)

def conditions (A B C : ℝ) : Prop :=
  (A + B + C) / 3 = 45 ∧
  (B + C) / 2 = 43 ∧
  B = 31

theorem average_A_B (A B C : ℝ) (h : conditions A B C) : (A + B) / 2 = 40 :=
by
  sorry

end average_A_B_l82_82609


namespace company_pays_300_per_month_l82_82632

theorem company_pays_300_per_month
  (length width height : ℝ)
  (total_volume : ℝ)
  (cost_per_box_per_month : ℝ)
  (h1 : length = 15)
  (h2 : width = 12)
  (h3 : height = 10)
  (h4 : total_volume = 1080000)
  (h5 : cost_per_box_per_month = 0.5) :
  (total_volume / (length * width * height)) * cost_per_box_per_month = 300 := by
  sorry

end company_pays_300_per_month_l82_82632


namespace joe_first_lift_weight_l82_82217

variables (x y : ℕ)

theorem joe_first_lift_weight (h1 : x + y = 600) (h2 : 2 * x = y + 300) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l82_82217


namespace odd_function_evaluation_l82_82235

theorem odd_function_evaluation
  (f : ℝ → ℝ)
  (h_odd : ∀ x, f (-x) = -f x)
  (h_def : ∀ x, x ≤ 0 → f x = 2 * x^2 - x) :
  f 1 = -3 :=
by {
  sorry
}

end odd_function_evaluation_l82_82235


namespace charity_amount_l82_82163

theorem charity_amount (total : ℝ) (charities : ℕ) (amount_per_charity : ℝ) 
  (h1 : total = 3109) (h2 : charities = 25) : 
  amount_per_charity = 124.36 :=
by
  sorry

end charity_amount_l82_82163


namespace abs_eq_four_l82_82038

theorem abs_eq_four (x : ℝ) (h : |x| = 4) : x = 4 ∨ x = -4 :=
by
  sorry

end abs_eq_four_l82_82038


namespace three_digit_integers_with_odd_factors_count_l82_82389

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l82_82389


namespace infinite_rational_points_on_circle_l82_82601

noncomputable def exists_infinitely_many_rational_points_on_circle : Prop :=
  ∃ f : ℚ → ℚ × ℚ, (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                   (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m)

theorem infinite_rational_points_on_circle :
  ∃ (f : ℚ → ℚ × ℚ), (∀ m : ℚ, (f m).1 ^ 2 + (f m).2 ^ 2 = 1) ∧ 
                     (∀ x y : ℚ, ∃ m : ℚ, (x, y) = f m) := sorry

end infinite_rational_points_on_circle_l82_82601


namespace cos_330_eq_sqrt3_div_2_l82_82666

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82666


namespace train_passes_tree_in_16_seconds_l82_82138

noncomputable def time_to_pass_tree (length_train : ℕ) (speed_train_kmh : ℕ) : ℕ :=
  let speed_train_ms := (speed_train_kmh * 1000) / 3600
  length_train / speed_train_ms

theorem train_passes_tree_in_16_seconds :
  time_to_pass_tree 280 63 = 16 :=
  by
    sorry

end train_passes_tree_in_16_seconds_l82_82138


namespace parabola_focus_l82_82750

theorem parabola_focus (a : ℝ) (h : a ≠ 0) (h_directrix : ∀ x y : ℝ, y^2 = a * x → x = -1) : 
    ∃ x y : ℝ, (y = 0 ∧ x = 1 ∧ y^2 = a * x) :=
sorry

end parabola_focus_l82_82750


namespace tens_digit_6_pow_45_l82_82974

theorem tens_digit_6_pow_45 : (6 ^ 45 % 100) / 10 = 0 := 
by 
  sorry

end tens_digit_6_pow_45_l82_82974


namespace bob_start_time_l82_82135

-- Define constants for the problem conditions
def yolandaRate : ℝ := 3 -- Yolanda's walking rate in miles per hour
def bobRate : ℝ := 4 -- Bob's walking rate in miles per hour
def distanceXY : ℝ := 10 -- Distance between point X and Y in miles
def bobDistanceWhenMet : ℝ := 4 -- Distance Bob had walked when they met in miles

-- Define the theorem statement
theorem bob_start_time : 
  ∃ T : ℝ, (yolandaRate * T + bobDistanceWhenMet = distanceXY) →
  (T = 2) →
  ∃ tB : ℝ, T - tB = 1 :=
by
  -- Insert proof here
  sorry

end bob_start_time_l82_82135


namespace product_mod_7_l82_82843

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82843


namespace sam_bought_new_books_l82_82297

   def books_question (a m u : ℕ) : ℕ := (a + m) - u

   theorem sam_bought_new_books (a m u : ℕ) (h1 : a = 13) (h2 : m = 17) (h3 : u = 15) :
     books_question a m u = 15 :=
   by sorry
   
end sam_bought_new_books_l82_82297


namespace constant_term_expansion_l82_82467

theorem constant_term_expansion : 
  ∃ r : ℕ, (9 - 3 * r / 2 = 0) ∧ 
  ∀ (x : ℝ) (hx : x ≠ 0), (2 * x - 1 / Real.sqrt x) ^ 9 = 672 := 
by sorry

end constant_term_expansion_l82_82467


namespace sum_of_coordinates_A_l82_82769

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l82_82769


namespace cos_330_eq_sqrt3_div_2_l82_82667

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82667


namespace cos_330_is_sqrt3_over_2_l82_82678

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l82_82678


namespace difference_in_percentage_l82_82508

noncomputable def principal : ℝ := 600
noncomputable def timePeriod : ℝ := 10
noncomputable def interestDifference : ℝ := 300

theorem difference_in_percentage (R D : ℝ) (h : 60 * (R + D) - 60 * R = 300) : D = 5 := 
by
  -- Proof is not provided, as instructed
  sorry

end difference_in_percentage_l82_82508


namespace find_complex_number_l82_82331

-- Define the complex number z and the condition
variable (z : ℂ)
variable (h : (conj z) / (1 + I) = 1 - 2 * I)

-- State the theorem
theorem find_complex_number (hz : h) : z = 3 + I := 
sorry

end find_complex_number_l82_82331


namespace num_odd_factors_of_three_digit_integers_eq_22_l82_82410

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l82_82410


namespace sum_coordinates_A_l82_82772

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l82_82772


namespace coin_loading_impossible_l82_82924

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l82_82924


namespace erick_total_money_collected_l82_82269

noncomputable def new_lemon_price (old_price increase : ℝ) : ℝ := old_price + increase
noncomputable def new_grape_price (old_price increase : ℝ) : ℝ := old_price + increase / 2

noncomputable def total_money_collected (lemons grapes : ℕ)
                                       (lemon_price grape_price lemon_increase : ℝ) : ℝ :=
  let new_lemon_price := new_lemon_price lemon_price lemon_increase
  let new_grape_price := new_grape_price grape_price lemon_increase
  lemons * new_lemon_price + grapes * new_grape_price

theorem erick_total_money_collected :
  total_money_collected 80 140 8 7 4 = 2220 := 
by
  sorry

end erick_total_money_collected_l82_82269


namespace remainder_of_product_mod_7_l82_82863

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82863


namespace bianca_next_day_run_l82_82301

-- Define the conditions
variable (miles_first_day : ℕ) (total_miles : ℕ)

-- Set the conditions for Bianca's run
def conditions := miles_first_day = 8 ∧ total_miles = 12

-- State the proposition we need to prove
def miles_next_day (miles_first_day total_miles : ℕ) : ℕ := total_miles - miles_first_day

-- The theorem stating the problem to prove
theorem bianca_next_day_run (h : conditions 8 12) : miles_next_day 8 12 = 4 := by
  unfold conditions at h
  simp [miles_next_day] at h
  sorry

end bianca_next_day_run_l82_82301


namespace arithmetic_sequence_y_value_l82_82118

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l82_82118


namespace ratio_of_numbers_l82_82618

-- Definitions for the conditions
variable (S L : ℕ)

-- Given conditions
def condition1 : Prop := S + L = 44
def condition2 : Prop := S = 20
def condition3 : Prop := L = 6 * S

-- The theorem to be proven
theorem ratio_of_numbers (h1 : condition1 S L) (h2 : condition2 S) (h3 : condition3 S L) : L / S = 6 := 
  sorry

end ratio_of_numbers_l82_82618


namespace Marley_fruit_count_l82_82807

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l82_82807


namespace zero_of_function_l82_82580

noncomputable def f (x : ℝ) : ℝ := Real.exp x + 2 * x - 4

theorem zero_of_function (x : ℝ) (h : f x = 0) (x1 x2 : ℝ)
  (h1 : -1 < x1 ∧ x1 < x)
  (h2 : x < x2 ∧ x2 < 2) :
  f x1 < 0 ∧ f x2 > 0 :=
by
  sorry

end zero_of_function_l82_82580


namespace value_of_other_bills_l82_82062

theorem value_of_other_bills (x : ℕ) : 
  (∃ (num_twenty num_x : ℕ), num_twenty = 3 ∧
                           num_x = 2 * num_twenty ∧
                           20 * num_twenty + x * num_x = 120) → 
  x * 6 = 60 :=
by
  intro h
  obtain ⟨num_twenty, num_x, h1, h2, h3⟩ := h
  have : num_twenty = 3 := h1
  have : num_x = 2 * num_twenty := h2
  have : x * 6 = 60 := sorry
  exact this

end value_of_other_bills_l82_82062


namespace bc_fraction_ad_l82_82595

theorem bc_fraction_ad
  (B C E A D : Type)
  (on_AD : ∀ P : Type, P = B ∨ P = C ∨ P = E)
  (AB BD AC CD DE EA: ℝ)
  (h1 : AB = 3 * BD)
  (h2 : AC = 5 * CD)
  (h3 : DE = 2 * EA)

  : ∃ BC AD: ℝ, BC = 1 / 12 * AD := 
sorry -- Proof is omitted

end bc_fraction_ad_l82_82595


namespace bert_ernie_ratio_l82_82299

theorem bert_ernie_ratio (berts_stamps ernies_stamps peggys_stamps : ℕ) 
  (h1 : peggys_stamps = 75) 
  (h2 : ernies_stamps = 3 * peggys_stamps) 
  (h3 : berts_stamps = peggys_stamps + 825) : 
  berts_stamps / ernies_stamps = 4 := 
by sorry

end bert_ernie_ratio_l82_82299


namespace range_of_a_l82_82753

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x - a ≤ -3) → a ∈ Set.Iic (-6) ∪ Set.Ici 2 :=
by
  intro h
  sorry

end range_of_a_l82_82753


namespace three_digit_integers_with_odd_number_of_factors_l82_82393

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l82_82393


namespace find_a_l82_82316

theorem find_a (r s a : ℚ) (h1 : s^2 = 16) (h2 : 2 * r * s = 15) (h3 : a = r^2) : a = 225/64 := by
  sorry

end find_a_l82_82316


namespace sum_of_coordinates_A_l82_82774

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l82_82774


namespace num_three_digit_integers_with_odd_factors_l82_82350

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l82_82350


namespace enrique_commission_l82_82714

theorem enrique_commission :
  let commission_rate : ℚ := 0.15
  let suits_sold : ℚ := 2
  let suits_price : ℚ := 700
  let shirts_sold : ℚ := 6
  let shirts_price : ℚ := 50
  let loafers_sold : ℚ := 2
  let loafers_price : ℚ := 150
  let total_sales := suits_sold * suits_price + shirts_sold * shirts_price + loafers_sold * loafers_price
  let commission := commission_rate * total_sales
  commission = 300 := by
begin
  sorry
end

end enrique_commission_l82_82714


namespace number_of_three_digit_integers_with_odd_number_of_factors_l82_82402

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l82_82402


namespace num_right_triangles_with_incenter_origin_l82_82236

theorem num_right_triangles_with_incenter_origin (p : ℕ) (hp : Nat.Prime p) :
  let M : ℤ × ℤ := (p * 1994, 7 * p * 1994)
  let is_lattice_point (x : ℤ × ℤ) : Prop := True  -- All points considered are lattice points
  let is_right_angle_vertex (M : ℤ × ℤ) : Prop := True
  let is_incenter_origin (M : ℤ × ℤ) : Prop := True
  let num_triangles (p : ℕ) : ℕ :=
    if p = 2 then 18
    else if p = 997 then 20
    else 36
  num_triangles p = if p = 2 then 18 else if p = 997 then 20 else 36 := (

  by sorry

 )

end num_right_triangles_with_incenter_origin_l82_82236


namespace John_traded_in_car_money_back_l82_82488

-- First define the conditions provided in the problem.
def UberEarnings : ℝ := 30000
def CarCost : ℝ := 18000
def UberProfit : ℝ := 18000

-- We need to prove that John got $6000 back when trading in the car.
theorem John_traded_in_car_money_back : 
  UberEarnings - UberProfit = CarCost - 6000 := 
by
  -- provide the detailed steps inside the proof block if needed
  sorry

end John_traded_in_car_money_back_l82_82488


namespace attraction_ticket_cost_l82_82443

theorem attraction_ticket_cost
  (cost_park_entry : ℕ)
  (cost_attraction_parent : ℕ)
  (total_paid : ℕ)
  (num_children : ℕ)
  (num_parents : ℕ)
  (num_grandmother : ℕ)
  (x : ℕ)
  (h_costs : cost_park_entry = 5)
  (h_attraction_parent : cost_attraction_parent = 4)
  (h_family : num_children = 4 ∧ num_parents = 2 ∧ num_grandmother = 1)
  (h_total_paid : total_paid = 55)
  (h_equation : (num_children + num_parents + num_grandmother) * cost_park_entry + (num_parents + num_grandmother) * cost_attraction_parent + num_children * x = total_paid) :
  x = 2 := by
  sorry

end attraction_ticket_cost_l82_82443


namespace factor_expression_l82_82538

theorem factor_expression (y z : ℝ) : 3 * y^2 - 75 * z^2 = 3 * (y + 5 * z) * (y - 5 * z) :=
by sorry

end factor_expression_l82_82538


namespace solve_for_x_l82_82040

/-- Let f(x) = 2 - 1 / (2 - x)^3.
Proof that f(x) = 1 / (2 - x)^3 implies x = 1. -/
theorem solve_for_x (x : ℝ) (h : 2 - 1 / (2 - x)^3 = 1 / (2 - x)^3) : x = 1 :=
  sorry

end solve_for_x_l82_82040


namespace drink_total_amount_l82_82284

theorem drink_total_amount (total_amount: ℝ) (grape_juice: ℝ) (grape_proportion: ℝ) 
  (h1: grape_proportion = 0.20) (h2: grape_juice = 40) : total_amount = 200 :=
by
  -- Definitions and assumptions
  let calculation := grape_juice / grape_proportion
  -- Placeholder for the proof
  sorry

end drink_total_amount_l82_82284


namespace solution_set_of_inequality_l82_82260

theorem solution_set_of_inequality :
  {x : ℝ | 3 * x ^ 2 - 7 * x - 10 ≥ 0} = {x : ℝ | x ≥ (10 / 3) ∨ x ≤ -1} :=
sorry

end solution_set_of_inequality_l82_82260


namespace oil_tank_depth_l82_82500

theorem oil_tank_depth (L r A : ℝ) (h : ℝ) (L_pos : L = 8) (r_pos : r = 2) (A_pos : A = 16) :
  h = 2 - Real.sqrt 3 ∨ h = 2 + Real.sqrt 3 :=
by
  sorry

end oil_tank_depth_l82_82500


namespace no_integral_solutions_l82_82994

theorem no_integral_solutions : ∀ (x : ℤ), x^5 - 31 * x + 2015 ≠ 0 :=
by
  sorry

end no_integral_solutions_l82_82994


namespace solution_set_of_quadratic_inequality_2_l82_82881

-- Definitions
variables {a b c x : ℝ}
def quadratic_inequality_1 (a b c x : ℝ) := a * x^2 + b * x + c < 0
def quadratic_inequality_2 (a b c x : ℝ) := a * x^2 - b * x + c > 0

-- Conditions
axiom condition_1 : ∀ x, quadratic_inequality_1 a b c x ↔ (x < -2 ∨ x > -1/2)
axiom condition_2 : a < 0
axiom condition_3 : ∃ x, a * x^2 + b * x + c = 0 ∧ (x = -2 ∨ x = -1/2)
axiom condition_4 : b = 5 * a / 2
axiom condition_5 : c = a

-- Proof Problem
theorem solution_set_of_quadratic_inequality_2 : ∀ x, quadratic_inequality_2 a b c x ↔ (1/2 < x ∧ x < 2) :=
by
  -- Proof goes here
  sorry

end solution_set_of_quadratic_inequality_2_l82_82881


namespace two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l82_82707

theorem two_pow_1000_mod_3 : 2^1000 % 3 = 1 := sorry
theorem two_pow_1000_mod_5 : 2^1000 % 5 = 1 := sorry
theorem two_pow_1000_mod_11 : 2^1000 % 11 = 1 := sorry
theorem two_pow_1000_mod_13 : 2^1000 % 13 = 3 := sorry

end two_pow_1000_mod_3_two_pow_1000_mod_5_two_pow_1000_mod_11_two_pow_1000_mod_13_l82_82707


namespace motorcyclist_cross_time_l82_82286

/-- Definitions and conditions -/
def speed_X := 2 -- Rounds per hour
def speed_Y := 4 -- Rounds per hour

/-- Proof statement -/
theorem motorcyclist_cross_time : (1 / (speed_X + speed_Y) * 60 = 10) :=
by
  sorry

end motorcyclist_cross_time_l82_82286


namespace product_mod_7_l82_82847

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82847


namespace sarah_age_ratio_l82_82082

theorem sarah_age_ratio 
  (S M : ℕ) 
  (h1 : S = 3 * (S / 3))
  (h2 : S - M = 5 * (S / 3 - 2 * M)) : 
  S / M = 27 / 2 := 
sorry

end sarah_age_ratio_l82_82082


namespace cos_330_eq_sqrt_3_div_2_l82_82654

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l82_82654


namespace alex_shirts_4_l82_82514

/-- Define the number of new shirts Alex, Joe, and Ben have. -/
def shirts_of_alex (alex_shirts : ℕ) (joe_shirts : ℕ) (ben_shirts : ℕ) : Prop :=
  joe_shirts = alex_shirts + 3 ∧ ben_shirts = joe_shirts + 8 ∧ ben_shirts = 15

theorem alex_shirts_4 {alex_shirts : ℕ} :
  ∃ joe_shirts ben_shirts, shirts_of_alex alex_shirts joe_shirts ben_shirts ∧ alex_shirts = 4 :=
by
  have joe_shirts := 4 + 3 by rfl
  have ben_shirts := 7 + 8 by rfl
  use joe_shirts, ben_shirts
  split
  . exact ⟨rfl, rfl, rfl⟩
  . exact rfl
  sorry

end alex_shirts_4_l82_82514


namespace measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l82_82057

noncomputable def C (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

#eval C (2 * Real.sqrt 2) 5 (Real.sqrt 13) -- Should evaluate to π/4

theorem measure_angle_C : C (2 * Real.sqrt 2) 5 (Real.sqrt 13) = π / 4 :=
sorry

noncomputable def sin_A (a c C : ℝ) : ℝ :=
  a * Real.sin C / c

#eval sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) -- Should evaluate to 2 * Real.sqrt 13 / 13

theorem value_of_sin_A : sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) = 2 * Real.sqrt 13 / 13 :=
sorry

noncomputable def sin_2A_plus_pi_over_4 (A : ℝ) : ℝ :=
  Real.sin (2 * A + π / 4)

theorem value_of_sin_2A_plus_pi_over_4 (A : ℝ) : sin_2A_plus_pi_over_4 (Real.arcsin (2 * Real.sqrt 13 / 13)) = 17 * Real.sqrt 2 / 26 :=
sorry

end measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l82_82057


namespace product_remainder_mod_7_l82_82874

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82874


namespace at_least_3_students_same_score_l82_82888

-- Conditions
def initial_points : ℕ := 6
def correct_points : ℕ := 4
def incorrect_points : ℤ := -1
def num_questions : ℕ := 6
def num_students : ℕ := 51

-- Question
theorem at_least_3_students_same_score :
  ∃ score : ℤ, ∃ students_with_same_score : ℕ, students_with_same_score ≥ 3 :=
by
  sorry

end at_least_3_students_same_score_l82_82888


namespace product_mod_7_l82_82848

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82848


namespace num_odd_factors_of_three_digit_integers_eq_22_l82_82413

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l82_82413


namespace three_digit_perfect_squares_count_l82_82379

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l82_82379


namespace log2_6_gt_2_sqrt_5_l82_82529

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end log2_6_gt_2_sqrt_5_l82_82529


namespace remainder_of_product_mod_7_l82_82858

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82858


namespace gcd_75_100_l82_82892

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l82_82892


namespace quadratic_min_value_max_l82_82342

theorem quadratic_min_value_max (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : b^2 - 4 * a * c ≥ 0) :
    (min (min ((b + c) / a) ((c + a) / b)) ((a + b) / c)) ≤ (5 / 4) :=
sorry

end quadratic_min_value_max_l82_82342


namespace marley_fruits_l82_82811

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l82_82811


namespace measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l82_82056

noncomputable def C (a b c : ℝ) : ℝ :=
  Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))

#eval C (2 * Real.sqrt 2) 5 (Real.sqrt 13) -- Should evaluate to π/4

theorem measure_angle_C : C (2 * Real.sqrt 2) 5 (Real.sqrt 13) = π / 4 :=
sorry

noncomputable def sin_A (a c C : ℝ) : ℝ :=
  a * Real.sin C / c

#eval sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) -- Should evaluate to 2 * Real.sqrt 13 / 13

theorem value_of_sin_A : sin_A (2 * Real.sqrt 2) (Real.sqrt 13) (π / 4) = 2 * Real.sqrt 13 / 13 :=
sorry

noncomputable def sin_2A_plus_pi_over_4 (A : ℝ) : ℝ :=
  Real.sin (2 * A + π / 4)

theorem value_of_sin_2A_plus_pi_over_4 (A : ℝ) : sin_2A_plus_pi_over_4 (Real.arcsin (2 * Real.sqrt 13 / 13)) = 17 * Real.sqrt 2 / 26 :=
sorry

end measure_angle_C_value_of_sin_A_value_of_sin_2A_plus_pi_over_4_l82_82056


namespace solve_fraction_equation_l82_82602

theorem solve_fraction_equation (x : ℚ) :
  (3 - x) / (x + 2) + (3 * x - 9) / (3 - x) = 2 ↔ x = -7 / 6 := 
by
  sorry

end solve_fraction_equation_l82_82602


namespace find_n_l82_82013

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l82_82013


namespace exists_line_intersecting_all_segments_l82_82503

theorem exists_line_intersecting_all_segments 
  (segments : List (ℝ × ℝ)) 
  (h1 : ∀ (P Q R : (ℝ × ℝ)), P ∈ segments → Q ∈ segments → R ∈ segments → ∃ (L : ℝ × ℝ → Prop), L P ∧ L Q ∧ L R) :
  ∃ (L : ℝ × ℝ → Prop), ∀ (S : (ℝ × ℝ)), S ∈ segments → L S :=
by
  sorry

end exists_line_intersecting_all_segments_l82_82503


namespace minimum_value_expression_l82_82234

noncomputable def expression (a b c d : ℝ) : ℝ :=
  (a + b) / c + (a + c) / d + (b + d) / a + (c + d) / b

theorem minimum_value_expression (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  expression a b c d ≥ 8 :=
by
  -- Proof goes here
  sorry

end minimum_value_expression_l82_82234


namespace equate_operations_l82_82310

theorem equate_operations :
  (15 * 5) / (10 + 2) = 3 → 8 / 4 = 2 → ((18 * 6) / (14 + 4) = 6) :=
by
sorry

end equate_operations_l82_82310


namespace extreme_point_at_1_l82_82549

noncomputable def f (a x : ℝ) : ℝ :=
  (1 / 2) * x^2 + (2 * a^3 - a^2) * Real.log x - (a^2 + 2 * a - 1) * x

theorem extreme_point_at_1 (a : ℝ) :
  (∃ x : ℝ, x = 1 ∧ ∀ x > 0, deriv (f a) x = 0 →
  a = -1) := sorry

end extreme_point_at_1_l82_82549


namespace three_digit_integers_with_odd_factors_l82_82373

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l82_82373


namespace minimum_bail_rate_l82_82966

theorem minimum_bail_rate
  (distance : ℝ) (leak_rate : ℝ) (rain_rate : ℝ) (sink_threshold : ℝ) (rowing_speed : ℝ) (time_in_minutes : ℝ) (bail_rate : ℝ) : 
  (distance = 2) → 
  (leak_rate = 15) → 
  (rain_rate = 5) →
  (sink_threshold = 60) →
  (rowing_speed = 3) →
  (time_in_minutes = (2 / 3) * 60) →
  (bail_rate = sink_threshold / (time_in_minutes) - (rain_rate + leak_rate)) →
  bail_rate ≥ 18.5 :=
by
  intros h_distance h_leak_rate h_rain_rate h_sink_threshold h_rowing_speed h_time_in_minutes h_bail_rate
  sorry

end minimum_bail_rate_l82_82966


namespace cos_330_eq_sqrt_3_div_2_l82_82676

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l82_82676


namespace find_x_l82_82722

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem find_x (x : ℝ) (h : log_base 3 (x - 3) + log_base (Real.sqrt 3) (x^3 - 3) + log_base (1/3) (x - 3) = 3 ∧ x > 3) : 
  x = Real.cbrt 12 :=
begin
  sorry
end

end find_x_l82_82722


namespace remaining_oak_trees_l82_82105

def initial_oak_trees : ℕ := 9
def cut_down_oak_trees : ℕ := 2

theorem remaining_oak_trees : initial_oak_trees - cut_down_oak_trees = 7 := 
by 
  sorry

end remaining_oak_trees_l82_82105


namespace part1_part2_l82_82550

def f (x : ℝ) : ℝ := x^2 - 1

theorem part1 (m x : ℝ) (hx : 1 ≤ x ∧ x ≤ 2) (ineq : 4 * m^2 * |f x| + 4 * f m ≤ |f (x-1)|) : 
    -1/2 ≤ m ∧ m ≤ 1/2 := 
sorry

theorem part2 (x1 : ℝ) (hx1 : 1 ≤ x1 ∧ x1 ≤ 2) : 
    (∃ x2 : ℝ, 1 ≤ x2 ∧ x2 ≤ 2 ∧ f x1 = |2 * f x2 - a * x2|) →
    (0 ≤ a ∧ a ≤ 3/2 ∨ a = 3) := 
sorry

end part1_part2_l82_82550


namespace middle_term_arithmetic_sequence_l82_82114

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l82_82114


namespace cos_330_eq_sqrt3_div_2_l82_82687

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82687


namespace cream_ratio_Joe_JoAnn_l82_82796

def Joe_initial_coffee := 15
def Joe_drank_coffee := 3
def Joe_added_cream := 3

def JoAnn_initial_coffee := 15
def JoAnn_added_cream := 3
def JoAnn_drank_total := 3

theorem cream_ratio_Joe_JoAnn :
  let Joe_final_cream := Joe_added_cream,
      JoAnn_total_volume := JoAnn_initial_coffee + JoAnn_added_cream,
      JoAnn_cream_concentration := (JoAnn_added_cream : ℚ) / JoAnn_total_volume,
      JoAnn_drank_cream := (JoAnn_drank_total : ℚ) * JoAnn_cream_concentration,
      JoAnn_remaining_cream := (JoAnn_added_cream : ℚ) - JoAnn_drank_cream,
      Joe_cream_amount := Joe_final_cream,
      JoAnn_cream_amount := JoAnn_remaining_cream
  in (Joe_cream_amount : ℚ) / JoAnn_cream_amount = 6 / 5 :=
by
  sorry

end cream_ratio_Joe_JoAnn_l82_82796


namespace distance_from_center_l82_82484

-- Define the circle equation as a predicate
def isCircle (x y : ℝ) : Prop :=
  x^2 + y^2 = 2 * x - 4 * y + 8

-- Define the center of the circle
def circleCenter : ℝ × ℝ := (1, -2)

-- Define the point in question
def point : ℝ × ℝ := (-3, 4)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the proof problem
theorem distance_from_center :
  ∀ (x y : ℝ), isCircle x y → distance circleCenter point = 2 * Real.sqrt 13 :=
by
  sorry

end distance_from_center_l82_82484


namespace remainder_of_product_mod_7_l82_82862

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82862


namespace gcd_75_100_l82_82909

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l82_82909


namespace greatest_common_divisor_456_108_lt_60_l82_82912

theorem greatest_common_divisor_456_108_lt_60 : 
  let divisors_456 := {d : ℕ | d ∣ 456}
  let divisors_108 := {d : ℕ | d ∣ 108}
  let common_divisors := divisors_456 ∩ divisors_108
  let common_divisors_lt_60 := {d ∈ common_divisors | d < 60}
  ∃ d, d ∈ common_divisors_lt_60 ∧ ∀ e ∈ common_divisors_lt_60, e ≤ d ∧ d = 12 := by {
    sorry
  }

end greatest_common_divisor_456_108_lt_60_l82_82912


namespace arithmetic_sequence_y_value_l82_82117

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l82_82117


namespace three_digit_odds_factors_count_l82_82419

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l82_82419


namespace abs_x_minus_y_l82_82193

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end abs_x_minus_y_l82_82193


namespace age_problem_l82_82137

-- Define the ages of a, b, and c
variables (a b c : ℕ)

-- State the conditions
theorem age_problem (h1 : a = b + 2) (h2 : b = 2 * c) (h3 : a + b + c = 22) : b = 8 :=
by
  sorry

end age_problem_l82_82137


namespace middle_term_arithmetic_sequence_l82_82116

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l82_82116


namespace min_value_of_a2_b2_l82_82426

noncomputable def f (x a b : ℝ) := Real.exp x + a * x + b

theorem min_value_of_a2_b2 {a b : ℝ} (h : ∃ t ∈ Set.Icc (1 : ℝ) (3 : ℝ), f t a b = 0) :
  a^2 + b^2 ≥ (Real.exp 1)^2 / 2 :=
by
  sorry

end min_value_of_a2_b2_l82_82426


namespace inequality_a2_b2_c2_geq_abc_l82_82806

theorem inequality_a2_b2_c2_geq_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_cond: a + b + c ≥ a * b * c) :
  a^2 + b^2 + c^2 ≥ a * b * c := 
sorry

end inequality_a2_b2_c2_geq_abc_l82_82806


namespace base7_addition_l82_82315

theorem base7_addition (Y X : Nat) (k m : Int) :
    (Y + 2 = X + 7 * k) ∧ (X + 5 = 4 + 7 * m) ∧ (5 = 6 + 7 * -1) → X + Y = 10 :=
by
  sorry

end base7_addition_l82_82315


namespace train_speed_kmph_l82_82157

def length_of_train : ℝ := 120
def length_of_bridge : ℝ := 255.03
def time_to_cross : ℝ := 30

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross * 3.6 = 45.0036 :=
by
  sorry

end train_speed_kmph_l82_82157


namespace last_two_digits_sum_factorials_l82_82719

theorem last_two_digits_sum_factorials : 
  (Finset.sum (Finset.range 2003) (λ n, n.factorial) % 100) = 13 := 
by 
  sorry

end last_two_digits_sum_factorials_l82_82719


namespace remainder_of_product_l82_82849

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82849


namespace average_weight_of_section_A_l82_82104

theorem average_weight_of_section_A (nA nB : ℕ) (WB WC : ℝ) (WA : ℝ) :
  nA = 50 →
  nB = 40 →
  WB = 70 →
  WC = 58.89 →
  50 * WA + 40 * WB = 58.89 * 90 →
  WA = 50.002 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_weight_of_section_A_l82_82104


namespace value_of_y_in_arithmetic_sequence_l82_82131

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l82_82131


namespace cos_330_eq_sqrt3_div_2_l82_82702

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82702


namespace value_of_x_squared_plus_y_squared_l82_82746

theorem value_of_x_squared_plus_y_squared
  (x y : ℝ)
  (h1 : (x + y)^2 = 4)
  (h2 : x * y = -1) :
  x^2 + y^2 = 6 :=
sorry

end value_of_x_squared_plus_y_squared_l82_82746


namespace expression_is_4045_l82_82611

theorem expression_is_4045 :
  ∃ q : ℕ, (4035 + 1) ∣ (2022 + 1) * 2022 ∧
  𝔽 2023 2022 - 𝔽 2022 2023 = 𝔽 4045 q ∧ 
  Nat.gcd 2022 q == 1 :=
by
  sorry

end expression_is_4045_l82_82611


namespace cubic_roots_cosines_l82_82195

theorem cubic_roots_cosines
  {p q r : ℝ}
  (h_eq : ∀ x : ℝ, x^3 + p * x^2 + q * x + r = 0)
  (h_roots : ∃ (α β γ : ℝ), (α > 0) ∧ (β > 0) ∧ (γ > 0) ∧ (α + β + γ = -p) ∧ 
             (α * β + β * γ + γ * α = q) ∧ (α * β * γ = -r)) :
  2 * r + 1 = p^2 - 2 * q :=
by
  sorry

end cubic_roots_cosines_l82_82195


namespace santino_total_fruits_l82_82462

theorem santino_total_fruits :
  ∃ (papaya_trees mango_trees papayas_per_tree mangos_per_tree total_fruits : ℕ),
    papaya_trees = 2 ∧
    papayas_per_tree = 10 ∧
    mango_trees = 3 ∧
    mangos_per_tree = 20 ∧
    total_fruits = (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree) ∧
    total_fruits = 80 :=
by
  -- Definitions
  let papaya_trees := 2
  let papayas_per_tree := 10
  let mango_trees := 3
  let mangos_per_tree := 20
  let total_fruits := (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree)

  -- Goal
  have : papaya_trees = 2 := rfl
  have : papayas_per_tree = 10 := rfl
  have : mango_trees = 3 := rfl
  have : mangos_per_tree = 20 := rfl
  have : total_fruits = 80 :=
    by
      calc
        total_fruits
          = (papaya_trees * papayas_per_tree) + (mango_trees * mangos_per_tree) : rfl
          ... = 20 + 60 : by simp [papaya_trees, papayas_per_tree, mango_trees, mangos_per_tree]
          ... = 80 : rfl
  exact ⟨papaya_trees, mango_trees, papayas_per_tree, mangos_per_tree, total_fruits, rfl, rfl, rfl, rfl, this⟩


end santino_total_fruits_l82_82462


namespace three_digit_integers_with_odd_factors_l82_82365

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l82_82365


namespace correct_inequality_l82_82134

def a : ℚ := -4 / 5
def b : ℚ := -3 / 4

theorem correct_inequality : a < b := 
by {
  -- Proof here
  sorry
}

end correct_inequality_l82_82134


namespace three_digit_perfect_squares_count_l82_82380

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l82_82380


namespace ab_power_2023_l82_82034

theorem ab_power_2023 (a b : ℤ) (h : |a + 2| + (b - 1) ^ 2 = 0) : (a + b) ^ 2023 = -1 :=
by
  sorry

end ab_power_2023_l82_82034


namespace num_three_digit_ints_with_odd_factors_l82_82359

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l82_82359


namespace number_of_three_digit_squares_l82_82385

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l82_82385


namespace three_digit_oddfactors_count_is_22_l82_82348

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l82_82348


namespace expected_value_Y_in_steady_state_l82_82473

-- Define the stationary functions X and Y
variables (X Y : ℝ → ℝ)

-- Define the expectations of X and Y
noncomputable def m_x : ℝ := 5
noncomputable def m_y : ℝ := 3 * m_x

-- Define the differential equation
axiom diff_eq : ∀ t, deriv Y t + 2 * Y t = 5 * deriv X t + 6 * X t

-- Define the stationarity conditions
axiom stationary_X : ∀ t, ∫ (τ : ℝ), deriv X τ = 0
axiom stationary_Y : ∀ t, ∫ (τ : ℝ), deriv Y τ = 0

-- The proof statement to be proved
theorem expected_value_Y_in_steady_state : m_y = 15 :=
  sorry

end expected_value_Y_in_steady_state_l82_82473


namespace total_pencils_correct_l82_82817

def Mitchell_pencils := 30
def Antonio_pencils := Mitchell_pencils - 6
def total_pencils := Antonio_pencils + Mitchell_pencils

theorem total_pencils_correct : total_pencils = 54 := by
  sorry

end total_pencils_correct_l82_82817


namespace overall_average_marks_is_57_l82_82762

-- Define the number of students and average mark per class
def students_class_A := 26
def avg_marks_class_A := 40

def students_class_B := 50
def avg_marks_class_B := 60

def students_class_C := 35
def avg_marks_class_C := 55

def students_class_D := 45
def avg_marks_class_D := 65

-- Define the total marks per class
def total_marks_class_A := students_class_A * avg_marks_class_A
def total_marks_class_B := students_class_B * avg_marks_class_B
def total_marks_class_C := students_class_C * avg_marks_class_C
def total_marks_class_D := students_class_D * avg_marks_class_D

-- Define the grand total of marks
def grand_total_marks := total_marks_class_A + total_marks_class_B + total_marks_class_C + total_marks_class_D

-- Define the total number of students
def total_students := students_class_A + students_class_B + students_class_C + students_class_D

-- Define the overall average marks
def overall_avg_marks := grand_total_marks / total_students

-- The target theorem we want to prove
theorem overall_average_marks_is_57 : overall_avg_marks = 57 := by
  sorry

end overall_average_marks_is_57_l82_82762


namespace positive_quadratic_if_and_only_if_l82_82453

variable (a : ℝ)
def p (x : ℝ) : ℝ := a * x^2 + 2 * x + 1

theorem positive_quadratic_if_and_only_if (h : ∀ x : ℝ, p a x > 0) : a > 1 := sorry

end positive_quadratic_if_and_only_if_l82_82453


namespace sum_of_roots_g_eq_3006_l82_82504

noncomputable def g (x : ℝ) : ℝ := 3 * x - 3 / x

theorem sum_of_roots_g_eq_3006 : 
  let T := ∑ x in {x : ℝ | g x = 3006}.to_finset in
  T = 1002 :=
 by
  sorry

end sum_of_roots_g_eq_3006_l82_82504


namespace speed_of_stream_l82_82635

-- Conditions
variables (b s : ℝ)

-- Downstream and upstream conditions
def downstream_speed := 150 = (b + s) * 5
def upstream_speed := 75 = (b - s) * 7

-- Goal statement
theorem speed_of_stream (h1 : downstream_speed b s) (h2 : upstream_speed b s) : s = 135/14 :=
by sorry

end speed_of_stream_l82_82635


namespace three_digit_integers_with_odd_number_of_factors_l82_82390

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l82_82390


namespace repair_cost_l82_82460

variable (R : ℝ)

theorem repair_cost (purchase_price transportation_charges profit_rate selling_price : ℝ) (h1 : purchase_price = 12000) (h2 : transportation_charges = 1000) (h3 : profit_rate = 0.5) (h4 : selling_price = 27000) :
  R = 5000 :=
by
  have total_cost := purchase_price + R + transportation_charges
  have selling_price_eq := 1.5 * total_cost
  have sp_eq_27000 := selling_price = 27000
  sorry

end repair_cost_l82_82460


namespace absolute_value_equation_sum_l82_82915

theorem absolute_value_equation_sum (x1 x2 : ℝ) (h1 : 3 * x1 - 12 = 6) (h2 : 3 * x2 - 12 = -6) : x1 + x2 = 8 := 
sorry

end absolute_value_equation_sum_l82_82915


namespace soccer_teams_participation_l82_82435

theorem soccer_teams_participation (total_games : ℕ) (teams_play : ℕ → ℕ) (x : ℕ) :
  (total_games = 20) → (teams_play x = x * (x - 1)) → x = 5 :=
by
  sorry

end soccer_teams_participation_l82_82435


namespace technician_round_trip_l82_82920

theorem technician_round_trip (D : ℝ) (hD : D > 0) :
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  (traveled_distance / round_trip * 100) = 65 := by
  -- Definitions based on the given conditions
  let round_trip := 2 * D
  let to_center := D
  let from_center_percent := 0.3 * D
  let traveled_distance := to_center + from_center_percent
  
  -- Placeholder for the proof to satisfy Lean syntax.
  sorry

end technician_round_trip_l82_82920


namespace find_certain_number_l82_82565

-- Define the given operation a # b
def sOperation (a b : ℝ) : ℝ :=
  a * b - b + b^2

-- State the theorem to find the value of the certain number
theorem find_certain_number (x : ℝ) (h : sOperation 3 x = 48) : x = 6 :=
sorry

end find_certain_number_l82_82565


namespace solve_equation_l82_82603

theorem solve_equation :
  ∀ (x : ℝ), x * (3 * x + 6) = 7 * (3 * x + 6) → (x = 7 ∨ x = -2) :=
by
  intro x
  sorry

end solve_equation_l82_82603


namespace ratio_of_m1_and_m2_l82_82233

theorem ratio_of_m1_and_m2 (m a b m1 m2 : ℝ) (h1 : a^2 * m - 3 * a * m + 2 * a + 7 = 0) (h2 : b^2 * m - 3 * b * m + 2 * b + 7 = 0) 
  (h3 : (a / b) + (b / a) = 2) (h4 : m1^2 * 9 - m1 * 28 + 4 = 0) (h5 : m2^2 * 9 - m2 * 28 + 4 = 0) : 
  (m1 / m2) + (m2 / m1) = 194 / 9 := 
sorry

end ratio_of_m1_and_m2_l82_82233


namespace visible_during_metaphase_l82_82456

-- Define the structures which could be present in a plant cell during mitosis.
inductive Structure
| Chromosomes
| Spindle
| CellWall
| MetaphasePlate
| CellMembrane
| Nucleus
| Nucleolus

open Structure

-- Define what structures are visible during metaphase.
def visibleStructures (phase : String) : Set Structure :=
  if phase = "metaphase" then
    {Chromosomes, Spindle, CellWall}
  else
    ∅

-- The proof statement
theorem visible_during_metaphase :
  visibleStructures "metaphase" = {Chromosomes, Spindle, CellWall} :=
by
  sorry

end visible_during_metaphase_l82_82456


namespace parking_spaces_remaining_l82_82884

-- Define the conditions as variables
variable (total_spaces : Nat := 30)
variable (spaces_per_caravan : Nat := 2)
variable (num_caravans : Nat := 3)

-- Prove the number of vehicles that can still park equals 24
theorem parking_spaces_remaining (total_spaces spaces_per_caravan num_caravans : Nat) :
    total_spaces - spaces_per_caravan * num_caravans = 24 :=
by
  -- Filling in the proof is required to fully complete this, but as per instruction we add 'sorry'
  sorry

end parking_spaces_remaining_l82_82884


namespace find_a_l82_82995

noncomputable def f (x a : ℝ) : ℝ := x^2 + real.log x - a * x

-- Assuming derivative calculation is correct
noncomputable def f_prime (x a : ℝ) : ℝ := 2 * x + 1 / x - a

-- The theorem to prove
theorem find_a (a : ℝ) :
  (f_prime 1 a = 2) ↔ (a = 1) :=
by
  simp [f_prime, f]
  sorry

end find_a_l82_82995


namespace gcd_75_100_l82_82902

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l82_82902


namespace beads_per_bracelet_l82_82298

def beads_bella_has : Nat := 36
def beads_bella_needs : Nat := 12
def total_bracelets : Nat := 6

theorem beads_per_bracelet : (beads_bella_has + beads_bella_needs) / total_bracelets = 8 :=
by
  sorry

end beads_per_bracelet_l82_82298


namespace Alex_shirt_count_l82_82515

variables (Ben_shirts Joe_shirts Alex_shirts : ℕ)

-- Conditions from the problem
def condition1 := Ben_shirts = 15
def condition2 := Ben_shirts = Joe_shirts + 8
def condition3 := Joe_shirts = Alex_shirts + 3

-- Statement to prove
theorem Alex_shirt_count : condition1 ∧ condition2 ∧ condition3 → Alex_shirts = 4 :=
by
  intros h
  sorry

end Alex_shirt_count_l82_82515


namespace find_b_l82_82996

theorem find_b (a b : ℝ) (f : ℝ → ℝ) (df : ℝ → ℝ) (x₀ : ℝ)
  (h₁ : ∀ x, f x = a * x + Real.log x)
  (h₂ : ∀ x, f x = 2 * x + b)
  (h₃ : x₀ = 1)
  (h₄ : f x₀ = a) :
  b = -1 := 
by
  sorry

end find_b_l82_82996


namespace no_14_consecutive_divisible_by_2_to_11_l82_82278

theorem no_14_consecutive_divisible_by_2_to_11 :
  ¬ ∃ (a : ℕ), ∀ i, i < 14 → ∃ p, Nat.Prime p ∧ 2 ≤ p ∧ p ≤ 11 ∧ (a + i) % p = 0 :=
by sorry

end no_14_consecutive_divisible_by_2_to_11_l82_82278


namespace probability_point_between_lines_l82_82533

theorem probability_point_between_lines {x y : ℝ} :
  (∀ x, y = -2 * x + 8) →
  (∀ x, y = -3 * x + 8) →
  0.33 = 0.33 :=
by
  intro hl hm
  sorry

end probability_point_between_lines_l82_82533


namespace river_flow_speed_l82_82153

/-- Speed of the ship in still water is 30 km/h,
    the distance traveled downstream is 144 km, and
    the distance traveled upstream is 96 km.
    Given that the time taken for both journeys is equal,
    the equation representing the speed of the river flow v is:
    144 / (30 + v) = 96 / (30 - v). -/
theorem river_flow_speed (v : ℝ) :
  (30 : ℝ) > 0 →
  real_equiv 144 (30 + v) (96 (30 - v)) := by
sorry

end river_flow_speed_l82_82153


namespace coin_loading_impossible_l82_82945

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l82_82945


namespace cos_330_eq_sqrt3_div_2_l82_82657

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82657


namespace move_digit_to_make_equation_correct_l82_82048

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end move_digit_to_make_equation_correct_l82_82048


namespace isosceles_triangle_perimeter_l82_82432

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 2) (h2 : b = 4) (h3 : a ≠ b) (h4 : a + b > b) (h5 : a + b > a) 
: ∃ p : ℝ, p = 10 :=
by
  -- Using the given conditions to determine the perimeter
  sorry

end isosceles_triangle_perimeter_l82_82432


namespace spherical_coordinates_convert_l82_82220

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end spherical_coordinates_convert_l82_82220


namespace small_stick_length_l82_82747

theorem small_stick_length 
  (x : ℝ) 
  (hx1 : 3 < x) 
  (hx2 : x < 9) 
  (hx3 : 3 + 6 > x) : 
  x = 4 := 
by 
  sorry

end small_stick_length_l82_82747


namespace fg_of_minus_three_l82_82801

-- Definitions of the functions f and g
def f (x : ℤ) : ℤ := 2 * x - 1
def g (x : ℤ) : ℤ := x * x + 4

-- The theorem to prove
theorem fg_of_minus_three : f (g (-3)) = 25 := by
  sorry

end fg_of_minus_three_l82_82801


namespace abcd_value_l82_82798

noncomputable def abcd_eval (a b c d : ℂ) : ℂ := a * b * c * d

theorem abcd_value (a b c d : ℂ) 
  (h1 : a + b + c + d = 5)
  (h2 : (5 - a)^4 + (5 - b)^4 + (5 - c)^4 + (5 - d)^4 = 125)
  (h3 : (a + b)^4 + (b + c)^4 + (c + d)^4 + (d + a)^4 + (a + c)^4 + (b + d)^4 = 1205)
  (h4 : a^4 + b^4 + c^4 + d^4 = 25) : 
  abcd_eval a b c d = 70 := 
sorry

end abcd_value_l82_82798


namespace three_digit_integers_with_odd_factors_l82_82371

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l82_82371


namespace min_blocks_to_remove_l82_82136

theorem min_blocks_to_remove (n : ℕ) (h₁ : n = 59) : ∃ k, ∃ m, (m*m*m ≤ n ∧ n < (m+1)*(m+1)*(m+1)) ∧ k = n - m*m*m ∧ k = 32 :=
by {
  sorry
}

end min_blocks_to_remove_l82_82136


namespace central_angle_of_probability_l82_82146

theorem central_angle_of_probability (x : ℝ) (h1 : x / 360 = 1 / 6) : x = 60 := by
  have h2 : x = 60 := by
    linarith
  exact h2

end central_angle_of_probability_l82_82146


namespace circle_complete_the_square_l82_82160

/-- Given the equation x^2 - 6x + y^2 - 10y + 18 = 0, show that it can be transformed to  
    (x - 3)^2 + (y - 5)^2 = 4^2 -/
theorem circle_complete_the_square :
  ∀ x y : ℝ, x^2 - 6 * x + y^2 - 10 * y + 18 = 0 ↔ (x - 3)^2 + (y - 5)^2 = 4^2 :=
by
  sorry

end circle_complete_the_square_l82_82160


namespace net_cannot_contain_2001_knots_l82_82144

theorem net_cannot_contain_2001_knots (knots : Nat) (ropes_per_knot : Nat) (total_knots : knots = 2001) (ropes_per_knot_eq : ropes_per_knot = 3) :
  false :=
by
  sorry

end net_cannot_contain_2001_knots_l82_82144


namespace asymptotes_tangent_to_circle_l82_82209

theorem asymptotes_tangent_to_circle {m : ℝ} (hm : m > 0) 
  (hyp_eq : ∀ x y : ℝ, y^2 - (x^2 / m^2) = 1) 
  (circ_eq : ∀ x y : ℝ, x^2 + y^2 - 4 * y + 3 = 0) : 
  m = (Real.sqrt 3) / 3 :=
sorry

end asymptotes_tangent_to_circle_l82_82209


namespace missing_number_l82_82002

theorem missing_number (x : ℤ) : 1234562 - 12 * x * 2 = 1234490 ↔ x = 3 :=
by
sorry

end missing_number_l82_82002


namespace cos_330_eq_sqrt3_div_2_l82_82698

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82698


namespace percent_increase_combined_cost_l82_82063

theorem percent_increase_combined_cost :
  let laptop_last_year := 500
  let tablet_last_year := 200
  let laptop_increase := 10 / 100
  let tablet_increase := 20 / 100
  let new_laptop_cost := laptop_last_year * (1 + laptop_increase)
  let new_tablet_cost := tablet_last_year * (1 + tablet_increase)
  let total_last_year := laptop_last_year + tablet_last_year
  let total_this_year := new_laptop_cost + new_tablet_cost
  let increase := total_this_year - total_last_year
  let percent_increase := (increase / total_last_year) * 100
  percent_increase = 13 :=
by
  sorry

end percent_increase_combined_cost_l82_82063


namespace relationship_x_x2_negx_l82_82022

theorem relationship_x_x2_negx (x : ℝ) (h : x^2 + x < 0) : x < x^2 ∧ x^2 < -x :=
by
  sorry

end relationship_x_x2_negx_l82_82022


namespace solution_set_l82_82086

noncomputable def satisfies_equations (x y : ℝ) : Prop :=
  (x^2 + 3 * x * y = 12) ∧ (x * y = 16 + y^2 - x * y - x^2)

theorem solution_set :
  {p : ℝ × ℝ | satisfies_equations p.1 p.2} = {(4, 1), (-4, -1), (-4, 1), (4, -1)} :=
by sorry

end solution_set_l82_82086


namespace num_three_digit_ints_with_odd_factors_l82_82360

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l82_82360


namespace stations_between_l82_82621

theorem stations_between (n : ℕ) (h : n * (n - 1) / 2 = 306) : n - 2 = 25 := 
by
  sorry

end stations_between_l82_82621


namespace range_of_a_decreasing_l82_82096

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else a / x

def is_decreasing (f : ℝ → ℝ) := ∀ x y : ℝ, x < y → f x ≥ f y

theorem range_of_a_decreasing (a : ℝ) :
  (∃ a : ℝ, (1/6) ≤ a ∧ a < (1/3)) ↔ is_decreasing (f a) :=
sorry

end range_of_a_decreasing_l82_82096


namespace value_b_2_100_l82_82585

section
open BigOperators

def sequence (b : ℕ → ℕ) : Prop :=
  (b 1 = 2) ∧ ∀ n : ℕ, b (2 * n) = (n + 1) * b n

theorem value_b_2_100 (b : ℕ → ℕ) (h : sequence b) : 
  b (2^100) = 2^2 * ∏ k in finset.range 100, (2^k + 1) := 
sorry

end

end value_b_2_100_l82_82585


namespace delaney_bus_miss_theorem_l82_82305

def delaneyMissesBus : Prop :=
  let busDeparture := 8 * 60               -- bus departure time in minutes (8:00 a.m.)
  let travelTime := 30                     -- travel time in minutes
  let departureTime := 7 * 60 + 50         -- departure time from home in minutes (7:50 a.m.)
  let arrivalTime := departureTime + travelTime -- arrival time at the pick-up point
  arrivalTime - busDeparture = 20 -- he misses the bus by 20 minutes

theorem delaney_bus_miss_theorem : delaneyMissesBus := sorry

end delaney_bus_miss_theorem_l82_82305


namespace three_digit_integers_with_odd_factors_l82_82409

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l82_82409


namespace coin_loading_impossible_l82_82933

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l82_82933


namespace three_digit_cubes_divisible_by_16_l82_82559

theorem three_digit_cubes_divisible_by_16 :
  (count (λ n : ℕ, 4 * n = n ∧ (100 ≤ (4 * n)^3 ∧ (4 * n)^3 ≤ 999)) {n | 1 ≤ n ∧ n ≤ 2}) = 1 :=
sorry

end three_digit_cubes_divisible_by_16_l82_82559


namespace find_b_from_root_and_constant_l82_82295

theorem find_b_from_root_and_constant
  (b k : ℝ)
  (h₁ : k = 44)
  (h₂ : ∃ (x : ℝ), x = 4 ∧ 2*x^2 + b*x - k = 0) :
  b = 3 :=
by
  sorry

end find_b_from_root_and_constant_l82_82295


namespace three_digit_integers_with_odd_factors_l82_82415

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l82_82415


namespace cos_330_eq_sqrt3_div_2_l82_82660

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82660


namespace polynomial_solution_l82_82064

noncomputable def P : ℝ → ℝ := sorry

theorem polynomial_solution (x : ℝ) :
  (∃ P : ℝ → ℝ, (∀ x, P x = (P 0) + (P 1) * x + (P 2) * x^2) ∧ 
  (P (-2) = 4)) →
  (P x = (4 * x^2 - 6 * x) / 7) :=
by
  sorry

end polynomial_solution_l82_82064


namespace bear_problem_l82_82527

-- Definitions of the variables
variables (W B Br : ℕ)

-- Given conditions
def condition1 : B = 2 * W := sorry
def condition2 : B = 60 := sorry
def condition3 : W + B + Br = 190 := sorry

-- The proof statement
theorem bear_problem : Br - B = 40 :=
by
  -- we would use the given conditions to prove this statement
  sorry

end bear_problem_l82_82527


namespace min_value_expression_l82_82006

theorem min_value_expression :
  ∃ x > 0, x^2 + 6 * x + 100 / x^3 = 3 * (50:ℝ)^(2/5) + 6 * (50:ℝ)^(1/5) :=
by
  sorry

end min_value_expression_l82_82006


namespace sum_of_dimensions_l82_82150

noncomputable def rectangular_prism_dimensions (A B C : ℝ) : Prop :=
  (A * B = 30) ∧ (A * C = 40) ∧ (B * C = 60)

theorem sum_of_dimensions (A B C : ℝ) (h : rectangular_prism_dimensions A B C) : A + B + C = 9 * Real.sqrt 5 :=
by
  sorry

end sum_of_dimensions_l82_82150


namespace savings_correct_l82_82577

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end savings_correct_l82_82577


namespace base8_1724_to_base10_l82_82267

/-- Define the base conversion function from base-eight to base-ten -/
def base8_to_base10 (d3 d2 d1 d0 : ℕ) : ℕ :=
  d3 * 8^3 + d2 * 8^2 + d1 * 8^1 + d0 * 8^0

/-- Base-eight representation conditions for the number 1724 -/
def base8_1724_digits := (1, 7, 2, 4)

/-- Prove the base-ten equivalent of the base-eight number 1724 is 980 -/
theorem base8_1724_to_base10 : base8_to_base10 1 7 2 4 = 980 :=
  by
    -- skipping the proof; just state that it is a theorem to be proved.
    sorry

end base8_1724_to_base10_l82_82267


namespace theater_ticket_difference_l82_82293

theorem theater_ticket_difference
  (O B V : ℕ) 
  (h₁ : O + B + V = 550) 
  (h₂ : 15 * O + 10 * B + 20 * V = 8000) : 
  B - (O + V) = 370 := 
sorry

end theater_ticket_difference_l82_82293


namespace arithmetic_seq_middle_term_l82_82122

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l82_82122


namespace arithmetic_seq_middle_term_l82_82123

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l82_82123


namespace fg_of_2_eq_513_l82_82070

def f (x : ℤ) : ℤ := x^3 + 1
def g (x : ℤ) : ℤ := 3*x + 2

theorem fg_of_2_eq_513 : f (g 2) = 513 := by
  sorry

end fg_of_2_eq_513_l82_82070


namespace find_a_b_solve_inequality_l82_82948

-- Definitions for the given conditions
def inequality1 (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 6 > 4
def sol_set1 (x : ℝ) (b : ℝ) : Prop := x < 1 ∨ x > b
def root_eq (a : ℝ) (x : ℝ) : Prop := a * x^2 - 3 * x + 2 = 0

-- The final Lean statements for the proofs
theorem find_a_b (a b : ℝ) : (∀ x, (inequality1 a x) ↔ (sol_set1 x b)) → a = 1 ∧ b = 2 :=
sorry

theorem solve_inequality (c : ℝ) : 
  (∀ x, (root_eq 1 x) ↔ (x = 1 ∨ x = 2)) → 
  (c > 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (2 < x ∧ x < c)) ∧
  (c < 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ (c < x ∧ x < 2)) ∧
  (c = 2 → ∀ x, (x^2 - (2 + c) * x + 2 * c < 0) ↔ false) :=
sorry

end find_a_b_solve_inequality_l82_82948


namespace remainder_product_l82_82872

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82872


namespace Marley_fruits_total_is_31_l82_82814

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l82_82814


namespace num_three_digit_perfect_cubes_divisible_by_16_l82_82560

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l82_82560


namespace max_jars_in_crate_l82_82501

-- Define the conditions given in the problem
def side_length_cardboard_box := 20 -- in cm
def jars_per_box := 8
def crate_width := 80 -- in cm
def crate_length := 120 -- in cm
def crate_height := 60 -- in cm
def volume_box := side_length_cardboard_box ^ 3
def volume_crate := crate_width * crate_length * crate_height
def boxes_per_crate := volume_crate / volume_box
def max_jars_per_crate := boxes_per_crate * jars_per_box

-- Statement that needs to be proved
theorem max_jars_in_crate : max_jars_per_crate = 576 := sorry

end max_jars_in_crate_l82_82501


namespace three_digit_integers_with_odd_factors_l82_82372

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l82_82372


namespace negation_of_existential_l82_82097

def divisible_by (n x : ℤ) := ∃ k : ℤ, x = k * n
def odd (x : ℤ) := ∃ k : ℤ, x = 2 * k + 1

def P (x : ℤ) := divisible_by 7 x ∧ ¬ odd x

theorem negation_of_existential :
  (¬ ∃ x : ℤ, P x) ↔ ∀ x : ℤ, divisible_by 7 x → odd x :=
by
  sorry

end negation_of_existential_l82_82097


namespace hyperbola_asymptotes_tangent_to_circle_l82_82210

theorem hyperbola_asymptotes_tangent_to_circle (m : ℝ) (h : m > 0) : 
  (∀ x y : ℝ, y^2 - (x^2 / m^2) = 1 → (x^2 + y^2 - 4*y + 3 = 0 → distance_center_to_asymptote (0, 2) (y = x / m) = 1)) → 
  m = real.sqrt(3) / 3 :=
sorry

end hyperbola_asymptotes_tangent_to_circle_l82_82210


namespace sum_of_zeros_l82_82427

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_zeros (h_symm : ∀ x, f (1 - x) = f (3 + x))
  (h_zeros : ∃ a b c, f a = 0 ∧ f b = 0 ∧ f c = 0 ∧ (a ≠ b ∧ b ≠ c ∧ c ≠ a)) :
  (∃ a b c, a + b + c = 6) :=
begin
  sorry
end

end sum_of_zeros_l82_82427


namespace income_percentage_less_l82_82073

-- Definitions representing the conditions
variables (T M J : ℝ)
variables (h1 : M = 1.60 * T) (h2 : M = 1.12 * J)

-- The theorem stating the problem
theorem income_percentage_less : (100 - (T / J) * 100) = 30 :=
by
  sorry

end income_percentage_less_l82_82073


namespace triangle_property_proof_l82_82059

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end triangle_property_proof_l82_82059


namespace problem1_problem2_l82_82980

theorem problem1 (a b x y : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < x ∧ 0 < y) : 
  (a^2 / x + b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (a * y = b * x → (a^2 / x + b^2 / y) = ((a + b)^2 / (x + y))) :=
sorry

theorem problem2 (x : ℝ) (h : 0 < x ∧ x < 1 / 2) :
  (∀ x, 0 < x ∧ x < 1 / 2 → ((2 / x + 9 / (1 - 2 * x)) ≥ 25)) ∧ (2 * (1 - 2 * (1 / 5)) = 9 * (1 / 5) → (2 / (1 / 5) + 9 / (1 - 2 * (1 / 5)) = 25)) :=
sorry

end problem1_problem2_l82_82980


namespace hajar_score_l82_82045

variables (F H : ℕ)

theorem hajar_score 
  (h1 : F - H = 21)
  (h2 : F + H = 69)
  (h3 : F > H) :
  H = 24 :=
sorry

end hajar_score_l82_82045


namespace solve_for_k_l82_82430

theorem solve_for_k (x : ℝ) (k : ℝ) (h₁ : 2 * x - 1 = 3) (h₂ : 3 * x + k = 0) : k = -6 :=
by
  sorry

end solve_for_k_l82_82430


namespace probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l82_82294

namespace Sprinters

def P_A : ℚ := 2 / 5
def P_B : ℚ := 3 / 4
def P_C : ℚ := 1 / 3

def P_all_qualified := P_A * P_B * P_C
def P_two_qualified := P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C
def P_at_least_one_qualified := 1 - (1 - P_A) * (1 - P_B) * (1 - P_C)

theorem probability_all_qualified : P_all_qualified = 1 / 10 :=
by 
  -- proof here
  sorry

theorem probability_two_qualified : P_two_qualified = 23 / 60 :=
by 
  -- proof here
  sorry

theorem probability_at_least_one_qualified : P_at_least_one_qualified = 9 / 10 :=
by 
  -- proof here
  sorry

end Sprinters

end probability_all_qualified_probability_two_qualified_probability_at_least_one_qualified_l82_82294


namespace three_digit_integers_with_odd_factors_l82_82395

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l82_82395


namespace stream_speed_l82_82633

theorem stream_speed (v : ℝ) (h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v)))) : 
  v = 5 / 3 :=
by
  -- Variables and assumptions
  have h1 : ∀ d : ℝ, d > 0 → ((1:ℝ) / (5 - v) = 2 * (1 / (5 + v))) := sorry
  -- To prove
  sorry

end stream_speed_l82_82633


namespace product_remainder_mod_7_l82_82878

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82878


namespace negation_of_proposition_l82_82474

theorem negation_of_proposition :
  ¬ (∃ x : ℝ, x < 1) ↔ ∀ x : ℝ, x ≥ 1 :=
by sorry

end negation_of_proposition_l82_82474


namespace total_pounds_of_peppers_l82_82304

def green_peppers : ℝ := 2.8333333333333335
def red_peppers : ℝ := 2.8333333333333335
def total_peppers : ℝ := 5.666666666666667

theorem total_pounds_of_peppers :
  green_peppers + red_peppers = total_peppers :=
by
  -- sorry: Proof is omitted
  sorry

end total_pounds_of_peppers_l82_82304


namespace product_remainder_mod_7_l82_82876

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82876


namespace arithmetic_sequence_50th_term_l82_82041

theorem arithmetic_sequence_50th_term :
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  a_n = 248 :=
by
  let a_1 := 3
  let d := 5
  let n := 50
  let a_n := a_1 + (n - 1) * d
  sorry

end arithmetic_sequence_50th_term_l82_82041


namespace three_digit_integers_with_odd_factors_count_l82_82388

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l82_82388


namespace cos_330_cos_30_val_answer_l82_82689

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l82_82689


namespace tan_alpha_20_l82_82206

theorem tan_alpha_20 (α : ℝ) 
  (h : Real.tan (α + 80 * Real.pi / 180) = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (α + 20 * Real.pi / 180) = Real.sqrt 3 / 7 := 
sorry

end tan_alpha_20_l82_82206


namespace sum_of_coefficients_256_l82_82026

theorem sum_of_coefficients_256 (n : ℕ) (h : (3 + 1)^n = 256) : n = 4 :=
sorry

end sum_of_coefficients_256_l82_82026


namespace three_digit_integers_with_odd_factors_l82_82414

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l82_82414


namespace solve_quadratic_equation_l82_82832

theorem solve_quadratic_equation (m : ℝ) : 9 * m^2 - (2 * m + 1)^2 = 0 → m = 1 ∨ m = -1/5 :=
by
  intro h
  sorry

end solve_quadratic_equation_l82_82832


namespace min_stamps_needed_l82_82166

theorem min_stamps_needed {c f : ℕ} (h : 3 * c + 4 * f = 33) : c + f = 9 :=
sorry

end min_stamps_needed_l82_82166


namespace weighted_avg_sales_increase_l82_82507

section SalesIncrease

/-- Define the weightages for each category last year. -/
def w_e : ℝ := 0.4
def w_c : ℝ := 0.3
def w_g : ℝ := 0.3

/-- Define the percent increases for each category this year. -/
def p_e : ℝ := 0.15
def p_c : ℝ := 0.25
def p_g : ℝ := 0.35

/-- Prove that the weighted average percent increase in sales this year is 0.24 or 24%. -/
theorem weighted_avg_sales_increase :
  ((w_e * p_e) + (w_c * p_c) + (w_g * p_g)) / (w_e + w_c + w_g) = 0.24 := 
by
  sorry

end SalesIncrease

end weighted_avg_sales_increase_l82_82507


namespace cos_330_eq_sqrt3_div_2_l82_82662

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l82_82662


namespace three_digit_oddfactors_count_l82_82356

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l82_82356


namespace three_digit_integers_with_odd_number_of_factors_count_l82_82401

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l82_82401


namespace three_digit_oddfactors_count_is_22_l82_82349

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l82_82349


namespace cos_330_eq_sqrt3_div_2_l82_82659

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82659


namespace Tim_income_percentage_less_than_Juan_l82_82592

-- Definitions for the problem
variables (T M J : ℝ)

-- Conditions based on the problem
def condition1 : Prop := M = 1.60 * T
def condition2 : Prop := M = 0.80 * J

-- Goal statement
theorem Tim_income_percentage_less_than_Juan :
  condition1 T M ∧ condition2 M J → T = 0.50 * J :=
by sorry

end Tim_income_percentage_less_than_Juan_l82_82592


namespace fraction_of_students_with_buddies_l82_82761

theorem fraction_of_students_with_buddies
  (s n : ℕ)
  (h : n / 4 = s / 2) :
  (n / 4 + s / 2) / (n + s) = 1 / 3 :=
by
  -- from the condition n / 4 = s / 2, we can derive n = 2s
  have : n = 2 * s := by
    calc
      n = 2 * s := sorry,
  -- substituting n = 2s in the required equation
  calc
    (n / 4 + s / 2) / (n + s)
        = ((2 * s) / 4 + s / 2) / ((2 * s) + s) := by rw this
    ... = (s / 2 + s / 2) / (3 * s)             := by norm_num
    ... = (s / s) * (1 / 3)                     := by rw [←div_add_div, div_self, div_eq_mul_inv]
    ... = 1 / 3                                 := by norm_num
  sorry

end fraction_of_students_with_buddies_l82_82761


namespace gymnastics_team_l82_82961

def number_of_rows (n m k : ℕ) : Prop :=
  n = k * (2 * m + k - 1) / 2

def members_in_first_row (n m k : ℕ) : Prop :=
  number_of_rows n m k ∧ 16 < k

theorem gymnastics_team : ∃ m k : ℕ, members_in_first_row 1000 m k ∧ k = 25 ∧ m = 28 :=
by
  sorry

end gymnastics_team_l82_82961


namespace ab_value_in_right_triangle_l82_82053

theorem ab_value_in_right_triangle (A B C : Type)
  [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
  (hA : ∠A = 90°) (BC : ℝ := 15) (tanC_eq_3sinC : ∀ (C : ℝ), tan C = 3 * sin C) :
  ∃ AB : ℝ, AB = (5 * Real.sqrt 6) / 4 := by
  sorry

end ab_value_in_right_triangle_l82_82053


namespace number_of_cuboids_painted_l82_82423

-- Define the problem conditions
def painted_faces (total_faces : ℕ) (faces_per_cuboid : ℕ) : ℕ :=
  total_faces / faces_per_cuboid

-- Define the theorem to prove
theorem number_of_cuboids_painted (total_faces : ℕ) (faces_per_cuboid : ℕ) :
  total_faces = 48 → faces_per_cuboid = 6 → painted_faces total_faces faces_per_cuboid = 8 :=
by
  intros h1 h2
  rw [h1, h2]
  exact rfl

end number_of_cuboids_painted_l82_82423


namespace equivalent_spherical_coords_l82_82222

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end equivalent_spherical_coords_l82_82222


namespace apples_given_by_anita_l82_82091

variable (initial_apples current_apples needed_apples : ℕ)

theorem apples_given_by_anita (h1 : initial_apples = 4) 
                               (h2 : needed_apples = 10)
                               (h3 : needed_apples - current_apples = 1) : 
  current_apples - initial_apples = 5 := 
by
  sorry

end apples_given_by_anita_l82_82091


namespace remainder_of_product_l82_82853

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82853


namespace emily_first_round_points_l82_82170

theorem emily_first_round_points (x : ℤ) 
  (second_round : ℤ := 33) 
  (last_round_loss : ℤ := 48) 
  (total_points_end : ℤ := 1) 
  (eqn : x + second_round - last_round_loss = total_points_end) : 
  x = 16 := 
by 
  sorry

end emily_first_round_points_l82_82170


namespace no_zeros_sin_log_l82_82422

open Real

theorem no_zeros_sin_log (x : ℝ) (h1 : 1 < x) (h2 : x < exp 1) : ¬ (sin (log x) = 0) :=
sorry

end no_zeros_sin_log_l82_82422


namespace least_positive_integer_l82_82913

theorem least_positive_integer (a : ℕ) :
  (a % 2 = 1) ∧ (a % 3 = 2) ∧ (a % 4 = 3) ∧ (a % 5 = 4) → a = 59 :=
by
  sorry

end least_positive_integer_l82_82913


namespace target_destroyed_probability_l82_82309

noncomputable def probability_hit (p1 p2 p3 : ℝ) : ℝ :=
  let miss1 := 1 - p1
  let miss2 := 1 - p2
  let miss3 := 1 - p3
  let prob_all_miss := miss1 * miss2 * miss3
  let prob_one_hit := (p1 * miss2 * miss3) + (miss1 * p2 * miss3) + (miss1 * miss2 * p3)
  let prob_destroyed := 1 - (prob_all_miss + prob_one_hit)
  prob_destroyed

theorem target_destroyed_probability :
  probability_hit 0.9 0.9 0.8 = 0.954 :=
sorry

end target_destroyed_probability_l82_82309


namespace remainder_of_product_mod_7_l82_82860

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82860


namespace Jackie_has_more_apples_l82_82512

def Adam_apples : Nat := 9
def Jackie_apples : Nat := 10

theorem Jackie_has_more_apples : Jackie_apples - Adam_apples = 1 := by
  sorry

end Jackie_has_more_apples_l82_82512


namespace cos_330_eq_sqrt3_over_2_l82_82669

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l82_82669


namespace three_digit_integers_with_odd_factors_l82_82416

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l82_82416


namespace product_mod_7_l82_82844

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82844


namespace original_price_l82_82499

theorem original_price (SP : ℝ) (gain_percent : ℝ) (P : ℝ) : SP = 1080 → gain_percent = 0.08 → SP = P * (1 + gain_percent) → P = 1000 :=
by
  intro hSP hGainPercent hEquation
  sorry

end original_price_l82_82499


namespace china_GDP_in_2016_l82_82042

noncomputable def GDP_2016 (a r : ℝ) : ℝ := a * (1 + r / 100)^5

theorem china_GDP_in_2016 (a r : ℝ) :
  GDP_2016 a r = a * (1 + r / 100)^5 :=
by
  -- proof
  sorry

end china_GDP_in_2016_l82_82042


namespace find_x_l82_82139

theorem find_x (x : ℝ) (h: 0.8 * 90 = 70 / 100 * x + 30) : x = 60 :=
by
  sorry

end find_x_l82_82139


namespace words_difference_l82_82074

-- Definitions based on conditions.
def right_hand_speed (words_per_minute : ℕ) := 10
def left_hand_speed (words_per_minute : ℕ) := 7
def time_duration (minutes : ℕ) := 5

-- Problem statement
theorem words_difference :
  let right_hand_words := right_hand_speed 0 * time_duration 0
  let left_hand_words := left_hand_speed 0 * time_duration 0
  (right_hand_words - left_hand_words) = 15 :=
by
  sorry

end words_difference_l82_82074


namespace largest_divisor_of_product_l82_82624

theorem largest_divisor_of_product (n : ℕ) (h : n % 3 = 0) : ∃ d, d = 288 ∧ ∀ n (h : n % 3 = 0), d ∣ (n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) := 
sorry

end largest_divisor_of_product_l82_82624


namespace geometric_sequence_value_a6_l82_82759

theorem geometric_sequence_value_a6
    (q a1 : ℝ) (a : ℕ → ℝ)
    (h1 : ∀ n, a n = a1 * q ^ (n - 1))
    (h2 : a 2 = 1)
    (h3 : a 8 = a 6 + 2 * a 4)
    (h4 : q > 0)
    (h5 : ∀ n, a n > 0) : 
    a 6 = 4 :=
by
  sorry

end geometric_sequence_value_a6_l82_82759


namespace alex_shirts_l82_82513

theorem alex_shirts (shirts_joe shirts_alex shirts_ben : ℕ) 
  (h1 : shirts_joe = shirts_alex + 3) 
  (h2 : shirts_ben = shirts_joe + 8) 
  (h3 : shirts_ben = 15) : shirts_alex = 4 :=
by
  sorry

end alex_shirts_l82_82513


namespace factor_expression_l82_82652

theorem factor_expression (x : ℝ) : 
  ((4 * x^3 + 64 * x^2 - 8) - (-6 * x^3 + 2 * x^2 - 8)) = 2 * x^2 * (5 * x + 31) := 
by sorry

end factor_expression_l82_82652


namespace num_odd_factors_of_three_digit_integers_eq_22_l82_82411

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l82_82411


namespace smallest_square_area_l82_82790

variable (M N : ℝ)

/-- Given that the largest square has an area of 1 cm^2, the middle square has an area M cm^2, and the smallest square has a vertex on the side of the middle square, prove that the area of the smallest square N is equal to ((1 - M) / 2)^2. -/
theorem smallest_square_area (h1 : 1 ≥ 0)
  (h2 : 0 ≤ M ∧ M ≤ 1)
  (h3 : 0 ≤ N) :
  N = (1 - M) ^ 2 / 4 := sorry

end smallest_square_area_l82_82790


namespace min_buses_needed_l82_82282

-- Given definitions from conditions
def students_per_bus : ℕ := 45
def total_students : ℕ := 495

-- The proposition to prove
theorem min_buses_needed : ∃ n : ℕ, 45 * n ≥ 495 ∧ (∀ m : ℕ, 45 * m ≥ 495 → n ≤ m) :=
by
  -- Preliminary calculations that lead to the solution
  let n := total_students / students_per_bus
  have h : total_students % students_per_bus = 0 := by sorry
  
  -- Conclude that the minimum n so that 45 * n ≥ 495 is indeed 11
  exact ⟨n, by sorry, by sorry⟩

end min_buses_needed_l82_82282


namespace T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l82_82444

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

end T_0_2006_correct_T_1_2006_correct_T_2_2006_correct_l82_82444


namespace product_mod_7_l82_82842

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82842


namespace simplify_expression_l82_82826

theorem simplify_expression (t : ℝ) : (t ^ 5 * t ^ 3) / t ^ 2 = t ^ 6 :=
by
  sorry

end simplify_expression_l82_82826


namespace inverse_proposition_l82_82489

theorem inverse_proposition (L₁ L₂ : Line) (a₁ a₂ : Angle) :
  -- Condition: If L₁ is parallel to L₂, then alternate interior angles are equal
  (L₁ ∥ L₂ → a₁ = a₂) →
  -- Proposition to prove: If alternate interior angles are equal, then L₁ is parallel to L₂
  (a₁ = a₂ → L₁ ∥ L₂) :=
by
  sorry

end inverse_proposition_l82_82489


namespace exp_value_l82_82726

theorem exp_value (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 3) : a^(m + 2 * n) = 18 := 
by
  sorry

end exp_value_l82_82726


namespace impossible_load_two_coins_l82_82929

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l82_82929


namespace mask_production_l82_82043

theorem mask_production (x : ℝ) :
  24 + 24 * (1 + x) + 24 * (1 + x)^2 = 88 :=
sorry

end mask_production_l82_82043


namespace april_plant_arrangement_l82_82963

theorem april_plant_arrangement :
    let nBasil := 5
    let nTomato := 4
    let nPairs := nTomato / 2
    let nUnits := nBasil + nPairs
    let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
    totalWays = 20160 := by
{
  let nBasil := 5
  let nTomato := 4
  let nPairs := nTomato / 2
  let nUnits := nBasil + nPairs
  let totalWays := (Nat.factorial nUnits) * (Nat.factorial nPairs) * (Nat.factorial (nPairs - 1))
  sorry
}

end april_plant_arrangement_l82_82963


namespace coin_loading_impossible_l82_82925

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l82_82925


namespace heath_average_carrots_per_hour_l82_82345

theorem heath_average_carrots_per_hour 
  (rows1 rows2 : ℕ)
  (plants_per_row1 plants_per_row2 : ℕ)
  (hours1 hours2 : ℕ)
  (h1 : rows1 = 200)
  (h2 : rows2 = 200)
  (h3 : plants_per_row1 = 275)
  (h4 : plants_per_row2 = 325)
  (h5 : hours1 = 15)
  (h6 : hours2 = 25) :
  ((rows1 * plants_per_row1 + rows2 * plants_per_row2) / (hours1 + hours2) = 3000) :=
  by
  sorry

end heath_average_carrots_per_hour_l82_82345


namespace three_digit_integers_with_odd_factors_count_l82_82386

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l82_82386


namespace isosceles_triangle_area_l82_82614

open Real

noncomputable def area_of_isosceles_triangle (b : ℝ) (h : ℝ) : ℝ :=
  (1/2) * b * h

theorem isosceles_triangle_area :
  ∃ (b : ℝ) (l : ℝ), h = 8 ∧ (2 * l + b = 32) ∧ (area_of_isosceles_triangle b h = 48) :=
by
  sorry

end isosceles_triangle_area_l82_82614


namespace part1_part2_part3_l82_82541

def pointM (m : ℝ) : ℝ × ℝ := (m - 1, 2 * m + 3)

-- Part 1
theorem part1 (m : ℝ) (h : 2 * m + 3 = 0) : pointM m = (-5 / 2, 0) :=
  sorry

-- Part 2
theorem part2 (m : ℝ) (h : 2 * m + 3 = -1) : pointM m = (-3, -1) :=
  sorry

-- Part 3
theorem part3 (m : ℝ) (h1 : |m - 1| = 2) : pointM m = (2, 9) ∨ pointM m = (-2, 1) :=
  sorry

end part1_part2_part3_l82_82541


namespace woman_lawyer_probability_l82_82949

theorem woman_lawyer_probability (total_members women_count lawyer_prob : ℝ) 
  (h1: total_members = 100) 
  (h2: women_count = 0.70 * total_members) 
  (h3: lawyer_prob = 0.40) : 
  (0.40 * 0.70) = 0.28 := by sorry

end woman_lawyer_probability_l82_82949


namespace sum_reciprocal_gt_log_l82_82200

theorem sum_reciprocal_gt_log (n : ℕ) (h : ∀ x ≥ 0, Real.exp x ≥ x + 1 ∧ (x = 0 → Real.exp x = x + 1)) :
  ∑ k in Finset.range n, (1 / (k + 1)) > Real.log (n + 1) :=
sorry

end sum_reciprocal_gt_log_l82_82200


namespace ab_cd_l82_82630

theorem ab_cd {a b c d : ℕ} {w x y z : ℕ}
  (hw : Prime w) (hx : Prime x) (hy : Prime y) (hz : Prime z)
  (horder : w < x ∧ x < y ∧ y < z)
  (hprod : w^a * x^b * y^c * z^d = 660) :
  (a + b) - (c + d) = 1 :=
by
  sorry

end ab_cd_l82_82630


namespace thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l82_82970

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem thirtieth_triangular_number : triangular_number 30 = 465 := 
by
  sorry

theorem sum_of_thirtieth_and_twentyninth_triangular_numbers : triangular_number 30 + triangular_number 29 = 900 := 
by
  sorry

end thirtieth_triangular_number_sum_of_thirtieth_and_twentyninth_triangular_numbers_l82_82970


namespace linear_system_solution_l82_82990

theorem linear_system_solution (a b : ℝ) 
  (h1 : 3 * a + 2 * b = 5) 
  (h2 : 2 * a + 3 * b = 4) : 
  a - b = 1 := 
by
  sorry

end linear_system_solution_l82_82990


namespace Marley_fruits_total_is_31_l82_82815

-- Define the given conditions

def Louis_oranges : Nat := 5
def Louis_apples : Nat := 3
def Samantha_oranges : Nat := 8
def Samantha_apples : Nat := 7

def Marley_oranges : Nat := 2 * Louis_oranges
def Marley_apples : Nat := 3 * Samantha_apples

-- The statement to be proved
def Marley_total_fruits : Nat := Marley_oranges + Marley_apples

theorem Marley_fruits_total_is_31 : Marley_total_fruits = 31 := by
  sorry

end Marley_fruits_total_is_31_l82_82815


namespace calculate_expression_l82_82983

theorem calculate_expression (a b c : ℝ) (A B C : ℝ)
  (hA : A ≠ 0 ∧ A ≠ π)
  (hB : B ≠ 0 ∧ B ≠ π)
  (hC : C ≠ 0 ∧ C ≠ π)
  (ha : a = 2)
  (hb : b = 3)
  (hc : c = 4)
  (cos_rule_A : c ^ 2 = a ^ 2 + b ^ 2 - 2 * a * b * Real.cos A)
  (cos_rule_B : b ^ 2 = a ^ 2 + c ^ 2 - 2 * a * c * Real.cos B)
  (cos_rule_C : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos C) :
  2 * b * c * Real.cos A + 2 * c * a * Real.cos B + 2 * a * b * Real.cos C = 29 :=
by
  sorry

end calculate_expression_l82_82983


namespace radius_of_inscribed_circle_l82_82570

variable (A p s r : ℝ)

theorem radius_of_inscribed_circle (h1 : A = 2 * p) (h2 : A = r * s) (h3 : p = 2 * s) : r = 4 := by
  sorry

end radius_of_inscribed_circle_l82_82570


namespace Ruth_school_hours_l82_82248

theorem Ruth_school_hours (d : ℝ) :
  0.25 * 5 * d = 10 → d = 8 :=
by
  sorry

end Ruth_school_hours_l82_82248


namespace find_x_in_acute_triangle_l82_82730

-- Definition of an acute triangle with given segment lengths due to altitudes
def acute_triangle_with_segments (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) : Prop :=
  BC = 4 + x ∧ AE = x ∧ BE = 8 ∧ (A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- The theorem to prove
theorem find_x_in_acute_triangle (A B C D E : Type) (BC AE BE : ℝ) (x : ℝ) 
  (h : acute_triangle_with_segments A B C D E BC AE BE x) : 
  x = 4 :=
by
  -- As the focus is on the statement, we add sorry to skip the proof.
  sorry

end find_x_in_acute_triangle_l82_82730


namespace find_number_l82_82916

theorem find_number (x : ℕ) : ((x * 12) / (180 / 3) + 70 = 71) → x = 5 :=
by
  sorry

end find_number_l82_82916


namespace find_constant_x_geom_prog_l82_82971

theorem find_constant_x_geom_prog (x : ℝ) :
  (30 + x) ^ 2 = (10 + x) * (90 + x) → x = 0 :=
by
  -- Proof omitted
  sorry

end find_constant_x_geom_prog_l82_82971


namespace graph_properties_l82_82553

noncomputable def f (x : ℝ) : ℝ := (x^2 - 5*x + 6) / (x - 1)

theorem graph_properties :
  (∀ x, x ≠ 1 → f x = (x-2)*(x-3)/(x-1)) ∧
  (∃ x, f x = 0 ∧ (x = 2 ∨ x = 3)) ∧
  (∀ ε > 0, ∃ δ > 0, ∀ x, abs (x - 1) < δ → abs (f x) > ε) ∧
  ((∀ ε > 0, ∃ M > 0, ∀ x > M, f x > ε) ∧ (∀ ε > 0, ∃ M < 0, ∀ x < M, f x < -ε)) := sorry

end graph_properties_l82_82553


namespace cos_330_eq_sqrt3_div_2_l82_82696

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82696


namespace coin_loading_impossible_l82_82942

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l82_82942


namespace complement_intersect_l82_82800

def U : Set ℤ := {-3, -2, -1, 0, 1, 2, 3}
def A : Set ℤ := {x | x^2 - 1 ≤ 0}
def B : Set ℤ := {x | -1 ≤ x ∧ x ≤ 3}
def C : Set ℤ := {x | x ∉ A ∧ x ∈ U} -- complement of A in U

theorem complement_intersect (U A B : Set ℤ) :
  (C ∩ B) = {2, 3} :=
by
  sorry

end complement_intersect_l82_82800


namespace B_time_to_finish_race_l82_82434

theorem B_time_to_finish_race (t : ℝ) 
  (race_distance : ℝ := 130)
  (A_time : ℝ := 36)
  (A_beats_B_by : ℝ := 26)
  (A_speed : ℝ := race_distance / A_time) 
  (B_distance_when_A_finishes : ℝ := race_distance - A_beats_B_by) 
  (B_speed := B_distance_when_A_finishes / t) :
  B_speed * (t - A_time) = A_beats_B_by → t = 48 := 
by
  intros h
  sorry

end B_time_to_finish_race_l82_82434


namespace zero_in_interval_l82_82989

noncomputable def f (x : ℝ) : ℝ := (1/2)^x - x^(1/3)

theorem zero_in_interval : ∃ x ∈ Set.Ioo (1/3 : ℝ) (1/2 : ℝ), f x = 0 :=
by
  -- The correct statement only
  sorry

end zero_in_interval_l82_82989


namespace relationship_t_s_l82_82441

variable {a b : ℝ}

theorem relationship_t_s (a b : ℝ) (t : ℝ) (s : ℝ) (ht : t = a + 2 * b) (hs : s = a + b^2 + 1) :
  t ≤ s := 
sorry

end relationship_t_s_l82_82441


namespace find_f_at_3_l82_82213

theorem find_f_at_3 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (2 * x + 1) = x ^ 2 - 2 * x) : f 3 = -1 :=
by {
  -- Proof would go here.
  sorry
}

end find_f_at_3_l82_82213


namespace remainder_correct_l82_82625

def dividend : ℕ := 165
def divisor : ℕ := 18
def quotient : ℕ := 9
def remainder : ℕ := 3

theorem remainder_correct {d q r : ℕ} (h1 : d = dividend) (h2 : q = quotient) (h3 : r = divisor * q) : d = 165 → q = 9 → 165 = 162 + remainder :=
by { sorry }

end remainder_correct_l82_82625


namespace evaluate_expression_l82_82111

theorem evaluate_expression : 
  60 + 120 / 15 + 25 * 16 - 220 - 420 / 7 + 3 ^ 2 = 197 :=
by
  sorry

end evaluate_expression_l82_82111


namespace gcd_75_100_l82_82895

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l82_82895


namespace attendees_chose_water_l82_82162

theorem attendees_chose_water
  (total_attendees : ℕ)
  (juice_percentage water_percentage : ℝ)
  (attendees_juice : ℕ)
  (h1 : juice_percentage = 0.7)
  (h2 : water_percentage = 0.3)
  (h3 : attendees_juice = 140)
  (h4 : total_attendees * juice_percentage = attendees_juice)
  : total_attendees * water_percentage = 60 := by
  sorry

end attendees_chose_water_l82_82162


namespace min_value_frac_sum_l82_82752

theorem min_value_frac_sum {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : 4 * a + b = 1): 
  (1 / (2 * a) + 2 / b) = 8 :=
sorry

end min_value_frac_sum_l82_82752


namespace ellipse_equation_l82_82186

theorem ellipse_equation (a b : ℝ) (A : ℝ × ℝ)
  (hA : A = (-3, 1.75))
  (he : 0.75 = Real.sqrt (a^2 - b^2) / a) 
  (hcond : (Real.sqrt (a^2 - b^2) / a) = 0.75) :
  (16 = a^2) ∧ (7 = b^2) :=
by
  have h1 : A = (-3, 1.75) := hA
  have h2 : Real.sqrt (a^2 - b^2) / a = 0.75 := hcond
  sorry

end ellipse_equation_l82_82186


namespace square_side_increase_l82_82259

theorem square_side_increase (p : ℝ) (h : (1 + p / 100)^2 = 1.69) : p = 30 :=
by {
  sorry
}

end square_side_increase_l82_82259


namespace additional_votes_in_revote_l82_82216

theorem additional_votes_in_revote (a b a' b' n : ℕ) :
  a + b = 300 →
  b - a = n →
  a' - b' = 3 * n →
  a' + b' = 300 →
  a' = (7 * b) / 6 →
  a' - a = 55 :=
by 
  intros h1 h2 h3 h4 h5
  sorry

end additional_votes_in_revote_l82_82216


namespace total_packs_equiv_117_l82_82821

theorem total_packs_equiv_117 
  (nancy_cards : ℕ)
  (melanie_cards : ℕ)
  (mary_cards : ℕ)
  (alyssa_cards : ℕ)
  (nancy_pack : ℝ)
  (melanie_pack : ℝ)
  (mary_pack : ℝ)
  (alyssa_pack : ℝ)
  (H_nancy : nancy_cards = 540)
  (H_melanie : melanie_cards = 620)
  (H_mary : mary_cards = 480)
  (H_alyssa : alyssa_cards = 720)
  (H_nancy_pack : nancy_pack = 18.5)
  (H_melanie_pack : melanie_pack = 22.5)
  (H_mary_pack : mary_pack = 15.3)
  (H_alyssa_pack : alyssa_pack = 24) :
  (⌊nancy_cards / nancy_pack⌋₊ + ⌊melanie_cards / melanie_pack⌋₊ + ⌊mary_cards / mary_pack⌋₊ + ⌊alyssa_cards / alyssa_pack⌋₊) = 117 :=
by
  sorry

end total_packs_equiv_117_l82_82821


namespace remainder_of_product_l82_82856

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82856


namespace export_volume_scientific_notation_l82_82511

theorem export_volume_scientific_notation :
  (234.1 * 10^6) = (2.341 * 10^8) := 
sorry

end export_volume_scientific_notation_l82_82511


namespace find_principal_l82_82492

noncomputable def principal_amount (P : ℝ) (r : ℝ) : Prop :=
  (800 = (P * r * 2) / 100) ∧ (820 = P * (1 + r / 100)^2 - P)

theorem find_principal (P : ℝ) (r : ℝ) (h : principal_amount P r) : P = 8000 :=
by
  sorry

end find_principal_l82_82492


namespace find_x_l82_82564

theorem find_x (x y : ℝ) (h1 : 0.65 * x = 0.20 * y)
  (h2 : y = 617.5 ^ 2 - 42) : 
  x = 117374.3846153846 :=
by
  sorry

end find_x_l82_82564


namespace number_of_three_digit_integers_with_odd_number_of_factors_l82_82403

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l82_82403


namespace add_fractions_l82_82158

theorem add_fractions :
  (11 / 12) + (7 / 8) + (3 / 4) = 61 / 24 :=
by
  sorry

end add_fractions_l82_82158


namespace calculation_result_l82_82647

theorem calculation_result : 7 * (9 + 2 / 5) + 3 = 68.8 :=
by
  sorry

end calculation_result_l82_82647


namespace digit_product_equality_l82_82622

theorem digit_product_equality (x y z : ℕ) (hx : x = 3) (hy : y = 7) (hz : z = 1) :
  x * (10 * x + y) = 111 * z :=
by
  -- Using hx, hy, and hz, the proof can proceed from here
  sorry

end digit_product_equality_l82_82622


namespace sum_of_coordinates_A_l82_82787

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l82_82787


namespace average_height_correct_l82_82466

noncomputable def initially_calculated_average_height 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) 
  (A : ℝ) : Prop :=
  let incorrect_sum := num_students * A
  let height_difference := incorrect_height - correct_height
  let actual_sum := num_students * actual_average
  incorrect_sum = actual_sum + height_difference

theorem average_height_correct 
  (num_students : ℕ) (incorrect_height correct_height : ℕ) 
  (actual_average : ℝ) :
  initially_calculated_average_height num_students incorrect_height correct_height actual_average 175 :=
by {
  sorry
}

end average_height_correct_l82_82466


namespace find_c_plus_one_over_b_l82_82089

theorem find_c_plus_one_over_b 
  (a b c : ℝ) 
  (a_pos : 0 < a) 
  (b_pos : 0 < b) 
  (c_pos : 0 < c) 
  (h1 : a * b * c = 1) 
  (h2 : a + 1 / c = 8) 
  (h3 : b + 1 / a = 20) : 
  c + 1 / b = 10 / 53 := 
sorry

end find_c_plus_one_over_b_l82_82089


namespace arithmetic_sequence_middle_term_l82_82126

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l82_82126


namespace min_distance_PQ_l82_82224

theorem min_distance_PQ :
  ∀ (P Q : ℝ × ℝ), (P.1 - P.2 - 4 = 0) → (Q.1^2 = 4 * Q.2) →
  ∃ (d : ℝ), d = dist P Q ∧ d = 3 * Real.sqrt 2 / 2 :=
sorry

end min_distance_PQ_l82_82224


namespace calculate_difference_l82_82452

def g (x : ℝ) : ℝ := 3 * x^2 + 4 * x + 5

theorem calculate_difference (x h : ℝ) : g (x + h) - g x = h * (6 * x + 3 * h + 4) :=
by
  sorry

end calculate_difference_l82_82452


namespace overlapping_area_of_rectangular_strips_l82_82110

theorem overlapping_area_of_rectangular_strips (theta : ℝ) (h_theta : theta ≠ 0) :
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  area = 2 / Real.sin theta :=
by
  let width := 2
  let diag_1 := width
  let diag_2 := width / Real.sin theta
  let area := (diag_1 * diag_2) / 2
  sorry

end overlapping_area_of_rectangular_strips_l82_82110


namespace hiker_total_distance_l82_82148

-- Define conditions based on the problem description
def day1_distance : ℕ := 18
def day1_speed : ℕ := 3
def day2_speed : ℕ := day1_speed + 1
def day1_time : ℕ := day1_distance / day1_speed
def day2_time : ℕ := day1_time - 1
def day3_speed : ℕ := 5
def day3_time : ℕ := 3

-- Define the total distance walked based on the conditions
def total_distance : ℕ :=
  day1_distance + (day2_speed * day2_time) + (day3_speed * day3_time)

-- The theorem stating the hiker walked a total of 53 miles
theorem hiker_total_distance : total_distance = 53 := by
  sorry

end hiker_total_distance_l82_82148


namespace expand_product_l82_82717

theorem expand_product (x : ℝ) :
  (x + 4) * (x - 5) = x^2 - x - 20 :=
by
  -- The proof will use algebraic identities and simplifications.
  sorry

end expand_product_l82_82717


namespace Marley_fruit_count_l82_82808

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l82_82808


namespace number_of_three_digit_squares_l82_82382

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l82_82382


namespace kevin_hop_distance_l82_82581

theorem kevin_hop_distance :
  (1/4) + (3/16) + (9/64) + (27/256) + (81/1024) + (243/4096) = 3367 / 4096 := 
by
  sorry 

end kevin_hop_distance_l82_82581


namespace cos_330_eq_sqrt3_div_2_l82_82685

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82685


namespace car_speed_second_hour_l82_82617

theorem car_speed_second_hour
  (speed_first_hour : ℝ)
  (avg_speed : ℝ)
  (hours : ℝ)
  (total_time : ℝ)
  (total_distance : ℝ)
  (distance_first_hour : ℝ)
  (distance_second_hour : ℝ) :
  speed_first_hour = 90 →
  avg_speed = 75 →
  hours = 2 →
  total_time = hours →
  total_distance = avg_speed * total_time →
  distance_first_hour = speed_first_hour * 1 →
  distance_second_hour = total_distance - distance_first_hour →
  distance_second_hour / 1 = 60 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end car_speed_second_hour_l82_82617


namespace expression_value_l82_82839

theorem expression_value (a b : ℝ) (h : a^2 * b^2 / (a^4 - 2 * b^4) = 1) : 
  (a^2 - b^2) / (a^2 + b^2) = 1 / 3 := 
by 
  sorry

end expression_value_l82_82839


namespace hyperbola_real_axis_length_l82_82029

theorem hyperbola_real_axis_length
    (a b : ℝ) 
    (h_pos_a : a > 0) 
    (h_pos_b : b > 0) 
    (h_eccentricity : a * Real.sqrt 5 = Real.sqrt (a^2 + b^2))
    (h_distance : b * a * Real.sqrt 5 / Real.sqrt (a^2 + b^2) = 8) :
    2 * a = 8 :=
sorry

end hyperbola_real_axis_length_l82_82029


namespace percentage_increase_l82_82445

theorem percentage_increase (x : ℝ) : 
  (1 + x / 100)^2 = 1.1025 → x = 5.024 := 
sorry

end percentage_increase_l82_82445


namespace total_votes_proof_l82_82822

noncomputable def total_votes (A : ℝ) (T : ℝ) := 0.40 * T = A
noncomputable def votes_in_favor (A : ℝ) := A + 68
noncomputable def total_votes_calc (T : ℝ) (Favor : ℝ) (A : ℝ) := T = Favor + A

theorem total_votes_proof (A T : ℝ) (Favor : ℝ) 
  (hA : total_votes A T) 
  (hFavor : votes_in_favor A = Favor) 
  (hT : total_votes_calc T Favor A) : 
  T = 340 :=
by
  sorry

end total_votes_proof_l82_82822


namespace distance_B_amusement_park_l82_82479

variable (d_A d_B v_A v_B t_A t_B : ℝ)

axiom h1 : v_A = 3
axiom h2 : v_B = 4
axiom h3 : d_B = d_A + 2
axiom h4 : t_A + t_B = 4
axiom h5 : t_A = d_A / v_A
axiom h6 : t_B = d_B / v_B

theorem distance_B_amusement_park:
  d_A / 3 + (d_A + 2) / 4 = 4 → d_B = 8 :=
by
  sorry

end distance_B_amusement_park_l82_82479


namespace arccos_cos_solution_l82_82463

theorem arccos_cos_solution (x : ℝ) (h₀ : 0 ≤ x) (h₁ : x ≤ (Real.pi / 2)) (h₂ : Real.arccos (Real.cos x) = 2 * x) : 
    x = 0 :=
by 
  sorry

end arccos_cos_solution_l82_82463


namespace cos_330_eq_sqrt3_div_2_l82_82688

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82688


namespace cos_330_eq_sqrt3_div_2_l82_82683

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l82_82683


namespace move_digit_to_make_equation_correct_l82_82049

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end move_digit_to_make_equation_correct_l82_82049


namespace range_of_a_l82_82196

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, ¬ ((a^2 - 4) * x^2 + (a + 2) * x - 1 ≥ 0)) →
  (-2:ℝ) ≤ a ∧ a < (6 / 5:ℝ) :=
by
  sorry

end range_of_a_l82_82196


namespace min_value_f_l82_82179

open Real

noncomputable def f (x : ℝ) : ℝ :=
  sqrt (15 - 12 * cos x) + 
  sqrt (4 - 2 * sqrt 3 * sin x) +
  sqrt (7 - 4 * sqrt 3 * sin x) +
  sqrt (10 - 4 * sqrt 3 * sin x - 6 * cos x)

theorem min_value_f : ∃ x : ℝ, f x = 6 := 
sorry

end min_value_f_l82_82179


namespace total_handshakes_l82_82296

-- Define the groups and their properties
def GroupA := 30
def GroupB := 15
def GroupC := 5
def KnowEachOtherA := true -- All 30 people in Group A know each other
def KnowFromB := 10 -- Each person in Group B knows 10 people from Group A
def KnowNoOneC := true -- Each person in Group C knows no one

-- Define the number of handshakes based on the conditions
def handshakes_between_A_and_B : Nat := GroupB * (GroupA - KnowFromB)
def handshakes_between_B_and_C : Nat := GroupB * GroupC
def handshakes_within_C : Nat := (GroupC * (GroupC - 1)) / 2
def handshakes_between_A_and_C : Nat := GroupA * GroupC

-- Prove the total number of handshakes
theorem total_handshakes : 
  handshakes_between_A_and_B +
  handshakes_between_B_and_C +
  handshakes_within_C +
  handshakes_between_A_and_C = 535 :=
by sorry

end total_handshakes_l82_82296


namespace f_bound_l82_82985

noncomputable def f : ℕ+ → ℝ := sorry

axiom f_1 : f 1 = 3 / 2
axiom f_ineq (x y : ℕ+) : f (x + y) ≥ (1 + y / (x + 1)) * f x + (1 + x / (y + 1)) * f y + x^2 * y + x * y + x * y^2

theorem f_bound (x : ℕ+) : f x ≥ 1 / 4 * x * (x + 1) * (2 * x + 1) := sorry

end f_bound_l82_82985


namespace three_digit_integers_with_odd_number_of_factors_l82_82392

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l82_82392


namespace total_legs_is_26_l82_82302

-- Define the number of puppies and chicks
def number_of_puppies : Nat := 3
def number_of_chicks : Nat := 7

-- Define the number of legs per puppy and per chick
def legs_per_puppy : Nat := 4
def legs_per_chick : Nat := 2

-- Calculate the total number of legs
def total_legs := (number_of_puppies * legs_per_puppy) + (number_of_chicks * legs_per_chick)

-- Prove that the total number of legs is 26
theorem total_legs_is_26 : total_legs = 26 := by
  sorry

end total_legs_is_26_l82_82302


namespace find_values_of_a_to_make_lines_skew_l82_82718

noncomputable def lines_are_skew (t u a : ℝ) : Prop :=
  ∀ t u,
    (1 + 2 * t = 4 + 5 * u ∧
     2 + 3 * t = 1 + 2 * u ∧
     a + 4 * t = u) → false

theorem find_values_of_a_to_make_lines_skew :
  ∀ a : ℝ, ¬ a = 3 ↔ lines_are_skew t u a :=
by
  sorry

end find_values_of_a_to_make_lines_skew_l82_82718


namespace tangent_lines_count_l82_82244

noncomputable def number_of_tangent_lines (r1 r2 : ℝ) (k : ℕ) : ℕ :=
if r1 = 2 ∧ r2 = 3 then 5 else 0

theorem tangent_lines_count: 
∃ k : ℕ, number_of_tangent_lines 2 3 k = 5 :=
by sorry

end tangent_lines_count_l82_82244


namespace arithmetic_seq_middle_term_l82_82124

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l82_82124


namespace problem_l82_82997

variable {f : ℝ → ℝ}

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≥ f y
def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop := ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y
def max_value_in (f : ℝ → ℝ) (a b : ℝ) (v : ℝ) : Prop := ∀ x, a ≤ x → x ≤ b → f x ≤ v ∧ (∃ z, a ≤ z ∧ z ≤ b ∧ f z = v)

theorem problem
  (h_even : even_function f)
  (h_decreasing : decreasing_on f (-5) (-2))
  (h_max : max_value_in f (-5) (-2) 7) :
  increasing_on f 2 5 ∧ max_value_in f 2 5 7 :=
by
  sorry

end problem_l82_82997


namespace car_speed_l82_82490

theorem car_speed (v : ℝ) (h1 : 1 / 900 * 3600 = 4) (h2 : 1 / v * 3600 = 6) : v = 600 :=
by
  sorry

end car_speed_l82_82490


namespace distinguishable_squares_count_l82_82889

theorem distinguishable_squares_count :
  let colors := 5  -- Number of different colors
  let total_corner_sets :=
    5 + -- All four corners the same color
    5 * 4 + -- Three corners the same color
    Nat.choose 5 2 * 2 + -- Two pairs of corners with the same color
    5 * 4 * 3 * 2 -- All four corners different
  let total_corner_together := total_corner_sets
  let total := 
    (4 * 5 + -- One corner color used
    3 * (5 * 4 + Nat.choose 5 2 * 2) + -- Two corner colors used
    2 * (5 * 4 * 3 * 2) + -- Three corner colors used
    1 * (5 * 4 * 3 * 2)) -- Four corner colors used
  total_corner_together * colors / 10
= 540 :=
by
  sorry

end distinguishable_squares_count_l82_82889


namespace OC_eq_l82_82729

variable {V : Type} [AddCommGroup V]

-- Given vectors a and b
variables (a b : V)

-- Conditions given in the problem
def OA := a + b
def AB := 3 • (a - b)
def CB := 2 • a + b

-- Prove that OC = 2a - 3b
theorem OC_eq : (a + b) + (3 • (a - b)) + (- (2 • a + b)) = 2 • a - 3 • b :=
by
  -- write your proof here
  sorry

end OC_eq_l82_82729


namespace n_squared_sum_of_squares_l82_82792

theorem n_squared_sum_of_squares (n a b c : ℕ) (h : n = a^2 + b^2 + c^2) : 
  ∃ x y z : ℕ, n^2 = x^2 + y^2 + z^2 :=
by 
  sorry

end n_squared_sum_of_squares_l82_82792


namespace triangle_area_l82_82643

noncomputable def area_of_right_triangle (a b c : ℝ) (h : a ^ 2 + b ^ 2 = c ^ 2) : ℝ :=
  (1 / 2) * a * b

theorem triangle_area (a b c : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_of_right_triangle a b c h4 = 54 := by
  rw [h1, h2, h3]
  sorry

end triangle_area_l82_82643


namespace sum_of_undefined_domain_values_l82_82308

theorem sum_of_undefined_domain_values :
  ∀ (x : ℝ), (x = 0 ∨ (1 + 1/x) = 0 ∨ (1 + 1/(1 + 1/x)) = 0 ∨ (1 + 1/(1 + 1/(1 + 1/x))) = 0) →
  x = 0 ∧ x = -1 ∧ x = -1/2 ∧ x = -1/3 →
  (0 + (-1) + (-1/2) + (-1/3) = -11/6) := sorry

end sum_of_undefined_domain_values_l82_82308


namespace number_of_orders_l82_82763

open Nat

theorem number_of_orders (total_targets : ℕ) (targets_A : ℕ) (targets_B : ℕ) (targets_C : ℕ)
  (h1 : total_targets = 10)
  (h2 : targets_A = 4)
  (h3 : targets_B = 3)
  (h4 : targets_C = 3)
  : total_orders = 80 :=
sorry

end number_of_orders_l82_82763


namespace middle_term_arithmetic_sequence_l82_82113

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l82_82113


namespace sequence_arithmetic_l82_82612

variable (a b : ℕ → ℤ)

theorem sequence_arithmetic :
  a 0 = 3 →
  (∀ n : ℕ, n > 0 → b n = a (n + 1) - a n) →
  b 3 = -2 →
  b 10 = 12 →
  a 8 = 3 :=
by
  intros h1 ha hb3 hb10
  sorry

end sequence_arithmetic_l82_82612


namespace arithmetic_sequence_middle_term_l82_82128

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l82_82128


namespace sequence_5th_term_l82_82569

theorem sequence_5th_term (a b c : ℚ) (h1 : a = 1 / 4 * (4 + b)) (h2 : b = 1 / 4 * (a + 40)) (h3 : 40 = 1 / 4 * (b + c)) : 
  c = 2236 / 15 := 
by 
  sorry

end sequence_5th_term_l82_82569


namespace average_condition_l82_82253

theorem average_condition (x : ℝ) :
  (1275 + x) / 51 = 80 * x → x = 1275 / 4079 :=
by
  sorry

end average_condition_l82_82253


namespace cos_330_eq_sqrt3_div_2_l82_82661

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l82_82661


namespace simplify_sqrt_eight_l82_82099

theorem simplify_sqrt_eight : Real.sqrt 8 = 2 * Real.sqrt 2 := sorry

end simplify_sqrt_eight_l82_82099


namespace cos_330_eq_sqrt3_div_2_l82_82668

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82668


namespace three_digit_integers_odd_factors_count_l82_82368

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l82_82368


namespace three_digit_odds_factors_count_l82_82420

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l82_82420


namespace largest_root_in_interval_l82_82169

theorem largest_root_in_interval :
  ∃ (r : ℝ), (2 < r ∧ r < 3) ∧ (∃ (a_2 a_1 a_0 : ℝ), 
    |a_2| ≤ 3 ∧ |a_1| ≤ 3 ∧ |a_0| ≤ 3 ∧ a_2 + a_1 + a_0 = -6 ∧ r^3 + a_2 * r^2 + a_1 * r + a_0 = 0) :=
sorry

end largest_root_in_interval_l82_82169


namespace cos_330_eq_sqrt3_div_2_l82_82663

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l82_82663


namespace radio_loss_percentage_l82_82468

theorem radio_loss_percentage :
  ∀ (cost_price selling_price : ℝ), 
    cost_price = 1500 → 
    selling_price = 1290 → 
    ((cost_price - selling_price) / cost_price) * 100 = 14 :=
by
  intros cost_price selling_price h_cp h_sp
  sorry

end radio_loss_percentage_l82_82468


namespace inequality_of_ab_l82_82020

theorem inequality_of_ab (a b : ℝ) (h₁ : a < 0) (h₂ : -1 < b ∧ b < 0) : ab > ab^2 ∧ ab^2 > a :=
by
  sorry

end inequality_of_ab_l82_82020


namespace maximum_profit_l82_82092

/-- 
Given:
- The fixed cost is 3000 (in thousand yuan).
- The revenue per hundred vehicles is 500 (in thousand yuan).
- The additional cost y is defined as follows:
  - y = 10*x^2 + 100*x for 0 < x < 40
  - y = 501*x + 10000/x - 4500 for x ≥ 40
  
Prove:
1. The profit S(x) (in thousand yuan) in 2020 is:
   - S(x) = -10*x^2 + 400*x - 3000 for 0 < x < 40
   - S(x) = 1500 - x - 10000/x for x ≥ 40
2. The production volume x (in hundreds of vehicles) to achieve the maximum profit is 100,
   and the maximum profit is 1300 (in thousand yuan).
-/
noncomputable def profit_function (x : ℝ) : ℝ :=
  if (0 < x ∧ x < 40) then
    -10 * x^2 + 400 * x - 3000
  else if (x ≥ 40) then
    1500 - x - 10000 / x
  else
    0 -- Undefined for other values, though our x will always be positive in our case

theorem maximum_profit : ∃ x : ℝ, 0 < x ∧ 
  (profit_function x = 1300 ∧ x = 100) ∧
  ∀ y, 0 < y → profit_function y ≤ 1300 :=
sorry

end maximum_profit_l82_82092


namespace coin_loading_impossible_l82_82921

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l82_82921


namespace distance_point_C_to_line_is_2_inch_l82_82724

/-- 
Four 2-inch squares are aligned in a straight line. The second square from the left is rotated 90 degrees, 
and then shifted vertically downward until it touches the adjacent squares. Prove that the distance from 
point C, the top vertex of the rotated square, to the original line on which the bases of the squares were 
placed is 2 inches.
-/
theorem distance_point_C_to_line_is_2_inch :
  ∀ (squares : Fin 4 → ℝ) (rotation : ℝ) (vertical_shift : ℝ) (C_position : ℝ),
  (∀ n : Fin 4, squares n = 2) →
  rotation = 90 →
  vertical_shift = 0 →
  C_position = 2 →
  C_position = 2 :=
by
  intros squares rotation vertical_shift C_position
  sorry

end distance_point_C_to_line_is_2_inch_l82_82724


namespace value_of_y_in_arithmetic_sequence_l82_82129

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l82_82129


namespace minimum_value_of_f_l82_82470

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 4^2)) + (Real.sqrt ((x + 1)^2 + 3^2))

theorem minimum_value_of_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 ∧ ∀ y : ℝ, f y ≥ f x :=
by
  use -3
  sorry

end minimum_value_of_f_l82_82470


namespace train_speed_l82_82918

-- Define the conditions given in the problem
def train_length : ℝ := 160
def time_to_cross_man : ℝ := 4

-- Define the statement to be proved
theorem train_speed (H1 : train_length = 160) (H2 : time_to_cross_man = 4) : train_length / time_to_cross_man = 40 :=
by
  sorry

end train_speed_l82_82918


namespace remainder_of_product_mod_7_l82_82861

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82861


namespace coin_loading_impossible_l82_82941

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l82_82941


namespace sum_of_digits_of_a_l82_82491

-- Define a as 10^10 - 47
def a : ℕ := (10 ^ 10) - 47

-- Function to compute the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

-- The theorem to prove that the sum of all the digits of a is 81
theorem sum_of_digits_of_a : sum_of_digits a = 81 := by
  sorry

end sum_of_digits_of_a_l82_82491


namespace num_three_digit_perfect_cubes_divisible_by_16_l82_82561

-- define what it means for an integer to be a three-digit number
def is_three_digit (n : ℤ) : Prop := 100 ≤ n ∧ n ≤ 999

-- define what it means for an integer to be a perfect cube
def is_perfect_cube (n : ℤ) : Prop := ∃ m : ℤ, m^3 = n

-- define what it means for an integer to be divisible by 16
def is_divisible_by_sixteen (n : ℤ) : Prop := n % 16 = 0

-- define the main theorem that combines these conditions
theorem num_three_digit_perfect_cubes_divisible_by_16 : 
  ∃ n, n = 2 := sorry

end num_three_digit_perfect_cubes_divisible_by_16_l82_82561


namespace pure_imaginary_condition_l82_82749

theorem pure_imaginary_condition (m : ℝ) (h : (m^2 - 3 * m) = 0) : (m = 0) :=
by
  sorry

end pure_imaginary_condition_l82_82749


namespace solve_for_x_add_y_l82_82018

theorem solve_for_x_add_y (x y : ℤ) 
  (h1 : y = 245) 
  (h2 : x - y = 200) : 
  x + y = 690 :=
by {
  -- Here we would provide the proof if needed
  sorry
}

end solve_for_x_add_y_l82_82018


namespace three_digit_integers_odd_factors_count_l82_82366

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l82_82366


namespace range_of_x_l82_82455

variable (x : ℝ)

def p := x^2 - 4 * x + 3 < 0
def q := (x^2 - x - 6 ≤ 0) ∧ (x^2 + 2 * x - 8 > 0)

theorem range_of_x : ¬ (p x ∧ q x) ∧ (p x ∨ q x) → (1 < x ∧ x ≤ 2) ∨ x = 3 :=
by 
  sorry

end range_of_x_l82_82455


namespace sequence_term_l82_82102

noncomputable def a_n (n : ℕ) : ℕ :=
  if n = 1 then 3 else 4 * n - 2

def S_n (n : ℕ) : ℕ :=
  2 * n^2 + 1

theorem sequence_term (n : ℕ) : a_n n = if n = 1 then S_n 1 else S_n n - S_n (n - 1) :=
by 
  sorry

end sequence_term_l82_82102


namespace integer_solutions_l82_82317

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end integer_solutions_l82_82317


namespace satisfies_equation_l82_82320

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end satisfies_equation_l82_82320


namespace sum_a_16_to_20_l82_82337

variable (a : ℕ → ℝ) (S : ℕ → ℝ)

-- Conditions
axiom S_def : ∀ n, S n = a 0 * (1 - (a 1 / a 0) ^ n) / (1 - (a 1 / a 0))
axiom S_5_eq_2 : S 5 = 2
axiom S_10_eq_6 : S 10 = 6

-- Theorem to prove
theorem sum_a_16_to_20 : a 16 + a 17 + a 18 + a 19 + a 20 = 16 :=
by
  sorry

end sum_a_16_to_20_l82_82337


namespace sum_of_coordinates_A_l82_82768

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l82_82768


namespace correct_operation_l82_82276

theorem correct_operation (a : ℝ) : 
  (-2 * a^2)^3 = -8 * a^6 :=
by sorry

end correct_operation_l82_82276


namespace telephone_number_problem_l82_82509

theorem telephone_number_problem
  (digits : Finset ℕ)
  (A B C D E F G H I J : ℕ)
  (h_digits : digits = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
  (h_distinct : [A, B, C, D, E, F, G, H, I, J].Nodup)
  (h_ABC : A > B ∧ B > C)
  (h_DEF : D > E ∧ E > F)
  (h_GHIJ : G > H ∧ H > I ∧ I > J)
  (h_DEF_consecutive_odd : D = E + 2 ∧ E = F + 2 ∧ (D % 2 = 1) ∧ (E % 2 = 1) ∧ (F % 2 = 1))
  (h_GHIJ_consecutive_even : G = H + 2 ∧ H = I + 2 ∧ I = J + 2 ∧ (G % 2 = 0) ∧ (H % 2 = 0) ∧ (I % 2 = 0) ∧ (J % 2 = 0))
  (h_sum_ABC : A + B + C = 15) :
  A = 9 :=
by
  sorry

end telephone_number_problem_l82_82509


namespace solve_chimney_bricks_l82_82164

noncomputable def chimney_bricks (x : ℝ) : Prop :=
  let brenda_rate := x / 8
  let brandon_rate := x / 12
  let combined_rate := brenda_rate + brandon_rate - 15
  (combined_rate * 6) = x

theorem solve_chimney_bricks : ∃ (x : ℝ), chimney_bricks x ∧ x = 360 :=
by
  use 360
  unfold chimney_bricks
  sorry

end solve_chimney_bricks_l82_82164


namespace arithmetic_sequence_l82_82442

variable (a : ℕ → ℕ)
variable (h : a 1 + 3 * a 8 + a 15 = 120)

theorem arithmetic_sequence (h : a 1 + 3 * a 8 + a 15 = 120) : a 2 + a 14 = 48 :=
sorry

end arithmetic_sequence_l82_82442


namespace total_notes_l82_82505

theorem total_notes (total_amount : ℤ) (num_50_notes : ℤ) (value_50 : ℤ) (value_500 : ℤ) (total_notes : ℤ) :
  total_amount = num_50_notes * value_50 + (total_notes - num_50_notes) * value_500 → 
  total_amount = 10350 → num_50_notes = 77 → value_50 = 50 → value_500 = 500 → total_notes = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_notes_l82_82505


namespace kristy_baked_cookies_l82_82582

theorem kristy_baked_cookies (C : ℕ) :
  (C - 3) - 8 - 12 - 16 - 6 - 14 = 10 ↔ C = 69 := by
  sorry

end kristy_baked_cookies_l82_82582


namespace gcf_75_100_l82_82898

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l82_82898


namespace farmer_rewards_l82_82954

theorem farmer_rewards (x y : ℕ) (h1 : x + y = 60) (h2 : 1000 * x + 3000 * y = 100000) : x = 40 ∧ y = 20 :=
by {
  sorry
}

end farmer_rewards_l82_82954


namespace cos_330_eq_sqrt3_div_2_l82_82699

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82699


namespace container_volumes_l82_82025

theorem container_volumes (a r : ℝ) (h1 : (2 * a)^3 = (4 / 3) * Real.pi * r^3) :
  ((2 * a + 2)^3 > (4 / 3) * Real.pi * (r + 1)^3) :=
by sorry

end container_volumes_l82_82025


namespace a_is_4_when_b_is_3_l82_82088

theorem a_is_4_when_b_is_3 
  (a : ℝ) (b : ℝ) (k : ℝ)
  (h1 : ∀ b, a * b^2 = k)
  (h2 : a = 9 ∧ b = 2) :
  a = 4 :=
by
  sorry

end a_is_4_when_b_is_3_l82_82088


namespace sum_coordinates_A_l82_82771

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l82_82771


namespace people_who_cannot_do_either_l82_82215

def people_total : ℕ := 120
def can_dance : ℕ := 88
def can_write_calligraphy : ℕ := 32
def can_do_both : ℕ := 18

theorem people_who_cannot_do_either : 
  people_total - (can_dance + can_write_calligraphy - can_do_both) = 18 := 
by
  sorry

end people_who_cannot_do_either_l82_82215


namespace area_of_rectangle_l82_82638

def length_fence (x : ℝ) : ℝ := 2 * x + 2 * x

theorem area_of_rectangle (x : ℝ) (h : length_fence x = 150) : x * 2 * x = 2812.5 :=
by
  sorry

end area_of_rectangle_l82_82638


namespace intersection_points_3_l82_82534

def eq1 (x y : ℝ) : Prop := (x - y + 3) * (2 * x + 3 * y - 9) = 0
def eq2 (x y : ℝ) : Prop := (2 * x - y + 2) * (x + 3 * y - 6) = 0

theorem intersection_points_3 :
  (∃ x y : ℝ, eq1 x y ∧ eq2 x y) ∧
  (∃ x1 y1 x2 y2 x3 y3 : ℝ, 
    eq1 x1 y1 ∧ eq2 x1 y1 ∧ 
    eq1 x2 y2 ∧ eq2 x2 y2 ∧ 
    eq1 x3 y3 ∧ eq2 x3 y3 ∧
    (x1, y1) ≠ (x2, y2) ∧ (x1, y1) ≠ (x3, y3) ∧ (x2, y2) ≠ (x3, y3)) :=
sorry

end intersection_points_3_l82_82534


namespace three_digit_integers_with_odd_factors_l82_82370

theorem three_digit_integers_with_odd_factors : 
  { n : ℕ // 100 ≤ n ∧ n ≤ 999 ∧ ∃ k : ℕ, n = k^2 }.card = 22 := 
by
  sorry

end three_digit_integers_with_odd_factors_l82_82370


namespace cos_330_eq_sqrt3_div_2_l82_82697

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82697


namespace gcf_75_100_l82_82897

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l82_82897


namespace total_people_in_school_l82_82437

def number_of_girls := 315
def number_of_boys := 309
def number_of_teachers := 772
def total_number_of_people := number_of_girls + number_of_boys + number_of_teachers

theorem total_people_in_school :
  total_number_of_people = 1396 :=
by sorry

end total_people_in_school_l82_82437


namespace remainder_of_product_l82_82850

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82850


namespace prove_statement_II_must_be_true_l82_82498

-- Definitions of the statements
def statement_I (d : ℕ) : Prop := d = 5
def statement_II (d : ℕ) : Prop := d ≠ 6
def statement_III (d : ℕ) : Prop := d = 7
def statement_IV (d : ℕ) : Prop := d ≠ 8

-- Condition: Exactly three of these statements are true and one is false
def exactly_three_true (P Q R S : Prop) : Prop :=
  (P ∧ Q ∧ R ∧ ¬S) ∨ (P ∧ Q ∧ ¬R ∧ S) ∨ (P ∧ ¬Q ∧ R ∧ S) ∨ (¬P ∧ Q ∧ R ∧ S)

-- Problem statement
theorem prove_statement_II_must_be_true (d : ℕ) (h : exactly_three_true (statement_I d) (statement_II d) (statement_III d) (statement_IV d)) : 
  statement_II d :=
by
  -- proof goes here
  sorry

end prove_statement_II_must_be_true_l82_82498


namespace middle_term_arithmetic_sequence_l82_82115

-- Definitions of the given conditions
def a := 3^2
def c := 3^4

-- Assertion that y is the middle term of the arithmetic sequence a, y, c
theorem middle_term_arithmetic_sequence : 
  let y := (a + c) / 2 in 
  y = 45 :=
by
  -- Since the final proof steps are not needed
  sorry

end middle_term_arithmetic_sequence_l82_82115


namespace total_votes_4500_l82_82438

theorem total_votes_4500 (V : ℝ) 
  (h : 0.60 * V - 0.40 * V = 900) : V = 4500 :=
by
  sorry

end total_votes_4500_l82_82438


namespace three_digit_oddfactors_count_is_22_l82_82346

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l82_82346


namespace range_of_m_l82_82198

theorem range_of_m (x1 x2 m : Real) (h_eq : ∀ x : Real, x^2 - 2*x + m + 2 = 0)
  (h_abs : |x1| + |x2| ≤ 3)
  (h_real : ∀ x : Real, ∃ y : Real, x^2 - 2*x + m + 2 = 0) : -13 / 4 ≤ m ∧ m ≤ -1 :=
by
  sorry

end range_of_m_l82_82198


namespace enrique_commission_l82_82715

theorem enrique_commission :
  let commission_rate : ℚ := 0.15
  let suits_sold : ℚ := 2
  let suits_price : ℚ := 700
  let shirts_sold : ℚ := 6
  let shirts_price : ℚ := 50
  let loafers_sold : ℚ := 2
  let loafers_price : ℚ := 150
  let total_sales := suits_sold * suits_price + shirts_sold * shirts_price + loafers_sold * loafers_price
  let commission := commission_rate * total_sales
  commission = 300 := by
begin
  sorry
end

end enrique_commission_l82_82715


namespace find_XY_base10_l82_82313

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end find_XY_base10_l82_82313


namespace line_passes_through_fixed_point_equal_intercepts_line_equation_l82_82736

open Real

theorem line_passes_through_fixed_point (m : ℝ) : ∃ P : ℝ × ℝ, P = (4, 1) ∧ (m + 2) * P.1 - (m + 1) * P.2 - 3 * m - 7 = 0 := 
sorry

theorem equal_intercepts_line_equation (m : ℝ) :
  ((3 * m + 7) / (m + 2) = -(3 * m + 7) / (m + 1)) → (m = -3 / 2) → 
  (∀ (x y : ℝ), (m + 2) * x - (m + 1) * y - 3 * m - 7 = 0 → x + y - 5 = 0) := 
sorry

end line_passes_through_fixed_point_equal_intercepts_line_equation_l82_82736


namespace factorize_expression_l82_82171

theorem factorize_expression : (x^2 + 9)^2 - 36*x^2 = (x + 3)^2 * (x - 3)^2 := 
by 
  sorry

end factorize_expression_l82_82171


namespace closest_point_on_line_l82_82182

-- Definition of the line and the given point
def line (x : ℝ) : ℝ := 2 * x - 4
def point : ℝ × ℝ := (3, -1)

-- Define the closest point we've computed
def closest_point : ℝ × ℝ := (9/5, 2/5)

-- Statement of the problem to prove the closest point
theorem closest_point_on_line : 
  ∃ (p : ℝ × ℝ), p = closest_point ∧ 
  ∀ (q : ℝ × ℝ), (line q.1 = q.2) → 
  (dist point p ≤ dist point q) :=
sorry

end closest_point_on_line_l82_82182


namespace coin_loading_impossible_l82_82944

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l82_82944


namespace number_of_distinct_collections_l82_82823

def mathe_matical_letters : Multiset Char :=
  {'M', 'A', 'T', 'H', 'E', 'M', 'A', 'T', 'I', 'C', 'A', 'L'}

def vowels : Multiset Char :=
  {'A', 'A', 'A', 'E', 'I'}

def consonants : Multiset Char :=
  {'M', 'T', 'H', 'M', 'T', 'C', 'L', 'C'}

def indistinguishable (s : Multiset Char) :=
  (s.count 'A' = s.count 'A' ∧
   s.count 'E' = 1 ∧
   s.count 'I' = 1 ∧
   s.count 'M' = 2 ∧
   s.count 'T' = 2 ∧
   s.count 'H' = 1 ∧
   s.count 'C' = 2 ∧
   s.count 'L' = 1)

theorem number_of_distinct_collections :
  5 * 16 = 80 :=
by
  -- proof would go here
  sorry

end number_of_distinct_collections_l82_82823


namespace remainder_product_l82_82870

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82870


namespace cos_330_eq_sqrt3_div_2_l82_82664

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = sqrt 3 / 2 := 
by 
  -- This is where the proof would go
  sorry

end cos_330_eq_sqrt3_div_2_l82_82664


namespace find_A_coordinates_sum_l82_82767

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l82_82767


namespace sum_of_coordinates_A_l82_82770

theorem sum_of_coordinates_A (a b : ℝ)
  (h1 : a ≠ 0)
  (h2 : ∃ x y : ℝ, y = a * x + 4 ∧ y = 2 * x + b ∧ y = (a / 2) * x + 8) :
  ∃ y : ℝ, y = 13 ∨ y = 20 :=
by
  sorry

end sum_of_coordinates_A_l82_82770


namespace ratio_n_over_p_l82_82532

theorem ratio_n_over_p (m n p : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : p ≠ 0) 
  (h4 : ∃ r1 r2 : ℝ, r1 + r2 = -p ∧ r1 * r2 = m ∧ 3 * r1 + 3 * r2 = -m ∧ 9 * r1 * r2 = n) :
  n / p = -27 := 
by
  sorry

end ratio_n_over_p_l82_82532


namespace moles_of_KCl_formed_l82_82721

variables (NaCl KNO3 KCl NaNO3 : Type) 

-- Define the moles of each compound
variables (moles_NaCl moles_KNO3 moles_KCl moles_NaNO3 : ℕ)

-- Initial conditions
axiom initial_NaCl_condition : moles_NaCl = 2
axiom initial_KNO3_condition : moles_KNO3 = 2

-- Reaction definition
axiom reaction : moles_KCl = moles_NaCl

theorem moles_of_KCl_formed :
  moles_KCl = 2 :=
by sorry

end moles_of_KCl_formed_l82_82721


namespace circle_center_radius_l82_82988

theorem circle_center_radius :
  ∃ (h k r : ℝ), (∀ x y : ℝ, (x + 1)^2 + (y - 1)^2 = 4 → (x - h)^2 + (y - k)^2 = r^2) ∧
    h = -1 ∧ k = 1 ∧ r = 2 :=
by
  sorry

end circle_center_radius_l82_82988


namespace three_digit_cubes_divisible_by_16_l82_82557

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l82_82557


namespace smallest_n_not_divisible_by_10_l82_82011

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l82_82011


namespace enrique_commission_l82_82713

def commission_earned (suits_sold: ℕ) (suit_price: ℝ) (shirts_sold: ℕ) (shirt_price: ℝ) 
                      (loafers_sold: ℕ) (loafers_price: ℝ) (commission_rate: ℝ) : ℝ :=
  let total_sales := (suits_sold * suit_price) + (shirts_sold * shirt_price) + (loafers_sold * loafers_price)
  total_sales * commission_rate

theorem enrique_commission :
  commission_earned 2 700 6 50 2 150 0.15 = 300 := by
  sorry

end enrique_commission_l82_82713


namespace pythagorean_triple_l82_82257

theorem pythagorean_triple {a b c : ℕ} (h : a * a + b * b = c * c) (gcd_abc : Nat.gcd (Nat.gcd a b) c = 1) :
  ∃ m n : ℕ, a = 2 * m * n ∧ b = m * m - n * n ∧ c = m * m + n * n :=
sorry

end pythagorean_triple_l82_82257


namespace min_sum_weights_l82_82566

theorem min_sum_weights (S : ℕ) (h1 : S > 280) (h2 : S % 70 = 30) : S = 310 :=
sorry

end min_sum_weights_l82_82566


namespace number_of_three_digit_integers_with_odd_number_of_factors_l82_82405

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l82_82405


namespace sum_first_15_nat_eq_120_l82_82339

-- Define a function to sum the first n natural numbers
def sum_natural_numbers (n : ℕ) : ℕ :=
  (n * (n + 1)) / 2

-- Define the theorem to show that the sum of the first 15 natural numbers equals 120
theorem sum_first_15_nat_eq_120 : sum_natural_numbers 15 = 120 := 
  by
    sorry

end sum_first_15_nat_eq_120_l82_82339


namespace total_beads_in_necklace_l82_82287

noncomputable def amethyst_beads : ℕ := 7
noncomputable def amber_beads : ℕ := 2 * amethyst_beads
noncomputable def turquoise_beads : ℕ := 19
noncomputable def total_beads : ℕ := amethyst_beads + amber_beads + turquoise_beads

theorem total_beads_in_necklace : total_beads = 40 := by
  sorry

end total_beads_in_necklace_l82_82287


namespace people_visited_neither_l82_82140

theorem people_visited_neither (total_people iceland_visitors norway_visitors both_visitors : ℕ)
  (h1 : total_people = 100)
  (h2 : iceland_visitors = 55)
  (h3 : norway_visitors = 43)
  (h4 : both_visitors = 61) :
  total_people - (iceland_visitors + norway_visitors - both_visitors) = 63 :=
by
  sorry

end people_visited_neither_l82_82140


namespace smallest_positive_integer_n_l82_82016

theorem smallest_positive_integer_n (n : ℕ) (cube : Finset (Fin 8)) :
    (∀ (coloring : Finset (Fin 8)), 
      coloring.card = n → 
      ∃ (v : Fin 8), 
        (∀ (adj : Finset (Fin 8)), adj.card = 3 → adj ⊆ cube → v ∈ adj → adj ⊆ coloring)) 
    ↔ n = 5 := 
by
  sorry

end smallest_positive_integer_n_l82_82016


namespace contrapositive_l82_82610

theorem contrapositive (x : ℝ) (h : x^2 ≥ 1) : x ≥ 0 ∨ x ≤ -1 :=
sorry

end contrapositive_l82_82610


namespace point_coordinates_l82_82329

/-- Given the vector from point A to point B, if point A is the origin, then point B will have coordinates determined by the vector. -/
theorem point_coordinates (A B: ℝ × ℝ) (v: ℝ × ℝ) 
  (h: A = (0, 0)) (h_v: v = (-2, 4)) (h_ab: B = (A.1 + v.1, A.2 + v.2)): 
  B = (-2, 4) :=
by
  sorry

end point_coordinates_l82_82329


namespace number_of_special_divisors_l82_82706

theorem number_of_special_divisors (a b c : ℕ) (n : ℕ) (h : n = 1806) :
  (∀ m : ℕ, m ∣ (2 ^ a * 3 ^ b * 101 ^ c) → (∃ x y z, m = 2 ^ x * 3 ^ y * 101 ^ z ∧ (x + 1) * (y + 1) * (z + 1) = 1806)) →
  (∃ count : ℕ, count = 2) := sorry

end number_of_special_divisors_l82_82706


namespace cos_330_is_sqrt3_over_2_l82_82680

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l82_82680


namespace fraction_identity_l82_82067

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := 2 * x - 3

theorem fraction_identity : 
  (f (g (f 3))) / (g (f (g 3))) = 59 / 19 := by
  sorry

end fraction_identity_l82_82067


namespace three_digit_perfect_squares_count_l82_82381

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l82_82381


namespace tangent_y_axis_circle_eq_l82_82192

theorem tangent_y_axis_circle_eq (h k r : ℝ) (hc : h = -2) (kc : k = 3) (rc : r = abs h) :
  (x + h)^2 + (y - k)^2 = r^2 ↔ (x + 2)^2 + (y - 3)^2 = 4 := by
  sorry

end tangent_y_axis_circle_eq_l82_82192


namespace no_real_solutions_quadratic_solve_quadratic_eq_l82_82250

-- For Equation (1)

theorem no_real_solutions_quadratic (a b c : ℝ) (h_eq : a = 3 ∧ b = -4 ∧ c = 5 ∧ (b^2 - 4 * a * c < 0)) :
  ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0 := 
by
  sorry

-- For Equation (2)

theorem solve_quadratic_eq {x : ℝ} (h_eq : (x + 1) * (x + 2) = 2 * x + 4) :
  x = -2 ∨ x = 1 :=
by
  sorry

end no_real_solutions_quadratic_solve_quadratic_eq_l82_82250


namespace jogging_track_circumference_l82_82616

def speed_Suresh_km_hr : ℝ := 4.5
def speed_wife_km_hr : ℝ := 3.75
def meet_time_min : ℝ := 5.28

theorem jogging_track_circumference : 
  let speed_Suresh_km_min := speed_Suresh_km_hr / 60
  let speed_wife_km_min := speed_wife_km_hr / 60
  let distance_Suresh_km := speed_Suresh_km_min * meet_time_min
  let distance_wife_km := speed_wife_km_min * meet_time_min
  let total_distance_km := distance_Suresh_km + distance_wife_km
  total_distance_km = 0.726 :=
by sorry

end jogging_track_circumference_l82_82616


namespace arithmetic_sequence_y_value_l82_82120

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l82_82120


namespace savings_by_december_l82_82579

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l82_82579


namespace correct_time_fraction_l82_82143

theorem correct_time_fraction : 
  (∀ hour : ℕ, hour < 24 → true) →
  (∀ minute : ℕ, minute < 60 → (minute ≠ 16)) →
  (fraction_of_correct_time = 59 / 60) :=
by
  intros h_hour h_minute
  sorry

end correct_time_fraction_l82_82143


namespace total_paths_A_to_C_via_B_l82_82991

-- Define the conditions
def steps_from_A_to_B : Nat := 6
def steps_from_B_to_C : Nat := 6
def right_moves_A_to_B : Nat := 4
def down_moves_A_to_B : Nat := 2
def right_moves_B_to_C : Nat := 3
def down_moves_B_to_C : Nat := 3

-- Define binomial coefficient function
def binom (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Calculate the number of paths for each segment
def paths_A_to_B : Nat := binom steps_from_A_to_B down_moves_A_to_B
def paths_B_to_C : Nat := binom steps_from_B_to_C down_moves_B_to_C

-- Theorem stating the total number of distinct paths
theorem total_paths_A_to_C_via_B : paths_A_to_B * paths_B_to_C = 300 :=
by
  sorry

end total_paths_A_to_C_via_B_l82_82991


namespace simplify_polynomial_l82_82084
-- Lean 4 statement to prove algebraic simplification


open Polynomial

variable (x : ℤ)

theorem simplify_polynomial :
  3 * (3 * (C x ^ 2) + 9 * C x - 4) - 2 * (C x ^ 2 + 7 * C x - 14) = 7 * (C x ^ 2) + 13 * C x + 16 :=
by
  -- The actual proof steps would be here
  sorry

end simplify_polynomial_l82_82084


namespace coin_loading_impossible_l82_82931

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l82_82931


namespace jane_reading_speed_second_half_l82_82793

-- Definitions from the problem's conditions
def total_pages : ℕ := 500
def first_half_pages : ℕ := total_pages / 2
def first_half_speed : ℕ := 10
def total_days : ℕ := 75

-- The number of days spent reading the first half
def first_half_days : ℕ := first_half_pages / first_half_speed

-- The number of days spent reading the second half
def second_half_days : ℕ := total_days - first_half_days

-- The number of pages in the second half
def second_half_pages : ℕ := total_pages - first_half_pages

-- The actual theorem stating that Jane's reading speed for the second half was 5 pages per day
theorem jane_reading_speed_second_half :
  second_half_pages / second_half_days = 5 :=
by
  sorry

end jane_reading_speed_second_half_l82_82793


namespace problem_positive_l82_82744

theorem problem_positive : ∀ x : ℝ, x < 0 → -3 * x⁻¹ > 0 :=
by 
  sorry

end problem_positive_l82_82744


namespace cos_330_eq_sqrt3_div_2_l82_82694

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82694


namespace expression_evaluation_l82_82648

theorem expression_evaluation : (3 * 15) + 47 - 27 * (2^3) / 4 = 38 := by
  sorry

end expression_evaluation_l82_82648


namespace santino_fruit_total_l82_82461

-- Definitions of the conditions
def numPapayaTrees : ℕ := 2
def numMangoTrees : ℕ := 3
def papayasPerTree : ℕ := 10
def mangosPerTree : ℕ := 20
def totalFruits (pTrees : ℕ) (pPerTree : ℕ) (mTrees : ℕ) (mPerTree : ℕ) : ℕ :=
  (pTrees * pPerTree) + (mTrees * mPerTree)

-- Theorem that states the total number of fruits is 80 given the conditions
theorem santino_fruit_total : totalFruits numPapayaTrees papayasPerTree numMangoTrees mangosPerTree = 80 := 
  sorry

end santino_fruit_total_l82_82461


namespace product_mod_7_l82_82845

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82845


namespace jogging_track_circumference_l82_82615

noncomputable def suresh_speed_km_hr : ℝ := 4.5
noncomputable def wife_speed_km_hr : ℝ := 3.75
noncomputable def meeting_time_min : ℝ := 5.28

def suresh_speed_km_min : ℝ := suresh_speed_km_hr / 60
def wife_speed_km_min : ℝ := wife_speed_km_hr / 60

def total_distance_km : ℝ := (suresh_speed_km_min + wife_speed_km_min) * meeting_time_min

theorem jogging_track_circumference :
  total_distance_km = 0.7257 :=
by
  sorry

end jogging_track_circumference_l82_82615


namespace xiao_cong_math_score_l82_82254

theorem xiao_cong_math_score :
  ∀ (C M E : ℕ),
    (C + M + E) / 3 = 122 → C = 118 → E = 125 → M = 123 :=
by
  intros C M E h1 h2 h3
  sorry

end xiao_cong_math_score_l82_82254


namespace gcd_75_100_l82_82893

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l82_82893


namespace emily_coloring_books_l82_82536

variable (initial_books : ℕ) (given_away : ℕ) (total_books : ℕ) (bought_books : ℕ)

theorem emily_coloring_books :
  initial_books = 7 →
  given_away = 2 →
  total_books = 19 →
  initial_books - given_away + bought_books = total_books →
  bought_books = 14 :=
by
  intros h1 h2 h3 h4
  sorry

end emily_coloring_books_l82_82536


namespace cos_alpha_plus_pi_six_l82_82728

theorem cos_alpha_plus_pi_six (α : ℝ) (hα_in_interval : 0 < α ∧ α < π / 2) (h_cos : Real.cos α = Real.sqrt 3 / 3) :
  Real.cos (α + π / 6) = (3 - Real.sqrt 6) / 6 := 
by
  sorry

end cos_alpha_plus_pi_six_l82_82728


namespace smallest_n_not_divisible_by_10_l82_82008

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l82_82008


namespace product_remainder_mod_7_l82_82877

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82877


namespace remainder_of_product_l82_82851

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82851


namespace second_train_speed_l82_82641

theorem second_train_speed (len1 len2 dist t : ℕ) (h1 : len1 = 100) (h2 : len2 = 150) (h3 : dist = 50) (h4 : t = 60) : 
  (len1 + len2 + dist) / t = 5 := 
  by
  -- Definitions from conditions
  have h_len1 : len1 = 100 := h1
  have h_len2 : len2 = 150 := h2
  have h_dist : dist = 50 := h3
  have h_time : t = 60 := h4
  
  -- Proof deferred
  sorry

end second_train_speed_l82_82641


namespace determine_x_value_l82_82745

theorem determine_x_value (x y : ℝ) (hx : x ≠ 0) (h1 : x / 3 = y ^ 3) (h2 : x / 9 = 9 * y) :
  x = 243 * Real.sqrt 3 ∨ x = -243 * Real.sqrt 3 := by 
  sorry

end determine_x_value_l82_82745


namespace three_digit_integers_with_odd_factors_count_l82_82387

-- Define the conditions
def is_three_digit_integer (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999

def has_odd_number_of_factors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Prove the main statement
theorem three_digit_integers_with_odd_factors_count : 
  { n : ℕ | is_three_digit_integer n ∧ has_odd_number_of_factors n }.to_finset.card = 22 :=
by
  sorry

end three_digit_integers_with_odd_factors_count_l82_82387


namespace cos_330_eq_sqrt3_div_2_l82_82695

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82695


namespace value_of_a_l82_82440

theorem value_of_a (a : ℝ) (x y : ℝ) : 
  (x + a^2 * y + 6 = 0 ∧ (a - 2) * x + 3 * a * y + 2 * a = 0) ↔ a = -1 :=
by
  sorry

end value_of_a_l82_82440


namespace sum_of_x_and_reciprocal_eq_3_5_l82_82547

theorem sum_of_x_and_reciprocal_eq_3_5
    (x : ℝ)
    (h : x^2 + (1 / x^2) = 10.25) :
    x + (1 / x) = 3.5 := 
by
  sorry

end sum_of_x_and_reciprocal_eq_3_5_l82_82547


namespace ones_digit_of_prime_in_sequence_l82_82187

open Nat

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ ∀ m : Nat, m ∣ n → m = 1 ∨ m = n

def valid_arithmetic_sequence (p1 p2 p3 p4: Nat) : Prop :=
  is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ is_prime p4 ∧
  (p2 = p1 + 4) ∧ (p3 = p2 + 4) ∧ (p4 = p3 + 4)

theorem ones_digit_of_prime_in_sequence (p1 p2 p3 p4 : Nat) (hp_seq : valid_arithmetic_sequence p1 p2 p3 p4) (hp1_gt_3 : p1 > 3) : 
  (p1 % 10) = 9 :=
sorry

end ones_digit_of_prime_in_sequence_l82_82187


namespace sum_of_fractions_le_half_l82_82024

theorem sum_of_fractions_le_half {a b c : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 1) :
  1 / (a^2 + 2 * b^2 + 3) + 1 / (b^2 + 2 * c^2 + 3) + 1 / (c^2 + 2 * a^2 + 3) ≤ 1 / 2 :=
by
  sorry

end sum_of_fractions_le_half_l82_82024


namespace three_digit_integers_odd_factors_count_l82_82367

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l82_82367


namespace constant_term_correct_l82_82226

variable (x : ℝ)

noncomputable def constant_term_expansion : ℝ :=
  let term := λ (r : ℕ) => (Nat.choose 9 r) * (-2)^r * x^((9 - 9 * r) / 2)
  term 1

theorem constant_term_correct : 
  constant_term_expansion x = -18 :=
sorry

end constant_term_correct_l82_82226


namespace odd_function_sin_cos_product_l82_82751

-- Prove that if the function f(x) = sin(x + α) - 2cos(x - α) is an odd function, then sin(α) * cos(α) = 2/5
theorem odd_function_sin_cos_product (α : ℝ)
  (hf : ∀ x, Real.sin (x + α) - 2 * Real.cos (x - α) = -(Real.sin (-x + α) - 2 * Real.cos (-x - α))) :
  Real.sin α * Real.cos α = 2 / 5 :=
  sorry

end odd_function_sin_cos_product_l82_82751


namespace find_m_of_parallel_lines_l82_82039

theorem find_m_of_parallel_lines (m : ℝ) 
  (H1 : ∃ x y : ℝ, m * x + 2 * y + 6 = 0) 
  (H2 : ∃ x y : ℝ, x + (m - 1) * y + m^2 - 1 = 0) : 
  m = -1 := 
by
  sorry

end find_m_of_parallel_lines_l82_82039


namespace num_three_digit_integers_with_odd_factors_l82_82353

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l82_82353


namespace sum_of_coordinates_A_l82_82775

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l82_82775


namespace number_of_extreme_points_zero_l82_82476

def f (x a : ℝ) : ℝ := x^3 + 3*x^2 + 4*x - a

theorem number_of_extreme_points_zero (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x, f x1 a = f x a → x = x1 ∨ x = x2) → False := 
by
  sorry

end number_of_extreme_points_zero_l82_82476


namespace compare_expressions_l82_82531

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end compare_expressions_l82_82531


namespace solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l82_82141

-- Define the conditions
variables (a b c x y z: ℝ) 

-- Define the system of equations
def system_of_equations (a b c x y z : ℝ) : Prop :=
  (a * y + b * x = c) ∧
  (c * x + a * z = b) ∧
  (b * z + c * y = a)

-- Define the general solution
def solution (a b c x y z : ℝ) : Prop :=
  x = (b^2 + c^2 - a^2) / (2 * b * c) ∧
  y = (a^2 + c^2 - b^2) / (2 * a * c) ∧
  z = (a^2 + b^2 - c^2) / (2 * a * b)

-- Define the proof problem statement
theorem solve_system_of_equations_general (a b c x y z : ℝ) (h : system_of_equations a b c x y z) 
      (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) : solution a b c x y z :=
  sorry

-- Special cases
theorem solve_system_of_equations_zero_case_1 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_2 (a b c x y z : ℝ) (h1 : a = 0) (h2 : b = 0) (h3: c ≠ 0) : c = 0 :=
  sorry

theorem solve_system_of_equations_zero_case_3 (b c x y z : ℝ) (h : system_of_equations 0 b c x y z) : x = c / b ∧ 
      (c * x = b) :=
  sorry

-- Following special cases more concisely
theorem solve_system_of_equations_special_cases (a b c x y z : ℝ) 
      (h : system_of_equations a b c x y z) (h1: a = 0 ∨ b = 0 ∨ c = 0): 
      (∃ k : ℝ, x = k ∧ y = -k ∧ z = k)  
    ∨ (∃ k : ℝ, x = k ∧ y = k ∧ z = -k)
    ∨ (∃ k : ℝ, x = -k ∧ y = k ∧ z = k) :=
  sorry

end solve_system_of_equations_general_solve_system_of_equations_zero_case_1_solve_system_of_equations_zero_case_2_solve_system_of_equations_zero_case_3_solve_system_of_equations_special_cases_l82_82141


namespace solve_for_m_l82_82425

-- Define the conditions as hypotheses
def hyperbola_equation (x y : Real) (m : Real) : Prop :=
  (x^2)/(m+9) + (y^2)/9 = 1

def eccentricity (e : Real) (a b : Real) : Prop :=
  e = 2 ∧ e^2 = 1 + (b^2)/(a^2)

-- Prove that m = -36 given the conditions
theorem solve_for_m (m : Real) (h : hyperbola_equation x y m) (h_ecc : eccentricity 2 3 (Real.sqrt (-(m+9)))) :
  m = -36 :=
sorry

end solve_for_m_l82_82425


namespace number_of_non_empty_subsets_of_M_inter_N_l82_82031

open Set

def M : Set ℤ := {x | -2 < x ∧ x ≤ 2}
def N : Set ℕ := {y | ∃ x ∈ M, y = x * x}
def common_elements : Set ℕ := M ∩ N

axiom count_non_empty_subsets_common_elements_eq_3 : 
  ∃ (n : ℕ), (n = size (common_elements.to_finset) - 1) = 3

theorem number_of_non_empty_subsets_of_M_inter_N : (∃ (n : ℕ), (n = 2^size (common_elements.to_finset) - 1)) := 
by 
  -- The proof will be filled in here
  sorry

end number_of_non_empty_subsets_of_M_inter_N_l82_82031


namespace fermat_coprime_l82_82629

theorem fermat_coprime (m n : ℕ) (hmn : m ≠ n) (hm_pos : m > 0) (hn_pos : n > 0) :
  gcd (2^(2^m) + 1) (2^(2^n) + 1) = 1 :=
sorry

end fermat_coprime_l82_82629


namespace probabilities_l82_82270

/-- Define events A and B for the probability space of rolling three six-sided dice. -/
def event_A (ω : Fin 6 × Fin 6 × Fin 6) : Prop := ω.1 ≠ ω.2 ∧ ω.2 ≠ ω.3 ∧ ω.1 ≠ ω.3
def event_B (ω : Fin 6 × Fin 6 × Fin 6) : Prop := ω.1 = 5 ∨ ω.2 = 5 ∨ ω.3 = 5

/-- Define the sample space of rolling three six-sided dice. -/
def sample_space : Finset (Fin 6 × Fin 6 × Fin 6) := 
  (Finset.fin_range 6) ×ˢ (Finset.fin_range 6) ×ˢ (Finset.fin_range 6)

/-- Define the probabilities P(AB) and P(B|A) respectively. -/
theorem probabilities (h : 0 < sample_space.card) :
    let P := (sample_space.filter (λ ω, event_A ω ∧ event_B ω)).card / sample_space.card
    let Q := (sample_space.filter (λ ω, event_A ω)).card / sample_space.card
    (P = 75 / 216) ∧ (P / Q = 5 / 8) :=
by
  sorry

end probabilities_l82_82270


namespace shaded_shape_area_l82_82623

/-- Define the coordinates and the conditions for the central square and triangles in the grid -/
def grid_size := 10
def central_square_side := 2
def central_square_area := central_square_side * central_square_side

def triangle_base := 5
def triangle_height := 5
def triangle_area := (1 / 2) * triangle_base * triangle_height

def number_of_triangles := 4
def total_triangle_area := number_of_triangles * triangle_area

def total_shaded_area := total_triangle_area + central_square_area

theorem shaded_shape_area : total_shaded_area = 54 :=
by
  -- We have defined each area component and summed them to the total shaded area.
  -- The statement ensures that the area of the shaded shape is equal to 54.
  sorry

end shaded_shape_area_l82_82623


namespace ab_cd_zero_l82_82737

theorem ab_cd_zero (a b c d : ℝ) 
  (h1 : a^2 + b^2 = 1) 
  (h2 : c^2 + d^2 = 1)
  (h3 : a * c + b * d = 0) : 
  a * b + c * d = 0 := 
by sorry

end ab_cd_zero_l82_82737


namespace coin_loading_impossible_l82_82943

theorem coin_loading_impossible (p q : ℝ) (hp : p ≠ 1 - p) (hq : q ≠ 1 - q) :
  ¬ (p * q = 1/4 ∧ p * (1 - q) = 1/4 ∧ (1 - p) * q = 1/4 ∧ (1 - p) * (1 - q) = 1/4) :=
sorry

end coin_loading_impossible_l82_82943


namespace fifth_team_points_l82_82017

theorem fifth_team_points (points_A points_B points_C points_D points_E : ℕ) 
(hA : points_A = 1) 
(hB : points_B = 2) 
(hC : points_C = 5) 
(hD : points_D = 7) 
(h_sum : points_A + points_B + points_C + points_D + points_E = 20) : 
points_E = 5 := 
sorry

end fifth_team_points_l82_82017


namespace solution_set_of_inequality_l82_82325

theorem solution_set_of_inequality :
  {x : ℝ | -x^2 - x + 2 ≥ 0} = {x : ℝ | -2 ≤ x ∧ x ≤ 1} :=
by
  sorry

end solution_set_of_inequality_l82_82325


namespace ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l82_82448

variables (x y k t : ℕ)

theorem ratio_brothers_sisters_boys (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  (x / (y+1)) = t := 
by simp [h2]

theorem ratio_brothers_sisters_girls (h1 : (x+1) / y = k) (h2 : x / (y+1) = t) :
  ((x+1) / y) = k := 
by simp [h1]

#check ratio_brothers_sisters_boys    -- Just for verification
#check ratio_brothers_sisters_girls   -- Just for verification

end ratio_brothers_sisters_boys_ratio_brothers_sisters_girls_l82_82448


namespace total_amount_paid_is_correct_l82_82291

-- Definitions for the conditions
def original_price : ℝ := 150
def sale_discount : ℝ := 0.30
def coupon_discount : ℝ := 10
def sales_tax : ℝ := 0.10

-- Calculation
def final_amount : ℝ :=
  let discounted_price := original_price * (1 - sale_discount)
  let price_after_coupon := discounted_price - coupon_discount
  let final_price_after_tax := price_after_coupon * (1 + sales_tax)
  final_price_after_tax

-- Statement to prove
theorem total_amount_paid_is_correct : final_amount = 104.50 := by
  sorry

end total_amount_paid_is_correct_l82_82291


namespace non_congruent_right_triangles_unique_l82_82321

theorem non_congruent_right_triangles_unique :
  ∃! (a: ℝ) (b: ℝ) (c: ℝ), a > 0 ∧ b = 2 * a ∧ c = a * Real.sqrt 5 ∧
  (3 * a + a * Real.sqrt 5 - a^2 = a * Real.sqrt 5) :=
by
  sorry

end non_congruent_right_triangles_unique_l82_82321


namespace number_of_three_digit_squares_l82_82384

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l82_82384


namespace roots_seventh_sum_l82_82805

noncomputable def x1 := (-3 + Real.sqrt 5) / 2
noncomputable def x2 := (-3 - Real.sqrt 5) / 2

theorem roots_seventh_sum :
  (x1 ^ 7 + x2 ^ 7) = -843 :=
by
  -- Given condition: x1 and x2 are roots of x^2 + 3x + 1 = 0
  have h1 : x1^2 + 3 * x1 + 1 = 0 := by sorry
  have h2 : x2^2 + 3 * x2 + 1 = 0 := by sorry
  -- Proof goes here
  sorry

end roots_seventh_sum_l82_82805


namespace cos_330_eq_sqrt3_div_2_l82_82658

theorem cos_330_eq_sqrt3_div_2
  (Q : Point)
  (hQ : Q.angle = 330) :
  cos 330 = sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82658


namespace cos_330_eq_sqrt3_over_2_l82_82671

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l82_82671


namespace total_players_is_60_l82_82243

-- Define the conditions
def Cricket_players : ℕ := 25
def Hockey_players : ℕ := 20
def Football_players : ℕ := 30
def Softball_players : ℕ := 18

def Cricket_and_Hockey : ℕ := 5
def Cricket_and_Football : ℕ := 8
def Cricket_and_Softball : ℕ := 3
def Hockey_and_Football : ℕ := 4
def Hockey_and_Softball : ℕ := 6
def Football_and_Softball : ℕ := 9

def Cricket_Hockey_and_Football_not_Softball : ℕ := 2

-- Define total unique players present on the ground
def total_unique_players : ℕ :=
  Cricket_players + Hockey_players + Football_players + Softball_players -
  (Cricket_and_Hockey + Cricket_and_Football + Cricket_and_Softball +
   Hockey_and_Football + Hockey_and_Softball + Football_and_Softball) +
  Cricket_Hockey_and_Football_not_Softball

-- Statement
theorem total_players_is_60:
  total_unique_players = 60 :=
by
  sorry

end total_players_is_60_l82_82243


namespace brooke_initial_l82_82165

variable (B : ℕ)

def brooke_balloons_initially (B : ℕ) :=
  let brooke_balloons := B + 8
  let tracy_balloons_initial := 6
  let tracy_added_balloons := 24
  let tracy_balloons := tracy_balloons_initial + tracy_added_balloons
  let tracy_popped_balloons := tracy_balloons / 2 -- Tracy having half her balloons popped.
  (brooke_balloons + tracy_popped_balloons = 35)

theorem brooke_initial (h : brooke_balloons_initially B) : B = 12 :=
  sorry

end brooke_initial_l82_82165


namespace smallest_integer_k_l82_82914

theorem smallest_integer_k : ∀ (k : ℕ), (64^k > 4^16) → k ≥ 6 :=
by
  sorry

end smallest_integer_k_l82_82914


namespace tank_filling_time_l82_82245

-- Define the rates at which pipes fill or drain the tank
def capacity : ℕ := 1200
def rate_A : ℕ := 50
def rate_B : ℕ := 35
def rate_C : ℕ := 20
def rate_D : ℕ := 40

-- Define the times each pipe is open
def time_A : ℕ := 2
def time_B : ℕ := 4
def time_C : ℕ := 3
def time_D : ℕ := 5

-- Calculate the total time for one cycle
def cycle_time : ℕ := time_A + time_B + time_C + time_D

-- Calculate the net amount of water added in one cycle
def net_amount_per_cycle : ℕ := (rate_A * time_A) + (rate_B * time_B) + (rate_C * time_C) - (rate_D * time_D)

-- Calculate the number of cycles needed to fill the tank
def num_cycles : ℕ := capacity / net_amount_per_cycle

-- Calculate the total time to fill the tank
def total_time : ℕ := num_cycles * cycle_time

-- Prove that the total time to fill the tank is 168 minutes
theorem tank_filling_time : total_time = 168 := by
  sorry

end tank_filling_time_l82_82245


namespace three_digit_integers_with_odd_factors_l82_82396

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l82_82396


namespace savings_by_december_l82_82578

-- Define the basic conditions
def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4

-- Define the final savings calculation
def final_savings : ℕ := initial_savings + total_income - total_expenses

-- The theorem to be proved
theorem savings_by_december : final_savings = 1340840 := by
  -- Proof placeholder
  sorry

end savings_by_december_l82_82578


namespace jindra_gray_fields_counts_l82_82493

-- Definitions for the problem setup
noncomputable def initial_gray_fields: ℕ := 7
noncomputable def rotation_90_gray_fields: ℕ := 8
noncomputable def rotation_180_gray_fields: ℕ := 4

-- Statement of the theorem to be proved
theorem jindra_gray_fields_counts:
  initial_gray_fields = 7 ∧
  rotation_90_gray_fields = 8 ∧
  rotation_180_gray_fields = 4 := by
  sorry

end jindra_gray_fields_counts_l82_82493


namespace impossible_load_two_coins_l82_82926

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l82_82926


namespace books_remainder_l82_82760

theorem books_remainder (total_books new_books_per_section sections : ℕ) 
  (h1 : total_books = 1521) 
  (h2 : new_books_per_section = 45) 
  (h3 : sections = 41) : 
  (total_books * sections) % new_books_per_section = 36 :=
by
  sorry

end books_remainder_l82_82760


namespace part_a_part_b_l82_82496

def fake_coin_min_weighings_9 (n : ℕ) : ℕ :=
  if n = 9 then 2 else 0

def fake_coin_min_weighings_27 (n : ℕ) : ℕ :=
  if n = 27 then 3 else 0

theorem part_a : fake_coin_min_weighings_9 9 = 2 := by
  sorry

theorem part_b : fake_coin_min_weighings_27 27 = 3 := by
  sorry

end part_a_part_b_l82_82496


namespace alex_mother_age_proof_l82_82594

-- Define the initial conditions
def alex_age_2004 : ℕ := 7
def mother_age_2004 : ℕ := 35
def initial_year : ℕ := 2004

-- Define the time variable and the relationship conditions
def years_after_2004 (x : ℕ) : Prop :=
  let alex_age := alex_age_2004 + x
  let mother_age := mother_age_2004 + x
  mother_age = 2 * alex_age

-- State the theorem to be proved
theorem alex_mother_age_proof : ∃ x : ℕ, years_after_2004 x ∧ initial_year + x = 2025 :=
by
  sorry

end alex_mother_age_proof_l82_82594


namespace product_of_two_numbers_l82_82890

theorem product_of_two_numbers :
  ∃ (a b : ℚ), (∀ k : ℚ, a = k + b) ∧ (∀ k : ℚ, a + b = 8 * k) ∧ (∀ k : ℚ, a * b = 40 * k) ∧ (a * b = 6400 / 63) :=
by {
  sorry
}

end product_of_two_numbers_l82_82890


namespace parabola_vertex_point_l82_82834

theorem parabola_vertex_point (a b c : ℝ) : 
  (∀ x y : ℝ, y = a * x^2 + b * x + c → 
  ∃ k : ℝ, ∃ h : ℝ, y = a * (x - h)^2 + k ∧ h = 2 ∧ k = -1 ∧ 
  (∃ y₀ : ℝ, 7 = a * (0 - h)^2 + k) ∧ y₀ = 7) 
  → (a = 2 ∧ b = -8 ∧ c = 7) := by
  sorry

end parabola_vertex_point_l82_82834


namespace fraction_identity_l82_82755

theorem fraction_identity (x y z v : ℝ) (hy : y ≠ 0) (hv : v ≠ 0)
    (h : x / y + z / v = 1) : x / y - z / v = (x / y) ^ 2 - (z / v) ^ 2 := by
  sorry

end fraction_identity_l82_82755


namespace cost_price_of_book_l82_82956

theorem cost_price_of_book (SP : ℝ) (rate_of_profit : ℝ) (CP : ℝ) 
  (h1 : SP = 90) 
  (h2 : rate_of_profit = 0.8) 
  (h3 : rate_of_profit = (SP - CP) / CP) : 
  CP = 50 :=
sorry

end cost_price_of_book_l82_82956


namespace min_value_of_quartic_function_l82_82720

theorem min_value_of_quartic_function : 
  ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ (∀ y : ℝ, (0 ≤ y ∧ y ≤ 1) → x^4 + (1 - x)^4 ≤ y^4 + (1 - y)^4) ∧ (x^4 + (1 - x)^4 = 1 / 8) :=
by
  sorry

end min_value_of_quartic_function_l82_82720


namespace nine_digit_not_perfect_square_l82_82517

theorem nine_digit_not_perfect_square (D : ℕ) (h1 : 100000000 ≤ D) (h2 : D < 1000000000)
  (h3 : ∀ c : ℕ, (c ∈ D.digits 10) → (c ≠ 0)) (h4 : D % 10 = 5) :
  ¬ ∃ A : ℕ, D = A ^ 2 := 
sorry

end nine_digit_not_perfect_square_l82_82517


namespace where_they_meet_l82_82644

/-- Define the conditions under which Petya and Vasya are walking. -/
structure WalkingCondition (n : ℕ) where
  lampposts : ℕ
  start_p : ℕ
  start_v : ℕ
  position_p : ℕ
  position_v : ℕ

/-- Initial conditions based on the problem statement. -/
def initialCondition : WalkingCondition 100 := {
  lampposts := 100,
  start_p := 1,
  start_v := 100,
  position_p := 22,
  position_v := 88
}

/-- Prove Petya and Vasya will meet at the 64th lamppost. -/
theorem where_they_meet (cond : WalkingCondition 100) : 64 ∈ { x | x = 64 } :=
  -- The formal proof would go here.
  sorry

end where_they_meet_l82_82644


namespace zengshan_suanfa_tongzong_l82_82962

-- Definitions
variables (x y : ℝ)
variables (h1 : x = y + 5) (h2 : (1 / 2) * x = y - 5)

-- Theorem
theorem zengshan_suanfa_tongzong :
  x = y + 5 ∧ (1 / 2) * x = y - 5 :=
by
  -- Starting with the given hypotheses
  exact ⟨h1, h2⟩

end zengshan_suanfa_tongzong_l82_82962


namespace sin_double_angle_l82_82335

theorem sin_double_angle (α : ℝ) (h : Real.tan (Real.pi + α) = 2) : Real.sin (2 * α) = 4 / 5 := 
by 
  sorry

end sin_double_angle_l82_82335


namespace algebraic_expression_correct_l82_82271

-- Definition of the problem
def algebraic_expression (x : ℝ) : ℝ :=
  2 * x + 3

-- Theorem statement
theorem algebraic_expression_correct (x : ℝ) :
  algebraic_expression x = 2 * x + 3 :=
by
  sorry

end algebraic_expression_correct_l82_82271


namespace total_pencils_is_54_l82_82819

def total_pencils (m a : ℕ) : ℕ :=
  m + a

theorem total_pencils_is_54 : 
  ∃ (m a : ℕ), (m = 30) ∧ (m = a + 6) ∧ total_pencils m a = 54 :=
by
  sorry

end total_pencils_is_54_l82_82819


namespace triangle_EF_value_l82_82054

variable (D E EF : ℝ)
variable (DE : ℝ)

theorem triangle_EF_value (h₁ : cos (2 * D - E) + sin (D + E) = 2) (h₂ : DE = 6) : EF = 3 :=
sorry

end triangle_EF_value_l82_82054


namespace cos_330_eq_sqrt3_over_2_l82_82672

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l82_82672


namespace savings_by_end_of_2019_l82_82575

variable (income_monthly : ℕ → ℕ) (expenses_monthly : ℕ → ℕ)
variable (initial_savings : ℕ)

noncomputable def total_income : ℕ :=
  (income_monthly 9 + income_monthly 10 + income_monthly 11 + income_monthly 12) * 4

noncomputable def total_expenses : ℕ :=
  (expenses_monthly 9 + expenses_monthly 10 + expenses_monthly 11 + expenses_monthly 12) * 4

noncomputable def final_savings (initial_savings : ℕ) (total_income : ℕ) (total_expenses : ℕ) : ℕ :=
  initial_savings + total_income - total_expenses

theorem savings_by_end_of_2019 :
  (income_monthly 9 = 55000) →
  (income_monthly 10 = 45000) →
  (income_monthly 11 = 10000) →
  (income_monthly 12 = 17400) →
  (expenses_monthly 9 = 40000) →
  (expenses_monthly 10 = 20000) →
  (expenses_monthly 11 = 5000) →
  (expenses_monthly 12 = 2000) →
  initial_savings = 1147240 →
  final_savings initial_savings total_income total_expenses = 1340840 :=
by
  intros h_income_9 h_income_10 h_income_11 h_income_12
         h_expenses_9 h_expenses_10 h_expenses_11 h_expenses_12
         h_initial_savings
  rw [final_savings, total_income, total_expenses]
  rw [h_income_9, h_income_10, h_income_11, h_income_12]
  rw [h_expenses_9, h_expenses_10, h_expenses_11, h_expenses_12]
  rw h_initial_savings
  sorry

end savings_by_end_of_2019_l82_82575


namespace find_m_for_opposite_solutions_l82_82754

theorem find_m_for_opposite_solutions (x y m : ℝ) 
  (h1 : x = -y)
  (h2 : 3 * x + 5 * y = 2)
  (h3 : 2 * x + 7 * y = m - 18) : 
  m = 23 :=
sorry

end find_m_for_opposite_solutions_l82_82754


namespace sum_of_coords_A_l82_82784

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l82_82784


namespace exists_six_consecutive_lcm_l82_82709

theorem exists_six_consecutive_lcm :
  ∃ n : ℕ, Nat.lcm (n) (n+1) (n+2) > Nat.lcm (n+3) (n+4) (n+5) := by
  sorry

end exists_six_consecutive_lcm_l82_82709


namespace gcd_75_100_l82_82904

-- Define the numbers
def a : ℕ := 75
def b : ℕ := 100

-- State the factorizations
def fact_a : a = 3 * 5^2 := by sorry
def fact_b : b = 2^2 * 5^2 := by sorry

-- Lean statement for the proof
theorem gcd_75_100 : Int.gcd a b = 25 := by
  rw [←fact_a, ←fact_b]
  -- Further steps to prove will be continued here
  sorry

end gcd_75_100_l82_82904


namespace find_XY_base10_l82_82312

theorem find_XY_base10 (X Y : ℕ) (h₁ : Y + 2 = X) (h₂ : X + 5 = 11) : X + Y = 10 := 
by 
  sorry

end find_XY_base10_l82_82312


namespace arcsin_add_arccos_eq_pi_div_two_l82_82483

open Real

theorem arcsin_add_arccos_eq_pi_div_two (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) : 
  arcsin x + arccos x = (π / 2) :=
sorry

end arcsin_add_arccos_eq_pi_div_two_l82_82483


namespace chicken_pieces_needed_l82_82266

theorem chicken_pieces_needed :
  let chicken_pasta_pieces := 2
      barbecue_chicken_pieces := 3
      fried_chicken_dinner_pieces := 8
      number_of_fried_chicken_dinner_orders := 2
      number_of_chicken_pasta_orders := 6
      number_of_barbecue_chicken_orders := 3
  in
  (number_of_fried_chicken_dinner_orders * fried_chicken_dinner_pieces +
   number_of_chicken_pasta_orders * chicken_pasta_pieces +
   number_of_barbecue_chicken_orders * barbecue_chicken_pieces) = 37 := by
  sorry

end chicken_pieces_needed_l82_82266


namespace problem_part1_problem_part2_l82_82546

noncomputable def f (x : ℝ) (h : x >= 0) : ℝ := x^2 - 4 * x

theorem problem_part1 (f : ℝ -> ℝ)
  (h1 : ∀ x, f(-x) = -f(x))
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 4 * x) :
  f(-3) + f(-2) + f(3) = 4 := 
  sorry
  
theorem problem_part2 (f : ℝ -> ℝ)
  (h1 : ∀ x, f(-x) = -f(x))
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 4 * x) :
  (∀ x, f x =
    if x ≥ 0 then x^2 - 4 * x 
    else - x^2 - 4 * x)
  ∧  (inter : set.Icc (-real.infinity) (-2) ∫ (2 : ℝ) (+real.infinity))
    := 
  sorry

end problem_part1_problem_part2_l82_82546


namespace twenty_four_game_l82_82551

theorem twenty_four_game : 8 / (3 - 8 / 3) = 24 := 
by
  sorry

end twenty_four_game_l82_82551


namespace three_digit_integers_with_odd_factors_l82_82407

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l82_82407


namespace no_prime_divisible_by_42_l82_82204

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → ¬ (42 ∣ p) :=
by sorry

end no_prime_divisible_by_42_l82_82204


namespace sums_of_coordinates_of_A_l82_82780

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l82_82780


namespace same_terminal_side_angle_in_range_0_to_2pi_l82_82052

theorem same_terminal_side_angle_in_range_0_to_2pi :
  ∃ k : ℤ, 0 ≤ 2 * k * π + (-4) * π / 3 ∧ 2 * k * π + (-4) * π / 3 ≤ 2 * π ∧
  2 * k * π + (-4) * π / 3 = 2 * π / 3 :=
by
  use 1
  sorry

end same_terminal_side_angle_in_range_0_to_2pi_l82_82052


namespace germination_percentage_l82_82185

theorem germination_percentage (seeds_plot1 seeds_plot2 : ℕ) (percent_germ_plot1 : ℕ) (total_percent_germ : ℕ) :
  seeds_plot1 = 300 →
  seeds_plot2 = 200 →
  percent_germ_plot1 = 20 →
  total_percent_germ = 26 →
  ∃ (percent_germ_plot2 : ℕ), percent_germ_plot2 = 35 :=
by
  sorry

end germination_percentage_l82_82185


namespace triangle_sides_l82_82055

theorem triangle_sides
  (D E : ℝ) (DE EF : ℝ)
  (h1 : Real.cos (2 * D - E) + Real.sin (D + E) = 2)
  (hDE : DE = 6) :
  EF = 3 :=
by
  -- Proof is omitted
  sorry

end triangle_sides_l82_82055


namespace not_p_and_q_equiv_not_p_or_not_q_l82_82999

variable (p q : Prop)

theorem not_p_and_q_equiv_not_p_or_not_q (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end not_p_and_q_equiv_not_p_or_not_q_l82_82999


namespace sum_of_coordinates_of_A_l82_82779

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l82_82779


namespace teacher_li_sheets_l82_82465

theorem teacher_li_sheets (x : ℕ)
    (h1 : ∀ (n : ℕ), n = 24 → (x / 24) = ((x / 32) + 2)) :
    x = 192 := by
  sorry

end teacher_li_sheets_l82_82465


namespace expectation_fraction_ident_distrib_l82_82247

open MeasureTheory ProbabilityTheory

variable {Ω : Type} {P : ProbabilitySpace Ω}

theorem expectation_fraction_ident_distrib 
  (X : ℕ → Ω → ℝ)
  (h_indep : IndepFun P X)
  (h_pos : ∀ i, ∀ ω, 0 < X i ω)
  (h_ident : ∀ i j, IdentDistrib P (X i) (X j))
  (n : ℕ) : 
  n > 0 → 
  E (λ ω, X 0 ω / ∑ i in Finset.range n, X i ω) = 1 / n := 
by
  sorry

end expectation_fraction_ident_distrib_l82_82247


namespace train_crossing_time_l82_82634

-- Define the length of the train
def train_length : ℝ := 120

-- Define the speed of the train
def train_speed : ℝ := 15

-- Define the target time to cross the man
def target_time : ℝ := 8

-- Proposition to prove
theorem train_crossing_time :
  target_time = train_length / train_speed :=
by
  sorry

end train_crossing_time_l82_82634


namespace find_a4_l82_82584

theorem find_a4 (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : ∀ n : ℕ, a (n + 1) = a n - 3) : a 4 = -8 :=
by {
  sorry
}

end find_a4_l82_82584


namespace log_domain_l82_82469

noncomputable def f (x : ℝ) : ℝ := Real.log (x - 1) / Real.log 2

theorem log_domain :
  ∀ x : ℝ, (∃ y : ℝ, f y = Real.log (x - 1) / Real.log 2) ↔ x ∈ Set.Ioi 1 :=
by {
  sorry
}

end log_domain_l82_82469


namespace three_digit_integers_with_odd_number_of_factors_count_l82_82400

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l82_82400


namespace minimum_value_f_is_correct_l82_82177

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end minimum_value_f_is_correct_l82_82177


namespace shortest_side_l82_82079

/-- 
Prove that if the lengths of the sides of a triangle satisfy the inequality \( a^2 + b^2 > 5c^2 \), 
then \( c \) is the length of the shortest side.
-/
theorem shortest_side (a b c : ℝ) (h : a^2 + b^2 > 5 * c^2) (triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b) : c ≤ a ∧ c ≤ b :=
by {
  -- Proof will be provided here.
  sorry
}

end shortest_side_l82_82079


namespace cos_330_cos_30_val_answer_l82_82691

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l82_82691


namespace transform_to_zero_set_l82_82587

def S (p : ℕ) : Finset ℕ := Finset.range p

def P (p : ℕ) (x : ℕ) : ℕ := 3 * x ^ ((2 * p - 1) / 3) + x ^ ((p + 1) / 3) + x + 1

def remainder (n p : ℕ) : ℕ := n % p

theorem transform_to_zero_set (p k : ℕ) (hp : Nat.Prime p) (h_cong : p % 3 = 2) (hk : 0 < k) :
  (∃ n : ℕ, ∀ i ∈ S p, remainder (P p i) p = n) ∨ (∃ n : ℕ, ∀ i ∈ S p, remainder (i ^ k) p = n) ↔
  Nat.gcd k (p - 1) > 1 :=
sorry

end transform_to_zero_set_l82_82587


namespace sums_of_coordinates_of_A_l82_82782

theorem sums_of_coordinates_of_A (A B C : ℝ × ℝ) (a b : ℝ)
  (hB : B.2 = 0)
  (hC : C.1 = 0)
  (hAB : ∃ f : ℝ → ℝ, (∀ x, f x = a * x + 4 ∨ f x = 2 * x + b ∨ f x = (a / 2) * x + 8) ∧ (f A.1 = A.2) ∧ (f B.1 = B.2))
  (hBC : ∃ g : ℝ → ℝ, (∀ x, g x = a * x + 4 ∨ g x = 2 * x + b ∨ g x = (a / 2) * x + 8) ∧ (g B.1 = B.2) ∧ (g C.1 = C.2))
  (hAC : ∃ h : ℝ → ℝ, (∀ x, h x = a * x + 4 ∨ h x = 2 * x + b ∨ h x = (a / 2) * x + 8) ∧ (h A.1 = A.2) ∧ (h C.1 = C.2)) :
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sums_of_coordinates_of_A_l82_82782


namespace significant_digits_of_square_side_l82_82098

theorem significant_digits_of_square_side (A : ℝ) (s : ℝ) (h : A = 0.6400) (hs : s^2 = A) : 
  s = 0.8000 :=
sorry

end significant_digits_of_square_side_l82_82098


namespace triangle_property_proof_l82_82058

noncomputable def triangleABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  a = 2 * Real.sqrt 2 ∧
  b = 5 ∧
  c = Real.sqrt 13 ∧
  C = Real.pi / 4 ∧
  ∃ sinA : ℝ, sinA = 2 * Real.sqrt 13 / 13 ∧
  ∃ sin_2A_plus_pi_4 : ℝ, sin_2A_plus_pi_4 = 17 * Real.sqrt 2 / 26

theorem triangle_property_proof :
  ∃ (A B C : ℝ), 
  triangleABC (2 * Real.sqrt 2) 5 (Real.sqrt 13) A B C
:= sorry

end triangle_property_proof_l82_82058


namespace snake_length_difference_l82_82446

theorem snake_length_difference :
  ∀ (jake_len penny_len : ℕ), 
    jake_len > penny_len →
    jake_len + penny_len = 70 →
    jake_len = 41 →
    jake_len - penny_len = 12 :=
by
  intros jake_len penny_len h1 h2 h3
  sorry

end snake_length_difference_l82_82446


namespace percentage_of_one_pair_repeated_digits_l82_82650

theorem percentage_of_one_pair_repeated_digits (n : ℕ) (h1 : 10000 ≤ n) (h2 : n ≤ 99999) :
  ∃ (percentage : ℝ), percentage = 56.0 :=
by
  sorry

end percentage_of_one_pair_repeated_digits_l82_82650


namespace cos_330_eq_sqrt3_div_2_l82_82681

theorem cos_330_eq_sqrt3_div_2 :
  ∀ Q : ℝ × ℝ, 
  Q = (cos (330 * (Real.pi / 180)), sin (330 * (Real.pi / 180))) →
  ∀ E : ℝ × ℝ, 
  E.1 = Q.1 ∧ E.2 = 0 →
  (E.1 > 0 ∧ E.2 = 0) → -- Q is in the fourth quadrant, hence is positive
  Q.1 = √3 / 2 := sorry

end cos_330_eq_sqrt3_div_2_l82_82681


namespace arithmetic_seq_a11_l82_82583

theorem arithmetic_seq_a11 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n, S n = n * (a 1 + a n) / 2)
  (h2 : S 21 = 105) : a 11 = 5 :=
sorry

end arithmetic_seq_a11_l82_82583


namespace gcd_75_100_l82_82900

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l82_82900


namespace three_digit_integers_with_odd_factors_l82_82408

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l82_82408


namespace not_unique_y_20_paise_l82_82636

theorem not_unique_y_20_paise (x y z w : ℕ) : 
  x + y + z + w = 750 → 10 * x + 20 * y + 50 * z + 100 * w = 27500 → ∃ (y₁ y₂ : ℕ), y₁ ≠ y₂ :=
by 
  intro h1 h2
  -- Without additional constraints on x, y, z, w,
  -- suppose that there are at least two different solutions satisfying both equations,
  -- demonstrating the non-uniqueness of y.
  sorry

end not_unique_y_20_paise_l82_82636


namespace least_m_for_no_real_roots_l82_82965

theorem least_m_for_no_real_roots : ∃ (m : ℤ), (∀ (x : ℝ), 3 * x * (m * x + 6) - 2 * x^2 + 8 ≠ 0) ∧ m = 4 := 
sorry

end least_m_for_no_real_roots_l82_82965


namespace three_digit_oddfactors_count_l82_82355

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l82_82355


namespace num_three_digit_integers_with_odd_factors_l82_82351

theorem num_three_digit_integers_with_odd_factors : 
  ∃ n : ℕ, n = 22 ∧ ∀ k : ℕ, (10 ≤ k ∧ k ≤ 31) ↔ (100 ≤ k^2 ∧ k^2 ≤ 999) := 
sorry

end num_three_digit_integers_with_odd_factors_l82_82351


namespace subset_range_a_l82_82539

def setA : Set ℝ := { x | (x^2 - 4 * x + 3) < 0 }
def setB (a : ℝ) : Set ℝ := { x | (2^(1 - x) + a) ≤ 0 ∧ (x^2 - 2*(a + 7)*x + 5) ≤ 0 }

theorem subset_range_a (a : ℝ) : setA ⊆ setB a ↔ -4 ≤ a ∧ a ≤ -1 := 
  sorry

end subset_range_a_l82_82539


namespace casey_pumping_minutes_l82_82525

theorem casey_pumping_minutes :
  let pump_rate := 3
  let corn_rows := 4
  let corn_plants_per_row := 15
  let water_needed_per_corn_plant := 0.5
  let num_pigs := 10
  let water_needed_per_pig := 4
  let num_ducks := 20
  let water_needed_per_duck := 0.25
  let total_water_needed := (corn_rows * corn_plants_per_row * water_needed_per_corn_plant) +
                            (num_pigs * water_needed_per_pig) +
                            (num_ducks * water_needed_per_duck)
  let minutes_needed := total_water_needed / pump_rate
  in minutes_needed = 25 :=
by 
  sorry

end casey_pumping_minutes_l82_82525


namespace geometric_sequence_property_l82_82238

variables {a : ℕ → ℝ} {S : ℕ → ℝ}

noncomputable def a_n (n : ℕ) : ℝ := 2 * 3^(n - 1)
noncomputable def S_n (n : ℕ) : ℝ := 
  if n = 0 then 0
  else (2 * (1 - 3^n)) / (1 - 3)

theorem geometric_sequence_property 
  (h₁ : a 1 + a 2 + a 3 = 26)
  (h₂ : S 6 = 728)
  (h₃ : ∀ n, a n = a_n n)
  (h₄ : ∀ n, S n = S_n n) :
  ∀ n, S (n + 1) ^ 2 - S n * S (n + 2) = 4 * 3 ^ n :=
by sorry

end geometric_sequence_property_l82_82238


namespace calculate_expression_value_l82_82167

theorem calculate_expression_value :
  5 * 7 + 6 * 9 + 13 * 2 + 4 * 6 = 139 :=
by
  -- proof can be added here
  sorry

end calculate_expression_value_l82_82167


namespace fraction_pow_rule_l82_82522

theorem fraction_pow_rule :
  (5 / 7)^4 = 625 / 2401 :=
by
  sorry

end fraction_pow_rule_l82_82522


namespace num_people_visited_iceland_l82_82436

noncomputable def total := 100
noncomputable def N := 43  -- Number of people who visited Norway
noncomputable def B := 61  -- Number of people who visited both Iceland and Norway
noncomputable def Neither := 63  -- Number of people who visited neither country
noncomputable def I : ℕ := 55  -- Number of people who visited Iceland (need to prove)

-- Lean statement to prove
theorem num_people_visited_iceland : I = total - Neither + B - N := by
  sorry

end num_people_visited_iceland_l82_82436


namespace log2_6_gt_2_sqrt_5_l82_82528

theorem log2_6_gt_2_sqrt_5 : 2 + Real.logb 2 6 > 2 * Real.sqrt 5 := by
  sorry

end log2_6_gt_2_sqrt_5_l82_82528


namespace standard_eq_circle_C_equation_line_AB_l82_82981

-- Define the center of circle C and the line l
def center_C : ℝ × ℝ := (2, 1)
def line_l (x y : ℝ) : Prop := x = 3

-- Define the standard equation of circle C
def eq_circle_C (x y : ℝ) : Prop :=
  (x - center_C.1)^2 + (y - center_C.2)^2 = 1

-- Equation of circle O
def eq_circle_O (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the condition that circle C intersects with circle O at points A and B
def intersects (x y : ℝ) : Prop :=
  eq_circle_C x y ∧ eq_circle_O x y

-- Define the equation of line AB in general form
def eq_line_AB (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Prove the standard equation of circle C is (x-2)^2 + (y-1)^2 = 1
theorem standard_eq_circle_C:
  eq_circle_C x y ↔ (x - 2)^2 + (y - 1)^2 = 1 :=
sorry

-- Prove that the equation of line AB is 2x + y - 4 = 0, given the intersection points A and B
theorem equation_line_AB (x y : ℝ) (h : intersects x y) :
  eq_line_AB x y :=
sorry

end standard_eq_circle_C_equation_line_AB_l82_82981


namespace problem_l82_82824

theorem problem
  (circle : Type)
  (A B C D : circle)
  (inscribed : InscribedQuadrilateral circle A B C D)
  (angle_BAC : ∠ B A C = 80)
  (angle_ADB : ∠ A D B = 35)
  (AD BC : ℝ)
  (AD_equals_5 : AD = 5)
  (BC_equals_7 : BC = 7):
  length (segment A C) = 7 * (real.sin 65) / (real.sin 80) :=
sorry

end problem_l82_82824


namespace original_wage_l82_82959

theorem original_wage (W : ℝ) 
  (h1: 1.40 * W = 28) : 
  W = 20 :=
sorry

end original_wage_l82_82959


namespace contest_result_l82_82184

-- Definition of the positions
def pos := Fin 5
-- Definition of students
inductive Student
| A | B | C | D | E

open Student 

-- Hypotheses based on given problem
variable (placement: pos → Student)
def prediction1 := [A, B, C, D, E]
def prediction2 := [D, A, E, C, B]

-- Conditions from the problem statement
def condition1 : Prop := ∀ i, placement i ≠ prediction1.get ⟨i.1⟩ sorry
def consecutive (s1 s2 : Student) (i j : pos) : Prop := (placement i = s1 ∧ placement j = s2 ∧ abs(i.1 - j.1) = 1)

def condition2 : Prop := ¬ ∃ i j : pos, consecutive (prediction1.get ⟨i.1⟩ sorry) (prediction1.get ⟨j.1⟩ sorry) i j

def condition3 : Prop := ∃ i j : pos, i ≠ j ∧ placement i = prediction2.get ⟨i.1⟩ sorry ∧ placement j = prediction2.get ⟨j.1⟩ sorry

def condition4 : Prop := ∃ si sj sk sl: Student, ∃ i j k l : pos, i ≠ j ∧ k ≠ l ∧ (consecutive si sj i j ∨ consecutive sk sl k l) ∧ ((si, sj) ∈ [(D, A), (A, E), (E, C), (C, B)] ∧ (sk, sl) ∈ [(D, A), (A, E), (E, C), (C, B)])

theorem contest_result : condition1 placement → condition2 placement → condition3 placement → condition4 placement → placement = λi, [E, D, A, C, B].get ⟨i.1⟩ sorry :=
by
  sorry

end contest_result_l82_82184


namespace cos_330_eq_sqrt_3_div_2_l82_82674

-- Definitions based on the conditions
def angle_330 := 330
def angle_30 := 30
def angle_360 := 360
def cos_30 : ℝ := real.cos (angle_30 * real.pi / 180) -- Known value in radians

-- Main theorem statement in Lean 4
theorem cos_330_eq_sqrt_3_div_2 :
  real.cos (angle_330 * real.pi / 180) = cos_30 :=
begin
  -- Given known values for cosine of specific angles
  have h_cos_30 : real.cos (angle_30 * real.pi / 180) = √3/2,
  { exact cos_30 },

  -- Prove that cos(330 degrees) = cos(30 degrees)
  calc
  real.cos (angle_330 * real.pi / 180)
      = real.cos ((angle_360 - angle_30) * real.pi / 180) : by norm_num
  ... = real.cos (angle_30 * real.pi / 180) : by rw real.cos_sub_pi
  ... = √3/2 : by exact h_cos_30,
end

end cos_330_eq_sqrt_3_div_2_l82_82674


namespace find_m_direct_proportion_l82_82188

theorem find_m_direct_proportion (m : ℝ) (h1 : m^2 - 3 = 1) (h2 : m ≠ 2) : m = -2 :=
by {
  -- here would be the proof, but it's omitted as per instructions
  sorry
}

end find_m_direct_proportion_l82_82188


namespace pointA_when_B_origin_pointB_when_A_origin_l82_82328

def vectorAB : ℝ × ℝ := (-2, 4)

-- Prove that when point B is the origin, the coordinates of point A are (2, -4)
theorem pointA_when_B_origin : vectorAB = (-2, 4) → (0, 0) - (-2, 4) = (2, -4) :=
by
  sorry

-- Prove that when point A is the origin, the coordinates of point B are (-2, 4)
theorem pointB_when_A_origin : vectorAB = (-2, 4) → (0, 0) + (-2, 4) = (-2, 4) :=
by
  sorry

end pointA_when_B_origin_pointB_when_A_origin_l82_82328


namespace product_mod_7_l82_82841

theorem product_mod_7 (a b c: ℕ) (ha: a % 7 = 2) (hb: b % 7 = 3) (hc: c % 7 = 4) :
  (a * b * c) % 7 = 3 :=
by
  sorry

end product_mod_7_l82_82841


namespace arithmetic_sequence_y_value_l82_82119

theorem arithmetic_sequence_y_value :
  ∃ y : ℤ, (∃ a1 a3 : ℤ, a1 = 9 ∧ a3 = 81 ∧ y = (a1 + a3) / 2) → y = 45 :=
by
  sorry

end arithmetic_sequence_y_value_l82_82119


namespace impossible_to_load_two_coins_l82_82940

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l82_82940


namespace remainder_of_product_mod_7_l82_82864

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82864


namespace arithmetic_geometric_means_l82_82544

theorem arithmetic_geometric_means (a b : ℝ) (h1 : 2 * a = 1 + 2) (h2 : b^2 = (-1) * (-16)) : a * b = 6 ∨ a * b = -6 :=
by
  sorry

end arithmetic_geometric_means_l82_82544


namespace smallest_n_not_divisible_by_10_l82_82009

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l82_82009


namespace largest_of_set_l82_82516

theorem largest_of_set : 
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  c = 2 ∧ (d < b ∧ b < a ∧ a < c) := by
  let a := 1 / 2
  let b := -1
  let c := abs (-2)
  let d := -3
  sorry

end largest_of_set_l82_82516


namespace find_x_value_l82_82552

theorem find_x_value (x : ℝ) (a b c : ℝ × ℝ × ℝ) 
  (h_a : a = (1, 1, x)) 
  (h_b : b = (1, 2, 1)) 
  (h_c : c = (1, 1, 1)) 
  (h_cond : (c - a) • (2 • b) = -2) : 
  x = 2 := 
by 
  -- the proof goes here
  sorry

end find_x_value_l82_82552


namespace number_of_diagonals_excluding_dividing_diagonals_l82_82649

theorem number_of_diagonals_excluding_dividing_diagonals (n : ℕ) (h1 : n = 150) :
  let totalDiagonals := n * (n - 3) / 2
  let dividingDiagonals := n / 2
  totalDiagonals - dividingDiagonals = 10950 :=
by
  sorry

end number_of_diagonals_excluding_dividing_diagonals_l82_82649


namespace number_of_three_digit_integers_with_odd_factors_l82_82375

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l82_82375


namespace ron_spending_increase_l82_82429

variable (P Q : ℝ) -- initial price and quantity
variable (X : ℝ)   -- intended percentage increase in spending

theorem ron_spending_increase :
  (1 + X / 100) * P * Q = 1.25 * P * (0.92 * Q) →
  X = 15 := 
by
  sorry

end ron_spending_increase_l82_82429


namespace rectangle_perimeter_l82_82472

theorem rectangle_perimeter (b : ℕ) (h1 : 3 * b * b = 192) : 2 * ((3 * b) + b) = 64 := 
by
  sorry

end rectangle_perimeter_l82_82472


namespace a_plus_b_values_l82_82189

theorem a_plus_b_values (a b : ℤ) (h1 : |a + 1| = 0) (h2 : b^2 = 9) :
  a + b = 2 ∨ a + b = -4 :=
by
  have ha : a = -1 := by sorry
  have hb1 : b = 3 ∨ b = -3 := by sorry
  cases hb1 with
  | inl b_pos =>
    left
    rw [ha, b_pos]
    exact sorry
  | inr b_neg =>
    right
    rw [ha, b_neg]
    exact sorry

end a_plus_b_values_l82_82189


namespace sum_of_coordinates_of_A_l82_82778

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l82_82778


namespace english_book_pages_l82_82159

def numPagesInOneEnglishBook (x y : ℕ) : Prop :=
  x = y + 12 ∧ 3 * x + 4 * y = 1275 → x = 189

-- The statement with sorry as no proof is required:
theorem english_book_pages (x y : ℕ) (h1 : x = y + 12) (h2 : 3 * x + 4 * y = 1275) : x = 189 :=
  sorry

end english_book_pages_l82_82159


namespace value_of_y_in_arithmetic_sequence_l82_82132

theorem value_of_y_in_arithmetic_sequence :
    ∃ y : ℤ, (arithmetic_sequence (3^2) y (3^4)) ∧ y = 45 := by
  -- Here we define the arithmetic sequence condition.
  def arithmetic_sequence (a b c : ℤ) : Prop := b = (a + c) / 2
  sorry

end value_of_y_in_arithmetic_sequence_l82_82132


namespace count_8_digit_numbers_l82_82992

theorem count_8_digit_numbers : 
  (∃ n : ℕ, n = 90_000_000) ↔ 
  (∃ L : Fin 9 → Fin 10, ∀ i : Fin 9, L i ≠ 0) :=
begin
  sorry
end

end count_8_digit_numbers_l82_82992


namespace slip_2_5_goes_to_B_l82_82229

-- Defining the slips and their values
def slips : List ℝ := [1.5, 2, 2, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 5, 5.5, 6]

-- Defining the total sum of slips
def total_sum : ℝ := 52

-- Defining the cup sum values
def cup_sums : List ℝ := [11, 10, 9, 8, 7]

-- Conditions: slip with 4 goes into cup A, slip with 5 goes into cup D
def cup_A_contains : ℝ := 4
def cup_D_contains : ℝ := 5

-- Proof statement
theorem slip_2_5_goes_to_B : 
  ∃ (cup_A cup_B cup_C cup_D cup_E : List ℝ), 
    (cup_A.sum = 11 ∧ cup_B.sum = 10 ∧ cup_C.sum = 9 ∧ cup_D.sum = 8 ∧ cup_E.sum = 7) ∧
    (4 ∈ cup_A) ∧ (5 ∈ cup_D) ∧ (2.5 ∈ cup_B) :=
sorry

end slip_2_5_goes_to_B_l82_82229


namespace rectangle_area_solution_l82_82708

theorem rectangle_area_solution (x : ℝ) (h1 : (x + 3) * (2*x - 1) = 12*x + 5) : 
  x = (7 + Real.sqrt 113) / 4 :=
by 
  sorry

end rectangle_area_solution_l82_82708


namespace largest_of_three_l82_82626

theorem largest_of_three (a b c : ℝ) (h₁ : a = 43.23) (h₂ : b = 2/5) (h₃ : c = 21.23) :
  max (max a b) c = a :=
by
  sorry

end largest_of_three_l82_82626


namespace coin_loading_impossible_l82_82932

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l82_82932


namespace trig_identity_example_l82_82495

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) -
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_example_l82_82495


namespace points_2_units_away_l82_82077

theorem points_2_units_away : (∃ x : ℝ, (x = -3 ∨ x = 1) ∧ (abs (x - (-1)) = 2)) :=
by
  sorry

end points_2_units_away_l82_82077


namespace Arthur_total_distance_l82_82518

/-- Arthur walks 8 blocks south and then 10 blocks west. Each block is one-fourth of a mile.
How many miles did Arthur walk in total? -/
theorem Arthur_total_distance (blocks_south : ℕ) (blocks_west : ℕ) (block_length_miles : ℝ) :
  blocks_south = 8 ∧ blocks_west = 10 ∧ block_length_miles = 1/4 →
  (blocks_south + blocks_west) * block_length_miles = 4.5 :=
by
  intro h
  have h1 : blocks_south = 8 := h.1
  have h2 : blocks_west = 10 := h.2.1
  have h3 : block_length_miles = 1 / 4 := h.2.2
  sorry

end Arthur_total_distance_l82_82518


namespace total_earnings_from_peaches_l82_82589

-- Definitions of the conditions
def total_peaches : ℕ := 15
def peaches_sold_to_friends : ℕ := 10
def price_per_peach_friends : ℝ := 2
def peaches_sold_to_relatives : ℕ :=  4
def price_per_peach_relatives : ℝ := 1.25
def peaches_for_self : ℕ := 1

-- We aim to prove the following statement
theorem total_earnings_from_peaches :
  (peaches_sold_to_friends * price_per_peach_friends) +
  (peaches_sold_to_relatives * price_per_peach_relatives) = 25 := by
  -- proof goes here
  sorry

end total_earnings_from_peaches_l82_82589


namespace value_range_of_f_in_interval_l82_82620

noncomputable def f (x : ℝ) : ℝ := x / (x + 2)

theorem value_range_of_f_in_interval : 
  ∀ x, (2 ≤ x ∧ x ≤ 4) → (1/2 ≤ f x ∧ f x ≤ 2/3) := 
by
  sorry

end value_range_of_f_in_interval_l82_82620


namespace sin_3x_sin_x_solutions_l82_82555

open Real

theorem sin_3x_sin_x_solutions :
  ∃ s : Finset ℝ, (∀ x ∈ s, 0 ≤ x ∧ x ≤ 2 * π ∧ sin (3 * x) = sin x) ∧ s.card = 7 := 
by sorry

end sin_3x_sin_x_solutions_l82_82555


namespace largest_angle_in_pentagon_l82_82568

-- Define the angles of the pentagon
variables (C D E : ℝ) 

-- Given conditions
def is_pentagon (A B C D E : ℝ) : Prop :=
  A = 75 ∧ B = 95 ∧ D = C + 10 ∧ E = 2 * C + 20 ∧ A + B + C + D + E = 540

-- Prove that the measure of the largest angle is 190°
theorem largest_angle_in_pentagon (C D E : ℝ) : 
  is_pentagon 75 95 C D E → max 75 (max 95 (max C (max (C + 10) (2 * C + 20)))) = 190 :=
by 
  sorry

end largest_angle_in_pentagon_l82_82568


namespace distance_between_adjacent_parallel_lines_l82_82978

noncomputable def distance_between_lines (r d : ℝ) : ℝ :=
  (49 * r^2 - 49 * 600.25 - (49 / 4) * d^2) / (1 - 49 / 4)

theorem distance_between_adjacent_parallel_lines :
  ∃ d : ℝ, ∀ (r : ℝ), 
    (r^2 = 506.25 + (1 / 4) * d^2 ∧ r^2 = 600.25 + (49 / 4) * d^2) →
    d = 2.8 :=
sorry

end distance_between_adjacent_parallel_lines_l82_82978


namespace train_speed_l82_82958

theorem train_speed (len_train len_bridge time : ℝ) (h_len_train : len_train = 120)
  (h_len_bridge : len_bridge = 150) (h_time : time = 26.997840172786177) :
  let total_distance := len_train + len_bridge
  let speed_m_s := total_distance / time
  let speed_km_h := speed_m_s * 3.6
  speed_km_h = 36 :=
by
  -- Proof goes here
  sorry

end train_speed_l82_82958


namespace weight_of_each_bar_l82_82449

theorem weight_of_each_bar 
  (num_bars : ℕ) 
  (cost_per_pound : ℝ) 
  (total_cost : ℝ) 
  (total_weight : ℝ) 
  (weight_per_bar : ℝ)
  (h1 : num_bars = 20)
  (h2 : cost_per_pound = 0.5)
  (h3 : total_cost = 15)
  (h4 : total_weight = total_cost / cost_per_pound)
  (h5 : weight_per_bar = total_weight / num_bars)
  : weight_per_bar = 1.5 := 
by
  sorry

end weight_of_each_bar_l82_82449


namespace sum_of_coords_A_l82_82783

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l82_82783


namespace twins_ages_sum_equals_20_l82_82450

def sum_of_ages (A K : ℕ) := 2 * A + K

theorem twins_ages_sum_equals_20 (A K : ℕ) (h1 : A = A) (h2 : A * A * K = 256) : 
  sum_of_ages A K = 20 :=
by
  sorry

end twins_ages_sum_equals_20_l82_82450


namespace find_A_coordinates_sum_l82_82765

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l82_82765


namespace product_remainder_mod_7_l82_82873

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82873


namespace three_digit_odds_factors_count_l82_82418

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l82_82418


namespace total_pieces_correct_l82_82265

-- Definition of the pieces of chicken required per type of order
def chicken_pieces_per_chicken_pasta : ℕ := 2
def chicken_pieces_per_barbecue_chicken : ℕ := 3
def chicken_pieces_per_fried_chicken_dinner : ℕ := 8

-- Definition of the number of each type of order tonight
def num_fried_chicken_dinner_orders : ℕ := 2
def num_chicken_pasta_orders : ℕ := 6
def num_barbecue_chicken_orders : ℕ := 3

-- Calculate the total number of pieces of chicken needed
def total_chicken_pieces_needed : ℕ :=
  (num_fried_chicken_dinner_orders * chicken_pieces_per_fried_chicken_dinner) +
  (num_chicken_pasta_orders * chicken_pieces_per_chicken_pasta) +
  (num_barbecue_chicken_orders * chicken_pieces_per_barbecue_chicken)

-- The proof statement
theorem total_pieces_correct : total_chicken_pieces_needed = 37 :=
by
  -- Our exact computation here
  sorry

end total_pieces_correct_l82_82265


namespace log_inequality_l82_82984

theorem log_inequality (a b c : ℝ) (h1 : b^2 - a * c < 0) :
  ∀ x y : ℝ, a * (Real.log x)^2 + 2 * b * (Real.log x) * (Real.log y) + c * (Real.log y)^2 = 1 
  → a * 1^2 + 2 * b * 1 * (-1) + c * (-1)^2 = 1 → 
  -1 / Real.sqrt (a * c - b^2) ≤ Real.log (x * y) ∧ Real.log (x * y) ≤ 1 / Real.sqrt (a * c - b^2) :=
by
  sorry

end log_inequality_l82_82984


namespace log_ordering_correct_l82_82947

noncomputable def log_ordering : Prop :=
  let a := 20.3
  let b := 0.32
  let c := Real.log b
  (0 < b ∧ b < 1) ∧ (c < 0) ∧ (c < b ∧ b < a)

theorem log_ordering_correct : log_ordering :=
by
  -- skipped proof
  sorry

end log_ordering_correct_l82_82947


namespace remainder_when_divided_by_r_minus_1_l82_82324

def f (r : Int) : Int := r^14 - r + 5

theorem remainder_when_divided_by_r_minus_1 : f 1 = 5 := by
  sorry

end remainder_when_divided_by_r_minus_1_l82_82324


namespace probability_at_least_two_same_l82_82081

theorem probability_at_least_two_same (n m : ℕ) (hn : n = 8) (hm : m = 6):
  (probability (λ (ω : vector (fin m) n), ∃ (i j : fin n), i ≠ j ∧ ω.nth i = ω.nth j)) = 1 :=
begin
  sorry
end

end probability_at_least_two_same_l82_82081


namespace number_of_sodas_bought_l82_82486

theorem number_of_sodas_bought
  (sandwich_cost : ℝ)
  (num_sandwiches : ℝ)
  (soda_cost : ℝ)
  (total_cost : ℝ)
  (h1 : sandwich_cost = 3.49)
  (h2 : num_sandwiches = 2)
  (h3 : soda_cost = 0.87)
  (h4 : total_cost = 10.46) :
  (total_cost - num_sandwiches * sandwich_cost) / soda_cost = 4 := 
sorry

end number_of_sodas_bought_l82_82486


namespace annual_profits_l82_82154

-- Define the profits of each quarter
def P1 : ℕ := 1500
def P2 : ℕ := 1500
def P3 : ℕ := 3000
def P4 : ℕ := 2000

-- State the annual profit theorem
theorem annual_profits : P1 + P2 + P3 + P4 = 8000 := by
  sorry

end annual_profits_l82_82154


namespace find_f_l82_82021

-- Define the conditions
def g (x : ℝ) : ℝ := 2 * x + 3
def f (x : ℝ) : ℝ := g (x + 2)

-- State the theorem
theorem find_f :
  ∀ x : ℝ, f x = 2 * x + 7 :=
by
  sorry

end find_f_l82_82021


namespace number_of_substitution_ways_mod_1000_l82_82639

theorem number_of_substitution_ways_mod_1000 :
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  total_ways % 1000 = 573 := by
  -- Definition
  let a_0 := 1
  let a_1 := 12 * 12 * a_0
  let a_2 := 12 * 11 * a_1
  let a_3 := 12 * 10 * a_2
  let a_4 := 12 * 9 * a_3
  let total_ways := a_0 + a_1 + a_2 + a_3 + a_4
  -- Proof is omitted
  sorry

end number_of_substitution_ways_mod_1000_l82_82639


namespace sum_of_primes_eq_100_l82_82100

theorem sum_of_primes_eq_100 : 
  ∃ (S : Finset ℕ), (∀ (x : ℕ), x ∈ S → Nat.Prime x) ∧ S.sum id = 100 ∧ S.card = 9 :=
by
  sorry

end sum_of_primes_eq_100_l82_82100


namespace new_ratio_l82_82840

theorem new_ratio (J: ℝ) (F: ℝ) (F_new: ℝ): 
  J = 59.99999999999997 → 
  F / J = 3 / 2 → 
  F_new = F + 10 → 
  F_new / J = 5 / 3 :=
by
  intros hJ hF hF_new
  sorry

end new_ratio_l82_82840


namespace scientific_notation_of_million_l82_82475

theorem scientific_notation_of_million (x : ℝ) (h : x = 2600000) : x = 2.6 * 10^6 := by
  sorry

end scientific_notation_of_million_l82_82475


namespace white_area_of_sign_l82_82261

theorem white_area_of_sign : 
  let total_area := 6 * 18
  let F_area := 2 * (4 * 1) + 6 * 1
  let O_area := 2 * (6 * 1) + 2 * (4 * 1)
  let D_area := 6 * 1 + 4 * 1 + 4 * 1
  let total_black_area := F_area + O_area + O_area + D_area
  total_area - total_black_area = 40 :=
by
  sorry

end white_area_of_sign_l82_82261


namespace function_symmetry_implies_even_l82_82998

theorem function_symmetry_implies_even (f : ℝ → ℝ) (h1 : ∃ x, f x ≠ 0)
  (h2 : ∀ x y, f x = y ↔ -f (-x) = -y) : ∀ x, f x = f (-x) :=
by
  sorry

end function_symmetry_implies_even_l82_82998


namespace max_value_of_x1_squared_plus_x2_squared_l82_82191

theorem max_value_of_x1_squared_plus_x2_squared :
  ∀ (k : ℝ), -4 ≤ k ∧ k ≤ -4 / 3 → (∃ x1 x2 : ℝ, x1^2 + x2^2 = 18) :=
by
  sorry

end max_value_of_x1_squared_plus_x2_squared_l82_82191


namespace triangle_AB_C_min_perimeter_l82_82060

noncomputable def minimum_perimeter (a b c : ℕ) (A B C : ℝ) : ℝ := a + b + c

theorem triangle_AB_C_min_perimeter
  (a b c : ℕ)
  (A B C : ℝ)
  (h1 : A = 2 * B)
  (h2 : C > π / 2)
  (h3 : a^2 = b * (b + c))
  (h4 : ∀ x : ℕ, x > 0 → a ≠ 0)
  (h5 :  a + b > c ∧ a + c > b ∧ b + c > a) :
  minimum_perimeter a b c A B C = 77 := 
sorry

end triangle_AB_C_min_perimeter_l82_82060


namespace minimize_AB_l82_82199

-- Definition of the circle C
def circleC (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 3 = 0

-- Definition of the point P
def P : ℝ × ℝ := (-1, 2)

-- Definition of the line l
def line_l (x y : ℝ) : Prop := x - y + 3 = 0

-- The goal is to prove that line_l is the line through P minimizing |AB|
theorem minimize_AB : 
  ∀ l : ℝ → ℝ → Prop, 
  (∀ x y, l x y → (∃ a b, circleC a b ∧ l a b ∧ circleC x y ∧ l x y ∧ (x ≠ a ∨ y ≠ b)) → False) 
  → l = line_l :=
by
  sorry

end minimize_AB_l82_82199


namespace sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l82_82756

-- Given conditions for the triangle ABC
variables {A B C a b c : ℝ}
axiom angle_C_eq_two_pi_over_three : C = 2 * Real.pi / 3
axiom c_squared_eq_five_a_squared_plus_ab : c^2 = 5 * a^2 + a * b

-- Proof statements
theorem sin_B_over_sin_A_eq_two (hAC: C = 2 * Real.pi / 3) (hCond: c^2 = 5 * a^2 + a * b) :
  Real.sin B / Real.sin A = 2 :=
sorry

theorem max_value_sin_A_sin_B (hAC: C = 2 * Real.pi / 3) :
  ∃ A B : ℝ, 0 < A ∧ A < Real.pi / 3 ∧ B = (Real.pi / 3 - A) ∧ Real.sin A * Real.sin B ≤ 1 / 4 :=
sorry

end sin_B_over_sin_A_eq_two_max_value_sin_A_sin_B_l82_82756


namespace total_packs_is_117_l82_82820

-- Defining the constants based on the conditions
def nancy_cards : ℕ := 540
def melanie_cards : ℕ := 620
def mary_cards : ℕ := 480
def alyssa_cards : ℕ := 720

def nancy_cards_per_pack : ℝ := 18.5
def melanie_cards_per_pack : ℝ := 22.5
def mary_cards_per_pack : ℝ := 15.3
def alyssa_cards_per_pack : ℝ := 24

-- Calculating the number of packs each person has
def nancy_packs := (nancy_cards : ℝ) / nancy_cards_per_pack
def melanie_packs := (melanie_cards : ℝ) / melanie_cards_per_pack
def mary_packs := (mary_cards : ℝ) / mary_cards_per_pack
def alyssa_packs := (alyssa_cards : ℝ) / alyssa_cards_per_pack

-- Rounding down the number of packs
def nancy_packs_rounded := nancy_packs.toNat
def melanie_packs_rounded := melanie_packs.toNat
def mary_packs_rounded := mary_packs.toNat
def alyssa_packs_rounded := alyssa_packs.toNat

-- Summing the total number of packs
def total_packs : ℕ := nancy_packs_rounded + melanie_packs_rounded + mary_packs_rounded + alyssa_packs_rounded

-- Proposition stating that the total number of packs is 117
theorem total_packs_is_117 : total_packs = 117 := by
  sorry

end total_packs_is_117_l82_82820


namespace hairstylist_monthly_earnings_l82_82147

noncomputable def hairstylist_earnings_per_month : ℕ :=
  let monday_wednesday_friday_earnings : ℕ := (4 * 10) + (3 * 15) + (1 * 22);
  let tuesday_thursday_earnings : ℕ := (6 * 10) + (2 * 15) + (3 * 30);
  let weekend_earnings : ℕ := (10 * 22) + (5 * 30);
  let weekly_earnings : ℕ :=
    (monday_wednesday_friday_earnings * 3) +
    (tuesday_thursday_earnings * 2) +
    (weekend_earnings * 2);
  weekly_earnings * 4

theorem hairstylist_monthly_earnings : hairstylist_earnings_per_month = 5684 := by
  -- Assertion based on the provided problem conditions
  sorry

end hairstylist_monthly_earnings_l82_82147


namespace seating_arrangement_7_people_l82_82764

theorem seating_arrangement_7_people (n : Nat) (h1 : n = 7) :
  let m := n - 1
  (m.factorial / m) * 2 = 240 :=
by
  sorry

end seating_arrangement_7_people_l82_82764


namespace find_g2_l82_82095

open Function

variable (g : ℝ → ℝ)

axiom g_condition : ∀ x : ℝ, g x + 2 * g (1 - x) = 5 * x ^ 2

theorem find_g2 : g 2 = -10 / 3 :=
by {
  sorry
}

end find_g2_l82_82095


namespace abs_x_minus_y_l82_82194

theorem abs_x_minus_y (x y : ℝ) (h₁ : x^3 + y^3 = 26) (h₂ : xy * (x + y) = -6) : |x - y| = 4 :=
by
  sorry

end abs_x_minus_y_l82_82194


namespace volume_after_increasing_edges_l82_82103

-- Defining the initial conditions and the theorem to prove regarding the volume.
theorem volume_after_increasing_edges {a b c : ℝ} 
  (h1 : a * b * c = 8) 
  (h2 : (a + 1) * (b + 1) * (c + 1) = 27) : 
  (a + 2) * (b + 2) * (c + 2) = 64 :=
sorry

end volume_after_increasing_edges_l82_82103


namespace road_construction_days_l82_82161

theorem road_construction_days
  (length_of_road : ℝ)
  (initial_men : ℕ)
  (completed_length : ℝ)
  (completed_days : ℕ)
  (extra_men : ℕ)
  (initial_days : ℕ)
  (remaining_length : ℝ)
  (remaining_days : ℕ)
  (total_men : ℕ) :
  length_of_road = 15 →
  initial_men = 30 →
  completed_length = 2.5 →
  completed_days = 100 →
  extra_men = 45 →
  initial_days = initial_days →
  remaining_length = length_of_road - completed_length →
  remaining_days = initial_days - completed_days →
  total_men = initial_men + extra_men →
  initial_days = 700 :=
by
  intros
  sorry

end road_construction_days_l82_82161


namespace crackers_count_l82_82072

theorem crackers_count (crackers_Marcus crackers_Mona crackers_Nicholas : ℕ) 
  (h1 : crackers_Marcus = 3 * crackers_Mona)
  (h2 : crackers_Nicholas = crackers_Mona + 6)
  (h3 : crackers_Marcus = 27) : crackers_Nicholas = 15 := 
by 
  sorry

end crackers_count_l82_82072


namespace regular_triangular_pyramid_volume_l82_82326

theorem regular_triangular_pyramid_volume (a γ : ℝ) : 
  ∃ V, V = (a^3 * Real.sin (γ / 2)^2) / (12 * Real.sqrt (1 - (Real.sin (γ / 2))^2)) := 
sorry

end regular_triangular_pyramid_volume_l82_82326


namespace condition_implies_at_least_one_gt_one_l82_82065

theorem condition_implies_at_least_one_gt_one (a b : ℝ) :
  (a + b > 2 → (a > 1 ∨ b > 1)) ∧ ¬(a^2 + b^2 > 2 → (a > 1 ∨ b > 1)) :=
by
  sorry

end condition_implies_at_least_one_gt_one_l82_82065


namespace find_line_through_and_perpendicular_l82_82972

def point (x y : ℝ) := (x, y)

def passes_through (P : ℝ × ℝ) (a b c : ℝ) :=
  a * P.1 + b * P.2 + c = 0

def is_perpendicular (a1 b1 a2 b2 : ℝ) :=
  a1 * a2 + b1 * b2 = 0

theorem find_line_through_and_perpendicular :
  ∃ c : ℝ, passes_through (1, -1) 1 1 c ∧ is_perpendicular 1 (-1) 1 1 → 
  c = 0 :=
by
  sorry

end find_line_through_and_perpendicular_l82_82972


namespace product_remainder_mod_7_l82_82880

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82880


namespace divide_polynomials_l82_82596

theorem divide_polynomials (n : ℕ) (h : ∃ (k : ℤ), n^2 + 3*n + 51 = 13 * k) : 
  ∃ (m : ℤ), 21*n^2 + 89*n + 44 = 169 * m := by
  sorry

end divide_polynomials_l82_82596


namespace find_y_when_z_is_three_l82_82605

theorem find_y_when_z_is_three
  (k : ℝ) (y z : ℝ)
  (h1 : y = 3)
  (h2 : z = 1)
  (h3 : y ^ 4 * z ^ 2 = k)
  (hc : z = 3) :
  y ^ 4 = 9 :=
sorry

end find_y_when_z_is_three_l82_82605


namespace tailor_trim_amount_l82_82156

variable (x : ℝ)

def original_side : ℝ := 22
def trimmed_side : ℝ := original_side - x
def fixed_trimmed_side : ℝ := original_side - 5
def remaining_area : ℝ := 120

theorem tailor_trim_amount :
  (original_side - x) * 17 = remaining_area → x = 15 :=
by
  intro h
  sorry

end tailor_trim_amount_l82_82156


namespace jennifer_fruits_left_l82_82794

open Nat

theorem jennifer_fruits_left :
  (p o a g : ℕ) → p = 10 → o = 20 → a = 2 * p → g = 2 → (p - g) + (o - g) + (a - g) = 44 :=
by
  intros p o a g h_p h_o h_a h_g
  rw [h_p, h_o, h_a, h_g]
  sorry

end jennifer_fruits_left_l82_82794


namespace solve_modulo_problem_l82_82112

theorem solve_modulo_problem (n : ℤ) :
  0 ≤ n ∧ n < 19 ∧ 38574 % 19 = n % 19 → n = 4 := by
  sorry

end solve_modulo_problem_l82_82112


namespace cos_330_eq_sqrt3_div_2_l82_82700

theorem cos_330_eq_sqrt3_div_2 :
  ∃ (θ : ℝ), θ = 330 ∧ cos θ = (√3)/2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82700


namespace probability_of_drawing_white_ball_l82_82439

def total_balls (red white : ℕ) : ℕ := red + white

def number_of_white_balls : ℕ := 2

def number_of_red_balls : ℕ := 3

def probability_of_white_ball (white total : ℕ) : ℚ := white / total

-- Theorem statement
theorem probability_of_drawing_white_ball :
  probability_of_white_ball number_of_white_balls (total_balls number_of_red_balls number_of_white_balls) = 2 / 5 :=
sorry

end probability_of_drawing_white_ball_l82_82439


namespace minimum_value_l82_82201

theorem minimum_value (a_n : ℕ → ℤ) (h : ∀ n, a_n n = n^2 - 8 * n + 5) : ∃ n, a_n n = -11 :=
by
  sorry

end minimum_value_l82_82201


namespace number_of_pencil_boxes_l82_82882

open Nat

def books_per_box : Nat := 46
def num_book_boxes : Nat := 19
def pencils_per_box : Nat := 170
def total_books_and_pencils : Nat := 1894

theorem number_of_pencil_boxes :
  (total_books_and_pencils - (num_book_boxes * books_per_box)) / pencils_per_box = 6 := 
by
  sorry

end number_of_pencil_boxes_l82_82882


namespace probability_of_three_primes_from_30_l82_82482

noncomputable def primes_up_to_30 : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19, 23, 29}

theorem probability_of_three_primes_from_30 :
  ((primes_up_to_30.card.choose 3) / ((Finset.range 31).card.choose 3)) = (6 / 203) :=
by
  sorry

end probability_of_three_primes_from_30_l82_82482


namespace incorrect_statement_A_l82_82946

/-- Let prob_beijing be the probability of rainfall in Beijing and prob_shanghai be the probability
of rainfall in Shanghai. We assert that statement (A) which claims "It is certain to rain in Beijing today, 
while it is certain not to rain in Shanghai" is incorrect given the probabilities. 
-/
theorem incorrect_statement_A (prob_beijing prob_shanghai : ℝ) 
  (h_beijing : prob_beijing = 0.8)
  (h_shanghai : prob_shanghai = 0.2)
  (statement_A : ¬ (prob_beijing = 1 ∧ prob_shanghai = 0)) : 
  true := 
sorry

end incorrect_statement_A_l82_82946


namespace area_of_square_l82_82887

-- Definitions
def radius_ratio (r R : ℝ) : Prop := R = 7 / 3 * r
def small_circle_circumference (r : ℝ) : Prop := 2 * Real.pi * r = 8
def square_side_length (R side : ℝ) : Prop := side = 2 * R
def square_area (side area : ℝ) : Prop := area = side * side

-- Problem statement
theorem area_of_square (r R side area : ℝ) 
    (h1 : radius_ratio r R)
    (h2 : small_circle_circumference r)
    (h3 : square_side_length R side)
    (h4 : square_area side area) :
    area = 3136 / (9 * Real.pi^2) := 
  by sorry

end area_of_square_l82_82887


namespace find_pairs_l82_82973

noncomputable def x (a b : ℝ) : ℝ := b^2 - (a - 1)/2
noncomputable def y (a b : ℝ) : ℝ := a^2 + (b + 1)/2
def valid_pair (a b : ℝ) : Prop := max (x a b) (y a b) ≤ 7 / 16

theorem find_pairs : valid_pair (1/4) (-1/4) :=
  sorry

end find_pairs_l82_82973


namespace remainder_product_l82_82871

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82871


namespace parking_lot_vehicle_spaces_l82_82885

theorem parking_lot_vehicle_spaces
  (total_spaces : ℕ)
  (spaces_per_caravan : ℕ)
  (num_caravans : ℕ)
  (remaining_spaces : ℕ) :
  total_spaces = 30 →
  spaces_per_caravan = 2 →
  num_caravans = 3 →
  remaining_spaces = total_spaces - (spaces_per_caravan * num_caravans) →
  remaining_spaces = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end parking_lot_vehicle_spaces_l82_82885


namespace impossible_to_load_two_coins_l82_82939

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l82_82939


namespace side_length_of_S2_l82_82080

variables (s r : ℕ)

-- Conditions
def combined_width_eq : Prop := 3 * s + 100 = 4000
def combined_height_eq : Prop := 2 * r + s = 2500

-- Conclusion we want to prove
theorem side_length_of_S2 : combined_width_eq s → combined_height_eq s r → s = 1300 :=
by
  intros h_width h_height
  sorry

end side_length_of_S2_l82_82080


namespace calculate_expression_l82_82524

theorem calculate_expression :
  (Real.sqrt 2 - 3)^0 - Real.sqrt 9 + |(-2: ℝ)| + ((-1/3: ℝ)⁻¹)^2 = 9 :=
by
  sorry

end calculate_expression_l82_82524


namespace arithmetic_geometric_inequality_l82_82246

theorem arithmetic_geometric_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a ≠ b) :
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  B < (a - b)^2 / (8 * (A - B)) ∧ (a - b)^2 / (8 * (A - B)) < A :=
by
  let A := (a + b) / 2
  let B := Real.sqrt (a * b)
  sorry

end arithmetic_geometric_inequality_l82_82246


namespace three_digit_oddfactors_count_l82_82357

open Int

theorem three_digit_oddfactors_count : 
  let S := {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, m * m = n}
  in S.card = 22 := by
  sorry

end three_digit_oddfactors_count_l82_82357


namespace wedge_volume_calculation_l82_82061

theorem wedge_volume_calculation :
  let r := 5 
  let h := 8 
  let V := (1 / 3) * (Real.pi * r^2 * h) 
  V = (200 * Real.pi) / 3 :=
by
  let r := 5
  let h := 8
  let V := (1 / 3) * (Real.pi * r^2 * h)
  -- Prove the equality step is omitted as per the prompt
  sorry

end wedge_volume_calculation_l82_82061


namespace coin_loading_impossible_l82_82923

theorem coin_loading_impossible (p q : ℝ) (h1 : p ≠ 1 - p) (h2 : q ≠ 1 - q) 
    (h3 : p * q = 1/4) (h4 : p * (1 - q) = 1/4) (h5 : (1 - p) * q = 1/4) (h6 : (1 - p) * (1 - q) = 1/4) : 
    false := 
by 
  sorry

end coin_loading_impossible_l82_82923


namespace three_digit_integers_with_odd_factors_l82_82397

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l82_82397


namespace largest_prime_divisor_36_squared_plus_81_squared_l82_82322

-- Definitions of the key components in the problem
def a := 36
def b := 81
def expr := a^2 + b^2
def largest_prime_divisor (n : ℕ) : ℕ := sorry -- Assume this function can compute the largest prime divisor

-- Theorem stating the problem
theorem largest_prime_divisor_36_squared_plus_81_squared : largest_prime_divisor (36^2 + 81^2) = 53 := 
  sorry

end largest_prime_divisor_36_squared_plus_81_squared_l82_82322


namespace smallest_n_not_divisible_by_10_l82_82012

theorem smallest_n_not_divisible_by_10 :
  ∃ n : ℕ, n > 2016 ∧ (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := by
  sorry

end smallest_n_not_divisible_by_10_l82_82012


namespace stable_performance_l82_82218

theorem stable_performance 
  (X_A_mean : ℝ) (X_B_mean : ℝ) (S_A_var : ℝ) (S_B_var : ℝ)
  (h1 : X_A_mean = 82) (h2 : X_B_mean = 82)
  (h3 : S_A_var = 245) (h4 : S_B_var = 190) : S_B_var < S_A_var :=
by {
  sorry
}

end stable_performance_l82_82218


namespace emily_dog_count_l82_82967

theorem emily_dog_count (dogs : ℕ) 
  (food_per_day_per_dog : ℕ := 250) 
  (vacation_days : ℕ := 14)
  (total_food_kg : ℕ := 14)
  (kg_to_grams : ℕ := 1000) 
  (total_food_grams : ℕ := total_food_kg * kg_to_grams)
  (food_needed_per_dog : ℕ := food_per_day_per_dog * vacation_days) 
  (total_food_needed : ℕ := dogs * food_needed_per_dog) 
  (h : total_food_needed = total_food_grams) : 
  dogs = 4 := 
sorry

end emily_dog_count_l82_82967


namespace integer_solutions_l82_82176

theorem integer_solutions (x y : ℤ) : 
  x^2 * y = 10000 * x + y ↔ 
  (x, y) = (-9, -1125) ∨ 
  (x, y) = (-3, -3750) ∨ 
  (x, y) = (0, 0) ∨ 
  (x, y) = (3, 3750) ∨ 
  (x, y) = (9, 1125) := 
by
  sorry

end integer_solutions_l82_82176


namespace intersection_complement_eq_l82_82202

open Set

variable (U A B : Set ℕ)

theorem intersection_complement_eq :
  (U = {1, 2, 3, 4, 5, 6}) →
  (A = {1, 3}) →
  (B = {3, 4, 5}) →
  A ∩ (U \ B) = {1} :=
by
  intros hU hA hB
  subst hU
  subst hA
  subst hB
  sorry

end intersection_complement_eq_l82_82202


namespace math_problem_l82_82741

noncomputable def log_8 := Real.log 8
noncomputable def log_27 := Real.log 27
noncomputable def expr := (9 : ℝ) ^ (log_8 / log_27) + (2 : ℝ) ^ (log_27 / log_8)

theorem math_problem : expr = 7 := by
  sorry

end math_problem_l82_82741


namespace remainder_is_correct_l82_82277

def dividend : ℕ := 725
def divisor : ℕ := 36
def quotient : ℕ := 20

theorem remainder_is_correct : ∃ (remainder : ℕ), dividend = (divisor * quotient) + remainder ∧ remainder = 5 := by
  sorry

end remainder_is_correct_l82_82277


namespace remainder_product_l82_82867

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82867


namespace sum_of_coords_A_l82_82785

variables (a b : ℝ)
noncomputable def point_A_coords := [(8, 12), (1, 12)]

theorem sum_of_coords_A : 
  ∀ (A : ℝ × ℝ), 
    A ∈ point_A_coords → 
    ∃ (x y : ℝ), A = (x, y) ∧ (x + y = 13 ∨ x + y = 20) :=
by
  intro A
  intro hA
  cases hA
  case inl =>
    use 8, 12
    split
    rfl
    right
    norm_num
  case inr =>
    use 1, 12
    split
    rfl
    left
    norm_num

end sum_of_coords_A_l82_82785


namespace number_of_8_digit_integers_l82_82993

theorem number_of_8_digit_integers : 
  ∃ n, n = 90000000 ∧ 
    (∀ (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ), 
     d1 ≠ 0 → 0 ≤ d1 ∧ d1 ≤ 9 ∧ 
     0 ≤ d2 ∧ d2 ≤ 9 ∧ 
     0 ≤ d3 ∧ d3 ≤ 9 ∧ 
     0 ≤ d4 ∧ d4 ≤ 9 ∧ 
     0 ≤ d5 ∧ d5 ≤ 9 ∧ 
     0 ≤ d6 ∧ d6 ≤ 9 ∧ 
     0 ≤ d7 ∧ d7 ≤ 9 ∧ 
     0 ≤ d8 ∧ d8 ≤ 9 →
     ∀ count, count = (if d1 ≠ 0 then 9 * 10^7 else 0)) :=
sorry

end number_of_8_digit_integers_l82_82993


namespace find_x_y_l82_82190

theorem find_x_y (a n x y : ℕ) (hx4 : 1000 ≤ x ∧ x < 10000) (hy4 : 1000 ≤ y ∧ y < 10000) 
  (h_yx : y > x) (h_y : y = a * 10 ^ n) 
  (h_sum : (x / 1000) + ((x % 1000) / 100) = 5 * a) 
  (ha : a = 2) (hn : n = 3) :
  x = 1990 ∧ y = 2000 := 
by 
  sorry

end find_x_y_l82_82190


namespace no_prime_divisible_by_42_l82_82203

theorem no_prime_divisible_by_42 : 
  ∀ p : ℕ, Prime p → ¬ (42 ∣ p) := 
by
  intro p hp hdiv
  have h2 : 2 ∣ p := dvd_of_mul_right_dvd hdiv
  have h3 : 3 ∣ p := dvd_of_mul_left_dvd (dvd_of_mul_right_dvd hdiv)
  have h7 : 7 ∣ p := dvd_of_mul_left_dvd hdiv
  sorry

end no_prime_divisible_by_42_l82_82203


namespace calculate_delta_nabla_l82_82723

-- Define the operations Δ and ∇
def delta (a b : ℤ) : ℤ := 3 * a + 2 * b
def nabla (a b : ℤ) : ℤ := 2 * a + 3 * b

-- Formalize the theorem
theorem calculate_delta_nabla : delta 3 (nabla 2 1) = 23 := 
by 
  -- Placeholder for proof, not required by the question
  sorry

end calculate_delta_nabla_l82_82723


namespace solution_l82_82307

theorem solution :
  ∀ (x : ℝ), x ≠ 0 → (9 * x) ^ 18 = (27 * x) ^ 9 → x = 1 / 3 :=
by
  intro x
  intro h
  intro h_eq
  sorry

end solution_l82_82307


namespace coin_loading_impossible_l82_82934

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l82_82934


namespace three_digit_oddfactors_count_is_22_l82_82347

theorem three_digit_oddfactors_count_is_22 :
  let lower_bound := 100
  let upper_bound := 999
  let smallest_square_root := 10
  let largest_square_root := 31
  let count := largest_square_root - smallest_square_root + 1
  ∀ n : ℤ, lower_bound ≤ n ∧ n ≤ upper_bound → (∃ k : ℤ, n = k * k →
  (smallest_square_root ≤ k ∧ k ≤ largest_square_root)) → count = 22 :=
by
  intros lower_bound upper_bound smallest_square_root largest_square_root count n hn h
  have smallest_square_root := 10
  have largest_square_root := 31
  have count := largest_square_root - smallest_square_root + 1
  sorry

end three_digit_oddfactors_count_is_22_l82_82347


namespace eval_expression_l82_82619

theorem eval_expression : 3 * 4^2 - (8 / 2) = 44 := by
  sorry

end eval_expression_l82_82619


namespace num_odd_factors_of_three_digit_integers_eq_22_l82_82412

theorem num_odd_factors_of_three_digit_integers_eq_22 : 
  let is_three_digit (n : ℕ) := 100 <= n ∧ n <= 999
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n
  ∧ let is_perfect_square (n : ℕ) := ∃ m : ℕ, m * m = n in
  ∃ finset.card (n ∈ finset.range 900) (is_three_digit n ∧ is_perfect_square n) = 22 :=
by sorry

end num_odd_factors_of_three_digit_integers_eq_22_l82_82412


namespace cos_330_eq_sqrt3_div_2_l82_82665

theorem cos_330_eq_sqrt3_div_2 : Real.cos (330 * Real.pi / 180) = Real.sqrt 3 / 2 := by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82665


namespace three_digit_integers_odd_factors_count_l82_82369

theorem three_digit_integers_odd_factors_count : ∃ n : ℕ, 
  (n ≥ 10 ∧ n ≤ 31) ∧
  (set.count (λ x, x ≥ 10 ∧ x ≤ 31) (λ x, x^2 ∈ set.Icc 100 999) = 22) := 
by {
  sorry
}

end three_digit_integers_odd_factors_count_l82_82369


namespace savings_by_end_of_2019_l82_82574

variable (income_monthly : ℕ → ℕ) (expenses_monthly : ℕ → ℕ)
variable (initial_savings : ℕ)

noncomputable def total_income : ℕ :=
  (income_monthly 9 + income_monthly 10 + income_monthly 11 + income_monthly 12) * 4

noncomputable def total_expenses : ℕ :=
  (expenses_monthly 9 + expenses_monthly 10 + expenses_monthly 11 + expenses_monthly 12) * 4

noncomputable def final_savings (initial_savings : ℕ) (total_income : ℕ) (total_expenses : ℕ) : ℕ :=
  initial_savings + total_income - total_expenses

theorem savings_by_end_of_2019 :
  (income_monthly 9 = 55000) →
  (income_monthly 10 = 45000) →
  (income_monthly 11 = 10000) →
  (income_monthly 12 = 17400) →
  (expenses_monthly 9 = 40000) →
  (expenses_monthly 10 = 20000) →
  (expenses_monthly 11 = 5000) →
  (expenses_monthly 12 = 2000) →
  initial_savings = 1147240 →
  final_savings initial_savings total_income total_expenses = 1340840 :=
by
  intros h_income_9 h_income_10 h_income_11 h_income_12
         h_expenses_9 h_expenses_10 h_expenses_11 h_expenses_12
         h_initial_savings
  rw [final_savings, total_income, total_expenses]
  rw [h_income_9, h_income_10, h_income_11, h_income_12]
  rw [h_expenses_9, h_expenses_10, h_expenses_11, h_expenses_12]
  rw h_initial_savings
  sorry

end savings_by_end_of_2019_l82_82574


namespace minimum_3x_4y_l82_82037

theorem minimum_3x_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = 5 * x * y) : 3 * x + 4 * y ≥ 5 :=
by
  sorry

end minimum_3x_4y_l82_82037


namespace remainder_of_product_l82_82854

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82854


namespace avg_weight_A_and_B_l82_82608

-- Definitions for the weights of A, B, and C
variables (A B C : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (A + B + C) / 3 = 45
def condition2 : Prop := (B + C) / 2 = 43
def condition3 : Prop := B = 31

-- Statement to be proved
theorem avg_weight_A_and_B : condition1 A B C ∧ condition2 A B C ∧ condition3 B → (A + B) / 2 = 40 :=
begin
  intros h,
  have h1 := h.1,
  have h2 := h.2.1,
  have h3 := h.2.2,
  sorry -- Proof goes here
end

end avg_weight_A_and_B_l82_82608


namespace katie_earnings_l82_82494

-- Define the constants for the problem
def bead_necklaces : Nat := 4
def gemstone_necklaces : Nat := 3
def cost_per_necklace : Nat := 3

-- Define the total earnings calculation
def total_necklaces : Nat := bead_necklaces + gemstone_necklaces
def total_earnings : Nat := total_necklaces * cost_per_necklace

-- Statement of the proof problem
theorem katie_earnings : total_earnings = 21 := by
  sorry

end katie_earnings_l82_82494


namespace erasers_left_l82_82480

/-- 
There are initially 250 erasers in a box. Doris takes 75 erasers, Mark takes 40 
erasers, and Ellie takes 30 erasers out of the box. Prove that 105 erasers are 
left in the box.
-/
theorem erasers_left (initial_erasers : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ)
  (h_initial : initial_erasers = 250)
  (h_doris : doris_takes = 75)
  (h_mark : mark_takes = 40)
  (h_ellie : ellie_takes = 30) :
  initial_erasers - doris_takes - mark_takes - ellie_takes = 105 :=
  by 
  sorry

end erasers_left_l82_82480


namespace determinant_zero_l82_82000

noncomputable def matrix_A (θ φ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 2 * Real.sin θ, -Real.cos θ],
    ![-2 * Real.sin θ, 0, Real.sin φ],
    ![Real.cos θ, -Real.sin φ, 0]]

theorem determinant_zero (θ φ : ℝ) : Matrix.det (matrix_A θ φ) = 0 := by
  sorry

end determinant_zero_l82_82000


namespace typing_problem_l82_82748

theorem typing_problem (a b m n : ℕ) (h1 : 60 = a * b) (h2 : 540 = 75 * n) (h3 : n = 3 * m) :
  a = 25 :=
by {
  -- sorry placeholder where the proof would go
  sorry
}

end typing_problem_l82_82748


namespace youngest_child_is_five_l82_82106

-- Define the set of prime numbers
def is_prime (n: ℕ) := n > 1 ∧ ∀ m: ℕ, m ∣ n → m = 1 ∨ m = n

-- Define the ages of the children
def youngest_child_age (x: ℕ) : Prop :=
  is_prime x ∧
  is_prime (x + 2) ∧
  is_prime (x + 6) ∧
  is_prime (x + 8) ∧
  is_prime (x + 12) ∧
  is_prime (x + 14)

-- The main theorem stating the age of the youngest child
theorem youngest_child_is_five : ∃ x: ℕ, youngest_child_age x ∧ x = 5 :=
  sorry

end youngest_child_is_five_l82_82106


namespace three_digit_integers_with_odd_factors_l82_82417

theorem three_digit_integers_with_odd_factors :
  ∃ n : ℕ, n = 22 ∧ (∀ x : ℕ, 100 ≤ x ∧ x ≤ 999 → (∃ y : ℕ, y * y = x → x ∈ (10^2 .. 31^2))) :=
sorry

end three_digit_integers_with_odd_factors_l82_82417


namespace savings_correct_l82_82576

def initial_savings : ℕ := 1147240
def total_income : ℕ := (55000 + 45000 + 10000 + 17400) * 4
def total_expenses : ℕ := (40000 + 20000 + 5000 + 2000 + 2000) * 4
def final_savings : ℕ := initial_savings + total_income - total_expenses

theorem savings_correct : final_savings = 1340840 :=
by
  sorry

end savings_correct_l82_82576


namespace sum_distances_between_l82_82573

noncomputable def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2).sqrt

theorem sum_distances_between (A B D : ℝ × ℝ)
  (hB : B = (0, 5))
  (hD : D = (8, 0))
  (hA : A = (20, 0)) :
  21 < distance A D + distance B D ∧ distance A D + distance B D < 22 :=
by
  sorry

end sum_distances_between_l82_82573


namespace cos_330_cos_30_val_answer_l82_82692

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l82_82692


namespace enrique_commission_l82_82710

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l82_82710


namespace remainder_product_l82_82866

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82866


namespace remainder_of_product_l82_82852

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82852


namespace functional_equation_initial_condition_unique_f3_l82_82066

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation (x y : ℝ) : f (f x + y) = f (x ^ 2 - y) + 2 * f x * y := sorry

theorem initial_condition : f 1 = 1 := sorry

theorem unique_f3 : f 3 = 9 := sorry

end functional_equation_initial_condition_unique_f3_l82_82066


namespace arithmetic_seq_middle_term_l82_82121

theorem arithmetic_seq_middle_term (a1 a3 y : ℤ) (h1 : a1 = 3^2) (h2 : a3 = 3^4)
    (h3 : y = (a1 + a3) / 2) : y = 45 :=
by
  rw [h1, h2] at h3
  simp at h3
  exact h3

end arithmetic_seq_middle_term_l82_82121


namespace wall_width_l82_82145

theorem wall_width
  (brick_length : ℝ) (brick_width : ℝ) (brick_height : ℝ)
  (wall_length : ℝ) (wall_height : ℝ)
  (num_bricks : ℕ)
  (brick_volume : ℝ := brick_length * brick_width * brick_height)
  (total_volume : ℝ := num_bricks * brick_volume) :
  brick_length = 0.20 → brick_width = 0.10 → brick_height = 0.08 →
  wall_length = 10 → wall_height = 8 → num_bricks = 12250 →
  total_volume = wall_length * wall_height * (0.245 : ℝ) :=
by 
  sorry

end wall_width_l82_82145


namespace batsman_average_increase_l82_82919

theorem batsman_average_increase (A : ℕ) (H1 : 16 * A + 85 = 17 * (A + 3)) : A + 3 = 37 :=
by {
  sorry
}

end batsman_average_increase_l82_82919


namespace shaded_fraction_l82_82292

theorem shaded_fraction {S : ℝ} (h : 0 < S) :
  let frac_area := ∑' n : ℕ, (1/(4:ℝ)^1) * (1/(4:ℝ)^n)
  1/3 = frac_area :=
by
  sorry

end shaded_fraction_l82_82292


namespace money_first_day_l82_82535

-- Define the total mushrooms
def total_mushrooms : ℕ := 65

-- Define the mushrooms picked on the second day
def mushrooms_day2 : ℕ := 12

-- Define the mushrooms picked on the third day
def mushrooms_day3 : ℕ := 2 * mushrooms_day2

-- Define the price per mushroom
def price_per_mushroom : ℕ := 2

-- Prove that the amount of money made on the first day is $58
theorem money_first_day : (total_mushrooms - mushrooms_day2 - mushrooms_day3) * price_per_mushroom = 58 := 
by
  -- Skip the proof
  sorry

end money_first_day_l82_82535


namespace sin_ineq_l82_82975

open Real

theorem sin_ineq (n : ℕ) (h : n > 0) : sin (π / (4 * n)) ≥ (sqrt 2) / (2 * n) :=
sorry

end sin_ineq_l82_82975


namespace figure_surface_area_calculation_l82_82231

-- Define the surface area of one bar
def bar_surface_area : ℕ := 18

-- Define the surface area lost at the junctions
def surface_area_lost : ℕ := 2

-- Define the effective surface area of one bar after accounting for overlaps
def effective_bar_surface_area : ℕ := bar_surface_area - surface_area_lost

-- Define the number of bars used in the figure
def number_of_bars : ℕ := 4

-- Define the total surface area of the figure
def total_surface_area : ℕ := number_of_bars * effective_bar_surface_area

-- The theorem stating the total surface area of the figure
theorem figure_surface_area_calculation : total_surface_area = 64 := by
  sorry

end figure_surface_area_calculation_l82_82231


namespace find_angle_A_l82_82757

theorem find_angle_A (a b c A B C : ℝ)
  (h1 : a^2 - b^2 = Real.sqrt 3 * b * c)
  (h2 : Real.sin C = 2 * Real.sqrt 3 * Real.sin B) :
  A = Real.pi / 6 :=
sorry

end find_angle_A_l82_82757


namespace gcd_75_100_l82_82911

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l82_82911


namespace cos_330_eq_sqrt_3_div_2_l82_82653

theorem cos_330_eq_sqrt_3_div_2 : Real.cos (330 * Real.pi / 180) = (Real.sqrt 3 / 2) :=
by
  sorry

end cos_330_eq_sqrt_3_div_2_l82_82653


namespace inequality_with_equality_condition_l82_82083

variable {a b c d : ℝ}

theorem inequality_with_equality_condition (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) : 
  (a^2 + b^2 + c^2 + d^2)^2 ≥ (a + b) * (b + c) * (c + d) * (d + a) ∧ 
  ((a^2 + b^2 + c^2 + d^2)^2 = (a + b) * (b + c) * (c + d) * (d + a) ↔ a = b ∧ b = c ∧ c = d) := sorry

end inequality_with_equality_condition_l82_82083


namespace sum_coordinates_A_l82_82773

-- Definitions and given conditions
variables {α : Type*} [linear_ordered_field α]
variables (a b : α)
variables (A : α × α) (B : α × α) (C : α × α)

-- Lines in the system specified
def line1 := λ (x : α), a * x + 4
def line2 := λ (x : α), 2 * x + b
def line3 := λ (x : α), (a / 2) * x + 8

-- Conditions on points B and C
def on_Ox_axis (P : α × α) : Prop := P.2 = 0
def on_Oy_axis (P : α × α) : Prop := P.1 = 0
def lines_intersect_at (l₁ l₂ : α → α) (P : α × α) : Prop := l₁ P.1 = P.2 ∧ l₂ P.1 = P.2

-- Statement to prove
theorem sum_coordinates_A :
  (on_Ox_axis B) →
  (on_Oy_axis C) →
  (lines_intersect_at line1 line2 B ∨ lines_intersect_at line2 line3 B) →
  (lines_intersect_at line1 line3 A) →
  (∃ s : α, s = A.1 + A.2 ∧ (s = 13 ∨ s = 20)) :=
begin
  intro hB,
  intro hC,
  intro hB_inter,
  intro hA_inter,
  sorry
end

end sum_coordinates_A_l82_82773


namespace roots_of_quadratic_l82_82338

theorem roots_of_quadratic (x1 x2 : ℝ) (h : ∀ x, x^2 - 3 * x - 2 = 0 → x = x1 ∨ x = x2) :
  x1 * x2 + x1 + x2 = 1 :=
sorry

end roots_of_quadratic_l82_82338


namespace unique_solution_condition_l82_82743

theorem unique_solution_condition (p q : ℝ) : 
  (∃! x : ℝ, 4 * x - 7 + p = q * x + 2) ↔ q ≠ 4 :=
by
  sorry

end unique_solution_condition_l82_82743


namespace original_recipe_calls_for_4_tablespoons_l82_82520

def key_limes := 8
def juice_per_lime := 1 -- in tablespoons
def juice_doubled := key_limes * juice_per_lime
def original_juice_amount := juice_doubled / 2

theorem original_recipe_calls_for_4_tablespoons :
  original_juice_amount = 4 :=
by
  sorry

end original_recipe_calls_for_4_tablespoons_l82_82520


namespace tangent_line_parallel_x_axis_coordinates_l82_82431

theorem tangent_line_parallel_x_axis_coordinates :
  (∃ P : ℝ × ℝ, P = (1, -2) ∨ P = (-1, 2)) ↔
  (∃ x y : ℝ, y = x^3 - 3 * x ∧ ∃ y', y' = 3 * x^2 - 3 ∧ y' = 0) :=
by
  sorry

end tangent_line_parallel_x_axis_coordinates_l82_82431


namespace cos_330_eq_sqrt3_div_2_l82_82701

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82701


namespace sum_of_coordinates_A_l82_82788

-- Define the problem settings and the required conditions
theorem sum_of_coordinates_A (a b : ℝ) (A B C : ℝ × ℝ) :
  -- Point B lies on the Ox axis
  B.snd = 0 →
  -- Point C lies on the Oy axis
  C.fst = 0 →
  -- Equations of lines given in some order
  (A.snd = a * A.fst + 4 ∧ C.snd = 2 * C.fst + b ∧ B.snd = (a / 2) * B.fst + 8) ∨ 
  (B.snd = a * A.fst + 4 ∧ A.snd = 2 * C.fst + b ∧ C.snd = (a / 2) * B.fst + 8) ∨
  (C.snd = a * A.fst + 4 ∧ B.snd = 2 * C.fst + b ∧ A.snd = (a / 2) * B.fst + 8) →
  -- Prove the sum of the coordinates of point A
  A.fst + A.snd = 13 ∨ A.fst + A.snd = 20 :=
by
  sorry

end sum_of_coordinates_A_l82_82788


namespace cos_330_eq_sqrt3_over_2_l82_82670

theorem cos_330_eq_sqrt3_over_2 :
  let θ := 330
  let complementaryAngle := 360 - θ
  let cos_pos30 := Real.cos 30 = (Real.sqrt 3 / 2)
  let sin_pos30 := Real.sin 30 = (1 / 2)
  let cos_θ := if (330 < 360 ∧ complementaryAngle = 30 ∧ θ = 330) then cos_pos30 else 0
  cos_θ = (Real.sqrt 3 / 2) := sorry

end cos_330_eq_sqrt3_over_2_l82_82670


namespace frosting_cupcakes_l82_82521

noncomputable def rate_cagney := 1 / 25  -- Cagney's rate in cupcakes per second
noncomputable def rate_lacey := 1 / 20  -- Lacey's rate in cupcakes per second

noncomputable def break_time := 30      -- Break time in seconds
noncomputable def work_period := 180    -- Work period in seconds before a break
noncomputable def total_time := 600     -- Total time in seconds (10 minutes)

noncomputable def combined_rate := rate_cagney + rate_lacey -- Combined rate in cupcakes per second

-- Effective work time after considering breaks
noncomputable def effective_work_time :=
  total_time - (total_time / work_period) * break_time

-- Total number of cupcakes frosted in the effective work time
noncomputable def total_cupcakes := combined_rate * effective_work_time

theorem frosting_cupcakes : total_cupcakes = 48 :=
by
  sorry

end frosting_cupcakes_l82_82521


namespace triangle_right_angled_l82_82227

theorem triangle_right_angled
  (a b c : ℝ) (A B C : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0)
  (h₄ : A + B + C = π)
  (h₅ : b * Real.cos C + c * Real.cos B = a * Real.sin A) :
  A = π / 2 ∨ B = π / 2 ∨ C = π / 2 :=
sorry

end triangle_right_angled_l82_82227


namespace gcd_75_100_l82_82894

theorem gcd_75_100 : ∀ (a b: ℕ), a = 75 → b = 100 → (Nat.gcd a b = 25) := 
by
  intros a b ha hb
  have h75 : a = 3 * 5^2 := by rw [ha]
  have h100 : b = 2^2 * 5^2 := by rw [hb]
  sorry

end gcd_75_100_l82_82894


namespace sum_of_areas_is_72_l82_82264

def base : ℕ := 2
def length1 : ℕ := 1
def length2 : ℕ := 8
def length3 : ℕ := 27

theorem sum_of_areas_is_72 : base * length1 + base * length2 + base * length3 = 72 :=
by
  sorry

end sum_of_areas_is_72_l82_82264


namespace people_left_first_hour_l82_82288

theorem people_left_first_hour 
  (X : ℕ)
  (h1 : X ≥ 0)
  (h2 : 94 - X + 18 - 9 = 76) :
  X = 27 := 
sorry

end people_left_first_hour_l82_82288


namespace discount_savings_difference_l82_82957

def cover_price : ℝ := 30
def discount_amount : ℝ := 5
def discount_percentage : ℝ := 0.25

theorem discount_savings_difference :
  let price_after_discount := cover_price - discount_amount
  let price_after_percentage_first := cover_price * (1 - discount_percentage)
  let new_price_after_percentage := price_after_discount * (1 - discount_percentage)
  let new_price_after_discount := price_after_percentage_first - discount_amount
  (new_price_after_percentage - new_price_after_discount) * 100 = 125 :=
by
  sorry

end discount_savings_difference_l82_82957


namespace correct_option_l82_82273

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end correct_option_l82_82273


namespace evaluate_expression_l82_82716

-- Define the integers a and b
def a := 2019
def b := 2020

-- The main theorem stating the equivalence
theorem evaluate_expression :
  (a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3 + 6) / (a * b) = 5 / (a * b) := 
by
  sorry

end evaluate_expression_l82_82716


namespace correct_option_l82_82274

-- Conditions as definitions
def optionA (a : ℝ) : Prop := a^2 * a^3 = a^6
def optionB (a : ℝ) : Prop := 3 * a - 2 * a = 1
def optionC (a : ℝ) : Prop := (-2 * a^2)^3 = -8 * a^6
def optionD (a : ℝ) : Prop := a^6 / a^2 = a^3

-- The statement to prove
theorem correct_option (a : ℝ) : optionC a :=
by 
  unfold optionC
  sorry

end correct_option_l82_82274


namespace sum_of_coordinates_A_l82_82776

-- Define points and equations
def point (x y : ℝ) := (x, y)

variable (a b : ℝ)

-- Lines defined by equations
def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, (a / 2) * x + 8

-- Conditions for points B and C
variable (xA yA : ℝ)
variable hA1 : a ≠ 0
variable hA2 : (point B on Ox axis)
variable hA3 : (point C on Oy axis)

-- Proof goal: Sum of coordinates of point A
theorem sum_of_coordinates_A :
    (∃ a b : ℝ, a ≠ 0
        ∧ (let l1 := line1 in
           let l2 := line2 in
           let l3 := line3 in
           let A := point xA yA in -- A is the intersection of any two lines based on given conditions
           (line1 xA = yA ∧ line2 xA = yA) ∨ -- A intersect line1 and line2
           (line2 xA = yA ∧ line3 xA = yA) ∨ -- A intersect line2 and line3
           (line1 xA = yA ∧ line3 xA = yA))  -- A intersect line1 and line3
        ∧ (xA + yA = 20 ∨ xA + yA = 13)) :=
sorry

end sum_of_coordinates_A_l82_82776


namespace probability_of_event_l82_82986

noncomputable def probability_gteq (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : Prop :=
  7 * x - 3 ≥ 0

theorem probability_of_event : 
  (∀ x, probability_gteq x → measure_theory.measure_space.probability_space.measure (set.Icc x 1) (measure_theory.measure_space.borel ℝ) = 4 / 7) :=
sorry

end probability_of_event_l82_82986


namespace equivalent_spherical_coords_l82_82223

theorem equivalent_spherical_coords (ρ θ φ : ℝ) (hρ : ρ = 4) (hθ : θ = 3 * π / 8) (hφ : φ = 9 * π / 5) :
  ∃ (ρ' θ' φ' : ℝ), ρ' = 4 ∧ θ' = 11 * π / 8 ∧ φ' = π / 5 ∧ 
  (ρ' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * π ∧ 0 ≤ φ' ∧ φ' ≤ π) :=
by
  sorry

end equivalent_spherical_coords_l82_82223


namespace num_three_digit_ints_with_odd_factors_l82_82361

theorem num_three_digit_ints_with_odd_factors : ∃ n, n = 22 ∧ ∀ k, 10 ≤ k ∧ k ≤ 31 → (∃ m, m = k * k ∧ 100 ≤ m ∧ m ≤ 999) :=
by
  -- outline of proof
  sorry

end num_three_digit_ints_with_odd_factors_l82_82361


namespace linear_eq_k_l82_82732

theorem linear_eq_k (k : ℕ) : (∀ x : ℝ, x^(k-1) + 3 = 0 ↔ k = 2) :=
by
  sorry

end linear_eq_k_l82_82732


namespace integer_solutions_l82_82318

-- Define the problem statement in Lean
theorem integer_solutions :
  {p : ℤ × ℤ | ∃ x y : ℤ, p = (x, y) ∧ x^2 + x = y^4 + y^3 + y^2 + y} =
  {(-1, -1), (0, -1), (-1, 0), (0, 0), (5, 2), (-6, 2)} :=
by
  sorry

end integer_solutions_l82_82318


namespace medians_square_sum_l82_82485

theorem medians_square_sum (a b c : ℝ) (ha : a = 13) (hb : b = 13) (hc : c = 10) :
  let m_a := (1 / 2 * (2 * b^2 + 2 * c^2 - a^2))^(1/2)
  let m_b := (1 / 2 * (2 * c^2 + 2 * a^2 - b^2))^(1/2)
  let m_c := (1 / 2 * (2 * a^2 + 2 * b^2 - c^2))^(1/2)
  m_a^2 + m_b^2 + m_c^2 = 432 :=
by
  sorry

end medians_square_sum_l82_82485


namespace tan_600_eq_sqrt3_l82_82742

theorem tan_600_eq_sqrt3 : (Real.tan (600 * Real.pi / 180)) = Real.sqrt 3 := 
by 
  -- sorry to skip the actual proof steps
  sorry

end tan_600_eq_sqrt3_l82_82742


namespace smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l82_82028

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

-- Statement A: The smallest positive period of f(x) is π.
theorem smallest_positive_period_pi : 
  ∀ x : ℝ, f (x + Real.pi) = f x :=
by sorry

-- Statement B: If f(x + θ) is an odd function, then one possible value of θ is π/4.
theorem not_odd_at_theta_pi_div_4 : 
  ¬ (∀ x : ℝ, f (x + Real.pi / 4) = -f x) :=
by sorry

-- Statement C: A possible axis of symmetry for f(x) is the line x = π / 3.
theorem axis_of_symmetry_at_pi_div_3 :
  ∀ x : ℝ, f (Real.pi / 3 - x) = f (Real.pi / 3 + x) :=
by sorry

-- Statement D: The maximum value of f(x) on [0, π / 4] is 1.
theorem max_value_not_1_on_interval : 
  ¬ (∀ x ∈ Set.Icc 0 (Real.pi / 4), f x ≤ 1) :=
by sorry

end smallest_positive_period_pi_not_odd_at_theta_pi_div_4_axis_of_symmetry_at_pi_div_3_max_value_not_1_on_interval_l82_82028


namespace num_divisors_with_divisor_count_1806_l82_82705

/-
Define the prime factorization of 1806 and its power.
-/
def prime_factors_1806 : List (ℕ × ℕ) := [(2, 1), (3, 2), (101, 1)]

/-
Define the prime factorization of 1806^1806.
-/
def prime_factors_1806_pow : List (ℕ × ℕ) := [(2, 1806), (3, 3612), (101, 1806)]

/-
Define a positive divisor of 1806^1806.
-/
def is_divisor (d : ℕ) : Prop :=
  ∃ (a b c : ℕ),
    d = 2^a * 3^b * 101^c ∧ (a + 1) * (b + 1) * (c + 1) = 1806

/-
Define the problem statement.
-/
theorem num_divisors_with_divisor_count_1806 :
  (Finset.univ.filter is_divisor).card = 36 :=
begin
  sorry
end

end num_divisors_with_divisor_count_1806_l82_82705


namespace max_b_c_l82_82542

theorem max_b_c (a b c : ℤ) (ha : a > 0) 
  (h1 : a - b + c = 4) 
  (h2 : 4 * a + 2 * b + c = 1) 
  (h3 : (b ^ 2) - 4 * a * c > 0) :
  -3 * a + 2 = -4 := 
sorry

end max_b_c_l82_82542


namespace number_of_three_digit_integers_with_odd_number_of_factors_l82_82404

theorem number_of_three_digit_integers_with_odd_number_of_factors :
  {n : ℕ | 100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k}.card = 22 :=
sorry

end number_of_three_digit_integers_with_odd_number_of_factors_l82_82404


namespace evaluate_Y_l82_82740

def Y (a b : ℤ) : ℤ := a^2 - 3 * a * b + b^2 + 3

theorem evaluate_Y : Y 2 5 = 2 :=
by
  sorry

end evaluate_Y_l82_82740


namespace base7_addition_l82_82314

theorem base7_addition (Y X : Nat) (k m : Int) :
    (Y + 2 = X + 7 * k) ∧ (X + 5 = 4 + 7 * m) ∧ (5 = 6 + 7 * -1) → X + Y = 10 :=
by
  sorry

end base7_addition_l82_82314


namespace find_A_coordinates_sum_l82_82766

-- Define points A, B, and C
structure Point :=
  (x : ℝ)
  (y : ℝ)

-- Define lines l1, l2, l3
def line1 (a : ℝ) := λ (x : ℝ), a * x + 4
def line2 (b : ℟) := λ (x : ℝ), 2 * x + b
def line3 (a : ℝ) := λ (x : ℝ), (a / 2) * x + 8

-- Define the conditions for the points A, B, and C
-- B lies on the x-axis at (xb, 0)
-- C lies on the y-axis at (0, yc)

noncomputable def A_coordinates (a b : ℝ) (A B C : Point) : Prop :=
  (A = ⟨B.x, line1 a B.x⟩ ∨ A = ⟨B.x, line2 b B.x⟩ ∨ A = ⟨C.y, line3 a C.y⟩) ∧
  (B = ⟨C.y, 0⟩)

-- Sum of coordinates of A
def sum_A (A : Point) : ℝ :=
  A.x + A.y

theorem find_A_coordinates_sum (a b : ℝ) (A B C : Point) 
  (A_coord : A_coordinates a b A B C) :
  sum_A A = 13 ∨ sum_A A = 20 :=
sorry

end find_A_coordinates_sum_l82_82766


namespace cos_330_eq_sqrt3_div_2_l82_82693

theorem cos_330_eq_sqrt3_div_2 :
  cos (330 * real.pi / 180) = sqrt 3 / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82693


namespace equal_angles_proof_l82_82833

/-- Proof Problem: After how many minutes will the hour and minute hands form equal angles with their positions at 12 o'clock? -/
noncomputable def equal_angle_time (x : ℝ) : Prop :=
  -- Defining the conditions for the problem
  let minute_hand_speed := 6 -- degrees per minute
  let hour_hand_speed := 0.5 -- degrees per minute
  let total_degrees := 360 * x -- total degrees of minute hand till time x
  let hour_hand_degrees := 30 * (x / 60) -- total degrees of hour hand till time x

  -- Equation for equal angles formed with respect to 12 o'clock
  30 * (x / 60) = 360 - 360 * (x / 60)

theorem equal_angles_proof :
  ∃ (x : ℝ), equal_angle_time x ∧ x = 55 + 5/13 :=
sorry

end equal_angles_proof_l82_82833


namespace cos_330_is_sqrt3_over_2_l82_82677

noncomputable def cos_330_degree : Real :=
  Real.cos (330 * Real.pi / 180)

theorem cos_330_is_sqrt3_over_2 :
  cos_330_degree = Real.sqrt 3 / 2 :=
sorry

end cos_330_is_sqrt3_over_2_l82_82677


namespace highest_value_of_a_for_divisibility_l82_82175

/-- Given a number in the format of 365a2_, where 'a' is a digit (0 through 9),
prove that the highest value of 'a' that makes the number divisible by 8 is 9. -/
theorem highest_value_of_a_for_divisibility :
  ∃ (a : ℕ), a ≤ 9 ∧ (∃ (d : ℕ), d < 10 ∧ (365 * 100 + a * 10 + 20 + d) % 8 = 0 ∧ a = 9) :=
sorry

end highest_value_of_a_for_divisibility_l82_82175


namespace expand_polynomial_l82_82003

noncomputable def p (x : ℝ) : ℝ := 7 * x ^ 2 + 5
noncomputable def q (x : ℝ) : ℝ := 3 * x ^ 3 + 2 * x + 1

theorem expand_polynomial (x : ℝ) : 
  (p x) * (q x) = 21 * x ^ 5 + 29 * x ^ 3 + 7 * x ^ 2 + 10 * x + 5 := 
by sorry

end expand_polynomial_l82_82003


namespace prime_cubic_solution_l82_82090

theorem prime_cubic_solution :
  ∃ p1 p2 : ℕ, (Nat.Prime p1 ∧ Nat.Prime p2) ∧ p1 ≠ p2 ∧
  (p1^3 + p1^2 - 18*p1 + 26 = 0) ∧ (p2^3 + p2^2 - 18*p2 + 26 = 0) :=
by
  sorry

end prime_cubic_solution_l82_82090


namespace paige_scored_17_points_l82_82078

def paige_points (total_points : ℕ) (num_players : ℕ) (points_per_player_exclusive : ℕ) : ℕ :=
  total_points - ((num_players - 1) * points_per_player_exclusive)

theorem paige_scored_17_points :
  paige_points 41 5 6 = 17 :=
by
  sorry

end paige_scored_17_points_l82_82078


namespace gcd_75_100_l82_82910

theorem gcd_75_100 : Nat.gcd 75 100 = 25 :=
by
  sorry

end gcd_75_100_l82_82910


namespace solve_abs_eq_l82_82085

theorem solve_abs_eq (x : ℝ) : |x - 4| = 3 - x ↔ x = 7 / 2 := by
  sorry

end solve_abs_eq_l82_82085


namespace find_value_of_reciprocal_sin_double_angle_l82_82197

open Real

noncomputable def point := ℝ × ℝ

def term_side_angle_passes_through (α : ℝ) (P : point) :=
  ∃ (r : ℝ), P = (r * cos α, r * sin α)

theorem find_value_of_reciprocal_sin_double_angle (α : ℝ) (P : point) (h : term_side_angle_passes_through α P) :
  P = (-2, 1) → (1 / sin (2 * α)) = -5 / 4 :=
by
  intro hP
  sorry

end find_value_of_reciprocal_sin_double_angle_l82_82197


namespace remainder_of_product_mod_7_l82_82859

theorem remainder_of_product_mod_7 (a b c : ℕ) 
  (ha: a % 7 = 2) 
  (hb: b % 7 = 3) 
  (hc: c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end remainder_of_product_mod_7_l82_82859


namespace students_with_grade_B_and_above_l82_82241

theorem students_with_grade_B_and_above (total_students : ℕ) (percent_below_B : ℕ) 
(h1 : total_students = 60) (h2 : percent_below_B = 40) : 
(total_students * (100 - percent_below_B) / 100) = 36 := by
  sorry

end students_with_grade_B_and_above_l82_82241


namespace geometric_sequence_sum_eq_five_l82_82731

/-- Given that {a_n} is a geometric sequence where each a_n > 0
    and the equation a_2 * a_4 + 2 * a_3 * a_5 + a_4 * a_6 = 25 holds,
    we want to prove that a_3 + a_5 = 5. -/
theorem geometric_sequence_sum_eq_five
  (a : ℕ → ℝ)
  (r : ℝ)
  (h_geom : ∀ n, a n = a 1 * r ^ (n - 1))
  (h_pos : ∀ n, a n > 0)
  (h_eq : a 2 * a 4 + 2 * a 3 * a 5 + a 4 * a 6 = 25) : a 3 + a 5 = 5 :=
sorry

end geometric_sequence_sum_eq_five_l82_82731


namespace tournament_trio_l82_82519

theorem tournament_trio
  (n : ℕ)
  (h_n : n ≥ 3)
  (match_result : Fin n → Fin n → Prop)
  (h1 : ∀ i j : Fin n, i ≠ j → (match_result i j ∨ match_result j i))
  (h2 : ∀ i : Fin n, ∃ j : Fin n, match_result i j)
:
  ∃ (A B C : Fin n), match_result A B ∧ match_result B C ∧ match_result C A :=
by
  sorry

end tournament_trio_l82_82519


namespace impossible_to_load_two_coins_l82_82938

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l82_82938


namespace cos_330_eq_sqrt3_div_2_l82_82703

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82703


namespace solve_equation_l82_82829

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end solve_equation_l82_82829


namespace cos_330_eq_sqrt3_div_2_l82_82686

theorem cos_330_eq_sqrt3_div_2 :
  let deg330 := 330
  let deg30 := 30
  cos (deg330 * (Float.pi / 180)) = Real.sqrt(3) / 2 :=
sorry

end cos_330_eq_sqrt3_div_2_l82_82686


namespace consecutive_numbers_even_count_l82_82600

def percentage_of_evens (n : ℕ) := 13 * n / 25

theorem consecutive_numbers_even_count (n : ℕ) (h1 : 52 * n / 100 = percentage_of_evens n) :
    percentage_of_evens 25 = 13 := 
begin
    sorry
end

end consecutive_numbers_even_count_l82_82600


namespace sam_dimes_l82_82597

theorem sam_dimes (dimes_original dimes_given : ℕ) :
  dimes_original = 9 → dimes_given = 7 → dimes_original + dimes_given = 16 :=
by
  intros h1 h2
  sorry

end sam_dimes_l82_82597


namespace three_digit_integers_with_odd_factors_l82_82394

theorem three_digit_integers_with_odd_factors : ∃ (count : ℕ), count = 22 ∧ 
  ∀ (n : ℕ), 100 ≤ n ∧ n ≤ 999 → (∃ k : ℕ, n = k * k) ↔ (n ≥ 10 * 10) ∧ (n ≤ 31 * 31) :=
by
  use 22
  intro n
  intro hn_range
  split;
  intro h
  -- proof steps omitted for brevity
  sorry

end three_digit_integers_with_odd_factors_l82_82394


namespace correct_operation_l82_82133

theorem correct_operation : 
  (3 - Real.sqrt 2) ^ 2 = 11 - 6 * Real.sqrt 2 :=
sorry

end correct_operation_l82_82133


namespace triangle_area_l82_82642

-- Define the sides of the triangle
def a : ℕ := 9
def b : ℕ := 12
def c : ℕ := 15

-- Define the property of being a right triangle via the Pythagorean theorem
def is_right_triangle (a b c : ℕ) : Prop := a^2 + b^2 = c^2

-- Define the area of a right triangle given base and height
def area_right_triangle (a b : ℕ) : ℕ := (a * b) / 2

-- The main theorem, stating that the area of the triangle with sides 9, 12, 15 is 54
theorem triangle_area : is_right_triangle a b c → area_right_triangle a b = 54 :=
by
  -- Proof is omitted
  sorry

end triangle_area_l82_82642


namespace average_output_l82_82645

theorem average_output (t1 t2: ℝ) (cogs1 cogs2 : ℕ) (h1 : t1 = cogs1 / 36) (h2 : t2 = cogs2 / 60) (h_sum_cogs : cogs1 = 60) (h_sum_more_cogs : cogs2 = 60) (h_sum_time : t1 + t2 = 60 / 36 + 60 / 60) : 
  (cogs1 + cogs2) / (t1 + t2) = 45 := by
  sorry

end average_output_l82_82645


namespace gcd_75_100_l82_82901

def is_prime_power (n : ℕ) (p : ℕ) (k : ℕ) : Prop :=
  n = p ^ k

def prime_factors (n : ℕ) (fs : List (ℕ × ℕ)) : Prop :=
  ∀ p k, (p, k) ∈ fs ↔ is_prime_power n p k

theorem gcd_75_100 : gcd 75 100 = 25 :=
by sorry

end gcd_75_100_l82_82901


namespace sum_of_coordinates_of_A_l82_82777

variables
  (a b : ℝ)
  (A B C : ℝ × ℝ)
  (AB BC AC : ℝ → ℝ)

def line1 := λ x : ℝ, a * x + 4
def line2 := λ x : ℝ, 2 * x + b
def line3 := λ x : ℝ, a / 2 * x + 8

def is_on_line (P : ℝ × ℝ) (L : ℝ → ℝ) := P.2 = L P.1

def conditions := 
  is_on_line A line1 ∧ is_on_line B line1 ∧ is_on_line A line3 ∧ is_on_line B line2 ∧ is_on_line C line2 ∧ is_on_line C line3 ∧
  B.2 = 0 ∧ C.1 = 0

theorem sum_of_coordinates_of_A :
  conditions a b A B C AB BC AC →
  (A.1 + A.2 = 13 ∨ A.1 + A.2 = 20) :=
sorry

end sum_of_coordinates_of_A_l82_82777


namespace shadedQuadrilateralArea_is_13_l82_82019

noncomputable def calculateShadedQuadrilateralArea : ℝ :=
  let s1 := 2
  let s2 := 4
  let s3 := 6
  let s4 := 8
  let bases := s1 + s2
  let height_small := bases * (10 / 20)
  let height_large := 10
  let alt := s4 - s3
  let area := (1 / 2) * (height_small + height_large) * alt
  13

theorem shadedQuadrilateralArea_is_13 :
  calculateShadedQuadrilateralArea = 13 := by
sorry

end shadedQuadrilateralArea_is_13_l82_82019


namespace volume_of_cut_pyramid_l82_82290

theorem volume_of_cut_pyramid
  (base_length : ℝ)
  (slant_length : ℝ)
  (cut_height : ℝ)
  (original_base_area : ℝ)
  (original_height : ℝ)
  (new_base_area : ℝ)
  (volume : ℝ)
  (h_base_length : base_length = 8 * Real.sqrt 2)
  (h_slant_length : slant_length = 10)
  (h_cut_height : cut_height = 3)
  (h_original_base_area : original_base_area = (base_length ^ 2) / 2)
  (h_original_height : original_height = Real.sqrt (slant_length ^ 2 - (base_length / Real.sqrt 2) ^ 2))
  (h_new_base_area : new_base_area = original_base_area / 4)
  (h_volume : volume = (1 / 3) * new_base_area * cut_height) :
  volume = 32 :=
by
  sorry

end volume_of_cut_pyramid_l82_82290


namespace sufficient_necessary_condition_l82_82155

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * a * x^3 + (1 / 2) * a * x^2 - 2 * a * x + 2 * a + 1

theorem sufficient_necessary_condition (a : ℝ) :
  (-6 / 5 < a ∧ a < -3 / 16) ↔
  (∃ x₁ x₂ : ℝ, f a x₁ = 0 ∧ f a x₂ = 0 ∧
   (∃ c₁ c₂ : ℝ, deriv (f a) c₁ = 0 ∧ deriv (f a) c₂ = 0 ∧
   deriv (deriv (f a)) c₁ < 0 ∧ deriv (deriv (f a)) c₂ > 0 ∧
   f a c₁ > 0 ∧ f a c₂ < 0)) := sorry

end sufficient_necessary_condition_l82_82155


namespace solve_for_a_l82_82545

theorem solve_for_a (x a : ℤ) (h1 : x = 3) (h2 : x + 2 * a = -1) : a = -2 :=
by
  sorry

end solve_for_a_l82_82545


namespace temperature_on_tuesday_l82_82255

variable (T W Th F : ℝ)

theorem temperature_on_tuesday :
  (T + W + Th = 156) ∧ (W + Th + 53 = 162) → T = 47 :=
by
  sorry

end temperature_on_tuesday_l82_82255


namespace abs_neg_five_l82_82497

theorem abs_neg_five : abs (-5) = 5 :=
by
  sorry

end abs_neg_five_l82_82497


namespace parking_lot_vehicle_spaces_l82_82886

theorem parking_lot_vehicle_spaces
  (total_spaces : ℕ)
  (spaces_per_caravan : ℕ)
  (num_caravans : ℕ)
  (remaining_spaces : ℕ) :
  total_spaces = 30 →
  spaces_per_caravan = 2 →
  num_caravans = 3 →
  remaining_spaces = total_spaces - (spaces_per_caravan * num_caravans) →
  remaining_spaces = 24 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  exact h4

end parking_lot_vehicle_spaces_l82_82886


namespace impossible_load_two_coins_l82_82930

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l82_82930


namespace marley_fruits_l82_82812

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l82_82812


namespace find_a_plus_b_l82_82606

open Function

theorem find_a_plus_b (a b : ℝ) (f g h : ℝ → ℝ)
  (h_f : ∀ x, f x = a * x - b)
  (h_g : ∀ x, g x = -4 * x - 1)
  (h_h : ∀ x, h x = f (g x))
  (h_h_inv : ∀ y, h⁻¹ y = y + 9) :
  a + b = -9 := 
by
  -- Proof goes here.
  sorry

end find_a_plus_b_l82_82606


namespace power_sum_roots_l82_82802

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end power_sum_roots_l82_82802


namespace abs_neg_eight_l82_82252

theorem abs_neg_eight : abs (-8) = 8 := by
  sorry

end abs_neg_eight_l82_82252


namespace largest_fraction_l82_82627

theorem largest_fraction :
  let f1 := (2 : ℚ) / 3
  let f2 := (3 : ℚ) / 4
  let f3 := (2 : ℚ) / 5
  let f4 := (11 : ℚ) / 15
  f2 > f1 ∧ f2 > f3 ∧ f2 > f4 :=
by
  sorry

end largest_fraction_l82_82627


namespace power_sum_roots_l82_82803

theorem power_sum_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + 3 * x₁ + 1 = 0) (h₂ : x₂^2 + 3 * x₂ + 1 = 0) : 
    x₁^7 + x₂^7 = -843 := 
by 
  sorry

end power_sum_roots_l82_82803


namespace cos_330_cos_30_val_answer_l82_82690

noncomputable def cos_330_eq : Prop :=
  let theta := 30 * (real.pi / 180) in
  real.cos (2 * real.pi - theta) = real.cos theta

theorem cos_330 : real.cos (330 * (real.pi / 180)) = real.cos (30 * (real.pi / 180)) := by
  sorry

theorem cos_30_val : real.cos (30 * (real.pi / 180)) = sqrt 3 / 2 := by
  sorry

theorem answer : real.cos (330 * (real.pi / 180)) = sqrt 3 / 2 := by
  rw [cos_330, cos_30_val]
  sorry

end cos_330_cos_30_val_answer_l82_82690


namespace equation_C_is_symmetric_l82_82272

def symm_y_axis (f : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), f x y ↔ f (-x) y

def equation_A (x y : ℝ) : Prop := x^2 - x + y^2 = 1
def equation_B (x y : ℝ) : Prop := x^2 * y + x * y^2 = 1
def equation_C (x y : ℝ) : Prop := x^2 - y^2 = 1
def equation_D (x y : ℝ) : Prop := x - y = 1

theorem equation_C_is_symmetric : symm_y_axis equation_C :=
by
  sorry

end equation_C_is_symmetric_l82_82272


namespace total_carrots_l82_82825

theorem total_carrots (sandy_carrots: Nat) (sam_carrots: Nat) (h1: sandy_carrots = 6) (h2: sam_carrots = 3) : sandy_carrots + sam_carrots = 9 :=
by
  sorry

end total_carrots_l82_82825


namespace esperanzas_tax_ratio_l82_82076

theorem esperanzas_tax_ratio :
  let rent := 600
  let food_expenses := (3 / 5) * rent
  let mortgage_bill := 3 * food_expenses
  let savings := 2000
  let gross_salary := 4840
  let total_expenses := rent + food_expenses + mortgage_bill + savings
  let taxes := gross_salary - total_expenses
  (taxes / savings) = (2 / 5) := by
  sorry

end esperanzas_tax_ratio_l82_82076


namespace find_n_l82_82015

theorem find_n (n : ℕ) (h : n > 2016) (h_not_divisible : ¬ (1^n + 2^n + 3^n + 4^n) % 10 = 0) : n = 2020 :=
sorry

end find_n_l82_82015


namespace mutually_exclusive_not_complementary_l82_82631

theorem mutually_exclusive_not_complementary :
  let balls := {b | b = "red" ∨ b = "black"}
  let pocket := {b ∈ balls | b = "red"} ∪ {b ∈ balls | b = "black"}
  let draw_two := ({x : Set String | Finset.card x = 2} : Finset (Set String))
  -- Define event C1: Having exactly one black ball
  let event_C1 := {x ∈ draw_two | Finset.card (x ∩ {"black"}) = 1}
  -- Define event C2: Having exactly two red balls
  let event_C2 := {x ∈ draw_two | x = {"red", "red"}}
  -- Prove the events are mutually exclusive but not complementary
  (event_C1 ∩ event_C2 = ∅) ∧ (event_C1 ∪ event_C2 ≠ draw_two) :=
by
  -- Event definitions
  sorry

end mutually_exclusive_not_complementary_l82_82631


namespace bicycle_cost_after_tax_l82_82281

theorem bicycle_cost_after_tax :
  let original_price := 300
  let first_discount := original_price * 0.40
  let price_after_first_discount := original_price - first_discount
  let second_discount := price_after_first_discount * 0.20
  let price_after_second_discount := price_after_first_discount - second_discount
  let tax := price_after_second_discount * 0.05
  price_after_second_discount + tax = 151.20 :=
by
  sorry

end bicycle_cost_after_tax_l82_82281


namespace remainder_when_divided_by_100_l82_82068

-- Define the given m
def m : ℕ := 76^2006 - 76

-- State the theorem
theorem remainder_when_divided_by_100 : m % 100 = 0 :=
by
  sorry

end remainder_when_divided_by_100_l82_82068


namespace no_prime_divisible_by_42_l82_82205

open Nat

theorem no_prime_divisible_by_42 : ∀ p : ℕ, Prime p → 42 ∣ p → p = 0 :=
by
  intros p hp hdiv
  sorry

end no_prime_divisible_by_42_l82_82205


namespace number_of_three_digit_integers_with_odd_factors_l82_82377

theorem number_of_three_digit_integers_with_odd_factors : 
  ∃ n, n = finset.card ((finset.range 32).filter (λ x, 100 ≤ x^2 ∧ x^2 < 1000)) ∧ n = 22 :=
by
  sorry

end number_of_three_digit_integers_with_odd_factors_l82_82377


namespace average_age_of_women_l82_82093

variable {A W : ℝ}

theorem average_age_of_women (A : ℝ) (h : 12 * (A + 3) = 12 * A - 90 + W) : 
  W / 3 = 42 := by
  sorry

end average_age_of_women_l82_82093


namespace total_savings_during_sale_l82_82725

theorem total_savings_during_sale :
  let regular_price_fox := 15
  let regular_price_pony := 20
  let pairs_fox := 3
  let pairs_pony := 2
  let total_discount := 22
  let discount_pony := 18.000000000000014
  let regular_total := (pairs_fox * regular_price_fox) + (pairs_pony * regular_price_pony)
  let discount_fox := total_discount - discount_pony
  (discount_fox / 100 * (pairs_fox * regular_price_fox)) + (discount_pony / 100 * (pairs_pony * regular_price_pony)) = 9 := by
  sorry

end total_savings_during_sale_l82_82725


namespace smallest_n_not_divisible_by_10_l82_82007

theorem smallest_n_not_divisible_by_10 :
  ∃ n > 2016, (1^n + 2^n + 3^n + 4^n) % 10 ≠ 0 ∧ n = 2020 := 
sorry

end smallest_n_not_divisible_by_10_l82_82007


namespace product_remainder_mod_7_l82_82875

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82875


namespace solve_for_x_l82_82036

theorem solve_for_x (x y : ℝ) (h : (x + 1) / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y - 5)) : 
  x = (y^2 + 3 * y - 1) / 7 := 
by 
  sorry

end solve_for_x_l82_82036


namespace remainder_of_product_l82_82855

theorem remainder_of_product (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 4) : 
  (a * b * c) % 7 = 3 := 
by
  sorry

end remainder_of_product_l82_82855


namespace range_of_m_l82_82727

noncomputable def f (m x : ℝ) : ℝ := m * x^2 - 2 * m * x + m + 3
noncomputable def g (x : ℝ) : ℝ := 2^(x - 2)

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x < 0 ∨ g x < 0) ↔ -4 < m ∧ m < 0 :=
by sorry

end range_of_m_l82_82727


namespace marble_count_calculation_l82_82562

theorem marble_count_calculation (y b g : ℕ) (x : ℕ)
  (h1 : y = 2 * x)
  (h2 : b = 3 * x)
  (h3 : g = 4 * x)
  (h4 : g = 32) : y + b + g = 72 :=
by
  sorry

end marble_count_calculation_l82_82562


namespace find_value_of_expression_l82_82739

theorem find_value_of_expression
  (x y z : ℝ)
  (h1 : 3 * x - 4 * y - 2 * z = 0)
  (h2 : x + 2 * y - 7 * z = 0)
  (hz : z ≠ 0) :
  (x^2 - 2 * x * y) / (y^2 + 4 * z^2) = -0.252 := 
sorry

end find_value_of_expression_l82_82739


namespace leftover_cents_l82_82149

noncomputable def total_cents (pennies nickels dimes quarters : Nat) : Nat :=
  (pennies * 1) + (nickels * 5) + (dimes * 10) + (quarters * 25)

noncomputable def total_cost (num_people : Nat) (cost_per_person : Nat) : Nat :=
  num_people * cost_per_person

theorem leftover_cents (h₁ : total_cents 123 85 35 26 = 1548)
                       (h₂ : total_cost 5 300 = 1500) :
  1548 - 1500 = 48 :=
sorry

end leftover_cents_l82_82149


namespace minimum_value_f_is_correct_l82_82178

noncomputable def f (x : ℝ) := 
  Real.sqrt (15 - 12 * Real.cos x) + 
  Real.sqrt (4 - 2 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (7 - 4 * Real.sqrt 3 * Real.sin x) + 
  Real.sqrt (10 - 4 * Real.sqrt 3 * Real.sin x - 6 * Real.cos x)

theorem minimum_value_f_is_correct :
  ∃ x : ℝ, f x = (9 / 2) * Real.sqrt 2 :=
sorry

end minimum_value_f_is_correct_l82_82178


namespace impossible_load_two_coins_l82_82927

-- Define the probabilities of landing heads and tails on two coins
def probability_of_heads_one_coin (p : ℝ) (hq : ℝ) : Prop :=
  (p ≠ 1 - p) ∧ (hq ≠ 1 - hq) ∧ 
  (p * hq = 1 / 4) ∧ (p * (1 - hq) = 1 / 4) ∧ ((1 - p) * hq = 1 / 4) ∧ ((1 - p) * (1 - hq) = 1 / 4)

-- State the theorem for part (a)
theorem impossible_load_two_coins (p q : ℝ) : ¬ (probability_of_heads_one_coin p q) :=
sorry

end impossible_load_two_coins_l82_82927


namespace option_c_same_function_l82_82563

theorem option_c_same_function :
  ∀ (x : ℝ), x ≠ 0 → (1 + (1 / x) = u ↔ u = 1 + (1 / (1 + 1 / x))) :=
by sorry

end option_c_same_function_l82_82563


namespace no_pre_period_decimal_representation_l82_82457

theorem no_pre_period_decimal_representation (m : ℕ) (h : Nat.gcd m 10 = 1) : ¬∃ k : ℕ, ∃ a : ℕ, 0 < a ∧ 10^a < m ∧ (10^a - 1) % m = k ∧ k ≠ 0 :=
sorry

end no_pre_period_decimal_representation_l82_82457


namespace solve_equation_l82_82830

theorem solve_equation : ∀ x : ℝ, 3 * x * (x - 1) = 2 * x - 2 ↔ (x = 1 ∨ x = 2 / 3) := 
by 
  intro x
  sorry

end solve_equation_l82_82830


namespace three_digit_perfect_squares_count_l82_82378

open Nat Real

theorem three_digit_perfect_squares_count : 
  (Finset.card (Finset.filter (λ n, (∃ k : ℕ, n = k^2) ∧ (100 ≤ n ∧ n ≤ 999)) (Finset.range 1000))) = 22 := 
by 
  sorry

end three_digit_perfect_squares_count_l82_82378


namespace three_digit_odds_factors_count_l82_82421

theorem three_digit_odds_factors_count : ∃ n, (∀ k, 100 ≤ k ∧ k ≤ 999 → (k.factors.length % 2 = 1) ↔ (k = 10^2 ∨ k = 11^2 ∨ k = 12^2 ∨ k = 13^2 ∨ k = 14^2 ∨ k = 15^2 ∨ k = 16^2 ∨ k = 17^2 ∨ k = 18^2 ∨ k = 19^2 ∨ k = 20^2 ∨ k = 21^2 ∨ k = 22^2 ∨ k = 23^2 ∨ k = 24^2 ∨ k = 25^2 ∨ k = 26^2 ∨ k = 27^2 ∨ k = 28^2 ∨ k = 29^2 ∨ k = 30^2 ∨ k = 31^2))
                ∧ n = 22 :=
sorry

end three_digit_odds_factors_count_l82_82421


namespace cos_330_eq_sqrt3_div_2_l82_82704

/-- 
Given the angles and cosine values:
1. \(330^\circ = 360^\circ - 30^\circ\)
2. \(\cos(360^\circ - \theta) = \cos \theta\)
3. \(\cos 30^\circ = \frac{\sqrt{3}}{2}\)

We want to prove that:
\(\cos 330^\circ = \frac{\sqrt{3}}{2}\)
-/
theorem cos_330_eq_sqrt3_div_2 
  (cos_30 : ℝ)
  (cos_sub : ∀ θ : ℝ, cos (360 - θ) = cos θ)
  (cos_30_value : cos 30 = (real.sqrt 3) / 2) : 
  cos 330 = (real.sqrt 3) / 2 :=
by
  sorry

end cos_330_eq_sqrt3_div_2_l82_82704


namespace cookie_cost_per_day_l82_82976

theorem cookie_cost_per_day
    (days_in_April : ℕ)
    (cookies_per_day : ℕ)
    (total_spent : ℕ)
    (total_cookies : ℕ := days_in_April * cookies_per_day)
    (cost_per_cookie : ℕ := total_spent / total_cookies) :
  days_in_April = 30 ∧ cookies_per_day = 3 ∧ total_spent = 1620 → cost_per_cookie = 18 :=
by
  sorry

end cookie_cost_per_day_l82_82976


namespace magnitude_of_z_8_l82_82172

def z : Complex := 2 + 3 * Complex.I

theorem magnitude_of_z_8 : Complex.abs (z ^ 8) = 28561 := by
  sorry

end magnitude_of_z_8_l82_82172


namespace matrix_product_l82_82523

-- Define matrix A
def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, -1, 3], ![0, 3, 2], ![1, -3, 4]]

-- Define matrix B
def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, 3, 0], ![2, 0, 4], ![3, 0, 1]]

-- Define the expected result matrix C
def C : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![9, 6, -1], ![12, 0, 14], ![7, 3, -8]]

-- The statement to prove
theorem matrix_product : A * B = C :=
by
  sorry

end matrix_product_l82_82523


namespace multiple_of_cans_of_corn_l82_82300

theorem multiple_of_cans_of_corn (peas corn : ℕ) (h1 : peas = 35) (h2 : corn = 10) (h3 : peas = 10 * x + 15) : x = 2 := 
by
  sorry

end multiple_of_cans_of_corn_l82_82300


namespace quadratic_inequality_l82_82537

theorem quadratic_inequality (k : ℝ) :
  (∀ x : ℝ, k * x^2 - k * x + 4 ≥ 0) ↔ 0 ≤ k ∧ k ≤ 16 :=
by sorry

end quadratic_inequality_l82_82537


namespace range_of_a_l82_82343

-- Define the sets A, B, and C
def set_A (x : ℝ) : Prop := -3 < x ∧ x ≤ 2
def set_B (x : ℝ) : Prop := -1 < x ∧ x < 3
def set_A_int_B (x : ℝ) : Prop := -1 < x ∧ x ≤ 2
def set_C (x : ℝ) (a : ℝ) : Prop := a < x ∧ x < a + 1

-- The target theorem to prove
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, set_C x a → set_A_int_B x) → 
  (-1 ≤ a ∧ a ≤ 1) :=
by 
  sorry

end range_of_a_l82_82343


namespace solve_equation_l82_82828

theorem solve_equation (x : ℝ) : 
  3 * x * (x - 1) = 2 * x - 2 ↔ x = 1 ∨ x = 2 / 3 :=
by
  sorry

end solve_equation_l82_82828


namespace inequalities_always_true_l82_82593

theorem inequalities_always_true (x y a b : ℝ) (hx : 0 < x) (hy : 0 < y) (ha : 0 < a) (hb : 0 < b) 
  (hxa : x ≤ a) (hyb : y ≤ b) : 
  (x + y ≤ a + b) ∧ (x - y ≤ a - b) ∧ (x * y ≤ a * b) ∧ (x / y ≤ a / b) := by
  sorry

end inequalities_always_true_l82_82593


namespace three_digit_integers_with_odd_factors_l82_82406

theorem three_digit_integers_with_odd_factors : 
  (∃ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ odd (nat.factors n).length) -> 
  (∃ k : ℕ, k = 22) :=
sorry

end three_digit_integers_with_odd_factors_l82_82406


namespace isosceles_triangle_side_length_l82_82835

theorem isosceles_triangle_side_length (base : ℝ) (area : ℝ) (congruent_side : ℝ) 
  (h_base : base = 30) (h_area : area = 60) : congruent_side = Real.sqrt 241 :=
by 
  sorry

end isosceles_triangle_side_length_l82_82835


namespace frequency_count_third_group_l82_82506

theorem frequency_count_third_group 
  (x n : ℕ)
  (h1 : n = 420 - x)
  (h2 : x / (n:ℚ) = 0.20) :
  x = 70 :=
by sorry

end frequency_count_third_group_l82_82506


namespace product_remainder_mod_7_l82_82879

theorem product_remainder_mod_7 (a b c : ℕ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
by 
  sorry

end product_remainder_mod_7_l82_82879


namespace investment_schemes_correct_l82_82953

-- Define the parameters of the problem
def num_projects : Nat := 3
def num_districts : Nat := 4

-- Function to count the number of valid investment schemes
def count_investment_schemes (num_projects num_districts : Nat) : Nat :=
  let total_schemes := num_districts ^ num_projects
  let invalid_schemes := num_districts
  total_schemes - invalid_schemes

-- Theorem statement
theorem investment_schemes_correct :
  count_investment_schemes num_projects num_districts = 60 := by
  sorry

end investment_schemes_correct_l82_82953


namespace part_a_part_b_l82_82256

variable (f : ℝ → ℝ)

-- Part (a)
theorem part_a (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) :
  ∀ x : ℝ, f (f x) ≤ 0 :=
sorry

-- Part (b)
theorem part_b (h : ∀ x y : ℝ, f (x + y) ≥ f x + y * f (f x)) (h₀ : f 0 ≥ 0) :
  ∀ x : ℝ, f x = 0 :=
sorry

end part_a_part_b_l82_82256


namespace eleven_billion_in_scientific_notation_l82_82571

namespace ScientificNotation

def Yi : ℝ := 10 ^ 8

theorem eleven_billion_in_scientific_notation : (11 * (10 : ℝ) ^ 9) = (1.1 * (10 : ℝ) ^ 10) :=
by 
  sorry

end ScientificNotation

end eleven_billion_in_scientific_notation_l82_82571


namespace solve_equation_l82_82478

theorem solve_equation (x : ℝ) (hx : x ≠ 1) : (x / (x - 1) - 1 = 1) → (x = 2) :=
by
  sorry

end solve_equation_l82_82478


namespace ratio_of_chris_to_amy_l82_82960

-- Definitions based on the conditions in the problem
def combined_age (Amy_age Jeremy_age Chris_age : ℕ) : Prop :=
  Amy_age + Jeremy_age + Chris_age = 132

def amy_is_one_third_jeremy (Amy_age Jeremy_age : ℕ) : Prop :=
  Amy_age = Jeremy_age / 3

def jeremy_age : ℕ := 66

-- The main theorem we need to prove
theorem ratio_of_chris_to_amy (Amy_age Chris_age : ℕ) (h1 : combined_age Amy_age jeremy_age Chris_age)
  (h2 : amy_is_one_third_jeremy Amy_age jeremy_age) : Chris_age / Amy_age = 2 :=
sorry

end ratio_of_chris_to_amy_l82_82960


namespace three_digit_integers_with_odd_factors_l82_82363

theorem three_digit_integers_with_odd_factors:
  let lower_bound := 100
  let upper_bound := 999 in
  let counts := (seq 10 22).filter (fun x => x * x >= lower_bound ∧ x * x <= upper_bound) in
  counts.length = 22 := by
  sorry

end three_digit_integers_with_odd_factors_l82_82363


namespace inequality_I_inequality_II_inequality_III_l82_82735

variable {a b c x y z : ℝ}

-- Assume the conditions
def conditions (a b c x y z : ℝ) : Prop :=
  x^2 < a ∧ y^2 < b ∧ z^2 < c

-- Prove the first inequality
theorem inequality_I (h : conditions a b c x y z) : x^2 * y^2 + y^2 * z^2 + z^2 * x^2 < a * b + b * c + c * a :=
sorry

-- Prove the second inequality
theorem inequality_II (h : conditions a b c x y z) : x^4 + y^4 + z^4 < a^2 + b^2 + c^2 :=
sorry

-- Prove the third inequality
theorem inequality_III (h : conditions a b c x y z) : x^2 * y^2 * z^2 < a * b * c :=
sorry

end inequality_I_inequality_II_inequality_III_l82_82735


namespace remainder_product_l82_82868

theorem remainder_product (a b c : ℤ) 
  (ha : a % 7 = 2) 
  (hb : b % 7 = 3) 
  (hc : c % 7 = 4) : 
  (a * b * c) % 7 = 3 :=
sorry

end remainder_product_l82_82868


namespace three_digit_integers_with_odd_number_of_factors_l82_82391

theorem three_digit_integers_with_odd_number_of_factors : 
  { n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ (∃ m : ℕ, n = m * m) }.toFinset.card = 22 := by
  sorry

end three_digit_integers_with_odd_number_of_factors_l82_82391


namespace bianca_points_l82_82646

theorem bianca_points : 
  let a := 5; let b := 8; let c := 10;
  let A1 := 10; let P1 := 5; let G1 := 5;
  let A2 := 3; let P2 := 2; let G2 := 1;
  (A1 * a - A2 * a) + (P1 * b - P2 * b) + (G1 * c - G2 * c) = 99 := 
by
  sorry

end bianca_points_l82_82646


namespace Marley_fruit_count_l82_82809

theorem Marley_fruit_count :
  ∀ (louis_oranges louis_apples samantha_oranges samantha_apples : ℕ)
  (marley_oranges marley_apples : ℕ),
  louis_oranges = 5 →
  louis_apples = 3 →
  samantha_oranges = 8 →
  samantha_apples = 7 →
  marley_oranges = 2 * louis_oranges →
  marley_apples = 3 * samantha_apples →
  marley_oranges + marley_apples = 31 :=
by
  intros
  sorry

end Marley_fruit_count_l82_82809


namespace derivative_at_pi_div_3_l82_82982

noncomputable def f (x : ℝ) : ℝ := (1 + Real.sqrt 2) * Real.sin x - Real.cos x

theorem derivative_at_pi_div_3 :
  deriv f (π / 3) = (1 / 2) * (1 + Real.sqrt 2 + Real.sqrt 3) :=
by
  sorry

end derivative_at_pi_div_3_l82_82982


namespace enrique_commission_l82_82711

-- Define parameters for the problem
def suit_price : ℝ := 700
def suits_sold : ℝ := 2

def shirt_price : ℝ := 50
def shirts_sold : ℝ := 6

def loafer_price : ℝ := 150
def loafers_sold : ℝ := 2

def commission_rate : ℝ := 0.15

-- Calculate total sales for each category
def total_suit_sales : ℝ := suit_price * suits_sold
def total_shirt_sales : ℝ := shirt_price * shirts_sold
def total_loafer_sales : ℝ := loafer_price * loafers_sold

-- Calculate total sales
def total_sales : ℝ := total_suit_sales + total_shirt_sales + total_loafer_sales

-- Calculate commission
def commission : ℝ := commission_rate * total_sales

-- Proof statement that Enrique's commission is $300
theorem enrique_commission : commission = 300 := sorry

end enrique_commission_l82_82711


namespace letter_at_position_in_pattern_l82_82306

/-- Determine the 150th letter in the repeating pattern XYZ is "Z"  -/
theorem letter_at_position_in_pattern :
  ∀ (pattern : List Char) (position : ℕ), pattern = ['X', 'Y', 'Z'] → position = 150 → pattern.get! ((position - 1) % pattern.length) = 'Z' :=
by
  intros pattern position
  intro hPattern hPosition
  rw [hPattern, hPosition]
  -- pattern = ['X', 'Y', 'Z'] and position = 150
  sorry

end letter_at_position_in_pattern_l82_82306


namespace johns_donation_l82_82035

theorem johns_donation
    (A T D : ℝ)
    (n : ℕ)
    (hA1 : A * 1.75 = 100)
    (hA2 : A = 100 / 1.75)
    (hT : T = 10 * A)
    (hD : D = 11 * 100 - T)
    (hn : n = 10) :
    D = 3700 / 7 := 
sorry

end johns_donation_l82_82035


namespace satisfies_equation_l82_82319

theorem satisfies_equation : 
  { (x, y) : ℤ × ℤ | x^2 + x = y^4 + y^3 + y^2 + y } = 
  { (0, -1), (-1, -1), (0, 0), (-1, 0), (5, 2), (-6, 2) } :=
by
  sorry

end satisfies_equation_l82_82319


namespace arithmetic_sequence_middle_term_l82_82125

theorem arithmetic_sequence_middle_term :
  let a1 := 3^2
  let a3 := 3^4
  let y := (a1 + a3) / 2
  y = 45 :=
by
  let a1 := (3:ℕ)^2
  let a3 := (3:ℕ)^4
  let y := (a1 + a3) / 2
  have : a1 = 9 := by norm_num
  have : a3 = 81 := by norm_num
  have : y = 45 := by norm_num
  exact this

end arithmetic_sequence_middle_term_l82_82125


namespace prove_expression_value_l82_82207

theorem prove_expression_value (a b c d : ℝ) (h1 : a + b = 0) (h2 : c = -1) (h3 : d = 1 ∨ d = -1) :
  2 * a + 2 * b - c * d = 1 ∨ 2 * a + 2 * b - c * d = -1 := 
by sorry

end prove_expression_value_l82_82207


namespace mary_walking_speed_l82_82591

-- Definitions based on the conditions:
def distance_sharon (t : ℝ) : ℝ := 6 * t
def distance_mary (x t : ℝ) : ℝ := x * t
def total_distance (x t : ℝ) : ℝ := distance_sharon t + distance_mary x t

-- Lean statement to prove that the speed x is 4 given the conditions
theorem mary_walking_speed (x : ℝ) (t : ℝ) (h1 : t = 0.3) (h2 : total_distance x t = 3) : x = 4 :=
by
  sorry

end mary_walking_speed_l82_82591


namespace hilt_miles_traveled_l82_82075

theorem hilt_miles_traveled (initial_miles lunch_additional_miles : Real) (h_initial : initial_miles = 212.3) (h_lunch : lunch_additional_miles = 372.0) :
  initial_miles + lunch_additional_miles = 584.3 :=
by
  sorry

end hilt_miles_traveled_l82_82075


namespace value_of_k_l82_82789

theorem value_of_k (k : ℝ) (x : ℝ) (y : ℝ) 
  (h1 : y = (k - 1) * x + k^2 - 1)
  (h2 : ∃ m : ℝ, y = m * x)
  (h3 : k ≠ 1) :
  k = -1 :=
by
  sorry

end value_of_k_l82_82789


namespace cream_ratio_l82_82797

-- Define the initial conditions for Joe and JoAnn
def initial_coffee : ℕ := 15
def initial_cup_size : ℕ := 20
def cream_added : ℕ := 3
def coffee_drank_by_joe : ℕ := 3
def mixture_stirred_by_joann : ℕ := 3

-- Define the resulting amounts of cream in Joe and JoAnn's coffee
def cream_in_joe : ℕ := cream_added
def cream_in_joann : ℝ := cream_added - (cream_added * (mixture_stirred_by_joann / (initial_coffee + cream_added)))

-- Prove the ratio of the amount of cream in Joe's coffee to that in JoAnn's coffee
theorem cream_ratio :
  (cream_in_joe : ℝ) / cream_in_joann = 6 / 5 :=
by
  -- The code is just a statement; the proof detail is omitted with sorry, and variables are straightforward math.
  sorry

end cream_ratio_l82_82797


namespace remainder_sum_modulo_eleven_l82_82323

theorem remainder_sum_modulo_eleven :
  (88132 + 88133 + 88134 + 88135 + 88136 + 88137 + 88138 + 88139 + 88140 + 88141) % 11 = 1 :=
by
  sorry

end remainder_sum_modulo_eleven_l82_82323


namespace sum_of_nine_numbers_l82_82211

theorem sum_of_nine_numbers (avg : ℝ) (n : ℕ) (h_avg : avg = 5.3) (h_n : n = 9) : 
  (9 * avg = 47.7) :=
by 
  rw [h_avg, h_n]
  norm_num

end sum_of_nine_numbers_l82_82211


namespace pentagon_area_l82_82225

theorem pentagon_area 
  (PQ QR RS ST TP : ℝ) 
  (angle_TPQ angle_PQR : ℝ) 
  (hPQ : PQ = 8) 
  (hQR : QR = 2) 
  (hRS : RS = 13) 
  (hST : ST = 13) 
  (hTP : TP = 8) 
  (hangle_TPQ : angle_TPQ = 90) 
  (hangle_PQR : angle_PQR = 90) : 
  PQ * QR + (1 / 2) * (TP - QR) * PQ + (1 / 2) * 10 * 12 = 100 := 
by
  sorry

end pentagon_area_l82_82225


namespace circle_equation_l82_82341

theorem circle_equation (x y : ℝ) (h1 : y^2 = 4 * x) (h2 : (x - 1)^2 + y^2 = 4) :
    x^2 + y^2 - 2 * x - 3 = 0 :=
sorry

end circle_equation_l82_82341


namespace parameter_values_for_roots_l82_82977

theorem parameter_values_for_roots (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 = 5 * x2 ∧ a * x1^2 - (2 * a + 5) * x1 + 10 = 0 ∧ a * x2^2 - (2 * a + 5) * x2 + 10 = 0)
  ↔ (a = 5 / 3 ∨ a = 5) := 
sorry

end parameter_values_for_roots_l82_82977


namespace polygon_diagonals_l82_82208

theorem polygon_diagonals (n : ℕ) (h : n - 3 = 4) : n = 7 :=
sorry

end polygon_diagonals_l82_82208


namespace spherical_coordinates_convert_l82_82221

theorem spherical_coordinates_convert (ρ θ φ ρ' θ' φ' : ℝ) 
  (h₀ : ρ > 0) 
  (h₁ : 0 ≤ θ ∧ θ < 2 * Real.pi) 
  (h₂ : 0 ≤ φ ∧ φ ≤ Real.pi) 
  (h_initial : (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5)) 
  (h_final : (ρ', θ', φ') = (4, (11 * Real.pi) / 8,  Real.pi / 5)) : 
  (ρ, θ, φ) = (4, (3 * Real.pi) / 8, (9 * Real.pi) / 5) → 
  (ρ, θ, φ) = (ρ', θ', φ') := 
by
  sorry

end spherical_coordinates_convert_l82_82221


namespace gcf_75_100_l82_82896

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l82_82896


namespace line_intersects_y_axis_at_origin_l82_82285

theorem line_intersects_y_axis_at_origin 
  (x₁ y₁ x₂ y₂ : ℤ) 
  (h₁ : (x₁, y₁) = (3, 9)) 
  (h₂ : (x₂, y₂) = (-7, -21)) 
  : 
  ∃ y : ℤ, (0, y) = (0, 0) := by
  sorry

end line_intersects_y_axis_at_origin_l82_82285


namespace dot_product_result_l82_82738

open scoped BigOperators

-- Define the vectors a and b
def a : ℝ × ℝ := (2, -3)
def b : ℝ × ℝ := (-1, 2)

-- Define the addition of two vectors
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2)

-- Define the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- The theorem to be proved
theorem dot_product_result : dot_product (vector_add a b) a = 5 := by
  sorry

end dot_product_result_l82_82738


namespace square_side_is_8_l82_82607

-- Definitions based on problem conditions
def rectangle_width : ℝ := 4
def rectangle_length : ℝ := 16
def rectangle_area : ℝ := rectangle_width * rectangle_length

def square_side_length (s : ℝ) : Prop := s^2 = rectangle_area

-- The theorem we need to prove
theorem square_side_is_8 (s : ℝ) : square_side_length s → s = 8 := by
  -- Proof to be filled in
  sorry

end square_side_is_8_l82_82607


namespace teams_in_each_group_l82_82047

theorem teams_in_each_group (n : ℕ) :
  (2 * (n * (n - 1) / 2) + 3 * n = 56) → n = 7 :=
by
  sorry

end teams_in_each_group_l82_82047


namespace parallel_lines_m_value_l82_82838

noncomputable def m_value_parallel (m : ℝ) : Prop :=
  (m-1) / 2 = 1 / -3

theorem parallel_lines_m_value :
  ∀ (m : ℝ), (m_value_parallel m) → m = 1 / 3 :=
by
  intro m
  intro h
  sorry

end parallel_lines_m_value_l82_82838


namespace find_c_value_l82_82214

theorem find_c_value (A B C : ℝ) (S1_area S2_area : ℝ) (b : ℝ) :
  S1_area = 40 * b + 1 →
  S2_area = 40 * b →
  ∃ c, AC + CB = c ∧ c = 462 :=
by
  intro hS1 hS2
  sorry

end find_c_value_l82_82214


namespace impossible_to_load_two_coins_l82_82937

theorem impossible_to_load_two_coins 
  (p q : ℝ) 
  (h0 : p ≠ 1 - p)
  (h1 : q ≠ 1 - q) 
  (hpq : p * q = 1/4)
  (hptq : p * (1 - q) = 1/4)
  (h1pq : (1 - p) * q = 1/4)
  (h1ptq : (1 - p) * (1 - q) = 1/4) : 
  false :=
sorry

end impossible_to_load_two_coins_l82_82937


namespace consecutive_even_numbers_l82_82599

theorem consecutive_even_numbers (n m : ℕ) (h : 52 * (2 * n - 1) = 100 * n) : n = 13 :=
by
  sorry

end consecutive_even_numbers_l82_82599


namespace sum_of_coefficients_eq_minus_36_l82_82613

noncomputable def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

theorem sum_of_coefficients_eq_minus_36 
  (a b c : ℝ)
  (h_min : ∀ x, quadratic a b c x ≥ -36)
  (h_points : quadratic a b c (-3) = 0 ∧ quadratic a b c 5 = 0)
  : a + b + c = -36 :=
sorry

end sum_of_coefficients_eq_minus_36_l82_82613


namespace number_of_three_digit_squares_l82_82383

-- Define the condition: three-digit number being a square with an odd number of factors
def is_three_digit_square (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000 ∧ ∃ k : ℕ, n = k * k

-- State the main theorem
theorem number_of_three_digit_squares : ∃ n : ℕ, n = 22 ∧ 
  (∀ m : ℕ, is_three_digit_square m → ∃! k : ℕ, m = k * k) :=
exists.intro 22 
  (and.intro rfl 
    (λ m h, sorry))

end number_of_three_digit_squares_l82_82383


namespace closest_point_to_line_l82_82181

theorem closest_point_to_line {x y : ℝ} (h : y = 2 * x - 4) :
  ∃ (closest_x closest_y : ℝ),
    closest_x = 9 / 5 ∧ closest_y = -2 / 5 ∧ closest_y = 2 * closest_x - 4 ∧
    ∀ (x' y' : ℝ), y' = 2 * x' - 4 → (closest_x - 3)^2 + (closest_y + 1)^2 ≤ (x' - 3)^2 + (y' + 1)^2 :=
by
  sorry

end closest_point_to_line_l82_82181


namespace coin_loading_impossible_l82_82935

theorem coin_loading_impossible (p q : ℝ) (h₁ : p ≠ 1 - p) (h₂ : q ≠ 1 - q)
  (h₃ : p * q = 1 / 4) (h₄ : p * (1 - q) = 1 / 4) (h₅ : (1 - p) * q = 1 / 4) (h₆ : (1 - p) * (1 - q) = 1 / 4) :
  false :=
by { sorry }

end coin_loading_impossible_l82_82935


namespace notebook_cost_l82_82289

theorem notebook_cost (n p : ℝ) (h1 : n + p = 2.40) (h2 : n = 2 + p) : n = 2.20 := by
  sorry

end notebook_cost_l82_82289


namespace area_of_bounded_curve_is_64_pi_l82_82964

noncomputable def bounded_curve_area : Real :=
  let curve_eq (x y : ℝ) : Prop := (2 * x + 3 * y + 5) ^ 2 + (x + 2 * y - 3) ^ 2 = 64
  let S : Real := 64 * Real.pi
  S

theorem area_of_bounded_curve_is_64_pi : bounded_curve_area = 64 * Real.pi := 
by
  sorry

end area_of_bounded_curve_is_64_pi_l82_82964


namespace three_digit_cubes_divisible_by_16_l82_82556

theorem three_digit_cubes_divisible_by_16 (n : ℤ) (x : ℤ) 
  (h_cube : x = n^3)
  (h_div : 16 ∣ x) 
  (h_3digit : 100 ≤ x ∧ x ≤ 999) : 
  x = 512 := 
by {
  sorry
}

end three_digit_cubes_divisible_by_16_l82_82556


namespace compare_expressions_l82_82530

-- Considering the conditions
noncomputable def log2 (x : ℝ) : ℝ := Real.log x / Real.log 2
noncomputable def sqrt5 := Real.sqrt 5
noncomputable def expr1 := (2 + log2 6)
noncomputable def expr2 := (2 * sqrt5)

-- The theorem statement
theorem compare_expressions : 
  expr1 > expr2 := 
  sorry

end compare_expressions_l82_82530


namespace river_flow_speed_eq_l82_82152

-- Definitions of the given conditions
def ship_speed : ℝ := 30
def distance_downstream : ℝ := 144
def distance_upstream : ℝ := 96

-- Lean 4 statement to prove the condition
theorem river_flow_speed_eq (v : ℝ) :
  (distance_downstream / (ship_speed + v) = distance_upstream / (ship_speed - v)) :=
by { sorry }

end river_flow_speed_eq_l82_82152


namespace gcf_75_100_l82_82899

theorem gcf_75_100 : Nat.gcd 75 100 = 25 := by
  -- Prime factorization for reference:
  -- 75 = 3 * 5^2
  -- 100 = 2^2 * 5^2
  sorry

end gcf_75_100_l82_82899


namespace mask_price_reduction_l82_82240

theorem mask_price_reduction 
  (initial_sales : ℕ)
  (initial_profit : ℝ)
  (additional_sales_factor : ℝ)
  (desired_profit : ℝ)
  (x : ℝ)
  (h_initial_sales : initial_sales = 500)
  (h_initial_profit : initial_profit = 0.6)
  (h_additional_sales_factor : additional_sales_factor = 100 / 0.1)
  (h_desired_profit : desired_profit = 240) :
  (initial_profit - x) * (initial_sales + additional_sales_factor * x) = desired_profit → x = 0.3 :=
sorry

end mask_price_reduction_l82_82240


namespace range_of_g_l82_82183

noncomputable def g (x : ℝ) : ℝ :=
  (cos x)^3 + 7 * (cos x)^2 + 2 * (cos x) + 3 * (1 - (cos x)^2) - 14

theorem range_of_g :
  ∀ x : ℝ, cos x ≠ 2 → 0.5 ≤ g x / (cos x - 2) ∧ g x / (cos x - 2) < 12.5 :=
by
  sorry

end range_of_g_l82_82183


namespace three_digit_integers_with_odd_number_of_factors_count_l82_82399

theorem three_digit_integers_with_odd_number_of_factors_count :
  ∃ n : ℕ, n = 22 ∧ ∀ x : ℕ, (100 ≤ x ∧ x ≤ 999 → (nat.factors_count_is_odd x ↔ (∃ k : ℕ, x = k^2 ∧ 10 ≤ k ∧ k ≤ 31))) := 
sorry

end three_digit_integers_with_odd_number_of_factors_count_l82_82399


namespace marley_fruits_l82_82810

theorem marley_fruits 
    (louis_oranges : ℕ := 5) (louis_apples : ℕ := 3)
    (samantha_oranges : ℕ := 8) (samantha_apples : ℕ := 7)
    (marley_oranges : ℕ := 2 * louis_oranges)
    (marley_apples : ℕ := 3 * samantha_apples) :
    marley_oranges + marley_apples = 31 := by
  sorry

end marley_fruits_l82_82810


namespace inequality_am_gm_l82_82454

theorem inequality_am_gm (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
    a^3 / (b * c) + b^3 / (c * a) + c^3 / (a * b) ≥ a + b + c :=
by {
    sorry
}

end inequality_am_gm_l82_82454


namespace jake_fewer_peaches_undetermined_l82_82447

theorem jake_fewer_peaches_undetermined 
    (steven_peaches : ℕ) 
    (steven_apples : ℕ) 
    (jake_fewer_peaches : steven_peaches > jake_peaches) 
    (jake_more_apples : jake_apples = steven_apples + 3) 
    (steven_peaches_val : steven_peaches = 9) 
    (steven_apples_val : steven_apples = 8) : 
    ∃ n : ℕ, jake_peaches = n ∧ ¬(∃ m : ℕ, steven_peaches - jake_peaches = m) := 
sorry

end jake_fewer_peaches_undetermined_l82_82447
