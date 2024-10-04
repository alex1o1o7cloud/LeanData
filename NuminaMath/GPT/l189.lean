import Mathlib

namespace sum_of_distances_from_points_to_BC_l189_189777

variables {A B C Q R P1 P2 : Type}
variables [HasDist A B C P1 Q R]

axiom triangle_ABC : triangle A B C
axiom sides : dist A B = 13 ∧ dist B C = 14 ∧ dist A C = 15
axiom right_angles : ∠B C Q = 90 ∧ ∠C B R = 90
axiom similarity1 : similar (triangle P1 Q R) (triangle A B C)
axiom similarity2 : similar (triangle P2 Q R) (triangle A B C)

theorem sum_of_distances_from_points_to_BC : 
  dist_from_bc P1 + dist_from_bc P2 = 48 :=
sorry

end sum_of_distances_from_points_to_BC_l189_189777


namespace arjun_becca_3_different_colors_l189_189561

open Classical

noncomputable def arjun_becca_probability : ℚ := 
  let arjun_initial := [2, 1, 1, 1] -- 2 red, 1 green, 1 yellow, 1 violet
  let becca_initial := [2, 1] -- 2 black, 1 orange
  
  -- possible cases represented as a list of probabilities
  let cases := [
    (2/5) * (1/4) * (3/5),    -- Case 1: Arjun does move a red ball to Becca, and then processes accordingly
    (3/5) * (1/2) * (1/5),    -- Case 2a: Arjun moves a non-red ball, followed by Becca moving a black ball, concluding in the defined manner
    (3/5) * (1/2) * (3/5)     -- Case 2b: Arjun moves a non-red ball, followed by Becca moving a non-black ball, again concluding appropriately
  ]
  
  -- sum of cases representing the total probability
  let total_probability := List.sum cases
  
  total_probability

theorem arjun_becca_3_different_colors : arjun_becca_probability = 3/10 := 
  by
    simp [arjun_becca_probability]
    sorry

end arjun_becca_3_different_colors_l189_189561


namespace midpoints_collinear_or_equilateral_l189_189081

variables {α : Type*} [metric_space α] [has_dist α] [add_comm_group α] [module ℝ α] 

structure triangle (α : Type*) :=
(A B C : α)
(is_equilateral : dist A B = dist B C ∧ dist B C = dist C A ∧ dist C A = dist A B)

def midpoint (a b : α) : α := (a + b) / 2

theorem midpoints_collinear_or_equilateral
  (T1 T2 : triangle α)
  (h_congruent : dist T1.A T1.B = dist T2.A T2.B ∧
                 dist T1.B T1.C = dist T2.B T2.C ∧
                 dist T1.C T1.A = dist T2.C T2.A)
  (h1 : T1.is_equilateral)
  (h2 : T2.is_equilateral)
  (A1 := midpoint T1.A T2.A)
  (B1 := midpoint T1.B T2.B)
  (C1 := midpoint T1.C T2.C) :
  collinear {A1, B1, C1} ∨ 
  equilateral {A1, B1, C1} :=
sorry

end midpoints_collinear_or_equilateral_l189_189081


namespace norm_two_u_l189_189619

variable {E : Type*} [NormedAddCommGroup E]

theorem norm_two_u (u : E) (h : ∥u∥ = 5) : ∥2 • u∥ = 10 := sorry

end norm_two_u_l189_189619


namespace min_points_in_set_M_l189_189074
-- Import the necessary library

-- Define the problem conditions and the result to prove
theorem min_points_in_set_M :
  ∃ (M : Finset ℝ) (C₁ C₂ C₃ C₄ C₅ C₆ C₇ : Finset ℝ),
  C₇.card = 7 ∧
  C₆.card = 6 ∧
  C₅.card = 5 ∧
  C₄.card = 4 ∧
  C₃.card = 3 ∧
  C₂.card = 2 ∧
  C₁.card = 1 ∧
  C₇ ⊆ M ∧
  C₆ ⊆ M ∧
  C₅ ⊆ M ∧
  C₄ ⊆ M ∧
  C₃ ⊆ M ∧
  C₂ ⊆ M ∧
  C₁ ⊆ M ∧
  M.card = 12 :=
sorry

end min_points_in_set_M_l189_189074


namespace partI_solution_set_partII_range_of_m_l189_189039

def f (x m : ℝ) : ℝ := |x - m| + |x + 6|

theorem partI_solution_set (x : ℝ) :
  ∀ (x : ℝ), f x 5 ≤ 12 ↔ (-13 / 2 ≤ x ∧ x ≤ 11 / 2) :=
by
  sorry

theorem partII_range_of_m (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 7) ↔ (m ≤ -13 ∨ m ≥ 1) :=
by
  sorry

end partI_solution_set_partII_range_of_m_l189_189039


namespace eiffel_tower_scale_l189_189548

theorem eiffel_tower_scale (height_model : ℝ) (height_actual : ℝ) (h_model : height_model = 30) (h_actual : height_actual = 984) : 
  height_actual / height_model = 32.8 := by
  sorry

end eiffel_tower_scale_l189_189548


namespace Brad_speed_calculation_l189_189789

noncomputable def Brad_running_speed (Distance_between_homes : ℝ) (Maxwell_speed : ℝ) (Maxwell_traveled_distance : ℝ) : ℝ :=
  let Distance_Brad = Distance_between_homes - Maxwell_traveled_distance
  let Time_Maxwell = Maxwell_traveled_distance / Maxwell_speed
  let Speed_Brad = Distance_Brad / Time_Maxwell
  Speed_Brad

theorem Brad_speed_calculation :
  Brad_running_speed 65 2 26 = 3 :=
by
  apply sorry

end Brad_speed_calculation_l189_189789


namespace count_012_in_base3_repr_l189_189147

def base3_rep (n : ℕ) : List ℕ := 
  if n = 0 then []
  else base3_rep (n / 3) ++ [n % 3]

def joined_base3_repr : List ℕ := List.join (List.map base3_rep (List.range' 1 729))

def count_substring_012 (l : List ℕ) : ℕ :=
  l.countp (λ i, l.drop i.take 3 = [0, 1, 2])

theorem count_012_in_base3_repr : 
  count_substring_012 joined_base3_repr = 148 := 
by sorry

end count_012_in_base3_repr_l189_189147


namespace consecutive_integer_product_sum_l189_189929

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189929


namespace zach_points_l189_189130

def football_game_points (b t z : ℝ) : Prop :=
  b = 21.0 ∧ t = 63 ∧ z = (t - b)

theorem zach_points (b t z : ℝ) (h : football_game_points b t z) : z = 42 :=
by
  cases h with b_eq rest
  cases rest with t_eq z_eq
  rw [b_eq, t_eq] at z_eq
  simp at z_eq
  exact z_eq

end zach_points_l189_189130


namespace distance_between_foci_of_ellipse_l189_189261

theorem distance_between_foci_of_ellipse :
  let E := 25 * x^2 - 100 * x + 4 * y^2 + 16 * y + 16
  in ( ∃ (a b : ℝ), ∀ (h : E = 0), 
    ∃ c : ℝ, c = real.sqrt (a^2 - b^2) ∧ 2 * c = 2 * real.sqrt 21 ) :=
sorry

end distance_between_foci_of_ellipse_l189_189261


namespace odd_prime_divides_seq_implies_power_of_two_divides_l189_189061

theorem odd_prime_divides_seq_implies_power_of_two_divides (a : ℕ → ℤ) (p n : ℕ)
  (h0 : a 0 = 2)
  (hk : ∀ k, a (k + 1) = 2 * (a k) ^ 2 - 1)
  (h_odd_prime : Nat.Prime p)
  (h_odd : p % 2 = 1)
  (h_divides : ↑p ∣ a n) :
  2^(n + 3) ∣ p^2 - 1 :=
sorry

end odd_prime_divides_seq_implies_power_of_two_divides_l189_189061


namespace number_of_parallel_lines_l189_189702

theorem number_of_parallel_lines (n : ℕ) (h : (n * (n - 1) / 2) * (8 * 7 / 2) = 784) : n = 8 :=
sorry

end number_of_parallel_lines_l189_189702


namespace hearts_total_shaded_area_l189_189444

theorem hearts_total_shaded_area (A B C D : ℕ) (hA : A = 1) (hB : B = 4) (hC : C = 9) (hD : D = 16) :
  (D - C) + (B - A) = 10 := 
by 
  sorry

end hearts_total_shaded_area_l189_189444


namespace cauchy_schwarz_inequality_max_sqrt_expression_l189_189145

theorem cauchy_schwarz_inequality (a b c d : ℝ) : 
  (a^2 + b^2) * (c^2 + d^2) ≥ (a * c + b * d)^2 := by
  sorry

theorem max_sqrt_expression (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 1) : 
  sqrt (3 * a + 1) + sqrt (3 * b + 1) ≤ sqrt 10 := by
    apply cauchy_schwarz_inequality
    sorry

end cauchy_schwarz_inequality_max_sqrt_expression_l189_189145


namespace triangle_side_length_l189_189278

theorem triangle_side_length (a : ℝ) (h1 : 4 < a) (h2 : a < 8) : a = 6 :=
sorry

end triangle_side_length_l189_189278


namespace min_wrapping_paper_side_l189_189154

theorem min_wrapping_paper_side (l w h : ℝ) : 
  ∃ s, s = sqrt ((l^2) / 2 + (w^2) / 2 + 2 * h^2) := by
  use sqrt ((l^2) / 2 + (w^2) / 2 + 2 * h^2)
  sorry

end min_wrapping_paper_side_l189_189154


namespace smallest_number_with_remainder_l189_189244

theorem smallest_number_with_remainder (a b m r : ℕ) (h1 : m = 64) (ha : a % m = r) (hb : b % m = r) (cond1 : a = 794) (cond2 : b = 858) (hr : r = 22) : ∃ n, (n > b ∧ n % m = r ∧ n = 922) := 
by
  use 922
  split
  -- Prove that 922 is greater than 858.
  exact sorry
  split
  -- Prove that 922 % 64 = 22. 
  exact sorry
  -- Prove equality.
  exact sorry

end smallest_number_with_remainder_l189_189244


namespace coefficient_x2_term_l189_189239

def poly1 : Polynomial ℝ := 2 * X^3 + 5 * X^2 - 3 * X
def poly2 : Polynomial ℝ := 3 * X^2 - 4 * X - 5

theorem coefficient_x2_term :
  (poly1 * poly2).coeff 2 = -37 :=
by
  sorry

end coefficient_x2_term_l189_189239


namespace find_f_3_l189_189651

noncomputable def f (a x : ℤ) : ℤ := a * x - 1

theorem find_f_3 (a : ℤ) (h : f a 2 = 3) : f 2 3 = 5 :=
by
  have ha : a = 2 := by sorry
  rw [ha]
  exact h

end find_f_3_l189_189651


namespace units_digit_3_pow_2004_l189_189979

-- Definition of the observed pattern of the units digits of powers of 3.
def pattern_units_digits : List ℕ := [3, 9, 7, 1]

-- Theorem stating that the units digit of 3^2004 is 1.
theorem units_digit_3_pow_2004 : (3 ^ 2004) % 10 = 1 :=
by
  sorry

end units_digit_3_pow_2004_l189_189979


namespace train_distance_difference_l189_189486

theorem train_distance_difference:
  ∀ (D1 D2 : ℕ) (t : ℕ), 
    (D1 = 20 * t) →            -- Slower train's distance
    (D2 = 25 * t) →           -- Faster train's distance
    (D1 + D2 = 450) →         -- Total distance between stations
    (D2 - D1 = 50) := 
by
  intros D1 D2 t h1 h2 h3
  sorry

end train_distance_difference_l189_189486


namespace remainder_when_divided_by_3_l189_189141

theorem remainder_when_divided_by_3 (N : ℕ) (digits : ℕ → ℕ) :
  (∀ d, digits d = 0 ∨ d = 3 ∨ d = 5 ∨ d = 7) →
  (∀ d, 0 ≤ digits d) →
  digits 3 + digits 5 + digits 7 = 1580 →
  digits 7 = digits 3 - 20 →
  (N = ∑ d in {3, 5, 7}, d * digits d) →
  N % 3 = 0 :=
by
  sorry

end remainder_when_divided_by_3_l189_189141


namespace elevator_stops_most_on_10th_floor_l189_189713

def people_on_floor : ℕ → ℕ
| 1 := 1
| 2 := 2
| 3 := 3
| 4 := 4
| 5 := 5
| 6 := 6
| 7 := 7
| 8 := 8
| 9 := 9
| 10 := 10
| _ := 0

theorem elevator_stops_most_on_10th_floor : ∀ n : ℕ, (n ≥ 1 ∧ n ≤ 10) → n = 10 := 
by 
  intros n hn,
  cases n with 
  | nat.zero        => simp at hn; linarith
  | nat.succ n_prev => sorry

end elevator_stops_most_on_10th_floor_l189_189713


namespace exponential_form_l189_189612

-- Define the variables representing the logarithms
variables (a m n : ℝ)
-- Given conditions
def log_a_2 : Prop := log a 2 = m
def log_a_3 : Prop := log a 3 = n

-- The theorem we need to prove
theorem exponential_form (h1 : log_a_2 a m) (h2 : log_a_3 a n) : a^(2*m + n) = 12 :=
by sorry

end exponential_form_l189_189612


namespace scaled_division_l189_189458

theorem scaled_division (hq : quotient 12 7 = 1) (hr : remainder 12 7 = 5) : 
  quotient (100 * 12) (100 * 7) = 1 ∧ remainder (100 * 12) (100 * 7) = 500 := 
sorry

end scaled_division_l189_189458


namespace consecutive_integers_sum_l189_189896

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189896


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189086

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189086


namespace max_connected_colour_l189_189488

-- Definition to capture the concept of a chess queen's move
structure QueenMove (n : ℕ) :=
  (x y : ℕ)
  (is_valid : ∀ (x' y' : ℕ), (x' = x ∨ y' = y ∨ abs (x' - x) = abs (y' - y)) → (x' < n) → (y' < n) → Prop)

-- Definition of a connected colour
def is_connected (board : ℕ → ℕ → ℕ) (color : ℕ) : Prop :=
  ∃ (x y : ℕ), ∀ (x' y' : ℕ), 
  (board x y = color) ∧ 
  (board x' y' = color) → 
  (∃ (move : QueenMove 2009), move.is_valid x' y')

-- Main theorem statement
theorem max_connected_colour {n : ℕ} :
  (∀ (board : ℕ → ℕ → ℕ), (∀ i j, board i j < n) → 
  ∃ color < n, is_connected board color) 
  → 
  n > 4017 :=
sorry

end max_connected_colour_l189_189488


namespace min_value_quadratic_l189_189249

theorem min_value_quadratic (x y : ℝ) : x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ 
                                        (∀ x y : ℝ, x^2 + 2*x*y + 2*y^2 = 0 → x = 0 ∧ y = 0) :=
by
  -- Part 1: Prove the expression is always non-negative
  have h1 : x^2 + 2*x*y + 2*y^2 = (x + y)^2 + y^2 := by
    ring,
  
  have h2: (x + y)^2 ≥ 0 := by
    apply sq_nonneg,

  have h3: y^2 ≥ 0 := by
    apply sq_nonneg,

  exact ⟨add_nonneg h2 h3, sorry⟩ -- This is a sketch. The remainder of the proof would involve showing the uniqueness of the minimum at (0, 0)

end min_value_quadratic_l189_189249


namespace tangent_triangle_side_length_l189_189378

theorem tangent_triangle_side_length : 
  ∀ (O1 O2 O3 : Type) [metric_space O1] [metric_space O2] [metric_space O3], 
  dist O1 O2 = 2 → dist O2 O3 = 2 → dist O3 O1 = 2 → 
  let r : ℝ := 1 in
  let side_length_of_triangle_XYZ := 4 in
  side_length_of_triangle_XYZ = 4 := 
by
  intros O1 O2 O3 h1 h2 h3 r side_length_of_triangle_XYZ
  sorry

end tangent_triangle_side_length_l189_189378


namespace asymptotes_of_hyperbola_l189_189590

noncomputable def is_asymptote (eq : ℝ → ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), eq x y → (x ≠ 0 → y ≠ 0 → y = (x * real.sqrt 5 / 2) ∨ y = -(x * real.sqrt 5 / 2))

theorem asymptotes_of_hyperbola :
  is_asymptote (λ x y, x^2 / 4 - y^2 / 5 = 0) :=
sorry

end asymptotes_of_hyperbola_l189_189590


namespace scooter_gain_percent_l189_189424

theorem scooter_gain_percent 
  (purchase_price : ℝ) (repair_costs : ℝ) (selling_price : ℝ) 
  (h1 : purchase_price = 800) (h2 : repair_costs = 200) (h3 : selling_price = 1200) : 
  ((selling_price - (purchase_price + repair_costs)) / (purchase_price + repair_costs)) * 100 = 20 :=
by
  sorry

end scooter_gain_percent_l189_189424


namespace factor_quadratic_expression_l189_189043

theorem factor_quadratic_expression (a b : ℤ) :
  (∃ a b : ℤ, (5 * a + 5 * b = -125) ∧ (a * b = -100) → (a + b = -25)) → (25 * x^2 - 125 * x - 100 = (5 * x + a) * (5 * x + b)) := 
by
  sorry

end factor_quadratic_expression_l189_189043


namespace find_initial_books_l189_189009

-- Define the initial number of books Paul had
def initial_books (B : ℕ) : Prop :=
  B - 94 + 150 = 58

-- The theorem statement that needs to be proven
theorem find_initial_books : ∃ B : ℕ, initial_books B ∧ B = 2 :=
by
  -- Construct the proof based on the conditions and given answer
  use 2
  unfold initial_books
  show 2 - 94 + 150 = 58
  sorry

end find_initial_books_l189_189009


namespace hyperbola_eccentricity_l189_189329

theorem hyperbola_eccentricity {
  (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
  (asymptote_tangent_hyperbola : ∀ (x y : ℝ),
    (x^2 / a^2 - y^2 / b^2 = 1) →
    (x^2 + (y-2)^2 = 1) →
    Real.abs (0 - 2 * a) / Real.sqrt (a^2 + b^2) = 1) :
  (e : ℝ) (h_ecc : e = Real.sqrt(1 + (b / a)^2)) :
  e = 2 := by
  sorry

end hyperbola_eccentricity_l189_189329


namespace multiple_of_Allyson_age_l189_189505

theorem multiple_of_Allyson_age (Hiram_age : ℕ) (Allyson_age : ℕ) (M : ℕ)
  (H : Hiram_age = 40)
  (A : Allyson_age = 28)
  (C : Hiram_age + 12 + 4 = M * Allyson_age) :
  M = 2 :=
by {
  rw [H, A] at C,
  norm_num at C,
  exact Nat.eq_of_mul_eq_mul_right (by norm_num) C,
}

end multiple_of_Allyson_age_l189_189505


namespace consecutive_integer_product_sum_l189_189937

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189937


namespace minimize_sum_pos_maximize_product_pos_l189_189469

def N : ℕ := 10^1001 - 1

noncomputable def find_min_sum_position : ℕ := 996

noncomputable def find_max_product_position : ℕ := 995

theorem minimize_sum_pos :
  ∀ m : ℕ, (m ≠ find_min_sum_position) → 
      (2 * 10^m + 10^(1001-m) - 10) ≥ (2 * 10^find_min_sum_position + 10^(1001-find_min_sum_position) - 10) := 
sorry

theorem maximize_product_pos :
  ∀ m : ℕ, (m ≠ find_max_product_position) → 
      ((2 * 10^m - 1) * (10^(1001 - m) - 9)) ≤ ((2 * 10^find_max_product_position - 1) * (10^(1001 - find_max_product_position) - 9)) :=
sorry

end minimize_sum_pos_maximize_product_pos_l189_189469


namespace right_triangle_not_isosceles_l189_189514

-- Define the triangle and the sides
structure Triangle :=
  (A B C : ℝ)
  (a b c : ℝ)
  (angle_a not_equal_90 : B ≠ 1)

-- The main statement translating the problem conditions to a formal Lean statement
theorem right_triangle_not_isosceles
  (ABC : Triangle)
  (h1 : log (sqrt ABC.b) ABC.a = log ABC.b (4 * ABC.a - 4))
  (h2 : log (sqrt ABC.b) ABC.c = log ABC.b (4 * ABC.c - 4))
  (h3 : C / A = 2)
  (h4 : sin B / sin A = 2)
  (h5 : ABC.angle_b = 90) :
  (A * max(sin A, sin C) ≠ max(sin C, sin B)) := sorry

end right_triangle_not_isosceles_l189_189514


namespace largest_non_sum_of_36_and_composite_l189_189094

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l189_189094


namespace integral_result_l189_189571

noncomputable def definite_integral : ℝ :=
  ∫ x in 0..(2 * Real.pi / 3), (Real.cos x) ^ 2 / (1 + Real.cos x + Real.sin x) ^ 2

theorem integral_result :
  definite_integral = (Real.sqrt 3) / 2 - Real.log 2 :=
by
  sorry

end integral_result_l189_189571


namespace cubes_diagonal_passes_through_l189_189152

noncomputable theory

/-- Prove that number of unit cubes an internal diagonal passes through in a rectangular solid of dimensions 120x280x360 is 600. -/
theorem cubes_diagonal_passes_through 
  (a b c : ℕ) (h1 : a = 120) (h2 : b = 280) (h3 : c = 360) :
  let gcd_ab := Nat.gcd a b in
  let gcd_bc := Nat.gcd b c in
  let gcd_ca := Nat.gcd c a in
  let gcd_abc := Nat.gcd (Nat.gcd a b) c in
  (a + b + c - gcd_ab - gcd_bc - gcd_ca + gcd_abc) = 600 := by
{
  sorry
}

end cubes_diagonal_passes_through_l189_189152


namespace chessboard_one_black_cell_impossible_l189_189157

theorem chessboard_one_black_cell_impossible :
  ¬ ∃ (m : ℕ) (n : ℕ) (flip : ℕ → ℕ → bool) (repaint : ℕ → Prop),
    let init_state := (m = 8) ∧ (n = 8) ∧ (flip (i : ℕ) (j : ℕ) := (i + j) % 2 = 0)
    in ∀ board : (ℕ × ℕ) → bool, 
         (∀ i j, board (i, j) = if flip i j then tt else ff) →
         (∃ k, (Π i j, let cell := board (i, j) in
                      k = if repaint i then !cell else cell
                    ) ∧ ( ∑ i j, (if (k i j) then 1 else 0)) = 1 ) :=
by
  sorry

end chessboard_one_black_cell_impossible_l189_189157


namespace percent_freshmen_psychology_majors_l189_189565

theorem percent_freshmen_psychology_majors
  (P_total F_total : ℝ)
  (freshmen_percent : F_total = 0.4 * P_total)
  (liberal_arts_percent_of_freshmen : ∀ f, f ∈ freshman_set → school f = "Liberal Arts" → f.percent = 0.6 * 1)
  (psychology_majors_percent_of_liberal_arts_freshmen: ∀ f, f ∈ liberal_arts_freshman_set → major f = "Psychology" → f.percent = 0.5 * 1)
  (total_percent := F_total * 0.6 * 0.5 / P_total) :
total_percent = 0.12 :=
by 
  sorry

end percent_freshmen_psychology_majors_l189_189565


namespace find_line_eq_l189_189832

noncomputable def circle_eq (x y: ℝ) : Prop := x^2 + y^2 - 2*x - 4*y = 0

noncomputable def line_perpendicular_eq (l: ℝ) (x y: ℝ) : Prop := 2*x - y + l = 0

noncomputable def center := (1 : ℝ, 2 : ℝ)

theorem find_line_eq (l: ℝ) :
  -- Conditions
  (∀ x y : ℝ, circle_eq x y → line_perpendicular_eq l x y) →
  (∀ (x y: ℝ), (x + 2 * y = 0) → (2*x - y +l = 0)) →
  -- Question
  l = 0 := 
by
  intros H1 H2
          sorry

end find_line_eq_l189_189832


namespace service_fee_correct_l189_189804
open Nat -- Open the natural number namespace

-- Define the conditions
def ticket_price : ℕ := 44
def num_tickets : ℕ := 3
def total_paid : ℕ := 150

-- Define the cost of tickets
def cost_of_tickets : ℕ := ticket_price * num_tickets

-- Define the service fee calculation
def service_fee : ℕ := total_paid - cost_of_tickets

-- The proof problem statement
theorem service_fee_correct : service_fee = 18 :=
by
  -- Omits the proof, providing a placeholder.
  sorry

end service_fee_correct_l189_189804


namespace convex_polygon_longest_sides_convex_polygon_shortest_sides_l189_189740

noncomputable def convex_polygon : Type := sorry

-- Definitions for the properties and functions used in conditions
def is_convex (P : convex_polygon) : Prop := sorry
def equal_perimeters (A B : convex_polygon) : Prop := sorry
def longest_side (P : convex_polygon) : ℝ := sorry
def shortest_side (P : convex_polygon) : ℝ := sorry

-- Problem part a
theorem convex_polygon_longest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ∃ (A B : convex_polygon), equal_perimeters A B ∧ longest_side A = longest_side B :=
sorry

-- Problem part b
theorem convex_polygon_shortest_sides (P : convex_polygon) (h_convex : is_convex P) :
  ¬(∀ (A B : convex_polygon), equal_perimeters A B → shortest_side A = shortest_side B) :=
sorry

end convex_polygon_longest_sides_convex_polygon_shortest_sides_l189_189740


namespace consecutive_integers_sum_l189_189902

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189902


namespace correct_statements_l189_189662

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem correct_statements :
  (∃ c : ℝ × ℝ, c = (-1, 1) ∧ ∀ x, f(x) = 1 - 1 / (x + 1)) ∧
  (∀ x : ℝ, x > -1 → f(x + 1) > f(x)) ∧
  (∑ i in (finset.range 2022), f(i + 1) + ∑ i in (finset.range 2022), f(1 / (i + 1)) = 2021) ∧
  (f 1 + 2021 = 2021.5)
:= by
  sorry

end correct_statements_l189_189662


namespace nice_sequence_characterization_l189_189785

-- Define a function that maps positive integers to positive integers
def f : ℕ+ → ℕ+ := sorry

-- Define what it means for a sequence to be "nice"
def is_nice_sequence (a : ℕ+ → ℤ) : Prop :=
  ∀ (i j n : ℕ+), (a i ≡ a j [MOD n]) ↔ (i ≡ j [MOD (f n)])

-- Main theorem statement in Lean 4
theorem nice_sequence_characterization (a : ℕ+ → ℤ) :
  is_nice_sequence a → (∃ (d : ℕ), d = 1 ∨ d = 2 ∨ 
    (∃ (α β : ℤ), ∀ (i : ℕ+), a i = α + ↑i * β))
  :=
  sorry

end nice_sequence_characterization_l189_189785


namespace percent_increase_l189_189511

theorem percent_increase (original value new_value : ℕ) (h1 : original_value = 20) (h2 : new_value = 25) :
  ((new_value - original_value) / original_value) * 100 = 25 :=
by
  -- Proof omitted
  sorry

end percent_increase_l189_189511


namespace fenced_area_l189_189531

theorem fenced_area (l w s : ℕ) (h1 : l = 20) (h2 : w = 18) (h3 : s = 4) :
  l * w - 2 * s * s = 328 :=
by
  rw [h1, h2, h3]
  -- l * w = 20 * 18
  -- 2 * s * s = 2 * 4 * 4
  norm_num
  -- 20 * 18 - 2 * 4 * 4 = 328
  norm_num
  sorry

end fenced_area_l189_189531


namespace smallest_powerful_integer_l189_189171

def is_powerful (k : ℕ) : Prop :=
  ∃ (p q r s t : ℕ), 
    p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t ∧
    p > 0 ∧ q > 0 ∧ r > 0 ∧ s > 0 ∧ t > 0 ∧
    p^2 ∣ k ∧ q^3 ∣ k ∧ r^5 ∣ k ∧ s^7 ∣ k ∧ t^{11} ∣ k

theorem smallest_powerful_integer : ∃ k : ℕ, is_powerful k ∧ k = 1024 :=
by
  sorry

end smallest_powerful_integer_l189_189171


namespace proof_perpendicular_ac_bs_l189_189186

variable (S A B C X Y : Type) [Nonempty S] [Nonempty A] [Nonempty B] [Nonempty C] [Nonempty X] [Nonempty Y]
variable (h1 : ∀ (α : Type) [Nonempty α], Type)
variable (h2 : h1 S) (h3 : h1 A) (h4 : h1 B) (h5 : h1 C) (h6 : h1 X) (h7 : h1 Y)

-- Faces of the pyramid are acute-angled triangles
def faces_acute_angled : Prop := ∀ (T : Type), acute_angled_triangle T

-- SX and SY are the altitudes of the faces ASB and BSC respectively
def is_altitude (P Q R S : Type) [Nonempty P] [Nonempty Q] [Nonempty R] [Nonempty S] : Prop :=
  ∀ (hP : P) (hQ : Q) (hR : R) (hS : S), altitude hP hQ hR hS

-- Quadrilateral AXYC is cyclic
def quad_cyclic (P Q R S : Type) [Nonempty P] [Nonempty Q] [Nonempty R] [Nonempty S] : Prop :=
  cyclic_quadrilateral P Q R S

-- Proving AC ⊥ BS
def perpendicularities (P Q R : Type) : Prop :=
  is_perpendicular P Q R

theorem proof_perpendicular_ac_bs (h1 : faces_acute_angled SABC)
  (h2 : is_altitude SX ASB) (h3 : is_altitude SY BSC)
  (h4 : quad_cyclic AXYC) :
  perpendicularities AC BS := 
sorry

end proof_perpendicular_ac_bs_l189_189186


namespace find_cost_of_fourth_cd_l189_189964

variables (cost1 cost2 cost3 cost4 : ℕ)
variables (h1 : (cost1 + cost2 + cost3) / 3 = 15)
variables (h2 : (cost1 + cost2 + cost3 + cost4) / 4 = 16)

theorem find_cost_of_fourth_cd : cost4 = 19 := 
by 
  sorry

end find_cost_of_fourth_cd_l189_189964


namespace solve_Dirichlet_Helmholtz_l189_189029

noncomputable def helmholtz_solution (r : ℝ) (φ : ℝ) : ℝ :=
  (3 / 4) * (BesselJ 1 (2 * r) / BesselJ 1 2) * sin φ - (1 / 4) * (BesselJ 3 (2 * r) / BesselJ 3 2) * sin (3 * φ)

theorem solve_Dirichlet_Helmholtz :
  (∀ (r φ : ℝ), 0 < r → r < 1 → (Laplacian (λ (r φ : ℝ), (helmholtz_solution r φ)) + 4 * (helmholtz_solution r φ) = 0))
  ∧ (∀ (φ : ℝ), (helmholtz_solution 1 φ = sin φ ^ 3)) :=
by
  sorry

end solve_Dirichlet_Helmholtz_l189_189029


namespace cow_cost_calculation_l189_189476

constant hearts_per_card : ℕ := 4
constant cards_in_deck : ℕ := 52
constant cost_per_cow : ℕ := 200

def total_hearts : ℕ := hearts_per_card * cards_in_deck
def number_of_cows : ℕ := 2 * total_hearts
def total_cost_of_cows : ℕ := number_of_cows * cost_per_cow

theorem cow_cost_calculation :
  total_cost_of_cows = 83200 := by
  -- Placeholder proof
  sorry

end cow_cost_calculation_l189_189476


namespace sum_series_eq_l189_189609

theorem sum_series_eq :
  (Finset.sum (Finset.range 128) (λ k, 1 / ((2 * (k + 1) - 1) * (2 * (k + 1) + 1)) )) = 128 / 257 := 
sorry

end sum_series_eq_l189_189609


namespace sum_of_consecutive_integers_with_product_812_l189_189941

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189941


namespace sophie_height_l189_189811

theorem sophie_height (tree_height : ℝ) (tree_shadow : ℝ) (sophie_shadow : ℝ) 
    (tree_height_eq : tree_height = 50) (tree_shadow_eq : tree_shadow = 25) 
    (sophie_shadow_eq : sophie_shadow = 18) : 
    ∃ sophie_height : ℝ, sophie_height = 36 := 
by
  have ratio : ℝ := tree_height / tree_shadow
  have sophie_height := ratio * sophie_shadow
  use sophie_height
  rw [tree_height_eq, tree_shadow_eq, sophie_shadow_eq]
  norm_num at *
  sorry

end sophie_height_l189_189811


namespace probability_heads_tails_heads_l189_189503

theorem probability_heads_tails_heads :
  let p : ℝ := 1 / 2 in
  (p) * (1 / 2) * (1 / 2) = (1 / 8) :=
by
  let p : ℝ := 1 / 2
  sorry

end probability_heads_tails_heads_l189_189503


namespace complex_multiplication_l189_189322

def imaginary_unit := Complex.I

theorem complex_multiplication (h : imaginary_unit^2 = -1) : (3 + 2 * imaginary_unit) * imaginary_unit = -2 + 3 * imaginary_unit :=
by
  sorry

end complex_multiplication_l189_189322


namespace min_value_quadratic_l189_189248

theorem min_value_quadratic (x y : ℝ) : x^2 + 2*x*y + 2*y^2 ≥ 0 ∧ 
                                        (∀ x y : ℝ, x^2 + 2*x*y + 2*y^2 = 0 → x = 0 ∧ y = 0) :=
by
  -- Part 1: Prove the expression is always non-negative
  have h1 : x^2 + 2*x*y + 2*y^2 = (x + y)^2 + y^2 := by
    ring,
  
  have h2: (x + y)^2 ≥ 0 := by
    apply sq_nonneg,

  have h3: y^2 ≥ 0 := by
    apply sq_nonneg,

  exact ⟨add_nonneg h2 h3, sorry⟩ -- This is a sketch. The remainder of the proof would involve showing the uniqueness of the minimum at (0, 0)

end min_value_quadratic_l189_189248


namespace solve_for_a_l189_189672

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end solve_for_a_l189_189672


namespace problem_l189_189842

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189842


namespace option_b_does_not_represent_5x_l189_189125

theorem option_b_does_not_represent_5x (x : ℝ) : 
  (∀ a, a = 5 * x ↔ a = x + x + x + x + x) →
  (¬ (5 * x = x * x * x * x * x)) :=
by
  intro h
  -- Using sorry to skip the proof.
  sorry

end option_b_does_not_represent_5x_l189_189125


namespace sum_of_consecutive_integers_with_product_812_l189_189918

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189918


namespace velocity_eq_l189_189834

noncomputable def s (t : ℝ) : ℝ := 2 * t * sin t + t

theorem velocity_eq (t : ℝ) : deriv (λ t, s t) t = 2 * sin t + 2 * t * cos t + 1 := 
sorry

end velocity_eq_l189_189834


namespace anna_ate_cupcakes_l189_189560

-- Given conditions
def total_cupcakes : Nat := 60
def cupcakes_given_away (total : Nat) : Nat := (4 * total) / 5
def cupcakes_remaining (total : Nat) : Nat := total - cupcakes_given_away total
def anna_cupcakes_left : Nat := 9

-- Proving the number of cupcakes Anna ate
theorem anna_ate_cupcakes : cupcakes_remaining total_cupcakes - anna_cupcakes_left = 3 := by
  sorry

end anna_ate_cupcakes_l189_189560


namespace meaningful_condition_l189_189705

theorem meaningful_condition (x : ℝ) :
  (1 - x >= 0) ∧ (x + 2 ≠ 0) ↔ (x ≤ 1) ∧ (x ≠ -2) :=
begin
  sorry,
end

end meaningful_condition_l189_189705


namespace num_sets_l189_189836

theorem num_sets {A : Set ℕ} :
  {1} ⊆ A ∧ A ⊆ {1, 2, 3, 4, 5} → ∃ n, n = 16 := 
by
  sorry

end num_sets_l189_189836


namespace worth_of_presents_l189_189749

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l189_189749


namespace ellipse_hyperbola_foci_same_a_value_l189_189294

theorem ellipse_hyperbola_foci_same_a_value (a : ℝ) 
  (h1 : a > 0)
  (h2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / 4 = 1)
  (h3 : ∀ x y : ℝ, x^2 / 9 - y^2 / 3 = 1)
  (h4 : ∃ c : ℝ, (h2.foci = c) ∧ (h3.foci = c)) :
  a = 4 :=
sorry

end ellipse_hyperbola_foci_same_a_value_l189_189294


namespace complement_cardinality_l189_189675

open Set

def A : Set ℕ := {4, 5, 7, 9}
def B : Set ℕ := {3, 4, 7, 8, 9}
def U : Set ℕ := A ∪ B

theorem complement_cardinality :
  (U \ (A ∩ B)).card = 3 := by
  sorry

end complement_cardinality_l189_189675


namespace number_of_zeros_l189_189535

theorem number_of_zeros (n : ℕ) (h : (∑ (i : ℕ) in (list.range n).map (λ _, 9), id i) = 252) : n = 28 :=
by
  sorry

end number_of_zeros_l189_189535


namespace parabola_directrix_standard_equation_l189_189331

-- The problem definition
def parabola_directrix_eq (directrix : ℝ → Prop) : Prop :=
  ∃ p : ℝ, 2 * p = 14 ∧ directrix (-7)

-- The standard form equation to prove
theorem parabola_directrix_standard_equation (directrix : ℝ → Prop) :
  parabola_directrix_eq directrix → ∃ p : ℝ, p = 7 ∧ ∀ x y : ℝ, directrix x → (y^2 = 4*p*x) :=
begin
  sorry
end

end parabola_directrix_standard_equation_l189_189331


namespace consecutive_integers_sum_l189_189850

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189850


namespace find_F_of_circle_l189_189295

def circle_equation (x y F : ℝ) : Prop :=
  x^2 + y^2 - 2*x + 2*y + F = 0

def is_circle_with_radius (x y F r : ℝ) : Prop := 
  ∃ k h, (x - k)^2 + (y + h)^2 = r

theorem find_F_of_circle {F : ℝ} :
  (∀ x y : ℝ, circle_equation x y F) ∧ 
  is_circle_with_radius 1 1 F 4 → F = -2 := 
by
  sorry

end find_F_of_circle_l189_189295


namespace slope_and_intercept_of_line_l189_189953

theorem slope_and_intercept_of_line :
  ∀ (x y : ℝ), 3 * x + 2 * y + 6 = 0 → y = - (3 / 2) * x - 3 :=
by
  intros x y h
  sorry

end slope_and_intercept_of_line_l189_189953


namespace consecutive_integer_product_sum_l189_189934

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189934


namespace max_subjects_per_teacher_l189_189174

theorem max_subjects_per_teacher (teachers_math : ℕ) (teachers_physics : ℕ) (teachers_chemistry : ℕ) (min_teachers : ℕ) :
  teachers_math = 11 ∧ teachers_physics = 8 ∧ teachers_chemistry = 5 ∧ min_teachers = 8 → 
  ∃ x : ℕ, (8 * x = (teachers_math + teachers_physics + teachers_chemistry)) ∧ x = 3 :=
begin
  sorry
end

end max_subjects_per_teacher_l189_189174


namespace find_tangent_line_equation_l189_189042

noncomputable def tangent_equation : Prop :=
  let f := (λ x : ℝ, x / (x + 2)) in
  let slope := deriv f in
  let m := slope (-1) in
  let point := (-1 : ℝ, -1 : ℝ) in
  y = m * (x - point.1) + point.2 = (y = 2 * (x + 1) - 1)
  
theorem find_tangent_line_equation : tangent_equation := sorry

end find_tangent_line_equation_l189_189042


namespace percentage_prefer_city_Y_l189_189414

theorem percentage_prefer_city_Y :
  ∀ (total_employees : ℕ)
    (percent_rel_X : ℕ)
    (percent_rel_Y : ℕ)
    (max_preferred_relocation : ℕ),
     total_employees = 200 →
     percent_rel_X = 30 →
     percent_rel_Y = 70 →
     max_preferred_relocation = 140 →
     percent_rel_Y = 70 :=
by
  intros total_employees percent_rel_X percent_rel_Y max_preferred_relocation
  intros h_total_employees h_percent_rel_X h_percent_rel_Y h_max_preferred_relocation
  rw [h_total_employees, h_percent_rel_X, h_percent_rel_Y, h_max_preferred_relocation]
  sorry

end percentage_prefer_city_Y_l189_189414


namespace consecutive_integers_sum_l189_189888

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189888


namespace lucka_cuts_l189_189787

theorem lucka_cuts (a b c d e : ℕ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 3) (h4 : d = 4) (h5 : e = 5)
  (h_difference : ∀ (x y : ℕ), x = 52341 → y = 23415 → x - y = 28926) :
  (cut1 cut2 : ℕ) (h_cut1 : cut1 = 1 ∧ cut2 = 5)
  (h_pos1 : cut1 > 0) (h_pos2 : cut2 < 6) (h_cut_valid : cut1 < cut2) : 
  cut1 = 2 ∧ cut2 = 4 :=
by
  assume a b c d e h1 h2 h3 h4 h5 h_difference cut1 cut2 h_cut1 h_pos1 h_pos2 h_cut_valid
  split
  case cut1_eq_2 : 
    sorry
  case cut2_eq_4 :
    sorry

end lucka_cuts_l189_189787


namespace log_c_a_lt_log_c_b_l189_189268

theorem log_c_a_lt_log_c_b 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a > b) 
  (ha_ne_one : a ≠ 1) (hb_ne_one : b ≠ 1) 
  (hc : 0 < c) (hc1 : c < 1) : 
  Real.log c a < Real.log c b := 
sorry

end log_c_a_lt_log_c_b_l189_189268


namespace area_triangle_DEF_l189_189601

open Real

noncomputable def D : Point := ⟨0, 0⟩
noncomputable def E : Point := ⟨2 * sqrt 3, 0⟩
noncomputable def F : Point := ⟨0, 2⟩

noncomputable def ∠EDF := 90
noncomputable def ∠DEF := 60
noncomputable def DF := 8

noncomputable def area_triangle (A B C : Point) : ℝ :=
  0.5 * abs ((B.x - A.x) * (C.y - A.y) - (B.y - A.y) * (C.x - A.x))

theorem area_triangle_DEF : area_triangle D E F = 32 * sqrt 3 / 3 :=
  sorry

end area_triangle_DEF_l189_189601


namespace rectangular_prism_prime_edge_length_form_l189_189446

theorem rectangular_prism_prime_edge_length_form (a b c : ℕ) (h1 : a.prime) (h2 : b.prime) (h3 : c.prime)
  (h4 : ∃ (p : ℕ) (n : ℕ), Nat.prime p ∧ (2 * (a * b + b * c + c * a) = p ^ n)) :
  (∃ k : ℕ, (a = 2 ^ k - 1) ∨ (b = 2 ^ k -1) ∨ (c = 2 ^ k - 1)) ∧
  (¬(∃ j : ℕ, (a = 2 ^ j - 1) ∧ (b = 2 ^ j - 1)) ∧ ¬(∃ j : ℕ, (a = 2 ^ j - 1) ∧ (c = 2 ^ j - 1)) ∧ ¬(∃ j : ℕ, (b = 2 ^ j - 1) ∧ (c = 2 ^ j - 1))) :=
sorry

end rectangular_prism_prime_edge_length_form_l189_189446


namespace problem_l189_189987

theorem problem (n : ℕ) (h : n > 0) : (n % 42 = 0) -> 
  ∃ k : ℤ, (n * (n + 2016) * (n + 2 * 2016) * ... * (n + 2015 * 2016)) = k * (2016!) :=
by
  sorry

end problem_l189_189987


namespace child_support_calculation_l189_189404

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l189_189404


namespace consecutive_integer_sum_l189_189867

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189867


namespace sounds_meet_at_x_l189_189567

theorem sounds_meet_at_x (d c s : ℝ) (h1 : 0 < d) (h2 : 0 < c) (h3 : 0 < s) :
  ∃ x : ℝ, x = d / 2 * (1 + s / c) ∧ x <= d ∧ x > 0 :=
by
  sorry

end sounds_meet_at_x_l189_189567


namespace math_problem_l189_189710

-- Define x, y, z, and w
variables (x y z w : ℝ)

-- Conditions provided in the problem
def cond1 : Prop := y = 1.75 * x
def cond2 : Prop := z = 0.60 * y
def cond3 : Prop := w = 1.25 * ((x + z) / 2)

-- Question and correct answer
def question : Proposition :=
  let percentage := ((w - x) / w) * 100 in
  percentage ≈ 21.95

theorem math_problem
  (cond1 : cond1)
  (cond2 : cond2)
  (cond3 : cond3) :
  question :=
sorry

end math_problem_l189_189710


namespace rational_coeff_terms_count_l189_189728

theorem rational_coeff_terms_count : 
  ∃ n : ℕ, 
  n = 6 ∧ 
  (∀ k : ℕ, 0 ≤ k ∧ k ≤ 30 → 
    (C (nat.succ 30) k * (-65) ^ k : ℚ).isRational ↔ k % 6 = 0) := 
sorry

end rational_coeff_terms_count_l189_189728


namespace length_of_AB_l189_189605

def A : ℝ × ℝ × ℝ := (2, -3, 5)
def B : ℝ × ℝ × ℝ := (2, -3, -5)

theorem length_of_AB : real.dist A B = 10 :=
by
  sorry

end length_of_AB_l189_189605


namespace point_not_on_line_pq_neg_l189_189327

theorem point_not_on_line_pq_neg (p q : ℝ) (h : p * q < 0) : ¬ (21 * p + q = -101) := 
by sorry

end point_not_on_line_pq_neg_l189_189327


namespace sqrt_sum_identity_l189_189317

theorem sqrt_sum_identity (α : ℝ) (h : (5 * real.pi / 2) ≤ α ∧ α ≤ (7 * real.pi / 2)) :
  sqrt (1 + sin α) + sqrt (1 - sin α) = sqrt (2 - cos α) :=
sorry

end sqrt_sum_identity_l189_189317


namespace simplify_expression_l189_189808

theorem simplify_expression (θ : ℝ) (h : 0 ≤ θ ∧ θ ≤ π) :
  sqrt (2 - cos θ - sin θ + sqrt (3 * (1 - cos θ) * (1 + cos θ - 2 * sin θ))) = 
  sqrt 3 * abs (sin (θ / 2)) + abs (cos (θ / 2)) * sqrt (1 - 2 * tan (θ / 2)) :=
by
  sorry

end simplify_expression_l189_189808


namespace verify_correct_answer_l189_189071

-- Definitions used in conditions
def oppo_numbers (x y : ℤ) := x = -y
def congruent (Δ₁ Δ₂ : Triangle) := congruent Δ₁ Δ₂
def real_roots (p q r : ℝ) := discriminant(p, q, r) ≥ 0

-- Propositions
def prop1 := ∀ x y : ℤ, x + y = 0 → oppo_numbers x y
def inv_prop1 := ∀ x y : ℤ, oppo_numbers x y → x + y = 0

def prop2 := ∀ Δ₁ Δ₂ : Triangle, congruent Δ₁ Δ₂ → area Δ₁ = area Δ₂
def neg_prop2 := ∃ Δ₁ Δ₂ : Triangle, congruent Δ₁ Δ₂ ∧ area Δ₁ ≠ area Δ₂

def prop3 := ∀ (q : ℝ), (q ≤ 1 → real_roots 1 2 q)
def contrap_prop3 := ∀ (q : ℝ), (¬real_roots 1 2 q → q > 1)

def prop4 := ∀ Δ : Triangle, interior_angles_equal Δ → scalene Δ
def inv_prop4 := ∀ Δ : Triangle, scalene Δ → ¬interior_angles_equal Δ

-- The problem tuple to verify the correct answer
theorem verify_correct_answer : 
  ((inv_prop1 = True) → (neg_prop2 = False) → (contrap_prop3 = True) → (inv_prop4 = False) → (answer = C)) := 
sorry

end verify_correct_answer_l189_189071


namespace spadesuit_computation_l189_189611

def spadesuit (x y : ℝ) := x - 1 / (y^2)

theorem spadesuit_computation : spadesuit 3 (spadesuit 3 3) = 1947 / 676 := by
  sorry

end spadesuit_computation_l189_189611


namespace smallest_number_divisible_l189_189513

theorem smallest_number_divisible (x : ℕ) :
  (∃ n : ℕ, x = n * 5 + 24) ∧
  (∃ n : ℕ, x = n * 10 + 24) ∧
  (∃ n : ℕ, x = n * 15 + 24) ∧
  (∃ n : ℕ, x = n * 20 + 24) →
  x = 84 :=
by
  sorry

end smallest_number_divisible_l189_189513


namespace derivative_y_l189_189692

-- Define the function y given the conditions
def y (x : ℝ) : ℝ := -2 * exp x * sin x

-- State the theorem to prove that the derivative of y equals the specified expression
theorem derivative_y (x : ℝ) : (deriv y x) = -2 * exp x * (sin x + cos x) :=
sorry

end derivative_y_l189_189692


namespace train_length_l189_189699

theorem train_length (x : ℕ) (h1 : (310 + x) / 18 = x / 8) : x = 248 :=
  sorry

end train_length_l189_189699


namespace smallest_class_size_l189_189337

theorem smallest_class_size
  (x : ℕ)
  (h1 : ∀ y : ℕ, y = x + 2)
  (total_students : 5 * x + 2 > 40) :
  ∃ (n : ℕ), n = 5 * x + 2 ∧ n = 42 :=
by
  sorry

end smallest_class_size_l189_189337


namespace inequality_holds_for_a_l189_189654

theorem inequality_holds_for_a (a : ℝ) (x : ℝ) (hx : x > 1) :
  (a = 3) → ¬ (∀ x, x > 1 → (a + 1) / x + log x > a) :=
begin
  intro ha,
  rw ha,
  contrapose,
  simp only [not_forall],
  use 3,
  simp,
end

end inequality_holds_for_a_l189_189654


namespace triangle_area_is_17_point_5_l189_189495

-- Define the points A, B, and C as tuples of coordinates
def A : (ℝ × ℝ) := (2, 2)
def B : (ℝ × ℝ) := (7, 2)
def C : (ℝ × ℝ) := (4, 9)

-- Function to calculate the area of a triangle given its vertices
noncomputable def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1 / 2) * abs ((x1 * (y2 - y3)) + (x2 * (y3 - y1)) + (x3 * (y1 - y2)))

-- The theorem statement asserting the area of the triangle is 17.5 square units
theorem triangle_area_is_17_point_5 :
  area_of_triangle A B C = 17.5 :=
by
  sorry -- Proof is omitted

end triangle_area_is_17_point_5_l189_189495


namespace consecutive_integers_sum_l189_189890

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189890


namespace sum_of_consecutive_integers_with_product_812_l189_189912

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189912


namespace Hari_joined_after_five_months_l189_189797

noncomputable def months_until_Hari_joins (x : ℕ) : Prop :=
  let Praveen_investment := 3220 * 12
  let Hari_investment := 8280 * (12 - x)
  (Praveen_investment : ℚ) / Hari_investment = (2 : ℚ) / 3

theorem Hari_joined_after_five_months : months_until_Hari_joins 5 :=
by 
  let Praveen_investment := 3220 * 12
  let Hari_investment := 8280 * (12 - 5)
  show (Praveen_investment : ℚ) / Hari_investment = (2 : ℚ) / 3,
  calc (Praveen_investment : ℚ) / Hari_investment
      = (3220 * 12 : ℚ)/ (8280 * 7) : by sorry
    ... = (2 : ℚ)/ 3 : by sorry

end Hari_joined_after_five_months_l189_189797


namespace mass_of_man_is_120_l189_189523

def length_of_boat : ℝ := 3
def breadth_of_boat : ℝ := 2
def height_water_rise : ℝ := 0.02
def density_of_water : ℝ := 1000
def volume_displaced : ℝ := length_of_boat * breadth_of_boat * height_water_rise
def mass_of_man := density_of_water * volume_displaced

theorem mass_of_man_is_120 : mass_of_man = 120 :=
by
  -- insert the detailed proof here
  sorry

end mass_of_man_is_120_l189_189523


namespace renaldo_distance_l189_189423

theorem renaldo_distance (R : ℕ) (h : R + (1/3 : ℝ) * R + 7 = 27) : R = 15 :=
by sorry

end renaldo_distance_l189_189423


namespace symmetrical_point_with_respect_to_origin_l189_189724

theorem symmetrical_point_with_respect_to_origin :
  (∀ (P : ℝ × ℝ), P = (3, 2) → ∃ (P' : ℝ × ℝ), P' = (-3, -2) ∧ P' = (-P.1, -P.2)) :=
by
  intro P hP
  use (-3, -2)
  simp [hP]
  constructor; apply rfl; sorry

end symmetrical_point_with_respect_to_origin_l189_189724


namespace trey_will_sell_bracelets_for_days_l189_189080

def cost : ℕ := 112
def price_per_bracelet : ℕ := 1
def bracelets_per_day : ℕ := 8

theorem trey_will_sell_bracelets_for_days :
  ∃ d : ℕ, d = cost / (price_per_bracelet * bracelets_per_day) ∧ d = 14 := by
  sorry

end trey_will_sell_bracelets_for_days_l189_189080


namespace question1_question2_l189_189308

-- Conditions
def A : set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (m : ℝ) : set ℝ := {x | m - 2 ≤ x ∧ x ≤ m + 2}

-- Question 1 Statement
theorem question1 (m : ℝ) : (A ∪ B m = A) → m = 1 := sorry

-- Question 2 Statement
theorem question2 (m : ℝ) : (A ∩ B m = {x | 0 ≤ x ∧ x ≤ 3}) → m = 2 := sorry

end question1_question2_l189_189308


namespace mens_tshirts_sales_interval_l189_189566

theorem mens_tshirts_sales_interval (
  womens_interval: ℕ,
  womens_price: ℕ,
  mens_price: ℕ,
  total_revenue_per_week: ℕ,
  open_hours_per_week: ℕ
): ℕ :=
  sorry

open_hours_per_week := 84
let womens_interval := 30
let womens_price := 18
let mens_price := 15
let total_revenue_per_week := 4914

example : mens_tshirts_sales_interval womens_interval womens_price mens_price total_revenue_per_week open_hours_per_week = 40 := 
by {
  -- the proof will go here
  sorry
}

end mens_tshirts_sales_interval_l189_189566


namespace sum_of_c_n_l189_189279

noncomputable def a_n (n : ℕ) : ℕ := 
  4 * n - 3

noncomputable def b_n (n : ℕ) : ℕ :=
  2^n - 1

noncomputable def c_n (n : ℕ) : ℝ :=
  (4 * n - 3) * (1 / 2)^n

noncomputable def T_n (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, c_n (i + 1)

theorem sum_of_c_n (n : ℕ) : 
  T_n n = 5 - (4 * n + 5) * (1 / 2)^n :=
sorry

end sum_of_c_n_l189_189279


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189100

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189100


namespace greatest_difference_of_units_l189_189958

theorem greatest_difference_of_units (d : ℕ) (h : d = 0 ∨ d = 5) : 
  ∃ diff : ℕ, diff = 5 ∧ (diff = |d - 0| ∨ diff = |5 - d|) :=
by 
  use 5
  split
  · rfl
  · cases h
    · left
      rw h
      exact nat.sub_self 0
    · right
      rw h
      exact rfl

end greatest_difference_of_units_l189_189958


namespace train_speed_l189_189970

theorem train_speed (v : ℕ) :
    let distance_between_stations := 155
    let speed_of_train_from_A := 20
    let start_time_train_A := 7
    let start_time_train_B := 8
    let meet_time := 11
    let distance_traveled_by_A := speed_of_train_from_A * (meet_time - start_time_train_A)
    let remaining_distance := distance_between_stations - distance_traveled_by_A
    let traveling_time_train_B := meet_time - start_time_train_B
    v * traveling_time_train_B = remaining_distance → v = 25 :=
by
  intros
  sorry

end train_speed_l189_189970


namespace counting_books_l189_189688

-- Definitions based on the given conditions
variables (mystery_books fantasy_books biographies science_fiction_books : ℕ)
variables (choices : ℕ)

-- Hypotheses based on the conditions
def distinct_books : Prop :=
  (mystery_books = 4) ∧
  (fantasy_books = 4) ∧
  (biographies = 3) ∧
  (science_fiction_books = 2)

-- The main theorem
theorem counting_books (h : distinct_books) : choices = 277 :=
sorry

end counting_books_l189_189688


namespace extreme_values_f_value_of_a_range_of_a_l189_189303

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2 * a * log x
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := 2 * a * x
noncomputable def h (a : ℝ) (x : ℝ) : ℝ := f a x - g a x

-- Theorem for the extreme values of f(x)
theorem extreme_values_f (a : ℝ) :
  (a ≤ 0 → ∀ x : ℝ, f a x ≠ min) ∧ 
  (a > 0 → ∃ x_min : ℝ, ∃ x_max : ℝ, x_min = sqrt a ∧ f a x_min = min ∧ ¬(∃ y : ℝ, f a y > f a x_min)) := 
sorry

-- Theorem for finding the value of 'a' (a = 1/2)
theorem value_of_a (a : ℝ) (ha : a > 0) :
  (∃ x : ℝ, h a x = 0 ∧ ∀ y : ℝ, h a y ≥ 0) ↔ a = 1 / 2 := 
sorry

-- Theorem for the range of 'a'
theorem range_of_a (a : ℝ) (ha : 0 < a ∧ a < 1) :
  (∀ x1 x2 : ℝ, (1 ≤ x1 ∧ x1 < x2 ∧ x2 ≤ 2) → |f a x1 - f a x2| > |g a x1 - g a x2|)
  → (0 < a ∧ a ≤ 1 / 2) :=
sorry

end extreme_values_f_value_of_a_range_of_a_l189_189303


namespace exists_nat_divisibility_l189_189598

theorem exists_nat_divisibility (n : ℕ) (prime : ℕ → Prop) (nat_divisibility : ∀ (n d : ℕ), n % d = 0) : 
  (n = 1806) → (∀ (p : ℕ), prime p → (nat_divisibility n p  ↔ nat_divisibility n (p - 1))) :=
by sorry

end exists_nat_divisibility_l189_189598


namespace geometric_figure_area_l189_189708

theorem geometric_figure_area :
  (∀ (z : ℂ),
     (0 < (z.re / 20)) ∧ ((z.re / 20) < 1) ∧ 
     (0 < (z.im / 20)) ∧ ((z.im / 20) < 1) ∧ 
     (0 < (20 / z.re)) ∧ ((20 / z.re) < 1) ∧ 
     (0 < (20 / z.im)) ∧ ((20 / z.im) < 1)) →
     (∃ (area : ℝ), area = 400 - 50 * Real.pi) :=
by
  sorry

end geometric_figure_area_l189_189708


namespace complex_number_on_imaginary_axis_l189_189821

theorem complex_number_on_imaginary_axis (a : ℝ) : (∃ z : ℂ, z = (-2 + a * complex.I) / (1 + complex.I) ∧ z.re = 0) -> a = 2 :=
by
    sorry

end complex_number_on_imaginary_axis_l189_189821


namespace consecutive_integers_sum_l189_189853

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189853


namespace complex_imaginary_axis_l189_189820

theorem complex_imaginary_axis (a : ℝ): 
  (∃ b : ℝ, ((-2 + a * complex.I) / (1 + complex.I)) = complex.I * b) → a = 2 := by
  sorry

end complex_imaginary_axis_l189_189820


namespace sum_of_consecutive_integers_with_product_812_l189_189926

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189926


namespace sum_of_consecutive_integers_with_product_812_l189_189909

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189909


namespace counterexample_to_strict_inequality_l189_189742

theorem counterexample_to_strict_inequality :
  ∃ (a1 a2 b1 b2 c1 c2 d1 d2 : ℕ),
  (0 < a1) ∧ (0 < a2) ∧ (0 < b1) ∧ (0 < b2) ∧ (0 < c1) ∧ (0 < c2) ∧ (0 < d1) ∧ (0 < d2) ∧
  (a1 * b2 < a2 * b1) ∧ (c1 * d2 < c2 * d1) ∧ ¬ (a1 + c1) * (b2 + d2) < (a2 + c2) * (b1 + d1) :=
sorry

end counterexample_to_strict_inequality_l189_189742


namespace equal_piecewise_paths_l189_189628

theorem equal_piecewise_paths {a n : ℤ} {b0 bn : ℝ} 
  (h_b0_pos : b0 > 0) 
  (h_bn_pos : bn > 0) : 
  (number_of_paths (a, b0) (a + n, bn) = number_of_paths (a, -b0) (a + n, bn)) := 
sorry

end equal_piecewise_paths_l189_189628


namespace haniMoreSitupsPerMinute_l189_189684

-- Define the conditions given in the problem
def totalSitups : Nat := 110
def situpsByDiana : Nat := 40
def rateDianaPerMinute : Nat := 4

-- Define the derived conditions from the solution steps
def timeDianaMinutes := situpsByDiana / rateDianaPerMinute -- 10 minutes
def situpsByHani := totalSitups - situpsByDiana -- 70 situps
def rateHaniPerMinute := situpsByHani / timeDianaMinutes -- 7 situps per minute

-- The theorem we need to prove
theorem haniMoreSitupsPerMinute : rateHaniPerMinute - rateDianaPerMinute = 3 :=
by
  -- Placeholder for the actual proof
  sorry

end haniMoreSitupsPerMinute_l189_189684


namespace problem_l189_189840

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189840


namespace maximal_sum_of_squares_of_sides_l189_189631

noncomputable def sum_of_squares_of_sides (vertices : List (ℝ × ℝ)) : ℝ :=
  vertices.zip (vertices.tail ++ [vertices.head]).sum_by (λ (v1, v2), (v1.1 - v2.1)^2 + (v1.2 - v2.2)^2)

def is_inscribed (vertices : List (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ) : Prop :=
  vertices.all (λ v, (v.1 - center.1)^2 + (v.2 - center.2)^2 = radius^2)

theorem maximal_sum_of_squares_of_sides 
  (center : ℝ × ℝ) (radius : ℝ) (vertices : List (ℝ × ℝ)) :
  is_inscribed vertices center radius →
  ∀ vertices', is_inscribed vertices' center radius → 
  sum_of_squares_of_sides vertices ≤ sum_of_squares_of_sides vertices' →
  (vertices' ≡ equilateral_triangle_inscribed center radius) :=
sorry

end maximal_sum_of_squares_of_sides_l189_189631


namespace consecutive_integers_sum_l189_189856

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189856


namespace consecutive_integers_sum_l189_189898

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189898


namespace triangle_construction_l189_189221

-- Definitions for the conditions
def length_of_side (c : ℝ) := c > 0
def ratio_of_sides (λ : ℝ) := λ > 0
def angle_difference (diff : ℝ) := diff > 0

-- Main theorem statement
theorem triangle_construction (c λ diff : ℝ) (hc : length_of_side c) (hλ : ratio_of_sides λ) (hdiff : angle_difference diff) :
  ∃ (a b : ℝ) (α β : ℝ), a > 0 ∧ b > 0 ∧ α > 0 ∧ β > 0
  ∧ a = λ * b ∧ (|α - β| = diff) ∧ valid_triangle c a b α β :=
sorry


end triangle_construction_l189_189221


namespace power_zero_equals_one_specific_case_l189_189492

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end power_zero_equals_one_specific_case_l189_189492


namespace tournament_games_l189_189150

theorem tournament_games (n : ℕ) (h : n = 50) : (n * (n - 1) / 2) * 4 = 4900 :=
by {
  rw h, 
  -- Compute the number of games played
  have pair_combinations : (50 * 49 / 2) = 1225, by norm_num,
  rw pair_combinations,
  norm_num,
  }

end tournament_games_l189_189150


namespace wage_increase_is_40_percent_l189_189184

-- Define the old and new wages
def old_wage : ℝ := 20
def new_wage : ℝ := 28

-- Define the percentage increase based on the given conditions
def percentage_increase (old_wage new_wage : ℝ) : ℝ :=
  ((new_wage - old_wage) / old_wage) * 100

-- The theorem stating the percentage increase is 40%
theorem wage_increase_is_40_percent :
  percentage_increase old_wage new_wage = 40 := by
  sorry

end wage_increase_is_40_percent_l189_189184


namespace lines_intersection_l189_189732

open Real

noncomputable def lines_intersect_at_single_point (A1 A2 A3 : ℝ × ℝ) (l : Set (ℝ × ℝ))
  (alpha1 alpha2 alpha3 : ℝ) : Prop :=
  let angle_with_l (A : ℝ × ℝ) (alpha : ℝ) := α - α  -- Dummy implementation
  ∃ P : ℝ × ℝ,
  (angle_with_l A1 (π - alpha1) = angle_with_l P l) ∧
  (angle_with_l A2 (π - alpha2) = angle_with_l P l) ∧
  (angle_with_l A3 (π - alpha3) = angle_with_l P l)

theorem lines_intersection (A1 A2 A3 : ℝ × ℝ) (l : Set (ℝ × ℝ))
  (alpha1 alpha2 alpha3 : ℝ)
  (h_alpha1 : 0 ≤ alpha1 ∧ alpha1 < 2 * π)
  (h_alpha2 : 0 ≤ alpha2 ∧ alpha2 < 2 * π)
  (h_alpha3 : 0 ≤ alpha3 ∧ alpha3 < 2 * π)
  (h_no_collinear : ¬ collinear ({A1, A2, A3} : Set (ℝ × ℝ))) :
  lines_intersect_at_single_point A1 A2 A3 l α1 α2 α3 := sorry

end lines_intersection_l189_189732


namespace smallest_side_of_triangle_l189_189019

variable {α : Type} [LinearOrderedField α]

theorem smallest_side_of_triangle (a b c : α) (h : a^2 + b^2 > 5*c^2) : c ≤ a ∧ c ≤ b :=
by
  sorry

end smallest_side_of_triangle_l189_189019


namespace total_donation_l189_189205

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l189_189205


namespace country_can_be_divided_into_three_provinces_l189_189070

noncomputable def divide_into_three_provinces (V : Type*) (edges : set (V × V)) :=
  ∃ (color : V → fin 3), ∀ (v1 v2 : V), (v1, v2) ∈ edges → color v1 ≠ color v2

theorem country_can_be_divided_into_three_provinces (V : Type*) [fintype V] (edges : set (V × V)) (h_tree : ∀ (u v : V), ∃! (p : list V), p.head = u ∧ p.last = v ∧ ∀ (w ∈ p.tail.erase_last), (p.head, w) ∈ edges ∨ (w, p.head) ∈ edges) :
  divide_into_three_provinces V edges :=
begin
  sorry
end

end country_can_be_divided_into_three_provinces_l189_189070


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189098

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189098


namespace ratio_of_areas_is_one_l189_189968

noncomputable def regular_pentagon := sorry -- Assume there is a definition of a regular pentagon.
noncomputable def first_circle (p : regular_pentagon) := sorry -- Circle tangent to AB and extended sides.
noncomputable def second_circle (p : regular_pentagon) := sorry -- Circle tangent to DE and extended sides.

theorem ratio_of_areas_is_one (p : regular_pentagon) :
  let A1 := π * (first_circle p)^2,
      A2 := π * (second_circle p)^2 in
  A2 / A1 = 1 := 
by
  sorry

end ratio_of_areas_is_one_l189_189968


namespace sum_of_consecutive_integers_with_product_812_l189_189925

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189925


namespace student_exam_limit_l189_189189

theorem student_exam_limit :
  tendsto (λ n : ℕ, 1 - (1 - 1 / (n : ℝ)) ^ n) at_top (𝓝 (1 - real.exp (-1))) :=
sorry

end student_exam_limit_l189_189189


namespace problem_solution_l189_189347

-- Define the parametric form of curve C
def param_C (α : ℝ) : ℝ × ℝ :=
  (sqrt 3 * cos α, sin α)

-- Standard form of the circle C
def standard_C (x y : ℝ) : Prop :=
  x^2 / 3 + y^2 = 1

-- Polar equation of line l
def polar_l (ρ θ : ℝ) : Prop :=
  sqrt 2 / 2 * ρ * cos (θ + π / 4) = -1

-- Cartesian form of line l
def cartesian_l (x y : ℝ) : Prop :=
  x - y + 2 = 0

-- Parametric form of line l_1
def param_l1 (t : ℝ) : ℝ × ℝ :=
  (-1 + sqrt 2 / 2 * t, sqrt 2 / 2 * t)

-- Point M
def point_M : ℝ × ℝ := (-1, 0)

-- Product of distances from M to points A and B
def product_distance (t1 t2 : ℝ) : ℝ :=
  abs (t1 * t2)

theorem problem_solution :
  (∀ α, standard_C (param_C α).fst (param_C α).snd) ∧
  (∀ ρ θ, polar_l ρ θ → cartesian_l ρ θ) ∧
  (∃ t1 t2, ∀ t, (standard_C (param_l1 t).fst (param_l1 t).snd) →
    param_l1 t = (A, B) ∧
    product_distance t1 t2 = 1) :=
sorry

end problem_solution_l189_189347


namespace solve_for_x_l189_189429

theorem solve_for_x (x : ℤ) (h : 3 * x - 7 = 11) : x = 6 :=
by
  sorry

end solve_for_x_l189_189429


namespace pyramid_divide_edge_ratio_l189_189216

def SquareBasePyramid (A B C D E : Point) : Prop :=
  base_vertices_square A B C D ∧ apex_vertex E ∧ equal_side_edges A B C D E

def PointDividesInRatio (P A E : Point) (r : ℚ) : Prop :=
  divides_segment_in_ratio A E P r

def Midpoint (Q C E : Point) : Prop :=
  midpoint Q C E

def PlaneDividesEdgeInRatio (D P Q B E : Point) (r : ℚ) : Prop :=
  plane_through_points D P Q ∧ divides_edge_in_ratio B E r

theorem pyramid_divide_edge_ratio (A B C D E P Q : Point) :
  SquareBasePyramid A B C D E →
  PointDividesInRatio P A E (3 / 1) →
  Midpoint Q C E →
  PlaneDividesEdgeInRatio D P Q B E (4 / 3) :=
sorry

end pyramid_divide_edge_ratio_l189_189216


namespace tangent_line_through_point_l189_189655

noncomputable def tangent_line_at_point (x : ℝ) (y : ℝ) : ℝ :=
  x^2 * (1 - x)

theorem tangent_line_through_point {x y : ℝ} (h : ∀ x, y = (1/3) * x ^ 3)-by sorry :=
  let m := x^2 in
  let P := (2, (1/3) * 2^3) in  
  m = 4 ∧ P = (12 * x - 3 * y - 16 = 0 ∨ 3 * y - 3 * x - (-2)) :=
  by sorry
  

end tangent_line_through_point_l189_189655


namespace initial_puppies_l189_189556

theorem initial_puppies (gave_away : ℕ) (left : ℕ) (initial : ℕ) (h1 : gave_away = 7) (h2 : left = 5) :
  initial = gave_away + left :=
by {
  unfold gave_away left initial,
  rfl,
  sorry
}

end initial_puppies_l189_189556


namespace total_pennies_donated_l189_189210

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l189_189210


namespace largest_non_sum_of_36_and_composite_l189_189092

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l189_189092


namespace salary_percentage_change_l189_189802

noncomputable def initial_salary (S : ℝ) : ℝ := S

noncomputable def first_decrease (S : ℝ) : ℝ := S * 0.40

noncomputable def first_increase (S : ℝ) : ℝ := first_decrease S * 1.60

noncomputable def second_decrease (S : ℝ) : ℝ := first_increase S * 0.55

noncomputable def second_increase (S : ℝ) : ℝ := second_decrease S * 1.20

noncomputable def final_decrease (S : ℝ) : ℝ := second_increase S * 0.75

def percentage_change (initial final : ℝ) : ℝ := (final - initial) / initial * 100

theorem salary_percentage_change (S : ℝ) : 
  percentage_change S (final_decrease S) = -68.32 := 
sorry

end salary_percentage_change_l189_189802


namespace problem_statement_l189_189203

variable (x y : ℝ)

theorem problem_statement :
  ({- (1 / 2) * x * y^2}^3 = - (1 / 8) * x^3 * y^6) :=
by
  sorry

end problem_statement_l189_189203


namespace person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l189_189421

-- Define the probability of hitting the target for Person A and Person B
def p_hit_A : ℚ := 2 / 3
def p_hit_B : ℚ := 3 / 4

-- Define the complementary probabilities (missing the target)
def p_miss_A := 1 - p_hit_A
def p_miss_B := 1 - p_hit_B

-- Prove the probability that Person A, shooting 4 times, misses the target at least once
theorem person_A_misses_at_least_once_in_4_shots :
  (1 - (p_hit_A ^ 4)) = 65 / 81 :=
by 
  sorry

-- Prove the probability that Person B stops shooting exactly after 5 shots
-- due to missing the target consecutively 2 times
theorem person_B_stops_after_5_shots_due_to_2_consecutive_misses :
  (p_hit_B * p_hit_B * p_miss_B * (p_miss_B * p_miss_B)) = 45 / 1024 :=
by
  sorry

end person_A_misses_at_least_once_in_4_shots_person_B_stops_after_5_shots_due_to_2_consecutive_misses_l189_189421


namespace angle_bisectors_l189_189376

open EuclideanGeometry

-- Definitions of points and conditions
variable (A B C D P Q O : Point)

-- Assuming intersection conditions
axiom H1 : ConvexQuadrilateral ABCD
axiom H2 : LineIntersects P A B ∧  LineIntersects P C D
axiom H3 : LineIntersects Q A D ∧  LineIntersects Q B C
axiom H4 : LineIntersects O A C ∧  LineIntersects O B D
axiom H5 : ∠ P O Q = 90

-- The statement to prove: PO is the bisector of ∠ AOD and OQ is the bisector of ∠ AOB
theorem angle_bisectors (H : ConvexQuadrilateral ABCD) (H2 : LineIntersects P A B ∧  LineIntersects P C D)
  (H3 : LineIntersects Q A D ∧  LineIntersects Q B C)
  (H4 : LineIntersects O A C ∧  LineIntersects O B D)
  (H5 : ∠ P O Q = 90) : 
  AngleBisector (LineThrough P O) (Angle A O D) ∧ AngleBisector (LineThrough Q O) (Angle A O B) :=
by
  sorry

end angle_bisectors_l189_189376


namespace solve_log_sin_eq_l189_189430

noncomputable def log_base (b : ℝ) (a : ℝ) : ℝ :=
  Real.log a / Real.log b

theorem solve_log_sin_eq :
  ∀ x : ℝ, 
  (0 < Real.sin x ∧ Real.sin x < 1) →
  log_base (Real.sin x) 4 * log_base (Real.sin x ^ 2) 2 = 4 →
  ∃ k : ℤ, x = (-1)^k * (Real.pi / 4) + Real.pi * k := 
by
  sorry

end solve_log_sin_eq_l189_189430


namespace max_value_arithmetic_sequence_l189_189725

theorem max_value_arithmetic_sequence
  (a : ℕ → ℝ)
  (a1 d : ℝ)
  (h1 : a 1 = a1)
  (h_diff : ∀ n : ℕ, a (n + 1) = a n + d)
  (ha1_pos : a1 > 0)
  (hd_pos : d > 0)
  (h1_2 : a1 + (a1 + d) ≤ 60)
  (h2_3 : (a1 + d) + (a1 + 2 * d) ≤ 100) :
  5 * a1 + (a1 + 4 * d) ≤ 200 :=
sorry

end max_value_arithmetic_sequence_l189_189725


namespace total_surface_area_of_cube_l189_189962

theorem total_surface_area_of_cube (edge_sum : ℕ) (h_edge_sum : edge_sum = 180) :
  ∃ (S : ℕ), S = 1350 := 
by
  sorry

end total_surface_area_of_cube_l189_189962


namespace consecutive_integer_sum_l189_189862

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189862


namespace explicit_expression_monotonically_increasing_intervals_l189_189305

-- Define the function based on given conditions
def f (x : ℝ) : ℝ := (5/2) * x^4 - (9/2) * x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) : ℝ := 10 * x^3 - 9 * x

-- Lean statements for the proof

-- 1. Prove that the function is correct
theorem explicit_expression :
  (∀ x : ℝ, f x = (5/2) * x^4 - (9/2) * x^2 + 1) :=
by sorry

-- 2. Prove the intervals where the function is increasing
theorem monotonically_increasing_intervals :
  (∀ x : ℝ, ((-3 * sqrt 10 / 10 < x ∧ x < 0) ∨ (3 * sqrt 10 / 10 < x)) → f' x > 0) :=
by sorry

end explicit_expression_monotonically_increasing_intervals_l189_189305


namespace tangent_line_intersects_twice_range_of_a_for_extreme_points_l189_189780

def f (a x : ℝ) : ℝ := (x - a) ^ 2
def g (x : ℝ) : ℝ := Real.log x
def F (a x : ℝ) : ℝ := f a x + g x
def G (a x : ℝ) : ℝ := f a x * g x

theorem tangent_line_intersects_twice (a : ℝ) : 
  let l := λ x : ℝ, (3 - 2 * a) * (x - 1) + (1 - a) ^ 2 
  in ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ F a x1 = l x1 ∧ F a x2 = l x2 := sorry

theorem range_of_a_for_extreme_points : 
  ∀ a : ℝ, (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (∂ (G a x) / ∂ x = 0 ∧ f a x * g x = 0)) 
  ↔ a ∈ Set.Ioo (-2 * Real.exp (-3 / 2)) 0 ∪ Set.Ioo 0 1 ∪ Set.Ioi 1 := sorry

end tangent_line_intersects_twice_range_of_a_for_extreme_points_l189_189780


namespace max_distinct_integers_l189_189230

theorem max_distinct_integers:
  ∃ n : ℕ, (∀ (k_1 k_2 ... k_n : ℕ), (∀ i, 1 ≤ k_i) ∧ (i ≠ j → k_i ≠ k_j)) →
  k_1^2 + k_2^2 + ... + k_n^2 = 2050 →
  n ≤ 16 :=
sorry

end max_distinct_integers_l189_189230


namespace determine_a_l189_189297

def f (x : ℝ) (a : ℝ) : ℝ := 
  if x ≠ 3 then 2 / (|x - 3|) else a

theorem determine_a (a : ℝ) :
  (∃ x1 x2 x3 : ℝ, x1 ≠ 3 ∧ x2 ≠ 3 ∧ x3 = 3 ∧ (f x1 a - 4 = 0) ∧ (f x2 a - 4 = 0) ∧ (f x3 a - 4 = 0)) →
  a = 4 := by
  sorry

end determine_a_l189_189297


namespace num_distinct_roots_abs_quadratic_eq_l189_189835

theorem num_distinct_roots_abs_quadratic_eq (x : ℝ) :
  (|x^2 - 6 * x| = 9).count_roots = 3 :=
sorry

end num_distinct_roots_abs_quadratic_eq_l189_189835


namespace find_larger_number_l189_189041

theorem find_larger_number (L S : ℕ) 
  (h1 : L - S = 1390)
  (h2 : L = 6 * S + 15) : 
  L = 1665 :=
sorry

end find_larger_number_l189_189041


namespace incenter_circumcenter_collinear_symm_l189_189794

variables (A B C H E F B' C' I O : Point)

-- The midpoint definition function
def is_midpoint (E : Point) (P Q : Point) : Prop :=
  dist E P = dist E Q / 2

-- The incenter and circumcenter collinearity proof
theorem incenter_circumcenter_collinear_symm (triangle : Triangle A B C)
  (H_orth : is_orthocenter H triangle)
  (E_mid : is_midpoint E H A)
  (incircle_touches_AB : incircle triangle touches AB at C')
  (incircle_touches_AC : incircle triangle touches AC at B')
  (symmetric_EF : is_symmetric E F (line B' C'))
  (incenter : is_incenter I triangle)
  (circumcenter : is_circumcenter O triangle) :
  lies_on_line F (line_through I O) :=
begin
  sorry
end

end incenter_circumcenter_collinear_symm_l189_189794


namespace valeries_thank_you_cards_l189_189972

variables (T R J B : ℕ)

theorem valeries_thank_you_cards :
  B = 2 →
  R = B + 3 →
  J = 2 * R →
  T + (B + 1) + R + J = 21 →
  T = 3 :=
by
  intros hB hR hJ hTotal
  sorry

end valeries_thank_you_cards_l189_189972


namespace existence_of_moments_l189_189384

noncomputable theory

-- Definitions
variables {X : ℝ → ℝ} {α : ℝ} {F : ℝ → ℝ}

-- Conditions
axiom non_neg_random_var : ∀ x, 0 ≤ X x
axiom continuous_distribution : ∀ x, continuous_at F x
axiom function_G : ∀ x, (1 - F x) ≠ 0 → (1 - F x) > 0

-- The function G(x)
def G (x : ℝ) : ℝ := (1 - F x)⁻¹/2

-- Problem statement
theorem existence_of_moments
  (hα : 0 < α ∧ α < 1)
  (finite_EG : ∫ x in 0..1, G (α * x) ^ 2 dF x < ∞) :
  ∀ n, ∫ x in 0..∞, x ^ n dF x < ∞ :=
sorry

end existence_of_moments_l189_189384


namespace complex_conjugate_of_fraction_l189_189243

theorem complex_conjugate_of_fraction :
  ∀ z : ℂ, z = (3 - complex.I) / (1 - complex.I) → conj z = 2 - complex.I :=
by
  intro z
  intro h
  sorry

end complex_conjugate_of_fraction_l189_189243


namespace streetlights_distance_l189_189474

theorem streetlights_distance (n : ℕ) (d : ℕ) (first last : ℕ) :
  n = 45 → d = 60 → first = 1 → last = n → 
  (last - first) * d / 1000 = 2.64 := 
by
  intros hn hd hfirst hlast
  rw [hn, hd, hfirst, hlast]
  norm_num
  sorry

end streetlights_distance_l189_189474


namespace scientific_notation_l189_189507

theorem scientific_notation (x : ℝ) (h : x = 0.000 815) : ∃ a n, (1 ≤ a) ∧ (a < 10) ∧ (n < 0) ∧ (x = a * 10^n) :=
by {
  use [8.15, -4],
  sorry
}

end scientific_notation_l189_189507


namespace area_of_region_enclosed_by_equation_l189_189494

noncomputable def enclosed_area : ℝ :=
  π / 2

theorem area_of_region_enclosed_by_equation :
  (∃ (x y : ℝ), x^2 + y^2 = |x| - |y|) → 
  (∃ area : ℝ, area = enclosed_area) :=
by
  sorry

end area_of_region_enclosed_by_equation_l189_189494


namespace symmetric_center_of_f_range_of_f_not_real_f_is_monotonically_increasing_sum_of_f_values_l189_189664

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem symmetric_center_of_f :
  ∃ x y, f(-1) = 1 :=
sorry

theorem range_of_f_not_real :
  set.range f ≠ set.univ :=
sorry

theorem f_is_monotonically_increasing :
  ∃ a b, ∀ x, a < x ∧ x < b → f(x) ≤ f(x + 1) :=
sorry

theorem sum_of_f_values :
  ∑ n in (finset.range 2022).image (λ n, f (n + 1)) + 
    ∑ n in (finset.range 2022).image (λ n, f (1 / (n + 1))) = 4043 / 2 :=
sorry

end symmetric_center_of_f_range_of_f_not_real_f_is_monotonically_increasing_sum_of_f_values_l189_189664


namespace largest_positive_integer_not_sum_of_36_and_composite_l189_189107

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l189_189107


namespace ellipse_semi_minor_axis_l189_189334

noncomputable def semiMinorAxis (c a : ℝ) : ℝ := 
  real.sqrt (a^2 - c^2)

theorem ellipse_semi_minor_axis :
  let center := (-3, 1)
  let focus := (-3, 0)
  let endpoint := (-3, 3)
  let c := real.abs (1 - 0)
  let a := real.abs (1 - 3)
  in semiMinorAxis c a = real.sqrt 3 :=
by
  sorry

end ellipse_semi_minor_axis_l189_189334


namespace max_elements_in_S_l189_189390

open Nat

/-- Let T be the set of all positive divisors of 2020^100. -/
def T : Set ℕ :=
  {d | ∃ (a b c : ℕ), d = 2^a * 5^b * 101^c ∧ 0 ≤ a ∧ a ≤ 200 ∧ 0 ≤ b ∧ b ≤ 100 ∧ 0 ≤ c ∧ c ≤ 100}

/-- Let S be the subset of T with the condition that no element of S is a multiple of another. -/
variable (S : Set ℕ)

axiom S_subset_T : S ⊆ T
axiom no_multiples_in_S : ∀ (x y : ℕ), x ∈ S → y ∈ S → (x ∣ y ∨ y ∣ x) → x = y

/-- Prove that the maximum number of elements in S is at most 10201. -/
theorem max_elements_in_S : ∃ (n : ℕ), n ≤ 101 ^ 2 ∧ ∀ t, t ∈ S → Cardinal.mk S ≤ 101 ^ 2 :=
by sorry

end max_elements_in_S_l189_189390


namespace cannot_get_105_one_stone_piles_l189_189996

def initial_piles : List ℕ := [51, 49, 5]

def combine_to_new_pile (a b : ℕ) : ℕ := a + b

def divide_even_pile (a : ℕ) (h : a % 2 = 0) : ℕ × ℕ := (a / 2, a / 2)

theorem cannot_get_105_one_stone_piles : 
  ¬ ∃ (piles : List ℕ),
    (initial_piles ∪ 
      {combine_to_new_pile a b | a b ∈ initial_piles ∨ 
       ∃ c (h : c % 2 = 0), c ∈ initial_piles}) ⊆ piles ∧
    piles.length = 105 ∧
    (∀ x ∈ piles, x = 1) :=
  sorry

end cannot_get_105_one_stone_piles_l189_189996


namespace amount_lent_to_B_l189_189536

theorem amount_lent_to_B
  (rate_of_interest_per_annum : ℝ)
  (P_C : ℝ)
  (years_C : ℝ)
  (total_interest : ℝ)
  (years_B : ℝ)
  (IB : ℝ)
  (IC : ℝ)
  (P_B : ℝ):
  (rate_of_interest_per_annum = 10) →
  (P_C = 3000) →
  (years_C = 4) →
  (total_interest = 2200) →
  (years_B = 2) →
  (IC = (P_C * rate_of_interest_per_annum * years_C) / 100) →
  (IB = (P_B * rate_of_interest_per_annum * years_B) / 100) →
  (total_interest = IB + IC) →
  P_B = 5000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end amount_lent_to_B_l189_189536


namespace approximate_value_accuracy_l189_189981

theorem approximate_value_accuracy (h : 48000 = 48000) :
  "Accurate to thousand" := by
  -- Add appropriate reasoning here
  sorry

end approximate_value_accuracy_l189_189981


namespace chord_length_correct_l189_189215

noncomputable def chord_length_square (r1 r2 r3 l: ℝ) (h1: r1 = 5) (h2: r2 = 10) (h3: r3 = 15) (h4: l = 722.2): 
  Prop := 
  ∃ PQ : ℝ, (PQ^2 = l) ∧ 
    (r1 + r2 = 15) ∧                       -- radius 5 and 10 are externally tangent
    (max r1 r2 + min r1 r2 = r3 - min r1 r2)    -- internally tangent to radius 15 circle

theorem chord_length_correct : chord_length_square 5 10 15 722.2 := 
  sorry

end chord_length_correct_l189_189215


namespace problem_xyz_divisibility_probability_l189_189796

-- Definitions for the given problem:
def is_divisible_by (m n : ℕ) := ∃ k, m = n * k

-- Probability calculation as a hypothesis using the given conditions.

theorem problem_xyz_divisibility_probability :
  let S := (Finset.range 2020).map Nat.succ in
  let probability (P : ℕ → ℕ → ℕ → Prop) :=
    (Finset.sum S (λ x, Finset.sum S (λ y, Finset.sum S (λ z, if P x y z then 1 else 0)))) / (2020.0 * 2020.0 * 2020.0) in
  let P x y z := is_divisible_by (x * y * z + x * y + x) 5 in 
  probability P = 33 / 125 :=
sorry

end problem_xyz_divisibility_probability_l189_189796


namespace exists_consecutive_naturals_with_one_prime_l189_189140

theorem exists_consecutive_naturals_with_one_prime (n : ℕ) :
  ∃ m, (∀ k, m ≤ k ∧ k < m + n → (k = m + n - 1 ∨ ¬ nat.prime k)) :=
sorry

end exists_consecutive_naturals_with_one_prime_l189_189140


namespace last_locker_opened_2046_l189_189524

def last_locker_opened (n : ℕ) : ℕ :=
  n - (n % 3)

theorem last_locker_opened_2046 : last_locker_opened 2048 = 2046 := by
  sorry

end last_locker_opened_2046_l189_189524


namespace find_g_l189_189386

noncomputable def bowtie (a b : ℝ) : ℝ := a + (ℕ → ℕ → ℝ) (λ n m, b.sqrt)[…]

theorem find_g (g : ℝ) (h : 5.bowtie g = 11) : g = 30 :=
by sorry

end find_g_l189_189386


namespace largest_consecutive_positive_elements_l189_189260

theorem largest_consecutive_positive_elements (a : ℕ → ℝ)
  (h₁ : ∀ n ≥ 2, a n = a (n-1) + a (n+2)) :
  ∃ m, m = 5 ∧ ∀ k < m, a k > 0 :=
sorry

end largest_consecutive_positive_elements_l189_189260


namespace largest_not_sum_of_36_and_composite_l189_189113

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l189_189113


namespace number_of_full_recipes_l189_189564

/-
Original number of students.
-/
def original_students := 150

/-
Expected drop in attendance (as a percentage).
-/
def attendance_drop_percent := 0.40

/-
Expected number of cookies per student.
-/
def cookies_per_student := 3

/-
Number of cookies per recipe.
-/
def cookies_per_recipe := 18

/-
Expected attendance after the drop.
-/
def expected_attendance := original_students * (1 - attendance_drop_percent)

/-
Total number of cookies needed.
-/
def total_cookies_needed := expected_attendance * cookies_per_student

/-
Number of full recipes required.
-/
def full_recipes_needed := total_cookies_needed / cookies_per_recipe

theorem number_of_full_recipes :
  full_recipes_needed = 15 :=
  sorry

end number_of_full_recipes_l189_189564


namespace find_square_value_l189_189169

theorem find_square_value (y : ℝ) (h : 4 * y^2 + 3 = 7 * y + 12) : (8 * y - 4)^2 = 202 := 
by
  sorry

end find_square_value_l189_189169


namespace mass_percentage_of_calcium_in_calcium_oxide_l189_189246

theorem mass_percentage_of_calcium_in_calcium_oxide
  (Ca_molar_mass : ℝ)
  (O_molar_mass : ℝ)
  (Ca_mass : Ca_molar_mass = 40.08)
  (O_mass : O_molar_mass = 16.00) :
  ((Ca_molar_mass / (Ca_molar_mass + O_molar_mass)) * 100) = 71.45 :=
by
  sorry

end mass_percentage_of_calcium_in_calcium_oxide_l189_189246


namespace infinite_geometric_series_sum_l189_189594

theorem infinite_geometric_series_sum :
  let a := 1 / 2
  let r := 1 / 3
  |r| < 1 → 
  let S := a / (1 - r)
  (S = 3 / 4) :=
by
  intro a r h
  let S := a / (1 - r)
  have hr : |r| < 1 := h
  show S = 3 / 4
  sorry

end infinite_geometric_series_sum_l189_189594


namespace train_meeting_distance_l189_189992

theorem train_meeting_distance :
  let distance := 150
  let time_x := 4
  let time_y := 3.5
  let speed_x := distance / time_x
  let speed_y := distance / time_y
  let relative_speed := speed_x + speed_y
  let time_to_meet := distance / relative_speed
  let distance_x_at_meeting := time_to_meet * speed_x
  distance_x_at_meeting = 70 := by
sorry

end train_meeting_distance_l189_189992


namespace percent_not_filler_l189_189530

theorem percent_not_filler
  (total_weight : ℕ)
  (vegetable_fillers : ℕ)
  (grain_fillers : ℕ)
  (h_total_weight : total_weight = 180)
  (h_vegetable_fillers : vegetable_fillers = 45)
  (h_grain_fillers : grain_fillers = 15)
  : (total_weight - (vegetable_fillers + grain_fillers)) / total_weight * 100 = 66.67 := 
sorry

end percent_not_filler_l189_189530


namespace norm_two_u_l189_189622

noncomputable def vector_u : ℝ × ℝ := sorry

theorem norm_two_u {u : ℝ × ℝ} (hu : ∥u∥ = 5) : ∥(2 : ℝ) • u∥ = 10 := by
  sorry

end norm_two_u_l189_189622


namespace part1_part2_l189_189296

-- Statement for Part 1
theorem part1 (a : ℝ) (x : ℝ) (h1 : 1 < x) (h2 : x < 3) :
  let fx := x * Real.log x - a * x^2 in
  (fx + a * x^2 - x + 2) / ((3 - x) * Real.exp x) > 1 / Real.exp 2 := sorry

-- Statement for Part 2
theorem part2 (a : ℝ) (h1 : 0 < a) (h2 : a < 1/Real.exp 1 ∨ 1/Real.exp 1 < a ∧ a < 1/2) :
  ∃ x : ℝ, 1 < x ∧ x < Real.exp 1 ∧
    ∃ F : ℝ → ℝ, ∀ t, F t = |t * Real.log t - a * t^2| ∧
    (∃ x0 : ℝ, 1 < x0 ∧ x0 < Real.exp 1 ∧ (∀ ε > 0, abs (F (x0 + ε) - F x0) < ε)) := sorry

end part1_part2_l189_189296


namespace simplify_fraction_l189_189630

variable (a b y : ℝ)
variable (h1 : y = (a + 2 * b) / a)
variable (h2 : a ≠ -2 * b)
variable (h3 : a ≠ 0)

theorem simplify_fraction : (2 * a + 2 * b) / (a - 2 * b) = (y + 1) / (3 - y) :=
by
  sorry

end simplify_fraction_l189_189630


namespace find_a_l189_189284

theorem find_a (a : ℝ) 
  (h : ∫ x in 1..a, (2 * x + 1 / x) = 3 + Real.log 2) : 
  a = 2 :=
by
  sorry

end find_a_l189_189284


namespace distinct_solutions_subtraction_l189_189771

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end distinct_solutions_subtraction_l189_189771


namespace consecutive_integer_sum_l189_189865

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189865


namespace line_has_equal_intercepts_find_a_l189_189670

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end line_has_equal_intercepts_find_a_l189_189670


namespace final_switches_in_A_l189_189073

open Set

-- Definition for the switch label
def switch_label (x y z : ℕ) : ℕ := 2^x * 3^y * 5^z

-- Condition: there are 1000 switches with specific labels
def valid_labels : Finset (ℕ × ℕ × ℕ) :=
  {(x, y, z) | 0 ≤ x ∧ x ≤ 9 ∧ 0 ≤ y ∧ y ≤ 9 ∧ 0 ≤ z ∧ z ≤ 9}

-- Condition: initially all switches are in position A (which we can define as 0)
def initial_position (label : ℕ × ℕ × ℕ) : Fin 4 := 0

-- Condition: A switch is advanced one step at each step i if its label divides the label of the i-th switch
def advance_step (i : ℕ) (pos : Fin 4) : Fin 4 := Fin.succ pos

-- Condition: A switch returns to position A if it is activated a multiple of 4 times
def is_position_A (steps : ℕ) : Prop := (steps % 4 = 0)

-- Main theorem to prove
theorem final_switches_in_A : 
  let total_steps (x y z : ℕ) : ℕ := (10-x)*(10-y)*(10-z) in
  let switches_in_A := (valid_labels.filter (λ (xyz : ℕ × ℕ × ℕ), is_position_A (total_steps xyz.1 xyz.2 xyz.3))).card in
  switches_in_A = 500 :=
sorry

end final_switches_in_A_l189_189073


namespace consecutive_integer_sum_l189_189870

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189870


namespace min_cubes_l189_189176

theorem min_cubes (a b c : ℕ) (h₁ : (a - 1) * (b - 1) * (c - 1) = 240) : a * b * c = 385 :=
  sorry

end min_cubes_l189_189176


namespace particular_solution_bounded_l189_189138

theorem particular_solution_bounded :
  ∃ y : ℝ → ℝ, 
    (∀ x : ℝ, (y'' x + 4 * y' x + 5 * y x = 8 * Real.cos x)) ∧ 
    (∀ x₀ : ℝ, (∀ x : ℝ, x < x₀ → |y x| < ∞)) ∧ 
    (∀ x : ℝ, y x = 2 * (Real.cos x + Real.sin x)) :=
begin
  sorry
end

end particular_solution_bounded_l189_189138


namespace workers_in_workshop_l189_189136

theorem workers_in_workshop :
  (∀ (W : ℕ), 8000 * W = 12000 * 7 + 6000 * (W - 7) → W = 21) :=
by
  intro W h
  sorry

end workers_in_workshop_l189_189136


namespace sum_of_consecutive_integers_with_product_812_l189_189908

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189908


namespace value_of_a_l189_189706

noncomputable def f (x a : ℝ) : ℝ := Real.log (1 - x) - Real.log (1 + x) + a

theorem value_of_a (a : ℝ) :
  let M := f (-1/2) a,
      N := f (1/2) a 
  in M + N = 1 → a = 1/2 :=
by
  let M := f (-1/2) a
  let N := f (1/2) a
  intro h
  sorry

end value_of_a_l189_189706


namespace find_even_decreasing_func_l189_189557

-- Define the functions
def func_A (x : ℝ) : ℝ := x^2
def func_B (x : ℝ) : ℝ := exp (-x)
def func_C (x : ℝ) : ℝ := x^(-2)
def func_D (x : ℝ) : ℝ := -x^3

-- Prove that func_C is the required function
theorem find_even_decreasing_func :
  (∀ x : ℝ, func_C x = func_C (-x)) ∧ (∀ x y : ℝ, 0 < x → x < y → func_C y < func_C x) →
  (∀ f : ℝ → ℝ, (func_A = f ∧ (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x)) 
  ∨ (func_B = f ∧ (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x)) 
  ∨ (func_C = f ∧ (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x)) 
  ∨ (func_D = f ∧ (∀ x : ℝ, f x = f (-x)) ∧ (∀ x y : ℝ, 0 < x → x < y → f y < f x)) 
  → f = func_C) :=
begin
  sorry
end

end find_even_decreasing_func_l189_189557


namespace additional_discount_percentage_l189_189010

def initial_price : ℝ := 2000
def gift_cards : ℝ := 200
def initial_discount_rate : ℝ := 0.15
def final_price : ℝ := 1330

theorem additional_discount_percentage :
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  additional_discount_percentage = 11.33 :=
by
  let discounted_price := initial_price * (1 - initial_discount_rate)
  let price_after_gift := discounted_price - gift_cards
  let additional_discount := price_after_gift - final_price
  let additional_discount_percentage := (additional_discount / price_after_gift) * 100
  show additional_discount_percentage = 11.33
  sorry

end additional_discount_percentage_l189_189010


namespace solve_for_a_l189_189671

-- Define the line equation and the condition of equal intercepts
def line_eq (a x y : ℝ) : Prop :=
  a * x + y - 2 - a = 0

def equal_intercepts (a : ℝ) : Prop :=
  (∀ x, line_eq a x 0 → x = 2 + a) ∧ (∀ y, line_eq a 0 y → y = 2 + a)

-- State the problem to prove the value of 'a'
theorem solve_for_a (a : ℝ) : equal_intercepts a → (a = -2 ∨ a = 1) :=
by
  sorry

end solve_for_a_l189_189671


namespace Cagney_and_Lacey_Cupcakes_l189_189199

-- Conditions
def CagneyRate := 1 / 25 -- cupcakes per second
def LaceyRate := 1 / 35 -- cupcakes per second
def TotalTimeInSeconds := 10 * 60 -- total time in seconds
def LaceyPrepTimeInSeconds := 1 * 60 -- Lacey's preparation time in seconds
def EffectiveWorkTimeInSeconds := TotalTimeInSeconds - LaceyPrepTimeInSeconds -- effective working time

-- Calculate combined rate
def CombinedRate := 1 / (1 / CagneyRate + 1 / LaceyRate) -- combined rate in cupcakes per second

-- Calculate the total number of cupcakes frosted
def TotalCupcakesFrosted := EffectiveWorkTimeInSeconds * CombinedRate -- total cupcakes frosted

-- We state the theorem that corresponds to our proof problem
theorem Cagney_and_Lacey_Cupcakes : TotalCupcakesFrosted = 37 := by
  sorry

end Cagney_and_Lacey_Cupcakes_l189_189199


namespace problem1_problem2_part1_problem2_part2_l189_189998

-- Problem 1
theorem problem1 (a b : ℝ) : 2 * a * b^2 + 4 * (a * b - 3 / 2 * a * b^2) - 4 * a * b = -4 * a * b^2 := by
  sorry

-- Problem 2
theorem problem2_part1 (m n : ℝ) : -2 * (m * n - 3 * m^2) - (m^2 - 5 * m * n + 2 * m * n) = 5 * m^2 + m * n := by
  sorry

theorem problem2_part2 : (-2 * (1 * 2 - 3 * 1^2) - (1^2 - 5 * 1 * 2 + 2 * 1 * 2)) = 7 := by
  have h : -2 * (1 * 2 - 3 * 1^2) - (1^2 - 5 * 1 * 2 + 2 * 1 * 2) = 5 * 1^2 + 1 * 2 := by
    apply problem2_part1
  rw h
  norm_num
  exact rfl

end problem1_problem2_part1_problem2_part2_l189_189998


namespace sum_a_n_first_n_terms_sum_b_n_first_n_terms_l189_189146

-- Definition of the sequence a_n = n + 3^(n-1)
def a (n : ℕ) : ℕ := n + 3^(n-1)

-- Definition of the sequence b_n = n * 3^(n-1)
def b (n : ℕ) : ℕ := n * 3^(n-1)

-- Sum of the first n terms of sequence a_n
def S (n : ℕ) : ℕ := ∑ k in Finset.range n, a (k+1)

-- Sum of the first n terms of sequence b_n
def T (n : ℕ) : ℕ := ∑ k in Finset.range n, b (k+1)

-- Proof statement for S_n
theorem sum_a_n_first_n_terms (n : ℕ) : S n = (3^n + n^2 + n - 1) / 2 := 
sorry

-- Proof statement for T_n
theorem sum_b_n_first_n_terms (n : ℕ) : T n = ((2 * n - 1) * 3^2 + 1) / 4 := 
sorry

end sum_a_n_first_n_terms_sum_b_n_first_n_terms_l189_189146


namespace solve_equation_l189_189027

theorem solve_equation {x : ℝ} (h : x ≠ -2) : (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) → x = 1 :=
by
  intro h_eq
  -- proof steps would go here
  sorry

end solve_equation_l189_189027


namespace minimum_value_and_attainment_intervals_where_f_is_decreasing_l189_189300

noncomputable def f (x: ℝ): ℝ := (Real.sin x)^2 + 2 * Real.sin x * Real.cos x + 3 * (Real.cos x)^2

theorem minimum_value_and_attainment :
  (∀ x : ℝ, f x ≥ 2 - Real.sqrt 2) ∧ 
  (∃ k : ℤ, ∀ x : ℝ, f x = 2 - Real.sqrt 2 ↔ x = -3*π/8 + k*π) :=
by sorry

theorem intervals_where_f_is_decreasing :
  ( ∀ k : ℤ, ∀ x : ℝ, x ∈ Set.Icc (π/8 + k*π) (5*π/8 + k*π) → ∀ x' : ℝ, x' > x → f x' < f x ) :=
by sorry

end minimum_value_and_attainment_intervals_where_f_is_decreasing_l189_189300


namespace child_support_owed_l189_189412

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l189_189412


namespace triangle_perimeter_l189_189173

-- Define the length of one leg (a) and the area of the right triangle
def a : ℝ := 30
def area : ℝ := 150

-- Define the length of the other leg (b)
def b : ℝ := 10

-- Define the length of the hypotenuse using the Pythagorean theorem
def c : ℝ := Real.sqrt (a^2 + b^2)

-- Define the perimeter of the triangle
def perimeter : ℝ := a + b + c

-- Prove that the perimeter is equal to 40 + 10 * Real.sqrt 10
theorem triangle_perimeter : perimeter = 40 + 10 * Real.sqrt 10 := 
by
  sorry

end triangle_perimeter_l189_189173


namespace josh_marbles_l189_189755

theorem josh_marbles (original_marble : ℝ) (given_marble : ℝ)
  (h1 : original_marble = 22.5) (h2 : given_marble = 20.75) :
  original_marble + given_marble = 43.25 := by
  sorry

end josh_marbles_l189_189755


namespace problem_l189_189844

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189844


namespace consecutive_integers_sum_l189_189892

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189892


namespace binomial_theorem_sum_abs_coeffs_l189_189779

theorem binomial_theorem_sum_abs_coeffs :
  let a := λ (n k : ℕ), (nat.choose n k) * (2^(n-k)) in
  (|a 6 1| + |a 6 2| + |a 6 3| + |a 6 4| + |a 6 5| + |a 6 6|) = 665 :=
by
  have step1: |nat.choose 6 1 * (2^(6-1))| = 192 := sorry
  have step2: |nat.choose 6 2 * (2^(6-2))| = 240 := sorry
  have step3: |nat.choose 6 3 * (2^(6-3))| = 160 := sorry
  have step4: |nat.choose 6 4 * (2^(6-4))| = 60  := sorry
  have step5: |nat.choose 6 5 * (2^(6-5))| = 12  := sorry
  have step6: |nat.choose 6 6 * (2^(6-6))| = 1   := sorry
  show (192 + 240 + 160 + 60 + 12 + 1) = 665, by sorry

end binomial_theorem_sum_abs_coeffs_l189_189779


namespace integral_value_complex_magnitude_l189_189521

-- Given the integral, show the value of the definite integral
theorem integral_value : (∫ x in -2 .. 1, |x^2 - 2|) = 1 :=
sorry

-- Given complex numbers z1 and z2, prove the magnitude condition
theorem complex_magnitude (a : ℝ) (hz1 : ℂ := a + 2 * complex.I) (hz2 : ℂ := 3 - 4 * complex.I) 
    (h_im : (hz1 / hz2).im = (hz1 / hz2)) : complex.abs hz1 = 10 / 3 :=
sorry

end integral_value_complex_magnitude_l189_189521


namespace problem1_problem2_problem3_l189_189510

-- Definitions
def p1 : Prop := 2 ∣ 4
def q1 : Prop := 2 ∣ 6
def p2 : Prop := ∀ (r : Type) [rectangle r], r.diagonals_equal
def q2 : Prop := ∀ (r : Type) [rectangle r], r.diagonals_bisect
def p3 : Prop := ∀ (r : real_roots), (r ∈ roots_of_eq (x^2 + x - 1 = 0)) → same_sign r
def q3 : Prop := ∀ (r : real_roots), (r ∈ roots_of_eq (x^2 + x - 1 = 0)) → equal_abs_values r

-- Proof goals
theorem problem1 : (p1 ∨ q1) = True ∧ (p1 ∧ q1) = True ∧ ¬p1 = False := by
  sorry

theorem problem2 : (p2 ∨ q2) = True ∧ (p2 ∧ q2) = True ∧ ¬p2 = False := by
  sorry

theorem problem3 : (p3 ∨ q3) = False ∧ (p3 ∧ q3) = False ∧ ¬p3 = True := by
  sorry

end problem1_problem2_problem3_l189_189510


namespace monotonicity_f_g_common_tangent_l189_189667

def f (m x : ℝ) : ℝ := m * Real.log (x + 1)
def g (x : ℝ) : ℝ := x / (x + 1)

def F (m x : ℝ) : ℝ := f m x - g x

theorem monotonicity_f_g (m : ℝ) :
  (∀ x > -1, (m ≤ 0 → (F m)' x < 0) ∧
   (m > 0  → (F m)' x < 0 ↔ x < -1 + 1/m) ∧
   (m > 0  → (F m)' x > 0 ↔ x > -1 + 1/m)) := sorry

theorem common_tangent (m : ℝ) :
  (∃ a b : ℝ, ∀ x > -1, 
   (deriv (f m) a = deriv (g) b) ∧ (f m a - deriv (f m) a * a = g b - deriv (g) b * b)) ↔ m = 1 := sorry

end monotonicity_f_g_common_tangent_l189_189667


namespace only_option_A_is_quadratic_l189_189122

def is_quadratic (expr : ℚ[X]) : Prop :=
  ∃ a b c : ℚ, a ≠ 0 ∧ expr = a * X^2 + b * X + c

def option_A := -5 * X^2 - X + 3
def option_B := (3 / X) + X^2 - 1
def option_C (a b c : ℚ) := a * X^2 + b * X + c
def option_D := 4 * X - 1

theorem only_option_A_is_quadratic :
  is_quadratic option_A ∧ 
  ¬ is_quadratic option_B ∧
  ∀ (a b c : ℚ), ¬ is_quadratic (option_C a b c) ∧
  ¬ is_quadratic option_D :=
by
  sorry

end only_option_A_is_quadratic_l189_189122


namespace janice_age_is_21_l189_189367

variables (current_year : Nat) (current_month : Nat)
variables (mark_birth_year : Nat) (mark_birth_month : Nat)
variables (graham_age_difference : Nat) (janice_age_ratio : Rational)

def mark_age : Nat := current_year - mark_birth_year
def graham_age : Nat := mark_age - graham_age_difference
def janice_age : Rational := graham_age * janice_age_ratio

theorem janice_age_is_21 (h1 : current_year = 2021)
  (h2 : current_month = 2)
  (h3 : mark_birth_year = 1976)
  (h4 : mark_birth_month = 1)
  (h5 : graham_age_difference = 3)
  (h6 : janice_age_ratio = 1/2) :
  janice_age = 21 := 
by
  sorry

end janice_age_is_21_l189_189367


namespace total_donation_l189_189207

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l189_189207


namespace sum_of_consecutive_integers_l189_189874

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189874


namespace arcsin_cos_area_l189_189227

noncomputable def integral_arcsin_cos : ℝ :=
  ∫ x in 0..4 * Real.pi, Real.arcsin (Real.cos x)

theorem arcsin_cos_area :
  integral_arcsin_cos = Real.pi ^ 3 :=
sorry

end arcsin_cos_area_l189_189227


namespace minimum_trees_chopped_l189_189237

def number_of_sharpenings (spend_axe resharpen_cost : ℕ) : ℕ := (spend_axe + resharpen_cost - 1) / resharpen_cost

def number_of_regrinds (spend_saw regrind_cost : ℕ) : ℕ := (spend_saw + regrind_cost - 1) / regrind_cost

theorem minimum_trees_chopped 
  (axe_trees_per_sharpen : ℕ) (saw_trees_per_regrind : ℕ)
  (spend_axe : ℕ) (resharpen_cost : ℕ)
  (spend_saw : ℕ) (regrind_cost : ℕ)
  (num_sharpenings := number_of_sharpenings spend_axe resharpen_cost) 
  (num_regrinds := number_of_regrinds spend_saw regrind_cost) 
  : axe_trees_per_sharpen = 25 → saw_trees_per_regrind = 20 → 
    resharpen_cost = 8 → regrind_cost = 10 → 
    spend_axe = 46 → spend_saw = 60 → 
    (num_sharpenings * axe_trees_per_sharpen) + (num_regrinds * saw_trees_per_regrind) = 270 := 
by
  intros h1 h2 h3 h4 h5 h6
  have h_num_sharpenings : num_sharpenings = 6, by sorry
  have h_num_regrinds : num_regrinds = 6, by sorry
  rw [h1, h2, h_num_sharpenings, h_num_regrinds]
  have h_axe : num_sharpenings * axe_trees_per_sharpen = 150, by sorry
  have h_saw : num_regrinds * saw_trees_per_regrind = 120, by sorry
  rw [h_axe, h_saw]
  exact rfl

end minimum_trees_chopped_l189_189237


namespace initial_average_weight_of_ABC_l189_189440

-- Conditions
variables (A B C D E : ℝ)
variables (W : ℝ)
hypothesis h1 : (E = D + 7)
hypothesis h2 : (3 * W + D = 320)
hypothesis h3 : (B + C + D + E = 316)

-- Goal: Prove that W = 110.33
theorem initial_average_weight_of_ABC : W = 110.33 :=
sorry

end initial_average_weight_of_ABC_l189_189440


namespace trapezoid_midsegment_length_l189_189736

/-- In trapezoid ABCD, where AD is parallel to BC,
    and angles B and C are 30 and 60 degrees respectively.
    Points E, M, F, and N are the midpoints of AB, BC, CD, and DA respectively.
    Given that BC = 7 and MN = 3, we aim to find the length of EF. -/
theorem trapezoid_midsegment_length 
  (A B C D E M F N : Point)
  (AD_parallel_BC : AD ∥ BC)
  (angle_B : ∠B = 30)
  (angle_C : ∠C = 60)
  (E_mid_AB : E = midpoint A B)
  (M_mid_BC : M = midpoint B C)
  (F_mid_CD : F = midpoint C D)
  (N_mid_DA : N = midpoint D A)
  (BC_length : BC = 7)
  (MN_length : MN = 3) :
  length EF = 4 := sorry

end trapezoid_midsegment_length_l189_189736


namespace exists_even_dance_count_l189_189533

theorem exists_even_dance_count (n : ℕ) (h_odd : Odd n) (d : Fin n → ℕ) 
  (h_even_sum : ∑ i, d i % 2 = 0) : ∃ i, d i % 2 = 0 := 
sorry

end exists_even_dance_count_l189_189533


namespace least_possible_integer_l189_189534

theorem least_possible_integer (N : ℕ) :
  ¬ (N % 23 = 0) → 
  ¬ (N % 24 = 0) → 
  (∀ k ∈ (finset.range 31).erase 23 24, k ∣ N) →
  N = 2230928700 :=
sorry

end least_possible_integer_l189_189534


namespace boxes_division_l189_189149

theorem boxes_division (total_eggs : ℚ) (eggs_per_box : ℚ) (number_of_boxes : ℚ) :
  total_eggs = 3 ∧ eggs_per_box = 1.5 -> number_of_boxes = 2 :=
begin
  intro h,
  cases h with ht hp,
  rw [ht, hp],
  norm_num,
end

end boxes_division_l189_189149


namespace inequality_on_a_l189_189659

variable {a b : ℝ}

def f (x : ℝ) : ℝ := Real.exp x - a * x ^ 2 - b * x - 1

theorem inequality_on_a (h1 : f 1 = 0) (h2 : ∃ x : ℝ, 0 < x ∧ x < 1 ∧ f x = 0) : Real.exp 1 - 2 < a ∧ a < 1 := by
  -- Proof would go here
  sorry

end inequality_on_a_l189_189659


namespace sum_of_squares_of_digits_l189_189069

theorem sum_of_squares_of_digits :
  let F := 1109700
  let p := 7
  let G := F + p
  let digits := Nat.digits 10 G
  let sum_of_squares := digits.map (λ d, d^2).sum
in sum_of_squares = 181 :=
by
  let F := 1109700
  let p := 7
  let G := F + p
  let digits := Nat.digits 10 G
  let sum_of_squares := digits.map (λ d, d^2).sum
  have digits_of_G : digits = [1, 1, 0, 9, 7, 0, 7] := by
    sorry -- Here, you would verify the digit list if needed.
  calc
    sum_of_squares = [1, 1, 0, 9, 7, 0, 7].map (λ d, d^2).sum := by sorry
    ... = 1 ^ 2 + 1 ^ 2 + 0 ^ 2 + 9 ^ 2  + 7 ^ 2 + 0 ^ 2 + 7 ^ 2 := by sorry
    ... = 1 + 1 + 0 + 81 + 49 + 0 + 49 := by sorry
    ... = 181 := by sorry

end sum_of_squares_of_digits_l189_189069


namespace consecutive_integers_sum_l189_189858

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189858


namespace david_english_marks_l189_189222

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end david_english_marks_l189_189222


namespace cowboy_hats_problem_l189_189481

def count_permutations (n : ℕ) : ℕ :=
  Nat.factorial n

def derangements (n : ℕ) : ℕ :=
  if n = 0 then 1 else
  if n = 1 then 0 else
  derangements (n - 1) * (n - 1) + derangements (n - 2) * (n - 1)

def derangement_probability (n : ℕ) :=
  (derangements n : ℚ) / count_permutations n

theorem cowboy_hats_problem : derangement_probability 3 = 1 / 3 :=
  by sorry

end cowboy_hats_problem_l189_189481


namespace polynomial_remainder_l189_189256

theorem polynomial_remainder :
  ∀ (x : ℂ), (x^1010 % (x^4 - 1)) = x^2 :=
sorry

end polynomial_remainder_l189_189256


namespace find_ab_and_monotonicity_l189_189660

noncomputable def f (x : ℝ) (a b : ℝ) := Real.exp x * (a * x + b) - x^2 - 4 * x

theorem find_ab_and_monotonicity (a b : ℝ)
  (h_fx : ∀ x, f x a b = Real.exp x * (a * x + b) - x^2 - 4 * x)
  (tangent_line_eq : ∀ f', deriv (f 0) = f')
  (h_tangent_line_eq : tangent_line_eq 4)
  (h_f0 : f 0 a b = 4)
  (h_df0 : deriv (f 0) = 4) :
  (a = 4 ∧ b = 4) ∧ 
  ((∀ x, x < -2 ∨ x > -Real.log 2 -> deriv (f x 4 4) > 0) ∧ 
   (∀ x, -2 < x ∧ x < -Real.log 2 -> deriv (f x 4 4) < 0)) :=
sorry

end find_ab_and_monotonicity_l189_189660


namespace sum_of_consecutive_integers_with_product_812_l189_189923

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189923


namespace real_z9_count_l189_189065

theorem real_z9_count (z : ℂ) (hz : z^18 = 1) : 
  (∃! z : ℂ, z^18 = 1 ∧ (z^9).im = 0) :=
sorry

end real_z9_count_l189_189065


namespace simplify_sqrt_fraction_l189_189616

theorem simplify_sqrt_fraction 
  {a b c : ℝ} 
  (h : sqrt (a - 5) + (b - 3) ^ 2 = sqrt (c - 4) + sqrt (4 - c)) : 
  (sqrt c) / (sqrt a - sqrt b) = sqrt 5 + sqrt 3 :=
by
  sorry

end simplify_sqrt_fraction_l189_189616


namespace determinant_value_l189_189649

theorem determinant_value (t₁ t₂ : ℤ)
    (h₁ : t₁ = 2 * 3 + 3 * 5)
    (h₂ : t₂ = 5) :
    Matrix.det ![
      ![1, -1, t₁],
      ![0, 1, -1],
      ![-1, t₂, -6]
    ] = 14 := by
  rw [h₁, h₂]
  -- Actual proof would go here
  sorry

end determinant_value_l189_189649


namespace min_value_f_not_exists_l189_189389

noncomputable def f (x : ℝ) : ℝ :=
  let a := log (sqrt (x^2 + 10) + x)
  let b := log (sqrt (x^2 + 10) - x)
  (1 + 1/a) * (1 + 2/b)

theorem min_value_f_not_exists : ¬ ∃ x, 0 < x ∧ x < 4.5 ∧ (
  ∀ y, 0 < y ∧ y < 4.5 → f(x) ≤ f(y)
) :=
begin
  sorry
end

end min_value_f_not_exists_l189_189389


namespace bisect_angle_AD_l189_189265

theorem bisect_angle_AD 
  (A B C D H E F : Type*)
  [AddGroup A] [AddGroup B] [AddGroup C]
  [AddGroup D] [AddGroup H]
  [AddGroup E] [AddGroup F]
  (h1 : D ∈ FindFoot A B C)
  (h2 : H ∈ LineThrough A D)
  (h3 : E ∈ LineIntersect (LineThrough B H) (LineThrough A C))
  (h4 : F ∈ LineIntersect (LineThrough C H) (LineThrough A B)) :
  BisectsAngle AD ED DF := sorry

end bisect_angle_AD_l189_189265


namespace total_pennies_l189_189211

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l189_189211


namespace rectangle_area_l189_189512

theorem rectangle_area (l w : ℝ) (h1 : 2 * l + 2 * w = 14) (h2 : l^2 + w^2 = 25) : l * w = 12 :=
by
  sorry

end rectangle_area_l189_189512


namespace minimum_quotient_is_10_point_75_l189_189252

open Nat

def minimum_quotient_condition := 
  ∃ a b c : ℕ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
               a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ 
               b = 2 * a ∧ 
               (120 * a + c) / (3 * a + c) = 10.75

theorem minimum_quotient_is_10_point_75 : minimum_quotient_condition :=
sorry

end minimum_quotient_is_10_point_75_l189_189252


namespace coefficient_of_x2_is_neg13_l189_189242

-- Define the two polynomials as functions from ℝ to ℝ
def poly1(x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 3 * x
def poly2(x : ℝ) : ℝ := 3 * x^2 - 4 * x - 5

-- Define a function to extract the coefficient of x^2 term in a given polynomial
def coefficient_x2 (p : ℝ → ℝ) : ℝ :=
  -- Coefficient for x^2 term is the partial derivative of p at x=0 divided by 2
  (D (D p) 0) / 2

-- Statement of the problem in Lean 4 form
theorem coefficient_of_x2_is_neg13 : coefficient_x2 (λ x, (poly1 x) * (poly2 x)) = -13 := 
by 
  sorry -- Proof of the theorem

end coefficient_of_x2_is_neg13_l189_189242


namespace sum_of_consecutive_integers_with_product_812_l189_189920

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189920


namespace mark_5_stones_odd_gaps_l189_189468

theorem mark_5_stones_odd_gaps : ∃ ways : ℕ, ways = 77 ∧ (
  ∃ S : finset ℕ, S.card = 5 ∧
  ∀ (a b : ℕ), a ∈ S → b ∈ S → a ≠ b → (a % 2 = b % 2) ∧ 1 ≤ a ∧ a ≤ 15 ∧
  ∃ S1 S2 : finset ℕ, S = S1 ∪ S2 ∧ S1.card = 5 ∧ S2.card = 0 ∧
  ∀ (a b : ℕ), a ∈ S1 → a < b → b ∈ S1 → (b - a) % 2 = 1
) := sorry

end mark_5_stones_odd_gaps_l189_189468


namespace correct_sequence_is_seqE_l189_189007

inductive ClockSequences
| seqA : ClockSequences
| seqB : ClockSequences
| seqC : ClockSequences
| seqD : ClockSequences
| seqE : ClockSequences

namespace ClockSequences

def validate : ClockSequences → bool 
| seqA := false
| seqB := false
| seqC := false
| seqD := false
| seqE := true

theorem correct_sequence_is_seqE : 
  validate seqE = true := 
by
  simp [validate]
  exact rfl

end ClockSequences

end correct_sequence_is_seqE_l189_189007


namespace modulus_complex_w_range_k_l189_189287

-- Prove the modulus of the complex number w = a + bi is 10
theorem modulus_complex_w (a b : ℝ) (z : ℂ) (hz : z^2 + 2 * z + 10 = 0) (Im_z_neg : z.im < 0) (cond : a / z + conj z = b * complex.I) : complex.abs (a + b * complex.I) = 10 := sorry

-- Prove the range of real values of k such that the inequality x^2 + kx - a ≥ 0 holds for all x ∈ [0, 5] is [-2√10, +∞)
theorem range_k (a : ℝ) (ha : a = -10) : {k : ℝ | ∀ x ∈ set.Icc (0:ℝ) 5, x^2 + k * x - a ≥ 0} = set.Ici (-(2 * real.sqrt 10)) := sorry

end modulus_complex_w_range_k_l189_189287


namespace line_tangent_and_not_in_fourth_l189_189450

-- Define the curve
def curve (x : ℝ) : ℝ := x^3

-- Define the point
def P : ℝ × ℝ := (1, 1)

-- Define the tangency condition
def tangent_at (l : ℝ → ℝ → Prop) (curve : ℝ → ℝ) (x0 : ℝ) : Prop :=
  l x0 (curve x0) ∧ ∃ m : ℝ, ∀ x : ℝ, x = x0 → l x (curve x) ∧ (curve' x0) = m

-- Define the condition that line does not pass through the fourth quadrant
def not_in_fourth_quadrant (l : ℝ → ℝ → Prop) : Prop :=
  ∀ x y : ℝ, l x y → ¬ (x > 0 ∧ y < 0)

def line_l (x y : ℝ) : Prop := 3 * x - 4 * y + 1 = 0

theorem line_tangent_and_not_in_fourth :
  tangent_at line_l curve x0 ∧
  (line_l 1 1) ∧
  not_in_fourth_quadrant line_l :=
sorry

end line_tangent_and_not_in_fourth_l189_189450


namespace find_height_from_B_to_BC_l189_189276

noncomputable def triangle_conditions 
  (b c : ℝ) (S : ℝ) :=
  b = 3 ∧ 
  c = 2 ∧ 
  S = (3 * Real.sqrt 3) / 2

noncomputable def angle_A 
  (b c S : ℝ) (A : ℝ) 
  (hA : triangle_conditions b c S) : Prop :=
  A = 120

noncomputable def height_from_B_to_BC 
  (b c S A : ℝ) (h : ℝ) 
  (hB : triangle_conditions b c S) 
  (hA : angle_A b c S A hB) : Prop :=
  h = (3 * Real.sqrt (57)) / 19

-- Statement without proof:
theorem find_height_from_B_to_BC 
  : ∃ (b c S A h : ℝ), 
      triangle_conditions b c S ∧ 
      angle_A b c S A (by assumption) ∧ 
      height_from_B_to_BC b c S A h (by assumption) (by assumption) := 
sorry

end find_height_from_B_to_BC_l189_189276


namespace powers_ratio_and_work_condition_l189_189361

noncomputable def pistons_power_ratio (initial_pressure : ℝ) (initial_volume : ℝ) (distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  let final_volume := initial_volume * ((distance - (speed * time)) / distance)
  let alpha := initial_volume / final_volume
  let final_pressure_nitrogen := alpha * initial_pressure
  let power_ratio := final_pressure_nitrogen / 1
  power_ratio

noncomputable def work_done (pressure: ℝ) (volume_change: ℝ) : ℝ :=
  pressure * volume_change

theorem powers_ratio_and_work_condition (initial_pressure : ℝ) (initial_volume : ℝ) (distance : ℝ) (speed : ℝ) (time : ℝ) (work_threshold : ℝ) (interval_time : ℝ) :
  (pistons_power_ratio initial_pressure initial_volume distance speed time = 2) ∧ 
  (work_done 1 (speed * (interval_time / 60) * initial_volume / distance) > work_threshold) :=
begin
  sorry
end

end powers_ratio_and_work_condition_l189_189361


namespace find_a_minus_b_l189_189324

theorem find_a_minus_b (a b : ℚ)
  (h1 : 2 = a + b / 2)
  (h2 : 7 = a - b / 2)
  : a - b = 19 / 2 := 
  sorry

end find_a_minus_b_l189_189324


namespace roof_length_width_diff_l189_189950

variable (w l : ℝ)
variable (h1 : l = 4 * w)
variable (h2 : l * w = 676)

theorem roof_length_width_diff :
  l - w = 39 :=
by
  sorry

end roof_length_width_diff_l189_189950


namespace bijection_and_inverse_l189_189377

def S_0 : Set ℂ := { z | complex.abs z = 1 ∧ z ≠ -1 }
def f (z : ℂ) : ℝ := complex.im z / (1 + complex.re z)
noncomputable def f_inv (y : ℝ) : ℂ := (1 - y^2) / (1 + y^2) + complex.I * (2 * y / (1 + y^2))

theorem bijection_and_inverse :
  (Bijective f) ∧ (∀ y : ℝ, f_inv y ∈ S_0 ∧ f (f_inv y) = y) := by
  sorry

end bijection_and_inverse_l189_189377


namespace consecutive_integer_sum_l189_189864

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189864


namespace rotation_270_of_4_minus_2i_is_2_minus_4i_l189_189191

-- Define the initial complex number
def initial_complex : ℂ := 4 - 2 * complex.i

-- Define the 270 degree counter-clockwise rotation as multiplication by -i
def rotation_270 (z : ℂ) : ℂ := -complex.i * z

-- State the theorem
theorem rotation_270_of_4_minus_2i_is_2_minus_4i : rotation_270 initial_complex = 2 - 4 * complex.i :=
  by
    sorry

end rotation_270_of_4_minus_2i_is_2_minus_4i_l189_189191


namespace largest_not_sum_of_36_and_composite_l189_189110

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l189_189110


namespace nhai_hiring_l189_189003

def man_hours (men : ℕ) (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  men * hours_per_day * days

theorem nhai_hiring :
  ∃ (new_employees : ℕ),
  let initial_man_hours := man_hours 100 8 50 in
  let completed_man_hours := man_hours 100 8 25 in
  let total_man_hours := 3 * completed_man_hours in
  let remaining_man_hours := total_man_hours - completed_man_hours in
  let working_hours_per_day := 10 in
  let remaining_days := 25 in
  let required_men := remaining_man_hours / (working_hours_per_day * remaining_days) in
  new_employees = required_men - 100 ∧
  new_employees = 60 :=
begin
  sorry
end

end nhai_hiring_l189_189003


namespace totalLightPathLengthInCube_l189_189761

noncomputable def lightPathLengthInCube (A B C G P Q: ℝ × ℝ × ℝ) :=
  let side_length : ℝ := 15
  let distance (v w : ℝ × ℝ × ℝ) : ℝ := real.sqrt ((v.1 - w.1) ^ 2 + (v.2 - w.2) ^ 2 + (v.3 - w.3) ^ 2)
  distance A P + distance P Q + distance Q A

theorem totalLightPathLengthInCube :
  let A : ℝ × ℝ × ℝ := (0, 0, 0)
  let B : ℝ × ℝ × ℝ := (15, 0, 0)
  let C : ℝ × ℝ × ℝ := (15, 15, 0)
  let G : ℝ × ℝ × ℝ := (15, 15, 15)
  let P : ℝ × ℝ × ℝ := (15, 9, 8)
  let Q : ℝ × ℝ × ℝ := (0, 9, 8)
  lightPathLengthInCube A B C G P Q = real.sqrt 370 + 15 + real.sqrt 145
:= by
  sorry

end totalLightPathLengthInCube_l189_189761


namespace correct_statements_l189_189663

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem correct_statements :
  (∃ c : ℝ × ℝ, c = (-1, 1) ∧ ∀ x, f(x) = 1 - 1 / (x + 1)) ∧
  (∀ x : ℝ, x > -1 → f(x + 1) > f(x)) ∧
  (∑ i in (finset.range 2022), f(i + 1) + ∑ i in (finset.range 2022), f(1 / (i + 1)) = 2021) ∧
  (f 1 + 2021 = 2021.5)
:= by
  sorry

end correct_statements_l189_189663


namespace rearranged_power_of_two_is_impossible_l189_189633

theorem rearranged_power_of_two_is_impossible (M N : ℕ) (a b : ℕ) 
  (hM : M = 2^a) (hN : N = 2^b) (ha_gt_hb : a > b) (h_rearrange : N ≠ M ∧ digit_list N = digit_list M) : 
  false :=
by sorry

end rearranged_power_of_two_is_impossible_l189_189633


namespace largest_not_sum_of_36_and_composite_l189_189115

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l189_189115


namespace consecutive_integer_sum_l189_189869

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189869


namespace value_of_d_to_nearest_tenth_l189_189501

-- Define the conditions
def d : ℝ := (0.889 * 55) / 9.97

-- The theorem statement proving the question equals the answer given the conditions
theorem value_of_d_to_nearest_tenth : (Real.round (d * 10) / 10) = 4.9 := by
  unfold d
  sorry

end value_of_d_to_nearest_tenth_l189_189501


namespace fraction_irreducible_l189_189426
-- Import necessary libraries

-- Define the problem to prove
theorem fraction_irreducible (n: ℕ) (h: n > 0) : gcd (21 * n + 4) (14 * n + 3) = 1 := 
  sorry

end fraction_irreducible_l189_189426


namespace sum_divisible_by_10_l189_189014

theorem sum_divisible_by_10 :
    (111 ^ 111 + 112 ^ 112 + 113 ^ 113) % 10 = 0 :=
by
  sorry

end sum_divisible_by_10_l189_189014


namespace log_bounds_l189_189064

theorem log_bounds
  (a b : ℤ)
  (log_10_548834: ℝ)
  (H1: 548834 > 10^a)
  (H2: 548834 < 10^b)
  (H3: log_10_548834 = Real.log 548834 / Real.log 10)
  (H4: a + 1 = b):
  a + b = 11 := by
begin
  have H5 : 100000 < 548834 := by sorry,
  have H6 : 548834 < 1000000 := by sorry,
  have H7 : log_10_548834 = Real.log 548834 / Real.log 10 := by sorry,
  have H8 : 5 + 1 = 6 := by sorry,
  sorry
end

end log_bounds_l189_189064


namespace consecutive_integers_sum_l189_189887

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189887


namespace consecutive_integers_sum_l189_189891

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189891


namespace find_irreducible_numerator_l189_189445

theorem find_irreducible_numerator :
  let frac_diff := (2024 : ℚ) / 2023 - (2023 / 2024) in
  ∃ p q, frac_diff = p / q ∧ Nat.gcd p q = 1 ∧ p = 4047 :=
by
  sorry

end find_irreducible_numerator_l189_189445


namespace first_player_wins_if_not_power_of_two_l189_189066

/-- 
  Prove that the first player can guarantee a win if and only if $n$ is not a power of two, under the given conditions. 
-/
theorem first_player_wins_if_not_power_of_two
  (n : ℕ) (h : n > 1) :
  (∃ k : ℕ, n = 2^k) ↔ false :=
sorry

end first_player_wins_if_not_power_of_two_l189_189066


namespace total_pennies_donated_l189_189574

theorem total_pennies_donated:
  let cassandra_pennies := 5000
  let james_pennies := cassandra_pennies - 276
  let stephanie_pennies := 2 * james_pennies
  cassandra_pennies + james_pennies + stephanie_pennies = 19172 :=
by
  sorry

end total_pennies_donated_l189_189574


namespace problem_l189_189290

def f : ℝ → ℝ
| x => if x > 0 then log (1/2) x else if x < 0 then log (1/2) (-x) else 0

theorem problem (x : ℝ) (h : -sqrt 5 < x ∧ x < sqrt 5) : f (x^2 - 1) > -2 := by
  sorry

end problem_l189_189290


namespace tom_pays_l189_189483

def cost_of_lemons : ℝ := 8 * 2
def cost_of_papayas : ℝ := 6 * 1
def cost_of_mangos : ℝ := 5 * 4
def cost_of_oranges : ℝ := 3 * 3
def cost_of_apples : ℝ := 8 * 1.5
def cost_of_pineapples : ℝ := 2 * 5

def total_cost_before_discounts : ℝ := 
  cost_of_lemons + cost_of_papayas + cost_of_mangos + cost_of_oranges + cost_of_apples + cost_of_pineapples

def discount_for_every_4_fruits : ℝ :=
  (8 + 6 + 5 + 3 + 8 + 2) / 4

def discount_for_two_types : ℝ := 2

def discount_for_more_than_6_lemons : ℝ := 8 * 0.5
def discount_for_more_than_6_apples : ℝ := 8 * 0.5

def total_discount : ℝ :=
  discount_for_every_4_fruits + discount_for_two_types + discount_for_more_than_6_lemons + discount_for_more_than_6_apples

theorem tom_pays : total_cost_before_discounts - total_discount = 55 := by
  sorry

end tom_pays_l189_189483


namespace f_at_8_5_l189_189824

def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom odd_function_shifted : ∀ x : ℝ, f (x - 1) = -f (1 - x)
axiom f_half : f 0.5 = 9

theorem f_at_8_5 : f 8.5 = 9 := by
  sorry

end f_at_8_5_l189_189824


namespace consecutive_integers_sum_l189_189886

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189886


namespace max_n_given_condition_l189_189783

def sum_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

theorem max_n_given_condition : ∃ n : ℕ, (n - sum_digits n = 2007 ∧ n = 2019) :=
sorry

end max_n_given_condition_l189_189783


namespace intersecting_curves_l189_189506

variables {x y a : ℝ}

def circle (a : ℝ) (x y : ℝ) := x^2 + y^2 = 4 * a^2
def parabola (a : ℝ) (x y : ℝ) := y = x^2 - 2 * a
def intersect_at_four_points (a : ℝ) : Prop :=
  let x_expr := x^2 * (x^2 - (4 * a - 1)) = 0 in
  (4 * a - 1 > 0) ∧  -- Ensure real and distinct roots for x
  let y_0 := -2 * a in
  let y_1 := 2 * a - 1 in
  (a > 0) ∧
  distinct_points [(0, y_0), (sqrt (4 * a - 1), y_1), (-sqrt (4 * a - 1), y_1)]  -- Ensure distinct points

def distinct_points (points : list (ℝ × ℝ)) : Prop :=
  ∀ (p1 p2 : ℝ × ℝ), p1 ≠ p2 → p1 ∈ points → p2 ∈ points → p1 ≠ p2

theorem intersecting_curves:
  ∀ a : ℝ, intersect_at_four_points a ↔ a > 1/4 := sorry

end intersecting_curves_l189_189506


namespace value_of_a_minus_b_l189_189693

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 6) (h2 : |b| = 2) (h3 : a + b > 0) :
  a - b = 4 ∨ a - b = 8 :=
  sorry

end value_of_a_minus_b_l189_189693


namespace scientific_notation_140000000_l189_189815

theorem scientific_notation_140000000 :
  140000000 = 1.4 * 10^8 := 
sorry

end scientific_notation_140000000_l189_189815


namespace part1_find_m_and_intervals_of_monotonic_increase_part2_range_of_a_minus_c_over_2_l189_189646

noncomputable def f (x : ℝ) (m ω : ℝ) : ℝ := m * sin (ω * x) - cos (ω * x)

theorem part1_find_m_and_intervals_of_monotonic_increase (x_0 : ℝ) (m ω : ℝ) 
  (h1 : x_0 = π / 3) 
  (h2 : ∀ x : ℝ, f (x + π / ω) m ω = f x m ω)
  (h3 : m > 0) : 
  m = sqrt 3 ∧ 
  ∃ k : ℤ, ∃ x : ℝ, x ∈ Icc (k*π - π / 6) (k*π + π / 3) :=
sorry

theorem part2_range_of_a_minus_c_over_2 (A B C a b c : ℝ) (f : ℝ)
  (h1 : ∀ (A B C : ℝ), A + B + C = π)
  (h2 : f = 2 * sin (2 * B - π / 6))
  (h3 : b = sqrt 3) 
  (h4 : 0 < B ∧ B < π) 
  (h5 : sin B = 1 / 2) : 
  ∃ r : ℝ, r = a - c / 2 ∧ r ∈ Ioo (-sqrt 3 / 2) sqrt 3 :=
sorry

end part1_find_m_and_intervals_of_monotonic_increase_part2_range_of_a_minus_c_over_2_l189_189646


namespace lagrange_four_squares_l189_189015

theorem lagrange_four_squares (n : ℕ) : ∃ a b c d : ℤ, a^2 + b^2 + c^2 + d^2 = n :=
by
  -- Conditions
  -- Any odd prime number can be expressed as the sum of four squares of integers.
  have h1 : ∀ p : ℕ, nat.prime p ∧ p % 2 = 1 → ∃ a b c d : ℤ, a^2 + b^2 + c^2 + d^2 = p := sorry,
  -- The number 2 can be expressed as the sum of four squares of integers.
  have h2 : ∃ a b c d : ℤ, a^2 + b^2 + c^2 + d^2 = 2 := by
    use 1, 1, 0, 0
    norm_num,
  -- The set of numbers representable as the sum of four squares is closed under multiplication.
  have h3 : ∀ a b : ℕ, (∃ a1 b1 c1 d1 : ℤ, a1^2 + b1^2 + c1^2 + d1^2 = a) → (∃ a2 b2 c2 d2 : ℤ, a2^2 + b2^2 + c2^2 + d2^2 = b) → (∃ a3 b3 c3 d3 : ℤ, a3^2 + b3^2 + c3^2 + d3^2 = a * b) := sorry,
  -- Proof of the main theorem
  sorry

end lagrange_four_squares_l189_189015


namespace mass_percentage_H_BaOH2_is_1_176_l189_189606

-- Define the molar masses as constants
def molar_mass_Ba : ℝ := 137.327
def molar_mass_O : ℝ := 15.999
def molar_mass_H : ℝ := 1.008

-- Define the molar mass of Barium hydroxide Ba(OH)₂
def molar_mass_BaOH2 : ℝ :=
  molar_mass_Ba + 2 * molar_mass_O + 2 * molar_mass_H

-- Define the total mass of hydrogen in Barium hydroxide Ba(OH)₂
def total_mass_H_in_BaOH2 : ℝ :=
  2 * molar_mass_H

-- Define the mass percentage of hydrogen in Barium hydroxide Ba(OH)₂
def mass_percentage_H_in_BaOH2 : ℝ :=
  (total_mass_H_in_BaOH2 / molar_mass_BaOH2) * 100

-- Prove that mass percentage of H in Ba(OH)₂ is approximately 1.176%
theorem mass_percentage_H_BaOH2_is_1_176 :
  abs (mass_percentage_H_in_BaOH2 - 1.176) < 0.01 := sorry

end mass_percentage_H_BaOH2_is_1_176_l189_189606


namespace certain_number_modulo_l189_189120

theorem certain_number_modulo (x : ℕ) : (57 * x) % 8 = 7 ↔ x = 1 := by
  sorry

end certain_number_modulo_l189_189120


namespace inequality_proof_l189_189668

open Real

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (sqrt (a^2 + 8 * b * c) / a + sqrt (b^2 + 8 * a * c) / b + sqrt (c^2 + 8 * a * b) / c) ≥ 9 :=
by 
  sorry

end inequality_proof_l189_189668


namespace units_digit_of_2_pow_2023_l189_189500

def units_digit (n : ℕ) : ℕ := n % 10

def cycle_units_digits : List ℕ := [2, 4, 8, 6]

theorem units_digit_of_2_pow_2023 : units_digit (2 ^ 2023) = 8 := by
  have cycle : ∀ k : ℕ, units_digit (2 ^ (k + 4)) = units_digit (2 ^ k) :=
    by
      intro k
      let cycle := [2, 4, 8, 6]
      norm_num
      sorry
  have mod_cycle : 2023 % 4 = 3 := by norm_num
  have units2_3 : units_digit (2 ^ 3) = 8 := by norm_num
  rw [←mod_cycle, cycle]
  exact units2_3
  sorry

end units_digit_of_2_pow_2023_l189_189500


namespace coefficient_of_x2_is_neg13_l189_189241

-- Define the two polynomials as functions from ℝ to ℝ
def poly1(x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 - 3 * x
def poly2(x : ℝ) : ℝ := 3 * x^2 - 4 * x - 5

-- Define a function to extract the coefficient of x^2 term in a given polynomial
def coefficient_x2 (p : ℝ → ℝ) : ℝ :=
  -- Coefficient for x^2 term is the partial derivative of p at x=0 divided by 2
  (D (D p) 0) / 2

-- Statement of the problem in Lean 4 form
theorem coefficient_of_x2_is_neg13 : coefficient_x2 (λ x, (poly1 x) * (poly2 x)) = -13 := 
by 
  sorry -- Proof of the theorem

end coefficient_of_x2_is_neg13_l189_189241


namespace expression_evaluation_l189_189826

theorem expression_evaluation : (6 * 111) - (2 * 111) = 444 :=
by
  sorry

end expression_evaluation_l189_189826


namespace min_value_of_x2_plus_y2_l189_189266

-- Define the problem statement
theorem min_value_of_x2_plus_y2 (x y : ℝ) (h : 3 * x + y = 10) : x^2 + y^2 ≥ 10 :=
sorry

end min_value_of_x2_plus_y2_l189_189266


namespace smallest_possible_sum_l189_189465

theorem smallest_possible_sum (a b c d : ℕ) (h : {a, b, c, d} = {3, 4, 5, 6}) :
  ab + bc + cd + da ≥ 81 :=
by
  sorry

end smallest_possible_sum_l189_189465


namespace sum_of_consecutive_integers_with_product_812_l189_189919

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189919


namespace vector_sum_is_correct_l189_189682

-- Definitions for vectors a and b
def vector_a := (1, -2)
def vector_b (m : ℝ) := (2, m)

-- Condition for parallel vectors a and b
def parallel_vectors (m : ℝ) : Prop :=
  1 * m - (-2) * 2 = 0

-- Defining the target calculation for given m
def calculate_sum (m : ℝ) : ℝ × ℝ :=
  let a := vector_a
  let b := vector_b m
  (3 * a.1 + 2 * b.1, 3 * a.2 + 2 * b.2)

-- Statement of the theorem to be proved
theorem vector_sum_is_correct (m : ℝ) (h : parallel_vectors m) : calculate_sum m = (7, -14) :=
by sorry

end vector_sum_is_correct_l189_189682


namespace find_number_l189_189999

theorem find_number (x : ℝ) (h : 0.9 * x = 0.0063) : x = 0.007 := 
by {
  sorry
}

end find_number_l189_189999


namespace length_of_CD_l189_189995

variable (A B C D E : Type) [Triangle A B C] 
variables (AB AC BC : ℝ)
variables (P : Predicates)

-- Edge lengths of triangle ABC
def segment_AB : ℝ := 55
def segment_AC : ℝ := 35
def segment_BC : ℝ := 72

-- Lengths involving points D and E
def segment_CD (x : ℝ) : ℝ := x
def segment_CE (x : ℝ) : ℝ := 81 - x

-- Condition on perimeters
def perimeter_condition (x : ℝ) : Prop := 81 - x = 81 - x

-- Condition on area 
def area_condition (x : ℝ) : Prop := x * (81 - x) = 1260

-- Final proof statement
theorem length_of_CD : ∃ x : ℝ, perimeter_condition x ∧ area_condition x ∧ (x = 60) :=
by
  sorry

end length_of_CD_l189_189995


namespace range_of_a_tangent_lines_inequality_l189_189707

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a*Real.log x - (1/2)*x^2 + a + (1/2)

-- Hypothesize the existence of zeros x1 and x2
variable (a : ℝ) (x1 x2 : ℝ)
variable (h1 : 0 < x1)
variable (h2 : 0 < x2)
variable (h3 : x1 < x2)
variable (hx1 : f a x1 = 0)
variable (hx2 : f a x2 = 0)

-- Define the tangent points intersection x3
noncomputable def x3 (a : ℝ) (x1 x2 : ℝ) : ℝ :=
  let slope1 := (a / x1 - x1)
  let slope2 := (a / x2 - x2)
  let l3 := (slope2 - slope1) / (x2 - x1)
  - (l3 * (x1 - x2)) / (l3 - slope1)

-- Open namespace Real
open Real

-- First part: proving the range of a
theorem range_of_a : ∀ a : ℝ, 
  (0 < a) ↔ (∃ (x1 : ℝ) (x2 : ℝ), x1 < x2 ∧ f a x1 = 0 ∧ f a x2 = 0) :=
sorry

-- Second part: proving the inequality
theorem tangent_lines_inequality (a : ℝ)
  (hx1 : f a x1 = 0)
  (hx2 : f a x2 = 0)
  (h3 : x1 < x2)
  (x3 := x3 a x1 x2) :
   2 * x3 < x1 + x2 :=
sorry

end range_of_a_tangent_lines_inequality_l189_189707


namespace integer_sided_isosceles_obtuse_triangles_l189_189253

theorem integer_sided_isosceles_obtuse_triangles (a c : ℕ) :
  (2 * a + c = 2008) → 
  (c ^ 2 > 2 * a ^ 2) → 
  (2 * a > c) →
  (502 < a ∧ a < 1177) →
  (674 : ℕ) :=
begin
  sorry
end

end integer_sided_isosceles_obtuse_triangles_l189_189253


namespace other_root_of_quadratic_l189_189674

theorem other_root_of_quadratic (p q r : ℝ)
  (h : eval 1 (λ x, p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)) = 0) :
  ∃ x, x ≠ 1 ∧ eval x (λ x, p * (q - r) * x^2 + q * (r - p) * x + r * (p - q)) = 0 :=
begin
  use (r * (p - q)) / (p * (q - r)),
  split,
  { intro h_eq,
    rw h_eq at h,
    have := h,
    simp [mul_comm] at this,
    linarith, },
  { simp [eval, mul_comm, mul_assoc] }
end

end other_root_of_quadratic_l189_189674


namespace sum_of_interior_angles_of_pentagon_l189_189956

theorem sum_of_interior_angles_of_pentagon : 
  ∑ (interior_angles : ℕ) (n : ℕ) (h : n = 5), interior_angles = (n - 2) * 180 := 
  sorry

end sum_of_interior_angles_of_pentagon_l189_189956


namespace find_b_l189_189466

theorem find_b (a b : ℝ) (k : ℝ) (h1 : a * b = k) (h2 : a + b = 40) (h3 : a - 2 * b = 10) (ha : a = 4) : b = 75 :=
  sorry

end find_b_l189_189466


namespace total_donation_l189_189206

-- Definitions
def cassandra_pennies : ℕ := 5000
def james_deficit : ℕ := 276
def james_pennies : ℕ := cassandra_pennies - james_deficit

-- Theorem to prove the total donation
theorem total_donation : cassandra_pennies + james_pennies = 9724 :=
by
  -- Proof is omitted
  sorry

end total_donation_l189_189206


namespace ratio_of_amount_lost_l189_189368

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

end ratio_of_amount_lost_l189_189368


namespace sum_of_consecutive_integers_l189_189876

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189876


namespace projection_locus_circle_l189_189518

theorem projection_locus_circle (O : ℝ × ℝ) (k : ℝ) (OA OB : ℝ)
  (h : 1 / OA^2 + 1 / OB^2 = 1 / k^2) :
  ∃ M : ℝ × ℝ, dist O M = k ∧ ∀ (A B : ℝ × ℝ), 
  dist O A = OA → dist O B = OB → collinear {O, A, B} → 
  is_orthogonal (O - A) (B - A) → dist O (orthogonal_projection O A B) = k :=
sorry

end projection_locus_circle_l189_189518


namespace fencing_cost_per_meter_l189_189544

-- Definitions based on given conditions
def area : ℚ := 1200
def short_side : ℚ := 30
def total_cost : ℚ := 1800

-- Definition to represent the length of the long side
def long_side := area / short_side

-- Definition to represent the diagonal of the rectangle
def diagonal := (long_side^2 + short_side^2).sqrt

-- Definition to represent the total length of the fence
def total_length := long_side + short_side + diagonal

-- Definition to represent the cost per meter
def cost_per_meter := total_cost / total_length

-- Theorem statement asserting that cost_per_meter == 15
theorem fencing_cost_per_meter : cost_per_meter = 15 := 
by 
  sorry

end fencing_cost_per_meter_l189_189544


namespace double_factorial_identity_l189_189798

theorem double_factorial_identity (n : ℕ) : 
  ((2 * n)!.toRational / n!.toRational) = 2^n * (double_factorial (2 * n - 1)) :=
by
  sorry

end double_factorial_identity_l189_189798


namespace no_integer_solutions_to_equation_l189_189599

theorem no_integer_solutions_to_equation (x y k : ℤ) : ¬(x ^ 2009 + y ^ 2009 = 7 ^ k) :=
sorry

end no_integer_solutions_to_equation_l189_189599


namespace operation_eval_l189_189781

def my_operation (a b : ℤ) := a * (b + 2) + a * (b + 1)

theorem operation_eval : my_operation 3 (-1) = 3 := by
  sorry

end operation_eval_l189_189781


namespace arithmetic_sequence_ratio_l189_189339

theorem arithmetic_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (a1 d : ℝ)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) (h2 : ∀ n, S n = n * (2 * a1 + (n - 1) * d) / 2)
  (h_nonzero: ∀ n, a n ≠ 0):
  (S 5) / (a 3) = 5 :=
by
  sorry

end arithmetic_sequence_ratio_l189_189339


namespace range_of_a_l189_189583

variable {a x : ℝ}

def proposition_p (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 + a * x + 1 > 0

def proposition_q (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 1 → (4 * x^2 - a * x) ≥ (4 * (x + 1)^2 - a * (x + 1))

theorem range_of_a (h : ¬ proposition_p a ∧ (proposition_p a ∨ proposition_q a)) : a ≤ 0 ∨ 4 ≤ a ∧ a ≤ 8 :=
sorry

end range_of_a_l189_189583


namespace max_f_theta_min_f_theta_integral_f_theta_l189_189584

-- Define the function f(θ)
noncomputable def f (θ : ℝ) : ℝ := ∫ x in 0..1, abs (sqrt (1 - x^2) - sin θ)

-- Prove the maximum value of f(θ) on the interval [0, π/2] is 1
theorem max_f_theta : ∀ θ ∈ Icc (0 : ℝ) (π / 2), f θ ≤ 1 := by
  sorry

-- Prove the minimum value of f(θ) on the interval [0, π/2] is √2 - 1
theorem min_f_theta : ∀ θ ∈ Icc (0 : ℝ) (π / 2), √2 - 1 ≤ f θ := by
  sorry

-- Prove the integral of f(θ) on the interval [0, π/2] is 4 - π/2
theorem integral_f_theta : ∫ θ in 0..(π / 2), f θ = 4 - π / 2 := by
  sorry

end max_f_theta_min_f_theta_integral_f_theta_l189_189584


namespace sum_of_consecutive_integers_l189_189873

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189873


namespace find_second_number_l189_189537

-- Define the given number
def given_number := 220070

-- Define the constants in the problem
def constant_555 := 555
def remainder := 70

-- Define the second number (our unknown)
variable (x : ℕ)

-- Define the condition as an equation
def condition : Prop :=
  given_number = (constant_555 + x) * 2 * (x - constant_555) + remainder

-- The theorem to prove that the second number is 343
theorem find_second_number : ∃ x : ℕ, condition x ∧ x = 343 :=
sorry

end find_second_number_l189_189537


namespace find_ax_6_by_6_l189_189320

variables (a b x y : ℝ)
variables (s_1 s_2 s_3 s_4 : ℝ)

# assumption, only those already identified in conditions in part (a)
hypothesis h1 : a * x + b * y = 5
hypothesis h2 : a * x^2 + b * y^2 = 11
hypothesis h3 : a * x^3 + b * y^3 = 26
hypothesis h4 : a * x^4 + b * y^4 = 58

theorem find_ax_6_by_6 : a * x^6 + b * y^6 = -220 :=
sorry

end find_ax_6_by_6_l189_189320


namespace sum_of_consecutive_integers_l189_189877

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189877


namespace min_distance_between_line_and_circle_l189_189517

noncomputable def point_on_line : set (ℝ × ℝ) :=
{ p | ∃ x y, 3 * x - 4 * y + 34 = 0 ∧ p = (x, y) }

noncomputable def point_on_circle : set (ℝ × ℝ) :=
{ p | ∃ x y, x^2 + y^2 - 8 * x + 2 * y - 8 = 0 ∧ p = (x, y) }

theorem min_distance_between_line_and_circle :
  ∀ M ∈ point_on_line, ∀ N ∈ point_on_circle, dist M N >= 5 :=
begin
  sorry
end

end min_distance_between_line_and_circle_l189_189517


namespace poly_remainder_mod_l189_189977

open Polynomial

noncomputable def poly_remainder (x : ℂ[X]) : ℂ :=
  (x - 1)^2015 %ₘ (x^2 - x + 1)

theorem poly_remainder_mod (x : ℂ[X]) :
  (x - 1)^2015 %ₘ (x^2 - x + 1) = -1 := by
  sorry

end poly_remainder_mod_l189_189977


namespace line_in_plane_parallel_to_other_plane_l189_189016

variables {Point Line Plane : Type} [EuclideanSpace Point Line Plane]

theorem line_in_plane_parallel_to_other_plane 
  (α β : Plane) (a : Line) 
  (h1 : α ∥ β) 
  (h2 : a ∈ α) : 
  a ∥ β :=
sorry

end line_in_plane_parallel_to_other_plane_l189_189016


namespace max_degree_q_for_horizontal_asymptote_l189_189228

theorem max_degree_q_for_horizontal_asymptote :
  ∀ (q p : Polynomial ℝ), p = 2 * Polynomial.X ^ 5 - 3 * Polynomial.X ^ 4 + 2 * Polynomial.X ^ 3 - 7 * Polynomial.X ^ 2 + Polynomial.X + 1 →
  degree p = 5 →
  ( degree q ≤ 5 ↔ ∃ c : ℝ, ∃ r : Polynomial ℝ, q = c * Polynomial.X^5 + r ∧ degree r < 5) :=
by
  sorry

end max_degree_q_for_horizontal_asymptote_l189_189228


namespace total_worth_of_presents_l189_189752

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l189_189752


namespace range_of_a_l189_189328

noncomputable def f (x : ℝ) : ℝ := 2^x - 5
noncomputable def g (x : ℝ) : ℝ := 4*x - x^2

theorem range_of_a (a : ℝ) (h₁ : 0 < a) (h₂ : ∀ x ∈ set.Icc a (a+2), f x ≥ g x) : 
  3 ≤ a :=
sorry

end range_of_a_l189_189328


namespace problem_l189_189847

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189847


namespace solution_set_l189_189632

theorem solution_set (f : ℝ → ℝ) (hf1 : f 1 = 1) (hf_deriv : ∀ x, fderiv ℝ f x < 1/2) :
  {x : ℝ | f x < x / 2 + 1 / 2} = {x : ℝ | x > 1} :=
by
  sorry

end solution_set_l189_189632


namespace three_g_two_plus_two_g_neg_four_l189_189218

def g (x : ℝ) : ℝ := 2 * x ^ 2 - 2 * x + 11

theorem three_g_two_plus_two_g_neg_four : 3 * g 2 + 2 * g (-4) = 147 := by
  sorry

end three_g_two_plus_two_g_neg_four_l189_189218


namespace smaller_circle_radius_l189_189733

theorem smaller_circle_radius
  (R : ℝ) (r : ℝ) 
  (h1 : R = 10) 
  (h2 : 2 * r + 2 * r = 2 * R) : r = 5 :=
by
  rw [h1] at h2
  linarith

# The theorem smaller_circle_radius states that given the radius of the larger circle R = 10 
# and the relationship 2 * r + 2 * r = 2 * R, it follows that the radius of the smaller circle r = 5.

end smaller_circle_radius_l189_189733


namespace circumcircle_I_if_and_only_if_circumcircle_K_l189_189681

variables {A B C D E F I J K : Type} [geometry ABCD] 
  (is_trapezoid : is_trapezoid ABCD)
  (side_AB_parallel_CD : parallel AB CD)
  (E_on_BC_outside : on_line E BC ∧ ¬(segment_in E BC))
  (F_on_AD_inside : on_line F AD ∧ segment_in F AD)
  (angle_DAE_eq_angle_CBF : ∠ DAE = ∠ CBF)
  (intersection_I : I = intersection CD EF)
  (intersection_J : J = intersection AB EF)
  (midpoint_K : K = midpoint EF)
  (K_not_on_AB : ¬on_line K AB)

theorem circumcircle_I_if_and_only_if_circumcircle_K :
  (on_circumcircle I (triangle AB K)) ↔ (on_circumcircle K (triangle CD J)) :=
sorry

end circumcircle_I_if_and_only_if_circumcircle_K_l189_189681


namespace sum_of_reciprocals_equal_one_l189_189805

theorem sum_of_reciprocals_equal_one (n : ℕ) (h : n ≥ 3) :
  ∃ (x : Fin n → ℕ), (∀ i j, i ≠ j → x i ≠ x j) ∧ (∑ i, 1 / (x i : ℚ) = 1) := 
sorry

end sum_of_reciprocals_equal_one_l189_189805


namespace david_marks_in_english_l189_189225

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end david_marks_in_english_l189_189225


namespace train_passes_man_in_approx_30_seconds_l189_189551

-- Define the input conditions
def train_length : ℝ := 550    -- Length of the train in meters
def train_speed_kmph : ℝ := 60 -- Speed of the train in km/h
def man_speed_kmph : ℝ := 6    -- Speed of the man in km/h

-- Conversion factor from km/h to m/s
def kmph_to_mps : ℝ := 1000 / 3600

-- Speeds in m/s
def train_speed_mps : ℝ := train_speed_kmph * kmph_to_mps
def man_speed_mps : ℝ := man_speed_kmph * kmph_to_mps

-- Relative speed (opposite directions)
def relative_speed_mps : ℝ := train_speed_mps + man_speed_mps

-- Time to pass the man
noncomputable def time_to_cross : ℝ := train_length / relative_speed_mps

-- Theorem stating the expected result
theorem train_passes_man_in_approx_30_seconds :
  |time_to_cross - 30| < 0.01 := by
  -- proof would go here
  sorry

end train_passes_man_in_approx_30_seconds_l189_189551


namespace prob_exactly_two_co_presidents_l189_189472

noncomputable def prob_two_prez_receive_books : ℚ :=
let p_club := 1/4 in
let prob_in_club (n : ℕ) : ℚ := 
  (Mathlib.Combinatorics.choose (n-2) 2 : ℚ) / (Mathlib.Combinatorics.choose n 4 : ℚ) in
p_club * (prob_in_club 6 + prob_in_club 7 + prob_in_club 8 + prob_in_club 9)

theorem prob_exactly_two_co_presidents : prob_two_prez_receive_books = 0.2667 := 
sorry

end prob_exactly_two_co_presidents_l189_189472


namespace sum_of_consecutive_integers_l189_189878

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189878


namespace find_second_sum_l189_189985

-- Define the total sum and the interest conditions
def total_sum := 2665
def interest_first_part (x : ℝ) := x * 3 * 5 / 100
def interest_second_part (x : ℝ) := (2665 - x) * 5 * 3 / 100

-- Define the equality of the two interests
def equal_interest (x : ℝ) := interest_first_part x = interest_second_part x

-- State the theorem
theorem find_second_sum (x : ℝ) (h1 : equal_interest x) : (2665 - x) = 1332.5 :=
by sorry

end find_second_sum_l189_189985


namespace consecutive_integer_product_sum_l189_189932

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189932


namespace range_of_a_l189_189309

theorem range_of_a (a : ℝ) : 
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ (a < -3 ∨ a > 1) :=
    sorry

end range_of_a_l189_189309


namespace initial_deadline_in_days_l189_189413

theorem initial_deadline_in_days
  (men_initial : ℕ)
  (days_initial : ℕ)
  (hours_per_day_initial : ℕ)
  (fraction_work_initial : ℚ)
  (additional_men : ℕ)
  (hours_per_day_additional : ℕ)
  (fraction_work_additional : ℚ)
  (total_work : ℚ := men_initial * days_initial * hours_per_day_initial)
  (remaining_days : ℚ := (men_initial * days_initial * hours_per_day_initial) / (additional_men * hours_per_day_additional * fraction_work_additional))
  (total_days : ℚ := days_initial + remaining_days) :
  men_initial = 100 →
  days_initial = 25 →
  hours_per_day_initial = 8 →
  fraction_work_initial = 1 / 3 →
  additional_men = 160 →
  hours_per_day_additional = 10 →
  fraction_work_additional = 2 / 3 →
  total_days = 37.5 :=
by
  intros
  sorry

end initial_deadline_in_days_l189_189413


namespace combination_identity_l189_189613

theorem combination_identity (x : ℕ) 
    (h : (nat.choose 10 x) = (nat.choose 8 (x - 2)) + (nat.choose 8 (x - 1)) + (nat.choose 9 (2 * x - 3))) :
    x = 3 ∨ x = 4 := sorry

end combination_identity_l189_189613


namespace childSupportOwed_l189_189408

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l189_189408


namespace curve_is_parabola_l189_189602

theorem curve_is_parabola (r θ : ℝ) (h : r = 6 * sin θ * (1 / cos θ)) : ∃ a b : ℝ, ∀ r θ : ℝ, r^2 * (cos θ)^2 = 6 * r * sin θ → (r * cos θ)^2 = 6 * r * sin θ :=
by sorry

end curve_is_parabola_l189_189602


namespace sum_of_consecutive_integers_l189_189875

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189875


namespace number_of_valid_bases_l189_189034

-- Define the main problem conditions
def base_representation_digits (n b : ℕ) := 
  let digits := (n.to_digits b).length 
  digits

def valid_bases_for_base10_256 (b : ℕ) : Prop := 
  b ≥ 2 ∧ base_representation_digits 256 b = 4

-- Theorem statement
theorem number_of_valid_bases : 
  finset.card (finset.filter valid_bases_for_base10_256 (finset.range (256 + 1))) = 2 := 
sorry

end number_of_valid_bases_l189_189034


namespace sum_of_exponentials_l189_189579

noncomputable def omega : complex := complex.exp (complex.pi * complex.I / 15)

theorem sum_of_exponentials :
  (∑ k in finset.range 14, omega ^ (k + 1)) = -1 :=
begin
  sorry
end

end sum_of_exponentials_l189_189579


namespace sum_of_consecutive_integers_with_product_812_l189_189921

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189921


namespace largest_tangential_quadrilaterals_l189_189766

-- Definitions and conditions
def convex_ngon {n : ℕ} (h : n ≥ 5) : Type := sorry -- Placeholder for defining a convex n-gon with ≥ 5 sides
def tangential_quadrilateral {n : ℕ} (h : n ≥ 5) (k : ℕ) : Prop := 
  -- Placeholder for the property that exactly k quadrilaterals out of all possible ones 
  -- in a convex n-gon have an inscribed circle
  sorry

theorem largest_tangential_quadrilaterals {n : ℕ} (h : n ≥ 5) : 
  ∃ k : ℕ, tangential_quadrilateral h k ∧ k = n / 2 :=
sorry

end largest_tangential_quadrilaterals_l189_189766


namespace sum_of_consecutive_integers_with_product_812_l189_189905

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189905


namespace consecutive_integers_sum_l189_189860

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189860


namespace range_of_m_l189_189735

noncomputable def system_of_equations (x y m : ℝ) : Prop :=
  (x + 2 * y = 1 - m) ∧ (2 * x + y = 3)

variable (x y m : ℝ)

theorem range_of_m (h : system_of_equations x y m) (hxy : x + y > 0) : m < 4 :=
by
  sorry

end range_of_m_l189_189735


namespace meet_distance_from_top_l189_189580

def time_to_reach_top (distance speed : ℝ) : ℝ := distance / speed

def distance_uphill (speed time : ℝ) : ℝ := speed * time

def distance_downhill (initial_distance speed time : ℝ) : ℝ := initial_distance - speed * time

-- Define the main problem
theorem meet_distance_from_top : 
  let 
    distance := 8 -- in kilometers
    alex_speed_up := 14 -- in km/hr
    alex_speed_down := 18 -- in km/hr
    betty_speed_up := 15 -- in km/hr
    betty_speed_down := 21 -- in km/hr
    alex_head_start := 2 / 15 -- in hours
    total_distance := 16 -- total distance for up and down the hill
    t := 128 / 231 -- the time when they meet
  in
  distance - (15 * (t - (2 / 15))) = 151 / 77 :=
by
  sorry

end meet_distance_from_top_l189_189580


namespace total_worth_of_presents_l189_189753

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l189_189753


namespace largest_non_sum_of_36_and_composite_l189_189095

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l189_189095


namespace tan_alpha_eq_3_over_4_l189_189641

variables (α : ℝ)
hypothesis (h1 : cos (π / 2 + α) = 3 / 5)
hypothesis (h2 : π / 2 < α ∧ α < 3 * π / 2)

theorem tan_alpha_eq_3_over_4 : tan α = 3 / 4 :=
sorry

end tan_alpha_eq_3_over_4_l189_189641


namespace sum_of_consecutive_integers_with_product_812_l189_189946

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189946


namespace angle_DQE_90_l189_189348

theorem angle_DQE_90
  {A B C D M N P Q F E : Type*}
  [triangle A B C]
  (hAD: altitude_triangle A B C D)
  (hAD_eq_BC : distance A D = distance B C)
  (hM_midpoint_CD : midpoint M C D)
  (hN_angle_bisector_ADC : angle_bisector_triangle A D C N)
  (hP_on_Gamma : on_circle P circle_ABC)
  (hBP_parallel_AC : parallel B P A C)
  (hDN_intersect_AM_F : intersect DN AM F)
  (hPF_intersect_Gamma_Q : intersect_again PF circle_ABC Q)
  (hAC_intersect_circumcircle_PNQ_E : intersect_again AC circumcircle_PNQ E) :
  ∠ D Q E = 90 :=
sorry

end angle_DQE_90_l189_189348


namespace perpendicular_GOA_BC_l189_189711

-- Definitions based on conditions
variables {A B C O_A O_B O_C G H I E D : Type}

axiom exists_circle_tangent_to_sides (A B C : Type) : ∃ (O_A O_B O_C : Type), 
  ∀ (AB_side : A → B → Prop) (BC_side : B → C → Prop) (CA_side : C → A → Prop)
  (tangent_to_AB : O_C → AB_side (A, B))
  (tangent_to_BC : O_A → BC_side (B, C))
  (tangent_to_CA : O_B → CA_side (C, A)), true

axiom circles_tangent (O_B O_C : Type) (BC_line : B → C → Prop) 
  (tangent_at_E : BC_line (E, B))
  (tangent_at_D : BC_line (D, C)), true

axiom circle_tangent_to_extensions (O_A : Type) (extension_of_AB : A → B → Prop) 
  (extension_of_AC : A → C → Prop) (tangent_at_H : extension_of_AB (H, A))
  (tangent_at_I : extension_of_AC (I, A)), true

axiom intersection_point (O_C O_B H I : Type) : ∃ (G : Type), 
  (line_through O_C H ∧ line_through O_B I → G)

-- Mathematical equivalence proof problem in Lean statement
theorem perpendicular_GOA_BC (A B C O_A O_B O_C G H I E D : Type)
  [exists_circle_tangent_to_sides A B C] 
  [circles_tangent O_B O_C (BC_side := λ b c, b = B ∧ c = C)] 
  [circle_tangent_to_extensions O_A (λ h_a a, h_a = A ∧ a ∈ [H,I]) (λ i_a a, i_a = A ∧ a ∈ [I,H])]
  [intersection_point O_C O_B H I] :
  perp GO_A BC := 
sorry

end perpendicular_GOA_BC_l189_189711


namespace number_of_books_is_10_l189_189714

def costPerBookBeforeDiscount : ℝ := 5
def discountPerBook : ℝ := 0.5
def totalPayment : ℝ := 45

theorem number_of_books_is_10 (n : ℕ) (h : (costPerBookBeforeDiscount - discountPerBook) * n = totalPayment) : n = 10 := by
  sorry

end number_of_books_is_10_l189_189714


namespace total_worth_of_presents_l189_189746

-- Definitions of the costs
def costOfRing : ℕ := 4000
def costOfCar : ℕ := 2000
def costOfBracelet : ℕ := 2 * costOfRing

-- Theorem statement
theorem total_worth_of_presents : 
  costOfRing + costOfCar + costOfBracelet = 14000 :=
begin
  -- by using the given definitions and the provided conditions, we assert the statement
  sorry
end

end total_worth_of_presents_l189_189746


namespace four_integers_inequality_l189_189720

theorem four_integers_inequality (S : Finset ℕ) (h_card : S.card = 9) (h_bound : ∀ n ∈ S, n ≤ 9000) :
  ∃ a b c d ∈ S, 4 + d ≤ a + b + c ∧ a + b + c ≤ 4 * d :=
by
  sorry

end four_integers_inequality_l189_189720


namespace probability_of_line_intersecting_all_colors_l189_189346

noncomputable def probability_red_white_blue : ℝ :=
  let center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 2
  let distance_from_circle : ℝ := 1
  let red_boundary : ℝ := 1
  let blue_boundary : ℝ := -1
  let total_angle : ℝ := 2 * Real.pi
  -- Angles where the line does not intersect all three colors
  let restricted_angle_range : ℝ := (2 * Real.pi) / 3
  let probability_all_three_colors := 1 - (restricted_angle_range / total_angle)
  probability_all_three_colors

-- The theorem that captures the problem statement
theorem probability_of_line_intersecting_all_colors :
  probability_red_white_blue = 2 / 3 :=
begin
  sorry
end

end probability_of_line_intersecting_all_colors_l189_189346


namespace product_of_numbers_l189_189464

theorem product_of_numbers (x y : ℤ) (h1 : x + y = 37) (h2 : x - y = 5) : x * y = 336 := by
  sorry

end product_of_numbers_l189_189464


namespace Δ_n_zero_iff_deg_le_l189_189489

open Polynomial

def Δ (P : Polynomial ℝ) : Polynomial ℝ := P.derivative - (X * P)

theorem Δ_n_zero_iff_deg_le (P : Polynomial ℝ) (n : ℕ) :
  (Δ^[n] P = 0) → P.degree ≤ (n - 1 : ℕ) :=
sorry

end Δ_n_zero_iff_deg_le_l189_189489


namespace number_of_valid_bases_l189_189035

-- Define the main problem conditions
def base_representation_digits (n b : ℕ) := 
  let digits := (n.to_digits b).length 
  digits

def valid_bases_for_base10_256 (b : ℕ) : Prop := 
  b ≥ 2 ∧ base_representation_digits 256 b = 4

-- Theorem statement
theorem number_of_valid_bases : 
  finset.card (finset.filter valid_bases_for_base10_256 (finset.range (256 + 1))) = 2 := 
sorry

end number_of_valid_bases_l189_189035


namespace math_problem_solution_l189_189362

noncomputable def problem_statement : Prop :=
  ∀ (initial_pressure : ℝ) (initial_volume : ℝ) (initial_distance : ℝ) (temp : ℝ)
    (piston_speed : ℝ) (movement_time : ℝ) (saturation_pressure_water_vapor : ℝ),
  let final_distance := initial_distance - piston_speed * movement_time in
  let final_volume := initial_volume * (final_distance / initial_distance) in
  let volume_ratio := initial_volume / final_volume in
  let final_pressure_nitrogen := volume_ratio * initial_pressure in
  let power_ratio := final_pressure_nitrogen / saturation_pressure_water_vapor in
  let total_time := 7.5 * 60 in -- convert minutes to seconds
  let interval_time := 30 in -- time in seconds
  let volume_change_interval := piston_speed * (interval_time / 60) * (initial_volume / initial_distance) in
  let work_done_vapor := saturation_pressure_water_vapor * volume_change_interval * 101325 in -- convert to Joules
  let work_threshold := 15 in
  let avg_pressure_needed := work_threshold / (volume_change_interval * (interval_time / total_time)) in
  power_ratio = 2 ∧ (work_done_vapor > work_threshold ∨ final_pressure_nitrogen > avg_pressure_needed)

theorem math_problem_solution : problem_statement :=
by sorry

end math_problem_solution_l189_189362


namespace female_students_next_to_each_other_l189_189008

theorem female_students_next_to_each_other (male female1 female2 : Type) :
  ∃ (arrangements : List (List Type)), 
     arrangements.length = 4 ∧ 
     (∀ (arrangement : List Type), arrangement ∈ arrangements → 
        (female1 ∈ arrangement.tail ∧ female2 ∈ arrangement.tail ∧ arrangement.tail.nth_le 0 _ = female1 ∧ arrangement.nth_le 1 _ = female2 ∨ 
         female1 ∈ arrangement.tail ∧ female2 ∈ arrangement.tail ∧ arrangement.tail.nth_le 0 _ = female2 ∧ arrangement.nth_le 1 _ = female1)) := 
sorry

end female_students_next_to_each_other_l189_189008


namespace equivalent_operation_l189_189126

theorem equivalent_operation : 
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  (initial_op * multiply_3_2) = (5 / 4 : ℝ) :=
by
  -- setup operations
  let initial_op := (5 / 6 : ℝ)
  let multiply_3_2 := (3 / 2 : ℝ)
  -- state the goal
  have h : (initial_op * multiply_3_2) = (5 / 4 : ℝ) := sorry
  exact h

end equivalent_operation_l189_189126


namespace veranda_width_l189_189447

def area_of_veranda (w : ℝ) : ℝ :=
  let room_area := 19 * 12
  let total_area := room_area + 140
  let total_length := 19 + 2 * w
  let total_width := 12 + 2 * w
  total_length * total_width - room_area

theorem veranda_width:
  ∃ w : ℝ, area_of_veranda w = 140 := by
  sorry

end veranda_width_l189_189447


namespace megan_folders_count_l189_189000

theorem megan_folders_count (init_files deleted_files files_per_folder : ℕ) (h₁ : init_files = 93) (h₂ : deleted_files = 21) (h₃ : files_per_folder = 8) :
  (init_files - deleted_files) / files_per_folder = 9 :=
by
  sorry

end megan_folders_count_l189_189000


namespace arithmetic_seq_properties_geometric_seq_properties_l189_189283

variable {n : ℕ}

def a_arithmetic_seq (n : ℕ) : ℕ := 2 * n - 1

def S_arithmetic_sum (n : ℕ) : ℕ := n^2

def b_geometric_seq (n : ℕ) : ℕ := 2 * 4^(n - 1)

def T_geometric_sum (n : ℕ) : ℕ := (2 / 3) * (4^n - 1)

theorem arithmetic_seq_properties :
  ∀ n : ℕ, (a_arithmetic_seq n = 2 * n - 1) ∧ (S_arithmetic_sum n = n^2) := 
sorry

theorem geometric_seq_properties :
  ∀ (n : ℕ), 
  let a_4 := 2 * 4 - 1 in
  let S_4 := 4^2 in
  (4 * 4 - 8 * 4 + 16 = 0) ∧ 
  (b_geometric_seq n = 2 * 4^(n - 1)) ∧ 
  (T_geometric_sum n = (2 / 3) * (4^n - 1)) :=
sorry

end arithmetic_seq_properties_geometric_seq_properties_l189_189283


namespace primitive_decomposition_square_exists_l189_189198

-- Definition of a primitive decomposition condition
def is_primitive_decomposition (n : ℕ) : Prop :=
  (n = 5 ∨ n ≥ 7)

-- Theorem stating the problem
theorem primitive_decomposition_square_exists (n : ℕ) :
  is_primitive_decomposition n ↔ ∃ (decomposition : finset (finset (ℕ × ℕ))),
    (∃ (squares : finset (ℕ × ℕ)), ∀ s ∈ squares, s ∈ decomposition ∧
    -- Each square is effectively decomposed into rectangles
    (∃ (a b : ℕ), a * b = s.fst * s.snd) ∧
    -- Ensure decomposition is primitive
    (∃ (p : finset (ℕ × ℕ)), p ∉ decomposition)) :=
by sorry

end primitive_decomposition_square_exists_l189_189198


namespace consecutive_integer_sum_l189_189866

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189866


namespace remainder_is_correct_l189_189255

noncomputable def p (x : ℝ) := x^4 - 2 * x^2 - 8
noncomputable def d (x : ℝ) := x^2 - 1
noncomputable def r (x : ℝ) := -x^2 - 8

theorem remainder_is_correct (x : ℝ) : IsDivRem p d r :=
by sorry

end remainder_is_correct_l189_189255


namespace consecutive_integer_product_sum_l189_189936

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189936


namespace volume_ratio_of_convex_shape_l189_189005

theorem volume_ratio_of_convex_shape (V_cube : ℝ) (midline_ratio : ℝ) 
  (h_ratio : midline_ratio = 1 / 3) :
  let V_convex := (1 / 2) * V_cube in 
  V_convex / V_cube = 1 / 2 := 
by
  sorry

end volume_ratio_of_convex_shape_l189_189005


namespace difference_in_points_l189_189729

theorem difference_in_points :
  let wildcats_score := 2.5 * 24,
      panthers_score := 1.3 * 24,
      tigers_score := 1.8 * 24,
      highest_score := max (max wildcats_score panthers_score) tigers_score,
      lowest_score := min (min wildcats_score panthers_score) tigers_score
  in
  highest_score - lowest_score = 28.8 :=
by
  let wildcats_score := 2.5 * 24
  let panthers_score := 1.3 * 24
  let tigers_score := 1.8 * 24
  let highest_score := max (max wildcats_score panthers_score) tigers_score
  let lowest_score := min (min wildcats_score panthers_score) tigers_score
  sorry

end difference_in_points_l189_189729


namespace remainder_a_cubed_l189_189767

theorem remainder_a_cubed {a n : ℤ} (hn : 0 < n) (hinv : a * a ≡ 1 [ZMOD n]) (ha : a ≡ -1 [ZMOD n]) : a^3 ≡ -1 [ZMOD n] := 
sorry

end remainder_a_cubed_l189_189767


namespace population_increase_l189_189264

theorem population_increase (k l m : ℝ) : 
  (1 + k/100) * (1 + l/100) * (1 + m/100) = 
  1 + (k + l + m)/100 + (k*l + k*m + l*m)/10000 + k*l*m/1000000 :=
by sorry

end population_increase_l189_189264


namespace log_expression_value_l189_189694

theorem log_expression_value (m n : ℕ) (hm : 0 < m) (hn : 0 < n) :
  log 2 ( ( (m^4 * n^(-4)) / (m^(-1) * n) )^(-3) / ( (m^(-2) * n^2) / (m * n^(-1)) )^5 ) = 0 :=
sorry

end log_expression_value_l189_189694


namespace average_speed_l189_189050

theorem average_speed (d t : ℕ) (h1: d = 160) (h2: t = 6) : (d / t = 80 / 3) :=
by {
  rw [h1, h2],
  norm_num,
}

end average_speed_l189_189050


namespace childSupportOwed_l189_189407

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l189_189407


namespace solve_quadratic_solution_l189_189462

theorem solve_quadratic_solution (x : ℝ) : (3 * x^2 - 6 * x = 0) ↔ (x = 0 ∨ x = 2) :=
sorry

end solve_quadratic_solution_l189_189462


namespace total_pennies_donated_l189_189208

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l189_189208


namespace triangle_subsegment_length_l189_189952

noncomputable def length_of_shorter_subsegment (PQ QR PR PS SR : ℝ) :=
  PQ < QR ∧ 
  PR = 15 ∧ 
  PQ / QR = 1 / 5 ∧ 
  PS + SR = PR ∧ 
  PS = PQ / QR * SR → 
  PS = 5 / 2

theorem triangle_subsegment_length (PQ QR PR PS SR : ℝ) 
  (h1 : PQ < QR) 
  (h2 : PR = 15) 
  (h3 : PQ / QR = 1 / 5) 
  (h4 : PS + SR = PR) 
  (h5 : PS = PQ / QR * SR) : 
  length_of_shorter_subsegment PQ QR PR PS SR := 
sorry

end triangle_subsegment_length_l189_189952


namespace min_coins_l189_189379

def sum_of_digits_base_two (a : ℕ) : ℕ :=
  Nat.popcount a

def total_coins (n : ℕ) : ℕ :=
  Nat.sum (Finset.range (2^n)) sum_of_digits_base_two

theorem min_coins (n : ℕ) (hn : 0 < n) :
  total_coins n = n * 2^(n-1) :=
by
  sorry

end min_coins_l189_189379


namespace smallest_difference_between_5_digit_numbers_l189_189966

theorem smallest_difference_between_5_digit_numbers : 
  ∃ (a b : ℕ), (a < 100000 ∧ b < 100000 ∧ a ≠ b ∧ (∀ d ∈ (Finset.range 10), 
  d ∈ (a.digits 10).to_finset ∨ d ∈ (b.digits 10).to_finset) ∧ 
  (∀ d ∈ (Finset.range 10), d ∈ (a.digits 10).to_finset → d ∉ (b.digits 10).to_finset) ∧ 
  |a - b| = 247) :=
sorry

end smallest_difference_between_5_digit_numbers_l189_189966


namespace area_AMCN_l189_189021

-- Define the points A, B, C, D, M, and N
structure Point :=
  (x : ℝ) (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨10, 0⟩
def C : Point := ⟨10, 5⟩
def D : Point := ⟨0, 5⟩
def M : Point := ⟨(B.x + C.x) / 2, (B.y + C.y) / 2⟩
def N : Point := ⟨(C.x + D.x) / 2, (C.y + D.y) / 2⟩

-- Define the function to calculate the area of a quadrilateral given its vertices
def area (P Q R S : Point) : ℝ :=
  1 / 2 * abs ((Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x)) +
  1 / 2 * abs ((R.x - P.x) * (S.y - P.y) - (R.y - P.y) * (S.x - P.x))

-- Statement: The area of quadrilateral AMCN is 31.25 square centimeters
theorem area_AMCN :
  area A M C N = 31.25 :=
sorry

end area_AMCN_l189_189021


namespace complex_number_on_imaginary_axis_l189_189822

theorem complex_number_on_imaginary_axis (a : ℝ) : (∃ z : ℂ, z = (-2 + a * complex.I) / (1 + complex.I) ∧ z.re = 0) -> a = 2 :=
by
    sorry

end complex_number_on_imaginary_axis_l189_189822


namespace sum_of_coefficients_of_polynomial_expansion_l189_189696

theorem sum_of_coefficients_of_polynomial_expansion :
  let f := (2 * x + 3) ^ 6
  let b : ℕ → ℤ := λ n, match f with
                        | polynomial.sum (fun m c => c * x ^ m) => c
                        end
  (polynomial.eval 1 f = 15625) → 
  b(6) + b(5) + b(4) + b(3) + b(2) + b(1) + b(0) = 15625 
by sorry

end sum_of_coefficients_of_polynomial_expansion_l189_189696


namespace part_one_part_two_l189_189782

noncomputable def M := Set.Ioo (-(1 : ℝ)/2) (1/2)

namespace Problem

variables {a b : ℝ}
def in_M (x : ℝ) := x ∈ M

theorem part_one (ha : in_M a) (hb : in_M b) :
  |(1/3 : ℝ) * a + (1/6) * b| < 1/4 :=
sorry

theorem part_two (ha : in_M a) (hb : in_M b) :
  |1 - 4 * a * b| > 2 * |a - b| :=
sorry

end Problem

end part_one_part_two_l189_189782


namespace largest_distance_to_vertices_less_than_twice_smallest_distance_to_sides_l189_189738

theorem largest_distance_to_vertices_less_than_twice_smallest_distance_to_sides 
  (A B C P A1 B1 C1 : Point)
  (triangle_acute : is_acute_triangle A B C)
  (P_inside : is_inside_triangle P A B C)
  (PA1_perpendicular : is_perpendicular (line_through P A1) (line_through B C))
  (PB1_perpendicular : is_perpendicular (line_through P B1) (line_through C A))
  (PC1_perpendicular : is_perpendicular (line_through P C1) (line_through A B))
  : max (dist P A) (max (dist P B) (dist P C)) < 
    2 * min (dist P A1) (min (dist P B1) (dist P C1)) :=
sorry

end largest_distance_to_vertices_less_than_twice_smallest_distance_to_sides_l189_189738


namespace maximum_cables_l189_189190

structure Organization (A B : Type) :=
  (employees : ℕ)
  (computers_A : A)
  (computers_B : B)
  (computers_count_A : ℕ)
  (computers_count_B : ℕ)
  (total_computers_A : computers_count_A = 20)
  (total_computers_B : computers_count_B = 20)
  (initially_unconnected : ∀ (a : A) (b : B), ¬connected a b)

def max_cables_for_communication (A B : Type) [fintype A] [fintype B] (org : Organization A B) : ℕ :=
if org.total_computers_A = 20 ∧ org.total_computers_B = 20 then
  20
else
  0

theorem maximum_cables {A B : Type} [fintype A] [fintype B] (org : Organization A B) :
  max_cables_for_communication A B org = 20 :=
by
  cases org with _ _ A B computers_count_A computers_count_B total_computers_A total_computers_B initially_unconnected,
  rw [max_cables_for_communication],
  rw [if_pos],
  sorry

end maximum_cables_l189_189190


namespace expectation_of_two_fair_dice_l189_189969

noncomputable def E_X : ℝ :=
  (2 * (1/36) + 3 * (2/36) + 4 * (3/36) + 5 * (4/36) + 6 * (5/36) + 7 * (6/36) + 
   8 * (5/36) + 9 * (4/36) + 10 * (3/36) + 11 * (2/36) + 12 * (1/36))

theorem expectation_of_two_fair_dice : E_X = 7 := by
  sorry

end expectation_of_two_fair_dice_l189_189969


namespace division_point_ratio_l189_189554

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def extendRay (A C : ℝ × ℝ) (k : ℝ) : ℝ × ℝ := 
  (A.1 + k * (C.1 - A.1), A.2 + k * (C.2 - A.2))

theorem division_point_ratio
    (A B C : ℝ × ℝ)
    (F : ℝ × ℝ := midpoint A B)
    (S : ℝ × ℝ := extendRay A C 2) :
  ∃ (k : ℝ), k = 2 / 3 :=
by
  sorry

end division_point_ratio_l189_189554


namespace sum_of_consecutive_integers_with_product_812_l189_189913

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189913


namespace find_MT_square_l189_189721

-- Definitions and conditions
variables (P Q R S L O M N T U : Type*)
variables (x : ℝ)
variables (PL PQ PS QR RS LO : finset ℝ)
variable (side_length_PQRS : ℝ) (area_PLQ area_QMTL area_SNUL area_RNMUT : ℝ)
variables (LO_MT_perpendicular LO_NU_perpendicular : Prop)

-- Stating the problem
theorem find_MT_square :
  (side_length_PQRS = 3) →
  (PL ⊆ PQ) →
  (PO ⊆ PS) →
  (PL = PO) →
  (PL = x) →
  (U ∈ LO) →
  (T ∈ LO) →
  (LO_MT_perpendicular) →
  (LO_NU_perpendicular) →
  (area_PLQ = 1) →
  (area_QMTL = 1) →
  (area_SNUL = 2) →
  (area_RNMUT = 2) →
  (x^2 / 2 = 1) → 
  (PL * LO = 1) →
  MT^2 = 1 / 2 :=
sorry

end find_MT_square_l189_189721


namespace polynomial_remainder_correct_l189_189949

noncomputable def remainder_polynomial (x : ℝ) : ℝ := x ^ 100

def divisor_polynomial (x : ℝ) : ℝ := x ^ 2 - 3 * x + 2

def polynomial_remainder (x : ℝ) : ℝ := 2 ^ 100 * (x - 1) - (x - 2)

theorem polynomial_remainder_correct : ∀ x : ℝ, (remainder_polynomial x) % (divisor_polynomial x) = polynomial_remainder x := by
  sorry

end polynomial_remainder_correct_l189_189949


namespace linseed_oil_remaining_l189_189475

-- Define the capacities of the cans such that they are integers
variables (x1 x2 x3 remaining : ℕ)

-- Define the conditions as hypotheses
hypothesis h_total : 30 = x1 + x2 + x3
hypothesis h_condition1 : x1 = (2 * x2) / 3
hypothesis h_condition2 : x1 = (3 * x3) / 5

-- Define the theorem that we want to prove
theorem linseed_oil_remaining : remaining = 30 - (x1 + x2 + x3) → remaining = 5 := 
by 
sorry

end linseed_oil_remaining_l189_189475


namespace solve_for_m_l189_189025

theorem solve_for_m (m : ℚ) (h1 : 3 ^ (2 * m + 3) = 1 / 81) (h2 : 1 / 81 = 3 ^ (-4)) : m = -7 / 2 :=
  sorry

end solve_for_m_l189_189025


namespace replacement_paint_intensity_l189_189030

theorem replacement_paint_intensity 
  (P_original : ℝ) (P_new : ℝ) (f : ℝ) (I : ℝ) :
  P_original = 50 →
  P_new = 45 →
  f = 0.2 →
  0.8 * P_original + f * I = P_new →
  I = 25 :=
by
  intros
  sorry

end replacement_paint_intensity_l189_189030


namespace find_ellipse_eq_max_area_MQN_l189_189396

/-- Given point \(A\) is on the ellipse \(E: \dfrac{x^2}{a^2} + \dfrac{y^2}{b^2} = 1\) (where \(a > b > 0\)). /--/
variables (A : Point) (a b : ℝ) (F1 F2 : Point) (M N Q : Point)

-- Conditions
variable (hA_on_ellipse : A ∈ {p : Point | p.x^2 / a^2 + p.y^2 / b^2 = 1})
variable (a_pos : a > 0)
variable (b_pos : b > 0)
variable (h_foci_dist : dist F1 F2 = 2 * sqrt 6)
variable (h_angle_A_F1F2 : angle A F1 F2 = 15)
variable (h_angle_A_F2F1 : angle A F2 F1 = 75)
variable (hM_on_ellipse : M ∈ {p : Point | p.x^2 / 9 + p.y^2 / 3 = 1})
variable (hN_sym_origin : N = Point.mirror_origin M)
variable (hQ_line_M_not_origin : passes_through M Q ∧ ¬ passes_through_origin M Q)

-- Goals to prove
theorem find_ellipse_eq : 
  ∃ (a b : ℝ), a = 3 ∧ b = sqrt(3) ∧ 
  (forall (p : Point), p ∈ {p : Point | p.x^2 / 9 + p.y^2 / 3 = 1}) := 
sorry

theorem max_area_MQN : 
  ∃ (maximum_area : ℝ), maximum_area = 3 * sqrt(3) :=
sorry

end find_ellipse_eq_max_area_MQN_l189_189396


namespace correct_operation_l189_189127

theorem correct_operation : 
  (sqrt (10 : ℝ) ^ (-2 : ℝ)) = 0.1 := 
by 
  sorry

end correct_operation_l189_189127


namespace inequality_holds_l189_189800

theorem inequality_holds (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    (2*x + y + z)^2 * (2*x^2 + (y + z)^2) + 
    (2*y + z + x)^2 * (2*y^2 + (z + x)^2) + 
    (2*z + x + y)^2 * (2*z^2 + (x + y)^2) ≤ 8 := 
begin
  sorry
end

end inequality_holds_l189_189800


namespace sum_of_consecutive_integers_with_product_812_l189_189911

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189911


namespace part1_part2_l189_189302

noncomputable def f (x : ℝ) : ℝ := x^2 + (1 / (x + 1))

theorem part1 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : f(x) ≥ x^2 - (4 / 9) * x + (8 / 9) :=
sorry

theorem part2 (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) : (68 / 81) < f(x) ∧ f(x) ≤ (3 / 2) :=
sorry

end part1_part2_l189_189302


namespace small_planter_capacity_l189_189419

-- Defining the parameters
def seeds_total := 200
def large_planters := 4
def seeds_per_large_planter := 20
def small_planters := 30

-- Derived quantities
def seeds_in_large_planters := large_planters * seeds_per_large_planter
def seeds_in_small_planters := seeds_total - seeds_in_large_planters
def seeds_per_small_planter := seeds_in_small_planters / small_planters

-- The theorem to prove
theorem small_planter_capacity :
  seeds_per_small_planter = 4 :=
by
  unfold seeds_per_small_planter seeds_in_small_planters seeds_in_large_planters
  simp [seeds_total, large_planters, seeds_per_large_planter, small_planters]
  sorry

end small_planter_capacity_l189_189419


namespace coefficient_of_x5_in_expansion_l189_189727

theorem coefficient_of_x5_in_expansion :
  let f := λ x : ℚ, x ^ 2 - 2 / x
  let general_term := λ (n a x : ℚ), n.choose a * ((-2)^a) * x^(2*(n-a) - a)
  let coeff := general_term 7 3
  coeff = -280 :=
by
  sorry

end coefficient_of_x5_in_expansion_l189_189727


namespace figure_single_part_l189_189226

-- Set up the problem conditions
def condition1 (x y : ℝ) : Prop := Real.sqrt (x^2 - 3 * y^2 + 4 * x + 4) ≤ 2 * x + 1
def condition2 (x y : ℝ) : Prop := x^2 + y^2 ≤ 4

-- Define the figure Φ as the set of points (x, y) satisfying both conditions
def φ (x y : ℝ) : Prop := condition1 x y ∧ condition2 x y

-- The proof problem
theorem figure_single_part : 
  ∀ (Φ : ℝ × ℝ → Prop), (Φ = λ p, φ p.1 p.2) → (∃! (region : ℝ), (Φ = λ p, φ p.1 p.2)) :=
by 
  sorry

end figure_single_part_l189_189226


namespace consecutive_integers_sum_l189_189889

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189889


namespace healthy_living_not_related_gender_95_prob_both_contact_persons_male_2_out_of_5_l189_189236

/-- Problem 1: Prove that based on given contingency table data and chi-squared formula,
{K^2} calculated is less than the critical value for 95% confidence,
hence "healthy living" and gender are not related with 95% confidence. -/
theorem healthy_living_not_related_gender_95 :
  let a := 30
  let b := 45
  let c := 15
  let d := 10
  let n := 100
  let k0 := 3.841
  {K^2} = (n * ((a * d - b * c)^2)) / ((a + b) * (c + d) * (a + c) * (b + d))
  K^2 < k0 := sorry

/-- Problem 2: Prove that the probability that both selected contact persons are male is 2/5. -/
theorem prob_both_contact_persons_male_2_out_of_5 :
  let total_combinations := Nat.choose 6 2
  let male_combinations := Nat.choose 4 2
  P(A) = (male_combinations / total_combinations)
  P(A) = 2/5 := sorry

end healthy_living_not_related_gender_95_prob_both_contact_persons_male_2_out_of_5_l189_189236


namespace smallest_percent_both_coffee_tea_l189_189004

noncomputable def smallest_percent_coffee_tea (P_C P_T P_not_C_or_T : ℝ) : ℝ :=
  let P_C_or_T := 1 - P_not_C_or_T
  let P_C_and_T := P_C + P_T - P_C_or_T
  P_C_and_T

theorem smallest_percent_both_coffee_tea :
  smallest_percent_coffee_tea 0.9 0.85 0.15 = 0.9 :=
by
  sorry

end smallest_percent_both_coffee_tea_l189_189004


namespace amount_paid_per_person_is_correct_l189_189163

noncomputable def amount_each_person_paid (total_bill : ℝ) (tip_rate : ℝ) (tax_rate : ℝ) (num_people : ℕ) : ℝ := 
  let tip_amount := tip_rate * total_bill
  let tax_amount := tax_rate * total_bill
  let total_amount := total_bill + tip_amount + tax_amount
  total_amount / num_people

theorem amount_paid_per_person_is_correct :
  amount_each_person_paid 425 0.18 0.08 15 = 35.7 :=
by
  sorry

end amount_paid_per_person_is_correct_l189_189163


namespace number_of_valid_bases_l189_189033

-- Define the main problem conditions
def base_representation_digits (n b : ℕ) := 
  let digits := (n.to_digits b).length 
  digits

def valid_bases_for_base10_256 (b : ℕ) : Prop := 
  b ≥ 2 ∧ base_representation_digits 256 b = 4

-- Theorem statement
theorem number_of_valid_bases : 
  finset.card (finset.filter valid_bases_for_base10_256 (finset.range (256 + 1))) = 2 := 
sorry

end number_of_valid_bases_l189_189033


namespace max_possible_score_l189_189351

theorem max_possible_score (s : ℝ) (h : 80 = s * 2) : s * 5 ≥ 100 :=
by
  -- sorry placeholder for the proof
  sorry

end max_possible_score_l189_189351


namespace diff_square_of_roots_eq_five_l189_189316

theorem diff_square_of_roots_eq_five :
  (α β : ℝ) (hα : α^2 - 3 * α + 1 = 0) (hβ : β^2 - 3 * β + 1 = 0) (hαβ : α ≠ β) :
  (α - β)^2 = 5 :=
sorry

end diff_square_of_roots_eq_five_l189_189316


namespace cube_surface_area_proof_l189_189172

-- Conditions
def prism_volume : ℕ := 10 * 5 * 20
def cube_volume : ℕ := 1000
def edge_length_of_cube : ℕ := 10
def cube_surface_area (s : ℕ) : ℕ := 6 * s * s

-- Theorem Statement
theorem cube_surface_area_proof : cube_volume = prism_volume → cube_surface_area edge_length_of_cube = 600 := 
by
  intros h
  -- Proof goes here
  sorry

end cube_surface_area_proof_l189_189172


namespace find_coordinates_of_P_l189_189656

-- Define the conditions and prove the coordinates of point P are (1, 1).
theorem find_coordinates_of_P : 
  ∃ P : ℝ × ℝ, (∃ (x y : ℝ), x > 0 ∧ y = exp x ∧ y = 1 / x ∧ 
  (let slope_tangent_ex := (deriv (λ x, exp x)) 0 in 
   let slope_tangent_1 := (deriv (λ x, 1 / x)) x in
   slope_tangent_ex = 1 ∧ slope_tangent_1 = -1 ∧ slope_tangent_ex * slope_tangent_1 = -1)) ∧ 
  P = (1, 1) :=
sorry

end find_coordinates_of_P_l189_189656


namespace problem_I_problem_II_problem_III_l189_189301

noncomputable def f (a x : ℝ) := a * x * Real.exp x
noncomputable def f' (a x : ℝ) := a * (1 + x) * Real.exp x

theorem problem_I (a : ℝ) (h : a ≠ 0) :
  (if a > 0 then ∀ x, (f' a x > 0 ↔ x > -1) ∧ (f' a x < 0 ↔ x < -1)
  else ∀ x, (f' a x > 0 ↔ x < -1) ∧ (f' a x < 0 ↔ x > -1)) :=
sorry

theorem problem_II (h : ∃ a : ℝ, a = 1) :
  ∃ (x : ℝ) (y : ℝ), x = -1 ∧ f 1 (-1) = -1 / Real.exp 1 ∧ ¬ ∃ y, ∀ x, y = f 1 x ∧ (f' 1 x) < 0 :=
sorry

theorem problem_III (h : ∃ m : ℝ, f 1 m = e * m * Real.exp m ∧ f' 1 m = e * (1 + m) * Real.exp m) :
  ∃ a : ℝ, a = 1 / 2 :=
sorry

end problem_I_problem_II_problem_III_l189_189301


namespace sum_of_consecutive_integers_with_product_812_l189_189948

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189948


namespace spaceship_travel_distance_l189_189550

-- Define each leg of the journey
def distance1 := 0.5
def distance2 := 0.1
def distance3 := 0.1

-- Define the total distance traveled
def total_distance := distance1 + distance2 + distance3

-- The statement to prove
theorem spaceship_travel_distance : total_distance = 0.7 := sorry

end spaceship_travel_distance_l189_189550


namespace boxes_per_case_5_l189_189756

variable (minutes_in_hour : Nat := 60)
variable (packing_rate : Nat := 10) -- boxes per minute
variable (total_cases : Nat := 240)
variable (hours_worked : Nat := 2)

def total_minutes (hours: Nat) : Nat := hours * minutes_in_hour
def total_boxes_packed (minutes: Nat) : Nat := minutes * packing_rate
def boxes_per_case (total_boxes: Nat) (cases: Nat) : Nat := total_boxes / cases

theorem boxes_per_case_5 :
  boxes_per_case (total_boxes_packed (total_minutes hours_worked)) total_cases = 5 := by
  sorry

end boxes_per_case_5_l189_189756


namespace minimum_expr_value_l189_189267

noncomputable def expr_min_value (a : ℝ) (h : a > 1) : ℝ :=
  a + 2 / (a - 1)

theorem minimum_expr_value (a : ℝ) (h : a > 1) :
  expr_min_value a h = 1 + 2 * Real.sqrt 2 :=
sorry

end minimum_expr_value_l189_189267


namespace trains_cross_time_correct_l189_189313

-- Definitions based on conditions
def train_length_1 : ℝ := 300 -- meters
def speed_train_1_kmph : ℝ := 72 -- km/h
def train_length_2 : ℝ := 500 -- meters
def speed_train_2_kmph : ℝ := 48 -- km/h
def wind_resistance_factor : ℝ := 0.9 -- 90%

-- Speed conversion from km/h to m/s
def speed_kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * (1000 / 3600)

-- Effective speed of the second train after wind resistance
def effective_speed_train_2_mps : ℝ :=
  speed_kmph_to_mps(speed_train_2_kmph) * wind_resistance_factor

-- Relative speed when both trains are moving in opposite directions
def relative_speed_mps : ℝ :=
  speed_kmph_to_mps(speed_train_1_kmph) + effective_speed_train_2_mps

-- Total distance to be covered for both trains to completely cross each other
def total_distance : ℝ :=
  train_length_1 + train_length_2

-- Time calculation using the formula Time = Distance / Speed
def crossing_time : ℝ :=
  total_distance / relative_speed_mps

-- Proof statement
theorem trains_cross_time_correct :
  crossing_time = 25 :=
by
  sorry

end trains_cross_time_correct_l189_189313


namespace ten_term_value_ninety_nine_over_hundred_not_in_sequence_a_n_in_open_interval_count_terms_in_interval_l189_189734

noncomputable def a_n (n : ℕ) : ℚ :=
  (9 * n^2 - 9 * n + 2) / (9 * n^2 - 1)

theorem ten_term_value : a_n 10 = 28 / 31 :=
sorry

theorem ninety_nine_over_hundred_not_in_sequence : 
  ¬ ∃ n : ℕ, a_n n = 99 / 100 :=
sorry

theorem a_n_in_open_interval : ∀ n : ℕ, n > 0 → 0 < a_n n ∧ a_n n < 1 :=
sorry

theorem count_terms_in_interval : 
  (∃ ! n : ℕ, 1 / 3 < a_n n ∧ a_n n < 2 / 3) :=
sorry

end ten_term_value_ninety_nine_over_hundred_not_in_sequence_a_n_in_open_interval_count_terms_in_interval_l189_189734


namespace consecutive_integers_sum_l189_189895

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189895


namespace triangle_similarity_l189_189480

theorem triangle_similarity
  {X A B C A' B' C' : Type}
  (h1 : ∀ (circ1 circ2 circ3 : Circle) (hX : X ∈ circ1 ∧ X ∈ circ2 ∧ X ∈ circ3),
    ∃ A B C : Point, (A ∈ circ1 ∧ A ≠ X) ∧ (B ∈ circ2 ∧ B ≠ X) ∧ (C ∈ circ3 ∧ C ≠ X))
  (h2 : ∀ (circBCX : Circle) (A AX : Line), A' = LineCircleIntersection AX circBCX)
  (h3 : ∀ (circCAX : Circle) (B BX : Line), B' = LineCircleIntersection BX circCAX)
  (h4 : ∀ (circABX : Circle) (C CX : Line), C' = LineCircleIntersection CX circABX) :
  Similar (Triangle ABC') (Triangle AB'C) ∧ Similar (Triangle ABC') (Triangle A'BC) ∧ Similar (Triangle A'BC) (Triangle AB'C) :=
sorry

end triangle_similarity_l189_189480


namespace magnitude_z_eq_l189_189644

def imaginary_unit : ℂ := complex.I
def equation (z : ℂ) : Prop := (2 - imaginary_unit) * z = 6 + 2 * imaginary_unit

theorem magnitude_z_eq :
  ∀ z : ℂ, equation z → complex.abs z = 2 * real.sqrt 2 := sorry

end magnitude_z_eq_l189_189644


namespace five_pow_sum_of_squares_l189_189393

theorem five_pow_sum_of_squares (n : ℕ) : 
  (∃ a b : ℕ, 5^n = a^2 + b^2 ∧ (a = b ∨ ∀ c d, 5^n = c^2 + d^2 → (c, d) = (a, b) ∨ (c, d) = (b, a))) → 
  ∃ k : ℕ, k = ⌊(n + 2) / 2⌋ := 
sorry

end five_pow_sum_of_squares_l189_189393


namespace perimeter_PQRST_l189_189352

-- Defining the points in the plane
variables (P Q R S T X : Point)

-- Given conditions
def PQ_length : ℝ := 3
def QR_length : ℝ := 3
def TS_length : ℝ := 7

-- Alias using specific point definitions
def PQ : ℝ := dist P Q
def QR : ℝ := dist Q R
def TX : ℝ := dist T X
def QX : ℝ := dist Q X
def PT : ℝ := dist P T
def XS : ℝ := dist X S
def RX : ℝ := dist R X
def RS : ℝ := dist R S
def perimeter : ℝ := PQ + QR + RS + TS + PT

-- Rectangle property
axiom PQXT_is_rectangle : rectangle P Q X T

-- Ensuring the distances from given conditions
axiom PQ_eq_QR : PQ = PQ_length
axiom QR_eq_QR_length : QR = QR_length
axiom TS_eq_TS_length : TS = TS_length

-- Prove the perimeter of polygon PQRST
theorem perimeter_PQRST : perimeter P Q R S T = 24 :=
by
  sorry

end perimeter_PQRST_l189_189352


namespace child_support_calculation_l189_189406

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l189_189406


namespace sum_seq_equals_2_pow_n_minus_1_l189_189220

-- Define the sequences a_n and b_n with given conditions
def a (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry
def b (n : ℕ) : ℕ := if n = 0 then 2 else if n = 1 then 4 else sorry

-- Relation for a_n: 2a_{n+1} = a_n + a_{n+2}
axiom a_relation (n : ℕ) : 2 * a (n + 1) = a n + a (n + 2)

-- Inequalities for b_n
axiom b_inequality_1 (n : ℕ) : b (n + 1) - b n < 2^n + 1 / 2
axiom b_inequality_2 (n : ℕ) : b (n + 2) - b n > 3 * 2^n - 1

-- Note that b_n ∈ ℤ is implied by the definition being in ℕ

-- Prove that the sum of the first n terms of the sequence { n * b_n / a_n }
theorem sum_seq_equals_2_pow_n_minus_1 (n : ℕ) : 
  (Finset.range n).sum (λ k => k * b k / a k) = 2^n - 1 := 
sorry

end sum_seq_equals_2_pow_n_minus_1_l189_189220


namespace f_at_five_over_six_l189_189269

def f : ℝ → ℝ :=
λ x, if x ≤ 0 then sin (π * x) else f (x - 1) + 1

theorem f_at_five_over_six : f (5 / 6) = 1 / 2 :=
sorry

end f_at_five_over_six_l189_189269


namespace parabola_circle_tangent_l189_189192

theorem parabola_circle_tangent : 
  ∃ r : ℝ, r = 1 / 4 ∧ ∀ x : ℝ, x^2 - x + r = 0 → (x - 1/2)^2 = 0 :=
begin
  use 1 / 4,
  split,
  { refl },
  { intro x,
    intro h,
    sorry
  }
end

end parabola_circle_tangent_l189_189192


namespace sum_of_consecutive_integers_with_product_812_l189_189907

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189907


namespace max_n_for_sparse_partition_l189_189259

def I (n : ℕ) : Set ℕ := { k | 1 ≤ k ∧ k ≤ n }

def P (n : ℕ) : Set ℝ := 
  { x | ∃ (m k : ℕ), m ∈ I n ∧ k ∈ I n ∧ x = m / Real.sqrt k }

def is_sparse_set (A : Set ℝ) : Prop :=
  ∀ x y ∈ A, ¬∃ z : ℕ, (x + y) = (z : ℝ)^2

theorem max_n_for_sparse_partition (n : ℕ) :
  (∃ (A B : Set ℝ), P n = A ∪ B ∧ A ∩ B = ∅ ∧ is_sparse_set A ∧ is_sparse_set B) ↔ n ≤ 14 :=
sorry

end max_n_for_sparse_partition_l189_189259


namespace possible_values_of_x_l189_189217

theorem possible_values_of_x (x : ℝ) (hx : 0 < x) :
  (∃ n : ℕ, sequence x 3000 n = 3001) ↔ 
  x = 3001 ∨ x = 1 ∨ x = 3001 / 9002999 ∨ x = 9002999 :=
sorry

noncomputable def sequence : ℝ → ℝ → ℕ → ℝ
| a, b, 0     := a
| a, b, 1     := b
| a, b, n + 2 := (sequence a b (n + 1) + 1) / sequence a b n

end possible_values_of_x_l189_189217


namespace area_of_triangle_for_curve_l189_189200

noncomputable def area_of_triangle (f : ℝ → ℝ) := 
  let x_intercepts : List ℝ := [0, 4] in
  let y_intercept : ℝ := 0 in
  let base := abs (x_intercepts.head! - x_intercepts.tail.head!) in
  let height := f ((x_intercepts.head! + x_intercepts.tail.head!) / 2) in
  (1 / 2) * base * height

theorem area_of_triangle_for_curve :
  area_of_triangle (λ x => x * (x - 4) ^ 2) = 16 :=
by
  sorry

end area_of_triangle_for_curve_l189_189200


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189088

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189088


namespace solve_factorial_equation_l189_189028

theorem solve_factorial_equation (x y z : ℕ) (h : (x, y, z) = (1, 1, 2) ∨ (x, y, z) = (2, 2, 1) ∨ (∃ a, a ≥ 3 ∧ (x, y, z) = (a, a, a))) : 
  (x + y) / z = (fintype.factorial x + fintype.factorial y) / fintype.factorial z :=
by 
  sorry

end solve_factorial_equation_l189_189028


namespace last_day_of_second_quarter_l189_189075

def is_common_year (days : ℕ) : Prop :=
  days = 365

def second_quarter_months : List String :=
  ["April", "May", "June"]

def days_in_june (days : ℕ) : Prop :=
  days = 30

theorem last_day_of_second_quarter (days: ℕ) (months: List String) (june_days: ℕ) (h1: is_common_year days) (h2: months = second_quarter_months) (h3: days_in_june june_days) :
  (months.get 2 = "June") ∧ (june_days = 30) := by
  sorry

end last_day_of_second_quarter_l189_189075


namespace min_diagonal_length_of_trapezoid_l189_189817

theorem min_diagonal_length_of_trapezoid (a b h d1 d2 : ℝ) 
  (h_area : a * h + b * h = 2)
  (h_diag : d1^2 + d2^2 = h^2 + (a + b)^2) 
  : d1 ≥ Real.sqrt 2 :=
sorry

end min_diagonal_length_of_trapezoid_l189_189817


namespace janice_age_is_21_l189_189366

variables (current_year : Nat) (current_month : Nat)
variables (mark_birth_year : Nat) (mark_birth_month : Nat)
variables (graham_age_difference : Nat) (janice_age_ratio : Rational)

def mark_age : Nat := current_year - mark_birth_year
def graham_age : Nat := mark_age - graham_age_difference
def janice_age : Rational := graham_age * janice_age_ratio

theorem janice_age_is_21 (h1 : current_year = 2021)
  (h2 : current_month = 2)
  (h3 : mark_birth_year = 1976)
  (h4 : mark_birth_month = 1)
  (h5 : graham_age_difference = 3)
  (h6 : janice_age_ratio = 1/2) :
  janice_age = 21 := 
by
  sorry

end janice_age_is_21_l189_189366


namespace cos_x_values_l189_189642

theorem cos_x_values (x : ℝ) (h : real.sec x - real.tan x = 3 / 4) : 
  real.cos x = 24 / 25 ∨ real.cos x = -24 / 25 :=
sorry

end cos_x_values_l189_189642


namespace largest_non_sum_of_36_and_composite_l189_189097

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l189_189097


namespace imag_part_of_complex_div_l189_189448

noncomputable def c1 : ℂ := 2 + Complex.i
noncomputable def c2 : ℂ := 3 - Complex.i

theorem imag_part_of_complex_div : Complex.imag (c1 / c2) = 1 / 2 := by
  sorry

end imag_part_of_complex_div_l189_189448


namespace norm_2u_equals_10_l189_189623

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end norm_2u_equals_10_l189_189623


namespace marble_color_196_l189_189345

theorem marble_color_196 :
  (∃ seq : ℕ → char, (∀ n, seq (n % 12) = if n % 12 < 3 then 'R' else if n % 12 < 8 then 'G' else 'B') ∧ seq 195 = 'G') :=
by
  sorry

end marble_color_196_l189_189345


namespace math_books_count_l189_189960

theorem math_books_count (total_books : ℕ) (history_books : ℕ) (geography_books : ℕ) (math_books : ℕ) 
  (h1 : total_books = 100) 
  (h2 : history_books = 32) 
  (h3 : geography_books = 25) 
  (h4 : math_books = total_books - history_books - geography_books) 
  : math_books = 43 := 
by 
  rw [h1, h2, h3] at h4;
  exact h4;
-- use 'sorry' to skip the proof if needed
-- sorry

end math_books_count_l189_189960


namespace isosceles_right_triangle_circle_area_ratio_l189_189718

theorem isosceles_right_triangle_circle_area_ratio (h r : ℝ) 
  (h_triangle : ∃ (a : ℝ), a = h / (real.sqrt 2) ∧
                           isosceles_right_triangle a a h)
  (h_circle_radius : r = h / (2 * (1 + real.sqrt 2))) :
  (π * r^2) / (1/2 * (h / (real.sqrt 2))^2) = π / (3 + 2 * real.sqrt 2) :=
by
  sorry

end isosceles_right_triangle_circle_area_ratio_l189_189718


namespace sum_of_consecutive_integers_with_product_812_l189_189914

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189914


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189087

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189087


namespace min_value_of_expression_l189_189976

theorem min_value_of_expression (x y : ℝ) : (2 * x * y - 3) ^ 2 + (x - y) ^ 2 ≥ 1 :=
sorry

end min_value_of_expression_l189_189976


namespace find_some_number_l189_189697

theorem find_some_number (a : ℕ) (h1 : a = 105) (h2 : a^3 = some_number * 35 * 45 * 35) : some_number = 1 := by
  sorry

end find_some_number_l189_189697


namespace polynomial_exists_f_eq_gh_l189_189392

theorem polynomial_exists_f_eq_gh 
  (f g h : ℤ[X])
  (hf : ∀ i, |coeff f i| ≤ 4)
  (hg : ∀ i, |coeff g i| ≤ 1)
  (hh : ∀ i, |coeff h i| ≤ 1)
  (h10 : eval 10 f = eval 10 g * eval 10 h) :
  f = g * h :=
sorry

end polynomial_exists_f_eq_gh_l189_189392


namespace nancy_money_l189_189002

-- Definitions based on problem conditions
def numQuarters : Nat := 12
def valuePerQuarter : Nat := 25 -- in cents
def cents_to_dollars(cents : Nat) : Nat := cents / 100

-- Question: How much money does Nancy have, given the conditions above
theorem nancy_money : cents_to_dollars(numQuarters * valuePerQuarter) = 3 :=
  by
    sorry

end nancy_money_l189_189002


namespace consecutive_integer_product_sum_l189_189928

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189928


namespace probability_from_first_to_last_floor_l189_189525

noncomputable def probability_of_open_path (n : ℕ) : ℚ :=
  let totalDoors := 2 * (n - 1)
  let halfDoors := n - 1
  let totalWays := Nat.choose totalDoors halfDoors
  let favorableWays := 2 ^ halfDoors
  favorableWays / totalWays

theorem probability_from_first_to_last_floor (n : ℕ) (h : n > 1) :
  probability_of_open_path n = 2 ^ (n - 1) / (Nat.choose (2 * (n - 1)) (n - 1)) := sorry

end probability_from_first_to_last_floor_l189_189525


namespace combined_percentage_of_students_preferring_tennis_is_39_l189_189563

def total_students_north : ℕ := 1800
def percentage_tennis_north : ℚ := 25 / 100
def total_students_south : ℕ := 3000
def percentage_tennis_south : ℚ := 50 / 100
def total_students_valley : ℕ := 800
def percentage_tennis_valley : ℚ := 30 / 100

def students_prefer_tennis_north : ℚ := total_students_north * percentage_tennis_north
def students_prefer_tennis_south : ℚ := total_students_south * percentage_tennis_south
def students_prefer_tennis_valley : ℚ := total_students_valley * percentage_tennis_valley

def total_students : ℕ := total_students_north + total_students_south + total_students_valley
def total_students_prefer_tennis : ℚ := students_prefer_tennis_north + students_prefer_tennis_south + students_prefer_tennis_valley

def percentage_students_prefer_tennis : ℚ := (total_students_prefer_tennis / total_students) * 100

theorem combined_percentage_of_students_preferring_tennis_is_39 :
  percentage_students_prefer_tennis = 39 := by
  sorry

end combined_percentage_of_students_preferring_tennis_is_39_l189_189563


namespace tom_fruits_left_l189_189078

theorem tom_fruits_left (oranges_initial apples_initial : ℕ) (sold_fraction_oranges sold_fraction_apples : ℚ)
    (h_oranges_initial : oranges_initial = 40)
    (h_apples_initial : apples_initial = 70)
    (h_sold_fraction_oranges : sold_fraction_oranges = 1 / 4)
    (h_sold_fraction_apples : sold_fraction_apples = 1 / 2) :
  let oranges_sold := oranges_initial * sold_fraction_oranges
      apples_sold := apples_initial * sold_fraction_apples
      total_fruits_initial := oranges_initial + apples_initial
      total_fruits_sold := oranges_sold + apples_sold
      fruits_left := total_fruits_initial - total_fruits_sold in
  fruits_left = 65 :=
by
  -- Proof goes here
  sorry

end tom_fruits_left_l189_189078


namespace consecutive_integers_sum_l189_189851

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189851


namespace coordinates_of_vertex_farthest_from_origin_l189_189433

noncomputable def eq_coordinates_of_vertex_farthest_from_origin : Prop :=
  let center := (4, 4 : ℝ × ℝ)
  let area := 36
  let scale_factor := 3
  let dilation_center := (0, 0 : ℝ × ℝ)
  let vertex_farthest_from_origin := (21, 21 : ℝ × ℝ)
  (vertex_farthest_from_origin = (21, 21))

theorem coordinates_of_vertex_farthest_from_origin :
  eq_coordinates_of_vertex_farthest_from_origin :=
  by sorry

end coordinates_of_vertex_farthest_from_origin_l189_189433


namespace cylinder_volume_ratio_l189_189529

theorem cylinder_volume_ratio (s : ℝ) :
  let r := s / 2
  let h := s
  let V_cylinder := π * r^2 * h
  let V_cube := s^3
  V_cylinder / V_cube = π / 4 :=
by
  sorry

end cylinder_volume_ratio_l189_189529


namespace sum_of_consecutive_integers_with_product_812_l189_189916

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189916


namespace rugby_team_new_avg_weight_l189_189068

noncomputable def new_average_weight (original_players : ℕ) (original_avg_weight : ℕ) 
  (new_player_weights : List ℕ) : ℚ :=
  let total_original_weight := original_players * original_avg_weight
  let total_new_weight := new_player_weights.foldl (· + ·) 0
  let new_total_weight := total_original_weight + total_new_weight
  let new_total_players := original_players + new_player_weights.length
  (new_total_weight : ℚ) / (new_total_players : ℚ)

theorem rugby_team_new_avg_weight :
  new_average_weight 20 180 [210, 220, 230] = 185.22 := by
  sorry

end rugby_team_new_avg_weight_l189_189068


namespace area_MDA_l189_189335

open Real EuclideanGeometry

variable (r : ℝ)
variable (O A B M D : Point)

-- Conditions
axiom h1 : dist O A = 2 * r
axiom h2 : dist O B = 2 * r
axiom h3 : dist A B = 2 * r
axiom h4 : midpoint M A B
axiom h5 : collinear O M A
axiom h6 : collinear O M B
axiom h7 : orthogonal (vector O M) (vector M A)
axiom h8 : orthogonal (vector M A) (vector M D)

theorem area_MDA : 
  let area := triangle_area M D A in
  area = (r ^ 2 * sqrt 3) / 8 :=
sorry

end area_MDA_l189_189335


namespace total_pennies_l189_189213

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l189_189213


namespace quadratic_solution_product_l189_189768

theorem quadratic_solution_product :
  let r := 9 / 2
  let s := -11
  (r + 4) * (s + 4) = -119 / 2 :=
by
  -- Define the quadratic equation and its solutions
  let r := 9 / 2
  let s := -11

  -- Prove the statement
  sorry

end quadratic_solution_product_l189_189768


namespace sum_reciprocal_eq_l189_189957

theorem sum_reciprocal_eq :
  ∃ (a b : ℕ), a + b = 45 ∧ Nat.lcm a b = 120 ∧ Nat.gcd a b = 5 ∧ 
  (1/a + 1/b = (3 : ℚ) / 40) := by
  sorry

end sum_reciprocal_eq_l189_189957


namespace recycle_cans_l189_189257

theorem recycle_cans (initial_cans : ℕ) (recycle_rate : ℕ) (n1 n2 n3 : ℕ)
  (h1 : initial_cans = 450)
  (h2 : recycle_rate = 5)
  (h3 : n1 = initial_cans / recycle_rate)
  (h4 : n2 = n1 / recycle_rate)
  (h5 : n3 = n2 / recycle_rate)
  (h6 : n3 / recycle_rate = 0) : 
  n1 + n2 + n3 = 111 :=
by
  sorry

end recycle_cans_l189_189257


namespace carrots_weight_l189_189532

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

end carrots_weight_l189_189532


namespace largest_positive_integer_not_sum_of_36_and_composite_l189_189108

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l189_189108


namespace T_positive_l189_189689

theorem T_positive (α : ℝ) (k : ℤ) (h₁ : α ≠ (1/2 : ℝ) * k * Real.pi) :
  let T := (Real.sin α + Real.sin α / Real.cos α) / (Real.cos α + Real.cos α / Real.sin α)
  in T > 0 :=
by
  let T := (Real.sin α + Real.sin α / Real.cos α) / (Real.cos α + Real.cos α / Real.sin α)
  have h_tmp: T ≥ 0, from sorry,
  exact h_tmp

end T_positive_l189_189689


namespace last_digit_of_one_over_729_l189_189974

def last_digit_of_decimal_expansion (n : ℕ) : ℕ := (n % 10)

theorem last_digit_of_one_over_729 : last_digit_of_decimal_expansion (1 / 729) = 9 :=
sorry

end last_digit_of_one_over_729_l189_189974


namespace rectangle_width_solution_l189_189024

noncomputable def solve_rectangle_width (W L w l : ℝ) :=
  L = 2 * W ∧ 3 * w = W ∧ 2 * l = L ∧ 6 * l * w = 5400

theorem rectangle_width_solution (W L w l : ℝ) :
  solve_rectangle_width W L w l → w = 10 * Real.sqrt 3 :=
by
  sorry

end rectangle_width_solution_l189_189024


namespace money_distribution_problem_l189_189349

theorem money_distribution_problem :
  ∃ n : ℕ, (3 * n + n * (n - 1) / 2 = 100 * n) ∧ n = 195 :=
by {
  use 195,
  sorry
}

end money_distribution_problem_l189_189349


namespace number_of_ways_to_choose_lineup_l189_189420

/--
   We have a basketball team of 16 players with a set of 3 twins and a set of 4 quadruplets.
   We want to find the number of ways to choose 6 starters such that exactly 2 of the quadruplets are included.
--/

theorem number_of_ways_to_choose_lineup : 
  ∑ (n : ℕ) in (∅ : Finset ℕ), (real.to_nat (nat.choose 4 2) * real.to_nat (nat.choose 14 4)) = 6006 :=
by
  sorry

end number_of_ways_to_choose_lineup_l189_189420


namespace solution_set_of_inequality_l189_189608

theorem solution_set_of_inequality :
  {x : ℝ | (1 - 2 * x) / (x + 3) ≥ 1} = set.Ioc (-3 : ℝ) (-2 / 3 : ℝ) := by
  sorry

end solution_set_of_inequality_l189_189608


namespace james_savings_l189_189744

-- Define the conditions
def cost_vest : ℝ := 250
def weight_plates_pounds : ℕ := 200
def cost_per_pound : ℝ := 1.2
def original_weight_vest_cost : ℝ := 700
def discount : ℝ := 100

-- Define the derived quantities based on conditions
def cost_weight_plates : ℝ := weight_plates_pounds * cost_per_pound
def total_cost_setup : ℝ := cost_vest + cost_weight_plates
def discounted_weight_vest_cost : ℝ := original_weight_vest_cost - discount
def savings : ℝ := discounted_weight_vest_cost - total_cost_setup

-- The statement to prove the savings
theorem james_savings : savings = 110 := by
  sorry

end james_savings_l189_189744


namespace smallest_positive_period_range_of_f_on_interval_l189_189310

variables (x : ℝ)

def m : ℝ × ℝ := (2 * sin x, -√3)
def n : ℝ × ℝ := (cos x, 2 * (cos x)^2 - 1)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2 + 1

-- 1. Smallest positive period of f(x)
theorem smallest_positive_period : ∀ x, f (x + π) = f x := sorry

-- 2. Range of f(x) on the interval [π/4, π/2]
theorem range_of_f_on_interval : set.range (λ x, f x) ⊆ set.Icc 2 3 := sorry

end smallest_positive_period_range_of_f_on_interval_l189_189310


namespace table_units_720_l189_189573

noncomputable def units_in_table (x : ℕ) : Prop :=
  let cara_rate := x / 12
  let carl_rate := x / 15
  let combined_rate := (cara_rate + carl_rate) - 12 in
  x = combined_rate * 6

theorem table_units_720 : units_in_table 720 :=
by
  -- This part of the proof is skipped.
  sorry

end table_units_720_l189_189573


namespace SD_eq_SM_l189_189344

theorem SD_eq_SM (ABC : Triangle) (CD : Altitude ABC) (M : Midpoint ABC.AB) 
  (K L : Point) (MK : Ray M CA) (ML : Ray M CB) (hMid : CK = CL) 
  (S : Circumcenter (Triangle.mk C K L)) : 
  SD = SM := 
by 
  sorry

end SD_eq_SM_l189_189344


namespace largest_positive_integer_not_sum_of_36_and_composite_l189_189109

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l189_189109


namespace sum_of_consecutive_integers_with_product_812_l189_189943

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189943


namespace arithmetic_progression_x_value_l189_189582

theorem arithmetic_progression_x_value :
  ∀ (x : ℝ), (3 * x + 2) - (2 * x - 4) = (5 * x - 1) - (3 * x + 2) → x = 9 :=
by
  intros x h
  sorry

end arithmetic_progression_x_value_l189_189582


namespace circle_area_k_l189_189437

theorem circle_area_k (C : ℝ) (hC : C = 36 * π) : ∃ k, k * π = π * 18^2 ∧ k = 324 :=
by
  have r : ℝ := 18
  have A : ℝ := π * r^2
  use (A / π)
  split
  · exact (mul_div_cancel_left (π * r^2) π)
  sorry

end circle_area_k_l189_189437


namespace letter_at_2023rd_position_l189_189084

-- Define the sequence and its length
def sequence : List Char :=
  ['F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'O', 'N', 'M', 'L', 'K', 'J', 'I', 'H', 'G', 'F']

def sequence_length : Nat := 18

-- Define the position based on the remainder calculation
def position := 2023 % sequence_length

-- State the theorem to prove that the 2023rd letter is 'P'
theorem letter_at_2023rd_position : sequence.get! (position - 1) = 'P' :=
  sorry

end letter_at_2023rd_position_l189_189084


namespace sets_equal_l189_189678

-- Definitions of sets M and N
def M := { u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l }
def N := { u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r }

-- Theorem statement asserting M = N
theorem sets_equal : M = N :=
by sorry

end sets_equal_l189_189678


namespace symmetrical_point_with_respect_to_origin_l189_189723

theorem symmetrical_point_with_respect_to_origin :
  (∀ (P : ℝ × ℝ), P = (3, 2) → ∃ (P' : ℝ × ℝ), P' = (-3, -2) ∧ P' = (-P.1, -P.2)) :=
by
  intro P hP
  use (-3, -2)
  simp [hP]
  constructor; apply rfl; sorry

end symmetrical_point_with_respect_to_origin_l189_189723


namespace pure_gala_trees_l189_189132

variable (T F : ℕ)

def G := T - F

theorem pure_gala_trees :
  (0.10 : ℝ) * T = 0.10 * T ∧
  F + 0.10 * T = 204 ∧
  F = 0.75 * T →
  G = 60 :=
by
  sorry

end pure_gala_trees_l189_189132


namespace compare_p_q_l189_189629

-- Define the context in which all variables are positive reals
variables {a b c d m n : ℝ}
variables [fact (0 < a)] [fact (0 < b)] [fact (0 < c)] [fact (0 < d)] [fact (0 < m)] [fact (0 < n)]

-- Definition of p and q based on given conditions
def p := real.sqrt (a * b) + real.sqrt (c * d)
def q := real.sqrt (m * a + n * c) * real.sqrt ((b / m) + (d / n))

-- The goal statement 
theorem compare_p_q : p ≤ q :=
sorry

end compare_p_q_l189_189629


namespace correct_statements_l189_189661

def f (x : ℝ) : ℝ := 2 * (abs (sin x) + cos x) * cos x - 1

theorem correct_statements (x : ℝ) :
  (f (-x) = f x) ∧ (∀ y, f y ∈ set.Icc (-real.sqrt 2) (real.sqrt 2)) :=
sorry

end correct_statements_l189_189661


namespace car_rental_cost_l189_189788

theorem car_rental_cost (D R M P C : ℝ) (hD : D = 5) (hR : R = 30) (hM : M = 500) (hP : P = 0.25) 
(hC : C = (R * D) + (P * M)) : C = 275 :=
by
  rw [hD, hR, hM, hP] at hC
  sorry

end car_rental_cost_l189_189788


namespace proof_problem_l189_189193

noncomputable def year_code : ℕ → ℕ :=
  λ year, year - 2016

noncomputable def stock : ℕ → ℝ
| 1 := 153.4
| 2 := 260.8
| 3 := 380.2
| 4 := 492
| 5 := 784
| _ := 0  -- Extend as needed for simplicity

noncomputable def regression_equation (x : ℕ) : ℝ :=
  (149.24 * x) - 33.64

def expected_stock_2023 : ℝ :=
  regression_equation (year_code 2023)

def residual_2021 : ℝ :=
  stock 5 - regression_equation (year_code 2021)

theorem proof_problem :
  (expected_stock_2023 > 1000) ∧
  (stock 1 < stock 2 ∧ 
   stock 2 < stock 3 ∧ 
   stock 3 < stock 4 ∧ 
   stock 4 < stock 5) ∧
  (residual_2021 = 71.44) :=
  by
  sorry

end proof_problem_l189_189193


namespace focal_length_ellipse_l189_189047

theorem focal_length_ellipse : 
  ∃ (c : ℝ), (2 * c = 2 * Real.sqrt 3) ∧ (∃ (a b : ℝ), (a^2 = 4) ∧ (b^2 = 1) ∧ (c = Real.sqrt (a^2 - b^2))). 
  sorry

end focal_length_ellipse_l189_189047


namespace child_support_owed_l189_189410

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l189_189410


namespace tangent_line_equation_l189_189604

noncomputable def f (x : ℝ) : ℝ := (2 + Real.sin x) / Real.cos x

theorem tangent_line_equation :
  let x0 : ℝ := 0
  let y0 : ℝ := f x0
  let m : ℝ := (2 * x0 + 1) / (Real.cos x0 ^ 2)
  ∃ (a b c : ℝ), a * x0 + b * y0 + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
by
  sorry

end tangent_line_equation_l189_189604


namespace slopes_of_midpoints_l189_189635

-- Define the necessary objects and properties
variables {A B C M N P O : Type} [Field A] 
variables (x1 x2 x3 y1 y2 y3 s1 s2 s3 t1 t2 t3 k1 k2 k3 : A)

-- Conditions
def hyperbola (x y : A) : Prop := (x^2) / 2 - (y^2) / 4 = 1
def slopes_sum (kab kbc kac : A) : Prop := kab + kbc + kac = -1
def midpoints (s t : A) (p1 p2 p : A) : Prop := s = (p1 + p2) / 2 ∧ t = (p1 + p2) / 2

-- Prove the theorem
theorem slopes_of_midpoints :
  hyperbola x1 y1 ∧ hyperbola x2 y2 ∧ hyperbola x3 y3 ∧ 
  slopes_sum (2 * (s1 / t1)) (2 * (s2 / t2)) (2 * (s3 / t3)) ∧ 
  midpoints s1 t1 (x1 + x2) ∧ midpoints s2 t2 (x2 + x3) ∧ midpoints s3 t3 (x1 + x3) ∧ 
  (k1 ≠ 0 ∧ k2 ≠ 0 ∧ k3 ≠ 0) → 
  (1 / k1) + (1 / k2) + (1 / k3) = -1 / 2 :=
sorry

end slopes_of_midpoints_l189_189635


namespace range_of_f_l189_189254

open Real

noncomputable def f (x : ℝ) : ℝ :=
  arcsin x + arccos x + arcsec x

theorem range_of_f :
  Set.range f = {π / 2, 3 * π / 2} :=
by
  sorry

end range_of_f_l189_189254


namespace find_x_l189_189238

theorem find_x (x : ℝ) : abs (2 * x - 1) = 3 * x + 6 ∧ x + 2 > 0 ↔ x = -1 := 
by
  sorry

end find_x_l189_189238


namespace find_divided_number_l189_189072

theorem find_divided_number:
  ∃ x : ℕ, (x % 127 = 6) ∧ (2037 % 127 = 5) ∧ x = 2038 :=
by
  sorry

end find_divided_number_l189_189072


namespace number_of_true_propositions_is_two_l189_189231

theorem number_of_true_propositions_is_two :
  let P1 := ¬(∀ x : ℝ, x^2 - x > 0)
  let P2 := ∀ x : ℕ+, 2 * x ^ 4 + 1 % 2 = 1
  let P3 := ∀ x : ℝ, |2 * x - 1| > 1 → (0 < 1 / x ∧ 1 / x < 1) ∨ 1 / x < 0
  (¬P1 ∧ P2 ∧ P3) ↔ 2 :=
by
  sorry

end number_of_true_propositions_is_two_l189_189231


namespace chimney_bricks_l189_189569

variable (h : ℕ)

/-- Brenda would take 8 hours to build a chimney alone. 
    Brandon would take 12 hours to build it alone. 
    When they work together, their efficiency is diminished by 15 bricks per hour due to their chatting. 
    If they complete the chimney in 6 hours when working together, then the total number of bricks in the chimney is 360. -/
theorem chimney_bricks
  (h : ℕ)
  (Brenda_rate : ℕ)
  (Brandon_rate : ℕ)
  (effective_rate : ℕ)
  (completion_time : ℕ)
  (h_eq : Brenda_rate = h / 8)
  (h_eq_alt : Brandon_rate = h / 12)
  (effective_rate_eq : effective_rate = (Brenda_rate + Brandon_rate) - 15)
  (completion_eq : 6 * effective_rate = h) :
  h = 360 := by 
  sorry

end chimney_bricks_l189_189569


namespace amina_reach_1_after_six_divisions_l189_189812

def iter_div2_floor (n : ℕ) : ℕ :=
  if n = 1 then 0 else 1 + iter_div2_floor (n / 2)

theorem amina_reach_1_after_six_divisions : iter_div2_floor 64 = 6 :=
by sorry

end amina_reach_1_after_six_divisions_l189_189812


namespace norm_2u_equals_10_l189_189625

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end norm_2u_equals_10_l189_189625


namespace sum_of_squares_of_roots_eq_63_l189_189233

theorem sum_of_squares_of_roots_eq_63 {y : Type} [char_zero y] (r s t : y) :
  ∀ (h1: r + s + t = 9) (h2: r * s + r * t + s * t = 9) (h3: r * s * t = 4), (r^2 + s^2 + t^2 = 63) :=
by
  intros
  sorry

end sum_of_squares_of_roots_eq_63_l189_189233


namespace car_speed_l189_189131

-- Define the given conditions
def distance_covered : ℝ := 624
def time_taken : ℝ := 2 + 2/5

-- State the theorem to be proved
theorem car_speed : distance_covered / time_taken = 260 :=
by
  -- Skipping proof with sorry
  sorry

end car_speed_l189_189131


namespace geometric_sequence_of_reciprocals_largest_n_for_Sn_l189_189273

def a (n : ℕ) : ℝ := 
  if n = 1 then 3 / 5 
  else function.iterate (λ x, 3 * x / (2 * x + 1)) (n-1) (3 / 5)

def b (n : ℕ) : ℝ := (1/a n) - 1

theorem geometric_sequence_of_reciprocals : ∃ r : ℝ, ∀ n : ℕ, b (n+1) = r * b n := 
sorry

def S (n : ℕ) : ℝ := ∑ i in finset.range n, (1 / a (i + 1))

theorem largest_n_for_Sn : ∃ n : ℕ, S n < 100 ∧ ∀ m > n, S m ≥ 100 :=
sorry

end geometric_sequence_of_reciprocals_largest_n_for_Sn_l189_189273


namespace sum_of_consecutive_integers_l189_189872

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189872


namespace graph_shift_right_by_pi_over_6_l189_189076

theorem graph_shift_right_by_pi_over_6 :
  ∀ x : ℝ, sin (2 * x - π / 3) = sin (2 * (x - π / 6)) :=
by
  intro x
  sorry

end graph_shift_right_by_pi_over_6_l189_189076


namespace count_officers_assignments_l189_189793

theorem count_officers_assignments (n : ℕ) (h : n = 15) :
  ∃ k : ℕ, k = 15 * 14 * 13 * 12 * 11 ∧ k = 360360 :=
by
  use 360360
  split
  · sorry -- Proof that 15 * 14 * 13 * 12 * 11 = 360360
  · refl -- Trivial proof that 360360 = 360360

end count_officers_assignments_l189_189793


namespace sequence_expression_l189_189760

theorem sequence_expression (c : ℕ) (h : c > 0) : 
  ∀ n, (x : ℕ) → 
    (x 1 = c) ∧ 
    (∀ n ≥ 2, x (n + 1) = x n + ⌊(2 * x n - (n + 2)) / n⌋ + 1) → 
    x n = c + (n * (n + 1)) / 2 := 
begin
  sorry
end

end sequence_expression_l189_189760


namespace at_least_one_not_less_than_2_l189_189774

noncomputable def x : ℝ := sorry
noncomputable def y : ℝ := sorry
noncomputable def z : ℝ := sorry

def a (x y : ℝ) : ℝ := x + 1 / y
def b (y z : ℝ) : ℝ := y + 1 / z
def c (z x : ℝ) : ℝ := z + 1 / x

theorem at_least_one_not_less_than_2 (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  a x y ≥ 2 ∨ b y z ≥ 2 ∨ c z x ≥ 2 := by
sorry

end at_least_one_not_less_than_2_l189_189774


namespace parallelogram_sticks_l189_189700

theorem parallelogram_sticks (a : ℕ) (h₁ : ∃ l₁ l₂, l₁ = 5 ∧ l₂ = 5 ∧ 
                                (l₁ = l₂) ∧ (a = 7)) : a = 7 :=
by sorry

end parallelogram_sticks_l189_189700


namespace david_english_marks_l189_189223

def david_marks (math physics chemistry biology avg : ℕ) : ℕ :=
  avg * 5 - (math + physics + chemistry + biology)

theorem david_english_marks :
  let math := 95
  let physics := 82
  let chemistry := 97
  let biology := 95
  let avg := 93
  david_marks math physics chemistry biology avg = 96 :=
by
  -- Proof is skipped
  sorry

end david_english_marks_l189_189223


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189090

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189090


namespace combined_work_days_l189_189983

-- Definitions for the conditions
def work_rate (days : ℕ) : ℚ := 1 / days
def combined_work_rate (days_a days_b : ℕ) : ℚ :=
  work_rate days_a + work_rate days_b

-- Theorem to prove
theorem combined_work_days (days_a days_b : ℕ) (ha : days_a = 15) (hb : days_b = 30) :
  1 / (combined_work_rate days_a days_b) = 10 :=
by
  rw [ha, hb]
  sorry

end combined_work_days_l189_189983


namespace max_value_of_a2_b2_c2_l189_189388

variables {a b c S : ℝ} {A B C : ℝ} {triangle_ABC : Prop}
hypothesis (h1 : S = (1 / 2) * c^2)
hypothesis (h2 : ab = sqrt 2)

theorem max_value_of_a2_b2_c2 : (a^2 + b^2 + c^2) ≤ 4 :=
sorry

end max_value_of_a2_b2_c2_l189_189388


namespace probability_of_red_ball_and_removed_red_balls_l189_189719

-- Conditions for the problem
def initial_red_balls : Nat := 10
def initial_yellow_balls : Nat := 2
def initial_blue_balls : Nat := 8
def total_balls : Nat := initial_red_balls + initial_yellow_balls + initial_blue_balls

-- Problem statement in Lean
theorem probability_of_red_ball_and_removed_red_balls :
  (initial_red_balls / total_balls = 1 / 2) ∧
  (∃ (x : Nat), -- Number of red balls removed
    ((initial_yellow_balls + x) / total_balls = 2 / 5) ∧
    (initial_red_balls - x = 10 - 6)) := 
by
  -- Lean will need the proofs here; we use sorry for now.
  sorry

end probability_of_red_ball_and_removed_red_balls_l189_189719


namespace add_to_both_num_and_denom_l189_189119

theorem add_to_both_num_and_denom (n : ℕ) : (4 + n) / (7 + n) = 7 / 8 ↔ n = 17 := by
  sorry

end add_to_both_num_and_denom_l189_189119


namespace line_plane_relationship_l189_189701

variables {V : Type} [InnerProductSpace ℝ V]

theorem line_plane_relationship {a b : V} (α : Set V) [Plane α] 
  (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a ⊥ b) (h4 : a ∈ α) : 
  b ∈ α ∨ (∀ {p : V}, p ∈ α → ¬ (b ≠ 0 ∧ b - p ∈ α)) ∨ ∃ q ∈ α, b = q + c • r :=
sorry

end line_plane_relationship_l189_189701


namespace limit_expression_l189_189201

theorem limit_expression : 
  tendsto (λ n : ℕ, (3^n - 1) / (3^(n+1) + 1)) at_top (𝓝 (1 / 3)) :=
by sorry

end limit_expression_l189_189201


namespace equal_degrees_of_all_vertices_l189_189504

theorem equal_degrees_of_all_vertices
  {V : Type*} [Fintype V] (G : SimpleGraph V)
  (h1 : ∀ {a b : V}, G.adj a b → ∀ {c : V}, G.adj a c → G.adj b c → false)
  (h2 : ∀ {a b : V}, ¬ G.adj a b → ∃! (c d : V), c ≠ d ∧ G.adj c a ∧ G.adj c b ∧ G.adj d a ∧ G.adj d b) :
  ∀ (v w : V), G.degree v = G.degree w := sorry

end equal_degrees_of_all_vertices_l189_189504


namespace arithmetic_difference_l189_189045

variable (S : ℕ → ℤ)
variable (n : ℕ)

-- Definitions as conditions from the problem
def is_arithmetic_sum (s : ℕ → ℤ) :=
  ∀ n : ℕ, s n = 2 * n ^ 2 - 5 * n

theorem arithmetic_difference :
  is_arithmetic_sum S →
  S 10 - S 7 = 87 :=
by
  intro h
  sorry

end arithmetic_difference_l189_189045


namespace limit_of_given_function_equals_e_pow_5_l189_189202

open Real

theorem limit_of_given_function_equals_e_pow_5 :
  tendsto (λ x, ( (2 * x - 1) / x ) ^ (1 / ((x^(1/5)) - 1))) (nhds 1) (nhds (exp 5)) :=
sorry

end limit_of_given_function_equals_e_pow_5_l189_189202


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189101

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189101


namespace least_number_to_subtract_l189_189118

theorem least_number_to_subtract (x y : ℕ) (h : x = 13605) (h1 : y = 87) : 
  ∃ r, r = x % y ∧ (x - r) % y = 0 :=
by
  -- Definitions to start the theorem
  let r := x % y
  use r
  have hr : r = x % y := rfl
  have h_sub : (x - r) % y = 0 := Nat.sub_mod_self x % y x%y rfl -- Known property
  exact ⟨hr, h_sub⟩

end least_number_to_subtract_l189_189118


namespace sum_of_digits_l189_189730

def digits (n : ℕ) : Prop := n ≥ 0 ∧ n < 10

def P := 1
def Q := 0
def R := 2
def S := 5
def T := 6

theorem sum_of_digits :
  digits P ∧ digits Q ∧ digits R ∧ digits S ∧ digits T ∧ 
  (10000 * P + 1000 * Q + 100 * R + 10 * S + T) * 4 = 41024 →
  P + Q + R + S + T = 14 :=
by
  sorry

end sum_of_digits_l189_189730


namespace find_BK_l189_189792

-- Given condition: rhombus ABCD
variables (A B C D K Q O: Type)
variable [h_rhombus : rhombus ABCD]

-- Given point K is on extension of AD beyond D
axiom K_on_extension_AD : ∃ (α : ℝ), α > 1 ∧ K = α • D

-- Given AK = 14
axiom AK_eq_14 : dist A K = 14

-- A, B, Q lie on a circle with radius 6
axiom circle_center_on_segment_AA : ∃ O, dist A O = 6 ∧ dist B O = 6 ∧ dist Q O = 6 ∧ lies_on_segment A O

-- Question to prove BK = 20
theorem find_BK : dist B K = 20 :=
sorry

end find_BK_l189_189792


namespace monthly_manufacturing_expenses_l189_189180

-- Definitions related to the problem conditions
def num_looms : ℕ := 125
def sales_value : ℕ := 500000
def establishment_charges : ℕ := 75000
def loss_per_idle_loom : ℕ := 2800

-- Goal statement in Lean 4
theorem monthly_manufacturing_expenses :
  let M := 150000 in (sales_value / num_looms - loss_per_idle_loom) * num_looms = M :=
by
  sorry

end monthly_manufacturing_expenses_l189_189180


namespace r_minus_s_l189_189769

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end r_minus_s_l189_189769


namespace consecutive_integers_sum_l189_189857

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189857


namespace area_parallelogram_l189_189570

def vector_space (V : Type*) := 
  (add_comm_group V) [module ℝ V]

variables (V : Type*) [vector_space V]
variable (p q a b : V)
variable (π : ℝ)

-- Magnitudes
variable [H1 : ∥p∥ = 1]
variable [H2 : ∥q∥ = 2]

-- Angle between p and q is π/6
variable [H3 : real.angle (p, q) = π/6]

-- vector definitions
def a : V := p + 2 • q
def b : V := 3 • p - q

theorem area_parallelogram :
  ∥ a × b ∥ = 7 :=
sorry

end area_parallelogram_l189_189570


namespace consecutive_integers_sum_l189_189883

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189883


namespace alper_cannot_guarantee_win_l189_189467

-- Define the game state and the conditions
def Game := ℕ -- Number of stones on the table
def canTake (n : ℕ) := n = 1 ∨ n = 2

-- Function to determine if a position is losing
def isLosingPosition : Game → Prop
| 0 => true
| 1 => false
| 2 => false
| n => (∀ m, canTake m → isLosingPosition (n - m))

-- Theorem stating that Alper cannot guarantee a win for k = 5,6,7,8,9
theorem alper_cannot_guarantee_win :
  let possible_ks := [5, 6, 7, 8, 9]
  ∀ k ∈ possible_ks, isLosingPosition k :=
by {
  let possible_ks := [5, 6, 7, 8, 9]
  intro k hk,
  cases hk,
  -- Here you should develop the proof for each specific case
  sorry, -- placeholders for the actual proof steps
  sorry,
  sorry,
  sorry,
  sorry
}

end alper_cannot_guarantee_win_l189_189467


namespace total_profit_is_50_l189_189167

-- Define the initial conditions
def initial_milk : ℕ := 80
def initial_water : ℕ := 20
def milk_cost_per_liter : ℕ := 22
def first_mixture_milk : ℕ := 40
def first_mixture_water : ℕ := 5
def first_mixture_price : ℕ := 19
def second_mixture_milk : ℕ := 25
def second_mixture_water : ℕ := 10
def second_mixture_price : ℕ := 18
def third_mixture_milk : ℕ := initial_milk - (first_mixture_milk + second_mixture_milk)
def third_mixture_water : ℕ := 5
def third_mixture_price : ℕ := 21

-- Define variables for revenue calculations
def first_mixture_revenue : ℕ := (first_mixture_milk + first_mixture_water) * first_mixture_price
def second_mixture_revenue : ℕ := (second_mixture_milk + second_mixture_water) * second_mixture_price
def third_mixture_revenue : ℕ := (third_mixture_milk + third_mixture_water) * third_mixture_price
def total_revenue : ℕ := first_mixture_revenue + second_mixture_revenue + third_mixture_revenue

-- Define the total milk cost
def total_milk_used : ℕ := first_mixture_milk + second_mixture_milk + third_mixture_milk
def total_cost : ℕ := total_milk_used * milk_cost_per_liter

-- Define the profit as the difference between total revenue and total cost
def profit : ℕ := total_revenue - total_cost

-- Prove that the total profit is Rs. 50
theorem total_profit_is_50 : profit = 50 := by
  sorry

end total_profit_is_50_l189_189167


namespace consecutive_integers_sum_l189_189900

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189900


namespace find_a_l189_189397

variable (a : ℝ)
def U := {2, 4, 3 - a^2}
def P := {2, a^2 - a + 2}
def complement_U_P := {-1}

theorem find_a : U = P ∪ complement_U_P → a = 2 := by
  sorry

end find_a_l189_189397


namespace jacket_price_restore_l189_189990

theorem jacket_price_restore (P : ℝ) (h₁ : P_1 = P * 0.85) (h₂ : P_2 = P_1 * 0.70)
  (h₃ : x = (P / P_2 - 1) * 100) : x ≈ 68.07 :=
by
  sorry

end jacket_price_restore_l189_189990


namespace david_marks_in_english_l189_189224

theorem david_marks_in_english
  (math phys chem bio : ℕ)
  (avg subs : ℕ) 
  (h_math : math = 95) 
  (h_phys : phys = 82) 
  (h_chem : chem = 97) 
  (h_bio : bio = 95) 
  (h_avg : avg = 93)
  (h_subs : subs = 5) :
  ∃ E : ℕ, (avg * subs = E + math + phys + chem + bio) ∧ E = 96 :=
by
  sorry

end david_marks_in_english_l189_189224


namespace quadratic_has_integer_roots_l189_189286

theorem quadratic_has_integer_roots (n : ℕ) (h : n ≠ 0): 
  (∃ x : ℤ, x^2 - 4 * x + n = 0) ↔ (n = 3 ∨ n = 4) := 
by {
  split; 
  intro h',
  sorry,
  sorry
}

end quadratic_has_integer_roots_l189_189286


namespace find_theta_l189_189980

theorem find_theta (Theta : ℕ) (h1 : 1 ≤ Theta ∧ Theta ≤ 9)
  (h2 : 294 / Theta = (30 + Theta) + 3 * Theta) : Theta = 6 :=
by sorry

end find_theta_l189_189980


namespace spike_hunts_20_crickets_per_day_l189_189031

/-- Spike the bearded dragon hunts 5 crickets every morning -/
def spike_morning_crickets : ℕ := 5

/-- Spike hunts three times the morning amount in the afternoon and evening -/
def spike_afternoon_evening_multiplier : ℕ := 3

/-- Total number of crickets Spike hunts per day -/
def spike_total_crickets_per_day : ℕ := spike_morning_crickets + spike_morning_crickets * spike_afternoon_evening_multiplier

/-- Prove that the total number of crickets Spike hunts per day is 20 -/
theorem spike_hunts_20_crickets_per_day : spike_total_crickets_per_day = 20 := 
by
  sorry

end spike_hunts_20_crickets_per_day_l189_189031


namespace rectangleArea_l189_189177

-- Definitions based on the conditions
def squareArea : ℕ := 25
def sideLengthOfSquare := Nat.sqrt squareArea
def widthOfRectangle := sideLengthOfSquare
def lengthOfRectangle := 2 * widthOfRectangle

-- Proof statement
theorem rectangleArea :
  let sideLength := Nat.sqrt squareArea in
  let width := sideLength in
  let length := 2 * width in
  let area := length * width in
  area = 50 := by
  sorry

end rectangleArea_l189_189177


namespace child_support_owed_l189_189411

noncomputable def income_first_3_years : ℕ := 3 * 30000
noncomputable def raise_per_year : ℕ := 30000 * 20 / 100
noncomputable def new_salary : ℕ := 30000 + raise_per_year
noncomputable def income_next_4_years : ℕ := 4 * new_salary
noncomputable def total_income : ℕ := income_first_3_years + income_next_4_years
noncomputable def total_child_support : ℕ := total_income * 30 / 100
noncomputable def amount_paid : ℕ := 1200
noncomputable def amount_owed : ℕ := total_child_support - amount_paid

theorem child_support_owed : amount_owed = 69000 := by
  sorry

end child_support_owed_l189_189411


namespace calculate_AE_l189_189484

variable {k : ℝ} (A B C D E : Type*)

namespace Geometry

def shared_angle (A B C : Type*) : Prop := sorry -- assumes triangles share angle A

def prop_constant_proportion (AB AC AD AE : ℝ) (k : ℝ) : Prop :=
  AB * AC = k * AD * AE

theorem calculate_AE
  (A B C D E : Type*) 
  (AB AC AD AE : ℝ)
  (h_shared : shared_angle A B C)
  (h_AB : AB = 5)
  (h_AC : AC = 7)
  (h_AD : AD = 2)
  (h_proportion : prop_constant_proportion AB AC AD AE k)
  (h_k : k = 1) :
  AE = 17.5 := 
sorry

end Geometry

end calculate_AE_l189_189484


namespace hexagon_side_length_l189_189716

theorem hexagon_side_length {a b c d e f : ℝ} (h_seq : [a, b, c, d, e, f] = [1, 2, 3, 4, 5, 6]) 
  (h_right_angles : ∀ i ∈ [1, 2, 3, 4, 5], (1:ℝ) * (1:ℝ) = 0) : 
  let s := sqrt ((2 + 4 + 6) ^ 2 + (1 + 3 + 5) ^ 2) / 2 in
  s = 15 / 2 :=
by
  -- Proof goes here
  sorry

end hexagon_side_length_l189_189716


namespace least_element_in_valid_set_l189_189383

noncomputable def is_valid_set (T : Set ℕ) : Prop :=
  ∀ c d ∈ T, c < d → ¬ (d % c = 0)

noncomputable def is_prime_or_gt_15 (n : ℕ) : Prop :=
  Prime n ∨ (15 < n ∧ n ≤ 18)

theorem least_element_in_valid_set :
  ∃ (T : Set ℕ), T ⊆ {n | 1 ≤ n ∧ n ≤ 18} ∧ is_valid_set T ∧ (∀ t ∈ T, is_prime_or_gt_15 t) ∧ (T.card = 8) ∧ (∀ t ∈ T, 2 ∈ T ∧ (∀ x ∈ T, 2 ≤ x)) :=
sorry

end least_element_in_valid_set_l189_189383


namespace minimum_area_integer_triangle_l189_189519

theorem minimum_area_integer_triangle :
  ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ (∃ (p q : ℤ), 2 ∣ (16 * p - 30 * q)) 
  → (∃ (area : ℝ), area = (1/2 : ℝ) * |16 * p - 30 * q| ∧ area = 1) :=
by
  sorry

end minimum_area_integer_triangle_l189_189519


namespace interest_percentage_correct_l189_189162

noncomputable def encyclopedia_cost : ℝ := 1200
noncomputable def down_payment : ℝ := 500
noncomputable def monthly_payment : ℝ := 70
noncomputable def final_payment : ℝ := 45
noncomputable def num_monthly_payments : ℕ := 12
noncomputable def total_installment_payments : ℝ := (num_monthly_payments * monthly_payment) + final_payment
noncomputable def total_cost_paid : ℝ := total_installment_payments + down_payment
noncomputable def amount_borrowed : ℝ := encyclopedia_cost - down_payment
noncomputable def interest_paid : ℝ := total_cost_paid - encyclopedia_cost
noncomputable def interest_percentage : ℝ := (interest_paid / amount_borrowed) * 100

theorem interest_percentage_correct : interest_percentage = 26.43 := by
  sorry

end interest_percentage_correct_l189_189162


namespace train_meeting_distance_l189_189991

/-- Two trains start from two stations at the same time, one at 20 km/h and the other at 25 km/h.
    When they meet, the faster train has traveled 75 km more than the slower train.
    Prove that the distance between the two stations is 675 km. -/
theorem train_meeting_distance (t : ℕ) (x : ℕ) (s1 s2 d : ℕ)
  (hs1 : s1 = 20) (hs2 : s2 = 25) (d_diff : d = 75)
  (hx1 : x = s1 * t) (hx2 : x + d_diff = s2 * t) :
  x + (x + d_diff) = 675 :=
by
  sorry

end train_meeting_distance_l189_189991


namespace tuning_day_method_pi_l189_189040

variable (x : ℝ)

-- Initial bounds and approximations
def initial_bounds (π : ℝ) := 31 / 10 < π ∧ π < 49 / 15

-- Definition of the "Tuning Day Method"
def tuning_day_method (a b c d : ℕ) (a' b' : ℝ) := a' = (b + d) / (a + c)

theorem tuning_day_method_pi :
  ∀ π : ℝ, initial_bounds π →
  (31 / 10 < π ∧ π < 16 / 5) ∧ 
  (47 / 15 < π ∧ π < 63 / 20) ∧
  (47 / 15 < π ∧ π < 22 / 7) →
  22 / 7 = 22 / 7 :=
by
  sorry

end tuning_day_method_pi_l189_189040


namespace amplitude_of_sine_l189_189568

-- Define the constants and the function
variables (a b c d : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_pos_d : 0 < d)
def f (x : ℝ) : ℝ := a * Real.sin (b * x + c) + d

-- Define the conditions
def max_value : ℝ := 5
def min_value : ℝ := -3

-- The proof statement that the amplitude a is 4
theorem amplitude_of_sine :
  a = (max_value - min_value) / 2 :=
by
  -- Assume the problem conditions (which will typically go here)
  sorry

end amplitude_of_sine_l189_189568


namespace rice_in_each_container_l189_189989

variable (weight_in_pounds : ℚ := 35 / 2)
variable (num_containers : ℕ := 4)
variable (pound_to_oz : ℕ := 16)

theorem rice_in_each_container :
  (weight_in_pounds * pound_to_oz) / num_containers = 70 :=
by
  sorry

end rice_in_each_container_l189_189989


namespace hydrochloric_acid_moles_required_l189_189312

open Nat

theorem hydrochloric_acid_moles_required :
  ∀ (AgNO3 HCl AgCl HNO3 : Type) 
  [HCl_is_acid : Inhabited HCl] 
  [AgNO3_is_nitrate : Inhabited AgNO3] 
  [AgCl_is_chloride : Inhabited AgCl] 
  [HNO3_is_acid : Inhabited HNO3],
  (∀ {n m : Nat}, n = 2 → m = n → m = 2) :=
by
  intros AgNO3 HCl AgCl HNO3 HCl_is_acid AgNO3_is_nitrate AgCl_is_chloride HNO3_is_acid
  intros n m h1 h2
  rw [h1, h2]
  rfl
  sorry

end hydrochloric_acid_moles_required_l189_189312


namespace consecutive_integers_sum_l189_189852

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189852


namespace compound_interest_period_l189_189496

theorem compound_interest_period :
  ∀ (P A r n : ℝ) (t : ℝ),
  P = 1500 →
  A = 1815 →
  r = 0.10 →
  n = 1 →
  A = P * (1 + r / n) ^ (n * t) →
  t = 2 :=
by
  intros P A r n t P_val A_val r_val n_val formula
  rw [P_val, A_val, r_val, n_val, ←formula]
  sorry

end compound_interest_period_l189_189496


namespace consecutive_integer_product_sum_l189_189935

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189935


namespace log_value_l189_189326

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_value (x : ℝ) (h : log_base 3 (5 * x) = 3) : log_base x 125 = 3 / 2 :=
  by
  sorry

end log_value_l189_189326


namespace sum_of_coefficients_l189_189032

theorem sum_of_coefficients (d : ℤ) (h : d ≠ 0) :
    let a := 3 + 2
    let b := 17 + 2
    let c := 10 + 5
    let e := 16 + 4
    a + b + c + e = 59 :=
by
  let a := 3 + 2
  let b := 17 + 2
  let c := 10 + 5
  let e := 16 + 4
  sorry

end sum_of_coefficients_l189_189032


namespace part1_part2_l189_189634

section
  variable {α : Type} [OrderedField α]

  noncomputable def a (n : ℕ) : α := sorry -- Define the sequence a_n
  def S (n : ℕ) : α := (Finset.range n).sum (λ i => a (i + 1))

  -- Given condition
  axiom a_n_condition : ∀ n : ℕ, n > 0 → a n * (2 * S n - a n) = 1

  theorem part1 (n : ℕ) (h : n > 0) : a n < 2 * Real.sqrt n := sorry 

  theorem part2 (n : ℕ) (h : n > 0) : a n * a (n + 1) < 1 := sorry 
end

end part1_part2_l189_189634


namespace meat_needed_l189_189022

theorem meat_needed (meat_per_hamburger : ℚ) (h_meat : meat_per_hamburger = (3 : ℚ) / 8) : 
  (24 * meat_per_hamburger) = 9 :=
by
  sorry

end meat_needed_l189_189022


namespace find_tan_angle_F2_F1_B_l189_189381

-- Definitions for the points and chord lengths
def F1 : Type := ℝ × ℝ
def F2 : Type := ℝ × ℝ
def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

-- Given distances
def F1A : ℝ := 3
def AB : ℝ := 4
def BF1 : ℝ := 5

-- The angle we want to find the tangent of
def angle_F2_F1_B (F1 F2 A B : Type) : ℝ := sorry -- Placeholder for angle calculation

-- The main theorem to prove
theorem find_tan_angle_F2_F1_B (F1 F2 A B : Type) (F1A_dist : F1A = 3) (AB_dist : AB = 4) (BF1_dist : BF1 = 5) :
  angle_F2_F1_B F1 F2 A B = 1 / 7 :=
sorry

end find_tan_angle_F2_F1_B_l189_189381


namespace right_triangle_decomposition_unique_l189_189151

theorem right_triangle_decomposition_unique (A B C D: Point)
  (hABC: Triangle A B C) 
  (h_right_angle_ACB: angle A C B = 90)
  (h_perpendicular_CD: altitude C D A B):
  (similar (Triangle A C D) (Triangle A B C))
  ∧ (similar (Triangle B C D) (Triangle A B C)) ->
  (unique (draw_altitude C [A, B])) :=
sorry

end right_triangle_decomposition_unique_l189_189151


namespace semicircles_ratio_l189_189425

theorem semicircles_ratio (r : ℝ) : 
  let semicircle_area := (1/4) * π * r^2
  let circle_area := π * r^2
  in (2 * semicircle_area) / circle_area = 1 / 2 := by
  -- Assume the radius r is given
  -- Define the area of each semicircle
  -- Define the area of the circle
  -- Calculate the ratio of combined areas of the semicircles to the area of the circle
  sorry

end semicircles_ratio_l189_189425


namespace max_f_find_a_l189_189520

-- Definition of the first function and the conditions
def f (x : ℝ) : ℝ := 4^(x - 1 / 2) - 3 * 2^x + 5

-- Definition of the second function and the conditions
def g (a : ℝ) (x : ℝ) : ℝ := a^(2 * x) + 2 * a^x - 1

-- Proof statement for the first part, finding the maximum value of f(x) in the interval [-2, 2]
theorem max_f : ∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → f x ≤ 137 / 32 ∧ f (-2) = 137 / 32 := sorry

-- Proof statement for the second part, finding the value of a when g(a, x) has a maximum value of 14
theorem find_a (a : ℝ) : (∀ (x : ℝ), -2 ≤ x ∧ x ≤ 2 → g a x ≤ 14) →
  (∃ (a : ℝ), (a = Real.sqrt 3 ∨ a = Real.sqrt 3 / 3)) := 
  sorry

end max_f_find_a_l189_189520


namespace minimum_lambda_l189_189282

theorem minimum_lambda (n : ℕ) (h : n ≥ 2) : 
  ∃ λ : ℝ, (∀ (a : Fin n → ℝ) (b : ℝ), 
  λ * (Finset.univ.sum (λ i, Real.sqrt (abs (a i - b)))) + Real.sqrt (n * abs (Finset.univ.sum a)) 
  ≥ Finset.univ.sum (λ i, (Real.sqrt (abs (a i))))) ∧
  λ = (n - 1 + Real.sqrt (n - 1)) / Real.sqrt n :=
by sorry

end minimum_lambda_l189_189282


namespace tom_fruits_left_l189_189077

theorem tom_fruits_left (oranges_initial apples_initial : ℕ) (sold_fraction_oranges sold_fraction_apples : ℚ)
    (h_oranges_initial : oranges_initial = 40)
    (h_apples_initial : apples_initial = 70)
    (h_sold_fraction_oranges : sold_fraction_oranges = 1 / 4)
    (h_sold_fraction_apples : sold_fraction_apples = 1 / 2) :
  let oranges_sold := oranges_initial * sold_fraction_oranges
      apples_sold := apples_initial * sold_fraction_apples
      total_fruits_initial := oranges_initial + apples_initial
      total_fruits_sold := oranges_sold + apples_sold
      fruits_left := total_fruits_initial - total_fruits_sold in
  fruits_left = 65 :=
by
  -- Proof goes here
  sorry

end tom_fruits_left_l189_189077


namespace median_of_six_numbers_l189_189833

theorem median_of_six_numbers 
  (x : ℝ)
  (H_mean : mean_of_set {87, 85, 80, 83, 84, x} = 83.5) :
  median_of_set {87, 85, 80, 83, 84, x} = 83.5 := 
sorry

-- Additional Definitions
def mean_of_set (s : set ℝ) : ℝ := (∑ x in s, x) / s.card

def median_of_set (s : set ℝ) : ℝ := 
  let sorted_list := s.to_list.qsort (≤)
  if h : sorted_list.length = 0 then 
    0 
  else if sorted_list.length % 2 = 1 then 
    sorted_list.nth_le (sorted_list.length / 2) (by linarith)
  else
    (sorted_list.nth_le (sorted_list.length / 2 - 1) (by linarith) + 
     sorted_list.nth_le (sorted_list.length / 2) (by linarith)) / 2

end median_of_six_numbers_l189_189833


namespace math_problem_solution_l189_189363

noncomputable def problem_statement : Prop :=
  ∀ (initial_pressure : ℝ) (initial_volume : ℝ) (initial_distance : ℝ) (temp : ℝ)
    (piston_speed : ℝ) (movement_time : ℝ) (saturation_pressure_water_vapor : ℝ),
  let final_distance := initial_distance - piston_speed * movement_time in
  let final_volume := initial_volume * (final_distance / initial_distance) in
  let volume_ratio := initial_volume / final_volume in
  let final_pressure_nitrogen := volume_ratio * initial_pressure in
  let power_ratio := final_pressure_nitrogen / saturation_pressure_water_vapor in
  let total_time := 7.5 * 60 in -- convert minutes to seconds
  let interval_time := 30 in -- time in seconds
  let volume_change_interval := piston_speed * (interval_time / 60) * (initial_volume / initial_distance) in
  let work_done_vapor := saturation_pressure_water_vapor * volume_change_interval * 101325 in -- convert to Joules
  let work_threshold := 15 in
  let avg_pressure_needed := work_threshold / (volume_change_interval * (interval_time / total_time)) in
  power_ratio = 2 ∧ (work_done_vapor > work_threshold ∨ final_pressure_nitrogen > avg_pressure_needed)

theorem math_problem_solution : problem_statement :=
by sorry

end math_problem_solution_l189_189363


namespace increase_decrease_l189_189522

theorem increase_decrease (initial : ℝ) (first_inc : ℝ) (second_dec : ℝ) (final_inc : ℝ) : 
  let first_step := initial * first_inc 
  let after_first := initial + first_step
  let second_step := after_first * second_dec
  let after_second := after_first - second_step
  let third_step := after_second * final_inc
  let after_third := after_second + third_step
  after_third = 181.125 :=
by
  let initial := 150
  let first_inc := 0.40
  let second_dec := 0.25
  let final_inc := 0.15
  let first_step := (initial * first_inc)
  let after_first := (initial + first_step)
  let second_step := (after_first * second_dec)
  let after_second := (after_first - second_step)
  let third_step := (after_second * final_inc)
  let final_total := (after_third := after_second + third_step)
  sorry

end increase_decrease_l189_189522


namespace no_possible_values_for_b_l189_189036

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_possible_values_for_b_l189_189036


namespace coefficient_x2_term_l189_189240

def poly1 : Polynomial ℝ := 2 * X^3 + 5 * X^2 - 3 * X
def poly2 : Polynomial ℝ := 3 * X^2 - 4 * X - 5

theorem coefficient_x2_term :
  (poly1 * poly2).coeff 2 = -37 :=
by
  sorry

end coefficient_x2_term_l189_189240


namespace range_of_x2_plus_y2_l189_189455

theorem range_of_x2_plus_y2 (f : ℝ → ℝ)
  (h_odd : ∀ x : ℝ, f x = -f (-x))
  (h_increasing : ∀ x y : ℝ, x < y → f x < f y)
  (x y : ℝ)
  (h_inequality : f (x^2 - 6 * x) + f (y^2 - 8 * y + 24) < 0) :
  16 < x^2 + y^2 ∧ x^2 + y^2 < 36 :=
sorry

end range_of_x2_plus_y2_l189_189455


namespace trapezoid_area_l189_189188

theorem trapezoid_area (R a : ℝ) (hR : 0 < R) (ha: 0 < a) :
  let AB := (4 * R^2) / a
  in let BH := 2 * R
  in (1/2) * (AB + AB) * BH = (8 * R^3) / a :=
by
  let AB := (4 * R^2) / a
  let BH := 2 * R
  calc
    (1/2) * (AB + AB) * BH
        = (1/2) * (2 * AB) * BH : by ring
    ... = AB * BH : by ring
    ... = ((4 * R^2) / a) * (2 * R) : rfl
    ... = (8 * R^3) / a : by ring

end trapezoid_area_l189_189188


namespace min_value_frac_sum_l189_189395

theorem min_value_frac_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (3 * z / (x + 2 * y) + 5 * x / (2 * y + 3 * z) + 2 * y / (3 * x + z)) ≥ 3 / 4 :=
by
  sorry

end min_value_frac_sum_l189_189395


namespace no_tiling_triminos_l189_189204

theorem no_tiling_triminos (board_size : ℕ) (trimino_size : ℕ) (remaining_squares : ℕ) 
  (H_board : board_size = 8) (H_trimino : trimino_size = 3) (H_remaining : remaining_squares = 63) : 
  ¬ ∃ (triminos : ℕ), triminos * trimino_size = remaining_squares :=
by {
  sorry
}

end no_tiling_triminos_l189_189204


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189103

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189103


namespace consecutive_integer_sum_l189_189861

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189861


namespace arithmetic_sequence_l189_189637

theorem arithmetic_sequence (
    a : ℕ → ℕ,
    S_n : ℕ → ℕ,
    b : ℕ → ℝ,
    T_n : ℕ → ℝ,
    h1 : a 3 = 3 * a 1,
    h2 : 2 * a 2 = 8,
    h3: S_n n = n * (n + 1),
    h4: ∀ n, b n = 1 / S_n n,
    h5: T_n n = ∑ i in range (n + 1), b i
) :
    (∀ n, a n = 2 * n) ∧
    (∀ n, T_n n = n / (n + 1))
:= by
  sorry

end arithmetic_sequence_l189_189637


namespace largest_prime_divisor_211020012_7_l189_189229

-- Definition and conditions
def base_7_to_decimal (a : List ℕ) : ℕ :=
  a.reverse.zipWith (λ n p, n * 7^p) (List.range a.length).sum

def largest_prime_divisor (n : ℕ) : ℕ :=
  n.factors.prime_divisors.max

def number_base_7 : List ℕ := [2, 1, 1, 0, 2, 0, 0, 1, 2]
def number_decimal := base_7_to_decimal number_base_7
def num_prime_factorization := [5, 41, 79, 769]

-- Statement to prove
theorem largest_prime_divisor_211020012_7 :
  largest_prime_divisor number_decimal = 769 :=
by sorry

end largest_prime_divisor_211020012_7_l189_189229


namespace minimum_value_quadratic_l189_189250

theorem minimum_value_quadratic (x y : ℝ) : ∃ z : ℝ, z = 0 ∧ (∀ a b : ℝ, a^2 + 2*a*b + 2*b^2 ≥ z) :=
begin
  use 0,
  split,
  { refl },
  { intros a b,
    -- Skipping the actual proof, we just say "sorry" here
    sorry
  }
end

end minimum_value_quadratic_l189_189250


namespace inequality_solution_set_conditions_l189_189652

theorem inequality_solution_set_conditions (a b c : ℝ) 
  (h1 : ∀ x, ax^2 + bx + c > 0 ↔ x ∈ set.Ioo (-3 : ℝ) 2) :
  a < 0 ∧ 
  (¬ ∀ x, bx - c > 0 ↔ x < 6) ∧ 
  a - b + c > 0 ∧ 
  (¬ ∀ x, cx^2 - bx + a < 0 ↔ (x ∈ set.Ioo (- ∞) (-1/2 : ℝ) ∪ set.Ioo (1/3 : ℝ) ∞)) :=
by
  sorry

end inequality_solution_set_conditions_l189_189652


namespace smallest_b_value_l189_189387

theorem smallest_b_value (a b : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a - b = 8)
  (h4 : Nat.gcd ((a^3 + b^3) / (a + b)) (a * b) = 16) : b = 4 :=
begin
  sorry
end

#quit

end smallest_b_value_l189_189387


namespace stratified_sampling_example_l189_189336

variable (total_students : ℕ) (male_students : ℕ) (female_students : ℕ) (sample_size : ℕ)

-- Given conditions
def is_stratified_sample (male_students female_students sample_size : ℕ) (sample_males sample_females : ℕ) : Prop :=
  male_students + female_students = total_students ∧
  sample_males + sample_females = sample_size ∧
  sample_males = (male_students * sample_size) / total_students ∧
  sample_females = (female_students * sample_size) / total_students

-- Prove the specific example
theorem stratified_sampling_example :
  is_stratified_sample 20 30 10 4 6 :=
by
  sorry

end stratified_sampling_example_l189_189336


namespace initial_riding_time_l189_189403

theorem initial_riding_time (t : ℝ) (h1 : t * 60 + 90 + 30 + 120 = 270) : t * 60 = 30 :=
by sorry

end initial_riding_time_l189_189403


namespace pieces_not_chewed_l189_189401

theorem pieces_not_chewed : 
  (8 * 7 - 54) = 2 := 
by 
  sorry

end pieces_not_chewed_l189_189401


namespace b_plus_c_plus_d_eq_minus1_l189_189391

variable {S : Set ℂ}
variable {a b c d : ℂ}

theorem b_plus_c_plus_d_eq_minus1 
  (h1 : a^2 = 1) 
  (h2 : b^2 = 1) 
  (h3 : c^2 = b) 
  (h4 : ∀ x y ∈ {a, b, c, d}, x * y ∈ {a, b, c, d}) : 
  b + c + d = (-1 : ℂ) :=
sorry

end b_plus_c_plus_d_eq_minus1_l189_189391


namespace impossible_to_place_tokens_l189_189364

theorem impossible_to_place_tokens :
  ∀ (W : ℕ), ∀ (circle : finset ℕ), 
  circle.card = 200 → 
  (∀ (i : ℕ), i ∈ circle → (i + 100) % 200 ∈ circle) → 
  (∀ (i : ℕ), i ∈ circle → ¬ ((i + 1) ∈ circle ∧ (i + 2) % 200 ∈ circle)) → 
  false :=
  sorry

end impossible_to_place_tokens_l189_189364


namespace line_J_MK_l189_189516

namespace Geometry

-- Definitions of circles and points
variables {Γ₁ Γ₂ : Circle} {A B K M P C Q L J O₂ : Point} 
variables [TangentsIntersection K Γ₁ A B] [VariablePoint M Γ₁ A B]
variables [SecondIntersection P (Line.mk M A) Γ₂ A] [SecondIntersection C (Line.mk M K) Γ₁ K]
variables [SecondIntersection Q (Line.mk A C) Γ₂ C] [FixedPoint L]
variables [Midpoint J P Q] [Center O₂ Γ₂]
variables [Concyclic O₂ L B K J]

-- The goal is to prove that J lies on the line (MK)
theorem line_J_MK : LiesOnLine J (Line.mk M K) :=
sorry

end Geometry

end line_J_MK_l189_189516


namespace binary_to_decimal_conversion_l189_189443

theorem binary_to_decimal_conversion : (1 * 2^2 + 1 * 2^1 + 0 * 2^0 = 6) := by
  sorry

end binary_to_decimal_conversion_l189_189443


namespace solve_system_of_trig_eqns_l189_189431

open Real

theorem solve_system_of_trig_eqns {x y : ℝ} :
  (sin(x) ^ 2 = sin(y) ∧ cos(x) ^ 4 = cos(y)) ↔
  (∃ l m : ℤ, x = l * π ∧ y = 2 * m * π) ∨
  (∃ l m : ℤ, x = l * π + π / 2 ∧ y = 2 * m * π + π / 2) :=
by
  sorry

end solve_system_of_trig_eqns_l189_189431


namespace inequality_positive_real_l189_189422

theorem inequality_positive_real (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 1) :
  (x^2 + y^2) / z + (y^2 + z^2) / x + (z^2 + x^2) / y ≥ 2 :=
sorry

end inequality_positive_real_l189_189422


namespace parabola_expression_l189_189456

theorem parabola_expression (a b c : ℝ) (h1 : ∀ x, (x = -1 → ax^2 + bx + c = 0) ∧ (x = 2 → ax^2 + bx + c = 0)) 
  (h2 : ∀ x, ax^2 + bx + c = -2x^2) : 
  (y = -2x^2 + 2x + 4) :=
by
  sorry

end parabola_expression_l189_189456


namespace f_value_at_3_l189_189271

def f (x : ℝ) := 2 * (x + 1) + 1

theorem f_value_at_3 : f 3 = 9 :=
by sorry

end f_value_at_3_l189_189271


namespace milk_fraction_correct_l189_189803

def fraction_of_milk_in_coffee_cup (coffee_initial : ℕ) (milk_initial : ℕ) : ℚ :=
  let coffee_transferred := coffee_initial / 3
  let milk_cup_after_transfer := milk_initial + coffee_transferred
  let coffee_left := coffee_initial - coffee_transferred
  let total_mixed := milk_cup_after_transfer
  let transfer_back := total_mixed / 2
  let coffee_back := transfer_back * (coffee_transferred / total_mixed)
  let milk_back := transfer_back * (milk_initial / total_mixed)
  let coffee_final := coffee_left + coffee_back
  let milk_final := milk_back
  milk_final / (coffee_final + milk_final)

theorem milk_fraction_correct (coffee_initial : ℕ) (milk_initial : ℕ)
  (h_coffee : coffee_initial = 6) (h_milk : milk_initial = 3) :
  fraction_of_milk_in_coffee_cup coffee_initial milk_initial = 3 / 13 :=
by
  sorry

end milk_fraction_correct_l189_189803


namespace square_area_and_perimeter_l189_189085

noncomputable def distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2)

theorem square_area_and_perimeter (A B C D : ℝ × ℝ) 
  (hxA : A = (0, 0)) (hxB : B = (-5, -3)) (hxC : C = (-4, -8)) (hxD : D = (1, -5)) : 
  let AB := distance A.1 A.2 B.1 B.2 in
  (AB^2 = 34 ∧ 4 * AB = 4 * real.sqrt 34) :=
by
  sorry

end square_area_and_perimeter_l189_189085


namespace number_of_monkeys_is_one_l189_189338

def AnimalType := {parrot : Prop, monkey : Prop, snake : Prop}

variables (Alex Bob Charlie Dave Eve : AnimalType)

axiom Alex_statement : Alex.parrot ↔ (Eve.parrot ↔ Alex.parrot)
axiom Bob_statement : Bob.parrot ↔ (¬(Alex.monkey ∧ Bob.monkey ∧ Charlie.monkey ∧ Dave.monkey ∧ Eve.monkey))
axiom Charlie_statement : Charlie.parrot ↔ (¬(Charlie.parrot ↔ Dave.parrot))
axiom Dave_statement : Dave.parrot ↔ Alex.parrot
axiom Eve_statement : Eve.parrot ↔ (Charlie.monkey)

theorem number_of_monkeys_is_one : 
  (if Alex.monkey then 1 else 0) + 
  (if Bob.monkey then 1 else 0) + 
  (if Charlie.monkey then 1 else 0) + 
  (if Dave.monkey then 1 else 0) + 
  (if Eve.monkey then 1 else 0) = 1 :=
sorry

end number_of_monkeys_is_one_l189_189338


namespace largest_positive_integer_not_sum_of_36_and_composite_l189_189106

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l189_189106


namespace product_is_negative_probability_l189_189965

theorem product_is_negative_probability :
  let S := {-4, -3, -2, 0, 0, 1, 2, 3} in
  ( fintype.card { s : finset Int // s.card = 3 ∧ s ⊆ S ∧ s.prod id < 0 } : ℚ ) 
  / ( fintype.card { s : finset Int // s.card = 3 ∧ s ⊆ S } : ℚ ) = 9 / 56 :=
by
  sorry

end product_is_negative_probability_l189_189965


namespace train_speed_in_kmph_l189_189182

-- Define the conditions as constants
constant train_length : ℕ := 155
constant bridge_length : ℕ := 220
constant crossing_time : ℕ := 30

-- Define the speed conversion factors
def m_per_s_to_km_per_h (mps : ℕ) : ℕ := mps * 36 / 10

-- Define the total distance covered by the train while crossing the bridge
def total_distance : ℕ := train_length + bridge_length

-- Calculate the speed of the train in m/s
def speed_m_per_s : ℕ := total_distance / crossing_time

-- Calculate the speed of the train in km/hr
def speed_km_per_h : ℕ := m_per_s_to_km_per_h speed_m_per_s

-- Statement to be proved
theorem train_speed_in_kmph : speed_km_per_h = 45 := by
  sorry

end train_speed_in_kmph_l189_189182


namespace polynomial_diff_l189_189291

theorem polynomial_diff (m n : ℤ) (h1 : 2 * m + 2 = 0) (h2 : n - 4 = 0) :
  (4 * m^2 * n - 3 * m * n^2) - 2 * (m^2 * n + m * n^2) = -72 := 
by {
  -- This is where the proof would go, so we put sorry for now
  sorry
}

end polynomial_diff_l189_189291


namespace proof_problem_l189_189298

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x - Real.pi / 6)

theorem proof_problem
  (x1 x2 : ℝ)
  (h1 : x1 ∈ Ioo (-17 * Real.pi / 12) (-2 * Real.pi / 3))
  (h2 : x2 ∈ Ioo (-17 * Real.pi / 12) (-2 * Real.pi / 3))
  (h3 : x1 ≠ x2)
  (h4 : f x1 = f x2) :
  f (x1 + x2) = -1 :=
sorry

end proof_problem_l189_189298


namespace median_mode_unchanged_l189_189610

open List

-- Define the original list of donations and the updated list after the additional donation
def original_donations := [5, 3, 6, 5, 10]
def updated_donations := [3, 5, 5, 6, 20]

-- Define functions to calculate the median and mode of a list
def median (l : List ℕ) : ℕ :=
  let sorted := sort l in
  sorted.nth_le (length l / 2) (by sorry)

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ m n, if l.count n > l.count m then n else m) 0

-- Define a theorem that the median and mode do not change after the additional donation
theorem median_mode_unchanged :
  median original_donations = median updated_donations ∧
  mode original_donations = mode updated_donations :=
by
  sorry

end median_mode_unchanged_l189_189610


namespace find_y_l189_189597

noncomputable def series_sum (y : ℝ) : ℝ :=
2 + 7 * y + 12 * y^2 + 17 * y^3 + ∑' n : ℕ, (5 + 5 * n) * y^(n + 4)

theorem find_y (y : ℝ) : series_sum y = 100 → y = 4 / 5 := sorry

end find_y_l189_189597


namespace quadratic_h_value_l189_189058

theorem quadratic_h_value (p q r h : ℝ) (hq : p*x^2 + q*x + r = 5*(x - 3)^2 + 15):
  let new_quadratic := 4* (p*x^2 + q*x + r)
  let m := 20
  let k := 60
  new_quadratic = m * (x - h) ^ 2 + k → h = 3 := by
  sorry

end quadratic_h_value_l189_189058


namespace problems_on_each_worksheet_l189_189185

-- Define the conditions
def worksheets_total : Nat := 9
def worksheets_graded : Nat := 5
def problems_left : Nat := 16

-- Define the number of remaining worksheets and the problems per worksheet
def remaining_worksheets : Nat := worksheets_total - worksheets_graded
def problems_per_worksheet : Nat := problems_left / remaining_worksheets

-- Prove the number of problems on each worksheet
theorem problems_on_each_worksheet : problems_per_worksheet = 4 :=
by
  sorry

end problems_on_each_worksheet_l189_189185


namespace average_marks_combined_l189_189178

theorem average_marks_combined (P C M B E : ℕ) (h : P + C + M + B + E = P + 280) : 
  (C + M + B + E) / 4 = 70 :=
by 
  sorry

end average_marks_combined_l189_189178


namespace largest_non_sum_of_36_and_composite_l189_189093

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l189_189093


namespace sum_of_consecutive_integers_with_product_812_l189_189906

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189906


namespace range_of_f_value_of_f_B_l189_189680

-- Definitions from problem conditions
def m (x : ℝ) : ℝ × ℝ := (real.sqrt 3 * real.sin x, 1 - real.sqrt 2 * real.sin x)
def n (x : ℝ) : ℝ × ℝ := (2 * real.cos x, 1 + real.sqrt 2 * real.sin x)
def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

-- Range of f(x) in the given interval
theorem range_of_f : ∀ x ∈ Icc 0 (π / 2), -1 ≤ f x ∧ f x ≤ 2 := by
  sorry

-- Definitions and conditions for triangle ABC
variable {a b c A B C : ℝ}
variable (ha : a > 0) (hb : b > 0) (hc : c > 0)
variable (h_ab : b / a = real.sqrt 3)
variable (h_sin : real.sin B * real.cos A / real.sin A = 2 - real.cos B)
variable (h_sine_rule : real.sin A / a = real.sin B / b = real.sin C / c)

-- Value of f(B)
theorem value_of_f_B (h_B_eq_pi_div_3 : B = π / 3) : f B = 1 := by
  sorry

end range_of_f_value_of_f_B_l189_189680


namespace jerry_task_duration_l189_189370

def earnings_per_task : ℕ := 40
def hours_per_day : ℕ := 10
def days_per_week : ℕ := 7
def total_earnings : ℕ := 1400

theorem jerry_task_duration :
  (10 * 7 = 70) →
  (1400 / 40 = 35) →
  (70 / 35 = 2) →
  (total_earnings / earnings_per_task = (hours_per_day * days_per_week) / h) →
  h = 2 :=
by
  intros h1 h2 h3 h4
  -- proof steps (omitted)
  sorry

end jerry_task_duration_l189_189370


namespace solve_for_y_l189_189026

theorem solve_for_y (y : ℝ) : 
  (4^(9^y) = 9^(4^y)) ↔ (y = real.log2(real.log2(3)) / real.log2(2.25)) :=
by
  sorry

end solve_for_y_l189_189026


namespace exists_divisible_by_11_sum_l189_189133

-- Define the sum of digits function
def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

-- Prove that for any set of 39 consecutive positive integers,
-- there is at least one integer in the set such that the sum of its digits is divisible by 11.
theorem exists_divisible_by_11_sum (n : ℕ) :
  ∃ i, i ∈ list.range 39 ∧ (sum_of_digits (n + i)) % 11 = 0 :=
by
  sorry

end exists_divisible_by_11_sum_l189_189133


namespace factor_of_quadratic_implies_m_value_l189_189144

theorem factor_of_quadratic_implies_m_value (m : ℤ) : (∀ x : ℤ, (x + 6) ∣ (x^2 - m * x - 42)) → m = 1 := by
  sorry

end factor_of_quadratic_implies_m_value_l189_189144


namespace seq_sum_to_2015_l189_189082

def seq_sum (n : ℕ) : ℤ :=
  if h : n % 4 = 3 then (n + 1) + (n + 2) - (n + 3)
  else 0

theorem seq_sum_to_2015 : 
  let full_groups := 503 * -4 in
  let remainder := 2013 + 2014 - 2015 in
  full_groups + remainder = -1 :=
by
  sorry

end seq_sum_to_2015_l189_189082


namespace hyperbola_focus_l189_189585

theorem hyperbola_focus :
  ∃ x y : ℝ, 
    (-2 * x^2 + 3 * y^2 - 8 * x - 24 * y + 8 = 0) ∧ 
    (x = -2) ∧ 
    (y = 4 + real.sqrt (80 / 3)) :=
begin
  -- The intuition and steps for solving the problem go here. 
  sorry
end

end hyperbola_focus_l189_189585


namespace inequality_true_for_all_real_l189_189806

theorem inequality_true_for_all_real (a : ℝ) : 
  3 * (1 + a^2 + a^4) ≥ (1 + a + a^2)^2 :=
sorry

end inequality_true_for_all_real_l189_189806


namespace cow_cost_calculation_l189_189477

constant hearts_per_card : ℕ := 4
constant cards_in_deck : ℕ := 52
constant cost_per_cow : ℕ := 200

def total_hearts : ℕ := hearts_per_card * cards_in_deck
def number_of_cows : ℕ := 2 * total_hearts
def total_cost_of_cows : ℕ := number_of_cows * cost_per_cow

theorem cow_cost_calculation :
  total_cost_of_cows = 83200 := by
  -- Placeholder proof
  sorry

end cow_cost_calculation_l189_189477


namespace convex_quadrilateral_probability_l189_189638

noncomputable def probability_convex_quadrilateral (n : ℕ) : ℚ :=
  if n = 6 then (Nat.choose 6 4 : ℚ) / (Nat.choose 15 4 : ℚ) else 0

theorem convex_quadrilateral_probability :
  probability_convex_quadrilateral 6 = 1 / 91 :=
by
  sorry

end convex_quadrilateral_probability_l189_189638


namespace paint_usage_total_l189_189559

theorem paint_usage_total
  (extra_large canvases_3 : ℕ) (large_canvases_5 : ℕ) (medium_canvases_6 : ℕ) (small_canvases_8 : ℕ)
  (red_extra : ℕ → ℕ := λ n, 5 * n) (red_large : ℕ → ℕ := λ n, 4 * n)
  (red_medium : ℕ → ℕ := λ n, 3 * n) (red_small : ℕ → ℕ := λ n, 1 * n)
  (blue_extra : ℕ → ℕ := λ n, 3 * n) (blue_large : ℕ → ℕ := λ n, 2 * n)
  (blue_medium : ℕ → ℕ := λ n, 1 * n) (blue_small : ℕ → ℕ := λ n, 1 * n)
  (yellow_extra : ℕ → ℕ := λ n, 2 * n) (yellow_large : ℕ → ℕ := λ n, 3 * n)
  (yellow_medium : ℕ → ℕ := λ n, 2 * n) (yellow_small : ℕ → ℕ := λ n, 1 * n)
  (green_extra : ℕ → ℕ := λ n, 1 * n) (green_large : ℕ → ℕ := λ n, 1 * n)
  (green_medium : ℕ → ℕ := λ n, 1 * n) (green_small : ℕ → ℕ := λ n, 1 * n) :
  red_extra 3 + red_large 5 + red_medium 6 + red_small 8 = 61 ∧
  blue_extra 3 + blue_large 5 + blue_medium 6 + blue_small 8 = 33 ∧
  yellow_extra 3 + yellow_large 5 + yellow_medium 6 + yellow_small 8 = 41 ∧
  green_extra 3 + green_large 5 + green_medium 6 + green_small 8 = 22 :=
by
  -- Here the different color paint usage calculation will be proved mathematically.
  sorry

end paint_usage_total_l189_189559


namespace proposition_equivalence_l189_189508

open Classical

theorem proposition_equivalence
  (p q : Prop) :
  ¬(p ∨ q) ↔ (¬p ∧ ¬q) :=
by sorry

end proposition_equivalence_l189_189508


namespace math_problem_l189_189358

open Classical
open Real

variables {A B C D E F L : Point}

-- Definitions of the problem's conditions
def is_triangle (A B C : Point) : Prop := 
  ∃ ABC : Triangle,
    ABC.A = A ∧ ABC.B = B ∧ ABC.C = C

def midpoint (A B F : Point) : Prop := 
  dist A F = dist B F

def circumscribed_diameter (A B C D E F : Point) : Prop :=
  ∃ (circle : Circle),
    circle_contains circle A ∧
    circle_contains circle B ∧
    circle_contains circle C ∧
    circle_contains circle D ∧
    circle_contains circle E ∧
    D ≠ E ∧
    is_diameter circle D E ∧
    on_same_side C E A B

def parallel (C A B DE : Line) : Prop :=
  ∃ (l1 : Line),
    l1.contains A ∧
    l1.contains B ∧
    parallel_lines l1 DE ∧
    ∃ (l2 : Line),
      l2.contains C ∧
      line_parallel l1 l2

def intersects (C DE L : Point) : Prop :=
  ∃ (line : Line),
    line.contains C ∧
    line.contains L ∧
    intersect_line_point DE line L

def proof_statement (A B C D E F L : Point) : Prop :=
  (dist A C + dist B C) ^ 2 = 4 * (dist D L) * (dist E F)

-- Lean statement
theorem math_problem (A B C D E F L : Point) 
  (h_triangle : is_triangle A B C) 
  (h_AC_gt_BC : dist A C > dist B C) 
  (h_midpoint : midpoint A B F) 
  (h_diameter : circumscribed_diameter A B C D E F) 
  (h_parallel : parallel C A B (mk_line D E)) 
  (h_intersect : intersects C (mk_line D E) L) :
  proof_statement A B C D E F L :=
sorry

end math_problem_l189_189358


namespace min_distance_at_sqrt2_div_2_l189_189673

noncomputable def f (x : ℝ) : ℝ := x^2
noncomputable def g (x : ℝ) : ℝ := Real.log x

theorem min_distance_at_sqrt2_div_2 :
  ∃ t : ℝ, (∀ x : ℝ, abs (f x - g x) ≥ abs (f (Real.sqrt (2) / 2) - g (Real.sqrt (2) / 2))) ∧
    t = Real.sqrt (2) / 2 := 
begin
  use Real.sqrt (2) / 2,
  split,
  { intro x,
    sorry, -- This part would contain the necessary proof steps.
  },
  { refl, }
end

end min_distance_at_sqrt2_div_2_l189_189673


namespace toothpicks_pattern_100th_stage_l189_189827

theorem toothpicks_pattern_100th_stage :
  let a_1 := 5
  let d := 4
  let n := 100
  (a_1 + (n - 1) * d) = 401 := by
  sorry

end toothpicks_pattern_100th_stage_l189_189827


namespace contrapositive_statement_l189_189690

def is_even (n : ℕ) : Prop := n % 2 = 0
def is_odd (n : ℕ) : Prop := ¬is_even n

theorem contrapositive_statement (a b : ℕ) :
  (¬(is_odd a ∧ is_odd b) ∧ ¬(is_even a ∧ is_even b)) → ¬is_even (a + b) :=
by
  sorry

end contrapositive_statement_l189_189690


namespace sum_of_consecutive_integers_with_product_812_l189_189924

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189924


namespace find_common_difference_l189_189280

def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Variance of first 5 terms is given as 2
def variance_is_two (a : ℕ → ℝ) : Prop :=
  let mean := (a 0 + a 1 + a 2 + a 3 + a 4) / 5 in
  (1 / 5) * ((a 0 - mean)^2 + (a 1 - mean)^2 + (a 2 - mean)^2 + (a 3 - mean)^2 + (a 4 - mean)^2) = 2

theorem find_common_difference (a : ℕ → ℝ) (d : ℝ)
  (h_arith : is_arithmetic_sequence a d)
  (h_var : variance_is_two a) :
  d = 1 ∨ d = -1 :=
sorry

end find_common_difference_l189_189280


namespace parabola_coefficients_l189_189581

theorem parabola_coefficients
    (vertex : (ℝ × ℝ))
    (passes_through : (ℝ × ℝ))
    (vertical_axis_of_symmetry : Prop)
    (hv : vertex = (2, -3))
    (hp : passes_through = (0, 1))
    (has_vertical_axis : vertical_axis_of_symmetry) :
    ∃ a b c : ℝ, ∀ x : ℝ, (x = 0 → (a * x^2 + b * x + c = 1)) ∧ (x = 2 → (a * x^2 + b * x + c = -3)) := sorry

end parabola_coefficients_l189_189581


namespace least_marbles_l189_189549

/-- 
  Prove that the least number of marbles that can be divided equally among
  2, 3, 4, 5, 6, and 8 friends is 120.
-/
theorem least_marbles : ∃ (n : ℕ), (2 ∣ n) ∧ (3 ∣ n) ∧ (4 ∣ n) ∧ (5 ∣ n) ∧ (6 ∣ n) ∧ (8 ∣ n) ∧ n = 120 :=
by
  have h1 : 2 ∣ 120 := by norm_num,
  have h2 : 3 ∣ 120 := by norm_num,
  have h3 : 4 ∣ 120 := by norm_num,
  have h4 : 5 ∣ 120 := by norm_num,
  have h5 : 6 ∣ 120 := by norm_num,
  have h6 : 8 ∣ 120 := by norm_num,
  existsi 120,
  split,
  -- prove all divisibility requirements
  exact h1,
  split,
  exact h2,
  split,
  exact h3,
  split,
  exact h4,
  split,
  exact h5,
  split,
  exact h6,
  -- prove equality
  rfl

end least_marbles_l189_189549


namespace sum_of_consecutive_integers_with_product_812_l189_189922

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189922


namespace prove_ln10_order_l189_189285

def ln10_order_proof : Prop :=
  let a := Real.log 10
  let b := Real.log 100
  let c := (Real.log 10) ^ 2
  c > b ∧ b > a

theorem prove_ln10_order : ln10_order_proof := 
sorry

end prove_ln10_order_l189_189285


namespace exponent_relation_l189_189691

theorem exponent_relation (a : ℝ) (m n : ℕ) (h1 : a^m = 9) (h2 : a^n = 3) : a^(m - n) = 3 := 
sorry

end exponent_relation_l189_189691


namespace sufficient_but_not_necessary_condition_l189_189639

noncomputable def alpha_sufficient_not_necessary (k : ℤ) (α : ℝ) : Prop :=
  α = (π / 6 + 2 * k * π) → cos (2 * α) = 1 / 2

theorem sufficient_but_not_necessary_condition (k : ℤ) (α : ℝ) :
  (∃ k : ℤ, α = π / 6 + 2 * k * π) ∧ ¬(α = π / 6 + 2 * k * π ↔ cos (2 * α) = 1 / 2) :=
sorry

end sufficient_but_not_necessary_condition_l189_189639


namespace exists_pair_with_diff_9_exists_pair_with_diff_10_exists_pair_with_diff_12_exists_pair_with_diff_13_possible_no_pair_with_diff_11_l189_189558

theorem exists_pair_with_diff_9 (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 100) (h₂ : s.card = 55) :
  ∃ x y ∈ s, x ≠ y ∧ (x - y).natAbs = 9 :=
sorry

theorem exists_pair_with_diff_10 (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 100) (h₂ : s.card = 55) :
  ∃ x y ∈ s, x ≠ y ∧ (x - y).natAbs = 10 :=
sorry

theorem exists_pair_with_diff_12 (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 100) (h₂ : s.card = 55) :
  ∃ x y ∈ s, x ≠ y ∧ (x - y).natAbs = 12 :=
sorry

theorem exists_pair_with_diff_13 (s : Finset ℕ) (h₁ : ∀ x ∈ s, x ≤ 100) (h₂ : s.card = 55) :
  ∃ x y ∈ s, x ≠ y ∧ (x - y).natAbs = 13 :=
sorry

theorem possible_no_pair_with_diff_11 :
  ∃ s : Finset ℕ, (∀ x ∈ s, x ≤ 100) ∧ s.card = 55 ∧ (∀ x y ∈ s, x ≠ y → (x - y).natAbs ≠ 11) :=
sorry

end exists_pair_with_diff_9_exists_pair_with_diff_10_exists_pair_with_diff_12_exists_pair_with_diff_13_possible_no_pair_with_diff_11_l189_189558


namespace abs_z_l189_189614

-- Define the problem as a Lean definition. 

theorem abs_z (z : ℂ) (hz : (1 - complex.I) / z = 1 + complex.I) : complex.abs z = 1 :=
sorry

end abs_z_l189_189614


namespace quadratic_inequality_solution_l189_189954

theorem quadratic_inequality_solution
  (a b c : ℝ)
  (h1: ∀ x : ℝ, (-1/3 < x ∧ x < 2) → (ax^2 + bx + c) > 0)
  (h2: a < 0):
  ∀ x : ℝ, ((-3 < x ∧ x < 1/2) ↔ (cx^2 + bx + a) < 0) :=
by
  sorry

end quadratic_inequality_solution_l189_189954


namespace square_field_area_l189_189439

theorem square_field_area (s : ℕ) (area cost_per_meter total_cost gate_width : ℕ):
  area = s^2 →
  cost_per_meter = 2 →
  total_cost = 1332 →
  gate_width = 1 →
  (4 * s - 2 * gate_width) * cost_per_meter = total_cost →
  area = 27889 :=
by
  intros h_area h_cost_per_meter h_total_cost h_gate_width h_equation
  sorry

end square_field_area_l189_189439


namespace quadratic_equation_identify_l189_189123

theorem quadratic_equation_identify {a b c x : ℝ} :
  ((3 - 5 * x^2 = x) ↔ true) ∧
  ((3 / x + x^2 - 1 = 0) ↔ false) ∧
  ((a * x^2 + b * x + c = 0) ↔ (a ≠ 0)) ∧
  ((4 * x - 1 = 0) ↔ false) :=
by
  sorry

end quadratic_equation_identify_l189_189123


namespace scientific_notation_of_2200_l189_189828

-- Define scientific notation criteria
def is_scientific_notation (a : ℝ) (n : ℤ) (x : ℝ) : Prop :=
  x = a * 10^n ∧ 1 ≤ a ∧ a < 10

-- Problem statement
theorem scientific_notation_of_2200 : ∃ (a : ℝ) (n : ℤ), is_scientific_notation a n 2200 ∧ a = 2.2 ∧ n = 3 :=
by {
  -- Proof can be added here.
  sorry
}

end scientific_notation_of_2200_l189_189828


namespace sum_of_consecutive_integers_with_product_812_l189_189944

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189944


namespace parallelogram_diagonal_square_condition_l189_189052

theorem parallelogram_diagonal_square_condition (a b m n : ℝ) (h1 : m^2 = a^2 + b^2 + 2*a*b*(Real.cos (45 * Real.pi / 180)))
                                                  (h2 : n^2 = a^2 + b^2 - 2*a*b*(Real.cos (45 * Real.pi / 180))) :
  a^4 + b^4 = m^2 * n^2 ↔ Real.cos (45 * Real.pi / 180) = 1 / sqrt 2 :=
by
  sorry

end parallelogram_diagonal_square_condition_l189_189052


namespace consecutive_integers_sum_l189_189903

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189903


namespace bumper_car_line_l189_189963
noncomputable theory

theorem bumper_car_line :
    ∃ t : ℕ → ℕ, 
    t 0 = 9 ∧ 
    (∀ n, (t (2 * n + 1) = max (0) (t (2 * n) - 6)) ∧ 
          (t (2 * n + 2) = t (2 * n + 1) + 3)) ∧
    ∀ n ≤ 6, t n ≤ 9 :=
by tidy [floor]

end bumper_car_line_l189_189963


namespace fixed_point_of_exponential_function_l189_189049

theorem fixed_point_of_exponential_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : 
  ∃ (x y : ℝ), (x = 3 ∧ y = 2 ∧ y = a^(x - 3) + 1) :=
by
  use 3
  use 2
  split
  { refl }
  split
  { refl }
  { sorry }

end fixed_point_of_exponential_function_l189_189049


namespace largest_positive_integer_not_sum_of_36_and_composite_l189_189104

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l189_189104


namespace consecutive_integer_product_sum_l189_189927

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189927


namespace largest_not_sum_of_36_and_composite_l189_189112

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l189_189112


namespace sector_COD_ratio_l189_189791

def angle_AOC : ℝ := 40 -- angle in degrees
def angle_DOB : ℝ := 30 -- angle in degrees
def angle_AOB : ℝ := 180 -- angle in degrees, since AB is a diameter

theorem sector_COD_ratio (angle_AOC = 40) (angle_DOB = 30) (angle_AOB = 180) : (angle_AOB - angle_AOC + angle_DOB) / 360 = 17 / 36 :=
by
  -- this is where the proof would go, but we are omitting it with sorry
  sorry

end sector_COD_ratio_l189_189791


namespace dog_run_distance_l189_189160

theorem dog_run_distance (r : ℝ) (h : r = 10) : dist = 10 * Real.pi :=
by
  sorry

end dog_run_distance_l189_189160


namespace problem_l189_189839

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189839


namespace prob_exactly_M_laws_included_in_Concept_expected_laws_in_Concept_l189_189139

variables (K N M : ℕ) (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

-- Part (a)
def prob_M_laws_included :
  ℝ :=
  Nat.choose K M * (1 - (1 - p)^N)^M * (1 - p)^N^(K - M)

theorem prob_exactly_M_laws_included_in_Concept :
  prob_M_laws_included K N M p h = Nat.choose K M * (1 - (1 - p)^N)^M * (1 - p)^N^(K - M) :=
sorry

-- Part (b)
def expected_num_laws_included : ℝ :=
  K * (1 - (1 - p)^N)

theorem expected_laws_in_Concept :
  expected_num_laws_included K N p h = K * (1 - (1 - p)^N) :=
sorry

end prob_exactly_M_laws_included_in_Concept_expected_laws_in_Concept_l189_189139


namespace ellipse_properties_l189_189281

noncomputable def standard_equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2) / 4 + y^2 = 1

noncomputable def trajectory_equation_midpoint (x y : ℝ) : Prop :=
  ((2 * x - 1)^2) / 4 + (2 * y - 1 / 2)^2 = 1

theorem ellipse_properties :
  (∀ x y : ℝ, standard_equation_of_ellipse x y) ∧
  (∀ x y : ℝ, trajectory_equation_midpoint x y) :=
by
  sorry

end ellipse_properties_l189_189281


namespace probability_of_double_l189_189164

def is_double (domino : ℕ × ℕ) : Prop := domino.1 = domino.2

def all_dominos : finset (ℕ × ℕ) :=
finset.product (finset.range 13) (finset.range 13)

def double_dominos : finset (ℕ × ℕ) :=
all_dominos.filter is_double

theorem probability_of_double :
  (double_dominos.card : ℚ) / (all_dominos.card : ℚ) = 1 / 7 :=
by
  sorry

end probability_of_double_l189_189164


namespace intersection_P_M_equals_I_l189_189148

def P := {x : ℤ | 0 ≤ x ∧ x < 3}
def M := {x : ℝ | x^2 ≤ 9}
def I := {0, 1, 2 : ℤ}

theorem intersection_P_M_equals_I : {x : ℤ | x ∈ P ∧ (x : ℝ) ∈ M} = I := by
  sorry

end intersection_P_M_equals_I_l189_189148


namespace grid_sum_non_negative_integers_l189_189343

theorem grid_sum_non_negative_integers (m : ℕ) (grid : ℕ → ℕ → ℕ)
  (h_cond : ∀ i j, grid i j = 0 → (∑ k, grid i k) + (∑ k, grid k j) ≥ m) :
  (∑ i, ∑ j, grid i j) ≥ m^2 / 2 :=
begin
  sorry
end

end grid_sum_non_negative_integers_l189_189343


namespace sum_of_consecutive_integers_l189_189880

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189880


namespace only_option_A_is_quadratic_l189_189121

def is_quadratic (expr : ℚ[X]) : Prop :=
  ∃ a b c : ℚ, a ≠ 0 ∧ expr = a * X^2 + b * X + c

def option_A := -5 * X^2 - X + 3
def option_B := (3 / X) + X^2 - 1
def option_C (a b c : ℚ) := a * X^2 + b * X + c
def option_D := 4 * X - 1

theorem only_option_A_is_quadratic :
  is_quadratic option_A ∧ 
  ¬ is_quadratic option_B ∧
  ∀ (a b c : ℚ), ¬ is_quadratic (option_C a b c) ∧
  ¬ is_quadratic option_D :=
by
  sorry

end only_option_A_is_quadratic_l189_189121


namespace avg_annual_growth_rate_profit_exceeds_340_l189_189195

variable (P2018 P2020 : ℝ)
variable (r : ℝ)

theorem avg_annual_growth_rate :
    P2018 = 200 → P2020 = 288 →
    (1 + r)^2 = P2020 / P2018 →
    r = 0.2 :=
by
  intros hP2018 hP2020 hGrowth
  sorry

theorem profit_exceeds_340 (P2020 : ℝ) (r : ℝ) :
    P2020 = 288 → r = 0.2 →
    P2020 * (1 + r) > 340 :=
by
  intros hP2020 hr
  sorry

end avg_annual_growth_rate_profit_exceeds_340_l189_189195


namespace directrix_of_parabola_l189_189288

theorem directrix_of_parabola (f : ℝ → ℝ)
  (h_f : ∀ (x : ℝ), f x = x^3 + x^2 + x + 3)
  (p : ℝ)
  (h_tangent : ∀ x, (deriv f 1 = 2) ∧ f (-1) = 2)
  (h_parabola : ∀ x y, y = 2 * p * x^2 → ∃ x0, 2 * p * x0 ^ 2 = 2 * x + 4) :
  ∃ a : ℝ, a = 1 := 
sorry

end directrix_of_parabola_l189_189288


namespace total_worth_of_presents_l189_189747

-- Definitions of the costs
def costOfRing : ℕ := 4000
def costOfCar : ℕ := 2000
def costOfBracelet : ℕ := 2 * costOfRing

-- Theorem statement
theorem total_worth_of_presents : 
  costOfRing + costOfCar + costOfBracelet = 14000 :=
begin
  -- by using the given definitions and the provided conditions, we assert the statement
  sorry
end

end total_worth_of_presents_l189_189747


namespace polynomial_factorization_l189_189509

theorem polynomial_factorization :
  ∃ (f g : Polynomial ℤ), f = (Polynomial.X ^ 2 + Polynomial.X + 1) ∧
  g = (Polynomial.X ^ 13 - Polynomial.X ^ 12 + Polynomial.X ^ 8 - 
       Polynomial.X ^ 7 + Polynomial.X ^ 6 - Polynomial.X + 1) ∧ 
  Polynomial.X ^ 15 + Polynomial.X ^ 8 + 1 = f * g := 
by 
  use Polynomial.X ^ 2 + Polynomial.X + 1
  use Polynomial.X ^ 13 - Polynomial.X ^ 12 + Polynomial.X ^ 8 - Polynomial.X ^ 7 + Polynomial.X ^ 6 - Polynomial.X + 1
  split; try { refl }
  split; try { refl }
  sorry

end polynomial_factorization_l189_189509


namespace hyperbola_triangle_area_l189_189640

theorem hyperbola_triangle_area 
  (F1 F2 P : ℝ × ℝ)
  (h_hyperbola : ∀ (x y : ℝ), x ∈ {p.1 | p.1^2 - p.2^2/24 = 1})
  (h_foci : F1 = (-5, 0) ∧ F2 = (5, 0))
  (h_pointP : (P.1 - F1.1)^2 + (P.2 - F1.2)^2 = 8^2 ∧ (P.1 - F2.1)^2 + (P.2 - F2.2)^2 = 6^2)
  (h_ratio : 3 * real.dist P F1 = 4 * real.dist P F2) :
  let area := λ a b c : ℝ, 0.5 * a * b * sin (real.angle_of_vectors a b) in
  area (real.dist P F1) (real.dist P F2) (real.dist F1 F2) = 24 :=
sorry

end hyperbola_triangle_area_l189_189640


namespace norm_two_u_l189_189620

noncomputable def vector_u : ℝ × ℝ := sorry

theorem norm_two_u {u : ℝ × ℝ} (hu : ∥u∥ = 5) : ∥(2 : ℝ) • u∥ = 10 := by
  sorry

end norm_two_u_l189_189620


namespace sum_of_consecutive_integers_with_product_812_l189_189917

theorem sum_of_consecutive_integers_with_product_812 (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := 
sorry

end sum_of_consecutive_integers_with_product_812_l189_189917


namespace angle_C_measure_l189_189786

theorem angle_C_measure 
  (p q : Prop) 
  (h1 : p) (h2 : q) 
  (A B C : ℝ) 
  (h_parallel : p = q) 
  (h_A_B : A = B / 10) 
  (h_straight_line : B + C = 180) 
  : C = 16.36 := 
sorry

end angle_C_measure_l189_189786


namespace min_distance_from_integer_point_to_line_l189_189453

theorem min_distance_from_integer_point_to_line : 
  ∃ (x y : ℤ), 
  let d := (25 * x - 15 * y + 12) / Real.sqrt (25^2 + (-15)^2) in
  ∀ (u v : ℤ), let d' := (25 * u - 15 * v + 12) / Real.sqrt (25^2 + (-15)^2) in d ≤ d' ∧ d = 2 := 
by 
  sorry

end min_distance_from_integer_point_to_line_l189_189453


namespace exists_large_m_and_abc_l189_189020

theorem exists_large_m_and_abc (n₀ : ℕ) :
  ∃ n₀ ∈ ℕ, ∀ (m : ℕ), m ≥ n₀ →
    ∃ (a b c : ℕ), m^3 < a ∧ a < b ∧ b < c ∧ c < (m + 1)^3 ∧ ∃ (k : ℕ), abc = k^3 :=
begin
  sorry
end

end exists_large_m_and_abc_l189_189020


namespace product_12_3460_l189_189055

theorem product_12_3460 : 12 * 3460 = 41520 :=
by
  sorry

end product_12_3460_l189_189055


namespace arithmetic_sum_24_l189_189726

theorem arithmetic_sum_24 {a : ℕ → ℤ} {d : ℤ} 
  (h_arith_seq : ∀ n : ℕ, a n = a 0 + n * d)
  (h_sum_condition : a 5 + a 10 + a 15 + a 20 = 20) : 
  let S24 := 12 * (a 0 + a 23) in
  S24 = 132 := 
by {
  -- Use h_arith_seq and h_sum_condition to obtain S24
  sorry
}

end arithmetic_sum_24_l189_189726


namespace value_of_a_star_b_l189_189837

variable (a b : ℤ)

def operation_star (a b : ℤ) : ℚ :=
  1 / a + 1 / b

theorem value_of_a_star_b (h1 : a + b = 7) (h2 : a * b = 12) :
  operation_star a b = 7 / 12 := by
  sorry

end value_of_a_star_b_l189_189837


namespace consecutive_integers_sum_l189_189893

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189893


namespace berries_difference_change_l189_189083

def berries_problem :=
  let B := 20 in
  let R := B + 10 in
  (R - B) + B = 50

theorem berries_difference_change : berries_problem := by
  sorry

end berries_difference_change_l189_189083


namespace covered_part_larger_than_uncovered_l189_189441

-- Definitions representing the conditions
variables {page : Type} [measure_space page]

-- Definitions for covered parts 1, 2, 3 and exposed parts 1', 2', 3'
variables (a b1 b2 b3 ap bp1 bp2 bp3 : page)
-- Areas of the regions
variables {μ : measure page}

-- Condition 1: A and B vertices of the upper page lie on the lower page
-- Not explicitly used in our Lean statement
-- Condition 2: The upper and lower pages are equal in size
axiom equal_size : μ a = μ ap
-- Condition 3: Areas of 1 + 2 + 3 are equal to 1' + 2' + 3'
axiom area_parts_eq : μ b1 + μ b2 + μ b3 = μ bp1 + μ bp2 + μ bp3

-- Additional covered area 4
variables (area4 : page)
axiom area4_gt_zero : μ area4 > 0

-- Lean statement to prove the main question
theorem covered_part_larger_than_uncovered :
  μ (b1 + b2 + b3 + area4) > μ (bp1 + bp2 + bp3) :=
by {
  -- apply the given axioms and prove the statement
  sorry
}

end covered_part_larger_than_uncovered_l189_189441


namespace parallelogram_side_length_sum_l189_189460

theorem parallelogram_side_length_sum (x y z : ℚ) 
  (h1 : 3 * x - 1 = 12)
  (h2 : 4 * z + 2 = 7 * y + 3) :
  x + y + z = 121 / 21 :=
by
  sorry

end parallelogram_side_length_sum_l189_189460


namespace range_of_a_l189_189333

theorem range_of_a
  (x0 : ℝ) (a : ℝ)
  (hx0 : x0 > 1)
  (hineq : (x0 + 1) * Real.log x0 < a * (x0 - 1)) :
  a > 2 :=
sorry

end range_of_a_l189_189333


namespace sum_of_consecutive_integers_with_product_812_l189_189940

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189940


namespace angle_O1N_O2B_is_90_degrees_l189_189442

noncomputable def isIsosceles (a b c : ℝ) : Prop :=
a = b ∨ b = c ∨ a = c

theorem angle_O1N_O2B_is_90_degrees
  (ω₁ ω₂ : Circle)
  (O₁ O₂ B K L A C N : Point)
  (h₁ : Center ω₁ = O₁)
  (h₂ : Center ω₂ = O₂)
  (h₃ : B ∈ ω₁ ∧ B ∈ ω₂)
  (h₄ : Line.extend O₂ B ∩ ω₁ = K)
  (h₅ : Line.extend O₁ B ∩ ω₂ = L)
  (h₆ : Parallels (Line.through B K) (Line.through A C))
  (h₇ : K ∈ ω₁)
  (h₈ : L ∈ ω₂)
  (h₉ : A ∈ ω₁)
  (h₁₀ : C ∈ ω₂)
  (h₁₁ : Radians.angle AK CL = N)
  : Angle (Line.through O₁ N) (Line.through O₂ B) = 90 :=
sorry

end angle_O1N_O2B_is_90_degrees_l189_189442


namespace scale_model_height_is_correct_l189_189181

noncomputable def actualHeight : ℤ := 1063
noncomputable def scaleRatio : ℤ := 50

def expectedScaleModelHeight : ℤ :=
  Int.round (actualHeight / scaleRatio.toReal)

theorem scale_model_height_is_correct :
  expectedScaleModelHeight = 21 :=
by
  sorry

end scale_model_height_is_correct_l189_189181


namespace square_inscribed_in_hexagon_has_side_length_l189_189432

-- Definitions for the conditions given
noncomputable def side_length_square (AB EF : ℝ) : ℝ :=
  if AB = 30 ∧ EF = 19 * (Real.sqrt 3 - 1) then 10 * Real.sqrt 3 else 0

-- The theorem stating the specified equality
theorem square_inscribed_in_hexagon_has_side_length (AB EF : ℝ)
  (hAB : AB = 30) (hEF : EF = 19 * (Real.sqrt 3 - 1)) :
  side_length_square AB EF = 10 * Real.sqrt 3 := 
by 
  -- This is the proof placeholder
  sorry

end square_inscribed_in_hexagon_has_side_length_l189_189432


namespace min_value_of_f_l189_189048

def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 1

theorem min_value_of_f : ∃ x : ℝ, f x = x^3 - 3 * x^2 + 1 ∧ (∀ y : ℝ, f y ≥ f 2) :=
by
  sorry

end min_value_of_f_l189_189048


namespace exists_grid_assignment_a_exists_grid_assignment_b_l189_189739

-- Definition of the assignment predicate for part (a)
def grid_assignment_a (g : ℤ × ℤ → ℤ) : Prop :=
  ∀ i j, (∑ x in finset.range 4, ∑ y in finset.range 6, g (i + x, j + y)) = 10

-- Part (a): Prove there exists a grid assignment such that every 4x6 subgrid sums to 10
theorem exists_grid_assignment_a : ∃ g : ℤ × ℤ → ℤ, grid_assignment_a g :=
by
  sorry

-- Definition of the assignment predicate for part (b)
def grid_assignment_b (g : ℤ × ℤ → ℤ) : Prop :=
  ∀ i j, (∑ x in finset.range 4, ∑ y in finset.range 6, g (i + x, j + y)) = 1

-- Part (b): Prove there exists a grid assignment such that every 4x6 subgrid sums to 1
theorem exists_grid_assignment_b : ∃ g : ℤ × ℤ → ℤ, grid_assignment_b g :=
by
  sorry

end exists_grid_assignment_a_exists_grid_assignment_b_l189_189739


namespace total_weight_10_moles_AlF3_l189_189116

-- Definitions
def atomic_weight_Al : ℝ := 26.98
def atomic_weight_F : ℝ := 19.00
def AlF3_composition : (ℕ × ℕ) := (1, 3) -- (1 Al atom, 3 F atoms)
def moles_AlF3 : ℕ := 10

-- Problem statement
theorem total_weight_10_moles_AlF3 :
  let molecular_weight_AlF3 := (AlF3_composition.1 * atomic_weight_Al + AlF3_composition.2 * atomic_weight_F) in
  let total_weight := molecular_weight_AlF3 * moles_AlF3 in
  total_weight = 839.8 :=
by
  sorry

end total_weight_10_moles_AlF3_l189_189116


namespace additional_pecks_needed_l189_189011

theorem additional_pecks_needed :
  let peck_to_bushel := 1 / 4
      bushel_to_barrel := 1 / 9
      pecks_already_picked := 1
      pecks_per_bushel := 4  -- Derived from 1 bushel = 4 pecks
      bushels_per_barrel := 9 -- Derived from 1 barrel = 9 bushels
  in
  (bushels_per_barrel * pecks_per_bushel) - pecks_already_picked = 35 :=
by sorry

end additional_pecks_needed_l189_189011


namespace count_multiples_of_13_and_7_l189_189686

theorem count_multiples_of_13_and_7 (a b : ℤ) (h1 : a = 300) (h2 : b = 600) :
  let lcm_13_7 := 13 * 7 in
  let first_multiple := Int.ceil (a / lcm_13_7) * lcm_13_7 in
  let last_multiple := Int.floor (b / lcm_13_7) * lcm_13_7 in
  let multiples_in_range := (last_multiple - first_multiple) / lcm_13_7 + 1 in
  multiples_in_range = 3 := 
by 
  sorry

end count_multiples_of_13_and_7_l189_189686


namespace juyeon_distance_l189_189374

theorem juyeon_distance (usual_distance : ℝ) (speed_ratio : ℝ) : usual_distance = 215 → speed_ratio = 0.32 → usual_distance * speed_ratio = 68.8 :=
by
  intros h1 h2
  rw [h1, h2]
  norm_num
  sorry

end juyeon_distance_l189_189374


namespace probability_of_getting_number_greater_than_3_l189_189717

def die_faces : Finset ℕ := {1, 2, 3, 4, 5, 6}

def favorable_outcomes (s : Finset ℕ) : Finset ℕ := s.filter (λ x, x > 3)

def probability (total favorable : ℕ) : ℚ := favorable / total

theorem probability_of_getting_number_greater_than_3 : probability die_faces.card (favorable_outcomes die_faces).card = 1 / 2 :=
by
  have h_total : die_faces.card = 6 := Finset.card_finset_eq.2 rfl,
  have h_favorable : (favorable_outcomes die_faces).card = 3 := rfl,
  simp [probability, h_total, h_favorable],
  norm_num

end probability_of_getting_number_greater_than_3_l189_189717


namespace polynomial_perfect_square_binomial_l189_189540

theorem polynomial_perfect_square_binomial (a : ℤ) :
    let p := 4 * x ^ 2 + 1 in
    (∃ m : ℤ, p + m = (2 * x + 1) ^ 2) ∨
    (∃ m : ℤ, p + m = (2 * x - 1) ^ 2) ∨
    (∃ m : ℤ, p + m = (2 * x ^ 2) ^ 2 + 4 * x ^ 2 + 1) :=
    a = 3 :=
sorry

end polynomial_perfect_square_binomial_l189_189540


namespace consecutive_integers_sum_l189_189885

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189885


namespace angle_opposite_side_l189_189342

theorem angle_opposite_side (a b c : ℝ) (h : (a + b + c) * (a + b - c) = 3 * a * b) : 
  ∃ θ : ℝ, θ = 60 ∧ cos θ = 1/2 ∧ cos θ = (a^2 + b^2 - c^2) / (2 * a * b) :=
by
  sorry

end angle_opposite_side_l189_189342


namespace sum_of_k_values_for_minimum_area_l189_189838

def area_of_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)).abs

theorem sum_of_k_values_for_minimum_area :
  let k_values : ℝ := [3, 5] in
  (∑ k in k_values, k) = 8 :=
by
  sorry

end sum_of_k_values_for_minimum_area_l189_189838


namespace total_surface_area_of_cube_l189_189063

theorem total_surface_area_of_cube : 
  ∀ (s : Real), 
  (12 * s = 36) → 
  (s * Real.sqrt 3 = 3 * Real.sqrt 3) → 
  6 * s^2 = 54 := 
by
  intros s h1 h2
  sorry

end total_surface_area_of_cube_l189_189063


namespace perc_diff_2_l189_189988

-- Define given conditions
def perc_55_of_40 : Real := (55 / 100) * 40
def frac_4_5_of_25 : Real := (4 / 5) * 25

-- State the problem
theorem perc_diff_2 : perc_55_of_40 - frac_4_5_of_25 = 2 := 
by 
sor

end perc_diff_2_l189_189988


namespace determine_product_of_distances_l189_189330

noncomputable def ellipse := {p : ℝ × ℝ | p.1 ^ 2 / 9 + p.2 ^ 2 / 16 = 1}
noncomputable def hyperbola := {p : ℝ × ℝ | p.1 ^ 2 / 5 - p.2 ^ 2 / 4 = 1}
def common_foci (F1 F2 : ℝ × ℝ) (ellipse hyperbola : set (ℝ × ℝ)) : Prop :=
  ∀ P ∈ ellipse ∩ hyperbola, |(dist P F1)| + |(dist P F2)| = 8 ∧ |(dist P F1)| - |(dist P F2)| = 4

theorem determine_product_of_distances
  (F1 F2 : ℝ × ℝ)
  (h_common_foci : common_foci F1 F2 ellipse hyperbola)
  (P : ℝ × ℝ)
  (h_P : P ∈ ellipse ∩ hyperbola) :
  |(dist P F1)| * |(dist P F2)| = 12 :=
  sorry

end determine_product_of_distances_l189_189330


namespace fifth_group_number_l189_189487

-- Definitions based on the conditions
def students : List ℕ := List.range' 1 161
def first_draw : ℕ := 3
def interval : ℕ := 8

-- Proving the number in the fifth group using systematic sampling
theorem fifth_group_number : 
  (students[4 * interval - (interval - first_draw) - 1]) = 35 := by
  sorry

end fifth_group_number_l189_189487


namespace insertion_sort_comparisons_range_l189_189018

theorem insertion_sort_comparisons_range (n : ℕ) (n_pos : 0 < n) :
  n - 1 ≤ number_of_comparisons_in_insertion_sort n ∧ 
  number_of_comparisons_in_insertion_sort n ≤ (n * (n - 1)) / 2 := 
sorry

end insertion_sort_comparisons_range_l189_189018


namespace domain_function_l189_189650

open Set

variable {α β : Type} [Ordered α]

noncomputable def domain_transformation (f : α → β) (dom2x_minus_3 domx : Set α) : Prop :=
  dom2x_minus_3 = Icc (-1 : α) 2 → domx = Icc (-5 : α) 1

theorem domain_function
  {f : ℝ → β} :
  domain_transformation f (Icc (-1 : ℝ) 2) (Icc (-5 : ℝ) 1) :=
by
  intro h
  sorry

end domain_function_l189_189650


namespace bob_number_correct_l189_189435

variables (a b : ℂ)

def alice_number := 7 + 4 * complex.I
def product := 48 - 16 * complex.I
def bob_number := (272 / 65) - (304 / 65) * complex.I

theorem bob_number_correct (h : a * b = product) (ha : a = alice_number) : b = bob_number :=
sorry

end bob_number_correct_l189_189435


namespace product_of_real_roots_l189_189457

theorem product_of_real_roots : 
  (∏ x in {x : ℝ | x ^ log x = 10}, x) = 1 := by
  sorry

end product_of_real_roots_l189_189457


namespace prob_xi_lt_2_l189_189543

noncomputable def xi : Type := sorry  -- Placeholder for the random variable type

axiom normal_dist : xi -> Prop -- xi follows normal distribution N(1, σ^2)
axiom P_lt_0 : ∀ (ξ : xi), P(ξ < 0) = 0.3 -- P(ξ < 0) = 0.3 for given ξ

theorem prob_xi_lt_2 (ξ : xi) (h : normal_dist ξ) : P(ξ < 2) = 0.7 := 
by
  have h_sym : P(ξ > 2) = 0.3 := sorry -- Leveraging symmetry
  have total_prob : P(ξ < 2) = 1 - P(ξ > 2) := sorry -- Using total probability
  rw [h_sym, total_prob]
  exact 0.7

end prob_xi_lt_2_l189_189543


namespace top_square_after_folds_l189_189539

/-- 
  A piece of paper consists of 20 squares in a 4x5 grid numbered row-wise from 1 to 20.
  The folding sequence is given:
  1. Fold the left third over the middle third.
  2. Fold the right third over the newly folded section.
  3. Fold the bottom half up over the top half.
  4. Fold the top half down over the bottom half.
  We need to prove that the square number 5 is on top after step 4.
-/
theorem top_square_after_folds (grid : list (list ℕ)) (folds : list (list (list ℕ) → list (list ℕ))) :
  (fold (λ g f => f g) grid folds).head.head = 5 := sorry

/-- Example 4x5 grid, numbered row-wise from 1 to 20. -/
def initial_grid : list (list ℕ) :=
[ [1, 2, 3, 4, 5],
  [6, 7, 8, 9, 10],
  [11, 12, 13, 14, 15],
  [16, 17, 18, 19, 20] ]

/-- Sequence of folding functions. -/
def folding_sequence : list (list (list ℕ) → list (list ℕ)) :=
[
  λ g => [[g[0][2], g[0][3], g[0][0], g[0][1], g[0][4]],
           [g[1][2], g[1][3], g[1][0], g[1][1], g[1][4]],
           [g[2][2], g[2][3], g[2][0], g[2][1], g[2][4]],
           [g[3][2], g[3][3], g[3][0], g[3][1], g[3][4]]],
  λ g => [[g[0][4], g[0][2], g[0][3], g[0][0], g[0][1]],
           [g[1][4], g[1][2], g[1][3], g[1][0], g[1][1]],
           [g[2][4], g[2][2], g[2][3], g[2][0], g[2][1]],
           [g[3][4], g[3][2], g[3][3], g[3][0], g[3][1]]],
  λ g => [[g[3][0], g[3][1], g[3][2], g[3][3], g[3][4]],
           [g[2][0], g[2][1], g[2][2], g[2][3], g[2][4]],
           [g[1][0], g[1][1], g[1][2], g[1][3], g[1][4]],
           [g[0][0], g[0][1], g[0][2], g[0][3], g[0][4]]],
  λ g => [[g[0][0], g[0][1], g[0][2], g[0][3], g[0][4]],
           [g[1][0], g[1][1], g[1][2], g[1][3], g[1][4]],
           [g[2][0], g[2][1], g[2][2], g[2][3], g[2][4]],
           [g[3][0], g[3][1], g[3][2], g[3][3], g[3][4]]]
  ]

#eval top_square_after_folds initial_grid folding_sequence

end top_square_after_folds_l189_189539


namespace total_possible_four_team_ranking_sequences_l189_189417

-- Define the teams
inductive Team
| A | B | C | D

open Team

-- Define the match outcomes
inductive Outcome
| win (winner : Team) (loser : Team)

open Outcome

-- Define the Saturday matches
def saturday_matches : list (Outcome × Outcome) :=
[
  (win A B, win C D),
  (win A B, win D C),
  (win B A, win C D),
  (win B A, win D C)
]

-- Define the possible final rankings based on the condition that there's a specific example
def final_rankings : list (list Team) := sorry

-- Prove the number of possible rankings
theorem total_possible_four_team_ranking_sequences : final_rankings.length = 16 :=
sorry

end total_possible_four_team_ranking_sequences_l189_189417


namespace train_length_l189_189552

theorem train_length (L : ℝ) (h1 : L + 110 / 15 = (L + 250) / 20) : L = 310 := 
sorry

end train_length_l189_189552


namespace Razorback_tshirt_problem_l189_189436

theorem Razorback_tshirt_problem
  (A T : ℕ)
  (h1 : A + T = 186)
  (h2 : 78 * T = 1092) :
  A = 172 := by
  sorry

end Razorback_tshirt_problem_l189_189436


namespace infinite_solutions_to_diophantine_eq_l189_189799

theorem infinite_solutions_to_diophantine_eq {a b : ℤ} (h_coprime : Int.gcd a b = 1) : 
  ∃ (infinitely_many_triples : ℕ → ℤ × ℤ × ℤ), 
    ∀ n : ℕ, 
    let (x, y, z) := infinitely_many_triples n in 
    a * x^2 + b * y^2 = z^3 ∧ (Int.gcd x y = 1) :=
by
  sorry

end infinite_solutions_to_diophantine_eq_l189_189799


namespace problem_1_problem_2_l189_189235

noncomputable def f (x t : ℝ) : ℝ := |x - 1| + |x - t|

theorem problem_1 (x : ℝ) :
  f x 2 > 2 ↔ x < 1/2 ∨ x > 5/2 :=
by
  sorry -- proof goes here

theorem problem_2 (a : ℝ) :
  (∀ t ∈ Icc (1 : ℝ) 2, ∀ x ∈ Icc (-1 : ℝ) 3, f x t ≥ a + x) → a ≤ -1 :=
by
  sorry -- proof goes here

end problem_1_problem_2_l189_189235


namespace inclination_angle_of_vertical_line_l189_189449

theorem inclination_angle_of_vertical_line :
  ∀ x : ℝ, x = Real.tan (60 * Real.pi / 180) → ∃ θ : ℝ, θ = 90 := by
  sorry

end inclination_angle_of_vertical_line_l189_189449


namespace bottles_used_first_game_l189_189156

theorem bottles_used_first_game:
  ∀ (total_cases total_bottles_per_case bottles_left_after_second_game bottles_used_second_game total_bottles_used FirstGameBottlesUsed : ℕ),
  total_cases = 10 →
  total_bottles_per_case = 20 →
  bottles_left_after_second_game = 20 →
  bottles_used_second_game = 110 →
  total_bottles_used = total_cases * total_bottles_per_case - bottles_left_after_second_game →
  FirstGameBottlesUsed = total_bottles_used - bottles_used_second_game →
  FirstGameBottlesUsed = 70 :=
begin
  intros,
  sorry
end

end bottles_used_first_game_l189_189156


namespace john_has_612_euros_l189_189400

-- Definitions of the conditions
def darwin := 45
def conversion_rate := 0.85
def mia := 2 * darwin + 20
def laura := 3 * (mia + darwin) - 30
def john := 1.5 * (laura + darwin)
def john_in_euros := john * conversion_rate

-- Statement to prove
theorem john_has_612_euros : john_in_euros = 612 := 
  sorry

end john_has_612_euros_l189_189400


namespace functionD_is_odd_and_monotonically_decreasing_l189_189187

-- Define the functions
def fA (x : ℝ) := log (1/2) x
def fB (x : ℝ) := 1 / x
def fC (x : ℝ) := -tan x
def fD (x : ℝ) := -x^3

-- Define what it means for a function to be odd
def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define what it means for a function to be monotonically decreasing
def isMonotonicallyDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x ≥ f y

-- State the proof problem
theorem functionD_is_odd_and_monotonically_decreasing :
  isOdd fD ∧ isMonotonicallyDecreasing fD :=
  sorry

end functionD_is_odd_and_monotonically_decreasing_l189_189187


namespace exists_x_l189_189775

noncomputable def exists_real_number (n : ℕ) (x : ℕ → ℝ) : Prop :=
  ∃ (x : ℝ), (finset.range n).sum (λ i, (x - x i) % 1) ≤ (n - 1) / 2

theorem exists_x (n : ℕ) (x : ℕ → ℝ) : exists_real_number n x :=
sorry

end exists_x_l189_189775


namespace reflection_of_point_D_over_axes_l189_189277

noncomputable def reflect_y (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
noncomputable def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

theorem reflection_of_point_D_over_axes :
  let D := (3 : ℝ, 3 : ℝ),
      D' := reflect_y D,
      D'' := reflect_x D' in
  D'' = (-3, -3) :=
by
  let D := (3 : ℝ, 3 : ℝ)
  let D' := reflect_y D
  let D'' := reflect_x D'
  simp [D, D', D'']
  sorry

end reflection_of_point_D_over_axes_l189_189277


namespace inequality_holds_l189_189515

def seq (n : ℕ) : ℝ :=
  if n = 1 then Real.sqrt 2019
  else Real.sqrt (2019 + seq (n-1))

theorem inequality_holds : ∀ n : ℕ, seq n < 2019 :=
by
  sorry

end inequality_holds_l189_189515


namespace consecutive_integers_sum_l189_189899

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189899


namespace surface_area_expansion_l189_189709

-- Definition of volume at initial and expanded states
def initial_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3
def expanded_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * (2 * r) ^ 3

-- Definition of surface area at initial and expanded states
def initial_surface_area (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2
def expanded_surface_area (r : ℝ) : ℝ := 4 * Real.pi * (2 * r) ^ 2

-- Theorem to prove
theorem surface_area_expansion (r : ℝ) (h : expanded_volume r = 8 * initial_volume r) : 
  expanded_surface_area r = 4 * initial_surface_area r :=
by
  sorry

end surface_area_expansion_l189_189709


namespace length_of_train_l189_189183

theorem length_of_train
    (speed_kmh : ℝ)
    (bridge_length_m : ℝ)
    (time_s : ℝ)
    (speed_ms : ℝ)
    (total_distance_m : ℝ) :
    speed_kmh = 50 →
    bridge_length_m = 140 →
    time_s = 36 →
    speed_ms = (speed_kmh * 1000 / 3600) →
    total_distance_m = speed_ms * time_s →
    total_distance_m = bridge_length_m + 360 :=
by
    intros h1 h2 h3 h4 h5
    rw [h1, h2, h3, h4] at h5
    sorry

end length_of_train_l189_189183


namespace find_lawn_width_l189_189545

/-- Given a rectangular lawn with a length of 80 m and roads each 10 m wide,
    one running parallel to the length and the other running parallel to the width,
    with a total travel cost of Rs. 3300 at Rs. 3 per sq m, prove that the width of the lawn is 30 m. -/
theorem find_lawn_width (w : ℕ) (h_area_road : 10 * w + 10 * 80 = 1100) : w = 30 :=
by {
  sorry
}

end find_lawn_width_l189_189545


namespace total_prime_ending_starting_numerals_l189_189685

def single_digit_primes : List ℕ := [2, 3, 5, 7]
def number_of_possible_digits := 10

def count_3digit_numerals : ℕ :=
  4 * number_of_possible_digits * 4

def count_4digit_numerals : ℕ :=
  4 * number_of_possible_digits * number_of_possible_digits * 4

theorem total_prime_ending_starting_numerals : 
  count_3digit_numerals + count_4digit_numerals = 1760 := by
sorry

end total_prime_ending_starting_numerals_l189_189685


namespace necessary_but_not_sufficient_condition_l189_189168

theorem necessary_but_not_sufficient_condition (a : ℝ) :
  (0 < a ∧ a ≤ 1) → (∀ x : ℝ, x^2 - 2*a*x + a > 0) :=
by
  sorry

end necessary_but_not_sufficient_condition_l189_189168


namespace consecutive_integers_sum_l189_189901

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189901


namespace elaine_earnings_increase_l189_189375

variable (E P : ℝ)

theorem elaine_earnings_increase :
  (0.25 * (E * (1 + P / 100)) = 1.4375 * 0.20 * E) → P = 15 :=
by
  intro h
  -- Start an intermediate transformation here
  sorry

end elaine_earnings_increase_l189_189375


namespace number_of_lattice_points_on_circle_l189_189415

theorem number_of_lattice_points_on_circle (center_x center_y radius : ℤ) 
(h_center_x : center_x = 199) (h_center_y : center_y = 0) (h_radius : radius = 199) :
  let circle_eq (x y : ℤ) := (x - center_x) ^ 2 + y ^ 2 = radius ^ 2 in
  ((∃ x y : ℤ, circle_eq x y ∧ x = 199 ∧ y = 0) ∧
   (∃ x y : ℤ, circle_eq x y ∧ x = -199 ∧ y = 0) ∧
   (∃ x y : ℤ, circle_eq x y ∧ x = 0 ∧ y = 199) ∧
   (∃ x y : ℤ, circle_eq x y ∧ x = 0 ∧ y = -199) ∧
   ¬(∃ x y : ℤ, circle_eq x y ∧ x ≠ 0 ∧ y ≠ 0)) :=
sorry

end number_of_lattice_points_on_circle_l189_189415


namespace max_value_of_sum_l189_189319

theorem max_value_of_sum (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) :
  a + b + c + d ≤ -5 := 
sorry

end max_value_of_sum_l189_189319


namespace sufficient_but_not_necessary_l189_189645

theorem sufficient_but_not_necessary (x : ℝ) (h : x > 2) : (1/x < 1/2 ∧ (∃ y : ℝ, 1/y < 1/2 ∧ y ≤ 2)) :=
by { sorry }

end sufficient_but_not_necessary_l189_189645


namespace hexagon_parallel_sides_l189_189311

theorem hexagon_parallel_sides
  (A B C D E F : Type)
  (inscribed : Set.Point.inscribed_in_circle {A, B, C, D, E, F})
  (AB_parallel_DE : AreParallel AB DE)
  (BC_parallel_EF : AreParallel BC EF)
  : AreParallel CD AF :=
begin
  sorry
end

end hexagon_parallel_sides_l189_189311


namespace max_number_of_integer_solutions_l189_189541

noncomputable def polynomial := ℤ[X]  -- Define polynomials with integer coefficients

def satisfies_condition (p : polynomial) : Prop := 
  p.eval 50 = 50

def max_integer_solutions (p : polynomial) : ℕ :=
  (multiset.card (multiset.filter (λ k : ℤ, p.eval k = k^3) (multiset.range (101)))) -- Here we filter integer solutions in a certain range for practical purposes

theorem max_number_of_integer_solutions (p : polynomial) (h : satisfies_condition p) : max_integer_solutions p ≤ 9 :=
  sorry

end max_number_of_integer_solutions_l189_189541


namespace time_to_complete_one_round_l189_189059

-- Definitions based on conditions
def length_breadth_ratio : ℝ := 4 / 1 -- Ratio between length and breadth
def area_park : ℝ := 102400 -- Area of park in square meters
def cycling_speed_kmh : ℝ := 12 -- Speed of man in km/hr
def cycling_speed_ms : ℝ := cycling_speed_kmh * 1000 / 3600 -- Speed in m/s
def length_park (x : ℝ) : ℝ := 4 * x
def breadth_park (x : ℝ) : ℝ := x

-- Problem statement
theorem time_to_complete_one_round : 
    ∃ t : ℝ, 
    (∀ x : ℝ, 
      (4 * x * x = area_park) → 
      t = (2 * length_park x + 2 * breadth_park x) / cycling_speed_ms) ∧
    t = 8 :=
begin
  sorry
end

end time_to_complete_one_round_l189_189059


namespace greatest_possible_individual_award_l189_189542

theorem greatest_possible_individual_award :
  ∀ (total_prize : ℕ) (num_winners : ℕ) (min_award : ℕ)
    (fraction_prize : ℚ) (fraction_winners : ℚ),
  total_prize = 500 →
  num_winners = 20 →
  min_award = 20 →
  fraction_prize = 2/5 →
  fraction_winners = 3/5 →
  let allocated_prize := fraction_prize * total_prize in
  let allocated_winners := fraction_winners * num_winners in
  let min_total_award := num_winners * min_award in
  let leftover_prize := total_prize - min_total_award in
  allocated_prize = 200 →
  allocated_winners = 12 →
  min_total_award = 400 →
  leftover_prize = 100 →
  ∃ (max_award : ℕ), max_award = 120 :=
begin
  sorry
end

end greatest_possible_individual_award_l189_189542


namespace plates_cost_l189_189214

theorem plates_cost :
  ∃ P : ℝ, (9 * P + 4 * 1.5 = 24) ∧ (P = 2) :=
by
  exists 2
  split
  . sorry  -- This would be the place where the mathematical proof would go
  . rfl    -- This simply states that the P we found is indeed 2

end plates_cost_l189_189214


namespace square_perimeter_l189_189498

def perimeter_of_square (side_length : ℝ) : ℝ :=
  4 * side_length

theorem square_perimeter (side_length : ℝ) (h : side_length = 5) : perimeter_of_square side_length = 20 := by
  sorry

end square_perimeter_l189_189498


namespace total_pennies_donated_l189_189209

def cassandra_pennies : ℕ := 5000
def james_pennies : ℕ := cassandra_pennies - 276
def total_pennies : ℕ := cassandra_pennies + james_pennies

theorem total_pennies_donated : total_pennies = 9724 := by
  sorry

end total_pennies_donated_l189_189209


namespace sum_of_five_distinct_integers_product_2022_l189_189056

theorem sum_of_five_distinct_integers_product_2022 :
  ∃ (a b c d e : ℤ), 
    a * b * c * d * e = 2022 ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
    c ≠ d ∧ c ≠ e ∧
    d ≠ e ∧ 
    (a + b + c + d + e = 342 ∨
     a + b + c + d + e = 338 ∨
     a + b + c + d + e = 336 ∨
     a + b + c + d + e = -332) :=
by 
  sorry

end sum_of_five_distinct_integers_product_2022_l189_189056


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189089

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189089


namespace cos_squared_sum_l189_189318

theorem cos_squared_sum (x y z : ℝ) 
  (h1 : sin x + sin y + sin z = 0 ∨ sin x + sin y - sin z = 0)
  (h2 : cos x + cos y + cos z = 0 ∨ cos x + cos y - cos z = 0) : 
  cos x ^ 2 + cos y ^ 2 + cos z ^ 2 = 3 / 2 := 
by
  sorry

end cos_squared_sum_l189_189318


namespace sphere_radius_l189_189289

-- Definitions and conditions
def PA := (1 : ℝ)
def PB := (2 : ℝ)
def PC := (3 : ℝ)

-- Mutual perpendicularity is implicit in the geometric configuration.

-- The radius R to be proven
def R := (Real.sqrt 14) / 2

-- The statement of the problem
theorem sphere_radius (P A B C : ℝ) (h1 : P = PA) (h2 : P = PB) (h3 : P = PC) : 
    R = (Real.sqrt 14) / 2 := 
sorry

end sphere_radius_l189_189289


namespace triply_perspective_triangles_l189_189994

-- Defining the points and the given conditions as hypotheses
variables {Point : Type} [AffineSpace ℝ Point]

-- Points of the triangles
variables {A B C A1 B1 C1 O O1 O2 O3 : Point}

-- Definition of collinearity and point incidence with lines
def collinear (p q r : Point) : Prop := ∃ l : AffineSubspace ℝ Point, p ∈ l ∧ q ∈ l ∧ r ∈ l
def on_line (p q : Point) (r : Point) : Prop := ∃ l : AffineSubspace ℝ Point, p ∈ l ∧ q ∈ l ∧ r ∈ l

-- Conditions translated into Lean hypotheses
axiom intersect_at_O : on_line A A1 O ∧ on_line B B1 O ∧ on_line C C1 O
axiom intersect_at_O1 : on_line A A1 O1 ∧ on_line B C1 O1 ∧ on_line C B1 O1
axiom intersect_at_O2 : on_line A C1 O2 ∧ on_line B B1 O2 ∧ on_line C A1 O2

-- Statement of the theorem to be proved
theorem triply_perspective_triangles :
  ∃ (O3 : Point), on_line A B1 O3 ∧ on_line B A1 O3 ∧ on_line C C1 O3 :=
sorry

end triply_perspective_triangles_l189_189994


namespace xyz_neg_l189_189643

theorem xyz_neg {a b c x y z : ℝ} 
  (ha : a < 0) (hb : b < 0) (hc : c < 0) 
  (h : |x - a| + |y - b| + |z - c| = 0) : 
  x * y * z < 0 :=
by 
  -- to be proven
  sorry

end xyz_neg_l189_189643


namespace tennis_tournament_l189_189341

theorem tennis_tournament (n : ℕ) :
    ( (n = 3) ∨ (n = 5) ∨ (n = 7) ∨ (n = 8) → 
      let total_players := 2 * n + 3 in
      let total_matches := total_players * (total_players - 1) / 2 in
      let women_wins := 3 * (total_matches / 5) in
      let men_wins := 2 * (total_matches / 5) in
      (((women_wins + men_wins = total_matches) ∧
      (total_matches % 5 = 0)) → false)
    ) :=
begin
  intro h,
  cases h,
  all_goals {
    intros total_players total_matches women_wins men_wins ht hmod,
    sorry,
  }
end

end tennis_tournament_l189_189341


namespace tree_height_l189_189369

theorem tree_height (rungs1 rungs2 : ℕ) (height1 : ℝ) (hr : rungs1 = 12) (hh : height1 = 6) (hr2 : rungs2 = 40) :
  (rungs2 * (height1 / rungs1 : ℝ)) = 20 :=
by
  rw [hr, hh, hr2]
  norm_num
  sorry

end tree_height_l189_189369


namespace right_triangle_hypotenuse_length_l189_189399

theorem right_triangle_hypotenuse_length 
  (x : ℝ) (h1 : 0 < x) (h2 : x < π / 2)
  (a b c : ℝ)
  (h3 : a = c * cos x) (h4 : b = c * sin x)
  (h5 : 1 = cos x ^ 2 + sin x ^ 2) -- Pythagorean identity
  (h6 : (cos x ^ 2 * c = a^2 * c/3 + b^2 * 2 * c/3  - 2 * c^3 / 9)) --Stewart's theorem for point D
  (h7 : sin x ^ 2 * c = a^2 * 2 * c/3 + b^2 * c / 3 - 2 * c ^ 3 / 9): 
  c = 3 * sqrt 5 / 5 :=
sorry

end right_triangle_hypotenuse_length_l189_189399


namespace alfonso_initial_money_l189_189555

def daily_earnings : ℕ := 6
def days_per_week : ℕ := 5
def total_weeks : ℕ := 10
def cost_of_helmet : ℕ := 340

theorem alfonso_initial_money :
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  cost_of_helmet - total_earnings = 40 :=
by
  let weekly_earnings := daily_earnings * days_per_week
  let total_earnings := weekly_earnings * total_weeks
  show cost_of_helmet - total_earnings = 40
  sorry

end alfonso_initial_money_l189_189555


namespace geometric_sequence_sum_a_l189_189332

theorem geometric_sequence_sum_a (a : ℤ) (S : ℕ → ℤ) (a_n : ℕ → ℤ) 
  (h1 : ∀ n : ℕ, S n = 2^n + a)
  (h2 : ∀ n : ℕ, a_n n = if n = 1 then S 1 else S n - S (n - 1)) :
  a = -1 :=
by
  sorry

end geometric_sequence_sum_a_l189_189332


namespace case_irrational_case_rational_l189_189258

def p (T : Set ℝ) : ℝ := Set.prod T

-- Case 1: All elements are irrational numbers.
theorem case_irrational (T : Set ℝ) (hT_count: T.cardinality = 2021)
  (hT_irrational: ∀ x ∈ T, irrational x) : 
  (∃ T, (∀ a ∈ T, ∃ b ∈ ℤ, p T - a = 2 * b + 1)) :=
sorry

-- Case 2: At least one element is a rational number.
theorem case_rational (T : Set ℝ) (hT_count: T.cardinality = 2021)
  (hT_rational: ∃ x ∈ T, rational x) : 
  ¬(∃ T, (∀ a ∈ T, ∃ b ∈ ℤ, p T - a = 2 * b + 1)) :=
sorry

end case_irrational_case_rational_l189_189258


namespace consecutive_integer_product_sum_l189_189931

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189931


namespace coal_removal_date_l189_189575

theorem coal_removal_date (m n : ℝ) (h1 : 0 < m) (h2 : 0 < n)
  (h3 : 25 * m + 9 * n = 0.5)
  (h4 : ∃ z : ℝ,  z * (n + m) = 0.5)
  (h5 : ∀ z : ℝ, z = 12 → (16 + z) * m = (9 + z) * n):
  ∃ t : ℝ, t = 28 := 
by 
{
  sorry
}

end coal_removal_date_l189_189575


namespace consecutive_integer_product_sum_l189_189933

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189933


namespace problem_l189_189841

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189841


namespace num_sequences_l189_189012

-- Define a property for sequences that respects the difference condition.
def valid_sequence (s : List ℕ) : Prop :=
  ∀ i, i < s.length - 1 → abs (s[i+1] - s[i]) ≤ 2

-- Define a property that the sequence must contain at least one 4 or 5.
def contains_4_or_5 (s : List ℕ) : Prop :=
  4 ∈ s ∨ 5 ∈ s

-- The main theorem stating the required result for sequences of length 100.
theorem num_sequences (n : ℕ) : 
  (length n = 100 → 
   valid_sequence n ∧ contains_4_or_5 n) 
   → S n = 5^100 - 3^100 :=
sorry

end num_sequences_l189_189012


namespace trim_length_l189_189818

theorem trim_length {π : ℝ} (r : ℝ)
  (π_approx : π = 22 / 7)
  (area : π * r^2 = 616) :
  2 * π * r + 5 = 93 :=
by
  sorry

end trim_length_l189_189818


namespace point_same_side_l189_189722

theorem point_same_side {x y : ℤ} (a b c : ℤ) (p : ℤ × ℤ) :
  a*x + b*y + c > 0 → a*fstsnd p + b*fstsnd p + c > 0 :=
begin
  sorry
end

end point_same_side_l189_189722


namespace find_B_plus_C_l189_189315

theorem find_B_plus_C 
(A B C : ℕ)
(h1 : A ≠ B)
(h2 : B ≠ C)
(h3 : C ≠ A)
(h4 : A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0)
(h5 : A < 5 ∧ B < 5 ∧ C < 5)
(h6 : 25 * A + 5 * B + C + 25 * B + 5 * C + A + 25 * C + 5 * A + B = 125 * A + 25 * A + 5 * A) : 
B + C = 4 * A := by
  sorry

end find_B_plus_C_l189_189315


namespace child_support_calculation_l189_189405

noncomputable def owed_child_support (yearly_salary : ℕ) (raise_pct: ℝ) 
(raise_years_additional_salary: ℕ) (payment_percentage: ℝ) 
(payment_years_salary_before_raise: ℕ) (already_paid : ℝ) : ℝ :=
  let initial_salary := yearly_salary * payment_years_salary_before_raise
  let increase_amount := yearly_salary * raise_pct
  let new_salary := yearly_salary + increase_amount
  let salary_after_raise := new_salary * raise_years_additional_salary
  let total_income := initial_salary + salary_after_raise
  let total_support_due := total_income * payment_percentage
  total_support_due - already_paid

theorem child_support_calculation:
  owed_child_support 30000 0.2 4 0.3 3 1200 = 69000 :=
by
  sorry

end child_support_calculation_l189_189405


namespace sum_of_remainders_l189_189823

open Nat

theorem sum_of_remainders {n : ℕ} (h1 : ∀ m, 0 ≤ m ∧ m ≤ 6) :
  (∑ m in Finset.range 7, (1111 * m + 123) % 41) = 120 := 
by 
  sorry

end sum_of_remainders_l189_189823


namespace exponent_product_equiv_l189_189572

variable (a b : ℝ) -- Assuming a and b are real numbers

-- The problem statement in Lean 4
theorem exponent_product_equiv :
  a^3 * b * (a⁻¹ * b)⁻² = a^5 * b⁻¹ :=
by sorry

end exponent_product_equiv_l189_189572


namespace Jamie_correct_percentage_l189_189790

theorem Jamie_correct_percentage (y : ℕ) : ((8 * y - 2 * y : ℕ) / (8 * y : ℕ) : ℚ) * 100 = 75 := by
  sorry

end Jamie_correct_percentage_l189_189790


namespace least_number_of_teams_l189_189528

/-- A coach has 30 players in a team. If he wants to form teams of at most 7 players each for a tournament, we aim to prove that the least number of teams that he needs is 5. -/
theorem least_number_of_teams (players teams : ℕ) 
  (h_players : players = 30) 
  (h_teams : ∀ t, t ≤ 7 → t ∣ players) : teams = 5 := by
  sorry

end least_number_of_teams_l189_189528


namespace island_knights_count_l189_189006

theorem island_knights_count (inhabitants : Fin 17 → Prop) (positions_3_to_6: ∀ i : Fin 4, i.val + 3 < 17 → inhabitants ⟨i.val + 3, Nat.lt_trans i.is_lt (Nat.lt_succ_self 16)⟩ = "There is a liar below me") (rest_positions: ∀ i : Fin 12, i.val + 1 < 3 ∨ i.val + 7 < 17 → inhabitants ⟨if i.val < 2 then i.val else i.val + 4, sorry⟩ = "There is a knight above me") : 
  ∃ (knights : Fin 17 → bool), (∑ i, if knights i then 1 else 0) = 14 := 
sorry

end island_knights_count_l189_189006


namespace area_of_triangle_XYZ_l189_189737

-- Defining the necessary elements: points, triangle, medians and centroid
variables (X Y Z M N O : Type) [AffineSpace X Y Z M N O]

-- Conditions from part (a)
def right_angle_at_O (XO YO : ℝ) : Prop := XO * YO / 2 = 96

def triangle_XYZ : Prop := 
  let XM := 18
  let YN := 24
  let XO := (2 / 3) * XM
  let YO := (2 / 3) * YN
  right_angle_at_O XO YO

-- Proof goal
theorem area_of_triangle_XYZ (h : triangle_XYZ) : 
  let area_XYZ := 3 * (96 : ℝ)
  area_XYZ = 288 :=
  sorry

end area_of_triangle_XYZ_l189_189737


namespace c_share_l189_189143

theorem c_share (A B C : ℕ) 
    (h1 : A = 1/2 * B) 
    (h2 : B = 1/2 * C) 
    (h3 : A + B + C = 406) : 
    C = 232 := by 
    sorry

end c_share_l189_189143


namespace goldbach_conjecture_132_largest_difference_l189_189955

theorem goldbach_conjecture_132_largest_difference :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ p + q = 132 ∧ (q - p) = 122 :=
sorry

end goldbach_conjecture_132_largest_difference_l189_189955


namespace janet_pairs_of_2_l189_189715

def total_pairs (x y z : ℕ) : Prop := x + y + z = 18

def total_cost (x y z : ℕ) : Prop := 2 * x + 5 * y + 7 * z = 60

theorem janet_pairs_of_2 (x y z : ℕ) (h1 : total_pairs x y z) (h2 : total_cost x y z) (hz : z = 3) : x = 12 :=
by
  -- Proof is currently skipped
  sorry

end janet_pairs_of_2_l189_189715


namespace consecutive_integers_sum_l189_189884

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by
  sorry

end consecutive_integers_sum_l189_189884


namespace arithmetic_sequence_l189_189272

variables {a1 a2 a3 d : ℝ}

def arithmetic (a1 a2 a3 : ℝ) : Prop := a1 + a2 + a3 = 21 ∧ a1 * a2 * a3 = 231

theorem arithmetic_sequence (h : arithmetic a1 a2 a3) (h_inc : a1 < a2 < a3) :
  a2 = 7 ∧ ∀ n : ℕ, 1 < n → ( ∃ d, d > 0 ∧ ∀ k : ℕ, a1 = a2 - d ∧ a3 = a2 + d ∧ ((k : ℝ) = (4 * k - 1))) :=
by
  sorry

end arithmetic_sequence_l189_189272


namespace sum_distances_focus_parabola_l189_189763

theorem sum_distances_focus_parabola : 
  let P := λ x : ℝ, x^2
  in ∃ (C : ℝ × ℝ → Prop) (p1 p2 p3 p4 : ℝ × ℝ),
    (C (-15, 225) ∧ C (-1, 1) ∧ C (14, 196) ∧ C p4 ∧ p4 = (2, 4)) ∧
    (∀ x, C (x, P x) ↔ (C (x, x^2))) ∧
    let focus := (0, 0.25)
    in let distance := λ (p : ℝ × ℝ), ((p.1 - focus.1)^2 + (p.2 - focus.2)^2)^0.5
    in distance (-15, 225) + distance (-1, 1) + distance (14, 196) + distance (2, 4) = 427 :=
by
  sorry

end sum_distances_focus_parabola_l189_189763


namespace line_PE_tangent_to_circle_ACE_l189_189485

-- Definitions for circles, points, and internal tangency
variables {P A C D E : Type*}

-- Assume there exist two circles that are internally tangent at A
axiom small_circle : ∃ (s1 : S1), s1.center = \(A\)
axiom big_circle : ∃ (s2 : S2), s2.center = \(A\)
axiom internal_tangency : (small_circle ∧ big_circle)

-- Assume C is a point on the smaller circle other than A
axiom C_on_small_circle : ∃ (C : point), C ∈ small_circle ∧ C ≠ A

-- The tangent line to the smaller circle at C meets the bigger circle at points D and E
axiom tangent_C_meets_big_circle : C ∈ small_circle ∧ (D ∈ big_circle ∨ E ∈ big_circle)

-- The line AC meets the bigger circle at points A and P
axiom line_AC_meets_big_circle : A ∈ line AC ∧ P ∈ line AC

-- Define the circle passing through points A, C, and E
axiom circle_through_ACE : ∃ (circle_ACE : circle), A ∈ circle_ACE ∧ C ∈ circle_ACE ∧ E ∈ circle_ACE 

-- Prove the statement
theorem line_PE_tangent_to_circle_ACE : 
  ∀ (P E : point), (P ∈ line PE) → (E ∈ circle_ACE) → tangent (line PE) (circle_ACE) :=
begin
  sorry -- Proof not required
end

end line_PE_tangent_to_circle_ACE_l189_189485


namespace sum_of_consecutive_integers_with_product_812_l189_189915

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189915


namespace coordinates_of_points_l189_189142

theorem coordinates_of_points
  (R : ℝ) (a b : ℝ)
  (hR : R = 10)
  (h_area : 1/2 * a * b = 600)
  (h_a_gt_b : a > b) :
  (a, 0) = (40, 0) ∧ (0, b) = (0, 30) ∧ (16, 18) = (16, 18) :=
  sorry

end coordinates_of_points_l189_189142


namespace bankers_discount_correct_l189_189137

-- Define the true discount (TD) and the face value (FV)
def TD : ℝ := 90
def FV : ℝ := 540

-- Present value (PV) is inferred from TD and FV
def PV : ℝ := FV - TD

-- Define the banker's discount (BD) using the given formula
def BD : ℝ := TD + (TD^2 / PV)

-- Prove that the banker's discount (BD) is 108
theorem bankers_discount_correct : BD = 108 := 
by
  have h1 : PV = 450 := by norm_num [PV, TD, FV]
  have h2 : (90^2 : ℝ) = 8100 := by norm_num
  have h3 : (8100 / 450 : ℝ) = 18 := by norm_num [h2]
  simp [BD, TD, h1, h3]
  linarith

end bankers_discount_correct_l189_189137


namespace problem_l189_189845

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189845


namespace fourth_term_seq_l189_189274

noncomputable def seq : ℕ → ℝ
| 0       := 1
| (n + 1) := 1 / 2 * seq n + 1 / 2

theorem fourth_term_seq : seq 3 = 1 :=
by {
  sorry
}

end fourth_term_seq_l189_189274


namespace number_of_ways_to_select_leader_and_manager_l189_189159

theorem number_of_ways_to_select_leader_and_manager (n : ℕ) (h : n = 8) : ∃ k, k = 56 :=
by
  use 8 * 7
  have h1 : 8 * 7 = 56 := rfl
  rw [h1]
  sorry

end number_of_ways_to_select_leader_and_manager_l189_189159


namespace num_ways_to_choose_program_l189_189179

-- Define all the courses
inductive Course
  | English
  | Algebra
  | Geometry
  | Calculus
  | History
  | Art
  | Science
  | Latin

open Course

-- Define sets of courses for brevity
def mathCourses : Set Course := { Algebra, Geometry, Calculus }
def humanitiesCourses : Set Course := { History, Art, Latin }
def allCourses : Set Course := { English, Algebra, Geometry, Calculus, History, Art, Science, Latin }
def remainingCourses : Set Course := allCourses.erase English

-- Define the main theorem
theorem num_ways_to_choose_program : 
  (Set.card {s : Set Course | s.card = 5 ∧ English ∈ s ∧ (∃ m, m ∈ s ∧ m ∈ mathCourses) ∧ (∃ h, h ∈ s ∧ h ∈ humanitiesCourses) }) = 33 := 
sorry

end num_ways_to_choose_program_l189_189179


namespace find_divisor_l189_189325

theorem find_divisor (h : 2994 / 14.5 = 171) : 29.94 / 1.75 = 17.1 :=
by
  sorry

end find_divisor_l189_189325


namespace consecutive_integer_product_sum_l189_189930

theorem consecutive_integer_product_sum (n : ℕ) (h : n * (n + 1) = 812) : n + (n + 1) = 57 :=
sorry

end consecutive_integer_product_sum_l189_189930


namespace sum_of_consecutive_integers_l189_189879

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189879


namespace sum_of_second_pair_l189_189810

theorem sum_of_second_pair :
  (6 + 7 = 12) →
  (5 + 6 = 10) →
  (7 + 8 = 14) →
  (3 + 3 = 5) →
  8 + 9 = 16 :=
by {
  intros h1 h2 h3 h4,
  -- The proof should be constructed here, but it's omitted for now
  sorry
}

end sum_of_second_pair_l189_189810


namespace inequality_a_b_l189_189626

theorem inequality_a_b (a b : ℝ) (h : a > b ∧ b > 0) : (1/a) < (1/b) := 
by
  sorry

end inequality_a_b_l189_189626


namespace symmetric_center_of_f_range_of_f_not_real_f_is_monotonically_increasing_sum_of_f_values_l189_189665

noncomputable def f (x : ℝ) : ℝ := x / (x + 1)

theorem symmetric_center_of_f :
  ∃ x y, f(-1) = 1 :=
sorry

theorem range_of_f_not_real :
  set.range f ≠ set.univ :=
sorry

theorem f_is_monotonically_increasing :
  ∃ a b, ∀ x, a < x ∧ x < b → f(x) ≤ f(x + 1) :=
sorry

theorem sum_of_f_values :
  ∑ n in (finset.range 2022).image (λ n, f (n + 1)) + 
    ∑ n in (finset.range 2022).image (λ n, f (1 / (n + 1))) = 4043 / 2 :=
sorry

end symmetric_center_of_f_range_of_f_not_real_f_is_monotonically_increasing_sum_of_f_values_l189_189665


namespace construct_numbers_from_1_to_10_l189_189971

noncomputable def construct_using_threes (n : ℕ) : ℕ := 
  match n with
  | 1 => ((3 / 3) ^ 333)
  | 2 => ((3 - 3 / 3) * 3 / 3)
  | 3 => 3 * (3 / 3) * (3 / 3)
  | 4 => (3 + 3 / 3) * (3 / 3)
  | 5 => 3 + 3 / 3 + 3 / 3
  | 6 => (3 + 3) * (3 / 3)^3
  | 7 => (3 + 3) + (3 / 3)^3
  | 8 => (3 * 3) - (3 / 3)^3
  | 9 => (3 * 3) * (3 / 3)^3
  | 10 => (3 * 3) + (3 / 3)^3
  | _ => 0

theorem construct_numbers_from_1_to_10 : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 10 → construct_using_threes n = n :=
by
  intros n hn
  cases n
  { simp at hn, cases hn }
  all_goals { sorry }

end construct_numbers_from_1_to_10_l189_189971


namespace dot_product_a_a_sub_2b_l189_189653

-- Define the vectors a and b
def a : (ℝ × ℝ) := (2, 3)
def b : (ℝ × ℝ) := (-1, 2)

-- Define the subtraction of vector a and 2 * vector b
def a_sub_2b : (ℝ × ℝ) := (a.1 - 2 * b.1, a.2 - 2 * b.2)

-- Define the dot product of two vectors
def dot_product (u v : (ℝ × ℝ)) : ℝ := u.1 * v.1 + u.2 * v.2

-- State that the dot product of a and (a - 2b) is 5
theorem dot_product_a_a_sub_2b : dot_product a a_sub_2b = 5 := 
by 
  -- proof omitted
  sorry

end dot_product_a_a_sub_2b_l189_189653


namespace difference_of_numbers_l189_189463

theorem difference_of_numbers (a b : ℕ) (h1 : a + b = 20460) (h2 : b % 12 = 0) (h3 : b / 10 = a) : b - a = 17314 :=
by
  sorry

end difference_of_numbers_l189_189463


namespace email_ratio_is_one_to_two_l189_189434

def initial_emails : ℕ := 400
def remaining_emails_after_trash (T : ℕ) : ℕ := initial_emails - T
def emails_moved_to_work_folder (T : ℕ) : ℕ := (4 * (initial_emails - T)) / 10
def emails_left_in_inbox (T : ℕ) : ℕ := remaining_emails_after_trash T - emails_moved_to_work_folder T
def ratio_of_trash_to_initial (T : ℕ) : ℕ → ℕ → Prop := λ a b, a / 200 = 1 ∧ b / 200 = 2

theorem email_ratio_is_one_to_two (T : ℕ) (h : emails_left_in_inbox T = 120) :
  ratio_of_trash_to_initial T initial_emails :=
by
  sorry

end email_ratio_is_one_to_two_l189_189434


namespace problem_l189_189843

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189843


namespace sufficient_condition_for_sets_l189_189677

theorem sufficient_condition_for_sets (A B : Set ℝ) (m : ℝ) :
    (∀ x, x ∈ A → x ∈ B) → (m ≥ 3 / 4 ∨ m ≤ -3 / 4) :=
by
    have A_def : A = {y | ∃ x, y = x^2 - (3 / 2) * x + 1 ∧ (1 / 4) ≤ x ∧ x ≤ 2} := sorry
    have B_def : B = {x | x ≥ 1 - m^2} := sorry
    sorry

end sufficient_condition_for_sets_l189_189677


namespace consecutive_integers_sum_l189_189855

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189855


namespace correct_calculation_result_l189_189538

theorem correct_calculation_result (x : ℝ) (h : x / 12 = 8) : 12 * x = 1152 :=
sorry

end correct_calculation_result_l189_189538


namespace cory_packs_l189_189586

theorem cory_packs (total_money_needed cost_per_pack : ℕ) (h1 : total_money_needed = 98) (h2 : cost_per_pack = 49) : total_money_needed / cost_per_pack = 2 :=
by 
  sorry

end cory_packs_l189_189586


namespace minimum_value_expr_min_value_reachable_l189_189773

noncomputable def expr (x y : ℝ) : ℝ :=
  4 * x^2 + 9 * y^2 + 16 / x^2 + 6 * y / x

theorem minimum_value_expr (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  expr x y ≥ (2 * Real.sqrt 564) / 3 :=
sorry

theorem min_value_reachable :
  ∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ expr x y = (2 * Real.sqrt 564) / 3 :=
sorry

end minimum_value_expr_min_value_reachable_l189_189773


namespace train_speed_l189_189982

theorem train_speed (D T : ℝ) (h1 : D = 160) (h2 : T = 16) : D / T = 10 :=
by 
  -- given D = 160 and T = 16, we need to prove D / T = 10
  sorry

end train_speed_l189_189982


namespace no_possible_values_for_b_l189_189038

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_possible_values_for_b_l189_189038


namespace area_ratio_of_triangle_angle_bisector_l189_189356

theorem area_ratio_of_triangle_angle_bisector
  (A B C D : Type)
  [triangle ABC]
  (AB AC BC : ℝ)
  (hAB : AB = 16)
  (hAC : AC = 24)
  (hBC : BC = 19)
  (hAD_bisector : is_angle_bisector_of_triangle_side A B C D) :
  area_ratio_of_angle_bisector_triangles A B C D = 2 / 3 :=
sorry

end area_ratio_of_triangle_angle_bisector_l189_189356


namespace complex_value_of_z_six_plus_z_inv_six_l189_189695

open Complex

theorem complex_value_of_z_six_plus_z_inv_six (z : ℂ) (h : z + z⁻¹ = 1) : z^6 + (z⁻¹)^6 = 2 := by
  sorry

end complex_value_of_z_six_plus_z_inv_six_l189_189695


namespace polyhedron_not_necessarily_regular_l189_189044

/--
  A polyhedron with equal regular polygonal faces does not necessarily imply that
  the polyhedron is a regular polyhedron.
-/
theorem polyhedron_not_necessarily_regular (P : Type) [polyhedron P] 
  (H : ∀ f g : face P, is_regular_polygon f ∧ congruent f g) :
  ¬ (is_regular_polyhedron P) :=
sorry

end polyhedron_not_necessarily_regular_l189_189044


namespace area_relation_triangle_l189_189418

-- Define the proof problem in Lean
theorem area_relation_triangle (
  (A B C D E F : Type) [Triangle A B C] [Point D] [Point E] [Point F] 
  (hE_AC : E ∈ LineSegment A C)
  (hDE_parallel_BC : Parallel Line DE Line BC)
  (hEF_parallel_AB : Parallel Line EF Line AB)
  (S_ADE S_BDE S_EFC S_BDEF : ℝ)
  (hBDEF : S_BDEF = S_BDE + S_EFC)
  (hS_BDE_ADE : S_BDE = S_ADE)
  ) : S_BDEF = 2 * sqrt (S_ADE * S_EFC) :=
  sorry

end area_relation_triangle_l189_189418


namespace boat_speed_still_water_l189_189135

/-- Proof that the speed of the boat in still water is 10 km/hr given the conditions -/
theorem boat_speed_still_water (V_b V_s : ℝ) 
  (cond1 : V_b + V_s = 15) 
  (cond2 : V_b - V_s = 5) : 
  V_b = 10 :=
by
  sorry

end boat_speed_still_water_l189_189135


namespace sum_of_squares_eq_product_l189_189459

-- Sequence definition
def x : ℕ → ℕ
| 0     := 1
| 1     := 2
| 2     := 3
| 3     := 4
| 4     := 5
| (n+5) := (List.prod (List.map x (List.range (n + 5)))) - 1

-- Theorem statement
theorem sum_of_squares_eq_product : 
  (∑ i in Finset.range 70, (x i) * (x i)) = (List.prod (List.map x (List.range 70))) := 
sorry

end sum_of_squares_eq_product_l189_189459


namespace sum_of_numbers_divisible_by_7_l189_189961

def cards : Set ℕ := {1, 2, 3, 5}

def valid_two_digit_numbers : Set ℕ :=
  { n | ∃ (a b : ℕ), a ∈ cards ∧ b ∈ cards ∧ a ≠ b ∧ n = 10 * a + b }

def divisible_by_7 (n : ℕ) : Prop := n % 7 = 0

theorem sum_of_numbers_divisible_by_7 :
  (∑ (n : ℕ) in valid_two_digit_numbers.filter divisible_by_7, n) = 56 :=
by
  sorry

end sum_of_numbers_divisible_by_7_l189_189961


namespace days_worked_together_l189_189153

noncomputable def work_completed (rate_per_day: ℚ) (days: ℚ): ℚ := rate_per_day * days

theorem days_worked_together 
  (A_rate: ℚ) (B_rate: ℚ) (C_rate: ℚ) (y: ℚ)
  (hA: A_rate = 1/20) 
  (hB: B_rate = 1/15)
  (hC: C_rate = 1/50)
  (hY: y = 5):
  ∃ (x: ℚ), 
    (work_completed (A_rate + B_rate) x) + (work_completed (A_rate + C_rate) y) = 1 ∧
    x ≈ 5 := 
begin
  sorry
end

end days_worked_together_l189_189153


namespace sum_of_consecutive_integers_with_product_812_l189_189945

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189945


namespace train_length_is_110_l189_189553

noncomputable def km_per_hour_to_m_per_second (speed_km_per_h : ℕ) : ℝ :=
  speed_km_per_h * 1000 / 3600

def train_length
  (train_speed_kmh : ℕ)
  (man_speed_kmh : ℕ)
  (time_seconds : ℕ) : ℝ :=
  let relative_speed_kmh := train_speed_kmh + man_speed_kmh
  let relative_speed_mps := km_per_hour_to_m_per_second relative_speed_kmh
  relative_speed_mps * time_seconds

theorem train_length_is_110
  (train_speed_kmh : ℕ)
  (man_speed_kmh : ℕ)
  (time_seconds : ℕ)
  (h_train_speed : train_speed_kmh = 30)
  (h_man_speed : man_speed_kmh = 3)
  (h_time : time_seconds = 12) :
  train_length train_speed_kmh man_speed_kmh time_seconds ≈ 110 :=
by
  sorry

end train_length_is_110_l189_189553


namespace inverse_function_sqrt3_l189_189666

-- Defining the function f
def f (x : ℝ) : ℝ := 3^x

-- Inverse function g that we need to prove g(√3) = 1/2
def g (x : ℝ) : ℝ := classical.some (exists_unique_inverse 3)

-- Statement to be proven
theorem inverse_function_sqrt3 : g (√3) = 1/2 :=
sorry

end inverse_function_sqrt3_l189_189666


namespace k_values_for_set_A_l189_189676

theorem k_values_for_set_A (k : ℝ) :
  (∀ x, (k+2) * x^2 + 2 * k * x + 1 = 0 → x ∈ (A : Set ℝ)) →
  (card A = 1 → (k = -2 ∨ k = -1 ∨ k = 2))
:=
by
  intros hA hCard
  sorry

end k_values_for_set_A_l189_189676


namespace childSupportOwed_l189_189409

def annualIncomeBeforeRaise : ℕ := 30000
def yearsBeforeRaise : ℕ := 3
def raisePercentage : ℕ := 20
def annualIncomeAfterRaise (incomeBeforeRaise raisePercentage : ℕ) : ℕ :=
  incomeBeforeRaise + (incomeBeforeRaise * raisePercentage / 100)
def yearsAfterRaise : ℕ := 4
def childSupportPercentage : ℕ := 30
def amountPaid : ℕ := 1200

def calculateChildSupport (incomeYears : ℕ → ℕ → ℕ) (supportPercentage : ℕ) (years : ℕ) : ℕ :=
  (incomeYears years supportPercentage) * supportPercentage / 100 * years

def totalChildSupportOwed : ℕ :=
  (calculateChildSupport (λ _ _ => annualIncomeBeforeRaise) childSupportPercentage yearsBeforeRaise) +
  (calculateChildSupport (λ _ _ => annualIncomeAfterRaise annualIncomeBeforeRaise raisePercentage) childSupportPercentage yearsAfterRaise)

theorem childSupportOwed : totalChildSupportOwed - amountPaid = 69000 :=
by trivial

end childSupportOwed_l189_189409


namespace r_minus_s_l189_189770

-- Define the equation whose roots are r and s
def equation (x : ℝ) := (6 * x - 18) / (x ^ 2 + 4 * x - 21) = x + 3

-- Define the condition that r and s are distinct roots of the equation and r > s
def is_solution_pair (r s : ℝ) :=
  equation r ∧ equation s ∧ r ≠ s ∧ r > s

-- The main theorem we need to prove
theorem r_minus_s (r s : ℝ) (h : is_solution_pair r s) : r - s = 12 :=
by
  sorry

end r_minus_s_l189_189770


namespace total_worth_of_presents_l189_189754

-- Define the costs as given in the conditions
def ring_cost : ℕ := 4000
def car_cost : ℕ := 2000
def bracelet_cost : ℕ := 2 * ring_cost

-- Define the total worth of the presents
def total_worth : ℕ := ring_cost + car_cost + bracelet_cost

-- Statement: Prove the total worth is 14000
theorem total_worth_of_presents : total_worth = 14000 :=
by
  -- Here is the proof statement
  sorry

end total_worth_of_presents_l189_189754


namespace computation_of_difference_of_squares_l189_189576

theorem computation_of_difference_of_squares : (65^2 - 35^2) = 3000 := sorry

end computation_of_difference_of_squares_l189_189576


namespace XY_parallel_AD_l189_189993

theorem XY_parallel_AD
  (A B C D X Y K : Point)
  (h_inscribed: ¬Collinear A B C ∧ ¬Collinear B C D ∧ ¬Collinear C D A ∧ ¬Collinear D A B)
  (h_cyclic: CyclicQuadrilateral A B C D)
  (h_midpoint: MidpointArc K A D)
  (h_interX: Intersects (Line B K) (Diagonal A C) X)
  (h_interY: Intersects (Line C K) (Diagonal B D) Y) :
  Parallel (Line X Y) (Line A D) := 
  sorry

end XY_parallel_AD_l189_189993


namespace minimum_value_quadratic_l189_189251

theorem minimum_value_quadratic (x y : ℝ) : ∃ z : ℝ, z = 0 ∧ (∀ a b : ℝ, a^2 + 2*a*b + 2*b^2 ≥ z) :=
begin
  use 0,
  split,
  { refl },
  { intros a b,
    -- Skipping the actual proof, we just say "sorry" here
    sorry
  }
end

end minimum_value_quadratic_l189_189251


namespace largest_not_sum_of_36_and_composite_l189_189114

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l189_189114


namespace geometric_sequence_product_l189_189784

variable (b : ℕ → ℝ)
variable (n : ℕ)

def geometric_product (b_1 b_n : ℝ) (n : ℕ) : ℝ :=
  Real.sqrt ((b_1 * b_n) ^ n)

theorem geometric_sequence_product (b_1 b_n : ℝ) (n : ℕ) (hb_n_pos : b_n > 0) (hn_pos: n > 0) :
  (∏ i in Finset.range n, b i) = geometric_product b_1 b_n n :=
sorry

end geometric_sequence_product_l189_189784


namespace part1_cosine_identity_part2_parallel_condition_l189_189615

-- Vectors a and b
def vec_a : ℝ × ℝ × ℝ := (1, 4, -2)
def vec_b : ℝ × ℝ × ℝ := (-2, 2, 4)

-- Vector c = 1/2 * b
def vec_c : ℝ × ℝ × ℝ := (-1, 1, 2)

-- Part 1: Prove cos(⟨a, c⟩) = -sqrt(14)/42
theorem part1_cosine_identity :
  let a := vec_a
  let b := vec_b
  let c := vec_c
  Real.cosine a c = -Real.sqrt 14 / 42 := sorry

-- Part 2: Prove that the value of k such that (k * a + b) is parallel to (a - 3 * b) is -1/3
theorem part2_parallel_condition :
  let a := vec_a
  let b := vec_b
  \exists k : ℝ, (k * a + b) ∥ (a - 3 * b) ∧ k = -1/3 := sorry

end part1_cosine_identity_part2_parallel_condition_l189_189615


namespace stock_investment_l189_189451

theorem stock_investment (market_value brokerage_rate income stock_percentage actual_market_value investment_amount : ℝ) 
  (h_market_value : market_value = 103.91666666666667)
  (h_brokerage_rate : brokerage_rate = 0.25 / 100)
  (h_income : income = 756)
  (h_stock_percentage : stock_percentage = 10.5)
  (h_actual_market_value : actual_market_value = market_value * (1 + brokerage_rate))
  (h_investment_amount : income = (investment_amount * stock_percentage) / 100) :
  investment_amount = 7200 :=
by 
  have h1 : actual_market_value = 103.91666666666667 * (1 + 0.0025), from h_actual_market_value,
  have h2 : 756 = (investment_amount * 10.5) / 100, from h_investment_amount,
  sorry

end stock_investment_l189_189451


namespace polar_to_cartesian_distance_l189_189340

noncomputable def distance (x₁ y₁ a b c : ℝ) : ℝ :=
  abs (a * x₁ + b * y₁ + c) / real.sqrt (a^2 + b^2)

theorem polar_to_cartesian_distance :
  let P := (2, (3 * real.pi / 2))
  let l := (3, -4, 3)
  let x := 2 * real.cos (3 * real.pi / 2)
  let y := 2 * real.sin (3 * real.pi / 2)
  distance x y l.fst l.snd l.snd.snd = 1 :=
by
  let P := (2 : ℝ, (3 * real.pi / 2))
  let l := (3 : ℝ, -4, 3)
  let x := 2 * real.cos (3 * real.pi / 2)
  let y := 2 * real.sin (3 * real.pi / 2)
  have h1 : x = 0 := by sorry
  have h2 : y = -2 := by sorry
  rw [h1, h2]
  have h3 : distance 0 (-2) 3 (-4) 3 = 1 := by sorry
  exact h3

end polar_to_cartesian_distance_l189_189340


namespace sum_prob_less_one_l189_189394

theorem sum_prob_less_one (x y z : ℝ) (hx : 0 < x ∧ x < 1) (hy : 0 < y ∧ y < 1) (hz : 0 < z ∧ z < 1) :
  x * (1 - y) * (1 - z) + (1 - x) * y * (1 - z) + (1 - x) * (1 - y) * z < 1 :=
by
  sorry

end sum_prob_less_one_l189_189394


namespace smallest_n_terminating_decimal_l189_189499

theorem smallest_n_terminating_decimal :
  ∃ n : ℕ, (0 < n) ∧ (∀ m : ℕ, (0 < m) ∧ (∃ k : ℕ, m = k + 107 ∧ ∀ p : ℕ, (nat.prime p) → ((p ∣ m) → (p = 2 ∨ p = 5))) → n ≤ m) ∧ n = 143 := 
sorry

end smallest_n_terminating_decimal_l189_189499


namespace rectangular_prism_diagonal_length_l189_189546

-- Definitions based on given conditions
def length : ℝ := 12
def width : ℝ := 15
def height : ℝ := 8

-- Theorem to prove the length of the diagonal
theorem rectangular_prism_diagonal_length :
  let base_diagonal := Real.sqrt (length^2 + width^2) in
  let space_diagonal := Real.sqrt (base_diagonal^2 + height^2) in
  space_diagonal = Real.sqrt 433 := by
  -- Skip the proof with sorry
  sorry

end rectangular_prism_diagonal_length_l189_189546


namespace proof_problem_l189_189292

noncomputable def a_n (n: ℕ) : ℕ := 2 * n
noncomputable def S_n (n: ℕ) : ℕ := n^2 + n
noncomputable def b_n (n: ℕ) : ℕ := 2^(n - 1) + 1
noncomputable def T_n (n: ℕ) : ℕ := 2^n + n - 1

-- Adding necessary assumptions as definitions.
def a1 : ℕ := 2
def a4 : ℕ := a_n 4
def S5 : ℕ := S_n 5
def geometric_condition : Prop := a4^2 = a1 * (S5 + 2)
def increasing_arithmetic_sequence : Prop := ∀ n m: ℕ, n < m → a_n n < a_n m

-- Formal statement of the proof problem
theorem proof_problem (n : ℕ) (h_ge_5 : n ≥ 5) :
  a_n n = 2 * n ∧
  S_n n = n^2 + n ∧
  T_n n = 2^n + n - 1 ∧
  ( ∀ n : ℕ, n ≥ 5 → T_n n > S_n n ) := 
sorry

end proof_problem_l189_189292


namespace norm_two_u_l189_189618

variable {E : Type*} [NormedAddCommGroup E]

theorem norm_two_u (u : E) (h : ∥u∥ = 5) : ∥2 • u∥ = 10 := sorry

end norm_two_u_l189_189618


namespace hyperbola_eq_l189_189776

theorem hyperbola_eq {a b : ℝ} (h1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1 → (x = a ∨ x = -a))
  (h2 : ∀ x y : ℝ, y ≥ 1) 
  (h3 : ∀ C D : ℝ × ℝ, (P = (0,1) → C ≠ D)
  (h4 : ∀ C D : ℝ × ℝ, tangent_line_at C ≠ tangent_line_at D)
  (h5 : area_equilateral_triangle Q C D = 16 * √3 / 27) :
  \(\frac{27}{4} x^2 - 3 y^2 = 1\) := 
begin
  sorry,
end

end hyperbola_eq_l189_189776


namespace sum_of_two_numbers_l189_189057

theorem sum_of_two_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h1 : x * y = 16)
  (h2 : 1/x = 3 * (1/y)) : 
  x + y = (16 * Real.sqrt 3) / 3 := 
by
  sorry

end sum_of_two_numbers_l189_189057


namespace work_completion_days_l189_189155

theorem work_completion_days (A_days : ℕ) (B_days : ℕ) (hA : A_days = 6) (hB : B_days = 12) : 
  let A_rate := 1 / (A_days : ℝ)
  let B_rate := 1 / (B_days : ℝ)
  let combined_rate := A_rate + B_rate
  let days_to_complete := 1 / combined_rate
  days_to_complete = 4 :=
by {
  rw [hA, hB],
  have A_rate : A_rate = 1/6 := by norm_num,
  have B_rate : B_rate = 1/12 := by norm_num,
  have combined_rate : combined_rate = 1/4 := by { rw [A_rate, B_rate], norm_num },
  have days_to_complete : days_to_complete = 4 := by { rw [combined_rate], norm_num },
  exact days_to_complete }

end work_completion_days_l189_189155


namespace consecutive_integer_sum_l189_189868

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189868


namespace consecutive_integers_sum_l189_189854

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189854


namespace integer_part_quotient_l189_189051

theorem integer_part_quotient :
  let a := 0.40
  let l := 0.59
  let d := 0.01
  let num_terms := ((l - a) / d).to_nat + 1
  let S_n := (num_terms:ℝ) / 2 * (a + l)
  let quotient := 28.816 / S_n
  quotient.floor = 2 :=
by
  let a := 0.40
  let l := 0.59
  let d := 0.01
  let num_terms := ((l - a) / d).to_nat + 1
  let S_n := (num_terms:ℝ) / 2 * (a + l)
  let quotient := 28.816 / S_n
  have h : num_terms = 20 := sorry
  have hS_n : S_n = 9.9 := sorry
  have hquotient : quotient = 28.816 / 9.9 := by rw [hS_n]
  have hfloor : (28.816 / 9.9).floor = 2 := sorry
  exact hfloor

end integer_part_quotient_l189_189051


namespace consecutive_integers_sum_l189_189859

theorem consecutive_integers_sum (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 := by
  sorry

end consecutive_integers_sum_l189_189859


namespace smallest_of_even_set_l189_189452

/--
Proposition:
Given a set of consecutive even integers where the median is 160 and the greatest integer is 170,
prove that the smallest integer in the set is 152.
-/
theorem smallest_of_even_set (median greatest : ℤ) (h1 : median = 160) (h2 : greatest = 170) :
  ∃ (smallest : ℤ), smallest = 152 := 
by
  use 152
  apply eq.refl 152

end smallest_of_even_set_l189_189452


namespace ordered_triples_count_l189_189762

theorem ordered_triples_count :
  (∃ (a b c : ℕ), (nat.lcm a b = 2000) ∧ (nat.lcm b c = 4000) ∧ (nat.lcm c a = 4000) ∧ 
  (let ⟨j,k⟩ := (nat.find_pick_exp a 2), ⟨m,n⟩ := (nat.find_pick_exp b 2), ⟨p,q⟩ := (nat.find_pick_exp c 2) in 
      j + m + p = 12))
  → ∃! count : ℕ, count = 10 :=
begin
  sorry
end

end ordered_triples_count_l189_189762


namespace min_max_abs_value_l189_189778

theorem min_max_abs_value (a b : ℝ) :
  ∀ a b, ∃ (a0 : ℝ) (b0 : ℝ), (max (abs (a0 + b0)) (max (abs (a0 - b0)) (abs (1 - b0)))) = 1/2 ∧ 
  (∀ a b, (max (abs (a + b)) (max (abs (a - b)) (abs (1 - b))) ≥ 1/2)) :=
sorry

end min_max_abs_value_l189_189778


namespace sum_of_integers_strictly_monotonic_sequences_l189_189647

noncomputable def sumOfSoughtIntegers : ℕ := 
  984374748

theorem sum_of_integers_strictly_monotonic_sequences (n : ℕ) (m : ℕ) (a : Fin (m + 1) → Fin 9) :
  (n = ∑ i in Finset.range (m + 1), (a i : ℕ) * 9^i) ∧ 
  (∀ i j, n < j → i < j → a i < a j) ∨ (∀ i j, i < j → a i > a j) →
  ∑ n = 984374748 := 
sorry

end sum_of_integers_strictly_monotonic_sequences_l189_189647


namespace find_b_and_range_l189_189683

def vec_a : ℝ × ℝ := (2, 2)
def angle_ba : ℝ := (3 * Real.pi / 4)
def dot_a_b (b : ℝ × ℝ) : ℝ := (vec_a.1 * b.1 + vec_a.2 * b.2)
def vec_t : ℝ × ℝ := (1, 0)
def perp_b_t (b : ℝ × ℝ) : Prop := (b.1 * vec_t.1 + b.2 * vec_t.2) = 0
def angles_arith_seq (A B C : ℝ) : Prop := (B = (A + C) / 2)
def vec_c (A C : ℝ) : ℝ × ℝ := (Real.cos A, Real.cos (C / 2) ^ 2)

noncomputable def vec_b : ℝ × ℝ := (0, -1)

theorem find_b_and_range (A B C : ℝ) 
  (h1 : angle_ba = 3 * Real.pi / 4)
  (h2 : dot_a_b vec_b = -2)
  (h3 : perp_b_t vec_b)
  (h4 : angles_arith_seq A B C)
  (h5 : B = Real.pi / 3) :
  vec_b = (0, -1) ∧ (∀ (vec_c_A_C : ℝ × ℝ), vec_c_A_C = vec_c A C → Real.sqrt 2 / 2 ≤ ∥vec_b + vec_c_A_C∥ < Real.sqrt 5 / 2) :=
sorry

end find_b_and_range_l189_189683


namespace students_in_section_A_l189_189067

-- Define the relevant conditions as variables and constants
variables (x : ℕ) -- Number of students in section A
constant b_students : ℕ := 34 -- Number of students in section B
constant a_average_weight : ℚ := 50 -- Average weight of section A
constant b_average_weight : ℚ := 30 -- Average weight of section B
constant total_average_weight : ℚ := 38.67 -- Average weight of the whole class

-- Define the relation for the total weight of both sections
def total_weight_A : ℚ := a_average_weight * x
def total_weight_B : ℚ := b_average_weight * b_students
def total_weight_class : ℚ := total_weight_A + total_weight_B
def total_students_class : ℕ := x + b_students

-- Define the equation based on the average weight of the whole class
def average_weight_equation : Prop :=
  total_average_weight * total_students_class = total_weight_class

-- The goal is to prove that x = 26
theorem students_in_section_A : x = 26 :=
  sorry

end students_in_section_A_l189_189067


namespace proof_part1_proof_part2_l189_189764

-- Defining the given geometric and trigonometric conditions
variables {A B C : ℝ}  -- Angles of the triangle ABC
variables {a b c : ℝ}  -- Sides of the triangle opposite to angles A, B, C respectively

-- Given conditions
noncomputable def condition1 : Prop :=
  3 * (sin A ^ 2 + sin C ^ 2) = 2 * (sin A * sin C) + 8 * (sin A * sin C * cos B)
  
noncomputable def condition2 : Prop :=
  cos B = 11 / 14

noncomputable def area_triangle : ℝ :=
  15 / 4 * sqrt 3

-- Proving the statement: a + c = 2b
theorem proof_part1 (h1 : condition1) : a + c = 2 * b := sorry

-- Finding the perimeter of △ABC
theorem proof_part2 (h1 : condition1) (h2 : condition2) (h3 : area (a, b, c) = area_triangle) : a + b + c = 15 := sorry

end proof_part1_proof_part2_l189_189764


namespace tangent_line_slope_ln_l189_189461

theorem tangent_line_slope_ln (x : ℝ) (h : x = 3) : 
  ∃ (m : ℝ), (∀ (x : ℝ), m = (deriv (fun x => log x)) 3) ∧ m = 1 / 3 :=
by
  sorry

end tangent_line_slope_ln_l189_189461


namespace fraction_of_AC_l189_189353

variables {Point : Type}
variables (A B C D A1 E1 : Point)
variables (division_points : Fin 2021 → Point)
variable (AD_segment_division : ∀ i, division_points i ∈ line_segment A D)
variable (BA1_AC_intersection : collinear B A1 E1 ∧ collinear A C E1)
variable (parallelogram : parallelogram_points A B C D)


theorem fraction_of_AC (hAD_division : ∀ i, division_points i = A + i • ((D - A) / 2021))
  (hE1_intersection : E1 = intersection_point (line_through B A1) (line_through A C)) :
  AE1 = (1 / 2022 : ℚ) * AC :=
sorry

end fraction_of_AC_l189_189353


namespace sum_of_consecutive_integers_with_product_812_l189_189947

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189947


namespace consecutive_integers_sum_l189_189894

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189894


namespace units_digit_17_pow_35_l189_189978

theorem units_digit_17_pow_35 : (17 ^ 35) % 10 = 3 := by
sorry

end units_digit_17_pow_35_l189_189978


namespace integral_one_over_x_from_inv_e_to_e_l189_189595

open Real
open IntervalIntegrable

theorem integral_one_over_x_from_inv_e_to_e : 
  (∫ x in (1 : ℝ) / real.exp 1 .. real.exp 1, 1 / x) = 2 := 
by
  sorry

end integral_one_over_x_from_inv_e_to_e_l189_189595


namespace find_c_value_l189_189829

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_c_value (c : ℝ) : 
  (∀ x y : ℝ, 2 * x + 3 * y = c → 
  (∃ m1 m2 : ℝ × ℝ, (m1 = (2, 6)) ∧ (m2 = (8, 10)) ∧ (midpoint m1 m2) = (x, y))) → 
  c = 34 :=
by
  sorry

end find_c_value_l189_189829


namespace largest_positive_integer_not_sum_of_36_and_composite_l189_189105

theorem largest_positive_integer_not_sum_of_36_and_composite :
  ∃ n : ℕ, n = 187 ∧ ∀ a (ha : a ∈ ℕ), ∀ b (hb : b ∈ ℕ) (h0 : 0 ≤ b) (h1: b < 36) (hcomposite: ∀ d, d ∣ b → d = 1 ∨ d = b), n ≠ 36 * a + b :=
sorry

end largest_positive_integer_not_sum_of_36_and_composite_l189_189105


namespace probability_three_same_color_l189_189314

-- Conditions
def red_plates : ℕ := 7
def blue_plates : ℕ := 5
def total_plates : ℕ := red_plates + blue_plates

-- Question restated as a proof problem.
theorem probability_three_same_color (r b : ℕ) (h_r : r = red_plates) (h_b : b = blue_plates) :
  let total_plates := r + b in
  let total_combinations := Nat.choose total_plates 3 in
  let red_combinations := Nat.choose r 3 in
  let blue_combinations := Nat.choose b 3 in
  (red_combinations + blue_combinations).toRat / total_combinations.toRat = (9 : ℚ) / 44 :=
sorry

end probability_three_same_color_l189_189314


namespace mu_minus_lambda_eq_half_l189_189359

variables {R : Type*} [Field R] [AddGroup R] [Module R R]
variables (A B C M : R) (λ μ : R)

def vector_AM_eq_3_MC : Prop := (A - M) = 3 * (M - C)
def vector_BM_eq_linear_combination : Prop := (B - M) = λ * (B - A) + μ * (B - C)

theorem mu_minus_lambda_eq_half
  (h1 : vector_AM_eq_3_MC A C M)
  (h2 : vector_BM_eq_linear_combination B A C M λ μ) :
  μ - λ = (1 / 2) :=
sorry

end mu_minus_lambda_eq_half_l189_189359


namespace max_non_empty_subsets_l189_189270

/-- Given n non-empty subsets of the set A = {1, 2, ..., 10}, 
we want to prove that the maximum n such that for any i, j,
A_i ∪ A_j ≠ A is 511. -/
theorem max_non_empty_subsets (A : Set ℕ) (hA : A = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}) :
  ∃ (n : ℕ) (A₁ A₂ ... An : Set ℕ),
  (∀ i j ∈ finset.range n, i ≠ j → A₁ ∪ A₂ ≠ A) → n ≤ 511 :=
sorry

end max_non_empty_subsets_l189_189270


namespace problem_statement_l189_189814

noncomputable def g : ℕ → ℕ
| 1 := 4
| 3 := 2
| 4 := 6
| _ := 0 -- handling any undefined g(x) as default 0 for simplicity

theorem problem_statement : 
  let (e, f) := (1, g(g 1))
  let (g_pt, h) := (3, g(g 3))
  in ef + gh = 24 :=
by
  let e := 1
  let f := g(g e)
  let g_pt := 3
  let h := g(g g_pt)
  have h1 : g 1 = 4 := rfl
  have h2 : g 3 = 2 := rfl
  have h3 : g 4 = 6 := rfl
  have f_def : f = 6,
  { rw [h1, h3] }
  have h_def : h = 6,
  {
    -- Assuming g(2) = 6 as a consistent value based on given context information
    let g_res := 2
    have h4 : g g_res = 6 := rfl
    exact h4
  }
  unfold ef gh
  rw [f_def, h_def]
  simp
  exact rfl

end problem_statement_l189_189814


namespace brandy_used_0_17_pounds_of_chocolate_chips_l189_189197

def weight_of_peanuts : ℝ := 0.17
def weight_of_raisins : ℝ := 0.08
def total_weight_of_trail_mix : ℝ := 0.42

theorem brandy_used_0_17_pounds_of_chocolate_chips :
  total_weight_of_trail_mix - (weight_of_peanuts + weight_of_raisins) = 0.17 :=
by
  sorry

end brandy_used_0_17_pounds_of_chocolate_chips_l189_189197


namespace root_of_function_implies_value_set_theta_l189_189304

noncomputable def value_set_theta : set ℝ :=
  { θ | ∃ k : ℤ, θ = (Real.pi / 2) + k * Real.pi }

theorem root_of_function_implies_value_set_theta (θ : ℝ) :
  (∃ x : ℝ, x^2 - 2 * x * Real.sin θ + 1 = 0) ↔ θ ∈ value_set_theta :=
by
  sorry

end root_of_function_implies_value_set_theta_l189_189304


namespace unvisited_cell_is_C1_l189_189158

-- Define the coordinates on the board.
structure Cell where
  x : Nat
  y : Nat
  deriving DecidableEq, Repr

-- Define the 5x5 board
def Board : Type := Cell

-- Define the neighboring cells for movement
def neighbors (c : Cell) : List Cell :=
  [ { x := c.x + 1, y := c.y },
    { x := c.x - 1, y := c.y },
    { x := c.x, y := c.y + 1 },
    { x := c.x, y := c.y - 1 } ].filter (λ n, n.x < 5 ∧ n.y < 5 ∧ n.x ≥ 0 ∧ n.y ≥ 0)

-- Define the main theorem
theorem unvisited_cell_is_C1 : ∀ (start : Cell) (path : List Cell),
  start ∈ path ∧
  (∀ c ∈ path, c ∈ neighbors start) ∧
  (path.head = start ∧ path.tail.last = start) ∧
  (∀ c ∈ path, c ≠ start → path.count c ≤ 1) →
  ∃ unvisited : Cell, unvisited = {x := 2, y := 0} :=
by
  intros start path cond
  sorry

end unvisited_cell_is_C1_l189_189158


namespace norm_two_u_l189_189621

noncomputable def vector_u : ℝ × ℝ := sorry

theorem norm_two_u {u : ℝ × ℝ} (hu : ∥u∥ = 5) : ∥(2 : ℝ) • u∥ = 10 := by
  sorry

end norm_two_u_l189_189621


namespace monika_watched_three_movies_l189_189402

-- Define the constants based on the conditions
def mall_expense : ℝ := 250
def beans_price : ℝ := 1.25
def bags_bought : ℝ := 20
def total_expense : ℝ := 347
def movie_ticket_price : ℝ := 24

-- Define the derived quantities
def farmer_market_expense := beans_price * bags_bought
def before_movies_expense := mall_expense + farmer_market_expense
def movies_expense := total_expense - before_movies_expense

-- Define the number of movies Monika watched
def number_of_movies := movies_expense / movie_ticket_price

-- State the main theorem
theorem monika_watched_three_movies : number_of_movies = 3 :=
by
  -- Proof goes here
  sorry

end monika_watched_three_movies_l189_189402


namespace kitchen_square_footage_l189_189587

-- Definitions and conditions from the problem
def kitchen_cost : ℕ := 20000
def bath_cost : ℕ := 12000
def bath_footage : ℕ := 150
def other_cost_per_sf : ℕ := 100
def total_footage : ℕ := 2000
def total_cost : ℕ := 174000
def num_bathrooms : ℕ := 2

-- The statement to be proven
theorem kitchen_square_footage (K R : ℕ) 
  (h1 : kitchen_cost = 20000) 
  (h2 : bath_cost = 12000) 
  (h3 : bath_footage = 150) 
  (h4 : other_cost_per_sf = 100) 
  (h5 : total_footage = 2000) 
  (h6 : total_cost = 174000)
  (h7 : num_bathrooms = 2)
  (h8 : 20000 + 2 * 12000 + 100 * R = 174000)
  (h9 : K + 2 * 150 + R = 2000) :
  K = 400 :=
begin
  -- proof would go here, but we will use sorry for now
  sorry
end

end kitchen_square_footage_l189_189587


namespace sequence_integer_terms_l189_189175

theorem sequence_integer_terms
    (a : ℕ → ℝ)
    (h1 : a 1 = 1)
    (hrec : ∀ n ≥ 1, a (n + 1) = 2 * a n + real.sqrt (3 * (a n) ^ 2 + 1)) :
    ∀ n, ∃ m : ℤ, a n = m :=
by
  sorry

end sequence_integer_terms_l189_189175


namespace students_interested_in_both_l189_189527

def numberOfStudentsInterestedInBoth (T S M N: ℕ) : ℕ := 
  S + M - (T - N)

theorem students_interested_in_both (T S M N: ℕ) (hT : T = 55) (hS : S = 43) (hM : M = 34) (hN : N = 4) : 
  numberOfStudentsInterestedInBoth T S M N = 26 := 
by 
  rw [hT, hS, hM, hN]
  sorry

end students_interested_in_both_l189_189527


namespace powers_ratio_and_work_condition_l189_189360

noncomputable def pistons_power_ratio (initial_pressure : ℝ) (initial_volume : ℝ) (distance : ℝ) (speed : ℝ) (time : ℝ) : ℝ :=
  let final_volume := initial_volume * ((distance - (speed * time)) / distance)
  let alpha := initial_volume / final_volume
  let final_pressure_nitrogen := alpha * initial_pressure
  let power_ratio := final_pressure_nitrogen / 1
  power_ratio

noncomputable def work_done (pressure: ℝ) (volume_change: ℝ) : ℝ :=
  pressure * volume_change

theorem powers_ratio_and_work_condition (initial_pressure : ℝ) (initial_volume : ℝ) (distance : ℝ) (speed : ℝ) (time : ℝ) (work_threshold : ℝ) (interval_time : ℝ) :
  (pistons_power_ratio initial_pressure initial_volume distance speed time = 2) ∧ 
  (work_done 1 (speed * (interval_time / 60) * initial_volume / distance) > work_threshold) :=
begin
  sorry
end

end powers_ratio_and_work_condition_l189_189360


namespace largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189091

theorem largest_positive_integer_not_sum_of_multiple_of_36_and_composite :
  ∃ (n : ℕ), n = 83 ∧ 
    (∀ (a : ℕ) (b : ℕ), a > 0 ∧ b > 0 ∧ b.prime → n ≠ 36 * a + b) :=
begin
  sorry
end

end largest_positive_integer_not_sum_of_multiple_of_36_and_composite_l189_189091


namespace sum_of_consecutive_integers_with_product_812_l189_189939

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189939


namespace distance_to_focus_is_4_l189_189704

-- Define the parabola y^2 = 4x
def is_on_parabola (P : ℝ × ℝ) : Prop := P.2^2 = 4 * P.1

-- Define the distance from point P to the y-axis (x = 0)
def distance_to_y_axis (P : ℝ × ℝ) : ℝ := abs (P.1)

-- Define the focus of the parabola y^2 = 4x, which is (1, 0)
def focus_of_parabola : ℝ × ℝ := (1, 0)

-- Define the Euclidean distance between two points in ℝ^2
def euclidean_distance (P Q : ℝ × ℝ) : ℝ := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Given conditions
variable (P : ℝ × ℝ)
hypothesis (h1 : is_on_parabola P)
hypothesis (h2 : distance_to_y_axis P = 3)

-- Prove that the distance from point P to the focus of the parabola is 4
theorem distance_to_focus_is_4 : euclidean_distance P focus_of_parabola = 4 := by
  sorry

end distance_to_focus_is_4_l189_189704


namespace find_a_plus_b_l189_189759

theorem find_a_plus_b (a b : ℕ) (positive_a : 0 < a) (positive_b : 0 < b)
  (condition : ∀ (n : ℕ), (n > 0) → (∃ m n : ℕ, n = m * a + n * b) ∨ (∃ k l : ℕ, n = 2009 + k * a + l * b))
  (not_expressible : ∃ m n : ℕ, 1776 = m * a + n * b): a + b = 133 :=
sorry

end find_a_plus_b_l189_189759


namespace expected_value_Y_correct_l189_189307

noncomputable def expected_value_Y (X Y : ℝ) (h1 : X + Y = 8) (h2 : X = 10 * 0.6) : ℝ :=
E(Y) 

theorem expected_value_Y_correct (X Y : ℝ) (h1 : X + Y = 8) (h2 : X = 10 * 0.6) : expected_value_Y X Y h1 h2 = 2 :=
by 
  sorry

end expected_value_Y_correct_l189_189307


namespace nonnegative_expr_interval_l189_189589

noncomputable def expr (x : ℝ) : ℝ := (2 * x - 15 * x ^ 2 + 56 * x ^ 3) / (9 - x ^ 3)

theorem nonnegative_expr_interval (x : ℝ) :
  expr x ≥ 0 ↔ 0 ≤ x ∧ x < 3 := by
  sorry

end nonnegative_expr_interval_l189_189589


namespace area_of_figure_formed_by_intersecting_lines_and_x_axis_l189_189959

noncomputable theory -- Required for certain calculations involving real numbers

open Real -- Opening the real number namespace for convenience

-- Definition of lines y = x and x = -8
def y_eq_x (x : ℝ) : ℝ := x
def x_eq_neg8 : ℝ := -8

-- Definition of points
def intersection_point : ℝ × ℝ := (-8, -8)
def origin : ℝ × ℝ := (0, 0)
def point_on_x_axis : ℝ × ℝ := (-8, 0)

-- Definition of distances (base and height)
def base_length : ℝ := dist (point_on_x_axis, origin)
def height_length : ℝ := dist (intersection_point, origin)

-- Definition of the area of the triangle
def triangle_area : ℝ := 1 / 2 * base_length * height_length

-- Main proof problem statement
theorem area_of_figure_formed_by_intersecting_lines_and_x_axis :
  triangle_area = 32 := 
sorry

end area_of_figure_formed_by_intersecting_lines_and_x_axis_l189_189959


namespace original_pension_l189_189161

-- Definitions and conditions
variables (c d r s : ℝ) (h_diff : d ≠ c)
variables (k x : ℝ) (h1 : k * (x - c)**0.5 = k * x**0.5 - r)
variables (h2 : k * (x - d)**0.5 = k * x**0.5 - s)

-- Lean 4 statement
theorem original_pension (c d r s : ℝ) (h_diff : d ≠ c) (k x : ℝ)
  (h1 : k * (x - c)**0.5 = k * x**0.5 - r)
  (h2 : k * (x - d)**0.5 = k * x**0.5 - s) :
  k * x**0.5 = (r^2 - s^2) / (2 * (r - s)) :=
  sorry

end original_pension_l189_189161


namespace measure_of_C_angle_maximum_area_triangle_l189_189712

-- Proof Problem 1: Measure of angle C
theorem measure_of_C_angle (A B C : ℝ) (a b c : ℝ)
  (h1 : 0 < C ∧ C < Real.pi)
  (m n : ℝ × ℝ)
  (h2 : m = (Real.sin A, Real.sin B))
  (h3 : n = (Real.cos B, Real.cos A))
  (h4 : m.1 * n.1 + m.2 * n.2 = -Real.sin (2 * C)) :
  C = 2 * Real.pi / 3 :=
sorry

-- Proof Problem 2: Maximum area of triangle ABC
theorem maximum_area_triangle (A B C : ℝ) (a b c S : ℝ)
  (h1 : c = 2 * Real.sqrt 3)
  (h2 : Real.cos C = -1 / 2)
  (h3 : S = 1 / 2 * a * b * Real.sin (2 * Real.pi / 3)): 
  S ≤ Real.sqrt 3 :=
sorry

end measure_of_C_angle_maximum_area_triangle_l189_189712


namespace anne_shorter_trip_percentage_l189_189743

theorem anne_shorter_trip_percentage :
  let j := 3 + 4
  let a := Real.sqrt (3 * 3 + 4 * 4)
  (j - a) / j * 100 ≈ 29 := 
by
  sorry

end anne_shorter_trip_percentage_l189_189743


namespace tasks_to_volunteers_l189_189591

theorem tasks_to_volunteers :
  ∀ (volunteers tasks : ℕ), 
  volunteers = 4 → 
  tasks = 5 → 
  (∃ g : ℕ, g = 2) → -- There is a group of 2 tasks handled by one volunteer.
  (volunteers * tasks ≥ tasks) →
  (∃ n : ℕ, n = (Nat.choose 5 2) * (Nat.factorial 4)) →
  n = 240 :=
by 
  intros volunteers tasks hvol htask hgroup hc hways
  cases hways with n hn
  have : n = (Nat.choose 5 2) * (Nat.factorial 4) := hn
  rw [Nat.choose, Nat.factorial] at this
  have : n = 10 * 24 := this
  have : n = 240 := by norm_num
  exact this

end tasks_to_volunteers_l189_189591


namespace simplify_fraction_l189_189809

theorem simplify_fraction (x y : ℝ) : (x - y) / (y - x) = -1 :=
sorry

end simplify_fraction_l189_189809


namespace polynomial_not_prime_l189_189741

open nat

def polynomial (n : ℕ) : ℕ := n^2 + n + 41

theorem polynomial_not_prime : ∃ n : ℕ, ¬ prime (polynomial n) :=
by {
  use 40,
  have h : polynomial 40 = 1681,
  calc
  polynomial 40 = 40^2 + 40 + 41 : rfl
  ... = 1600 + 40 + 41 : by norm_num
  ... = 1681 : by norm_num,
  rw h,
  have not_prime_1681 : ¬ prime 1681,
  { rw prime_def_lt,
    split,
    { norm_num },
    { use 41,
      norm_num } },
  exact not_prime_1681
}

end polynomial_not_prime_l189_189741


namespace arithmetic_sequence_sum_eight_l189_189293

noncomputable theory

-- Definitions and conditions
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- Main theorem
theorem arithmetic_sequence_sum_eight
  (a : ℕ → ℝ)
  (h_seq : arithmetic_sequence a)
  (h_sum : S a 8 = 32) :
  a 2 + 2 * a 5 + a 6 = 16 :=
by sorry

end arithmetic_sequence_sum_eight_l189_189293


namespace total_worth_of_presents_l189_189748

-- Definitions of the costs
def costOfRing : ℕ := 4000
def costOfCar : ℕ := 2000
def costOfBracelet : ℕ := 2 * costOfRing

-- Theorem statement
theorem total_worth_of_presents : 
  costOfRing + costOfCar + costOfBracelet = 14000 :=
begin
  -- by using the given definitions and the provided conditions, we assert the statement
  sorry
end

end total_worth_of_presents_l189_189748


namespace sum_of_reciprocals_no_dependence_abs_diff_of_reciprocals_no_dependence_l189_189997

-- Define the geometric conditions for the problems
variable {M A B : Point} -- Points in the Euclidean plane
variable {circle : Circle} -- An existing circle
variable (inside_circle : M ∈ circle.interior)
variable (outside_circle : M ∈ circle.exterior)
variable (tangent_dist_A : ℝ) -- Distance from M to tangent at A
variable (tangent_dist_B : ℝ) -- Distance from M to tangent at B

-- First problem: Prove the sum of reciprocals of distances
theorem sum_of_reciprocals_no_dependence
  (inside_circle : M ∈ circle.interior)
  (chord : Line AB)
  (tangent_dist_A : distance M (tangent_line A circle) = m)
  (tangent_dist_B : distance M (tangent_line B circle) = n) :
  ∃ k : ℝ, 1 / tangent_dist_A + 1 / tangent_dist_B = k :=
sorry

-- Second problem: Prove the absolute difference of reciprocals of distances
theorem abs_diff_of_reciprocals_no_dependence
  (outside_circle : M ∈ circle.exterior)
  (chord : Line AB)
  (tangent_dist_A : distance M (tangent_line A circle) = m)
  (tangent_dist_B : distance M (tangent_line B circle) = n) :
  ∃ k : ℝ, |1 / tangent_dist_A - 1 / tangent_dist_B| = k :=
sorry

end sum_of_reciprocals_no_dependence_abs_diff_of_reciprocals_no_dependence_l189_189997


namespace gcd_660_924_l189_189973

theorem gcd_660_924 : Nat.gcd 660 924 = 132 := by
  sorry

end gcd_660_924_l189_189973


namespace norm_two_u_l189_189617

variable {E : Type*} [NormedAddCommGroup E]

theorem norm_two_u (u : E) (h : ∥u∥ = 5) : ∥2 • u∥ = 10 := sorry

end norm_two_u_l189_189617


namespace polynomial_identity_l189_189801

def P (x : ℝ) := (x - 1/2)^2001 + 1/2

theorem polynomial_identity : ∀ x : ℝ, P x + P (1 - x) = 1 :=
by
  sorry

end polynomial_identity_l189_189801


namespace sum_of_consecutive_integers_with_product_812_l189_189938

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189938


namespace complex_sum_l189_189385

noncomputable def omega : ℂ := sorry
axiom h1 : omega^11 = 1
axiom h2 : omega ≠ 1

theorem complex_sum 
: omega^10 + omega^14 + omega^18 + omega^22 + omega^26 + omega^30 + omega^34 + omega^38 + omega^42 + omega^46 + omega^50 + omega^54 + omega^58 
= -omega^10 :=
sorry

end complex_sum_l189_189385


namespace rationalize_denominator_l189_189023

theorem rationalize_denominator : (√12 + √2) / (√3 + √2) = 4 - √6 := 
by 
  sorry

end rationalize_denominator_l189_189023


namespace john_sold_books_on_monday_l189_189372

theorem john_sold_books_on_monday :
  ∀ (initial_stock tuesday_sales wednesday_sales thursday_sales friday_sales : ℕ)
    (unsold_percentage : ℝ),
  initial_stock = 700 →
  tuesday_sales = 82 →
  wednesday_sales = 60 →
  thursday_sales = 48 →
  friday_sales = 40 →
  unsold_percentage = 0.60 →
  let unsold_books := unsold_percentage * initial_stock,
      total_sold := initial_stock - unsold_books,
      other_days_sales := tuesday_sales + wednesday_sales + thursday_sales + friday_sales,
      monday_sales := total_sold - other_days_sales in
  monday_sales = 50 :=
by
  intros initial_stock tuesday_sales wednesday_sales thursday_sales friday_sales unsold_percentage 
  h_initial_stock h_tuesday_sales h_wednesday_sales h_thursday_sales h_friday_sales h_unsold_percentage;
  let unsold_books := unsold_percentage * initial_stock;
  let total_sold := initial_stock - unsold_books;
  let other_days_sales := tuesday_sales + wednesday_sales + thursday_sales + friday_sales;
  let monday_sales := total_sold - other_days_sales;
  sorry

end john_sold_books_on_monday_l189_189372


namespace number_of_distinct_classes_l189_189547

def bead : Type := Prop -- A bead can be red or blue

def ring (positions : Fin 4 → bead) := True -- Ring with four positions

def symmetry_group : Finset (Equiv (Fin 4)) :=
  {⟨λ i, (i + 1) % 4,  λ i, (i + 3) % 4, by decide⟩, -- rotation by 90 degrees
   ⟨λ i, (i + 2) % 4,  λ i, (i + 2) % 4, by decide⟩, -- rotation by 180 degrees
   ⟨λ i, (i + 3) % 4,  λ i, (i + 1) % 4, by decide⟩, -- rotation by 270 degrees
   ⟨λ i, (3 - i) % 4, λ i, (3 - i) % 4, by decide⟩ -- reflection (0 <-> 3, 1 <-> 2)
  }

def count_distinct_arrangements : Nat :=
  let orbits := Fix.StandardPartition.symm_action On symmetry_group;
  orbits.card

theorem number_of_distinct_classes : count_distinct_arrangements = 6 := by
  sorry

end number_of_distinct_classes_l189_189547


namespace power_zero_equals_one_specific_case_l189_189493

theorem power_zero_equals_one 
    (a b : ℤ) 
    (h : a ≠ 0)
    (h2 : b ≠ 0) : 
    (a / b : ℚ) ^ 0 = 1 := 
by {
  sorry
}

-- Specific case
theorem specific_case : 
  ( ( (-123456789 : ℤ) / (9876543210 : ℤ) : ℚ ) ^ 0 = 1 ) := 
by {
  apply power_zero_equals_one;
  norm_num;
  sorry
}

end power_zero_equals_one_specific_case_l189_189493


namespace no_possible_values_for_b_l189_189037

theorem no_possible_values_for_b : ¬ ∃ b : ℕ, 2 ≤ b ∧ b^3 ≤ 256 ∧ 256 < b^4 := by
  sorry

end no_possible_values_for_b_l189_189037


namespace solve_for_b_l189_189062

theorem solve_for_b (b x : ℝ) (h : 9^(x + 8) = 10^x) : b = (10 / 9) :=
by 
  sorry

end solve_for_b_l189_189062


namespace melanie_total_payment_l189_189001

noncomputable def totalCost (rentalCostPerDay : ℝ) (insuranceCostPerDay : ℝ) (mileageCostPerMile : ℝ) (days : ℕ) (miles : ℕ) : ℝ :=
  (rentalCostPerDay * days) + (insuranceCostPerDay * days) + (mileageCostPerMile * miles)

theorem melanie_total_payment :
  totalCost 30 5 0.25 3 350 = 192.5 :=
by
  sorry

end melanie_total_payment_l189_189001


namespace remainder_eval_at_4_l189_189170

def p : ℚ → ℚ := sorry

def r (x : ℚ) : ℚ := sorry

theorem remainder_eval_at_4 :
  (p 1 = 2) →
  (p 3 = 5) →
  (p (-2) = -2) →
  (∀ x, ∃ q : ℚ → ℚ, p x = (x - 1) * (x - 3) * (x + 2) * q x + r x) →
  r 4 = 38 / 7 :=
sorry

end remainder_eval_at_4_l189_189170


namespace books_sold_online_l189_189165

theorem books_sold_online (X : ℤ) 
  (h1: 743 = 502 + (37 + X) + (74 + X + 34) - 160) : 
  X = 128 := 
by sorry

end books_sold_online_l189_189165


namespace neg_proposition_l189_189053

theorem neg_proposition :
  ¬ (∀ x : ℝ, sin x ≤ 1) ↔ ∃ x : ℝ, sin x > 1 := 
by 
  sorry

end neg_proposition_l189_189053


namespace second_quadrant_necessary_not_sufficient_l189_189013

open Classical

-- Definitions
def isSecondQuadrant (α : ℝ) : Prop := 90 < α ∧ α < 180
def isObtuseAngle (α : ℝ) : Prop := 90 < α ∧ α < 180 ∨ 180 < α ∧ α < 270

-- The theorem statement
theorem second_quadrant_necessary_not_sufficient (α : ℝ) :
  (isSecondQuadrant α → isObtuseAngle α) ∧ ¬(isSecondQuadrant α ↔ isObtuseAngle α) :=
by
  sorry

end second_quadrant_necessary_not_sufficient_l189_189013


namespace complex_imaginary_axis_l189_189819

theorem complex_imaginary_axis (a : ℝ): 
  (∃ b : ℝ, ((-2 + a * complex.I) / (1 + complex.I)) = complex.I * b) → a = 2 := by
  sorry

end complex_imaginary_axis_l189_189819


namespace problem_l189_189846

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189846


namespace find_OD_l189_189679

structure Vector2D :=
  (x : ℝ)
  (y : ℝ)

def perpendicular (v1 v2 : Vector2D) : Prop :=
  v1.x * v2.x + v1.y * v2.y = 0

def parallel (v1 v2 : Vector2D) : Prop :=
  ∃ k : ℝ, v1.x = k * v2.x ∧ v1.y = k * v2.y

variables (OA OB OC OD : Vector2D)
variables (x y : ℝ)

-- Given conditions
def condition1 : OA = ⟨3, 1⟩ := rfl
def condition2 : OB = ⟨-1, 2⟩ := rfl
def condition3 : perpendicular OC OB := by sorry
def condition4 : parallel ⟨OC.x - OB.x, OC.y - OB.y⟩ OA := by sorry
def condition5 : OD.x + 3 = OC.x ∧ OD.y + 1 = OC.y := by sorry

-- Prove that OD = {11, 6}
theorem find_OD : OD = ⟨11, 6⟩ :=
  by sorry

end find_OD_l189_189679


namespace prob_exactly_two_co_presidents_l189_189473

noncomputable def prob_two_prez_receive_books : ℚ :=
let p_club := 1/4 in
let prob_in_club (n : ℕ) : ℚ := 
  (Mathlib.Combinatorics.choose (n-2) 2 : ℚ) / (Mathlib.Combinatorics.choose n 4 : ℚ) in
p_club * (prob_in_club 6 + prob_in_club 7 + prob_in_club 8 + prob_in_club 9)

theorem prob_exactly_two_co_presidents : prob_two_prez_receive_books = 0.2667 := 
sorry

end prob_exactly_two_co_presidents_l189_189473


namespace number_of_n_complete_sets_l189_189380

def n_complete (n : ℕ) (A : Finset ℕ) : Prop :=
  A.card = n ∧ ∀ i ∈ A, i < 2^n ∧ (Finset.powerset A).image (λ s, (s.sum id) % 2^n) = (Finset.range (2^n)).toFinset

theorem number_of_n_complete_sets (n : ℕ) (h : 0 < n) :
  ∃ k, k = 2 ^ (n * (n - 1) / 2) ∧
       ∀ (A : Finset ℕ), n_complete n A → A.card = k :=
by
  sorry

end number_of_n_complete_sets_l189_189380


namespace integer_values_expression_l189_189491

theorem integer_values_expression (a b : ℕ) :
  ∃ k : ℤ, (√5 - √2) * (√a + √b) = 3 * k :=
sorry

end integer_values_expression_l189_189491


namespace initial_winnings_l189_189745

theorem initial_winnings (X : ℝ) 
  (h1 : X - 0.25 * X = 0.75 * X)
  (h2 : 0.75 * X - 0.10 * (0.75 * X) = 0.675 * X)
  (h3 : 0.675 * X - 0.15 * (0.675 * X) = 0.57375 * X)
  (h4 : 0.57375 * X = 240) :
  X = 418 := by
  sorry

end initial_winnings_l189_189745


namespace lara_bouncy_house_time_l189_189757

theorem lara_bouncy_house_time :
  let run1_time := (3 * 60 + 45) + (2 * 60 + 10) + (1 * 60 + 28)
  let door_time := 73
  let run2_time := (2 * 60 + 55) + (1 * 60 + 48) + (1 * 60 + 15)
  run1_time + door_time + run2_time = 874 := by
    let run1_time := 225 + 130 + 88
    let door_time := 73
    let run2_time := 175 + 108 + 75
    sorry

end lara_bouncy_house_time_l189_189757


namespace two_co_presidents_probability_l189_189471

noncomputable def probability_two_co_presidents_receive_books : ℝ :=
  let prob_each_club := 1 / 4 in
  let prob_club_1 := (binomial 2 2 * binomial (6-2) 2) / binomial 6 4 in
  let prob_club_2 := (binomial 2 2 * binomial (7-2) 2) / binomial 7 4 in
  let prob_club_3 := (binomial 2 2 * binomial (8-2) 2) / binomial 8 4 in
  let prob_club_4 := (binomial 2 2 * binomial (9-2) 2) / binomial 9 4 in
  prob_each_club * (prob_club_1 + prob_club_2 + prob_club_3 + prob_club_4)

theorem two_co_presidents_probability : probability_two_co_presidents_receive_books ≈ 0.267 :=
by
  sorry

end two_co_presidents_probability_l189_189471


namespace smallest_period_f_max_min_f_on_interval_l189_189299

noncomputable def f (x : ℝ) := 
    cos x * sin (x + (Real.pi / 3)) - sqrt 3 * (cos x)^2 + (sqrt 3) / 4

theorem smallest_period_f :
  ∃ T : ℝ, T > 0 ∧ (∀ x : ℝ, f (x + T) = f x) ∧ T = Real.pi :=
sorry

theorem max_min_f_on_interval :
  ∃ (max_val min_val : ℝ),
    (∀ x : ℝ, -Real.pi / 4 <= x ∧ x <= Real.pi / 4 → f x ≤ max_val) ∧
    (∀ x : ℝ, -Real.pi / 4 <= x ∧ x <= Real.pi / 4 → f x ≥ min_val) ∧
    max_val = 1 / 4 ∧ min_val = -1 / 2 :=
sorry

end smallest_period_f_max_min_f_on_interval_l189_189299


namespace consecutive_integers_sum_l189_189897

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189897


namespace staff_discounted_price_l189_189984

theorem staff_discounted_price (d : ℝ) (h_nonneg : 0 ≤ d) :
  let discounted_price := d * 0.35 in
  let staff_discounted_price := discounted_price * 0.40 in
  staff_discounted_price = d * 0.14 :=
by
  sorry

end staff_discounted_price_l189_189984


namespace inequality_for_natural_n_l189_189017

theorem inequality_for_natural_n (n : ℕ) : (2 * n + 1)^n ≥ (2 * n)^n + (2 * n - 1)^n :=
by sorry

end inequality_for_natural_n_l189_189017


namespace parts_supplier_total_amount_received_l189_189986

noncomputable def total_amount_received (total_packages: ℕ) (price_per_package: ℚ) (discount_factor: ℚ)
  (X_percentage: ℚ) (Y_percentage: ℚ) : ℚ :=
  let X_packages := X_percentage * total_packages
  let Y_packages := Y_percentage * total_packages
  let Z_packages := total_packages - X_packages - Y_packages
  let discounted_price := discount_factor * price_per_package
  let cost_X := X_packages * price_per_package
  let cost_Y := Y_packages * price_per_package
  let cost_Z := 10 * price_per_package + (Z_packages - 10) * discounted_price
  cost_X + cost_Y + cost_Z

-- Given conditions
def total_packages : ℕ := 60
def price_per_package : ℚ := 20
def discount_factor : ℚ := 4 / 5
def X_percentage : ℚ := 0.20
def Y_percentage : ℚ := 0.15

theorem parts_supplier_total_amount_received :
  total_amount_received total_packages price_per_package discount_factor X_percentage Y_percentage = 1084 := 
by 
  -- Here we need the proof, but we put sorry to skip it as per instructions
  sorry

end parts_supplier_total_amount_received_l189_189986


namespace square_area_multiplier_l189_189816

theorem square_area_multiplier 
  (perimeter_square : ℝ) (length_rectangle : ℝ) (width_rectangle : ℝ)
  (perimeter_square_eq : perimeter_square = 800) 
  (length_rectangle_eq : length_rectangle = 125) 
  (width_rectangle_eq : width_rectangle = 64)
  : (perimeter_square / 4) ^ 2 / (length_rectangle * width_rectangle) = 5 := 
by
  sorry

end square_area_multiplier_l189_189816


namespace computation_of_difference_of_squares_l189_189577

theorem computation_of_difference_of_squares : (65^2 - 35^2) = 3000 := sorry

end computation_of_difference_of_squares_l189_189577


namespace largest_non_sum_of_36_and_composite_l189_189096

theorem largest_non_sum_of_36_and_composite :
  ∃ (n : ℕ), (∀ (a b : ℕ), n = 36 * a + b → b < 36 → b = 0 ∨ b = 1 ∨ b = 2 ∨ b = 3 ∨ b = 4 ∨ b = 5 ∨ b = 6 ∨ b = 8 ∨ b = 9 ∨ b = 10 ∨ b = 11 ∨ b = 12 ∨ b = 13 ∨ b = 14 ∨ b = 15 ∨ b = 16 ∨ b = 17 ∨ b = 18 ∨ b = 19 ∨ b = 20 ∨ b = 21 ∨ b = 22 ∨ b = 23 ∨ b = 24 ∨ b = 25 ∨ b = 26 ∨ b = 27 ∨ b = 28 ∨ b = 29 ∨ b = 30 ∨ b = 31 ∨ b = 32 ∨ b = 33 ∨ b = 34 ∨ b = 35) ∧ n = 188 :=
by
  use 188,
  intros a b h1 h2,
  -- rest of the proof that checks the conditions
  sorry

end largest_non_sum_of_36_and_composite_l189_189096


namespace find_P_l189_189355

def Point := ℝ × ℝ × ℝ

def distance (P Q : Point) : ℝ :=
  let (x1, y1, z1) := P
  let (x2, y2, z2) := Q
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2)

def A : Point := (1, -2, 1)
def B : Point := (2, 2, 2)
def P (z : ℝ) : Point := (0, 0, z)

theorem find_P : ∃ z : ℝ, distance (P z) A = distance (P z) B ∧ P z = (0, 0, 3) :=
  by
    sorry

end find_P_l189_189355


namespace degree_of_poly_l189_189497

-- Define the given polynomial
def poly : Polynomial ℤ := 3 + 7 * Polynomial.X ^ 5 - 4 * Polynomial.X ^ 2 + (1 / 3) * Polynomial.X ^ 5 + 11

-- Define the degree of the polynomial
def degree_poly (p : Polynomial ℤ) : ℤ := Polynomial.degree p

-- The theorem to prove the degree of the given polynomial is 5
theorem degree_of_poly : degree_poly poly = 5 := by
  sorry

end degree_of_poly_l189_189497


namespace two_point_distribution_p_value_l189_189703

noncomputable def X : Type := ℕ -- discrete random variable (two-point)
def p (E_X2 : ℝ): ℝ := E_X2 -- p == E(X)

theorem two_point_distribution_p_value (var_X : ℝ) (E_X : ℝ) (E_X2 : ℝ) 
    (h1 : var_X = 2 / 9) 
    (h2 : E_X = p E_X2) 
    (h3 : E_X2 = E_X): 
    E_X = 1 / 3 ∨ E_X = 2 / 3 :=
by
  sorry

end two_point_distribution_p_value_l189_189703


namespace radius_of_sphere_correct_l189_189830

noncomputable def radius_of_sphere (height : ℝ) (a b c : ℝ) : ℝ :=
  sqrt 6

theorem radius_of_sphere_correct (height : ℝ) (a b c : ℝ) (h_height : height = 5) 
                                   (h_sides : a = 7 ∧ b = 8 ∧ c = 9) :
  radius_of_sphere height a b c = sqrt 6 :=
by
  sorry

end radius_of_sphere_correct_l189_189830


namespace stratified_sampling_correct_l189_189831

-- Definitions based on the conditions
def total_employees : ℕ := 300
def over_40 : ℕ := 50
def between_30_and_40 : ℕ := 150
def under_30 : ℕ := 100
def sample_size : ℕ := 30
def stratified_ratio : ℕ := 1 / 10  -- sample_size / total_employees

-- Function to compute the number of individuals sampled from each age group
def sampled_from_age_group (group_size : ℕ) : ℕ :=
  group_size * stratified_ratio

-- Mathematical properties to be proved
theorem stratified_sampling_correct :
  sampled_from_age_group over_40 = 5 ∧ 
  sampled_from_age_group between_30_and_40 = 15 ∧ 
  sampled_from_age_group under_30 = 10 := by
  sorry

end stratified_sampling_correct_l189_189831


namespace find_ellipse_equation_find_fixed_point_through_line_P_l189_189657

-- Definitions and statements for the given conditions

noncomputable def ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) : Prop :=
  ∀ x y : ℝ, ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ (a^2 = 8 ∧ b^2 = 4)

theorem find_ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let focus := (2 : ℝ, 0),
      midpoint := (1 : ℝ, sqrt 2 / 2)
  in ellipse_equation a b h1 h2 :=
sorry

noncomputable def line_through_point_P (b : ℝ) : Prop :=
  ∀ k m x1 x2 y1 y2 : ℝ, 
  (x1 + x2 = -(4 * k * m / (1 + 2 *k^2)) ∧
   x1 * x2 = (2 * m^2 - 8) / (1 + 2 * k^2) ∧
   (2 * k - 1) * (m + 2) = 2 * k * m) →
  (sum_of_slopes_of_lines (0, b) (x1, y1) (x2, y2) = 1 →
  ∃ fixed_point, fixed_point = (-4, -2))

theorem find_fixed_point_through_line_P (x1 x2 y1 y2 : ℝ) (b : ℝ) (h1 : b > 0):
  (x1 + x2 ≠ 0) → 
  line_through_point_P b :=
sorry

end find_ellipse_equation_find_fixed_point_through_line_P_l189_189657


namespace ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l189_189731

theorem ab_parallel_to_x_axis_and_ac_parallel_to_y_axis
  (a b : ℝ)
  (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (a, -1))
  (hB : B = (2, 3 - b))
  (hC : C = (-5, 4))
  (hAB_parallel_x : A.2 = B.2)
  (hAC_parallel_y : A.1 = C.1) : a + b = -1 := by
  sorry


end ab_parallel_to_x_axis_and_ac_parallel_to_y_axis_l189_189731


namespace parabola_and_circle_tangency_l189_189306

noncomputable def parabola_directrix_tangent_to_circle (p : ℝ) (h : p > 0) : Prop :=
  let circle_eq := ∀ x y : ℝ, x^2 + y^2 + 6*x + 8 = 0
  let directrix := ∀ x y : ℝ, -x = p / 2
  let center_circle := ∀ x y : ℝ, (x+3)^2 + y^2 = 1
  ∀ (C : ℝ × ℝ), C = (-3, 0) → 
  let distance_directrix_to_center := (abs (3 - p / 2)) = 1
  ∀ (D : ℝ), distance_directrix_to_center → (p = 4) ∨ (p = 8)

theorem parabola_and_circle_tangency (p : ℝ) (h : parabola_directrix_tangent_to_circle p h) : 
  p = 4 ∨ p = 8 := 
sorry

end parabola_and_circle_tangency_l189_189306


namespace total_pennies_l189_189212

variable (C J : ℕ)

def cassandra_pennies : ℕ := 5000
def james_pennies (C : ℕ) : ℕ := C - 276

theorem total_pennies (hC : C = cassandra_pennies) (hJ : J = james_pennies C) :
  C + J = 9724 :=
by
  sorry

end total_pennies_l189_189212


namespace cos_angle_PMN_l189_189382

-- Definitions based on the conditions
variables (P A B C D M N : Type) -- P is the apex, ABCD is the square base, M and N are midpoints
variables (s h : ℝ) -- s is the side length of the square base, h is the height of the pyramid
variables [metric_space P] [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space M] [metric_space N]

-- The hypothesis about the geometry of the square pyramid
hypothesis pyramid_sq_base : dist A B = s ∧ dist B C = s ∧ dist C D = s ∧ dist D A = s
hypothesis midpoint_AB_M : dist A M = s / 2 ∧ dist B M = s / 2
hypothesis midpoint_CD_N : dist C N = s / 2 ∧ dist D N = s / 2
hypothesis height_pyramid : dist P (midpoint A B C D) = h -- where "midpoint A B C D" is the center of the square base

-- The proof problem statement
theorem cos_angle_PMN : cos_angle P M N = (h^2 - s^2 / 4) / (s^2 / 4 + h^2) := 
by 
  sorry

end cos_angle_PMN_l189_189382


namespace johnny_marbles_l189_189373

noncomputable def choose_at_least_one_red : ℕ :=
  let total_marbles := 8
  let red_marbles := 1
  let other_marbles := 7
  let choose_4_out_of_8 := Nat.choose total_marbles 4
  let choose_3_out_of_7 := Nat.choose other_marbles 3
  let choose_4_with_at_least_1_red := choose_3_out_of_7
  choose_4_with_at_least_1_red

theorem johnny_marbles : choose_at_least_one_red = 35 :=
by
  -- Sorry, proof is omitted
  sorry

end johnny_marbles_l189_189373


namespace max_value_problem_l189_189813

theorem max_value_problem (x y : ℝ)
  (h₁ : x > 0 ∧ y > 0)
  (h₂ : x^2 + y^2 / 2 = 1) :
  ∃ a, a = x * sqrt (1 + y^2) ∧ a ≤ 3 * sqrt 2 / 4 :=
sorry

end max_value_problem_l189_189813


namespace profit_benny_wants_to_make_l189_189196

noncomputable def pumpkin_pies : ℕ := 10
noncomputable def cherry_pies : ℕ := 12
noncomputable def cost_pumpkin_pie : ℝ := 3
noncomputable def cost_cherry_pie : ℝ := 5
noncomputable def price_per_pie : ℝ := 5

theorem profit_benny_wants_to_make : 5 * (pumpkin_pies + cherry_pies) - (pumpkin_pies * cost_pumpkin_pie + cherry_pies * cost_cherry_pie) = 20 :=
by
  sorry

end profit_benny_wants_to_make_l189_189196


namespace original_price_sarees_l189_189951

theorem original_price_sarees (P : ℝ) (h : 0.80 * P * 0.85 = 231.2) : P = 340 := 
by sorry

end original_price_sarees_l189_189951


namespace other_student_in_sample_18_l189_189526

theorem other_student_in_sample_18 (class_size sample_size : ℕ) (all_students : Finset ℕ) (sample_students : List ℕ)
  (h_class_size : class_size = 60)
  (h_sample_size : sample_size = 4)
  (h_all_students : all_students = Finset.range 60) -- students are numbered from 1 to 60
  (h_sample : sample_students = [3, 33, 48])
  (systematic_sampling : ℕ → ℕ → List ℕ) -- systematic_sampling function that generates the sample based on first element and k
  (k : ℕ) (h_k : k = class_size / sample_size) :
  systematic_sampling 3 k = [3, 18, 33, 48] := 
  sorry

end other_student_in_sample_18_l189_189526


namespace asymptotes_of_hyperbola_l189_189825

theorem asymptotes_of_hyperbola : 
  ∀ (x y : ℝ), 9 * y^2 - 25 * x^2 = 169 → (y = (5/3) * x ∨ y = -(5/3) * x) :=
by 
  sorry

end asymptotes_of_hyperbola_l189_189825


namespace norm_2u_equals_10_l189_189624

-- Define u as a vector in ℝ² and the function for its norm.
variable (u : ℝ × ℝ)

-- Define the condition that the norm of u is 5.
def norm_eq_5 : Prop := Real.sqrt (u.1^2 + u.2^2) = 5

-- Statement of the proof problem
theorem norm_2u_equals_10 (h : norm_eq_5 u) : Real.sqrt ((2 * u.1)^2 + (2 * u.2)^2) = 10 :=
by
  sorry

end norm_2u_equals_10_l189_189624


namespace shaded_region_area_l189_189562

theorem shaded_region_area (x y : ℝ) (OA OB : ℝ) (P A O B : ℝ) 
  (h : P ∈ segment ℝ A B ∨ P ∈ line ℝ A O B) 
  (h_eq : (P - O) = x * (A - O) + y * (B - O))
  (h_cond : y = 4 ∧ x ≤ 0) : 
  ∃ (area : ℝ), area = 9 / 2 := 
by 
  sorry

end shaded_region_area_l189_189562


namespace diff_of_squares_div_l189_189502

-- Definitions from the conditions
def a : ℕ := 125
def b : ℕ := 105

-- The main statement to be proved
theorem diff_of_squares_div {a b : ℕ} (h1 : a = 125) (h2 : b = 105) : (a^2 - b^2) / 20 = 230 := by
  sorry

end diff_of_squares_div_l189_189502


namespace distinct_solutions_subtraction_l189_189772

theorem distinct_solutions_subtraction (r s : ℝ) (h_eq : ∀ x ≠ 3, (6 * x - 18) / (x^2 + 4 * x - 21) = x + 3) 
  (h_r : (6 * r - 18) / (r^2 + 4 * r - 21) = r + 3) 
  (h_s : (6 * s - 18) / (s^2 + 4 * s - 21) = s + 3) 
  (h_distinct : r ≠ s) 
  (h_order : r > s) : 
  r - s = 10 := 
by 
  sorry

end distinct_solutions_subtraction_l189_189772


namespace total_cost_correct_l189_189478

-- Condition C1: There are 13 hearts in a deck of 52 playing cards. 
def hearts_in_deck : ℕ := 13

-- Condition C2: The number of cows is twice the number of hearts.
def cows_in_Devonshire : ℕ := 2 * hearts_in_deck

-- Condition C3: Each cow is sold at $200.
def cost_per_cow : ℕ := 200

-- Question Q1: Calculate the total cost of the cows.
def total_cost_of_cows : ℕ := cows_in_Devonshire * cost_per_cow

-- Final statement we need to prove
theorem total_cost_correct : total_cost_of_cows = 5200 := by
  -- This will be proven in the proof body
  sorry

end total_cost_correct_l189_189478


namespace on_time_departure_rate_l189_189046

-- Definitions and conditions
def first_flight_late : Prop := true
def next_three_flights_on_time : Prop := true
def additional_on_time_flights : ℕ := 4

-- Calculation of the total number of flights
def total_flights : ℕ := 1 + 3 + additional_on_time_flights

-- On-time flights
def on_time_flights : ℕ := 3 + additional_on_time_flights

-- Target on-time departure rate
def target_percentage : ℚ := 7 / 8 

theorem on_time_departure_rate (P : ℚ) (hP : P < target_percentage * 100) : 
  (on_time_flights : ℚ) / total_flights > P / 100 := 
by {
  have h : (on_time_flights : ℚ) = 7 := rfl,
  have hn : (total_flights : ℚ) = 8 := rfl,
  calc
    (on_time_flights : ℚ) / total_flights
      = 7 / 8 : by rw [h, hn]
    ...     > P / 100 : sorry -- This part captures the essence of the problem
}

end on_time_departure_rate_l189_189046


namespace length_GP_in_triangle_DEF_with_medians_l189_189357

theorem length_GP_in_triangle_DEF_with_medians 
  (DE DF EF : ℝ) (H1 : DE = 10) (H2 : DF = 15) (H3 : EF = 17) 
  (median_DG_median_DM : ∀ (D G P M N O : Type), DM = (1/3) * DG)
  (centroid_of_medians_intersect : ∀ (D G M N O P : Type), G = centroid D M N O )
  (altitude_from_G_to_EF : ∀ (D G P E F : Type), altitude G P E F) : 
  ∃ (G P : Type), length_GP G P = (4*sqrt(154))/(17) := by
  sorry

end length_GP_in_triangle_DEF_with_medians_l189_189357


namespace mass_percentage_H_in_Ammonium_chloride_l189_189247

theorem mass_percentage_H_in_Ammonium_chloride :
  let molar_mass_N := 14.01
  let molar_mass_H := 1.01
  let molar_mass_Cl := 35.45
  let number_of_H := 4
  let molar_mass_NH4Cl := molar_mass_N + number_of_H * molar_mass_H + molar_mass_Cl
  let total_mass_H_NH4Cl := number_of_H * molar_mass_H
  let mass_percentage_H := (total_mass_H_NH4Cl / molar_mass_NH4Cl) * 100
  molar_mass_NH4Cl = 53.50 →
  mass_percentage_H ≈ 7.55 :=
by
  intros molar_mass_N molar_mass_H molar_mass_Cl number_of_H molar_mass_NH4Cl total_mass_H_NH4Cl mass_percentage_H h
  sorry

end mass_percentage_H_in_Ammonium_chloride_l189_189247


namespace cross_section_hexagon_l189_189354

-- Define the vertices and midpoint conditionally
structure HexagonalPrism where
  A B C D E F A' B' C' D' E' F' : Type
  M : Type
  is_midpoint_M : M = (D, E)

-- Define the proof problem
theorem cross_section_hexagon (hp : HexagonalPrism) : 
  shape_of_cross_section hp.A' hp.C hp.M = "hexagon" := 
sorry

end cross_section_hexagon_l189_189354


namespace same_graph_l189_189128

def f (x : ℝ) : ℝ := |x|

def g (x : ℝ) : ℝ :=
  if x ≥ 0 then x else -x

theorem same_graph : ∀ (x : ℝ), f x = g x :=
by {
  sorry -- Proof is not required as per instructions
}

end same_graph_l189_189128


namespace measure_of_angle_A_value_of_b2_plus_c2_l189_189275

-- Definitions of the conditions
def triangle_conditions (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = π ∧
  0 < A < π / 2 ∧
  sqrt 3 * b = 2 * a * sin B

def triangle_area (a b : ℝ) (A : ℝ) (area : ℝ) : Prop :=
  area = 1/2 * b * a * sin A

-- Theorem to prove angle A
theorem measure_of_angle_A (A B C a b c : ℝ) (h_conditions : triangle_conditions A B C a b c) :
  A = π / 3 :=
sorry

-- Theorem to prove b^2 + c^2
theorem value_of_b2_plus_c2 (A B C a b c : ℝ) (area : ℝ) (h_area : triangle_area a b (π / 3) area) (ha : a = 7) (h_area_eq : area = 10 * sqrt 3):
  b^2 + c^2 = 89 :=
sorry

end measure_of_angle_A_value_of_b2_plus_c2_l189_189275


namespace limit_tanxy_over_y_l189_189245

theorem limit_tanxy_over_y (f : ℝ×ℝ → ℝ) :
  (∀ ε > 0, ∃ δ > 0, ∀ x y, abs (x - 3) < δ ∧ abs y < δ → abs (f (x, y) - 3) < ε) :=
sorry

end limit_tanxy_over_y_l189_189245


namespace sum_of_first_2000_terms_is_2800_l189_189060

-- Define the sequence based on the given conditions
def seq : ℕ → ℕ 
| n :=
  let b := n + 1 in
  let m := (b * (b + 1)) / 2 in
  if h : n ≥ m then
    1
  else 
    3

-- Define the sum of the first 2000 terms of the sequence
def sum_seq_first_2000_terms : ℕ :=
  (List.range 2000).map seq |> List.sum

-- The main theorem
theorem sum_of_first_2000_terms_is_2800 : sum_seq_first_2000_terms = 2800 := 
by
  sorry -- Proof to be completed

end sum_of_first_2000_terms_is_2800_l189_189060


namespace simplify_expression_l189_189427

noncomputable def expr1 := (Real.sqrt 462) / (Real.sqrt 330)
noncomputable def expr2 := (Real.sqrt 245) / (Real.sqrt 175)
noncomputable def expr_simplified := (12 * Real.sqrt 35) / 25

theorem simplify_expression :
  expr1 + expr2 = expr_simplified :=
sorry

end simplify_expression_l189_189427


namespace sum_of_consecutive_integers_with_product_812_l189_189942

theorem sum_of_consecutive_integers_with_product_812 : ∃ (x : ℕ), (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) :=
by 
  sorry

end sum_of_consecutive_integers_with_product_812_l189_189942


namespace sum_of_consecutive_integers_l189_189882

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189882


namespace true_statements_proved_l189_189592

-- Conditions
def A : Prop := ∃ n : ℕ, 25 = 5 * n
def B : Prop := (∃ m1 : ℕ, 209 = 19 * m1) ∧ (¬ ∃ m2 : ℕ, 63 = 19 * m2)
def C : Prop := (¬ ∃ k1 : ℕ, 90 = 30 * k1) ∧ (¬ ∃ k2 : ℕ, 49 = 30 * k2)
def D : Prop := (∃ l1 : ℕ, 34 = 17 * l1) ∧ (¬ ∃ l2 : ℕ, 68 = 17 * l2)
def E : Prop := ∃ q : ℕ, 140 = 7 * q

-- Correct statements
def TrueStatements : Prop := A ∧ B ∧ E ∧ ¬C ∧ ¬D

-- Lean statement to prove
theorem true_statements_proved : TrueStatements := 
by
  sorry

end true_statements_proved_l189_189592


namespace boys_meeting_problem_l189_189967

theorem boys_meeting_problem (d : ℝ) (t : ℝ)
  (speed1 speed2 : ℝ)
  (h1 : speed1 = 6) 
  (h2 : speed2 = 8) 
  (h3 : t > 0)
  (h4 : ∀ n : ℤ, n * (speed1 + speed2) * t ≠ d) : 
  0 = 0 :=
by 
  sorry

end boys_meeting_problem_l189_189967


namespace donald_juice_l189_189593

variable (P D : ℕ)

theorem donald_juice (h1 : P = 3) (h2 : D = 2 * P + 3) : D = 9 := by
  sorry

end donald_juice_l189_189593


namespace least_positive_integer_is_4619_l189_189975

noncomputable def least_positive_integer (N : ℕ) : Prop :=
  N % 4 = 3 ∧
  N % 5 = 4 ∧
  N % 6 = 5 ∧
  N % 7 = 6 ∧
  N % 11 = 10 ∧
  ∀ M : ℕ, (M % 4 = 3 ∧ M % 5 = 4 ∧ M % 6 = 5 ∧ M % 7 = 6 ∧ M % 11 = 10) → N ≤ M

theorem least_positive_integer_is_4619 : least_positive_integer 4619 :=
  sorry

end least_positive_integer_is_4619_l189_189975


namespace worth_of_presents_l189_189751

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l189_189751


namespace number_of_linear_eqs_l189_189658

def is_linear_eq_in_one_var (eq : String) : Bool :=
  match eq with
  | "0.3x = 1" => true
  | "x/2 = 5x + 1" => true
  | "x = 6" => true
  | _ => false

theorem number_of_linear_eqs :
  let eqs := ["x - 2 = 2 / x", "0.3x = 1", "x/2 = 5x + 1", "x^2 - 4x = 3", "x = 6", "x + 2y = 0"]
  (eqs.filter is_linear_eq_in_one_var).length = 3 :=
by
  sorry

end number_of_linear_eqs_l189_189658


namespace loan_balance_formula_l189_189398

variable (c V : ℝ) (t n : ℝ)

theorem loan_balance_formula :
  V = c / (1 + t)^(3 * n) →
  n = (Real.log (c / V)) / (3 * Real.log (1 + t)) :=
by sorry

end loan_balance_formula_l189_189398


namespace no_intersect_M1_M2_l189_189758

theorem no_intersect_M1_M2 (A B : ℤ) : ∃ C : ℤ, 
  ∀ x y : ℤ, (x^2 + A * x + B) ≠ (2 * y^2 + 2 * y + C) := by
  sorry

end no_intersect_M1_M2_l189_189758


namespace two_co_presidents_probability_l189_189470

noncomputable def probability_two_co_presidents_receive_books : ℝ :=
  let prob_each_club := 1 / 4 in
  let prob_club_1 := (binomial 2 2 * binomial (6-2) 2) / binomial 6 4 in
  let prob_club_2 := (binomial 2 2 * binomial (7-2) 2) / binomial 7 4 in
  let prob_club_3 := (binomial 2 2 * binomial (8-2) 2) / binomial 8 4 in
  let prob_club_4 := (binomial 2 2 * binomial (9-2) 2) / binomial 9 4 in
  prob_each_club * (prob_club_1 + prob_club_2 + prob_club_3 + prob_club_4)

theorem two_co_presidents_probability : probability_two_co_presidents_receive_books ≈ 0.267 :=
by
  sorry

end two_co_presidents_probability_l189_189470


namespace consecutive_integer_sum_l189_189863

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189863


namespace no_integer_r_exists_l189_189234

theorem no_integer_r_exists (n : ℕ) : ¬ ∃ r : ℤ, ∀ n : ℕ, (n > 0 →  (n! / 2^(n - r)) ∈ int) :=
sorry

end no_integer_r_exists_l189_189234


namespace largest_not_sum_of_36_and_composite_l189_189111

theorem largest_not_sum_of_36_and_composite :
  ∃ (n : ℕ), n = 304 ∧ ∀ (a b : ℕ), 0 ≤ b ∧ b < 36 ∧ (b + 36 * a) ∈ range n →
  (∀ k < a, Prime (b + 36 * k) ∧ n = 36 * (n / 36) + n % 36) :=
begin
  use 304,
  split,
  { refl },
  { intros a b h0 h1 hsum,
    intros k hk,
    split,
    { sorry }, -- Proof for prime
    { unfold range at hsum,
      exact ⟨n / 36, n % 36⟩, },
  }
end

end largest_not_sum_of_36_and_composite_l189_189111


namespace find_unknown_rate_of_blankets_l189_189166

theorem find_unknown_rate_of_blankets (x : ℕ) 
  (h1 : 3 * 100 = 300) 
  (h2 : 5 * 150 = 750)
  (h3 : 3 + 5 + 2 = 10) 
  (h4 : 10 * 160 = 1600) 
  (h5 : 300 + 750 + 2 * x = 1600) : 
  x = 275 := 
sorry

end find_unknown_rate_of_blankets_l189_189166


namespace positive_difference_arithmetic_sequence_l189_189117

theorem positive_difference_arithmetic_sequence :
  let a := 3
  let d := 5
  let a₁₀₀ := a + (100 - 1) * d
  let a₁₁₀ := a + (110 - 1) * d
  a₁₁₀ - a₁₀₀ = 50 :=
by
  sorry

end positive_difference_arithmetic_sequence_l189_189117


namespace silver_coin_value_l189_189428

--- Definitions from the conditions
def total_value_hoard (value_silver : ℕ) := 100 * 3 * value_silver + 60 * value_silver + 33

--- Statement of the theorem to prove
theorem silver_coin_value (x : ℕ) (h : total_value_hoard x = 2913) : x = 8 :=
by {
  sorry
}

end silver_coin_value_l189_189428


namespace sum_and_product_l189_189454

theorem sum_and_product (c d : ℝ) (h1 : 2 * c = -8) (h2 : c^2 - d = 4) : c + d = 8 := by
  sorry

end sum_and_product_l189_189454


namespace tom_total_lifting_capacity_l189_189482

-- Definitions based on conditions
def initial_farmer_lift_kg : ℝ := 100
def increase150_pct : ℝ := initial_farmer_lift_kg * 1.5
def farmer_after_150_pct : ℝ := initial_farmer_lift_kg + increase150_pct
def increase25_pct : ℝ := farmer_after_150_pct * 0.25
def farmer_after_25_pct : ℝ := farmer_after_150_pct + increase25_pct
def increase10_pct : ℝ := farmer_after_25_pct * 0.1
def farmer_final_per_hand_kg : ℝ := farmer_after_25_pct + increase10_pct
def total_farmer_lift_kg : ℝ := farmer_final_per_hand_kg * 2

def initial_deadlift_lb : ℝ := 400
def increase20_pct : ℝ := initial_deadlift_lb * 0.2
def deadlift_after_20_pct : ℝ := initial_deadlift_lb + increase20_pct
def lb_to_kg : ℝ := 1 / 2.20462
def deadlift_final_kg : ℝ := deadlift_after_20_pct * lb_to_kg

-- Total lifting capacity in kg
def total_lifting_capacity_kg : ℝ := total_farmer_lift_kg + deadlift_final_kg

-- Theorem statement: Tom's total lifting capacity for the farmer handles and deadlift combined is 905.22 kg
theorem tom_total_lifting_capacity : total_lifting_capacity_kg = 905.22 :=
by
  -- This part is skipped
  sorry

end tom_total_lifting_capacity_l189_189482


namespace sum_of_consecutive_integers_with_product_812_l189_189910

theorem sum_of_consecutive_integers_with_product_812 :
  ∃ x : ℕ, (x * (x + 1) = 812) ∧ (x + (x + 1) = 57) := 
begin
  sorry
end

end sum_of_consecutive_integers_with_product_812_l189_189910


namespace josephine_filled_two_liter_containers_l189_189416

-- Definitions of the conditions
def total_milk := 10
def containers_075 := 2
def milk_per_container_075 := 0.75
def containers_05 := 5
def milk_per_container_05 := 0.5

-- Question restated as a Lean statement
theorem josephine_filled_two_liter_containers : 
  let smaller_containers_milk := containers_075 * milk_per_container_075 + containers_05 * milk_per_container_05
  let two_liter_containers_milk := total_milk - smaller_containers_milk
  let number_of_two_liter_containers := two_liter_containers_milk / 2
  number_of_two_liter_containers = 3 :=
by
  sorry

end josephine_filled_two_liter_containers_l189_189416


namespace find_tan_P_l189_189596

theorem find_tan_P (PQ QR : ℝ) (hPQ : PQ = 30) (hQR : QR = 54) (right_triangle : ∃ P Q R : Point, is_right_triangle P Q R ∧ PQ = 30 ∧ QR = 54 ∧ angle QPR = π / 2) :
  tan (angle P) = 2 * real.sqrt 14 / 5 :=
by
  sorry

end find_tan_P_l189_189596


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189102

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189102


namespace slope_of_tangent_at_1_l189_189765

variable {f : ℝ → ℝ}

theorem slope_of_tangent_at_1
  (hf : Differentiable ℝ f)
  (hlim : (filter.tendsto (λ x, (f 1 - f (1 - x)) / (2 * x)) (nhds 0) (nhds (-1)))) :
  deriv f 1 = -2 :=
by
  sorry

end slope_of_tangent_at_1_l189_189765


namespace monotonicity_and_extrema_l189_189194

-- Define the function f
def f (x : ℝ) : ℝ := sin x - cos x + x + 1

-- Define the gradient f'
noncomputable def f' (x : ℝ) : ℝ := cos x + sin x + 1

-- Prove the intervals of monotonicity and extrema
theorem monotonicity_and_extrema : 
  (∀ x, 0 < x ∧ x < π → f' x > 0) ∧
  (∀ x, π < x ∧ x < (3 * π) / 2 → f' x < 0) ∧
  (∀ x, (3 * π) / 2 < x ∧ x < 2 * π → f' x > 0) ∧
  (∃ x, x = π ∧ f x = π + 2) ∧
  (∃ x, x = (3 * π) / 2 ∧ f x = (3 * π) / 2) :=
sorry

end monotonicity_and_extrema_l189_189194


namespace total_time_to_virgo_l189_189079

def train_ride : ℝ := 5
def first_layover : ℝ := 1.5
def bus_ride : ℝ := 4
def second_layover : ℝ := 0.5
def first_flight : ℝ := 6
def third_layover : ℝ := 2
def second_flight : ℝ := 3 * bus_ride
def fourth_layover : ℝ := 3
def car_drive : ℝ := 3.5
def first_boat_ride : ℝ := 1.5
def fifth_layover : ℝ := 0.75
def second_boat_ride : ℝ := 2 * first_boat_ride - 0.5
def final_walk : ℝ := 1.25

def total_time : ℝ := train_ride + first_layover + bus_ride + second_layover + first_flight + third_layover + second_flight + fourth_layover + car_drive + first_boat_ride + fifth_layover + second_boat_ride + final_walk

theorem total_time_to_virgo : total_time = 44 := by
  simp [train_ride, first_layover, bus_ride, second_layover, first_flight, third_layover, second_flight, fourth_layover, car_drive, first_boat_ride, fifth_layover, second_boat_ride, final_walk, total_time]
  sorry

end total_time_to_virgo_l189_189079


namespace area_of_ellipse_l189_189600

-- Defining the given equation of the ellipse
def ellipse_eq (x y : ℝ) : Prop :=
  2 * x^2 + 8 * x + 3 * y^2 + 12 * y + 18 = 0

-- Statement of the theorem to be proved
theorem area_of_ellipse :
  (∀ x y, ellipse_eq x y) →
  ∃ (A : ℝ), A = π * sqrt (2 / 3) :=
by
  intros
  sorry

end area_of_ellipse_l189_189600


namespace sum_coordinates_D_is_13_l189_189795

theorem sum_coordinates_D_is_13 
  (A B C D : ℝ × ℝ) 
  (hA : A = (4, 8))
  (hB : B = (2, 2))
  (hC : C = (6, 4))
  (hD : D = (8, 5))
  (h_mid1 : (A.1 + B.1) / 2 = 3 ∧ (A.2 + B.2) / 2 = 5)
  (h_mid2 : (B.1 + C.1) / 2 = 4 ∧ (B.2 + C.2) / 2 = 3)
  (h_mid3 : (C.1 + D.1) / 2 = 7 ∧ (C.2 + D.2) / 2 = 4.5)
  (h_mid4 : (D.1 + A.1) / 2 = 6 ∧ (D.2 + A.2) / 2 = 6.5)
  (h_square : ((A.1 + B.1) / 2, (A.2 + B.2) / 2) = (3, 5) ∧
               ((B.1 + C.1) / 2, (B.2 + C.2) / 2) = (4, 3) ∧
               ((C.1 + D.1) / 2, (C.2 + D.2) / 2) = (7, 4.5) ∧
               ((D.1 + A.1) / 2, (D.2 + A.2) / 2) = (6, 6.5))
  : (8 + 5) = 13 :=
by
  sorry

end sum_coordinates_D_is_13_l189_189795


namespace product_of_common_divisors_l189_189607

open Nat

-- Define the sets of divisors
def divisors_210 := {n : ℤ | ∃ k : ℤ, 210 = k * n}
def divisors_35 := {n : ℤ | ∃ k : ℤ, 35 = k * n}
def common_divisors := {n : ℤ | n ∈ divisors_210 ∧ n ∈ divisors_35}

-- Define the set of common divisors explicitly
def common_divisors_explicit := {1, -1, 5, -5, 7, -7, 35, -35}

-- Prove that the product is 1500625
theorem product_of_common_divisors : 
  (∏ d in common_divisors_explicit, d) = 1500625 := by
  sorry

end product_of_common_divisors_l189_189607


namespace product_minus_200_l189_189263

-- Definitions and conditions
variables (P Q R S : ℤ)
variable (x : ℤ)
variable (h1 : P + 5 = x)
variable (h2 : Q - 5 = x)
variable (h3 : R * 2 = x)
variable (h4 : S / 2 = x)
variable (h_sum : P + Q + R + S = 104)

-- Theorem statement
theorem product_minus_200 : P * Q * R * S - 200 = 267442.5 :=
by sorry

end product_minus_200_l189_189263


namespace triangle_max_area_in_quarter_ellipse_l189_189490

theorem triangle_max_area_in_quarter_ellipse (a b c : ℝ) (h : c^2 = a^2 - b^2) :
  ∃ (T_max : ℝ), T_max = b / 2 :=
by sorry

end triangle_max_area_in_quarter_ellipse_l189_189490


namespace solution_of_triples_l189_189588

theorem solution_of_triples (m n p : ℤ) (prime_p : nat.prime p) :
  n ^ (2 * p) = m ^ 2 + n ^ 2 + ↑p + 1 → 
  ( (m = 3 ∧ n = 2 ∧ p = 2) ∨ (m = 3 ∧ n = -2 ∧ p = 2) ∨ 
    (m = -3 ∧ n = 2 ∧ p = 2) ∨ (m = -3 ∧ n = -2 ∧ p = 2)) := 
by 
  sorry

end solution_of_triples_l189_189588


namespace quadratic_equation_identify_l189_189124

theorem quadratic_equation_identify {a b c x : ℝ} :
  ((3 - 5 * x^2 = x) ↔ true) ∧
  ((3 / x + x^2 - 1 = 0) ↔ false) ∧
  ((a * x^2 + b * x + c = 0) ↔ (a ≠ 0)) ∧
  ((4 * x - 1 = 0) ↔ false) :=
by
  sorry

end quadratic_equation_identify_l189_189124


namespace smallest_x_multiple_of_53_l189_189232

theorem smallest_x_multiple_of_53 : ∃ (x : Nat), (x > 0) ∧ ( ∀ (n : Nat), (n > 0) ∧ ((3 * n + 43) % 53 = 0) → x ≤ n ) ∧ ((3 * x + 43) % 53 = 0) :=
sorry

end smallest_x_multiple_of_53_l189_189232


namespace sum_of_consecutive_integers_l189_189881

theorem sum_of_consecutive_integers (x : ℕ) (h : x * (x + 1) = 812) : x + (x + 1) = 57 :=
by sorry

end sum_of_consecutive_integers_l189_189881


namespace verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189099

noncomputable def largest_integer_not_sum_of_multiple_of_36_and_composite_integer : ℕ :=
  209

theorem verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer :
  ∀ m : ℕ, ∀ a b : ℕ, (m = 36 * a + b) → (0 ≤ b ∧ b < 36) →
  ((b % 3 = 0 → b = 3) ∧ 
   (b % 3 = 1 → ∀ k, is_prime (b + 36 * k) → k = 2 → b ≠ 4) ∧ 
   (b % 3 = 2 → ∀ k, is_prime (b + 36 * k) → b = 29)) → 
  m ≤ 209 :=
sorry

end verify_largest_integer_not_sum_of_multiple_of_36_and_composite_integer_l189_189099


namespace line_has_equal_intercepts_find_a_l189_189669

theorem line_has_equal_intercepts (a : ℝ) :
  (∃ l : ℝ, (l = 0 → ax + y - 2 - a = 0) ∧ (l = 1 → (a = 1 ∨ a = -2))) := sorry

-- formalizing the problem
theorem find_a (a : ℝ) (h_eq_intercepts : ∀ x y : ℝ, (a * x + y - 2 - a = 0 ↔ (x = 2 + a ∧ y = -2 - a))) :
  a = 1 ∨ a = -2 := sorry

end line_has_equal_intercepts_find_a_l189_189669


namespace proof_problem_l189_189648

variable (α β : Type) [Plane α] [Plane β]
variable (l1 l2 : Type) [Line l1] [Line l2]

-- Conditions
variable (perp_α : PerpendicularLinePlane l1 α)
variable (in_β : ContainedLinePlane l2 β)

-- Propositions to evaluate
def proposition_1 : Prop := (ParallelPlanes α β) → (PerpendicularLines l1 l2)
def proposition_3 : Prop := (ParallelLines l1 l2) → (PerpendicularPlanes α β)

-- The full statement to show the correct answer is D
theorem proof_problem : (proposition_1 perp_α in_β) ∧ (proposition_3 perp_α in_β) ∧ 
                         ¬ (proposition_2 perp_α in_β) ∧ ¬(proposition_4 perp_α in_β) :=
by
  sorry

end proof_problem_l189_189648


namespace Cameron_task_completion_l189_189365

theorem Cameron_task_completion (C : ℝ) (h1 : ∃ x, x = 9 / C) (h2 : ∃ y, y = 1 / 2) (total_work : ∃ z, z = 1):
  9 - 9 / C + 1/2 = 1 -> C = 18 := by
  sorry

end Cameron_task_completion_l189_189365


namespace num_real_solutions_equation_l189_189687

theorem num_real_solutions_equation :
  ∃ (xs : Finset ℝ), (∀ x ∈ xs, 2^(2*x + 2) - 2^(x + 4) - 2^x + 4 = 0) ∧ xs.card = 2 :=
by {
  sorry
}

end num_real_solutions_equation_l189_189687


namespace increasing_function_positive_l189_189321

theorem increasing_function_positive 
  {f : ℝ → ℝ} {a b : ℝ} (h_deriv_pos : ∀ x, a < x ∧ x < b → f' x > 0) 
  (h_fa_nonneg : f a ≥ 0) :
  ∀ x, a < x ∧ x < b → f x > 0 :=
sorry

end increasing_function_positive_l189_189321


namespace range_of_y_l189_189627

theorem range_of_y (a b y : ℝ) (hab : a + b = 2) (hbl : b ≤ 2) (hy : y = a^2 + 2*a - 2) : y ≥ -2 :=
by
  sorry

end range_of_y_l189_189627


namespace height_drawn_to_hypotenuse_l189_189438

-- Definitions for the given problem
variables {A B C D : Type}
variables {area : ℝ}
variables {angle_ratio : ℝ}
variables {h : ℝ}

-- Given conditions
def is_right_triangle (A B C : Type) : Prop := -- definition for the right triangle
sorry

def area_of_triangle (A B C : Type) (area: ℝ) : Prop := 
area = ↑(2 : ℝ) * Real.sqrt 3  -- area given as 2√3 cm²

def angle_bisector_ratios (A B C D : Type) (ratio: ℝ) : Prop :=
ratio = 1 / 2  -- given ratio 1:2

-- Question statement
theorem height_drawn_to_hypotenuse (A B C D : Type) 
  (right_triangle : is_right_triangle A B C)
  (area_cond : area_of_triangle A B C area)
  (angle_ratio_cond : angle_bisector_ratios A B C D angle_ratio):
  h = Real.sqrt 3 :=
sorry

end height_drawn_to_hypotenuse_l189_189438


namespace horner_method_multiplications_additions_l189_189636

def polynomial (n : ℕ) : Type :=
  { a : ℕ → ℝ // ∀ m, m > n → a m = 0 }

theorem horner_method_multiplications_additions (n : ℕ) (a : polynomial n) (x : ℝ) :
  ∃ m a, (eval_horner a x).num_mul = n ∧ (eval_horner a x).num_add = n := sorry

structure operations :=
  (num_mul : ℕ)
  (num_add : ℕ)

def eval_horner {n : ℕ} (a : polynomial n) (x : ℝ) : operations :=
  let f := λ (p : operations) (i : ℕ), { num_mul := p.num_mul + 1, num_add := p.num_add + 1 }
  in { num_mul := n, num_add := n }

noncomputable def eval (n : ℕ) (a : polynomial n) (x : ℝ) : ℝ :=
  let rec eval_aux (i : ℕ) (acc : ℝ) :=
    if h : i > n then acc
    else eval_aux (i+1) (a.val i + x * acc)
  in eval_aux 0 0

example (n : ℕ) (a : polynomial n) (x : ℝ) : eval_horner a x = { num_mul := n, num_add := n } := sorry

end horner_method_multiplications_additions_l189_189636


namespace total_cost_correct_l189_189479

-- Condition C1: There are 13 hearts in a deck of 52 playing cards. 
def hearts_in_deck : ℕ := 13

-- Condition C2: The number of cows is twice the number of hearts.
def cows_in_Devonshire : ℕ := 2 * hearts_in_deck

-- Condition C3: Each cow is sold at $200.
def cost_per_cow : ℕ := 200

-- Question Q1: Calculate the total cost of the cows.
def total_cost_of_cows : ℕ := cows_in_Devonshire * cost_per_cow

-- Final statement we need to prove
theorem total_cost_correct : total_cost_of_cows = 5200 := by
  -- This will be proven in the proof body
  sorry

end total_cost_correct_l189_189479


namespace arithmetic_sequence_a3_l189_189350

theorem arithmetic_sequence_a3 (a : ℕ → ℕ) (h1 : a 6 = 6) (h2 : a 9 = 9) : a 3 = 3 :=
by
  -- proof goes here
  sorry

end arithmetic_sequence_a3_l189_189350


namespace consecutive_integer_sum_l189_189871

noncomputable def sum_of_consecutive_integers (x : ℕ) : ℕ :=
x + (x + 1)

theorem consecutive_integer_sum (x : ℕ) (h : x * (x + 1) = 812) : sum_of_consecutive_integers x = 57 :=
sorry

end consecutive_integer_sum_l189_189871


namespace worth_of_presents_l189_189750

def ring_value := 4000
def car_value := 2000
def bracelet_value := 2 * ring_value

def total_worth := ring_value + car_value + bracelet_value

theorem worth_of_presents : total_worth = 14000 := by
  sorry

end worth_of_presents_l189_189750


namespace num_nat_least_prime_divisor_17_in_range_l189_189578

/-- 
Compute the number of natural numbers \(1 \leq n \leq 10^6\)
such that the least prime divisor of \(n\) is \(17\). 
--/
def leastPrimeDivisorIsSeventeen (n : ℕ) : Prop :=
  nat.find_min (λ p, nat.prime p ∧ p ∣ n) = 17

theorem num_nat_least_prime_divisor_17_in_range :
  (finset.filter leastPrimeDivisorIsSeventeen (finset.Icc 1 1000000)).card = 11323 :=
begin
  sorry
end

end num_nat_least_prime_divisor_17_in_range_l189_189578


namespace infinite_powers_of_two_in_sequence_l189_189807

theorem infinite_powers_of_two_in_sequence :
  ∃ᶠ n in at_top, ∃ k : ℕ, ∃ a : ℕ, (a = ⌊n * Real.sqrt 2⌋ ∧ a = 2^k) :=
sorry

end infinite_powers_of_two_in_sequence_l189_189807


namespace problem_l189_189848

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189848


namespace xy_value_l189_189323

namespace ProofProblem

variables {x y : ℤ}

theorem xy_value (h1 : x * (x + y) = x^2 + 12) (h2 : x - y = 3) : x * y = 12 :=
by
  -- The proof is not required here
  sorry

end ProofProblem

end xy_value_l189_189323


namespace consecutive_integers_sum_l189_189904

theorem consecutive_integers_sum (x y : ℕ) (h : x * y = 812) (hc : y = x + 1) : x + y = 57 :=
sorry

end consecutive_integers_sum_l189_189904


namespace vegetable_weights_l189_189371

variable (B : ℕ) (B_s : ℕ) (V_t : ℕ) (C P B_v : ℕ)

def initial_conditions (B : ℕ) : Prop := B = 4

def beef_used_in_soup (B B_s : ℕ) : Prop := B_s = B - 1

def total_weight_of_vegetables (B_s V_t : ℕ) : Prop := V_t = 2 * B_s

def equal_amounts_of_carrots_and_potatoes (C P : ℕ) : Prop := C = P

def bell_peppers_twice_of_carrots (C B_v : ℕ) : Prop := B_v = 2 * C

def total_weight_relation (C P B_v V_t : ℕ) : Prop := C + P + B_v = V_t

theorem vegetable_weights (B B_s V_t C P B_v : ℕ)
  (h1 : initial_conditions B)
  (h2 : beef_used_in_soup B B_s)
  (h3 : total_weight_of_vegetables B_s V_t)
  (h4 : equal_amounts_of_carrots_and_potatoes C P)
  (h5 : bell_peppers_twice_of_carrots C B_v)
  (h6 : total_weight_relation C P B_v V_t) :
  C = 1.5 ∧ P = 1.5 ∧ B_v = 3 :=
by
  sorry

end vegetable_weights_l189_189371


namespace inclination_angle_of_line_l189_189054

theorem inclination_angle_of_line (
  a b c : ℝ
  (h_line_eq : ∀ x y : ℝ, a * x - y + c = 0)
  (h_point_on_line : h_line_eq (sqrt 3) 4 = true)
) : ∃ θ : ℝ, θ = 60 := 
by
  sorry

end inclination_angle_of_line_l189_189054


namespace time_spent_on_Type_A_problems_l189_189134

theorem time_spent_on_Type_A_problems (total_questions : ℕ)
  (exam_duration_minutes : ℕ) (type_A_problems : ℕ)
  (time_ratio_A_B : ℕ) : ℕ :=
  let total_questions = 200 
  let exam_duration_minutes = 180 
  let type_A_problems = 25 
  let time_ratio_A_B = 2 
  40 := 
  sorry

end time_spent_on_Type_A_problems_l189_189134


namespace find_value_of_m_l189_189262

theorem find_value_of_m : ∃ m : ℤ, 2^4 - 3 = 5^2 + m ∧ m = -12 :=
by
  use -12
  sorry

end find_value_of_m_l189_189262


namespace find_line_equation_l189_189603

-- Definitions and conditions from the problem
def point : ℝ × ℝ := (1, 0)
def line1 (x y : ℝ) : Prop := 3 * x + y - 6 = 0
def line2 (x y : ℝ) : Prop := 3 * x + y + 3 = 0
def segment_length : ℝ := (9 * Real.sqrt 10) / 10

-- The proof goal
theorem find_line_equation (x y : ℝ) :
  (∃ x y, line1 x y ∧ line2 x y ∧ Real.sqrt ( (x - y)^2 + (x - y)^2 ) = segment_length ∧ (point = (1, 0))) → 
  (x - 3 * y - 1 = 0) :=
sorry

end find_line_equation_l189_189603


namespace problem_l189_189849

noncomputable def consecutive_integers_sum (x : ℕ) : ℕ :=
  x + (x + 1)

theorem problem (x : ℕ) (hx : x * (x + 1) = 812) : consecutive_integers_sum x = 57 := by
  sorry

end problem_l189_189849


namespace certain_number_is_two_l189_189129

variable (x : ℕ)  -- x is the certain number

-- Condition: Given that adding 6 incorrectly results in 8
axiom h1 : x + 6 = 8

-- The mathematically equivalent proof problem Lean statement
theorem certain_number_is_two : x = 2 :=
by
  sorry

end certain_number_is_two_l189_189129


namespace parabola_x_intercepts_l189_189219

noncomputable def parabola (y : ℝ) : ℝ := -3 * y^2 + 3 * y + 3

theorem parabola_x_intercepts : {x : ℝ // ∃ y : ℝ, parabola y = x} = 3 := by
  sorry

end parabola_x_intercepts_l189_189219


namespace max_area_of_rect_l189_189698

theorem max_area_of_rect (x y : ℝ) (h1 : x + y = 10) : 
  x * y ≤ 25 :=
by 
  sorry

end max_area_of_rect_l189_189698
