import Mathlib

namespace expedition_ratios_l2267_226731

theorem expedition_ratios (F : ℕ) (S : ℕ) (L : ℕ) (R : ℕ) 
  (h1 : F = 3) 
  (h2 : S = F + 2) 
  (h3 : F + S + L = 18) 
  (h4 : L = R * S) : 
  R = 2 := 
sorry

end expedition_ratios_l2267_226731


namespace smallest_a₁_l2267_226704

-- We define the sequence a_n and its recurrence relation
def a (n : ℕ) (a₁ : ℝ) : ℝ :=
  match n with
  | 0     => 0  -- this case is not used, but included for function completeness
  | 1     => a₁
  | (n+2) => 11 * a (n+1) a₁ - (n+2)

theorem smallest_a₁ : ∃ a₁ : ℝ, (a₁ = 21 / 100) ∧ ∀ n > 1, a n a₁ > 0 := 
  sorry

end smallest_a₁_l2267_226704


namespace inequality_relation_l2267_226780

open Real

theorem inequality_relation (x : ℝ) :
  ¬ ((∀ x, (x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
     (∀ x, (x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) := 
by
  sorry

end inequality_relation_l2267_226780


namespace no_such_integers_exist_l2267_226749

theorem no_such_integers_exist :
  ¬(∃ (a b c d : ℤ), a * 19^3 + b * 19^2 + c * 19 + d = 1 ∧ a * 62^3 + b * 62^2 + c * 62 + d = 2) :=
by
  sorry

end no_such_integers_exist_l2267_226749


namespace max_expression_value_l2267_226705

theorem max_expression_value (a b c d : ℝ) 
  (h1 : -6.5 ≤ a ∧ a ≤ 6.5) 
  (h2 : -6.5 ≤ b ∧ b ≤ 6.5) 
  (h3 : -6.5 ≤ c ∧ c ≤ 6.5) 
  (h4 : -6.5 ≤ d ∧ d ≤ 6.5) : 
  a + 2*b + c + 2*d - a*b - b*c - c*d - d*a ≤ 182 :=
sorry

end max_expression_value_l2267_226705


namespace vending_machine_users_l2267_226739

theorem vending_machine_users (p_fail p_double p_single : ℚ) (total_snacks : ℕ) (P : ℕ) :
  p_fail = 1 / 6 ∧ p_double = 1 / 10 ∧ p_single = 1 - 1 / 6 - 1 / 10 ∧
  total_snacks = 28 →
  P = 30 :=
by
  intros h
  sorry

end vending_machine_users_l2267_226739


namespace evaluate_expression_l2267_226771

variable (b x : ℝ)

theorem evaluate_expression (h : x = b + 9) : x - b + 4 = 13 := by
  sorry

end evaluate_expression_l2267_226771


namespace simplify_and_evaluate_expr_l2267_226743

theorem simplify_and_evaluate_expr (a : ℝ) (h1 : -1 < a) (h2 : a < Real.sqrt 5) (h3 : a = 2) :
  (a - (a^2 / (a^2 - 1))) / (a^2 / (a^2 - 1)) = 1 / 2 :=
by
  sorry

end simplify_and_evaluate_expr_l2267_226743


namespace sum_of_squares_not_7_mod_8_l2267_226701

theorem sum_of_squares_not_7_mod_8 (a b c : ℤ) : (a^2 + b^2 + c^2) % 8 ≠ 7 :=
sorry

end sum_of_squares_not_7_mod_8_l2267_226701


namespace probability_of_green_tile_l2267_226706

theorem probability_of_green_tile :
  let total_tiles := 100
  let green_tiles := 14
  let probability := green_tiles / total_tiles
  probability = 7 / 50 :=
by
  sorry

end probability_of_green_tile_l2267_226706


namespace find_c_value_l2267_226796

theorem find_c_value (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 4) (h3 : x2 = 5) (h4 : y2 = 0) (c : ℝ)
  (h5 : 3 * ((x1 + x2) / 2) - 2 * ((y1 + y2) / 2) = c) : c = 5 :=
sorry

end find_c_value_l2267_226796


namespace bridge_length_l2267_226725

   noncomputable def walking_speed_km_per_hr : ℝ := 6
   noncomputable def walking_time_minutes : ℝ := 15

   noncomputable def length_of_bridge (speed_km_per_hr : ℝ) (time_min : ℝ) : ℝ :=
     (speed_km_per_hr * 1000 / 60) * time_min

   theorem bridge_length :
     length_of_bridge walking_speed_km_per_hr walking_time_minutes = 1500 := 
   by
     sorry
   
end bridge_length_l2267_226725


namespace polygon_sides_exterior_angle_l2267_226758

theorem polygon_sides_exterior_angle (n : ℕ) (h : 360 / 24 = n) : n = 15 := by
  sorry

end polygon_sides_exterior_angle_l2267_226758


namespace distance_from_original_position_l2267_226786

/-- Definition of initial problem conditions and parameters --/
def square_area (l : ℝ) : Prop :=
  l * l = 18

def folded_area_relation (x : ℝ) : Prop :=
  0.5 * x^2 = 2 * (18 - 0.5 * x^2)

/-- The main statement that needs to be proved --/
theorem distance_from_original_position :
  ∃ (A_initial A_folded_dist : ℝ),
    square_area A_initial ∧
    (∃ x : ℝ, folded_area_relation x ∧ A_folded_dist = 2 * Real.sqrt 6 * Real.sqrt 2) ∧
    A_folded_dist = 4 * Real.sqrt 3 :=
by
  -- The proof is omitted here; providing structure for the problem.
  sorry

end distance_from_original_position_l2267_226786


namespace primes_and_one_l2267_226733

-- Given conditions:
variables {a n : ℕ}
variable (ha : a > 100 ∧ a % 2 = 1)  -- a is an odd natural number greater than 100
variable (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime (a - n^2) / 4)  -- for all n ≤ √(a / 5), (a - n^2) / 4 is prime

-- Theorem: For all n > √(a / 5), (a - n^2) / 4 is either prime or 1
theorem primes_and_one {a : ℕ} (ha : a > 100 ∧ a % 2 = 1)
  (hn_bound : ∀ n ≤ Nat.sqrt (a / 5), Prime ((a - n^2) / 4)) :
  ∀ n > Nat.sqrt (a / 5), Prime ((a - n^2) / 4) ∨ ((a - n^2) / 4) = 1 :=
sorry

end primes_and_one_l2267_226733


namespace room_length_l2267_226781

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sq_meter : ℝ) (length : ℝ) : 
  width = 3.75 ∧ total_cost = 14437.5 ∧ cost_per_sq_meter = 700 → length = 5.5 :=
by
  sorry

end room_length_l2267_226781


namespace certain_fraction_ratio_l2267_226772

theorem certain_fraction_ratio :
  (∃ (x y : ℚ), (x / y) / (6 / 5) = (2 / 5) / 0.14285714285714288) →
  (∃ (x y : ℚ), x / y = 84 / 25) := 
  by
    intros h_ratio
    have h_rat := h_ratio
    sorry

end certain_fraction_ratio_l2267_226772


namespace tan_2alpha_value_beta_value_l2267_226785

variable (α β : ℝ)
variable (h1 : 0 < β ∧ β < α ∧ α < π / 2)
variable (h2 : Real.cos α = 1 / 7)
variable (h3 : Real.cos (α - β) = 13 / 14)

theorem tan_2alpha_value : Real.tan (2 * α) = - (8 * Real.sqrt 3 / 47) :=
by
  sorry

theorem beta_value : β = π / 3 :=
by
  sorry

end tan_2alpha_value_beta_value_l2267_226785


namespace evariste_stairs_l2267_226714

def num_ways (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else num_ways (n - 1) + num_ways (n - 2)

theorem evariste_stairs (n : ℕ) : num_ways n = u_n :=
  sorry

end evariste_stairs_l2267_226714


namespace approximation_accuracy_l2267_226752

noncomputable def radius (k : Circle) : ℝ := sorry
def BG_equals_radius (BG : ℝ) (r : ℝ) := BG = r
def DB_equals_radius_sqrt3 (DB DG r : ℝ) := DB = DG ∧ DG = r * Real.sqrt 3
def cos_beta (cos_beta : ℝ) := cos_beta = 1 / (2 * Real.sqrt 3)
def sin_beta (sin_beta : ℝ) := sin_beta = Real.sqrt 11 / (2 * Real.sqrt 3)
def angle_BCH (angle_BCH : ℝ) (beta : ℝ) := angle_BCH = 120 - beta
def side_nonagon (a_9 r : ℝ) := a_9 = 2 * r * Real.sin 20
def bounds_sin_20 (sin_20 : ℝ) := 0.34195 < sin_20 ∧ sin_20 < 0.34205
def error_margin_low (BH_low a_9 r : ℝ) := 0.6839 * r < a_9
def error_margin_high (BH_high a_9 r : ℝ) := a_9 < 0.6841 * r

theorem approximation_accuracy
  (r : ℝ) (BG DB DG : ℝ) (beta : ℝ) (a_9 BH_low BH_high : ℝ)
  (h1 : BG_equals_radius BG r)
  (h2 : DB_equals_radius_sqrt3 DB DG r)
  (h3 : cos_beta (1 / (2 * Real.sqrt 3)))
  (h4 : sin_beta (Real.sqrt 11 / (2 * Real.sqrt 3)))
  (h5 : angle_BCH (120 - beta) beta)
  (h6 : side_nonagon a_9 r)
  (h7 : bounds_sin_20 (Real.sin 20))
  (h8 : error_margin_low BH_low a_9 r)
  (h9 : error_margin_high BH_high a_9 r) : 
  0.6861 * r < BH_high ∧ BH_low < 0.6864 * r :=
sorry

end approximation_accuracy_l2267_226752


namespace min_value_x_plus_one_over_2x_l2267_226732

theorem min_value_x_plus_one_over_2x (x : ℝ) (hx : x > 0) : 
  x + 1 / (2 * x) ≥ Real.sqrt 2 := sorry

end min_value_x_plus_one_over_2x_l2267_226732


namespace triangle_non_existence_triangle_existence_l2267_226748

-- Definition of the triangle inequality theorem for a triangle with given sides.
def triangle_exists (a b c : ℕ) : Prop := (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_non_existence (h : ¬ triangle_exists 2 3 7) : true := by
  sorry

theorem triangle_existence (h : triangle_exists 5 5 5) : true := by
  sorry

end triangle_non_existence_triangle_existence_l2267_226748


namespace average_mpg_correct_l2267_226790

noncomputable def average_mpg (initial_miles final_miles : ℕ) (refill1 refill2 refill3 : ℕ) : ℚ :=
  let distance := final_miles - initial_miles
  let total_gallons := refill1 + refill2 + refill3
  distance / total_gallons

theorem average_mpg_correct :
  average_mpg 32000 33100 15 10 22 = 23.4 :=
by
  sorry

end average_mpg_correct_l2267_226790


namespace min_product_sum_l2267_226768

theorem min_product_sum (a : Fin 7 → ℕ) (b : Fin 7 → ℕ) 
  (h2 : ∀ i, 2 ≤ a i) 
  (h3 : ∀ i, a i ≤ 166) 
  (h4 : ∀ i, a i ^ b i % 167 = a (i + 1) % 7 + 1 ^ 2 % 167) : 
  b 0 * b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * (b 0 + b 1 + b 2 + b 3 + b 4 + b 5 + b 6) = 675 := sorry

end min_product_sum_l2267_226768


namespace calculate_nabla_l2267_226767

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calculate_nabla : nabla (nabla 2 3) 4 = 11 / 9 :=
by
  sorry

end calculate_nabla_l2267_226767


namespace factorization1_factorization2_factorization3_l2267_226774

-- (1) Prove x^3 - 6x^2 + 9x == x(x-3)^2
theorem factorization1 (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

-- (2) Prove (x-2)^2 - x + 2 == (x-2)(x-3)
theorem factorization2 (x : ℝ) : (x - 2)^2 - x + 2 = (x - 2) * (x - 3) :=
by sorry

-- (3) Prove (x^2 + y^2)^2 - 4x^2*y^2 == (x + y)^2(x - y)^2
theorem factorization3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
by sorry

end factorization1_factorization2_factorization3_l2267_226774


namespace units_digit_of_result_l2267_226773

def tens_plus_one (a b : ℕ) : Prop := a = b + 1

theorem units_digit_of_result (a b : ℕ) (h : tens_plus_one a b) :
  ((10 * a + b) - (10 * b + a)) % 10 = 9 :=
by
  -- Let's mark this part as incomplete using sorry.
  sorry

end units_digit_of_result_l2267_226773


namespace isabel_initial_candy_l2267_226712

theorem isabel_initial_candy (total_candy : ℕ) (candy_given : ℕ) (initial_candy : ℕ) :
  candy_given = 25 → total_candy = 93 → total_candy = initial_candy + candy_given → initial_candy = 68 :=
by
  intros h_candy_given h_total_candy h_eq
  rw [h_candy_given, h_total_candy] at h_eq
  sorry

end isabel_initial_candy_l2267_226712


namespace find_x_l2267_226798

theorem find_x (x : ℚ) (h : (35 / 100) * x = (40 / 100) * 50) : 
  x = 400 / 7 :=
sorry

end find_x_l2267_226798


namespace select_team_l2267_226799

-- Definition of the problem conditions 
def boys : Nat := 10
def girls : Nat := 12
def team_size : Nat := 8
def boys_in_team : Nat := 4
def girls_in_team : Nat := 4

-- Given conditions reflect in the Lean statement that needs proof
theorem select_team : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 103950 :=
by
  sorry

end select_team_l2267_226799


namespace point_not_on_line_l2267_226778

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : ¬ ((2023, 0) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) :=
by
  -- proof is omitted
  sorry

end point_not_on_line_l2267_226778


namespace solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l2267_226746

-- Equation (1)
theorem solve_quadratic_eq1 (x : ℝ) : x^2 + 16 = 8*x ↔ x = 4 := by
  sorry

-- Equation (2)
theorem solve_quadratic_eq2 (x : ℝ) : 2*x^2 + 4*x - 3 = 0 ↔ 
  x = -1 + (Real.sqrt 10) / 2 ∨ x = -1 - (Real.sqrt 10) / 2 := by
  sorry

-- Equation (3)
theorem solve_quadratic_eq3 (x : ℝ) : x*(x - 1) = x ↔ x = 0 ∨ x = 2 := by
  sorry

-- Equation (4)
theorem solve_quadratic_eq4 (x : ℝ) : x*(x + 4) = 8*x - 3 ↔ x = 3 ∨ x = 1 := by
  sorry

end solve_quadratic_eq1_solve_quadratic_eq2_solve_quadratic_eq3_solve_quadratic_eq4_l2267_226746


namespace tangent_line_condition_l2267_226753

theorem tangent_line_condition (a b : ℝ):
  ((a = 1 ∧ b = 1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) ∧
  ( (a = -1 ∧ b = -1) → ∀ x y : ℝ, x + y = 0 → (x - a)^2 + (y - b)^2 = 2 → x = 0 ∧ y = 0) →
  (a = 1 ∧ b = 1) ∨ (a = -1 ∧ b = -1) :=
by
  sorry

end tangent_line_condition_l2267_226753


namespace negation_of_proposition_l2267_226700

theorem negation_of_proposition (m : ℤ) : 
  (¬ (∃ x : ℤ, x^2 + 2*x + m ≤ 0)) ↔ ∀ x : ℤ, x^2 + 2*x + m > 0 :=
sorry

end negation_of_proposition_l2267_226700


namespace caloprian_lifespan_proof_l2267_226723

open Real

noncomputable def timeDilation (delta_t : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  delta_t * sqrt (1 - (v ^ 2) / (c ^ 2))

noncomputable def caloprianMinLifeSpan (d : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  let earth_time := (d / v) * 2
  timeDilation earth_time v c

theorem caloprian_lifespan_proof :
  caloprianMinLifeSpan 30 0.3 1 = 20 * sqrt 91 :=
sorry

end caloprian_lifespan_proof_l2267_226723


namespace find_positive_integer_l2267_226737

variable (z : ℕ)

theorem find_positive_integer
  (h1 : (4 * z)^2 - z = 2345)
  (h2 : 0 < z) :
  z = 7 :=
sorry

end find_positive_integer_l2267_226737


namespace range_of_a_l2267_226766

theorem range_of_a
  (P : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) :
  ¬P → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l2267_226766


namespace sum_of_first_n_terms_l2267_226784

-- Define the sequence aₙ
def a (n : ℕ) : ℕ := 2 * n - 1

-- Prove that the sum of the first n terms of the sequence is n²
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range (n+1)).sum a = n^2 :=
by sorry -- Proof is skipped

end sum_of_first_n_terms_l2267_226784


namespace a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l2267_226742

noncomputable def a_n (n : ℕ) : ℕ := 3 * n

noncomputable def b_n (n : ℕ) : ℕ := 3 * n + 2^(n - 1)

noncomputable def S_n (n : ℕ) : ℕ := (3 * n * (n + 1) / 2) + (2^n - 1)

theorem a_n_is_arithmetic_sequence (n : ℕ) :
  (a_n 1 = 3) ∧ (a_n 4 = 12) ∧ (∀ n : ℕ, a_n n = 3 * n) :=
by
  sorry

theorem b_n_is_right_sequence (n : ℕ) :
  (b_n 1 = 4) ∧ (b_n 4 = 20) ∧ (∀ n : ℕ, b_n n = 3 * n + 2^(n - 1)) ∧ 
  (∀ n : ℕ, b_n n - a_n n = 2^(n - 1)) :=
by
  sorry

theorem sum_first_n_terms_b_n (n : ℕ) :
  S_n n = 3 * (n * (n + 1) / 2) + 2^n - 1 :=
by
  sorry

end a_n_is_arithmetic_sequence_b_n_is_right_sequence_sum_first_n_terms_b_n_l2267_226742


namespace whole_number_M_l2267_226729

theorem whole_number_M (M : ℤ) (hM : 9 < (M : ℝ) / 4 ∧ (M : ℝ) / 4 < 10) : M = 37 ∨ M = 38 ∨ M = 39 := by
  sorry

end whole_number_M_l2267_226729


namespace an_geometric_l2267_226765

-- Define the functions and conditions
def f (x : ℝ) (b : ℝ) : ℝ := b * x + 1

def g (n : ℕ) (b : ℝ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => f (g n b) b

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ :=
  g (n + 1) b - g n b

-- Prove that a_n is a geometric sequence
theorem an_geometric (b : ℝ) (h : b ≠ 1) : 
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) b = q * a n b :=
sorry

end an_geometric_l2267_226765


namespace sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l2267_226760

theorem sqrt_99_eq_9801 : 99^2 = 9801 := by
  sorry

theorem expr_2000_1999_2001_eq_1 : 2000^2 - 1999 * 2001 = 1 := by
  sorry

end sqrt_99_eq_9801_expr_2000_1999_2001_eq_1_l2267_226760


namespace fraction_studying_japanese_l2267_226747

variable (J S : ℕ)
variable (hS : S = 3 * J)

def fraction_of_seniors_studying_japanese := (1 / 3 : ℚ) * S
def fraction_of_juniors_studying_japanese := (3 / 4 : ℚ) * J

def total_students := S + J

theorem fraction_studying_japanese (J S : ℕ) (hS : S = 3 * J) :
  ((1 / 3 : ℚ) * S + (3 / 4 : ℚ) * J) / (S + J) = 7 / 16 :=
by {
  -- proof to be filled in
  sorry
}

end fraction_studying_japanese_l2267_226747


namespace not_perfect_square_l2267_226763

theorem not_perfect_square (n : ℕ) : ¬ ∃ k : ℕ, 7 * n + 3 = k^2 := 
by
  sorry

end not_perfect_square_l2267_226763


namespace problem_l2267_226783

theorem problem (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z)
  (coprime : Nat.gcd X (Nat.gcd Y Z) = 1)
  (h : X * Real.log 3 / Real.log 100 + Y * Real.log 4 / Real.log 100 = Z):
  X + Y + Z = 4 :=
sorry

end problem_l2267_226783


namespace f_monotonic_increasing_l2267_226779

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

theorem f_monotonic_increasing :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 > x2 → f x1 > f x2 :=
by
  intros x1 x2 hx1 hx2 h
  sorry

end f_monotonic_increasing_l2267_226779


namespace polygon_sides_l2267_226727

theorem polygon_sides (n : ℕ) (h : 180 * (n - 2) = 720) : n = 6 :=
sorry

end polygon_sides_l2267_226727


namespace warehouse_width_l2267_226721

theorem warehouse_width (L : ℕ) (circles : ℕ) (total_distance : ℕ)
  (hL : L = 600)
  (hcircles : circles = 8)
  (htotal_distance : total_distance = 16000) : 
  ∃ W : ℕ, 2 * L + 2 * W = (total_distance / circles) ∧ W = 400 :=
by
  sorry

end warehouse_width_l2267_226721


namespace size_of_each_group_l2267_226717

theorem size_of_each_group 
  (skittles : ℕ) (erasers : ℕ) (groups : ℕ)
  (h_skittles : skittles = 4502) (h_erasers : erasers = 4276) (h_groups : groups = 154) :
  (skittles + erasers) / groups = 57 :=
by
  sorry

end size_of_each_group_l2267_226717


namespace solve_first_train_length_l2267_226791

noncomputable def first_train_length (time: ℝ) (speed1_kmh: ℝ) (speed2_kmh: ℝ) (length2: ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * 1000 / 3600
  let speed2_ms := speed2_kmh * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem solve_first_train_length :
  first_train_length 7.0752960452818945 80 65 165 = 120.28 :=
by
  simp [first_train_length]
  norm_num
  sorry

end solve_first_train_length_l2267_226791


namespace selene_sandwiches_l2267_226754

-- Define the context and conditions in Lean
variables (S : ℕ) (sandwich_cost hamburger_cost hotdog_cost juice_cost : ℕ)
  (selene_cost tanya_cost total_cost : ℕ)

-- Each item prices
axiom sandwich_price : sandwich_cost = 2
axiom hamburger_price : hamburger_cost = 2
axiom hotdog_price : hotdog_cost = 1
axiom juice_price : juice_cost = 2

-- Purchases
axiom selene_purchase : selene_cost = sandwich_cost * S + juice_cost
axiom tanya_purchase : tanya_cost = hamburger_cost * 2 + juice_cost * 2

-- Total spending
axiom total_spending : selene_cost + tanya_cost = 16

-- Goal: Prove that Selene bought 3 sandwiches
theorem selene_sandwiches : S = 3 :=
by {
  sorry
}

end selene_sandwiches_l2267_226754


namespace find_triple_abc_l2267_226718

theorem find_triple_abc (a b c : ℝ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
    (h_sum : a + b + c = 3)
    (h2 : a^2 - a ≥ 1 - b * c)
    (h3 : b^2 - b ≥ 1 - a * c)
    (h4 : c^2 - c ≥ 1 - a * b) :
    a = 1 ∧ b = 1 ∧ c = 1 :=
by
  sorry

end find_triple_abc_l2267_226718


namespace sum_of_solutions_l2267_226710

theorem sum_of_solutions (x1 x2 : ℝ) (h1 : (x1 - 8) ^ 2 = 49) (h2 : (x2 - 8) ^ 2 = 49) : x1 + x2 = 16 :=
sorry

end sum_of_solutions_l2267_226710


namespace ratio_of_amounts_l2267_226769

theorem ratio_of_amounts (B J P : ℝ) (hB : B = 60) (hP : P = (1 / 3) * B) (hJ : J = B - 20) : J / P = 2 :=
by
  have hP_val : P = 20 := by sorry
  have hJ_val : J = 40 := by sorry
  have ratio : J / P = 40 / 20 := by sorry
  show J / P = 2
  sorry

end ratio_of_amounts_l2267_226769


namespace election_votes_l2267_226744

theorem election_votes (V : ℝ) (h1 : 0.70 * V - 0.30 * V = 200) : V = 500 :=
sorry

end election_votes_l2267_226744


namespace arsenic_acid_concentration_equilibrium_l2267_226734

noncomputable def dissociation_constants 
  (Kd1 Kd2 Kd3 : ℝ) (H3AsO4 H2AsO4 HAsO4 AsO4 H : ℝ) : Prop :=
  Kd1 = (H * H2AsO4) / H3AsO4 ∧ Kd2 = (H * HAsO4) / H2AsO4 ∧ Kd3 = (H * AsO4) / HAsO4

theorem arsenic_acid_concentration_equilibrium :
  dissociation_constants 5.6e-3 1.7e-7 2.95e-12 0.1 (2e-2) (1.7e-7) (0) (2e-2) :=
by sorry

end arsenic_acid_concentration_equilibrium_l2267_226734


namespace car_speed_travel_l2267_226792

theorem car_speed_travel (v : ℝ) :
  600 = 3600 / 6 ∧
  (6 : ℝ) = (3600 / v) + 2 →
  v = 900 :=
by
  sorry

end car_speed_travel_l2267_226792


namespace system_solution_a_l2267_226728

theorem system_solution_a (x y z : ℤ) (h1 : x^2 + x * y + y^2 = 7) (h2 : y^2 + y * z + z^2 = 13) (h3 : z^2 + z * x + x^2 = 19) :
  (x = 2 ∧ y = 1 ∧ z = 3) ∨ (x = -2 ∧ y = -1 ∧ z = -3) :=
sorry

end system_solution_a_l2267_226728


namespace construct_right_triangle_l2267_226788

theorem construct_right_triangle (hypotenuse : ℝ) (ε : ℝ) (h_positive : 0 < ε) (h_less_than_ninety : ε < 90) :
    ∃ α β : ℝ, α + β = 90 ∧ α - β = ε ∧ 45 < α ∧ α < 90 :=
by
  sorry

end construct_right_triangle_l2267_226788


namespace _l2267_226735

/-- This theorem states that if the GCD of 8580 and 330 is diminished by 12, the result is 318. -/
example : (Int.gcd 8580 330) - 12 = 318 :=
by
  sorry

end _l2267_226735


namespace triangle_has_120_degree_l2267_226713

noncomputable def angles_of_triangle (α β γ : Real) : Prop :=
  α + β + γ = 180

theorem triangle_has_120_degree (α β γ : Real)
    (h1 : angles_of_triangle α β γ)
    (h2 : Real.cos (3 * α) + Real.cos (3 * β) + Real.cos (3 * γ) = 1) :
  γ = 120 :=
  sorry

end triangle_has_120_degree_l2267_226713


namespace find_number_l2267_226787

theorem find_number (x n : ℤ) (h1 : |x| = 9 * x - n) (h2 : x = 2) : n = 16 := by 
  sorry

end find_number_l2267_226787


namespace max_value_of_quadratic_l2267_226750

theorem max_value_of_quadratic :
  ∃ y : ℝ, (∀ x : ℝ, y ≥ -x^2 + 5 * x - 4) ∧ y = 9 / 4 :=
sorry

end max_value_of_quadratic_l2267_226750


namespace remaining_oranges_l2267_226782

/-- Define the conditions of the problem. -/
def oranges_needed_Michaela : ℕ := 20
def oranges_needed_Cassandra : ℕ := 2 * oranges_needed_Michaela
def total_oranges_picked : ℕ := 90

/-- State the proof problem. -/
theorem remaining_oranges : total_oranges_picked - (oranges_needed_Michaela + oranges_needed_Cassandra) = 30 := 
sorry

end remaining_oranges_l2267_226782


namespace perimeter_of_square_l2267_226757

theorem perimeter_of_square (s : ℕ) (h : s = 13) : 4 * s = 52 :=
by {
  sorry
}

end perimeter_of_square_l2267_226757


namespace exists_xn_gt_yn_l2267_226709

noncomputable def x_sequence : ℕ → ℝ := sorry
noncomputable def y_sequence : ℕ → ℝ := sorry

theorem exists_xn_gt_yn
    (x1 x2 y1 y2 : ℝ)
    (hx1 : 1 < x1)
    (hx2 : 1 < x2)
    (hy1 : 1 < y1)
    (hy2 : 1 < y2)
    (h_x_seq : ∀ n, x_sequence (n + 2) = x_sequence n + (x_sequence (n + 1))^2)
    (h_y_seq : ∀ n, y_sequence (n + 2) = (y_sequence n)^2 + y_sequence (n + 1)) :
    ∃ n : ℕ, x_sequence n > y_sequence n :=
sorry

end exists_xn_gt_yn_l2267_226709


namespace f_g_of_neg2_l2267_226789

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x - 1)^2

theorem f_g_of_neg2 : f (g (-2)) = 29 := by
  -- We need to show f(g(-2)) = 29 given the definitions of f and g
  sorry

end f_g_of_neg2_l2267_226789


namespace corresponding_angles_equal_l2267_226751

theorem corresponding_angles_equal (α β γ : ℝ) (h1 : α + β + γ = 180) (h2 : α = 90) :
  (180 - α = 90 ∧ β + γ = 90 ∧ α = 90) :=
by
  sorry

end corresponding_angles_equal_l2267_226751


namespace meet_at_starting_point_l2267_226745

theorem meet_at_starting_point (track_length : Nat) (speed_A_kmph speed_B_kmph : Nat)
  (h_track_length : track_length = 1500)
  (h_speed_A : speed_A_kmph = 36)
  (h_speed_B : speed_B_kmph = 54) :
  let speed_A_mps := speed_A_kmph * 1000 / 3600
  let speed_B_mps := speed_B_kmph * 1000 / 3600
  let time_A := track_length / speed_A_mps
  let time_B := track_length / speed_B_mps
  let lcm_time := Nat.lcm time_A time_B
  lcm_time = 300 :=
by
  sorry

end meet_at_starting_point_l2267_226745


namespace problem_statement_l2267_226708

theorem problem_statement (x : ℤ) (h : Even (3 * x + 1)) : Odd (7 * x + 4) :=
  sorry

end problem_statement_l2267_226708


namespace train_bus_ratio_is_two_thirds_l2267_226722

def total_distance : ℕ := 1800
def distance_by_plane : ℕ := total_distance / 3
def distance_by_bus : ℕ := 720
def distance_by_train : ℕ := total_distance - (distance_by_plane + distance_by_bus)
def train_to_bus_ratio : ℚ := distance_by_train / distance_by_bus

theorem train_bus_ratio_is_two_thirds :
  train_to_bus_ratio = 2 / 3 := by
  sorry

end train_bus_ratio_is_two_thirds_l2267_226722


namespace first_number_is_45_l2267_226703

theorem first_number_is_45 (a b : ℕ) (h1 : a / gcd a b = 3) (h2 : b / gcd a b = 4) (h3 : lcm a b = 180) : a = 45 := by
  sorry

end first_number_is_45_l2267_226703


namespace part_I_part_II_l2267_226794

noncomputable def f (x : ℝ) := Real.sin x
noncomputable def f' (x : ℝ) := Real.cos x

theorem part_I (x : ℝ) (h : 0 < x) : f' x > 1 - x^2 / 2 := sorry

theorem part_II (a : ℝ) : (∀ x, 0 < x ∧ x < Real.pi / 2 → f x + f x / f' x > a * x) ↔ a ≤ 2 := sorry

end part_I_part_II_l2267_226794


namespace living_room_area_is_60_l2267_226719

-- Define the conditions
def carpet_length : ℝ := 4
def carpet_width : ℝ := 9
def carpet_area : ℝ := carpet_length * carpet_width
def coverage_fraction : ℝ := 0.60

-- Define the target area of the living room floor
def target_living_room_area (A : ℝ) : Prop :=
  coverage_fraction * A = carpet_area

-- State the Theorem
theorem living_room_area_is_60 (A : ℝ) (h : target_living_room_area A) : A = 60 := by
  -- Proof omitted
  sorry

end living_room_area_is_60_l2267_226719


namespace arithmetic_sequence_problem_l2267_226793

theorem arithmetic_sequence_problem : 
  ∀ (a : ℕ → ℕ) (d : ℕ), 
  a 1 = 1 →
  (a 3 + a 4 + a 5 + a 6 = 20) →
  a 8 = 9 :=
by
  intros a d h₁ h₂
  -- We skip the proof, leaving a placeholder.
  sorry

end arithmetic_sequence_problem_l2267_226793


namespace problem_I_problem_II_l2267_226730

noncomputable def f (x : ℝ) : ℝ := x - 2 * Real.sin x

theorem problem_I :
  ∀ x ∈ Set.Icc 0 Real.pi, (f x) ≥ (f (Real.pi / 3) - Real.sqrt 3) ∧ (f x) ≤ f Real.pi :=
sorry

theorem problem_II :
  ∀ a : ℝ, ((∃ x : ℝ, (0 < x ∧ x < Real.pi / 2) ∧ f x < a * x) ↔ a > -1) :=
sorry

end problem_I_problem_II_l2267_226730


namespace positive_difference_of_squares_l2267_226716

theorem positive_difference_of_squares (a b : ℕ) (h1 : a + b = 70) (h2 : a - b = 20) : a^2 - b^2 = 1400 :=
by
sorry

end positive_difference_of_squares_l2267_226716


namespace non_adjacent_ball_arrangements_l2267_226777

-- Statement only, proof is omitted
theorem non_adjacent_ball_arrangements :
  let n := (3: ℕ) -- Number of identical yellow balls
  let white_red_positions := (4: ℕ) -- Positions around the yellow unit
  let choose_positions := Nat.choose white_red_positions 2
  let arrange_balls := (2: ℕ) -- Ways to arrange the white and red balls in the chosen positions
  let total_arrangements := choose_positions * arrange_balls
  total_arrangements = 12 := 
by
  sorry

end non_adjacent_ball_arrangements_l2267_226777


namespace modular_inverse_of_2_mod_199_l2267_226762

theorem modular_inverse_of_2_mod_199 : (2 * 100) % 199 = 1 := 
by sorry

end modular_inverse_of_2_mod_199_l2267_226762


namespace range_of_m_l2267_226770

-- Define the discriminant of a quadratic equation
def discriminant(a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Proposition p: The equation x^2 - 2x + m = 0 has two distinct real roots
def p (m : ℝ) : Prop := discriminant 1 (-2) m > 0

-- Proposition q: The function y = (m + 2)x - 1 is monotonically increasing
def q (m : ℝ) : Prop := m + 2 > 0

-- The main theorem stating the conditions and proving the range of m
theorem range_of_m (m : ℝ) (hpq : p m ∨ q m) (hpnq : ¬(p m ∧ q m)) : m ≤ -2 ∨ m ≥ 1 := sorry

end range_of_m_l2267_226770


namespace part1_part2_l2267_226795

variable (a b : ℝ)

theorem part1 : ((-a)^2 * (a^2)^2 / a^3) = a^3 := sorry

theorem part2 : (a + b) * (a - b) - (a - b)^2 = 2 * a * b - 2 * b^2 := sorry

end part1_part2_l2267_226795


namespace percentage_discount_l2267_226724

theorem percentage_discount (P S : ℝ) (hP : P = 50) (hS : S = 35) : (P - S) / P * 100 = 30 := by
  sorry

end percentage_discount_l2267_226724


namespace brad_running_speed_l2267_226711

variable (dist_between_homes : ℕ)
variable (maxwell_speed : ℕ)
variable (time_maxwell_walks : ℕ)
variable (maxwell_start_time : ℕ)
variable (brad_start_time : ℕ)

#check dist_between_homes = 94
#check maxwell_speed = 4
#check time_maxwell_walks = 10
#check brad_start_time = maxwell_start_time + 1

theorem brad_running_speed (dist_between_homes : ℕ) (maxwell_speed : ℕ) (time_maxwell_walks : ℕ) (maxwell_start_time : ℕ) (brad_start_time : ℕ) :
  dist_between_homes = 94 →
  maxwell_speed = 4 →
  time_maxwell_walks = 10 →
  brad_start_time = maxwell_start_time + 1 →
  (dist_between_homes - maxwell_speed * time_maxwell_walks) / (time_maxwell_walks - (brad_start_time - maxwell_start_time)) = 6 :=
by
  intros
  sorry

end brad_running_speed_l2267_226711


namespace ethan_expected_wins_l2267_226736

-- Define the conditions
def P_win := 2 / 5
def P_tie := 2 / 5
def P_loss := 1 / 5

-- Define the adjusted probabilities
def adj_P_win := P_win / (P_win + P_loss)
def adj_P_loss := P_loss / (P_win + P_loss)

-- Define Ethan's expected number of wins before losing
def expected_wins_before_loss : ℚ := 2

-- The theorem to prove 
theorem ethan_expected_wins :
  ∃ E : ℚ, 
    E = (adj_P_win * (E + 1) + adj_P_loss * 0) ∧ 
    E = expected_wins_before_loss :=
by
  sorry

end ethan_expected_wins_l2267_226736


namespace arithmetic_sequence_properties_l2267_226707

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
∀ n, a (n + 1) = a n + d

def sum_of_first_n_terms (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
∀ n, S n = n * (a 1 + a n) / 2

def condition_S10_pos (S : ℕ → ℝ) : Prop :=
S 10 > 0

def condition_S11_neg (S : ℕ → ℝ) : Prop :=
S 11 < 0

-- Main statement
theorem arithmetic_sequence_properties {a : ℕ → ℝ} {S : ℕ → ℝ} {d : ℝ}
  (ar_seq : is_arithmetic_sequence a d)
  (sum_first_n : sum_of_first_n_terms S a)
  (S10_pos : condition_S10_pos S)
  (S11_neg : condition_S11_neg S) :
  (∀ n, (S n) / n = a 1 + (n - 1) / 2 * d) ∧
  (a 2 = 1 → -2 / 7 < d ∧ d < -1 / 4) :=
by
  sorry

end arithmetic_sequence_properties_l2267_226707


namespace debate_club_girls_l2267_226740

theorem debate_club_girls (B G : ℕ) 
  (h1 : B + G = 22)
  (h2 : B + (1/3 : ℚ) * G = 14) : G = 12 :=
sorry

end debate_club_girls_l2267_226740


namespace angle_B_in_triangle_tan_A_given_c_eq_3a_l2267_226776

theorem angle_B_in_triangle (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) : B = π / 3 := 
sorry

theorem tan_A_given_c_eq_3a (a b c A B C : ℝ) (h1 : a^2 + c^2 - b^2 = ac) (h2 : c = 3 * a) : 
(Real.tan A) = Real.sqrt 3 / 5 :=
sorry

end angle_B_in_triangle_tan_A_given_c_eq_3a_l2267_226776


namespace AM_QM_Muirhead_Inequality_l2267_226702

open Real

theorem AM_QM_Muirhead_Inequality (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) :
  ((a + b + c) / 3 = sqrt ((a^2 + b^2 + c^2) / 3) ↔ a = b ∧ b = c) ∧
  (sqrt ((a^2 + b^2 + c^2) / 3) = ((ab / c) + (bc / a) + (ca / b)) / 3 ↔ a = b ∧ b = c) :=
by sorry

end AM_QM_Muirhead_Inequality_l2267_226702


namespace ratio_price_16_to_8_l2267_226741

def price_8_inch := 5
def P : ℝ := sorry
def price_16_inch := 5 * P
def daily_earnings := 3 * price_8_inch + 5 * price_16_inch
def three_day_earnings := 3 * daily_earnings
def total_earnings := 195

theorem ratio_price_16_to_8 : total_earnings = three_day_earnings → P = 2 :=
by
  sorry

end ratio_price_16_to_8_l2267_226741


namespace prime_square_mod_six_l2267_226715

theorem prime_square_mod_six (p : ℕ) (hp : Nat.Prime p) (h : p > 5) : p^2 % 6 = 1 :=
by
  sorry

end prime_square_mod_six_l2267_226715


namespace sample_variance_is_two_l2267_226756

theorem sample_variance_is_two (a : ℝ) (h_avg : (a + 0 + 1 + 2 + 3) / 5 = 1) : 
  (1 / 5) * ((a - 1)^2 + (0 - 1)^2 + (1 - 1)^2 + (2 - 1)^2 + (3 - 1)^2) = 2 :=
sorry

end sample_variance_is_two_l2267_226756


namespace tonya_stamps_left_l2267_226761

theorem tonya_stamps_left 
    (stamps_per_matchbook : ℕ) 
    (matches_per_matchbook : ℕ) 
    (tonya_initial_stamps : ℕ) 
    (jimmy_initial_matchbooks : ℕ) 
    (stamps_per_match : ℕ) 
    (tonya_final_stamps_expected : ℕ)
    (h1 : stamps_per_matchbook = 1) 
    (h2 : matches_per_matchbook = 24) 
    (h3 : tonya_initial_stamps = 13) 
    (h4 : jimmy_initial_matchbooks = 5) 
    (h5 : stamps_per_match = 12)
    (h6 : tonya_final_stamps_expected = 3) :
    tonya_initial_stamps - jimmy_initial_matchbooks * (matches_per_matchbook / stamps_per_match) = tonya_final_stamps_expected :=
by
  sorry

end tonya_stamps_left_l2267_226761


namespace multiples_of_15_between_17_and_202_l2267_226755

theorem multiples_of_15_between_17_and_202 : 
  ∃ n : ℕ, (∀ k : ℤ, 17 < k * 15 ∧ k * 15 < 202 → k = n + 1) ∧ n = 12 :=
sorry

end multiples_of_15_between_17_and_202_l2267_226755


namespace prob_one_tails_in_three_consecutive_flips_l2267_226775

-- Define the probability of heads and tails
def P_H : ℝ := 0.5
def P_T : ℝ := 0.5

-- Define the probability of a sequence of coin flips resulting in exactly one tails in three flips
def P_one_tails_in_three_flips : ℝ :=
  P_H * P_H * P_T + P_H * P_T * P_H + P_T * P_H * P_H

-- The statement we need to prove
theorem prob_one_tails_in_three_consecutive_flips :
  P_one_tails_in_three_flips = 0.375 :=
by
  sorry

end prob_one_tails_in_three_consecutive_flips_l2267_226775


namespace polynomial_102_l2267_226738

/-- Proving the value of the polynomial expression using the Binomial Theorem -/
theorem polynomial_102 :
  102^4 - 4 * 102^3 + 6 * 102^2 - 4 * 102 + 1 = 100406401 :=
by
  sorry

end polynomial_102_l2267_226738


namespace negation_of_p_l2267_226759

theorem negation_of_p :
  (¬ (∀ x : ℝ, x^3 + 2 < 0)) = ∃ x : ℝ, x^3 + 2 ≥ 0 := 
  by sorry

end negation_of_p_l2267_226759


namespace problem1_problem2_problem3_l2267_226726

theorem problem1 : (x : ℝ) → ((x + 1)^2 = 9 → (x = -4 ∨ x = 2)) :=
by
  intro x
  sorry

theorem problem2 : (x : ℝ) → (x^2 - 12*x - 4 = 0 → (x = 6 + 2*Real.sqrt 10 ∨ x = 6 - 2*Real.sqrt 10)) :=
by
  intro x
  sorry

theorem problem3 : (x : ℝ) → (3*(x - 2)^2 = x*(x - 2) → (x = 2 ∨ x = 3)) :=
by
  intro x
  sorry

end problem1_problem2_problem3_l2267_226726


namespace problem_divisible_by_480_l2267_226797

theorem problem_divisible_by_480 (a : ℕ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) : ∃ k : ℕ, a * (a^2 - 1) * (a^2 - 4) = 480 * k :=
by
  sorry

end problem_divisible_by_480_l2267_226797


namespace rational_abs_neg_l2267_226720

theorem rational_abs_neg (a : ℚ) (h : abs a = -a) : a ≤ 0 :=
by 
  sorry

end rational_abs_neg_l2267_226720


namespace largest_integer_less_than_hundred_with_remainder_five_l2267_226764

theorem largest_integer_less_than_hundred_with_remainder_five (n : ℤ) :
  n < 100 ∧ n % 8 = 5 → n = 93 :=
by
  sorry

end largest_integer_less_than_hundred_with_remainder_five_l2267_226764
