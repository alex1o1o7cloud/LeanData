import Mathlib

namespace gcd_max_digits_l169_169059

theorem gcd_max_digits (a b : ℕ) (h1 : 10^6 ≤ a ∧ a < 10^7) (h2 : 10^6 ≤ b ∧ b < 10^7) (h3 : 10^12 ≤ Nat.lcm a b ∧ Nat.lcm a b < 10^13) : 
  (Nat.gcd a b).digits ≤ 2 := 
sorry

end gcd_max_digits_l169_169059


namespace term_value_in_sequence_l169_169037

theorem term_value_in_sequence (a : ℕ → ℕ) (n : ℕ) (h : ∀ n, a n = n * (n + 2) / 2) (h_val : a n = 220) : n = 20 :=
  sorry

end term_value_in_sequence_l169_169037


namespace bounded_figure_one_center_of_symmetry_l169_169558

theorem bounded_figure_one_center_of_symmetry (F : set ℝ^2) (bounded : is_bounded F)
  (c₁ c₂ : ℝ^2) (h₁ : is_center_of_symmetry F c₁) (h₂ : is_center_of_symmetry F c₂) :
  c₁ = c₂ := 
sorry

end bounded_figure_one_center_of_symmetry_l169_169558


namespace question_eq_answer_l169_169101

theorem question_eq_answer (w x y z k : ℕ) 
  (h : 2^w * 3^x * 5^y * 7^z * 11^k = 2520) : 
  2 * w + 3 * x + 5 * y + 7 * z + 11 * k = 24 :=
sorry

end question_eq_answer_l169_169101


namespace find_a_plus_b_l169_169446

-- Define the given conditions as Lean definitions
noncomputable def f (a b x : ℝ) : ℝ := a * x + b

-- Define the recurrence relation
noncomputable def f_n (a b : ℝ) : ℕ → (ℝ → ℝ)
| 1 => f a b
| (n + 1) => λ x, f a b (f_n a b n x)

-- State the theorem to prove
theorem find_a_plus_b (a b : ℝ) (h_f7 : f_n a b 7 = λ x, 128 * x + 381) : a + b = 5 := 
sorry

end find_a_plus_b_l169_169446


namespace arithmetic_sequence_sum_of_cubes_l169_169914

theorem arithmetic_sequence_sum_of_cubes (x : ℤ) (n : ℕ) (h : n > 5) 
  (hyp : ∑ k in finset.range (n + 1), (x + 2 * k) ^ 3 = 1458) : 
  n = 6 := 
sorry

end arithmetic_sequence_sum_of_cubes_l169_169914


namespace inequality_solution_set_l169_169795

noncomputable def greatest_integer_not_exceeding (x : ℝ) := int.floor x

theorem inequality_solution_set (x : ℝ) : 
  greatest_integer_not_exceeding x ^ 2 - 5 * greatest_integer_not_exceeding x + 6 ≤ 0 ↔ 2 ≤ x ∧ x < 4 := by
  sorry

end inequality_solution_set_l169_169795


namespace inequality_proof_l169_169112

theorem inequality_proof 
  (a b c d : ℝ)
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  (sum_eq : a + b + c + d = 3) : 
  1 / a^2 + 1 / b^2 + 1 / c^2 + 1 / d^2 ≤ 1 / (a * b * c * d)^2 := 
by 
  sorry

end inequality_proof_l169_169112


namespace integer_pairs_count_l169_169321

theorem integer_pairs_count :
  (∃ a b : ℤ, ∃ x y : ℤ, a * x + b * y = 1 ∧ x^2 + y^2 = 50) → 72 := sorry

end integer_pairs_count_l169_169321


namespace sum_positive_factors_30_l169_169548

theorem sum_positive_factors_30 : 
  ∑ d in {d | d ∣ 30}, d = 72 :=
by
  -- Prime factorization based definition of 30
  have h30 : 30 = 2^1 * 3^1 * 5^1 := by norm_num
  -- The result follows by computing the sum of the divisors directly
  sorry

end sum_positive_factors_30_l169_169548


namespace interior_angle_of_regular_heptagon_l169_169205

-- Define the problem statement in Lean
theorem interior_angle_of_regular_heptagon : 
  let n := 7 in (n - 2) * 180 / n = 900 / 7 := 
by 
  let n := 7
  show (n - 2) * 180 / n = 900 / 7
  sorry

end interior_angle_of_regular_heptagon_l169_169205


namespace triangle_ratio_equal_sqrt2_l169_169695

variable {A B C : ℝ} {a b c : ℝ}

-- Define the internal angles and sides of triangle ABC
axiom angle_A : A = RealAngle
axiom angle_B : B = RealAngle
axiom angle_C : C = RealAngle

-- Define the conditions
axiom condition1 : 2 * b * Real.sin (2 * A) = 3 * a * Real.sin B
axiom condition2 : c = 2 * b

-- Define the goal
theorem triangle_ratio_equal_sqrt2 : a / b = Real.sqrt 2 := by
  sorry

end triangle_ratio_equal_sqrt2_l169_169695


namespace chord_division_ratio_l169_169754

theorem chord_division_ratio
  (A B C D M : Point)
  (O : Circle)
  (h_perpendicular : ⊥ AB CD)
  (h_ratio_AB : divides CD AB = 1/5)
  (h_ratio_longer_arc : arc_ratio_longer CD AB = 1/2) :
  divides AB CD = 1/3 := 
by
  sorry

end chord_division_ratio_l169_169754


namespace find_p_l169_169340

noncomputable section
open ProbabilityTheory

theorem find_p
  (X : ℕ → ℝ)
  (h1 : ∀ n, X n = binomial 2 p)
  (h2 : P (set_of (λ x, X x ≥ 1)) = 3 / 4)
  : p = 1 / 2 := by
  sorry

end find_p_l169_169340


namespace range_a_l169_169744

-- Conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + Real.log x
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1 / x

-- Theorem to prove the range of a
theorem range_a (a : ℝ) : (∀ x > 0, f' a x → ℝ → ∃ c : ℝ, a < 0) :=
sorry

end range_a_l169_169744


namespace simplify_expression_l169_169194

theorem simplify_expression : (3 + 3 + 5) / 2 - 1 / 2 = 5 := by
  sorry

end simplify_expression_l169_169194


namespace exist_segment_on_line_l169_169686

noncomputable theory

open EuclideanGeometry

-- Given Definitions
variables (e : Line) (A B : Point) (ϕ : Angle) 

-- Conditions
axiom A_not_on_e : ¬ incident A e
axiom B_not_on_e : ¬ incident B e

-- Proof Statement
theorem exist_segment_on_line (e : Line) (A B : Point) (ϕ : Angle)
  (ha : ¬ incident A e)
  (hb : ¬ incident B e) : 
  ∃ (C D : Point), incident C e ∧ incident D e ∧ 
    ∠ A C D = ϕ ∧ ∠ B C D = ϕ := sorry

end exist_segment_on_line_l169_169686


namespace lilly_fish_l169_169836

-- Define the conditions
def total_fish : ℕ := 18
def rosy_fish : ℕ := 8

-- Statement: Prove that Lilly has 10 fish
theorem lilly_fish (h1 : total_fish = 18) (h2 : rosy_fish = 8) :
  total_fish - rosy_fish = 10 :=
by sorry

end lilly_fish_l169_169836


namespace quadratic_has_imaginary_roots_iff_l169_169905

noncomputable def has_imaginary_roots (λ : ℝ) : Prop :=
  ∀ (x : ℂ), (1 - complex.I) * x^2 + (λ + complex.I) * x + (1 + complex.I * λ) = 0 → x.im ≠ 0

theorem quadratic_has_imaginary_roots_iff (λ : ℝ) :
  has_imaginary_roots λ ↔ λ ≠ 2 :=
sorry

end quadratic_has_imaginary_roots_iff_l169_169905


namespace share_equally_l169_169646

variable (Emani Howard : ℕ)
axiom h1 : Emani = 150
axiom h2 : Emani = Howard + 30

theorem share_equally : (Emani + Howard) / 2 = 135 :=
by sorry

end share_equally_l169_169646


namespace find_a_l169_169026

theorem find_a
  (f : ℝ → ℝ)
  (a : ℝ)
  (h1 : f = λ x, a / x^2)
  (h2 : let f2 := f 2 in let f' := deriv f in f2 = a / 4 ∧ f' 2 = -a / 4)
  (h3 : let f2 := f 2 in ∃ (x1 y1 : ℝ), x1 = 1 ∧ y1 = 2 ∧ (f' 2) = (y1 - f2) / (x1 - 2)) :
  a = 4 :=
by
  sorry

end find_a_l169_169026


namespace neg_p_sufficient_but_not_necessary_for_q_l169_169004

variable (x : ℝ) (p : Prop) (q : Prop)

def p : Prop := x ≤ 1
def q : Prop := 1/x < 1

theorem neg_p_sufficient_but_not_necessary_for_q : (¬p → q) ∧ ¬(q → ¬p) :=
by sorry

end neg_p_sufficient_but_not_necessary_for_q_l169_169004


namespace nested_radical_solution_l169_169654

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l169_169654


namespace distance_between_points_is_sqrt2_l169_169892

def intersection_points_distance : ℝ :=
  let eq1 (x y : ℝ) := x^2 + y = 10
  let eq2 (x y : ℝ) := x + y = 10
  let point1 := (0, 10)
  let point2 := (1, 9)
  let distance := Real.sqrt ((point2.1 - point1.1) ^ 2 + (point2.2 - point1.2) ^ 2)
  distance

theorem distance_between_points_is_sqrt2 : intersection_points_distance = Real.sqrt 2 := by
  sorry

end distance_between_points_is_sqrt2_l169_169892


namespace regular_heptagon_interior_angle_l169_169198

theorem regular_heptagon_interior_angle :
  ∀ (n : ℕ), n = 7 → (∑ i in finset.range n, 180 / n) = 128.57 :=
  by
  intros n hn
  rw hn
  sorry

end regular_heptagon_interior_angle_l169_169198


namespace proof_expr_l169_169628

noncomputable def simplify_expr := 
  (0.002 : ℝ) ^ (-1 / 2) - 10 * (Real.sqrt 5 - 2) ^ (-1) + (Real.sqrt 2 - Real.sqrt 3) ^ 0

theorem proof_expr : simplify_expr = -19 := by
  sorry

end proof_expr_l169_169628


namespace regular_heptagon_interior_angle_l169_169202

theorem regular_heptagon_interior_angle :
  ∀ (S : Type) [decidable_instance S] [fintype S], ∀ (polygon : set S), is_regular polygon → card polygon = 7 → 
    (sum_of_interior_angles polygon / 7 = 128.57) :=
by
  intros S dec inst polygon h_reg h_card
  sorry

end regular_heptagon_interior_angle_l169_169202


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169971

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169971


namespace correct_statement_D_for_function_y_eq_2x_minus_1_l169_169328

theorem correct_statement_D_for_function_y_eq_2x_minus_1 :
  ∀ (x y : ℝ),
  y = 2 * x - 1 →
  ¬ (x < 0 ∧ y > 0) :=
begin
  sorry
end

end correct_statement_D_for_function_y_eq_2x_minus_1_l169_169328


namespace one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169543

theorem one_hundred_fiftieth_digit_of_3_div_11_is_7 :
  let decimal_repetition := "27"
  let length := 2
  150 % length = 0 →
  (decimal_repetition[1] = '7')
: sorry

end one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169543


namespace greatest_sum_of_consecutive_integers_product_less_500_l169_169997

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l169_169997


namespace cos_minus_sin_value_l169_169352

theorem cos_minus_sin_value (θ : ℝ) (h1 : θ ∈ Ioo (3 * Real.pi / 4) Real.pi) (h2 : sin θ * cos θ = -Real.sqrt 3 / 2) :
  cos θ - sin θ = -Real.sqrt (1 + Real.sqrt 3) :=
by
  sorry

end cos_minus_sin_value_l169_169352


namespace area_of_circle_l169_169580

-- Given the circumference of a circle
def circumference : ℝ := 30 * Real.pi

-- Assuming the radius definition according to the circumference
def radius (C : ℝ) : ℝ := C / (2 * Real.pi)

-- Define the area of the circle in terms of the radius
def area (r : ℝ) : ℝ := Real.pi * r^2

-- The statement to be proved
theorem area_of_circle : area (radius circumference) = 225 * Real.pi := sorry

end area_of_circle_l169_169580


namespace find_f_prime_2_l169_169354

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * f' 2

theorem find_f_prime_2 (f : ℝ → ℝ) (f' : ℝ → ℝ) (h_deriv : ∀ x, deriv f x = f' x)
  (h_eq : ∀ x, f x = x^2 + 3 * x * f' 2) : f' 2 = -2 :=
by
  sorry

end find_f_prime_2_l169_169354


namespace cos_x_plus_3y_values_l169_169084

theorem cos_x_plus_3y_values (x y : ℝ) (h1 : (cos (3 * x)) / ((2 * cos (2 * x) - 1) * cos y) = 2 / 5 + (cos (x + y)) ^ 2)
                               (h2 : (sin (3 * x)) / ((2 * cos (2 * x) + 1) * sin y) = 3 / 5 + (sin (x + y)) ^ 2) :
  ∃ v₁ v₂ : ℝ, v₁ ≠ v₂ ∧ (cos (x + 3 * y) = v₁ ∨ cos (x + 3 * y) = v₂) ∧ (v₁ = -1 ∨ v₁ = -1/5) ∧ (v₂ = -1 ∨ v₂ = -1/5) :=
by
  sorry

end cos_x_plus_3y_values_l169_169084


namespace Andy_more_white_socks_than_black_l169_169615

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end Andy_more_white_socks_than_black_l169_169615


namespace train_crosses_pole_in_12_seconds_l169_169606

noncomputable def time_to_cross_pole (speed train_length : ℕ) : ℕ := 
  train_length / speed

theorem train_crosses_pole_in_12_seconds 
  (speed : ℕ) (platform_length : ℕ) (time_to_cross_platform : ℕ) (train_crossing_time : ℕ)
  (h_speed : speed = 10) 
  (h_platform_length : platform_length = 320) 
  (h_time_to_cross_platform : time_to_cross_platform = 44) 
  (h_train_crossing_time : train_crossing_time = 12) :
  time_to_cross_pole speed 120 = train_crossing_time := 
by 
  sorry

end train_crosses_pole_in_12_seconds_l169_169606


namespace extremum_interval_a_l169_169833

theorem extremum_interval_a (a : ℝ) :
  (∃ x ∈ Ioo (1/e : ℝ) e, deriv (λ x : ℝ, x + a * log x) x = 0) →
  -e < a ∧ a < -1/e :=
by
  have h₁ : ∀ x > 0, deriv (λ x : ℝ, x + a * log x) x = 1 + a / x := sorry
  have h₂ : deriv (λ x : ℝ, x + a * log x) (1/e) * deriv (λ x : ℝ, x + a * log x) e < 0 := sorry
  exact sorry

end extremum_interval_a_l169_169833


namespace basis_vector_sets_problem_l169_169380

theorem basis_vector_sets_problem
  (e1 e2 : ℝ × ℝ)
  (h_basis : ¬ collinear e1 e2) :
  let set1 := (e1, (e1.1 + e2.1, e1.2 + e2.2))
  let set2 := ((e1.1 - 2 * e2.1, e1.2 - 2 * e2.2), (4 * e2.1 - 2 * e1.1, 4 * e2.2 - 2 * e1.2))
  let set3 := ((e1.1 + e2.1, e1.2 + e2.2), (e1.1 - e2.1, e1.2 - e2.2))
  let set4 := ((2 * e1.1 - e2.1, 2 * e1.2 - e2.2), (1 / 2 * e2.1 - e1.1, 1 / 2 * e2.2 - e1.2)) in
  [set1, set2, set3, set4].count (λ set, collinear set.1 set.2) = 2 :=
by
  sorry

end basis_vector_sets_problem_l169_169380


namespace range_values_PM_PN_l169_169096

open_locale real

def ellipse (P : ℝ × ℝ) : Prop :=
  (P.1 / 5) ^ 2 + (P.2 / 4) ^ 2 = 1

def circle1 (M : ℝ × ℝ) : Prop :=
  (M.1 + 3) ^ 2 + M.2 ^ 2 = 1

def circle2 (N : ℝ × ℝ) : Prop :=
  (N.1 - 3) ^ 2 + N.2 ^ 2 = 4

def distance (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1) ^ 2 + (A.2 - B.2) ^ 2)

def range_PM_PN (P M N : ℝ × ℝ) : Prop :=
  7 ≤ distance P M + distance P N ∧ distance P M + distance P N ≤ 13

theorem range_values_PM_PN (P M N : ℝ × ℝ) (hP : ellipse P) (hM : circle1 M) (hN : circle2 N) :
  range_PM_PN P M N :=
sorry

end range_values_PM_PN_l169_169096


namespace largest_number_l169_169497

theorem largest_number (a b c : ℕ) (h1 : a = 1) (h2 : b = 7) (h3 : c = 0) : 
  nat.lt (nat.le.digits (b :: a :: c :: [])) 710 = 710 := 
sorry

end largest_number_l169_169497


namespace box_weight_no_apples_l169_169576

variable (initialWeight : ℕ) (halfWeight : ℕ) (totalWeight : ℕ)
variable (boxWeight : ℕ)

-- Given conditions
axiom initialWeight_def : initialWeight = 9
axiom halfWeight_def : halfWeight = 5
axiom appleWeight_consistent : ∃ w : ℕ, ∀ n : ℕ, n * w = totalWeight

-- Question: How many kilograms does the empty box weigh?
theorem box_weight_no_apples : (initialWeight - totalWeight) = boxWeight :=
by
  -- The proof steps are omitted as indicated by the 'sorry' placeholder.
  sorry

end box_weight_no_apples_l169_169576


namespace interior_edges_sum_l169_169255

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l169_169255


namespace min_max_value_l169_169313

def min_max_expression : ℝ :=
  2

theorem min_max_value : ∀ (y : ℝ), 
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 1 ∧ (|x^2 + x * y + cos y| ≤ min_max_expression) := 
sorry

end min_max_value_l169_169313


namespace mixture_weight_l169_169283

theorem mixture_weight
  (weight_A weight_B weight_C weight_D weight_E : ℕ)
  (ratio_A ratio_B ratio_C ratio_D ratio_E total_volume : ℕ)
  (total_weight_in_kg : ℝ) :
  weight_A = 950 →
  weight_B = 850 →
  weight_C = 900 →
  weight_D = 920 →
  weight_E = 875 →
  ratio_A = 7 →
  ratio_B = 4 →
  ratio_C = 2 →
  ratio_D = 3 →
  ratio_E = 5 →
  total_volume = 21 →
  total_weight_in_kg = (7 * 950 + 4 * 850 + 2 * 900 + 3 * 920 + 5 * 875) / 1000 →
  total_weight_in_kg = 18.985 :=
by
  intros hA hB hC hD hE hrA hrB hrC hrD hrE htv htw
  rw [hA, hB, hC, hD, hE, hrA, hrB, hrC, hrD, hrE, htv] at htw
  sorry

end mixture_weight_l169_169283


namespace polynomial_factorization_l169_169830

noncomputable def p (x : ℚ) (b c : ℚ) : ℚ := x^2 + b * x + c

theorem polynomial_factorization (b c : ℤ) (h1 : ∃ (p : polynomial ℚ), p = polynomial.X ^ 2 + polynomial.C (b : ℚ) * polynomial.X + polynomial.C (c : ℚ)
  ∧ (polynomial.X ^ 4 + 8 * polynomial.X ^ 3 + 6 * polynomial.X ^ 2 + polynomial.C 36).is_factor p
  ∧ (3 * polynomial.X ^ 4 + 6 * polynomial.X ^ 3 + 5 * polynomial.X ^ 2 + 42 * polynomial.X + polynomial.C 15).is_factor p) :
  p 5 b c = 11 :=
sorry

end polynomial_factorization_l169_169830


namespace positive_integer_pairs_divisibility_l169_169291

theorem positive_integer_pairs_divisibility (a b : ℕ) (h : a * b^2 + b + 7 ∣ a^2 * b + a + b) :
  (a = 11 ∧ b = 1) ∨ (a = 49 ∧ b = 1) ∨ ∃ k : ℕ, k > 0 ∧ a = 7 * k^2 ∧ b = 7 * k :=
sorry

end positive_integer_pairs_divisibility_l169_169291


namespace days_required_proof_l169_169736

variable (y : ℕ)

-- Given conditions
def cows := y
def cans := y + 2
def days := y + 3

-- Question equivalent to finding days_required
def days_required_to_produce_cans (cows' cans' : ℕ) : ℕ :=
  (cows' * days * cans') / (cows * cans)

theorem days_required_proof :
  days_required_to_produce_cans (y + 4) (y + 7) = (y * (y + 3) * (y + 7)) / ((y + 4) * (y + 2)) := by
  sorry

end days_required_proof_l169_169736


namespace sqrt_triangle_inequality_l169_169191

theorem sqrt_triangle_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (habc1 : a + b > c) (habc2 : a + c > b) (habc3 : b + c > a) :
  (sqrt a + sqrt b > sqrt c) ∧ (sqrt a + sqrt c > sqrt b) ∧ (sqrt b + sqrt c > sqrt a) := 
by
  sorry

end sqrt_triangle_inequality_l169_169191


namespace digit_150_of_3_div_11_l169_169535

theorem digit_150_of_3_div_11 : 
  let n := 150
  let digits := "27"
  let cycle_length := 2
  let digit_150 := digits[(n % cycle_length)]
  in digit_150 = '7' :=
by {
  sorry
}

end digit_150_of_3_div_11_l169_169535


namespace color_of_1543_over_275_l169_169930

-- Labels for colors
inductive Color
| white
| black

-- Assume rational numbers with color assignment
def RationalCol (x : ℚ) : Color

-- Conditions
axiom color_one_white : RationalCol 1 = Color.white
axiom color_never_same : ∀ x : ℚ, RationalCol x ≠ RationalCol (x + 1)
axiom color_reciprocal_same : ∀ x : ℚ, RationalCol x = RationalCol (1 / x)

theorem color_of_1543_over_275 : RationalCol (1543 / 275) = Color.white :=
by
sory

end color_of_1543_over_275_l169_169930


namespace range_of_phi_l169_169031

noncomputable def f (x φ : ℝ) : ℝ :=
  2 * Real.cos (3 * x + φ) + 3

theorem range_of_phi
  (φ : ℝ)
  (h₁ : ∀ x ∈ Ioo (-Real.pi / 6) (Real.pi / 12), f x φ > 3)
  (h₂ : |φ| ≤ Real.pi / 2) :
  0 ≤ φ ∧ φ ≤ Real.pi / 4 :=
sorry

end range_of_phi_l169_169031


namespace standard_pairs_parity_l169_169568

theorem standard_pairs_parity (m n : ℕ) (h_m : m ≥ 3) (h_n : n ≥ 3) (color : ℕ × ℕ → bool) :
  let S := count_standard_pairs m n color in
  even S ↔ even (count_blue_edges m n color) :=
sorry

def count_standard_pairs (m n : ℕ) (color : ℕ × ℕ → bool) : ℕ :=
-- A definition for counting standard pairs (adjacent squares with different colors)
sorry

def count_blue_edges (m n : ℕ) (color : ℕ × ℕ → bool) : ℕ :=
-- A definition for counting blue squares in Category 2 (edges excluding corners)
sorry

-- Definition to check if a number is even
def even (n : ℕ) : Prop := n % 2 = 0

end standard_pairs_parity_l169_169568


namespace necessary_and_sufficient_l169_169819

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.cos x + b * Real.sin x

-- State the theorem
theorem necessary_and_sufficient (a b : ℝ) : 
  (∀ x, f a b x = f a b (-x)) ↔ b = 0 := 
sorry

end necessary_and_sufficient_l169_169819


namespace area_of_square_l169_169121

-- Defining the points A and B as given in the conditions.
def A : ℝ × ℝ := (1, 2)
def B : ℝ × ℝ := (4, 6)

-- Theorem statement: proving that the area of the square given the endpoints A and B is 12.5.
theorem area_of_square : 
  ∀ (A B : ℝ × ℝ),
  A = (1, 2) → B = (4, 6) → 
  ∃ (area : ℝ), area = 12.5 := 
by
  intros A B hA hB
  sorry

end area_of_square_l169_169121


namespace fraction_is_correct_l169_169712

def f (x : ℕ) : ℕ := 3 * x + 2
def g (x : ℕ) : ℕ := 2 * x - 3

theorem fraction_is_correct : (f (g (f 3))) / (g (f (g 3))) = 59 / 19 :=
by
  sorry

end fraction_is_correct_l169_169712


namespace find_AB_length_l169_169063

variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]
  [T_triangleABC : Triangle A B C] [B_right_angle : RightAngle B]
  (AC_length : dist A C = 4) (BC_length : dist B C = 3)

theorem find_AB_length :
  dist A B = Real.sqrt 7 :=
by
  sorry

end find_AB_length_l169_169063


namespace arithmetic_sequence_a7_value_l169_169453

noncomputable def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_a7_value
  (a : ℕ → ℝ) (d : ℝ)
  (h₀ : d ≠ 0)
  (h₁ : a 1 ^ 2 + a 2 ^ 2 = a 3 ^ 2 + a 4 ^ 2)
  (h₂ : ∑ i in Finset.range 5, a (i + 1) = 5) :
  a 7 = 9 :=
by
  sorry

end arithmetic_sequence_a7_value_l169_169453


namespace vector_difference_magnitude_l169_169804

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def unit_vectors (a b : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 1

theorem vector_difference_magnitude (a b : V)
  (h : unit_vectors a b) : ∥a - b∥ = real.sqrt 3 :=
by sorry

end vector_difference_magnitude_l169_169804


namespace first_book_pages_l169_169619

variable (x : ℕ) -- representing the total number of pages in the first book

-- Define the conditions given in the problem
def needs_to_finish (total_pages : ℕ) : Prop :=
  total_pages = 800

def has_read_80_percent (pages_read : ℕ) : Prop :=
  pages_read = 0.80 * x

def has_read_fifth_of_1000 (pages_read : ℕ) : Prop :=
  pages_read = 1 / 5 * 1000

def needs_to_read_more (pages_left : ℕ) : Prop :=
  pages_left = 200

-- The main theorem to prove
theorem first_book_pages (x : ℕ) (total_pages : ℕ) (pages_read_1 : ℕ) (pages_read_2 : ℕ) (pages_left : ℕ) :
  needs_to_finish total_pages →
  has_read_80_percent x pages_read_1 →
  has_read_fifth_of_1000 pages_read_2 →
  needs_to_read_more pages_left →
  pages_read_1 + pages_read_2 = total_pages - pages_left →
  x = 500 :=
by
  sorry

end first_book_pages_l169_169619


namespace hypotenuse_length_l169_169212

-- Let a and b be the lengths of the non-hypotenuse sides of a right triangle.
-- We are given that a = 6 and b = 8, and we need to prove that the hypotenuse c is 10.
theorem hypotenuse_length (a b c : ℕ) (h₁ : a = 6) (h₂ : b = 8) (h₃ : c ^ 2 = a ^ 2 + b ^ 2) : c = 10 :=
by
  -- The proof goes here.
  sorry

end hypotenuse_length_l169_169212


namespace cleaning_time_l169_169557

def john_cleaning_rate : ℝ := 1 / 6
def nick_cleaning_rate (t : ℝ) : ℝ := 1 / t
def combined_cleaning_rate (r1 r2 : ℝ) : ℝ := r1 + r2
def time_to_clean_house (combined_rate : ℝ) : ℝ := 1 / combined_rate

theorem cleaning_time (t : ℝ) (h1 : 1 / 3 * t = 3) : 
  time_to_clean_house (combined_cleaning_rate john_cleaning_rate (nick_cleaning_rate t)) = 3.6 :=
by
  -- We put the proof here
  sorry

end cleaning_time_l169_169557


namespace range_of_independent_variable_l169_169167

theorem range_of_independent_variable (x : ℝ) (hx : 1 - 2 * x ≥ 0) : x ≤ 0.5 :=
sorry

end range_of_independent_variable_l169_169167


namespace modulus_difference_l169_169356

def z1 : Complex := 1 + 2 * Complex.I
def z2 : Complex := 2 + Complex.I

theorem modulus_difference :
  Complex.abs (z2 - z1) = Real.sqrt 2 := by sorry

end modulus_difference_l169_169356


namespace number_of_schools_l169_169308

-- Define the conditions as parameters and assumptions
structure CityContest (n : ℕ) :=
  (students_per_school : ℕ := 4)
  (total_students : ℕ := students_per_school * n)
  (andrea_percentile : ℕ := 75)
  (andrea_highest_team : Prop)
  (beth_rank : ℕ := 20)
  (carla_rank : ℕ := 47)
  (david_rank : ℕ := 78)
  (andrea_position : ℕ)
  (h3 : andrea_position = (3 * total_students + 1) / 4)
  (h4 : 3 * n > 78)

-- Define the main theorem statement
theorem number_of_schools (n : ℕ) (contest : CityContest n) (h5 : contest.andrea_highest_team) : n = 20 :=
  by {
    -- You would insert the detailed proof of the theorem based on the conditions here.
    sorry
  }

end number_of_schools_l169_169308


namespace find_k_value_l169_169719

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end find_k_value_l169_169719


namespace sum_of_interior_edges_l169_169254

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l169_169254


namespace problem_conditions_l169_169746

open Set

theorem problem_conditions (a b : ℝ) (h : {1, a, b / a} = {0, a^2, a + b}) : a^2015 + b^2016 = -1 :=
by
  sorry

end problem_conditions_l169_169746


namespace does_not_necessarily_hold_l169_169735

variables {a b c : ℝ}
hypothesis (h1 : c < b)
hypothesis (h2 : b < a)
hypothesis (h3 : a * c < 0)

theorem does_not_necessarily_hold : ¬(c * b^2 < a * b^2) :=
sorry

end does_not_necessarily_hold_l169_169735


namespace even_x_satisfies_remainder_l169_169499

theorem even_x_satisfies_remainder 
  (z : ℕ) 
  (hz : z % 4 = 0) : 
  ∃ (x : ℕ), x % 2 = 0 ∧ (z * (2 + x + z) + 3) % 2 = 1 := 
by
  sorry

end even_x_satisfies_remainder_l169_169499


namespace sum_k_div_3_pow_k_l169_169307

theorem sum_k_div_3_pow_k (n : ℕ) (h : n ≥ 1) :
  (∑ k in Finset.range n.succ, k * 3^(-k) : ℝ) = 3/2 - 1/(2 * 3^n) :=
by
  sorry

end sum_k_div_3_pow_k_l169_169307


namespace vector_difference_magnitude_l169_169807

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def unit_vectors (a b : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 1

theorem vector_difference_magnitude (a b : V)
  (h : unit_vectors a b) : ∥a - b∥ = real.sqrt 3 :=
by sorry

end vector_difference_magnitude_l169_169807


namespace remainder_conditions_l169_169553

variables (P D R : Polynomial ℤ) (m b : ℤ)

-- Let P(x) = x^5 - 7x^4 + 21x^3 - 28x^2 + 19x - 6
noncomputable def P := X^5 - 7*X^4 + 21*X^3 - 28*X^2 + 19*X - 6

-- Let D(x) = x^2 - 3x + m
def D := X^2 - 3*X + C m

-- Given that the remainder R(x) = 2x + b when P(x) is divided by D(x)
def R := 2*X + C b

-- Prove m = 19 and b = 100
theorem remainder_conditions :
  ((∀ x : Polynomial ℤ, (P = D * x + R)) ∧ (degree R < degree D)) →
  m = 19 ∧ b = 100 :=
by
  sorry

end remainder_conditions_l169_169553


namespace geometric_sequence_ratio_l169_169404

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : 6 * a 7 = (a 8 + a 9) / 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = a 1 * (1 - q^n) / (1 - q)) :
  S 6 / S 3 = 28 :=
by
  -- The proof goes here
  sorry

end geometric_sequence_ratio_l169_169404


namespace attended_college_percentage_l169_169222

noncomputable def percentage_of_class_attended_college : ℕ :=
let boys := 160
let girls := 200
let boys_attended := 0.75 * boys
let girls_not_attended := 0.40 * girls
let girls_attended := girls - girls_not_attended
let total_students := boys + girls
let students_attended := boys_attended + girls_attended
(students_attended / total_students) * 100

theorem attended_college_percentage :
  percentage_of_class_attended_college = 66.67 :=
by 
  sorry

end attended_college_percentage_l169_169222


namespace larger_number_of_two_l169_169377

theorem larger_number_of_two (x y : ℝ) (h1 : x * y = 30) (h2 : x + y = 13) : max x y = 10 :=
sorry

end larger_number_of_two_l169_169377


namespace probability_loses_first_two_wins_third_probability_wins_exactly_three_mean_and_variance_binomial_l169_169574

noncomputable def problem_conditions := 
  ∀ (n : ℕ) (p : ℚ), n = 6 ∧ p = 1 / 3

theorem probability_loses_first_two_wins_third :
  problem_conditions → 
  (1 - 1/3)^2 * (1/3) = 4/27 := sorry

theorem probability_wins_exactly_three :
  problem_conditions → 
  ∑ C(6, k) * (1/3)^k * (2/3)^(6-k) = 160/729 := sorry

theorem mean_and_variance_binomial :
  problem_conditions →
  let X : ℕ → ℚ := λ k => C(6, k) * (1/3)^k * (2/3)^(6-k) in
  (∑ k in range 7, k * X k = 2) ∧ (∑ k in range 7, (k - 2)^2 * X k = 4/3) := sorry

end probability_loses_first_two_wins_third_probability_wins_exactly_three_mean_and_variance_binomial_l169_169574


namespace triangle_area_ratio_of_trapezoid_l169_169517

theorem triangle_area_ratio_of_trapezoid 
  {A B C D E B1 : Type} 
  (ABC : Triangle A B C) 
  (BB1 : Altitude B B1) 
  (DE_parallel_AC : Line DE ∥ Line AC) 
  (midpoint_B1 : Midpoint B1 BB1) :
  ratio (area (Triangle B D E)) (area (Trapezoid A D E C)) = 1 / 3 := 
sorry

end triangle_area_ratio_of_trapezoid_l169_169517


namespace insert_eights_composite_l169_169164

theorem insert_eights_composite (n : ℕ) : 
  let N_n := 200 * 10^n + (88 * 10^n - 88) / 9 + 21
  in ¬Prime N_n :=
sorry

end insert_eights_composite_l169_169164


namespace greatest_sum_of_consecutive_integers_l169_169962

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169962


namespace ways_to_distribute_books_into_bags_l169_169623

theorem ways_to_distribute_books_into_bags : 
  let books := 5
  let bags := 4
  ∃ (ways : ℕ), ways = 41 := 
sorry

end ways_to_distribute_books_into_bags_l169_169623


namespace sum_sequence_identity_l169_169366

-- Definitions based on conditions
def a (n : ℕ) : ℕ := n * 2^(n-1)

def S (n : ℕ) : ℕ := ∑ i in range n, a (i+1)

-- Prove the main statement
theorem sum_sequence_identity (n : ℕ) : S n = (n-1) * 2^(n+1) + 2 :=
by
  sorry

end sum_sequence_identity_l169_169366


namespace proof_problems_l169_169637

def otimes (a b : ℝ) : ℝ :=
  a * (1 - b)

theorem proof_problems :
  (otimes 2 (-2) = 6) ∧
  ¬ (∀ (a b : ℝ), otimes a b = otimes b a) ∧
  (∀ (a b : ℝ), a + b = 0 → otimes a a + otimes b b = 2 * a * b) ∧
  ¬ (∀ (a b : ℝ), otimes a b = 0 → a = 0) :=
by
  sorry
 
end proof_problems_l169_169637


namespace population_increase_leap_year_l169_169413

theorem population_increase_leap_year :
  (∀ (b_d: ℕ) (d_d: ℕ) (days: ℕ), -- declaring parameters for births/day, deaths/day, and days/year
    (b_d = 24 / 6) ∧  -- Condition for births per day
    (d_d = 2) ∧       -- Condition for deaths per day
    (days = 366) →
    let net_increase := b_d - d_d in
    let annual_increase := net_increase * days in
    annual_increase.round_nearest 100 = 700) := 
by
  sorry

end population_increase_leap_year_l169_169413


namespace julios_spending_on_limes_l169_169778

theorem julios_spending_on_limes 
    (days : ℕ) (lime_juice_per_day : ℕ) (lime_juice_per_lime : ℕ) (limes_per_dollar : ℕ) 
    (total_spending : ℝ) 
    (h1 : days = 30) 
    (h2 : lime_juice_per_day = 1) 
    (h3 : lime_juice_per_lime = 2) 
    (h4 : limes_per_dollar = 3) 
    (h5 : total_spending = 5) :
    let lime_juice_needed := days * lime_juice_per_day,
        total_limes := lime_juice_needed / lime_juice_per_lime,
        cost := (total_limes / limes_per_dollar : ℕ) in
    (cost : ℝ) = total_spending := 
by 
    sorry

end julios_spending_on_limes_l169_169778


namespace parabola_directrix_l169_169889

theorem parabola_directrix {x y : ℝ} (h : y^2 = 6 * x) : x = -3 / 2 := 
sorry

end parabola_directrix_l169_169889


namespace triangle_right_count_l169_169915

noncomputable def is_right_triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  (A = Real.pi / 2 ∨ B = Real.pi / 2 ∨ C = Real.pi / 2)

theorem triangle_right_count :
  ∀ (a b c : ℝ) (A B C : ℝ),
  (A = B - C) →
  (3 * A = 4 * B ∧ 4 * B = 5 * C) →
  (a^2 = (b + c) * (b - c)) →
  (5 * a = 12 * b ∧ 12 * b = 13 * c) →
  (nat.card {t : ℕ // is_right_triangle a b c A B C}) = 3 :=
by 
  intros a b c A B C h1 h2 h3 h4,
  sorry

end triangle_right_count_l169_169915


namespace negation_of_proposition_l169_169365

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, 1 < x → (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) > 4)) ↔
  (∃ x : ℝ, 1 < x ∧ (Real.log x / Real.log 2) + 4 * (Real.log 2 / Real.log x) ≤ 4) :=
sorry

end negation_of_proposition_l169_169365


namespace triangle_ratio_sine_equality_l169_169462

open Real

noncomputable def triangle (A B C : Type*) := ∀ (A B C), is_triangle A B C
noncomputable def point_on_side (A B : Type*) (P : Type*) := ∃ (P : Type*) (A B P : Type*), is_on_side P A B

theorem triangle_ratio_sine_equality 
  (A B C : Type*) 
  (triangle_ABC : triangle A B C)
  (A1 : Type*) (on_A1 : point_on_side B C A1)
  (B1 : Type*) (on_B1 : point_on_side C A B1)
  (C1 : Type*) (on_C1 : point_on_side A B C1) :
  (dist A C1 / dist C1 B) * (dist B A1 / dist A1 C) * (dist C B1 / dist B1 A) = 
  (sin (angle A C C1) / sin (angle C1 C B)) * (sin (angle B A A1) / sin (angle A1 A C)) * (sin (angle C B B1) / sin (angle B1 B A)) :=
sorry

end triangle_ratio_sine_equality_l169_169462


namespace intersection_of_M_and_N_l169_169454

-- Define the set M
def M : Set ℝ := {x | 2^(x-1) < 1}

-- Define the set N
def N : Set ℝ := {x | log (1/2) x < 0}

-- Theorem stating the intersection of M and N
theorem intersection_of_M_and_N : (M ∩ N) = {x | (1/2 : ℝ) < x ∧ x < 1} := 
by
  sorry

end intersection_of_M_and_N_l169_169454


namespace abs_diff_squares_l169_169952

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169952


namespace sum_of_interior_edges_l169_169259

theorem sum_of_interior_edges {f : ℝ} {w : ℝ} (h_frame_area : f = 32) (h_outer_edge : w = 7) (h_frame_width : 2) :
  let i_length := w - 2 * h_frame_width in
  let i_other_length := (f - (w * (w  - 2 * h_frame_width))) / (w  + 2 * h_frame_width) in
  i_length + i_other_length + i_length + i_other_length = 8 :=
by
  let i_length := w - 2 * 2
  let i_other_length := (32 - (i_length * w)) / (w  + 2 * 2)
  let sum := i_length + i_other_length + i_length + i_other_length
  have h_sum : sum = 8, by sorry
  exact h_sum

end sum_of_interior_edges_l169_169259


namespace dot_product_zero_l169_169353

variables (a b : EuclideanSpace ℝ (Fin 2))
variable (angle_ab : Real.Angle.rad = π/4)
variable (norm_a : ∥a∥ = 2)
variable (norm_b : ∥b∥ = 2)

theorem dot_product_zero :
    a • (a - (real.sqrt 2) • b) = 0 :=
by
  sorry

end dot_product_zero_l169_169353


namespace find_positive_number_l169_169737

variable (m : ℝ)

def root1 : ℝ := 2 * m - 1
def root2 : ℝ := 2 - m

def pos_number : ℝ := root1^2

theorem find_positive_number (h : root1 + root2 = 0) : pos_number = 9 := 
by
  sorry

end find_positive_number_l169_169737


namespace vectors_sum_bounded_l169_169755

variable {α : Type*} [NormedAddCommGroup α] [NormedSpace ℝ α]

noncomputable def vectors_indexed_correctly (n : ℕ) (vectors : Fin n → α) (h_len : ∀ i, ∥vectors i∥ = 1)
  (h_sum : ∑ i, vectors i = 0) : Prop :=
  ∀ k : ℕ, k ≤ n → ∥∑ i in Finset.range k, vectors ⟨i, Fin.is_lt i k⟩∥ ≤ 2

theorem vectors_sum_bounded (n : ℕ) (vectors : Fin n → α)
  (h_len : ∀ i, ∥vectors i∥ = 1)
  (h_sum : ∑ i, vectors i = 0) :
  vectors_indexed_correctly n vectors h_len h_sum :=
sorry

end vectors_sum_bounded_l169_169755


namespace DE_length_l169_169620

theorem DE_length (A B C D E : Type) [metric_space B] [metric_space D]
  (ABC_triangle : triangle A B C)
  (AC_eq_BC : dist A C = dist B C)
  (angle_ACB_right : ∠ A C B = 90)
  (AD_perpendicular_CE : ∃ (D' : Type), ⟦ AD ∥ CE ⟧ ∧ ⟦ HD = 8 ⟧)
  (BE_perpendicular_CE : ∃ (E' : Type), ⟦ BE ∥ CE ⟧ ∧ ⟦ HE = 3 ⟧) :
  dist D E = 5 :=
by
  sorry

end DE_length_l169_169620


namespace square_divided_perimeter_l169_169172

/-- Given a square with vertices (-2a, -2a), (2a, -2a), (-2a, 2a), (2a, 2a), and a line y = 3/4 * x,
prove that the perimeter of one of the parts formed by this line, divided by 2a, is equal to 3. -/
theorem square_divided_perimeter (a : ℝ) (h : 0 < a) :
    let P1 := (-2 * a, -2 * a)
    let P2 := (2 * a, -2 * a)
    let P3 := (-2 * a, 2 * a)
    let P4 := (2 * a, 2 * a)
    let l (x : ℝ) := 3 * x / 4
    let I1 := (2 * a, 3 * a / 2)
    let I2 := (-2 * a, -3 * a / 2)
    sqrt ((4 * a) ^ 2 + (3 * a) ^ 2) + 2 * (a / 2) / (2 * a) = 3 := sorry

end square_divided_perimeter_l169_169172


namespace evaluate_expression_l169_169653

theorem evaluate_expression : 
  (Int.floor ((Real.ceil ((11/6 : Real) ^ 2)) + 19/5)) = 7 := 
by
  sorry

end evaluate_expression_l169_169653


namespace height_relation_l169_169926

namespace CylinderProof

variables {r1 r2 h1 h2 : ℝ}

-- Define the conditions
def equal_volume (r1 r2 h1 h2 : ℝ) : Prop :=
  π * r1^2 * h1 = π * r2^2 * h2

def radius_relation (r1 r2 : ℝ) : Prop :=
  r2 = 1.2 * r1

-- Prove the desired relationship
theorem height_relation (r1 r2 h1 h2 : ℝ) (hv : equal_volume r1 r2 h1 h2)
  (hr : radius_relation r1 r2) : h1 = 1.44 * h2 := by
  sorry

end CylinderProof

end height_relation_l169_169926


namespace length_of_YZ_l169_169230

theorem length_of_YZ
  (PQ : ℝ) (XY : ℝ) (QR : ℝ) (YZ : ℝ)
  (h1 : Δ PQR ≈ Δ XYZ)
  (h2 : PQ = 12)
  (h3 : QR = 7)
  (h4 : XY = 4) :
  YZ = 7 / 3 :=
by
  -- skip the proof
  sorry

end length_of_YZ_l169_169230


namespace evaluate_expression_l169_169660

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l169_169660


namespace jerry_showers_l169_169302

variable (water_allowance : ℕ) (drinking_cooking : ℕ) (water_per_shower : ℕ) (pool_length : ℕ) 
  (pool_width : ℕ) (pool_height : ℕ) (gallons_per_cubic_foot : ℕ)

/-- Jerry can take 15 showers in July given the conditions. -/
theorem jerry_showers :
  water_allowance = 1000 →
  drinking_cooking = 100 →
  water_per_shower = 20 →
  pool_length = 10 →
  pool_width = 10 →
  pool_height = 6 →
  gallons_per_cubic_foot = 1 →
  (water_allowance - (drinking_cooking + (pool_length * pool_width * pool_height) * gallons_per_cubic_foot)) / water_per_shower = 15 :=
by
  intros h_water_allowance h_drinking_cooking h_water_per_shower h_pool_length h_pool_width h_pool_height h_gallons_per_cubic_foot
  sorry

end jerry_showers_l169_169302


namespace largest_positive_integer_n_l169_169668

def alpha_condition (n : ℕ) (alphas : Fin n → ℝ) : Prop :=
  ∑ i in Finset.range n, (alphas i ^ 2 - alphas i * alphas ((i + 1) % n)) / 
    (alphas i ^ 2 + alphas ((i + 1) % n) ^ 2) ≥ 0

def largest_n : ℕ :=
  3

theorem largest_positive_integer_n (n : ℕ) (h : ∀ (alphas : Fin n → ℝ), (∀ i, alphas i > 0) → alpha_condition n alphas) :
  n ≤ 3 :=
  sorry

end largest_positive_integer_n_l169_169668


namespace intersection_points_count_l169_169496

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x
noncomputable def g (x : ℝ) : ℝ := x^2 - 4 * x + 5

theorem intersection_points_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 = g x1 ∧ f x2 = g x2 ∧ (∀ x, f x = g x → x = x1 ∨ x = x2) :=
by
  sorry

end intersection_points_count_l169_169496


namespace negation_of_proposition_l169_169495

theorem negation_of_proposition (a : ℝ) : 
  ¬(a = -1 → a^2 = 1) ↔ (a ≠ -1 → a^2 ≠ 1) :=
by sorry

end negation_of_proposition_l169_169495


namespace find_solutions_l169_169891

noncomputable def f (x : ℝ) : ℝ := - (5 / 3) * x + (10 / 3) * (1 / x)

theorem find_solutions :
  (∀ x : ℝ, x ≠ 0 → f(x) + 2 * f(x⁻¹) = 5 * x) →
  (∀ x : ℝ, f(x) = -f(-x) → x = real.sqrt 2 ∨ x = -real.sqrt 2) :=
by
  intros h_cond h_eq
  sorry

end find_solutions_l169_169891


namespace lemoine_point_is_centroid_l169_169792

-- Defining the triangle ABC and Lemoine point K
variables {A B C K A₁ B₁ C₁ : Type} [affine_space ℝ ℝ^2]

-- Assumption that A₁, B₁ and C₁ are the perpendicular projections of K onto the sides of triangle ABC.
variable (is_projection : ∀ (P : ℝ^2), P ∈ set_of (λ P, dist P K = dist P (line_through B C))
  ∨ P ∈ set_of (λ P, dist P K = dist P (line_through C A))
  ∨ P ∈ set_of (λ P, dist P K = dist P (line_through A B)))

-- Statement to be proven: K is the centroid of the triangle formed by projections
theorem lemoine_point_is_centroid (A B C K A₁ B₁ C₁ : ℝ^2) 
  (hA₁ : A₁ ∈ line_through B C) (hB₁ : B₁ ∈ line_through C A) (hC₁ : C₁ ∈ line_through A B)
  (projA₁ : dist A₁ K = dist A₁ (line_through B C)) 
  (projB₁ : dist B₁ K = dist B₁ (line_through C A)) 
  (projC₁ : dist C₁ K = dist C₁ (line_through A B)) : 
  affine_combination [A₁, B₁, C₁] [1/3, 1/3, 1/3] = K :=
by
  sorry

end lemoine_point_is_centroid_l169_169792


namespace value_of_M_l169_169450

noncomputable def a : ℝ := (Real.sqrt (Real.sqrt 8 + 3) + Real.sqrt (Real.sqrt 8 - 3)) / Real.sqrt (Real.sqrt 8 + 2)
noncomputable def b : ℝ := Real.sqrt (5 - 2 * Real.sqrt 6)
noncomputable def M : ℝ := a - b

theorem value_of_M : M = 4 :=
by
  sorry

end value_of_M_l169_169450


namespace angle_CAB_is_60_degree_l169_169148

theorem angle_CAB_is_60_degree
    {A B C D E F G : Type} -- Points in a geometrical space
    (h1 : dist A C = dist B C) 
    (h2 : dist A C = dist C D) 
    (h3 : dist C D = dist G D)
    (h4 : dist G D = dist D F)
    (h5 : dist D F = dist E F)
    (h6 : angle A C B = angle E F D) :
    angle A C B = 60 := 
    sorry

end angle_CAB_is_60_degree_l169_169148


namespace nested_radical_eq_6_l169_169657

theorem nested_radical_eq_6 (x : ℝ) (h : x = Real.sqrt (18 + x)) : x = 6 :=
by 
  have h_eq : x^2 = 18 + x,
  { rw h, exact pow_two (Real.sqrt (18 + x)) },
  have quad_eq : x^2 - x - 18 = 0,
  { linarith [h_eq] },
  have factored : (x - 6) * (x + 3) = x^2 - x - 18,
  { ring },
  rw [←quad_eq, factored] at h,
  sorry

end nested_radical_eq_6_l169_169657


namespace brushes_cost_l169_169118

-- Define the conditions
def canvas_cost (B : ℝ) : ℝ := 3 * B
def paint_cost : ℝ := 5 * 8
def total_material_cost (B : ℝ) : ℝ := B + canvas_cost B + paint_cost
def earning_from_sale : ℝ := 200 - 80

-- State the question as a theorem in Lean
theorem brushes_cost (B : ℝ) (h : total_material_cost B = earning_from_sale) : B = 20 :=
sorry

end brushes_cost_l169_169118


namespace count_valid_numbers_l169_169048

-- Define the range of numbers we are analyzing
def range := finset.Icc 1 500

-- Define the condition for being a multiple of both 4 and 6 (i.e., multiple of 12)
def is_multiple_of_12 (n : ℕ) : Prop := n % 12 = 0

-- Define the condition for not being a multiple of 5
def not_multiple_of_5 (n : ℕ) : Prop := ¬ (n % 5 = 0)

-- Define the condition for not being a multiple of 9
def not_multiple_of_9 (n : ℕ) : Prop := ¬ (n % 9 = 0)

-- Define the final set of numbers according to the conditions specified
def valid_numbers := range.filter (λ n, is_multiple_of_12 n ∧ not_multiple_of_5 n ∧ not_multiple_of_9 n)

-- Define the theorem we want to prove
theorem count_valid_numbers : valid_numbers.card = 26 :=
by
  sorry

end count_valid_numbers_l169_169048


namespace chord_length_l169_169897

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end chord_length_l169_169897


namespace translate_parabola_l169_169185

-- Define the original function
def original_parabola (x : ℝ) : ℝ := x^2

-- Define the translation function
def translate_left (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f (x + a)
def translate_down (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f x - b

-- The theorem that states the resulting function after the translations
theorem translate_parabola :
  (translate_down (translate_left original_parabola 2) 3) = 
  (λ x, (x + 2)^2 - 3) :=
by {
  sorry
}

end translate_parabola_l169_169185


namespace Andy_more_white_socks_than_black_l169_169616

def num_black_socks : ℕ := 6
def initial_num_white_socks : ℕ := 4 * num_black_socks
def final_num_white_socks : ℕ := initial_num_white_socks / 2
def more_white_than_black : ℕ := final_num_white_socks - num_black_socks

theorem Andy_more_white_socks_than_black :
  more_white_than_black = 6 :=
sorry

end Andy_more_white_socks_than_black_l169_169616


namespace probability_of_match_ending_after_5th_game_l169_169525

theorem probability_of_match_ending_after_5th_game :
  let first_3_win_prob_A := 3 / 5
  let subsequent_win_prob_A := 2 / 5
  let total_prob := ((3.choose 2 * (first_3_win_prob_A^2) * (1 - subsequent_win_prob_A) * subsequent_win_prob_A)
                   + (3.choose 1 * first_3_win_prob_A * (1 - first_3_win_prob_A)^2 * subsequent_win_prob_A^2)
                   + (3.choose 2 * ((1 - first_3_win_prob_A)^2) * first_3_win_prob_A * (1 - subsequent_win_prob_A) * subsequent_win_prob_A)
                   + (3.choose 1 * (1 - first_3_win_prob_A) * (first_3_win_prob_A)^2 * (1 - subsequent_win_prob_A)^2)) / 
                  625,
  total_prob = 234 / 625 := sorry

end probability_of_match_ending_after_5th_game_l169_169525


namespace initial_students_per_group_l169_169913

-- Define the conditions
variables {x : ℕ} (h : 3 * x - 2 = 22)

-- Lean 4 statement of the proof problem
theorem initial_students_per_group (x : ℕ) (h : 3 * x - 2 = 22) : x = 8 :=
sorry

end initial_students_per_group_l169_169913


namespace only_height_weight_is_correlation_l169_169360

def relationship_square_area_side_length : Prop := ∀ (s : ℝ), ∃ (A : ℝ), A = s * s
def relationship_vehicle_distance_time : Prop := ∀ (v t : ℝ), ∃ (d : ℝ), d = v * t
def relationship_sphere_radius_volume : Prop := ∀ (r : ℝ), ∃ (V : ℝ), V = (4 / 3) * π * r^3
def relationship_height_weight : Prop := ∃ (h w : ℝ), correlation h w

def prove_correlation_relationship : Prop :=
  relationship_height_weight ∧
  ¬ relationship_square_area_side_length ∧
  ¬ relationship_vehicle_distance_time ∧
  ¬ relationship_sphere_radius_volume

theorem only_height_weight_is_correlation : prove_correlation_relationship :=
sorry

end only_height_weight_is_correlation_l169_169360


namespace find_length_of_BC_l169_169270

section
variables {A B C : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C]

structure IsoscelesTriangle (A B C : Type) extends Triangle A B C :=
  (AB_eq_AC : AB = AC)

def Triangle (AB AC BC : ℝ) : Prop :=
  ∃ (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C],
    let isosceles := IsoscelesTriangle.mk A B C AB AC BC sorry in
    ∃ {M N: A}, M_is_midpoint : midpoint B C M ∧ N_is_midpoint : midpoint A C N ∧
    (distance A M = 5) ∧ (distance B N = 3 * sqrt 5)

theorem find_length_of_BC (AB AC : ℝ) (M N : A) :
  let B := AB,
  let C := AC,
  let AM := 5,
  let BN := 3 * sqrt 5 in
  IsoscelesTriangle A B C →
  (midpoint B C M) →
  (midpoint A C N) →
  (distance A M = 5) →
  (distance B N = 3 * sqrt 5) →
  distance B C = 4 * sqrt 5 := sorry

end

end find_length_of_BC_l169_169270


namespace cost_of_50_roses_l169_169622

def cost_of_dozen_roses : ℝ := 24

def is_proportional (n : ℕ) (cost : ℝ) : Prop :=
  cost = (cost_of_dozen_roses / 12) * n

def has_discount (n : ℕ) : Prop :=
  n ≥ 45

theorem cost_of_50_roses :
  ∃ (cost : ℝ), is_proportional 50 cost ∧ has_discount 50 ∧ cost * 0.9 = 90 :=
by
  sorry

end cost_of_50_roses_l169_169622


namespace find_minuend_l169_169551

variable (x y : ℕ)

-- Conditions
axiom h1 : x - y = 8008
axiom h2 : x - 10 * y = 88

-- Theorem statement
theorem find_minuend : x = 8888 :=
by
  sorry

end find_minuend_l169_169551


namespace problem_one_problem_two_problem_three_l169_169043

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos (3*x / 2), Real.sin (3*x / 2))
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.cos (x / 2), -Real.sin (x / 2))
noncomputable def f (x m : ℝ) : ℝ :=
  let dot_product := a x.1 * b x.1 + a x.2 * b x.2
  let norm := Real.sqrt ((a x.1 + b x.1)^2 + (a x.2 + b x.2)^2)
  dot_product - m * norm + 1

-- 1. When m = 0, prove f(π/6) = 3/2
theorem problem_one : f (π/6) 0 = 3/2 := by
  sorry

-- 2. Prove m = sqrt(2) when the minimum value of f(x) is -1
theorem problem_two (H : ∀ x, -π/3 ≤ x ∧ x ≤ π/4 → f x $sqrt(2)$ = -1) : 
  m = sqrt(2) := by 
  sorry

-- 3. Prove the range of m for g(x) = f(x) + (24/49)m^2 to have four distinct zeros
noncomputable def g (x m : ℝ) : ℝ := f x m + (24/49)*m^2

theorem problem_three (H : ∀ x, -π/3 ≤ x ∧ x ≤ π/4 → g x m = 0) :
  ∃  -- ∃ means "there exists"
  m, (7 * sqrt(2) / 6) ≤ m ∧ m < (7 / 4) ∧
  ∀ x y,
    -π/3 ≤ x ∧ x ≤ π/4 ∧ -π/3 ≤ y ∧ y ≤ π/4 ∧
    g x m = 0 ∧ g y m = 0 ∧ x ≠ y :=
    by sorry

end problem_one_problem_two_problem_three_l169_169043


namespace find_CD_length_l169_169759

theorem find_CD_length (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (h1 : ∠ BAC = ∠ BDC) (h2 : ∠ ABD = ∠ CBD) (AB : ℝ) (BD : ℝ) (BC : ℝ)
  (h3 : AB = 15) (h4 : BD = 12) (h5 : BC = 7) :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (CD = m / n) ∧ (m + n = 33) :=
begin
  sorry
end

end find_CD_length_l169_169759


namespace garden_to_land_area_ratio_l169_169501

variables (l_ter w_ter l_gard w_gard : ℝ)

-- Condition 1: Width of the land rectangle is 3/5 of its length
def land_conditions : Prop := w_ter = (3 / 5) * l_ter

-- Condition 2: Width of the garden rectangle is 3/5 of its length
def garden_conditions : Prop := w_gard = (3 / 5) * l_gard

-- Problem: Ratio of the area of the garden to the area of the land is 36%.
theorem garden_to_land_area_ratio
  (h_land : land_conditions l_ter w_ter)
  (h_garden : garden_conditions l_gard w_gard) :
  (l_gard * w_gard) / (l_ter * w_ter) = 0.36 := sorry

end garden_to_land_area_ratio_l169_169501


namespace max_coins_as_pleases_max_coins_equally_distributed_l169_169925

-- Part a
theorem max_coins_as_pleases {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 31) := 
by
  sorry

-- Part b
theorem max_coins_equally_distributed {N : ℕ} (N_warriors : N = 33) (total_coins : ℕ := 240) : 
  ∃ k : ℕ, k ≤ N ∧ (∃ remaining_coins : ℕ, remaining_coins ≤ total_coins ∧ remaining_coins = 30) := 
by
  sorry

end max_coins_as_pleases_max_coins_equally_distributed_l169_169925


namespace constant_term_equality_l169_169098

theorem constant_term_equality (a : ℝ) 
  (h1 : ∃ T, T = (x : ℝ)^2 + 2 / x ∧ T^9 = 64 * ↑(Nat.choose 9 6)) 
  (h2 : ∃ T, T = (x : ℝ) + a / (x^2) ∧ T^9 = a^3 * ↑(Nat.choose 9 3)):
  a = 4 := 
sorry

end constant_term_equality_l169_169098


namespace hyperbola_eccentricity_l169_169640

/-- Given the equation of a hyperbola, this theorem proves that its eccentricity (e) is calculated correctly. -/
theorem hyperbola_eccentricity : 
  ∀ (x y : ℝ), (∃ a b c e : ℝ, a^2 = 3 ∧ b^2 = 6 ∧ c^2 = a^2 + b^2 ∧ c = 3 ∧ e = c / a ∧ e = sqrt 3) :=
by
  intros x y
  use sqrt 3, sqrt 6, 3, sqrt 3 -- values for a, b, c, e
  split
  { rw [pow_two, ← eq_comm], norm_num },
  split
  { rw [pow_two, ← eq_comm], norm_num },
  split
  { rw [pow_two, pow_two, ← eq_comm], norm_num }, -- c^2 = a^2 + b^2
  split
  { norm_num }, -- c = 3
  split
  { rw [div_eq_inv_mul, ← mul_self_inj_of_nonneg, sqrt_mul_self, inv_eq_one_div, ← mul_assoc, one_mul],
    norm_num,
    exact le_of_lt (real.sqrt_pos.2 zero_lt_three), exact le_of_lt (zero_lt_iff.2 zero_ne_three) },
  { rw [eq_comm, sqrt_eq, ← sqrt_eq_sqrt_iff (le_of_lt (by norm_num)) (real.sqrt_nonneg _)],
    norm_num} -- e = sqrt 3

-- Sorry is added for placeholders of the non-trivial parts of the proof
sorry

end hyperbola_eccentricity_l169_169640


namespace triangle_DEF_area_l169_169250

noncomputable def area_of_triangle (D E F : (ℝ × ℝ)) : ℝ :=
  0.5 * abs (D.1 * (E.2 - F.2) + E.1 * (F.2 - D.2) + F.1 * (D.2 - E.2))

theorem triangle_DEF_area :
  let D := (0, 1)
  let E := (6, 0)
  let F := (3, 3)
  area_of_triangle D E F = 10.5 :=
by
  let D := (0, 1)
  let E := (6, 0)
  let F := (3, 3)
  have h : area_of_triangle D E F = 10.5 := by
    calc
      area_of_triangle D E F
          = 0.5 * abs (0 * (0 - 3) + 6 * (3 - 1) + 3 * (1 - 0)) : by rfl
      ... = 0.5 * abs (0 + 18 + 3) : by rfl
      ... = 0.5 * 21 : by rfl
      ... = 10.5 : by rfl
  exact h

end triangle_DEF_area_l169_169250


namespace area_of_triangle_ABC_l169_169033

theorem area_of_triangle_ABC :
  let A := (-3/4 : ℝ, - (Real.sqrt 2) / 2),
  let B := (1/4 : ℝ, (Real.sqrt 2) / 2),
  let C := (5/4 : ℝ, - (Real.sqrt 2) / 2) in
  let AC := Real.sqrt ((5/4 + 3/4)^2 + (0)^2) in
  AC = Real.sqrt 2 →
  ∃ (A B C : ℝ × ℝ), 
    let area := (1 / 2) * AC * Real.sqrt 2 in
    area = Real.sqrt 2 := sorry

end area_of_triangle_ABC_l169_169033


namespace carson_pumps_needed_l169_169277

theorem carson_pumps_needed :
  let tire_volume := 500 -- cubic inches
  let first_two_tires := (2 : ℕ)
  let first_two_rate := 50 -- cubic inches per pump
  let third_tire_initial := 0.4 * tire_volume -- cubic inches
  let third_tire_rate := 40 -- cubic inches per pump
  let third_tire_deflate_rate := 10 -- cubic inches per pump
  let last_tire_initial := 0.7 * tire_volume -- cubic inches
  let last_tire_rate := 60 -- cubic inches per pump
  let first_two_tires_needed_pumps := (tire_volume / first_two_rate : ℝ)
  let third_tire_needed_pumps :=
    ((tire_volume - third_tire_initial) / (third_tire_rate - third_tire_deflate_rate) : ℝ)
  let last_tire_needed_pumps :=
    (tire_volume - last_tire_initial) / last_tire_rate
in
  2 * first_two_tires_needed_pumps + third_tire_needed_pumps + last_tire_needed_pumps = 33 := sorry

end carson_pumps_needed_l169_169277


namespace max_good_set_size_l169_169598

def is_composite (n : ℕ) : Prop :=
  ¬ n.prime ∧ 1 < n

def good_set (S : Finset ℕ) : Prop :=
  ∀ a b ∈ S, a ≠ b → Nat.gcd a b = 1

theorem max_good_set_size (S : Finset ℕ) :
  (∀ n ∈ S, is_composite n) →
  good_set S →
  S.card ≤ 14 :=
begin
  sorry -- proof will be here
end

end max_good_set_size_l169_169598


namespace lines_are_perpendicular_l169_169039

-- Definitions of the conditions
variables {Point : Type*} [AffineSpace Point ℝ] 
variables (Γ₁ Γ₂ : Set Point) -- circles
variables (A B O C : Point) -- intersections and center
variables (D E : Point) -- intersections with lines

def is_center (O : Point) (Γ : Set Point) := sorry -- requires definition of a circle center
def is_on_circle (P : Point) (Γ : Set Point) := sorry -- requires definition of being on a circle

-- Problem setup as conditions
variables (hI : ∃ (A B : Point), A ≠ B ∧ A ∈ Γ₁ ∧ B ∈ Γ₁ ∧ A ∈ Γ₂ ∧ B ∈ Γ₂)
variables (hO : is_center O Γ₁)
variables (hC : C ∈ Γ₁ ∧ C ≠ A ∧ C ≠ B)
variables (hD : ∃ (P : Point), P ∈ Γ₂ ∧ P ∈ line_span ℝ {A, C})
variables (hE : ∃ (Q : Point), Q ∈ Γ₂ ∧ Q ∈ line_span ℝ {B, C})

theorem lines_are_perpendicular
  (hOc : line_span ℝ {O, C} = line_span ℝ {O, C})
  (hDE : ∃ (D E : Point), line_span ℝ {D, E} = line_span ℝ {D, E}) : 
  ⊥ (line_span ℝ {O, C}) (line_span ℝ {D, E}) := 
sorry

end lines_are_perpendicular_l169_169039


namespace Julio_limes_expense_l169_169781

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end Julio_limes_expense_l169_169781


namespace triangle_inequality_l169_169445

variable {a b c : ℝ}

theorem triangle_inequality (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (habc1 : a + b > c) (habc2 : a + c > b) (habc3 : b + c > a) :
  (a / (b + c) + b / (c + a) + c / (a + b) < 2) :=
sorry

end triangle_inequality_l169_169445


namespace locus_of_point_X_l169_169317

-- Definitions
variable {A B C X : Point}
variable (△ABC : Triangle)
variable (h_eq : Equilateral △ABC)
variable (X_inside : X ∈ interior △ABC)

-- The statement we want to prove
theorem locus_of_point_X (h1 : ∠ X A B + ∠ X B C + ∠ X C A = 90°) : 
  X ∈ set_of_points_on_altitudes_of △ABC :=
sorry

end locus_of_point_X_l169_169317


namespace domain_of_f_l169_169452

def f (x : ℝ) : ℝ := log (3 - x) / log 2

theorem domain_of_f :
  {x : ℝ | f x ∈ ℝ } = {x : ℝ | x < 3} :=
 sorry

end domain_of_f_l169_169452


namespace chord_length_l169_169898

theorem chord_length (t : ℝ) :
  (∃ x y, x = 1 + 2 * t ∧ y = 2 + t ∧ x ^ 2 + y ^ 2 = 9) →
  ((1.8 - (-3)) ^ 2 + (2.4 - 0) ^ 2 = (12 / 5 * Real.sqrt 5) ^ 2) :=
by
  sorry

end chord_length_l169_169898


namespace parallel_vectors_l169_169176

open_locale classical

theorem parallel_vectors (x : ℝ) : (∃ k : ℝ, (2, -1) = k • (-4, x)) → x = 2 :=
by
  intro h
  cases h with k hk
  sorry

end parallel_vectors_l169_169176


namespace regular_heptagon_interior_angle_l169_169195

theorem regular_heptagon_interior_angle :
  ∀ (n : ℕ), n = 7 → (∑ i in finset.range n, 180 / n) = 128.57 :=
  by
  intros n hn
  rw hn
  sorry

end regular_heptagon_interior_angle_l169_169195


namespace hexagon_largest_angle_l169_169481

theorem hexagon_largest_angle (x : ℝ) (hx : 6*x + 15 = 720) :
  (x + 5) = 122.5 :=
by {
  have h1: x = 117.5, {
    linarith, },
  rw h1,
  linarith,
}

end hexagon_largest_angle_l169_169481


namespace yen_per_pound_l169_169086

theorem yen_per_pound 
  (pounds_initial : ℕ) 
  (euros : ℕ) 
  (yen_initial : ℕ) 
  (pounds_per_euro : ℕ) 
  (yen_total : ℕ) 
  (hp : pounds_initial = 42) 
  (he : euros = 11) 
  (hy : yen_initial = 3000) 
  (hpe : pounds_per_euro = 2) 
  (hy_total : yen_total = 9400) 
  : (yen_total - yen_initial) / (pounds_initial + euros * pounds_per_euro) = 100 := 
by
  sorry

end yen_per_pound_l169_169086


namespace find_m_l169_169368

theorem find_m (m : ℝ) :
  (∃ x : ℝ, x^2 - m * x + m^2 - 19 = 0 ∧ (x = 2 ∨ x = 3))
  ∧ (∀ x : ℝ, x^2 - m * x + m^2 - 19 = 0 → x ≠ 2 ∧ x ≠ -4) 
  → m = -2 :=
by
  sorry

end find_m_l169_169368


namespace abs_diff_squares_l169_169955

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169955


namespace min_distance_proof_l169_169888

noncomputable def min_distance_ellipse_to_line : ℝ :=
  let ellipse := λ x y : ℝ, (x^2) / 4 + y^2 = 1
  let line := λ x y : ℝ, 2 * x - 3 * y + 6 = 0
  (6 - Real.sqrt 13) / Real.sqrt 13

theorem min_distance_proof :
  ∀ (x y : ℝ), (ellipse x y ∧ line x y → | 2 * x - 3 * y + 6 | / √(2^2 + (-3)^2) = min_distance_ellipse_to_line := 
by
  sorry

end min_distance_proof_l169_169888


namespace area_triangle_ABC_l169_169188

open Real

def point (a b : ℝ) : Type :=
  { x // x = a } × { y // y = b }

def line (m b : ℝ) : ℝ → ℝ := 
  fun x => m * x + b

def area_of_triangle (A B C : point) : ℝ :=
  abs (fst A.1 * (snd B.1 - snd C.1) + fst B.1 * (snd C.1 - snd A.1) 
      + fst C.1 * (snd A.1 - snd B.1)) / 2

-- Define vertices
def A : point := (⟨3, rfl⟩, ⟨0, rfl⟩)
def B : point := (⟨0, rfl⟩, ⟨3, rfl⟩)
def C : point := (⟨5, rfl⟩, ⟨5 - 5, rfl⟩)  -- The y-coordinate follows the line x - y = 5

theorem area_triangle_ABC : area_of_triangle A B C = 3 :=
  sorry

end area_triangle_ABC_l169_169188


namespace paint_coverage_l169_169115

theorem paint_coverage
  (wall_area : ℚ)
  (coats : ℕ)
  (gallons : ℕ)
  (total_area : ℚ)
  (coverage : ℚ) :
  wall_area = 600 →
  coats = 2 →
  gallons = 3 →
  total_area = wall_area * coats →
  coverage = total_area / gallons →
  coverage = 400 := 
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4
  rw [h4] at h5
  exact h5

end paint_coverage_l169_169115


namespace odd_factors_of_420_l169_169730

/-- 
Proof problem: Given that 420 can be factorized as \( 2^2 \times 3 \times 5 \times 7 \), 
we need to prove that the number of odd factors of 420 is 8.
-/
def number_of_odd_factors (n : ℕ) : ℕ :=
  let odd_factors := n.factors.filter (λ x, x % 2 ≠ 0)
  in odd_factors.length

theorem odd_factors_of_420 : number_of_odd_factors 420 = 8 := 
sorry

end odd_factors_of_420_l169_169730


namespace graph_cycles_equal_length_l169_169003

theorem graph_cycles_equal_length {ε : ℝ} (hε : 0 < ε) :
  ∃ (v_0 : ℕ), ∀ (v : ℕ), v > v_0 → ∀ (G : SimpleGraph (fin v)),
  G.edge_count ≥ (1 + ε) * v →
  ∃ (C1 C2 : SimpleGraph.Cycle (fin v)), C1 ≠ C2 ∧ C1.length = C2.length :=
by
  sorry

end graph_cycles_equal_length_l169_169003


namespace euler_totient_inequality_l169_169787

open Int

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k : ℕ, (Nat.Prime p) ∧ (k ≥ 1) ∧ (m = p^k)

def φ (n m : ℕ) (h : m ≠ 1) : ℕ := -- This is a placeholder, you would need an actual implementation for φ
  sorry

theorem euler_totient_inequality (m : ℕ) (h : m ≠ 1) :
  (is_power_of_prime m) ↔ (∀ n > 0, (φ n m h) / n ≥ (φ m m h) / m) :=
sorry

end euler_totient_inequality_l169_169787


namespace dima_age_l169_169125

variable (x : ℕ)

-- Dima's age is x years
def age_of_dima := x

-- Dima's age is twice his brother's age
def age_of_brother := x / 2

-- Dima's age is three times his sister's age
def age_of_sister := x / 3

-- The average age of Dima, his sister, and his brother is 11 years
def average_age := (x + age_of_brother x + age_of_sister x) / 3 = 11

theorem dima_age (h1 : age_of_brother x = x / 2) 
                 (h2 : age_of_sister x = x / 3) 
                 (h3 : average_age x) : x = 18 := 
by sorry

end dima_age_l169_169125


namespace magnitude_diff_unit_vectors_l169_169812

variables (a b : ℝ^3) -- Define vector variables a and b in 3-dimensional real space

-- Define the conditions
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def sum_is_one (v1 v2 : ℝ^3) : Prop := ‖v1 + v2‖ = 1

-- State the theorem
theorem magnitude_diff_unit_vectors (a b : ℝ^3) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h_sum : sum_is_one a b) : 
  ‖a - b‖ = real.sqrt 3 :=
sorry

end magnitude_diff_unit_vectors_l169_169812


namespace average_remaining_numbers_l169_169483

theorem average_remaining_numbers (numbers : List ℝ) 
  (h_len : numbers.length = 50) 
  (h_avg : (numbers.sum / 50 : ℝ) = 38) 
  (h_discard : 45 ∈ numbers ∧ 55 ∈ numbers) :
  let new_sum := numbers.sum - 45 - 55
  let new_len := 50 - 2
  (new_sum / new_len : ℝ) = 37.5 :=
by
  sorry

end average_remaining_numbers_l169_169483


namespace best_cashback_categories_l169_169466

def transport_expense : ℕ := 2000
def groceries_expense : ℕ := 5000
def clothing_expense : ℕ := 3000
def entertainment_expense : ℕ := 3000
def sports_goods_expense : ℕ := 1500

def transport_cashback : ℕ → ℕ := λ exp, exp * 5 / 100
def groceries_cashback : ℕ → ℕ := λ exp, exp * 3 / 100
def clothing_cashback : ℕ → ℕ := λ exp, exp * 4 / 100
def entertainment_cashback : ℕ → ℕ := λ exp, exp * 5 / 100
def sports_goods_cashback : ℕ → ℕ := λ exp, exp * 6 / 100

theorem best_cashback_categories :
  ({2, 3, 4} : set ℕ) ⊆ {2, 3, 4, 5} ∧
  ∀ categories : set ℕ,
    categories ⊆ {1, 2, 3, 4, 5} →
    (∀ cat ∈ categories, cat = 2 ∨ cat = 3 ∨ cat = 4) →
    (∑ cat in categories, if cat = 1 then transport_cashback transport_expense
                  else if cat = 2 then groceries_cashback groceries_expense
                  else if cat = 3 then clothing_cashback clothing_expense
                  else if cat = 4 then entertainment_cashback entertainment_expense
                  else sports_goods_cashback sports_goods_expense)
    ≤ (600 + 10) :=   -- Adjusted amount to reflect equivalent condition
sorry

end best_cashback_categories_l169_169466


namespace minimum_distance_sum_l169_169107

variables {A B C M : EPoint} {a b c : ℝ}

-- Define the sides of the triangle
def length_a (A B : EPoint) : ℝ := R2.dist A B
def length_b (B C : EPoint) : ℝ := R2.dist B C
def length_c (C A : EPoint) : ℝ := R2.dist C A

-- Main statement to prove
theorem minimum_distance_sum
  (hA: length_a A B = a)
  (hB: length_b B C = b)
  (hC: length_c C A = c):
  ∃ G : EPoint, (∀ M : EPoint, (R2.dist M A)^2 + (R2.dist M B)^2 + (R2.dist M C)^2 ≥ (R2.dist G A)^2 + (R2.dist G B)^2 + (R2.dist G C)^2) ∧ 
  ((R2.dist G A)^2 + (R2.dist G B)^2 + (R2.dist G C)^2 = (a ^ 2 + b ^ 2 + c ^ 2) / 3) :=
by { sorry }

end minimum_distance_sum_l169_169107


namespace digit_150_of_3_div_11_l169_169534

theorem digit_150_of_3_div_11 : 
  let n := 150
  let digits := "27"
  let cycle_length := 2
  let digit_150 := digits[(n % cycle_length)]
  in digit_150 = '7' :=
by {
  sorry
}

end digit_150_of_3_div_11_l169_169534


namespace carbonic_acid_formation_l169_169316

-- Definition of amounts of substances involved
def moles_CO2 : ℕ := 3
def moles_H2O : ℕ := 3

-- Stoichiometric condition derived from the equation CO2 + H2O → H2CO3
def stoichiometric_ratio (a b c : ℕ) : Prop := (a = b) ∧ (a = c)

-- The main statement to prove
theorem carbonic_acid_formation : 
  stoichiometric_ratio moles_CO2 moles_H2O 3 :=
by
  sorry

end carbonic_acid_formation_l169_169316


namespace number_of_k_solutions_l169_169375

theorem number_of_k_solutions :
  ∃ (n : ℕ), n = 1006 ∧
  (∀ k, (∃ a b : ℕ+, (a ≠ b) ∧ (k * (a + b) = 2013 * Nat.lcm a b)) ↔ k ≤ n ∧ 0 < k) :=
by
  sorry

end number_of_k_solutions_l169_169375


namespace rick_bought_30_guppies_l169_169849

theorem rick_bought_30_guppies (G : ℕ) (T C : ℕ) 
  (h1 : T = 4 * C) 
  (h2 : C = 2 * G) 
  (h3 : G + C + T = 330) : 
  G = 30 := 
by 
  sorry

end rick_bought_30_guppies_l169_169849


namespace find_a3_l169_169706

-- Definitions of various terms used in the problem
variables (n : ℕ) (a S : ℕ → ℝ) (a_1 a_3 d : ℝ)

-- Conditions provided in the problem
def sum_first_n_terms (n : ℕ) : ℝ := S n
def first_term : ℝ := a 1
def sum_first_term : ℝ := S 1
def sum_third_term : ℝ := S 3

-- Specific conditions stated in the problem
def condition_1 : sum_first_term = a_1 := rfl
def condition_2 : sum_third_term = 3 * a_1 + 3 * d := rfl
def condition_3 : 3 * sum_first_term - 2 * sum_third_term = 15 := 
  by sorry -- This would be proven based on the definition of the sequence sum

-- The goal to prove
theorem find_a3 :
  a_3 = -5 :=
sorry

end find_a3_l169_169706


namespace interior_angle_heptagon_l169_169209

theorem interior_angle_heptagon : 
  ∀ (n : ℕ), n = 7 → (5 * 180 / n : ℝ) = 128.57142857142858 :=
by
  intros n hn
  rw hn
  -- The proof is skipped
  sorry

end interior_angle_heptagon_l169_169209


namespace equation_of_line_l_l169_169244

-- Definitions of points and circle
def M : ℝ × ℝ := (1, 2)
def O : ℝ × ℝ := (2, 0)
def CircleA : set (ℝ × ℝ) := {p | (p.1 - 2) ^ 2 + (p.2) ^ 2 = 9}

-- Definition of the line connecting M and O
def LineMO : set (ℝ × ℝ) := {p | ∃ (k : ℝ), p.2 = -2 * p.1 + k}

-- Definition of line l passing through M and is perpendicular to LineMO
def LineL : set (ℝ × ℝ) := {p | p.2 = (1 / 2) * p.1 + 3/2}

-- The theorem to prove
theorem equation_of_line_l : ∀ p : ℝ × ℝ, p ∈ LineL ↔ p.1 - 2 * p.2 + 3 = 0 := by
  sorry

end equation_of_line_l_l169_169244


namespace danil_claim_false_l169_169287

theorem danil_claim_false (E O : ℕ) (hE : E % 2 = 0) (hO : O % 2 = 0) (h : O = E + 15) : false :=
by sorry

end danil_claim_false_l169_169287


namespace problem_union_M_N_l169_169835

open Set

-- Definitions based on problem conditions
def M : Set ℝ := {x | x^2 = x}
def N : Set ℝ := {x | real.log10 x ≤ 0}

-- Proof statement (needs a proof)
theorem problem_union_M_N : M ∪ N = Icc (0 : ℝ) 1 := by
  sorry

end problem_union_M_N_l169_169835


namespace speed_of_second_train_l169_169526

theorem speed_of_second_train
  (length_first_train : ℝ)
  (length_second_train : ℝ)
  (speed_first_train : ℝ)
  (cross_time_seconds : ℝ)
  (relative_speed : ℝ)
  (total_distance_km : ℝ)
  : 
  length_first_train = 140 
  → length_second_train = 160
  → speed_first_train = 60
  → cross_time_seconds = 10.799136069114471
  → relative_speed = (total_distance_km / (cross_time_seconds / 3600))
  → total_distance_km = ((length_first_train + length_second_train) / 1000)
  → relative_speed = speed_first_train + 40 :=
begin
  sorry
end

end speed_of_second_train_l169_169526


namespace find_B_find_a_l169_169422

noncomputable theory

-- Define the problem conditions
variables {A B C : ℝ} {a b c : ℝ}

-- Given conditions
def cond1 : Prop := a * Mathlib.cos A * Mathlib.cos B - b * Mathlib.sin A ^ 2 - c * Mathlib.cos A = 2 * b * Mathlib.cos B
def cond2 : Prop := b = Mathlib.sqrt 7 * a
def area_triangle : ℝ := 2 * Mathlib.sqrt 3

-- The first part of the problem: Finding B.
theorem find_B (h1: cond1) : B = 2 * Mathlib.pi / 3 :=
sorry

-- The second part of the problem: Finding a.
theorem find_a (h2: cond2) (h3 : area_triangle = 2 * Mathlib.sqrt 3) : a = 2 :=
sorry

end find_B_find_a_l169_169422


namespace range_of_m_l169_169345

variable (m : ℝ)

def A (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 3 * m - 1 }
def B : Set ℝ := {x | 1 ≤ x ∧ x ≤ 10}

theorem range_of_m (h : A m ∪ B = B) : m ≤ 11 / 3 := by
  sorry

end range_of_m_l169_169345


namespace least_number_subtracted_l169_169550

-- Define the original number and the divisor
def original_number : ℕ := 427398
def divisor : ℕ := 14

-- Define the least number to be subtracted
def remainder := original_number % divisor
def least_number := remainder

-- The statement to be proven
theorem least_number_subtracted : least_number = 6 :=
by
  sorry

end least_number_subtracted_l169_169550


namespace quadrilaterals_with_same_lengths_are_not_necessarily_equal_l169_169491

noncomputable def are_quadrilaterals_equal (l1 l2 : List ℝ) : Prop := 
  ∃ (q1 q2 : Quadrilateral), 
    ordered_lengths q1 = l1 ∧ 
    ordered_lengths q2 = l2 ∧ 
    l1 = l2 ∧ 
    q1 ≠ q2

theorem quadrilaterals_with_same_lengths_are_not_necessarily_equal (l1 l2 : List ℝ) :
  are_quadrilaterals_equal l1 l2 :=
begin
  sorry
end

end quadrilaterals_with_same_lengths_are_not_necessarily_equal_l169_169491


namespace hannah_age_is_48_l169_169044

-- Define the ages of the brothers
def num_brothers : ℕ := 3
def age_each_brother : ℕ := 8

-- Define the sum of brothers' ages
def sum_brothers_ages : ℕ := num_brothers * age_each_brother

-- Define the age of Hannah
def hannah_age : ℕ := 2 * sum_brothers_ages

-- The theorem to prove Hannah's age is 48 years
theorem hannah_age_is_48 : hannah_age = 48 := by
  sorry

end hannah_age_is_48_l169_169044


namespace trajectory_eq_l169_169075

theorem trajectory_eq :
  ∀ (x y : ℝ), abs x * abs y = 1 → (x * y = 1 ∨ x * y = -1) :=
by
  intro x y h
  sorry

end trajectory_eq_l169_169075


namespace slope_extension_l169_169514

theorem slope_extension (h : ℝ) (l₁ : ℝ) (θ₁ : ℝ) (θ₂ : ℝ)
  (hl₁: l₁ = 100) (hθ₁ : θ₁ = 45) (hθ₂ : θ₂ = 30)
  (height_unchanged : h = l₁ * real.sin (θ₁ * real.pi / 180)) :
  let l₂ = h / real.sin (θ₂ * real.pi / 180) in l₂ - l₁ ≈ 15.4 :=
by
  let l₂ := h / real.sin (θ₂ * real.pi / 180)
  have h : h = 100 := 
    by rw [hl₁, hθ₁]
       exact height_unchanged
  have l₂ := 100 * 2 / real.sqrt 3
  have extension := (l₂ - 100)
  show extension ≈ 15.4
  sorry

end slope_extension_l169_169514


namespace optimal_garden_dimensions_l169_169133

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), (2 * l + 2 * w = 400 ∧
                l ≥ 100 ∧
                w ≥ 0 ∧ 
                l * w = 10000) :=
by
  sorry

end optimal_garden_dimensions_l169_169133


namespace five_person_permutation_l169_169410

-- Define the main proposition
theorem five_person_permutation (five_people : Finset ℕ) (oldest_not_last : ∀ last : ℕ, last ≠ 4 → ∃ p ∈ five_people, p ≠ 4) :
  finset.card { l : List ℕ // l.perm five_people.to_list ∧ l.ilast ≠ 4 } = 96 :=
by
  sorry

end five_person_permutation_l169_169410


namespace greatest_sum_of_consecutive_integers_l169_169963

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169963


namespace cricket_average_increase_l169_169239

theorem cricket_average_increase (innings_played : ℕ) (initial_average next_innings_runs : ℕ) :
  innings_played = 20 →
  initial_average = 36 →
  next_innings_runs = 120 →
  ( 21 * (initial_average + 4) = (innings_played * initial_average + next_innings_runs + initial_average + 4) ) :=
begin
  intros,
  sorry
end

end cricket_average_increase_l169_169239


namespace avg_percentage_students_l169_169559

-- Define the function that calculates the average percentage of all students
def average_percent (n1 n2 : ℕ) (p1 p2 : ℕ) : ℕ :=
  (n1 * p1 + n2 * p2) / (n1 + n2)

-- Define the properties of the numbers of students and their respective percentages
def students_avg : Prop :=
  average_percent 15 10 70 90 = 78

-- The main theorem: Prove that given the conditions, the average percentage is 78%
theorem avg_percentage_students : students_avg :=
  by
    -- The proof will be provided here.
    sorry

end avg_percentage_students_l169_169559


namespace evaluate_expression_l169_169652

theorem evaluate_expression : 
  (Int.floor ((Real.ceil ((11/6 : Real) ^ 2)) + 19/5)) = 7 := 
by
  sorry

end evaluate_expression_l169_169652


namespace intercept_form_impossible_values_l169_169899

-- Define the problem statement
theorem intercept_form_impossible_values (m : ℝ) :
  (¬ (∃ a b c : ℝ, m ≠ 0 ∧ a * m = 0 ∧ b * m = 0 ∧ c * m = 1) ↔ (m = 4 ∨ m = -3 ∨ m = 5)) :=
sorry

end intercept_form_impossible_values_l169_169899


namespace factorization_m_minus_n_l169_169312

theorem factorization_m_minus_n :
  ∃ (m n : ℤ), (6 * (x:ℝ)^2 - 5 * x - 6 = (6 * x + m) * (x + n)) ∧ (m - n = 5) :=
by {
  sorry
}

end factorization_m_minus_n_l169_169312


namespace sum_possible_x_isosceles_l169_169408

theorem sum_possible_x_isosceles (x : ℝ) 
(iso_triangle : ∃ a b c : ℝ, a + b + c = 180 ∧ (a = b ∨ a = c ∨ b = c) ∧ (50 ∈ {a, b, c}) ∧ (x ∈ {a, b, c})) : 
(x = 50 ∨ x = 65 ∨ x = 80) → 50 + 65 + 80 = 195 :=
by sorry

end sum_possible_x_isosceles_l169_169408


namespace area_of_triangle_OPQ_l169_169443

variables (O F P Q : Type) [metric_space O] [metric_space F] [metric_space P] [metric_space Q]
variable (a b : ℝ)
variable (O_is_vertex : isVertex O) (F_is_focus : isFocus O F)

-- Define conditions
axiom h1 : dist O F = a
axiom h2 : dist P Q = b
axiom h3 : chord_through_focus P Q F

-- Define the problem to prove
theorem area_of_triangle_OPQ : triangle_area O P Q = a * sqrt (a * b) := 
sorry

end area_of_triangle_OPQ_l169_169443


namespace decreasing_interval_of_f_l169_169886

noncomputable def f (x : ℝ) : ℝ := -x^2 - x + 4

theorem decreasing_interval_of_f :
  ∃ (a b : ℝ), a = -1 ∧ b = -1 ∧
                 (- ∃ x ∈ set.Ioi (-b / (2 * a)), deriv f x < 0) :=
begin
  apply exists.intro (-1), -- For coefficient a
  apply exists.intro (-1), -- For coefficient b
  sorry
end

end decreasing_interval_of_f_l169_169886


namespace solve_color_problem_l169_169771

variables (R B G C : Prop)

def color_problem (R B G C : Prop) : Prop :=
  (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C) → C ∧ (R ∨ B)

theorem solve_color_problem (R B G C : Prop) (h : (C → (R ∨ B)) ∧ (¬C → (¬R ∧ ¬G)) ∧ ((B ∨ G) → C)) : C ∧ (R ∨ B) :=
  by {
    sorry
  }

end solve_color_problem_l169_169771


namespace infinite_solutions_ax2_by2_eq_z3_l169_169788

theorem infinite_solutions_ax2_by2_eq_z3 
  (a b : ℤ) 
  (coprime_ab : Int.gcd a b = 1) :
  ∃ (x y z : ℤ), (∀ n : ℤ, ∃ (x y z : ℤ), a * x^2 + b * y^2 = z^3 
  ∧ Int.gcd x y = 1) := 
sorry

end infinite_solutions_ax2_by2_eq_z3_l169_169788


namespace min_symmetric_set_size_l169_169261

def is_symmetric_about_origin (T : set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-a, -b) ∈ T

def is_symmetric_about_x (T : set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (a, -b) ∈ T

def is_symmetric_about_y (T : set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-a, b) ∈ T

def is_symmetric_about_y_eq_neg_x (T : set (ℝ × ℝ)) : Prop :=
  ∀ (a b : ℝ), (a, b) ∈ T → (-b, -a) ∈ T

def contains_point (T : set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  p ∈ T

def min_points_in_symmetric_set (T : set (ℝ × ℝ)) : ℕ :=
  if is_symmetric_about_origin T ∧ is_symmetric_about_x T ∧ is_symmetric_about_y T ∧ is_symmetric_about_y_eq_neg_x T ∧ contains_point T (1, 3) then 8 else 0

theorem min_symmetric_set_size : 
  (∃ (T : set (ℝ × ℝ)), is_symmetric_about_origin T ∧ is_symmetric_about_x T ∧ is_symmetric_about_y T ∧ is_symmetric_about_y_eq_neg_x T ∧ contains_point T (1, 3)) → min_points_in_symmetric_set T = 8 :=
sorry

end min_symmetric_set_size_l169_169261


namespace trains_cross_time_l169_169235

noncomputable def time_to_cross_trains : ℝ :=
  let l1 := 220 -- length of the first train in meters
  let s1 := 120 * (5 / 18) -- speed of the first train in meters per second
  let l2 := 280.04 -- length of the second train in meters
  let s2 := 80 * (5 / 18) -- speed of the second train in meters per second
  let relative_speed := s1 + s2 -- relative speed in meters per second
  let total_length := l1 + l2 -- total length to be crossed in meters
  total_length / relative_speed -- time in seconds

theorem trains_cross_time :
  abs (time_to_cross_trains - 9) < 0.01 := -- Allowing a small error to account for approximation
by
  sorry

end trains_cross_time_l169_169235


namespace trigonometric_identity_l169_169017

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + π / 3)
noncomputable def g (x : ℝ) : ℝ := Real.cos (2 * x + π / 3)

theorem trigonometric_identity : 
  ∀ x : ℝ, g(x) = f(π / 4 + x) :=
sorry

end trigonometric_identity_l169_169017


namespace abs_diff_squares_105_95_l169_169932

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l169_169932


namespace smallest_number_of_pens_l169_169852

theorem smallest_number_of_pens (pencils_per_package : ℕ) (pens_per_package : ℕ) 
  (h_pencils : pencils_per_package = 15) (h_pens : pens_per_package = 12) : 
  ∃ (smallest_number_of_pens : ℕ), smallest_number_of_pens = Nat.lcm pens_per_package pencils_per_package := 
by 
  use Nat.lcm pens_per_package pencils_per_package
  sorry

end smallest_number_of_pens_l169_169852


namespace max_sum_first_n_terms_l169_169143

-- Definitions and conditions
def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def condition1 (a : ℕ → ℝ) : Prop :=
  3 * a 7 = 5 * a 12

def condition2 (a : ℕ → ℝ) : Prop :=
  a 0 > 0

-- Define the sum of the first n terms of the sequence
def sum_seq (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, a i

-- Problem statement in Lean
theorem max_sum_first_n_terms (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : arithmetic_seq a)
  (h_cond1 : condition1 a) (h_cond2 : condition2 a) :
  ∀ n : ℕ, sum_seq a n ≤ sum_seq a 20 :=
sorry

end max_sum_first_n_terms_l169_169143


namespace find_AX_l169_169417

theorem find_AX
  (AB AC BC : ℕ)
  (h1 : AB = 60)
  (h2 : AC = 40)
  (h3 : BC = 20)
  (h4 : AC + BC = AB)
  (angle_bisector_theorem : ∀ AX BX: ℕ, AX / AC = BX / BC → AX = 2 * BX) :
  ∃ AX BX : ℕ, AX = 40 :=
by
  have AC_BC_eq : AC + BC = AB := by sorry
  rw [AC_BC_eq, h1] at h2 h3 -- This ensures AC + BC = AB assuming AC = 40 and BC = 20
  obtain ⟨AX, BX, hAX, hBX⟩ := angle_bisector_theorem (AX) (BX)
  use AX, BX
  have : AX = 40 := by sorry
  exact this

end find_AX_l169_169417


namespace expand_product_equivalence_l169_169309

variable (x : ℝ)  -- Assuming x is a real number

theorem expand_product_equivalence : (x + 5) * (x + 7) = x^2 + 12 * x + 35 :=
by
  sorry

end expand_product_equivalence_l169_169309


namespace polar_coordinate_C1_distance_AB_l169_169767

section

variables (t α θ : ℝ) 

-- Given the parametric equations of line l
def line_l_x (t : ℝ) : ℝ := 2 + real.sqrt 2 * t
def line_l_y (t : ℝ) : ℝ := 2 + real.sqrt 2 * t

-- Given the parametric equations of curve C1
def curve_C1_x (α : ℝ) : ℝ := 2 + real.sqrt 2 * real.cos α
def curve_C1_y (α : ℝ) : ℝ := real.sqrt 2 * real.sin α

-- Given the polar coordinate equation of curve C2
def curve_C2_polar (ρ θ : ℝ) : Prop := ρ * (real.sin θ)^2 = 4 * real.cos θ

-- Prove the polar coordinate equation of curve C1 is ρ^2 - 4ρcosθ + 2 = 0
theorem polar_coordinate_C1 (ρ θ : ℝ) :
  (exists α, curve_C1_x α = ρ * real.cos θ ∧ curve_C1_y α = ρ * real.sin θ) →
  ρ^2 - 4 * ρ * real.cos θ + 2 = 0 :=
sorry

-- Prove the distance |AB| = 3√2 where A and B are the intersection points of line l with curves C1 and C2 respectively.
theorem distance_AB :
  (exists t, line_l_x t = 1 ∧ line_l_y t = 1) ∧
  (exists t, line_l_x t = 4 ∧ line_l_y t = 4) →
  dist (1, 1) (4, 4) = 3 * real.sqrt 2 :=
sorry

end

end polar_coordinate_C1_distance_AB_l169_169767


namespace number_of_different_lines_l169_169675

theorem number_of_different_lines (a b : ℕ) (h : a ∈ {0, 1, 2, 3} ∧ b ∈ {0, 1, 2, 3}) :
  {d : ℕ // d = 9} ≠ ∅ :=
begin
    -- Note: This is a placeholder statement to illustrate the problem requirements.
    sorry
end

end number_of_different_lines_l169_169675


namespace correct_sample_size_l169_169578

variable {StudentScore : Type} {scores : Finset StudentScore} (extract_sample : Finset StudentScore → Finset StudentScore)

noncomputable def is_correct_statement : Prop :=
  ∀ (total_scores : Finset StudentScore) (sample_scores : Finset StudentScore),
  (total_scores.card = 1000) →
  (extract_sample total_scores = sample_scores) →
  (sample_scores.card = 100) →
  sample_scores.card = 100

theorem correct_sample_size (total_scores sample_scores : Finset StudentScore)
  (H_total : total_scores.card = 1000)
  (H_sample : extract_sample total_scores = sample_scores)
  (H_card : sample_scores.card = 100) :
  sample_scores.card = 100 :=
sorry

end correct_sample_size_l169_169578


namespace eggs_leftover_l169_169608

theorem eggs_leftover (eggs_abigail eggs_beatrice eggs_carson cartons : ℕ)
  (h_abigail : eggs_abigail = 37)
  (h_beatrice : eggs_beatrice = 49)
  (h_carson : eggs_carson = 14)
  (h_cartons : cartons = 12) :
  ((eggs_abigail + eggs_beatrice + eggs_carson) % cartons) = 4 :=
by
  sorry

end eggs_leftover_l169_169608


namespace double_acute_angle_l169_169012

theorem double_acute_angle (α : ℝ) (h : 0 < α ∧ α < π / 2) : 0 < 2 * α ∧ 2 * α < π :=
by
  sorry

end double_acute_angle_l169_169012


namespace vector_difference_magnitude_l169_169803

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def unit_vectors (a b : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 1

theorem vector_difference_magnitude (a b : V)
  (h : unit_vectors a b) : ∥a - b∥ = real.sqrt 3 :=
by sorry

end vector_difference_magnitude_l169_169803


namespace subsidy_fund_calculation_determine_m_l169_169475

-- Problem 1
def sales_2008 : ℝ := 1.6 * 10^9 -- total sales in yuan
def subsidy_rate : ℝ := 0.13 -- subsidy rate of 13%
def expected_subsidy_funds : ℝ := 2.08 * 10^8 -- expected subsidy funds in yuan

theorem subsidy_fund_calculation :
  sales_2008 * subsidy_rate = expected_subsidy_funds := by
  sorry

-- Problem 2
def total_jobs_2008_2010 : ℕ := 247000 -- total jobs created from 2008 to 2010
def jobs_2010_increase_ratio : ℝ := 10 / 81 -- increase ratio of jobs in 2010 compared to 2009
def jobs_2008 : ℕ := 75000 -- jobs created in 2008 for 1.7 percentage point increase
def percentage_increase_2008 : ℝ := 1.7 -- percentage point increase in 2008
def percentage_increase_2010 : ℝ := 0.5 -- percentage point increase in 2010 compared to 2009
def expected_m : ℝ := 20000 -- expected value of m

theorem determine_m :
  (total_jobs_2008_2010 - jobs_2008 * 3 - 
  (jobs_2008 * (1 + jobs_2010_increase_ratio) + jobs_2008)) / 
  (percentage_increase_2010 * 100) = expected_m := by
  sorry

end subsidy_fund_calculation_determine_m_l169_169475


namespace lim_p_over_q_at_1_l169_169111

noncomputable def p (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3) * (x - 4)

noncomputable def q (x : ℝ) : ℝ := (x - 1) * (x - 1/2) * (x - 1/3) * (x - 1/4)

theorem lim_p_over_q_at_1 : 
  (∀ x : ℝ, p x ≠ 0 → q x ≠ 0 → p x = (x - 1) * (x - 2) * (x - 3) * (x - 4)) ∧ 
  (∀ x : ℝ, p x ≠ 0 → q x ≠ 0 → q x = (x - 1) * (x - 1/2) * (x - 1/3) * (x - 1/4)) →
  (∃ L : ℝ, Filter.Tendsto (λ x, (p x) / (q x)) (nhds 1) (nhds L) ∧ L = -24) :=
by
  intro h
  sorry

end lim_p_over_q_at_1_l169_169111


namespace prove_math_problem_l169_169420

noncomputable def ellipse_foci : Prop := 
  ∃ (a b : ℝ), 
  a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ),
  (x^2 / a^2 + y^2 / b^2 = 1) → 
  a = 2 ∧ b^2 = 3)

noncomputable def intersect_and_rhombus : Prop :=
  ∃ (m : ℝ) (t : ℝ),
  (3 * m^2 + 4) > 0 ∧ 
  t = 1 / (3 * m^2 + 4) ∧ 
  0 < t ∧ t < 1 / 4

theorem prove_math_problem : ellipse_foci ∧ intersect_and_rhombus :=
by sorry

end prove_math_problem_l169_169420


namespace Tara_loss_point_l169_169480

theorem Tara_loss_point :
  ∀ (clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal) 
  (H1 : initial_savings = 10)
  (H2 : clarinet_cost = 90)
  (H3 : book_price = 5)
  (H4 : total_books_sold = 25)
  (H5 : books_sold_to_goal = (clarinet_cost - initial_savings) / book_price)
  (H6 : additional_books = total_books_sold - books_sold_to_goal),
  additional_books * book_price = 45 :=
by
  intros clarinet_cost initial_savings book_price total_books_sold additional_books books_sold_to_goal
  intros H1 H2 H3 H4 H5 H6
  sorry

end Tara_loss_point_l169_169480


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169967

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169967


namespace magnitude_diff_unit_vectors_l169_169808

variables (a b : ℝ^3) -- Define vector variables a and b in 3-dimensional real space

-- Define the conditions
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def sum_is_one (v1 v2 : ℝ^3) : Prop := ‖v1 + v2‖ = 1

-- State the theorem
theorem magnitude_diff_unit_vectors (a b : ℝ^3) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h_sum : sum_is_one a b) : 
  ‖a - b‖ = real.sqrt 3 :=
sorry

end magnitude_diff_unit_vectors_l169_169808


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169968

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169968


namespace combined_area_is_256_l169_169602

-- Define the conditions
def side_length : ℝ := 16
def area_square : ℝ := side_length ^ 2

-- Define the property of the sides r and s
def r_s_property (r s : ℝ) : Prop :=
  (r + s)^2 + (r - s)^2 = side_length^2

-- The combined area of the four triangles
def combined_area_of_triangles (r s : ℝ) : ℝ :=
  2 * (r ^ 2 + s ^ 2)

-- Prove the final statement
theorem combined_area_is_256 (r s : ℝ) (h : r_s_property r s) :
  combined_area_of_triangles r s = 256 := by
  sorry

end combined_area_is_256_l169_169602


namespace original_volume_of_wooden_block_l169_169274

theorem original_volume_of_wooden_block
  (l : ℝ) (a b : ℝ)
  (num_sections : ℕ)
  (area_increase : ℝ)
  (volume_orig : ℝ) :
  l = 100 ∧
  num_sections = 6 ∧
  area_increase = 100 ∧
  (area a b = l * a * b) ∧
  volume_orig = 1000 :=
by
  sorry

end original_volume_of_wooden_block_l169_169274


namespace find_g_2002_l169_169364

noncomputable def f : ℝ → ℝ := sorry

noncomputable def g (x : ℝ) : ℝ := f(x) + 1 - x

theorem find_g_2002 : f 1 = 1 →
  (∀ x, f(x + 5) ≥ f(x) + 5) →
  (∀ x, f(x + 1) ≤ f(x) + 1) →
  g 2002 = 1 :=
begin
  sorry
end

end find_g_2002_l169_169364


namespace weight_of_piece_l169_169863

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end weight_of_piece_l169_169863


namespace find_length_of_AC_find_cos_acute_l169_169019

-- Definitions based on conditions
def AB : ℝ := 6
def BC : ℝ := 5
def area_ABC : ℝ := 9
def AC1 : ℝ := Real.sqrt 13
def AC2 : ℝ := Real.sqrt 109

-- Statement 1: Calculate AC
theorem find_length_of_AC (h1 : AB = 6) (h2 : BC = 5) (h3 : 1 / 2 * AB * BC * Real.sin (angleABC AB BC) = 9) :
    AC = AC1 ∨ AC = AC2 :=
sorry

-- Given additional condition that triangle is acute
theorem find_cos_acute (h1 : AB = 6) (h2 : BC = 5) (h4 : AC = AC1) :
    Real.cos (2 * angleA AB AC BC + Real.pi / 6) = (-5 * Real.sqrt 3 - 12) / 26 :=
sorry

end find_length_of_AC_find_cos_acute_l169_169019


namespace average_price_of_tshirts_l169_169862

theorem average_price_of_tshirts
  (A : ℝ)
  (total_cost_seven_remaining : ℝ := 7 * 505)
  (total_cost_three_returned : ℝ := 3 * 673)
  (total_cost_eight : ℝ := total_cost_seven_remaining + 673) -- since (1 t-shirt with price is included in the total)
  (total_cost_eight_eq : total_cost_eight = 8 * A) :
  A = 526 :=
by sorry

end average_price_of_tshirts_l169_169862


namespace solve_for_k_l169_169721

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end solve_for_k_l169_169721


namespace area_quadrilateral_ABCD_l169_169472

-- Define points and distances in the geometrical setting.
structure Point :=
  (x : ℝ)
  (y : ℝ)

def A : Point := ⟨0, 0⟩
def B : Point := ⟨13, 0⟩
def C : Point := ⟨5, 12⟩
def D : Point := ⟨0, 12⟩

-- Definitions of distances based on given conditions.
def distance (p1 p2 : Point) : ℝ := 
  real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

-- Conditions
def AB_eq_13 : Prop := distance A B = 13
def AC_eq_13 : Prop := distance A C = 13
def BC_eq_10 : Prop := distance B C = 10
def CD_eq_5 : Prop := distance C D = 5
def AD_eq_12 : Prop := distance A D = 12

-- Define quadrilateral area calculation (sum of areas of triangles)
def area_triangle (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

def area_ABCD : ℝ :=
  area_triangle A B C + area_triangle A D C

-- Lean statement to prove the total area
theorem area_quadrilateral_ABCD : 
  AB_eq_13 ∧ AC_eq_13 ∧ BC_eq_10 ∧ CD_eq_5 ∧ AD_eq_12 → area_ABCD = 90 := 
by
  sorry

end area_quadrilateral_ABCD_l169_169472


namespace interior_angle_of_regular_heptagon_l169_169203

-- Define the problem statement in Lean
theorem interior_angle_of_regular_heptagon : 
  let n := 7 in (n - 2) * 180 / n = 900 / 7 := 
by 
  let n := 7
  show (n - 2) * 180 / n = 900 / 7
  sorry

end interior_angle_of_regular_heptagon_l169_169203


namespace magnitude_difference_sqrt3_l169_169815

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Condition 1: a and b are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∥v∥ = 1

-- Condition 2: |a + b| = 1
def sum_of_unit_vectors_has_unit_norm : Prop :=
  ∥a + b∥ = 1

-- Question: |a - b| = sqrt(3)
theorem magnitude_difference_sqrt3 (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hab : sum_of_unit_vectors_has_unit_norm a b) :
  ∥a - b∥ = Real.sqrt 3 :=
sorry

end magnitude_difference_sqrt3_l169_169815


namespace min_max_weight_147_l169_169424

theorem min_max_weight_147 : 
  ∃ (weights : Fin 20 → ℕ), 
    (∀ i j, i < j → weights i ≤ weights j) ∧ 
    (∀ k : ℕ, 1 ≤ k ∧ k ≤ 2014 → 
      ∃ (s : Finset (Fin 20)), k = s.sum (λ i, weights i)) ∧ 
    (weights 19 = 147) :=
sorry

end min_max_weight_147_l169_169424


namespace abs_diff_squares_l169_169939

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169939


namespace jerry_showers_l169_169300

theorem jerry_showers :
  ∀ gallons_total gallons_drinking_cooking gallons_per_shower pool_length pool_width pool_height gallons_per_cubic_foot,
    gallons_total = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    pool_length = 10 →
    pool_width = 10 →
    pool_height = 6 →
    gallons_per_cubic_foot = 1 →
    let pool_volume := pool_length * pool_width * pool_height in
    pool_volume = 600 →
    let gallons_for_pool := pool_volume * gallons_per_cubic_foot in
    let gallons_for_showers := gallons_total - gallons_drinking_cooking - gallons_for_pool in
    let number_of_showers := gallons_for_showers / gallons_per_shower in
    number_of_showers = 15 :=
by
  intros
  sorry

end jerry_showers_l169_169300


namespace log_inequality_solution_set_l169_169214

theorem log_inequality_solution_set (a x : ℝ) (h1 : x < 0) (h2 : a > 0) (h3 : a ≠ 1) (h4 : a^x > 1) :
  (∀ x, log a x > 0 → 0 < x ∧ x < 1) :=
sorry

end log_inequality_solution_set_l169_169214


namespace slippers_total_cost_l169_169841

def slippers_original_cost : ℝ := 50.00
def slipper_discount_rate : ℝ := 0.10
def embroidery_cost_per_shoe : ℝ := 5.50
def shipping_cost : ℝ := 10.00
def number_of_shoes : ℕ := 2

theorem slippers_total_cost :
  let discount := slippers_original_cost * slipper_discount_rate,
      slippers_cost_after_discount := slippers_original_cost - discount,
      embroidery_cost := embroidery_cost_per_shoe * number_of_shoes,
      total_cost := slippers_cost_after_discount + embroidery_cost + shipping_cost
  in total_cost = 66.00 := by
  sorry

end slippers_total_cost_l169_169841


namespace probability_same_number_of_flips_l169_169268

theorem probability_same_number_of_flips (p_head : ℝ) (p_tail : ℝ) :
  (p_head = (1 / 3)) → (p_tail = (2 / 3)) →
  (∑ n in (set.Icc 1 (99 : ℕ)), ((p_tail) ^ (n - 1) * p_head) ^ 4) = (81 / 65) :=
by
  -- Given the conditions that p_head = 1/3 and p_tail = 2/3,
  assume h1 : p_head = 1 / 3,
  assume h2 : p_tail = 2 / 3,
  sorry

end probability_same_number_of_flips_l169_169268


namespace smallest_number_among_10_11_12_l169_169513

theorem smallest_number_among_10_11_12 : min (min 10 11) 12 = 10 :=
by sorry

end smallest_number_among_10_11_12_l169_169513


namespace value_of_expression_l169_169325

def star (x y : ℝ) (h : x ≠ y) : ℝ := (x + y) / (x - y)

theorem value_of_expression : ((star (-2) (1/2) (by norm_num)) ▸ star (3/5) (-3/4) (by norm_num)) = -1/9 := 
by sorry

end value_of_expression_l169_169325


namespace vector_problem_l169_169372

variables (a b : ℝ^3)

-- The given conditions
def condition1 : Prop := ‖a‖ = 4
def condition2 : Prop := ‖b‖ = 3
def condition3 : Prop := (2 • a - 3 • b) • (2 • a + b) = 61

-- The conclusions to be proved
def conclusion1 : Prop := a • b = -6
def conclusion2 : Prop := ‖a - 2 • b‖ = 2 * Real.sqrt 7

-- The theorem statement
theorem vector_problem (h1 : condition1 a) (h2 : condition2 b) (h3 : condition3 a b) : conclusion1 a b ∧ conclusion2 a b := by
  sorry

end vector_problem_l169_169372


namespace box_volume_correct_l169_169150

variables (length width height : ℕ)

def volume_of_box (length width height : ℕ) : ℕ :=
  length * width * height

theorem box_volume_correct :
  volume_of_box 20 15 10 = 3000 :=
by
  -- This is where the proof would go
  sorry 

end box_volume_correct_l169_169150


namespace regular_heptagon_interior_angle_l169_169197

theorem regular_heptagon_interior_angle :
  ∀ (n : ℕ), n = 7 → (∑ i in finset.range n, 180 / n) = 128.57 :=
  by
  intros n hn
  rw hn
  sorry

end regular_heptagon_interior_angle_l169_169197


namespace plot_width_l169_169251

theorem plot_width (cost_per_sq_meter : ℝ) (total_cost : ℝ) (path_width : ℝ) (plot_length : ℝ) (w : ℝ) : 
  cost_per_sq_meter = 0.4 ∧ total_cost = 340 ∧ path_width = 2.5 ∧ plot_length = 110 →
  let total_length := plot_length + 2 * path_width in
  let total_width := w + 2 * path_width in
  let total_area := total_length * total_width in
  let plot_area := plot_length * w in
  let path_area := total_area - plot_area in
  let cost := path_area * cost_per_sq_meter in
  cost = total_cost →
  w = 55 := 
by
  intros h hc,
  sorry

end plot_width_l169_169251


namespace expression_evaluation_l169_169278

noncomputable def compute_expression : ℝ :=
  ∏ i in (Finset.range 216).filter (λ x, x % 2 = 0) \ 
                ((Finset.image (λ x, x ^ 3) ((Finset.range 6).image (λ x, x.succ.succ)).filter (λ x, x ≤ 216))),
  (⌊real.cbrt (i - 1)⌋ / ⌊real.cbrt i⌋)

theorem expression_evaluation : compute_expression = 1 / 6 :=
by
  sorry

end expression_evaluation_l169_169278


namespace odd_factors_of_420_l169_169731

/-- 
Proof problem: Given that 420 can be factorized as \( 2^2 \times 3 \times 5 \times 7 \), 
we need to prove that the number of odd factors of 420 is 8.
-/
def number_of_odd_factors (n : ℕ) : ℕ :=
  let odd_factors := n.factors.filter (λ x, x % 2 ≠ 0)
  in odd_factors.length

theorem odd_factors_of_420 : number_of_odd_factors 420 = 8 := 
sorry

end odd_factors_of_420_l169_169731


namespace twelve_million_plus_twelve_thousand_l169_169520

theorem twelve_million_plus_twelve_thousand :
  12000000 + 12000 = 12012000 :=
by
  sorry

end twelve_million_plus_twelve_thousand_l169_169520


namespace number_of_monic_polynomials_l169_169631

noncomputable def num_of_monic_polynomials_satisfying_condition : Nat :=
  119

theorem number_of_monic_polynomials
  (q : Polynomial ℤ)
  (hq1 : Polynomial.Monic q)
  (hq2 : q.degree = 12)
  (h_cond : ∃ p : Polynomial ℤ, q * p = Polynomial.eval₂ Polynomial.C Polynomial.X q Polynomial.X^2) :
  ∃ n : Nat, n = num_of_monic_polynomials_satisfying_condition :=  
sorry

end number_of_monic_polynomials_l169_169631


namespace analysis_method_correct_answer_l169_169271

axiom analysis_def (conclusion: Prop): 
  ∃ sufficient_conditions: (Prop → Prop), 
    (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)

theorem analysis_method_correct_answer :
  ∀ (conclusion : Prop) , ∃ sufficient_conditions: (Prop → Prop), 
  (∀ proof_conclusion: Prop, proof_conclusion = conclusion → sufficient_conditions proof_conclusion)
:= by 
  intros 
  sorry

end analysis_method_correct_answer_l169_169271


namespace coefficient_of_one_over_x_in_expansion_l169_169763

theorem coefficient_of_one_over_x_in_expansion : 
  (coeff_of_term (x - (2 / x))^7 (-1)) = 560 :=
sorry

end coefficient_of_one_over_x_in_expansion_l169_169763


namespace probability_half_correct_l169_169873

noncomputable def probability_at_least_half_correct : ℚ :=
  ∑ k in Finset.range (20 - 10 + 1), (Nat.choose 20 (10 + k)) * (1/2)^(20)

theorem probability_half_correct (n : ℕ) (p : ℚ) (N : ℕ)
  (h_n : n = 20) (h_p : p = 1/2) (h_N : N = 10) :
  probability_at_least_half_correct = 1/2 :=
by
  have eq : (∑ k in Finset.range (n - N + 1), (Nat.choose n (N + k)) * (p)^(n)) = 1/2 :=
    sorry
  exact eq

end probability_half_correct_l169_169873


namespace abs_diff_squares_105_95_l169_169933

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l169_169933


namespace first_four_same_as_last_four_l169_169584

variable {α : Type*} [LinearOrder α] [AddGroup α] [DecidableEq α]

def unique_length_5 (S : List α) : Prop :=
  ∀ (i j : ℕ), i < j → i + 5 ≤ S.length → j + 5 ≤ S.length → 
  (S.drop i).take 5 ≠ (S.drop j).take 5

def extend_breaks_uniqueness (S : List α) : Prop :=
  ¬ unique_length_5 (0 :: S) ∧ ¬ unique_length_5 (S ++ [0])

theorem first_four_same_as_last_four (S : List α) :
  unique_length_5 S ∧ extend_breaks_uniqueness S →
  (S.take 4 = S.drop (S.length - 4)).take 4 :=
by
  sorry

end first_four_same_as_last_four_l169_169584


namespace seq_2017_l169_169036

-- Define the sequence recursively
def seq : ℕ → ℚ
| 0     := 2
| (n+1) := 1 - 1 / seq n

-- The proof statement
theorem seq_2017 : seq 2016 = 2 :=
by
  sorry

end seq_2017_l169_169036


namespace total_alphabets_written_l169_169617

-- Define the number of vowels and the number of times each is written
def num_vowels : ℕ := 5
def repetitions : ℕ := 4

-- The theorem stating the total number of alphabets written on the board
theorem total_alphabets_written : num_vowels * repetitions = 20 := by
  sorry

end total_alphabets_written_l169_169617


namespace abs_diff_squares_l169_169940

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169940


namespace cost_to_paint_new_room_l169_169919

def wall_area (height: ℝ) (width: ℝ) : ℝ := height * width
def window_area (height: ℝ) (width: ℝ) : ℝ := height * width
def door_area (height: ℝ) (width: ℝ) : ℝ := height * width

def total_painted_area (orig_walls: List (ℝ × ℝ)) (num_windows: ℕ) (window_dim: ℝ × ℝ)
  (num_doors: ℕ) (door_dim: ℝ × ℝ) (scale_factor: ℝ) (new_num_windows: ℕ) (new_num_doors: ℕ) : ℝ := 
  let orig_wall_area := orig_walls.map (λ d => wall_area (d.1) (d.2)).sum
  let orig_window_area := num_windows * window_area window_dim.1 window_dim.2
  let orig_door_area := num_doors * door_area door_dim.1 door_dim.2
  let net_orig_wall_area := orig_wall_area - (orig_window_area + orig_door_area)

  let new_wall_area := orig_walls.map (λ d => wall_area (d.1 * scale_factor) (d.2 * scale_factor)).sum
  let new_window_area := new_num_windows * window_area window_dim.1 window_dim.2
  let new_door_area := new_num_doors * door_area door_dim.1 door_dim.2
  let net_new_wall_area := new_wall_area - (new_window_area + new_door_area + orig_window_area + orig_door_area)
  net_new_wall_area

theorem cost_to_paint_new_room :
  let walls := [(10, 8), (10, 9), (15, 8), (15, 9)] in
  let num_windows := 2 in
  let window_dim := (3, 4) in
  let num_doors := 1 in
  let door_dim := (3, 7) in
  let cost_per_sqft := 2 in
  let scale_factor := 1.5 in
  let new_num_windows := 4 in
  let new_num_doors := 2 in
  total_painted_area walls num_windows window_dim num_doors door_dim scale_factor new_num_windows new_num_doors * cost_per_sqft = 1732.50 := sorry

end cost_to_paint_new_room_l169_169919


namespace number_is_composite_all_n_l169_169161

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

def construct_number (n : ℕ) : ℕ :=
  200 * 10^n + 88 * ((10^n - 1) / 9) + 21

theorem number_is_composite_all_n (n : ℕ) :
  let N := construct_number n in
  is_composite N := 
by
  sorry

end number_is_composite_all_n_l169_169161


namespace abs_diff_squares_105_95_l169_169935

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l169_169935


namespace three_bodies_with_triangle_front_view_l169_169587

def has_triangle_front_view (b : Type) : Prop :=
  -- Placeholder definition for example purposes
  sorry

theorem three_bodies_with_triangle_front_view :
  ∃ (body1 body2 body3 : Type),
  has_triangle_front_view body1 ∧
  has_triangle_front_view body2 ∧
  has_triangle_front_view body3 :=
sorry

end three_bodies_with_triangle_front_view_l169_169587


namespace two_marbles_different_colors_probability_l169_169242

-- Definitions
def red_marbles : Nat := 3
def green_marbles : Nat := 4
def white_marbles : Nat := 5
def blue_marbles : Nat := 3
def total_marbles : Nat := red_marbles + green_marbles + white_marbles + blue_marbles

-- Combinations of different colored marbles
def red_green : Nat := red_marbles * green_marbles
def red_white : Nat := red_marbles * white_marbles
def red_blue : Nat := red_marbles * blue_marbles
def green_white : Nat := green_marbles * white_marbles
def green_blue : Nat := green_marbles * blue_marbles
def white_blue : Nat := white_marbles * blue_marbles

-- Total favorable outcomes
def total_favorable : Nat := red_green + red_white + red_blue + green_white + green_blue + white_blue

-- Total outcomes when drawing 2 marbles from the jar
def total_outcomes : Nat := Nat.choose total_marbles 2

-- Probability calculation
noncomputable def probability_different_colors : Rat := total_favorable / total_outcomes

-- Proof that the probability is 83/105
theorem two_marbles_different_colors_probability :
  probability_different_colors = 83 / 105 := by
  sorry

end two_marbles_different_colors_probability_l169_169242


namespace find_value_of_m_l169_169114

/-- Given the universal set U, set A, and the complement of A in U, we prove that m = -2. -/
theorem find_value_of_m (m : ℤ) (U : Set ℤ) (A : Set ℤ) (complement_U_A : Set ℤ) 
  (h1 : U = {2, 3, m^2 + m - 4})
  (h2 : A = {m, 2})
  (h3 : complement_U_A = {3}) 
  (h4 : U = A ∪ complement_U_A) 
  (h5 : A ∩ complement_U_A = ∅) 
  : m = -2 :=
sorry

end find_value_of_m_l169_169114


namespace restricted_students_arrange_separate_l169_169409

def students : Type := Fin 5 -- Define a type for the five students

def restricted_students : Finset (Fin 5) := {0, 1, 2} -- Define the three specific students who refuse to stand next to each other

noncomputable def number_of_arrangements_without_restriction : ℕ := 5.factorial

noncomputable def number_of_restricted_arrangements : ℕ :=
  let any_two_together := choose 3 2 * 4.factorial * 2.factorial in
  let all_three_together := 3.factorial * 3.factorial in
  any_two_together - all_three_together

theorem restricted_students_arrange_separate : 
  (number_of_arrangements_without_restriction - number_of_restricted_arrangements) = 12 := 
sorry -- Proof omitted

end restricted_students_arrange_separate_l169_169409


namespace certain_number_is_47_l169_169055

theorem certain_number_is_47 (x : ℤ) (h : 34 + x - 53 = 28) : x = 47 :=
by
  sorry

end certain_number_is_47_l169_169055


namespace slope_of_line_l169_169714

theorem slope_of_line
  (k : ℝ) 
  (hk : 0 < k) 
  (h1 : ¬ (2 / Real.sqrt (k^2 + 1) = 3 * 2 * Real.sqrt (1 - 8 * k^2) / Real.sqrt (k^2 + 1))) 
  : k = 1 / 3 :=
sorry

end slope_of_line_l169_169714


namespace isosceles_triangle_perimeter_l169_169407

-- Define the perimeter of an isosceles triangle where two sides are 3 and 4
noncomputable def isosceles_triangle_possible_perimeters : Set ℕ :=
  {10, 11}

-- Definition of a generic isosceles triangle
structure IsoscelesTriangle (a b : ℕ) :=
  (equal_sides : a ∈ {3, 4})
  (base : b ∈ {3, 4})
  (isosceles : a ≠ b)

-- The theorem stating the possible perimeters
theorem isosceles_triangle_perimeter (a b : ℕ) (h : IsoscelesTriangle a b) : 
  (a + b + b ∈ isosceles_triangle_possible_perimeters) ∨ (a + a + b ∈ isosceles_triangle_possible_perimeters) := by
  cases h with
  | intro equal_sides base isosceles =>
    sorry

end isosceles_triangle_perimeter_l169_169407


namespace distance_center_to_line_l169_169296

def circle_center : ℝ × ℝ := (-2, 2)
def line_eq (x y : ℝ) : Prop := x - y + 3 = 0

theorem distance_center_to_line : 
  let a := 1
  let b := -1
  let c := 3
  let x0 := -2
  let y0 := 2
  let d := (|a * x0 + b * y0 + c|) / (real.sqrt (a^2 + b^2))
  d = (3 * real.sqrt 2) / 2 :=
by
  -- Proof to be provided
  sorry

end distance_center_to_line_l169_169296


namespace abs_diff_squares_105_95_l169_169943

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l169_169943


namespace solve_quadratic_equation_l169_169140

theorem solve_quadratic_equation (x : ℝ) : 4 * (2 * x + 1) ^ 2 = 9 * (x - 3) ^ 2 ↔ x = -11 ∨ x = 1 := 
by sorry

end solve_quadratic_equation_l169_169140


namespace length_HD_is_3_l169_169765

noncomputable def square_side : ℝ := 8

noncomputable def midpoint_AD : ℝ := square_side / 2

noncomputable def length_FD : ℝ := midpoint_AD

theorem length_HD_is_3 :
  ∃ (x : ℝ), 0 < x ∧ x < square_side ∧ (8 - x) ^ 2 = x ^ 2 + length_FD ^ 2 ∧ x = 3 :=
by
  sorry

end length_HD_is_3_l169_169765


namespace magnitude_diff_unit_vectors_l169_169809

variables (a b : ℝ^3) -- Define vector variables a and b in 3-dimensional real space

-- Define the conditions
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def sum_is_one (v1 v2 : ℝ^3) : Prop := ‖v1 + v2‖ = 1

-- State the theorem
theorem magnitude_diff_unit_vectors (a b : ℝ^3) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h_sum : sum_is_one a b) : 
  ‖a - b‖ = real.sqrt 3 :=
sorry

end magnitude_diff_unit_vectors_l169_169809


namespace largest_number_of_acute_angles_in_convex_octagon_l169_169211

theorem largest_number_of_acute_angles_in_convex_octagon :
  ∀ (angles : Fin 8 → ℝ), (∀ i, angles i < 90) → (∑ i, angles i = 1080) → 
  ∃ (acute_count : ℕ), acute_count ≤ 4 :=
by sorry

end largest_number_of_acute_angles_in_convex_octagon_l169_169211


namespace term_2018_is_256_l169_169569

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def a : ℕ → ℕ
| 0     := 2018
| (n+1) := (sum_of_digits (a n))^2

theorem term_2018_is_256 : a 2018 = 256 :=
by sorry

end term_2018_is_256_l169_169569


namespace not_jog_probability_eq_l169_169903

def P_jog : ℚ := 5 / 8

theorem not_jog_probability_eq :
  1 - P_jog = 3 / 8 :=
by
  sorry

end not_jog_probability_eq_l169_169903


namespace number_of_correct_conclusions_l169_169327

noncomputable def A (x : ℝ) : ℝ := 2 * x^2
noncomputable def B (x : ℝ) : ℝ := x + 1
noncomputable def C (x : ℝ) : ℝ := -2 * x
noncomputable def D (y : ℝ) : ℝ := y^2
noncomputable def E (x y : ℝ) : ℝ := 2 * x - y

def conclusion1 (y : ℤ) : Prop := 
  0 < ((B (0 : ℝ)) * (C (0 : ℝ)) + A (0 : ℝ) + D y + E (0) (y : ℝ))

def conclusion2 : Prop := 
  ∃ (x y : ℝ), A x + D y + 2 * E x y = -2

def M (A B C : ℝ → ℝ) (x m : ℝ) : ℝ :=
  3 * (A x - B x) + m * B x * C x

def linear_term_exists (m : ℝ) : Prop :=
  (0 : ℝ) ≠ -3 - 2 * m

def conclusion3 : Prop := 
 ∀ m : ℝ, (¬ linear_term_exists m ∧ M A B C (0 : ℝ) m > -3) 

def p (x y : ℝ) := 
  2 * (x + 1) ^ 2 + (y - 1) ^ 2 = 1

theorem number_of_correct_conclusions : Prop := 
  (¬ conclusion1 1) ∧ (conclusion2) ∧ (¬ conclusion3)

end number_of_correct_conclusions_l169_169327


namespace inverse_B_P_sq_l169_169379

open Matrix

noncomputable def B_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![3, 7], ![-2, -4]]

noncomputable def P : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![1, 0], ![0, 2]]

theorem inverse_B_P_sq :
  let B := B_inv⁻¹
  let BP := B ⬝ P
  (BP * BP)⁻¹ = ![![8, 28], ![-4, -12]] :=
by
  let B := B_inv⁻¹
  let BP := B ⬝ P
  have h : (BP * BP)⁻¹ = ![![8, 28], ![-4, -12]], from sorry
  exact h

end inverse_B_P_sq_l169_169379


namespace triangle_ABC_BC_length_l169_169392

theorem triangle_ABC_BC_length :
  ∀ {A B C X : Type} [has_dist A B] [has_dist A C] [has_dist B X] [has_dist C X] [has_dist B C],
  has_dist.dist A B = 75 →
  has_dist.dist A C = 85 →
  (∀ (P : Type), ∃ (H : ∃ {r : ℝ}, has_dist.dist A P = r), P = B ∨ P = X) →
  (∃ (d1 d2 : ℕ), is_integer d1 ∧ is_integer d2 ∧ d1 + d2 = has_dist.dist B C) →
  has_dist.dist B C = 89 :=
by
  intros A B C X hAB hAC intersec int_lengths
  sorry

end triangle_ABC_BC_length_l169_169392


namespace walking_speed_l169_169593

theorem walking_speed (run_speed : ℝ) (total_time : ℝ) (half_distance : ℝ) (walk_speed : ℝ) : 
  run_speed = 8 → 
  total_time = 1.5 → 
  half_distance = 4 → 
  (half_distance / walk_speed) + (half_distance / run_speed) = total_time → 
  walk_speed = 4 :=
by 
  intros h_run_speed h_total_time h_half_distance h_equation
  have h1 : half_distance / run_speed = 1 / 2, from (by rw [h_half_distance, h_run_speed]; exact rfl),
  rw [h1] at h_equation,
  linarith,
  sorry

end walking_speed_l169_169593


namespace length_of_perpendicular_l169_169690

theorem length_of_perpendicular 
(DEF : Triangle)
(right_triangle : is_right_triangle DEF)
(DE : DE.length = 5)
(EF : EF.length = 12) : 
  length_of_perpendicular_from_hypotenuse_to_midpoint_angle_bisector DEF = 5 := 
sorry

end length_of_perpendicular_l169_169690


namespace abs_diff_squares_l169_169954

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169954


namespace chess_tournament_exists_l169_169448

theorem chess_tournament_exists (n : ℕ) (h_pos_n : n > 0) 
  (t : Fin n → ℕ) (h_strictly_increasing : ∀ i j, i < j → t i < t j) 
  (h_all_positive : ∀ i, t i > 0) :
  ∃ (games_played : Fin (t (Fin.last n) + 1) → ℕ), 
    (∀ p, ∃ i, games_played p = t i) ∧ 
    (∀ i, ∃ p, games_played p = t i) :=
sorry

end chess_tournament_exists_l169_169448


namespace largest_in_set_l169_169440

variable (a b : ℤ)

theorem largest_in_set (ha : a = -3) (hb : b = 2) : 
  let s := {-5 * a, 3 * b, 36 / a, a^2 + b, -1}
  in -5 * a ∈ s ∧ ∀ x ∈ s, x ≤ -5 * a := by
  sorry

end largest_in_set_l169_169440


namespace sum_pn_2009_l169_169449

noncomputable def p : ℕ → (ℝ → ℝ)
| 0       := id
| (n + 1) := λ x, ∫ t in 0..x, p n t

def P (x : ℝ) : ℝ :=
if h : ∃ n : ℕ, n ≤ x ∧ x ≤ n + 1 then p (classical.some h) x else 0

theorem sum_pn_2009 :
  continuous_on P (Ici 0) →
  (∑' n, p n 2009) = real.exp 2010 - real.exp 2009 - 1 :=
sorry

end sum_pn_2009_l169_169449


namespace area_of_R_in_rhombus_l169_169859

theorem area_of_R_in_rhombus :
  let ABCD : Type := ℝ×ℝ×ℝ×ℝ
  in ∀ (A B C D : ABCD)
     (side_length : ℝ)
     (angle_B : ℝ)
     (R : set (ℝ × ℝ)),
     side_length = 3 →
     angle_B = 90 →
     (forall P ∈ R, dist P B < dist P A ∧ dist P B < dist P D) →
     -- Proving the area of the region R is equal to 9π/16
     (area R = (9 * π) / 16) :=
begin
  sorry
end

end area_of_R_in_rhombus_l169_169859


namespace digit_150_of_3_div_11_l169_169537

theorem digit_150_of_3_div_11 :
  let rep := "27"
  ∃ seq : ℕ → ℕ,
  (∀ n, seq (2 * n) = 2) ∧ (∀ n, seq (2 * n + 1) = 7) ∧
  150 % 2 = 0 ∧ seq (150 - 1) = 7 :=
by
  let rep := "27"
  use (λ n : ℕ => if n % 2 = 0 then 2 else 7)
  split
  { intro n
    exact rfl }
  { intro n
    exact rfl }
  { exact rfl }
  { exact rfl }

end digit_150_of_3_div_11_l169_169537


namespace abs_diff_squares_105_95_l169_169950

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l169_169950


namespace find_x_l169_169383

def operation (a b : ℝ) : ℝ := a * b^(1/2)

theorem find_x (x : ℝ) : operation x 9 = 12 → x = 4 :=
by
  intro h
  sorry

end find_x_l169_169383


namespace collinear_vectors_y_l169_169749

theorem collinear_vectors_y (y : ℝ) : (∃ k : ℝ, (2, 3) = k • (-4, y)) → y = -6 :=
by
  intro h
  sorry

end collinear_vectors_y_l169_169749


namespace number_of_solutions_l169_169733

open Real

theorem number_of_solutions :
  ∀ x : ℝ, (0 < x ∧ x < 3 * π) → (3 * cos x ^ 2 + 2 * sin x ^ 2 = 2) → 
  ∃ (L : Finset ℝ), L.card = 3 ∧ ∀ y ∈ L, 0 < y ∧ y < 3 * π ∧ 3 * cos y ^ 2 + 2 * sin y ^ 2 = 2 :=
by 
  sorry

end number_of_solutions_l169_169733


namespace cyclist_average_rate_l169_169220

noncomputable def average_rate_round_trip (D : ℝ) : ℝ :=
  let time_to_travel := D / 10
  let time_to_return := D / 9
  let total_distance := 2 * D
  let total_time := time_to_travel + time_to_return
  (total_distance / total_time)

theorem cyclist_average_rate (D : ℝ) (hD : D > 0) :
  average_rate_round_trip D = 180 / 19 :=
by
  sorry

end cyclist_average_rate_l169_169220


namespace sum_S_eq_31_div_16_l169_169701

-- Definitions based on conditions
def sequence_a (n : ℕ) : ℝ := 
  match n with
  | 1 => 1
  | 2 => 1 / 2 
  | _ => 1 / (2 ^ (n-1))

def sum_S (n : ℕ) : ℝ := 
  ∑ k in Finset.range n, sequence_a (k + 1)

-- Main statement we need to prove
theorem sum_S_eq_31_div_16 : sum_S 5 = 31 / 16 :=
begin
  sorry
end

end sum_S_eq_31_div_16_l169_169701


namespace greatest_divisor_l169_169385

theorem greatest_divisor (k : ℕ) (h1 : k = 5) (h2 : k = 1) :
  ∀ k, (5^k ∣ nat.factorial 25) ↔ k ≤ 6 :=
sorry

end greatest_divisor_l169_169385


namespace projection_transform_correct_l169_169439

def projection_matrix (v : ℝ × ℝ) : (ℝ × ℝ) × (ℝ × ℝ) :=
  let n := 1 / (v.1 * v.1 + v.2 * v.2)
  ((n * v.1 * v.1, n * v.1 * v.2), (n * v.2 * v.1, n * v.2 * v.2))

theorem projection_transform_correct :
  let P1 := projection_matrix (4, 2)
  let P2 := projection_matrix (2, -1)
  let M := ((P2.1.1 * P1.1.1 + P2.1.2 * P1.2.1, P2.1.1 * P1.1.2 + P2.1.2 * P1.2.2),
            (P2.2.1 * P1.1.1 + P2.2.2 * P1.2.1, P2.2.1 * P1.1.2 + P2.2.2 * P1.2.2))
  M = ((12/25 : ℝ, 0), (0, 1/25)) :=
by
  sorry

end projection_transform_correct_l169_169439


namespace extreme_value_and_extremes_l169_169677

theorem extreme_value_and_extremes (a b : ℝ) (f : ℝ → ℝ) (f' : ℝ → ℝ)
  (h₁ : f = λ x, x^3 + 3 * a * x^2 + b * x)
  (h₂ : f' = λ x, 3 * x^2 + 6 * a * x + b)
  (h_extreme : f(-1) = 0 ∧ f'(-1) = 0)
  (a_val : a = 2 / 3)
  (b_val : b = 1) :
  (∀ x ∈ set.Icc (-2 : ℝ) (-1 / 4), f(x) ≥ -2 ∧ f(x) ≤ 0) := 
sorry

#check extreme_value_and_extremes

end extreme_value_and_extremes_l169_169677


namespace car_clock_correctness_l169_169883

variables {t_watch t_car : ℕ} 
--  Variable declarations for time on watch (accurate) and time on car clock.

-- Define the initial times at 8:00 AM
def initial_time_watch : ℕ := 8 * 60 -- 8:00 AM in minutes
def initial_time_car : ℕ := 8 * 60 -- also 8:00 AM in minutes

-- Define the known times in the afternoon
def afternoon_time_watch : ℕ := 14 * 60 -- 2:00 PM in minutes
def afternoon_time_car : ℕ := 14 * 60 + 10 -- 2:10 PM in minutes

-- Car clock runs 37 minutes in the time the watch runs 36 minutes
def car_clock_rate : ℕ × ℕ := (37, 36)

-- Check the car clock time when the accurate watch shows 10:00 PM
def car_time_at_10pm_watch : ℕ := 22 * 60 -- 10:00 PM in minutes

-- Define the actual time that we need to prove
def actual_time_at_10pm_car : ℕ := 21 * 60 + 47 -- 9:47 PM in minutes

theorem car_clock_correctness : 
  (t_watch = actual_time_at_10pm_car) ↔ 
  (t_car = car_time_at_10pm_watch) ∧ 
  (initial_time_watch = initial_time_car) ∧ 
  (afternoon_time_watch = 14 * 60) ∧ 
  (afternoon_time_car = 14 * 60 + 10) ∧ 
  (car_clock_rate = (37, 36)) :=
sorry

end car_clock_correctness_l169_169883


namespace max_trig_expression_l169_169318

theorem max_trig_expression (x : ℝ) : 
  (∃ x : ℝ, 
    (∃ t : ℝ, t = Float.cos x^2 ∧ 
      Float.sin^4 x + Float.cos^4 x + 2 = 1)) :=
sorry

end max_trig_expression_l169_169318


namespace range_of_a_for_min_f_l169_169676

def f (a x : ℝ) : ℝ :=
  if x < a then (1 - a) * x + 1 else x + 4 / x - 4

theorem range_of_a_for_min_f :
  ∀ (a : ℝ), (∃ (m : ℝ), ∀ (x : ℝ), f a x ≥ m) ↔ (1 ≤ a ∧ a ≤ (1 + Real.sqrt 5) / 2) :=
by
  sorry

end range_of_a_for_min_f_l169_169676


namespace regular_heptagon_interior_angle_l169_169201

theorem regular_heptagon_interior_angle :
  ∀ (S : Type) [decidable_instance S] [fintype S], ∀ (polygon : set S), is_regular polygon → card polygon = 7 → 
    (sum_of_interior_angles polygon / 7 = 128.57) :=
by
  intros S dec inst polygon h_reg h_card
  sorry

end regular_heptagon_interior_angle_l169_169201


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169985

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169985


namespace cube_side_length_is_30_l169_169146

theorem cube_side_length_is_30
  (cost_per_kg : ℝ) (coverage_per_kg : ℝ) (total_cost : ℝ) (s : ℝ)
  (h1 : cost_per_kg = 40)
  (h2 : coverage_per_kg = 20)
  (h3 : total_cost = 10800)
  (total_surface_area : ℝ) (W : ℝ) (C : ℝ)
  (h4 : total_surface_area = 6 * s^2)
  (h5 : W = total_surface_area / coverage_per_kg)
  (h6 : C = W * cost_per_kg)
  (h7 : C = total_cost) :
  s = 30 :=
by
  sorry

end cube_side_length_is_30_l169_169146


namespace fibonacci_sum_lt_one_l169_169908

def fib : ℕ → ℕ
| 0       => 1
| 1       => 1
| (n + 2) => fib n + fib (n + 1)

def a (n : ℕ) : ℚ := 1 / (fib n * fib (n + 2))

theorem fibonacci_sum_lt_one : ∀ m : ℕ, (finset.range (m + 1)).sum (λ i, a i) < 1 :=
by
  sorry

end fibonacci_sum_lt_one_l169_169908


namespace max_value_l169_169018

theorem max_value (x y : ℝ) (h1 : x + y = 2) (h2 : x > 0) (h3 : y > 0) : 
  x^2 + y^2 + 4 * real.sqrt (x * y) ≤ 6 := 
sorry

end max_value_l169_169018


namespace greatest_perimeter_of_pieces_l169_169431

open Real

theorem greatest_perimeter_of_pieces
  (base height : ℝ) (equal_areas_pieces : ℕ) 
  (equal_areas_pieces = 5)
  (base = 10) (height = 12) : 
  ∃ P : ℕ → ℝ, P 2 ≈ 26.82 := 
sorry

end greatest_perimeter_of_pieces_l169_169431


namespace evaluate_sum_l169_169144

theorem evaluate_sum (a : Fin 2002 → ℝ)
  (h1 : ∑ k in Finset.range 2002, a k / (k + 2) = 4 / 3)
  (h2 : ∑ k in Finset.range 2002, a k / (k + 3) = 4 / 5)
  (h3 : ∀ n ∈ Finset.range 2002, ∑ k in Finset.range 2002, a k / (k + (n + 2)) = 4 / (n + 3)) :
  ∑ k in Finset.range 2002, a k / (2 * k + 1) = 1 - 1 / 4005 ^ 2 :=
begin
  sorry,
end

end evaluate_sum_l169_169144


namespace trapezoids_not_necessarily_congruent_l169_169928

-- Define trapezoid structure
structure Trapezoid (α : Type) [LinearOrderedField α] :=
(base1 base2 side1 side2 diag1 diag2 : α) -- sides and diagonals
(angle1 angle2 angle3 angle4 : α)        -- internal angles

-- Conditions about given trapezoids
variables {α : Type} [LinearOrderedField α]
variables (T1 T2 : Trapezoid α)

-- The condition that corresponding angles of the trapezoids are equal
def equal_angles := 
  T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 ∧ 
  T1.angle3 = T2.angle3 ∧ T1.angle4 = T2.angle4

-- The condition that diagonals of the trapezoids are equal
def equal_diagonals := 
  T1.diag1 = T2.diag1 ∧ T1.diag2 = T2.diag2

-- The statement to prove
theorem trapezoids_not_necessarily_congruent :
  equal_angles T1 T2 ∧ equal_diagonals T1 T2 → ¬ (T1 = T2) := by
  sorry

end trapezoids_not_necessarily_congruent_l169_169928


namespace quadratic_inequality_solution_l169_169473

theorem quadratic_inequality_solution (a b : ℝ)
  (h_solution : ∀ x : ℝ, x ∈ set.Ioo 1 2 → ax^2 + x + b > 0) :
  a + b = -1 :=
sorry

end quadratic_inequality_solution_l169_169473


namespace find_a_range_l169_169389

def is_monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x ≤ y → f x ≤ f y

def f (a : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + x - 7

theorem find_a_range (a : ℝ)
  (h : is_monotonically_increasing (f a)) : -real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3 :=
  sorry

end find_a_range_l169_169389


namespace parallelepiped_lateral_surface_area_l169_169505

theorem parallelepiped_lateral_surface_area (a b : ℝ) (h : 60° = real.pi / 3) :
  let height := real.sqrt(3 * (a^2 + b^2)) in
  let S := 2 * (a + b) * height in
  S = 2 * (a + b) * real.sqrt(3 * (a^2 + b^2)) :=
by
  -- Definitions
  let height := real.sqrt(3 * (a^2 + b^2))
  let S := 2 * (a + b) * height
  -- Statement to prove
  show S = 2 * (a + b) * real.sqrt(3 * (a^2 + b^2))
sorry

end parallelepiped_lateral_surface_area_l169_169505


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169984

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169984


namespace regular_pentagonal_pyramid_angle_l169_169069

noncomputable def angle_between_slant_height_and_non_intersecting_edge (base_edge_slant_height : ℝ) : ℝ :=
  -- Assuming the base edge and slant height are given as input and equal
  if base_edge_slant_height > 0 then 36 else 0

theorem regular_pentagonal_pyramid_angle
  (base_edge_slant_height : ℝ)
  (h : base_edge_slant_height > 0) :
  angle_between_slant_height_and_non_intersecting_edge base_edge_slant_height = 36 :=
by
  -- omitted proof steps
  sorry

end regular_pentagonal_pyramid_angle_l169_169069


namespace exists_non_zero_sequence_divisible_by_1001_l169_169696

theorem exists_non_zero_sequence_divisible_by_1001 (a : Fin 10 → ℤ) :
  ∃ (z : Fin 10 → ℤ), 
    (∀ i, z i ∈ {-1, 0, 1}) ∧
    (∑ i, z i * a i) % 1001 = 0 ∧
    (∃ i, z i ≠ 0) :=
sorry

end exists_non_zero_sequence_divisible_by_1001_l169_169696


namespace all_numbers_equal_l169_169682

theorem all_numbers_equal 
  (x : Fin 2007 → ℝ)
  (h : ∀ (I : Finset (Fin 2007)), I.card = 7 → ∃ (J : Finset (Fin 2007)), J.card = 11 ∧ 
  (1 / 7 : ℝ) * I.sum x = (1 / 11 : ℝ) * J.sum x) :
  ∃ c : ℝ, ∀ i : Fin 2007, x i = c :=
by sorry

end all_numbers_equal_l169_169682


namespace find_mn_pairs_l169_169667

theorem find_mn_pairs : 
  (∀ m n : ℕ, m ≥ 3 ∧ n ≥ 3 ∧ (∃ᶠ a in Filter.atTop, a > 0 ∧ (a^m + a - 1) % (a^n + a^2 - 1) = 0 ) → (m = 5 ∧ n = 3) :=
by
  intros m n h1 h2
  sorry

end find_mn_pairs_l169_169667


namespace sin_alpha_of_tan_root_l169_169011

theorem sin_alpha_of_tan_root (α : ℝ) (hα : 0 < α ∧ α < π / 2) (h : ∃ x, x^2 + 2 * x - 3 = 0 ∧ tan α = x) : sin α = Real.sqrt 2 / 2 := 
by
  sorry

end sin_alpha_of_tan_root_l169_169011


namespace distance_A_B_midpoint_A_B_l169_169544

-- Define the points A and B
def pointA : ℝ × ℝ := (2, -3)
def pointB : ℝ × ℝ := (8, 5)

-- Define the function to calculate the distance between two points
def distance (p₁ p₂ : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p₂.1 - p₁.1)^2 + (p₂.2 - p₁.2)^2)

-- Define the function to calculate the midpoint of two points
def midpoint (p₁ p₂ : ℝ × ℝ) : ℝ × ℝ :=
  ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)

theorem distance_A_B :
  distance pointA pointB = 10 := 
sorry

theorem midpoint_A_B :
  midpoint pointA pointB = (5, 1) :=
sorry

end distance_A_B_midpoint_A_B_l169_169544


namespace projection_of_b_on_a_l169_169723

-- Define the problem conditions
variables (a b : ℝ) (a b : ℝ → ℝ → ℝ)

-- Given conditions:
def cond1 {a b : ℝ × ℝ} (ha : a ≠ 0) (hb : b ≠ 0) : Prop := 
  true

def cond2 (a : ℝ × ℝ) : Prop := 
  ∥a∥ = 2

def cond3 (a b : ℝ × ℝ) : Prop := 
  dot_product a (a + (2 • b)) = 0

-- We want to prove:
theorem projection_of_b_on_a (a b : ℝ × ℝ) (h1 : cond1 a b) (h2 : cond2 a) (h3 : cond3 a b) :
  projection b a = -1 / 4 := 
sorry

-- Definitions
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

def projection (b a : ℝ × ℝ) :=
  (dot_product b a) / (dot_product a a)

end projection_of_b_on_a_l169_169723


namespace sum_of_interior_edges_l169_169260

theorem sum_of_interior_edges {f : ℝ} {w : ℝ} (h_frame_area : f = 32) (h_outer_edge : w = 7) (h_frame_width : 2) :
  let i_length := w - 2 * h_frame_width in
  let i_other_length := (f - (w * (w  - 2 * h_frame_width))) / (w  + 2 * h_frame_width) in
  i_length + i_other_length + i_length + i_other_length = 8 :=
by
  let i_length := w - 2 * 2
  let i_other_length := (32 - (i_length * w)) / (w  + 2 * 2)
  let sum := i_length + i_other_length + i_length + i_other_length
  have h_sum : sum = 8, by sorry
  exact h_sum

end sum_of_interior_edges_l169_169260


namespace initial_cookie_count_l169_169844

theorem initial_cookie_count (eaten : ℕ) (left : ℕ) (total : ℕ) :
  eaten = 9 → left = 23 → total = 32 → eaten + left = total :=
by 
  sorry

# Example usage
example : initial_cookie_count 9 23 32 := 
by
  sorry

end initial_cookie_count_l169_169844


namespace robert_ate_10_more_chocolates_than_nickel_l169_169474

variable (S : ℕ) (N : ℕ) (R : ℕ)
hypothesis hS : S = 15
hypothesis hN : N = S - 5
hypothesis hR : R = 2 * N

theorem robert_ate_10_more_chocolates_than_nickel : R - N = 10 :=
by
  -- This part will contain the actual proof, which we are omitting.
  sorry

end robert_ate_10_more_chocolates_than_nickel_l169_169474


namespace range_a_l169_169743

-- Conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + Real.log x
def f' (a : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 1 / x

-- Theorem to prove the range of a
theorem range_a (a : ℝ) : (∀ x > 0, f' a x → ℝ → ∃ c : ℝ, a < 0) :=
sorry

end range_a_l169_169743


namespace vector_diff_magnitude_l169_169802

variables {ℝ_vector: Type*} [inner_product_space ℝ ℝ_vector]

open_locale real_inner_product_space

def is_unit_vector (v : ℝ_vector) : Prop :=
  ∥v∥ = 1

theorem vector_diff_magnitude (a b : ℝ_vector) (h_a_unit: is_unit_vector a) (h_b_unit: is_unit_vector b) (h_sum_unit: ∥a + b∥ = 1) :
  ∥a - b∥ = real.sqrt 3 :=
begin
  sorry
end

end vector_diff_magnitude_l169_169802


namespace greatest_sum_of_consecutive_integers_l169_169961

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169961


namespace scientific_notation_n_of_80_million_l169_169609

theorem scientific_notation_n_of_80_million :
  ∃ n : ℤ, 80000000 = 8 * 10^n ∧ n = 7 :=
by
  use 7
  split
  { sorry } -- Here you would show 80000000 = 8 * 10^7
  { refl }

end scientific_notation_n_of_80_million_l169_169609


namespace Q_no_negative_roots_and_at_least_one_positive_root_l169_169294

def Q (x : ℝ) : ℝ := x^7 - 2 * x^6 - 6 * x^4 - 4 * x + 16

theorem Q_no_negative_roots_and_at_least_one_positive_root :
  (∀ x, x < 0 → Q x > 0) ∧ (∃ x, x > 0 ∧ Q x = 0) := 
sorry

end Q_no_negative_roots_and_at_least_one_positive_root_l169_169294


namespace range_of_a_for_three_zeros_l169_169362

def f (x a : ℝ) : ℝ := if x > a then x + 2 else x^2 + 5 * x + 2

def g (x a : ℝ) : ℝ := f x a - 2 * x

theorem range_of_a_for_three_zeros : { a : ℝ | ∃ h : ∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ g x1 a = 0 ∧ g x2 a = 0 ∧ g x3 a = 0 ∧ x1 ≠ a ∧ x2 ≠ a ∧ x3 ≠ a } = set.Ico (-1 : ℝ) 2 :=
by
  sorry

end range_of_a_for_three_zeros_l169_169362


namespace find_expression_axis_of_symmetry_center_of_symmetry_transformation_range_of_f_l169_169709

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

axiom A_pos : 2 > 0
axiom omega : ℝ := 2
axiom omega_pos : omega > 0
axiom phi : ℝ := Real.pi / 6
axiom phi_pos : 0 < phi ∧ phi < Real.pi / 2
axiom lowest_point : f (2 * Real.pi / 3) = -2

theorem find_expression :
  f = (λ x : ℝ, 2 * Real.sin (2 * x + Real.pi / 6)) :=
sorry

theorem axis_of_symmetry (k : ℤ) :
  (∃ k : ℤ, (λ x : ℝ, 2 * Real.sin (2 * x + Real.pi / 6)) = k * ℝ.pi / 2 + ℝ.pi / 6) :=
sorry

theorem center_of_symmetry (k : ℤ) : 
  (∃ k : ℤ, (2 * k * Real.pi - Real.pi / 12, 0)) :=
sorry

theorem transformation :
  (λ x : ℝ, f x) = 2 * (λ x : ℝ, Real.sin (2 * x + Real.pi / 6)) :=
sorry

theorem range_of_f : 
  (∀ x ∈ Set.Icc (Real.pi / 12) (Real.pi / 2), 
  (-1 : ℝ ≤ f x) ∧ (f x ≤ 2)) :=
sorry

end find_expression_axis_of_symmetry_center_of_symmetry_transformation_range_of_f_l169_169709


namespace focus_of_parabola_l169_169885

theorem focus_of_parabola (x y : ℝ) (h : y = 2 * x^2) : 
  ∃ f : ℝ × ℝ, f = (0, 1 / 8) :=
by
  sorry

end focus_of_parabola_l169_169885


namespace part_I_solution_set_part_II_m_range_l169_169834

def f (x : ℝ) : ℝ := 2 * |x - 1| + |x + 2|

theorem part_I_solution_set :
  {x : ℝ | f x ≥ 4} = set.Iic 0 ∪ set.Ici (4 / 3) := sorry

theorem part_II_m_range (m : ℝ) :
  (∃ x : ℝ, f x < |m - 2|) → m ∈ set.Iio (-1) ∪ set.Ioi 5 := sorry

end part_I_solution_set_part_II_m_range_l169_169834


namespace jacqueline_guavas_l169_169426

theorem jacqueline_guavas 
  (G : ℕ) 
  (plums : ℕ := 16) 
  (apples : ℕ := 21) 
  (given : ℕ := 40) 
  (remaining : ℕ := 15) 
  (initial_fruits : ℕ := plums + G + apples)
  (total_fruits_after_given : ℕ := remaining + given) : 
  initial_fruits = total_fruits_after_given → G = 18 := 
by
  intro h
  sorry

end jacqueline_guavas_l169_169426


namespace limit_f_log2005_div_n_eq_0_l169_169618

open Nat

-- Define sets A and B as described
def A : Set ℕ := { n | n = 0 ∨ n = 2 ∨ n = 8 ∨ n = 10 ∨ (∃ (k : ℕ), n = (bits_to_nat (list.reverse_bin_list [false] [false] [false] ...) + ...))}
def B : Set ℕ := { n | n = 1 ∨ n = 3 ∨ n = 4 ∨ n = 5 ∨ n = 6 ∨ n = 7 ∨ n = 9 ∨ (∃ (k : ℕ), n = (bits_to_nat (list.reverse_bin_list [true] [false] [false] ...) + ...))}

-- Define function f as described
def f (n : ℕ) : ℕ := cardinal.mk (A ∩ {k | k < n})

-- State the problem to prove the limit as given
theorem limit_f_log2005_div_n_eq_0 :
  tendsto (λ n, (f n * (log n)^2005) / n) at_top (𝓝 0) := by
  sorry

end limit_f_log2005_div_n_eq_0_l169_169618


namespace find_x_values_l169_169698

theorem find_x_values (x : ℤ) 
  (h1 : |x-1| ≥ 2) 
  (h2 : x ∈ Set.univ ℤ) 
  (h3 : ¬((|x-1| ≥ 2) ∧ (x ∈ Set.univ ℤ)))
  (h4 : ¬(¬x ∈ Set.univ ℤ)) : 
  x = 0 ∨ x = 1 ∨ x = 2 :=
by
  sorry

end find_x_values_l169_169698


namespace geometric_sequence_ratio_l169_169403

theorem geometric_sequence_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) (q : ℝ) (h1 : 6 * a 7 = (a 8 + a 9) / 2)
  (h2 : ∀ n : ℕ, a (n + 1) = a n * q)
  (h3 : ∀ n : ℕ, S n = a 1 * (1 - q^n) / (1 - q)) :
  S 6 / S 3 = 28 :=
by
  -- The proof goes here
  sorry

end geometric_sequence_ratio_l169_169403


namespace digit_150_of_3_div_11_l169_169536

theorem digit_150_of_3_div_11 :
  let rep := "27"
  ∃ seq : ℕ → ℕ,
  (∀ n, seq (2 * n) = 2) ∧ (∀ n, seq (2 * n + 1) = 7) ∧
  150 % 2 = 0 ∧ seq (150 - 1) = 7 :=
by
  let rep := "27"
  use (λ n : ℕ => if n % 2 = 0 then 2 else 7)
  split
  { intro n
    exact rfl }
  { intro n
    exact rfl }
  { exact rfl }
  { exact rfl }

end digit_150_of_3_div_11_l169_169536


namespace union_M_N_l169_169717

def M : Set ℝ := {y | ∃ x, x ∈ Ici (0 : ℝ) ∧ y = (1 / 2) ^ x }
def N : Set ℝ := {y | ∃ x, x ∈ Ioc (0 : ℝ) 1 ∧ y = Real.log x / Real.log 2 }

theorem union_M_N : M ∪ N = {y | y <= 1} :=
by
  sorry

end union_M_N_l169_169717


namespace mean_of_six_numbers_l169_169507

theorem mean_of_six_numbers (numbers : Fin 6 → ℚ)
  (h : ∑ i, numbers i = 2/3) : 
  (∑ i, numbers i) / 6 = 1/9 :=
by {
  sorry -- The proof is omitted, as per the instructions.
}

end mean_of_six_numbers_l169_169507


namespace vector_diff_magnitude_l169_169798

variables {ℝ_vector: Type*} [inner_product_space ℝ ℝ_vector]

open_locale real_inner_product_space

def is_unit_vector (v : ℝ_vector) : Prop :=
  ∥v∥ = 1

theorem vector_diff_magnitude (a b : ℝ_vector) (h_a_unit: is_unit_vector a) (h_b_unit: is_unit_vector b) (h_sum_unit: ∥a + b∥ = 1) :
  ∥a - b∥ = real.sqrt 3 :=
begin
  sorry
end

end vector_diff_magnitude_l169_169798


namespace verify_phone_prices_and_profits_l169_169122

noncomputable def phone_prices_and_profits : Prop :=
  let x := 2400 in
  let y := x + 800 in
  let budget : ℕ := 28000 in
  let price_A := y in
  let price_B := x in
  let profit_A : ℕ := 3700 - price_A in
  let profit_B : ℕ := 2700 - price_B in
  let total_profit (m n : ℕ) : ℕ := m * profit_A + n * profit_B in
    (price_B = 2400) ∧ 
    (price_A = 3200) ∧
    (∃ (m n : ℕ), 4 * m + 3 * n = 35 ∧ budget = 3200 * m + 2400 * n ∧ (m, n) = (8, 1) ∧ total_profit m n = 4300)

theorem verify_phone_prices_and_profits : phone_prices_and_profits := 
sorry

end verify_phone_prices_and_profits_l169_169122


namespace putty_volume_l169_169376

theorem putty_volume (k : ℝ) : 0 < k → 
  (∃ V : ℝ, V = k / 2 - 4 / 3 ∧ V = (Volume to seal the rectangular window frame with a perimeter of k cm)) :=
sorry

end putty_volume_l169_169376


namespace angle_Congruence_l169_169567

open EuclideanGeometry

variables (A B C P H M N O D E : Point)
variables [AcuteAngleTriangle A B C]
variables (H_on_CP : H ∈ altitude A B C P)
variables (AH_intersection : intersect_lines A H B C = M)
variables (BH_intersection : intersect_lines B H A C = N)
variables (MN_intersection : intersect_lines M N C P = O)
variables (line_through_O : ∃ l, ∃ m, l = line_through O C M D ∧ m = line_through O N H E ∧ intersect_lines l m = O)

theorem angle_Congruence : ∠ EPC = ∠ DPC :=
by
  sorry

end angle_Congruence_l169_169567


namespace no_prime_natural_solution_exists_l169_169129

theorem no_prime_natural_solution_exists (p : ℕ) (m : ℕ) (h1 : p.prime) (h2 : 5 < p) : ¬ ((p - 1)! + 1 = p^m) :=
by sorry

end no_prime_natural_solution_exists_l169_169129


namespace martin_class_number_l169_169298

theorem martin_class_number (b : ℕ) (h1 : 100 < b) (h2 : b < 200) 
  (h3 : b % 3 = 2) (h4 : b % 4 = 1) (h5 : b % 5 = 1) : 
  b = 101 ∨ b = 161 := 
by
  sorry

end martin_class_number_l169_169298


namespace train_crossing_time_l169_169081

-- Define the conditions.

def train_length : ℝ := 100  -- Train length in meters.
def train_speed_km_per_hr : ℝ := 36  -- Train speed in km/hr.

-- Convert speed from km/hr to m/s.
def train_speed_m_per_s : ℝ := train_speed_km_per_hr * (1000 / 3600)

-- Calculate the time to cross the pole.
def time_to_cross : ℝ := train_length / train_speed_m_per_s

-- The theorem to prove.
theorem train_crossing_time : time_to_cross = 10 := by
  sorry

end train_crossing_time_l169_169081


namespace julios_spending_on_limes_l169_169779

theorem julios_spending_on_limes 
    (days : ℕ) (lime_juice_per_day : ℕ) (lime_juice_per_lime : ℕ) (limes_per_dollar : ℕ) 
    (total_spending : ℝ) 
    (h1 : days = 30) 
    (h2 : lime_juice_per_day = 1) 
    (h3 : lime_juice_per_lime = 2) 
    (h4 : limes_per_dollar = 3) 
    (h5 : total_spending = 5) :
    let lime_juice_needed := days * lime_juice_per_day,
        total_limes := lime_juice_needed / lime_juice_per_lime,
        cost := (total_limes / limes_per_dollar : ℕ) in
    (cost : ℝ) = total_spending := 
by 
    sorry

end julios_spending_on_limes_l169_169779


namespace fraction_equation_solution_l169_169151

theorem fraction_equation_solution (a : ℤ) (hpos : a > 0) (h : (a : ℝ) / (a + 50) = 0.870) : a = 335 :=
by {
  sorry
}

end fraction_equation_solution_l169_169151


namespace smallest_positive_solution_tan2x_tan3x_sec3x_l169_169670

theorem smallest_positive_solution_tan2x_tan3x_sec3x :
  ∃ x > 0, tan (2 * x) + tan (3 * x) = sec (3 * x) ∧ x = π / 14 :=
by
  use π / 14
  split
  { exact div_pos pi_pos (by norm_num) }
  split
  { sorry } -- Trigonometric identities and verification step
  { refl }

end smallest_positive_solution_tan2x_tan3x_sec3x_l169_169670


namespace ratio_of_saturday_to_friday_customers_l169_169192

def tips_per_customer : ℝ := 2.0
def customers_friday : ℕ := 28
def customers_sunday : ℕ := 36
def total_tips : ℝ := 296

theorem ratio_of_saturday_to_friday_customers :
  let tips_friday := customers_friday * tips_per_customer
  let tips_sunday := customers_sunday * tips_per_customer
  let tips_friday_and_sunday := tips_friday + tips_sunday
  let tips_saturday := total_tips - tips_friday_and_sunday
  let customers_saturday := tips_saturday / tips_per_customer
  (customers_saturday / customers_friday : ℝ) = 3 := 
by
  sorry

end ratio_of_saturday_to_friday_customers_l169_169192


namespace area_of_border_is_correct_l169_169594

-- Definitions based on the given conditions
def photograph_height : ℕ := 12
def photograph_width : ℕ := 15
def border_width : ℕ := 3

-- Problem statement
theorem area_of_border_is_correct : 
  (let photograph_area := photograph_height * photograph_width in
   let frame_height := photograph_height + 2 * border_width in
   let frame_width := photograph_width + 2 * border_width in
   let frame_area := frame_height * frame_width in
   let border_area := frame_area - photograph_area in
   border_area = 198) :=
by simp [photograph_height, photograph_width, border_width]; sorry

end area_of_border_is_correct_l169_169594


namespace packages_required_l169_169156

def room_numbers (n : ℕ) : Prop :=
  (n >= 101 ∧ n <= 150) ∨ (n >= 201 ∧ n <= 250) ∨ (n >= 301 ∧ n <= 350)

def digit_count (d : ℕ) (count : ℕ) : Prop :=
  ∀ floor_start, (floor_start ∈ [100, 200, 300]) →
    (count = (1 + 5 + 15) ∨
     (floor_start + d ∈ range 10 ∧ count = 15))

theorem packages_required :
  ∃ k, ∀ d, digit_count d 50 → k = 50 :=
sorry

end packages_required_l169_169156


namespace basketball_statistics_no_increase_after_7th_game_l169_169434

theorem basketball_statistics_no_increase_after_7th_game :
  let scores := [38, 55, 40, 59, 42, 57]
  let new_score := 46
  let updated_scores := scores ++ [new_score]
  -- Define range, median, mean, mode, and mid-range for initial and updated scores
  let initial_range := (scores.maximumD 0) - (scores.minimumD 0)
  let new_range := (updated_scores.maximumD 0) - (updated_scores.minimumD 0)
  let initial_median := if let scores_sorted := scores.sort, n := scores.length, n % 2 == 0 then 
                         (scores_sorted.get! (n / 2 - 1) + scores_sorted.get! (n / 2)) / 2 
                       else 
                         scores_sorted.get! (n / 2)
  let new_median := if let updated_scores_sorted := updated_scores.sort, n := updated_scores.length, n % 2 == 0 then 
                      (updated_scores_sorted.get! (n / 2 - 1) + updated_scores_sorted.get! (n / 2)) / 2
                    else 
                      updated_scores_sorted.get! (n / 2)
  let initial_mean := (scores.sum) / (scores.length)
  let new_mean := (updated_scores.sum) / (updated_scores.length)
  let initial_mode := if let freq_map := scores.toMultiset, mode_set := freq_map.toFinset.filter (λ x, freq_map.count x > 1) then 
                        if mode_set.isEmpty then none else some mode_set 
                      else none
  let new_mode := if let freq_map := updated_scores.toMultiset, mode_set := freq_map.toFinset.filter (λ x, freq_map.count x > 1) then 
                    if mode_set.isEmpty then none else some mode_set 
                  else none
  let initial_midrange := ((scores.maximumD 0) + (scores.minimumD 0)) / 2
  let new_midrange := ((updated_scores.maximumD 0) + (updated_scores.minimumD 0)) / 2
  -- Proving that none of these statistics increased
  initial_range ≥ new_range ∧ initial_median ≥ new_median ∧ initial_mean ≥ new_mean ∧ initial_mode = new_mode ∧ initial_midrange = new_midrange :=
sorry

end basketball_statistics_no_increase_after_7th_game_l169_169434


namespace best_cashback_categories_l169_169467

def transport_expense : ℕ := 2000
def groceries_expense : ℕ := 5000
def clothing_expense : ℕ := 3000
def entertainment_expense : ℕ := 3000
def sports_goods_expense : ℕ := 1500

def transport_cashback : ℕ → ℕ := λ exp, exp * 5 / 100
def groceries_cashback : ℕ → ℕ := λ exp, exp * 3 / 100
def clothing_cashback : ℕ → ℕ := λ exp, exp * 4 / 100
def entertainment_cashback : ℕ → ℕ := λ exp, exp * 5 / 100
def sports_goods_cashback : ℕ → ℕ := λ exp, exp * 6 / 100

theorem best_cashback_categories :
  ({2, 3, 4} : set ℕ) ⊆ {2, 3, 4, 5} ∧
  ∀ categories : set ℕ,
    categories ⊆ {1, 2, 3, 4, 5} →
    (∀ cat ∈ categories, cat = 2 ∨ cat = 3 ∨ cat = 4) →
    (∑ cat in categories, if cat = 1 then transport_cashback transport_expense
                  else if cat = 2 then groceries_cashback groceries_expense
                  else if cat = 3 then clothing_cashback clothing_expense
                  else if cat = 4 then entertainment_cashback entertainment_expense
                  else sports_goods_cashback sports_goods_expense)
    ≤ (600 + 10) :=   -- Adjusted amount to reflect equivalent condition
sorry

end best_cashback_categories_l169_169467


namespace relationship_between_line_and_circle_l169_169010

variables {a b r : ℝ} (M : ℝ × ℝ) (l m : ℝ → ℝ)

def point_inside_circle_not_on_axes 
    (M : ℝ × ℝ) (r : ℝ) : Prop := 
    (M.fst^2 + M.snd^2 < r^2) ∧ (M.fst ≠ 0) ∧ (M.snd ≠ 0)

def line_eq (a b r : ℝ) (x y : ℝ) : Prop := 
    a * x + b * y = r^2

def chord_midpoint (M : ℝ × ℝ) (m : ℝ → ℝ) : Prop := 
    ∃ x1 y1 x2 y2, 
    (M.fst = (x1 + x2) / 2 ∧ M.snd = (y1 + y2) / 2) ∧ 
    (m x1 = y1 ∧ m x2 = y2)

def circle_external (O : ℝ → ℝ) (l : ℝ → ℝ) : Prop := 
    ∀ x y, O x = y → l x ≠ y

theorem relationship_between_line_and_circle
    (M_inside : point_inside_circle_not_on_axes M r)
    (M_chord : chord_midpoint M m)
    (line_eq_l : line_eq a b r M.fst M.snd) :
    (m (M.fst) = - (a / b) * M.snd) ∧ 
    (∀ x, l x ≠ m x) :=
sorry

end relationship_between_line_and_circle_l169_169010


namespace evaluate_expression_l169_169651

theorem evaluate_expression :
  (floor ((ceil (121 / 36 : ℚ) + 19 / 5) : ℚ)) = 7 :=
by
  sorry

end evaluate_expression_l169_169651


namespace disputed_piece_weight_l169_169865

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end disputed_piece_weight_l169_169865


namespace correct_statements_count_l169_169680

noncomputable def f (x : ℝ) := (1 / 2) * Real.sin (2 * x)
noncomputable def g (x : ℝ) := (1 / 2) * Real.sin (2 * x + Real.pi / 4)

theorem correct_statements_count :
  let complexity := True
    -- Condition 1: Smallest positive period of f(x) is 2π
    (∀ x : ℝ, f (x + 2 * Real.pi) = f x) = False ∧
    -- Condition 2: f(x) is monotonically increasing on [-π/4, π/4]
    (∀ x y : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y) = True ∧
    -- Condition 3: Range of f(x) when x ∈ [-π/6, π/3]
    (∀ y : ℝ, ∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x = y ↔ -Real.sqrt(3) / 4 ≤ y ∧ y ≤ Real.sqrt(3) / 4) = False ∧
    -- Condition 4: Graph of f(x) can be obtained by shifting g(x) to the left by π/8
    (∀ x : ℝ, f x = g (x + Real.pi / 8)) = False →
  true := sorry

end correct_statements_count_l169_169680


namespace sequence_term_l169_169173

noncomputable def S (n : ℕ) : ℤ := n^2 - 3 * n

theorem sequence_term (n : ℕ) (h : n ≥ 1) : 
  ∃ a : ℕ → ℤ, a n = 2 * n - 4 := 
  sorry

end sequence_term_l169_169173


namespace shekar_marks_in_math_l169_169136

theorem shekar_marks_in_math (M : ℕ) : 
  (65 + 82 + 67 + 75 + M) / 5 = 73 → M = 76 :=
by
  intros h
  sorry

end shekar_marks_in_math_l169_169136


namespace base_7_units_digit_of_product_359_72_l169_169486

def base_7_units_digit (n : ℕ) : ℕ := n % 7

theorem base_7_units_digit_of_product_359_72 : base_7_units_digit (359 * 72) = 4 := 
by
  sorry

end base_7_units_digit_of_product_359_72_l169_169486


namespace sum_of_point_P1_l169_169510

theorem sum_of_point_P1 (k m : ℕ) : 
  let A := (0, 0)
  let B := (0, 420)
  let C := (560, 0)
  let P1 := (k, m)
  let P2 := (448, 4)
  let P3 := (224, 212)
  let P4 := (112, 316)
  let P5 := (56, 368)
  let P6 := (28, 184)
  let P7 := (14, 92)
  P1 = (336, 8) →
  k + m = 344 :=
begin
  sorry,
end

end sum_of_point_P1_l169_169510


namespace part1_part2_part3_l169_169099

-- Assumptions and definitions
def f (x : ℝ) : ℝ := abs (2 * x - 2) + abs (x + 2)

-- Part 1: Prove the inequality solution set
theorem part1 (x : ℝ) : f x ≤ 5 - 2 * x ↔ -5 ≤ x ∧ x ≤ 1 :=
by sorry

-- Part 2: Prove the minimum value of f(x)
theorem part2 : (∀ x, f x ≥ 3) ∧ (∃ x, f x = 3) :=
by sorry

-- Part 3: Prove the inequality involving a and b
theorem part3 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a^2 + b^2 + 2 * b = 3) :
  a + b ≤ 2 * real.sqrt 2 - 1 :=
by sorry

end part1_part2_part3_l169_169099


namespace mean_salaries_l169_169224

noncomputable def salaries : List ℝ := [1000, 2500, 3100, 3650, 1500, 2000]
noncomputable def mean (l : List ℝ) : ℝ := l.sum / l.length

theorem mean_salaries : mean salaries = 2458.33 := by
  sorry

end mean_salaries_l169_169224


namespace min_area_of_B_l169_169692

noncomputable def setA := { p : ℝ × ℝ | abs (p.1 - 2) + abs (p.2 - 3) ≤ 1 }

noncomputable def setB (D E F : ℝ) := { p : ℝ × ℝ | p.1^2 + p.2^2 + D * p.1 + E * p.2 + F ≤ 0 ∧ D^2 + E^2 - 4 * F > 0 }

theorem min_area_of_B (D E F : ℝ) (h : setA ⊆ setB D E F) : 
  ∃ r : ℝ, (∀ p ∈ setB D E F, p.1^2 + p.2^2 ≤ r^2) ∧ (π * r^2 = 2 * π) :=
sorry

end min_area_of_B_l169_169692


namespace maximum_value_of_p_l169_169394

noncomputable def C (n k : ℕ) : ℕ := (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

noncomputable def p (n : ℕ) (h : n > 0) : ℝ :=
  10 * n / ((n + 5) * (n + 4))

theorem maximum_value_of_p :
  ∃ n : ℕ, n > 0 ∧ p n (by linarith) = 5 / 9 :=
sorry

end maximum_value_of_p_l169_169394


namespace proportion_of_overcrowded_cars_is_40_percent_min_prop_of_passengers_in_overcrowded_cars_is_49_percent_passengers_prop_cannot_be_less_than_cars_l169_169155

def condition (percent_10_19 : ℕ) (percent_20_29 : ℕ) (percent_30_39 : ℕ) (percent_40_49 : ℕ) 
              (percent_50_59 : ℕ) (percent_60_69 : ℕ) (percent_70_79 : ℕ) 
              (is_overcrowded : ℕ → Prop) :=
  percent_10_19 = 4 ∧ percent_20_29 = 6 ∧ percent_30_39 = 12 ∧ percent_40_49 = 18 ∧ 
  percent_50_59 = 20 ∧ percent_60_69 = 14 ∧ percent_70_79 = 6 ∧ 
  (∀ n : ℕ, is_overcrowded n ↔ n ≥ 60)

def proportion_of_overcrowded_cars (percent_10_19 percent_20_29 percent_30_39 percent_40_49 
                                    percent_50_59 percent_60_69 percent_70_79 : ℕ) : ℕ :=
  percent_60_69 + percent_70_79

theorem proportion_of_overcrowded_cars_is_40_percent (percent_10_19 percent_20_29 percent_30_39 
                                                     percent_40_49 percent_50_59 percent_60_69 
                                                     percent_70_79 : ℕ) 
                                                     (is_overcrowded : ℕ → Prop) 
                                                     (h : condition percent_10_19 percent_20_29 
                                                     percent_30_39 percent_40_49 percent_50_59 
                                                     percent_60_69 percent_70_79 is_overcrowded) :   
  (proportion_of_overcrowded_cars percent_50_59 percent_60_69 percent_70_79 = 40) :=
  sorry

def min_prop_of_passengers_in_overcrowded_cars (N : ℕ) : ℕ :=
  -- Calculation steps skipped for brevity
  49

theorem min_prop_of_passengers_in_overcrowded_cars_is_49_percent (N : ℕ) 
                                                                (percent_10_19 percent_20_29 
                                                                percent_30_39 percent_40_49 
                                                                percent_50_59 percent_60_69 
                                                                percent_70_79 : ℕ) 
                                                                (is_overcrowded : ℕ → Prop) 
                                                                (h : condition percent_10_19 
                                                                percent_20_29 percent_30_39 
                                                                percent_40_49 percent_50_59 
                                                                percent_60_69 percent_70_79 
                                                                is_overcrowded) :   
  (min_prop_of_passengers_in_overcrowded_cars N = 49) :=
  sorry

def can_passengers_prop_be_less_than_cars (N n M m : ℕ) 
                                          (avg_passengers_per_wagon : ℕ → ℕ → ℕ) 
                                          (avg_passengers_per_overcrowded_wagon : ℕ → ℕ → ℕ) 
                                          (h : avg_passengers_per_wagon n N ≤ 
                                                avg_passengers_per_overcrowded_wagon m M) : Prop :=
  not (M / N < m / n)

theorem passengers_prop_cannot_be_less_than_cars (N n M m : ℕ) 
                                                 (avg_passengers_per_wagon : ℕ → ℕ → ℕ) 
                                                 (avg_passengers_per_overcrowded_wagon : ℕ → ℕ → ℕ) 
                                                 (h : can_passengers_prop_be_less_than_cars N n M m 
                                                 avg_passengers_per_wagon 
                                                 avg_passengers_per_overcrowded_wagon h) : 
  can_passengers_prop_be_less_than_cars N n M m avg_passengers_per_wagon 
                                       avg_passengers_per_overcrowded_wagon h :=
  sorry

end proportion_of_overcrowded_cars_is_40_percent_min_prop_of_passengers_in_overcrowded_cars_is_49_percent_passengers_prop_cannot_be_less_than_cars_l169_169155


namespace sqrt_19_range_l169_169648

theorem sqrt_19_range : 4 < Real.sqrt 19 ∧ Real.sqrt 19 < 5 := 
by
  have h1 : Real.sqrt 16 = 4 := by norm_num
  have h2 : Real.sqrt 25 = 5 := by norm_num
  have h3 : Real.sqrt 16 < Real.sqrt 19 := by exact Real.sqrt_lt.mpr (by norm_num)
  have h4 : Real.sqrt 19 < Real.sqrt 25 := by exact Real.sqrt_lt.mpr (by norm_num)
  exact ⟨by linarith, by linarith⟩

end sqrt_19_range_l169_169648


namespace inequality_unique_solution_l169_169058

theorem inequality_unique_solution (p : ℝ) :
  (∃ x : ℝ, 0 ≤ x^2 + p * x + 5 ∧ x^2 + p * x + 5 ≤ 1) →
  (∃ x : ℝ, x^2 + p * x + 4 = 0) → p = 4 ∨ p = -4 :=
sorry

end inequality_unique_solution_l169_169058


namespace ratio_of_areas_l169_169165

theorem ratio_of_areas (P C : ℝ) (w r : ℝ)
  (Perimeter_eq : 2 * (2 * w + w) = 2 * real.pi * r)
  : (2 * w^2) / (real.pi * r^2) = 2 / 9 * real.pi :=
by
  sorry

end ratio_of_areas_l169_169165


namespace integral_zero_l169_169790

variables {f : ℝ → ℝ} {F : ℝ → ℝ} (T : ℝ) (n : ℕ)

-- Conditions
def periodic (f : ℝ → ℝ) (T : ℝ) := ∀ x, f (x + T) = f x
def continuous (f : ℝ → ℝ) := ∀ x, continuous_at f x
def antiderivative (F f : ℝ → ℝ) := ∀ x, deriv F x = f x

theorem integral_zero (h_periodic : periodic f T)
                      (h_continuous : continuous f)
                      (h_antiderivative : antiderivative F f) :
  ∫ x in 0 .. T, (F (n * x) - F x - f x * ((n - 1) * T) / 2) = 0 :=
sorry

end integral_zero_l169_169790


namespace sequence_formula_sum_geq_2013_l169_169770

noncomputable def a (n : ℕ) : ℤ := 3 * (-2) ^ (n - 1)

noncomputable def S (n : ℕ) : ℤ := (List.range n).sum (fun i => a (i + 1))

theorem sequence_formula :
  (forall n, n >= 1 -> a n = 3 * (-2) ^ (n - 1)) ∧
  (S 2 + S 4 - 2 * S 3 = 0) ∧
  ((a 2) + (a 3) + (a 4) = -18) :=
by
  sorry

theorem sum_geq_2013 :
  (∃ k : ℕ, k >= 5 ∧ S (2 * k + 1) >= 2013) :=
by
  sorry

end sequence_formula_sum_geq_2013_l169_169770


namespace radius_of_C_greater_than_middle_r_l169_169579

variables {O1 O2 O3 : Type} [MetricSpace O1] [MetricSpace O2] [MetricSpace O3]
variables (C : Type) [MetricSpace C]
variables (r1 r2 r3 R : ℝ)
variables (touches : ∀ {X Y : Type} [MetricSpace X] [MetricSpace Y], Prop)

-- Conditions
def centers_collinear (O1 O2 O3 : Type) [MetricSpace O1] [MetricSpace O2] [MetricSpace O3] : Prop :=
  colinear [O1, O2, O3]

def pairwise_disjoint (O1 O2 O3 : Type) [MetricSpace O1] [MetricSpace O2] [MetricSpace O3] : Prop :=
  disjoint (dist_metric O1) (dist_metric O2) (dist_metric O3)

def circle_C_touches (C O1 O2 O3 : Type) [MetricSpace C] [MetricSpace O1] [MetricSpace O2] [MetricSpace O3] : Prop :=
  touches C O1 ∧ touches C O2 ∧ touches C O3

-- Statement
theorem radius_of_C_greater_than_middle_r (h_collinear: centers_collinear O1 O2 O3)
    (h_disjoint: pairwise_disjoint O1 O2 O3)
    (h_touches: circle_C_touches C O1 O2 O3)
    (h_radii_pos: 0 < r1 ∧ 0 < r2 ∧ 0 < r3):
  R > r2 :=
sorry

end radius_of_C_greater_than_middle_r_l169_169579


namespace similar_triangles_l169_169103

open EuclideanGeometry

theorem similar_triangles (A B C H_A H_B : Point)
  (h_A_perp : Perpendicular A B C H_A)
  (h_B_perp : Perpendicular B A C H_B) :
  Similar (triangle C H_A H_B) (triangle C A B) :=
sorry

end similar_triangles_l169_169103


namespace cat_count_l169_169458

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end cat_count_l169_169458


namespace independence_test_l169_169766

theorem independence_test (H0 : Prop) (k2 : ℝ) (p_k2_ge_10_83 : ℝ) 
  (h0_unrelated : H0) (h_est : p_k2_ge_10_83 ≈ 0.001) : 
  (1 - p_k2_ge_10_83) = 0.999 := 
by
  sorry

end independence_test_l169_169766


namespace parallelogram_configuration_l169_169076

theorem parallelogram_configuration
  (A B C D E : Point)
  (h_convex : convex_pentagon A B C D E)
  (h_angles_equal : ∠CAB = ∠BCA)
  (P : Point)
  (h_midpoint : is_midpoint P B D)
  (h_collinear : collinear C P E) :
  is_parallelogram B C D E :=
begin
  sorry
end

end parallelogram_configuration_l169_169076


namespace taequan_dice_game_l169_169145

theorem taequan_dice_game (p : ℝ) (ways_eight : ℕ → ℕ) (H1 : p = 0.1388888888888889)
  (H2 : ∀ n, (ways_eight n / 6^n)^2 = p → ways_eight 2 = 5 ) : 
  ∃ n, (ways_eight n / 6^n)^2 = p ∧ n = 2 := by 
begin 
  sorry 
end

end taequan_dice_game_l169_169145


namespace functional_equation_solution_l169_169642

theorem functional_equation_solution (f : ℤ → ℝ) (hf : ∀ x y : ℤ, f (↑((x + y) / 3)) = (f x + f y) / 2) :
    ∃ c : ℝ, ∀ x : ℤ, x ≠ 0 → f x = c :=
sorry

end functional_equation_solution_l169_169642


namespace percent_increase_to_restore_l169_169500

noncomputable def original_price : ℝ := 100
noncomputable def first_discount : ℝ := original_price * 0.25
noncomputable def price_after_first_discount : ℝ := original_price - first_discount
noncomputable def second_discount : ℝ := price_after_first_discount * 0.10
noncomputable def price_after_second_discount : ℝ := price_after_first_discount - second_discount
noncomputable def required_increase : ℝ := (original_price - price_after_second_discount) / price_after_second_discount * 100

theorem percent_increase_to_restore :
  required_increase ≈ 48.15 :=
sorry

end percent_increase_to_restore_l169_169500


namespace max_cashback_categories_l169_169464

-- Define the expenditure and cashback percentages
def expenses : ℕ → ℕ
| 1 := 2000
| 2 := 5000
| 3 := 3000
| 4 := 3000
| 5 := 1500
| _ := 0

def cashback_rate : ℕ → ℝ
| 1 := 0.05
| 2 := 0.03
| 3 := 0.04
| 4 := 0.05
| 5 := 0.06
| _ := 0

-- Define the cashback amount for each category
def cashback_amount (n : ℕ) : ℝ :=
expenses n * cashback_rate n

-- Define the set of categories
def categories := {1, 2, 3, 4, 5}

-- Prove that selecting Groceries, Entertainment, and Clothing maximizes the cashback
theorem max_cashback_categories :
  let selected := {2, 3, 4} in
  ∀ s ⊆ categories, s.card = 3 → ∑ i in s, cashback_amount i ≤ ∑ i in selected, cashback_amount i :=
begin
  -- The proof will be filled in here
  sorry
end

end max_cashback_categories_l169_169464


namespace equal_functions_pair_d_l169_169218

-- Define the functions for pair D
def f : ℝ → ℝ := λ x, x
def g : ℝ → ℝ := λ x, 3 * x^3

-- Statement of the proof problem
theorem equal_functions_pair_d : (∀ x : ℝ, f x = g x) :=
by
  sorry -- The proof is omitted for this example, only the statement is required

end equal_functions_pair_d_l169_169218


namespace number_of_boys_l169_169485

theorem number_of_boys (n : ℕ)
    (incorrect_avg_weight : ℝ)
    (misread_weight new_weight : ℝ)
    (correct_avg_weight : ℝ)
    (h1 : incorrect_avg_weight = 58.4)
    (h2 : misread_weight = 56)
    (h3 : new_weight = 66)
    (h4 : correct_avg_weight = 58.9)
    (h5 : n * correct_avg_weight = n * incorrect_avg_weight + (new_weight - misread_weight)) :
  n = 20 := by
  sorry

end number_of_boys_l169_169485


namespace sum_f_2015_l169_169241

def f : ℝ → ℝ
| x => if h : (x % 6) > -3 ∧ (x % 6) ≤ -1 then -(x % 6 + 2) ^ 2 
       else if h : (x % 6) ≥ -1 ∧ (x % 6) ≤ 3 then x % 6 
       else 0 -- This branch is somewhat arbitrary, as f is only fully defined in these intervals.

theorem sum_f_2015 : (∑ i in Finset.range 2015, f (i + 1)) = 1680 := by
  sorry

end sum_f_2015_l169_169241


namespace log_geom_seq_sum_l169_169700

variable {a : ℕ → ℝ}
variable (h_geom_seq : ∀ n, a n > 0)
variable (h_arith : a 5 * a 6 = 10)

theorem log_geom_seq_sum :
  ∑ k in Finset.range 10, Real.log (a (k+1)) = 5 :=
by
  sorry

end log_geom_seq_sum_l169_169700


namespace total_players_l169_169233

-- Definitions based on problem conditions.
def players_kabadi : Nat := 10
def players_kho_kho_only : Nat := 20
def players_both_games : Nat := 5

-- Proof statement for the total number of players.
theorem total_players : (players_kabadi + players_kho_kho_only - players_both_games) = 25 := by
  sorry

end total_players_l169_169233


namespace turnback_difference_sum_l169_169760

theorem turnback_difference_sum (m n : ℤ) : 
  let sequence := λ k, match k % 6 with
    | 0 => m
    | 1 => n
    | 2 => n - m
    | 3 => -m
    | 4 => -n
    | 5 => -n + m
    | _ => 0 -- This case can't be hit
  in 
  (sequence 0 + sequence 1 + sequence 2) = 2 * n :=
by sorry

end turnback_difference_sum_l169_169760


namespace correct_proposition_l169_169006

section
  variables {a b : Type} {α : set Type}
  variable parallel : (a → b → Prop)
  variable perpendicular : (a → b → Prop)
  variable contained : (b → α → Prop)

  -- Hypotheses for propositions
  hypothesis prop1 : parallel a b ∧ contained b α → parallel a α
  hypothesis prop2 : parallel a α ∧ contained b α → parallel a b
  hypothesis prop3 : parallel a α ∧ parallel b α → parallel a b
  hypothesis prop4 : perpendicular a α ∧ parallel b α → perpendicular a b

  -- Statement to be proven
  theorem correct_proposition : prop1 = False ∧ prop2 = False ∧ prop3 = False ∧ prop4 = True :=
  by
    sorry
end

end correct_proposition_l169_169006


namespace convex_polygon_has_covering_triple_l169_169854

-- Assuming the existence of convex polygon type and circle coverage definition
structure Polygon where
  vertices : List Point

def convex (poly : Polygon) : Prop := sorry
def covers (circle : Circle) (poly : Polygon) : Prop := sorry
def passThrough (circle : Circle) (p q r : Point) : Prop := sorry
def consecutive (poly : Polygon) (p q r : Point) : Prop := sorry

theorem convex_polygon_has_covering_triple (poly : Polygon) (h_convex : convex poly) :
  ∃ (P Q R : Point), consecutive poly P Q R ∧
                     ∃ circle, passThrough circle P Q R ∧ covers circle poly :=
  sorry

end convex_polygon_has_covering_triple_l169_169854


namespace sin_alpha_plus_7pi_over_6_l169_169333

theorem sin_alpha_plus_7pi_over_6 (α : ℝ) (h : sin (π / 3 + α) + sin α = 4 * sqrt 3 / 5) : 
  sin (α + 7 * π / 6) = -4 / 5 :=
sorry

end sin_alpha_plus_7pi_over_6_l169_169333


namespace revised_multiplication_table_odd_fraction_l169_169070

theorem revised_multiplication_table_odd_fraction :
  let factors := Finset.range 16
  let is_odd (n : ℕ) := n % 2 = 1
  let total_products := factors.card * factors.card
  let odd_factors := factors.filter is_odd
  let odd_products := odd_factors.card * odd_factors.card
  round (odd_products / total_products) = 0.25 :=
by
  sorry

end revised_multiplication_table_odd_fraction_l169_169070


namespace greatest_sum_of_consecutive_integers_product_less_500_l169_169999

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l169_169999


namespace profit_percentage_is_30_l169_169591

-- Define the cost price and selling price.
def cost_price : ℝ := 350
def selling_price : ℝ := 455

-- Define the condition that the percentage of profit.
def percentage_profit (CP SP : ℝ) : ℝ := ((SP - CP) / CP) * 100

-- The statement we want to prove:
theorem profit_percentage_is_30 (CP SP : ℝ) (h1 : CP = 350) (h2 : SP = 455) :
  percentage_profit CP SP = 30 :=
by 
  rw [h1, h2]
  unfold percentage_profit
  simp
  sorry

end profit_percentage_is_30_l169_169591


namespace ab_cd_sum_l169_169479

-- Define the function f and the given points conditionally
def f : ℕ → ℕ
| 1 := 2
| 2 := 4
| 3 := 6
| 4 := 3
| _ := 0  -- the rest values if needed

-- Prove that the points (1, 4) and (4, 6) are on the graph of y = f(f(x))
theorem ab_cd_sum : f (f 1) = 4 ∧ f (f 4) = 6 ∧ (1 * 4 + 4 * 6 = 28) :=
by
  have h1 : f 1 = 2 := rfl
  have h2 : f 2 = 4 := rfl
  have h3 : f 3 = 6 := rfl
  have h4 : f 4 = 3 := rfl
  have hff1 : f (f 1) = f 2 := by rw [h1]
  have hff1' : f (f 1) = 4 := by rw [hff1, h2]
  have hff4 : f (f 4) = f 3 := by rw [h4]
  have hff4' : f (f 4) = 6 := by rw [hff4, h3]
  have sum : (1 * 4 + 4 * 6 = 28) := rfl
  exact ⟨hff1', hff4', sum⟩

end ab_cd_sum_l169_169479


namespace probability_five_chords_form_convex_pentagon_l169_169476

noncomputable def probability_convex_pentagon (n k : ℕ) : ℚ :=
(combinatorics.choose n k) / (combinatorics.choose (combinatorics.choose 7 2) 5)

theorem probability_five_chords_form_convex_pentagon :
  probability_convex_pentagon 7 5 = 1 / 969 := sorry

end probability_five_chords_form_convex_pentagon_l169_169476


namespace greatest_sum_consecutive_integers_lt_500_l169_169994

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169994


namespace ratio_kid_to_adult_ticket_l169_169478

theorem ratio_kid_to_adult_ticket (A : ℝ) : 
  (6 * 5 + 2 * A = 50) → (5 / A = 1 / 2) :=
by
  sorry

end ratio_kid_to_adult_ticket_l169_169478


namespace no_solution_exists_l169_169666

   theorem no_solution_exists (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : 
     ¬ (3 / a + 4 / b = 12 / (a + b)) := 
   sorry
   
end no_solution_exists_l169_169666


namespace third_client_multiple_l169_169272

theorem third_client_multiple :
  let dashboard := 4000
  let first_client := 1 / 2 * dashboard
  let second_client := (2 / 5) * first_client + first_client
  let total_dashboard := dashboard + first_client + second_client
  ∃ x, total_dashboard + x * (first_client + second_client) = 18400 ∧ x = 2 :=
by
  let dashboard := 4000
  let first_client := 1 / 2 * dashboard
  let second_client := (2 / 5) * first_client + first_client
  let total_dashboard := dashboard + first_client + second_client
  use 2
  simp
  sorry

end third_client_multiple_l169_169272


namespace expected_total_bunnies_after_matings_l169_169119

theorem expected_total_bunnies_after_matings :
  (Marlon_had : 30 = number_of_initial_bunnies) (gave_away : 12 = (2 / 5) * number_of_initial_bunnies)
  (remaining_bunnies : 18 = number_of_initial_bunnies - (2 / 5) * number_of_initial_bunnies)
  (first_mating_avg : 2 = (1 + 2 + 3) / 3)
  (second_mating_avg : 3 = (2 + 3 + 4) / 3)
  (expected_first_mating_kittens : 36 = remaining_bunnies * first_mating_avg)
  (expected_second_mating_kittens : 54 = remaining_bunnies * second_mating_avg) :
  108 = remaining_bunnies + expected_first_mating_kittens + expected_second_mating_kittens := 
sorry

end expected_total_bunnies_after_matings_l169_169119


namespace find_b_l169_169386

theorem find_b (n : ℝ) (b : ℝ) (h1 : n = 2 ^ 0.25) (h2 : n ^ b = 8) : b = 12 :=
by
  sorry

end find_b_l169_169386


namespace parallel_line_length_l169_169072

def isosceles_triangle (A B C : Type) [has_length has_area] : Prop :=
  is_triangle A B C ∧ (length A B = length B C)

theorem parallel_line_length (A B C D E : Type) [has_length has_area] 
  (h_iso : isosceles_triangle A B C)
  (base_length : length A C = 24) 
  (parallel_line_area_ratio : area (trilateral A D E) = 1/4 * area (trilateral A B C))
  (parallel_to_base : parallel D E A C) :
  length D E = 12 :=
sorry

end parallel_line_length_l169_169072


namespace sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l169_169349

theorem sin_pi_six_minus_alpha_eq_one_third_cos_two_answer
  (α : ℝ) (h1 : Real.sin (π / 6 - α) = 1 / 3) :
  2 * Real.cos (π / 6 + α / 2) ^ 2 - 1 = 1 / 3 := by
  sorry

end sin_pi_six_minus_alpha_eq_one_third_cos_two_answer_l169_169349


namespace greatest_sum_of_consecutive_integers_l169_169960

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169960


namespace cookie_combinations_l169_169237

theorem cookie_combinations (total_cookies kinds : Nat) (at_least_one : kinds > 0 ∧ ∀ k : Nat, k < kinds → k > 0) : 
  (total_cookies = 8 ∧ kinds = 4) → 
  (∃ comb : Nat, comb = 34) := 
by 
  -- insert proof here 
  sorry

end cookie_combinations_l169_169237


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169973

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169973


namespace total_wheels_in_three_garages_l169_169511

theorem total_wheels_in_three_garages : 
  let garage1_bicycles := 5
  let garage1_missing_bicycle_wheels := 2
  let garage1_tricycles := 6
  let garage1_unicycles := 9
  let garage1_quadracycles := 3
  let garage2_bicycles := 2
  let garage2_tricycle := 1
  let garage2_unicycles := 3
  let garage2_quadracycles := 4
  let garage2_pentacycles := 2
  let garage2_missing_pentacycle_wheels := 2
  let garage3_bicycles := 3
  let garage3_tricycles := 4
  let garage3_unicycles := 2
  let garage3_missing_unicycle_wheels := 1
  let garage3_quadracycles := 2
  let garage3_pentacycle := 1
  let garage3_hexacycle_with_sidecar := 1
  let garage3_missing_hexacycle_wheel := 1
  (2*(garage1_bicycles - garage1_missing_bicycle_wheels) + garage1_missing_bicycle_wheels
  + 3*garage1_tricycles
  + 1*garage1_unicycles
  + 4*garage1_quadracycles)
  + (2*garage2_bicycles
  + 3*garage2_tricycle
  + 1*garage2_unicycles
  + 4*garage2_quadracycles
  + 5*(garage2_pentacycles - (garage2_missing_pentacycle_wheels // 2)) + (garage2_missing_pentacycle_wheels mod 2))
  + (2*garage3_bicycles
  + 3*garage3_tricycles
  + 1*(garage3_unicycles - garage3_missing_unicycle_wheels) + garage3_missing_unicycle_wheels
  + 4*garage3_quadracycles
  + 5*garage3_pentacycle
  + (6*garage3_hexacycle_with_sidecar + garage3_hexacycle_with_sidecar - garage3_missing_hexacycle_wheel))
  = 119 := 
by
  sorry

end total_wheels_in_three_garages_l169_169511


namespace number_of_terms_in_arithmetic_sequence_l169_169293

theorem number_of_terms_in_arithmetic_sequence 
  (a : ℕ)
  (d : ℕ)
  (an : ℕ)
  (h1 : a = 3)
  (h2 : d = 4)
  (h3 : an = 47) :
  ∃ n : ℕ, an = a + (n - 1) * d ∧ n = 12 :=
by
  sorry

end number_of_terms_in_arithmetic_sequence_l169_169293


namespace combined_weight_is_correct_l169_169663

def EvanDogWeight := 63
def IvanDogWeight := EvanDogWeight / 7
def CombinedWeight := EvanDogWeight + IvanDogWeight

theorem combined_weight_is_correct 
: CombinedWeight = 72 :=
by 
  sorry

end combined_weight_is_correct_l169_169663


namespace f_inv_at_4_l169_169875

def f : ℝ → ℝ := sorry
def f_inv : ℝ → ℝ := inverse f

axiom symmetry_property (a b : ℝ) : f (2 * a - 1) = 2 * b - 2 → f (4) = 0 → (a, b) = (1, 2)
axiom f_has_inverse : ∃ f_inv, f (f_inv x) = x ∧ f_inv (f y) = y
axiom f_at_4 : f 4 = 0

theorem f_inv_at_4 : f_inv 4 = -2 :=
by
  apply_theorems f_inv_at_4 sorry

end f_inv_at_4_l169_169875


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169970

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169970


namespace initial_boggies_count_l169_169605

def train_speed (n : ℕ) := 15 * n / 9
def train_speed_after_detach (n : ℕ) := 15 * (n - 1) / 8.25

theorem initial_boggies_count : 
  ∀ (n : ℕ), train_speed n = train_speed_after_detach n → n = 12 :=
by
  intro n
  assume h : train_speed n = train_speed_after_detach n
  sorry

end initial_boggies_count_l169_169605


namespace number_of_painting_schemes_l169_169607

theorem number_of_painting_schemes :
  let vertices := ({'A', 'B', 'C', 'D', 'E', 'F'} : set char),
      colors := ({'c1', 'c2', 'c3', 'c4', 'c5'} : set char),
      neq_color := (λ (v1 v2 : char), v1 ≠ v2) in
  ∃ (f : char → char), 
    (∀ v, v ∈ vertices → f v ∈ colors) ∧
    (∀ (v1 v2 : char), (v1, v2) ∈ ({('A', 'B'), ('A', 'C'), ('B', 'C'),
                                   ('D', 'E'), ('D', 'F'), ('E', 'F'),
                                   ('A', 'D'), ('B', 'E'), ('C', 'F'),
                                   ('A', 'E'), ('A', 'F'), ('B', 'D'),
                                   ('B', 'F'), ('C', 'D'), ('C', 'E')} : set (char × char))
                                   → neq_color (f v1) (f v2))
    → ∃ n, n = 1920 := sorry

end number_of_painting_schemes_l169_169607


namespace partition_inequality_l169_169827

-- Definition of b_n (for sets of natural numbers)
def b (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual definition/formula of b_n

-- Definition of c_n (partitions with at least two elements)
def c (n : ℕ) : ℕ := 
  sorry -- Placeholder for the actual definition/formula of c_n

-- The statement of the theorem to be proven
theorem partition_inequality (n : ℕ) : 
  c n = ∑ i in finset.range n, (-1) ^ (i + 1) * b (n-i) :=
sorry

end partition_inequality_l169_169827


namespace simplify_expression_l169_169138

theorem simplify_expression (x y : ℝ) (h1 : x = 1) (h2 : y = 2) : 
  ((x + y) * (x - y) - (x - y)^2 + 2 * y * (x - y)) / (4 * y) = -1 :=
by
  sorry

end simplify_expression_l169_169138


namespace inner_circle_radius_is_sqrt_2_l169_169263

noncomputable def radius_of_inner_circle (side_length : ℝ) : ℝ :=
  let semicircle_radius := side_length / 4
  let distance_from_center_to_semicircle_center :=
    Real.sqrt ((side_length / 2) ^ 2 + (side_length / 2) ^ 2)
  let inner_circle_radius := (distance_from_center_to_semicircle_center - semicircle_radius)
  inner_circle_radius

theorem inner_circle_radius_is_sqrt_2 (side_length : ℝ) (h: side_length = 4) : 
  radius_of_inner_circle side_length = Real.sqrt 2 :=
by
  sorry

end inner_circle_radius_is_sqrt_2_l169_169263


namespace vector_sum_eq_l169_169042

def vector_a : (ℝ × ℝ) := (2, 1)
def vector_b : (ℝ × ℝ) := (-3, 4)

theorem vector_sum_eq : 3 • vector_a + 4 • vector_b = (-6, 19) :=
by
  sorry

end vector_sum_eq_l169_169042


namespace time_in_future_is_4_l169_169916

def current_time := 5
def future_hours := 1007
def modulo := 12
def future_time := (current_time + future_hours) % modulo

theorem time_in_future_is_4 : future_time = 4 := by
  sorry

end time_in_future_is_4_l169_169916


namespace sum_of_tetrahedron_properties_eq_14_l169_169088

-- Define the regular tetrahedron properties
def regular_tetrahedron_edges : ℕ := 6
def regular_tetrahedron_vertices : ℕ := 4
def regular_tetrahedron_faces : ℕ := 4

-- State the theorem that needs to be proven
theorem sum_of_tetrahedron_properties_eq_14 :
  regular_tetrahedron_edges + regular_tetrahedron_vertices + regular_tetrahedron_faces = 14 :=
by
  sorry

end sum_of_tetrahedron_properties_eq_14_l169_169088


namespace projection_sum_inequality_l169_169040

variables {n m : ℕ}
variables (a : Fin n → ℝ) (b : Fin m → ℝ)
variables (α : Fin n → ℝ) (β : Fin m → ℝ)

theorem projection_sum_inequality
  (H : ∀ (φ : ℝ), ∑ i, (a i) * |cos (φ - α i)| ≤ ∑ j, (b j) * |cos (φ - β j)|) :
  ∑ i, a i ≤ ∑ j, b j :=
begin
  -- The proof would go here
  sorry
end

end projection_sum_inequality_l169_169040


namespace abs_diff_squares_105_95_l169_169948

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l169_169948


namespace minimum_apples_to_guarantee_18_one_color_l169_169575

theorem minimum_apples_to_guarantee_18_one_color :
  let red := 32
  let green := 24
  let yellow := 22
  let blue := 15
  let orange := 14
  ∀ n, (n >= 81) →
  (∃ red_picked green_picked yellow_picked blue_picked orange_picked : ℕ,
    red_picked + green_picked + yellow_picked + blue_picked + orange_picked = n
    ∧ red_picked ≤ red ∧ green_picked ≤ green ∧ yellow_picked ≤ yellow ∧ blue_picked ≤ blue ∧ orange_picked ≤ orange
    ∧ (red_picked = 18 ∨ green_picked = 18 ∨ yellow_picked = 18 ∨ blue_picked = 18 ∨ orange_picked = 18)) :=
by {
  -- The proof is omitted for now.
  sorry
}

end minimum_apples_to_guarantee_18_one_color_l169_169575


namespace max_value_f_on_interval_l169_169158

def f (x : ℝ) : ℝ := x * Real.cos x - Real.sin x

theorem max_value_f_on_interval : 
  ∀ x ∈ Set.Icc (-Real.pi) 0, 
  f x ≤ f (-Real.pi) := 
by 
  sorry

end max_value_f_on_interval_l169_169158


namespace angle_BAC_equilateral_triangle_l169_169282

/-- 
  Consider a circle with an equilateral triangle and a regular pentagon inscribed in it. 
  The triangle and pentagon share a common vertex. Label the vertices of the triangle as 
  A, B, and C where A is the shared vertex, and the vertices of the pentagon as A, P1, P2, P3, P4 in clockwise order. 
  Prove the measure of ∠BAC within the triangle is 60°.
-/
theorem angle_BAC_equilateral_triangle :
  ∀ (A B C P1 P2 P3 P4 : Type) [Circumcircle A B C] [Circumcircle A P1 P2 P3 P4],
  (equilateral_triangle A B C) →
  (regular_pentagon A P1 P2 P3 P4) →
  ∠ BAC = 60 :=
by
  -- Conditions 
  intros A B C P1 P2 P3 P4 hCircumcircle1 hCircumcircle2 hEqTri hRegPent
  
  -- Proof sketch: 
  -- Using the definitions of equilateral triangle and regular pentagon, we show that the measure of ∠BAC is 60°.
  -- In an equilateral triangle, each angle measures 60°.
  -- Since ∠BAC is an internal angle of the equilateral triangle ABC, it is equal to 60°.
  sorry

end angle_BAC_equilateral_triangle_l169_169282


namespace proof_of_geometry_problem_l169_169401

-- Define the given conditions
variables (A B C I H B1 C1 B2 C2 K A1 : Type)
variables (triangle_ABC : Triangle A B C)
variables (acute_triangle : ∀ {X Y Z : Type}, Triangle X Y Z → acute X Y Z)
variables (incenter : incenter I A B C)
variables (orthocenter : orthocenter H A B C)
variables (midpoints : midpoint B1 A C ∧ midpoint C1 A B)
variables (intersects : ∃ B2 ≠ B, (B1I intersects AB at B2))
variables (extension_intersect : ∃ C2, ray (C1I) ∩ (extension AC) = C2)
variables (line_intersects_bc : (B2C2) ∩ BC = K)
variables (circumcenter : circumcenter A1 B H C)

-- Define the theorem to be proved
theorem proof_of_geometry_problem :
  collinear A I A1 ↔ triangle_area_eq (area_triangle B K B2) (area_triangle C K C2) :=
sorry

end proof_of_geometry_problem_l169_169401


namespace correct_statements_count_l169_169681

noncomputable def f (x : ℝ) := (1 / 2) * Real.sin (2 * x)
noncomputable def g (x : ℝ) := (1 / 2) * Real.sin (2 * x + Real.pi / 4)

theorem correct_statements_count :
  let complexity := True
    -- Condition 1: Smallest positive period of f(x) is 2π
    (∀ x : ℝ, f (x + 2 * Real.pi) = f x) = False ∧
    -- Condition 2: f(x) is monotonically increasing on [-π/4, π/4]
    (∀ x y : ℝ, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y) = True ∧
    -- Condition 3: Range of f(x) when x ∈ [-π/6, π/3]
    (∀ y : ℝ, ∀ x : ℝ, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 3 → f x = y ↔ -Real.sqrt(3) / 4 ≤ y ∧ y ≤ Real.sqrt(3) / 4) = False ∧
    -- Condition 4: Graph of f(x) can be obtained by shifting g(x) to the left by π/8
    (∀ x : ℝ, f x = g (x + Real.pi / 8)) = False →
  true := sorry

end correct_statements_count_l169_169681


namespace digit_150_of_decimal_3_div_11_l169_169531

theorem digit_150_of_decimal_3_div_11 : 
  (let digits := [2, 7] in digits[(150 % digits.length)]) = 7 :=
by
  sorry

end digit_150_of_decimal_3_div_11_l169_169531


namespace centroid_positions_count_l169_169877

noncomputable def centroid_possible_positions (A B C : (ℕ × ℕ)) : ℕ :=
  if (A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ 
      A.1 ≤ 8 ∧ A.2 ≤ 8 ∧ 
      B.1 ≤ 8 ∧ B.2 ≤ 8 ∧ 
      C.1 ≤ 8 ∧ C.2 ≤ 8) then 
    let G_x := (A.1 + B.1 + C.1) / 3
    let G_y := (A.2 + B.2 + C.2) / 3
    (G_x, G_y)
  else (0, 0)

theorem centroid_positions_count : 
  ∃ n, n = 529 ∧ ∀ (A B C : ℕ × ℕ), 
  A ≠ B -> B ≠ C -> A ≠ C ->
  A.1 ≤ 8 -> A.2 ≤ 8 -> 
  B.1 ≤ 8 -> B.2 ≤ 8 -> 
  C.1 ≤ 8 -> C.2 ≤ 8 -> 
  centroid_possible_positions A B C ∈ 
  {(p, q) | ∃ (i j : ℤ), 1 ≤ i ∧ i ≤ 23 ∧ 1 ≤ j ∧ j ≤ 23 ∧ 
                  p = (i : ℕ) ∧ q = (j : ℕ)} := 
sorry

end centroid_positions_count_l169_169877


namespace roots_product_l169_169831

noncomputable def quadratic_eq := ∀ x : ℝ, x^2 - x - 4 = 0

theorem roots_product {x₁ x₂ : ℝ}
  (h₁ : quadratic_eq x₁)
  (h₂ : quadratic_eq x₂) :
  (x₁^5 - 20 * x₁) * (x₂^4 + 16) = 1296 :=
sorry

end roots_product_l169_169831


namespace total_fencing_cost_l169_169909

theorem total_fencing_cost (A_cost_per_meter B_cost_per_meter : ℝ) (area : ℝ) (ratio_long_short : ℝ) :
  A_cost_per_meter = 0.80 → B_cost_per_meter = 1.20 →
  area = 3750 →
  ratio_long_short = 3 / 2 →
  let x := real.sqrt (area / (3 * 2)) in
  let length := 3 * x in
  let width := 2 * x in
  let diagonal := real.sqrt (length^2 + width^2) in
  let cost_A := 2 * length * A_cost_per_meter in
  let cost_B := (2 * width * B_cost_per_meter + diagonal * B_cost_per_meter) in
  cost_A + cost_B ≈ 348.17 :=
by intros A_cost_per_meter B_cost_per_meter area ratio_long_short
   sorry

end total_fencing_cost_l169_169909


namespace calculate_expression_l169_169630

theorem calculate_expression : -1^4 * 8 - 2^3 / (-4) * (-7 + 5) = -12 := 
by 
  /-
  In Lean, we typically perform arithmetic simplifications step by step;
  however, for the purpose of this example, only stating the goal:
  -/
  sorry

end calculate_expression_l169_169630


namespace evaluate_expression_l169_169661

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l169_169661


namespace coin_bag_amount_l169_169243

-- Definitions based on the conditions
def num_pennies (x : ℕ) : ℕ := x
def num_dimes (x : ℕ) : ℕ := 2 * x
def num_quarters (x : ℕ) : ℕ := 6 * x

-- The total value in the bag
def total_value (x : ℕ) : ℚ := 0.01 * x + 0.20 * (num_dimes x).toRat + 1.50 * (num_quarters x).toRat

-- The problem statement to be proved
theorem coin_bag_amount (x : ℕ) : total_value x = 342.0 ↔ ∃ k : ℕ, 1.71 * (k : ℚ) = 342 := 
by sorry

end coin_bag_amount_l169_169243


namespace count_valid_numbers_l169_169049

-- Define the range of numbers we are analyzing
def range := finset.Icc 1 500

-- Define the condition for being a multiple of both 4 and 6 (i.e., multiple of 12)
def is_multiple_of_12 (n : ℕ) : Prop := n % 12 = 0

-- Define the condition for not being a multiple of 5
def not_multiple_of_5 (n : ℕ) : Prop := ¬ (n % 5 = 0)

-- Define the condition for not being a multiple of 9
def not_multiple_of_9 (n : ℕ) : Prop := ¬ (n % 9 = 0)

-- Define the final set of numbers according to the conditions specified
def valid_numbers := range.filter (λ n, is_multiple_of_12 n ∧ not_multiple_of_5 n ∧ not_multiple_of_9 n)

-- Define the theorem we want to prove
theorem count_valid_numbers : valid_numbers.card = 26 :=
by
  sorry

end count_valid_numbers_l169_169049


namespace point_inside_circle_l169_169021

theorem point_inside_circle (r d : ℝ) (P_inside : d < r) : P_inside := sorry

-- Given conditions in the problem
def radius : ℝ := 8
def distance_to_center : ℝ := 6

-- Formalize the given condition as an assumption in Lean
lemma point_P_inside_circle : distance_to_center < radius := by
  have P_inside : distance_to_center < radius := by norm_num
  exact P_inside

#check point_inside_circle radius distance_to_center point_P_inside_circle

end point_inside_circle_l169_169021


namespace monotonic_increase_inequality_proof_l169_169363

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := real.exp x - a * x^2

theorem monotonic_increase (a : ℝ) :
  (∀ x : ℝ, 0 < x → f x a ≥ 0) ↔ a ≤ real.exp 1 / 2 :=
sorry

theorem inequality_proof (a : ℝ) (h₀ : a ≤ 1) :
  ∀ x : ℝ, 0 ≤ x → f x a ≥ (real.exp 1 - 2) * x + a :=
sorry

end monotonic_increase_inequality_proof_l169_169363


namespace k_range_for_two_zeros_of_f_l169_169020

noncomputable def f (x k : ℝ) : ℝ := x^2 - x * (Real.log x) - k * (x + 2) + 2

theorem k_range_for_two_zeros_of_f :
  ∀ k : ℝ, (∃ x1 x2 : ℝ, (1/2 < x1) ∧ (x1 < x2) ∧ f x1 k = 0 ∧ f x2 k = 0) ↔ 1 < k ∧ k ≤ (9 + 2 * Real.log 2) / 10 :=
by
  sorry

end k_range_for_two_zeros_of_f_l169_169020


namespace circumcircles_tangent_l169_169856
  
theorem circumcircles_tangent 
  {A B C P S T O M : Point} 
  (hM_center_pst : ∀ (X : Point), X ∈ {P, S, T} → dist M X = dist M P)
  (hA_O_perp_EF : AO ⊥ EF)
  (hAO_parallel_MP : AO ∥ MP) :
  tangents_at O (circumcircle ABC) (circumcircle PST) :=
sorry

end circumcircles_tangent_l169_169856


namespace measure_angle_E_l169_169611

open Real

-- Define the properties and the problem
structure ConvexPentagon (A B C D E : Type) :=
(side_length : ℝ)
(angleA : ℝ)
(angleB : ℝ)

def pentagon ABCDE := ConvexPentagon (A B C D E)
-- Conditions
axiom equal_sides : ∀ (p : ConvexPentagon A B C D E), 
  p.side_length = p.side_length -- All sides are equal
axiom angle_A_90 : ∀ (p : ConvexPentagon A B C D E), 
  p.angleA = 90
axiom angle_B_90 : ∀ (p : ConvexPentagon A B C D E), 
  p.angleB = 90 

-- The lean statement declaring to prove the angle E is 150 degrees
theorem measure_angle_E (ABCDE : ConvexPentagon A B C D E)
  (h1 : equal_sides ABCDE)
  (h2 : angle_A_90 ABCDE)
  (h3 : angle_B_90 ABCDE) : 
  ∃ (angleE : ℝ), angleE = 150 :=
sorry -- Proof omitted

end measure_angle_E_l169_169611


namespace smallest_number_of_students_l169_169621

-- Define the conditions as given in the problem
def eight_to_six_ratio : ℕ × ℕ := (5, 3) -- ratio of 8th-graders to 6th-graders
def eight_to_nine_ratio : ℕ × ℕ := (7, 4) -- ratio of 8th-graders to 9th-graders

theorem smallest_number_of_students (a b c : ℕ)
  (h1 : a = 5 * b) (h2 : b = 3 * c) (h3 : a = 7 * c) : a + b + c = 76 := 
sorry

end smallest_number_of_students_l169_169621


namespace negation_of_all_cuboids_are_prisms_with_four_lateral_faces_l169_169900

-- Assume that Cuboid is a type and PrismWithFourLateralFaces is a predicate on Cuboid
def Cuboid := Type
def PrismWithFourLateralFaces (c : Cuboid) := Prop

-- The problem statement: for all cuboids, they are prisms with four lateral faces
def all_cuboids_are_prisms_with_four_lateral_faces := ∀ (c : Cuboid), PrismWithFourLateralFaces c

-- We need to prove the negation
theorem negation_of_all_cuboids_are_prisms_with_four_lateral_faces :
  ¬ all_cuboids_are_prisms_with_four_lateral_faces ↔ ∃ (c : Cuboid), ¬ PrismWithFourLateralFaces c :=
begin
  sorry,
end

end negation_of_all_cuboids_are_prisms_with_four_lateral_faces_l169_169900


namespace value_of_2a_plus_b_l169_169384

theorem value_of_2a_plus_b (a b : ℝ) (h1 : real.sqrt (a + 5) = -3) (h2 : real.cbrt b = -2) : 2 * a + b = 0 :=
sorry

end value_of_2a_plus_b_l169_169384


namespace greatest_sum_consecutive_integers_lt_500_l169_169996

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169996


namespace equation_ellipse_C_minimum_value_MN_l169_169001

-- Given conditions
def ellipse (a b : ℝ) := (a > b ∧ b > 0)
def vertex_at_0_2 (b : ℝ) := (0, b)
def eccentricity_condition (a b c : ℝ) := (c = (real.sqrt (a^2 - b^2)) / a)
def circle := ∀ x y : ℝ, x^2 + y^2 = 1

-- Goals
theorem equation_ellipse_C : 
  ∀ (a b : ℝ), ellipse a b → vertex_at_0_2 b = 2 → eccentricity_condition a b (real.sqrt 5 / 3) → 
  (∀ x y: ℝ, (x^2 / a^2) + (y^2 / b^2) = 1) = (∀ x y: ℝ, (x^2 / 9) + (y^2 / 4) = 1) :=
sorry

theorem minimum_value_MN :
  ∀ (P M N : ℝ × ℝ), 
  (∀ x y: ℝ, (x^2 / 9) + (y^2 / 4) = 1) → circle → 
  P ∈ ∀ x y: ℝ, (x^2 / 9) + (y^2 / 4) = 1 → 
  (M ∈ ∃ y: ℝ, M = (0, 1/y)) → 
  (N ∈ ∃ x: ℝ, N = (1/x, 0)) → 
  min_value (abs (dist M N)) = 5/6 :=
sorry

end equation_ellipse_C_minimum_value_MN_l169_169001


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169978

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169978


namespace interior_edges_sum_l169_169256

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l169_169256


namespace probability_of_even_product_l169_169524

theorem probability_of_even_product :
  let spinner1 := [6, 7, 8, 9]
  let spinner2 := [10, 11, 12, 13, 14]
  (1 - (4 / ((4:ℚ) * 5))) = 4 / 5 := by
  sorry

end probability_of_even_product_l169_169524


namespace minimum_value_of_expression_l169_169369

theorem minimum_value_of_expression
  (a b : ℝ)
  (circle1 : ∀ x y : ℝ, x^2 + y^2 = 4)
  (circle2 : ∀ x y : ℝ, (x-1)^2 + (y-3)^2 = 4)
  (P : ∀ M N : ℝ × ℝ, M ∈ {(x, y) | circle1 x y} → N ∈ {(x, y) | circle2 x y} → (a, b) ∈ line_through M N → (dist (a, b) M = dist (a, b) N))
  (line_eq : a + 3 * b - 5 = 0)
: (a^2 + b^2 - 6 * a - 4 * b + 13 = 8 / 5) := sorry

end minimum_value_of_expression_l169_169369


namespace Julio_limes_expense_l169_169780

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end Julio_limes_expense_l169_169780


namespace interior_edges_sum_l169_169257

theorem interior_edges_sum 
  (frame_thickness : ℝ)
  (frame_area : ℝ)
  (outer_length : ℝ)
  (frame_thickness_eq : frame_thickness = 2)
  (frame_area_eq : frame_area = 32)
  (outer_length_eq : outer_length = 7) 
  : ∃ interior_edges_sum : ℝ, interior_edges_sum = 8 := 
by
  sorry

end interior_edges_sum_l169_169257


namespace abs_diff_squares_105_95_l169_169945

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l169_169945


namespace savings_percentage_is_correct_l169_169589

-- Definitions for given conditions
def jacket_original_price : ℕ := 100
def shirt_original_price : ℕ := 50
def shoes_original_price : ℕ := 60

def jacket_discount : ℝ := 0.30
def shirt_discount : ℝ := 0.40
def shoes_discount : ℝ := 0.25

-- Definitions for savings
def jacket_savings : ℝ := jacket_original_price * jacket_discount
def shirt_savings : ℝ := shirt_original_price * shirt_discount
def shoes_savings : ℝ := shoes_original_price * shoes_discount

-- Definition for total savings and total original cost
def total_savings : ℝ := jacket_savings + shirt_savings + shoes_savings
def total_original_cost : ℕ := jacket_original_price + shirt_original_price + shoes_original_price

-- The theorems to be proven
theorem savings_percentage_is_correct : (total_savings / total_original_cost * 100) = 30.95 := by
  sorry

end savings_percentage_is_correct_l169_169589


namespace greatest_sum_of_consecutive_integers_l169_169966

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169966


namespace omega_range_l169_169703

theorem omega_range (ω : ℝ) :
  (∀ x1 x2 : ℝ, - (real.pi / 3) ≤ x1 ∧ x1 < x2 ∧ x2 ≤ real.pi / 3 → sin (ω * x1) < sin (ω * x2)) →
  0 < ω ∧ ω ≤ 3 / 2 :=
sorry

end omega_range_l169_169703


namespace part_I_period_part_I_monotonicity_interval_part_II_range_l169_169025

noncomputable def f (x : ℝ) : ℝ :=
  4 * Real.cos x * Real.sin (x + Real.pi / 6) - 1

theorem part_I_period : ∀ x, f (x + Real.pi) = f x := by
  sorry

theorem part_I_monotonicity_interval (k : ℤ) :
  ∀ x, k * Real.pi + Real.pi / 6 ≤ x ∧ x ≤ k * Real.pi + 2 * Real.pi / 3 → f (x + Real.pi) = f x := by
  sorry

theorem part_II_range :
  ∀ x, -Real.pi / 6 ≤ x ∧ x ≤ Real.pi / 4 → f x ∈ Set.Icc (-1) 2 := by
  sorry

end part_I_period_part_I_monotonicity_interval_part_II_range_l169_169025


namespace john_purchased_small_bottles_l169_169432

theorem john_purchased_small_bottles :
  ∃ S : ℕ, (1300 * 1.89 + S * 1.38) / (1300 + S) = 1.7034 ∧ S = 750 :=
by
  use 750
  split
  · apply_eq_of_eq
    -- This is where the proof would go, we'll use sorry
    sorry
  · rfl

end john_purchased_small_bottles_l169_169432


namespace minimum_boxes_to_eliminate_l169_169398

theorem minimum_boxes_to_eliminate (boxes : fin 30 → real)
  (h₁ : boxes.values.to_finset = {0.50, 2, 10, 20, 40, 60, 80, 100, 200, 300, 400, 500, 750, 1000, 800, 1500, 3000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000})
  (h₂ : ∀ i j, (i ≠ j) → (boxes i ≠ boxes j)) :
  (∃ k, k = 7 ∧ ∀ (remaining_boxes : finset (fin 30)),
    remaining_boxes.card = 30 - k →
    (finset.filter (λ x, boxes x ≥ 75000) remaining_boxes).card ≥ 8 → 
    remaining_boxes.card ≤ 23) :=
begin
  sorry
end

end minimum_boxes_to_eliminate_l169_169398


namespace at_most_n_pairs_with_distance_d_l169_169756

theorem at_most_n_pairs_with_distance_d
  (n : ℕ) (hn : n ≥ 3)
  (points : Fin n → ℝ × ℝ)
  (d : ℝ)
  (hd : ∀ i j, i ≠ j → dist (points i) (points j) ≤ d)
  (dmax : ∃ i j, i ≠ j ∧ dist (points i) (points j) = d) :
  ∃ (pairs : Finset (Fin n × Fin n)), ∀ p ∈ pairs, dist (points p.1) (points p.2) = d ∧ pairs.card ≤ n := 
sorry

end at_most_n_pairs_with_distance_d_l169_169756


namespace prove_n_value_l169_169052

theorem prove_n_value (n : ℝ) : (23 ^ (5 * n) = (1 / 23) ^ (2 * n - 47)) → n = 47 / 7 := by
  sorry

end prove_n_value_l169_169052


namespace find_angle_QOR_l169_169187

-- Definitions to capture the problem setup
variables {P Q R : Type} [triangle P Q R]
variables {O : Type} [circle O]

-- Angles in degrees
def degree (a b c : Type) : ℝ := sorry -- Just a placeholder for the actual angle measures

variable (angle_QPR : degree Q P R = 50)

-- The theorem to prove
theorem find_angle_QOR : degree Q O R = 230 :=
by sorry

end find_angle_QOR_l169_169187


namespace range_of_tangent_lines_l169_169182

noncomputable def function_f (x : ℝ) := x * Real.exp x
noncomputable def point_P (t : ℝ) := (t, -1 : ℝ)

theorem range_of_tangent_lines (t : ℝ) :
  (∃ (x0 : ℝ), x0 ≠ t ∧ (function_f x0).derivative (x0 + 1) * Real.exp x0 = -1) ↔ t > Real.exp 2 - 4 :=
begin
  sorry
end

end range_of_tangent_lines_l169_169182


namespace frac_sum_eq_l169_169387

theorem frac_sum_eq (a b : ℝ) (h1 : a^2 + a - 1 = 0) (h2 : b^2 + b - 1 = 0) : 
  (a / b + b / a = 2) ∨ (a / b + b / a = -3) := 
sorry

end frac_sum_eq_l169_169387


namespace curves_segments_intersection_l169_169632

theorem curves_segments_intersection (n m : ℕ) (segments : fin n → ℝ × ℝ × ℝ × ℝ)
  (curves : fin m → ℝ → ℝ → ℝ → Prop)
  (h1 : ∀ i j : fin m, i ≠ j → curves i ≠ curves j)
  (h2 : ∀ i : fin m, ∃ a b : fin n, ∃ x1 x2 y1 y2 : ℝ, 
    curves i (segments a).1 (segments a).2 (segments b).1 ∧
    curves i (segments b).3 (segments b).4 (segments a).3 ∧
    i ≠ j → (segments i).2 ≠ (segments j).4)
  (h3 : ∃ C : ℝ, ∀ n, m ≤ C * n) :
  m = O(n) := sorry

end curves_segments_intersection_l169_169632


namespace magnitude_diff_unit_vectors_l169_169810

variables (a b : ℝ^3) -- Define vector variables a and b in 3-dimensional real space

-- Define the conditions
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def sum_is_one (v1 v2 : ℝ^3) : Prop := ‖v1 + v2‖ = 1

-- State the theorem
theorem magnitude_diff_unit_vectors (a b : ℝ^3) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h_sum : sum_is_one a b) : 
  ‖a - b‖ = real.sqrt 3 :=
sorry

end magnitude_diff_unit_vectors_l169_169810


namespace greatest_sum_consecutive_integers_lt_500_l169_169987

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169987


namespace sufficient_condition_for_perpendicular_l169_169007

variables (m n : Line) (α β : Plane)

def are_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem sufficient_condition_for_perpendicular :
  (are_parallel m n) ∧ (line_perpendicular_to_plane n α) → (line_perpendicular_to_plane m α) :=
sorry

end sufficient_condition_for_perpendicular_l169_169007


namespace functions_monotonic_in_interval_l169_169179

theorem functions_monotonic_in_interval :
  (∀ x ∈ Ioi 0 ∩ Iio (π / 2), deriv (λ x : ℝ, sin x - cos x) x > 0)
  ∧ (∀ x ∈ Ioi 0 ∩ Iio (π / 2), deriv (λ x : ℝ, sin x / cos x) x > 0) :=
by
  sorry

end functions_monotonic_in_interval_l169_169179


namespace magnitude_projection_l169_169797

variables (v w : ℝ^3)
noncomputable def dot_product (x y : ℝ^3) : ℝ := x.1 * y.1 + x.2 * y.2 + x.3 * y.3
noncomputable def magnitude (x : ℝ^3) : ℝ := real.sqrt (dot_product x x)

axiom v_dot_w : dot_product v w = 4
axiom w_magnitude : magnitude w = 8

theorem magnitude_projection : magnitude ((dot_product v w / (magnitude w ^ 2)) • w) = 4 :=
sorry

end magnitude_projection_l169_169797


namespace sum_f_neg_l169_169023

noncomputable def f (x : ℝ) : ℝ := -x - x^3

theorem sum_f_neg {x1 x2 x3 : ℝ}
  (h1 : x1 + x2 > 0)
  (h2 : x2 + x3 > 0)
  (h3 : x3 + x1 > 0) :
  f x1 + f x2 + f x3 < 0 :=
by
  sorry

end sum_f_neg_l169_169023


namespace abs_diff_squares_105_95_l169_169942

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l169_169942


namespace noah_energy_usage_is_1026_l169_169847

noncomputable def total_energy_usage : ℕ := 
  let bedroom_light := 6
  let desktop := 60
  let office_light := 6 * 3
  let laptop := 40
  let living_room_light := 6 * 4
  let tv := 100
  let kitchen_light := 6 * 2
  let microwave := 900
  let bathroom_light := 6 * 5
  let hairdryer := 1000
  
  let bedroom_light_usage := bedroom_light * 2
  let desktop_usage := desktop * 1
  let office_light_usage := office_light * 3
  let laptop_usage := laptop * 2
  let living_room_light_usage := living_room_light * 4
  let tv_usage := tv * 2
  let kitchen_light_usage := kitchen_light * 1
  let microwave_usage := microwave * (20 / 60)
  let bathroom_light_usage := bathroom_light * 1.5
  let hairdryer_usage := hairdryer * (10 / 60)
  
  bedroom_light_usage + desktop_usage + office_light_usage + laptop_usage + living_room_light_usage + 
  tv_usage + kitchen_light_usage + (microwave_usage.to_int) + bathroom_light_usage.to_int + 
  (hairdryer_usage.to_int)

theorem noah_energy_usage_is_1026 : total_energy_usage = 1026 :=
  sorry

end noah_energy_usage_is_1026_l169_169847


namespace regular_heptagon_interior_angle_l169_169200

theorem regular_heptagon_interior_angle :
  ∀ (S : Type) [decidable_instance S] [fintype S], ∀ (polygon : set S), is_regular polygon → card polygon = 7 → 
    (sum_of_interior_angles polygon / 7 = 128.57) :=
by
  intros S dec inst polygon h_reg h_card
  sorry

end regular_heptagon_interior_angle_l169_169200


namespace nested_radical_solution_l169_169655

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l169_169655


namespace janet_earns_more_as_freelancer_l169_169428

-- Definitions for the problem conditions
def current_job_weekly_hours : ℕ := 40
def current_job_hourly_rate : ℕ := 30

def freelance_client_a_hours_per_week : ℕ := 15
def freelance_client_a_hourly_rate : ℕ := 45

def freelance_client_b_hours_project1_per_week : ℕ := 5
def freelance_client_b_hours_project2_per_week : ℕ := 10
def freelance_client_b_hourly_rate : ℕ := 40

def freelance_client_c_hours_per_week : ℕ := 20
def freelance_client_c_rate_range : ℕ × ℕ := (35, 42)

def weekly_fica_taxes : ℕ := 25
def monthly_healthcare_premiums : ℕ := 400
def monthly_increased_rent : ℕ := 750
def monthly_business_phone_internet : ℕ := 150
def business_expense_percentage : ℕ := 10

def weeks_in_month : ℕ := 4

-- Define the calculations
def current_job_monthly_earnings := current_job_weekly_hours * current_job_hourly_rate * weeks_in_month

def freelance_client_a_weekly_earnings := freelance_client_a_hours_per_week * freelance_client_a_hourly_rate
def freelance_client_b_weekly_earnings := (freelance_client_b_hours_project1_per_week + freelance_client_b_hours_project2_per_week) * freelance_client_b_hourly_rate
def freelance_client_c_weekly_earnings := freelance_client_c_hours_per_week * ((freelance_client_c_rate_range.1 + freelance_client_c_rate_range.2) / 2)

def total_freelance_weekly_earnings := freelance_client_a_weekly_earnings + freelance_client_b_weekly_earnings + freelance_client_c_weekly_earnings
def total_freelance_monthly_earnings := total_freelance_weekly_earnings * weeks_in_month

def total_additional_expenses := (weekly_fica_taxes * weeks_in_month) + monthly_healthcare_premiums + monthly_increased_rent + monthly_business_phone_internet

def business_expense_deduction := (total_freelance_monthly_earnings * business_expense_percentage) / 100
def adjusted_freelance_earnings_after_deduction := total_freelance_monthly_earnings - business_expense_deduction
def adjusted_freelance_earnings_after_expenses := adjusted_freelance_earnings_after_deduction - total_additional_expenses

def earnings_difference := adjusted_freelance_earnings_after_expenses - current_job_monthly_earnings

-- The theorem to be proved
theorem janet_earns_more_as_freelancer :
  earnings_difference = 1162 :=
sorry

end janet_earns_more_as_freelancer_l169_169428


namespace chord_length_l169_169895

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end chord_length_l169_169895


namespace triangle_angle_not_greater_than_60_l169_169469

theorem triangle_angle_not_greater_than_60 (A B C : Real) (h1 : A + B + C = 180) 
  : A ≤ 60 ∨ B ≤ 60 ∨ C ≤ 60 :=
by {
  sorry
}

end triangle_angle_not_greater_than_60_l169_169469


namespace divisibility_of_a_b_by_n_l169_169829

theorem divisibility_of_a_b_by_n
  (n : ℕ) (a b : ℤ)
  (h_diff : a ≠ b)
  (h_div : ∀ m : ℕ, n^m ∣ (a^m - b^m)) :
  n ∣ a ∧ n ∣ b := 
sorry

end divisibility_of_a_b_by_n_l169_169829


namespace sol_sells_more_candy_each_day_l169_169871

variable {x : ℕ}

-- Definition of the conditions
def sells_candy (first_day : ℕ) (rate : ℕ) (days : ℕ) : ℕ :=
  first_day + rate * (days - 1) * days / 2

def earns (bars_sold : ℕ) (price_cents : ℕ) : ℕ :=
  bars_sold * price_cents

-- Problem statement in Lean:
theorem sol_sells_more_candy_each_day
  (first_day_sales : ℕ := 10)
  (days : ℕ := 6)
  (price_cents : ℕ := 10)
  (total_earnings : ℕ := 1200) :
  earns (sells_candy first_day_sales x days) price_cents = total_earnings → x = 76 :=
sorry

end sol_sells_more_candy_each_day_l169_169871


namespace root_in_interval_l169_169009

noncomputable def f (a b x : ℝ) : ℝ := a^x + x - b

theorem root_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b ∧ b < 1) : 
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f a b x = 0 :=
by {
  sorry
}

end root_in_interval_l169_169009


namespace equilateral_triangle_condition_l169_169127
noncomputable def triangle_inequality (A B C : Real) (ha : A + B + C = π) : Prop :=
  (sin A + sin B + sin C) / (sin A * sin B * sin C) ≥ 4

theorem equilateral_triangle_condition (A B C : Real) (ha : A + B + C = π) :
  triangle_inequality A B C ha ↔ (A = B ∧ B = C) :=
sorry

end equilateral_triangle_condition_l169_169127


namespace greatest_sum_consecutive_integers_lt_500_l169_169995

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169995


namespace digit_150_of_3_div_11_l169_169539

theorem digit_150_of_3_div_11 :
  let rep := "27"
  ∃ seq : ℕ → ℕ,
  (∀ n, seq (2 * n) = 2) ∧ (∀ n, seq (2 * n + 1) = 7) ∧
  150 % 2 = 0 ∧ seq (150 - 1) = 7 :=
by
  let rep := "27"
  use (λ n : ℕ => if n % 2 = 0 then 2 else 7)
  split
  { intro n
    exact rfl }
  { intro n
    exact rfl }
  { exact rfl }
  { exact rfl }

end digit_150_of_3_div_11_l169_169539


namespace part1_part2_l169_169711

def f (x : ℝ) : ℝ := abs (x - 2)

theorem part1 (x : ℝ) : f x > 4 - abs (x + 1) ↔ x < -3 / 2 ∨ x > 5 / 2 := 
sorry

theorem part2 (a b : ℝ) (ha : 0 < a ∧ a < 1/2) (hb : 0 < b ∧ b < 1/2)
  (h : f (1 / a) + f (2 / b) = 10) : a + b / 2 ≥ 2 / 7 := 
sorry

end part1_part2_l169_169711


namespace no_real_roots_for_pair_2_2_3_l169_169331

noncomputable def discriminant (A B : ℝ) : ℝ :=
  let a := 1 - 2 * B
  let b := -B
  let c := -A + A * B
  b ^ 2 - 4 * a * c

theorem no_real_roots_for_pair_2_2_3 : discriminant 2 (2 / 3) < 0 := by
  sorry

end no_real_roots_for_pair_2_2_3_l169_169331


namespace dirichlet_props_l169_169489

noncomputable def dirichlet_function (x : ℝ) : ℝ :=
  if x ∈ set_of is_rat then 1 else 0

theorem dirichlet_props :
  (∀ x : ℝ, dirichlet_function (dirichlet_function x) = 1) ∧
  (∀ x : ℝ, dirichlet_function x = dirichlet_function (-x)) ∧
  (∀ (x : ℝ) (T : ℚ), T ≠ 0 → dirichlet_function (x + T) = dirichlet_function x) ∧
  (∃ (x1 x2 x3 : ℝ), 
      x1 = - real.sqrt 3 / 3 ∧ x2 = 0 ∧ x3 = real.sqrt 3 / 3 ∧ 
      let A := (x1, dirichlet_function x1) in
      let B := (x2, dirichlet_function x2) in
      let C := (x3, dirichlet_function x3) in
      A.2 = 0 ∧ B.2 = 1 ∧ C.2 = 0 ∧ 
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = (B.1 - C.1)^2 + (B.2 - C.2)^2 ∧
      (A.1 - B.1)^2 + (A.2 - B.2)^2 = (C.1 - A.1)^2 + (C.2 - A.2)^2
      ) :=
by sorry

end dirichlet_props_l169_169489


namespace goshawk_eurasian_reserve_hawks_l169_169751

variable (H P : ℝ)

theorem goshawk_eurasian_reserve_hawks :
  P = 100 ∧
  (35 / 100) * P = P - (H + (40 / 100) * (P - H) + (25 / 100) * (40 / 100) * (P - H))
    → H = 25 :=
by sorry

end goshawk_eurasian_reserve_hawks_l169_169751


namespace students_not_enrolled_in_any_l169_169395

open Finset

variables (U : Finset ℕ) -- Universe of students

variables (F G S : Finset ℕ) -- Sets of students taking French, German, and Spanish
variables (h_card_U : U.card = 150)
variables (h_card_F : F.card = 60)
variables (h_card_G : G.card = 50)
variables (h_card_S : S.card = 40)
variables (h_card_FG : (F ∩ G).card = 20)
variables (h_card_FS : (F ∩ S).card = 15)
variables (h_card_GS : (G ∩ S).card = 10)
variables (h_card_FGS : (F ∩ G ∩ S).card = 5)

theorem students_not_enrolled_in_any :
  (U.card - (F ∪ G ∪ S).card) = 40 :=
by
  sorry

end students_not_enrolled_in_any_l169_169395


namespace number_of_perfect_cubes_in_range_l169_169374

/-
Problem: Prove that the number of integers n where 100 ≤ n^3 ≤ 1000 is equal to 6.
-/

theorem number_of_perfect_cubes_in_range : 
  ∃ (count : ℕ), count = 6 ∧ (count = (finset.card (finset.filter (λ n : ℕ, 100 ≤ n^3 ∧ n^3 ≤ 1000) (finset.range 11)))) :=
  by {
   sorry
  }

end number_of_perfect_cubes_in_range_l169_169374


namespace distance_between_A_and_B_l169_169769

theorem distance_between_A_and_B :
  let A := (2, -3, 3)
  let B := (2, 1, 0)
  dist (A : ℝ×ℝ×ℝ) B = 5 :=
by {
  sorry
}

end distance_between_A_and_B_l169_169769


namespace powers_of_2_but_not_16_count_l169_169732

theorem powers_of_2_but_not_16_count :
  let upper_bound := 1000000000
  let powers_of_2 := { n | ∃ k, n = 2^k ∧ n < upper_bound }
  let powers_of_16 := { n | ∃ m, n = 16^m ∧ n < upper_bound }
  let powers_of_2_not_16 := powers_of_2 \ powers_of_16
  in powers_of_2_not_16.card = 23 :=
by
  sorry

end powers_of_2_but_not_16_count_l169_169732


namespace find_m_range_l169_169793

theorem find_m_range (m : ℝ) (x : ℝ) (h : ∃ c d : ℝ, (c ≠ 0) ∧ (∀ x, (c * x + d)^2 = x^2 + (12 / 5) * x + (2 * m / 5))) : 3.5 ≤ m ∧ m ≤ 3.7 :=
by
  sorry

end find_m_range_l169_169793


namespace axis_of_symmetry_exists_l169_169488

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + (Real.pi / 3))

theorem axis_of_symmetry_exists :
  ∃ k : ℤ, ∃ x : ℝ, (x = -5 * Real.pi / 12 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi))
  ∨ (x = Real.pi / 12 + k * Real.pi / 2 ∧ f x = Real.sin (Real.pi / 2 + k * Real.pi)) :=
sorry

end axis_of_symmetry_exists_l169_169488


namespace product_probability_multiple_of_4_l169_169776

-- Define the dice rolls and their properties
def is_multiple_of_4 (n : ℕ) : Prop := n % 4 = 0

-- Fair die rolls ranging for Juan and Amal
def juan_rolls := {n // 1 ≤ n ∧ n ≤ 12}
def amal_rolls := {n // 1 ≤ n ∧ n ≤ 10}

-- Ranges for fair dice representing all possible outcomes
def juan_roll_range := finset.filter is_multiple_of_4 (finset.range 13)
def amal_roll_range := finset.filter is_multiple_of_4 (finset.range 11)

-- Calculate probability of event occurring
def probability (s : finset ℕ) (total : ℕ) : ℚ := s.card / total

-- Conditions
def prob_juan_multiple_of_4 := probability juan_roll_range 12
def prob_amal_multiple_of_4 := probability amal_roll_range 10
def prob_neither_multiple_of_4 := (1 - prob_juan_multiple_of_4) * (1 - prob_amal_multiple_of_4)
def prob_at_least_one_multiple_of_4 := 1 - prob_neither_multiple_of_4

-- Statement to prove
theorem product_probability_multiple_of_4 :
  prob_at_least_one_multiple_of_4 = 2 / 5 := sorry

end product_probability_multiple_of_4_l169_169776


namespace arithmetic_seq_sum_l169_169739

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h_arith : ∀ n, a (n + 1) = a n + (a 2 - a 1)) 
(h_given : a 2 + a 8 = 10) : 
a 3 + a 7 = 10 :=
sorry

end arithmetic_seq_sum_l169_169739


namespace average_weight_increase_l169_169066

theorem average_weight_increase (A : ℝ) :
  let initial_total_weight := 10 * A
  let new_total_weight := initial_total_weight - 65 + 97
  let new_average := new_total_weight / 10
  let increase := new_average - A
  increase = 3.2 :=
by
  sorry

end average_weight_increase_l169_169066


namespace volleyball_quality_test_l169_169463

theorem volleyball_quality_test : 
  let quality_data := [275, 263, 278, 270, 261, 277, 282, 269] in
  let acceptable_quality_range (x : ℕ) := 260 ≤ x ∧ x ≤ 280 in
  (quality_data.filter (λ x, ¬(acceptable_quality_range x))).length = 1 :=
by
  let quality_data := [275, 263, 278, 270, 261, 277, 282, 269]
  let acceptable_quality_range (x : ℕ) := 260 ≤ x ∧ x ≤ 280
  show (quality_data.filter (λ x, ¬(acceptable_quality_range x))).length = 1
  sorry

end volleyball_quality_test_l169_169463


namespace interior_angle_heptagon_l169_169208

theorem interior_angle_heptagon : 
  ∀ (n : ℕ), n = 7 → (5 * 180 / n : ℝ) = 128.57142857142858 :=
by
  intros n hn
  rw hn
  -- The proof is skipped
  sorry

end interior_angle_heptagon_l169_169208


namespace parallel_lines_l169_169273

variables {α : Type*} [EuclideanGeometry α]

-- Define the points and conditions
variables (A B C D E P Q R S P' Q' R' S' : α)

-- Assuming the conditions
variable (h1 : ∃ (A B C D E : α), ∀ (P Q R S : α), ∃ (P' Q' R' S' : α),
  ∃ (a b c d e f g h : ℝ),
  (P ∈ line_segment A B) ∧ (Q ∈ line_segment B C) ∧ (R ∈ line_segment C D) ∧ (S ∈ line_segment D A) ∧
  (P' ∈ line_segment C D) ∧ (Q' ∈ line_segment D A) ∧ (R' ∈ line_segment A B) ∧ (S' ∈ line_segment B C) ∧
  (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ∧ A ≠ C ∧ B ≠ D) ∧
  (IsPerpendicular (line_through A B) (line_through E P)) ∧
  (IsPerpendicular (line_through B C) (line_through E Q)) ∧
  (IsPerpendicular (line_through C D) (line_through E R)) ∧
  (IsPerpendicular (line_through D A) (line_through E S)) ∧
  (IsIntersection (line_through E P) (line_through C D) = P') ∧
  (IsIntersection (line_through E Q) (line_through D A) = Q') ∧
  (IsIntersection (line_through E R) (line_through A B) = R') ∧
  (IsIntersection (line_through E S) (line_through B C) = S'))

-- Prove the statements
theorem parallel_lines (h : h1):
  Parallel (line_through R' S') (line_through Q' P') ∧
  Parallel (line_through Q' P') (line_through A C) ∧
  Parallel (line_through R' Q') (line_through S' P') ∧
  Parallel (line_through S' P') (line_through B D) :=
sorry

end parallel_lines_l169_169273


namespace find_integer_pairs_l169_169315

-- This definition corresponds to the condition of m and n being positive integers
def are_positive_integers (m n : ℕ) : Prop :=
  m > 0 ∧ n > 0

-- This definition corresponds to the existence of the polynomials P and Q with the required properties
def exist_poly_diff (m n : ℕ) : Prop :=
  ∃ P Q : ℝ[X], P.monic ∧ Q.monic ∧ P.degree = m ∧ Q.degree = n ∧ ∀ t : ℝ, P.eval (Q.eval t) ≠ Q.eval (P.eval t)

-- This theorem statement combines the conditions and the proof goal
theorem find_integer_pairs : ∀ m n : ℕ, are_positive_integers m n → exist_poly_diff m n :=
by
  -- Here we just state the theorem, the proof is omitted
  intros m n hmn
  sorry

end find_integer_pairs_l169_169315


namespace parallel_line_to_two_planes_l169_169555

theorem parallel_line_to_two_planes
    {l : Line} {p₁ p₂ : Plane}
    (h₁ : IsParallel l p₁)
    (h₂ : IsParallel l p₂)
    (h_inter : Intersect p₁ p₂) :
    IsParallel l (IntersectionLine p₁ p₂) := 
sorry

end parallel_line_to_two_planes_l169_169555


namespace weight_of_piece_l169_169864

theorem weight_of_piece (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := 
by
  sorry

end weight_of_piece_l169_169864


namespace geometric_sequence_150th_term_l169_169418

noncomputable def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ := a * r ^ (n - 1)

theorem geometric_sequence_150th_term :
  geometric_sequence 8 (-1 / 2) 150 = -8 * (1 / 2) ^ 149 :=
by
  -- This is the proof placeholder
  sorry

end geometric_sequence_150th_term_l169_169418


namespace relationship_between_f_x1_and_f_x2_l169_169820

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

-- Conditions:
variable (h_even : ∀ x, f x = f (-x))          -- f is even
variable (h_increasing : ∀ a b, 0 < a → a < b → f a < f b)  -- f is increasing on (0, +∞)
variable (h_x1_neg : x1 < 0)                   -- x1 < 0
variable (h_x2_pos : 0 < x2)                   -- x2 > 0
variable (h_abs : |x1| > |x2|)                 -- |x1| > |x2|

-- Goal:
theorem relationship_between_f_x1_and_f_x2 : f x1 > f x2 :=
by
  sorry

end relationship_between_f_x1_and_f_x2_l169_169820


namespace count_special_divisors_of_2028_l169_169639

theorem count_special_divisors_of_2028 : 
  let d := 202^8
  ∃ n : ℕ, n = 30 ∧ ∀ x : ℕ, (x | d) → (∃ a b : ℕ, x = 2^a * 101^b ∧ 0 ≤ a ∧ a ≤ 8 ∧ 0 ≤ b ∧ b ≤ 8) → 
          (∃ c d : ℕ, x = 2^c * 101^d ∧ ((c % 2 = 0 ∧ d % 2 = 0) ∨ (c % 3 = 0 ∧ d % 3 = 0) ∨ (c % 6 = 0 ∧ d % 6 = 0))) :=
by 
  sorry

end count_special_divisors_of_2028_l169_169639


namespace negation_of_universal_proposition_l169_169159

theorem negation_of_universal_proposition :
  ¬(∀ x : ℝ, x^2 + 2*x + 2 > 0) ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_universal_proposition_l169_169159


namespace equal_tuesdays_and_thursdays_in_31_days_l169_169246

noncomputable def num_start_days_with_equal_tuesdays_thursdays : ℕ :=
  3

theorem equal_tuesdays_and_thursdays_in_31_days :
  ∃ days : ℕ, days = num_start_days_with_equal_tuesdays_thursdays :=
begin
  existsi num_start_days_with_equal_tuesdays_thursdays,
  refl,
end

end equal_tuesdays_and_thursdays_in_31_days_l169_169246


namespace max_d_n_l169_169907

def sequence_a (n : ℕ) : ℤ := 100 + n^2

def d_n (n : ℕ) : ℤ := Int.gcd (sequence_a n) (sequence_a (n + 1))

theorem max_d_n : ∃ n, d_n n = 401 :=
by
  -- Placeholder for the actual proof
  sorry

end max_d_n_l169_169907


namespace problem1_problem2_l169_169231

-- Problem 1
theorem problem1 (x : ℝ) : x * (x - 1) - 3 * (x - 1) = 0 → (x = 1) ∨ (x = 3) :=
by sorry

-- Problem 2
theorem problem2 (x : ℝ) : x^2 + 2*x - 1 = 0 → (x = -1 + Real.sqrt 2) ∨ (x = -1 - Real.sqrt 2) :=
by sorry

end problem1_problem2_l169_169231


namespace value_of_Z_4_3_l169_169288

def Z (a b : ℤ) : ℤ := a^3 - 3 * a^2 * b + 3 * a * b^2 - b^3

theorem value_of_Z_4_3 : Z 4 3 = 1 := by
  sorry

end value_of_Z_4_3_l169_169288


namespace general_term_formula_sn_le_60n_plus_800_when_an_eq_2_min_n_for_sn_gt_60n_plus_800_when_an_eq_4n_minus_2_l169_169000

-- Define the arithmetic sequence
def arithmetic_seq (n : ℕ) (d : ℤ) (a1 : ℤ) : ℤ := a1 + (n - 1) * d

-- Conditions
def a1 : ℤ := 2
def a2 : ℤ := arithmetic_seq 2 d a1
def a5 : ℤ := arithmetic_seq 5 d a1

-- Question 1: Find the general term formula
theorem general_term_formula : (d = 0 ∨ d = 4) → (∀ n : ℕ, n > 0 → (arithmetic_seq n d a1 = 2 ∨ arithmetic_seq n d a1 = 4 * n - 2)) :=
sorry

-- Sum of first n terms of arithmetic sequence
def sum_seq (n : ℕ) (a_seq : ℕ → ℤ) : ℤ := sum (range n).map a_seq

-- Question 2(a): When a_n = 2, prove S_n <= 60n + 800 for all n > 0
theorem sn_le_60n_plus_800_when_an_eq_2 : (∀ n > 0, sum_seq n (λ (n : ℕ), 2) <= 60 * n + 800) :=
sorry

-- Question 2(b): When a_n = 4n - 2, prove the minimum n such that S_n > 60n + 800 is 41
theorem min_n_for_sn_gt_60n_plus_800_when_an_eq_4n_minus_2 :
  (∀ n : ℕ, n > 0 → (sum_seq n (λ (n : ℕ), 4 * n - 2) > 60 * n + 800 ↔ n ≥ 41)) :=
sorry

end general_term_formula_sn_le_60n_plus_800_when_an_eq_2_min_n_for_sn_gt_60n_plus_800_when_an_eq_4n_minus_2_l169_169000


namespace coincidence_probability_l169_169923

theorem coincidence_probability (m n : ℕ) (h : m > n) :
  let total_prob := (2 : ℝ) * n / (m + n)
  in total_prob = (2 : ℝ) * n / (m + n) :=
by
  sorry

end coincidence_probability_l169_169923


namespace vector_diff_magnitude_l169_169799

variables {ℝ_vector: Type*} [inner_product_space ℝ ℝ_vector]

open_locale real_inner_product_space

def is_unit_vector (v : ℝ_vector) : Prop :=
  ∥v∥ = 1

theorem vector_diff_magnitude (a b : ℝ_vector) (h_a_unit: is_unit_vector a) (h_b_unit: is_unit_vector b) (h_sum_unit: ∥a + b∥ = 1) :
  ∥a - b∥ = real.sqrt 3 :=
begin
  sorry
end

end vector_diff_magnitude_l169_169799


namespace maximize_daily_profit_l169_169586

variable (x : ℝ) -- price reduction in yuan per pot
variable (y : ℝ) -- daily profit in yuan

def daily_profit (x : ℝ) : ℝ := (40 - x) * (20 + 2 * x)

theorem maximize_daily_profit : (0 ≤ x ∧ x < 40) → (∃ x_max : ℝ, x_max = 15 ∧ daily_profit x_max = 1250) :=
by
  intro h
  use 15
  split
  · refl
  · rw [daily_profit, ← sub_eq_add_neg, sub_left_eq_sub_right, mul_add, mul_comm]
    simp
    sorry

end maximize_daily_profit_l169_169586


namespace good_arrangement_iff_coprime_l169_169527

-- Definitions for the concepts used
def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_good_arrangement (n m : ℕ) : Prop :=
  ∃ k₀, ∀ i, (n * k₀ * i) % (m + n) = (i % (m + n))

theorem good_arrangement_iff_coprime (n m : ℕ) : is_good_arrangement n m ↔ is_coprime n m := 
sorry

end good_arrangement_iff_coprime_l169_169527


namespace negative_coefficient_in_expansion_l169_169855

theorem negative_coefficient_in_expansion : 
  ∃ (k : ℤ), k < 0 ∧ k ∈ (coefficients (polynomial.expand (Polynomial.of_coefficients [(2:ℤ), -1, 1] 2014)))
:= sorry

end negative_coefficient_in_expansion_l169_169855


namespace quadratic_solution_l169_169170

theorem quadratic_solution :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := sorry

end quadratic_solution_l169_169170


namespace find_243xyz_divisible_by_396_l169_169323

/--
Find the values of x, y, and z such that the number 243xyz is divisible by 396.
-/
theorem find_243xyz_divisible_by_396 :
  ∃ (x y z : ℕ),
    (y * 10 + z) % 4 = 0 ∧
    (2 + 4 + 3 + x + y + z) % 9 = 0 ∧
    ((2 + 3 + y) - (4 + x + z)) % 11 = 0 ∧
    (243 * 1000 + x * 100 + y * 10 + z) % 396 = 0 :=
begin
  use [5, 4, 0], -- one possible solution (243540)
  split,
  { exact dec_trivial }, -- y*10+z=54, 54 mod 4 = 0
  split,
  { exact dec_trivial }, -- 2+4+3+x+y+z=18, 18 mod 9 = 0
  split,
  { exact dec_trivial }, -- (2+3+y) - (4+x+z)=5-5, 0 mod 11 = 0
  { exact dec_trivial }  -- 243540 mod 396 = 0
end

end find_243xyz_divisible_by_396_l169_169323


namespace reduced_price_per_kg_l169_169597

-- Assume the constants in the conditions
variables (P R : ℝ)
variables (h1 : R = P - 0.40 * P) -- R = 0.60P
variables (h2 : 2000 / P + 10 = 2000 / R) -- extra 10 kg for the same 2000 rs

-- State the target we want to prove
theorem reduced_price_per_kg : R = 80 :=
by
  -- The steps and details of the proof
  sorry

end reduced_price_per_kg_l169_169597


namespace max_difference_in_masses_of_two_flour_bags_l169_169124

theorem max_difference_in_masses_of_two_flour_bags :
  ∀ (x y : ℝ), (24.8 ≤ x ∧ x ≤ 25.2) → (24.8 ≤ y ∧ y ≤ 25.2) → |x - y| ≤ 0.4 :=
by
  sorry

end max_difference_in_masses_of_two_flour_bags_l169_169124


namespace find_b_c_d_l169_169078

noncomputable def a_n_sequence : ℕ → ℕ
| n := -- Here define the function to generate the sequence {1, 3, 3, 3, 5, 5, 5, 5, 5, ...}
by sorry

lemma floor_sqrt_eq_k (n c : ℕ) (k : ℕ) : 
    (k ∈ (ℕ \ {0}) → floor (sqrt (Int.ofNat (n + c))) = k ) :=
begin
  -- The proof that the floor function on sqrt term leads to integer k
  sorry
end

theorem find_b_c_d :
  ∃ b c d, (∀ n : ℕ, a_n_sequence n = b * int.floor (real.sqrt (n + c)) + d) ∧ b + c + d = 2 :=
begin
  use [2, -1, 1],
  split,
  { intro n,
    have a : int.floor (real.sqrt (n - 1)) = what_is_the_expected_output  -- Define expected output per n
    sorry,
    },
  { simp, 
    linarith }
end

end find_b_c_d_l169_169078


namespace find_gallons_of_15_percent_solution_l169_169045

-- Definitions for the problem conditions
def alc_total (x y : ℝ) : ℝ := 0.15 * x + 0.35 * y
def total_volume (x y : ℝ) : ℝ := x + y

-- Given these conditions
axiom h1 : total_volume x y = 250
axiom h2 : alc_total x y = 52.5

-- We need to prove that x = 175
theorem find_gallons_of_15_percent_solution (x y : ℝ) (h1 : total_volume x y = 250) (h2 : alc_total x y = 52.5) : x = 175 := sorry

end find_gallons_of_15_percent_solution_l169_169045


namespace seth_has_winning_strategy_l169_169867

-- Defining the set of pairs (k, 2k) for k in the range 1 to 25
def valid_pairs : set (ℕ × ℕ) :=
  { (k, 2 * k) | k ∈ finset.range (25 + 1) }

-- Define a function that represents whether Seth has a winning strategy
-- under the initial conditions of the game
def seth_wins : Prop :=
  ∃ strategy : list (set (ℕ × ℕ)), 
  (∀ turn : ℕ, turn % 2 = 1 → strategy.nth turn ∈ valid_pairs) ∧
  (∀ used_pairs : set (ℕ × ℕ), ∀ round : ℕ, 
    (∀ step < round, 
      step % 2 = 1 → (strategy.nth step) ∉ used_pairs ∧
      ∃ step_pair ∈ valid_pairs, step_pair ∉ used_pairs) →
    (strategy.nth round) ∉ used_pairs ∧
    ∃ next_pair ∈ valid_pairs, next_pair ∉ used_pairs)

theorem seth_has_winning_strategy :
  seth_wins :=
sorry

end seth_has_winning_strategy_l169_169867


namespace circumradii_equal_l169_169468

theorem circumradii_equal 
  (A B C D M N : Point)
  (h1 : D is the circumcenter of triangle(A, B, C))
  (h2 : Circle_through (A, B, D) intersects sides (A, C) and (B, C) at points M and N) :
  circumradius (triangle(A, B, D)) = circumradius (triangle(M, N, C)) := 
sorry

end circumradii_equal_l169_169468


namespace min_value_expr_l169_169823

theorem min_value_expr (x y z : ℝ) (h : x > 0 ∧ y > 0 ∧ z > 0) (h_xyz : x * y * z = 1) : 
  x^2 + 4 * x * y + 9 * y^2 + 8 * y * z + 3 * z^2 ≥ 9^(10/9) :=
sorry

end min_value_expr_l169_169823


namespace clock_angle_proof_l169_169729

-- Define the given conditions
def hour_deg := 30  -- degrees per hour mark
def min_deg_per_min := 6  -- degrees per minute for minute hand
def hour_deg_per_min := 0.5  -- degrees per minute for hour hand

-- Given time
def hour := 2
def minutes := 20

-- Calculate the position of the minute hand
def minute_hand_angle := min_deg_per_min * minutes

-- Calculate the position of the hour hand
def hour_hand_angle := hour * hour_deg + hour_deg_per_min * minutes

-- Calculate the smaller angle between the two hands
def smaller_angle := abs (minute_hand_angle - hour_hand_angle)

-- The theorem to prove
theorem clock_angle_proof : smaller_angle = 50 :=
by
  sorry

end clock_angle_proof_l169_169729


namespace slippers_total_cost_l169_169842

theorem slippers_total_cost (original_price discount_rate embroidery_cost_per_shoe shipping_cost : ℝ)
                            (num_shoes : ℕ)
                            (h_original_price : original_price = 50.00)
                            (h_discount_rate : discount_rate = 0.10)
                            (h_embroidery_cost_per_shoe : embroidery_cost_per_shoe = 5.50)
                            (h_shipping_cost : shipping_cost = 10.00)
                            (h_num_shoes : num_shoes = 2) :
  let discounted_price := original_price * (1 - discount_rate)
      total_embroidery_cost := embroidery_cost_per_shoe * num_shoes
      total_cost := discounted_price + total_embroidery_cost + shipping_cost
  in total_cost = 66.00 := by
  sorry

end slippers_total_cost_l169_169842


namespace hexagon_sum_distances_l169_169853

/-- Point M lies on the side of a regular hexagon with side length 10. 
    Prove that the sum of the distances from point M to the lines containing 
    the other five sides of the hexagon is 30√3. -/
theorem hexagon_sum_distances (a : ℝ) (h_a : a = 10) (M : ℝ → ℝ) :
  (∑ i in (Finset.range 5), distance M (line_containing_side_i a)) = 30 * (sqrt 3) := 
sorry

end hexagon_sum_distances_l169_169853


namespace jenny_improvements_value_l169_169429

-- Definitions based on the conditions provided
def property_tax_rate : ℝ := 0.02
def initial_house_value : ℝ := 400000
def rail_project_increase : ℝ := 0.25
def affordable_property_tax : ℝ := 15000

-- Statement of the theorem
theorem jenny_improvements_value :
  let new_house_value := initial_house_value * (1 + rail_project_increase)
  let max_affordable_house_value := affordable_property_tax / property_tax_rate
  let value_of_improvements := max_affordable_house_value - new_house_value
  value_of_improvements = 250000 := 
by
  sorry

end jenny_improvements_value_l169_169429


namespace four_at_six_l169_169053

-- Definition of the operation
def at (a b : ℕ) : ℕ := 4 * a - 2 * b + a * b

-- Theorem we want to prove
theorem four_at_six : at 4 6 = 28 := by
  -- Proof goes here
  sorry

end four_at_six_l169_169053


namespace union_of_sets_l169_169094

open Set

theorem union_of_sets :
  let A := {1, 2}
  let B := {2, 3}
  A ∪ B = {1, 2, 3} :=
by
  let A := {1, 2}
  let B := {2, 3}
  show A ∪ B = {1, 2, 3}
  sorry

end union_of_sets_l169_169094


namespace first_term_geometric_sequence_l169_169641

theorem first_term_geometric_sequence (a r : ℕ) (h₁ : a * r^5 = 32) (h₂ : r = 2) : a = 1 := by
  sorry

end first_term_geometric_sequence_l169_169641


namespace find_larger_triangle_area_l169_169095

-- Assume the necessary geometric setup and definitions
noncomputable def area_of_triangle_GHI : ℝ :=
100 -- Given area of △GHI

-- Define the problem context and the main statement
theorem find_larger_triangle_area (hexagon : Type)
  (A B C D E F G H I : hexagon)
  (is_regular_hexagon : ∀ (X Y Z : hexagon), (is_side A B ∧ is_side B C ∧ is_side C D ∧ is_side D E ∧ is_side E F ∧ is_side F A) ∧
    (is_midpoint G A B ∧ is_midpoint H C D ∧ is_midpoint I E F))
  (J K L : hexagon)
  (vertices_midpoints : J = midpoint B C ∧ K = midpoint D E ∧ L = midpoint F A) :
  area_of_triangleₓ J K L = 300 :=
sorry

end find_larger_triangle_area_l169_169095


namespace system_solutions_l169_169292

noncomputable def f (t : ℝ) : ℝ := 4 * t^2 / (1 + 4 * t^2)

theorem system_solutions (x y z : ℝ) :
  (f x = y ∧ f y = z ∧ f z = x) ↔ (x = 0 ∧ y = 0 ∧ z = 0) ∨ (x = 1/2 ∧ y = 1/2 ∧ z = 1/2) :=
by
  sorry

end system_solutions_l169_169292


namespace water_drain_rate_l169_169154

theorem water_drain_rate
  (total_volume : ℕ)
  (total_time : ℕ)
  (H1 : total_volume = 300)
  (H2 : total_time = 25) :
  total_volume / total_time = 12 := 
by
  sorry

end water_drain_rate_l169_169154


namespace evaluate_expression_l169_169306

theorem evaluate_expression : (24^36 / 72^18) = 8^18 := by
  sorry

end evaluate_expression_l169_169306


namespace South_Eastbay_population_increase_l169_169411

theorem South_Eastbay_population_increase : 
  (let births_per_day := 24 / 6
   let deaths_per_day := 1 / 2
   let net_increase_per_day := births_per_day - deaths_per_day
   let net_increase_per_year := net_increase_per_day * 365
   let rounded_increase := round (net_increase_per_year / 100) * 100
   in rounded_increase) = 1300 := 
by
  -- Definitions from the conditions
  let births_per_day := 24 / 6
  let deaths_per_day := 1 / 2
  let net_increase_per_day := births_per_day - deaths_per_day
  let net_increase_per_year := net_increase_per_day * 365
  let rounded_increase := round (net_increase_per_year / 100) * 100
  -- The statement to prove
  have : rounded_increase = 1300 := sorry
  exact sorry

end South_Eastbay_population_increase_l169_169411


namespace socks_count_l169_169614

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end socks_count_l169_169614


namespace distance_A_C_l169_169768

noncomputable def circle_center := (0 : ℝ, 2 : ℝ)

noncomputable def point_A := (2 * real.sqrt 3, 2 : ℝ)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

theorem distance_A_C : distance circle_center point_A = 2 * real.sqrt 3 := 
by 
  sorry

end distance_A_C_l169_169768


namespace fred_initial_balloons_l169_169332

theorem fred_initial_balloons (balloons_given : ℕ) (balloons_left : ℕ) (h1 : balloons_given = 221) (h2 : balloons_left = 488) : 
  balloons_given + balloons_left = 709 :=
by
  rw [h1, h2]
  show 221 + 488 = 709
  sorry

end fred_initial_balloons_l169_169332


namespace abs_diff_squares_105_95_l169_169947

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l169_169947


namespace greatest_sum_consecutive_integers_lt_500_l169_169988

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169988


namespace units_digit_product_l169_169213

theorem units_digit_product :
  let sum_squares_units := (list.range (2 * 2011 - 1)).filter (λ x, x % 2 = 1).sum (λ x, (x^2 % 10)) % 10,
      sum_units := (list.range (2 * 4005 - 1)).filter (λ x, x % 2 = 1).sum % 10
  in (sum_squares_units * sum_units) % 10 = 0 := 
by {
  let n := 2011,
  let m := 4005,
  let sum_squares := (list.range (2 * n - 1)).filter (λ x, x % 2 = 1).map (λ x, x ^ 2),
  let sum_odd := (list.range (2 * m - 1)).filter (λ x, x % 2 = 1),
  have h_sq : sum_squares_units = sum_squares.sum % 10 := rfl,
  have h_sum : sum_units = sum_odd.sum % 10 := rfl,
  rw [h_sq, h_sum],
  sorry
}

end units_digit_product_l169_169213


namespace total_capacity_sum_is_224_l169_169638

-- Define the capacity function
def capacity (S : Set ℕ) : ℕ :=
  S.toFinset.sum id

-- Define the condition for the subsets
def satisfies_condition (A : Set ℕ) : Prop :=
  ∀ a ∈ A, (8 - a) ∈ A

-- Define the universe of elements
def universe := {1, 2, 3, 4, 5, 6, 7}

-- Define the total capacity sum function
noncomputable def total_sum_of_capacities : ℕ :=
  (universe.powerset.filter (λ A => satisfies_condition A ∧ A.nonempty)).sum capacity

-- State the theorem
theorem total_capacity_sum_is_224 : total_sum_of_capacities = 224 :=
by
  sorry

end total_capacity_sum_is_224_l169_169638


namespace find_center_of_tangent_circle_l169_169238

theorem find_center_of_tangent_circle :
  ∃ (a b : ℝ), (abs a = 5) ∧ (abs b = 5) ∧ (4 * a - 3 * b + 10 = 25) ∧ (a = -5) ∧ (b = 5) :=
by {
  -- Here we would provide the proof in Lean, but for now, we state the theorem
  -- and leave the proof as an exercise.
  sorry
}

end find_center_of_tangent_circle_l169_169238


namespace probability_two_girls_l169_169600

-- Definitions based on the conditions
def total_members : ℕ := 15
def boys : ℕ := 7
def girls : ℕ := 8

-- Proof statement of the main problem
theorem probability_two_girls (h_total : total_members = 15) (h_boys : boys = 7) (h_girls : girls = 8) :
  (gir_count.choose 2 / total_members.choose 2) = 4 / 15   :=
begin
  sorry  -- Omitted proof
end

end probability_two_girls_l169_169600


namespace area_of_triangle_PSL_l169_169077

theorem area_of_triangle_PSL {P Q R S L' L T U : Type} 
  (area_PQR : ℝ) (PT_eq_TS : Prop) (QS_eq_2RS : Prop)
  (intersect_PT_QU_T : T ∈ (PS ∩ QU))
  (area_PSL : ℝ) :
  (area_PQR = 150) → (PT_eq_TS) → (QS_eq_2RS) → (area_PSL = 20) :=
begin
  sorry
end

end area_of_triangle_PSL_l169_169077


namespace nested_radical_eq_6_l169_169658

theorem nested_radical_eq_6 (x : ℝ) (h : x = Real.sqrt (18 + x)) : x = 6 :=
by 
  have h_eq : x^2 = 18 + x,
  { rw h, exact pow_two (Real.sqrt (18 + x)) },
  have quad_eq : x^2 - x - 18 = 0,
  { linarith [h_eq] },
  have factored : (x - 6) * (x + 3) = x^2 - x - 18,
  { ring },
  rw [←quad_eq, factored] at h,
  sorry

end nested_radical_eq_6_l169_169658


namespace greatest_sum_of_consecutive_integers_l169_169964

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169964


namespace mutually_exclusive_not_opposite_l169_169644

universe u

-- Define the colors and people involved
inductive Color
| black
| red
| white

inductive Person 
| A
| B
| C

-- Define a function that distributes the cards amongst the people
def distributes (cards : List Color) (people : List Person) : People -> Color :=
  sorry

-- Define events as propositions
def A_gets_red (d : Person -> Color) : Prop :=
  d Person.A = Color.red

def B_gets_red (d : Person -> Color) : Prop :=
  d Person.B = Color.red

-- The main theorem stating the problem
theorem mutually_exclusive_not_opposite 
  (d : Person -> Color)
  (h : A_gets_red d → ¬ B_gets_red d) : 
  ¬ ( ∀ (p : Prop), A_gets_red d ↔ p ) → B_gets_red d :=
sorry

end mutually_exclusive_not_opposite_l169_169644


namespace function_property_l169_169152

noncomputable def f (x : ℝ) : ℝ := sorry
variable (a x1 x2 : ℝ)

-- Conditions
axiom f_defined_on_R : ∀ x : ℝ, f x ≠ 0
axiom f_increasing_on_left_of_a : ∀ x y : ℝ, x < y → y < a → f x < f y
axiom f_even_shifted_by_a : ∀ x : ℝ, f (x + a) = f (-(x + a))
axiom ordering : x1 < a ∧ a < x2
axiom distance_comp : |x1 - a| < |x2 - a|

-- Proof Goal
theorem function_property : f (2 * a - x1) > f (2 * a - x2) :=
by
  sorry

end function_property_l169_169152


namespace eq_root_count_l169_169322

theorem eq_root_count (p : ℝ) : 
  (∀ x : ℝ, (2 * x^2 - 3 * p * x + 2 * p = 0 → (9 * p^2 - 16 * p = 0))) →
  (∃! p1 p2 : ℝ, (9 * p1^2 - 16 * p1 = 0) ∧ (9 * p2^2 - 16 * p2 = 0) ∧ p1 ≠ p2) :=
sorry

end eq_root_count_l169_169322


namespace correct_statement_c_l169_169217

theorem correct_statement_c : (Real.cbrt(0) = 0) :=
by
  sorry

end correct_statement_c_l169_169217


namespace find_n_l169_169153

noncomputable def a_n (n : ℕ) : ℝ := 1 / (Real.sqrt n + Real.sqrt (n + 1))
noncomputable def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, a_n i

theorem find_n :
  S_n 99 = 9 →
  (∀ n, S_n n = ∑ i in Finset.range n, Real.sqrt (i + 1) - Real.sqrt i) →
  ∑ i in Finset.range 99, (Real.sqrt (i + 1) - Real.sqrt i) - 1 = 9 ↔ 99 = 99 :=
by { intros, sorry }

end find_n_l169_169153


namespace part_a_part_b_l169_169221

open Nat

-- Define the sum of digits function
noncomputable def S (n : Nat) : Nat :=
  if n = 0 then 0 else n.digits 10 |>.sum

-- Part (a) statement
theorem part_a (K : Nat) : S(K) ≤ 8 * S(8 * K) :=
  sorry

-- If k is of the form 2^r * 5^q
def is_special_form (k : Nat) : Prop :=
  ∃ r q : Nat, k = (2^r) * (5^q)

-- Find the largest suitable value of ck
theorem part_b (k : Nat) (hk : is_special_form k) : ∃ c_k : ℝ, (∀ N : Nat, (S(k * N) / S(N) : ℝ) ≥ c_k) ∧ c_k = (1 : ℝ) / S(2 * 5) :=
  sorry

end part_a_part_b_l169_169221


namespace shortest_side_le_sqrt2_l169_169477

theorem shortest_side_le_sqrt2 (Q : Quadrilateral) (h : InscribedInCircle Q 1) :
  shortest_side Q ≤ Real.sqrt 2 := by
  sorry

end shortest_side_le_sqrt2_l169_169477


namespace probability_compare_l169_169166

-- Conditions
def v : ℝ := 0.1
def n : ℕ := 998

-- Binomial distribution formula
noncomputable def binom_prob (n k : ℕ) (v : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * (v ^ k) * ((1 - v) ^ (n - k))

-- Theorem to prove
theorem probability_compare :
  binom_prob n 99 v > binom_prob n 100 v :=
by
  sorry

end probability_compare_l169_169166


namespace tan_mul_eq_veca2_plus_dot_l169_169726

-- Definitions for the given problem
variables {α β : ℝ} (k : ℤ)
def vec_a : ℝ × ℝ := (Real.sin α, Real.sin β)
def vec_b : ℝ × ℝ := (Real.cos (α - β), -1)
def vec_c : ℝ × ℝ := (Real.cos (α + β), 2)

-- Conditions that α and β are not equal to kπ + π/2
def not_special_angle (α β : ℝ) : Prop := ∀ k : ℤ, α ≠ k * Real.pi + Real.pi / 2 ∧ β ≠ k * Real.pi + Real.pi / 2

-- 1. Prove that if vec_b is parallel to vec_c, then tan α * tan β = -3.
theorem tan_mul_eq (h1 : vec_b α β = (λ a b, (a, b)) (Real.cos (α + β)) (λ x, -2*x) vec_c α β ∧ not_special_angle α β) :
  Real.tan α * Real.tan β = -3 :=
sorry

-- 2. Prove that vec_a^2 + vec_b · vec_c = -1
theorem veca2_plus_dot (h2 : not_special_angle α β) :
  (Real.sin α)^2 + (Real.sin β)^2 + (Real.cos (α - β) * Real.cos (α + β) - 1) = -1 :=
sorry

end tan_mul_eq_veca2_plus_dot_l169_169726


namespace daysRequired_l169_169425

-- Defining the structure of the problem
structure WallConstruction where
  m1 : ℕ    -- Number of men in the first scenario
  d1 : ℕ    -- Number of days in the first scenario
  m2 : ℕ    -- Number of men in the second scenario

-- Given values
def wallConstructionProblem : WallConstruction :=
  WallConstruction.mk 20 5 30

-- The total work constant
def totalWork (wc : WallConstruction) : ℕ :=
  wc.m1 * wc.d1

-- Proving the number of days required for m2 men
theorem daysRequired (wc : WallConstruction) (k : ℕ) : 
  k = totalWork wc → (wc.m2 * (k / wc.m2 : ℚ) = k) → (k / wc.m2 : ℚ) = 3.3 :=
by
  intro h1 h2
  sorry

end daysRequired_l169_169425


namespace Marie_final_erasers_l169_169839

def Marie_initial_erasers : ℝ := 95.5
def Marie_bought_erasers : ℝ := 42.75
def discount_rate : ℝ := 0.15

theorem Marie_final_erasers :
  let discount := discount_rate * Marie_bought_erasers in
  let erasers_bought_after_discount := Marie_bought_erasers - discount in
  Marie_initial_erasers + erasers_bought_after_discount = 131.8375 := by
  sorry

end Marie_final_erasers_l169_169839


namespace initial_marbles_l169_169279

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

end initial_marbles_l169_169279


namespace one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169542

theorem one_hundred_fiftieth_digit_of_3_div_11_is_7 :
  let decimal_repetition := "27"
  let length := 2
  150 % length = 0 →
  (decimal_repetition[1] = '7')
: sorry

end one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169542


namespace two_std_dev_less_than_mean_l169_169561

def mean : ℝ := 14.0
def std_dev : ℝ := 1.5

theorem two_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.0 := 
by sorry

end two_std_dev_less_than_mean_l169_169561


namespace cost_difference_l169_169169

-- Definitions based on given conditions
def length_width_ratio := (7, 4)
def park_area := 4265.892
def cost_per_meter_wood := 75.90
def cost_per_meter_metal := 98.45

-- The mathematically equivalent proof problem
theorem cost_difference (length_width_ratio : ℝ × ℝ)
                        (park_area : ℝ) 
                        (cost_per_meter_wood : ℝ) 
                        (cost_per_meter_metal : ℝ) : 
  (length_width_ratio.fst / length_width_ratio.snd = 7 / 4) ∧
  (park_area = 4265.892) ∧
  (cost_per_meter_wood = 75.90) ∧
  (cost_per_meter_metal = 98.45) →
  ∃ cost_difference : ℝ, cost_difference = 6134.85 :=
begin
  sorry
end

end cost_difference_l169_169169


namespace simplify_expression_l169_169359

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (x ^ (-2) + y ^ (-2)) ^ (-1) = x ^ 2 * y ^ 2 / (x ^ 2 + y ^ 2) :=
sorry

end simplify_expression_l169_169359


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169982

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169982


namespace find_interest_rate_l169_169861

-- Define the conditions
def total_amount : ℝ := 2500
def second_part_rate : ℝ := 0.06
def annual_income : ℝ := 145
def first_part_amount : ℝ := 500.0000000000002
noncomputable def interest_rate (r : ℝ) : Prop :=
  first_part_amount * r + (total_amount - first_part_amount) * second_part_rate = annual_income

-- State the theorem
theorem find_interest_rate : interest_rate 0.05 :=
by
  sorry

end find_interest_rate_l169_169861


namespace locus_of_points_is_line_extension_l169_169669

-- Definitions for conditions.
variables {P Q R M U X U' S l : Type}
variables [HasMidpoint M Q R]  -- M is the midpoint of Q and R on line l.
variables [IsIncircle C (triangle P Q R)]  -- C is the incircle of triangle PQR.

-- Main statement: The locus of points P.
theorem locus_of_points_is_line_extension
  (h_midpoint : ∀ (Q R : l), M = midpoint Q R)
  (h_incircle : ∀ (P Q R : Type), IsIncircle C (triangle P Q R)) :
  locus_of_points P = extension_line U' X := 
sorry

end locus_of_points_is_line_extension_l169_169669


namespace rebecca_has_more_eggs_than_marbles_l169_169131

-- Given conditions
def eggs : Int := 20
def marbles : Int := 6

-- Mathematically equivalent statement to prove
theorem rebecca_has_more_eggs_than_marbles :
    eggs - marbles = 14 :=
by
    sorry

end rebecca_has_more_eggs_than_marbles_l169_169131


namespace point_in_first_quadrant_l169_169904

-- Define the complex number z
def z := (3 + complex.i) / (1 + complex.i) + 3 * complex.i

-- Define the condition which states that z = 2 + 2i (as computed)
def z_simplified : z = 2 + 2 * complex.i := sorry

-- Prove that z is in the first quadrant
theorem point_in_first_quadrant (h : z = 2 + 2 * complex.i) : 
  (complex.re z > 0) ∧ (complex.im z > 0) :=
by
  rw h
  exact ⟨by norm_num, by norm_num⟩

end point_in_first_quadrant_l169_169904


namespace Emily_total_cost_l169_169647

theorem Emily_total_cost :
  let cost_curtains := 2 * 30
  let cost_prints := 9 * 15
  let installation_cost := 50
  let total_cost := cost_curtains + cost_prints + installation_cost
  total_cost = 245 := by
{
 sorry
}

end Emily_total_cost_l169_169647


namespace regular_heptagon_interior_angle_l169_169196

theorem regular_heptagon_interior_angle :
  ∀ (n : ℕ), n = 7 → (∑ i in finset.range n, 180 / n) = 128.57 :=
  by
  intros n hn
  rw hn
  sorry

end regular_heptagon_interior_angle_l169_169196


namespace simplify_expression_l169_169137

theorem simplify_expression (x : ℝ) (h : x ≠ 0) : 
  (5 / (4 * x^(-4)) + (4 * x^3) / 5) = (x^3 * (25 * x + 16)) / 20 := 
by
  sorry

end simplify_expression_l169_169137


namespace general_term_a_sum_bn_l169_169402

variables {a_n : ℕ → ℤ} {b_n : ℕ → ℤ} (n : ℕ)

-- Define conditions for the arithmetic sequence {a_n}
def a_2 := (a_n 2) = 1
def a_5 := (a_n 5) = 4

-- Statement to prove the general term formula a_n = n - 1
theorem general_term_a (h1 : a_2) (h2 : a_5) : a_n n = n - 1 := sorry

-- Define the sequence {b_n} = 2^{a_n}
def b_seq := ∀ n, b_n n = 2^(a_n n)

-- Statement to prove the sum of the first n terms of the sequence {b_n}
theorem sum_bn (h1 : a_2) (h2 : a_5) (h3 : b_seq) : ∑ i in finset.range n, b_n i = 2^n - 1 := sorry

end general_term_a_sum_bn_l169_169402


namespace find_m_n_and_area_l169_169338

def A : ℝ × ℝ := (0, 2)
def B : ℝ × ℝ := (4, 0)
def C : ℝ × ℝ := (7, 3)
def D (m n : ℝ) : ℝ × ℝ := (m, n)

theorem find_m_n_and_area : 
  (∃ m n : ℝ, m = 3 / 5 ∧ n = 31 / 5 ∧ 
  let 
    AB := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
  let 
    CD := ((C.1 - m)^2 + (C.2 - n)^2) in
  let 
    d := abs ((7 + 2*3) * 1 / sqrt 5) in 
  let 
    S := 1 / 2 * ((sqrt AB) + (sqrt CD)) * d in
  S = 117 / 5) :=
sorry

end find_m_n_and_area_l169_169338


namespace quadratic_trinomial_complete_square_form_l169_169249

theorem quadratic_trinomial_complete_square_form (a b : ℝ) :
  (∃ (m n : ℝ), x^4 - 6 * x^3 + 7 * x^2 + a * x + b = (x^2 + m * x + n)^2) →
  (a = -2 * 3 ∧ b = (-3)^2 - 2 * (-1)) →
  (ƒorm = x^2 - 3 * x - 1 ∨ form = -x^2 + 3 * x + 1) :=
begin
  intros h1 h2,
  sorry
end

end quadratic_trinomial_complete_square_form_l169_169249


namespace nim_sequence_eventual_const_l169_169324

-- Define the game and function 
def nim_game (S : Set ℕ) (n : ℕ) : Prop :=
  ∃ f : ℕ → ℕ, ∀ n ∈ f S, (n > 0 ∧ ∀ n' ∈ S, n' ≤ n)

-- Function f(S) indicating Deric's winning strategy
def f (S : Set ℕ) : Set ℕ :=
  { n : ℕ | ∃ m ∈ S, nim_game S (n - m) }

-- Proof statement 
theorem nim_sequence_eventual_const (T : Set ℕ) : 
  ∃ N : ℕ, ∀ n m ≥ N, f^[n] T = f^[m] T :=
sorry

end nim_sequence_eventual_const_l169_169324


namespace find_vector_at_t_neg_2_l169_169393

-- Definitions for the given conditions
def point1 := (2, 0, -3)
def point2 := (7, -2, 1)
def point3 := (17, -6, 9)
def t1 := 1
def t2 := 2
def t3 := 4
def t4 := -2

-- The property we need to prove
def vector_position_at_t := (t: ℝ) → ℝ × ℝ × ℝ

-- The proof problem
theorem find_vector_at_t_neg_2 (a d: ℝ × ℝ × ℝ)
  (h1: vector_position_at_t t1 = point1)
  (h2: vector_position_at_t t2 = point2)
  (h3: vector_position_at_t t3 = point3):
  vector_position_at_t t4 = (-1, 3, -9) :=
sorry

end find_vector_at_t_neg_2_l169_169393


namespace number_of_ordered_triples_l169_169444

open Finset Nat

-- Define the set S
def S : Finset ℕ := (range 66).filter (λ n, n > 0)

-- Define the problem statement
theorem number_of_ordered_triples (hS : ∀ n ∈ S, n ≤ 65) :
  (S.card = 65) →
  (∑ z in S, ∑ x in (S.filter (λ t, t < z)), ∑ y in (S.filter (λ t, t < z)), if (x < y) then 2 else 1) = 89440 :=
by
  sorry

end number_of_ordered_triples_l169_169444


namespace finland_forest_percentage_l169_169223

noncomputable def finland_forested_area : ℝ := 53.42-- million hectares
noncomputable def world_forested_area_billion : ℝ := 8.076-- billion hectares
noncomputable def world_forested_area_million : ℝ := world_forested_area_billion * 1000-- converting billion to million

theorem finland_forest_percentage :
  (finland_forested_area / world_forested_area_million) * 100 ≈ 0.66 :=
by
  -- proof can be provided here
  sorry

end finland_forest_percentage_l169_169223


namespace insert_eights_composite_l169_169163

theorem insert_eights_composite (n : ℕ) : 
  let N_n := 200 * 10^n + (88 * 10^n - 88) / 9 + 21
  in ¬Prime N_n :=
sorry

end insert_eights_composite_l169_169163


namespace min_value_of_function_l169_169320

theorem min_value_of_function :
  ∃ x : ℝ, (∃ t ∈ set.Icc (-real.sqrt 2) (real.sqrt 2), y = (4 - 3 * real.sin x) * (4 - 3 * real.cos x) ∧ t = real.sin x + real.cos x ∧ sin x * cos x = (t^2 - 1)/2) → 
  (y = 16 - 12 * t + 9 * sin x * cos x = 1/2 * (9 * t^2 - 24 * t + 23)) → 
  y = 7/2 := 
sorry

end min_value_of_function_l169_169320


namespace max_cashback_categories_l169_169465

-- Define the expenditure and cashback percentages
def expenses : ℕ → ℕ
| 1 := 2000
| 2 := 5000
| 3 := 3000
| 4 := 3000
| 5 := 1500
| _ := 0

def cashback_rate : ℕ → ℝ
| 1 := 0.05
| 2 := 0.03
| 3 := 0.04
| 4 := 0.05
| 5 := 0.06
| _ := 0

-- Define the cashback amount for each category
def cashback_amount (n : ℕ) : ℝ :=
expenses n * cashback_rate n

-- Define the set of categories
def categories := {1, 2, 3, 4, 5}

-- Prove that selecting Groceries, Entertainment, and Clothing maximizes the cashback
theorem max_cashback_categories :
  let selected := {2, 3, 4} in
  ∀ s ⊆ categories, s.card = 3 → ∑ i in s, cashback_amount i ≤ ∑ i in selected, cashback_amount i :=
begin
  -- The proof will be filled in here
  sorry
end

end max_cashback_categories_l169_169465


namespace exists_d_for_divisibility_l169_169826

theorem exists_d_for_divisibility (a n : ℕ) (h_coprime : Nat.gcd a n = 1) :
  ∃ d : ℕ, ∀ b : ℕ, n ∣ (a^b - 1) ↔ d ∣ b :=
begin
  sorry
end

end exists_d_for_divisibility_l169_169826


namespace abs_diff_squares_105_95_l169_169944

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l169_169944


namespace multiple_with_0_1_digits_l169_169868

theorem multiple_with_0_1_digits (n : ℕ) (hn : 0 < n) :
  ∃ m : ℕ, (n ∣ m) ∧ (m.digits 10).all (λ d, d = 0 ∨ d = 1) ∧ m.digits 10).length ≤ n :=
sorry

end multiple_with_0_1_digits_l169_169868


namespace Fibonacci_has_term_with_8_zeroes_l169_169437

noncomputable def Fibonacci : ℕ → ℕ
| 0 := 0
| 1 := 1
| (n + 2) := Fibonacci n + Fibonacci (n + 1)

theorem Fibonacci_has_term_with_8_zeroes : ∃ n ≤ 10000000000000002, Fibonacci n % 100000000 = 0 := by
sorry

end Fibonacci_has_term_with_8_zeroes_l169_169437


namespace mooncake_packaging_problem_l169_169177

theorem mooncake_packaging_problem :
  ∃ x y : ℕ, 9 * x + 4 * y = 35 ∧ x + y = 5 :=
by
  -- Proof is omitted
  sorry

end mooncake_packaging_problem_l169_169177


namespace three_rugs_area_l169_169564

theorem three_rugs_area (A B C X Y Z: ℝ)
  (h1: A + B + C = 200)
  (h2: X + Y + Z = 140)
  (h3: Y = 22)
  (h4: X + 2 * Y + 3 * Z = 200) : 
  Z = 19 := 
begin
  sorry
end

end three_rugs_area_l169_169564


namespace evaluate_expression_l169_169662

theorem evaluate_expression : ∃ x : ℝ, (x = Real.sqrt (18 + x)) ∧ (x = (1 + Real.sqrt 73) / 2) := by
  sorry

end evaluate_expression_l169_169662


namespace range_of_a_for_perpendicular_tangent_line_l169_169742

theorem range_of_a_for_perpendicular_tangent_line (a : ℝ) :
  (∃ x > 0, ∃ y : ℝ, (f : ℝ → ℝ) = (λ x, a*x^3 + Real.log x) ∧ (f' : ℝ → ℝ) = (λ x, 3*a*x^2 + 1/x) ∧ (f'' : ℝ → ℝ) = (λ x, 6*a*x - 1/x^2) ∧ (∀ x, f'' x ≠ 0) ∧ (∀ x > 0, f' x → ∞)) → a < 0 := 
begin
  sorry
end

end range_of_a_for_perpendicular_tangent_line_l169_169742


namespace center_of_circle_in_second_or_fourth_quadrant_l169_169745

theorem center_of_circle_in_second_or_fourth_quadrant
  (α : ℝ) 
  (hyp1 : ∀ x y : ℝ, x^2 * Real.cos α - y^2 * Real.sin α + 2 = 0 → Real.cos α * Real.sin α > 0)
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 + 2*x*Real.cos α - 2*y*Real.sin α = 0) :
  (-Real.cos α > 0 ∧ Real.sin α > 0) ∨ (-Real.cos α < 0 ∧ Real.sin α < 0) :=
sorry

end center_of_circle_in_second_or_fourth_quadrant_l169_169745


namespace log_function_sum_l169_169710

-- Define the function f
def f (x : ℝ) : ℝ := x / (1 + x)

-- Define the logarithmic identities as constants for simplicity
def log2_3 : ℝ := real.log_b 2 3
def log3_5 : ℝ := real.log_b 3 5
def log3_2 : ℝ := 1 / log2_3
def log5_3 : ℝ := 1 / log3_5

-- State the theorem
theorem log_function_sum : f log2_3 + f log3_5 + f log3_2 + f log5_3 = 2 :=
by
  sorry

end log_function_sum_l169_169710


namespace greatest_sum_of_consecutive_integers_product_less_500_l169_169998

theorem greatest_sum_of_consecutive_integers_product_less_500 :
  ∃ n : ℤ, n * (n + 1) < 500 ∧ (n + (n + 1)) = 43 :=
by
  sorry

end greatest_sum_of_consecutive_integers_product_less_500_l169_169998


namespace interior_angle_of_regular_heptagon_l169_169204

-- Define the problem statement in Lean
theorem interior_angle_of_regular_heptagon : 
  let n := 7 in (n - 2) * 180 / n = 900 / 7 := 
by 
  let n := 7
  show (n - 2) * 180 / n = 900 / 7
  sorry

end interior_angle_of_regular_heptagon_l169_169204


namespace abs_diff_squares_l169_169938

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169938


namespace son_l169_169245

theorem son's_age (F S : ℕ) (h1 : F + S = 75) (h2 : F = 8 * (S - (F - S))) : S = 27 :=
sorry

end son_l169_169245


namespace cost_of_limes_after_30_days_l169_169784

def lime_juice_per_mocktail : ℝ := 1  -- tablespoons per mocktail
def days : ℕ := 30  -- number of days
def lime_juice_per_lime : ℝ := 2  -- tablespoons per lime
def limes_per_dollar : ℝ := 3  -- limes per dollar

theorem cost_of_limes_after_30_days : 
  let total_lime_juice := (lime_juice_per_mocktail * days),
      number_of_limes  := (total_lime_juice / lime_juice_per_lime),
      total_cost       := (number_of_limes / limes_per_dollar)
  in total_cost = 5 :=
by
  sorry

end cost_of_limes_after_30_days_l169_169784


namespace trains_clear_time_l169_169565

def length_train1 : ℝ := 141
def length_train2 : ℝ := 165
def speed_train1_kmh : ℝ := 80
def speed_train2_kmh : ℝ := 65
def kmh_to_ms_factor : ℝ := 1000 / 3600

def total_distance : ℝ := length_train1 + length_train2
def relative_speed_kmh : ℝ := speed_train1_kmh + speed_train2_kmh
def relative_speed_ms : ℝ := relative_speed_kmh * kmh_to_ms_factor

theorem trains_clear_time : total_distance / relative_speed_ms = 7.59 :=
by
  sorry

end trains_clear_time_l169_169565


namespace all_values_equal_l169_169624

noncomputable def f : ℤ × ℤ → ℕ :=
sorry

theorem all_values_equal (f : ℤ × ℤ → ℕ)
  (h_pos : ∀ p, 0 < f p)
  (h_mean : ∀ x y, f (x, y) = 1/4 * (f (x+1, y) + f (x-1, y) + f (x, y+1) + f (x, y-1))) :
  ∀ (x1 y1 x2 y2 : ℤ), f (x1, y1) = f (x2, y2) := 
sorry

end all_values_equal_l169_169624


namespace simplify_sqrt_expression_l169_169890

theorem simplify_sqrt_expression (x : ℝ) : sqrt(9 * x^6 + 3 * x^4) = sqrt(3) * x^2 * sqrt(3 * x^2 + 1) :=
by
  sorry

end simplify_sqrt_expression_l169_169890


namespace ACP_P_trajectory_l169_169337

noncomputable def trajectory_circle_center (a b : ℝ) : ℝ × ℝ :=
  ( (a - (Real.sqrt 3) * b) / 2, ((Real.sqrt 3) * a + b) / 2 )

theorem ACP_P_trajectory (O : ℝ × ℝ) (r : ℝ) (a b : ℝ)
  (A : ℝ × ℝ := (a, b)) (C : ℝ → ℝ × ℝ := λ θ, (r * Real.cos θ, r * Real.sin θ))
  (equilateral_ACP : ∀ θ,  -- A condition ensuring ACP is equilateral
    let
      P := (
        a + (1 / 2) * (r * Real.cos θ - a) - (Real.sqrt 3 / 2) * (r * Real.sin θ - b),
        b + (Real.sqrt 3 / 2) * (r * Real.cos θ - a) + (1 / 2) * (r * Real.sin θ - b)
      )
    in
      True -- Placeholder for ACP being equilateral
    ) :
  ∃ (M : ℝ × ℝ), M = trajectory_circle_center a b ∧
    (∀ θ, let (Px, Py) := (
      a + (1 / 2) * (r * Real.cos θ - a) - (Real.sqrt 3 / 2) * (r * Real.sin θ - b),
      b + (Real.sqrt 3 / 2) * (r * Real.cos θ - a) + (1 / 2) * (r * Real.sin θ - b)
    )
    in (Px - M.1) ^ 2 + (Py - M.2) ^ 2 = r ^ 2) :=
begin
  sorry
end

end ACP_P_trajectory_l169_169337


namespace tan_value_l169_169348

theorem tan_value (θ : ℝ) (h : Real.sin (12 * Real.pi / 5 + θ) + 2 * Real.sin (11 * Real.pi / 10 - θ) = 0) :
  Real.tan (2 * Real.pi / 5 + θ) = 2 :=
by
  sorry

end tan_value_l169_169348


namespace vector_diff_magnitude_l169_169801

variables {ℝ_vector: Type*} [inner_product_space ℝ ℝ_vector]

open_locale real_inner_product_space

def is_unit_vector (v : ℝ_vector) : Prop :=
  ∥v∥ = 1

theorem vector_diff_magnitude (a b : ℝ_vector) (h_a_unit: is_unit_vector a) (h_b_unit: is_unit_vector b) (h_sum_unit: ∥a + b∥ = 1) :
  ∥a - b∥ = real.sqrt 3 :=
begin
  sorry
end

end vector_diff_magnitude_l169_169801


namespace abs_diff_squares_105_95_l169_169946

theorem abs_diff_squares_105_95 : abs ((105:ℤ)^2 - (95:ℤ)^2) = 2000 := by
  sorry

end abs_diff_squares_105_95_l169_169946


namespace measure_angle_C_and_area_l169_169064

noncomputable def triangleProblem (a b c A B C : ℝ) :=
  (a + b = 5) ∧ (c = Real.sqrt 7) ∧ (4 * Real.sin ((A + B) / 2)^2 - Real.cos (2 * C) = 7 / 2)

theorem measure_angle_C_and_area (a b c A B C : ℝ) (h: triangleProblem a b c A B C) :
  C = Real.pi / 3 ∧ (1 / 2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 2 :=
by
  obtain ⟨ha, hb, hc⟩ := h
  sorry

end measure_angle_C_and_area_l169_169064


namespace locus_of_M_on_tangent_segment_l169_169183

theorem locus_of_M_on_tangent_segment (O : Point) (R a : ℝ) (A M : Point) 
  (h_circle : dist O A = R) (h_tangent : dist A M = a ∧ ⊥ ≠ tangent A circle_circum(O, R)) :
    locus M (OM = sqrt (R^2 + a^2)) :=
by
  sorry

end locus_of_M_on_tangent_segment_l169_169183


namespace gas_usage_l169_169773

def distance_dermatologist : ℕ := 30
def distance_gynecologist : ℕ := 50
def car_efficiency : ℕ := 20

theorem gas_usage (d_1 d_2 e : ℕ) (H1 : d_1 = distance_dermatologist) (H2 : d_2 = distance_gynecologist) (H3 : e = car_efficiency) :
  (2 * d_1 + 2 * d_2) / e = 8 :=
by
  rw [H1, H2, H3]
  norm_num
  sorry

end gas_usage_l169_169773


namespace parallel_lines_transitive_l169_169857

theorem parallel_lines_transitive {a b c : Line} (h1 : Parallel a c) (h2 : Parallel b c) : Parallel a b :=
  sorry

end parallel_lines_transitive_l169_169857


namespace fraction_difference_l169_169275

theorem fraction_difference :
  (18/42 : ℚ) - (3/11 : ℚ) = 12/77 := 
by
  -- The proof can be filled in here, but is omitted for the exercise.
  sorry

end fraction_difference_l169_169275


namespace total_students_surveyed_l169_169588

variable (T : ℕ)
variable (F : ℕ)

theorem total_students_surveyed :
  (F = 20 + 60) → (F = 40 * (T / 100)) → (T = 200) :=
by
  intros h1 h2
  sorry

end total_students_surveyed_l169_169588


namespace washed_clothes_l169_169846

def detergent_per_pound : ℝ := 2

def total_detergent_used : ℝ := 18

def pounds_of_clothes_washed : ℝ := total_detergent_used / detergent_per_pound

theorem washed_clothes :
  pounds_of_clothes_washed = 9 := by
  sorry

end washed_clothes_l169_169846


namespace b_2023_is_1_l169_169108

def sequence (n : ℕ) : ℚ :=
  if n = 1 then 3
  else if n = 2 then 4
  else (sequence (n-1) + 1) / sequence (n-2)

theorem b_2023_is_1 : sequence 2023 = 1 := 
  sorry

end b_2023_is_1_l169_169108


namespace time_with_X_and_Y_l169_169516

variable (X Y Z : ℝ) -- Define the rates for each valve
variable (tank_capacity : ℝ) -- Define the tank capacity

-- Conditions
def condition1 : X + Y + Z = 1 / 2 := sorry
def condition2 : X + Z = 1 / 3 := sorry
def condition3 : Y + Z = 1 / 4 := sorry

-- Goal: Prove the time it takes to fill the tank with X and Y open
theorem time_with_X_and_Y (h1 : condition1 X Y Z) (h2 : condition2 X Z) (h3 : condition3 Y Z) : 
  1 / (X + Y) = 2.4 := sorry

end time_with_X_and_Y_l169_169516


namespace king_william_probability_l169_169092

theorem king_william_probability :
  let m := 2
  let n := 15
  m + n = 17 :=
by
  sorry

end king_william_probability_l169_169092


namespace interior_angle_heptagon_l169_169210

theorem interior_angle_heptagon : 
  ∀ (n : ℕ), n = 7 → (5 * 180 / n : ℝ) = 128.57142857142858 :=
by
  intros n hn
  rw hn
  -- The proof is skipped
  sorry

end interior_angle_heptagon_l169_169210


namespace arithmetic_progression_geometric_progression_l169_169738

/-- Problem 1: Arithmetic Progression -/
theorem arithmetic_progression {a : ℕ → ℝ} {n : ℕ} 
  (h : 2 ≤ n) 
  (H : ∀ k, 2 ≤ k ∧ k ≤ n - 1 → a k = (a (k - 1) + a (k + 1)) / 2) :
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → (j - i) * (a 2 - a 1) = a j - a i :=
begin
  sorry
end

/-- Problem 2: Geometric Progression -/
theorem geometric_progression {a : ℕ → ℝ} {n : ℕ} 
  (h : 2 ≤ n) 
  (H : ∀ k, 2 ≤ k ∧ k ≤ n - 1 → a k = sqrt (a (k - 1) * a (k + 1))) :
  ∃ r, ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ n → a i * r ^ (j - i) = a j :=
begin
  sorry
end

end arithmetic_progression_geometric_progression_l169_169738


namespace first_year_payment_l169_169142

theorem first_year_payment (x : ℝ) 
  (second_year : ℝ := x + 2)
  (third_year : ℝ := x + 5)
  (fourth_year : ℝ := x + 9)
  (total_payment : ℝ := x + second_year + third_year + fourth_year)
  (h : total_payment = 96) : x = 20 := 
by
  sorry

end first_year_payment_l169_169142


namespace hall_reunion_attendees_l169_169522

theorem hall_reunion_attendees
  (total_guests : ℕ)
  (oates_attendees : ℕ)
  (both_attendees : ℕ)
  (h : total_guests = 100 ∧ oates_attendees = 50 ∧ both_attendees = 12) :
  ∃ (hall_attendees : ℕ), hall_attendees = 62 :=
by
  sorry

end hall_reunion_attendees_l169_169522


namespace number_of_correct_propositions_l169_169370

open Set

variable {α β : Type} [Preorder α] [Preorder β]

-- Definitions for the problem
def line (ℝ : Type) : Type := ℝ × ℝ
def plane (ℝ : Type) : Type := ℝ × ℝ × ℝ

variable (m n : line ℝ)
variable (α β : plane ℝ)

-- Given conditions as modular definitions
def non_coincident_lines (m n : line ℝ) : Prop := m ≠ n
def non_coincident_planes (α β : plane ℝ) : Prop := α ≠ β

variable (h_mn : non_coincident_lines m n)
variable (h_ab : non_coincident_planes α β)

-- Propositions
def prop1 : Prop := (m ⟂ n ∧ m ⟂ α) → (n ∥ α)
def prop2 : Prop := (m ⟂ α ∧ n ⟂ β ∧ m ∥ n) → (α ∥ β)
def prop3 : Prop := (m_skew_n : ¬ ∃ p, m = n ∧ m ∘ p = n ∘ p) → 
  (m ⊆ α ∧ n ⊆ β ∧ m ∥ β ∧ n ∥ α) → (α ∥ β)
def prop4 : Prop := (α ⟂ β ∧ α ∩ β = m ∧ n ⊆ β ∧ n ⟂ m) → (n ⟂ α)

-- Problem: among these propositions, the number of correct ones is 3
theorem number_of_correct_propositions : prop1 m n α β = false ∧ prop2 m n α β = true ∧ prop3 m n α β = true ∧ prop4 m n α β = true → 3 = 3 := 
sorry

end number_of_correct_propositions_l169_169370


namespace digit_415_of_17_over_39_is_4_l169_169931

-- Define the repeating sequence and its length
def seq : List ℕ := [4, 3, 5, 8, 9, 7]
def seq_length : ℕ := 6

-- Define a function to get the n-th digit in the repeating sequence
def nth_digit_of_repeating_seq (n : ℕ) : ℕ := seq[(n % seq_length)]

-- The main theorem to be proven
theorem digit_415_of_17_over_39_is_4 : nth_digit_of_repeating_seq 415 = 4 := 
by
sory

end digit_415_of_17_over_39_is_4_l169_169931


namespace royalties_amount_l169_169073

/--
Given the following conditions:
1. No tax for royalties up to 800 yuan.
2. For royalties exceeding 800 yuan but not exceeding 4000 yuan, tax is levied at 14% on the amount exceeding 800 yuan.
3. For royalties exceeding 4000 yuan, tax is levied at 11% of the total royalties.

If someone has paid 420 yuan in taxes for publishing a book, prove that their royalties amount to 3800 yuan.
-/
theorem royalties_amount (r : ℝ) (h₁ : ∀ r, r ≤ 800 → 0 = r * 0 / 100)
  (h₂ : ∀ r, 800 < r ∧ r ≤ 4000 → 0.14 * (r - 800) = r * 0.14 / 100)
  (h₃ : ∀ r, r > 4000 → 0.11 * r = 420) : r = 3800 := sorry

end royalties_amount_l169_169073


namespace log_comparison_l169_169684

theorem log_comparison (m n : ℝ) (hmn : 2^m > 2^n) (hn4 : 2^n > 4) : log m 2 < log n 2 :=
by
  sorry

end log_comparison_l169_169684


namespace digit_150_of_decimal_3_div_11_l169_169528

theorem digit_150_of_decimal_3_div_11 : 
  (let digits := [2, 7] in digits[(150 % digits.length)]) = 7 :=
by
  sorry

end digit_150_of_decimal_3_div_11_l169_169528


namespace sum_arithmetic_seq_eq_l169_169761

variables {α : Type*} [linear_ordered_field α]

def arithmetic_seq (a1 d : α) (n : ℕ) : α := a1 + (n - 1) * d

theorem sum_arithmetic_seq_eq :
  ∀ (a1 d : α) (n : ℕ),
  (a1 + arithmetic_seq a1 d 4 = 10) →
  (arithmetic_seq a1 d 2 - arithmetic_seq a1 d 3 = 2) →
  (finset.range (n + 1)).sum (λ k, arithmetic_seq a1 d (k + 1)) = 9 * n - n^2 :=
by
  sorry

end sum_arithmetic_seq_eq_l169_169761


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169976

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169976


namespace fraction_value_l169_169014

theorem fraction_value (a b : ℝ) (h : 1 / a - 1 / b = 4) : 
    (a - 2 * a * b - b) / (2 * a + 7 * a * b - 2 * b) = 6 :=
by
  sorry

end fraction_value_l169_169014


namespace smallest_gcd_bc_l169_169054

theorem smallest_gcd_bc (a b c : ℕ) (h1 : Nat.gcd a b = 240) (h2 : Nat.gcd a c = 1001) : Nat.gcd b c = 1 :=
sorry

end smallest_gcd_bc_l169_169054


namespace prime_number_probability_l169_169190

-- Define the sets of numbers on Spinner A and Spinner B.
def spinnerA : set ℕ := {1, 4, 7, 9}
def spinnerB : set ℕ := {2, 3, 5, 8}

-- Define a predicate to check for prime numbers.
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Define the set of prime sums when spinning both spinners once.
def prime_sums : set ℕ := {n | ∃ a ∈ spinnerA, ∃ b ∈ spinnerB, n = a + b ∧ is_prime n}

-- Calculate the probability of prime sums.
def prime_probability : ℚ := (prime_sums.size : ℚ) / (spinnerA.size * spinnerB.size)

-- Theorem: The probability that the sum of the numbers on which the spinners land is prime is 1/4.
theorem prime_number_probability : prime_probability = 1 / 4 := sorry

end prime_number_probability_l169_169190


namespace range_of_a_l169_169702

variable {α : Type*} [LinearOrderedField α]

def isEven (f : α → α) : Prop := ∀ x, f x = f (-x)
def monotonicIncreasing (f : α → α) : Prop := ∀ {x y}, x < y → f x ≤ f y

theorem range_of_a (f : α → α) (a : α) 
  (h_even : isEven f) 
  (h_mon_inc : monotonicIncreasing f)
  (h_interval : ∀ x, x ∈ Iio (0:α) → f x < f 0) 
  (h_cond : f (2 ^ |a - 1|) > f (-real.sqrt 2)) : 
  (1 / 2 : α) < a ∧ a < (3 / 2 : α) := 
sorry

end range_of_a_l169_169702


namespace parking_spots_l169_169248

def numberOfLevels := 5
def openSpotsOnLevel1 := 4
def openSpotsOnLevel2 := openSpotsOnLevel1 + 7
def openSpotsOnLevel3 := openSpotsOnLevel2 + 6
def openSpotsOnLevel4 := 14
def openSpotsOnLevel5 := openSpotsOnLevel4 + 5
def totalOpenSpots := openSpotsOnLevel1 + openSpotsOnLevel2 + openSpotsOnLevel3 + openSpotsOnLevel4 + openSpotsOnLevel5

theorem parking_spots :
  openSpotsOnLevel5 = 19 ∧ totalOpenSpots = 65 := by
  sorry

end parking_spots_l169_169248


namespace friend_separation_lineup_l169_169758

-- Declare six students and specify the conditions
constant students : Fin 6 → Type
constant friends : Fin 3 → Type
constant student_not_next_to : students → students → Prop
constant is_friend : friends → students
constant is_not_adjacent (s1 s2 : students) : Prop := ¬ student_not_next_to s1 s2

-- Main statement
theorem friend_separation_lineup :
  (count_permutations (students) : Nat) - (count_prohibited (students, friends, is_not_adjacent) : Nat) = 576 := by
  sorry

end friend_separation_lineup_l169_169758


namespace solve_expression_l169_169141

theorem solve_expression : 
  let d := 0.76 * 0.76 + 0.76 * 0.2 + 0.04 in
  (0.76 * 0.76 * 0.76 - 0.008) / d = 0.560 :=
by
  -- Definitions and computation steps would go within the proof block if provided
  sorry

end solve_expression_l169_169141


namespace num_interesting_numbers_l169_169289

def g (n : ℕ) : ℕ :=
if n = 1 then 1 else 
let factors := n.factorization in
factors.keys.prod (λ p, (p + 2) ^ (factors p - 1))

def g_m (m n : ℕ) : ℕ :=
Nat.iterate m g n

def is_bounded_sequence (sequence : ℕ → ℕ) : Prop :=
∃ B, ∀ m, sequence m ≤ B

def is_interesting (n : ℕ) : Prop :=
is_bounded_sequence (g_m · n)

theorem num_interesting_numbers :
  {n | 1 ≤ n ∧ n ≤ 500 ∧ is_interesting n}.card = 9 :=
sorry

end num_interesting_numbers_l169_169289


namespace square_circle_area_ratio_l169_169601

-- Definitions
variable (r : ℝ)
def s := 2 * r
def area_square := s ^ 2
def area_circle := π * r ^ 2

-- Theorem
theorem square_circle_area_ratio : ∀ r > 0, (area_square / area_circle) = 4 / π :=
by 
  intro r hr_pos
  have h_s : s = 2 * r := by sorry
  have h_area_square : area_square = (2 * r) ^ 2 := by sorry
  have h_area_square_eval : area_square = 4 * r ^ 2 := by sorry
  have h_area_circle : area_circle = π * r ^ 2 := by sorry
  have h_ratio : (area_square / area_circle) = (4 * r ^ 2) / (π * r ^ 2) := by sorry
  have h_cancel_r_sq : (4 * r ^ 2) / (π * r ^ 2) = 4 / π := by sorry
  exact h_cancel_r_sq

end square_circle_area_ratio_l169_169601


namespace max_distance_and_coverage_area_l169_169184

theorem max_distance_and_coverage_area (r : ℝ) (radars : ℕ) (width : ℝ) :
  r = 15 → radars = 8 → width = 18 →
  ∃ (max_distance : ℝ) (area : ℝ),
    max_distance = 12 / (Real.sin (Real.pi / 8)) ∧
    area = 432 * Real.pi / (Real.tan (Real.pi / 8)) :=
by
  intros hr hrad hw
  use (12 / (Real.sin (Real.pi / 8)))
  use (432 * Real.pi / (Real.tan (Real.pi / 8)))
  refine ⟨_, _⟩
  . simp [hr, hrad, hw]
  . simp [hr, hrad, hw]
  sorry

end max_distance_and_coverage_area_l169_169184


namespace vector_difference_magnitude_l169_169805

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def unit_vectors (a b : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 1

theorem vector_difference_magnitude (a b : V)
  (h : unit_vectors a b) : ∥a - b∥ = real.sqrt 3 :=
by sorry

end vector_difference_magnitude_l169_169805


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169969

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169969


namespace abs_diff_squares_l169_169937

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169937


namespace quiz_score_of_dropped_student_l169_169562

theorem quiz_score_of_dropped_student (avg_16_students : ℤ) (avg_15_students : ℤ)
    (h1 : avg_16_students = 62.5)
    (h2 : avg_15_students = 62.0) :
    let total_16 := 16 * avg_16_students
    let total_15 := 15 * avg_15_students
    let x := total_16 - total_15
    x = 70 :=
by
  -- Definitions from the problem conditions
  let total_16 := 16 * 62.5
  let total_15 := 15 * 62.0
  let x := total_16 - total_15
  -- The proof would follow from the definitions
  sorry

end quiz_score_of_dropped_student_l169_169562


namespace eight_digit_numbers_l169_169506

theorem eight_digit_numbers (digits : Finset ℕ) (non_zero_first_digit : Finset ℕ) (n : ℕ) (h1 : digits = {2, 0, 1, 9, 20, 19}) 
  (h2 : non_zero_first_digit = {2, 1, 9, 20, 19}) : 
  ∃ numbers_count : ℕ, numbers_count = 498 :=
by {
  -- Use the given conditions
  have condition1 : digits = {2, 0, 1, 9, 20, 19} := h1,
  have condition2 : non_zero_first_digit = {2, 1, 9, 20, 19} := h2,
  -- Goal is to show that the number of 8-digit numbers as described is 498
  have target : 498 = 498, by refl,
  exact ⟨498, target⟩,
}

end eight_digit_numbers_l169_169506


namespace product_of_elements_l169_169693

noncomputable def satisfies_condition (A : set ℝ) := ∀ a ∈ A, (1 + a) / (1 - a) ∈ A

theorem product_of_elements (A : set ℝ) (hA : satisfies_condition A) :
  ∃ p : ℝ, (∀ a ∈ A, a ≠ 1 ∧ a ≠ -1 ∧ a ≠ 0) ∧ p = 1 := 
sorry

end product_of_elements_l169_169693


namespace faye_pencils_allocation_l169_169664

theorem faye_pencils_allocation (pencils total_pencils rows : ℕ) (h_pencils : total_pencils = 6) (h_rows : rows = 2) (h_allocation : pencils = total_pencils / rows) : pencils = 3 := by
  sorry

end faye_pencils_allocation_l169_169664


namespace min_value_S_l169_169860

-- Define the condition "rolling a standard die"
def die_roll (i: ℕ) : Prop := 1 ≤ i ∧ i ≤ 6

-- Define the condition of the sum 1994 being greater than 0
def sum_reachable (n target: ℕ) : Prop :=
  ∃ seq: Fin n → ℕ, (∀ i, die_roll (seq i)) ∧ (Finset.univ.sum seq = target)

-- Define the minimum value of S
def min_value (n: ℕ) (S: ℕ)  : Prop :=
  S = 334

-- The theorem statement in Lean
theorem min_value_S : ∃ S, (min_value 333 S) ∧ (sum_reachable 333 1994) ∧ (sum_reachable 333 S) := 
begin
  use 334,
  split,
  { exact rfl },
  split,
  { sorry },
  { sorry },
end

end min_value_S_l169_169860


namespace only_zero_function_satisfies_inequality_l169_169351

noncomputable def f (x : ℝ) : ℝ := sorry

theorem only_zero_function_satisfies_inequality (α β : ℝ) (hα : α ≠ 0) (hβ : β ≠ 0) :
  (∀ x y : ℝ, 0 < x → 0 < y →
    f x * f y ≥ (y^α / (x^α + x^β)) * (f x)^2 + (x^β / (y^α + y^β)) * (f y)^2) →
  ∀ x : ℝ, 0 < x → f x = 0 :=
sorry

end only_zero_function_satisfies_inequality_l169_169351


namespace find_x_set_l169_169636

-- Define the conditions of the problem
def is_even {R : Type*} [LinearOrderedField R] (f : R → R) : Prop :=
  ∀ x, f x = f (-x)

def is_decreasing {R : Type*} [LinearOrderedField R] [Preorder R] 
  (f : R → R) : Prop :=
  ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x

-- Define the logarithmic condition + is_zero condition on given value
def log_base (B x : ℝ) := log x / log B 

-- The main theorem to prove correct answer
theorem find_x_set (f : ℝ → ℝ) (hf_even : is_even f) 
  (hf_decreasing : is_decreasing f) (hf_half : f (1 / 2) = 0) :
  {x | f (log_base (1 / 4) x) < 0} = {x | 0 < x ∧ x < 1 / 2} ∪ {x | x > 2} :=
sorry

end find_x_set_l169_169636


namespace circle_area_in_sq_cm_l169_169546

theorem circle_area_in_sq_cm (diameter_meters : ℝ) (h : diameter_meters = 5) : 
  let radius_meters := diameter_meters / 2
  let area_square_meters := π * radius_meters^2
  let area_square_cm := area_square_meters * 10000
  area_square_cm = 62500 * π :=
by
  sorry

end circle_area_in_sq_cm_l169_169546


namespace greatest_sum_consecutive_integers_lt_500_l169_169991

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169991


namespace sum_of_squares_divisibility_l169_169822

theorem sum_of_squares_divisibility
  (p : ℕ) (hp : Nat.Prime p)
  (x y z : ℕ)
  (hx : 0 < x) (hxy : x < y) (hyz : y < z) (hzp : z < p)
  (hmod_eq : ∀ a b c : ℕ, a^3 % p = b^3 % p → b^3 % p = c^3 % p → a^3 % p = c^3 % p) :
  (x^2 + y^2 + z^2) % (x + y + z) = 0 := by
  sorry

end sum_of_squares_divisibility_l169_169822


namespace minimum_balls_each_color_minimum_balls_ten_one_color_l169_169753

section

variables (white red black : ℕ)
-- Conditions for the problem
def conditions := white = 5 ∧ red = 12 ∧ black = 20

-- Problem (a): Minimum number of balls to get at least one ball of each color
theorem minimum_balls_each_color : ∀ (white red black : ℕ), conditions white red black → 
  ∃ n, n = 33 :=
by
  intros white red black h
  have h_white : white = 5 := h.left
  have h_red : red = 12 := h.right.left
  have h_black : black = 20 := h.right.right
  use 33
  sorry

-- Problem (b): Minimum number of balls to get 10 balls of one color
theorem minimum_balls_ten_one_color : ∀ (white red black : ℕ), conditions white red black → 
  ∃ n, n = 24 :=
by
  intros white red black h
  have h_white : white = 5 := h.left
  have h_red : red = 12 := h.right.left
  have h_black : black = 20 := h.right.right
  use 24
  sorry

end

end minimum_balls_each_color_minimum_balls_ten_one_color_l169_169753


namespace wholesale_cost_per_bag_l169_169674

theorem wholesale_cost_per_bag 
  (gross_profit_percent : ℝ)
  (selling_price : ℝ)
  (wholesale_cost : ℝ)
  (profit_ratio : gross_profit_percent = 0.16)
  (selling_price_value : selling_price = 28)
  (profit_eqn : selling_price = (1 + gross_profit_percent) * wholesale_cost) : 
  wholesale_cost = 24.14 :=
by
  rw [selling_price_value, profit_ratio, profit_eqn]
  sorry

end wholesale_cost_per_bag_l169_169674


namespace calculate_costs_l169_169461

structure FamilyCosts where
  highway_mileage    : ℕ
  highway_mpg        : ℕ
  city_mileage       : ℕ
  city_mpg           : ℕ
  highway_fuel_price : ℝ
  city_fuel_price    : ℝ

def maintenance_cost (mileage : ℕ) (cost_per_mile : ℝ) : ℝ :=
  mileage * cost_per_mile

def fuel_cost (miles : ℕ) (mpg : ℕ) (price_per_gal : ℝ) : ℝ :=
  (miles / mpg) * price_per_gal

def total_family_cost (fc : FamilyCosts) : ℝ :=
  let highway_fuel_cost := fuel_cost fc.highway_mileage fc.highway_mpg fc.highway_fuel_price
  let city_fuel_cost := fuel_cost fc.city_mileage fc.city_mpg fc.city_fuel_price
  let highway_maintenance := maintenance_cost fc.highway_mileage 0.05
  let city_maintenance := maintenance_cost fc.city_mileage 0.07
  highway_fuel_cost + city_fuel_cost + highway_maintenance + city_maintenance

noncomputable def jensen_family : FamilyCosts := {
  highway_mileage := 210,
  highway_mpg := 35,
  city_mileage := 54,
  city_mpg := 18,
  highway_fuel_price := 3.70,
  city_fuel_price := 4.20
}

noncomputable def smith_family : FamilyCosts := {
  highway_mileage := 240,
  highway_mpg := 30,
  city_mileage := 60,
  city_mpg := 15,
  highway_fuel_price := 3.85,
  city_fuel_price := 4.00
}

noncomputable def greens_family : FamilyCosts := {
  highway_mileage := 260,
  highway_mpg := 32,
  city_mileage := 48,
  city_mpg := 20,
  highway_fuel_price := 3.75,
  city_fuel_price := 4.10
}

theorem calculate_costs : total_family_cost jensen_family = 49.08 ∧ 
                         total_family_cost smith_family = 63.00 ∧
                         total_family_cost greens_family = 56.67 ∧
                         smith_family.total_cost = max jensen_family.total_cost (max smith_family.total_cost greens_family.total_cost) ∧
                         jensen_family.total_cost = min jensen_family.total_cost (min smith_family.total_cost greens_family.total_cost) ∧
                         total_family_cost jensen_family + total_family_cost smith_family + total_family_cost greens_family = 168.75 := 
by
  sorry

end calculate_costs_l169_169461


namespace exists_subset_no_three_ap_l169_169106

-- Define the set S_n
def S (n : ℕ) : Finset ℕ := (Finset.range ((3^n + 1) / 2 + 1)).image (λ i => i + 1)

-- Define the property of no three elements forming an arithmetic progression
def no_three_form_ap (M : Finset ℕ) : Prop :=
  ∀ a b c, a ∈ M → b ∈ M → c ∈ M → a < b → b < c → 2 * b ≠ a + c

-- Define the theorem statement
theorem exists_subset_no_three_ap (n : ℕ) :
  ∃ M : Finset ℕ, M ⊆ S n ∧ M.card = 2^n ∧ no_three_form_ap M :=
sorry

end exists_subset_no_three_ap_l169_169106


namespace sum_of_products_of_subsets_l169_169794

open Set

def X : Set ℝ := {1 / 2, 1 / 3, 1 / 4, 1 / 5, 1 / 6}

def product_of_set (G : Set ℝ) : ℝ :=
  if G.nonempty then G.prod id else 1

theorem sum_of_products_of_subsets :
  ∑ G in (X.powerset \ {∅}), product_of_set G = 5 / 2 :=
by
  sorry

end sum_of_products_of_subsets_l169_169794


namespace percentage_error_in_area_l169_169612

-- Definitions based on conditions
def actual_side (s : ℝ) := s
def measured_side (s : ℝ) := s * 1.01
def actual_area (s : ℝ) := s^2
def calculated_area (s : ℝ) := (measured_side s)^2

-- Theorem statement of the proof problem
theorem percentage_error_in_area (s : ℝ) : 
  (calculated_area s - actual_area s) / actual_area s * 100 = 2.01 := 
by 
  -- Proof is omitted
  sorry

end percentage_error_in_area_l169_169612


namespace four_times_six_pow_n_plus_five_pow_n_plus_one_minus_nine_is_multiple_of_twenty_days_later_3100_days_friday_l169_169571

-- Problem 1:
theorem four_times_six_pow_n_plus_five_pow_n_plus_one_minus_nine_is_multiple_of_twenty 
  (n : ℕ) (hn : 0 < n) : 20 ∣ (4 * 6^n + 5^n + 1 - 9) :=
sorry

-- Problem 2:
theorem days_later_3100_days_friday
  (today_is_monday : true) : true -- To assert that 3100 days from Monday is Friday
  :=
begin
  have day_of_week_3100 := (3100 % 7), /-* Use the built-in modulus operator to find the remainder *-/
  have day_of_week := 4, -- Given 3100 mod 7 = 4 means it will be a Friday.
  exact true.intro -- No actual Lean statement required for such proved assertion,
end

end four_times_six_pow_n_plus_five_pow_n_plus_one_minus_nine_is_multiple_of_twenty_days_later_3100_days_friday_l169_169571


namespace calculation_result_l169_169570

theorem calculation_result
  (a b c : ℤ)
  (h₁ : a = 786)
  (h₂ : b = 74)
  (h₃ : c = 30) :
  ((a * b) / c : ℤ) = 1,938 := by
  sorry

end calculation_result_l169_169570


namespace exists_linear_eq_exactly_m_solutions_l169_169471

theorem exists_linear_eq_exactly_m_solutions (m : ℕ) (hm : 0 < m) :
  ∃ (a b c : ℤ), ∀ (x y : ℕ), a * x + b * y = c ↔
    (1 ≤ x ∧ 1 ≤ y ∧ x + y = m + 1) :=
by
  sorry

end exists_linear_eq_exactly_m_solutions_l169_169471


namespace julios_spending_on_limes_l169_169777

theorem julios_spending_on_limes 
    (days : ℕ) (lime_juice_per_day : ℕ) (lime_juice_per_lime : ℕ) (limes_per_dollar : ℕ) 
    (total_spending : ℝ) 
    (h1 : days = 30) 
    (h2 : lime_juice_per_day = 1) 
    (h3 : lime_juice_per_lime = 2) 
    (h4 : limes_per_dollar = 3) 
    (h5 : total_spending = 5) :
    let lime_juice_needed := days * lime_juice_per_day,
        total_limes := lime_juice_needed / lime_juice_per_lime,
        cost := (total_limes / limes_per_dollar : ℕ) in
    (cost : ℝ) = total_spending := 
by 
    sorry

end julios_spending_on_limes_l169_169777


namespace marble_fraction_l169_169396

theorem marble_fraction (total_marbles : ℕ) (frac_green : ℚ) (green_increase : ℚ) (yellow_mult : ℕ) :
  total_marbles = 120 →
  frac_green = 4 / 9 →
  green_increase = 1.5 →
  yellow_mult = 3 →
  let green_marbles := (frac_green * total_marbles).toNat,
      yellow_marbles := ((1 - frac_green) * total_marbles).toNat,
      new_green := (green_marbles * green_increase).toNat,
      new_yellow := yellow_marbles * yellow_mult,
      new_total := new_green + new_yellow in
  (new_green : ℚ) / new_total = 80 / 281 :=
by { intros, sorry }

end marble_fraction_l169_169396


namespace sum_divides_exp_sum_l169_169688

theorem sum_divides_exp_sum (p a b c d : ℕ) [Fact (Nat.Prime p)] 
  (h1 : 0 < a) (h2 : a < b) (h3 : b < c) (h4 : c < d) (h5 : d < p)
  (h6 : a^4 % p = b^4 % p) (h7 : b^4 % p = c^4 % p) (h8 : c^4 % p = d^4 % p) :
  (a + b + c + d) ∣ (a^2013 + b^2013 + c^2013 + d^2013) :=
sorry

end sum_divides_exp_sum_l169_169688


namespace geometric_sum_ratio_l169_169405

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

theorem geometric_sum_ratio (a : ℕ → ℝ) (q : ℝ)
  (h₀ : is_geometric_sequence a q)
  (h₁ : 6 * a 7 = (a 8 + a 9) / 2) :
  sum_first_n_terms a 6 / sum_first_n_terms a 3 = 28 :=
by
  sorry

end geometric_sum_ratio_l169_169405


namespace next_palindromic_year_next_palindromic_odd_year_next_palindromic_prime_year_l169_169232

def is_palindromic (n: ℕ) : Prop := 
  n.toString.reverse.toInt = n

def is_prime (n: ℕ) : Prop := 
  n > 1 ∧ ∀ (k : ℕ), k ∣ n → k = 1 ∨ k = n

noncomputable def next_palindromic (y: ℕ) : ℕ :=
  if y <= 2002 then 2112 else if y <= 3003 then 3003 else 10301

theorem next_palindromic_year: 
  ∃ y, is_palindromic y ∧ y > 2002 ∧ y = 2112 :=
by
  use 2112
  have h1 : is_palindromic 2112 := sorry
  have h2 : 2112 > 2002 := by norm_num
  have h3 : 2112 = 2112 := by rfl
  exact ⟨h1, ⟨h2, h3⟩⟩

theorem next_palindromic_odd_year: 
  ∃ y, is_palindromic y ∧ y % 2 = 1 ∧ y > 1991 ∧ y = 3003 :=
by
  use 3003
  have h1 : is_palindromic 3003 := sorry
  have h2 : 3003 % 2 = 1 := by norm_num
  have h3 : 3003 > 1991 := by norm_num
  have h4 : 3003 = 3003 := by rfl
  exact ⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩

theorem next_palindromic_prime_year: 
  ∃ y, is_palindromic y ∧ is_prime y ∧ y > 929 ∧ y = 10301 :=
by
  use 10301
  have h1 : is_palindromic 10301 := sorry
  have h2 : is_prime 10301 := sorry
  have h3 : 10301 > 929 := by norm_num
  have h4 : 10301 = 10301 := by rfl
  exact ⟨h1, ⟨h2, ⟨h3, h4⟩⟩⟩

end next_palindromic_year_next_palindromic_odd_year_next_palindromic_prime_year_l169_169232


namespace ratio_mercedes_jonathan_l169_169089

theorem ratio_mercedes_jonathan (M : ℝ) (J : ℝ) (D : ℝ) 
  (h1 : J = 7.5) 
  (h2 : D = M + 2) 
  (h3 : M + D = 32) : M / J = 2 :=
by
  sorry

end ratio_mercedes_jonathan_l169_169089


namespace pen_collection_l169_169566

/-- You collect pens. You start with 25 pens.
Mike gives you 22 more pens.
Cindy doubles your pens.
You give away 19 pens.
Prove that you have 75 pens at the end. -/
theorem pen_collection (starting_pens : ℕ) (mikes_pens : ℕ) (sharons_pens : ℕ)
  (cindy_double : ℕ -> ℕ) :
  starting_pens = 25 →
  mikes_pens = 22 →
  cindy_double = λ n, 2 * n →
  sharons_pens = 19 →
  let total_pens := cindy_double (starting_pens + mikes_pens) - sharons_pens in
  total_pens = 75 :=
by
  intros h_start h_mike h_cindy h_sharon
  let total_pens := cindy_double (starting_pens + mikes_pens) - sharons_pens
  have : total_pens = 75 := sorry
  exact this

end pen_collection_l169_169566


namespace sum_ratio_l169_169343

variable {a_n : ℕ → ℕ}
variable {S_n : ℕ → ℕ}
variable {d : ℕ}

axiom arithmetic_sequence : ∀ n, a_n n = a_n 1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n, S_n n = n * (a_n 1 + a_n n) / 2
axiom condition_a4 : a_n 4 = 2 * (a_n 2 + a_n 3)
axiom non_zero_difference : d ≠ 0

theorem sum_ratio : S_n 7 / S_n 4 = 7 / 4 := 
by
  sorry

end sum_ratio_l169_169343


namespace average_computer_time_per_person_is_95_l169_169178

def people : ℕ := 8
def computers : ℕ := 5
def work_time : ℕ := 152 -- total working day minutes

def total_computer_time : ℕ := work_time * computers
def average_time_per_person : ℕ := total_computer_time / people

theorem average_computer_time_per_person_is_95 :
  average_time_per_person = 95 := 
by
  sorry

end average_computer_time_per_person_is_95_l169_169178


namespace max_value_fraction_l169_169832

theorem max_value_fraction (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) :
  (∀ x y k : ℝ, 0 < x → 0 < y → 0 < k → 
  let expr := (kx + y)^2 / (x^2 + y^2)
  in expr ≤ k^2 + 1) :=
sorry

end max_value_fraction_l169_169832


namespace interior_angle_of_regular_heptagon_l169_169206

-- Define the problem statement in Lean
theorem interior_angle_of_regular_heptagon : 
  let n := 7 in (n - 2) * 180 / n = 900 / 7 := 
by 
  let n := 7
  show (n - 2) * 180 / n = 900 / 7
  sorry

end interior_angle_of_regular_heptagon_l169_169206


namespace description_of_M_l169_169367

theorem description_of_M :
  let M := {(x, y) | y = x^2} in
  M = {(x, y) | y = x^2} :=
by
  sorry

end description_of_M_l169_169367


namespace cos_identity_l169_169013

noncomputable def cos_value_condition (α : ℝ) : Prop :=
  π < α ∧ α < 3 * π / 2 ∧ cos (π + α) = sqrt 3 / 2

theorem cos_identity (α : ℝ) (h : cos_value_condition α) : cos (2 * π - α) = - (sqrt 3 / 2) :=
by
  sorry

end cos_identity_l169_169013


namespace imaginary_part_of_exp_neg_pi_div_6_l169_169649

theorem imaginary_part_of_exp_neg_pi_div_6 :
  ∃ (z : ℂ), z = exp (-π / 6 * I) ∧ z.im = -1 / 2 :=
by
  -- Definitions and assumptions from the conditions
  let θ := -π / 6
  have h_euler : exp (θ * I) = complex.of_real (cos θ) + complex.I * (sin θ), by apply exp_mul_I
  have h_cos : cos θ = sqrt 3 / 2, from by norm_num
  have h_sin : sin θ = -1 / 2, from by norm_num
  -- The proof is skipped
  sorry

end imaginary_part_of_exp_neg_pi_div_6_l169_169649


namespace sequence_properties_l169_169329

theorem sequence_properties (n : ℕ) (h : 0 < n) :
  let S_n := 3 * n - 2 * n^2
  let a_n := -4 * n + 5
  S_n = (finset.range n).sum a_n ∧ S_n ≥ n * a_n :=
sorry

end sequence_properties_l169_169329


namespace cross_product_self_zero_l169_169041

variables (a b : ℝ × ℝ × ℝ)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.2 * w.3 - v.3 * w.2.2, v.3 * w.1 - v.1 * w.3, v.1 * w.2.2 - v.2.2 * w.1)

theorem cross_product_self_zero
  (h : cross_product a b = (-3, 6, 2)) :
  cross_product (2 • a + 3 • b) (2 • a + 3 • b) = (0, 0, 0) :=
  sorry

end cross_product_self_zero_l169_169041


namespace focus_of_parabola_x_squared_eq_4y_is_0_1_l169_169884

theorem focus_of_parabola_x_squared_eq_4y_is_0_1 :
  ∃ (x y : ℝ), (0, 1) = (x, y) ∧ (∀ a b : ℝ, a^2 = 4 * b → (x, y) = (0, 1)) :=
sorry

end focus_of_parabola_x_squared_eq_4y_is_0_1_l169_169884


namespace find_norms_sum_l169_169097

open Real EuclideanSpace Finset

-- Defining the conditions
-- Definitions and conditions given in step a)
def midpoint_condition (a b m : EuclideanSpace ℝ (Fin 2)) : Prop :=
  m = (a + b) / 2

def given_conditions (a b m : EuclideanSpace ℝ (Fin 2)) : Prop :=
  m = ⟨4, 5⟩ ∧ (a ⬝ b = 12)

-- The theorem statement
theorem find_norms_sum (a b m : EuclideanSpace ℝ (Fin 2)) (h1 : midpoint_condition a b m)
  (h2 : given_conditions a b m) : ∥a∥^2 + ∥b∥^2 = 140 :=
by
  sorry

end find_norms_sum_l169_169097


namespace isosceles_trapezoid_height_l169_169482

/-- Given an isosceles trapezoid with area 100 and diagonals that are mutually perpendicular,
    we want to prove that the height of the trapezoid is 10. -/
theorem isosceles_trapezoid_height (BC AD h : ℝ) 
    (area_eq_100 : 100 = (1 / 2) * (BC + AD) * h)
    (height_eq_half_sum : h = (1 / 2) * (BC + AD)) :
    h = 10 :=
by
  sorry

end isosceles_trapezoid_height_l169_169482


namespace magnitude_difference_sqrt3_l169_169813

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Condition 1: a and b are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∥v∥ = 1

-- Condition 2: |a + b| = 1
def sum_of_unit_vectors_has_unit_norm : Prop :=
  ∥a + b∥ = 1

-- Question: |a - b| = sqrt(3)
theorem magnitude_difference_sqrt3 (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hab : sum_of_unit_vectors_has_unit_norm a b) :
  ∥a - b∥ = Real.sqrt 3 :=
sorry

end magnitude_difference_sqrt3_l169_169813


namespace Julio_limes_expense_l169_169782

/-- Julio's expense on limes after 30 days --/
theorem Julio_limes_expense :
  ((30 * (1 / 2)) / 3) * 1 = 5 := 
by
  sorry

end Julio_limes_expense_l169_169782


namespace alberto_vs_bjorn_distance_difference_l169_169160

noncomputable def alberto_distance (t : ℝ) : ℝ := (3.75 / 5) * t
noncomputable def bjorn_distance (t : ℝ) : ℝ := (3.4375 / 5) * t

theorem alberto_vs_bjorn_distance_difference :
  alberto_distance 5 - bjorn_distance 5 = 0.3125 :=
by
  -- proof goes here
  sorry

end alberto_vs_bjorn_distance_difference_l169_169160


namespace quadratic_solution_l169_169171

theorem quadratic_solution :
  ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := sorry

end quadratic_solution_l169_169171


namespace smallest_difference_l169_169265

-- Define the main theorem
theorem smallest_difference (PQ QR PR : ℤ)
  (h1 : PQ + QR + PR = 3030)
  (h2 : PQ < QR ∧ QR ≤ PR)
  (h3 : nat.prime PQ)
  (h4 : PQ + QR > PR)
  (h5 : PQ + PR > QR)
  (h6 : QR + PR > PQ) :
  ∃ d : ℤ, d = QR - PQ ∧ ∀ d' : ℤ, (d' = QR - PQ ∧ d' < d) → d' ≥ 15 := 
begin
  sorry
end

end smallest_difference_l169_169265


namespace total_distance_l169_169433

-- Definitions based on the conditions in the problem statement
def Jonathan_distance : ℝ := 7.5
def Mercedes_distance : ℝ := 2.5 * Jonathan_distance
def Davonte_distance : ℝ := Mercedes_distance + 3.25
def Felicia_distance : ℝ := Davonte_distance - 1.75
def Emilia_distance : ℝ := (Jonathan_distance + Davonte_distance + Felicia_distance) / 3

-- The main statement we want to prove
theorem total_distance :
  let total_distance := Mercedes_distance + Davonte_distance + Felicia_distance + Emilia_distance
  in total_distance = 77.5833 := by
  -- Proof will be constructed here based on definitions
  sorry

end total_distance_l169_169433


namespace train_crossing_time_l169_169051
#check 36 * (1000 / 3600) = 10 

-- Define constants for the problem
constant length_train : ℕ := 120
constant length_bridge : ℕ := 200
constant speed_train_mps : ℕ := (36 * 1000) / 3600

-- The theorem we want to prove
theorem train_crossing_time : 
  let total_distance := length_train + length_bridge in
  let time := total_distance / speed_train_mps in
  time = 32 :=
by
  let total_distance := length_train + length_bridge
  let time := total_distance / speed_train_mps
  show time = 32 from sorry

end train_crossing_time_l169_169051


namespace magnitude_difference_sqrt3_l169_169817

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Condition 1: a and b are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∥v∥ = 1

-- Condition 2: |a + b| = 1
def sum_of_unit_vectors_has_unit_norm : Prop :=
  ∥a + b∥ = 1

-- Question: |a - b| = sqrt(3)
theorem magnitude_difference_sqrt3 (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hab : sum_of_unit_vectors_has_unit_norm a b) :
  ∥a - b∥ = Real.sqrt 3 :=
sorry

end magnitude_difference_sqrt3_l169_169817


namespace sum_of_possible_values_l169_169414

theorem sum_of_possible_values :
  ∀ x, (|x - 5| - 4 = 3) → x = 12 ∨ x = -2 → (12 + (-2) = 10) :=
by
  sorry

end sum_of_possible_values_l169_169414


namespace greatest_sum_consecutive_integers_lt_500_l169_169989

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169989


namespace number_of_correct_statements_l169_169678

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * Real.sin (2 * x)

def statement_1 : Prop := ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi
def statement_2 : Prop := ∀ x y, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y
def statement_3 : Prop := ∀ y, -Real.pi / 6 ≤ y ∧ y ≤ Real.pi / 3 → -Real.sqrt 3 / 4 ≤ f y ∧ f y ≤ Real.sqrt 3 / 4
def statement_4 : Prop := ∀ x, f x = (1 / 2 * Real.sin (2 * x + Real.pi / 4) - Real.pi / 8)

theorem number_of_correct_statements : 
  (¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4) = true :=
sorry

end number_of_correct_statements_l169_169678


namespace lucia_hiphop_classes_l169_169116

def cost_hiphop_class : Int := 10
def cost_ballet_class : Int := 12
def cost_jazz_class : Int := 8
def num_ballet_classes : Int := 2
def num_jazz_classes : Int := 1
def total_cost : Int := 52

def num_hiphop_classes : Int := (total_cost - (num_ballet_classes * cost_ballet_class + num_jazz_classes * cost_jazz_class)) / cost_hiphop_class

theorem lucia_hiphop_classes : num_hiphop_classes = 2 := by
  sorry

end lucia_hiphop_classes_l169_169116


namespace sets_equal_l169_169105

theorem sets_equal :
  {u : ℤ | ∃ (m n l : ℤ), u = 12 * m + 8 * n + 4 * l} =
  {u : ℤ | ∃ (p q r : ℤ), u = 20 * p + 16 * q + 12 * r} := 
sorry

end sets_equal_l169_169105


namespace digit_150_of_3_div_11_l169_169533

theorem digit_150_of_3_div_11 : 
  let n := 150
  let digits := "27"
  let cycle_length := 2
  let digit_150 := digits[(n % cycle_length)]
  in digit_150 = '7' :=
by {
  sorry
}

end digit_150_of_3_div_11_l169_169533


namespace dodecahedron_interior_diagonals_count_l169_169373

def is_dodecahedron (polyhedron : Type) : Prop :=
  ∃ (faces vertices : ℕ), faces = 12 ∧ vertices = 20 ∧ 
  ∀ (v : polyhedron), (vertex_degree v = 3) ∧ (faces_are_pentagons polyhedron)

def is_interior_diagonal (d : polyhedron → polyhedron → Prop) (polyhedron : Type) : Prop :=
  ∀ (v1 v2 : polyhedron), d v1 v2 → ¬(share_an_edge v1 v2)

theorem dodecahedron_interior_diagonals_count (polyhedron : Type) [is_dodecahedron polyhedron] :
  ∃ (d : polyhedron → polyhedron → Prop), is_interior_diagonal d polyhedron ∧ count_interior_diagonals d polyhedron = 160 := 
sorry

end dodecahedron_interior_diagonals_count_l169_169373


namespace number_of_pairs_l169_169344

theorem number_of_pairs (a b p : ℤ) (h1 : |a - b| + (a + b)^2 = p) (h2 : prime p) : 
  (∃ n, n = 6 ∧ ∃ pairs : list (ℤ × ℤ), (list.filter (λ (ab : ℤ × ℤ), |ab.1 - ab.2| + (ab.1 + ab.2)^2 = p) pairs).length = n) :=
by sorry

end number_of_pairs_l169_169344


namespace slippers_total_cost_l169_169840

def slippers_original_cost : ℝ := 50.00
def slipper_discount_rate : ℝ := 0.10
def embroidery_cost_per_shoe : ℝ := 5.50
def shipping_cost : ℝ := 10.00
def number_of_shoes : ℕ := 2

theorem slippers_total_cost :
  let discount := slippers_original_cost * slipper_discount_rate,
      slippers_cost_after_discount := slippers_original_cost - discount,
      embroidery_cost := embroidery_cost_per_shoe * number_of_shoes,
      total_cost := slippers_cost_after_discount + embroidery_cost + shipping_cost
  in total_cost = 66.00 := by
  sorry

end slippers_total_cost_l169_169840


namespace convex_n_hedral_angle_l169_169509

theorem convex_n_hedral_angle (n : ℕ) 
  (sum_plane_angles : ℝ) (sum_dihedral_angles : ℝ) 
  (h1 : sum_plane_angles = sum_dihedral_angles)
  (h2 : sum_plane_angles < 2 * Real.pi)
  (h3 : sum_dihedral_angles > (n - 2) * Real.pi) :
  n = 3 := 
by 
  sorry

end convex_n_hedral_angle_l169_169509


namespace product_of_ratios_l169_169921

theorem product_of_ratios:
  ∀ (x1 y1 x2 y2 x3 y3 : ℝ),
    (x1^3 - 3 * x1 * y1^2 = 2023) ∧ (y1^3 - 3 * x1^2 * y1 = 2022) →
    (x2^3 - 3 * x2 * y2^2 = 2023) ∧ (y2^3 - 3 * x2^2 * y2 = 2022) →
    (x3^3 - 3 * x3 * y3^2 = 2023) ∧ (y3^3 - 3 * x3^2 * y3 = 2022) →
    (1 - x1/y1) * (1 - x2/y2) * (1 - x3/y3) = 1 / 2023 :=
by
  intros x1 y1 x2 y2 x3 y3
  sorry

end product_of_ratios_l169_169921


namespace find_possible_k_values_l169_169786

noncomputable def possible_values_of_k (n : ℕ) (d : Finset ℕ) (k : ℕ) :=
  (1 ∈ d ∧ n ∈ d ∧ ∀ x ∈ d, x | n) ∧
    (d.card = k) ∧
      (n = (d.toList.nth 1).get_or_else 0 * (d.toList.nth 2).get_or_else 0 +
           (d.toList.nth 1).get_or_else 0 * (d.toList.nth 4).get_or_else 0 +
           (d.toList.nth 2).get_or_else 0 * (d.toList.nth 4).get_or_else 0)

theorem find_possible_k_values (n : ℕ) (d : Finset ℕ) (k : ℕ) :
  possible_values_of_k n d k → (k = 9 ∧ n = 36) ∨ 
                                (k = 8 ∧ 
                                 ∃ p q, p < q ∧ 
                                        Nat.prime (p + q + 1) ∧ 
                                        n = p * q * (1 + p + q)) :=
sorry

end find_possible_k_values_l169_169786


namespace coefficients_not_necessarily_divisible_by_10_l169_169085

theorem coefficients_not_necessarily_divisible_by_10 (a : Fin 2022 → ℤ) :
  (∀ (n : ℕ) (hn : 0 < n), (Polynomial.eval (Polynomial.ofFinFun a) n) % 10 = 0) →
  ¬ (∀ i, a i % 10 = 0) := 
sorry

end coefficients_not_necessarily_divisible_by_10_l169_169085


namespace equilateral_triangle_area_sum_l169_169882

theorem equilateral_triangle_area_sum (r : ℝ) (h : r = 20) :
  let P1P2 := 2 * r in
  let side_length := P1P2 in
  let a := 3 in
  let b := 400 in
  let area := (400 * Real.sqrt 3) in
  (Real.sqrt a + Real.sqrt b = Real.sqrt 3 + Real.sqrt 400) ∧ (a + b = 403) :=
by
  -- Define side length as 40
  let P1P2 := 2 * r
  -- Define a and b
  let a := 3
  let b := 400
  -- Define the area using given formula
  let area := (400 * Real.sqrt 3)
  
  -- Prove the area expression holds
  have h1: Real.sqrt a + Real.sqrt b = Real.sqrt 3 + Real.sqrt 400, from sorry,
  
  -- Also prove the sum a + b is correct
  have h2: a + b = 403, from sorry,
  
  -- Return the conjunction of both
  exact ⟨h1, h2⟩

end equilateral_triangle_area_sum_l169_169882


namespace functional_equation_solution_l169_169109

-- The mathematical problem statement in Lean 4

theorem functional_equation_solution (f : ℝ → ℝ) 
  (h_monotonic : ∀ x y : ℝ, (f x) * (f y) = f (x + y))
  (h_mono : ∀ x y : ℝ, x < y → f x < f y ∨ f x > f y) :
  ∃ a : ℝ, a > 0 ∧ a ≠ 1 ∧ ∀ x : ℝ, f x = a^x :=
sorry

end functional_equation_solution_l169_169109


namespace pyramid_lateral_surface_area_l169_169470

noncomputable def lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) : ℝ :=
  n * S

theorem pyramid_lateral_surface_area (S : ℝ) (n : ℕ) (α : ℝ) (A : ℝ) :
  A = n * S * (Real.cos α) →
  lateral_surface_area S n α = A / (Real.cos α) :=
by
  sorry

end pyramid_lateral_surface_area_l169_169470


namespace abs_diff_squares_105_95_l169_169949

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l169_169949


namespace yellow_peaches_count_l169_169917

variables {Y : ℕ}

theorem yellow_peaches_count :
  ∃ (Y : ℕ), 14 = Y + 8 ∧ Y = 6 :=
by
  use (6)
  refine ⟨rfl, rfl⟩

end yellow_peaches_count_l169_169917


namespace quotient_calculation_l169_169547

theorem quotient_calculation (dividend divisor remainder expected_quotient : ℕ)
  (h₁ : dividend = 166)
  (h₂ : divisor = 18)
  (h₃ : remainder = 4)
  (h₄ : dividend = divisor * expected_quotient + remainder) :
  expected_quotient = 9 :=
by
  sorry

end quotient_calculation_l169_169547


namespace sum_of_first_8_terms_l169_169508

noncomputable def sum_of_geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
a * (1 - r^n) / (1 - r)

theorem sum_of_first_8_terms 
  (a r : ℝ)
  (h₁ : sum_of_geometric_sequence a r 4 = 5)
  (h₂ : sum_of_geometric_sequence a r 12 = 35) :
  sum_of_geometric_sequence a r 8 = 15 := 
sorry

end sum_of_first_8_terms_l169_169508


namespace total_expenditure_is_3000_l169_169560

/-- Define the Hall dimensions -/
def length : ℝ := 20
def width : ℝ := 15
def cost_per_square_meter : ℝ := 10

/-- Statement to prove --/
theorem total_expenditure_is_3000 
  (h_length : length = 20)
  (h_width : width = 15)
  (h_cost : cost_per_square_meter = 10) : 
  length * width * cost_per_square_meter = 3000 :=
sorry

end total_expenditure_is_3000_l169_169560


namespace simplify_expression_l169_169139

theorem simplify_expression (t : ℝ) (t_ne_zero : t ≠ 0) : (t^5 * t^3) / t^4 = t^4 := 
by
  sorry

end simplify_expression_l169_169139


namespace digit_150_of_decimal_3_div_11_l169_169530

theorem digit_150_of_decimal_3_div_11 : 
  (let digits := [2, 7] in digits[(150 % digits.length)]) = 7 :=
by
  sorry

end digit_150_of_decimal_3_div_11_l169_169530


namespace find_a_l169_169358

noncomputable def perpendicular_tangent (a : ℝ) : Prop :=
  let curve := λ x : ℝ, Real.exp (a * x)
  let tangent_slope := (λ x : ℝ, a * Real.exp (a * x))
  let target_point := (0 : ℝ, 1 : ℝ)
  let target_line := (λ x y : ℝ, x + 2 * y + 1 = 0)
  ∃ slope : ℝ, slope * 2 = -1

theorem find_a (a : ℝ) (h_perpendicular : perpendicular_tangent a) : a = 2 :=
  by sorry

end find_a_l169_169358


namespace min_value_2a_plus_b_l169_169016

theorem min_value_2a_plus_b :
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ (3 * a + b = a ^ 2 + a * b) ∧ (∀ a' b', a' > 0 ∧ b' > 0 ∧ (3 * a' + b' = a' ^ 2 + a' * b') → 2 * a' + b' ≥ 3 + 2 * Real.sqrt 2) :=
begin
  sorry
end

end min_value_2a_plus_b_l169_169016


namespace cone_base_radius_l169_169174

theorem cone_base_radius (a r l : ℝ) (h_surface_area : π * (r^2 + r * l) = π * a) (h_lateral_surface : π * l = 2 * π * r) : r = (sqrt (3 * a)) / 3 := by
sorrry

end cone_base_radius_l169_169174


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169975

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169975


namespace problem_solution_l169_169724

noncomputable def a : ℕ → ℕ
| 1     := 1
| 2     := 2
| n     := sorry -- No further values needed for a_i in this problem

noncomputable def b : ℕ → ℕ
| 1     := 2
| n     := if n > 1 then b (n - 1) + 1 else 0 -- Derived from the solution pattern, b_2 = 3, b_3 = 4, etc.

axiom a_b_condition (i j k l : ℕ) (h : i + j = k + l) : a i + b j = a k + b l

theorem problem_solution : (1 / 2013 : ℚ) * ∑ i in finset.range (2013 + 1), (a i + b i) = 2015 :=
by {
  -- Here, we will need the actual proof steps.
  sorry
}

end problem_solution_l169_169724


namespace inequality_range_l169_169490

theorem inequality_range (a : ℝ) : (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by
  sorry

end inequality_range_l169_169490


namespace trigonometric_series_identity_l169_169130

theorem trigonometric_series_identity (n : ℕ) (α : ℝ) 
: 1 + (Finset.range n).sum (λ k, 2 * cos (2 * (k + 1) * α)) = sin ((2 * n + 1) * α) / sin α := 
sorry

end trigonometric_series_identity_l169_169130


namespace parallel_vectors_magnitude_l169_169725

-- Define the vectors a and b
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, 6)

-- Define the parallel condition
def are_parallel (a b : ℝ × ℝ) : Prop :=
  (a.snd * b.fst = a.fst * b.snd)

-- Define the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.fst ^ 2 + v.snd ^ 2)

theorem parallel_vectors_magnitude :
  ∀ x : ℝ, are_parallel vector_a (vector_b x) → magnitude (vector_a.1 - vector_b x.1, vector_a.2 - vector_b x.2) = 2 * Real.sqrt 5 := by
  sorry

end parallel_vectors_magnitude_l169_169725


namespace negation_of_exists_solution_l169_169494

theorem negation_of_exists_solution :
  ¬ (∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔ ∀ c : ℝ, c > 0 → ¬ (∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_exists_solution_l169_169494


namespace value_of_b_l169_169193

theorem value_of_b (a b : ℕ) (q : ℝ)
  (h1 : q = 0.5)
  (h2 : a = 2020)
  (h3 : q = a / b) : b = 4040 := by
  sorry

end value_of_b_l169_169193


namespace BobWins_l169_169610

noncomputable def AliceBobGame := nat → bool

theorem BobWins : ∀ n : ℕ, n > 0 → (∃ f : AliceBobGame, f n = true) :=
by
    intro n
    intro hn
    -- Proof omitted, but exists based on the given problem
    sorry

end BobWins_l169_169610


namespace birds_joined_l169_169572

-- Definitions based on the identified conditions
def initial_birds : ℕ := 3
def initial_storks : ℕ := 2
def total_after_joining : ℕ := 10

-- Theorem statement that follows from the problem setup
theorem birds_joined :
  total_after_joining - (initial_birds + initial_storks) = 5 := by
  sorry

end birds_joined_l169_169572


namespace digit_150_of_3_div_11_l169_169532

theorem digit_150_of_3_div_11 : 
  let n := 150
  let digits := "27"
  let cycle_length := 2
  let digit_150 := digits[(n % cycle_length)]
  in digit_150 = '7' :=
by {
  sorry
}

end digit_150_of_3_div_11_l169_169532


namespace find_t_l169_169876

-- Define the roots and basic properties
variables (a b c : ℝ)
variables (r s t : ℝ)

-- Define conditions from the first cubic equation
def first_eq_roots : Prop :=
  a + b + c = -5 ∧ a * b * c = 13

-- Define conditions from the second cubic equation with shifted roots
def second_eq_roots : Prop :=
  t = -(a * b * c + a * b + a * c + b * c + a + b + c + 1)

-- The theorem stating the value of t
theorem find_t (h₁ : first_eq_roots a b c) (h₂ : second_eq_roots a b c t) : t = -15 :=
sorry

end find_t_l169_169876


namespace altitude_eq_4r_l169_169080

variable (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]

-- We define the geometrical relations and constraints
def AC_eq_BC (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (AC BC : ℝ) : Prop :=
AC = BC

def in_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (incircle_radius r : ℝ) : Prop :=
incircle_radius = r

def ex_circle_radius_eq_r (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D] (excircle_radius r : ℝ) : Prop :=
excircle_radius = r

-- Main theorem to prove
theorem altitude_eq_4r 
  (A B C D : Type) [DecidableEq A] [DecidableEq B] [DecidableEq C] [DecidableEq D]
  (AC BC : ℝ) (r : ℝ)
  (h : ℝ)
  (H1 : AC_eq_BC A B C D AC BC)
  (H2 : in_circle_radius_eq_r A B C D r r)
  (H3 : ex_circle_radius_eq_r A B C D r r) :
  h = 4 * r :=
  sorry

end altitude_eq_4r_l169_169080


namespace education_budget_l169_169175

-- Definitions of the conditions
def total_budget : ℕ := 32 * 10^6  -- 32 million
def policing_budget : ℕ := total_budget / 2
def public_spaces_budget : ℕ := 4 * 10^6  -- 4 million

-- The theorem statement
theorem education_budget :
  total_budget - (policing_budget + public_spaces_budget) = 12 * 10^6 :=
by
  sorry

end education_budget_l169_169175


namespace second_number_less_than_first_l169_169226

theorem second_number_less_than_first (X : ℝ) : 
  let first_num := 0.70 * X,
      second_num := 0.63 * X in
  (second_num / first_num) = 0.9 :=
by 
  sorry

end second_number_less_than_first_l169_169226


namespace probability_of_valid_committee_l169_169879

noncomputable def science_club_probability : ℚ :=
  let total_committees := nat.choose 24 5 in
  let invalid_committees := (nat.choose 12 0) * (nat.choose 12 5) +
                            (nat.choose 12 1) * (nat.choose 12 4) +
                            (nat.choose 12 4) * (nat.choose 12 1) +
                            (nat.choose 12 5) * (nat.choose 12 0) in
  let valid_committees := total_committees - invalid_committees in
  valid_committees / total_committees

theorem probability_of_valid_committee :
  science_club_probability = 4704 / 7084 := by
  sorry

end probability_of_valid_committee_l169_169879


namespace three_digit_number_12_times_sum_of_digits_l169_169549

theorem three_digit_number_12_times_sum_of_digits :
  ∃ A : ℕ, 100 ≤ A ∧ A < 1000 ∧ (∃ S : ℕ, S = (A % 10 + (A % 100) / 10 + A / 100) ∧ A = 12 * S) → A = 108 :=
begin
  sorry
end

end three_digit_number_12_times_sum_of_digits_l169_169549


namespace power_of_square_l169_169629

variable {R : Type*} [CommRing R] (a : R)

theorem power_of_square (a : R) : (3 * a^2)^2 = 9 * a^4 :=
by sorry

end power_of_square_l169_169629


namespace sequence_not_all_prime_l169_169336

theorem sequence_not_all_prime (x0 a b : ℕ) (h : ∀ n, ∃ x : ℕ, x = (iterated (λ x, x * a + b) n x0) ∧ ¬prime x) :
  ∃ n, ¬ prime (iterated (λ x, x * a + b) n x0) :=
begin
  sorry
end

end sequence_not_all_prime_l169_169336


namespace orthocenters_collinear_of_four_lines_l169_169460

-- Define properties about general position of lines and orthocenters
variables {L1 L2 L3 L4 : Type} -- Types for the lines
variables [line L1] [line L2] [line L3] [line L4]

-- General position of the lines: no two lines are parallel and no three lines meet at a single point
def general_position (L1 L2 L3 L4 : Type) [line L1] [line L2] [line L3] [line L4] : Prop :=
  ∀ (a b : Type) [line a] [line b], a ≠ b → ¬parallel a b ∧
  ∀ (a b c : Type) [line a] [line b] [line c], (a ≠ b ∧ b ≠ c ∧ a ≠ c) → ¬concurrent a b c

-- Existing theorems about orthocenters and collinearity are assumed
axiom orthocenter_collinear {a b c : Type} [triangle a b c] : ∃ (l : Type), collinear (orthocenter a b c) (orthocenter a c b) (orthocenter b c a)

-- Main proof problem statement
theorem orthocenters_collinear_of_four_lines 
  (hgp : general_position L1 L2 L3 L4) :
  ∃ l : Type, collinear (orthocenter_triangle L1 L2) (orthocenter_triangle L2 L3)
  (orthocenter_triangle L3 L4) (orthocenter_triangle L4 L1) :=
sorry

end orthocenters_collinear_of_four_lines_l169_169460


namespace reduced_price_is_3_84_l169_169596

noncomputable def reduced_price_per_dozen (original_price : ℝ) (bananas_for_40 : ℕ) : ℝ := 
  let reduced_price := 0.6 * original_price
  let total_bananas := bananas_for_40 + 50
  let price_per_banana := 40 / total_bananas
  12 * price_per_banana

theorem reduced_price_is_3_84 
  (original_price : ℝ) 
  (bananas_for_40 : ℕ) 
  (h₁ : 40 = bananas_for_40 * original_price) 
  (h₂ : bananas_for_40 = 75) 
    : reduced_price_per_dozen original_price bananas_for_40 = 3.84 :=
sorry

end reduced_price_is_3_84_l169_169596


namespace peanuts_weight_correct_l169_169625

def trail_mix_weight : ℝ := 0.4166666666666667
def chocolate_chips_weight : ℝ := 0.16666666666666666
def raisins_weight : ℝ := 0.08333333333333333

-- We are proving that the weight of the peanuts is 0.1666666666666667 pounds.
theorem peanuts_weight_correct :
  let peanuts_weight := trail_mix_weight - (chocolate_chips_weight + raisins_weight) in
  peanuts_weight = 0.1666666666666667 :=
by
  sorry

end peanuts_weight_correct_l169_169625


namespace Cameron_answered_110_questions_today_l169_169276

-- Define the conditions
def num_questions_per_tourist : ℕ := 2
def tours : ℕ := 6
def group_sizes : List ℕ := [6, 11, 8, 5, 9, 7]

def special_questions : List (ℕ → ℕ) := [
  λ q, q,         -- First group: normal questions
  λ q, q,         -- Second group: normal questions
  λ q, if q == 1 then 6 else q * num_questions_per_tourist, -- Third group: one special individual
  λ q, if q == 1 then 8 else if q == 2 then 0 else q * num_questions_per_tourist, -- Fourth group: two special individuals
  λ q, if q ≤ 3 then 4 else if q > 6 then 0 else q * num_questions_per_tourist, -- Fifth group: five special individuals
  λ q, if q > 5 then 3 else q * num_questions_per_tourist -- Sixth group: two special individuals
]

-- Define a function to compute total questions for specific group size and special questions handler
def total_questions_for_group (n : ℕ) (special_handler : ℕ → ℕ) : ℕ :=
(List.range n).sum (λ i, special_handler (i + 1))

def total_questions : ℕ :=
(List.zipWith total_questions_for_group group_sizes special_questions).sum

-- Assertion of the final result
theorem Cameron_answered_110_questions_today : total_questions = 110 := by
  sorry

end Cameron_answered_110_questions_today_l169_169276


namespace elena_flower_petals_l169_169305

theorem elena_flower_petals :
  let lilies := 8 in
  let tulips := 5 in
  let petals_per_lily := 6 in
  let petals_per_tulip := 3 in
  (lilies * petals_per_lily + tulips * petals_per_tulip) = 63 :=
by
  let lilies := 8
  let tulips := 5
  let petals_per_lily := 6
  let petals_per_tulip := 3
  show lilies * petals_per_lily + tulips * petals_per_tulip = 63
  calc
    lilies * petals_per_lily + tulips * petals_per_tulip
      = 8 * 6 + 5 * 3 : rfl
  ... = 48 + 15 : rfl
  ... = 63 : rfl

end elena_flower_petals_l169_169305


namespace profit_percentage_l169_169592

theorem profit_percentage (CP SP : ℝ) (h1 : CP = 500) (h2 : SP = 650) : 
  (SP - CP) / CP * 100 = 30 :=
by
  sorry

end profit_percentage_l169_169592


namespace evaluate_expression_l169_169650

theorem evaluate_expression :
  (floor ((ceil (121 / 36 : ℚ) + 19 / 5) : ℚ)) = 7 :=
by
  sorry

end evaluate_expression_l169_169650


namespace vector_AB_complex_l169_169347

-- Define the complex numbers corresponding to vectors
def OA : ℂ := 1 - 2 * complex.i
def OB : ℂ := 2 + complex.i

-- State the proof problem
theorem vector_AB_complex :
  OB - OA = 1 + 3 * complex.i :=
by
  sorry

end vector_AB_complex_l169_169347


namespace ratio_of_x_y_eq_one_l169_169126

theorem ratio_of_x_y_eq_one (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : (|((λ n, (((2 * n) % 2 : ℕ) * x + (((2 * n + 1) % 2 : ℕ) * y) - ((2 * n + 2) % 2) * (x + y))) (2019) - y)) - x| = 
       |((λ n, (((2 * n) % 2 : ℕ) * y + (((2 * n + 1) % 2 : ℕ) * x) - ((2 * n + 2) % 2) * (x + y))) (2019) - y)) - x|))
: x / y = 1 := 
sorry

-- This Lean statement represents the mathematical proof problem translated into Lean syntax
-- Conditions:
-- x > 0 (h1)
-- y > 0 (h2)
-- The given equation with 2019 nested absolute value signs on each side (encoded in h3)
-- To prove: x / y = 1

end ratio_of_x_y_eq_one_l169_169126


namespace remainder_of_sum_14_x_l169_169102

theorem remainder_of_sum_14_x (x : ℤ) (hx : 0 < x) (hmod : 7 * x ≡ 1 [MOD 31]) : (14 + x) % 31 = 23 := 
by
  sorry

end remainder_of_sum_14_x_l169_169102


namespace greatest_sum_of_consecutive_integers_l169_169958

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169958


namespace find_BP_l169_169228

theorem find_BP
  (A B C D P : Type) 
  (AP PC BP DP : ℝ)
  (hAP : AP = 8) 
  (hPC : PC = 1)
  (hBD : BD = 6)
  (hBP_less_DP : BP < DP) 
  (hPower_of_Point : AP * PC = BP * DP)
  : BP = 2 := 
by {
  sorry
}

end find_BP_l169_169228


namespace max_value_of_quadratic_l169_169157

-- Define the given quadratic function
def f (x : ℝ) : ℝ := -x^2 + 2 * x + 1

-- Statement asserting that the function f(x) has a maximum value of 2
theorem max_value_of_quadratic : ∃ x : ℝ, ∀ y : ℝ, f(y) ≤ 2 :=
by
  -- Placeholder for proof
  sorry

end max_value_of_quadratic_l169_169157


namespace distance_from_point_to_plane_l169_169339

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def vectorSub (p q : Point3D) : Point3D :=
  { x := p.x - q.x, y := p.y - q.y, z := p.z - q.z }

def dotProduct (u v : Point3D) : ℝ :=
  u.x * v.x + u.y * v.y + u.z * v.z

def vectorMagnitude (v : Point3D) : ℝ :=
  Real.sqrt (v.x ^ 2 + v.y ^ 2 + v.z ^ 2)

def distancePointToPlane (M N : Point3D) (n : Point3D) : ℝ :=
  let MO := vectorSub M N 
  (dotProduct MO n) / vectorMagnitude n

def M : Point3D := { x := -1, y := 1, z := -2 }

def N : Point3D := { x := 0, y := 0, z := 0 }

def n : Point3D := { x := 1, y := -2, z := 2 }

theorem distance_from_point_to_plane :
  distancePointToPlane M N n = 7 / 3 := by
  sorry

end distance_from_point_to_plane_l169_169339


namespace sample_size_calculation_l169_169582

theorem sample_size_calculation (n : ℕ) (ratio_A_B_C q_A q_B q_C : ℕ) 
  (ratio_condition : ratio_A_B_C = 2 ∧ ratio_A_B_C * q_A = 2 ∧ ratio_A_B_C * q_B = 3 ∧ ratio_A_B_C * q_C = 5)
  (sample_A_units : q_A = 16) : n = 80 :=
sorry

end sample_size_calculation_l169_169582


namespace cat_catches_mouse_thm_l169_169577

noncomputable def cat_catches_mouse : Prop :=
  ∃ (c m : ℝ → ℝ × ℝ)
    (h_cat_initial : c 0 = (1, 1))
    (h_mouse_initial : m 0 = (0, 0))
    (h_cat_diff : ∀ t ∈ (Icc 0 1), differentiable_at ℝ (λ t, c t))
    (h_mouse_diff : ∀ t ∈ (Icc 0 1), differentiable_at ℝ (λ t, m t))
    (h_cat_speed : ∀ t ∈ (Icc 0 1), (fderiv ℝ (λ t, c t) t).norm = √2)
    (h_mouse_speed : ∀ t ∈ (Icc 0 1), (fderiv ℝ (λ t, m t) t).norm = 1)
    (h_cat_nonneg : ∀ t ∈ (Icc 0 1), (c t).fst ≥ 0 ∧ (c t).snd ≥ 0)
    (h_mouse_nonneg : ∀ t ∈ (Icc 0 1), (m t).fst ≥ 0 ∧ (m t).snd ≥ 0)
    (h_cat_info : ∀ t ∈ (Icc 0 1), ∃ τ ∈ (Icc 0 1), c τ = m τ), 
  ∃ τ ∈ (Icc 0 1), c τ = m τ

theorem cat_catches_mouse_thm : cat_catches_mouse :=
sorry

end cat_catches_mouse_thm_l169_169577


namespace interval_of_monotonic_increase_l169_169893

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.exp x
noncomputable def y' (x : ℝ) : ℝ := 2 * x * Real.exp x + x^2 * Real.exp x

theorem interval_of_monotonic_increase :
  ∀ x : ℝ, (y' x ≥ 0 ↔ (x ∈ Set.Ici 0 ∨ x ∈ Set.Iic (-2))) :=
by
  sorry

end interval_of_monotonic_increase_l169_169893


namespace double_integral_iterated_l169_169310

variable (f : ℝ → ℝ → ℝ)

def region_D (x y : ℝ) : Prop :=
  -2 ≤ x ∧ x ≤ 2 ∧ -1 ≤ y ∧ y ≤ 3 - x^2

theorem double_integral_iterated {f : ℝ → ℝ → ℝ} :
  (∫ x in -2..2, ∫ y in -1..(3 - x^2), f x y) = 
  (∫ y in -1..3, ∫ x in - ∞..√(3 - y), f x y) + 
  (∫ y in -1..3, ∫ x in -√(3 - y)..∞, f x y) :=
by
  sorry

end double_integral_iterated_l169_169310


namespace rainwater_collection_tuesday_l169_169427

-- Let us define the conditions as variables
variables (gallons_per_inch : ℕ) 
          (rain_monday : ℕ) 
          (price_per_gallon : ℝ) 
          (total_money : ℝ)

-- Let the values of the conditions be assigned as given in the problem
def James_gallons_per_inch := 15
def James_rain_monday := 4
def James_price_per_gallon := 1.2
def James_total_money := 126

-- Define the collected rainwater on Tuesday
def rain_tuesday (gallons_per_inch price_per_gallon : ℕ) (rain_monday : ℕ) (total_money : ℝ) : ℕ :=
  (total_money - ((rain_monday * gallons_per_inch) * price_per_gallon)) / (gallons_per_inch * price_per_gallon)

theorem rainwater_collection_tuesday :
  rain_tuesday James_gallons_per_inch James_price_per_gallon James_rain_monday James_total_money = 3 :=
sorry

end rainwater_collection_tuesday_l169_169427


namespace merchant_import_tax_l169_169552

def value (total_value : ℝ) : ℝ := if total_value > 1000 then total_value - 1000 else 0
def tax_rate : ℝ := 0.07
def import_tax (total_value : ℝ) : ℝ := tax_rate * value total_value

theorem merchant_import_tax (total_value : ℝ) (h : total_value = 2570) : import_tax total_value = 109.90 :=
by {
  rw h,
  unfold import_tax value,
  norm_num,
  }

end merchant_import_tax_l169_169552


namespace apex_angle_of_fourth_cone_l169_169515

-- Define the conditions as Lean definitions
def identical_cones (A : Point) (cones : List Cone) : Prop :=
  ∀ cone ∈ cones, cone.apex_angle = π / 3 ∧ 
                  ∀ otherCone ∈ cones, cone ≠ otherCone → cones_touch_externally cone otherCone

def cone_touches_internally (inner outer : Cone) : Prop :=
  ∀ point ∈ inner.base_circle, is_point_inside_cone point outer ∧ inner.apex = outer.apex

-- Define the Lean theorem statement
theorem apex_angle_of_fourth_cone (A : Point) (cones : List Cone) (fourth_cone : Cone)
  (h1 : identical_cones A cones)
  (h2 : ∀ cone ∈ cones, cone_touches_internally cone fourth_cone)
  (h3 : ∀ cone ∈ cones, cone.apex = fourth_cone.apex)  :
  fourth_cone.apex_angle = π / 3 + 2 * arcsin (1 / sqrt 3) := 
begin
  sorry, -- Proof omitted as requested
end

end apex_angle_of_fourth_cone_l169_169515


namespace farmer_planting_problem_l169_169583

theorem farmer_planting_problem (total_acres : ℕ) (flax_acres : ℕ) (sunflower_acres : ℕ)
  (h1 : total_acres = 240)
  (h2 : flax_acres = 80)
  (h3 : sunflower_acres = total_acres - flax_acres) :
  sunflower_acres - flax_acres = 80 := by
  sorry

end farmer_planting_problem_l169_169583


namespace tangent_line_at_M_find_f_one_plus_f_prime_one_l169_169912

noncomputable def f (x : ℝ) : ℝ := sorry

theorem tangent_line_at_M :
  tangent_line f 1 = λ x, 3 * x - 2 := sorry

theorem find_f_one_plus_f_prime_one :
  f 1 + (deriv f 1) = 4 := 
by
  have h_tangent : tangent_line f 1 = λ x, 3 * x - 2 := sorry,
  have f_prime_1 : deriv f 1 = 3 := sorry,
  have f_1 : f 1 = 1 := sorry,
  show 1 + 3 = 4,
  linarith

end tangent_line_at_M_find_f_one_plus_f_prime_one_l169_169912


namespace triangle_ABC_area_l169_169906

theorem triangle_ABC_area :
  let r := 3
  let AB := 2 * r
  let BD := 4
  let AD := AB + BD
  let ED := 6
  let AE := Real.sqrt (AD^2 + ED^2)
  let EC := 27 / Real.sqrt 136
  let AC := AE - EC
  let BC := Real.sqrt (AB^2 - AC^2)
  (1 / 2) * AC * BC = 3052 / 136 :=
by
  let r := 3
  let AB := 2 * r
  let BD := 4
  let AD := AB + BD
  let ED := 6
  let AE := Real.sqrt (AD^2 + ED^2)
  let EC := 27 / Real.sqrt 136
  let AC := AE - EC
  let BC := Real.sqrt (AB^2 - AC^2)
  show (1 / 2) * AC * BC = 3052 / 136
  sorry

end triangle_ABC_area_l169_169906


namespace number_of_correct_statements_l169_169679

noncomputable def f (x : ℝ) : ℝ := 1 / 2 * Real.sin (2 * x)

def statement_1 : Prop := ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = 2 * Real.pi
def statement_2 : Prop := ∀ x y, -Real.pi / 4 ≤ x ∧ x ≤ y ∧ y ≤ Real.pi / 4 → f x ≤ f y
def statement_3 : Prop := ∀ y, -Real.pi / 6 ≤ y ∧ y ≤ Real.pi / 3 → -Real.sqrt 3 / 4 ≤ f y ∧ f y ≤ Real.sqrt 3 / 4
def statement_4 : Prop := ∀ x, f x = (1 / 2 * Real.sin (2 * x + Real.pi / 4) - Real.pi / 8)

theorem number_of_correct_statements : 
  (¬ statement_1 ∧ statement_2 ∧ ¬ statement_3 ∧ ¬ statement_4) = true :=
sorry

end number_of_correct_statements_l169_169679


namespace largest_n_possible_l169_169902

noncomputable def max_n (a : ℕ → ℤ) : ℕ :=
  let n := 16 in
  if ∀ i < n, (λ j, ∑ k in finRange 7, a (i + k)) i < 0 ∧ (λ j, ∑ k in finRange 11, a (i + k)) i > 0
  then n
  else sorry -- Placeholder for further implementation

-- Finally, state the theorem
theorem largest_n_possible (a : ℕ → ℤ) : max_n a = 16 :=
sorry

end largest_n_possible_l169_169902


namespace percentage_democrats_voting_X_l169_169065

variables (k : ℝ) (P : ℝ) (V : ℝ)
variables (R D : ℝ)
variables (Votes_X : ℝ)
variables (Percentage_X_Win : ℝ)

-- Conditions
def registered_voters_sum : Prop := R + D = V
def republicans_to_democrats_ratio : Prop := R = 3 * k ∧ D = 2 * k
def percentage_republicans_X : Prop := 0.7 * R = 0.7 * 3 * k
def percentage_democrats_X : Prop := Votes_X = 0.7 * 3 * k + (P / 100) * 2 * k
def expected_win_percentage : Prop := Percentage_X_Win = 0.539999999999999853 * V
def candidates_X_win : Prop := 3.9999999999999853 = (Percentage_X_Win - 0.5) * 100 -- This implies winning by 3.9999999999999853 percent.

-- Theorem to prove
theorem percentage_democrats_voting_X :
  registered_voters_sum ∧
  republicans_to_democrats_ratio ∧
  percentage_republicans_X ∧
  percentage_democrats_X ∧
  expected_win_percentage ∧
  candidates_X_win →
  P = 30 := by
  sorry

end percentage_democrats_voting_X_l169_169065


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169977

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169977


namespace num_ways_for_A_B_diff_l169_169236

def num_ways_at_least_one_diff (n m k : ℕ) : ℕ :=
 ∑ i in (finset.powerset (finset.range n)).filter (λ s, s.card = k), 
   (finset.powerset (finset.univ.filter (λ x, x ∉ s))).card

theorem num_ways_for_A_B_diff : 
  num_ways_at_least_one_diff 4 2 2 = 30 := 
begin 
  -- problem-specific lean code logic goes here 
  sorry 
end

end num_ways_for_A_B_diff_l169_169236


namespace number_of_roosters_l169_169920

-- Define the basic constants and variables involved.
def total_chickens : ℕ := 9000
def parts_roosters : ℕ := 2
def parts_hens : ℕ := 1
def parts_chicks : ℕ := 3
def total_parts : ℕ := parts_roosters + parts_hens + parts_chicks

-- The key statement to be proven.
theorem number_of_roosters :
  let chickens_per_part := total_chickens / total_parts in
  let roosters := chickens_per_part * parts_roosters in
  roosters = 3000 := by
  sorry

end number_of_roosters_l169_169920


namespace total_rent_is_correct_l169_169234

noncomputable def total_rent_field 
  (A_cows : Nat) (A_months : Nat) (B_cows : Nat) (B_months : Nat)
  (C_cows : Nat) (C_months : Nat) (D_cows : Nat) (D_months : Nat)
  (A_rent : Nat) : Nat :=
  let A_cow_months := A_cows * A_months
  let B_cow_months := B_cows * B_months
  let C_cow_months := C_cows * C_months
  let D_cow_months := D_cows * D_months
  let total_cow_months := A_cow_months + B_cow_months + C_cow_months + D_cow_months
  let rent_per_cow_month := A_rent / A_cow_months
  rent_per_cow_month * total_cow_months

theorem total_rent_is_correct:
  ∀ (A_cows A_months B_cows B_months C_cows C_months D_cows D_months A_rent : Nat),
    A_cows = 24 →
    A_months = 3 →
    B_cows = 10 →
    B_months = 5 →
    C_cows = 35 →
    C_months = 4 →
    D_cows = 21 →
    D_months = 3 →
    A_rent = 720 →
    total_rent_field A_cows A_months B_cows B_months C_cows C_months D_cows D_months A_rent = 3250 :=
by {
  intros,
  sorry
}

end total_rent_is_correct_l169_169234


namespace calc_f_neg_2016_l169_169361

noncomputable def f : ℝ → ℝ
| x := if x > 2 then f (x + 5)
       else if -2 ≤ x ∧ x ≤ 2 then Real.exp x
       else f (-x)

theorem calc_f_neg_2016 : f (-2016) = Real.exp 1 := 
sorry

end calc_f_neg_2016_l169_169361


namespace minimal_stone_removal_l169_169752

-- Formalize the conditions
def chessboard := fin 7 × fin 8
def stones_placed (c : chessboard) : Prop := true -- initially, every square has a stone

def adjacent (a b : chessboard) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨  -- horizontal
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1)) ∨  -- vertical
  ((a.1 = b.1 + 1 ∨ a.1 = b.1 - 1) ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) -- diagonal

def no_five_adjacent_stones (S : set chessboard) : Prop :=
  ∀ c1 c2 c3 c4 c5 : chessboard,
  (c1 ∈ S ∧ c2 ∈ S ∧ c3 ∈ S ∧ c4 ∈ S ∧ c5 ∈ S) →
  ¬ (adjacent c1 c2 ∧ adjacent c2 c3 ∧ adjacent c3 c4 ∧ adjacent c4 c5)

def minimal_removals (S : set chessboard) : ℕ :=
  56 - S.card

-- Problem statement
theorem minimal_stone_removal :
  ∃ (S : set chessboard), no_five_adjacent_stones S ∧ minimal_removals S = 10 :=
sorry

end minimal_stone_removal_l169_169752


namespace problem1_problem2_problem3_l169_169715

-- Definition of conditions and vectors a and b
def vector_a (x : ℝ) : ℝ × ℝ := (1, x)
def vector_b (x : ℝ) : ℝ × ℝ := (2 * x + 3, -x)

-- Absolute value of a vector difference
def vector_norm (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- Proof statement for problem 1: Parallel vectors
theorem problem1 (x : ℝ) (h : vector_a x = vector_b x) : vector_norm (vector_a x - vector_b x) = 2 ∨ vector_norm (vector_a x - vector_b x) = 2 * Real.sqrt 5 := sorry

-- Proof statement for problem 2: Acute angle
theorem problem2 (x : ℝ) (hx : 2 * x + 3 - x ^ 2 > 0) : -1 < x ∧ x < 3 := sorry

-- Unit vector orthogonal to a
def is_perpendicular (v1 v2 : ℝ × ℝ) : Bool := v1.1 * v2.1 + v1.2 * v2.2 = 0
def is_unit_vector (v : ℝ × ℝ) : Bool := Real.sqrt (v.1 ^ 2 + v.2 ^ 2) = 1

theorem problem3 (x : ℝ) (hx : Real.sqrt (1 + x ^ 2) = 2) :
  ∃ (m n : ℝ), is_perpendicular (vector_a x) (m, n) ∧ is_unit_vector (m, n) := sorry

end problem1_problem2_problem3_l169_169715


namespace at_least_two_roots_l169_169722

noncomputable def Quadratic (b c : ℝ) : ℝ → ℝ :=
  λ x, x^2 + b * x + c

theorem at_least_two_roots
  (b1 b2 c1 c2 : ℝ)
  (h : ∃ x, (Quadratic b1 c1 x + Quadratic b2 c2 x + Quadratic ((b1 + b2) / 2) ((c1 + c2) / 2) x) = 0) :
  (∃ x, Quadratic b1 c1 x = 0 ∨ Quadratic b2 c2 x = 0) :=
begin
  sorry
end

end at_least_two_roots_l169_169722


namespace range_of_a_l169_169391

-- Given the conditions as variables
variable (a x : ℝ)

-- Definition
def no_solution (a : ℝ) : Prop :=
  ∀ x : ℝ, ¬ (a + 2 < x ∧ x < 3a - 2)

-- Theorem to prove
theorem range_of_a (a : ℝ) : no_solution a → a ≤ 2 :=
by 
  sorry

end range_of_a_l169_169391


namespace proof_part_II_l169_169030

def f (x : ℝ) : ℝ := abs (x + 1)

def M : set ℝ := { x | x < -1 ∨ x > 1 }

theorem proof_part_II (a b : ℝ) (ha : a ∈ M) (hb : b ∈ M) : 
  f (a * b) > f (a) - f (-b) := by
sorry

end proof_part_II_l169_169030


namespace slippers_total_cost_l169_169843

theorem slippers_total_cost (original_price discount_rate embroidery_cost_per_shoe shipping_cost : ℝ)
                            (num_shoes : ℕ)
                            (h_original_price : original_price = 50.00)
                            (h_discount_rate : discount_rate = 0.10)
                            (h_embroidery_cost_per_shoe : embroidery_cost_per_shoe = 5.50)
                            (h_shipping_cost : shipping_cost = 10.00)
                            (h_num_shoes : num_shoes = 2) :
  let discounted_price := original_price * (1 - discount_rate)
      total_embroidery_cost := embroidery_cost_per_shoe * num_shoes
      total_cost := discounted_price + total_embroidery_cost + shipping_cost
  in total_cost = 66.00 := by
  sorry

end slippers_total_cost_l169_169843


namespace distinct_books_permutation_count_l169_169297

theorem distinct_books_permutation_count : (∃ n : ℕ, n = 6) → (∏ i in finset.range 6 + 1, i) = 720 :=
by
  sorry

end distinct_books_permutation_count_l169_169297


namespace number_of_factors_P_l169_169824

/-- The number of positive integer factors of the expression 
  \( P = 35^5 + 5 \times 35^4 + 10 \times 35^3 + 10 \times 35^2 + 5 \times 35 + 1 \)
  is 121. -/
theorem number_of_factors_P :
  let P := 35^5 + 5 * 35^4 + 10 * 35^3 + 10 * 35^2 + 5 * 35 + 1 
  in nat.num_divisors P = 121 :=
by
  sorry

end number_of_factors_P_l169_169824


namespace T8_plus_T9_is_minus_199_l169_169716

def a_n (n : ℕ) : ℤ := 2^n + n
def b_n (n : ℕ) : ℤ := if n = 1 then 3 else -3 * 2^(n - 1) - 2 * n + 1
def c_n (n : ℕ) : ℤ := if n % 2 = 1 then a_n n else b_n n

def T_n (n : ℕ) : ℤ := ∑ i in Finset.range n, c_n (i + 1) -- sum of first n terms, i + 1 for 1-based indexing

theorem T8_plus_T9_is_minus_199 : T_n 8 + T_n 9 = -199 :=
sorry

end T8_plus_T9_is_minus_199_l169_169716


namespace greatest_sum_consecutive_integers_lt_500_l169_169990

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169990


namespace cricket_average_score_l169_169484

theorem cricket_average_score (A : ℝ)
    (h1 : 3 * 30 = 90)
    (h2 : 5 * 26 = 130) :
    2 * A + 90 = 130 → A = 20 :=
by
  intros h
  linarith

end cricket_average_score_l169_169484


namespace log_half_decreasing_l169_169734

section

variables {m n : ℝ} (h : 0 < m) (h' : m < n)

theorem log_half_decreasing (h : 0 < m) (h' : m < n) : log (1/2) m > log (1/2) n :=
sorry

end

end log_half_decreasing_l169_169734


namespace fuel_consumption_l169_169304

open Real

theorem fuel_consumption (initial_fuel : ℝ) (final_fuel : ℝ) (distance_covered : ℝ) (consumption_rate : ℝ) (fuel_left : ℝ) (x : ℝ) :
  initial_fuel = 60 ∧ final_fuel = 50 ∧ distance_covered = 100 ∧ 
  consumption_rate = (initial_fuel - final_fuel) / distance_covered ∧ consumption_rate = 0.1 ∧ 
  fuel_left = initial_fuel - consumption_rate * x ∧ x = 260 →
  fuel_left = 34 :=
by
  sorry

end fuel_consumption_l169_169304


namespace fixed_point_P_circle_eq_D_E_F_right_angle_triangle_exists_l169_169685

-- Condition (1): The line always passes through point P.
def line_passing_through_P (k : ℝ) : Prop :=
  ∀x y, (k-1)*x - 2*y + 5 - 3*k = 0 → (x, y) = (3, 1)

-- Condition (2): The circle passes through (4,0) and (3,1), and its center is on the line x - 2y + 1 = 0.
def circle_passing_conditions (D E F : ℝ) : Prop :=
  (4*4 + 0*0 + D*4 + E*0 + F = 0) ∧
  (3*3 + 1*1 + D*3 + E*1 + F = 0) ∧
  (-D/2) - 2*(-E/2) + 1 = 0

-- Condition (3): Point P is one endpoint of the diameter, the other endpoint is point Q.
def circle_diameter_point (x0 y0 : ℝ) : Prop :=
  (3 + x0 = 14) ∧ (1 + y0 = 8) ∧ (x0, y0) = (11, 7)

-- Conclusion (1): The coordinates of the fixed point P are (3, 1).
theorem fixed_point_P : (∃ k : ℝ, line_passing_through_P k) ↔ (3, 1) := sorry

-- Conclusion (2): The equation of circle C is x^2 + y^2 - 14x - 8y + 40 = 0.
theorem circle_eq_D_E_F : circle_passing_conditions (-14) (-8) 40 ↔
                          ∀ x y, x^2 + y^2 - 14*x - 8*y + 40 = 0 := sorry

-- Conclusion (3): Possible values of m for right-angled triangle PMQ are 5 and 65/3.
theorem right_angle_triangle_exists (m : ℝ) : 
  circle_diameter_point 11 7 →
  ((m = 5) ∨ (m = 65 / 3)) :=
by
  intros
  admit


end fixed_point_P_circle_eq_D_E_F_right_angle_triangle_exists_l169_169685


namespace fraction_inequality_l169_169285

theorem fraction_inequality (x : ℝ) (h : -3 ≤ x ∧ x ≤ 1) :
  3 * x + 8 ≥ 3 * (5 - 2 * x) → (7 / 9) ≤ x ∧ x ≤ 1 :=
by {
  intros,
  sorry
}

end fraction_inequality_l169_169285


namespace percentage_of_loss_is_correct_l169_169590

def CP : ℝ := 600
def SP : ℝ := 480
def loss (cp sp : ℝ) : ℝ := cp - sp
def percentage_of_loss (cp sp : ℝ) : ℝ := (loss cp sp / cp) * 100

theorem percentage_of_loss_is_correct :
  percentage_of_loss CP SP = 20 :=
by
  sorry

end percentage_of_loss_is_correct_l169_169590


namespace imaginary_part_of_z_is_negative_one_l169_169355

-- Definition of complex number and the given condition
def z (x : ℂ) : ℂ := (3 + x) / (1 + x)

-- Theorem stating that the imaginary part of z is -1
theorem imaginary_part_of_z_is_negative_one : (z Complex.I).im = -1 :=
sorry

end imaginary_part_of_z_is_negative_one_l169_169355


namespace part1_part2_l169_169697

open Real

variables (α : ℝ) (A : (ℝ × ℝ)) (B : (ℝ × ℝ)) (C : (ℝ × ℝ))

def points_coordinates : Prop :=
A = (3, 0) ∧ B = (0, 3) ∧ C = (cos α, sin α) ∧ π / 2 < α ∧ α < 3 * π / 2

theorem part1 (h : points_coordinates α A B C) (h1 : dist (3, 0) (cos α, sin α) = dist (0, 3) (cos α, sin α)) : 
  α = 5 * π / 4 :=
sorry

theorem part2 (h : points_coordinates α A B C) (h2 : ((cos α - 3) * cos α + (sin α) * (sin α - 3)) = -1) : 
  (2 * sin α * sin α + sin (2 * α)) / (1 + tan α) = -5 / 9 :=
sorry

end part1_part2_l169_169697


namespace complex_expression_value_l169_169100

-- Define the conditions
def z := 1 + Complex.i
def conjugate_z := Complex.conj z

-- Define the main problem
theorem complex_expression_value : (z / Complex.i + Complex.i * conjugate_z) = 2 := 
by
-- The proof will go here
sorry

end complex_expression_value_l169_169100


namespace maximum_value_of_expression_l169_169319

open BigOperators

noncomputable def f (x: ℝ) : ℝ := 10^x - 100^x

theorem maximum_value_of_expression :
  ∃ x : ℝ, f(x) = (1 / 4) ∧ 
  ∀ y : ℝ, f(y) ≤ (1 / 4) :=
sorry

end maximum_value_of_expression_l169_169319


namespace sequence_property_l169_169691

theorem sequence_property :
  ∃ a1 : ℕ, 
    (∀ n < 100000, nat.even (nat.rec_on n a1 (λ _ an, nat.floor ((3 * an) / 2) + 1))) ∧ 
    nat.odd (nat.rec_on 100000 a1 (λ _ an, nat.floor ((3 * an) / 2) + 1)) :=
sorry

end sequence_property_l169_169691


namespace weight_of_b_l169_169881

noncomputable def weights (a b c d e : ℝ) : Prop :=
  (a + b + c + d + e = 300) ∧
  (a + b + c = 165) ∧
  (b + c + d = 174) ∧
  (c + d + e = 186)

theorem weight_of_b {a b c d e : ℝ} (h : weights a b c d e) : b = 114 :=
by
  cases h with h1 h234
  cases h234 with h2 h34
  cases h34 with h3 h4
  sorry

end weight_of_b_l169_169881


namespace find_n_l169_169068

-- Definition of the interior angle of a regular hexagon
def interior_angle_hexagon : ℝ := 120

-- Definition of the exterior angle of a regular n-gon
def exterior_angle_ngon (n : ℕ) : ℝ := 360 / n

-- The target value of n we want to prove
def target_n : ℕ := 15

-- Proof statement: Prove that n = 15 given the conditions
theorem find_n (n : ℕ) (h1 : interior_angle_hexagon = 5 * exterior_angle_ngon n) : 
  n = target_n :=
sorry

end find_n_l169_169068


namespace find_k_for_tangent_circle_l169_169008

theorem find_k_for_tangent_circle :
  (∃ l1 l2 : ℝ → ℝ, 
     (l1 0 = 7/3 ∧ l1 7 = 0) ∧ 
     (l2 2 = 1 ∧ ∃ k : ℝ, l2 3 = k + 1) ∧
     (∀ x y : ℝ, x ≠ y → ((l1 x - l1 y)/(x - y)) * ((l2 x - l2 y)/(x - y)) = -1) ∧
     ∃ t : ℝ, t ∈ {0, 1} → ∃ c : ℝ, c belongs to L1 ∩ L2 ∧ c = tangent_point)) →
    ∃ k : ℝ, k = 3 :=
sorry

end find_k_for_tangent_circle_l169_169008


namespace product_of_two_consecutive_not_n_powered_product_of_n_consecutive_not_n_powered_l169_169791

-- Conditions from Part (1) and Part (2)
def is_n_powered_number (n p : ℕ) : Prop :=
  ∃ p : ℕ, p > 0 ∧ p^n = n

-- Part (1)
theorem product_of_two_consecutive_not_n_powered (n x : ℕ) (h1 : n ≥ 2) :
  ¬ is_n_powered_number n (x * (x + 1)) :=
sorry

-- Part (2)
theorem product_of_n_consecutive_not_n_powered (n k : ℕ) (h1 : n ≥ 2) :
  ¬ is_n_powered_number n (∏ i in finset.range n, k + i) :=
sorry

end product_of_two_consecutive_not_n_powered_product_of_n_consecutive_not_n_powered_l169_169791


namespace marvin_birthday_on_monday_2022_l169_169120

-- Definition of leap year
def isLeapYear (y : ℕ) : Prop := (y % 4 = 0) ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)

-- Condition 1: Marvin's birthday on January 31, 2019, was on a Thursday
def birthday_on_2019_is_thursday : true := trivial

-- Function to calculate the day advancement for leap and non-leap years
def day_of_week_progression (start_year target_year : ℕ) : ℕ :=
let years := list.range' (start_year + 1) (target_year - start_year) in
years.foldl (λ acc y, acc + if isLeapYear y then 2 else 1) 0

-- Proof problem statement
theorem marvin_birthday_on_monday_2022 (year : ℕ) (h : year = 2022)
  (initial_day_of_week : ℕ) (LeapDayIncrement : ℕ := 2) (NonLeapDayIncrement : ℕ := 1)
  (initial_day_of_week_eq_thursday : initial_day_of_week = 4) :
  (initial_day_of_week + (day_of_week_progression 2019 year)) % 7 = 1 := by
  rw [h, initial_day_of_week_eq_thursday]
  sorry

end marvin_birthday_on_monday_2022_l169_169120


namespace greatest_sum_of_consecutive_integers_l169_169957

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169957


namespace number_is_composite_all_n_l169_169162

def is_composite (n : ℕ) : Prop := ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ a * b = n

def construct_number (n : ℕ) : ℕ :=
  200 * 10^n + 88 * ((10^n - 1) / 9) + 21

theorem number_is_composite_all_n (n : ℕ) :
  let N := construct_number n in
  is_composite N := 
by
  sorry

end number_is_composite_all_n_l169_169162


namespace john_paid_more_than_lily_by_three_l169_169837

-- Given conditions:
def p_slices : ℕ := 12
def p_total : ℝ := 12
def p_pepperoni_cost : ℝ := 3
def p_pepperoni_slices : ℝ := (1 / 3) * p_slices
def john_slices : ℝ := p_pepperoni_slices + 2
def lily_slices : ℝ := p_slices - john_slices
def john_payment : ℝ := (john_slices * ((p_total + p_pepperoni_cost) / p_slices))
def lily_payment : ℝ := lily_slices * (p_total / p_slices)

-- Lean 4 statement to prove:
theorem john_paid_more_than_lily_by_three :
  john_payment - lily_payment = 3 :=
by
  sorry

end john_paid_more_than_lily_by_three_l169_169837


namespace asymptotes_of_hyperbola_l169_169038

def sequence (a : ℕ → ℝ) := ∀ n : ℕ, n > 0 → a n = 1 / (n * (n + 1))

def sum_sequence (S : ℕ → ℝ) := ∀ n : ℕ, S n = 1 - 1 / (n + 1)

theorem asymptotes_of_hyperbola :
  (∀ n : ℕ, n > 0 → (sum_sequence (λ k, 1 - 1 / (k + 1)) n = 9 / 10)) →
  (∀ n : ℕ, n > 0 → (sequence (λ k, 1 / (k * (k + 1))) n = 1 / (n * (n + 1)))) →
  let n := 9 in
  ∀ n' : ℕ, n' = 10 →
  ∀ m' : ℕ, m' = 9 →
  ∃ y, ∀ x : ℝ, y = x * (±(3 * real.sqrt 10 / 10)) :=
begin
  sorry
end

end asymptotes_of_hyperbola_l169_169038


namespace primitive_roots_at_least_half_l169_169435

open Nat

noncomputable def totient (n : ℕ) : ℕ :=
  (n.factorization.map fun _ => 1).prod * (1 - 1 / n.factorization.keys.prod)

theorem primitive_roots_at_least_half (p : ℕ) (k : ℕ) 
  (hp : p = 4 * k + 1) (hpp : Nat.Prime p) :
  ∃ g : ℕ, is_primitive_root g p → p ∃ g' : ℕ, g ≠ (p - g') ∧ is_primitive_root g' p :=
  sorry

end primitive_roots_at_least_half_l169_169435


namespace overlap_area_after_folding_l169_169595

theorem overlap_area_after_folding (w : ℝ) : 
  let length := 2 * w,
      section_width := length / 4,
      overlap_length := w / 2,
      overlapping_area := w * (w / 2) in
  overlapping_area = (w^2 / 2) :=
by
  sorry

end overlap_area_after_folding_l169_169595


namespace statues_ratio_l169_169728

theorem statues_ratio :
  let y1 := 4                  -- Number of statues after first year.
  let y2 := 4 * y1             -- Number of statues after second year.
  let y3 := (y2 + 12) - 3      -- Number of statues after third year.
  let y4 := 31                 -- Number of statues after fourth year.
  let added_fourth_year := y4 - y3  -- Statues added in the fourth year.
  let broken_third_year := 3        -- Statues broken in the third year.
  added_fourth_year / broken_third_year = 2 :=
by
  sorry

end statues_ratio_l169_169728


namespace count_multiples_4_6_not_5_9_l169_169047

/-- The number of integers between 1 and 500 that are multiples of both 4 and 6 but not of either 5 or 9 is 22. -/
theorem count_multiples_4_6_not_5_9 :
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22 :=
by
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  show count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22
  sorry

end count_multiples_4_6_not_5_9_l169_169047


namespace greatest_m_value_l169_169502

noncomputable def find_greatest_m : ℝ := sorry

theorem greatest_m_value :
  ∃ m : ℝ, 
    (∀ x, x^2 - m * x + 8 = 0 → x ∈ {x | ∃ y, y^2 = 116}) ∧ 
    m = 2 * Real.sqrt 29 :=
sorry

end greatest_m_value_l169_169502


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169974

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169974


namespace digit_150_of_3_div_11_l169_169538

theorem digit_150_of_3_div_11 :
  let rep := "27"
  ∃ seq : ℕ → ℕ,
  (∀ n, seq (2 * n) = 2) ∧ (∀ n, seq (2 * n + 1) = 7) ∧
  150 % 2 = 0 ∧ seq (150 - 1) = 7 :=
by
  let rep := "27"
  use (λ n : ℕ => if n % 2 = 0 then 2 else 7)
  split
  { intro n
    exact rfl }
  { intro n
    exact rfl }
  { exact rfl }
  { exact rfl }

end digit_150_of_3_div_11_l169_169538


namespace alan_needs_17_votes_l169_169181

-- Variables and conditions
variables (total_votes : ℕ) (sally_votes : ℕ) (katie_votes : ℕ) (alan_votes : ℕ)
constants (total_students : 130) (sally_v : 24) (katie_v : 29) (alan_v : 37)

-- Definition that Alan needs at least 17 more votes to ensure he finishes with the most votes
theorem alan_needs_17_votes (total_votes = 130) (sally_votes = 24) (katie_votes = 29) (alan_votes = 37) : alan_votes + 17 > katie_votes + (total_votes - sally_votes - katie_votes - alan_votes - 17) :=
-- Proof step placeholder
sorry

end alan_needs_17_votes_l169_169181


namespace no_four_positive_integers_l169_169423

theorem no_four_positive_integers :
  ¬ ∃ (n1 n2 n3 n4 : ℕ), 0 < n1 ∧ 0 < n2 ∧ 0 < n3 ∧ 0 < n4 ∧
    (∀ i j, i ≠ j → (([n1, n2, n3, n4].nth i).get_or_else 0 * ([n1, n2, n3, n4].nth j).get_or_else 0 + 2002).is_square) :=
by
  sorry

end no_four_positive_integers_l169_169423


namespace solve_eq_l169_169748

theorem solve_eq (x : ℝ) (h : 2 - 1 / (2 - x) = 1 / (2 - x)) : x = 1 := 
sorry

end solve_eq_l169_169748


namespace exists_four_numbers_divisible_l169_169082

def unique_digits (n : ℕ) : Prop :=
  (n.digits 10).nodup

def contains_all_digits (a b c d : ℕ) : Prop :=
  let used_digits := (a.digits 10) ++ (b.digits 10) ++ (c.digits 10) ++ (d.digits 10)
  List.perm used_digits [1, 2, 3, 4, 5, 6, 7, 8, 9]

theorem exists_four_numbers_divisible :
  ∃ a b c d : ℕ, (a % 16 = 0 ∧ b % 17 = 0 ∧ c % 18 = 0 ∧ d % 19 = 0) ∧ contains_all_digits a b c d :=
sorry

end exists_four_numbers_divisible_l169_169082


namespace total_pies_bigger_event_l169_169845

def pies_last_week := 16.5
def apple_pies_last_week := 14.25
def cherry_pies_last_week := 12.75

def pecan_multiplier := 4.3
def apple_multiplier := 3.5
def cherry_multiplier := 5.7

theorem total_pies_bigger_event :
  (pies_last_week * pecan_multiplier) + 
  (apple_pies_last_week * apple_multiplier) + 
  (cherry_pies_last_week * cherry_multiplier) = 193.5 :=
by
  sorry

end total_pies_bigger_event_l169_169845


namespace card_arrangement_possible_l169_169851

-- Define the set of integers from 1 to 25.
def card_numbers : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}

-- Predicate to check if two numbers share at least one prime factor.
def shares_prime_factor (a b : ℕ) : Prop :=
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ a ∧ p ∣ b

-- Target set size we want to prove it can be arranged satisfying the condition.
def target_size : ℕ := 20

-- Main theorem to prove.
theorem card_arrangement_possible :
  ∃ (l : List ℕ), l.length = target_size ∧ (∀ (i : ℕ), i < l.length - 1 → shares_prime_factor (l.nth_le i _) (l.nth_le (i + 1) _)) :=
by
  sorry

end card_arrangement_possible_l169_169851


namespace total_cost_is_correct_l169_169518

-- Definitions based on the given conditions
def cost_of_goat := 400
def num_goats := 3
def num_llamas := 2 * num_goats
def cost_increase_factor := 1.5

-- The cost of one llama is 50% more than the cost of one goat
def cost_of_llama := cost_of_goat * cost_increase_factor

-- The total cost is the sum of the costs of goats and llamas
def total_cost := (num_goats * cost_of_goat) + (num_llamas * cost_of_llama)

-- The theorem to prove the total cost
theorem total_cost_is_correct : total_cost = 4800 :=
by
  -- Proof goes here, but for now we use sorry
  sorry

end total_cost_is_correct_l169_169518


namespace find_compound_interest_principal_l169_169910

noncomputable def SI (P R T: ℝ) := (P * R * T) / 100
noncomputable def CI (P R T: ℝ) := P * (1 + R / 100)^T - P

theorem find_compound_interest_principal :
  let SI_amount := 3500.000000000004
  let SI_years := 2
  let SI_rate := 6
  let CI_years := 2
  let CI_rate := 10
  let SI_value := SI SI_amount SI_rate SI_years
  let P := 4000
  (SI_value = (CI P CI_rate CI_years) / 2) →
  P = 4000 :=
by
  intros
  sorry

end find_compound_interest_principal_l169_169910


namespace general_term_sum_first_11_terms_l169_169504

variable (a : Nat → Int)
variable (a_1 : Int := 21)
variable (a_10 : Int := 3)
variable (d : Int := (a_10 - a_1)/9)
variable (n : Nat)

-- Condition: a is an arithmetic progression such that a 1 = 21 and a 10 = 3
axiom arithmetic_progression (a : Nat → Int) (a_1 : Int) (a_10 : Int) : Prop :=
∀ n, a n = a_1 + (n - 1) * d

theorem general_term (h : arithmetic_progression a a_1 a_10) :
  a n = -2 * n + 23 :=
sorry

theorem sum_first_11_terms (h : arithmetic_progression a a_1 a_10) :
  (finset.sum (finset.range 11) (λ n, a (n + 1))) = 121 :=
sorry

end general_term_sum_first_11_terms_l169_169504


namespace segment_intersection_l169_169168

open Set

noncomputable
def A : ℕ → Set ℝ := sorry
noncomputable
def B : ℕ → Set ℝ := sorry

theorem segment_intersection :
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ 1977 → ∃ l : ℝ, l ∈ A k ∧ l ∈ Icc (k - 1) (k + 1)) →
  (A 1977 ∩ B 1).Nonempty →
  (A 1 ∩ B 1977).Nonempty →
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ 1977 ∧ (A k ∩ B k).Nonempty := 
sorry

end segment_intersection_l169_169168


namespace abs_diff_squares_105_95_l169_169951

def abs_diff_squares (a b : ℕ) : ℕ :=
  abs ((a ^ 2) - (b ^ 2))

theorem abs_diff_squares_105_95 : abs_diff_squares 105 95 = 2000 :=
by {
  let a := 105;
  let b := 95;
  have h1 : abs ((a ^ 2) - (b ^ 2)) = abs_diff_squares a b,
  simp [abs_diff_squares],
  sorry
}

end abs_diff_squares_105_95_l169_169951


namespace distance_midpoints_l169_169123

-- Define the initial coordinates of points A and B
variables (a b c d : ℝ)

-- Define the initial midpoint M of points A and B
def M : ℝ × ℝ := ((a + c) / 2, (b + d) / 2)

-- Define the new coordinates of points A and B after movement
def A' : ℝ × ℝ := (a + 3, b + 5)
def B' : ℝ × ℝ := (c - 5, d - 3)

-- Define the new midpoint M' of the moved points A' and B'
def M' : ℝ × ℝ := ((a + 3 + (c - 5)) / 2, (b + 5 + (d - 3)) / 2)

-- Prove the distance between M and M' is √2
theorem distance_midpoints :
  let M' := ((a + c - 2) / 2, (b + d + 2) / 2) in
  let distance := (λ (x y : ℝ × ℝ), 
                   (real.sqrt ((x.1 - y.1) ^ 2 + (x.2 - y.2) ^ 2))) in
  distance M M' = real.sqrt 2 :=
by
  -- Add the proof here
  sorry

end distance_midpoints_l169_169123


namespace submerged_spherical_segment_l169_169599

-- Define the problem conditions.
variables (r m : ℝ) (V_sec V_s : ℝ)
noncomputable def V_sec := (2 * π * m^3) / 3
noncomputable def V_s := (π * m^2 * (3 * r - m)) / 3

-- State the problem in Lean 4 as a theorem with the given conditions.
theorem submerged_spherical_segment (r m : ℝ) 
  (h1 : V_s = V_sec / 2) 
  (h2 : V_sec = (2 * π * m^3) / 3) 
  (h3 : V_s = (π * m^2 * (3 * r - m)) / 3) : 
  m = (r / 2) * (3 - Real.sqrt 5) :=
by sorry

end submerged_spherical_segment_l169_169599


namespace parabola_through_points_l169_169247

theorem parabola_through_points :
  ∃ (b c : ℝ),
    (∀ x : ℝ, (-2, -16) ∈ set_of (λ pt, pt.2 = x^2 + b * x + c)) ∧
    (∀ x : ℝ, (2, 8) ∈ set_of (λ pt, pt.2 = x^2 + b * x + c)) ∧
    (∀ x : ℝ, (4, 36) ∈ set_of (λ pt, pt.2 = x^2 + b * x + c)) ∧
    b = 6 ∧ c = -8 :=
begin
  use [6, -8],
  split,
  { intro x,
    rw [set.mem_set_of_eq],
    simp [sq, add_mul],
    norm_num,
    sorry },
  split,
  { intro x,
    rw [set.mem_set_of_eq],
    simp [sq, add_mul],
    norm_num,
    sorry },
  split,
  { intro x,
    rw [set.mem_set_of_eq],
    simp [sq, add_mul],
    norm_num,
    sorry }
end

end parabola_through_points_l169_169247


namespace sum_of_prime_f_values_l169_169672

def is_prime (p : ℕ) : Prop :=
  p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

def f (n : ℕ) : ℕ := n^4 - 400 * n^2 + 841

theorem sum_of_prime_f_values : 
  ∑ n in ((finset.range 21).filter (λ n, is_prime (f n))), f n = 1283 :=
sorry

end sum_of_prime_f_values_l169_169672


namespace greatest_sum_of_consecutive_integers_l169_169959

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169959


namespace max_sum_diagonals_of_rhombus_l169_169689

theorem max_sum_diagonals_of_rhombus (a b : ℝ) (h1 : 0 ≤ a ∧ a ≤ 6) (h2 : 6 ≤ b ∧ b ≥ 0) 
(h_eq1 : ∀ c, c = 2 * 5 * real.sin (a / 2))
(h_eq2 : ∀ d, d = 2 * 5 * real.cos (a / 2)):
  10 * (real.sin (a / 2) + real.cos (a / 2)) = 14 := 
sorry 

end max_sum_diagonals_of_rhombus_l169_169689


namespace fraction_irreducibility_l169_169330

theorem fraction_irreducibility (n : ℕ) : ¬ reducible (n ^ 2 + 2, n * (n + 1)) ↔ (∃ k : ℕ, n = 6 * k + 1) ∨ (∃ k : ℕ, n = 6 * k + 3) := 
by 
  sorry

end fraction_irreducibility_l169_169330


namespace greatest_sum_consecutive_integers_lt_500_l169_169993

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169993


namespace math_proof_problem_l169_169002

variables (f : ℝ → ℝ)

-- Conditions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x ≤ y → y ≤ b → f x ≤ f y

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) - f (1 + x) + 2 * x = 0

-- The proof problem
theorem math_proof_problem
  (h1 : is_odd_function f)
  (h2 : differentiable ℝ f)
  (h3 : functional_equation f)
  (h4 : is_monotonically_increasing_on f 0 1) :
  f 2 = 2 ∧ f 2024 = 2024 ∧ deriv f 2023 = 1 :=
sorry

end math_proof_problem_l169_169002


namespace abs_diff_squares_105_95_l169_169934

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l169_169934


namespace range_of_a_l169_169713

theorem range_of_a (a : ℝ) :
  (∀ θ ∈ Icc (0 : ℝ) (π/2),
    √2 * (2 * a + 3) * Real.cos (θ - π/4) 
      + 6 / (Real.sin θ + Real.cos θ)
      - 2 * Real.sin 2 * θ < 3 * a + 6) →
  a > 3 := sorry

end range_of_a_l169_169713


namespace minimum_value_a_2b_3c_l169_169699

theorem minimum_value_a_2b_3c (a b c : ℝ)
  (h : ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3) :
  (a + 2*b - 3*c) = -4 :=
sorry

end minimum_value_a_2b_3c_l169_169699


namespace tire_swap_distance_l169_169894

theorem tire_swap_distance : ∃ x : ℕ, 
  (1 - x / 11000) * 9000 = (1 - x / 9000) * 11000 ∧ x = 4950 := 
by
  sorry

end tire_swap_distance_l169_169894


namespace total_fish_l169_169456

theorem total_fish (fish_lilly fish_rosy : ℕ) (hl : fish_lilly = 10) (hr : fish_rosy = 14) :
  fish_lilly + fish_rosy = 24 := 
by 
  sorry

end total_fish_l169_169456


namespace definite_integral_sin2_cos6_l169_169627

theorem definite_integral_sin2_cos6 :
  ∫ x in 0..(2 * Real.pi), (Real.sin x)^2 * (Real.cos x)^6 = (3 * Real.pi) / 32 :=
by
  sorry

end definite_integral_sin2_cos6_l169_169627


namespace cost_of_limes_after_30_days_l169_169785

def lime_juice_per_mocktail : ℝ := 1  -- tablespoons per mocktail
def days : ℕ := 30  -- number of days
def lime_juice_per_lime : ℝ := 2  -- tablespoons per lime
def limes_per_dollar : ℝ := 3  -- limes per dollar

theorem cost_of_limes_after_30_days : 
  let total_lime_juice := (lime_juice_per_mocktail * days),
      number_of_limes  := (total_lime_juice / lime_juice_per_lime),
      total_cost       := (number_of_limes / limes_per_dollar)
  in total_cost = 5 :=
by
  sorry

end cost_of_limes_after_30_days_l169_169785


namespace equivalency_statement_1_equivalency_statement_2_equivalency_statement_3_equivalency_statement_4_l169_169633

variables (p q : Prop) (a : ℝ)

-- Given conditions
axiom p_true : p := sorry
axiom q_false : ¬q := sorry

axiom x_condition : ∀ x : ℝ, x^2 + x + 2 ≥ 0 := sorry

axiom necessary_condition_skew_lines : 
  ∀ lines_a lines_b, (¬(∃ x, lines_a = lines_b x)) → (are_skew lines_a lines_b ↔ true) := sorry

axiom sufficiency_parallel_lines : 
  ((x + a*y + 6 = 0) ∧ ((a-2)*x + 3*y + 2*a = 0)) ↔ (a = -1) := sorry

-- Proofs to ensure the statements' equivalency
theorem equivalency_statement_1 : p ∨ q := by
  exact or.inl p_true

theorem equivalency_statement_2 : ¬∃ x : ℝ, x^2 + x + 2 < 0 := by
  exact λ h, match h with ⟨x, hx⟩ => lt_irrefl (x^2 + x + 2) (lt_of_lt_of_le hx (x_condition x))

theorem equivalency_statement_3 : 
  ∀ lines_a lines_b, (¬(∃ x, lines_a = lines_b x)) → (are_skew lines_a lines_b ↔ true) := 
  by exact necessary_condition_skew_lines

theorem equivalency_statement_4 : ((x + a*y + 6 = 0) ∧ ((a-2)*x + 3*y + 2a = 0)) ↔ (a = -1) := 
  by exact sufficiency_parallel_lines

end equivalency_statement_1_equivalency_statement_2_equivalency_statement_3_equivalency_statement_4_l169_169633


namespace solve_sailboat_speed_l169_169634

variable {v : ℝ}
variable (v_12 : ℝ) (d : ℝ) (t_diff : ℝ)

def sailboat_speed_24_square_foot : Prop :=
  (d / v + t_diff = d / v_12) → v = 50

theorem solve_sailboat_speed (h : sailboat_speed_24_square_foot v_12 d t_diff) : v = 50 :=
by {
  have h1 : d / v + t_diff = d / v_12 := by exact h,
  
  -- Assuming the values from the conditions provided
  have hd : d = 200 := rfl,
  have hv_12 : v_12 = 20 := rfl,
  have h_t_diff : t_diff = 6 := rfl,

  calc
    d / v + t_diff
      = 200 / v + 6     : by rw [hd, h_t_diff]
  ... = 200 / 20        : by rw hv_12
  ... = 10              : by norm_num
  ... = 200 / 4         : by norm_num
  ... = d / (200 / 4)   : by rw [hd] at *
  ... = 50              : by norm_num
}

end solve_sailboat_speed_l169_169634


namespace eventual_ones_l169_169280

theorem eventual_ones (k : ℕ) (a : Fin (2^k) → ℤ)
  (h : ∀ i, a i = 1 ∨ a i = -1)
  (transform : ∀ n : ℕ, ∃ an : Fin (2^k) → ℤ, ∀ i, an i = a i * a ((i + 1) % 2^k)) :
  ∃ n : ℕ, ∀ i, (transform^[n] a) i = 1 := sorry

end eventual_ones_l169_169280


namespace dihedral_angle_between_planes_l169_169523

noncomputable def plane1 := { x y z : ℝ // 5 * x - 4 * y + z - 1 = 0 }
noncomputable def plane2 := { x y z : ℝ // 10 * x + 2 * y + 4 * z - 7 = 0 }

noncomputable def normal_vector1 : ℝ × ℝ × ℝ := (5, -4, 1)
noncomputable def normal_vector2 : ℝ × ℝ × ℝ := (10, 2, 4)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2 + a.3 * b.3

noncomputable def magnitude (a : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt (a.1 ^ 2 + a.2 ^ 2 + a.3 ^ 2)

noncomputable def cos_angle_between_normals : ℝ :=
  dot_product normal_vector1 normal_vector2 / (magnitude normal_vector1 * magnitude normal_vector2)

theorem dihedral_angle_between_planes : Real.arccos (cos_angle_between_normals) = 49 :=
sorry

end dihedral_angle_between_planes_l169_169523


namespace parabola_has_fixed_point_l169_169687

noncomputable def parabola_fixed_point : Prop :=
  ∃ a b : ℝ, 
  (∀ x y : ℝ, y^2 = 4 * x ↔ x ≥ 0) ∧  -- Standard equation of parabola
  (∀ m n : ℝ, (∃ y : ℝ, y^2 - 4 * m * y + 4 * n = 0) → (my = x + (n = -2))) ∧
  (∃ x : ℝ, (x = 2) ∧ (my = x - 2))

theorem parabola_has_fixed_point :
  parabola_fixed_point := sorry

end parabola_has_fixed_point_l169_169687


namespace math_problem_l169_169419

-- Define the curves C1 and C2 in rectangular and polar coordinates
def C1_parametric (α : ℝ) : ℝ × ℝ :=
  (3 * Real.cos α, Real.sin α)

def C1 (x y : ℝ) : Prop :=
  (x / 3) ^ 2 + y ^ 2 = 1

def C2_polar (ρ θ : ℝ) : Prop :=
  ρ ^ 2 - 8 * ρ * Real.sin θ + 15 = 0

def C2_rectangular (x y : ℝ) : Prop :=
  (x - 4) ^ 2 + y ^ 2 = 1

-- Define the maximum distance between points on the curves
def PQ_max_distance : ℝ :=
  8

-- Statement of the problem to be proven
theorem math_problem :
  (∀ (α : ℝ), C1_parametric α = (3 * Real.cos α, Real.sin α)) ∧
  (∀ (ρ θ : ℝ), C2_polar ρ θ ↔ C2_rectangular (ρ * Real.cos θ) (ρ * Real.sin θ)) ∧
  (∃ P Q : ℝ × ℝ, C1 P.1 P.2 ∧ C2_rectangular Q.1 Q.2 ∧ (Real.sqrt ((P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2)) = PQ_max_distance) :=
by
  -- Rewriting the proof statement hereby including "sorry" as no proof is required
  sorry

end math_problem_l169_169419


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169986

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169986


namespace proof_problem_l169_169032

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log x - a * (x - 1)

noncomputable def g (x : ℝ) : ℝ := exp x

def is_monotonic_interval (a : ℝ) : Prop :=
  (∀ x, 0 < x → x < 1 → deriv (λ x, f x a) x > 0) ∧ 
  (∀ x, 1 < x → deriv (λ x, f x a) x < 0)

def range_of_f (a : ℝ) : list (ℝ × ℝ) :=
if a ≤ 1/exp 1 then
  [(0, 1 + a - a * exp 1)]
else if 1/exp 1 < a ∧ a ≤ 1/(exp 1 - 1) then
  [(0, -log a - 1 + a)]
else if 1/(exp 1 - 1) < a ∧ a < 1 then
  [(1 + a - a * exp 1, -log a - 1 + a)]
else [(1 + a - a * exp 1, 0)]

def tangent_slope_condition (a : ℝ) : Prop :=
∃ x₀, f 1 a = (1 / exp x₀) ∧ deriv g x₀ = 1 / deriv (λ x, f x a) 1

theorem proof_problem (a : ℝ) (h_a : 0 < a) :
  is_monotonic_interval 1 ∧ 
  (range_of_f a = [(0, 1 + a - a * exp 1)] ∨ 
   range_of_f a = [(0, -log a - 1 + a)] ∨ 
   range_of_f a = [(1 + a - a * exp 1, -log a - 1 + a)] ∨ 
   range_of_f a = [(1 + a - a * exp 1, 0)]) ∧ 
  (tangent_slope_condition a → (exp 1 - 1) / exp 1 < a ∧ a < (exp 2 - 1) / exp 1) :=
begin
  sorry
end

end proof_problem_l169_169032


namespace complex_modulus_l169_169740

theorem complex_modulus (z : ℂ) (h : (1+2*complex.I)*z = 1-complex.I) : abs (conj z) = sqrt 10 / 5 :=
by
  sorry

end complex_modulus_l169_169740


namespace brian_gallons_usage_l169_169626

/-
Brian’s car gets 20 miles per gallon. 
On his last trip, he traveled 60 miles. 
How many gallons of gas did he use?
-/

theorem brian_gallons_usage (miles_per_gallon : ℝ) (total_miles : ℝ) (gallons_used : ℝ) 
    (h1 : miles_per_gallon = 20) 
    (h2 : total_miles = 60) 
    (h3 : gallons_used = total_miles / miles_per_gallon) : 
    gallons_used = 3 := 
by
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end brian_gallons_usage_l169_169626


namespace general_formula_calculate_a2016_is_2016_element_of_sequence_l169_169421

-- Define the sequence conditions
def sequence (n : ℕ) : ℤ := 4 * n - 2

-- State the conditions
lemma initial_conditions :
  sequence 1 = 2 ∧ sequence 17 = 66 :=
by
  split;
  simp [sequence]

-- Prove the general formula
theorem general_formula (n : ℕ) :
  ∃ k b, (sequence n = k * n + b ∧
  sequence 1 = 2 ∧ sequence 17 = 66 ∧
  sequence n = 4 * n - 2 ∧ k = 4 ∧ b = -2 ) :=
by
  use [4, -2];
  simp [sequence];
  sorry

-- Calculate \( a_{2016} \)
theorem calculate_a2016 :
  sequence 2016 = 8062 :=
by
  simp [sequence]

-- Check whether 2016 is an element of the sequence
theorem is_2016_element_of_sequence :
  ∃ n : ℤ, sequence n = 2016 ↔ 2016 ∉ sequence :=
by
  intro n;
  simp [sequence];
  norm_num;
  sorry

end general_formula_calculate_a2016_is_2016_element_of_sequence_l169_169421


namespace first_day_exceeds_500_bacteria_l169_169397

-- Conditions: The colony starts with 5 bacteria and triples every day.
def bacteria_count (n : ℕ) : ℕ := 5 * 3^n

-- Proof problem: We need to find the smallest n such that bacteria_count n > 500
theorem first_day_exceeds_500_bacteria :
  ∃ n : ℕ, (bacteria_count n > 500) ∧ ∀ m : ℕ, (0 ≤ m < n → bacteria_count m ≤ 500) := by
  sorry

end first_day_exceeds_500_bacteria_l169_169397


namespace sum_of_remainders_l169_169215

theorem sum_of_remainders
  (a b c : ℕ)
  (h₁ : a % 36 = 15)
  (h₂ : b % 36 = 22)
  (h₃ : c % 36 = 9) :
  (a + b + c) % 36 = 10 :=
by
  sorry

end sum_of_remainders_l169_169215


namespace jerry_showers_l169_169303

variable (water_allowance : ℕ) (drinking_cooking : ℕ) (water_per_shower : ℕ) (pool_length : ℕ) 
  (pool_width : ℕ) (pool_height : ℕ) (gallons_per_cubic_foot : ℕ)

/-- Jerry can take 15 showers in July given the conditions. -/
theorem jerry_showers :
  water_allowance = 1000 →
  drinking_cooking = 100 →
  water_per_shower = 20 →
  pool_length = 10 →
  pool_width = 10 →
  pool_height = 6 →
  gallons_per_cubic_foot = 1 →
  (water_allowance - (drinking_cooking + (pool_length * pool_width * pool_height) * gallons_per_cubic_foot)) / water_per_shower = 15 :=
by
  intros h_water_allowance h_drinking_cooking h_water_per_shower h_pool_length h_pool_width h_pool_height h_gallons_per_cubic_foot
  sorry

end jerry_showers_l169_169303


namespace turn_off_all_bulbs_l169_169135

/-- Several light bulbs are lit on a dashboard. There are several buttons such that pressing a button changes the state of
the bulbs it is connected to. It is known that for any set of bulbs, there is a button connected to an odd number of bulbs
in this set. We need to prove that by pressing the buttons, it is possible to turn off all the bulbs. --/
theorem turn_off_all_bulbs (n : ℕ) (bulbs : Fin n → Bool) (buttons : Fin n → Finset (Fin n)) 
  (h : ∀ s : Finset (Fin n), ∃ b : Fin n, (buttons b ∩ s).card % 2 = 1) :
  ∃ presses : Fin n → Bool, bulbs = λ i, bulbs i = (∑ j in (buttons j).filter presses, 1) % 2 = 0 :=
sorry

end turn_off_all_bulbs_l169_169135


namespace abs_diff_squares_105_95_l169_169936

theorem abs_diff_squares_105_95 : abs ((105 : ℤ)^2 - (95 : ℤ)^2) = 2000 := 
by
  sorry

end abs_diff_squares_105_95_l169_169936


namespace regular_heptagon_interior_angle_l169_169199

theorem regular_heptagon_interior_angle :
  ∀ (S : Type) [decidable_instance S] [fintype S], ∀ (polygon : set S), is_regular polygon → card polygon = 7 → 
    (sum_of_interior_angles polygon / 7 = 128.57) :=
by
  intros S dec inst polygon h_reg h_card
  sorry

end regular_heptagon_interior_angle_l169_169199


namespace solution_exists_l169_169382

noncomputable def problem_statement : Prop :=
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ (2 * log y x + 2 * log x y = 8) ∧ (x * y = 256) ∧ ((x + y) / 2 = 16)

theorem solution_exists : problem_statement :=
by {
  sorry
}

end solution_exists_l169_169382


namespace symmetry_center_of_tangent_shift_l169_169022

noncomputable def symmetry_center (k : ℤ) : ℝ × ℝ := 
  (k * (π / 2) + π / 3, 0)

theorem symmetry_center_of_tangent_shift (k : ℤ) :
  let y := λ x : ℝ, Real.tan (x - π / 3)
  in (∃ c : ℝ × ℝ, c = symmetry_center k) :=
by {
  let y := λ x : ℝ, Real.tan (x - π / 3),
  use (symmetry_center k),
  sorry
}

end symmetry_center_of_tangent_shift_l169_169022


namespace triangle_problem_l169_169186

noncomputable def AD_value (AB BC CA : ℕ) (R : ℝ) (m n : ℕ) : ℝ :=
  (m : ℝ) + real.sqrt (n : ℝ)

theorem triangle_problem
  (AB BC CA : ℕ)
  (hAB : AB = 43)
  (hBC : BC = 13)
  (hCA : CA = 48)
  (AD m n : ℕ)
  (hAD : AD = 6 * real.sqrt 43)
  : ∃ m n : ℕ, ⌊AD_value AB BC CA 43 m n⌋ = 12 :=
by 
  sorry

end triangle_problem_l169_169186


namespace lean_problem_l169_169062

section Geometry

variables {A B C P M N : Type*} [MetricSpace A] [MetricSpace B] 
          [MetricSpace C] [MetricSpace P] [MetricSpace M] [MetricSpace N]

-- Defining the conditions for the geometrical problem
variables (angle_ABC angle_ACB angle_PCB : ℝ)
variables (on_angle_bisector : ∀ {x : ℝ}, x = angle_ABC / 2)
variables (BP_intersects_AC_at_M : ∃ M : Type*, is_intersection (line_through B P) (line_through A C) M)
variables (CP_intersects_AB_at_N : ∃ N : Type*, is_intersection (line_through C P) (line_through A B) N)

-- The conditions given in the problem
def conditions : Prop :=
  angle_ABC = 40 ∧
  angle_ACB = 30 ∧
  (on_angle_bisector angle_ABC) ∧
  angle_PCB = 10 ∧
  BP_intersects_AC_at_M ∧
  CP_intersects_AB_at_N

-- Problem to prove
theorem lean_problem (h : conditions) : distance P M = distance A N :=
sorry

end Geometry

end lean_problem_l169_169062


namespace determine_f_l169_169436

theorem determine_f:
  ∀ (f : ℕ → ℕ), (∀ n m : ℕ, n ≠ m → f (n) * f (m) = f ((n*m)^2021)) → 
  (∀ x : ℕ, 2 ≤ x → f(x) = 1) :=
sorry

end determine_f_l169_169436


namespace distributive_like_laws_false_l169_169290

section
  variables (a b x y z : ℕ) -- You can adjust the type of x, y, z, a, b as needed ℕ, ℚ, etc.
  
  def hash (a b : ℕ) : ℕ := a * b + 1

  theorem distributive_like_laws_false :
    (∀ x y z : ℕ, hash x (y + z) ≠ hash x y + hash x z) ∧
    (∀ x y z : ℕ, x + hash y z ≠ hash (x + y) (x + z)) ∧
    (∀ x y z : ℕ, hash x (hash y z) ≠ hash (hash x y) (hash x z)) :=
  by
    apply And.intro
    { intros x y z
      simp [hash]
      sorry }
    apply And.intro
    { intros x y z
      simp [hash]
      sorry }
    { intros x y z
      simp [hash]
      sorry }
end

end distributive_like_laws_false_l169_169290


namespace exists_polynomial_no_integer_zero_and_divisible_l169_169869

noncomputable def P (x : ℤ) : ℤ := (x^2 + 1) * (x^2 - 2) * (x^2 + 2) * (x^2 + 7)

theorem exists_polynomial_no_integer_zero_and_divisible (n : ℕ) (hn : 0 < n) :
  (¬ ∃ x : ℤ, P x = 0) ∧ (∃ x : ℤ, n ∣ P x) :=
by
  sorry

end exists_polynomial_no_integer_zero_and_divisible_l169_169869


namespace value_of_x_l169_169643

/-
Given the following conditions:
  x = a + 7,
  a = b + 9,
  b = c + 15,
  c = d + 25,
  d = 60,
Prove that x = 116.
-/

theorem value_of_x (a b c d x : ℤ) 
    (h1 : x = a + 7)
    (h2 : a = b + 9)
    (h3 : b = c + 15)
    (h4 : c = d + 25)
    (h5 : d = 60) : x = 116 := 
  sorry

end value_of_x_l169_169643


namespace angle_measure_l169_169545

theorem angle_measure (x : ℝ) (h1 : 180 - x = 6 * (90 - x)) : x = 72 := by
  sorry

end angle_measure_l169_169545


namespace root_difference_geom_prog_l169_169295

theorem root_difference_geom_prog
  (x1 x2 x3 : ℝ)
  (h1 : 8 * x1^3 - 22 * x1^2 + 15 * x1 - 2 = 0)
  (h2 : 8 * x2^3 - 22 * x2^2 + 15 * x2 - 2 = 0)
  (h3 : 8 * x3^3 - 22 * x3^2 + 15 * x3 - 2 = 0)
  (geom_prog : ∃ (a r : ℝ), x1 = a / r ∧ x2 = a ∧ x3 = a * r) :
  |x3 - x1| = 33 / 14 :=
by
  sorry

end root_difference_geom_prog_l169_169295


namespace cat_count_l169_169459

def initial_cats : ℕ := 2
def female_kittens : ℕ := 3
def male_kittens : ℕ := 2
def total_kittens : ℕ := female_kittens + male_kittens
def total_cats : ℕ := initial_cats + total_kittens

theorem cat_count : total_cats = 7 := by
  unfold total_cats
  unfold initial_cats total_kittens
  unfold female_kittens male_kittens
  rfl

end cat_count_l169_169459


namespace vector_diff_magnitude_l169_169800

variables {ℝ_vector: Type*} [inner_product_space ℝ ℝ_vector]

open_locale real_inner_product_space

def is_unit_vector (v : ℝ_vector) : Prop :=
  ∥v∥ = 1

theorem vector_diff_magnitude (a b : ℝ_vector) (h_a_unit: is_unit_vector a) (h_b_unit: is_unit_vector b) (h_sum_unit: ∥a + b∥ = 1) :
  ∥a - b∥ = real.sqrt 3 :=
begin
  sorry
end

end vector_diff_magnitude_l169_169800


namespace find_k_value_l169_169718

theorem find_k_value (x y k : ℝ) 
  (h1 : 2 * x + y = 1) 
  (h2 : x + 2 * y = k - 2) 
  (h3 : x - y = 2) : 
  k = 1 := 
by
  sorry

end find_k_value_l169_169718


namespace magnitude_difference_sqrt3_l169_169814

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Condition 1: a and b are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∥v∥ = 1

-- Condition 2: |a + b| = 1
def sum_of_unit_vectors_has_unit_norm : Prop :=
  ∥a + b∥ = 1

-- Question: |a - b| = sqrt(3)
theorem magnitude_difference_sqrt3 (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hab : sum_of_unit_vectors_has_unit_norm a b) :
  ∥a - b∥ = Real.sqrt 3 :=
sorry

end magnitude_difference_sqrt3_l169_169814


namespace apple_pairing_l169_169918

-- Define the problem setup.
noncomputable def apples : List ℝ := sorry

-- Conditions
axiom apple_count : apples.length = 300
axiom apple_weight_condition : ∀ (i j : Fin 300), apples[i] / apples[j] ≤ 3

-- Define the pairs and their weight relations.
def pairs : List (ℝ × ℝ) := (List.zip (List.take 150 apples) (List.drop 150 apples))

axiom pair_weight_condition : ∀ (i j : Fin 150), let (a, b) := pairs[i]; let (c, d) := pairs[j] in (a + b) / (c + d) ≤ 2

-- The theorem to prove.
theorem apple_pairing :
  ∃ (pairs : List (ℝ × ℝ)), pairs.length = 150 ∧ ∀ (i j : Fin 150), let (a, b) := pairs[i]; let (c, d) := pairs[j] in (a + b) / (c + d) ≤ 2 :=
by
  -- Proof goes here
  sorry

end apple_pairing_l169_169918


namespace one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169540

theorem one_hundred_fiftieth_digit_of_3_div_11_is_7 :
  let decimal_repetition := "27"
  let length := 2
  150 % length = 0 →
  (decimal_repetition[1] = '7')
: sorry

end one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169540


namespace hexagon_inscribed_circle_area_problem_l169_169858

theorem hexagon_inscribed_circle_area_problem
  (r : ℝ)
  (Q B1 B2 B3 B4 B5 B6 : ℝ)
  (h_circ : π * r^2 = 1)
  (h_segm_1_2 : (1 / 6) = π * r^2 / 6 - (π * r^2 / 6 - (π * r^2 / 6 - sqrt 3 * r^2 / 4 / sqrt π))
  (h_segm_4_5 : (1 / 10) = π * r^2 / 10 - (π * r^2 / 6 - sqrt 3 * r^2 / 4 / sqrt π))
  (h_segm_3_4 : (1 / 7 - sqrt 3 / (m : ℝ)) = π * r^2 / 7 - (π * r^2 / 6 - sqrt 3 * r^2 / 4 / sqrt π)) :
  m = 504 :=
sorry

end hexagon_inscribed_circle_area_problem_l169_169858


namespace percentage_needed_to_pass_l169_169603

-- Definitions for conditions
def obtained_marks : ℕ := 125
def failed_by : ℕ := 40
def total_marks : ℕ := 500
def passing_marks := obtained_marks + failed_by

-- Assertion to prove
theorem percentage_needed_to_pass : (passing_marks : ℕ) * 100 / total_marks = 33 := by
  sorry

end percentage_needed_to_pass_l169_169603


namespace range_of_even_and_increasing_l169_169390

noncomputable def even_function (f: ℝ → ℝ) := ∀ x: ℝ, f(x) = f(-x)
noncomputable def increasing_function (f: ℝ → ℝ) (a b: ℝ) := a ≤ b → f(a) ≤ f(b) 

theorem range_of_even_and_increasing (f: ℝ → ℝ)
  (h_even: even_function f)
  (h_increasing: increasing_function f) :
  ∀ x, f(x) < f(2) ↔ x ∈ {x | x < -2} ∪ {x | x > 2} :=
by
  sorry

end range_of_even_and_increasing_l169_169390


namespace range_of_x_l169_169029

noncomputable def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x) + 1

theorem range_of_x (x : ℝ) (h : f (2 * x - 1) + f (4 - x^2) > 2) : x ∈ Set.Ioo (-1 : ℝ) 3 :=
by
  sorry

end range_of_x_l169_169029


namespace sum_first_eleven_terms_l169_169412

open Real

variable {a : ℕ → ℝ} (h_arith : ∃ d : ℝ, ∀ n : ℕ, a(n + 1) = a(n) + d)
variable (h4_8 : a 4 + a 8 = 16)

theorem sum_first_eleven_terms : (∑ i in Finset.range 11, a i) = 88 :=
sorry

end sum_first_eleven_terms_l169_169412


namespace line_through_fixed_point_max_distance_range_length_MN_l169_169708

-- Definitions based on conditions
def line_l (m : ℝ) : ℝ → ℝ → Prop := λ x y, 2 * x + (1 + m) * y + 2 * m = 0
def P : ℝ × ℝ := (-1, 0)
def Q : ℝ × ℝ := (1, -2)
def N : ℝ × ℝ := (2, 1)

-- Problem (1) statement
theorem line_through_fixed_point :
  ∀(m : ℝ), line_l m (1:ℝ) (-2:ℝ) :=
by sorry

-- Problem (2) statement
theorem max_distance (m : ℝ) :
  let dist (a b : ℝ × ℝ) := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  dist P Q = 2 * real.sqrt 2 :=
by sorry

-- Problem (3) statement
theorem range_length_MN (m : ℝ) :
  let dist (a b : ℝ × ℝ) := real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  ∀ M : ℝ × ℝ, line_l m M.1 M.2 → (∃(k : ℝ), dist P M = k ∧ k = 2 * real.sqrt 2) →
  ∃ (MN_length : ℝ), (real.sqrt 2 ≤ MN_length ∧ MN_length ≤ 3 * real.sqrt 2) :=
by sorry

end line_through_fixed_point_max_distance_range_length_MN_l169_169708


namespace minimum_value_is_12_l169_169441

noncomputable def smallest_possible_value (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) : ℝ :=
(a + b + c + d) * ((1 / (a + b)) + (1 / (a + c)) + (1 / (a + d)) + (1 / (b + c)) + (1 / (b + d)) + (1 / (c + d)))

theorem minimum_value_is_12 (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (h5 : a ≤ b) (h6 : b ≤ c) (h7 : c ≤ d) :
  smallest_possible_value a b c d h1 h2 h3 h4 h5 h6 h7 ≥ 12 :=
sorry

end minimum_value_is_12_l169_169441


namespace no_such_natural_number_exists_l169_169299

theorem no_such_natural_number_exists :
  ¬ ∃ n : ℕ, ∃ m : ℕ, 3^n + 2 * 17^n = m^2 :=
by sorry

end no_such_natural_number_exists_l169_169299


namespace inequality_proof_l169_169789

variable {a b c : ℝ}

theorem inequality_proof (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_mul : a * b * c = 1) : 
  (a - 1) / c + (c - 1) / b + (b - 1) / a ≥ 0 :=
sorry

end inequality_proof_l169_169789


namespace jungkook_red_balls_l169_169090

-- Definitions from conditions
def num_boxes : ℕ := 2
def red_balls_per_box : ℕ := 3

-- Theorem stating the problem
theorem jungkook_red_balls : (num_boxes * red_balls_per_box) = 6 :=
by sorry

end jungkook_red_balls_l169_169090


namespace verify_optionD_is_correct_l169_169554

-- Define the equations as options
def optionA : Prop := -abs (-6) = 6
def optionB : Prop := -(-6) = -6
def optionC : Prop := abs (-6) = -6
def optionD : Prop := -(-6) = 6

-- The proof problem to verify option D is correct
theorem verify_optionD_is_correct : optionD :=
by
  sorry

end verify_optionD_is_correct_l169_169554


namespace midpoint_locus_l169_169929

-- Definitions:
def center_O := (0 : ℝ, 0 : ℝ)
def radius_large := 10
def radius_small := 5
def point_P := (15 : ℝ, 0 : ℝ)

-- Theorem statement:
theorem midpoint_locus :
  ∀ Q : ℝ × ℝ, 
    (dist Q center_O = radius_large) →
      ∃ M : ℝ × ℝ, 
        (dist M (7.5, 0) = radius_small) 
        ∧ M = ((point_P.1 + Q.1) / 2, (point_P.2 + Q.2) / 2) :=
sorry

end midpoint_locus_l169_169929


namespace distance_z6_from_origin_l169_169635

noncomputable def z : ℕ → ℂ
| 0       := 1
| (n + 1) := (z n)^2 + 1 + complex.i

theorem distance_z6_from_origin :
  complex.abs (z 5) = real.sqrt 3835182225545 :=
by
  sorry

end distance_z6_from_origin_l169_169635


namespace sum_of_interior_edges_l169_169252

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l169_169252


namespace disputed_piece_weight_l169_169866

theorem disputed_piece_weight (x d : ℝ) (h1 : x - d = 300) (h2 : x + d = 500) : x = 400 := by
  sorry

end disputed_piece_weight_l169_169866


namespace second_tap_empty_time_l169_169581

theorem second_tap_empty_time (hours_fill_first_tap : ℝ) (hours_fill_both_taps : ℝ) 
(hrs_1_pos : 0 < hours_fill_first_tap) (hrs_2_pos : 0 < hours_fill_both_taps) : 
    let F := 1 / hours_fill_first_tap in
    let combined_rate := 1 / hours_fill_both_taps in
    ∃ x : ℝ, (0 < x) ∧ (F - (1 / x) = combined_rate) ∧ x = 9 :=
by
  let F := 1 / hours_fill_first_tap
  let combined_rate := 1 / hours_fill_both_taps
  use 9
  split
  sorry

end second_tap_empty_time_l169_169581


namespace nested_radical_solution_l169_169656

theorem nested_radical_solution :
  (∃ x : ℝ, x = Real.sqrt (18 + x) ∧ x ≥ 0) ∧ ∀ x : ℝ, x = Real.sqrt (18 + x) → x ≥ 0 → x = 6 :=
by
  sorry

end nested_radical_solution_l169_169656


namespace johnson_and_martinez_tied_at_may_l169_169240

def home_runs_johnson (m : String) : ℕ :=
  if m = "January" then 2 else
  if m = "February" then 12 else
  if m = "March" then 20 else
  if m = "April" then 15 else
  if m = "May" then 9 else 0

def home_runs_martinez (m : String) : ℕ :=
  if m = "January" then 5 else
  if m = "February" then 9 else
  if m = "March" then 15 else
  if m = "April" then 20 else
  if m = "May" then 9 else 0

def cumulative_home_runs (player_home_runs : String → ℕ) (months : List String) : ℕ :=
  months.foldl (λ acc m => acc + player_home_runs m) 0

def months_up_to_may : List String :=
  ["January", "February", "March", "April", "May"]

theorem johnson_and_martinez_tied_at_may :
  cumulative_home_runs home_runs_johnson months_up_to_may
  = cumulative_home_runs home_runs_martinez months_up_to_may :=
by
    sorry

end johnson_and_martinez_tied_at_may_l169_169240


namespace greatest_sum_of_consecutive_integers_l169_169965

def consecutiveSumCondition (n : ℤ) : Prop :=
  n * (n + 1) < 500 

theorem greatest_sum_of_consecutive_integers : 
  ∃ n : ℤ, consecutiveSumCondition n ∧ ∀ m : ℤ, consecutiveSumCondition m → n + (n + 1) ≥ m + (m + 1) :=
sorry

end greatest_sum_of_consecutive_integers_l169_169965


namespace triangle_type_l169_169750

theorem triangle_type (A B C : ℝ) (h : sin A * cos A = sin B * cos B) :
  (2 * A + 2 * B = 180) ∨ (A = B) → (A + B = 90) ∨ (A = B) := 
by
  sorry

end triangle_type_l169_169750


namespace carol_fewest_pastries_l169_169229

-- Definitions based on the conditions
def alice_area := 15
def bob_area := 9
def carol_area := 19.625
def dave_area := 10
def carol_pastries := 20

-- The total dough P used by each friend is the same
variable (P : ℝ)

-- Number of pastries each friend can make, based on the same total dough
def alice_pastries := P / alice_area
def bob_pastries := P / bob_area
def carol_pastries_calculated := P / carol_area
def dave_pastries := P / dave_area

-- The proof problem to be stated in Lean
theorem carol_fewest_pastries (hP_pos : 0 < P) : 
  carol_pastries_calculated = 20 ∧ carol_pastries_calculated ≤ min (min alice_pastries bob_pastries) dave_pastries := 
sorry

end carol_fewest_pastries_l169_169229


namespace trapezoids_not_necessarily_congruent_l169_169927

-- Define trapezoid structure
structure Trapezoid (α : Type) [LinearOrderedField α] :=
(base1 base2 side1 side2 diag1 diag2 : α) -- sides and diagonals
(angle1 angle2 angle3 angle4 : α)        -- internal angles

-- Conditions about given trapezoids
variables {α : Type} [LinearOrderedField α]
variables (T1 T2 : Trapezoid α)

-- The condition that corresponding angles of the trapezoids are equal
def equal_angles := 
  T1.angle1 = T2.angle1 ∧ T1.angle2 = T2.angle2 ∧ 
  T1.angle3 = T2.angle3 ∧ T1.angle4 = T2.angle4

-- The condition that diagonals of the trapezoids are equal
def equal_diagonals := 
  T1.diag1 = T2.diag1 ∧ T1.diag2 = T2.diag2

-- The statement to prove
theorem trapezoids_not_necessarily_congruent :
  equal_angles T1 T2 ∧ equal_diagonals T1 T2 → ¬ (T1 = T2) := by
  sorry

end trapezoids_not_necessarily_congruent_l169_169927


namespace avg_score_all_matches_l169_169563

-- Definitions from the conditions
variable (score1 score2 : ℕ → ℕ) 
variable (avg1 avg2 : ℕ)
variable (count1 count2 : ℕ)

-- Assumptions from the conditions
axiom avg_score1 : avg1 = 30
axiom avg_score2 : avg2 = 40
axiom count1_matches : count1 = 2
axiom count2_matches : count2 = 3

-- The proof statement
theorem avg_score_all_matches : 
  ((score1 0 + score1 1) + (score2 0 + score2 1 + score2 2)) / (count1 + count2) = 36 := 
  sorry

end avg_score_all_matches_l169_169563


namespace sum_of_interior_edges_l169_169258

theorem sum_of_interior_edges {f : ℝ} {w : ℝ} (h_frame_area : f = 32) (h_outer_edge : w = 7) (h_frame_width : 2) :
  let i_length := w - 2 * h_frame_width in
  let i_other_length := (f - (w * (w  - 2 * h_frame_width))) / (w  + 2 * h_frame_width) in
  i_length + i_other_length + i_length + i_other_length = 8 :=
by
  let i_length := w - 2 * 2
  let i_other_length := (32 - (i_length * w)) / (w  + 2 * 2)
  let sum := i_length + i_other_length + i_length + i_other_length
  have h_sum : sum = 8, by sorry
  exact h_sum

end sum_of_interior_edges_l169_169258


namespace three_tan_C_eq_79_l169_169400

open Real

theorem three_tan_C_eq_79 (ABC : Triangle)
  (h₁ : ∠A = A ∧ ∠B = B ∧ ∠C = C)
  (acute : ∀ (θ : ABC.Angle), θ < π / 2)
  (h₂ : sin A = 3 / 5)
  (h₃ : tan (A - B) = -1 / 3) : 3 * tan C = 79 :=
by
  sorry

end three_tan_C_eq_79_l169_169400


namespace determine_d_l169_169455

variables (u v : ℝ × ℝ × ℝ) -- defining u and v as 3D vectors

noncomputable def cross_product (a b : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (a.2.2 * b.2.1 - a.2.1 * b.2.2, a.1 * b.2.2 - a.2.2 * b.1 , a.2.1 * b.1 - a.1 * b.2.1)

noncomputable def dot_product (a b : ℝ × ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

noncomputable def i : ℝ × ℝ × ℝ := (1, 0, 0)
noncomputable def j : ℝ × ℝ × ℝ := (0, 1, 0)
noncomputable def k : ℝ × ℝ × ℝ := (0, 0, 1)

theorem determine_d (u : ℝ × ℝ × ℝ) (v : ℝ × ℝ × ℝ) :
  cross_product i (cross_product (u + v) i) +
  cross_product j (cross_product (u + v) j) +
  cross_product k (cross_product (u + v) k) =
  2 * (u + v) :=
sorry

end determine_d_l169_169455


namespace greatest_sum_consecutive_integers_lt_500_l169_169992

theorem greatest_sum_consecutive_integers_lt_500 : 
  ∃ n : ℤ, (n * (n + 1) < 500) ∧ n + (n + 1) = 43 := 
by {
  sorry -- Proof needed
}

end greatest_sum_consecutive_integers_lt_500_l169_169992


namespace count_multiples_4_6_not_5_9_l169_169046

/-- The number of integers between 1 and 500 that are multiples of both 4 and 6 but not of either 5 or 9 is 22. -/
theorem count_multiples_4_6_not_5_9 :
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22 :=
by
  let lcm_4_6 := (Nat.lcm 4 6)
  let lcm_4_6_5 := (Nat.lcm lcm_4_6 5)
  let lcm_4_6_9 := (Nat.lcm lcm_4_6 9)
  let lcm_4_6_5_9 := (Nat.lcm lcm_4_6_5 9)
  let count_multiples (x : Nat) := (500 / x)
  show count_multiples lcm_4_6 - count_multiples lcm_4_6_5 - count_multiples lcm_4_6_9 + count_multiples lcm_4_6_5_9 = 22
  sorry

end count_multiples_4_6_not_5_9_l169_169046


namespace magnitude_difference_sqrt3_l169_169816

variables {a b : EuclideanSpace ℝ (Fin 2)}

-- Condition 1: a and b are unit vectors
def is_unit_vector (v : EuclideanSpace ℝ (Fin 2)) : Prop :=
  ∥v∥ = 1

-- Condition 2: |a + b| = 1
def sum_of_unit_vectors_has_unit_norm : Prop :=
  ∥a + b∥ = 1

-- Question: |a - b| = sqrt(3)
theorem magnitude_difference_sqrt3 (ha : is_unit_vector a) (hb : is_unit_vector b)
  (hab : sum_of_unit_vectors_has_unit_norm a b) :
  ∥a - b∥ = Real.sqrt 3 :=
sorry

end magnitude_difference_sqrt3_l169_169816


namespace abs_diff_squares_l169_169953

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169953


namespace vector_magnitude_is_five_l169_169342

noncomputable def vectors_magnitude : ℝ :=
  let AB := (1 : ℝ) in
  let AC := (2 : ℝ) in
  let AP := (3 : ℝ) in
  let cos_60 := (1 / 2 : ℝ) in
  let dot_AB_AC := AB * AC * cos_60 in
  let dot_AC_AP := AC * AP * cos_60 in
  let dot_AB_AP := AB * AP * cos_60 in
  Real.sqrt ((AB^2) + (AC^2) + (AP^2) + 2 * dot_AB_AC + 2 * dot_AC_AP + 2 * dot_AB_AP)

theorem vector_magnitude_is_five (h1 : ∠ PAB = 60) (h2 : ∠ BAC = 60) (h3 : ∠ PAC = 60) 
                                  (h4 : |AB| = 1) (h5 : |AC| = 2) (h6 : |AP| = 3) :
  vectors_magnitude = 5 := 
sorry

end vector_magnitude_is_five_l169_169342


namespace log_base_change_l169_169818

theorem log_base_change (a b : ℝ) (h1 : a = log 4 484) (h2 : b = log 2 22) : a = b := by
  sorry

end log_base_change_l169_169818


namespace max_wins_gsw_possible_l169_169878

/-- Define the problem parameters --/
def games_played : ℕ := 100
def avg_points_scored_per_game : ℕ := 7
def avg_points_allowed_per_game : ℕ := 8
def max_point_diff : ℤ := 10
def total_point_diff : ℤ := -100

/-- Translating conditions and goal statement in Lean --/
theorem max_wins_gsw_possible :
  ∃ W : ℕ, W ≤ 81 ∧
  (∀ L : ℕ, W + L = games_played ∧ 
           total_point_diff = games_played * (avg_points_scored_per_game - avg_points_allowed_per_game) ∧ 
           (-max_point_diff ≤ (avg_points_scored_per_game - avg_points_allowed_per_game) ∧ 
           (avg_points_scored_per_game - avg_points_allowed_per_game) ≤ max_point_diff) →
           L = games_played - W) :=
begin
  sorry
end

end max_wins_gsw_possible_l169_169878


namespace sum_A_k_div_k_l169_169326

noncomputable def A (k : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1 ∧ d ≤ Nat.sqrt (2 * k - 1)) (Finset.range k)).card

noncomputable def sumExpression : ℝ :=
  ∑' k, (-1)^(k-1) * (A k / k : ℝ)

theorem sum_A_k_div_k : sumExpression = Real.pi^2 / 8 :=
  sorry

end sum_A_k_div_k_l169_169326


namespace pollen_particle_diameter_in_scientific_notation_l169_169266

theorem pollen_particle_diameter_in_scientific_notation :
  0.0000078 = 7.8 * 10^(-6) :=
by
  sorry

end pollen_particle_diameter_in_scientific_notation_l169_169266


namespace min_value_in_interval_l169_169492

noncomputable def f (x : ℝ) : ℝ := x^3 + x^2 - x + 1

theorem min_value_in_interval : ∃ x ∈ set.Icc (-2 : ℝ) 1, f x = -1 :=
by
  sorry

end min_value_in_interval_l169_169492


namespace transformed_set_mean_and_variance_l169_169694

variables {n : ℕ} (x : Fin n → ℝ)
noncomputable def mean (x : Fin n → ℝ) : ℝ := (∑ i, x i) / n
noncomputable def variance (x : Fin n → ℝ) : ℝ := (∑ i, (x i - mean x)^2) / n

theorem transformed_set_mean_and_variance
  (x : Fin n → ℝ)
  (mean_x : ℝ)
  (s2 : ℝ)
  (h_mean : mean x = mean_x)
  (h_variance : variance x = s2) :
  mean (fun i => sqrt 3 * x i + sqrt 2) = sqrt 3 * mean_x + sqrt 2 ∧
  variance (fun i => sqrt 3 * x i + sqrt 2) = 3 * s2 :=
by
  sorry

end transformed_set_mean_and_variance_l169_169694


namespace largest_value_expression_l169_169764

theorem largest_value_expression (a b c : ℝ) (ha : a ∈ ({1, 2, 4} : Set ℝ)) (hb : b ∈ ({1, 2, 4} : Set ℝ)) (hc : c ∈ ({1, 2, 4} : Set ℝ)) (habc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c) :
  (a / 2) / (b / c) ≤ 4 :=
sorry

end largest_value_expression_l169_169764


namespace rational_function_solution_eq_l169_169314

theorem rational_function_solution_eq (f : ℚ → ℚ) (h₀ : f 0 = 0) 
  (h₁ : ∀ x y : ℚ, f (f x + f y) = x + y) :
  (∀ x : ℚ, f x = x) ∨ (∀ x : ℚ, f x = -x) := 
sorry

end rational_function_solution_eq_l169_169314


namespace range_of_x_for_obtuse_angle_l169_169388

def vectors_are_obtuse (a b : ℝ × ℝ) : Prop :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  dot_product < 0

theorem range_of_x_for_obtuse_angle :
  ∀ (x : ℝ), vectors_are_obtuse (1, 3) (x, -1) ↔ (x < -1/3 ∨ (-1/3 < x ∧ x < 3)) :=
by
  sorry

end range_of_x_for_obtuse_angle_l169_169388


namespace collinearity_condition_l169_169493

variables {R : Type*} [Nontrivial R] [AddCommGroup R] [Module ℝ R]

def collinear (a b : R) : Prop :=
  ∃ (λ1 λ2 : ℝ), λ1 ≠ 0 ∧ λ2 ≠ 0 ∧ λ1 • a + λ2 • b = 0

theorem collinearity_condition (a b : R) : collinear a b :=
sorry

end collinearity_condition_l169_169493


namespace magnitude_diff_unit_vectors_l169_169811

variables (a b : ℝ^3) -- Define vector variables a and b in 3-dimensional real space

-- Define the conditions
def is_unit_vector (v : ℝ^3) : Prop := ‖v‖ = 1
def sum_is_one (v1 v2 : ℝ^3) : Prop := ‖v1 + v2‖ = 1

-- State the theorem
theorem magnitude_diff_unit_vectors (a b : ℝ^3) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (h_sum : sum_is_one a b) : 
  ‖a - b‖ = real.sqrt 3 :=
sorry

end magnitude_diff_unit_vectors_l169_169811


namespace solution_set_is_m_x7_l169_169704

-- Define the conditions

variable {R : Type*} [LinearOrder R] [Preorder R] [TopologicalSpace R]
variables (f : R → R) [DecidablePred (λ x, f x > 0)]
variable {x : R}

-- The function is decreasing
def decreasing (f : R → R) : Prop :=
  ∀ ⦃a b : R⦄, a < b → f a ≥ f b 

-- The conditions given in the problem
def condition1 (f : R → R) : Prop :=
  decreasing f

def condition2 (f : R → R) : Prop :=
  f 0 = 1

def condition3 (f : R → R) : Prop :=
  f 1 = 0

-- The theorem statement
theorem solution_set_is_m_x7: 
  condition1 f → condition2 f → condition3 f → 
  ∀ x, x < 1 → f x > 0 := 
begin 
  sorry -- To be filled in with the complete proof later
end

end solution_set_is_m_x7_l169_169704


namespace range_of_a_for_perpendicular_tangent_line_l169_169741

theorem range_of_a_for_perpendicular_tangent_line (a : ℝ) :
  (∃ x > 0, ∃ y : ℝ, (f : ℝ → ℝ) = (λ x, a*x^3 + Real.log x) ∧ (f' : ℝ → ℝ) = (λ x, 3*a*x^2 + 1/x) ∧ (f'' : ℝ → ℝ) = (λ x, 6*a*x - 1/x^2) ∧ (∀ x, f'' x ≠ 0) ∧ (∀ x > 0, f' x → ∞)) → a < 0 := 
begin
  sorry
end

end range_of_a_for_perpendicular_tangent_line_l169_169741


namespace stagecoaches_encountered_on_journey_l169_169061
-- Import the necessary Lean library

-- Define the conditions as Lean definitions and assumptions
def daily_coaches_bratislava_to_brasov : ℕ := 2
def daily_coaches_brasov_to_bratislava : ℕ := 2
def journey_days : ℕ := 10

-- The Lean statement for the problem
theorem stagecoaches_encountered_on_journey :
  ∀ (daily_coaches_bratislava_to_brasov daily_coaches_brasov_to_bratislava journey_days : ℕ),
  (daily_coaches_bratislava_to_brasov = 2) ->
  (daily_coaches_brasov_to_bratislava = 2) ->
  (journey_days = 10) ->
  (2 * journey_days = 20) :=
by
  intros daily_coaches_bratislava_to_brasov daily_coaches_brasov_to_bratislava journey_days h1 h2 h3
  rw [h1, h2, h3]
  show 2 * 10 = 20
  rfl

end stagecoaches_encountered_on_journey_l169_169061


namespace sum_of_interior_edges_l169_169253

theorem sum_of_interior_edges (frame_width : ℕ) (frame_area : ℕ) (outer_edge : ℕ) 
  (H1 : frame_width = 2) (H2 : frame_area = 32) (H3 : outer_edge = 7) : 
  2 * (outer_edge - 2 * frame_width) + 2 * (x : ℕ) = 8 :=
by
  sorry

end sum_of_interior_edges_l169_169253


namespace simplify_expression_l169_169870

theorem simplify_expression : 
  (sqrt 507 / sqrt 48 - sqrt 175 / sqrt 112) = 2 := 
by
  sorry

end simplify_expression_l169_169870


namespace books_sold_on_tuesday_l169_169774

def total_stock : ℕ := 620
def sold_monday : ℕ := 50
def sold_wednesday : ℕ := 60
def sold_thursday : ℕ := 48
def sold_friday : ℕ := 40
def unsold_percentage : ℚ := 54.83870967741935 / 100

def unsold_books : ℚ := total_stock * unsold_percentage
def unsold_books_rounded : ℤ := unsold_books.toReal.floor + 1 -- rounded up

def sold_books : ℤ := total_stock - unsold_books_rounded
def sold_mon_wed_thu_fri : ℕ := sold_monday + sold_wednesday + sold_thursday + sold_friday

theorem books_sold_on_tuesday : (sold_books - sold_mon_wed_thu_fri) = 82 :=
sorry

end books_sold_on_tuesday_l169_169774


namespace min_value_M_l169_169015

theorem min_value_M (a b c : ℝ) (h1 : a < b) (h2 : ∀ x : ℝ, a * x^2 + b * x + c ≥ 0): 
  ∃ M : ℝ, M = 8 ∧ M = (a + 2 * b + 4 * c) / (b - a) :=
sorry

end min_value_M_l169_169015


namespace sara_museum_visit_l169_169134

theorem sara_museum_visit (S : Finset ℕ) (hS : S.card = 6) :
  ∃ count : ℕ, count = 720 ∧ 
  (∀ M A : Finset ℕ, M.card = 3 → A.card = 3 → M ∪ A = S → 
    count = (S.card.choose M.card) * M.card.factorial * A.card.factorial) :=
by
  sorry

end sara_museum_visit_l169_169134


namespace spider_legs_total_l169_169772

-- Definitions based on given conditions
def spiders : ℕ := 4
def legs_per_spider : ℕ := 8

-- Theorem statement
theorem spider_legs_total : (spiders * legs_per_spider) = 32 := by
  sorry

end spider_legs_total_l169_169772


namespace product_of_consecutive_integers_l169_169430

theorem product_of_consecutive_integers (l : List ℤ) (h1 : l.length = 2019) (h2 : l.sum = 2019) : l.prod = 0 := 
sorry

end product_of_consecutive_integers_l169_169430


namespace jane_exercises_per_day_l169_169087

-- Conditions
variables (total_hours : ℕ) (total_weeks : ℕ) (days_per_week : ℕ)
variable (goal_achieved : total_hours = 40 ∧ total_weeks = 8 ∧ days_per_week = 5)

-- Statement
theorem jane_exercises_per_day : ∃ hours_per_day : ℕ, hours_per_day = (total_hours / total_weeks) / days_per_week :=
by
  sorry

end jane_exercises_per_day_l169_169087


namespace range_of_a_l169_169028

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2^(1 + a * x) - x

noncomputable def F (a : ℝ) (x : ℝ) : ℝ := f a (f a x) - x

theorem range_of_a (a : ℝ) : (F a 0 = 0 → F a e = 0) → 
  (0 < a ∧ a < (1 / (Real.exp 1 * Real.log 2))) :=
by
  sorry

end range_of_a_l169_169028


namespace quadratic_unique_solution_k_neg_l169_169673

theorem quadratic_unique_solution_k_neg (k : ℝ) :
  (∃ x : ℝ, 9 * x^2 + k * x + 36 = 0 ∧ ∀ y : ℝ, 9 * y^2 + k * y + 36 = 0 → y = x) →
  k = -36 :=
by
  sorry

end quadratic_unique_solution_k_neg_l169_169673


namespace wave_number_count_l169_169585

/-- Definition of a wave number -/
def is_wave_number (n : ℕ) : Prop :=
  let digits := [((n / 10000) % 10), ((n / 1000) % 10), ((n / 100) % 10), ((n / 10) % 10), (n % 10)]
  (digits[1] > digits[0] && digits[1] > digits[2]) && 
  (digits[3] > digits[2] && digits[3] > digits[4]) &&
  digits.nodup && 
  digits ∈ permutations [1, 2, 3, 4, 5]

/-- The main theorem to prove the number of wave numbers -/
theorem wave_number_count : 
  (finset.range 100000).filter is_wave_number).card = 16 :=
sorry

end wave_number_count_l169_169585


namespace bicycle_not_in_motion_time_l169_169267

def distance : ℝ := 22.5
def bert_ride_speed : ℝ := 8
def bert_walk_speed : ℝ := 5
def al_walk_speed : ℝ := 4
def al_ride_speed : ℝ := 10

theorem bicycle_not_in_motion_time : 
  let x := 10 in
  let bert_time := x / bert_ride_speed + (distance - x) / bert_walk_speed in
  let al_time := x / al_walk_speed + (distance - x) / al_ride_speed in
  bert_time = al_time → 
  75 * 60 = 75
:= by
  sorry

end bicycle_not_in_motion_time_l169_169267


namespace union_A_B_intersection_complement_A_B_subset_A_C_implies_a_gt_7_l169_169346

variable (a : ℝ)

def A : set ℝ := {x | 3 ≤ x ∧ x ≤ 7}
def B : set ℝ := {x | 2 < x ∧ x < 10}
def C (a : ℝ) : set ℝ := {x | x < a}

theorem union_A_B : A ∪ B = {x | 2 < x ∧ x < 10} :=
sorry

theorem intersection_complement_A_B : (C \ A) ∩ B = ({x | 2 < x ∧ x < 3} ∪ {x | 7 < x ∧ x < 10}) :=
sorry

theorem subset_A_C_implies_a_gt_7 : (A ⊆ C a) → a > 7 :=
sorry

end union_A_B_intersection_complement_A_B_subset_A_C_implies_a_gt_7_l169_169346


namespace ordered_quadruples_div_100_l169_169442

theorem ordered_quadruples_div_100 :
  let m := finset.card { (x1 : ℕ, x2 : ℕ, x3 : ℕ, x4 : ℕ) |
    (x1 > 0) ∧ (x2 > 0) ∧ (x3 > 0) ∧ (x4 > 0) ∧
    (x1 % 2 = 0) ∧ (x2 % 2 = 0) ∧ (x3 % 2 = 0) ∧ (x4 % 2 = 0) ∧
    (x1 + x2 + x3 + x4 = 100) } in
  (m / 100) = 221 :=
by
  sorry

end ordered_quadruples_div_100_l169_169442


namespace unique_function_prime_set_l169_169796

open Function

noncomputable def P : Set ℕ := {p : ℕ | Nat.Prime p}

theorem unique_function_prime_set (f : ℕ → ℕ) (hf : ∀ p q ∈ P, (f p)^(f q) + q^p = (f q)^(f p) + p^q) :
  ∀ p ∈ P, f p = p :=
by
  sorry

end unique_function_prime_set_l169_169796


namespace abs_diff_squares_l169_169956

-- Definitions for the numbers 105 and 95
def a : ℕ := 105
def b : ℕ := 95

-- Statement to prove: The absolute value of the difference between the squares of 105 and 95 is 2000.
theorem abs_diff_squares : |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169956


namespace standard_circle_eqn_cartesian_line_eqn_min_distance_midpoint_l169_169074

section Problem

-- Definitions of parametric equations for the circle C
def parametric_circle_x (t : ℝ) : ℝ := 2 * sin t
def parametric_circle_y (t : ℝ) : ℝ := 2 * cos t

-- Polar equation of the line l
def polar_line (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = 2 * sqrt 2

-- Point A
def point_A : ℝ × ℝ := (2, 0)

-- Standard equation of circle C
theorem standard_circle_eqn : ∀ t : ℝ, (parametric_circle_x t)^2 + (parametric_circle_y t)^2 = 4 := 
sorry

-- Conversion of polar to Cartesian coordinates for line l
theorem cartesian_line_eqn : ∀ (ρ θ : ℝ), polar_line ρ θ → ∃ x y : ℝ, x + y - 4 = 0 := 
sorry

-- Minimum distance from midpoint M to line l
theorem min_distance_midpoint : ∃ α : ℝ, let M := ((cos α) + 1, sin α) in 
  ∀ (x y : ℝ), (x + y - 4 = 0) → (min (abs ((cos α) + (sin α) - 3) / sqrt 2) = (3 * sqrt 2 - 2) / 2) :=
sorry

end Problem

end standard_circle_eqn_cartesian_line_eqn_min_distance_midpoint_l169_169074


namespace greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169972

theorem greatest_possible_sum_of_two_consecutive_integers_lt_500 (n : ℕ) (h : n * (n + 1) < 500) : n + (n + 1) ≤ 43 := by
  sorry

end greatest_possible_sum_of_two_consecutive_integers_lt_500_l169_169972


namespace distance_dot_product_l169_169451

noncomputable def ellipse_equation (a b : ℝ) (h : 0 < b ∧ b < a) (ec : a / b ≥ 2) (min_dist : ℝ) : 
  ∀ x y, 
  (x ^ 2 / a ^ 2) + (y ^ 2 / b ^ 2) = 1 ↔ (x ^ 2 / 4) + (y ^ 2 / 3) = 1 :=
by sorry

theorem distance_dot_product (x y : ℝ) (k : ℝ) (hline : y = k*x + 2) :
  (∀ a b : ℝ, 
  (a, b) ∈ ellipse_intersection (0, y)) → 
  (a, b) ∈ ellipse_intersection (x, y) →
  (- ∞, 13 / 4] :=
by sorry

end distance_dot_product_l169_169451


namespace abs_diff_squares_l169_169941

theorem abs_diff_squares (a b : ℤ) (h_a : a = 105) (h_b : b = 95):
  |a^2 - b^2| = 2000 := by
  sorry

end abs_diff_squares_l169_169941


namespace has_local_maximum_l169_169132

noncomputable def func (x : ℝ) : ℝ := (1 / 3) * x ^ 3 - 4 * x + 4

theorem has_local_maximum :
  ∃ x, x = -2 ∧ func x = 28 / 3 :=
by
  sorry

end has_local_maximum_l169_169132


namespace socks_count_l169_169613

theorem socks_count
  (black_socks : ℕ) 
  (white_socks : ℕ)
  (H1 : white_socks = 4 * black_socks)
  (H2 : black_socks = 6)
  (H3 : white_socks / 2 = white_socks - (white_socks / 2)) :
  (white_socks / 2) - black_socks = 6 := by
  sorry

end socks_count_l169_169613


namespace number_of_distinct_triangles_l169_169286

noncomputable def triangle_integers (x y : ℕ): Prop :=
  40 * x + y = 2030 ∧ x + 2 * y = 900

theorem number_of_distinct_triangles : 
  ∃ n : ℕ, n = 156 ∧ 
    ∀ O P Q : (ℕ × ℕ), 
    O = (0, 0) →
    P ≠ Q →
    triangle_integers P.1 P.2 ∧
    triangle_integers Q.1 Q.2 →
    ((P.1 ≠ Q.1) ∧ (P.2 ≠ Q.2)) →
    ((|P.1 - Q.1| % 2 = 0) → ∃Δ, Δ = n) :=
by
  sorry

end number_of_distinct_triangles_l169_169286


namespace area_enclosed_by_cos_l169_169880

open Real

/-- The area enclosed by the curve y = cos x for 0 <= x <= π is 2. -/
theorem area_enclosed_by_cos : ∫ x in 0..π, |cos x| = 2 := by
  sorry

end area_enclosed_by_cos_l169_169880


namespace unpainted_cubes_count_l169_169487

theorem unpainted_cubes_count :
  let L := 6
  let W := 6
  let H := 3
  (L - 2) * (W - 2) * (H - 2) = 16 :=
by
  sorry

end unpainted_cubes_count_l169_169487


namespace max_rook_fence_jumps_l169_169645

theorem max_rook_fence_jumps : ∃ n, n = 47 ∧ ∀ (buchess_board : list (list bool)) 
  (rook_move : nat → (nat × nat)),
  (∀ i, ∃ x y, rook_move i = (x, y)) →
  (∀ i j, i ≠ j → rook_move i ≠ rook_move j) →
  (∀ i, ¬ is_fence_cell (rook_move i).fst (rook_move i).snd) →
  (∀ i j, is_jump_over_fence (rook_move i) (rook_move j) → ¬ intermediate_step (rook_move i) (rook_move j)) → n ≤ 47 :=
by sorry

end max_rook_fence_jumps_l169_169645


namespace digit_150_of_decimal_3_div_11_l169_169529

theorem digit_150_of_decimal_3_div_11 : 
  (let digits := [2, 7] in digits[(150 % digits.length)]) = 7 :=
by
  sorry

end digit_150_of_decimal_3_div_11_l169_169529


namespace circles_internally_tangent_l169_169498

theorem circles_internally_tangent :
  ∀ (x y : ℝ),
    (x^2 + y^2 - 6 * x + 4 * y + 12 = 0) ∧
    ((x - 7)^2 + (y - 1)^2 = 36) →
    (distance ⟨3, -2⟩ ⟨7, 1⟩ = real.sqrt ((3 - 7)^2 + (-2 - 1)^2)) →
    (distance ⟨3, -2⟩ ⟨7, 1⟩ = 6 - 1) →
    "The circles are internally tangent" :=
by
  sorry

end circles_internally_tangent_l169_169498


namespace value_of_f_is_negative_l169_169024

theorem value_of_f_is_negative {a b c : ℝ} (h1 : a + b < 0) (h2 : b + c < 0) (h3 : c + a < 0) :
  2 * a ^ 3 + 4 * a + 2 * b ^ 3 + 4 * b + 2 * c ^ 3 + 4 * c < 0 := by
sorry

end value_of_f_is_negative_l169_169024


namespace expr_for_arithmetic_seq_l169_169350

variable {d a₁ a₂ a₃ a₇ : ℝ}

def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ (d : ℝ), ∀ n : ℕ, a (n + 1) = a n + d

def geometric_seq (a : ℕ → ℝ) (i j k : ℕ) : Prop :=
  a j ^ 2 = a i * a k

theorem expr_for_arithmetic_seq
  (a : ℕ → ℝ)
  (h_arith : arithmetic_seq a)
  (h_nonzero_d : ∃ d, d ≠ 0 ∧ (arith_seq_prop a d))
  (h_geom : geometric_seq a 2 3 7)
  (h_cond : 2 * a 1 + a 2 = 1) :
  ∀ n : ℕ, a n = 5/3 - n := 
  by sorry

noncomputable def arith_seq_prop (a : ℕ → ℝ) (d: ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

end expr_for_arithmetic_seq_l169_169350


namespace vector_difference_magnitude_l169_169806

open_locale real_inner_product_space

variables {V : Type*} [inner_product_space ℝ V]

noncomputable def unit_vectors (a b : V) : Prop :=
  ∥a∥ = 1 ∧ ∥b∥ = 1 ∧ ∥a + b∥ = 1

theorem vector_difference_magnitude (a b : V)
  (h : unit_vectors a b) : ∥a - b∥ = real.sqrt 3 :=
by sorry

end vector_difference_magnitude_l169_169806


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169983

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169983


namespace jerry_showers_l169_169301

theorem jerry_showers :
  ∀ gallons_total gallons_drinking_cooking gallons_per_shower pool_length pool_width pool_height gallons_per_cubic_foot,
    gallons_total = 1000 →
    gallons_drinking_cooking = 100 →
    gallons_per_shower = 20 →
    pool_length = 10 →
    pool_width = 10 →
    pool_height = 6 →
    gallons_per_cubic_foot = 1 →
    let pool_volume := pool_length * pool_width * pool_height in
    pool_volume = 600 →
    let gallons_for_pool := pool_volume * gallons_per_cubic_foot in
    let gallons_for_showers := gallons_total - gallons_drinking_cooking - gallons_for_pool in
    let number_of_showers := gallons_for_showers / gallons_per_shower in
    number_of_showers = 15 :=
by
  intros
  sorry

end jerry_showers_l169_169301


namespace circle_radius_is_3_l169_169707

theorem circle_radius_is_3 (m : ℝ) (r : ℝ) :
  (∀ (M N : ℝ × ℝ), (M ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + m * p.2 - 4 = 0} ∧
                     N ∈ {p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 + m * p.2 - 4 = 0} ∧
                     M + N = (-(M.1 + N.1), -(M.2 + N.2))) →
  r = 3) :=
sorry

end circle_radius_is_3_l169_169707


namespace find_f_zero_l169_169027

def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ := a * x + b

theorem find_f_zero (a b : ℝ)
  (h1 : f 3 a b = 7)
  (h2 : f 5 a b = -1) : f 0 a b = 19 :=
by
  sorry

end find_f_zero_l169_169027


namespace graduation_photo_arrangements_l169_169180

def numArrangements := 1200

theorem graduation_photo_arrangements : 
  let students : Fin 7 := ⟨[A, B, C, D, E, F, G], 7⟩ 
  (count (permutations students) (λ p, 
    (∃ i, (p[i] = A ∧ p[i + 1] = B) ∨ (p[i] = B ∧ p[i + 1] = A)) = false 
    ∧ (∀ j, p[j] = B → (p[j + 1] = C ∨ p[j - 1] = C))) = numArrangements :=
begin
  sorry
end

end graduation_photo_arrangements_l169_169180


namespace students_per_bus_correct_l169_169503

def total_students : ℝ := 28
def number_of_buses : ℝ := 2.0
def students_per_bus : ℝ := 14

theorem students_per_bus_correct :
  total_students / number_of_buses = students_per_bus := 
by
  -- Proof should go here
  sorry

end students_per_bus_correct_l169_169503


namespace smaller_circle_radius_l169_169521

theorem smaller_circle_radius (r R : ℝ) (A1 A2 : ℝ) (hR : R = 5.0) (hA : A1 + A2 = 25 * Real.pi)
  (hap : A2 = A1 + 25 * Real.pi / 2) : r = 5 * Real.sqrt 2 / 2 :=
by
  -- Placeholder for the actual proof
  sorry

end smaller_circle_radius_l169_169521


namespace angle_B_30_degrees_side_b_sqrt_7_l169_169757

noncomputable def acute_triangle (A B C a b c : ℝ) : Prop :=
  ∀ (A B C : ℝ), 
    0 < A ∧ A < π/2 ∧ 
    0 < B ∧ B < π/2 ∧ 
    0 < C ∧ C < π/2 ∧ 
    A + B + C = π 

theorem angle_B_30_degrees (A B C a b c : ℝ) 
  (h₁ : acute_triangle A B C a b c)
  (h₂ : a = 2 * b * Real.sin A) :
  B = π / 6 :=
sorry

theorem side_b_sqrt_7 (A B C a b c : ℝ) 
  (h₁ : acute_triangle A B C a b c)
  (h₂ : a = 3 * Real.sqrt 3)
  (h₃ : c = 5)
  (h₄ : B = π / 6) :
  b = Real.sqrt 7 :=
sorry

end angle_B_30_degrees_side_b_sqrt_7_l169_169757


namespace sum_of_digits_can_be_arbitrarily_large_l169_169128

theorem sum_of_digits_can_be_arbitrarily_large :
  ∀ k : ℕ, ∃ n : ℕ, 0 < n ∧ sum_of_digits (2 ^ n) > k :=
by
  sorry

end sum_of_digits_can_be_arbitrarily_large_l169_169128


namespace constant_term_expansion_l169_169415

theorem constant_term_expansion :
  (∃ (c : ℤ), c = 20 ∧ ∀ (x : ℚ), (x ≠ 0 → ((x + 2 + 1/x) ^ 3).coeff 0 = c)) :=
by
  sorry

end constant_term_expansion_l169_169415


namespace closest_to_circle_is_D_l169_169216

def is_closer_to_circle (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2) / a

def ellipse_A : Prop :=
  let a := 1
  let b := 1 / 2
  ∃ e : ℝ, e = is_closer_to_circle a b ∧ e = Real.sqrt 3 / 2

def ellipse_B : Prop :=
  let a := 1
  let b := Real.sqrt (1 / 2)
  ∃ e : ℝ, e = is_closer_to_circle a b ∧ e = Real.sqrt 2 / 2

def ellipse_C : Prop :=
  let a := 3
  let b := 1
  ∃ e : ℝ, e = is_closer_to_circle a b ∧ e = 2 * Real.sqrt 2 / 3

def ellipse_D : Prop :=
  let a := Real.sqrt 3
  let b := 1
  ∃ e : ℝ, e = is_closer_to_circle a b ∧ e = Real.sqrt 6 / 3

theorem closest_to_circle_is_D (A B C D : Prop) : D = ellipse_D :=
  sorry

end closest_to_circle_is_D_l169_169216


namespace calculate_perimeter_l169_169416

-- Declare the condition
variables {width height removed_area total_area : ℝ}

-- The main condition as identified from the solution
def condition : Prop := 
  width = 11 ∧
  height = 96 / 11 ∧
  removed_area = 12 ∧
  total_area = 84 ∧
  (width * height - removed_area = total_area)

-- The conclusion we aim to prove
def perimeter (width height : ℝ) : ℝ := 
  2 * (width + height - 1) + 22

-- Lean statement to prove the problem
theorem calculate_perimeter (h : condition) : 
  perimeter width height = 327 / 11 :=
begin
  sorry
end

end calculate_perimeter_l169_169416


namespace g_periodic_6_l169_169378

def g (a b c : ℝ) : ℝ × ℝ × ℝ :=
  (a + b, b + c, a + c)

def g_iter (n : Nat) (triple : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  match n with
  | 0 => triple
  | n + 1 => g (g_iter n triple).1 (g_iter n triple).2.1 (g_iter n triple).2.2

theorem g_periodic_6 {a b c : ℝ} (h : ∃ n : Nat, n > 0 ∧ g_iter n (a, b, c) = (a, b, c))
  (h' : (a, b, c) ≠ (0, 0, 0)) : g_iter 6 (a, b, c) = (a, b, c) :=
by
  sorry

end g_periodic_6_l169_169378


namespace range_of_median_AD_in_acute_triangle_l169_169071

variable {A B C D : Type}
variable (a b c : ℝ)
variable (hBC : BC = 2)
variable (hSinEquality : sin B + sin C = 2 * sin A)

theorem range_of_median_AD_in_acute_triangle 
    (h_acute_triangle : acute_triangle A B C)
    (h_sine_relation : sin B + sin C = 2 * sin A)
    (h_side_BC : b + c = 4) :
    ∃ AD_range, AD_range = [real.sqrt 3, real.sqrt 13 / 2) := 
sorry

end range_of_median_AD_in_acute_triangle_l169_169071


namespace part_a_part_b_part_c_part_d_l169_169727

noncomputable def z1 : Complex := 3 * (Complex.cos (5 * Real.pi / 4) + Complex.sin (5 * Real.pi / 4) * Complex.I)
noncomputable def z2 : Complex := 5 * (Complex.cos (Real.pi / 2) + Complex.sin (Real.pi / 2) * Complex.I)

theorem part_a : z1 * z2 = 7.5 * Real.sqrt 2 - 7.5 * Real.sqrt 2 * Complex.I := 
begin
  sorry
end

theorem part_b : z1 / z2 = -0.3 * Real.sqrt 2 + 0.3 * Real.sqrt 2 * Complex.I := 
begin
  sorry
end

theorem part_c : z1 ^ 5 = 121.5 * Real.sqrt 2 + 121.5 * Real.sqrt 2 * Complex.I := 
begin
  sorry
end

theorem part_d : 
  ∃ (w : Complex), w ∈ { sqrt 3 * (Complex.cos (5 * Real.pi / 8) + Complex.sin (5 * Real.pi / 8) * Complex.I), 
                         sqrt 3 * (Complex.cos (13 * Real.pi / 8) + Complex.sin (13 * Real.pi / 8) * Complex.I) } := 
begin
  sorry
end

end part_a_part_b_part_c_part_d_l169_169727


namespace nested_radical_eq_6_l169_169659

theorem nested_radical_eq_6 (x : ℝ) (h : x = Real.sqrt (18 + x)) : x = 6 :=
by 
  have h_eq : x^2 = 18 + x,
  { rw h, exact pow_two (Real.sqrt (18 + x)) },
  have quad_eq : x^2 - x - 18 = 0,
  { linarith [h_eq] },
  have factored : (x - 6) * (x + 3) = x^2 - x - 18,
  { ring },
  rw [←quad_eq, factored] at h,
  sorry

end nested_radical_eq_6_l169_169659


namespace possible_values_of_f2011_l169_169381

noncomputable theory

open Classical

-- Define the function property
def f_property (f : ℤ → ℝ) : Prop :=
  ∀ a b : ℤ, f (a + b) = 1 / f a + 1 / f b

-- State the problem in Lean
theorem possible_values_of_f2011 (f : ℤ → ℝ) (h : f_property f) :
  f 2011 = sqrt 2 ∨ f 2011 = -sqrt 2 :=
sorry

end possible_values_of_f2011_l169_169381


namespace rational_range_l169_169060

theorem rational_range (a : ℚ) (h : a - |a| = 2 * a) : a ≤ 0 := 
sorry

end rational_range_l169_169060


namespace movie_of_the_year_condition_l169_169225

theorem movie_of_the_year_condition : 
  let total_lists := 795 in
  let fraction := 1 / 4 in
  let required_lists := Nat.ceil (fraction * total_lists) in
  required_lists = 199 :=
by
  sorry

end movie_of_the_year_condition_l169_169225


namespace distance_to_cemetery_l169_169848

-- Definitions from the problem conditions
def increased_speed_by_one_fifth (v : ℝ) : ℝ := v + v / 5
def increased_speed_by_one_third (v : ℝ) : ℝ := v + v / 3

-- Given conditions
variables (T : ℝ) (v : ℝ) (D : ℝ)
variable h1 : T - 5 / 6 * T = 10 / 60  -- 10 minutes earlier when speed increased by 1/5 after 1 hour
variable h2 : D = v * T                 -- total distance is speed times time

-- Prove the total distance D is 180 km
theorem distance_to_cemetery (v : ℝ) (D : ℝ) :
  (T - 5 / 6 * T = 10 / 60) ∧
  (D = 60 / (v * (2/3))) ∧
  (T = 2) →
  D = 180 :=
begin
  sorry,
end

end distance_to_cemetery_l169_169848


namespace integer_part_4S_l169_169825

theorem integer_part_4S (S : ℝ) 
  (hS : S = ∑ k in Finset.range (2011+1), 1 / (k+1)^3) : 
  ⌊4 * S⌋ = 4 := by
  sorry

end integer_part_4S_l169_169825


namespace paint_cells_impossible_l169_169083

def num_colors : ℕ := 8
def num_color_pairs : ℕ := (num_colors * (num_colors - 1)) / 2
def num_adjacent_pairs : ℕ := 24

theorem paint_cells_impossible :
  num_color_pairs > num_adjacent_pairs →
  ∀ (table : Matrix (Fin 4) (Fin 4) (Fin num_colors)), ∃ (c1 c2 : Fin num_colors), ¬ (∃ (r1 r2 : Fin 4) (c1 c2 : Fin 4), (abs ((r1 - r2 : Int).natAbs) + abs ((c1 - c2 : Int).natAbs) = 1) ∧ (table ⟨r1, h1⟩ = c1) ∧ (table ⟨r2, h2⟩ = c2)) :=
begin
  sorry
end

end paint_cells_impossible_l169_169083


namespace wine_remaining_l169_169573

-- Define the variables and main theorem
variables (v : ℝ) (n : ℕ) (a : ℕ → ℝ)

-- Theorem stating the amount of wine remaining after n-th extraction
theorem wine_remaining (h_v_pos : v > 0) (h_a_le_v : ∀ i, i < n → a i ≤ v) :
  let b : ℕ → ℝ := λ k, 
    if k = 0 then v 
    else (b (k - 1) * (v - a k) / v) 
  in 
  b n = (∏ i in finrange n, (v - a i)) / v^(n-1) :=
by 
  sorry

end wine_remaining_l169_169573


namespace number_and_sum_of_possible_g1_values_l169_169821

noncomputable def g (x : ℝ) : ℝ := sorry

axiom functional_eq (x y : ℝ) :
  g((x + y)^2) = g(x)^2 - 2 * x * g(y) + 2 * y^2

theorem number_and_sum_of_possible_g1_values :
  let g1 := g 1 in
  g1 = Real.sqrt 2 ∧ 
  (∃! a : ℝ, a = g1) ∧ 
  (∀ b : ℝ, b = g1 → b = Real.sqrt 2) :=
by
  sorry

end number_and_sum_of_possible_g1_values_l169_169821


namespace arrangement_condition_l169_169227

theorem arrangement_condition 
  (k n r : ℕ) 
  (h1 : 0 < k) 
  (h2 : 0 < n) 
  (h3 : 0 < r) 
  (h4 : r < n) 
  (h5 : 2 * (k * n + r) = (k * n + r) + (k * n + r)) : 
  (∃ (seq : list ℕ), (∀ i : ℕ, i < 2*(k*n + r) - 2*n → let segment := seq.drop i in 
    ∃ (b w : ℕ), b ≠ w ∧ b + w = 2 * n) ↔ (r ≥ k ∧ n > k + r)) := 
sorry

end arrangement_condition_l169_169227


namespace perimeter_of_square_d_l169_169872

noncomputable def square_d_perimeter : ℝ :=
  let side_length_c : ℝ := 32 / 4
  let area_c : ℝ := side_length_c ^ 2
  let area_d : ℝ := area_c / 3
  let side_length_d : ℝ := real.sqrt (area_d)
  4 * side_length_d

theorem perimeter_of_square_d : square_d_perimeter = 32 * real.sqrt 3 / 3 := by
  sorry

end perimeter_of_square_d_l169_169872


namespace determine_vertices_l169_169850

structure PathGraph (n : ℕ) :=
(vertices : Finset ℕ)
(edges : Finset (ℕ × ℕ))
(degree : ℕ → ℕ)
(isPath : ∀ v ∈ vertices, degree v = 1 ∨ degree v = 2)

def reachable (U : PathGraph) (start : ℕ) (steps : ℕ) : Prop := sorry

theorem determine_vertices (U : PathGraph) (start : ℕ) (steps : ℕ)
  (h : reachable U start steps) : ∃ n : ℕ, n = U.vertices.card :=
sorry

end determine_vertices_l169_169850


namespace cylinder_height_eq_diameter_l169_169149

noncomputable def volume_cylinder (r h : ℝ) : ℝ :=
  π * r^2 * h

theorem cylinder_height_eq_diameter (h : ℝ) (r : ℝ) (V : ℝ) : 
  h = 2 * r ∧ volume_cylinder r h = 16 * π → h = 4 :=
by
  intro hyp
  have hr : r^3 = 8 := by 
    rw [<-hyp.2, volume_cylinder, hyp.1]
    sorry
  have r_eq_2 : r = 2 := by
    -- Prove that r = 2 from r^3 = 8
    sorry
  have h_eq_4 : h = 4 := by
    rw [hyp.1, r_eq_2]
    sorry
  exact h_eq_4

end cylinder_height_eq_diameter_l169_169149


namespace inscribed_quadrilateral_exists_l169_169556

theorem inscribed_quadrilateral_exists 
  (a b c : ℝ) 
  (α : ℝ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (hα : 0 < α ∧ α < 180) :
  ∃ A B C D : Point,
    is_quadrilateral A B C D ∧
    side_length A B = a ∧
    side_length B C = b ∧
    side_length C D = c ∧
    angle_between A B C = α :=
  sorry

end inscribed_quadrilateral_exists_l169_169556


namespace number_of_subsets_of_M_is_2_l169_169901

-- Define the set M using the given condition
def M : Set ℕ := {x | x * (x + 2) ≤ 0}

-- The goal is to prove that the number of subsets of M is 2
theorem number_of_subsets_of_M_is_2 : Fintype.card (Set ℕ) = 2 :=
sorry

end number_of_subsets_of_M_is_2_l169_169901


namespace bipyramid_total_volume_l169_169281

-- Definitions of geometric parameters based on the given problem
def a : ℝ := 2
def angle_APB : ℝ := Real.pi / 3
def side_length : ℝ := 2
def base_area : ℝ := 4
def height : ℝ := 2 / Real.sqrt 3

-- Statement of the theorem to be proved
theorem bipyramid_total_volume :
  let volume_of_one_pyramid := (1 / 3) * base_area * height in
  let total_volume := 2 * volume_of_one_pyramid in
  total_volume = 16 * Real.sqrt 3 / 9 :=
by
  sorry

end bipyramid_total_volume_l169_169281


namespace derivative_of_y_l169_169887

-- Define the function y
def y (x : ℝ) : ℝ := x / (1 - Real.cos x)

-- State the theorem
theorem derivative_of_y (x : ℝ) :
  deriv y x = (1 - Real.cos x - x * Real.sin x) / (1 - Real.cos x)^2 :=
by
  sorry

end derivative_of_y_l169_169887


namespace projection_is_q_l169_169371

def vector_1 : ℝ × ℝ := (5, 2)
def vector_2 : ℝ × ℝ := (2, 6)
def direction_vector : ℝ × ℝ := ( -3, 4)
def q : ℝ × ℝ := (104 / 25, 78 / 25)

theorem projection_is_q : 
  let line_param := λ t : ℝ, (vector_1.1 + t * direction_vector.1, vector_1.2 + t * direction_vector.2) in
  ∃ t : ℝ, line_param t = q ∧ (q.1 - vector_1.1) * direction_vector.1 + (q.2 - vector_1.2) * direction_vector.2 = 0 :=
sorry

end projection_is_q_l169_169371


namespace max_marks_l169_169264

theorem max_marks (M : ℝ) (h1 : 0.33 * M = 92 + 40) : M = 400 :=
by
  sorry

end max_marks_l169_169264


namespace solve_for_k_l169_169720

variable {x y k : ℝ}

theorem solve_for_k (h1 : 2 * x + y = 1) (h2 : x + 2 * y = k - 2) (h3 : x - y = 2) : k = 1 :=
by
  sorry

end solve_for_k_l169_169720


namespace chord_length_l169_169896

theorem chord_length (x y t : ℝ) (h₁ : x = 1 + 2 * t) (h₂ : y = 2 + t) (h_circle : x^2 + y^2 = 9) : 
  ∃ l, l = 12 / 5 * Real.sqrt 5 := 
sorry

end chord_length_l169_169896


namespace expectation_of_xi_l169_169519

noncomputable def xi_expectation : ℚ :=
  let P (n : ℕ) : ℚ := (Nat.CasesOn n (1/8)
    (Nat.CasesOn (n.pred)
      (3/8)
      (Nat.CasesOn (n.pred.pred)
        (3/8)
        (Nat.CasesOn (n.pred.pred.pred)
          (1/8)
          0)))
    0)
  in (0 * P 0 + 1 * P 1 + 2 * P 2 + 3 * P 3)

theorem expectation_of_xi :
  xi_expectation = 3/2 :=
by
  sorry

end expectation_of_xi_l169_169519


namespace exists_bijective_function_f_l169_169665

/-- 
  There exists a bijective function f: ℕ → ℕ that satisfies 
  the equation ∀ (m n : ℕ), f(3 * m * n + m + n) = 4 * f(m) * f(n) + f(m) + f(n).
-/
theorem exists_bijective_function_f :
  ∃ (f : ℕ → ℕ), function.bijective f ∧ 
    ∀ (m n : ℕ), f(3 * m * n + m + n) = 4 * f(m) * f(n) + f(m) + f(n) :=
sorry

end exists_bijective_function_f_l169_169665


namespace range_of_BC_in_triangle_l169_169057

theorem range_of_BC_in_triangle 
  (A B C : ℝ) 
  (a c BC : ℝ)
  (h1 : c = Real.sqrt 2)
  (h2 : a * Real.cos C = c * Real.sin A)
  (h3 : 0 < C ∧ C < Real.pi)
  (h4 : BC = 2 * Real.sin A)
  (h5 : ∃ A1 A2, 0 < A1 ∧ A1 < Real.pi / 2 ∧ Real.pi / 2 < A2 ∧ A2 < Real.pi ∧ Real.sin A = Real.sin A1 ∧ Real.sin A = Real.sin A2)
  : BC ∈ Set.Ioo (Real.sqrt 2) 2 :=
sorry

end range_of_BC_in_triangle_l169_169057


namespace AT_eq_TB_l169_169447

open Classical
noncomputable theory

-- Definitions of the geometric conditions
variables {Point : Type} [EuclideanPoint Point]

structure Circle (Point : Type) :=
(center : Point)
(radius : ℝ)

structure Line (Point : Type) :=
(points : set Point)
(is_line : ∀ p1 p2 ∈ points, p1 ≠ p2 → ∃ l, (l p1) ∧ (l p2))

variable {k1 k2 : Circle Point}
variable {l : Line Point}
variable {A B C D T : Point}

-- Given conditions
axiom k1_circ : Circle Point
axiom l_line : Line Point
axiom A_B_on_k1 : A ∈ k1_circ ∧ B ∈ k1_circ ∧ A ≠ B
axiom k2_circ : Circle Point
axiom k2_touches_k1_at_C : let C := Point, C ∈ k1_circ → C ∈ k2_circ
axiom k2_touches_l_at_D : let D := Point, D ∈ l_line.points
axiom T_second_intersection_k1_CD : let T := Point, T ≠ C → ∃ T, (T ∈ k1_circ ∧ T ∈ {p : Point | ∃ q, (q ∈ (↑l_line).points ∧ p = q)})

-- Theorem to prove
theorem AT_eq_TB : (A_T : ℝ) = (T_B : ℝ) := sorry -- This is left as an exercise.

end AT_eq_TB_l169_169447


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169980

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169980


namespace interior_angle_heptagon_l169_169207

theorem interior_angle_heptagon : 
  ∀ (n : ℕ), n = 7 → (5 * 180 / n : ℝ) = 128.57142857142858 :=
by
  intros n hn
  rw hn
  -- The proof is skipped
  sorry

end interior_angle_heptagon_l169_169207


namespace probability_A_does_not_lose_l169_169189

theorem probability_A_does_not_lose (p_tie p_A_win : ℚ) (h_tie : p_tie = 1 / 2) (h_A_win : p_A_win = 1 / 3) :
  p_tie + p_A_win = 5 / 6 :=
by sorry

end probability_A_does_not_lose_l169_169189


namespace width_of_prism_l169_169056

theorem width_of_prism (w : ℝ) : 
  (∃ (w : ℝ), (w^2 + 89 = 100)) → w = sqrt 11 :=
by
  sorry

end width_of_prism_l169_169056


namespace surface_area_correct_l169_169671

def w := 3 -- width in cm
def l := 4 -- length in cm
def h := 5 -- height in cm

def surface_area : Nat := 
  2 * (h * w) + 2 * (l * w) + 2 * (l * h)

theorem surface_area_correct : surface_area = 94 := 
  by
    sorry

end surface_area_correct_l169_169671


namespace compare_abc_l169_169334

theorem compare_abc (a b c : ℝ) (h1 : a = Real.sqrt 2)
                             (h2 : b = 2 ^ 0.8)
                             (h3 : c = 2 * Real.logb 5 2) : c < a ∧ a < b := 
by {
  sorry
}

end compare_abc_l169_169334


namespace construct_80_construct_160_construct_20_l169_169924

-- Define the notion of constructibility from an angle
inductive Constructible : ℝ → Prop
| base (a : ℝ) : a = 40 → Constructible a
| add (a b : ℝ) : Constructible a → Constructible b → Constructible (a + b)
| sub (a b : ℝ) : Constructible a → Constructible b → Constructible (a - b)

-- Lean statements for proving the constructibility
theorem construct_80 : Constructible 80 :=
sorry

theorem construct_160 : Constructible 160 :=
sorry

theorem construct_20 : Constructible 20 :=
sorry

end construct_80_construct_160_construct_20_l169_169924


namespace rope_purchases_l169_169093

theorem rope_purchases (last_week_rope_feet : ℕ) (less_rope : ℕ) (feet_to_inches : ℕ) 
  (h1 : last_week_rope_feet = 6) 
  (h2 : less_rope = 4) 
  (h3 : feet_to_inches = 12) : 
  (last_week_rope_feet * feet_to_inches) + ((last_week_rope_feet - less_rope) * feet_to_inches) = 96 := 
by
  sorry

end rope_purchases_l169_169093


namespace cos_sum_seventh_root_of_unity_l169_169683

theorem cos_sum_seventh_root_of_unity (z : ℂ) (α : ℝ) 
  (h1 : z ^ 7 = 1) (h2 : z ≠ 1) (h3 : ∃ k : ℤ, α = (2 * k * π) / 7 ) :
  (Real.cos α + Real.cos (2 * α) + Real.cos (4 * α)) = -1 / 2 :=
by 
  sorry

end cos_sum_seventh_root_of_unity_l169_169683


namespace tangent_angle_expression_correct_l169_169357

noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3

theorem tangent_angle_expression_correct :
  let α := Real.arctan (f' 1) in
  (sin α ^ 2 - cos α ^ 2) / (2 * sin α * cos α + cos α ^ 2) = 3 / 5 :=
by {
  sorry,
}

end tangent_angle_expression_correct_l169_169357


namespace arithmetic_sequence_of_square_negative_one_sequence_kn_sequence_possible_constant_sequence_if_both_l169_169005

def equal_square_difference_sequence (a : ℕ → ℝ) (p : ℝ) : Prop :=
∀ n, n ≥ 2 → (a n)^2 - (a (n - 1))^2 = p

theorem arithmetic_sequence_of_square {a : ℕ → ℝ} {p : ℝ} :
equal_square_difference_sequence a p → 
∃ c d, ∀ n, a n = c + n * d :=
sorry

theorem negative_one_sequence : 
equal_square_difference_sequence (λ n, (-1)^n) 0 :=
sorry

theorem kn_sequence_possible {a : ℕ → ℝ} {p : ℝ} (k : ℕ) :
equal_square_difference_sequence a p → 
equal_square_difference_sequence (λ n, a (k * n)) p :=
sorry

theorem constant_sequence_if_both {a : ℕ → ℝ} {p d : ℝ} :
equal_square_difference_sequence a p ∧
(∃ c d, ∀ n, a n = c + n * d) → 
(∀ n, a n = a 2) :=
sorry

end arithmetic_sequence_of_square_negative_one_sequence_kn_sequence_possible_constant_sequence_if_both_l169_169005


namespace pens_bought_l169_169457

-- Define the given conditions
def num_notebooks : ℕ := 10
def cost_per_pen : ℕ := 2
def total_paid : ℕ := 30
def cost_per_notebook : ℕ := 0  -- Assumption that notebooks are free

-- Converted condition that 10N + 2P = 30 and N = 0
def equation (N P : ℕ) : Prop := (10 * N + 2 * P = total_paid)

-- Statement to prove that if notebooks are free, 15 pens were bought
theorem pens_bought (N : ℕ) (P : ℕ) (hN : N = cost_per_notebook) (h : equation N P) : P = 15 :=
by sorry

end pens_bought_l169_169457


namespace geometric_sum_ratio_l169_169406

noncomputable def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (finset.range n).sum a

theorem geometric_sum_ratio (a : ℕ → ℝ) (q : ℝ)
  (h₀ : is_geometric_sequence a q)
  (h₁ : 6 * a 7 = (a 8 + a 9) / 2) :
  sum_first_n_terms a 6 / sum_first_n_terms a 3 = 28 :=
by
  sorry

end geometric_sum_ratio_l169_169406


namespace count_ordered_pairs_l169_169050

theorem count_ordered_pairs : 
  let count := (λ b : ℕ, ∃ a : ℝ, a > 0 ∧ (log b a)^2023 = log b (a^2023))
  (1 ≤ b ∧ b ≤ 201) in 
  (∑ b in finset.Icc 1 201, if count b then 1 else 0) = 603 := 
sorry

end count_ordered_pairs_l169_169050


namespace relationship_among_a_b_c_l169_169335

noncomputable def a : ℝ := (3 : ℝ) ^ (-1 / 2)
noncomputable def b : ℝ := Real.log (1 / 2) / Real.log 3
noncomputable def c : ℝ := Real.log 3 / Real.log 2

theorem relationship_among_a_b_c : c > a ∧ a > b :=
by { 
    sorry 
}

end relationship_among_a_b_c_l169_169335


namespace union_of_sets_l169_169113

def A (x : ℤ) : Set ℤ := {x^2, 2*x - 1, -4}
def B (x : ℤ) : Set ℤ := {x - 5, 1 - x, 9}

theorem union_of_sets (x : ℤ) (hx : x = -3) (h_inter : A x ∩ B x = {9}) :
  A x ∪ B x = {-8, -4, 4, -7, 9} :=
by
  sorry

end union_of_sets_l169_169113


namespace distance_O_B_l169_169079

def O : ℝ × ℝ × ℝ := (0, 0, 0)
def A : ℝ × ℝ × ℝ := (1, 2, 3)
def B : ℝ × ℝ × ℝ := (1, 0, 3)

def distance (P Q : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2 + (P.3 - Q.3)^2)

theorem distance_O_B : distance O B = Real.sqrt 10 := by
  sorry

end distance_O_B_l169_169079


namespace coeff_sq_term_l169_169284

noncomputable def S (n : ℕ) : ℚ := (n + 1) / (2 * ((n - 1)!))

noncomputable def T (n : ℕ) : ℚ := sorry

theorem coeff_sq_term (n : ℕ) (hn : 2 ≤ n) : 
  (T n) / (S n) = (1/4 : ℚ) * n^2 - (1/12 : ℚ) * n - (1/6 : ℚ) := 
sorry

end coeff_sq_term_l169_169284


namespace squares_in_figure_100_l169_169219

def f : ℕ → ℕ
| 0 := 1
| 1 := 7
| 2 := 19
| 3 := 37
| n := 3 * n * n + 3 * n + 1

theorem squares_in_figure_100 : f 100 = 30301 :=
sorry

end squares_in_figure_100_l169_169219


namespace solution_l169_169828

def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def is_even (g : ℝ → ℝ) : Prop :=
  ∀ y : ℝ, g (-y) = g y

def problem (f g : ℝ → ℝ) : Prop :=
  (∀ x y : ℝ, f (x + y) + f (x - y) = 2 * f x * g y) ∧
  (f 0 = 0) ∧
  (∃ x : ℝ, f x ≠ 0)

theorem solution (f g : ℝ → ℝ) (h : problem f g) : is_odd f ∧ is_even g :=
sorry

end solution_l169_169828


namespace proposition_d_is_false_l169_169269

theorem proposition_d_is_false (Q : Type) [Quadrilateral Q]
    (has_one_pair_of_parallel_sides : has_one_pair_of_parallel_sides Q)
    (has_one_pair_of_equal_sides : has_one_pair_of_equal_sides Q) : ¬ is_parallelogram Q := by
  sorry

end proposition_d_is_false_l169_169269


namespace quadratic_roots_inequality_l169_169747

-- Define the conditions given in the problem
def quadratic_inequality (a b : ℝ) := ∀ x : ℝ, ax^2 + bx + 2 < 0 ↔ (x < -1/2 ∨ x > 1/3)

-- State the problem as a theorem in Lean
theorem quadratic_roots_inequality (a b : ℝ) (h : quadratic_inequality a b) :
  (a ≠ 0) → (b ≠ 0) → (a - b) / a = 5 / 6 :=
by
  sorry

end quadratic_roots_inequality_l169_169747


namespace find_f_and_its_monotonicity_l169_169110

def evenFunctionOnNegOneToOne := ∀ x : ℝ, x ∈ [-1, 1] → f (-x) = f x
def symmetricWithRespectToLineOne := ∀ x y : ℝ, g (1 - x) = f x

noncomputable def g (x: ℝ) : ℝ := 2 * a * (x - 2) - 3 * (x - 2) ^ 3
def a_greater_than_9_div_2 (a : ℝ) := a > 9 / 2

def f (x: ℝ) : ℝ :=
if x ≤ 0 then
  3 * x^3 - 2 * a * x
else
  -3 * x^3 + 2 * a * x

theorem find_f_and_its_monotonicity
  (f_is_even: evenFunctionOnNegOneToOne)
  (f_symmetric: symmetricWithRespectToLineOne)
  (g_form : ∀ x, 2 ≤ x ∧ x ≤ 3 → g x = 2 * a * (x - 2) - 3 * (x - 2) ^ 3)
  (a_bound: a_greater_than_9_div_2 a) :
  (∀ x, x ∈ [-1, 0] ∨ x ∈ (0, 1] →
    (f x = (if x ≤ 0 then 3 * x ^ 3 - 2 * a * x else -3 * x ^ 3 + 2 * a * x)))
  ∧ (∀ x1 x2, x1 < x2 ∧ ((x1 ∈ [-1, 0] ∧ x2 ∈ [-1, 0]) ∨ (x1 ∈ (0, 1] ∧ x2 ∈ (0, 1])) → f x1 > f x2) :=
sorry

end find_f_and_its_monotonicity_l169_169110


namespace IM_eq_IN_l169_169104

theorem IM_eq_IN 
  (A B C D I P M E F N : Type)
  (BC AD AM PB PC AC : A → B → C)
  (hI_on_BC : I ∈ BC)
  (hP_on_AD : P ∈ AD)
  (hLine_through_I : line_through I)
  (hLine_intersects_M : intersects AB PB line_through I M E)
  (hLine_intersects_F : intersects AC PC line_through I F N)
  (hDE_eq_DF : DE = DF)
  : IM = IN := sorry

end IM_eq_IN_l169_169104


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169981

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169981


namespace ordered_pair_l169_169838

-- Define variables and conditions
variables {V : Type*} [AddCommGroup V] [Module ℝ V]
variables (A B Q : V)
variables (a b : ℝ)

-- Given conditions
def seg_eq_ratio (a b : ℝ) : Prop := AQ / QB = a / b

-- Statement to prove
theorem ordered_pair (h : seg_eq_ratio 7 2):
  ∃ (x y : ℝ), Q = x • A + y • B ∧ (x, y) = ((2:ℝ)/9, (7:ℝ)/9) :=
sorry

end ordered_pair_l169_169838


namespace greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169979

theorem greatest_possible_sum_of_consecutive_integers_product_less_500 :
  ∃ (n : ℤ), (n * (n + 1) < 500) ∧ (∀ (m : ℤ), (m * (m + 1) < 500) → (n + (n + 1) ≥ m + (m + 1))) :=
begin
  sorry
end

end greatest_possible_sum_of_consecutive_integers_product_less_500_l169_169979


namespace number_of_ways_to_choose_positions_l169_169067

-- Definition of the problem conditions
def number_of_people : ℕ := 8

-- Statement of the proof problem
theorem number_of_ways_to_choose_positions : 
  (number_of_people) * (number_of_people - 1) * (number_of_people - 2) = 336 := by
  -- skipping the proof itself
  sorry

end number_of_ways_to_choose_positions_l169_169067


namespace final_score_for_5_hours_l169_169604

-- 1. Direct proportion condition
def grade (t : ℝ) : ℝ := 45 * t

-- 2. The 10% bonus condition
def final_score_with_bonus (G : ℝ) : ℝ := G * 1.10

-- 3. The student's grade based on given conditions
def grade_for_2_hours : ℝ := grade 2 = 90
def grade_for_5_hours : ℝ := grade 5

-- The theorem statement
theorem final_score_for_5_hours (h1 : grade_for_2_hours) : final_score_with_bonus (grade_for_5_hours) = 247.5 :=
by
  sorry

end final_score_for_5_hours_l169_169604


namespace students_on_Korabelnaya_Street_l169_169512

theorem students_on_Korabelnaya_Street (n : ℕ) (h1 : n < 50) 
  (h2 : (n / 7 + n / 3 + n / 2 = n - 1)) : n ∣ 42 :=
begin
  sorry
end

end students_on_Korabelnaya_Street_l169_169512


namespace kenny_played_basketball_for_10_hours_l169_169091

noncomputable def kenny_hours_basketball : ℕ :=
  let trumpet_hours := 40 in
  let run_hours := trumpet_hours / 2 in
  let basketball_hours := run_hours / 2 in
  basketball_hours

theorem kenny_played_basketball_for_10_hours :
  kenny_hours_basketball = 10 :=
by
  unfold kenny_hours_basketball
  sorry

end kenny_played_basketball_for_10_hours_l169_169091


namespace one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169541

theorem one_hundred_fiftieth_digit_of_3_div_11_is_7 :
  let decimal_repetition := "27"
  let length := 2
  150 % length = 0 →
  (decimal_repetition[1] = '7')
: sorry

end one_hundred_fiftieth_digit_of_3_div_11_is_7_l169_169541


namespace race_finish_difference_l169_169117

theorem race_finish_difference :
  ∀ (distance speed_malcolm speed_joshua : ℕ),
    speed_malcolm = 5 →
    speed_joshua = 7 →
    distance = 12 →
    (distance * speed_joshua - distance * speed_malcolm) = 24 :=
by
  intros distance speed_malcolm speed_joshua h_speed_malcolm h_speed_joshua h_distance
  simp [h_speed_malcolm, h_speed_joshua, h_distance]
  sorry

end race_finish_difference_l169_169117


namespace ribbon_length_l169_169147

theorem ribbon_length (a b : ℝ) (h : a = 10) (k : b = 15) : 
  let base_top_length := 6 * a,
      lateral_length := 3 * b,
      total_length := base_top_length + lateral_length
  in total_length = 105 := by
  -- conditions
  have h1 : base_top_length = 60, by rw [<- h, show 6 * 10 = 60, by norm_num],
  have h2 : lateral_length = 45, by rw [<- k, show 3 * 15 = 45, by norm_num],
  -- conclusion
  show base_top_length + lateral_length = 105, by rw [h1, h2, show 60 + 45 = 105, by norm_num]


end ribbon_length_l169_169147


namespace shopkeeper_loss_percentage_l169_169262

theorem shopkeeper_loss_percentage
  (total_stock_value : ℝ)
  (overall_loss : ℝ)
  (first_part_percentage : ℝ)
  (first_part_profit_percentage : ℝ)
  (remaining_part_loss : ℝ)
  (total_worth_first_part : ℝ)
  (first_part_profit : ℝ)
  (remaining_stock_value : ℝ)
  (remaining_stock_loss : ℝ)
  (loss_percentage : ℝ) :
  total_stock_value = 16000 →
  overall_loss = 400 →
  first_part_percentage = 0.10 →
  first_part_profit_percentage = 0.20 →
  total_worth_first_part = total_stock_value * first_part_percentage →
  first_part_profit = total_worth_first_part * first_part_profit_percentage →
  remaining_stock_value = total_stock_value * (1 - first_part_percentage) →
  remaining_stock_loss = overall_loss + first_part_profit →
  loss_percentage = (remaining_stock_loss / remaining_stock_value) * 100 →
  loss_percentage = 5 :=
by intros; sorry

end shopkeeper_loss_percentage_l169_169262


namespace cost_of_limes_after_30_days_l169_169783

def lime_juice_per_mocktail : ℝ := 1  -- tablespoons per mocktail
def days : ℕ := 30  -- number of days
def lime_juice_per_lime : ℝ := 2  -- tablespoons per lime
def limes_per_dollar : ℝ := 3  -- limes per dollar

theorem cost_of_limes_after_30_days : 
  let total_lime_juice := (lime_juice_per_mocktail * days),
      number_of_limes  := (total_lime_juice / lime_juice_per_lime),
      total_cost       := (number_of_limes / limes_per_dollar)
  in total_cost = 5 :=
by
  sorry

end cost_of_limes_after_30_days_l169_169783


namespace vector_dot_product_parallel_l169_169035

theorem vector_dot_product_parallel (m : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h_a : a = (1, 2))
  (h_b : b = (m, -4))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  (a.1 * b.1 + a.2 * b.2) = -10 := by
  sorry

end vector_dot_product_parallel_l169_169035


namespace eccentricity_of_ellipse_is_one_seventh_l169_169034

theorem eccentricity_of_ellipse_is_one_seventh
  (c : ℝ) (a b : ℝ) 
  (h_focus_parabola : c = 1 / 4) 
  (h_focus_ellipse : c ^ 2 + b ^ 2 = a ^ 2) 
  (h_b_value : b = sqrt 3) 
  (h_a_value : a = sqrt ((1 / 4) ^ 2 + 3)) : 
  (c / a = 1 / 7) :=
by
  sorry

end eccentricity_of_ellipse_is_one_seventh_l169_169034


namespace find_y_l169_169762

variables (BAC ABC ACB CDE ADE AED DEB : ℝ)

theorem find_y (h1 : BAC = 45)
               (h2 : ABC = 45)
               (h3 : ACB = 90)
               (h4 : CDE = 72)
               (h5 : ADE + DEB = 180) :
  DEB = 153 :=
by
  calc
  DEB = 180 - AED : by sorry
     ... = 180 - (180 - ADE - BAC) : by sorry
     ... = 180 - (180 - 108 - 45) : by sorry
     ... = 180 - 27 : by sorry
     ... = 153 : by sorry

end find_y_l169_169762


namespace average_age_of_town_l169_169399

-- Definitions based on conditions
def ratio_of_women_to_men (nw nm : ℕ) : Prop := nw * 8 = nm * 9

def young_men (nm : ℕ) (n_young_men : ℕ) (average_age_young : ℕ) : Prop :=
  n_young_men = 40 ∧ average_age_young = 25

def remaining_men_average_age (nm n_young_men : ℕ) (average_age_remaining : ℕ) : Prop :=
  average_age_remaining = 35

def women_average_age (average_age_women : ℕ) : Prop :=
  average_age_women = 30

-- Complete problem statement we need to prove
theorem average_age_of_town (nw nm : ℕ) (total_avg_age : ℕ) :
  ratio_of_women_to_men nw nm →
  young_men nm 40 25 →
  remaining_men_average_age nm 40 35 →
  women_average_age 30 →
  total_avg_age = 32 * 17 + 6 :=
sorry

end average_age_of_town_l169_169399


namespace quadratic_to_vertex_form_l169_169311

theorem quadratic_to_vertex_form : ∃ m n : ℝ, (∀ x : ℝ, x^2 - 8*x + 3 = 0 ↔ (x - m)^2 = n) ∧ m + n = 17 :=
by sorry

end quadratic_to_vertex_form_l169_169311


namespace tank_total_weight_l169_169775

-- Definition for the condition of height and diameter of cylindrical tank
def tank_radius : ℝ := 4 / 2 -- Given diameter of tank is 4 feet
def tank_height : ℝ := 10

-- Definition of volumes in terms of feet and gallons
def full_tank_vol_cubic_feet : ℝ := real.pi * tank_radius^2 * tank_height
def full_tank_vol_gallons : ℝ := full_tank_vol_cubic_feet * 7.48

-- Given tank capacity in gallons
def tank_capacity_gallons : ℝ := 200
def tank_empty_weight_pounds : ℝ := 80
def mix_ratio_X : ℝ := 3
def mix_ratio_water : ℝ := 7
def total_mix_ratio : ℝ := mix_ratio_X + mix_ratio_water

-- Definitions for weights per gallon
def weight_per_gallon_water : ℝ := 8
def weight_per_gallon_liquid_X : ℝ := 12

-- Percentage to fill the tank
def fill_percentage : ℝ := 0.6
def filled_volume_gallons : ℝ := fill_percentage * tank_capacity_gallons

-- Definitions for proportions of liquids
def gallons_liquid_X : ℝ := filled_volume_gallons * (mix_ratio_X / total_mix_ratio)
def gallons_water : ℝ := filled_volume_gallons * (mix_ratio_water / total_mix_ratio)

-- Determine the total weight of the liquid in the tank
def weight_liquid_X : ℝ := gallons_liquid_X * weight_per_gallon_liquid_X
def weight_water : ℝ := gallons_water * weight_per_gallon_water
def total_liquid_weight : ℝ := weight_liquid_X + weight_water

-- Prove total weight of the tank
theorem tank_total_weight : tank_empty_weight_pounds + total_liquid_weight = 1184 := by
  sorry

end tank_total_weight_l169_169775


namespace problem1_problem2_l169_169705

-- Definitions and conditions
def arithmetic_seq (a_n : ℕ → ℤ) (a1 d : ℤ) : Prop :=
  ∀ n, a_n n = a1 + (n - 1) * d

def b_seq (a_n b_n : ℕ → ℤ) : Prop :=
  ∀ n, (n % 2 = 1 ∧ b_n n = a_n n + 5) ∨ (n % 2 = 0 ∧ b_n n = -a_n n + 10)

def T_n (b_n : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ i in range (n + 1), b_n i

variables {a_n b_n : ℕ → ℤ} {a1 d : ℤ}

-- Given conditions
axiom arithmetic_seq_given : arithmetic_seq a_n a1 d
axiom b_seq_given : b_seq a_n b_n
axiom T3_eq_24 : T_n b_n 2 = 24
axiom T4_eq_24 : T_n b_n 3 = 24

-- To be proved
theorem problem1 : a1 + d = 4 → -a1 - 3 * d + 10 = 0 → ∀ n, a_n n = 3 * n - 2 :=
  sorry

theorem problem2 : (T_n b_n n = 96) → (T_n b_n m = 96) → (n < m) → n = 11 ∧ m = 16 :=
  sorry

end problem1_problem2_l169_169705


namespace minimum_sum_first_row_l169_169922

def grid := list (list ℕ)

def valid_grid (g : grid) : Prop :=
  (∀ (col : list ℕ), size col = 9) ∧ 
  (∀ (row ∈ g), ∀ (n : ℕ), n ∈ row → 1 ≤ n ∧ n ≤ 2004) ∧
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 2004 → (count n (flatten g) = 9)) ∧
  (∀ (i : ℕ) (h : i < 2004) (col := column i g), ∀ (n : ℕ), count n col ≤ 3)

-- We assume "column" and "flatten" are predefined functions where:
-- "flatten g" concatenates all rows into a single list.
-- "column i g" retrieves the ith column from the grid g.

theorem minimum_sum_first_row (g : grid) (h : valid_grid g) : 
  sum (nth g 0 []) = 2005004 :=
sorry

end minimum_sum_first_row_l169_169922


namespace base7_to_decimal_l169_169874

theorem base7_to_decimal (x y : ℕ) (h0 : 451 = 7 ^ 2 * 4 + 7 * 5 + 1)
    (h1 : 232 = 100 * x + 10 * y + 2) : 
  x * y / 10 = 0.6 := by
  -- Proof to be provided here
  sorry

end base7_to_decimal_l169_169874


namespace slope_passing_through_points_10_11_l169_169911

theorem slope_passing_through_points_10_11 (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (a1 : ℝ)
  (h1 : a 4 = 15) (h2 : S 5 = 55)
  (h3 : ∀ n, a n = a1 + (n - 1) * d)
  (h4 : ∀ n, S n = n * (a1 + a (n + 1)) / 2) :
  let a10 := a 10,
      a11 := a 11 in
  (a11 - a10) / (11 - 10) = 4 :=
by
  sorry

end slope_passing_through_points_10_11_l169_169911


namespace total_volume_of_spheres_in_prism_l169_169341

theorem total_volume_of_spheres_in_prism :
  let h := 3
  let θ := Real.pi / 3
  let volume := (4 * Real.pi / 3) * (∑ n in Finset.range (n + 1), (1 / 27) ^ n)
  h = 3 ∧ θ = Real.pi / 3 → volume = 18 * Real.pi / 13 :=
by
  sorry

end total_volume_of_spheres_in_prism_l169_169341


namespace arithmetic_sequence_sum_7_l169_169438

-- Define the arithmetic sequence and its properties
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n k : ℕ, n > 0 → a (n + k) = a n + k * (a 2 - a 1)

def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n * (a 1 + a n)) / 2

-- Given conditions translated to Lean
variables (a : ℕ → ℝ)
variable h_arithmetic : is_arithmetic a
variable h_condition : a 2 + a 4 = 9 - a 6

-- Theorem statement
theorem arithmetic_sequence_sum_7 : sum_first_n_terms a 7 = 21 :=
by
  sorry

end arithmetic_sequence_sum_7_l169_169438
