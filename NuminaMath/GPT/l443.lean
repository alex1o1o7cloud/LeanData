import Mathlib

namespace unpainted_area_five_inch_board_l443_443460

theorem unpainted_area_five_inch_board :
  ∀ (w1 w2 : ℝ) (θ : ℝ),
  w1 = 5 ∧ w2 = 6 ∧ θ = π/4 →
  let hypotenuse := w1 * real.sqrt 2 in
  let area := hypotenuse * w2 in
  area = 30 * real.sqrt 2 :=
by
  intros w1 w2 θ h,
  cases h with h_w1 h_rest,
  cases h_rest with h_w2 h_θ,
  rw [h_w1, h_w2, h_θ],
  let hypotenuse := 5 * real.sqrt 2,
  let area := hypotenuse * 6,
  have: area = 30 * real.sqrt 2 := by
    calc
      area = (5 * real.sqrt 2) * 6 : by rfl
           ... = 30 * real.sqrt 2    : by ring,
  exact this

end unpainted_area_five_inch_board_l443_443460


namespace smallest_five_digit_number_tens_place_l443_443815

theorem smallest_five_digit_number_tens_place :
  ∃ n : ℕ, 
  n = 12365 ∧ 
  (∀ m : ℕ, (digits m = [5, 6, 3, 2, 1] → m ≥ n)) ∧ 
  (∃ k : ℕ, m = 3 * k) ∧  
  (∃ l : ℕ, m = 5 * l) →
  digit_at_tens_place 12365 = 6 :=
sorry

end smallest_five_digit_number_tens_place_l443_443815


namespace ratio_AB_BC_l443_443641

variables (O A B C : Type)
variables [has_norm ℝ A] [has_norm ℝ B] [has_norm ℝ C] [has_norm ℝ O]
variables [has_scalar ℝ (O → ℝ)]

-- Non-collinear points
axiom non_collinear : ¬collinear O A B C

-- Given condition: OB = (1/3)OA + (2/3)OC
axiom OB_eq : ∀ (OB OA OC : O → ℝ), 
  OB = (1/3 : ℝ) • OA + (2/3 : ℝ) • OC

-- The ratio of |AB| to |BC| is 2:1
theorem ratio_AB_BC : ∀ (AB BC AC : O → ℝ), 
  ∥AB∥ / ∥BC∥ = 2 / 1 := sorry

end ratio_AB_BC_l443_443641


namespace contrapositive_eq_l443_443481

variables (P Q : Prop)

theorem contrapositive_eq : (¬P → Q) ↔ (¬Q → P) := 
by {
    sorry
}

end contrapositive_eq_l443_443481


namespace value_of_f_neg_t_l443_443631

def f (x : ℝ) : ℝ := 3 * x + Real.sin x + 1

theorem value_of_f_neg_t (t : ℝ) (h : f t = 2) : f (-t) = 0 :=
by
  sorry

end value_of_f_neg_t_l443_443631


namespace arc_length_correct_l443_443307

-- Define the radius and the angle in degrees
def radius : Real := 1
def angle_deg : Real := 60

-- Convert the angle from degrees to radians
def angle_rad (angle_deg : Real) : Real := angle_deg * (Real.pi / 180)

-- Define the length of the arc formula for a given radius and angle in radians
def arc_length (r : Real) (theta : Real) : Real := r * theta

-- Specify that the central angle in radians is π/3
def angle_rad_60 : Real := angle_rad 60

theorem arc_length_correct : arc_length radius angle_rad_60 = Real.pi / 3 := by
  sorry

end arc_length_correct_l443_443307


namespace remainder_when_sum_divided_by_11_l443_443993

def sum_of_large_numbers : ℕ :=
  100001 + 100002 + 100003 + 100004 + 100005 + 100006 + 100007

theorem remainder_when_sum_divided_by_11 : sum_of_large_numbers % 11 = 2 := by
  sorry

end remainder_when_sum_divided_by_11_l443_443993


namespace largest_average_l443_443484

theorem largest_average:
  let multiples (n : ℕ) (N : ℕ) := {x : ℕ | x > 0 ∧ x ≤ N ∧ x % n = 0}
  let average (s : Set ℕ) := (Set.sum s) / (Set.card s)
  average (multiples 10 150) > average (multiples 7 150)
  ∧ average (multiples 10 150) > average (multiples 8 150)
  ∧ average (multiples 10 150) > average (multiples 9 150)
  ∧ average (multiples 10 150) > average (multiples 11 150) :=
by
  sorry

end largest_average_l443_443484


namespace total_cost_is_83_50_l443_443076

-- Definitions according to the conditions
def cost_adult_ticket : ℝ := 5.50
def cost_child_ticket : ℝ := 3.50
def total_tickets : ℝ := 21
def adult_tickets : ℝ := 5
def child_tickets : ℝ := total_tickets - adult_tickets

-- Total cost calculation based on the conditions
def cost_adult_total : ℝ := adult_tickets * cost_adult_ticket
def cost_child_total : ℝ := child_tickets * cost_child_ticket
def total_cost : ℝ := cost_adult_total + cost_child_total

-- The theorem to prove that the total cost is $83.50
theorem total_cost_is_83_50 : total_cost = 83.50 := by
  sorry

end total_cost_is_83_50_l443_443076


namespace smallest_factor_of_32_not_8_l443_443226

theorem smallest_factor_of_32_not_8 : ∃ n : ℕ, n = 16 ∧ (n ∣ 32) ∧ ¬(n ∣ 8) ∧ ∀ m : ℕ, (m ∣ 32) ∧ ¬(m ∣ 8) → n ≤ m :=
by
  sorry

end smallest_factor_of_32_not_8_l443_443226


namespace milton_books_l443_443776

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l443_443776


namespace cos_phi_zero_of_odd_function_l443_443652

theorem cos_phi_zero_of_odd_function (φ : ℝ) (f : ℝ → ℝ) (h1 : ∀ x, f(x) = cos(2 * x + φ)) (h2 : ∀ x, f(-x) = -f(x)) : cos(φ) = 0 :=
by
  sorry

end cos_phi_zero_of_odd_function_l443_443652


namespace distinct_terms_in_expansion_l443_443683

theorem distinct_terms_in_expansion :
  let P1 := (x + y + z)
  let P2 := (u + v + w + x + y)
  ∃ n : ℕ, n = 14 ∧ 
    ∀ a b, 
      (a ∈ {x, y, z} ∧ b ∈ {u, v, w, x, y}) → 
      (a * b ∈ expansion_of P1 P2)
:= sorry

end distinct_terms_in_expansion_l443_443683


namespace product_of_repeating_decimal_l443_443574

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l443_443574


namespace distance_between_lines_l443_443415

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  abs (C2 - C1) / real.sqrt (A^2 + B^2)

theorem distance_between_lines : 
  distance_between_parallel_lines 3 4 (-12) 3 = 3 :=
by
  sorry

end distance_between_lines_l443_443415


namespace odd_and_monotonically_increasing_l443_443545

noncomputable def f (x : ℝ) : ℝ := exp x - exp (-x) - 2 * x

-- Definition of an odd function
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f (x)

-- Definition of a monotonically increasing function on an interval (a, b)
def is_monotonically_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → f x ≤ f y

theorem odd_and_monotonically_increasing :
  is_odd f ∧ is_monotonically_increasing_on f 0 +∞ :=
by
  sorry

end odd_and_monotonically_increasing_l443_443545


namespace inequality_proof_l443_443351

noncomputable def a : ℝ := Real.logBase π 3
noncomputable def b : ℝ := 2 ^ 0.3
noncomputable def c : ℝ := Real.logBase 3 (Real.sin (Real.pi / 6))

theorem inequality_proof : b > a ∧ a > c :=
by
  have h1 : 0 < a := sorry
  have h2 : a < 1 := sorry
  have h3 : b > 1 := sorry
  have h4 : c < 0 := sorry
  exact ⟨h3, h1, h2, h4⟩

end inequality_proof_l443_443351


namespace rectangle_area_l443_443139

variable (a b c : ℝ)

theorem rectangle_area (h : a^2 + b^2 = c^2) : a * b = area :=
by sorry

end rectangle_area_l443_443139


namespace infinite_n_exist_l443_443589

def S (n : ℕ) : ℕ := 
  n.digits 10 |>.sum

theorem infinite_n_exist (p : ℕ) [Fact (Nat.Prime p)] : 
  ∃ᶠ n in at_top, S n ≡ n [MOD p] :=
sorry

end infinite_n_exist_l443_443589


namespace isosceles_triangle_angles_l443_443741

noncomputable 
def is_triangle_ABC_isosceles (A B C : ℝ) (alpha beta : ℝ) (AB AC : ℝ) 
  (h1 : AB = AC) (h2 : alpha = 2 * beta) : Prop :=
  180 - 3 * beta = C ∧ C / 2 = 90 - 1.5 * beta

theorem isosceles_triangle_angles (A B C C1 C2 : ℝ) (alpha beta : ℝ) (AB AC : ℝ)
  (h1 : AB = AC) (h2 : alpha = 2 * beta) :
  (180 - 3 * beta) / 2 = 90 - 1.5 * beta :=
by sorry

end isosceles_triangle_angles_l443_443741


namespace product_of_repeating_decimal_l443_443577

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l443_443577


namespace find_integer_divisible_by_24_l443_443969

theorem find_integer_divisible_by_24 : 
  ∃ n : ℕ, (n % 24 = 0) ∧ (9 < real.sqrt (real.cbrt n)) ∧ (real.sqrt (real.cbrt n) < 9.1) := 
by
  let n := 744
  use n
  have h1 : n % 24 = 0 := by norm_num
  have h2 : 9 < real.sqrt (real.cbrt n) := by norm_num
  have h3 : real.sqrt (real.cbrt n) < 9.1 := by norm_num
  exact ⟨h1, h2, h3⟩

end find_integer_divisible_by_24_l443_443969


namespace arithmetic_sequence_term_difference_l443_443471

theorem arithmetic_sequence_term_difference :
  let d := 8
  let a₁ := -5
  ∀ n : ℕ, n > 0 →
  let a := λ n, a₁ + (n - 1) * d in
  |a 1508 - a 1500| = 64 :=
by
  let d := 8
  let a₁ := -5
  assume n _,
  let a := λ n, a₁ + (n - 1) * d,
  sorry

end arithmetic_sequence_term_difference_l443_443471


namespace surface_area_of_prism_l443_443540

-- Given conditions in Lean definitions
def diameter_of_sphere : ℝ := 2
def edge_length_of_base_of_prism : ℝ := 1

-- Main theorem: The statement to be proven.
theorem surface_area_of_prism : 
  ∃ (h : ℝ), (diameter_of_sphere = real.sqrt (edge_length_of_base_of_prism ^ 2 + edge_length_of_base_of_prism ^ 2 + h ^ 2)) ∧
  (2 * (edge_length_of_base_of_prism ^ 2) + 4 * edge_length_of_base_of_prism * h = 2 + 4 * real.sqrt 2) :=
begin
  sorry
end

end surface_area_of_prism_l443_443540


namespace area_swept_by_AP_l443_443750

def sin (x : ℝ) : ℝ := Real.sin x
def cos (x : ℝ) : ℝ := Real.cos x

def A : ℝ × ℝ := (2, 0)
def P (t : ℝ) : ℝ × ℝ := (sin (2 * t - Real.pi / 3), cos (2 * t - Real.pi / 3))

theorem area_swept_by_AP (t : ℝ) (h1 : t = Real.pi / 12) (h2 : t = Real.pi / 4) : 
  let swept_area := (Real.pi / 6) in
  swept_area = (Real.pi / 6) :=
by
  sorry

end area_swept_by_AP_l443_443750


namespace sum_of_complex_powers_l443_443560

theorem sum_of_complex_powers :
  let i := complex.I in
  let z := ∑ k in finset.range 2010, i ^ (k + 1) in
  z = i :=
by
  have h1 : i^2 = -1 := by simp [complex.I]
  have h2 : i^3 = -(complex.I) := by simp [complex.I]
  have h3 : i^4 = 1 := by simp [complex.I]
  sorry

end sum_of_complex_powers_l443_443560


namespace production_rate_equation_l443_443510

-- Original definition of the production rate
def original_production_rate (x : ℝ) : ℝ := x

-- New production rate after technology improvement
def new_production_rate (x : ℝ) : ℝ := x + 3

-- The time taken to produce 120 tons at the original rate
def time_to_produce_120 (x : ℝ) : ℝ := 120 / original_production_rate x

-- The time taken to produce 180 tons at the new rate
def time_to_produce_180 (x : ℝ) : ℝ := 180 / new_production_rate x

-- The proof problem: prove the times are equal
theorem production_rate_equation (x : ℝ) : time_to_produce_120 x = time_to_produce_180 x :=
by
  rw [time_to_produce_120, time_to_produce_180, original_production_rate, new_production_rate]
  sorry

end production_rate_equation_l443_443510


namespace diane_coins_in_third_piggy_bank_l443_443485

theorem diane_coins_in_third_piggy_bank :
  ∀ n1 n2 n4 n5 n6 : ℕ, n1 = 72 → n2 = 81 → n4 = 99 → n5 = 108 → n6 = 117 → (n4 - (n4 - 9)) = 90 :=
by
  -- sorry is needed to avoid an incomplete proof, as only the statement is required.
  sorry

end diane_coins_in_third_piggy_bank_l443_443485


namespace max_x_add_inv_x_value_l443_443064

noncomputable def max_x_add_inv_x (nums : List ℝ) (h_pos : ∀ x ∈ nums, x > 0)
  (h_sum : List.sum nums = 1010) (h_sum_inv : List.sum (nums.map (λ x, x⁻¹)) = 1010) : ℝ :=
  real.sup (λ x, x + x⁻¹) (set_of (λ x, x ∈ nums))

theorem max_x_add_inv_x_value :
  ∃ (nums : List ℝ) (h_pos : ∀ x ∈ nums, x > 0)
    (h_sum : List.sum nums = 1010) (h_sum_inv : List.sum (nums.map (λ x, x⁻¹)) = 1010),
    max_x_add_inv_x nums h_pos h_sum h_sum_inv = 2029 / 1010 := sorry

end max_x_add_inv_x_value_l443_443064


namespace smallest_m_l443_443393

open Set Topology MeasureTheory

-- Define the conditions of the problem
variables (x y z : ℝ) (m : ℕ)

-- Required conditions
def conditions (m : ℕ) :=
  (0 : ℝ) ≤ x ∧ x ≤ ↑m ∧
  (0 : ℝ) ≤ y ∧ y ≤ ↑m ∧
  (0 : ℝ) ≤ z ∧ z ≤ ↑m ∧
  (abs (x - y) ≥ 2) ∧ (abs (y - z) ≥ 2) ∧ (abs (z - x) ≥ 2) ∧
  (x + y + z ≥ ↑m)

-- Define the probability expression
def probability (m : ℕ) := 
  volume {p : ℝ × ℝ × ℝ | conditions m p.1 p.2 p.3} / volume (Icc (0 : ℝ) m) ^ 3

-- The statement of the problem
theorem smallest_m (m : ℕ) (h : probability m > (3 / 5 : ℝ)) : m = 30 :=
sorry

end smallest_m_l443_443393


namespace john_spent_l443_443748

/-- John bought 9.25 meters of cloth at a cost price of $44 per meter.
    Prove that the total amount John spent on the cloth is $407. -/
theorem john_spent :
  let length_of_cloth := 9.25
  let cost_per_meter := 44
  let total_cost := length_of_cloth * cost_per_meter
  total_cost = 407 := by
  sorry

end john_spent_l443_443748


namespace minimum_BC_length_l443_443500

theorem minimum_BC_length (AB AC DC BD BC : ℕ)
  (h₁ : AB = 5) (h₂ : AC = 12) (h₃ : DC = 8) (h₄ : BD = 20) (h₅ : BC > 12) : BC = 13 :=
by
  sorry

end minimum_BC_length_l443_443500


namespace amber_can_win_l443_443063

theorem amber_can_win (k : ℕ) (h : k > 0) : 
  (∀ m n, m < k → n < k → m ≠ n → -- any two distinct non-empty piles
  let pile := (fun (i: ℕ) => 2020 : ℕ) in 
  pile m = 0 ∨ pile n = 0) ↔ k = 4039 :=
sorry

end amber_can_win_l443_443063


namespace other_leg_length_l443_443142

theorem other_leg_length (a b c : ℕ) (h1 : a = 9) (h2 : c = 15) (h3 : c * c = a * a + b * b) : b = 12 :=
by
  have h_a : 9 * 9 = a * a, by rw ← h1; ring
  have h_c : 15 * 15 = c * c, by rw ← h2; ring
  rw h_a at h3
  rw h_c at h3
  linarith [h3]
  sorry

end other_leg_length_l443_443142


namespace simplify_factorial_expression_l443_443568

theorem simplify_factorial_expression (N : ℕ) (h : N ≥ 2) :
  (N-2)! * (N-1) * N / (N+2)! = 1 / ((N+2) * (N+1)) :=
by
  sorry

end simplify_factorial_expression_l443_443568


namespace perpendicular_vectors_parallel_vectors_l443_443671

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (2, x)
noncomputable def vector_b (x : ℝ) : ℝ × ℝ := (x - 1, 1)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem perpendicular_vectors (x : ℝ) :
  dot_product (vector_a x) (vector_b x) = 0 ↔ x = 2 / 3 :=
by sorry

theorem parallel_vectors (x : ℝ) :
  (2 / (x - 1) = x) ∨ (x - 1 = 0) ∨ (2 = 0) ↔ (x = 2 ∨ x = -1) :=
by sorry

end perpendicular_vectors_parallel_vectors_l443_443671


namespace clean_house_together_time_l443_443107

noncomputable def time_to_clean (house : Type) (John_rate Nick_rate Mary_rate : ℝ) : ℝ :=
  1 / (John_rate + Nick_rate + Mary_rate)

theorem clean_house_together_time :
  ∃ N : ℝ, ∃ John_rate Nick_rate Mary_rate combined_rate: ℝ,
    (John_rate = 1 / 6) ∧
    (Nick_rate = 1 / 9) ∧
    (Mary_rate = 1 / 11) ∧
    (combined_rate = John_rate + Nick_rate + Mary_rate) ∧
    time_to_clean ℝ John_rate Nick_rate Mary_rate = 198 / 73 :=
begin
  use [9, 1/6, 1/9, 1/11, (1/6 + 1/9 + 1/11)],
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { sorry },
end

end clean_house_together_time_l443_443107


namespace remainder_3005_98_l443_443473

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end remainder_3005_98_l443_443473


namespace min_cost_to_win_l443_443903

theorem min_cost_to_win (n : ℕ) : 
  (∀ m : ℕ, m = 0 →
  (∀ cents : ℕ, 
  (n = 5 * m ∨ n = m + 1) ∧ n > 2008 ∧ n % 100 = 42 → 
  cents = 35)) :=
sorry

end min_cost_to_win_l443_443903


namespace problem_1_l443_443175

theorem problem_1 (a : ℝ) (h : a^(1/2) + a^(-1/2) = 3) :
  (a^(3/2) + a^(-3/2) + 2) / (a^2 + a^(-2) + 3) = 2 / 5 :=
sorry

end problem_1_l443_443175


namespace exists_600_digit_number_l443_443463

-- Define the concept of digit addition (complexification)
def complexify (n : ℕ) : ℕ := n + 1

-- Define the initial number of digits
def initial_digits : ℕ := 500

-- The number of times complexification is performed
def times_complexified : ℕ := 100

-- The target number of digits after complexifications
def target_digits : ℕ := initial_digits + times_complexified

theorem exists_600_digit_number :
  ∃ n, n = target_digits := 
begin
  use target_digits,
  refl,
end

end exists_600_digit_number_l443_443463


namespace books_arrangement_l443_443067

theorem books_arrangement :
  let books := [A, B, C, D, E, F] in
  let ends := [A, B] in
  let adjacent := [C, D] in
  ∃ (arrangements : Finset (List books)),
    (∀ arrangement ∈ arrangements, arrangement.head = A ∨ arrangement.head = B) ∧
    (∀ arrangement ∈ arrangements, arrangement.last = A ∨ arrangement.last = B) ∧
    (∀ arrangement ∈ arrangements, adjacent ∈ arrangement.adjacent_subs) ∧
    arrangements.card = 24 :=
by sorry

end books_arrangement_l443_443067


namespace probability_angie_carlos_opposite_bridget_not_adjacent_l443_443162

noncomputable def count_permutations : ℕ → ℕ :=
λ n, if n = 0 then 1 else n * count_permutations (n - 1)

theorem probability_angie_carlos_opposite_bridget_not_adjacent :
  ∃ n : ℚ, n = (1 : ℚ) / 6 :=
begin
  -- Definitions
  let positions := 5,
  let total_permutations := count_permutations (positions - 1),
  let favorable_permutations :=
    (count_permutations 2) * 2,

  -- Calculate probability as a fraction
  let probability := (favorable_permutations : ℚ) / (total_permutations : ℚ),
  
  -- Prove the probability is 1/6
  use probability,
  norm_num [count_permutations],
  norm_cast,
  simp only [total_permutations, favorable_permutations],
  norm_num,
  exact rfl,
end

end probability_angie_carlos_opposite_bridget_not_adjacent_l443_443162


namespace find_a_l443_443212

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end find_a_l443_443212


namespace no_solution_system_l443_443931

noncomputable def system_inconsistent : Prop :=
  ∀ x y : ℝ, ¬ (3 * x - 4 * y = 8 ∧ 6 * x - 8 * y = 12)

theorem no_solution_system : system_inconsistent :=
by
  sorry

end no_solution_system_l443_443931


namespace max_expression_value_l443_443028

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443028


namespace arrangement_of_people_l443_443624

variable (A B C D E : Type)

/-- Given five distinct people (A, B, C, D, E) standing in a row,
the number of different arrangements such that A and B are not adjacent
and A and C are not adjacent is 36. -/
theorem arrangement_of_people (arrangements : Finset (List (A ⊕ B ⊕ C ⊕ D ⊕ E))) :
  (arrangements.filter (λ l, ¬ (adjacent A B l) ∧ ¬ (adjacent A C l))).card = 36 :=
sorry

-- Auxiliary definition to check adjacency in a list 
def adjacent (x y : Type) : List (Type) → Bool
| []       => false
| [a]      => false
| (a::b::t) => if (a = x ∧ b = y) ∨ (a = y ∧ b = x) then true else adjacent x y (b::t)


end arrangement_of_people_l443_443624


namespace fraction_problem_l443_443943

theorem fraction_problem :
  (3 / 7 + 5 / 8) / (5 / 12 + 2 / 9) = 531 / 322 :=
by sorry

end fraction_problem_l443_443943


namespace cube_surface_area_l443_443062

-- Define the volume condition
def volume (s : ℕ) : ℕ := s^3

-- Define the surface area function
def surface_area (s : ℕ) : ℕ := 6 * s^2

-- State the theorem to be proven
theorem cube_surface_area (s : ℕ) (h : volume s = 729) : surface_area s = 486 :=
by
  sorry

end cube_surface_area_l443_443062


namespace min_sum_a_b_l443_443237

theorem min_sum_a_b (a b : ℝ) (h_cond: 1/a + 4/b = 1) (a_pos : 0 < a) (b_pos : 0 < b) : 
  a + b ≥ 9 :=
sorry

end min_sum_a_b_l443_443237


namespace find_circle_equation_l443_443217

theorem find_circle_equation (M : ℝ × ℝ) (C : ℝ × ℝ)
    (center_C : C = (-1, -2)) (passes_through_M : 
        dist M C = sqrt (1 + 2)^2 + 1^2 = sqrt 13 ) :
  ∃ (r : ℝ), (x + 1)^2 + (y + 2)^2 = r :=
  sorry

end find_circle_equation_l443_443217


namespace area_of_isosceles_right_triangle_l443_443459

theorem area_of_isosceles_right_triangle 
  (XYZ : Type) 
  (XY YZ XZ : ℝ)
  (isosceles_right_triangle : XY^2 + YZ^2 = XZ^2 ∧ XY = YZ ∨ XY = XZ ∨ YZ = XZ)
  (xy_longer_yz : XY > YZ)
  (xy_length : XY = 12.000000000000002) : 
  √(YZ * YZ) = √(YZ) ∧ XY = YZ implies 
  let a := 12.000000000000002 / √2 in 
  let area := 0.5 * a * a in 
  area = 36.000000000000015 := 
sorry

end area_of_isosceles_right_triangle_l443_443459


namespace distance_center_point_is_10_l443_443468

-- Definition of the circle equation and the given point
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 6 * x - 8 * y + 24
def point := (-3, 4 : ℝ × ℝ)

-- Definition of the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Definition of the circle's center derived from the given equation
def circle_center : ℝ × ℝ := (3, -4)

-- Statement to prove that the distance between the circle's center and the given point is 10
theorem distance_center_point_is_10 : distance circle_center point = 10 :=
sorry

end distance_center_point_is_10_l443_443468


namespace max_value_of_expression_l443_443094

theorem max_value_of_expression : ∃ t ∈ ℝ, (∀ t' ∈ ℝ, 
  (2^t' - 3 * t') * t' / 4^t' ≤ (2^t - 3 * t) * t / 4^t' ) ∧ 
  ( (2^t - 3 * t) * t / 4^t = 1/12) :=
sorry

end max_value_of_expression_l443_443094


namespace product_of_repeating_decimal_l443_443572

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l443_443572


namespace square_rectangle_area_ratio_l443_443538

theorem square_rectangle_area_ratio (l1 l2 : ℕ) (h1 : l1 = 32) (h2 : l2 = 64) (p : ℕ) (s : ℕ) 
  (h3 : p = 256) (h4 : s = p / 4)  :
  (s * s) / (l1 * l2) = 2 := 
by
  sorry

end square_rectangle_area_ratio_l443_443538


namespace max_value_of_fraction_l443_443041

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443041


namespace butternut_wood_figurines_l443_443544

theorem butternut_wood_figurines (B : ℕ) (basswood_blocks : ℕ) (aspen_blocks : ℕ) (butternut_blocks : ℕ) 
  (basswood_figurines_per_block : ℕ) (aspen_figurines_per_block : ℕ) (total_figurines : ℕ) 
  (h_basswood_blocks : basswood_blocks = 15)
  (h_aspen_blocks : aspen_blocks = 20)
  (h_butternut_blocks : butternut_blocks = 20)
  (h_basswood_figurines_per_block : basswood_figurines_per_block = 3)
  (h_aspen_figurines_per_block : aspen_figurines_per_block = 2 * basswood_figurines_per_block)
  (h_total_figurines : total_figurines = 245) :
  B = 4 :=
by
  -- Definitions based on the given conditions
  let basswood_figurines := basswood_blocks * basswood_figurines_per_block
  let aspen_figurines := aspen_blocks * aspen_figurines_per_block
  let figurines_from_butternut := total_figurines - basswood_figurines - aspen_figurines
  -- Calculate the number of figurines per block of butternut wood
  let butternut_figurines_per_block := figurines_from_butternut / butternut_blocks
  -- The objective is to prove that the number of figurines per block of butternut wood is 4
  exact sorry

end butternut_wood_figurines_l443_443544


namespace max_expression_value_l443_443007

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443007


namespace max_f_find_m_find_a_l443_443663

noncomputable def f (x : ℝ) : ℝ := x / Real.exp(x)

theorem max_f : ∃ x : ℝ, (0 < x) ∧ ∀ y : ℝ, y > 0 → f(y) ≤ f(x) ∧ x = 1 :=
sorry

theorem find_m (m : ℝ) : (∃ x : ℝ, x > 0 ∧ f(x) - m = 0) ∧ (∃ x' : ℝ, x' > 0 ∧ x' ≠ x ∧ f(x') - m = 0) ↔ (0 < m ∧ m < 1 / Real.exp(1)) :=
sorry

theorem find_a (a : ℝ) : (∃! x : ℤ, f(x) > 0 ∧ f^2(x.to_real) - a * f(x.to_real) > 0) ↔ 2 / (Real.exp 2) ≤ a ∧ a < 1 / Real.exp(1) :=
sorry

end max_f_find_m_find_a_l443_443663


namespace differences_in_set_l443_443294

theorem differences_in_set : 
  let s := {1, 3, 5, 7, 9, 11}
  in (#{d | ∃ x y, x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ d = x - y ∧ d > 0}.card) = 5 := 
by
  sorry

end differences_in_set_l443_443294


namespace sin_cos_identity_l443_443648
noncomputable theory

variables {α : ℝ} (tanα : ℝ) (sinα cosα : ℝ)

-- Condition
def condition := tanα = 1 / 2

-- Theorem statement
theorem sin_cos_identity (h : condition) : sinα * cosα = 2 / 5 :=
sorry

end sin_cos_identity_l443_443648


namespace total_books_l443_443376

def number_of_zoology_books : ℕ := 16
def number_of_botany_books : ℕ := 4 * number_of_zoology_books

theorem total_books : number_of_zoology_books + number_of_botany_books = 80 := by
  sorry

end total_books_l443_443376


namespace work_completion_days_l443_443137

variables (M D X : ℕ) (W : ℝ)

-- Original conditions
def original_men : ℕ := 15
def planned_days : ℕ := 40
def men_absent : ℕ := 5

-- Theorem to prove
theorem work_completion_days :
  M = original_men →
  D = planned_days →
  W > 0 →
  (M - men_absent) * X * W = M * D * W →
  X = 60 :=
by
  intros hM hD hW h_work
  sorry

end work_completion_days_l443_443137


namespace find_x_squared_l443_443453

theorem find_x_squared :
  ∃ x : ℝ, 0 < x ∧ (cos (arctan (2 * x)) = x / 2) →
  x^2 = (Real.sqrt 17 - 1) / 4 :=
by
  sorry

end find_x_squared_l443_443453


namespace find_integer_l443_443219

theorem find_integer (n : ℤ) (h₀ : 4 ≤ n) (h₁ : n ≤ 10) (h₂ : n ≡ 11783 [MOD 7]) : n = 5 := 
sorry

end find_integer_l443_443219


namespace find_positive_Y_for_nine_triangle_l443_443590

def triangle_relation (X Y : ℝ) : ℝ := X^2 + 3 * Y^2

theorem find_positive_Y_for_nine_triangle (Y : ℝ) : (9^2 + 3 * Y^2 = 360) → Y = Real.sqrt 93 := 
by
  sorry

end find_positive_Y_for_nine_triangle_l443_443590


namespace count_false_propositions_l443_443827

/-- 
Definition of the propositions:
1. Some real numbers are non-repeating decimals.
2. Some triangles are not isosceles triangles.
3. Some rhombi are squares.
-/
def proposition_1 : Prop := ∃ x : ℝ, irrational x
def proposition_2 : Prop := ∃ (a b c : ℝ), triangle a b c ∧ ¬ isosceles a b c
def proposition_3 : Prop := ∃ (r : Type) [∀ x y : r, decidable (x = y)], ∀ x y : r, (is_rhombus x y r) → (is_square x y r)

/-- 
The theorem states that there are 0 false propositions 
among the given three propositions.
-/
theorem count_false_propositions : 
  (¬proposition_1 = false) ∧ (¬proposition_2 = false) ∧ (¬proposition_3 = false) → 
  num_false_propositions(proposition_1, proposition_2, proposition_3) = 0 :=
by
  sorry

end count_false_propositions_l443_443827


namespace andrey_stamps_l443_443387

theorem andrey_stamps :
  ∃ (x : ℕ), 
    x % 3 = 1 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 5 ∧ 
    150 < x ∧ 
    x ≤ 300 ∧ 
    x = 208 :=
begin
  sorry
end

end andrey_stamps_l443_443387


namespace answer_to_rarely_infrequently_word_l443_443900

-- Declare variables and definitions based on given conditions
-- In this context, we'll introduce a basic definition for the word "seldom".

noncomputable def is_word_meaning_rarely (w : String) : Prop :=
  w = "seldom"

-- Now state the problem in the form of a Lean theorem
theorem answer_to_rarely_infrequently_word : ∃ w, is_word_meaning_rarely w :=
by
  use "seldom"
  unfold is_word_meaning_rarely
  rfl

end answer_to_rarely_infrequently_word_l443_443900


namespace plane_through_line_and_point_l443_443233

-- Definitions from the conditions
def line (x y z : ℝ) : Prop :=
  (x - 1) / 2 = (y - 3) / 4 ∧ (x - 1) / 2 = z / (-1)

def pointP1 : ℝ × ℝ × ℝ := (1, 5, 2)

-- Correct answer
def plane_eqn (x y z : ℝ) : Prop :=
  5 * x - 2 * y + 2 * z + 1 = 0

-- The theorem to prove
theorem plane_through_line_and_point (x y z : ℝ) :
  line x y z → plane_eqn x y z := by
  sorry

end plane_through_line_and_point_l443_443233


namespace distinct_terms_in_expansion_l443_443686

theorem distinct_terms_in_expansion:
  (∀ (x y z u v w: ℝ), (x + y + z) * (u + v + w + x + y) = 0 → false) →
  3 * 5 = 15 := by sorry

end distinct_terms_in_expansion_l443_443686


namespace smallest_number_of_students_l443_443526

theorem smallest_number_of_students 
    (ratio_9th_10th : Nat := 3 / 2)
    (ratio_9th_11th : Nat := 5 / 4)
    (ratio_9th_12th : Nat := 7 / 6) :
  ∃ N9 N10 N11 N12 : Nat, 
  N9 / N10 = 3 / 2 ∧ N9 / N11 = 5 / 4 ∧ N9 / N12 = 7 / 6 ∧ N9 + N10 + N11 + N12 = 349 :=
by {
  sorry
}

#print axioms smallest_number_of_students

end smallest_number_of_students_l443_443526


namespace min_value_S_l443_443061

noncomputable def min_sum_seq_2003 : ℕ :=
  let a : ℕ → ℕ := sorry, --sequence definition placeholder
  S := (finset.range 2003).sum a
  in S

theorem min_value_S : min_sum_seq_2003 ≥ 4003 :=
begin
  -- sequence and constraints definition
  have h1 : ∀ i, 1 ≤ a i ∧ a i < 100,
  { intro i, sorry },
  
  have h2 : ∀ i j, i < j → i + 1 = j → (a i + a j ≠ 100),
  { intros i j hij hij1, sorry },

  -- proving the lower bound
  sorry
end

end min_value_S_l443_443061


namespace find_t_l443_443672

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l443_443672


namespace solve_for_z_l443_443399

theorem solve_for_z : ∀ z : ℂ, (5 - 3 * complex.I * z = 2 + 5 * complex.I * z) ↔ (z = -3 * complex.I / 8) :=
by
  intros z
  split
  {
    -- Proof of the forward direction (if 5 - 3iz = 2 + 5iz, then z = -3i/8)
    assume h : 5 - 3 * complex.I * z = 2 + 5 * complex.I * z
    sorry
  }
  {
    -- Proof of the backward direction (if z = -3i/8, then 5 - 3iz = 2 + 5iz)
    assume h : z = -3 * complex.I / 8
    sorry
  }

end solve_for_z_l443_443399


namespace equal_angles_point_p_l443_443181

noncomputable def ellipse := 
  { (x : ℝ) (y : ℝ) | (x^2 / 8 + y^2 / 4 = 1) }

def focus : ℝ × ℝ := (2, 0)

theorem equal_angles_point_p :
  ∃ (p : ℝ), p > 0 ∧
  (∀ A B : ℝ × ℝ, (A ∈ ellipse) → (B ∈ ellipse) → (A ≠ B) → A.1 * B.2 - A.2 * B.1 = 0 → 
    let P := (p, 0)
    in ∠A P focus = ∠B P focus) ↔ p = 2 := 
by
  sorry

end equal_angles_point_p_l443_443181


namespace intersection_point_l443_443323

noncomputable def curve1 (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ Real.pi / 2) : ℝ × ℝ := 
  (Real.sqrt 5 * Real.cos θ, Real.sqrt 5 * Real.sin θ)

noncomputable def curve2 (t : ℝ) : ℝ × ℝ := 
  (1 - (Real.sqrt 2) / 2 * t, - (Real.sqrt 2) / 2 * t)

theorem intersection_point :
  ∃ θ t, 0 ≤ θ ∧ θ ≤ Real.pi / 2 ∧ curve1 θ ⟨0, (Real.pi / 2).le_refl⟩ = curve2 t ∧ curve1 θ ⟨0, (Real.pi / 2).le_refl⟩ = (2, 1) :=
begin
  sorry
end

end intersection_point_l443_443323


namespace average_mpg_correct_l443_443169

noncomputable def average_mpg (initial_miles final_miles : ℕ) (refill1 refill2 refill3 : ℕ) : ℚ :=
  let distance := final_miles - initial_miles
  let total_gallons := refill1 + refill2 + refill3
  distance / total_gallons

theorem average_mpg_correct :
  average_mpg 32000 33100 15 10 22 = 23.4 :=
by
  sorry

end average_mpg_correct_l443_443169


namespace range_of_k_l443_443232

-- Define the sequence a_n and its excellent value H_n
def excellent_value (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (a 1 + (∑ i in finset.range (n - 1), 2 ^ i * a (i + 2))) / n

-- Define the known condition on the excellent value
axiom excellent_value_known (a : ℕ → ℝ) (n : ℕ) : excellent_value a n = 2 ^ (n + 1)

-- Define the sum of the first n terms S_n of the sequence a_n - kn
def S_n (a : ℕ → ℝ) (k : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, (a (i + 1) - k * (i + 1))

-- Define the condition that the sum of the first n terms is bounded by S_6
axiom sum_bounded (a : ℕ → ℝ) (k : ℝ) : ∀ n : ℕ, 0 < n → S_n a k n ≤ S_n a k 6

-- Prove the range for k
theorem range_of_k (a : ℕ → ℝ) (k : ℝ) :
  (∃ k : ℝ, (16 / 7) ≤ k ∧ k ≤ (7 / 3)) :=
sorry

end range_of_k_l443_443232


namespace max_sum_of_products_l443_443838

theorem max_sum_of_products : 
  ∃ (a : Fin 2009 → ℤ), 
    (∀ i, a i = 1 ∨ a i = -1) ∧ 
    ¬ (∀ i, a i = 1) ∧ 
    ¬ (∀ i, a i = -1) ∧ 
    (∑ i : Fin 2009, 
      ∏ j in Finset.range 10, a ((i + j) % 2009)) = 2005 :=
sorry

end max_sum_of_products_l443_443838


namespace find_number_l443_443983

theorem find_number (n : ℕ) (h1 : 9 < real.cbrt n) (h2 : real.cbrt n < 9.1) (h3 : n % 24 = 0) : n = 744 :=
sorry

end find_number_l443_443983


namespace minimum_red_edges_l443_443605

theorem minimum_red_edges (coloring : Fin 12 → Bool) (h : ∀ (f : Fin 6), ∃ (e : Fin 4), coloring (face_edge f e) = false) : 
  ∃ n, n = 6 ∧ (∀ red_edges_count < n, ¬ satisfies_condition coloring red_edges_count) :=
  sorry

end minimum_red_edges_l443_443605


namespace max_value_of_fraction_l443_443017

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443017


namespace roots_of_equation_l443_443437

theorem roots_of_equation :
  ∀ x : ℝ, x * (x + 2) = -x - 2 ↔ x = -2 ∨ x = -1 :=
by
  intro x
  split
  intro h
  sorry -- Proof steps are not required, so we insert sorry here.
  intro hx
  cases hx with h1 h2
  { rw h1
    sorry -- Proof steps are not required, so we insert sorry here. }
  { rw h2
    sorry -- Proof steps are not required, so we insert sorry here. }

end roots_of_equation_l443_443437


namespace K4_planar_K33_non_planar_K5_non_planar_l443_443396

-- Definitions of the graphs
def K4 : SimpleGraph (Fin 4) := ⊤
def K33 : SimpleGraph (Sum (Fin 3) (Fin 3)) :=
  {Adj := λ x y, if x <.1 y = .2 then true else false , simp [SimpleGraph.completeBipartite (Fin 3) (Fin 3)]}
def K5 : SimpleGraph (Fin 5) := ⊤

-- Planarity conditions
def is_planar (G : SimpleGraph α) : Prop := sorry  -- Assume the definition of planarity

-- The theorem statements
theorem K4_planar : is_planar K4 := sorry

theorem K33_non_planar : ¬ is_planar K33 := sorry

theorem K5_non_planar : ¬ is_planar K5 := sorry

end K4_planar_K33_non_planar_K5_non_planar_l443_443396


namespace find_derivative_l443_443861

noncomputable def f (a b x : ℝ) : ℝ := a * real.log x + (b / x)

theorem find_derivative (a b : ℝ) (h₁ : f a b 1 = -2) (h₂ : differentiable ℝ (f a b)) :
  b = -2 ∧ a = -2 → deriv (f a b) 2 = -1 / 2 :=
by
  intros h
  cases h with hb ha
  rw [hb, ha]
  sorry

end find_derivative_l443_443861


namespace cos_squared_value_l443_443647

theorem cos_squared_value (x : ℝ) (h : Real.sin (x + π / 6) = 1 / 4) : 
  Real.cos (π / 3 - x) ^ 2 = 1 / 16 := 
sorry

end cos_squared_value_l443_443647


namespace distinct_diff_count_l443_443289

theorem distinct_diff_count :
  (∃ S : set ℕ, S = {1, 3, 5, 7, 9, 11} ∧ 
    (∑ x in S, ∑ y in S, if x > y then 1 else 0) = 10) :=
begin
  let S := {1, 3, 5, 7, 9, 11},
  use S,
  split,
  {
    refl,
  },
  {
    sorry
  }
end

end distinct_diff_count_l443_443289


namespace sin_alpha_value_l443_443644

theorem sin_alpha_value (α : ℝ) (h1 : cos (π - α) = 4 / 5) (h2 : π / 2 < α ∧ α < π) : sin α = 3 / 5 :=
sorry

end sin_alpha_value_l443_443644


namespace star_value_l443_443629

def star (a b : ℝ) : ℝ := a^3 + 3 * a^2 * b + 3 * a * b^2 + b^3

theorem star_value : star 3 2 = 125 :=
by
  sorry

end star_value_l443_443629


namespace commute_time_l443_443899

theorem commute_time (d s1 s2 : ℝ) (h1 : s1 = 45) (h2 : s2 = 30) (h3 : d = 18) : (d / s1 + d / s2 = 1) :=
by
  -- Definitions and assumptions
  rw [h1, h2, h3]
  -- Total time calculation
  exact sorry

end commute_time_l443_443899


namespace arielle_age_l443_443834

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end arielle_age_l443_443834


namespace unique_increasing_sequence_l443_443074

theorem unique_increasing_sequence (a : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n+1))
  (h_sum : 2 ^ 305 + 1 = (Finset.range 319).sum (λ i, 2 ^ (a i))) :
  ∃ k, k = 319 :=
by 
  use 319
  sorry

end unique_increasing_sequence_l443_443074


namespace number_of_circumcenter_quadrilaterals_l443_443865

-- Definitions for each type of quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

def is_square (q : Quadrilateral) : Prop := sorry
def is_rectangle (q : Quadrilateral) : Prop := sorry
def is_rhombus (q : Quadrilateral) : Prop := sorry
def is_kite (q : Quadrilateral) : Prop := sorry
def is_trapezoid (q : Quadrilateral) : Prop := sorry
def has_circumcenter (q : Quadrilateral) : Prop := sorry

-- List of quadrilaterals
def square : Quadrilateral := sorry
def rectangle : Quadrilateral := sorry
def rhombus : Quadrilateral := sorry
def kite : Quadrilateral := sorry
def trapezoid : Quadrilateral := sorry

-- Proof that the number of quadrilaterals with a point equidistant from all vertices is 2
theorem number_of_circumcenter_quadrilaterals :
  (has_circumcenter square) ∧
  (has_circumcenter rectangle) ∧
  ¬ (has_circumcenter rhombus) ∧
  ¬ (has_circumcenter kite) ∧
  ¬ (has_circumcenter trapezoid) →
  2 = 2 :=
by
  sorry

end number_of_circumcenter_quadrilaterals_l443_443865


namespace octahedron_coloring_l443_443382

theorem octahedron_coloring : 
  ∃ (n : ℕ), n = 6 ∧
  ∀ (F : Fin 8 → Fin 4), 
    (∀ (i j : Fin 8), i ≠ j → F i ≠ F j) ∧
    (∃ (pairs : Fin 8 → (Fin 4 × Fin 4)), 
      (∀ (i : Fin 8), ∃ j : Fin 4, pairs i = (j, j)) ∧ 
      (∀ j, ∃ (i : Fin 8), F i = j)) :=
by
  sorry

end octahedron_coloring_l443_443382


namespace intersection_points_count_l443_443598

theorem intersection_points_count:
  let line1 := { p : ℝ × ℝ | ∃ x y : ℝ, 4 * y - 3 * x = 2 ∧ (p.1 = x ∧ p.2 = y) }
  let line2 := { p : ℝ × ℝ | ∃ x y : ℝ, x + 3 * y = 3 ∧ (p.1 = x ∧ p.2 = y) }
  let line3 := { p : ℝ × ℝ | ∃ x y : ℝ, 6 * x - 8 * y = 6 ∧ (p.1 = x ∧ p.2 = y) }
  ∃! p1 p2 : ℝ × ℝ, p1 ∈ line1 ∧ p1 ∈ line2 ∧ p2 ∈ line2 ∧ p2 ∈ line3 :=
by
  sorry

end intersection_points_count_l443_443598


namespace sum_of_ages_five_years_ago_l443_443939

theorem sum_of_ages_five_years_ago :
  ∀ (Djibo_age today sister_age today : ℕ), 
    Djibo_age = 17 → sister_age = 28 →
    (Djibo_age - 5) + (sister_age - 5) = 35 :=
by
  intro Djibo_age sister_age
  intros h1 h2
  rw [h1, h2]
  sorry

end sum_of_ages_five_years_ago_l443_443939


namespace area_of_triangle_PQR_l443_443091

def Point := (ℝ × ℝ)
def area_of_triangle (P Q R : Point) : ℝ :=
  0.5 * abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2))

def P : Point := (1, 1)
def Q : Point := (4, 5)
def R : Point := (7, 2)

theorem area_of_triangle_PQR :
  area_of_triangle P Q R = 10.5 := by
  sorry

end area_of_triangle_PQR_l443_443091


namespace sum_of_circumradius_l443_443908

-- Define the geometrical setup
variables (AB CD h : ℝ)
variables (E : Type) (is_isosceles_trapezoid : Prop)

-- Define the radii of the circumcircles of triangles ABE and CDE
variables (R M : ℝ)

-- Assume the given conditions
def given_conditions : Prop :=
  AB = 13 ∧ CD = 17 ∧ h = 3 ∧ is_isosceles_trapezoid

-- The problem statement
theorem sum_of_circumradius
  (h_condition : given_conditions)
  (h_isosceles : is_isosceles_trapezoid) :
  R + M = 39 := by
  sorry

end sum_of_circumradius_l443_443908


namespace find_integer_divisible_by_24_with_cube_root_in_range_l443_443965

theorem find_integer_divisible_by_24_with_cube_root_in_range :
  ∃ (n : ℕ), (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ (24 ∣ n) ∧ n = 744 := by
    sorry

end find_integer_divisible_by_24_with_cube_root_in_range_l443_443965


namespace solve_exp_eqn_l443_443690

theorem solve_exp_eqn (x : ℝ) : (2^x + 3^x + 6^x = 7^x) ↔ (x = 2) := 
by
  sorry

end solve_exp_eqn_l443_443690


namespace perfect_square_pairs_l443_443986

theorem perfect_square_pairs (x y : ℕ) (a b : ℤ) :
  (x^2 + 8 * ↑y = a^2 ∧ y^2 - 8 * ↑x = b^2) →
  (∃ n : ℕ, x = n ∧ y = n + 2) ∨ (x = 7 ∧ y = 15) ∨ (x = 33 ∧ y = 17) ∨ (x = 45 ∧ y = 23) :=
by
  sorry

end perfect_square_pairs_l443_443986


namespace sequences_properties_l443_443670

-- Definitions for properties P and P'
def is_property_P (seq : List ℕ) : Prop := sorry
def is_property_P' (seq : List ℕ) : Prop := sorry

-- Define sequences
def sequence1 := [1, 2, 3, 1]
def sequence2 := [1, 234, 5]  -- Extend as needed

-- Conditions
def bn_is_permutation_of_an (a b : List ℕ) : Prop := sorry -- Placeholder for permutation check

-- Main Statement 
theorem sequences_properties :
  is_property_P sequence1 ∧
  is_property_P' sequence2 := 
by
  sorry

-- Additional theorem to check permutation if needed
-- theorem permutation_check :
--  bn_is_permutation_of_an sequence1 sequence2 :=
-- by
--  sorry

end sequences_properties_l443_443670


namespace product_of_repeating_decimal_l443_443579

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l443_443579


namespace pentagon_area_l443_443912

noncomputable def square_area (side_length : ℤ) : ℤ :=
  side_length * side_length

theorem pentagon_area (CF : ℤ) (a b : ℤ) (CE : ℤ) (ED : ℤ) (EF : ℤ) :
  (CF = 5) →
  (a = CE + ED) →
  (b = EF) →
  (CE < ED) →
  CF * CF = CE * CE + EF * EF →
  square_area a + square_area b - (CE * EF / 2) = 71 :=
by
  intros hCF ha hb hCE_lt_ED hPythagorean
  sorry

end pentagon_area_l443_443912


namespace mass_percentage_Ca_in_CaOH2_l443_443221

-- Conditions
def molar_mass_Ca : ℝ := 40.08 -- g/mol
def molar_mass_O : ℝ := 16.00 -- g/mol
def molar_mass_H : ℝ := 1.01 -- g/mol

-- Given the chemical formula of Calcium hydroxide: Ca(OH)₂
def molar_mass_OH : ℝ := molar_mass_O + molar_mass_H
def molar_mass_CaOH2 : ℝ := molar_mass_Ca + 2 * molar_mass_OH

-- Calculate the mass percentage of Ca in Ca(OH)2
def mass_percentage_Ca : ℝ := (molar_mass_Ca / molar_mass_CaOH2) * 100

-- Theorem stating mass percentage of Ca in Ca(OH)2 is approximately 54.09%
theorem mass_percentage_Ca_in_CaOH2 : abs (mass_percentage_Ca - 54.09) < 1e-2 :=
by
  sorry

end mass_percentage_Ca_in_CaOH2_l443_443221


namespace xiaoming_mirrored_time_l443_443552

-- Define the condition: actual time is 7:10 AM.
def actual_time : (ℕ × ℕ) := (7, 10)

-- Define a function to compute the mirrored time given an actual time.
def mirror_time (h m : ℕ) : (ℕ × ℕ) :=
  let mirrored_minute := if m = 0 then 0 else 60 - m
  let mirrored_hour := if m = 0 then if h = 12 then 12 else (12 - h) % 12
                        else if h = 12 then 11 else (11 - h) % 12
  (mirrored_hour, mirrored_minute)

-- Our goal is to verify that the mirrored time of 7:10 is 4:50.
theorem xiaoming_mirrored_time : mirror_time 7 10 = (4, 50) :=
by
  -- Proof will verify that mirror_time (7, 10) evaluates to (4, 50).
  sorry

end xiaoming_mirrored_time_l443_443552


namespace total_pages_read_proof_l443_443394

noncomputable def reading_rate (pages_per_60_minutes : ℝ) : ℝ :=
  pages_per_60_minutes / 60

def reading_time : ℝ := 480 -- in minutes

def Rene_pages : ℝ := (reading_rate 30) * reading_time
def Lulu_pages : ℝ := (reading_rate 27) * reading_time
def Cherry_pages : ℝ := (reading_rate 25) * reading_time
def Max_pages : ℝ := (reading_rate 34) * reading_time
def Rosa_pages : ℝ := (reading_rate 29) * reading_time

def total_pages_read := Rene_pages + Lulu_pages + Cherry_pages + Max_pages + Rosa_pages

theorem total_pages_read_proof : total_pages_read = 1160 := by
  sorry

end total_pages_read_proof_l443_443394


namespace problem1_problem2_problem3_problem4_l443_443920

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end problem1_problem2_problem3_problem4_l443_443920


namespace deductive_reasoning_example_l443_443906

theorem deductive_reasoning_example :
  (∀ x ∈ {gold, silver, copper, iron}, conducts_electricity x) →
  (sequence_gen_formula ∈ {a_n = 1 / (n * (n + 1)) | n ∈ ℕ}) →
  (∀ r : ℝ, area_of_circle r = π * r * r) →
  (equation_of_circle ∈ {(x - a)^2 + (y - b)^2 = r^2 | x y a b r : ℝ}) →
  is_deductive_reasoning 
    (from_major_minor_premises_to_conclusion
       (area_of_circle 1 = π)) := 
  by
    sorry

end deductive_reasoning_example_l443_443906


namespace cos_double_angle_l443_443235

theorem cos_double_angle (x : ℝ) (h : sin (π / 4 + x / 2) = 3 / 5) : cos (2 * x) = -7 / 25 := 
sorry

end cos_double_angle_l443_443235


namespace travis_flight_cost_l443_443457

theorem travis_flight_cost 
  (cost_leg1 : ℕ := 1500) 
  (cost_leg2 : ℕ := 1000) 
  (discount_leg1 : ℕ := 25) 
  (discount_leg2 : ℕ := 35) : 
  cost_leg1 - (discount_leg1 * cost_leg1 / 100) + cost_leg2 - (discount_leg2 * cost_leg2 / 100) = 1775 :=
by
  sorry

end travis_flight_cost_l443_443457


namespace num_j_integers_l443_443352

def is_divisor (a b : ℕ) : Prop := b % a = 0

def sum_divisors (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ x => is_divisor x n).sum

def perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, 2 ≤ m → m * m ≤ n → ¬is_divisor m n

noncomputable def count_primes_below (n : ℕ) : ℕ :=
  (Finset.range n).filter is_prime |> Finset.card

theorem num_j_integers :
  let g := sum_divisors 
  (Finset.range 1001).filter (λ j =>
    g j = 1 + Nat.sqrt j + j).card = count_primes_below 32 := by
  sorry

end num_j_integers_l443_443352


namespace find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443973

theorem find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1 :
    ∃ n : ℕ, n > 0 ∧ (n % 24 = 0) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ n = 744 :=
by
  sorry

end find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443973


namespace max_expression_value_l443_443034

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443034


namespace root_in_interval_l443_443643

theorem root_in_interval (a b c : ℝ) (h_a : a ≠ 0)
    (h_table : ∀ x y, (x = 1.2 ∧ y = -1.16) ∨ (x = 1.3 ∧ y = -0.71) ∨ (x = 1.4 ∧ y = -0.24) ∨ (x = 1.5 ∧ y = 0.25) ∨ (x = 1.6 ∧ y = 0.76) → y = a * x^2 + b * x + c ) :
  ∃ x₁, 1.4 < x₁ ∧ x₁ < 1.5 ∧ a * x₁^2 + b * x₁ + c = 0 :=
by sorry

end root_in_interval_l443_443643


namespace triangle_area_l443_443305

/-!
  In a triangle ABC, the sides opposite to angles A, B, and C are a, b, and c respectively.
  Given the following conditions:
  1. \( b^2 + c^2 = a^2 + bc \)
  2. \( \overrightarrow{AC} \cdot \overrightarrow{AB} = 4 \)
  Prove that the area of triangle ABC is \( 2 \sqrt{3} \).
-/

theorem triangle_area (a b c : ℝ) (A B C : ℝ) 
  (h1 : b^2 + c^2 = a^2 + b * c) 
  (h2 : b * c * Real.cos (A) = 4) 
  (angle_A : A = Real.acos (1 / 2)) :
  (1 / 2) * b * c * Real.sin (A) = 2 * Real.sqrt 3 :=
by  
  sorry

end triangle_area_l443_443305


namespace problem_statement_l443_443728

-- Given conditions
def parametric_curve (α : ℝ) : ℝ × ℝ :=
  (sqrt 5 * cos α, sin α)

def polar_line (ρ θ : ℝ) : Prop :=
  ρ * cos (θ + π / 4) = sqrt 2

def point_P := (0, -2 : ℝ)

-- Statement of the problem
theorem problem_statement :
  (∀ x y α, (x, y) = parametric_curve α → (x^2 / 5) + y^2 = 1) ∧
  (∀ ρ θ, polar_line ρ θ → ∀ x y, (x, y) = (ρ * cos θ, ρ * sin θ) → y = x - 2) ∧
  (|PA| + |PB| = 10 * sqrt 2 / 3) :=
by
  sorry

end problem_statement_l443_443728


namespace jerry_time_proof_l443_443456

noncomputable def tom_walk_speed (step_length_tom : ℕ) (pace_tom : ℕ) : ℕ := 
  step_length_tom * pace_tom

noncomputable def tom_distance_to_office (walk_speed_tom : ℕ) (time_tom : ℕ) : ℕ :=
  walk_speed_tom * time_tom

noncomputable def jerry_walk_speed (step_length_jerry : ℕ) (pace_jerry : ℕ) : ℕ :=
  step_length_jerry * pace_jerry

noncomputable def jerry_time_to_office (distance_to_office : ℕ) (walk_speed_jerry : ℕ) : ℚ :=
  distance_to_office / walk_speed_jerry

theorem jerry_time_proof :
  let step_length_tom := 80
  let pace_tom := 85
  let time_tom := 20
  let step_length_jerry := 70
  let pace_jerry := 110
  let office_distance := tom_distance_to_office (tom_walk_speed step_length_tom pace_tom) time_tom
  let jerry_speed := jerry_walk_speed step_length_jerry pace_jerry
  jerry_time_to_office office_distance jerry_speed = 53/3 := 
by
  sorry

end jerry_time_proof_l443_443456


namespace find_value_of_square_sums_l443_443642

variable (x y z : ℝ)

-- Define the conditions
def weighted_arithmetic_mean := (2 * x + 2 * y + 3 * z) / 8 = 9
def weighted_geometric_mean := Real.rpow (x^2 * y^2 * z^3) (1 / 7) = 6
def weighted_harmonic_mean := 7 / ((2 / x) + (2 / y) + (3 / z)) = 4

-- State the theorem to be proved
theorem find_value_of_square_sums
  (h1 : weighted_arithmetic_mean x y z)
  (h2 : weighted_geometric_mean x y z)
  (h3 : weighted_harmonic_mean x y z) :
  x^2 + y^2 + z^2 = 351 :=
by sorry

end find_value_of_square_sums_l443_443642


namespace geostationary_orbit_distance_l443_443129

noncomputable def distance_between_stations (earth_radius : ℝ) (orbit_altitude : ℝ) (num_stations : ℕ) : ℝ :=
  let θ : ℝ := 360 / num_stations
  let R : ℝ := earth_radius + orbit_altitude
  let sin_18 := (Real.sqrt 5 - 1) / 4
  2 * R * sin_18

theorem geostationary_orbit_distance :
  distance_between_stations 3960 22236 10 = -13098 + 13098 * Real.sqrt 5 :=
by
  sorry

end geostationary_orbit_distance_l443_443129


namespace green_chips_count_l443_443070

theorem green_chips_count (total_chips : ℕ) (blue_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ)
  (h1 : total_chips = 60)
  (h2 : blue_chips = total_chips / 6)
  (h3 : red_chips = 34)
  (h4 : green_chips = total_chips - blue_chips - red_chips) :
  green_chips = 16 :=
by {
  -- Define the intermediate steps
  have h_blue_calculation : blue_chips = 10,
  { rw h1, exact Nat.div_eq_of_eq_mul_right (Nat.succ_pos 5) rfl },

  -- Assume that 34 red chips are given
  have h_red_count : red_chips = 34 := h3,

  -- Define the total number of chips
  have h_total_calculation : green_chips = 60 - 10 - 34
    by { rw [h1, h_blue_calculation, h_red_count] },

  -- Conclusion
  exact by { rw h_total_calculation, norm_num }
}

end green_chips_count_l443_443070


namespace seating_arrangements_count_l443_443314

theorem seating_arrangements_count : 
  let total_arrangements := Nat.factorial 10,
      super_person_arrangements := Nat.factorial 7,
      internal_arrangements := Nat.factorial 4,
      invalid_arrangements := super_person_arrangements * internal_arrangements
  in total_arrangements - invalid_arrangements = 3507840 := 
by
  sorry

end seating_arrangements_count_l443_443314


namespace quadratic_coefficients_l443_443047

theorem quadratic_coefficients :
  ∀ (x : ℝ), (5 * x^2 = 6 * x - 8) → (5 * x^2 - 6 * x + 8 = 0) :=
begin
  intro x,
  intro h,
  sorry
end

end quadratic_coefficients_l443_443047


namespace axis_of_symmetry_transformed_function_l443_443404

def original_function (x : ℝ) : ℝ :=
  2 * cos (4 * x - π / 3) + 1

def transformed_function (x : ℝ) : ℝ :=
  2 * cos (2 * x + π / 3) + 1

theorem axis_of_symmetry_transformed_function :
  ∃ x : ℝ, x = -π / 6 :=
sorry

end axis_of_symmetry_transformed_function_l443_443404


namespace num_consecutive_prime_products_le_900_l443_443688

def is_consecutive_prime_product (n : ℕ) : Prop :=
  ∃ k, ∃ primes : List ℕ, primes.length >= 2 ∧
    (∀ i, primes.nth i = prime) ∧ 
    (∀ i j, i < j → primes.nth i < primes.nth j) ∧
    primes.product = n

theorem num_consecutive_prime_products_le_900 : 
  (Finset.filter (λ n, is_consecutive_prime_product n) (Finset.range 900)).card = 14 := 
sorry

end num_consecutive_prime_products_le_900_l443_443688


namespace find_ellipse_eqn_line_mn_fixed_point_l443_443246

noncomputable def ellipse_eqn (a b : ℝ) (h : a > b ∧ b > 0 ∧ b = 1) : Prop :=
  (a^2 = b^2 + 1^2) ∧ (a^2 = 2)

theorem find_ellipse_eqn (a b : ℝ) (h : a > b ∧ b > 0 ∧ b = 1) :
  ellipse_eqn a b h → (C : set (ℝ × ℝ), (C = { p | (p.1^2 / 2) + p.2^2 = 1})) :=
  sorry

noncomputable def line_mn_through_fixed_point (k : ℝ) (h : - (Real.sqrt 2) / 2 < k ∧ k < 0) : Prop :=
  ∃ m : ℝ, (m = -2 * k) ∧ (∀ x : ℝ, y = k * (x - 2) → y = 0 → x = 2)

theorem line_mn_fixed_point (k : ℝ) (h : - (Real.sqrt 2) / 2 < k ∧ k < 0) :
  line_mn_through_fixed_point k h :=
  sorry

end find_ellipse_eqn_line_mn_fixed_point_l443_443246


namespace expandProduct_l443_443942

theorem expandProduct (x : ℝ) : 4 * (x - 5) * (x + 8) = 4 * x^2 + 12 * x - 160 := 
by 
  sorry

end expandProduct_l443_443942


namespace find_number_l443_443042

theorem find_number (f : ℝ → ℝ) (x : ℝ)
  (h : f (x * 0.004) / 0.03 = 9.237333333333334)
  (h_linear : ∀ a, f a = a) :
  x = 69.3 :=
by
  -- Proof goes here
  sorry

end find_number_l443_443042


namespace cone_to_sphere_volume_ratio_l443_443992

-- Define the radius r as a positive real number
variables (r : ℝ) (h : ℝ)
-- Assume the height h of the cone is equal to the radius r
hypothesis h_eq_r : h = r

-- Define the volume of the sphere with radius r
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

-- Define the volume of the cone with radius r and height h
def volume_cone (r h : ℝ) : ℝ := (1 / 3) * Real.pi * r^2 * h

-- The theorem to be proven
theorem cone_to_sphere_volume_ratio (r : ℝ) (h : ℝ) (h_eq_r : h = r) :
  (volume_cone r h) / (volume_sphere r) = 1 / 4 :=
by
  sorry

end cone_to_sphere_volume_ratio_l443_443992


namespace permutation_digits_yields_5445_l443_443938

theorem permutation_digits_yields_5445 :
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (9 * (a - b) = a + b) ∧ (10 * a + b) = 54 ∧ (10 * b + a) = 45 ∧ 
  (10 * (10 * a + b) + (10 * b + a) = 5445) :=
by
  unfold nat
  sorry

end permutation_digits_yields_5445_l443_443938


namespace product_union_perfect_square_l443_443998

noncomputable def H1 (n : ℕ) : Set ℕ := {x | ∃ i : ℕ, i < n ∧ x = 2 * i + 1}
noncomputable def H2 (n k : ℕ) : Set ℕ := {x | ∃ i : ℕ, i < n ∧ x = 2 * i + 1 + k}
noncomputable def product (s : Set ℕ) : ℕ := s.toFinset.prod id

theorem product_union_perfect_square (n k : ℕ) (hn : 0 < n) (hk : k = 2 * n + 1) :
  ∃ m : ℕ, product (H1 n ∪ H2 n k) = m * m := 
sorry

end product_union_perfect_square_l443_443998


namespace isosceles_triangle_formed_by_lines_l443_443432

theorem isosceles_triangle_formed_by_lines :
  let P1 := (1/4, 4)
  let P2 := (-3/2, -3)
  let P3 := (2, -3)
  let d12 := ((1/4 + 3/2)^2 + (4 + 3)^2)
  let d13 := ((1/4 - 2)^2 + (4 + 3)^2)
  let d23 := ((-3/2 - 2)^2)
  (d12 = d13) ∧ (d12 ≠ d23) → 
  ∃ (A B C : ℝ × ℝ), 
    A = P1 ∧ B = P2 ∧ C = P3 ∧ 
    ((dist A B = dist A C) ∧ (dist B C ≠ dist A B)) :=
by
  sorry

end isosceles_triangle_formed_by_lines_l443_443432


namespace triangle_isosceles_l443_443244

theorem triangle_isosceles 
  (α β γ δ : ℝ)
  (h_sum_angles : α + β + γ = 180)
  (h_quad_sum : (α + β) + (β + γ) + (α + γ) + δ = 360)
  (h_sum_exists : ∀ (x y ∈ {α, β, γ}), ∃ (z ∈ {α + β, β + γ, α + γ, δ}), z = x + y) :
  α = β ∨ β = γ ∨ γ = α := 
sorry

end triangle_isosceles_l443_443244


namespace v4_value_at_neg4_l443_443462

def polynomial (x : ℤ) : ℤ :=
  12 + 35 * x - 8 * x ^ 2 + 79 * x ^ 3 + 6 * x ^ 4 + 5 * x ^ 5 + 3 * x ^ 6

def v4 (x : ℤ) :=
  ((((3 * x + 5) * x + 6) * x + 79) * x - 8)

theorem v4_value_at_neg4 :
  v4 (-4) = 220 :=
by
  have v : ℤ := (-4)
  have p : ℤ := v4 v
  show p = 220
  sorry

end v4_value_at_neg4_l443_443462


namespace largest_possible_markers_in_package_l443_443152

theorem largest_possible_markers_in_package (alex_markers jordan_markers : ℕ) 
  (h1 : alex_markers = 56)
  (h2 : jordan_markers = 42) :
  Nat.gcd alex_markers jordan_markers = 14 :=
by
  sorry

end largest_possible_markers_in_package_l443_443152


namespace sqrt_a_squared_not_always_eq_a_l443_443479

variable (a : ℝ)

theorem sqrt_a_squared_not_always_eq_a : (sqrt (a ^ 2) ≠ a) :=
sorry

end sqrt_a_squared_not_always_eq_a_l443_443479


namespace min_value_of_T_l443_443276

noncomputable def T_min_value (a b c : ℝ) : ℝ :=
  (5 + 2*a*b + 4*a*c) / (a*b + 1)

theorem min_value_of_T :
  ∀ (a b c : ℝ),
  a < 0 →
  b > 0 →
  b^2 ≤ (4 * c) / a →
  c ≤ (1/4) * a * b^2 →
  T_min_value a b c ≥ 4 ∧ (T_min_value a b c = 4 ↔ a * b = -3) :=
by
  intros
  sorry

end min_value_of_T_l443_443276


namespace geometric_and_harmonic_mean_l443_443082

noncomputable def find_numbers : ℝ × ℝ :=
(list.to_finset
[ (5 + Real.sqrt 5) / 2,
  (5 - Real.sqrt 5) / 2]).prod !

theorem geometric_and_harmonic_mean
  (a b : ℝ)
  (h1: a * b = 5)
  (h2: 2 / (1/a + 1/b) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by {
  sorry
}

end geometric_and_harmonic_mean_l443_443082


namespace cricket_problem_l443_443309

theorem cricket_problem
  (x : ℕ)
  (run_rate_initial : ℝ := 3.8)
  (overs_remaining : ℕ := 40)
  (run_rate_remaining : ℝ := 6.1)
  (target_runs : ℕ := 282) :
  run_rate_initial * x + run_rate_remaining * overs_remaining = target_runs → x = 10 :=
by
  -- proof goes here
  sorry

end cricket_problem_l443_443309


namespace anna_money_ratio_l443_443196

theorem anna_money_ratio (total_money spent_furniture left_money given_to_Anna : ℕ)
  (h_total : total_money = 2000)
  (h_spent : spent_furniture = 400)
  (h_left : left_money = 400)
  (h_after_furniture : total_money - spent_furniture = given_to_Anna + left_money) :
  (given_to_Anna / left_money) = 3 :=
by
  have h1 : total_money - spent_furniture = 1600 := by sorry
  have h2 : given_to_Anna = 1200 := by sorry
  have h3 : given_to_Anna / left_money = 3 := by sorry
  exact h3

end anna_money_ratio_l443_443196


namespace trip_time_is_correct_l443_443335

noncomputable def total_trip_time : ℝ :=
  let wrong_direction_time := 100 / 60
  let return_time := 100 / 45
  let detour_time := 30 / 45
  let normal_trip_time := 300 / 60
  let stop_time := 2 * (15 / 60)
  wrong_direction_time + return_time + detour_time + normal_trip_time + stop_time

theorem trip_time_is_correct : total_trip_time = 10.06 :=
  by
    -- Proof steps are omitted
    sorry

end trip_time_is_correct_l443_443335


namespace example_number_divisibility_and_digits_l443_443796

def example_number : ℕ := 98987676545431312020

theorem example_number_divisibility_and_digits :
  ∃ example_number : ℕ,
    (example_number % 2020 = 0) ∧
    (∀ digit : ℕ, digit ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} → 
      count_digits example_number digit = (number_of_digits example_number) / 10) :=
begin
  use example_number,
  split,
  { -- example_number % 2020 = 0
    sorry },
  { -- Each digit from 0 to 9 appears exactly the same number of times
    intro digit,
    sorry }
end

end example_number_divisibility_and_digits_l443_443796


namespace M_lt_N_l443_443702

/-- M is the coefficient of x^4 y^2 in the expansion of (x^2 + x + 2y)^5 -/
def M : ℕ := 120

/-- N is the sum of the coefficients in the expansion of (3/x - x)^7 -/
def N : ℕ := 128

/-- The relationship between M and N -/
theorem M_lt_N : M < N := by 
  dsimp [M, N]
  sorry

end M_lt_N_l443_443702


namespace two_numbers_ratio_l443_443809

theorem two_numbers_ratio (A B : ℕ) (h_lcm : Nat.lcm A B = 30) (h_sum : A + B = 25) :
  ∃ x y : ℕ, x = 2 ∧ y = 3 ∧ A / B = x / y := 
sorry

end two_numbers_ratio_l443_443809


namespace find_t_l443_443669

variables (t : ℝ)

def a : ℝ × ℝ := (-1, 1)
def b : ℝ × ℝ := (3, t)
def a_plus_b : ℝ × ℝ := (2, 1 + t)

def are_parallel (u v : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, u = (k * v.1, k * v.2)

theorem find_t (t : ℝ) :
  are_parallel (3, t) (2, 1 + t) ↔ t = -3 :=
sorry

end find_t_l443_443669


namespace duration_of_talking_segments_l443_443891

theorem duration_of_talking_segments
  (total_show_time : ℕ)
  (ad_breaks : ℕ)
  (ad_break_duration : ℕ)
  (songs_duration : ℕ)
  (talking_segments : ℕ)
  (h1 : total_show_time = 180)
  (h2 : ad_breaks = 5)
  (h3 : ad_break_duration = 5)
  (h4 : songs_duration = 125)
  (h5 : talking_segments = 3) :
  (total_show_time - (ad_breaks * ad_break_duration + songs_duration)) / talking_segments = 10 :=
by
  rw [h1, h2, h3, h4, h5]
  norm_num
  sorry

end duration_of_talking_segments_l443_443891


namespace part_I_part_II_part_III_l443_443243

variable {λ : ℝ} {a : ℕ → ℝ} {S : ℕ → ℝ}

noncomputable def a_n (n : ℕ) : ℝ := 
  if n = 0 then 0 else (2 * n + 1) * λ ^ (n - 1)

def S_n (n : ℕ) : ℝ := ∑ i in Finset.range n, a_n i

theorem part_I (h₁ : λ > 0) (n : ℕ) (hn : n > 0) :
  a_n n = (2 * n + 1) * λ ^ (n - 1) := 
by sorry

theorem part_II (h₂ : λ = 4) (r s t : ℕ) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hne : r ≠ s) :
  ¬(∃ r s t, (a_n r) / (a_n s) = (a_n s) / (a_n t)) := 
by sorry

theorem part_III (n : ℕ) (hn : n > 0) :
  (1 - λ) * S_n n + λ * a_n n ≥ 3 :=
by sorry

end part_I_part_II_part_III_l443_443243


namespace product_of_repeating_decimal_l443_443580

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l443_443580


namespace number_of_fractions_l443_443907

noncomputable def is_fraction (expr : ℤ → ℤ → Prop) : Prop :=
  ∃ p q : ℤ, q ≠ 0 ∧ expr p q

noncomputable def expr1 (x : ℤ) : ℤ → ℤ → Prop := λ p q, p = 1 ∧ q = x ∧ x ≠ 0
noncomputable def expr2 : ℤ → ℤ → Prop := λ p q, p = 1 ∧ q = 2
noncomputable def expr3 (x : ℤ) : ℤ → ℤ → Prop := 
  if x ≠ 1 then λ p q, p = x^2 - 1 ∧ q = x - 1 else λ p q, false 
noncomputable def expr4 (x y : ℤ) : ℤ → ℤ → Prop := λ p q, p = 3 * x * y ∧ q = π.toInt
noncomputable def expr5 (x y : ℤ) : ℤ → ℤ → Prop := λ p q, p = 3 ∧ q = x + y ∧ x + y ≠ 0

theorem number_of_fractions (x y : ℤ) : 
  (is_fraction (expr1 x) ∧ is_fraction expr2 ∧ is_fraction (expr3 x) ∧ is_fraction (expr4 x y) ∧ is_fraction (expr5 x y)) ↔ 3 :=
sorry

end number_of_fractions_l443_443907


namespace frequency_of_sixth_group_l443_443640

theorem frequency_of_sixth_group :
  ∀ (total_data_points : ℕ)
    (freq1 freq2 freq3 freq4 : ℕ)
    (freq5_ratio : ℝ),
    total_data_points = 40 →
    freq1 = 10 →
    freq2 = 5 →
    freq3 = 7 →
    freq4 = 6 →
    freq5_ratio = 0.10 →
    (total_data_points - (freq1 + freq2 + freq3 + freq4) - (total_data_points * freq5_ratio)) = 8 :=
by
  sorry

end frequency_of_sixth_group_l443_443640


namespace matthew_total_time_on_malfunctioning_day_l443_443369

-- Definitions for conditions
def assembling_time : ℝ := 1
def normal_baking_time : ℝ := 1.5
def malfunctioning_baking_time : ℝ := 2 * normal_baking_time
def decorating_time : ℝ := 1

-- The theorem statement
theorem matthew_total_time_on_malfunctioning_day :
  assembling_time + malfunctioning_baking_time + decorating_time = 5 :=
by
  -- This is where the proof would go
  sorry

end matthew_total_time_on_malfunctioning_day_l443_443369


namespace even_derivative_is_odd_odd_derivative_is_even_l443_443170

variables {α : Type*} [normed_field α] [normed_space α]

def even_fun (f : α → α) : Prop := ∀ x, f (-x) = f x
def odd_fun (f : α → α) : Prop := ∀ x, f (-x) = -f x
def differentiable_fun (f : α → α) : Prop := differentiable α f

theorem even_derivative_is_odd (f : α → α) (h1 : differentiable_fun f) (h2 : even_fun f) : odd_fun (deriv f) :=
sorry

theorem odd_derivative_is_even (f : α → α) (h1 : differentiable_fun f) (h2 : odd_fun f) : even_fun (deriv f) :=
sorry

end even_derivative_is_odd_odd_derivative_is_even_l443_443170


namespace rectangle_circle_area_ratio_l443_443828

noncomputable def area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) : ℝ :=
  (2 * w^2) / (Real.pi * r^2)

theorem rectangle_circle_area_ratio (w r : ℝ) (h : 3 * w = Real.pi * r) :
  area_ratio w r h = 18 / (Real.pi * Real.pi) :=
by
  sorry

end rectangle_circle_area_ratio_l443_443828


namespace tire_circumference_l443_443867

theorem tire_circumference (rpm : ℕ) (speed_kmh : ℕ) (C : ℝ) 
  (h1 : rpm = 400) 
  (h2 : speed_kmh = 144) 
  (h3 : (speed_kmh * 1000 / 60) = (rpm * C)) : 
  C = 6 :=
by
  sorry

end tire_circumference_l443_443867


namespace greatest_cardinality_l443_443894

noncomputable def T_cardinality (T : Set ℕ) : Prop :=
  (∀ x : ℕ, x ∈ T -> (∑ y in T \ {x}, y) % (T.card - 1) = 0) ∧
  1 ∈ T ∧ 2 ∈ T ∧ (1001 ∈ T ∧ ∀ y : ℕ, y ∈ T -> y ≤ 1001) ∧
  T.card ≤ 26

theorem greatest_cardinality : 
  ∃ (T : Set ℕ), T_cardinality T :=
sorry

end greatest_cardinality_l443_443894


namespace amount_of_tin_in_new_mixture_l443_443520

def tin_in_alloy_A (weight_A : ℚ) : ℚ := (3/4) * weight_A
def tin_in_alloy_B (weight_B : ℚ) : ℚ := (3/8) * weight_B
def tin_in_alloy_C (weight_C : ℚ) : ℚ := (1/5) * weight_C

theorem amount_of_tin_in_new_mixture :
  tin_in_alloy_A 170 + tin_in_alloy_B 250 + tin_in_alloy_C 120 = 245.25 :=
by
  -- Proof goes here
  sorry

end amount_of_tin_in_new_mixture_l443_443520


namespace sum_of_ai_within_24_hours_is_16560_l443_443807

theorem sum_of_ai_within_24_hours_is_16560 :
  let a_i := λ i : ℕ => (720 * i) / 11
  let n := 22
  let S := ∑ i in (finset.range (n + 1)).filter (λ i => i > 0), a_i i
  S = 16560 :=
by
  sorry

end sum_of_ai_within_24_hours_is_16560_l443_443807


namespace number_of_digits_n_l443_443760

def is_perfect_cube (x : ℕ) : Prop := ∃ y : ℕ, y ^ 3 = x
def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y ^ 2 = x

theorem number_of_digits_n (n : ℕ) (h_pos : 0 < n) (h_div : 30 ∣ n)
  (h_cube : is_perfect_cube (n^2)) (h_square : is_perfect_square (n^3)) :
  nat.log10 (2^6 * 3^6 * 5^6) + 1 = 9 :=
by {
  sorry
}

end number_of_digits_n_l443_443760


namespace garden_area_change_l443_443140

noncomputable def rectangle_length : ℝ := 60
noncomputable def rectangle_width : ℝ := 15

noncomputable def rectangle_area : ℝ := rectangle_length * rectangle_width

noncomputable def rectangle_perimeter : ℝ := 2 * (rectangle_length + rectangle_width)

noncomputable def equilateral_triangle_side : ℝ := rectangle_perimeter / 3

noncomputable def equilateral_triangle_area : ℝ := (real.sqrt 3 / 4) * equilateral_triangle_side^2

noncomputable def area_difference : ℝ := equilateral_triangle_area - rectangle_area

theorem garden_area_change :
  let expected_area_change : ℝ := 625 * real.sqrt 3 - 900 in
  abs (area_difference - expected_area_change) < 1e-2 :=
by
  sorry

end garden_area_change_l443_443140


namespace maximize_sum_arithmetic_sequence_l443_443317

theorem maximize_sum_arithmetic_sequence :
  ∀ (a : ℕ → ℤ),
    a 2 = 2008 →
    a 2008 = a 2004 - 16 →
    let d := a 2004 - a 2008 in
    a 1 = a 2 - d →
    ∃ n, (n = 503 ∨ n = 504) ∧ ∀ m, m ≠ n → sum (fun i => a i) 1 m ≤ sum (fun i => a i) 1 n :=
sorry

end maximize_sum_arithmetic_sequence_l443_443317


namespace range_of_m_l443_443274

noncomputable def f (m x : ℝ) : ℝ := (m - 2 * Real.sin x) / (Real.cos x)

theorem range_of_m (h : ∀ x ∈ Ioo 0 (Real.pi / 2), ∀ y ∈ Ioo 0 (Real.pi / 2), x < y → f m x > f m y) : m ≤ 2 :=
  sorry

end range_of_m_l443_443274


namespace inscribed_circle_circumference_eq_arc_length_l443_443455

-- Definition of the given conditions
def arc_angle : ℝ := 120 * (Real.pi / 180)  -- 120 degrees in radians
def original_circle_radius (R : ℝ) : Prop := R > 0
def inscribed_circle_radius (R r : ℝ) : Prop := r = R / 3

-- The statement of the proof
theorem inscribed_circle_circumference_eq_arc_length
  (R r : ℝ)
  (h1 : original_circle_radius R)
  (h2 : inscribed_circle_radius R r)
  (h3 : arc_angle = 2 * Real.pi / 3) :
  2 * Real.pi * r = (2 * Real.pi * R) / 3 :=
by
  sorry

end inscribed_circle_circumference_eq_arc_length_l443_443455


namespace taller_tree_height_l443_443833

-- Define the heights of the trees and their relationship
variables (h : ℝ)
variables (ratio: ℝ) (height_difference: ℝ) (added_height: ℝ)

-- Define the conditions in Lean
def condition_1 (h : ℝ) := ratio = 5 / 7
def condition_2 (h : ℝ) := height_difference = 24
def condition_3 (h : ℝ) := added_height = 10
def condition_4 (h : ℝ) := ratio = ((h - height_difference) / h)

-- Goal is to show that the height of the taller tree is 84 feet
theorem taller_tree_height (h : ℝ) 
  (c1: condition_1 h) (c2: condition_2 h) (c3: condition_3 h) (c4: condition_4 h) : 
  h = 84 :=
sorry

end taller_tree_height_l443_443833


namespace fraction_paint_remaining_l443_443534

theorem fraction_paint_remaining 
  (original_paint : ℝ)
  (h_original : original_paint = 2) 
  (used_first_day : ℝ)
  (h_used_first_day : used_first_day = (1 / 4) * original_paint) 
  (remaining_after_first : ℝ)
  (h_remaining_first : remaining_after_first = original_paint - used_first_day) 
  (used_second_day : ℝ)
  (h_used_second_day : used_second_day = (1 / 3) * remaining_after_first) 
  (remaining_after_second : ℝ)
  (h_remaining_second : remaining_after_second = remaining_after_first - used_second_day) : 
  remaining_after_second / original_paint = 1 / 2 :=
by
  -- Proof goes here.
  sorry

end fraction_paint_remaining_l443_443534


namespace part1_part2_l443_443271

variables {α : Type*} [linear_ordered_field α]

-- Define the function f(x) = |x - 1| - 2|x + a|
def f (a x : α) : α := abs (x - 1) - 2 * abs (x + a)

-- Define the function g(x) = 1/2 * x + b
def g (b x : α) : α := (1 / 2) * x + b

-- Part 1: Prove the solution set of the inequality f(x) ≤ 0 given a = 1/2.
theorem part1 (x : α) :
  f (1 / 2 : α) x ≤ 0 ↔ x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 1) := sorry

-- Part 2: Prove that 2b - 3a > 2 given a ≥ -1 
-- and the function g(x) is always above the function f(x).
theorem part2 (a b : α) (h : a ≥ -1)
  (h_above : ∀ x, g b x ≥ f a x) :
  2 * b - 3 * a > 2 := sorry

end part1_part2_l443_443271


namespace smallest_int_x_log_condition_l443_443224

theorem smallest_int_x_log_condition (z : ℕ) (hz : 0 < z) :
  ∃ x : ℕ, (log 3 (x ^ 2) > log 27 (3 ^ (24 + 4 ^ (2 * z - 1)))) ∧ (∀ y : ℕ, 
  log 3 (y ^ 2) > log 27 (3 ^ (24 + 4 ^ (2 * z - 1))) → x ≤ y) :=
sorry

end smallest_int_x_log_condition_l443_443224


namespace sum_f_l443_443184

noncomputable def f : ℝ → ℝ :=
λ x, if -3 ≤ x ∧ x < -1 then -(x+2)^2 else if -1 ≤ x ∧ x < 3 then x else f (x - 6 * ⌊x / 6⌋)

theorem sum_f (n : ℕ) (h : n = 2012) :
  (∑ i in Finset.range n, f (i + 1)) = 338 := by
  sorry

end sum_f_l443_443184


namespace find_integer_divisible_by_24_with_cube_root_in_range_l443_443962

theorem find_integer_divisible_by_24_with_cube_root_in_range :
  ∃ (n : ℕ), (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ (24 ∣ n) ∧ n = 744 := by
    sorry

end find_integer_divisible_by_24_with_cube_root_in_range_l443_443962


namespace max_expression_value_l443_443010

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443010


namespace array_inversion_l443_443625

def is_inversion (arr : List ℕ) (p q : ℕ) : Prop :=
  p < q ∧ arr[p] > arr[q]

def inversion_count (arr : List ℕ) : ℕ :=
  List.foldr (λ p acc => 
    List.foldr (λ q acc_inner => 
      acc_inner + if is_inversion arr p q then 1 else 0) 0 (List.range (arr.length - p))) 0 (List.range (arr.length))

theorem array_inversion (arr : List ℕ) (h : arr = [2, 4, 3, 1]) : inversion_count arr = 4 :=
by
  sorry

end array_inversion_l443_443625


namespace brady_total_earnings_l443_443172

-- Let's define all the conditions given in the problem
def pay_per_card : ℝ := 0.70
def bonus_per_100_cards : ℝ := 10.0
def total_cards_transcribed : ℕ := 200

-- Now we will define the proof statement
theorem brady_total_earnings :
  (total_cards_transcribed * pay_per_card) + ((total_cards_transcribed / 100) * bonus_per_100_cards) = 160 := 
begin
  sorry, -- we do not provide the actual proof, just the statement
end

end brady_total_earnings_l443_443172


namespace joey_return_speed_l443_443746

-- Joey's distance for one-way trip
def distance_one_way : ℝ := 5

-- Joey's time for one-way trip
def time_one_way : ℝ := 1

-- Joey's one-way speed 
def speed_one_way : ℝ := distance_one_way / time_one_way

-- Total distance of round trip
def distance_round_trip : ℝ := 2 * distance_one_way

-- Average speed of round trip
def average_speed_round_trip : ℝ := 8

-- Total time for the round trip
def total_time_round_trip : ℝ := distance_round_trip / average_speed_round_trip

-- Time taken for the return trip
def time_return_trip : ℝ := total_time_round_trip - time_one_way

-- The return speed is the distance of the one-way trip divided by the return time.
theorem joey_return_speed : speed_one_way = 5 ∧ distance_round_trip = 10 ∧ average_speed_round_trip = 8 →
  (distance_one_way / (total_time_round_trip - time_one_way)) = 20 :=
by
  -- Placeholder for proof
  sorry

end joey_return_speed_l443_443746


namespace max_expression_value_l443_443032

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443032


namespace find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443976

theorem find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1 :
    ∃ n : ℕ, n > 0 ∧ (n % 24 = 0) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ n = 744 :=
by
  sorry

end find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443976


namespace smallest_n_l443_443600

theorem smallest_n (n : ℕ) (h : n > 2) :
  (∃ (a : ℕ → ℕ), (Nat.gcd (List.foldl Nat.gcd 0 (List.range n).map a)) =
    ∑ k in List.range (n - 1), ∑ j in List.range (n - k), 1 / Nat.gcd (a k) (a (k + j))) ↔ n = 4 :=
by sorry

end smallest_n_l443_443600


namespace f_even_f_increasing_on_positives_find_x_l443_443263

noncomputable theory

-- Define the function f
def f (x : ℝ) : ℝ := 2^x + 2^(-x)

-- 1. Prove that f(x) is an even function
theorem f_even : ∀ x : ℝ, f (-x) = f x :=
by
  sorry

-- 2. Prove that f(x) is increasing on (0, +∞)
theorem f_increasing_on_positives : ∀ x₁ x₂ : ℝ, (0 < x₁ ∧ x₁ < x₂) → f x₁ < f x₂ :=
by
  sorry

-- 3. Given f(x) = 5 * 2^(-x) + 3, find x
theorem find_x (x : ℝ) : f x = 5 * 2^(-x) + 3 → x = 2 :=
by
  sorry

end f_even_f_increasing_on_positives_find_x_l443_443263


namespace sum_of_powers_simplified_value_l443_443475

theorem sum_of_powers_simplified_value :
  let sum := -1^2022 + (-1)^2023 + 1^2024 - 1^2025
  in sum = -2 :=
by
  sorry

end sum_of_powers_simplified_value_l443_443475


namespace problem_A_l443_443356

def f (x : ℝ) := (2*x + 3) / (x + 2)

def S := {y : ℝ | ∃ x : ℝ, x ≥ 0 ∧ y = f x}

noncomputable def M := real.Sup S
noncomputable def m := real.Inf S

theorem problem_A :
  m = 3/2 ∧ m ∈ S ∧ M = 2 ∧ M ∉ S :=
by
  sorry

end problem_A_l443_443356


namespace sum_of_arithmetic_series_51_to_100_l443_443123

theorem sum_of_arithmetic_series_51_to_100 :
  let first_term := 51
  let last_term := 100
  let n := (last_term - first_term) + 1
  2 * (n / 2) * (first_term + last_term) / 2 = 3775 :=
by
  sorry

end sum_of_arithmetic_series_51_to_100_l443_443123


namespace time_spent_per_bone_l443_443402

theorem time_spent_per_bone
  (total_hours : ℤ) (number_of_bones : ℤ) 
  (h1 : total_hours = 206) 
  (h2 : number_of_bones = 206) :
  (total_hours / number_of_bones = 1) := 
by {
  -- proof would go here
  sorry
}

end time_spent_per_bone_l443_443402


namespace reflection_triangle_incicle_l443_443488

/-!
# Triangle reflection problem

Given an acute-angled triangle \( A_1 A_2 A_3 \), the foot of the altitude from \( A_i \) is \( K_i \), and the incircle touches the side opposite \( A_i \) at \( L_i \). The line \( K_1 K_2 \) is reflected in the line \( L_1 L_2 \), the line \( K_2 K_3 \) is reflected in the line \( L_2 L_3 \), and the line \( K_3 K_1 \) is reflected in the line \( L_3 L_1 \). Show that the three new lines form a triangle with vertices on the incircle.
-/

variable (A1 A2 A3 K1 K2 K3 L1 L2 L3 : Point)

-- Definitions and conditions
def acute_angled_triangle (A1 A2 A3 : Point) : Prop :=
  ∠A1 A2 A3 < 90 ∧ ∠A2 A3 A1 < 90 ∧ ∠A3 A1 A2 < 90

def foot_of_altitude (A_i K_i : Point) (△ : Triangle) : Prop :=
  -- Specification of foot of altitude condition

def incircle_touch (L_i : Point) (△ : Triangle) : Prop :=
  -- Specification that L_i is the point where the incircle touches the side opposite A_i

def reflected_line (K_i K_j L_i L_j : Point) : Line :=
  -- Specification of the line reflection

-- Theorem statement
theorem reflection_triangle_incicle :
  ∀ (A1 A2 A3 K1 K2 K3 L1 L2 L3 : Point)
  (h_triangle : acute_angled_triangle A1 A2 A3)
  (h_altitude1 : foot_of_altitude A1 K1 ⟨A1, A2, A3⟩)
  (h_altitude2 : foot_of_altitude A2 K2 ⟨A1, A2, A3⟩)
  (h_altitude3 : foot_of_altitude A3 K3 ⟨A1, A2, A3⟩)
  (h_incircle1 : incircle_touch L1 ⟨A1, A2, A3⟩)
  (h_incircle2 : incircle_touch L2 ⟨A1, A2, A3⟩)
  (h_incircle3 : incircle_touch L3 ⟨A1, A2, A3⟩)
  (h_reflect1 : reflected_line K1 K2 L1 L2)
  (h_reflect2 : reflected_line K2 K3 L2 L3)
  (h_reflect3 : reflected_line K3 K1 L3 L1),
  ∃ (T1 T2 T3 : Point), incircle_touch T1 ⟨A1, A2, A3⟩ ∧ incircle_touch T2 ⟨A1, A2, A3⟩ ∧ incircle_touch T3 ⟨A1, A2, A3⟩ ∧
  T1, T2, T3 form a triangle :=
by
  sorry

end reflection_triangle_incicle_l443_443488


namespace golden_section_AC_l443_443511

def golden_ratio := (1 + Real.sqrt 5) / 2
def golden_ratio_reciprocal := golden_ratio - 1

theorem golden_section_AC (AB : ℝ) (C : ℝ) (h : AB = 20 ∧ C = (AB * ((Real.sqrt 5 - 1) / 2) ∨ C = (20 - AB * ((Real.sqrt 5 - 1) / 2)))) :
  C = 10 * (Real.sqrt 5 - 1) ∨ C = 30 - 10 * Real.sqrt 5 :=
by
  sorry

end golden_section_AC_l443_443511


namespace equation_of_the_line_l443_443411

noncomputable def line_equation (t : ℝ) : (ℝ × ℝ) := (3 * t + 6, 5 * t - 7)

theorem equation_of_the_line : ∃ m b : ℝ, (∀ t : ℝ, ∃ (x y : ℝ), line_equation t = (x, y) ∧ y = m * x + b) ∧ m = 5 / 3 ∧ b = -17 :=
by
  sorry

end equation_of_the_line_l443_443411


namespace quadratic_root_inequality_l443_443213

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end quadratic_root_inequality_l443_443213


namespace product_of_repeating_decimal_l443_443576

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l443_443576


namespace calvin_weeks_buying_chips_l443_443564

variable (daily_spending : ℝ := 0.50)
variable (days_per_week : ℝ := 5)
variable (total_spending : ℝ := 10)
variable (spending_per_week := daily_spending * days_per_week)

theorem calvin_weeks_buying_chips :
  total_spending / spending_per_week = 4 := by
  sorry

end calvin_weeks_buying_chips_l443_443564


namespace find_XN_l443_443740

-- Given Conditions
variables {X Y Z N : Type} [topological_space X] [metric_space X]
variable (d : X → X → ℝ)
variables (XY YZ XZ : ℝ)

-- Midpoint condition
def is_midpoint (A B M : X) [metric_space X] : Prop :=
  d A M = d B M ∧ d A B = d A M + d B M

-- Equivalent Problem Statement
theorem find_XN 
  (XY YZ XZ : ℝ)
  (dXY : d X Y = 24)
  (dYZ : d Y Z = 24)
  (dXZ : d X Z = 28)
  (midpoint_N: is_midpoint Y Z N) :
  d X N = 12 * real.sqrt 3 :=
  sorry

end find_XN_l443_443740


namespace no_poly_2008_poly_exists_iff_l443_443876

-- Definition and statements for part (a)
def is_divisor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem no_poly_2008 : 
  ¬ ∃ P : ℕ → ℤ, ∀ d, is_divisor 2008 d → P d = 2008 / d :=
sorry

-- Definition and statements for part (b)
theorem poly_exists_iff {n : ℕ} : 
  (∃ P : ℕ → ℤ, ∀ d, is_divisor n d → P d = n / d) ↔ (n = 1 ∨ ∃ p, nat.prime p ∧ n = p) :=
sorry

end no_poly_2008_poly_exists_iff_l443_443876


namespace distance_to_Big_Rock_l443_443143

theorem distance_to_Big_Rock (rower_speed : ℝ) (river_speed : ℝ) (total_time : ℝ) : 
  rower_speed = 6 → river_speed = 2 → total_time = 1 → 
  let D := 8 / 3 in
  D = 2.67 := 
by 
  intros h1 h2 h3 
  let D := 8 / 3
  exact (eq.refl 2.67 : 8 / 3 = 2.67)

end distance_to_Big_Rock_l443_443143


namespace number_of_valid_permutations_l443_443763

noncomputable def is_permutation_of_set {α : Type*} [DecidableEq α] (s : Finset α) (l : List α) : Prop :=
l.perm s.to_list

theorem number_of_valid_permutations :
  {l : List ℕ // is_permutation_of_set {1, 2, 3, 4} l ∧ 
  (Nat.abs (l.nthLe 0 (by simp [Nat.lt_succ_self])) - 1) +
  (Nat.abs (l.nthLe 1 (by simp [Nat.lt_succ_self])) - 2) +
  (Nat.abs (l.nthLe 2 (by simp [Nat.lt_succ_self])) - 3) +
  (Nat.abs (l.nthLe 3 (by simp [Nat.lt_succ_self])) - 4) = 6 } = 9 := 
sorry

end number_of_valid_permutations_l443_443763


namespace find_positive_integer_l443_443954

theorem find_positive_integer : ∃ (n : ℤ), n > 0 ∧ (24 : ℤ) ∣ n ∧ (9 : ℝ) < (n : ℝ).cbrt ∧ (n : ℝ).cbrt < 9.1 ∧ n = 744 := by
  sorry

end find_positive_integer_l443_443954


namespace bricks_required_l443_443884

theorem bricks_required (L_courtyard W_courtyard L_brick W_brick : Real)
  (hcourtyard : L_courtyard = 35) 
  (wcourtyard : W_courtyard = 24) 
  (hbrick_len : L_brick = 0.15) 
  (hbrick_wid : W_brick = 0.08) : 
  (L_courtyard * W_courtyard) / (L_brick * W_brick) = 70000 := 
by
  sorry

end bricks_required_l443_443884


namespace repeating_decimal_to_fraction_l443_443610

-- Definition of the repeating decimal 0.7(36) as the fraction 27/37
theorem repeating_decimal_to_fraction : (0.7 + 0.0036 + 0.000036 + ...) = (27 / 37) := by
sorry

end repeating_decimal_to_fraction_l443_443610


namespace bob_speed_l443_443045

theorem bob_speed (v : ℝ) : (∀ v_a : ℝ, v_a > 120 → 30 / v_a < 30 / v - 0.5) → v = 40 :=
by
  sorry

end bob_speed_l443_443045


namespace expected_score_of_three_people_probability_bicycle_on_second_day_days_bicycle_higher_probability_l443_443941

-- Proof problem for Part 1
theorem expected_score_of_three_people : 
  let p1 := 2/3
  let p2 := 1/3
  let E (X : ℕ → ℝ) := 3 * ((p1 : ℝ)^3) + 4 * (3*(p1^2*p2)) + 5 * (3*(p1*p2^2)) + 6 * (p2^3)
  in E (λ x, x) = 4 :=
by
  sorry

-- Proof problem for Part 2 (i)
theorem probability_bicycle_on_second_day : 
  let p_bicycle_first_day := (4:ℝ)/5
  let p_bicycle_given_bicycle := (1:ℝ)/4
  let p_tram_given_bicycle := 1 - p_bicycle_given_bicycle
  let p_bicycle_given_tram := (2:ℝ)/3
  let p_tram_given_tram := 1 - p_bicycle_given_tram
  in (p_bicycle_first_day * p_bicycle_given_bicycle + (1 - p_bicycle_first_day) * p_bicycle_given_tram) = 1/3 :=
by
  sorry

-- Proof problem for Part 2 (ii)
theorem days_bicycle_higher_probability :
  let p_bicycle_first_day := (4:ℝ)/5
  let p_bicycle_given_bicycle := (1:ℝ)/4
  let p_tram_given_bicycle := 1 - p_bicycle_given_bicycle
  let p_bicycle_given_tram := (2:ℝ)/3
  let p_tram_given_tram := 1 - p_bicycle_given_tram
  let P_n : ℕ → ℝ := λ n, (8/17) + (28/85) * ((-5/12)^(n-1))
  in (list.filter (λ n, P_n n > 1/2) (list.range 16)).length = 2 :=
by
  sorry

end expected_score_of_three_people_probability_bicycle_on_second_day_days_bicycle_higher_probability_l443_443941


namespace range_of_a_l443_443661

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 
  if x >= 2 then (a - 1 / 2) * x 
  else a^x - 4

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) ↔ (1 < a ∧ a ≤ 3) :=
sorry

end range_of_a_l443_443661


namespace John_paid_4000_l443_443336

/-- John has to hire a lawyer and pays some amount upfront. The lawyer charges $100 per hour 
and works 50 hours in court, plus 2 times that long in prep time. John's brother pays half the fee.
The total payment was $8000. Prove that John paid $4000. -/
theorem John_paid_4000 
  (hourly_rate : ℕ := 100)
  (court_hours : ℕ := 50)
  (prep_hours : ℕ := 2 * court_hours)
  (total_hours : ℕ := court_hours + prep_hours)
  (total_fee : ℕ := total_hours * hourly_rate)
  (total_payment : ℕ := 8000)
  (brother_share : ℕ := total_payment / 2) :
  (john_paid : ℕ := total_payment / 2) = 4000 :=
by
  -- sorry to skip the proof
  sorry

end John_paid_4000_l443_443336


namespace smallest_value_expression_l443_443854

theorem smallest_value_expression (n : ℕ) (hn : n > 0) : (n = 8) ↔ ((n / 2) + (32 / n) = 8) := by
  sorry

end smallest_value_expression_l443_443854


namespace parallelogram_area_l443_443422

noncomputable def polynomial : Complex → Complex := fun z =>
  z^4 + 4 * Complex.I * z^3 + (-5 + 7 * Complex.I) * z^2 + (-10 - 4 * Complex.I) * z + (1 - 8 * Complex.I)

theorem parallelogram_area :
  let roots := polynomial.roots in
  roots.length = 4 →
  let a := roots[0]
  let b := roots[1]
  let c := roots[2]
  let d := roots[3]
  is_parallelogram a b c d →
  parallelogram_area a b c d = Real.sqrt 10 :=
by sorry

end parallelogram_area_l443_443422


namespace inscribed_circle_radius_l443_443897

theorem inscribed_circle_radius (a b c : ℝ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5)
  (h4 : a^2 + b^2 = c^2) : 
  ∃ r : ℝ, r = 1 :=
by
  -- Introducing variables and conditions
  let s := (a + b + c) / 2 -- semi-perimeter
  have hs1 : s = 6 := sorry -- Given sides a, b, c
  let area := (1 / 2) * a * b -- area of the right triangle
  have ha1 : area = 6 := sorry
  let r := area / s -- radius of the inscribed circle
  have hr1 : r = 1 := sorry
  exact ⟨1, hr1⟩ -- The radius r is 1

end inscribed_circle_radius_l443_443897


namespace OA_perp_OB_l443_443055

noncomputable def line_eq (x : ℝ) : ℝ := x - 2
noncomputable def parabola_eq (y : ℝ) : ℝ := y^2 / 2

theorem OA_perp_OB :
  ∃ (A B : ℝ × ℝ),
    (A.snd = line_eq (A.fst)) ∧
    (B.snd = line_eq (B.fst)) ∧
    (A.fst = parabola_eq (A.snd)) ∧
    (B.fst = parabola_eq (B.snd)) ∧
    let O := (0, 0) in
    ∥(O.1 - A.1, O.2 - A.2)∥ * ∥(O.1 - B.1, O.2 - B.2)∥ = ∥(A.1 - B.1, A.2 - B.2)∥^2 :=
sorry

end OA_perp_OB_l443_443055


namespace kittens_count_l443_443159

def cats_taken_in : ℕ := 12
def cats_initial : ℕ := cats_taken_in / 2
def cats_post_adoption : ℕ := cats_taken_in + cats_initial - 3
def cats_now : ℕ := 19

theorem kittens_count :
  ∃ k : ℕ, cats_post_adoption + k - 1 = cats_now :=
by
  use 5
  sorry

end kittens_count_l443_443159


namespace number_of_rectangles_sum_of_perimeters_l443_443759

theorem number_of_rectangles (n : ℕ) (hn : 0 < n) : 
  ∃ total_rectangles : ℕ, total_rectangles = (n^2 * (n-1)^2) / 4 := 
by 
  have valid_n := hn
  let total_rectangles := (n^2 * (n-1)^2) / 4 
  exact ⟨total_rectangles, sorry⟩

theorem sum_of_perimeters (n : ℕ) (hn : 0 < n) :
  ∃ total_perimeter : ℕ, total_perimeter = -- insert the derived formula here :=
by
  have valid_n := hn
  let total_perimeter := -- derived formula for sum of perimeters
  exact ⟨total_perimeter, sorry⟩

end number_of_rectangles_sum_of_perimeters_l443_443759


namespace find_larger_number_l443_443443

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end find_larger_number_l443_443443


namespace nine_questions_insufficient_l443_443466

/--
We have 5 stones with distinct weights and we are allowed to ask nine questions of the form
"Is it true that A < B < C?". Prove that nine such questions are insufficient to always determine
the unique ordering of these stones.
-/
theorem nine_questions_insufficient (stones : Fin 5 → Nat) 
  (distinct_weights : ∀ i j : Fin 5, i ≠ j → stones i ≠ stones j) :
  ¬ (∃ f : { q : Fin 125 | q.1 ≤ 8 } → (Fin 5 → Fin 5 → Fin 5 → Bool),
    ∀ w1 w2 w3 w4 w5 : Fin 120,
      (f ⟨0, sorry⟩) = sorry  -- This line only represents the existence of 9 questions
      )
:=
sorry

end nine_questions_insufficient_l443_443466


namespace second_box_capacity_l443_443508

-- Given conditions
def height1 := 4 -- height of the first box in cm
def width1 := 2 -- width of the first box in cm
def length1 := 6 -- length of the first box in cm
def clay_capacity1 := 48 -- weight capacity of the first box in grams

def height2 := 3 * height1 -- height of the second box in cm
def width2 := 2 * width1 -- width of the second box in cm
def length2 := length1 -- length of the second box in cm

-- Hypothesis: weight capacity increases quadratically with height
def quadratic_relationship (h1 h2 : ℕ) (capacity1 : ℕ) : ℕ :=
  (h2 / h1) * (h2 / h1) * capacity1

-- The proof problem
theorem second_box_capacity :
  quadratic_relationship height1 height2 clay_capacity1 = 432 :=
by
  -- proof omitted
  sorry

end second_box_capacity_l443_443508


namespace panda_on_stilts_height_l443_443194

theorem panda_on_stilts_height (x : ℕ) (h_A : ℕ) 
  (h1 : h_A = x / 4) -- A Bao's height accounts for 1/4 of initial total height
  (h2 : x - 40 = 3 * h_A) -- After breaking 20 dm off each stilt, the new total height is such that A Bao's height accounts for 1/3 of this new height
  : x = 160 := 
by
  sorry

end panda_on_stilts_height_l443_443194


namespace sum_of_x_coordinates_of_fourth_vertex_l443_443527

theorem sum_of_x_coordinates_of_fourth_vertex
    (A B C : ℝ×ℝ)
    (Ax_coord Ay_coord Bx_coord By_coord Cx_coord Cy_coord : ℝ)
    (hA : A = (Ax_coord, Ay_coord))
    (hB : B = (Bx_coord, By_coord))
    (hC : C = (Cx_coord, Cy_coord))
    (hA_coord : A = (1,2))
    (hB_coord : B = (3,8))
    (hC_coord : C = (4,1)) :
    let D1 := (Bx_coord + Cx_coord - Ax_coord, By_coord + Cy_coord - Ay_coord),
        D2 := (Ax_coord + Cx_coord - Bx_coord, Ay_coord + Cy_coord - By_coord),
        D3 := (Ax_coord + Bx_coord - Cx_coord, Ay_coord + By_coord - Cy_coord) in
    D1.1 + D2.1 + D3.1 = 8 := by
  sorry

end sum_of_x_coordinates_of_fourth_vertex_l443_443527


namespace maximum_value_is_17_l443_443022

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443022


namespace product_of_repeating_decimal_l443_443578

   -- Definitions
   def repeating_decimal : ℚ := 456 / 999  -- 0.\overline{456}

   -- Problem Statement
   theorem product_of_repeating_decimal (t : ℚ) (h : t = repeating_decimal) : (t * 7) = 1064 / 333 :=
   by
     sorry
   
end product_of_repeating_decimal_l443_443578


namespace experiment_arrangement_count_l443_443547

theorem experiment_arrangement_count :
  let experiments := [0, 1, 2, 3, 4, 5]
  ∃ n : ℕ, n = 300 ∧ 
    (n = (finset.permutations (experiments.erase_nth 0)).card / 2) := by
  sorry

end experiment_arrangement_count_l443_443547


namespace proof_c_value_l443_443775

open ProbabilityTheory

noncomputable def c_value (ξ : ℝ → Measure ℝ) (c : ℝ) : Prop :=
  c = 3 ∧
  (∀ x, ξ = Normal 2 9) ∧
  (probability_measure ξ) ∧
  (P (measurable_set (Set.Ioi (c + 1))) = P (measurable_set (Set.Iio (c - 3))))

theorem proof_c_value :
  c_value Normal(2, 9) 3 := sorry

end proof_c_value_l443_443775


namespace order_stats_dirichlet_dist_l443_443496

/--
Given independent random variables \(X_1, \ldots, X_n\) uniformly distributed on \([0,1]\),
and their order statistics \(X_{1:n}, \ldots, X_{n:n}\), for any 
\(1 \leq r_1 < \ldots < r_k \leq n\), the random vector 
\( \left( X_{r_1: n}, X_{r_2: n} - X_{r_1: n}, \ldots, X_{r_k: n} - X_{r_{k-1}: n} \right) \) 
has a Dirichlet distribution with parameters 
\( \left( r_1, r_2 - r_1, \ldots, r_k - r_{k-1}, n - r_k + 1 \right) \).
-/
theorem order_stats_dirichlet_dist
  {X : Type*} [measure_space X] [is_U :: random_variable X] 
  [∀ i, uniform (X i)]: 
  ∀ (n k : ℕ) (r : vector ℕ k), 
  (∀ i, 1 ≤ r i ∧ r i ≤ n) ∧ (∀ i j, i < j → r i < r j) →
  ∃ f : X → vector ℝ k, 
    (λ x, let ys := vector.of_fn (λ i, nth_order_stat x (r i) n) in
    let diff := vector.of_fn (λ i, ys (i + 1) - ys i) in 
    vector.cons (ys 0) diff (k - 1) :=
    @dirichlet_distribution (k + 1) 
      (vector.cons r.head (vector.map2 sub r.tail r.head) 
      ((vector.last r) - r.tail.head + 1))
      sorry

end order_stats_dirichlet_dist_l443_443496


namespace sixth_group_points_l443_443638

-- Definitions of conditions
def total_data_points : ℕ := 40

def group1_points : ℕ := 10
def group2_points : ℕ := 5
def group3_points : ℕ := 7
def group4_points : ℕ := 6
def group5_frequency : ℝ := 0.10

def group5_points : ℕ := (group5_frequency * total_data_points).toInt

-- Theorem: The number of data points in the sixth group
theorem sixth_group_points :
  group1_points + group2_points + group3_points + group4_points + group5_points + x = total_data_points →
  x = 8 :=
by
  sorry

end sixth_group_points_l443_443638


namespace volume_of_cube_l443_443449

theorem volume_of_cube (SA : ℝ) (h : SA = 486) : ∃ V : ℝ, V = 729 :=
by
  sorry

end volume_of_cube_l443_443449


namespace find_number_l443_443982

theorem find_number (n : ℕ) (h1 : 9 < real.cbrt n) (h2 : real.cbrt n < 9.1) (h3 : n % 24 = 0) : n = 744 :=
sorry

end find_number_l443_443982


namespace find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443957

theorem find_integer_divisible_by_24_and_cube_root_between_9_and_9_1 : 
  ∃ (n : ℕ), 
  (n % 24 = 0) ∧ 
  (9 < (n : ℚ) ^ (1 / 3 : ℚ)) ∧ 
  ((n : ℚ) ^ (1 / 3 : ℚ) < 9.1) ∧ 
  (n = 744) := by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443957


namespace integrality_condition_l443_443230

noncomputable def binom (n k : ℕ) : ℕ := 
  n.choose k

theorem integrality_condition (n k : ℕ) (h : 1 ≤ k) (h1 : k < n) (h2 : (k + 1) ∣ (n^2 - 3*k^2 - 2)) : 
  ∃ m : ℕ, m = (n^2 - 3*k^2 - 2) / (k + 1) ∧ (m * binom n k) % 1 = 0 :=
sorry

end integrality_condition_l443_443230


namespace circles_externally_tangent_l443_443433

-- Setting up the problem with given conditions as definitions
def circle1 : set (ℝ × ℝ) := {p | (p.1)^2 + (p.2)^2 = 1}

def circle2 : set (ℝ × ℝ) := {p | (p.1)^2 + (p.2)^2 - 8 * p.1 + 6 * p.2 + 9 = 0}

-- Define the distance formula between two points in ℝ²
def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

-- Define the centers and radii of the given circles using their standard forms
def center1 := (0, 0)
def radius1 := 1
def center2 := (4, -3)
def radius2 := 4

-- Statement to be proved: the circles are externally tangent
theorem circles_externally_tangent :
  distance center1 center2 = radius1 + radius2 :=
by sorry

end circles_externally_tangent_l443_443433


namespace min_sum_is_minimum_l443_443595

noncomputable def min_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem min_sum_is_minimum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ min_sum a b c ha hb hc = 3 / real.cbrt (162 : ℝ) :=
sorry

end min_sum_is_minimum_l443_443595


namespace a1998_eq_4494_l443_443438

def sequence (n : ℕ) : ℕ 
| 0 => 1
| (n + 1) => Nat.find (λ m => m > sequence n ∧ ∀ i j k ≤ (n + 1), i + j = 3 * k → False)

theorem a1998_eq_4494 : sequence 1997 = 4494 := 
by 
  sorry

end a1998_eq_4494_l443_443438


namespace matthew_total_time_on_failure_day_l443_443370

-- Define the conditions as variables
def assembly_time : ℝ := 1 -- hours
def usual_baking_time : ℝ := 1.5 -- hours
def decoration_time : ℝ := 1 -- hours
def baking_factor : ℝ := 2 -- Factor by which baking time increased on that day

-- Prove that the total time taken is 5 hours
theorem matthew_total_time_on_failure_day : 
  (assembly_time + (usual_baking_time * baking_factor) + decoration_time) = 5 :=
by {
  sorry
}

end matthew_total_time_on_failure_day_l443_443370


namespace number_of_worksheets_correct_l443_443909

noncomputable def study_time_each_day : Float := 4 - 1.67
noncomputable def total_study_time : Float := study_time_each_day * 4
noncomputable def time_for_chapters : Float := 3 * 2
noncomputable def remaining_time_for_worksheets : Float := total_study_time - time_for_chapters
noncomputable def worksheet_time : Float := 1.5
noncomputable def number_of_worksheets : Int := Int.floor (remaining_time_for_worksheets / worksheet_time)

theorem number_of_worksheets_correct :
  number_of_worksheets = 2 :=
  by
  sorry

end number_of_worksheets_correct_l443_443909


namespace cost_price_of_radio_l443_443410

theorem cost_price_of_radio (SP : ℝ) (L_p : ℝ) (C : ℝ) (h₁ : SP = 3200) (h₂ : L_p = 0.28888888888888886) 
  (h₃ : SP = C - (C * L_p)) : C = 4500 :=
by
  sorry

end cost_price_of_radio_l443_443410


namespace systematic_sampling_number_l443_443714

theorem systematic_sampling_number {n m s a b c d : ℕ} (h_n : n = 60) (h_m : m = 4) 
  (h_s : s = 3) (h_a : a = 33) (h_b : b = 48) 
  (h_gcd_1 : ∃ k, s + k * (n / m) = a) (h_gcd_2 : ∃ k, a + k * (n / m) = b) :
  ∃ k, s + k * (n / m) = d → d = 18 := by
  sorry

end systematic_sampling_number_l443_443714


namespace andrey_stamps_count_l443_443384

theorem andrey_stamps_count (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x ∧ x ≤ 300) → x = 208 := 
by 
  sorry

end andrey_stamps_count_l443_443384


namespace total_juice_consumed_l443_443147

noncomputable def cone_volume (r h : ℝ) : ℝ :=
  (1/3) * π * r^2 * h

def height_after_sips (h : ℝ) (sips : ℕ → ℝ) (m : ℕ) : ℝ :=
  h - ∑ n in range (m + 1), sips n

noncomputable def volume_after_sips (initial_volume : ℝ) (h : ℝ) (sips : ℕ → ℝ) (m : ℕ) : ℝ :=
  let hm := height_after_sips h sips m in
  initial_volume * (hm / h)^3

def sips (n : ℕ) : ℝ := 1 / (n ^ 2)

theorem total_juice_consumed (r h : ℝ) (ho : h = 9) (ro : r = 3) :
  (let in_volume := cone_volume r h in
   let remaining_volume := lim (λ m, volume_after_sips in_volume h sips m) in
   in_volume - remaining_volume = (216 * π^3 - 2187 * real.sqrt 3) / (8 * π^2)) := by
  admit

end total_juice_consumed_l443_443147


namespace milton_books_l443_443778

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l443_443778


namespace find_QS_l443_443458

-- Define the sides of the triangle PQR
def PQ : ℝ := 8
def QR : ℝ := 10
def PR : ℝ := 12

-- Define the speeds of the two bugs
def speed_bug1 : ℝ := 2
def speed_bug2 : ℝ := 3

-- Define the perimeter of the triangle
def perimeter : ℝ := PQ + QR + PR

-- Define the combined speed of the bugs
def combined_speed : ℝ := speed_bug1 + speed_bug2

-- Suppose the bugs meet at point S such that QS is the distance we want to prove
def meet_time : ℝ := perimeter / combined_speed
def QS : ℝ := 3 -- This is the result we aim to prove

theorem find_QS : QS = QR - (QR - (meet_time * speed_bug1 - PQ)) := by
  sorry

end find_QS_l443_443458


namespace unattainable_y_l443_443229

theorem unattainable_y (x : ℝ) (h1 : x ≠ -3/2) : y = (1 - x) / (2 * x + 3) -> ¬(y = -1 / 2) :=
by sorry

end unattainable_y_l443_443229


namespace find_c_l443_443739

noncomputable def cos_deg (θ : ℝ) : ℝ := Real.cos (θ * Real.pi / 180)

theorem find_c (a b c S : ℝ) (C : ℝ) 
  (ha : a = 3) 
  (hC : C = 120) 
  (hS : S = 15 * Real.sqrt 3 / 4) 
  (hab : a * b = 15)
  (hc2 : c^2 = a^2 + b^2 - 2 * a * b * cos_deg C) :
  c = 7 :=
by 
  sorry

end find_c_l443_443739


namespace evaluate_f_prime_at_two_l443_443859

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem evaluate_f_prime_at_two (a b : ℝ)
  (h₁ : f a b 1 = -2)
  (h₂ : (f' a b 1) = 0) :
  (f' (-2) (-2) 2) = -1/2 :=
by
  sorry

end evaluate_f_prime_at_two_l443_443859


namespace exists_special_permutation_l443_443193

/--
  There exists a permutation of all positive integers such that for any integer l (l ≥ 2),
  the sum of any l consecutive numbers in the permutation cannot be expressed in the form a^b,
  where a and b are both positive integers greater than or equal to 2.
-/
theorem exists_special_permutation : 
  ∃ (perm : ℕ → ℕ), 
  (∀ n : ℕ, 
    ∃ l : ℕ, 
    l ≥ 2 → 
    (∀ i : ℕ,
      0 ≤ i ∧ i + l ≤ n → 
      ¬ (∃ a b : ℕ, a ≥ 2 ∧ b ≥ 2 ∧ 
        (finset.range l).sum (λ k, perm (i + k)) = a^b)))
  :=
sorry

end exists_special_permutation_l443_443193


namespace factor_by_resultant_is_three_l443_443525

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end factor_by_resultant_is_three_l443_443525


namespace circumscribed_spheres_equal_l443_443756

-- Conditions:
variables {T₁ T₂ : Polyhedron}
variables (R₁ r₁ R₂ r₂ : ℝ)
variables (inscribed_equal : r₁ = r₂)

-- Question: If the inscribed spheres of T₁ and T₂ are equal, then the circumscribed spheres are also equal.
theorem circumscribed_spheres_equal 
    (dual_regular_polyhedra : Polyhedron.DualRegular T₁ T₂)
    (inscribed_spheres_equal : r₁ = r₂) 
    (circumscribed_r1 : T₁.CircumscribedSphereRadius = R₁)
    (inscribed_r1 : T₁.InscribedSphereRadius = r₁)
    (circumscribed_r2 : T₂.CircumscribedSphereRadius = R₂)
    (inscribed_r2 : T₂.InscribedSphereRadius = r₂) 
    : R₁ = R₂ :=
sorry

end circumscribed_spheres_equal_l443_443756


namespace find_integer_divisible_by_24_l443_443972

theorem find_integer_divisible_by_24 : 
  ∃ n : ℕ, (n % 24 = 0) ∧ (9 < real.sqrt (real.cbrt n)) ∧ (real.sqrt (real.cbrt n) < 9.1) := 
by
  let n := 744
  use n
  have h1 : n % 24 = 0 := by norm_num
  have h2 : 9 < real.sqrt (real.cbrt n) := by norm_num
  have h3 : real.sqrt (real.cbrt n) < 9.1 := by norm_num
  exact ⟨h1, h2, h3⟩

end find_integer_divisible_by_24_l443_443972


namespace series_sum_l443_443582

noncomputable def series_term (n : ℕ) : ℝ :=
  (4 * n + 2) / ((6 * n - 5)^2 * (6 * n + 1)^2)

theorem series_sum :
  (∑' n : ℕ, series_term (n + 1)) = 1 / 6 :=
by
  sorry

end series_sum_l443_443582


namespace geometric_sequence_fourth_term_l443_443515

theorem geometric_sequence_fourth_term :
  ∃ (a r : ℕ), a = 5 ∧ (a * r^4 = 1280) ∧ (a * r^3 = 320) :=
by {
  use 5, use 4,
  split, exact rfl,
  split,
  { norm_num, exact rfl },
  { norm_num, exact rfl }
}

end geometric_sequence_fourth_term_l443_443515


namespace domain_of_f_parity_of_f_range_of_g_l443_443264

noncomputable def f (x : ℝ) : ℝ := Real.logBase 3 ((1 - x) / (1 + x))

theorem domain_of_f : {x : ℝ | x ∈ set.Ioo (-1 : ℝ) 1} = {x : ℝ | (1 - x) / (1 + x) > 0} := by sorry

theorem parity_of_f : ∀ x : ℝ, f (-x) = -f (x) := by sorry

noncomputable def g (x : ℝ) : ℝ := f x

theorem range_of_g : ∀ x : ℝ, x ∈ set.Icc (-1 / 2) (1 / 2) → g x ∈ set.Icc (-1 : ℝ) 1 := by sorry

end domain_of_f_parity_of_f_range_of_g_l443_443264


namespace calculation_proof_l443_443554

theorem calculation_proof : (96 / 6) * 3 / 2 = 24 := by
  sorry

end calculation_proof_l443_443554


namespace Q_subset_P_l443_443755

def P : Set ℝ := { x | x < 4 }
def Q : Set ℝ := { x | x^2 < 4 }

theorem Q_subset_P : Q ⊆ P := by
  sorry

end Q_subset_P_l443_443755


namespace max_expression_value_l443_443033

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443033


namespace quadruple_count_l443_443635

theorem quadruple_count (n : ℕ) (hn : 0 < n) :
  (∑ (a : ℕ) in finset.range (n+1), ∑ (b : ℕ) in finset.range (a+1), 
   ∑ (c : ℕ) in finset.range (b+1), ∑ (d : ℕ) in finset.range (c+1), 1) = nat.choose (n+4) 4 :=
sorry

end quadruple_count_l443_443635


namespace modulus_of_z_l443_443302

-- Define the given condition
def condition (z : ℂ) : Prop := (z - 3) * (1 - 3 * Complex.I) = 10

-- State the main theorem
theorem modulus_of_z (z : ℂ) (h : condition z) : Complex.abs z = 5 :=
sorry

end modulus_of_z_l443_443302


namespace probability_of_three_draws_l443_443509

def chips : List ℕ := [1, 2, 3, 4, 5, 6, 7]

def draws_sum_exceeds (n : ℕ) (draws : List ℕ) : Prop :=
  ∑ draw in draws, draw > n

def exactly_three_draws (draws : List ℕ) : Prop :=
  draws.length = 3

def probability_three_draws_to_exceed_8 : ℚ :=
  5 / 21

theorem probability_of_three_draws (d : List ℕ) :
  (∃ draws, draws.perm d ∧ exactly_three_draws draws ∧ draws_sum_exceeds 8 draws) →
  probability_three_draws_to_exceed_8 = 5 / 21 :=
sorry

end probability_of_three_draws_l443_443509


namespace ball_total_distance_l443_443505

-- A helper function to calculate the distance traveled by the ball
noncomputable def total_distance (h0 : ℝ) (r : ℝ) (bounces : ℕ) : ℝ :=
  let ascent_height := λ n, h0 * (r ^ n)
  let descent_distance := λ n, if n = 0 then h0 else h0 * (r ^ (n - 1))
  (Finset.range bounces).sum descent_distance + (Finset.range bounces).sum ascent_height

theorem ball_total_distance :
  let h0 := 24
  let r := 5 / 8
  let bounces := 4
  total_distance h0 r bounces = 88.13 :=
by
  let h0 := 24
  let r := 5 / 8
  let bounces := 4
  have h_total_distance : total_distance h0 r bounces = 88.131859375 := sorry
  -- Rounding to 2 decimal places
  have h_rounded : Real.round (total_distance h0 r bounces * 100) / 100 = 88.13 := sorry
  exact h_rounded

end ball_total_distance_l443_443505


namespace polar_to_cartesian_l443_443125

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 4 * real.cos θ) :
  (let x := ρ * real.cos θ in
   let y := ρ * real.sin θ in
   (x - 2) ^ 2 + y ^ 2 = 4) :=
by
  sorry

end polar_to_cartesian_l443_443125


namespace augmented_matrix_correct_l443_443408

-- Defining the system of linear equations as assumptions
variables {x y : ℝ}

-- First equation
def eq1 : Prop := 3 * x + 2 * y = 5
-- Second equation
def eq2 : Prop := x + 2 * y = -1

-- The augmented matrix for the given system
def augmented_matrix : Matrix (Fin 2) (Fin 3) ℝ := ![
  [3, 2, 5],
  [1, 2, -1]
]

-- Tactic block containing the theorem statement
theorem augmented_matrix_correct {x y : ℝ} (h1 : eq1) (h2 : eq2) : 
  augmented_matrix = ![
    [3, 2, 5],
    [1, 2, -1]
  ] :=
by
  sorry

end augmented_matrix_correct_l443_443408


namespace calculate_value_of_rational_exponent_l443_443174

noncomputable def mixed_number_to_improper_fraction (a b c : ℤ) := a * c + b

theorem calculate_value_of_rational_exponent :
  (- (3 + 3 / 8 : ℚ)) ^ (- 2 / 3 : ℚ) = (4 / 9 : ℚ) :=
by
  sorry

end calculate_value_of_rational_exponent_l443_443174


namespace average_weight_estimation_exclude_friend_l443_443911

theorem average_weight_estimation_exclude_friend
    (w : ℝ)
    (H1 : 62.4 < w ∧ w < 72.1)
    (H2 : 60.3 < w ∧ w < 70.6)
    (H3 : w ≤ 65.9)
    (H4 : 63.7 < w ∧ w < 66.3)
    (H5 : 75.0 ≤ w ∧ w ≤ 78.5) :
    False ∧ ((63.7 < w ∧ w ≤ 65.9) → (w = 64.8)) :=
by
  sorry

end average_weight_estimation_exclude_friend_l443_443911


namespace part1_i_part1_ii_part2_l443_443278

-- Defining the sequence
def sequence (n : ℕ) : ℤ := 2^n - (-1)^n

-- Problem 1 (i): Prove that if a_{n_1}, a_{n_1 + 1}, a_{n_1 + 2} form an arithmetic sequence, then n_1 = 2.
theorem part1_i (n₁ : ℕ) (h_arith : 2 * sequence (n₁ + 1) = sequence n₁ + sequence (n₁ + 2)) : n₁ = 2 :=
sorry

-- Problem 1 (ii): Prove that if n₁ = 1 and a_1, a_{n₂}, a_{n₃} form an arithmetic sequence, then n₃ - n₂ = 1.
theorem part1_ii (n₂ n₃ : ℕ) (h_arith : 2 * sequence n₂ = sequence 1 + sequence n₃) : n₃ - n₂ = 1 :=
sorry

-- Problem 2: Prove that if a_{n₁}, a_{n₂}, ..., a_{nₜ} form an arithmetic sequence, then the maximum value of t is 3.
theorem part2 (t : ℕ) (hs : ∀ (n₁ n₂ nₜ : ℕ) (h1 : n₁ < n₂) (h2 : n₂ < nₜ) (h_arith : 2 * sequence n₂ = sequence n₁ + sequence nₜ), t <= 3) : t ≤ 3 :=
sorry

end part1_i_part1_ii_part2_l443_443278


namespace find_factor_l443_443522

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end find_factor_l443_443522


namespace find_functions_l443_443612

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem find_functions (f : ℝ × ℝ → ℝ) :
  (is_non_decreasing (λ x => f (0, x))) →
  (∀ x y, f (x, y) = f (y, x)) →
  (∀ x y z, (f (x, y) - f (y, z)) * (f (y, z) - f (z, x)) * (f (z, x) - f (x, y)) = 0) →
  (∀ x y a, f (x + a, y + a) = f (x, y) + a) →
  (∃ a : ℝ, (∀ x y, f (x, y) = a + min x y) ∨ (∀ x y, f (x, y) = a + max x y)) :=
  by sorry

end find_functions_l443_443612


namespace second_agency_charge_per_mile_l443_443049

theorem second_agency_charge_per_mile:
  ∀ (x : ℝ),
  let cost1 := 20.25 + 0.14 * 25.0 in
  let cost2 := 18.25 + x * 25.0 in
  cost1 = cost2 → x = 0.22 :=
by
  intro x
  let cost1 := 20.25 + 0.14 * 25.0
  let cost2 := 18.25 + x * 25.0
  intro h
  have : cost1 = cost2 := h
  sorry

end second_agency_charge_per_mile_l443_443049


namespace intersection_point_l443_443338

-- Definitions for the given functions with conditions
def f (x : ℝ) : ℝ := (x^2 - 8 * x + 12) / (2 * x - 6)

def g (x : ℝ) : ℝ := (-2 * x - 4 + 27 / (x - 3))

-- Prove that the given point is an intersection of f(x) and g(x) and does not lie on x = -3
theorem intersection_point : f 2.8 = g 2.8 ∧ 2.8 ≠ -3 :=
by 
  sorry

end intersection_point_l443_443338


namespace find_a_l443_443665

noncomputable def f : ℝ → ℝ :=
λ x, if x ≤ 0 then 2^x else x - 1

theorem find_a (a : ℝ) (h : f a = 1 / 2) : a = -1 ∨ a = 3 / 2 :=
sorry

end find_a_l443_443665


namespace residue_neg999_mod_25_is_1_l443_443599

theorem residue_neg999_mod_25_is_1 : ∃ r : ℤ, 0 ≤ r ∧ r < 25 ∧ (-999 % 25) = r := by
  let r := -999 % 25
  use r
  have : 0 ≤ r ∧ r < 25 := Int.mod_nonneg _ (by norm_num : 25 ≠ 0)
  sorry

end residue_neg999_mod_25_is_1_l443_443599


namespace right_square_pyramid_height_l443_443636

theorem right_square_pyramid_height :
  ∀ (h x : ℝ),
    let topBaseSide := 3
    let bottomBaseSide := 6
    let lateralArea := 4 * (1/2) * (topBaseSide + bottomBaseSide) * x
    let baseAreasSum := topBaseSide^2 + bottomBaseSide^2
    lateralArea = baseAreasSum →
    x = 5/2 →
    h = 2 :=
by
  intros h x topBaseSide bottomBaseSide lateralArea baseAreasSum lateralEq baseEq
  sorry

end right_square_pyramid_height_l443_443636


namespace sum_of_common_divisors_l443_443841

def common_divisors_sum (a b c d e : ℤ) : ℤ :=
  let common_divisors := set.to_finset {n | n > 0 ∧ n ∣ a ∧ n ∣ b ∧ n ∣ c ∧ n ∣ d ∧ n ∣ e}
  in common_divisors.sum id

theorem sum_of_common_divisors : 
  common_divisors_sum 48 64 (-16) 128 112 = 15 :=
sorry

end sum_of_common_divisors_l443_443841


namespace triangle_problem_proof_l443_443355

variables {A B C D E P : Type} [inner_point : P ∈ (triangle A B C)]
variables {AP BP AD AC BE : ℝ}

-- Given conditions
axiom equal_segments_AP_BP : AP = BP
axiom equal_segments_BE_CE : BE = CE

-- Definition of segments AD and AC
def AP_segments (AP AD AC : ℝ) : Prop :=
  (1 / AP) = (1 / AD) + (1 / AC)

-- The proof problem
theorem triangle_problem_proof : equal_segments_AP_BP ∧ equal_segments_BE_CE → AP_segments AP AD AC :=
by
  intros h
  sorry

end triangle_problem_proof_l443_443355


namespace find_absolute_difference_l443_443429

noncomputable def C := (c : ℝ) → (c^2 - 4/3, c)
noncomputable def D := (d : ℝ) → (d^2 - 4/3, d)
noncomputable def Q : ℝ × ℝ := (2 * Real.sqrt 3, 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem find_absolute_difference
  (c d : ℝ)
  (hC : C c ∈ {p : ℝ × ℝ | 3 * p.2^2 = 3 * p.1 + 4})
  (hD : D d ∈ {p : ℝ × ℝ | 3 * p.2^2 = 3 * p.1 + 4})
  (h_line : ∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.2 - 2 * p.1 * Real.sqrt 3 + 4 = 0} → p ∈ {C c, D d, Q}) :
  abs (distance (C c) Q - distance (D d) Q) = Real.sqrt 13 / 12 :=
sorry

end find_absolute_difference_l443_443429


namespace even_function_a_is_0_l443_443257

def f (a : ℝ) (x : ℝ) : ℝ := (a+1) * x^2 + 3 * a * x + 1

theorem even_function_a_is_0 (a : ℝ) : 
  (∀ x : ℝ, f a x = f a (-x)) → a = 0 :=
by sorry

end even_function_a_is_0_l443_443257


namespace find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443959

theorem find_integer_divisible_by_24_and_cube_root_between_9_and_9_1 : 
  ∃ (n : ℕ), 
  (n % 24 = 0) ∧ 
  (9 < (n : ℚ) ^ (1 / 3 : ℚ)) ∧ 
  ((n : ℚ) ^ (1 / 3 : ℚ) < 9.1) ∧ 
  (n = 744) := by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443959


namespace sine_of_angle_between_plane_and_diagonal_of_parallelepiped_l443_443817

theorem sine_of_angle_between_plane_and_diagonal_of_parallelepiped
    (a b c : ℝ) (hac : a = 3) (hbc : b = 4) (hcc : c = 12) :
    ∃ θ : ℝ, θ = real.sin (angle_between_plane_and_diagonal a b c) ∧ θ = 24/65 :=
by
  sorry

end sine_of_angle_between_plane_and_diagonal_of_parallelepiped_l443_443817


namespace initial_roses_l443_443843

theorem initial_roses (R : ℕ) (h : R + 16 = 23) : R = 7 :=
sorry

end initial_roses_l443_443843


namespace problem_sum_congruent_mod_11_l443_443607

theorem problem_sum_congruent_mod_11 : 
  (2 + 333 + 5555 + 77777 + 999999 + 11111111 + 222222222) % 11 = 3 := 
by
  -- Proof needed here
  sorry

end problem_sum_congruent_mod_11_l443_443607


namespace time_saved_by_both_trains_trainB_distance_l443_443086

-- Define the conditions
def trainA_speed_reduced := 360 / 12  -- 30 miles/hour
def trainB_speed_reduced := 360 / 8   -- 45 miles/hour

def trainA_speed := trainA_speed_reduced / (2 / 3)  -- 45 miles/hour
def trainB_speed := trainB_speed_reduced / (1 / 2)  -- 90 miles/hour

def trainA_time_saved := 12 - (360 / trainA_speed)  -- 4 hours
def trainB_time_saved := 8 - (360 / trainB_speed)   -- 4 hours

-- Prove that total time saved by both trains running at their own speeds is 8 hours
theorem time_saved_by_both_trains : trainA_time_saved + trainB_time_saved = 8 := by
  sorry

-- Prove that the distance between Town X and Town Y for Train B is 360 miles
theorem trainB_distance : 360 = 360 := by
  rfl

end time_saved_by_both_trains_trainB_distance_l443_443086


namespace midpoint_equidistant_l443_443849

-- Define the basic geometrical elements required
variables {A B C M N K P Q S : Point}
variable {Ω : Circle}

-- Conditions given in the problem
axiom AB_gt_BC : A.distance_to B > B.distance_to C
axiom inscribed : Triangle ABC.inscribed_in Ω
axiom AM_CN_eq : A.distance_to M = C.distance_to N
axiom MN_AC_intersect_K : Line MN ∩ Line AC = {K}
axiom P_incenter_AMK : incenter_of_triangle P A M K
axiom Q_excenter_CNK : excenter_of_triangle Q C N K (side_CN)
axiom S_midpoint_arc_ABC : midpoint_of_arc S A B C Ω

-- The statement to prove
theorem midpoint_equidistant (h : AB_gt_BC) : distance S P = distance S Q :=
by
  sorry

end midpoint_equidistant_l443_443849


namespace find_x_l443_443762

theorem find_x (x : ℚ) (h1 : 8 * x^2 + 9 * x - 2 = 0) (h2 : 16 * x^2 + 35 * x - 4 = 0) : 
  x = 1 / 8 :=
by sorry

end find_x_l443_443762


namespace find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443955

theorem find_integer_divisible_by_24_and_cube_root_between_9_and_9_1 : 
  ∃ (n : ℕ), 
  (n % 24 = 0) ∧ 
  (9 < (n : ℚ) ^ (1 / 3 : ℚ)) ∧ 
  ((n : ℚ) ^ (1 / 3 : ℚ) < 9.1) ∧ 
  (n = 744) := by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443955


namespace parabola_line_chord_length_l443_443991

-- Define the parabola and line as functions.
def parabola (x : ℝ) : ℝ := (12 * x)^(1/2)
def line (x : ℝ) : ℝ := 2 * x + 1

-- The length of the chord formed by their intersection.
def chord_length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2)

-- The main statement to be proved.
theorem parabola_line_chord_length :
  chord_length
    (classical.some (exists_quad_solution (λ y, y^2 = 12 * (y - 1)/2 + 1)))
    (classical.some (exists_quad_solution (λ y, y^2 = 12 * (y - 1)/2 + 1))) = real.sqrt 15 :=
sorry

end parabola_line_chord_length_l443_443991


namespace sum_of_undefined_points_l443_443936

theorem sum_of_undefined_points : 
  (0 + (-1) + (-2/3) = -5/3) ↔ 
  (∀ x, g(x) = (1 / (2 + (1 / (1 + (1 / x))))) → (x = 0 ∨ x = -1 ∨ x = -2/3)) := sorry

end sum_of_undefined_points_l443_443936


namespace fraction_value_l443_443299

theorem fraction_value (x y : ℕ) (hx : x = 3) (hy : y = 4) :
  (1 / (y : ℚ) / (1 / (x : ℚ))) = 3 / 4 :=
by
  rw [hx, hy]
  norm_num

end fraction_value_l443_443299


namespace digit_count_satisfying_conditions_l443_443681

theorem digit_count_satisfying_conditions : 
  (∃ f : Fin 4 → Fin 10, (f 0 = 1 ∨ f 0 = 4 ∨ f 0 = 5) ∧ 
                        (f 1 = 1 ∨ f 1 = 4 ∨ f 1 = 5) ∧ 
                        (f 2 = 5 ∨ f 2 = 7 ∨ f 2 = 8) ∧ 
                        (f 3 = 5 ∨ f 3 = 7 ∨ f 3 = 8) ∧ 
                        f 2 ≠ f 3) →
                        (∃ n : ℕ, n = 54) :=
begin
  sorry
end

end digit_count_satisfying_conditions_l443_443681


namespace fatima_donates_75_sq_inches_l443_443205

/-- Fatima starts with 100 square inches of cloth and cuts it in half twice.
    The total amount of cloth she donates should be 75 square inches. -/
theorem fatima_donates_75_sq_inches:
  ∀ (cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second: ℕ),
  cloth_initial = 100 → 
  cloth_after_first_cut = cloth_initial / 2 →
  cloth_donated_first = cloth_initial / 2 →
  cloth_after_second_cut = cloth_after_first_cut / 2 →
  cloth_donated_second = cloth_after_first_cut / 2 →
  cloth_donated_first + cloth_donated_second = 75 := 
by
  intros cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second
  intros h_initial h_after_first h_donated_first h_after_second h_donated_second
  sorry

end fatima_donates_75_sq_inches_l443_443205


namespace find_integer_divisible_by_24_with_cube_root_in_range_l443_443963

theorem find_integer_divisible_by_24_with_cube_root_in_range :
  ∃ (n : ℕ), (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ (24 ∣ n) ∧ n = 744 := by
    sorry

end find_integer_divisible_by_24_with_cube_root_in_range_l443_443963


namespace smallest_value_abs_z_add_i_is_three_l443_443770

noncomputable def smallest_value_abs_z_add_i (z : ℂ) (h : |z^2 + 16| = |z * (z + 4 * complex.I)|) : ℝ :=
  Inf (set_of (λ r, ∃ x : ℂ, |x + complex.I| = r))
  
theorem smallest_value_abs_z_add_i_is_three (z : ℂ) (h : |z^2 + 16| = |z * (z + 4 * complex.I)|) : 
  smallest_value_abs_z_add_i z h = 3 :=
sorry

end smallest_value_abs_z_add_i_is_three_l443_443770


namespace sin_double_angle_sum_l443_443254

theorem sin_double_angle_sum (α : ℝ) (h₁ : 0 < α ∧ α < π / 2) (h₂ : cos (α + π / 6) = -1 / 3): 
  sin (2 * α + π / 12) = (7 * sqrt 2 - 8) / 18 :=
by
  sorry

end sin_double_angle_sum_l443_443254


namespace coloring_books_gotten_rid_of_l443_443144

theorem coloring_books_gotten_rid_of :
  ∀ (initial_books remaining_books books_per_shelf shelves : ℕ),
    initial_books = 86 →
    books_per_shelf = 7 →
    shelves = 7 →
    remaining_books = books_per_shelf * shelves →
    initial_books - remaining_books = 37 :=
by
  intros initial_books remaining_books books_per_shelf shelves
  assume h1 : initial_books = 86
  assume h2 : books_per_shelf = 7
  assume h3 : shelves = 7
  assume h4 : remaining_books = books_per_shelf * shelves
  sorry

end coloring_books_gotten_rid_of_l443_443144


namespace cone_surface_area_l443_443701

theorem cone_surface_area 
  (A : Real) (r l : Real)
  (h_eq_triangle : A = sqrt 3)
  (axial_section : A = (sqrt 3 / 4) * r^2)
  (height : r^2 - (r / 2)^2 = 3) :
  (2 * π * sqrt 3 + 3 * π) = π * (2 * sqrt 3 + 3) :=
by
  sorry

end cone_surface_area_l443_443701


namespace focal_length_curve_l443_443821
-- Lean 4 statement


theorem focal_length_curve (k : ℝ) :
  (sqrt (1 + 5 / k) = 2 ∨ sqrt (-5 / k - 1) = 2) → (k = 5 / 3 ∨ k = -1) :=
by 
  sorry

end focal_length_curve_l443_443821


namespace diagonals_not_parallel_to_sides_in_32_polygon_l443_443682

theorem diagonals_not_parallel_to_sides_in_32_polygon : 
  let n := 32,
      total_diagonals := (n * (n - 3)) / 2,
      pairs_of_parallel_sides := n / 2,
      diagonals_parallel_to_each_pair := (n - 4) / 2,
      total_diagonals_parallel := 14 * pairs_of_parallel_sides
  in total_diagonals - total_diagonals_parallel = 240 :=
by
  let n := 32
  let total_diagonals := (n * (n - 3)) / 2
  let pairs_of_parallel_sides := n / 2
  let diagonals_parallel_to_each_pair := (n - 4) / 2
  let total_diagonals_parallel := 14 * pairs_of_parallel_sides
  have h_total_diagonals : total_diagonals = 464 := by sorry
  have h_total_diagonals_parallel_sides : total_diagonals_parallel = 224 := by sorry
  calc 
    total_diagonals - total_diagonals_parallel
        = 464 - 224 : by rw [h_total_diagonals, h_total_diagonals_parallel_sides]
    ... = 240 : by norm_num

end diagonals_not_parallel_to_sides_in_32_polygon_l443_443682


namespace retail_price_per_book_l443_443486

theorem retail_price_per_book (n r w : ℝ)
  (h1 : r * n = 48)
  (h2 : w = r - 2)
  (h3 : w * (n + 4) = 48) :
  r = 6 := by
  sorry

end retail_price_per_book_l443_443486


namespace find_integer_divisible_by_24_l443_443970

theorem find_integer_divisible_by_24 : 
  ∃ n : ℕ, (n % 24 = 0) ∧ (9 < real.sqrt (real.cbrt n)) ∧ (real.sqrt (real.cbrt n) < 9.1) := 
by
  let n := 744
  use n
  have h1 : n % 24 = 0 := by norm_num
  have h2 : 9 < real.sqrt (real.cbrt n) := by norm_num
  have h3 : real.sqrt (real.cbrt n) < 9.1 := by norm_num
  exact ⟨h1, h2, h3⟩

end find_integer_divisible_by_24_l443_443970


namespace green_chips_count_l443_443069

theorem green_chips_count (total_chips : ℕ) (blue_chips : ℕ) (red_chips : ℕ) (green_chips : ℕ)
  (h1 : total_chips = 60)
  (h2 : blue_chips = total_chips / 6)
  (h3 : red_chips = 34)
  (h4 : green_chips = total_chips - blue_chips - red_chips) :
  green_chips = 16 :=
by {
  -- Define the intermediate steps
  have h_blue_calculation : blue_chips = 10,
  { rw h1, exact Nat.div_eq_of_eq_mul_right (Nat.succ_pos 5) rfl },

  -- Assume that 34 red chips are given
  have h_red_count : red_chips = 34 := h3,

  -- Define the total number of chips
  have h_total_calculation : green_chips = 60 - 10 - 34
    by { rw [h1, h_blue_calculation, h_red_count] },

  -- Conclusion
  exact by { rw h_total_calculation, norm_num }
}

end green_chips_count_l443_443069


namespace find_a_l443_443658

theorem find_a (a : ℝ) : ∃ a, (∀ x : ℝ, y = x^4 + a * x^2 + 1 ∧ 
                                     (tangent_slope (x) = 4 * (-1)^3 + 2 * a * (-1) = 8 ∧ 
                                     point_on_curve (-1) = a + 2)) → 
                                     a = -6 := by sorry

end find_a_l443_443658


namespace tree_planting_activity_l443_443517

theorem tree_planting_activity (x y : ℕ) 
  (h1 : y = 2 * x + 15)
  (h2 : x = y / 3 + 6) : 
  y = 81 ∧ x = 33 := 
by sorry

end tree_planting_activity_l443_443517


namespace stock_decrease_l443_443799

theorem stock_decrease (v : ℝ) : 
  let value_after_day_one := 0.7 * v in
  let value_after_day_two := value_after_day_one + 0.4 * value_after_day_one in
  value_after_day_two = 0.98 * v :=
by
  let value_after_day_one := 0.7 * v
  let value_after_day_two := value_after_day_one + 0.4 * value_after_day_one
  calc
    value_after_day_two = value_after_day_one + 0.4 * value_after_day_one : by rfl
    ... = 0.7 * v + 0.4 * (0.7 * v) : by rfl
    ... = 0.7 * v + 0.28 * v : by rfl
    ... = 0.98 * v : by ring

end stock_decrease_l443_443799


namespace calc1_calc2_calc3_calc4_l443_443922

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end calc1_calc2_calc3_calc4_l443_443922


namespace find_factor_l443_443523

-- Defining the given conditions
def original_number : ℕ := 7
def resultant (x: ℕ) : ℕ := 2 * x + 9
def condition (x f: ℕ) : Prop := (resultant x) * f = 69

-- The problem statement
theorem find_factor : ∃ f: ℕ, condition original_number f ∧ f = 3 :=
by
  sorry

end find_factor_l443_443523


namespace quadratic_roots_square_diff_l443_443303

theorem quadratic_roots_square_diff (α β : ℝ) (h : α ≠ β)
    (hα : α^2 - 3 * α + 2 = 0) (hβ : β^2 - 3 * β + 2 = 0) :
    (α - β)^2 = 1 :=
sorry

end quadratic_roots_square_diff_l443_443303


namespace find_last_product_l443_443151

theorem find_last_product (v w x y z : ℕ)
  (h : ∀ (a b c : ℕ), {a, b, c} ⊆ {v, w, x, y, z} →
    (∀ p ∈ {v, w, x, y, z}.powerset.filter (λ s, s.card = 3).map (λ s, s.prod), 
      p ∈ {1, 2, 3, 4, 5, 6, 10, 12, 15, 30})) :
  ∃ last_product, last_product = 30 := 
sorry

end find_last_product_l443_443151


namespace sin_ratio_area_of_triangle_l443_443324

noncomputable theory

variables {A B C a b c : ℝ}

-- Definitions based on conditions
def law_of_sines := (∀ a b c A B C : ℝ , b ≠ 0 ∧ c ≠ 0 ∧ a ≠ 0 → (sin A / a = sin B / b ∧ sin B / b = sin C / c ∧ sin A / a = sin C / c))
def cosine_condition := b * (cos A - 2 * cos C) = (2 * c - a) * cos B
def given_cos_B := cos B = 1 / 4
def given_b := b = 2

-- Problem 1
theorem sin_ratio (h1 : cosine_condition) (h2 : law_of_sines a b c A B C) : sin A / sin C = 1 / 2 := 
sorry

-- Problem 2
theorem area_of_triangle (h1 : given_cos_B) (h2 : given_b) (h3 : sin_ratio cosine_condition (law_of_sines a b c A B C))
  (h4 : 0 < B ∧ B < π) : 
  let a := 1,
      c := 2,
      sin_B := sqrt (1 - (cos B) ^ 2) 
  in 
  1 / 2 * a * c * sin_B = sqrt 15 / 4 := 
sorry

end sin_ratio_area_of_triangle_l443_443324


namespace polygon_chessboard_probability_l443_443531

theorem polygon_chessboard_probability 
  (n : ℕ) 
  (square_side_length : Real := 4) 
  (polygon_radius : Real := 1) 
  : 
  let P := (4 * n / Real.pi) * Real.sin (Real.pi / (4 * n)) - 
           (n / (8 * Real.pi)) * Real.sin (Real.pi / (2 * n)) - 
           (1 / 8) 
  in 
  True := sorry

end polygon_chessboard_probability_l443_443531


namespace multiplication_of_monomials_l443_443915

-- Define the constants and assumptions
def a : ℝ := -2
def b : ℝ := 4
def e1 : ℤ := 4
def e2 : ℤ := 5
def result : ℝ := -8
def result_exp : ℤ := 9

-- State the theorem to be proven
theorem multiplication_of_monomials :
  (a * 10^e1) * (b * 10^e2) = result * 10^result_exp := 
by
  sorry

end multiplication_of_monomials_l443_443915


namespace binomial_probability_l443_443532

theorem binomial_probability (n : ℕ) (p : ℝ) (h1 : (n * p = 300)) (h2 : (n * p * (1 - p) = 200)) :
    p = 1 / 3 :=
by
  sorry

end binomial_probability_l443_443532


namespace milton_books_l443_443779

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l443_443779


namespace sum_of_special_numbers_eq_31_l443_443621

/-- 
  We aim to prove the sum of all positive integers n that have the following properties:
  1. All digits of n are less than 5.
  2. The representation of n in base 5 is the reverse of its representation in base 9.
  We need to show that this sum equals 31.
--/
theorem sum_of_special_numbers_eq_31 : 
  ∑ n in { n : ℕ | ∃ (d : ℕ) (a : ℕ → ℕ) (h1 : ∀ i ≤ d, a i < 5) (h2 : n = ∑ i in Finset.range (d+1), 5^i * a i) 
    (h3 : n = ∑ i in Finset.range (d+1), 9^(d-i) * a i)}, n = 31 := 
by
  sorry

end sum_of_special_numbers_eq_31_l443_443621


namespace quadrilateral_is_square_l443_443872

theorem quadrilateral_is_square
  (A B C D P Q R S : Point)
  (AB BC CD DA : ℝ)
  (h1 : dist A B = AB) (h2 : dist B C = BC)(h3 : dist C D = CD)(h4 : dist D A = DA)
  (h5 : dist A P = 2 * AB) (h6 : dist B Q = 2 * BC)(h7 : dist C R = 2 * CD)(h8 : dist D S = 2 * DA)
  (h9 : is_square P Q R S) : 
  is_square A B C D :=
sorry

end quadrilateral_is_square_l443_443872


namespace at_least_one_less_than_two_l443_443632

theorem at_least_one_less_than_two (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y > 2) : 
  (1 + y) / x < 2 ∨ (1 + x) / y < 2 := 
by 
  sorry

end at_least_one_less_than_two_l443_443632


namespace find_number_multiplied_by_48_eq_173_times_240_l443_443097

theorem find_number_multiplied_by_48_eq_173_times_240 : ∃ x : ℕ, 48 * x = 173 * 240 :=
by
  use 865
  sorry

end find_number_multiplied_by_48_eq_173_times_240_l443_443097


namespace sphere_volume_l443_443446

theorem sphere_volume (π : ℝ) (r : ℝ):
  4 * π * r^2 = 144 * π →
  (4 / 3) * π * r^3 = 288 * π :=
by
  sorry

end sphere_volume_l443_443446


namespace complex_number_solution_l443_443657

theorem complex_number_solution (z : ℂ) (i : ℂ) (h : i^2 = -1) (h_i : z * i = 2 + i) : z = 1 - 2 * i := by
  sorry

end complex_number_solution_l443_443657


namespace larger_number_solution_l443_443445

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end larger_number_solution_l443_443445


namespace cube_surface_area_l443_443622

theorem cube_surface_area (P : ℝ) (hP : P = 24) : 
  let s := P / 4 in
  let SA := 6 * s^2 in
  SA = 216 := 
by
  sorry

end cube_surface_area_l443_443622


namespace sample_size_proof_l443_443513

-- Define the quantities produced by each workshop
def units_A : ℕ := 120
def units_B : ℕ := 80
def units_C : ℕ := 60

-- Define the number of units sampled from Workshop C
def samples_C : ℕ := 3

-- Calculate the total sample size n
def total_sample_size : ℕ :=
  let sampling_fraction := samples_C / units_C
  let samples_A := sampling_fraction * units_A
  let samples_B := sampling_fraction * units_B
  samples_A + samples_B + samples_C

-- The theorem we want to prove
theorem sample_size_proof : total_sample_size = 13 :=
by sorry

end sample_size_proof_l443_443513


namespace part1_part2_l443_443272

noncomputable def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|
noncomputable def g (x b : ℝ) : ℝ := 0.5 * x + b

theorem part1 (a : ℝ) (h : a = 1/2) : 
  { x : ℝ | f x a ≤ 0 } = { x : ℝ | x ≤ -2 ∨ x ≥ 0 } :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ -1) (h2 : ∀ x, g x b ≥ f x a) : 
  2 * b - 3 * a > 2 :=
sorry

end part1_part2_l443_443272


namespace vector_sum_projections_le_l443_443284

theorem vector_sum_projections_le 
  (n m : ℕ)
  (a : ℕ → ℝ) (b : ℕ → ℝ) -- lengths of vectors in first and second set respectively
  (α : ℕ → ℝ) (β : ℕ → ℝ) -- angles of vectors in first and second set w.r.t. Ox-axis
  (h : ∀ (ϕ : ℝ), 
    ∑ i in finset.range n, |a i * real.cos (ϕ - α i)| 
    ≤ ∑ j in finset.range m, |b j * real.cos (ϕ - β j)|
  ) :
  ∑ i in finset.range n, a i ≤ ∑ j in finset.range m, b j :=
begin
  sorry
end

end vector_sum_projections_le_l443_443284


namespace cubic_reflection_translation_linear_l443_443885

theorem cubic_reflection_translation_linear (a b c d : ℝ) :
    let f := λ x : ℝ, a * (x - 10)^3 + b * (x - 10)^2 + c * (x - 10) + d,
        g := λ x : ℝ, -a * (x + 10)^3 - b * (x + 10)^2 - c * (x + 10) - d in
    (λ x : ℝ, f x + g x) = λ x, -20 * c * x :=
by {
    intros,
    sorry
}

end cubic_reflection_translation_linear_l443_443885


namespace simplify_f_value_of_f_l443_443261

def f (α : ℝ) : ℝ := 
  (sin (π / 2 - α) * cos (10 * π - α) * tan (-α + 3 * π)) / 
  (tan (π + α) * sin (5 * π / 2 + α))

theorem simplify_f : ∀ α : ℝ, f α = -cos α := 
by {
  intro α,
  sorry
}

theorem value_of_f (α : ℝ) 
  (h_range : 0 < α ∧ α < π / 2) 
  (h_sin : sin (α - π / 6) = 1 / 3) : 
  f α = (1 - 2 * sqrt 6) / 6 := 
by {
  sorry
}

end simplify_f_value_of_f_l443_443261


namespace max_value_of_fraction_l443_443040

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443040


namespace value_of_r_minus_p_l443_443115

variable (p q r : ℝ)

-- The conditions given as hypotheses
def arithmetic_mean_pq := (p + q) / 2 = 10
def arithmetic_mean_qr := (q + r) / 2 = 25

-- The goal is to prove that r - p = 30
theorem value_of_r_minus_p (h1: arithmetic_mean_pq p q) (h2: arithmetic_mean_qr q r) :
  r - p = 30 := by
  sorry

end value_of_r_minus_p_l443_443115


namespace max_f_value_find_period_max_area_of_triangle_l443_443360

noncomputable def f (x : ℝ) : ℝ :=
  cos (2 * x + π / 3) + 2 * cos x ^ 2

theorem max_f_value :
  ∃ x : ℝ, f x = 1 + sqrt 3 ∧ (∀ y : ℝ, f y ≤ 1 + sqrt 3) := sorry

theorem find_period :
  ∃ T : ℝ, T = π ∧ ∀ x : ℝ, f (x + T) = f x := sorry

theorem max_area_of_triangle (A B C a b c S : ℝ) 
  (h1 : f (C / 2) = 1) 
  (h2 : c = 2) 
  (h3 : C = π / 3) :
  ∃ a b : ℝ, a = 2 ∧ b = 2 ∧ S = sqrt 3 ∧ 
  (area : ℝ) (h : a * b * sin C / 2 = S) := sorry

end max_f_value_find_period_max_area_of_triangle_l443_443360


namespace palindromic_prime_sum_l443_443227

def is_prime (n : ℕ) : Prop :=
  nat.prime n

def reverse_digits (n : ℕ) : ℕ := 
  -- Reverses the digits of a number
  n.digits 10.reverse.foldl (λ acc d, acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  n < 100 ∧ is_prime n ∧ is_prime (reverse_digits n)

noncomputable def sum_of_palindromic_primes : ℕ :=
  (finset.range 100).filter is_palindromic_prime |>.sum id

theorem palindromic_prime_sum : sum_of_palindromic_primes = 429 :=
sorry

end palindromic_prime_sum_l443_443227


namespace stock_price_end_of_second_year_l443_443195

theorem stock_price_end_of_second_year 
(initial_price : ℝ)
(increase_rate_year1 : ℝ)
(decrease_rate_year2 : ℝ)
(h_initial : initial_price = 120)
(h_increase : increase_rate_year1 = 0.50)
(h_decrease : decrease_rate_year2 = 0.30) : 
let price_end_year1 := initial_price + (increase_rate_year1 * initial_price)
let price_end_year2 := price_end_year1 - (decrease_rate_year2 * price_end_year1)
in price_end_year2 = 126 :=
by
  have h_price_end_year1 : price_end_year1 = 180 := by sorry
  have h_decrease_amount : decrease_rate_year2 * price_end_year1 = 54 := by sorry
  have h_price_end_year2 : price_end_year2 = 126 := by sorry
  exact h_price_end_year2

end stock_price_end_of_second_year_l443_443195


namespace power_of_product_l443_443822

variable (a b : ℝ) (m : ℕ)
theorem power_of_product (h : 0 < m) : (a * b)^m = a^m * b^m :=
sorry

end power_of_product_l443_443822


namespace milton_zoology_books_l443_443782

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l443_443782


namespace round_2_0249_to_nearest_hundredth_l443_443798

theorem round_2_0249_to_nearest_hundredth : round (2.0249 : ℝ) 2 = 2.02 := 
sorry

end round_2_0249_to_nearest_hundredth_l443_443798


namespace set_intersection_l443_443362

def A := {x : ℝ | x^2 - 3*x ≥ 0}
def B := {x : ℝ | x < 1}
def intersection := {x : ℝ | x ≤ 0}

theorem set_intersection : A ∩ B = intersection :=
  sorry

end set_intersection_l443_443362


namespace median_of_free_throws_l443_443506

def list_of_free_throws : List ℕ := [5, 17, 16, 14, 12, 10, 20, 18, 15, 11]

theorem median_of_free_throws 
  (free_throws : List ℕ) 
  (h : free_throws = list_of_free_throws) : 
  median free_throws = 14.5 :=
by 
  sorry

end median_of_free_throws_l443_443506


namespace different_positive_integers_as_differences_l443_443290

structure Condition where
  (S : Finset ℕ)
  (distinct_members : ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b)

noncomputable def number_of_positive_differences (c : Condition) : ℕ :=
  let differences := {d | ∃ a b : ℕ, a ∈ c.S ∧ b ∈ c.S ∧ a ≠ b ∧ d = abs (a - b)}.toFinset
  differences.card

theorem different_positive_integers_as_differences : 
  ∀ c : Condition, 
    c.S = {1, 3, 5, 7, 9, 11} →
    number_of_positive_differences c = 5 := 
by
  sorry

end different_positive_integers_as_differences_l443_443290


namespace angle_E_is_120_l443_443155

noncomputable def equal_length_pentagon (ABCDE : Type) [plane_geometry ABCDE] : Prop :=
  side_length A B = side_length B C ∧
  side_length B C = side_length C D ∧
  side_length C D = side_length D E ∧
  side_length D E = side_length E A

noncomputable def angle_A (ABCDE : Type) [plane_geometry ABCDE] : Prop :=
  measure_angle A = 90

noncomputable def angle_B (ABCDE : Type) [plane_geometry ABCDE] : Prop :=
  measure_angle B = 90

noncomputable def angle_C (ABCDE : Type) [plane_geometry ABCDE] : Prop :=
  measure_angle C = 120

theorem angle_E_is_120
  (ABCDE : Type) [plane_geometry ABCDE]
  (h1 : equal_length_pentagon ABCDE)
  (h2 : angle_A ABCDE)
  (h3 : angle_B ABCDE)
  (h4 : angle_C ABCDE) :
  measure_angle E = 120 := by
  sorry -- Placeholder for the proof

end angle_E_is_120_l443_443155


namespace velma_daphne_visibility_difference_l443_443090

-- Define the visibility distances for Veronica, Freddie, Velma, and Daphne
def veronica_visibility_distance : ℝ := 1000
def freddie_visibility_distance : ℝ := 3 * veronica_visibility_distance
def velma_visibility_distance : ℝ := 5 * freddie_visibility_distance - 2000
def daphne_visibility_distance : ℝ :=
  (veronica_visibility_distance + freddie_visibility_distance + velma_visibility_distance) / 3

-- Define the target sum of visibility distances
def target_sum_visibility_distance : ℝ := 40000

-- Calculate the sum of all their flashlights' visibility distances
def sum_visibility_distances : ℝ :=
  veronica_visibility_distance + freddie_visibility_distance + velma_visibility_distance + daphne_visibility_distance

-- Prove that Velma's visibility distance is 7666.67 feet more than Daphne's visibility distance
theorem velma_daphne_visibility_difference :
  velma_visibility_distance - daphne_visibility_distance = 7666.67 ∧ sum_visibility_distances = target_sum_visibility_distance :=
by
  sorry

end velma_daphne_visibility_difference_l443_443090


namespace distinct_integers_from_special_fractions_l443_443176

def is_special_fraction (a b : ℕ) : Prop := a > 0 ∧ b > 0 ∧ a + b = 18

def is_integer (x : ℚ) : Prop := ∃ n : ℤ, x = n

theorem distinct_integers_from_special_fractions : 
  (∃ (sums : Finset ℤ), 
      (∀ (a b c d : ℕ), is_special_fraction a b → is_special_fraction c d → 
        is_integer ((a : ℚ) / b + (c : ℚ) / d) ∧ ((a : ℚ) / b + (c : ℚ) / d : ℚ) ∈ sums)
      ∧ sums.card = 13) :=
begin
  sorry
end

end distinct_integers_from_special_fractions_l443_443176


namespace greatest_possible_gcd_l443_443358

theorem greatest_possible_gcd (d : ℕ) (a : ℕ → ℕ) (h_sum : (a 0) + (a 1) + (a 2) + (a 3) + (a 4) + (a 5) + (a 6) + (a 7) = 595)
  (h_gcd : ∀ i, d ∣ a i) : d ≤ 35 :=
sorry

end greatest_possible_gcd_l443_443358


namespace borrowed_sheets_l443_443679

-- Defining the page sum function
def sum_pages (n : ℕ) : ℕ := n * (n + 1)

-- Formulating the main theorem statement
theorem borrowed_sheets (b c : ℕ) (H : c + b ≤ 30) (H_avg : (sum_pages b + sum_pages (30 - b - c) - sum_pages (b + c)) * 2 = 25 * (60 - 2 * c)) :
  c = 10 :=
sorry

end borrowed_sheets_l443_443679


namespace geometric_and_harmonic_mean_l443_443083

noncomputable def find_numbers : ℝ × ℝ :=
(list.to_finset
[ (5 + Real.sqrt 5) / 2,
  (5 - Real.sqrt 5) / 2]).prod !

theorem geometric_and_harmonic_mean
  (a b : ℝ)
  (h1: a * b = 5)
  (h2: 2 / (1/a + 1/b) = 2) :
  (a = (5 + Real.sqrt 5) / 2 ∧ b = (5 - Real.sqrt 5) / 2) ∨
  (a = (5 - Real.sqrt 5) / 2 ∧ b = (5 + Real.sqrt 5) / 2) :=
by {
  sorry
}

end geometric_and_harmonic_mean_l443_443083


namespace find_x0_l443_443268

def f (x : ℝ) : ℝ := if x >= 0 then 2^x else -x

theorem find_x0 (x0 : ℝ) (hx : f x0 = 2) : x0 = 1 ∨ x0 = -2 :=
sorry

end find_x0_l443_443268


namespace differing_remainders_l443_443928

-- Define the set of all possible 100-digit numbers where each digit is either 1 or 2
def possibleNumbers : Finset ℕ :=
  Finset.univ.filter (λ n, n < 3 ^ 100 ∧ ∀ k, k < 100 → ((λ d, 0 < d ∧ d < 3) (n.divMod 10).fst.applyDigit k))

-- Define the main theorem
theorem differing_remainders : Finset.card (possibleNumbers.image (λ n, n % 1024)) = 1024 := by
  sorry

end differing_remainders_l443_443928


namespace maximum_value_is_17_l443_443027

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443027


namespace max_value_of_fraction_l443_443019

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443019


namespace volume_of_pyramid_l443_443141

theorem volume_of_pyramid
  (hexagon : Point → Prop)
  (regular_hexagon : ∀ A B C D E F : Point, hexagon A ∧ hexagon B ∧ hexagon C ∧ hexagon D ∧ hexagon E ∧ hexagon F → 
                    hexagon A ∧ hexagon B ∧ hexagon C ∧ hexagon D ∧ hexagon E ∧ hexagon F ∧
                    (∀ X Y, (hexagon X ∧ hexagon Y) → dist X Y = 6 ∨ dist X Y = 6 * sqrt(3))) 
  (equilateral_triangle_PAD : ∀ P A D : Point, 
                              (hexagon A ∧ (dist P A = 10) ∧ (dist P D = 10) ∧ (dist A D = 10)))
  (side_length_hexagon : ∀ A B : Point, hexagon A ∧ hexagon B → dist A B = 6)
  : ∀ P A B C D E F : Point, hexagon A ∧ hexagon B ∧ hexagon C ∧ hexagon D ∧ hexagon E ∧ hexagon F ∧
    (dist P A = 10) ∧ (dist P D = 10) ∧ (dist A D = 10) →
    volume (pyramid P A B C D E F) = 270 :=
by
  sorry

def Point := ℝ × ℝ

def dist (p1 p2 : Point) : ℝ := 
  let (x1, y1) := p1
  let (x2, y2) := p2
  sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2) 

structure Pyramid :=
  (vertices : List Point)
  (apex : Point)

def volume : Pyramid → ℝ
| ⟨vertices, apex⟩ =>
  sorry 

end volume_of_pyramid_l443_443141


namespace speed_of_person_b_l443_443790

noncomputable def PersonASpeed := 40 -- Person A's initial speed in km/h
noncomputable def TravelTimeAtoB := 2 -- Time taken by Person A from point A to point B in hours
noncomputable def DistanceAtoB := 80 -- Distance from point A to point B in kilometers
noncomputable def NewSpeedA := 80 -- New speed of Person A from point B to point C
noncomputable def TravelTimeAtoC := 2 -- Time taken by Person A from point B to point C in hours
noncomputable def DistanceBtoC := 160 -- Distance from point B to point C in kilometers
noncomputable def ExtraTimeB := 0.5 -- Additional time that Person B had been traveling before Person A reached B
noncomputable def TotalTimeB := 2.5 -- Total time for Person B to travel from B to C

theorem speed_of_person_b : (DistanceBtoC / TotalTimeB) = 64 := by
  -- Definitions based on conditions
  have PersonBSpeed := DistanceBtoC / TotalTimeB
  exact (eq.refl 64)
  sorry

end speed_of_person_b_l443_443790


namespace smallest_m_l443_443541

theorem smallest_m (m : ℕ) (x : ℕ) (h1 : 1.05 * x = 100 * m) : m = 21 :=
sorry

end smallest_m_l443_443541


namespace monthly_rate_is_24_l443_443528

noncomputable def weekly_rate : ℝ := 10
noncomputable def weeks_per_year : ℕ := 52
noncomputable def months_per_year : ℕ := 12
noncomputable def yearly_savings : ℝ := 232

theorem monthly_rate_is_24 (M : ℝ) (h : weeks_per_year * weekly_rate - months_per_year * M = yearly_savings) : 
  M = 24 :=
by
  sorry

end monthly_rate_is_24_l443_443528


namespace find_real_numbers_with_means_l443_443084

theorem find_real_numbers_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_geom : Real.sqrt (a * b) = Real.sqrt 5) 
  (h_harm : 2 / ((1 / a) + (1 / b)) = 2) : 
  ( {a, b} = { (5 + Real.sqrt 5) / 2, (5 - Real.sqrt 5) / 2 } ) :=
by
  sorry

end find_real_numbers_with_means_l443_443084


namespace geometry_problem_l443_443926

-- Defining the geometry and properties
variables (y x : ℝ)

-- Conditions of the problem
def length_of_rectangle := 3 * y
def width_of_rectangle := y

def length_of_smaller_rectangle := x
def width_of_smaller_rectangle := y - x

def perimeter_of_triangle := x + (3 * y - x) + real.sqrt (2 * x^2 - 6 * y * x + 9 * y^2)
def area_of_smaller_rectangle := x * (y - x)

-- The statement of the theorem (problem statement reformulated in Lean)
theorem geometry_problem 
  (h_length : length_of_rectangle y = 3 * y)
  (h_width : width_of_rectangle y = y)
  (h_small_length : length_of_smaller_rectangle x = x)
  (h_small_width : width_of_smaller_rectangle x y = y - x) :
  perimeter_of_triangle x y = 3 * y + real.sqrt (2 * x^2 - 6 * y * x + 9 * y^2) ∧ 
  area_of_smaller_rectangle x y = x * y - x^2 :=
sorry

end geometry_problem_l443_443926


namespace valid_number_of_orderings_l443_443786

noncomputable def total_orderings (H : Type) [Fintype H] (l g a m : H) (houses : List H) :=
  -- count valid orderings where x precedes y
  (houses.permutations.count (λ order, 
    order.indexOf l < order.indexOf g ∧ 
    order.indexOf a < order.indexOf m))

theorem valid_number_of_orderings {H : Type} [DecidableEq H] [Fintype H] (l g a m : H)
  (houses : List H) (h_distinct : houses.nodup):
l ∈ houses → g ∈ houses → a ∈ houses → m ∈ houses → houses.length = 5 →
 total_orderings houses l g a m houses = 24 :=
  by 
  sorry

end valid_number_of_orderings_l443_443786


namespace platform_length_proof_l443_443489

variables (length_train time_pole time_platform : ℝ)
variable (L : ℝ) -- Length of the platform

-- Given conditions
def train_speed (length_train time_pole : ℝ) : ℝ := length_train / time_pole
def platform_length (length_train time_platform : ℝ) (V : ℝ) : ℝ := V * time_platform - length_train

-- Proof statement
theorem platform_length_proof (h1 : length_train = 300) 
                             (h2 : time_pole = 36)
                             (h3 : time_platform = 39)
                             (V : ℝ) 
                             (hV : V = train_speed length_train time_pole) :
                             platform_length length_train time_platform V = 25 :=
by 
  sorry

end platform_length_proof_l443_443489


namespace area_is_correct_l443_443753

noncomputable def f (x : ℝ) : ℝ :=
  if h : 0 ≤ x ∧ x ≤ 3 then x^2
  else if h : 3 ≤ x ∧ x ≤ 10 then 3 * x - 6
  else 0

def area_under_curve : ℝ :=
  (∫ x in 0..3, x^2) + (∫ x in 3..10, 3 * x - 6)

theorem area_is_correct : area_under_curve = 94.5 :=
  sorry

end area_is_correct_l443_443753


namespace probability_queen_then_club_l443_443080

-- Define the problem conditions using the definitions
def deck_size : ℕ := 52
def num_queens : ℕ := 4
def num_clubs : ℕ := 13
def num_club_queens : ℕ := 1

-- Define a function that computes the probability of the given event
def probability_first_queen_second_club : ℚ :=
  let prob_first_club_queen := (num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_club_queen := (num_clubs - 1 : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_1 := prob_first_club_queen * prob_second_club_given_first_club_queen
  let prob_first_non_club_queen := (num_queens - num_club_queens : ℚ) / (deck_size : ℚ)
  let prob_second_club_given_first_non_club_queen := (num_clubs : ℚ) / (deck_size - 1 : ℚ)
  let prob_case_2 := prob_first_non_club_queen * prob_second_club_given_first_non_club_queen
  prob_case_1 + prob_case_2

-- The statement to be proved
theorem probability_queen_then_club : probability_first_queen_second_club = 1 / 52 := by
  sorry

end probability_queen_then_club_l443_443080


namespace find_angle_Y_l443_443236

-- Defining angles as Real numbers within degrees
def angle (x : ℝ) := x

-- Conditions given in the problem
variables (A B X Y Z : ℝ)
variable h1 : A + B = 180
variable h2 : A = 50
variable h3 : X = Y
variable h4 : B + Z = 180
variable h5 : Z = 50

-- The goal is to prove that angle Y = 25 degrees
theorem find_angle_Y : Y = 25 :=
by
  sorry

end find_angle_Y_l443_443236


namespace parabola_x_intercept_y_intercept_point_l443_443731

theorem parabola_x_intercept_y_intercept_point (a b w : ℝ) 
  (h1 : a = -1) 
  (h2 : b = 4) 
  (h3 : ∀ x : ℝ, x = 0 → w = 8): 
  ∃ (w : ℝ), w = 8 := 
by
  sorry

end parabola_x_intercept_y_intercept_point_l443_443731


namespace unique_side_length_of_square_l443_443549

theorem unique_side_length_of_square {
  AF DH BG AE : ℝ,
  quadrilateral_area : ℝ,
  side_length : ℝ
}
(AF_val : AF = 7)
(DH_val : DH = 4)
(BG_val : BG = 5)
(AE_val : AE = 1)
(quadrilateral_area_val : quadrilateral_area = 78)
: side_length = 12 :=
by
  sorry

end unique_side_length_of_square_l443_443549


namespace no_perfect_number_of_form_p3q_l443_443391
open Real Nat

theorem no_perfect_number_of_form_p3q (p q : ℕ) (hp : Prime p) (hq : Prime q) (h_distinct: p ≠ q) :
  ¬∃ n : ℕ, n = p^3 * q ∧ isPerfect n :=
sorry

end no_perfect_number_of_form_p3q_l443_443391


namespace carbonate_weight_l443_443618

namespace MolecularWeight

def molecular_weight_Al2_CO3_3 : ℝ := 234
def molecular_weight_Al : ℝ := 26.98
def num_Al_atoms : ℕ := 2

theorem carbonate_weight :
  molecular_weight_Al2_CO3_3 - (num_Al_atoms * molecular_weight_Al) = 180.04 :=
sorry

end MolecularWeight

end carbonate_weight_l443_443618


namespace tangent_line_at_origin_l443_443266

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 2

noncomputable def f_prime (x : ℝ) : ℝ := Real.exp x + x * Real.exp x

theorem tangent_line_at_origin :
  let tangent_point := (0 : ℝ, f 0)
  f_prime 0 = 1 ∧ tangent_point = (0, 2) ∧
  ∀ x y : ℝ, y = f′ 0 * (x - 0) + f 0 ↔ x - y + 2 = 0 :=
by
  have tangent_point : ℝ × ℝ := (0, f 0)
  have slope := f_prime 0
  have is_tangent := ∀ x y, y = slope * (x - 0) + tangent_point.2
  have line_eq := ∀ x y, x - y + 2 = 0
  sorry

end tangent_line_at_origin_l443_443266


namespace revenue_december_times_average_l443_443700

variable {D : ℝ} -- the revenue in December

-- Given conditions in the problem
def revenue_in_november (D : ℝ) := (3/5 : ℝ) * D
def revenue_in_january (D : ℝ) := (1/3 : ℝ) * revenue_in_november D

-- Define the average of revenues in November and January
def average_revenue_nov_jan (D : ℝ) := ((revenue_in_november D) + (revenue_in_january D)) / 2

-- Prove that December's revenue is 2.5 times the average of revenues in November and January
theorem revenue_december_times_average (D : ℝ) :
  D = 2.5 * average_revenue_nov_jan D :=
by
  sorry -- Proof to be added

end revenue_december_times_average_l443_443700


namespace cube_net_sums_l443_443165

theorem cube_net_sums (A B C : ℕ) (x y z : ℕ) (h₁ : A + x = B + y) (h₂ : B + y = C + z) (h₃ : A + x = C + z)
  (h₄ : {A, B, C} = {4, 5, 6})
  (h₅ : A ≠ B) (h₆ : B ≠ C) (h₇ : A ≠ C) :
  (A = 5) ∧ (B = 4) ∧ (C = 6) ∨ (A = 4) ∧ (B = 6) ∧ (C = 5) :=
by
  sorry

end cube_net_sums_l443_443165


namespace LCM_of_fractions_LCM_of_fractions_test_LCM_of_inverses_l443_443469

theorem LCM_of_fractions (x : ℕ) (h5 : x ≠ 0) (h10 : 10 * x ≠ 0) (h15 : 15 * x ≠ 0) :
  Nat.lcm (5 * x) (Nat.lcm (10 * x) (15 * x)) = 30 * x :=
by {
  have h1 : Nat.lcm (10 * x) (15 * x) = 30 * x := sorry,
  rw h1,
  apply Nat.lcm_comm,
}

theorem LCM_of_fractions_test (x : ℕ) (h : x ≠ 0) :
  Nat.lcm (LCM_of_fractions x h (by simp [h]) (by simp [h])) = 30 * x := sorry

theorem LCM_of_inverses (x : ℕ) (hx : x ≠ 0) :
  (1 : ℚ) / (5 * x) = 1 / 30 * x := LCM_of_fractions_test x hx

end LCM_of_fractions_LCM_of_fractions_test_LCM_of_inverses_l443_443469


namespace expand_and_simplify_l443_443609

theorem expand_and_simplify (x : ℝ) : (2*x + 6)*(x + 9) = 2*x^2 + 24*x + 54 :=
by
  sorry

end expand_and_simplify_l443_443609


namespace scientific_notation_of_3100000_l443_443149

theorem scientific_notation_of_3100000 :
  ∃ (a : ℝ) (n : ℤ), 3100000 = a * 10^n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 3.1 ∧ n = 6 :=
  sorry

end scientific_notation_of_3100000_l443_443149


namespace room_width_l443_443054

-- We state our conditions and to be proven theorem
theorem room_width
    (length : ℝ)
    (cost_per_sqm : ℝ)
    (total_cost : ℝ)
    (total_area : length * cost_per_sqm / total_cost = length * 1)
    (area_eq : total_cost / cost_per_sqm = 20.625)
    (length_eq : length = 5.5)
    : total_area / length = 3.75 := 
by
    sorry

end room_width_l443_443054


namespace part1_part2_l443_443270

variables {α : Type*} [linear_ordered_field α]

-- Define the function f(x) = |x - 1| - 2|x + a|
def f (a x : α) : α := abs (x - 1) - 2 * abs (x + a)

-- Define the function g(x) = 1/2 * x + b
def g (b x : α) : α := (1 / 2) * x + b

-- Part 1: Prove the solution set of the inequality f(x) ≤ 0 given a = 1/2.
theorem part1 (x : α) :
  f (1 / 2 : α) x ≤ 0 ↔ x ≤ -2 ∨ (0 ≤ x ∧ x ≤ 1) := sorry

-- Part 2: Prove that 2b - 3a > 2 given a ≥ -1 
-- and the function g(x) is always above the function f(x).
theorem part2 (a b : α) (h : a ≥ -1)
  (h_above : ∀ x, g b x ≥ f a x) :
  2 * b - 3 * a > 2 := sorry

end part1_part2_l443_443270


namespace expected_digits_of_fair_icosahedral_die_l443_443406

noncomputable def expected_num_of_digits : ℚ :=
  (9 / 20) * 1 + (11 / 20) * 2

theorem expected_digits_of_fair_icosahedral_die :
  expected_num_of_digits = 1.55 := by
  sorry

end expected_digits_of_fair_icosahedral_die_l443_443406


namespace solution_set_of_inequality_l443_443816

variable (f : ℝ → ℝ)

axiom domain_f : ∀ x : ℝ, True
axiom value_at_neg1 : f (-1) = 2
axiom derivative_pos : ∀ x : ℝ, deriv f x > 2

theorem solution_set_of_inequality : 
  { x : ℝ | f x > 2 * x + 4 } = Ioi (-1) := by
  sorry

end solution_set_of_inequality_l443_443816


namespace antonio_weight_l443_443910

-- Let A be the weight of Antonio
variable (A : ℕ)

-- Conditions:
-- 1. Antonio's sister weighs A - 12 kilograms.
-- 2. The total weight of Antonio and his sister is 88 kilograms.

theorem antonio_weight (A: ℕ) (h1: A - 12 >= 0) (h2: A + (A - 12) = 88) : A = 50 := by
  sorry

end antonio_weight_l443_443910


namespace lizette_third_quiz_score_l443_443365

theorem lizette_third_quiz_score :
  ∀ (x : ℕ),
  (2 * 95 + x) / 3 = 94 → x = 92 :=
by
  intro x h
  have h1 : 2 * 95 = 190 := by norm_num
  have h2 : 3 * 94 = 282 := by norm_num
  sorry

end lizette_third_quiz_score_l443_443365


namespace find_k_inv_h_8_l443_443808

variable (h k : ℝ → ℝ)

-- Conditions
axiom h_inv_k_x (x : ℝ) : h⁻¹ (k x) = 3 * x - 4
axiom h_3x_minus_4 (x : ℝ) : k x = h (3 * x - 4)

-- The statement we want to prove
theorem find_k_inv_h_8 : k⁻¹ (h 8) = 8 := 
  sorry

end find_k_inv_h_8_l443_443808


namespace ming_estimate_less_l443_443846

theorem ming_estimate_less (x y δ : ℝ) (h1 : x > y) (h2 : y > 0) (h3 : δ > 0) : 
  (x + δ) - (y + 2 * δ) < x - y :=
by 
  sorry

end ming_estimate_less_l443_443846


namespace four_colorable_l443_443119

variables (V : Type) [Fintype V] (G : SimpleGraph V) [DecidableRel G.Adj]

-- Definitions based on the conditions:
def is_connected (G : SimpleGraph V) : Prop :=
  ∀ u v : V, G.connected (u, v)

def odd_cycle_disconnection (G : SimpleGraph V) : Prop :=
  ∀ (C : List V), G.is_cycle C → C.length % 2 = 1 → G.edge_set ∩ (set_of (λ e, e ∈ C.to_set)) = ∅ → ¬(∀ u v : V, G.connected (u, v))

-- Theorem statement:
theorem four_colorable (G : SimpleGraph V)
  [is_connected G]
  [odd_cycle_disconnection G] :
  ∃ f : V → Fin 4, G.proper_coloring f :=
sorry

end four_colorable_l443_443119


namespace razorback_tshirt_profit_l443_443810

theorem razorback_tshirt_profit :
  let profit_per_tshirt := 9
  let cost_per_tshirt := 4
  let num_tshirts_sold := 245
  let discount := 0.2
  let selling_price := profit_per_tshirt + cost_per_tshirt
  let discount_amount := discount * selling_price
  let discounted_price := selling_price - discount_amount
  let total_revenue := discounted_price * num_tshirts_sold
  let total_production_cost := cost_per_tshirt * num_tshirts_sold
  let total_profit := total_revenue - total_production_cost
  total_profit = 1568 :=
by
  sorry

end razorback_tshirt_profit_l443_443810


namespace equilateral_and_same_centroid_l443_443148

universe u

noncomputable theory

variables {A B C : EuclideanGeometry.Point} 
variables {S1 S2 S3 Sa Sb Sc : EuclideanGeometry.Point}

-- Define that S1, S2, S3 are centroids of the externally constructed equilateral triangles on the sides of ABC
def are_external_centroids (A B C S1 S2 S3 : EuclideanGeometry.Point) : Prop :=
  ∃ (D₁ D₂ D₃ : EuclideanGeometry.Point),
    EuclideanGeometry.EquilateralTriangle A B D₁ ∧
    EuclideanGeometry.Centroid A B D₁ S1 ∧
    EuclideanGeometry.EquilateralTriangle B C D₂ ∧
    EuclideanGeometry.Centroid B C D₂ S2 ∧
    EuclideanGeometry.EquilateralTriangle C A D₃ ∧
    EuclideanGeometry.Centroid C A D₃ S3

-- Define that Sa, Sb, Sc are centroids of the internally constructed equilateral triangles on the sides of ABC
def are_internal_centroids (A B C Sa Sb Sc : EuclideanGeometry.Point) : Prop :=
  ∃ (D₁ D₂ D₃ : EuclideanGeometry.Point),
    EuclideanGeometry.EquilateralTriangle A B D₁ ∧
    EuclideanGeometry.Centroid A B D₁ Sa ∧
    EuclideanGeometry.EquilateralTriangle B C D₂ ∧
    EuclideanGeometry.Centroid B C D₂ Sb ∧
    EuclideanGeometry.EquilateralTriangle C A D₃ ∧
    EuclideanGeometry.Centroid C A D₃ Sc

-- The main theorem statement
theorem equilateral_and_same_centroid
  (h_ext : are_external_centroids A B C S1 S2 S3)
  (h_int : are_internal_centroids A B C Sa Sb Sc) :
  EuclideanGeometry.EquilateralTriangle S1 S2 S3 ∧
  EuclideanGeometry.EquilateralTriangle Sa Sb Sc ∧
  EuclideanGeometry.Centroid S1 S2 S3 (EuclideanGeometry.Centroid A B C) ∧
  EuclideanGeometry.Centroid Sa Sb Sc (EuclideanGeometry.Centroid A B C) :=
sorry

end equilateral_and_same_centroid_l443_443148


namespace max_value_of_fraction_l443_443004

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443004


namespace bianca_songs_l443_443499

theorem bianca_songs (initial_songs deleted_songs new_songs : ℕ) 
  (h1 : initial_songs = 34) 
  (h2 : deleted_songs = 14) 
  (h3 : new_songs = 44) : 
  (initial_songs - deleted_songs + new_songs = 64) :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end bianca_songs_l443_443499


namespace brianne_savings_in_may_l443_443306

-- Definitions based on conditions from a)
def initial_savings_jan : ℕ := 20
def multiplier : ℕ := 3
def additional_income : ℕ := 50

-- Savings in successive months
def savings_feb : ℕ := multiplier * initial_savings_jan
def savings_mar : ℕ := multiplier * savings_feb + additional_income
def savings_apr : ℕ := multiplier * savings_mar + additional_income
def savings_may : ℕ := multiplier * savings_apr + additional_income

-- The main theorem to verify
theorem brianne_savings_in_may : savings_may = 2270 :=
sorry

end brianne_savings_in_may_l443_443306


namespace line_equation_parametric_to_implicit_l443_443814

theorem line_equation_parametric_to_implicit (t : ℝ) :
  ∀ x y : ℝ, (x = 3 * t + 6 ∧ y = 5 * t - 7) → y = (5 / 3) * x - 17 :=
by
  intros x y h
  obtain ⟨hx, hy⟩ := h
  sorry

end line_equation_parametric_to_implicit_l443_443814


namespace subgroup_finite_index_isomorphic_l443_443749

open AddCommGroup

-- Let \( A \) be an abelian additive group
variable (A: Type*) [AddCommGroup A]

-- All nonzero elements of \( A \) have infinite order.
variable (h_order : ∀ a : A, a ≠ 0 → ∀ n : ℕ, n ≠ 0 → (n • a) ≠ 0)

-- For each prime number \( p \), |A/pA| ≤ p.
variable (h_quotient : ∀ p : ℕ, Prime p → (AddSubgroup.zmultiples ((p : ℕ) • (1 : A))).index ≤ p)

-- Prove each subgroup of \( A \) of finite index is isomorphic to \( A \).
theorem subgroup_finite_index_isomorphic : 
  ∀ (B : AddSubgroup A), B.index < ∞ → B ≃+ A :=
sorry

end subgroup_finite_index_isomorphic_l443_443749


namespace part1_part2_l443_443273

noncomputable def f (x a : ℝ) : ℝ := |x - 1| - 2 * |x + a|
noncomputable def g (x b : ℝ) : ℝ := 0.5 * x + b

theorem part1 (a : ℝ) (h : a = 1/2) : 
  { x : ℝ | f x a ≤ 0 } = { x : ℝ | x ≤ -2 ∨ x ≥ 0 } :=
sorry

theorem part2 (a b : ℝ) (h1 : a ≥ -1) (h2 : ∀ x, g x b ≥ f x a) : 
  2 * b - 3 * a > 2 :=
sorry

end part1_part2_l443_443273


namespace a_2023_value_l443_443927

theorem a_2023_value :
  ∀ (a : ℕ → ℚ),
  a 1 = 5 ∧
  a 2 = 5 / 11 ∧
  (∀ n, 3 ≤ n → a n = (a (n - 2)) * (a (n - 1)) / (3 * (a (n - 2)) - (a (n - 1)))) →
  a 2023 = 5 / 10114 ∧ 5 + 10114 = 10119 :=
by
  sorry

end a_2023_value_l443_443927


namespace find_t_l443_443675

-- Define vectors a and b
def a := (3 : ℝ, 4 : ℝ)
def b := (1 : ℝ, 0 : ℝ)

-- Define the vector c as a function of t
def c (t : ℝ) := (a.1 + t * b.1, a.2 + t * b.2)

-- Statement of the theorem to be proven
theorem find_t (t : ℝ) :
  (a.1 * (a.1 + t * b.1) + a.2 * (a.2 + t * b.2)) = (b.1 * (a.1 + t * b.1) + b.2 * (a.2 + t * b.2)) →
  t = 5 :=
by
  sorry

end find_t_l443_443675


namespace coefficient_of_x4_in_binomial_expansion_l443_443191

theorem coefficient_of_x4_in_binomial_expansion :
  let T_r := λ r : ℕ, choose 5 r * (x^2)^(5 - r) * (-1 / x)^r in
  (∃ r : ℕ, 10 - 3*r = 4 ∧ T_r r = 10 * x^4) :=
by
  let T_r := λ r : ℕ, choose 5 r * (x^2)^(5 - r) * (-1 / x)^r
  existsi 2
  split
  sorry

end coefficient_of_x4_in_binomial_expansion_l443_443191


namespace differences_in_set_l443_443295

theorem differences_in_set : 
  let s := {1, 3, 5, 7, 9, 11}
  in (#{d | ∃ x y, x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ d = x - y ∧ d > 0}.card) = 5 := 
by
  sorry

end differences_in_set_l443_443295


namespace induction_inequality_l443_443087

variable (n : ℕ) (h₁ : n ∈ Set.Icc 2 (2^n - 1))

theorem induction_inequality : 1 + 1/2 + 1/3 < 2 := 
  sorry

end induction_inequality_l443_443087


namespace angle_bisectors_perpendicular_or_coincide_l443_443850

-- Definitions of the circles, points, and the secant
variables {circle1 circle2 : Type} [metric_space circle1] [metric_space circle2]
variables {T A B C D : circle1}
variables {secant : line}

-- Conditions from the problem
variables (hT : circle1.touch circle2 at T)
variables (hSecant1 : secant.intersects circle1 at A B)
variables (hSecant2 : secant.intersects circle2 at C D)

-- Main theorem statement
theorem angle_bisectors_perpendicular_or_coincide
  (hT : circle1.touch circle2 at T)
  (hSecant1 : secant.intersects circle1 at A B)
  (hSecant2 : secant.intersects circle2 at C D) :
  let α := ∠ ATB in 
  let β := ∠ CTD in
  angle_bisector α = angle_bisector β ∨ angle_bisector α ⊥ angle_bisector β :=
sorry

end angle_bisectors_perpendicular_or_coincide_l443_443850


namespace area_PQR_l443_443190

-- Define the coordinates of the points
def P : ℝ × ℝ := (-3, 4)
def Q : ℝ × ℝ := (4, 9)
def R : ℝ × ℝ := (5, -3)

-- Function to calculate the area of a triangle given three points
def area_of_triangle (A B C : ℝ × ℝ) : ℝ :=
  0.5 * (abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)))

-- Statement to prove the area of triangle PQR is 44.5
theorem area_PQR : area_of_triangle P Q R = 44.5 := sorry

end area_PQR_l443_443190


namespace distinct_positive_rationals_sum_squares_l443_443390

theorem distinct_positive_rationals_sum_squares (n : ℕ) (h : n > 0) : 
  ∃ (r : fin n → ℚ), 
  (∀ i j, i ≠ j → r i ≠ r j) ∧ 
  (∀ i, 0 < r i) ∧ 
  (∑ i, (r i)^2 = n) := 
sorry

end distinct_positive_rationals_sum_squares_l443_443390


namespace find_angle4_l443_443874

noncomputable def angle_1 := 70
noncomputable def angle_2 := 110
noncomputable def angle_3 := 35
noncomputable def angle_4 := 35

theorem find_angle4 (h1 : angle_1 + angle_2 = 180) (h2 : angle_3 = angle_4) :
  angle_4 = 35 :=
by
  have h3: angle_1 + 70 + 40 = 180 := by sorry
  have h4: angle_2 + angle_3 + angle_4 = 180 := by sorry
  sorry

end find_angle4_l443_443874


namespace f_is_odd_varphi_is_even_u_is_neither_even_nor_odd_l443_443602

def is_even (Q : ℝ → ℝ) : Prop := ∀ x, Q (-x) = Q x
def is_odd (Q : ℝ → ℝ) : Prop := ∀ x, Q (-x) = -Q x

def f (x : ℝ) : ℝ := x^2 / sin (2 * x)
def varphi (x : ℝ) : ℝ := 4 - 2 * x^4 + sin x ^ 2
def u (x : ℝ) : ℝ := x^3 + 2 * x - 1
noncomputable def y (x : ℝ) : ℝ := (1 + a^kx) / (1 - a^kx)

theorem f_is_odd : is_odd f := 
by sorry

theorem varphi_is_even : is_even varphi :=
by sorry

theorem u_is_neither_even_nor_odd : ¬is_even u ∧ ¬is_odd u := 
by sorry

noncomputable theorem y_is_odd : is_odd y := 
by sorry

end f_is_odd_varphi_is_even_u_is_neither_even_nor_odd_l443_443602


namespace sum_adjacent_to_9_l443_443434

def divisors (n : ℕ) := { d : ℕ | d > 0 ∧ n % d = 0 }
def is_in_circle (lst : List ℕ) := ∀ i, gcd (list.nth_le lst i sorry) (list.nth_le lst ((i + 1) % lst.length) sorry) > 1

theorem sum_adjacent_to_9 :
  ∃ (lst : List ℕ),
    lst.perm (divisors 189).toList ∧
    is_in_circle lst ∧
    (∃ (i : ℕ), list.nth_le lst i sorry = 9 ∧ list.nth_le lst ((i + 1) % lst.length) sorry + list.nth_le lst ((i - 1 + lst.length) % lst.length) sorry = 30) :=
sorry

end sum_adjacent_to_9_l443_443434


namespace evaluate_f_prime_at_two_l443_443858

noncomputable def f (a b x : ℝ) : ℝ := a * Real.log x + b / x

def f' (a b x : ℝ) : ℝ := (a * x + 2) / x^2

theorem evaluate_f_prime_at_two (a b : ℝ)
  (h₁ : f a b 1 = -2)
  (h₂ : (f' a b 1) = 0) :
  (f' (-2) (-2) 2) = -1/2 :=
by
  sorry

end evaluate_f_prime_at_two_l443_443858


namespace equidistant_fixed_point_exists_l443_443081

theorem equidistant_fixed_point_exists {α : Type} [metric_space α] {x y : ι → α} (hx : ∀ t1 t2 : ι, dist (x t1) (x t2) = dist (y t1) (y t2)) :
  ∃ m : α, ∀ t : ι, dist m (x t) = dist m (y t) :=
by sorry

end equidistant_fixed_point_exists_l443_443081


namespace max_value_of_fraction_l443_443018

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443018


namespace probability_of_sum_greater_than_six_l443_443234

-- Define the set and the condition of choosing two different numbers
def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5}
def num_choices := 2

-- Define the total number of ways to select two different numbers
def total_ways : ℕ := set_of_numbers.card.choose num_choices

-- Define a function that checks if the sum of two numbers is greater than 6
def sum_greater_than_six (a b : ℕ) : Prop := a + b > 6

-- Define the number of favorable outcomes where the sum is greater than 6
def favorable_ways : ℕ := 
  ({(2, 5), (3, 4), (3, 5), (4, 5)}.filter (λ (x : ℕ × ℕ), sum_greater_than_six x.1 x.2)).card

-- Define the probability
def probability : ℚ := favorable_ways / total_ways

-- Proof statement
theorem probability_of_sum_greater_than_six :
  probability = 2 / 5 :=
sorry

end probability_of_sum_greater_than_six_l443_443234


namespace black_eq_white_area_iff_median_l443_443342

theorem black_eq_white_area_iff_median 
  (ABC : Triangle)
  (O : Point)
  (AO BO CO : Line)
  (hAO : is_inside (O : Point) (ABC : Triangle))
  (hdiv : divide_into_six_regions (AO BO CO) (ABC))
  (α β γ : Real)
  (hO : barycentric_coordinates O α β γ)
  (P Q R : Point)
  (hP : intersection (AO) (BC))
  (hQ : intersection (BO) (AC))
  (hR : intersection (CO) (AB)) :
  (area_black_regions ABC O AO BO CO P Q R = area_white_regions ABC O AO BO CO P Q R) ↔
  is_on_median (O) (ABC) :=
sorry

end black_eq_white_area_iff_median_l443_443342


namespace find_element_in_n2o_l443_443989

-- Definitions and conditions given in the problem
def atomic_mass_nitrogen : ℝ := 14.01
def atomic_mass_oxygen : ℝ := 16.00
def mass_percentage_nitrogen (molecular_mass : ℝ) : ℝ :=
  (2 * atomic_mass_nitrogen / molecular_mass) * 100

-- Total molecular mass of N2O
def molecular_mass_n2o : ℝ := (2 * atomic_mass_nitrogen) + atomic_mass_oxygen

-- Main theorem stating the proof problem
theorem find_element_in_n2o (H : mass_percentage_nitrogen molecular_mass_n2o ≈ 63.64) : true :=
by sorry

end find_element_in_n2o_l443_443989


namespace trig_identity_l443_443581

theorem trig_identity :
  (sin (real.pi * 47 / 180) - (sin (real.pi * 17 / 180)) * (cos (real.pi * 30 / 180))) / (cos (real.pi * 17 / 180)) = 1 / 2 :=
by
  sorry

end trig_identity_l443_443581


namespace percentage_increase_time_second_half_l443_443375

noncomputable def total_distance : ℝ := 640
noncomputable def first_half_distance : ℝ := total_distance / 2
noncomputable def first_half_speed : ℝ := 80
noncomputable def total_trip_speed : ℝ := 40

def time (distance speed : ℝ) : ℝ := distance / speed

theorem percentage_increase_time_second_half :
  let first_half_time := time first_half_distance first_half_speed in
  let total_trip_time := time total_distance total_trip_speed in
  let second_half_time := total_trip_time - first_half_time in
  let percentage_increase := ((second_half_time - first_half_time) / first_half_time) * 100 in
  percentage_increase = 200 := by
  sorry

end percentage_increase_time_second_half_l443_443375


namespace number_of_dress_designs_is_correct_l443_443135

-- Define the number of choices for colors, patterns, and fabric types as conditions
def num_colors : Nat := 4
def num_patterns : Nat := 5
def num_fabric_types : Nat := 2

-- Define the total number of dress designs
def total_dress_designs : Nat := num_colors * num_patterns * num_fabric_types

-- Prove that the total number of different dress designs is 40
theorem number_of_dress_designs_is_correct : total_dress_designs = 40 := by
  sorry

end number_of_dress_designs_is_correct_l443_443135


namespace num_parts_divided_by_planes_l443_443743

-- Define the conditions
def three_planes_intersect_one_point (planes : ℕ) : Prop :=
  ∀ (a b c : plane), a ≠ b → b ≠ c → a ≠ c → intersection_point a b c ∈ plane_point

def no_four_planes_common_point (planes : ℕ) : Prop :=
  ∀ (a b c d : plane), a ≠ b → b ≠ c → c ≠ d → a ≠ d → b ≠ d → a ≠ c → 
  intersection_point a b c ≠ intersection_point c d a

-- Define the main function
def parts_divided_by_planes (n : ℕ) [hp: three_planes_intersect_one_point n] [hq: no_four_planes_common_point n] : ℕ :=
  (n^3 + 5 * n + 6) / 6
  
-- Prove the main theorem
theorem num_parts_divided_by_planes (n : ℕ) [hp: three_planes_intersect_one_point n] [hq: no_four_planes_common_point n] : 
  parts_divided_by_planes n = (n^3 + 5 * n + 6) / 6 := 
  sorry

end num_parts_divided_by_planes_l443_443743


namespace find_k_l443_443046

-- Define the given ellipse and its focus
def ellipse (x y : ℝ) (k : ℝ) := x^2 + k * y^2 / 5 = 1
def focus := (0, 2) : ℝ × ℝ

-- Prove that k = 1 given the ellipse equation and the focus
theorem find_k
  (h: focus = (0, 2))
  (heq : ∀ x y : ℝ, ellipse x y k)
  : k = 1 := by
  sorry

end find_k_l443_443046


namespace subset_1_index_index_211_subset_l443_443248

-- Given definitions and conditions
def E := {a_1, a_2, a_3, a_4, a_5, a_6, a_7, a_8, a_9, a_10}

def subset_index (subset : Set) : Nat :=
  subset.fold (λ acc x => acc + 2^(x - 1)) 0

-- Prove the equivalencies
theorem subset_1_index :
  subset_index {a_1, a_3} = 5 := 
sorry

theorem index_211_subset :
  [a_1, a_2, a_5, a_7, a_8] ∈ E :=
sorry

end subset_1_index_index_211_subset_l443_443248


namespace product_of_repeating_decimal_l443_443573

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l443_443573


namespace find_m_value_l443_443825

noncomputable def circle_tangent_line (m : ℝ) : Prop :=
  let x := -1
  let y := -1
  let line := λ y : ℝ, m*y + 2
  let distance := |x + m*y - 2| / Real.sqrt (m^2 + 1)
  let radius := Real.sqrt 2
  distance = radius

theorem find_m_value (m : ℝ) :
  circle_tangent_line m ↔ m = 1 ∨ m = -7 :=
by
  sorry

end find_m_value_l443_443825


namespace find_t_l443_443673

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l443_443673


namespace jenny_problem_l443_443332

def round_to_nearest_ten (n : ℤ) : ℤ :=
  if n % 10 < 5 then n - (n % 10) else n + (10 - n % 10)

theorem jenny_problem : round_to_nearest_ten (58 + 29) = 90 := 
by
  sorry

end jenny_problem_l443_443332


namespace find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443958

theorem find_integer_divisible_by_24_and_cube_root_between_9_and_9_1 : 
  ∃ (n : ℕ), 
  (n % 24 = 0) ∧ 
  (9 < (n : ℚ) ^ (1 / 3 : ℚ)) ∧ 
  ((n : ℚ) ^ (1 / 3 : ℚ) < 9.1) ∧ 
  (n = 744) := by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443958


namespace problem1_problem2_l443_443758

-- Definition for f(x)
def f (x : ℝ) : ℝ := 4^x / (4^x + 2)

-- Problem 1: Prove that f(x) + f(1-x) = 1
theorem problem1 (x : ℝ) : f(x) + f(1 - x) = 1 :=
sorry

-- Problem 2: Prove that the sum f(1/2014) + f(2/2014) + ... + f(2013/2014) = 1006.5
theorem problem2 : 
  ∑ i in finset.range(2013), f((i+1) / 2014) = 1006.5 :=
sorry

end problem1_problem2_l443_443758


namespace cars_minus_trucks_l443_443604

theorem cars_minus_trucks (total : ℕ) (trucks : ℕ) (h_total : total = 69) (h_trucks : trucks = 21) :
  (total - trucks) - trucks = 27 :=
by
  sorry

end cars_minus_trucks_l443_443604


namespace sum_of_squares_of_polynomial_coefficients_l443_443857

def polynomial_sum_of_squares (p : Polynomial ℤ) : ℤ :=
(p.coeff 5) ^ 2 + (p.coeff 3) ^ 2 + (p.coeff 1) ^ 2 + (p.coeff 0) ^ 2

theorem sum_of_squares_of_polynomial_coefficients :
  polynomial_sum_of_squares (6 * (Polynomial.Coeff X 5 + 2 * (Polynomial.Coeff X 3) + 5 * (Polynomial.Coeff X 1) + Polynomial.Coeff X 0)) = 1116 :=
sorry

end sum_of_squares_of_polynomial_coefficients_l443_443857


namespace find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443956

theorem find_integer_divisible_by_24_and_cube_root_between_9_and_9_1 : 
  ∃ (n : ℕ), 
  (n % 24 = 0) ∧ 
  (9 < (n : ℚ) ^ (1 / 3 : ℚ)) ∧ 
  ((n : ℚ) ^ (1 / 3 : ℚ) < 9.1) ∧ 
  (n = 744) := by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443956


namespace full_batches_needed_l443_443168

def students : Nat := 150
def cookies_per_student : Nat := 3
def cookies_per_batch : Nat := 20
def attendance_rate : Rat := 0.70

theorem full_batches_needed : 
  let attendees := (students : Rat) * attendance_rate
  let total_cookies_needed := attendees * (cookies_per_student : Rat)
  let batches_needed := total_cookies_needed / (cookies_per_batch : Rat)
  batches_needed.ceil = 16 :=
by
  sorry

end full_batches_needed_l443_443168


namespace warehouse_workers_wage_per_hour_l443_443331

/-- Define the constants and conditions given in the problem -/
def num_workers := 4
def num_managers := 2
def manager_wage_per_hour := 20
def days_per_month := 25
def hours_per_day := 8
def total_cost := 22000
def tax_rate := 0.10

/-- Define the assertion -/
theorem warehouse_workers_wage_per_hour : 
  (∀ W : ℝ, W ≠ 0 →
   let manager_monthly_wage := num_managers * manager_wage_per_hour * days_per_month * hours_per_day in
   let worker_monthly_wage := num_workers * W * days_per_month * hours_per_day in
   let wages := manager_monthly_wage + worker_monthly_wage in
   let total_wages_with_taxes := wages + tax_rate * wages in
   total_wages_with_taxes = total_cost) →
  (∃ W : ℝ, W = 15) :=
by
  intro H
  use 15
  sorry

end warehouse_workers_wage_per_hour_l443_443331


namespace max_branch_diameter_l443_443845

theorem max_branch_diameter (d : ℝ) (w : ℝ) (angle : ℝ) (H: w = 1 ∧ angle = 90) :
  d ≤ 2 * Real.sqrt 2 + 2 := 
sorry

end max_branch_diameter_l443_443845


namespace find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443960

theorem find_integer_divisible_by_24_and_cube_root_between_9_and_9_1 : 
  ∃ (n : ℕ), 
  (n % 24 = 0) ∧ 
  (9 < (n : ℚ) ^ (1 / 3 : ℚ)) ∧ 
  ((n : ℚ) ^ (1 / 3 : ℚ) < 9.1) ∧ 
  (n = 744) := by
  sorry

end find_integer_divisible_by_24_and_cube_root_between_9_and_9_1_l443_443960


namespace calculate_three_times_neg_two_l443_443173

-- Define the multiplication of a positive and a negative number resulting in a negative number
def multiply_positive_negative (a b : Int) (ha : a > 0) (hb : b < 0) : Int :=
  a * b

-- Define the absolute value multiplication
def absolute_value_multiplication (a b : Int) : Int :=
  abs a * abs b

-- The theorem that verifies the calculation
theorem calculate_three_times_neg_two : 3 * (-2) = -6 :=
by
  -- Using the given conditions to conclude the result
  sorry

end calculate_three_times_neg_two_l443_443173


namespace bird_families_before_migration_l443_443863

-- Define the variables for the number of bird families
variables (A B : ℕ)

-- The conditions
def condition1 := A = 42
def condition2 := B = 31
def condition3 := A = B + 11

-- The statement to prove
theorem bird_families_before_migration (A B : ℕ) (h1 : condition1) (h2 : condition2) (h3 : condition3) : A + B = 73 :=
sorry

end bird_families_before_migration_l443_443863


namespace BK_and_DK_equivalence_l443_443412

-- Definitions related to the given problem
variable (A B C D K : Point)
variable (square_ABC : square A B C D)
variable (right_triangle_ACK : right_triangle A C K)
variable (B_and_K_same_side_AC : same_side B K AC)

-- The statement in Lean
theorem BK_and_DK_equivalence :
  let AK := dist A K in
  let CK := dist C K in
  let BK := dist B K in
  let DK := dist D K in
  BK = (abs (AK - CK)) / (real.sqrt 2) ∧
  DK = (AK + CK) / (real.sqrt 2) :=
sorry

end BK_and_DK_equivalence_l443_443412


namespace triangle_fraction_l443_443788

theorem triangle_fraction (A B C K M E: Point) (h_isosceles : AC = BC)
  (h_E_on_AC : E ∈ Line AC) 
  (h_K_on_AB : K ∈ Line AB) 
  (h_M_on_BC : M ∈ Line BC) 
  (h_KE_parallel_BC : Parallel KE BC) 
  (h_EM_parallel_AB : Parallel EM AB)
  (h_ratio_BM_ME : BM / ME = 2 / 3) :
  area (Δ KEM) = (6 / 25) * area (Δ ABC) :=
sorry

end triangle_fraction_l443_443788


namespace total_number_of_bricks_l443_443109

variable (length_courtyard : ℕ := 25) -- in meters
variable (width_courtyard : ℕ := 16) -- in meters
variable (length_brick_cm : ℕ := 20) -- in centimeters
variable (width_brick_cm : ℕ := 10) -- in centimeters

theorem total_number_of_bricks :
  (let area_courtyard_m2 := length_courtyard * width_courtyard in -- in square meters
   let area_courtyard_cm2 := area_courtyard_m2 * 10000 in -- converting to square centimeters
   let area_brick_cm2 := length_brick_cm * width_brick_cm in
   area_courtyard_cm2 / area_brick_cm2 = 20000) :=
by
  sorry

end total_number_of_bricks_l443_443109


namespace smallest_operation_result_l443_443705

theorem smallest_operation_result : 
  let sqrt18 := real.sqrt 18
  let sqrt8 := real.sqrt 8 in
  min (sqrt18 + sqrt8) (min (sqrt18 - sqrt8) (min (sqrt18 * sqrt8) (sqrt18 / sqrt8))) = sqrt18 - sqrt8 :=
by
  sorry

end smallest_operation_result_l443_443705


namespace product_of_repeating_decimal_l443_443570

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l443_443570


namespace father_daughter_age_l443_443514

theorem father_daughter_age :
  ∃ (x : ℕ), (40 + x = 2 * (10 + x)) ↔ (x = 20) := 
begin
  -- Define the variables for father's age and daughter's age
  let father_current_age := 40,
  let daughter_current_age := 10,
  
  -- Define the number of years later
  existsi (20 : ℕ),
  split,
  { -- Prove the forward implication
    intro h,
    linarith, },
  { -- Prove the reverse implication
    intro h,
    rw h,
    ring },
end

end father_daughter_age_l443_443514


namespace max_sinA_sinC_l443_443304

variable (a b c A B C : ℝ)

-- Assuming the sides opposite to angles A, B, C in triangle ABC are a, b, c respectively.
axiom triangle_ABC: a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

-- Given condition a ∙ cos(A) = b ∙ sin(A)
axiom a_cosA_eq_b_sinA: a * Real.cos A = b * Real.sin A

-- Given condition B > π / 2
axiom B_gt_half_pi: B > Real.pi / 2

-- Angle sum in triangle A + B + C = π
axiom angle_sum: A + B + C = Real.pi

-- Prove that the maximum value of Real.sin A + Real.sin C equals 9 / 8
theorem max_sinA_sinC : Real.sin A + Real.sin C ≤ 9 / 8 :=
by
  sorry

end max_sinA_sinC_l443_443304


namespace total_distance_of_journey_l443_443106

theorem total_distance_of_journey :
  ∃ (D : ℝ), 
  (∀ (T : ℝ), D = 40 * T) ∧
  (∀ (T : ℝ), D = 35 * (T + 0.25)) → 
  D = 70 := 
by
  assume D T,
  sorry

end total_distance_of_journey_l443_443106


namespace max_value_of_fraction_l443_443001

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443001


namespace parallel_case_perpendicular_case_l443_443706

noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b (x : ℝ) : ℝ × ℝ := (x, -6)

theorem parallel_case (x : ℝ) :
  a.2 * b x.1 + a.1 * b x.2 = 0 ↔ x = -4 :=
by sorry

theorem perpendicular_case (x : ℝ) :
  a.1 * b x.1 + a.2 * b x.2 = 0 ↔ x = 9 :=
by sorry

end parallel_case_perpendicular_case_l443_443706


namespace range_of_omega_l443_443051

noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.sin (ω * x + φ)

variables (ω : ℝ) (φ : ℝ)
hypothesis (h1 : 0 < ω)
hypothesis (h2 : 0 < φ ∧ φ < Real.pi / 2)
hypothesis (h3 : f ω φ 0 = Real.sqrt 2 / 2)
hypothesis (h4 : ∀ x₁ x₂ : ℝ, (x₁ ∈ Set.Ioo (Real.pi / 2) Real.pi) → (x₂ ∈ Set.Ioo (Real.pi / 2) Real.pi) → x₁ ≠ x₂ → (x₁ - x₂) / (f ω φ x₁ - f ω φ x₂) < 0)

theorem range_of_omega : 1 / 2 ≤ ω ∧ ω ≤ 5 / 4 :=
sorry

end range_of_omega_l443_443051


namespace find_m_l443_443623

theorem find_m (m : ℤ) : 72519 * m = 724827405 → m = 9999 :=
begin
  intro h,
  sorry
end

end find_m_l443_443623


namespace max_norm_b_l443_443364

/-- Definitions -/
def vector_a : ℝ × ℝ := (2, 0)
def vector_b (x y : ℝ) : ℝ × ℝ := (x, y)
def angle_between_vectors (u v : ℝ × ℝ) : ℝ := 
  let dot := u.1 * v.1 + u.2 * v.2
  let norm_u := real.sqrt (u.1^2 + u.2^2)
  let norm_v := real.sqrt (v.1^2 + v.2^2)
  real.acos (dot / (norm_u * norm_v))

/-- Conditions -/
def condition1 (x y : ℝ) : Prop := vector_a = (2, 0)
def condition2 (x y : ℝ) : Prop := angle_between_vectors (vector_b x y) (vector_b x y - vector_a) = real.pi / 6

/-- Theorem -/
theorem max_norm_b (x y : ℝ) (h1 : condition1 x y) (h2 : condition2 x y) : ∥vector_b x y∥ ≤ 4 :=
by sorry

end max_norm_b_l443_443364


namespace canonical_line_eq_l443_443864

-- Define the system of linear equations
def system_of_equations (x y z : ℝ) : Prop :=
  (2 * x - 3 * y - 2 * z + 6 = 0 ∧ x - 3 * y + z + 3 = 0)

-- Define the canonical equation of the line
def canonical_equation (x y z : ℝ) : Prop :=
  (x + 3) / 9 = y / 4 ∧ (x + 3) / 9 = z / 3 ∧ y / 4 = z / 3

-- The theorem to prove equivalence
theorem canonical_line_eq : 
  ∀ (x y z : ℝ), system_of_equations x y z → canonical_equation x y z :=
by
  intros x y z H
  sorry

end canonical_line_eq_l443_443864


namespace basketball_team_lineup_count_l443_443881

theorem basketball_team_lineup_count :
  ∃ (team : Finset ℕ), team.card = 15 ∧
  (∀ (Tom Tim : ℕ), Tom ∈ team ∧ Tim ∈ team →
  ∃ (starters : Finset ℕ), starters.card = 5 ∧
  Tom ∈ starters ∧ Tim ∈ starters ∧
  (starters \ {Tom, Tim}).card = 3 ∧
  (starters \ {Tom, Tim}).card.comb 13 3 = 286) := sorry

end basketball_team_lineup_count_l443_443881


namespace find_x_values_l443_443996

-- Definitions of functions f and f⁻¹
def f_inv (y : ℝ) : ℝ := (y^2 + 1) / y^2
def f (y : ℝ) : ℝ := 1 / (y^2 * (1 / y^2 + 1))

-- Main theorem
theorem find_x_values (k : ℤ) :
  let x := real.pi / 3 + 2 * real.pi * k in
  real.cos x * f_inv (real.sin x) * f (real.arcsin x) = 2 / 3 :=
sorry

end find_x_values_l443_443996


namespace candy_bar_cost_l443_443588

theorem candy_bar_cost :
  ∃ C : ℕ, (C + 1 = 3) → (C = 2) :=
by
  use 2
  intros h
  linarith

end candy_bar_cost_l443_443588


namespace max_value_of_fraction_l443_443037

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443037


namespace macys_weekly_running_goal_l443_443366

theorem macys_weekly_running_goal :
  (3 * 6 + 6 = 24) :=
by
  -- Calculate miles run in 6 days
  have run_in_six_days : 3 * 6 = 18 := by norm_num
  -- Total miles goal including remaining 6 miles
  have total_goal := run_in_six_days + 6
  show total_goal = 24 by norm_num

end macys_weekly_running_goal_l443_443366


namespace carpet_total_cost_l443_443892

theorem carpet_total_cost (length_floor : ℕ) (width_floor : ℕ) (side_carpet : ℕ) (area_irregular : ℕ) (cost_per_square : ℕ) :
    length_floor = 24 →
    width_floor = 64 →
    side_carpet = 8 →
    area_irregular = 128 →
    cost_per_square = 24 →
    let area_floor := length_floor * width_floor,
        area_carpet := side_carpet * side_carpet,
        num_carpet_floor := area_floor / area_carpet,
        num_carpet_irregular := (area_irregular / area_carpet) + 1,
        total_carpet_squares := num_carpet_floor + num_carpet_irregular,
        total_cost := total_carpet_squares * cost_per_square
    in total_cost = 648 :=
by
  intros h_length h_width h_side h_area h_cost
  simp only [h_length, h_width, h_side, h_area, h_cost]
  let area_floor := 24 * 64
  let area_carpet := 8 * 8
  let num_carpet_floor := area_floor / area_carpet
  let num_carpet_irregular := (128 / area_carpet) + 1
  let total_carpet_squares := num_carpet_floor + num_carpet_irregular
  let total_cost := total_carpet_squares * 24
  have : total_cost = 648 := by sorry
  exact this

end carpet_total_cost_l443_443892


namespace max_value_of_fraction_l443_443015

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443015


namespace find_positive_integer_l443_443949

theorem find_positive_integer : ∃ (n : ℤ), n > 0 ∧ (24 : ℤ) ∣ n ∧ (9 : ℝ) < (n : ℝ).cbrt ∧ (n : ℝ).cbrt < 9.1 ∧ n = 744 := by
  sorry

end find_positive_integer_l443_443949


namespace calculate_eccentricity_ratio_l443_443252

noncomputable def e1 : ℝ := sorry
noncomputable def e2 : ℝ := sorry
constant P : (ℝ × ℝ)
constant F1 F2 : (ℝ × ℝ)
constant O : (ℝ × ℝ)
axiom h_curve_ellipse : ∀ (point : (ℝ × ℝ)), true -- ellipse properties here (placeholder)
axiom h_curve_hyperbola : ∀ (point : (ℝ × ℝ)), true -- hyperbola properties here (placeholder)
axiom h_common_foci : F1 = (-1, 0) ∧ F2 = (1, 0) -- assuming common foci for illustration
axiom h_common_point : 2 * dist O P = dist F1 F2

theorem calculate_eccentricity_ratio : 
  ∃ e1 e2 : ℝ, (∃ P : ℝ × ℝ, (e1 = e2) ∧ true ∧ true) →
  ∀ P : ℝ × ℝ, 2 * dist O P = dist F1 F2 →
  (e1 * e2) / (real.sqrt (e1 ^ 2 + e2 ^ 2)) = (real.sqrt 2) / 2 :=
by sorry

end calculate_eccentricity_ratio_l443_443252


namespace number_of_sets_satisfying_condition_l443_443059

def sets_union_condition (M : Set Nat) : Prop :=
  M ∪ {1} = {1, 2, 3}

theorem number_of_sets_satisfying_condition : 
  (Finset.filter sets_union_condition (Finset.powerset {2, 3})).card = 3 :=
sorry

end number_of_sets_satisfying_condition_l443_443059


namespace find_roses_last_year_l443_443997

-- Definitions based on conditions
def roses_last_year : ℕ := sorry
def roses_this_year := roses_last_year / 2
def roses_needed := 2 * roses_last_year
def rose_cost := 3 -- cost per rose in dollars
def total_spent := 54 -- total spent in dollars

-- Formulate the problem
theorem find_roses_last_year (h : 2 * roses_last_year - roses_this_year = 18)
  (cost_eq : total_spent / rose_cost = 18) :
  roses_last_year = 12 :=
by
  sorry

end find_roses_last_year_l443_443997


namespace probability_Bm_l443_443120

variables {Ω : Type*} [ProbabilitySpace Ω]

noncomputable def indicator (A : set Ω) : Ω → ℝ := λ ω, if ω ∈ A then 1 else 0

noncomputable def sigma_sum (n : ℕ) (A : fin n → set Ω) : Ω → ℕ :=
  λ ω, finset.univ.sum (λ i, indicator (A i) ω)

def Sm (m n : ℕ) (A : fin n → set Ω) : ℝ :=
  (finset.powerset_len m (finset.univ : finset (fin n))).sum
    (λ J, P (⋂ j ∈ J, A j))

noncomputable def G (n : ℕ) (A : fin n → set Ω) (s : ℝ) : ℝ :=
  ∑ m in finset.range (n+1), Sm m n A * (s - 1)^m

theorem probability_Bm (m n : ℕ) (A : fin n → set Ω) :
  ∀ (B : ℕ → set Ω), B m = {ω : Ω | sigma_sum n A ω = m} →
  P (B m) = ∑ k in finset.Icc m n, (-1)^(k - m) * (nat.choose k m) * Sm k n A :=
sorry

end probability_Bm_l443_443120


namespace part1_part2_1_part2_2_part2_3_l443_443189

def equivalence_point (f : ℝ → ℝ) (x : ℝ) : Prop := x = f x

theorem part1 :
  ∃ x > 1, equivalence_point (λ x: ℝ, 1 / (x - 1)) x ∧
    x = (1 + Real.sqrt 5) / 2 :=
sorry

theorem part2_1 (m n : ℝ) (h₁ : m^2 - m - 2 = 0) (h₂ : n^2 - n - 2 = 0) (h₃ : m ≠ n) :
  m^2 * n + m * n^2 = -2 :=
sorry

theorem part2_2 (p q : ℝ) (h₁ : p^2 = p + 2) (h₂ : 2 * q^2 = q + 1) (h₃ : p ≠ 2 * q) :
  p^2 + 4 * q^2 = 5 :=
sorry

def W1 (x : ℝ) : Prop := x ≥ 1 ∧ (λ x, x^2 - 2) x = x
def W2 (x : ℝ) : Prop := (x - 2)^2 - 2 = x
def W : ℝ → ℝ → Prop := λ x y, (W1 x ∧ W1 y) ∨ (W2 x ∧ W2 y)

theorem part2_3 :
  ∃ (x y : ℝ), W x y ∧
    ((x = 2 ∧ y = 2) ∨ (x = (5 - Real.sqrt 17) / 2 ∧ y = (5 - Real.sqrt 17) / 2)) :=
sorry

end part1_part2_1_part2_2_part2_3_l443_443189


namespace find_positive_integer_l443_443953

theorem find_positive_integer : ∃ (n : ℤ), n > 0 ∧ (24 : ℤ) ∣ n ∧ (9 : ℝ) < (n : ℝ).cbrt ∧ (n : ℝ).cbrt < 9.1 ∧ n = 744 := by
  sorry

end find_positive_integer_l443_443953


namespace measure_of_angle_x_l443_443730

namespace angle_proof

-- Define the parallel lines and angles
variable (k ℓ : Line) (A B C D F G H : Point)
variable (x : ℝ)

-- Conditions of the problem
axiom parallel_k_ell : k ∥ ℓ
axiom angle_A_40 : angle A B C = 40
axiom angle_B_90 : angle B C D = 90
axiom angle_C_30 : angle D A C = 30

-- Theorem to be proved
theorem measure_of_angle_x :
  x = 70 :=
sorry

end angle_proof

end measure_of_angle_x_l443_443730


namespace triangle_arithmetic_angles_l443_443407

/-- The angles in a triangle are in arithmetic progression and the side lengths are 6, 7, and y.
    The sum of the possible values of y equals a + sqrt b + sqrt c,
    where a, b, and c are positive integers. Prove that a + b + c = 68. -/
theorem triangle_arithmetic_angles (y : ℝ) (a b c : ℕ) (h1 : a = 3) (h2 : b = 22) (h3 : c = 43) :
    (∃ y1 y2 : ℝ, y1 = 3 + Real.sqrt 22 ∧ y2 = Real.sqrt 43 ∧ (y = y1 ∨ y = y2))
    → a + b + c = 68 :=
by
  sorry

end triangle_arithmetic_angles_l443_443407


namespace inequality_condition_l443_443889

theorem inequality_condition (a : ℝ) : 
  (∀ x : ℝ, x^2 - 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) ∨ (False) := 
sorry

end inequality_condition_l443_443889


namespace volume_of_sphere_is_correct_l443_443533

noncomputable def volume_of_sphere_dividing_cone (r m : ℝ) (r_nonneg : r = 7) (m_nonneg : m = 24) : ℝ :=
  let V_cone := (1 / 3) * Real.pi * r ^ 2 * m
  let V_half_cone := V_cone / 2
  let R := Real.cbrt (V_half_cone * 3 / (2 * Real.pi))
  (4 / 3) * Real.pi * R ^ 3

theorem volume_of_sphere_is_correct : volume_of_sphere_dividing_cone 7 24 7 rfl = 9800 * Real.pi :=
by
  sorry

end volume_of_sphere_is_correct_l443_443533


namespace king_of_addition_sum_l443_443708

theorem king_of_addition_sum : ∑ i in finset.Icc 6 21, i = 216 := 
by 
  sorry

end king_of_addition_sum_l443_443708


namespace find_real_numbers_with_means_l443_443085

theorem find_real_numbers_with_means (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (h_geom : Real.sqrt (a * b) = Real.sqrt 5) 
  (h_harm : 2 / ((1 / a) + (1 / b)) = 2) : 
  ( {a, b} = { (5 + Real.sqrt 5) / 2, (5 - Real.sqrt 5) / 2 } ) :=
by
  sorry

end find_real_numbers_with_means_l443_443085


namespace smallest_x_l443_443440

noncomputable def digit_sum (n : ℕ) : ℕ :=
n.digits.sum

theorem smallest_x (x : ℕ) (a : ℕ) :
  a = (100 * x + 4950) ∧ digit_sum a = 50 →
  ∃ y : ℕ, y = 99950 ∧ (∀ z : ℕ, a = (100 * z + 4950) ∧ digit_sum a = 50 → y ≤ z) :=
begin
  intros h,
  sorry
end

end smallest_x_l443_443440


namespace trigonometric_identity_l443_443866

theorem trigonometric_identity (x : ℝ) : 
  cos x * cos (2 * x) = sin (π / 4 + x) * sin (π / 4 + 4 * x) + sin (3 * π / 4 + 4 * x) * cos (7 * π / 4 - 5 * x) := 
sorry

end trigonometric_identity_l443_443866


namespace sum_of_alternating_sums_n_8_l443_443228

def alternating_sum (s : List ℕ) : ℤ :=
  s.head! * 2 + s.tail!.enum.map (λ ⟨i, a⟩, if i % 2 = 0 then a else -a).sum

def subset_alternating_sum (subset : Finset ℤ) : ℤ :=
  subset.powerset.filter (λ s, ¬s.is_empty).sum (λ s, alternating_sum (s.sort (· > ·)).val)

theorem sum_of_alternating_sums_n_8 : 
    subset_alternating_sum (Finset.range 9) = 3840 :=
  by
    sorry

end sum_of_alternating_sums_n_8_l443_443228


namespace max_product_of_three_l443_443862

theorem max_product_of_three (a b c d e : ℤ) (h : {a, b, c, d, e} = {-5, -4, -1, 6, 7}) :
  max (((a * b * c), (a * b * d), (a * b * e), (a * c * d), (a * c * e), 
        (a * d * e), (b * c * d), (b * c * e), (b * d * e), 
        (c * d * e)) : ℝ) 140 :=
by sorry

end max_product_of_three_l443_443862


namespace eval_expr_at_n_eq_3_l443_443606

theorem eval_expr_at_n_eq_3 : 
  ∀ n : ℤ, n = 3 → (n - 2) * (n - 1) * n * (n + 1) * (n + 2) + 10 = 130 :=
by
  intro n hn
  rw hn
  norm_num
  done

end eval_expr_at_n_eq_3_l443_443606


namespace cosine_double_angle_second_quadrant_l443_443650

theorem cosine_double_angle_second_quadrant 
  (α : ℝ) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : sin α + cos α = (√3) / 3) :
  cos (2 * α) = - (√5) / 3 :=
by
  sorry

end cosine_double_angle_second_quadrant_l443_443650


namespace arielle_age_l443_443835

theorem arielle_age (E A : ℕ) (h1 : E = 10) (h2 : E + A + E * A = 131) : A = 11 := by 
  sorry

end arielle_age_l443_443835


namespace product_value_l443_443937

theorem product_value (x : ℝ) (h : (Real.sqrt (6 + x) + Real.sqrt (21 - x) = 8)) : (6 + x) * (21 - x) = 1369 / 4 :=
by
  sorry

end product_value_l443_443937


namespace sum_of_roots_l443_443856

theorem sum_of_roots (a b c : ℚ) (h_eq : 6 * a^3 + 7 * a^2 - 12 * a = 0) (h_eq_b : 6 * b^3 + 7 * b^2 - 12 * b = 0) (h_eq_c : 6 * c^3 + 7 * c^2 - 12 * c = 0) : 
  a + b + c = -7/6 := 
by
  -- Insert proof steps here
  sorry

end sum_of_roots_l443_443856


namespace find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443978

theorem find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1 :
    ∃ n : ℕ, n > 0 ∧ (n % 24 = 0) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ n = 744 :=
by
  sorry

end find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443978


namespace find_number_l443_443699

-- Define the main condition and theorem.
theorem find_number (x : ℤ) : 45 - (x - (37 - (15 - 19))) = 58 ↔ x = 28 :=
by
  sorry  -- placeholder for the proof

end find_number_l443_443699


namespace check_correct_conditional_expression_l443_443478
-- importing the necessary library for basic algebraic constructions and predicates

-- defining a predicate to denote the symbolic representation of conditional expressions validity
def valid_conditional_expression (expr: String) : Prop :=
  expr = "x <> 1" ∨ expr = "x > 1" ∨ expr = "x >= 1" ∨ expr = "x < 1" ∨ expr = "x <= 1" ∨ expr = "x = 1"

-- theorem to check for the valid conditional expression among the given options
theorem check_correct_conditional_expression :
  (valid_conditional_expression "1 < x < 2") = false ∧ 
  (valid_conditional_expression "x > < 1") = false ∧ 
  (valid_conditional_expression "x <> 1") = true ∧ 
  (valid_conditional_expression "x ≤ 1") = true :=
by sorry

end check_correct_conditional_expression_l443_443478


namespace even_not_div_by_4_not_sum_consecutive_odds_l443_443389

theorem even_not_div_by_4_not_sum_consecutive_odds
  (e : ℤ) (h_even: e % 2 = 0) (h_nondiv4: ¬ (e % 4 = 0)) :
  ∀ n : ℤ, e ≠ n + (n + 2) :=
by
  sorry

end even_not_div_by_4_not_sum_consecutive_odds_l443_443389


namespace added_number_mean_eq_median_and_multiple_of_three_l443_443150

theorem added_number_mean_eq_median_and_multiple_of_three 
    (n : ℤ) 
    (h_multiple_3 : n % 3 = 0)
    (h_mean_eq_median : (35 + n) / 5 = (list.median ([4, 7, 11, 13] ++ [n]))) : 
    n = 0 :=
  sorry

end added_number_mean_eq_median_and_multiple_of_three_l443_443150


namespace find_positive_integer_l443_443952

theorem find_positive_integer : ∃ (n : ℤ), n > 0 ∧ (24 : ℤ) ∣ n ∧ (9 : ℝ) < (n : ℝ).cbrt ∧ (n : ℝ).cbrt < 9.1 ∧ n = 744 := by
  sorry

end find_positive_integer_l443_443952


namespace number_of_n_digit_numbers_with_even_5s_l443_443192

/-- The main statement we want to prove. -/
theorem number_of_n_digit_numbers_with_even_5s (n : ℕ) (hn : 1 ≤ n) : 
    nat.rec_on n 
      (8 = 8)  -- base case n = 1: trivial match with f_1 = 8
      (λ n' fn', ∃ m: ℕ, f n = ½ * (7 * 8^(n-1) + 9 * 10^(n-1))) := 
  sorry -- Proof is omitted.

end number_of_n_digit_numbers_with_even_5s_l443_443192


namespace vector_sum_possible_values_l443_443285

noncomputable def vector_a (x : ℝ) := (2 : ℝ, 4 : ℝ, x)
noncomputable def vector_b (y : ℝ) := (2 : ℝ, y, 2 : ℝ)

def magnitude (a : ℝ × ℝ × ℝ) : ℝ := real.sqrt (a.1 * a.1 + a.2.1 * a.2.1 + a.2.2 * a.2.2)

def dot_product (a b : ℝ × ℝ × ℝ) : ℝ := a.1 * b.1 + a.2.1 * b.2.1 + a.2.2 * b.2.2

theorem vector_sum_possible_values (x y : ℝ) :
  magnitude (vector_a x) = 6 →
  dot_product (vector_a x) (vector_b y) = 0 →
  (x + y = 1 ∨ x + y = -3) :=
by {
  sorry
}

end vector_sum_possible_values_l443_443285


namespace u2008_is_5898_l443_443761

-- Define the sequence as given in the problem.
def u (n : ℕ) : ℕ := sorry  -- The nth term of the sequence defined in the problem.

-- The main theorem stating u_{2008} = 5898.
theorem u2008_is_5898 : u 2008 = 5898 := sorry

end u2008_is_5898_l443_443761


namespace sculpture_height_is_34_inches_l443_443177

-- Define the height of the base in inches
def height_of_base_in_inches : ℕ := 2

-- Define the total height in feet
def total_height_in_feet : ℕ := 3

-- Convert feet to inches (1 foot = 12 inches)
def total_height_in_inches (feet : ℕ) : ℕ := feet * 12

-- The height of the sculpture, given the base and total height
def height_of_sculpture (total_height base_height : ℕ) : ℕ := total_height - base_height

-- State the theorem that the height of the sculpture is 34 inches
theorem sculpture_height_is_34_inches :
  height_of_sculpture (total_height_in_inches total_height_in_feet) height_of_base_in_inches = 34 := by
  sorry

end sculpture_height_is_34_inches_l443_443177


namespace angle_AMN_eq_30_l443_443718

-- This theorem concerns a quadrilateral ABCD with given properties.
theorem angle_AMN_eq_30
  (A B C D M N: Type) -- Points are of some type
  (h1: (AB = BC)) -- AB = BC
  (h2: (angle A = 20)) -- ∠A = 20°
  (h3: (angle B = 20)) -- ∠B = 20°
  (h4: (angle C = 30)) -- ∠C = 30°
  : (angle AMN = 30) := sorry

end angle_AMN_eq_30_l443_443718


namespace train_crosses_signal_pole_in_24_seconds_l443_443879

theorem train_crosses_signal_pole_in_24_seconds
  (train_length : ℝ)
  (platform_length : ℝ)
  (time_to_cross_platform : ℝ)
  (combined_length : train_length + platform_length = 487.5)
  (time_to_cross_platform_eq : time_to_cross_platform = 39) :
  let speed := combined_length / time_to_cross_platform in
  let time_to_cross_pole := train_length / speed in
  train_length = 300 →
  platform_length = 187.5 →
  time_to_cross_pole = 24 := 
by {
  intros,
  let speed := combined_length / time_to_cross_platform,
  have speed_eq : speed = 487.5 / 39 := by
    sorry,
  let time_to_cross_pole := train_length / speed,
  have time_eq : time_to_cross_pole = 300 / (487.5 / 39) := by
    sorry,
  rw speed_eq at time_eq,
  norm_num at time_eq,
  exact time_eq,
}

end train_crosses_signal_pole_in_24_seconds_l443_443879


namespace sorted_sum_inequality_equality_conditions_l443_443871

theorem sorted_sum_inequality {n : ℕ} {a b : fin n → ℝ} {i : fin n → fin n} (ha : ∀ j k, j ≤ k → a j ≤ a k) (hb : ∀ j k, j ≤ k → b j ≤ b k) (hi : function.bijective i) :
  (∑ j in finset.range n, a j * b (fin.reverse_cast j)) ≤ (∑ j in finset.range n, a j * b (i j)) ∧ (∑ j in finset.range n, a j * b (i j)) ≤ (∑ j in finset.range n, a j * b j) :=
sorry

theorem equality_conditions {n : ℕ} {a b : fin n → ℝ} (ha : ∀ j k, j ≤ k → a j ≤ a k) (hb : ∀ j k, j ≤ k → b j ≤ b k) :
  (∑ j in finset.range n, a j * b (fin.reverse_cast j) = ∑ j in finset.range n, a j * b (i j) ∨ ∑ j in finset.range n, a j * b (i j) = ∑ j in finset.range n, a j * b j) →
  (∀ j k, a j = a k) ∨ (∀ j k, b j = b k) :=
sorry

end sorted_sum_inequality_equality_conditions_l443_443871


namespace milton_books_l443_443780

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l443_443780


namespace balls_sum_divisibility_l443_443160

open NatRat

def probability_sum_divisible_by_3 (n : ℕ) : ℚ :=
  let xs := Finset.range n \ Finset.single 0
  let divisible_by_3 := xs.filter (λ x, x % 3 = 0)
  let mod1 := xs.filter (λ x, x % 3 = 1)
  let mod2 := xs.filter (λ x, x % 3 = 2)
  let favorable := (Finset.card divisible_by_3.choose₂.card) + (Finset.card mod1 * Finset.card mod2)
  let total := xs.choose₂.card
  favorable / total

theorem balls_sum_divisibility :
  probability_sum_divisible_by_3 20 = 32 / 95 :=
sorry

end balls_sum_divisibility_l443_443160


namespace expected_heads_l443_443334

def coin_flips : Nat := 64

def prob_heads (tosses : ℕ) : ℚ :=
  1 / 2^(tosses + 1)

def total_prob_heads : ℚ :=
  prob_heads 0 + prob_heads 1 + prob_heads 2 + prob_heads 3

theorem expected_heads : (coin_flips : ℚ) * total_prob_heads = 60 := by
  sorry

end expected_heads_l443_443334


namespace NaOH_combined_l443_443988

theorem NaOH_combined (n : ℕ) (h : n = 54) : 
  (2 * n) / 2 = 54 :=
by
  sorry

end NaOH_combined_l443_443988


namespace graphs_symmetric_about_a_axis_of_symmetry_l443_443875

def graph_symmetric_about_line (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a - x) = f (x - a)

theorem graphs_symmetric_about_a (f : ℝ → ℝ) (a : ℝ) :
  ∀ x, f (x - a) = f (a - (x - a)) :=
sorry

theorem axis_of_symmetry (f : ℝ → ℝ) :
  (∀ x : ℝ, f (1 + 2 * x) = f (1 - 2 * x)) →
  ∀ x, f x = f (2 - x) := 
sorry

end graphs_symmetric_about_a_axis_of_symmetry_l443_443875


namespace triangle_problem_l443_443738

structure Triangle (α : Type*) [Add α] where
  A B C M N O : α

variables {α : Type*} [Add α] [Eq α]

def on_side (x y : α) : Prop := sorry

theorem triangle_problem
  (A B C M N O : α) 
  (hM : on_side M A B) 
  (hN : on_side N B C)
  (hO : intersect CM AN O) 
  (h_eq : AM + AN = CM + CN) :
  AO + AB = CO + CB :=
sorry

end triangle_problem_l443_443738


namespace external_angle_bisector_ratio_l443_443327

-- Define the conditions
variables {A B C N' : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace N']
variables (vector b c : A → B)

-- Define AB and AC
def AB := c
def AC := b

-- Define the bisector condition
axiom bisector_of_external_angle_at_A (A B C N' : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace N'] 
  (AB : A → B) (AC : A → C) (N' : A → A)
  : isAngleBisectorOfLineSegmentA B C N'

-- State the theorem
theorem external_angle_bisector_ratio (A B C N' : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace N'] 
  (c : A → B) (b : A → C) 
  (h_ne : c ≠ b)
  (h_bisector : isAngleBisectorOfLineSegmentA B C N')
  : dist B N' / dist C N' = dist A B / dist A C :=
sorry

end external_angle_bisector_ratio_l443_443327


namespace larger_number_solution_l443_443444

theorem larger_number_solution (x y : ℝ) (h1 : x + y = 40) (h2 : x - y = 10) : x = 25 :=
by
  sorry

end larger_number_solution_l443_443444


namespace length_OP_l443_443344

def triangle (A B C : Type) := true

def is_centroid {A B C : Type} (O : Type) (A B C : triangle A B C) : Prop :=
true

def median_length_A (A : Type) (P : Type) : Nat := 30

def median_length_C (C : Type) (Q : Type) : Nat := 15

def segment_length (O Q : Type) : Nat := 5

theorem length_OP
  {A B C O P Q : Type}
  (h1 : is_centroid O (triangle A B C))
  (h2 : segment_length O Q = 5)
  (h3 : median_length_A A P = 30) :
  ∃ OP_len : Nat, OP_len = 20 :=
sorry

end length_OP_l443_443344


namespace shorter_stick_length_l443_443044

variable (L S : ℝ)

theorem shorter_stick_length
  (h1 : L - S = 12)
  (h2 : (2 / 3) * L = S) :
  S = 24 := by
  sorry

end shorter_stick_length_l443_443044


namespace cone_distance_proof_l443_443535

def cone_height (R r : ℝ) : ℝ :=
  sqrt (R^2 - r^2)

def distance_to_table (h r R : ℝ) : ℝ :=
  2 * (h * r) / R

theorem cone_distance_proof (r R : ℝ) (h : ℝ) (hR : R = 2 * r) 
  (hr : h = sqrt (R^2 - r^2)) :
  distance_to_table h r R = sqrt 3 := 
by
  sorry

end cone_distance_proof_l443_443535


namespace maximum_value_is_17_l443_443026

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443026


namespace angle_B_solution_l443_443713

variables (A B C a b c : ℝ)
hypothesis (h : a * Real.cos B - b * Real.cos A = (1 / 2) * c)
noncomputable def prove_angle_B : Prop :=
  B = Real.pi / 6

theorem angle_B_solution (h : a * Real.cos B - b * Real.cos A = (1 / 2) * c) : prove_angle_B A B C a b c :=
sorry

end angle_B_solution_l443_443713


namespace find_f_neg2016_l443_443662

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 1

theorem find_f_neg2016 (a : ℝ) (h1 : f a 2016 = 5) : f a (-2016) = -7 :=
by 
-- Definitions given in the conditions
  have h : a * 2016^3 - 1 = 5 := h1,
  sorry

end find_f_neg2016_l443_443662


namespace functional_equation_solution_l443_443766

noncomputable theory

-- Defining the set D as the set of real numbers excluding -1
def D : Set ℝ := { x : ℝ | x ≠ -1 }

-- Define the function f and its domain and codomain
def f (x : ℝ) : ℝ := (1 - x) / (1 + x)

-- Main theorem 
theorem functional_equation_solution (x y : ℝ) (hx : x ∈ D) (hy : y ∈ D) (h1 : x ≠ 0) (h2 : y ≠ -x) : 
  (f (f x) + y) * f (y / x) + f (f y) = x :=
by 
  sorry

end functional_equation_solution_l443_443766


namespace MN_passes_through_incenter_l443_443722

-- Definitions of the geometrical entities 
variables {A B C A1 B1 C1 M N I : Type}

-- Definition of the triangle ABC being acute-angled
def is_acute_angled (A B C : Type) : Prop := sorry

-- Definitions of angle bisectors intersecting the circumcircle
variable (circumcircle : Type)
def intersects_circumcircle (A B C A1 B1 C1 : Type) : Prop := sorry

-- Definitions of intersections M and N
def intersection_M (A B B1 C1 M : Type) : Prop := sorry
def intersection_N (B C A1 B1 N : Type) : Prop := sorry

-- Definition of the incenter I
def incenter (A B C I : Type) : Prop := sorry

-- The main theorem to be proved
theorem MN_passes_through_incenter (A B C A1 B1 C1 M N I : Type)
  [is_acute_angled A B C] 
  [intersects_circumcircle A B C A1 B1 C1]
  [intersection_M A B B1 C1 M] 
  [intersection_N B C A1 B1 N]
  [incenter A B C I] :
  passes_through I M N := 
sorry

end MN_passes_through_incenter_l443_443722


namespace max_value_of_fraction_l443_443000

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443000


namespace negation_proposition_l443_443826

theorem negation_proposition :
  (¬ ∀ x : ℝ, x^2 + 2 * x + 2 > 0) ↔ (∃ x : ℝ, x^2 + 2 * x + 2 ≤ 0) :=
by
  sorry

end negation_proposition_l443_443826


namespace problem_solution_l443_443121

-- Definitions of the conditions
def ABCDEF_inscribed_in_circle : Prop := ∃ O : Point, is_circumscribed_around_circle ABCDEF O
def AB_length : Prop := dist AB = 5
def BC_length : Prop := dist BC = 5
def CD_length : Prop := dist CD = 5
def DE_length : Prop := dist DE = 5
def EF_length : Prop := dist EF = 2
def FA_length : Prop := dist FA = 2

-- Definition of the theorem to prove
theorem problem_solution (ABCDEF_inscribed : ABCDEF_inscribed_in_circle)
                         (AB_eq_5 : AB_length)
                         (BC_eq_5 : BC_length)
                         (CD_eq_5 : CD_length)
                         (DE_eq_5 : DE_length)
                         (EF_eq_2 : EF_length)
                         (FA_eq_2 : FA_length) :
    (1 - cos (angle B)) * (1 - cos (angle AEF)) = 1 :=
begin
  sorry -- proof term would go here
end

end problem_solution_l443_443121


namespace problem_statement_l443_443281

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4}

-- Define the complement of B in U
def C_U_B : Set ℕ := { x | x ∈ U ∧ x ∉ B }

-- State the theorem
theorem problem_statement : (A ∩ C_U_B) = {1, 2} :=
by {
  -- Proof is omitted
  sorry
}

end problem_statement_l443_443281


namespace circle_condition_l443_443431

theorem circle_condition (m : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 4 * x - 2 * y + 5 * m = 0) ↔ m < 1 := by
  sorry

end circle_condition_l443_443431


namespace connect_segment_l443_443088

noncomputable def segment_exists (A B : Point) (d : ℝ) : Prop :=
  ∃ (f g : Point → Point → List LineSegment), by
    let reach AB_segment : Prop := |A - B| > (1 : ℝ)
    let compass_limit : LineSegment → Prop := λ AB, all (λ AB_segment, length AB_segment ≤ (10 : Real))
    let ruler_limit : Prop := ∀ AB_segment, ∃ points, List.sublist.points AB_segment.points
    have h1 : ∃ (l: List LineSegment), ∀ f g, f A B = l → True := sorry 
    exact reach & compass_limit & ruler_limit

variables {A B : Point} 

theorem connect_segment : segment_exists A B :=
  sorry 

end connect_segment_l443_443088


namespace problem_to_theorem_l443_443548

-- Definitions
variables {A B C D P Q H : Type} [LinearOrderedField A]
variables (BC DC BP BQ CH BH : A)
variables (AB BC PC : A) -- lengths
variables [fintype A] [decidable_rel A] -- useful for finiteness considerations

-- Assuming P and Q lie on sides AB and BC respectively, and given further conditions
def square_ABCD (A B C D : A) := 
  ∀  (P Q : A), P ∈ segment A B ∧ Q ∈ segment B C ∧ BP = BQ → 
  is_perpendicular D H Q → 
  is_perpendicular DH HQ

-- Now, we state the actual theorem to be proven
theorem problem_to_theorem:
  ∀  (P Q H : A), P ∈ segment A B ∧ Q ∈ segment B C ∧ BP = BQ ∧ 
  foot_perpendicular B PC H → 
  (is_perpendicular D H Q):=
begin
  sorry
end

end problem_to_theorem_l443_443548


namespace smaug_silver_coins_l443_443804

theorem smaug_silver_coins :
  ∀ (num_gold num_copper num_silver : ℕ)
  (value_per_silver value_per_gold conversion_factor value_total : ℕ),
  num_gold = 100 →
  num_copper = 33 →
  value_per_silver = 8 →
  value_per_gold = 3 →
  conversion_factor = value_per_gold * value_per_silver →
  value_total = 2913 →
  (num_gold * conversion_factor + num_silver * value_per_silver + num_copper = value_total) →
  num_silver = 60 :=
by
  intros num_gold num_copper num_silver value_per_silver value_per_gold conversion_factor value_total
  intros h1 h2 h3 h4 h5 h6 h_eq
  sorry

end smaug_silver_coins_l443_443804


namespace shaded_area_l443_443550

theorem shaded_area (side_square : ℝ) (leg_triangle : ℝ) (pi_value : ℝ) 
  (Hs : side_square = 20) (Ht : leg_triangle = 10) (Hp : pi_value = 3.14) :
  (area_shaded : ℝ) (Harea : area_shaded = 286) := by
  sorry

end shaded_area_l443_443550


namespace sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l443_443744

theorem sum_of_squares_multiple_of_five :
  ( (-1)^2 + 0^2 + 1^2 + 2^2 + 3^2 ) % 5 = 0 :=
by
  sorry

theorem sum_of_consecutive_squares_multiple_of_five 
  (n : ℤ) :
  ((n - 2)^2 + (n - 1)^2 + n^2 + (n + 1)^2 + (n + 2)^2) % 5 = 0 :=
by
  sorry

end sum_of_squares_multiple_of_five_sum_of_consecutive_squares_multiple_of_five_l443_443744


namespace max_diff_inequality_l443_443353

noncomputable def P (x y : list ℝ) : ℝ := 
  (list.zip x y).map (λ (xi, yi), xi - yi).maximum.get_or_else 0

noncomputable def G (x y : list ℝ) : ℝ :=
  x.maximum.get_or_else 0 - y.minimum.get_or_else 0

theorem max_diff_inequality (x y : list ℝ) (h : x.minimum.get_or_else 0 >= y.maximum.get_or_else 0) :
  P x y ≤ G x y ∧ G x y ≤ 2 * (P x y) := 
by
  sorry

end max_diff_inequality_l443_443353


namespace part1_part2_l443_443166

theorem part1 (m : ℝ) (h1 : m^2 - 5 * m + 6 = 0) (h2 : m^2 - 3 * m ≠ 0) : m = 2 := 
  sorry

theorem part2 (p q : ℝ) (h3 : (-2 + complex.I * 2) * (-2 + complex.I * 2) + p * (-2 + complex.I * 2) + q = 0) : p = 4 ∧ q = 8 :=
  sorry

end part1_part2_l443_443166


namespace primes_condition_l443_443934

theorem primes_condition (p : ℕ) (hp : nat.prime p) :
  ∃ (q : ℕ), q * q = 5^p + 4 * p^4 → p = 5 :=
by {
  sorry
}

end primes_condition_l443_443934


namespace minimum_value_16_sub_8sqrt3_l443_443350

noncomputable def min_value (a b c : ℝ) : ℝ :=
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (6 / c - 1)^2

theorem minimum_value_16_sub_8sqrt3 :
  ∀ (a b c : ℝ), 1 ≤ a ∧ a ≤ 2 ∧ 2 ≤ b ∧ b ≤ c ∧ c ≤ 6 →
  min_value a b c ≥ 16 - 8 * real.sqrt 3 :=
by
  sorry

end minimum_value_16_sub_8sqrt3_l443_443350


namespace persimmons_picked_l443_443745

theorem persimmons_picked : 
  ∀ (J H : ℕ), (4 * J = H - 3) → (H = 35) → (J = 8) := 
by
  intros J H hJ hH
  sorry

end persimmons_picked_l443_443745


namespace ellipse_equation_distance_sum_constant_max_triangle_area_l443_443245

-- Define the conditions for the ellipse
def a : ℝ := 2
def b : ℝ := 1
def c : ℝ := sqrt 3

axiom eccentricity : c / a = sqrt 3 / 2
axiom dot_product : (-c, -b) • (c, -b) = -2

-- The equation of the ellipse
theorem ellipse_equation : (x y : ℝ) (hx : x / sqrt 4 = x^2 / 4) (hy : y^2 = y^2) :
  x^2 / 4 + y^2 = 1 := 
begin
  sorry -- proof not needed
end

-- Line l intersects the ellipse C and the origin
def line_l (k m x : ℝ) : ℝ := k*x + m

-- Proving OA and OB's squared distance sum remains constant
theorem distance_sum_constant (k : ℝ) (m : ℝ) (hm : m > 0) :
  |(0, 0) - (x1, y1)|^2 + (|(0, 0) - (x2, y2)|^2 = constant :=
begin
  sorry -- proof not needed
end

-- Prove the maximum triangle area
theorem max_triangle_area (k : ℝ) (m : ℝ) (hm : m > 0) :
  k = 1/2 -> max_area := 
begin
  sorry -- proof not needed
end

end ellipse_equation_distance_sum_constant_max_triangle_area_l443_443245


namespace number_of_points_on_curve_C_l443_443060

/-- The parametric equations of curve C. -/
def parametric_curve_C (θ : ℝ) : ℝ × ℝ :=
  (2 + 3 * Real.cos θ, 1 + 3 * Real.sin θ)

/-- The equation of line l is x - 3y + 2 = 0. -/
def line_l (x y : ℝ) : Prop := x - 3 * y + 2 = 0

/-- The given distance from the curve to the line. -/
def distance := 7 * Real.sqrt 10 / 10

/-- The center of the curve C in rectangular form. -/
def center_C : ℝ × ℝ := (2, 1)

/-- The radius of the circle C. -/
def radius_C := 3

/-- The distance from the center of the circle to the line using formula. -/
def distance_to_line (x0 y0 : ℝ) (A B C : ℝ) : ℝ :=
  Real.abs (A * x0 + B * y0 + C) / Real.sqrt (A^2 + B^2)

/-- Proof that there are exactly 4 points on the curve that are the given distance away from the line. -/
theorem number_of_points_on_curve_C (θ : ℝ) :
  ∃ (x y : ℝ), 
    (x, y) = parametric_curve_C θ ∧
    Real.abs ((x - 2)^2 + (y - 1)^2 - 9) < 0.0001 ∧
    distance_to_line 2 1 1 (-3) 2 = (Real.sqrt 10 / 10) →
    (Real.abs (3 - Real.sqrt 10 / 10 - distance) < 0.0001 ∧ Real.abs (3 + Real.sqrt 10 / 10 - distance) < 0.0001) →
    ∃ (P : set (ℝ × ℝ)), 
      P = {p | p = (2 + 3 * Real.cos θ, 1 + 3 * Real.sin θ) ∧
                distance_between_point_and_line p 1 (-3) 2 = distance} ∧
      P.card = 4 := sorry

end number_of_points_on_curve_C_l443_443060


namespace find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443977

theorem find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1 :
    ∃ n : ℕ, n > 0 ∧ (n % 24 = 0) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ n = 744 :=
by
  sorry

end find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443977


namespace max_value_of_fraction_l443_443005

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443005


namespace max_value_of_fraction_l443_443006

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443006


namespace simplify_expression_l443_443397

variables {α : ℝ} {k : ℤ}

theorem simplify_expression (α : ℝ) (k : ℤ) :
    (2*k*π < α ∧ α < (2*k + 0.5)*π) ∨
    ((2*k + 1)*π < α ∧ α < (2*k + 1.5)*π) ∨
    ((2*k + 1.5)*π < α ∧ α < (2*k + 2)*π) →
    ((∃ m : ℤ, 2*m*π < α ∧ α < (2*m + 0.5)*π ∨ (2*m + 1)*π < α ∧ α < (2*m + 1.5)*π → 
      \(\frac{\csc α}{\sqrt{1+\operatorname{ctg}^{2} \alpha}} - \frac{\cos α}{\sqrt{1-\sin^2 α}} = 0 \)) ∧
     (∃ m : ℤ, (2*m + 0.5)*π < α ∧ α < (2*m + 1)*π → 
      \(\frac{\csc α}{\sqrt{1+\operatorname{ctg}^{2} \alpha}} - \frac{\cos α}{\sqrt{1-\sin^2 α}} = 2 \)) ∧
     (∃ m : ℤ, (2*m + 1.5)*π < α ∧ α < (2*m + 2)*π → 
      \(\frac{\csc α}{\sqrt{1+\operatorname{ctg}^{2} \alpha}} - \frac{\cos α}{\sqrt{1-\sin^2 α}} = -2 \))
    := by sorry

end simplify_expression_l443_443397


namespace part1_solution_part2_solution_l443_443322

-- Definitions for the given mathematical problem

def C1_parametric_equations (θ : ℝ) : ℝ × ℝ :=
  (cos θ, 1 + sin θ)

def C2_polar_equations (ρ θ : ℝ) (a : ℝ) : Prop :=
  ρ^2 = 2 * ρ * cos θ + a

def line_parametric_equations (t : ℝ) : ℝ × ℝ :=
  (3 + (sqrt 2 / 2) * t, -1 + (sqrt 2 / 2) * t)

-- Part (1) Lean statement
theorem part1_solution {θ : ℝ} (a : ℝ) (h : a = 0) :
  let C1_eq := (λ (θ : ℝ), (cos θ, 1 + sin θ)) in
  let C2_eq := (λ (ρ θ : ℝ), ρ^2 = 2 * ρ * cos θ + a) in
  (x_c1 y_c1 : ℝ) (hC1 : (x_c1, y_c1) = C1_eq θ)
  (x_c2 y_c2 : ℝ) (hC2 : (x_c2, y_c2) = C2_eq ρ θ) :
  set.prod {P | x_c1^2 + (y_c1 - 1)^2 = 1} {P | x_c2^2 + y_c2^2 = 2 * x_c2} = 2 :=
sorry

-- Part (2) Lean statement
theorem part2_solution {a : ℝ} :
  let line_eq := (λ (t : ℝ), (3 + (sqrt 2 / 2) * t, -1 + (sqrt 2 / 2) * t)) in
  let C2_eq := (λ (ρ θ : ℝ), ρ^2 = 2 * ρ * cos θ + a) in
  ∃ t1 t2 : ℝ, (forall t, line_eq t = (x, y)) → x * y = 1 → abs (4 - a) = 1 :=
sorry

end part1_solution_part2_solution_l443_443322


namespace two_digit_numbers_with_perfect_square_products_l443_443689

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def digits_product (n : ℕ) (d1 d2 : ℕ) : Prop :=
  n = d1 * 10 + d2 ∧ d1 * d2 = n

theorem two_digit_numbers_with_perfect_square_products :
  ∃ S : finset ℕ, S.card = 21 ∧ ∀ n ∈ S, 10 ≤ n ∧ n ≤ 99 ∧ (∃ d1 d2 : ℕ, n = d1 * 10 + d2 ∧ is_perfect_square (d1 * d2)) :=
sorry

end two_digit_numbers_with_perfect_square_products_l443_443689


namespace cost_gravelling_path_example_l443_443111

noncomputable def cost_of_gravelling_path (
    plot_length : ℝ,
    plot_breadth : ℝ,
    path_width : ℝ,
    cost_per_sqm : ℝ
) : ℝ :=
    let inner_length := plot_length - 2 * path_width in
    let inner_breadth := plot_breadth - 2 * path_width in
    let outer_area := plot_length * plot_breadth in
    let inner_area := inner_length * inner_breadth in
    let path_area := outer_area - inner_area in
    path_area * cost_per_sqm

theorem cost_gravelling_path_example :
  cost_of_gravelling_path 110 0.65 0.05 0.8 = 8.844 :=
by
  sorry

end cost_gravelling_path_example_l443_443111


namespace pencil_weight_l443_443651

theorem pencil_weight (total_weight : ℝ) (empty_case_weight : ℝ) (num_pencils : ℕ)
  (h1 : total_weight = 11.14) 
  (h2 : empty_case_weight = 0.5) 
  (h3 : num_pencils = 14) :
  (total_weight - empty_case_weight) / num_pencils = 0.76 := by
  sorry

end pencil_weight_l443_443651


namespace find_a_plus_b_l443_443298

theorem find_a_plus_b (x a b : ℝ) (ha : x = a + Real.sqrt b)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_eq : x^2 + 5 * x + 4/x + 1/(x^2) = 34) : a + b = 5 :=
sorry

end find_a_plus_b_l443_443298


namespace cosine_of_angle_between_a_and_b_l443_443283

variables (a b : ℝ × ℝ)

def a_def : a = (2, 4) := rfl
def b_def : a - (2, 2) * b = (0, 8) := by
  simp [a, b]
  sorry

theorem cosine_of_angle_between_a_and_b : 
  let cos_theta := ((a.1 * b.1 + a.2 * b.2) / (real.sqrt (a.1^2 + a.2^2) * real.sqrt (b.1^2 + b.2^2))) in
  a = (2, 4) ∧ a - 2 • b = (0, 8) → cos_theta = -3/5 :=
begin
  assume h,
  sorry
end

end cosine_of_angle_between_a_and_b_l443_443283


namespace hyperbola_equation_from_conditions_l443_443256

theorem hyperbola_equation_from_conditions :
  (∃ a b : ℝ, (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) ∧ 
  (∀ F : ℝ × ℝ, F = (1, 0)) ∧
  (∀ A B : ℝ × ℝ, A = (-1, b / a) ∧ B = (-1, - b / a)) ∧
  (∀ S : ℝ, S = sqrt 3 / 3)
  ∧ ((a^2 + b^2 = 1) ∧ (b / a = S)))) →
  (∀ x y : ℝ, (x^2 / (3 / 7) - y^2 / (4 / 7) = 1)) := 
sorry

end hyperbola_equation_from_conditions_l443_443256


namespace greatest_digit_sum_base7_l443_443093

theorem greatest_digit_sum_base7 (n : ℕ) (h1 : 0 < n) (h2 : n < 1729) :
  ∃ s, s = 22 ∧ s = List.sum (n.base7Digits s) :=
sorry

end greatest_digit_sum_base7_l443_443093


namespace block_fraction_visible_above_water_l443_443476

-- Defining constants
def weight_of_block : ℝ := 30 -- N
def buoyant_force_submerged : ℝ := 50 -- N

-- Defining the proof problem
theorem block_fraction_visible_above_water (W Fb : ℝ) (hW : W = weight_of_block) (hFb : Fb = buoyant_force_submerged) :
  (1 - W / Fb) = 2 / 5 :=
by
  -- Proof is omitted
  sorry

end block_fraction_visible_above_water_l443_443476


namespace chairs_to_remove_is_33_l443_443893

-- Definitions for the conditions
def chairs_per_row : ℕ := 11
def total_chairs : ℕ := 110
def students : ℕ := 70

-- Required statement
theorem chairs_to_remove_is_33 
  (h_divisible_by_chairs_per_row : ∀ n, n = total_chairs - students → ∃ k, n = chairs_per_row * k) :
  ∃ rem_chairs : ℕ, rem_chairs = total_chairs - 77 ∧ rem_chairs = 33 := sorry

end chairs_to_remove_is_33_l443_443893


namespace maximum_value_is_17_l443_443023

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443023


namespace xyz_product_l443_443697

-- Defining the variables and conditions
variables (x y z : ℝ)
variables (h1 : xy = 20 * real.cbrt 2)
variables (h2 : xz = 35 * real.cbrt 2)
variables (h3 : yz = 14 * real.cbrt 2)

-- The statement to be proved
theorem xyz_product : x * y * z = 140 :=
by
  sorry

end xyz_product_l443_443697


namespace avg_score_of_boys_l443_443537

variables (g b : ℕ) (m_b m_g : ℝ)

def students_total := 180
def average_score_total := 90
def boys_percentage_more_than_girls := 0.2
def girls_percentage_less_than_boys := 0.25

def number_of_boys := g + g * boys_percentage_more_than_girls
def girls_total := g
def boys_total := b
def game_participants := g + b = students_total

def girl_average_score := m_b * (1 - girls_percentage_less_than_boys)
def total_average_score := (students_total : ℝ) * average_score_total
def score_contribution := (g : ℝ) * m_g + (b : ℝ) * m_b

theorem avg_score_of_boys
  (H1 : number_of_boys = b)
  (H2 : g + b = students_total)
  (H3 : m_g = m_b * (1 - girls_percentage_less_than_boys))
  (H4 : (g : ℝ) * m_g + (b : ℝ) * m_b = total_average_score) :
  m_b = 102 := 
by 
  sorry

end avg_score_of_boys_l443_443537


namespace value_three_in_range_of_g_l443_443501

theorem value_three_in_range_of_g (a : ℝ) : ∀ (a : ℝ), ∃ (x : ℝ), x^2 + a * x + 1 = 3 :=
by
  sorry

end value_three_in_range_of_g_l443_443501


namespace altitude_of_isosceles_triangle_on_diagonal_l443_443242

theorem altitude_of_isosceles_triangle_on_diagonal (a b : ℝ) (h : ℝ) :
  let d := Real.sqrt (a ^ 2 + b ^ 2) in
  (1 / 2) * a * b = (1 / 2) * d * h →
  h = a * b / d :=
by
  intros
  sorry

end altitude_of_isosceles_triangle_on_diagonal_l443_443242


namespace find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443974

theorem find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1 :
    ∃ n : ℕ, n > 0 ∧ (n % 24 = 0) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ n = 744 :=
by
  sorry

end find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443974


namespace ratio_boys_to_girls_l443_443308

variable {S G : ℕ} (h : 1 / 2 * G = 1 / 5 * S)

theorem ratio_boys_to_girls (S G : ℕ) (h : 1 / 2 * G = 1 / 5 * S) : 
  let B := S - G
  in B / G = 3 / 2 := by
  sorry

end ratio_boys_to_girls_l443_443308


namespace exactly_one_box_empty_l443_443452

noncomputable def num_ways_to_place_balls : ℕ :=
  let boxes := (Finset.range 3).card,
  let balls := (Finset.range 4).card,
  -- Ways to choose 2 boxes out of 3 boxes
  let choose_boxes := Nat.choose 3 2,
  -- Each ball has 2 choices of two chosen boxes: 2^4
  let place_balls := 2 ^ 4,
  -- Subtract cases where all balls are in one box
  let valid_placements := place_balls - 2 in
  choose_boxes * valid_placements

theorem exactly_one_box_empty :
  num_ways_to_place_balls = 42 :=
sorry

end exactly_one_box_empty_l443_443452


namespace larger_number_is_3289_l443_443117

noncomputable def hcf : ℕ := 23
noncomputable def lcm_factors : List ℕ := [11, 13, 225]

def larger_number (A B : ℕ) : Prop :=
  let factors_A : List ℕ := A.factorization.keys.toList
  let factors_B : List ℕ := B.factorization.keys.toList
  A > B ∧ 
  A = hcf * 11 * 13 ∧ 
  B = hcf * 225 ∧ 
  23 ∈ factors_A ∧ 
  23 ∈ factors_B ∧ 
  (11 ∈ factors_A ∧ 13 ∈ factors_A ∧ 225 ∈ factors_B)

theorem larger_number_is_3289 : ∃ A B : ℕ, larger_number A B ∧ A = 3289 := 
by
  sorry

end larger_number_is_3289_l443_443117


namespace function_decreases_l443_443933

def op (m n : ℝ) : ℝ := - (m * n) + n

def f (x : ℝ) : ℝ := op x 2

theorem function_decreases (x1 x2 : ℝ) (h : x1 < x2) : f x1 > f x2 :=
by sorry

end function_decreases_l443_443933


namespace max_days_to_eat_candies_l443_443853

theorem max_days_to_eat_candies :
  ∃ a n : ℕ, n ≤ 39 ∧ n * (2 * a + n - 1) = 1554 ∧ n = 37 :=
begin
  sorry
end

end max_days_to_eat_candies_l443_443853


namespace sum_of_first_11_terms_l443_443729

theorem sum_of_first_11_terms (a : ℕ → ℝ) (d : ℝ) 
  (h1 : a 3 + a 6 + a 9 = 12)
  (h2 : ∀ n, a n = a 1 + (n - 1) * d) :
  Finset.sum (Finset.range 11) (λ n, a (n + 1)) = 44 :=
by sorry

end sum_of_first_11_terms_l443_443729


namespace max_value_of_fraction_l443_443014

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443014


namespace todd_initial_gum_l443_443078

-- Define the conditions and the final result
def initial_gum (final_gum: Nat) (given_gum: Nat) : Nat := final_gum - given_gum

theorem todd_initial_gum :
  initial_gum 54 16 = 38 :=
by
  -- Use the initial_gum definition to state the problem
  -- The proof is skipped with sorry
  sorry

end todd_initial_gum_l443_443078


namespace find_lambda_l443_443247

variables {A B C O : Type} 
variables [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] [vector_space ℝ O]
variables (OA OB OC : O)
variables (λ : ℝ)

-- collinear condition for points A, B, C
def collinear (A B C : Type) [vector_space ℝ A] [vector_space ℝ B] [vector_space ℝ C] :=
  ∃ x y : ℝ, x + y = 1 ∧ OA = x • OB + y • OC

-- given condition
def given_condition (λ : ℝ) := (4 : ℝ) • OA = (2 * λ) • OB + (3 : ℝ) • OC

theorem find_lambda (h1 : collinear A B C) (h2 : given_condition λ) : λ = 1 / 2 :=
by
  sorry

end find_lambda_l443_443247


namespace rate_percent_calculation_l443_443472

theorem rate_percent_calculation 
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ) 
  (h1 : SI = 3125) 
  (h2 : P = 12500) 
  (h3 : T = 7) 
  (h4 : SI = P * R * T / 100) :
  R = 3.57 :=
by
  sorry

end rate_percent_calculation_l443_443472


namespace equilateral_triangle_l443_443614

-- Definition of a triangle with side lengths a, b, c
structure Triangle :=
(a b c R : ℝ)
-- Assumption: a, b, c > 0, R > 0
(positive_sides : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0)
-- Given the equation involving cosines and sines
(given_eq : (a * Real.cos (angle_at_A a b c) + b * Real.cos (angle_at_B a b c) + c * Real.cos (angle_at_C a b c)) /
             (a * Real.sin (angle_at_A a b c) + b * Real.sin (angle_at_B a b c) + c * Real.sin (angle_at_C a b c)) =
             (a + b + c) / (9 * R))

-- Helper functions to calculate angles
def angle_at_A (a b c : ℝ) : ℝ := 
  sorry -- Substitute with actual implementation

def angle_at_B (a b c : ℝ) : ℝ := 
  sorry -- Substitute with actual implementation

def angle_at_C (a b c : ℝ) : ℝ := 
  sorry -- Substitute with actual implementation

theorem equilateral_triangle (t : Triangle) : t.a = t.b ∧ t.b = t.c :=
by
  sorry -- The actual proof goes here

end equilateral_triangle_l443_443614


namespace total_green_and_red_marbles_l443_443800

variable (S G R B : Type) [AddGrp G] [AddGrp R] [AddGrp B]
variable (sara_green sara_red sara_blue : G) 
variable (tom_green tom_red tom_blue : G) 
variable (lisa_green lisa_red lisa_blue : G)

-- conditions
def Sara := (sara_green = 3) ∧ (sara_red = 5) ∧ (sara_blue = 6)
def Tom := (tom_green = 4) ∧ (tom_red = 7) ∧ (tom_blue = 2)
def Lisa := (lisa_green = 5) ∧ (lisa_red = 3) ∧ (lisa_blue = 7)

-- question to prove
theorem total_green_and_red_marbles (hSara: Sara) (hTom: Tom) (hLisa: Lisa) : 
  (sara_green + tom_green + lisa_green) + (sara_red + tom_red + lisa_red) = 27 :=
by
  sorry

end total_green_and_red_marbles_l443_443800


namespace position_of_2010_is_correct_l443_443163

-- Definition of the arithmetic sequence and row starting points
def first_term : Nat := 1
def common_difference : Nat := 2
def S (n : Nat) : Nat := (n * (2 * first_term + (n - 1) * common_difference)) / 2

-- Definition of the position where number 2010 appears
def row_of_number (x : Nat) : Nat :=
  let n := (Nat.sqrt x) + 1
  if (n - 1) * (n - 1) < x && x <= n * n then n else n - 1

def column_of_number (x : Nat) : Nat :=
  let row := row_of_number x
  x - (S (row - 1)) + 1

-- Main theorem
theorem position_of_2010_is_correct :
  row_of_number 2010 = 45 ∧ column_of_number 2010 = 74 :=
by
  sorry

end position_of_2010_is_correct_l443_443163


namespace g_at_neg_one_l443_443824

noncomputable def g (x : ℝ) : ℝ := sorry

theorem g_at_neg_one : (∀ x : ℝ, x ≠ 2 / 3 → g x + g ((x + 2) / (2 - 3 * x)) = 2 * x) → g (-1) = -61 / 65 :=
begin
  intros h,
  sorry
end

end g_at_neg_one_l443_443824


namespace matrix_A5B_invertible_l443_443880

open Matrix 

-- Define the necessary types and variables
variables {α : Type*} [linear_ordered_ring α] 
variables (A B : Matrix (Fin 2) (Fin 2) α)

-- Define a theorem representing the problem
theorem matrix_A5B_invertible 
  (hA_inv : (A⁻¹).is_integral) 
  (hA_B_inv : ((A + B)⁻¹).is_integral)
  (hA_2B_inv : ((A + 2 * B)⁻¹).is_integral)
  (hA_3B_inv : ((A + 3 * B)⁻¹).is_integral)
  (hA_4B_inv : ((A + 4 * B)⁻¹).is_integral) : 
  ((A + 5 * B)⁻¹).is_integral :=
sorry

end matrix_A5B_invertible_l443_443880


namespace find_intersection_points_max_AB_value_l443_443315

def curve1 (t : ℝ) (α : ℝ) (hα : 0 ≤ α ∧ α < π) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

def curve2 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 4 * Real.sin θ in (ρ * Real.cos θ, ρ * Real.sin θ)

def curve3 (θ : ℝ) : ℝ × ℝ :=
  let ρ := 4 * Real.sqrt 3 * Real.cos θ in (ρ * Real.cos θ, ρ * Real.sin θ)

theorem find_intersection_points :
  let p1 := (0, 0)
  let p2 := (Real.sqrt 3, 3)
  ∃ (points : List (ℝ × ℝ)), points = [p1, p2] ∧ 
  ∀ p ∈ points, (∃ θ, curve2 θ = p) ∧ (∃ θ, curve3 θ = p) := sorry

theorem max_AB_value (α : ℝ) (hα : 0 ≤ α ∧ α < π) :
  let A := curve2 α
  let B := curve3 α
  ‖A.1 - B.1‖ + ‖A.2 - B.2‖ ≤ 8 := sorry

end find_intersection_points_max_AB_value_l443_443315


namespace probability_preferred_colors_l443_443840

-- The conditions: defining the set of balls and preferences
def balls : List String := ["red", "yellow", "blue", "green", "purple"]

def prefersA (color : String) : Prop :=
  color = "red" ∨ color = "yellow"

def prefersB (color : String) : Prop :=
  color = "yellow" ∨ color = "green" ∨ color = "purple"

-- The main theorem statement: the probability of both drawing their preferred color
theorem probability_preferred_colors :
  (let total_outcomes := 5 * 4 in
   let favorable_outcomes := 3 + 2 in
   favorable_outcomes / total_outcomes = 1 / 4) :=
by
  sorry

end probability_preferred_colors_l443_443840


namespace area_of_DEF_l443_443321

-- Defining the given conditions
def square_area_UVWX : ℝ := 64
def small_square_side : ℝ := 2
def isosceles_base_length : ℝ := 4
def isosceles_height : ℝ := 8

-- The main theorem to be proved
theorem area_of_DEF : 
  let area_UVWX := square_area_UVWX,
      side_UVWX := real.sqrt area_UVWX,
      side_DN := isosceles_height,
      side_EF := isosceles_base_length in
  side_UVWX = 8 ∧ side_DN = 8 ∧ side_EF = 4 →
  (1 / 2 * side_EF * side_DN = 16) := 
by 
  sorry

end area_of_DEF_l443_443321


namespace num_valid_functions_l443_443668

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {1, 2, 3, 4, 5}
def f (x : ℕ) (hx : x ∈ M) : ℕ := sorry -- Define a total function from M to N

theorem num_valid_functions :
    (∃ (f : ℕ → ℕ) (hf : ∀ x ∈ M, f(x) ∈ N), 
      ∀ (A B C : ℕ × ℕ) (D : ℕ × ℕ), 
      (A = (1, f 1) ∧ B = (2, f 2) ∧ C = (3, f 3)) ∧ 
      (
        -- Given condition simplified in abstract form
        (let DA := sorry -- use appropriate coordinates for D, A, B, C
         let DB := sorry
         let DC := sorry
         ∀ (λ : ℝ), μ * DA + DC = λ * DB)
      ) 
    ) 
    → ∃ (count : ℕ), count = 20 :=
begin
  sorry
end

end num_valid_functions_l443_443668


namespace sixth_group_points_l443_443637

-- Definitions of conditions
def total_data_points : ℕ := 40

def group1_points : ℕ := 10
def group2_points : ℕ := 5
def group3_points : ℕ := 7
def group4_points : ℕ := 6
def group5_frequency : ℝ := 0.10

def group5_points : ℕ := (group5_frequency * total_data_points).toInt

-- Theorem: The number of data points in the sixth group
theorem sixth_group_points :
  group1_points + group2_points + group3_points + group4_points + group5_points + x = total_data_points →
  x = 8 :=
by
  sorry

end sixth_group_points_l443_443637


namespace ratio_of_11th_terms_l443_443282

variable {ℕ : Type}

-- Given conditions
variable (S T : ℕ → ℝ)
variable (a b : ℕ → ℝ)
variable (n : ℕ)

-- Conditions from the problem
axiom sum_condition : ∀ n : ℕ, S n ≠ 0 ∧ T n ≠ 0
axiom ratio_condition : ∀ n : ℕ, S n / T n = 2 * n / (3 * n + 1)

theorem ratio_of_11th_terms :
  a 11 / b 11 = 21 / 32 := by
  sorry

end ratio_of_11th_terms_l443_443282


namespace sum_remainder_l443_443186

def seq : ℕ → ℕ
| 0 => 2
| 1 => 2
| 2 => 2
| (n + 3) => seq (n + 2) + seq (n + 1) + seq n

theorem sum_remainder :
  (∀ n, seq (n + 3) = seq (n + 2) + seq (n + 1) + seq n) →
  seq 22 = 2555757 →
  seq 23 = 4700770 →
  seq 24 = 8651555 →
  ((∑ k in finset.range 25, seq k) % 1000 = 656) :=
by
  sorry

end sum_remainder_l443_443186


namespace non_negative_integer_solutions_system_eq_l443_443222

theorem non_negative_integer_solutions_system_eq :
  ∀ (x y z t : ℕ), 
  (x + y = z + t) ∧ (z + t = x * y) →
    (x, y, z, t) ∈ {
      (0,0,0,0), (2,2,2,2), (1,5,2,3), (5,1,2,3), (1,5,3,2), (5,1,3,2),
      (2,3,1,5), (2,3,5,1), (3,2,1,5), (3,2,5,1)
    } :=
by
  sorry

end non_negative_integer_solutions_system_eq_l443_443222


namespace racers_in_final_segment_l443_443565

def initial_racers := 200

def racers_after_segment_1 (initial: ℕ) := initial - 10
def racers_after_segment_2 (after_segment_1: ℕ) := after_segment_1 - after_segment_1 / 3
def racers_after_segment_3 (after_segment_2: ℕ) := after_segment_2 - after_segment_2 / 4
def racers_after_segment_4 (after_segment_3: ℕ) := after_segment_3 - after_segment_3 / 3
def racers_after_segment_5 (after_segment_4: ℕ) := after_segment_4 - after_segment_4 / 2
def racers_after_segment_6 (after_segment_5: ℕ) := after_segment_5 - (3 * after_segment_5 / 4)

theorem racers_in_final_segment : racers_after_segment_6 (racers_after_segment_5 (racers_after_segment_4 (racers_after_segment_3 (racers_after_segment_2 (racers_after_segment_1 initial_racers))))) = 8 :=
  by
  sorry

end racers_in_final_segment_l443_443565


namespace find_vector_AM_l443_443792

variables {V : Type*} [add_comm_group V] [vector_space ℝ V]
          (A B C M : V)
          (a b : V) -- vectors corresponding to AB and AC

-- Conditions
axiom H1 : M = (2 / 7) • B + (5 / 7) • C
axiom H2 : B - A = a
axiom H3 : C - A = b

-- Proof that the result vector AM is (2/7)a + (5/7)b
theorem find_vector_AM : A - M = (2 / 7) • a + (5 / 7) • b := sorry

end find_vector_AM_l443_443792


namespace greatest_price_max_l443_443566

open Real

def average_price (total_revenue : ℝ) (num_products : ℕ) := total_revenue / num_products

def company_c_conditions (products : Finₓ 30 → ℝ) : Prop :=
  (average_price (∑ i, products i) 30 = 1500) ∧
  (∀ i, products i ≥ 600) ∧
  (∃ (S₁ : Finset (Finₓ 30)), S₁.card = 15 ∧ (∀ i ∈ S₁, products i < 1200)) ∧
  (∃ (S₂ : Finset (Finₓ 30)), S₂.card ≤ 5 ∧ (∀ i ∈ S₂, products i > 2000))

theorem greatest_price_max (products : Finₓ 30 → ℝ) (h : company_c_conditions products) :
  ∃ P_max, P_max = 14000 ∧ ∃ i, products i = P_max :=
sorry

end greatest_price_max_l443_443566


namespace factorization_solution_l443_443818

def factorization_problem : Prop :=
  ∃ (a b c : ℤ), (∀ (x : ℤ), x^2 + 17 * x + 70 = (x + a) * (x + b)) ∧ 
                 (∀ (x : ℤ), x^2 - 18 * x + 80 = (x - b) * (x - c)) ∧ 
                 (a + b + c = 28)

theorem factorization_solution : factorization_problem :=
sorry

end factorization_solution_l443_443818


namespace find_c_interval_l443_443207

theorem find_c_interval :
  {c : ℝ | (4 * c / 3 ≤ 8 + 4 * c) ∧ (8 + 4 * c < -3 * (1 + c))} = set.Icc (-3:ℝ) (-11 / 7) :=
by {
  -- Proof would go here, but it's omitted as instructed
  sorry
}

end find_c_interval_l443_443207


namespace milton_books_l443_443781

theorem milton_books (Z B : ℕ) (h1 : B = 4 * Z) (h2 : Z + B = 80) : Z = 16 :=
sorry

end milton_books_l443_443781


namespace quadrilateral_angle_BAD_l443_443392

theorem quadrilateral_angle_BAD {A B C D: Type*} [has_angle A B C D]
  (h₁ : AB = BC)
  (h₂ : BC = CD)
  (h₃ : angle ABC = 90)
  (h₄ : angle BCD = 150) : angle BAD = 75 :=
sorry

end quadrilateral_angle_BAD_l443_443392


namespace determinant_new_matrix_l443_443752

variables {α : Type*} [LinearOrderedField α]
variables (a b c : EuclideanSpace α (Fin 3))

noncomputable def D := Determinant ![a, b, c]

theorem determinant_new_matrix (a b c : EuclideanSpace α (Fin 3)) :
  Let D := a ⬝ (b ⨯ c)
  Let D' := Determinant ![a, b - a, c + a]
  D = D' :=
by
  sorry

end determinant_new_matrix_l443_443752


namespace Amaya_total_marks_l443_443905

theorem Amaya_total_marks (A M S Ma : ℕ) (hM : M = 70)
  (hS : S = M + 10) (hMa_art : Ma = (9/10 : ℚ) * A) 
  (hMa_art2 : Ma = A - 20) :
  A = 200 → Ma = 180 → S = 80 → M = 70 → (M + S + A + Ma = 530).
Proof.
  sorry

end Amaya_total_marks_l443_443905


namespace length_XY_l443_443498

variables (A B C D X Y : ℝ × ℝ)

def is_square (A B C D : ℝ × ℝ) : Prop :=
  A = (0, 0) ∧ B = (1, 0) ∧ C = (1, 1) ∧ D = (0, 1)

def equilateral_triangle (A B Y : ℝ × ℝ) : Prop :=
  dist A B = dist B Y ∧ dist B Y = dist Y A

theorem length_XY (h1 : is_square A B C D) 
  (h2 : equilateral_triangle A B Y ∧ equilateral_triangle C D X)
  (h3 : A = (0, 0)) (h4 : B = (1, 0)) (h5 : C = (1, 1)) (h6 : D = (0, 1))
  (hY : Y = (1/2, √3 / 2)) (hX : X = (1 / 2, 1 - √3 / 2)) :
  dist X Y = √3 - 1 := 
by sorry

end length_XY_l443_443498


namespace hares_cuts_l443_443427

-- Definitions representing the given conditions
def intermediates_fallen := 10
def end_pieces_fixed := 2
def total_logs := intermediates_fallen + end_pieces_fixed

-- Theorem statement
theorem hares_cuts : total_logs - 1 = 11 := by 
  sorry

end hares_cuts_l443_443427


namespace time_increase_percentage_l443_443373

theorem time_increase_percentage (total_distance : ℝ) (first_half_distance : ℝ)
  (first_half_speed : ℝ) (total_avg_speed : ℝ) :
  let time_first_half := first_half_distance / first_half_speed
  let total_time := total_distance / total_avg_speed
  let time_second_half := total_time - time_first_half
  let percentage_increase := ((time_second_half - time_first_half) / time_first_half) * 100
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_speed = 80 →
  total_avg_speed = 40 →
  percentage_increase = 200 :=
by
  intros total_distance_eq first_half_distance_eq first_half_speed_eq total_avg_speed_eq
  let time_first_half := first_half_distance / first_half_speed
  let total_time := total_distance / total_avg_speed
  let time_second_half := total_time - time_first_half
  let percentage_increase := ((time_second_half - time_first_half) / time_first_half) * 100
  have h1 : first_half_distance = 320 := by
    rw [total_distance_eq]
    exact first_half_distance_eq
  have h2 : time_first_half = 4 := by
    rw [h1, first_half_speed_eq]
    norm_num
  have h3 : total_time = 16 := by
    rw [total_distance_eq, total_avg_speed_eq]
    norm_num
  have h4 : time_second_half = 12 := by
    rw [h3, h2]
    norm_num
  have h5 : percentage_increase = 200 := by
    rw [h4, h2]
    norm_num
  exact h5

end time_increase_percentage_l443_443373


namespace frequency_of_sixth_group_l443_443639

theorem frequency_of_sixth_group :
  ∀ (total_data_points : ℕ)
    (freq1 freq2 freq3 freq4 : ℕ)
    (freq5_ratio : ℝ),
    total_data_points = 40 →
    freq1 = 10 →
    freq2 = 5 →
    freq3 = 7 →
    freq4 = 6 →
    freq5_ratio = 0.10 →
    (total_data_points - (freq1 + freq2 + freq3 + freq4) - (total_data_points * freq5_ratio)) = 8 :=
by
  sorry

end frequency_of_sixth_group_l443_443639


namespace max_value_of_fraction_l443_443016

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443016


namespace fatima_donates_75_square_inches_l443_443202

theorem fatima_donates_75_square_inches :
  ∀ (cloth: ℚ), cloth = 100 →
  (∃ (c1 c2 c3: ℚ), c1 = cloth / 2 ∧ c2 = c1 / 2 ∧ c3 = 75) →
  (c1 + c2 = c3) :=
by
  assume cloth
  assume h1 : cloth = 100
  assume h2 : ∃ (c1 c2 c3: ℚ), c1 = cloth / 2 ∧ c2 = c1 / 2 ∧ c3 = 75
  sorry

end fatima_donates_75_square_inches_l443_443202


namespace volume_region_between_spheres_l443_443454

-- Defining the volume of a sphere given its radius
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * (r ^ 3)

-- Stating the theorem
theorem volume_region_between_spheres :
  let r_middle := 8
  let r_largest := 12
  volume_sphere r_largest - volume_sphere r_middle = (4608 / 3) * Real.pi :=
by
  sorry

end volume_region_between_spheres_l443_443454


namespace both_propositions_false_l443_443765

def is_M_point (f : ℝ → ℝ) (a b c : ℝ) (h_c_ab : c ∈ set.Icc a b) :=
  ∃ (I : set ℝ), (c ∈ (set.Ioi a ∩ set.Iio b) ∩ I) ∧ ∀ (x ∈ (I ∩ set.Icc a b)), x ≠ c → f x < f c

theorem both_propositions_false (f : ℝ → ℝ) (a b x₀ : ℝ) (h_max : f x₀ = Finset.sup (set.Icc a b) f)
  (h_M_points : ∀ (a b : ℝ), a < b → is_M_point f a b b) : false :=
sorry

end both_propositions_false_l443_443765


namespace university_students_l443_443913

theorem university_students (C P : ℕ) (N n : ℕ)
    (h_total : 3000 = C + P - (C ∩ P))
    (h_C : 1500 ≤ C ∧ C ≤ 1800)
    (h_P : 750 ≤ P ∧ P ≤ 1050)
    (h_n : n = 850)
    (h_N : N = 250) :
    N - n = -600 :=
begin
  sorry 
end

end university_students_l443_443913


namespace option_B_correct_option_C_correct_l443_443751

variables (A B : Set ω) 

section 
  variable [ProbabilitySpace Ω]
  variable {A B : Event Ω}

  -- Conditions for Option B (independence, probabilities)
  variable (h_indep : indep_events A B)
  variable (h_PA : P A = 1/2) (h_PB : P B = 1/3)
  
  theorem option_B_correct : P (A ∪ B) = 2/3 := sorry

  -- Conditions for Option C (given probabilities, find P(B))
  variable (h_PA : P A = 1/2)
  variable (h_PnA_given_B : P (Aᶜ ∣ B) = 3/4)
  variable (h_PnA_given_nB : P (Aᶜ ∣ Bᶜ) = 3/8)
  
  theorem option_C_correct : P B = 1/3 := sorry
end

end option_B_correct_option_C_correct_l443_443751


namespace binomial_even_sum_l443_443329

theorem binomial_even_sum (n : ℕ) (h : nat.choose n 3 = nat.choose n 7) : (∑ k in finset.range (n + 1), if k % 2 = 0 then nat.choose n k else 0) = 2 ^ 9 :=
by
  -- Exact mathematical transformation of the problem using the given condition
  have h1 : n = 10 := sorry,
  -- Computing the sum of the binomial coefficients of even indices
  simp [h1],
  sorry

end binomial_even_sum_l443_443329


namespace find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443975

theorem find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1 :
    ∃ n : ℕ, n > 0 ∧ (n % 24 = 0) ∧ (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ n = 744 :=
by
  sorry

end find_integer_divisible_by_24_with_cube_root_between_9_and_9_point_1_l443_443975


namespace smallest_n_to_prevent_Bob_winning_l443_443068

theorem smallest_n_to_prevent_Bob_winning :
  let boxes := 60
  let initial_pebbles := n
  let smallest_n := 960
  (∃ (n : ℕ) (h : n > 0), n < smallest_n → Bob_wins game_conditions) → (n = smallest_n) :=
by
  sorry

end smallest_n_to_prevent_Bob_winning_l443_443068


namespace initial_water_amount_l443_443450

theorem initial_water_amount (players : ℕ) (water_per_player spill leftover : ℕ) 
  (h_players : players = 30) 
  (h_water_per_player : water_per_player = 200) 
  (h_spill : spill = 250) 
  (h_leftover : leftover = 1750) :
  (players * water_per_player + spill + leftover) / 1000 = 8 :=
by
  have h1 : players * water_per_player = 6000, by
    rw [h_players, h_water_per_player]
    norm_num,
  have h2 : players * water_per_player + spill = 6250, by
    rw [h1, h_spill]
    norm_num,
  have h3 : players * water_per_player + spill + leftover = 8000, by
    rw [h2, h_leftover]
    norm_num,
  rw [h3]
  norm_num
  rfl

end initial_water_amount_l443_443450


namespace evaluate_expression_at_minus_half_l443_443803

noncomputable def complex_expression (x : ℚ) : ℚ :=
  (x - 3)^2 + (x + 3) * (x - 3) - 2 * x * (x - 2) + 1

theorem evaluate_expression_at_minus_half :
  complex_expression (-1 / 2) = 2 :=
by
  sorry

end evaluate_expression_at_minus_half_l443_443803


namespace intersection_lines_l443_443430

theorem intersection_lines (a b : ℝ) (h1 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → x = 1/3 * y + a)
                          (h2 : ∀ x y : ℝ, (x = 3 ∧ y = 1) → y = 1/3 * x + b) :
  a + b = 8 / 3 :=
sorry

end intersection_lines_l443_443430


namespace max_value_of_fraction_l443_443039

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443039


namespace juniors_to_freshmen_ratio_l443_443167

variable (f s j : ℕ)

def participated_freshmen := 3 * f / 7
def participated_sophomores := 5 * s / 7
def participated_juniors := j / 2

-- The statement
theorem juniors_to_freshmen_ratio
    (h1 : participated_freshmen = participated_sophomores)
    (h2 : participated_freshmen = participated_juniors) :
    j = 6 * f / 7 ∧ f = 7 * j / 6 :=
by
  sorry

end juniors_to_freshmen_ratio_l443_443167


namespace discount_percentage_is_20_l443_443138

-- Definitions based on the given conditions
def CP : ℝ := 100          -- Cost price
def markup : ℝ := 0.5      -- Markup percentage
def profit : ℝ := 0.2      -- Profit percentage
def MP : ℝ := CP + CP * markup  -- Marked price
def SP : ℝ := CP + CP * profit  -- Selling price
def D : ℝ := MP - SP       -- Discount amount
def D_perc : ℝ := (D / MP) * 100  -- Discount percentage

-- Statement to prove
theorem discount_percentage_is_20 : D_perc = 20 := 
by
  sorry

end discount_percentage_is_20_l443_443138


namespace find_a_l443_443211

theorem find_a :
  ∀ (a : ℝ), 
  (∀ x : ℝ, 2 * x^2 - 2016 * x + 2016^2 - 2016 * a - 1 = a^2) → 
  (∃ x1 x2 : ℝ, 2 * x1^2 - 2016 * x1 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 2 * x2^2 - 2016 * x2 + 2016^2 - 2016 * a - 1 - a^2 = 0 ∧
                 x1 < a ∧ a < x2) → 
  2015 < a ∧ a < 2017 :=
by sorry

end find_a_l443_443211


namespace sequence_first_number_l443_443721

theorem sequence_first_number (a : ℕ) (seq : List ℕ) (h₁ : seq = [2, 16, 4, 14, 6, 12, 8]) : a = 2 :=
by
  have h₂ : 2 = 2 := rfl
  exact h₂

end sequence_first_number_l443_443721


namespace oranges_per_group_l443_443839

theorem oranges_per_group (total_oranges groups : ℕ) (h1 : total_oranges = 384) (h2 : groups = 16) :
  total_oranges / groups = 24 := by
  sorry

end oranges_per_group_l443_443839


namespace maximum_value_is_17_l443_443024

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443024


namespace find_parameter_a_exactly_two_solutions_l443_443215

noncomputable def system_has_two_solutions (a : ℝ) : Prop :=
∃ (x y : ℝ), |y - 3 - x| + |y - 3 + x| = 6 ∧ (|x| - 4)^2 + (|y| - 3)^2 = a

theorem find_parameter_a_exactly_two_solutions :
  {a : ℝ | system_has_two_solutions a} = {1, 25} :=
by
  sorry

end find_parameter_a_exactly_two_solutions_l443_443215


namespace find_x_l443_443985

def integer_part (x : ℝ) : ℤ := ⌊x⌋

theorem find_x (x : ℝ) : x^2 - 10 * (integer_part x) + 9 = 0 →
  (x = 1) ∨ (x = sqrt 61) ∨ (x = sqrt 71) ∨ (x = 9) :=
begin
  intro h,
  sorry
end

end find_x_l443_443985


namespace log_base_3_condition_l443_443124

theorem log_base_3_condition (a b : ℝ) (h1 : log 3 a > log 3 b) : a > b :=
sorry

end log_base_3_condition_l443_443124


namespace angle_A₁_B₁_B_eq_30_l443_443127

open EuclideanGeometry

-- Given conditions
variables {A B C A₁ B₁ : Point}
variable h_isosceles : IsIsoscelesTriangle A B C
variable h_angle_C : ∠ A C B = 20
variable h_secant_50 : AngleLineThroughPoint (LineThrough A B) (LineThrough A A₁) 50 = true
variable h_secant_60 : AngleLineThroughPoint (LineThrough A B) (LineThrough B B₁) 60 = true
variable h_secant_intersection_A₁ : Intersects (LineThrough A A₁) (LineThrough B C) A₁
variable h_secant_intersection_B₁ : Intersects (LineThrough B B₁) (LineThrough A C) B₁

-- Proposition to prove
theorem angle_A₁_B₁_B_eq_30 :
  ∠ A₁ B₁ B = 30 :=
by
  sorry -- proof goes here after solving interactively


end angle_A₁_B₁_B_eq_30_l443_443127


namespace total_area_of_house_is_2300_l443_443518

-- Definitions based on the conditions in the problem
def area_living_room_dining_room_kitchen : ℕ := 1000
def area_master_bedroom_suite : ℕ := 1040
def area_guest_bedroom : ℕ := area_master_bedroom_suite / 4

-- Theorem to state the total area of the house
theorem total_area_of_house_is_2300 :
  area_living_room_dining_room_kitchen + area_master_bedroom_suite + area_guest_bedroom = 2300 :=
by
  sorry

end total_area_of_house_is_2300_l443_443518


namespace greatest_t_value_l443_443616

theorem greatest_t_value :
  ∃ t_max : ℝ, (∀ t : ℝ, ((t ≠  8) ∧ (t ≠ -7) → (t^2 - t - 90) / (t - 8) = 6 / (t + 7) → t ≤ t_max)) ∧ t_max = -1 :=
sorry

end greatest_t_value_l443_443616


namespace average_grade_of_female_students_is_92_l443_443812

noncomputable def female_average_grade 
  (overall_avg : ℝ) (male_avg : ℝ) (num_males : ℕ) (num_females : ℕ) : ℝ :=
  let total_students := num_males + num_females
  let total_score := total_students * overall_avg
  let male_total_score := num_males * male_avg
  let female_total_score := total_score - male_total_score
  female_total_score / num_females

theorem average_grade_of_female_students_is_92 :
  female_average_grade 90 83 8 28 = 92 := 
by
  -- Proof steps to be completed
  sorry

end average_grade_of_female_students_is_92_l443_443812


namespace intersection_set_eq_l443_443280

-- Define M
def M : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1^2 / 16) + (p.2^2 / 9) = 1 }

-- Define N
def N : Set (ℝ × ℝ) := { p : ℝ × ℝ | (p.1 / 4) + (p.2 / 3) = 1 }

-- Define the intersection of M and N
def M_intersection_N := { x : ℝ | -4 ≤ x ∧ x ≤ 4 }

-- The theorem to be proved
theorem intersection_set_eq : 
  { p : ℝ × ℝ | p ∈ M ∧ p ∈ N } = { p : ℝ × ℝ | p.1 ∈ M_intersection_N } :=
sorry

end intersection_set_eq_l443_443280


namespace polynomial_factor_theorem_l443_443692

theorem polynomial_factor_theorem
  (P Q R S : Polynomial ℝ)
  (h : ∀ x : ℝ, P(x^5) + x * Q(x^5) + x^2 * R(x^5) = (x^4 + x^3 + x^2 + x + 1) * S(x)) :
  P 1 = 0 :=
sorry

end polynomial_factor_theorem_l443_443692


namespace find_set_of_natural_numbers_l443_443987

theorem find_set_of_natural_numbers (a b c d : ℕ) (h1 : a * b * c % d = 1) (h2 : a * b * d % c = 1)
  (h3 : a * c * d % b = 1) (h4 : b * c * d % a = 1) : {a, b, c, d} = {1, 2, 3, 4} :=
sorry

end find_set_of_natural_numbers_l443_443987


namespace find_larger_number_l443_443442

theorem find_larger_number (a b : ℝ) (h1 : a + b = 40) (h2 : a - b = 10) : a = 25 := by
  sorry

end find_larger_number_l443_443442


namespace slower_pump_time_l443_443870

def pool_problem (R : ℝ) :=
  (∀ t : ℝ, (2.5 * R * t = 1) → (t = 5))
  ∧ (∀ R1 R2 : ℝ, (R1 = 1.5 * R) → (R1 + R = 2.5 * R))
  ∧ (∀ t : ℝ, (R * t = 1) → (t = 12.5))

theorem slower_pump_time (R : ℝ) : pool_problem R :=
by
  -- Assume that the combined rates take 5 hours to fill the pool
  sorry

end slower_pump_time_l443_443870


namespace matthew_total_time_on_malfunctioning_day_l443_443368

-- Definitions for conditions
def assembling_time : ℝ := 1
def normal_baking_time : ℝ := 1.5
def malfunctioning_baking_time : ℝ := 2 * normal_baking_time
def decorating_time : ℝ := 1

-- The theorem statement
theorem matthew_total_time_on_malfunctioning_day :
  assembling_time + malfunctioning_baking_time + decorating_time = 5 :=
by
  -- This is where the proof would go
  sorry

end matthew_total_time_on_malfunctioning_day_l443_443368


namespace matrix_pow_eight_l443_443178

theorem matrix_pow_eight :
  let A := Matrix.of 2 2 (λ i j, if (i, j) = (0, 0) then 1 else if (i, j) = (0, 1) then -1 else if (i, j) = (1, 0) then 1 else 1) in
  A^8 = Matrix.scalar 2 16 :=
by
  let A := Matrix.of 2 2 (λ i j, if (i, j) = (0, 0) then 1 else if (i, j) = (0, 1) then -1 else if (i, j) = (1, 0) then 1 else 1)
  have h : A^8 = Matrix.scalar 2 16, from sorry 
  exact h

end matrix_pow_eight_l443_443178


namespace circle_tangent_line_standard_equation_l443_443995

-- Problem Statement:
-- Prove that the standard equation of the circle with center at (1,1)
-- and tangent to the line x + y = 4 is (x - 1)^2 + (y - 1)^2 = 2
theorem circle_tangent_line_standard_equation :
  (forall (x y : ℝ), (x + y = 4) -> (x - 1)^2 + (y - 1)^2 = 2) := by
  sorry

end circle_tangent_line_standard_equation_l443_443995


namespace element_in_set_l443_443361

def M : Set (ℤ × ℤ) := {(1, 2)}

theorem element_in_set : (1, 2) ∈ M :=
by
  sorry

end element_in_set_l443_443361


namespace stratified_sampling_correct_l443_443883

-- Definitions based on conditions
def num_classes_first_year : ℕ := 20
def students_per_class_first_year : ℕ := 50
def num_classes_second_year : ℕ := 24
def students_per_class_second_year : ℕ := 45
def total_students_to_be_selected : ℕ := 208

-- Total number of students in the first and second years
def total_students_first_year : ℕ := num_classes_first_year * students_per_class_first_year := by sorry
def total_students_second_year : ℕ := num_classes_second_year * students_per_class_second_year := by sorry

-- Total number of students in both grades
def total_students : ℕ := total_students_first_year + total_students_second_year := by sorry

-- Number of students to be selected from each year
def students_selected_first_year : ℕ := (total_students_first_year * total_students_to_be_selected) / total_students := by sorry
def students_selected_second_year : ℕ := (total_students_second_year * total_students_to_be_selected) / total_students := by sorry

-- Lean 4 theorem statement
theorem stratified_sampling_correct :
  students_selected_first_year = 100 ∧
  students_selected_second_year = 108 ∧
  (100 / total_students_first_year) = (108 / total_students_second_year) :=
begin
  sorry
end

end stratified_sampling_correct_l443_443883


namespace amount_less_than_twice_the_number_l443_443882

theorem amount_less_than_twice_the_number (n : ℕ) (h : n = 16) : ∃ x : ℕ, 2 * n - x = 20 ∧ x = 12 :=
by
  use 12
  split
  · rw [h]
    norm_num
  · rfl

end amount_less_than_twice_the_number_l443_443882


namespace valid_mapping_A_l443_443483

theorem valid_mapping_A : 
  let M := {-2, 0, 2}
  let P := {-4, 0, 4}
  let f := λ x : ℤ, x * x
  ∀ x ∈ M, ∃! y ∈ P, f x = y := 
by
  sorry

end valid_mapping_A_l443_443483


namespace last_three_digits_of_8_pow_105_l443_443220

theorem last_three_digits_of_8_pow_105 : (8 ^ 105) % 1000 = 992 :=
by
  sorry

end last_three_digits_of_8_pow_105_l443_443220


namespace num_triangles_l443_443787

theorem num_triangles (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 4) (hy : 1 ≤ y ∧ y ≤ 4) :
  (finset.univ.product finset.univ).filter (λ z, 1 ≤ z.1 ∧ z.1 ≤ 4 ∧ 1 ≤ z.2 ∧ z.2 ≤ 4).card = 16 → 516 :=
begin
  sorry
end

end num_triangles_l443_443787


namespace average_age_constant_l443_443877

theorem average_age_constant 
  (average_age_3_years_ago : ℕ) 
  (number_of_members_3_years_ago : ℕ) 
  (baby_age_today : ℕ) 
  (number_of_members_today : ℕ) 
  (H1 : average_age_3_years_ago = 17) 
  (H2 : number_of_members_3_years_ago = 5) 
  (H3 : baby_age_today = 2) 
  (H4 : number_of_members_today = 6) : 
  average_age_3_years_ago = (average_age_3_years_ago * number_of_members_3_years_ago + baby_age_today + 3 * number_of_members_3_years_ago) / number_of_members_today := 
by sorry

end average_age_constant_l443_443877


namespace complex_numbers_product_in_polar_form_l443_443555

open Real

-- Given definitions from the conditions

def complex_in_polar_form (r : ℝ) (θ : ℝ) : ℂ :=
  r * (cos θ + I * sin θ)

theorem complex_numbers_product_in_polar_form :
  let z1 := complex_in_polar_form 4 (25 * pi / 180)
  let z2 := complex_in_polar_form (-3) (48 * pi / 180)
  let product := z1 * z2
  let r := abs (4 * (-3))
  let θ := ((25 + 48 + 180) % 360) * pi / 180
  (r, θ) = (12, 253 * pi / 180) := 
by
  -- This will be the proof statement.
  sorry

end complex_numbers_product_in_polar_form_l443_443555


namespace choir_members_count_l443_443057

theorem choir_members_count (n : ℕ) (h1 : n % 10 = 4) (h2 : n % 11 = 5) (h3 : 200 ≤ n) (h4 : n ≤ 300) : n = 234 := 
sorry

end choir_members_count_l443_443057


namespace base_8_digits_sum_l443_443297

theorem base_8_digits_sum
    (X Y Z : ℕ)
    (h1 : 1 ≤ X ∧ X < 8)
    (h2 : 1 ≤ Y ∧ Y < 8)
    (h3 : 1 ≤ Z ∧ Z < 8)
    (h4 : X ≠ Y)
    (h5 : Y ≠ Z)
    (h6 : Z ≠ X)
    (h7 : 8^2 * X + 8 * Y + Z + 8^2 * Y + 8 * Z + X + 8^2 * Z + 8 * X + Y = 8^3 * X + 8^2 * X + 8 * X) :
  Y + Z = 7 * X :=
by
  sorry

end base_8_digits_sum_l443_443297


namespace coeff_x_in_expansion_l443_443553

theorem coeff_x_in_expansion : 
  (coeff (x : ℚ) ((1 + x) * (2 - x)^4)) = -16 :=
sorry

end coeff_x_in_expansion_l443_443553


namespace sequence_bounds_l443_443930

noncomputable def a_seq : ℕ → ℝ
| 0       := 1 / 2
| (k + 1) := a_seq k + (1 / (k+1)) * (a_seq k) ^ 2

theorem sequence_bounds (n : ℕ) (hn : 0 < n) : 
  1 - 1/n.to_real < a_seq n ∧ a_seq n < 1 :=
sorry

end sequence_bounds_l443_443930


namespace regular_n_gon_of_orthocenters_l443_443340

-- Definitions and assumptions
variables {n : ℕ} (hne : 0 < n)
variables (ω : points.inscribed_circle) (A : ι → ω.point) (P : ω)
variables {H : ι → ω.symm_complex}

-- Proposition to be proved
theorem regular_n_gon_of_orthocenters
  (regular_2n_gon : ∀ i, A (2*i) = ω.to_point i ∧ A (2*i+1) = ω.to_point (i+1))
  (orthocenter_def : ∀ i, H i = P + A (2*i) + A (2*i+1))
  : is_regular_n_gon (H : ι → points) :=
sorry

end regular_n_gon_of_orthocenters_l443_443340


namespace true_propositions_l443_443260

-- Definitions based on conditions

def proposition1 (a b c : ℝ) : Prop :=
  (a * c^2 > b * c^2) → (a > b)

def proposition2 : Prop :=
  let f1 : ℝ := 0    -- Placeholder values, not used in the theorems
  let f2 : ℝ := 0    -- Placeholder values, not used in the theorems
  let A : ℝ := 0     -- Placeholder values, not used in the theorems
  let B : ℝ := 0     -- Placeholder values, not used in the theorems
  let len_ch A : ℝ := 10 -- Major axis length
  (len_ch A = 10) ∧ (false) -- Incorrect due to incorrect condition

def proposition3 (p q : Prop) : Prop :=
  (¬ p) ∧ (p ∨ q) → q

def p : Prop := ∃ x : ℝ, x^2 + x + 1 < 0
def proposition4 : Prop :=
  (¬ p) ↔ (∀ x : ℝ, x^2 + x + 1 > 0)

-- The main proof statement
theorem true_propositions :
  (proposition1 ∧ proposition3) ∧ (¬ proposition2) ∧ (¬ proposition4) :=
by
  sorry

end true_propositions_l443_443260


namespace local_minimum_of_f_l443_443698

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x^2 + a * x - 1) * real.exp(x - 1)

theorem local_minimum_of_f (a : ℝ) (h : ∃ x, x = -2 ∧ (∃ x', (differentiable_at ℝ (λ x, f x a) x') ∧ deriv (λ x, f x a) (-2) = 0)) : f 1 a = -1 :=
by
  sorry

end local_minimum_of_f_l443_443698


namespace product_lcm_gcd_eq_product_original_numbers_l443_443556

theorem product_lcm_gcd_eq_product_original_numbers :
  let a := 12
  let b := 18
  (Int.gcd a b) * (Int.lcm a b) = a * b :=
by
  sorry

end product_lcm_gcd_eq_product_original_numbers_l443_443556


namespace exists_natural_n_l443_443794

theorem exists_natural_n (k : ℕ) : ∃ (n : ℕ), (real.sqrt (n + 1981^k) + real.sqrt n = (real.sqrt 1982 + 1)^k) := 
sorry

end exists_natural_n_l443_443794


namespace right_triangles_with_conditions_l443_443296

theorem right_triangles_with_conditions :
  (∃ (a b : ℤ), b < 50 ∧ a ^ 2 + b ^ 2 = (b + 2) ^ 2) → 
  (finset.count (λ (a b : ℤ), b < 50 ∧ a ^ 2 + b ^ 2 = (b + 2) ^ 2) finset.univ = 6) := 
sorry

end right_triangles_with_conditions_l443_443296


namespace similar_triangles_l443_443328

noncomputable def triangle_similarity_coefficient : ℝ :=
  ((4:ℝ) / 9)^2021

theorem similar_triangles (AB BC AC : ℝ) (h1 : AB = 5) (h2 : BC = 6) (h3 : AC = 4):
  ∃ (k : ℝ), k = triangle_similarity_coefficient ∧
  similar_triangles ABC (construct_triangle_ABC AB BC AC h1 h2 h3) (construct_triangle_2021 h1 h2 h3) :=
begin
  sorry
end

-- Helper Functions
def similar_triangles : (triangle → triangle → Prop) := sorry
def construct_triangle_ABC : (ℝ → ℝ → ℝ → triangle) := sorry
def construct_triangle_2021 : (ℝ → ℝ → ℝ → triangle) := sorry

end similar_triangles_l443_443328


namespace hyperbola_eccentricity_l443_443249

variables (a b c : ℝ) (F1 F2 M : ℝ × ℝ)
variable [fact (a > 0)]
variable [fact (b > 0)]

-- Conditions
def is_hyperbola (P : ℝ × ℝ) :=
  P.1^2 / a^2 - P.2^2 / b^2 = 1

def is_regular_triangle (F1 F2 M : ℝ × ℝ) :=
  dist F1 F2 = dist F2 M ∧ dist F2 M = dist M F1

def is_midpoint_on_hyperbola (P Q : ℝ × ℝ) :=
  is_hyperbola ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

-- Eccentricity of the hyperbola
def eccentricity (a b : ℝ) := 
  sqrt (1 + b^2 / a^2)

-- Theorem Statement
theorem hyperbola_eccentricity
  (h1 : is_hyperbola (F1, F2))
  (h2 : is_regular_triangle (F1, F2, M))
  (h3 : is_midpoint_on_hyperbola (M, F1)) :
  eccentricity a b = sqrt 3 + 1 :=
sorry

end hyperbola_eccentricity_l443_443249


namespace probability_angle_BPC_greater_than_90_proof_l443_443354

noncomputable def probability_angle_BPC_greater_than_90 (ABCD : set (ℝ × ℝ)) (P : ℝ × ℝ → ℝ) 
  (h_square : is_square ABCD) (h_P_inside : ∀ (p : ℝ × ℝ), p ∈ ABCD → P p ∈ ABCD) :
  ℝ := 
1 - (Real.pi / 8)

theorem probability_angle_BPC_greater_than_90_proof (ABCD : set (ℝ × ℝ)) (P : ℝ × ℝ → ℝ) 
  (h_square : is_square ABCD) (h_P_inside : ∀ (p : ℝ × ℝ), p ∈ ABCD → P p ∈ ABCD) :
  probability_angle_BPC_greater_than_90 ABCD P h_square h_P_inside = 1 - (Real.pi / 8) 
:= 
sorry

end probability_angle_BPC_greater_than_90_proof_l443_443354


namespace geometric_sequence_seventh_term_l443_443819

noncomputable def seventh_term (a r : ℕ) : ℕ := a * r^6

theorem geometric_sequence_seventh_term :
  ∀ (a r : ℕ), 
  (a * r^4 = 32) ∧ (a * r^10 = 2) → 
  (seventh_term a r = 8) :=
by
  intro a r
  assume h
  sorry

end geometric_sequence_seventh_term_l443_443819


namespace sum_even_odd_difference_l443_443467

theorem sum_even_odd_difference :
  let even_sum := (2023 / 2) * (0 + 4044)
  let odd_sum := (2023 / 2) * (1 + 4045) - 100
  even_sum - odd_sum = -2385 :=
by
  -- Define even_sum, odd_sum
  let even_sum : ℕ := (2023 / 2) * (0 + 4044)
  let odd_sum : ℕ := (2023 / 2) * (1 + 4045) - 100
  -- State the main equality theorem
  have : even_sum - odd_sum = -2385 := sorry,
  exact this

end sum_even_odd_difference_l443_443467


namespace tank_full_time_l443_443869

def tank_capacity : ℕ := 900
def fill_rate_A : ℕ := 40
def fill_rate_B : ℕ := 30
def drain_rate_C : ℕ := 20
def cycle_time : ℕ := 3
def net_fill_per_cycle : ℕ := fill_rate_A + fill_rate_B - drain_rate_C

theorem tank_full_time :
  (tank_capacity / net_fill_per_cycle) * cycle_time = 54 :=
by
  sorry

end tank_full_time_l443_443869


namespace right_triangles_count_l443_443048

-- Define points and their properties
variables (A B C D E : Type) [Point A] [Point B] [Point C] [Point D] [Point E]

-- Define the rectangle
def rectangle (ABCD : set (Point)) := ∃ (A B C D : Point), 
  (A ≠ B) ∧ (B ≠ C) ∧ (C ≠ D) ∧ (D ≠ A) ∧ 
  (AB ‖ CD) ∧ (BC ‖ AD) ∧ (AC ⊥ BD)

-- Conditions
axiom diagonal_AC (h1 : rectangle ABCD) : ∃ (AC : Line), (A ≠ C) ∧ (A, C ∈ ABCD) ∧ 
  (∀ (X : Point), X ∈ AC ↔ X ∈ ABCD) ∧ (∃ (E : Point), (E ∈ AC) ∧ ratio AE EC = 2)

-- Proof statement: Prove that the number of right triangles using {A, B, C, D, E} is 2
theorem right_triangles_count (h1 : rectangle ABCD) (h2 : diagonal_AC) : 
  num_right_triangles {A, B, C, D, E} = 2 :=
sorry

end right_triangles_count_l443_443048


namespace find_n_eq_l443_443666

theorem find_n_eq :
  ∀ n : ℕ, n > 0 → f_n(x) = (x : ℝ) / (2 ^ (n + 1) - 2 + 2 ^ n) :=
sorry

end find_n_eq_l443_443666


namespace jackson_weekly_mileage_increase_l443_443747

theorem jackson_weekly_mileage_increase :
  ∃ (weeks : ℕ), weeks = (7 - 3) / 1 := by
  sorry

end jackson_weekly_mileage_increase_l443_443747


namespace max_six_digit_multiple_of_6_is_965328_l443_443851

-- Define the given digits set
def digits : List ℕ := [2, 3, 5, 6, 8, 9]

-- Define the condition: the number must use each digit exactly once and be a multiple of 6
def is_valid (n : ℕ) : Prop :=
  (n.digits 10).perm digits ∧ (n % 6 = 0)

-- Define the greatest possible six-digit number that meets the conditions
def max_six_digit_multiple_of_6 := 965328

-- The theorem statement
theorem max_six_digit_multiple_of_6_is_965328 : 
  is_valid max_six_digit_multiple_of_6 :=
sorry

end max_six_digit_multiple_of_6_is_965328_l443_443851


namespace remainder_sum_mod_13_l443_443620

theorem remainder_sum_mod_13 : (1230 + 1231 + 1232 + 1233 + 1234) % 13 = 0 :=
by
  sorry

end remainder_sum_mod_13_l443_443620


namespace sequence_sum_fraction_l443_443667

def sequence_a : ℕ → ℕ
| 0       := 1  -- This will handle the 1-based indexing, where a₁ = 1
| (n + 1) := sequence_a n + (n + 1) + 1

theorem sequence_sum_fraction :
  (∑ k in Finset.range 2019, (1 : ℚ) / sequence_a (k + 1)) = 2019 / 1010 :=
by
  sorry

end sequence_sum_fraction_l443_443667


namespace andrey_stamps_l443_443386

theorem andrey_stamps :
  ∃ (x : ℕ), 
    x % 3 = 1 ∧ 
    x % 5 = 3 ∧ 
    x % 7 = 5 ∧ 
    150 < x ∧ 
    x ≤ 300 ∧ 
    x = 208 :=
begin
  sorry
end

end andrey_stamps_l443_443386


namespace difference_xy_l443_443707

theorem difference_xy (x y : ℝ) (h1 : x + y = 9) (h2 : x^2 - y^2 = 27) : x - y = 3 := sorry

end difference_xy_l443_443707


namespace find_positive_integer_l443_443951

theorem find_positive_integer : ∃ (n : ℤ), n > 0 ∧ (24 : ℤ) ∣ n ∧ (9 : ℝ) < (n : ℝ).cbrt ∧ (n : ℝ).cbrt < 9.1 ∧ n = 744 := by
  sorry

end find_positive_integer_l443_443951


namespace total_amount_paid_l443_443113

theorem total_amount_paid (hrs_a hrs_b hrs_c : ℕ) (amount_b : ℝ) :
  hrs_a = 9 → hrs_b = 10 → hrs_c = 13 → amount_b = 225 →
  (hrs_a * amount_b / hrs_b) + amount_b + (hrs_c * amount_b / hrs_b) = 720 :=
by
  intros ha hb hc hab
  have ra := 225 / 10
  have amount_a := ra * 9
  have amount_c := ra * 13
  calc amount_a + 225 + amount_c = 202.5 + 225 + 292.5 : sorry
                        ... = 720 : sorry

end total_amount_paid_l443_443113


namespace point_on_line_example_l443_443482

variable {l : Type}

def is_point_on_line (p : Type) (m : ℝ) (b : ℝ) : Prop :=
  ∃ (x y : ℝ), p = (x, y) ∧ (y = m * x + b)

theorem point_on_line_example :
  is_point_on_line (1, -2) (-2) 0 :=
sorry

end point_on_line_example_l443_443482


namespace proof_correctness_l443_443521

-- Define the new operation
def new_op (a b : ℝ) : ℝ := (a + b)^2 - (a - b)^2

-- Definitions for the conclusions
def conclusion_1 : Prop := new_op 1 (-2) = -8
def conclusion_2 : Prop := ∀ a b : ℝ, new_op a b = new_op b a
def conclusion_3 : Prop := ∀ a b : ℝ, new_op a b = 0 → a = 0
def conclusion_4 : Prop := ∀ a b : ℝ, a + b = 0 → (new_op a a + new_op b b = 8 * a^2)

-- Specify the correct conclusions
def correct_conclusions : Prop := conclusion_1 ∧ conclusion_2 ∧ ¬conclusion_3 ∧ conclusion_4

-- State the theorem
theorem proof_correctness : correct_conclusions := by
  sorry

end proof_correctness_l443_443521


namespace blue_segments_count_proof_l443_443495

theorem blue_segments_count_proof :
  let rows := 20
  let cols := 20
  let total_points := rows * cols
  let total_segments := (rows - 1) * cols * 2
  let red_points := 219
  let boundary_red_points := 39
  let black_segments := 237
  let internal_red_points := red_points - boundary_red_points
  let total_red_endpoints := boundary_red_points * 3 + internal_red_points * 4
  let total_non_black_segments := total_segments - black_segments
  -- Calculate the number of red segments k
  let k := (total_red_endpoints - black_segments) / 2
in total_non_black_segments - k = 223 := by
  -- This is where the calculation proof steps would go
  sorry

end blue_segments_count_proof_l443_443495


namespace smallest_m_4_and_n_229_l443_443802

def satisfies_condition (m n : ℕ) : Prop :=
  19 * m + 8 * n = 1908

def is_smallest_m (m n : ℕ) : Prop :=
  ∀ m' n', satisfies_condition m' n' → m' > 0 → n' > 0 → m ≤ m'

theorem smallest_m_4_and_n_229 : ∃ (m n : ℕ), satisfies_condition m n ∧ is_smallest_m m n ∧ m = 4 ∧ n = 229 :=
by
  sorry

end smallest_m_4_and_n_229_l443_443802


namespace find_X_in_rectangle_diagram_l443_443318

theorem find_X_in_rectangle_diagram :
  ∀ (X : ℝ),
  (1 + 1 + 1 + 2 + X = 1 + 2 + 1 + 6) → X = 5 :=
by
  intros X h
  sorry

end find_X_in_rectangle_diagram_l443_443318


namespace diameter_of_triangle_l443_443447

noncomputable def diameter_of_circumscribed_circle (a b c : ℕ) (h : a = 25 ∧ b = 39 ∧ c = 40) : ℚ := 
  let p := (a + b + c) / 2
  let S_Δ := Real.sqrt (p * (p - a) * (p - b) * (p - c))
  let R := (a * b * c) / (4 * S_Δ)
  2 * R

theorem diameter_of_triangle (h : 25 = 25 ∧ 39 = 39 ∧ 40 = 40) :
  diameter_of_circumscribed_circle 25 39 40 h = 125 / 3 :=
sorry

end diameter_of_triangle_l443_443447


namespace find_integer_divisible_by_24_with_cube_root_in_range_l443_443961

theorem find_integer_divisible_by_24_with_cube_root_in_range :
  ∃ (n : ℕ), (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ (24 ∣ n) ∧ n = 744 := by
    sorry

end find_integer_divisible_by_24_with_cube_root_in_range_l443_443961


namespace log2_q_value_l443_443206

/-- There are 50 teams playing a tournament where every team plays every other team exactly once. 
No ties occur and each team has an equal (50%) chance of winning any match. 
The probability that no two teams win the same number of games is p/q, 
where p and q are relatively prime positive integers. We need to find log2(q). -/
theorem log2_q_value :
  let n := 50,
  let total_games := n * (n - 1) / 2,
  let possible_outcomes := 2 ^ total_games,
  let factorial := nat.factorial n,
  let log_p_q := nat.floor_log2 possible_outcomes,
  ∀ (p q : ℕ),
    nat.gcd p q = 1 ∧
    q = 2 ^ (log_p_q - nat.pack (λ k, k + 1, n) (λ k, total_games / 2 ^ k)) →
  log2 q = 1178 :=
sorry

end log2_q_value_l443_443206


namespace shortest_distance_exists_l443_443346

noncomputable def point_R (u : ℝ) : ℝ × ℝ × ℝ :=
  (u + 4, 3 * u, 1)

noncomputable def point_S (v : ℝ) : ℝ × ℝ × ℝ :=
  (-v + 2, v + 4, 2 * v + 5)

noncomputable def distance_squared (u v : ℝ) : ℝ :=
  let RU := point_R u
  let SV := point_S v
  (RU.1 - SV.1)^2 + (RU.2 - SV.2)^2 + (RU.3 - SV.3)^2

theorem shortest_distance_exists :
  ∃ (u v : ℝ), distance_squared u v = 1 :=
by
  use (1 / 2), (-5 / 2)
  simp [distance_squared, point_R, point_S]
  sorry

end shortest_distance_exists_l443_443346


namespace symmetric_origin_coordinates_l443_443727

-- Given the coordinates (m, n) of point P
variables (m n : ℝ)
-- Define point P
def P := (m, n)

-- Define point P' which is symmetric to P with respect to the origin O
def P'_symmetric_origin : ℝ × ℝ := (-m, -n)

-- Prove that the coordinates of P' are (-m, -n)
theorem symmetric_origin_coordinates :
  P'_symmetric_origin m n = (-m, -n) :=
by
  -- Proof content goes here but we're skipping it with sorry
  sorry

end symmetric_origin_coordinates_l443_443727


namespace more_karabases_than_barabases_l443_443320

/-- In the fairy-tale land of Perra-Terra, each Karabas is acquainted with nine Barabases, 
    and each Barabas is acquainted with ten Karabases. We aim to prove that there are more Karabases than Barabases. -/
theorem more_karabases_than_barabases (K B : ℕ) (h1 : 9 * K = 10 * B) : K > B := 
by {
    -- Following the conditions and conclusion
    sorry
}

end more_karabases_than_barabases_l443_443320


namespace rectangle_2_letter_l443_443413

theorem rectangle_2_letter (rect1 rect2 rect3 rect4 rect5 : set char)
  (h1 : 'P' ∈ rect1 ∧ 'R' ∈ rect1 ∧ 'I' ∈ rect1 ∧ 'S' ∈ rect1 ∧ 'M' ∈ rect1)
  (h2 : 'P' ∈ rect2 ∧ 'R' ∈ rect2 ∧ 'I' ∈ rect2 ∧ 'S' ∈ rect2 ∧ 'M' ∈ rect2)
  (h3 : 'P' ∈ rect3 ∧ 'R' ∈ rect3 ∧ 'I' ∈ rect3 ∧ 'S' ∈ rect3 ∧ 'M' ∈ rect3)
  (h4 : rect4 = {'S'})
  (h5 : 'P' ∈ rect5 ∧ 'R' ∈ rect5 ∧ 'I' ∈ rect5 ∧ 'S' ∈ rect5 ∧ 'M' ∈ rect5)
  (distinct : ∀ i j, i ≠ j → ∃ c, c ∈ i ∧ c ∉ j) :
  'R' ∈ rect2 :=
by
  sorry

end rectangle_2_letter_l443_443413


namespace right_triangle_area_l443_443801

-- Define the initial lengths and the area calculation function.
def area_right_triangle (base height : ℕ) : ℕ :=
  (1 / 2) * base * height

theorem right_triangle_area
  (a : ℕ) (b : ℕ) (c : ℕ)
  (h1 : a = 18)
  (h2 : b = 24)
  (h3 : c = 30)  -- Derived from the solution steps
  (h4 : a ^ 2 + b ^ 2 = c ^ 2) :
  area_right_triangle a b = 216 :=
sorry

end right_triangle_area_l443_443801


namespace cos_75_degree_proof_l443_443924

theorem cos_75_degree_proof :
  ∀ (cos60 sin60 cos15 sin15: ℝ), 
  cos60 = 1 / 2 →
  sin60 = sqrt 3 / 2 →
  cos15 = (sqrt 6 + sqrt 2) / 4 →
  sin15 = (sqrt 6 - sqrt 2) / 4 →
  cos (75 * (Real.pi / 180)) = (sqrt 6 - sqrt 2) / 4 :=
by
  intros cos60 sin60 cos15 sin15 hcos60 hsin60 hcos15 hsin15
  rw [Real.cos_add, hcos60, hsin60, hcos15, hsin15]
  sorry

end cos_75_degree_proof_l443_443924


namespace incenter_orthocenter_parallel_l443_443405

open EuclideanGeometry

variables {A B C I O H D E F X : Point}

-- Define the properties of the points
axiom incenter_I : Incenter I A B C
axiom circumcenter_O : Circumcenter O A B C
axiom orthocenter_H : Orthocenter H A B C
axiom D_def : D = Line.intersection (line_through A I) (line_through B C)
axiom E_def : E = Line.intersection (line_through B I) (line_through A C)
axiom F_def : F = Line.intersection (line_through C I) (line_through A B)
axiom orthocenter_X_DEF : Orthocenter X D E F

-- The theorem to prove
theorem incenter_orthocenter_parallel :
  Parallel (line_through I X) (line_through O H) :=
sorry

end incenter_orthocenter_parallel_l443_443405


namespace fourth_square_area_l443_443461

-- Definitions based on given conditions
variables {XYZ XZW : Type*}
variables a b c d : ℕ
variables (h_area_XY : a = 25)
variables (h_area_YZ : b = 4)
variables (h_area_XW : c = 49)

-- Problem statement: prove the area of the fourth square
theorem fourth_square_area (h1 : a = 25) (h2 : b = 4) (h3 : c = 49) :
  d = 78 :=
begin
  sorry
end

end fourth_square_area_l443_443461


namespace evaluate_expression_l443_443198

theorem evaluate_expression : sqrt (9 / 4) - sqrt (4 / 9) = 5 / 6 := by
  sorry

end evaluate_expression_l443_443198


namespace milton_books_l443_443777

variable (z b : ℕ)

theorem milton_books (h₁ : z + b = 80) (h₂ : b = 4 * z) : z = 16 :=
by
  sorry

end milton_books_l443_443777


namespace limit_problem_l443_443925

theorem limit_problem:
  (∀ ε > 0, ∃ N : ℝ, ∀ n : ℝ, n > N → abs((n * sqrt(n) * (n - real.cbrt(n^5 - 5))) - 0) < ε) := 
by 
  sorry

end limit_problem_l443_443925


namespace integral_evaluation_l443_443199

def integral_expression : ℝ := ∫ x in -1 .. 1, (real.sqrt (1 - x^2) - 1)

theorem integral_evaluation : integral_expression = (Real.pi / 2 - 2) := by
  sorry

end integral_evaluation_l443_443199


namespace solve_for_Q_l443_443398

theorem solve_for_Q (Q : ℝ) (h : sqrt (Q^4) = 32 * real.sqrt (64^(1/6))) : Q = 8 :=
begin
  sorry
end

end solve_for_Q_l443_443398


namespace distance_inequality_even_distance_inequality_odd_l443_443401

variable {n : ℕ}
variable {A : Fin n → ℝ × ℝ}
variable {P : ℝ}
variable {d : ℝ}
variable {O : ℝ × ℝ}

def centroid (A : Fin n → ℝ × ℝ) : ℝ × ℝ :=
  let (x_sum, y_sum) := Finset.univ.fold (λ (acc : ℝ × ℝ) (i : Fin n), (acc.1 + (A i).1, acc.2 + (A i).2)) (0, 0)
  (x_sum / n, y_sum / n)

def perimeter (A : Fin n → ℝ × ℝ) : ℝ :=
  Finset.univ.fold (λ acc (i : Fin n), acc + (dist (A i) (A ((i : ℕ + 1) % n).natAbs))) 0

theorem distance_inequality_even (h_even : n % 2 = 0) (h_sum_dist : d = Finset.univ.sum (λ i, dist (A i) O))
  (h_centroid : O = centroid A) (h_perimeter : P = perimeter A) : 
  d ≤ (n / 4) * P := sorry

theorem distance_inequality_odd (h_odd : n % 2 = 1) (h_sum_dist : d = Finset.univ.sum (λ i, dist (A i) O))
  (h_centroid : O = centroid A) (h_perimeter : P = perimeter A) : 
  d ≤ ((n^2 - 1) / (4 * n)) * P := sorry

end distance_inequality_even_distance_inequality_odd_l443_443401


namespace length_of_AB_l443_443732

noncomputable def is_isosceles_triangle (A B C : Type) [HasMetric (A B C)] : Prop :=
∃ (S : Type), HasMetric S → (B = S) ∧ (C = S)

theorem length_of_AB {A B C D : Type} [HasMetric (A B C D)]
  (isosceles_ABC : is_isosceles_triangle A B C)
  (isosceles_CBD : is_isosceles_triangle C B D)
  (perimeter_CBD : Metric.perimeter (C B D) = 19)
  (perimeter_ABC : Metric.perimeter (A B C) = 20)
  (BD_length : Metric.length (B D) = 7) :
  Metric.length (A B) = 8 :=
sorry

end length_of_AB_l443_443732


namespace correct_statement_is_D_l443_443156

theorem correct_statement_is_D :
  (¬(∀ A B C : Point, A ≠ B ∧ B ≠ C ∧ A ≠ C → ∃ α : Plane, A ∈ α ∧ B ∈ α ∧ C ∈ α)) ∧
  (¬(∀ l₁ l₂ : Line, l₁ ≠ l₂ ∧ (∃ P : Point, P ∈ l₁ ∧ P ∈ l₂) → ∃ α : Plane, ∀ Q : Point, (Q ∈ l₁ ∨ Q ∈ l₂) → Q ∈ α)) ∧
  (¬(∀ l₁ l₂ l₃ : Line, (∃ P₁ : Point, P₁ ∈ l₁ ∧ P₁ ∈ l₂) ∧ (∃ P₂ : Point, P₂ ∈ l₂ ∧ P₂ ∈ l₃) ∧ (∃ P₃ : Point, P₃ ∈ l₃ ∧ P₃ ∈ l₁) → ∃ α : Plane, ∀ R : Point, (R ∈ l₁ ∨ R ∈ l₂ ∨ R ∈ l₃) → R ∈ α)) ∧
  (∃ l₁ l₂ l₃ : Line, (∃ P : Point, P ∈ l₁ ∧ P ∈ l₂ ∧ P ∈ l₃) ∧ ¬(∃ α : Plane, ∀ S : Point, (S ∈ l₁ ∨ S ∈ l₂ ∨ S ∈ l₃) → S ∈ α)) :=
by
  sorry

end correct_statement_is_D_l443_443156


namespace acute_angle_AC_BD_l443_443716

noncomputable def acute_angle_between_diagonals (ABCD : Type) (AC : Type) (BD : Type) : angle :=
  sorry

theorem acute_angle_AC_BD (ABCD : Type) (h1 : convex_quadrilateral ABCD) (h2 : ¬ parallel_sides ABCD)
  (angle_list : list ℝ) (h3 : angle_list = [16, 19, 55, 55])
  (AC BD : Type) : acute_angle_between_diagonals ABCD AC BD = 87 :=
by
  sorry

end acute_angle_AC_BD_l443_443716


namespace fraction_unshaded_area_l443_443319

theorem fraction_unshaded_area (r : ℝ) (rL : ℝ) (h1 : rL = 3 * r) : 
  (λ A B, A / B = (8 : ℝ) / 9) (9 * Real.pi * r ^ 2 - Real.pi * r ^ 2) (9 * Real.pi * r ^ 2) :=
by
  -- We assume the variables and conditions directly
  have h_area_small : ℝ := Real.pi * r ^ 2
  have h_area_large : ℝ := Real.pi * (3 * r) ^ 2
  sorry

end fraction_unshaded_area_l443_443319


namespace min_value_of_reciprocals_l443_443423

theorem min_value_of_reciprocals (m n : ℝ) (h1 : m + n = 2) (h2 : m * n > 0) : 
  (1 / m) + (1 / n) = 2 :=
by
  -- the proof needs to be completed here.
  sorry

end min_value_of_reciprocals_l443_443423


namespace isosceles_triangle_area_l443_443848

theorem isosceles_triangle_area :
  ∀ (P Q R S : ℝ) (h1 : dist P Q = 26) (h2 : dist P R = 26) (h3 : dist Q R = 50),
  ∃ (area : ℝ), area = 25 * Real.sqrt 51 :=
by
  sorry

end isosceles_triangle_area_l443_443848


namespace find_derivative_l443_443860

noncomputable def f (a b x : ℝ) : ℝ := a * real.log x + (b / x)

theorem find_derivative (a b : ℝ) (h₁ : f a b 1 = -2) (h₂ : differentiable ℝ (f a b)) :
  b = -2 ∧ a = -2 → deriv (f a b) 2 = -1 / 2 :=
by
  intros h
  cases h with hb ha
  rw [hb, ha]
  sorry

end find_derivative_l443_443860


namespace cot_thirty_equals_twice_cosine_l443_443946

theorem cot_thirty_equals_twice_cosine (h1 : Real.tan (Real.pi / 6) = 1 / Real.sqrt 3) (h2 : Real.cos (Real.pi / 6) = Real.sqrt 3 / 2) :
  Real.cot (Real.pi / 6) = Real.sqrt 3 ∧ Real.cot (Real.pi / 6) = 2 * Real.cos (Real.pi / 6) :=
by
  sorry

end cot_thirty_equals_twice_cosine_l443_443946


namespace distinct_diff_count_l443_443287

theorem distinct_diff_count :
  (∃ S : set ℕ, S = {1, 3, 5, 7, 9, 11} ∧ 
    (∑ x in S, ∑ y in S, if x > y then 1 else 0) = 10) :=
begin
  let S := {1, 3, 5, 7, 9, 11},
  use S,
  split,
  {
    refl,
  },
  {
    sorry
  }
end

end distinct_diff_count_l443_443287


namespace min_value_of_f_l443_443617

def f (x y : ℝ) : ℝ := x^2 / (y - 2) + y^2 / (x - 2)

theorem min_value_of_f (x y : ℝ) (hx : x > 2) (hy : y > 2) : 
  ∃ c, c ≥ 0 ∧ (∀ x y : ℝ, x > 2 → y > 2 → f x y ≥ c) ∧ c = 12 :=
by
  sorry

end min_value_of_f_l443_443617


namespace weight_of_new_man_l443_443494

theorem weight_of_new_man (avg_increase : ℝ) (num_oarsmen : ℕ) (old_weight : ℝ) (weight_increase : ℝ) 
  (h1 : avg_increase = 1.8) (h2 : num_oarsmen = 10) (h3 : old_weight = 53) (h4 : weight_increase = num_oarsmen * avg_increase) :
  ∃ W : ℝ, W = old_weight + weight_increase :=
by
  sorry

end weight_of_new_man_l443_443494


namespace least_possible_value_of_z_l443_443656

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end least_possible_value_of_z_l443_443656


namespace incorrect_statement_l443_443104

def consecutive_interior_angles_are_supplementary (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ + θ₂ = 180 → l1 = l2

def alternate_interior_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def corresponding_angles_are_equal (l1 l2 : ℝ) : Prop :=
  ∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2

def complementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 90

def supplementary_angles (θ₁ θ₂ : ℝ) : Prop :=
  θ₁ + θ₂ = 180

theorem incorrect_statement :
  ¬ (∀ (θ₁ θ₂ : ℝ), θ₁ = θ₂ → l1 = l2) →
    consecutive_interior_angles_are_supplementary l1 l2 →
    alternate_interior_angles_are_equal l1 l2 →
    corresponding_angles_are_equal l1 l2 →
    (∀ (θ₁ θ₂ : ℝ), supplementary_angles θ₁ θ₂) →
    (∀ (θ₁ θ₂ : ℝ), complementary_angles θ₁ θ₂) :=
sorry

end incorrect_statement_l443_443104


namespace necessary_condition_l443_443890

theorem necessary_condition (m : ℝ) (h : ∀ x : ℝ, x^2 - x + m > 0) : m > 0 := 
sorry

end necessary_condition_l443_443890


namespace calc_fraction_l443_443918

theorem calc_fraction:
  (125: ℕ) = 5 ^ 3 →
  (25: ℕ) = 5 ^ 2 →
  (25 ^ 40) / (125 ^ 20) = 5 ^ 20 :=
by
  intros h1 h2
  sorry

end calc_fraction_l443_443918


namespace determine_a_if_odd_function_l443_443694

-- Define the function f
def f (a x : ℝ) : ℝ :=
  3 * a - 2 / (3 ^ x + 1)

-- Define the condition for f to be an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- State the proof problem
theorem determine_a_if_odd_function :
  (∀ a : ℝ, (is_odd_function (f a)) → a = 1/3) :=
sorry

end determine_a_if_odd_function_l443_443694


namespace find_abc_digits_l443_443829

theorem find_abc_digits (N : ℕ) (abcd : ℕ) (a b c d : ℕ) (hN : N % 10000 = abcd) (hNsq : N^2 % 10000 = abcd)
  (ha_ne_zero : a ≠ 0) (hb_ne_six : b ≠ 6) (hc_ne_six : c ≠ 6) : (a * 100 + b * 10 + c) = 106 :=
by
  -- The proof is omitted.
  sorry

end find_abc_digits_l443_443829


namespace sequence_fifth_term_l443_443720

-- Lean definition of the sequence
def sequence : ℕ → ℕ
| 1     := 6
| n     := n * 6

theorem sequence_fifth_term :
  sequence 5 = 30 :=
by 
  unfold sequence
  simp 
  sorry

end sequence_fifth_term_l443_443720


namespace sum_of_consecutive_integers_starting_from_3_l443_443470

open Int

/-
Problem:
Given the sum of consecutive integers starting from 3, determine the maximum count of such integers that sums up to less than 500.
Proof that the correct answer is 29.
-/

theorem sum_of_consecutive_integers_starting_from_3 {n : ℕ} (h₁ : 3 + (n - 1) + 3 + (n - 2) + ... + 3 ≤ 500) : n = 29 := by
  -- Convert to sum formula based on given conditions
  have sum_formula : ∀ n : ℕ, 3 * n + (n * (n - 1) / 2) = 500
  sorry
  -- Convert the sum formula to quadratic inequality
  have quadratic_ineq : n^2 + 5 * n - 1000 = 0
  sorry
  -- Solve the quadratic equation 
  have quadr_solution : quadratic_formula 1 5 (-1000) (by linarith)
  have discriminant := by simpa using quadr_discriminant_ne_neg_real (by linarith)
  have settles : (5 + sqrt (discriminant)) / 2 = 63.39
  sorry
  -- Validate the max integer less than or equal to the equation
  have validate_n : int.floor (63.39) = 29
  sorry
  -- Hence the result
  exact 29

#eval sum_of_consecutive_integers_starting_from_3 (3 * 29 + ((29 * 28) / 2) ≤ 500)

end sum_of_consecutive_integers_starting_from_3_l443_443470


namespace oomyapeck_eyes_count_l443_443330

-- Define the various conditions
def number_of_people : ℕ := 3
def fish_per_person : ℕ := 4
def eyes_per_fish : ℕ := 2
def eyes_given_to_dog : ℕ := 2

-- Compute the total number of fish
def total_fish : ℕ := number_of_people * fish_per_person

-- Compute the total number of eyes from the total number of fish
def total_eyes : ℕ := total_fish * eyes_per_fish

-- Compute the number of eyes Oomyapeck eats
def eyes_eaten_by_oomyapeck : ℕ := total_eyes - eyes_given_to_dog

-- The proof statement
theorem oomyapeck_eyes_count : eyes_eaten_by_oomyapeck = 22 := by
  sorry

end oomyapeck_eyes_count_l443_443330


namespace even_function_l443_443480

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

def f (x : ℝ) : ℝ := (x + 2)^2 + (2 * x - 1)^2

theorem even_function : is_even_function f :=
by
  sorry

end even_function_l443_443480


namespace analytic_method_finds_sufficient_condition_l443_443161

theorem analytic_method_finds_sufficient_condition :
  (∀ (P Q : Prop), (analytic_method P Q → sufficient_condition (P → Q))) := 
begin
  sorry
end

end analytic_method_finds_sufficient_condition_l443_443161


namespace percentage_of_315_out_of_900_is_35_l443_443128

theorem percentage_of_315_out_of_900_is_35 :
  (315 : ℝ) / 900 * 100 = 35 := 
by
  sorry

end percentage_of_315_out_of_900_is_35_l443_443128


namespace number_of_correct_propositions_l443_443597

theorem number_of_correct_propositions :
  let p1 := ¬(∀ (x : ℝ), x^2 ≥ 0)
  let p2 := ∀ (x : ℝ), ¬(x ≠ 3 → x ≠ 3)
  let p3 := ¬(∀ (m : ℝ), m ≤ 1/2 → ∃ (x : ℝ), mx^2 + 2x + 2 = 0) in
  (if p1 then 1 else 0) + (if p2 then 1 else 0) + (if p3 then 1 else 0) = 1 :=
by
  let p1 := ¬(∀ (x : ℝ), x^2 ≥ 0)
  have hp1 : p1 = false :=
    by
      -- Proof of ¬(∀ (x : ℝ), x^2 ≥ 0)
      have h1 : ¬(∃ (x : ℝ), x^2 < 0) :=
        by simp [ge]
      exact h1
  let p2 := ∀ (x : ℝ), ¬(x ≠ 3 → x ≠ 3)
  have hp2 : p2 = false :=
    by
      -- x ≠ 3 ↔ x ≠ 3 is true, thus negation is false
      have h2 : ∀ (x : ℝ), (x ≠ 3 → x ≠ 3) :=
        by intro x; exact id
      simp [h2]
  let p3 := ¬(∀ (m : ℝ), m ≤ 1/2 → ∃ (x : ℝ), mx^2 + 2x + 2 = 0)
  have hp3 : p3 = true :=
    by
      -- Proof of ¬(∀ (m : ℝ), m ≤ 1/2 → mx^2 + 2x + 2 = 0 has real roots)
      have h3 : ∀ (m : ℝ), m > 1/2 → mx^2 + 2x + 2 = 0 has no real roots :=
        by
          intro m hm
          -- Discriminant calc
          have delta : 4 - 8*m < 0 := by linarith
          simp [delta]
      exact h3
  exact
    by
      simp [hp1, hp2, hp3]
      linarith

end number_of_correct_propositions_l443_443597


namespace total_length_vertex_to_centroid_l443_443773

-- Define the side length of the equilateral triangle
def side_length : ℝ := 1

-- Define a function to calculate the median of an equilateral triangle
def median (a : ℝ) : ℝ := (sqrt 3 / 2) * a

-- Define a function to calculate the length from the vertex to the centroid
def segment_length_from_vertex_to_centroid (a : ℝ) : ℝ :=
  (2 / 3) * median a

-- Define a theorem stating the total length of segments from vertices to centroid
theorem total_length_vertex_to_centroid : 
  3 * segment_length_from_vertex_to_centroid side_length = sqrt 3 := 
sorry

end total_length_vertex_to_centroid_l443_443773


namespace max_adjacent_diff_pairs_l443_443633

-- Define the grid and the problem conditions
structure Grid :=
  (rows : ℕ)
  (cols : ℕ)
  (cells : Fin (rows * cols) → Bool) -- cells indexed as in a single array for simplicity

-- Define the properties of the grid for our problem
def is_valid_grid (g : Grid) : Prop :=
  (g.rows = 100) ∧
  (g.cols = 100) ∧
  (∀ j : Fin g.cols, (Finset.univ.filter (λ i : Fin g.rows, g.cells (i * g.cols + j) = tt)).card = 50) ∧
  ((Finset.univ : Finset (Fin (g.rows))).to_list.map (λ i, 
    (Finset.univ.filter (λ j : Fin g.cols, g.cells (i * g.cols + j) = tt)).card)).nodup ∧
  (Finset.univ.sum (λ i : Fin g.rows, 
    (Finset.univ.filter (λ j : Fin g.cols, g.cells (i * g.cols + j) = tt)).card) = 5000)

-- Final proof problem statement
theorem max_adjacent_diff_pairs (g : Grid) (h : is_valid_grid g) : 
  max_adjacent_diff_pairs g = 14751 :=
sorry

end max_adjacent_diff_pairs_l443_443633


namespace eccentricity_of_hyperbola_l443_443052

noncomputable def hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) (c : ℝ)
  (hc : c^2 = a^2 + b^2) : ℝ :=
  (1 + Real.sqrt 5) / 2

theorem eccentricity_of_hyperbola (a b c e : ℝ)
  (ha : a > 0) (hb : b > 0) (h_hyperbola : c^2 = a^2 + b^2)
  (h_eccentricity : e = (1 + Real.sqrt 5) / 2) :
  e = hyperbola_eccentricity a b ha hb c h_hyperbola :=
by
  sorry

end eccentricity_of_hyperbola_l443_443052


namespace calculate_expression_l443_443561

theorem calculate_expression :
  (| (Real.sqrt 3 - 1) | - ((- (Real.sqrt 3)) ^ 2) - (12 * (- (1 / 3)))) = Real.sqrt 3 :=
by
suffices h1 : | (Real.sqrt 3 - 1) | = Real.sqrt 3 - 1, from by
suffices h2 : ((- (Real.sqrt 3)) ^ 2) = 3, from by
suffices h3 : 12 * (- (1 / 3)) = -4, from by
simp [h1, h2, h3],
sorry,
sorry,
sorry

end calculate_expression_l443_443561


namespace cdf_correct_probability_correct_l443_443608

noncomputable def p (x y : ℝ) : ℝ :=
if (0 ≤ x ∧ x ≤ π / 2 ∧ 0 ≤ y ∧ y ≤ π / 2) then 0.5 * sin (x + y) else 0

def F (x y : ℝ) : ℝ :=
0.5 * (sin(x) + sin(y) - sin(x + y))

theorem cdf_correct (x y : ℝ) :
  (0 ≤ x ∧ x ≤ π / 2 ∧ 0 ≤ y ∧ y ≤ π / 2) →
  F x y = ∫ (u : ℝ) in 0..x, ∫ (v : ℝ) in 0..y, p u v :=
sorry

theorem probability_correct :
  F (π / 6) (π / 6) = (2 - Real.sqrt 3) / 4 :=
sorry

end cdf_correct_probability_correct_l443_443608


namespace odd_function_sin_lambda_alpha_l443_443267

def f (x : ℝ) (λ α : ℝ) : ℝ :=
  if x ≥ 0 then x^2 + 2017 * x + Math.sin x
  else -x^2 + λ * x + Math.cos (x + α)

theorem odd_function_sin_lambda_alpha (λ α : ℝ) 
  (h : ∀ x : ℝ, f x λ α = -f (-x) λ α) : 
  Math.sin (λ * α) = -1 :=
by 
sorry

end odd_function_sin_lambda_alpha_l443_443267


namespace intersection_point_l443_443990

theorem intersection_point :
  let p := (50 / 17, - 13 / 17) in
  (5 * p.1 - 3 * p.2 = 17) ∧ (8 * p.1 + 2 * p.2 = 22) :=
by
  let p := (50 / 17, - 13 / 17)
  have h1 : 5 * p.1 - 3 * p.2 = 17 := sorry
  have h2 : 8 * p.1 + 2 * p.2 = 22 := sorry
  exact ⟨h1, h2⟩

end intersection_point_l443_443990


namespace arithmetic_series_sum_l443_443916

theorem arithmetic_series_sum :
  let a₁ := 9
  let d := 8
  let aₙ := 177
  let n := ((aₙ - a₁) / d) + 1
  ∃ S, S = n * (a₁ + aₙ) / 2 := 
by
  simp [a₁, d, aₙ, n]
  exact ⟨2046, sorry⟩  -- Here we skip the proof

end arithmetic_series_sum_l443_443916


namespace gambler_largest_amount_received_l443_443491

def largest_amount_received_back (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : ℕ :=
  3000 - (30 * a + 100 * b)

theorem gambler_largest_amount_received (x y a b : ℕ) (h1: 30 * x + 100 * y = 3000)
    (h2: a + b = 16) (h3: a = b + 2) : 
    largest_amount_received_back x y a b h1 h2 h3 = 2030 :=
by sorry

end gambler_largest_amount_received_l443_443491


namespace discount_price_l443_443145

theorem discount_price (original_price : ℝ) : 
  original_price > 0 →
  let sale_price := 0.67 * original_price in
  let final_price := 0.75 * sale_price in
  final_price = 0.5025 * original_price :=
by
  intros h
  let sale_price := 0.67 * original_price
  let final_price := 0.75 * sale_price
  have : final_price = 0.75 * (0.67 * original_price) := rfl
  rw [this]
  have : 0.75 * 0.67 = 0.5025 := sorry
  rw [this]
  exact rfl

end discount_price_l443_443145


namespace back_wheel_revolutions_l443_443381

noncomputable def front_wheel_radius : ℝ := 1.5
noncomputable def back_wheel_radius : ℝ := 0.5
noncomputable def front_wheel_revolutions : ℕ := 120

theorem back_wheel_revolutions :
  let circumference (r : ℝ) := 2 * Real.pi * r in
  let distance_traveled := circumference front_wheel_radius * front_wheel_revolutions in
  let back_wheel_circumference := circumference back_wheel_radius in
  let revolutions := distance_traveled / back_wheel_circumference in
  revolutions = 360 :=
by
  sorry

end back_wheel_revolutions_l443_443381


namespace eval_expression_l443_443197

theorem eval_expression : (-25 - 5 * (8 / 4)) = -35 :=
by
  have h1 : 8 / 4 = 2 := by sorry
  have h2 : 5 * 2 = 10 := by sorry
  show (-25 - 10) = -35, from sorry

end eval_expression_l443_443197


namespace hannah_games_l443_443286

theorem hannah_games (total_points : ℕ) (avg_points_per_game : ℕ) (h1 : total_points = 312) (h2 : avg_points_per_game = 13) :
  total_points / avg_points_per_game = 24 :=
sorry

end hannah_games_l443_443286


namespace find_x_for_salt_solution_l443_443490

theorem find_x_for_salt_solution : ∀ (x : ℝ),
  (1 + x) * 0.10 = (x * 0.50) →
  x = 0.25 :=
by
  intros x h
  sorry

end find_x_for_salt_solution_l443_443490


namespace sum_mod_500_l443_443188

open Function Int

noncomputable def setQ : Finset ℕ := 
  (Finset.range 200).image (λ k => (3^k % 500).natAbs)

theorem sum_mod_500 (M : ℕ) : 
  (∑ x in setQ, x) % 500 = M :=
by
  sorry

end sum_mod_500_l443_443188


namespace triangle_obtuse_l443_443530

noncomputable def is_obtuse_triangle : Prop :=
  ∃ (a b c: ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    (a / 14 = b / 10 = c / 5) ∧ 
    ((b + c > a) ∧ (a + c > b) ∧ (a + b > c)) ∧
    (let cosA := (b^2 + c^2 - a^2) / (2 * b * c) in cosA < 0)

theorem triangle_obtuse : is_obtuse_triangle := 
sorry

end triangle_obtuse_l443_443530


namespace symmetric_point_l443_443223

-- Define the given point M
def point_M : ℝ × ℝ × ℝ := (1, 0, -1)

-- Define the line in parametric form
def line (t : ℝ) : ℝ × ℝ × ℝ :=
  (3.5 + 2 * t, 1.5 + 2 * t, 0)

-- Define the symmetric point M'
def point_M' : ℝ × ℝ × ℝ := (2, -1, 1)

-- Statement: Prove that M' is the symmetric point to M with respect to the given line
theorem symmetric_point (M M' : ℝ × ℝ × ℝ) (line : ℝ → ℝ × ℝ × ℝ) :
  M = (1, 0, -1) →
  line (t) = (3.5 + 2 * t, 1.5 + 2 * t, 0) →
  M' = (2, -1, 1) :=
sorry

end symmetric_point_l443_443223


namespace product_of_repeating_decimal_l443_443571

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l443_443571


namespace max_value_of_fraction_l443_443036

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443036


namespace ratio_of_radii_l443_443420

theorem ratio_of_radii (a r R : ℝ) (h_triangle : ∀ x y z : ℝ, x^2 + y^2 + z^2 = (x + y + z)^2 / 3) 
  (h1 : r = a * sqrt 3 / 2) (h2 : R = a) : (R / r) = sqrt (7 / 3) :=
by
  sorry

end ratio_of_radii_l443_443420


namespace correct_proposition_l443_443050

axiom α : Type
axiom β : Type
axiom A : α
axiom B : α
axiom a : Set α
axiom α_set : Set α
axiom β_set : Set β

-- Define the conditions
axiom A_in_alpha : A ∈ α_set
axiom B_in_alpha : B ∈ α_set
axiom a_in_alpha : a ∈ α_set
axiom a_in_beta : a ∈ β_set
axiom alpha_inter_beta_eq_a : α_set ∩ β_set = a
axiom A_in_a : A ∈ a
axiom a_subset_alpha : a ⊆ α_set
axiom A_notin_a : A ∉ a

-- Propositions in Lean
def proposition_A := A_in_alpha ∧ B_in_alpha -> (A, B) ∈ α_set   -- Mistake: tuple instead of line AB
def proposition_B := a_in_alpha ∧ a_in_beta -> α_set ∩ β_set = {a}  -- Mistake: confusion of set a
def proposition_C := A_in_a ∧ a_subset_alpha -> A_in_alpha         -- Correct
def proposition_D := A_notin_a ∧ a_subset_alpha -> A ∉ α_set     -- Mistake: Logic doesn't always hold

-- Main statement to prove
theorem correct_proposition : proposition_C :=
sorry

end correct_proposition_l443_443050


namespace probability_two_sets_of_three_l443_443464

-- Define a predicate that checks if a hand contains two sets of three cards, each from different ranks
def two_sets_of_three (hand : Finset (Fin 52)) : Prop :=
  ∃ (r1 r2 : Fin 13), r1 ≠ r2 ∧
    (∀ c1 ∈ hand.filter (λ c, c / 4 = r1), ∃ (n1 : Fin 4), c1 = r1 * 4 + n1) ∧
    (hand.filter (λ c, c / 4 = r1)).card = 3 ∧
    (∀ c2 ∈ hand.filter (λ c, c / 4 = r2), ∃ (n2 : Fin 4), c2 = r2 * 4 + n2) ∧
    (hand.filter (λ c, c / 4 = r2)).card = 3

-- Prove that the probability of a hand containing exactly two sets of three cards from different ranks is 13/106470
theorem probability_two_sets_of_three :
  ∃ (P : ℚ), P = (13 : ℚ) / 106470 ∧
    ∀ (hand : Finset (Fin 52)), hand.card = 6 → 
    ((two_sets_of_three hand) → ∃ k ∈ (Finset.powersetLen 6 (Finset.univ : Finset (Fin 52))), P = k.card / nat.choose 52 6) :=
by
  sorry

end probability_two_sets_of_three_l443_443464


namespace PA_perpendicular_BC_l443_443164

noncomputable theory

variables {O₁ O₂ : Type*} [metric_space O₁] [metric_space O₂]
variables {A B C E F G H P : Type*} [point A] [point B] [point C]
          [point E] [point F] [point G] [point H] [point P] -- Points A, B, C, E, F, G, H, and P
variables {PA BC : line} -- Lines PA and BC
variables {EG FH : line} -- Extensions of lines EG and FH

-- Conditions
variables (EG_tangent : is_tangent E G O₁)
variables (FH_tangent : is_tangent F H O₂)
variables (EG_intersects_FH : intersects EG FH P)

-- Theorem stating PA is perpendicular to BC
theorem PA_perpendicular_BC (EG_tangent : is_tangent E G O₁)
                            (FH_tangent : is_tangent F H O₂)
                            (EG_intersects_FH : intersects EG FH P)
                            (tangent_condition : ∀ x y ∈ {E, F, G, H}, on_circle O₁ x y ↔ on_circle O₂ x y)
                            : is_perpendicular (PA P A) (BC B C) :=
begin
  sorry,
end

end PA_perpendicular_BC_l443_443164


namespace problem1_problem2_l443_443563

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end problem1_problem2_l443_443563


namespace find_number_l443_443984

theorem find_number (n : ℕ) (h1 : 9 < real.cbrt n) (h2 : real.cbrt n < 9.1) (h3 : n % 24 = 0) : n = 744 :=
sorry

end find_number_l443_443984


namespace infinite_pairs_exists_l443_443435

theorem infinite_pairs_exists (p q : ℕ) (hp : 0 < p) (hq : 0 < q) :
  ∃ (p q : ℕ), ∀ x : ℝ, (x + 1)^p * (x - 3)^q = x^(p + q) + a_1 * x^(p + q - 1) + a_2 * x^(p + q - 2) + ... + a_n ∧ a_1 = a_2 :=
by
  sorry

end infinite_pairs_exists_l443_443435


namespace distinct_terms_in_expansion_l443_443684

theorem distinct_terms_in_expansion :
  let P1 := (x + y + z)
  let P2 := (u + v + w + x + y)
  ∃ n : ℕ, n = 14 ∧ 
    ∀ a b, 
      (a ∈ {x, y, z} ∧ b ∈ {u, v, w, x, y}) → 
      (a * b ∈ expansion_of P1 P2)
:= sorry

end distinct_terms_in_expansion_l443_443684


namespace max_expression_value_l443_443009

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443009


namespace shortest_distance_phenomenon_l443_443421

def phenomenon1 : Prop :=
  ∀ (p1 p2 : Point), fix_with_two_nails (p1, p2) → ¬(shortest_distance(p1, p2))

def phenomenon2 : Prop :=
  ∀ (p1 p2 : Point), straighten_curved_road(p1, p2) → shortest_distance(p1, p2)

def phenomenon3 : Prop :=
  ∀ (p1 p2 : Point), measure_long_jump(p1, p2) → ¬(shortest_distance(p1, p2))

theorem shortest_distance_phenomenon :
  (phenomenon1 → false) ∧ (phenomenon2) ∧ (phenomenon3 → false) :=
by
  split
  { intro h, sorry },
  { intro h, sorry },
  { intro h, sorry }

end shortest_distance_phenomenon_l443_443421


namespace find_x_y_l443_443363

-- Define the sets and the complements
def universal_set (x : ℕ) : Set ℕ := {2, 3, x^2 + 2*x - 3}
def set_A := {5}
def complement_I_A (I : Set ℕ) (A : Set ℕ) (y : ℕ) : Set ℕ := {n | n ∈ I ∧ n ∉ A}

-- The main theorem statement
theorem find_x_y (x y : ℕ) :
  let I := universal_set x
  let A := set_A
  let complement_A := complement_I_A I A y
  I = {2, 3, x^2 + 2*x - 3} →
  A = {5} →
  complement_A = {2, y} →
  (x = -4 ∨ x = 2) ∧ y = 3 :=
by 
  intros;
  sorry

end find_x_y_l443_443363


namespace triangle_def_area_l443_443847

variable (D E F : Type) (d e f : ℝ)
variable (h_triangle : triangle D E F)
variable (h_isosceles : isosceles_triangle D E F d e f)
variable (h_side_lengths : sides D E F [26, 26, 48])

theorem triangle_def_area : area D E F = 240 := by
  sorry

end triangle_def_area_l443_443847


namespace differences_in_set_l443_443293

theorem differences_in_set : 
  let s := {1, 3, 5, 7, 9, 11}
  in (#{d | ∃ x y, x ∈ s ∧ y ∈ s ∧ x ≠ y ∧ d = x - y ∧ d > 0}.card) = 5 := 
by
  sorry

end differences_in_set_l443_443293


namespace find_parabola_focus_l443_443218

-- The condition and the definition of the parabola equation
def parabola_eq (x : ℝ) : ℝ :=
  4 * x^2 + 2 * x - 3

-- The definition of the focus calculation as a proof problem
theorem find_parabola_focus :
  let h := -1/4
  let k := -13/4
  let a := 4
  (h, k + 1/(4 * a)) = (-1/4 : ℝ, -51/16 : ℝ) :=
by
  -- Definitions and calculations go here; these are skipped with sorry.
  sorry

end find_parabola_focus_l443_443218


namespace systematic_sampling_method_l443_443887

theorem systematic_sampling_method :
  ∀ (num_classes num_students_per_class selected_student : ℕ),
    num_classes = 12 →
    num_students_per_class = 50 →
    selected_student = 40 →
    (∃ (start_interval: ℕ) (interval: ℕ) (total_population: ℕ), 
      total_population > 100 ∧ start_interval < interval ∧ interval * num_classes = total_population ∧
      ∀ (c : ℕ), c < num_classes → (start_interval + c * interval) % num_students_per_class = selected_student - 1) →
    "Systematic Sampling" = "Systematic Sampling" :=
by
  intros num_classes num_students_per_class selected_student h_classes h_students h_selected h_conditions
  sorry

end systematic_sampling_method_l443_443887


namespace no_valid_point_C_l443_443311

noncomputable def point (x y : ℝ) : Type := ℝ × ℝ

noncomputable def distance (P Q : point) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

noncomputable def perimeter (A B C : point) : ℝ :=
  distance A B + distance A C + distance B C

noncomputable def area (A B C : point) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem no_valid_point_C
    (A B : point) (C : point)
    (hAB : distance A B = 12)
    (h_perimeter : perimeter A B C = 60)
    (h_area : area A B C = 120) :
    ¬∃ (C : point), true :=
  sorry

end no_valid_point_C_l443_443311


namespace average_middle_three_terms_l443_443451

def arithmetic_sequence (a_1 d : ℤ) (n : ℕ) : ℤ := a_1 + (n - 1) * d

theorem average_middle_three_terms 
  (a_1 a_2 a_3 a_4 a_5 a_6 a_7 d : ℤ)
  (h_seq : ∀ n, 1 ≤ n ∧ n ≤ 7 → a_1 + (n - 1) * d = match n with
                                                      | 1 => a_1
                                                      | 2 => a_2
                                                      | 3 => a_3
                                                      | 4 => a_4
                                                      | 5 => a_5
                                                      | 6 => a_6
                                                      | 7 => a_7
                                                      | _ => 0
                                                      end)
  (h_first_three_avg : (a_1 + a_2 + a_3) / 3 = 20)
  (h_last_three_avg : (a_5 + a_6 + a_7) / 3 = 24) :
  (a_3 + a_4 + a_5) / 3 = 22 :=
by
  -- proof goes here
  sorry

end average_middle_three_terms_l443_443451


namespace reflection_matrix_squared_identity_l443_443349

open Matrix

theorem reflection_matrix_squared_identity :
  let u : Vector ℝ 2 := ![1, 3]
  let u_norm := (1 / Real.sqrt 10) • u
  let uuᵀ : Matrix (Fin 2) (Fin 2) ℝ := (vec_mul u_norm u_norm)
  let R := 2 • uuᵀ - 1
  R * R = (1 : Matrix (Fin 2) (Fin 2) ℝ) := by
  -- let u : Vector ℝ 2 := ![1, 3]
  -- let u_norm := (1 / Real.sqrt 10) • u
  -- let uuᵀ : Matrix (Fin 2) (Fin 2) ℝ := (vec_mul u_norm u_norm)
  -- let R := 2 • uuᵀ - 1
  -- show R * R = (1 : Matrix (Fin 2) (Fin 2) ℝ)
  sorry

end reflection_matrix_squared_identity_l443_443349


namespace only_setB_is_proportional_l443_443102

-- Definitions for the line segments
def setA := (3, 4, 5, 6)
def setB := (5, 15, 2, 6)
def setC := (4, 8, 3, 5)
def setD := (8, 4, 1, 3)

-- Definition to check if a set of line segments is proportional
def is_proportional (s : ℕ × ℕ × ℕ × ℕ) : Prop :=
  let (a, b, c, d) := s
  a * d = b * c

-- Theorem proving that the only proportional set is set B
theorem only_setB_is_proportional :
  is_proportional setA = false ∧
  is_proportional setB = true ∧
  is_proportional setC = false ∧
  is_proportional setD = false :=
by
  sorry

end only_setB_is_proportional_l443_443102


namespace ratio_of_x_and_y_l443_443114

theorem ratio_of_x_and_y (x y : ℝ) (h : 0.80 * x = 0.20 * y) : x / y = 1 / 4 :=
by
  sorry

end ratio_of_x_and_y_l443_443114


namespace perimeter_calculation_l443_443709

theorem perimeter_calculation (A : ℝ) (width : ℝ) (length : ℝ) (total_area : ℝ) :
  (3 * A + width * length = 130) ∧ (A = width * width) ∧ (length = 2 * width) →
  3 * width + 2 * (width + length) = 11 * Real.sqrt 26 :=
begin
  sorry
end

end perimeter_calculation_l443_443709


namespace orange_pyramid_total_l443_443516

theorem orange_pyramid_total :
  let base_length := 7
  let base_width := 9
  -- layer 1 -> dimensions (7, 9)
  -- layer 2 -> dimensions (6, 8)
  -- layer 3 -> dimensions (5, 6)
  -- layer 4 -> dimensions (4, 5)
  -- layer 5 -> dimensions (3, 3)
  -- layer 6 -> dimensions (2, 2)
  -- layer 7 -> dimensions (1, 1)
  (base_length * base_width) + ((base_length - 1) * (base_width - 1))
  + ((base_length - 2) * (base_width - 3)) + ((base_length - 3) * (base_width - 4))
  + ((base_length - 4) * (base_width - 6)) + ((base_length - 5) * (base_width - 7))
  + ((base_length - 6) * (base_width - 8)) = 175 := sorry

end orange_pyramid_total_l443_443516


namespace math_proof_problem_l443_443659

noncomputable def prop_1 (x : ℝ) : Prop := f x = 4 * cos (2 * x + π / 3)
noncomputable def center_symmetry_1 : Prop := let center := (-5 * π / 12, 0) in f center.1 = 0
def prop_3 (a b : α) (ha : ∥a + b∥ = ∥a∥ - ∥b∥) : Prop := ∃ λ : ℝ, b = λ • a
def prop_4 (a b : ℝ) (B : ℝ) (h : 0 < B ∧ B < π / 2) : Prop := 
  (let c := Real.sin B, g := 40 * Real.sin B in g < 40 * (Real.sin (π / 6)) ∧ 20 < 40)
def correct_propositions : set ℕ := {1, 3, 4}

theorem math_proof_problem :
  (prop_1 (-5 * π / 12) = 0) ∧ (prop_3 a b ha) ∧ (prop_4 40 20 (25 * π / 180) (and.intro (by linarith) (by linarith))) ↔ correct_propositions = {1, 3, 4} :=
sorry

end math_proof_problem_l443_443659


namespace max_value_of_fraction_l443_443038

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443038


namespace find_a2_l443_443830

noncomputable def sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 20) ∧
  (a 10 = 100) ∧
  (∀ n ≥ 3, a n = (∑ i in finset.range (n - 1), a (i + 1)) / (n - 1))

theorem find_a2 (a : ℕ → ℝ) (h : sequence a) : a 2 = 180 :=
by
  sorry

end find_a2_l443_443830


namespace max_min_sum_l443_443797

theorem max_min_sum (x y z : ℝ) (N n : ℝ) 
  (h : 5 * (x + y + z) = x^2 + y^2 + z^2 + 1)
  (N_def : N = max (xy + yz + zx))
  (n_def : n = min (xy + yz + zx)) : 
  N + 6 * n = 11.33 := 
sorry

end max_min_sum_l443_443797


namespace trig_relationship_l443_443630

noncomputable def a := Real.cos 1
noncomputable def b := Real.cos 2
noncomputable def c := Real.sin 2

theorem trig_relationship : c > a ∧ a > b := by
  sorry

end trig_relationship_l443_443630


namespace max_expression_value_l443_443008

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443008


namespace domain_of_function_l443_443416

theorem domain_of_function : 
  { x : ℝ | (x + 1 ≥ 0) ∧ (2 - x > 0) } = set.Ico (-1:ℝ) 2 :=
by
  sorry

end domain_of_function_l443_443416


namespace final_race_time_l443_443734

noncomputable def john_final_time (initial_times : List ℝ) (final_median : ℝ) :=
  let sorted_initial_times := initial_times.sort
  let add_to_list := sorted_initial_times ++ [12.1] -- final attempt time x to be tested
  let sorted_final_times := add_to_list.sort
  let new_median := (sorted_final_times[2] + sorted_final_times[3]) / 2
  new_median = final_median

theorem final_race_time :
  john_final_time [12.1, 12.7, 11.8, 12.3, 12.5] 12.2 = True :=
by
  sorry

end final_race_time_l443_443734


namespace sufficient_not_necessary_condition_l443_443253

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ) (h_pos : x > 0) :
  (a = 4 → x + a / x ≥ 4) ∧ (∃ b : ℝ, b ≠ 4 ∧ ∃ x : ℝ, x > 0 ∧ x + b / x ≥ 4) :=
by
  sorry

end sufficient_not_necessary_condition_l443_443253


namespace max_third_side_length_l443_443428

theorem max_third_side_length (x : ℕ) (h1 : 28 + x > 47) (h2 : 47 + x > 28) (h3 : 28 + 47 > x) :
  x = 74 :=
sorry

end max_third_side_length_l443_443428


namespace f_periodic_f_at_3_l443_443240

-- Define the function f
def f : ℝ → ℝ :=
λ x, if -1 < x ∧ x < 0 then 1 else if 0 ≤ x ∧ x < 1 then 0 else 0

theorem f_periodic :
    ∀ x : ℝ, f (x + 1) = -f x ∧ f (x + 2) = f x :=
begin
  intro x,
  split,
  -- First part: f (x + 1) = -f x
  {
    sorry  -- Proof part to be filled
  },
  -- Second part: f (x + 2) = f x
  {
    sorry  -- Proof part to be filled
  }
end

theorem f_at_3 : f 3 = 0 :=
begin
  -- Assuming the periodicity property and the given conditions
  have h₀ : f (3) = f (1),
  { apply (f_periodic 1).2 },
  have h₁ : f 1 = -f 0,
  { apply (f_periodic 0).1 },
  have h₂ : f 0 = 0,
  { unfold f, simp only [if_pos], -- This depends on the intervals defined
    linarith },
  rw [h₁, h₂],
  simp,
end

end f_periodic_f_at_3_l443_443240


namespace problem_specific_case_l443_443275

def f (x : ℝ) : ℝ := 2^x
def g (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem problem (x : ℝ) : f (g x) + g (f x) = 2 * x :=
by
  sorry

theorem specific_case : f (g 2019) + g (f 2019) = 4038 :=
by
  sorry

end problem_specific_case_l443_443275


namespace determine_all_pairs_l443_443591

noncomputable def polynomial_solution (P Q : Polynomial ℂ) : Prop :=
  P.monic ∧ Q.monic ∧ (Q^2 + 1).divides P ∧ (P^2 + 1).divides Q ∧ 
  ∃ (c : ℂ), c = 2 ∨ c = -2 ∧ (Q = P + c * Complex.I ∨ Q = P - c * Complex.I)

theorem determine_all_pairs (P Q : Polynomial ℂ) (h : polynomial_solution P Q) : 
  ∃ c : ℂ, c = 2 ∨ c = -2 ∧ (Q = P + c * Complex.I ∨ Q = P - c * Complex.I) :=
sorry

end determine_all_pairs_l443_443591


namespace min_intersection_card_l443_443999

variable (A B C : Set α)
variable [Fintype α]

-- Define |S| and n(S)
def card (S : Set α) : ℕ := Fintype.card S
def num_subsets (S : Set α) : ℕ := 2 ^ card S

-- Given conditions
variable (h1 : card A = 50)
variable (h2 : card B = 50)
variable (h3 : card C = 50)
variable (h4 : num_subsets A + num_subsets B + num_subsets C = 2 * num_subsets (A ∪ B ∪ C))

-- Question to prove
theorem min_intersection_card : card (A ∩ B ∩ C) = 51 := sorry

end min_intersection_card_l443_443999


namespace domain_of_function_l443_443417

theorem domain_of_function : 
  { x : ℝ | (x + 1 ≥ 0) ∧ (2 - x > 0) } = set.Ico (-1:ℝ) 2 :=
by
  sorry

end domain_of_function_l443_443417


namespace maximum_additional_payment_expected_value_difference_l443_443200

-- Add the conditions as definitions
def a1 : ℕ := 1298
def a2 : ℕ := 1347
def a3 : ℕ := 1337
def b1 : ℕ := 1402
def b2 : ℕ := 1310
def b3 : ℕ := 1298

-- Prices in rubles per kilowatt-hour
def peak_price : ℝ := 4.03
def night_price : ℝ := 1.01
def semi_peak_price : ℝ := 3.39

-- Actual consumptions in kilowatt-hour
def ΔP : ℝ := 104
def ΔN : ℝ := 37
def ΔSP : ℝ := 39

-- Correct payment calculated by the company
def correct_payment : ℝ := 660.72

-- Statements to prove
theorem maximum_additional_payment : 397.34 = (104 * 4.03 + 39 * 3.39 + 37 * 1.01 - 660.72) :=
by
  sorry

theorem expected_value_difference : 19.3 = ((5 * 1402 + 3 * 1347 + 1337 - 1298 - 3 * 1270 - 5 * 1214) / 15 * 8.43 - 660.72) :=
by
  sorry

end maximum_additional_payment_expected_value_difference_l443_443200


namespace remainder_sum_of_numbers_l443_443619

theorem remainder_sum_of_numbers :
  ((123450 + 123451 + 123452 + 123453 + 123454 + 123455) % 7) = 5 :=
by
  sorry

end remainder_sum_of_numbers_l443_443619


namespace basketball_team_win_requirement_l443_443130

noncomputable def basketball_win_percentage_goal (games_played_so_far games_won_so_far games_remaining win_percentage_goal : ℕ) : ℕ :=
  let total_games := games_played_so_far + games_remaining
  let required_wins := (win_percentage_goal * total_games) / 100
  required_wins - games_won_so_far

theorem basketball_team_win_requirement :
  basketball_win_percentage_goal 60 45 50 75 = 38 := 
by
  sorry

end basketball_team_win_requirement_l443_443130


namespace vector_addition_example_l443_443567

theorem vector_addition_example : 
  let v1 := (⟨-5, 3⟩ : ℝ × ℝ)
  let v2 := (⟨7, -6⟩ : ℝ × ℝ)
  v1 + v2 = (⟨2, -3⟩ : ℝ × ℝ) := 
by {
  sorry
}

end vector_addition_example_l443_443567


namespace max_expression_value_l443_443011

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443011


namespace profit_share_difference_l443_443493

theorem profit_share_difference 
  (A_capital B_capital C_capital : ℕ) 
  (B_profit : ℕ) 
  (hA : A_capital = 8000) 
  (hB : B_capital = 10000) 
  (hC : C_capital = 12000) 
  (hB_profit : B_profit = 1500) : 
  let total_profit := (B_profit * (hB / Nat.gcd hA hB hC)) in
  let A_share := A_capital / Nat.gcd hA hB hC * total_profit / (A_capital / Nat.gcd hA hB hC + B_capital / Nat.gcd hA hB hC + C_capital / Nat.gcd hA hB hC) in
  let C_share := C_capital / Nat.gcd hA hB hC * total_profit / (A_capital / Nat.gcd hA hB hC + B_capital / Nat.gcd hA hB hC + C_capital / Nat.gcd hA hB hC) in
  C_share - A_share = 600 :=
sorry

end profit_share_difference_l443_443493


namespace subset_singleton_natural_l443_443546

/-
  Problem Statement:
  Prove that the set {2} is a subset of the set of natural numbers.
-/

open Set

theorem subset_singleton_natural :
  {2} ⊆ (Set.univ : Set ℕ) :=
by
  sorry

end subset_singleton_natural_l443_443546


namespace min_numbers_on_board_l443_443380

-- Define the range of natural numbers from 1 to 50
def range_1_to_50 := {n : ℕ | 1 ≤ n ∧ n ≤ 50}

-- Use the conditions to define the predicate for Vasya's action
def can_erase (a b : ℕ) : Prop := gcd a b > 1

-- Define the minimum number of numbers that will remain on the board
def min_numbers_remaining : ℕ := 8

-- The main statement to prove
theorem min_numbers_on_board : ∃ S ⊆ range_1_to_50, 
  (∀ a ∈ S, ∀ b ∈ S, a ≠ b → ¬ can_erase a b) ∧ 
  (∀ T ⊆ range_1_to_50, (∀ a ∈ T, ∀ b ∈ T, a ≠ b → ¬ can_erase a b) → min_numbers_remaining ≤ card T) :=
begin
  sorry
end

end min_numbers_on_board_l443_443380


namespace common_factor_condition_l443_443053

def seq_a : Nat → Int 
| 0       => 1
| 1       => 3
| (n + 1) => seq_a n + seq_a (n - 1)

theorem common_factor_condition (n : Nat) (h : n ≥ 1) : 
  n % 5 = 3 ↔ Nat.gcd (n * seq_a (n + 1) + seq_a n) (n * seq_a n + seq_a (n - 1)) > 1 :=
by
  sorry

end common_factor_condition_l443_443053


namespace div_mul_fraction_eq_neg_81_over_4_l443_443557

theorem div_mul_fraction_eq_neg_81_over_4 : 
  -4 / (4 / 9) * (9 / 4) = - (81 / 4) := 
by
  sorry

end div_mul_fraction_eq_neg_81_over_4_l443_443557


namespace calc_fraction_l443_443917

theorem calc_fraction : (36 + 12) / (6 - 3) = 16 :=
by
  sorry

end calc_fraction_l443_443917


namespace nancy_flooring_area_l443_443378

def area_of_rectangle (length : ℕ) (width : ℕ) : ℕ :=
  length * width

theorem nancy_flooring_area :
  let central_area_length := 10
  let central_area_width := 10
  let hallway_length := 6
  let hallway_width := 4
  let central_area := area_of_rectangle central_area_length central_area_width
  let hallway_area := area_of_rectangle hallway_length hallway_width
  let total_area := central_area + hallway_area
  total_area = 124 :=
by
  rfl  -- This is where the proof would go.

end nancy_flooring_area_l443_443378


namespace units_digit_sum_l443_443448

theorem units_digit_sum (n : ℕ) (h : n > 0) : (35^n % 10) + (93^45 % 10) = 8 :=
by
  -- Since the units digit of 35^n is always 5 
  have h1 : 35^n % 10 = 5 := sorry
  -- Since the units digit of 93^45 is 3 (since 45 mod 4 = 1 and the pattern repeats every 4),
  have h2 : 93^45 % 10 = 3 := sorry
  -- Therefore, combining the units digits
  calc
    (35^n % 10) + (93^45 % 10)
    = 5 + 3 := by rw [h1, h2]
    _ = 8 := by norm_num

end units_digit_sum_l443_443448


namespace initial_elephants_count_l443_443852

def exodus_rate : ℕ := 2880
def exodus_time : ℕ := 4
def entrance_rate : ℕ := 1500
def entrance_time : ℕ := 7
def final_elephants : ℕ := 28980

theorem initial_elephants_count :
  final_elephants - (exodus_rate * exodus_time) + (entrance_rate * entrance_time) = 27960 := by
  sorry

end initial_elephants_count_l443_443852


namespace product_of_y_coordinates_l443_443388

theorem product_of_y_coordinates :
  (∃ y : ℝ, (4, y.from_R8_dist(-2, 5) = 13 → (5 + real.sqrt 133) * (5 - real.sqrt 133) = -108 :=
begin
  sorry
end

end product_of_y_coordinates_l443_443388


namespace fatima_donates_75_sq_inches_l443_443204

/-- Fatima starts with 100 square inches of cloth and cuts it in half twice.
    The total amount of cloth she donates should be 75 square inches. -/
theorem fatima_donates_75_sq_inches:
  ∀ (cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second: ℕ),
  cloth_initial = 100 → 
  cloth_after_first_cut = cloth_initial / 2 →
  cloth_donated_first = cloth_initial / 2 →
  cloth_after_second_cut = cloth_after_first_cut / 2 →
  cloth_donated_second = cloth_after_first_cut / 2 →
  cloth_donated_first + cloth_donated_second = 75 := 
by
  intros cloth_initial cloth_after_first_cut cloth_after_second_cut cloth_donated_first cloth_donated_second
  intros h_initial h_after_first h_donated_first h_after_second h_donated_second
  sorry

end fatima_donates_75_sq_inches_l443_443204


namespace sum_of_possible_values_l443_443896

theorem sum_of_possible_values (n : ℕ) (h1 : 7 + 11 > n) (h2 : 7 + n > 11) : 
  ∑ i in finset.Icc 5 17, i = 143 := 
sorry

end sum_of_possible_values_l443_443896


namespace adjacent_difference_arithmetic_mean_difference_greater_than_eight_specific_n_30_example_l443_443073

theorem adjacent_difference (n : ℕ) (hn : 0 < n) (x : Fin n → ℝ)
  (hsum : (Finset.univ.sum x = 0)) (hone : ∃ i, x i = 1) :
  ∃ i, |x (i + 1) % n - x i| ≥ 4 / n :=
sorry

theorem arithmetic_mean_difference (n : ℕ) (hn : 0 < n) (x : Fin n → ℝ)
  (hsum : (Finset.univ.sum x = 0)) (hone : ∃ i, x i = 1) :
  ∃ i, |(x ((i + 1) % n) + x ((i + n - 1) % n)) / 2 - x i| ≥ 8 / n^2 :=
sorry

theorem greater_than_eight (n : ℕ) (hn : 0 < n) (x : Fin n → ℝ)
  (hsum : (Finset.univ.sum x = 0)) (hone : ∃ i, x i = 1) :
  ∃ k > 8, ( ∃ i, |(x ((i + 1) % n) + x ((i + n - 1) % n)) / 2 - x i| ≥ k / n^2 ) :=
sorry

theorem specific_n_30_example (x : Fin 30 → ℝ)
  (hsum : (Finset.univ.sum x = 0)) (hone : ∃ i, x i = 1) :
  (∃ i, |(x ((i + 1) % 30) + x ((i + 29) % 30)) / 2 - x i| ≥ 2 / 113)
  ∧ (∀ i, |(x ((i + 1) % 30) + x ((i + 29) % 30)) / 2 - x i| ≤ 2 / 113) :=
sorry

end adjacent_difference_arithmetic_mean_difference_greater_than_eight_specific_n_30_example_l443_443073


namespace transformed_sum_l443_443536

open BigOperators -- Open namespace to use big operators like summation

theorem transformed_sum (n : ℕ) (x : Fin n → ℝ) (s : ℝ) 
  (h_sum : ∑ i, x i = s) : 
  ∑ i, ((3 * (x i + 10)) - 10) = 3 * s + 20 * n :=
by
  sorry

end transformed_sum_l443_443536


namespace num_int_values_in_abs_x_lt_4pi_l443_443687

theorem num_int_values_in_abs_x_lt_4pi : 
  let four_pi := 4 * Real.pi in 
  Finset.card (Finset.filter (λ x, abs x < four_pi) (Finset.range (Int.ceiling four_pi * 2 + 1))) = 25 :=
by
  let four_pi := 4 * Real.pi
  sorry

end num_int_values_in_abs_x_lt_4pi_l443_443687


namespace quadratic_reciprocal_sum_l443_443654

theorem quadratic_reciprocal_sum (x1 x2 : ℝ) 
  (h : ∀ x, x^2 + x - 5x - 6 = 0) : 
  (1 / x1 + 1 / x2 = -2 / 3) :=
sorry

end quadratic_reciprocal_sum_l443_443654


namespace number_of_odd_teams_l443_443065

-- Define the teams and their members
def team_members : List ℕ := [27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10]

-- Calculate the cumulative number for the last participant for each team
def cumulative_members (members : List ℕ) : List ℕ :=
  members.scanl (+) 0

-- Check if the last participant's number is odd
def is_odd (n : ℕ) : Bool :=
  n % 2 = 1

-- Count the number of teams where the last participant's number is odd
def count_odd_teams (members : List ℕ) : ℕ :=
  let cumulative = cumulative_members members
  cumulative.filter is_odd |>.length

theorem number_of_odd_teams :
  count_odd_teams team_members = 10 :=
by
  -- Proof part
  sorry

end number_of_odd_teams_l443_443065


namespace flour_to_add_l443_443333

theorem flour_to_add (total_flour : ℕ) (flour_added : ℕ) (h1 : total_flour = 8) (h2 : flour_added = 4) : total_flour - flour_added = 4 :=
by
  rw [h1, h2]
  exact Nat.sub_self 4 -- proves 8 - 4 = 4

end flour_to_add_l443_443333


namespace proof_same_axis_no_same_center_l443_443424

noncomputable def same_axis_no_same_center : Prop :=
  ∀ (k1 k2: ℤ),
    (∃ x: ℝ, x = k1 * (π / 2) + (π / 3) ∧ x = k2 * π + (π / 3)) ∧
    ∀ (k1 k2: ℤ),
      (∃ x: ℝ, x = (k1 * (π / 2) + (π / 12), 0) ∧
            x ≠ (k2 * π + (5 * π / 6), 0))

theorem proof_same_axis_no_same_center : same_axis_no_same_center :=
begin
  sorry
end

end proof_same_axis_no_same_center_l443_443424


namespace domain_f_when_a_is_3_max_val_of_a_if_fx_geq_2_l443_443269

def f (x : ℝ) (a : ℝ) : ℝ := log (2, abs (x + 1) + abs (x - 1) - a)

theorem domain_f_when_a_is_3 : 
  ∀ x : ℝ, (0 < log (2, abs (x + 1) + abs (x - 1) - 3)) ↔ (x < -3 / 2 ∨ x > 3 / 2) :=
by sorry

theorem max_val_of_a_if_fx_geq_2 :
  (∀ x : ℝ, f x a ≥ 2) ↔ a ≤ -2 :=
by sorry

end domain_f_when_a_is_3_max_val_of_a_if_fx_geq_2_l443_443269


namespace max_books_borrowed_l443_443868

theorem max_books_borrowed (total_students : ℕ) (students_no_books : ℕ) (students_1_book : ℕ)
  (students_2_books : ℕ) (avg_books_per_student : ℕ) (remaining_students_borrowed_at_least_3 :
  ∀ (s : ℕ), s ≥ 3) :
  total_students = 25 →
  students_no_books = 3 →
  students_1_book = 11 →
  students_2_books = 6 →
  avg_books_per_student = 2 →
  ∃ (max_books : ℕ), max_books = 15 :=
  by
  sorry

end max_books_borrowed_l443_443868


namespace line_passing_through_point_perpendicular_to_polar_axis_l443_443653

theorem line_passing_through_point_perpendicular_to_polar_axis (P : ℝ × ℝ) (hP : P = (π, π)) :
  ∃ ρ θ, ρ = -π / cos θ ∧ (∃ r, P = (r * cos θ, r * sin θ)) := sorry

end line_passing_through_point_perpendicular_to_polar_axis_l443_443653


namespace G_at_16_l443_443771

noncomputable def G (x : ℝ) : ℝ := sorry

theorem G_at_16 :
  (∀ x : ℝ, G(4 * x) / G(x + 4) = 16 - (64 * x + 80) / (x^2 + 8*x + 16)) →
  G(8) = 28 →
  G(16) = 120 :=
sorry

end G_at_16_l443_443771


namespace find_shortest_side_of_triangle_l443_443712

def Triangle (A B C : Type) := true -- Dummy definition for a triangle

structure Segments :=
(BD DE EC : ℝ)

def angle_ratios (AD AE : ℝ) (r1 r2 : ℕ) := true -- Dummy definition for angle ratios

def triangle_conditions (ABC : Type) (s : Segments) (r1 r2 : ℕ)
  (h1 : angle_ratios AD AE r1 r2)
  (h2 : s.BD = 4)
  (h3 : s.DE = 2)
  (h4 : s.EC = 5) : Prop := True

noncomputable def shortestSide (ABC : Type) (s : Segments) (r1 r2 : ℕ) : ℝ := 
  if true then sorry else 0 -- Placeholder for the shortest side length function

theorem find_shortest_side_of_triangle (ABC : Type) (s : Segments)
  (h1 : angle_ratios AD AE 2 3) (h2 : angle_ratios AE AD 1 1)
  (h3 : s.BD = 4) (h4 : s.DE = 2) (h5 : s.EC = 5) :
  shortestSide ABC s 2 3 = 30 / 11 :=
sorry

end find_shortest_side_of_triangle_l443_443712


namespace find_p_q_r_sum_l443_443180

open Real

theorem find_p_q_r_sum :
  ∃ (p q r : ℕ), (p > 0) ∧ (q > 0) ∧ (r > 0) ∧
  (∃ x : ℝ, 16 * x ^ 3 - 4 * x ^ 2 - 4 * x - 1 = 0 ∧ 
  x = (root_three p + root_three q + 1) / r) ∧
  p + q + r = 288 :=
by
  sorry

end find_p_q_r_sum_l443_443180


namespace triangle_angles_l443_443901

theorem triangle_angles
  (A B C D E P : Type)
  (angle : Type)
  (ABC_triang : triangle A B C)
  (BD_bisector : angle A B D = angle D B C)
  (E_on_AB : point E A B)
  (ACE_condition : angle A C E = (2/5 : ℝ) * angle A C B)
  (meet_at_P : meet D C P)
  (ED_DC_CP : segment E D = segment D C ∧ segment D C = segment C P) :
  angle A = 45 ∧ angle B = 60 ∧ angle C = 75 :=
sorry

end triangle_angles_l443_443901


namespace find_integer_divisible_by_24_with_cube_root_in_range_l443_443966

theorem find_integer_divisible_by_24_with_cube_root_in_range :
  ∃ (n : ℕ), (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ (24 ∣ n) ∧ n = 744 := by
    sorry

end find_integer_divisible_by_24_with_cube_root_in_range_l443_443966


namespace find_f_neg_8_l443_443649

noncomputable def log2 (x : ℝ) := Real.log x / Real.log 2

def f (x : ℝ) : ℝ :=
  if x > 0 then log2 x
  else if x < 0 then - log2 (-x)
  else 0

theorem find_f_neg_8 : f(-8) = -3 :=
by
  sorry

end find_f_neg_8_l443_443649


namespace evaluate_expression_when_x_is_3_l443_443098

theorem evaluate_expression_when_x_is_3 :
  (let x := 3 in
  (sqrt (x - 2 * sqrt 2) / sqrt (x * x - 4 * sqrt 2 * x + 8)) -
  (sqrt (x + 2 * sqrt 2) / sqrt (x * x + 4 * sqrt 2 * x + 8)) = 2) := by
  sorry

end evaluate_expression_when_x_is_3_l443_443098


namespace difference_of_squares_count_l443_443592

theorem difference_of_squares_count : (finset.filter (λ n, ∃ (a b : ℕ), n = a^2 - b^2) (finset.Icc 1 1200)).card = 900 :=
by
  sorry

end difference_of_squares_count_l443_443592


namespace distance_between_planes_l443_443216

-- Define the planes
def plane1 (x y z : ℝ) : Prop := x + 2 * y - 2 * z + 3 = 0
def plane2 (x y z : ℝ) : Prop := 2 * x + 4 * y - 4 * z + 2 = 0

-- Define a point
def point1 : ℝ × ℝ × ℝ := (-3, 0, 0)

-- Define the normalized plane2
def plane2_normalized (x y z : ℝ) : Prop := x + 2 * y - 2 * z + 1 = 0

-- Define the distance calculation between a point (x0, y0, z0) and a plane ax + by + cz + d = 0
def point_to_plane_distance (x0 y0 z0 a b c d : ℝ) : ℝ := 
  |a * x0 + b * y0 + c * z0 + d| / Real.sqrt (a^2 + b^2 + c^2)

-- Prove that the distance between plane1 and plane2 is 2/3
theorem distance_between_planes : 
  point_to_plane_distance (-3) 0 0 1 2 (-2) 1 = 2 / 3 :=
by
  sorry

end distance_between_planes_l443_443216


namespace find_f_prime_at_0_l443_443239

theorem find_f_prime_at_0 :
  ∃ f : ℝ → ℝ, (∀ x, f x = x^2 + 2 * x * f' 1) → f' 0 = -4 :=
sorry

end find_f_prime_at_0_l443_443239


namespace max_expression_value_l443_443029

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443029


namespace solution_l443_443646

noncomputable def given_equation (α : ℝ) : Prop :=
  sin (2 * α) - 2 = 2 * cos (2 * α)

noncomputable def desired_value (α : ℝ) : ℝ :=
  sin α ^ 2 + sin (2 * α)

theorem solution (α : ℝ) (h : given_equation α) : 
  (desired_value α = 1) ∨ (desired_value α = 8 / 5) :=
by
  sorry

end solution_l443_443646


namespace problem_remainder_l443_443337

def N : ℕ :=
  ∑ a_1 in Finset.range 3,
  ∑ a_2 in Finset.range (a_1 + 1),
  ∑ a_3 in Finset.range (a_2 + 1),
  -- ... Continue nested sums pattern
  ∑ a_2011 in Finset.range (a_2010 + 1),
  (List.range 2012).prod (fun n => a_n)

theorem problem_remainder : (N % 1000) = 183 := by
  sorry

end problem_remainder_l443_443337


namespace alice_password_probability_l443_443153

-- Definitions
def two_digit_count : ℕ := 100
def even_two_digit_count : ℕ := 50
def symbol_set : finset char := {'$', '%', '@', '!', '#'}
def favorable_symbols : finset char := {'$', '%', '@'}
def favorable_symbol_count : ℕ := 3
def symbol_count : ℕ := 5

-- Theorem Statement
theorem alice_password_probability : 
  (even_two_digit_count / two_digit_count) * 
  (favorable_symbol_count / symbol_count) * 
  (even_two_digit_count / two_digit_count) = 3 / 20 := by
{
  sorry
}

end alice_password_probability_l443_443153


namespace green_chips_count_l443_443071

def total_chips : ℕ := 60
def fraction_blue_chips : ℚ := 1 / 6
def num_red_chips : ℕ := 34

theorem green_chips_count :
  let num_blue_chips := total_chips * fraction_blue_chips
  let chips_not_green := num_blue_chips + num_red_chips
  let num_green_chips := total_chips - chips_not_green
  num_green_chips = 16 := by
    let num_blue_chips := total_chips * fraction_blue_chips
    let chips_not_green := num_blue_chips + num_red_chips
    let num_green_chips := total_chips - chips_not_green
    show num_green_chips = 16
    sorry

end green_chips_count_l443_443071


namespace percentage_increase_time_second_half_l443_443374

noncomputable def total_distance : ℝ := 640
noncomputable def first_half_distance : ℝ := total_distance / 2
noncomputable def first_half_speed : ℝ := 80
noncomputable def total_trip_speed : ℝ := 40

def time (distance speed : ℝ) : ℝ := distance / speed

theorem percentage_increase_time_second_half :
  let first_half_time := time first_half_distance first_half_speed in
  let total_trip_time := time total_distance total_trip_speed in
  let second_half_time := total_trip_time - first_half_time in
  let percentage_increase := ((second_half_time - first_half_time) / first_half_time) * 100 in
  percentage_increase = 200 := by
  sorry

end percentage_increase_time_second_half_l443_443374


namespace part1_part2_l443_443326

-- Definitions for the sides and the target equations
def triangleSides (a b c : ℝ) (A B C : ℝ) : Prop :=
  b * Real.sin (C / 2) ^ 2 + c * Real.sin (B / 2) ^ 2 = a / 2

-- The first part of the problem
theorem part1 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  b + c = 2 * a :=
  sorry

-- The second part of the problem
theorem part2 (a b c A B C : ℝ) (hTriangleSides : triangleSides a b c A B C) :
  A ≤ π / 3 :=
  sorry

end part1_part2_l443_443326


namespace hari_joined_after_5_months_l443_443793

theorem hari_joined_after_5_months
    (praveen_investment : ℝ) (hari_investment : ℝ) 
    (profit_ratio_num : ℝ) (profit_ratio_denom : ℝ) 
    (praveen_investment_time : ℝ) (hari_investment_time : ℝ)
    (time_period : ℝ) :
    praveen_investment = 3920 →
    hari_investment = 10080 →
    profit_ratio_num = 2 →
    profit_ratio_denom = 3 →
    praveen_investment_time = 12 →
    time_period = 12 →
    let x := time_period - (profit_ratio_num * praveen_investment_time) / (profit_ratio_denom * (hari_investment / praveen_investment)) in
    x ≈ 5 := 
by
  intros _ _ _ _ _ _ _
  sorry

end hari_joined_after_5_months_l443_443793


namespace average_age_of_mentors_l443_443813

theorem average_age_of_mentors
  (total_members : ℕ)
  (average_age_total : ℕ)
  (num_girls : ℕ)
  (num_boys : ℕ)
  (num_mentors : ℕ)
  (average_age_girls : ℕ)
  (average_age_boys : ℕ)
  (h_total_members : total_members = 50)
  (h_average_age_total : average_age_total = 20)
  (h_num_girls : num_girls = 25)
  (h_num_boys : num_boys = 20)
  (h_num_mentors : num_mentors = 5)
  (h_average_age_girls : average_age_girls = 18)
  (h_average_age_boys : average_age_boys = 19) : 
  ∑ m in set.to_finset (finset.range 5), m / 5 =
  34 := 
  sorry

end average_age_of_mentors_l443_443813


namespace disjoint_quads_possible_l443_443241

theorem disjoint_quads_possible (P : Finset (ℝ × ℝ)) (h_size : P.card = 2020) (h_collinear : ∀ (A B C : ℝ × ℝ), A ∈ P → B ∈ P → C ∈ P → A ≠ B → B ≠ C → A ≠ C → ¬ collinear A B C) : 
  ∃ (Q : Finset (Finset (ℝ × ℝ))), Q.card = 505 ∧ ∀ q ∈ Q, q.card = 4 ∧ (∀ (x ∈ q) (y ∈ q), x ≠ y) ∧ (∀ (q1 q2 ∈ Q), q1 ≠ q2 → disjoint q1 q2) :=
sorry

end disjoint_quads_possible_l443_443241


namespace remainder_3005_98_l443_443474

theorem remainder_3005_98 : 3005 % 98 = 65 :=
by sorry

end remainder_3005_98_l443_443474


namespace floor_abs_sum_eq_seven_l443_443947

theorem floor_abs_sum_eq_seven :
  (Int.floor (Real.abs 3.7) + Int.natAbs (Int.floor (-3.7))) = 7 :=
by sorry

end floor_abs_sum_eq_seven_l443_443947


namespace correct_option_C_l443_443100

variable (a : ℝ)

theorem correct_option_C : (a^2 * a = a^3) :=
by sorry

end correct_option_C_l443_443100


namespace count_non_equivalent_pile_ways_l443_443403

-- Definitions based on the conditions
variables {n m : ℕ} (hn : n ≥ 1) (hm : m ≥ 1)

-- The problem statement itself
theorem count_non_equivalent_pile_ways (hn : n ≥ 1) (hm : m ≥ 1) :
  let C := (m + n - 1).choose (n - 1) in
  C * C = (m + n - 1).choose (n - 1)^2 :=
by
  sorry

end count_non_equivalent_pile_ways_l443_443403


namespace cosine_of_angle_between_a_b_l443_443645

variables (e1 e2 : ℝ) (a b : ℝ)
variables (u v : ℝ)

-- Definitions for unit vectors e1 and e2
constants (e1_len : ℝ) (e2_len : ℝ)
constants (cos_theta : ℝ)

-- Definition of given vectors a and b
def a := 2 * e1 + e2
def b := -3 * e1 + 2 * e2

-- Conditions as per the problem
axiom e1_unit : e1_len = 1
axiom e2_unit : e2_len = 1
axiom e1_e2_angle : cos_theta = (1 / 2) -- cos(60°)

-- Proof statement
theorem cosine_of_angle_between_a_b : (a * b) / (real.sqrt (a * a) * real.sqrt (b * b)) = - (1 / 2) :=
sorry

end cosine_of_angle_between_a_b_l443_443645


namespace min_sum_is_minimum_l443_443596

noncomputable def min_sum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem min_sum_is_minimum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ∃ a b c, 0 < a ∧ 0 < b ∧ 0 < c ∧ min_sum a b c ha hb hc = 3 / real.cbrt (162 : ℝ) :=
sorry

end min_sum_is_minimum_l443_443596


namespace laptop_price_l443_443395

theorem laptop_price (cost upfront : ℝ) (upfront_percentage : ℝ) (upfront_eq : upfront = 240) (upfront_percentage_eq : upfront_percentage = 20) : 
  cost = 1200 :=
by
  sorry

end laptop_price_l443_443395


namespace find_number_l443_443979

theorem find_number (n : ℕ) (h1 : 9 < real.cbrt n) (h2 : real.cbrt n < 9.1) (h3 : n % 24 = 0) : n = 744 :=
sorry

end find_number_l443_443979


namespace difference_between_mean_and_median_l443_443715
noncomputable theory

def test_scores : List ℚ := (List.replicate 6 60) ++ (List.replicate 8 75) ++ (List.replicate 10 85) ++ (List.replicate 4 90) ++ (List.replicate 12 100)

def mean (l : List ℚ) : ℚ :=
  l.sum / l.length

def median (l : List ℚ) : ℚ :=
  let sorted := l.sort
  if h : (l.length % 2 = 0) then
    (sorted.get (l.length / 2) + sorted.get (l.length / 2 - 1)) / 2
  else
    sorted.get (l.length / 2)

theorem difference_between_mean_and_median : 
  mean test_scores - median test_scores = -0.75 := 
by
  sorry

end difference_between_mean_and_median_l443_443715


namespace parabola_equation_l443_443519

theorem parabola_equation {p : ℝ} (hp : 0 < p)
  (h_cond : ∃ A B : ℝ × ℝ, (A.1^2 = 2 * A.2 * p) ∧ (B.1^2 = 2 * B.2 * p) ∧ (A.2 = A.1 - p / 2) ∧ (B.2 = B.1 - p / 2) ∧ (|A.1 - B.1|^2 + |A.2 - B.2|^2 = 4))
  : y^2 = 2 * x := sorry

end parabola_equation_l443_443519


namespace no_even_degree_polyhedron_l443_443603

theorem no_even_degree_polyhedron :
  ¬ (∃ (P : Polyhedron), 
    (∀ (f ∈ P.faces), f ≠ P.pentagon → f.is_triangle) ∧ 
    P.pentagon.is_pentagon ∧ 
    (∀ (v ∈ P.vertices), even (v.degree))) := sorry

end no_even_degree_polyhedron_l443_443603


namespace circle_equation_through_points_l443_443994

theorem circle_equation_through_points (A B: ℝ × ℝ) (C : ℝ × ℝ)
  (hA : A = (1, -1)) (hB : B = (-1, 1)) (hC : C.1 + C.2 = 2)
  (hAC : dist A C = dist B C) :
  (x - C.1) ^ 2 + (y - C.2) ^ 2 = 4 :=
by
  sorry

end circle_equation_through_points_l443_443994


namespace num_blue_parrots_l443_443379

-- Definitions from conditions
def total_parrots : ℕ := 160
def fraction_green : ℚ := 5 / 8
def fraction_blue : ℚ := 1 - fraction_green
def total_blue_parrots : ℕ := (fraction_blue * total_parrots).toNat

-- Proof statement
theorem num_blue_parrots : total_blue_parrots = 60 := by
  sorry

end num_blue_parrots_l443_443379


namespace maximum_value_f_l443_443265

def f (x : ℝ) : ℝ := sqrt x + sqrt (6 - 2 * x)

theorem maximum_value_f : (∀ x ∈ set.Icc (0 : ℝ) 3, f x ≤ 3) ∧ (∃ x ∈ set.Icc (0 : ℝ) 3, f x = 3) :=
by
  sorry

end maximum_value_f_l443_443265


namespace find_BC_value_l443_443325

noncomputable def BC_of_triangle (A B C : Type*) [EuclideanGeometry ℝ A B C]
  (angleB : ∠B = 45) (AB AC : ℝ) (AB_value : AB = 100 * Real.sqrt 2) (AC_value : AC = 100) : ℝ :=
  let BC := dist B C
  BC

theorem find_BC_value {A B C : Type*} [EuclideanGeometry ℝ A B C] 
  (hB : ∠B = 45) (hAB : dist A B = 100 * Real.sqrt 2) (hAC : dist A C = 100) :
  BC_of_triangle A B C hB hAB hAC = 100 :=
begin
  sorry,
end

end find_BC_value_l443_443325


namespace min_quots_exists_l443_443497

noncomputable def solve_minimal_cyclic_permutation (A : ℕ) (Z : ℕ) : Prop :=
  let A_str := A.to_digits 10;
  let Z_str := Z.to_digits 10;
  A_str.length = 1001 ∧ 
  Z_str.length = 1001 ∧ 
  Z_str = A_str.rotate_left 1 ∧ 
  A > Z ∧
  A = (10^500 * 9 + 10^499 * 8 + 10^(501)-1-mod 10^499 ) -- The number with 501 nines, an eight, followed by 499 nines ensures minimal quotient strictly greater than 1

theorem min_quots_exists : 
  ∃ A Z, solve_minimal_cyclic_permutation A Z := sorry

end min_quots_exists_l443_443497


namespace correct_proposition_is_D_l443_443101

def prop_A : Prop := ∃ n, (n = 10 ∨ n = 15) ∧ n % 5 = 0
def prop_B : Prop := ∃ (x y : ℝ), (x^2 - 3*x - 4 = 0) ∧ (y^2 - 3*y - 4 = 0) ∧ ((x = -4 ∧ y = 1) ∨ (x = 4 ∧ y = -1))
def prop_C : Prop := ¬∃ x : ℝ, x^2 + 1 = 0
def prop_D : Prop := ∃ (T : Type) [triangle T], has_angle T 45 ∧ has_angle T 45 ∧ is_isosceles_right_triangle T

theorem correct_proposition_is_D : (prop_D) ∧ (∃ p q, prop_D = (p ∧ q)) := 
by 
  sorry

end correct_proposition_is_D_l443_443101


namespace isosceles_triangle_inequality_l443_443795

variable {A B C D M : Type} [metric_space A]
variables (a b c : A) (d : A) (m : A)
variables (AB AC BC BD DC DM AM : ℝ)
variables (eq_AB_AC : AB = AC)
variables (eq_BD_DC : BD = DC)

theorem isosceles_triangle_inequality (h1 : ∀ (x y : A), dist x y = dist y x)
  (h2 : dist a b = AB) (h3 : dist a c = AC) (h4 : dist b c = BC)
  (h5 : dist b d = BD) (h6 : dist d c = DC) (h7 : dist d m = DM)
  (h8 : dist a m = AM) :
  BD - DM < AB - AM := by
  sorry

end isosceles_triangle_inequality_l443_443795


namespace concentric_but_different_radius_l443_443678

noncomputable def circleF (x y : ℝ) : ℝ :=
  x^2 + y^2 - 1

def pointP (x : ℝ) : ℝ × ℝ :=
  (x, x)

def circleEquation (x y : ℝ) : Prop :=
  circleF x y = 0

def circleEquation' (x y : ℝ) : Prop :=
  circleF x y - circleF x y = 0

theorem concentric_but_different_radius (x : ℝ) (hP : circleF x x ≠ 0) (hCenter : x ≠ 0):
  ∃ r1 r2 : ℝ, r1 ≠ r2 ∧
    ∀ x y, (circleEquation x y ↔ x^2 + y^2 = 1) ∧ 
           (circleEquation' x y ↔ x^2 + y^2 = 2) :=
by
  sorry

end concentric_but_different_radius_l443_443678


namespace initial_innings_l443_443409

/-- The number of innings a player played initially given the conditions described in the problem. -/
theorem initial_innings (n : ℕ)
  (average_runs : ℕ)
  (additional_runs : ℕ)
  (new_average_increase : ℕ)
  (h1 : average_runs = 42)
  (h2 : additional_runs = 86)
  (h3 : new_average_increase = 4) :
  42 * n + 86 = 46 * (n + 1) → n = 10 :=
by
  intros h
  linarith

end initial_innings_l443_443409


namespace set_expression_l443_443944

def is_natural_number (x : ℚ) : Prop :=
  ∃ n : ℕ, x = n

theorem set_expression :
  {x : ℕ | is_natural_number (6 / (5 - x) : ℚ)} = {2, 3, 4} :=
sorry

end set_expression_l443_443944


namespace floor_abs_sum_eq_seven_l443_443948

theorem floor_abs_sum_eq_seven :
  (Int.floor (Real.abs 3.7) + Int.natAbs (Int.floor (-3.7))) = 7 :=
by sorry

end floor_abs_sum_eq_seven_l443_443948


namespace square_free_m_l443_443210

theorem square_free_m (m : ℕ) : 
  (∀ (d : ℕ), d ≤ m ∧ gcd d m ≠ 1 → ∃ (a : Fin 2020 → ℕ), 
    (∀ k, gcd (a k) m = 1) ∧ ∃ n p, m + ∑ i in finRange 2020, a i * d ^ (i + 1) = p ^ n) 
  → (∀ p k, nat.prime p → p ^ 2 ∣ m → (p ∣ m → false)) :=
begin
  sorry
end

end square_free_m_l443_443210


namespace balance_expenses_l443_443904

-- Define the basic amounts paid by Alice, Bob, and Carol
def alicePaid : ℕ := 120
def bobPaid : ℕ := 150
def carolPaid : ℕ := 210

-- The total expenditure
def totalPaid : ℕ := alicePaid + bobPaid + carolPaid

-- Each person's share of the total expenses
def eachShare : ℕ := totalPaid / 3

-- Amount Alice should give to balance the expenses
def a : ℕ := eachShare - alicePaid

-- Amount Bob should give to balance the expenses
def b : ℕ := eachShare - bobPaid

-- The statement to be proven
theorem balance_expenses : a - b = 30 :=
by
  sorry

end balance_expenses_l443_443904


namespace min_dist_l443_443754

noncomputable def P : Set (ℝ × ℝ) := { p | ∃ y, p = (y^2 / 4, y) }
def F : ℝ × ℝ := (1, 0)
def B : ℝ × ℝ := (3, 2)

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem min_dist : ∃ P ∈ P, dist P B + dist P F = 4 :=
sorry

end min_dist_l443_443754


namespace sequence_is_arithmetic_l443_443628

theorem sequence_is_arithmetic (a b c : ℝ) (h1 : 2^a = 3) (h2 : 2^b = 6) (h3 : 2^c = 12) :
  (b - a = c - b) :=
by {
  sorry
}

end sequence_is_arithmetic_l443_443628


namespace max_expression_value_l443_443030

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443030


namespace amount_invested_l443_443158

theorem amount_invested (P : ℝ) :
  P * (1.03)^2 - P = 0.08 * P + 6 → P = 314.136 := by
  sorry

end amount_invested_l443_443158


namespace existence_of_xy_iff_prime_divisor_l443_443768

theorem existence_of_xy_iff_prime_divisor (n : ℕ) (hn1 : n > 1) (hn2 : n % 2 = 1) :
  (∃ x y : ℕ, 4 / n = 1 / x + 1 / y) ↔ ∃ p : ℕ, prime p ∧ p ∣ n ∧ ∃ k : ℕ, p = 4 * k - 1 :=
sorry

end existence_of_xy_iff_prime_divisor_l443_443768


namespace green_chips_count_l443_443072

def total_chips : ℕ := 60
def fraction_blue_chips : ℚ := 1 / 6
def num_red_chips : ℕ := 34

theorem green_chips_count :
  let num_blue_chips := total_chips * fraction_blue_chips
  let chips_not_green := num_blue_chips + num_red_chips
  let num_green_chips := total_chips - chips_not_green
  num_green_chips = 16 := by
    let num_blue_chips := total_chips * fraction_blue_chips
    let chips_not_green := num_blue_chips + num_red_chips
    let num_green_chips := total_chips - chips_not_green
    show num_green_chips = 16
    sorry

end green_chips_count_l443_443072


namespace dihedral_angle_range_of_regular_polyhedron_l443_443313

theorem dihedral_angle_range_of_regular_polyhedron (n : ℕ) (h : n ≥ 3) : 
  ∃ θ : ℝ, θ ∈ (n-2) * ℝ.pi / n ∧ θ < ℝ.pi :=
sorry

end dihedral_angle_range_of_regular_polyhedron_l443_443313


namespace polynomial_identity_l443_443209

theorem polynomial_identity (P : ℝ → ℝ) :
  (∀ x, (x - 1) * P (x + 1) - (x + 2) * P x = 0) ↔ ∃ a : ℝ, ∀ x, P x = a * (x^3 - x) :=
by
  sorry

end polynomial_identity_l443_443209


namespace compound_interest_rate_l443_443157

theorem compound_interest_rate (P r : ℝ) (h1 : 17640 = P * (1 + r / 100)^8)
                                (h2 : 21168 = P * (1 + r / 100)^12) :
  4 * (r / 100) = 18.6 :=
by
  sorry

end compound_interest_rate_l443_443157


namespace different_positive_integers_as_differences_l443_443292

structure Condition where
  (S : Finset ℕ)
  (distinct_members : ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b)

noncomputable def number_of_positive_differences (c : Condition) : ℕ :=
  let differences := {d | ∃ a b : ℕ, a ∈ c.S ∧ b ∈ c.S ∧ a ≠ b ∧ d = abs (a - b)}.toFinset
  differences.card

theorem different_positive_integers_as_differences : 
  ∀ c : Condition, 
    c.S = {1, 3, 5, 7, 9, 11} →
    number_of_positive_differences c = 5 := 
by
  sorry

end different_positive_integers_as_differences_l443_443292


namespace smallest_n_trailing_zeros_diff_l443_443225

theorem smallest_n_trailing_zeros_diff :
  ∃ (n : ℕ), (∀ k : ℕ, (k ≥ n) → (count_trailing_zeros ((k+20)!) - count_trailing_zeros (k!)) = 2020) ∧ n = 5 ^ 2017 - 20 := sorry

/-- Function to count the number of trailing zeros in a factorial -/
def count_trailing_zeros (m : ℕ) : ℕ :=
  (nat.floor (m.nat_succ.log 5)) +
  (nat.floor (m.nat_succ.log 25)) +
  (nat.floor (m.nat_succ.log 125)) +
  -- Continue summing for higher powers of 5
  (nat.floor (m.nat_succ.log 625)) +
  (nat.floor (m.nat_succ.log 3125)) +
  -- Adding a few more terms for practical purposes
  (nat.floor (m.nat_succ.log (5^6))) +
  (nat.floor (m.nat_succ.log (5^7))) +
  sorry -- Continue as needed

end smallest_n_trailing_zeros_diff_l443_443225


namespace marbles_distribution_l443_443066

open BigOperators

/-- There are 52 marbles in total in five bags with no two bags containing the same number of marbles.
    Show that there exists a distribution satisfying these conditions.
    Show that in any such distribution, one bag contains exactly 12 marbles. -/
theorem marbles_distribution :
  ∃ (bags : Finset ℕ), (bags.card = 5) ∧ (∑ x in bags, x = 52) ∧ bags.pairwise (≠) ∧ ∃ x ∈ bags, x = 12 :=
begin
  sorry
end

end marbles_distribution_l443_443066


namespace probability_sum_is_3_or_6_l443_443627

theorem probability_sum_is_3_or_6 :
  let s := {1, 2, 3, 4, 5}
  let pairs := {(a, b) | a ∈ s ∧ b ∈ s ∧ a ≠ b}
  let favorable_pairs := {(1, 2), (1, 5), (2, 4)}
  (favorable_pairs.card : ℝ) / (pairs.card : ℝ) = 3 / 10 :=
by
  sorry

end probability_sum_is_3_or_6_l443_443627


namespace balloon_counts_l443_443118

theorem balloon_counts 
  (red_balloon_count : ℕ)
  (yellow_balloon_shortfall : ℕ)
  (blue_balloon_count : ℕ) 
  (h1 : red_balloon_count = 40)
  (h2 : yellow_balloon_shortfall = 3)
  (h3 : ∃ y : ℕ, y = (red_balloon_count - 1 - yellow_balloon_shortfall))
  (h4 : ∃ b : ℕ, b = (red_balloon_count + h3.some - 1 + 2)) :
  h3.some = 36 ∧ h4.some = 77 :=
by 
sorriousousousorry

end balloon_counts_l443_443118


namespace find_constants_l443_443348

variable (M : Matrix (Fin 2) (Fin 2) ℚ)
variable (a b : ℚ)

def M := !![3, 1; 0, 4]
def a := -1/12
def b := 7/12

theorem find_constants :
  M⁻¹ = a • M + b • (1 : Matrix (Fin 2) (Fin 2) ℚ) := by
  sorry

end find_constants_l443_443348


namespace equal_sum_sequence_a18_l443_443187

theorem equal_sum_sequence_a18
    (a : ℕ → ℕ)
    (h1 : a 1 = 2)
    (h2 : ∀ n, a n + a (n + 1) = 5) :
    a 18 = 3 :=
sorry

end equal_sum_sequence_a18_l443_443187


namespace peter_total_distance_l443_443383

noncomputable def total_distance : ℝ :=
  let distance_spain_russia := 7019
  let distance_spain_germany := 1615
  let distance_germany_france := 956
  let distance_france_russia := 6180
  let headwind_increase := 0.05
  let tailwind_decrease := 0.03
  let effective_distance_france_russia := distance_france_russia * (1 + headwind_increase)
  let distance_russia_germany_spain := distance_spain_russia + distance_spain_germany
  let effective_distance_russia_germany_spain := distance_russia_germany_spain * (1 - tailwind_decrease)
  effective_distance_france_russia + effective_distance_russia_germany_spain

theorem peter_total_distance (dist_esp_rus: ℝ)
                             (dist_esp_ger: ℝ)
                             (dist_ger_fra: ℝ)
                             (dist_fra_rus: ℝ)
                             (headwind: ℝ)
                             (tailwind: ℝ) :
  dist_esp_rus = 7019 →
  dist_esp_ger = 1615 →
  dist_ger_fra = 956 →
  dist_fra_rus = 6180 →
  headwind = 0.05 →
  tailwind = 0.03 →
  dist_fra_rus * (1 + headwind) + (dist_esp_rus + dist_esp_ger) * (1 - tailwind) = 14863.98 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end peter_total_distance_l443_443383


namespace domain_of_f_l443_443418

noncomputable def domain_f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem domain_of_f : domain_f = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end domain_of_f_l443_443418


namespace at_least_one_greater_than_one_l443_443757

theorem at_least_one_greater_than_one (a b : ℝ) (h : a + b > 2) : a > 1 ∨ b > 1 :=
sorry

end at_least_one_greater_than_one_l443_443757


namespace alcohol_percentage_in_combined_mixture_l443_443502

theorem alcohol_percentage_in_combined_mixture :
  let initial_solution_volume := 11
  let initial_alcohol_percentage := 0.42
  let added_water_volume := 3
  let solution_60_volume := 2
  let solution_60_alcohol_percentage := 0.60
  let solution_80_volume := 1
  let solution_80_alcohol_percentage := 0.80
  let solution_35_volume := 1.5
  let solution_35_alcohol_percentage := 0.35
  let total_initial_volume := initial_solution_volume + added_water_volume
  let total_alcohol := initial_solution_volume * initial_alcohol_percentage + 
                       solution_60_volume * solution_60_alcohol_percentage + 
                       solution_80_volume * solution_80_alcohol_percentage + 
                       solution_35_volume * solution_35_alcohol_percentage
  let final_volume := total_initial_volume + solution_60_volume + 
                      solution_80_volume + solution_35_volume
  let alcohol_percentage := (total_alcohol / final_volume) * 100
  alcohol_percentage ≈ 38.62 :=
by
  sorry

end alcohol_percentage_in_combined_mixture_l443_443502


namespace blithe_toy_count_l443_443171

-- Define the initial number of toys, the number lost, and the number found.
def initial_toys := 40
def toys_lost := 6
def toys_found := 9

-- Define the total number of toys after the changes.
def total_toys_after_changes := initial_toys - toys_lost + toys_found

-- The proof statement.
theorem blithe_toy_count : total_toys_after_changes = 43 :=
by
  -- Placeholder for the proof
  sorry

end blithe_toy_count_l443_443171


namespace quadratic_root_inequality_l443_443214

theorem quadratic_root_inequality (a : ℝ) :
  2015 < a ∧ a < 2017 ↔ 
  ∃ x₁ x₂ : ℝ, (2 * x₁^2 - 2016 * (x₁ - 2016 + a) - 1 = a^2) ∧ 
               (2 * x₂^2 - 2016 * (x₂ - 2016 + a) - 1 = a^2) ∧
               x₁ < a ∧ a < x₂ :=
sorry

end quadratic_root_inequality_l443_443214


namespace choose_rectangles_l443_443539

theorem choose_rectangles (n : ℕ) (hn : n ≥ 2) :
  ∃ (chosen_rectangles : Finset (ℕ × ℕ)), 
    (chosen_rectangles.card = 2 * n ∧
     ∀ (r1 r2 : ℕ × ℕ), r1 ∈ chosen_rectangles → r2 ∈ chosen_rectangles →
      (r1.fst ≤ r2.fst ∧ r1.snd ≤ r2.snd) ∨ 
      (r2.fst ≤ r1.fst ∧ r2.snd ≤ r1.snd) ∨ 
      (r1.fst ≤ r2.snd ∧ r1.snd ≤ r2.fst) ∨ 
      (r2.fst ≤ r1.snd ∧ r2.snd <= r1.fst)) :=
sorry

end choose_rectangles_l443_443539


namespace problem_part1_problem_part2_l443_443693

open Real

theorem problem_part1 (α : ℝ) (h : (sin (π - α) * cos (2 * π - α)) / (tan (π - α) * sin (π / 2 + α) * cos (π / 2 - α)) = 1 / 2) :
  (cos α - 2 * sin α) / (3 * cos α + sin α) = 5 := sorry

theorem problem_part2 (α : ℝ) (h : tan α = -2) :
  1 - 2 * sin α * cos α + cos α ^ 2 = 2 / 5 := sorry

end problem_part1_problem_part2_l443_443693


namespace total_detergent_used_l443_443785

-- Define the parameters of the problem
def total_pounds_of_clothes : ℝ := 9
def pounds_of_cotton : ℝ := 4
def pounds_of_woolen : ℝ := 5
def detergent_per_pound_cotton : ℝ := 2
def detergent_per_pound_woolen : ℝ := 1.5

-- Main theorem statement
theorem total_detergent_used : 
  (pounds_of_cotton * detergent_per_pound_cotton) + (pounds_of_woolen * detergent_per_pound_woolen) = 15.5 :=
by
  sorry

end total_detergent_used_l443_443785


namespace tree_planting_cost_l443_443680

variable (fence_length : ℝ) (type1_width : ℝ) (type1_cost : ℝ)
variable (type2_width : ℝ) (type2_cost : ℝ)
variable (type3_width : ℝ) (type3_cost : ℝ)
variable (total_cost : ℝ)

noncomputable def total_tree_cost (total_length : ℝ) (width1 width2 width3 : ℝ) 
  (cost1 cost2 cost3 : ℝ) : ℝ :=
let cycle_width := width1 + width2 + width3
let cycles := (total_length / cycle_width).to_nat
let cost := cycles * (cost1 + cost2 + cost3)
cost

axiom Holly_trees : 
  fence_length = 124 ∧ 
  type1_width = 2 ∧ type1_cost = 12 ∧
  type2_width = 3.5 ∧ type2_cost = 18 ∧
  type3_width = 1.5 ∧ type3_cost = 10 ∧
  total_cost = 680

theorem tree_planting_cost : total_tree_cost 124 2 3.5 1.5 12 18 10 = 680 :=
by 
  apply Holly_trees
  sorry

end tree_planting_cost_l443_443680


namespace age_of_15th_student_l443_443116

theorem age_of_15th_student (avg15: ℕ) (avg5: ℕ) (avg9: ℕ) (x: ℕ)
  (h1: avg15 = 15) (h2: avg5 = 14) (h3: avg9 = 16)
  (h4: 15 * avg15 = x + 5 * avg5 + 9 * avg9) : x = 11 :=
by
  -- Proof will be added here
  sorry

end age_of_15th_student_l443_443116


namespace find_s_l443_443587

theorem find_s (s : ℝ) (t : ℝ) (h1 : t = 4) (h2 : t = 12 * s^2 + 2 * s) : s = 0.5 ∨ s = -2 / 3 :=
by
  sorry

end find_s_l443_443587


namespace evaluate_expression_l443_443929

noncomputable def g (x : ℝ) : ℝ := x^3 + 3 * x^(1/2)

theorem evaluate_expression : 3 * g 3 - 2 * g 9 = -1395 + 9 * real.sqrt 3 :=
by 
  sorry

end evaluate_expression_l443_443929


namespace ratio_of_XT_to_TY_is_one_l443_443179

-- Define the properties of the shape described in the conditions

-- Assume a shape created from 12 unit squares
def unit_square (A : Type*) := sorry -- Placeholder for the unit square type

-- Define specific layout of the shape
def terraced_pattern_layout (shape : Type*) (unit_square : Type*) : Prop :=
  sorry -- Placeholder for the definition of the specific polygon arrangement

-- Define line bisecting horizontally
def horizontal_bisected_line (shape : Type*) (line : Type*) : Prop :=
  sorry -- Placeholder for the line \( \overline{RS} \) properties

-- Define the midpoint and endpoints of \( \overline{RS} \)
def midpoint_endpoint_ratio (line : Type*) (X T Y : Type*) (ratio : ℚ) : Prop :=
  sorry -- Placeholder for the definition involving midpoint \( T \) and ends \( X \) and \( Y \)

theorem ratio_of_XT_to_TY_is_one
  (shape : Type*)
  (unit_square : Type*)
  (line : Type*)
  (X T Y : Type*)
  (h_shape : terraced_pattern_layout shape unit_square)
  (h_bisect : horizontal_bisected_line shape line)
  (h_mid : midpoint_endpoint_ratio line X T Y 1) :
  (∃ r : ℚ, r = 1) :=
sorry -- Placeholder to prove the ratio is 1

end ratio_of_XT_to_TY_is_one_l443_443179


namespace max_value_of_fraction_l443_443035

-- Define the problem statement:
theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) 
  (hmean : (x + y + z) / 3 = 60) : ∃ x y z, (∀ x y z, (10 ≤ x ∧ x < 100) ∧ (10 ≤ y ∧ y < 100) ∧ (10 ≤ z ∧ z < 100) ∧ (x + y + z) / 3 = 60 → 
  (x + y) / z ≤ 17) ∧ ((x + y) / z = 17) :=
by
  sorry

end max_value_of_fraction_l443_443035


namespace function_has_local_minimum_in_interval_l443_443823

open Real

theorem function_has_local_minimum_in_interval {b : ℝ} :
  (0 < b ∧ b < 1) ↔ ∃ x ∈ Ioo (0 : ℝ) 1, is_local_min (λ x, x^3 - 3 * b * x + 3 * b) x :=
by
  sorry

end function_has_local_minimum_in_interval_l443_443823


namespace solution_set_quadratic_l443_443831

-- Define the quadratic equation as a function
def quadratic_eq (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- The theorem to prove
theorem solution_set_quadratic :
  {x : ℝ | quadratic_eq x = 0} = {1, 2} :=
by
  sorry

end solution_set_quadratic_l443_443831


namespace point_coordinates_l443_443791

namespace CoordinateProof

structure Point where
  x : ℝ
  y : ℝ

def isSecondQuadrant (P : Point) : Prop := P.x < 0 ∧ P.y > 0
def distToXAxis (P : Point) : ℝ := |P.y|
def distToYAxis (P : Point) : ℝ := |P.x|

theorem point_coordinates (P : Point) (h1 : isSecondQuadrant P) (h2 : distToXAxis P = 3) (h3 : distToYAxis P = 7) : P = ⟨-7, 3⟩ :=
by
  sorry

end CoordinateProof

end point_coordinates_l443_443791


namespace max_expression_value_l443_443012

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443012


namespace cos_C_in_triangle_l443_443710

noncomputable def cos_C (A B C : ℝ) (cos_A cos_B : ℝ) : ℝ :=
  let sin_A := sqrt (1 - cos_A^2)
  let sin_B := sqrt (1 - cos_B^2)
  - (cos_A * cos_B) + (sin_A * sin_B)

theorem cos_C_in_triangle (cos_A cos_B cos_C: ℝ) : 
  cos_A = (√5) / 5 → cos_B = 4 / 5 → cos_C = (2 * √5) / 25 :=
  by
    intros h1 h2
    rw [h1, h2]
    sorry

end cos_C_in_triangle_l443_443710


namespace lambda_unique_value_l443_443251

noncomputable def lambda_value (e1 e2 : Vector ℝ 2) (AB BC : Vector ℝ 2) (lambda : ℝ) : ℝ :=
  let CD : Vector ℝ 2 := lambda • e1 - e2
  let BD : Vector ℝ 2 := BC + CD
  if h : ∃ β : ℝ, AB = β • BD then lambda else 0

theorem lambda_unique_value 
  (e1 e2 : Vector ℝ 2)
  (h_non_collinear : ¬(Vector.collinear e1 e2))
  (AB BC : Vector ℝ 2)
  (h_AB : AB = 2 • e1 + e2)
  (h_BC : BC = -e1 + 3 • e2)
  (h_collinear : ∃ λ : ℝ, (2 • e1 + e2) = (1 / 2) • ((λ - 1) • e1 + 2 • e2)) :
  (2 • e1 + e2 = (1 / 2) • ((5 - 1) • e1 + 2 • e2)) :=
by
  sorry

end lambda_unique_value_l443_443251


namespace different_positive_integers_as_differences_l443_443291

structure Condition where
  (S : Finset ℕ)
  (distinct_members : ∀ a b : ℕ, a ∈ S → b ∈ S → a ≠ b)

noncomputable def number_of_positive_differences (c : Condition) : ℕ :=
  let differences := {d | ∃ a b : ℕ, a ∈ c.S ∧ b ∈ c.S ∧ a ≠ b ∧ d = abs (a - b)}.toFinset
  differences.card

theorem different_positive_integers_as_differences : 
  ∀ c : Condition, 
    c.S = {1, 3, 5, 7, 9, 11} →
    number_of_positive_differences c = 5 := 
by
  sorry

end different_positive_integers_as_differences_l443_443291


namespace permutations_6_participants_4_positions_l443_443154

/-- There are 6 participants in a race and no ties are allowed.
We need to calculate the number of different outcomes for the first four positions.
Let P(n, k) be the number of permutations of n items taken k at a time, using the formula:
P(n, k) = n! / (n - k)!
For n = 6 and k = 4, the number of different 1st-2nd-3rd-4th place outcomes is equal to 360. -/
theorem permutations_6_participants_4_positions : 
  Nat.choose (6, 4) * Nat.factorial 4 = 360 :=
by
  sorry

end permutations_6_participants_4_positions_l443_443154


namespace calc1_calc2_calc3_calc4_l443_443923

theorem calc1 : 23 + (-16) - (-7) = 14 :=
by
  sorry

theorem calc2 : (3/4 - 7/8 - 5/12) * (-24) = 13 :=
by
  sorry

theorem calc3 : ((7/4 - 7/8 - 7/12) / (-7/8)) + ((-7/8) / (7/4 - 7/8 - 7/12)) = -10/3 :=
by
  sorry

theorem calc4 : -1^4 - (1 - 0.5) * (1/3) * (2 - (-3)^2) = 1/6 :=
by
  sorry

end calc1_calc2_calc3_calc4_l443_443923


namespace abscissa_of_A_is_5_l443_443726

theorem abscissa_of_A_is_5
  (A : ℝ × ℝ) (B : ℝ × ℝ)
  (hA : A.1 = A.2 ∧ A.1 > 0)
  (hB : B = (5, 0))
  (C : ℝ × ℝ) (D : ℝ × ℝ)
  (hC : C = ((A.1 + 5) / 2, A.2 / 2))
  (hD : D = (5 / 2, 5 / 2))
  (dot_product_eq : (B.1 - A.1, B.2 - A.2) • (D.1 - C.1, D.2 - C.2) = 0) :
  A.1 = 5 :=
sorry

end abscissa_of_A_is_5_l443_443726


namespace adventure_books_count_l443_443914

theorem adventure_books_count :
  ∀ (a m t : ℕ), m = 17 ∧ t = 30 → a + m = t → a = 13 := 
by
  intros a m t  h₁ h₂
  cases h₁ with h₁m h₁t
  rw [h₁m, h₁t] at h₂
  linarith

end adventure_books_count_l443_443914


namespace right_triangle_ineq_l443_443719

variable (a b c : ℝ)
variable (h : c^2 = a^2 + b^2)

theorem right_triangle_ineq (a b c : ℝ) (h : c^2 = a^2 + b^2) : (a^3 + b^3 + c^3) / (a * b * (a + b + c)) ≥ Real.sqrt 2 :=
by
  sorry

end right_triangle_ineq_l443_443719


namespace find_V_l443_443873

theorem find_V 
  (c : ℝ)
  (R₁ V₁ W₁ R₂ W₂ V₂ : ℝ)
  (h1 : R₁ = c * (V₁ / W₁))
  (h2 : R₁ = 6)
  (h3 : V₁ = 2)
  (h4 : W₁ = 3)
  (h5 : R₂ = 25)
  (h6 : W₂ = 5)
  (h7 : V₂ = R₂ * W₂ / 9) :
  V₂ = 125 / 9 :=
by sorry

end find_V_l443_443873


namespace negation_of_all_cats_not_pets_l443_443058

variables (Cat Pet : Type)
variables (is_cat : Cat → Prop) (is_pet : Pet → Prop)

theorem negation_of_all_cats_not_pets :
  (∀ x, is_cat x → ¬ is_pet x) → (∃ x, is_cat x ∧ is_pet x) :=
sorry

end negation_of_all_cats_not_pets_l443_443058


namespace negation_of_p_l443_443277

theorem negation_of_p :
  (∃ x : ℝ, x < 0 ∧ x + (1 / x) > -2) ↔ ¬ (∀ x : ℝ, x < 0 → x + (1 / x) ≤ -2) :=
by {
  sorry
}

end negation_of_p_l443_443277


namespace third_height_of_triangle_l443_443439

theorem third_height_of_triangle 
  (a b c ha hb hc : ℝ)
  (h_abc_diff : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_heights : ∃ (h1 h2 h3 : ℕ), h1 = 3 ∧ h2 = 10 ∧ h3 ≠ h1 ∧ h3 ≠ h2) :
  ∃ (h3 : ℕ), h3 = 4 :=
by
  sorry

end third_height_of_triangle_l443_443439


namespace least_possible_value_of_z_l443_443655

theorem least_possible_value_of_z (x y z : ℤ) 
  (hx : Even x) 
  (hy : Odd y) 
  (hz : Odd z) 
  (h1 : y - x > 5) 
  (h2 : z - x = 9) : 
  z = 11 := 
by
  sorry

end least_possible_value_of_z_l443_443655


namespace angle_MNP_is_45_l443_443122

-- Define the geometric context and notation
def right_triangle (a b c : ℝ) : Prop :=
a^2 + b^2 = c^2

def midpoint (a b c : ℝ) : Prop :=
2 * c = a + b

-- Define the main problem components
variable {M N P Q O : ℝ}
variable h1 : right_triangle M N P
variable h2 : Q = (N + P) / 2
variable h3 : MQ = QO
variable h4 : midpoint N P O

-- State the theorem to be proven
theorem angle_MNP_is_45 :
angle M N P = 45 := 
by
sor purchasesdatch


end angle_MNP_is_45_l443_443122


namespace product_of_repeating_decimal_l443_443575

theorem product_of_repeating_decimal :
  let s := (456 : ℚ) / 999 in
  7 * s = 1064 / 333 :=
by
  let s := (456 : ℚ) / 999
  sorry

end product_of_repeating_decimal_l443_443575


namespace complex_polar_form_l443_443558

noncomputable def complexSumReTheta : ℂ :=
  5 * exp (real.pi * complex.I / 12) + 5 * exp (13 * real.pi * complex.I / 24)

theorem complex_polar_form :
  ∃ r θ : ℝ, complexSumReTheta = r * exp (θ * complex.I)
    ∧ r = 10 * real.cos (11 * real.pi / 48)
    ∧ θ = 5 * real.pi / 16 :=
by
  sorry

end complex_polar_form_l443_443558


namespace range_of_z_l443_443769

theorem range_of_z (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
(hxy : x + y = x * y) (hxyz : x + y + z = x * y * z) : 1 < z ∧ z ≤ sqrt 3 :=
by
  sorry

end range_of_z_l443_443769


namespace find_positive_integer_l443_443950

theorem find_positive_integer : ∃ (n : ℤ), n > 0 ∧ (24 : ℤ) ∣ n ∧ (9 : ℝ) < (n : ℝ).cbrt ∧ (n : ℝ).cbrt < 9.1 ∧ n = 744 := by
  sorry

end find_positive_integer_l443_443950


namespace bucket_weight_proof_l443_443132

variables (p q x y : ℝ)

def bucket_full_weight : ℝ :=
  q + 8/5 * (p - q)

theorem bucket_weight_proof
  (h1 : x + 3/4 * y = p)
  (h2 : x + 1/3 * y = q) :
  x + y = bucket_full_weight p q :=
sorry

end bucket_weight_proof_l443_443132


namespace distinct_terms_in_expansion_l443_443685

theorem distinct_terms_in_expansion:
  (∀ (x y z u v w: ℝ), (x + y + z) * (u + v + w + x + y) = 0 → false) →
  3 * 5 = 15 := by sorry

end distinct_terms_in_expansion_l443_443685


namespace polynomial_decomposition_l443_443359

noncomputable def s (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 1
noncomputable def t (x : ℝ) : ℝ := x + 18

def g (x : ℝ) : ℝ := 3 * x^4 + 9 * x^3 - 7 * x^2 + 2 * x + 6
def e (x : ℝ) : ℝ := x^2 + 2 * x - 3

theorem polynomial_decomposition : s 1 + t (-1) = 27 :=
by
  sorry

end polynomial_decomposition_l443_443359


namespace max_expression_value_l443_443031

noncomputable def max_value : ℕ := 17

theorem max_expression_value 
  (x y z : ℕ) 
  (hx : 10 ≤ x ∧ x < 100) 
  (hy : 10 ≤ y ∧ y < 100) 
  (hz : 10 ≤ z ∧ z < 100) 
  (mean_eq : (x + y + z) / 3 = 60) : 
  (x + y) / z ≤ max_value :=
sorry

end max_expression_value_l443_443031


namespace base8_to_base10_conversion_l443_443131

-- Variables and definitions from the problem
def num_base8: ℕ := 543
def expected_base10: ℕ := 355

-- Definitions of converting base-8 to base-10
def convert_base8_to_base10 (n: ℕ): ℕ :=
  let d0 := (n % 10) in
  let d1 := (n / 10 % 10) in
  let d2 := (n / 100 % 10) in
  d0 * (8^0) + d1 * (8^1) + d2 * (8^2)

-- The theorem to prove that the conversion of 543_8 equals 355 in base-10
theorem base8_to_base10_conversion: convert_base8_to_base10 num_base8 = expected_base10 := 
  sorry

end base8_to_base10_conversion_l443_443131


namespace correct_statement_l443_443103

theorem correct_statement
  (H1 : ∀ n > 0, n = 1 → true)
  (H2 : ∀ x, -x < x)
  (H3 : ∀ x, |x| = x → x ≥ 0)
  (H4 : ∀ x, |x| > 0)
  : "The smallest positive integer is 1" := by
  sorry

end correct_statement_l443_443103


namespace CB1_perpendicular_plane_ABC1_MN_parallel_plane_ABC1_l443_443126

variables (A B C A1 B1 C1 M N : Point)
variable {p : Plane}
variables (right_prism : prism ABC A1 B1 C1)
variables (angle_ABC_90 : ∠A B C = 90)
variables (equal_edges : dist B C = dist C C1)
variables (M_midpoint : midpoint B B1 M)
variables (N_midpoint : midpoint A1 C1 N)

/-- (1) CB1 is perpendicular to plane ABC1 -/
theorem CB1_perpendicular_plane_ABC1 (CB1 : Line) :
  CB1 ⊥ p :=
sorry

/-- (2) MN is parallel to plane ABC1 -/
theorem MN_parallel_plane_ABC1 (MN : Line) :
  MN ∥ p :=
sorry

end CB1_perpendicular_plane_ABC1_MN_parallel_plane_ABC1_l443_443126


namespace problem1_problem2_problem3_problem4_l443_443921

theorem problem1 : 23 + (-16) - (-7) = 14 := by
  sorry

theorem problem2 : (3/4 - 7/8 - 5/12) * (-24) = 13 := by
  sorry

theorem problem3 : (7/4 - 7/8 - 7/12) / (-7/8) + (-7/8) / (7/4 - 7/8 - 7/12) = -(10/3) := by
  sorry

theorem problem4 : -1 ^ 4 - (1 - 0.5) * (1/3) * (2 - (-3) ^ 2) = 1/6 := by 
  sorry

end problem1_problem2_problem3_problem4_l443_443921


namespace find_number_l443_443980

theorem find_number (n : ℕ) (h1 : 9 < real.cbrt n) (h2 : real.cbrt n < 9.1) (h3 : n % 24 = 0) : n = 744 :=
sorry

end find_number_l443_443980


namespace perpendicular_line_eq_l443_443092

theorem perpendicular_line_eq (x y : ℝ) :
  (∃ (p : ℝ × ℝ), p = (-2, 3) ∧ 
    ∀ y₀ x₀, 3 * x - y = 6 ∧ y₀ = 3 ∧ x₀ = -2 → y = -1 / 3 * x + 7 / 3) :=
sorry

end perpendicular_line_eq_l443_443092


namespace croissants_left_l443_443367

theorem croissants_left (total_croissants : ℕ) (neighbors : ℕ) (left_croissants : ℕ) : 
  total_croissants = 59 →
  neighbors = 8 →
  left_croissants = total_croissants % neighbors →
  left_croissants = 3 :=
begin
  intros h1 h2 h3,
  rw [h1, h2] at h3,
  exact h3,
end

end croissants_left_l443_443367


namespace range_of_k_l443_443664

noncomputable def f (k x : ℝ) : ℝ :=
  (4^x - k * 2^(x + 1) + 1) / (4^x + 2^x + 1)

theorem range_of_k (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, let a := f k x1; let b := f k x2; let c := f k x3 in
    a + b > c ∧ a + c > b ∧ b + c > a ) ↔ k ∈ Set.Icc (-2 : ℝ) (1/4 : ℝ) :=
sorry

end range_of_k_l443_443664


namespace a_101_is_100_l443_443735

def sequence (a : ℕ → ℕ) : Prop := 
a 1 = 2 ∧ ∀ n, a (2 * n + 1) = 2 * a n + 1

theorem a_101_is_100 (a : ℕ → ℕ) (h : sequence a) : 
  a 101 = 100 := 
sorry

end a_101_is_100_l443_443735


namespace find_t_l443_443676

-- Define vectors a and b
def a := (3 : ℝ, 4 : ℝ)
def b := (1 : ℝ, 0 : ℝ)

-- Define the vector c as a function of t
def c (t : ℝ) := (a.1 + t * b.1, a.2 + t * b.2)

-- Statement of the theorem to be proven
theorem find_t (t : ℝ) :
  (a.1 * (a.1 + t * b.1) + a.2 * (a.2 + t * b.2)) = (b.1 * (a.1 + t * b.1) + b.2 * (a.2 + t * b.2)) →
  t = 5 :=
by
  sorry

end find_t_l443_443676


namespace product_of_repeating_decimal_l443_443569

theorem product_of_repeating_decimal (x : ℚ) (h : x = 456 / 999) : 7 * x = 355 / 111 :=
by
  sorry

end product_of_repeating_decimal_l443_443569


namespace max_expression_value_l443_443013

theorem max_expression_value (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (hmean : (x + y + z) / 3 = 60) :
  (x + y) / z ≤ 17 :=
sorry

end max_expression_value_l443_443013


namespace find_t_l443_443674

variable (t : ℝ)

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (1, 0)
def c (t : ℝ) : ℝ × ℝ := (3 + t, 4)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

theorem find_t (h : dot_product (a) (c t) = dot_product (b) (c t)) : t = 5 := 
by 
  sorry

end find_t_l443_443674


namespace seating_arrangements_l443_443723

theorem seating_arrangements {n k : ℕ} (h1 : n = 8) (h2 : k = 6) :
  ∃ c : ℕ, c = (n - 1) * Nat.factorial k ∧ c = 20160 :=
by
  sorry

end seating_arrangements_l443_443723


namespace max_marks_l443_443112

theorem max_marks (T : ℝ) (h : 0.33 * T = 165) : T = 500 := 
by {
  sorry
}

end max_marks_l443_443112


namespace right_triangle_BGF_l443_443316

open EuclideanGeometry
open Triangle

-- Define triangle ABC as acute, with specific geometric properties and relationships.
variable {A B C D E F G H : Point}
variable {ABC : acute_triangle A B C}
variable {angle_bisector_AD : angle_bisector A D B C}
variable {point_D_on_BC : lies_on D B C}
variable {perpendicular_DE_AC : perpendicular D E A C}
variable {perpendicular_DF_AB : perpendicular D F A B}
variable {intersection_BE_CF_at_H : intersects BE CF H}
variable {G_AFH_circumcircle_intersect_BE : AFH_circumcircle.intersects BE G}

-- Conclusion
theorem right_triangle_BGF :
    right_triangle B G F :=
sorry

end right_triangle_BGF_l443_443316


namespace best_statistical_chart_for_comparisons_changes_math_scores_studentsA_B_l443_443077

def students (A B : Type) := true
def compared_changes (X : Type) := true

theorem best_statistical_chart_for_comparisons_changes_math_scores_studentsA_B 
  (A B : Type) (X : Type) 
  (h1 : students A B) (h2 : compared_changes X) : 
  X = "compound line chart" :=
sorry

end best_statistical_chart_for_comparisons_changes_math_scores_studentsA_B_l443_443077


namespace max_tile_difference_l443_443878

theorem max_tile_difference :
  ∀ (tiles : ℕ) (dim : ℕ), tiles = 95 → dim = 10 →
  (∃ (black white : ℕ), black - white = 77 ∧
  (∀ r : fin dim, ∃ c : fin dim, black > grid(r, c)) ∧ 
  (∀ c : fin dim, ∃ r : fin dim, white > grid(r, c))) :=
begin
  sorry

end max_tile_difference_l443_443878


namespace log_base_4_of_32_l443_443940

theorem log_base_4_of_32 : log 4 32 = 5 / 2 :=
by sorry

end log_base_4_of_32_l443_443940


namespace solve_quadratic_l443_443806

theorem solve_quadratic : 
  ∀ x : ℝ, x^2 + 6 * x + 5 = 0 ↔ (x = -1 ∨ x = -5) :=
by
  intro x
  split
  sorry

end solve_quadratic_l443_443806


namespace sum_of_even_indexed_angles_l443_443836

-- Given definitions and conditions
def z_is_complex_and_satisfies_equation (z : ℂ) : Prop := (z^24 - z^12 - 1 = 0) ∧ (abs z = 1)

def complex_to_angle (z : ℂ) : ℝ := 
  if imag_part z > 0 then real.atan2 (complex.imag z) (complex.re z) * (180 / real.pi) 
  else real.atan2 (complex.imag z) (complex.re z) * (180 / real.pi) + 360

-- Necessary fact: angle is in range
def angles_in_order (zs : list ℂ) : Prop :=
  ∀ i j, i < j → (0 ≤ complex_to_angle (zs.nth_le i sorry)) ∧ 
                 (complex_to_angle (zs.nth_le i sorry) < complex_to_angle (zs.nth_le j sorry))∧ 
                 (complex_to_angle (zs.nth_le j sorry) < 360)

-- Main theorem statement
theorem sum_of_even_indexed_angles (zs : list ℂ) (h : list.pairwise (λ z1 z2, (abs z1 - abs z2) = 0) zs) 
  (hz_conds : ∀ z, z ∈ zs → z_is_complex_and_satisfies_equation z) 
  (h_len : zs.length = 6) 
  (h_angle_order : angles_in_order zs) :
  (list.sum (list.map (complex_to_angle ∘ (ls.take_even_indexes)) zs)) = 480 := sorry

def ls.take_even_indexes {α : Type} [inhabited α] (l : list α) : list α := 
list.filter (λ x, ((list.index_of x l) % 2 = 1)) l

end sum_of_even_indexed_angles_l443_443836


namespace exists_zeros_in_intervals_l443_443774

noncomputable def f (x : ℝ) : ℝ := Real.log x - x^2 + 1

theorem exists_zeros_in_intervals : 
  (∃ x ∈ Ioo 0 1, f x = 0) ∧ (∃ x ∈ Ioo 1 2, f x = 0) :=
by
  sorry

end exists_zeros_in_intervals_l443_443774


namespace nylon_cord_length_l443_443110

theorem nylon_cord_length (π : ℝ) (hπ : π ≈ 3.14) : 
  ∃ q : ℝ, q ≈ 9.55 ∧ ∃ r : ℝ, r * π = 30 ∧ q = r :=
by
  sorry

end nylon_cord_length_l443_443110


namespace cylinder_lateral_surface_area_l443_443886

theorem cylinder_lateral_surface_area 
  (r h : ℝ) 
  (radius_eq : r = 2) 
  (height_eq : h = 5) : 
  2 * Real.pi * r * h = 62.8 :=
by
  -- Proof steps go here
  sorry

end cylinder_lateral_surface_area_l443_443886


namespace wednesday_temp_proof_l443_443079

def monday_temp (M : ℝ) : Prop := true
def tuesday_temp (M : ℝ) : ℝ := M + 4
def wednesday_temp (M : ℝ) : ℝ := M - 6
def tuesday_actual_temp : ℝ := 22

theorem wednesday_temp_proof (M : ℝ) (h_tuesday : tuesday_temp M = tuesday_actual_temp) :
  wednesday_temp M = 12 :=
by 
  unfold tuesday_temp at h_tuesday
  have h : M = 18 := by linarith
  unfold wednesday_temp
  rw h
  norm_num

end wednesday_temp_proof_l443_443079


namespace bounded_function_solution_l443_443611

def bounded_func (f : ℝ → ℝ) :=
  ∃ M > 0, ∀ x, |f x| ≤ M

theorem bounded_function_solution (f : ℝ → ℝ) (h_bounded : bounded_func f) :
  (∀ x y : ℝ, f(x * f(y)) + y * f(x) = x * f(y) + f(x * y)) →
  (∀ x : ℝ, (x ≥ 0 → f(x) = 0) ∧ (x < 0 → f(x) = -2 * x)) :=
begin
  sorry
end

end bounded_function_solution_l443_443611


namespace zero_one_law_for_gaussian_systems_l443_443767

def is_gaussian_random_sequence (X : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, is_gaussian (X n)

def linear_subspace_of_R_infty (L : Set (ℕ → ℝ)) : Prop :=
  ∀ f1 f2 ∈ L, ∀ (a b : ℝ), (λ n, a * f1 n + b * f2 n) ∈ L ∧ (λ n, f1 n - f2 n) ∈ L

theorem zero_one_law_for_gaussian_systems {X : ℕ → ℝ} {L : Set (ℕ → ℝ)} 
  (hX : is_gaussian_random_sequence X) 
  (hL : linear_subspace_of_R_infty L) : 
  (Prob (X ∈ L) = 0 ∨ Prob (X ∈ L) = 1) 
  ∧ (Prob (λ ω, ∃ n, |X ω n| < ∞) = 0 ∨ Prob (λ ω, ∃ n, |X ω n| < ∞) = 1) := 
sorry

end zero_one_law_for_gaussian_systems_l443_443767


namespace part1_part2_l443_443262

def f (x a b : ℝ) := -3 * x^2 + a * (5 - a) * x + b

theorem part1 (a b : ℝ) (h1: ∀ x : ℝ, x ∈ Ioo (-1) 3 → f x a b > 0) : 
  (a = 2 ∧ b = 9) ∨ (a = 3 ∧ b = 9) :=
sorry

theorem part2 (b : ℝ) (h2: f 1 (a : ℝ) b < 0) :
  (b < -13 / 4 → ∀ a : ℝ, True)
  ∧ (b = -13 / 4 → ∀ (a : ℝ), a ≠ 5 / 2)
  ∧ (b > -13 / 4 → ∀ (a : ℝ), a > (5 + Real.sqrt (4 * b + 13)) / 2 ∨ a < (5 - Real.sqrt (4 * b + 13)) / 2) :=
sorry

end part1_part2_l443_443262


namespace second_derivative_l443_443764

variable (x : ℝ)
noncomputable def y : ℝ := (1 - x^2) / (Real.sin x)

theorem second_derivative (h₁ : y = (1 - x^2) / (Real.sin x)) :
  (deriv (deriv y)) x = (-2 * x * Real.sin x - (1 - x^2) * Real.cos x) / (Real.sin x)^2 := sorry

end second_derivative_l443_443764


namespace sequence_sum_l443_443279

theorem sequence_sum (n : ℕ) : 
  (\sum k in finset.range n, 1 / ((2 * k + 1) * (2 * k + 3))) = n / (2 * n + 1) :=
by
  sorry

end sequence_sum_l443_443279


namespace difference_between_two_numbers_l443_443441

theorem difference_between_two_numbers :
  ∃ a b : ℕ, 
    a + 5 * b = 23405 ∧ 
    (∃ b' : ℕ, b = 10 * b' + 5 ∧ b' = 5 * a) ∧ 
    5 * b - a = 21600 :=
by {
  sorry
}

end difference_between_two_numbers_l443_443441


namespace sum_of_quotient_and_remainder_l443_443477

noncomputable def some_number := 7 * 13 + 1

theorem sum_of_quotient_and_remainder :
  let N := some_number in
  (N / 8) + (N % 8) = 15 :=
by
  let N := some_number
  have H : N = 7 * 13 + 1 := rfl
  have N_val: N = 92 := by rw [H]; norm_num
  have quotient: N / 8 = 11 := by rw [N_val]; norm_num
  have remainder: N % 8 = 4 := by rw [N_val]; norm_num
  rw [quotient, remainder]
  norm_num

end sum_of_quotient_and_remainder_l443_443477


namespace maximum_value_is_17_l443_443021

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443021


namespace no_real_solution_l443_443601

theorem no_real_solution (n : ℝ) : (∀ x : ℝ, (x+6)*(x-3) = n + 4*x → false) ↔ n < -73/4 := by
  sorry

end no_real_solution_l443_443601


namespace number_of_polynomials_l443_443586

def polynomial (coeffs : List ℤ) (x : ℝ) : ℝ :=
  coeffs.enum.map (λ ⟨i, a_i⟩, a_i * (x ^ i)).sum

def num_such_polynomials : ℕ :=
  10

theorem number_of_polynomials :
  ∃ (p : List ℤ) (n : ℕ),
    p.length = n + 1 ∧
    (n + p.map Int.natAbs).sum = 4 ∧
    num_such_polynomials = 10 :=
by
  sorry

end number_of_polynomials_l443_443586


namespace area_of_triangle_ABC_l443_443737

theorem area_of_triangle_ABC {a b c : ℝ} (b_eq : b = 1) (c_eq : c = sqrt 3) (C_eq : ∠C = (2 * π) / 3) :
  ∃ (S : ℝ), S = (sqrt 3) / 4 :=
by
  sorry

end area_of_triangle_ABC_l443_443737


namespace find_third_angle_l443_443820

variable (A B C : ℝ)

theorem find_third_angle
  (hA : A = 32)
  (hB : B = 3 * A)
  (hC : C = 2 * A - 12) :
  C = 52 := by
  sorry

end find_third_angle_l443_443820


namespace hcf_of_two_numbers_l443_443426

-- Define the integers A, B and the condition that A is the larger number
def A := 345
def B : ℕ

-- Define the function gcd and lcm
def gcd (a b : ℕ) : ℕ := if a = 0 then b else if b = 0 then a else Nat.gcd a b
def lcm (a b : ℕ) : ℕ := a * b / gcd a b

-- State the condition that the lcm of A and B includes 14 and 15 as factors
def lcm_contains_factors (a b x y : ℕ) : Prop := ∃ H : ℕ, lcm a b = H * x * y

-- Define the conditions given in the problem
axiom hcf_lcm_conditions : lcm_contains_factors A B 14 15

-- State the theorem to prove that the hcf of the two numbers is 5
theorem hcf_of_two_numbers : gcd A B = 5 :=
by sorry

end hcf_of_two_numbers_l443_443426


namespace slope_angle_of_line_l443_443704

theorem slope_angle_of_line (A B : ℝ × ℝ) (hA : A = (0, real.sqrt 3)) (hB : B = (2, 3 * real.sqrt 3)) :
  ∃ α : ℝ, α = real.pi / 3 := 
by
  -- the proof will go here
  sorry

end slope_angle_of_line_l443_443704


namespace max_value_of_fraction_l443_443020

theorem max_value_of_fraction (x y z : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : 10 ≤ y ∧ y < 100) (hz : 10 ≤ z ∧ z < 100) (h : x + y + z = 180) : 
  (x + y) / z ≤ 17 :=
sorry

end max_value_of_fraction_l443_443020


namespace min_value_sum_is_minimal_l443_443593

noncomputable def min_value_sum (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem min_value_sum_is_minimal :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → min_value_sum a b c ≥ 1 / (2 * real.cbrt 2) :=
by
  intros a b c ha hb hc
  let s := min_value_sum a b c
  have am_gm_ineq := am_gm s sorry -- Set up necessary details for the AM-GM application
  calc
    s ≥ 3 * real.cbrt (s)   : am_gm_ineq
      ... = 1 / (2 * real.cbrt 2) : by sorry

end min_value_sum_is_minimal_l443_443593


namespace gcd_lcm_sum_eq_l443_443855

-- Define the two numbers
def a : ℕ := 72
def b : ℕ := 8712

-- Define the GCD and LCM functions.
def gcd_ab : ℕ := Nat.gcd a b
def lcm_ab : ℕ := Nat.lcm a b

-- Define the sum of the GCD and LCM.
def sum_gcd_lcm : ℕ := gcd_ab + lcm_ab

-- The theorem we want to prove
theorem gcd_lcm_sum_eq : sum_gcd_lcm = 26160 := by
  -- Details of the proof would go here
  sorry

end gcd_lcm_sum_eq_l443_443855


namespace area_of_shaded_region_l443_443724

def parallelogram_exists (EFGH : Type) : Prop :=
  ∃ (E F G H : EFGH) (EJ JH EH : ℝ) (height : ℝ), EJ + JH = EH ∧ EH = 12 ∧ JH = 8 ∧ height = 10

theorem area_of_shaded_region {EFGH : Type} (h : parallelogram_exists EFGH) : 
  ∃ (area_shaded : ℝ), area_shaded = 100 := 
by
  sorry

end area_of_shaded_region_l443_443724


namespace product_of_chords_l443_443341

theorem product_of_chords :
  let A := 3
  let B := -3
  let radius := 3
  let num_points := 8
  let ω := Complex.exp (2 * Real.pi * Complex.I / 16)
  let C_i (i : ℕ) : Complex := radius * ω ^ i
  let AC_i (i : ℕ) : ℝ := Complex.abs (A - C_i i)
  let BC_i (i : ℕ) : ℝ := Complex.abs (B - C_i i)
  (∏ i in Finset.range 7, AC_i (i + 1)) * (∏ i in Finset.range 7, BC_i (i + 1)) = 393216 := 
sorry

end product_of_chords_l443_443341


namespace correct_calculation_l443_443099

noncomputable def is_correct : Prop :=
  ∃ B, B = (2 * Real.sqrt 3)

theorem correct_calculation :
  ¬((-Real.sqrt 3) ^ 2 = -3) ∧ ¬((Real.cbrt (-1)) = 1) ∧ ¬(((Real.sqrt 2) + 1) * ((Real.sqrt 2) - 1) = 3)
  → (Real.sqrt 12 = 2 * Real.sqrt 3) :=
by {
  -- proof omitted
  sorry
}

end correct_calculation_l443_443099


namespace shaded_region_perimeter_l443_443414

theorem shaded_region_perimeter :
  let side_length := 1
  let diagonal_length := Real.sqrt 2 * side_length
  let arc_TRU_length := (1 / 4) * (2 * Real.pi * diagonal_length)
  let arc_VPW_length := (1 / 4) * (2 * Real.pi * side_length)
  let arc_UV_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  let arc_WT_length := (1 / 4) * (2 * Real.pi * (Real.sqrt 2 - side_length))
  (arc_TRU_length + arc_VPW_length + arc_UV_length + arc_WT_length) = (2 * Real.sqrt 2 - 1) * Real.pi :=
by
  sorry

end shaded_region_perimeter_l443_443414


namespace product_zero_l443_443201

theorem product_zero :
  (∀ (n : ℕ), 1 ≤ n ∧ n ≤ 151 →
  (∏ i in finset.range 150, (1 - 2 / (i + 2)) = 0)) :=
by
  sorry

end product_zero_l443_443201


namespace hyperbolas_properties_l443_443626

-- Define the hyperbola C1 and C2
def C1 (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1
def C2 (x y : ℝ) : Prop := y^2 / 9 - x^2 / 16 = 1

-- Define the necessary properties
def have_same_asymptotes := ∀ x : ℝ, y = 3 / 4 * x ∨ y = -3 / 4 * x
def no_common_points := ∀ x y : ℝ, ¬(C1 x y ∧ C2 x y)
def equal_focal_distances := 
  let c1 := sqrt ((16 : ℝ) + 9) in
  let c2 := sqrt ((9 : ℝ) + 16) in
  2 * c1 = 2 * c2 ∧ c1 = 5 ∧ c2 = 5

-- The theorem to be proven
theorem hyperbolas_properties :
  have_same_asymptotes ∧ no_common_points ∧ equal_focal_distances :=
sorry

end hyperbolas_properties_l443_443626


namespace telescoping_series_sum_l443_443583

theorem telescoping_series_sum :
  (∑ n in Finset.range 48 \ Finset.range 2, (1 / (n + 2) * (n + 2 - 2)^(1 / 3) + ((n + 2 - 2) / (n + 2))^(1 / 3))) =
    1 - ((50 : ℝ)^(1 / 3))⁻¹ :=
  sorry

end telescoping_series_sum_l443_443583


namespace fixed_point_of_function_l443_443043

variable (a : ℝ)

noncomputable def f (x : ℝ) : ℝ := a^(1 - x) - 2

theorem fixed_point_of_function (ha_pos : 0 < a) (ha_ne_one : a ≠ 1) : f a 1 = -1 := by
  sorry

end fixed_point_of_function_l443_443043


namespace solve_inequality_l443_443613

theorem solve_inequality : 
  {x : ℝ | (1 / (x^2 + 1)) > (4 / x) + (21 / 10)} = {x : ℝ | -2 < x ∧ x < 0} :=
by
  sorry

end solve_inequality_l443_443613


namespace centroid_iff_l443_443742

noncomputable def is_centroid (P A B C : Point) : Prop :=
  ∃ (A1 B1 C1 : Point), 
    (A1 ∈ LineSegment B C) ∧ (B1 ∈ LineSegment C A) ∧ (C1 ∈ LineSegment A B) ∧
    Concurrent (Line A A1) (Line B B1) (Line C C1) P ∧
    ∀ (P : Point), Concurrent (Line A A1) (Line B B1) (Line C C1) P →
    Line P A = Line P B ∧ Line P B = Line P C

theorem centroid_iff (A B C P A1 B1 C1 : Point) 
  (hA1 : A1 ∈ LineSegment B C) 
  (hB1 : B1 ∈ LineSegment C A) 
  (hC1 : C1 ∈ LineSegment A B) 
  (hConcurrent : Concurrent (Line A A1) (Line B B1) (Line C C1) P) : 
  is_centroid P A B C ↔ is_centroid P A1 B1 C1 :=
sorry

end centroid_iff_l443_443742


namespace find_g10_l443_443585

noncomputable def g : ℕ → ℝ := sorry

axiom g_def : ∀ m n, m ≥ n → g(m + n) + g(m - n) = (g(3m) + g(3n)) / 3
axiom g_init : g 1 = 1

theorem find_g10 : g 10 = 100 := 
by 
  sorry

end find_g10_l443_443585


namespace journey_speed_l443_443898

theorem journey_speed (t_total : ℝ) (d_total : ℝ) (d_half : ℝ) (v_half2 : ℝ) (time_half2 : ℝ) (time_total : ℝ) (v_half1 : ℝ) :
  t_total = 5 ∧ d_total = 112 ∧ d_half = d_total / 2 ∧ v_half2 = 24 ∧ time_half2 = d_half / v_half2 ∧ time_total = t_total - time_half2 ∧ v_half1 = d_half / time_total → v_half1 = 21 :=
by
  intros h
  sorry

end journey_speed_l443_443898


namespace find_integer_divisible_by_24_l443_443968

theorem find_integer_divisible_by_24 : 
  ∃ n : ℕ, (n % 24 = 0) ∧ (9 < real.sqrt (real.cbrt n)) ∧ (real.sqrt (real.cbrt n) < 9.1) := 
by
  let n := 744
  use n
  have h1 : n % 24 = 0 := by norm_num
  have h2 : 9 < real.sqrt (real.cbrt n) := by norm_num
  have h3 : real.sqrt (real.cbrt n) < 9.1 := by norm_num
  exact ⟨h1, h2, h3⟩

end find_integer_divisible_by_24_l443_443968


namespace meal_combinations_correct_l443_443487

noncomputable def number_of_meal_combinations (total_dishes : ℕ) (special_dishes : ℕ) : ℕ :=
  let total_combinations := total_dishes^3
  let restricted_combinations := special_dishes * 14^2 * 3
  total_combinations - restricted_combinations

theorem meal_combinations_correct : number_of_meal_combinations 15 3 = 1611 :=
by 
  unfold number_of_meal_combinations
  norm_num
  sorry

end meal_combinations_correct_l443_443487


namespace AD_perpendicular_MN_l443_443711

-- Define the points and segments
variables {A B C D E F M N : Type}

-- Assume triangle ABC, and the given conditions in Lean 4
variables (triangle : ∀ {A B C : Type}, Prop)
variables (angle_bisector : ∀ {A B C D : Type}, Prop)
variables (line_extension : ∀ {A B E : Type}, Prop)
variables (parallel1 : CE ∥ BD)
variables (parallel2 : BF ∥ CD)
variables (midpoint1 : ∀ {CE M : Type}, Prop)
variables (midpoint2 : ∀ {BF N : Type}, Prop)

-- Statement to be proved
theorem AD_perpendicular_MN :
  ∀ (A B C D E F M N : Type) (angle_bisector : ∀ {A B C D : Type}, Prop)
  (line_extension : ∀ {A B E : Type}, Prop) (parallel1 : CE ∥ BD) (parallel2 : BF ∥ CD)
  (midpoint1 : ∀ {CE M : Type}, Prop) (midpoint2 : ∀ {BF N : Type}, Prop),
  AD ⊥ MN :=
begin
  sorry
end

end AD_perpendicular_MN_l443_443711


namespace solve_equation_l443_443805

theorem solve_equation (x : ℝ) (h : 2^(2 * x) + 16 = 12 * 2^x) : x^2 - 1 = 0 ∨ x^2 - 1 = 8 :=
sorry

end solve_equation_l443_443805


namespace fern_pays_228_11_usd_l443_443945

open Real

noncomputable def high_heels_price : ℝ := 66
noncomputable def ballet_slippers_price : ℝ := (2 / 3) * high_heels_price
noncomputable def purse_price : ℝ := 49.5
noncomputable def scarf_price : ℝ := 27.5
noncomputable def high_heels_discount : ℝ := 0.10 * high_heels_price
noncomputable def discounted_high_heels_price : ℝ := high_heels_price - high_heels_discount
noncomputable def total_cost_before_tax : ℝ := discounted_high_heels_price + ballet_slippers_price + purse_price + scarf_price
noncomputable def sales_tax : ℝ := 0.075 * total_cost_before_tax
noncomputable def total_cost_after_tax : ℝ := total_cost_before_tax + sales_tax
noncomputable def exchange_rate : ℝ := 1 / 0.85
noncomputable def total_cost_in_usd : ℝ := total_cost_after_tax * exchange_rate

theorem fern_pays_228_11_usd: total_cost_in_usd = 228.11 := by
  sorry

end fern_pays_228_11_usd_l443_443945


namespace find_number_l443_443981

theorem find_number (n : ℕ) (h1 : 9 < real.cbrt n) (h2 : real.cbrt n < 9.1) (h3 : n % 24 = 0) : n = 744 :=
sorry

end find_number_l443_443981


namespace no_equilateral_right_or_obtuse_triangle_l443_443105

theorem no_equilateral_right_or_obtuse_triangle :
  (¬ ∃ (T : Type*) [triangle T], equilateral T ∧ right_triangle T) ∧
  (¬ ∃ (T : Type*) [triangle T], equilateral T ∧ obtuse_triangle T) :=
by
  -- Proof omitted for brevity
  sorry

end no_equilateral_right_or_obtuse_triangle_l443_443105


namespace three_digit_even_distinct_sum_mod_1000_l443_443559

theorem three_digit_even_distinct_sum_mod_1000 :
  ( ∑ n in { m | even m ∧ 100 ≤ m ∧ m < 1000 ∧ ∀ i j, i ≠ j → 
    (digit m i) ≠ (digit m j) }, n ) % 1000 = 120 := 
sorry

end three_digit_even_distinct_sum_mod_1000_l443_443559


namespace x_pow_a_not_satisfy_any_l443_443660

theorem x_pow_a_not_satisfy_any (a : Real) :
  (∀ f : Real → Real, (∃ x y : Real, f (x * y) ≠ f x + f y) ∧ 
   (∃ x y : Real, f (x + y) ≠ f x * f y) ∧ 
   (∃ x y : Real, f (x + y) ≠ f x + f y)) →
  f = fun x => x^a → 
  (∃ x y : Real, f (x * y) ≠ f x + f y) ∧ 
  (∃ x y : Real, f (x + y) ≠ f x * f y) ∧ 
  (∃ x y : Real, f (x + y) ≠ f x + f y) := by
  sorry

end x_pow_a_not_satisfy_any_l443_443660


namespace conic_eccentricity_l443_443703

theorem conic_eccentricity (m : ℝ) (h : 0 < -m) (h2 : (Real.sqrt (1 + (-1 / m))) = 2) : m = -1/3 := 
by
  -- Proof can be added here
  sorry

end conic_eccentricity_l443_443703


namespace andrey_stamps_count_l443_443385

theorem andrey_stamps_count (x : ℕ) : 
  (x % 3 = 1) ∧ (x % 5 = 3) ∧ (x % 7 = 5) ∧ (150 < x ∧ x ≤ 300) → x = 208 := 
by 
  sorry

end andrey_stamps_count_l443_443385


namespace order_of_areas_l443_443551

-- Definitions for the areas given the conditions
def S1 : ℝ := 0.57
def S2 : ℝ := 0.215
def S3 : ℝ := 0.5
def S4 : ℝ := S2 + 0.1525 -- Since S4 = S2 + "extra part" = 0.215 + 0.1525

-- Theorem to prove the ascending order
theorem order_of_areas : S2 < S4 ∧ S4 < S3 ∧ S3 < S1 :=
by {
  -- Adding the values derived for clarity, though proofs are not included
  have h1 : S1 = 0.57, by rfl,
  have h2 : S2 = 0.215, by rfl,
  have h3 : S3 = 0.5, by rfl,
  have h4 : S4 = 0.3675, by rfl,
  -- Proof steps are not included
  sorry -- the proof has been omitted
}

end order_of_areas_l443_443551


namespace max_geometric_sequence_terms_l443_443310

theorem max_geometric_sequence_terms (a r : ℝ) (n : ℕ) (h_r : r > 1) 
    (h_seq : ∀ (k : ℕ), 1 ≤ k ∧ k ≤ n → 100 ≤ a * r^(k-1) ∧ a * r^(k-1) ≤ 1000) :
  n ≤ 6 :=
sorry

end max_geometric_sequence_terms_l443_443310


namespace fort_blocks_count_l443_443136

noncomputable def volume_of_blocks (l w h : ℕ) (wall_thickness floor_thickness top_layer_volume : ℕ) : ℕ :=
  let interior_length := l - 2 * wall_thickness
  let interior_width := w - 2 * wall_thickness
  let interior_height := h - floor_thickness
  let volume_original := l * w * h
  let volume_interior := interior_length * interior_width * interior_height
  volume_original - volume_interior + top_layer_volume

theorem fort_blocks_count : volume_of_blocks 15 12 7 2 1 180 = 912 :=
by
  sorry

end fort_blocks_count_l443_443136


namespace part1_evaluation_part2_evaluation_part3_evaluation_l443_443919

-- Statement for Part 1
theorem part1_evaluation : 0.027^(-1/3) - (-(1/7))^(-2) + 256^(3/4) - 3^(-1) + (sqrt 2 - 1)^0 = 16 :=
by sorry

-- Statement for Part 2
theorem part2_evaluation : log 2.5 6.25 + log 10 0.01 + log (Math.E.pow 1/2) - 2^(1 + log 2 3) = -11/2 :=
by sorry

-- Statement for Part 3
theorem part3_evaluation : log 10 (5^2) + (2/3) * log 10 8 + log 10 5 * log 10 20 + (log 10 2)^2 = 3.6 :=
by sorry

end part1_evaluation_part2_evaluation_part3_evaluation_l443_443919


namespace milton_zoology_books_l443_443783

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l443_443783


namespace complex_problem_l443_443238

theorem complex_problem 
  (a : ℝ) 
  (ha : a^2 - 9 = 0) :
  (a + (Complex.I ^ 19)) / (1 + Complex.I) = 1 - 2 * Complex.I := by
  sorry

end complex_problem_l443_443238


namespace distinct_diff_count_l443_443288

theorem distinct_diff_count :
  (∃ S : set ℕ, S = {1, 3, 5, 7, 9, 11} ∧ 
    (∑ x in S, ∑ y in S, if x > y then 1 else 0) = 10) :=
begin
  let S := {1, 3, 5, 7, 9, 11},
  use S,
  split,
  {
    refl,
  },
  {
    sorry
  }
end

end distinct_diff_count_l443_443288


namespace eccentricity_difference_is_correct_l443_443182

noncomputable def eccentricity_difference : ℝ :=
  let a := some positive_real
  let b := some (positive_real_lt a)
  let f := λ θ, _ /- The function f(θ) representing the eccentricity based on given conditions -/
  f (2 * Real.pi / 3) - f (Real.pi / 3)

theorem eccentricity_difference_is_correct :
  eccentricity_difference = (2 * Real.sqrt 3) / 3 :=
sorry

end eccentricity_difference_is_correct_l443_443182


namespace circle_covers_extrema_l443_443301

noncomputable def f (x k : ℝ) : ℝ := (real.sqrt 3) * real.sin (real.pi * x / k)

theorem circle_covers_extrema (k : ℝ) : (∀ x y : ℝ, x^2 + y^2 = k^2 → (∃ xₘ : ℝ, f xₘ k = real.sqrt 3) ∧ (∃ xₘ : ℝ, f xₘ k = -real.sqrt 3)) → 2 ≤ k := 
begin
  sorry
end

end circle_covers_extrema_l443_443301


namespace solve_system_l443_443696

theorem solve_system (x y : ℝ) (h1 : 4 * x - y = 2) (h2 : 3 * x - 2 * y = -1) : x - y = -1 := 
by
  sorry

end solve_system_l443_443696


namespace decompose_five_eighths_l443_443183

theorem decompose_five_eighths : 
  ∃ a b c : ℕ, 0 < a ∧ 0 < b ∧ 0 < c ∧ (5 : ℚ) / 8 = 1 / (a : ℚ) + 1 / (b : ℚ) + 1 / (c : ℚ) := 
by
  sorry

end decompose_five_eighths_l443_443183


namespace dice_digit_distribution_l443_443465

theorem dice_digit_distribution : ∃ n : ℕ, n = 10 ∧ 
  (∀ (d1 d2 : Finset ℕ), d1.card = 6 ∧ d2.card = 6 ∧
  (0 ∈ d1) ∧ (1 ∈ d1) ∧ (2 ∈ d1) ∧ 
  (0 ∈ d2) ∧ (1 ∈ d2) ∧ (2 ∈ d2) ∧
  ({3, 4, 5, 6, 7, 8} ⊆ (d1 ∪ d2)) ∧ 
  (∀ i, i ∈ d1 ∪ d2 → i ∈ (Finset.range 10))) := 
  sorry

end dice_digit_distribution_l443_443465


namespace sum_of_values_l443_443425

def r (x : ℝ) : ℝ := abs (x + 1) - 3
def s (x : ℝ) : ℝ := -(abs (x + 2))

theorem sum_of_values :
  (s (r (-5)) + s (r (-4)) + s (r (-3)) + s (r (-2)) + s (r (-1)) + s (r (0)) + s (r (1)) + s (r (2)) + s (r (3))) = -37 :=
by {
  sorry
}

end sum_of_values_l443_443425


namespace marble_problem_l443_443504

theorem marble_problem 
  (B : ℕ)
  (h : (3/(3 + B) * 2/(2 + B) * 1/(1 + B)) + (B/(3 + B) * (B - 1)/(2 + B) * (B - 2)/(1 + B)) = 0.1) 
  : B = 3 := 
sorry

end marble_problem_l443_443504


namespace domain_of_f_l443_443419

noncomputable def domain_f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem domain_of_f : domain_f = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end domain_of_f_l443_443419


namespace additional_men_to_finish_work_earlier_l443_443543

theorem additional_men_to_finish_work_earlier:
  let M := 15 in
  let T := 10 in
  let t := 7 in
  let N : ℕ := 6 in
  (7 * (M + N) = 10 * M) :=
sorry

end additional_men_to_finish_work_earlier_l443_443543


namespace correct_propositions_l443_443259

variables {L1 L2 : Type} [linear_order L1] [linear_order L2]
variables {P P1 P2 : Type} [plane P] [plane P1] [plane P2]
variables {l : Type} [line l]

-- Proposition 3:
axiom perp_same_plane_parallel (L1 L2 : Type) (P : Type) [line L1] [line L2] [plane P] : 
  (⊥ L1 P) ∧ (⊥ L2 P) → (// L1 L2)

-- Proposition 4:
axiom perp_planes_nonperp_intersection (P1 P2 : Type) (L l : Type) [plane P1] [plane P2] [line L] [line l] : 
  (⊥ P1 P2) ∧ (⊆ L P1) ∧ (¬ (⊥ L l)) ∧ (intersect_line P1 P2 = l) → (¬ (⊥ L P2))

theorem correct_propositions (L1 L2 : Type) (P P1 P2 : Type) (L l : Type) [line L1] [line L2] [plane P] [plane P1] [plane P2] [line L] [line l] : 
  (⊥ L1 P) ∧ (⊥ L2 P) → (// L1 L2) ∧ 
  (⊥ P1 P2) ∧ (⊆ L P1) ∧ (¬ (⊥ L l)) ∧ (intersect_line P1 P2 = l) → (¬ (⊥ L P2)) :=
begin
  sorry
end

end correct_propositions_l443_443259


namespace cost_of_four_stamps_l443_443789

-- Define the cost of one stamp
def cost_of_one_stamp := 0.34

-- Define that the cost remains the same for multiple stamps
def cost_of_stamps (n : ℕ) := n * cost_of_one_stamp

-- State the main theorem to prove the cost of 4 stamps
theorem cost_of_four_stamps : cost_of_stamps 4 = 1.36 :=
sorry

end cost_of_four_stamps_l443_443789


namespace twenty_five_billion_scientific_notation_l443_443725

theorem twenty_five_billion_scientific_notation :
  (25 * 10^9 : ℝ) = 2.5 * 10^10 := 
by simp only [←mul_assoc, ←@pow_add ℝ, pow_one, two_mul];
   norm_num

end twenty_five_billion_scientific_notation_l443_443725


namespace max_total_profit_max_avg_annual_profit_l443_443542

noncomputable def total_profit (x : ℕ) : ℝ := - (x : ℝ)^2 + 18 * x - 36
noncomputable def avg_annual_profit (x : ℕ) : ℝ := (total_profit x) / x

theorem max_total_profit : ∃ x : ℕ, total_profit x = 45 ∧ x = 9 :=
  by sorry

theorem max_avg_annual_profit : ∃ x : ℕ, avg_annual_profit x = 6 ∧ x = 6 :=
  by sorry

end max_total_profit_max_avg_annual_profit_l443_443542


namespace min_perimeter_triangle_l443_443932

open set

variable {A B C : Point}
variable {l : Line}

theorem min_perimeter_triangle (A B : Point) (l : Line) : 
  ∃ C : Point, C ∈ l ∧ (∀ C' : Point, C' ∈ l → (dist A C + dist C B ≤ dist A C' + dist C' B)) :=
sorry

end min_perimeter_triangle_l443_443932


namespace fatima_donates_75_square_inches_l443_443203

theorem fatima_donates_75_square_inches :
  ∀ (cloth: ℚ), cloth = 100 →
  (∃ (c1 c2 c3: ℚ), c1 = cloth / 2 ∧ c2 = c1 / 2 ∧ c3 = 75) →
  (c1 + c2 = c3) :=
by
  assume cloth
  assume h1 : cloth = 100
  assume h2 : ∃ (c1 c2 c3: ℚ), c1 = cloth / 2 ∧ c2 = c1 / 2 ∧ c3 = 75
  sorry

end fatima_donates_75_square_inches_l443_443203


namespace train_speed_kmph_l443_443146

-- defining the conditions
def train_length : ℝ := 155
def bridge_length : ℝ := 220.03
def crossing_time : ℝ := 30

-- defining constants
def total_distance : ℝ := train_length + bridge_length
def speed_mps : ℝ := total_distance / crossing_time
def mps_to_kmph : ℝ := 3.6

noncomputable def speed_kmph : ℝ := speed_mps * mps_to_kmph

-- the theorem statement
theorem train_speed_kmph : speed_kmph = 45.0036 := by
    sorry

end train_speed_kmph_l443_443146


namespace quadratic_inequality_solution_set_l443_443832

theorem quadratic_inequality_solution_set (p q : ℝ) :
  (∀ x : ℝ, 2 < x ∧ x < 3 → x^2 - p * x - q < 0) →
  p = 5 ∧ q = -6 ∧
  (∀ x : ℝ, - (1 : ℝ) / 2 < x ∧ x < - (1 : ℝ) / 3 → 6 * x^2 + 5 * x + 1 < 0) :=
by
  sorry

end quadratic_inequality_solution_set_l443_443832


namespace two_unique_lines_l443_443255

noncomputable def skew_lines (A B : Type) [Add A] [Mul A] := { a // ∃ (c : A), a * c = 0 }
noncomputable def point_space (P : Type) := P

theorem two_unique_lines {A B P : Type} [Add A] [Mul A] [point_space P] (angle_ab : ℝ) (angle_pa : ℝ) (angle_pb : ℝ) 
(h_ab : angle_ab = 50) (h_pa : angle_pa = 30) (h_pb : angle_pb = 30) :
  ∃! (l : skew_lines A B), ∃ (p : point_space P) (a b : skew_lines A B),
    (angle a b = angle_ab) ∧ (angle p a = angle_pa) ∧ (angle p b = angle_pb) :=
  sorry

end two_unique_lines_l443_443255


namespace area_inequality_l443_443312

-- Define the region covered by the triangles S
def region_S (S : Set Triangle) : Set Point :=
  ⋃ i, S i

-- Define the region covered by the medial triangles T
def region_T (T : Set Triangle) : Set Point :=
  ⋃ i, T i

-- Define the areas for regions S and T
def area_S (S : Set Triangle) : ℝ :=
  ∑ i, S i.area

def area_T (T : Set Triangle) : ℝ :=
  ∑ i, T i.area

-- The necessary condition for each medial triangle T_i within its corresponding S_i
axiom medial_triangle (S_i T_i : Triangle) : T_i ⊆ S_i ∧ S_i.area = 4 * T_i.area

-- The main theorem to prove
theorem area_inequality (S T : Set Triangle)
  (h1 : ∀ i, medial_triangle (S i) (T i)) :
  area_S S ≤ 4 * area_T T :=
sorry

end area_inequality_l443_443312


namespace solve_for_A_l443_443691

def clubsuit (A B : ℤ) : ℤ := 3 * A + 2 * B + 7

theorem solve_for_A (A : ℤ) : (clubsuit A 6 = 70) -> (A = 17) :=
by
  sorry

end solve_for_A_l443_443691


namespace triangle_obtuse_l443_443250

theorem triangle_obtuse (α : ℝ) (hα : 0 < α ∧ α < π) (h_trig : sin α + cos α = 2 / 3) : π / 2 < α ∧ α < π :=
sorry

end triangle_obtuse_l443_443250


namespace maximize_triangle_groups_l443_443837

/-- There are 1989 points in space, none of which are collinear. 
These points are divided into 30 groups with different numbers of points in each group. 
From each of three different groups, one point is taken to form a triangle. 
To maximize the total number of triangles, show that each group should have the following number of points:
{51, 52, 53, 54, 55, 56, 58, 59, ..., 81} -/
theorem maximize_triangle_groups :
  ∃ (m : Fin 30 → ℕ), (∀ i j : Fin 30, i < j → m i < m j) ∧ 
  (∑ i, m i = 1989) ∧ 
  (∀ i j k : Fin 30, i < j → j < k → m i < m j ∧ m j < m k) ∧
  m = ![51, 52, 53, 54, 55, 56, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81] :=
sorry

end maximize_triangle_groups_l443_443837


namespace angle_remains_unchanged_l443_443089

-- Definition of magnification condition (though it does not affect angle in mathematics, we state it as given)
def magnifying_glass (magnification : ℝ) (initial_angle : ℝ) : ℝ := 
  initial_angle  -- Magnification does not change the angle in this context.

-- Given condition
def initial_angle : ℝ := 30

-- Theorem we want to prove
theorem angle_remains_unchanged (magnification : ℝ) (h_magnify : magnification = 100) :
  magnifying_glass magnification initial_angle = initial_angle :=
by
  sorry

end angle_remains_unchanged_l443_443089


namespace triangle_right_angle_l443_443300

theorem triangle_right_angle
  (A B : ℝ)  -- Angles A and B
  (hA : 0 < A ∧ A < π / 2)  -- A is an acute angle
  (hB : 0 < B ∧ B < π / 2)  -- B is an acute angle
  (h : sin A ^ 2 + sin B ^ 2 = sin (A + B)) :
  ∃ C : ℝ, C = π / 2 ∧ A + B + C = π :=  -- Triangle is right-angled with angle C = π / 2
sorry

end triangle_right_angle_l443_443300


namespace max_value_of_fraction_l443_443002

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443002


namespace flagpole_arrangement_modulo_500_l443_443842

theorem flagpole_arrangement_modulo_500 :
  ∃ (M : ℕ), 
    (∀ (arrangement : ℕ → ℕ → Prop), 
      (∀ (pole1_cnt pole2_cnt : ℕ), 
        pole1_cnt ≠ pole2_cnt ∧
        pole1_cnt < pole2_cnt ∧
        no_adjacent_green (arrangement pole1_cnt) ∧
        no_adjacent_green (arrangement pole2_cnt) →
        valid_arrangement arrangement pole1_cnt pole2_cnt 20 12 8) →
      M = count_valid_arrangements arrangement) →
    M % 500 = 222 :=
by 
  sorry

-- Definitions for no_adjacent_green, valid_arrangement, and count_valid_arrangements
-- need to be appropriately defined based on the problem's constraints.

end flagpole_arrangement_modulo_500_l443_443842


namespace chord_length_of_circle_l443_443634

theorem chord_length_of_circle (a : ℝ) (ha : a ≠ 0) :
  let y := (1/2) * a^2 in 
  let M := (a, y) in
  let r := Math.sqrt(a^2 + ((1/2) * a^2 - 1)^2) in
  let circle_eq := (x - a) ^ 2 + (y - (1/2 * a ^ 2)) ^ 2 = a ^ 2 + (1 / 2 * a ^ 2 - 1) ^ 2 in
  let PQ_length := |a + 1 - (a - 1)| in
  PQ_length = 2 := 
by 
  sorry

end chord_length_of_circle_l443_443634


namespace circle_intersection_relation_l443_443345

variables {P A B C D : Type} 
  [point P] [point A] [point B] [point C] [point D]
  (C1 C2 C3 C4 : Type) 
  [circle C1] [circle C2] [circle C3] [circle C4]

noncomputable def externally_tangent_at (C1 C2 : Type) [circle C1] [circle C2] (P : Type) [point P] : Prop := 
sorry -- definition of externally tangent at P

-- Hypotheses according to conditions
variables 
  (h1 : externally_tangent_at C1 C3 P)
  (h2 : externally_tangent_at C2 C4 P)
  (h3 : intersection_point C1 C2 A) -- A != P ensured by definition of intersection_point
  (h4 : intersection_point C2 C3 B)
  (h5 : intersection_point C3 C4 C)
  (h6 : intersection_point C4 C1 D)

-- Proving the required relationship
theorem circle_intersection_relation :
  AB * BC / (AD * DC) = PB^2 / P^2 :=
sorry

end circle_intersection_relation_l443_443345


namespace probability_of_jack_king_ace_l443_443075

theorem probability_of_jack_king_ace :
  let prob_jack := (4 : ℚ) / 52,
      prob_king := (4 : ℚ) / 51,
      prob_ace := (4 : ℚ) / 50 in
  prob_jack * prob_king * prob_ace = 16 / 33150 := 
by
  sorry

end probability_of_jack_king_ace_l443_443075


namespace square_circle_union_area_l443_443895

theorem square_circle_union_area (side_length : ℝ) (h : side_length = 8) :
  let square_area := side_length ^ 2 in
  let circle_radius := side_length / 2 in
  let circle_area := π * circle_radius ^ 2 in
  circle_area <= square_area →
  square_area = 64 :=
by
  intros
  simp [square_area, circle_radius, circle_area] at *
  sorry

end square_circle_union_area_l443_443895


namespace lacy_correct_percentage_is_80_l443_443717

-- Define the total number of problems
def total_problems (x : ℕ) : ℕ := 5 * x + 10

-- Define the number of problems Lacy missed
def problems_missed (x : ℕ) : ℕ := x + 2

-- Define the number of problems Lacy answered correctly
def problems_answered (x : ℕ) : ℕ := total_problems x - problems_missed x

-- Define the fraction of problems Lacy answered correctly
def fraction_answered_correctly (x : ℕ) : ℚ :=
  (problems_answered x : ℚ) / (total_problems x : ℚ)

-- The main theorem to prove the percentage of problems correctly answered is 80%
theorem lacy_correct_percentage_is_80 (x : ℕ) : 
  fraction_answered_correctly x = 4 / 5 := 
by 
  sorry

end lacy_correct_percentage_is_80_l443_443717


namespace milton_zoology_books_l443_443784

theorem milton_zoology_books
  (z b : ℕ)
  (h1 : z + b = 80)
  (h2 : b = 4 * z) :
  z = 16 :=
by sorry

end milton_zoology_books_l443_443784


namespace min_value_sum_is_minimal_l443_443594

noncomputable def min_value_sum (a b c : ℝ) : ℝ :=
  a / (3 * b) + b / (6 * c) + c / (9 * a)

theorem min_value_sum_is_minimal :
  ∀ (a b c : ℝ), 0 < a → 0 < b → 0 < c → min_value_sum a b c ≥ 1 / (2 * real.cbrt 2) :=
by
  intros a b c ha hb hc
  let s := min_value_sum a b c
  have am_gm_ineq := am_gm s sorry -- Set up necessary details for the AM-GM application
  calc
    s ≥ 3 * real.cbrt (s)   : am_gm_ineq
      ... = 1 / (2 * real.cbrt 2) : by sorry

end min_value_sum_is_minimal_l443_443594


namespace land_area_square_l443_443108

theorem land_area_square (s : ℕ) (h1 : s = 32) : s * s = 1024 :=
by
  rw [h1]
  sorry -- To be proved

end land_area_square_l443_443108


namespace find_f_2008_l443_443185

noncomputable def f : ℝ → ℝ :=
  sorry

theorem find_f_2008 (f : ℝ → ℝ) (h1 : ∀ x, f(x) = f(4 - x)) (h2 : ∀ x, f(2 - x) + f(x - 2) = 0) : f(2008) = 0 :=
by
  -- sorry for skipping the proof
  sorry

end find_f_2008_l443_443185


namespace find_integer_divisible_by_24_l443_443971

theorem find_integer_divisible_by_24 : 
  ∃ n : ℕ, (n % 24 = 0) ∧ (9 < real.sqrt (real.cbrt n)) ∧ (real.sqrt (real.cbrt n) < 9.1) := 
by
  let n := 744
  use n
  have h1 : n % 24 = 0 := by norm_num
  have h2 : 9 < real.sqrt (real.cbrt n) := by norm_num
  have h3 : real.sqrt (real.cbrt n) < 9.1 := by norm_num
  exact ⟨h1, h2, h3⟩

end find_integer_divisible_by_24_l443_443971


namespace matthew_total_time_on_failure_day_l443_443371

-- Define the conditions as variables
def assembly_time : ℝ := 1 -- hours
def usual_baking_time : ℝ := 1.5 -- hours
def decoration_time : ℝ := 1 -- hours
def baking_factor : ℝ := 2 -- Factor by which baking time increased on that day

-- Prove that the total time taken is 5 hours
theorem matthew_total_time_on_failure_day : 
  (assembly_time + (usual_baking_time * baking_factor) + decoration_time) = 5 :=
by {
  sorry
}

end matthew_total_time_on_failure_day_l443_443371


namespace cow_difference_l443_443902

variables (A M R : Nat)

def Aaron_has_four_times_as_many_cows_as_Matthews : Prop := A = 4 * M
def Matthews_has_cows : Prop := M = 60
def Total_cows_for_three := A + M + R = 570

theorem cow_difference (h1 : Aaron_has_four_times_as_many_cows_as_Matthews A M) 
                       (h2 : Matthews_has_cows M)
                       (h3 : Total_cows_for_three A M R) :
  (A + M) - R = 30 :=
by
  sorry

end cow_difference_l443_443902


namespace angle_ABC_eq_90_l443_443844

variables (x y z t : ℝ)
axiom car1 : x / 12 + y / 10 + z / 15 = t
axiom car2 : x / 15 + y / 15 + z / 10 = t
axiom car3 : x / 10 + y / 20 + z / 12 = t

theorem angle_ABC_eq_90 (h : x / 12 + y / 10 + z / 15 = t) 
                        (h' : x / 15 + y / 15 + z / 10 = t) 
                        (h'' : x / 10 + y / 20 + z / 12 = t) :
  ∠ ABC = 90 :=
sorry

end angle_ABC_eq_90_l443_443844


namespace exists_lim_at_infty_eq_zero_l443_443339

variable (f : ℝ → ℝ)
variable [∀ x, DifferentiableAt ℝ f x]

theorem exists_lim_at_infty_eq_zero
  (h1 : ∀ x ≥ 0, |f x| ≤ 5)
  (h2 : ∀ x ≥ 0, f x * (deriv f x) ≥ Real.sin x) :
  ∃ l : ℝ, tendsto f at_top (nhds l) ∧ l = 0 := by 
  sorry

end exists_lim_at_infty_eq_zero_l443_443339


namespace eccentricity_mn_sum_l443_443935

theorem eccentricity_mn_sum (h : ℝ := -2 / 5) (a2 : ℚ := 64 / 25) (b2 : ℚ := 441 / 125) :
  let c2 := a2 - b2 in
  let e := real.sqrt (c2 / a2) in
  let m := 49 in -- Numerator
  let n := 320 in -- Denominator
  m + n = 369 :=
by sorry

end eccentricity_mn_sum_l443_443935


namespace spadesuit_evaluation_l443_443231

def spadesuit (x y : ℝ) : ℝ := x - (1 / y)

theorem spadesuit_evaluation : spadesuit 3 (spadesuit 2 (5 / 3)) = 16 / 7 :=
by
  sorry

end spadesuit_evaluation_l443_443231


namespace ratio_AB_BM_l443_443056

theorem ratio_AB_BM (A B C M : Point) (hBM : median B M A C)
  (angle_ABM : angle A B M = 40) (angle_CBM : angle C B M = 70) :
  (AB / BM = cos 20 / sin 40) := 
  sorry

end ratio_AB_BM_l443_443056


namespace segment_CD_length_is_50_over_3_l443_443436
-- Define the conditions and parameters
def length_of_segment_CD (V : ℝ) (r h : ℝ) : Prop :=
  let V_cylinder := (π * r^2 * h)
  let V_hemisphere := (1 / 2 * (4 / 3 * π * r^3))
  let total_volume := V_cylinder + 2 * V_hemisphere
  total_volume = V

theorem segment_CD_length_is_50_over_3 :
  length_of_segment_CD (352 * π) 4 (50 / 3) :=
by
  sorry

end segment_CD_length_is_50_over_3_l443_443436


namespace remainder_of_3_pow_45_mod_17_l443_443095

theorem remainder_of_3_pow_45_mod_17 : 3^45 % 17 = 15 := 
by {
  sorry
}

end remainder_of_3_pow_45_mod_17_l443_443095


namespace find_t_l443_443677

-- Define vectors a and b
def a := (3 : ℝ, 4 : ℝ)
def b := (1 : ℝ, 0 : ℝ)

-- Define the vector c as a function of t
def c (t : ℝ) := (a.1 + t * b.1, a.2 + t * b.2)

-- Statement of the theorem to be proven
theorem find_t (t : ℝ) :
  (a.1 * (a.1 + t * b.1) + a.2 * (a.2 + t * b.2)) = (b.1 * (a.1 + t * b.1) + b.2 * (a.2 + t * b.2)) →
  t = 5 :=
by
  sorry

end find_t_l443_443677


namespace distance_traveled_by_circle_center_l443_443134

theorem distance_traveled_by_circle_center
  (r : ℝ) (a b c : ℝ) 
  (h₁ : r = 2)
  (h₂ : a = 9)
  (h₃ : b = 12)
  (h₄ : c = 15)
  (h₅ : a^2 + b^2 = c^2) :
  let DE := a, DF := b, EF := c,
      XYZ :=
        { xy := DE - 2*r,
          xz := DF - 2*r,
          yz := EF - 2*r } in
  XYZ.xy + XYZ.xz + XYZ.yz = 24 :=
by
  sorry

end distance_traveled_by_circle_center_l443_443134


namespace find_m_in_triangle_ABC_l443_443736

theorem find_m_in_triangle_ABC (A B C E : Type)
  [AddCommGroup A] [Module ℝ A]
  (AB BC AC BE m : ℝ) 
  (h1 : AB = 4) 
  (h2 : BC = 5) 
  (h3 : AC = 6) 
  (h4 : BE = m * √3) 
  (angle_bisector : true) -- This line just holds a place for the angle bisector condition
  : m = 10 / 3 :=
by
  sorry

end find_m_in_triangle_ABC_l443_443736


namespace not_always_possible_triangle_sides_l443_443347

theorem not_always_possible_triangle_sides (α β γ δ : ℝ) 
  (h1 : α + β + γ + δ = 360) 
  (h2 : α < 180) 
  (h3 : β < 180) 
  (h4 : γ < 180) 
  (h5 : δ < 180) : 
  ¬ (∀ (x y z : ℝ), (x = α ∨ x = β ∨ x = γ ∨ x = δ) ∧ (y = α ∨ y = β ∨ y = γ ∨ y = δ) ∧ (z = α ∨ z = β ∨ z = γ ∨ z = δ) ∧ (x ≠ y) ∧ (x ≠ z) ∧ (y ≠ z) → x + y > z ∧ x + z > y ∧ y + z > x)
:= sorry

end not_always_possible_triangle_sides_l443_443347


namespace time_increase_percentage_l443_443372

theorem time_increase_percentage (total_distance : ℝ) (first_half_distance : ℝ)
  (first_half_speed : ℝ) (total_avg_speed : ℝ) :
  let time_first_half := first_half_distance / first_half_speed
  let total_time := total_distance / total_avg_speed
  let time_second_half := total_time - time_first_half
  let percentage_increase := ((time_second_half - time_first_half) / time_first_half) * 100
  total_distance = 640 →
  first_half_distance = total_distance / 2 →
  first_half_speed = 80 →
  total_avg_speed = 40 →
  percentage_increase = 200 :=
by
  intros total_distance_eq first_half_distance_eq first_half_speed_eq total_avg_speed_eq
  let time_first_half := first_half_distance / first_half_speed
  let total_time := total_distance / total_avg_speed
  let time_second_half := total_time - time_first_half
  let percentage_increase := ((time_second_half - time_first_half) / time_first_half) * 100
  have h1 : first_half_distance = 320 := by
    rw [total_distance_eq]
    exact first_half_distance_eq
  have h2 : time_first_half = 4 := by
    rw [h1, first_half_speed_eq]
    norm_num
  have h3 : total_time = 16 := by
    rw [total_distance_eq, total_avg_speed_eq]
    norm_num
  have h4 : time_second_half = 12 := by
    rw [h3, h2]
    norm_num
  have h5 : percentage_increase = 200 := by
    rw [h4, h2]
    norm_num
  exact h5

end time_increase_percentage_l443_443372


namespace work_completion_days_l443_443695

theorem work_completion_days (m r e d : ℕ) :
  let W := m * d * e in 
  let W_new := m * e + r * (e + 1) in 
  W / W_new = m * d * e / (m * e + r * e + r) :=
by
  sorry

end work_completion_days_l443_443695


namespace coprime_solutions_are_one_and_two_prod_l443_443208

def coprime_pairs (x y : ℕ) : Prop :=
  x ∣ (y^2 + 210) ∧ y ∣ (x^2 + 210) ∧ Nat.gcd x y = 1

theorem coprime_solutions_are_one_and_two_prod {x y : ℕ} (h : coprime_pairs x y) :
  (x = 1 ∧ y = 1) ∨ (x = 1 ∧ y = 211) :=
begin
  sorry
end

end coprime_solutions_are_one_and_two_prod_l443_443208


namespace geometric_seq_formula_max_sum_seq_l443_443733

theorem geometric_seq_formula (q : ℝ) (n : ℕ) (a_n: ℕ → ℝ): 
  (0 < q) → (a₁ = 2) → (∀ m : ℕ, a (m+1) = q * a m) → 
  (2 * a 1 = a 3 ∧ a 3 = 3 * a 2) →
  (∀ n : ℕ, a n = 2 ^ n ) :=
by
  sorry

theorem max_sum_seq (b : ℕ → ℝ) (T : ℕ → ℝ):
  (b n = 11 - 2 * log (2, a n)) → 
  (a n = 2 ^ n) → 
  (∀ n : ℕ, T n = (n * (9 + 11 - 2 * n)) / 2 ) →
  ∃ n : ℕ, T n ≤ 25 :=
by
  sorry

end geometric_seq_formula_max_sum_seq_l443_443733


namespace problem1_problem2_l443_443562

-- Problem 1
theorem problem1 (a b : ℝ) : 4 * a^4 * b^3 / (-2 * a * b)^2 = a^2 * b :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (3 * x - y)^2 - (3 * x + 2 * y) * (3 * x - 2 * y) = 5 * y^2 - 6 * x * y :=
by
  sorry

end problem1_problem2_l443_443562


namespace matrix_A_squared_unique_l443_443772

open Matrix

variables {R : Type*} [CommRing R] {A : Matrix (Fin 2) (Fin 2) R}

def matrix_A_condition (A : Matrix (Fin 2) (Fin 2) ℝ) : Prop := A ^ 4 = 0

theorem matrix_A_squared_unique (A : Matrix (Fin 2) (Fin 2) ℝ) (h : matrix_A_condition A) : A ^ 2 = 0 :=
sorry

end matrix_A_squared_unique_l443_443772


namespace max_value_of_fraction_l443_443003

open Nat 

theorem max_value_of_fraction {x y z : ℕ} (hx : 10 ≤ x ∧ x ≤ 99) (hy : 10 ≤ y ∧ y ≤ 99) (hz : 10 ≤ z ∧ z ≤ 99) 
  (h_mean : (x + y + z) / 3 = 60) : (max ((x + y) / z) 17) = 17 :=
sorry

end max_value_of_fraction_l443_443003


namespace find_integer_divisible_by_24_with_cube_root_in_range_l443_443964

theorem find_integer_divisible_by_24_with_cube_root_in_range :
  ∃ (n : ℕ), (9 < real.cbrt n) ∧ (real.cbrt n < 9.1) ∧ (24 ∣ n) ∧ n = 744 := by
    sorry

end find_integer_divisible_by_24_with_cube_root_in_range_l443_443964


namespace sum_not_prime_if_product_equality_l443_443357

theorem sum_not_prime_if_product_equality 
  (a b c d : ℕ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (h : a * b = c * d) : ¬Nat.Prime (a + b + c + d) := 
by
  sorry

end sum_not_prime_if_product_equality_l443_443357


namespace find_area_of_M_l443_443343

noncomputable def area_of_M : ℝ :=
  let M := {p : ℝ × ℝ | ∃ a b : ℝ, (p.1 - a)^2 + (p.2 - b)^2 ≤ 8 ∧ a^2 + b^2 ≤ min (-4 * a + 4 * b) 8} in
  let area : ℝ := -- Define the area calculation function in further detailed steps here.
  sorry

theorem find_area_of_M : area_of_M = 24 * Real.pi - 4 * Real.sqrt 3 :=
by
  sorry

end find_area_of_M_l443_443343


namespace find_integer_divisible_by_24_l443_443967

theorem find_integer_divisible_by_24 : 
  ∃ n : ℕ, (n % 24 = 0) ∧ (9 < real.sqrt (real.cbrt n)) ∧ (real.sqrt (real.cbrt n) < 9.1) := 
by
  let n := 744
  use n
  have h1 : n % 24 = 0 := by norm_num
  have h2 : 9 < real.sqrt (real.cbrt n) := by norm_num
  have h3 : real.sqrt (real.cbrt n) < 9.1 := by norm_num
  exact ⟨h1, h2, h3⟩

end find_integer_divisible_by_24_l443_443967


namespace estimate_probability_l443_443133

-- Define the data given in the problem
def number_of_shots := [20, 80, 100, 200, 400, 800, 1000]
def hits_above_9_rings := [18, 68, 82, 166, 330, 664, 832]
def frequencies := [0.90, 0.85, 0.82, 0.83, 0.825, 0.83, 0.832]

-- Problem statement: Estimate the probability of hitting above 9 rings in one shot
theorem estimate_probability :
  ∃ p : ℝ, (∀ f ∈ frequencies, abs (p - f) < 0.1) → (abs (p - 0.83) < 0.01) := 
sorry

end estimate_probability_l443_443133


namespace expression_value_at_two_l443_443096

theorem expression_value_at_two : (a : ℝ) (ha : a = 2) → (3 * a⁻² + a⁻² / 3) / a² = 5 / 24 :=
by
  intro a ha
  sorry

end expression_value_at_two_l443_443096


namespace cloth_gain_percentage_l443_443492

theorem cloth_gain_percentage 
  (x : ℝ) -- x represents the cost price of 1 meter of cloth
  (CP : ℝ := 30 * x) -- CP of 30 meters of cloth
  (profit : ℝ := 10 * x) -- profit from selling 30 meters of cloth
  (SP : ℝ := CP + profit) -- selling price of 30 meters of cloth
  (gain_percentage : ℝ := (profit / CP) * 100) : 
  gain_percentage = 33.33 := 
sorry

end cloth_gain_percentage_l443_443492


namespace shortest_distance_to_line_l443_443888

-- Define the conditions in Cartesian coordinates
def Circle (x y : ℝ) : Prop := (x^2 + (y + 2)^2 = 4)
def Line (x y : ℝ) : Prop := (x + y - 2 = 0)

-- Define the shortest distance calculation
noncomputable def shortestDistance : ℝ := ((|0 - 2 - 2|) / Real.sqrt 2) - 2

theorem shortest_distance_to_line :
  ∀ P : ℝ × ℝ, Circle P.1 P.2 → ∃ d : ℝ, d = shortestDistance := 
by
  intro P hP
  use shortestDistance
  -- Skip the proof here
  sorry

end shortest_distance_to_line_l443_443888


namespace factor_by_resultant_is_three_l443_443524

theorem factor_by_resultant_is_three
  (x : ℕ) (f : ℕ) (h1 : x = 7)
  (h2 : (2 * x + 9) * f = 69) :
  f = 3 :=
sorry

end factor_by_resultant_is_three_l443_443524


namespace students_posted_one_photo_l443_443503

variables (students photos : ℕ) (grades : ℕ → ℕ) 

-- Conditions
def total_students : ℕ := 50
def total_photos : ℕ := 60
def minimum_photos_per_student : Prop := ∀ n, 1 ≤ grades n
def equal_photos_in_same_grade : Prop := ∀ i j, i ≠ j → grades i ≠ grades j
def different_photos_in_different_grades : Prop := ∀ i j, i ≠ j → grades i ≠ grades j

-- Theorem stating the number of students who posted exactly one photo
theorem students_posted_one_photo :
  minimum_photos_per_student grades → 
  equal_photos_in_same_grade grades → 
  different_photos_in_different_grades grades →
  ∑ i in finset.range total_students, grades i = total_photos →
  (finset.filter (λ (n : ℕ), grades n = 1) (finset.range total_students)).card = 46 :=
by sorry

end students_posted_one_photo_l443_443503


namespace tangent_lines_to_origin_l443_443615

open Real

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x^2 + 2 * x

noncomputable def f' (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 2

theorem tangent_lines_to_origin :
  ∃ (m1 m2 : ℝ), m1 = 2 ∧ m2 = -1 / 4 ∧ ∀ x,
    (tangent_at_origin : (f'(0) = 2 ∧ f 0 = 0) ∨ (∃ x0 : ℝ, f'(x0) = m2 ∧ f x0 / x0 = -1 / 4 ∧ x0 ≠ 0)) :=
by
  sorry

end tangent_lines_to_origin_l443_443615


namespace find_a_l443_443258

noncomputable def curve1 : ℝ → ℝ := λ x, x + Real.log x
noncomputable def curve2 (a : ℝ) : ℝ → ℝ := λ x, a * x^2 + (2 * a + 3) * x + 1

theorem find_a :
  ∀ a : ℝ,
  (∃ x : ℝ, curve1 x = (2 * x - 1) ∧ curve1 x = curve2 a x) →
  a = 0 ∨ a = 1 / 2 :=
by
  sorry

end find_a_l443_443258


namespace distance_to_grandma_l443_443377

-- Definitions based on the conditions
def miles_per_gallon : ℕ := 20
def gallons_needed : ℕ := 5

-- The theorem statement to prove the distance is 100 miles
theorem distance_to_grandma : miles_per_gallon * gallons_needed = 100 := by
  sorry

end distance_to_grandma_l443_443377


namespace diagonal_difference_l443_443584

theorem diagonal_difference :
  let original_matrix := [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19], [20, 21, 22, 23, 24], [25, 26, 27, 28, 29]]
  let updated_matrix := [[5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [19, 18, 17, 16, 15], [20, 21, 22, 23, 24], [29, 28, 27, 26, 25]]
  let main_diagonal_sum := (updated_matrix 0 0) + (updated_matrix 1 1) + (updated_matrix 2 2) + (updated_matrix 3 3) + (updated_matrix 4 4)
  let anti_diagonal_sum := (updated_matrix 0 4) + (updated_matrix 1 3) + (updated_matrix 2 2) + (updated_matrix 3 1) + (updated_matrix 4 0)
  abs (anti_diagonal_sum - main_diagonal_sum) = 8 :=
by
  sorry

end diagonal_difference_l443_443584


namespace solve_frac_equation_l443_443400

def complex_solutions_of_equation : Prop :=
  ∀ x : ℂ, (3 * x^2 - 1) / (4 * x - 4) = 2 / 3 ↔ 
          x = (8 / 18 : ℂ) + (complex.I * (real.sqrt 116) / 18) ∨ 
          x = (8 / 18 : ℂ) - (complex.I * (real.sqrt 116) / 18)

theorem solve_frac_equation :
  complex_solutions_of_equation := 
  sorry

end solve_frac_equation_l443_443400


namespace rise_in_water_level_l443_443512

noncomputable def edge : ℝ := 15.0
noncomputable def base_length : ℝ := 20.0
noncomputable def base_width : ℝ := 15.0
noncomputable def volume_cube : ℝ := edge ^ 3
noncomputable def base_area : ℝ := base_length * base_width

theorem rise_in_water_level :
  (volume_cube / base_area) = 11.25 :=
by
  sorry

end rise_in_water_level_l443_443512


namespace maximum_value_is_17_l443_443025

noncomputable def maximum_expression_value (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : ℕ :=
  max (180 / z - 1)

theorem maximum_value_is_17 (x y z : ℕ) (h₁ : 10 ≤ x ∧ x < 100) (h₂ : 10 ≤ y ∧ y < 100) (h₃ : 10 ≤ z ∧ z < 100)
  (h₄ : x + y + z = 180) : maximum_expression_value x y z h₁ h₂ h₃ h₄ = 17 :=
  sorry

end maximum_value_is_17_l443_443025


namespace downstream_distance_correct_l443_443507

-- Definitions based on the conditions
def still_water_speed : ℝ := 22
def stream_speed : ℝ := 5
def travel_time : ℝ := 3

-- The effective speed downstream is the sum of the still water speed and the stream speed
def effective_speed_downstream : ℝ := still_water_speed + stream_speed

-- The distance covered downstream is the product of effective speed and travel time
def downstream_distance : ℝ := effective_speed_downstream * travel_time

-- The theorem to be proven
theorem downstream_distance_correct : downstream_distance = 81 := by
  sorry

end downstream_distance_correct_l443_443507


namespace water_speed_l443_443529

theorem water_speed (swim_speed : ℝ) (time : ℝ) (distance : ℝ) (v : ℝ) 
  (h1: swim_speed = 10) (h2: time = 2) (h3: distance = 12) 
  (h4: distance = (swim_speed - v) * time) : 
  v = 4 :=
by
  sorry

end water_speed_l443_443529


namespace area_of_sin_x_segment_l443_443811

theorem area_of_sin_x_segment : 
  ∫ x in 0..(Real.pi / 2), Real.sin x = ∫ x in 0..(Real.pi / 2), Real.sin x :=
by
  sorry

end area_of_sin_x_segment_l443_443811
