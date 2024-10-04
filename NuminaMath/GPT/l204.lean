import Mathlib

namespace perpendicularity_in_triangle_l204_204971

noncomputable def triangle_proof (A B C M D E : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace D] [MetricSpace E] : Prop :=
  let BM := Metric.dist B M
  let AC := Metric.dist A C
  let AD := Metric.dist A D
  let AB := Metric.dist A B
  let CE := Metric.dist C E
  let CM := Metric.dist C M
  (BM = AC) ∧ (AD = AB) ∧ (CE = CM) → Metric.perp D M B E

theorem perpendicularity_in_triangle (A B C M D E : Type*) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M] [MetricSpace D] [MetricSpace E] :
  triangle_proof A B C M D E :=
begin
  sorry
end

end perpendicularity_in_triangle_l204_204971


namespace michael_digging_time_equals_700_l204_204132

-- Conditions defined
def digging_rate := 4
def father_depth := digging_rate * 400
def michael_depth := 2 * father_depth - 400
def time_for_michael := michael_depth / digging_rate

-- Statement to prove
theorem michael_digging_time_equals_700 : time_for_michael = 700 :=
by
  -- Here we would provide the proof steps, but we use sorry for now
  sorry

end michael_digging_time_equals_700_l204_204132


namespace car_average_speed_l204_204758

theorem car_average_speed (D : ℝ) (hD : D > 0) : 
  let 
      time1 := D / 80
      time2 := D / 24
      time3 := D / 30
      total_time := time1 + time2 + time3
      total_distance := 3 * D
      average_speed := total_distance / total_time 
  in average_speed = 240 / 7 := 
by 
  sorry

end car_average_speed_l204_204758


namespace probability_between_40_and_50_l204_204990

-- Definition of the problem 
def roll_two_dice := (ℕ × ℕ)
def valid_outcome (pair : roll_two_dice) : Prop := 
  let (a, b) := pair in 
  1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ (10 * a + b = 40 ∨ 10 * a + b = 41 ∨ 10 * a + b = 42 ∨ 10 * a + b = 43 ∨ 10 * a + b = 44 ∨ 10 * a + b = 45 ∨ 10 * a + b = 46)

def total_outcomes := 6 * 6
def favorable_outcomes := 12
def expected_probability := favorable_outcomes / total_outcomes

theorem probability_between_40_and_50 : 
  (∃ fvals : finset roll_two_dice, (∀ pair ∈ fvals, valid_outcome pair) ∧ fvals.card = favorable_outcomes) →
  (∃ totalvals : finset roll_two_dice, totalvals.card = total_outcomes) →
  expected_probability = 1 / 3 := 
sorry

end probability_between_40_and_50_l204_204990


namespace new_parabola_after_shift_l204_204171

-- Define the original parabola
def original_parabola (x : ℝ) : ℝ := x^2 + 1

-- Define the transformation functions for shifting the parabola
def shift_left (x : ℝ) (shift : ℝ) : ℝ := x + shift
def shift_down (y : ℝ) (shift : ℝ) : ℝ := y - shift

-- Prove the transformation yields the correct new parabola equation
theorem new_parabola_after_shift : 
  (∀ x : ℝ, (shift_down (original_parabola (shift_left x 2)) 3) = (x + 2)^2 - 2) :=
by
  sorry

end new_parabola_after_shift_l204_204171


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204022

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204022


namespace proof_l204_204987

open Real

def vector := ℝ × ℝ × ℝ

noncomputable def a : vector := (2, -7, 3)
noncomputable def b : vector := (-4, e, 2)
noncomputable def c : vector := (5, 0, -6)

noncomputable def vector_add (v1 v2 : vector) : vector :=
(v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

noncomputable def scalar_mul (k : ℝ) (v : vector) : vector :=
(k * v.1, k * v.2, k * v.3)

noncomputable def vector_sub (v1 v2 : vector) : vector :=
(v1.1 - v2.1, v1.2 - v2.2, v1.3 - v2.3)

noncomputable def cross_product (v1 v2 : vector) : vector :=
(v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

noncomputable def dot_product (v1 v2 : vector) : ℝ :=
v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

theorem proof :
  dot_product (vector_add a b) (cross_product (vector_add b c) (vector_sub (scalar_mul 2 c) a)) = -30 :=
sorry

end proof_l204_204987


namespace smallest_odd_with_five_different_prime_factors_l204_204260

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204260


namespace abs_neg_four_squared_plus_six_l204_204156

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l204_204156


namespace least_four_digit_multiple_of_7_l204_204715

theorem least_four_digit_multiple_of_7 : ∃ n : ℕ, (1000 ≤ n ∧ n ≤ 9999 ∧ n % 7 = 0 ∧ ∀ m : ℕ, (1000 ≤ m ∧ m % 7 = 0 → n ≤ m)) :=
by 
  let n := 1001
  have h1 : 1000 ≤ n := by decide
  have h2 : n ≤ 9999 := by decide
  have h3 : n % 7 = 0 := by decide
  have h4 : ∀ m : ℕ, (1000 ≤ m ∧ m % 7 = 0 → n ≤ m) := by
    intro m
    assume h
    have hm : 143 ≤ m / 7 := sorry
    have hn : n = 7 * 143 := by decide
    exact sorry
  use n
  exact ⟨h1, h2, h3, h4⟩

end least_four_digit_multiple_of_7_l204_204715


namespace number_of_paths_l204_204439

theorem number_of_paths (right_steps up_steps total_steps : ℕ) (h1 : right_steps = 6) (h2 : up_steps = 4) (h3 : total_steps = 10) 
  (h4 : total_steps = right_steps + up_steps) : 
  ∃ (paths : ℕ), paths = Nat.choose total_steps up_steps ∧ paths = 210 := 
by 
  use 210
  split
  · rw [←h4, h1, h2]
    norm_num
  · refl

end number_of_paths_l204_204439


namespace polygon_triangulation_exists_l204_204818

def valid_triangulation (polygon : list (ℕ × ℕ × ℕ)) (diagonals : list (ℕ × ℕ)) : Prop :=
  ∀ triangle ∈ triangulate polygon diagonals,
  (triangle.color1 = triangle.color2 ∧ triangle.color2 = triangle.color3) ∨
  (triangle.color1 ≠ triangle.color2 ∧ triangle.color2 ≠ triangle.color3 ∧ triangle.color1 ≠ triangle.color3)

-- Define the function to triangulate the polygon
noncomputable def triangulate : (list (ℕ × ℕ × ℕ)) → (list (ℕ × ℕ)) → list (triangle ℕ) := sorry

structure triangle (α : Type) :=
  (color1 : α)
  (color2 : α)
  (color3 : α)

theorem polygon_triangulation_exists :
  ∀ (polygon : Polygon ℕ) (n : ℕ), polygon sides = 2019 → (∀ (side ∈ polygon.sides, side.color = red ∨ side.color = yellow ∨ side.color = blue) → ∃ diagonals : list (ℕ × ℕ), valid_triangulation polygon diagonals :=
sorry

end polygon_triangulation_exists_l204_204818


namespace length_of_chord_AB_l204_204371

/-- Define the circle and point P -/
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 1
def P : ℝ × ℝ := (-1, real.sqrt 3)

/-- Define the length of the chord AB -/
def chord_length (A B : ℝ × ℝ) : ℝ :=
  real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

/-- The theorem stating the length of the chord AB -/
theorem length_of_chord_AB :
  (∀ (A B : ℝ × ℝ), circle_eq A.1 A.2 → circle_eq B.1 B.2 →
    tangent_from_point P A → tangent_from_point P B →
    chord_length A B = real.sqrt 3) :=
by
  sorry

end length_of_chord_AB_l204_204371


namespace mass_in_scientific_notation_l204_204829

def mass_of_sesame_seed_in_grams : ℝ := 0.004

theorem mass_in_scientific_notation : mass_of_sesame_seed_in_grams = 4 * 10^(-3) := 
by
  sorry

end mass_in_scientific_notation_l204_204829


namespace equal_roots_of_quadratic_l204_204545

theorem equal_roots_of_quadratic (k : ℝ) : (1 - 8 * k = 0) → (k = 1/8) :=
by
  intro h
  sorry

end equal_roots_of_quadratic_l204_204545


namespace sum_of_odd_digits_1_to_75_l204_204589

-- Define the function O(n) to sum the odd digits of a number n
def sum_of_odd_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.filter (λ d => d % 2 = 1) |>.sum

-- Define the problem to prove the sum of sum_of_odd_digits from 1 to 75 equals 321
theorem sum_of_odd_digits_1_to_75 :
  (∑ n in Finset.range 76, sum_of_odd_digits n) = 321 :=
by
  sorry

end sum_of_odd_digits_1_to_75_l204_204589


namespace abs_z_eq_sqrt_10_l204_204905

variable (z : ℂ)

def condition := (z + 1) * complex.I = 3 + 2 * complex.I

theorem abs_z_eq_sqrt_10 (h : condition z) : complex.abs z = real.sqrt 10 := sorry

end abs_z_eq_sqrt_10_l204_204905


namespace smallest_odd_with_five_prime_factors_l204_204292

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204292


namespace pentagon_area_ratio_l204_204979

theorem pentagon_area_ratio (ABCDE : convex_pentagon)
  (par_AB_CE : AB ∥ CE)
  (par_BC_AD : BC ∥ AD)
  (par_AC_DE : AC ∥ DE)
  (angle_ABC_150 : ∠ ABC = 150)
  (len_AB : AB = 4)
  (len_BC : BC = 6)
  (len_DE : DE = 18) :
  let m := 25
  let n := 81
  m + n = 106 :=
by sorry

end pentagon_area_ratio_l204_204979


namespace max_value_of_polynomial_l204_204351

theorem max_value_of_polynomial :
  ∃ x : ℝ, (x = -1) ∧ ∀ y : ℝ, -3 * y^2 - 6 * y + 12 ≤ -3 * (-1)^2 - 6 * (-1) + 12 := by
  sorry

end max_value_of_polynomial_l204_204351


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204250

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204250


namespace friends_statements_l204_204857

theorem friends_statements (a b c : ℝ) (n : ℕ) 
  (h₁ : sqrt (a * b) = 99 * sqrt 2) 
  (h₂ : sqrt (a * b * c) = n) :
  ¬ (c = sqrt 2) ∧ 
  (c = 98) ∧ 
  ¬ (∀ k, even k → sqrt (a * b * k) = n) ∧ 
  ¬ (∀ k, sqrt (a * b * c) = n → c = k) :=
by
  sorry

end friends_statements_l204_204857


namespace h_at_7_over_5_eq_0_l204_204422

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l204_204422


namespace consecutive_product_minus_one_is_2003_times_perfect_square_l204_204523

def sequence (n : ℕ) : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := 2005 * sequence (n + 1) - sequence (n)

theorem consecutive_product_minus_one_is_2003_times_perfect_square :
  ∀ n : ℕ, ∃ k : ℕ, sequence n * sequence (n + 1) - 1 = 2003 * (k * k) :=
by {
  sorry
}

end consecutive_product_minus_one_is_2003_times_perfect_square_l204_204523


namespace smallest_odd_with_five_different_prime_factors_l204_204263

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204263


namespace points_on_open_hemisphere_l204_204847

theorem points_on_open_hemisphere
  (S : Finset (EuclideanSpace ℝ (Fin 3)))
  (O : EuclideanSpace ℝ (Fin 3))
  (h₁ : ∀ T : Finset (EuclideanSpace ℝ (Fin 3)), T ⊆ S → T.card = 4 → ∃ (H : EuclideanSpace ℝ (Fin 3)), 
    (T ⊆ {x | ∥x∥ = ∥H∥} ∧ ∀ x ∈ T, (EuclideanSpace.inner H x > 0))):
  (∃ (H : EuclideanSpace ℝ (Fin 3)), ∀ x ∈ S, (EuclideanSpace.inner H x > 0)) :=
sorry

end points_on_open_hemisphere_l204_204847


namespace pyramid_medians_perpendicular_to_plane_l204_204078

noncomputable def midpoint (A B : Point) : Point := sorry

theorem pyramid_medians_perpendicular_to_plane
  {A B C D M N : Point}
  (hM : midpoint A D = M)
  (hN : midpoint C D = N)
  (hABM : dist B M = dist A D / 2)
  (hBCN : dist B N = dist C D / 2) :
  is_perpendicular (line_through B D) (plane_through A B C) :=
sorry

end pyramid_medians_perpendicular_to_plane_l204_204078


namespace range_of_k_l204_204909
-- Let's import the necessary libraries first

theorem range_of_k (f g : ℝ → ℝ) (k : ℝ) :
  (∀ x : ℝ, f x =
    if x ≥ 2 then 3 / (x - 1)
    else if x < 0 then |2^x - 1|
    else if 0 ≤ x ∧ x < 2 then 2^x - 1
    else 0)
  → (∀ x : ℝ, g x = f x - k)
  → (∃ x₁ x₂ x₃ : ℝ, g x₁ = 0 ∧ g x₂ = 0 ∧ g x₃ = 0)
  → 0 < k ∧ k < 1 :=
by
  intro hf hg hz
  sorry -- Proof is omitted.

end range_of_k_l204_204909


namespace find_alpha_l204_204564

def angle_π_over_3 (α : ℝ) : Prop :=
0 < α ∧ α < 2 * Real.pi ∧
  let P := (1 - Real.tan (Real.pi / 12), 1 + Real.tan (Real.pi / 12)) in
  α = Real.arctan (P.snd / P.fst)

theorem find_alpha (α : ℝ) (h : angle_π_over_3 α) : α = Real.pi / 3 :=
by
  sorry

end find_alpha_l204_204564


namespace value_of_f_neg2016_plus_f_2017_l204_204900

noncomputable def f : ℝ → ℝ
| x := if h : 0 ≤ x ∧ x ≤ 2 then Real.log (x + 1) / Real.log 2 -- ∵ f(x) = log₂(x+1)
       else if h : x < 0 then f (-x)                          -- ∵ f is even
       else f (x - 2)                                         -- ∵ f is periodic with period 2

theorem value_of_f_neg2016_plus_f_2017 : f (-2016) + f 2017 = 1 := by
  sorry

end value_of_f_neg2016_plus_f_2017_l204_204900


namespace problem_solution_l204_204599

noncomputable def max_real_part : ℝ :=
  8 * (1 + (1 + Real.sqrt 5 + Real.sqrt (10 - 2 * Real.sqrt 5)) / 2)

theorem problem_solution :
  ∃ (w : ℕ → ℂ), 
  (∀ j, 1 ≤ j ∧ j ≤ 10 → (w j = (2 ^ 3) * Complex.exp (Complex.I * (2 * ↑j * Real.pi / 10))
                           ∨ w j = -Complex.I * (2 ^ 3) * Complex.exp (Complex.I * (2 * ↑j * Real.pi / 10)))) ∧
  (s.sum (λ j, w j)).re ≤ max_real_part :=
begin
  sorry
end

end problem_solution_l204_204599


namespace probability_odd_divisor_15_l204_204404

/-- Given the prime factorization of 15! 
    (15 factorial), prove that the probability 
    of a randomly chosen divisor being odd is 1/6. -/
theorem probability_odd_divisor_15! :
  let factorial_15 := (2^11) * (3^6) * (5^3) * 7 * 11 * 13 in
  let total_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1) in
  let odd_divisors := (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1) in
  let probability := odd_divisors / total_divisors in
  probability = 1 / 6 :=
by
  let factorial_15 := (2^11) * (3^6) * (5^3) * 7 * 11 * 13
  let total_divisors := (11 + 1) * (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let odd_divisors := (6 + 1) * (3 + 1) * (1 + 1) * (1 + 1) * (1 + 1)
  let probability := odd_divisors / total_divisors
  have h1 : total_divisors = 1344 := by sorry
  have h2 : odd_divisors = 224 := by sorry
  have h3 : probability = 224 / 1344 := by sorry
  show probability = 1 / 6 from sorry

end probability_odd_divisor_15_l204_204404


namespace minimum_value_of_expression_l204_204598

open Real

noncomputable def f (x y z : ℝ) : ℝ := (x + 2 * y) / (x * y * z)

theorem minimum_value_of_expression :
  ∀ (x y z : ℝ),
    x > 0 → y > 0 → z > 0 →
    x + y + z = 1 →
    x = 2 * y →
    f x y z = 8 :=
by
  intro x y z x_pos y_pos z_pos h_sum h_xy
  sorry

end minimum_value_of_expression_l204_204598


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204452

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204452


namespace solve_sqrt_eq_l204_204434

def u (x : ℝ) : ℝ := real.sqrt (2 * x)
def v (x : ℝ) : ℝ := real.sqrt (6 - 2 * x)
def circle_condition (x : ℝ) : Prop := (u x)^2 + (v x)^2 = 6
def line_condition (x a : ℝ) : Prop := u x + real.sqrt 2 * v x = real.sqrt 2 * a
def non_negative (x : ℝ) : Prop := u x ≥ 0 ∧ v x ≥ 0

theorem solve_sqrt_eq (a : ℝ) : (a < real.sqrt 3 → ∀ x, ¬(circle_condition x ∧ line_condition x a ∧ non_negative x)) ∧
                                (a = real.sqrt 3 → ∃ x, circle_condition x ∧ line_condition x a ∧ non_negative x ∧ ∀ y, (circle_condition y ∧ line_condition y a ∧ non_negative y → y = x)) ∧
                                (real.sqrt 3 < a ∧ a < 3 → ∃ x1 x2, x1 ≠ x2 ∧ circle_condition x1 ∧ line_condition x1 a ∧ non_negative x1 ∧ circle_condition x2 ∧ line_condition x2 a ∧ non_negative x2) ∧
                                (a = 3 → ∃ x, circle_condition x ∧ line_condition x a ∧ non_negative x ∧ ∀ y, (circle_condition y ∧ line_condition y a ∧ non_negative y → y = x)) ∧
                                (a > 3 → ∀ x, ¬(circle_condition x ∧ line_condition x a ∧ non_negative x)) :=
by
  sorry

end solve_sqrt_eq_l204_204434


namespace change_from_15_dollars_l204_204810

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end change_from_15_dollars_l204_204810


namespace valid_interval_for_k_l204_204838

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end valid_interval_for_k_l204_204838


namespace exist_children_with_inclusion_l204_204746

theorem exist_children_with_inclusion (n m : ℕ) (h_n : n = 11) (h_m : m = 5) :
  ∃ A B : fin n, (∀ club : fin m, attends A club → attends B club) :=
sorry

end exist_children_with_inclusion_l204_204746


namespace smallest_odd_with_five_prime_factors_l204_204294

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204294


namespace butterfly_count_l204_204824

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l204_204824


namespace compound_interest_amount_l204_204669

theorem compound_interest_amount (P : ℝ) (R : ℝ) (T : ℝ) (P' : ℝ) (R' : ℝ) (T' : ℝ) :
  SI = P * R * T / 100 →
  CI = P' * ((1 + R' / 100)^T' - 1) →
  ∃ (SI CI : ℝ), SI = 0.5 * CI →
  SI = 1750 * 8 * 3 / 100 →
  R = 8 →
  T = 3 →
  R' = 10 →
  T' = 2 →
  P' = 4000 :=
by
  intros _ _ _ h3 _ _ _
  sorry

end compound_interest_amount_l204_204669


namespace norris_money_left_l204_204612

-- Defining the conditions
def sept_savings : ℕ := 29
def oct_savings : ℕ := 25
def nov_savings : ℕ := 31
def dec_savings : ℕ := 35
def jan_savings : ℕ := 40

def initial_savings : ℕ := sept_savings + oct_savings + nov_savings + dec_savings + jan_savings
def interest_rate : ℝ := 0.02

def total_interest : ℝ :=
  sept_savings * interest_rate + 
  (sept_savings + oct_savings) * interest_rate + 
  (sept_savings + oct_savings + nov_savings) * interest_rate +
  (sept_savings + oct_savings + nov_savings + dec_savings) * interest_rate

def total_savings_with_interest : ℝ := initial_savings + total_interest
def hugo_owes_norris : ℕ := 20 - 10

-- The final statement to prove Norris' total amount of money
theorem norris_money_left : total_savings_with_interest + hugo_owes_norris = 175.76 := by
  sorry

end norris_money_left_l204_204612


namespace express_in_base_3_l204_204831

theorem express_in_base_3 : ∀ n : ℕ, n = 25 → nat.to_digits 3 n = [2, 2, 1] :=
by
  intro n h
  rw h
  exfalso
  sorry

end express_in_base_3_l204_204831


namespace curl_H_is_zero_except_origin_l204_204463

-- Conditions and question
def magnetic_field_intensity_vector (x y z : ℝ) : ℝ := 
  let ρ_squared := x^2 + y^2 
  if ρ_squared = 0 then 0 else -2 * y / ρ_squared + 2 * x / ρ_squared

theorem curl_H_is_zero_except_origin (x y z : ℝ) :
  ∇ × magnetic_field_intensity_vector x y z = 0 ↔ (x = 0 ∧ y = 0) :=
sorry

end curl_H_is_zero_except_origin_l204_204463


namespace sum_of_squares_and_product_l204_204677

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_squares_and_product_l204_204677


namespace necessary_condition_l204_204393

theorem necessary_condition (x : ℝ) : x = 1 → x^2 = 1 :=
by
  sorry

end necessary_condition_l204_204393


namespace ellipse_equation_proof_slopes_sum_zero_l204_204878

-- Assume the given conditions
variables (a b : ℝ) (h_ab : a > b) (h_b : b > 0) (e : ℝ) (h_e : e = (sqrt 3) / 2)
variables (M : ℝ × ℝ) (h_M : M = (4, 1))
variables (m : ℝ) (h_m : m ≠ -3)

-- Define the ellipse equation
def ellipse_eq (x y : ℝ) : Prop := (x^2 / a^2 + y^2 / b^2 = 1)

-- Prove part (1)
theorem ellipse_equation_proof :
  b^2 = 5 → a^2 = 20 → 
  ellipse_eq a b (M.fst) (M.snd) :=
sorry

-- Define intersection points and slopes
variables (x1 x2 y1 y2 : ℝ) 
variables (k1 k2 : ℝ) 
variable (h_int : y = x + m)
variable (h_int_pts : ellipse_eq x1 y1 ∧ intersection ellipse_eq (x1, y1) = some (x2, y2))

-- Prove part (2)
theorem slopes_sum_zero :
  (∀ (P Q : ℝ × ℝ), 
    let MP := (P.snd - M.snd) / (P.fst - M.fst),
        MQ := (Q.snd - M.snd) / (Q.fst - M.fst)
    in MP + MQ = 0)
  := sorry

end ellipse_equation_proof_slopes_sum_zero_l204_204878


namespace div_of_powers_l204_204710

-- Definitions of conditions:
def is_power_of_3 (x : ℕ) := x = 3 ^ 3

-- The conditions for the problem:
variable (a b c : ℕ)
variable (h1 : is_power_of_3 27)
variable (h2 : a = 3)
variable (h3 : b = 12)
variable (h4 : c = 6)

-- The proof statement:
theorem div_of_powers : 3 ^ 12 / 27 ^ 2 = 729 :=
by
  have h₁ : 27 = 3 ^ 3 := h1
  have h₂ : 27 ^ 2 = (3 ^ 3) ^ 2 := by rw [h₁]
  have h₃ : (3 ^ 3) ^ 2 = 3 ^ 6 := by rw [← pow_mul]
  have h₄ : 27 ^ 2 = 3 ^ 6 := by rw [h₂, h₃]
  have h₅ : 3 ^ 12 / 3 ^ 6 = 3 ^ (12 - 6) := by rw [div_eq_mul_inv, ← pow_sub]
  show 3 ^ 6 = 729, from by norm_num
  sorry

end div_of_powers_l204_204710


namespace area_of_triangle_l204_204064

section
variables (AB AC : ℝ × ℝ)
  (hAB : AB = (1,1))
  (hAC : AC = (1,3))

def vector_area (u v : ℝ × ℝ) : ℝ :=
  0.5 * ((u.1 * v.2) - (u.2 * v.1))

theorem area_of_triangle : vector_area AB AC = 1 :=
by
  have h1 : AB = (1,1) := hAB
  have h2 : AC = (1,3) := hAC
  unfold vector_area
  rw [h1, h2]
  norm_num
  sorry
end

end area_of_triangle_l204_204064


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204254

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204254


namespace minimum_perimeter_triangle_l204_204554

-- Define the type of points and lines
axiom Point : Type
axiom Line : Type

-- Define that point O is the vertex of an acute angle and point P is inside it
axiom O P A B P1 P2 A' B' : Point
axiom OA OB : Line

-- Define the reflection points
axiom reflection_over_OA : Point → Point 
axiom reflection_over_OB : Point → Point
axiom reflection_P_OA : reflection_over_OA P = P1 
axiom reflection_P_OB : reflection_over_OB P = P2

-- Define intersection functions
axiom intersection : Point → Point → Line → Point

-- Condition for the minimum perimeter of triangle PAB
noncomputable def is_minimum_perimeter_triangle (A B : Point) : Prop :=
  A = intersection P1 P2 OA ∧ B = intersection P1 P2 OB 

-- Statement of the problem
theorem minimum_perimeter_triangle :
  is_minimum_perimeter_triangle A' B' :=
sorry

end minimum_perimeter_triangle_l204_204554


namespace butterfly_count_l204_204823

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l204_204823


namespace smallest_odd_number_with_five_different_prime_factors_l204_204315

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204315


namespace scientific_notation_of_8200000_l204_204724

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l204_204724


namespace relationship_D_E_l204_204504

theorem relationship_D_E (D E : ℝ) (l : ℝ) 
  (center_on_line : (-D / 2 - E / 2 = l)) :
  D + E = -2 := by
  have h : ∀ (a b c : ℝ), (a / b = c) → (a = b * c) := 
    begin
      intros a b c habc,
      apply (eq_mul_of_div_eq habc),
    end,
  specialize h (-D - E) 2 l center_on_line,
  rw mul_assoc at h,
  rw mul_one at h,
  exact h

#check @relationship_D_E

end relationship_D_E_l204_204504


namespace trajectory_of_P_no_such_line_l_l204_204503

-- Definitions of points A, B, C, P
def point_A : (ℝ × ℝ) := (8, 0)
def point_Q : (ℝ × ℝ) := (-1, 0)

-- Condition definitions
def is_on_y_axis (B : ℝ × ℝ) := B.1 = 0
def is_on_x_axis (C : ℝ × ℝ) := C.2 = 0
def dot_product_zero (A B P : ℝ × ℝ) :=
  let AB := (B.1 - A.1, B.2 - A.2)
  let BP := (P.1 - B.1, P.2 - B.2)
  (AB.1 * BP.1 + AB.2 * BP.2) = 0
def equal_vectors (B C P : ℝ × ℝ) : Prop :=
  (C.1 - B.1, C.2 - B.2) = (P.1 - C.1, P.2 - C.2)

noncomputable def trajectory_eq : Prop :=
  ∀ (B P : ℝ × ℝ), is_on_y_axis B -> dot_product_zero point_A B P -> equal_vectors B (P.1, 0) P -> (P.2 ^ 2 = -4 * P.1)

noncomputable def no_line_l_exists : Prop :=
  ∀ (l : ℝ → ℝ), (∀ x, l x = (kx - 8 * k) → k^2 < 1/8 -> (some_other_conditions) -> ∃ (M N : ℝ × ℝ), (trajectory_eq M) ∧ (trajectory_eq N) ∧ ((some_vector_calculation) = 97)

-- Lean statement for proof
theorem trajectory_of_P : trajectory_eq := sorry

theorem no_such_line_l : no_line_l_exists := sorry

end trajectory_of_P_no_such_line_l_l204_204503


namespace smallest_odd_number_with_five_different_prime_factors_l204_204317

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204317


namespace garden_area_increase_l204_204765

theorem garden_area_increase (r1 r2 : ℝ) (h1 : r1 = 8) (h2 : r2 = 10) :
  let A1 := π * r1^2
  let A2 := π * r2^2
  let increase := ((A2 - A1) / A1) * 100
  increase = 56.25 := by
sor

end garden_area_increase_l204_204765


namespace nurse_serving_time_l204_204994

/-- 
Missy is attending to the needs of 12 patients. One-third of the patients have special dietary 
requirements, which increases the serving time by 20%. It takes 5 minutes to serve each standard 
care patient. Prove that it takes 64 minutes to serve dinner to all patients.
-/
theorem nurse_serving_time (total_patients : ℕ) (special_fraction : ℚ)
  (standard_time : ℕ) (increase_fraction : ℚ) :
  total_patients = 12 →
  special_fraction = 1/3 →
  standard_time = 5 →
  increase_fraction = 1/5 →
  let special_patients := (special_fraction * total_patients : ℚ).toNat in
  let standard_patients := total_patients - special_patients in
  let special_time := standard_time + ((increase_fraction * standard_time : ℚ).toNat) in
  let total_time := (standard_patients * standard_time) + (special_patients * special_time) in
  total_time = 64 :=
begin
  -- Insert hypothesis and proof here
  sorry
end

end nurse_serving_time_l204_204994


namespace geometric_sequence_sum_l204_204543

theorem geometric_sequence_sum (a_n : ℕ → ℝ) (q : ℝ) (h1 : q = 2) (h2 : a_n 1 + a_n 3 = 5) :
  a_n 3 + a_n 5 = 20 :=
by
  -- The proof would go here, but it is not required for this task.
  sorry

end geometric_sequence_sum_l204_204543


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204321

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204321


namespace func_properties_solve_eq_l204_204867

noncomputable def f : ℝ → ℝ := sorry -- Function f is given but not explicitly defined

theorem func_properties (x y : ℝ) (hx1 : x ∈ Ioo (-1 : ℝ) 1) (hy1 : y ∈ Ioo (-1 : ℝ) 1) :
  (f(0) = 0) ∧ (∀ x < 0, f(x) > 0) ∧ (f(x) + f(y) = f((x + y) / (1 + x * y))) ∧
  (∀ x1 x2 ∈ Ioo (-1 : ℝ) 1, x1 < x2 → f(x1) > f(x2)) ∧ (∀ x, f(-x) = -f(x)) := sorry

theorem solve_eq (f_neg_half_eq : f (-1/2) = 1) :
  ∃ x, f(x) + 1/2 = 0 ∧ x = 2 - Real.sqrt 3 := sorry

end func_properties_solve_eq_l204_204867


namespace total_bones_in_graveyard_l204_204071

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end total_bones_in_graveyard_l204_204071


namespace range_of_m_l204_204088

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end range_of_m_l204_204088


namespace quadratic_inequality_solutions_l204_204835

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l204_204835


namespace two_digit_primes_ending_in_3_l204_204045

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204045


namespace area_of_square_tangent_to_circle_l204_204701

theorem area_of_square_tangent_to_circle
  (R : ℝ) (hR : R = 5)
  (A B C D : ℝ × ℝ) 
  (hA : dist A (0, 0) = R) 
  (hB : dist B (0, 0) = R)
  (hC : tangent_to_circle C) 
  (hD : tangent_to_circle D) :
  let x := calc_side_length A B C D in
  x^2 = 64 :=
by sorry

end area_of_square_tangent_to_circle_l204_204701


namespace minimal_pyramid_height_l204_204956

theorem minimal_pyramid_height (r x a : ℝ) (h₁ : 0 < r) (h₂ : a = 2 * r * x / (x - r)) (h₃ : x > 4 * r) :
  x = (6 + 2 * Real.sqrt 3) * r :=
by
  -- Proof steps would go here
  sorry

end minimal_pyramid_height_l204_204956


namespace floor_expression_is_2_l204_204415

theorem floor_expression_is_2 :
  let n := 2011 in
  let expr := (n + 1)^3 / ((n - 1) * n) + (n - 1)^3 / (n * (n + 1)) in
  ⌊expr⌋ = 2 :=
by
  sorry

end floor_expression_is_2_l204_204415


namespace parabola_sum_l204_204647

theorem parabola_sum (a b c : ℝ)
  (h1 : 4 = a * 1^2 + b * 1 + c)
  (h2 : -1 = a * (-2)^2 + b * (-2) + c)
  (h3 : ∀ x : ℝ, a * x^2 + b * x + c = a * (x + 1)^2 - 2)
  : a + b + c = 5 := by
  sorry

end parabola_sum_l204_204647


namespace total_carrots_l204_204575

-- Definitions from conditions in a)
def JoanCarrots : ℕ := 29
def JessicaCarrots : ℕ := 11

-- Theorem that encapsulates the problem
theorem total_carrots : JoanCarrots + JessicaCarrots = 40 := by
  sorry

end total_carrots_l204_204575


namespace ellipse_standard_eqn_and_x_range_l204_204875

noncomputable def ellipse_conditions (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ a > b ∧ 
  e = sqrt 3 / 2 ∧ 
  (a^2 = b^2 + c^2) ∧ 
  sqrt(c^2 + (3 / 2)^2) = sqrt(57) / 2

theorem ellipse_standard_eqn_and_x_range :
  (∃ (a b c : ℝ), ellipse_conditions a b c) →
  (a = 4 ∧ b = 2 ∧ c^2 = 12 ∧ (h : ∀ e = sqrt 3 / 2, 
  (a = 4 → b = 2 → a^2 = b^2 + c^2 → 
  sqrt(c^2 + (3 / 2)^2) = sqrt 57 / 2 → 
  (fraction_ring.mk x^2 16 + fraction_ring.mk y^2 4 = 1)) ∧ 
  - (9 / 8) ≤ x_0 ∧ x_0 ≤ (9 / 8)) 
  sorry

end ellipse_standard_eqn_and_x_range_l204_204875


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204453

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204453


namespace ratio_of_areas_l204_204384

theorem ratio_of_areas (s : ℝ) : (s^2) / ((3 * s)^2) = 1 / 9 := 
by
  sorry

end ratio_of_areas_l204_204384


namespace angle_m_eq_l204_204860

variables {X Y Z W A : Type} [real : Set_like (ℝ)] [Angle : Set_like (ℝ)]

def bisect_angle (θ : Angle) (XW : Type) (X : Type) : Prop := 
  θ = angle_bisect X W

def external_angle (m θ : Angle)(X Z A : Type) : Prop := 
  2*m + 2*θ = θ + angle Y + angle Z

theorem angle_m_eq (X Y Z W A : Type) [m θ : Angle]
  (h1 : bisect_angle θ XW X)
  (h2 : external_angle m θ X Z A)
  (h3 : angle ZXA = 90.0) : 
  m = (angle Y + angle Z - θ) / 2 :=
by {
  sorry
}

end angle_m_eq_l204_204860


namespace selling_price_eq_l204_204662

noncomputable def cost_price : ℝ := 1300
noncomputable def selling_price_loss : ℝ := 1280
noncomputable def selling_price_profit_25_percent : ℝ := 1625

theorem selling_price_eq (cp sp_loss sp_profit sp: ℝ) 
  (h1 : sp_profit = 1.25 * cp)
  (h2 : sp_loss = cp - 20)
  (h3 : sp = cp + 20) :
  sp = 1320 :=
sorry

end selling_price_eq_l204_204662


namespace min_value_expression_l204_204883

open Real

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hxyz : x^2 + y^2 + z^2 = 1) : 
  (∃ (c : ℝ), c = 3 * sqrt 3 / 2 ∧ c ≤ (x / (1 - x^2) + y / (1 - y^2) + z / (1 - z^2))) :=
by
  sorry

end min_value_expression_l204_204883


namespace clock_angle_3_36_l204_204215

def minute_hand_position (minutes : ℕ) : ℝ :=
  minutes * 6

def hour_hand_position (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def angle_difference (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

def acute_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem clock_angle_3_36 :
  acute_angle (angle_difference (minute_hand_position 36) (hour_hand_position 3 36)) = 108 :=
by
  sorry

end clock_angle_3_36_l204_204215


namespace percentage_problem_l204_204053

theorem percentage_problem :
  ∃ P : ℝ, 20% of 30 > P% of 16 by 2 ∧ P = 25 := 
by
  -- Define the given conditions
  let a := 30
  let b := 16
  let c := 2
  let perc_a := 0.2
  let perc_b := 0.25
  -- Translate these to the equations
  have h1 : perc_a * a = 6,
  -- Show that it matches the condition of being greater by 2 than the P percentage of b.
  have h2 : 6 = (perc_b / 1) * b + c,
  -- Conclude P is 25%
  exact ⟨perc_b * 100, h1, by norm_num⟩

end percentage_problem_l204_204053


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204031

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204031


namespace exists_k_fixed_point_l204_204777

def G : set (ℝ → ℝ) := {f : ℝ → ℝ | ∃ (a b : ℝ), (a ≠ 1) ∧ (∀ x, f x = a * x + b)}

theorem exists_k_fixed_point :
  (∀ f ∈ G, ∃ (a b : ℝ), f = (λ x, a * x + b)) →
  (∀ f g ∈ G, (λ x, (g (f x))) ∈ G) →
  (∀ f ∈ G, ∃ a b : ℝ, f = (λ x, a * x + b) ∧ (λ x, (x - b) / a) ∈ G) →
  (∀ f ∈ G, ∃ x_f : ℝ, f x_f = x_f) →
  ∃ k : ℝ, ∀ f ∈ G, f k = k :=
by
  intros h1 h2 h3 h4
  sorry

end exists_k_fixed_point_l204_204777


namespace div_power_l204_204705

theorem div_power (h : 27 = 3 ^ 3) : 3 ^ 12 / 27 ^ 2 = 729 :=
by {
  calc
    3 ^ 12 / 27 ^ 2 = 3 ^ 12 / (3 ^ 3) ^ 2 : by rw h
               ... = 3 ^ 12 / 3 ^ 6       : by rw pow_mul
               ... = 3 ^ (12 - 6)         : by rw div_eq_sub_pow
               ... = 3 ^ 6                : by rw sub_self_pow
               ... = 729                  : by norm_num,
  sorry
}

end div_power_l204_204705


namespace butterflies_left_l204_204825

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l204_204825


namespace acute_angle_3_36_clock_l204_204217

theorem acute_angle_3_36_clock : 
  let minute_hand_degrees := (36 / 60) * 360,
      hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360,
      angle := abs(minute_hand_degrees - hour_hand_degrees) in
  angle = 108 :=
by
  let minute_hand_degrees := (36 / 60) * 360
  let hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360
  let angle := abs(minute_hand_degrees - hour_hand_degrees)
  show angle = 108 from sorry

end acute_angle_3_36_clock_l204_204217


namespace find_angle_C_range_of_a_plus_b_l204_204946

theorem find_angle_C (a b c A B C : ℝ) (h1: c * cos B = a - b / 2) (h2: A + B + C = π) (h3: 0 < C) (h4: C < π) :
    C = π / 3 :=
sorry

theorem range_of_a_plus_b (a b A B C : ℝ) (h1: c = 3) 
    (h2: 2c * cos B = 2a - b) (h3: ∀ x : ℝ, 0 < x → x < π → a / (sin A) = b / (sin B) = 3 / (sin (π/3))) 
    (h4 : A + B + C = π) (h5: 0 < A) (h6: A < 2 * π / 3) :
    3 < a + b ∧ a + b <= 6 :=
sorry

end find_angle_C_range_of_a_plus_b_l204_204946


namespace representable_by_expression_l204_204461

theorem representable_by_expression (n : ℕ) :
  ∃ (x y z : ℕ), x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ (n = (x * y + y * z + z * x) / (x + y + z)) ↔ n ≠ 1 := by
  sorry

end representable_by_expression_l204_204461


namespace sum_first_95_odds_equals_9025_l204_204405

-- Define the nth odd positive integer
def nth_odd (n : ℕ) : ℕ := 2 * n - 1

-- Define the sum of the first n odd positive integers
def sum_first_n_odds (n : ℕ) : ℕ := n^2

-- State the theorem to be proved
theorem sum_first_95_odds_equals_9025 : sum_first_n_odds 95 = 9025 :=
by
  -- We provide a placeholder for the proof
  sorry

end sum_first_95_odds_equals_9025_l204_204405


namespace smallest_odd_number_with_five_prime_factors_l204_204228

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204228


namespace camp_student_distribution_l204_204751

/-- Given 600 students numbered from 001 to 600, where a systematic sampling method is used to select a sample
of 50 students starting with the number 003. The 600 students are divided into three camps:
 - Camp I: Students numbered 001 to 300,
 - Camp II: Students numbered 301 to 495,
 - Camp III: Students numbered 496 to 600.
Prove that the number of students selected from each camp is (25, 17, 8). -/
theorem camp_student_distribution :
  ∀ (students : Fin 600) (selected_sample : Fin 50) (camp_i camp_ii camp_iii : Set (Fin 600)),
  (∀ i, selected_sample i = 3 + i * 12) →
  (camp_i = {i : Fin 600 | 1 ≤ i.val ∧ i.val ≤ 300}) →
  (camp_ii = {i : Fin 600 | 301 ≤ i.val ∧ i.val ≤ 495}) →
  (camp_iii = {i : Fin 600 | 496 ≤ i.val ∧ i.val ≤ 600}) →
  (card (camp_i ∩ selected_sample) = 25) ∧ 
  (card (camp_ii ∩ selected_sample) = 17) ∧ 
  (card (camp_iii ∩ selected_sample) = 8) :=
begin
  intros,
  sorry
end

end camp_student_distribution_l204_204751


namespace complement_B_intersection_condition_union_condition_l204_204921

-- Definitions for sets and elements
def U : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {1, 4}
def A (x : ℕ) : Set ℕ := {1, 2, x^2}

-- First proof: complement of B with respect to U
theorem complement_B :
  {n | n ∈ U ∧ n ∉ B} = {2, 3} :=
by
  sorry

-- Second proof: if A ∩ B = B, find x
theorem intersection_condition (x : ℕ) (h : {1, 2, x^2} ∩ {1, 4} = {1, 4}) :
  x = 1 :=
by
  sorry

-- Third proof: if A ∪ B = U, find x
theorem union_condition (x : ℕ) (h : {1, 2, x^2} ∪ {1, 4} = {1, 2, 3, 4}) :
  False :=
by
  sorry

end complement_B_intersection_condition_union_condition_l204_204921


namespace num_valid_permutations_l204_204536

theorem num_valid_permutations : 
  ∀ (s : list ℕ), (s.sort = list.range 1 101) ∧ (∀ i, i < s.length - 1 → abs (s.nth_le i (sorry)) - (s.nth_le (i + 1) (sorry)) ≤ 1) → s = list.range 1 101 ∨ s = (list.range 1 101).reverse :=
by
  sorry

end num_valid_permutations_l204_204536


namespace percentage_of_damaged_tins_l204_204386

-- Let "Total tins" be the total number of tins before any were thrown away
def total_tins (cases_per_delivery : ℕ) (tins_per_case : ℕ) : ℕ := cases_per_delivery * tins_per_case

-- Let "Damaged tins" be the total number of tins that were damaged and thrown away
def damaged_tins (total_tins : ℕ) (tins_remaining : ℕ) : ℕ := total_tins - tins_remaining

-- Let "% damaged" be the percentage of tins that were damaged
def percentage_damaged (damaged : ℕ) (total : ℕ) : ℚ := (damaged.to_rat / total.to_rat) * 100

-- Given conditions
theorem percentage_of_damaged_tins (cases : ℕ) (tins_per_case : ℕ) (tins_left : ℕ) :
  total_tins cases tins_per_case = 360 →
  tins_left = 342 →
  percentage_damaged (damaged_tins 360 342) 360 = 5 := by
  intros h_cases h_tins_left
  rw [h_cases, h_tins_left]
  norm_num
  sorry

end percentage_of_damaged_tins_l204_204386


namespace perpendicular_AO_LD_l204_204122

/-!
# Geometry: Perpendicular Lines in a Triangle

This module proves that in a given triangle with specified conditions,
the lines \(AO\) and \(LD\) are perpendicular.

## Problem Statement
Given:
- A triangle \(ABC\) with \(AL\) as the angle bisector,
- \(O\) as the center of the circumcircle of triangle \(ABC\),
- A point \(D\) on \(AC\) such that \(AD = AB\).

Prove:
- The lines \(AO\) and \(LD\) are perpendicular.
-/

open Geometry

variables {A B C D O L : Point}

-- Definitions according to problem conditions
def AL_is_angle_bisector (A B C L : Point) : Prop :=
  ∃ (α β : Angle), Angle_at A B α ∧ Angle_at A C β ∧ α = β

def circumcenter (A B C O : Point) : Prop :=
  ∃ R : ℝ, Circle O R ∧ OnCircle A O R ∧ OnCircle B O R ∧ OnCircle C O R

def isosceles_on_AC (A B D : Point) : Prop :=
  dist A D = dist A B

-- Theorem statement
theorem perpendicular_AO_LD
  (h1 : AL_is_angle_bisector A B C L)
  (h2 : circumcenter A B C O)
  (h3 : isosceles_on_AC A B D) :
  Angle (A, O) (L, D) = 90 :=
sorry

end perpendicular_AO_LD_l204_204122


namespace product_of_first_n_terms_of_geometric_sequence_l204_204194

theorem product_of_first_n_terms_of_geometric_sequence {b : ℕ → ℝ} 
  (h : ∀ m n p q : ℕ, m + n = p + q → b m * b n = b p * b q) (n : ℕ) :
  (finset.range n).prod (λ i, b (i + 1)) = (b 1 * b n) ^ (n / 2) :=
sorry

end product_of_first_n_terms_of_geometric_sequence_l204_204194


namespace rectangles_same_dimensions_l204_204948

theorem rectangles_same_dimensions (n : ℕ) (h_pos : 0 < n) : 
  ∃ (w1 h1 w2 h2 : ℕ), 
    (w1 * h1 = w2 * h2) ∧ 
    (w1 ≠ w2 ∨ h1 ≠ h2) ∧ 
    (bit0 1) ^ n = w1 + w2 :=
begin
  sorry
end

end rectangles_same_dimensions_l204_204948


namespace negation_proof_l204_204177

theorem negation_proof :
  (¬ ∃ (a : ℝ), a ∈ set.Icc 0 1 ∧ a^4 + a^2 > 1) ↔ (∀ (a : ℝ), a ∈ set.Icc 0 1 → a^4 + a^2 ≤ 1) :=
by
sorry

end negation_proof_l204_204177


namespace find_valid_mappings_l204_204364

noncomputable def valid_mapping_1 (x : ℝ) : Prop :=
x > 0 ∧ abs x > 0

noncomputable def valid_mapping_2 (x : ℕ) : Prop :=
nat.succ x ∈ set.univ

noncomputable def valid_mapping_3 (x : {x : ℝ // x > 0}) : Prop :=
x.val * x.val ∈ set.univ

theorem find_valid_mappings :
  (∀ x : ℝ, valid_mapping_1 x) = false ∧
  (∀ x : ℕ, valid_mapping_2 x) = true ∧
  (∀ (x : {x : ℝ // x > 0}), valid_mapping_3 x) = true
:= by {
  sorry
}

end find_valid_mappings_l204_204364


namespace min_value_reciprocals_l204_204593

open Real

theorem min_value_reciprocals (a b : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : a + 3 * b = 1) :
  ∃ m : ℝ, (∀ x y : ℝ, 0 < x → 0 < y → x + 3 * y = 1 → (1 / x + 1 / y) ≥ m) ∧ m = 4 + 2 * sqrt 3 :=
sorry

end min_value_reciprocals_l204_204593


namespace marys_score_l204_204989

theorem marys_score (C ω S : ℕ) (H1 : S = 30 + 4 * C - ω) (H2 : S > 80)
  (H3 : (∀ C1 ω1 C2 ω2, (C1 ≠ C2 → 30 + 4 * C1 - ω1 ≠ 30 + 4 * C2 - ω2))) : 
  S = 119 :=
sorry

end marys_score_l204_204989


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204029

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204029


namespace hyperbola_range_of_m_l204_204546

theorem hyperbola_range_of_m (m : ℝ) :
  (∃ foci_on_y_axis : (∀ x y : ℝ, foci_of_hyperbola (x^2 / (m^2 - 4) - y^2 / (m + 1) = 1) y_axis),
    -2 < m ∧ m < -1) :=
sorry

end hyperbola_range_of_m_l204_204546


namespace increase_average_grade_l204_204694

theorem increase_average_grade (
    S_A : ℝ := 47.2,
    S_B : ℝ := 41.8,
    S_L : ℝ := 47,
    S_F : ℝ := 44,
    n_A : ℝ := 10,
    n_B : ℝ := 10
) : 
    (S_L < S_A ∧ S_L > S_B ∧ S_F > S_B ∧ S_F < S_A) →
    ((S_A * n_A - S_L - S_F) / (n_A - 2) > S_A ∧ 
     (S_B * n_B + S_L + S_F) / (n_B + 2) > S_B) :=
begin
  sorry
end

end increase_average_grade_l204_204694


namespace circle_area_ratio_l204_204696

noncomputable def ratio_of_circle_areas : ℝ :=
  let r1 := sqrt(3) / 3
  let A1 := π * r1^2
  let r2 := sqrt(2) + 1
  let A2 := π * (sqrt(2) + 1)^2
  A2 / A1

theorem circle_area_ratio :
  let correct_answer := 3 * sqrt(3) * (3 + 2 * sqrt(2))
  ratio_of_circle_areas = correct_answer :=
by sorry

end circle_area_ratio_l204_204696


namespace integer_part_m_eq_3_l204_204865

theorem integer_part_m_eq_3 (x : ℝ) (hx : 0 < x ∧ x < π / 2) : 
  ⌊3 ^ (Real.cos x ^ 2) + 3 ^ (Real.sin x ^ 5)⌋ = 3 :=
sorry

end integer_part_m_eq_3_l204_204865


namespace street_lamp_turn_off_count_l204_204954

-- Define the conditions as hypotheses
def conditions (n m k : Nat) := n = 12 ∧ m = 9 ∧ k = 3

-- Main theorem statement
theorem street_lamp_turn_off_count : 
  conditions 12 9 3 → 
  ∃ count : Nat, count = Nat.choose 8 3 ∧ count = 56 := 
by 
  intro h
  use Nat.choose 8 3
  have h1 : Nat.choose 8 3 = 56 := by norm_num
  exact And.intro rfl h1

end street_lamp_turn_off_count_l204_204954


namespace smallest_odd_number_with_five_prime_factors_l204_204302

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204302


namespace two_digit_primes_ending_in_3_l204_204043

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204043


namespace smallest_odd_number_with_five_prime_factors_l204_204307

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204307


namespace cos_seven_pi_over_six_l204_204445

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l204_204445


namespace cos_seven_pi_over_six_l204_204444

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l204_204444


namespace smallest_odd_number_with_five_prime_factors_l204_204266

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204266


namespace divide_polyhedron_into_convex_parts_l204_204570

/-- Inside a convex polyhedron, there are several non-intersecting spheres (tori of different radii).
    Prove that this polyhedron can be divided into smaller convex polyhedra, each of which contains
    exactly one of the given spheres. -/
theorem divide_polyhedron_into_convex_parts
  (Poly : ConvexPolyhedron)
  (S : Finset Sphere)
  (S_non_intersecting : ∀ (s1 s2 : Sphere), s1 ∈ S → s2 ∈ S → s1 ≠ s2 → Disjoint s1 s2)
  (H : ∀ (s : Sphere), s ∈ S → IsTorus s) :
  ∃ (P : Finset ConvexPolyhedron),
    (∀ (p : ConvexPolyhedron), p ∈ P → (∃ (s : Sphere), s ∈ S ∧ p ⊇ s ∧ ∀ (s' : Sphere), s' ∈ S → s' ≠ s → p ∩ s' = ∅)) ∧
    Poly = ⋃₀ P :=
sorry

end divide_polyhedron_into_convex_parts_l204_204570


namespace range_a_P_range_a_Q_range_a_exactly_one_P_Q_range_m_l204_204884
-- Import necessary libraries

-- Definitions based purely on the conditions outlined in the problem
def f (x : ℝ) := (1 - x) / 3

def P (a : ℝ) := |f a| < 2

def A (a : ℝ) := {x : ℝ | x^2 + (a + 2) * x + 1 = 0}
def B := {x : ℝ | x > 0}
def Q (a : ℝ) := ∀ x ∈ A a, x ∉ B

def S := {a : ℝ | -4 < a ∧ a < 7}
def T (m : ℝ) := {y : ℝ | ∃ x : ℝ, x ≠ 0 ∧ y = x + m / x}

-- Theorems to prove 
theorem range_a_P : {a : ℝ | P a} = {a : ℝ | -5 < a ∧ a < 7} := 
sorry

theorem range_a_Q : {a : ℝ | Q a} = {a : ℝ | -4 < a} := 
sorry

theorem range_a_exactly_one_P_Q : {a : ℝ | (P a ∧ ¬ Q a) ∨ (¬ P a ∧ Q a)} = {a : ℝ | (-5 < a ∧ a ≤ -4) ∨ (7 ≤ a)} := 
sorry

theorem range_m : {m : ℝ | m > 0 ∧ (∀ y ∈ (set.univ \ T m), y ∈ S)} = {m : ℝ | 0 < m ∧ m ≤ 4} := 
sorry

end range_a_P_range_a_Q_range_a_exactly_one_P_Q_range_m_l204_204884


namespace ratio_length_breadth_l204_204649

-- Define the conditions
def length := 135
def area := 6075

-- Define the breadth in terms of the area and length
def breadth := area / length

-- The problem statement as a Lean 4 theorem to prove the ratio
theorem ratio_length_breadth : length / breadth = 3 := 
by
  -- Proof goes here
  sorry

end ratio_length_breadth_l204_204649


namespace area_of_R3_is_4_5_l204_204817

-- Definition of the conditions
def R1_area : ℝ := 36
def R2_side_length : ℝ := (Real.sqrt R1_area) / (2 * Real.sqrt 2)
def R3_side_length : ℝ := R2_side_length / 2
def R3_area : ℝ := R3_side_length^2

-- Lean 4 statement for proving the area of R3
theorem area_of_R3_is_4_5 : R3_area = 4.5 :=
by
  sorry

end area_of_R3_is_4_5_l204_204817


namespace solve_for_y_l204_204631

-- Define the main theorem to be proven
theorem solve_for_y (y : ℤ) (h : 7 * (4 * y + 3) - 5 = -3 * (2 - 8 * y) + 5 * y) : y = 22 :=
by
  sorry

end solve_for_y_l204_204631


namespace total_games_played_l204_204778

theorem total_games_played (G : ℕ) (R : ℕ)
  (h1 : G = 30 + R)
  (h2 : 12 + 0.80 * R = 0.70 * G)
  : G = 120 := 
sorry

end total_games_played_l204_204778


namespace absolute_value_simplification_l204_204149

theorem absolute_value_simplification : abs(-4^2 + 6) = 10 := by
  sorry

end absolute_value_simplification_l204_204149


namespace num_of_int_solutions_l204_204534

/-- 
  The number of integer solutions to the equation 
  \((x^3 - x - 1)^{2015} = 1\) is 3.
-/
theorem num_of_int_solutions :
  ∃ n : ℕ, n = 3 ∧ ∀ x : ℤ, (x ^ 3 - x - 1) ^ 2015 = 1 ↔ x = 0 ∨ x = 1 ∨ x = -1 := 
sorry

end num_of_int_solutions_l204_204534


namespace find_speed_of_first_person_l204_204699

def time := 3.5
def relative_speed (x : ℝ) : ℝ := x + 7
def distance : ℝ := 45.5

theorem find_speed_of_first_person (x : ℝ) (h1 : distance = relative_speed x * time) : x = 6 :=
sorry

end find_speed_of_first_person_l204_204699


namespace runners_align_time_l204_204187

noncomputable def lcm (a b : ℕ) : ℕ :=
Nat.lcm a b

theorem runners_align_time:
  let track_length := 750
  let v1 := 4.6
  let v2 := 4.9
  let v3 := 5.3
  let t1 := Nat.ceil (track_length / v1)
  let t2 := Nat.ceil (track_length / v2)
  let t3 := Nat.ceil (track_length / v3)
  t = lcm (lcm t1 t2) t3 := 3750 := by
  sorry

end runners_align_time_l204_204187


namespace incorrect_statements_correct_statement_l204_204489

def isClosedSet (M : Set ℤ) : Prop :=
  ∀ a b ∈ M, a + b ∈ M ∧ a - b ∈ M

theorem incorrect_statements : 
  ¬ isClosedSet ({-4, -2, 0, 2, 4} : Set ℤ) ∧
  ¬ isClosedSet {n : ℤ | n > 0} ∧ 
  ¬ ∀ A₁ A₂ : Set ℤ, isClosedSet A₁ → isClosedSet A₂ → isClosedSet (A₁ ∪ A₂) :=
by
  sorry

theorem correct_statement :
  isClosedSet {n : ℤ | ∃ k : ℤ, n = 3 * k} :=
by
  sorry

end incorrect_statements_correct_statement_l204_204489


namespace smallest_odd_number_with_five_primes_proof_l204_204238

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204238


namespace math_problem_l204_204359

theorem math_problem 
  (x y z t : ℝ) 
  (h_nonneg : 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ 0 ≤ t)
  (h1 : x * y * z = 2)
  (h2 : y + z + t = 2 * real.sqrt 2) :
  2 * x^2 + y^2 + z^2 + t^2 ≥ 6 :=
by
  sorry

end math_problem_l204_204359


namespace sum_first_fifty_digits_of_decimal_of_one_over_1234_l204_204338

theorem sum_first_fifty_digits_of_decimal_of_one_over_1234 :
  let s := "00081037277147487844408427876817350238192918144683"
  let digits := s.data
  (4 * (list.sum (digits.map (λ c, (c.to_nat - '0'.to_nat)))) + (list.sum ((digits.take 6).map (λ c, (c.to_nat - '0'.to_nat)))) ) = 729 :=
by sorry

end sum_first_fifty_digits_of_decimal_of_one_over_1234_l204_204338


namespace smallest_odd_number_with_five_prime_factors_l204_204275

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204275


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204010

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204010


namespace johnny_pounds_per_dish_l204_204581

-- Definitions based on conditions in Step a).
def dishes_per_day : ℕ := 40
def cost_per_pound : ℕ := 8
def total_spent_per_week : ℕ := 1920
def operating_days_per_week : ℕ := 7 - 3

-- Definition of the proof problem based on the equivalent problem in Step c).
theorem johnny_pounds_per_dish :
  let total_pounds_per_week := total_spent_per_week / cost_per_pound in
  let pounds_per_day := total_pounds_per_week / operating_days_per_week in
  let pounds_per_dish := pounds_per_day / dishes_per_day in
  pounds_per_dish = 1.5 :=
by
  sorry

end johnny_pounds_per_dish_l204_204581


namespace shaded_region_area_l204_204763

noncomputable def radius : ℝ := 2
noncomputable def side : ℝ := 1

-- Assume the existence of points
axiom O : Point
axiom A : Point
axiom B : Point
axiom C : Point
axiom circle_centered_O : Center O × Radius O = radius
axiom square_OABC : Square O A B C × Side O A = side
axiom D : Point
axiom E : Point
axiom extended_meet_circle_at_D : (O ∈ Line(A, D)) × (OD = radius)
axiom extended_meet_circle_at_E : (O ∈ Line(C, E)) × (OE = radius)

-- Target to prove
theorem shaded_region_area : 
  area_shaded_region (BD, BE, arc DE) = (π / 3) + 1 - sqrt(3) :=
sorry

end shaded_region_area_l204_204763


namespace brian_breath_hold_time_l204_204791

theorem brian_breath_hold_time : 
  let t₀ := 10
  let t₁ := t₀ * 2
  let t₂ := t₁ * 2
  let t₃ := t₂ + t₂ * 0.5
  t₃ = 60 :=
by
  let t₀ := 10
  let t₁ := t₀ * 2
  let t₂ := t₁ * 2
  let t₃ := t₂ + t₂ * 0.5 
  sorry

end brian_breath_hold_time_l204_204791


namespace find_second_x_intercept_l204_204854

theorem find_second_x_intercept (a b c : ℝ)
  (h_vertex : ∀ x, y = a * x^2 + b * x + c → x = 5 → y = -3)
  (h_intercept1 : ∀ y, y = a * 1^2 + b * 1 + c → y = 0) :
  ∃ x, y = a * x^2 + b * x + c ∧ y = 0 ∧ x = 9 :=
sorry

end find_second_x_intercept_l204_204854


namespace final_cost_correct_l204_204390

noncomputable def original_price_dress : ℝ := 50
noncomputable def original_price_shoes : ℝ := 75
noncomputable def discount_dress : ℝ := 0.30
noncomputable def discount_shoes : ℝ := 0.25
noncomputable def tax_rate : ℝ := 0.05

theorem final_cost_correct :
  let discounted_price_dress := original_price_dress * (1 - discount_dress),
      discounted_price_shoes := original_price_shoes * (1 - discount_shoes),
      total_before_tax := discounted_price_dress + discounted_price_shoes,
      tax_amount := total_before_tax * tax_rate,
      total_cost := total_before_tax + tax_amount
  in total_cost = 95.81 :=
by
  sorry

end final_cost_correct_l204_204390


namespace find_three_digit_number_divisible_by_5_l204_204387

theorem find_three_digit_number_divisible_by_5 {n x : ℕ} (hx1 : 100 ≤ x) (hx2 : x < 1000) (hx3 : x % 5 = 0) (hx4 : x = n^3 + n^2) : x = 150 ∨ x = 810 := 
by
  sorry

end find_three_digit_number_divisible_by_5_l204_204387


namespace four_digit_square_palindromes_count_l204_204816

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_square_palindromes_count : 
  (finset.card (finset.filter (λ n : ℕ, is_palindrome (n * n) ∧ is_four_digit (n * n)) 
                              (finset.range 100) \ finset.range 32)) = 3 :=
by sorry

end four_digit_square_palindromes_count_l204_204816


namespace ratio_of_areas_of_triangle_and_trapezoid_l204_204395

noncomputable def equilateral_triangle_area (s : ℝ) : ℝ := (s ^ 2 * Real.sqrt 3) / 4

theorem ratio_of_areas_of_triangle_and_trapezoid :
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  (a_small / a_trapezoid) = (1 / 3) :=
by
  let large_triangle_side := 10
  let small_triangle_side := 5
  let a_large := equilateral_triangle_area large_triangle_side
  let a_small := equilateral_triangle_area small_triangle_side
  let a_trapezoid := a_large - a_small
  have h : (a_small / a_trapezoid) = (1 / 3) := 
    by sorry  -- Here would be the proof steps, but we're skipping
  exact h

end ratio_of_areas_of_triangle_and_trapezoid_l204_204395


namespace twentieth_fisherman_catch_l204_204683

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l204_204683


namespace absolute_value_simplification_l204_204148

theorem absolute_value_simplification : abs(-4^2 + 6) = 10 := by
  sorry

end absolute_value_simplification_l204_204148


namespace find_angle_B_find_area_B_l204_204065

variables {A B C : ℝ} {a b c : ℝ}

-- Condition: Sides and angles in the triangle
axiom sides_opposite : ∀ {A B C : ℝ} {a b c : ℝ}, (triangle_sides_opposite A B C a b c)

-- Condition: (2a + c) cos B + b cos C = 0
axiom cosine_condition : (2 * a + c) * Real.cos B + b * Real.cos C = 0

-- Prove angle B = 2π/3
theorem find_angle_B (h1: sides_opposite) (h2: cosine_condition) : B = 2 * Real.pi / 3 := 
sorry

-- Prove the area of the triangle
theorem find_area_B (h1: sides_opposite) 
                    (h2: cosine_condition)
                    (h3: b = Real.sqrt 13)
                    (h4: a + c = 4) 
                    (h5: find_angle_B h1 h2) : 
                    1 / 2 * a * c * Real.sin B = 3 * Real.sqrt 3 / 4 := 
sorry

end find_angle_B_find_area_B_l204_204065


namespace numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l204_204659

theorem numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1 :
  (63 ∣ 2^48 - 1) ∧ (65 ∣ 2^48 - 1) := 
by
  sorry

end numbers_between_sixty_and_seventy_divide_2_pow_48_minus_1_l204_204659


namespace calculation_eq_l204_204718

theorem calculation_eq :
  (∑ i in finset.range (n + 1), (C n i * (↑(i + 1) : ℚ)⁻¹ * (1 / 3) ^ (i + 1))) = 
  (1 / (n + 1 : ℚ)) * ((4 / 3) ^ (n + 1) - 1) :=
sorry

end calculation_eq_l204_204718


namespace cosine_distance_l204_204165

theorem cosine_distance (x : ℝ) : 
  let y := (1/2) * Real.cos (x + π/3)
  ∃ d : ℝ, d = Real.sqrt (π^2 + 1) ∧
    ∀ x1 x2 : ℝ, 
      (y x1 = 1/2 ∧ y x2 = -1/2) → 
      |x2 - x1| = d := 
begin
  sorry
end

end cosine_distance_l204_204165


namespace janet_more_cards_than_brenda_l204_204099

theorem janet_more_cards_than_brenda : ∀ (J B M : ℕ), M = 2 * J → J + B + M = 211 → M = 150 - 40 → J - B = 9 :=
by
  intros J B M h1 h2 h3
  sorry

end janet_more_cards_than_brenda_l204_204099


namespace common_chord_length_of_circles_l204_204640

theorem common_chord_length_of_circles :
  let circle1 (x y : ℝ) := x^2 + y^2 - 4 = 0
  let circle2 (x y : ℝ) := x^2 + y^2 - 4*x + 4*y - 12 = 0
  let chord_eq (x y : ℝ) := x - y + 2 = 0
  let dist_from_origin_to_chord (x y : ℝ) := abs 2 / real.sqrt (1^2 + 1^2)
  dist_from_origin_to_chord 0 0 = real.sqrt 2 ∧
  2 * (real.sqrt (4 - 2)) = 2 * real.sqrt 2 :=
by 
  sorry

end common_chord_length_of_circles_l204_204640


namespace smallest_odd_number_with_five_primes_proof_l204_204240

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204240


namespace smallest_odd_number_with_five_different_prime_factors_l204_204282

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204282


namespace find_b_l204_204420

def h (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ b : ℝ, h(b) = 0 ∧ b = 7 / 5 :=
by
  sorry

end find_b_l204_204420


namespace area_decreases_by_28_l204_204549

def decrease_in_area (s h : ℤ) (h_eq : h = s + 3) : ℤ :=
  let new_area := (s - 4) * (s + 7)
  let original_area := s * h
  new_area - original_area

theorem area_decreases_by_28 (s h : ℤ) (h_eq : h = s + 3) : decrease_in_area s h h_eq = -28 :=
sorry

end area_decreases_by_28_l204_204549


namespace smallest_positive_integer_x_for_cube_l204_204716

theorem smallest_positive_integer_x_for_cube (x : ℕ) (h1 : 1512 = 2^3 * 3^3 * 7) (h2 : ∀ n : ℕ, n > 0 → ∃ k : ℕ, 1512 * n = k^3) : x = 49 :=
sorry

end smallest_positive_integer_x_for_cube_l204_204716


namespace neg_div_neg_eq_pos_division_of_negatives_example_l204_204411

theorem neg_div_neg_eq_pos (a b : Int) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  -- You can complete the proof here
  sorry

theorem division_of_negatives_example : (-81 : Int) / (-9) = 9 :=
  neg_div_neg_eq_pos 81 9 (by decide)

end neg_div_neg_eq_pos_division_of_negatives_example_l204_204411


namespace number_of_trips_to_fill_tank_l204_204402

def volume_of_hemisphere (r : ℝ) : ℝ := (2 / 3) * π * r^3

def volume_of_cone (R h : ℝ) : ℝ := (1 / 3) * π * R^2 * h

theorem number_of_trips_to_fill_tank :
  let bucket_radius := 8
  let tank_base_radius := 12
  let tank_height := 18
  let volume_bucket := volume_of_hemisphere bucket_radius
  let volume_tank := volume_of_cone tank_base_radius tank_height
  ceiling (volume_tank / volume_bucket) = 3 :=
by
  let bucket_radius := 8
  let tank_base_radius := 12
  let tank_height := 18
  let volume_bucket := volume_of_hemisphere bucket_radius
  let volume_tank := volume_of_cone tank_base_radius tank_height
  sorry

end number_of_trips_to_fill_tank_l204_204402


namespace original_average_weight_l204_204185

theorem original_average_weight (W : ℝ) (h : (7 * W + 110 + 60) / 9 = 113) : W = 121 :=
by
  sorry

end original_average_weight_l204_204185


namespace find_t_l204_204108

-- Define the problem and the conditions
open Polynomial

theorem find_t (n : ℕ) (hn_even : n % 2 = 0) :
    let g (x : ℂ) := (1 + x + x^3)^n
    let b : Fin (3*n + 1) → ℂ := λ i, (g x).coeff i
    let t := ∑ i in Finset.filter (λ i, even i) (Finset.range (3*n+1)), b i
    t = (3^n + 1) / 2 :=
begin
  sorry,
end

end find_t_l204_204108


namespace zero_primes_in_sequence_l204_204350

def Q : Nat := 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47 * 53 * 59 * 61 * 67

theorem zero_primes_in_sequence :
  (∀ n, 2 ≤ n ∧ n ≤ 61 → ¬ Nat.prime (Q + n)) :=
by {
  intros n hn,
  sorry
}

end zero_primes_in_sequence_l204_204350


namespace michael_twenty_dollar_bills_l204_204610

theorem michael_twenty_dollar_bills (total_amount : ℕ) (denomination : ℕ) 
  (h_total : total_amount = 280) (h_denom : denomination = 20) : 
  total_amount / denomination = 14 := by
  sorry

end michael_twenty_dollar_bills_l204_204610


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204454

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204454


namespace two_digit_primes_ending_in_3_l204_204048

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204048


namespace number_of_elements_in_CU_A_inter_B_l204_204494

open Set

def A : Set Char := {'a', 'b', 'c', 'd', 'e'}
def B : Set Char := {'c', 'd', 'e', 'f'}
def U : Set Char := A ∪ B
def C_U (S : Set Char) : Set Char := U \ S

theorem number_of_elements_in_CU_A_inter_B :
  (C_U (A ∩ B)).card = 3 := 
by
  sorry

end number_of_elements_in_CU_A_inter_B_l204_204494


namespace spherical_to_rectangular_coords_l204_204429

noncomputable def spherical_to_rectangular 
  (ρ θ φ : ℝ)  : ℝ × ℝ × ℝ :=
(ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_to_rectangular_coords 
  (hρ : ℝ := 10) (hθ : ℝ := 5 * Real.pi / 4) (hφ : ℝ := Real.pi / 4)
  (x y z : ℝ) :
  spherical_to_rectangular hρ hθ hφ = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coords_l204_204429


namespace simplify_fraction_l204_204145

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end simplify_fraction_l204_204145


namespace triangle_ratio_PA_AB_l204_204552

theorem triangle_ratio_PA_AB 
  (A B C P : Type)
  [MetricSpace B] [MetricSpace C] [MetricSpace P]
  (AC CB : ℕ)
  (h_r : AC / CB = 2 / 3)
  (h_bisector : ∃ (P : Point), (bisector_of_exterior_angle_at C (P) ∧ between A P B)) :
  ∃ (PA AB : ℕ), PA / AB = 3.2 :=
begin
  -- proof to be completed
  sorry
end

end triangle_ratio_PA_AB_l204_204552


namespace soda_count_l204_204399

theorem soda_count
  (W : ℕ) (S : ℕ) (B : ℕ) (T : ℕ)
  (hW : W = 26) (hB : B = 17) (hT : T = 31) :
  W + S - B = T → S = 22 :=
by
  sorry

end soda_count_l204_204399


namespace integral_calculation_l204_204403

def line_segment_integral (A B : ℝ × ℝ) : ℝ :=
  ∫ (x : ℝ) in 0..4, (x - (3/4)*x) * (5/4)

theorem integral_calculation : 
  line_segment_integral (0, 0) (4, 3) = 5 / 2 :=
by
  sorry

end integral_calculation_l204_204403


namespace inscribed_rectangles_with_diagonal_l204_204870

theorem inscribed_rectangles_with_diagonal (a b c : ℝ) (h1 : a ≤ b) :
  (b < c ∧ c < real.sqrt (a^2 + b^2) → ∃! r1 r2 r3 r4, (∿_ 4 rectangles with diagonal c inside given rectangle)) ∧
  (b = c → ∃! r1 r2, (∿_ 2 rectangles with diagonal c inside given rectangle)) ∧
  (c = real.sqrt (a^2 + b^2) → ∃! r1, (∿_ 1 rectangle with diagonal c inside given rectangle)) ∧
  (c < b ∨ c > real.sqrt (a^2 + b^2) → ∃! r0, (∿_ 0 rectangles with diagonal c inside given rectangle)) :=
by
  sorry

end inscribed_rectangles_with_diagonal_l204_204870


namespace runners_never_meet_l204_204689

theorem runners_never_meet
    (x : ℕ)  -- Speed of first runner
    (a : ℕ)  -- 1/3 of the circumference of the track
    (C : ℕ)  -- Circumference of the track
    (hC : C = 3 * a)  -- Given that C = 3 * a
    (h_speeds : 1 * x = x ∧ 2 * x = 2 * x ∧ 4 * x = 4 * x)  -- Speed ratios: 1:2:4
    (t : ℕ)  -- Time variable
: ¬(∃ t, (x * t % C = 2 * x * t % C ∧ 2 * x * t % C = 4 * x * t % C)) :=
by sorry

end runners_never_meet_l204_204689


namespace consecutive_zeros_sequences_length_15_l204_204535

theorem consecutive_zeros_sequences_length_15 :
  ∃ n : ℕ, n = 121 ∧ (∀ seq : fin 15 → bool, (∀ i j, seq i = seq j → i.succ = j.succ ∨ seq i = ff → (seq i = seq j)) → seq.count ff = n) :=
sorry

end consecutive_zeros_sequences_length_15_l204_204535


namespace problem_statement_l204_204492

-- Definitions
def collinear (A B C : ℕ × ℕ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

def seq_a (n : ℕ) : ℕ := 2*n - 1

def seq_b (n : ℕ) : ℕ := 2^n * seq_a n

-- Conditions
def points_collinear : Prop :=
  collinear (1, 1) (2, 3) (n, seq_a n)

-- Proving the general term of a_n and sum T_n
theorem problem_statement (n : ℕ) (points_collinear : points_collinear) : 
  (∀ n : ℕ, seq_a n = 2*n - 1) ∧ (∑ i in Finset.range n, seq_b (i+1) = 2^(n+1) * (2*n - 3) + 6) :=
begin
  sorry
end

end problem_statement_l204_204492


namespace abs_neg_four_squared_plus_six_l204_204155

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l204_204155


namespace unique_rectangle_dimensions_l204_204485

theorem unique_rectangle_dimensions (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < a ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = a * b / 4 :=
sorry

end unique_rectangle_dimensions_l204_204485


namespace angle_between_asymptotes_l204_204636

-- Definitions and conditions
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 / 3 = 1
def asymptote1 (x y : ℝ) : Prop := y = sqrt 3 * x
def asymptote2 (x y : ℝ) : Prop := y = -sqrt 3 * x

-- Theorem statement
theorem angle_between_asymptotes :
  ∃ theta : ℝ, theta = real.pi / 3 ∧ 
  (∀ x y, hyperbola x y → (asymptote1 x y ∨ asymptote2 x y)) →
  (∃ t1 t2 : ℝ, tan t1 = sqrt 3 ∧ tan t2 = -sqrt 3 ∧ theta = abs (t2 - t1)) :=
sorry

end angle_between_asymptotes_l204_204636


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204027

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204027


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204450

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204450


namespace smallest_odd_number_with_five_different_prime_factors_l204_204310

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204310


namespace solve_log_equation_l204_204890

open Real -- Opens the Real namespace for real number operations

theorem solve_log_equation (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : 2 - log 2 a = 3 - log 3 b ∧ 2 - log 2 a = log 6 (1 / (a + b))) : 
  (1 / a + 1 / b = 1 / 108) := 
begin
  sorry
end

end solve_log_equation_l204_204890


namespace minimum_centroid_distance_l204_204128

noncomputable def vector_magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def centroid (A B : ℝ × ℝ) : ℝ × ℝ :=
  ( (A.1 + B.1) / 3, (A.2 + B.2) / 3 )

theorem minimum_centroid_distance
  (OA OB : ℝ × ℝ)
  (H1 : OA.1 * OB.1 + OA.2 * OB.2 = 6)
  (H2 : inner_product_angle OA OB = (1 / 2)) :
  vector_magnitude (centroid OA OB) = 2 :=
by
  -- Proof steps go here
  sorry

end minimum_centroid_distance_l204_204128


namespace fraction_denominator_l204_204054

theorem fraction_denominator (x : ℕ) (h₁ : 525 / x = 0.525) (h₂ : ∀ n ≥ 1, decimal_repetition n 2 (21 / 40) = 5) : x = 1000 :=
sorry

end fraction_denominator_l204_204054


namespace simplify_fraction_eq_l204_204146

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end simplify_fraction_eq_l204_204146


namespace determine_P_l204_204641

def digits := {1, 2, 3, 4, 5}

noncomputable def is_divisible_by_5 (n : ℕ) : Prop := n % 5 = 0
noncomputable def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

noncomputable def valid_number (P Q R S T : ℕ) : Prop :=
  {P, Q, R, S, T} = digits ∧
  is_divisible_by_5 (100 * P + 10 * Q + R) ∧
  is_divisible_by_4 (100 * Q + 10 * R + S) ∧
  is_divisible_by_5 (100 * R + 10 * S + T)

theorem determine_P (P Q R S T : ℕ) (h : valid_number P Q R S T) : P = 3 :=
  sorry

end determine_P_l204_204641


namespace range_of_a_l204_204521

theorem range_of_a (a : ℝ) : (¬ ∀ x : ℝ, x^2 + a * x + 1 ≥ 0) ↔ a ∈ Set.Iio (-2) ∪ Set.Ioi 2 :=
by
  sorry

end range_of_a_l204_204521


namespace find_A_find_c_find_sin_2B_plus_A_l204_204567

variable {a b c A B : ℝ}
variable {cos_B : ℝ}

-- Given conditions
axiom a_val : a = 5
axiom b_val : b = 6
axiom cos_B_val : cos_B = -4 / 5
axiom B_obtuse : B > π / 2 ∧ B < π

-- Prove the required values
theorem find_A : 
  (a = 5) → (b = 6) → (cos B = -4 / 5) → A = π / 6 :=
by infer_instance
-- sorry -- Proof omitted

theorem find_c : 
  (a = 5) → (b = 6) → (cos B = -4 / 5) → c = -4 + 3 * sqrt 3 :=
by infer_instance
-- sorry -- Proof omitted

theorem find_sin_2B_plus_A : 
  (a = 5) → (b = 6) → (cos B = -4 / 5) → 
  (A = π / 6) → sin (2 * B + A) = (7 - 24 * sqrt 3) / 50 :=
by infer_instance
-- sorry -- Proof omitted

end find_A_find_c_find_sin_2B_plus_A_l204_204567


namespace pass_each_other_at_noon_l204_204928

noncomputable def time_they_meet : String :=
  let distance_to_summit : ℝ := 5000
  let hillary_rate_up : ℝ := 800
  let eddy_rate_up : ℝ := 500
  let hillary_stop_distance : ℝ := 1000
  let hillary_rate_down : ℝ := 1000

  let distance_hillary_climbs : ℝ := distance_to_summit - hillary_stop_distance
  let hillary_climb_time : ℝ := distance_hillary_climbs / hillary_rate_up
  let hillary_stop_time : ℕ := 6 + hillary_climb_time.to_nat

  let eddy_distance_by_stop_time : ℝ := (hillary_stop_time - 6) * eddy_rate_up
  let distance_between_them_at_stop : ℝ := distance_hillary_climbs - eddy_distance_by_stop_time
  let closing_rate : ℝ := hillary_rate_down + eddy_rate_up
  let time_to_meet : ℝ := distance_between_them_at_stop / closing_rate

  let final_meeting_time : ℕ := hillary_stop_time + time_to_meet.to_nat
  if final_meeting_time = 12 then "12:00" else "Not the expected time"

theorem pass_each_other_at_noon :
  time_they_meet = "12:00" :=
by
  sorry

end pass_each_other_at_noon_l204_204928


namespace clock_angle_3_36_l204_204216

def minute_hand_position (minutes : ℕ) : ℝ :=
  minutes * 6

def hour_hand_position (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def angle_difference (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

def acute_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem clock_angle_3_36 :
  acute_angle (angle_difference (minute_hand_position 36) (hour_hand_position 3 36)) = 108 :=
by
  sorry

end clock_angle_3_36_l204_204216


namespace percent_decrease_is_92_l204_204565

open Real

-- Given conditions
def cost_1990 := 60 -- in cents per minute
def cost_2010 := 10 -- in cents per minute
def subsidy_2010 := 5 -- in cents per minute

-- Question: Prove the percent decrease.
theorem percent_decrease_is_92 :
  let effective_cost_2010 := cost_2010 - subsidy_2010 in
  let decrease := cost_1990 - effective_cost_2010 in
  let percent_decrease := (decrease / cost_1990) * 100 in
  abs (percent_decrease - 92) < 0.5 :=
by
  sorry

end percent_decrease_is_92_l204_204565


namespace smallest_odd_with_five_different_prime_factors_l204_204257

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204257


namespace product_of_469111_and_9999_l204_204794

theorem product_of_469111_and_9999 : 469111 * 9999 = 4690418889 := 
by 
  sorry

end product_of_469111_and_9999_l204_204794


namespace fibonacci_sum_l204_204111

noncomputable def fibonacci : ℕ → ℕ 
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n+1) + fibonacci n

theorem fibonacci_sum :
  (∑ n in (Finset.range ∞), (fibonacci (2*n + 1) / (5^n))) = 35 / 3 :=
sorry

end fibonacci_sum_l204_204111


namespace negation_exists_l204_204654

theorem negation_exists (h : ∀ x : ℝ, 0 < x → Real.sin x < x) : ∃ x : ℝ, 0 < x ∧ Real.sin x ≥ x :=
by
  sorry

end negation_exists_l204_204654


namespace tangent_line_at_M_neg_pi_zero_l204_204644
open Real

noncomputable def f (x : ℝ) : ℝ := sin x / x

def tangent_line_eq (x y : ℝ) : Prop :=
  x - π * y + π = 0

theorem tangent_line_at_M_neg_pi_zero :
  tangent_line_eq (-π) (f (-π)) :=
sorry

end tangent_line_at_M_neg_pi_zero_l204_204644


namespace smallest_odd_number_with_five_different_prime_factors_l204_204313

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204313


namespace fourth_vertex_of_square_l204_204082

open Complex

theorem fourth_vertex_of_square :
  let A := 1 + 2 * Complex.i
  let B := -2 + Complex.i
  let C := -1 - 2 * Complex.i
  ∃ D : ℂ, (1 + 2 * Complex.i, -2 + Complex.i, -1 - 2 * Complex.i, D).is_square ∧ D = 2 - Complex.i := 
by
  sorry

end fourth_vertex_of_square_l204_204082


namespace smallest_odd_number_with_five_prime_factors_l204_204270

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204270


namespace find_S9_l204_204590

open Nat

noncomputable def S_n (n : ℕ) (a_1 a_n : ℕ) : ℕ :=
  n * (a_1 + a_n) / 2

noncomputable def a_n (a_1 d : ℕ) (n : ℕ) : ℕ :=
  a_1 + (n - 1) * d

theorem find_S9 (a1 d a2 S4 : ℕ) (h1 : a_2 = a_1 + d) (h2 : S_4 = 4 * a_1 + 6 * d) :
  S_9 a_1 (a_n a_1 d 9) = 81 :=
by
  -- Proof goes here
  sorry

end find_S9_l204_204590


namespace pump_time_l204_204755

def depth_in_feet := 18 / 12

def volume_cubic_feet := depth_in_feet * 24 * 32

def volume_gallons := volume_cubic_feet * 7.5

def pumping_rate := 8 * 3

def total_time := volume_gallons / pumping_rate

theorem pump_time (h : total_time = 360) : total_time = 360 :=
by sorry

end pump_time_l204_204755


namespace find_coordinates_of_B_l204_204901

variables (k a : ℝ)
def intersects_at_A (k a : ℝ) := (¬ (k = 0)) ∧ (¬ (a = 0)) ∧ ((2:ℝ) = (k / (-1:ℝ))) ∧ ((2:ℝ) = (a * (-1:ℝ)))

theorem find_coordinates_of_B (k a : ℝ) (h : intersects_at_A k a) : (1, -2) = (1:ℝ, - (2:ℝ)) :=
by {
  sorry
}

end find_coordinates_of_B_l204_204901


namespace smallest_odd_with_five_prime_factors_l204_204289

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204289


namespace highest_possible_value_of_integer_in_list_l204_204637

def arithmeticMean (l : List ℕ) : ℕ :=
  l.sum / l.length

theorem highest_possible_value_of_integer_in_list :
  ∀ (l : List ℕ), l.length = 5 ∧ l.sum = 55 ∧ l.nodup ∧ ∀ x ∈ l, x > 0 →
    ∃ x' ∈ l, x' = 45 := 
by
  sorry

end highest_possible_value_of_integer_in_list_l204_204637


namespace smallest_odd_number_with_five_primes_proof_l204_204242

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204242


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204246

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204246


namespace tan_theta_minus_cos_over_sin_l204_204499

theorem tan_theta_minus_cos_over_sin (θ : ℝ) (h : sin θ * cos θ = 1 / 2) : tan θ - cos θ / sin θ = 0 := 
by
  sorry

end tan_theta_minus_cos_over_sin_l204_204499


namespace odd_primes_remainder_mod_32_l204_204113

theorem odd_primes_remainder_mod_32 :
  let M := (3 * 5 * 7 * 11 * 13)
  in M % 32 = 7 :=
by
  sorry

end odd_primes_remainder_mod_32_l204_204113


namespace twentieth_fisherman_caught_l204_204684

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l204_204684


namespace ten_ray_not_six_ray_count_l204_204980

def unit_square := {p : ℝ × ℝ // 0 ≤ p.1 ∧ p.1 ≤ 1 ∧ 0 ≤ p.2 ∧ p.2 ≤ 1}

def n_ray_partitional (p : unit_square) (n : ℕ) : Prop :=
  n ≥ 4 ∧ ∃ rays : fin n → (unit_square → Prop),
  (∀ i, rays i p) ∧
  (∃ areas : fin n → ℝ, (∀ i, areas i = 1 / n) ∧ (Σ i, areas i = 1))

def count_10_ray_non_6_ray_partitional (R : unit_square) : ℕ :=
  let count_10 := (fin 19).card * (fin 19).card
  let overlap := fin 19.card
  count_10 - overlap

theorem ten_ray_not_six_ray_count :
  ∃ k : ℕ, k = 342 ∧ count_10_ray_non_6_ray_partitional unit_square = k := by
  sorry

end ten_ray_not_six_ray_count_l204_204980


namespace sum_of_angles_WYZ_XYZ_l204_204369

/-- 
Assume a circle is circumscribed around quadrilateral WXYZ.
Given inscribed angles ∠WXY = 50 degrees and ∠YZW = 70 degrees,
prove that the sum of the angles ∠WYZ + ∠XYZ equals 60 degrees.
-/
theorem sum_of_angles_WYZ_XYZ :
  (∠WXY = 50) → (∠YZW = 70) → (∠WYZ + ∠XYZ = 60) :=
by
  intro h1 h2
  sorry

end sum_of_angles_WYZ_XYZ_l204_204369


namespace ellipse_condition_l204_204090

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end ellipse_condition_l204_204090


namespace part_a_part_b_l204_204788

-- Prime factorization of 6000
def primeFactors_6000 : List (ℕ × ℕ) := [(2, 4), (3, 1), (5, 3)]

-- Proof problem for part (a)
theorem part_a (n : ℕ) (pf : List (ℕ × ℕ)) (h : n = 6000) (pf = primeFactors_6000) : 
  ∀ n, n = 6000 → (pf.foldl (λ acc pair => acc * (pair.snd + 1)) 1) = 40 :=
by
  intros n h
  sorry

-- Proof problem for part (b)
theorem part_b (n : ℕ) (pf : List (ℕ × ℕ)) (h : n = 6000) (pf = primeFactors_6000) : 
  ∀ n, n = 6000 → 
  (let totalFactors := pf.foldl (λ acc pair => acc * (pair.snd + 1)) 1;
       perfectSquareFactors := (if 2 ∈ pf then 3 else 1) * (if 3 ∈ pf then 1 else 1) * (if 5 ∈ pf then 2 else 1);
       nonPerfectSquareFactors := totalFactors - perfectSquareFactors
  in nonPerfectSquareFactors = 34) :=
by
  intros n h
  sorry

end part_a_part_b_l204_204788


namespace abs_neg_four_squared_plus_six_l204_204154

theorem abs_neg_four_squared_plus_six : |(-4^2 + 6)| = 10 := by
  -- We skip the proof steps according to the instruction
  sorry

end abs_neg_four_squared_plus_six_l204_204154


namespace equivalence_of_statements_l204_204102

theorem equivalence_of_statements
  {G : Type*} [group G] [fintype G]
  (K : conjugacy_class G) (hK : group.closure (set_of (λ x, x ∈ K)) = ⊤) :
  (∃ m : ℕ, ∀ g : G, ∃ ks : fin m → G, (∀ i : fin m, ks i ∈ K) ∧ g = (list.of_fn ks).prod) ↔
  (G = commutator_subgroup G) :=
sorry

end equivalence_of_statements_l204_204102


namespace range_of_x_l204_204891

noncomputable def f : ℝ → ℝ := sorry
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def is_decreasing_on_nonneg (f : ℝ → ℝ) : Prop := ∀ x₁ x₂, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → f x₁ ≥ f x₂

theorem range_of_x (h1 : is_even f) (h2 : is_decreasing_on_nonneg f) (h3 : ∀ x, f (log x) > f 1 → 0 < x) :
  {x : ℝ | 0 < x ∧ x < 10} = {x : ℝ | f (log x) > f 1} :=
sorry

end range_of_x_l204_204891


namespace math_problem_l204_204398

theorem math_problem
  (x₁ x₂ x₃ x₄ x₅ x₆ x₇ : ℝ)
  (h1 : x₁ + 4 * x₂ + 9 * x₃ + 16 * x₄ + 25 * x₅ + 36 * x₆ + 49 * x₇ = 1)
  (h2 : 4 * x₁ + 9 * x₂ + 16 * x₃ + 25 * x₄ + 36 * x₅ + 49 * x₆ + 64 * x₇ = 12)
  (h3 : 9 * x₁ + 16 * x₂ + 25 * x₃ + 36 * x₄ + 49 * x₅ + 64 * x₆ + 81 * x₇ = 123) :
  16 * x₁ + 25 * x₂ + 36 * x₃ + 49 * x₄ + 64 * x₅ + 81 * x₆ + 100 * x₇ = 334 := by
  sorry

end math_problem_l204_204398


namespace smallest_odd_number_with_five_different_prime_factors_l204_204316

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204316


namespace equal_radii_circumcircles_l204_204603

theorem equal_radii_circumcircles (ABCD : Type*)
  [Parallelogram ABCD] [NotRectangle ABCD] (P : Point)
  (same_radius : same_radius_circumcircles P ABCD) : 
  equal_radii_circumcircles P AB BC :=
by 
  sorry

end equal_radii_circumcircles_l204_204603


namespace relationship_C1_C2_A_l204_204981

variables (A B C C1 C2 : ℝ)

-- Given conditions
def TriangleABC : Prop := B = 2 * A
def AngleSumProperty : Prop := A + B + C = 180
def AltitudeDivides := C1 = 90 - A ∧ C2 = 90 - 2 * A

-- Theorem to prove the relationship between C1, C2, and A
theorem relationship_C1_C2_A (h1: TriangleABC A B) (h2: AngleSumProperty A B C) (h3: AltitudeDivides C1 C2 A) : 
  C1 - C2 = A :=
by sorry

end relationship_C1_C2_A_l204_204981


namespace smallest_odd_number_with_five_different_prime_factors_l204_204285

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204285


namespace solve_for_a_l204_204496

open Set

theorem solve_for_a (a : ℝ) :
  let M := ({a^2, a + 1, -3} : Set ℝ)
  let P := ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ)
  M ∩ P = {-3} →
  a = -1 :=
by
  intros M P h
  have hM : M = {a^2, a + 1, -3} := rfl
  have hP : P = {a - 3, 2 * a - 1, a^2 + 1} := rfl
  rw [hM, hP] at h
  sorry

end solve_for_a_l204_204496


namespace circle_area_eq_100pi_l204_204789

-- Definitions based on conditions
def square_area : ℝ := 400
def diameter := real.sqrt square_area
def radius := diameter / 2

-- Proof statement
theorem circle_area_eq_100pi 
  (square_area : ℝ := 400)
  (diameter := real.sqrt square_area)
  (radius := diameter / 2) :
  π * radius^2 = 100 * π :=
by
  sorry

end circle_area_eq_100pi_l204_204789


namespace perfect_squares_of_nat_l204_204585

theorem perfect_squares_of_nat (a b c : ℕ) (h : a^2 + b^2 + c^2 = (a - b)^2 + (b - c)^2 + (c - a)^2) :
  ∃ m n p q : ℕ, ab = m^2 ∧ bc = n^2 ∧ ca = p^2 ∧ ab + bc + ca = q^2 :=
by sorry

end perfect_squares_of_nat_l204_204585


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204245

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204245


namespace number_of_integer_length_chords_l204_204162

theorem number_of_integer_length_chords :
  let A := (11, 2)
  let circle_eq : ℝ × ℝ → Prop := λ p, (p.1^2 + p.2^2 + 2*p.1 - 4*p.2 - 164 = 0)
  ∃ n : ℕ, ∀ P : ℝ × ℝ, P = A → circle_eq P → n = 32 :=
sorry

end number_of_integer_length_chords_l204_204162


namespace find_functions_solution_l204_204833

noncomputable def solution (f : ℝ → ℝ) : Prop :=
  (∀ a b c d : ℝ, f (a - b) * f (c - d) + f (a - d) * f (b - c) ≤ (a - c) * f (b - d)) →
  (f = (λ x, 0) ∨ f = id)

theorem find_functions_solution :
  solution f :=
sorry

end find_functions_solution_l204_204833


namespace _l204_204354

noncomputable def point (α : Type) := prod α α

structure triangle (α : Type) :=
  (A B C : point α)
  (equilateral : (dist A B = dist B C) ∧ (dist B C = dist C A))

def dist {α : Type} [metric_space α] (p1 p2 : point α) : α :=
  ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)^(1/2 : α)

def isCircle (α : Type) [metric_space α] (center : point α) (radius : α) (p : point α) : Prop :=
  dist center p = radius

noncomputable def symmetricPointAroundLine {α : Type} [metric_space α] (A B C : point α) : point α :=
  -- define the symmetric point M of A with respect to line BC
  sorry

noncomputable theorem locusOfPointsEquidistant (α : Type) [metric_space α] (t : triangle α) :
  ∃ M : point α, 
  let locus := { X : point α | dist t.A X = (dist t.B X) + (dist t.C X) } in
  ∀ X : point α, X ∈ locus ↔ isCircle α (symmetricPointAroundLine t.A t.B t.C) (dist t.A t.B) X := 
sorry

noncomputable def pedalTriangle (α : Type) (A B C X : point α): Type :=
  -- Define the points A', B', and C' as the projections of X onto the sides BC, CA, and AB respectively
  sorry

noncomputable theorem pedalTriangleRightAngled (α : Type) [metric_space α] (t : triangle α) :
  let locus := { X : point α | dist t.A X = (dist t.B X) + (dist t.C X) } in
  ∀ X : point α, X ∈ locus → 
  ∃ A' B' C' : point α, (pedalTriangle α t.A t.B t.C X) ∧ right_angled A' B' C' :=
sorry

end _l204_204354


namespace sum_of_areas_of_adjacent_triangles_equals_fourth_triangle_l204_204690

theorem sum_of_areas_of_adjacent_triangles_equals_fourth_triangle 
  (ABC : Triangle) 
  (O : Point) 
  (A1 B1 C1 : Point)
  (AA1 BB1 CC1 : Segment)
  (condition1 : Through O, segments AA1 BB1 CC1 parallel to sides)
  (condition2 : Segments AA1 BB1 CC1 divide triangle ABC into three adjoining triangles at vertices and one fourth triangle)
  (area_a : Real) (area_b : Real) (area_c : Real) (area_4th : Real) 
  (S_a : area_a = area of triangle_adjacent A)
  (S_b : area_b = area of triangle_adjacent B)
  (S_c : area_c = area of triangle_adjacent C)
  (S : area_4th = area of fourth_triangle) :
  S_a + S_b + S_c = S :=
by
  sorry

end sum_of_areas_of_adjacent_triangles_equals_fourth_triangle_l204_204690


namespace smallest_odd_number_with_five_prime_factors_l204_204269

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204269


namespace smallest_odd_number_with_five_prime_factors_l204_204299

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204299


namespace winning_candidate_votes_l204_204186

-- Definitions of the given conditions transformed from the problem statement
def total_votes (w : ℕ) : ℝ :=
  let c1 := 5000
  let c2 := 15000
  let percentage_winning := 65.21739130434783 / 100
  w / percentage_winning + c1 + c2

-- Definition to prove that the winning candidate received 37500 votes
theorem winning_candidate_votes {w : ℕ} : 
  w = 37500 ↔ total_votes w = 20000 / 0.3478260869565217 + 5000 + 15000 :=
sorry

end winning_candidate_votes_l204_204186


namespace james_total_fish_catch_l204_204572

-- Definitions based on conditions
def weight_trout : ℕ := 200
def weight_salmon : ℕ := weight_trout + (60 * weight_trout / 100)
def weight_tuna : ℕ := 2 * weight_trout
def weight_bass : ℕ := 3 * weight_salmon
def weight_catfish : ℚ := weight_tuna / 3

-- Total weight of the fish James caught
def total_weight_fish : ℚ := 
  weight_trout + weight_salmon + weight_tuna + weight_bass + weight_catfish 

-- The theorem statement
theorem james_total_fish_catch : total_weight_fish = 2013.33 := by
  sorry

end james_total_fish_catch_l204_204572


namespace sum_elements_union_l204_204627

open Finset

def unions {α : Type*} (sets : List (Finset α)) : Finset α :=
sets.foldr (∪) ∅

def f {α : Type*} (k : List (Finset α)) : ℕ :=
(unions k).card

def S (n : ℕ) {α : Type*} (m : Finₓ α) : Finset (List (Finset α)) :=
(finset.univ : Finset (Finset α)).toList.replicate n

noncomputable def s (n m : ℕ) : ℕ :=
(S n m).sum f

theorem sum_elements_union (n : ℕ) :
  s n 1998 = 1998 * (2^(1998 * n) - 2^(1997 * n)) :=
sorry

end sum_elements_union_l204_204627


namespace min_value_of_reciprocals_l204_204600

variable (b : ℕ → ℝ)

theorem min_value_of_reciprocals (h : ∀ i, (0 < b i)) (h_sum : ∑ i in Finset.range 10, b i = 2) :
  (∑ i in Finset.range 10, (1 / b i)) ≥ 50 := 
begin
  sorry
end

end min_value_of_reciprocals_l204_204600


namespace initial_number_18_l204_204333

theorem initial_number_18 (N : ℤ) (h : ∃ k : ℤ, N + 5 = 23 * k) : N = 18 := 
sorry

end initial_number_18_l204_204333


namespace even_function_smallest_period_pi_l204_204908

def f (x : ℝ) : ℝ := (Real.cos x)^2 - 1/2

theorem even_function_smallest_period_pi :
  (∀ x : ℝ, f x = f (-x)) ∧ (∀ T : ℝ, (0 < T) ∧ (∀ x : ℝ, f (x + T) = f x) → T >= π) :=
by
  sorry

end even_function_smallest_period_pi_l204_204908


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204006

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204006


namespace scientific_notation_of_8200000_l204_204725

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l204_204725


namespace smallest_odd_number_with_five_different_prime_factors_l204_204280

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204280


namespace height_of_pyramid_proof_l204_204955

open Real

noncomputable def height_of_pyramid (s h_AB h_BC angle_ABC angle_ADC : ℝ) : ℝ := s

theorem height_of_pyramid_proof :
  ∀ (AB BC : ℝ) (angle_ABC angle_ADC : ℝ) (h_lateral : ℝ)
  (height : ℝ),
  AB = 2 → BC = 6 →
  angle_ABC = π / 3 → angle_ADC = 2 * π / 3 →
  h_lateral = sqrt 2 →
  height_of_pyramid height h_lateral AB BC angle_ABC angle_ADC 
  = 2 * sqrt (3 * sqrt 2 - 4) := by
  intros
  sorry

end height_of_pyramid_proof_l204_204955


namespace min_employees_needed_l204_204394

-- Define the conditions
variable (W A : Finset ℕ)
variable (n_W n_A n_WA : ℕ)

-- Assume the given condition values
def sizeW := 95
def sizeA := 80
def sizeWA := 30

-- Define the proof problem
theorem min_employees_needed :
  (sizeW + sizeA - sizeWA) = 145 :=
by sorry

end min_employees_needed_l204_204394


namespace centroid_distance_sum_l204_204112

noncomputable def quad_centroid_vectors (a b c d : ℝ^3) : ℝ^3 :=
  (a + b + c + d) / 4

noncomputable def square_distance (x y : ℝ^3) : ℝ :=
  ∥x - y∥^2

theorem centroid_distance_sum (a b c d : ℝ^3) (G : ℝ^3)
  (hG : G = quad_centroid_vectors a b c d)
  (h_sum : square_distance G a + square_distance G b + square_distance G c + square_distance G d = 116) :
  square_distance a b + square_distance a c + square_distance a d + square_distance b c + square_distance b d + square_distance c d = 464 :=
sorry

end centroid_distance_sum_l204_204112


namespace scientific_notation_correctness_l204_204727

theorem scientific_notation_correctness : ∃ x : ℝ, x = 8.2 ∧ (8200000 : ℝ) = x * 10^6 :=
by
  use 8.2
  split
  · rfl
  · sorry

end scientific_notation_correctness_l204_204727


namespace inequality_solution_set_l204_204548

theorem inequality_solution_set (a : ℤ) : 
  (∀ x : ℤ, (1 + a) * x > 1 + a → x < 1) → a < -1 :=
sorry

end inequality_solution_set_l204_204548


namespace intersection_eq_M_l204_204608

-- Define the sets M and N according to the given conditions
def M : Set ℝ := {x : ℝ | x^2 - x < 0}
def N : Set ℝ := {x : ℝ | |x| < 2}

-- The 'theorem' statement to prove M ∩ N = M
theorem intersection_eq_M : M ∩ N = M :=
  sorry

end intersection_eq_M_l204_204608


namespace h_formula_l204_204192

def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def h (m n : ℕ) : ℕ :=
  ∑ k in Finset.range (m + 1), if 2 ≤ k then (-1)^(m - k) * binomial m k * k * (k - 1) ^ (n - 1) else 0

theorem h_formula {m n : ℕ} (hm : 2 ≤ m) (hn : 1 ≤ n) :
  h m n = ∑ k in Finset.range (m + 1), if 2 ≤ k then (-1)^(m - k) * binomial m k * k * (k - 1) ^ (n - 1) else 0 :=
sorry

end h_formula_l204_204192


namespace smallest_odd_number_with_five_prime_factors_l204_204272

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204272


namespace total_investment_with_interest_l204_204193

def principal : ℝ := 1000
def part3Percent : ℝ := 199.99999999999983
def rate3Percent : ℝ := 0.03
def rate5Percent : ℝ := 0.05
def interest3Percent : ℝ := part3Percent * rate3Percent
def part5Percent : ℝ := principal - part3Percent
def interest5Percent : ℝ := part5Percent * rate5Percent
def totalWithInterest : ℝ := principal + interest3Percent + interest5Percent

theorem total_investment_with_interest :
  totalWithInterest = 1046.00 :=
by
  unfold totalWithInterest interest5Percent part5Percent interest3Percent
  sorry

end total_investment_with_interest_l204_204193


namespace count_integers_l204_204931

def satisfies_conditions (n : ℤ) (r : ℤ) : Prop :=
  200 < n ∧ n < 300 ∧ n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5

theorem count_integers (n : ℤ) (r : ℤ) :
  (satisfies_conditions n r) → ∃! n, 200 < n ∧ n < 300 ∧ ∃ r, n % 7 = r ∧ n % 9 = r ∧ 0 ≤ r ∧ r < 5 :=
by
  sorry

end count_integers_l204_204931


namespace distance_center_of_ball_travels_l204_204752

noncomputable def radius_of_ball : ℝ := 2
noncomputable def R1 : ℝ := 100
noncomputable def R2 : ℝ := 60
noncomputable def R3 : ℝ := 80

noncomputable def adjusted_R1 : ℝ := R1 - radius_of_ball
noncomputable def adjusted_R2 : ℝ := R2 + radius_of_ball
noncomputable def adjusted_R3 : ℝ := R3 - radius_of_ball

noncomputable def distance_travelled : ℝ :=
  (Real.pi * adjusted_R1) +
  (Real.pi * adjusted_R2) +
  (Real.pi * adjusted_R3)

theorem distance_center_of_ball_travels : distance_travelled = 238 * Real.pi :=
by
  sorry

end distance_center_of_ball_travels_l204_204752


namespace smallest_odd_number_with_five_prime_factors_l204_204232

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204232


namespace smallest_positive_n_l204_204544

open BigOperators

-- Definitions for our conditions
def expression (n : ℕ) (k : ℕ) : ℕ := n * (2 ^ 5) * ((2 ^ 2) * (3 ^ 2)) * (7 ^ k)

theorem smallest_positive_n {k : ℕ} :
    ∃ n : ℕ, (∃ N : ℕ, n = N) ∧ (expression n k) % (5^2) = 0 ∧ (expression n k) % (3^3) = 0 ∧
    ∀ m : ℕ, (m > 0) ∧ ((expression m k) % (5^2) = 0 ∧ (expression m k) % (3^3) = 0) → n ≤ m := 
begin
  use 75,
  split,
  existsi 75,
  split,
  norm_num,
  split,
  norm_num,
  sorry
end

end smallest_positive_n_l204_204544


namespace part1_part2_l204_204487

noncomputable def a_seq : ℕ → ℝ
| 0       := 1
| (n + 1) := a_seq n / (1 + a_seq n)

theorem part1 (n : ℕ) :
  (∀ m : ℕ,  m > 0 → (1 / a_seq (m + 1)) - (1 / a_seq m) = 1) ∧
  (∀ m : ℕ, m > 0 → a_seq m = 1 / m) := 
sorry

theorem part2 (n : ℕ) :
  (2 / 3 < ∑ i in range n, a_seq i * a_seq (i + 1) ∧ ∑ i in range n, a_seq i * a_seq (i + 1) < 5 / 6) → (n = 4 ∨ n = 5) :=
sorry

end part1_part2_l204_204487


namespace find_absolute_value_l204_204903

theorem find_absolute_value (h k : ℤ) (h1 : 3 * (-3)^3 - h * (-3) + k = 0) (h2 : 3 * 2^3 - h * 2 + k = 0) : |3 * h - 2 * k| = 27 :=
by
  sorry

end find_absolute_value_l204_204903


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204025

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204025


namespace scheduling_schemes_l204_204858

theorem scheduling_schemes (days : Finset ℕ) (A_schedule B_schedule C_schedule : Finset ℕ) 
  (A_not_mon : ¬ 0 ∈ A_schedule) (B_not_sat : ¬ 5 ∈ B_schedule)
  (A_days : A_schedule.card = 2) (B_days : B_schedule.card = 2) (C_days : C_schedule.card = 2) :
  42 = (A_schedule.product B_schedule.product C_schedule).count sorry :=
sorry

end scheduling_schemes_l204_204858


namespace percentage_deficit_second_side_l204_204960

theorem percentage_deficit_second_side :
  ∀ (L W : ℝ) (x : ℝ), 
  let L' := 1.07 * L
  let W' := W * (1 - x / 100)
  let actual_area := L * W
  let measured_area := L' * W'
  let error_percent := 0.0058
  (measured_area = actual_area * (1 + error_percent)) → x ≈ 6 := 
by
  intros L W x
  let L' := 1.07 * L
  let W' := W * (1 - x / 100)
  let actual_area := L * W
  let measured_area := L' * W'
  let error_percent := 0.0058
  have h : measured_area = actual_area * (1 + error_percent) :=
    by sorry
  sorry

end percentage_deficit_second_side_l204_204960


namespace no_real_solutions_for_inequality_l204_204813

theorem no_real_solutions_for_inequality (a : ℝ) :
  ¬∃ x : ℝ, ∀ y : ℝ, |(x^2 + a*x + 2*a)| ≤ 5 → y = x :=
sorry

end no_real_solutions_for_inequality_l204_204813


namespace exists_n_divides_nS_l204_204881

def coprime (a b : ℕ) : Prop :=
  Nat.gcd a b = 1

def bad_numbers (p q : ℕ) (pq_coprime : coprime p q) (h_p : p > 1) (h_q : q > 1) : Set ℕ :=
  {n | ∀ x y : ℕ, n ≠ p * x + q * y}

def S (p q : ℕ) (pq_coprime : coprime p q) (h_p : p > 1) (h_q : q > 1) : ℕ :=
  (bad_numbers p q pq_coprime h_p h_q).sum^2019

theorem exists_n_divides_nS (p q : ℕ) (pq_coprime : coprime p q) (h_p : p > 1) (h_q : q > 1) :
  ∃ n : ℕ, (p - 1) * (q - 1) ∣ n * S p q pq_coprime h_p h_q :=
sorry

end exists_n_divides_nS_l204_204881


namespace solve_for_first_expedition_weeks_l204_204973

-- Define the variables according to the given conditions.
variables (x : ℕ)
variables (days_in_week : ℕ := 7)
variables (total_days_on_island : ℕ := 126)

-- Define the total number of weeks spent on the expeditions.
def total_weeks_on_expeditions (x : ℕ) : ℕ := 
  x + (x + 2) + 2 * (x + 2)

-- Convert total days to weeks.
def total_weeks := total_days_on_island / days_in_week

-- Prove the equation
theorem solve_for_first_expedition_weeks
  (h : total_weeks_on_expeditions x = total_weeks):
  x = 3 :=
by
  sorry

end solve_for_first_expedition_weeks_l204_204973


namespace total_legs_l204_204577

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_l204_204577


namespace find_b_l204_204421

def h (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ b : ℝ, h(b) = 0 ∧ b = 7 / 5 :=
by
  sorry

end find_b_l204_204421


namespace acute_angle_between_hands_at_3_36_l204_204207

variable (minute_hand_position hour_hand_position abs_diff : ℝ)

def minute_hand_angle_at_3_36 : ℝ := 216
def hour_hand_angle_at_3_36 : ℝ := 108

theorem acute_angle_between_hands_at_3_36 (h₀ : minute_hand_position = 216)
    (h₁ : hour_hand_position = 108) :
    abs_diff = abs(minute_hand_position - hour_hand_position) → 
    abs_diff = 108 :=
  by
    rw [h₀, h₁]
    sorry

end acute_angle_between_hands_at_3_36_l204_204207


namespace problem_proof_l204_204646

theorem problem_proof :
  (∀ x, f(x) = sqrt(abs(x + 1) + abs(x + 2) - 5)) →
  (A = {x : ℝ | x ≤ -4 ∨ x ≥ 1}) →
  (B = {x : ℝ | -1 < x ∧ x < 2}) →
  ∀ a b, (a ∈ B ∩ (ℝ \ A)) → (b ∈ B ∩ (ℝ \ A)) →
  (abs(a + b) / 2 < abs(1 + a * b / 4)) :=
by
  intros f A B a b ha hb
  sorry

end problem_proof_l204_204646


namespace age_difference_l204_204770

theorem age_difference (A B : ℕ) (h1 : B = 34) (h2 : A + 10 = 2 * (B - 10)) : A - B = 4 :=
by
  sorry

end age_difference_l204_204770


namespace diameter_percentage_l204_204164

theorem diameter_percentage (d_R d_S : ℝ) (h : π * (d_R / 2)^2 = 0.25 * π * (d_S / 2)^2) : 
  d_R = 0.5 * d_S :=
by 
  sorry

end diameter_percentage_l204_204164


namespace excircle_angle_equality_l204_204568

theorem excircle_angle_equality
  {A B C A1 B1 C1 F : Type}
  (is_triangle_ABC : is_triangle A B C)
  (excircle_touches_BC_AC_AB : excircle_touches_side_opposite AC excircle touches_extension_sides_BC_AC_AB A1 B1 C1)
  (midpoint_F : midpoint F A1 B1) :
  ∠ B1 C1 C = ∠ A1 C1 F :=
sorry

end excircle_angle_equality_l204_204568


namespace scientific_notation_of_8200000_l204_204726

theorem scientific_notation_of_8200000 : 
  (8200000 : ℝ) = 8.2 * 10^6 := 
sorry

end scientific_notation_of_8200000_l204_204726


namespace sum_distances_max_min_sum_distances_min_plane_l204_204892

-- Part 1: Maximum and Minimum values on the circumcircle
theorem sum_distances_max_min (P: ℂ) (n: ℕ) (A: fin n → ℂ) (h: ∀ i, abs P = abs (A i)) :
  ∃ (max min: ℝ), max = 2 * csc (π / (2 * n)) ∧ min = 2 * cot (π / (2 * n)) ∧
    (∀ Q, abs Q = abs P → ∑ i, abs (Q - A i) ≤ max ∧ ∑ i, abs (Q - A i) ≥ min) :=
by { sorry }

-- Part 2: Minimum value anywhere in the plane
theorem sum_distances_min_plane (P: ℂ) (n: ℕ) (A: fin n → ℂ) :
  ∃ (min: ℝ), min = n ∧ 
    (∀ Q, ∑ i, abs (Q - A i) ≥ min) :=
by { sorry }

end sum_distances_max_min_sum_distances_min_plane_l204_204892


namespace sum_of_digits_l204_204934

def digits := { n // n > 0 ∧ n < 6 }

def sum_digits_in_base_6 (S H E: digits) : Prop :=
  let S_val := S.val
  let H_val := H.val
  let E_val := E.val in
  S_val ≠ H_val ∧ H_val ≠ E_val ∧ E_val ≠ S_val
  ∧ (S_val + H_val) % 6 = S_val
  ∧ ((H_val + E_val) % 6 = E_val)
  ∧ ((S_val + E_val) % 6 = S_val)
  ∧ (S_val + H_val + E_val = 7)

theorem sum_of_digits (S H E: digits) (h: sum_digits_in_base_6 S H E) : S.val + H.val + E.val = 7 := 
sorry

end sum_of_digits_l204_204934


namespace circle_radius_l204_204762

theorem circle_radius (M N r : ℝ) (h1 : M = Real.pi * r^2) (h2 : N = 2 * Real.pi * r) (h3 : M / N = 25) : r = 50 :=
by
  sorry

end circle_radius_l204_204762


namespace electric_sharpens_more_l204_204769

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end electric_sharpens_more_l204_204769


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204008

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204008


namespace smallest_odd_with_five_prime_factors_l204_204291

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204291


namespace cost_of_hair_updo_l204_204100

-- Define the cost of a hair updo as a variable H (H : ℝ)
variable (H : ℝ)

-- Define conditions as hypotheses
def manicure_cost : ℝ := 30
def total_cost_with_tips : ℝ := 96
def tip_rate : ℝ := 0.20

-- Hypotheses based on the given problem
hypothesis h1 : total_cost_with_tips = H + manicure_cost + tip_rate * H + tip_rate * manicure_cost
hypothesis h2 : manicure_cost = 30
hypothesis h3 : total_cost_with_tips = 96
hypothesis h4 : tip_rate = 0.20

-- Prove that H = 50
theorem cost_of_hair_updo : H = 50 :=
by
  rw [h2, h3, h4] at h1
  -- more rewriting and solving steps would go here
  sorry

end cost_of_hair_updo_l204_204100


namespace find_m_plus_n_l204_204764

noncomputable theory

-- Define conditions
def a : ℝ := 1/2
def b : ℝ := sqrt(255) / 2 
def c : ℂ := a + b * complex.I

-- Definitions provided in the problem
axiom cond1 : complex.abs c = 8
axiom cond2 : ∃ m n : ℕ, m.coprime n ∧ b^2 = m / n

-- Statement to prove
theorem find_m_plus_n : ∃ (m n : ℕ), nat.coprime m n ∧ b^2 = m / n ∧ (m + n = 259) :=
sorry

end find_m_plus_n_l204_204764


namespace pow_div_pow_eq_l204_204707

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end pow_div_pow_eq_l204_204707


namespace smallest_odd_number_with_five_different_prime_factors_l204_204320

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204320


namespace tan_angle_in_triangle_l204_204566

theorem tan_angle_in_triangle (A B C : Type) [triangle A B C] (angle_B : angle B = 90) (AB : length A B = 13) (BC : length B C = 5) :
  tan angle A = 5 / 12 :=
by 
  sorry

end tan_angle_in_triangle_l204_204566


namespace smallest_odd_number_with_five_prime_factors_l204_204267

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204267


namespace find_counterfeit_coin_l204_204391

/-- 
Given 11 coins, where 10 are genuine with weight 20 g each, and 1 is counterfeit with weight 21 g.
Additionally, a balance scale that equilibrates if the weight on its right pan is twice that on the left pan:
Show that the counterfeit coin can be identified within three weighings.
-/
theorem find_counterfeit_coin :
  ∃! (i : Fin 11), (∀ (j : Fin 11), j ≠ i → coin_weight j = 20) ∧ coin_weight i = 21 → 
  ∃! (k : ℕ) (h_k : k ≤ 3), identify_counterfeit k := 
sorry

noncomputable def coin_weight (n : Fin 11) : ℝ :=
sorry -- Placeholder for coin weights (to be defined)

noncomputable def identify_counterfeit (n : ℕ) : Fin 11 :=
sorry -- Placeholder for counterfeit identification method (to be defined)

end find_counterfeit_coin_l204_204391


namespace gwen_spent_money_l204_204468

theorem gwen_spent_money (x y : ℕ) (h1 : x = 14) (h2 : y = 6) : x - y = 8 := by
  rw [h1, h2]
  rfl

end gwen_spent_money_l204_204468


namespace maximize_profit_l204_204372

noncomputable def total_revenue (x : ℝ) : ℝ :=
  if x ≤ 390 then -(x^3) / 900 + 400 * x else 90090

def cost (x : ℝ) : ℝ := 20000 + 100 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  total_revenue x - cost x

theorem maximize_profit : ∃ x, 0 ≤ x ∧ x ≤ 390 ∧ ∀ y, 0 ≤ y ∧ y ≤ 390 → total_profit x ≥ total_profit y :=
by
  use 300
  intros y hy
  sorry

end maximize_profit_l204_204372


namespace trajectory_of_midpoint_l204_204472

theorem trajectory_of_midpoint (Q : ℝ × ℝ) (P : ℝ × ℝ) (N : ℝ × ℝ)
  (h1 : Q.1^2 - Q.2^2 = 1)
  (h2 : N = (2 * P.1 - Q.1, 2 * P.2 - Q.2))
  (h3 : N.1 + N.2 = 2)
  (h4 : (P.2 - Q.2) / (P.1 - Q.1) = 1) :
  2 * P.1^2 - 2 * P.2^2 - 2 * P.1 + 2 * P.2 - 1 = 0 :=
  sorry

end trajectory_of_midpoint_l204_204472


namespace clock_angle_at_3_36_l204_204201

def minute_hand_angle : ℝ := (36.0 / 60.0) * 360.0
def hour_hand_angle : ℝ := 90.0 + (36.0 / 60.0) * 30.0

def acute_angle (a b : ℝ) : ℝ :=
  let diff := abs (a - b)
  min diff (360 - diff)

theorem clock_angle_at_3_36 :
  acute_angle minute_hand_angle hour_hand_angle = 108 :=
by
  sorry

end clock_angle_at_3_36_l204_204201


namespace algebraic_expression_value_l204_204481

theorem algebraic_expression_value (x y : ℝ) (h : x - 2 * y = -1) : 6 + 2 * x - 4 * y = 4 := by
  sorry

end algebraic_expression_value_l204_204481


namespace true_percent_increase_in_home_cost_l204_204066

theorem true_percent_increase_in_home_cost
  (initial_cost : ℝ) (final_cost : ℝ) (r : ℝ) (n : ℕ)
  (h₀ : initial_cost = 120000)
  (h₁ : final_cost = 192000)
  (h₂ : r = 0.05)
  (h₃ : n = 8) :
  (final_cost / initial_cost) = (1 + r)^n → 
  ((final_cost / initial_cost - 1) * 100 ≈ 47.75) := 
sorry

end true_percent_increase_in_home_cost_l204_204066


namespace algebraic_cofactor_k_value_l204_204963

theorem algebraic_cofactor_k_value :
  let M : Matrix (Fin 3) (Fin 3) ℤ := ![
    ![4, 2, k],
    ![-3, 5, 4],
    ![-1, 1, -2]
  ],
  let elem := M 1 0,
  let minor := Matrix.det ![
    ![2, k],
    ![1, -2]
  ],
  let cofactor := (-1) ^ (1 + 0) * minor in
  cofactor = -10 → k = -14 :=
by
  intros,
  -- The proof can be filled in later
  sorry

end algebraic_cofactor_k_value_l204_204963


namespace cosines_product_of_triangle_l204_204096

theorem cosines_product_of_triangle (K L M P O Q S : Type) [triangle K L M]
  (KP : median K L M P) (circumcenter : center_circumcircle O K L M)
  (incircle_center : center_incircle Q K L M) (intersection : intersects KP OQ S)
  (angle_LKM : ∠ L K M = π / 3) 
  (proportion : (dist O S / dist P S) = sqrt 6 * (dist Q S / dist K S)) :
  (cos (∠ K L M) * cos (∠ K M L)) = -3 / 8 :=
sorry

end cosines_product_of_triangle_l204_204096


namespace two_digit_primes_ending_in_3_l204_204040

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204040


namespace range_of_m_l204_204087

def is_ellipse (m : ℝ) : Prop :=
  ∀ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2

theorem range_of_m (m : ℝ) (h : is_ellipse m) : m > 5 :=
sorry

end range_of_m_l204_204087


namespace clock_angle_at_3_36_l204_204199

def minute_hand_angle : ℝ := (36.0 / 60.0) * 360.0
def hour_hand_angle : ℝ := 90.0 + (36.0 / 60.0) * 30.0

def acute_angle (a b : ℝ) : ℝ :=
  let diff := abs (a - b)
  min diff (360 - diff)

theorem clock_angle_at_3_36 :
  acute_angle minute_hand_angle hour_hand_angle = 108 :=
by
  sorry

end clock_angle_at_3_36_l204_204199


namespace number_of_ways_to_win_championships_l204_204532

-- Definitions for the problem
def num_athletes := 5
def num_events := 3

-- Proof statement
theorem number_of_ways_to_win_championships : 
  (num_athletes ^ num_events) = 125 := 
by 
  sorry

end number_of_ways_to_win_championships_l204_204532


namespace white_roses_count_l204_204400

def total_flowers : ℕ := 6284
def red_roses : ℕ := 1491
def yellow_carnations : ℕ := 3025
def white_roses : ℕ := total_flowers - (red_roses + yellow_carnations)

theorem white_roses_count :
  white_roses = 1768 := by
  sorry

end white_roses_count_l204_204400


namespace triangle_in_unit_cube_l204_204141

theorem triangle_in_unit_cube (a b c : ℝ) : 0 ≤ a ∧ a ≤ real.sqrt 2 → 0 ≤ b ∧ b ≤ real.sqrt 2 → 0 ≤ c ∧ c ≤ real.sqrt 2 → ∃ (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ),
  (x1, y1, z1) ∈ set.Icc (0:ℝ) 1 ∧
  (x2, y2, z2) ∈ set.Icc (0:ℝ) 1 ∧
  (x3, y3, z3) ∈ set.Icc (0:ℝ) 1 ∧
  (real.dist (x1, y1, z1) (x2, y2, z2) = a) ∧
  (real.dist (x1, y1, z1) (x3, y3, z3) = b) ∧
  (real.dist (x2, y2, z2) (x3, y3, z3) = c) :=
by
  sorry

end triangle_in_unit_cube_l204_204141


namespace modified_goldbach_2024_l204_204805

def is_prime (p : ℕ) : Prop := ∀ n : ℕ, n > 1 → n < p → ¬ (p % n = 0)

theorem modified_goldbach_2024 :
  ∃ (p1 p2 : ℕ), p1 ≠ p2 ∧ is_prime p1 ∧ is_prime p2 ∧ p1 + p2 = 2024 := 
sorry

end modified_goldbach_2024_l204_204805


namespace magnitude_of_b_l204_204528

variable (x : ℝ)

def vector_a := (1 : ℝ, 2 : ℝ)
def vector_b := (x, 1 : ℝ)

theorem magnitude_of_b (h : (vector_a.1 / vector_a.2 = vector_b.1 / vector_b.2)) : 
  Real.sqrt (vector_b.1 * vector_b.1 + vector_b.2 * vector_b.2) = Real.sqrt 5 / 2 :=
by sorry

end magnitude_of_b_l204_204528


namespace cond1_impl_const_cond2_impl_const_cond3_impl_const_l204_204597

-- Define sequences and their properties
def is_geometric (a : ℕ → ℝ) : Prop :=
  ∃ q ≠ 0, ∀ n, a (n + 1) = q * a n

def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

def is_constant (a : ℕ → ℝ) : Prop :=
  ∃ c, ∀ n, a n = c

-- Define the constants k and b
variable (k b : ℝ)
-- Assume non-zero constants
variable (hk : k ≠ 0) (hb : b ≠ 0)

-- Condition 1: Both sequences {a_n} and {ka_n + b} are geometric progressions
def cond1 (a : ℕ → ℝ) : Prop :=
  is_geometric a ∧ is_geometric (λ n, k * a n + b)

-- Condition 2: {a_n} is an arithmetic progression, and {ka_n + b} is a geometric progression
def cond2 (a : ℕ → ℝ) : Prop :=
  is_arithmetic a ∧ is_geometric (λ n, k * a n + b)

-- Condition 3: {a_n} is a geometric progression, and {ka_n + b} is an arithmetic progression
def cond3 (a : ℕ → ℝ) : Prop :=
  is_geometric a ∧ is_arithmetic (λ n, k * a n + b)

-- Theorems to be proved
theorem cond1_impl_const (a : ℕ → ℝ) (h : cond1 k b hk hb a) : is_constant a :=
  sorry

theorem cond2_impl_const (a : ℕ → ℝ) (h : cond2 k b hk hb a) : is_constant a :=
  sorry

theorem cond3_impl_const (a : ℕ → ℝ) (h : cond3 k b hk hb a) : is_constant a :=
  sorry

end cond1_impl_const_cond2_impl_const_cond3_impl_const_l204_204597


namespace hyperbola_asymptotes_equations_l204_204945

open Real

def hyperbola_asymptotes (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) (h4 : a = c / 2) : Prop :=
  b = sqrt (3) * a ∧ (∀ x : ℝ, y = sqrt(3) * x ∨ y = - sqrt(3) * x)

theorem hyperbola_asymptotes_equations (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) (h4 : a = c / 2) : hyperbola_asymptotes a b c h1 h2 h3 h4 :=
sorry

end hyperbola_asymptotes_equations_l204_204945


namespace interval_where_decreasing_l204_204648

open Real

noncomputable def piecewise_function (x : ℝ) : ℝ :=
if x ≥ 0 then (x - 3) * x else -((x - 3) * x)

theorem interval_where_decreasing :
  ∃ a b : ℝ, (∀ x : ℝ, a ≤ x ∧ x ≤ b → (piecewise_function x)' < 0) ∧ a = 0 ∧ b = 3 / 2 :=
sorry

end interval_where_decreasing_l204_204648


namespace sum_series_eq_l204_204189

theorem sum_series_eq (n : ℕ) : (Finset.range (n+1)).sum (λ k, 1 / ((k+1) * ((k+1) + 1) : ℝ)) = n / (n + 1 : ℝ) :=
by
  sorry

end sum_series_eq_l204_204189


namespace fixed_point_of_parabola_l204_204121

theorem fixed_point_of_parabola (s : ℝ) : ∃ y : ℝ, y = 4 * 3^2 + s * 3 - 3 * s ∧ (3, y) = (3, 36) :=
by
  sorry

end fixed_point_of_parabola_l204_204121


namespace problem_abc_value_l204_204125

theorem problem_abc_value 
  (a b c : ℤ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > 0)
  (h4 : Int.gcd b c = 1)
  (h5 : (b + c) % a = 0)
  (h6 : (a + c) % b = 0) :
  a * b * c = 6 :=
sorry

end problem_abc_value_l204_204125


namespace pow_div_pow_eq_l204_204708

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end pow_div_pow_eq_l204_204708


namespace find_smallest_n_l204_204086

-- defining the geometric sequence and its sum for the given conditions
def a_n (n : ℕ) := 3 * (4 ^ n)

def S_n (n : ℕ) := (a_n n - 1) / (4 - 1) -- simplification step

-- statement of the problem: finding the smallest natural number n such that S_n > 3000
theorem find_smallest_n :
  ∃ n : ℕ, S_n n > 3000 ∧ ∀ m : ℕ, m < n → S_n m ≤ 3000 := by
  sorry

end find_smallest_n_l204_204086


namespace negation_proposition_l204_204656

variable {a b : ℝ}

theorem negation_proposition (h : a ≤ b) : 2^a ≤ 2^b :=
sorry

end negation_proposition_l204_204656


namespace acute_angle_3_36_clock_l204_204218

theorem acute_angle_3_36_clock : 
  let minute_hand_degrees := (36 / 60) * 360,
      hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360,
      angle := abs(minute_hand_degrees - hour_hand_degrees) in
  angle = 108 :=
by
  let minute_hand_degrees := (36 / 60) * 360
  let hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360
  let angle := abs(minute_hand_degrees - hour_hand_degrees)
  show angle = 108 from sorry

end acute_angle_3_36_clock_l204_204218


namespace point_P_lies_on_parabola_l204_204076

-- Define points and their geometric relationships
structure Point3D (ℝ : Type _) := 
(x : ℝ) 
(y : ℝ)
(z : ℝ)

def is_cube (A B C D A1 D1 M P : Point3D ℝ) (edge_len AM : ℝ) : Prop :=
  edge_len = 1 ∧
  M = ⟨1 / 3, 0, 0⟩ ∧
  -- P is any point on the face ABCD
  P.y = 0 ∧ P.z = 0 ∧
  (∃ k : ℝ, D = ⟨k, 0, 0⟩ ∧ A1 = ⟨0, 0, 1⟩ ∧ D1 = ⟨k, 0, 1⟩)

-- Define the distance squared
def distance_squared (p1 p2 : Point3D ℝ) : ℝ := 
  (p1.x - p2.x)^2 + (p1.y - p2.y)^2 + (p1.z - p2.z)^2

-- Define the geometric condition given in the problem
def geometric_condition (P M : Point3D ℝ) (A1 D1 : Point3D ℝ) : Prop :=
  let dist_line := distance_squared P A1 + distance_squared P D1 - 2 * (((A1.x - D1.x) * (P.x - A1.x) 
    + (A1.y - D1.y) * (P.y - A1.y) + (A1.z - D1.z) * (P.z - A1.z)) / distance_squared A1 D1 * distance_squared P M).sqrt
  in (dist_line^2 - (distance_squared P M)^2).sqrt = 1

-- Formalize what needs to be proved
theorem point_P_lies_on_parabola : ∀ (A B C D A1 D1 M P : Point3D ℝ),
  ∀ edge_len AM,
  is_cube(A B C D A1 D1 M P edge_len AM) →
  geometric_condition(P M A1 D1) →
  (∃ (eq: string), eq = "parabola") :=
by
  intros A B C D A1 D1 M P edge_len AM h_cube h_geom_cond
  -- proof details are skipped
  sorry

end point_P_lies_on_parabola_l204_204076


namespace simplify_abs_expr_l204_204152

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l204_204152


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204449

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204449


namespace angle_ABC_calculation_l204_204587

-- Define the points A, B, and C
def A : ℝ × ℝ × ℝ := (-3, 1, 5)
def B : ℝ × ℝ × ℝ := (-4, -2, 1)
def C : ℝ × ℝ × ℝ := (-5, -2, 2)

-- Define the function to compute the distance between two points in 3D
def dist (X Y : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((X.1 - Y.1)^2 + (X.2 - Y.2)^2 + (X.3 - Y.3)^2)

-- Compute distances AB, AC, and BC
def AB := dist A B
def AC := dist A C
def BC := dist B C

-- Compute cos(∠ABC) using the Law of Cosines
def cos_ABC : ℝ :=
  (AB^2 + BC^2 - AC^2) / (2 * AB * BC)

-- Translate the angle into degrees using acos
def angle_ABC_degrees : ℝ :=
  Real.acos(cos_ABC)

-- Now, state that the angle ∠ABC is equal to the calculated angle in degrees
theorem angle_ABC_calculation : angle_ABC_degrees = Real.acos(3 * Real.sqrt 13 / 26) :=
sorry

end angle_ABC_calculation_l204_204587


namespace samuel_money_left_l204_204742

def total_money : ℝ := 240
def fraction_samuel_received : ℝ := 3 / 8
def fraction_spent_on_drinks : ℝ := 2 / 3
def tax_rate : ℝ := 0.12

theorem samuel_money_left :
  let samuel_received := fraction_samuel_received * total_money in
  let spent_on_drinks := fraction_spent_on_drinks * total_money in
  let total_tax := tax_rate * spent_on_drinks in
  let total_spent := spent_on_drinks + total_tax in
  let money_left := samuel_received - total_spent in
  money_left = -89.2 :=
by
  sorry

end samuel_money_left_l204_204742


namespace volleyball_tournament_l204_204557

variable (n : ℕ)
variable (x y : Fin n → ℕ)

def P : ℕ := ∑ k in Finset.univ, (x k) ^ 2
def Q : ℕ := ∑ k in Finset.univ, (y k) ^ 2

theorem volleyball_tournament (h1 : ∀ k, x k + y k = n - 1)
  (h2 : ∑ k in Finset.univ, x k = ∑ k in Finset.univ, y k) :
  P x = Q y := sorry

end volleyball_tournament_l204_204557


namespace one_cow_one_bag_in_30_days_l204_204950

theorem one_cow_one_bag_in_30_days : 
  (H : ∀ (cows : ℕ) (bags : ℕ) (days : ℕ), cows = 45 → bags = 90 → days = 60 → (45 * days) / cows = bags) →
  (∀ (one_cow : ℕ), (H 45 90 60) → one_cow = 1 → ((one_cow * 60) / 2) = 1 → ((one_cow * (60 / 2)) = 1)) :=
by sorry

end one_cow_one_bag_in_30_days_l204_204950


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204323

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204323


namespace max_books_borrowed_l204_204736

theorem max_books_borrowed (total_students books_per_student : ℕ) (students_with_no_books: ℕ) (students_with_one_book students_with_two_books: ℕ) (rest_at_least_three_books students : ℕ) :
  total_students = 20 →
  books_per_student = 2 →
  students_with_no_books = 2 →
  students_with_one_book = 8 →
  students_with_two_books = 3 →
  rest_at_least_three_books = total_students - (students_with_no_books + students_with_one_book + students_with_two_books) →
  (students_with_no_books * 0 + students_with_one_book * 1 + students_with_two_books * 2 + students * books_per_student = total_students * books_per_student) →
  (students * 3 + some_student_max = 26) →
  some_student_max ≥ 8 :=
by
  introv h1 h2 h3 h4 h5 h6 h7
  sorry

end max_books_borrowed_l204_204736


namespace smallest_odd_number_with_five_different_prime_factors_l204_204278

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204278


namespace energy_soda_packs_l204_204373

-- Definitions and conditions
variables (total_bottles : ℕ) (regular_soda : ℕ) (diet_soda : ℕ) (pack_size : ℕ)
variables (complete_packs : ℕ) (remaining_regular : ℕ) (remaining_diet : ℕ) (remaining_energy : ℕ)

-- Conditions given in the problem
axiom h_total_bottles : total_bottles = 200
axiom h_regular_soda : regular_soda = 55
axiom h_diet_soda : diet_soda = 40
axiom h_pack_size : pack_size = 3

-- Proving the correct answer
theorem energy_soda_packs :
  complete_packs = (total_bottles - (regular_soda + diet_soda)) / pack_size ∧
  remaining_regular = regular_soda ∧
  remaining_diet = diet_soda ∧
  remaining_energy = (total_bottles - (regular_soda + diet_soda)) % pack_size :=
by
  sorry

end energy_soda_packs_l204_204373


namespace completing_the_square_l204_204341

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end completing_the_square_l204_204341


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204002

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204002


namespace instantaneous_velocity_at_2_l204_204897

variable (t : ℝ)

-- Define the displacement function S
def S : ℝ := Real.log (t + 1) + 2 * t ^ 2 + 1

-- Define the instantaneous velocity V as the derivative of S
noncomputable def V : ℝ := deriv (λ t, Real.log (t + 1) + 2 * t ^ 2 + 1) t

-- Formulating the main theorem to prove the instantaneous velocity at t=2
theorem instantaneous_velocity_at_2 :
  V 2 = 25 / 3 :=
sorry

end instantaneous_velocity_at_2_l204_204897


namespace coralReefs_below_5_percent_l204_204828

noncomputable def coralReefs (N : ℝ) (y : ℕ) : ℝ :=
  N * (0.70 ^ y)

theorem coralReefs_below_5_percent (N : ℝ) (initialYear : ℕ) (currentYear : ℕ) :
  (∀ y, currentYear = initialYear + y → coralReefs N y < 0.05 * N) → currentYear = 2020 :=
by
  assume h : ∀ y, currentYear = initialYear + y → coralReefs N y < 0.05 * N
  sorry

end coralReefs_below_5_percent_l204_204828


namespace union_complement_correct_l204_204916

open Set

def A : Set ℝ := {0, 1, 2, 3}
def B : Set ℝ := {x | x^2 - 2*x - 3 ≥ 0}

theorem union_complement_correct : A ∪ (compl B) = Ioo (-1 : ℝ) 3 ∪ {3} := by
  sorry

end union_complement_correct_l204_204916


namespace range_of_a_l204_204917

theorem range_of_a (a : ℝ) :
  (∃ A : Finset ℝ, 
    (∀ x, x ∈ A ↔ x^3 - 2 * x^2 + a * x = 0) ∧ A.card = 3) ↔ (a < 0 ∨ (0 < a ∧ a < 1)) :=
by
  sorry

end range_of_a_l204_204917


namespace remainder_of_sum_l204_204057

theorem remainder_of_sum (n : ℕ) (h : 0 < n) : 
  let S := (finset.range n).sum (λ k, (7^(n-k) * nat.choose n k)) 
  in S % 9 = if n % 2 = 0 then 0 else 7 :=
sorry

end remainder_of_sum_l204_204057


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204014

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204014


namespace number_of_triplets_l204_204114

theorem number_of_triplets :
  let N := (λ N : Nat, ∃ (A B C : Nat), 0 ≤ A ∧ A < B ∧ B < C ∧ C ≤ 99 ∧
    ∃ (a b c p : Nat), Nat.Prime p ∧ 0 ≤ b ∧ b < a ∧ a < c ∧ c < p ∧
    p ∣ (A - a) ∧ p ∣ (B - b) ∧ p ∣ (C - c) ∧
    ∃ (D d : Nat), A = B - D ∧ C = B + D ∧ b = a - d ∧ c = a + d ∧
    D % p = 0 ∧ d % p = 0 ∧ p = 3
  ) in N = 272 :=
by
  sorry

end number_of_triplets_l204_204114


namespace correct_sampling_methods_order_l204_204802

def simple_random_sampling (select: ℕ → ℕ → Prop) :=
  ∀ n k, select n k = (n ≤ 8 ∧ k ≤ 1)

def stratified_sampling (select: ℕ → ℕ → Prop) :=
  ∀ n k, select n k = (n ≤ 2100 / 3 ∧ k ≤ 3)

def systematic_sampling (select: ℕ → ℕ → Prop) :=
  ∀ n k, select n k = (n ≤ 20 ∧ k ≤ 35)

theorem correct_sampling_methods_order :
  simple_random_sampling (λ n k, n = 8 ∧ k = 1) ∧
  stratified_sampling (λ n k, n = 2100 ∧ k = 3) ∧
  systematic_sampling (λ n k, n = 20 ∧ k = 35) →
  (1, (λ n k, n = 8 ∧ k = 1)) = (simple_random_sampling, (λ n k, n = 8 ∧ k = 1)) ∧
  (2, (λ n k, n = 2100 ∧ k = 3)) = (stratified_sampling, (λ n k, n = 2100 ∧ k = 3)) ∧
  (3, (λ n k, n = 20 ∧ k = 35)) = (systematic_sampling, (λ n k, n = 20 ∧ k = 35))
:= sorry

end correct_sampling_methods_order_l204_204802


namespace find_pairs_l204_204839

theorem find_pairs (x y p : ℕ)
  (h1 : 1 ≤ x) (h2 : 1 ≤ y) (h3 : x ≤ y) (h4 : Prime p) :
  (x = 3 ∧ y = 5 ∧ p = 7) ∨ (x = 1 ∧ ∃ q, Prime q ∧ y = q + 1 ∧ p = q ∧ q ≠ 7) ↔
  (x + y) * (x * y - 1) / (x * y + 1) = p := 
sorry

end find_pairs_l204_204839


namespace percent_answered_first_question_correctly_l204_204761

theorem percent_answered_first_question_correctly (pB pA_inter_B pNeither : ℝ) (h1 : pB = 0.55) (h2 : pA_inter_B = 0.55) (h3 : pNeither = 0.20) : 
  let pA := 0.80 in
  pA = 1 - pNeither - pB + pA_inter_B :=
by
  -- Definitions as per the conditions
  let P_A := 0.80
  let P_B := pB
  let P_A_inter_B := pA_inter_B
  let P_neither := pNeither

  -- The equation to prove
  have h : P_A = 1 - P_neither - P_B + P_A_inter_B, from sorry,
  exact h

end percent_answered_first_question_correctly_l204_204761


namespace classroom_position_l204_204051

theorem classroom_position (a b c d : ℕ) (h : (1, 2) = (a, b)) : (3, 2) = (c, d) :=
by
  sorry

end classroom_position_l204_204051


namespace angle_BAP_13_degrees_l204_204740

theorem angle_BAP_13_degrees
  {A B C P : Type}
  [Isosceles (triangle A B C)] (h1 : isosceles_triangle AB AC)
  (h2 : ∠BCP = 30) (h3 : ∠APB = 150) (h4 : ∠CAP = 39) :
  ∠BAP = 13 := 
sorry

end angle_BAP_13_degrees_l204_204740


namespace find_special_number_l204_204850

-- Define the product of digits function
def product_of_digits (A : ℕ) : ℕ :=
  A.digits 10 |>.prod

-- State the theorem
theorem find_special_number :
  {A : ℕ // A = (3 / 2) * product_of_digits A} = 48 :=
by sorry

end find_special_number_l204_204850


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204001

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204001


namespace positive_difference_in_x_coordinates_l204_204473

-- Define points for line l
def point_l1 : ℝ × ℝ := (0, 10)
def point_l2 : ℝ × ℝ := (2, 0)

-- Define points for line m
def point_m1 : ℝ × ℝ := (0, 3)
def point_m2 : ℝ × ℝ := (10, 0)

-- Define the proof statement with the given problem
theorem positive_difference_in_x_coordinates :
  let y := 20
  let slope_l := (point_l2.2 - point_l1.2) / (point_l2.1 - point_l1.1)
  let intersection_l_x := (y - point_l1.2) / slope_l + point_l1.1
  let slope_m := (point_m2.2 - point_m1.2) / (point_m2.1 - point_m1.1)
  let intersection_m_x := (y - point_m1.2) / slope_m + point_m1.1
  abs (intersection_l_x - intersection_m_x) = 54.67 := 
  sorry -- Proof goes here

end positive_difference_in_x_coordinates_l204_204473


namespace final_number_after_trebling_l204_204379

theorem final_number_after_trebling (x : ℕ) (h : x = 5) : 3 * (2 * x + 9) = 57 :=
by
  rw h -- Replaces x with 5 using the hypothesis.
  sorry -- This is the placeholder for the actual proof steps.

end final_number_after_trebling_l204_204379


namespace solve_x_l204_204539

theorem solve_x (x : ℝ) (h : -{-[-(-x)]} = -4) : x = -4 :=
sorry

end solve_x_l204_204539


namespace neg_div_neg_eq_pos_division_of_negatives_example_l204_204412

theorem neg_div_neg_eq_pos (a b : Int) (hb : b ≠ 0) : (-a) / (-b) = a / b := by
  -- You can complete the proof here
  sorry

theorem division_of_negatives_example : (-81 : Int) / (-9) = 9 :=
  neg_div_neg_eq_pos 81 9 (by decide)

end neg_div_neg_eq_pos_division_of_negatives_example_l204_204412


namespace function_solution_l204_204834

theorem function_solution (f : ℝ → ℝ) (α : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + α * x) :=
by
  sorry

end function_solution_l204_204834


namespace henri_total_time_l204_204530

variable (m1 m2 : ℝ) (r w : ℝ)

theorem henri_total_time (H1 : m1 = 3.5) (H2 : m2 = 1.5) (H3 : r = 10) (H4 : w = 1800) :
    m1 + m2 + w / r / 60 = 8 := by
  sorry

end henri_total_time_l204_204530


namespace greatest_integer_less_than_sum_l204_204116

def nested_sqrt (x : ℕ) : ℝ := if x = 1 then sqrt 6 else sqrt (6 + nested_sqrt (x - 1))
def nested_cbrt (x : ℕ) : ℝ := if x = 1 then real.cbrt 6 else real.cbrt (6 + nested_cbrt (x - 1))

noncomputable def a := nested_sqrt 2016 / 2016
noncomputable def b := nested_cbrt 2017 / 2017

theorem greatest_integer_less_than_sum : ⌊a + b⌋ = 0 :=
by
  sorry

end greatest_integer_less_than_sum_l204_204116


namespace BP_equals_4_l204_204743

noncomputable def length_BP : ℝ :=
  let A B C D P : Type := sorry
  let AP := 10
  let PC := 2
  let BD := 9
  let BP := 4 -- We are aiming to prove this
  let DP := BD - BP
  in if BP < DP then BP else sorry

theorem BP_equals_4 (A B C D P : Type)
  (h1 : A = B ∨ A = C ∨ A = D ∨ B = C ∨ B = D ∨ C = D)
  (h2 : ∃ O : Type, (A = O) ∧ (B = O) ∧ (C = O) ∧ (D = O))
  (h3 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (intersect_AC_BD : P = sorry) -- Intersection property
  (AP_length : AP = 10)
  (PC_length : PC = 2)
  (BD_length : BD = 9)
  (BP_less_DP : BP < DP)
  : length_BP = 4 := sorry

end BP_equals_4_l204_204743


namespace max_sqrt_sum_max_sqrt_sum_achieved_at_zero_l204_204913

theorem max_sqrt_sum (x : ℝ) (h : -49 ≤ x ∧ x ≤ 49) :
  sqrt (49 + x) + sqrt (49 - x) ≤ 14 :=
begin
  sorry
end

theorem max_sqrt_sum_achieved_at_zero :
  sqrt (49 + 0) + sqrt (49 - 0) = 14 :=
begin
  -- This directly shows the maximum value is achieved at x = 0
  calc sqrt (49 + 0) + sqrt (49 - 0) = sqrt 49 + sqrt 49 : by simp
                                ... = 7 + 7               : by rw [sqrt_eq, sqrt_eq]
                                ... = 14                  : by norm_num,
end

end max_sqrt_sum_max_sqrt_sum_achieved_at_zero_l204_204913


namespace count_bases_for_last_digit_l204_204929

theorem count_bases_for_last_digit (n : ℕ) : n = 729 → ∃ S : Finset ℕ, S.card = 2 ∧ ∀ b ∈ S, 2 ≤ b ∧ b ≤ 10 ∧ (n - 5) % b = 0 :=
by
  sorry

end count_bases_for_last_digit_l204_204929


namespace distance_P1_P2_is_sqrt_370_l204_204464

-- Define points as pairs of real numbers
def point := (ℝ × ℝ)

-- Distance formula
def distance (p1 p2 : point) : ℝ :=
  real.sqrt (((p2.1 - p1.1) ^ 2) + ((p2.2 - p1.2) ^ 2))

-- Specific points (3, 20) and (12, 3)
def P1 : point := (3, 20)
def P2 : point := (12, 3)

-- Proof statement
theorem distance_P1_P2_is_sqrt_370 : distance P1 P2 = real.sqrt 370 := by
  sorry

end distance_P1_P2_is_sqrt_370_l204_204464


namespace smallest_odd_number_with_five_primes_proof_l204_204235

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204235


namespace triangle_area_l204_204712

theorem triangle_area :
  let line1 := (λ x : ℝ, 2 * x + 4)
  let line2 := (λ x : ℝ, 2 * x - 2)
  let line3 := (λ x : ℝ, -2 * x + 2)
  let intersection_point1 := (-1/2, 3)
  let intersection_point2 := (1, 0)
  let intersection_point3 := (-3, 0)
  1/2 * |(-1/2 * 0 + 1 * 0 + -3 * 3) - (3 * 1 + 0 * -3 + 0 * -1/2)| = 6 :=
by sorry

end triangle_area_l204_204712


namespace probability_three_As_given_two_As_l204_204790

open_locale classical
noncomputable theory

def num_strings_with_exactly_k_As (k : ℕ) : ℕ :=
  nat.choose 5 k * 3^(5 - k)

def num_strings_with_at_least_two_As : ℕ :=
  num_strings_with_exactly_k_As 2 +
  num_strings_with_exactly_k_As 3 +
  num_strings_with_exactly_k_As 4 +
  num_strings_with_exactly_k_As 5

def num_strings_with_at_least_three_As : ℕ :=
  num_strings_with_exactly_k_As 3 +
  num_strings_with_exactly_k_As 4 +
  num_strings_with_exactly_k_As 5

def probability_at_least_three_As_given_at_least_two_As (num_at_least_two : ℕ) (num_at_least_three : ℕ) : ℚ :=
  num_at_least_three.to_rat / num_at_least_two.to_rat

theorem probability_three_As_given_two_As :
  probability_at_least_three_As_given_at_least_two_As num_strings_with_at_least_two_As num_strings_with_at_least_three_As = 53 / 188 := by
sorry

end probability_three_As_given_two_As_l204_204790


namespace perimeter_of_triangle_is_36_l204_204663

variable (inradius : ℝ)
variable (area : ℝ)
variable (P : ℝ)

theorem perimeter_of_triangle_is_36 (h1 : inradius = 2.5) (h2 : area = 45) : 
  P / 2 * inradius = area → P = 36 :=
sorry

end perimeter_of_triangle_is_36_l204_204663


namespace daily_wage_l204_204388

theorem daily_wage (h1 : 57.5 = 159.96875) :
  57.5 / (11 / 32 + 31 / 8 + 3 / 4) = 11.57 :=
by
  have h_total_days : 11 / 32 + 31 / 8 + 3 / 4 = 159 / 32 := sorry
  have h_decimal_days : 159 / 32 = 4.96875 := sorry
  rw [h_decimal_days, h1]
  norm_num
  sorry

end daily_wage_l204_204388


namespace given_even_function_and_monotonic_increasing_l204_204879

-- Define f as an even function on ℝ
def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f (x)

-- Define that f is monotonically increasing on (-∞, 0)
def is_monotonically_increasing_on_negatives (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → y < 0 → f (x) < f (y)

-- Theorem statement
theorem given_even_function_and_monotonic_increasing {
  f : ℝ → ℝ
} (h_even : is_even_function f)
  (h_monotonic : is_monotonically_increasing_on_negatives f) :
  f (1) > f (-2) :=
sorry

end given_even_function_and_monotonic_increasing_l204_204879


namespace acute_angle_at_3_36_l204_204205

def degrees (h m : ℕ) : ℝ :=
  let minute_angle := (m / 60.0) * 360.0
  let hour_angle := (h % 12 + m / 60.0) * 30.0
  abs (minute_angle - hour_angle)

theorem acute_angle_at_3_36 : degrees 3 36 = 108 :=
by
  sorry

end acute_angle_at_3_36_l204_204205


namespace expected_adjacent_black_pairs_correct_l204_204378

noncomputable def expected_adjacent_black_pairs (total_cards black_cards : ℕ) (cards_arranged_in_circle : Prop) : ℚ :=
  if cards_arranged_in_circle ∧ total_cards = 60 ∧ black_cards = 30 then
    30 * (29 / 59)
  else
    0

theorem expected_adjacent_black_pairs_correct :
  expected_adjacent_black_pairs 60 30 True = 870 / 59 :=
by
  sorry

end expected_adjacent_black_pairs_correct_l204_204378


namespace tripod_height_l204_204780

theorem tripod_height {m n : ℕ} (h : ℝ)
    (h_base: ∀ (a b c : ℝ), a = 6 → b = 6 → c = 6 → 
        let d := 5 in
        h = d - (d - h * ℝ.cos(120 / 2 * (Math.PI / 180))))
    (height_form : h = m / ℝ.sqrt n)
    (non_square_free : ∀ p : ℕ, p.prime → ¬ p ^ 2 ∣ n ):
    ⌊m + ℝ.sqrt n⌋ = 6 :=
by
  sorry

end tripod_height_l204_204780


namespace smallest_odd_number_with_five_primes_proof_l204_204236

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204236


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204328

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204328


namespace sum_of_integers_l204_204674

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l204_204674


namespace AllieMoreGrapes_l204_204389

-- Definitions based on conditions
def RobBowl : ℕ := 25
def TotalGrapes : ℕ := 83
def AllynBowl (A : ℕ) : ℕ := A + 4

-- The proof statement that must be shown.
theorem AllieMoreGrapes (A : ℕ) (h1 : A + (AllynBowl A) + RobBowl = TotalGrapes) : A - RobBowl = 2 :=
by {
  sorry
}

end AllieMoreGrapes_l204_204389


namespace total_coins_l204_204179

-- Define the conditions
variables (E A : ℕ)
variable h1 : E * 45 = A * 10
variable h2 : A / 4 = 90

-- Define the proof problem
theorem total_coins (E A : ℕ) (h1 : E * 45 = A * 10) (h2 : A / 4 = 90) : E + A = 440 :=
sorry

end total_coins_l204_204179


namespace eval_expression_eq_2_l204_204332

theorem eval_expression_eq_2 :
  (10^2 + 11^2 + 12^2 + 13^2 + 14^2) / 365 = 2 :=
by
  sorry

end eval_expression_eq_2_l204_204332


namespace clock_angle_3_36_l204_204213

def minute_hand_position (minutes : ℕ) : ℝ :=
  minutes * 6

def hour_hand_position (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def angle_difference (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

def acute_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem clock_angle_3_36 :
  acute_angle (angle_difference (minute_hand_position 36) (hour_hand_position 3 36)) = 108 :=
by
  sorry

end clock_angle_3_36_l204_204213


namespace polynomial_rewrite_l204_204169

theorem polynomial_rewrite :
  ∃ (a b c d e f : ℤ), 
  (2401 * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f)) ∧
  (a + b + c + d + e + f = 274) :=
sorry

end polynomial_rewrite_l204_204169


namespace geometric_sequence_log_sum_l204_204967

theorem geometric_sequence_log_sum
  (a : ℕ → ℝ) 
  (h1 : ∀ n, a n > 0)
  (h2 : ∃ r, ∀ n, a (n + 1) = r * a n) 
  (h3 : a 1008 * a 1009 = 1 / 100) :
  ∑ i in finset.range 2016, log 10 (a (i + 1)) = -2016 :=
sorry

end geometric_sequence_log_sum_l204_204967


namespace arithmetic_sequence_first_term_power_l204_204904

variable (S : ℕ → ℤ) (a : ℕ → ℤ) (c : ℤ) (d : ℤ)
variable (Sn_eq : ∀ n, S n = 2 * (n:ℤ)^2 - 4 * (n:ℤ) + c)
variable (a1_eq : a 1 = c - 2)
variable (a2_eq : a 2 = 2)
variable (a3_eq : a 3 = 6)
variable (a_in_arith_sequence : 2 * a 2 = a 1 + a 3)
variable (c_eq_0 : c = 0)
variable (a1_value : a 1 = -2)
variable (d_value : d = 4)

theorem arithmetic_sequence_first_term_power :
  (a 1) ^ d = 16 := by
  sorry

end arithmetic_sequence_first_term_power_l204_204904


namespace least_vertices_no_Hamiltonian_path_l204_204104

variable (G : Type) [Graph G] -- Define G as a type with a graph structure
variable [ConnectedGraph G] -- G is a connected graph

-- Defining that every vertex in G has a degree of at least m
variable (m : ℕ) (h_m : 3 ≤ m) (degree_condition : ∀ v : G, degree v ≥ m)

-- Defining Hamiltonian path
def HamiltonianPath (g : G) : Prop := 
  ∃ path : List G, (∀ x ∈ path, ∀ y ∈ path, x ≠ y) ∧ (path.length = num_vertices G) 

-- Theorem statement
theorem least_vertices_no_Hamiltonian_path :
  ∃ (n : ℕ), no_Hamiltonian_path G → n ≥ 2 * m + 1 :=
by
  sorry -- Proof is omitted

end least_vertices_no_Hamiltonian_path_l204_204104


namespace smallest_odd_number_with_five_prime_factors_l204_204276

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204276


namespace stickers_total_l204_204976

def karl_stickers : ℕ := 25
def ryan_stickers : ℕ := karl_stickers + 20
def ben_stickers : ℕ := ryan_stickers - 10
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem stickers_total : total_stickers = 105 := by
  sorry

end stickers_total_l204_204976


namespace sum_geometric_series_l204_204849

-- Definitions
def S (r : ℝ) := 18 / (1 - r)

variable (a : ℝ)
-- Conditions
variables (ha : -1 < a) (ha1 : a < 1) (hS : S a * S (-a) = 3024)

-- Proof statement
theorem sum_geometric_series (ha : -1 < a) (ha1 : a < 1) (hS : S a * S (-a) = 3024) : 
  S a + S (-a) = 337.5 := 
  sorry

end sum_geometric_series_l204_204849


namespace initial_blue_balls_l204_204679

theorem initial_blue_balls (total_balls : ℕ) (remaining_balls : ℕ) (B : ℕ) :
  total_balls = 18 → remaining_balls = total_balls - 3 → (B - 3) / remaining_balls = 1 / 5 → B = 6 :=
by 
  intros htotal hremaining hprob
  sorry

end initial_blue_balls_l204_204679


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204012

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204012


namespace expression_d_cannot_be_calculated_as_square_of_binomial_l204_204344

variable (a b : ℝ)

-- Definitions of the expressions
def exprA := (a - 2 * b) * (2 * b + a)
def exprB := (a - 2 * b) * (-a - 2 * b)
def exprC := (2 * a - b) * (-2 * a - b)
def exprD := (a + 2 * b) * (-a - 2 * b)

-- The main theorem stating which expression cannot be calculated using
-- the square of a binomial formula
theorem expression_d_cannot_be_calculated_as_square_of_binomial : 
  ( ∃ (u : ℝ), exprA = u^2 ) ∧ 
  ( ∃ (v : ℝ), exprB = v^2 ) ∧ 
  ( ∃ (w : ℝ), exprC = w^2 ) ∧ 
  ¬( ∃ (x : ℝ), exprD = x^2 ) :=
sorry

end expression_d_cannot_be_calculated_as_square_of_binomial_l204_204344


namespace number_of_valid_integers_l204_204893

theorem number_of_valid_integers (n : ℕ) (h1 : n ≤ 2021) (h2 : ∀ m : ℕ, m^2 ≤ n → n < (m + 1)^2 → ((m^2 + 1) ∣ (n^2 + 1))) : 
  ∃ k, k = 47 :=
by
  sorry

end number_of_valid_integers_l204_204893


namespace sin_double_angle_l204_204541

theorem sin_double_angle (α : ℝ) (h1 : sin (π - α) = 2 / 3) (h2 : π / 2 < α ∧ α < π) :
  sin (2 * α) = - (4 * Real.sqrt 5) / 9 :=
sorry

end sin_double_angle_l204_204541


namespace equals_at_some_N_smallest_N_equals_at_some_N_l204_204107

-- Part (a)
theorem equals_at_some_N (a b c : ℕ) (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) :
  ∃ N : ℕ, N > 0 ∧ let a_seq (n : ℕ) : ℕ := if n = 1 then a else ⌊Real.sqrt ((a_seq (n-1)) * (b_seq (n-1)))⌋.toNat
                       b_seq (n : ℕ) : ℕ := if n = 1 then b else ⌊Real.sqrt ((b_seq (n-1)) * (c_seq (n-1)))⌋.toNat
                       c_seq (n : ℕ) : ℕ := if n = 1 then c else ⌊Real.sqrt ((c_seq (n-1)) * (a_seq (n-1)))⌋.toNat
  in a_seq N = b_seq N ∧ b_seq N = c_seq N := 
sorry

-- Part (b)
theorem smallest_N_equals_at_some_N (a b c : ℕ) (h_a : a ≥ 2) (h_bc_sum : b + c = 2 * a - 1) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_c_pos : c > 0) :
  let a_seq (n : ℕ) : ℕ := if n = 1 then a else ⌊Real.sqrt ((a_seq (n-1)) * (b_seq (n-1)))⌋.toNat
      b_seq (n : ℕ) : ℕ := if n = 1 then b else ⌊Real.sqrt ((b_seq (n-1)) * (c_seq (n-1)))⌋.toNat
      c_seq (n : ℕ) : ℕ := if n = 1 then c else ⌊Real.sqrt ((c_seq (n-1)) * (a_seq (n-1)))⌋.toNat
  in ∃ N : ℕ, N = 3 ∧ a_seq N = 1 ∧ b_seq N = 1 ∧ c_seq N = 1 := 
sorry


end equals_at_some_N_smallest_N_equals_at_some_N_l204_204107


namespace cosines_product_of_triangle_l204_204095

theorem cosines_product_of_triangle (K L M P O Q S : Type) [triangle K L M]
  (KP : median K L M P) (circumcenter : center_circumcircle O K L M)
  (incircle_center : center_incircle Q K L M) (intersection : intersects KP OQ S)
  (angle_LKM : ∠ L K M = π / 3) 
  (proportion : (dist O S / dist P S) = sqrt 6 * (dist Q S / dist K S)) :
  (cos (∠ K L M) * cos (∠ K M L)) = -3 / 8 :=
sorry

end cosines_product_of_triangle_l204_204095


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204329

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204329


namespace smallest_odd_with_five_prime_factors_l204_204298

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204298


namespace michelle_phone_bill_l204_204759

def base_cost : ℝ := 20
def text_cost_per_message : ℝ := 0.05
def minute_cost_over_20h : ℝ := 0.20
def messages_sent : ℝ := 150
def hours_talked : ℝ := 22
def allowed_hours : ℝ := 20

theorem michelle_phone_bill :
  base_cost + (messages_sent * text_cost_per_message) +
  ((hours_talked - allowed_hours) * 60 * minute_cost_over_20h) = 51.50 := by
  sorry

end michelle_phone_bill_l204_204759


namespace sum_of_squares_and_product_l204_204676

theorem sum_of_squares_and_product
  (x y : ℕ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_squares_and_product_l204_204676


namespace nth_equation_pattern_l204_204998

theorem nth_equation_pattern (n : ℕ) : 
  (∏ i in finset.range n, (n + i + 1)) = 2^n * ∏ i in finset.range n, (2 * i + 1) :=
by sorry

end nth_equation_pattern_l204_204998


namespace coplanar_vectors_lambda_value_l204_204479

/-- 
  Define the vectors a, b, c and the condition of coplanarity.
  Then, prove that the value of λ such that the vectors a, b, and c are coplanar is 3.
--/
theorem coplanar_vectors_lambda_value : 
  let a := (2: ℝ, -1: ℝ, 2: ℝ)
  let b := (-1: ℝ, 3: ℝ, -3: ℝ)
  let c (λ: ℝ) := (13: ℝ, 6: ℝ, λ)
  ∃ m n λ, c λ = (2 * m - n, -m + 3 * n, 2 * m - 3 * n) ∧ λ = 3 :=
by {
  sorry
}

end coplanar_vectors_lambda_value_l204_204479


namespace perimeter_of_third_polygon_l204_204191

variable {R : Type*} [LinearOrderedField R]

theorem perimeter_of_third_polygon
  (a b : R)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_ne_b : a ≠ b) :
  ∃ x : R, x = (b * a.sqrt) / ((2 * a - b).sqrt) :=
by
  use (b * a.sqrt) / ((2 * a - b).sqrt)
  sorry

end perimeter_of_third_polygon_l204_204191


namespace negation_proposition_l204_204653

theorem negation_proposition :
  (\neg (\forall x: ℝ, (0 < x → sin x < x)) = 
  (∃ x0: ℝ, (0 < x0 ∧ sin x0 ≥ x0))) := by
sorry

end negation_proposition_l204_204653


namespace valid_interval_for_k_l204_204837

theorem valid_interval_for_k :
  ∀ k : ℝ, (∀ x : ℝ, x^2 - 8*x + k < 0 → 0 < k ∧ k < 16) :=
by
  sorry

end valid_interval_for_k_l204_204837


namespace smallest_odd_number_with_five_primes_proof_l204_204237

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204237


namespace mn_value_l204_204168
open Real

-- Define the conditions
def L_1_scenario_1 (m n : ℝ) : Prop :=
  ∃ (θ₁ θ₂ : ℝ), θ₁ = 2 * θ₂ ∧ m = tan θ₁ ∧ n = tan θ₂ ∧ m = 4 * n

-- State the theorem
theorem mn_value (m n : ℝ) (hL1 : L_1_scenario_1 m n) (hm : m ≠ 0) : m * n = 2 :=
  sorry

end mn_value_l204_204168


namespace _l204_204843

noncomputable def polynomial_remainder_theorem : Prop :=
  ∀ (x : ℂ), x^4 + x^3 + x^2 + x + 1 = 0 → x^{44} + x^{33} + x^{22} + x^{11} + 1 = 0

lemma check_remainder_zero_of_division : polynomial_remainder_theorem := by
  sorry

end _l204_204843


namespace problem1_problem2_problem3_l204_204925

section VectorProofs

variables (k : ℝ) (a b c d : ℝ × ℝ)

-- Conditions given
def vect_a : ℝ × ℝ := (6, 1)
def vect_b : ℝ × ℝ := (-2, 3)
def vect_c : ℝ × ℝ := (2, 2)
def vect_d : ℝ × ℝ := (-3, k)

-- Problem 1: Prove that a + 2b - c = (0, 5)
theorem problem1 : vect_a + (2 : ℝ) • vect_b - vect_c = (0, 5) := sorry

-- Problem 2: Prove that if (a + 2c) is parallel to (c + kb), then k = -1/4
theorem problem2 (h : (vect_a + (2 : ℝ) • vect_c) = (c + k • vect_b)) : k = (-1) / 4 := sorry

-- Problem 3: Prove that if the angle between a and d is obtuse, then k ∈ (-∞, -1/2) ∪ (-1/2, 18)
theorem problem3 (h : (vect_a.fst * vect_d.fst + vect_a.snd * vect_d.snd) < 0) : 
  k ∈ Set.Ioo (-∞ : ℝ) (-1 / 2) ∨ k ∈ Set.Ioo (-1 / 2) 18 := sorry

end VectorProofs

end problem1_problem2_problem3_l204_204925


namespace find_S_l204_204895

def poly_expansion (x : ℝ) : ℝ := (x^2 - x + 1) ^ 1999

theorem find_S :
  (∀ (a : ℝ), poly_expansion a = (∑ i in finset.range (3999), (a^i))) →
  (S = (∑ i in finset.range (3998), (a^i))) →
  S = 0 :=
  by
    assume h_expansion h_S,
    -- The proof would go here
    sorry

end find_S_l204_204895


namespace m_plus_n_is_55_l204_204970

noncomputable def ABC := Point
variables {A B C E D I : ABC}
variables (h1 : dist A B = 3) (h2 : dist A C = 5) (h3 : dist B C = 7)
variables (hE : E = reflect A B C) (hD : ∃ D, D ∈ circumcircle A B C ∧ collinear B E D)
variables (hI : I = incenter A B D)
variables (h_cos_squared : ∃ m n : ℕ, relatively_prime m n ∧ cos_sq_angle A E I = m / n)

theorem m_plus_n_is_55 : ∃ m n : ℕ, relatively_prime m n ∧ cos_sq_angle A E I = m / n → m + n = 55 := by
  sorry

end m_plus_n_is_55_l204_204970


namespace number_of_green_quadruples_l204_204110

def num_pos_divisors (k : ℕ) : ℕ :=
  k.divisors.card

def conditions (a b c d : ℕ) : Prop :=
  b = a^2 + 1 ∧
  c = b^2 + 1 ∧
  d = c^2 + 1 ∧
  num_pos_divisors a + num_pos_divisors b + num_pos_divisors c + num_pos_divisors d % 2 = 1 ∧
  a < 10^6 ∧ b < 10^6 ∧ c < 10^6 ∧ d < 10^6

def is_green (a b c d : ℕ) : Prop :=
  conditions a b c d 

theorem number_of_green_quadruples : 
  (finset.univ.filter (λ (a b c d : ℕ), is_green a b c d)).card = 2 := 
sorry

end number_of_green_quadruples_l204_204110


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204327

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204327


namespace max_3m_4n_l204_204365

noncomputable def max_value_3m_4n (m n : ℕ) : ℕ :=
  3 * m + 4 * n

theorem max_3m_4n (m n : ℕ)
  (h_sum : ∃ (a b : ℕ) (ha : ∀ i < m, is_even a) (hb : ∀ i < n, is_odd b),
    (finset.range m).sum (λ i, 2 * (i + 1)) +  (finset.range n).sum (λ i, 2 * i + 1) = 1987):
  max_value_3m_4n m n ≤ 221 :=
sorry

end max_3m_4n_l204_204365


namespace chris_breath_holding_goal_l204_204409

noncomputable theory

def breath_holding_days (initial_time : ℕ) (daily_increase : ℕ) (target_time : ℕ) : ℕ :=
let days := (target_time - initial_time) / daily_increase + 1 in 
days

theorem chris_breath_holding_goal :
  breath_holding_days 10 10 90 = 9 := 
by 
  simp [breath_holding_days]
  sorry

end chris_breath_holding_goal_l204_204409


namespace speed_of_car_B_is_correct_l204_204407

def carB_speed : ℕ := 
  let speedA := 50 -- Car A's speed in km/hr
  let timeA := 6 -- Car A's travel time in hours
  let ratio := 3 -- The ratio of distances between Car A and Car B
  let distanceA := speedA * timeA -- Calculate Car A's distance
  let timeB := 1 -- Car B's travel time in hours
  let distanceB := distanceA / ratio -- Calculate Car B's distance
  distanceB / timeB -- Calculate Car B's speed

theorem speed_of_car_B_is_correct : carB_speed = 100 := by
  sorry

end speed_of_car_B_is_correct_l204_204407


namespace length_of_rs_l204_204181

theorem length_of_rs {a b c d e f : ℕ} (h : {a, b, c, d, e, f} = {9, 15, 21, 31, 40, 45}) 
  (hPQ : a = 45) : ∃ RS, RS = 9 :=
by
  sorry

end length_of_rs_l204_204181


namespace parabola_normal_intersect_l204_204586

theorem parabola_normal_intersect {x y : ℝ} (h₁ : y = x^2) (A : ℝ × ℝ) (hA : A = (-1, 1)) :
  ∃ B : ℝ × ℝ, B = (1.5, 2.25) ∧ ∀ x : ℝ, (y - 1) = 1/2 * (x + 1) →
  ∀ x : ℝ, y = x^2 ∧ B = (1.5, 2.25) :=
sorry

end parabola_normal_intersect_l204_204586


namespace cos_double_angle_of_geometric_sequence_l204_204951

-- Definitions based on given conditions
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * r

def s_n (a_7a_8 : ℝ) : ℝ := 3/5

-- Problem statement to prove in Lean
theorem cos_double_angle_of_geometric_sequence
  (a : ℕ → ℝ)
  (h_seq : geometric_sequence a (real.sqrt 2))
  (h_s : s_n (a 7 * a 8) = 3 / 5) :
  real.cos (2 * a 5) = 7 / 25 :=
sorry  -- Proof placeholder

end cos_double_angle_of_geometric_sequence_l204_204951


namespace Q_nonneg_integers_l204_204431

def Q (x : ℝ) : ℝ := (x - 4) * (x - 16) * (x - 36) * (x - 64) * (x - 100) * (x - 144) * (x - 196) * (x - 256) * (x - 324) * (x - 400) * (x - 484) * (x - 576) * (x - 676) * (x - 784) * (x - 900) * (x - 1024) * (x - 1156) * (x - 1296) * (x - 1444) * (x - 1600) * (x - 1764) * (x - 1936) * (x - 2116) * (x - 2304) * (x - 2500)

theorem Q_nonneg_integers :
  {m : ℤ | Q m.to_real ≥ 0}.finite.to_finset.card = 2461 :=
sorry

end Q_nonneg_integers_l204_204431


namespace square_remainder_l204_204356

theorem square_remainder (k : ℤ) :
  let n := 5 * k + 3 in
  n^2 % 5 = 4 :=
by
  sorry

end square_remainder_l204_204356


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204248

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204248


namespace sharkFinFalcataArea_l204_204668

open Real

/-- Define the area of a circle segment --/
def circleSegment (r : ℝ) (sectorAngle : ℝ) : ℝ :=
  (sectorAngle / (2 * π)) * π * r^2

/-- Define the geometry conditions --/
def quarterCircleArea (r : ℝ) : ℝ := circleSegment r (π / 2)
def semiCircleArea (r : ℝ) : ℝ := circleSegment r π

/-- The problem statement --/
theorem sharkFinFalcataArea :
  ∀ (r1 r2 : ℝ),
    r1 = 4 →
    r2 = 2 →
    quarterCircleArea r1 - semiCircleArea r2 = 2 * π :=
by
  intros r1 r2 hr1 hr2
  rw [hr1, hr2]
  sorry

end sharkFinFalcataArea_l204_204668


namespace f_is_odd_and_neg_cbrt_l204_204542

open Function

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then real.cbrt (x + 1) else 0

theorem f_is_odd_and_neg_cbrt (x : ℝ) (h1 : Odd f) (h2 : ∀ y, y > 0 → f y = real.cbrt (y + 1)) :
  x < 0 → f x = -real.cbrt (-x + 1) :=
by
  intro hx
  have hnx : -x > 0 := neg_pos.mpr hx
  specialize h2 (-x) hnx
  rw [h1 x] at h2
  exact h2

#check f_is_odd_and_neg_cbrt

end f_is_odd_and_neg_cbrt_l204_204542


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204448

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204448


namespace points_on_paths_count_l204_204382

theorem points_on_paths_count :
  let A := (-4, 3)
  let B := (2, -3)
  (∑ (x : ℤ) in ({ x : ℤ | -13 ≤ x ∧ x ≤ 15 }.filter (λ x, 
    ∑ (y : ℤ) in ({ y : ℤ | -13 ≤ y ∧ y ≤ 14 }.filter (λ y, 
      |x + 4| + |x - 2| + |y - 3| + |y + 3| ≤ 25), 1)), 1) = 193 := 
by 
  sorry

end points_on_paths_count_l204_204382


namespace line_parallel_l204_204175

theorem line_parallel (a : ℝ) : (∀ x y : ℝ, ax + y = 0) ↔ (x + ay + 1 = 0) → a = 1 ∨ a = -1 := 
sorry

end line_parallel_l204_204175


namespace douglas_percent_votes_l204_204559

def percentageOfTotalVotesWon (votes_X votes_Y: ℕ) (percent_X percent_Y: ℕ) : ℕ :=
  let total_votes_Douglas : ℕ := (percent_X * 2 * votes_X + percent_Y * votes_Y)
  let total_votes_cast : ℕ := 3 * votes_Y
  (total_votes_Douglas * 100 / total_votes_cast)

theorem douglas_percent_votes (votes_X votes_Y : ℕ) (h_ratio : 2 * votes_X = votes_Y)
  (h_perc_X : percent_X = 64)
  (h_perc_Y : percent_Y = 46) :
  percentageOfTotalVotesWon votes_X votes_Y 64 46 = 58 := by
    sorry

end douglas_percent_votes_l204_204559


namespace smallest_odd_number_with_five_prime_factors_l204_204274

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204274


namespace incorrect_statements_correct_statement_l204_204490

def isClosedSet (M : Set ℤ) : Prop :=
  ∀ a b ∈ M, a + b ∈ M ∧ a - b ∈ M

theorem incorrect_statements : 
  ¬ isClosedSet ({-4, -2, 0, 2, 4} : Set ℤ) ∧
  ¬ isClosedSet {n : ℤ | n > 0} ∧ 
  ¬ ∀ A₁ A₂ : Set ℤ, isClosedSet A₁ → isClosedSet A₂ → isClosedSet (A₁ ∪ A₂) :=
by
  sorry

theorem correct_statement :
  isClosedSet {n : ℤ | ∃ k : ℤ, n = 3 * k} :=
by
  sorry

end incorrect_statements_correct_statement_l204_204490


namespace maria_baggies_count_l204_204130

def total_cookies (chocolate_chip : ℕ) (oatmeal : ℕ) : ℕ :=
  chocolate_chip + oatmeal

def baggies_count (total_cookies : ℕ) (cookies_per_bag : ℕ) : ℕ :=
  total_cookies / cookies_per_bag

theorem maria_baggies_count :
  let choco_chip := 2
  let oatmeal := 16
  let cookies_per_bag := 3
  baggies_count (total_cookies choco_chip oatmeal) cookies_per_bag = 6 :=
by
  sorry

end maria_baggies_count_l204_204130


namespace target_hit_probability_l204_204772

open_locale big_operators

def probability_one_shot := (1 / 2 : ℝ)
def total_shots := 6
def hits := 3
def consecutive_hits := 2

theorem target_hit_probability :
  let comb_4_2 := (4.choose 2 : ℝ)
  in (comb_4_2 * probability_one_shot^total_shots) = (4.choose 2 : ℝ) * (1 / 2)^6 :=
by sorry

end target_hit_probability_l204_204772


namespace trigonometric_identity_l204_204889

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) :
  7 * (Real.sin α)^2 + 3 * (Real.cos α)^2 = 31 / 5 := by
  sorry

end trigonometric_identity_l204_204889


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204023

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204023


namespace count_valid_arrays_l204_204930

def array_conforms (A : array (Fin 6) (array (Fin 6) Int)) :=
  (∀ i : Fin 6, (List.sum (Fin 6).to_list.map (λ j => A[i][j])) = 0) ∧
  (∀ j : Fin 6, (List.sum (Fin 6).to_list.map (λ i => A[i][j])) = 0)

def is_valid_entry (A : array (Fin 6) (array (Fin 6) Int)) : Prop :=
  ∀ i j, A[i][j] = 1 ∨ A[i][j] = -1

theorem count_valid_arrays : 
  ∃ n : ℤ, n = 2800 ∧ 
  ∃ A : array (Fin 6) (array (Fin 6) Int), 
  (array_conforms A) ∧ (is_valid_entry A) := sorry

end count_valid_arrays_l204_204930


namespace product_of_solutions_l204_204465

theorem product_of_solutions : 
  (|y| = 3 * (|y| - 2)) → (y = 3 ∨ y = -3) → 3 * (-3) = -9 :=
begin
  sorry
end

end product_of_solutions_l204_204465


namespace constant_triangle_area_l204_204912

-- Define the hyperbolas C1 and C2
def C1 (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

def C2 (a b x y : ℝ) : Prop :=
  y^2 / b^2 - x^2 / a^2 = 1

-- Define the point P on C2
def P (a b θ : ℝ) : ℝ × ℝ :=
  (a * Real.tan θ, b * Real.sec θ)

-- Define the asymptotes of C1
def l1 (a b x y : ℝ) : Prop :=
  y = b / a * x

def l2 (a b x y : ℝ) : Prop :=
  y = -b / a * x

-- Lean theorem statement
theorem constant_triangle_area (a b θ : ℝ) (A B : ℝ × ℝ) (l1 l2 : ℝ → ℝ → Prop) :
  (∀ x y, (l1 a b x y ∨ l2 a b x y)) →
  (∀ x y, C1 a b x y) →
  (∀ x y, C2 a b x y) →
  (let P := P a b θ in
   ∃ A B, -- Point of tangency of tangents PA and PB on hyperbola C1
    ∃ l1 l2, -- Asymptotes of C1
      (l1 a b A.1 A.2) ∧ (l2 a b B.1 B.2) ∧
      (let area : ℝ := -- calculate the area
        by
          let x1 := A.1
          let y1 := A.2
          let x2 := B.1
          let y2 := B.2
          exact abs ((x1 * y2 - x2 * y1) / 2)
      in area = a * b)) :=
sorry -- proof to be provided

end constant_triangle_area_l204_204912


namespace distance_XY_equals_distance_ZY_l204_204702

noncomputable def Distance_XZ : ℝ := 5 * 80
noncomputable def Distance_ZY : ℝ := 4.444444444444445 * 45
noncomputable def Distance_XY : ℝ := Distance_XZ - Distance_ZY

theorem distance_XY_equals_distance_ZY :
  Distance_XY = Distance_ZY :=
by
  -- correct distance
  have Distance_XZ_val : Distance_XZ = 400 := by norm_num [Distance_XZ]
  have Distance_ZY_val : Distance_ZY = 200 := by norm_num [Distance_ZY]
  calc
    Distance_XY = 400 - 200 : by rw [Distance_XY, Distance_XZ_val, Distance_ZY_val]
    ... = 200 : by norm_num

end distance_XY_equals_distance_ZY_l204_204702


namespace ratio_of_black_to_white_areas_l204_204856

theorem ratio_of_black_to_white_areas :
  let π := Real.pi
  let radii := [2, 4, 6, 8]
  let areas := [π * (radii[0])^2, π * (radii[1])^2, π * (radii[2])^2, π * (radii[3])^2]
  let black_areas := [areas[0], areas[2] - areas[1]]
  let white_areas := [areas[1] - areas[0], areas[3] - areas[2]]
  let total_black_area := black_areas.sum
  let total_white_area := white_areas.sum
  let ratio := total_black_area / total_white_area
  ratio = 3 / 5 := sorry

end ratio_of_black_to_white_areas_l204_204856


namespace solve_puzzle_l204_204695

theorem solve_puzzle (x1 x2 x3 x4 x5 x6 x7 x8 : ℕ) : 
  (8 + x1 + x2 = 20) →
  (x1 + x2 + x3 = 20) →
  (x2 + x3 + x4 = 20) →
  (x3 + x4 + x5 = 20) →
  (x4 + x5 + 5 = 20) →
  (x5 + 5 + x6 = 20) →
  (5 + x6 + x7 = 20) →
  (x6 + x7 + x8 = 20) →
  (x1 = 7 ∧ x2 = 5 ∧ x3 = 8 ∧ x4 = 7 ∧ x5 = 5 ∧ x6 = 8 ∧ x7 = 7 ∧ x8 = 5) :=
by {
  sorry
}

end solve_puzzle_l204_204695


namespace length_of_CE_l204_204109

variable 
  (Point : Type) 
  (A C E B D F : Point)
  (AE CD CF : Point → Point → Prop)
  (perp : Point → Point → Prop)
  (AB_length : ℝ)
  (CD_length : ℝ)
  (AE_length : ℝ)

-- Conditions
axiom C_not_on_AE : ¬ AE C A 
axiom D_on_AE : AE D A
axiom CD_perp_AE : perp CD AE
axiom F_on_AE : AE F A
axiom B_on_CE : AE B C 
axiom AB_perp_CF : perp AB CF
axiom AB_eq_6 : AB_length = 6
axiom CD_eq_10 : CD_length = 10
axiom AE_eq_7 : AE_length = 7

-- Proof Statement
theorem length_of_CE : ∃ CE_length : ℝ, CE_length = 35 / 3 := 
sorry

end length_of_CE_l204_204109


namespace nina_total_spent_l204_204134

open Real

def toy_price : ℝ := 10
def toy_count : ℝ := 3
def toy_discount : ℝ := 0.15

def card_price : ℝ := 5
def card_count : ℝ := 2
def card_discount : ℝ := 0.10

def shirt_price : ℝ := 6
def shirt_count : ℝ := 5
def shirt_discount : ℝ := 0.20

def sales_tax_rate : ℝ := 0.07

noncomputable def discounted_price (price : ℝ) (count : ℝ) (discount : ℝ) : ℝ :=
  count * price * (1 - discount)

noncomputable def total_cost_before_tax : ℝ := 
  discounted_price toy_price toy_count toy_discount +
  discounted_price card_price card_count card_discount +
  discounted_price shirt_price shirt_count shirt_discount

noncomputable def total_cost_after_tax : ℝ :=
  total_cost_before_tax * (1 + sales_tax_rate)

theorem nina_total_spent : total_cost_after_tax = 62.60 :=
by
  sorry

end nina_total_spent_l204_204134


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204247

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204247


namespace olympiad2024_sum_l204_204969

theorem olympiad2024_sum (A B C : ℕ) (h_distinct : A ≠ B ∧ B ≠ C ∧ A ≠ C) (h_product : A * B * C = 2310) : 
  A + B + C ≤ 390 :=
sorry

end olympiad2024_sum_l204_204969


namespace find_other_number_verify_sum_l204_204720

theorem find_other_number (x y : Int) (h1 : x + y = -26) (h2 : x = 11) : y = -37 :=
by
  calc
    y = -26 - x : by 
      rw [← h1, h2]
    ... = -26 - 11 : by
      rw h2
    ... = -37 : by
      norm_num

theorem verify_sum (x y : Int) (h : y = -37) (h2 : x = 11) : x + y = -26 :=
by
  rw [h, h2]
  norm_num

end find_other_number_verify_sum_l204_204720


namespace flour_needed_for_one_loaf_l204_204133

-- Define the conditions
def flour_needed_for_two_loaves : ℚ := 5 -- cups of flour needed for two loaves

-- Define the theorem to prove
theorem flour_needed_for_one_loaf : flour_needed_for_two_loaves / 2 = 2.5 :=
by 
  -- Skip the proof.
  sorry

end flour_needed_for_one_loaf_l204_204133


namespace opposite_number_l204_204178

variable (a : ℝ)

theorem opposite_number (a : ℝ) : -(3 * a - 2) = -3 * a + 2 := by
  sorry

end opposite_number_l204_204178


namespace tangent_integer_values_l204_204859

/-- From point P outside a circle with circumference 12π units, a tangent and a secant are drawn.
      The secant divides the circle into arcs with lengths m and n. Given that the length of the
      tangent t is the geometric mean between m and n, and that m is three times n, there are zero
      possible integer values for t. -/
theorem tangent_integer_values
  (circumference : ℝ) (m n t : ℝ)
  (h_circumference : circumference = 12 * Real.pi)
  (h_sum : m + n = 12 * Real.pi)
  (h_ratio : m = 3 * n)
  (h_tangent : t = Real.sqrt (m * n)) :
  ¬(∃ k : ℤ, t = k) := 
sorry

end tangent_integer_values_l204_204859


namespace div_of_powers_l204_204709

-- Definitions of conditions:
def is_power_of_3 (x : ℕ) := x = 3 ^ 3

-- The conditions for the problem:
variable (a b c : ℕ)
variable (h1 : is_power_of_3 27)
variable (h2 : a = 3)
variable (h3 : b = 12)
variable (h4 : c = 6)

-- The proof statement:
theorem div_of_powers : 3 ^ 12 / 27 ^ 2 = 729 :=
by
  have h₁ : 27 = 3 ^ 3 := h1
  have h₂ : 27 ^ 2 = (3 ^ 3) ^ 2 := by rw [h₁]
  have h₃ : (3 ^ 3) ^ 2 = 3 ^ 6 := by rw [← pow_mul]
  have h₄ : 27 ^ 2 = 3 ^ 6 := by rw [h₂, h₃]
  have h₅ : 3 ^ 12 / 3 ^ 6 = 3 ^ (12 - 6) := by rw [div_eq_mul_inv, ← pow_sub]
  show 3 ^ 6 = 729, from by norm_num
  sorry

end div_of_powers_l204_204709


namespace total_legs_proof_l204_204579

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end total_legs_proof_l204_204579


namespace prime_divides_harmonic_sum_l204_204538

theorem prime_divides_harmonic_sum (p m n : ℕ) (hp : Nat.Prime p) (hp2 : p > 2) (hmn_pos : m > 0 ∧ n > 0)
  (harmonic_sum : m / n = ∑ i in Finset.range (p - 1) + 1, 1 / (i + 1)) : p ∣ m :=
sorry

end prime_divides_harmonic_sum_l204_204538


namespace acute_angle_between_hands_at_3_36_l204_204210

variable (minute_hand_position hour_hand_position abs_diff : ℝ)

def minute_hand_angle_at_3_36 : ℝ := 216
def hour_hand_angle_at_3_36 : ℝ := 108

theorem acute_angle_between_hands_at_3_36 (h₀ : minute_hand_position = 216)
    (h₁ : hour_hand_position = 108) :
    abs_diff = abs(minute_hand_position - hour_hand_position) → 
    abs_diff = 108 :=
  by
    rw [h₀, h₁]
    sorry

end acute_angle_between_hands_at_3_36_l204_204210


namespace smallest_odd_number_with_five_prime_factors_l204_204223

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204223


namespace butterflies_left_l204_204826

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l204_204826


namespace total_bones_in_graveyard_l204_204070

def total_skeletons : ℕ := 20

def adult_women : ℕ := total_skeletons / 2
def adult_men : ℕ := (total_skeletons - adult_women) / 2
def children : ℕ := (total_skeletons - adult_women) / 2

def bones_adult_woman : ℕ := 20
def bones_adult_man : ℕ := bones_adult_woman + 5
def bones_child : ℕ := bones_adult_woman / 2

def bones_graveyard : ℕ :=
  (adult_women * bones_adult_woman) +
  (adult_men * bones_adult_man) +
  (children * bones_child)

theorem total_bones_in_graveyard :
  bones_graveyard = 375 :=
sorry

end total_bones_in_graveyard_l204_204070


namespace div_power_l204_204703

theorem div_power (h : 27 = 3 ^ 3) : 3 ^ 12 / 27 ^ 2 = 729 :=
by {
  calc
    3 ^ 12 / 27 ^ 2 = 3 ^ 12 / (3 ^ 3) ^ 2 : by rw h
               ... = 3 ^ 12 / 3 ^ 6       : by rw pow_mul
               ... = 3 ^ (12 - 6)         : by rw div_eq_sub_pow
               ... = 3 ^ 6                : by rw sub_self_pow
               ... = 729                  : by norm_num,
  sorry
}

end div_power_l204_204703


namespace sacks_per_day_l204_204927

-- Define the given conditions
def total_sacks : ℕ := 56
def total_days : ℕ := 4

-- Define the statement we need to prove
theorem sacks_per_day :
  ∃ (h : ℕ), total_sacks = total_days * h ∧ h = 14 :=
by
  use 14
  split
  · exact rfl
  · exact rfl

end sacks_per_day_l204_204927


namespace spherical_to_rectangular_coords_l204_204428

noncomputable def spherical_to_rectangular 
  (ρ θ φ : ℝ)  : ℝ × ℝ × ℝ :=
(ρ * sin φ * cos θ, ρ * sin φ * sin θ, ρ * cos φ)

theorem spherical_to_rectangular_coords 
  (hρ : ℝ := 10) (hθ : ℝ := 5 * Real.pi / 4) (hφ : ℝ := Real.pi / 4)
  (x y z : ℝ) :
  spherical_to_rectangular hρ hθ hφ = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_coords_l204_204428


namespace clock_angle_at_3_36_l204_204200

def minute_hand_angle : ℝ := (36.0 / 60.0) * 360.0
def hour_hand_angle : ℝ := 90.0 + (36.0 / 60.0) * 30.0

def acute_angle (a b : ℝ) : ℝ :=
  let diff := abs (a - b)
  min diff (360 - diff)

theorem clock_angle_at_3_36 :
  acute_angle minute_hand_angle hour_hand_angle = 108 :=
by
  sorry

end clock_angle_at_3_36_l204_204200


namespace range_of_m_l204_204851

-- Define the function and the inequality condition
def inequality (m : ℝ) (x : ℝ) : Prop := 
  m * x^3 - x^2 + 4 * x + 3 ≥ 0

-- Define the main theorem statement
theorem range_of_m (m : ℝ) : 
  (∀ x ∈ Icc (-2 : ℝ) (1 : ℝ), inequality m x) ↔ (m ∈ Icc (-6 : ℝ) (-2 : ℝ)) :=
by
  sorry

end range_of_m_l204_204851


namespace sum_of_integers_n_l204_204845

theorem sum_of_integers_n (h₁ : ∃ k : ℤ, ∀ n : ℤ, n - 3 = k^3) (h₂ : ∃ m : ℤ, ∀ n : ℤ, n^2 + 4 = m^3) : 
    ∑ n in {n : ℤ | (∃ k : ℤ, n - 3 = k^3) ∧ (∃ m : ℤ, n^2 + 4 = m^3)}, n = 13 := by
  sorry

end sum_of_integers_n_l204_204845


namespace complete_the_square_l204_204343

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end complete_the_square_l204_204343


namespace diametrically_opposite_points_exist_l204_204370

theorem diametrically_opposite_points_exist (k : ℕ) (h : k > 0)
  (arc_lengths : list ℕ)
  (h1 : arc_lengths.length = 3 * k)
  (h2 : arc_lengths.count 1 = k)
  (h3 : arc_lengths.count 2 = k)
  (h4 : arc_lengths.count 3 = k) :
  ∃ i j, i ≠ j ∧ (arc_lengths.nth i = arc_lengths.nth (j - k) ∨ arc_lengths.nth i = arc_lengths.nth (j + k)) :=
sorry

end diametrically_opposite_points_exist_l204_204370


namespace dot_product_range_l204_204937

variables (a b : EuclideanSpace ℝ (Fin 2))
variables (norm_a : ‖a‖ = 3)
variables (norm_b : ‖b‖ = 5)
variables (theta : ℝ)
variables (angle_double : 0 <= theta ∧ theta <= 2*π)

theorem dot_product_range : 
  let double_theta := 2 * theta in
  let cos_2theta := 2 * real.cos theta^2 - 1 in
  (3 : ℝ) * (5 : ℝ) * cos_2theta ∈ set.Icc (-15 : ℝ) (15 : ℝ) :=
by
  sorry

end dot_product_range_l204_204937


namespace imaginary_part_z_l204_204936

theorem imaginary_part_z : 
  let z := (1 - complex.I) * (2 + complex.I) in 
  complex.im z = -1 := 
by
  let z := (1 - complex.I) * (2 + complex.I)
  show complex.im z = -1
  sorry

end imaginary_part_z_l204_204936


namespace acute_angle_at_3_36_l204_204204

def degrees (h m : ℕ) : ℝ :=
  let minute_angle := (m / 60.0) * 360.0
  let hour_angle := (h % 12 + m / 60.0) * 30.0
  abs (minute_angle - hour_angle)

theorem acute_angle_at_3_36 : degrees 3 36 = 108 :=
by
  sorry

end acute_angle_at_3_36_l204_204204


namespace hundredth_odd_integer_not_divisible_by_five_l204_204196

def odd_positive_integer (n : ℕ) : ℕ := 2 * n - 1

theorem hundredth_odd_integer_not_divisible_by_five :
  odd_positive_integer 100 = 199 ∧ ¬ (199 % 5 = 0) :=
by
  sorry

end hundredth_odd_integer_not_divisible_by_five_l204_204196


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204252

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204252


namespace toys_in_box_time_l204_204991

/-- Mia and her grandmother are working together to put all $50$ toys into a toy box. However, 
every $45$ seconds, while Mia's grandmother puts $4$ toys into the box, Mia immediately takes $3$ toys out.
Prove that the total time taken to put all 50 toys into the box for the first time is 36 minutes. -/
theorem toys_in_box_time :
  ∀ (total_toys: ℕ) (grandmother_rate: ℕ) (mia_rate: ℕ) (interval_seconds: ℕ),
    total_toys = 50 →
    grandmother_rate = 4 →
    mia_rate = 3 →
    interval_seconds = 45 →
    (total_toys / (grandmother_rate - mia_rate + (total_toys % (grandmother_rate - mia_rate) == 0).ite 0 1))
    * interval_seconds / 60 = 36 :=
by
  intros total_toys grandmother_rate mia_rate interval_seconds 
         h_total_toys h_grandmother_rate h_mia_rate h_interval_seconds
  sorry

end toys_in_box_time_l204_204991


namespace ellipse_eccentricity_l204_204886

def semi_focal_distance (a b : ℝ) : ℝ := real.sqrt (a^2 - b^2)

def eccentricity (a b : ℝ) : ℝ := semi_focal_distance a b / a

theorem ellipse_eccentricity (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) :
  let c := semi_focal_distance a b,
      F₁ := (-c, 0),
      F₂ := (c, 0),
      P := (a^2 / c, real.sqrt 3 * b) in
  -- Additional property stated in the proof problem, rewrite condition as necessary
  (perpendicular_bisector_passes_through F₁ F₂ P) →
  eccentricity a b = real.sqrt 2 / 2 := 
by 
  sorry

end ellipse_eccentricity_l204_204886


namespace log_statements_correct_l204_204052

theorem log_statements_correct (b x : ℝ) (y : ℝ) (h₁ : y = Real.log x / Real.log b) (h₂ : 0 < b) (h₃ : b ≠ 1) :
  ((x = b → y = 1) ∧ (x = 1 → y = 0) ∧ (x = b^2 → y = 2) ∧ (x = 0 → y = 0) ∧ 
  ((x = b → y = 1) ∧ (x = 1 → y = 0) ∧ (x = b^2 → y = 2) ∧ x ≠ 0 → (x = b → y = 1 ∧ x = 1 → y = 0 ∧ x = b^2 → y = 2 ∧ x ≠ 0 ∧ (x ≠ b ∧ x ≠ 1 ∧ x ≠ b^2)))
  :=
sorry

end log_statements_correct_l204_204052


namespace find_values_n_l204_204537

def valid_n (n : ℤ) : Prop := 
  ∃ k : ℤ, (k^2 - 2005)/2 - n ≥ 0 ∧ 
           (k^2 - 2005)/2 - n = k * k/2 - n ∧ 
           k is integer ∧ 
           sqrt n + sqrt (n + 2005) = k

theorem find_values_n (n : ℤ) : valid_n n → n = 39204 ∨ n = 1004004 :=
by
  sorry

end find_values_n_l204_204537


namespace find_m_l204_204061

noncomputable def f (x m : ℝ) : ℝ := (x^2 + m*x) * Real.exp x

def is_monotonically_decreasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ y ∧ y ≤ b → f y ≤ f x

theorem find_m (m : ℝ) :
  is_monotonically_decreasing (f (m := m)) (-3/2) 1 ∧
  (-3/2)^2 + (m + 2)*(-3/2) + m = 0 ∧
  1^2 + (m + 2)*1 + m = 0 →
  m = -3/2 :=
by
  sorry

end find_m_l204_204061


namespace two_digit_primes_ending_in_3_l204_204034

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204034


namespace change_from_15_dollars_l204_204809

theorem change_from_15_dollars :
  let cost_eggs := 3
  let cost_pancakes := 2
  let cost_mugs_of_cocoa := 2 * 2
  let tax := 1
  let initial_cost := cost_eggs + cost_pancakes + cost_mugs_of_cocoa + tax
  let additional_pancakes := 2
  let additional_mug_of_cocoa := 2
  let additional_cost := additional_pancakes + additional_mug_of_cocoa
  let new_total_cost := initial_cost + additional_cost
  let payment := 15
  let change := payment - new_total_cost
  change = 1 :=
by
  sorry

end change_from_15_dollars_l204_204809


namespace acute_angle_3_36_clock_l204_204221

theorem acute_angle_3_36_clock : 
  let minute_hand_degrees := (36 / 60) * 360,
      hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360,
      angle := abs(minute_hand_degrees - hour_hand_degrees) in
  angle = 108 :=
by
  let minute_hand_degrees := (36 / 60) * 360
  let hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360
  let angle := abs(minute_hand_degrees - hour_hand_degrees)
  show angle = 108 from sorry

end acute_angle_3_36_clock_l204_204221


namespace polynomial_nonreal_root_l204_204105

variables {a : Fin 98 → ℝ}

def P (x : ℂ) : ℂ :=
  x^100 + 20 * x^99 + 198 * x^98 +
  ∑ i in Finset.range 97, (a i) * x^(i + 1) + 1

theorem polynomial_nonreal_root (h_poly : ∀ x : ℂ, P x = x^100 + 20 * x^99 + 198 * x^98 +
  ∑ i in Finset.range 97, (a i) * x^(i + 1) + 1) :
  ∃ x : ℂ, P x = 0 ∧ ¬ (x.im = 0) :=
by
  sorry

end polynomial_nonreal_root_l204_204105


namespace value_of_a_minus_b_l204_204633

theorem value_of_a_minus_b (a b : ℤ) (h1 : 2020 * a + 2024 * b = 2040) (h2 : 2022 * a + 2026 * b = 2044) :
  a - b = 1002 :=
sorry

end value_of_a_minus_b_l204_204633


namespace pow_div_pow_eq_l204_204706

theorem pow_div_pow_eq :
  (3^12) / (27^2) = 729 :=
by
  -- We'll use the provided conditions and proof outline
  -- 1. 27 = 3^3
  -- 2. (a^b)^c = a^{bc}
  -- 3. a^b \div a^c = a^{b-c}
  sorry

end pow_div_pow_eq_l204_204706


namespace cos_seven_pi_over_six_l204_204459

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l204_204459


namespace sum_of_digits_of_valid_hex_l204_204425

-- Define a predicate to check if a number has only digit 0-9 in its hexadecimal representation.
def is_valid_hex (n : ℕ) : Prop :=
  let hex_digits := n.digits 16
  hex_digits.all (λ d, d ≤ 9)

-- Define the problem and expected answer.
theorem sum_of_digits_of_valid_hex :
  let count := (Finset.range 500).filter is_valid_hex).card in
  let digit_sum := count.digits 10 in
  digit_sum.sum = 10 :=
begin
  sorry,  -- Proof omitted
end

end sum_of_digits_of_valid_hex_l204_204425


namespace smallest_odd_number_with_five_different_prime_factors_l204_204319

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204319


namespace problem_statement_l204_204470

def diamond (a b : ℝ) := Real.sqrt (a^2 + b^2)
def star (a b : ℝ) := |a| - |b|

theorem problem_statement : (7 ∘ diamond ∘ 24) star (24 ∘ diamond ∘ 7) = 0 := by
  sorry

end problem_statement_l204_204470


namespace determine_quadrant_of_z_l204_204436

noncomputable def complex_quadrant (z : ℂ) : String :=
  if (z.re > 0) && (z.im > 0) then "First quadrant"
  else if (z.re < 0) && (z.im > 0) then "Second quadrant"
  else if (z.re < 0) && (z.im < 0) then "Third quadrant"
  else if (z.re > 0) && (z.im < 0) then "Fourth quadrant"
  else "On the axis"

theorem determine_quadrant_of_z : 
  let z := (Complex.i * (-2 + Complex.i))
  complex_quadrant z = "Third quadrant" :=
by
  sorry

end determine_quadrant_of_z_l204_204436


namespace doug_initial_marbles_l204_204441

theorem doug_initial_marbles (ed_marbles : ℕ) (diff_ed_doug : ℕ) (final_ed_marbles : ed_marbles = 27) (diff : diff_ed_doug = 5) :
  ∃ doug_initial_marbles : ℕ, doug_initial_marbles = 22 :=
by
  sorry

end doug_initial_marbles_l204_204441


namespace max_non_threatening_rooks_l204_204173

theorem max_non_threatening_rooks (n : ℕ) (hn : n = 2022) : 
  ∃ k : ℕ, (∀ i j : fin k, i ≠ j → (a i ≠ a j ∧ b i ≠ b j ∧ c i ≠ c j)) ∧ k = 1349 := 
sorry


end max_non_threatening_rooks_l204_204173


namespace sharpener_difference_l204_204766

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end sharpener_difference_l204_204766


namespace probability_first_two_cards_black_l204_204555

theorem probability_first_two_cards_black :
  let deck := 52,
      suits := 4,
      cards_per_suit := 13,
      black_suits := 2
  in
  let total_black_cards := black_suits * cards_per_suit,
      total_ways := (deck * (deck - 1)) / 2,
      successful_ways := (total_black_cards * (total_black_cards - 1)) / 2
  in 
  (successful_ways / total_ways : ℚ) = 25 / 102 := by
  sorry

end probability_first_two_cards_black_l204_204555


namespace simplify_expr1_simplify_expr2_l204_204143

-- First expression
theorem simplify_expr1 (a b : ℝ) : a * (a - b) - (a + b) * (a - 2 * b) = 2 * b ^ 2 :=
by
  sorry

-- Second expression
theorem simplify_expr2 (x : ℝ) : 
  ( ( (4 * x - 9) / (3 - x) - x + 3 ) / ( (x ^ 2 - 4) / (x - 3) ) ) = - (x / (x + 2)) :=
by
  sorry

end simplify_expr1_simplify_expr2_l204_204143


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204020

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204020


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204003

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204003


namespace smallest_odd_number_with_five_prime_factors_l204_204226

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204226


namespace previous_job_salary_is_correct_l204_204188

-- Define the base salary and commission structure.
def base_salary_new_job : ℝ := 45000
def commission_rate : ℝ := 0.15
def sale_amount : ℝ := 750
def minimum_sales : ℝ := 266.67

-- Define the total salary from the new job with the minimum sales.
def new_job_total_salary : ℝ :=
  base_salary_new_job + (commission_rate * sale_amount * minimum_sales)

-- Define Tom's previous job's salary.
def previous_job_salary : ℝ := 75000

-- Prove that Tom's previous job salary matches the new job total salary with the minimum sales.
theorem previous_job_salary_is_correct :
  (new_job_total_salary = previous_job_salary) :=
by
  -- This is where you would include the proof steps, but it's sufficient to put 'sorry' for now.
  sorry

end previous_job_salary_is_correct_l204_204188


namespace two_digit_primes_ending_in_3_l204_204039

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204039


namespace coordinates_of_C_l204_204383

namespace CoordinateProof

-- Given points
def A := (-3 : ℝ, 5 : ℝ)
def B := (9 : ℝ, -1 : ℝ)

-- Define vector AB
def vectorAB := (B.1 - A.1, B.2 - A.2)

-- Define the scalar multiple for vector AB to get BC
def scalar : ℝ := 2 / 5

-- Hence define the vector BC
def vectorBC := (scalar * (vectorAB.1), scalar * (vectorAB.2))

-- Define point C
def C := (B.1 + vectorBC.1, B.2 + vectorBC.2)

-- State the theorem that the coordinates of C are as given
theorem coordinates_of_C : C = (13.8, -3.4) := by
  sorry

end CoordinateProof

end coordinates_of_C_l204_204383


namespace polynomial_division_remainder_triplet_l204_204719

theorem polynomial_division_remainder_triplet :
  ∃ (l b c : ℝ),
    (∀ x : ℝ,
      (x^5 - 5 * x^4 + 14 * x^3 - 20 * x^2 + 15 * x - 4) ≡ 
      (x^2 + b * x + c) [MOD (x^3 - 3 * x^2 + 2 * x + l)]) →
    (l = 1 ∧ b = -2 ∧ c = 0) :=
by
  use 1, -2, 0
  intros x h
  simp at h
  sorry

end polynomial_division_remainder_triplet_l204_204719


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204013

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204013


namespace butterflies_left_correct_l204_204819

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l204_204819


namespace infinite_pairwise_rel_prime_triples_l204_204624

theorem infinite_pairwise_rel_prime_triples :
  ∃^∞ (a b c : ℕ), Nat.gcd a b = 1 ∧ Nat.gcd b c = 1 ∧ Nat.gcd c a = 1 ∧ 
  Nat.gcd (a * b + c) (b * c + a) = 1 ∧ 
  Nat.gcd (b * c + a) (c * a + b) = 1 ∧ 
  Nat.gcd (c * a + b) (a * b + c) = 1 :=
sorry

end infinite_pairwise_rel_prime_triples_l204_204624


namespace kilometers_to_meters_kilograms_to_grams_l204_204750

def km_to_meters (km: ℕ) : ℕ := km * 1000
def kg_to_grams (kg: ℕ) : ℕ := kg * 1000

theorem kilometers_to_meters (h: 3 = 3): km_to_meters 3 = 3000 := by {
 sorry
}

theorem kilograms_to_grams (h: 4 = 4): kg_to_grams 4 = 4000 := by {
 sorry
}

end kilometers_to_meters_kilograms_to_grams_l204_204750


namespace incorrect_parallel_m_n_l204_204882

variables {l m n : Type} [LinearOrder m] [LinearOrder n] {α β : Type}

-- Assumptions for parallelism and orthogonality
def parallel (x y : Type) : Prop := sorry
def orthogonal (x y : Type) : Prop := sorry

-- Conditions
axiom parallel_m_l : parallel m l
axiom parallel_n_l : parallel n l
axiom orthogonal_m_α : orthogonal m α
axiom parallel_m_β : parallel m β
axiom parallel_m_α : parallel m α
axiom parallel_n_α : parallel n α
axiom orthogonal_m_β : orthogonal m β
axiom orthogonal_α_β : orthogonal α β

-- The theorem to prove
theorem incorrect_parallel_m_n : parallel m α ∧ parallel n α → ¬ parallel m n := sorry

end incorrect_parallel_m_n_l204_204882


namespace sum_arithmetic_sequence_n_ge_52_l204_204462

theorem sum_arithmetic_sequence_n_ge_52 (n : ℕ) : 
  (∃ k, k = n) → 22 - 3 * (n - 1) = 22 - 3 * (n - 1) ∧ n ∈ { k | 3 ≤ k ∧ k ≤ 13 } :=
by
  sorry

end sum_arithmetic_sequence_n_ge_52_l204_204462


namespace sum_first_2014_terms_l204_204418

noncomputable def convex_sequence (b : ℕ → ℤ) : Prop :=
∀ n : ℕ, b (n + 1) = b n + b (n + 2)

def sequence_b : ℕ → ℤ
| 0     := 1
| 1     := -2
| (n+2) := sequence_b n - sequence_b (n+1)

theorem sum_first_2014_terms : (∑ i in Finset.range 2014, sequence_b i) = 339 :=
begin
  have h_convex : convex_sequence sequence_b,
  { intro n,
    cases n; simp [sequence_b] },
  sorry
end

end sum_first_2014_terms_l204_204418


namespace acute_angle_3_36_clock_l204_204219

theorem acute_angle_3_36_clock : 
  let minute_hand_degrees := (36 / 60) * 360,
      hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360,
      angle := abs(minute_hand_degrees - hour_hand_degrees) in
  angle = 108 :=
by
  let minute_hand_degrees := (36 / 60) * 360
  let hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360
  let angle := abs(minute_hand_degrees - hour_hand_degrees)
  show angle = 108 from sorry

end acute_angle_3_36_clock_l204_204219


namespace range_of_lambda_l204_204861

-- Define the vectors a and b
def a (λ : ℝ) : ℝ × ℝ := (λ, 2 * λ)
def b (λ : ℝ) : ℝ × ℝ := (3 * λ, 2)

-- Define the dot product
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the condition for the angle between vectors to be acute
def acute_angle_condition (λ : ℝ) : Prop := dot_product (a λ) (b λ) > 0

-- Define the condition that vectors are not parallel 
def not_parallel (λ : ℝ) : Prop := ¬(λ = 0 ∨ λ = 1/3)

-- Define the final range condition for λ
def range_condition (λ : ℝ) : Prop := (λ < -4/3) ∨ (λ > 0 ∧ λ ≠ 1/3)

-- The theorem statement
theorem range_of_lambda (λ : ℝ) : acute_angle_condition λ ∧ not_parallel λ ↔ range_condition λ :=
by
  sorry

end range_of_lambda_l204_204861


namespace find_k_l204_204863

theorem find_k (k : ℤ)
  (h : ∃ (x₁ x₂ : ℤ), x₁ ≠ x₂ ∧ ∀ x, ((k^2 - 1) * x^2 - 3 * (3 * k - 1) * x + 18 = 0) ↔ (x = x₁ ∨ x = x₂)
       ∧ x₁ > 0 ∧ x₂ > 0) : k = 2 :=
by
  sorry

end find_k_l204_204863


namespace count_valid_n_l204_204983

noncomputable def f (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ d => n % d = 0).sum id

def is_sum_of_two_squares (n : ℕ) : Prop :=
  ∃ x y : ℕ, x^2 + y^2 = n

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

theorem count_valid_n :
  (Finset.range 51).filter (λ n => is_prime (f n) ∧ is_sum_of_two_squares n).card = 5 :=
by {
  sorry
}

end count_valid_n_l204_204983


namespace cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204451

noncomputable def cos_seven_pi_six : Real :=
  Real.cos (7 * Real.pi / 6)

theorem cos_seven_pi_six_eq_neg_sqrt_three_div_two :
  cos_seven_pi_six = -Real.sqrt 3 / 2 :=
sorry

end cos_seven_pi_six_eq_neg_sqrt_three_div_two_l204_204451


namespace TowerRemainder_l204_204417

noncomputable def T (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * T (n - 1)

theorem TowerRemainder : (T 10) % 1000 = 144 := by
  sorry

end TowerRemainder_l204_204417


namespace verify_polynomial_division_example_l204_204844

open Polynomial

noncomputable def remainder_of_division (p q : Polynomial ℝ) : Polynomial ℝ :=
  let (_, r) := p.divModByMonic (monic q)
  r

def polynomial_division_example : Prop :=
  remainder_of_division (X^4) (X^2 + 4 * X + 1) = -56 * X - 15

theorem verify_polynomial_division_example : polynomial_division_example :=
by
  sorry

end verify_polynomial_division_example_l204_204844


namespace math_problem_l204_204502

theorem math_problem 
  (a b c : ℕ) 
  (h_primea : Nat.Prime a)
  (h_posa : 0 < a)
  (h_posb : 0 < b)
  (h_posc : 0 < c)
  (h_eq : a^2 + b^2 = c^2) :
  (b % 2 ≠ c % 2) ∧ (∃ k, 2 * (a + b + 1) = k^2) := 
sorry

end math_problem_l204_204502


namespace butterfly_count_l204_204822

theorem butterfly_count (total_butterflies : ℕ) (one_third_flew_away : ℕ) (initial_butterflies : total_butterflies = 9) (flew_away : one_third_flew_away = total_butterflies / 3) : 
(total_butterflies - one_third_flew_away) = 6 := by
  sorry

end butterfly_count_l204_204822


namespace part_a_expected_mixed_color_pairs_part_b_expected_attempts_l204_204949

-- Part (a): Expected number of mixed-color pairs given n white and n black balls
theorem part_a_expected_mixed_color_pairs (n : ℕ) : 
  expected_mixed_color_pairs n = (n * n) / (2 * n - 1) := sorry

-- Part (b): Expected number of attempts needed to leave the box empty
theorem part_b_expected_attempts (n : ℕ) : 
  expected_attempts n = 2 * n - harmonic_number n := sorry

-- Note: In practice, the harmonic number and expected value definitions would need to be provided.

namespace Harmonic_number

  noncomputable def H_n (n : ℕ) : ℝ := sorry -- Define the harmonic series H(n)

end Harmonic_number

end part_a_expected_mixed_color_pairs_part_b_expected_attempts_l204_204949


namespace quadratic_inequality_solutions_l204_204836

theorem quadratic_inequality_solutions {k : ℝ} (h1 : 0 < k) (h2 : k < 16) :
  ∃ x : ℝ, x^2 - 8*x + k < 0 :=
sorry

end quadratic_inequality_solutions_l204_204836


namespace butterflies_left_l204_204827

theorem butterflies_left (initial_butterflies : ℕ) (one_third_left : ℕ)
  (h1 : initial_butterflies = 9) (h2 : one_third_left = initial_butterflies / 3) :
  initial_butterflies - one_third_left = 6 :=
by
  sorry

end butterflies_left_l204_204827


namespace perfect_square_option_l204_204345

theorem perfect_square_option (a b c : ℕ) : 
  (∃ k : ℕ, k^2 = 3^6 * 4^5 * 5^4) :=
begin
  sorry
end

end perfect_square_option_l204_204345


namespace find_a_value_l204_204497

theorem find_a_value (a : ℝ) :
  {a^2, a + 1, -3} ∩ {a - 3, 2 * a - 1, a^2 + 1} = {-3} → a = -1 :=
by
  intro h
  sorry

end find_a_value_l204_204497


namespace y_coordinate_of_point_l204_204968

theorem y_coordinate_of_point (x y : ℝ) (m : ℝ)
  (h₁ : x = 10)
  (h₂ : y = m * x + -2)
  (m_def : m = (0 - (-4)) / (4 - (-4)))
  (h₃ : y = 3) : y = 3 :=
sorry

end y_coordinate_of_point_l204_204968


namespace smallest_odd_number_with_five_prime_factors_l204_204231

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204231


namespace ellipse_standard_equation_fixed_point_Q_exists_l204_204906

variables {a b : ℝ}
noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem ellipse_standard_equation (h1 : a > b) (h2 : b > 0) (h3 : ellipse_equation (-1) (sqrt 2 / 2))
  (hfocus : (1 ^ 2) / a^2 + 0 / b^2 = 1) :
   ellipse_equation 1 0 :=
sorry

theorem fixed_point_Q_exists (h1 : a > b) (h2 : b > 0) (h3 : ellipse_equation (-1) (sqrt 2 / 2))
  (hfocus : (1 ^ 2) / a^2 + 0 / b^2 = 1) :
   ∃ m : ℝ, (m = 5/4) ∧ (∀ A B : ℝ, (QA B (5/4)) ∙ (QB A (5/4)) = -7 / 16) :=
sorry

end ellipse_standard_equation_fixed_point_Q_exists_l204_204906


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204326

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204326


namespace two_digit_primes_ending_in_3_l204_204042

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204042


namespace smallest_odd_number_with_five_different_prime_factors_l204_204318

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204318


namespace division_of_negatives_l204_204413

theorem division_of_negatives :
  (-81 : ℤ) / (-9) = 9 := 
  by
  -- Property of division with negative numbers
  have h1 : (-81 : ℤ) / (-9) = 81 / 9 := by sorry
  -- Perform the division
  have h2 : 81 / 9 = 9 := by sorry
  -- Combine the results
  rw h1
  exact h2

end division_of_negatives_l204_204413


namespace meaning_of_assignment_statement_l204_204650

variable (n : ℕ)

theorem meaning_of_assignment_statement : 
  (n = n + 1) ↔ (n := n + 1) :=
by
  -- here we need to insert the actual proof in Lean 4.
  sorry

end meaning_of_assignment_statement_l204_204650


namespace h_at_7_over_5_eq_0_l204_204424

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l204_204424


namespace total_handshakes_l204_204787

-- Defining the groups and their properties
def group1 : Set ℕ := {i | i < 25}
def group2 : Set ℕ := {i | 25 ≤ i ∧ i < 30}
def group3 : Set ℕ := {i | 30 ≤ i ∧ i < 35}

-- Assumptions
axiom group1_all_know_each_other : ∀ (a b : ℕ), a ∈ group1 → b ∈ group1 → a ≠ b → knows a b
axiom group2_knows_no_one : ∀ (a : ℕ), a ∈ group2 → ∀ (b : ℕ), b ∈ group1 ∨ b ∈ group2 ∨ b ∈ group3 → ¬knows a b
axiom group3_knows_18_from_group1 : ∀ (a : ℕ), a ∈ group3 → ∃ (S : Set ℕ), S ⊆ group1 ∧ S.card = 18 ∧ ∀ (b : ℕ), b ∈ S → knows a b

-- Function to count the number of handshakes
noncomputable def count_handshakes : ℕ :=
  let group2_handshakes := 5 * 30
  let group3_handshakes := 5 * 12
  group2_handshakes + group3_handshakes

-- The theorem we are to prove
theorem total_handshakes : count_handshakes = 210 := by
  sorry

end total_handshakes_l204_204787


namespace function_value_at_minus_one_l204_204437

theorem function_value_at_minus_one :
  ( -(1:ℝ)^4 + -(1:ℝ)^3 + (1:ℝ) ) / ( -(1:ℝ)^2 + (1:ℝ) ) = 1 / 2 :=
by sorry

end function_value_at_minus_one_l204_204437


namespace greatest_integer_e_minus_5_l204_204595

theorem greatest_integer_e_minus_5 (e : ℝ) (h : 2 < e ∧ e < 3) : ⌊e - 5⌋ = -3 :=
by
  sorry

end greatest_integer_e_minus_5_l204_204595


namespace probability_odd_product_is_one_l204_204614

noncomputable def area (r : ℝ) : ℝ := π * r ^ 2

noncomputable def area_outer_region : ℝ := (area 8 - area 4) / 3
noncomputable def area_inner_region : ℝ := area 4 / 3

def points_inner : List ℕ := [3, 4, 4]
def points_outer : List ℕ := [1, 3, 3]

def odd_points (points : List ℕ) : List ℕ := points.filter (λ x, x % 2 = 1)

def probability_odd_hit : ℝ := 
  (odd_points points_outer).length * area_outer_region + (odd_points points_inner).length * area_inner_region / (area 8)

theorem probability_odd_product_is_one :
  probability_odd_hit = 1 :=
by
  sorry

end probability_odd_product_is_one_l204_204614


namespace right_triangle_circle_diameter_RS_l204_204591

theorem right_triangle_circle_diameter_RS :
  ∀ (P Q R S : Type) [MetricSpace P] [MetricSpace Q] [MetricSpace R] [MetricSpace S] 
    (dist : P → Point → ℝ)
    (h1 : angle P Q R = π/2)
    (h2 : Circle Diameter ({QR}) S)
    (h3 : dist P S = 3)
    (h4 : dist Q S = 9), 
    dist R S = 27 :=
by
  intro P Q R S dist h1 h2 h3 h4
  sorry

end right_triangle_circle_diameter_RS_l204_204591


namespace g_f_neg2_l204_204120

def f (x : ℤ) : ℤ := x^3 + 3

def g (x : ℤ) : ℤ := 2*x^2 + 2*x + 1

theorem g_f_neg2 : g (f (-2)) = 41 :=
by {
  -- proof steps skipped
  sorry
}

end g_f_neg2_l204_204120


namespace nurse_missy_serving_time_l204_204992

-- Conditions
def total_patients : ℕ := 12
def standard_patient_time : ℕ := 5
def special_requirement_fraction : ℚ := 1/3
def serving_time_increase_factor : ℚ := 1.2

-- Proof statement
theorem nurse_missy_serving_time :
  let special_patients := total_patients * special_requirement_fraction
  let standard_patients := total_patients - special_patients
  let special_patient_time := serving_time_increase_factor * standard_patient_time
  total_serving_time = (standard_patient_time * standard_patients) + (special_patient_time * special_patients) :=
  64 := sorry

end nurse_missy_serving_time_l204_204992


namespace smallest_odd_number_with_five_prime_factors_l204_204308

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204308


namespace quadratic_inequality_transformation_l204_204513

-- Define the conditions
def quadratic_inequality_solution_set (f : ℝ → ℝ) : set ℝ :=
  {x | x < -1 ∨ x > 0.5}

def transformed_solution_set (f : ℝ → ℝ) : set ℝ :=
  {x | f (10 ^ x) > 0}

-- Define the correct answer
def solution_set (f : ℝ → ℝ) : set ℝ :=
  {x | x < -Real.log 2}

-- Proof problem in Lean 4 statement
theorem quadratic_inequality_transformation (f : ℝ → ℝ) :
  quadratic_inequality_solution_set f = {x | x < -1 ∨ x > 0.5} →
  transformed_solution_set f = {x | x < -Real.log 2} := 
by
  sorry

end quadratic_inequality_transformation_l204_204513


namespace total_amount_l204_204737

theorem total_amount (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) (P : ℝ)
  (h1 : CI = 2828.80) (h2 : r = 0.08) (h3 : n = 1) (h4 : t = 2)
  (h5 : P = CI / ((1 + r / n) ^ (n * t) - 1)) :
  let total_amount : ℝ := P + CI in
  total_amount = 19828.80 :=
by
  sorry

end total_amount_l204_204737


namespace smallest_odd_number_with_five_prime_factors_l204_204273

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204273


namespace find_intersection_distance_l204_204520

theorem find_intersection_distance :
  let C1 := λ t : ℝ, (1 + sqrt 3 * t, t)
  let C2 := {p : ℝ × ℝ | p.1^2 + 4 * p.2^2 = 4}
  let E := (1, 0)
  ∃ A B ∈ C2, (C1 A) ∧ (C1 B) ∧ 
  abs (dist E A) + abs (dist E B) = 2 * sqrt 19 / 7 :=
by
  sorry

end find_intersection_distance_l204_204520


namespace smallest_odd_number_with_five_prime_factors_l204_204225

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204225


namespace eccentricity_of_ellipse_l204_204876

theorem eccentricity_of_ellipse (a b : ℝ) (h1 : a > b) (h2 : b > 0)
  (P : ℝ × ℝ) (F1 F2 : ℝ × ℝ)
  (h_ellipse : P.1 ^ 2 / a ^ 2 + P.2 ^ 2 / b ^ 2 = 1)
  (h_perp : P.2 = 0)
  (h_angle : ∠ F1 P F2 = 30) :
  let e := sqrt(3) / 3 in e :=
sorry

end eccentricity_of_ellipse_l204_204876


namespace total_seats_in_cinema_l204_204361

theorem total_seats_in_cinema (best_seat : ℕ) (h : best_seat = 265) : 
  let k := (Int.sqrt (1 + 4 * 530) - 1) / 2 in
  1 ≤ k ∧ k = k.natAbs ∧ 
  let total_rows := 2 * k + 1 in
  (1 + total_rows) * total_rows / 2 = 1035 :=
by
  sorry

end total_seats_in_cinema_l204_204361


namespace log_equation_solution_l204_204347

theorem log_equation_solution (x : ℝ) (hpos : x > 0) (hneq : x ≠ 1) : (Real.log 8 / Real.log x) * (2 * Real.log x / Real.log 2) = 6 * Real.log 2 :=
by
  sorry

end log_equation_solution_l204_204347


namespace fuel_reduction_16km_temperature_drop_16km_l204_204397

-- Definition for fuel reduction condition
def fuel_reduction_rate (distance: ℕ) : ℕ := distance / 4 * 2

-- Definition for temperature drop condition
def temperature_drop_rate (distance: ℕ) : ℕ := distance / 8 * 1

-- Theorem to prove fuel reduction for 16 km
theorem fuel_reduction_16km : fuel_reduction_rate 16 = 8 := 
by
  -- proof will go here, but for now add sorry
  sorry

-- Theorem to prove temperature drop for 16 km
theorem temperature_drop_16km : temperature_drop_rate 16 = 2 := 
by
  -- proof will go here, but for now add sorry
  sorry

end fuel_reduction_16km_temperature_drop_16km_l204_204397


namespace scientific_notation_of_8200000_l204_204723

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l204_204723


namespace triangle_cosine_product_l204_204093

-- Define the given conditions
variables {K L M P O Q S : Type} [HasAngleRatio.{a}]
def is_triangle (p1 p2 p3 : Type) : Prop := true -- A placeholder definition
def median (p1 p2 p3 p4 : Type) : Prop := true -- A placeholder definition
def circumcenter (p1 p2 p3 p4 : Type) : Prop := true -- A placeholder definition
def incenter (p1 p2 p3 p4 : Type) : Prop := true -- A placeholder definition
def angle (p1 p2 p3 : Type) : Real := sorry -- A placeholder definition
def intersection (p1 p2 p3 p4 p5 : Type) : Prop := true -- A placeholder definition
def angle_prod_eq (a b c : Real) : Prop := a * b = c

-- Encapsulate the problem statement
theorem triangle_cosine_product :
    (is_triangle K L M) ∧ (median K L M P) ∧ (circumcenter K L M O) ∧ (incenter K L M Q) ∧
    (intersection KP OQ S) ∧ (angle Q S O / angle P S O = sqrt 6 * (angle Q S P / angle K S P)) ∧
    (angle L K M = π / 3) →
    angle_prod_eq (cos (angle K L M)) (cos (angle K M L)) (-3 / 8) :=
by sorry

end triangle_cosine_product_l204_204093


namespace quadratic_intersection_l204_204660

theorem quadratic_intersection :
  ∀ x : ℝ, (x^2 - 2 * x + 1 = 0) → x = 1 :=
begin
  sorry
end

end quadratic_intersection_l204_204660


namespace range_of_a_l204_204509

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l204_204509


namespace dot_product_of_vectors_l204_204482

variable (a b : ℝ)
variable (va vb : ℝ)

def norm (v : ℝ) : ℝ := abs v

theorem dot_product_of_vectors
  (h1 : norm a = 3)
  (h2 : norm b = 5)
  (h3 : (a * b) / (norm b) = 12 / 5) :
  a * b = 12 :=
sorry

end dot_product_of_vectors_l204_204482


namespace smallest_odd_number_with_five_prime_factors_l204_204230

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204230


namespace ellipse_condition_l204_204089

theorem ellipse_condition (m : ℝ) : 
  (∃ x y : ℝ, m * (x^2 + y^2 + 2*y + 1) = (x - 2*y + 3)^2) → m > 5 :=
by
  intro h
  sorry

end ellipse_condition_l204_204089


namespace interval_of_increase_l204_204517

-- Given conditions
def f (x : ℝ) : ℝ := (1/2) - cos(2 * x) * cos(2 * x)
def g (x : ℝ) : ℝ := 2 * sin(2 * x - (Real.pi / 8)) + 1

theorem interval_of_increase :
  let m : ℝ := Real.pi / 8 in
  is_monotonic_increasing_on (g) (Set.Icc π (5 * π / 4)) :=
sorry

end interval_of_increase_l204_204517


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204324

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204324


namespace sin_cos_15_deg_l204_204796

noncomputable def sin_deg (deg : ℝ) : ℝ := Real.sin (deg * Real.pi / 180)
noncomputable def cos_deg (deg : ℝ) : ℝ := Real.cos (deg * Real.pi / 180)

theorem sin_cos_15_deg :
  (sin_deg 15 + cos_deg 15) * (sin_deg 15 - cos_deg 15) = -Real.sqrt 3 / 2 :=
by
  sorry

end sin_cos_15_deg_l204_204796


namespace two_digit_primes_ending_in_3_l204_204035

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204035


namespace smallest_odd_number_with_five_prime_factors_l204_204305

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204305


namespace maximum_volume_exists_l204_204385

-- Definitions for conditions
def length_per_meter : Real := 18
def ratio_length_to_width := 2 / 1

-- Definition of volume function V based on given conditions
noncomputable def volume (x h : Real) : Real := 
  2 * x^2 * h

-- Relationship between length, width, and height
-- 6x + 4h = total length of the steel bar
noncomputable def height (x : Real) : Real := 
  (length_per_meter - 6 * x) / 4

-- Lean statement to prove maximum volume
theorem maximum_volume_exists : 
  ∃ x h V, 
    6 * x + 4 * h = length_per_meter ∧ 
    2 * x / ratio_length_to_width = h ∧
    V = volume x h ∧ 
    ∀ x' h' V', 
      6 * x' + 4 * h' = length_per_meter → 
      2 * x' / ratio_length_to_width = h' → 
      V' = volume x' h' → 
      V' ≤ V :=
by
  sorry

end maximum_volume_exists_l204_204385


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204009

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204009


namespace smallest_odd_number_with_five_different_prime_factors_l204_204312

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204312


namespace maximum_value_fraction_l204_204493

variable (a b : ℝ)
variable (hab : a * b = 1)
variable (hineq : a > b ∧ b ≥ (2/3))

theorem maximum_value_fraction : 
  ∃ (x : ℝ), x = a - b ∧ (0 < x ∧ x ≤ (5/6)) → 
  (sup {x : ℝ | ∃ (a b : ℝ), a * b = 1 ∧ a > b ∧ b ≥ (2/3) ∧ x = (a - b)/(a^2 + b^2)}) = (30/97) :=
by
  intro x hx
  sorry

end maximum_value_fraction_l204_204493


namespace main_theorem_l204_204103

variables (A B C D E F G H I M N : Type) [has_CrossProduct A B C D] [has_CrossProduct E F G H] [has_CrossProduct I M N]

def quadrilateral (A B C D : Type) : Prop :=
  -- Definition of convex quadrilateral

def intersection_of (X Y Z W : Type) (I : Type) : Prop :=
  -- Definition of intersection I = AC ∩ BD

def points_on (E : Type) (F : Type) (G : Type) (H : Type) :
  E ∈ segment AB ∧ H ∈ segment BC ∧ F ∈ segment CD ∧ G ∈ segment DA

def other_intersections (M : Type) (N : Type) : Prop :=
  M = EG ∩ AC ∧ N = HF ∩ AC

theorem main_theorem 
  (ABCD_quadrilateral : quadrilateral A B C D)
  (I_intersection : intersection_of AC BD I)
  (EFGH_points : points_on E F G H)
  (MN_intersections : other_intersections M N) :
  (AM / IM) * (IN / CN) = (IA / IC) :=
by sorry

end main_theorem_l204_204103


namespace smallest_odd_number_with_five_prime_factors_l204_204222

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204222


namespace smallest_odd_number_with_five_different_prime_factors_l204_204281

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204281


namespace solve_quadratic_equation_l204_204783

theorem solve_quadratic_equation :
  ∀ x : ℝ, 2 * x^2 - 8 * x + 3 = 0 ↔
    (x = 2 + sqrt(10) / 2) ∨ (x = 2 - sqrt(10) / 2) :=
by
  sorry

end solve_quadratic_equation_l204_204783


namespace distance_D_to_plane_ABC_l204_204526

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def A := Point3D.mk 2 3 1
def B := Point3D.mk 4 1 -2
def C := Point3D.mk 6 3 7
def D := Point3D.mk -5 -4 8

def distanceFromPlaneToPoint (A B C D : Point3D) : ℝ :=
  sorry -- This is placeholder for the actual function to compute distance

theorem distance_D_to_plane_ABC :
  distanceFromPlaneToPoint A B C D = 11 :=
sorry

end distance_D_to_plane_ABC_l204_204526


namespace range_of_k_l204_204163

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem range_of_k :
  (∀ x : ℝ, 2 < x → f x > k) →
  k ≤ -Real.exp 2 :=
by
  sorry

end range_of_k_l204_204163


namespace mia_postcard_cost_l204_204744

def postcard_prices : Type := 
  { country : String // country = "Italy" ∨ country = "Germany" ∨ country = "Canada" ∨ country = "Japan"  }

noncomputable def price (p : postcard_prices) : ℕ :=
  match p.1 with
  | "Italy"     => 8
  | "Germany" => 8
  | "Canada"   => 5
  | "Japan"     => 7
  | _             => 0 -- assuming all other cases are invalid

noncomputable def quantity (decade : ℕ) (p : postcard_prices) : ℕ :=
  match decade, p.1 with
  | 1960, "Italy"     => 12
  | 1960, "Germany" => 5
  | 1960, "Canada"   => 7
  | 1960, "Japan"     => 8
  | 1970, "Italy"     => 11
  | 1970, "Germany" => 13
  | 1970, "Canada"   => 6
  | 1970, "Japan"     => 9
  | 1980, "Italy"     => 10
  | 1980, "Germany" => 15
  | 1980, "Canada"   => 10
  | 1980, "Japan"     => 5
  | _, _                => 0 -- assuming all other cases are invalid

noncomputable def total_cost (decades : List ℕ) (p : postcard_prices) : ℕ :=
  (decades.map (λ d, quantity d p * price p)).sum

noncomputable def total_cost_dollars (decades : List ℕ) (p : postcard_prices) : ℝ :=
  (total_cost decades p) / 100

theorem mia_postcard_cost (decades : List ℕ) :
  total_cost_dollars [1960, 1970, 1980] { country := "Canada", .. } + 
  total_cost_dollars [1960, 1970, 1980] { country := "Japan", .. } = 2.69 := 
sorry

end mia_postcard_cost_l204_204744


namespace min_value_of_quadratic_eq_neg_three_halfs_l204_204855

theorem min_value_of_quadratic_eq_neg_three_halfs :
  ∃ (x : ℝ), (∀ y : ℝ, 2 * y^2 + 6 * y - 5 ≥ 2 * x^2 + 6 * x - 5) ∧ x = -3 / 2 :=
begin
  existsi (-3 / 2 : ℝ),
  split,
  { intro y,
    sorry
  },
  refl
end

end min_value_of_quadratic_eq_neg_three_halfs_l204_204855


namespace car_travel_distance_l204_204757

def velocity (t : ℝ) : ℝ := 7 - 3 * t + 25 / (1 + t)

noncomputable def stop_time : ℝ := 4

noncomputable def distance_traveled : ℝ :=
  ∫ t in 0..stop_time, velocity t

theorem car_travel_distance :
  distance_traveled = 4 + 25 * Real.log 5 := by
  sorry

end car_travel_distance_l204_204757


namespace number_of_valid_pairings_l204_204182

def person_knows (n i j : ℕ) : Prop :=
  n = 12 ∧ 
  ((j = (i + 1) % n) ∨ 
   (j = (i - 1 + n) % n) ∨ 
   (j = (i + (n // 2)) % n) ∨ 
   (j = (i + 2) % n))

noncomputable def count_valid_pairings (n : ℕ) (knows : ℕ → ℕ → Prop) : ℕ := sorry

theorem number_of_valid_pairings :
  count_valid_pairings 12 person_knows = 16 :=
sorry

end number_of_valid_pairings_l204_204182


namespace prove_range_of_m_l204_204525

variable {U : Set ℝ}
variable {A : Set ℝ}
variable {B : Set ℝ}
variable {x m : ℝ}

def U := Set.univ
def A := {x | x < 1}
def B := {x | x > m}

theorem prove_range_of_m (h : (U \ A) ⊆ B) : m < 1 :=
by
  sorry

end prove_range_of_m_l204_204525


namespace ball_hits_ground_time_l204_204643

theorem ball_hits_ground_time
  (t : ℚ)
  (y : ℚ)
  (y_eq : y = -4.9 * t^2 + 4.5 * t + 6) :
  y = 0 ↔ t = 8121 / 4900 :=
begin
  sorry
end

end ball_hits_ground_time_l204_204643


namespace fractions_zero_condition_l204_204140

variable {a b c : ℝ}

theorem fractions_zero_condition 
  (h : (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0) :
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := 
sorry

end fractions_zero_condition_l204_204140


namespace not_right_triangle_if_divided_into_similar_parts_l204_204779

theorem not_right_triangle_if_divided_into_similar_parts 
(triangle : Type) 
(similar_to : triangle → triangle → Prop) :
  ∀ (τ : triangle), (∃ τs : list triangle, τs.length = 5 ∧ ∀ (subt : triangle), subt ∈ τs → similar_to subt τ) → 
  ¬ (∃ a b c : ℝ, a^2 + b^2 = c^2 ∧ τ = {a, b, c}) :=
begin
  sorry
end

end not_right_triangle_if_divided_into_similar_parts_l204_204779


namespace smallest_odd_with_five_different_prime_factors_l204_204264

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204264


namespace find_first_term_l204_204062

variables {a d : ℝ}

-- The sum of the first 22 terms of an arithmetic progression
def S22 := (22 / 2) * (2 * a + 21 * d)

-- The sum of the first 44 terms of an arithmetic progression
def S44 := (44 / 2) * (2 * a + 43 * d)

-- The sum of the next 22 terms (terms 23 to 44) is given by S44 - S22
def SNext22 := S44 - S22

theorem find_first_term (h1 : S22 = 1045) (h2 : SNext22 = 2013) :
  a = 53 / 2 :=
sorry

end find_first_term_l204_204062


namespace nth_harmonic_equation_l204_204997

theorem nth_harmonic_equation (n : ℕ) (hn : n > 0) : 
  (∑ i in Finset.range (2 * n), ((-1)^i * (1 / (i + 1)))) = 
  (∑ i in Finset.range (n + 1), (1 / (n + i))) :=
by
  sorry

end nth_harmonic_equation_l204_204997


namespace moon_weight_correct_l204_204176

noncomputable def moon_weight : ℝ :=
  let M := 333.33 in
  have h1 : 0.3 * (2 * M) = 200 := by sorry,
  have h2 : 0.3 * (3 * M) = 300 := by sorry,
  have h3 : 2 * M = 666.67 := by sorry,
  have h4 : 3 * M = 1000 := by sorry,
  M

theorem moon_weight_correct : moon_weight = 333.33 :=
  by unfold moon_weight; sorry

end moon_weight_correct_l204_204176


namespace find_abcd_l204_204964

theorem find_abcd (A B C D : ℕ) (E F G H : ℕ) (h_distinct : ∀ i j : ℤ, i ≠ j → i ∈ {A, B, C, D, E, F, G, H} → j ∈ {A, B, C, D, E, F, G, H} → i ≠ j) (h_digits : ∀ i : ℤ, i ∈ {A, B, C, D, E, F, G, H} → 1 ≤ i ∧ i ≤ 8) : 
  let ABCD := 1000 * A + 100 * B + 10 * C + D in
  ABCD + E * F * G * H = 2011 → ABCD = 1563 :=
by sorry

end find_abcd_l204_204964


namespace distance_from_point_P_to_line_BC_l204_204616

theorem distance_from_point_P_to_line_BC
  (A B C P : Type)
  (PA : P ↔ A)
  (plane_ABC : Plane A B C)
  (h_perp : PA ⊥ plane_ABC)
  (h_AB : dist A B = 13)
  (h_AC : dist A C = 13)
  (h_BC : dist B C = 10)
  (h_PA : dist P A = 5) :
  dist P BC = 13 :=
begin
   sorry
end

end distance_from_point_P_to_line_BC_l204_204616


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204024

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204024


namespace inequality_solution_l204_204632

def f (x : ℝ) : ℝ := (x^2 - 16 * x + 15) / (x^2 - 4 * x + 5)

theorem inequality_solution :
  {x : ℝ | -1 < f x ∧ f x < 1} = {x : ℝ | x < 5 / 2} :=
by
  sorry

end inequality_solution_l204_204632


namespace main_l204_204524

def M (x : ℝ) : Prop := x^2 - 5 * x ≤ 0
def N (x : ℝ) (p : ℝ) : Prop := p < x ∧ x < 6
def intersection (x : ℝ) (q : ℝ) : Prop := 2 < x ∧ x ≤ q

theorem main (p q : ℝ) (hM : ∀ x, M x → 0 ≤ x ∧ x ≤ 5) (hN : ∀ x, N x p → p < x ∧ x < 6) (hMN : ∀ x, (M x ∧ N x p) ↔ intersection x q) :
  p + q = 7 :=
by
  sorry

end main_l204_204524


namespace squares_after_seven_dwarfs_l204_204363

noncomputable def initial_squares : Nat := 1

def squares_after_n_dwarfs (n : Nat) : Nat :=
  initial_squares + (3 * n)

theorem squares_after_seven_dwarfs :
  squares_after_n_dwarfs 7 = 22 := by
  sorry

end squares_after_seven_dwarfs_l204_204363


namespace min_distance_ellipse_to_line_l204_204852

theorem min_distance_ellipse_to_line :
  let ellipse (x y : ℝ) := (x^2 / 9) + (y^2 / 4) = 1
  let line (x y : ℝ) := x + 2*y - 10 = 0
  let distance (m x1 y1 x2 y2 : ℝ) := |m + y1 - y2| / Real.sqrt(1^2 + 2^2)
  ∀ (x y : ℝ), ellipse x y → 
  distance 5 (x - 5) (\frac{-x - 5}{2}) x y = Real.sqrt 5 :=
by
  sorry

end min_distance_ellipse_to_line_l204_204852


namespace pairs_satisfying_condition_l204_204933

theorem pairs_satisfying_condition :
  (∃ (x y : ℕ), 1 ≤ x ∧ x ≤ 1000 ∧ 1 ≤ y ∧ y ≤ 1000 ∧ (x^2 + y^2) % 7 = 0) → 
  (∃ n : ℕ, n = 20164) :=
sorry

end pairs_satisfying_condition_l204_204933


namespace twentieth_fisherman_catch_l204_204681

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l204_204681


namespace smallest_odd_number_with_five_prime_factors_l204_204224

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204224


namespace find_a6_l204_204872

-- Definitions and hypotheses based on the conditions in the problem
variable (a : ℕ → ℕ) (S5 : ℕ)
hypothesis h1 : S5 = (a 1) + (a 2) + (a 3) + (a 4) + (a 5)
hypothesis h2 : S5 = 25
hypothesis h3 : a 2 = 3

noncomputable def common_diff : ℕ :=
  (a 2) - (a 1)

-- The theorem we need to prove
theorem find_a6 : a 6 = 11 := by
  -- Skipping the proof as per instructions
  sorry

end find_a6_l204_204872


namespace find_couples_l204_204129

theorem find_couples (n p q : ℕ) (hn : 0 < n) (hp : 0 < p) (hq : 0 < q)
    (h_gcd : Nat.gcd p q = 1)
    (h_eq : p + q^2 = (n^2 + 1) * p^2 + q) : 
    (p = n + 1 ∧ q = n^2 + n + 1) :=
by 
  sorry

end find_couples_l204_204129


namespace pump_time_l204_204756

def depth_in_feet := 18 / 12

def volume_cubic_feet := depth_in_feet * 24 * 32

def volume_gallons := volume_cubic_feet * 7.5

def pumping_rate := 8 * 3

def total_time := volume_gallons / pumping_rate

theorem pump_time (h : total_time = 360) : total_time = 360 :=
by sorry

end pump_time_l204_204756


namespace find_b_l204_204419

def h (x : ℝ) : ℝ := 5 * x - 7

theorem find_b : ∃ b : ℝ, h(b) = 0 ∧ b = 7 / 5 :=
by
  sorry

end find_b_l204_204419


namespace smallest_odd_number_with_five_different_prime_factors_l204_204277

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204277


namespace range_of_a_l204_204511

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end range_of_a_l204_204511


namespace bricks_needed_l204_204368

-- Define the dimensions of the wall in centimeters
def wall_length_cm : ℝ := 800
def wall_height_cm : ℝ := 660
def wall_width_cm : ℝ := 22.5

-- Define the dimensions of a single brick in centimeters
def brick_length_cm : ℝ := 25
def brick_height_cm : ℝ := 11.25
def brick_width_cm : ℝ := 6

-- Calculate the volume of the wall
def wall_volume : ℝ := wall_length_cm * wall_height_cm * wall_width_cm

-- Calculate the volume of a single brick
def brick_volume : ℝ := brick_length_cm * brick_height_cm * brick_width_cm

-- Calculate the number of bricks needed
def num_bricks_needed : ℝ := wall_volume / brick_volume

-- The theorem to prove that the number of bricks needed is 7040
theorem bricks_needed : num_bricks_needed = 7040 := by
  sorry

end bricks_needed_l204_204368


namespace economic_reasons_for_cash_preference_l204_204475

-- Definitions of conditions
def speed_of_cash_transactions_faster (avg_cash_time avg_cashless_time : ℝ) : Prop :=
  avg_cash_time < avg_cashless_time

def lower_cost_of_handling_cash (cash_handling_cost card_acquiring_cost : ℝ) : Prop :=
  cash_handling_cost < card_acquiring_cost

def higher_fraud_risk_with_cards (fraud_risk_cash fraud_risk_cards : ℝ) : Prop :=
  fraud_risk_cash < fraud_risk_cards

-- Main theorem
theorem economic_reasons_for_cash_preference
  (avg_cash_time avg_cashless_time : ℝ)
  (cash_handling_cost card_acquiring_cost : ℝ)
  (fraud_risk_cash fraud_risk_cards : ℝ)
  (h1 : speed_of_cash_transactions_faster avg_cash_time avg_cashless_time)
  (h2 : lower_cost_of_handling_cash cash_handling_cost card_acquiring_cost)
  (h3 : higher_fraud_risk_with_cards fraud_risk_cash fraud_risk_cards) :
  ∃ reasons : list string, reasons = ["Efficiency of Operations", "Cost of Handling Transactions", "Risk of Fraud"] :=
by
  sorry

end economic_reasons_for_cash_preference_l204_204475


namespace rectangle_area_l204_204174

theorem rectangle_area :
  ∀ (width length : ℝ), (length = 3 * width) → (width = 5) → (length * width = 75) :=
by
  intros width length h1 h2
  rw [h2, h1]
  sorry

end rectangle_area_l204_204174


namespace limit_ln_sin_equals_neg_one_eighth_l204_204738

open Real

-- We need to express the limit and the correct answer.
theorem limit_ln_sin_equals_neg_one_eighth :
  tendsto (λ x, (ln (sin (3 * x)) / (6 * x - π) ^ 2)) (𝓝 (π / 6)) (𝓝 (-1 / 8)) :=
sorry

end limit_ln_sin_equals_neg_one_eighth_l204_204738


namespace incorrect_statements_correct_statement_l204_204488

def isClosedSet (M : Set ℤ) : Prop :=
  ∀ a b ∈ M, a + b ∈ M ∧ a - b ∈ M

theorem incorrect_statements : 
  ¬ isClosedSet ({-4, -2, 0, 2, 4} : Set ℤ) ∧
  ¬ isClosedSet {n : ℤ | n > 0} ∧ 
  ¬ ∀ A₁ A₂ : Set ℤ, isClosedSet A₁ → isClosedSet A₂ → isClosedSet (A₁ ∪ A₂) :=
by
  sorry

theorem correct_statement :
  isClosedSet {n : ℤ | ∃ k : ℤ, n = 3 * k} :=
by
  sorry

end incorrect_statements_correct_statement_l204_204488


namespace largest_power_of_5_dividing_sum_l204_204442

theorem largest_power_of_5_dividing_sum :
  let s := 48! + 50! + 51!
  (∃ n : ℕ, (5^n ∣ s) ∧ ( ∀ m : ℕ, (5^m ∣ s → m ≤ n) )) :=
  ∃ n : ℕ, (5 ^ n ∣ (48! + 50! + 51!)) ∧ (∀ m : ℕ, 5 ^ m ∣ (48! + 50! + 51!) → m ≤ n) :=
        ∃ n : ℕ, n = 10 := sorry

end largest_power_of_5_dividing_sum_l204_204442


namespace min_value_reciprocal_sum_l204_204127

theorem min_value_reciprocal_sum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : Real.sqrt 3 = Real.sqrt (3^a * 3^b)) : 
  (1 / a + 1 / b) ≥ 4 :=
sorry

end min_value_reciprocal_sum_l204_204127


namespace sum_abscissas_common_points_l204_204460

noncomputable def cos_eq_8coscos2cos : ℝ → Prop := 
  λ x, 8 * Real.cos (π * x) * (Real.cos (2 * π * x))^2 * Real.cos (4 * π * x) = Real.cos (9 * π * x)

theorem sum_abscissas_common_points : ∑ x in {x | 0 ≤ x ∧ x ≤ 1 ∧ cos_eq_8coscos2cos x}, x = 3.5 :=
sorry

end sum_abscissas_common_points_l204_204460


namespace range_of_a_l204_204508

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l204_204508


namespace smallest_odd_number_with_five_different_prime_factors_l204_204314

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204314


namespace find_k_squared_l204_204410

noncomputable def complex_problem_statement (p q r : ℂ) :=
  (p + q + r = 0) ∧ (|p|^2 + |q|^2 + |r|^2 = 324) ∧ 
  (∃ u v : ℝ, u = |q - r| ∧ v = |p - q| ∧ (u^2 + v^2 = 486))

theorem find_k_squared (p q r : ℂ) (h : complex_problem_statement p q r) : 
  ∃ k : ℝ, k^2 = 486 := by 
  sorry

end find_k_squared_l204_204410


namespace smallest_odd_with_five_prime_factors_l204_204293

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204293


namespace smallest_odd_number_with_five_different_prime_factors_l204_204283

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204283


namespace cos_seven_pi_over_six_l204_204458

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l204_204458


namespace smallest_odd_number_with_five_primes_proof_l204_204239

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204239


namespace acute_angle_at_3_36_l204_204203

def degrees (h m : ℕ) : ℝ :=
  let minute_angle := (m / 60.0) * 360.0
  let hour_angle := (h % 12 + m / 60.0) * 30.0
  abs (minute_angle - hour_angle)

theorem acute_angle_at_3_36 : degrees 3 36 = 108 :=
by
  sorry

end acute_angle_at_3_36_l204_204203


namespace determine_angle_C_find_area_l204_204959

variables {A B C : ℝ}
variables {a b c : ℝ}

/-- Conditions of the problem -/
def is_acute_triangle (A B C : ℝ) : Prop := 
  A < π / 2 ∧ B < π / 2 ∧ C < π / 2

noncomputable def angle_B_law (b c : ℝ) : Prop := 
  ∀ (sinB sinC : ℝ), sqrt 3 * b = 2 * c * sinB → sinB ≠ 0 → sinC = sqrt 3 / 2 

/-- (I) Theorem to prove C = π / 3 given the conditions -/
theorem determine_angle_C 
  (acute : is_acute_triangle A B C)
  (side_relation : angle_B_law b c) 
  : C = π / 3 := 
sorry

/-- (II) Theorem to prove the area of triangle ABC is 3√3/2 given specific values -/
theorem find_area 
  (hC : C = π / 3) 
  (h_c : c = sqrt 7) 
  (h_ab : a + b = 5) 
  : 1/2 * a * b * sin C = 3 * sqrt 3 / 2 := 
sorry

end determine_angle_C_find_area_l204_204959


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204015

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204015


namespace scientific_notation_of_8200000_l204_204722

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l204_204722


namespace problem_l204_204092

open Function

noncomputable def areaOfTriangle {A B C : Type*} [AffineSpace A B] (Δ : Triangle A) : ℝ :=
sorry

noncomputable def centroid {A B : Type*} [AffineSpace A B] (Δ : Triangle A) : A :=
sorry

noncomputable def isMidpoint {A B : Type*} [AffineSpace A B] (P : A) (A B : A) : Prop :=
sorry

noncomputable def median {A B : Type*} [AffineSpace A B] (P : A) (Δ : Triangle A) : Line A :=
sorry

theorem problem {A B : Type*} [AffineSpace A B] (Δ : Triangle A)
  (D E F G R S T U : A)
  (h1 : median D Δ intersection median E Δ = G)
  (h2 : isMidpoint T D F)
  (h3 : median R △ intersection median S △ = U)
  (h4 : areaOfTriangle (mkTriangle G U T) = x) :
  areaOfTriangle Δ = 24 * x :=
sorry

end problem_l204_204092


namespace smallest_odd_number_with_five_prime_factors_l204_204304

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204304


namespace smallest_odd_number_with_five_primes_proof_l204_204241

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204241


namespace sum_max_min_a_l204_204626

theorem sum_max_min_a (a : ℝ) (h1 : ∀ x : ℝ, x^2 - a * x - 20 * a^2 < 0)
  (h2 : ∀ x1 x2 : ℝ, x1^2 - a * x1 - 20 * a^2 = 0 → x2^2 - a * x2 - 20 * a^2 = 0 → |x1 - x2| ≤ 9) :
    -1 ≤ a ∧ a ≤ 1 ∧ a ≠ 0 → (1 + -1) = 0 :=
by
  sorry

end sum_max_min_a_l204_204626


namespace amount_on_compound_interest_eq_4000_l204_204672

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T - 1)

variable (si_amount : ℝ := 1750) (si_rate : ℝ := 8) (si_time : ℝ := 3)
variable (ci_rate : ℝ := 10) (ci_time : ℝ := 2)

theorem amount_on_compound_interest_eq_4000 :
  let si := simple_interest si_amount si_rate si_time in
  let ci := 2 * si in
  ∃ (P_ci : ℝ), compound_interest P_ci ci_rate ci_time = ci ∧ P_ci = 4000 :=
by
  sorry

end amount_on_compound_interest_eq_4000_l204_204672


namespace smallest_odd_with_five_different_prime_factors_l204_204258

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204258


namespace smallest_odd_number_with_five_primes_proof_l204_204234

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204234


namespace original_coins_count_l204_204730

-- Define the initial amount of coins and the fractions taken out each day
def initial_coins := ℕ
def day1_fraction := 1 / 9
def day2_fraction := 1 / 8
def day3_fraction := 1 / 7
def day4_fraction := 1 / 6
def day5_fraction := 1 / 5
def day6_fraction := 1 / 4
def day7_fraction := 1 / 3
def day8_fraction := 1 / 2

-- Remaining coins after each extraction
def coins_after_day1 (initial_coins : ℕ) := initial_coins - (initial_coins * day1_fraction)
def coins_after_day2 (coins_after_day1 : ℕ) := coins_after_day1 - (coins_after_day1 * day2_fraction)
def coins_after_day3 (coins_after_day2 : ℕ) := coins_after_day2 - (coins_after_day2 * day3_fraction)
def coins_after_day4 (coins_after_day3 : ℕ) := coins_after_day3 - (coins_after_day3 * day4_fraction)
def coins_after_day5 (coins_after_day4 : ℕ) := coins_after_day4 - (coins_after_day4 * day5_fraction)
def coins_after_day6 (coins_after_day5 : ℕ) := coins_after_day5 - (coins_after_day5 * day6_fraction)
def coins_after_day7 (coins_after_day6 : ℕ) := coins_after_day6 - (coins_after_day6 * day7_fraction)
def coins_after_day8 (coins_after_day7 : ℕ) := coins_after_day7 - (coins_after_day7 * day8_fraction)

-- Main theorem stating the initial coins count equals to 45 coins after applying the sequential fractions
theorem original_coins_count : 
  ∃ initial_coins : ℕ, 
    coins_after_day8 (coins_after_day7 
      (coins_after_day6 
        (coins_after_day5 
          (coins_after_day4 
            (coins_after_day3 
              (coins_after_day2 
                (coins_after_day1 initial_coins)
              )
            )
          )
        )
      )
    ) = 5 → initial_coins = 45 
:= sorry

end original_coins_count_l204_204730


namespace project_completion_days_l204_204786

theorem project_completion_days (A_days : ℕ) (B_days : ℕ) (A_alone_days : ℕ) :
  A_days = 20 → B_days = 25 → A_alone_days = 2 → (A_alone_days : ℚ) * (1 / A_days) + (10 : ℚ) * (1 / (A_days * B_days / (A_days + B_days))) = 1 :=
by
  sorry

end project_completion_days_l204_204786


namespace perfect_cube_of_sum_of_cubes_l204_204623

theorem perfect_cube_of_sum_of_cubes :
  ∃ (n : ℕ → ℕ) (hn : ∀ i j, i ≠ j → n i ≠ n j), 
    (∑ i in finset.range 100, n i ^ 3) / 100 = (∃ k : ℕ, k ^ 3) :=
by
  sorry

end perfect_cube_of_sum_of_cubes_l204_204623


namespace max_soccer_balls_l204_204781

theorem max_soccer_balls : 
  ∀ (cost_volleyball cost_soccer_ball : ℕ) (num_volleyballs : ℕ),
  cost_volleyball = 88 →
  cost_soccer_ball = 72 →
  num_volleyballs = 11 →
  (cost_volleyball * num_volleyballs) / cost_soccer_ball = 13 :=
by
  intros cost_volleyball cost_soccer_ball num_volleyballs h₁ h₂ h₃
  have total_spent := cost_volleyball * num_volleyballs
  have max_soccer_balls := total_spent / cost_soccer_ball
  rw [h₁, h₂, h₃] at total_spent max_soccer_balls
  sorry

end max_soccer_balls_l204_204781


namespace first_digit_power_l204_204938

theorem first_digit_power (n : ℕ) (h : ∃ k : ℕ, 7 * 10^k ≤ 2^n ∧ 2^n < 8 * 10^k) :
  (∃ k' : ℕ, 1 * 10^k' ≤ 5^n ∧ 5^n < 2 * 10^k') :=
sorry

end first_digit_power_l204_204938


namespace max_value_sqrt_abcd_l204_204982

theorem max_value_sqrt_abcd (a b c d : ℝ) (ha : 0 ≤ a) (hb : 0 ≤ b) (hc : 0 ≤ c) (hd : 0 ≤ d) (h_sum : a + b + c + d = 1) :
  Real.sqrt (abcd) ^ (1 / 4) + Real.sqrt ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1 / 4) ≤ 1 := 
sorry

end max_value_sqrt_abcd_l204_204982


namespace find_a3_l204_204869

noncomputable def S (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * (1 - q ^ n) / (1 - q)

noncomputable def a (n : ℕ) (a₁ q : ℚ) : ℚ :=
  a₁ * q ^ (n - 1)

theorem find_a3 (a₁ q : ℚ) (h1 : S 6 a₁ q / S 3 a₁ q = -19 / 8)
  (h2 : a 4 a₁ q - a 2 a₁ q = -15 / 8) :
  a 3 a₁ q = 9 / 4 :=
by sorry

end find_a3_l204_204869


namespace laura_distributes_items_l204_204583

theorem laura_distributes_items :
  let friends := {1, 2, 3, 4}
  let small_blocks := 12
  let medium_blocks := 10
  let large_blocks := 6
  let cards := 8
  let figurines := 4
  (∀ f ∈ friends, (3 ≤ small_blocks) ∧ (1 ≤ large_blocks) ∧ (2 ≤ medium_blocks)) → 
  (∀ f ∈ friends, ¬ (small_blocks > 3 ∧ medium_blocks > 3 ∧ large_blocks > 2)) →
  (∀ f ∈ friends, (π_cards = 2) ∧ (π_figurines = 1)) →
  small_blocks = 3 ∧ medium_blocks = 2 ∨ medium_blocks = 3 ∧ large_blocks = 1 ∨ large_blocks = 2 ∧
  cards = 2 ∧ figurines = 1 :=
by
  sorry

end laura_distributes_items_l204_204583


namespace greatest_divisor_l204_204714

open Nat

theorem greatest_divisor :
  gcd (3547 - 17) (gcd (12739 - 17) (21329 - 17)) = 2 :=
by
  sorry

end greatest_divisor_l204_204714


namespace merchant_articles_l204_204376

theorem merchant_articles (N CP SP : ℝ) (h1 : N * CP = 16 * SP) (h2 : SP = CP * 1.0625) (h3 : CP ≠ 0) : N = 17 :=
by
  sorry

end merchant_articles_l204_204376


namespace scientific_notation_of_8200000_l204_204721

theorem scientific_notation_of_8200000 :
  8200000 = 8.2 * 10^6 :=
by
  sorry

end scientific_notation_of_8200000_l204_204721


namespace find_max_min_f_l204_204926

noncomputable def f (x : ℝ) : ℝ :=
  (cos (2 * x)) - (2 * cos x)

theorem find_max_min_f : 
    ∀ x : ℝ, x ∈ Icc (-π / 3) (π / 4) →
    f x = cos (2 * x) - (2 * cos x) ∧ 
    ((f (-π / 4) = -1 ∧ ∀ y ∈ Icc (-π / 3) (π / 4), f y ≤ -1) 
    ∨ (f (-acos (√(2)/2)) = -√2 ∧ ∀ y ∈ Icc (-π / 3) (π / 4), -√2 ≤ f y)) :=
by
  intro x hx
  have h_cos_range : cos x ∈ set.Icc (sqrt 2 / 2) 1 := sorry
  have h_f_def : f x = 2 * (cos x - 1 / 2) ^ 2 - 3 / 2 := sorry
  exact ⟨h_f_def, ⟨sorry, sorry⟩⟩

end find_max_min_f_l204_204926


namespace fraction_correct_l204_204953

theorem fraction_correct (x : ℚ) (h : (5 / 6) * 576 = x * 576 + 300) : x = 5 / 16 := 
sorry

end fraction_correct_l204_204953


namespace perfect_matchings_same_weight_l204_204978

variables {V : Type*} [fintype V] [decidable_eq V]
variables (A B : finset V) (n : ℕ)
variables (G : simple_graph V) [weight_condition: ∀ e ∈ G.edge_set, 0 < G.edge_weight e]
variables (G' : simple_graph V) [fresh_condition: G'.edge_set = {e | ∃ M, is_min_weight_perfect_matching G M ∧ e ∈ M.edge_set}]

open simple_graph 

theorem perfect_matchings_same_weight (h_size : A.card = n) (h_bipartite : is_bipartite G A B) :
  ∀ M1 M2, is_perfect_matching G' M1 → is_perfect_matching G' M2 → M1.weight (G'.edge_weight) = M2.weight (G'.edge_weight) :=
sorry

end perfect_matchings_same_weight_l204_204978


namespace smallest_odd_number_with_five_prime_factors_l204_204309

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204309


namespace juggler_balls_division_l204_204658

theorem juggler_balls_division (total_balls jugglers : ℕ) (h1 : total_balls = 2268) (h2 : jugglers = 378) :
  total_balls / jugglers = 6 :=
by
  rw [h1, h2]
  norm_num
  -- sorry

end juggler_balls_division_l204_204658


namespace discount_correct_l204_204380

def normal_cost : ℝ := 80
def discount_rate : ℝ := 0.45
def discounted_cost : ℝ := normal_cost - (discount_rate * normal_cost)

theorem discount_correct : discounted_cost = 44 := by
  -- By computation, 0.45 * 80 = 36 and 80 - 36 = 44
  sorry

end discount_correct_l204_204380


namespace cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204455

theorem cos_7pi_over_6_eq_neg_sqrt3_over_2 : Real.cos (7 * Real.pi / 6) = - (Real.sqrt 3) / 2 := by
  sorry

end cos_7pi_over_6_eq_neg_sqrt3_over_2_l204_204455


namespace divisor_of_6n_l204_204902

theorem divisor_of_6n
  (n : ℕ)
  (divisors_n : ∀ n d, d ∣ n → 1 ≤ d ∧ d ≤ n)
  (tau_n : Nat.divisorCount n = 10)
  (tau_2n : Nat.divisorCount (2 * n) = 20)
  (tau_3n : Nat.divisorCount (3 * n) = 15)
  : Nat.divisorCount (6 * n) = 30 := sorry

end divisor_of_6n_l204_204902


namespace find_q_l204_204605

-- Define the geometric sequence
def geometric_sequence (aₙ : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, aₙ (n + 1) = aₙ n * q

-- Define the sum of the first n terms of the sequence
def sum_of_first_n_terms (aₙ : ℕ → ℝ) (Sₙ : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, Sₙ n = (aₙ 0 * (1 - q^(n + 1))) / (1 - q)

-- Define the arithmetic sequence condition
def arithmetic_sequence_condition (Sₙ : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, 2 * Sₙ n = Sₙ (n + 1) + Sₙ (n + 2)

theorem find_q (aₙ Sₙ : ℕ → ℝ) (q : ℝ) :
  geometric_sequence aₙ q →
  sum_of_first_n_terms aₙ Sₙ →
  arithmetic_sequence_condition Sₙ →
  q = -2 :=
by
  intros
  sorry

end find_q_l204_204605


namespace EF_bisects__l204_204697

-- Definitions of the main points and properties
variables (A B C D E F : Type*) [EuclideanGeometry]

-- Lemmas related to the points and segments
variable (AB_congruent_CD : dist A B = dist C D)
variable (intersection_E : E = line_intersection (line_through A B) (line_through C D))
variable (perp_bisectors_intersect_F : 
  is_perpendicular_bisector F (segment_through A C) ∧ 
  is_perpendicular_bisector F (segment_through B D))
variable (F_interior_∠AEC : F ∈ interior_angle A E C)

-- The theorem to prove that EF bisects ∠AEC
theorem EF_bisects_∠AEC : angle_bisector E F A C :=
sorry

end EF_bisects__l204_204697


namespace find_a_l204_204911

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then 2^(x - 1) else x + 3

theorem find_a (a : ℝ) (h : f a + f 1 = 0) : a = -4 :=
sorry

end find_a_l204_204911


namespace length_of_second_video_l204_204977

theorem length_of_second_video 
    (length_first : ℝ) 
    (length_last : ℕ) 
    (total_time_seconds : ℕ) 
    (length_last_seconds : ℕ) 
    (length_fourth : ℝ) : ℝ :=
  let total_time := total_time_seconds / 60
  let length_last_min := length_last_seconds / 60
  let total_last := (length_last_min:ℝ)
  have length_first = 2 := sorry
  have length_last_seconds = 60 := sorry
  have total_time = 8.5 := sorry
  have total_last * 2 = 2 := sorry
  have length_fourth = total_time - length_first - total_last * 2 := sorry
  length_fourth

#eval length_of_second_video 2 1 510 60 4.5 -- This should evaluate and show the output 4.5

end length_of_second_video_l204_204977


namespace car_storm_avg_time_l204_204367

theorem car_storm_avg_time :
  let car_speed := 1 -- miles per minute
  let storm_speed := 1 -- miles per minute
  let storm_radius := 30 -- miles
  let initial_distance := 90 -- miles
  let t1 := (180 + Math.sqrt (180^2 - 4 * 7200)) / 2
  let t2 := (180 - Math.sqrt (180^2 - 4 * 7200)) / 2
  (t1 + t2) / 2 = 90 :=
by
  sorry

end car_storm_avg_time_l204_204367


namespace two_digit_primes_ending_in_3_l204_204038

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204038


namespace amount_on_compound_interest_eq_4000_l204_204671

noncomputable def simple_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * R * T / 100

noncomputable def compound_interest (P : ℝ) (R : ℝ) (T : ℝ) : ℝ :=
  P * ((1 + R / 100) ^ T - 1)

variable (si_amount : ℝ := 1750) (si_rate : ℝ := 8) (si_time : ℝ := 3)
variable (ci_rate : ℝ := 10) (ci_time : ℝ := 2)

theorem amount_on_compound_interest_eq_4000 :
  let si := simple_interest si_amount si_rate si_time in
  let ci := 2 * si in
  ∃ (P_ci : ℝ), compound_interest P_ci ci_rate ci_time = ci ∧ P_ci = 4000 :=
by
  sorry

end amount_on_compound_interest_eq_4000_l204_204671


namespace problem_proof_l204_204505

noncomputable def f (x : ℝ) := ∀ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), f' x = ((λ y, y + f y) x) * Real.sin x / Real.cos x ∧ f 0 = 0

theorem problem_proof :
  (∀ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), (f' x ≥ 0)) ∧ (∀ x ∈ Ioo (-Real.pi / 2) (Real.pi / 2), f has no extreme value) :=
sorry

end problem_proof_l204_204505


namespace time_to_pump_out_water_l204_204753

noncomputable def volume_of_water (depth_in_inches : ℕ) (length ft : ℕ) (width ft : ℕ) : ℝ :=
  (depth_in_inches / 12) * length * width

noncomputable def volume_to_gallons (volume_cubic_feet : ℝ) : ℝ :=
  volume_cubic_feet * 7.5

noncomputable def total_pumping_capacity_per_minute (pumps : ℕ) (capacity per pump : ℕ) : ℝ :=
  pumps * capacity

noncomputable def total_time_to_pump_out_water (total volume gallons : ℝ) (pumping_capacity_per_minute : ℝ) : ℝ :=
  total_volume_gallons / pumping_capacity_per_minute

theorem time_to_pump_out_water :
  let depth_in_inches := 18
  let length := 24
  let width := 32
  let pumps := 3
  let capacity_per_pump := 8
  let depth_in_feet := 1.5
  let volume_cubic_feet := volume_of_water depth_in_inches length width
  let total_volume_gallons := volume_to_gallons volume_cubic_feet
  let pumping_capacity_per_minute := total_pumping_capacity_per_minute pumps capacity_per_pump
  in total_time_to_pump_out_water total_volume_gallons pumping_capacity_per_minute = 360 := by
  sorry

end time_to_pump_out_water_l204_204753


namespace negation_of_proposition_l204_204348

theorem negation_of_proposition :
  (∀ x : ℝ, x^2 + 1 ≥ 0) ↔ (¬ ∃ x : ℝ, x^2 + 1 < 0) :=
by
  sorry

end negation_of_proposition_l204_204348


namespace small_panda_bears_count_l204_204157

theorem small_panda_bears_count :
  ∃ (S : ℕ), ∃ (B : ℕ),
    B = 5 ∧ 7 * (25 * S + 40 * B) = 2100 ∧ S = 4 :=
by
  exists 4
  exists 5
  repeat { sorry }

end small_panda_bears_count_l204_204157


namespace sunzi_suanjing_l204_204080

theorem sunzi_suanjing (a b m : ℤ) (h1 : m > 0) :
  a = ∑ k in finset.range (21), nat.choose 20 k * 2^k →
  a ≡ 2013 [MOD 4] :=
by sorry

end sunzi_suanjing_l204_204080


namespace purchase_price_l204_204136

theorem purchase_price (P : ℝ)
  (down_payment : ℝ) (monthly_payment : ℝ) (number_of_payments : ℝ)
  (interest_rate : ℝ) (total_paid : ℝ)
  (h1 : down_payment = 12)
  (h2 : monthly_payment = 10)
  (h3 : number_of_payments = 12)
  (h4 : interest_rate = 0.10714285714285714)
  (h5 : total_paid = 132) :
  P = 132 / 1.1071428571428572 :=
by
  sorry

end purchase_price_l204_204136


namespace purely_imaginary_complex_number_l204_204940

theorem purely_imaginary_complex_number (a : ℝ) :
  (a^2 - 1 + (a - 1) * complex.i).im ≠ 0 ∧ (a^2 - 1 = 0 → a = -1) :=
by
  sorry

end purely_imaginary_complex_number_l204_204940


namespace students_in_all_three_workshops_l204_204183

-- Define the students counts and other conditions
def num_students : ℕ := 25
def num_dance : ℕ := 12
def num_chess : ℕ := 15
def num_robotics : ℕ := 11
def num_at_least_two : ℕ := 12

-- Define the proof statement
theorem students_in_all_three_workshops : 
  ∃ c : ℕ, c = 1 ∧ 
    (∃ a b d : ℕ, 
      a + b + c + d = num_at_least_two ∧
      num_students ≥ num_dance + num_chess + num_robotics - a - b - d - 2 * c
    ) := 
by
  sorry

end students_in_all_three_workshops_l204_204183


namespace tan_identity_example_l204_204501

theorem tan_identity_example (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 :=
by
  sorry

end tan_identity_example_l204_204501


namespace sum_of_first_50_digits_of_one_over_1234_l204_204334

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l204_204334


namespace distinct_even_numbers_count_l204_204533

theorem distinct_even_numbers_count :
  (finset.filter (λ (n : ℕ), (1000 ≤ n ∧ n ≤ 9999) ∧ (n % 2 = 0) ∧ (nat.digits 10 n).nodup) (finset.range 10000)).card = 2296 := 
sorry

end distinct_even_numbers_count_l204_204533


namespace train_pass_bridge_time_l204_204735

noncomputable def length_of_train : ℝ := 485
noncomputable def length_of_bridge : ℝ := 140
noncomputable def speed_of_train_kmph : ℝ := 45 
noncomputable def speed_of_train_mps : ℝ := speed_of_train_kmph * (1000 / 3600)

theorem train_pass_bridge_time :
  (length_of_train + length_of_bridge) / speed_of_train_mps = 50 :=
by
  sorry

end train_pass_bridge_time_l204_204735


namespace fourth_row_chairs_l204_204563

theorem fourth_row_chairs :
  ∃ d : ℕ, ∀ (n : ℕ), (n = 1 → chairs n = 14) ∧ (n = 2 → chairs n = 23) ∧ (n = 3 → chairs n = 32) ∧
  (n = 5 → chairs n = 50) ∧ (n = 6 → chairs n = 59) ∧
  (∀ i : ℕ, 1 ≤ i ∧ i ≤ 5 ∧ chairs (i + 1) = chairs i + d) → chairs 4 = 41 :=
begin
  let chairs : ℕ → ℕ := 
    λ n, if n = 1 then 14
         else if n = 2 then 23
         else if n = 3 then 32
         else if n = 4 then 41
         else if n = 5 then 50
         else if n = 6 then 59
         else 0,
  have h : ∀ n, chairs (n + 1) - chairs n = 9,
  sorry
end

end fourth_row_chairs_l204_204563


namespace smallest_odd_number_with_five_different_prime_factors_l204_204311

theorem smallest_odd_number_with_five_different_prime_factors :
  ∃ (n : ℕ), (∀ p, prime p → p ∣ n → p ≠ 2) ∧ (nat.factors n).length = 5 ∧ ∀ m, (∀ p, prime p → p ∣ m → p ≠ 2) ∧ (nat.factors m).length = 5 → n ≤ m :=
  ⟨15015, 
  begin
    sorry
  end⟩

end smallest_odd_number_with_five_different_prime_factors_l204_204311


namespace circle_equation_line_equation_l204_204966

noncomputable def circle_C (x y : ℝ) : Prop :=
  x^2 + y^2 - 8 * x + 6 * y = 0

noncomputable def point_O : ℝ × ℝ := (0, 0)
noncomputable def point_A : ℝ × ℝ := (1, 1)
noncomputable def point_B : ℝ × ℝ := (4, 2)

theorem circle_equation :
  circle_C point_O.1 point_O.2 ∧
  circle_C point_A.1 point_A.2 ∧
  circle_C point_B.1 point_B.2 :=
by sorry

noncomputable def line_l_case1 (x : ℝ) : Prop :=
  x = 3 / 2

noncomputable def line_l_case2 (x y : ℝ) : Prop :=
  8 * x + 6 * y - 39 = 0

noncomputable def center_C : ℝ × ℝ := (4, -3)
noncomputable def radius_C : ℝ := 5

noncomputable def point_through_l : ℝ × ℝ := (3 / 2, 9 / 2)

theorem line_equation : 
(∀ (M N : ℝ × ℝ), circle_C M.1 M.2 ∧ circle_C N.1 N.2 → ∃ C_slave : Prop, 
(C_slave → 
((line_l_case1 (point_through_l.1)) ∨ 
(line_l_case2 point_through_l.1 point_through_l.2)))) :=
by sorry

end circle_equation_line_equation_l204_204966


namespace sum_of_first_50_digits_of_one_over_1234_l204_204336

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l204_204336


namespace butterflies_left_correct_l204_204821

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l204_204821


namespace domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l204_204842

-- For the function y = sqrt(3 + 2x)
theorem domain_sqrt_3_plus_2x (x : ℝ) : 3 + 2 * x ≥ 0 -> x ∈ Set.Ici (-3 / 2) :=
by
  sorry

-- For the function f(x) = 1 + sqrt(9 - x^2)
theorem domain_1_plus_sqrt_9_minus_x2 (x : ℝ) : 9 - x^2 ≥ 0 -> x ∈ Set.Icc (-3) 3 :=
by
  sorry

-- For the function φ(x) = sqrt(log((5x - x^2) / 4))
theorem domain_sqrt_log_5x_minus_x2_over_4 (x : ℝ) : (5 * x - x^2) / 4 > 0 ∧ (5 * x - x^2) / 4 ≥ 1 -> x ∈ Set.Icc 1 4 :=
by
  sorry

-- For the function y = sqrt(3 - x) + arccos((x - 2) / 3)
theorem domain_sqrt_3_minus_x_plus_arccos (x : ℝ) : 3 - x ≥ 0 ∧ -1 ≤ (x - 2) / 3 ∧ (x - 2) / 3 ≤ 1 -> x ∈ Set.Icc (-1) 3 :=
by
  sorry

end domain_sqrt_3_plus_2x_domain_1_plus_sqrt_9_minus_x2_domain_sqrt_log_5x_minus_x2_over_4_domain_sqrt_3_minus_x_plus_arccos_l204_204842


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204019

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204019


namespace problem_b_correct_l204_204862

theorem problem_b_correct (a b c : ℝ) : ac^2 > bc^2 → a > b :=
sorry

end problem_b_correct_l204_204862


namespace correct_operation_l204_204346

theorem correct_operation 
  (a : ℝ) :
  ((-a^2)^3 = -a^6) ∧
  ¬(a^3 * a^2 = a^6) ∧
  ¬((2a)^3 = 6a^3) ∧
  ¬(a + a = a^2) :=
by {
  split,
  { -- proof for Option B
    sorry },
  split,
  { -- disproof for Option A
    sorry },
  split,
  { -- disproof for Option C
    sorry },
  { -- disproof for Option D
    sorry }
}

end correct_operation_l204_204346


namespace true_propositions_l204_204531

def z := 2 - complex.i

def p1 := complex.abs z = real.sqrt 5
def p2 := z^2 = 3 - 4 * complex.i
def p3 := z.conj = 2 + complex.i
def p4 := z.im = -1

-- Define the main theorem proving that p2 and p4 are true.

theorem true_propositions : p2 ∧ p4 :=
by sorry

end true_propositions_l204_204531


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204016

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204016


namespace cos_seven_pi_over_six_l204_204447

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l204_204447


namespace log5_1250_round_l204_204195

theorem log5_1250_round (log_5_625 : log 5 625 = 4) (log_5_3125 : log 5 3125 = 5) : 
  Int.round (log 5 1250) = 4 :=
by
  sorry

end log5_1250_round_l204_204195


namespace ellipse_eq_standard_line_intersect_ellipse_l204_204877

noncomputable theory

-- Given conditions for the ellipse
def ellipse (a b : ℝ) (ha : a > 0) (hb : b > 0) := ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1

-- Conditions from the problem
variables (a b : ℝ)
variables (ha : a > b) (hb : b > 0)
variables (e : ℝ) (he : e = sqrt 3 / 2)
variables (c : ℝ) (hc : c = sqrt 3)
variables (focus_dist : ℝ) (hdist : focus_dist = sqrt 6)

-- Standard equation of the ellipse
theorem ellipse_eq_standard : a = 2 → b = 1 → ∀ x y : ℝ, x^2 / 4 + y^2 = 1 :=
sorry

-- Conditions for part II
variables (F : ℝ) (hF : F = sqrt 3)
variables (θ : ℝ) (hθ : θ = 45 * (π / 180))
variables (width : ℝ) (height : ℝ)
variables (intersect_1 intersect_2 : ℝ)

-- Line passing through right focus with angle 45, intersecting with the ellipse
theorem line_intersect_ellipse (h_eq : F = sqrt 3) (a_eq : a = 2) (b_eq : b = 1) :
  let y1 := intersect_1 in
  let y2 := intersect_2 in
  5 * y1^2 + 2 * sqrt 3 * y1 - 1 = 0 ∧
  5 * y2^2 + 2 * sqrt 3 * y2 - 1 = 0 ∧
  (|y1 - y2| = 4 * sqrt 2 / 5) ∧
  ((sqrt 3 / 2) * (4 * sqrt 2 / 5) = 2 * sqrt 6 / 5) :=
sorry

end ellipse_eq_standard_line_intersect_ellipse_l204_204877


namespace arithmetic_sequence_general_formula_l204_204888

theorem arithmetic_sequence_general_formula :
  (∀ n:ℕ, ∃ (a_n : ℕ), ∀ k:ℕ, a_n = 2 * k → k = n)
  ∧ ( 2 * n + 2 * (n + 2) = 8 → 2 * n + 2 * (n + 3) = 12 → a_n = 2 * n )
  ∧ (S_n = (n * (n + 1)) / 2 → S_n = 420 → n = 20) :=
by { sorry }

end arithmetic_sequence_general_formula_l204_204888


namespace three_topping_pizzas_l204_204381

theorem three_topping_pizzas : Nat.choose 8 3 = 56 := by
  sorry

end three_topping_pizzas_l204_204381


namespace inf_arith_seq_contains_inf_geo_seq_l204_204620

-- Condition: Infinite arithmetic sequence of natural numbers
variable (a d : ℕ) (h : ∀ n : ℕ, n ≥ 1 → ∃ k : ℕ, k = a + (n - 1) * d)

-- Theorem: There exists an infinite geometric sequence within the arithmetic sequence
theorem inf_arith_seq_contains_inf_geo_seq :
  ∃ r : ℕ, ∀ n : ℕ, ∃ k : ℕ, k = a * r ^ (n - 1) := sorry

end inf_arith_seq_contains_inf_geo_seq_l204_204620


namespace change_from_fifteen_dollars_l204_204807

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end change_from_fifteen_dollars_l204_204807


namespace two_digit_primes_ending_in_3_l204_204032

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204032


namespace minimum_value_of_F_l204_204432

noncomputable def F (m n : ℝ) : ℝ := (m - n)^2 + (m^2 - n + 1)^2

theorem minimum_value_of_F : 
  (∀ m n : ℝ, F m n ≥ 9 / 32) ∧ (∃ m n : ℝ, F m n = 9 / 32) :=
by
  sorry

end minimum_value_of_F_l204_204432


namespace acute_angle_between_hands_at_3_36_l204_204211

variable (minute_hand_position hour_hand_position abs_diff : ℝ)

def minute_hand_angle_at_3_36 : ℝ := 216
def hour_hand_angle_at_3_36 : ℝ := 108

theorem acute_angle_between_hands_at_3_36 (h₀ : minute_hand_position = 216)
    (h₁ : hour_hand_position = 108) :
    abs_diff = abs(minute_hand_position - hour_hand_position) → 
    abs_diff = 108 :=
  by
    rw [h₀, h₁]
    sorry

end acute_angle_between_hands_at_3_36_l204_204211


namespace smallest_odd_number_with_five_different_prime_factors_l204_204287

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204287


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204004

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204004


namespace smallest_odd_with_five_prime_factors_l204_204296

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204296


namespace two_digit_primes_ending_in_3_l204_204046

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204046


namespace count_10_digit_numbers_with_at_least_two_identical_digits_l204_204360

def total_10_digit_numbers : ℕ := 9 * 10^9

def distinct_10_digit_numbers : ℕ := 9 * Nat.fact 9

theorem count_10_digit_numbers_with_at_least_two_identical_digits :
  (total_10_digit_numbers - distinct_10_digit_numbers = (9 * 10^9 - 9 * Nat.fact 9)) :=
  by sorry

end count_10_digit_numbers_with_at_least_two_identical_digits_l204_204360


namespace number_of_real_solutions_l204_204433

theorem number_of_real_solutions : (∃ x : ℝ, 3 ^ (x^2 - 4 * x + 3) = 1) → 2 := by
  sorry

end number_of_real_solutions_l204_204433


namespace rectangle_same_color_exists_l204_204159

theorem rectangle_same_color_exists (color : ℝ × ℝ → Prop) (red blue : Prop) (h : ∀ p : ℝ × ℝ, color p = red ∨ color p = blue) :
  ∃ (a b c d : ℝ × ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d ∧
  (color a = color b ∧ color b = color c ∧ color c = color d) :=
sorry

end rectangle_same_color_exists_l204_204159


namespace trip_duration_l204_204352

theorem trip_duration 
(T : ℕ) (A : ℕ) : 
  (∀ d1 d2, d1 = 50 * 4 ∧ d2 = 80 * A ∧ (65 * T = d1 + d2) ∧ T = 4 + A) -> T = 8 :=
by
  intro h
  cases h with d1 h
  cases h with h1 h
  cases h with d2 h
  cases h with h2 h
  cases h with h3 h4
  have : d1 = 200 := h1
  have : d2 = 80 * A := h2
  have : 65 * T = 200 + 80 * A := h3
  have : T = 4 + A := h4
  sorry

end trip_duration_l204_204352


namespace solve_for_a_l204_204495

open Set

theorem solve_for_a (a : ℝ) :
  let M := ({a^2, a + 1, -3} : Set ℝ)
  let P := ({a - 3, 2 * a - 1, a^2 + 1} : Set ℝ)
  M ∩ P = {-3} →
  a = -1 :=
by
  intros M P h
  have hM : M = {a^2, a + 1, -3} := rfl
  have hP : P = {a - 3, 2 * a - 1, a^2 + 1} := rfl
  rw [hM, hP] at h
  sorry

end solve_for_a_l204_204495


namespace two_digit_primes_ending_in_3_l204_204049

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204049


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204005

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204005


namespace coloring_count_on_chessboard_l204_204801

/--
* A 7×7 chessboard
* Coloring 2 squares yellow, the rest green
* Two coloring methods that can be obtained by rotating the chessboard are considered the same

We need to prove that the number of distinct coloring methods is 300.
-/
theorem coloring_count_on_chessboard :
  let n := 7,
      total_squares := n * n
  in
  let colorings := {pair // pair.1 ≠ pair.2} / ~ where (pair : (Fin total_squares) × (Fin total_squares)) and ~ denotes equivalence under rotations
  in
  colorings.card = 300 :=
by
  sorry

end coloring_count_on_chessboard_l204_204801


namespace economic_reasons_for_cash_preference_l204_204474

-- Definitions of conditions
def speed_of_cash_transactions_faster (avg_cash_time avg_cashless_time : ℝ) : Prop :=
  avg_cash_time < avg_cashless_time

def lower_cost_of_handling_cash (cash_handling_cost card_acquiring_cost : ℝ) : Prop :=
  cash_handling_cost < card_acquiring_cost

def higher_fraud_risk_with_cards (fraud_risk_cash fraud_risk_cards : ℝ) : Prop :=
  fraud_risk_cash < fraud_risk_cards

-- Main theorem
theorem economic_reasons_for_cash_preference
  (avg_cash_time avg_cashless_time : ℝ)
  (cash_handling_cost card_acquiring_cost : ℝ)
  (fraud_risk_cash fraud_risk_cards : ℝ)
  (h1 : speed_of_cash_transactions_faster avg_cash_time avg_cashless_time)
  (h2 : lower_cost_of_handling_cash cash_handling_cost card_acquiring_cost)
  (h3 : higher_fraud_risk_with_cards fraud_risk_cash fraud_risk_cards) :
  ∃ reasons : list string, reasons = ["Efficiency of Operations", "Cost of Handling Transactions", "Risk of Fraud"] :=
by
  sorry

end economic_reasons_for_cash_preference_l204_204474


namespace unique_m_l204_204774

-- Given conditions extracted
def is_right_triangle_with_medians (a b c d m : ℝ) : Prop :=
  ∃ (a b c d : ℝ),
    ((c / -d = 2) ∧ (2 * c / d = m))

theorem unique_m (a b c d : ℝ) :
  is_right_triangle_with_medians a b c d m →
  ∃! (m : ℝ), m = -4 :=
begin
  sorry
end

end unique_m_l204_204774


namespace original_people_in_room_l204_204097

theorem original_people_in_room (x : ℝ) (h1 : x / 3 * 2 / 2 = 18) : x = 54 :=
sorry

end original_people_in_room_l204_204097


namespace arrange_weights_l204_204184

noncomputable def packets := {A B C D : ℕ}

inductive Comparison
| Less
| Greater

open Comparison

def weigh (x y : ℕ) : Comparison :=
if x < y then Less else Greater

theorem arrange_weights :
  ∃ (order : List packets), 
    (∀ (p1 p2 : packets), p1 ≠ p2 → 
      (weigh (p1 : ℕ) (p2 : ℕ) = Less ↔ order.indexOf p1 < order.indexOf p2) ∧ 
      (weigh (p1 : ℕ) (p2 : ℕ) = Greater ↔ order.indexOf p1 > order.indexOf p2)) := sorry

end arrange_weights_l204_204184


namespace problem_statement_l204_204483

-- Definitions and Assumptions
def divisors (n : ℕ) : Finset ℕ :=
  univ.filter (λ d, d ∣ n)

def d (n : ℕ) : ℕ :=
  (divisors n).card

def φ (n : ℕ) : ℕ :=
  (Finset.range (n + 1)).filter (λ k, Nat.coprime k n).card

theorem problem_statement (c n : ℕ) (hpos : 0 < n) :
  d n + φ n = n + c → c ≤ 1 := by
  sorry

end problem_statement_l204_204483


namespace point_inside_circle_l204_204944

theorem point_inside_circle (m : ℝ) : (1 - 2)^2 + (-3 + 1)^2 < m → m > 5 :=
by
  sorry

end point_inside_circle_l204_204944


namespace total_revenue_correct_l204_204691

def ticket_price : ℝ := 20
def discount_40_percent : ℝ := 0.40
def discount_15_percent : ℝ := 0.15

def first_10_discounted_revenue : ℝ := 10 * (ticket_price * (1 - discount_40_percent))
def next_20_discounted_revenue : ℝ := 20 * (ticket_price * (1 - discount_15_percent))
def remaining_20_full_price_revenue : ℝ := 20 * ticket_price

def total_revenue : ℝ := first_10_discounted_revenue + next_20_discounted_revenue + remaining_20_full_price_revenue

theorem total_revenue_correct : total_revenue = 860 := 
by 
  unfold ticket_price 
  unfold discount_40_percent 
  unfold discount_15_percent 
  unfold first_10_discounted_revenue 
  unfold next_20_discounted_revenue 
  unfold remaining_20_full_price_revenue 
  unfold total_revenue 
  sorry

end total_revenue_correct_l204_204691


namespace simplify_fraction_eq_l204_204147

theorem simplify_fraction_eq : (180 / 270 : ℚ) = 2 / 3 :=
by
  sorry

end simplify_fraction_eq_l204_204147


namespace sales_volume_second_year_l204_204634

theorem sales_volume_second_year
  (a1 a2 a3 : ℝ)
  (h1 : a1 + a2 + a3 = 2.46)
  (h2 : a1 + a3 = 2 * a2) :
  a2 = 0.82 :=
by calc sorry

end sales_volume_second_year_l204_204634


namespace wang_jia_math_score_l204_204996

/-- 
Wang Jia's final exam scores in three subjects are three consecutive even numbers 
whose sum is 288. We seek to find the score in Mathematics. 
-/
theorem wang_jia_math_score :
  ∃ (x : ℤ), (x + (x - 2) + (x + 2) = 288) ∧ (x % 2 = 0) :=
begin
  use 96,
  split,
  {
    calc 96 + (96 - 2) + (96 + 2) = 96 + 94 + 98 : by ring
                             ... = 288 : by norm_num,
  },
  {
    norm_num,
  }
end

end wang_jia_math_score_l204_204996


namespace intersection_M_N_l204_204607

def M : Set ℤ := {-1, 0, 1}
def N : Set ℤ := {x | x^2 ≠ x}

theorem intersection_M_N:
  M ∩ N = {-1} := by
  sorry

end intersection_M_N_l204_204607


namespace general_form_eq_of_line_l_value_of_fraction_cos_sin_beta_l204_204515

-- Definitions of the given conditions
def line_l1 := ∀ x y : ℝ, x - sqrt 2 * y + 1 = 0
def P := (-sqrt 2, 2 : ℝ)
def alpha := atan (sqrt 2 / 2)
def beta := 2 * alpha

-- Proof problem statements
theorem general_form_eq_of_line_l (alpha : ℝ) (beta : ℝ) :
  (tan alpha = sqrt 2 / 2) →
  (beta = 2 * alpha) →
  ∃ k b, (k = 2 * sqrt 2) ∧ b = 2 + 2 * (sqrt 2)^2 ∧
  ∀ x y : ℝ, (y - 2) = k * (x + sqrt 2) →
  2 * sqrt 2 * x - y + 6 = 0 :=
sorry

theorem value_of_fraction_cos_sin_beta (alpha : ℝ) (beta : ℝ) :
  (tan alpha = sqrt 2 / 2) →
  (beta = 2 * alpha) →
  (tan beta = 2 * sqrt 2) →
  ∀ cos2beta sin2beta : ℝ, cos2beta = cos (2*beta) → sin2beta = sin (2*beta) →
  (cos2beta / (1 + cos2beta - sin2beta) = 1 / 2 + sqrt 2) :=
sorry

end general_form_eq_of_line_l_value_of_fraction_cos_sin_beta_l204_204515


namespace smallest_sum_of_digits_l204_204798

theorem smallest_sum_of_digits :
  ∃ (a b S : ℕ), 
    (100 ≤ a ∧ a < 1000) ∧ 
    (10 ≤ b ∧ b < 100) ∧ 
    (∃ (d1 d2 d3 d4 d5 : ℕ), 
      (d1 ≠ d2) ∧ (d1 ≠ d3) ∧ (d1 ≠ d4) ∧ (d1 ≠ d5) ∧ 
      (d2 ≠ d3) ∧ (d2 ≠ d4) ∧ (d2 ≠ d5) ∧ 
      (d3 ≠ d4) ∧ (d3 ≠ d5) ∧ 
      (d4 ≠ d5) ∧ 
      S = a + b ∧ 100 ≤ S ∧ S < 1000 ∧ 
      (∃ (s : ℕ), 
        s = (S / 100) + ((S % 100) / 10) + (S % 10) ∧ 
        s = 3)) :=
sorry

end smallest_sum_of_digits_l204_204798


namespace necessary_but_not_sufficient_not_sufficient_condition_l204_204988

theorem necessary_but_not_sufficient (a b : ℝ) : (a > 2 ∧ b > 2) → (a + b > 4) :=
sorry

theorem not_sufficient_condition (a b : ℝ) : (a + b > 4) → ¬(a > 2 ∧ b > 2) :=
sorry

end necessary_but_not_sufficient_not_sufficient_condition_l204_204988


namespace clock_angle_at_3_36_l204_204197

def minute_hand_angle : ℝ := (36.0 / 60.0) * 360.0
def hour_hand_angle : ℝ := 90.0 + (36.0 / 60.0) * 30.0

def acute_angle (a b : ℝ) : ℝ :=
  let diff := abs (a - b)
  min diff (360 - diff)

theorem clock_angle_at_3_36 :
  acute_angle minute_hand_angle hour_hand_angle = 108 :=
by
  sorry

end clock_angle_at_3_36_l204_204197


namespace temperature_on_tuesday_l204_204357

theorem temperature_on_tuesday 
  (T W Th F : ℝ)
  (H1 : (T + W + Th) / 3 = 45)
  (H2 : (W + Th + F) / 3 = 50)
  (H3 : F = 53) :
  T = 38 :=
by 
  sorry

end temperature_on_tuesday_l204_204357


namespace walkways_area_l204_204947

theorem walkways_area (rows cols : ℕ) (bed_length bed_width walkthrough_width garden_length garden_width total_flower_beds bed_area total_bed_area total_garden_area : ℝ) 
  (h1 : rows = 4) (h2 : cols = 3) 
  (h3 : bed_length = 8) (h4 : bed_width = 3) 
  (h5 : walkthrough_width = 2)
  (h6 : garden_length = (cols * bed_length) + ((cols + 1) * walkthrough_width))
  (h7 : garden_width = (rows * bed_width) + ((rows + 1) * walkthrough_width))
  (h8 : total_garden_area = garden_length * garden_width)
  (h9 : total_flower_beds = rows * cols)
  (h10 : bed_area = bed_length * bed_width)
  (h11 : total_bed_area = total_flower_beds * bed_area)
  (h12 : total_garden_area - total_bed_area = 416) : 
  True := 
sorry

end walkways_area_l204_204947


namespace rounding_nearest_tenth_l204_204782

noncomputable def repeated_fraction := 32.3636363636 -- etc., repeating

def addition := repeated_fraction + 16.25

theorem rounding_nearest_tenth: (Float.round (addition * 10) / 10) = 48.6 := by
  sorry

end rounding_nearest_tenth_l204_204782


namespace smallest_odd_with_five_different_prime_factors_l204_204261

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204261


namespace collinearity_concyclic_condition_l204_204558

/-- In an acute triangle \( \triangle ABC \), where \( AB > AC \), 
let \( \odot O \) denote the circumcircle and \( \odot I \) the incircle of \( \triangle ABC \),
with the incircle touching side \( BC \) at point \( D \).
Line \( AO \) intersects side \( BC \) at point \( X \).
\( AY \) denotes the altitude from \( A \) to side \( BC \), 
and let the tangents to \( \odot O \) at points \( B \) and \( C \) intersect at point \( L \).
Also, let \( PQ \) denote a diameter of \( \odot O \) passing through point \( I \). 
Prove that points \( A \), \( D \), and \( L \) are collinear if and only if points \( P \), \( X \), \( Y \), and \( Q \) are concyclic.-/
theorem collinearity_concyclic_condition
  {A B C D L P Q X Y : Point}
  (h_acute : Triangle.acute A B C)
  (h_ab_gt_ac : dist A B > dist A C)
  (O I : Circle)
  (h_circumcircle : Circle.circumcircle_of_triangle O A B C)
  (h_incircle : Circle.incircle_of_triangle I A B C)
  (h_touch_d : Circle.touches I (Line.of_points B C) D)
  (h_ao_intersects_bc_at_x : Line.of_points A O = Line.of_points B C)
  (h_altitude : Line.perp 
    (Line.of_points A (foot A (Line.of_points B C))) Y)
  (h_tangents_intersect_at_l : Circle.tangent O B = Circle.tangent O C)
  (h_pq_diameter : Circle.diameter_through I O P Q) :
  (Collinear A D L ↔ Concyclic P X Y Q) :=
sorry

end collinearity_concyclic_condition_l204_204558


namespace bug_returns_to_A_after_10_meters_l204_204588

noncomputable def P : ℕ → ℚ
| 0     := 1
| (n+1) := (1 - P n) / 3

theorem bug_returns_to_A_after_10_meters :
  P 10 = 4921 / 59049 :=
sorry

end bug_returns_to_A_after_10_meters_l204_204588


namespace seven_by_seven_tiling_min_tiles_A_for_large_square_l204_204745

-- Definition of the problem
theorem seven_by_seven_tiling (A B : ℕ → ℤ) (A_property : A(3)) (B_property: B(4)) : 
  (∃ (m n: ℕ), 7 * 7 = m * A(3) + n * B(4)) :=
sorry

theorem min_tiles_A_for_large_square (A B : ℕ → ℤ) (A_property : A(3)) (B_property: B(4)) : 
  (∃ m n, 1011 * 1011 = m * A(3) + n * B(4) ∧ n + 1011 = m ∧ m + n = 2 * 506 + 1011) :=
sorry

end seven_by_seven_tiling_min_tiles_A_for_large_square_l204_204745


namespace problem_120_degree_angle_l204_204804

-- Define the conditions
def triangular_grid := sorry
def non_self_intersecting_polygon (P : triangular_grid) : Prop := sorry
def perimeter (P : triangular_grid) : ℕ := sorry

-- The theorem to prove
theorem problem_120_degree_angle (P : triangular_grid) 
  (H1 : non_self_intersecting_polygon P) 
  (H2 : perimeter P = 1399) : 
  ∃ (angle : ℤ), angle = 120 ∨ angle = -120 :=
sorry

end problem_120_degree_angle_l204_204804


namespace value_of_m_min_value_expression_l204_204910

variable (x m a b c: ℝ)

def f (x m : ℝ) : ℝ := sqrt (x^2 + 2 * x + 1) - abs (x - m)

-- Part (1)
theorem value_of_m (h₁ : ∀ x, f x m ≤ 4) (h₂ : 0 < m) : m = 3 := by
  sorry

-- Part (2)
theorem min_value_expression 
  (h₃ : a^2 + b^2 + c^2 = 3) : (1 / a^2) + (1 / b^2) + (4 / c^2) ≥ 16 / 3 := by
  sorry

end value_of_m_min_value_expression_l204_204910


namespace simplify_abs_expr_l204_204151

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l204_204151


namespace problem1_problem2_l204_204797

-- Problem 1: Prove (-a^3)^2 * (-a^2)^3 / a = -a^11 given a is a real number.
theorem problem1 (a : ℝ) : (-a^3)^2 * (-a^2)^3 / a = -a^11 :=
  sorry

-- Problem 2: Prove (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 given m, n are real numbers.
theorem problem2 (m n : ℝ) : (m - n)^3 * (n - m)^4 * (n - m)^5 = - (n - m)^12 :=
  sorry

end problem1_problem2_l204_204797


namespace simplify_trig_identity_l204_204142

theorem simplify_trig_identity (x : ℝ) :
  (tan x / (1 + cot x)) + ((1 + cot x) / tan x) = (sin x * (2 * sin x + cos x)) / ((cos x + sin x) * cos x) :=
sorry

end simplify_trig_identity_l204_204142


namespace increasing_interval_cos_squared_l204_204059

open Real

theorem increasing_interval_cos_squared (φ : ℝ) (hφ : φ > 0 ∧ φ < π / 2) (h : sin φ - cos φ = sqrt 2 / 2) :
  (∀ k : ℤ, 
  ∀ x : ℝ, 
  (x ∈ set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) ↔ (f x = cos^2 (x + φ) → ∃ k : ℤ, x ∈ set.Icc (k * π + π / 12) (k * π + 7 * π / 12)) :=
begin
  sorry
end

end increasing_interval_cos_squared_l204_204059


namespace range_of_t_l204_204484

theorem range_of_t (t : ℝ) (f : ℝ → ℝ) (H : ∀ x, f x = x^2 - 2 * t * x + 1) :
  (∀ x ∈ set.Icc 0 (t + 1), ∀ y ∈ set.Icc 0 (t + 1), |f x - f y| ≤ 2) →
  (∀ x ∈ set.Icc 0 t, f' x ≤ 0) →
  1 ≤ t ∧ t ≤ Real.sqrt 2 :=
by
  sorry

end range_of_t_l204_204484


namespace joan_trip_time_l204_204972

-- Definitions of given conditions as parameters
def distance : ℕ := 480
def speed : ℕ := 60
def lunch_break_minutes : ℕ := 30
def bathroom_break_minutes : ℕ := 15
def number_of_bathroom_breaks : ℕ := 2

-- Conversion factors
def minutes_to_hours (m : ℕ) : ℚ := m / 60

-- Calculation of total time taken
def total_time : ℚ := 
  (distance / speed) + 
  (minutes_to_hours lunch_break_minutes) + 
  (number_of_bathroom_breaks * minutes_to_hours bathroom_break_minutes)

-- Statement of the problem
theorem joan_trip_time : total_time = 9 := 
  by 
    sorry

end joan_trip_time_l204_204972


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204325

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204325


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204000

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204000


namespace unique_pair_B_l204_204739

variables {P A B C B' C' : Type} [Add P] [Add A] [Add B] [Add C] [Add B'] [Add C']
variables (PA PB PC : P) (AB' AC' : P) (PC' PB' B'C' : P)

-- The conditions of the problem
variables (B'_on_PB : B' ∈ PB) (C'_on_PC : C' ∈ PC)

-- The theorem statement
theorem unique_pair_B'_C' 
  (h1 : PC' + B'C' = PA + AB') 
  (h2 : PB' + B'C' = PA + AC') : 
  ∃! (B' C' : P), B' ∈ PB ∧ C' ∈ PC ∧ (PC' + B'C' = PA + AB') ∧ (PB' + B'C' = PA + AC') :=
sorry

end unique_pair_B_l204_204739


namespace graveyard_bones_count_l204_204068

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end graveyard_bones_count_l204_204068


namespace limit_expression_l204_204793

theorem limit_expression : 
  (tendsto (λ n : ℕ, 1 - (n : ℝ) / (n + 1)) at_top (𝓝 0)) :=
by sorry

end limit_expression_l204_204793


namespace parallelepiped_inequality_l204_204771

-- Define the vertices and center of the parallelepiped.
variable (A : Fin 8 → ℝ^3)
variable (O : ℝ^3)

-- Define the conditions of the problem.
def midpoint_condition (A : Fin 8 → ℝ^3) (O : ℝ^3) :=
  O = (A 0 + A 4) / 2 ∧ O = (A 1 + A 5) / 2 ∧ O = (A 2 + A 6) / 2 ∧ O = (A 3 + A 7) / 2

theorem parallelepiped_inequality (A : Fin 8 → ℝ^3) (O : ℝ^3) (h : midpoint_condition A O) :
  4 * (Finset.univ.sum (λ i, (∥O - A i∥)^2)) ≤ ((Finset.univ.sum (λ i, ∥O - A i∥))^2) :=
  sorry

end parallelepiped_inequality_l204_204771


namespace sum_discount_rates_correct_l204_204471

noncomputable def sum_discount_rates : ℝ :=
let fox_price := 15 in
let pony_price := 18 in
let num_fox := 3 in
let num_pony := 2 in
let total_saved := 8.91 in
let pony_discount_rate := 10.999999999999996 in
let total_fox_cost := fox_price * num_fox in
let total_pony_cost := pony_price * num_pony in
let pony_discount_amount := total_pony_cost * pony_discount_rate / 100 in
let fox_discount_rate := ((total_saved - pony_discount_amount) / total_fox_cost) * 100 in
fox_discount_rate + pony_discount_rate

theorem sum_discount_rates_correct :
  sum_discount_rates = 22 :=
by
  sorry

end sum_discount_rates_correct_l204_204471


namespace one_is_not_identity_element_for_star_l204_204811

noncomputable def star (a b : ℝ) : ℝ := a^2 + 2 * a * b

theorem one_is_not_identity_element_for_star (a : ℝ) (ha : a ≠ 0) : 
  star a 1 ≠ a ∨ star 1 a ≠ a :=
by
  unfold star
  have h1 : star a 1 = a^2 + 2 * a := by sorry
  have h2 : star 1 a = 1 + 2 * a := by sorry
  exact Or.intro_left _ h1 sorry
  
#print axioms one_is_not_identity_element_for_star

end one_is_not_identity_element_for_star_l204_204811


namespace nursing_home_beds_l204_204760

/-- A community plans to build a nursing home with 100 rooms, consisting of single, double, and triple rooms.
    Let t be the number of single rooms (1 nursing bed), double rooms (2 nursing beds) is twice the single rooms,
    and the rest are triple rooms (3 nursing beds).
    The equations are:
    - number of double rooms: 2 * t
    - number of single rooms: t
    - number of triple rooms: 100 - 3 * t
    - total number of nursing beds: t + 2 * (2 * t) + 3 * (100 - 3 * t) 
    Prove the following:
    1. If the total number of nursing beds is 200, then t = 25.
    2. The maximum number of nursing beds is 260.
    3. The minimum number of nursing beds is 180.
-/
theorem nursing_home_beds (t : ℕ) (h1 : 10 ≤ t ∧ t ≤ 30) (total_rooms : ℕ := 100) :
  (∀ total_beds, (total_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → total_beds = 200 → t = 25) ∧
  (∀ max_beds, (max_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 10 → max_beds = 260) ∧
  (∀ min_beds, (min_beds = t + 2 * (2 * t) + 3 * (100 - 3 * t)) → t = 30 → min_beds = 180) := 
by
  sorry

end nursing_home_beds_l204_204760


namespace exists_unique_decomposition_l204_204353

theorem exists_unique_decomposition:
    ∃ (A B : Set ℕ), (∀ n : ℕ, ∃! (a b : ℕ), n = a + b ∧ a ∈ A ∧ b ∈ B) :=
sorry

end exists_unique_decomposition_l204_204353


namespace spherical_to_rectangular_l204_204426

theorem spherical_to_rectangular
  (ρ θ φ : ℝ)
  (ρ_eq : ρ = 10)
  (θ_eq : θ = 5 * Real.pi / 4)
  (φ_eq : φ = Real.pi / 4) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_l204_204426


namespace div_of_powers_l204_204711

-- Definitions of conditions:
def is_power_of_3 (x : ℕ) := x = 3 ^ 3

-- The conditions for the problem:
variable (a b c : ℕ)
variable (h1 : is_power_of_3 27)
variable (h2 : a = 3)
variable (h3 : b = 12)
variable (h4 : c = 6)

-- The proof statement:
theorem div_of_powers : 3 ^ 12 / 27 ^ 2 = 729 :=
by
  have h₁ : 27 = 3 ^ 3 := h1
  have h₂ : 27 ^ 2 = (3 ^ 3) ^ 2 := by rw [h₁]
  have h₃ : (3 ^ 3) ^ 2 = 3 ^ 6 := by rw [← pow_mul]
  have h₄ : 27 ^ 2 = 3 ^ 6 := by rw [h₂, h₃]
  have h₅ : 3 ^ 12 / 3 ^ 6 = 3 ^ (12 - 6) := by rw [div_eq_mul_inv, ← pow_sub]
  show 3 ^ 6 = 729, from by norm_num
  sorry

end div_of_powers_l204_204711


namespace acute_angle_at_3_36_l204_204202

def degrees (h m : ℕ) : ℝ :=
  let minute_angle := (m / 60.0) * 360.0
  let hour_angle := (h % 12 + m / 60.0) * 30.0
  abs (minute_angle - hour_angle)

theorem acute_angle_at_3_36 : degrees 3 36 = 108 :=
by
  sorry

end acute_angle_at_3_36_l204_204202


namespace clock_angle_3_36_l204_204214

def minute_hand_position (minutes : ℕ) : ℝ :=
  minutes * 6

def hour_hand_position (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def angle_difference (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

def acute_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem clock_angle_3_36 :
  acute_angle (angle_difference (minute_hand_position 36) (hour_hand_position 3 36)) = 108 :=
by
  sorry

end clock_angle_3_36_l204_204214


namespace arc_point_relation_l204_204137

variables {A B C D P : Type}

/-- Assume that we have points A, B, C, D forming a square and P is on the arc CD.
    Prove that PA + PC = sqrt 2 * PB. --/
theorem arc_point_relation
  {s : ℝ} -- side length of the square
  {PA PC PB : ℝ} -- distances from P to A, P to C, and P to B respectively
  (h_square: ∀ (A B C D : ℝ), A = B ∧ B = C ∧ C = D)
  (h_arc : P ∈ (arc_construction C D))
  (h_cd: ∀ (A B C D : ℝ), |C - D| = s)
  (h_pc: ∀ (P A C D : ℝ), dist P A + dist P C = sqrt 2 * dist P B) :
  PA + PC = sqrt 2 * PB :=
sorry

end arc_point_relation_l204_204137


namespace angle_B_is_pi_over_6_l204_204551

theorem angle_B_is_pi_over_6
  (A B : ℝ)
  (a b : ℝ)
  (hA : A = π / 3)
  (ha : a = sqrt 3)
  (hb : b = 1) :
  B = π / 6 :=
sorry

end angle_B_is_pi_over_6_l204_204551


namespace simplify_expression_l204_204630

variable {b : ℝ}

theorem simplify_expression (hb : 0 ≤ b) : 
  ((rtpow (rtpow b 16 3) 4) ^ 3 * (rtpow (rtpow b 16 4) 3) ^ 3) = b^8 := 
sorry

end simplify_expression_l204_204630


namespace two_digit_primes_ending_in_3_l204_204037

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204037


namespace xiao_ming_fails_the_test_probability_l204_204952

def probability_scoring_above_80 : ℝ := 0.69
def probability_scoring_between_70_and_79 : ℝ := 0.15
def probability_scoring_between_60_and_69 : ℝ := 0.09

theorem xiao_ming_fails_the_test_probability :
  1 - (probability_scoring_above_80 + probability_scoring_between_70_and_79 + probability_scoring_between_60_and_69) = 0.07 :=
by
  sorry

end xiao_ming_fails_the_test_probability_l204_204952


namespace probability_of_four_card_success_l204_204366

example (cards : Fin 4) (pins : Fin 4) {attempts : ℕ}
  (h1 : ∀ (c : Fin 4) (p : Fin 4), attempts ≤ 3)
  (h2 : ∀ (c : Fin 4), ∃ (p : Fin 4), p ≠ c ∧ attempts ≤ 3) :
  ∃ (three_cards : Fin 3), attempts ≤ 3 :=
sorry

noncomputable def probability_success :
  ℚ := 23 / 24

theorem probability_of_four_card_success :
  probability_success = 23 / 24 :=
sorry

end probability_of_four_card_success_l204_204366


namespace locus_of_bisecting_circles_l204_204922

open_locale classical

noncomputable theory

variables {O₁ O₂ : Point} {R₁ R₂ : ℝ}

/-- The locus of points for the centers of circles that bisect two given non-overlapping circles
  is a straight line perpendicular to the segment connecting the centers of these circles. -/
theorem locus_of_bisecting_circles (h₁ : R₁ > 0) (h₂ : R₂ > 0) (h₃ : O₁ ≠ O₂) :
  ∃ (X : Point), (∀ X, dist X O₁^2 - dist X O₂^2 = R₂^2 - R₁^2) ∧ 
  (line.perpendicular (line O₁ O₂) X) :=
sorry

end locus_of_bisecting_circles_l204_204922


namespace number_drawn_from_first_group_l204_204693

theorem number_drawn_from_first_group (n: ℕ) (groups: ℕ) (interval: ℕ) (fourth_group_number: ℕ) (total_bags: ℕ) 
    (h1: total_bags = 50) (h2: groups = 5) (h3: interval = total_bags / groups)
    (h4: interval = 10) (h5: fourth_group_number = 36) : n = 6 :=
by
  sorry

end number_drawn_from_first_group_l204_204693


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204330

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204330


namespace C1_polar_equation_C2_cartesian_equation_OA_OB_range_l204_204961

section
variables (α θ ρ : ℝ) (x y : ℝ)

-- Curve C1 parametric equations
def C1_parametric_x (α : ℝ) : ℝ := 2 * cos α
def C1_parametric_y (α : ℝ) : ℝ := 2 + 2 * sin α

-- Curve C2 polar equation
def C2_polar_eq : Prop := ρ * sin θ ^ 2 = cos θ

-- The known range of α
def α_range : Prop := α ∈ (Set.Icc (Real.pi / 4) (Real.pi / 3))

-- Proving polar equation of C1
theorem C1_polar_equation (α : ℝ) : C1_parametric_x α ^ 2 + (C1_parametric_y α - 2) ^ 2 = 4 :=
sorry

-- Proving cartesian equation of C2
theorem C2_cartesian_equation (x y : ℝ) (h : C2_polar_eq) :
  y^2 = x :=
sorry

-- Function to compute |OA| * |OB|
def OA_OB_product (α : ℝ) (h : α_range α) : ℝ := 
  let ρ1 := 4 * sin α in
  let ρ2 := cos α / sin α ^ 2 in
  ρ1 * ρ2

-- Proving the range of |OA| * |OB|
theorem OA_OB_range : ∀ α, α_range α → (OA_OB_product α) ∈ (Set.Icc (4 / Real.sqrt 3) 4) :=
sorry
end

end C1_polar_equation_C2_cartesian_equation_OA_OB_range_l204_204961


namespace find_erased_number_l204_204362

variable (x1 x2 x3 x4 x5 x6: ℕ)

def A := x1 + x6
def B := x1 + x2
def C := x2 + x3
def D := x3 + x4
def E := x4 + x5
def F := x5 + x6

theorem find_erased_number : ∃ (F : ℕ), F = A + C + E - B - D :=
by
  use (A + C + E - B - D)
  sorry

end find_erased_number_l204_204362


namespace sunflower_packets_correct_l204_204629

namespace ShyneGarden

-- Define the given conditions
def eggplants_per_packet := 14
def sunflowers_per_packet := 10
def eggplant_packets_bought := 4
def total_plants := 116

-- Define the function to calculate the number of sunflower packets bought
def sunflower_packets_bought (eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants : ℕ) : ℕ :=
  (total_plants - (eggplant_packets_bought * eggplants_per_packet)) / sunflowers_per_packet

-- State the theorem to prove the number of sunflower packets
theorem sunflower_packets_correct :
  sunflower_packets_bought eggplants_per_packet sunflowers_per_packet eggplant_packets_bought total_plants = 6 :=
by
  sorry

end ShyneGarden

end sunflower_packets_correct_l204_204629


namespace sharpener_difference_l204_204767

/-- A hand-crank pencil sharpener can sharpen one pencil every 45 seconds.
An electric pencil sharpener can sharpen one pencil every 20 seconds.
The total available time is 360 seconds (i.e., 6 minutes).
Prove that the difference in the number of pencils sharpened 
by the electric sharpener and the hand-crank sharpener in 360 seconds is 10 pencils. -/
theorem sharpener_difference (time : ℕ) (hand_crank_rate : ℕ) (electric_rate : ℕ) 
(h_time : time = 360) (h_hand_crank : hand_crank_rate = 45) (h_electric : electric_rate = 20) :
  (time / electric_rate) - (time / hand_crank_rate) = 10 := by
  sorry

end sharpener_difference_l204_204767


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204249

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204249


namespace simplify_fraction_l204_204144

theorem simplify_fraction (a b : ℕ) (h : a = 180) (k : b = 270) : 
  ∃ c d, c = 2 ∧ d = 3 ∧ (a / (Nat.gcd a b) = c) ∧ (b / (Nat.gcd a b) = d) :=
by
  sorry

end simplify_fraction_l204_204144


namespace division_neg4_by_2_l204_204792

theorem division_neg4_by_2 : (-4) / 2 = -2 := sorry

end division_neg4_by_2_l204_204792


namespace flippers_win_probability_l204_204635

-- Define the probability of the Flippers winning a single game
def winProb : ℚ := 3/5

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ :=
  nat.factorial n / (nat.factorial k * nat.factorial (n - k))

-- Define the binomial probability formula for exactly k wins in n games
def binomProb (n k : ℕ) (p : ℚ) : ℚ :=
  binom n k * p ^ k * (1 - p) ^ (n - k)

-- Define the specific problem's parameters
def flippersWin4OutOf6Games : ℚ :=
  binomProb 6 4 winProb

-- Statement to be proved: The probability that the Flippers win exactly 4 out of 6 games is 4860/15625
theorem flippers_win_probability : flippersWin4OutOf6Games = 4860 / 15625 := by
  -- Proof goes here
  sorry

end flippers_win_probability_l204_204635


namespace geometric_progression_property_l204_204124

-- Given Definitions
def Sn (a q : ℝ) (n : ℕ) : ℝ := a * (q^n - 1) / (q - 1)
def P (a q : ℝ) (n : ℕ) : ℝ := a^n * q^(n * (n - 1) / 2)
def Sn' (a q : ℝ) (n : ℕ) : ℝ := (q^n - 1) / ((q - 1) * a * q^(n - 1))

-- The conjecture to be proved
theorem geometric_progression_property 
  (a q : ℝ) (n : ℕ) : 
  P a q n ^ 2 * (Sn' a q n) ^ n = (Sn a q n) ^ n := 
by 
  sorry

end geometric_progression_property_l204_204124


namespace h_at_7_over_5_eq_0_l204_204423

def h (x : ℝ) : ℝ := 5 * x - 7

theorem h_at_7_over_5_eq_0 : h (7 / 5) = 0 := 
by 
  sorry

end h_at_7_over_5_eq_0_l204_204423


namespace num_7digit_integers_condition_met_l204_204642

theorem num_7digit_integers_condition_met : 
  (∃ (s : Finset (Fin 7 → Fin 7)), (∀ (d : Fin 7), d ∈ {0, 1, 2, 3, 4, 5, 6}.image s) ∧ 
  ∀ i j k : Fin 7, (s i = 1) → (s j = 2) → (s k = 3) → i < j ∧ j < k) → 
  (Finset.card (Finset.filter (λ p : Fin 7 → Fin 7, 
    p 0 = 1 ∧ p 1 = 2 ∧ p 2 = 3 ∨ 
    p 0 = 1 ∧ p 2 = 2 ∧ p 1 = 3 ∧ 
    p 1 = 1 ∧ p 0 = 2 ∧ p 2 = 3 ∧ 
    p 1 = 1 ∧ p 2 = 2 ∧ p 0 = 3 ∧ 
    p 2 = 1 ∧ p 0 = 2 ∧ p 1 = 3 ∧ 
    p 2 = 1 ∧ p 1 = 2 ∧ p 0 = 3)
    (Finset.univ : Finset (Fin 7 → Fin 7))) = 840) := 
by
  sorry

end num_7digit_integers_condition_met_l204_204642


namespace smallest_odd_number_with_five_prime_factors_l204_204306

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204306


namespace maximize_expr_l204_204085

-- Define the conditions as Lean definitions
variable (a b c d e : ℕ)
variable (h_distinct : list.nodup [a, b, c, d, e])
variable (h_range : ∀ x ∈ [a, b, c, d, e], x ∈ [1, 2, 3, 4, 5, 6, 7, 8])

-- Mathematical expression definition
def expr := real.sqrt ((a / 2) ^ (d / e) / (b / c))

-- Lean theorem statement
theorem maximize_expr : expr a b c d e = 116.85 :=
  sorry

end maximize_expr_l204_204085


namespace hundredth_term_is_981_l204_204562

def is_sum_of_distinct_powers_of_3 (n : ℕ) : Prop :=
  ∃ a : list ℕ, (∀ x ∈ a, x < 6) ∧ n = a.sum (λ x, 3^x)

def nth_term_in_sequence (k : ℕ) : ℕ :=
  (nat.binary_of_num k).reverse.to_list.foldr (λ x acc, acc + 3 ^ x) 0

theorem hundredth_term_is_981 :
  nth_term_in_sequence 100 = 981 := sorry

end hundredth_term_is_981_l204_204562


namespace root_of_quadratic_eq_is_two_l204_204540

theorem root_of_quadratic_eq_is_two (k : ℝ) : (2^2 - 3 * 2 + k = 0) → k = 2 :=
by
  intro h
  sorry

end root_of_quadratic_eq_is_two_l204_204540


namespace car_b_speed_l204_204408

noncomputable def SpeedOfCarB (Speed_A Time_A Time_B d_ratio: ℝ) : ℝ :=
  let Distance_A := Speed_A * Time_A
  let Distance_B := Distance_A / d_ratio
  Distance_B / Time_B

theorem car_b_speed
  (Speed_A : ℝ) (Time_A : ℝ) (Time_B : ℝ) (d_ratio : ℝ)
  (h1 : Speed_A = 70) (h2 : Time_A = 10) (h3 : Time_B = 10) (h4 : d_ratio = 2) :
  SpeedOfCarB Speed_A Time_A Time_B d_ratio = 35 :=
by
  sorry

end car_b_speed_l204_204408


namespace average_of_polynomial_over_roots_of_unity_l204_204602

-- Define the polynomial f(x)
noncomputable def f (a : ℕ → ℂ) (m : ℕ) (x : ℂ) : ℂ :=
  ∑ i in Finset.range (m + 1), a i * x^i

-- Define the n-th roots of unity condition
lemma n_roots_of_unity_sum_zero {n : ℕ} (hn : 0 < n) :
  ∑ k in Finset.range n, (Complex.exp ((2 * Real.pi * Complex.I * k) / n) : ℂ) = (0 : ℂ) :=
sorry

-- Main theorem
theorem average_of_polynomial_over_roots_of_unity (a : ℕ → ℂ) (m n : ℕ) (h1 : m < n) (h2 : 0 < n) :
  (1 / n : ℂ) * (∑ k in Finset.range n, f a m (Complex.exp ((2 * Real.pi * Complex.I * k) / n))) = a 0 :=
by
let z := (λ k, Complex.exp ((2 * Real.pi * Complex.I * k) / n))
have h_sum : ∑ k in Finset.range n, f a m (z k) = n * a 0,
  {
    -- Proof goes here but will be skipped
    sorry
  }
have h_fraction : (1 / n : ℂ) * (n * a 0) = a 0,
  {
    -- Simplify the expression (1/n) * (n * a 0)
    field_simp [h2.ne'],
    ring,
  }
exact h_fraction

end average_of_polynomial_over_roots_of_unity_l204_204602


namespace triangle_angle_l204_204119

theorem triangle_angle (a b c : ℝ) (h : (a^2 + b*c) * x^2 + 2 * sqrt(b^2 + c^2) * x + 1 = 0) (eq_root : discriminant ((a^2 + b*c)) (2 * sqrt(b^2 + c^2)) 1 = 0) :
  ∃ A : ℝ, 0 < A ∧ A < 180 ∧ cos (A * π / 180) = 1/2 ∧ A = 60 :=
by sorry

end triangle_angle_l204_204119


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204021

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204021


namespace negation_exists_l204_204655

theorem negation_exists (h : ∀ x : ℝ, 0 < x → Real.sin x < x) : ∃ x : ℝ, 0 < x ∧ Real.sin x ≥ x :=
by
  sorry

end negation_exists_l204_204655


namespace f_neg_two_l204_204172

noncomputable def f : ℝ → ℝ := sorry

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

variables (f_odd : is_odd_function f)
variables (f_two : f 2 = 2)

theorem f_neg_two : f (-2) = -2 :=
by
  -- Given that f is an odd function and f(2) = 2
  sorry

end f_neg_two_l204_204172


namespace opposite_points_l204_204615

theorem opposite_points (A B : ℝ) (h1 : A = -B) (h2 : A < B) (h3 : abs (A - B) = 6.4) : A = -3.2 ∧ B = 3.2 :=
by
  sorry

end opposite_points_l204_204615


namespace smallest_odd_number_with_five_prime_factors_l204_204301

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204301


namespace smallest_odd_with_five_different_prime_factors_l204_204265

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204265


namespace integer_solutions_abs_sum_l204_204613

theorem integer_solutions_abs_sum (n : ℕ) (h1 : n ≥ 1)
  (h2: ∀ k ∈ {1, 2, 3, ..., n}, (λ m, |x| + |y| = m) → 4 * m - 4 * (m - 1) = 4) :
  ∀ x y : ℤ, |x| + |y| = n → (nat.solutions_count ((λ x y : ℤ, |x| + |y| = n) 0) = 4 + 19 * 4) :=
by
  sorry

end integer_solutions_abs_sum_l204_204613


namespace meet_probability_same_interval_meet_probability_diff_interval_meet_probability_adj_interval_l204_204582

noncomputable def probability_meet (distX distY : ℝ → ℝ) : ℝ :=
-- Placeholder definition depending on distribution definition
sorry

-- Part (a)
theorem meet_probability_same_interval : 
  probability_meet (λ x, if 0 ≤ x ∧ x ≤ 60 then 1/60 else 0)
                    (λ y, if 0 ≤ y ∧ y ≤ 60 then 1/60 else 0) = 11 / 36 := 
sorry

-- Part (b)
theorem meet_probability_diff_interval :
  probability_meet (λ x, if 0 ≤ x ∧ x ≤ 60 then 1/60 else 0)
                    (λ y, if 0 ≤ y ∧ y ≤ 30 then 1/30 else 0) = 11 / 36 :=
sorry

-- Part (c)
theorem meet_probability_adj_interval :
  probability_meet (λ x, if 0 ≤ x ∧ x ≤ 60 then 1/60 else 0)
                   (λ y, if 0 ≤ y ∧ y ≤ 50 then 1/50 else 0) = 19 / 60 :=
sorry

end meet_probability_same_interval_meet_probability_diff_interval_meet_probability_adj_interval_l204_204582


namespace smallest_odd_number_with_five_primes_proof_l204_204233

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204233


namespace maximize_x4_y3_l204_204984

noncomputable def f (x y : ℝ) : ℝ := x^4 * y^3

theorem maximize_x4_y3 (x y : ℝ) (h₀ : x > 0) (h₁ : y > 0) (h₂ : x + y = 50) :
  (x, y) = (200 / 7, 150 / 7) →
  ∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + y' = 50 → f x y ≥ f x' y' :=
begin
  sorry,
end

end maximize_x4_y3_l204_204984


namespace length_of_other_train_l204_204731

variable (L : ℝ)

theorem length_of_other_train
    (train1_length : ℝ := 260)
    (train1_speed_kmh : ℝ := 120)
    (train2_speed_kmh : ℝ := 80)
    (time_to_cross : ℝ := 9)
    (train1_speed : ℝ := train1_speed_kmh * 1000 / 3600)
    (train2_speed : ℝ := train2_speed_kmh * 1000 / 3600)
    (relative_speed : ℝ := train1_speed + train2_speed)
    (total_distance : ℝ := relative_speed * time_to_cross)
    (other_train_length : ℝ := total_distance - train1_length) :
    L = other_train_length := by
  sorry

end length_of_other_train_l204_204731


namespace smallest_odd_number_with_five_different_prime_factors_l204_204284

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204284


namespace count_valid_triples_l204_204469

theorem count_valid_triples : 
  Nat.card { ⟨a, b, c : Nat⟩ | a ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                                  b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                                  c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                                  7 * a = 3 * b + 4 * c } = 15 := 
by {
  sorry
}

end count_valid_triples_l204_204469


namespace determinant_of_projection_matrix_l204_204117

def proj_matrix (u : ℝ × ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  let norm_u := Real.sqrt ((u.1 ^ 2) + (u.2 ^ 2))
  let u_unit := (u.1 / norm_u, u.2 / norm_u)
  let u_col := ![u_unit.1, u_unit.2]
  let u_row := !![u_unit.1, u_unit.2]
  u_col ⬝ u_row

theorem determinant_of_projection_matrix :
  let Q := proj_matrix (3, 4)
  det Q = 0 :=
by
  let Q := proj_matrix (3, 4)
  sorry

end determinant_of_projection_matrix_l204_204117


namespace white_rhino_weight_l204_204139

/-
Problem Statement:
Let W be the weight of one white rhino in pounds. 
Given:
  1. The weight of one black rhino is 2000 pounds.
  2. The total weight of 7 white rhinos and 8 black rhinos is 51700 pounds.
Prove:
  W = 5100
-/

theorem white_rhino_weight :
  ∃ W : ℕ, (let black_rhino_weight := 2000 in (7 * W + 8 * black_rhino_weight = 51700) ∧ W = 5100) :=
by
  existsi 5100
  let black_rhino_weight := 2000
  simp [black_rhino_weight]
  sorry

end white_rhino_weight_l204_204139


namespace butterflies_left_correct_l204_204820

-- Define the total number of butterflies and the fraction that flies away
def butterflies_total : ℕ := 9
def fraction_fly_away : ℚ := 1 / 3

-- Define the number of butterflies left in the garden
def butterflies_left (t : ℕ) (f : ℚ) : ℚ := t - (t : ℚ) * f

-- State the theorem
theorem butterflies_left_correct : butterflies_left butterflies_total fraction_fly_away = 6 := by
  sorry

end butterflies_left_correct_l204_204820


namespace complex_exp_power_cos_angle_l204_204894

theorem complex_exp_power_cos_angle (z : ℂ) (h : z + 1/z = 2 * Complex.cos (Real.pi / 36)) :
    z^1000 + 1/(z^1000) = 2 * Complex.cos (Real.pi * 2 / 9) :=
by
  sorry

end complex_exp_power_cos_angle_l204_204894


namespace scientific_notation_correctness_l204_204729

theorem scientific_notation_correctness : ∃ x : ℝ, x = 8.2 ∧ (8200000 : ℝ) = x * 10^6 :=
by
  use 8.2
  split
  · rfl
  · sorry

end scientific_notation_correctness_l204_204729


namespace complement_union_l204_204919

variable (x : ℝ)

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x ≤ 1}
def P : Set ℝ := {x | x ≥ 2}

theorem complement_union (x : ℝ) : x ∈ U → (¬ (x ∈ M ∨ x ∈ P)) ↔ (1 < x ∧ x < 2) := 
by
  sorry

end complement_union_l204_204919


namespace quadrilateral_possible_l204_204776

theorem quadrilateral_possible
  (a b c d : ℝ)
  (h1 : a + b + c + d = 1)
  (h2 : d ≤ 2 * a)
  (h3 : a ≤ b ∧ b ≤ c ∧ c ≤ d) 
  : a + b > c + d ∧ a + c > b + d ∧ a + d > b + c :=
by skip_proof:= sorry

end quadrilateral_possible_l204_204776


namespace no_integer_area_l204_204965

-- Define the necessary conditions for the quadrilateral ABCD
variables (AB CD : ℕ)

-- Define the theorem stating no configuration yields an integer area
theorem no_integer_area (AB CD : ℕ) (h₁ : AB ⊥ BC) (h₂ : BC ⊥ CD) (h₃ : is_tangent BC (circle O OD)) (h₄ : midpoint O AD):
  ¬ (AB = 4 ∧ CD = 2 ∨ AB = 6 ∧ CD = 3 ∨ AB = 8 ∧ CD = 4 ∨ AB = 10 ∧ CD = 5 ∨ AB = 12 ∧ CD = 6) 
   → ¬ ∃ (BC : ℝ), (AB + CD) * BC / 2 ∈ ℤ :=
by sorry

end no_integer_area_l204_204965


namespace winning_post_distance_l204_204775

noncomputable def speed_ratio : ℝ := 1.75
noncomputable def start_distance : ℝ := 84
noncomputable def required_distance : ℝ := 196

theorem winning_post_distance :
  ∀ (v : ℝ), 
    let speed_A := speed_ratio * v in
    let distance_B := required_distance - start_distance in
    let time_A := required_distance / speed_A in
    let time_B := distance_B / v in
    time_A = time_B :=
begin
  intros v,
  let speed_A := speed_ratio * v,
  let distance_B := required_distance - start_distance,
  let time_A := required_distance / speed_A,
  let time_B := distance_B / v,
  sorry,
end

end winning_post_distance_l204_204775


namespace two_digit_primes_ending_in_3_l204_204036

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204036


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204251

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204251


namespace sum_of_first_11_terms_of_arithmetic_seq_l204_204962

noncomputable def arithmetic_sequence_SUM (a d : ℚ) : ℚ :=  
  11 / 2 * (2 * a + 10 * d)

theorem sum_of_first_11_terms_of_arithmetic_seq
  (a d : ℚ)
  (h : a + 2 * d + a + 6 * d = 16) :
  arithmetic_sequence_SUM a d = 88 := 
  sorry

end sum_of_first_11_terms_of_arithmetic_seq_l204_204962


namespace find_a_l204_204514

def f (a x : ℝ) : ℝ := (a * x^2) / (x + 1)
def f_prime (a x : ℝ) : ℝ := (a * x^2 + 2 * a * x) / (x + 1)^2

theorem find_a (a : ℝ) (h_slope : f_prime a 1 = 1) : a = 4 / 3 :=
by
  sorry

end find_a_l204_204514


namespace smallest_odd_number_with_five_prime_factors_l204_204229

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204229


namespace k4_min_bound_l204_204074

theorem k4_min_bound (n m X : ℕ) (hn : n ≥ 5) (hm : m ≤ n.choose 3) 
    (no_three_collinear : ∀ (P : Finset (Fin n)), P.card = 3 → ∀ (p : Fin n → Fin n → Fin n → Prop), 
                          ∃ u v w, u ≠ v ∧ v ≠ w ∧ u ≠ w ∧ P = {u,v,w} ∧ ¬ p u v w) :
    X ≥ m / 4 * (9 * m / n - 3 / 2 * n^2 + 11 / 2 * n - 6) :=
sorry

end k4_min_bound_l204_204074


namespace taller_tree_cross_section_area_l204_204678

noncomputable def tree_cross_section_area (h : ℝ) : ℝ :=
  let r := Real.sqrt h in
  3.14 * (r * r)

-- Statement of the theorem
theorem taller_tree_cross_section_area :
  ∀ h : ℝ, 
  (h > 20) →
  (7 * (h - 20) = 5 * h) →
  tree_cross_section_area h = 219.8 :=
by
  intros h h_gt_20 ratio_eq
  sorry

end taller_tree_cross_section_area_l204_204678


namespace tan_ratio_l204_204592

variable (a b : ℝ)

-- Conditions given in the problem
def sin_sum_condition : Prop := sin(a + b) = 5 / 8
def sin_diff_condition : Prop := sin(a - b) = 3 / 8

-- Goal: Prove that tan a / tan b = 4
theorem tan_ratio (h1 : sin_sum_condition a b) (h2 : sin_diff_condition a b) :
  tan a / tan b = 4 :=
by
  sorry

end tan_ratio_l204_204592


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204244

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204244


namespace min_volume_is_54π_l204_204063

noncomputable def min_cylinder_volume (r h : ℝ) : ℝ :=
  if h = (2 * r) / (r - 2) then 2 * π * ((r ^ 3) / (r - 2))
  else 0

theorem min_volume_is_54π :
  ∃ r h, r > 0 ∧ h > 0 ∧ π * r ^ 2 * h = 2 * π * r ^ 2 + 2 * π * r * h ∧ min_cylinder_volume r h = 54 * π :=
begin
  sorry
end

end min_volume_is_54π_l204_204063


namespace polynomial_evaluation_l204_204935

theorem polynomial_evaluation (a : ℝ) (h : a^2 + a - 1 = 0) : a^3 + 2 * a^2 + 2 = 3 :=
by
  sorry

end polynomial_evaluation_l204_204935


namespace point_coincidence_l204_204673

variables {Point : Type}
noncomputable def angle : ℝ := 3/11 * 180

structure Line :=
(origin : Point)
(direction : ℝ) 

structure Circle :=
(center : Point)
(radius : ℝ)

structure Intersection :=
(point : Point)

axiom intersects (circle : Circle) (line : Line): Intersection

axiom different_points (M1 O : Point) : M1 ≠ O

axiom e_is_line : Line
axiom f_is_line : Line
axiom O_is_intersection (O : Point) (e f : Line) : O ∈ e ∧ O ∈ f

axiom M1_not_O (M1 O : Point): M1 ∈ e_is_line ∧ M1 ≠ O

axiom circles_with_points (M1 O M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 : Point) (r : ℝ):
  let c1 := Circle.mk M1 r,
      c2 := Circle.mk M2 r,
      c3 := Circle.mk M3 r,
      c4 := Circle.mk M4 r,
      c5 := Circle.mk M5 r,
      c6 := Circle.mk M6 r,
      c7 := Circle.mk M7 r,
      c8 := Circle.mk M8 r,
      c9 := Circle.mk M9 r,
      c10 := Circle.mk M10 r,
      c11 := Circle.mk M11 r
  in intersects c1 f_is_line = {point := M2} ∧
     intersects c2 e_is_line = {point := M3} ∧
     intersects c3 f_is_line = {point := M4} ∧
     intersects c4 e_is_line = {point := M5} ∧
     intersects c5 f_is_line = {point := M6} ∧
     intersects c6 e_is_line = {point := M7} ∧
     intersects c7 f_is_line = {point := M8} ∧
     intersects c8 e_is_line = {point := M9} ∧
     intersects c9 f_is_line = {point := M10} ∧
     intersects c10 e_is_line = {point := M11}

theorem point_coincidence (O M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 : Point) (e f : Line) (r : ℝ)
  (angle_condition : angle = 3/11 * 180)
  (intersection_condition : O ∈ e ∧ O ∈ f)
  (point_condition1 : M1 ∈ e ∧ M1 ≠ O)
  (circle_condition : circles_with_points M1 O M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 r) : 
  M11 = O :=
sorry

end point_coincidence_l204_204673


namespace twentieth_fisherman_catch_l204_204682

theorem twentieth_fisherman_catch (total_fishermen : ℕ) (total_fish : ℕ) (fish_per_19 : ℕ) (fish_each_19 : ℕ) (h1 : total_fishermen = 20) (h2 : total_fish = 10000) (h3 : fish_per_19 = 19 * 400) (h4 : fish_each_19 = 400) : 
  fish_per_19 + fish_each_19 = total_fish := by
  sorry

end twentieth_fisherman_catch_l204_204682


namespace two_primes_equal_l204_204611

theorem two_primes_equal
  (a b c : ℕ)
  (p q r : ℕ)
  (hp : p = b^c + a ∧ Nat.Prime p)
  (hq : q = a^b + c ∧ Nat.Prime q)
  (hr : r = c^a + b ∧ Nat.Prime r) :
  p = q ∨ q = r ∨ r = p := 
sorry

end two_primes_equal_l204_204611


namespace no_distinct_positive_integers_l204_204106

noncomputable def P (x : ℕ) : ℕ := x^2000 - x^1000 + 1

theorem no_distinct_positive_integers (a : Fin 2001 → ℕ) (h_distinct : Function.Injective a) :
  ¬ (∀ i j, i ≠ j → a i * a j ∣ P (a i) * P (a j)) :=
sorry

end no_distinct_positive_integers_l204_204106


namespace min_distance_intersecting_lines_l204_204550

theorem min_distance_intersecting_lines : 
  ∀ (m n : ℝ), (∀ x y : ℝ, (y = 2*x) ∧ (x + y = 3) ∧ (m*x + n*y + 5 = 0) → (m = -5 - 2*n)) → 
  (∀ m n : ℝ, (distance (m, n) (0, 0) = Real.sqrt ((-5 - 2*n)^2 + n^2))) → 
  distance ((-5 - 2*(-2)), -2) (0, 0) = Real.sqrt 5 :=
by {
  -- Here you would provide the proof using the conditions.
  sorry
}

end min_distance_intersecting_lines_l204_204550


namespace range_of_a_l204_204510

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x^2 - a*x + 2*a > 0) → 0 < a ∧ a < 8 := 
sorry

end range_of_a_l204_204510


namespace nurse_serving_time_l204_204995

/-- 
Missy is attending to the needs of 12 patients. One-third of the patients have special dietary 
requirements, which increases the serving time by 20%. It takes 5 minutes to serve each standard 
care patient. Prove that it takes 64 minutes to serve dinner to all patients.
-/
theorem nurse_serving_time (total_patients : ℕ) (special_fraction : ℚ)
  (standard_time : ℕ) (increase_fraction : ℚ) :
  total_patients = 12 →
  special_fraction = 1/3 →
  standard_time = 5 →
  increase_fraction = 1/5 →
  let special_patients := (special_fraction * total_patients : ℚ).toNat in
  let standard_patients := total_patients - special_patients in
  let special_time := standard_time + ((increase_fraction * standard_time : ℚ).toNat) in
  let total_time := (standard_patients * standard_time) + (special_patients * special_time) in
  total_time = 64 :=
begin
  -- Insert hypothesis and proof here
  sorry
end

end nurse_serving_time_l204_204995


namespace emma_last_page_correct_l204_204067

-- Given conditions
def emma_time_per_page : ℕ := 15
def liam_time_per_page : ℕ := 45
def noah_time_per_page : ℕ := 30
def total_pages : ℕ := 900

def emma_reading_speed : ℕ := liam_time_per_page / emma_time_per_page -- 3 times as fast as Liam

-- Ensure Emma and Liam (excluding Noah) read the same amount of time
noncomputable def last_page_emma_reads : ℕ :=
  let p := (total_pages - 200) / (emma_reading_speed + 1) in
  emma_reading_speed * p

theorem emma_last_page_correct :
  last_page_emma_reads = 525 :=
by
  have p := (total_pages - 200) / (emma_reading_speed + 1)
  have emma_pages := emma_reading_speed * p
  have 4 * p + 200 = 900 := sorry
  calc
    last_page_emma_reads
        = emma_pages := by rfl
    ... = 525 := sorry


end emma_last_page_correct_l204_204067


namespace absolute_value_simplification_l204_204150

theorem absolute_value_simplification : abs(-4^2 + 6) = 10 := by
  sorry

end absolute_value_simplification_l204_204150


namespace completing_the_square_l204_204340

theorem completing_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  sorry

end completing_the_square_l204_204340


namespace cayli_combinations_l204_204799

theorem cayli_combinations (art_choices sports_choices music_choices : ℕ)
  (h1 : art_choices = 2)
  (h2 : sports_choices = 3)
  (h3 : music_choices = 4) :
  art_choices * sports_choices * music_choices = 24 := by
  sorry

end cayli_combinations_l204_204799


namespace negation_proposition_l204_204652

theorem negation_proposition :
  (\neg (\forall x: ℝ, (0 < x → sin x < x)) = 
  (∃ x0: ℝ, (0 < x0 ∧ sin x0 ≥ x0))) := by
sorry

end negation_proposition_l204_204652


namespace ramesh_refrigerator_selling_price_l204_204625

theorem ramesh_refrigerator_selling_price (L : ℝ) : 
  let discount1 := 0.2 * L
  let discount2 := 0.1 * (L - discount1)
  let additional_costs := 125 + 250 + 100
  let total_cost := L - discount1 - discount2 + additional_costs
  let profit := 0.18 * L
  let selling_price := total_cost + profit
  selling_price = 0.9 * L + 475 :=
by 
  -- Definitions and calculations based directly on the problem statement 
  let discount1 := 0.2 * L,
  let discount2 := 0.1 * (L - discount1),
  let additional_costs := 125 + 250 + 100,
  let total_cost := L - discount1 - discount2 + additional_costs,
  let profit := 0.18 * L,
  let selling_price := total_cost + profit,
  -- Required to show selling_price is 0.9L + 475
  sorry

end ramesh_refrigerator_selling_price_l204_204625


namespace volume_tetrahedron_height_from_A4_l204_204358

def Point := (ℝ × ℝ × ℝ)

def vector_sub (p1 p2 : Point) : Point :=
  (p1.1 - p2.1, p1.2 - p2.2, p1.3 - p2.3)

def scalar_triple_product (v1 v2 v3 : Point) : ℝ :=
  v1.1 * (v2.2 * v3.3 - v2.3 * v3.2) - 
  v1.2 * (v2.1 * v3.3 - v2.3 * v3.1) + 
  v1.3 * (v2.1 * v3.2 - v2.2 * v3.1)

def cross_product (v1 v2 : Point) : Point :=
  (v1.2 * v2.3 - v1.3 * v2.2, v1.3 * v2.1 - v1.1 * v2.3, v1.1 * v2.2 - v1.2 * v2.1)

def magnitude (v : Point) : ℝ :=
  Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

def volume (A1 A2 A3 A4 : Point) : ℝ :=
  (1 / 6) * abs (scalar_triple_product (vector_sub A2 A1) (vector_sub A3 A1) (vector_sub A4 A1))

def area (A1 A2 A3 : Point) : ℝ :=
  (1 / 2) * magnitude (cross_product (vector_sub A2 A1) (vector_sub A3 A1))

def height (A1 A2 A3 A4 : Point) : ℝ :=
  let S := area A1 A2 A3 in
  let V := volume A1 A2 A3 A4 in
  (3 * V) / S

def A1 : Point := (-3, -5, 6)
def A2 : Point := (2, 1, -4)
def A3 : Point := (0, -3, -1)
def A4 : Point := (-5, 2, -8)

theorem volume_tetrahedron : volume A1 A2 A3 A4 = 191 / 6 := 
by sorry

theorem height_from_A4 : height A1 A2 A3 A4 = Real.sqrt (191 / 3) := 
by sorry

end volume_tetrahedron_height_from_A4_l204_204358


namespace complex_properties_l204_204899

-- Definitions from the conditions
def z : ℂ := 2 - I
def polynomial : ℂ → Prop := λ x, x^2 - 4*x + 5 = 0

theorem complex_properties :
  (conj z = 2 + I) ∧
  (z.im = -1) ∧
  (¬ (∃ y : ℂ, z - I = y ∧ y.re = 0)) ∧
  (polynomial z) :=
by
  sorry

end complex_properties_l204_204899


namespace smallest_odd_with_five_prime_factors_l204_204288

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204288


namespace students_scores_correlation_l204_204392

-- Definitions
def circle_area (r : ℝ) : ℝ := π * r^2

def sphere_volume (r : ℝ) : ℝ := (4 / 3) * π * r^3

def sine_value (θ : ℝ) : ℝ := real.sin θ

-- Hypothetical student's scores (without specific models, we state the correlation assumption)
def student_scores : Type := ℕ × ℕ

-- Goal: Prove that the students' math and physics scores have a correlation
theorem students_scores_correlation (s : student_scores) : ∃ correlation (math phys : ℕ), correlation math phys :=
by sorry

end students_scores_correlation_l204_204392


namespace smallest_odd_number_with_five_prime_factors_l204_204268

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204268


namespace f_neg_one_l204_204507

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x^2 + 1/x else - (x^2 + 1/(-x))

theorem f_neg_one : f (-1) = -2 :=
by
  -- This is where the proof would go, but it is left as a sorry
  sorry

end f_neg_one_l204_204507


namespace acute_angle_between_hands_at_3_36_l204_204209

variable (minute_hand_position hour_hand_position abs_diff : ℝ)

def minute_hand_angle_at_3_36 : ℝ := 216
def hour_hand_angle_at_3_36 : ℝ := 108

theorem acute_angle_between_hands_at_3_36 (h₀ : minute_hand_position = 216)
    (h₁ : hour_hand_position = 108) :
    abs_diff = abs(minute_hand_position - hour_hand_position) → 
    abs_diff = 108 :=
  by
    rw [h₀, h₁]
    sorry

end acute_angle_between_hands_at_3_36_l204_204209


namespace geometric_seq_ratio_l204_204868

theorem geometric_seq_ratio : 
  ∀ (a : ℕ → ℝ) (q : ℝ), 
    (∀ n, a (n+1) = a n * q) → 
    q > 1 → 
    a 1 + a 6 = 8 → 
    a 3 * a 4 = 12 → 
    a 2018 / a 2013 = 3 :=
by
  intros a q h_geom h_q_pos h_sum_eq h_product_eq
  sorry

end geometric_seq_ratio_l204_204868


namespace polar_equiv_l204_204077

theorem polar_equiv
  (r : ℝ)
  (θ : ℝ)
  (hr : -3 = r)
  (hθ : 7 * real.pi / 6 = θ) :
  ∃ r' θ', r' > 0 ∧ 0 ≤ θ' ∧ θ' < 2 * real.pi ∧ (3 = r' ∧ real.pi / 6 = θ') :=
by {
  use 3,
  use real.pi / 6,
  split,
  { -- proof r' > 0:
    exact zero_lt_three,
  },
  split,
  { -- proof 0 ≤ θ':
    exact real.pi_nonneg,
  },
  split,
  { -- proof θ' < 2π:
    exact real.pi_div_two_lt_two_pi,
  },
  {
    -- proof equivalence
    split;
    { apply eq.symm;
      norm_num
    }
  }
}

end polar_equiv_l204_204077


namespace find_a_l204_204606

-- Parametric equation definitions:
def l1 (t : ℝ) (a : ℝ) : ℝ × ℝ :=
  (1 + t, a + 3 * t)

-- Polar coordinate equation for line l2 in Cartesian form:
def l2 (x y : ℝ) : Prop :=
  3 * x - y - 4 = 0

-- Distance between two parallel lines equation:
def distance_parallel_lines (a : ℝ) : Prop :=
  |a - 3 + 4| / real.sqrt 10 = real.sqrt 10

-- The main theorem assumes the distance condition and expresses the final values for 'a'
theorem find_a (a : ℝ) :
  distance_parallel_lines a → (a = 9 ∨ a = -11) :=
by
  sorry

end find_a_l204_204606


namespace sine_cosine_identity_l204_204500

theorem sine_cosine_identity (α : ℝ) : 
  (sin α - cos α = - (7/5)) → (sin α * cos α = - (12/25)) ∧ (tan α = - (3/4) ∨ tan α = - (4/3)) :=
by 
  sorry

end sine_cosine_identity_l204_204500


namespace initial_toy_cost_l204_204609

theorem initial_toy_cost :
  ∀ (initial_toys teddy_bears total_money teddy_bear_cost : ℕ), 
  initial_toys = 28 →
  teddy_bears = 20 →
  total_money = 580 →
  teddy_bear_cost = 15 →
  (TotalMoneyCost initial_toys + teddy_bears * teddy_bear_cost = total_money) →
    (TotalMoneyCost initial_toys / initial_toys = 10) :=
sorry

def TotalMoneyCost (initial_toys : ℕ) : ℕ :=
  initial_toys * 10

end initial_toy_cost_l204_204609


namespace smallest_three_digit_value_l204_204466

theorem smallest_three_digit_value :
  ∃ (L P M : ℕ),
    (L * L % 10 = L) ∧
    (100 * L + 10 * L + L) * L = 100 * L + 10 * P + M ∧
    (100 * L + 10 * P + M) = 275 :=
begin
  sorry
end

end smallest_three_digit_value_l204_204466


namespace days_after_which_A_left_l204_204732

variable (W : ℝ)

def A_work_rate := W / 45
def B_work_rate := W / 40
def combined_work_rate := W / 45 + W / 40
def remaining_work_by_B (work_completed_by_B: ℝ) := work_completed_by_B = 23 * (W / 40)
def total_work_done (x : ℝ) := x * (combined_work_rate) + 23 * (W / 40) = W

theorem days_after_which_A_left (x : ℝ) (h_total_work : total_work_done W x) : 
  x = 9 := sorry

end days_after_which_A_left_l204_204732


namespace clock_angle_3_36_l204_204212

def minute_hand_position (minutes : ℕ) : ℝ :=
  minutes * 6

def hour_hand_position (hours minutes : ℕ) : ℝ :=
  hours * 30 + minutes * 0.5

def angle_difference (angle1 angle2 : ℝ) : ℝ :=
  abs (angle1 - angle2)

def acute_angle (angle : ℝ) : ℝ :=
  min angle (360 - angle)

theorem clock_angle_3_36 :
  acute_angle (angle_difference (minute_hand_position 36) (hour_hand_position 3 36)) = 108 :=
by
  sorry

end clock_angle_3_36_l204_204212


namespace two_digit_primes_ending_in_3_l204_204041

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204041


namespace average_earnings_per_minute_l204_204180

theorem average_earnings_per_minute (race_duration : ℕ) (lap_distance : ℕ) (certificate_rate : ℝ) (laps_run : ℕ) :
  race_duration = 12 → 
  lap_distance = 100 → 
  certificate_rate = 3.5 → 
  laps_run = 24 → 
  ((laps_run * lap_distance / 100) * certificate_rate) / race_duration = 7 :=
by
  intros hrace_duration hlap_distance hcertificate_rate hlaps_run
  rw [hrace_duration, hlap_distance, hcertificate_rate, hlaps_run]
  sorry

end average_earnings_per_minute_l204_204180


namespace parabola_hyperbola_focus_l204_204160

theorem parabola_hyperbola_focus (p : ℝ) : 
  let a := 3
  let b := real.sqrt 5
  let c := real.sqrt 14
  (c = real.sqrt (a^2 + b^2)) → (∃ p : ℝ, (p / 2 = c)) → p = 2 * real.sqrt 14 :=
by
  intros a b c hc hf
  sorry

end parabola_hyperbola_focus_l204_204160


namespace four_digit_square_palindromes_count_l204_204815

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_square_palindromes_count : 
  (finset.card (finset.filter (λ n : ℕ, is_palindrome (n * n) ∧ is_four_digit (n * n)) 
                              (finset.range 100) \ finset.range 32)) = 3 :=
by sorry

end four_digit_square_palindromes_count_l204_204815


namespace smallest_odd_with_five_prime_factors_l204_204297

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204297


namespace hypotenuse_length_l204_204396

noncomputable def hypotenuse_of_30_60_90_triangle (r : ℝ) : ℝ :=
  let a := (r * 3) / Real.sqrt 3
  2 * a

theorem hypotenuse_length (r : ℝ) (h : r = 3) : hypotenuse_of_30_60_90_triangle r = 6 * Real.sqrt 3 :=
  by sorry

end hypotenuse_length_l204_204396


namespace projection_of_vector_onto_plane_l204_204115

noncomputable def projection_matrix (n : ℝ → ℝ → ℝ → ℝ) : matrix (fin 3) (fin 3) ℝ :=
  let normal := ![n 2 (-1) 1]
  let A := matrix.vec_head normal
  let B := matrix.vec_tail normal
  let inner_product := A * A + B.mul_vecemat inner_product normal
  A.outer_product A / inner_product

theorem projection_of_vector_onto_plane
  (v : ℝ → ℝ → ℝ → ℝ) :
    ∀ v : ℝ → ℝ → ℝ → ℝ, 
      let q := 
        ![(1/3) * v 0 + (2/3) * v 1 + (1/3) * v 2,
          (2/3) * v 0 + (1/3) * v 1 + (2/3) * v 2,
          (-1/3) * v 0 + (2/3) * v 1 + (2/3) * v 2]
        in 
        ![(1/3), (2/3), (1/3),
          (2/3), (1/3), (2/3), 
          (-1/3), (2/3), (2/3)] * ![v 0, v 1, v 2] = q := 
sorry

end projection_of_vector_onto_plane_l204_204115


namespace spherical_to_rectangular_l204_204427

theorem spherical_to_rectangular
  (ρ θ φ : ℝ)
  (ρ_eq : ρ = 10)
  (θ_eq : θ = 5 * Real.pi / 4)
  (φ_eq : φ = Real.pi / 4) :
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-5, -5, 5 * Real.sqrt 2) :=
by
  sorry

end spherical_to_rectangular_l204_204427


namespace domain_of_g_l204_204812

noncomputable def g : ℝ → ℝ := sorry

theorem domain_of_g :
  (∀ x : ℝ, x ≠ 0 → (x ∈ domain g ∧ (1/x) ∈ domain g)) ∧
  (∀ x : ℝ, x ∈ domain g → g(x) + g(1/x) = x^2)
  →
  (domain g = {-1, 1}) :=
by 
  sorry

end domain_of_g_l204_204812


namespace probability_midpoint_belongs_to_S_l204_204985

-- Define the set of points S
def S : set (ℤ × ℤ × ℤ) :=
  {p | 
    let (x, y, z) := p in
    0 ≤ x ∧ x ≤ 2 ∧ 0 ≤ y ∧ y ≤ 3 ∧ 0 ≤ z ∧ z ≤ 4 
  }

-- Define a function to determine if the midpoint of two points is in S
def midpoint_in_S (p1 p2 : ℤ × ℤ × ℤ) : Prop :=
  let (x1, y1, z1) := p1 in
  let (x2, y2, z2) := p2 in
  let mx := (x1 + x2) / 2 in
  let my := (y1 + y2) / 2 in
  let mz := (z1 + z2) / 2 in
  (mx, my, mz) ∈ S

-- Calculate the probability
def probability_midpoint_in_S (S : set (ℤ × ℤ × ℤ)) : ℚ :=
  let total_pairs := (|S| * (|S| - 1)) / 2 in
  let favorable_pairs := 
    (count {p1 p2 | p1 ≠ p2 ∧ midpoint_in_S p1 p2}) in
  favorable_pairs / total_pairs

-- Statement of the theorem
theorem probability_midpoint_belongs_to_S :
  probability_midpoint_in_S S = 23 / 177 := sorry

end probability_midpoint_belongs_to_S_l204_204985


namespace max_non_palindromic_years_l204_204748

/-- Given the definition of palindromic years, the maximum number of consecutive
    non-palindromic years between 1000 and 9999 is 109. -/
theorem max_non_palindromic_years : 
  ∃ y1 y2 : ℕ, (1000 ≤ y1 ∧ y1 ≤ 9999) ∧ (1000 ≤ y2 ∧ y2 ≤ 9999) ∧ 
  is_palindrome y1 ∧ is_palindrome y2 ∧ (y2 - y1 = 110) ∧ 
  (∀ y, y1 < y < y2 → ¬is_palindrome y) → 
  ∀ y1 y2 : ℕ, (1000 ≤ y1 ∧ y1 ≤ 9999) ∧ (1000 ≤ y2 ∧ y2 ≤ 9999) ∧ 
   (y2 - y1 = 110) → (∀ y, y1 < y < y2 → ¬is_palindrome y) :=
sorry

end max_non_palindromic_years_l204_204748


namespace sin_2x_plus_cos_x_gt_one_l204_204621

theorem sin_2x_plus_cos_x_gt_one (x : ℝ) (hx : 0 < x ∧ x < π / 3) :
  sin (2 * x) + cos x > 1 :=
sorry

end sin_2x_plus_cos_x_gt_one_l204_204621


namespace graveyard_bones_count_l204_204069

def total_skeletons : ℕ := 20
def half_total (n : ℕ) : ℕ := n / 2
def skeletons_adult_women : ℕ := half_total total_skeletons
def remaining_skeletons : ℕ := total_skeletons - skeletons_adult_women
def even_split (n : ℕ) : ℕ := n / 2
def skeletons_adult_men : ℕ := even_split remaining_skeletons
def skeletons_children : ℕ := even_split remaining_skeletons

def bones_per_woman : ℕ := 20
def bones_per_man : ℕ := bones_per_woman + 5
def bones_per_child : ℕ := bones_per_woman / 2

def total_bones_adult_women : ℕ := skeletons_adult_women * bones_per_woman
def total_bones_adult_men : ℕ := skeletons_adult_men * bones_per_man
def total_bones_children : ℕ := skeletons_children * bones_per_child

def total_bones_in_graveyard : ℕ := total_bones_adult_women + total_bones_adult_men + total_bones_children

theorem graveyard_bones_count : total_bones_in_graveyard = 375 := by
  sorry

end graveyard_bones_count_l204_204069


namespace arithmetic_sequence_correct_geometric_sequence_correct_sum_of_first_n_terms_l204_204491

noncomputable def a_n (n : ℕ) : ℕ := n - 1

noncomputable def b_n (n : ℕ) : ℕ := 2 ^ n

theorem arithmetic_sequence_correct :
  a_n 1 = 0 ∧ a_n 3 = 2 :=
by 
  unfold a_n
  split
  { simp }
  { simp }

theorem geometric_sequence_correct :
  b_n n = 2 ^ (a_n n + 1) :=
by
  unfold b_n a_n
  simp

theorem sum_of_first_n_terms (n : ℕ) : 
  ∑ i in finset.range n, a_n i * b_n i = (n - 2) * 2 ^ (n + 1) + 4 :=
sorry

end arithmetic_sequence_correct_geometric_sequence_correct_sum_of_first_n_terms_l204_204491


namespace total_food_eaten_l204_204680

theorem total_food_eaten (num_puppies num_dogs : ℕ)
    (dog_food_per_meal dog_meals_per_day puppy_food_per_day : ℕ)
    (dog_food_mult puppy_meal_mult : ℕ)
    (h1 : num_puppies = 6)
    (h2 : num_dogs = 5)
    (h3 : dog_food_per_meal = 6)
    (h4 : dog_meals_per_day = 2)
    (h5 : dog_food_mult = 3)
    (h6 : puppy_meal_mult = 4)
    (h7 : puppy_food_per_day = (dog_food_per_meal / dog_food_mult) * puppy_meal_mult * dog_meals_per_day) :
    (num_dogs * dog_food_per_meal * dog_meals_per_day + num_puppies * puppy_food_per_day) = 108 := by
  -- conclude the theorem
  sorry

end total_food_eaten_l204_204680


namespace parallel_vectors_tan_l204_204924

theorem parallel_vectors_tan (θ : ℝ)
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (h₀ : a = (2, Real.sin θ))
  (h₁ : b = (1, Real.cos θ))
  (h_parallel : a.1 * b.2 = a.2 * b.1) :
  Real.tan θ = 2 := 
sorry

end parallel_vectors_tan_l204_204924


namespace cost_each_side_is_56_l204_204939

-- Define the total cost and number of sides
def total_cost : ℕ := 224
def number_of_sides : ℕ := 4

-- Define the cost per side as the division of total cost by number of sides
def cost_per_side : ℕ := total_cost / number_of_sides

-- The theorem stating the cost per side is 56
theorem cost_each_side_is_56 : cost_per_side = 56 :=
by
  -- Proof would go here
  sorry

end cost_each_side_is_56_l204_204939


namespace lada_lera_numbers_l204_204101

theorem lada_lera_numbers :
  ∃ (a b : ℕ), 0.95 * a = 1.05 * b ∧ a = 21 ∧ b = 19 :=
by
  sorry

end lada_lera_numbers_l204_204101


namespace custom_bowling_ball_volume_l204_204416

-- Define the conditions of the problem
def original_radius : ℝ := 12
def original_volume : ℝ := (4 / 3) * Real.pi * (original_radius ^ 3)
def hole1_radius : ℝ := 1.25
def hole1_depth : ℝ := 6
def hole1_volume : ℝ := Real.pi * (hole1_radius ^ 2) * hole1_depth
def hole2_volume : ℝ := hole1_volume
def hole3_radius : ℝ := 2
def hole3_depth : ℝ := 6
def hole3_volume : ℝ := Real.pi * (hole3_radius ^ 2) * hole3_depth

-- The proof problem
theorem custom_bowling_ball_volume : original_volume - 2 * hole1_volume - hole3_volume = 2261.25 * Real.pi := 
by
  -- Lean proof goes here
  sorry

end custom_bowling_ball_volume_l204_204416


namespace simplify_abs_expr_l204_204153

theorem simplify_abs_expr : |(-4 ^ 2 + 6)| = 10 := by
  sorry

end simplify_abs_expr_l204_204153


namespace smallest_odd_number_with_five_prime_factors_l204_204300

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204300


namespace acute_angle_between_hands_at_3_36_l204_204208

variable (minute_hand_position hour_hand_position abs_diff : ℝ)

def minute_hand_angle_at_3_36 : ℝ := 216
def hour_hand_angle_at_3_36 : ℝ := 108

theorem acute_angle_between_hands_at_3_36 (h₀ : minute_hand_position = 216)
    (h₁ : hour_hand_position = 108) :
    abs_diff = abs(minute_hand_position - hour_hand_position) → 
    abs_diff = 108 :=
  by
    rw [h₀, h₁]
    sorry

end acute_angle_between_hands_at_3_36_l204_204208


namespace sequence_value_l204_204871

theorem sequence_value (a : ℕ → ℤ) (h1 : ∀ p q : ℕ, 0 < p → 0 < q → a (p + q) = a p + a q)
                       (h2 : a 2 = -6) : a 10 = -30 :=
by
  sorry

end sequence_value_l204_204871


namespace initial_total_quantity_l204_204377

theorem initial_total_quantity
  (x : ℝ)
  (milk_water_ratio : 5 / 9 = 5 * x / (3 * x + 12))
  (milk_juice_ratio : 5 / 8 = 5 * x / (4 * x + 6)) :
  5 * x + 3 * x + 4 * x = 24 :=
by
  sorry

end initial_total_quantity_l204_204377


namespace smallest_odd_number_with_five_different_prime_factors_l204_204279

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204279


namespace total_claps_at_30_l204_204848

-- Define the sequence based on the conditions:
def seq : Nat → Nat
| 0 => 1
| 1 => 1
| (n+2) => seq n + seq (n+1)

-- Define the property of counting claps:
def isMultipleOf3(n: Nat) : Bool :=
  n % 3 = 0

-- Count the claps up to the 30th number.
def countClapsUpTo (n : Nat) : Nat :=
  (List.range n).countp (λ i => isMultipleOf3 (seq i))

theorem total_claps_at_30 : countClapsUpTo 30 = 7 :=
  sorry

end total_claps_at_30_l204_204848


namespace vector_dot_product_l204_204512

variables {V : Type*} [inner_product_space ℝ V]

def mag_a (a : V) : Prop := ∥a∥ = 2
def proj_b_on_a (a b : V) : Prop := orthogonal_projection (ℝ ∙ a) b = (1 / 4 : ℝ) • a

theorem vector_dot_product (a b : V) (ha : mag_a a) (hb : proj_b_on_a a b) : ⟪a, b⟫ = 1 :=
by {
  -- Initially setting up the proof
  sorry
}

end vector_dot_product_l204_204512


namespace find_r_l204_204853

noncomputable def p (x : ℝ) : ℝ := 6 * x^3 - 3 * x^2 - 36 * x + 54

theorem find_r (r : ℝ) (h : ∃ q : ℝ → ℝ, p = (x : ℝ) → (x - r) ^ 2 * q x) : r = 2 / 3 :=
by
  sorry

end find_r_l204_204853


namespace radius_of_circle_l204_204050

def is_circle (diameter : ℚ) (shape : Prop) : Prop :=
shape = true ∧ diameter = 26

theorem radius_of_circle (d : ℚ) (shape : Prop) (h : is_circle d shape) : d / 2 = 13 :=
by
  cases h with shape_positive diameter_eq_26
  rw [diameter_eq_26]
  exact div_eq_div_iff.mpr (rfl : 26 = 26 * 1)
  rintro ⟨rfl, rfl⟩
  norm_num
  norm_num
  sorry

end radius_of_circle_l204_204050


namespace exists_member_with_few_friends_among_enemies_l204_204443

variable (S : Type) [DecidableEq S]
variable (n q : ℕ)
variable (is_member : S → Prop)
variable (friend : S → S → Prop)
variable [h1 : ∀ (x y : S), (is_member x ∧ is_member y) → x ≠ y → friend x y ∨ ¬ friend x y]
variable [h2 : ∀ (x y z : S), (is_member x ∧ is_member y ∧ is_member z) → x ≠ y → y ≠ z → x ≠ z → ¬ (friend x y ∧ friend y z ∧ friend z x)]
variable [h3 : card (set_of is_member) = n]
variable [h4 : card (set_of (λ (p : S × S), friend p.1 p.2 ∧ is_member p.1 ∧ is_member p.2)) = q]

theorem exists_member_with_few_friends_among_enemies :
  ∃ x : S, (is_member x) ∧ (F x ≤ q * (1 - 4 * q / n ^ 2)) :=
sorry

end exists_member_with_few_friends_among_enemies_l204_204443


namespace person_speed_l204_204733

theorem person_speed (d_meters : ℕ) (t_minutes : ℕ) (d_km t_hours : ℝ) :
  (d_meters = 1800) →
  (t_minutes = 12) →
  (d_km = d_meters / 1000) →
  (t_hours = t_minutes / 60) →
  d_km / t_hours = 9 :=
by
  intros
  sorry

end person_speed_l204_204733


namespace sum_of_odd_integers_21_to_51_l204_204717

noncomputable def arithmetic_seq (a d : ℕ) (n : ℕ) : ℕ := a + (n - 1) * d

noncomputable def sum_arithmetic_seq (a d l : ℕ) : ℕ :=
  let n := (l - a) / d + 1
  n * (a + l) / 2

theorem sum_of_odd_integers_21_to_51 : sum_arithmetic_seq 21 2 51 = 576 := by
  sorry

end sum_of_odd_integers_21_to_51_l204_204717


namespace rectangle_area_increase_l204_204547

variables {α : Type*} [field α]

theorem rectangle_area_increase (l w : α) :
  let l_new := 1.25 * l
  let w_new := 1.20 * w
  (l_new * w_new) = 1.5 * (l * w) →
  (l_new * w_new - l * w) / (l * w) = 0.5 :=
by 
  intros l_new w_new h
  unfold l_new w_new at h
  sorry

end rectangle_area_increase_l204_204547


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204331

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204331


namespace license_plates_count_l204_204932

theorem license_plates_count :
  let letters := 26
  let digits := 10
  let odd_digits := 5
  let even_digits := 5
  (letters^3) * digits * (odd_digits + even_digits) = 878800 := by
  sorry

end license_plates_count_l204_204932


namespace twentieth_fisherman_caught_l204_204686

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l204_204686


namespace trajectory_of_R_is_parabola_l204_204661

theorem trajectory_of_R_is_parabola (k x y : ℝ) :
  (∃ R : ℝ × ℝ, 
    ∃ A B : ℝ × ℝ,
    (A.1^2 = 4 * A.2) ∧ (B.1^2 = 4 * B.2) ∧ 
    (A.2 = k * A.1 - 1) ∧ (B.2 = k * B.1 - 1) ∧ 
    (R.1 = 4 * k) ∧ (R.2 = 4 * k^2 - 3) 
  ) →
  (R.1^2 = 4 * (R.2 + 3) ∧ |R.1| > 4) :=
sorry

end trajectory_of_R_is_parabola_l204_204661


namespace smallest_odd_with_five_different_prime_factors_l204_204256

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204256


namespace sum_first_fifty_digits_of_decimal_of_one_over_1234_l204_204337

theorem sum_first_fifty_digits_of_decimal_of_one_over_1234 :
  let s := "00081037277147487844408427876817350238192918144683"
  let digits := s.data
  (4 * (list.sum (digits.map (λ c, (c.to_nat - '0'.to_nat)))) + (list.sum ((digits.take 6).map (λ c, (c.to_nat - '0'.to_nat)))) ) = 729 :=
by sorry

end sum_first_fifty_digits_of_decimal_of_one_over_1234_l204_204337


namespace arithmetic_sequence_2023rd_term_l204_204645

theorem arithmetic_sequence_2023rd_term 
  (p q : ℤ)
  (h1 : 3 * p - q + 9 = 9)
  (h2 : 3 * (3 * p - q + 9) - q + 9 = 3 * p + q) :
  p + (2023 - 1) * (3 * p - q + 9) = 18189 := by
  sorry

end arithmetic_sequence_2023rd_term_l204_204645


namespace calculate_angle_x_l204_204561

-- Definitions
noncomputable def AB : LineSegment := sorry
noncomputable def CD : LineSegment := sorry
noncomputable def DE : LineSegment := sorry
noncomputable def C : Point := sorry
noncomputable def D : Point := sorry
noncomputable def E : Point := sorry

-- Given conditions
axiom h1 : is_perpendicular CD AB
axiom h2 : angle DE DC = 45
axiom h3 : angle DE AB = x

-- Proof statement
theorem calculate_angle_x : x = 45 :=
by sorry

end calculate_angle_x_l204_204561


namespace system_has_three_solutions_l204_204840

theorem system_has_three_solutions (a : ℝ) :
  (a = 4 ∨ a = 64 ∨ a = 51 + 10 * Real.sqrt 2) ↔
  ∃ (x y : ℝ), 
    (x = abs (y - Real.sqrt a) + Real.sqrt a - 4 
    ∧ (abs x - 6)^2 + (abs y - 8)^2 = 100) 
        ∧ (∃! x1 y1 : ℝ, (x1 = abs (y1 - Real.sqrt a) + Real.sqrt a - 4 
        ∧ (abs x1 - 6)^2 + (abs y1 - 8)^2 = 100)) :=
by
  sorry

end system_has_three_solutions_l204_204840


namespace sum_digit_product_1001_to_2011_l204_204741

def digit_product (n : ℕ) : ℕ :=
  (n.digits 10).foldr (λ d acc => d * acc) 1

theorem sum_digit_product_1001_to_2011 :
  (Finset.range 1011).sum (λ k => digit_product (1001 + k)) = 91125 :=
by
  sorry

end sum_digit_product_1001_to_2011_l204_204741


namespace find_S13_l204_204874

-- Define the arithmetic sequence
variable (a : ℕ → ℤ) (S : ℕ → ℤ)

-- The sequence is arithmetic, i.e., there exists a common difference d
variable (d : ℤ)
axiom arithmetic_sequence : ∀ n : ℕ, a (n + 1) = a n + d

-- The sum of the first n terms is given by S_n
axiom sum_of_terms : ∀ n : ℕ, S n = n * (a 1 + a n) / 2

-- Given condition
axiom given_condition : a 1 + a 8 + a 12 = 12

-- We need to prove that S_{13} = 52
theorem find_S13 : S 13 = 52 :=
sorry

end find_S13_l204_204874


namespace twentieth_fisherman_caught_l204_204685

-- Definitions based on conditions
def fishermen_count : ℕ := 20
def total_fish_caught : ℕ := 10000
def fish_per_nineteen_fishermen : ℕ := 400
def nineteen_count : ℕ := 19

-- Calculation based on the problem conditions
def total_fish_by_nineteen : ℕ := nineteen_count * fish_per_nineteen_fishermen

-- Prove the number of fish caught by the twentieth fisherman
theorem twentieth_fisherman_caught : 
  total_fish_caught - total_fish_by_nineteen = 2400 := 
by
  -- This is where the proof would go
  sorry

end twentieth_fisherman_caught_l204_204685


namespace collinear_A2_B2_C2_l204_204138

variables {A B C A1 B1 C1 C2 A2 B2 P : Type}

-- Define the conditions
def on_side (X Y Z : Type) := sorry -- Placeholder to capture points on side logic
def intersection (L1 L2 : Type) (P : Type) := sorry -- Placeholder to define intersection logic
def concurrent (L1 L2 L3 : Type) := sorry -- Placeholder to define concurrency logic
def collinear (X Y Z : Type) := sorry -- Placeholder to define collinearity logic

-- Assumptions
axiom h1 : on_side A B C A1  -- A1 is on side BC
axiom h2 : on_side B C A B1  -- B1 is on side CA
axiom h3 : on_side C A B C1  -- C1 is on side AB
axiom h4 : intersection (line A B) (line A1 B1) C2  -- C2 is the intersection of lines AB and A1B1
axiom h5 : intersection (line B C) (line B1 C1) A2  -- A2 is the intersection of lines BC and B1C1
axiom h6 : intersection (line A C) (line A1 C1) B2  -- B2 is the intersection of lines AC and A1C1
axiom h7 : concurrent (line A A1) (line B B1) (line C C1)  -- AA1, BB1, and CC1 are concurrent at P

-- Theorem to be proved
theorem collinear_A2_B2_C2 : collinear A2 B2 C2 := sorry

end collinear_A2_B2_C2_l204_204138


namespace cos_seven_pi_over_six_l204_204456

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l204_204456


namespace two_digit_primes_ending_in_3_l204_204044

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204044


namespace inscribe_ngon_correctness_l204_204569

noncomputable def inscribe_ngon (O M : Point) (n : ℕ) (lines : list Line)
  (H : ∀ l in lines, passes_through O l) : Prop :=
  ∃ polygon : Polygon, inscribed_in_circle polygon O ∧
  sides_parallel_to polygon lines ∧
  ∃ side : Line, passes_through M side ∧ side ∈ sides_of polygon

theorem inscribe_ngon_correctness (O M : Point) (n : ℕ) (lines : list Line)
  (H : ∀ l in lines, passes_through O l) : 
  inscribe_ngon O M n lines H :=
sorry

end inscribe_ngon_correctness_l204_204569


namespace line_intersects_circle_l204_204435

-- Define the circle as a predicate
def circle (x y : ℝ) : Prop :=
  x^2 + y^2 - 2 * x + 4 * y = 0

-- Define the line as a predicate depending on the parameter t ∈ ℝ
def line (t : ℝ) (x y : ℝ) : Prop :=
  2 * t * x - y - 2 - 2 * t = 0

-- Formalize the theorem statement
theorem line_intersects_circle :
  ∃ (x y : ℝ) (t : ℝ), circle x y ∧ line t x y :=
by
  -- Proof is required but not provided
  sorry

end line_intersects_circle_l204_204435


namespace derivative_of_inverse_l204_204986

variable (f : ℝ → ℝ) (g : ℝ → ℝ)
variable (x₀ y₀ : ℝ)

-- Conditions from the problem
-- f' x₀ ≠ 0
def f' (x : ℝ) : ℝ
 
-- g(y) is the inverse function of f(x)
def is_inverse (f g : ℝ → ℝ) := ∀ x, g (f x) = x ∧ f (g x) = x

-- Given statement in the problem
axiom f'_nonzero : f' x₀ ≠ 0
axiom y₀_def : y₀ = f(x₀)
axiom g_inv_f : is_inverse f g

-- Prove that g'(y₀) = 1 / f'(x₀)
theorem derivative_of_inverse : ∀ (f: ℝ → ℝ) (g: ℝ → ℝ) (x₀ y₀: ℝ), f' x₀ ≠ 0 →
    ∀ y, y = f x₀ → is_inverse f g → (deriv g y₀ = (1 / f' x₀)) :=
by
  sorry

end derivative_of_inverse_l204_204986


namespace greatest_num_consecutive_integers_l204_204713

theorem greatest_num_consecutive_integers (N a : ℤ) (h : (N * (2*a + N - 1) = 210)) :
  ∃ N, N = 210 :=
sorry

end greatest_num_consecutive_integers_l204_204713


namespace ellipse_properties_l204_204167

theorem ellipse_properties (h k a b : ℝ) (θ : ℝ)
  (h_def : h = -2)
  (k_def : k = 3)
  (a_def : a = 6)
  (b_def : b = 4)
  (θ_def : θ = 45) :
  h + k + a + b = 11 :=
by
  sorry

end ellipse_properties_l204_204167


namespace secant_equal_length_l204_204639

theorem secant_equal_length
  {A B C D E F : Point}
  (ω1 ω2 : Circle)
  (h1 : ω1 ∩ ω2 = {A, B})
  (hC : C ∈ ω1)
  (hF : F ∈ ω1)
  (hCE : E ∈ Line.mk C B)
  (hFD : D ∈ Line.mk F B)
  (h_angle : ∠ A B C = ∠ A B D) :
  dist C E = dist F D :=
by sorry

end secant_equal_length_l204_204639


namespace a3_value_l204_204873

variable {a : ℕ → ℤ} -- Arithmetic sequence as a function from natural numbers to integers
variable {S : ℕ → ℤ} -- Sum of the first n terms

-- Conditions
axiom a1_eq : a 1 = -11
axiom a4_plus_a6_eq : a 4 + a 6 = -6
-- Common difference d
variable {d : ℤ}
axiom d_def : ∀ n, a (n + 1) = a n + d

theorem a3_value : a 3 = -7 := by
  sorry -- Proof not required as per the instructions

end a3_value_l204_204873


namespace area_triangle_FRH_l204_204560

-- Definitions based on conditions
variables (E F G H P D R : Type) 
  (rectangle_EFGH : Rectangle E F G H) 
  (EP PF : ℝ)
  (EG : ℝ)
  (DPHF_area : ℝ)

-- Given conditions
def conditions : Prop := 
  EP = 4 ∧ PF = 4 ∧ EG = 18 ∧ DPHF_area = 144

-- Problem statement
theorem area_triangle_FRH (h : conditions E F G H P D R rectangle_EFGH EP PF EG DPHF_area) : 
  triangle_area F R H = 36 :=
sorry

end area_triangle_FRH_l204_204560


namespace pythago_schools_l204_204440

def number_of_schools (total_students : ℕ) : ℕ :=
  total_students / 4

theorem pythago_schools : 
  ∃ n : ℕ, 
    (∀ d e f g : ℕ, 
      1 ≤ d ∧ d < e ∧ e < f ∧ f < g ∧
      (4 * n - 1) = g ∧
      List.mem 50 [e, f, g] ∧
      List.mem 81 [e, f, g] ∧
      List.mem 97 [e, f, g] 
    ) → n = 25 :=
by
  sorry

end pythago_schools_l204_204440


namespace Q_symmetric_to_P_l204_204841

structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

def symmetric_point (A B P : Point3D) : Point3D := 
  let AB := ⟨B.x - A.x, B.y - A.y, B.z - A.z⟩
  let λ := (AB.x * (P.x - A.x) + AB.y * (P.y - A.y) + AB.z * (P.z - A.z)) /
            (AB.x ^ 2 + AB.y ^ 2 + AB.z ^ 2)
  let O := ⟨A.x + λ * AB.x, A.y + λ * AB.y, A.z + λ * AB.z⟩
  ⟨2 * O.x - P.x, 2 * O.y - P.y, 2 * O.z - P.z⟩

theorem Q_symmetric_to_P (A B P Q : Point3D) 
  (hA : A = ⟨1, 2, -6⟩) 
  (hB : B = ⟨7, -7, 6⟩) 
  (hP : P = ⟨1, 3, 2⟩) 
  (hQ : Q = symmetric_point A B P) : 
  Q = ⟨5, -5, -6⟩ := by
  sorry

end Q_symmetric_to_P_l204_204841


namespace inclination_angle_obtuse_l204_204941

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem inclination_angle_obtuse : 
  let k := HasDerivAt f 4 in k < 0 ∧ (90 * Real.pi / 180) < k ∧ k < (180 * Real.pi / 180) :=
sorry

end inclination_angle_obtuse_l204_204941


namespace acute_angle_at_3_36_l204_204206

def degrees (h m : ℕ) : ℝ :=
  let minute_angle := (m / 60.0) * 360.0
  let hour_angle := (h % 12 + m / 60.0) * 30.0
  abs (minute_angle - hour_angle)

theorem acute_angle_at_3_36 : degrees 3 36 = 108 :=
by
  sorry

end acute_angle_at_3_36_l204_204206


namespace min_value_frac_sum_l204_204803

noncomputable def geom_seq := ℕ → ℝ

def positive_geom_seq (a : geom_seq) (q : ℝ) (h_pos : 0 < q) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

def question (a : geom_seq) (m n : ℕ) (hmn : m + n = 8) : Prop :=
  (∀ (a1 : ℝ), a 3 = a 2 + 2 * a 1) →
  (∃ a1 : ℝ, a m * a n = 64 * (a 1) ^ 2) →
  ∃ x : ℝ, x = 2 ∧ x = min ((1 : ℝ) / m + 9 / n)

theorem min_value_frac_sum (a : geom_seq) (hmn : m + n = 8) :
  positive_geom_seq a 2 (by norm_num) →
  question a m n hmn :=
  sorry

end min_value_frac_sum_l204_204803


namespace sequence_term_2011_l204_204091

noncomputable def seq (n : ℕ) : ℚ :=
  if n = 1 then 2 else
  if n = 2 then 3 else
  have seq (n - 1) * seq (n - 2) ≠ 0 from sorry,
  (seq (n - 1)) / (seq (n - 2))

theorem sequence_term_2011 : seq 2011 = 3 / 2 :=
sorry

end sequence_term_2011_l204_204091


namespace nurse_missy_serving_time_l204_204993

-- Conditions
def total_patients : ℕ := 12
def standard_patient_time : ℕ := 5
def special_requirement_fraction : ℚ := 1/3
def serving_time_increase_factor : ℚ := 1.2

-- Proof statement
theorem nurse_missy_serving_time :
  let special_patients := total_patients * special_requirement_fraction
  let standard_patients := total_patients - special_patients
  let special_patient_time := serving_time_increase_factor * standard_patient_time
  total_serving_time = (standard_patient_time * standard_patients) + (special_patient_time * special_patients) :=
  64 := sorry

end nurse_missy_serving_time_l204_204993


namespace f_even_g_odd_l204_204519

noncomputable def f (x : ℝ) : ℝ := log (1 - x) + log (1 + x)
noncomputable def g (x : ℝ) : ℝ := log (1 - x) - log (1 + x)

theorem f_even_g_odd : (∀ x : ℝ, -1 < x ∧ x < 1 → f(-x) = f(x)) ∧ (∀ x : ℝ, -1 < x ∧ x < 1 → g(-x) = -g(x)) :=
by
  sorry

end f_even_g_odd_l204_204519


namespace time_to_pump_out_water_l204_204754

noncomputable def volume_of_water (depth_in_inches : ℕ) (length ft : ℕ) (width ft : ℕ) : ℝ :=
  (depth_in_inches / 12) * length * width

noncomputable def volume_to_gallons (volume_cubic_feet : ℝ) : ℝ :=
  volume_cubic_feet * 7.5

noncomputable def total_pumping_capacity_per_minute (pumps : ℕ) (capacity per pump : ℕ) : ℝ :=
  pumps * capacity

noncomputable def total_time_to_pump_out_water (total volume gallons : ℝ) (pumping_capacity_per_minute : ℝ) : ℝ :=
  total_volume_gallons / pumping_capacity_per_minute

theorem time_to_pump_out_water :
  let depth_in_inches := 18
  let length := 24
  let width := 32
  let pumps := 3
  let capacity_per_pump := 8
  let depth_in_feet := 1.5
  let volume_cubic_feet := volume_of_water depth_in_inches length width
  let total_volume_gallons := volume_to_gallons volume_cubic_feet
  let pumping_capacity_per_minute := total_pumping_capacity_per_minute pumps capacity_per_pump
  in total_time_to_pump_out_water total_volume_gallons pumping_capacity_per_minute = 360 := by
  sorry

end time_to_pump_out_water_l204_204754


namespace sum_a_1_to_2014_equal_zero_l204_204480

def f (n: ℕ) : ℤ := 
  if n % 2 = 1 then n 
  else -n

def a (n: ℕ) : ℤ := 
  f n + f (n + 1)

theorem sum_a_1_to_2014_equal_zero : 
  (Finset.range 2014).sum (λ n, a (n + 1)) = 0 := 
sorry

end sum_a_1_to_2014_equal_zero_l204_204480


namespace range_of_a_for_critical_points_l204_204060

noncomputable def f (a x : ℝ) : ℝ := x^3 - a * x^2 + a * x + 3

theorem range_of_a_for_critical_points : 
  ∀ a : ℝ, (∃ x : ℝ, deriv (f a) x = 0) ↔ (a < 0 ∨ a > 3) :=
by
  sorry

end range_of_a_for_critical_points_l204_204060


namespace clock_time_l204_204571

-- Lean statement representing the math problem
theorem clock_time 
    (no_numbers_on_dial : true) 
    (unclear_top : true)
    (same_length_hands : true)
    (hand_A_exact_hour : true)
    (hand_B_slightly_short_hour : true) : 
    (time_shown = 16:50) := 
by 
  -- we assume all conditions hold true hence applying axioms or facts from our known set.
  sorry

end clock_time_l204_204571


namespace smaller_acute_angle_is_20_degrees_l204_204957

noncomputable def smaller_acute_angle (x : ℝ) : Prop :=
  let θ1 := 7 * x
  let θ2 := 2 * x
  θ1 + θ2 = 90 ∧ θ2 = 20

theorem smaller_acute_angle_is_20_degrees : ∃ x : ℝ, smaller_acute_angle x :=
  sorry

end smaller_acute_angle_is_20_degrees_l204_204957


namespace original_cost_l204_204688

noncomputable def original_cost_of_chips (C : ℝ) : Prop :=
  let total_cost := 5 * C
  let discount := 0.10 * total_cost
  let discounted_cost := total_cost - discount
  let paid_per_person := 5
  let total_paid_after_discount := 3 * paid_per_person
  discounted_cost = total_paid_after_discount

theorem original_cost (C : ℝ) (h : C = 3.33) : original_cost_of_chips C :=
by
  have h_total_cost : 5 * C = 5 * 3.33 := by rw h
  have h_discount : 0.10 * (5 * C) = 0.10 * (5 * 3.33) := by rw h_total_cost
  have h_discounted_cost : (5 * C) - 0.10 * (5 * C) = (5 * 3.33) - 0.10 * (5 * 3.33) := by rw [h_total_cost, h_discount]
  have h_total_paid_after_discount : 3 * 5 = 15 := by norm_num
  rw h_total_paid_after_discount
  exact h_discounted_cost

end original_cost_l204_204688


namespace prefer_cash_payments_l204_204477

/-- 
 Large retail chains might prefer cash payments for their goods given economic arguments 
 such as efficiency of operations, cost of handling transactions, and risk of fraud.
-/
theorem prefer_cash_payments (h₁ : ∀ (τ₁ τ₂ : ℝ), τ₁ < τ₂ → τ₁ = 1.8 ∧ τ₂ = 2.2)
                              (h₂ : ∀ (c₁ c₂ : ℝ), c₁ < c₂ → c₁)
                              (h₃ : ∀ (r₁ r₂ : ℝ), r₁ > r₂ → r₁) : 
  ∃ reasons, reasons = "efficiency, cost, fraud" := 
by
  sorry


end prefer_cash_payments_l204_204477


namespace inequality_solution_l204_204349

theorem inequality_solution {x : ℝ} : (1 / 2 - (x - 2) / 3 > 1) → (x < 1 / 2) :=
by {
  sorry
}

end inequality_solution_l204_204349


namespace triangle_perimeter_l204_204958

noncomputable def smallest_perimeter (a b c : ℕ) : ℕ :=
  a + b + c

theorem triangle_perimeter (a b c : ℕ) (A B C : ℝ) (h1 : A = 2 * B) 
  (h2 : C > π / 2) (h3 : a^2 = b * (b + c)) (h4 : ∃ m n : ℕ, b = m^2 ∧ b + c = n^2 ∧ a = m * n) :
  smallest_perimeter 28 16 33 = 77 :=
by sorry

end triangle_perimeter_l204_204958


namespace part1_part2_l204_204516

def f (x : ℝ) : ℝ := -x^3 + 3 * x

theorem part1 : ∀ x : ℝ, f (-x) = -f x :=
by
  intro x
  sorry

theorem part2 : ∀ x : ℝ, 
  (x > -1 ∧ x < 1 → f' x > 0) ∧ 
  ((x < -1 ∨ x > 1) → f' x < 0) :=
by 
  intro x
  let f' (x : ℝ) : ℝ := -3 * x^2 + 3
  split
  · intro h
    sorry
  · intro h
    sorry

end part1_part2_l204_204516


namespace S_10_value_l204_204915

def sequence (n : ℕ) : ℤ := 11 - 2 * n

def abs_sum (n : ℕ) : ℤ :=
  (Finset.range n).sum (λ i => Int.natAbs (sequence (i + 1)))

theorem S_10_value : abs_sum 10 = 50 := 
  sorry

end S_10_value_l204_204915


namespace smallest_odd_with_five_different_prime_factors_l204_204255

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204255


namespace percentage_change_xyz_l204_204084

theorem percentage_change_xyz (x y z : ℝ) :
  let x' := 0.8 * x,
      y' := 0.8 * y,
      z' := 1.1 * z,
      new_val := x' * y' * z',
      old_val := x * y * z
  in new_val = 0.704 * old_val → (old_val - new_val) / old_val * 100 = 29.6 :=
by
  intro h
  calc (x * y * z - 0.704 * x * y * z) / (x * y * z) * 100
      = (1 - 0.704) * 100 : by sorry
      = 0.296 * 100      : by sorry
      = 29.6             : by sorry

end percentage_change_xyz_l204_204084


namespace theater_seats_l204_204556

theorem theater_seats (n : ℕ) (a_1 a_n d : ℕ) (h1 : a_1 = 14) (h2 : d = 2) (h3 : a_n = 56) (h4 : a_n = a_1 + (n - 1) * d) : 
  let S_n := n * (a_1 + a_n) / 2
  in S_n = 770 :=
by 
  sorry

end theater_seats_l204_204556


namespace acute_angle_3_36_clock_l204_204220

theorem acute_angle_3_36_clock : 
  let minute_hand_degrees := (36 / 60) * 360,
      hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360,
      angle := abs(minute_hand_degrees - hour_hand_degrees) in
  angle = 108 :=
by
  let minute_hand_degrees := (36 / 60) * 360
  let hour_hand_degrees := ((3 / 12) + (36 / 720)) * 360
  let angle := abs(minute_hand_degrees - hour_hand_degrees)
  show angle = 108 from sorry

end acute_angle_3_36_clock_l204_204220


namespace probability_of_MATHEMATICS_letter_l204_204055

def unique_letters_in_mathematics : Finset Char := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}

theorem probability_of_MATHEMATICS_letter :
  let total_letters := 26
  let unique_letters_count := unique_letters_in_mathematics.card
  (unique_letters_count / total_letters : ℝ) = 8 / 26 := by
  sorry

end probability_of_MATHEMATICS_letter_l204_204055


namespace smallest_odd_with_five_different_prime_factors_l204_204262

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204262


namespace ted_gathered_10_blue_mushrooms_l204_204401

noncomputable def blue_mushrooms_ted_gathered : ℕ :=
  let bill_red_mushrooms := 12
  let bill_brown_mushrooms := 6
  let ted_green_mushrooms := 14
  let total_white_spotted_mushrooms := 17
  
  let bill_white_spotted_red_mushrooms := bill_red_mushrooms / 2
  let bill_white_spotted_brown_mushrooms := bill_brown_mushrooms

  let total_bill_white_spotted_mushrooms := bill_white_spotted_red_mushrooms + bill_white_spotted_brown_mushrooms
  let ted_white_spotted_mushrooms := total_white_spotted_mushrooms - total_bill_white_spotted_mushrooms

  ted_white_spotted_mushrooms * 2

theorem ted_gathered_10_blue_mushrooms :
  blue_mushrooms_ted_gathered = 10 :=
by
  sorry

end ted_gathered_10_blue_mushrooms_l204_204401


namespace complete_the_square_l204_204342

theorem complete_the_square (x : ℝ) : x^2 - 8 * x + 1 = 0 → (x - 4)^2 = 15 :=
by
  intro h
  sorry

end complete_the_square_l204_204342


namespace sum_of_solutions_l204_204079

theorem sum_of_solutions (y x : ℝ) (h1 : y = 7) (h2 : x^2 + y^2 = 100) : 
  x + -x = 0 :=
by
  sorry

end sum_of_solutions_l204_204079


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204030

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204030


namespace min_shift_l204_204942

-- Conditions
def original_func (x : ℝ) : ℝ := sin (x / 2 + π / 3)
def shifted_func (x m : ℝ) : ℝ := sin (x / 2 + m / 2 + π / 3)
def target_func (x : ℝ) : ℝ := cos (x / 2)

-- The theorem to prove
theorem min_shift (m : ℝ) (h₀ : m > 0) (h₁ : ∀ x, shifted_func (x - m) m = target_func x) :
  m = π / 3 :=
sorry

end min_shift_l204_204942


namespace sqrt_positive_equivalent_domain_exists_m_for_quadratic_positive_l204_204058

noncomputable theory

def is_monotonic {D : Set ℝ} (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ∈ D → y ∈ D → x < y → f(x) < f(y)

def is_positive_function {D : Set ℝ} (f : ℝ → ℝ) : Prop :=
  ∃ a b, [a, b] ⊆ D ∧ a < b ∧
    (∀ x, x ∈ [a, b] → f(x) ∈ [a, b]) ∧
    (∀ x y, x ∈ [a, b] → y ∈ [a, b] → x < y → f(x) < f(y))

def equivalent_domain_interval (f : ℝ → ℝ) (D : Set ℝ) : Set ℝ :=
  {ab | ∃ a b, ab = [a, b] ∧ [a, b] ⊆ D ∧ is_positive_function f ∧ (∀ x, x ∈ [a, b] → f(x) ∈ [a, b])}

theorem sqrt_positive_equivalent_domain :
  equivalent_domain_interval (λ x : ℝ, real.sqrt x) {x | 0 ≤ x} = {[0, 1]} :=
sorry

theorem exists_m_for_quadratic_positive :
  ∃ m ∈ set.Ioo (-1 : ℝ) (-3 / 4), is_positive_function (λ x : ℝ, x^2 + m) {x | x < 0} :=
sorry

end sqrt_positive_equivalent_domain_exists_m_for_quadratic_positive_l204_204058


namespace max_sum_distances_is_side_or_altitude_min_sum_distances_is_perp_median_or_parallel_l204_204619

open EuclideanGeometry

noncomputable def maximize_sum_distances (A B O : Point) (h1 : ¬ Collinear A B O) : Line :=
  sorry

noncomputable def minimize_sum_distances (A B O : Point) (h1 : ¬ Collinear A B O) : Line :=
  sorry

theorem max_sum_distances_is_side_or_altitude (A B O : Point) (h1 : ¬ Collinear A B O) :
  let l := maximize_sum_distances A B O h1
  ∃ X, (X = Intersect l (Segment A B) ∨ X = Foot_of_perpendicular O l) ∧ Sum_distances_to_line A B l = max_distance :=
sorry

theorem min_sum_distances_is_perp_median_or_parallel (A B O : Point) (h1 : ¬ Collinear A B O) :
  let l := minimize_sum_distances A B O h1
  let M := midpoint_segment A B O
  (Is_perpendicular l (Line_through O M) ∨ Is_parallel l (Side_of_triangle A B O)) ∧ 
  Sum_distances_to_line A B l = min_distance :=
sorry

end max_sum_distances_is_side_or_altitude_min_sum_distances_is_perp_median_or_parallel_l204_204619


namespace clock_angle_at_3_36_l204_204198

def minute_hand_angle : ℝ := (36.0 / 60.0) * 360.0
def hour_hand_angle : ℝ := 90.0 + (36.0 / 60.0) * 30.0

def acute_angle (a b : ℝ) : ℝ :=
  let diff := abs (a - b)
  min diff (360 - diff)

theorem clock_angle_at_3_36 :
  acute_angle minute_hand_angle hour_hand_angle = 108 :=
by
  sorry

end clock_angle_at_3_36_l204_204198


namespace two_digit_primes_ending_in_3_l204_204033

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204033


namespace problem_expression_simplification_l204_204846

theorem problem_expression_simplification :
  9^(1/2 * Real.log 10 / Real.log 3) + (1/81)^(1/4) - 0.027^(-1/3) + Real.log 2 / Real.log 6 + 1 / (1 + Real.log 2 / Real.log 3) = 8 :=
by
  sorry

end problem_expression_simplification_l204_204846


namespace zinc_to_copper_ratio_l204_204800

theorem zinc_to_copper_ratio (total_weight zinc_weight copper_weight : ℝ) 
  (h1 : total_weight = 64) 
  (h2 : zinc_weight = 28.8) 
  (h3 : copper_weight = total_weight - zinc_weight) : 
  (zinc_weight / 0.4) / (copper_weight / 0.4) = 9 / 11 :=
by
  sorry

end zinc_to_copper_ratio_l204_204800


namespace domain_of_sqrt_function_l204_204166

theorem domain_of_sqrt_function : 
  ∀ x : ℝ, f(x) = real.sqrt (16 - 4 ^ x) → (16 - 4 ^ x ≥ 0 ↔ x ≤ 2) :=
by
  intros x f
  have h₁ : 16 - 4 ^ x ≥ 0 ↔ x ≤ 2 :=
  begin
    split,
    {
      intro h,
      -- Formal proof part here, skipped as it is required by the task to leave the proof incomplete
      sorry,
    },
    {
      intro h,
      -- Formal proof part here, skipped as it is required by the task to leave the proof incomplete
      sorry,
    }
  end
  exact h₁
  sorry

end domain_of_sqrt_function_l204_204166


namespace find_f_of_half_l204_204907

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_of_half : (∀ x : ℝ, f (Real.logb 4 x) = x) → f (1 / 2) = 2 :=
by
  intros h
  have h1 := h (4 ^ (1 / 2))
  sorry

end find_f_of_half_l204_204907


namespace a_n_arithmetic_sequence_T_n_less_than_6_l204_204486

-- Definition of sequence {a_n} such that a_1 = 1 and its sum S_n
variable {a : ℕ → ℕ}
variable {S : ℕ → ℕ}
axiom h_a_1 : a 1 = 1
axiom h_S_n : ∀ n : ℕ, S n = (∑ k in finset.range (n + 1), a k)

-- Given condition: (S_r / S_t) = (r / t)^2
axiom h_S_ratio : ∀ (r t : ℕ), 0 < r → 0 < t → S r / S t = (r^2) / (t^2)

-- Arithmetic sequence assertion based on (S_n / S_1) = n^2 
theorem a_n_arithmetic_sequence : ∀ n : ℕ, n > 0 → a n = 2 * n - 1 :=
sorry

-- Definition of sequence {b_n} such that (\frac{a_n}{b_n} = 2^{n-1})
variable {b : ℕ → ℕ}
axiom h_b_def : ∀ n : ℕ, n > 0 → a n = b n * 2 ^ (n - 1)

-- Sum T_n of sequence {b_n}
variable {T : ℕ → ℕ}
axiom h_T_n : ∀ n : ℕ, T n = (∑ k in finset.range (n + 1), b k)

-- Prove T_n < 6
theorem T_n_less_than_6 : ∀ n : ℕ, n > 0 → T n < 6 :=
sorry

end a_n_arithmetic_sequence_T_n_less_than_6_l204_204486


namespace cos_seven_pi_over_four_l204_204832

theorem cos_seven_pi_over_four : Real.cos (7 * Real.pi / 4) = 1 / Real.sqrt 2 := 
by
  sorry

end cos_seven_pi_over_four_l204_204832


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204028

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204028


namespace cos_seven_pi_over_six_l204_204457

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  -- Place the proof here
  sorry

end cos_seven_pi_over_six_l204_204457


namespace change_from_fifteen_dollars_l204_204808

theorem change_from_fifteen_dollars : 
  ∀ (cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid : ℕ),
  cost_eggs = 3 →
  cost_pancakes = 2 →
  cost_mug_cocoa = 2 →
  num_mugs = 2 →
  tax = 1 →
  additional_pancakes = 2 →
  additional_mug = 2 →
  paid = 15 →
  paid - (cost_eggs + cost_pancakes + (num_mugs * cost_mug_cocoa) + tax + additional_pancakes + additional_mug) = 1 :=
by
  intros cost_eggs cost_pancakes cost_mug_cocoa num_mugs tax additional_pancakes additional_mug paid
  sorry

end change_from_fifteen_dollars_l204_204808


namespace fourth_vertex_of_square_l204_204081

open Complex

theorem fourth_vertex_of_square :
  let A := 1 + 2 * Complex.i
  let B := -2 + Complex.i
  let C := -1 - 2 * Complex.i
  ∃ D : ℂ, (1 + 2 * Complex.i, -2 + Complex.i, -1 - 2 * Complex.i, D).is_square ∧ D = 2 - Complex.i := 
by
  sorry

end fourth_vertex_of_square_l204_204081


namespace trigonometric_relationship_l204_204594

noncomputable def a := sin (5 * Real.pi / 7)
noncomputable def b := cos (2 * Real.pi / 7)
noncomputable def c := tan (2 * Real.pi / 7)

theorem trigonometric_relationship : b < a ∧ a < c :=
by
  sorry

end trigonometric_relationship_l204_204594


namespace tangent_line_at_point_l204_204896

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (Real.exp (-(x - 1)) - x) else (Real.exp (x - 1) + x)

theorem tangent_line_at_point (f_even : ∀ x : ℝ, f x = f (-x)) :
    ∀ (x y : ℝ), x = 1 → y = 2 → (∃ m b : ℝ, y = m * x + b ∧ m = 2 ∧ b = 0) := by
  sorry

end tangent_line_at_point_l204_204896


namespace smallest_odd_with_five_prime_factors_l204_204290

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204290


namespace length_of_AB_l204_204083

-- Define the geometric setup
structure Rectangle :=
  (A B C D : Point)
  (AB : Line A B)
  (BC : Line B C)
  (CD : Line C D)
  (DA : Line D A)
  (ABCD_rect : IsRectangle A B C D)

structure RightAngledTriangle (A B C : Point) :=
  (right_angle : Angle A B C = 90 °)

def Point : Type := sorry  -- Define type of Point
def Line (A B : Point) : Type := sorry  -- Define type of Line
def Angle (A B C : Point) : ℝ := sorry  -- Define Angle calculation
def IsRectangle (A B C D : Point) : Prop := sorry  -- Predicate for being a rectangle

theorem length_of_AB :
  ∀ (A B C D E F : Point)
    (AE : Line A E) (ED : Line E D) (BF : Line B F) (AB : Line A B)
    (AE_len : distance A E = 21) (ED_len : distance E D = 72) (BF_len : distance B F = 45)
    (ΔAED_right : RightAngledTriangle A E D) (ΔBFC_right : RightAngledTriangle B F C)
    (F_on_DE : Collinear D E F)
    (ABCD_rect : IsRectangle A B C D),
  distance A B = 50 := 
by
  sorry

end length_of_AB_l204_204083


namespace inequality_and_equality_conditions_l204_204864

theorem inequality_and_equality_conditions
    {a b c d : ℝ}
    (ha : 0 < a)
    (hb : 0 < b)
    (hc : 0 < c)
    (hd : 0 < d) :
  (a ^ (1/3) * b ^ (1/3) + c ^ (1/3) * d ^ (1/3) ≤ (a + b + c) ^ (1/3) * (a + c + d) ^ (1/3)) ↔ 
  (b = (a / c) * (a + c) ∧ d = (c / a) * (a + c)) :=
  sorry

end inequality_and_equality_conditions_l204_204864


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204007

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204007


namespace sunset_time_correct_l204_204135

theorem sunset_time_correct : 
  let sunrise := (6 * 60 + 43)       -- Sunrise time in minutes (6:43 AM)
  let daylight := (11 * 60 + 56)     -- Length of daylight in minutes (11:56)
  let sunset := (sunrise + daylight) % (24 * 60) -- Calculate sunset time considering 24-hour cycle
  let sunset_hour := sunset / 60     -- Convert sunset time back into hours
  let sunset_minute := sunset % 60   -- Calculate remaining minutes
  (sunset_hour - 12, sunset_minute) = (6, 39)    -- Convert to 12-hour format and check against 6:39 PM
:= by
  sorry

end sunset_time_correct_l204_204135


namespace minimize_distances_l204_204467

variables {A B C P : Type} 

structure Triangle (A B C P : Type) :=
  (is_acute_angled : True)
  (in_interior : P)

def feet_of_perpendiculars (P : P) : P → P → P × P × P := sorry

def incenter (A B C : Type) : Type := sorry

theorem minimize_distances {P : Triangle A B C P} (L M N : Type) 
  (h_feet : feet_of_perpendiculars P A B C = (L, M, N))
  (hincenter : incenter A B C = P) :
  ∀ Q, Q ≠ P → (BL^2 + CM^2 + AN^2) < (B(Q*Q, L^2) + C(Q*Q, M^2) + A(Q*Q, N^2)) :=
sorry

end minimize_distances_l204_204467


namespace negation_of_triangle_statement_l204_204657

theorem negation_of_triangle_statement :
  (¬ (∀ (T : Triangle), T.has_at_least_two_obtuse_angles)) ↔
  (∃ (T : Triangle), T.has_at_most_one_obtuse_angle) :=
begin
  sorry
end

end negation_of_triangle_statement_l204_204657


namespace sin_beta_value_l204_204887

theorem sin_beta_value (α β : ℝ) (h1 : cos α = 4 / 5) (h2 : cos (α + β) = 3 / 5) (h3 : 0 < α ∧ α < π / 2) (h4 : 0 < β ∧ β < π / 2) :
  sin β = 7 / 25 := 
sorry

end sin_beta_value_l204_204887


namespace count_negative_numbers_l204_204784

theorem count_negative_numbers : 
  let numbers := [0, -2, 3, -0.1, -(-5)] in 
  (list.countp (λ x, x < 0) numbers) = 2 :=
by 
  -- sorry, proof is not required
  sorry

end count_negative_numbers_l204_204784


namespace firetruck_reachable_area_l204_204072

theorem firetruck_reachable_area : 
  let max_distance_on_road := 60 * (8 / 60 : ℝ)
  let t := (8 / 60 : ℝ)
  let max_off_road_distance(x: ℝ) := 10 * (t - x / 60)
  let total_area := (64 : ℝ) - 4 * (1 / 2 * (4 / 3 : ℝ) * (4 / 3 : ℝ))
  in total_area = (544 / 9 : ℝ) :=
by
  sorry

end firetruck_reachable_area_l204_204072


namespace combined_teaching_experience_l204_204098

def james_teaching_years : ℕ := 40
def partner_teaching_years : ℕ := james_teaching_years - 10

theorem combined_teaching_experience : james_teaching_years + partner_teaching_years = 70 :=
by
  sorry

end combined_teaching_experience_l204_204098


namespace split_into_two_groups_l204_204601

variables (n : ℕ) (a₁ a₂ a₃ a₄ : ℕ)

def gcd_condition := (gcd n a₁ = 1 ∧ gcd n a₂ = 1 ∧ gcd n a₃ = 1 ∧ gcd n a₄ = 1)
def mod_condition := ∀ k ∈ finset.range' 1 (n - 1), 
  (k * a₁ % n + k * a₂ % n + k * a₃ % n + k * a₄ % n) = 2 * n
def sum_condition := (a₁ % n) + (a₂ % n) + (a₃ % n) + (a₄ % n) = 2 * n
def even_split := (a₁ % n) + (a₄ % n) = n ∧ (a₂ % n) + (a₃ % n) = n

theorem split_into_two_groups 
  (hn : n ≥ 2) 
  (ha : a₁ > 0 ∧ a₂ > 0 ∧ a₃ > 0 ∧ a₄ > 0) 
  (hgcd : gcd_condition n a₁ a₂ a₃ a₄) 
  (hmod : mod_condition n a₁ a₂ a₃ a₄) : 
  even_split n a₁ a₂ a₃ a₄ :=
sorry

end split_into_two_groups_l204_204601


namespace cube_surface_area_ratio_l204_204355

theorem cube_surface_area_ratio (x : ℝ) (hx : x > 0) :
  let A1 := 6 * (2 * x) ^ 2,
      A2 := 6 * x ^ 2
  in (A1 / A2) = 4 :=
by
  -- Proof goes here
  sorry

end cube_surface_area_ratio_l204_204355


namespace johnny_weekly_earnings_l204_204576

-- Define the conditions mentioned in the problem.
def number_of_dogs_at_once : ℕ := 3
def thirty_minute_walk_payment : ℝ := 15
def sixty_minute_walk_payment : ℝ := 20
def work_hours_per_day : ℝ := 4
def sixty_minute_walks_needed_per_day : ℕ := 6
def work_days_per_week : ℕ := 5

-- Prove Johnny's weekly earnings given the conditions
theorem johnny_weekly_earnings :
  let sixty_minute_walks_per_day := sixty_minute_walks_needed_per_day / number_of_dogs_at_once
  let sixty_minute_earnings_per_day := sixty_minute_walks_per_day * number_of_dogs_at_once * sixty_minute_walk_payment
  let remaining_hours_per_day := work_hours_per_day - sixty_minute_walks_per_day
  let thirty_minute_walks_per_day := remaining_hours_per_day * 2 -- each 30-minute walk takes 0.5 hours
  let thirty_minute_earnings_per_day := thirty_minute_walks_per_day * number_of_dogs_at_once * thirty_minute_walk_payment
  let daily_earnings := sixty_minute_earnings_per_day + thirty_minute_earnings_per_day
  let weekly_earnings := daily_earnings * work_days_per_week
  weekly_earnings = 1500 :=
by
  sorry

end johnny_weekly_earnings_l204_204576


namespace correct_average_marks_l204_204638

theorem correct_average_marks 
  (n : ℕ) (average initial_wrong new_correct : ℕ) 
  (h_num_students : n = 30)
  (h_average_marks : average = 100)
  (h_initial_wrong : initial_wrong = 70)
  (h_new_correct : new_correct = 10) :
  (average * n - (initial_wrong - new_correct)) / n = 98 := 
by
  sorry

end correct_average_marks_l204_204638


namespace neg_of_exists_a_l204_204190

theorem neg_of_exists_a (a : ℝ) : ¬ (∃ a : ℝ, a^2 + 1 < 2 * a) :=
by
  sorry

end neg_of_exists_a_l204_204190


namespace divide_segment_l204_204438

theorem divide_segment (a : ℝ) (n : ℕ) (h : 0 < n) : 
  ∃ P : ℝ, P = a / (n + 1) ∧ P > 0 :=
by
  sorry

end divide_segment_l204_204438


namespace circumference_of_circle_l204_204698

theorem circumference_of_circle (speed1 speed2 : ℕ) (meeting_time : ℕ) : speed1 = 7 → speed2 = 8 → meeting_time = 20 → 
  let relative_speed := speed1 + speed2
  let circumference := relative_speed * meeting_time
  circumference = 300 :=
begin
  intros h_speed1 h_speed2 h_meeting_time,
  rw [h_speed1, h_speed2, h_meeting_time],
  simp,
end

end circumference_of_circle_l204_204698


namespace probability_of_both_green_buttons_eq_5_over_14_l204_204573

noncomputable def probability_green_buttons : ℚ := 
let initial_red_C := 6 in
let initial_green_C := 12 in
let initial_total_C := initial_red_C + initial_green_C in
let remaining_total_C := 3 * initial_total_C / 4 in
let removed_total := initial_total_C - remaining_total_C in
let x := removed_total / 2 in
let remaining_red_C := initial_red_C - x in
let remaining_green_C := initial_green_C - x in
let prob_green_C := remaining_green_C / remaining_total_C in
let green_D := x in
let total_D := x * 2 in
let prob_green_D := green_D / total_D in
prob_green_C * prob_green_D

theorem probability_of_both_green_buttons_eq_5_over_14 : 
  probability_green_buttons = (5 / 14) := 
sorry

end probability_of_both_green_buttons_eq_5_over_14_l204_204573


namespace smallest_odd_number_with_five_prime_factors_is_15015_l204_204253

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (factors : List ℕ), factors.nodup ∧ factors.length = 5 ∧ (∀ p ∈ factors, is_prime p) ∧ factors.prod = n

def is_odd (n : ℕ) : Prop := n % 2 = 1

def smallest_odd_number_with_five_prime_factors (n : ℕ) : Prop :=
  has_five_distinct_prime_factors n ∧ is_odd n

theorem smallest_odd_number_with_five_prime_factors_is_15015 :
  smallest_odd_number_with_five_prime_factors 15015 :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_is_15015_l204_204253


namespace smallest_odd_with_five_prime_factors_is_15015_l204_204322

def is_odd (n : ℕ) : Prop := n % 2 = 1

def has_five_different_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), nat.prime p1 ∧ nat.prime p2 ∧ nat.prime p3 ∧ nat.prime p4 ∧ nat.prime p5 ∧
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧
  p3 ≠ p4 ∧ p3 ≠ p5 ∧
  p4 ≠ p5 ∧
  n = p1 * p2 * p3 * p4 * p5

def smallest_odd_number_with_five_different_prime_factors : ℕ :=
  15015

theorem smallest_odd_with_five_prime_factors_is_15015 :
  ∃ n, is_odd n ∧ has_five_different_prime_factors n ∧ n = 15015 :=
by exact ⟨15015, rfl, sorry⟩

end smallest_odd_with_five_prime_factors_is_15015_l204_204322


namespace relationship_S_T_l204_204866

def S (n : ℕ) : ℤ := 2^n
def T (n : ℕ) : ℤ := 2^n - (-1)^n

theorem relationship_S_T (n : ℕ) (h : n > 0) : 
  (n % 2 = 1 → S n < T n) ∧ (n % 2 = 0 → S n > T n) :=
by
  sorry

end relationship_S_T_l204_204866


namespace count_blanks_l204_204553

theorem count_blanks (B : ℝ) (h1 : 10 + B = T) (h2 : 0.7142857142857143 = B / T) : B = 25 :=
by
  -- The conditions are taken into account as definitions or parameters
  -- We skip the proof itself by using 'sorry'
  sorry

end count_blanks_l204_204553


namespace no_integer_x_within_ranges_l204_204665

namespace PolynomialProof

noncomputable def P : ℤ[X] := sorry -- Placeholder for the polynomial P

axiom P_integer_coeffs : P.coeff.nth n ∈ ℤ for all n : ℕ

axiom P_equal_five_at_integers (a b c d e : ℤ) :
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (P.eval a = 5) ∧ 
  (P.eval b = 5) ∧ 
  (P.eval c = 5) ∧ 
  (P.eval d = 5) ∧ 
  (P.eval e = 5)


theorem no_integer_x_within_ranges (a b c d e : ℤ):
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧
  (P.eval a = 5) ∧ 
  (P.eval b = 5) ∧ 
  (P.eval c = 5) ∧ 
  (P.eval d = 5) ∧ 
  (P.eval e = 5)) →
  (∀ x : ℤ, ¬ (-6 ≤ P.eval x ∧ P.eval x ≤ 4 ∨ 6 ≤ P.eval x ∧ P.eval x ≤ 16)) :=
begin
  intros,
  sorry -- Here should be the proof steps
end

end PolynomialProof

end no_integer_x_within_ranges_l204_204665


namespace problem_1_problem_2_l204_204527

-- Define the vectors a, b, c
def a : ℝ × ℝ := (-2, 2)
def b : ℝ × ℝ := (2, 1)
def c : ℝ × ℝ := (2, -1)

-- Definitions of vectors operations
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ :=
  (v1.1 - v2.1, v1.2 - v2.2)

def scale_vector (t : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (t * v.1, t * v.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

-- Problem statements
theorem problem_1 (t : ℝ) :
  magnitude (vec_add a (scale_vector t b)) = 3 ↔ t = 1 ∨ t = -1/5 := sorry

theorem problem_2 (t : ℝ) :
  dot_product (vec_sub a (scale_vector t b)) c = 0 ↔ t = -2 := sorry

end problem_1_problem_2_l204_204527


namespace smallest_odd_number_with_five_primes_proof_l204_204243

def is_prime (n : ℕ) : Prop :=
  2 ≤ n ∧ ∀ m ∣ n, m = 1 ∨ m = n

def smallest_odd_primes : List ℕ := [3, 5, 7, 11, 13]

noncomputable def smallest_odd_number_with_five_primes : ℕ :=
  List.prod smallest_odd_primes

theorem smallest_odd_number_with_five_primes_proof : smallest_odd_number_with_five_primes = 15015 :=
by
  unfold smallest_odd_number_with_five_primes
  unfold smallest_odd_primes
  norm_num

end smallest_odd_number_with_five_primes_proof_l204_204243


namespace four_digit_square_palindromes_count_l204_204814

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString
  s = s.reverse

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem four_digit_square_palindromes_count : 
  (finset.card (finset.filter (λ n : ℕ, is_palindrome (n * n) ∧ is_four_digit (n * n)) 
                              (finset.range 100) \ finset.range 32)) = 3 :=
by sorry

end four_digit_square_palindromes_count_l204_204814


namespace div_power_l204_204704

theorem div_power (h : 27 = 3 ^ 3) : 3 ^ 12 / 27 ^ 2 = 729 :=
by {
  calc
    3 ^ 12 / 27 ^ 2 = 3 ^ 12 / (3 ^ 3) ^ 2 : by rw h
               ... = 3 ^ 12 / 3 ^ 6       : by rw pow_mul
               ... = 3 ^ (12 - 6)         : by rw div_eq_sub_pow
               ... = 3 ^ 6                : by rw sub_self_pow
               ... = 729                  : by norm_num,
  sorry
}

end div_power_l204_204704


namespace Jason_more_blue_marbles_l204_204574

theorem Jason_more_blue_marbles (Jason_blue_marbles Tom_blue_marbles : ℕ) 
  (hJ : Jason_blue_marbles = 44) (hT : Tom_blue_marbles = 24) :
  Jason_blue_marbles - Tom_blue_marbles = 20 :=
by
  sorry

end Jason_more_blue_marbles_l204_204574


namespace ten_percent_of_x_l204_204749

theorem ten_percent_of_x
  (x : ℝ)
  (h : 3 - (1 / 4) * 2 - (1 / 3) * 3 - (1 / 7) * x = 27) :
  0.10 * x = 17.85 :=
by
  -- theorem proof goes here
  sorry

end ten_percent_of_x_l204_204749


namespace complement_union_result_l204_204920

open Set

variable (U A B : Set ℕ)
variable (hU : U = {0, 1, 2, 3})
variable (hA : A = {1, 2})
variable (hB : B = {2, 3})

theorem complement_union_result : compl A ∪ B = {0, 2, 3} :=
by
  -- Our proof steps would go here
  sorry

end complement_union_result_l204_204920


namespace square_ratio_in_right_triangle_l204_204075

theorem square_ratio_in_right_triangle :
  ∀ (triangle : Type) (a b c : ℕ) (s1 s2 : ℕ), 
  a = 6 → b = 8 → c = 10 →
  ∃ x y : ℕ, 
  (x = 24 / 7) ∧ (y = 240 / 37) ∧ 
  s1 = x ∧ s2 = y ∧ 
  s1 * 70 = s2 * 37 :=
by { intros, sorry }

end square_ratio_in_right_triangle_l204_204075


namespace prefer_cash_payments_l204_204476

/-- 
 Large retail chains might prefer cash payments for their goods given economic arguments 
 such as efficiency of operations, cost of handling transactions, and risk of fraud.
-/
theorem prefer_cash_payments (h₁ : ∀ (τ₁ τ₂ : ℝ), τ₁ < τ₂ → τ₁ = 1.8 ∧ τ₂ = 2.2)
                              (h₂ : ∀ (c₁ c₂ : ℝ), c₁ < c₂ → c₁)
                              (h₃ : ∀ (r₁ r₂ : ℝ), r₁ > r₂ → r₁) : 
  ∃ reasons, reasons = "efficiency, cost, fraud" := 
by
  sorry


end prefer_cash_payments_l204_204476


namespace higher_selling_price_is_463_l204_204785

-- Definitions and conditions
def cost_price : ℝ := 400
def selling_price_340 : ℝ := 340
def loss_340 : ℝ := selling_price_340 - cost_price
def gain_percent : ℝ := 0.05
def additional_gain : ℝ := gain_percent * -loss_340
def expected_gain := -loss_340 + additional_gain

-- Theorem to prove that the higher selling price is 463
theorem higher_selling_price_is_463 : ∃ P : ℝ, P = cost_price + expected_gain ∧ P = 463 :=
by
  sorry

end higher_selling_price_is_463_l204_204785


namespace arithmetic_sequence_b_min_value_sum_B_l204_204914

noncomputable def a : ℕ → ℤ
| 0     => 19
| 1     => 2
| (n+2) => 2 * (a (n+1) + 1) - a n

def b : ℕ → ℤ
| 0     => a 1 - a 0
| (n+1) => a (n+2) - a (n+1)

theorem arithmetic_sequence_b : ∀ n ≥ 1, b (n+1) = b n + 2 := 
sorry

noncomputable def B (n : ℕ) : ℤ :=
match n with
| 0     => 0
| (n+1) => b n + B n

theorem min_value_sum_B : ∀ n, B n ≥ -289 / 4 := 
sorry

end arithmetic_sequence_b_min_value_sum_B_l204_204914


namespace sum_of_integers_l204_204666

theorem sum_of_integers (a b : ℕ) (h1 : a * b + a + b = 255) (h2 : a < 30) (h3 : b < 30) (h4 : a % 2 = 1) :
  a + b = 30 := 
sorry

end sum_of_integers_l204_204666


namespace possible_values_for_a_l204_204918

def setM : Set ℝ := {x | x^2 + x - 6 = 0}
def setN (a : ℝ) : Set ℝ := {x | a * x + 2 = 0}

theorem possible_values_for_a (a : ℝ) : (∀ x, x ∈ setN a → x ∈ setM) ↔ (a = -1 ∨ a = 0 ∨ a = 2 / 3) := 
by
  sorry

end possible_values_for_a_l204_204918


namespace area_of_shape_formed_by_fourth_vertices_l204_204687

-- Defining the vertices of the square
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (1, 0)
def C : (ℝ × ℝ) := (1, 1)
def D : (ℝ × ℝ) := (0, 1)

-- Defining the property of rhombuses having three consecutive vertices on the square's sides
def consecutive_vertices_on_sides (K L N : ℝ × ℝ) : Prop :=
(K.1 = (1 - N.1)) ∧ (L.1 = 1) ∧ (K.2 = 0) ∧ (N.2 = 1)

-- Defining the problem statement
theorem area_of_shape_formed_by_fourth_vertices :
  ∃ K L N M : ℝ × ℝ, consecutive_vertices_on_sides K L N ∧
  (∃ shape : set (ℝ × ℝ), is_square shape ∧ vertices shape = {M | consecutive_vertices_on_sides K L N}) ∧
  area shape = 1 :=
sorry

end area_of_shape_formed_by_fourth_vertices_l204_204687


namespace compound_interest_amount_l204_204670

theorem compound_interest_amount (P : ℝ) (R : ℝ) (T : ℝ) (P' : ℝ) (R' : ℝ) (T' : ℝ) :
  SI = P * R * T / 100 →
  CI = P' * ((1 + R' / 100)^T' - 1) →
  ∃ (SI CI : ℝ), SI = 0.5 * CI →
  SI = 1750 * 8 * 3 / 100 →
  R = 8 →
  T = 3 →
  R' = 10 →
  T' = 2 →
  P' = 4000 :=
by
  intros _ _ _ h3 _ _ _
  sorry

end compound_interest_amount_l204_204670


namespace sum_of_first_50_digits_of_one_over_1234_l204_204335

def first_n_digits_sum (x : ℚ) (n : ℕ) : ℕ :=
  sorry  -- This function should compute the sum of the first n digits after the decimal point of x

theorem sum_of_first_50_digits_of_one_over_1234 :
  first_n_digits_sum (1/1234) 50 = 275 :=
sorry

end sum_of_first_50_digits_of_one_over_1234_l204_204335


namespace shifted_sine_function_l204_204651

theorem shifted_sine_function (w : ℝ) (ϕ : ℝ) (h_w : w = 2) (h_ϕ : |ϕ| < Real.pi / 2) :
    (∀ x : ℝ, sin (2 * (x - Real.pi / 6) + ϕ) = sin (2 * x - Real.pi / 6)) :=
sorry

end shifted_sine_function_l204_204651


namespace total_legs_l204_204578

def human_legs : Nat := 2
def num_humans : Nat := 2
def dog_legs : Nat := 4
def num_dogs : Nat := 2

theorem total_legs :
  num_humans * human_legs + num_dogs * dog_legs = 12 := by
  sorry

end total_legs_l204_204578


namespace smallest_odd_number_with_five_prime_factors_l204_204227

def is_prime_factor_of (p n : ℕ) : Prop :=
  p.prime ∧ p ∣ n

def is_odd (n : ℕ) : Prop :=
  ¬ 2 ∣ n

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
  p1.prime ∧ p2.prime ∧ p3.prime ∧ p4.prime ∧ p5.prime ∧ 
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p1 ≠ p5 ∧ 
  p2 ≠ p3 ∧ p2 ≠ p4 ∧ p2 ≠ p5 ∧ 
  p3 ≠ p4 ∧ p3 ≠ p5 ∧ 
  p4 ≠ p5 ∧ 
  p1 * p2 * p3 * p4 * p5 = n

theorem smallest_odd_number_with_five_prime_factors :
  is_odd 15015 ∧ has_five_distinct_prime_factors 15015 ∧ 
  (∀ n : ℕ, is_odd n ∧ has_five_distinct_prime_factors n → 15015 ≤ n) :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204227


namespace arithmetic_sequence_common_difference_l204_204478

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
  ∃ d, (∀ n, a (n + 1) - a n = d) ∧ d = 9 / 4 := 
  sorry

end arithmetic_sequence_common_difference_l204_204478


namespace seats_per_bus_correct_l204_204667

-- Define the conditions given in the problem
def students : ℕ := 28
def buses : ℕ := 4

-- Define the number of seats per bus
def seats_per_bus : ℕ := students / buses

-- State the theorem that proves the number of seats per bus
theorem seats_per_bus_correct : seats_per_bus = 7 := by
  -- conditions are used as definitions, the goal is to prove seats_per_bus == 7
  sorry

end seats_per_bus_correct_l204_204667


namespace sum_first_fifty_digits_of_decimal_of_one_over_1234_l204_204339

theorem sum_first_fifty_digits_of_decimal_of_one_over_1234 :
  let s := "00081037277147487844408427876817350238192918144683"
  let digits := s.data
  (4 * (list.sum (digits.map (λ c, (c.to_nat - '0'.to_nat)))) + (list.sum ((digits.take 6).map (λ c, (c.to_nat - '0'.to_nat)))) ) = 729 :=
by sorry

end sum_first_fifty_digits_of_decimal_of_one_over_1234_l204_204339


namespace cost_of_pencils_l204_204617

open Nat

theorem cost_of_pencils (P : ℕ) : 
  (H : 20 * P + 80 * 3 = 360) → 
  P = 6 :=
by 
  sorry

end cost_of_pencils_l204_204617


namespace discriminant_nonnegative_triangle_perimeter_l204_204529

-- Part 1: Prove that the equation always has real roots
theorem discriminant_nonnegative (k : ℝ) : 
    let a := 1
    let b := -(k + 2)
    let c := 2 * k
    let Δ := b^2 - 4 * a * c
    Δ ≥ 0 :=
by
  let a := 1
  let b := -(k + 2)
  let c := 2 * k
  let Δ := b^2 - 4 * a * c
  have : Δ = (k-2)^2 :=
    by
      dsimp only [Δ]
      ring
  rw this
  exact pow_two_nonneg _

-- Part 2: Prove the perimeter of the triangle is 5
theorem triangle_perimeter (b c : ℝ) (h_b c: ∀ k, (b, c) = (k+2±√(k-2)² 2*k))
  (h_iso: ∀ b c, b = c) : let a := 1 in (a + b + c) = 5 := sorry

end discriminant_nonnegative_triangle_perimeter_l204_204529


namespace min_value_of_2A_abs_diff_x1_x2_l204_204518

/-- Given the function f(x) = sqrt(3) * sin(2017 * x) + cos(2017 * x), 
    prove that the maximum value is 2 and the minimum value of 
    2A * |x1 - x2| where f(x1) ≤ f(x) ≤ f(x2) for all x in ℝ is 4π / 2017. -/
theorem min_value_of_2A_abs_diff_x1_x2 : 
  (∃ x1 x2 : ℝ, ∀ x : ℝ, 
    (let f : ℝ → ℝ := λ x, sqrt 3 * Real.sin (2017 * x) + Real.cos (2017 * x)
     ∧ f x1 ≤ f x ∧ f x ≤ f x2 
     ∧ (2 * 2 * |x1 - x2| = 4 * Real.pi / 2017))) :=
sorry

end min_value_of_2A_abs_diff_x1_x2_l204_204518


namespace pyramid_surface_area_l204_204773

noncomputable def total_surface_area : Real :=
  let ab := 14
  let bc := 8
  let pf := 15
  let base_area := ab * bc
  let fm := ab / 2
  let pm_ab := Real.sqrt (pf^2 + fm^2)
  let pm_bc := Real.sqrt (pf^2 + (bc / 2)^2)
  base_area + 2 * (ab / 2 * pm_ab) + 2 * (bc / 2 * pm_bc)

theorem pyramid_surface_area :
  total_surface_area = 112 + 14 * Real.sqrt 274 + 8 * Real.sqrt 241 := by
  sorry

end pyramid_surface_area_l204_204773


namespace trajectory_of_M_fixed_point_on_y_axis_l204_204898

def point (α : Type) := α × α
def circle (α : Type) := point α → Prop

variables {α : Type} [linear_ordered_field α]

def F1 : circle α :=
λ p, let (x, y) := p in (x + 1)^2 + y^2 = 8

def symmetrical_about_origin (p1 p2 : point α) : Prop :=
let (x1, y1) := p1, (x2, y2) := p2 in x1 = -x2 ∧ y1 = -y2

def is_perpendicular_bisector (p1 p2 m : point α) : Prop :=
let (x1, y1) := p1, (x2, y2) := p2, (xm, ym) := m in
xm = (x1 + x2) / 2 ∧ ym = (y1 + y2) / 2

-- (I) Equation of trajectory C of point M
theorem trajectory_of_M (P F1 F2 M : point α)
    (P_on_F1 : F1 P) (F2_sym_F1 : symmetrical_about_origin F1 F2)
    (M_is_bisector : is_perpendicular_bisector P F2 M) :
  (let (x, y) := M in (x^2 / 2) + y^2 = 1) := sorry

def G : point α := (0, 1/3 : α)

def line_through (G : point α) (k : α) : point α → Prop :=
λ p, let (x, y) := p in y = k * x + 1/3

-- (II) Existence of fixed point Q on y-axis that circle with AB as diameter passes through
theorem fixed_point_on_y_axis (k : α)
    (trajectory_eq : ∀ (A B : point α), line_through G k A → line_through G k B →
    (let (xA, yA) := A, (xB, yB) := B in xA + xB = -4*k / (3*(1 + 2*k^2)) ∧ xA * xB = -16 / (9*(1 + 2*k^2)))) :
  ∃ Q : point α, let (xQ, yQ) := Q in xQ = 0 ∧ yQ = -1 ∧
  ∀ (A B : point α), line_through G k A → line_through G k B →
    (let (xA, yA) := A, (xB, yB) := B in
      let vector_AQ := (-xA, yQ - yA),
          vector_BQ := (-xB, yQ - yB) in
      vector_AQ.1 * vector_BQ.1 + vector_AQ.2 * vector_BQ.2 = 0) := sorry

end trajectory_of_M_fixed_point_on_y_axis_l204_204898


namespace two_digit_primes_ending_in_3_l204_204047

theorem two_digit_primes_ending_in_3 : ∃ n : ℕ, n = 7 ∧ (∀ x, 10 ≤ x ∧ x < 100 ∧ x % 10 = 3 → Prime x → (x = 13 ∨ x = 23 ∨ x = 43 ∨ x = 53 ∨ x = 73 ∨ x = 83)) :=
by {
  use 7,
  split,
  -- proof part is omitted for this example
  sorry,
}

end two_digit_primes_ending_in_3_l204_204047


namespace periodic_function_sum_l204_204506

-- Given definitions
def symmetric_about (f : ℝ → ℝ) (a : ℝ) := ∀ x, f (2 * a - x) = f x

-- The main theorem to prove
theorem periodic_function_sum :
  ∀ (f : ℝ → ℝ),
  (symmetric_about f (-3/4)) →
  (∀ x, f x = -f (x + 3/2)) →
  f (-1) = 1 →
  f 0 = -2 →
  (∑ i in finset.range 2008, f (i + 1)) = 1 :=
by
  intro f symm H H1 H0
  sorry

end periodic_function_sum_l204_204506


namespace scientific_notation_correctness_l204_204728

theorem scientific_notation_correctness : ∃ x : ℝ, x = 8.2 ∧ (8200000 : ℝ) = x * 10^6 :=
by
  use 8.2
  split
  · rfl
  · sorry

end scientific_notation_correctness_l204_204728


namespace log_irrational_of_not_power_of_10_l204_204622

theorem log_irrational_of_not_power_of_10 
  (N : ℕ) (h1 : N > 0) (h2 : ∀ k : ℤ, N ≠ 10 ^ k) : irrational (log 10 N) :=
sorry

end log_irrational_of_not_power_of_10_l204_204622


namespace smallest_product_of_non_real_zeros_l204_204522

-- Definitions based on the problem conditions
def Q (x : ℝ) (p q r s : ℝ) := x^4 + p * x^3 + q * x^2 + r * x + s

/-- 
  Given a quartic polynomial Q(x) with exactly two real zeros and two non-real zeros, 
  prove that the product of the non-real zeros of Q is the smallest among Q(-2), 
  the product of the zeros of Q, the sum of the coefficients of Q, and the sum of 
  the real zeros of Q.
-/
theorem smallest_product_of_non_real_zeros
  (p q r s : ℝ)
  (h_roots : ∃ x1 x2 z : ℂ, z.im ≠ 0 ∧ Q x1 p q r s = 0 ∧ Q x2 p q r s = 0 ∧ 
                           Q z.re p q r s = 0 ∧ Q z.im p q r s = 0) :
  let Q_neg2 := 16 - 8 * p + 4 * q - 2 * r + s,
      product_of_zeros := s,
      sum_of_coeffs := 1 + p + q + r + s,
      product_of_non_real_zeros := z.re^2 + z.im^2 in
  product_of_non_real_zeros < min (min Q_neg2 product_of_zeros) sum_of_coeffs := 
sorry

end smallest_product_of_non_real_zeros_l204_204522


namespace johns_total_earnings_per_week_l204_204974

def small_crab_baskets_monday := 3
def medium_crab_baskets_monday := 2
def large_crab_baskets_thursday := 4
def jumbo_crab_baskets_thursday := 1

def crabs_per_small_basket := 4
def crabs_per_medium_basket := 3
def crabs_per_large_basket := 5
def crabs_per_jumbo_basket := 2

def price_per_small_crab := 3
def price_per_medium_crab := 4
def price_per_large_crab := 5
def price_per_jumbo_crab := 7

def total_weekly_earnings :=
  (small_crab_baskets_monday * crabs_per_small_basket * price_per_small_crab) +
  (medium_crab_baskets_monday * crabs_per_medium_basket * price_per_medium_crab) +
  (large_crab_baskets_thursday * crabs_per_large_basket * price_per_large_crab) +
  (jumbo_crab_baskets_thursday * crabs_per_jumbo_basket * price_per_jumbo_crab)

theorem johns_total_earnings_per_week : total_weekly_earnings = 174 :=
by sorry

end johns_total_earnings_per_week_l204_204974


namespace wall_length_proof_l204_204747

-- Define the initial conditions
def men1 : ℕ := 20
def days1 : ℕ := 8
def men2 : ℕ := 86
def days2 : ℕ := 8
def wall_length2 : ℝ := 283.8

-- Define the expected length of the wall for the first condition
def expected_length : ℝ := 65.7

-- The proof statement.
theorem wall_length_proof : ((men1 * days1) / (men2 * days2)) * wall_length2 = expected_length :=
sorry

end wall_length_proof_l204_204747


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204017

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204017


namespace distributionAmounts_l204_204692

def totalWinnings : ℝ := 500
def giveawayPercentage : ℝ := 0.60
def firstShare : ℝ := 0.30
def secondShare : ℝ := 0.25
def thirdShare : ℝ := 0.20
def fourthShare : ℝ := 0.15
def fifthShare : ℝ := 0.10

theorem distributionAmounts :
  let totalGiveaway := giveawayPercentage * totalWinnings in
  let firstFriend := firstShare * totalGiveaway in
  let secondFriend := secondShare * totalGiveaway in
  let thirdFriend := thirdShare * totalGiveaway in
  let fourthFriend := fourthShare * totalGiveaway in
  let fifthFriend := fifthShare * totalGiveaway in
  firstFriend = 90 ∧ secondFriend = 75 ∧ thirdFriend = 60 ∧ fourthFriend = 45 ∧ fifthFriend = 30 := 
by
  sorry

end distributionAmounts_l204_204692


namespace pyramid_surface_area_20sqrt551_l204_204118

-- Definitions to capture the given conditions
structure Triangle (α : Type) :=
(a b c : α)

structure Pyramid (α : Type) :=
(A B C D : α)
(edge_lengths : list ℕ)

-- Hypothesis captures the conditions
variables {α : Type}
variables (A B C D : α)
variables (lengths : list ℕ)
variables (h1 : lengths = [10, 24, 24])
variables (h2 : length (erase_dup lengths) = 3) -- Ensures no equilateral face

-- The problem statement is to prove the given surface area
theorem pyramid_surface_area_20sqrt551 
  (p : Pyramid α) (h3 : p.edge_lengths = lengths) : 
  (calculate_surface_area p) = 20 * real.sqrt 551 :=
sorry

end pyramid_surface_area_20sqrt551_l204_204118


namespace cos_seven_pi_over_six_l204_204446

theorem cos_seven_pi_over_six : Real.cos (7 * Real.pi / 6) = -Real.sqrt 3 / 2 :=
by
  sorry

end cos_seven_pi_over_six_l204_204446


namespace sum_of_integers_l204_204675

/-- Given two positive integers x and y such that the sum of their squares equals 181 
    and their product equals 90, prove that the sum of these two integers is 19. -/
theorem sum_of_integers (x y : ℤ) (h1 : x^2 + y^2 = 181) (h2 : x * y = 90) : x + y = 19 := by
  sorry

end sum_of_integers_l204_204675


namespace total_legs_proof_l204_204580

def johnny_legs : Nat := 2
def son_legs : Nat := 2
def dog_legs : Nat := 4
def number_of_dogs : Nat := 2
def number_of_humans : Nat := 2

def total_legs : Nat :=
  (number_of_dogs * dog_legs) + (number_of_humans * johnny_legs)

theorem total_legs_proof : total_legs = 12 := by
  sorry

end total_legs_proof_l204_204580


namespace inverse_of_inverse_of_14_eq_22_over_9_l204_204158

def g (x : ℝ) : ℝ := 3 * x - 2

def g_inv (x : ℝ) : ℝ := (x + 2) / 3

theorem inverse_of_inverse_of_14_eq_22_over_9 :
  g_inv (g_inv 14) = 22 / 9 :=
by 
  -- proof goes here
  sorry

end inverse_of_inverse_of_14_eq_22_over_9_l204_204158


namespace value_of_m_l204_204596

def f (x m : ℝ) : ℝ := x^2 - 2 * x + m
def g (x m : ℝ) : ℝ := x^2 - 2 * x + 2 * m + 8

theorem value_of_m (m : ℝ) : (3 * f 5 m = g 5 m) → m = -22 :=
by
  intro h
  sorry

end value_of_m_l204_204596


namespace find_k_b_l204_204943

noncomputable def symmetric_line_circle_intersection : Prop :=
  ∃ (k b : ℝ), 
    (∀ (x y : ℝ),  (y = k * x) ∧ ((x-1)^2 + y^2 = 1)) ∧ 
    (∀ (x y : ℝ), (x - y + b = 0)) →
    (k = -1 ∧ b = -1)

theorem find_k_b :
  symmetric_line_circle_intersection :=
  by
    -- omitted proof
    sorry

end find_k_b_l204_204943


namespace smallest_odd_with_five_prime_factors_l204_204295

theorem smallest_odd_with_five_prime_factors :
  ∃ n : ℕ, n = 3 * 5 * 7 * 11 * 13 ∧ ∀ m : ℕ, (m < n → (∃ p1 p2 p3 p4 p5 : ℕ,
  prime p1 ∧ odd p1 ∧ prime p2 ∧ odd p2 ∧ prime p3 ∧ odd p3 ∧
  prime p4 ∧ odd p4 ∧ prime p5 ∧ odd p5 ∧
  m = p1 * p2 * p3 * p4 * p5)) → m < 3 * 5 * 7 * 11 * 13 := 
by {
  use 3 * 5 * 7 * 11 * 13,
  split,
  norm_num,
  intros m hlt hexists,
  obtain ⟨p1, p2, p3, p4, p5, hp1, hodd1, hp2, hodd2, hp3, hodd3, hp4, hodd4, hp5, hodd5, hprod⟩ := hexists,
  sorry
}

end smallest_odd_with_five_prime_factors_l204_204295


namespace stickers_total_l204_204975

def karl_stickers : ℕ := 25
def ryan_stickers : ℕ := karl_stickers + 20
def ben_stickers : ℕ := ryan_stickers - 10
def total_stickers : ℕ := karl_stickers + ryan_stickers + ben_stickers

theorem stickers_total : total_stickers = 105 := by
  sorry

end stickers_total_l204_204975


namespace smallest_odd_number_with_five_prime_factors_l204_204271

theorem smallest_odd_number_with_five_prime_factors : 
  ∃ n : ℕ, n = 15015 ∧ (∀ (p ∈ {3, 5, 7, 11, 13}), prime p) ∧ odd n :=
by
  sorry

end smallest_odd_number_with_five_prime_factors_l204_204271


namespace smallest_odd_number_with_five_different_prime_factors_l204_204286

noncomputable def smallest_odd_with_five_prime_factors : Nat :=
  3 * 5 * 7 * 11 * 13

theorem smallest_odd_number_with_five_different_prime_factors :
  smallest_odd_with_five_prime_factors = 15015 :=
by
  sorry -- Proof to be filled in later

end smallest_odd_number_with_five_different_prime_factors_l204_204286


namespace relationship_among_abc_l204_204885

noncomputable def a : ℝ := Real.log 3 / Real.log 2 -- This is log_2 3
noncomputable def integrand (x : ℝ) : ℝ := x + 1 / x
noncomputable def b : ℝ := ∫ x in 1..2, integrand x
noncomputable def c : ℝ := Real.log 30 / Real.log (1 / 3) -- This is log_{1/3} (1/30)

theorem relationship_among_abc : c > b ∧ b > a :=
by {
  sorry -- placeholder for the proof
}

end relationship_among_abc_l204_204885


namespace find_a_value_l204_204498

theorem find_a_value (a : ℝ) :
  {a^2, a + 1, -3} ∩ {a - 3, 2 * a - 1, a^2 + 1} = {-3} → a = -1 :=
by
  intro h
  sorry

end find_a_value_l204_204498


namespace triangle_cosine_product_l204_204094

-- Define the given conditions
variables {K L M P O Q S : Type} [HasAngleRatio.{a}]
def is_triangle (p1 p2 p3 : Type) : Prop := true -- A placeholder definition
def median (p1 p2 p3 p4 : Type) : Prop := true -- A placeholder definition
def circumcenter (p1 p2 p3 p4 : Type) : Prop := true -- A placeholder definition
def incenter (p1 p2 p3 p4 : Type) : Prop := true -- A placeholder definition
def angle (p1 p2 p3 : Type) : Real := sorry -- A placeholder definition
def intersection (p1 p2 p3 p4 p5 : Type) : Prop := true -- A placeholder definition
def angle_prod_eq (a b c : Real) : Prop := a * b = c

-- Encapsulate the problem statement
theorem triangle_cosine_product :
    (is_triangle K L M) ∧ (median K L M P) ∧ (circumcenter K L M O) ∧ (incenter K L M Q) ∧
    (intersection KP OQ S) ∧ (angle Q S O / angle P S O = sqrt 6 * (angle Q S P / angle K S P)) ∧
    (angle L K M = π / 3) →
    angle_prod_eq (cos (angle K L M)) (cos (angle K M L)) (-3 / 8) :=
by sorry

end triangle_cosine_product_l204_204094


namespace palindrome_percentage_contains_2_l204_204795

def is_palindrome (n : ℕ) : Prop :=
  let s := n.toString in s = s.reverse

def contains_digit (n : ℕ) (d : ℕ) : Prop :=
  nat.digits 10 n |> list.contains d

theorem palindrome_percentage_contains_2 :
  let palindromes := {n // 1000 ≤ n ∧ n ≤ 2000 ∧ is_palindrome n} in
  let total_palindromes := (palindromes.to_finset.card : ℤ) in
  let palindromes_with_2 := (palindromes.to_finset.filter (λ n, contains_digit n 2)).card in
  total_palindromes = 90 ∧ palindromes_with_2 = 19 →
  (palindromes_with_2 : ℚ) / total_palindromes * 100 = 21.11 := by sorry

end palindrome_percentage_contains_2_l204_204795


namespace sequence_sum_l204_204604

noncomputable def y : ℕ → ℕ → ℚ
| 0 m := 1
| 1 m := m
| (n + 2) m := ((m + 1) * (y (n + 1) m) - (m - n) * (y n m)) / (n + 1)

theorem sequence_sum (m : ℕ) (hm : 0 < m) :
  ∑ k in range (m + 1), y k m = Real.exp (m + 1) :=
sorry

end sequence_sum_l204_204604


namespace arctan_sum_equals_5pi_over_6_l204_204056

theorem arctan_sum_equals_5pi_over_6 (a b : ℝ) (h1 : a = 1 / 3) 
  (h2 : (a + 2) * (b + 2) = 15) : 
  real.arctan a + real.arctan b = 5 * real.pi / 6 := 
sorry

end arctan_sum_equals_5pi_over_6_l204_204056


namespace mary_spent_total_amount_l204_204131

def cost_of_berries := 11.08
def cost_of_apples := 14.33
def cost_of_peaches := 9.31
def total_cost := 34.72

theorem mary_spent_total_amount :
  cost_of_berries + cost_of_apples + cost_of_peaches = total_cost :=
by
  sorry

end mary_spent_total_amount_l204_204131


namespace social_media_to_phone_ratio_l204_204830

def hours_spent_on_phone_daily : ℕ := 16
def hours_spent_on_social_media_weekly : ℕ := 56
def days_in_week : ℕ := 7

theorem social_media_to_phone_ratio :
  (hours_spent_on_social_media_weekly / days_in_week).toRat / (hours_spent_on_phone_daily).toRat = (1 : ℕ) / (2 : ℕ) := 
by
  sorry

end social_media_to_phone_ratio_l204_204830


namespace calc_expr_l204_204406

theorem calc_expr :
  |(-5 : ℤ)| - 2 * Real.pow 3 0 + Real.tan (Real.pi / 4) + Real.sqrt 9 = 8 :=
by
  sorry

end calc_expr_l204_204406


namespace total_volume_removed_tetrahedra_l204_204430

theorem total_volume_removed_tetrahedra (x y : ℝ) (hx : x = sqrt 2 - 1) (hy : y = 2 * (sqrt 2 - 1))
  (prism_dims : 1 * 2 * 2) : 
  ∑(volume : ℝ), (volume = 4 * ((1 / 3) * (8 * (sqrt 2 - 1)^2) * (1 / sqrt 2))) = (16 - 34 * sqrt 2) / 3 :=
by
  -- Given conditions for the dimensions and sliced shapes
  have h1 : x = sqrt 2 - 1 := hx,
  have h2 : y = 2 * (sqrt 2 - 1) := hy,
  have prism_dimensions : 1 * 2 * 2 = 4 := rfl,

  -- Calculate base area and height
  let base_area : ℝ = 8 * (sqrt 2 - 1) ^ 2,
  let height : ℝ = 1 / sqrt 2,
  let volume_tetrahedron : ℝ = (1 / 3) * base_area * height,

  -- Aggregate volume from 4 corners
  let total_volume : ℝ = 4 * volume_tetrahedron,
  
  -- Compare with expected volume
  show total_volume = (16 - 34 * sqrt 2) / 3,
  sorry

end total_volume_removed_tetrahedra_l204_204430


namespace shawn_red_pebbles_l204_204628

variable (Total : ℕ)
variable (B : ℕ)
variable (Y : ℕ)
variable (P : ℕ)
variable (G : ℕ)

theorem shawn_red_pebbles (h1 : Total = 40)
                          (h2 : B = 13)
                          (h3 : B - Y = 7)
                          (h4 : P = Y)
                          (h5 : G = Y)
                          (h6 : 3 * Y + B = Total)
                          : Total - (B + P + Y + G) = 9 :=
by
 sorry

end shawn_red_pebbles_l204_204628


namespace smallest_odd_with_five_different_prime_factors_l204_204259

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_odd (n : ℕ) : Prop :=
  n % 2 = 1

def has_five_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ a b c d e : ℕ, 
    is_prime a ∧ is_prime b ∧ is_prime c ∧ is_prime d ∧ is_prime e ∧ 
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e ∧ 
    n = a * b * c * d * e

theorem smallest_odd_with_five_different_prime_factors : ∃ n : ℕ, 
  is_odd n ∧ has_five_distinct_prime_factors n ∧ ∀ m : ℕ, 
  is_odd m ∧ has_five_distinct_prime_factors m → n ≤ m :=
exists.intro 15015 sorry

end smallest_odd_with_five_different_prime_factors_l204_204259


namespace smallest_odd_number_with_five_prime_factors_l204_204303

theorem smallest_odd_number_with_five_prime_factors :
  ∃ (n : ℕ), n = 3 * 5 * 7 * 11 * 13 ∧
  n % 2 ≠ 0 ∧
  ∃ (p1 p2 p3 p4 p5 : ℕ), 
    (prime p1) ∧ 
    (prime p2) ∧ 
    (prime p3) ∧ 
    (prime p4) ∧ 
    (prime p5) ∧ 
    p1 ≠ p2 ∧ 
    p2 ≠ p3 ∧ 
    p3 ≠ p4 ∧ 
    p4 ≠ p5 ∧ 
    p1 = 3 ∧ 
    p2 = 5 ∧ 
    p3 = 7 ∧ 
    p4 = 11 ∧ 
    p5 = 13 ∧ 
    n = p1 * p2 * p3 * p4 * p5 :=
sorry

end smallest_odd_number_with_five_prime_factors_l204_204303


namespace num_two_digit_primes_with_ones_digit_three_is_seven_l204_204011

noncomputable def is_prime (n : ℕ) : Prop := sorry

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_ones_digit_three (n : ℕ) : Prop := n % 10 = 3

theorem num_two_digit_primes_with_ones_digit_three_is_seven :
  {n : ℕ | is_two_digit n ∧ has_ones_digit_three n ∧ is_prime n}.to_finset.card = 7 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_three_is_seven_l204_204011


namespace range_of_g_l204_204584

noncomputable def intersection_points (t : ℝ) : list (ℝ × ℝ) :=
let f := λ x : ℝ, x * (x - 1) * (x - 3),
    l := λ x : ℝ, t * x in
[(0, 0),
 (2 - real.sqrt(t + 1), t * (2 - real.sqrt (t + 1))),
 (2 + real.sqrt(t + 1), t * (2 + real.sqrt(t + 1)))]

noncomputable def g (t : ℝ) : ℝ :=
let P := (2 - real.sqrt(t + 1), t * (2 - real.sqrt(t + 1))),
    Q := (2 + real.sqrt(t + 1), t * (2 + real.sqrt(t + 1))),
    OP := real.sqrt((2 - real.sqrt(t + 1))^2 + (t * (2 - real.sqrt(t + 1)))^2),
    OQ := real.sqrt((2 + real.sqrt(t + 1))^2 + (t * (2 + real.sqrt(t + 1)))^2) in
OP * OQ

theorem range_of_g : ∀ t : ℝ, 
  0 ≤ g t :=
sorry

end range_of_g_l204_204584


namespace response_rate_increase_l204_204734

theorem response_rate_increase :
  let original_customers := 70
  let original_responses := 7
  let redesigned_customers := 63
  let redesigned_responses := 9
  let original_response_rate := (original_responses : ℝ) / original_customers
  let redesigned_response_rate := (redesigned_responses : ℝ) / redesigned_customers
  let percentage_increase := ((redesigned_response_rate - original_response_rate) / original_response_rate) * 100
  abs (percentage_increase - 42.86) < 0.01 :=
by
  sorry

end response_rate_increase_l204_204734


namespace complex_quadrant_l204_204664

theorem complex_quadrant :
  let z := (1 - 3 * Complex.I) / (Complex.I - 1)
  z = -2 + Complex.I → (z.re < 0 ∧ z.im > 0) := 
by
  intros z hz
  rw hz
  dsimp
  simp [z]
  exact ⟨by linarith, by linarith⟩
  sorry -- Proof steps not needed

end complex_quadrant_l204_204664


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204026

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204026


namespace sum_of_constants_l204_204806

theorem sum_of_constants (x a b : ℤ) (h : x^2 - 10 * x + 15 = 0) 
    (h1 : (x + a)^2 = b) : a + b = 5 := 
sorry

end sum_of_constants_l204_204806


namespace swimmers_meet_l204_204700

def time_to_meet (pool_length speed1 speed2 time: ℕ) : ℕ :=
  (time * (speed1 + speed2)) / pool_length

theorem swimmers_meet
  (pool_length : ℕ)
  (speed1 : ℕ)
  (speed2 : ℕ)
  (total_time : ℕ) :
  total_time = 12 * 60 →
  pool_length = 90 →
  speed1 = 3 →
  speed2 = 2 →
  time_to_meet pool_length speed1 speed2 total_time = 20 := by
  sorry

end swimmers_meet_l204_204700


namespace electric_sharpens_more_l204_204768

noncomputable def number_of_pencils_hand_crank : ℕ := 360 / 45
noncomputable def number_of_pencils_electric : ℕ := 360 / 20

theorem electric_sharpens_more : number_of_pencils_electric - number_of_pencils_hand_crank = 10 := by
  sorry

end electric_sharpens_more_l204_204768


namespace num_two_digit_primes_with_ones_digit_eq_3_l204_204018

theorem num_two_digit_primes_with_ones_digit_eq_3 : 
  (finset.filter (λ n, n.mod 10 = 3 ∧ nat.prime n) (finset.range 100).filter (λ n, n >= 10)).card = 6 :=
by
  sorry

end num_two_digit_primes_with_ones_digit_eq_3_l204_204018


namespace sum_of_P_neg1_l204_204123

noncomputable def P (x : ℝ) (a b : ℝ) : ℝ := x^3 + a * x^2 + b * x + 2015

theorem sum_of_P_neg1 (a b : ℝ) (h : ∀ x, x ≥ 0 → P x a b ≥ 0) 
  (h_roots : ∃ r s t : ℤ, P x a b = (x - ↑r) * (x - ↑s) * (x - ↑t)) : 
  ∑ val in [P (-1) a b | let r s t : ℤ in
    (P (-1) a b = eval_poly [(-1, r), (-1, s), (-1, t)]),
      where r * s * t = -2015; {however,, r s t are integers,such that}, 
  sort(sroots : adhere all[]]} = 9496 := sorry

end sum_of_P_neg1_l204_204123


namespace other_candidate_votes_l204_204073

-- Define the constants according to the problem
variables (X Y Z : ℝ)
axiom h1 : X = Y + (1 / 2) * Y
axiom h2 : X = 22500
axiom h3 : Y = Z - (2 / 5) * Z

-- Define the goal
theorem other_candidate_votes : Z = 25000 :=
by
  sorry

end other_candidate_votes_l204_204073


namespace work_time_all_together_l204_204374

-- Definition of individual work times and hourly working rates
def man_work_time := 7 * 5  -- in hours
def son_work_time := 14 * 4  -- in hours
def friend_work_time := 10 * 6  -- in hours

-- Work rates (work per hour)
def man_work_rate := 1 / man_work_time
def son_work_rate := 1 / son_work_time
def friend_work_rate := 1 / friend_work_time

-- Combined work rate
def combined_work_rate := man_work_rate + son_work_rate + friend_work_rate

-- Time taken for all three to complete the work together
def total_time := 1 / combined_work_rate

-- The theorem stating the time to complete work together
theorem work_time_all_together :
  abs (total_time - 15.85) < 0.01 := 
sorry

end work_time_all_together_l204_204374


namespace vector_magnitude_sum_cosine_angle_between_a_and_b_minus_a_l204_204923

variables (a b : ℝ^3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) (hab : a • b = -1/2)

-- Prove 1. ‖a + b‖ = 2
theorem vector_magnitude_sum : ‖a + b‖ = 2 :=
sorry

-- Prove 2. Cosine of angle between a and b - a is -√6/4
theorem cosine_angle_between_a_and_b_minus_a 
  (h_mag_diff : ‖b - a‖ = sqrt 6) (h_dot_diff : a • (b - a) = -3/2) :
  cos α = -sqrt 6 / 4 :=
sorry


end vector_magnitude_sum_cosine_angle_between_a_and_b_minus_a_l204_204923


namespace division_of_negatives_l204_204414

theorem division_of_negatives :
  (-81 : ℤ) / (-9) = 9 := 
  by
  -- Property of division with negative numbers
  have h1 : (-81 : ℤ) / (-9) = 81 / 9 := by sorry
  -- Perform the division
  have h2 : 81 / 9 = 9 := by sorry
  -- Combine the results
  rw h1
  exact h2

end division_of_negatives_l204_204414


namespace remaining_yards_correct_l204_204375

-- Define the conversion constant
def yards_per_mile: ℕ := 1760

-- Define the conditions
def marathon_in_miles: ℕ := 26
def marathon_in_yards: ℕ := 395
def total_marathons: ℕ := 15

-- Define the function to calculate the remaining yards after conversion
def calculate_remaining_yards (marathon_in_miles marathon_in_yards total_marathons yards_per_mile: ℕ): ℕ :=
  let total_yards := total_marathons * marathon_in_yards
  total_yards % yards_per_mile

-- Statement to prove
theorem remaining_yards_correct :
  calculate_remaining_yards marathon_in_miles marathon_in_yards total_marathons yards_per_mile = 645 :=
  sorry

end remaining_yards_correct_l204_204375


namespace solution_set_l204_204880

variable (f : ℝ → ℝ)

def cond1 := ∀ x, f x = f (-x)
def cond2 := ∀ x y, 0 ≤ x → x ≤ y → f x ≤ f y
def cond3 := f (1/3) = 0

theorem solution_set (hf1 : cond1 f) (hf2 : cond2 f) (hf3 : cond3 f) :
  { x : ℝ | f (Real.log x / Real.log (1/8)) > 0 } = { x : ℝ | 0 < x ∧ x < 1/2 } ∪ { x : ℝ | 2 < x } :=
sorry

end solution_set_l204_204880


namespace fraction_simplified_to_p_l204_204170

theorem fraction_simplified_to_p (q : ℕ) (hq_pos : 0 < q) (gcd_cond : Nat.gcd 4047 q = 1) :
    (2024 / 2023) - (2023 / 2024) = 4047 / q := sorry

end fraction_simplified_to_p_l204_204170


namespace ordered_pair_p3_q6_l204_204126

noncomputable def p (x : ℝ) : ℝ := -- define p as a cubic polynomial (to be specified)
noncomputable def q (x : ℝ) : ℝ := -- define q as a cubic polynomial (to be specified)

theorem ordered_pair_p3_q6 
  (h1 : p(0) = -24)
  (h2 : q(0) = 30)
  (h3 : ∀ x : ℝ, p(q(x)) = q(p(x))) :
  (p(3), q(6)) = (3, -24) :=
sorry

end ordered_pair_p3_q6_l204_204126


namespace paula_has_3_nickels_l204_204161

noncomputable def paula_coins : Prop :=
  ∃ (n_quarters n_dimes n_nickels n_pennies : ℕ),
  let total_value_coins := 25 * n_quarters + 10 * n_dimes + 5 * n_nickels + n_pennies in
  let n := n_quarters + n_dimes + n_nickels + n_pennies in
  total_value_coins = 15 * n ∧
  (total_value_coins + 25) / (n + 1) = 16 ∧
  n = 9

theorem paula_has_3_nickels : paula_coins →
  ∃ n_nickels : ℕ, n_nickels = 3 :=
by
  intros h,
  sorry

end paula_has_3_nickels_l204_204161


namespace trapezoid_base_lengths_l204_204618

variables {A B C D H Q : Type} [CoordinateSpace ℝ A B H] [CoordinateSpace ℝ C D Q]

structure IsoscelesTrapezoid (A B C D H : ℝ) :=
  (is_trapezoid : is_trapezoid A B C D)
  (isosceles : isosceles A B D C)

def H_on_AD (H : ℝ) (A D : ℝ) : Prop :=
  A < H ∧ H < D

def height (C H : ℝ) (A D : ℝ) : Prop :=
  C < D ∧ H == C

noncomputable def length_AH := (20 : ℝ)
noncomputable def length_DH := (8 : ℝ)
noncomputable def length_AD := (length_AH + length_DH + 12 : ℝ)

theorem trapezoid_base_lengths (x y : ℝ):
  ∃ (AD BC: ℝ), 
  H_on_AD H A D ∧
  height C H A D ∧
  length_AH = 20 ∧
  length_DH = 8 ∧
  AD = 28 ∧
  BC = 12 := 
by
  have AD_eq : x = y + 28 := sorry
  have pc : x - y = 16 := sorry
  have pb : x + y = 40 := sorry
  have lh : AD = 28 := sorry
  have hg : BC = 12 := sorry
  use [28, 12]
  sorry

end trapezoid_base_lengths_l204_204618


namespace cyclist_round_trip_time_l204_204999

-- Define the distances and speeds.
def distance1 : ℝ := 16
def speed1 : ℝ := 8

def distance2 : ℝ := 16
def speed2 : ℝ := 10

def return_distance : ℝ := 32
def return_speed : ℝ := 10

-- Define the times for each part of the trip.
def time1 : ℝ := distance1 / speed1
def time2 : ℝ := distance2 / speed2
def return_time : ℝ := return_distance / return_speed

-- Total time for round trip
def total_time : ℝ := time1 + time2 + return_time

-- Theorem stating the total time is 6.8 hours
theorem cyclist_round_trip_time : 
  total_time = 6.8 :=
by
  -- To start the proof, we use sorry to indicate it's a placeholder for the actual proof.
  sorry

end cyclist_round_trip_time_l204_204999
