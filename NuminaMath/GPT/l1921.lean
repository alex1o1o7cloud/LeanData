import Mathlib

namespace fraction_of_water_l1921_192106

/-- 
  Prove that the fraction of the mixture that is water is (\frac{2}{5}) 
  given the total weight of the mixture is 40 pounds, 
  1/4 of the mixture is sand, 
  and the remaining 14 pounds of the mixture is gravel. 
-/
theorem fraction_of_water 
  (total_weight : ℝ)
  (weight_sand : ℝ)
  (weight_gravel : ℝ)
  (weight_water : ℝ)
  (h1 : total_weight = 40)
  (h2 : weight_sand = (1/4) * total_weight)
  (h3 : weight_gravel = 14)
  (h4 : weight_water = total_weight - (weight_sand + weight_gravel)) :
  (weight_water / total_weight) = 2/5 :=
by
  sorry

end fraction_of_water_l1921_192106


namespace range_of_a_l1921_192124

open Real

theorem range_of_a (a x y : ℝ)
  (h1 : (x - a) ^ 2 + (y - (a + 2)) ^ 2 = 1)
  (h2 : ∃ M : ℝ × ℝ, (M.1 - a) ^ 2 + (M.2 - (a + 2)) ^ 2 = 1
                       ∧ dist M (0, 3) = 2 * dist M (0, 0)) :
  -3 ≤ a ∧ a ≤ 0 :=
sorry

end range_of_a_l1921_192124


namespace man_swim_upstream_distance_l1921_192156

theorem man_swim_upstream_distance (c d : ℝ) (h1 : 15.5 + c ≠ 0) (h2 : 15.5 - c ≠ 0) :
  (15.5 + c) * 2 = 36 ∧ (15.5 - c) * 2 = d → d = 26 := by
  sorry

end man_swim_upstream_distance_l1921_192156


namespace sqrt_of_4_equals_2_l1921_192170

theorem sqrt_of_4_equals_2 : Real.sqrt 4 = 2 :=
by sorry

end sqrt_of_4_equals_2_l1921_192170


namespace a10_eq_neg12_l1921_192196

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d a1 : ℤ)

-- Conditions of the problem
axiom arithmetic_sequence : ∀ n : ℕ, a_n n = a1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n : ℕ, S_n n = n * (2 * a1 + (n - 1) * d) / 2
axiom a2_eq_4 : a_n 2 = 4
axiom S8_eq_neg8 : S_n 8 = -8

-- The statement to prove
theorem a10_eq_neg12 : a_n 10 = -12 :=
sorry

end a10_eq_neg12_l1921_192196


namespace symmetric_point_origin_l1921_192197

-- Define the original point
def original_point : ℝ × ℝ := (4, -1)

-- Define a function to find the symmetric point with respect to the origin
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)

-- State the theorem
theorem symmetric_point_origin : symmetric_point original_point = (-4, 1) :=
sorry

end symmetric_point_origin_l1921_192197


namespace eliana_steps_total_l1921_192193

def eliana_walks_first_day_steps := 200 + 300
def eliana_walks_second_day_steps := 2 * eliana_walks_first_day_steps
def eliana_walks_third_day_steps := eliana_walks_second_day_steps + 100
def eliana_total_steps := eliana_walks_first_day_steps + eliana_walks_second_day_steps + eliana_walks_third_day_steps

theorem eliana_steps_total : eliana_total_steps = 2600 := by
  sorry

end eliana_steps_total_l1921_192193


namespace penny_remaining_money_l1921_192149

theorem penny_remaining_money (initial_money : ℤ) (socks_pairs : ℤ) (socks_cost_per_pair : ℤ) (hat_cost : ℤ) :
  initial_money = 20 → socks_pairs = 4 → socks_cost_per_pair = 2 → hat_cost = 7 → 
  initial_money - (socks_pairs * socks_cost_per_pair + hat_cost) = 5 := 
by
  intros h₁ h₂ h₃ h₄
  sorry

end penny_remaining_money_l1921_192149


namespace find_values_of_a_and_c_l1921_192122

theorem find_values_of_a_and_c
  (a c : ℝ)
  (h1 : ∀ x : ℝ, (1 / 3 < x ∧ x < 1 / 2) ↔ a * x^2 + 5 * x + c > 0) :
  a = -6 ∧ c = -1 :=
by
  sorry

end find_values_of_a_and_c_l1921_192122


namespace length_breadth_difference_l1921_192148

theorem length_breadth_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 288) : L - W = 12 :=
by
  sorry

end length_breadth_difference_l1921_192148


namespace second_car_mileage_l1921_192185

theorem second_car_mileage (x : ℝ) : 
  (150 / 50) + (150 / x) + (150 / 15) = 56 / 2 → x = 10 :=
by
  intro h
  sorry

end second_car_mileage_l1921_192185


namespace sin_C_in_right_triangle_l1921_192109

theorem sin_C_in_right_triangle
  (A B C : ℝ)
  (sin_A : ℝ)
  (sin_B : ℝ)
  (B_right_angle : B = π / 2)
  (sin_A_value : sin_A = 3 / 5)
  (sin_B_value : sin_B = 1)
  (sin_of_C : ℝ)
  (tri_ABC : A + B + C = π ∧ A > 0 ∧ C > 0) :
    sin_of_C = 4 / 5 :=
by
  -- Skipping the proof
  sorry

end sin_C_in_right_triangle_l1921_192109


namespace arithmetic_sequence_solution_l1921_192101

variable (a d : ℤ)
variable (n : ℕ)

/-- Given the following conditions:
1. The sum of the first three terms of an arithmetic sequence is -3.
2. The product of the first three terms is 8,
This theorem proves that:
1. The general term formula of the sequence is 3 * n - 7.
2. The sum of the first n terms is (3 / 2) * n ^ 2 - (11 / 2) * n.
-/
theorem arithmetic_sequence_solution
  (h1 : (a - d) + a + (a + d) = -3)
  (h2 : (a - d) * a * (a + d) = 8) :
  (∃ a d : ℤ, (∀ n : ℕ, (n ≥ 1) → (3 * n - 7 = a + (n - 1) * d) ∧ (∃ S : ℕ → ℤ, S n = (3 / 2) * n ^ 2 - (11 / 2) * n))) :=
by
  sorry

end arithmetic_sequence_solution_l1921_192101


namespace complement_union_l1921_192130

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}
def complement_U_A : Set ℕ := U \ A

theorem complement_union (U A B : Set ℕ) (hU : U = {0, 1, 2, 3, 4}) (hA : A = {1, 2, 3}) (hB : B = {2, 4}) :
  (complement_U_A ∪ B) = {0, 2, 4} := by
  sorry

end complement_union_l1921_192130


namespace Valley_Forge_High_School_winter_carnival_l1921_192158

noncomputable def number_of_girls (total_students : ℕ) (total_participants : ℕ) (fraction_girls_participating : ℚ) (fraction_boys_participating : ℚ) : ℕ := sorry

theorem Valley_Forge_High_School_winter_carnival
  (total_students : ℕ)
  (total_participants : ℕ)
  (fraction_girls_participating : ℚ)
  (fraction_boys_participating : ℚ)
  (h_total_students : total_students = 1500)
  (h_total_participants : total_participants = 900)
  (h_fraction_girls : fraction_girls_participating = 3 / 4)
  (h_fraction_boys : fraction_boys_participating = 2 / 3) :
  number_of_girls total_students total_participants fraction_girls_participating fraction_boys_participating = 900 := sorry

end Valley_Forge_High_School_winter_carnival_l1921_192158


namespace original_number_increased_by_110_l1921_192152

-- Define the conditions and the proof statement without the solution steps
theorem original_number_increased_by_110 {x : ℝ} (h : x + 1.10 * x = 1680) : x = 800 :=
by 
  sorry

end original_number_increased_by_110_l1921_192152


namespace digit_7_count_correct_l1921_192153

def base8ToBase10 (n : Nat) : Nat :=
  -- converting base 8 number 1000 to base 10
  1 * 8^3 + 0 * 8^2 + 0 * 8^1 + 0 * 8^0

def countDigit7 (n : Nat) : Nat :=
  -- counts the number of times the digit '7' appears in numbers from 1 to n
  let digits := (List.range (n + 1)).map fun x => x.digits 10
  digits.foldl (fun acc ds => acc + ds.count 7) 0

theorem digit_7_count_correct : countDigit7 512 = 123 := by
  sorry

end digit_7_count_correct_l1921_192153


namespace gcd_64_144_l1921_192157

theorem gcd_64_144 : Nat.gcd 64 144 = 16 := by
  sorry

end gcd_64_144_l1921_192157


namespace range_of_a_l1921_192173

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 1| - |x - 2| < a^2 + a + 1) →
  (a < -1 ∨ a > 0) :=
by
  sorry

end range_of_a_l1921_192173


namespace move_digit_to_make_equation_correct_l1921_192107

theorem move_digit_to_make_equation_correct :
  101 - 102 ≠ 1 → (101 - 10^2 = 1) :=
by
  sorry

end move_digit_to_make_equation_correct_l1921_192107


namespace standard_equation_of_parabola_l1921_192164

theorem standard_equation_of_parabola (F : ℝ × ℝ) (hF : F.1 + 2 * F.2 + 3 = 0) :
  (∃ y₀: ℝ, y₀ < 0 ∧ F = (0, y₀) ∧ ∀ x: ℝ, x ^ 2 = - 6 * y₀ * x) ∨
  (∃ x₀: ℝ, x₀ < 0 ∧ F = (x₀, 0) ∧ ∀ y: ℝ, y ^ 2 = - 12 * x₀ * y) :=
sorry

end standard_equation_of_parabola_l1921_192164


namespace find_selling_price_l1921_192167

def cost_price : ℝ := 59
def selling_price_for_loss : ℝ := 52
def loss := cost_price - selling_price_for_loss

theorem find_selling_price (sp : ℝ) : (sp - cost_price = loss) → sp = 66 :=
by
  sorry

end find_selling_price_l1921_192167


namespace staff_meeting_doughnuts_l1921_192129

theorem staff_meeting_doughnuts (n_d n_s n_l : ℕ) (h₁ : n_d = 50) (h₂ : n_s = 19) (h₃ : n_l = 12) :
  (n_d - n_l) / n_s = 2 :=
by
  sorry

end staff_meeting_doughnuts_l1921_192129


namespace derivative_y_l1921_192137

noncomputable def u (x : ℝ) := 4 * x - 1 + Real.sqrt (16 * x ^ 2 - 8 * x + 2)
noncomputable def v (x : ℝ) := Real.sqrt (16 * x ^ 2 - 8 * x + 2) * Real.arctan (4 * x - 1)

noncomputable def y (x : ℝ) := Real.log (u x) - v x

theorem derivative_y (x : ℝ) :
  deriv y x = (4 * (1 - 4 * x)) / (Real.sqrt (16 * x ^ 2 - 8 * x + 2)) * Real.arctan (4 * x - 1) :=
by
  sorry

end derivative_y_l1921_192137


namespace negation_of_exists_l1921_192165

theorem negation_of_exists (x : ℝ) : 
  ¬ (∃ x : ℝ, 2 * x^2 + 2 * x - 1 ≤ 0) ↔ ∀ x : ℝ, 2 * x^2 + 2 * x - 1 > 0 :=
by
  sorry

end negation_of_exists_l1921_192165


namespace range_of_a_l1921_192150

noncomputable def equation_has_two_roots (a m : ℝ) : Prop :=
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ 
    x₁ + a * (2 * x₁ + 2 * m - 4 * Real.exp 1 * x₁) * (Real.log (x₁ + m) - Real.log x₁) = 0 ∧ 
    x₂ + a * (2 * x₂ + 2 * m - 4 * Real.exp 1 * x₂) * (Real.log (x₂ + m) - Real.log x₂) = 0

theorem range_of_a (m : ℝ) (hm : 0 < m) : 
  (∃ a, equation_has_two_roots a m) ↔ (a < 0 ∨ a > 1 / (2 * Real.exp 1)) := 
sorry

end range_of_a_l1921_192150


namespace total_fruit_punch_eq_21_l1921_192184

def orange_punch : ℝ := 4.5
def cherry_punch := 2 * orange_punch
def apple_juice := cherry_punch - 1.5

theorem total_fruit_punch_eq_21 : orange_punch + cherry_punch + apple_juice = 21 := by 
  -- This is where the proof would go
  sorry

end total_fruit_punch_eq_21_l1921_192184


namespace sum_of_possible_g9_values_l1921_192177

def f (x : ℝ) : ℝ := x^2 - 6 * x + 14

def g (y : ℝ) : ℝ := 3 * y + 2

theorem sum_of_possible_g9_values : ∀ {x1 x2 : ℝ}, f x1 = 9 → f x2 = 9 → g x1 + g x2 = 22 := by
  intros
  sorry

end sum_of_possible_g9_values_l1921_192177


namespace sparse_real_nums_l1921_192160

noncomputable def is_sparse (r : ℝ) : Prop :=
  ∃n > 0, ∀s : ℝ, s^n = r → s = 1 ∨ s = -1 ∨ s = 0

theorem sparse_real_nums (r : ℝ) : is_sparse r ↔ r = -1 ∨ r = 0 ∨ r = 1 := 
by
  sorry

end sparse_real_nums_l1921_192160


namespace arithmetic_sequence_a1_a9_l1921_192119

variable (a : ℕ → ℝ)

-- This statement captures if given condition holds, prove a_1 + a_9 = 18.
theorem arithmetic_sequence_a1_a9 (h : a 4 + a 5 + a 6 = 27)
    (h_seq : ∀ (n : ℕ), a (n + 1) = a n + (a 2 - a 1)) :
    a 1 + a 9 = 18 :=
sorry

end arithmetic_sequence_a1_a9_l1921_192119


namespace find_ab_sum_eq_42_l1921_192169

noncomputable def find_value (a b : ℝ) : ℝ := a^2 * b + a * b^2

theorem find_ab_sum_eq_42 (a b : ℝ) (h1 : a + b = 6) (h2 : a * b = 7) : find_value a b = 42 := by
  sorry

end find_ab_sum_eq_42_l1921_192169


namespace Amanda_needs_12_more_marbles_l1921_192113

theorem Amanda_needs_12_more_marbles (K A M : ℕ)
  (h1 : M = 5 * K)
  (h2 : M = 85)
  (h3 : M = A + 63) :
  A + 12 = 2 * K := 
sorry

end Amanda_needs_12_more_marbles_l1921_192113


namespace find_n_sequence_sum_l1921_192141

theorem find_n_sequence_sum 
  (a : ℕ → ℝ) (S : ℕ → ℝ)
  (h₀ : ∀ n, a n = (2^n - 1) / 2^n)
  (h₁ : S 6 = 321 / 64) :
  ∃ n, S n = 321 / 64 ∧ n = 6 := 
by 
  sorry

end find_n_sequence_sum_l1921_192141


namespace f_2013_value_l1921_192142

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def g (x : ℝ) : ℝ := sorry

axiom h1 : ∀ x : ℝ, x ≠ 1 → f (2 * x + 1) + g (3 - x) = x
axiom h2 : ∀ x : ℝ, x ≠ 1 → f ((3 * x + 5) / (x + 1)) + 2 * g ((2 * x + 1) / (x + 1)) = x / (x + 1)

theorem f_2013_value : f 2013 = 1010 / 1007 :=
by
  sorry

end f_2013_value_l1921_192142


namespace arithmetic_sequence_terms_l1921_192194

theorem arithmetic_sequence_terms (a : ℕ → ℕ) (n : ℕ)
  (h1 : a 1 + a 2 + a 3 = 34)
  (h2 : a n + a (n - 1) + a (n - 2) = 146)
  (h3 : n * (a 1 + a n) = 780) : n = 13 :=
sorry

end arithmetic_sequence_terms_l1921_192194


namespace find_solutions_l1921_192147

def satisfies_inequality (x : ℝ) : Prop :=
  (Real.cos x)^2018 + (1 / (Real.sin x))^2019 ≤ (Real.sin x)^2018 + (1 / (Real.cos x))^2019

def in_intervals (x : ℝ) : Prop :=
  (x ∈ Set.Ico (-Real.pi / 3) 0) ∨
  (x ∈ Set.Ico (Real.pi / 4) (Real.pi / 2)) ∨
  (x ∈ Set.Ioc Real.pi (5 * Real.pi / 4)) ∨
  (x ∈ Set.Ioc (3 * Real.pi / 2) (5 * Real.pi / 3))

theorem find_solutions :
  ∀ x : ℝ, x ∈ Set.Icc (-Real.pi / 3) (5 * Real.pi / 3) →
  satisfies_inequality x ↔ in_intervals x := 
  by sorry

end find_solutions_l1921_192147


namespace trigonometric_inequality_l1921_192118

theorem trigonometric_inequality (x : Real) (n : Int) :
  (9.286 * (Real.sin x)^3 * Real.sin ((Real.pi / 2) - 3 * x) +
   (Real.cos x)^3 * Real.cos ((Real.pi / 2) - 3 * x) > 
   3 * Real.sqrt 3 / 8) →
   (x > (Real.pi / 12) + (Real.pi * n / 2) ∧
   x < (5 * Real.pi / 12) + (Real.pi * n / 2)) :=
by
  sorry

end trigonometric_inequality_l1921_192118


namespace quadratic_is_perfect_square_l1921_192100

theorem quadratic_is_perfect_square (a b c : ℤ) 
  (h : ∀ x : ℤ, ∃ d e : ℤ, a*x^2 + b*x + c = (d*x + e)^2) : 
  ∃ d e : ℤ, a = d^2 ∧ b = 2*d*e ∧ c = e^2 :=
sorry

end quadratic_is_perfect_square_l1921_192100


namespace P2011_1_neg1_is_0_2_pow_1006_l1921_192132

def P1 (x y : ℤ) : ℤ × ℤ := (x + y, x - y)

def Pn : ℕ → ℤ → ℤ → ℤ × ℤ 
| 0, x, y => (x, y)
| (n + 1), x, y => P1 (Pn n x y).1 (Pn n x y).2

theorem P2011_1_neg1_is_0_2_pow_1006 : Pn 2011 1 (-1) = (0, 2^1006) := by
  sorry

end P2011_1_neg1_is_0_2_pow_1006_l1921_192132


namespace find_x_l1921_192166

variables (a b x : ℝ)
variables (pos_a : a > 0) (pos_b : b > 0) (pos_x : x > 0)

theorem find_x : ((2 * a) ^ (2 * b) = (a^2) ^ b * x ^ b) → (x = 4) := by
  sorry

end find_x_l1921_192166


namespace white_wash_cost_l1921_192175

noncomputable def room_length : ℝ := 25
noncomputable def room_width : ℝ := 15
noncomputable def room_height : ℝ := 12
noncomputable def door_height : ℝ := 6
noncomputable def door_width : ℝ := 3
noncomputable def window_height : ℝ := 4
noncomputable def window_width : ℝ := 3
noncomputable def num_windows : ℕ := 3
noncomputable def cost_per_sqft : ℝ := 3

theorem white_wash_cost :
  let wall_area := 2 * (room_length * room_height + room_width * room_height)
  let door_area := door_height * door_width
  let window_area := window_height * window_width
  let total_non_white_wash_area := door_area + ↑num_windows * window_area
  let white_wash_area := wall_area - total_non_white_wash_area
  let total_cost := white_wash_area * cost_per_sqft
  total_cost = 2718 :=  
by
  sorry

end white_wash_cost_l1921_192175


namespace remainder_problem_l1921_192139

theorem remainder_problem (n m q1 q2 : ℤ) (h1 : n = 11 * q1 + 1) (h2 : m = 17 * q2 + 3) :
  ∃ r : ℤ, (r = (5 * n + 3 * m) % 11) ∧ (r = (7 * q2 + 3) % 11) :=
by
  sorry

end remainder_problem_l1921_192139


namespace missed_both_shots_l1921_192181

variables (p q : Prop)

theorem missed_both_shots : (¬p ∧ ¬q) ↔ ¬(p ∨ q) :=
by sorry

end missed_both_shots_l1921_192181


namespace smallest_obtuse_triangles_l1921_192162

def obtuseTrianglesInTriangulation (n : Nat) : Nat :=
  if n < 3 then 0 else (n - 2) - 2

theorem smallest_obtuse_triangles (n : Nat) (h : n = 2003) :
  obtuseTrianglesInTriangulation n = 1999 := by
  sorry

end smallest_obtuse_triangles_l1921_192162


namespace a_parallel_b_l1921_192102

variable {Line : Type} (a b c : Line)

-- Definition of parallel lines
def parallel (x y : Line) : Prop := sorry

-- Conditions
axiom a_parallel_c : parallel a c
axiom b_parallel_c : parallel b c

-- Theorem to prove a is parallel to b given the conditions
theorem a_parallel_b : parallel a b :=
by
  sorry

end a_parallel_b_l1921_192102


namespace total_votes_l1921_192105

-- Conditions
variables (V : ℝ)
def candidate_votes := 0.31 * V
def rival_votes := 0.31 * V + 2451

-- Problem statement
theorem total_votes (h : candidate_votes V + rival_votes V = V) : V = 6450 :=
sorry

end total_votes_l1921_192105


namespace minimum_value_of_xy_l1921_192103

noncomputable def minimum_value_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : ℝ :=
  if hmin : 4 * x + y + 12 = x * y then 36 else sorry

theorem minimum_value_of_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : 4 * x + y + 12 = x * y) : 
  minimum_value_xy x y hx hy h = 36 :=
sorry

end minimum_value_of_xy_l1921_192103


namespace ellipse_foci_coordinates_l1921_192126

theorem ellipse_foci_coordinates :
  ∀ (x y : ℝ),
  (x^2 / 9 + y^2 / 5 = 1) →
  (x, y) = (2, 0) ∨ (x, y) = (-2, 0) :=
by
  sorry

end ellipse_foci_coordinates_l1921_192126


namespace even_function_A_value_l1921_192135

-- Given function definition
def f (x : ℝ) (A : ℝ) : ℝ := (x + 1) * (x - A)

-- Statement to prove
theorem even_function_A_value (A : ℝ) (h : ∀ x : ℝ, f x A = f (-x) A) : A = 1 :=
by
  sorry

end even_function_A_value_l1921_192135


namespace sufficient_condition_l1921_192199

-- Definitions of propositions p and q
variables (p q : Prop)

-- Theorem statement
theorem sufficient_condition (h : ¬(p ∨ q)) : ¬p :=
by sorry

end sufficient_condition_l1921_192199


namespace apples_handed_out_to_students_l1921_192136

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

end apples_handed_out_to_students_l1921_192136


namespace water_added_l1921_192121

theorem water_added (x : ℝ) (salt_percent_initial : ℝ) (evaporation_fraction : ℝ) 
(salt_added : ℝ) (resulting_salt_percent : ℝ) 
(hx : x = 119.99999999999996) (h_initial_salt : salt_percent_initial = 0.20) 
(h_evap_fraction : evaporation_fraction = 1/4) (h_salt_added : salt_added = 16)
(h_resulting_salt_percent : resulting_salt_percent = 1/3) : 
∃ (water_added : ℝ), water_added = 30 :=
by
  sorry

end water_added_l1921_192121


namespace tangent_line_at_1_intervals_of_monotonicity_and_extrema_l1921_192145

open Real

noncomputable def f (x : ℝ) := 6 * log x + (1 / 2) * x^2 - 5 * x

theorem tangent_line_at_1 :
  let f' (x : ℝ) := (6 / x) + x - 5
  (f 1 = -9 / 2) →
  (f' 1 = 2) →
  (∀ x y : ℝ, y + 9 / 2 = 2 * (x - 1) → 4 * x - 2 * y - 13 = 0) := 
by
  sorry

theorem intervals_of_monotonicity_and_extrema :
  let f' (x : ℝ) := (x^2 - 5 * x + 6) / x
  (∀ x, 0 < x ∧ x < 2 → f' x > 0) → 
  (∀ x, 3 < x → f' x > 0) →
  (∀ x, 2 < x ∧ x < 3 → f' x < 0) →
  (f 2 = -8 + 6 * log 2) →
  (f 3 = -21 / 2 + 6 * log 3) :=
by
  sorry

end tangent_line_at_1_intervals_of_monotonicity_and_extrema_l1921_192145


namespace probability_x_add_y_lt_4_in_square_l1921_192144

noncomputable def square_area : ℝ := 3 * 3

noncomputable def triangle_area : ℝ := (1 / 2) * 2 * 2

noncomputable def region_area : ℝ := square_area - triangle_area

noncomputable def probability (A B : ℝ) : ℝ := A / B

theorem probability_x_add_y_lt_4_in_square :
  probability region_area square_area = 7 / 9 :=
by 
  sorry

end probability_x_add_y_lt_4_in_square_l1921_192144


namespace probability_of_exactly_one_red_ball_l1921_192174

-- Definitions based on the conditions:
def total_balls : ℕ := 5
def red_balls : ℕ := 2
def white_balls : ℕ := 3
def draw_count : ℕ := 2

-- Required to calculate combinatory values
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Definitions of probabilities (though we won't use them explicitly for the statement):
def total_events : ℕ := choose total_balls draw_count
def no_red_ball_events : ℕ := choose white_balls draw_count
def one_red_ball_events : ℕ := choose red_balls 1 * choose white_balls 1

-- Probability Functions (for context):
def probability (events : ℕ) (total_events : ℕ) : ℚ := events / total_events

-- Lean 4 statement:
theorem probability_of_exactly_one_red_ball :
  probability one_red_ball_events total_events = 3/5 := by
  sorry

end probability_of_exactly_one_red_ball_l1921_192174


namespace cos_value_l1921_192186

theorem cos_value (α : ℝ) (h : Real.sin (π/4 + α) = 1/3) :
  Real.cos (π/2 - 2*α) = -7/9 :=
sorry

end cos_value_l1921_192186


namespace smallest_value_of_n_l1921_192117

/-- Given that Casper has exactly enough money to buy either 
  18 pieces of red candy, 20 pieces of green candy, 
  25 pieces of blue candy, or n pieces of purple candy where 
  each purple candy costs 30 cents, prove that the smallest 
  possible value of n is 30.
-/
theorem smallest_value_of_n
  (r g b n : ℕ)
  (h : 18 * r = 20 * g ∧ 20 * g = 25 * b ∧ 25 * b = 30 * n) : 
  n = 30 :=
sorry

end smallest_value_of_n_l1921_192117


namespace total_distance_traveled_l1921_192111

/-- Defining the distance Greg travels in each leg of his trip -/
def distance_workplace_to_market : ℕ := 30

def distance_market_to_friend : ℕ := distance_workplace_to_market + 10

def distance_friend_to_aunt : ℕ := 5

def distance_aunt_to_grocery : ℕ := 7

def distance_grocery_to_home : ℕ := 18

/-- The total distance Greg traveled during his entire trip is the sum of all individual distances -/
theorem total_distance_traveled :
  distance_workplace_to_market + distance_market_to_friend + distance_friend_to_aunt + distance_aunt_to_grocery + distance_grocery_to_home = 100 :=
by
  sorry

end total_distance_traveled_l1921_192111


namespace age_of_father_now_l1921_192128

variable (M F : ℕ)

theorem age_of_father_now :
  (M = 2 * F / 5) ∧ (M + 14 = (F + 14) / 2) → F = 70 :=
by 
sorry

end age_of_father_now_l1921_192128


namespace part_one_part_two_l1921_192168

variable {a b c : ℝ}
variable (h_pos : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_eq : a^2 + b^2 + 4*c^2 = 3)

theorem part_one (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) :
  a + b + 2*c ≤ 3 :=
sorry

theorem part_two (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 + b^2 + 4*c^2 = 3) (h_b_eq_2c : b = 2*c) :
  1/a + 1/c ≥ 3 :=
sorry

end part_one_part_two_l1921_192168


namespace find_greater_number_l1921_192195

theorem find_greater_number (x y : ℕ) 
  (h1 : x + y = 40)
  (h2 : x - y = 12) : x = 26 :=
by
  sorry

end find_greater_number_l1921_192195


namespace isabela_spent_2800_l1921_192155

/-- Given:
1. Isabela bought twice as many cucumbers as pencils.
2. Both cucumbers and pencils cost $20 each.
3. Isabela got a 20% discount on the pencils.
4. She bought 100 cucumbers.
Prove that the total amount Isabela spent is $2800. -/
theorem isabela_spent_2800 :
  ∀ (pencils cucumbers : ℕ) (pencil_cost cucumber_cost : ℤ) (discount rate: ℚ)
    (total_cost pencils_cost cucumbers_cost discount_amount : ℤ),
  cucumbers = 100 →
  pencils * 2 = cucumbers →
  pencil_cost = 20 →
  cucumber_cost = 20 →
  rate = 20 / 100 →
  pencils_cost = pencils * pencil_cost →
  discount_amount = pencils_cost * rate →
  total_cost = pencils_cost - discount_amount + cucumbers * cucumber_cost →
  total_cost = 2800 := by
  intros _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
  sorry

end isabela_spent_2800_l1921_192155


namespace probability_all_even_l1921_192180

theorem probability_all_even :
  let die1_even_count := 3
  let die1_total := 6
  let die2_even_count := 3
  let die2_total := 7
  let die3_even_count := 4
  let die3_total := 9
  let prob_die1_even := die1_even_count / die1_total
  let prob_die2_even := die2_even_count / die2_total
  let prob_die3_even := die3_even_count / die3_total
  let probability_all_even := prob_die1_even * prob_die2_even * prob_die3_even
  probability_all_even = 1 / 10.5 :=
by
  sorry

end probability_all_even_l1921_192180


namespace linda_max_servings_is_13_l1921_192110

noncomputable def max_servings 
  (recipe_bananas : ℕ) (recipe_yogurt : ℕ) (recipe_honey : ℕ)
  (linda_bananas : ℕ) (linda_yogurt : ℕ) (linda_honey : ℕ)
  (servings_for_recipe : ℕ) : ℕ :=
  min 
    (linda_bananas * servings_for_recipe / recipe_bananas) 
    (min 
      (linda_yogurt * servings_for_recipe / recipe_yogurt)
      (linda_honey * servings_for_recipe / recipe_honey)
    )

theorem linda_max_servings_is_13 : 
  max_servings 3 2 1 10 9 4 4 = 13 :=
  sorry

end linda_max_servings_is_13_l1921_192110


namespace average_age_l1921_192198

variable (John Mary Tonya : ℕ)

theorem average_age (h1 : John = 2 * Mary) (h2 : John = Tonya / 2) (h3 : Tonya = 60) : 
  (John + Mary + Tonya) / 3 = 35 :=
by
  sorry

end average_age_l1921_192198


namespace even_numbers_average_18_l1921_192146

variable (n : ℕ)
variable (avg : ℕ)

theorem even_numbers_average_18 (h : avg = 18) : n = 17 := 
    sorry

end even_numbers_average_18_l1921_192146


namespace candles_lit_time_correct_l1921_192189

noncomputable def candle_time : String :=
  let initial_length := 1 -- Since the length is uniform, we use 1
  let rateA := initial_length / (6 * 60) -- Rate at which Candle A burns out
  let rateB := initial_length / (8 * 60) -- Rate at which Candle B burns out
  let t := 320 -- The time in minutes that satisfy the condition
  let time_lit := (16 * 60 - t) / 60 -- Convert minutes to hours
  if time_lit = 10 + 40 / 60 then "10:40 AM" else "Unknown"

theorem candles_lit_time_correct :
  candle_time = "10:40 AM" := 
by
  sorry

end candles_lit_time_correct_l1921_192189


namespace find_f_m_l1921_192104

-- Definitions based on the conditions
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 3

axiom condition (m a : ℝ) : f (-m) a = 1

-- The statement to be proven
theorem find_f_m (m a : ℝ) (hm : f (-m) a = 1) : f m a = 5 := 
by sorry

end find_f_m_l1921_192104


namespace find_K_l1921_192116

theorem find_K : ∃ K : ℕ, (64 ^ (2 / 3) * 16 ^ 2) / 4 = 2 ^ K :=
by
  use 10
  sorry

end find_K_l1921_192116


namespace alyssa_puppies_l1921_192143

theorem alyssa_puppies (total_puppies : ℕ) (given_away : ℕ) (remaining_puppies : ℕ) 
  (h1 : total_puppies = 7) (h2 : given_away = 5) 
  : remaining_puppies = total_puppies - given_away → remaining_puppies = 2 :=
by
  intro h
  rw [h1, h2] at h
  exact h

end alyssa_puppies_l1921_192143


namespace avg_age_of_women_l1921_192123

theorem avg_age_of_women (T : ℕ) (W : ℕ) (T_avg : ℕ) (H1 : T_avg = T / 10)
  (H2 : (T_avg + 6) = ((T - 18 - 22 + W) / 10)) : (W / 2) = 50 :=
sorry

end avg_age_of_women_l1921_192123


namespace intersection_of_lines_l1921_192120

theorem intersection_of_lines :
  ∃ (x y : ℝ), 10 * x - 5 * y = 5 ∧ 8 * x + 2 * y = 22 ∧ x = 2 ∧ y = 3 := by
  sorry

end intersection_of_lines_l1921_192120


namespace rhombus_area_l1921_192154

theorem rhombus_area (d1 d2 : ℝ) (h1 : d1 = 30) (h2 : d2 = 12) : (d1 * d2) / 2 = 180 :=
by
  sorry

end rhombus_area_l1921_192154


namespace problem_part1_problem_part2_l1921_192112

def U : Set ℕ := {x | 0 < x ∧ x < 9}

def S : Set ℕ := {1, 3, 5}

def T : Set ℕ := {3, 6}

theorem problem_part1 : S ∩ T = {3} := by
  sorry

theorem problem_part2 : U \ (S ∪ T) = {2, 4, 7, 8} := by
  sorry

end problem_part1_problem_part2_l1921_192112


namespace students_exceed_guinea_pigs_l1921_192188

theorem students_exceed_guinea_pigs :
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  total_students - total_guinea_pigs = 85 :=
by
  -- using the conditions and correct answer identified above
  let classrooms := 5
  let students_per_classroom := 20
  let guinea_pigs_per_classroom := 3
  let total_students := classrooms * students_per_classroom
  let total_guinea_pigs := classrooms * guinea_pigs_per_classroom
  show total_students - total_guinea_pigs = 85
  sorry

end students_exceed_guinea_pigs_l1921_192188


namespace problem_statement_l1921_192171

section

variable {f : ℝ → ℝ}

-- Conditions
axiom even_function (h : ∀ x : ℝ, f (-x) = f x) : ∀ x, f (-x) = f x 
axiom monotonically_increasing (h : ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y

-- Goal
theorem problem_statement 
  (h_even : ∀ x, f (-x) = f x)
  (h_mono : ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f x ≤ f y) :
  f (-Real.log 2 / Real.log 3) > f (Real.log 2 / Real.log 3) ∧ f (Real.log 2 / Real.log 3) > f 0 := 
sorry

end

end problem_statement_l1921_192171


namespace gcd_459_357_l1921_192191

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_459_357_l1921_192191


namespace equation_of_line_with_x_intercept_and_slope_l1921_192138

theorem equation_of_line_with_x_intercept_and_slope :
  ∃ (a b c : ℝ), a * x - b * y + c = 0 ∧ a = 1 ∧ b = -1 ∧ c = 2 :=
sorry

end equation_of_line_with_x_intercept_and_slope_l1921_192138


namespace percentage_deficit_for_second_side_l1921_192134

theorem percentage_deficit_for_second_side
  (L W : ℝ) 
  (measured_first_side : ℝ := 1.12 * L) 
  (error_in_area : ℝ := 1.064) : 
  (∃ x : ℝ, (1.12 * L) * ((1 - 0.01 * x) * W) = 1.064 * (L * W) → x = 5) :=
by
  sorry

end percentage_deficit_for_second_side_l1921_192134


namespace angle_negative_225_in_second_quadrant_l1921_192159

def inSecondQuadrant (angle : Int) : Prop :=
  angle % 360 > -270 ∧ angle % 360 <= -180

theorem angle_negative_225_in_second_quadrant :
  inSecondQuadrant (-225) :=
by
  sorry

end angle_negative_225_in_second_quadrant_l1921_192159


namespace numDogsInPetStore_l1921_192190

-- Definitions from conditions
variables {D P : Nat}

-- Theorem statement - no proof provided
theorem numDogsInPetStore (h1 : D + P = 15) (h2 : 4 * D + 2 * P = 42) : D = 6 :=
by
  sorry

end numDogsInPetStore_l1921_192190


namespace maria_sold_in_first_hour_l1921_192178

variable (x : ℕ)

-- Conditions
def sold_in_first_hour := x
def sold_in_second_hour := 2
def average_sold_in_two_hours := 6

-- Proof Goal
theorem maria_sold_in_first_hour :
  (sold_in_first_hour + sold_in_second_hour) / 2 = average_sold_in_two_hours → sold_in_first_hour = 10 :=
by
  sorry

end maria_sold_in_first_hour_l1921_192178


namespace hash_op_8_4_l1921_192172

def hash_op (a b : ℕ) : ℕ := a + a / b - 2

theorem hash_op_8_4 : hash_op 8 4 = 8 := 
by 
  -- The proof is left as an exercise, indicated by sorry.
  sorry

end hash_op_8_4_l1921_192172


namespace tina_jumps_more_than_cindy_l1921_192127

def cindy_jumps : ℕ := 12
def betsy_jumps : ℕ := cindy_jumps / 2
def tina_jumps : ℕ := betsy_jumps * 3

theorem tina_jumps_more_than_cindy : tina_jumps - cindy_jumps = 6 := by
  sorry

end tina_jumps_more_than_cindy_l1921_192127


namespace intersection_complement_l1921_192187

open Set

-- Definitions from the problem
def U : Set ℝ := univ
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {y | 0 < y}

-- The proof statement
theorem intersection_complement : A ∩ (compl B) = Ioc (-1 : ℝ) 0 := by
  sorry

end intersection_complement_l1921_192187


namespace gcd_factorial_8_6_squared_l1921_192161

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_factorial_8_6_squared :
  Nat.gcd (factorial 8) ((factorial 6) ^ 2) = 7200 :=
by
  sorry

end gcd_factorial_8_6_squared_l1921_192161


namespace correct_average_weight_l1921_192192

-- Definitions
def initial_average_weight : ℝ := 58.4
def number_of_boys : ℕ := 20
def misread_weight_initial : ℝ := 56
def misread_weight_correct : ℝ := 68

-- Correct average weight
theorem correct_average_weight : 
  let initial_total_weight := initial_average_weight * (number_of_boys : ℝ)
  let difference := misread_weight_correct - misread_weight_initial
  let correct_total_weight := initial_total_weight + difference
  let correct_average_weight := correct_total_weight / (number_of_boys : ℝ)
  correct_average_weight = 59 :=
by
  -- Insert the proof steps if needed
  sorry

end correct_average_weight_l1921_192192


namespace solution_set_inequality_l1921_192163

theorem solution_set_inequality {a b c : ℝ} (h₁ : a < 0)
  (h₂ : ∀ x : ℝ, (a * x^2 + b * x + c <= 0) ↔ (x <= -(1/3) ∨ 2 <= x)) :
  (∀ x : ℝ, (c * x^2 + b * x + a > 0) ↔ (x < -3 ∨ 1/2 < x)) :=
by
  sorry

end solution_set_inequality_l1921_192163


namespace cost_of_blue_hat_is_six_l1921_192114

-- Given conditions
def total_hats : ℕ := 85
def green_hats : ℕ := 40
def blue_hats : ℕ := total_hats - green_hats
def cost_green_hat : ℕ := 7
def total_cost : ℕ := 550
def total_cost_green_hats : ℕ := green_hats * cost_green_hat
def total_cost_blue_hats : ℕ := total_cost - total_cost_green_hats
def cost_blue_hat : ℕ := total_cost_blue_hats / blue_hats

-- Proof statement
theorem cost_of_blue_hat_is_six : cost_blue_hat = 6 := sorry

end cost_of_blue_hat_is_six_l1921_192114


namespace ellipse_major_axis_focal_distance_l1921_192115

theorem ellipse_major_axis_focal_distance (m : ℝ) (h1 : 10 - m > 0) (h2 : m - 2 > 0) 
  (h3 : ∀ x y, x^2 / (10 - m) + y^2 / (m - 2) = 1) 
  (h4 : ∃ c, 2 * c = 4 ∧ c^2 = (m - 2) - (10 - m)) : m = 8 :=
by
  sorry

end ellipse_major_axis_focal_distance_l1921_192115


namespace desired_value_l1921_192176

noncomputable def find_sum (a b c : ℝ) (p q r : ℝ) : ℝ :=
  a / p + b / q + c / r

theorem desired_value (a b c : ℝ) (h1 : p = a / 2) (h2 : q = b / 2) (h3 : r = c / 2) :
  find_sum a b c p q r = 6 :=
by
  sorry

end desired_value_l1921_192176


namespace parabola_exists_l1921_192183

noncomputable def parabola_conditions (a b : ℝ) : Prop :=
  (a + b = -3) ∧ (4 * a - 2 * b = 12)

noncomputable def translated_min_equals_six (m : ℝ) : Prop :=
  (m > 0) ∧ ((-1 - 2 + m)^2 - 3 = 6) ∨ ((3 - 2 - m)^2 - 3 = 6)

theorem parabola_exists (a b m : ℝ) (x y : ℝ) :
  parabola_conditions a b → y = x^2 + b * x + 1 → translated_min_equals_six m →
  (y = x^2 - 4 * x + 1) ∧ (m = 6 ∨ m = 4) := 
by 
  sorry

end parabola_exists_l1921_192183


namespace greatest_three_digit_multiple_of_17_l1921_192140

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, n < 1000 ∧ n ≥ 100 ∧ (n % 17 = 0) ∧ ∀ m : ℕ, m < 1000 ∧ m ≥ 100 ∧ (m % 17 = 0) → m ≤ n := by
  sorry

end greatest_three_digit_multiple_of_17_l1921_192140


namespace carol_rectangle_width_l1921_192131

def carol_width (lengthC : ℕ) (widthJ : ℕ) (lengthJ : ℕ) (widthC : ℕ) :=
  lengthC * widthC = lengthJ * widthJ

theorem carol_rectangle_width 
  {lengthC widthJ lengthJ : ℕ} (h1 : lengthC = 8)
  (h2 : widthJ = 30) (h3 : lengthJ = 4)
  (h4 : carol_width lengthC widthJ lengthJ 15) : 
  widthC = 15 :=
by 
  subst h1
  subst h2
  subst h3
  sorry -- proof not required

end carol_rectangle_width_l1921_192131


namespace min_value_of_expression_l1921_192125

open Real

theorem min_value_of_expression {a b c d e f : ℝ} (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f)
    (h_sum : a + b + c + d + e + f = 10) :
    (∃ x, x = 44.1 ∧ ∀ y, y = 1 / a + 4 / b + 9 / c + 16 / d + 25 / e + 36 / f → x ≤ y) :=
sorry

end min_value_of_expression_l1921_192125


namespace initial_visual_range_is_90_l1921_192133

-- Define the initial visual range without the telescope (V).
variable (V : ℝ)

-- Define the condition that the visual range with the telescope is 150 km.
variable (condition1 : V + (2 / 3) * V = 150)

-- Define the proof problem statement.
theorem initial_visual_range_is_90 (V : ℝ) (condition1 : V + (2 / 3) * V = 150) : V = 90 :=
sorry

end initial_visual_range_is_90_l1921_192133


namespace vector_addition_l1921_192182

def a : ℝ × ℝ := (2, -1)
def b : ℝ × ℝ := (-3, 4)

theorem vector_addition :
  2 • a + b = (1, 2) :=
by
  sorry

end vector_addition_l1921_192182


namespace tan_315_eq_neg_1_l1921_192108

theorem tan_315_eq_neg_1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_1_l1921_192108


namespace solve_congruence_l1921_192179

theorem solve_congruence :
  ∃ a m : ℕ, (8 * (x : ℕ) + 1) % 12 = 5 % 12 ∧ m ≥ 2 ∧ a < m ∧ x ≡ a [MOD m] ∧ a + m = 5 :=
by
  sorry

end solve_congruence_l1921_192179


namespace gcd_2pow_2025_minus_1_2pow_2016_minus_1_l1921_192151

theorem gcd_2pow_2025_minus_1_2pow_2016_minus_1 :
  Nat.gcd (2^2025 - 1) (2^2016 - 1) = 511 :=
by sorry

end gcd_2pow_2025_minus_1_2pow_2016_minus_1_l1921_192151
