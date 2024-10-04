import Mathlib

namespace at_least_93_are_one_l775_775986

theorem at_least_93_are_one (a : Fin 100 → ℕ) (hpos : ∀ i, a i > 0) (h_sum_prod : ∑ i, a i = ∏ i, a i) : 
  ∃ (S : Finset (Fin 100)), S.card ≥ 93 ∧ ∀ i ∈ S, a i = 1 :=
sorry

end at_least_93_are_one_l775_775986


namespace sum_of_exponents_l775_775222

noncomputable def uniqueSum (s : ℕ) (m : ℕ → ℕ) (b : ℕ → ℤ) : Prop :=
  (∀ i j, i ≠ j → m i ≠ m j) ∧
  (∀ k, b k = 1 ∨ b k = -1) ∧
  ∃ (hs : Finset ℕ), hs.card = s ∧ (∑ k in hs, b k * (3 ^ m k)) = 2023

theorem sum_of_exponents (s : ℕ) (m : ℕ → ℕ) (b : ℕ → ℤ)
    (h : uniqueSum s m b) : (∑ k in Finset.range s, m k) = 21 :=
by
  revert s m b h
  sorry

end sum_of_exponents_l775_775222


namespace smallest_hot_dog_packages_l775_775307

theorem smallest_hot_dog_packages (d : ℕ) (b : ℕ) (hd : d = 10) (hb : b = 15) :
  ∃ n : ℕ, n * d = m * b ∧ n = 3 :=
by
  sorry

end smallest_hot_dog_packages_l775_775307


namespace trajectory_ellipse_l775_775262

-- definition of the problem conditions
variables (F1 F2 M : Type) [metric_space F1] [metric_space F2] [metric_space M]
variable (distance : F1 → F2 → ℝ)

def fixed_points (F1 F2 : Type) [metric_space F1] [metric_space F2] := 
  distance (classical.arbitrary F1) (classical.arbitrary F2) = 6

def moving_point (M : Type) (F1 F2 : Type) [metric_space M] [metric_space F1] [metric_space F2]
  (distance : M → F1 → ℝ) (distance2 : M → F2 → ℝ) :=
  ∀ (m : M), distance m (classical.arbitrary F1) + distance2 m (classical.arbitrary F2) = 8

theorem trajectory_ellipse (F1 F2 M : Type) [metric_space F1] [metric_space F2] [metric_space M] 
  (distance : M → F1 → ℝ) (distance2 : M → F2 → ℝ) :
  fixed_points F1 F2 → moving_point M F1 F2 distance distance2 →
  ∃ (trajectory : Type), ∀ (m : M), trajectory = ellipse F1 F2 :=
sorry

end trajectory_ellipse_l775_775262


namespace tan_4410_is_undefined_l775_775309

theorem tan_4410_is_undefined : ∀ (θ : ℝ), θ = 4410 → real.tan θ = real.tan 90 := by
  intros θ hθ
  rw hθ
  ring
  have h : 4410 = 12 * 360 + 90 := by norm_num
  rw h
  rw real.tan_add_int_mul_two_pi
  norm_num
  sorry

end tan_4410_is_undefined_l775_775309


namespace find_larger_number_l775_775193

theorem find_larger_number (S L : ℕ) (h1 : L - S = 1365) (h2 : L = 6 * S + 10) : L = 1636 := 
by
  sorry

end find_larger_number_l775_775193


namespace length_of_QR_in_cube_l775_775604

theorem length_of_QR_in_cube (edge_length : ℝ) (h_edge : edge_length = 2) :
  let PS := real.sqrt (edge_length ^ 2 + edge_length ^ 2),
      QS := PS / 2,
      RS := edge_length
  in real.sqrt ((QS ^ 2) + (RS ^ 2)) = real.sqrt 6 :=
by {
  sorry
}

end length_of_QR_in_cube_l775_775604


namespace train_length_72kmphr_9sec_180m_l775_775249

/-- Given speed in km/hr and time in seconds, calculate the length of the train in meters -/
theorem train_length_72kmphr_9sec_180m : ∀ (speed_kmph : ℕ) (time_sec : ℕ),
  speed_kmph = 72 → time_sec = 9 → 
  (speed_kmph * 1000 / 3600) * time_sec = 180 :=
by
  intros speed_kmph time_sec h1 h2
  sorry

end train_length_72kmphr_9sec_180m_l775_775249


namespace min_distance_l775_775000

theorem min_distance (x y : ℝ) (h : 5 * x + 12 * y = 60) : sqrt (x^2 + y^2) = 60 / 13 :=
sorry

end min_distance_l775_775000


namespace minimum_value_R_l775_775011

def floor_div (m k : ℕ) : ℕ := m / k

def R (k : ℕ) : ℚ :=
  (1 + ∑ n in Finset.range 200, ite (floor_div n k + floor_div (200 - n) k = floor_div 200 k) 1 0) / 199

theorem minimum_value_R :
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ 199 → R 100 = 1 / 2 :=
by
  sorry

end minimum_value_R_l775_775011


namespace mass_percentage_of_P_in_AlPO4_is_25_40_l775_775346

-- Define the molar masses of the elements
def molar_mass_Al : ℝ := 26.98
def molar_mass_P : ℝ := 30.97
def molar_mass_O : ℝ := 16.00

-- Define the count of Oxygen atoms in AlPO4
def count_O_in_AlPO4 : ℝ := 4

-- Calculate the total molar mass of AlPO4
def molar_mass_AlPO4 : ℝ := molar_mass_Al + molar_mass_P + count_O_in_AlPO4 * molar_mass_O

-- Define the mass percentage formula
def mass_percentage_P_in_AlPO4 : ℝ := (molar_mass_P / molar_mass_AlPO4) * 100

-- Theorem to show the mass percentage of P in AlPO4
theorem mass_percentage_of_P_in_AlPO4_is_25_40 : mass_percentage_P_in_AlPO4 = 25.40 :=
by
  unfold molar_mass_Al molar_mass_P molar_mass_O count_O_in_AlPO4 molar_mass_AlPO4 mass_percentage_P_in_AlPO4
  rw [←div_eq_inv_mul, ←mul_assoc]
  norm_num
  sorry

end mass_percentage_of_P_in_AlPO4_is_25_40_l775_775346


namespace mean_equality_l775_775005

theorem mean_equality (x : ℤ) (h : (8 + 10 + 24) / 3 = (16 + x + 18) / 3) : x = 8 := by 
sorry

end mean_equality_l775_775005


namespace gcd_18_eq_6_l775_775727

theorem gcd_18_eq_6 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 200) : (nat.gcd 18 n = 6) ↔ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 22 ∧ n = 6 * k ∧ ¬(n = 18 * (n / 18))) :=
begin
  sorry
end

end gcd_18_eq_6_l775_775727


namespace no_infinite_layering_l775_775920

-- Definitions for the problem context
def Grid (n : ℕ) := fin n → fin n → bool
-- True represents a black cube, false represents a white cube.

-- Conditions for each cell
def valid_first_layer (g : Grid n) : Prop :=
∀ (i j : fin n), 
(g i j = true → even (count_adjacent_white g i j)) ∧ 
(g i j = false → odd (count_adjacent_black g i j))

def count_adjacent_white (g : Grid n) (i j : fin n) : ℕ :=
countb (λ (p : fin n × fin n), is_adjacent p (i, j) ∧ g p.1 p.2 = false) (adjacent_cells i j)

def count_adjacent_black (g : Grid n) (i j : fin n) : ℕ :=
countb (λ (p : fin n × fin n), is_adjacent p (i, j) ∧ g p.1 p.2 = true) (adjacent_cells i j)

def is_adjacent (p q : fin n × fin n) : Prop :=
(abs (int.of_nat p.1.val - int.of_nat q.1.val) ≤ 1) ∧
(abs (int.of_nat p.2.val - int.of_nat q.2.val) ≤ 1)

def adjacent_cells (i j : fin n) : finset (fin n × fin n) :=
{(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)}

-- Statement: there does not exist an arrangement such that the process never ends
theorem no_infinite_layering (n : ℕ) (g : Grid n) (H : valid_first_layer g) : 
¬ ∃ (layers : ℕ → Grid n), 
(layers 0 = g) ∧ 
(∀ k, valid_first_layer (layers k) → valid_first_layer (layers (k + 1))) ∧ 
(∀ m ≠ n, layers m ≠ layers n) :=
sorry

end no_infinite_layering_l775_775920


namespace part_I_part_II_part_III_l775_775759

def distinct_sums (P : Set ℝ) : ℕ :=
  Set.card {x + y | x ∈ P ∧ y ∈ P ∧ x < y}

theorem part_I : distinct_sums ({1, 3, 5, 7, 9} : Set ℝ) = 7 := sorry

theorem part_II (n : ℕ) (hn : n > 2) :
  distinct_sums ({4^i | i : ℕ ∧ i < n} : Set ℝ) = n * (n - 1) / 2 := sorry

theorem part_III (P : Set ℝ) (hP : ∀ x y ∈ P, x < y → x + y ∈ P) :
  ∃ n > 2, distinct_sums P ≥ 2 * n - 3 := sorry

end part_I_part_II_part_III_l775_775759


namespace polynomial_value_at_neg_one_l775_775148

theorem polynomial_value_at_neg_one (P : ℕ → ℕ) 
  (h0 : P 0 = 1)
  (h1 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ 2022 → P k = k) :
  P (-1) = 2022 := 
sorry

end polynomial_value_at_neg_one_l775_775148


namespace candy_last_days_l775_775709

-- Definitions based on conditions
def pieces_from_neighbors : ℕ := 5
def pieces_from_sister : ℕ := 13
def pieces_per_day : ℕ := 9

-- The total number of pieces of candy Emily received
def total_pieces : ℕ := pieces_from_neighbors + pieces_from_sister

-- Expected number of days the candy will last
def expected_days : ℕ := 2

-- Lean 4 theorem to prove the number of days
theorem candy_last_days (total_pieces = pieces_from_neighbors + pieces_from_sister) 
  (daily_eating = pieces_per_day) :
  total_pieces / pieces_per_day = expected_days := 
by 
  -- jumping to the interactive proof mode
  sorry

end candy_last_days_l775_775709


namespace domain_of_f_sqrt_frac_l775_775547

def domain_of_function (x : ℝ) : Prop :=
  (x + 3 ≥ 0) ∧ (x + 1 ≠ 0)

theorem domain_of_f_sqrt_frac :
  {x : ℝ | domain_of_function x} = {x : ℝ | x ≥ -3} \ {x : ℝ | x = -1} :=
by
  sorry

end domain_of_f_sqrt_frac_l775_775547


namespace selina_sold_shirts_l775_775528

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end selina_sold_shirts_l775_775528


namespace categorize_numbers_l775_775335

def numbers := [-18, -3/5, 0, 2023, -22/7, -0.142857, 95/100]

def positiveNumberSet := {2023, 95/100}
def negativeNumberSet := {-18, -3/5, -22/7, -0.142857}
def integerSet := {-18, 0, 2023}
def fractionSet := {-3/5, -22/7, -0.142857, 95/100}

theorem categorize_numbers :
  ∀ x ∈ numbers,
  (x ∈ positiveNumberSet ↔ x = 2023 ∨ x = 95/100)
  ∧ (x ∈ negativeNumberSet ↔ x = -18 ∨ x = -3/5 ∨ x = -22/7 ∨ x = -0.142857)
  ∧ (x ∈ integerSet ↔ x = -18 ∨ x = 0 ∨ x = 2023)
  ∧ (x ∈ fractionSet ↔ x = -3/5 ∨ x = -22/7 ∨ x = -0.142857 ∨ x = 95/100) :=
by 
  sorry

end categorize_numbers_l775_775335


namespace incorrect_conclusion_symmetry_l775_775785

/-- Given the function f(x) = sin(1/5 * x + 13/6 * π), we define another function g(x) as the
translated function of f rightward by 10/3 * π units. We need to show that the graph of g(x)
is not symmetrical about the line x = π/4. -/
theorem incorrect_conclusion_symmetry (f g : ℝ → ℝ)
  (h₁ : ∀ x, f x = Real.sin (1/5 * x + 13/6 * Real.pi))
  (h₂ : ∀ x, g x = f (x - 10/3 * Real.pi)) :
  ¬ (∀ x, g (2 * (Real.pi / 4) - x) = g x) :=
sorry

end incorrect_conclusion_symmetry_l775_775785


namespace problem_l775_775911

noncomputable theory

def sequence (a : ℕ → ℝ) (a1 : ℝ) : Prop :=
  a 2 = 4 * a1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) = (a n) ^ 2 + 2 * (a n) 

def log_sequence (a : ℕ → ℝ) : ℕ → ℝ :=
  λ n, Real.logBase 3 (1 + a n)

def geometric_sequence (u : ℕ → ℝ) (r b : ℝ) : Prop :=
  u 0 = b ∧ ∀ n ≥ 0, u (n + 1) = r * u n

def sum_log_sequence (u : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, u i

theorem problem (a : ℕ → ℝ) (a1 : ℝ) (h_pos : ∀ n, 0 < a n) 
  (h_seq : sequence a a1) :
  (geometric_sequence (log_sequence a) 2 1) ∧
  ∃ n : ℕ, n ≥ 10 ∧ (sum_log_sequence (log_sequence a) n) > 520 :=
by
  sorry

end problem_l775_775911


namespace sum_two_digit_integers_ends_with_36_l775_775587

/-!
  Prove that the sum of all two-digit positive integers whose squares end with the digits 36 is 130.
-/

def is_two_digit_integer (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def ends_with_36 (n : ℕ) : Prop :=
  (n * n) % 100 = 36

theorem sum_two_digit_integers_ends_with_36 :
  (∑ n in finset.filter (λ n, is_two_digit_integer n ∧ ends_with_36 n) (finset.range 100), n) = 130 :=
by {
  sorry
}

end sum_two_digit_integers_ends_with_36_l775_775587


namespace isosceles_triangle_FGH_l775_775941

-- Define the geometry of triangle and orthocenter
variables {A B C H F G M : Type} [AffineSpace ℝ (Point A)] [AffineSpace ℝ (Point B)]
  [AffineSpace ℝ (Point C)] [AffineSpace ℝ (Point H)] [AffineSpace ℝ (Point F)] 
  [AffineSpace ℝ (Point G)] [AffineSpace ℝ (Point M)]

-- Define the conditions for the acute-angled scalene triangle and orthocenter intersecting the angle bisectors
axiom acute_scalene (ABC : Triangle) (ha : acute ABC) (hs : scalene ABC) : Prop
axiom orthocenter_int (H : Point) (A C : Point) (F G : Point) : Prop
axiom altitudes_angle_bisector (ABC : Triangle) (F G : Point) : Prop

-- Statement to prove
theorem isosceles_triangle_FGH 
  (ABC : Triangle) (H : Point) (A C : Point) (F G : Point) 
  (h_acu : acute_scalene ABC ha hs) 
  (h_orth : orthocenter_int H A C F G) 
  (h_alt_bis : altitudes_angle_bisector ABC F G) : 
  isosceles (Triangle.mk F G H) := 
sorry

end isosceles_triangle_FGH_l775_775941


namespace complex_expression_evaluation_l775_775246

theorem complex_expression_evaluation :
  1.047 * ((sqrt(561^2 - 459^2) / (4 * 2 / 7 * 0.15 + (4 * 2 / 7) / (20 / 3)) + 4 * sqrt(10)) / (1 / 3 * sqrt(40))) = 125 := 
sorry

end complex_expression_evaluation_l775_775246


namespace magnitude_vector_sum_l775_775063

variable {α : Type*} [InnerProductSpace ℝ α]

theorem magnitude_vector_sum (a b : α) (h₀ : ⟪a, b⟫ = 0) (ha : ∥a∥ = 2) (hb : ∥b∥ = 1) :
  ∥a + 2 • b∥ = 2 * Real.sqrt 2 :=
by
  sorry

end magnitude_vector_sum_l775_775063


namespace find_digits_l775_775341

/-- 
  Find distinct digits A, B, C, and D such that 9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B).
 -/
theorem find_digits
  (A B C D : ℕ)
  (hA : A ≠ B) (hA : A ≠ C) (hA : A ≠ D)
  (hB : B ≠ C) (hB : B ≠ D)
  (hC : C ≠ D)
  (hNonZeroB : B ≠ 0) :
  9 * (100 * A + 10 * B + C) = B * (1000 * B + 100 * C + 10 * D + B) ↔ (A = 2 ∧ B = 1 ∧ C = 9 ∧ D = 7) := by
  sorry

end find_digits_l775_775341


namespace percentage_passed_eng_students_l775_775845

variable (total_male_students : ℕ := 120)
variable (total_female_students : ℕ := 100)
variable (total_international_students : ℕ := 70)
variable (total_disabilities_students : ℕ := 30)

variable (male_eng_percentage : ℕ := 25)
variable (female_eng_percentage : ℕ := 20)
variable (intern_eng_percentage : ℕ := 15)
variable (disab_eng_percentage : ℕ := 10)

variable (male_pass_percentage : ℕ := 20)
variable (female_pass_percentage : ℕ := 25)
variable (intern_pass_percentage : ℕ := 30)
variable (disab_pass_percentage : ℕ := 35)

def total_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100) +
  (total_female_students * female_eng_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100)

def total_passed_engineering_students : ℕ :=
  (total_male_students * male_eng_percentage / 100 * male_pass_percentage / 100) +
  (total_female_students * female_eng_percentage / 100 * female_pass_percentage / 100) +
  (total_international_students * intern_eng_percentage / 100 * intern_pass_percentage / 100) +
  (total_disabilities_students * disab_eng_percentage / 100 * disab_pass_percentage / 100)

def passed_eng_students_percentage : ℕ :=
  total_passed_engineering_students * 100 / total_engineering_students

theorem percentage_passed_eng_students :
  passed_eng_students_percentage = 23 :=
sorry

end percentage_passed_eng_students_l775_775845


namespace prob_part1_prob_part2_l775_775500

-- Define the probability that Person A hits the target
def pA : ℚ := 2 / 3

-- Define the probability that Person B hits the target
def pB : ℚ := 3 / 4

-- Define the number of shots
def nShotsA : ℕ := 3
def nShotsB : ℕ := 2

-- The problem posed to Person A
def probA_miss_at_least_once : ℚ := 1 - (pA ^ nShotsA)

-- The problem posed to Person A (exactly twice in 2 shots)
def probA_hits_exactly_twice : ℚ := pA ^ 2

-- The problem posed to Person B (exactly once in 2 shots)
def probB_hits_exactly_once : ℚ :=
  2 * (pB * (1 - pB))

-- The combined probability for Part 2
def combined_prob : ℚ := probA_hits_exactly_twice * probB_hits_exactly_once

theorem prob_part1 :
  probA_miss_at_least_once = 19 / 27 := by
  sorry

theorem prob_part2 :
  combined_prob = 1 / 6 := by
  sorry

end prob_part1_prob_part2_l775_775500


namespace find_sum_fusion_number_l775_775083

def sum_fusion_number (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 4 * (2 * k + 1)

theorem find_sum_fusion_number (n : ℕ) :
  n = 2020 ↔ sum_fusion_number n :=
sorry

end find_sum_fusion_number_l775_775083


namespace find_a_if_parallel_l775_775408

-- Definitions of the vectors and the scalar a
def vector_m : ℝ × ℝ := (2, 1)
def vector_n (a : ℝ) : ℝ × ℝ := (4, a)

-- Condition for parallel vectors
def are_parallel (m n : ℝ × ℝ) : Prop :=
  m.1 / n.1 = m.2 / n.2

-- Lean 4 statement
theorem find_a_if_parallel (a : ℝ) (h : are_parallel vector_m (vector_n a)) : a = 2 :=
by
  sorry

end find_a_if_parallel_l775_775408


namespace cheryl_material_used_l775_775250

noncomputable def total_material_needed : ℚ :=
  (5 / 11) + (2 / 3)

noncomputable def material_left : ℚ :=
  25 / 55

noncomputable def material_used : ℚ :=
  total_material_needed - material_left

theorem cheryl_material_used :
  material_used = 22 / 33 :=
by
  sorry

end cheryl_material_used_l775_775250


namespace range_of_x_for_inequality_l775_775784

noncomputable def f (x : ℝ) := log x / (-1 : ℝ) * exp(1 : ℝ) (x ^ 2 + 1 / exp(1 : ℝ)) - x / exp(1)

theorem range_of_x_for_inequality :
  ∀ x : ℝ, (1 / 2 < x ∧ x < 2) → (0 < x) ∧ f(x + 1) < f(2 * x - 1) :=
by
  sorry

end range_of_x_for_inequality_l775_775784


namespace parallelogram_AB_length_l775_775748

open EuclideanGeometry

/-- Given a parallelogram ABCD such that BD = 2 and 2(AD ⋅ AB) = |BC|^2, 
    then |AB| = 2. --/
theorem parallelogram_AB_length (A B C D : Point)
  (h_parallelogram : is_parallelogram A B C D)
  (h_BD : dist B D = 2)
  (h_AD_AB : 2 * (vector.dot (A - D) (A - B)) = real.norm_sq (B - C)) :
  dist A B = 2 := by
  sorry

end parallelogram_AB_length_l775_775748


namespace remainder_of_polynomial_l775_775696

-- Define the polynomial and the divisor
def f (x : ℝ) := x^3 - 4 * x + 6
def a := -3

-- State the theorem
theorem remainder_of_polynomial :
  f a = -9 := by
  sorry

end remainder_of_polynomial_l775_775696


namespace range_of_a_l775_775552

theorem range_of_a (a : ℝ) : (∀ x1 x2 ∈ Icc (2 : ℝ) 4, (x1 ≤ x2 → -(x1^2) + 2*(a-1)*x1 + 2 ≤ -(x2^2) + 2*(a-1)*x2 + 2) ∨ (x1 ≤ x2 → -(x1^2) + 2*(a-1)*x1 + 2 ≥ -(x2^2) + 2*(a-1)*x2 + 2)) → (a ≤ 3 ∨ a ≥ 5) :=
sorry

end range_of_a_l775_775552


namespace selina_sells_5_shirts_l775_775526

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end selina_sells_5_shirts_l775_775526


namespace base10_to_base4_156_eq_2130_l775_775577

def base10ToBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem base10_to_base4_156_eq_2130 :
  base10ToBase4 156 = [2, 1, 3, 0] := sorry

end base10_to_base4_156_eq_2130_l775_775577


namespace max_sum_select_seven_numbers_l775_775257

theorem max_sum_select_seven_numbers :
  let grid := 
    [[1, 2, 3, 4, 5, 6, 7],
     [8, 9, 10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19, 20, 21],
     [22, 23, 24, 25, 26, 27, 28],
     [29, 30, 31, 32, 33, 34, 35],
     [36, 37, 38, 39, 40, 41, 42],
     [43, 44, 45, 46, 47, 48, 49]]
  in
  let selected_numbers := [49, 43, 37, 31, 25, 19, 13]
  in
  (selected_numbers.sum = 217) :=
begin
  let grid := 
    [[1, 2, 3, 4, 5, 6, 7],
     [8, 9, 10, 11, 12, 13, 14],
     [15, 16, 17, 18, 19, 20, 21],
     [22, 23, 24, 25, 26, 27, 28],
     [29, 30, 31, 32, 33, 34, 35],
     [36, 37, 38, 39, 40, 41, 42],
     [43, 44, 45, 46, 47, 48, 49]],
  let selected_numbers := [49, 43, 37, 31, 25, 19, 13],
  show (selected_numbers.sum = 217),
  by sorry
end

end max_sum_select_seven_numbers_l775_775257


namespace flag_height_l775_775651

theorem flag_height
  (d1 d2 d3 : ℕ × ℕ)
  (length : ℕ)
  (area1 : d1 = (8, 5))
  (area2 : d2 = (10, 7))
  (area3 : d3 = (5, 5))
  (flag_length : length = 15)
  : (d1.1 * d1.2 + d2.1 * d2.2 + d3.1 * d3.2) / length = 9 :=
by
  have h1 : d1.1 * d1.2 = 8 * 5, from congr_arg (*) area1,
  have h2 : d2.1 * d2.2 = 10 * 7, from congr_arg (*) area2,
  have h3 : d3.1 * d3.2 = 5 * 5, from congr_arg (*) area3,
  simp [flag_length] at *,
  sorry

end flag_height_l775_775651


namespace valid_subsets_equal_128_l775_775068

open Finset

noncomputable def count_valid_subsets : ℕ :=
  let S := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
  (S.powerset - {∅}).filter (λ T, 
    (∀ x ∈ T, ∀ y ∈ T, x ≠ y → (x = y + 1 → false) ∧ 
    (∃ (k : ℕ), T.card = k ∧ ∀ z ∈ T, z ≥ k))
  ).card

theorem valid_subsets_equal_128 : count_valid_subsets = 128 :=
sorry

end valid_subsets_equal_128_l775_775068


namespace jane_change_l775_775871

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l775_775871


namespace math_problem_l775_775179

noncomputable def sequence (x₁ : ℝ) : ℕ → ℝ
| 0       := x₁
| (n + 1) := 1 - 1 / sequence x₁ n

theorem math_problem
  (h1 : sequence 3 7 = 2 / 3)
  (h2 : (∀ n, sequence (-1) 1 n + sequence (-1) 2 n + sequence (-1) 3 n ≠ 1011))
  (h3 : ∀ x₁ x₂, (x₁ - 1) * (2 * x₂ + real.sqrt 2) * sequence x₁ 8 = -2 → x₁ = real.sqrt 2)
  : 2 = 2 :=
begin
  sorry
end

end math_problem_l775_775179


namespace probability_first_two_red_third_spade_l775_775288

/-- A standard deck of cards with 52 cards divided into four suits of 13 cards each.
Two of the suits are red (hearts and diamonds), and two are black (spades and clubs).
The cards are shuffled into a random order. -/
def standard_deck : Type := sorry

/-- Probability that the first two cards drawn are red and the third card is a spade. -/
theorem probability_first_two_red_third_spade (deck : standard_deck) :
    probability (draw_two_red_then_spade deck) = 13 / 204 := sorry

end probability_first_two_red_third_spade_l775_775288


namespace inequality_cannot_hold_l775_775740

noncomputable def f (a b c x : ℝ) := a * x ^ 2 + b * x + c

theorem inequality_cannot_hold
  (a b c : ℝ)
  (h_symm : ∀ x, f a b c x = f a b c (2 - x)) :
  ¬ (f a b c (1 - a) < f a b c (1 - 2 * a) ∧ f a b c (1 - 2 * a) < f a b c 1) :=
by {
  sorry
}

end inequality_cannot_hold_l775_775740


namespace num_ordered_pairs_satisfying_condition_l775_775912

def S := {0, 1, 2, 3, 4}

def A_i_plus_A_j (i j : ℕ) : ℕ :=
  |i - j|

theorem num_ordered_pairs_satisfying_condition :
  (∑ i in S, ∑ j in S, if (A_i_plus_A_j (A_i_plus_A_j i j) 2) = 1 then 1 else 0) = 12 := sorry

end num_ordered_pairs_satisfying_condition_l775_775912


namespace compute_x_y_div_2_l775_775081

theorem compute_x_y_div_2 (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h1 : Real.logBase y x + Real.logBase x y = 11 / 3)
    (h2 : x * y = 169) : (x + y) / 2 = 7 * Real.sqrt 13 := by
  sorry

end compute_x_y_div_2_l775_775081


namespace darts_game_score_l775_775159

variable (S1 S2 S3 : ℕ)
variable (n : ℕ)

theorem darts_game_score :
  n = 8 →
  S2 = 2 * S1 →
  S3 = (3 * S1) →
  S2 = 48 :=
by
  intros h1 h2 h3
  sorry

end darts_game_score_l775_775159


namespace find_a_l775_775027

/- Definitions -/
def f (a : ℝ) (x : ℝ) : ℝ :=
if x < 1 then 2 * x + a else -x - 2 * a

/- Theorem Statement -/
theorem find_a (a : ℝ) (h : a ≠ 0) (h_eq : f a (1 - a) = f a (1 + a)) : a = -3 / 4 :=
by
  /- Proof goes here -/
  sorry

end find_a_l775_775027


namespace find_integer_modulo_l775_775344

theorem find_integer_modulo : ∃ n : ℕ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ 123456 [MOD 11] := by
  use 3
  sorry

end find_integer_modulo_l775_775344


namespace masha_cards_extra_l775_775156

noncomputable def can_be_given_to_petya (cards : Finset ℕ) (numerators : Finset ℕ) : Finset ℕ :=
  cards \ numerators

theorem masha_cards_extra (cards : Finset ℕ) (numerators : Finset ℕ) :
  cards = \{1, 2, 3, 4, 5, 6, 7, 8, 9\} →
  numerators = \{1, 6, 9, 8\} →
  can_be_given_to_petya cards numerators = \{5, 7\} :=
by
  intros h_cards h_numerators
  rw [h_cards, h_numerators]
  sorry

end masha_cards_extra_l775_775156


namespace optionC_is_only_quadratic_l775_775242

-- Define a quadratic equation in one variable
def is_quadratic_equation_in_one_variable (eq : Expr) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ eq = a * x^2 + b * x + c = 0

-- Define the candidate equations
def optionA := 2 * x + 1 = 0
def optionB := x^2 + y = 1
def optionC := x^2 + 2 * x + 1 = 0
def optionD := x^2 + 1 / x = 1

-- Theorem stating that option C is the only quadratic equation in one variable
theorem optionC_is_only_quadratic :
  is_quadratic_equation_in_one_variable optionC ∧
  ¬ is_quadratic_equation_in_one_variable optionA ∧
  ¬ is_quadratic_equation_in_one_variable optionB ∧
  ¬ is_quadratic_equation_in_one_variable optionD :=
sorry

end optionC_is_only_quadratic_l775_775242


namespace increasing_interval_of_f_l775_775958

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

theorem increasing_interval_of_f :
  ∃ a b : ℝ, (1 < a ∧ ∃ k, a < k ∧ (∀ x : ℝ, k < x → differentiable_at ℝ f x ∧ derivate f x > 0)) :=
sorry

end increasing_interval_of_f_l775_775958


namespace number_of_folded_strips_is_odd_l775_775199

theorem number_of_folded_strips_is_odd :
  let cube_face_size := 9
  let strips := 2
  let total_faces := 6
  ∀ (faces : Fin cube_face_size × Fin cube_face_size → Prop)
    (strips_cover : ∀ (s : Fin strips), Fin cube_face_size × Fin cube_face_size → Prop),
  (∃ k : ℕ, ∃ odd_check : Fin 2, 
      odd_check * k = Fin (total_faces * cube_face_size * cube_face_size) 
      ∧ ∀ (f : Fin total_faces), 
            (∑ b in (Fin cube_face_size × Fin cube_face_size), faces b = 41) → 
            ∀ (str : Fin strips), strips_cover str (cube_face_size, cube_face_size)) 
  → k % 2 = 1 :=
begin
  sorry
end

end number_of_folded_strips_is_odd_l775_775199


namespace circle_line_distance_condition_l775_775830

theorem circle_line_distance_condition :
  ∀ (c : ℝ), 
    (∃ (x y : ℝ), x^2 + y^2 - 4*x - 4*y - 8 = 0 ∧ (x - y + c = 2 ∨ x - y + c = -2)) →
    -2*Real.sqrt 2 ≤ c ∧ c ≤ 2*Real.sqrt 2 := 
sorry

end circle_line_distance_condition_l775_775830


namespace circle_center_and_radius_l775_775777

theorem circle_center_and_radius :
  ∃ (center : ℝ × ℝ) (radius : ℝ),
  let circle := {P : ℝ × ℝ | (P.1)^2 + (P.2)^2 + P.1 - 6 * P.2 + 3 = 0},
      line := {P : ℝ × ℝ | P.1 + 2 * P.2 - 3 = 0},
      O := (0, 0 : ℝ),
      P Q : ℝ × ℝ,
      P_in_circle : P ∈ circle,
      Q_in_circle : Q ∈ circle,
      P_in_line : P ∈ line,
      Q_in_line : Q ∈ line,
      O_perp_PQ : P.1 * Q.1 + P.2 * Q.2 = 0
  in center = (-1/2, 3) ∧ radius = 5/2 := sorry

end circle_center_and_radius_l775_775777


namespace roots_sum_and_product_l775_775404

-- Given the quadratic equation x^2 - 3x + 1 = 0 with roots x1 and x2, 
-- we need to prove that x1 + x2 - x1 * x2 = 2.
theorem roots_sum_and_product (x1 x2 : ℝ) (h1 : x1 * x1 - 3 * x1 + 1 = 0) (h2 : x2 * x2 - 3 * x2 + 1 = 0) :
  x1 + x2 - x1 * x2 = 2 :=
sorry

end roots_sum_and_product_l775_775404


namespace pencils_more_than_200_on_saturday_l775_775568

theorem pencils_more_than_200_on_saturday 
    (p : ℕ → ℕ) 
    (h_start : p 1 = 3)
    (h_next_day : ∀ n, p (n + 1) = (p n + 2) * 2) 
    : p 6 > 200 :=
by
  -- Proof steps can be filled in here.
  sorry

end pencils_more_than_200_on_saturday_l775_775568


namespace complex_real_eq_imag_l775_775827

theorem complex_real_eq_imag (a : ℝ) (h : (1 + a * Complex.i) / (2 - Complex.i)).re = ((1 + a * Complex.i) / (2 - Complex.i)).im) : 
  a = 1/3 := 
sorry

end complex_real_eq_imag_l775_775827


namespace largest_angle_of_triangle_l775_775953

theorem largest_angle_of_triangle 
  (α β γ : ℝ) 
  (h1 : α = 60) 
  (h2 : β = 70) 
  (h3 : α + β + γ = 180) : 
  max α (max β γ) = 70 := 
by 
  sorry

end largest_angle_of_triangle_l775_775953


namespace fraction_is_three_halves_l775_775070

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end fraction_is_three_halves_l775_775070


namespace diagonal_AC_length_l775_775850

noncomputable def length_diagonal_AC (AB BC CD DA : ℝ) (angle_ADC : ℝ) : ℝ :=
  (CD^2 + DA^2 - 2 * CD * DA * Real.cos angle_ADC).sqrt

theorem diagonal_AC_length :
  ∀ (AB BC CD DA : ℝ) (angle_ADC : ℝ),
  AB = 10 → BC = 10 → CD = 17 → DA = 17 → angle_ADC = 2 * Real.pi / 3 →
  length_diagonal_AC AB BC CD DA angle_ADC = Real.sqrt 867 :=
begin
  intros AB BC CD DA angle_ADC hAB hBC hCD hDA hangle_ADC,
  rw [hCD, hDA, hangle_ADC],
  sorry
end

end diagonal_AC_length_l775_775850


namespace parabola_equation_and_x_range_l775_775856

open Real

noncomputable def equation_of_parabola (p : ℝ) : ℝ :=
  y^2 = 2 * p * x

theorem parabola_equation_and_x_range :
  (∀ (F M : Point) (p : ℝ), p > 0 → F = (p/2, 0) → M ∈ (C : parabola, y^2 = 2 * p * x) 
    → distance(center_of_circumcircle(Δ OFM), axis_of_symmetry(C)) = 3/2
    → equation_of_parabola p = y^2 = 4x) ∧
  (∀ (K : Point)(λ ∈ Icc 2 3)(G : Point), 
    line_through K intersects (C : parabola, y^2 = 4x) at points A B 
    → (KA = λ * KB) 
    → (G ∈ x_axis) ∧ (distance(GA, GB) = GA - GB)
    → (x0 = 2*m^2 + 1) ∈ Icc (13/4) (11/3)) := 
 begin 
  sorry 
end

end parabola_equation_and_x_range_l775_775856


namespace smallest_x_for_g_l775_775631

noncomputable def g : ℝ → ℝ := sorry

axiom g_property_1 (x : ℝ) : x > 0 → g (4 * x) = 4 * g x

axiom g_property_2 (x : ℝ) : 2 ≤ x ∧ x ≤ 4 → g x = 1 - |x - 3|

theorem smallest_x_for_g :
  ∃ x : ℝ,  x > 0 ∧ (g x = g 2048) ∧ ∀ y : ℝ, (g y = g 2048) → y ≥ x :=
begin
  sorry
end

end smallest_x_for_g_l775_775631


namespace max_unique_rankings_l775_775926

theorem max_unique_rankings (n : ℕ) : 
  ∃ (contestants : ℕ), 
    (∀ (scores : ℕ → ℕ), 
      (∀ i, 0 ≤ scores i ∧ scores i ≤ contestants) ∧
      (∀ i j, i ≠ j → scores i ≠ scores j)) 
    → contestants = 2^n := 
sorry

end max_unique_rankings_l775_775926


namespace value_of_m_has_positive_root_l775_775426

theorem value_of_m_has_positive_root (x m : ℝ) (hx : x ≠ 3) :
    ((x + 5) / (x - 3) = 2 - m / (3 - x)) → x > 0 → m = 8 := 
sorry

end value_of_m_has_positive_root_l775_775426


namespace gcd_18_n_eq_6_in_range_l775_775725

theorem gcd_18_n_eq_6_in_range :
  {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 18 n = 6}.card = 22 :=
by
  -- To skip the proof
  sorry

end gcd_18_n_eq_6_in_range_l775_775725


namespace range_of_m_l775_775798

noncomputable def set_A : set ℝ := { x | -2 ≤ x ∧ x ≤ 7 }
noncomputable def set_B (m : ℝ) : set ℝ := { x | m - 1 ≤ x ∧ x ≤ 2 * m + 1 }

theorem range_of_m (m : ℝ) (h : (set_B m) ⊆ set_A) : m ≤ -2 ∨ (-1 ≤ m ∧ m ≤ 3) :=
by sorry

end range_of_m_l775_775798


namespace cone_radius_solution_l775_775963

noncomputable def cone_radius_problem (CSA : ℝ) (l : ℝ) : Prop :=
  let π := Real.pi in
  let r := CSA / (π * l) in
  r ≈ 8

theorem cone_radius_solution : cone_radius_problem 452.3893421169302 18 :=
by
  let CSA := 452.3893421169302
  let l := 18
  let π := Real.pi
  let r := CSA / (π * l)
  show r ≈ 8
  sorry

end cone_radius_solution_l775_775963


namespace diagonal_AC_length_l775_775851

noncomputable def length_diagonal_AC (AB BC CD DA : ℝ) (angle_ADC : ℝ) : ℝ :=
  (CD^2 + DA^2 - 2 * CD * DA * Real.cos angle_ADC).sqrt

theorem diagonal_AC_length :
  ∀ (AB BC CD DA : ℝ) (angle_ADC : ℝ),
  AB = 10 → BC = 10 → CD = 17 → DA = 17 → angle_ADC = 2 * Real.pi / 3 →
  length_diagonal_AC AB BC CD DA angle_ADC = Real.sqrt 867 :=
begin
  intros AB BC CD DA angle_ADC hAB hBC hCD hDA hangle_ADC,
  rw [hCD, hDA, hangle_ADC],
  sorry
end

end diagonal_AC_length_l775_775851


namespace convex_polygon_can_be_divided_l775_775513

-- Define convex polygon and the requirement for question
def convex_polygon (n : ℕ) := sorry -- To be defined, as defining convexity is complex and out of scope for this exercise.

-- Definition for being able to divide a polygon into convex pentagons
def can_be_divided_into_pentagons (P : Type) [convex_polygon P] : Prop := sorry -- Again, the detailed definition is complex

theorem convex_polygon_can_be_divided (n : ℕ) 
  (h₁ : n ≥ 6) 
  (P : Type) 
  [convex_polygon P] : 
  can_be_divided_into_pentagons P :=
sorry -- Proof goes here

end convex_polygon_can_be_divided_l775_775513


namespace count_congruent_to_2_mod_7_l775_775815

theorem count_congruent_to_2_mod_7 : 
  let count := (1 to 300).count (λ x, x % 7 = 2)
  count = 43 := by
  sorry

end count_congruent_to_2_mod_7_l775_775815


namespace quarters_range_difference_l775_775524

theorem quarters_range_difference (n d q : ℕ) (h1 : n + d + q = 150) (h2 : 5 * n + 10 * d + 25 * q = 2000) :
  let max_quarters := 0
  let min_quarters := 62
  (max_quarters - min_quarters) = 62 :=
by
  let max_quarters := 0
  let min_quarters := 62
  sorry

end quarters_range_difference_l775_775524


namespace sum_of_digits_10_pow_95_sub_97_l775_775323

-- Define the number 10^95 - 97
def big_number : ℕ := 10 ^ 95 - 97

-- Define a function to calculate the sum of the digits of a number
def sum_of_digits (n : ℕ) : ℕ :=
  n.toString.foldl (λ acc c, acc + (c.toNat - '0'.toNat)) 0

-- State the theorem
theorem sum_of_digits_10_pow_95_sub_97 : sum_of_digits big_number = 840 :=
by
  -- The proof is omitted
  sorry

end sum_of_digits_10_pow_95_sub_97_l775_775323


namespace hyperbola_parabola_foci_l775_775428

-- Definition of the hyperbola
def hyperbola (k : ℝ) (x y : ℝ) : Prop := y^2 / 5 - x^2 / k = 1

-- Definition of the parabola
def parabola (x y : ℝ) : Prop := x^2 = 12 * y

-- Condition that both curves have the same foci
def same_foci (focus : ℝ) (x y : ℝ) : Prop := focus = 3 ∧ (parabola x y → ((0, focus) : ℝ×ℝ) = (0, 3)) ∧ (∃ k : ℝ, hyperbola k x y ∧ ((0, focus) : ℝ×ℝ) = (0, 3))

theorem hyperbola_parabola_foci (k : ℝ) (x y : ℝ) : same_foci 3 x y → k = -4 := 
by {
  sorry
}

end hyperbola_parabola_foci_l775_775428


namespace convex_ngon_can_be_divided_l775_775510

open Convex

theorem convex_ngon_can_be_divided (n : ℕ) (h : n ≥ 6) :
  ∃ (S : finset (fin 5 → ℝ)) (S' : finset (fin 5 → ℝ)),
    (∀ s ∈ S, Convex ℝ (finset.image id s)) ∧ 
    (∀ s' ∈ S', Convex ℝ (finset.image id s')) ∧
    (Convex ℝ (⋃₀ (S ∪ S'))) ∧
    (finset.card S + finset.card S' = n) :=
by
  sorry

end convex_ngon_can_be_divided_l775_775510


namespace line_intersects_circle_range_of_a_l775_775090

theorem line_intersects_circle_range_of_a {a : ℝ} :
  (∃ x y : ℝ, (x - y + 1 = 0) ∧ ((x - a)^2 + y^2 = 2)) → a ∈ (set.Icc (-3 : ℝ) 1) :=
sorry

end line_intersects_circle_range_of_a_l775_775090


namespace limit_sum_sequence_l775_775900

def a (n : ℕ) : ℝ :=
if n = 1 then 2^(n-1)
else if n = 2 then 2^(n-1)
else 1 / 3^n

def S (n : ℕ) : ℝ :=
∑ i in finset.range n, a(i+1)

theorem limit_sum_sequence :
  filter.tendsto S filter.at_top (nhds (3 + 1/18)) :=
sorry

end limit_sum_sequence_l775_775900


namespace bridge_angles_sum_l775_775268

noncomputable def is_isosceles (A B C : Type) [metric_space A] (a b c : A) : Prop :=
dist a b = dist a c

noncomputable def angle (A B C : Type) [metric_space A] (a b c : A) : ℝ :=
sorry -- Definition of the angle measure

theorem bridge_angles_sum
  {A B C D E F : Type} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
  (a b c d e f : A)
  (isosceles_abc : is_isosceles A a b c)
  (isosceles_def : is_isosceles A d e f)
  (angle_bac : angle A b a c = 25)
  (angle_edf : angle A e d f = 35)
  (parallel_ad_ce : sorry) : -- need a proper definition for parallelism
  angle A d a c + angle A d e = 150 :=
sorry

end bridge_angles_sum_l775_775268


namespace range_of_x_l775_775780

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 2
  else Real.log (-x) / Real.log (1 / 2)

theorem range_of_x (x : ℝ) : f x > f (-x) ↔ (x > 1) ∨ (-1 < x ∧ x < 0) :=
by
  sorry

end range_of_x_l775_775780


namespace jane_received_change_l775_775874

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l775_775874


namespace num_ints_congruent_to_2_mod_7_l775_775813

theorem num_ints_congruent_to_2_mod_7 :
  ∃ n : ℕ, (∀ k, 1 ≤ 7 * k + 2 ∧ 7 * k + 2 ≤ 300 ↔ 0 ≤ k ≤ 42) ∧ n = 43 :=
sorry

end num_ints_congruent_to_2_mod_7_l775_775813


namespace toy_poodle_height_l775_775209

theorem toy_poodle_height (standard_poodle_height miniature_poodle_height toy_poodle_height : ℕ)
    (h1 : standard_poodle_height = miniature_poodle_height + 8)
    (h2 : miniature_poodle_height = toy_poodle_height + 6)
    (h3 : standard_poodle_height = 28) :
    toy_poodle_height = 14 :=
begin
  sorry
end

end toy_poodle_height_l775_775209


namespace ellipse_closer_to_circle_l775_775377

variables (a : ℝ)

-- Conditions: 1 < a < 2 + sqrt 5
def in_range_a (a : ℝ) : Prop := 1 < a ∧ a < 2 + Real.sqrt 5

-- Ellipse eccentricity should decrease as 'a' increases for the given range 1 < a < 2 + sqrt 5
theorem ellipse_closer_to_circle (h_range : in_range_a a) :
    ∃ b : ℝ, b = Real.sqrt (1 - (a^2 - 1) / (4 * a)) ∧ ∀ a', (1 < a' ∧ a' < 2 + Real.sqrt 5 ∧ a < a') → b > Real.sqrt (1 - (a'^2 - 1) / (4 * a')) := 
sorry

end ellipse_closer_to_circle_l775_775377


namespace number_of_arrangements_l775_775218

theorem number_of_arrangements (n : ℕ) (h : n = 5) :
  ∃ (m : ℕ), (exactly_one_person_between_A_and_B n) → m = 36 := by
  sorry

-- Define exactly_one_person_between_A_and_B
def exactly_one_person_between_A_and_B (n : ℕ) : Prop :=
  ∀ {a b : ℕ} (ha : a < n) (hb : b < n) (hab : a ≠ b) (p : list ℕ),
    p.length = n → (((p.take (a.succ)).drop a = p.take (b.succ)).drop b ≠ p)

end number_of_arrangements_l775_775218


namespace value_of_composed_operations_l775_775350

def op1 (x : ℝ) : ℝ := 9 - x
def op2 (x : ℝ) : ℝ := x - 9

theorem value_of_composed_operations : op2 (op1 15) = -15 :=
by
  sorry

end value_of_composed_operations_l775_775350


namespace collinear_vectors_l775_775763

open Real

theorem collinear_vectors (λ : ℝ) : 
  let A : ℝ × ℝ := (1, 1),
      B : ℝ × ℝ := (4, 2),
      a : ℝ × ℝ := (2, λ),
      AB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2) in
  a.1 * AB.2 = a.2 * AB.1 → λ = 2 / 3 := 
begin
  sorry
end

end collinear_vectors_l775_775763


namespace Natasha_avg_speed_climb_l775_775918

-- Definitions for conditions
def distance_to_top : ℝ := sorry -- We need to find this
def time_up := 3 -- time in hours to climb up
def time_down := 2 -- time in hours to climb down
def avg_speed_journey := 3 -- avg speed in km/hr for the whole journey

-- Equivalent math proof problem statement
theorem Natasha_avg_speed_climb (distance_to_top : ℝ) 
  (h1 : time_up = 3)
  (h2 : time_down = 2)
  (h3 : avg_speed_journey = 3)
  (h4 : (2 * distance_to_top) / (time_up + time_down) = avg_speed_journey) : 
  (distance_to_top / time_up) = 2.5 :=
sorry -- Proof not required

end Natasha_avg_speed_climb_l775_775918


namespace haley_trees_grown_after_typhoon_l775_775066

def original_trees := 9
def trees_died := 4
def current_trees := 10

theorem haley_trees_grown_after_typhoon (newly_grown_trees : ℕ) :
  (original_trees - trees_died) + newly_grown_trees = current_trees → newly_grown_trees = 5 :=
by
  sorry

end haley_trees_grown_after_typhoon_l775_775066


namespace find_x0_l775_775398

noncomputable def f (a c : ℝ) (x : ℝ) : ℝ := a * x^2 + c

theorem find_x0 (a c x0 : ℝ) (ha : a ≠ 0) (hx0 : 0 ≤ x0 ∧ x0 ≤ 1)
  (h_integral : ∫ x in 0..1, f a c x = f a c x0) : x0 = Real.sqrt 3 / 3 :=
by
  -- sorry represents the absence of proof
  sorry

end find_x0_l775_775398


namespace simple_interest_factor_l775_775834

theorem simple_interest_factor (P : ℝ) (R T : ℕ) (hR : R = 10) (hT : T = 10) :
  let SI := P * (R / 100) * T in
  let A := P + SI in
  A = 2 * P :=
by
  sorry

end simple_interest_factor_l775_775834


namespace new_class_mean_l775_775093

theorem new_class_mean {X Y : ℕ} {mean_a mean_b : ℚ}
  (hx : X = 30) (hy : Y = 6) 
  (hmean_a : mean_a = 72) (hmean_b : mean_b = 78) :
  (X * mean_a + Y * mean_b) / (X + Y) = 73 := 
by 
  sorry

end new_class_mean_l775_775093


namespace enclosed_region_area_l775_775146

noncomputable def g (x : ℝ) : ℝ := 1 - 2 * real.sqrt (1 - x^2)

theorem enclosed_region_area :
  ∀ f : ℝ → ℝ,
  f = g →
  ∃ area : ℝ, area = (Real.pi * Real.sqrt 3 / 8) ∧ area ≈ 0.68 :=
begin
  sorry
end

end enclosed_region_area_l775_775146


namespace log_relationship_l775_775769

theorem log_relationship (a b c: ℝ) (ha : a = Real.log 3 / Real.log 2)
  (hb : b = Real.log 4 / Real.log 3) (hc : c = Real.log 11 / (2 * Real.log 2)) :
  b < a ∧ a < c :=
by
  sorry

end log_relationship_l775_775769


namespace Ivan_can_safely_make_the_journey_l775_775978

def eruption_cycle_first_crater (t : ℕ) : Prop :=
  ∃ n : ℕ, t = 1 + 18 * n

def eruption_cycle_second_crater (t : ℕ) : Prop :=
  ∃ m : ℕ, t = 1 + 10 * m

def is_safe (start_time : ℕ) : Prop :=
  ∀ t, start_time ≤ t ∧ t < start_time + 16 → 
    ¬ eruption_cycle_first_crater t ∧ 
    ¬ (t ≥ start_time + 12 ∧ eruption_cycle_second_crater t)

theorem Ivan_can_safely_make_the_journey : ∃ t : ℕ, is_safe (38 + t) :=
sorry

end Ivan_can_safely_make_the_journey_l775_775978


namespace minimize_MN_length_l775_775163

-- Given conditions as definitions
variables {A O B M N : Type*} [metric_space O]
variables (OA OB : ℝ) -- Distances from O to A and B
variables (x : ℝ) -- Lengths AM and BN

-- Conditions
axiom OA_greater_OB : OA > OB
axiom AM_eq_BN : ∀ (M : O) (A : M) (N : O) (B : N), dist' O A M = dist' O B N

-- Prove minimum length of MN
theorem minimize_MN_length (OA OB : ℝ) (h : OA > OB) :
  (∃ x, x = (OA - OB) / 2) :=
begin
  sorry
end

end minimize_MN_length_l775_775163


namespace solve_fractional_equation_l775_775965

theorem solve_fractional_equation (x : ℝ) (hx : x ≠ 0) : (x + 1) / x = 2 / 3 ↔ x = -3 :=
by
  sorry

end solve_fractional_equation_l775_775965


namespace bucket_full_weight_l775_775997

variable (p q r : ℚ)
variable (x y : ℚ)

-- Define the conditions
def condition1 : Prop := p = r + (3 / 4) * y
def condition2 : Prop := q = r + (1 / 3) * y
def condition3 : Prop := x = r

-- Define the conclusion
def conclusion : Prop := x + y = (4 * p - r) / 3

-- The theorem stating that the conclusion follows from the conditions
theorem bucket_full_weight (h1 : condition1 p r y) (h2 : condition2 q r y) (h3 : condition3 x r) : conclusion x y p r :=
by
  sorry

end bucket_full_weight_l775_775997


namespace mutually_exclusive_complementary_event_l775_775282

-- Definitions of events
def hitting_target_at_least_once (shots: ℕ) : Prop := shots > 0
def not_hitting_target_at_all (shots: ℕ) : Prop := shots = 0

-- The statement to prove
theorem mutually_exclusive_complementary_event : 
  ∀ (shots: ℕ), (not_hitting_target_at_all shots ↔ ¬ hitting_target_at_least_once shots) :=
by 
  sorry

end mutually_exclusive_complementary_event_l775_775282


namespace problem_abs_diff_sum_div_eq_negative_l775_775366

theorem problem_abs_diff_sum_div_eq_negative:
  ∀ (a b : ℝ), abs a = 3 → abs b = 4 → a < b → (frac (a - b) (a + b) = -1/7 ∨ frac (a - b) (a + b) = -7) := by
  sorry

end problem_abs_diff_sum_div_eq_negative_l775_775366


namespace candles_must_be_odd_l775_775972

theorem candles_must_be_odd (n k : ℕ) (h : n * k = (n * (n + 1)) / 2) : n % 2 = 1 :=
by
  -- Given that the total burn time for all n candles = k * n
  -- And the sum of the first n natural numbers = (n * (n + 1)) / 2
  -- We have the hypothesis h: n * k = (n * (n + 1)) / 2
  -- We need to prove that n must be odd
  sorry

end candles_must_be_odd_l775_775972


namespace max_elements_is_50_l775_775672

def max_elements_in_S (S : Set ℕ) : Prop :=
  (∀ a ∈ S, a ≤ 100) ∧
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd (a + b) c = 1) ∧
  (∀ a b ∈ S, a ≠ b → ∃ c ∈ S, Nat.gcd (a + b) c > 1)

theorem max_elements_is_50 : ∃ S : Set ℕ, max_elements_in_S S ∧ S.size = 50 :=
sorry

end max_elements_is_50_l775_775672


namespace no_solution_for_inequality_l775_775530

theorem no_solution_for_inequality (x : ℝ) (h : |x| > 2) : ¬ (5 * x^2 + 6 * x + 8 < 0) := 
by
  sorry

end no_solution_for_inequality_l775_775530


namespace three_a_greater_three_b_l775_775764

variable (a b : ℝ)

theorem three_a_greater_three_b (h : a > b) : 3 * a > 3 * b :=
  sorry

end three_a_greater_three_b_l775_775764


namespace x_in_set_l775_775364

theorem x_in_set {x : ℝ} : x ∈ ({1, 2, x^2} : Set ℝ) → x ∈ ({0, 2} : Set ℝ) :=
by
  intro h
  sorry

end x_in_set_l775_775364


namespace pr_plus_one_not_divide_p_pow_p_minus_one_l775_775135

theorem pr_plus_one_not_divide_p_pow_p_minus_one {p r : ℕ} (hp : p.prime) (hp_odd : p % 2 = 1) (hr_odd : r % 2 = 1) :
  ¬ (p * r + 1 ∣ p ^ p - 1) :=
by
  sorry

end pr_plus_one_not_divide_p_pow_p_minus_one_l775_775135


namespace formation_of_H2O_l775_775687

noncomputable def amount_of_H2O (NaOH HCl : ℕ) : ℕ := 
  if NaOH = HCl then NaOH else min NaOH HCl

theorem formation_of_H2O (NaOH HCl : ℕ) : NaOH = 2 → HCl = 2 → amount_of_H2O NaOH HCl = 2 := by
  intros hNaOH hHCl
  rw [hNaOH, hHCl]
  simp [amount_of_H2O]
  sorry

end formation_of_H2O_l775_775687


namespace part_a_part_b_l775_775744

-- Assume n is a fixed integer >= 2
variable (n : ℕ) (h : n ≥ 2)
variable (x : Fin n → ℝ)
variable (x_nonneg : ∀ i : Fin n, 0 ≤ x i)

-- Define the sum in the inequality
def lhs : ℝ := 
  ∑ i in (Finset.range n).val.pairwise id, 
    let i₁ := i.1 in
    let i₂ := i.2 in
    x i₁ * x i₂ * (x i₁ ^ 2 + x i₂ ^ 2)

def rhs : ℝ := 
  (∑ i in Finset.finRange n, x i) ^ 4

-- smallest c value such that lhs ≤ c * rhs
theorem part_a : lhs x ≤ (1 / 8) * rhs x :=
sorry

-- Necessary and sufficient conditions for equality when c = 1 / 8
theorem part_b : (lhs x = (1 / 8) * rhs x) ↔ 
  (∃ i j : Fin n, i ≠ j ∧ x i = x j ∧ ∀ k : Fin n, k ≠ i ∧ k ≠ j → x k = 0) :=
sorry

end part_a_part_b_l775_775744


namespace min_distance_from_circle_to_line_l775_775057

open Real

def line_l (x y : ℝ) : Prop := x - y + 4 = 0
def circle_C (x y : ℝ) (θ : ℝ) : Prop := x = 1 + 2 * cos θ ∧ y = 1 + 2 * sin θ

theorem min_distance_from_circle_to_line :
  let center : ℝ × ℝ := (1, 1)
  let radius : ℝ := 2
  let distance_center_to_line : ℝ := |(1 - 1) - (1 - 1) + 4| / sqrt (1^2 + (-1)^2)
  min_distance := distance_center_to_line - radius
  min_distance = 2 * sqrt 2 - 2 :=
sorry

end min_distance_from_circle_to_line_l775_775057


namespace circumcenter_on_perpendicular_bisector_l775_775621

noncomputable def cyclic_quadrilateral (A B X C O D E : Type) : Prop :=
  ∃ (Q : cyclic_quadrilateral), Q.circumcenter = O ∧ 
  ∃ (D : line BX), |AD| = |BD| ∧
  ∃ (E : line CX), |AE| = |CE|

theorem circumcenter_on_perpendicular_bisector 
  {A B X C O D E : Type} 
  (hABXC : cyclic_quadrilateral A B X C O)
  (hD_on_BX : D ∈ line BX)
  (hE_on_CX : E ∈ line CX)
  (hAD_eq_BD : |AD| = |BD|)
  (hAE_eq_CE : |AE| = |CE|) : 
  let O' := circumcenter of triangle DEX in
  O' lies on the perpendicular bisector of OA :=
begin
  sorry
end

end circumcenter_on_perpendicular_bisector_l775_775621


namespace work_days_together_l775_775597

-- Conditions
variable {W : ℝ} (h_a_alone : ∀ (W : ℝ), W / a_work_time = W / 16)
variable {a_work_time : ℝ} (h_work_time_a : a_work_time = 16)

-- Question translated to proof problem
theorem work_days_together (D : ℝ) :
  (10 * (W / D) + 12 * (W / 16) = W) → D = 40 :=
by
  intros h
  have eq1 : 10 * (W / D) + 12 * (W / 16) = W := h
  sorry

end work_days_together_l775_775597


namespace normal_intersects_again_at_B_l775_775887

noncomputable def point := (ℚ, ℚ)

def A : point := (2, 4)

def parabola (x : ℚ) : ℚ := x^2

def is_on_parabola (p : point) : Prop := (parabola p.1) = p.2

def normal_slope (x : ℚ) : ℚ := - (1 / (2 * x))

def normal_line (p : point) (x : ℚ) : ℚ := - (1/(4 * p.1) * (x - p.1)) + p.2

theorem normal_intersects_again_at_B : 
  ∃ B : point,
    is_on_parabola A ∧
    is_on_parabola B ∧
    B ≠ A ∧
    ∀ x, x ≠ 2 → (parabola x) = (normal_line A x) ↔ x = -9/4 :=
  sorry

end normal_intersects_again_at_B_l775_775887


namespace tangent_circles_parallel_l775_775499

theorem tangent_circles_parallel {A B C D C1 A1 : Point} (h : Triangle A B C)
  (hD : OnLine D A C)
  (tangent_BDC : TangentAt D (Circumcircle B D C) C1)
  (tangent_ADB : TangentAt D (Circumcircle A D B) A1)
  (hC1 : OnLine C1 A B)
  (hA1 : OnLine A1 B C) :
  Parallel A1 C1 A C :=
by 
  sorry

end tangent_circles_parallel_l775_775499


namespace find_range_of_weights_l775_775215

theorem find_range_of_weights :
  let weights := [41, 48, 50, 53, 49, 50, 53, 53, 51, 67] in 
  (list.max weights).iget - (list.min weights).iget = 26 := by sorry

end find_range_of_weights_l775_775215


namespace cos_5_theta_l775_775821

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end cos_5_theta_l775_775821


namespace coefficient_of_x5y4_in_binomial_expansion_l775_775580


theorem coefficient_of_x5y4_in_binomial_expansion :
  (nat.choose 9 5) = 126 := by
  -- The proof is not required, so we are leaving it as sorry
  sorry

end coefficient_of_x5y4_in_binomial_expansion_l775_775580


namespace arithmetic_sequence_general_term_and_sum_l775_775376

variable (a_n : ℕ → ℝ)
variable (S_n : ℕ → ℝ)
variable (a_3 a_4 a_7 : ℝ)
variable (d a_1 : ℝ)

axiom h1 : a_3 = 1
axiom h2 : a_4 = real.sqrt (a_3 * a_7)
axiom h3 : ∀ n : ℕ, a_n = a_1 + (n - 1) * d

theorem arithmetic_sequence_general_term_and_sum :
  (∀ n : ℕ, a_n = 2 * n - 5) ∧ (∀ n : ℕ, S_n = n^2 - 4 * n) :=
by
  exists a_1 d,
  have : a_4 = a_1 + 3 * d,
  have : a_3 = a_1 + 2 * d,
  sorry

end arithmetic_sequence_general_term_and_sum_l775_775376


namespace angle_OBM_eq_pi_div_2_l775_775760

-- Definitions based on the conditions
variables {A B C K N M O : Type*}
variables [triangle A B C]
variables [circumcircle O A C]
variables [meets AKB : A ∈ circle(O) ∧ K ∈ circle(O) ∧ B ∈ circle(O)]
variables [meets BNC : N ∈ circle(O) ∧ B ∈ circle(O) ∧ C ∈ circle(O)]
variables [circumcircle_ABC : ∃ (c₁ : circle A B C), B ∈ c₁ ∧ M ∈ c₁]
variables [circumcircle_KBN : ∃ (c₂ : circle K B N), B ∈ c₂ ∧ M ∈ c₂]

-- Goal: Prove that ∠OBM = π / 2
theorem angle_OBM_eq_pi_div_2 : ∠ O B M = π / 2 :=
by sorry

end angle_OBM_eq_pi_div_2_l775_775760


namespace picture_area_l775_775069

theorem picture_area (x y : ℕ) (h1 : 1 < x) (h2 : 1 < y)
  (h3 : (3*x + 3) * (y + 2) = 110) : x * y = 28 :=
by {
  sorry
}

end picture_area_l775_775069


namespace least_three_digit_divisible_by_2_5_7_3_l775_775991

theorem least_three_digit_divisible_by_2_5_7_3 : 
  ∃ n, n = 210 ∧ (100 ≤ n) ∧ 
           (n < 1000) ∧ 
           (n % 2 = 0) ∧ 
           (n % 5 = 0) ∧ 
           (n % 7 = 0) ∧ 
           (n % 3 = 0) :=
by
  use 210
  split
  rfl
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  norm_num

end least_three_digit_divisible_by_2_5_7_3_l775_775991


namespace term_containing_x6_l775_775049

noncomputable def binomial_expansion (a b : ℝ) (x : ℝ) (n : ℕ) := (a * x + b / x) ^ n

theorem term_containing_x6
  (a b : ℝ) (n : ℕ)
  (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_A : 2 ^ n = 256)
  (h_B : (a + b) ^ n = 256)
  (h_C : binomial_coeff n 4 * a ^ 4 * b ^ 4 = 70) :
  (∃ k : ℕ, 8 * x ^ (n - 2 * k) = 8 * x ^ 6) :=
begin
  sorry
end

end term_containing_x6_l775_775049


namespace vector_magnitude_sub_l775_775411

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (theta : ℝ) (h_theta : theta = Real.pi / 3)

/-- Given vectors a and b with magnitudes 2 and 3 respectively, and the angle between them is 60 degrees,
    we need to prove that the magnitude of the vector a - b is sqrt(7). -/
theorem vector_magnitude_sub : ‖a - b‖ = Real.sqrt 7 :=
by
  sorry

end vector_magnitude_sub_l775_775411


namespace find_c_l775_775009

theorem find_c
    (unit_squares : ℝ)
    (slanted_line  : ℝ → ℝ)
    (area_division : ℝ → Prop)
    (line_eqn      : ℝ → ℝ)
    (eq_area_2     : ℝ)
    (c_solution    : ℝ) :
    (∀ x, unit_squares = 1) →
    (∀ x, slanted_line (c:ℝ) := (x - c) * 3 / (3 - c)) →
    (∀ y, line_eqn (c: ℝ) = 3 * (3 - c) / 2 - 1) →
    (∀ a, area_division (c) → (3 * (3 - c) / 2 - 1 = 2.5)) →
    (∃ (c : ℝ), eq_area_2 = 2.5 → c_solution = 2 / 3) :=
begin
  sorry
end

end find_c_l775_775009


namespace min_magnitude_l775_775147

open Complex

theorem min_magnitude
  (x y z : ℕ)
  (hx : x ≠ y ∧ x ≠ z ∧ y ≠ z)
  (θ : ℂ)
  (hθ : θ ^ 4 = 1 ∧ θ ≠ 1) :
  ∃ t : ℝ, t = sqrt 23 ∧ (∀ a b c : ℕ, (a ≠ b ∧ a ≠ c ∧ b ≠ c) → (|a + b * θ + c * θ^3| ≥ t)) :=
begin
  -- Proof would go here
  sorry
end

end min_magnitude_l775_775147


namespace sum_of_odd_index_l775_775757

def sequence (r : ℝ) : ℕ → ℤ
| 0       := 0
| 1       := 1
| n + 2   := r * (sequence r (n + 1)) - (sequence r n)

theorem sum_of_odd_index (r : ℝ) (m : ℕ) (h : m ≥ 1) :
  (∑ k in finset.range m, sequence r (2 * k + 1)) = (sequence r m) ^ 2 := sorry

end sum_of_odd_index_l775_775757


namespace square_area_percentage_error_l775_775598

theorem square_area_percentage_error {s : ℝ} (h : s > 0) :
    let s' := s * 1.02 in
    let A := s^2 in
    let A' := s'^2 in
    (A' - A) / A * 100 = 4.04 := 
by 
  sorry

end square_area_percentage_error_l775_775598


namespace sum_series_fraction_l775_775326

theorem sum_series_fraction :
  (∑ n in Finset.range 9, (1 : ℚ) / ((n + 2) * (n + 3))) = 9 / 22 := sorry

end sum_series_fraction_l775_775326


namespace areas_equal_l775_775904

open EuclideanGeometry

-- Define the configuration of points and the circumcenter
variables {A B C O A' B' C' : Point}

-- Assume O is the circumcenter of triangle ABC
axiom circumcenter_of_triangle :
  IsCircumcenter O A B C

-- Assume rays AO, BO, and CO intersect circumcircle of triangle ABC at A', B', C' respectively
axiom rays_intersect_circumcircle :
  RayIntersectsCircumcircle AO O A A' ∧ 
  RayIntersectsCircumcircle BO O B B' ∧ 
  RayIntersectsCircumcircle CO O C C'

-- Define the areas of the respective triangles
noncomputable def area_ABC := AreaTriangle A B C
noncomputable def area_A'BC := AreaTriangle A' B C
noncomputable def area_B'CA := AreaTriangle B' C A
noncomputable def area_C'AB := AreaTriangle C' A B

-- The statement to be proved
theorem areas_equal :
  area_A'BC + area_B'CA + area_C'AB = area_ABC := sorry

end areas_equal_l775_775904


namespace cheese_placement_distinct_ways_l775_775231

theorem cheese_placement_distinct_ways (total_wedges : ℕ) (selected_wedges : ℕ) : 
  total_wedges = 18 ∧ selected_wedges = 6 → 
  ∃ (distinct_ways : ℕ), distinct_ways = 130 :=
by
  sorry

end cheese_placement_distinct_ways_l775_775231


namespace smoothies_from_strawberries_l775_775116

theorem smoothies_from_strawberries (smoothies_per_three_strawberries : ℕ) :
  smoothies_per_three_strawberries = 15 -> (18 * (15 / 3)) = 90 :=
by
  intros h
  rw h
  calc
    18 * (15 / 3) = 18 * 5 : by norm_num
                   ... = 90 : by norm_num

end smoothies_from_strawberries_l775_775116


namespace average_age_parents_and_children_l775_775943

noncomputable def avg_combined_age (n₁ n₂ : ℕ) (a₁ a₂ : ℝ) : ℝ :=
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂)

theorem average_age_parents_and_children :
  let sixth_graders := 45
  let sixth_graders_avg_age := 12
  let parents := 60
  let parents_avg_age := 35
  round (avg_combined_age sixth_graders parents sixth_graders_avg_age parents_avg_age) 2 = 25.14 :=
sorry

end average_age_parents_and_children_l775_775943


namespace cos_5_theta_l775_775820

theorem cos_5_theta (θ : ℝ) (h : Real.cos θ = 2 / 5) : Real.cos (5 * θ) = 2762 / 3125 := 
sorry

end cos_5_theta_l775_775820


namespace minimum_cost_is_correct_l775_775881

noncomputable def rectangular_area (length width : ℝ) : ℝ :=
  length * width

def flower_cost_per_sqft (flower : String) : ℝ :=
  match flower with
  | "Marigold" => 1.00
  | "Sunflower" => 1.75
  | "Tulip" => 1.25
  | "Orchid" => 2.75
  | "Iris" => 3.25
  | _ => 0.00

def min_garden_cost : ℝ :=
  let areas := [rectangular_area 5 2, rectangular_area 7 3, rectangular_area 5 5, rectangular_area 2 4, rectangular_area 5 4]
  let costs := [flower_cost_per_sqft "Orchid" * 8, 
                flower_cost_per_sqft "Iris" * 10, 
                flower_cost_per_sqft "Sunflower" * 20, 
                flower_cost_per_sqft "Tulip" * 21, 
                flower_cost_per_sqft "Marigold" * 25]
  costs.sum

theorem minimum_cost_is_correct :
  min_garden_cost = 140.75 :=
  by
    -- Proof omitted
    sorry

end minimum_cost_is_correct_l775_775881


namespace exist_point_S_l775_775574

-- Define the vertices of the acute-angled triangle
variables {A B C : Point} (h_acute : acute ∠BCA ∧ acute ∠CAB ∧ acute ∠ABC)

-- Define the orthocenter of the triangle
def orthocenter : Point :=
  sorry

-- Define the diameter of sphere for vertices
noncomputable def sphere_diameter_A (A' : Point) : Real :=
  distance A A'

noncomputable def sphere_diameter_B (B' : Point) : Real :=
  distance B B'

noncomputable def sphere_diameter_C (C' : Point) : Real :=
  distance C C'

-- Define the transversal and the construction involving spheres
def transversal (P : Point) : Line :=
  line_through P B ∧ line_through P C

theorem exist_point_S (P S : Point) :
  acute_triangle A B C → 
  (∀ A' B' C' : Point, A' = foot A B C ∧ B' = foot B C A ∧ C' = foot C A B →
   (∃ S : Point, ∀ T : Point, T ∈ line_through S T →
    right_angle (transversal A T) ∧ right_angle (transversal B T) ∧ right_angle (transversal C T))) :=
sorry

end exist_point_S_l775_775574


namespace transformed_properties_l775_775089

noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := cos (2 * x + φ)

noncomputable def symmetry_condition (φ : ℝ) : Prop :=
  2 * (4 * π / 3) + φ = int.cast (4 * π / 3) + φ = int.cast (φ)

noncomputable def transformed_function (x : ℝ) : ℝ := cos (2 * (x + π / 3) - π / 6)

theorem transformed_properties (h_symmetry : symmetry_condition (φ := - π / 6)) 
    (h_range : - π / 2 < (- π / 6) ∧ (- π / 6) < π / 2) : 
  (∀ x, transformed_function x = -sin (2 * x)) ∧ 
  odd_function (transformed_function) ∧ 
  monotone_decreasing_on transformed_function (set.Ioo 0 (π / 4)) :=
sorry

end transformed_properties_l775_775089


namespace milan_billed_minutes_l775_775715

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l775_775715


namespace students_catching_up_on_homework_correct_l775_775103

-- Define the conditions
def total_students : ℕ := 60
def fraction_silent_reading : ℚ := 3 / 8
def fraction_board_games : ℚ := 1 / 4

-- Define the calculation for number of students catching up on homework
def students_catching_up_on_homework (total : ℕ) (frac_reading frac_games : ℚ) : ℕ :=
  let reading := Int.to_nat (frac_reading * total).round
  let games := Int.to_nat (frac_games * total).round
  total - (reading + games)

-- The statement we need to prove
theorem students_catching_up_on_homework_correct :
  students_catching_up_on_homework total_students fraction_silent_reading fraction_board_games = 22 :=
by
  sorry

end students_catching_up_on_homework_correct_l775_775103


namespace normal_intersects_again_at_B_l775_775889

noncomputable def point := (ℚ, ℚ)

def A : point := (2, 4)

def parabola (x : ℚ) : ℚ := x^2

def is_on_parabola (p : point) : Prop := (parabola p.1) = p.2

def normal_slope (x : ℚ) : ℚ := - (1 / (2 * x))

def normal_line (p : point) (x : ℚ) : ℚ := - (1/(4 * p.1) * (x - p.1)) + p.2

theorem normal_intersects_again_at_B : 
  ∃ B : point,
    is_on_parabola A ∧
    is_on_parabola B ∧
    B ≠ A ∧
    ∀ x, x ≠ 2 → (parabola x) = (normal_line A x) ↔ x = -9/4 :=
  sorry

end normal_intersects_again_at_B_l775_775889


namespace probability_sum_3_or_6_l775_775431

def balls := {1, 2, 3, 4, 5}

def possible_pairs := { (a, b) | a ∈ balls ∧ b ∈ balls ∧ a < b }

def favorable_pairs := { (1, 2), (1, 5), (2, 4) }

theorem probability_sum_3_or_6 : (favorable_pairs.card : ℚ) / possible_pairs.card = 3 / 10 :=
by
  sorry

end probability_sum_3_or_6_l775_775431


namespace sum_of_S_values_l775_775894

-- Definition of S_n according to the problem
def S (n : ℕ) : ℤ :=
  ∑ i in finset.range n, if (i % 3 = 2) then -(i + 1) else (i + 1)

-- WorldView proves
theorem sum_of_S_values : S 18 + S 34 + S 51 = 79 :=
by
  sorry

end sum_of_S_values_l775_775894


namespace length_QR_right_triangle_DEF_l775_775934

theorem length_QR_right_triangle_DEF {DE EF DF: ℝ} (h1 : DE = 9) (h2 : EF = 12) (h3 : DF = 15) 
    (hQ : ∃Q, (circle_tangent_to_line_at_point Q DE D ∧ circle_passes_through Q F)) 
    (hR : ∃R, (circle_tangent_to_line_at_point R DF F ∧ circle_passes_through R E)) :
    let QR := length_segment Q R in
    QR = 15.375 :=
by
  sorry

-- Auxiliary definitions
structure Point (ℝ : Type) :=
  (x : ℝ) (y : ℝ)

def length_segment (P Q : Point ℝ) : ℝ :=
    sqrt ((Q.x - P.x)^2 + (Q.y - P.y)^2)

def circle_tangent_to_line_at_point (C : Point ℝ) (L : ℝ → ℝ) (P : Point ℝ) : Prop :=
  -- Definition omitted

def circle_passes_through (C : Point ℝ) (P : Point ℝ) : Prop :=
  -- Definition omitted

end length_QR_right_triangle_DEF_l775_775934


namespace philips_painting_total_l775_775504

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end philips_painting_total_l775_775504


namespace handball_tournament_l775_775098

theorem handball_tournament :
  let points (win: ℕ) (draw: ℕ) (loss: ℕ) := 2 * win + draw;
  let total_teams := 14;
  let games_played (n: ℕ) := n * (n - 1) / 2;
  let total_points (games: ℕ) := 2 * games;
  let each_team_once := ∀ i j : ℕ, i ≠ j → played i j;
  let points_unique := ∀ i j : ℕ, i ≠ j → get_points i ≠ get_points j;
  ∀ (win draw loss : ℕ) (played get_points : ℕ → ℕ),
  (∀ team : ℕ, team < total_teams → points (win team) (draw team) (loss team) = get_points team) →
  (∀ i j : ℕ, i < total_teams → j < total_teams → i ≠ j → each_team_once i j) →
  (points_unique get_points) →
  let top_teams := {i | i < 3};
  let bottom_teams := {i | i ≥ total_teams - 3};
  ¬ ∀ t b, t ∈ top_teams → b ∈ bottom_teams → lost_to t b
:=
by sorry

end handball_tournament_l775_775098


namespace area_of_triangle_OAB_is_constant_equation_of_circle_given_intersections_l775_775065

-- 1. Prove the area of triangle OAB is a constant value
theorem area_of_triangle_OAB_is_constant {t : ℝ} (ht : t ≠ 0) :
  let C := (t, 2 / t) in
  let O := (0, 0) in
  let A := (2 * t, 0) in
  let B := (0, 4 / t) in
  let area := 1 / 2 * |2 * t * (4 / t)| in
  area = 4 :=
sorry

-- 2. Find the equation of circle C given line intersection condition
theorem equation_of_circle_given_intersections {t : ℝ} (ht : t ≠ 0) :
  let C := (t, 2 / t) in
  let line_eq := -2 * x + 4 in
  (OM = ON) →  -- condition that O lies on the perpendicular bisector of MN
  (t = 2 ∨ t = -2) →
  -- Possible equations 
  ((x - 2)^2 + (y - 1)^2 = 5 ∨ (x + 2)^2 + (y + 1)^2 = 5) →
  (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

end area_of_triangle_OAB_is_constant_equation_of_circle_given_intersections_l775_775065


namespace find_volume_of_pyramid_l775_775102

noncomputable def volume_of_pyramid
  (a : ℝ) (α : ℝ)
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) : ℝ :=
  (a ^ 3 * abs (Real.cos α)) / 3

--and the theorem to prove the statement
theorem find_volume_of_pyramid
  (a α : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < α ∧ α < π) 
  (h3 : ∀ θ, θ = α ∨ θ = π - α ∨ θ = 2 * π - α) :
  volume_of_pyramid a α h1 h2 h3 = (a ^ 3 * abs (Real.cos α)) / 3 :=
sorry

end find_volume_of_pyramid_l775_775102


namespace area_triangle_QRS_l775_775929

open EuclideanGeometry

/-- The statement of the problem:
  Points P, Q, R, S, and T are located in 3-dimensional space.
  P and Q, Q and R, R and S, S and T, and T and P each has a distance of 3.
  The angles PQR, RST, and STP are each 120 degrees.
  The plane of triangle PQR is perpendicular to the line segment ST.
  Given these conditions, the area of triangle QRS is 9√3/4. -/
theorem area_triangle_QRS {P Q R S T : EuclideanGeometry.Point}
  (hPQ : dist P Q = 3) (hQR : dist Q R = 3) 
  (hRS : dist R S = 3) (hST : dist S T = 3) (hTP : dist T P = 3)
  (h_anglePQR : angle P Q R = 2 * π / 3)
  (h_angleRST : angle R S T = 2 * π / 3)
  (h_angleSTP : angle S T P = 2 * π / 3)
  (h_plane_perpendicular : is_perpendicular (line_through P Q) 
                                           (line_through S T)) :
  area (triangle Q R S) = 9 * Real.sqrt 3 / 4 :=
begin
  sorry
end

end area_triangle_QRS_l775_775929


namespace allocation_schemes_l775_775708

theorem allocation_schemes : ∃ n : ℕ, 
  let volunteers := 5 in 
  let projects := 4 in
  (∀ v, v < volunteers → (∃ p, p < projects ∧ p.volunteer = v)) ∧ 
  (∀ p, p < projects → (∃ v, v < volunteers ∧ v.project = p)) → 
  n = 240 :=
begin
  sorry
end

end allocation_schemes_l775_775708


namespace polynomial_inequality_l775_775885

noncomputable def poly_deg_n_real_coeffs_real_roots (f : ℝ → ℝ) : Prop :=
  ∃ n (r : Fin n → ℝ) (b : ℝ), b ≠ 0 ∧ f = λ x, b * ∏ i, (x - r i)

theorem polynomial_inequality
  (f : ℝ → ℝ) 
  (h_f : poly_deg_n_real_coeffs_real_roots f) :
  ∀ x : ℝ, f(x) * (deriv^[2] f) x ≤ ((deriv f) x)^2 := 
begin
  sorry
end

end polynomial_inequality_l775_775885


namespace sufficient_condition_l775_775670

theorem sufficient_condition (a : ℝ) (h : a ≥ 5) : ∀ x ∈ set.Icc 1 2, x^2 - a ≤ 0 := 
sorry

end sufficient_condition_l775_775670


namespace slower_speed_is_35_l775_775593

-- Define the given conditions
def distance : ℝ := 70 -- distance is 70 km
def speed_on_time : ℝ := 40 -- on-time average speed is 40 km/hr
def delay : ℝ := 0.25 -- delay is 15 minutes or 0.25 hours

-- This is the statement we need to prove
theorem slower_speed_is_35 :
  ∃ slower_speed : ℝ, 
    slower_speed = distance / (distance / speed_on_time + delay) ∧ slower_speed = 35 :=
by
  sorry

end slower_speed_is_35_l775_775593


namespace least_positive_integer_solution_l775_775584

theorem least_positive_integer_solution :
  ∃ N : ℕ, N > 0 ∧ (N % 5 = 4) ∧ (N % 6 = 5) ∧ (N % 7 = 6) ∧ (N % 8 = 7) ∧ (N % 9 = 8) ∧ (N % 10 = 9) ∧ (N % 11 = 10) ∧ N = 27719 :=
by
  -- the proof is omitted
  sorry

end least_positive_integer_solution_l775_775584


namespace boxes_and_balls_l775_775166

theorem boxes_and_balls : 
  let balls := {"A", "B", "C"}
  let boxes := {1, 2, 3, 4}
  (∃ (f : balls → boxes), ∃ b ∈ balls, f b = 1) → 
    37 := sorry

end boxes_and_balls_l775_775166


namespace sqrt_product_l775_775988

noncomputable def sqrt_fifth_32 : ℝ := 32 ^ (1/5 : ℝ)
noncomputable def sqrt_fourth_16 : ℝ := 16 ^ (1/4 : ℝ)
noncomputable def sqrt_25 : ℝ := 25 ^ (1/2 : ℝ)

theorem sqrt_product :
  (sqrt_fifth_32 * sqrt_fourth_16 * sqrt_25) = 20 :=
by {
  have h1 : sqrt_fifth_32 = 2, by sorry,
  have h2 : sqrt_fourth_16 = 2, by sorry,
  have h3 : sqrt_25 = 5, by sorry,
  rw [h1, h2, h3],
  norm_num,
  exact 20,
}

end sqrt_product_l775_775988


namespace product_of_solutions_neg49_l775_775237

noncomputable def product_of_roots (a b c : ℝ) : ℝ := c / a

theorem product_of_solutions_neg49 :
  ∀ (x : ℝ), -49 = -x^2 - 4 * x →
    product_of_roots 1 4 (-49) = -49 :=
begin
  -- We define the quadratic equation constants.
  intros x h,
  sorry
end

end product_of_solutions_neg49_l775_775237


namespace cos_5theta_l775_775818

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end cos_5theta_l775_775818


namespace subset_bound_l775_775466

theorem subset_bound {n : ℕ} (A : Finset ℕ) (h1 : ∀ x ∈ A, x ∣ n ∨ x = n) 
  (h2 : ∀ x ∈ A, ∃! y ∈ A, x ∣ y) :
  (2 * n / 3 : ℤ) ≤ (A.card : ℤ) ∧ (A.card : ℤ) ≤ ⌈ (3 * n / 4 : ℤ) ⌉ := 
sorry

end subset_bound_l775_775466


namespace train_cross_tree_time_l775_775611

theorem train_cross_tree_time :
  ∀ (train_length platform_length : ℕ) (time_to_pass_platform : ℝ), 
  train_length = 1200 →
  platform_length = 1100 →
  time_to_pass_platform = 230 →
  let speed := (train_length + platform_length) / time_to_pass_platform in
  (train_length / speed) = 120 :=
by
  intros train_length platform_length time_to_pass_platform h_train_length h_platform_length h_time_to_pass_platform
  simp [h_train_length, h_platform_length, h_time_to_pass_platform]
  let speed := (1200 + 1100) / 230
  have h_speed : speed = 10 := by norm_num
  simp [h_speed]
  norm_num
  sorry

end train_cross_tree_time_l775_775611


namespace tangent_line_eqn_l775_775197

theorem tangent_line_eqn : 
  ∀ (x y : ℝ), y = cos x - x / 2 → (0, 1) ∈ set_of (λ p : ℝ × ℝ, p.2 = cos p.1 - p.1 / 2) → 
  (∃ C : ℝ, ∀ x0, y0, y0 = cos x0 - x0 / 2 → (x0, y0) = (0, 1) → C = -1 / 2 → x + 2 * y - 2 = 0) :=
by 
  sorry

end tangent_line_eqn_l775_775197


namespace pool_capacity_is_80_percent_l775_775940

noncomputable def current_capacity_percentage (width length depth rate time : ℝ) : ℝ :=
  let total_volume := width * length * depth
  let water_removed := rate * time
  (water_removed / total_volume) * 100

theorem pool_capacity_is_80_percent :
  current_capacity_percentage 50 150 10 60 1000 = 80 :=
by
  sorry

end pool_capacity_is_80_percent_l775_775940


namespace find_positive_integer_solutions_l775_775684

theorem find_positive_integer_solutions :
  { x : ℤ × ℤ × ℤ // x.1 > 0 ∧ x.2.1 > 0 ∧ x.2.2 > 0 ∧ 2 ^ x.1 * 3 ^ x.2.1 + 9 = x.2.2 ^ 2 } =
  { (4, 0, 5), (4, 5, 51), (3, 3, 15), (4, 3, 21), (3, 2, 9) } :=
by
  sorry

end find_positive_integer_solutions_l775_775684


namespace axis_of_symmetry_transformed_function_l775_775531

-- Define the original function
def cos_fn (x : ℝ) : ℝ := Real.cos x

-- Define the function after stretching the x-coordinates by a factor of 2
def stretched_cos_fn (x : ℝ) : ℝ := Real.cos (x / 2)

-- Define the function after translating to the left by π units
def transformed_cos_fn (x : ℝ) : ℝ := Real.cos ((x + Real.pi) / 2)

-- Define the statement of the problem
theorem axis_of_symmetry_transformed_function : ∃ k : ℤ, transformed_cos_fn (-Real.pi) = transformed_cos_fn (k * 2 * Real.pi - Real.pi) :=
by
  sorry

end axis_of_symmetry_transformed_function_l775_775531


namespace find_constants_C_D_l775_775681

theorem find_constants_C_D : 
  (∃ C D : ℚ, 
    C = 81 / 16 ∧ D = -49 / 16 ∧ 
    (∀ x : ℚ, x ≠ 12 → x ≠ -4 → 
      (7 * x - 3) / (x^2 - 8 * x - 48) = C / (x - 12) + D / (x + 4))) :=
by
  existsi (81 / 16, -49 / 16)
  sorry

end find_constants_C_D_l775_775681


namespace estimate_pi_correct_l775_775523

noncomputable def estimate_pi (pairs : List (ℝ × ℝ)) (m : ℕ) : ℚ :=
  if h₁ : 1 > 0 ∧ 1 > 0 then (pairs.filter (fun p => p.1^2 + p.2^2 < 1 ∧ p.1 < 1 ∧ p.2 < 1 ∧ p.1 + p.2 > 1)).length
  else 0

theorem estimate_pi_correct (pairs : List (ℝ × ℝ))
  (h_pairs : ∀ (p : ℝ × ℝ), p ∈ pairs → 0 < p.1 ∧ p.1 < 1 ∧ 0 < p.2 ∧ p.2 < 1)
  (m : ℕ)
  (h_m : length (pairs.filter (fun p => p.1^2 + p.2^2 < 1 ∧ p.1 < 1 ∧ p.2 < 1 ∧ p.1 + p.2 > 1)) = m)
  (h_m_value : m = 56)
  (h_total_pairs : length pairs = 200)
  : estimate_pi pairs m = 78 / 25 :=
  sorry

end estimate_pi_correct_l775_775523


namespace solution_set_of_inequality_l775_775964

theorem solution_set_of_inequality (x : ℝ) : 
  abs ((x + 2) / x) < 1 ↔ x < -1 :=
by
  sorry

end solution_set_of_inequality_l775_775964


namespace afb_leq_bfa_l775_775948

open Real

variable {f : ℝ → ℝ}

theorem afb_leq_bfa
  (h_nonneg : ∀ x > 0, f x ≥ 0)
  (h_diff : ∀ x > 0, DifferentiableAt ℝ f x)
  (h_cond : ∀ x > 0, x * (deriv (deriv f) x) - f x ≤ 0)
  (a b : ℝ)
  (h_a_pos : 0 < a)
  (h_b_pos : 0 < b)
  (h_a_lt_b : a < b) :
  a * f b ≤ b * f a := 
sorry

end afb_leq_bfa_l775_775948


namespace golden_triangle_area_of_hyperbola_l775_775969

noncomputable def golden_triangle_area : ℝ := 
  let a : ℝ := 2
  let b : ℝ := 2
  let c : ℝ := 2 * Real.sqrt 2
  let right_focus := (c, 0)
  let right_vertex := (a, 0)
  let endpoint_conjugate_axis := (0, b)
  let area := 1 / 2 * (c - a) * b
  area

theorem golden_triangle_area_of_hyperbola : 
  (golden_triangle_area: ℝ) = 2 * Real.sqrt 2 - 2 :=
begin
  sorry
end

end golden_triangle_area_of_hyperbola_l775_775969


namespace prime_factor_Φ_n_condition_l775_775367

open Nat

def is_prime (p : ℕ) : Prop := p.prime

theorem prime_factor_Φ_n_condition 
  (n : ℕ) (x₀ : ℤ) (p : ℕ) 
  (h1 : 0 < n) 
  (h2 : is_prime p) 
  (h3 : p ∣ (cyclotomic n ℤ).eval x₀) 
  : (n ∣ (p - 1)) ∨ (p ∣ n) :=
sorry

end prime_factor_Φ_n_condition_l775_775367


namespace count_congruent_to_2_mod_7_l775_775814

theorem count_congruent_to_2_mod_7 : 
  let count := (1 to 300).count (λ x, x % 7 = 2)
  count = 43 := by
  sorry

end count_congruent_to_2_mod_7_l775_775814


namespace cos_double_angle_tan_sum_angles_l775_775379

variable (α β : ℝ)
variable (α_acute : 0 < α ∧ α < π / 2)
variable (β_acute : 0 < β ∧ β < π / 2)
variable (tan_alpha : Real.tan α = 4 / 3)
variable (sin_alpha_minus_beta : Real.sin (α - β) = - (Real.sqrt 5) / 5)

/- Prove that cos 2α = -7/25 given the conditions -/
theorem cos_double_angle :
  Real.cos (2 * α) = -7 / 25 :=
by
  sorry

/- Prove that tan (α + β) = -41/38 given the conditions -/
theorem tan_sum_angles :
  Real.tan (α + β) = -41 / 38 :=
by
  sorry

end cos_double_angle_tan_sum_angles_l775_775379


namespace proof_problem_l775_775786

-- Define the function f
def f (a : ℝ) (x : ℝ) := a * Real.cos (x + Real.pi / 6)

-- Condition 1: The function passes through the given point
def passesThrough (a : ℝ) := f(a, Real.pi / 2) = -1/2

-- Condition 2: \sin \theta = 1/3 with 0 < \theta < \pi / 2
def sinTheta (θ : ℝ) := Real.sin θ = 1/3 ∧ 0 < θ ∧ θ < Real.pi / 2

-- Definitions of a and f(θ)
theorem proof_problem (a : ℝ) (θ : ℝ) (h₁ : passesThrough a) (h₂ : sinTheta θ) :
  a = 1 ∧ f 1 θ = (2 * Real.sqrt 6 - 1) / 6 :=
by sorry

end proof_problem_l775_775786


namespace gcd_18_n_eq_6_in_range_l775_775723

theorem gcd_18_n_eq_6_in_range :
  {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 18 n = 6}.card = 22 :=
by
  -- To skip the proof
  sorry

end gcd_18_n_eq_6_in_range_l775_775723


namespace sequence_prime_count_l775_775595

variable (Q : ℕ) (m : ℕ)

def is_prime_in_sequence : Prop :=
  Q = List.prod (List.filter Nat.Prime (List.range' 2 58)) ∧
    (∀ (m : ℕ), 2 ≤ m ∧ m ≤ 61 → m ≠ 61 → ¬Nat.Prime (Q + m)) ∧
    (Nat.Prime (Q + 61) ↔ Nat.Prime (Q + 61))

theorem sequence_prime_count :
  ∃ (Q : ℕ), (Q = List.prod (List.filter Nat.Prime (List.range' 2 58))) →
  (∑ m in Finset.range 60, ite (Nat.Prime (Q + (m + 2))) 1 0 = if Nat.Prime (Q + 61) then 1 else 0) :=
by
  sorry

end sequence_prime_count_l775_775595


namespace find_r_l775_775914

noncomputable def a : ℝ × ℝ × ℝ := (2, 3, -1)
noncomputable def b : ℝ × ℝ × ℝ := (-1, 1, 2)
noncomputable def c : ℝ × ℝ × ℝ := (5, 2, -3)
noncomputable def cross_prod (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.2 * v.3 - u.3 * v.2, u.3 * v.1 - u.1 * v.3, u.1 * v.2 - u.2 * v.1)

def dot_prod (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_r (p q r : ℝ) :
  c = (p • a) + (q • b) + (r • cross_prod a b) → r = 2 / 15 :=
by 
  -- The proof would go here, showing the computations and the final result
  sorry

end find_r_l775_775914


namespace max_value_of_f_l775_775261

noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x) ^ 2 + 2 * Real.cos x

theorem max_value_of_f :
  ∃ x ∈ Icc 0 (Real.pi / 2), f x = 1 + (3 / 2) * Real.sqrt 3 :=
sorry

end max_value_of_f_l775_775261


namespace last_number_deleted_on_blackboard_l775_775469

theorem last_number_deleted_on_blackboard 
  (m : ℕ) (hm : m > 1) : 
  let N := m^2017 + 1,
      X := (m^2017 + 1) / (m + 1) + m 
  in last_number_deleted N = X :=
sorry

end last_number_deleted_on_blackboard_l775_775469


namespace base_4_representation_has_four_digits_l775_775861

theorem base_4_representation_has_four_digits (n : ℕ) (h : n = 73) : 
  nat.digits 4 n = [1, 1, 2, 1] :=
by {
  have h1 : nat.digits 4 73 = [1, 1, 2, 1] := by rfl,
  exact h1,
}

end base_4_representation_has_four_digits_l775_775861


namespace remaining_area_l775_775622

theorem remaining_area (x : ℝ) :
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  A_large - A_hole = - x^2 + 22 * x + 52 := by
  let A_large := (2 * x + 8) * (x + 6)
  let A_hole := (3 * x - 4) * (x + 1)
  have hA_large : A_large = 2 * x^2 + 20 * x + 48 := by
    sorry
  have hA_hole : A_hole = 3 * x^2 - 2 * x - 4 := by
    sorry
  calc
    A_large - A_hole = (2 * x^2 + 20 * x + 48) - (3 * x^2 - 2 * x - 4) := by
      rw [hA_large, hA_hole]
    _ = -x^2 + 22 * x + 52 := by
      ring

end remaining_area_l775_775622


namespace find_intersection_of_normal_at_A_with_parabola_l775_775891

theorem find_intersection_of_normal_at_A_with_parabola {x : ℝ} {y : ℝ} 
  (A : (ℝ × ℝ)) (hA_on_parabola : A = (2, 4)) (hParabola : y = x^2) :
  let B := (-2.25, 5.0625) in 
  ∃ B : (ℝ × ℝ), (B.2 = B.1^2) ∧ (B.1 = -2.25) ∧ (B.2 = 5.0625) ∧
  (∃ c : ℝ, 
    (y - 4 = -1/4 * (x - 2)) ∧ 
    (y = -1/4 * x + c) ∧ 
    (y = x^2) ∧ 
    (B.1 ≠ 2)) := by sorry

end find_intersection_of_normal_at_A_with_parabola_l775_775891


namespace proof_problem_l775_775639

variable (A B C : ℕ)

-- Defining the conditions
def condition1 : Prop := A + B + C = 700
def condition2 : Prop := B + C = 600
def condition3 : Prop := C = 200

-- Stating the proof problem
theorem proof_problem (h1 : condition1 A B C) (h2 : condition2 B C) (h3 : condition3 C) : A + C = 300 :=
sorry

end proof_problem_l775_775639


namespace combined_yearly_return_is_19_percent_l775_775610

-- Defining the investment amounts and their respective returns.
def A1 : ℕ := 500
def A2 : ℕ := 1500
def R1 : ℝ := 0.07
def R2 : ℝ := 0.23

-- Calculate returns from investments.
def return1 := A1 * R1
def return2 := A2 * R2

-- Calculate total investment and total return.
def total_investment := A1 + A2
def total_return := return1 + return2

-- Calculate combined yearly return percentage.
def combined_yearly_return_percentage := (total_return / total_investment) * 100

-- Theorem to prove combined yearly return percentage is 19%.
theorem combined_yearly_return_is_19_percent :
  combined_yearly_return_percentage = 19 := by
  sorry

end combined_yearly_return_is_19_percent_l775_775610


namespace sum_of_coordinates_of_other_endpoint_of_segment_l775_775923

theorem sum_of_coordinates_of_other_endpoint_of_segment {x y : ℝ}
  (h1 : (6 + x) / 2 = 3)
  (h2 : (1 + y) / 2 = 7) :
  x + y = 13 := by
  sorry

end sum_of_coordinates_of_other_endpoint_of_segment_l775_775923


namespace variance_unchanged_l775_775946

variable {α : Type} [LinearOrder α] [Add α]

-- Define the ages of Yuan Yuan's family
def ages (father_age mother_age yuanyuan_age : α) := (father_age, mother_age, yuanyuan_age)

-- Define a function to calculate the variance of a triplet of ages
def variance (x y z : α) : α :=
  let mean := (x + y + z) / 3
  in ((x - mean) ^ 2 + (y - mean) ^ 2 + (z - mean) ^ 2) / 3

-- Statement: Prove that the variance remains unchanged when increasing each age by a constant (in this case, 5 years).
theorem variance_unchanged (x y z : α) (c : α) :
  variance x y z = variance (x + c) (y + c) (z + c) :=
by sorry

end variance_unchanged_l775_775946


namespace polygon_angle_sum_and_regularity_l775_775895

-- Definitions based on the conditions
def polygon (Q : Type) := finitely_many_vertices Q

def interior_angle (a : Q → ℝ) := ∀ v : Q, 9 * exterior_angle v = a v

noncomputable def sum_of_interior_angles (a : Q → ℝ) : ℝ :=
  ∑ v in vertices Q, a v

-- Statement to prove
theorem polygon_angle_sum_and_regularity (Q : Type) [polygon Q] (a : Q → ℝ) (b : Q → ℝ) 
  (h1 : ∀ v : Q, a v = 9 * b v) (h2 : ∑ v in vertices Q, b v = 360) :
  sum_of_interior_angles a = 3240 ∧ (Q.regular ∨ ¬Q.regular) :=
by
  sorry

end polygon_angle_sum_and_regularity_l775_775895


namespace sum_coefficients_even_l775_775134

open Real Polynomial

noncomputable def m := (-1 + sqrt 17) / 2

theorem sum_coefficients_even
  (P : Polynomial ℤ)
  (hP : P.eval m = 2018)
  (coeff_positive : ∀ i, 0 ≤ P.coeff i)
  (coeff_int : ∀ i, P.coeff i ∈ ℤ)
  (degree_pos : 0 < P.natDegree) :
  (P.coeff 0 + P.coeff 1 + P.coeff 2 + ... + P.coeff P.natDegree) % 2 = 0 :=
by
  sorry

end sum_coefficients_even_l775_775134


namespace sum_series_to_fraction_l775_775332

theorem sum_series_to_fraction :
  (∑ n in Finset.range 9, (1 / ((n + 2) * (n + 3) : ℚ))) = 9 / 22 := 
begin
  sorry
end

end sum_series_to_fraction_l775_775332


namespace nina_homework_total_l775_775919

def ruby_math_homework : ℕ := 6

def ruby_reading_homework : ℕ := 2

def nina_math_homework : ℕ := ruby_math_homework * 4 + ruby_math_homework

def nina_reading_homework : ℕ := ruby_reading_homework * 8 + ruby_reading_homework

def nina_total_homework : ℕ := nina_math_homework + nina_reading_homework

theorem nina_homework_total :
  nina_total_homework = 48 :=
by
  unfold nina_total_homework
  unfold nina_math_homework
  unfold nina_reading_homework
  unfold ruby_math_homework
  unfold ruby_reading_homework
  sorry

end nina_homework_total_l775_775919


namespace altitude_in_scientific_notation_l775_775191

theorem altitude_in_scientific_notation : 
  (389000 : ℝ) = 3.89 * (10 : ℝ) ^ 5 :=
by
  sorry

end altitude_in_scientific_notation_l775_775191


namespace ln_gt_ln_sufficient_for_x_gt_y_l775_775260

noncomputable def ln : ℝ → ℝ := sorry  -- Assuming ln is imported from Mathlib

-- Conditions
variable (x y : ℝ)
axiom ln_gt_ln_of_x_gt_y (hxy : x > y) (hx_pos : 0 < x) (hy_pos : 0 < y) : ln x > ln y

theorem ln_gt_ln_sufficient_for_x_gt_y (h : ln x > ln y) : x > y := sorry

end ln_gt_ln_sufficient_for_x_gt_y_l775_775260


namespace angle_supplement_complement_l775_775386

theorem angle_supplement_complement (a : ℝ) (h : 180 - a = 3 * (90 - a)) : a = 45 :=
by
  sorry

end angle_supplement_complement_l775_775386


namespace even_digit_sum_l775_775471

def E (n : ℕ) : ℕ :=
  let digits := (List.ofDigits 10 (Nat.digits 10 n)).filter (λ x => x % 2 = 0)
  let even_sum := digits.sum id
  let additional_five := if n % 10 = 0 then 5 else 0
  even_sum + additional_five

theorem even_digit_sum : (List.range 200).sum (λ n => E (n + 1)) = 902 :=
  sorry

end even_digit_sum_l775_775471


namespace constant_term_in_expansion_l775_775105

-- Define the binomial expansion expression
def binomial_expansion_expr (x : ℝ) := (1 / x - x^3) ^ 4

-- Theorem statement for finding the constant term
theorem constant_term_in_expansion : 
  ∀ (x : ℝ) (h : x ≠ 0), is_constant_term (binomial_expansion_expr x) (-4) :=
begin
  -- By using the conditions to state our theorem
  intros x h,
  sorry, -- Proof goes here
end

-- The helper function to define what we mean by constant term
def is_constant_term (f : ℝ -> ℝ) (c : ℝ) : Prop :=
  f = λ x, c

end constant_term_in_expansion_l775_775105


namespace domain_ln_2sinx_sub_1_l775_775195

theorem domain_ln_2sinx_sub_1 :
  (λ x : ℝ, ∃ k : ℤ, (π / 6) + 2 * k * π < x ∧ x < (5 * π / 6) + 2 * k * π) = { x : ℝ | ∃ k : ℤ, (π / 6) + 2 * k * π < x ∧ x < (5 * π / 6) + 2 * k * π } :=
by
  sorry

end domain_ln_2sinx_sub_1_l775_775195


namespace cafeteria_green_apples_l775_775559

def number_of_green_apples (G : ℕ) : Prop :=
  42 + G - 9 = 40 → G = 7

theorem cafeteria_green_apples
  (red_apples : ℕ)
  (students_wanting_fruit : ℕ)
  (extra_fruit : ℕ)
  (G : ℕ)
  (h1 : red_apples = 42)
  (h2 : students_wanting_fruit = 9)
  (h3 : extra_fruit = 40)
  : number_of_green_apples G :=
by
  -- Place for proof omitted intentionally
  sorry

end cafeteria_green_apples_l775_775559


namespace pizza_slices_per_pizza_l775_775227

theorem pizza_slices_per_pizza (num_coworkers slices_per_person num_pizzas : ℕ) (h1 : num_coworkers = 12) (h2 : slices_per_person = 2) (h3 : num_pizzas = 3) :
  (num_coworkers * slices_per_person) / num_pizzas = 8 :=
by
  sorry

end pizza_slices_per_pizza_l775_775227


namespace complex_div_eq_neg_i_l775_775673

theorem complex_div_eq_neg_i : (1 - complex.i) / (1 + complex.i) = -complex.i :=
by
  sorry

end complex_div_eq_neg_i_l775_775673


namespace angle_ECD_is_75_l775_775092

/-- In triangle ABC, given AC = BC, ∠DCB = 30°, D lies on BC, and DE ∥ AB, prove that ∠ECD = 75°. -/
theorem angle_ECD_is_75
  (A B C D E : Type) 
  [MetricGeometry A B C D E]
  (h1 : is_isosceles_triangle A B C)
  (h2 : ∠ D C B = 30)
  (h3 : lies_on D B C)
  (h4 : parallel D E A B) :
  ∠ E C D = 75 := by
  sorry

end angle_ECD_is_75_l775_775092


namespace sum_f_values_l775_775032

noncomputable def f : ℝ → ℝ := sorry

axiom odd_property (x : ℝ) : f (-x) = -f (x)
axiom periodicity (x : ℝ) : f (x) = f (x + 4)
axiom f1 : f 1 = -1

theorem sum_f_values : f 1 + f 2 + f 3 + f 4 + f 5 + f 6 + f 7 + f 8 + f 9 + f 10 = -1 :=
by
  sorry

end sum_f_values_l775_775032


namespace polynomial_remainder_l775_775698

theorem polynomial_remainder (x : ℝ) : 
  let f := λ x : ℝ, x^3 - 4 * x + 6 in
  f (-3) = -9 :=
by
  let f := λ x : ℝ, x^3 - 4 * x + 6
  show f (-3) = -9
  sorry

end polynomial_remainder_l775_775698


namespace minimize_expression_10_l775_775212

theorem minimize_expression_10 (n : ℕ) (h : 0 < n) : 
  (∃ m : ℕ, 0 < m ∧ (∀ k : ℕ, 0 < k → (n = k) → (n = 10))) :=
by
  sorry

end minimize_expression_10_l775_775212


namespace moles_of_NaOH_combined_l775_775692

-- Given conditions
def moles_AgNO3 := 3
def moles_AgOH := 3
def balanced_ratio_AgNO3_NaOH := 1 -- 1:1 ratio as per the equation

-- Problem statement
theorem moles_of_NaOH_combined : 
  moles_AgOH = moles_AgNO3 → balanced_ratio_AgNO3_NaOH = 1 → 
  (∃ moles_NaOH, moles_NaOH = 3) := by
  sorry

end moles_of_NaOH_combined_l775_775692


namespace measure_angle_A_l775_775298

-- Angles A and B are supplementary
def supplementary (A B : ℝ) : Prop :=
  A + B = 180

-- Definition of the problem conditions
def problem_conditions (A B : ℝ) : Prop :=
  supplementary A B ∧ A = 4 * B

-- The measure of angle A
def measure_of_A := 144

-- The statement to prove
theorem measure_angle_A (A B : ℝ) :
  problem_conditions A B → A = measure_of_A := 
by
  sorry

end measure_angle_A_l775_775298


namespace negation_of_original_prop_l775_775204

variable (a : ℝ)
def original_prop (x : ℝ) : Prop := x^2 + a * x + 1 < 0

theorem negation_of_original_prop :
  ¬ (∃ x : ℝ, original_prop a x) ↔ ∀ x : ℝ, ¬ original_prop a x :=
by sorry

end negation_of_original_prop_l775_775204


namespace intervals_monotonic_increase_range_of_m_l775_775867

-- Define the functions and given conditions
noncomputable def f (x : ℝ) : ℝ := 1 - Math.sin (2 * x + Real.pi / 6)
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * Math.cos (x + Real.pi / 3) - m + 2

-- Define the intervals of monotonic increase for f
theorem intervals_monotonic_increase (k : ℤ) : 
  ∀ x : ℝ, x ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) → 
  ∀ y : ℝ, y ∈ Set.Icc (k * Real.pi + Real.pi / 6) (k * Real.pi + 2 * Real.pi / 3) → 
  x < y → f x < f y := sorry

-- Prove that m >= 4 given the condition on f and g
theorem range_of_m (m : ℝ) : 
  (∀ x1 x2 : ℝ, x1 ∈ Set.Icc 0 Real.pi → x2 ∈ Set.Icc 0 Real.pi → f x1 >= g m x2) → 
  m ≥ 4 := sorry

end intervals_monotonic_increase_range_of_m_l775_775867


namespace length_AB_is_two_l775_775750

open EuclideanGeometry

noncomputable def parallelogram_ABCD (A B C D : Point) : Prop :=
  parallelogram A B C D

theorem length_AB_is_two (A B C D : Point) (BD_norm : ∥B - D∥ = 2)
  (dot_product_condition : 2 * (D - A) • (B - A) = ∥C - B∥^2)
  (h_para : parallelogram_ABCD A B C D) :
  ∥B - A∥ = 2 := 
sorry

end length_AB_is_two_l775_775750


namespace intersection_A_complement_B_l775_775799

noncomputable def U := set.univ ℝ
def A := { x : ℝ | x^2 - 2 * x < 0 }
def B := { x : ℝ | x >= 1 }
def complement_B := { x : ℝ | x < 1 }
def A_intersect_complement_B := { x : ℝ | 0 < x ∧ x < 1 }

theorem intersection_A_complement_B :
  (A ∩ complement_B) = A_intersect_complement_B :=
sorry

end intersection_A_complement_B_l775_775799


namespace neg_prop_p_equiv_l775_775034

variable {x : ℝ}

def prop_p : Prop := ∃ x ≥ 0, 2^x = 3

theorem neg_prop_p_equiv : ¬prop_p ↔ ∀ x ≥ 0, 2^x ≠ 3 :=
by sorry

end neg_prop_p_equiv_l775_775034


namespace sum_of_squares_of_roots_l775_775701

theorem sum_of_squares_of_roots (s_1 s_2 : ℝ) (h : s_1^2 - 17 * s_1 + 8 = 0) (h' : s_2^2 - 17 * s_2 + 8 = 0) :
  s_1 + s_2 = 17 ∧ s_1 * s_2 = 8 → s_1^2 + s_2^2 = 273 :=
by
  intro h_vieta
  cases h_vieta with sum_roots prod_roots
  -- Apply the algebraic identity: s_1^2 + s_2^2 = (s_1 + s_2)^2 - 2 * (s_1 * s_2)
  have identity : s_1^2 + s_2^2 = (s_1 + s_2) ^ 2 - 2 * s_1 * s_2 := by
    ring
  -- Substitute the values from Vieta's formulas
  rw [sum_roots, prod_roots] at identity
  simp at identity
  -- The final equality
  exact identity

-- Placeholder proof
sorry

end sum_of_squares_of_roots_l775_775701


namespace trigonometric_fraction_value_l775_775736

theorem trigonometric_fraction_value (α : ℝ) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) /
  (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end trigonometric_fraction_value_l775_775736


namespace jane_change_l775_775872

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l775_775872


namespace area_of_intersecting_hexagon_l775_775629

def regular_hexagon_area (r : ℝ) : ℝ :=
  2 * r^2 * Real.sqrt 3

theorem area_of_intersecting_hexagon (a r : ℝ)
  (h_tri : r = (a * Real.sqrt 3) / 6)
  (h_geom : ∀ (O1 O2 O3 : ℝ × ℝ),
              let O1 := (0, -r),
                  O2 := (r * Real.sqrt 3 / 2, r / 2),
                  O3 := (-r * Real.sqrt 3 / 2, r / 2) in
              True): 
  (hex_area : ℝ) :=
  hex_area = regular_hexagon_area r

end area_of_intersecting_hexagon_l775_775629


namespace philips_painting_total_l775_775505

def total_paintings_after_days (daily_paintings : ℕ) (initial_paintings : ℕ) (days : ℕ) : ℕ :=
  initial_paintings + daily_paintings * days

theorem philips_painting_total (daily_paintings initial_paintings days : ℕ) 
  (h1 : daily_paintings = 2) (h2 : initial_paintings = 20) (h3 : days = 30) : 
  total_paintings_after_days daily_paintings initial_paintings days = 80 := 
by
  sorry

end philips_painting_total_l775_775505


namespace function_decreasing_interval_l775_775400

-- Definitions based on conditions
def quadratic_function (x : ℝ) : ℝ := -x^2 + 4*x - 3

-- Problem statement to be proved
theorem function_decreasing_interval :
  ∀ x₁ x₂ : ℝ, (2 ≤ x₁ → x₁ ≤ x₂ → quadratic_function(x₁) ≥ quadratic_function(x₂)) := 
by
  sorry

end function_decreasing_interval_l775_775400


namespace part1_part2_l775_775139

-- Definitions for the conditions
def U : Set ℝ := Set.univ
def A : Set ℝ := { x | 0 ≤ x ∧ x ≤ 3 }
def B (m : ℝ) : Set ℝ := { x | m - 1 ≤ x ∧ x ≤ 2 * m }

-- Part (1)
theorem part1 (m : ℝ) (hm : m = 3) :
  A ∩ (U \ (B m)) = { x | 0 ≤ x ∧ x < 2 } :=
by
  sorry

-- Part (2)
theorem part2 (m : ℝ) (h : B m ∪ A = A) :
  m ∈ Iio (-1) ∪ Icc 1 (3 / 2) :=
by
  sorry

end part1_part2_l775_775139


namespace gcd_is_13_eval_at_neg1_l775_775264

-- Define the GCD problem
def gcd_117_182 : ℕ := gcd 117 182

-- Define the polynomial evaluation problem
def f (x : ℝ) : ℝ := 1 - 9 * x + 8 * x^2 - 4 * x^4 + 5 * x^5 + 3 * x^6

-- Formalize the statements to be proved
theorem gcd_is_13 : gcd_117_182 = 13 := 
by sorry

theorem eval_at_neg1 : f (-1) = 12 := 
by sorry

end gcd_is_13_eval_at_neg1_l775_775264


namespace line_equation_M_l775_775403

theorem line_equation_M (x y : ℝ) :
  (∃ (m c : ℝ), y = m * x + c ∧ m = -5/4 ∧ c = -3)
  ∧ (∃ (slope intercept : ℝ), slope = 2 * (-5/4) ∧ intercept = (1/2) * -3 ∧ (y - 2 = slope * (x + 4)))
  → ∃ (a b : ℝ), y = a * x + b ∧ a = -5/2 ∧ b = -8 :=
by
  sorry

end line_equation_M_l775_775403


namespace parabola_focus_l775_775551

open Real

def parabola (x y : ℝ) : Prop := y^2 = 16 * x

def focus : Point := ⟨4, 0⟩

def onYAxis (p : Point) : Prop := p.x = 0

def distanceFromOrigin (a b : Point) : ℝ :=
  Real.sqrt (a.x^2 + a.y^2)

theorem parabola_focus (A : Point) :
  onYAxis A → 
  distanceFromOrigin A = distanceFromOrigin focus →
  let B := ⟨-4, 0⟩ in
  let FA := ⟨A.x - focus.x, A.y - focus.y⟩ in
  let AB := ⟨B.x - A.x, B.y - A.y⟩ in
  FA.x * AB.x + FA.y * AB.y = 0 :=
by
  sorry

end parabola_focus_l775_775551


namespace optimal_garden_dimensions_l775_775176

theorem optimal_garden_dimensions :
  ∃ (l w : ℝ), 2 * l + 2 * w = 400 ∧ l ≥ 100 ∧ w ≥ 50 ∧ l ≥ w + 20 ∧ l * w = 9600 :=
by
  sorry

end optimal_garden_dimensions_l775_775176


namespace find_intersection_of_normal_at_A_with_parabola_l775_775892

theorem find_intersection_of_normal_at_A_with_parabola {x : ℝ} {y : ℝ} 
  (A : (ℝ × ℝ)) (hA_on_parabola : A = (2, 4)) (hParabola : y = x^2) :
  let B := (-2.25, 5.0625) in 
  ∃ B : (ℝ × ℝ), (B.2 = B.1^2) ∧ (B.1 = -2.25) ∧ (B.2 = 5.0625) ∧
  (∃ c : ℝ, 
    (y - 4 = -1/4 * (x - 2)) ∧ 
    (y = -1/4 * x + c) ∧ 
    (y = x^2) ∧ 
    (B.1 ≠ 2)) := by sorry

end find_intersection_of_normal_at_A_with_parabola_l775_775892


namespace angle_value_l775_775444

theorem angle_value (x : ℝ) (h₁ : (90 : ℝ) = 44 + x) : x = 46 :=
by
  sorry

end angle_value_l775_775444


namespace exists_disjoint_scaled_cover_l775_775133
open Classical

variable {α : Type} [TopologicalSpace α]

def open_disc (center : α) (radius : ℝ) : Set α := 
  { p | dist p center < radius }

def scaled_disc (center : α) (radius : ℝ) (factor : ℝ) : Set α :=
  { p | dist p center < factor * radius }

theorem exists_disjoint_scaled_cover (E : Set (ℝ × ℝ)) 
  (F : Finset (Set (ℝ × ℝ))) (hF : ∀ D ∈ F, ∃ center : ℝ × ℝ, ∃ radius : ℝ, radius > 0 ∧ D = open_disc center radius)
  (hE : E ⊆ ⋃₀ F) :
  ∃ (D' : Finset (Set (ℝ × ℝ))), (∀ D ∈ D', ∃ center : ℝ × ℝ, ∃ radius : ℝ, radius > 0 ∧ D = open_disc center radius)
  ∧ (∀ D₁ ∈ D', ∀ D₂ ∈ D', D₁ ≠ D₂ → D₁ ∩ D₂ = ∅)
  ∧ (E ⊆ ⋃ (D ∈ D'), scaled_disc (classical.some (classical.some_spec (hF D (finset.mem D')))) (classical.some (classical.some_spec (classical.some_spec (hF D (finset.mem D'))))) 3) :=
sorry

end exists_disjoint_scaled_cover_l775_775133


namespace regular_tetrahedron_volume_regular_tetrahedron_surface_area_l775_775752

noncomputable def base_edge_length : ℝ := 2 * real.sqrt 6
noncomputable def height : ℝ := 1

/-- Volume of the regular tetrahedron is 2 * sqrt 3 -/
theorem regular_tetrahedron_volume 
  (h_edge : base_edge_length = 2 * real.sqrt 6)
  (h_height : height = 1) :
  ∃ V : ℝ, V = 2 * real.sqrt 3 := 
sorry

/-- Total surface area of the regular tetrahedron is 9 * sqrt 2 + 6 * sqrt 3 -/
theorem regular_tetrahedron_surface_area 
  (h_edge : base_edge_length = 2 * real.sqrt 6)
  (h_height : height = 1) :
  ∃ S : ℝ, S = 9 * real.sqrt 2 + 6 * real.sqrt 3 := 
sorry

end regular_tetrahedron_volume_regular_tetrahedron_surface_area_l775_775752


namespace sum_of_signs_20222023_impossible_l775_775457

/--
It is impossible to place "+" or "-" signs between each pair of adjacent digits in the number 20222023 such that the resulting expression equals zero.
-/
theorem sum_of_signs_20222023_impossible :
  ∀ (signs : list ℤ), signs.length = 7 →
  (∀ i, i < signs.length → (signs.nth i = some 1 ∨ signs.nth i = some (-1))) →
  2 * signs.nth 0 + 0 * signs.nth 1 + 2 * signs.nth 2 + 2 * signs.nth 3 + 2 * signs.nth 4 + 0 * signs.nth 5 + 2 * signs.nth 6 + 3 * signs.nth 7 ≠ 0 := sorry

end sum_of_signs_20222023_impossible_l775_775457


namespace correct_choices_l775_775294

-- Definitions based on each condition and correctness of choices
def sampling_example (x : Nat) : Prop := 
  "A quality inspector takes a sample from a uniformly moving production line every 10 minutes for a certain indicator test." ≠ 
  "This is stratified sampling."

def histogram_area : Prop :=
  "In the frequency distribution histogram, the sum of the areas of all small rectangles is 1."

def regression_line : Prop :=
  ∀ (x y : ℕ), "In the regression line equation, \\(\overset{\land }{y}=0.2x+12\\), when the variable \\(x\\) increases by one unit, the variable \\(y\\) definitely increases by 0.2 units." ≠ 
  "Variable y increases on average by 0.2 units."

def categorical_variables (X Y : Type) (k : ℕ) : Prop :=
  "For two categorical variables \\(X\\) and \\(Y\\), after calculating the statistic \\(K^{2}\\)'s observed value \\(k\\), 
   the larger the observed value \\(k\\), the more confident we are that X and Y are related."

-- The proof statement
theorem correct_choices : 
  (sampling_example → False) ∧ histogram_area ∧ (regression_line → False) ∧ categorical_variables :=
  by
    sorry

end correct_choices_l775_775294


namespace jane_change_l775_775877

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l775_775877


namespace lcm_of_a_c_l775_775950

theorem lcm_of_a_c (a b c : ℕ) (h1 : Nat.lcm a b = 20) (h2 : Nat.lcm b c = 24) : Nat.lcm a c = 30 := by
  sorry

end lcm_of_a_c_l775_775950


namespace number_of_solutions_eq_l775_775519

theorem number_of_solutions_eq (a n : ℕ) (x y : ℕ → ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ n → y i ≥ 0) ∧ 
  (∑ i in Finset.range n, (i + 1) * x (i + 1)) = a  ↔ 
  (∑ i in Finset.range n, (i + 1) * y (i + 1)) = a - (n * (n + 1)) / 2 :=
begin
  sorry
end

end number_of_solutions_eq_l775_775519


namespace Trent_onions_chopped_per_pot_l775_775982

theorem Trent_onions_chopped_per_pot (x : ℕ) (h1: ∀ (n : ℕ), Trent_cries_2_tears_for_every_3_onions (n) = (2 * n) / 3) 
                                      (h2: Trent_makes_pots_of_soup = 6) 
                                      (h3: Trent_cries_total_tears = 16) : x = 4 := 
by
  have h_total_onions := 6 * x
  have h_total_tears := (2 / 3) * h_total_onions
  rw h1 at h_total_tears
  sorry

end Trent_onions_chopped_per_pot_l775_775982


namespace remaining_blocks_correct_l775_775522

-- Define the initial number of blocks
def initial_blocks : ℕ := 59

-- Define the number of blocks used
def used_blocks : ℕ := 36

-- Define the remaining blocks equation
def remaining_blocks : ℕ := initial_blocks - used_blocks

-- Prove that the number of remaining blocks is 23
theorem remaining_blocks_correct : remaining_blocks = 23 := by
  sorry

end remaining_blocks_correct_l775_775522


namespace num_valid_segment_sets_l775_775884

-- Define the main theorem for the problem
theorem num_valid_segment_sets 
  (A B C D E : Point)
  (on_circle : ∀ P : Point, P ∈ {A, B, C, D, E}) :
  (∃ S T : Finset Point, S ≠ ∅ ∧ T ≠ ∅ ∧ S ∩ T = ∅ ∧ S ∪ T = {A, B, C, D, E}
    ∧ (∀ (x ∈ S) (y ∈ T), connected x y))
   → card ({ S : Finset (Point × Point) |
       ∃ (S T : Finset Point), S ≠ ∅ ∧ T ≠ ∅ ∧ S ∩ T = ∅ ∧ S ∪ T = {A, B, C, D, E}
       ∧ (∀ (x ∈ S) (y ∈ T), (x, y) ∈ S) ∧
         (∀ (P : Point), P ∈ {A, B, C, D, E} → (∃ (Q : Point), Q ∈ {A, B, C, D, E} ∧ (P, Q) ∈ S))
    }) = 195 :=
sorry

end num_valid_segment_sets_l775_775884


namespace omega_value_increasing_intervals_l775_775773

noncomputable def f (omega x : ℝ) : ℝ := (sin(omega * x) + cos(omega * x))^2 + 2 * (cos(omega * x))^2

theorem omega_value (omega : ℝ) (h1: ω > 0) (h2: ∀ x, f ω (x + π / (3 * ω)) = f ω x): ω = 3 / 2 := 
sorry

noncomputable def g (omega x : ℝ) : ℝ := f omega (x - π / 2)

theorem increasing_intervals (ω : ℝ) (k : ℤ) (h_omega: ω = 3 / 2) :
  ∀ x, (2 / 3 * k * π + π / 4 ≤ x ∧ x ≤ 2 / 3 * k * π + 7 * π / 12) ↔ 0 < cos (3 / 2 * x - π / 2) :=
sorry

end omega_value_increasing_intervals_l775_775773


namespace quadrilateral_incircle_iff_l775_775131

section
variables {A B C D P E F G H : Type*} [euclidean_geometry] (ABCD : quadrilateral A B C D)
variables (h1 : ∃ (P : point), P ∈ line AC ∩ line BD)
variables (h2 : perpendicular P E (line_segment A B))
variables (h3 : perpendicular P F (line_segment B C))
variables (h4 : perpendicular P G (line_segment C D))
variables (h5 : perpendicular P H (line_segment D A))

/-- A convex quadrilateral ABCD has an incircle if and only if 
    1 / PE + 1 / PG = 1 / PF + 1 / PH. -/
theorem quadrilateral_incircle_iff (h : ∀ (p : P), quadrilateral.has_incircle ABCD ↔ (1 / length (line_segment P E) + 1 / length (line_segment P G) = 1 / length (line_segment P F) + 1 / length (line_segment P H))) : 
  ∀ {PE PF PG PH : ℝ}, quadrilateral.has_incircle ABCD ↔ ((1 / PE) + (1 / PG) = (1 / PF) + (1 / PH)) :=
sorry
end

end quadrilateral_incircle_iff_l775_775131


namespace transformed_data_stddev_l775_775829

theorem transformed_data_stddev {x : ℕ → ℝ} (n : ℕ) (h : n = 10) (σ : ℝ) (hσ : σ = 8) :
  let y := λ i, 3 * x i - 1 in
  stddev (finset.range n) y = 24 :=
by sorry

end transformed_data_stddev_l775_775829


namespace find_c_l775_775007

theorem find_c (c : ℝ) : 
  (let A := 5 in
  let half_area := A / 2 in
  let line_eq := λ x,  3 * (x - c) / (3 - c) in
  let base := 3 - c in
  let height := 3 in
  let triangle_area := (base * height) / 2 in
  let shaded_area := triangle_area - 1 in
  shaded_area = half_area) → 
  c = 2 / 3 := 
begin
  intros h,
  -- Proof omitted for demonstration
  sorry
end

end find_c_l775_775007


namespace police_officers_on_duty_l775_775164

-- Define the conditions as given in the problem statement
def percent := 0.17
def female_officers := 500
def female_officers_on_duty := percent * female_officers
def total_officers_on_duty := 2 * female_officers_on_duty

-- The proof statement
theorem police_officers_on_duty : total_officers_on_duty = 170 := 
  sorry

end police_officers_on_duty_l775_775164


namespace parabola_focus_coordinates_l775_775343

theorem parabola_focus_coordinates (x y : ℝ) (h : y = 4 * x^2) : (0, 1/16) = (0, 1/16) :=
by
  sorry

end parabola_focus_coordinates_l775_775343


namespace trapezoids_count_l775_775634

theorem trapezoids_count :
  ∃ (n : ℕ), n = 49 ∧
  ∃ (b d : ℕ),  ∀ (x y z w : ℕ),
  (A = (0, 0)) ∧ (B = (x, 2 * x)) ∧ (D = (y, 3 * y)) ∧
  (C = (z, w)) ∧ (0 < x) ∧ (0 < y) ∧ (area ((0, 0), (y, 3 * y), (z, w), (x, 2 * x)) = 500000) :=
sorry

end trapezoids_count_l775_775634


namespace percentage_of_boring_grinding_approx_l775_775868

noncomputable def boringGrindingPercentage (total_original_gameplay : ℕ) (expansion_gameplay : ℕ) (enjoyable_gameplay : ℕ) : ℚ :=
  let total_gameplay := total_original_gameplay + expansion_gameplay
  let boring_grinding := total_gameplay - enjoyable_gameplay
  (boring_grinding * 100) / total_gameplay

theorem percentage_of_boring_grinding_approx :
  boringGrindingPercentage 100 30 50 ≈ 61.54 :=
by
  sorry

end percentage_of_boring_grinding_approx_l775_775868


namespace quadrilateral_area_l775_775733

theorem quadrilateral_area 
  (z : ℂ) (h1 : (z * (conj z)^3 + (conj z) * z^3 = 450))
  (h2 : ∃ (x y : ℤ), z = x + y * complex.I) :
  ∃ (area : ℕ), area = 80 := 
sorry

end quadrilateral_area_l775_775733


namespace lines_perpendicular_l775_775899

noncomputable def slope_1 (A a : ℝ) : ℝ := sin A / -a
noncomputable def slope_2 (B b : ℝ) : ℝ := b / sin B

theorem lines_perpendicular (A B a b c : ℝ) (h1 : sin A + a * (0 : ℝ) + c = 0) 
  (h2 : b * (0 : ℝ) - (0 : ℝ) * sin B + c = 0) :
  slope_1 A a * slope_2 B b = -1 :=
by
  sorry

end lines_perpendicular_l775_775899


namespace slower_tourist_l775_775560

theorem slower_tourist (a : ℝ) (h : 0 < a) : 11 * (0.9 * a) < 10 * a :=
by
  calc 
    11 * (0.9 * a) = 9.9 * a : by ring
    ... < 10 * a : by linarith

end slower_tourist_l775_775560


namespace cycling_speed_l775_775557

-- Definitions based on given conditions.
def ratio_L_B : ℕ := 1
def ratio_B_L : ℕ := 2
def area_of_park : ℕ := 20000
def time_in_minutes : ℕ := 6

-- The question translated to Lean 4 statement.
theorem cycling_speed (L B : ℕ) (h1 : ratio_L_B * B = ratio_B_L * L)
  (h2 : L * B = area_of_park)
  (h3 : B = 2 * L) :
  (2 * L + 2 * B) / (time_in_minutes / 60) = 6000 := by
  sorry

end cycling_speed_l775_775557


namespace mean_of_numbers_is_10_l775_775942

-- Define the list of numbers
def numbers : List ℕ := [6, 8, 9, 11, 16]

-- Define the length of the list
def n : ℕ := numbers.length

-- Define the sum of the list
def sum_numbers : ℕ := numbers.sum

-- Define the mean (average) calculation for the list
def average : ℕ := sum_numbers / n

-- Prove that the mean of the list is 10
theorem mean_of_numbers_is_10 : average = 10 := by
  sorry

end mean_of_numbers_is_10_l775_775942


namespace f_neg_five_l775_775553

def f (a b x : ℝ) := a * x + b * Real.sin x + 1

theorem f_neg_five (a b : ℝ) (h : f a b 5 = 7) : f a b (-5) = -5 :=
by 
  sorry

end f_neg_five_l775_775553


namespace mixture_cost_correct_l775_775543

-- Definitions and conditions from a)
abbreviation C := ℝ -- cost per pound of milk powder and coffee in June
constant priceInJulyMilk : ℝ := 0.4 -- cost per pound of milk powder in July
constant priceInJulyCoffee : ℝ
constant mixtureCost3lbs : ℝ

axiom eq_price_june : ∀ c : ℝ, priceInJulyCoffee = 3 * c  
axiom priceMilkJuly: ∀ c : ℝ, 0.4 * c = 0.4 → c = 1

-- Question statement with its conditions
theorem mixture_cost_correct :
  ∀ c : ℝ,
  priceInJulyCoffee = 3 * c →
  priceInJulyMilk = 0.4 →
  c = 1 →
  mixtureCost3lbs = 1.5 * priceInJulyMilk + 1.5 * priceInJulyCoffee →
  mixtureCost3lbs = 5.1 :=
by
  intros
  sorry

end mixture_cost_correct_l775_775543


namespace quadruples_solution_l775_775167

noncomputable
def valid_quadruples (x1 x2 x3 x4 : ℝ) : Prop :=
  (x1 + x2 * x3 * x4 = 2) ∧
  (x2 + x1 * x3 * x4 = 2) ∧
  (x3 + x1 * x2 * x4 = 2) ∧
  (x4 + x1 * x2 * x3 = 2) ∧
  (x1 ≠ 0) ∧ (x2 ≠ 0) ∧ (x3 ≠ 0) ∧ (x4 ≠ 0)

theorem quadruples_solution (x1 x2 x3 x4 : ℝ) :
  valid_quadruples x1 x2 x3 x4 ↔ 
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) := 
by sorry

end quadruples_solution_l775_775167


namespace trajectory_equation_range_DE_FG_l775_775370

section Problem1

/-- Given a circle with center A and equation (x-2)^2 + y^2 = 64, moving point M on the circle,
    point B(-2,0), and the perpendicular bisector of segment BM intersects AM at point P,
    prove that the equation of trajectory E of point P is given by (x^2) / 16 + (y^2) / 12 = 1. -/
theorem trajectory_equation :
  (∃ A B : ℝ × ℝ,
    ∀ M : ℝ × ℝ,
    (M.1 - 2) ^ 2 + M.2 ^ 2 = 64 →
    let P : ℝ × ℝ := (A.1 + B.1) / 2, (A.2 + B.2) / 2 in
    (P.1) ^ 2 / 16 + (P.2) ^ 2 / 12 = 1) :=
sorry

end Problem1

section Problem2

/-- Given two mutually perpendicular lines l₁ and l₂ through point A intersecting curve E 
    at points D, E, F, and G respectively, and the equation of E as (x^2) / 16 + (y^2) / 12 = 1,
    prove that the range of values for |DE| + |FG| is [96 / 7, 14). -/
theorem range_DE_FG :
  (∃ A : ℝ × ℝ,
    ∀ (l₁ l₂ : ℝ × ℝ → Prop),
      (∀ D E F G : ℝ × ℝ,
        (l₁ ∧ l₂) →
        let |DE| := dist D E, |FG| := dist F G in
        |DE| + |FG| = [96 / 7, 14)) :=
sorry

end Problem2

end trajectory_equation_range_DE_FG_l775_775370


namespace circle_equation_tangent_lines_l775_775369

theorem circle_equation {a b : ℝ} (ha : b = a + 1) (hP : (5 - a)^2 + (4 - b)^2 = 2) (hQ : (3 - a)^2 + (6 - b)^2 = 2) : 
  (∃ (a b : ℝ), (x - 4)^2 + (y - 5)^2 = 2) :=
by {
  -- Provided conditions
  have h_center : b = a + 1 := ha,
  have h_point_P : (5 - a)^2 + (4 - b)^2 = 2 := hP,
  have h_point_Q : (3 - a)^2 + (6 - b)^2 = 2 := hQ,
  -- Solution proving (x - 4)^2 + (y - 5)^2 = 2
  sorry
}

theorem tangent_lines {x y : ℝ} (h_center : x = 4) (h_y_center : y = 5) (h_radius : (x - 4)^2 + (y - 5)^2 = 2):
  (∃ k : ℝ, y = k * (x - 1) ∧ (|4 * k - 5 - k| / real.sqrt (1 + k^2)) = sqrt 2) :=
by {
  -- Provided conditions
  have h1_center : x = 4 := h_center,
  have h1_y_center : y = 5 := h_y_center,
  have h1_radius : (x - 4)^2 + (y - 5)^2 = 2 := h_radius,
  -- Solution proving tangent lines equations
  sorry
}

end circle_equation_tangent_lines_l775_775369


namespace find_q_l775_775949

def is_quadratic (q : ℚ[X]) : Prop :=
  q.degree = 2

theorem find_q {q : ℚ[X]} (h_q_is_quadratic : is_quadratic q)
  (h_asymptote_neg1 : eval (-1 : ℚ) q = 0)
  (h_asymptote_pos1 : eval (1 : ℚ) q = 0)
  (h_q2 : eval (2 : ℚ) q = 6) :
  q = (2 : ℚ) * X^2 - 2 :=
by {
  sorry
}

end find_q_l775_775949


namespace range_of_k_line_equation_l775_775410
noncomputable def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 = 1
def line_eq (x y k : ℝ) : Prop := y = k * x - 1

theorem range_of_k (k : ℝ) : 
  (∀ x y, hyperbola_eq x y → line_eq x y k) →
  1 < k ∧ k < sqrt 2 := sorry

theorem line_equation (k : ℝ) :
  (∀ x y, hyperbola_eq x y → line_eq x y k) →
  abs (distance (k - sqrt(6)/2) (k * (sqrt(6)/2 - x) -1)) = 2 * sqrt 5 → 
  y = sqrt(6)/2 * x - 1 := sorry

end range_of_k_line_equation_l775_775410


namespace cost_of_blue_hat_is_six_l775_775985

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

end cost_of_blue_hat_is_six_l775_775985


namespace mean_of_three_numbers_l775_775190

theorem mean_of_three_numbers (a : Fin 12 → ℕ) (x y z : ℕ) 
  (h1 : (Finset.univ.sum a) / 12 = 40)
  (h2 : ((Finset.univ.sum a) + x + y + z) / 15 = 50) :
  (x + y + z) / 3 = 90 := 
by
  sorry

end mean_of_three_numbers_l775_775190


namespace trees_in_backyard_l775_775916

-- Define the initial counts and the operations
def initial_trees : ℕ := 13
def trees_removed : ℕ := 3
def new_trees_bought : ℕ := 18
def trees_planted_initially : ℕ := 12
def additional_percentage : ℚ := 0.25

-- Define the functions to calculate intermediate values
def remaining_trees : ℕ := initial_trees - trees_removed
def total_planted_initially : ℕ := trees_planted_initially
def additional_trees_planted : ℕ :=
  (trees_planted_initially * additional_percentage).to_int

def final_trees : ℕ :=
  remaining_trees + total_planted_initially + additional_trees_planted

-- Lean proposition stating the proof problem
theorem trees_in_backyard : final_trees = 25 := 
by
  sorry

end trees_in_backyard_l775_775916


namespace product_of_diagonal_lengths_l775_775521

noncomputable def prod_diagonal_lengths (n : ℕ) (h : 0 < n) : ℂ :=
  ∏ k in finset.range (n), complex.abs (1 - complex.exp (2 * real.pi * complex.I * k / n))

theorem product_of_diagonal_lengths (n : ℕ) (h : 1 < n) :
  prod_diagonal_lengths n h = ↑n := sorry

end product_of_diagonal_lengths_l775_775521


namespace max_value_f_l775_775318

open Real

/-- Determine the maximum value of the function f(x) = 1 / (1 - x * (1 - x)). -/
theorem max_value_f (x : ℝ) : 
  ∃ y, y = (1 / (1 - x * (1 - x))) ∧ y ≤ 4/3 ∧ ∀ z, z = (1 / (1 - x * (1 - x))) → z ≤ 4/3 :=
by
  sorry

end max_value_f_l775_775318


namespace only_n_is_three_l775_775357

/-- For all integers \( n \geq 2 \), the only \( n \) such that 
  there exists an arrangement of \( 1, 2, \ldots, n \) in a row 
  where the sum of the first \( k \) numbers is divisible by \( k \) 
  for all \( 1 \leq k \leq n \), is \( n = 3 \). -/
theorem only_n_is_three (n : ℕ) (h : n ≥ 2) :
  (∃ f : Fin n → Fin n, 
    ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (∑ i in Finset.range k, (f i).val) % k = 0) ↔ n = 3 := 
by 
  sorry

end only_n_is_three_l775_775357


namespace sequence_remainder_prime_l775_775470

theorem sequence_remainder_prime (p : ℕ) (hp : Nat.Prime p) (x : ℕ → ℕ)
  (h1 : ∀ i, 0 ≤ i ∧ i < p → x i = i)
  (h2 : ∀ n, n ≥ p → x n = x (n-1) + x (n-p)) :
  (x (p^3) % p) = p - 1 :=
sorry

end sequence_remainder_prime_l775_775470


namespace sequence_general_term_l775_775774

-- Define the sequence and the sum of the sequence
def Sn (n : ℕ) : ℕ := 3 + 2^n

def an (n : ℕ) : ℕ :=
  if n = 1 then 5 else 2^(n - 1)

-- Proposition stating the equivalence
theorem sequence_general_term (n : ℕ) : 
  (n = 1 → an n = 5) ∧ (n ≠ 1 → an n = 2^(n - 1)) :=
by 
  sorry

end sequence_general_term_l775_775774


namespace correct_conclusions_l775_775762

-- Define the arithmetic sequence and its sum
variables {a_n : ℕ → ℤ} {d a_1 : ℤ}

-- Sum function for the first n terms of the arithmetic sequence
def S (n : ℤ) := n * (2 * a_1 + (n - 1) * d) / 2

-- Given condition: S₆ = S₁₂
axiom sum_condition : S 6 = S 12

-- Prove that the following conclusions are correct
theorem correct_conclusions :
  (2 * a_1 + 17 * d = 0) ∧
  (S 18 = 0) ∧
  (d > 0 → a_n 6 + a_n 14 > 0) ∧
  ¬ (d < 0 → |a_n 6| > |a_n 14|) :=
by
  -- Variables, assumptions, and goals are established here, proof would follow
  sorry

end correct_conclusions_l775_775762


namespace largest_number_with_digits_sum_17_l775_775236

noncomputable def sum_of_digits (n : ℕ) : ℕ := 
  (Integer.digits 10 n).sum

def is_digit_4_or_5 (n : ℕ) : Prop := 
  ∀ (d ∈ Integer.digits 10 n), d = 4 ∨ d = 5

theorem largest_number_with_digits_sum_17 : 
  ∃ n, is_digit_4_or_5 n ∧ sum_of_digits n = 17 ∧ ∀ m, is_digit_4_or_5 m ∧ sum_of_digits m = 17 → m ≤ n :=
begin
  sorry
end

end largest_number_with_digits_sum_17_l775_775236


namespace semicircle_property_l775_775630

-- Define the semicircle and the given points A, B, M, P, and Q
open_locale real

structure Semicircle (A B M : ℝ × ℝ) :=
(center : ℝ × ℝ := M) (radius : ℝ := dist A M)
(property_A : dist A M = radius)
(property_B : dist B M = radius)

noncomputable def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
((A.1 + B.1) / 2, (A.2 + B.2) / 2)

noncomputable def intersection_point
  (A B P Q M : ℝ × ℝ) (h_parallel : ∃ k : ℝ, Q = (P.1 + k, P.2))
  : ℝ × ℝ := sorry  -- intersection point calculation

theorem semicircle_property
  (A B P Q M : ℝ × ℝ)
  (h_semicircle : Semicircle A B M)
  (h_P_semicircle : dist P M = h_semicircle.radius)
  (h_point_diff : P ≠ A ∧ P ≠ B)
  (h_midpoint : Q = midpoint A P)
  (h_parallel : ∃ k : ℝ, Q = (P.1 + k, P.2))
  (S := intersection_point A B P Q M h_parallel)
  : dist P M = dist P S :=
by
  sorry -- Proof to be constructed

end semicircle_property_l775_775630


namespace construct_triangle_from_intersections_l775_775312

-- Define the existence of the circumcircle and the three points M, F, and S
variables {A B C : Point} {O : Point} (M F S : Point)
variable h_distinct_noncollinear : M ≠ F ∧ F ≠ S ∧ M ≠ S ∧ ¬ collinear M F S

-- Define that M, F, and S are the second intersections of the altitude, angle bisector, and median from A respectively
variable hM : isSecondIntersectionOfAltitude M A O
variable hF : isSecondIntersectionOfAngleBisector F A O
variable hS : isSecondIntersectionOfMedian S A O

-- Define that the circumcircle exists
variable h_circumcircle : ∃ k : Circle, k.center = O ∧ k.contains A ∧ k.contains B ∧ k.contains C

-- State the theorem
theorem construct_triangle_from_intersections (h_distinct_noncollinear : h_distinct_noncollinear)
  (hM : hM) (hF : hF) (hS : hS) (h_circumcircle : h_circumcircle) :
  ∃ (A B C : Point), triangle A B C :=
sorry

end construct_triangle_from_intersections_l775_775312


namespace tuning_day_method_l775_775533

theorem tuning_day_method :
  let pi_approx := 3.14159
  ∃ p q : ℕ,
  let first := 31 / 10, second := 49 / 15, next := (31 + 49) / (10 + 15) in
  (first < pi_approx ∧ pi_approx < second) →
  (let third := (47 / 15 + 63 / 20) / 2 in
  (47 / 15 < pi_approx ∧ pi_approx < 16 / 5) →
  (63 / 20 < pi_approx ∧ pi_approx < 22 / 7) → q = 7 ∧ p = 22) sorry

end tuning_day_method_l775_775533


namespace triangle_quadrilateral_perimeter_l775_775635

theorem triangle_quadrilateral_perimeter
  (triangle_sides : ℕ × ℕ × ℕ)
  (quad_side : ℕ)
  (triangle_perimeter : ℕ)
  (quad_perimeter : ℕ)
  (total_perimeter : ℕ) :
    triangle_sides = (6, 8, 10) →
    quad_side = 5 →
    triangle_perimeter = 24 →
    quad_perimeter = 20 →
    total_perimeter = triangle_perimeter + quad_perimeter →
    total_perimeter = 44 :=
by
  intros
  simp *
  sorry

end triangle_quadrilateral_perimeter_l775_775635


namespace four_x_equals_four_l775_775078

theorem four_x_equals_four (x : ℝ) (n : ℝ) (e : ℝ) (h1 : 1 = x) (h2 : 2 = 2 * x) (h3 : 2 * x * e = 10) (h4 : e = 5) : n = 4 :=
by {
  -- use h1 to replace x
  have hx: x = 1 := eq.symm h1,
  -- the rest is a trivial steps
  -- substitute x by 1 in h3 and h4
  -- sorry used to skip the proof
  sorry
}

end four_x_equals_four_l775_775078


namespace pension_amount_l775_775618

theorem pension_amount 
  (a b p q : ℝ) (b_neq_a : b ≠ a)
  (k x y : ℝ)
  (h1 : y = k * real.sqrt x)
  (h2 : y + p = k * real.sqrt (x + a))
  (h3 : y + q = k * real.sqrt (x + b)) : 
  y = (a * q^2 - b * p^2) / (2 * (b * p - a * q)) :=
sorry

end pension_amount_l775_775618


namespace max_value_of_f_in_interval_l775_775554

noncomputable def f : ℝ → ℝ := λ x, 2 * x^3 - 9 * x^2 + 12 * x + 1

theorem max_value_of_f_in_interval : ∃ x ∈ (set.Icc 0 3), f x = 10 := 
by {
  sorry
}

end max_value_of_f_in_interval_l775_775554


namespace total_leftover_tarts_l775_775637

def cherry_tarts := 0.08
def blueberry_tarts := 0.75
def peach_tarts := 0.08

theorem total_leftover_tarts : cherry_tarts + blueberry_tarts + peach_tarts = 0.91 := by
  sorry

end total_leftover_tarts_l775_775637


namespace intersected_area_of_triangles_l775_775061

noncomputable def area_of_intersected_figure :=
  let s₁ := 4
  let s₂ := 6
  let s₃ := 8
  let θ₂ := 45
  let θ₃ := 90
  let area := 4 * Real.sqrt 3
  area

theorem intersected_area_of_triangles :
  let s₁ := 4
  let s₂ := 6
  let s₃ := 8
  let θ₂ := 45
  let θ₃ := 90
  calculate_intersected_area s₁ s₂ s₃ θ₂ θ₃ = 4 * Real.sqrt 3 :=
by
  sorry

end intersected_area_of_triangles_l775_775061


namespace orthocenter_lies_on_AD_l775_775854

open EuclideanGeometry

-- Definitions for the problem

variables (A B C D T O1 O2 O3 : Point) (triangle : Triangle) (circumcenter : Triangle → Point)
(h1 : ∠A < 90) (h2 : isOnLine T BC) (h3 : acuteTriangle A T D)
(h4 : circumcenter (triangle A B T) = O1) (h5 : circumcenter (triangle A D T) = O2) 
(h6 : circumcenter (triangle C D T) = O3) 

-- Goal to prove
theorem orthocenter_lies_on_AD : 
  orthocenter (triangle O1 O2 O3) ∈ AD := 
sorry

end orthocenter_lies_on_AD_l775_775854


namespace average_rainfall_correct_l775_775660

-- Define the monthly rainfall
def january_rainfall := 150
def february_rainfall := 200
def july_rainfall := 366
def other_months_rainfall := 100

-- Calculate total yearly rainfall
def total_yearly_rainfall := 
  january_rainfall + 
  february_rainfall + 
  july_rainfall + 
  (9 * other_months_rainfall)

-- Calculate total hours in a year
def days_per_month := 30
def total_days_in_year := 12 * days_per_month
def hours_per_day := 24
def total_hours_in_year := total_days_in_year * hours_per_day

-- Calculate average rainfall per hour
def average_rainfall_per_hour := 
  total_yearly_rainfall / total_hours_in_year

theorem average_rainfall_correct :
  average_rainfall_per_hour = (101 / 540) := sorry

end average_rainfall_correct_l775_775660


namespace joan_exam_time_difference_l775_775119

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l775_775119


namespace value_of_abs_diff_l775_775205

-- Let a_1 and b_1 be the largest elements in the representations of 2021 based on the conditions
def satisfies_conditions (a b : ℕ) : Prop :=
  ∃ (a_2 a_3 ⋯ a_m b_2 b_3 ⋯ b_n : ℕ), 
  2021 = (fact a * fact a_2 * fact a_3 * ⋯ * fact a_m) / (fact b * fact b_2 * fact b_3 * ⋯ * fact b_n) ∧
  a ≥ a_2 ∧ a_2 ≥ ⋯ ∧ a_m ∧ b ≥ b_2 ∧ b_2 ≥ ⋯ ∧ b_n ∧
  ∀ (x y : ℕ), x ≥ a ∨ y ≥ b → x + y ≥ a + b

-- The main theorem we want to prove
theorem value_of_abs_diff :
  ∃ (a b : ℕ), satisfies_conditions a b ∧ |a - b| = 4
  :=
sorry

end value_of_abs_diff_l775_775205


namespace impossible_to_have_only_stacks_of_three_l775_775987

theorem impossible_to_have_only_stacks_of_three (n J : ℕ) (h_initial_n : n = 1) (h_initial_J : J = 1001) :
  (∀ n J, (n + J = 1002) → (∀ k : ℕ, 3 * k ≤ J → k + 3 * k ≠ 1002)) 
  :=
sorry

end impossible_to_have_only_stacks_of_three_l775_775987


namespace simplify_expression_l775_775739

theorem simplify_expression (a : ℝ) (h : a > 0) : 
  (a^2 / (a * (a^3) ^ (1 / 2)) ^ (1 / 3)) = a^(7 / 6) :=
sorry

end simplify_expression_l775_775739


namespace sequence_2018_value_l775_775405

theorem sequence_2018_value :
  ∃ a : ℕ → ℤ, a 1 = 3 ∧ a 2 = 6 ∧ (∀ n, a (n + 2) = a (n + 1) - a n) ∧ a 2018 = -3 :=
sorry

end sequence_2018_value_l775_775405


namespace log_base_equation_l775_775084

theorem log_base_equation (x : ℝ) (h : log x 16 = 0.8) : x = 32 := 
sorry

end log_base_equation_l775_775084


namespace inequality_gt_zero_l775_775514

theorem inequality_gt_zero (x y : ℝ) : x^2 + 2*y^2 + 2*x*y + 6*y + 10 > 0 :=
  sorry

end inequality_gt_zero_l775_775514


namespace necessary_and_sufficient_condition_l775_775062

variable (k1 k2 : ℝ) 

def l1_parallel_l2 (k1 k2 : ℝ) : Prop := 
  k1 = k2

theorem necessary_and_sufficient_condition (k1 k2 : ℝ) :
  (∀ x y : ℝ, (k1 * x + y + 1 = 0) → (k2 * x + y - 1 = 0)) ↔ k1 = k2 :=
begin
  sorry
end

end necessary_and_sufficient_condition_l775_775062


namespace find_distance_between_posters_and_wall_l775_775596

-- Definitions for given conditions
def poster_width : ℝ := 29.05
def num_posters : ℕ := 8
def wall_width : ℝ := 394.4

-- The proof statement: find the distance 'd' between posters and ends
theorem find_distance_between_posters_and_wall :
  ∃ d : ℝ, (wall_width - num_posters * poster_width) / (num_posters + 1) = d ∧ d = 18 := 
by {
  -- The proof would involve showing that this specific d meets the constraints.
  sorry
}

end find_distance_between_posters_and_wall_l775_775596


namespace area_of_triangle_l775_775141

def a : ℝ × ℝ := (3, 2)
def b : ℝ × ℝ := (-1, 5)

theorem area_of_triangle : 
  let ab_mat := ![[3, -1], [2, 5]] in 
  (1 / 2) * |ab_mat.det|  = 8.5 :=
by
  sorry

end area_of_triangle_l775_775141


namespace octahedral_die_painting_l775_775643

theorem octahedral_die_painting (faces : Finset ℕ) (h_faces : faces = {1, 2, 3, 4, 5, 6, 7, 8}) :
  (∃ (red_faces : Finset ℕ), red_faces.card = 3 ∧ (∀ (a b c ∈ red_faces), a + b + c ≠ 9) ∧ red_faces ⊆ faces) →
  (finset.card {red_faces : Finset ℕ | red_faces.card = 3 ∧ (∀ (a b c ∈ red_faces, a + b + c ≠ 9)) ∧ red_faces ⊆ faces}) = 32 :=
sorry

end octahedral_die_painting_l775_775643


namespace part1_probability_part2_distribution_expectation_l775_775849

theorem part1_probability :
  ((P (C after B) = 3/4) ∧ (P (A after C) = 2/5))
  → (P (A on third) = 3/10) := by
  sorry

theorem part2_distribution_expectation :
  ((P (B after A) = 1/3) ∧ (P (C after A) = 2/3) ∧
   (P (A after B) = 1/4) ∧ (P (C after B) = 3/4) ∧
   (P (A after C) = 2/5) ∧ (P (B after C) = 3/5))
  → (P (X = 1) = 13/20) ∧ (P (X = 2) = 7/20) ∧ (E(X) = 27/20) := by
  sorry

end part1_probability_part2_distribution_expectation_l775_775849


namespace functional_eq_solution_l775_775746

theorem functional_eq_solution (f : ℝ → ℝ)
  (h : ∀ x y : ℝ, f (2 * x + f y) = x + y + f x) :
  ∀ x : ℝ, f x = x :=
sorry

end functional_eq_solution_l775_775746


namespace sum_of_exponents_l775_775221

noncomputable def uniqueSum (s : ℕ) (m : ℕ → ℕ) (b : ℕ → ℤ) : Prop :=
  (∀ i j, i ≠ j → m i ≠ m j) ∧
  (∀ k, b k = 1 ∨ b k = -1) ∧
  ∃ (hs : Finset ℕ), hs.card = s ∧ (∑ k in hs, b k * (3 ^ m k)) = 2023

theorem sum_of_exponents (s : ℕ) (m : ℕ → ℕ) (b : ℕ → ℤ)
    (h : uniqueSum s m b) : (∑ k in Finset.range s, m k) = 21 :=
by
  revert s m b h
  sorry

end sum_of_exponents_l775_775221


namespace probability_red_first_blue_second_is_1_over_11_l775_775266

noncomputable def probability_red_first_blue_second : ℚ :=
let total_marbles : ℚ := 12
let red_marbles : ℚ := 4
let blue_marbles : ℚ := 3
let prob_red_first : ℚ := red_marbles / total_marbles
let prob_blue_second_given_red_first : ℚ := blue_marbles / (total_marbles - 1)
in prob_red_first * prob_blue_second_given_red_first

theorem probability_red_first_blue_second_is_1_over_11 :
  probability_red_first_blue_second = 1 / 11 :=
by
  sorry

end probability_red_first_blue_second_is_1_over_11_l775_775266


namespace culture_growth_l775_775274

/-- Define the initial conditions and growth rates of the bacterial culture -/
def initial_cells : ℕ := 5

def growth_rate1 : ℕ := 3
def growth_rate2 : ℕ := 2

def cycle_duration : ℕ := 3
def first_phase_duration : ℕ := 6
def second_phase_duration : ℕ := 6

def total_duration : ℕ := 12

/-- Define the hypothesis that calculates the number of cells at any point in time based on the given rules -/
theorem culture_growth : 
    (initial_cells * growth_rate1^ (first_phase_duration / cycle_duration) 
    * growth_rate2^ (second_phase_duration / cycle_duration)) = 180 := 
sorry

end culture_growth_l775_775274


namespace card_sum_probability_l775_775228

-- Given two cards drawn at random from a standard 52-card deck,
-- which both can individually be numbers from 2 through 10,
-- prove that the probability that their sum equals 13 is 40/663.
theorem card_sum_probability :
  let deck := (2:ℕ) :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: []
  52 = deck.cardinality * 4 -> 
  let total_combinations := (52.choose 2) in
  let favorable_combinations := 
    ((6 * 4) * 4 + 2 * 1 * 4) * (51.choose 1) in
  ((favorable_combinations : ℚ) / (total_combinations : ℚ)) = 40 / 663 :=
by sorry

end card_sum_probability_l775_775228


namespace ellipse_area_irrational_l775_775664

theorem ellipse_area_irrational 
  (a b : ℚ) : 
  let A := π * a * b in irrational A := 
sorry

end ellipse_area_irrational_l775_775664


namespace gcd_18_n_eq_6_l775_775720

theorem gcd_18_n_eq_6 (num_valid_n : Nat) :
  (num_valid_n = (List.range 200).count (λ n, (1 ≤ n ∧ n ≤ 200) ∧ (6 ∣ n) ∧ ¬(9 ∣ n))) →
  num_valid_n = 22 := by
  sorry

end gcd_18_n_eq_6_l775_775720


namespace fraction_simplification_l775_775239

theorem fraction_simplification : 1 + 1 / (1 - 1 / (2 + 1 / 3)) = 11 / 4 :=
by
  sorry

end fraction_simplification_l775_775239


namespace winning_candidate_votes_l775_775975

theorem winning_candidate_votes (V : ℝ) (h1 : 0.52 * V - 0.48 * V = 288) : 0.52 * V = 3744 := 
by 
  -- From the equation, solve for V
  have h2 : 0.04 * V = 288 := by linarith
  -- Then, solve for V
  have h3 : V = 288 / 0.04 := by exact (eq_div_of_mul_eq _ _ (ne_of_gt (by norm_num)) h2.symm).symm
  -- Substitute V in the winning candidate's vote expression
  have h4 : 0.52 * V = 0.52 * (288 / 0.04) := by rw [h3]
  -- Simplify the right-hand side
  have h5 : 0.52 * (288 / 0.04) = 0.52 * 7200 := by congr
  -- Simplify the multiplication
  show 0.52 * 7200 = 3744 from by norm_num

end winning_candidate_votes_l775_775975


namespace length_of_QR_in_cube_l775_775605

theorem length_of_QR_in_cube (edge_length : ℝ) (h_edge : edge_length = 2) :
  let PS := real.sqrt (edge_length ^ 2 + edge_length ^ 2),
      QS := PS / 2,
      RS := edge_length
  in real.sqrt ((QS ^ 2) + (RS ^ 2)) = real.sqrt 6 :=
by {
  sorry
}

end length_of_QR_in_cube_l775_775605


namespace max_f_value_l775_775010

open Real

noncomputable def f (x y : ℝ) : ℝ := min x (y / (x^2 + y^2))

theorem max_f_value : ∃ (x₀ y₀ : ℝ), (0 < x₀) ∧ (0 < y₀) ∧ (∀ (x y : ℝ), (0 < x) → (0 < y) → f x y ≤ f x₀ y₀) ∧ f x₀ y₀ = 1 / sqrt 2 :=
by 
  sorry

end max_f_value_l775_775010


namespace num_permutations_4_transpositions_l775_775693

/-- Define the permutation type on {1, 2, 3, 4, 5, 6} -/
def perm6 := equiv.perm (fin 6)

/-- Define the identity permutation on {1, 2, 3, 4, 5, 6} -/
def identity_perm6 : perm6 := equiv.refl (fin 6)

/-- Define a function that calculates the number of transpositions needed to reach the identity -/
noncomputable def num_transpositions (σ : perm6) : ℕ := 
  sorry

/-- Set of permutations of {1, 2, 3, 4, 5, 6} that require exactly 4 transpositions -/
def permutations_4_transpositions : finset perm6 := 
  finset.filter (λ σ, num_transpositions σ = 4) 
                (finset.univ)

theorem num_permutations_4_transpositions : 
  permutations_4_transpositions.card = 304 :=
sorry

end num_permutations_4_transpositions_l775_775693


namespace milan_billed_minutes_l775_775717

-- Define the conditions
def monthly_fee : ℝ := 2
def cost_per_minute : ℝ := 0.12
def total_bill : ℝ := 23.36

-- Define the number of minutes based on the above conditions
def minutes := (total_bill - monthly_fee) / cost_per_minute

-- Prove that the number of minutes is 178
theorem milan_billed_minutes : minutes = 178 := by
  -- Proof steps would go here, but as instructed, we use 'sorry' to skip the proof.
  sorry

end milan_billed_minutes_l775_775717


namespace sum_series_fraction_l775_775325

theorem sum_series_fraction :
  (∑ n in Finset.range 9, (1 : ℚ) / ((n + 2) * (n + 3))) = 9 / 22 := sorry

end sum_series_fraction_l775_775325


namespace sum_ps_at_10_l775_775473

def S : Finset (Vector (Fin 10) (Fin 2)) := 
  Finset.univ

def ps (s : Vector (Fin 10) (Fin 2)) : Polynomial (Fin 2) :=
  Polynomial.ofFinset (s.toFinset)

theorem sum_ps_at_10 : (∑ s in S, ps s.eval 10) = 512 := sorry

end sum_ps_at_10_l775_775473


namespace left_focus_distance_hyperbola_l775_775952

-- Definitions based on conditions
noncomputable def hyperbola (m : ℝ) (P : ℝ × ℝ) : Prop :=
  let right_focus := (m / 2, 0)
  let left_focus := (-m / 2, 0)
  let real_axis := m
  ∃ x y : ℝ, P = (x, y) ∧
  dist P right_focus = m ∧
  dist P left_focus = ? -- require this part to complete the proof definition

theorem left_focus_distance_hyperbola (m : ℝ) (P : ℝ × ℝ) 
  (h₁ : ∃ x y : ℝ, P = (x, y) ∧ 
    dist P (m / 2, 0) = m) :
  let left_focus_distance := dist P (-m / 2, 0) 
  left_focus_distance = 2 * m := sorry

end left_focus_distance_hyperbola_l775_775952


namespace circumcircle_centers_collinear_l775_775903

variables {A B C I A1 B1 C1 : Type}
  [NonIsoscelesTriangle ABC]
  [CircleCenter I]
  [TouchesAt A1 BC]
  [TouchesAt B1 CA]
  [TouchesAt C1 AB]

theorem circumcircle_centers_collinear :
  collinear (center_of_circumcircle_triangle A I A1)
            (center_of_circumcircle_triangle B I B1)
            (center_of_circumcircle_triangle C I C1) :=
sorry

end circumcircle_centers_collinear_l775_775903


namespace expand_polynomial_product_l775_775677

theorem expand_polynomial_product :
  (λ x : ℝ, (x^2 - 3*x + 3) * (x^2 + 3*x + 3)) = (λ x : ℝ, x^4 - 3*x^2 + 9) :=
by {
  funext x,
  -- The detailed proof steps would go here
  sorry
}

end expand_polynomial_product_l775_775677


namespace altitudes_feet_l775_775922

-- Define the type for points in the plane
variables {Point : Type*} [EuclideanGeometry Point]

-- Given points and conditions in the problem
variables (A B C A1 B1 C1 : Point)
variables (h1 : ∠ B1 A1 C = ∠ B A1 C1)
variables (h2 : ∠ A1 B1 C = ∠ A B1 C1)
variables (h3 : ∠ A1 C1 B = ∠ A C1 B1)

-- Translate the problem into a Lean theorem statement
theorem altitudes_feet (ABC : triangle A B C) : (is_feet_of_altitude A1 B1 C1 ABC) :=
by {
  sorry -- Proof to be filled in
}

end altitudes_feet_l775_775922


namespace factor_polynomial_l775_775679

theorem factor_polynomial (z : ℝ) : (70 * z ^ 20 + 154 * z ^ 40 + 224 * z ^ 60) = 14 * z ^ 20 * (5 + 11 * z ^ 20 + 16 * z ^ 40) := 
sorry

end factor_polynomial_l775_775679


namespace probability_distance_more_than_8km_l775_775442

-- Define basic constants and conditions.
def speed := 5 -- speed of each geologist in km/h
def roads := 8 -- number of roads

-- Probability that the distance between two geologists is more than 8 km after one hour.
theorem probability_distance_more_than_8km : 
  let possible_outcomes := roads * roads in
  let favorable_outcomes := 24 in
  (favorable_outcomes : ℝ) / possible_outcomes = 0.375 := 
by 
  sorry -- proof not required

end probability_distance_more_than_8km_l775_775442


namespace find_coefficient_b_l775_775925

noncomputable def polynomial_f (a b c d : ℝ) (x : ℝ) : ℝ :=
  a * x^3 + b * x^2 + c * x + d

theorem find_coefficient_b 
  (a b c d : ℝ)
  (h1 : polynomial_f a b c d (-2) = 0)
  (h2 : polynomial_f a b c d 0 = 0)
  (h3 : polynomial_f a b c d 2 = 0)
  (h4 : polynomial_f a b c d (-1) = 3) :
  b = 0 :=
sorry

end find_coefficient_b_l775_775925


namespace proof_problem_l775_775832

-- Defining a right triangle ΔABC with ∠BCA=90°
structure RightTriangle :=
(a b c : ℝ)  -- sides a, b, c with c as the hypotenuse
(hypotenuse_eq : c^2 = a^2 + b^2)  -- Pythagorean relation

-- Define the circles K1 and K2 with radii r1 and r2 respectively
structure CirclesOnTriangle (Δ : RightTriangle) :=
(r1 r2 : ℝ)  -- radii of the circles K1 and K2

-- Prove the relationship r1 + r2 = a + b - c
theorem proof_problem (Δ : RightTriangle) (C : CirclesOnTriangle Δ) :
  C.r1 + C.r2 = Δ.a + Δ.b - Δ.c := by
  sorry

end proof_problem_l775_775832


namespace inverse_value_l775_775075

-- Define the function g
noncomputable def g (x : ℝ) : ℝ := 25 / (7 + 2 * x)

-- Define the goal of the proof
theorem inverse_value {g : ℝ → ℝ}
  (h : ∀ y, g (g⁻¹ y) = y) :
  ((g⁻¹ 5)⁻¹) = -1 :=
by
  sorry

end inverse_value_l775_775075


namespace allocation_schemes_beijing_olympics_l775_775706

theorem allocation_schemes_beijing_olympics : 
  ∃ schemes : ℕ, schemes = 240 ∧ 
  ( ∀ (volunteers : Fin 5 → Fin 4), 
    ∃ assignments : volunteers → Fin 4, 
    (∀ v, ∃ p, assignments v = p) ∧ 
    (∀ p, ∃ v, assignments v = p) ) :=
sorry

end allocation_schemes_beijing_olympics_l775_775706


namespace polynomial_remainder_l775_775697

theorem polynomial_remainder (x : ℝ) : 
  let f := λ x : ℝ, x^3 - 4 * x + 6 in
  f (-3) = -9 :=
by
  let f := λ x : ℝ, x^3 - 4 * x + 6
  show f (-3) = -9
  sorry

end polynomial_remainder_l775_775697


namespace oliver_total_earnings_l775_775497

/-- Rates for different types of laundry items -/
def rate_regular : ℝ := 3
def rate_delicate : ℝ := 4
def rate_bulky : ℝ := 5

/-- Quantity of laundry items washed over three days -/
def quantity_day1_regular : ℝ := 7
def quantity_day1_delicate : ℝ := 4
def quantity_day1_bulky : ℝ := 2

def quantity_day2_regular : ℝ := 10
def quantity_day2_delicate : ℝ := 6
def quantity_day2_bulky : ℝ := 3

def quantity_day3_regular : ℝ := 20
def quantity_day3_delicate : ℝ := 4
def quantity_day3_bulky : ℝ := 0

/-- Discount on delicate clothes for the third day -/
def discount : ℝ := 0.2

/-- The expected earnings for each day and total -/
def earnings_day1 : ℝ :=
  rate_regular * quantity_day1_regular +
  rate_delicate * quantity_day1_delicate +
  rate_bulky * quantity_day1_bulky

def earnings_day2 : ℝ :=
  rate_regular * quantity_day2_regular +
  rate_delicate * quantity_day2_delicate +
  rate_bulky * quantity_day2_bulky

def earnings_day3 : ℝ :=
  rate_regular * quantity_day3_regular +
  (rate_delicate * quantity_day3_delicate * (1 - discount)) +
  rate_bulky * quantity_day3_bulky

def total_earnings : ℝ := earnings_day1 + earnings_day2 + earnings_day3

theorem oliver_total_earnings : total_earnings = 188.80 := by
  sorry

end oliver_total_earnings_l775_775497


namespace special_solutions_l775_775004

variable {C₁ C₂ : ℝ}

def F (x y y' : ℝ) : ℝ := y'^2 - (6 * x + y) * y' + 6 * x * y

theorem special_solutions :
  (∀ x, ∃ C₁ : ℝ, y(x) = C₁ * Real.exp x /- y1 = C₁ e^x -/) ∨
  (∀ x, ∃ C₂ : ℝ, y(x) = 3 * x^2 + C₂ /- y2 = 3x^2 + C₂ -/) :=
sorry

end special_solutions_l775_775004


namespace determine_crow_count_l775_775841

inductive Species
| Parrot
| Crow

open Species

structure Bird :=
(species : Species)
(statement : Bird → Prop)

def Alice := Bird.mk _ (λ D, Alice.species ≠ D.species)
def Ben := Bird.mk _ (λ E, E.species = Crow)
def Carla := Bird.mk _ (λ B, B.species = Crow)
def David := Bird.mk _ (λ birds, (birds.count (λ b, b.species = Parrot)) ≥ 3)
def Eliza := Bird.mk _ (λ C, C.species = Parrot)

def birds := [Alice, Ben, Carla, David, Eliza]

noncomputable def crow_count := count (λ b, b.species = Crow) birds

theorem determine_crow_count : crow_count = 4 :=
sorry

end determine_crow_count_l775_775841


namespace evaluate_expression_l775_775606

noncomputable def a : ℝ := real.cbrt 8
noncomputable def b : ℝ := (real.sqrt 5 - 1) ^ 0
noncomputable def c : ℝ := -real.sqrt (1 / 4)
noncomputable def d : ℝ := 2 ^ (-1)

theorem evaluate_expression : a + b + c + d = 3 :=
by
  sorry

end evaluate_expression_l775_775606


namespace number_of_integers_with_property_l775_775413

theorem number_of_integers_with_property :
  ∃ (n : ℕ), n = 226 ∧
    card {x | 100 ≤ x ∧ x ≤ 999 ∧
               ∃ (y : ℕ), 100 ≤ y ∧ y ≤ 999 ∧ y ≠ x ∧
                           y ∈ (perm (digits x)) ∧
                           11 ∣ int y} = 226 :=
sorry

end number_of_integers_with_property_l775_775413


namespace selina_sold_shirts_l775_775527

/-- Selina's selling problem -/
theorem selina_sold_shirts :
  let pants_price := 5
  let shorts_price := 3
  let shirts_price := 4
  let num_pants := 3
  let num_shorts := 5
  let remaining_money := 30 + (2 * 10)
  let money_from_pants := num_pants * pants_price
  let money_from_shorts := num_shorts * shorts_price
  let total_money_from_pants_and_shorts := money_from_pants + money_from_shorts
  let total_money_from_shirts := remaining_money - total_money_from_pants_and_shorts
  let num_shirts := total_money_from_shirts / shirts_price
  num_shirts = 5 := by
{
  sorry
}

end selina_sold_shirts_l775_775527


namespace price_per_box_l775_775981

theorem price_per_box (total_apples : ℕ) (apples_per_box : ℕ) (total_revenue : ℕ) : 
  total_apples = 10000 → apples_per_box = 50 → total_revenue = 7000 → 
  total_revenue / (total_apples / apples_per_box) = 35 :=
by
  intros h1 h2 h3
  -- we can skip the actual proof with sorry. This indicates that the proof is not provided,
  -- but the statement is what needs to be proven.
  sorry

end price_per_box_l775_775981


namespace parabola_and_chord_l775_775047

-- Define the conditions of the given problem
def vertex_at_origin (C : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C (0, 0)

def directrix (x : ℝ) := x = 2

noncomputable def parabola_eq (p : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, y^2 = -4 * p * x

def line_eq (x : ℝ) (y : ℝ) := y = x + 2

theorem parabola_and_chord
  (C : ℝ × ℝ → Prop) 
  (p : ℝ) 
  (hvertex : vertex_at_origin C)
  (hdirectrix : directrix p)
  (hparabola : ∀ x y, parabola_eq 4 (x, y))
  (hline : ∀ x y, line_eq x y) :
  (∀ x y, C (x, y) → parabola_eq 4 (x, y)) ∧ 
  ∃ A B : ℝ × ℝ, C A ∧ line_eq A.1 A.2 ∧ C B ∧ line_eq B.1 B.2 ∧ 
  (let D := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
   sqrt D = 4 * sqrt 6) :=
  sorry

end parabola_and_chord_l775_775047


namespace integral_abs_eq_five_half_l775_775676

theorem integral_abs_eq_five_half :
  ∫ x in -1..2, |x| = 5 / 2 := by
  sorry

end integral_abs_eq_five_half_l775_775676


namespace least_three_digit_divisible_by_2_5_7_3_l775_775992

theorem least_three_digit_divisible_by_2_5_7_3 : 
  ∃ n, n = 210 ∧ (100 ≤ n) ∧ 
           (n < 1000) ∧ 
           (n % 2 = 0) ∧ 
           (n % 5 = 0) ∧ 
           (n % 7 = 0) ∧ 
           (n % 3 = 0) :=
by
  use 210
  split
  rfl
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  split
  norm_num
  norm_num

end least_three_digit_divisible_by_2_5_7_3_l775_775992


namespace intersect_DN_SM_and_P_circumcenter_l775_775109

noncomputable def midpoint (A B: Point): Point := sorry
noncomputable def parallel (line1 line2: Line): Prop := sorry
noncomputable def centroid (A B C: Point): Point := sorry
noncomputable def intersection (line1 line2: Line): Point := sorry
noncomputable def circumcenter (tetra: Tetrahedron): Point := sorry

theorem intersect_DN_SM_and_P_circumcenter (S A B C M D N DN SM P: Point)
  (h1 : D = midpoint A B)
  (h2 : M = centroid A B C)
  (h3 : parallel DN SC)
  (h4 : DN ∩ SM = P)
  (h5 : circumcenter (Tetrahedron.mk S A B C) = P) :
  DN ∩ SM = P ∧ circumcenter (Tetrahedron.mk S A B C) = P := 
by sorry

end intersect_DN_SM_and_P_circumcenter_l775_775109


namespace integer_roots_of_polynomial_l775_775336

theorem integer_roots_of_polynomial :
  ∀ x : ℤ, x^3 - 4*x^2 - 11*x + 24 = 0 ↔ x = 2 ∨ x = -3 ∨ x = 4 := 
by 
  sorry

end integer_roots_of_polynomial_l775_775336


namespace seating_arrangements_l775_775099

theorem seating_arrangements :
  let total_arrangements := Nat.factorial 8
  let jwp_together := (Nat.factorial 6) * (Nat.factorial 3)
  total_arrangements - jwp_together = 36000 := by
  sorry

end seating_arrangements_l775_775099


namespace sequence_average_geq_neg_half_l775_775962

theorem sequence_average_geq_neg_half (n : ℕ) (a : ℕ → ℤ)
  (h1 : a 1 = 0)
  (h2 : ∀ k, 1 ≤ k ∧ k < n → |a (k + 1)| = |a k + 1|) :
  (a 1 + a 2 + a 3 + ⋯ + a n : ℤ) / ↑n ≥ -1 / 2 :=
by sorry

end sequence_average_geq_neg_half_l775_775962


namespace barrels_oil_total_l775_775646

theorem barrels_oil_total :
  let A := 3 / 4
  let B := A + 1 / 10
  A + B = 8 / 5 := by
  sorry

end barrels_oil_total_l775_775646


namespace min_questions_for_order_l775_775556
-- Importing the necessary library for general mathematics

-- The main theorem statement
theorem min_questions_for_order 
  (f : (Fin 50 → Fin 100) → (Fin 50 → Fin 100)) : 
  ∃ n : Nat, n = 5 ∧ ∀ p : (Fin 100 → Fin 100), p.orderable (min_query_count := n) :=
sorry

end min_questions_for_order_l775_775556


namespace paintings_after_30_days_l775_775503

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end paintings_after_30_days_l775_775503


namespace sum_of_squares_of_distances_l775_775194

theorem sum_of_squares_of_distances 
  (n : ℕ) (d r : ℝ) (X : Point) (center : Point) (inscribed_radius : ℝ) 
  (X_center_distance : dist X center = d) 
  (inscribed_circle_radius : inscribed_radius = r) :
  sum_of_squares_of_distances_to_sides X n = n * (r^2 + (1/2) * d^2) :=
sorry

end sum_of_squares_of_distances_l775_775194


namespace order_of_a_b_c_l775_775020

noncomputable def ln : ℝ → ℝ := Real.log
noncomputable def a : ℝ := ln 3 / 3
noncomputable def b : ℝ := ln 5 / 5
noncomputable def c : ℝ := ln 6 / 6

theorem order_of_a_b_c : a > b ∧ b > c := by
  sorry

end order_of_a_b_c_l775_775020


namespace runs_in_last_match_l775_775624

theorem runs_in_last_match (initial_average : ℝ) (wickets_last : ℕ) (average_decrease : ℝ) (wickets_before : ℝ) :
  initial_average = 12.4 →
  wickets_last = 4 →
  average_decrease = 0.4 →
  wickets_before = 55 →
  let final_average := initial_average - average_decrease in
  let wickets_after := wickets_before + wickets_last in
  let total_runs_before := wickets_before * initial_average in
  let total_runs_after := wickets_after * final_average in
  let runs_last := total_runs_after - total_runs_before in
  runs_last = 26 :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end runs_in_last_match_l775_775624


namespace irene_overtime_pay_per_hour_l775_775866

def irene_base_pay : ℝ := 500
def irene_base_hours : ℕ := 40
def irene_total_hours_last_week : ℕ := 50
def irene_total_income_last_week : ℝ := 700

theorem irene_overtime_pay_per_hour :
  (irene_total_income_last_week - irene_base_pay) / (irene_total_hours_last_week - irene_base_hours) = 20 := 
by
  sorry

end irene_overtime_pay_per_hour_l775_775866


namespace milan_billed_minutes_l775_775711

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l775_775711


namespace history_books_count_l775_775564

theorem history_books_count :
  ∀ (total_books reading_books math_books science_books history_books : ℕ),
  total_books = 10 →
  reading_books = 2 * total_books / 5 →
  math_books = 3 * total_books / 10 →
  science_books = math_books - 1 →
  history_books = total_books - (reading_books + math_books + science_books) →
  history_books = 1 :=
by {
  intros total_books reading_books math_books science_books history_books,
  intro h_total,
  intro h_reading,
  intro h_math,
  intro h_science,
  intro h_history,
  rw [h_total, h_reading, h_math, h_science, h_history],
  norm_num,
  sorry
}

end history_books_count_l775_775564


namespace common_ratio_solution_l775_775025

-- Define the problem condition
def geometric_sum_condition (a1 : ℝ) (q : ℝ) : Prop :=
  (a1 * (1 - q^3)) / (1 - q) = 3 * a1

-- Define the theorem we want to prove
theorem common_ratio_solution (a1 : ℝ) (q : ℝ) (h : geometric_sum_condition a1 q) :
  q = 1 ∨ q = -2 :=
sorry

end common_ratio_solution_l775_775025


namespace least_l_width_l775_775130

-- Definitions and Conditions
def Hexagon (H : Type) : Prop :=
  ∃ (A B C D E F : H),
    -- parallel opposite sides
    parallel (A, B) (D, E) ∧
    parallel (B, C) (E, F) ∧
    parallel (C, D) (F, A) ∧
    -- any 3 vertices can be covered with a strip of width 1
    (∀ (X Y Z : H), ∃ (l l' : H → ℝ), strip_width l l' X Y Z = 1)

def family_of_hexagons (F : Type → Prop) : Prop :=
  ∀ H, F H → Hexagon H

-- Statement to prove
theorem least_l_width (F : Type → Prop) :
  family_of_hexagons F →
  ∃ (ℓ : ℝ), (∀ H, F H → strip_cover_width H ℓ) ∧ (∀ k, k < ℓ → ¬ (∀ H, F H → strip_cover_width H k)) :=
  sorry

end least_l_width_l775_775130


namespace find_positive_integer_l775_775682

theorem find_positive_integer (n : ℕ) (h1 : n % 14 = 0) (h2 : 676 ≤ n ∧ n ≤ 702) : n = 700 :=
sorry

end find_positive_integer_l775_775682


namespace triangle_equilateral_l775_775864

theorem triangle_equilateral
  (A B C : ℝ)
  (h1 : 0 < A)
  (h2 : A < 180)
  (h3 : 0 < B)
  (h4 : B < 180)
  (h5 : 0 < C)
  (h6 : C < 180)
  (h7 : cos (A - B) * cos (B - C) * cos (C - A) = 1) :
  A = B ∧ B = C ∧ C = A :=
by
  sorry

end triangle_equilateral_l775_775864


namespace line_intersects_y_axis_line_intersects_y_axis_l775_775303

theorem line_intersects_y_axis : ∃ y, 5 * y + 3 * 0 = 15 ∧ (0, y) = (0, 3) :=
by
  exists 3
  simp
  trivial

# Alternatively if we write without actually proving all details, just "members"

theorem line_intersects_y_axis' : ∃ y, 5 * y + 3 * 0 = 15 ∧ (0, y) = (0, 3) :=
sorry

end line_intersects_y_axis_line_intersects_y_axis_l775_775303


namespace find_x_values_l775_775144

def f (x : ℝ) : ℝ := 3 * x^2 - 8

noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Placeholder for the inverse function

theorem find_x_values:
  ∃ x : ℝ, (f x = f_inv x) ↔ (x = (1 + Real.sqrt 97) / 6 ∨ x = (1 - Real.sqrt 97) / 6) := sorry

end find_x_values_l775_775144


namespace line_equation_l775_775548

theorem line_equation (x y : ℝ) : 
  (∃ l : ℝ → ℝ, (l 0 = 1 ∧ ∀ θ, θ = 45 → tan θ = 1 ∧ ∀ x, l x = x + 1)) → (x - y + 1 = 0) :=
sorry

end line_equation_l775_775548


namespace bad_segments_even_l775_775371

/-- Given a closed polygonal chain where each segment touches a circle, the number of bad segments,
where a segment is bad if its continuation touches the circle, is even. -/
theorem bad_segments_even
  (n : ℕ)
  (closed_polygon : Fin (n + 1) → ℝ × ℝ)
  (circle : (ℝ × ℝ) × ℝ)
  (touches : ∀ i : Fin n, ∃ p : ℝ × ℝ, line_through (closed_polygon i) (closed_polygon (i + 1)) ∩ (circle.1, circle.2) = {p}) :
  ∃ m : ℕ, 2 * m = n :=
sorry

end bad_segments_even_l775_775371


namespace selina_sells_5_shirts_l775_775525

theorem selina_sells_5_shirts
    (pants_price shorts_price shirts_price : ℕ)
    (pants_sold shorts_sold shirts_bought remaining_money : ℕ)
    (total_earnings : ℕ) :
  pants_price = 5 →
  shorts_price = 3 →
  shirts_price = 4 →
  pants_sold = 3 →
  shorts_sold = 5 →
  shirts_bought = 2 →
  remaining_money = 30 →
  total_earnings = remaining_money + shirts_bought * 10 →
  total_earnings = 50 →
  total_earnings = pants_sold * pants_price + shorts_sold * shorts_price + 20 →
  20 / shirts_price = 5 :=
by
  sorry

end selina_sells_5_shirts_l775_775525


namespace nonzero_integral_for_n_3_4_7_8_zero_integral_otherwise_l775_775488

def f_n (n : ℕ) (x : ℝ) : ℝ := (List.range n).foldr (λ i acc, Real.cos ((i + 1 : ℕ) * x) * acc) 1

def fn_nonzero_integral (n : ℕ) : Prop :=
  ∫ x in 0..2*Real.pi, f_n n x ≠ 0

theorem nonzero_integral_for_n_3_4_7_8 :
  ∀ (n : ℕ), n ∈ [3, 4, 7, 8] → fn_nonzero_integral n :=
by
  assume n h
  sorry

theorem zero_integral_otherwise :
  ∀ (n : ℕ), n ∈ [1, 2, 5, 6, 9, 10] → ¬fn_nonzero_integral n :=
by
  assume n h
  sorry

end nonzero_integral_for_n_3_4_7_8_zero_integral_otherwise_l775_775488


namespace expand_polynomial_product_l775_775678

theorem expand_polynomial_product :
  (λ x : ℝ, (x^2 - 3*x + 3) * (x^2 + 3*x + 3)) = (λ x : ℝ, x^4 - 3*x^2 + 9) :=
by {
  funext x,
  -- The detailed proof steps would go here
  sorry
}

end expand_polynomial_product_l775_775678


namespace find_XY_reflection_XM_l775_775859

noncomputable def triangle_reflection_XM (XN NZ YE : ℝ) (X Y Z M : Point) : ℝ :=
let XY' := reflect X M Y,
    Z'  := reflect X M Z,
    E   := midpoint Y Z,
    N   := reflect X M E, 
    -- Define lengths given
    XN := 8,
    NZ := 16,
    YE := 12,
    -- Compute relevant lengths following the problem's steps.
    XE := XN,
    Z'E := NZ,
    EM := (XE * YE) / Z'E in
-- Law of Cosines application for XY
 sqrt (2 * (XE^2) * (YE^2)) - 2 * (XE) * (YE) * (cos_angle X E Y) = 2 * sqrt 61

theorem find_XY_reflection_XM : (XY' X Y) = 2 * sqrt 61 := sorry

end find_XY_reflection_XM_l775_775859


namespace notebook_cost_l775_775836

theorem notebook_cost (s n c : ℕ) (h1 : s > 17) (h2 : n > 2 ∧ n % 2 = 0) (h3 : c > n) (h4 : s * c * n = 2013) : c = 61 :=
sorry

end notebook_cost_l775_775836


namespace formula_for_an_sum_of_series_l775_775753

-- Defining sequences a_n and b_n according to the problem's conditions
def a (n : ℕ) : ℕ := 2 ^ n

def S (n : ℕ) : ℕ := (list.range (n + 1)).map (λ k, a k).sum

def b (n : ℕ) : ℤ := -2 * n

-- Given conditions in Lean
axiom condition1 : ∀ n : ℕ, n ≠ 0 → (2 * a n = 2 + S n)

axiom condition2 : a 1 = 2

axiom positive_terms : ∀ n : ℕ, n ≠ 0 → a n > 0

axiom condition3 : ∀ n : ℕ, a n ^ 2 = (1 / 2) ^ (b n)

-- Proof statements
theorem formula_for_an (n : ℕ) (h : n ≠ 0) : a n = 2 ^ n :=
by sorry

theorem sum_of_series (n : ℕ) : ∑ k in finset.range n, (1 / (b k * b (k + 1))) = n / (4 * (n + 1)) :=
by sorry

end formula_for_an_sum_of_series_l775_775753


namespace sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l775_775509

open Real

theorem sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x * y * z ≥ x * y + y * z + z * x) :
  sqrt (x * y * z) ≥ sqrt x + sqrt y + sqrt z :=
by
  sorry

end sqrt_xyz_ge_sqrt_x_add_sqrt_y_add_sqrt_z_l775_775509


namespace milan_billed_minutes_l775_775719

-- Define the conditions
def monthly_fee : ℝ := 2
def cost_per_minute : ℝ := 0.12
def total_bill : ℝ := 23.36

-- Define the number of minutes based on the above conditions
def minutes := (total_bill - monthly_fee) / cost_per_minute

-- Prove that the number of minutes is 178
theorem milan_billed_minutes : minutes = 178 := by
  -- Proof steps would go here, but as instructed, we use 'sorry' to skip the proof.
  sorry

end milan_billed_minutes_l775_775719


namespace a_general_formula_T_remainder_div3_l775_775138

-- Definitions based on conditions
def a : ℕ → ℕ 
| 1 := 1
| n := n

def S (n : ℕ) : ℕ := (n * (n + 1)) / 2

def b (n : ℕ) : ℕ := 2 ^ (a n)

def T (n : ℕ) : ℕ := ∑ i in range n, b i

-- Theorem statements to prove
theorem a_general_formula (n : ℕ) : a n = n := 
sorry

theorem T_remainder_div3 (n : ℕ) : (T (2 * n - 1)) % 3 = 2 := 
sorry

end a_general_formula_T_remainder_div3_l775_775138


namespace sum_f_1_to_100_l775_775487

-- Define the function f(n) as described
def f (n : ℕ) : ℕ :=
  let digits := n.digits 10
  let evenDigits := digits.filter (fun d => d % 2 = 0)
  if evenDigits.length = 0 then 0 else evenDigits.foldl (fun acc d => acc * d) 1

-- Prove the theorem that the sum of f(n) for n from 1 to 100 is 1308
theorem sum_f_1_to_100 : (Finset.range 101).sum f = 1308 :=
begin
  sorry
end

end sum_f_1_to_100_l775_775487


namespace intersection_of_M_and_N_l775_775059

def M := {x : ℝ | 3 * x - x^2 > 0}
def N := {x : ℝ | x^2 - 4 * x + 3 > 0}
def I := {x : ℝ | 0 < x ∧ x < 1}

theorem intersection_of_M_and_N : M ∩ N = I :=
by
  sorry

end intersection_of_M_and_N_l775_775059


namespace complex_conjugate_l775_775380

noncomputable theory

open Complex

theorem complex_conjugate (x y : ℝ) (i : ℂ) (h : x / (1 + i) = 1 - y * i) : 
  conj (x + y * i) = 2 - i :=
by {
  -- The statement is constructed assuming x, y are reals, i is the imaginary unit, and h is the condition.
  sorry
}

end complex_conjugate_l775_775380


namespace k_values_exist_l775_775339

open Real

noncomputable def matrixA : Matrix (Fin 2) (Fin 2) ℝ := ![![3, 4], ![6, 2]]

theorem k_values_exist (k : ℝ) :
  ∃ u : Vector (Fin 2) ℝ, u ≠ ![0, 0] ∧ matrixA.mulVec u = k • u ↔
  k = (5 + Real.sqrt 97) / 2 ∨ k = (5 - Real.sqrt 97) / 2 :=
by
  sorry

end k_values_exist_l775_775339


namespace cardinality_of_union_l775_775035

def A : Set ℤ := {-1, 1, 2}
def B : Set ℤ := {0, 1, 2, 7}
def AuB : Set ℤ := A ∪ B

theorem cardinality_of_union :
  (Finset.card (Set.toFinset AuB)) = 5 := by
  sorry

end cardinality_of_union_l775_775035


namespace min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l775_775245

-- Statements for minimum questions required for different number of cards 

theorem min_questions_30_cards (cards : Fin 30 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 10 :=
by
  sorry

theorem min_questions_31_cards (cards : Fin 31 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 11 :=
by
  sorry

theorem min_questions_32_cards (cards : Fin 32 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 12 :=
by
  sorry

theorem min_questions_50_cards_circle (cards : Fin 50 → Int) (h : ∀ i, cards i = 1 ∨ cards i = -1) :
  ∃ n, n = 50 :=
by
  sorry

end min_questions_30_cards_min_questions_31_cards_min_questions_32_cards_min_questions_50_cards_circle_l775_775245


namespace find_k_l775_775776

-- Define the sequence and its sum
def Sn (k : ℝ) (n : ℕ) : ℝ := k + 3^n
def an (k : ℝ) (n : ℕ) : ℝ := Sn k n - (if n = 0 then 0 else Sn k (n - 1))

-- Define the condition that a sequence is geometric
def is_geometric (a : ℕ → ℝ) (r : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = r * a n

theorem find_k (k : ℝ) :
  is_geometric (an k) (an k 1 / an k 0) → k = -1 := 
by sorry

end find_k_l775_775776


namespace longest_tape_length_l775_775251

/-!
  Problem: Find the length of the longest tape that can exactly measure the lengths 
  24 m, 36 m, and 54 m in cm.
  
  Solution: Convert the given lengths to the same unit (cm), then find their GCD.
  
  Given: Lengths are 2400 cm, 3600 cm, and 5400 cm.
  To Prove: gcd(2400, 3600, 5400) = 300.
-/

theorem longest_tape_length (a b c : ℕ) : a = 2400 → b = 3600 → c = 5400 → Nat.gcd (Nat.gcd a b) c = 300 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  -- omitted proof steps
  sorry

end longest_tape_length_l775_775251


namespace find_a_range_m_l775_775478

-- Define the function f
def f (x : ℝ) (a : ℝ) := a * x - 1 + Real.exp x

-- Statement for Part 1
theorem find_a (a : ℝ) :
  (∃ x₀ : ℝ, f x₀ a = 2 * x₀ ∧ (f' : ℝ → ℝ) x₀ = 2) → a = 1 :=
sorry

-- Define the function f for Part 2
def f_2 (x : ℝ) := x - 1 + Real.exp x

-- Statement for Part 2
theorem range_m (m : ℝ) :
  (∀ x ∈ Set.Ico 0 (Real.pi / 2), f_2 x ≥ m * Real.sin (2 * x)) → m ≤ 1 :=
sorry

end find_a_range_m_l775_775478


namespace find_k_value_l775_775108

-- Define the given conditions
structure RightTriangle (A B O : ℝ × ℝ) :=
(O_right : O.1 = 0 ∧ O.2 = 0)
(A_pos : A = (0, 3))
(OP_dist : dist O (4 * real.sqrt 2, 4 * real.sqrt 2) = 4 * real.sqrt 2)

-- Define the function y = k / x and the points mapped to E and F
def hyperbolic_function (k x : ℝ) : ℝ :=
  k / x

noncomputable def k_value (D : ℝ × ℝ) : ℝ := sorry  -- Placeholder for the computed k-value

theorem find_k_value :
  ∃ k, (RightTriangle (0, 3) (3, 0) (0, 0)) → (D = (5, 8)) →
  (hyperbolic_function k D.1 = D.2) ∧ (k = 24) :=
begin
  sorry  -- Separate proof steps are needed
end

end find_k_value_l775_775108


namespace exists_large_value_l775_775022

noncomputable def polynomial {n : ℕ} (C : Fin (n + 1) → ℂ) : ℂ → ℂ :=
  fun z => ∑ i in Finset.range (n + 1), C ⟨i, Nat.lt_succ_self n⟩ * z^i

theorem exists_large_value {n : ℕ} (C : Fin (n + 1) → ℂ) :
  ∃ z_0 : ℂ, Complex.abs z_0 ≤ 1 ∧ Complex.abs (polynomial C z_0) ≥ Complex.abs (C 0) + Complex.abs (C n) :=
sorry

end exists_large_value_l775_775022


namespace general_term_a_max_partial_sum_b_l775_775755

open Real

noncomputable def f (x : ℝ) : ℝ := -1/2 * x + 1/2

def seq_a (n : ℕ) : ℝ := (1/3) ^ n

def partial_sum_a (n : ℕ) : ℝ := (1 / 2) * (1 - (1 / 3) ^ n)

def seq_b (n : ℕ) : ℝ := 3 / 2 * log 3 (1 - 2 * partial_sum_a n) + 10

def partial_sum_b (n : ℕ) : ℝ := (n : ℝ) / 2 * (17 / 2 + 1)

theorem general_term_a (n : ℕ) : seq_a n = (1 / 3) ^ n :=
by
  sorry

theorem max_partial_sum_b : partial_sum_b 6 = 57 / 2 :=
by
  sorry

end general_term_a_max_partial_sum_b_l775_775755


namespace average_price_comparison_l775_775933

-- Define the conditions
variables (a b : ℝ) (ha : a > 0) (hb : b > 0) (hne : a ≠ b)

-- Define the average unit prices
def m1 : ℝ := 2 * a * b / (a + b)
def m2 : ℝ := (a + b) / 2

-- State the theorem
theorem average_price_comparison : m1 a b < m2 a b :=
sorry

end average_price_comparison_l775_775933


namespace number_of_sheep_l775_775253

theorem number_of_sheep (S H : ℕ) 
  (h1 : S / H = 5 / 7)
  (h2 : H * 230 = 12880) : 
  S = 40 :=
by
  sorry

end number_of_sheep_l775_775253


namespace sum_of_ages_after_10_years_l775_775129

theorem sum_of_ages_after_10_years (Kareem_age son's_age : ℕ) 
  (h1 : Kareem_age = 42) 
  (h2 : son's_age = 14) : 
  Kareem_age + 10 + (son's_age + 10) = 76 :=
by 
  rw [h1, h2]
  simp
  done

#print sum_of_ages_after_10_years

end sum_of_ages_after_10_years_l775_775129


namespace intersecting_parabolas_l775_775310

theorem intersecting_parabolas :
  let k_vals : Finset Int := {1, 2}
  let a_vals : Finset Int := {-3, -2, -1, 0, 1, 2, 3}
  let b_vals : Finset Int := {-4, -3, -2, -1, 1, 2, 3, 4}
  ∀ (parabolas : Finset (ℝ × ℝ → ℝ)) 
    (condition : ∀ p ∈ parabolas, ∃ k a b, k ∈ k_vals ∧ a ∈ a_vals ∧ b ∈ b_vals ∧ 
      (∀ x y, (focusless_point : (x, y), y = k * (a * x + b))),
    Finset.count (intersections parabolas) = 2366 :=
by
  sorry

end intersecting_parabolas_l775_775310


namespace proof_l775_775468

noncomputable def triangle (α β γ : Type) := {a b c : Type}

variables {α β γ δ ε ζ η θ : Type}

def circle (α β : Type) := {center : α, radius : β}

variables {A B C : triangle α β γ}
variables {K : circle δ ε}

def tangent (l : ζ) (C : circle δ ε) : Prop := sorry

def intersects (l m : ζ) (p : θ) : Prop := sorry

def circumcircle (a b c : triangle α β γ) : circle δ ε := sorry

def collinear (p q r : θ) : Prop := sorry

def parallel (l m : ζ) : Prop := sorry

axiom condition1 : tangent (AC : ζ) (K)
axiom condition2 : tangent (AB : ζ) (K)
axiom condition3 : intersects (BC : ζ) (K) (M : θ)
axiom condition4 : intersects (BC : ζ) (K) (N : θ)
axiom condition5 : intersects (FM : ζ) (EN : ζ) (I : θ)
axiom condition6 : intersects (circumcircle (I F N)) (circumcircle (I E M)) (J : θ)
axiom distinct_IJ : I ≠ J

theorem proof (A B C : triangle α β γ) (K : circle δ ε)
  (E F M N I J : θ)
  (h1 : tangent (AC : ζ) K)
  (h2 : tangent (AB : ζ) K)
  (h3 : intersects (BC : ζ) K M)
  (h4 : intersects (BC : ζ) K N)
  (h5 : intersects (FM : ζ) (EN : ζ) I)
  (h6 : intersects (circumcircle (I F N)) (circumcircle (I E M)) J)
  (h7 : I ≠ J) :
  collinear (I J A) ∧ perpendicular (KJ : ζ) (IJ : ζ) := sorry

end proof_l775_775468


namespace truck_travel_distance_l775_775290

theorem truck_travel_distance (b t : ℝ) (h1 : t > 0) :
  (300 * (b / 4) / t) / 3 = (25 * b) / t :=
by
  sorry

end truck_travel_distance_l775_775290


namespace simplify_expression_l775_775182

open Nat

theorem simplify_expression (x : ℤ) : 2 - (3 - (2 - (5 - (3 - x)))) = -1 - x :=
by
  sorry

end simplify_expression_l775_775182


namespace remainder_theorem_example_l775_775347

theorem remainder_theorem_example (x : ℝ) : 
  let Q := fun x => 8*x^4 - 18*x^3 - 6*x^2 + 4*x - 30 
  in Q x = 8*x^4 - 18*x^3 - 6*x^2 + 4*x - 30 ∧ Q 4 = 786 :=
by
  sorry

end remainder_theorem_example_l775_775347


namespace ways_to_sum_1800_l775_775416

theorem ways_to_sum_1800 : 
  (∃ n1 n2 n3 n4 : ℕ, 
    n1 * 3 * 2 + n2 * 2 * 3 + n3 * 6 + n4 * 6 = 1800 ∧ 
    n1 + n2 + n3 + n4 = 300) → 
  ∑ i in finset.range 304, (nat.choose 303 3) = 1515051 := 
sorry

end ways_to_sum_1800_l775_775416


namespace positive_difference_of_perimeters_l775_775801

-- Definitions of the problem conditions and perimeters
def first_rectangle_height : ℕ := 2
def first_rectangle_width : ℕ := 5

def second_rectangle_height : ℕ := 3
def second_rectangle_width : ℕ := 7

def perimeter (height width : ℕ) : ℕ := 2 * (height + width)

-- Main Lean statement to prove the equivalent proof problem
theorem positive_difference_of_perimeters : 
  |perimeter second_rectangle_height second_rectangle_width - perimeter first_rectangle_height first_rectangle_width| = 6 :=
by
  sorry

end positive_difference_of_perimeters_l775_775801


namespace general_solution_nonhomogeneous_system_l775_775691

theorem general_solution_nonhomogeneous_system :
  ∃ (C1 C2 : ℝ), 
  (∀ t : ℝ,
  let x := 2 * C1 * exp(2 * t) + C2 * exp(3 * t) - (3 / 2) * exp(t) + 2 * t * exp(2 * t) in
  let y := -C1 * exp(2 * t) - C2 * exp(3 * t) + (1 / 2) * exp(t) - (t + 1) * exp(2 * t) in
  (differentiation x t = x - 2 * y + exp(t)) ∧ 
  (differentiation y t = x + 4 * y + exp(2 * t))) :=
sorry

end general_solution_nonhomogeneous_system_l775_775691


namespace num_4digit_greater_than_1000_using_digits_2012_l775_775412

theorem num_4digit_greater_than_1000_using_digits_2012 : 
  let digits := [2, 0, 1, 2] in
  let count_gt_1000 := Nat.card {n : Nat // (1000 ≤ n ∧ 
                    (∀ c ∈ (to_string n).to_list, c ∈ digits) ∧ 
                    (multiset.of_list (to_string n).to_list) = multiset.of_list digits)} in
  count_gt_1000 = 9 :=
  by
  -- Sorry is a placeholder for the proof
  sorry

end num_4digit_greater_than_1000_using_digits_2012_l775_775412


namespace sum_of_table_correct_l775_775356

noncomputable def sum_of_table (n : ℕ) (hn : 1 < n) : ℕ :=
  let S := Finset.range (n + 1)
  let subsets_of_S := Finset.powerset S
  let card_subsets := 2^n
  let A_i := λ(i : ℕ), (subsets_of_S.to_list.nth i).getOrElse ∅
  let symm_diff := λ (A B : Finset ℕ), (A \ B) ∪ (B \ A)
  let sum_elements := λ (set : Finset ℕ), set.sum id
  let M := λ (i j : ℕ), sum_elements (symm_diff (A_i i) (A_i j))
  let total_sum := Finset.sum (Finset.range card_subsets) (λ i, 
                           Finset.sum (Finset.range card_subsets) (λ j, M i j))
  in 2^(2*n-2) * n * (n + 1)

theorem sum_of_table_correct
  (n : ℕ) (hn : 1 < n) : sum_of_table n hn = 2^(2*n-2) * n * (n + 1) :=
sorry

end sum_of_table_correct_l775_775356


namespace both_selected_probability_l775_775570

-- Define the probabilities of selection for X and Y
def P_X := 1 / 7
def P_Y := 2 / 9

-- Statement to prove that the probability of both being selected is 2 / 63
theorem both_selected_probability :
  (P_X * P_Y) = (2 / 63) :=
by
  -- Proof skipped
  sorry

end both_selected_probability_l775_775570


namespace initial_distance_proof_l775_775984

-- Define the conditions
variables (speedX speedY : ℝ)
variables (time : ℝ)
variables (extraDistance : ℝ)

-- Define the initial distance
def initialDistance (speedX speedY time extraDistance : ℝ) := (speedY - speedX) * time - extraDistance

-- Given conditions
axiom speedX_36 : speedX = 36 -- Speed of vehicle X, 36 mph
axiom speedY_45 : speedY = 45 -- Speed of vehicle Y, 45 mph
axiom time_5 : time = 5 -- Time taken for vehicle Y to overtake, 5 hours
axiom extraDistance_23 : extraDistance = 23 -- Extra distance vehicle Y is ahead, 23 miles

-- Prove that the initial distance is 22 miles
theorem initial_distance_proof : initialDistance speedX speedY time extraDistance = 22 :=
by
  rw [speedX_36, speedY_45, time_5, extraDistance_23]
  sorry

end initial_distance_proof_l775_775984


namespace min_length_segment_ab_l775_775927
noncomputable def min_distance_segment_ab : ℝ :=
  let A (a : ℝ) : ℝ × ℝ := (a, (12 / 5) * a - 3)
  let B (b : ℝ) : ℝ × ℝ := (b, b^2)
  let distance (a b : ℝ) : ℝ :=
    real.sqrt ((b - a)^2 + (b^2 - ((12 / 5) * a - 3))^2)
  sInf (set.image (λ (ab : ℝ × ℝ), distance ab.1 ab.2) (set.univ.prod set.univ))
  
theorem min_length_segment_ab : min_distance_segment_ab = 3 / 5 := 
sorry

end min_length_segment_ab_l775_775927


namespace number_of_bottles_l775_775214

-- Define the weights and total weight based on given conditions
def weight_of_two_bags_chips : ℕ := 800
def total_weight_five_bags_and_juices : ℕ := 2200
def weight_difference_chip_Juice : ℕ := 350

-- Considering 1 bag of chips weighs 400 g (derived from the condition)
def weight_of_one_bag_chips : ℕ := 400
def weight_of_one_bottle_juice : ℕ := weight_of_one_bag_chips - weight_difference_chip_Juice

-- Define the proof of the question
theorem number_of_bottles :
  (total_weight_five_bags_and_juices - (5 * weight_of_one_bag_chips)) / weight_of_one_bottle_juice = 4 := by sorry

end number_of_bottles_l775_775214


namespace base_radius_of_cone_l775_775082

theorem base_radius_of_cone (r : ℝ) (h : r > 0) :
  (r = 2) → (2 * r * Real.pi / (2 * Real.pi) = 1) :=
by
  intro hr
  rw hr
  norm_num
  use 1


end base_radius_of_cone_l775_775082


namespace haleigh_candle_problem_l775_775804

theorem haleigh_candle_problem 
  (wax_left : ℕ → ℕ → ℕ)
  (recoverable_wax : ℕ → ℕ)
  (total_wax : ℕ)
  (candles : ℕ → ℕ)
  (candles_20oz : ℕ := 5)
  (candles_5oz : ℕ := 5)
  (candles_1oz : ℕ := 25)
  (total_wax_recovered : ℕ) 
  (candles_made : ℕ) 
  (each_canldle_10_percent_wax : ∀ (oz : ℕ), wax_left oz 10 = oz / 10)
  (calculate_recoverable_wax : recoverable_wax candles_20oz + recoverable_wax candles_5oz + recoverable_wax candles_1oz = total_wax) 
  (calculate_total_wax_recovered : total_wax = total_wax_recovered)
  (calculate_candles_made : total_wax_recovered / 5 = candles_made) :
  candles_made = 3 := 
by:
  sorry

end haleigh_candle_problem_l775_775804


namespace system_solution_l775_775187

theorem system_solution (x : Fin 1995 → ℤ) :
  (∀ i : (Fin 1995),
    x (i + 1) ^ 2 = 1 + x ((i + 1993) % 1995) * x ((i + 1994) % 1995)) →
  (∀ n : (Fin 1995),
    (x n = 0 ∧ x (n + 1) = 1 ∧ x (n + 2) = -1) ∨
    (x n = 0 ∧ x (n + 1) = -1 ∧ x (n + 2) = 1)) :=
by sorry

end system_solution_l775_775187


namespace find_base_s_l775_775860

theorem find_base_s :
  ∃ s : ℕ, (5 * s^2 + 2 * s + 3) + (4 * s^2 + 5 * s + 3) = s^3 + s^2 ∧ s = 7 :=
begin
  use 7,
  split,
  { sorry },  -- Proof of the equation
  { refl }    -- Proof that s = 7
end

end find_base_s_l775_775860


namespace rectangular_prism_sides_multiples_of_5_l775_775208

noncomputable def rectangular_prism_sides_multiples_product_condition 
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) (prod_eq_450 : l * w = 450) : Prop :=
  l ∣ 450 ∧ w ∣ 450

theorem rectangular_prism_sides_multiples_of_5
  (l w : ℕ) (hl : l % 5 = 0) (hw : w % 5 = 0) :
  rectangular_prism_sides_multiples_product_condition l w hl hw (by sorry) :=
sorry

end rectangular_prism_sides_multiples_of_5_l775_775208


namespace relationship_among_abc_l775_775382

def a : ℝ := real.sqrt 0.5
def b : ℝ := real.sqrt 0.9
def c : ℝ := real.log 5 0.3

theorem relationship_among_abc : b > a ∧ a > c := by
  have h1 : 0 < a := real.sqrt_pos.mpr (by norm_num)
  have h2 : a < b := real.sqrt_lt (by norm_num) (by norm_num)
  have h3 : b < 1 := real.sqrt_lt_one (by norm_num)
  have h4 : c < 0 := real.log_lt_zero (by norm_num)
  exact ⟨h2, h4⟩

end relationship_among_abc_l775_775382


namespace employees_count_l775_775540

-- Let E be the number of employees excluding the manager
def E (employees : ℕ) : ℕ := employees

-- Let T be the total salary of employees excluding the manager
def T (employees : ℕ) : ℕ := employees * 1500

-- Conditions given in the problem
def average_salary (employees : ℕ) : ℕ := T employees / E employees
def new_average_salary (employees : ℕ) : ℕ := (T employees + 22500) / (E employees + 1)

theorem employees_count : (average_salary employees = 1500) ∧ (new_average_salary employees = 2500) ∧ (manager_salary = 22500) → (E employees = 20) :=
  by sorry

end employees_count_l775_775540


namespace find_x_l775_775064

noncomputable def a (x : ℝ) : ℝ × ℝ :=
  (Real.cos (3 * x / 2), Real.sin (3 * x / 2))

noncomputable def b (x : ℝ) : ℝ × ℝ :=
  (Real.cos (x / 2), -Real.sin (x / 2))

noncomputable def norm_sq (v : ℝ × ℝ) : ℝ :=
  v.1 * v.1 + v.2 * v.2

theorem find_x (x : ℝ) :
  (0 ≤ x ∧ x ≤ Real.pi)
  ∧ (norm_sq (a x) + norm_sq (b x) + 2 * ((a x).1 * (b x).1 + (a x).2 * (b x).2) = 1)
  → (x = Real.pi / 3 ∨ x = 2 * Real.pi / 3) :=
by
  intro h
  sorry

end find_x_l775_775064


namespace eunji_rank_l775_775158

open Nat

theorem eunji_rank (minyoung_rank : ℕ) (places_after : ℕ) 
  (h1 : minyoung_rank = 33) (h2 : places_after = 11) : 
  (minyoung_rank + places_after) = 44 :=
by
  rw [h1, h2]
  exact rfl

end eunji_rank_l775_775158


namespace sum_two_digit_squares_ends_with_36_l775_775589

theorem sum_two_digit_squares_ends_with_36 : 
  (∑ n in Finset.filter (λ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n^2 % 100 = 36) (Finset.range 100), n) = 194 :=
by
  sorry

end sum_two_digit_squares_ends_with_36_l775_775589


namespace proportion_option_B_true_l775_775074

theorem proportion_option_B_true {a b c d : ℚ} (h : a / b = c / d) : 
  (a + c) / c = (b + d) / d := 
by 
  sorry

end proportion_option_B_true_l775_775074


namespace tan_sum_20_40_l775_775213

theorem tan_sum_20_40 :
  tan 20 * Real.pi / 180 + tan 40 * Real.pi / 180 + Real.sqrt 3 * (tan 20 * Real.pi / 180 * tan 40 * Real.pi / 180) = Real.sqrt 3 :=
by
  sorry

end tan_sum_20_40_l775_775213


namespace johns_average_speed_l775_775464

-- Definitions
def distance_to_library : ℝ := 2 -- km
def time_to_library : ℕ := 40 -- minutes
def reading_time : ℕ := 20 -- minutes
def distance_back_home : ℝ := 2 -- km
def time_back_home : ℕ := 20 -- minutes

-- Calculate total distance and total time
def total_distance : ℝ := distance_to_library + distance_back_home
def total_time_minutes : ℕ := time_to_library + reading_time + time_back_home
def total_time_hours : ℝ := total_time_minutes / 60.0 -- convert minutes to hours

-- Theorem stating John's average speed is 3 km/hr
theorem johns_average_speed : total_distance / total_time_hours = 3 := by
  sorry

end johns_average_speed_l775_775464


namespace diagonal_length_of_regular_hexagon_l775_775001

theorem diagonal_length_of_regular_hexagon (s : ℝ) (h : s = 10) :
  let D := λ A : ℝ, 2 * s * Real.sqrt 3 in
  D s = 10 * Real.sqrt 3 :=
by
  sorry

end diagonal_length_of_regular_hexagon_l775_775001


namespace complex_quadrant_l775_775273

noncomputable def i : ℂ := complex.I

theorem complex_quadrant (z : ℂ) (h : (i^3) * z = 2 + i) : z.re < 0 ∧ z.im > 0 :=
by
  -- Start the proof (to be filled in)
  sorry -- Placeholder for the proof

end complex_quadrant_l775_775273


namespace complex_equality_l775_775363

theorem complex_equality (a b : ℝ) (h : (⟨0, 1⟩ : ℂ) ^ 3 = ⟨a, -b⟩) : a + b = 1 :=
by
  sorry

end complex_equality_l775_775363


namespace circumference_of_circle_x_l775_775662

theorem circumference_of_circle_x 
  (r_y : ℝ) (h : r_y / 2 = 4.5) (A_x : ℝ) (A_y : ℝ) (r_x : ℝ)
  (h_s : A_x = A_y) (h_area_y : A_y = π * r_y^2) (h_area_x : A_x = π * r_x^2) :
  2 * π * r_x = 18 * π :=
by
  -- Assume conditions (these will be transformed into the assumptions 'h')
  have r_y_eq : r_y = 9, from
    calc r_y = 2 * 4.5 : by linarith
          ... = 9     : by norm_num,
  -- Given two circles have the same area A_x = A_y = 81π
  have h_area_y_eq : 81 * π = π * r_y^2, from
    calc 81 * π = π * (9 ^ 2) : by norm_num
           ... = π * r_y^2    : by rw [r_y_eq],
  -- Equate A_x and A_y
  have A_x_eq : A_x = 81 * π, from
    calc A_x = A_y : by rw [h_s]
          ... = 81 * π : by exact h_area_y_eq,
  -- solve for r_x
  have r_x_eq : r_x = 9, from
    calc r_x = sqrt (81) : by rw [←h_area_x, A_x_eq]; linarith
        ... = 9         : by norm_num,
  -- the actual goal
  rw [r_x_eq],
  norm_num

end circumference_of_circle_x_l775_775662


namespace intersection_unique_point_l775_775415

theorem intersection_unique_point :
  ∃! p : ℝ × ℝ, let (x, y) := p in y = |3 * x + 4| ∧ y = 5 - |2 * x - 1| ∧ (x, y) = (0, 4) :=
sorry

end intersection_unique_point_l775_775415


namespace cos_5theta_l775_775819

theorem cos_5theta (theta : ℝ) (h : Real.cos theta = 2 / 5) : Real.cos (5 * theta) = 2762 / 3125 := 
sorry

end cos_5theta_l775_775819


namespace an_general_formula_Tn_formula_l775_775058

open Nat
open BigOperators

-- Given conditions 
def Sn (n : ℕ) : ℕ := (n * n + n) / 2
def an (n : ℕ) : ℕ := if n = 1 then 1 else (Sn n - Sn (n - 1))
def bn (n : ℕ) : ℕ := an n * 2 ^ an (2 * n)

-- Lean statement for part 1
theorem an_general_formula (n : ℕ) : an n = n :=
by sorry

-- Lean statement for part 2
theorem Tn_formula (n : ℕ) : 
  (∑ k in Finset.range n, bn (k + 1)) = ((n / 3) - (1 / 9)) * 4^(n + 1) + (4 / 9) :=
by sorry

end an_general_formula_Tn_formula_l775_775058


namespace probability_sequence_l775_775976

theorem probability_sequence:
  let total_cards := 52
  let num_aces := 4
  let num_tens := 4
  let num_queens := 4
  let first_card_ace_probability := (num_aces : ℚ) / total_cards
  let second_card_ten_probability := (num_tens : ℚ) / (total_cards - 1)
  let third_card_queen_probability := (num_queens : ℚ) / (total_cards - 2)
  in first_card_ace_probability * second_card_ten_probability * third_card_queen_probability = 8 / 16575 :=
sorry

end probability_sequence_l775_775976


namespace correct_propositions_l775_775390

-- Define the propositions
def proposition_1 := ∀ (cylinder : Type) (upper_base lower_base : Set Point) (p1 : Point) (p2 : Point),
  p1 ∈ upper_base → p2 ∈ lower_base → line_segment p1 p2 ∈ generatrix cylinder

def proposition_2 := ∀ (cone : Type) (vertex : Point) (base : Set Point) (p : Point),
  p ∈ base → line_segment vertex p ∈ generatrix cone

def proposition_3 := ∀ (frustum : Type) (upper_base lower_base : Set Point) (p1 : Point) (p2 : Point),
  p1 ∈ upper_base → p2 ∈ lower_base → line_segment p1 p2 ∈ generatrix frustum

def proposition_4 := ∀ (cylinder : Type) (g1 g2 : generatrix cylinder),
  parallel (line_containing g1) (line_containing g2)

-- Define the theorem to be proved (D is correct)
theorem correct_propositions : proposition_2 ∧ proposition_4 :=
by
  sorry

end correct_propositions_l775_775390


namespace committee_count_is_correct_l775_775095

-- Definitions of the problem conditions
def total_people : ℕ := 10
def committee_size : ℕ := 5
def remaining_people := total_people - 1
def members_to_choose := committee_size - 1

-- The combinatorial function for selecting committee members
def binomial (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def number_of_ways_to_form_committee : ℕ :=
  binomial remaining_people members_to_choose

-- Statement of the problem to prove the number of ways is 126
theorem committee_count_is_correct :
  number_of_ways_to_form_committee = 126 :=
by
  sorry

end committee_count_is_correct_l775_775095


namespace largest_of_five_l775_775591

def a : ℝ := 0.994
def b : ℝ := 0.9399
def c : ℝ := 0.933
def d : ℝ := 0.9940
def e : ℝ := 0.9309

theorem largest_of_five : (a > b ∧ a > c ∧ a ≥ d ∧ a > e) := by
  -- We add sorry here to skip the proof
  sorry

end largest_of_five_l775_775591


namespace largest_power_of_3_l775_775996

theorem largest_power_of_3 (q : ℝ) (h : q = ∑ k in finset.range 8 + 1, k * real.log (2 * k)) : 
  (∃ n : ℕ, e^q = 3^9 * n) :=
sorry

end largest_power_of_3_l775_775996


namespace trajectory_equation_max_value_ratio_l775_775033

-- Introduce points and lines
def point (x y : ℝ) := (x, y)
def line (y : ℝ → ℝ) := y

-- Fixed points and line
def F := point 0 1
def l := line (λ x, -1)

-- Define condition
axiom trajectory_condition (P Q : ℝ × ℝ) : 
  P.2 ≠ -1 → Q = (P.1, -1) → 
  let QF := (Q.1 - F.1, Q.2 - F.2),
      FP := (F.1 - P.1, F.2 - P.2) in
  (Q.1 - P.1) * QF.1 + (Q.2 - P.2) * QF.2 = FP.1 * (F.1 - Q.1) + FP.2 * (F.2 - Q.2)

-- Define the statement of the first problem
theorem trajectory_equation (P : ℝ × ℝ) : 
  (P.2 ≠ -1 → P = (P.1, P.2) → 
  (∃ Q : ℝ × ℝ, Q = (P.1, -1) ∧ trajectory_condition P Q) → 
  P.1^2 = 4 * P.2) := 
  sorry

-- Define the second problem parameters based on center of circle moving on the trajectory C
def D := point 0 2

def circle (M : ℝ × ℝ) (r : ℝ) (x y : ℝ) := 
  (x - M.1)^2 + (y - M.2)^2 = r^2

theorem max_value_ratio (a b : ℝ) (M := point a b) : 
  a^2 = 4 * b →
  let l1 := real.sqrt ((a - 2)^2 + 4),
      l2 := real.sqrt ((a + 2)^2 + 4) in
  b = 2 → 
  (circle M (real.sqrt (a^2 + (b - 2)^2)) M.1 0 ∧ circle M (real.sqrt (a^2 + (b - 2)^2)) D.1 D.2) → 
  real.sup_set (λ a b, (l1 / l2 + l2 / l1)) = 2 * real.sqrt 2 := 
  sorry

end trajectory_equation_max_value_ratio_l775_775033


namespace simplify_exponents_l775_775238

theorem simplify_exponents :
  (2 : ℝ) ^ 0.3 * (2 : ℝ) ^ 0.7 * (2 : ℝ) ^ 0.5 * (2 : ℝ) ^ 0.4 * (2 : ℝ) ^ 0.1 = 4 :=
by sorry

end simplify_exponents_l775_775238


namespace distinct_triangles_in_octahedron_l775_775067

theorem distinct_triangles_in_octahedron : 
  let vertices := 8
  let edges_per_vertex := 4
  ∃ triangles : Nat, -- Existential quantifier for the number of distinct triangles
    triangles = 8 ∧ -- The number of distinct triangles is 8
    (∀ triangle, -- For any given triangle
      (triangle ∈ triangles) →  -- If the triangle is a member of the set of triangles
      (∀ vertex1 vertex2 vertex3, -- The triangle is formed by three vertices
        set.mem vertex1 [triangle] ∧ -- The vertices are in the triangle
        set.mem vertex2 [triangle] ∧
        set.mem vertex3 [triangle] → 
        (vertex1 ≠ vertex2 ∧ 
         vertex2 ≠ vertex3 ∧
         vertex1 ≠ vertex3 ∧ 
         ¬ (edge vertex1 vertex2) ∈ edges ∧ -- None of the edges of the triangle are edges of the octahedron
         ¬ (edge vertex2 vertex3) ∈ edges ∧
         ¬ (edge vertex3 vertex1) ∈ edges))) := 
sorry

end distinct_triangles_in_octahedron_l775_775067


namespace mike_optimal_strategy_l775_775917

theorem mike_optimal_strategy (p : ℝ) (h1 : 0 < p) (h2 : p < 1) :
  ((1/2 < p ∧ p < 1) → (3 : ℕ)) ∧ ((0 < p ∧ p < 1/2) → (1 : ℕ)) :=
by
  -- We assert the proof but don't need to provide it
  sorry

end mike_optimal_strategy_l775_775917


namespace eval_g_l775_775311

def g (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 + x + 1

theorem eval_g : 3 * g 2 + 2 * g (-2) = -9 := 
by {
  sorry
}

end eval_g_l775_775311


namespace radar_coverage_correct_l775_775355

noncomputable def radar_coverage (r : ℝ) (width : ℝ) : ℝ × ℝ :=
  let θ := Real.pi / 7
  let distance := 40 / Real.sin θ
  let area := 1440 * Real.pi / Real.tan θ
  (distance, area)

theorem radar_coverage_correct : radar_coverage 41 18 = 
  (40 / Real.sin (Real.pi / 7), 1440 * Real.pi / Real.tan (Real.pi / 7)) :=
by
  sorry

end radar_coverage_correct_l775_775355


namespace simplify_f_value_of_f_l775_775362

noncomputable def f (α : ℝ) : ℝ :=
  (sin (α - (π / 2)) * cos ((3 * π / 2) - α) * tan (π + α) * cos ((π / 2) + α)) / 
  (sin (2 * π - α) * tan (-α - π) * sin (-α - π))

theorem simplify_f (α : ℝ) : f(α) = -cos(α) :=
  sorry

theorem value_of_f : f(-31 * π / 3) = -1 / 2 :=
  sorry

end simplify_f_value_of_f_l775_775362


namespace sum_abs_arithmetic_sequence_l775_775441

variable (a : ℕ → ℝ)

noncomputable def S (n : ℕ) : ℝ := (Finset.range n).sum (λ k, a (k + 1))

theorem sum_abs_arithmetic_sequence (h1 : 0 < a 1)
  (h2 : a 10 * a 11 < 0)
  (hS10 : S a 10 = 36)
  (hS18 : S a 18 = 12) :
  (Finset.range 18).sum (λ k, |a (k + 1)|) = 60 :=
sorry

end sum_abs_arithmetic_sequence_l775_775441


namespace correct_arrangements_count_l775_775573

def valid_arrangements_count : Nat :=
  let houses := ['O', 'R', 'B', 'Y', 'G']
  let arrangements := houses.permutations
  let valid_arr := arrangements.filter (fun a =>
    let o_idx := a.indexOf 'O'
    let r_idx := a.indexOf 'R'
    let b_idx := a.indexOf 'B'
    let y_idx := a.indexOf 'Y'
    let constraints_met :=
      o_idx < r_idx ∧       -- O before R
      b_idx < y_idx ∧       -- B before Y
      (b_idx + 1 != y_idx) ∧ -- B not next to Y
      (r_idx + 1 != b_idx) ∧ -- R not next to B
      (b_idx + 1 != r_idx)   -- symmetrical R not next to B

    constraints_met)
  valid_arr.length

theorem correct_arrangements_count : valid_arrangements_count = 5 :=
  by
    -- To be filled with proof steps.
    sorry

end correct_arrangements_count_l775_775573


namespace percent_boys_in_class_l775_775094

-- Define the conditions given in the problem
def initial_ratio (b g : ℕ) : Prop := b = 3 * g / 4

def total_students_after_new_girls (total : ℕ) (new_girls : ℕ) : Prop :=
  total = 42 ∧ new_girls = 4

-- Define the percentage calculation correctness
def percentage_of_boys (boys total : ℕ) (percentage : ℚ) : Prop :=
  percentage = (boys : ℚ) / (total : ℚ) * 100

-- State the theorem to be proven
theorem percent_boys_in_class
  (b g : ℕ)   -- Number of boys and initial number of girls
  (total new_girls : ℕ) -- Total students after new girls joined and number of new girls
  (percentage : ℚ) -- The percentage of boys in the class
  (h_initial_ratio : initial_ratio b g)
  (h_total_students : total_students_after_new_girls total new_girls)
  (h_goals : g + new_girls = total - b)
  (h_correct_calc : percentage = 35.71) :
  percentage_of_boys b total percentage :=
by
  sorry

end percent_boys_in_class_l775_775094


namespace minimal_value_box_l775_775319

theorem minimal_value_box (a b : ℤ) (h1 : a ≠ b ∧ b ≠ 332 ∧ 332 ≠ a) (h2 : (ax + b) * (bx + 2a) = 36x^2 + 332x + 72) :
  2 * a^2 + b^2 = 332 :=
by
  sorry

end minimal_value_box_l775_775319


namespace definite_integral_cos_exp_l775_775320

open Real

theorem definite_integral_cos_exp :
  ∫ x in -π..0, (cos x + exp x) = 1 - (1 / exp π) :=
by
  sorry

end definite_integral_cos_exp_l775_775320


namespace find_y_relation_l775_775038

variables {α β x y : ℝ}

-- Conditions 
def isAcuteAngle (θ : ℝ) : Prop := 0 < θ ∧ θ < Real.pi / 2
def condition1 := isAcuteAngle α ∧ isAcuteAngle β
def condition2 := cos (α + β) = -4 / 5
def condition3 := sin β = x
def condition4 := cos α = y
def validRange := 4 / 5 < x ∧ x < 1

-- The function relation to be proven
def yRelation (x : ℝ) : ℝ := -4 / 5 * Real.sqrt (1 - x^2) + 3 / 5 * x

-- The main theorem to be proven
theorem find_y_relation (h1 : condition1) 
                        (h2 : condition2) 
                        (h3 : condition3) 
                        (h4 : condition4) 
                        (h5 : validRange) : 
   y = yRelation x :=
sorry

end find_y_relation_l775_775038


namespace math_vs_english_time_difference_l775_775124

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l775_775124


namespace eventually_constant_mod_l775_775669

def f : Nat → Nat
| 1     => 2
| (n+1) => 2 * f n

theorem eventually_constant_mod (m : Nat) :
  ∃ k, ∀ n ≥ k, f n % m = f (n+1) % m :=
sorry

end eventually_constant_mod_l775_775669


namespace geometric_intersection_l775_775107

noncomputable def cartesian_equation_line_l : ∀ (x y: ℝ), Prop := λ x y, 
  sqrt 3 * x - y + 2 * sqrt 3 - 3 = 0

noncomputable def cartesian_equation_curve_c : ∀ (x y: ℝ), Prop := λ x y, 
  (x - 2)^2 + (y - 2)^2 = 8

theorem geometric_intersection
  (polar_eq_C : ∀ θ: ℝ, 4 * sqrt 2 * sin (θ + π / 4) = sqrt (x^2 + y^2))
  (param_eq_line_l : ∀ t: ℝ, let x := -2 + (1 / 2) * t in let y := -3 + (sqrt 3 / 2) * t in true)
  (point_P : ∃ (x y: ℝ), x = -2 ∧ y = -3)
  : ∃ (A B: ℝ × ℝ), --existence of points A and B
    (A.1, A.2) = (-2 + (1 / 2) * t1, -3 + (sqrt 3 / 2) * t1) ∧
    (B.1, B.2) = (-2 + (1 / 2) * t2, -3 + (sqrt 3 / 2) * t2) ∧
    cartesian_equation_line_l A.1 A.2 ∧ 
    cartesian_equation_line_l B.1 B.2 ∧
    cartesian_equation_curve_c A.1 A.2 ∧ 
    cartesian_equation_curve_c B.1 B.2 ∧
    |PA| * |PB| =33 := 
sorry

end geometric_intersection_l775_775107


namespace number_of_teams_l775_775438

theorem number_of_teams (n : ℕ) (h : (n * (n - 1)) / 2 = 21) : n = 7 :=
sorry

end number_of_teams_l775_775438


namespace aaron_brothers_l775_775640

theorem aaron_brothers :
  ∃ A : ℕ, 6 = 2 * A - 2 ∧ A = 4 :=
begin
  sorry
end

end aaron_brothers_l775_775640


namespace num_ints_congruent_to_2_mod_7_l775_775811

theorem num_ints_congruent_to_2_mod_7 :
  ∃ n : ℕ, (∀ k, 1 ≤ 7 * k + 2 ∧ 7 * k + 2 ≤ 300 ↔ 0 ≤ k ≤ 42) ∧ n = 43 :=
sorry

end num_ints_congruent_to_2_mod_7_l775_775811


namespace parallel_coordinates_max_value_l775_775803

open Real

-- Definition of the vectors and their properties
def vec_a (x : ℝ) : ℝ × ℝ := (sin x, 1)
def vec_b (x : ℝ) : ℝ × ℝ := (sin x, cos x + 1)

-- Parallel condition (I)
theorem parallel_coordinates (x : ℝ) (hx : sin x * (cos x + 1) = sin x) :
  (vec_a x = (0, 1) ∧ (vec_b x = (0, 2) ∨ vec_b x = (0, 0))) ∨
  (vec_a x = (1, 1) ∧ vec_b x = (1, 1)) ∨
  (vec_a x = (-1, 1) ∧ vec_b x = (-1, 2)) := 
sorry

-- Maximum value condition (II)
def f (x : ℝ) : ℝ := (vec_a x).1 * (vec_b x).1 + (vec_a x).2 * (vec_b x).2

theorem max_value (x : ℝ) (hx : x ∈ Icc (-(π/2)) (π/2)) :
  ∃ (c : ℝ), f c = 9 / 4 ∧ (c = π / 3 ∨ c = -π / 3) := 
sorry

end parallel_coordinates_max_value_l775_775803


namespace wrapping_paper_needed_l775_775306

-- Define the conditions as variables in Lean
def wrapping_paper_first := 3.5
def wrapping_paper_second := (2 / 3) * wrapping_paper_first
def wrapping_paper_third := wrapping_paper_second + 0.5 * wrapping_paper_second
def wrapping_paper_fourth := wrapping_paper_first + wrapping_paper_second
def wrapping_paper_fifth := wrapping_paper_third - 0.25 * wrapping_paper_third

-- Define the total wrapping paper needed
def total_wrapping_paper := wrapping_paper_first + wrapping_paper_second + wrapping_paper_third + wrapping_paper_fourth + wrapping_paper_fifth

-- Statement to prove the final equivalence
theorem wrapping_paper_needed : 
  total_wrapping_paper = 17.79 := 
sorry  -- Proof is omitted

end wrapping_paper_needed_l775_775306


namespace function_monotonic_increasing_l775_775088

theorem function_monotonic_increasing (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (x1 - x2) * (f a x1 - f a x2) > 0) ↔ 4 ≤ a ∧ a < 8 
  where
  f : ℝ → ℝ → ℝ
  | a, x => if x > 1 then a^x else (4 - a / 2) * x + 2 :=
sorry

end function_monotonic_increasing_l775_775088


namespace hexagon_area_l775_775165

theorem hexagon_area (I B : ℕ) : I = 11 → B = 6 → I + B / 2 - 1 = 13 :=
by
  intros hI hB
  rw [hI, hB]
  let A := I + B / 2 - 1
  have : A = 11 + 6 / 2 - 1 := by
    rw [←hI, ←hB]
    exact rfl
  exact this
sorry

end hexagon_area_l775_775165


namespace water_balloons_packs_l775_775496

theorem water_balloons_packs
    (x : ℕ)
    (three_packs_own : ℕ := 3 * x)
    (two_packs_neighbor : ℕ := 2 * x)
    (total_balloons : ℕ := three_packs_own + two_packs_neighbor)
    (half_balloons : ℕ := total_balloons / 2)
    (milly_balloons : ℕ := half_balloons + 7)
    (floretta_balloons : ℕ := half_balloons - 7)
    (floretta_final_balloons : ℕ := 8) :
    x = 6 :=
 by
    have total_balloons := three_packs_own + two_packs_neighbor
    have split_evenly : total_balloons / 2 = floretta_final_balloons + 7
    have floretta_balloons_calc := floretta_final_balloons + 7
    have total_balloons_calc := floretta_balloons_calc * 2
    have x_calc : 5 * x = total_balloons_calc
    sorry


end water_balloons_packs_l775_775496


namespace sum_series_to_fraction_l775_775331

theorem sum_series_to_fraction :
  (∑ n in Finset.range 9, (1 / ((n + 2) * (n + 3) : ℚ))) = 9 / 22 := 
begin
  sorry
end

end sum_series_to_fraction_l775_775331


namespace total_time_to_meet_l775_775644

-- Definitions of conditions
def length_of_park : ℝ := 24
def closing_rate : ℝ := 0.8
def biking_time : ℝ := 7
def waiting_time : ℝ := 3

-- Definitions related to speeds
def v_L := closing_rate / 3
def v_A := 2 * v_L

-- Proof
theorem total_time_to_meet :
  let distance_covered := closing_rate * biking_time;
  let remaining_distance := length_of_park - distance_covered;
  let time_to_cover_remaining := remaining_distance / v_L;
  let total_time := biking_time + waiting_time + time_to_cover_remaining in
  total_time = 79 :=
by
  sorry

end total_time_to_meet_l775_775644


namespace fourth_machine_works_for_12_hours_daily_l775_775432

noncomputable def hours_fourth_machine_works (m1_hours m1_production_rate: ℕ) (m2_hours m2_production_rate: ℕ) (price_per_kg: ℕ) (total_earning: ℕ) :=
  let m1_total_production := m1_hours * m1_production_rate
  let m1_total_output := 3 * m1_total_production
  let m1_revenue := m1_total_output * price_per_kg
  let remaining_revenue := total_earning - m1_revenue
  let m2_total_production := remaining_revenue / price_per_kg
  m2_total_production / m2_production_rate

theorem fourth_machine_works_for_12_hours_daily : hours_fourth_machine_works 23 2 (sorry) (sorry) 50 8100 = 12 := by
  sorry

end fourth_machine_works_for_12_hours_daily_l775_775432


namespace sum_even_integers_602_to_700_l775_775561

-- Definitions based on the conditions and the problem statement
def sum_first_50_even_integers := 2550
def n_even_602_700 := 50
def first_term_602_to_700 := 602
def last_term_602_to_700 := 700

-- Theorem statement
theorem sum_even_integers_602_to_700 : 
  sum_first_50_even_integers = 2550 → 
  n_even_602_700 = 50 →
  (n_even_602_700 / 2) * (first_term_602_to_700 + last_term_602_to_700) = 32550 :=
by
  sorry

end sum_even_integers_602_to_700_l775_775561


namespace part1_part2_l775_775112

variables {A B C : ℝ} {a b c : ℝ}

-- conditions of the problem
def condition_1 (a b c : ℝ) (C : ℝ) : Prop :=
  a * Real.cos C + Real.sqrt 3 * Real.sin C - b - c = 0

def condition_2 (C : ℝ) : Prop :=
  0 < C ∧ C < Real.pi

-- Part 1: Proving the value of angle A
theorem part1 (a b c C : ℝ) (h1 : condition_1 a b c C) (h2 : condition_2 C) : 
  A = Real.pi / 3 :=
sorry

-- Part 2: Range of possible values for the perimeter, given c = 3
def is_acute_triangle (A B C : ℝ) : Prop :=
  A < Real.pi / 2 ∧ B < Real.pi / 2 ∧ C < Real.pi / 2

theorem part2 (a b A B C : ℝ) (h1 : condition_1 a b 3 C) (h2 : condition_2 C) 
           (h3 : A = Real.pi / 3) (h4 : is_acute_triangle A B C) :
  ∃ p, p ∈ Set.Ioo ((3 * Real.sqrt 3 + 9) / 2) (9 + 3 * Real.sqrt 3) :=
sorry

end part1_part2_l775_775112


namespace count_periodic_values_l775_775368

noncomputable def sequence_relation (x : ℝ) (n : ℕ) : ℝ :=
  if 2 * x < 1.25 then 2 * x
  else 2 * x - 1.25

def periodic (x0 : ℝ) : Prop :=
  ∀ n : ℕ, sequence_relation x0 n = sequence_relation x0 (n + 7)

theorem count_periodic_values :
  ∃ (n : ℕ), periodic n = 128 :=
sorry

end count_periodic_values_l775_775368


namespace fraction_sum_zero_implies_square_sum_zero_l775_775906

theorem fraction_sum_zero_implies_square_sum_zero (a b c : ℝ) (h₀ : a ≠ b) (h₁ : b ≠ c) (h₂ : c ≠ a)
  (h : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a^2 / (b - c)^2 + b^2 / (c - a)^2 + c^2 / (a - b)^2 = 0 := 
by
  sorry

end fraction_sum_zero_implies_square_sum_zero_l775_775906


namespace violet_needs_water_l775_775230

/-- Violet needs 800 ml of water per hour hiked, her dog needs 400 ml of water per hour,
    and they can hike for 4 hours. We need to prove that Violet needs 4.8 liters of water
    for the hike. -/
theorem violet_needs_water (hiking_hours : ℝ)
  (violet_water_per_hour : ℝ)
  (dog_water_per_hour : ℝ)
  (violet_water_needed : ℝ)
  (dog_water_needed : ℝ)
  (total_water_needed_ml : ℝ)
  (total_water_needed_liters : ℝ) :
  hiking_hours = 4 ∧
  violet_water_per_hour = 800 ∧
  dog_water_per_hour = 400 ∧
  violet_water_needed = 3200 ∧
  dog_water_needed = 1600 ∧
  total_water_needed_ml = 4800 ∧
  total_water_needed_liters = 4.8 →
  total_water_needed_liters = 4.8 :=
by sorry

end violet_needs_water_l775_775230


namespace percentage_of_green_shirts_l775_775844

theorem percentage_of_green_shirts (n b r o g : ℕ)
  (h₁ : n = 700)
  (h₂ : b = 0.45 * 700)
  (h₃ : r = 0.23 * 700)
  (h₄ : o = 119)
  (h₅ : g = n - (b + r + o)) :
  g * 100 / n = 15 :=
by {
  sorry
}

end percentage_of_green_shirts_l775_775844


namespace count_integers_congruent_to_2_mod_7_up_to_300_l775_775808

theorem count_integers_congruent_to_2_mod_7_up_to_300 : 
  (Finset.card (Finset.filter (λ n : ℕ, n % 7 = 2) (Finset.range 301))) = 43 := 
by
  sorry

end count_integers_congruent_to_2_mod_7_up_to_300_l775_775808


namespace solve_equation_l775_775937

theorem solve_equation : ∀ x : ℝ, 2 * x^2 - 3 * x = 0 ↔ (x = 0 ∨ x = 3 / 2) :=
by
  intro x
  split
  . intro h
    have := eq_zero_or_eq_zero_of_mul_eq_zero h
    apply Or.imp _ _ this
    . intro h1
      exact h1
    . intro h2
      have := eq_div_of_mul_eq h2
      rwa [mul_comm, mul_one] at this

  . intro hx
    cases hx
    . rw hx
      ring
    
    . rw [hx, mul_div_cancel']
      ring

end solve_equation_l775_775937


namespace sqrt_eq_self_iff_eq_zero_l775_775555

theorem sqrt_eq_self_iff_eq_zero (x : ℝ) : sqrt x = x ↔ x = 0 :=
by
  sorry

end sqrt_eq_self_iff_eq_zero_l775_775555


namespace total_handshakes_l775_775609

theorem total_handshakes (total_people : ℕ) (first_meeting_people : ℕ) (second_meeting_new_people : ℕ) (common_people : ℕ)
  (total_people_is : total_people = 12)
  (first_meeting_people_is : first_meeting_people = 7)
  (second_meeting_new_people_is : second_meeting_new_people = 5)
  (common_people_is : common_people = 2)
  (first_meeting_handshakes : ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2)
  (second_meeting_handshakes: ℕ := (first_meeting_people * (first_meeting_people - 1)) / 2 - (common_people * (common_people - 1)) / 2):
  first_meeting_handshakes + second_meeting_handshakes = 41 := 
sorry

end total_handshakes_l775_775609


namespace compute_expression_l775_775308

theorem compute_expression : 3 * 3^4 - 9^19 / 9^17 = 162 := by
  sorry

end compute_expression_l775_775308


namespace evaluate_expression_l775_775322

theorem evaluate_expression :
  sqrt (1 + 2) + sqrt (1 + 2 + 3) + sqrt (1 + 2 + 3 + 4) + sqrt (1 + 2 + 3 + 4 + 5) + 2 =
  sqrt 3 + sqrt 6 + sqrt 10 + sqrt 15 + 2 := 
sorry

end evaluate_expression_l775_775322


namespace time_to_traverse_nth_mile_l775_775626

-- Define the conditions
def speed (k : ℝ) (d : ℕ) : ℝ := k / (2 * d + 1)

def time_to_traverse (k : ℝ) (n : ℕ) : ℝ := 1 / speed k (n - 1)

-- The main theorem to prove
theorem time_to_traverse_nth_mile (n : ℕ) (h₃ : time_to_traverse (5 / 3) 3 = 3) : 
    time_to_traverse (5 / 3) n = 3 * (2 * n - 1) / 5 := 
sorry

end time_to_traverse_nth_mile_l775_775626


namespace breadth_of_brick_l775_775619

theorem breadth_of_brick (length_courtyard breadth_courtyard length_brick number_bricks : ℕ) 
(h1 : length_courtyard = 2000) 
(h2 : breadth_courtyard = 1600)
(h3 : length_brick = 20)
(h4 : number_bricks = 16000) 
(h5 : length_courtyard * breadth_courtyard = 3200000) 
(h6 : area_courtyard = length_courtyard * breadth_courtyard)
(h7 : ∀ breadth_brick, 20 * breadth_brick = 320000 * breadth_brick)
(h8 : 320000 * breadth_brick = area_courtyard) :
breadth_of_brick = 10 :=
by
  -- We are given and we should use these hypotheses,
  -- h1, h2, h3, h4, h5, h6, h7
  sorry

end breadth_of_brick_l775_775619


namespace meaningful_expression_range_l775_775423

theorem meaningful_expression_range (x : ℝ) : 
  (x - 1 ≥ 0) ∧ (x ≠ 3) ↔ (x ≥ 1 ∧ x ≠ 3) := 
by
  sorry

end meaningful_expression_range_l775_775423


namespace vertex_of_quadratic_function_l775_775703

-- Define the function and constants
variables (p q : ℝ)
  (hp : p > 0)
  (hq : q > 0)

-- State the theorem
theorem vertex_of_quadratic_function : 
  ∀ p q : ℝ, p > 0 → q > 0 → 
  (∀ x : ℝ, x = - (2 * p) / (2 : ℝ) → x = -p) := 
sorry

end vertex_of_quadratic_function_l775_775703


namespace joan_exam_time_difference_l775_775125

theorem joan_exam_time_difference :
  (let english_questions := 30
       math_questions := 15
       english_time_hours := 1
       math_time_hours := 1.5
       english_time_minutes := english_time_hours * 60
       math_time_minutes := math_time_hours * 60
       time_per_english_question := english_time_minutes / english_questions
       time_per_math_question := math_time_minutes / math_questions
    in time_per_math_question - time_per_english_question = 4) :=
by
  sorry

end joan_exam_time_difference_l775_775125


namespace boys_without_microscopes_l775_775835

theorem boys_without_microscopes
    (total_boys : ℕ)
    (total_students_with_microscopes : ℕ)
    (girls_with_microscopes : ℕ)
    (total_boys = 24)
    (total_students_with_microscopes = 30)
    (girls_with_microscopes = 18) :
    (total_boys - (total_students_with_microscopes - girls_with_microscopes) = 12) :=
by
  sorry

end boys_without_microscopes_l775_775835


namespace trapezoid_circumradius_l775_775449

theorem trapezoid_circumradius (A B C D E : Point) (AD BC : ℝ)
  (h1 : AD = 9) (h2 : BC = 2)
  (h3 : ∠A = Real.arctan 4) (h4 : ∠D = Real.arctan (2/3))
  (h5 : E is_intersection_of AC BD)
  (hTrapezoid : is_trapezoid A B C D) :
  circumradius_triangle C B E = (5 * Real.sqrt 5) / 11 :=
sorry

end trapezoid_circumradius_l775_775449


namespace three_digit_sum_26_l775_775729

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem three_digit_sum_26 : 
  ∃! (n : ℕ), is_three_digit n ∧ digit_sum n = 26 := 
sorry

end three_digit_sum_26_l775_775729


namespace probability_odd_function_l775_775037

def f1 (x : ℝ) : ℝ := x
def f2 (x : ℝ) : ℝ := Real.sin x
def f3 (x : ℝ) : ℝ := Real.cos x
def f4 (x : ℝ) : ℝ := x⁻¹

theorem probability_odd_function (f1 f2 f3 f4 : ℝ → ℝ) :
  let functions := [f1, f2, f3, f4]
  let odd_functions := [f1, f2, f4]
  let even_functions := [f3]
  let pairs := (finset.univ : finset (ℕ × ℕ)).filter (λ ⟨i, j⟩, i < j)
  let odd_pairs := pairs.filter (λ ⟨i, j⟩, functions.nth i ∈ odd_functions ∧ functions.nth j ∈ even_functions ∨
                                      functions.nth i ∈ even_functions ∧ functions.nth j ∈ odd_functions)
  (odd_pairs.card : ℚ) / (pairs.card : ℚ) = 1 / 2 :=
by
  sorry

end probability_odd_function_l775_775037


namespace find_n_l775_775338

theorem find_n (n : ℕ) (h : n ≥ 2) : 
  (∀ (i j : ℕ), 0 ≤ i ∧ i ≤ n ∧ 0 ≤ j ∧ j ≤ n → (i + j) % 2 = (Nat.choose n i + Nat.choose n j) % 2) ↔ ∃ k : ℕ, k ≥ 1 ∧ n = 2^k - 2 :=
by
  sorry

end find_n_l775_775338


namespace sum_three_numbers_l775_775562

theorem sum_three_numbers 
  (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 52) 
  (h2 : ab + bc + ca = 72) : 
  a + b + c = 14 := 
by 
  sorry

end sum_three_numbers_l775_775562


namespace sabrina_cookies_l775_775174

theorem sabrina_cookies :
  let cookies_initial := 84 in
  let cookies_after_brother := cookies_initial - 20 in
  let cookies_after_mother := cookies_after_brother + (20 * 2) in
  let cookies_after_sister := cookies_after_mother - Int((cookies_after_mother : ℚ) / 5) in
  let cookies_after_father := cookies_after_sister + 36 / 4 in
  let cookies_after_grandmother := cookies_after_father + 15 in
  let cookies_after_cousin := cookies_after_grandmother - Int((cookies_after_grandmother : ℚ) * (3 / 7)) in
  let cookies_after_best_friend := cookies_after_cousin - Int((cookies_after_cousin : ℚ) / 4) in
  cookies_after_best_friend = 46 :=
by
  sorry

end sabrina_cookies_l775_775174


namespace fraction_is_three_halves_l775_775071

theorem fraction_is_three_halves (a b : ℝ) (hb : b ≠ 0) (h : 2 * a = 3 * b) : a / b = 3 / 2 :=
sorry

end fraction_is_three_halves_l775_775071


namespace contrapositive_equivalence_l775_775542

variable (a b : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem contrapositive_equivalence :
  (∀ a b, is_odd a → is_odd b → is_odd (a * b)) ↔ (∀ a b, ¬ is_odd (a * b) → ¬ (is_odd a ∧ is_odd b)) :=
begin
  sorry
end

end contrapositive_equivalence_l775_775542


namespace blue_bird_chess_team_arrangements_l775_775534

theorem blue_bird_chess_team_arrangements : 
  let boys := 3
  let girls := 3
  (∃ arrangements : ℕ, arrangements = 36) :=
by
  let total_arrangements := nat.factorial boys * nat.factorial girls
  existsi total_arrangements
  sorry

end blue_bird_chess_team_arrangements_l775_775534


namespace aml_1996_pn_pos_real_numbers_l775_775886

theorem aml_1996_pn_pos_real_numbers
  (n : ℕ) (h1 : n > 0)
  (x : Fin n → ℝ)
  (hx_pos : ∀ i, 0 < x i)
  (hx_diff : ∀ i j, i < j → |x i - x j| ≤ 1) :
  (∑ i in Fin.range n, x i / x ((i + 1) % n) ) ≥ 
  (∑ i in Fin.range n, (x ((i + 1) % n) + 1) / (x i + 1)) :=
by
  sorry

end aml_1996_pn_pos_real_numbers_l775_775886


namespace total_surface_area_of_tower_l775_775013

def volume_to_side_length (v : ℕ) : ℕ :=
  nat.root 3 v

def cube_surface_area (s : ℕ) : ℕ :=
  6 * s * s

def adjusted_cuboid_area (current_side : ℕ) (previous_side : ℕ) : ℕ :=
  cube_surface_area current_side - previous_side ^ 2

noncomputable def total_adjusted_surface_area : ℕ :=
  let sides := [volume_to_side_length 1, volume_to_side_length 27, volume_to_side_length 125, volume_to_side_length 343]
  let adjusted_areas := [sides.head, adjusted_cuboid_area sides.nth 1 sides.head, adjusted_cuboid_area sides.nth 2 sides.nth 1, cube_surface_area (sides.nth 3)]
  adjusted_areas.sum

theorem total_surface_area_of_tower : total_adjusted_surface_area = 494 := by
  sorry

end total_surface_area_of_tower_l775_775013


namespace sum_partial_fractions_series_l775_775330

theorem sum_partial_fractions_series :
  (∑ n in finset.range 9, 1 / ((n + 2) * (n + 3): ℚ)) = 9 / 22 := sorry

end sum_partial_fractions_series_l775_775330


namespace set_C_cannot_form_triangle_l775_775295

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Given conditions
def set_A := (3, 6, 8)
def set_B := (3, 8, 9)
def set_C := (3, 6, 9)
def set_D := (6, 8, 9)

theorem set_C_cannot_form_triangle : ¬ is_triangle 3 6 9 :=
by
  -- Proof is omitted
  sorry

end set_C_cannot_form_triangle_l775_775295


namespace blueberries_frozen_l775_775537

-- Lean Statement for the given math problem
theorem blueberries_frozen (total_production : ℝ)
  (percentage_mixed : ℝ)
  (percentage_frozen : ℝ)
  (percentage_direct : ℝ)
  (h1 : total_production = 4.8)
  (h2 : percentage_mixed = 0.25)
  (h3 : percentage_frozen = 0.40)
  (h4 : percentage_direct = 0.60) :
  let remaining := total_production * (1 - percentage_mixed)
  let frozen := remaining * percentage_frozen in
  Float.round_decimals 1 frozen = 1.4 :=
by
  sorry

end blueberries_frozen_l775_775537


namespace cyclic_quadrilateral_condition_l775_775907

theorem cyclic_quadrilateral_condition
  (A B C D O : Type)
  [planar_geometry ABCD O]
  (h_intersection : intersection AC BD = O)
  (h_condition : OA * sin (angle A) + OC * sin (angle C) = OB * sin (angle B) + OD * sin (angle D)) :
  is_cyclic_quadrilateral ABCD :=
sorry

end cyclic_quadrilateral_condition_l775_775907


namespace correct_judgements_l775_775668

noncomputable def f : ℝ → ℝ :=
  sorry

axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_period_1 : ∀ x : ℝ, f (x + 1) = -f x
axiom f_increasing_0_1 : ∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ 1 → f x ≤ f y

theorem correct_judgements : 
  (∀ x : ℝ, f (x + 2) = f x) ∧ 
  (∀ x : ℝ, f (1 - x) = f (1 + x)) ∧ 
  (∀ x y : ℝ, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f x ≥ f y) ∧ 
  ¬(∀ x y : ℝ, -2 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≥ f y) :=
by 
  sorry

end correct_judgements_l775_775668


namespace slope_of_obtuse_angle_line_l775_775491

-- Definitions of the parabola and points
def parabola (x y : ℝ) : Prop :=
  y^2 = 4*x

def focus : (ℝ × ℝ) := (1, 0)

def line_through_focus (k x : ℝ) : ℝ :=
  k * (x - 1)

-- Given conditions
def line_passing_through (F : ℝ × ℝ) (k x y : ℝ) : Prop :=
  y = k * (x - F.1)

def points_on_parabola (A B : ℝ × ℝ) : Prop :=
  parabola A.1 A.2 ∧ parabola B.1 B.2

def segment_length (A B : ℝ × ℝ) (len : ℝ) : Prop :=
  (real.dist A B) = len

-- Theorem to be proven
theorem slope_of_obtuse_angle_line
  (A B : ℝ × ℝ)
  (F := focus)
  (k : ℝ)
  (h1 : points_on_parabola A B)
  (h2 : line_passing_through F k A.1 A.2)
  (h3 : line_passing_through F k B.1 B.2)
  (h4 : segment_length A B (16/3)) :
  k = -real.sqrt 3 := sorry

end slope_of_obtuse_angle_line_l775_775491


namespace meaningful_expression_range_l775_775424

theorem meaningful_expression_range (x : ℝ) : 
  (x - 1 ≥ 0) ∧ (x ≠ 3) ↔ (x ≥ 1 ∧ x ≠ 3) := 
by
  sorry

end meaningful_expression_range_l775_775424


namespace graph_contains_minor_subgraph_l775_775617

-- Define the main theorem stating that a graph G contains K^5 or K_{3,3} as a subgraph if and only if it contains K^5 or K_{3,3} as a minor.
theorem graph_contains_minor_subgraph (G : Graph) :
  (∃ H : Graph, (H.subgraph G) ∧ (H.isomorphic_to K_5 ∨ H.isomorphic_to K_{3,3})) ↔
  (∃ H : Graph, (H.minor_of G) ∧ (H.isomorphic_to K_5 ∨ H.isomorphic_to K_{3,3})) := 
sorry

end graph_contains_minor_subgraph_l775_775617


namespace distinct_real_solutions_l775_775789

open Real Nat

noncomputable def p_n : ℕ → ℝ → ℝ 
| 0, x => x
| (n+1), x => (p_n n (x^2 - 2))

theorem distinct_real_solutions (n : ℕ) : 
  ∃ S : Finset ℝ, S.card = 2^n ∧ ∀ x ∈ S, p_n n x = x ∧ (∀ y ∈ S, x ≠ y → x ≠ y) := 
sorry

end distinct_real_solutions_l775_775789


namespace card_A_l775_775796

def A : Set ℤ := {x : ℤ | (x - 1) * (5 - x) ≥ 0}

theorem card_A : (A.toFinset.card) = 5 := by {
  sorry
}

end card_A_l775_775796


namespace jane_received_change_l775_775875

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l775_775875


namespace find_lambda_l775_775802

noncomputable def vector_2d := ℝ × ℝ

variable (a b : vector_2d)
variable (λ : ℝ)

def magnitude (v : vector_2d) : ℝ :=
(real.sqrt (v.1^2 + v.2^2))

axiom a_value : a = (1, real.sqrt 3)
axiom b_magnitude : magnitude b = 1
axiom equation : a + λ • b = (0, 0)

theorem find_lambda : λ = 2 ∨ λ = -2 :=
sorry

end find_lambda_l775_775802


namespace math_vs_english_time_difference_l775_775122

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l775_775122


namespace equation_solution_l775_775012

theorem equation_solution (x y : ℝ) (h : x^2 + (1 - y)^2 + (x - y)^2 = (1 / 3)) : 
  x = (1 / 3) ∧ y = (2 / 3) := 
  sorry

end equation_solution_l775_775012


namespace trees_in_one_row_l775_775463

theorem trees_in_one_row (total_revenue : ℕ) (price_per_apple : ℕ) (apples_per_tree : ℕ) (trees_per_row : ℕ)
  (revenue_condition : total_revenue = 30)
  (price_condition : price_per_apple = 1 / 2)
  (apples_condition : apples_per_tree = 5)
  (trees_condition : trees_per_row = 4) :
  trees_per_row = 4 := by
  sorry

end trees_in_one_row_l775_775463


namespace part1_part2_l775_775375

/-- Part 1 -/
def sequence_a (n : ℕ) : ℕ :=
  if n = 0 then 0 else 4 * n - 3

def A (n : ℕ) : ℕ :=
  ∑ i in List.range (n + 1), sequence_a i.succ

def B (n : ℕ) : ℕ :=
  ∑ i in List.range (n + 1), sequence_a (i + 2)

def C (n : ℕ) : ℕ :=
  ∑ i in List.range (n + 1), sequence_a (i + 3)

theorem part1 :
  (∀ n : ℕ, sequence_a 1 = 1 ∧ sequence_a 2 = 5) →
  (∀ n, A n, B n, C n are in arithmetic progression) →
  ∀ n : ℕ, sequence_a (n + 1) = 4 * (n + 1) - 3 :=
by
  sorry

/-- Part 2 -/
theorem part2 :
  (∀ q > 0, (∀ n : ℕ, sequence_a (n + 1) = sequence_a n * q) ↔
  (∀ n, A n, B n, C n are in geometric progression with ratio q)) :=
by
  sorry

end part1_part2_l775_775375


namespace milan_billed_minutes_l775_775712

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l775_775712


namespace coefficient_term_without_x_in_expansion_l775_775689

theorem coefficient_term_without_x_in_expansion 
  (x y : ℝ) : 
  ∃ c : ℝ, (∀ r : ℕ, r = 4 → c = (-1)^r * (Nat.choose 8 r) * y^(8-r)) ∧ c = 70 :=
by {
  use (Nat.choose 8 4 : ℝ),
  split,
  { intros r hr,
    rw hr,
    simp, },
  { norm_num }
}

end coefficient_term_without_x_in_expansion_l775_775689


namespace matrix_not_invertible_implies_fraction_sum_l775_775353

variables {a b c : ℝ}

theorem matrix_not_invertible_implies_fraction_sum :
  det (Matrix.of (λ i j, if i = j then (if i = 0 then a^2 else if i = 1 then b^2 else c^2)
                         else (if (i = 0 ∧ j = 1) ∨ (i = 1 ∧ j = 0) then b^2 else if (i = 0 ∧ j = 2) ∨ (i = 2 ∧ j = 0) then c^2 else a^2))) = 0 →
  (a^2 / (b^2 + c^2)) + (b^2 / (a^2 + c^2)) + (c^2 / (a^2 + b^2)) = 3 / 2 :=
begin
  sorry
end

end matrix_not_invertible_implies_fraction_sum_l775_775353


namespace det_B_eq_8D_l775_775372

-- Define vectors a, b, and c in a suitable vector space
variables {V : Type*} [inner_product_space ℝ V]
variables a b c : V

-- Define the determinant D
def D : ℝ := (inner_product_space.cross_product _ _ _ a (b × c))

-- Define the column vectors of matrix B
noncomputable def B_col1 : V := 2 • a + b
noncomputable def B_col2 : V := 2 • b + c
noncomputable def B_col3 : V := 2 • c + a

-- State the theorem to be proved
theorem det_B_eq_8D : inner_product_space.det _ _ _ B_col1 B_col2 B_col3 = 8 * D :=
sorry

end det_B_eq_8D_l775_775372


namespace donations_received_l775_775289

def profit : Nat := 960
def half_profit: Nat := profit / 2
def goal: Nat := 610
def extra: Nat := 180
def total_needed: Nat := goal + extra
def donations: Nat := total_needed - half_profit

theorem donations_received :
  donations = 310 := by
  -- Proof omitted
  sorry

end donations_received_l775_775289


namespace geom_transform_symmetry_x_axis_l775_775928

theorem geom_transform_symmetry_x_axis :
  ∀ (A B : ℝ × ℝ),
  (A = (4, 3)) →
  (B = (4, -3)) →
  (∃ (T : ℝ × ℝ → ℝ × ℝ),
    (T (4,3) = (4,-3)) ∧
    ∀ (x y : ℝ), (T (x, y)) = (x, -y)) :=
by 
  intros A B hA hB 
  use (λ p: ℝ × ℝ, (p.1, -p.2))
  sorry

end geom_transform_symmetry_x_axis_l775_775928


namespace probability_al_bill_cal_l775_775292

theorem probability_al_bill_cal :
  let nums := (Finset.range 15).map (λ x, x + 1) in
  let total_ways := nat.factorial 15 / (nat.factorial 12) in
  let valid_assignments := 7 in
  valid_assignments.fdiv total_ways == 1 / 390 :=
by
  let nums := (Finset.range 15).map (λ x, x + 1)
  let total_ways := nat.factorial 15 / (nat.factorial 12)
  let valid_assignments := 7
  have : valid_assignments.fdiv total_ways = 1 / 390 := sorry
  exact this

end probability_al_bill_cal_l775_775292


namespace sum_of_exponents_2023_l775_775223

theorem sum_of_exponents_2023 :
  ∃ (s : ℕ) (m : Fin s → ℕ) (b : Fin s → ℤ), 
  (∀ i j, i < j → m i > m j) ∧
  (∀ k, b k = 1 ∨ b k = -1) ∧
  (∑ i, b i * 3 ^ (m i) = 2023) ∧
  (∑ i, m i = 22) := 
by 
  sorry

end sum_of_exponents_2023_l775_775223


namespace total_bill_correct_l775_775300

def first_family_adults := 2
def first_family_children := 3
def second_family_adults := 4
def second_family_children := 2
def third_family_adults := 3
def third_family_children := 4

def adult_meal_cost := 8
def child_meal_cost := 5
def drink_cost_per_person := 2

def calculate_total_cost 
  (adults1 : ℕ) (children1 : ℕ) 
  (adults2 : ℕ) (children2 : ℕ) 
  (adults3 : ℕ) (children3 : ℕ)
  (adult_cost : ℕ) (child_cost : ℕ)
  (drink_cost : ℕ) : ℕ := 
  let meal_cost1 := (adults1 * adult_cost) + (children1 * child_cost)
  let meal_cost2 := (adults2 * adult_cost) + (children2 * child_cost)
  let meal_cost3 := (adults3 * adult_cost) + (children3 * child_cost)
  let drink_cost1 := (adults1 + children1) * drink_cost
  let drink_cost2 := (adults2 + children2) * drink_cost
  let drink_cost3 := (adults3 + children3) * drink_cost
  meal_cost1 + drink_cost1 + meal_cost2 + drink_cost2 + meal_cost3 + drink_cost3
   
theorem total_bill_correct :
  calculate_total_cost
    first_family_adults first_family_children
    second_family_adults second_family_children
    third_family_adults third_family_children
    adult_meal_cost child_meal_cost drink_cost_per_person = 153 :=
  sorry

end total_bill_correct_l775_775300


namespace S6_correct_l775_775152

-- Definitions based on conditions
variables (a₁ r : ℝ) (S : ℕ → ℝ)

-- Sum of first n terms in geometric sequence 
def S_n (a₁ r : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - r^(n : ℝ)) / (1 - r)

-- Given conditions
axiom S2_eq : S_n a₁ r 2 = 3
axiom S4_eq : S_n a₁ r 4 = 15

-- Theorem we need to prove
theorem S6_correct : S_n a₁ r 6 = 63 :=
sorry

end S6_correct_l775_775152


namespace tangent_line_at_0_l775_775198

noncomputable def tangent_line_eq : Prop :=
  ∃ (m b : ℝ), ∀ (x y : ℝ), (y = exp (-x) + 1) → (x + y - 2 = 0)
  
theorem tangent_line_at_0 (x : ℝ) (y : ℝ) (h : y = exp (-x) + 1) : 
  x = 0 → y = 2 → x + y - 2 = 0 :=
by
  sorry

end tangent_line_at_0_l775_775198


namespace greatest_sum_first_quadrant_l775_775971

theorem greatest_sum_first_quadrant (x y : ℤ) (hx_pos : 0 < x) (hy_pos : 0 < y) (h_circle : x^2 + y^2 = 49) : x + y ≤ 7 :=
sorry

end greatest_sum_first_quadrant_l775_775971


namespace rotated_tetrahedron_l775_775172

/- Given a regular tetrahedron, rotating each edge by 180 degrees around the opposite edge results in a regular tetrahedron with three times the original edge length -/
theorem rotated_tetrahedron (T : set ℝ) (is_regular_tetrahedron : regular_tetrahedron T) : 
  ∃ T' : set ℝ, regular_tetrahedron T' ∧ (edge_length T' = 3 * edge_length T) :=
sorry

end rotated_tetrahedron_l775_775172


namespace fraction_ratio_l775_775072

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end fraction_ratio_l775_775072


namespace part1_part2_l775_775381
noncomputable def root : ℝ := (1 - real.sqrt 5) / 2

axiom eq1 : root^2 = root + 1
axiom eq2 : root^3 = 2 * root + 1
axiom eq3 : root^4 = 3 * root + 2
axiom eq4 : root^5 = 5 * root + 3
axiom eq5 : root^6 = 8 * root + 5

theorem part1 : root^7 = 13 * root + 8 :=
by sorry

theorem part2 (n : ℕ) (α β : ℝ) (hn : root^n = α * root + β) : root^(n+1) = (α + β) * root + α :=
by sorry

end part1_part2_l775_775381


namespace prove_nine_lines_l775_775150

noncomputable def exists_nine_lines (k m : ℤ) : Prop :=
  let ellipse_eq := (λ (x y : ℝ), x^2 / 16 + y^2 / 12 = 1)
  let hyperbola_eq := (λ (x y : ℝ), x^2 / 4 - y^2 / 12 = 1)
  let line := (λ (x : ℝ), k * x + m)
  ∃ (A B C D : ℝ × ℝ), 
    (A ∈ {p : ℝ × ℝ | ellipse_eq p.1 p.2} ∧ B ∈ {p : ℝ × ℝ | ellipse_eq p.1 p.2}) ∧
    (C ∈ {p : ℝ × ℝ | hyperbola_eq p.1 p.2} ∧ D ∈ {p : ℝ × ℝ | hyperbola_eq p.1 p.2}) ∧
    (A.2 = line A.1 ∧ B.2 = line B.1 ∧ C.2 = line C.1 ∧ D.2 = line D.1) ∧
    (A ≠ B ∧ C ≠ D) ∧
    let x1 := A.1, x2 := B.1, x3 := C.1, x4 := D.1 in
    x1 + x3 = x2 + x4

theorem prove_nine_lines : ∃ (k m : ℤ), ∃! (l : ℝ → ℝ) (l = λ x, (k * x + m)), exists_nine_lines k m :=
sorry

end prove_nine_lines_l775_775150


namespace gain_per_year_is_120_l775_775281

def principal := 6000
def rate_borrow := 4
def rate_lend := 6
def time := 2

def simple_interest (P R T : Nat) : Nat := P * R * T / 100

def interest_earned := simple_interest principal rate_lend time
def interest_paid := simple_interest principal rate_borrow time
def gain_in_2_years := interest_earned - interest_paid
def gain_per_year := gain_in_2_years / 2

theorem gain_per_year_is_120 : gain_per_year = 120 :=
by
  sorry

end gain_per_year_is_120_l775_775281


namespace unique_a_when_theta_determined_l775_775142

variables {a b : ℝ} {θ : ℝ}

def minimum_value_condition (a b : ℝ) (θ : ℝ) (t : ℝ) : Prop :=
  let expr := a^2 * sin(θ)^2 in
    expr = 1

theorem unique_a_when_theta_determined (a b theta : ℝ) :
  (∀ t : ℝ, (minimum_value_condition a b θ t)) → (∃! a', a' = abs a) :=
by
  sorry

end unique_a_when_theta_determined_l775_775142


namespace calculate_slopes_difference_l775_775766

noncomputable def point_on_ellipse_and_line (x y : ℝ) : Prop :=
  (x^2 / 4 + y^2 / 3 = 1) ∧ (y - x - 3 = 0) ∧ (x ≠ -3) ∧ (x ≠ sqrt 3) ∧ (x ≠ - sqrt 3)

noncomputable def foci_F1 := (-1 : ℝ, 0 : ℝ)
noncomputable def foci_F2 := (1 : ℝ, 0 : ℝ)

noncomputable def slope_PF1 (x y : ℝ) : ℝ :=
  y / (x + 1)

noncomputable def slope_PF2 (x y : ℝ) : ℝ :=
  y / (x - 1)

theorem calculate_slopes_difference (x y : ℝ)
  (h : point_on_ellipse_and_line x y) :
  1 / slope_PF2 x y - 2 / slope_PF1 x y = -1 :=
by sorry

end calculate_slopes_difference_l775_775766


namespace gcd_18_eq_6_l775_775728

theorem gcd_18_eq_6 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 200) : (nat.gcd 18 n = 6) ↔ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 22 ∧ n = 6 * k ∧ ¬(n = 18 * (n / 18))) :=
begin
  sorry
end

end gcd_18_eq_6_l775_775728


namespace complex_division_l775_775663

theorem complex_division (i : ℂ) (hi : i = Complex.I) : (1 + i) / (1 - i) = i :=
by
  sorry

end complex_division_l775_775663


namespace complement_of_intersection_l775_775153

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_of_intersection :
  (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end complement_of_intersection_l775_775153


namespace convex_polygon_can_be_divided_l775_775512

-- Define convex polygon and the requirement for question
def convex_polygon (n : ℕ) := sorry -- To be defined, as defining convexity is complex and out of scope for this exercise.

-- Definition for being able to divide a polygon into convex pentagons
def can_be_divided_into_pentagons (P : Type) [convex_polygon P] : Prop := sorry -- Again, the detailed definition is complex

theorem convex_polygon_can_be_divided (n : ℕ) 
  (h₁ : n ≥ 6) 
  (P : Type) 
  [convex_polygon P] : 
  can_be_divided_into_pentagons P :=
sorry -- Proof goes here

end convex_polygon_can_be_divided_l775_775512


namespace trig_identity_l775_775017

theorem trig_identity (a : ℝ) (h : (1 + Real.sin a) / Real.cos a = -1 / 2) : 
  (Real.cos a / (Real.sin a - 1)) = 1 / 2 := by
  -- Proof goes here
  sorry

end trig_identity_l775_775017


namespace probability_blue_or_purple_is_correct_l775_775613

def total_jelly_beans : ℕ := 7 + 8 + 9 + 10 + 4

def blue_jelly_beans : ℕ := 10

def purple_jelly_beans : ℕ := 4

def blue_or_purple_jelly_beans : ℕ := blue_jelly_beans + purple_jelly_beans

def probability_blue_or_purple : ℚ := blue_or_purple_jelly_beans / total_jelly_beans

theorem probability_blue_or_purple_is_correct :
  probability_blue_or_purple = 7 / 19 :=
by
  sorry

end probability_blue_or_purple_is_correct_l775_775613


namespace hyperbola_equation_l775_775791

noncomputable def hyperbola (a b x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

noncomputable def foci (x y : ℝ) : Prop :=
  (x, y) = (-2, 0) ∨ (x, y) = (2, 0)

noncomputable def point_on_hyperbola (x y : ℝ) (a b : ℝ) : Prop :=
  hyperbola a b x y

theorem hyperbola_equation (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b)
  (h₂ : foci (-2) 0) (h₃ : foci 2 0)
  (h₄ : point_on_hyperbola 3 (Real.sqrt 7) a b) :
  (∀ x y, hyperbola a b x y ↔ x^2 - y^2 = 2) ∧
  (∀ k : ℝ, (let l y x := y = k * x + 2
             in (area_of_triangle_with_origin O E F = 2 * Real.sqrt 2 →
                 (l y x = Real.sqrt 2 * x + 2 ∨ l y x = -Real.sqrt 2 * x + 2)))) :=
by
  sorry

end hyperbola_equation_l775_775791


namespace find_t_value_l775_775792

noncomputable def a : ℕ → ℝ
| 0       := 1
| 1       := 1
| (n + 2) := Real.sqrt ((n + 2) / 2 + a (n + 1) * a n)

theorem find_t_value :
  let t := Real.sqrt 6 / 6 in
  ∀ n : ℕ, 0 < n → t * n < a n ∧ a n < t * n + 1 :=
by sorry

end find_t_value_l775_775792


namespace num_digits_of_prime_started_numerals_l775_775973

theorem num_digits_of_prime_started_numerals (n : ℕ) (h : 4 * 10^(n-1) = 400) : n = 3 := 
  sorry

end num_digits_of_prime_started_numerals_l775_775973


namespace circumcircle_intersection_l775_775945

-- Define our problem context
variables {A B C D E F K O : Point}

-- Conditions
def conditions : Prop :=
  ¬(O ∈ Line A C) ∧
  ¬(O ∈ Line B D) ∧
  intersect (Line A B) (Line C D) = some E ∧
  intersect (Line A D) (Line B C) = some F

-- Proof statement
theorem circumcircle_intersection (h : conditions) :
  ∃ K, (K ∈ circumcircle (Triangle A B F)) ∧
       (K ∈ circumcircle (Triangle C D F)) ∧
       (K ∈ circumcircle (Triangle B E C)) ∧
       (K ∈ circumcircle (Triangle A D E)) ∧
       (K ∈ circumcircle (Triangle B O D)) ∧
       (K ∈ circumcircle (Triangle A O C)) ∧
       (K ∈ Line E F) ∧
       (perpendicular (Line E F) (Line O K)) :=
sorry

end circumcircle_intersection_l775_775945


namespace sasha_coins_l775_775177

theorem sasha_coins (q n d : ℕ) (hq : 0.25 * q + 0.05 * n + 0.10 * d = 4.80)
    (hqn : q = n) (hqd : q = d) : q = 12 :=
by
    sorry

end sasha_coins_l775_775177


namespace remainder_of_x_pow_150_div_by_x_minus_1_cubed_l775_775003

theorem remainder_of_x_pow_150_div_by_x_minus_1_cubed :
  (x : ℤ) → (x^150 % (x - 1)^3) = (11175 * x^2 - 22200 * x + 11026) :=
by
  intro x
  sorry

end remainder_of_x_pow_150_div_by_x_minus_1_cubed_l775_775003


namespace base10_to_base4_156_eq_2130_l775_775576

def base10ToBase4 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec loop (n : ℕ) (acc : List ℕ) : List ℕ :=
      if n = 0 then acc
      else loop (n / 4) ((n % 4) :: acc)
    loop n []

theorem base10_to_base4_156_eq_2130 :
  base10ToBase4 156 = [2, 1, 3, 0] := sorry

end base10_to_base4_156_eq_2130_l775_775576


namespace Matt_overall_profit_l775_775495

def initialValue : ℕ := 8 * 6

def valueGivenAwayTrade1 : ℕ := 2 * 6
def valueReceivedTrade1 : ℕ := 3 * 2 + 9

def valueGivenAwayTrade2 : ℕ := 2 + 6
def valueReceivedTrade2 : ℕ := 2 * 5 + 8

def valueGivenAwayTrade3 : ℕ := 5 + 9
def valueReceivedTrade3 : ℕ := 3 * 3 + 10 + 1

def valueGivenAwayTrade4 : ℕ := 2 * 3 + 8
def valueReceivedTrade4 : ℕ := 2 * 7 + 4

def overallProfit : ℕ :=
  (valueReceivedTrade1 - valueGivenAwayTrade1) +
  (valueReceivedTrade2 - valueGivenAwayTrade2) +
  (valueReceivedTrade3 - valueGivenAwayTrade3) +
  (valueReceivedTrade4 - valueGivenAwayTrade4)

theorem Matt_overall_profit : overallProfit = 23 :=
by
  unfold overallProfit valueReceivedTrade1 valueGivenAwayTrade1 valueReceivedTrade2 valueGivenAwayTrade2 valueReceivedTrade3 valueGivenAwayTrade3 valueReceivedTrade4 valueGivenAwayTrade4
  linarith

end Matt_overall_profit_l775_775495


namespace max_min_value_monotonic_f_l775_775395

noncomputable def f (x : ℝ) (a : ℝ) := x^2 + 2 * a * x + 3

theorem max_min_value (a : ℝ) : a = -2 →
  (∀ x ∈ Set.Icc (-4 : ℝ) 6, f x a ≤ 35) ∧ 
  (∃ x ∈ Set.Icc (-4 : ℝ) 6, f x a = 35) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 6, f x a ≥ -1) ∧ 
  (∃ x ∈ Set.Icc (-4 : ℝ) 6, f x a = -1) :=
begin
  intro ha,
  rw ha,
  sorry,
end

theorem monotonic_f (a : ℝ) :
  (∀ x ∈ Set.Icc (-4 : ℝ) 6, Differentiable ℝ (λ x, f x a)) →
  ∃ S : Set ℝ, S = {a | a < -6 ∨ a > 4} ∧ 
  (∀ a ∈ S, ∀ x1 x2 ∈ Set.Icc (-4 : ℝ) 6, x1 ≤ x2 → f' x1 a ≤ f' x2 a) :=
begin
  intro h,
  use {a | a < -6 ∨ a > 4},
  split,
  { refl },
  { intro a_haS,
    intro x1 x2,
    intro hx1,
    intro hx2,
    intro hle,
    sorry,
  }
end

end max_min_value_monotonic_f_l775_775395


namespace find_angle_between_vectors_l775_775771

variables {a b : EuclideanSpace ℝ (Fin 2)}

def vector_length (v : EuclideanSpace ℝ (Fin 2)) : ℝ :=
  (∑ i, (v i) ^ 2).sqrt

theorem find_angle_between_vectors
  (h1 : vector_length a = sqrt 3)
  (h2 : vector_length b = 4)
  (h3 : inner a (2 • a - b) = 0) :
  real.arccos (inner a b / (vector_length a * vector_length b)) = π / 6 :=
sorry

end find_angle_between_vectors_l775_775771


namespace probability_inside_sphere_in_cube_l775_775628

noncomputable def cube_volume : ℝ := 4^3

noncomputable def sphere_volume : ℝ := (4 * Real.pi / 3) * 2^3

noncomputable def probability : ℝ := sphere_volume / cube_volume

theorem probability_inside_sphere_in_cube :
  -2 ≤ x ∧ x ≤ 2 ∧ -2 ≤ y ∧ y ≤ 2 ∧ -2 ≤ z ∧ z ≤ 2 → (Real.random (x^2 + y^2 + z^2 ≤ 4) = probability) := 
by simp [cube_volume, sphere_volume, probability]; sorry

end probability_inside_sphere_in_cube_l775_775628


namespace area_of_triangle_l775_775474

noncomputable def a : ℝ × ℝ := (4, -1)
noncomputable def b : ℝ × ℝ := (1, 3)
noncomputable def a_plus_3b : ℝ × ℝ := (a.1 + 3 * b.1, a.2 + 3 * b.2)

theorem area_of_triangle : 
  let parallelogram_area := (a.1 * (3 * b.2)) - ((-1) * (3 * b.1))
  let triangle_area := parallelogram_area / 2
  triangle_area = 19.5 :=
by
  let parallelogram_area := (a.1 * (3 * b.2)) - ((-1) * (3 * b.1)) -- |4 * 9 - (-1) * 3|
  let triangle_area := parallelogram_area / 2 -- |(4 * 9 + (-1) * 3) / 2|
  -- Since we are not proving the steps, we just state the result
  exact Eq.refl 19.5

end area_of_triangle_l775_775474


namespace tens_digit_of_8_pow_1701_l775_775995

theorem tens_digit_of_8_pow_1701 :
  let n := 8 ^ 1701 in
  (n % 100) / 10 = 0 :=
by
  -- Introducing the value n
  let n := 8 ^ 1701
  -- Extracting the tens digit
  have h : (n % 100) / 10 = 0
  {
    sorry
  }
  -- Conclusion
  exact h

end tens_digit_of_8_pow_1701_l775_775995


namespace sum_series_fraction_l775_775327

theorem sum_series_fraction :
  (∑ n in Finset.range 9, (1 : ℚ) / ((n + 2) * (n + 3))) = 9 / 22 := sorry

end sum_series_fraction_l775_775327


namespace find_height_of_door_l775_775545

-- Define the basic variables based on the problem conditions
def length_of_room : ℝ := 25
def width_of_room : ℝ := 15
def height_of_room : ℝ := 12
def cost_per_sqft : ℝ := 2
def area_of_one_window : ℝ := 4 * 3
def number_of_windows : ℝ := 3
def total_cost : ℝ := 1812

-- We need to prove the height of the door is 3 feet
def height_of_door (h : ℝ) : Prop :=
  2 * (height_of_room * (length_of_room + width_of_room) - 6 * h - (number_of_windows * area_of_one_window)) = total_cost

theorem find_height_of_door : ∃ h : ℝ, height_of_door h ∧ h = 3 :=
begin
  use 3,
  unfold height_of_door,
  simp,
  norm_num,
  sorry
end

end find_height_of_door_l775_775545


namespace max_min_exponents_l775_775016

-- Definitions based on the conditions
variables {a b : ℝ}
hypothesis h : 0 < a ∧ a < b ∧ b < 1

-- A statement of the maximum and minimum of the four values
theorem max_min_exponents (h : 0 < a ∧ a < b ∧ b < 1) :
  let M := max (max (a^a) (a^b)) (max (b^a) (b^b)),
      m := min (min (a^a) (a^b)) (min (b^a) (b^b))
  in M = b^a ∧ m = a^b :=
sorry

end max_min_exponents_l775_775016


namespace centroid_of_triangle_is_intersection_of_medians_l775_775425

theorem centroid_of_triangle_is_intersection_of_medians 
  {A B C : Point} (hABC : triangle A B C) :
  let G := centroid A B C in
  let D := midpoint A B,
      E := midpoint B C,
      F := midpoint C A in
  let medians_intersect := ∃ G, is_intersection_of (line A E) (line B F) in
  medians_intersect G :=
sorry

end centroid_of_triangle_is_intersection_of_medians_l775_775425


namespace parabola_focus_directrix_distance_l775_775947

noncomputable def focus_to_directrix_distance (a : ℝ) (h : a ≠ 0) : ℝ :=
  let c := a / 4 in
  (c - (-c)).abs

theorem parabola_focus_directrix_distance (a : ℝ) (h : a ≠ 0) :
  focus_to_directrix_distance a h = a / 2 := by
    sorry

end parabola_focus_directrix_distance_l775_775947


namespace ratio_sum_of_arithmetic_sequences_l775_775657

-- Definitions for the arithmetic sequences
def a_num := 3
def d_num := 3
def l_num := 99

def a_den := 4
def d_den := 4
def l_den := 96

-- Number of terms in each sequence
def n_num := (l_num - a_num) / d_num + 1
def n_den := (l_den - a_den) / d_den + 1

-- Sum of the sequences using the sum formula for arithmetic series
def S_num := n_num * (a_num + l_num) / 2
def S_den := n_den * (a_den + l_den) / 2

-- The theorem statement
theorem ratio_sum_of_arithmetic_sequences : S_num / S_den = 1683 / 1200 := by sorry

end ratio_sum_of_arithmetic_sequences_l775_775657


namespace max_ratio_of_n_m_l775_775731

theorem max_ratio_of_n_m
  (m n k : ℕ)
  (hm : m > 0)
  (hn : n > 0)
  (hk : k > 0)
  (h : |m^k - n.factorial| ≤ n)
  : ∃ mn : ℕ, mn = (n / m) ∧ mn ≤ 2 :=
sorry

end max_ratio_of_n_m_l775_775731


namespace even_and_decreasing_function_l775_775202

theorem even_and_decreasing_function (a : ℤ) : 
  (∃ a ∈ {0, 1, 2, 4}, (∃ (k : ℤ), a^2 - 4 * a - 9 = 2 * k) ∧ (a^2 - 4 * a - 9 < 0)) → a = 1 := 
by
  intro h
  sorry

end even_and_decreasing_function_l775_775202


namespace triangle_angle_B_l775_775456

theorem triangle_angle_B (a b A B : ℝ) (h1 : a * Real.cos B = 3 * b * Real.cos A) (h2 : B = A - Real.pi / 6) : 
  B = Real.pi / 6 := by
  sorry

end triangle_angle_B_l775_775456


namespace evaluate_expression_l775_775675

theorem evaluate_expression : 
  (3 * Real.sqrt 10) / (Real.sqrt 3 + Real.sqrt 5 + 2 * Real.sqrt 2) = (3 / 2) * (Real.sqrt 6 + Real.sqrt 2 - 0.8 * Real.sqrt 5) :=
by
  sorry

end evaluate_expression_l775_775675


namespace perimeter_of_inscribed_quadrilateral_eq_twice_diameter_l775_775284

theorem perimeter_of_inscribed_quadrilateral_eq_twice_diameter 
  (rect : Type*) [rectangle rect] 
  (circle : Type*) [circle circle]
  (inscribed : rect ⊆ circle) 
  (midpoints_consecutive_connected : ∀ (midpoints : quadrilateral), midpoints.vertices ⟷ rect.sides.midpoints) 
  : ∀ (P Q R S : Point),
    quadrilateral P Q R S →
    rectangle.midpoints_connected_to_form_quadrilateral P Q R S →
    ∃ d : ℝ,
      (circle.diameter = d) ∧
      (perimeter_of (quadrilateral P Q R S) = 2 * d) :=
by {
  sorry
}

end perimeter_of_inscribed_quadrilateral_eq_twice_diameter_l775_775284


namespace mode_of_dataSet_is_3_l775_775956

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end mode_of_dataSet_is_3_l775_775956


namespace min_top_block_value_l775_775840

-- Define the layers with their respective block counts
def pyramid_structure : ℕ × ℕ × ℕ × ℕ := (10, 4, 2, 1)

-- Define the range of numbers for blocks in the base layer
def layer1_blocks : Finset ℕ := Finset.range 11

-- Define the condition of sum rule for constructing upper layers
def sum_rule (layer1 layer2 layer3 layer4 : List ℕ) : Prop :=
  -- Layer 2 numbers are sums of specific combinations from Layer 1
  layer2 = [1 + 2 + 3 + 4, 5 + 6 + 7 + 8, 2 + 3 + 5 + 6, 4 + 5 + 7 + 8] ∧
  -- Layer 3 numbers are sums of Layer 2 blocks
  layer3 = [layer2.head! + layer2.nthLe 2 sorry, layer2.nthLe 1 sorry + layer2.nthLe 3 sorry] ∧
  -- Layer 4 number is sum of Layer 3 blocks
  layer4 = [layer3.head! + layer3.nthLe 1 sorry]

-- Define the computation of the top block's value based on previous layers
def top_block_value (layer4 : List ℕ) : ℕ :=
  layer4.head!

-- Define the equivalence between given conditions and the minimum top block value
theorem min_top_block_value :
  ∃ layer1 layer2 layer3 layer4,
    layer1 ⊆ layer1_blocks ∧
    layer2.length = 4 ∧ layer3.length = 2 ∧ layer4.length = 1 ∧
    sum_rule layer1 layer2 layer3 layer4 ∧
    top_block_value layer4 = 55 :=
by sorry

end min_top_block_value_l775_775840


namespace irrational_distance_from_grid_point_to_square_vertex_l775_775097

theorem irrational_distance_from_grid_point_to_square_vertex :
  ∀ (m n : ℤ) (x y : ℤ), 
  (x = 0 ∨ x = 1) ∧ (y = 0 ∨ y = 1) → 
  ∃ (i : ℕ), let (dx, dy) := [(x - m)^2 + (y - n)^2, (m - x)^2 + (n - y)^2, (x - m)^2 + (n - y)^2, (m - x)^2 + (y - n)^2] in 
    irrational (real.sqrt (dx i)) :=
begin
  sorry 
end

end irrational_distance_from_grid_point_to_square_vertex_l775_775097


namespace intersection_is_circumcenter_l775_775454

open EuclideanGeometry

-- Define points, lines, and angles in the Euclidean plane
variables {A B C M N O : Point}

-- Assume the conditions provided in the problem
axiom angle_ABC_eq_60 (h : Triangle A B C) : ∠ B = 60
axiom points_M_N (h : Triangle A B C) : OnLine M A B ∧ OnLine N B C
axiom equal_segments (h : Triangle A B C) : dist A M = dist M N ∧ dist M N = dist N C

-- The theorem statement
theorem intersection_is_circumcenter (h : Triangle A B C) 
    (h_angle_ABC : angle_ABC_eq_60 h) 
    (h_points_M_N : points_M_N h)
    (h_equal_segments : equal_segments h) : 
    is_circumcenter O A B C ↔ intersection (line C M) (line A N) = O :=
sorry

end intersection_is_circumcenter_l775_775454


namespace fred_weekly_allowance_l775_775359

-- Given conditions
variables (A : ℝ) -- Fred's weekly allowance
-- Fred spent half of his allowance on the movies
variables (spent : ℝ) -- amount spent on the movies
-- He earned 6 dollars
variables (earned : ℝ) -- amount earned
-- Final amount he ended with
variables (final_amount : ℝ) -- final amount he ended with

-- Definitions for the problem
def spent_amount (A : ℝ) := A / 2
def earned_amount : ℝ := 6
def final_amount_value : ℝ := 14

-- The main proof problem statement
theorem fred_weekly_allowance (A : ℝ) (spent : ℝ) (earned : ℝ) (final_amount : ℝ) :
  spent = spent_amount A ∧ earned = earned_amount ∧ final_amount = final_amount_value →
  A = 16 := 
by
  sorry

end fred_weekly_allowance_l775_775359


namespace spooky_sequence_property_l775_775296

noncomputable def spooky_sequence (a : ℕ → ℝ) : Prop :=
  (a 1 = 1) ∧ ∀ n > 1, (∑ i in finset.range n, (n + 1 - i) * a i.succ < 0) ∧ (∑ i in finset.range n, (n + 1 - i)^2 * a i.succ > 0)

theorem spooky_sequence_property (a : ℕ → ℝ) (h : spooky_sequence a) :
  (2013^3 * a 1 + 2012^3 * a 2 + 2011^3 * a 3 + ... + 2^3 * a 2012 + a 2013 < 12345) :=
sorry

end spooky_sequence_property_l775_775296


namespace inequality_proof_l775_775151

variable (D : Set ℝ) (f : ℝ → ℝ)

theorem inequality_proof
  (hD_pos : ∀ x ∈ D, x > 0)
  (f_pos : ∀ x ∈ D, f x > 0)
  (cond1 : ∀ x1 x2 ∈ D, f (sqrt (x1 * x2)) ≤ sqrt (f x1 * f x2))
  (cond2 : ∀ x1 x2 ∈ D, f x1 ^ 2 + f x2 ^ 2 ≥ 2 * f (sqrt ((x1 ^ 2 + x2 ^ 2) / 2)) ^ 2)
  (x1 x2 : ℝ) (hx1 : x1 ∈ D) (hx2 : x2 ∈ D) :
  f x1 + f x2 ≥ 2 * f ((x1 + x2) / 2) := by
  sorry

end inequality_proof_l775_775151


namespace difference_of_two_numbers_l775_775211

def nat_sum := 22305
def a := ∃ a: ℕ, 5 ∣ a
def is_b (a b: ℕ) := b = a / 10 + 3

theorem difference_of_two_numbers (a b : ℕ) (h : a + b = nat_sum) (h1 : 5 ∣ a) (h2 : is_b a b) : a - b = 14872 :=
by
  sorry

end difference_of_two_numbers_l775_775211


namespace calculation_one_calculation_two_l775_775654

-- Proof problem 1
theorem calculation_one :
  4 * (sqrt 5 - 4)^4 + 3 * (sqrt 5 - 4)^3 + 2^(-2) * (9/4)^(-1/2) - (0.01)^(1/2) = 826 + 1/15 := 
by 
  sorry

-- Proof problem 2
theorem calculation_two (a : ℝ) (ha : a ≠ 0) :
  (3 * a^(9/2) * sqrt(a^(-3))) / (sqrt((a^(-7/3)) * (a^(13/3)))) = 3 * a^3 := 
by 
  sorry

end calculation_one_calculation_two_l775_775654


namespace installment_value_approx_l775_775959

noncomputable def calculate_installment_value (tv_price : ℝ) (n_installments: ℕ) (interest_rate : ℝ) (last_installment_value : ℝ) : ℝ :=
  let n := n_installments - 1
  let interest := interest_rate / 100
  let sum_natural_numbers := (n * (n + 1)) / 2 
  let I := (tv_price * interest * sum_natural_numbers) / (12 * n)
  let total_paid := tv_price + I
  let excluded_last := total_paid - last_installment_value
  excluded_last / n

theorem installment_value_approx :
  calculate_installment_value 10000 20 6 9000 ≈ 55.40 := 
by
  sorry

end installment_value_approx_l775_775959


namespace positive_integer_solution_equiv_nonnegative_integer_solution_l775_775517

theorem positive_integer_solution_equiv_nonnegative_integer_solution
    (n a : ℕ) (h_pos_a : 0 < a) :
    (∃ x : Fin n → ℕ, (∀ i, 0 < x i) ∧ (∑ i in Finset.range n, (i + 1) * x ⟨i, by { simp [Fin.is_lt] }⟩) = a) ↔
    (∃ y : Fin n → ℕ, (∀ i, 0 ≤ y i) ∧ (∑ i in Finset.range n, (i + 1) * y ⟨i, by { simp [Fin.is_lt] }⟩) = a - (n * (n + 1)) / 2) :=
sorry

end positive_integer_solution_equiv_nonnegative_integer_solution_l775_775517


namespace f_half_and_minus_half_l775_775779

noncomputable def f (x : ℝ) : ℝ :=
  1 - x + Real.log (1 - x) / Real.log 2 - Real.log (1 + x) / Real.log 2

theorem f_half_and_minus_half :
  f (1 / 2) + f (-1 / 2) = 2 := by
  sorry

end f_half_and_minus_half_l775_775779


namespace solution_to_inequality_l775_775340

theorem solution_to_inequality (x : ℝ) : 
  (∃ x ∈ set.Icc (-∞ : ℝ) (∞ : ℝ), x ∈ (set.Ioo (-2 : ℝ) (1 : ℝ)) ∧ x ≠ 1) → 
  ∀ x ∈ set.Icc (-2 : ℝ) (4/3 : ℝ), (3 * x - 4) * (x + 2) / (x - 1) ≤ 0 := 
by
  sorry

end solution_to_inequality_l775_775340


namespace greetingCards_l775_775607

variable (n : ℕ) (students : Fin n → Type)
variable (envelopes : ∀ s : Fin n, Fin (n - 1) → Fin n)
variable (cards : ∀ s : Fin n, ℕ)

theorem greetingCards : (∀ s : Fin n, cards s ≥ 1) →
  (∀ (s : Fin n) (r : Fin n), ∃ (envelope : Fin (n - 1)), envelopes s envelope = r) →
  (∀ (k : ℕ) (P : Fin k → Fin n),
    (P k - 1 ≠ P 0) ∧ (∀ i, i < k - 1 → P i ≠ P (i + 1)) →
    cards (P 0) = cards (P (k - 1))) →
  ∀ s : Fin n, cards s > 0 ∧ (∀ (k : ℕ) (P : Fin k → Fin n),
    (P k - 1 ≠ P 0) ∧ (∀ i, i < k - 1 → P i ≠ P (i + 1)) →
    cards (P 0) = cards (P (k - 1))) :=
by
  intros h1 h2 h3
  sorry

end greetingCards_l775_775607


namespace distinct_graphs_l775_775241

theorem distinct_graphs :
  let eq1 := λ x, x^2 - 3,
      eq2 := λ x, if x ≠ 3 then (x^3 - 27) / (x - 3) else 0,
      eq3 := λ x, if x ≠ 3 then (x^3 - 27) / (x - 3) else 0
  in ∀ x : ℝ, (eq1 x ≠ eq2 x) ∧ (eq1 x ≠ eq3 x) ∧ (eq2 x ≠ eq3 x) := by
  sorry

end distinct_graphs_l775_775241


namespace three_digit_number_increase_l775_775653

theorem three_digit_number_increase (n : ℕ) (h1 : 100 ≤ n) (h2 : n ≤ 999) :
  (n * 1001 / n) = 1001 :=
by
  sorry

end three_digit_number_increase_l775_775653


namespace solve_fraction_identity_l775_775186

theorem solve_fraction_identity (x : ℝ) (hx : (x + 5) / (x - 3) = 4) : x = 17 / 3 :=
by
  sorry

end solve_fraction_identity_l775_775186


namespace find_a_l775_775787

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) (h3 : a + a^2 = 12) : a = 3 :=
by sorry

end find_a_l775_775787


namespace min_operations_to_positives_l775_775734

theorem min_operations_to_positives (n : ℕ) (h : n ≠ 0) (f : Fin n → ℝ) (hf : ∀ i, f i ≠ 0) :
  ∃ m, (∀ k (hk : k = ⌊n / 2⌋ + (n % 2)), m = k) :=
by
  sorry

end min_operations_to_positives_l775_775734


namespace proof_of_length_QR_l775_775603

def length_QR (edge_length : ℝ) : ℝ :=
  let diagonal_length := real.sqrt (edge_length * edge_length + edge_length * edge_length)
  let QS := diagonal_length / 2
  real.sqrt (QS * QS  + edge_length * edge_length)

theorem proof_of_length_QR :
  length_QR 2 = real.sqrt 6 :=
by
  sorry

end proof_of_length_QR_l775_775603


namespace average_percentage_l775_775823

theorem average_percentage (n1 n2 : ℕ) (avg1 avg2 : ℕ) (total_students : ℕ) (total_average : ℕ) :
  n1 = 15 → avg1 = 75 → n2 = 10 → avg2 = 95 → total_students = 25 → total_average = 83 → 
  (n1 * avg1 + n2 * avg2) / total_students = total_average :=
by 
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end average_percentage_l775_775823


namespace how_many_oranges_put_back_l775_775302

variables (A O x : ℕ)

-- Conditions: prices and initial selection.
def price_apple (A : ℕ) : ℕ := 40 * A
def price_orange (O : ℕ) : ℕ := 60 * O
def total_fruit := 20
def average_price_initial : ℕ := 56 -- Average price in cents

-- Conditions: equation from initial average price.
def total_initial_cost := total_fruit * average_price_initial
axiom initial_cost_eq : price_apple A + price_orange O = total_initial_cost
axiom total_fruit_eq : A + O = total_fruit

-- New conditions: desired average price and number of fruits
def average_price_new : ℕ := 52 -- Average price in cents
axiom new_cost_eq : price_apple A + price_orange (O - x) = (total_fruit - x) * average_price_new

-- The statement to be proven
theorem how_many_oranges_put_back : 40 * A + 60 * (O - 10) = (total_fruit - 10) * 52 → x = 10 :=
sorry

end how_many_oranges_put_back_l775_775302


namespace age_of_B_l775_775968

theorem age_of_B (A B C : ℕ) (h1 : A + B + C = 90)
                  (h2 : (A - 10) = (B - 10) / 2)
                  (h3 : (B - 10) / 2 = (C - 10) / 3) : 
                  B = 30 :=
by sorry

end age_of_B_l775_775968


namespace sum_partial_fractions_series_l775_775329

theorem sum_partial_fractions_series :
  (∑ n in finset.range 9, 1 / ((n + 2) * (n + 3): ℚ)) = 9 / 22 := sorry

end sum_partial_fractions_series_l775_775329


namespace problem_l775_775056

noncomputable def hyperbola_eccentricity : ℝ → ℝ → ℝ := 
  λ a b, (real.sqrt (a^2 + b^2)) / a

theorem problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
  (H : |(3 * b * real.sqrt (a^2 + b^2) / (real.sqrt 3 * b - 3 * a))|  =
       |3 * real.sqrt (a^2 + b^2)|) :
  hyperbola_eccentricity a b = 2 * real.sqrt 3 / 3 :=
sorry

end problem_l775_775056


namespace johns_final_amount_l775_775462

def initial_amount : ℝ := 45.7
def deposit_amount : ℝ := 18.6
def withdrawal_amount : ℝ := 20.5

theorem johns_final_amount : initial_amount + deposit_amount - withdrawal_amount = 43.8 :=
by
  sorry

end johns_final_amount_l775_775462


namespace joan_exam_time_difference_l775_775126

theorem joan_exam_time_difference :
  (let english_questions := 30
       math_questions := 15
       english_time_hours := 1
       math_time_hours := 1.5
       english_time_minutes := english_time_hours * 60
       math_time_minutes := math_time_hours * 60
       time_per_english_question := english_time_minutes / english_questions
       time_per_math_question := math_time_minutes / math_questions
    in time_per_math_question - time_per_english_question = 4) :=
by
  sorry

end joan_exam_time_difference_l775_775126


namespace acute_triangle_angle_C_l775_775435

theorem acute_triangle_angle_C (ABC : Triangle)
  (A B C D E F O : Point)
  (h_acutely : isAcuteTriangle ABC)
  (h_AD : isAltitude A D)
  (h_BE : isMedian B E)
  (h_CF : isAngleBisector C F)
  (h_intersect : areConcurrent [AD, BE, CF] O)
  (h_OE_2OC : distance O E = 2 * distance O C) :
  angle ABC. ∠ C = 2 * arccos (1 / 7) :=
sorry

end acute_triangle_angle_C_l775_775435


namespace length_AB_is_two_l775_775751

open EuclideanGeometry

noncomputable def parallelogram_ABCD (A B C D : Point) : Prop :=
  parallelogram A B C D

theorem length_AB_is_two (A B C D : Point) (BD_norm : ∥B - D∥ = 2)
  (dot_product_condition : 2 * (D - A) • (B - A) = ∥C - B∥^2)
  (h_para : parallelogram_ABCD A B C D) :
  ∥B - A∥ = 2 := 
sorry

end length_AB_is_two_l775_775751


namespace f_2023_l775_775041

noncomputable def f : ℝ → ℝ := sorry

axiom f_domain : ∀ x : ℝ, x ∈ set.univ
axiom f_prop1 : ∀ x : ℝ, f(x) + f(4 - x) = 0
axiom f_prop2 : ∀ x : ℝ, f(-x) = -f(x)
axiom f_defn : ∀ x : ℝ, x ∈ set.Icc 0 2 → f(x) = -x^2 + 2x

theorem f_2023 : f 2023 = -1 := sorry

end f_2023_l775_775041


namespace area_triang_ABD_l775_775858

-- Define the convex quadrilateral
variables {A B C D E F : Point} 

-- Assume E and F are midpoints
variables (hE : midpoint E B C) (hF : midpoint F C D)

-- Define the conditions: four triangles' areas as consecutive natural numbers
variables {n : ℕ} (hAreas : area A E F = n ∧ area A F D = n + 1 ∧ area A B E = n + 2 ∧ area E F C = n + 3)

-- Statement: The greatest possible area of triangle ABD is 6
theorem area_triang_ABD (hConvex : convex_quadrilateral A B C D) : area A B D = 6 :=
by {
  sorry
}

end area_triang_ABD_l775_775858


namespace number_of_correct_propositions_l775_775897

section ClosedGeometricSequence

variable (a : ℕ → ℝ) (q : ℝ)
variable [CompleteLattice ℝ]

def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  ∀ n m : ℕ, a (n + m) = a n * q ^ m

def is_closed_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop := 
  is_geometric_sequence a q ∧ ∀ m n : ℕ, ∃ k : ℕ, a (m + n) = a k

variables (a₁ a₂ : ℕ → ℝ) (q₁ q₂ : ℝ)

def prop1 : Prop := 
  a 1 = 3 ∧ q = 2 → ¬ is_closed_geometric_sequence a q

def prop2 : Prop := 
  a 1 = 1 / 2 ∧ q = 2 → is_closed_geometric_sequence a q

def prop3 : Prop := 
  is_closed_geometric_sequence a₁ q₁ ∧ is_closed_geometric_sequence a₂ q₂ → ¬ is_closed_geometric_sequence (λ n, a₁ n + a₂ n) (q₁ + q₂)

def prop4 : Prop := 
  ¬ ∃ a : ℕ → ℝ, ∃ q : ℝ, is_closed_geometric_sequence a q ∧ is_closed_geometric_sequence (λ n, (a n)^2) q

theorem number_of_correct_propositions : 
  (prop1 a q) + (prop2 a q) + (prop3 a₁ a₂ q₁ q₂) + (prop4 a q) = 1 :=
sorry

end ClosedGeometricSequence

end number_of_correct_propositions_l775_775897


namespace weight_around_59_3_l775_775171

noncomputable def weight_at_height (height: ℝ) : ℝ := 0.75 * height - 68.2

theorem weight_around_59_3 (x : ℝ) (h : x = 170) : abs (weight_at_height x - 59.3) < 1 :=
by
  sorry

end weight_around_59_3_l775_775171


namespace initial_candy_bobby_l775_775305

-- Definitions given conditions
def initial_candy (x : ℕ) : Prop :=
  (x + 42 = 70)

-- Theorem statement
theorem initial_candy_bobby : ∃ x : ℕ, initial_candy x ∧ x = 28 :=
by {
  sorry
}

end initial_candy_bobby_l775_775305


namespace mark_age_in_5_years_l775_775155

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end mark_age_in_5_years_l775_775155


namespace customers_in_other_countries_l775_775614

-- Given 
def total_customers : ℕ := 7422
def customers_in_us : ℕ := 723

-- To Prove
theorem customers_in_other_countries : (total_customers - customers_in_us) = 6699 := 
by
  sorry

end customers_in_other_countries_l775_775614


namespace sum_of_terms_k0_b3_pneg4_general_term_k1_b0_p0_closed_sequence_constraints_l775_775756

-- Problem 1: Sum of Terms for k=0, b=3, p=-4
theorem sum_of_terms_k0_b3_pneg4 (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : ∀ n : ℕ, n > 0 → (3 * (a 0 + a n) - 4 = 2 * (S n))) :
  (S : ℕ → ℕ) (n : ℕ) (S n = (3^n - 1) / 2) := sorry

-- Problem 2: General Term given a3 = 3 and a9 = 15, for k=1, b=0, p=0
theorem general_term_k1_b0_p0 (a : ℕ → ℕ)
  (h : ∀ n : ℕ, n > 0 → (n * (a 0 + a n) = 2 * (S n)) ∧ a 3 = 3 ∧ a 9 = 15):
  (a n = 2 * n - 3) := sorry

-- Problem 3: Closed Sequence, for k=1, b=0, p=0, and a2 - a1 = 2
theorem closed_sequence_constraints (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : ∀ n : ℕ, n > 0 → (n * (a 0 + a n) = 2 * (S n)))
  (h2 : a 2 - a 1 = 2)
  (h3 : ∀ n : ℕ, n > 0 → (S n ≠ 0))
  (h4 : ∃ (a1 : ℕ), a1 ∈ {4, 6, 8, 10})
  (ineq : ∀ n : ℕ, 1 / 12 < ∑ i in range (n+1), 1 / S i < 11 / 18) :
  (a1 = 4 ∨ a1 = 6 ∨ a1 = 8 ∨ a1 = 10) := sorry

end sum_of_terms_k0_b3_pneg4_general_term_k1_b0_p0_closed_sequence_constraints_l775_775756


namespace least_number_of_square_tiles_l775_775583

theorem least_number_of_square_tiles (length : ℝ) (breadth : ℝ) (h₁ : length = 15.17) (h₂ : breadth = 9.02) :
  ∃ n : ℕ, n = 814 ∧
    ∀ t : ℝ, t = (length * 100) / 41 ∧ t = (breadth * 100) / 41 → n = (t * t) :=
by
  have lm := length * 100
  have bm := breadth * 100
  have gcd := Int.gcd_nat (Real.toNat lm) (Real.toNat bm)
  have tile_side := gcd
  have tiles_length := lm / tile_side
  have tiles_breadth := bm / tile_side
  have total_tiles := tiles_length * tiles_breadth
  use total_tiles
  exact sorry

end least_number_of_square_tiles_l775_775583


namespace inverse_f_1_l775_775399

noncomputable def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

theorem inverse_f_1 : ∃ x : ℝ, f x = 1 ∧ x = 2 := by
sorry

end inverse_f_1_l775_775399


namespace find_single_point_domain_l775_775902

def f1 (x : ℝ) := real.sqrt (2 - x)

def fn : ℕ → (ℝ → ℝ)
| 1 := f1
| n := λ x, fn (n - 1) (real.sqrt ((n + 1)^2 - x))

theorem find_single_point_domain : ∃ c, ∀ x, fn 2 x = real.sqrt 2 - 0 ↔ x = 9 :=
by sorry

end find_single_point_domain_l775_775902


namespace original_stone_145_l775_775358

theorem original_stone_145 :
  ∃ k : ℕ, (k < 15 + 1) ∧ (145 ≡ k [MOD 26]) :=
begin
  use 2,
  split,
  {
    -- verify k < 15 + 1
    linarith,
  },
  {
    -- verify 145 ≡ 2 [MOD 26]
    norm_num,
  }
end

end original_stone_145_l775_775358


namespace happy_dictionary_problem_l775_775436

def smallest_positive_integer : ℕ := 1
def largest_negative_integer : ℤ := -1
def smallest_abs_rational : ℚ := 0

theorem happy_dictionary_problem : 
  smallest_positive_integer - largest_negative_integer + smallest_abs_rational = 2 := 
by
  sorry

end happy_dictionary_problem_l775_775436


namespace integer_solutions_count_l775_775807

theorem integer_solutions_count :
  let y := Int
  let inequality1 := ∀ y, -4 * y ≥ y + 9
  let inequality2 := ∀ y, -3 * y ≤ 15
  let inequality3 := ∀ y, -5 * y ≥ 2 * y + 20
  let interval := [-5, -4, -3]
in
  ∃ y, y ∈ interval ∧ inequality1 y ∧ inequality2 y ∧ inequality3 y :=
sorry

end integer_solutions_count_l775_775807


namespace mode_of_dataSet_is_3_l775_775957

-- Define the data set
def dataSet : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

-- Define what it means to be the mode of a list
def is_mode (l : List ℕ) (n : ℕ) : Prop :=
  ∀ m, l.count n ≥ l.count m

-- Prove the mode of the data set
theorem mode_of_dataSet_is_3 : is_mode dataSet 3 :=
by
  sorry

end mode_of_dataSet_is_3_l775_775957


namespace vector_projection_l775_775365

open Function

variables (a e : ℝ^3) (ha : ∥a∥ = 6) (he : ∥e∥ = 1) (angle_ae : real.angle a e = real.pi * (2 / 3))

noncomputable def projection_of_a_onto_e : ℝ := ∥a∥ * real.cos (2 * real.pi / 3)

theorem vector_projection (ha : ∥a∥ = 6) (he : ∥e∥ = 1) (angle_ae : real.angle a e = real.pi * (2 / 3)) :
  projection_of_a_onto_e a e = -3 :=
by {
  unfold projection_of_a_onto_e,
  rw [ha, real.angle_eq_iff_eq_or_eq_neg_pi],
  simp,
  sorry
}

end vector_projection_l775_775365


namespace find_derivative_at_1_l775_775741

-- Given condition definition
def f (x : ℝ) : ℝ := x^3 + 2 * x * (f 1) + x

-- The problem statement we need to prove
theorem find_derivative_at_1 : (deriv f) 1 = -4 := 
by {
  -- Proof steps would go here, but for now we use 'sorry' to denote it is to be filled
  sorry
}

end find_derivative_at_1_l775_775741


namespace mixture_volume_l775_775563

def weight_a (Va : ℝ) : ℝ := Va * 900
def weight_b (Vb : ℝ) : ℝ := Vb * 800
def total_weight (Va Vb : ℝ) : ℝ := weight_a Va + weight_b Vb
def given_ratio (Va Vb : ℝ) : Prop := Va / Vb = 3 / 2

theorem mixture_volume (Va Vb : ℝ)
  (h1 : given_ratio Va Vb)
  (h2 : total_weight Va Vb = 3440) : 
  Va + Vb = 4 :=
begin
  sorry
end

end mixture_volume_l775_775563


namespace projection_of_a_in_direction_of_b_l775_775360

theorem projection_of_a_in_direction_of_b :
  let a := (1, -2)
  let b := (3, 4)
  (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -1 :=
by
  let a := (1, -2)
  let b := (3, 4)
  have h_dot_product : a.1 * b.1 + a.2 * b.2 = 3 - 8 := by sorry
  have h_magnitude : Real.sqrt (b.1^2 + b.2^2) = 5 := by sorry
  show (a.1 * b.1 + a.2 * b.2) / Real.sqrt (b.1^2 + b.2^2) = -1, from
    by
      rw [h_dot_product, h_magnitude]
      norm_num

end projection_of_a_in_direction_of_b_l775_775360


namespace sort_athletes_two_rounds_l775_775216

theorem sort_athletes_two_rounds (n : ℕ) (h_n : n > 0) (a : Fin n → Fin n) :
  ∃ b : Fin n → Fin n → Fin n, (∀ i, b 1 i = a i) ∧ ∀ j,  b 2 j = j := sorry

end sort_athletes_two_rounds_l775_775216


namespace second_number_value_l775_775210

theorem second_number_value (A B C : ℝ) 
    (h1 : A + B + C = 98) 
    (h2 : A = (2/3) * B) 
    (h3 : C = (8/5) * B) : 
    B = 30 :=
by 
  sorry

end second_number_value_l775_775210


namespace fill_entire_cistern_l775_775283

theorem fill_entire_cistern (t : ℝ) (part : ℝ) (whole : ℝ)
  (h1 : part = 1 / 11)
  (h2 : t = 3) :
  whole = 11 * t :=
by
  have h3 : whole = 11 * part * t := sorry
  have h4 : 11 * part = 1 := sorry
  rw [h1, h2] at h3
  exact h3

end fill_entire_cistern_l775_775283


namespace vertex_angle_of_cone_l775_775702

theorem vertex_angle_of_cone (α : ℝ) (h_α : 0 < α ∧ α < 2 * π) :
  let vertex_angle := 2 * real.arcsin (α / (2 * real.pi)) in 
  true := by
  sorry

end vertex_angle_of_cone_l775_775702


namespace Lakers_win_in_7_games_l775_775536

-- Variables for probabilities given in the problem
variable (p_Lakers_win : ℚ := 1 / 4) -- Lakers' probability of winning a single game
variable (p_Celtics_win : ℚ := 3 / 4) -- Celtics' probability of winning a single game

-- Probabilities and combinations
def binom (n k : ℕ) : ℕ := Nat.choose n k

noncomputable def probability_Lakers_win_game7 : ℚ :=
  let first_6_games := binom 6 3 * (p_Lakers_win ^ 3) * (p_Celtics_win ^ 3)
  let seventh_game := p_Lakers_win
  first_6_games * seventh_game

theorem Lakers_win_in_7_games : probability_Lakers_win_game7 = 540 / 16384 := by
  sorry

end Lakers_win_in_7_games_l775_775536


namespace sum_of_consecutive_primes_up_to_151_l775_775590

theorem sum_of_consecutive_primes_up_to_151 :
  ∑ k in (Finset.filter Nat.Prime (Finset.range 152)), k = 2427 := 
by
  sorry

end sum_of_consecutive_primes_up_to_151_l775_775590


namespace avg_speed_while_climbing_l775_775162

-- Definitions for conditions
def totalClimbTime : ℝ := 4
def restBreaks : ℝ := 0.5
def descentTime : ℝ := 2
def avgSpeedWholeJourney : ℝ := 1.5
def totalDistance : ℝ := avgSpeedWholeJourney * (totalClimbTime + descentTime)

-- The question: Prove Natasha's average speed while climbing to the top, excluding the rest breaks duration.
theorem avg_speed_while_climbing :
  (totalDistance / 2) / (totalClimbTime - restBreaks) = 1.29 := 
sorry

end avg_speed_while_climbing_l775_775162


namespace fourth_term_geometric_sequence_l775_775200

theorem fourth_term_geometric_sequence :
  let a := (6: ℝ)^(1/2)
  let b := (6: ℝ)^(1/6)
  let c := (6: ℝ)^(1/12)
  b = a * r ∧ c = a * r^2 → (a * r^3) = 1 := 
by
  sorry

end fourth_term_geometric_sequence_l775_775200


namespace expected_winnings_value_l775_775160

-- Conditions
def is_prime (n : ℕ) : Prop := n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
def is_perfect_square (n : ℕ) : Prop := n = 1 ∨ n = 4
def winnings (n : ℕ) : ℕ :=
  if is_prime n then n
  else if is_perfect_square n then 2 * n
  else 0

noncomputable def expected_value : ℚ :=
  (4 / 8 : ℚ) * (2 + 3 + 5 + 7) + (2 / 8 : ℚ) * (2 * 1 + 2 * 4) + (2 / 8 : ℚ) * 0

-- Statement
theorem expected_winnings_value : expected_value = 11 := 
sorry

end expected_winnings_value_l775_775160


namespace tangent_line_at_one_l775_775549

noncomputable def f (x : ℝ) : ℝ := x^2 + 1/x

theorem tangent_line_at_one :
  let f' := (λ x, 2*x - 1/x^2) in
  let x := 1 in
  let y := f x in
  let slope := f' x in
  ∃ (c1 c2 : ℝ), y = c1 + slope*(x - c2) ∧ c1 = 1 ∧ c2 = 0 :=
begin
  let f' := (λ x, 2*x - 1/x^2),
  let x := 1,
  let y := f x,
  let slope := f' x,
  use [2, 1],
  split,
  { rw [f, f'],
    dsimp,
    sorry, -- This is where the proof would go
  },
  split;
  refl, -- trivial facts
end

end tangent_line_at_one_l775_775549


namespace problem_statement_l775_775848

variable (a : ℕ → ℝ)
variable (q : ℝ)
variable (S : ℕ → ℝ)

-- Conditions
def increasing_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def condition1 := a 1 = 1
def condition2 := (a 3 + a 4) / (a 1 + a 2) = 4
def increasing := q > 0

-- Definition of S_n
def sum_geom (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = a 1 * (1 - q ^ n) / (1 - q)

theorem problem_statement (a : ℕ → ℝ) (q : ℝ) (S : ℕ → ℝ) 
  (h_geom : increasing_geometric_sequence a q) 
  (h_condition1 : condition1 a) 
  (h_condition2 : condition2 a) 
  (h_increasing : increasing q)
  (h_sum_geom : sum_geom a q S) : 
  S 5 = 31 :=
sorry

end problem_statement_l775_775848


namespace sum_g_values_l775_775909

noncomputable def g (x : ℝ) : ℝ :=
if x > 3 then x^2 - 1 else
if x >= -3 then 3 * x + 2 else 4

theorem sum_g_values : g (-4) + g 0 + g 4 = 21 :=
by
  sorry

end sum_g_values_l775_775909


namespace part1_concyclic_points_part2_tangent_line_BD_l775_775029

-- Define the geometric setup and hypotheses
variables {A B C D E H N P : Type}
variables [triangle ABC] [circle O ABC] [tangent O A BC D]
variables [reflect A BC E] [perpendicular A H BE H] [midpoint N A H]
variables [intersect_circle BN O P]

-- Part 1: Prove that points A, N, M, P are concyclic
theorem part1_concyclic_points
    (circumcircle : circle O ABC)
    (angle_BAC : angle BAC = 90)
    (angle_ABC_lt_angle_ACB : angle ABC < angle ACB)
    (tangent_at_A : tangent O A BC D)
    (reflection_E : reflect A BC E)
    (perpendicular_AH_BEH : perpendicular A H BE H)
    (midpoint_N_AH : midpoint N A H)
    (intersect_BN_P : intersect_circle BN O P) :
    cyclic A N M P :=
sorry

-- Part 2: Prove that line BD is tangent to the circumcircle of triangle ADP
theorem part2_tangent_line_BD
    (circumcircle : circle O ABC)
    (angle_BAC : angle BAC = 90)
    (angle_ABC_lt_angle_ACB : angle ABC < angle ACB)
    (tangent_at_A : tangent O A BC D)
    (reflection_E : reflect A BC E)
    (perpendicular_AH_BEH : perpendicular A H BE H)
    (midpoint_N_AH : midpoint N A H)
    (intersect_BN_P : intersect_circle BN O P) :
    tangent BD (circumcircle_triangle ADP) :=
sorry

end part1_concyclic_points_part2_tangent_line_BD_l775_775029


namespace jane_change_l775_775876

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l775_775876


namespace isosceles_trapezoid_from_pentagon_l775_775683

theorem isosceles_trapezoid_from_pentagon (ABCD : Type) (A B C D : ABCD) 
  (AD BC AC : ℝ) (isosceles_trapezoid : IsoscelesTrapezoid ABCD A B C D AD BC)
  (h1 : AD > BC)
  (isosceles_triangles : IsoscelesTriangle ABCD A B C ∧ IsoscelesTriangle ABCD A D C) :
  ∃ P : Type, RegularPentagon P ∧ SubType P ABCD :=
by
  sorry

end isosceles_trapezoid_from_pentagon_l775_775683


namespace total_black_nodes_in_first_20_rows_eq_10945_l775_775446

def black_nodes : ℕ → ℕ
| 0     := 0  -- No solid circles in the 1st row (problem starts with 2nd row in essence)
| 1     := 1  -- 1 solid circle in the 2nd row
| (n+2) := black_nodes (n+1) + black_nodes n

def solid_circles_in_first_20_rows : ℕ :=
  (List.range 20).sum black_nodes

theorem total_black_nodes_in_first_20_rows_eq_10945 :
  solid_circles_in_first_20_rows = 10945 :=
by
  sorry

end total_black_nodes_in_first_20_rows_eq_10945_l775_775446


namespace product_of_neg_ints_abs3_to_5_l775_775960

theorem product_of_neg_ints_abs3_to_5 : 
    ∏ (x : ℤ) in ({-3, -4, -5} : Finset ℤ), x = -60 := 
by 
  sorry

end product_of_neg_ints_abs3_to_5_l775_775960


namespace equation_solution_l775_775936

theorem equation_solution (x : ℝ) : 
  (x - 3)^4 = 16 → x = 5 :=
by
  sorry

end equation_solution_l775_775936


namespace fg_sum_at_2_l775_775901

noncomputable def f (x : ℚ) : ℚ := (5 * x^3 + 4 * x^2 - 2 * x + 3) / (x^3 - 2 * x^2 + 3 * x + 1)
noncomputable def g (x : ℚ) : ℚ := x^2 - 2

theorem fg_sum_at_2 : f (g 2) + g (f 2) = 468 / 7 := by
  sorry

end fg_sum_at_2_l775_775901


namespace min_tests_to_determine_sceptervirus_l775_775974

-- Define the problem of finding the smallest number of tests required
-- to determine the presence of the sceptervirus given the conditions.
theorem min_tests_to_determine_sceptervirus (n : ℕ) : 
  ∃ (k : ℕ), k = n ∧
  (∀ (S : finset (fin n)), (∀ t, t ∈ S → (t > 0) ∨ (t = 0) ∨ (t < 0)) → 
    (∃ (x : fin n → ℤ), 
      (∀ r, x r ∈ {-1, 0, 1}) ∧ 
      (∀ i, (∑ r in S, x r) = 0))) :=
begin
  sorry
end

end min_tests_to_determine_sceptervirus_l775_775974


namespace expression_in_terms_of_p_and_q_l775_775417

theorem expression_in_terms_of_p_and_q (x : ℝ) :
  let p := (1 - Real.cos x) * (1 + Real.sin x)
  let q := (1 + Real.cos x) * (1 - Real.sin x)
  (Real.cos x ^ 2 - Real.cos x ^ 4 - Real.sin (2 * x) + 2) = p * q - (p + q) :=
by
  sorry

end expression_in_terms_of_p_and_q_l775_775417


namespace complex_point_quadrant_l775_775021

theorem complex_point_quadrant 
  (i : Complex) 
  (h_i_unit : i = Complex.I) : 
  (Complex.re ((i - 3) / (1 + i)) < 0) ∧ (Complex.im ((i - 3) / (1 + i)) > 0) :=
by {
  sorry
}

end complex_point_quadrant_l775_775021


namespace determine_f_101_l775_775132

theorem determine_f_101 (f : ℕ → ℕ) (h : ∀ m n : ℕ, m * n + 1 ∣ f m * f n + 1) : 
  ∃ k : ℕ, k % 2 = 1 ∧ f 101 = 101 ^ k :=
sorry

end determine_f_101_l775_775132


namespace max_mogs_l775_775304

theorem max_mogs : ∃ x y z : ℕ, 1 ≤ x ∧ 1 ≤ y ∧ 1 ≤ z ∧ 3 * x + 4 * y + 8 * z = 100 ∧ z = 10 :=
by
  sorry

end max_mogs_l775_775304


namespace meeting_successful_probability_is_correct_l775_775625

/-- Define the structure of the meeting problem with associates and manager's arrival constraints -/
structure MeetingScenario where
  (x : ℝ) (y : ℝ) (z : ℝ)
  (hx_range : 0 ≤ x ∧ x ≤ 4)
  (hy_range : 0 ≤ y ∧ y ≤ 4)
  (hz_range : 0 ≤ z ∧ z ≤ 4)
  (hx_manager : x ≤ z ∧ z ≤ x + 0.5)
  (hy_manager : y ≤ z ∧ z ≤ y + 0.5)
  (h_associates : |x - y| ≤ 1.5)

/-- Probability that the meeting successfully occurs -/
def probability_of_successful_meeting (scenario : MeetingScenario) : ℝ := 0.1

-- The main statement to be proved
theorem meeting_successful_probability_is_correct :
  ∀ (scenario : MeetingScenario), probability_of_successful_meeting(scenario) = 0.1 := by
  sorry

end meeting_successful_probability_is_correct_l775_775625


namespace find_donna_bananas_l775_775667

-- Given conditions
variables (total_bananas : ℕ) (dawn_bananas lydia_bananas donna_bananas : ℕ)
variables (h1 : total_bananas = 350) (h2 : dawn_bananas = lydia_bananas + 70) (h3 : lydia_bananas = 90)

-- Required to prove
theorem find_donna_bananas : donna_bananas = 100 :=
by
  calc 
    total_bananas : 350 := h1
    dawn_bananas : lydia_bananas + 70 := h2
    lydia_bananas : 90 := h3
    donna_bananas = total_bananas - (dawn_bananas + lydia_bananas) : sorry

end find_donna_bananas_l775_775667


namespace exists_N_binary_representation_l775_775465

theorem exists_N_binary_representation (n p : ℕ) (h_composite : ∃ d : ℕ, d > 1 ∧ d < n ∧ n % d = 0) (h_proper_divisor : p > 0 ∧ p < n ∧ n % p = 0) :
  ∃ N : ℕ, ((1 + 2^p + 2^(n-p)) * N) % 2^n = 1 % 2^n :=
by
  sorry

end exists_N_binary_representation_l775_775465


namespace max_m_l775_775439

open Real

/-- Given points A(0,4) and B(2,0), and a circle with equation x^2 + y^2 + 2x + 4y + 5 = m,
    if there is a point P on the circle such that ∠APB is a right angle,
    then the maximum value of the real number m is 45. -/
theorem max_m (m : ℝ) : 
  (∃ P : ℝ × ℝ, (P.1^2 + P.2^2 + 2 * P.1 + 4 * P.2 + 5 = m) ∧ 
    ∃ A : ℝ × ℝ, A = (0, 4) ∧
    ∃ B : ℝ × ℝ, B = (2, 0) ∧
    ∃ P' : ℝ × ℝ, P' = P ∧
    angle (A, P', B) = π / 2) → 
  m ≤ 45 := 
sorry

end max_m_l775_775439


namespace find_common_ratio_l775_775136

theorem find_common_ratio (S : ℕ → ℝ) (a : ℕ → ℝ) (q : ℝ)
  (h1 : S 3 = a 1 + a 2 + a 3)
  (h2 : S 2 = a 1 + a 2)
  (h3 : 3 * S 3 = a 4 - 2)
  (h4 : 3 * S 2 = a 3 - 2)
  (h5 : ∀ n : ℕ, a (n+1) = q * a n) : q = 4 := sorry

end find_common_ratio_l775_775136


namespace region_passes_four_quadrants_l775_775828

theorem region_passes_four_quadrants
  (x y λ : Real)
  (h1 : x ≤ 1)
  (h2 : y ≤ 3)
  (h3 : 2 * x - y + λ - 1 ≥ 0) :
  1 < λ := sorry

end region_passes_four_quadrants_l775_775828


namespace solve_eq1_solve_eq2_l775_775939

theorem solve_eq1 {x : ℝ} : 2 * x^2 - 1 = 49 ↔ x = 5 ∨ x = -5 := 
  sorry

theorem solve_eq2 {x : ℝ} : (x + 3)^3 = 64 ↔ x = 1 := 
  sorry

end solve_eq1_solve_eq2_l775_775939


namespace problem_statement_l775_775735

variable {Ω : Type*} -- Sample space
variables {P : MeasureTheory.Measure Ω} -- Probability measure
variables {A B : Set Ω} -- Events in the sample space

theorem problem_statement (h1 : P A = P (Set.compl A))
                          (h2 : P (Set.compl B ∩ A) / P A > P (B ∩ Set.compl A) / P (Set.compl A)) :
  P (A ∩ Set.compl B) > P (Set.compl A ∩ B) ∧ P (A ∩ B) < P (Set.compl A ∩ Set.compl B) :=
by
  sorry

end problem_statement_l775_775735


namespace remainder_of_polynomial_l775_775695

-- Define the polynomial and the divisor
def f (x : ℝ) := x^3 - 4 * x + 6
def a := -3

-- State the theorem
theorem remainder_of_polynomial :
  f a = -9 := by
  sorry

end remainder_of_polynomial_l775_775695


namespace tangent_line_equation_slope_sum_condition_l775_775391

noncomputable def f (a : ℝ) : ℝ → ℝ := λ x, (1/3) * x^3 - a * x + 2 * a

theorem tangent_line_equation (x : ℝ) (hx : x = 2) (a : ℝ) (ha : a = 1) :
  9 * (2 : ℝ) - 3 * ((1/3) * 2^3 - 1 * 2 + 2) - 10 = 0 := by
  sorry

theorem slope_sum_condition (a : ℝ) (h : ∀ x₀ : ℝ, 
  (x₀ = 0 ∨ x₀ = 3) →
  (x₀ ^ 2 - a - (9 - a) = 1)) :
  a = 4 := by
  sorry

end tangent_line_equation_slope_sum_condition_l775_775391


namespace function_identity_l775_775140

theorem function_identity (f : ℕ+ → ℕ+) (h : ∀ m n : ℕ+, f m + f n ∣ m + n) : ∀ m : ℕ+, f m = m := by
  sorry

end function_identity_l775_775140


namespace sample_proportion_l775_775838

theorem sample_proportion (total_students boys girls sample_size : ℕ)
  (total_eq : total_students = 700)
  (boys_eq : boys = 385)
  (girls_eq : girls = 315)
  (sample_size_eq : sample_size = 60) :
  let boys_in_sample := (sample_size * boys) / total_students in
  let girls_in_sample := (sample_size * girls) / total_students in
  boys_in_sample = 33 ∧ girls_in_sample = 27 :=
by
  sorry

end sample_proportion_l775_775838


namespace flour_ratio_correct_l775_775494

-- Definitions based on conditions
def initial_sugar : ℕ := 13
def initial_flour : ℕ := 25
def initial_baking_soda : ℕ := 35
def initial_cocoa_powder : ℕ := 60

def added_sugar : ℕ := 12
def added_flour : ℕ := 8
def added_cocoa_powder : ℕ := 15

-- Calculate remaining ingredients
def remaining_flour : ℕ := initial_flour - added_flour
def remaining_sugar : ℕ := initial_sugar - added_sugar
def remaining_cocoa_powder : ℕ := initial_cocoa_powder - added_cocoa_powder

-- Calculate ratio
def total_remaining_sugar_and_cocoa : ℕ := remaining_sugar + remaining_cocoa_powder
def flour_to_sugar_cocoa_ratio : ℕ × ℕ := (remaining_flour, total_remaining_sugar_and_cocoa)

-- Proposition stating the desired ratio
theorem flour_ratio_correct : flour_to_sugar_cocoa_ratio = (17, 46) := by
  sorry

end flour_ratio_correct_l775_775494


namespace ineq_five_times_x_minus_six_gt_one_l775_775334

variable {x : ℝ}

theorem ineq_five_times_x_minus_six_gt_one (x : ℝ) : 5 * x - 6 > 1 :=
sorry

end ineq_five_times_x_minus_six_gt_one_l775_775334


namespace walking_ring_width_l775_775291

theorem walking_ring_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 20 * Real.pi) :
  r₁ - r₂ = 10 :=
by
  sorry

end walking_ring_width_l775_775291


namespace max_value_of_expression_l775_775883

theorem max_value_of_expression (A M C : ℕ) (h1 : A + M + C = 10) : 
  A * M * C + A * M + M * C + C * A ≤ 69 := 
begin
  sorry
end

end max_value_of_expression_l775_775883


namespace terry_spent_total_l775_775532

def total_amount_spent (monday_spent tuesday_spent wednesday_spent : ℕ) : ℕ := 
  monday_spent + tuesday_spent + wednesday_spent

theorem terry_spent_total 
  (monday_spent : ℕ)
  (hmonday : monday_spent = 6)
  (tuesday_spent : ℕ)
  (htuesday : tuesday_spent = 2 * monday_spent)
  (wednesday_spent : ℕ)
  (hwednesday : wednesday_spent = 2 * (monday_spent + tuesday_spent)) :
  total_amount_spent monday_spent tuesday_spent wednesday_spent = 54 :=
by
  sorry

end terry_spent_total_l775_775532


namespace triple_n_8_l775_775354

def sum_proper_divisors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, d < n) (Finset.divisors n)).sum

def triple_n (n : ℕ) : ℕ :=
  sum_proper_divisors (sum_proper_divisors (sum_proper_divisors n))

theorem triple_n_8 : triple_n 8 = 0 :=
  by
    sorry

end triple_n_8_l775_775354


namespace sum_of_squares_l775_775348

theorem sum_of_squares (n : ℕ) : 
  ∑ k in Finset.range (n + 1), k ^ 2 = n * (n + 1) * (2 * n + 1) / 6 := 
by
  sorry

end sum_of_squares_l775_775348


namespace sum_of_distinct_integers_l775_775475

noncomputable def a : ℤ := 11
noncomputable def b : ℤ := 9
noncomputable def c : ℤ := 4
noncomputable def d : ℤ := 2
noncomputable def e : ℤ := 1

def condition : Prop := (6 - a) * (6 - b) * (6 - c) * (6 - d) * (6 - e) = 120
def distinct_integers : Prop := a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e

theorem sum_of_distinct_integers (h1 : condition) (h2 : distinct_integers) : a + b + c + d + e = 27 :=
by
  sorry

end sum_of_distinct_integers_l775_775475


namespace arithmetic_sequence_k_l775_775440

theorem arithmetic_sequence_k :
  ∀ (a : ℕ → ℤ) (d : ℤ) (k : ℕ),
  d ≠ 0 →
  (∀ n : ℕ, a n = a 0 + n * d) →
  a 0 = 0 →
  a k = a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 →
  k = 22 :=
by
  intros a d k hdnz h_arith h_a1_zero h_ak_sum
  sorry

end arithmetic_sequence_k_l775_775440


namespace pythagorean_relationship_l775_775632

theorem pythagorean_relationship (a b c : ℝ) (h : c^2 = a^2 + b^2) : c^2 = a^2 + b^2 :=
by
  sorry

end pythagorean_relationship_l775_775632


namespace find_cost_price_l775_775544

/-- 
Given:
- SP = 1290 (selling price)
- LossP = 14.000000000000002 (loss percentage)
Prove that: CP = 1500 (cost price)
--/
theorem find_cost_price (SP : ℝ) (LossP : ℝ) (CP : ℝ) (h1 : SP = 1290) (h2 : LossP = 14.000000000000002) : CP = 1500 :=
sorry

end find_cost_price_l775_775544


namespace chef_bought_almonds_l775_775271

theorem chef_bought_almonds (total_nuts pecans : ℝ)
  (h1 : total_nuts = 0.52) (h2 : pecans = 0.38) :
  total_nuts - pecans = 0.14 :=
by
  sorry

end chef_bought_almonds_l775_775271


namespace existence_of_square_with_conditions_l775_775743

noncomputable def problem_statement : Prop :=
  ∃ (k O A B C D E F G : ℝ × ℝ) (r : ℝ),
    (k.center = O ∧ ‖O - A‖ = 1) ∧
    (is_perpendicular (line_A_O) (line_A_B)) ∧
    (‖A - B‖ = sqrt 7 - 2) ∧
    (B ∈ line_O_B) ∧
    (C ∈ k ∧ OB ∩ k = C) ∧
    (D = midpoint O C) ∧
    (E = reflect D (line_A_O)) ∧
    (square_on k DEFG) ∧
    (F G ∈ k)

theorem existence_of_square_with_conditions : problem_statement :=
sorry

end existence_of_square_with_conditions_l775_775743


namespace modular_inverse_sum_eq_14_l775_775233

theorem modular_inverse_sum_eq_14 : 
(9 + 13 + 15 + 16 + 12 + 3 + 14) % 17 = 14 := by
  sorry

end modular_inverse_sum_eq_14_l775_775233


namespace LCM_of_36_and_220_l775_775535

theorem LCM_of_36_and_220:
  let A := 36
  let B := 220
  let productAB := A * B
  let HCF := 4
  let LCM := (A * B) / HCF
  LCM = 1980 := 
by
  sorry

end LCM_of_36_and_220_l775_775535


namespace eval_expression_l775_775665

-- Define the function g(x)
def g (x : ℝ) : ℝ := x^3 + 2 * x^2 + 3 * Real.sqrt x

-- State the theorem to be proven
theorem eval_expression : 3 * g 3 - 2 * g 9 = -1665 + 9 * Real.sqrt 3 :=
by
  -- Proof goes here
  sorry

end eval_expression_l775_775665


namespace sequence_value_a2017_l775_775793

noncomputable theory

def a_sequence (a : ℕ → ℚ) :=
  ∀ n, a (n + 1) = if 0 ≤ a n ∧ a n < 1/2 then 2 * a n else 2 * a n - 1

def initial_condition (a : ℕ → ℚ) := a 1 = 6 / 7

theorem sequence_value_a2017 (a : ℕ → ℚ) (h_seq : a_sequence a) (h_initial : initial_condition a) :
  a 2017 = 6 / 7 :=
sorry

end sequence_value_a2017_l775_775793


namespace circle_intersection_condition_l775_775600

theorem circle_intersection_condition
  (A B C D E F G H K M U : Point)
  (r CU : ℝ)
  (ABC_isosceles_right : is_isosceles_right_triangle A B C)
  (right_angle_C : ∠ A C B = 90)
  (U_midpoint : midpoint A B U)
  (M_on_CU : M ∈ segment U C)
  (CM_condition : dist C M = (sqrt 5 - 1) / 2 * dist C U)
  (r_condition : r = dist C U * sqrt (sqrt 5 - 2))
  (D_on_AB : D ∈ segment A B) (E_on_AB : E ∈ segment A B)
  (F_on_BC : F ∈ segment B C) (G_on_BC : G ∈ segment B C)
  (H_on_CA : H ∈ segment C A) (K_on_CA : K ∈ segment C A)
  (DE_FG_HK_condition : dist D E / dist F G = dist A B / dist B C 
    ∧ dist D E / dist H K = dist A B / dist C A) :
  ∠ D M E + ∠ F M G + ∠ H M K = 180 :=
by
  sorry

end circle_intersection_condition_l775_775600


namespace find_f_inv_486_l775_775822

-- Assuming function f: ℝ → ℝ
variable (f : ℝ → ℝ)

-- Given conditions
axiom f_cond1 : f 4 = 2
axiom f_cond2 : ∀ x : ℝ, f (3 * x) = 3 * f x

-- Proof problem: Prove that f⁻¹(486) = 972
theorem find_f_inv_486 : (∃ x : ℝ, f x = 486 ∧ x = 972) :=
sorry

end find_f_inv_486_l775_775822


namespace geometric_sequence_of_distances_l775_775863

noncomputable def curve_C (ρ θ a : ℝ) : Prop :=
  ρ * (Real.sin θ) ^ 2 = 2 * a * (Real.cos θ)

def line_l (x y t : ℝ) : Prop :=
  x = -2 + t ∧ y = -4 + t

def intersects (x y a : ℝ) : Prop :=
  y = x - 2 ∧ y ^ 2 = 2 * a * x

theorem geometric_sequence_of_distances (a : ℝ) (h_pos_a : a > 0) : 
  (∃ P M N : ℝ × ℝ, 
    let x1 := (P.1 + 2) ^ 2 + (P.2 + 4) ^ 2,
    let x2 := (P.1 + 2) ^ 2 + (P.2 + 4) ^ 2 in
    P = (-2, -4) ∧ 
    intersects M.1 M.2 a ∧ 
    intersects N.1 N.2 a ∧ 
    ((M.1 - P.1) ^ 2 + (M.2 - P.2) ^ 2) ^ 0.5 * 
    ((N.1 - P.1) ^ 2 + (N.2 - P.2) ^ 2) ^ 0.5 = 
    ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) ^ 0.5 * 
    ((M.1 - N.1) ^ 2 + (M.2 - N.2) ^ 2) ^ 0.5) → a = 1 :=
sorry

end geometric_sequence_of_distances_l775_775863


namespace largest_number_l775_775263

theorem largest_number (P Q R S T : ℕ) 
  (hP_digits_prime : ∃ p1 p2, P = 10 * p1 + p2 ∧ Prime P ∧ Prime (p1 + p2))
  (hQ_multiple_of_5 : Q % 5 = 0)
  (hR_odd_non_prime : Odd R ∧ ¬ Prime R)
  (hS_prime_square : ∃ p, Prime p ∧ S = p * p)
  (hT_mean_prime : T = (P + Q) / 2 ∧ Prime T)
  (hP_range : 10 ≤ P ∧ P ≤ 99)
  (hQ_range : 2 ≤ Q ∧ Q ≤ 19)
  (hR_range : 2 ≤ R ∧ R ≤ 19)
  (hS_range : 2 ≤ S ∧ S ≤ 19)
  (hT_range : 2 ≤ T ∧ T ≤ 19) :
  max P (max Q (max R (max S T))) = Q := 
by 
  sorry

end largest_number_l775_775263


namespace solve_equation_l775_775185

theorem solve_equation (x : ℝ) : 
  (x - 4)^6 + (x - 6)^6 = 64 → x = 4 ∨ x = 6 :=
by
  sorry

end solve_equation_l775_775185


namespace pirates_initial_coins_l775_775983

/-- 
Given a scenario where two pirates repeatedly lose half of their gold coins,
we prove that the initial number of coins the first pirate had is 24,
given the conditions of the game and the final amounts of coins.
-/
theorem pirates_initial_coins
  (coins_final_p1 : ℕ := 15)
  (coins_final_p2 : ℕ := 33)
  (coins_initial : ℕ := 24) :
  ∃ coins1 coins2 coins3 coins4, 
  let coins2 := coins_initial / 2 + coins_initial / 2 in
  let coins3 := coins2 / 2 * 2 in
  let coins4 := coins3 / 2 + coins3 / 2 in
    coins1 = coins_final_p1 → 
    coins4 = coins_final_p2 → 
    coins_initial = 24 := 
begin
  sorry
end

end pirates_initial_coins_l775_775983


namespace angle_EDF_is_55_degrees_l775_775865

open EuclideanGeometry

theorem angle_EDF_is_55_degrees :
  ∀ {A B C D E F : Point},
    IsoscelesTriangle A B C →
    MeasureAngle A = 70 →
    LiesOn D (line B C) →
    LiesOn E (line A C) →
    LiesOn F (line A B) →
    Dist C E = Dist C D →
    Dist B F = Dist B D →
    MeasureAngle E D F = 55 :=
by
  -- This is where the proof would go
  apply sorry

end angle_EDF_is_55_degrees_l775_775865


namespace complex_quadrant_l775_775388

-- Statement to prove the quadrant of the complex number
theorem complex_quadrant (z : ℂ) (h : z = complex.mk 3 1 / complex.mk 1 (-1)) : 
  (0 < z.re) ∧ (0 < z.im) := by
  sorry

end complex_quadrant_l775_775388


namespace orthocenter_reflection_l775_775113

open Real

structure Triangle :=
  (A B C : ℝ)
  (angle_C angle_B : ℝ)
  (angle_condition : angle_C = 90 + angle_B)

def isReflectionOverBC (T : Triangle) : Prop :=
  -- Dummy definition, in practice this should be the rigorous geometric condition
  sorry

theorem orthocenter_reflection (T : Triangle) (h : T.angle_condition) :
  isReflectionOverBC T :=
sorry

end orthocenter_reflection_l775_775113


namespace yellow_bean_ratio_is_correct_l775_775732

def bagA := 30
def bagB := 32
def bagC := 34
def bagD := 36

def yellowRatioA := 0.40
def yellowRatioB := 0.30
def yellowRatioC := 0.25
def yellowRatioD := 0.10

def yellowBeansA := round (bagA * yellowRatioA)
def yellowBeansB := round (bagB * yellowRatioB)
def yellowBeansC := round (bagC * yellowRatioC)
def yellowBeansD := round (bagD * yellowRatioD)

def totalYellowBeans := yellowBeansA + yellowBeansB + yellowBeansC + yellowBeansD
def totalBeans := bagA + bagB + bagC + bagD

def yellowRatio := (totalYellowBeans / totalBeans) * 100

theorem yellow_bean_ratio_is_correct : yellowRatio ≈ 26.52 := by
  sorry

end yellow_bean_ratio_is_correct_l775_775732


namespace number_of_subsets_of_P_l775_775794

noncomputable def P : Set ℝ := {x | x^2 - 2*x + 1 = 0}

theorem number_of_subsets_of_P : ∃ (n : ℕ), n = 2 ∧ ∀ S : Set ℝ, S ⊆ P → S = ∅ ∨ S = {1} := by
  sorry

end number_of_subsets_of_P_l775_775794


namespace num_ints_congruent_to_2_mod_7_l775_775812

theorem num_ints_congruent_to_2_mod_7 :
  ∃ n : ℕ, (∀ k, 1 ≤ 7 * k + 2 ∧ 7 * k + 2 ≤ 300 ↔ 0 ≤ k ≤ 42) ∧ n = 43 :=
sorry

end num_ints_congruent_to_2_mod_7_l775_775812


namespace range_of_m_l775_775169

variable (m : ℝ)
def p := m > 1
def q := ∀ x : ℝ, m * x^2 + m * x + 1 > 0 
def p_and_q := ¬ (p ∧ q)
def p_or_q := p ∨ q

theorem range_of_m : {m : ℝ | ¬ (p m ∧ q m) ∧ (p m ∨ q m)} = {m | (0 ≤ m ∧ m ≤ 1) ∨ (4 ≤ m)} :=
by
  sorry

end range_of_m_l775_775169


namespace minimum_sum_of_products_l775_775908

-- Define the distinct positive integer sequence and their constraint
def sequence (a : Fin 10 → ℕ) : Prop :=
  (∀ i j, i ≠ j → a i ≠ a j) ∧ (∑ i, a i = 1995)

-- Define the sum S as a circular sum of products.
def sum_of_products (a : Fin 10 → ℕ) : ℕ :=
  ∑ i : Fin 10, a i * a ((i + 1) % 10)

theorem minimum_sum_of_products :
  ∃ (a : Fin 10 → ℕ), sequence a ∧ sum_of_products a = 6044 := 
sorry

end minimum_sum_of_products_l775_775908


namespace paintings_after_30_days_l775_775502

theorem paintings_after_30_days (paintings_per_day : ℕ) (initial_paintings : ℕ) (days : ℕ)
    (h1 : paintings_per_day = 2)
    (h2 : initial_paintings = 20)
    (h3 : days = 30) :
    initial_paintings + paintings_per_day * days = 80 := by
  sorry

end paintings_after_30_days_l775_775502


namespace noncongruent_integer_tris_l775_775414

theorem noncongruent_integer_tris : 
  ∃ S : Finset (ℕ × ℕ × ℕ), S.card = 18 ∧ 
    ∀ (a b c : ℕ), (a, b, c) ∈ S → 
      (a + b > c ∧ a + b + c < 20 ∧ a < b ∧ b < c ∧ a^2 + b^2 ≠ c^2) :=
sorry

end noncongruent_integer_tris_l775_775414


namespace P_xi_gt_30_l775_775170

noncomputable def letter_weight : ℝ → ℂ := sorry -- ξ (weight of a letter in grams)

axiom P_xi_lt_10 : letter_weight(ξ) < 10 = 0.3
axiom P_10_leq_xi_leq_30 : 10 ≤ letter_weight(ξ) ≤ 30 = 0.4

theorem P_xi_gt_30 : letter_weight(ξ) > 30 = 0.3 :=
by 
  have h1 : P(letter_weight(ξ) < 10) + P(10 ≤ letter_weight(ξ) ≤ 30) + P(letter_weight(ξ) > 30) = 1 := sorry
  rw [P_xi_lt_10, P_10_leq_xi_leq_30] at h1
  sorry

end P_xi_gt_30_l775_775170


namespace bullet_train_length_l775_775269

def kmph_to_mps (v : ℝ) : ℝ :=
  v * (5 / 18)

theorem bullet_train_length
  (v_t : ℝ) (v_m : ℝ) (t : ℝ)
  (hv_t : v_t = 69)
  (hv_m : v_m = 3)
  (ht : t = 10) :
  let relative_speed := kmph_to_mps (v_t + v_m)
  in relative_speed * t = 200 :=
by
  have h_rel_speed : relative_speed = kmph_to_mps (69 + 3) := by rw [hv_t, hv_m]
  have h_rel_speed_mps : relative_speed = 20 := by norm_num [kmph_to_mps, h_rel_speed]
  have h_time : t = 10 := by rw [ht]
  have h_length : 20 * 10 = 200 := by norm_num
  rw [h_rel_speed_mps, h_time, h_length]
  exact h_length

end bullet_train_length_l775_775269


namespace find_m_perpendicular_l775_775623

theorem find_m_perpendicular (m : ℝ) : 
  (∃ k : ℝ, (k = 2 / 3) ∧ (∀ p1 p2 : ℤ × ℤ, (p1 = (1, 2)) ∧ (p2 = (m, 3)) → 
  (2 * fst p1 - 3 * snd p1 + 1 = 0 ∧ 2 * fst p2 - 3 * snd p2 + 1 = 0) → 
  (∃ k' : ℝ, k' = -(1 / k)) → 
  (snd p2 - snd p1)/(fst p2 - fst p1) = k' → m = 1 / 3))

end find_m_perpendicular_l775_775623


namespace quad_is_parallelogram_l775_775168

variables {A B C D P Q : Type}
variables [Point A] [Point B] [Point C] [Point D] [Point P] [Point Q]
variables (proj_circP: Circle P) (proj_circQ: Circle Q)
variables (conc_center: Point) (diff_radii: ℝ → ℝ → Prop)

-- We assume projections onto sidelines lie on given circles.
axiom proj_on_circles : ∀ (proj_pts : Set Point), 
  (proj_pts ⊆ Circle_points proj_circP ∨ proj_pts ⊆ Circle_points proj_circQ) 
  ∧ ∃ r1 r2, diff_radii r1 r2 ∧ r1 ≠ r2

-- We add concentric condition
axiom concentric_circles : circles_are_concentric proj_circP proj_circQ conc_center

-- The definition of parallelogram
def is_parallelogram (A B C D: Type) [Point A] [Point B] [Point C] [Point D] : Prop :=
  are_parallel (Line.mk A B) (Line.mk C D) ∧ are_parallel (Line.mk A D) (Line.mk B C)

-- The theorem we aim to prove
theorem quad_is_parallelogram 
  (ABCD : quadrilateral A B C D)
  (projections_condition : proj_on_circles (sidelines_projections A B C D P Q))
  (concentric_condition : concentric_circles ) : 
  is_parallelogram A B C D :=
sorry

end quad_is_parallelogram_l775_775168


namespace chess_team_selection_l775_775161

theorem chess_team_selection:
  let boys := 10
  let girls := 12
  let team_size := 8     -- total team size
  let boys_selected := 5 -- number of boys to select
  let girls_selected := 3 -- number of girls to select
  ∃ (w : ℕ), 
  (w = Nat.choose boys boys_selected * Nat.choose girls girls_selected) ∧ 
  w = 55440 :=
by
  sorry

end chess_team_selection_l775_775161


namespace length_of_AC_l775_775852

-- Definitions from the problem
variable (AB BC CD DA : ℝ)
variable (angle_ADC : ℝ)
variable (AC : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  AB = 10 ∧ BC = 10 ∧ CD = 17 ∧ DA = 17 ∧ angle_ADC = 120

-- The mathematically equivalent proof statement
theorem length_of_AC (h : conditions AB BC CD DA angle_ADC) : AC = Real.sqrt 867 := sorry

end length_of_AC_l775_775852


namespace train_overtake_distance_correct_l775_775599

-- Define the conditions
def train_A_speed : ℝ := 30
def train_B_speed : ℝ := 42
def train_B_delay : ℝ := 2

-- Define the function to find the distance where Train A is overtaken by Train B
def overtake_distance (train_A_speed train_B_speed train_B_delay : ℝ) : ℝ :=
  let distance_ahead := train_A_speed * train_B_delay
  let relative_speed := train_B_speed - train_A_speed
  let overtake_time := distance_ahead / relative_speed
  train_A_speed * overtake_time

-- Define the theorem to be proven
theorem train_overtake_distance_correct : 
  overtake_distance train_A_speed train_B_speed train_B_delay = 150 :=
by 
  -- skip the proof
  sorry

end train_overtake_distance_correct_l775_775599


namespace least_three_digit_with_factors_l775_775994

theorem least_three_digit_with_factors (n : ℕ) :
  (n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n) → n = 210 := by
  sorry

end least_three_digit_with_factors_l775_775994


namespace jane_change_l775_775870

def cost_of_skirt := 13
def cost_of_blouse := 6
def skirts_bought := 2
def blouses_bought := 3
def amount_paid := 100

def total_cost_skirts := skirts_bought * cost_of_skirt
def total_cost_blouses := blouses_bought * cost_of_blouse
def total_cost := total_cost_skirts + total_cost_blouses
def change_received := amount_paid - total_cost

theorem jane_change : change_received = 56 :=
by
  -- Proof goes here, but it's skipped with sorry
  sorry

end jane_change_l775_775870


namespace fraction_ratio_l775_775073

variable {α : Type*} [DivisionRing α] (a b : α)

theorem fraction_ratio (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := 
by sorry

end fraction_ratio_l775_775073


namespace weight_polynomial_sum_l775_775905

noncomputable def weight (P : Polynomial ℤ) : ℕ :=
  P.coeffs.count is_odd

theorem weight_polynomial_sum (n : ℕ) (i : Finₓ n → ℕ) (h : ∀ j : Finₓ n, i j < i j.succ) :
  weight (∑ j in Finset.range n, (1 : Polynomial ℤ) + Polynomial.C (X : Polynomial ℤ) ^ (i j)) ≥ weight ((1 : Polynomial ℤ) + Polynomial.C (X : Polynomial ℤ) ^ (i 0)) :=
begin
  sorry
end

end weight_polynomial_sum_l775_775905


namespace least_m_properly_placed_polygons_l775_775704

noncomputable def min_lines_through_origin (polygons : List Polygon) : ℕ :=
  2

theorem least_m_properly_placed_polygons :
  ∀ (polygons : List Polygon), (∀ (P1 P2 : Polygon), ∃ (line : Line),
    (line.through_origin ∧ line.cuts_polygon P1 ∧ line.cuts_polygon P2)) → min_lines_through_origin polygons = 2 :=
by
  -- Conditions and definitions would be expanded here in a complete proof
  sorry

end least_m_properly_placed_polygons_l775_775704


namespace math_problem_l775_775434

noncomputable theory

variables {A B C H D E F : Type}
variables (BD CD : ℝ) (AH HD : ℝ)
variables [add_comm_group A] [vector_space ℝ A]
variables [add_comm_group B] [vector_space ℝ B]
variables [add_comm_group C] [vector_space ℝ C]
variables [add_comm_group H] [vector_space ℝ H]
variables [add_comm_group D] [vector_space ℝ D]
variables [add_comm_group E] [vector_space ℝ E]
variables [add_comm_group F] [vector_space ℝ F]

-- Given conditions
def conditions : Prop :=
  BD = 3 ∧
  CD = 7 ∧
  AH / HD = 5 / 7 ∧
  IsOrthocenter H ∧
  IsFootOfAltitude D A ∧
  IntersectsCircumcircle B H C AC E ∧
  IntersectsCircumcircle B H C AB F 

-- Definition of the area of triangle AEF in simplified form
def area_AEF := (120 : ℤ) / 17

-- The final problem statement
theorem math_problem : conditions → (100 * 120 + 17) = 12017 :=
sorry

end math_problem_l775_775434


namespace bounded_sum_square_inequality_l775_775026

-- Define the necessary conditions and variables
variables (n : ℕ) (a : ℕ → ℕ) (x : ℝ)
  (h1 : n ≥ 2)
  (h2 : ∀ i j, i < j → i < n → j < n → a i < a j)
  (h3 : ∑ i in finset.range n, (1 : ℝ) / a i ≤ 1)

-- State the theorem to be proved
theorem bounded_sum_square_inequality :
  (∑ i in finset.range n, 1 / (a i ^ 2 + x ^ 2)) ^ 2 ≤ 1 / 2 * (1 / (a 0 * (a 0 - 1) + x ^ 2)) := 
  sorry

end bounded_sum_square_inequality_l775_775026


namespace min_sum_areas_l775_775100

theorem min_sum_areas (y₁ y₂ : ℝ) (hyp1 : (y₁^2 / 2) ^ 2 + y₁ y₂ = -1) :
  (y₁ * y₂ = -2) → 
  min ((1 / 4) * (abs y₁ + abs y₂)) = ((Real.sqrt 2) / 2) := 
sorry

end min_sum_areas_l775_775100


namespace distance_between_points_l775_775581

-- Defining the points
def p1 : ℝ × ℝ := (1, 3)
def p2 : ℝ × ℝ := (4, 6)

--The theorem to prove the distance between points p1 and p2 is 3 * (sqrt 2)
theorem distance_between_points :
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 3 * real.sqrt 2 :=
by
  -- Calculation can be inserted here.
  sorry

end distance_between_points_l775_775581


namespace james_has_more_balloons_l775_775869

theorem james_has_more_balloons (james_balloons : ℕ) (amy_balloons : ℕ) :
  james_balloons = 232 → amy_balloons = 101 → james_balloons - amy_balloons = 131 :=
by
  intros h₁ h₂
  rw [h₁, h₂]
  exact Nat.sub_eq_sub_min 232 101

end james_has_more_balloons_l775_775869


namespace jean_pairs_of_pants_l775_775115

theorem jean_pairs_of_pants
  (retail_price : ℝ)
  (discount_rate : ℝ)
  (tax_rate : ℝ)
  (total_paid : ℝ)
  (number_of_pairs : ℝ)
  (h1 : retail_price = 45)
  (h2 : discount_rate = 0.20)
  (h3 : tax_rate = 0.10)
  (h4 : total_paid = 396)
  (h5 : number_of_pairs = total_paid / ((retail_price * (1 - discount_rate)) * (1 + tax_rate))) :
  number_of_pairs = 10 :=
by
  sorry

end jean_pairs_of_pants_l775_775115


namespace volume_ratio_l775_775085

noncomputable def ratio_volumes (d : ℝ) (cylinder_volume sphere_volume cone_volume : ℝ) : Prop :=
  let r := d / 2
  let V_cylinder := 2 * π * r ^ 3
  let V_sphere := (4 / 3) * π * r ^ 3
  let V_cone := (2 / 3) * π * r ^ 3
  (cylinder_volume = V_cylinder) ∧ (sphere_volume = V_sphere) ∧ (cone_volume = V_cone) ∧
  (cylinder_volume / sphere_volume = 3) ∧ (sphere_volume / sphere_volume = 1) ∧ (cone_volume / sphere_volume = 1 / 2)

theorem volume_ratio (d : ℝ) : ∃ cylinder_volume sphere_volume cone_volume, ratio_volumes d cylinder_volume sphere_volume cone_volume :=
by
  let r := d / 2
  let V_cylinder := 2 * π * r ^ 3
  let V_sphere := (4 / 3) * π * r ^ 3
  let V_cone := (2 / 3) * π * r ^ 3
  use [V_cylinder, V_sphere, V_cone]
  unfold ratio_volumes
  split
  · refl
  split
  · refl
  split
  · refl
  split
  · field_simp [V_sphere, V_cylinder]
    norm_num
  split
  · field_simp [V_sphere]
    norm_num
  · field_simp [V_sphere, V_cone]
    norm_num
  sorry

end volume_ratio_l775_775085


namespace number_of_false_statements_l775_775419

-- Definitions for the conditions
variable (P : Type) (l m : Type)
variable [affine_space P] [line l] [line m]
axiom skew_lines : skew l m

-- Definitions for the statements
def statement_1 (P : Type) (l m : Type) [affine_space P] [line l] [line m] : Prop :=
  ∃! L, L ∥ l ∧ L ∥ m ∧ passes_through L P

def statement_2 (P : Type) (l m : Type) [affine_space P] [line l] [line m] : Prop :=
  ∃! L, L ⟂ l ∧ L ⟂ m ∧ passes_through L P

def statement_3 (P : Type) (l m : Type) [affine_space P] [line l] [line m] : Prop :=
  ∃! L, intersects L l ∧ intersects L m ∧ passes_through L P

def statement_4 (P : Type) (l m : Type) [affine_space P] [line l] [line m] : Prop :=
  ∃! L, is_skew_to L l ∧ is_skew_to L m ∧ passes_through L P

-- Theorem stating there is exactly one false statement
theorem number_of_false_statements (P : Type) (l m : Type) 
  [affine_space P] [line l] [line m] (skew_lines l m):
  ∃ f1 f2 f3 f4 : Prop, ¬ statement_1 P l m ∧ statement_2 P l m ∧ statement_3 P l m ∧ statement_4 P l m ∧ (f1 ∧ f2 ∧ f3 ∧ f4 = 1) :=
sorry

end number_of_false_statements_l775_775419


namespace vector_addition_correct_l775_775680

def vec1 : ℤ × ℤ := (5, -9)
def vec2 : ℤ × ℤ := (-8, 14)
def vec_sum (v1 v2 : ℤ × ℤ) : ℤ × ℤ := (v1.1 + v2.1, v1.2 + v2.2)

theorem vector_addition_correct :
  vec_sum vec1 vec2 = (-3, 5) :=
by
  -- Proof omitted
  sorry

end vector_addition_correct_l775_775680


namespace positive_integer_solution_equiv_nonnegative_integer_solution_l775_775518

theorem positive_integer_solution_equiv_nonnegative_integer_solution
    (n a : ℕ) (h_pos_a : 0 < a) :
    (∃ x : Fin n → ℕ, (∀ i, 0 < x i) ∧ (∑ i in Finset.range n, (i + 1) * x ⟨i, by { simp [Fin.is_lt] }⟩) = a) ↔
    (∃ y : Fin n → ℕ, (∀ i, 0 ≤ y i) ∧ (∑ i in Finset.range n, (i + 1) * y ⟨i, by { simp [Fin.is_lt] }⟩) = a - (n * (n + 1)) / 2) :=
sorry

end positive_integer_solution_equiv_nonnegative_integer_solution_l775_775518


namespace some_number_proof_l775_775710

def g (n : ℕ) : ℕ :=
  if n < 3 then 1 else 
  if n % 2 = 0 then g (n - 1) else 
    g (n - 2) * n

theorem some_number_proof : g 106 - g 103 = 105 :=
by sorry

end some_number_proof_l775_775710


namespace greatest_three_digit_divisible_by_4_and_9_l775_775235

theorem greatest_three_digit_divisible_by_4_and_9 : 
  ∃ n, 100 ≤ n ∧ n ≤ 999 ∧ (n % 4 = 0 ∧ n % 9 = 0) ∧ ∀ m, (100 ≤ m ∧ m ≤ 999 ∧ m % 4 = 0 ∧ m % 9 = 0 → m ≤ n) :=
begin
  use 972,
  split,
  { exact dec_trivial },            -- 100 ≤ 972 is true trivially
  split,
  { exact dec_trivial },            -- 972 ≤ 999 is true trivially
  split,
  { split, 
    { exact dec_trivial },          -- 972 % 4 = 0 is true trivially
    { exact dec_trivial }           -- 972 % 9 = 0 is true trivially
  },
  { intros m h,
    cases h with hl hr,
    cases hr with hr1 hr2,
    cases hr2 with hr21 hr22,
    dsimp at *,
    linarith }                        -- The maximum is achieved, so m ≤ 972 holds for all m in range
end

end greatest_three_digit_divisible_by_4_and_9_l775_775235


namespace most_balls_l775_775565

def soccerballs : ℕ := 50
def basketballs : ℕ := 26
def baseballs : ℕ := basketballs + 8

theorem most_balls :
  max (max soccerballs basketballs) baseballs = soccerballs := by
  sorry

end most_balls_l775_775565


namespace probability_a_in_B_l775_775406

noncomputable def setA : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
noncomputable def setB : Set ℝ := { y | 0 ≤ y ∧ y ≤ (5 / 2) }

theorem probability_a_in_B (a : ℝ) (ha : a ∈ setA) : 
  let p := (5 / 2) / 5 in
  p = 1 / 2 := 
by
  sorry

end probability_a_in_B_l775_775406


namespace probability_of_getting_all_books_l775_775572

open Classical

def total_possible_scenarios : ℕ := 8
def favorable_scenarios : ℕ := 2

theorem probability_of_getting_all_books :
  (favorable_scenarios : ℚ) / total_possible_scenarios = 1 / 4 := 
  sorry

end probability_of_getting_all_books_l775_775572


namespace find_monotonic_intervals_and_value_a_l775_775397

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log a x - x^3

theorem find_monotonic_intervals_and_value_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1)
: ( (0 < a ∧ a < 1) →
      (∀ x > 0, f x a < 0)) ∧ 
  ((a > 1) →
      (∀ x, (0 < x ∧ x < (1 / (3 * Real.log a))) → 
        f x a > 0) ∧
      (∀ x, (x > (1 / (3 * Real.log a))) → 
        f x a < 0)) ∧ 
  (f (1 / (3 * Real.log a)) a = (1 / 3) * Real.log a (2 / 3) - (2 / 3) → 
   a = Real.sqrt Real.e) := sorry

end find_monotonic_intervals_and_value_a_l775_775397


namespace arithmetic_series_sum_plus_100_l775_775656

theorem arithmetic_series_sum_plus_100 : 
  let a₁ := 10
  let aₙ := 100
  let d := 1
  let n := ((aₙ - a₁) / d) + 1 
  let S := n * (a₁ + aₙ) / 2
  S + 100 = 5105 := 
by
  -- Definitions
  let a₁ := 10
  let aₙ := 100
  let d := 1
  let n := ((aₙ - a₁) / d) + 1
  let S := n * (a₁ + aₙ) / 2
  -- Goals
  have h₁ : S = 5005 := sorry
  have h₂ : S + 100 = 5105 := by rw [h₁]; norm_num
  exact h₂

end arithmetic_series_sum_plus_100_l775_775656


namespace length_equality_l775_775157

theorem length_equality (A B C D E F : Point) (ABCD_square : square A B C D) (E_on_CD : on_line_segment E C D) (F_on_BC : on_bisector (BAE_bisector A B E) B F) :
  length_segment A E = length_segment B F + length_segment E D :=
sorry

end length_equality_l775_775157


namespace perfect_square_factors_of_18000_l775_775806

theorem perfect_square_factors_of_18000 :
  let a_range := [0, 2];
  let b_range := [0, 2];
  let c_range := [0, 2];
  finset.prod (finset.of_list a_range) (λ a, finset.prod (finset.of_list b_range) (λ b, finset.prod (finset.of_list c_range) (λ c, 1))) = 8 :=
by sorry

end perfect_square_factors_of_18000_l775_775806


namespace sum_partial_fractions_series_l775_775328

theorem sum_partial_fractions_series :
  (∑ n in finset.range 9, 1 / ((n + 2) * (n + 3): ℚ)) = 9 / 22 := sorry

end sum_partial_fractions_series_l775_775328


namespace initial_points_l775_775529

theorem initial_points (n : ℕ) (h : 16 * n - 15 = 225) : n = 15 :=
sorry

end initial_points_l775_775529


namespace count_integers_congruent_to_2_mod_7_up_to_300_l775_775809

theorem count_integers_congruent_to_2_mod_7_up_to_300 : 
  (Finset.card (Finset.filter (λ n : ℕ, n % 7 = 2) (Finset.range 301))) = 43 := 
by
  sorry

end count_integers_congruent_to_2_mod_7_up_to_300_l775_775809


namespace find_focus_with_larger_x_coordinate_l775_775690

def hyperbola_center (x y : ℝ) : Bool :=
  x = 5 ∧ y = 10

def hyperbola_a : ℝ := 7
def hyperbola_b : ℝ := 15
def hyperbola_c (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

def hyperbola_focus_with_larger_x_coordinate
  (x y a b : ℝ) : Prop :=
  x = 5 + Real.sqrt (a^2 + b^2) ∧ y = 10

theorem find_focus_with_larger_x_coordinate :
  ∀ x y : ℝ, 
  ∀ a b : ℝ,
  hyperbola_center 5 10 →
  a = hyperbola_a →
  b = hyperbola_b →
  hyperbola_focus_with_larger_x_coordinate x y a b :=
sorry

end find_focus_with_larger_x_coordinate_l775_775690


namespace sum_of_intersections_correct_l775_775747

structure Point where
  x : ℝ
  y : ℝ

def graph : List (Point × Point) := [
  (Point.mk (-4) (-5), Point.mk (-2) (-1)),
  (Point.mk (-2) (-1), Point.mk (-1) (-2)),
  (Point.mk (-1) (-2), Point.mk (1) (2)),
  (Point.mk (1) (2), Point.mk (2) (1)),
  (Point.mk (2) (1), Point.mk (4) (5))
]

def intersects_with_line (P1 P2 : Point) (m b : ℝ) : Option ℝ :=
  let x1 := P1.x
  let y1 := P1.y
  let x2 := P2.x
  let y2 := P2.y
  -- Ensure segment P1P2 is not vertical
  if x1 ≠ x2 then
    let slope := (y2 - y1) / (x2 - x1)
    let intercept := y1 - slope * x1
    -- Solve for intersection with y = mx + b
    let x_intersection := (intercept - b) / (m - slope)
    -- Check if intersection is within the segment
    if x1 ≤ x_intersection ∧ x_intersection ≤ x2 ∨ x2 ≤ x_intersection ∧ x_intersection ≤ x1 then
      return some x_intersection
  return none

def sum_of_intersections (segments : List (Point × Point)) (m b : ℝ) : ℝ :=
  segments.foldl (λ acc (P1 P2) => 
    match intersects_with_line P1 P2 m b with
    | some x => acc + x
    | none => acc) 0

theorem sum_of_intersections_correct :
  sum_of_intersections graph 1 2 = 7 := by
  sorry

end sum_of_intersections_correct_l775_775747


namespace points_symmetric_about_x_axis_l775_775558

def point := ℝ × ℝ

def symmetric_x_axis (A B : point) : Prop :=
  A.1 = B.1 ∧ A.2 = -B.2

theorem points_symmetric_about_x_axis : symmetric_x_axis (-1, 3) (-1, -3) :=
by
  sorry

end points_symmetric_about_x_axis_l775_775558


namespace botanical_garden_path_length_l775_775276

theorem botanical_garden_path_length
  (scale : ℝ)
  (path_length_map : ℝ)
  (path_length_real : ℝ)
  (h_scale : scale = 500)
  (h_path_length_map : path_length_map = 6.5)
  (h_path_length_real : path_length_real = path_length_map * scale) :
  path_length_real = 3250 :=
by
  sorry

end botanical_garden_path_length_l775_775276


namespace polar_to_cartesian_parabola_l775_775317

theorem polar_to_cartesian_parabola (r theta : ℝ) (x y : ℝ) :
  (r = 2 * sin theta * sec theta) →
  (r = real.sqrt (x^2 + y^2)) →
  (sin theta = y / real.sqrt (x^2 + y^2)) →
  (cos theta = x / real.sqrt (x^2 + y^2)) →
  (x^2 = 2 * y) :=
by
  -- Proof steps would go here
  sorry

end polar_to_cartesian_parabola_l775_775317


namespace number_of_women_at_tables_l775_775636

def num_tables : ℝ := 9.0
def num_men : ℝ := 3.0
def avg_customers_per_table : ℝ := 1.111111111

theorem number_of_women_at_tables : 
  let total_customers := num_tables * avg_customers_per_table in
  let num_women := total_customers - num_men in
  num_women = 7.0 := 
by 
  sorry

end number_of_women_at_tables_l775_775636


namespace sum_of_solutions_l775_775145

noncomputable theory

-- Define the function f
def f (x : ℝ) : ℝ := -12 * x + 5

-- Define the inverse of f
def f_inv (y : ℝ) : ℝ := (5 - y) / -12

-- Define the transformation for the composition
def g (x : ℝ) : ℝ := (3 * x)⁻¹

-- Define the statement to prove
theorem sum_of_solutions :
  let solutions := {x | f_inv x = f (g x)} in
  ∑ x in solutions, x = 55 := by
  sorry

end sum_of_solutions_l775_775145


namespace sum_series_to_fraction_l775_775333

theorem sum_series_to_fraction :
  (∑ n in Finset.range 9, (1 / ((n + 2) * (n + 3) : ℚ))) = 9 / 22 := 
begin
  sorry
end

end sum_series_to_fraction_l775_775333


namespace cricket_initial_average_l775_775944

theorem cricket_initial_average:
  ∃ (A : ℕ), 
    let total_runs_initial := 10 * A in
    let total_runs_next := 10 * A + 59 in
    let new_average := (total_runs_next : ℚ) / 11 in
    total_runs_initial > 0 ∧ 
    total_runs_next / 11 = A + 4 ∧
    new_average = 15 := 
sorry

end cricket_initial_average_l775_775944


namespace incorrect_statement_D_l775_775592

-- Define the necessary conditions
def distance_to_x_axis (P : ℝ × ℝ) : ℝ := |P.2|

def point_in_second_quadrant (a : ℝ) : Prop :=
  let P := (-|a| - 1, a^2 + 1)
  P.1 < 0 ∧ P.2 > 0

def angle_bisector_property (x y : ℝ) : Prop :=
  (x > 0 ∧ y > 0 ∧ x = y) ∨ (x < 0 ∧ y < 0 ∧ x = y)

def cube_root_of_square (x : ℝ) : Prop :=
  x^2 = 64 → real.cbrt x = 2

-- Define the statement that ensures the correctness of the conditions
theorem incorrect_statement_D :
  (∀ P : ℝ × ℝ, P = (3, -2) → distance_to_x_axis P = 2) →
  (∀ a : ℝ, point_in_second_quadrant a) →
  (∀ x y : ℝ, angle_bisector_property x y) →
  ¬(∀ x : ℝ, cube_root_of_square x) :=
by
  intros hA hB hC hD
  -- Further proof steps would go here
  sorry

end incorrect_statement_D_l775_775592


namespace doughnut_unit_l775_775267

theorem doughnut_unit (boxes_units : ℕ) (ate_doughnuts : ℕ) (left_doughnuts : ℕ) 
    (H1 : boxes_units = 2) (H2 : ate_doughnuts = 8) (H3 : left_doughnuts = 16) : 
    boxes_units * (left_doughnuts + ate_doughnuts) = 24 ∧ 24 / boxes_units = 12 :=
by
  have H4 : left_doughnuts + ate_doughnuts = 24, from sorry
  have H5 : 24 / boxes_units = 12, from sorry
  exact ⟨H4, H5⟩

end doughnut_unit_l775_775267


namespace find_T_100_l775_775401

noncomputable def a (n : ℕ) : ℕ := 2^(n-1)

def b (n k : ℕ) : ℕ :=
if n > k ∧ n ≤ k + 1
then (-1)^k * k
else if n = k + 1
then a (n + 1)
else a n

def T (n : ℕ) : ℕ :=
∑ i in Finset.range n, b i i -- sum of the first n terms of the sequence {b_n}

theorem find_T_100 : T 100 = 8152 := by
  sorry

end find_T_100_l775_775401


namespace CE_eq_DE_l775_775014

variables {A B C D E : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E]

-- Assume all points lie in a metric space and are distinct
noncomputable def point_A : A := sorry
noncomputable def point_B : B := sorry
noncomputable def point_C : C := sorry
noncomputable def point_D : D := sorry
noncomputable def point_E : E := sorry

-- Assume D and C lie on the same side of the line AB, AB = AD + BC
axiom same_side (line AB : set A) (D C : A) : sorry
axiom length_eq (AB AD BC : ℝ) : AB = AD + BC

-- Assume the bisectors intersect at point E
axiom angle_bisect_ABE (A B C D E : A) : sorry
axiom angle_bisect_BAD (A B C D E : A) : sorry

-- Prove that CE = DE
theorem CE_eq_DE (A B C D E : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] :
  (dist (point_E : metric_space) (point_C : metric_space)) = (dist (point_E : metric_space) (point_D : metric_space)) :=
sorry

end CE_eq_DE_l775_775014


namespace evaluate_expression_l775_775666

def g (x : ℝ) : ℝ := x^3 - 3 * Real.sqrt x

theorem evaluate_expression :
  g 3 * g 1 - g 9 = -774 + 6 * Real.sqrt 3 :=
by
  sorry

end evaluate_expression_l775_775666


namespace function_eq_l775_775087

-- Given condition
def cond (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (exp x) = x + 1

-- Conclusion to prove
theorem function_eq (f : ℝ → ℝ) (h : cond f) : ∀ x : ℝ, f x = log x + 1 :=
by
  sorry

end function_eq_l775_775087


namespace sum_sequence_a_b_eq_1033_l775_775961

def a (n : ℕ) : ℕ := n + 1
def b (n : ℕ) : ℕ := 2^(n-1)

theorem sum_sequence_a_b_eq_1033 : 
  (a (b 1)) + (a (b 2)) + (a (b 3)) + (a (b 4)) + (a (b 5)) + 
  (a (b 6)) + (a (b 7)) + (a (b 8)) + (a (b 9)) + (a (b 10)) = 1033 := by
  sorry

end sum_sequence_a_b_eq_1033_l775_775961


namespace number_of_uniforms_l775_775015

def pants_cost := 20
def shirt_cost := 2 * pants_cost
def tie_cost := shirt_cost / 5
def socks_cost := 3
def uniform_cost := pants_cost + shirt_cost + tie_cost + socks_cost
def total_spending := 355

theorem number_of_uniforms (pants_cost: ℕ) (shirt_cost: ℕ) (tie_cost: ℕ) (socks_cost: ℕ) (uniform_cost: ℕ) (total_spending: ℕ) : 
  pants_cost = 20 →
  shirt_cost = 2 * pants_cost →
  tie_cost = shirt_cost / 5 →
  socks_cost = 3 →
  uniform_cost = pants_cost + shirt_cost + tie_cost + socks_cost →
  total_spending = 355 →
  total_spending / uniform_cost = 5 :=
begin
  intros hpants hshirt htie hsocks huniform htotal,
  rw [hpants, hshirt, htie, hsocks, huniform, htotal],
  norm_num,
end

end number_of_uniforms_l775_775015


namespace KalebBooks_l775_775128

variable (initial_books : Nat)
variable (sold_books : Nat)
variable (new_books : Nat)

-- Defining the conditions
def initialBooks := 34
def soldBooks := 17
def newBooks := 7

-- Statement of the math proof problem
theorem KalebBooks : (initialBooks - soldBooks + newBooks = 24) := by
  -- These definitions are equivalent to the conditions given in part a.
  let initialBooks := 34
  let soldBooks := 17
  let newBooks := 7
  -- The proof calculation using the initial_books, sold_books, and new_books variables.
  have sub_result : initialBooks - soldBooks = 17 := by
    sorry -- This is a simplified subtraction that can be computed.
  have add_result : sub_result + newBooks = 24 := by
    sorry -- This again is a simple addition that can be computed.
  exact add_result -- Conclusion based on the obtained results.

end KalebBooks_l775_775128


namespace determine_radius_of_circle_l775_775206

noncomputable def radius_of_circle (x : ℝ) : ℝ := 
sqrt ((-7) ^ 2 + 36)

theorem determine_radius_of_circle :
  ∃ x : ℝ, (x^2 + 36 = (x - 2)^2 + 4) ∧ (radius_of_circle x = sqrt 85) :=
begin
  use -7,
  split,
  { dsimp, linarith, },
  { dsimp [radius_of_circle],  -- calculation of the radius directly
    sorry 
  }
end

end determine_radius_of_circle_l775_775206


namespace midpoint_parallel_midline_l775_775451

/-
 Given a triangle ABC with altitudes AH1, BH2, CH3. Let M be the midpoint of H2H3
 and K the intersection of line AM with H2H1. Prove that K lies on the midline
 of triangle ABC that is parallel to AC.
-/
theorem midpoint_parallel_midline {A B C H1 H2 H3 M K : Point}
(altitude_A : altitude A B C H1)
(altitude_B : altitude B A C H2)
(altitude_C : altitude C A B H3)
(midpoint_M : Midpoint M H2 H3)
(intersection_K : Intersect (Line A M) (Segment H2 H1) K) :
  OnMidlineParallel K A B C :=
sorry

end midpoint_parallel_midline_l775_775451


namespace rate_of_Y_l775_775277

noncomputable def rate_X : ℝ := 2
noncomputable def time_to_cross : ℝ := 0.5

theorem rate_of_Y (rate_Y : ℝ) : rate_X * time_to_cross = 1 → rate_Y * time_to_cross = 1 → rate_Y = rate_X :=
by
    intros h_rate_X h_rate_Y
    sorry

end rate_of_Y_l775_775277


namespace jonah_started_with_l775_775880

theorem jonah_started_with (x : ℕ) : 
  x - 1 = 11 → x = 12 :=
by {
  intro h,
  linarith,
}

end jonah_started_with_l775_775880


namespace jane_received_change_l775_775873

def cost_of_skirt : ℕ := 13
def skirts_bought : ℕ := 2
def cost_of_blouse : ℕ := 6
def blouses_bought : ℕ := 3
def amount_paid : ℕ := 100

theorem jane_received_change : 
  (amount_paid - ((cost_of_skirt * skirts_bought) + (cost_of_blouse * blouses_bought))) = 56 := 
by
  sorry

end jane_received_change_l775_775873


namespace range_of_f_l775_775392

def f (x : ℕ) : ℤ := 2 * (x : ℤ) - 3

theorem range_of_f : set.range f = {-1, 1, 3, 5, 7} :=
by 
  sorry

end range_of_f_l775_775392


namespace remainder_of_M_div_45_l775_775893

-- Given conditions as definitions
def M := "123456789101112...4950".to_nat

-- Lean statement asserting the theorem
theorem remainder_of_M_div_45 : M % 45 = 15 :=
by sorry

end remainder_of_M_div_45_l775_775893


namespace walk_back_steps_walk_back_distance_l775_775096

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m | n → m = 1 ∨ m = n

def steps (n : ℕ) : ℤ :=
  if is_prime n then 2 else -3

def total_steps : ℤ :=
  (Finset.range 20).sum_of steps

theorem walk_back_steps : total_steps = -17 :=
by
  sorry

theorem walk_back_distance : total_steps = -17 → abs total_steps = 17 :=
by
  intro h
  rw h
  norm_num

end walk_back_steps_walk_back_distance_l775_775096


namespace ral_current_age_l775_775932

variable (ral suri : ℕ)

-- Conditions
axiom age_relation : ral = 3 * suri
axiom suri_future_age : suri + 3 = 16

-- Statement
theorem ral_current_age : ral = 39 := by
  sorry

end ral_current_age_l775_775932


namespace P_investment_time_l775_775207

noncomputable def investment_in_months 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop)
  (time_Q : ℕ)
  (time_P : ℕ)
  (x : ℕ) : Prop :=
  investment_ratio_PQ 7 5 ∧ 
  profit_ratio_PQ 7 9 ∧ 
  time_Q = 9 ∧ 
  (7 * time_P) / (5 * time_Q) = 7 / 9

theorem P_investment_time 
  (investment_ratio_PQ : ℕ → ℕ → Prop)
  (profit_ratio_PQ : ℕ → ℕ → Prop) 
  (x : ℕ) : Prop :=
  ∀ (t : ℕ), investment_in_months investment_ratio_PQ profit_ratio_PQ 9 t x → t = 5

end P_investment_time_l775_775207


namespace initial_coins_l775_775183

theorem initial_coins (y : ℚ) 
  (h : y = (81*y - 1200 + 30)/81) : y = 1210 / 81 := 
by 
  sorry

end initial_coins_l775_775183


namespace inverse_function_l775_775655

noncomputable def f (x : ℝ) := 2^(x + 1)

noncomputable def f_inv (x : ℝ) := Real.log x / Real.log 2 - 1

theorem inverse_function : 
  ∀ x : ℝ, x > 0 → f (f_inv x) = x ∧ f_inv (f x) = x := 
by
  assume x hx
  sorry

end inverse_function_l775_775655


namespace total_time_correct_l775_775966

-- Declarations for given conditions
def speed_boat : ℝ := 22  -- speed of the boat in standing water (kmph)
def speed_stream : ℝ := 4  -- speed of the stream (kmph)
def distance : ℝ := 12000  -- distance to the place (km)

-- Derived speeds downstream and upstream
def speed_downstream : ℝ := speed_boat + speed_stream
def speed_upstream : ℝ := speed_boat - speed_stream

-- Time calculations for downstream and upstream
def time_downstream : ℝ := distance / speed_downstream
def time_upstream : ℝ := distance / speed_upstream

-- Total time for the round trip
def total_time : ℝ := time_downstream + time_upstream

-- Theorem to be proved
theorem total_time_correct : total_time ≈ 1128.21 :=
by
  sorry

end total_time_correct_l775_775966


namespace min_value_zero_l775_775477

noncomputable def smallest_possible_value (a b c : ℤ) (ω : ℂ) : ℝ :=
  |a + b * ω + c * ω^3|

theorem min_value_zero (a b c : ℤ) (ω : ℂ) (h1 : a ≠ b) (h2 : a ≠ c) (h3 : b ≠ c)
    (h4 : ω^4 = 1) (h5 : ω ≠ 1) : ∃ a b c : ℤ, smallest_possible_value a b c ω = 0 := by
  sorry

end min_value_zero_l775_775477


namespace pyramid_angle_l775_775541

noncomputable def theta (S V α : ℝ) : ℝ :=
  Real.arctan ((3 * V * Real.cos (α / 2)) / S * Real.sqrt (2 * Real.sin α / S))

theorem pyramid_angle (S V α : ℝ) (hS : 0 < S) (hα : 0 < α) (hV : 0 < V) :
  ∃ θ, θ = Real.arctan ((3 * V * Real.cos (α / 2)) / S * Real.sqrt (2 * Real.sin α / S)) :=
begin
  use theta S V α,
  sorry
end

end pyramid_angle_l775_775541


namespace no_good_subset_888_exists_good_subset_666_l775_775742

-- Define the size of the table
def table_size : ℕ := 32 * 32

-- Define the mouse behavior and conditions
structure MouseBehavior :=
  (move_forward : Bool)
  (turn_right : Bool)

-- Define the condition of a good subset
def is_good_subset (subset_size : ℕ) : Prop :=
  ∀ path : List MouseBehavior, 
    (path.length = subset_size + 1) → 
    (∀ (i : ℕ), i < subset_size → path[i].turn_right = tt → path[i+1].move_forward = tt) ∧
    (path.get_last no_confusion).move_forward = ff

-- Proof statement for part (a)
theorem no_good_subset_888 : ¬ is_good_subset 888 :=
  sorry

-- Proof statement for part (b)
theorem exists_good_subset_666 : ∃ subset_size : ℕ, subset_size ≥ 666 ∧ is_good_subset subset_size :=
  sorry

end no_good_subset_888_exists_good_subset_666_l775_775742


namespace exponential_inequality_l775_775036

theorem exponential_inequality (x : ℝ) (h : 2^(3-2*x) < 2^(3*x-4)) : x > 7/5 := 
by 
  sorry

end exponential_inequality_l775_775036


namespace probability_ABP_l775_775627

noncomputable def square (A B C D : Point) : Prop :=
  dist A B = dist B C ∧ dist B C = dist C D ∧ dist C D = dist D A ∧ dist A C = dist B D

def prob_tri_area_gt (A B C D P : Point) :=
  let square_ABCD := square A B C D in
  let area_ABP := triangle_area A B P in
  let area_BCP := triangle_area B C P in
  let area_CDP := triangle_area C D P in
  let area_DAP := triangle_area D A P in
  ∃ P, (P in square_ABCD) → (area_ABP > area_BCP) ∧ (area_ABP > area_CDP) ∧ (area_ABP > area_DAP)

theorem probability_ABP : ∀ (A B C D P : Point), 
  prob_tri_area_gt A B C D P → (P in square A B C D) → 
  probability (P in triangle_area A O B) = 1 / 4 := 
sorry

end probability_ABP_l775_775627


namespace sophie_aunt_money_l775_775189

noncomputable def totalMoneyGiven (shirts: ℕ) (shirtCost: ℝ) (trousers: ℕ) (trouserCost: ℝ) (additionalItems: ℕ) (additionalItemCost: ℝ) : ℝ :=
  shirts * shirtCost + trousers * trouserCost + additionalItems * additionalItemCost

theorem sophie_aunt_money : totalMoneyGiven 2 18.50 1 63 4 40 = 260 := 
by
  sorry

end sophie_aunt_money_l775_775189


namespace problem_statement_l775_775788

theorem problem_statement (m n : ℝ) :
  (m^2 - 1840 * m + 2009 = 0) → (n^2 - 1840 * n + 2009 = 0) → 
  (m^2 - 1841 * m + 2009) * (n^2 - 1841 * n + 2009) = 2009 := 
by
  intros h1 h2
  sorry

end problem_statement_l775_775788


namespace expression_is_odd_l775_775420

-- Define positive integers
def is_positive (n : ℕ) := n > 0

-- Define odd integer
def is_odd (n : ℕ) := n % 2 = 1

-- Define multiple of 3
def is_multiple_of_3 (n : ℕ) := ∃ k : ℕ, n = 3 * k

-- The Lean 4 statement to prove the problem
theorem expression_is_odd (a b c : ℕ)
  (ha : is_positive a) (hb : is_positive b) (hc : is_positive c)
  (h_odd_a : is_odd a) (h_odd_b : is_odd b) (h_mult_3_c : is_multiple_of_3 c) :
  is_odd (5^a + (b-1)^2 * c) :=
by
  sorry

end expression_is_odd_l775_775420


namespace triangle_area_l775_775111

variable {A B C a b c : ℝ}
variable {sin cos tan : ℝ → ℝ}
variable {sqrt : ℝ → ℝ}

lemma angle_B (h1 : b * sin A = 2 * sqrt 3 * a * (cos (B / 2))^2 - sqrt 3 * a) :
  B = 60 :=
  sorry

theorem triangle_area (h2 : b = 4 * sqrt 3) 
  (h3 : sin A * cos B + cos A * sin B = 2 * sin A) 
  (h4 : B = 60) :
  let a := 4 in
  let c := 2 * a in
  (1 / 2) * a * c * (sqrt 3 / 2) = 8 * sqrt 3 :=
  sorry

end triangle_area_l775_775111


namespace number_of_solutions_eq_l775_775520

theorem number_of_solutions_eq (a n : ℕ) (x y : ℕ → ℕ) :
  (∀ i, 1 ≤ i ∧ i ≤ n → x i > 0) ∧ 
  (∀ i, 1 ≤ i ∧ i ≤ n → y i ≥ 0) ∧ 
  (∑ i in Finset.range n, (i + 1) * x (i + 1)) = a  ↔ 
  (∑ i in Finset.range n, (i + 1) * y (i + 1)) = a - (n * (n + 1)) / 2 :=
begin
  sorry
end

end number_of_solutions_eq_l775_775520


namespace tangent_line_at_P_tangent_lines_through_P_tangent_lines_with_slope_4_l775_775050

noncomputable def curve (x : ℝ) : ℝ := (1/3) * x ^ 3 + (4/3)

noncomputable def derivative_curve (x : ℝ) : ℝ := x ^ 2

theorem tangent_line_at_P :
  ∀ (P : ℝ × ℝ),
  P = (2, 4) →
  (∃ (k : ℝ) (y : ℝ → ℝ), k = derivative_curve 2 ∧ k = 4 ∧ y = λ x, k * (x - 2) + 4 ∧ y 2 = 4 ∧ ∀ x, y x = 4 * x - 4) := 
by
  intros P HP
  use (4 : ℝ), (λ x, 4 * (x - 2) + 4)
  simp [derivative_curve, HP]
  sorry

theorem tangent_lines_through_P :
  ∀ (P : ℝ × ℝ),
  P = (2, 4) →
  (∃ x0, x0 = 2 ∨ x0 = -1 ∧ ∀ x, (curve x = (curve x0) + (derivative_curve x0) * (x - x0) → (x0 = 2 → (4 * x - x - 4 = 0)) ∧ (x0 = -1 → (x - x - 2 = 0)))) :=
by
  intros P HP
  use (2 : ℝ), (-1 : ℝ)
  sorry

theorem tangent_lines_with_slope_4 :
  ∀ (P : ℝ × ℝ),
  P = (2, 4) →
  (∃ x0, derivative_curve x0 = 4 ∧ (x0 = 2 ∨ x0 = -2) ∧ ∀ x, (x0 = 2 → (4 * x - x - 4 = 0)) ∧ (x0 = -2 → (12 * x - 3 * x + 20 = 0))) := 
by
  intros P HP
  use (2 : ℝ), (-2 : ℝ)
  sorry

end tangent_line_at_P_tangent_lines_through_P_tangent_lines_with_slope_4_l775_775050


namespace center_circle_sum_l775_775349

theorem center_circle_sum (x y : ℝ) (h : x^2 + y^2 = 4 * x + 10 * y - 12) : x + y = 7 := 
sorry

end center_circle_sum_l775_775349


namespace positive_solution_of_logarithmic_equation_l775_775002
open Real

theorem positive_solution_of_logarithmic_equation 
  (x : ℝ) (h₀ : 0 < x)
  (h₁ : log 3 (x - 3) + log (sqrt 3) (x^2 - 3) + log (1/3) (x - 3) = 3) :
  x = sqrt (3 + 3 * sqrt 3) :=
sorry

end positive_solution_of_logarithmic_equation_l775_775002


namespace percentage_increase_biking_time_l775_775240

theorem percentage_increase_biking_time
  (time_young_hours : ℕ)
  (distance_young_miles : ℕ)
  (time_now_hours : ℕ)
  (distance_now_miles : ℕ)
  (time_young_minutes : ℕ := time_young_hours * 60)
  (time_now_minutes : ℕ := time_now_hours * 60)
  (time_per_mile_young : ℕ := time_young_minutes / distance_young_miles)
  (time_per_mile_now : ℕ := time_now_minutes / distance_now_miles)
  (increase_in_time_per_mile : ℕ := time_per_mile_now - time_per_mile_young)
  (percentage_increase : ℕ := (increase_in_time_per_mile * 100) / time_per_mile_young) :
  percentage_increase = 100 :=
by
  -- substitution of values for conditions
  have time_young_hours := 2
  have distance_young_miles := 20
  have time_now_hours := 3
  have distance_now_miles := 15
  sorry

end percentage_increase_biking_time_l775_775240


namespace max_a3_b3_c3_d3_l775_775898

-- Define that a, b, c, d are real numbers that satisfy the given conditions.
theorem max_a3_b3_c3_d3 (a b c d : ℝ) 
  (h1 : a^2 + b^2 + c^2 + d^2 = 16)
  (h2 : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ a ∧ a ≠ c ∧ b ≠ d) :
  a^3 + b^3 + c^3 + d^3 ≤ 64 :=
sorry

end max_a3_b3_c3_d3_l775_775898


namespace BS_greater_than_CS_l775_775280

-- Variables for the vertices of the pentagon and the pyramid's apex.
variables {A B C D E F S P : Type*}

-- Conditions as hypotheses: pentagon is inscribed in a circle, 
-- and the length inequalities between the sides.
def PentagonInscribedInCircle (A B C D E F : Type*) : Prop := sorry
axiom BC_less_CD : ∀ {A B C D E F : Type*}, BC < CD (A B C D E F : ℝ)
axiom AB_less_DE : ∀ {A B C D E F : Type*}, AB < DE (A B C D E F : ℝ)
axiom AS_longest_edge : ∀ {S A B C D E F : Type*}, longest_edge S A B C D E F A

-- Hypothesis of projection:
def Projection (S A B C D E F P : Type*) : Prop := sorry

-- The theorem statement that we need to prove:
theorem BS_greater_than_CS
  (h1 : PentagonInscribedInCircle A B C D E F)
  (h2 : BC_less_CD A B C D E F)
  (h3 : AB_less_DE A B C D E F)
  (h4 : AS_longest_edge S A B C D E F)
  (h5 : Projection S A B C D E F P) :
  BS > CS (S A B C D E F) := 
sorry

end BS_greater_than_CS_l775_775280


namespace difference_of_second_largest_and_smallest_l775_775220

theorem difference_of_second_largest_and_smallest :
  let numbers := [35, 68, 57, 95] in
  let sorted := List.sort numbers in
  (sorted.nthLe 2 (by simp [sorted_length, -sorted_nthLe, length_sort]; norm_num [sorted_length])) - (sorted.nthLe 0 (by simp [sorted_length, -sorted_nthLe, length_sort]; norm_num [sorted_length])) = 33 :=
by
  sorry

end difference_of_second_largest_and_smallest_l775_775220


namespace salad_calories_l775_775460

theorem salad_calories :
  let lettuce_calories := 30
  let cucumber_calories := 80
  let crouton_calories := 20
  let number_of_croutons := 12
  in lettuce_calories + cucumber_calories + (number_of_croutons * crouton_calories) = 350 :=
by
  let lettuce_calories := 30
  let cucumber_calories := 80
  let crouton_calories := 20
  let number_of_croutons := 12
  calc
    lettuce_calories + cucumber_calories + (number_of_croutons * crouton_calories)
      = 30 + 80 + (12 * 20)  : by sorry -- just to follow the procedure, placeholder here
      = 30 + 80 + 240       : by sorry
      = 350                 : by sorry

end salad_calories_l775_775460


namespace find_c_l775_775008

theorem find_c
    (unit_squares : ℝ)
    (slanted_line  : ℝ → ℝ)
    (area_division : ℝ → Prop)
    (line_eqn      : ℝ → ℝ)
    (eq_area_2     : ℝ)
    (c_solution    : ℝ) :
    (∀ x, unit_squares = 1) →
    (∀ x, slanted_line (c:ℝ) := (x - c) * 3 / (3 - c)) →
    (∀ y, line_eqn (c: ℝ) = 3 * (3 - c) / 2 - 1) →
    (∀ a, area_division (c) → (3 * (3 - c) / 2 - 1 = 2.5)) →
    (∃ (c : ℝ), eq_area_2 = 2.5 → c_solution = 2 / 3) :=
begin
  sorry
end

end find_c_l775_775008


namespace emily_age_l775_775674

theorem emily_age
  (rachel_age_now : ℕ)
  (rachel_age_then : ℕ)
  (emily_age_then : ℕ)
  (rachel_age_now_condition : rachel_age_now = 24)
  (rachel_age_then_condition : rachel_age_then = 8)
  (emily_age_then_condition : emily_age_then = rachel_age_then / 2)
  (age_difference : ℕ := rachel_age_then - emily_age_then) :
  emily_age := rachel_age_now - age_difference :=
by {
   have emily_age_then_condition_translated : emily_age_then = 4,
     by linarith [rachel_age_then_condition, emily_age_then_condition],
   have age_difference_eqn : age_difference = 4,
     by linarith [rachel_age_then_condition, emily_age_then_condition_translated],
   show emily_age = 20,
     by linarith [rachel_age_now_condition, age_difference_eqn],
   sorry
}

end emily_age_l775_775674


namespace tenth_term_of_sequence_l775_775758

noncomputable def a : ℕ → ℕ
| 1       := 1
| (n + 1) := a n + 2

theorem tenth_term_of_sequence : a 10 = 19 := 
by sorry

end tenth_term_of_sequence_l775_775758


namespace circumference_of_smaller_circle_l775_775772

variable (R : ℝ)
variable (A_shaded : ℝ)

theorem circumference_of_smaller_circle :
  (A_shaded = (32 / π) ∧ 3 * (π * R ^ 2) - π * R ^ 2 = A_shaded) → 
  2 * π * R = 4 :=
by
  sorry

end circumference_of_smaller_circle_l775_775772


namespace O_O1_O2_O3_cyclic_l775_775409

-- Definitions and assumptions based on conditions
variable {A B C O : Point}
variable {O1 O2 O3 : Point}
variable {circle : Set Point}  -- Circle is defined by a set of points

axiom A_B_C_collinear : Collinear A B C  -- A, B, C are collinear
axiom O_not_on_line : ¬Collinear A B O  -- O is not collinear with A, B

-- O1, O2, O3 are circumcenters of respective triangles
axiom O1_circumcenter : Circumcenter O1 O A B
axiom O2_circumcenter : Circumcenter O2 O A C
axiom O3_circumcenter : Circumcenter O3 O B C

-- The theorem to prove: O, O1, O2, O3 are concyclic
theorem O_O1_O2_O3_cyclic :
  ∃ circle, circle O ∧ circle O1 ∧ circle O2 ∧ circle O3
:= sorry

end O_O1_O2_O3_cyclic_l775_775409


namespace value_of_m_l775_775418

theorem value_of_m (a a1 a2 a3 a4 a5 a6 m : ℝ) (x : ℝ)
  (h1 : (1 + m * x)^6 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6) 
  (h2 : a + a1 + a2 + a3 + a4 + a5 + a6 = 64) :
  (m = 1 ∨ m = -3) :=
sorry

end value_of_m_l775_775418


namespace int_part_of_seven_minus_sqrt_five_l775_775316

noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋ -- Define the floor function

theorem int_part_of_seven_minus_sqrt_five : floor (7 - Real.sqrt 5) = 4 := by
  -- Translate the conditions and proof problem
  have h1 : 2 < Real.sqrt 5 := Real.lt_sqrt_of_sq_lt_dec (4 : ℝ) (le_refl 4) (lt_of_le_of_ne_dec (le_refl 5) (ne_of_not_lt (by norm_num))),
  have h2 : Real.sqrt 5 < 3 := Real.sqrt_lt (by norm_num) (4 : ℝ) (by norm_num),
  have h3 : 4 < 7 - Real.sqrt 5 := sub_lt_sub_left h2 7,
  have h4 : 7 - Real.sqrt 5 < 5 := sub_lt_sub_left h1 7,
  -- Conclusion that deals with the integer part
  exact ModEq_floor.off_by_one 4 (7 - Real.sqrt 5) (And.intro h3 h4)

end int_part_of_seven_minus_sqrt_five_l775_775316


namespace find_c_plus_d_l775_775506

/-- Let \(Q\) lie on diagonal \(AC\) of square \(EFGH\) such that \(EQ > GQ\).
Let \(O_3\) and \(O_4\) be the circumcenters of triangles \(EFQ\) and \(GHQ\) respectively.
Given that \(EF = 14\) and \(\angle O_3 Q O_4 = 150^\circ\),
prove that the length of \(EQ = \sqrt{c} + \sqrt{d}\) where \(c\) and \(d\) are positive integers,
and show that \(c + d = 196\). -/
theorem find_c_plus_d :
  ∃ (c d : ℕ), 
    ∃ (Q : ℝ × ℝ), 
      Q.1^2 + Q.2^2 = 14^2 / 2 ∧ 
      ∃ (O₃ O₄ : ℝ × ℝ),
        ∃ (angle_O3_Q_O4 : ℝ),
          angle_O3_Q_O4 = 150 ∧
          c > 0 ∧ d > 0 ∧
          Q.1 + sqrt c + sqrt d = measure_of some_eq ∧ 
          c + d = 196 := 
begin
  sorry
end

end find_c_plus_d_l775_775506


namespace fourth_root_sq_eq_sixteen_l775_775998

theorem fourth_root_sq_eq_sixteen (x : ℝ) (h : (x^(1/4))^2 = 16) : x = 256 :=
sorry

end fourth_root_sq_eq_sixteen_l775_775998


namespace sum_of_first_three_terms_of_geometric_sequence_l775_775137

noncomputable def a1 : ℝ := 1/2
noncomputable def q : ℝ := (1/2)

-- Define geometric sequence
def a (n : ℕ) : ℝ := a1 * q^(n-1)

-- Statement to prove
theorem sum_of_first_three_terms_of_geometric_sequence :
  a 1 = 1/2 ∧ (a 3)^2 = a 6 → 
  (a 1 + a 2 + a 3) = 7/8 :=
by
  sorry

end sum_of_first_three_terms_of_geometric_sequence_l775_775137


namespace find_k_l775_775052

theorem find_k (k : ℝ) 
  (h1 : ∀ a b : ℝ, a + b = -k → a * b = 8 → 
    ∀ (roots_eq : (x : ℝ) → (x - (a + 3)) * (x - (b + 3)) = 0), 
    a + b + 6 = k) : k = 3 :=
begin
  sorry
end

end find_k_l775_775052


namespace gcd_18_n_eq_6_l775_775722

theorem gcd_18_n_eq_6 (num_valid_n : Nat) :
  (num_valid_n = (List.range 200).count (λ n, (1 ≤ n ∧ n ≤ 200) ∧ (6 ∣ n) ∧ ¬(9 ∣ n))) →
  num_valid_n = 22 := by
  sorry

end gcd_18_n_eq_6_l775_775722


namespace parabola_directrix_l775_775045

open Real

/--
Given the vertex of the parabola \( C \) is at the origin and the directrix has the equation \( x = 2 \).
1. Prove the equation of the parabola \( C \) is \( y^2 = -8x \).
2. Prove the length of the chord \( AB \) where the line \( l: y = x + 2 \) intersects \( C \).
-/
theorem parabola_directrix (C : ℝ → ℝ) :
  (∀ x y : ℝ, C y = y^2 / (-8 : ℝ)) ∧
  (∃ x1 x2 : ℝ, ∃ y1 y2 : ℝ, (y1 = x1 + 2 ∧ y2 = x2 + 2) ∧ 
    ((y1^2 = -8*x1) ∧ (y2^2 = -8*x2)) ∧
    (chord_length ==== (some_correct_length))) :=
  sorry

end parabola_directrix_l775_775045


namespace product_simplification_l775_775351

variables {a b c : ℝ}

theorem product_simplification (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((a + b + c)⁻¹ * (a⁻¹ + b⁻¹ + c⁻¹) * (ab + bc + ac) * ((ab)⁻¹ + (bc)⁻¹ + (ac)⁻¹)) = 
  ((ab + bc + ac)^2) / (abc) := 
sorry

end product_simplification_l775_775351


namespace tan_of_negative_55_over_6_pi_l775_775970

theorem tan_of_negative_55_over_6_pi : 
    ∀ (θ : ℝ) (n : ℤ),
    (tan (θ + n * π) = tan θ) →
    (tan (-θ) = -tan θ) →
    (tan (π / 6) = sqrt 3 / 3) →
    tan (-55 / 6 * π) = -sqrt 3 / 3 :=
by
  intros θ n h_periodic h_odd h_special
  sorry

end tan_of_negative_55_over_6_pi_l775_775970


namespace propositions_correct_l775_775642

-- Definitions for the propositions

-- Proposition ①
def is_mapping_to (f : ℝ → ℝ) (A B : set ℝ) : Prop :=
∀ x ∈ A, f x ∈ B

-- Proposition ②
def functional_equation (f : ℝ → ℝ) : Prop :=
∀ x y, f (x + y) = f x * f y ∧ f 1 ≠ 0 → f 0 = 1

-- Proposition ③
def infinitely_many_odd_even_functions : Prop :=
∃ f : ℝ → ℝ, ∃ D : set ℝ, 
  (∀ x ∈ D, f (-x) = -f x) ∧ (∀ x ∈ D, f (-x) = f x) ∧ -- odd and even
  infinite D

-- Proposition ④
def even_function_mul_pos (f : ℝ → ℝ) : Prop :=
(∀ x, f x = f (-x)) → ¬ (∀ x, f (x) * f (-x) > 0)

-- Proposition ⑤
def max_value_of_function (f : ℝ → ℝ) (M : ℝ) : Prop :=
(∀ x, f x ≤ M) → M = ⨆ x, f x

-- Main theorem statement
theorem propositions_correct :
  (is_mapping_to (fun x => x^2) {x | x > 0} set.univ) ∧
  (functional_equation (fun x => real.exp x) ∧ real.exp 1 ≠ 0 → real.exp 0 = 1) ∧
  infinitely_many_odd_even_functions ∧
  even_function_mul_pos (fun x => x^2) ∧
  max_value_of_function (fun x => x) (0) :=
by sorry

end propositions_correct_l775_775642


namespace convert_base_5_to_decimal_l775_775314

-- Define the base-5 number 44 and its decimal equivalent
def base_5_number : ℕ := 4 * 5^1 + 4 * 5^0

-- Prove that the base-5 number 44 equals 24 in decimal
theorem convert_base_5_to_decimal : base_5_number = 24 := by
  sorry

end convert_base_5_to_decimal_l775_775314


namespace coeff_x3_y6_l775_775688

noncomputable def binomial_coefficient : ℕ → ℕ → ℕ
| n 0 := 1
| 0 k := 0
| n k := (binomial_coefficient (n - 1) (k - 1)) + (binomial_coefficient (n - 1) k)

theorem coeff_x3_y6 (x y : ℂ) : 
  coeff (expand_polynomial ((x - y)^2 * (x + y)^7) (x^3 * y^6)) = 0 :=
by
  sorry

end coeff_x3_y6_l775_775688


namespace gcd_prime_exists_l775_775486

-- Define the finite nonempty set S of positive integers greater than 1
def finiteNonemptySet (S : Set ℕ) : Prop :=
  ∃ s : ℕ, s ∈ S ∧ S ≠ ∅ ∧ ∀ x ∈ S, x > 1

-- Define the property for set S
def propertyOfSet (S : Set ℕ) : Prop :=
  ∃ s ∈ S, ∀ n : ℕ, n > 0 → (gcd s n = 1 ∨ gcd s n = s)

theorem gcd_prime_exists (S : Set ℕ) (h1 : finiteNonemptySet S) (h2 : propertyOfSet S) :
  ∃ s t ∈ S, Prime (gcd s t) :=
by
  sorry

end gcd_prime_exists_l775_775486


namespace Danai_can_buy_more_decorations_l775_775315

theorem Danai_can_buy_more_decorations :
  let skulls := 12
  let broomsticks := 4
  let spiderwebs := 12
  let pumpkins := 24 -- 2 times the number of spiderwebs
  let cauldron := 1
  let planned_total := 83
  let budget_left := 10
  let current_decorations := skulls + broomsticks + spiderwebs + pumpkins + cauldron
  current_decorations = 53 → -- 12 + 4 + 12 + 24 + 1
  let additional_decorations_needed := planned_total - current_decorations
  additional_decorations_needed = 30 → -- 83 - 53
  (additional_decorations_needed - budget_left) = 20 → -- 30 - 10
  True := -- proving the statement
sorry

end Danai_can_buy_more_decorations_l775_775315


namespace velvet_needed_for_one_cloak_l775_775459

variables (hatsPerYard cloaksPerYard totalHats totalVelvet of cloaks : ℕ)

def velvetForHats (hats : ℕ) : ℕ := hats / hatsPerYard

theorem velvet_needed_for_one_cloak :
  let hatsPerYard := 4
  let totalHats := 12
  let totalVelvet := 21
  let cloaks := 6

  -- Total velvet for hats
  let velvetForHatsTotal := totalHats / hatsPerYard

  -- Remaining velvet for cloaks
  let velvetForCloaksTotal := totalVelvet - velvetForHatsTotal

  -- Velvet needed per cloak
  velvetForCloaksTotal / cloaks = 3 :=
by 
  rw [hatsPerYard, totalHats, totalVelvet, cloaks]
  -- Proof omitted
  sorry

end velvet_needed_for_one_cloak_l775_775459


namespace car_A_faster_than_car_B_l775_775659

noncomputable def car_A_speed := 
  let t_A1 := 50 / 60 -- time for the first 50 miles at 60 mph
  let t_A2 := 50 / 40 -- time for the next 50 miles at 40 mph
  let t_A := t_A1 + t_A2 -- total time for Car A
  100 / t_A -- average speed of Car A

noncomputable def car_B_speed := 
  let t_B := 1 + (1 / 4) + 1 -- total time for Car B, including a 15-minute stop
  100 / t_B -- average speed of Car B

theorem car_A_faster_than_car_B : car_A_speed > car_B_speed := 
by sorry

end car_A_faster_than_car_B_l775_775659


namespace gggg_is_odd_one_out_l775_775924

/-- Define the words and conditions -/
def is_vowel (c : Char) : Prop := 
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' 

def starts_with_vowel (s : String) : Prop :=
  s.front.isAlpha ∧ is_vowel s.front.toUpper

def first_and_last_different (s : String) : Prop :=
  s.front ≠ s.back

def not_four_letter_word (s : String) : Prop :=
  s.length ≠ 4

def is_real_word (s : String) : Prop := 
  -- This is a simplification assumption we must make, as checking real words is complex
  s ≠ "GGGG"

def is_alphabetically_ordered (s : String) : Prop :=
  s = s.toList.asArray.quicksort.toList.asString

def odd_one_out (words : List (String × (String → Prop))) : String :=
  (words.filter (λ (p : String × (String → Prop)), ¬ p.2 p.1)).head!.fst

/-- Prove 'GGGG' is the odd-one-out among the given words -/
theorem gggg_is_odd_one_out :
  odd_one_out [ ("ARFA", starts_with_vowel),
                ("BANT", first_and_last_different),
                ("VOLKODAV", not_four_letter_word),
                ("GGGG", is_real_word ∘ not),
                ("SOUS", is_alphabetically_ordered ∘ not) ] = "GGGG" :=
by sorry

end gggg_is_odd_one_out_l775_775924


namespace math_vs_english_time_difference_l775_775123

-- Definitions based on the conditions
def english_total_questions : ℕ := 30
def math_total_questions : ℕ := 15
def english_total_time_minutes : ℕ := 60 -- 1 hour = 60 minutes
def math_total_time_minutes : ℕ := 90 -- 1.5 hours = 90 minutes

noncomputable def time_per_english_question : ℕ :=
  english_total_time_minutes / english_total_questions

noncomputable def time_per_math_question : ℕ :=
  math_total_time_minutes / math_total_questions

-- Theorem based on the question and correct answer
theorem math_vs_english_time_difference :
  (time_per_math_question - time_per_english_question) = 4 :=
by
  -- Proof here
  sorry

end math_vs_english_time_difference_l775_775123


namespace projectile_height_at_time_l775_775196

theorem projectile_height_at_time
  (y : ℝ)
  (t : ℝ)
  (h_eq : y = -16 * t ^ 2 + 64 * t) :
  ∃ t₀ : ℝ, t₀ = 3 ∧ y = 49 :=
by sorry

end projectile_height_at_time_l775_775196


namespace shaded_region_area_l775_775104

/-
Define the problem:
- Three rectangles with specified dimensions.
- A triangle with a specific base and height.
- Prove that the area of the shaded region is 32.
-/

theorem shaded_region_area :
  let area_rectangle_1 := 5 * 4 in
  let area_rectangle_2 := 6 * 6 in
  let area_rectangle_3 := 5 * 8 in
  let total_grid_area := area_rectangle_1 + area_rectangle_2 + area_rectangle_3 in
  let base_triangle := 16 in
  let height_triangle := 8 in
  let area_triangle := (base_triangle * height_triangle) / 2 in
  let shaded_area := total_grid_area - area_triangle in
  shaded_area = 32 :=
by
  sorry

end shaded_region_area_l775_775104


namespace range_of_a_l775_775770

variable (x a : ℝ)

def p : Prop := (1 / 2 ≤ x ∧ x ≤ 1)
def q : Prop := ((x - a) * (x - a - 1) > 0)

theorem range_of_a (h : p x a → ¬ q x a) (h' : ¬ (¬ p x a) → ¬ q x a) : 0 ≤ a ∧ a ≤ 1 / 2 := by
  sorry

end range_of_a_l775_775770


namespace find_k_minus_r_l775_775977

theorem find_k_minus_r : 
  ∃ (k r : ℕ), k > 1 ∧ r < k ∧ 
  (1177 % k = r) ∧ (1573 % k = r) ∧ (2552 % k = r) ∧ 
  (k - r = 11) :=
sorry

end find_k_minus_r_l775_775977


namespace number_of_a_values_l775_775730

theorem number_of_a_values (a : ℝ) :
  (∃ x : ℝ, y = x + 2*a ∧ y = x^3 - 3*a*x + a^3) → a = 0 :=
by
  sorry

end number_of_a_values_l775_775730


namespace three_digit_factorions_l775_775278

def is_factorion (n : ℕ) : Prop :=
  let digits := (n / 100, (n % 100) / 10, n % 10)
  let (a, b, c) := digits
  n = Nat.factorial a + Nat.factorial b + Nat.factorial c

theorem three_digit_factorions : ∀ n : ℕ, (100 ≤ n ∧ n < 1000) → is_factorion n → n = 145 :=
by
  sorry

end three_digit_factorions_l775_775278


namespace area_swept_by_centroid_l775_775915

-- Lean 4 statement for the problem
theorem area_swept_by_centroid (A B O C G : Point) (r : ℝ) 
  (hAB : distance A B = 30)
  (hO : O = midpoint A B)
  (h_radius : r = 15)
  (hC : C lies on semicircle with diameter O)
  (hG : G = centroid A B C):
  (area_swept_by_centroid G) = (25 / 2) * π := 
sorry

end area_swept_by_centroid_l775_775915


namespace angle_bisector_proof_l775_775616

open_locale real_inner_product_space
noncomputable theory

variables {O O' E P A B : Type*}

-- Assume the circles are internally tangent and touch at E
variables (touching : tangent_circles O O' E)

-- Assume P is a point on the smaller circle, A and B are points on the larger circle where the tangent at P intersects
variables (on_smaller_circle : point_on_circle P O')
          (tangent_line_intersects : points_on_larger_circle A B O)

-- Definition of the angle bisector relationship
def angle_bisector (center : O) (P A B : Type*) : Prop :=
  let EP := line_through E P in
  let \[\angle AEB\] := angle A E B in
  is_angle_bisector EP \[\angle AEB\]

-- Prove EP bisects the angle AEB
theorem angle_bisector_proof :
  angle_bisector E P A B :=
sorry

end angle_bisector_proof_l775_775616


namespace largest_integer_l775_775421

def bin_op (n : ℤ) : ℤ := n - 5 * n

theorem largest_integer (n : ℤ) (h : 0 < n) (h' : bin_op n < 18) : n = 4 := sorry

end largest_integer_l775_775421


namespace ratio_arithmetic_progression_l775_775846

theorem ratio_arithmetic_progression (a d : ℕ) 
  (h1 : a ≠ 0) 
  (h2 : d ≠ 0) 
  (h3 : 15 * (2 * a + 14 * d) = 3 * 8 * (2 * a + 7 * d)) :
  a / d = 7 / 3 :=
begin
  -- Proof is not needed as per instructions.
  sorry
end

end ratio_arithmetic_progression_l775_775846


namespace f_nx_eq_Pn_fx_l775_775778

noncomputable def f (x : ℝ) : ℝ :=
  (Real.exp x + Real.exp (-x)) / 2

theorem f_nx_eq_Pn_fx (n : ℕ) (x : ℝ) :
  ∃ P_n : Polynomial ℤ, f (n * x) = Polynomial.eval (f x) P_n ∧ Polynomial.degree P_n = n :=
begin
  sorry
end

end f_nx_eq_Pn_fx_l775_775778


namespace base_4_representation_156_l775_775579

theorem base_4_representation_156 :
  ∃ b3 b2 b1 b0 : ℕ,
    156 = b3 * 4^3 + b2 * 4^2 + b1 * 4^1 + b0 * 4^0 ∧
    b3 = 2 ∧ b2 = 1 ∧ b1 = 3 ∧ b0 = 0 :=
by
  have h1 : 156 = 2 * 4^3 + 28 := by norm_num
  have h2 : 28 = 1 * 4^2 + 12 := by norm_num
  have h3 : 12 = 3 * 4^1 + 0 := by norm_num
  refine ⟨2, 1, 3, 0, _, rfl, rfl, rfl, rfl⟩
  rw [h1, h2, h3]
  norm_num

end base_4_representation_156_l775_775579


namespace flag_height_l775_775650

theorem flag_height
  (d1 d2 d3 : ℕ × ℕ)
  (length : ℕ)
  (area1 : d1 = (8, 5))
  (area2 : d2 = (10, 7))
  (area3 : d3 = (5, 5))
  (flag_length : length = 15)
  : (d1.1 * d1.2 + d2.1 * d2.2 + d3.1 * d3.2) / length = 9 :=
by
  have h1 : d1.1 * d1.2 = 8 * 5, from congr_arg (*) area1,
  have h2 : d2.1 * d2.2 = 10 * 7, from congr_arg (*) area2,
  have h3 : d3.1 * d3.2 = 5 * 5, from congr_arg (*) area3,
  simp [flag_length] at *,
  sorry

end flag_height_l775_775650


namespace local_minimum_f_eval_integral_part_f_l775_775882

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sin x * Real.sqrt (1 - Real.cos x))

theorem local_minimum_f :
  (0 < x) -> (x < π) -> f x >= 1 :=
  by sorry

theorem eval_integral_part_f :
  ∫ x in (↑(π / 2))..(↑(2 * π / 3)), f x = sorry :=
  by sorry

end local_minimum_f_eval_integral_part_f_l775_775882


namespace sum_of_angles_sphere_inscribed_in_pyramid_l775_775287

-- Define the pyramid SABC and the points where the inscribed sphere touches
variable (S A B C D E F : Type)
variable [SABC_pyramid : is_pyramid S A B C]
variable [inscribed_sphere_touches_faces : 
  (sphere_touches_face S A B D) ∧ 
  (sphere_touches_face S B C E) ∧
  (sphere_touches_face S C A F)]

-- Define the angles formed at the touch points
variable (angle_SDA : ℝ)
variable (angle_SEB : ℝ)
variable (angle_SFC : ℝ)
variable [has_angle : has_angle S D A angle_SDA]
variable [has_angle : has_angle S E B angle_SEB]
variable [has_angle : has_angle S F C angle_SFC]

-- State the theorem
theorem sum_of_angles_sphere_inscribed_in_pyramid : 
  angle_SDA + angle_SEB + angle_SFC = 360 :=
sorry

end sum_of_angles_sphere_inscribed_in_pyramid_l775_775287


namespace problem_statement_l775_775234

theorem problem_statement (a : ℤ) (n m : ℕ) (h1 : n % 2 = 0 → (-a)^n = a^n)
  (h2 : n % 2 = 1 → (-a)^n = -a^n) (h3 : (a^m)^n = a^(m*n))
  (h4 : (1 : ℤ) / (a^n) = a^(-n)) :
  (1 : ℤ) / (-5^2)^4 * (-5)^9 = -5 :=
by
  sorry

end problem_statement_l775_775234


namespace altitudes_eq_cosine_l775_775831

theorem altitudes_eq_cosine (a A : ℝ) (EF : ℝ)
  (h₁ : ∃ (B C : ℝ), ∃ (BE CF : ℝ), are_altitudes BE CF a A B C) : 
  EF = a * Real.cos A :=
sorry

-- Auxiliary definition to express the concept of altitudes
def are_altitudes (BE CF a A B C : ℝ) : Prop :=
  ∃ (E F : ℝ), BE = a * Real.sin C ∧ CF = a * Real.cos B ∧ EF = a * Real.sin C * Real.cos A

end altitudes_eq_cosine_l775_775831


namespace negate_proposition_p_l775_775114

theorem negate_proposition_p (f : ℝ → ℝ) :
  (¬ ∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) >= 0) ↔ ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
sorry

end negate_proposition_p_l775_775114


namespace jill_total_earnings_l775_775879

def hourly_wage := 4.00
def tip_rate := 0.15
def shifts := 3
def hours_per_shift := 8
def avg_orders_per_hour := 40.00

theorem jill_total_earnings :
  let total_hours := shifts * hours_per_shift,
      wage_earnings := total_hours * hourly_wage,
      total_orders := total_hours * avg_orders_per_hour,
      tip_earnings := tip_rate * total_orders,
      total_earnings := wage_earnings + tip_earnings
  in total_earnings = 240.00 :=
by {
  let total_hours := shifts * hours_per_shift,
  let wage_earnings := total_hours * hourly_wage,
  let total_orders := total_hours * avg_orders_per_hour,
  let tip_earnings := tip_rate * total_orders,
  let total_earnings := wage_earnings + tip_earnings,
  show total_earnings = 240.00, from sorry
}

end jill_total_earnings_l775_775879


namespace area_difference_10_l775_775445

noncomputable def area (a b : ℝ) : ℝ := 1/2 * a * b

theorem area_difference_10
  (angle_EAB_right : ∀ (A E B : ℝ), ∠ EAB = 90)
  (angle_ABC_right : ∀ (A B C : ℝ), ∠ ABC = 90)
  (AB : ℝ)
  (BC : ℝ)
  (AE : ℝ)
  (AC_BE_intersect_D : ∃ D, LineThrough점 (AC) ∩ LineThrough점 (BE) = {D})
  (h_AB : AB = 5)
  (h_BC : BC = 3)
  (h_AE : AE = 7)
  : area AE AB - area AB BC = 10 :=
by
  sorry

end area_difference_10_l775_775445


namespace solve_equation_l775_775601

noncomputable def heaviside (x : ℝ) : ℝ :=
  if x >= 0 then 1 else 0

theorem solve_equation (x : ℝ) (k : ℤ) :
  (Real.cot (2 * x * (heaviside (x + 3 * Real.pi) - heaviside (x - 6 * Real.pi))) = 1 / Real.sin x - 1 / Real.cos x) →
  x = Real.pi / 4 + Real.pi * k ∧ -3 ≤ k ∧ k ≤ 5 :=
by
  sorry

end solve_equation_l775_775601


namespace find_ω_l775_775054

noncomputable def f (ω φ x : ℝ) : ℝ :=
  2 * Real.sin (ω * x + φ)

noncomputable def ω_range (ω : ℝ) : Prop :=
  (8 / 3) < ω ∧ ω ≤ (30 / 11)

theorem find_ω (ω : ℝ) (φ : ℝ) 
  (cond1 : ω > 0)
  (cond2 : ∀ x ∈ Set.Ioo (7 * Real.pi / 12) (51 * Real.pi / 60), Monotone (λ x, f ω φ x))
  (cond3 : f ω φ (7 * Real.pi / 12) = - f ω φ (3 * Real.pi / 4))
  (cond4 : ∃! zs, zs = {x | x ∈ Set.Ico (2 * Real.pi / 3) (13 * Real.pi / 6) ∧ f ω φ x = 0} ∧ zs.card = 5) : ω_range ω :=
sorry

end find_ω_l775_775054


namespace equivalent_mod_l775_775765

theorem equivalent_mod (h : 5^300 ≡ 1 [MOD 1250]) : 5^9000 ≡ 1 [MOD 1000] :=
by 
  sorry

end equivalent_mod_l775_775765


namespace complement_set_l775_775795

theorem complement_set (U M: Set ℕ) (hU: U = {1, 2, 3, 4, 5, 6, 7}) 
    (hM: M = {x | x^2 - 6 * x + 5 ≤ 0 ∧ x ∈ Set.univ ℤ}) : 
    U \ M = {6, 7} := by
  sorry

end complement_set_l775_775795


namespace find_a_find_AB_l775_775767

noncomputable def ellipse := { a : ℝ // a > 1 }

variables (a : ellipse)
def f1 := (-a, 0)
def f2 := (a, 0)
def ellipse_eq (x y : ℝ) := (x^2)/(a.val^2) + (y^2)/(a.val^2 - 1) = 1
def line (angle : ℝ) (x y : ℝ) := y = x + 1
def triangle_perimeter (|AF1| |AF2| |BF1| : ℝ) := |AF1| + |AF2| + |BF1|

theorem find_a (h : triangle_perimeter 2a = 8) : a.val = 2 :=
sorry

theorem find_AB
  (a_val : a.val = 2)
  (h : triangle_perimeter 2 * (f1, f2, A, B) = 8)
  (angle : ℝ)
  (h1 : angle = π/4):
  |AB| = 24/7 :=
sorry

end find_a_find_AB_l775_775767


namespace largest_sum_is_1173_l775_775990

def largest_sum_of_two_3digit_numbers : Prop :=
  ∃ a b c d e f : ℕ, 
  (a = 6 ∧ b = 5 ∧ c = 4 ∧ d = 3 ∧ e = 2 ∧ f = 1) ∧
  100 * (a + b) + 10 * (c + d) + (e + f) = 1173

theorem largest_sum_is_1173 : largest_sum_of_two_3digit_numbers :=
  by
  sorry

end largest_sum_is_1173_l775_775990


namespace participants_begin_competition_l775_775843

theorem participants_begin_competition (x : ℝ) 
  (h1 : 0.4 * x * (1 / 4) = 16) : 
  x = 160 := 
by
  sorry

end participants_begin_competition_l775_775843


namespace repack_books_l775_775839

theorem repack_books :
  let num_boxes := 1573
      books_per_box := 42
      books_per_new_box := 45 in
  let total_books := num_boxes * books_per_box in
  total_books % books_per_new_box = 6 :=
by
  let num_boxes := 1573
  let books_per_box := 42
  let books_per_new_box := 45
  let total_books := num_boxes * books_per_box
  show total_books % books_per_new_box = 6
  sorry

end repack_books_l775_775839


namespace allocation_schemes_l775_775707

theorem allocation_schemes : ∃ n : ℕ, 
  let volunteers := 5 in 
  let projects := 4 in
  (∀ v, v < volunteers → (∃ p, p < projects ∧ p.volunteer = v)) ∧ 
  (∀ p, p < projects → (∃ v, v < volunteers ∧ v.project = p)) → 
  n = 240 :=
begin
  sorry
end

end allocation_schemes_l775_775707


namespace milan_billed_minutes_l775_775716

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l775_775716


namespace max_cone_volume_l775_775641

theorem max_cone_volume {c : ℝ} (hc : 0 < c) :
  ∃ (x y : ℝ), (x^2 + y^2 = c^2) ∧ 
               (∃ h : ℝ, h = c / (Real.sqrt 3) ∧ y = c * (Real.sqrt (2 / 3)) ∧ 
               (1 / 3 * Real.pi * y^2 * h = (2 * Real.pi * Real.sqrt 3 * c^2) / 27)) :=
begin
  sorry
end

end max_cone_volume_l775_775641


namespace percentage_of_oysters_with_pearls_l775_775270

def jamie_collects_oysters (oysters_per_dive dives total_pearls : ℕ) : ℕ :=
  oysters_per_dive * dives

def percentage_with_pearls (total_pearls total_oysters : ℕ) : ℕ :=
  (total_pearls * 100) / total_oysters

theorem percentage_of_oysters_with_pearls :
  ∀ (oysters_per_dive dives total_pearls : ℕ),
  oysters_per_dive = 16 →
  dives = 14 →
  total_pearls = 56 →
  percentage_with_pearls total_pearls (jamie_collects_oysters oysters_per_dive dives total_pearls) = 25 :=
by
  intros
  sorry

end percentage_of_oysters_with_pearls_l775_775270


namespace cosine_between_diagonals_l775_775279

def vector1 : ℝ × ℝ × ℝ := (3, 2, 1)
def vector2 : ℝ × ℝ × ℝ := (2, -1, -2)

def diag1 : ℝ × ℝ × ℝ := (vector1.1 + vector2.1, vector1.2 + vector2.2, vector1.3 + vector2.3)
def diag2 : ℝ × ℝ × ℝ := (vector1.1 - vector2.1, vector1.2 - vector2.2, vector1.3 - vector2.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def norm (v : ℝ × ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2 + v.3^2)

theorem cosine_between_diagonals :
  let cos_theta := (dot_product diag1 diag2) / ((norm diag1) * (norm diag2))
  cos_theta = 5 / Real.sqrt 513 :=
by
  sorry

end cosine_between_diagonals_l775_775279


namespace find_length_of_segment_AC_l775_775661

noncomputable def length_of_ac (C : Circle) (A B : Point) (r : ℝ)
  (h1 : C.circumference = 18 * real.pi)
  (h2 : C.diameter = segment AB)
  (h3 : angle C A C = 45) :
  ℝ :=
  18 * real.sin (22.5 * real.pi / 180)

theorem find_length_of_segment_AC
  (C : Circle)
  (A B : Point)
  (r : ℝ)
  (h1 : C.circumference = 18 * real.pi)
  (h2 : C.diameter = segment AB)
  (h3 : angle (join A C) (join A B) = 45) 
  :
  length_of_ac C A B r h1 h2 h3 = 18 * real.sin (22.5 * real.pi / 180) :=
sorry

end find_length_of_segment_AC_l775_775661


namespace clever_functions_l775_775745

-- Define what it means to be a clever value point
def clever_value_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀, f x₀ = (has_deriv_at f (f x₀) x₀).deriv

-- Define the functions to be checked
def f1 (x : ℝ) : ℝ := x^2
def f2 (x : ℝ) : ℝ := 1 / exp x
def f3 (x : ℝ) : ℝ := log x
def f4 (x : ℝ) : ℝ := tan x
def f5 (x : ℝ) : ℝ := x + (1 / x)

-- Prove exactly the specified functions have clever value points
theorem clever_functions : 
  (clever_value_point f1 ∧ clever_value_point f3 ∧ clever_value_point f5) ∧
  (¬ clever_value_point f2) ∧ (¬ clever_value_point f4) :=
by
  sorry

end clever_functions_l775_775745


namespace cameron_typing_speed_l775_775647

theorem cameron_typing_speed (w: ℕ) (h1: wpm_aft: ℕ) (h2: diff: ℕ) :
  wpm_aft = 8 →
  diff = 10 →
  5 * w - 5 * wpm_aft = diff →
  w = 10 := by
  intros h1 h2 h3
  sorry

end cameron_typing_speed_l775_775647


namespace least_three_digit_with_factors_l775_775993

theorem least_three_digit_with_factors (n : ℕ) :
  (n ≥ 100 ∧ n < 1000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 7 ∣ n ∧ 3 ∣ n) → n = 210 := by
  sorry

end least_three_digit_with_factors_l775_775993


namespace max_f_value_l775_775783

def f (x : ℝ) : ℝ := cos (π / 2 + x) + sin (π / 2 + x)^2

theorem max_f_value : ∀ x ∈ set.Icc (-π) 0, f x ≤ (5 / 4) ∧ ∃ x ∈ set.Icc (-π) 0, f x = (5 / 4) := sorry

end max_f_value_l775_775783


namespace find_DF_l775_775452

variables {D E F N : Type} [metric_space D] [metric_space E] [metric_space F]
variable [metric_space N]

def DE : ℝ := 7
def EF : ℝ := 9
def DN : ℝ := 9 / 2

theorem find_DF (DF : ℝ) (h1 : dist D E = DE) (h2 : dist E F = EF) (h3 : dist D N = DN) : DF = Real.sqrt 130 :=
by
  -- Proof goes here
  sorry

end find_DF_l775_775452


namespace calculate_expression_l775_775658

theorem calculate_expression :
  (∛(27 : ℝ) - |(1 : ℝ) - real.sqrt 3| + 2 * real.sqrt 3 = 4 + real.sqrt 3) :=
by
  sorry

end calculate_expression_l775_775658


namespace Wendy_uploaded_pictures_l775_775232

theorem Wendy_uploaded_pictures : 
  ∀ (album1_pics : ℕ) (num_albums : ℕ) (pics_per_album : ℕ),
  album1_pics = 27 →
  num_albums = 9 →
  pics_per_album = 2 →
  album1_pics + num_albums * pics_per_album = 45 := 
by
  intros album1_pics num_albums pics_per_album h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end Wendy_uploaded_pictures_l775_775232


namespace ZP_passing_through_fixed_point_l775_775024

variable {α : Type*}
variables {A B C P I₁ I₂ Z : α}

-- Assuming the basic structure of triangle and points
variables (ABC_circumcircle : Set α)  -- Circumcircle of triangle ABC
variables (hABC : Triangle ABC_circumcircle A B C)  -- Triangle ABC assumed to exist on the circumcircle
variables (P_move : α)  -- Point P moving along the circumcircle
variables (P_on_circumcircle : P ∈ ABC_circumcircle)  -- P always on the circumcircle

-- The definitions and conditions
def intersects_chord (AP BC : Set α) : Prop :=
  exists (X : α), X ∈ AP ∧ X ∈ BC  -- Intersection of lines AP and BC

variables (AP BC : Set α)  -- the lines AP and BC
variables (I₁_circle : inscribed_circle (Triangle BCP) I₁)
variables (I₂_circle : inscribed_circle (Triangle APC) I₂)
variables (I₁I₂_line : line_through I₁ I₂)

theorem ZP_passing_through_fixed_point (h_intersect : intersects_chord AP BC) :
  ∃ (F : α), ∀ (P : α), Z = inter (I₁I₂_line) (BC) → line_through Z P F :=
sorry

end ZP_passing_through_fixed_point_l775_775024


namespace find_x_l775_775018

theorem find_x (x y : ℕ) (h1 : y = 30) (h2 : x / y = 5 / 2) : x = 75 := by
  sorry

end find_x_l775_775018


namespace marble_problem_l775_775297

theorem marble_problem (a : ℚ) :
  let brian := 3 * a - 4 in
  let caden := 2 * brian + 2 in
  let daryl := 4 * caden in
  a + brian + caden + daryl = 122 → a = 78 / 17 := 
by 
  intros 
  unfold brian caden daryl 
  sorry

end marble_problem_l775_775297


namespace range_of_a_l775_775781

def f (x a : ℝ) : ℝ := x + a / x

theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ m n p ∈ set.Icc (1 / 3) 1, f m a + f n a > f p a ∧ 
    f m a + f p a > f n a ∧ f n a + f p a > f m a) ↔ 
  a ∈ set.Ioo (1 / 15) (1 / 9) ∪ set.Ico 1 (5 / 3) :=
sorry

end range_of_a_l775_775781


namespace knights_seating_arrangement_l775_775301

open Set

-- Define the knight problem
def knight_problem (n : ℕ) : Prop :=
  ∃ (seating : Fin (2 * n) → Fin (2 * n)),
    (∀ (i : Fin (2 * n)), 
      let k := (seating i) in
      let k' := (seating ((i + 1) % (2 * n))) in
      k ≠ k' ∧ ∀ (enemy : Fin (2 * n)), enemy ≠ k' → enemy ≠ k)

-- Define a round seating arrangement proof for knights
theorem knights_seating_arrangement (n : ℕ) (knights : Fin (2 * n) → Finset (Fin (2 * n))) 
  (h : ∀ k, (knights k).card ≤ n - 1) :
  knight_problem n :=
  sorry

end knights_seating_arrangement_l775_775301


namespace sum_of_edges_of_geometric_progression_solid_l775_775286

theorem sum_of_edges_of_geometric_progression_solid
  (a : ℝ)
  (r : ℝ)
  (volume_eq : a^3 = 512)
  (surface_eq : 2 * (64 / r + 64 * r + 64) = 352)
  (r_value : r = 1.25 ∨ r = 0.8) :
  4 * (8 / r + 8 + 8 * r) = 97.6 := by
  sorry

end sum_of_edges_of_geometric_progression_solid_l775_775286


namespace mode_of_dataset_l775_775954

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end mode_of_dataset_l775_775954


namespace b_seq_inequality_l775_775754

-- Define the sequence a_n
def a_seq (n : ℕ) : ℕ := if n = 0 then 2 else 3 * 2^(n-1) - a_seq (n-1)

-- The auxiliary sequence b_n
def b_seq (n : ℕ) : ℚ := (a_seq n + 1) / (a_seq n - 1)

-- Prove that a_n = 2^n.
lemma seq_is_geometric : ∀ n : ℕ, a_seq n = 2^n :=
by sorry

-- Prove the inequality for b_n
theorem b_seq_inequality (n : ℕ) : (∑ k in finset.range n, b_seq (k+1)) < n + 4 :=
by sorry

end b_seq_inequality_l775_775754


namespace sum_of_arithmetic_sequence_l775_775044

variable {α : Type*} [LinearOrderedField α]

def sum_arithmetic_sequence (a₁ d : α) (n : ℕ) : α :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_arithmetic_sequence {a₁ d : α}
  (h₁ : sum_arithmetic_sequence a₁ d 10 = 12) :
  (a₁ + 4 * d) + (a₁ + 5 * d) = 12 / 5 :=
by
  sorry

end sum_of_arithmetic_sequence_l775_775044


namespace machines_900_bottles_4_minutes_l775_775173

def bottles_per_minute_per_machine (total_bottles : ℕ) (machines : ℕ) : ℕ :=
  total_bottles / machines

def time_to_produce (bottle_goal : ℕ) (production_rate : ℕ) : ℕ :=
  bottle_goal / production_rate

theorem machines_900_bottles_4_minutes :
  (b : ℕ) (m6 : ℕ) (m5 : ℕ) (bpm : ℕ) (r : ℕ) : b = 900 → m6 = 6 → m5 = 5 → bpm = 270 → 
  r = bottles_per_minute_per_machine bpm m6 * m5 → time_to_produce b r = 4 := by
  intros
  sorry

end machines_900_bottles_4_minutes_l775_775173


namespace find_b_l775_775448

-- Defining the quadratic equation and its properties
def quadratic_eq (b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, (x₂ * x₁ = 20) ∧ (x₁ + x₂ = 12) ∧ (∀ x, x² - b * x + 20 = 0 → (x = x₁) ∨ (x = x₂))

-- Main statement to prove that b = -12 under the given conditions
theorem find_b (b : ℝ) : quadratic_eq b → b = -12 :=
sorry

end find_b_l775_775448


namespace interval_of_monotonicity_value_of_a_when_maximum_l775_775396

noncomputable def f (a x : ℝ) : ℝ := (1/3)^((a * x^2) - 4 * x + 3)

theorem interval_of_monotonicity (f : ℝ → ℝ) (a : ℝ) (hx : a = -1) :
  (∀ x, (1/3)^(-x^2 - 4 * x + 3) = f x) →
  ∃ I1 I2 : Set ℝ, I1 = Set.Ioo (-∞) (-2 : ℝ) ∧ I2 = Set.Ioo (-2 : ℝ) (∞),
  StrictAntiOn f I1 ∧ StrictMonoOn f I2 := sorry

theorem value_of_a_when_maximum (f : ℝ → ℝ) (h_max : ∃ x, f x = 3) :
  (∀ x, (1/3)^((a * x^2) - 4 * x + 3) = f x) →
  ∃ a : ℝ, a = 1 := sorry

end interval_of_monotonicity_value_of_a_when_maximum_l775_775396


namespace train_crossing_time_l775_775633

-- Definitions corresponding to the problem conditions
def train_speed_kmh : ℝ := 90
def train_length_m : ℝ := 250
def conversion_factor : ℝ := 1000 / 3600  -- Conversion factor from km/hr to m/s

-- The problem statement that requires proof
theorem train_crossing_time :
  (train_length_m / (train_speed_kmh * conversion_factor)) = 10 :=
by
  sorry

end train_crossing_time_l775_775633


namespace solve_equation_l775_775938

theorem solve_equation (x : ℝ) :
  x^2 - 2 * |x - 1| - 2 = 0 → (x = 2 ∨ x = -1 - real.sqrt 5) :=
by
  sorry

end solve_equation_l775_775938


namespace joan_carrots_grown_correct_l775_775118

variable (total_carrots : ℕ) (jessica_carrots : ℕ) (joan_carrots : ℕ)

theorem joan_carrots_grown_correct (h1 : total_carrots = 40) (h2 : jessica_carrots = 11) (h3 : total_carrots = joan_carrots + jessica_carrots) : joan_carrots = 29 :=
by
  sorry

end joan_carrots_grown_correct_l775_775118


namespace water_level_rise_large_pool_l775_775567

theorem water_level_rise_large_pool
  (side_large side_medium side_small : ℝ)
  (rise_medium rise_small : ℝ)
  (h_large : side_large = 6)
  (h_medium : side_medium = 3)
  (h_small : side_small = 2)
  (h_rise_medium : rise_medium = 0.06)
  (h_rise_small : rise_small = 0.04) :
  let volume_medium := side_medium ^ 2 * rise_medium in
  let volume_small := side_small ^ 2 * rise_small in
  let total_volume := volume_medium + volume_small in
  let surface_area_large := side_large ^ 2 in
  let water_level_increase := total_volume / surface_area_large in
  water_level_increase = 35 / 18 :=
by
  sorry

end water_level_rise_large_pool_l775_775567


namespace problem1_problem2_l775_775857

-- Definitions for the problem

/-- Definition of point P in Cartesian coordinate system -/
def P (x : ℝ) : ℝ × ℝ :=
  (x - 2, x)

-- First proof problem statement
theorem problem1 (x : ℝ) (h : (x - 2) * x < 0) : x = 1 :=
sorry

-- Second proof problem statement
theorem problem2 (x : ℝ) (h1 : x - 2 < 0) (h2 : x > 0) : 0 < x ∧ x < 2 :=
sorry

end problem1_problem2_l775_775857


namespace gcd_459_357_l775_775608

theorem gcd_459_357 : Nat.gcd 459 357 = 51 :=
by
  sorry

end gcd_459_357_l775_775608


namespace problem_prove_divisibility_l775_775930

theorem problem_prove_divisibility (n : ℕ) : 11 ∣ (5^(2*n) + 3^(n+2) + 3^n) :=
sorry

end problem_prove_divisibility_l775_775930


namespace alani_income_goal_l775_775293

theorem alani_income_goal (h1_rate : 15) (h2_rate : 15) (h3_rate : 15)
  (h1_hours : ℕ) (h2_hours : ℕ) (h3_hours : ℕ)
  (income_goal : 375)
  (w1_rate : 45 / 3 = h1_rate)
  (w2_rate : 90 / 6 = h2_rate)
  (w3_rate : 30 / 2 = h3_rate) :
  (h1_hours * h1_rate) + (h2_hours * h2_rate) + (h3_hours * h3_rate) = income_goal →
  (h1_hours + h2_hours + h3_hours) = 25 :=
by
  sorry

end alani_income_goal_l775_775293


namespace count_congruent_to_2_mod_7_l775_775816

theorem count_congruent_to_2_mod_7 : 
  let count := (1 to 300).count (λ x, x % 7 = 2)
  count = 43 := by
  sorry

end count_congruent_to_2_mod_7_l775_775816


namespace line_through_points_on_parabola_l775_775373

theorem line_through_points_on_parabola
  (p q : ℝ)
  (hpq : p^2 - 4 * q > 0) :
  ∃ (A B : ℝ × ℝ),
    (exists (x₁ x₂ : ℝ), x₁^2 + p * x₁ + q = 0 ∧ x₂^2 + p * x₂ + q = 0 ∧
                         A = (x₁, x₁^2 / 3) ∧ B = (x₂, x₂^2 / 3) ∧
                         (∀ x y, (x, y) = A ∨ (x, y) = B → px + 3 * y + q = 0)) :=
sorry

end line_through_points_on_parabola_l775_775373


namespace gcd_18_eq_6_l775_775726

theorem gcd_18_eq_6 {n : ℕ} (hn : 1 ≤ n ∧ n ≤ 200) : (nat.gcd 18 n = 6) ↔ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ 22 ∧ n = 6 * k ∧ ¬(n = 18 * (n / 18))) :=
begin
  sorry
end

end gcd_18_eq_6_l775_775726


namespace circumcenter_orthocenter_distance_eq_l775_775299

open EuclideanGeometry

-- Define points and triangle
variables (A B C D O H E F O' : Point)

-- Statement adapted to Lean
theorem circumcenter_orthocenter_distance_eq 
  (triangle_ABC : Triangle A B C)
  (D_on_BC : PointOnLineSegment D B C)
  (O_ABC : Circumcenter O A B C)
  (H_ABC : Orthocenter H A B C)
  (E_on_AB : PointOnLine E A B)
  (F_on_AC : PointOnLine F A C)
  (E_perp_BD : PerpendicularBisector E B D)
  (F_perp_CD : PerpendicularBisector F C D)
  (O'_DEF : Circumcenter O' D E F) : 
  dist O H = dist O O' :=
sorry

end circumcenter_orthocenter_distance_eq_l775_775299


namespace variance_of_temperatures_l775_775402

-- Define the temperatures
def temperatures : List ℝ := [28, 29, 25, 25, 28]

-- Define the average temperature
def average_temperature (temps : List ℝ) : ℝ :=
  (temps.sum) / (temps.length)

-- Define the variance calculation
def variance (temps : List ℝ) : ℝ :=
  let avg := average_temperature temps
  (temps.foldl (λ acc x => acc + (x - avg) ^ 2) 0) / (temps.length)

-- Define the main statement requiring proof
theorem variance_of_temperatures : variance temperatures = 14 / 5 := sorry

end variance_of_temperatures_l775_775402


namespace rate_per_square_meter_l775_775951

/--

Given:
- The length of the room is 5.5 meters.
- The width of the room is 3.75 meters.
- The total cost of paving the floor is Rs. 16500.

Prove:
- The rate per square meter for paving the floor is Rs. 800.

--/

theorem rate_per_square_meter
  (length : ℝ) (width : ℝ) (total_cost : ℝ)
  (h_length : length = 5.5)
  (h_width : width = 3.75)
  (h_total_cost : total_cost = 16500) :
  let area := length * width in
  let rate_per_sqm := total_cost / area in
  rate_per_sqm = 800 :=
by 
  sorry

end rate_per_square_meter_l775_775951


namespace h_eq_h_2_l775_775378

open Set

-- Define the conditions in Lean
variable {k : ℕ} (hk : 2 ≤ k)
def P (k : ℕ) := {x | x ≤ 2*k-1 ∧ 
                  (∀ x ∈ {1, 2, 3, ..., 2*k-1}, (x ∈ P → 2*k - x ∈ P))}

-- Define the function h(k)
def h (k : ℕ) : ℕ := 2^k - 1

-- State the theorem we want to prove
theorem h_eq (k : ℕ) (hk : 2 ≤ k) : h k = 2^k - 1 := sorry

-- Specific case for k = 2
theorem h_2 : h 2 = 3 := by
  rw [h]
  norm_num

end h_eq_h_2_l775_775378


namespace part_I_part_II_l775_775031

noncomputable def a (n : ℕ) : ℕ := sorry

def b (n : ℕ) : ℕ := a n + 3^n

def Sn (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), b i

theorem part_I (h1 : a 5 = 11) (h2 : a 2 + a 6 = 18) : 
  ∀ (n : ℕ), a n = 2 * n + 1 := sorry

theorem part_II (h1 : a 5 = 11) (h2 : a 2 + a 6 = 18) : 
  ∀ (n : ℕ), Sn n = n^2 + 2 * n + 3^(n + 1) / 2 - 3 / 2 := sorry

end part_I_part_II_l775_775031


namespace parabola_and_chord_l775_775048

-- Define the conditions of the given problem
def vertex_at_origin (C : ℝ × ℝ → Prop) : Prop :=
  ∀ x y, C (0, 0)

def directrix (x : ℝ) := x = 2

noncomputable def parabola_eq (p : ℝ) : ℝ × ℝ → Prop :=
  λ ⟨x, y⟩, y^2 = -4 * p * x

def line_eq (x : ℝ) (y : ℝ) := y = x + 2

theorem parabola_and_chord
  (C : ℝ × ℝ → Prop) 
  (p : ℝ) 
  (hvertex : vertex_at_origin C)
  (hdirectrix : directrix p)
  (hparabola : ∀ x y, parabola_eq 4 (x, y))
  (hline : ∀ x y, line_eq x y) :
  (∀ x y, C (x, y) → parabola_eq 4 (x, y)) ∧ 
  ∃ A B : ℝ × ℝ, C A ∧ line_eq A.1 A.2 ∧ C B ∧ line_eq B.1 B.2 ∧ 
  (let D := (A.1 - B.1)^2 + (A.2 - B.2)^2 in
   sqrt D = 4 * sqrt 6) :=
  sorry

end parabola_and_chord_l775_775048


namespace simplify_expression_l775_775935

theorem simplify_expression (x : ℝ) :
  (3 * x)^3 - (4 * x^2) * (2 * x^3) = 27 * x^3 - 8 * x^5 :=
by
  sorry

end simplify_expression_l775_775935


namespace intersection_of_M_and_N_l775_775060

def M : Set ℝ := {x | x^2 - 3*x - 28 ≤ 0}
def N : Set ℝ := {x | x^2 - x - 6 > 0}
def intersection := {x : ℝ | -4 ≤ x ∧ x ≤ -2 ∨ 3 < x ∧ x ≤ 7}

theorem intersection_of_M_and_N : M ∩ N = intersection := by
  sorry

end intersection_of_M_and_N_l775_775060


namespace smallest_prime_divisor_and_cube_root_l775_775931

theorem smallest_prime_divisor_and_cube_root (N : ℕ) (p : ℕ) (q : ℕ)
  (hN_composite : N > 1 ∧ ¬ (∃ p : ℕ, p > 1 ∧ p < N ∧ N = p))
  (h_divisor : N = p * q)
  (h_p_prime : Nat.Prime p)
  (h_min_prime : ∀ (d : ℕ), Nat.Prime d → d ∣ N → p ≤ d)
  (h_cube_root : p > Nat.sqrt (Nat.sqrt N)) :
  Nat.Prime q := 
sorry

end smallest_prime_divisor_and_cube_root_l775_775931


namespace parallel_vectors_k_l775_775019

theorem parallel_vectors_k (k : ℝ) (a b : ℝ × ℝ) (h₁ : a = (2 - k, 3)) (h₂ : b = (2, -6)) (h₃ : a.1 * b.2 = a.2 * b.1) : k = 3 :=
sorry

end parallel_vectors_k_l775_775019


namespace race_time_differences_l775_775493

theorem race_time_differences :
  ∀ (distance : ℕ)
    (speed_malcolm speed_joshua speed_emily : ℕ),
    distance = 15 →
    speed_malcolm = 5 →
    speed_joshua = 7 →
    speed_emily = 6 →
    (speed_joshua * distance - speed_malcolm * distance = 30) ∧ (speed_emily * distance - speed_malcolm * distance = 15) :=
by
  intros distance speed_malcolm speed_joshua speed_emily h1 h2 h3 h4
  split
  · calc speed_joshua * distance - speed_malcolm * distance = 7 * 15 - 5 * 15 : by rw [h1, h2, h3]
    ... = 105 - 75 : by rw [mul_comm]
    ... = 30 : by norm_num
  · calc speed_emily * distance - speed_malcolm * distance = 6 * 15 - 5 * 15 : by rw [h1, h2, h4]
    ... = 90 - 75 : by rw [mul_comm]
    ... = 15 : by norm_num

end race_time_differences_l775_775493


namespace area_of_triangle_l775_775430

theorem area_of_triangle (AB BC AC : ℝ) (h₁ : AB = 30) (h₂ : BC = 29) (h₃ : AC = 42) : 
  let s := (AB + BC + AC) / 2 
  in let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC)) 
     in 21225 / 145 <= area ∧ area <= 21225 / 145 + 1 := 
by 
  -- This is a simplification to show the approximate area corresponding to 146
  sorry

end area_of_triangle_l775_775430


namespace circle_area_l775_775508

theorem circle_area
  (A B : ℝ × ℝ)
  (x : ℝ)
  (A_on_circle : A = (8, 17))
  (B_on_circle : B = (16, 15))
  (tangent_intersection_x : x = 7)
  (intersection_property : true): -- Placeholder, because Lean requires an additional hypothesis or premise
  (π * (4930 / 281)) = 177; -- This assertion says the area formula holds (multiplying by π is implied in the context, but not actually executable directly as such in Lean).
sorry

end circle_area_l775_775508


namespace find_a_value_l775_775255

noncomputable def problem (a : ℝ) : Prop :=
  let A := (1, 1)
  let B := (1 / 2, 0)
  let C := (3 / 2, 0)
  let AB_line (x : ℝ) := 2 * x - 1
  let AC_line (x : ℝ) := -2 * x + 3
  let G (x y : ℝ) : Prop := 
    y < AB_line x ∧ y < AC_line x ∧ y < a ∧ 0 < x ∧ x < 2 ∧ 0 < y ∧ y < 1
  let rectangular_region_area := 2 * 1
  let G_area := (1 - a) * a / 2
  G_area / rectangular_region_area = 1 / 16

theorem find_a_value : ∃ a : ℝ, 0 < a ∧ a < 1 ∧ problem a := 
by { use 1 / 2, sorry }

end find_a_value_l775_775255


namespace find_alpha_l775_775385

noncomputable def alpha_satisfying (alpha : ℝ) : Prop :=
∀ (x : ℝ), f (x) = x ^ alpha → f (1/2) = sqrt 2 / 2

theorem find_alpha :
  alpha_satisfying (1/2) :=
by
  intros x f
  sorry

end find_alpha_l775_775385


namespace AC_div_AB_eq_13_l775_775467

-- Define a triangle $ABC$ with the given conditions
variable (A B C : Point) (ABC_triangle : Triangle A B C) 

-- Given conditions
def angle_BAC_eq_90 := angle A B C = 90
def length_AB_lt_AC := length A B < length A C

-- Define the sets S1, S2, S3, S4, S5, S6
def S1 (P : Point) := ABC_triangle.contains P ∧ distance P A < distance P B ∧ distance P B < distance P C
def S2 (P : Point) := ABC_triangle.contains P ∧ distance P A < distance P C ∧ distance P C < distance P B
def S3 (P : Point) := ABC_triangle.contains P ∧ distance P B < distance P A ∧ distance P A < distance P C
def S4 (P : Point) := ABC_triangle.contains P ∧ distance P B < distance P C ∧ distance P C < distance P A
def S5 (P : Point) := ABC_triangle.contains P ∧ distance P C < distance P A ∧ distance P A < distance P B
def S6 (P : Point) := ABC_triangle.contains P ∧ distance P C < distance P B ∧ distance P B < distance P A

-- Given the ratio of the largest region to the smallest non-empty region is 49:1
def ratio_largest_to_smallest_non_empty_area : ℝ := 49

-- The required theorem
theorem AC_div_AB_eq_13 
    (h1 : angle_BAC_eq_90) 
    (h2 : length_AB_lt_AC) 
    (h3 : ∃ (largest smallest : Set Point), ratio (area largest) (area smallest) = ratio_largest_to_smallest_non_empty_area)
    : ratio (length A C) (length A B) = 13 := 
sorry

end AC_div_AB_eq_13_l775_775467


namespace total_available_space_l775_775652

-- Define the problem conditions
variable (S : ℕ) -- The total floor space of the second floor in square feet
variable (H_first_floor_size : 2 * S = S1) 
variable (H_boxes_use : 5000 = (1/4) * S)

-- Define the problem statement
theorem total_available_space (S1 S2 : ℕ) (h1: 2 * S = S1) (h2: 5000 = 1/4 * S):
  S1 + (3 / 4 * S) = 55000 :=
begin
  sorry
end

end total_available_space_l775_775652


namespace allocation_schemes_beijing_olympics_l775_775705

theorem allocation_schemes_beijing_olympics : 
  ∃ schemes : ℕ, schemes = 240 ∧ 
  ( ∀ (volunteers : Fin 5 → Fin 4), 
    ∃ assignments : volunteers → Fin 4, 
    (∀ v, ∃ p, assignments v = p) ∧ 
    (∀ p, ∃ v, assignments v = p) ) :=
sorry

end allocation_schemes_beijing_olympics_l775_775705


namespace mode_of_dataset_l775_775955

def dataset : List ℕ := [0, 1, 2, 2, 3, 1, 3, 3]

def frequency (n : ℕ) (l : List ℕ) : ℕ :=
  l.count n

theorem mode_of_dataset :
  (∀ n ≠ 3, frequency n dataset ≤ 3) ∧ frequency 3 dataset = 3 :=
by
  sorry

end mode_of_dataset_l775_775955


namespace compare_sums_l775_775738

theorem compare_sums (a b c : ℝ) (h : a > b ∧ b > c) : a^2 * b + b^2 * c + c^2 * a > a * b^2 + b * c^2 + c * a^2 := by
  sorry

end compare_sums_l775_775738


namespace conjugate_of_z_l775_775023

open Complex

theorem conjugate_of_z (z : ℂ) (h : (1 + I) * z = abs (1 - I) * I) : 
  conj z = (√2 / 2) + (√2 / 2) * I :=
begin
  -- Proof goes here
  sorry
end

end conjugate_of_z_l775_775023


namespace time_to_pass_tree_l775_775612

-- Define the conditions
def length_of_train : ℝ := 500  -- the length of the train in meters
def initial_velocity : ℝ := 0    -- the train starts from rest, so initial velocity is 0 m/s
def acceleration : ℝ := 0.5      -- uniform acceleration of the train in m/s²

-- Define the proof goal
theorem time_to_pass_tree :
  ∃ t : ℝ, length_of_train = (initial_velocity * t) + (0.5 * acceleration * t^2) ∧ t ≈ 44.72 :=
sorry

end time_to_pass_tree_l775_775612


namespace max_type_A_toys_l775_775259

-- Definitions for unit prices and quantities
def price_A (price_B : ℝ) := 1.6 * price_B
def quantity_A (total_amount : ℝ) (price_A : ℝ) := total_amount / price_A

-- Conditions
def condition1 (price_B : ℝ) (quantity_A : ℝ) (quantity_B : ℝ) :=
  quantity_A - quantity_B = 30

def condition2 (price_A price_B : ℝ) :=
  price_A = 1.6 * price_B

def max_quantity_A (price_A price_B total_cost total_quantity : ℝ) : ℝ :=
  (total_cost - price_B * total_quantity) / (price_A - price_B)

-- Lean theorem
theorem max_type_A_toys (price_A price_B : ℝ) (total_cost total_quantity : ℝ) :
  price_A = 8 ∧ price_B = 5 ∧ total_cost = 1350 ∧ total_quantity = 200 →
  max_quantity_A price_A price_B total_cost total_quantity = 116 :=
sorry

end max_type_A_toys_l775_775259


namespace PM_parallel_GH_of_triangle_properties_l775_775433

open EuclideanGeometry

/-- Given triangle ABC with orthocenter H. 
    BH intersects AC at E, and CH intersects AB at F. 
    The tangent line to the circumcircle of triangle ABC passing through A intersects BC at P.
    M is the midpoint of AH.
    EF intersects BC at G.
    Prove that PM is parallel to GH.
-/
theorem PM_parallel_GH_of_triangle_properties (A B C H E F P M G : Point)
  (h_orthocenter : orthocenter A B C H)
  (h_BE_AC : (line_through B H) ∩ (line_through A C) = {E})
  (h_CF_AB : (line_through C H) ∩ (line_through A B) = {F})
  (h_tangent_at_A : is_tangent (circumcircle A B C) (line_through A P))
  (h_tangent_intersects_BC : (line_through A P) ∩ (line_through B C) = {P})
  (h_M_midpoint_AH : midpoint M A H)
  (h_EF_BC_G : (line_through E F) ∩ (line_through B C) = {G}) :
  parallel (line_through P M) (line_through G H) :=
sorry

end PM_parallel_GH_of_triangle_properties_l775_775433


namespace remainder_of_n_div_1000_l775_775896

noncomputable def setS : Set ℕ := {x | 1 ≤ x ∧ x ≤ 15}

def n : ℕ :=
  let T := {x | 4 ≤ x ∧ x ≤ 15}
  (3^12 - 2^12) / 2

theorem remainder_of_n_div_1000 : (n % 1000) = 672 := 
  by sorry

end remainder_of_n_div_1000_l775_775896


namespace floor_covered_by_three_layers_l775_775225

theorem floor_covered_by_three_layers 
    (total_rug_area : ℝ)
    (covered_area : ℝ)
    (area_two_layers : ℝ)
    (h1 : total_rug_area = 200)
    (h2 : covered_area = 140)
    (h3 : area_two_layers = 24) :
    let overlap_area := total_rug_area - covered_area in
    let k := (overlap_area - area_two_layers) / 2 in
    k = 18 :=
by
  simp [h1, h2, h3]
  unfold overlap_area
  unfold k
  sorry

end floor_covered_by_three_layers_l775_775225


namespace sum_two_digit_integers_ends_with_36_l775_775586

/-!
  Prove that the sum of all two-digit positive integers whose squares end with the digits 36 is 130.
-/

def is_two_digit_integer (n : ℕ) : Prop :=
  n >= 10 ∧ n < 100

def ends_with_36 (n : ℕ) : Prop :=
  (n * n) % 100 = 36

theorem sum_two_digit_integers_ends_with_36 :
  (∑ n in finset.filter (λ n, is_two_digit_integer n ∧ ends_with_36 n) (finset.range 100), n) = 130 :=
by {
  sorry
}

end sum_two_digit_integers_ends_with_36_l775_775586


namespace log_problem_l775_775184

theorem log_problem (x : ℝ) (h : log 8 x + 3 * log 2 x = 12) : x = 2 ^ 3.6 :=
by
  sorry  -- Proof goes here

end log_problem_l775_775184


namespace find_smallest_omega_l775_775053

noncomputable def smallest_omega (ω : ℝ) : Prop :=
  ω > 0 ∧ ∃ k : ℤ, (π / 3) = k * (2 * π / ω)

theorem find_smallest_omega : smallest_omega 6 :=
by
  sorry

end find_smallest_omega_l775_775053


namespace find_x_l775_775826

def inequality_holds (x : ℝ) : Prop :=
  ∀ (m : ℝ), (0 < m) → (mx - 1) * (3m^2 - (x + 1)m - 1) ≥ 0

theorem find_x (x : ℝ) (h : inequality_holds x) : x = 1 :=
sorry

end find_x_l775_775826


namespace incorrect_result_l775_775800

theorem incorrect_result (a b : ℤ) :
  (-a + b = -1) → (a + b = 5) → (4a + b = 14) → ¬ (2a + b = 7) :=
by
  intro hA hB hD
  -- We skip the proof using sorry, as it's not required in the task
  sorry

end incorrect_result_l775_775800


namespace min_sqrt_m2_n2_l775_775476

theorem min_sqrt_m2_n2 (a b m n : ℝ) (h1 : a^2 + b^2 = 3) (h2 : m * a + n * b = 3) : 
∃ (x : ℝ), x = sqrt (m^2 + n^2) ∧ x ≥ sqrt 3 := 
sorry

end min_sqrt_m2_n2_l775_775476


namespace chord_angle_measure_l775_775040

theorem chord_angle_measure (AB_ratio : ℕ) (circ : ℝ) (h : AB_ratio = 1 + 5) : 
  ∃ θ : ℝ, θ = (1 / 6) * circ ∧ θ = 60 :=
by
  sorry

end chord_angle_measure_l775_775040


namespace fraction_not_shaded_l775_775507

-- Variables and definitions
variables (s : ℝ)

def point (x y : ℝ) : Type := (x, y)
def R := point 0 (s / 3)
def S := point s (2 * s / 3)

-- The total area of the square
def area_square := s * s

-- Areas of the shapes forming the unshaded region
def area_triangle := (1 / 2) * (2 * s / 3) * (s / 3)
def area_quadrilateral := (2 * s / 3) * (s / 3)

-- The total area of the unshaded region
def white_area := 2 * area_triangle + area_quadrilateral

-- Proving the fraction of the square that is not shaded
theorem fraction_not_shaded : 1 - (white_area / area_square) = 5 / 9 :=
by 
  unfold white_area area_triangle area_square area_quadrilateral
  sorry  -- Proof placeholder

end fraction_not_shaded_l775_775507


namespace problem_1_problem_2_l775_775768

open Real

theorem problem_1 (α : ℝ) (hα1 : sin (α + π / 2) = -sqrt 5 / 5) (hα2 : 0 < α) (hα3 : α < π) :
  (cos^2 (π / 4 + α / 2) - cos^2 (π / 4 - α / 2)) / (sin (π - α) + cos (3 * π + α)) = -2 / 3 :=
sorry

theorem problem_2 (α : ℝ) (hα1 : sin (α + π / 2) = -sqrt 5 / 5) (hα2 : 0 < α) (hα3 : α < π) :
  cos (2 * α - 3 * π / 4) = -sqrt 2 / 10 :=
sorry

end problem_1_problem_2_l775_775768


namespace gcd_18_n_eq_6_in_range_l775_775724

theorem gcd_18_n_eq_6_in_range :
  {n : ℕ | 1 ≤ n ∧ n ≤ 200 ∧ Nat.gcd 18 n = 6}.card = 22 :=
by
  -- To skip the proof
  sorry

end gcd_18_n_eq_6_in_range_l775_775724


namespace sin_2x_eq_cos_x_solution_set_l775_775700

theorem sin_2x_eq_cos_x_solution_set :
  {x : ℝ | x ∈ set.Icc 0 (2 * Real.pi) ∧ Real.sin (2 * x) = Real.cos x} = 
  {Real.pi / 2, 3 * Real.pi / 2, Real.pi / 6, 5 * Real.pi / 6} :=
by
  sorry

end sin_2x_eq_cos_x_solution_set_l775_775700


namespace gcd_of_72_90_120_l775_775582

theorem gcd_of_72_90_120 : Nat.gcd (Nat.gcd 72 90) 120 = 6 := 
by 
  have h1 : 72 = 2^3 * 3^2 := by norm_num
  have h2 : 90 = 2 * 3^2 * 5 := by norm_num
  have h3 : 120 = 2^3 * 3 * 5 := by norm_num
  sorry

end gcd_of_72_90_120_l775_775582


namespace combined_flock_size_after_5_years_l775_775324

noncomputable def initial_flock_size : ℕ := 100
noncomputable def ducks_killed_per_year : ℕ := 20
noncomputable def ducks_born_per_year : ℕ := 30
noncomputable def years_passed : ℕ := 5
noncomputable def other_flock_size : ℕ := 150

theorem combined_flock_size_after_5_years
  (init_size : ℕ := initial_flock_size)
  (killed_per_year : ℕ := ducks_killed_per_year)
  (born_per_year : ℕ := ducks_born_per_year)
  (years : ℕ := years_passed)
  (other_size : ℕ := other_flock_size) :
  init_size + (years * (born_per_year - killed_per_year)) + other_size = 300 := by
  -- The formal proof would go here.
  sorry

end combined_flock_size_after_5_years_l775_775324


namespace minimum_distance_curve_line_l775_775178

noncomputable def general_equation_curve_C : String := "general equation of curve C"
noncomputable def cartesian_equation_line_l : String := "Cartesian coordinate equation of line l"

theorem minimum_distance_curve_line 
  (α : Real)
  (P : Real × Real)
  (hp : P = (sqrt 3 * Real.cos α, Real.sin α))
  (dist_to_line : Real)
  (hline : dist_to_line = (abs (2 * Real.sin (α + Real.pi / 3) - 4) / sqrt 2)) :
  dist_to_line = (abs (2 * Real.sin (α + Real.pi / 3) - 4) / sqrt 2) := 
sorry

end minimum_distance_curve_line_l775_775178


namespace proof_problem_l775_775737

-- Define the conditions
variable (θ : Real) 
variable (tan_θ : Real)
variable (sin_θ : Real)
variable (cos_θ : Real)

noncomputable def given_conditions (θ : Real) :=
  tan_θ = 3 ∧ sin^2 θ + cos^2 θ = 1

-- Define the expression to be proved
noncomputable def target_expression (θ : Real) : Real :=
  ((1 - cos_θ ^ 2) / sin_θ) - (sin_θ / (1 + cos_θ))

-- Prove that the above expression equals to the evaluated value given the conditions
theorem proof_problem (θ : Real) (tan_θ : Real) (sin_θ : Real) (cos_θ : Real) :
  (given_conditions θ) → 
  target_expression θ = (3 / Real.sqrt 10) - (3 / (Real.sqrt 10 + 1)) :=
by
  sorry

end proof_problem_l775_775737


namespace smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l775_775585

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, 1 < m ∧ m < n → ¬(m ∣ n)
def is_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n
def has_prime_factor_less_than (n k : ℕ) : Prop := ∃ p : ℕ, p < k ∧ is_prime p ∧ p ∣ n

theorem smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60 :
  ∃ m : ℕ, 
    m = 4091 ∧ 
    ¬is_prime m ∧ 
    ¬is_square m ∧ 
    ¬has_prime_factor_less_than m 60 ∧ 
    (∀ n : ℕ, ¬is_prime n ∧ ¬is_square n ∧ ¬has_prime_factor_less_than n 60 → 4091 ≤ n) :=
by
  sorry

end smallest_integer_neither_prime_nor_square_no_prime_factor_less_than_60_l775_775585


namespace joan_exam_time_difference_l775_775127

theorem joan_exam_time_difference :
  (let english_questions := 30
       math_questions := 15
       english_time_hours := 1
       math_time_hours := 1.5
       english_time_minutes := english_time_hours * 60
       math_time_minutes := math_time_hours * 60
       time_per_english_question := english_time_minutes / english_questions
       time_per_math_question := math_time_minutes / math_questions
    in time_per_math_question - time_per_english_question = 4) :=
by
  sorry

end joan_exam_time_difference_l775_775127


namespace sequence_bounds_l775_775265

theorem sequence_bounds (a : ℕ → ℕ) (T : ℕ → ℝ) 
  (h1 : ∀ n, a n = 2 * n)
  (h2 : ∀ n, T n = (∑ i in range n, (2:ℝ) / (a i * ((a (i + 1)) + 2)))) :
  ∀ n, (1/6 : ℝ) ≤ T n ∧ T n < (3/8 : ℝ) := sorry

end sequence_bounds_l775_775265


namespace line_intersect_ellipse_l775_775051

-- Definition of the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 5 + y^2 / 4 = 1

-- Definition of the line
def line (m x y : ℝ) : Prop :=
  m * x + y + m - 1 = 0

-- Proving the intersection
theorem line_intersect_ellipse (m : ℝ) : 
  ∃ x y : ℝ, ellipse x y ∧ line m x y :=
begin
  use [-1, 1],
  split,
  { dsimp [ellipse],
    have h₁ : (-1 : ℝ)^2 / 5 = 1 / 5 := by norm_num,
    have h₂ : (1 : ℝ)^2 / 4 = 1 / 4 := by norm_num,
    rw [h₁, h₂],
    norm_num },
  { dsimp [line],
    norm_num }
end

end line_intersect_ellipse_l775_775051


namespace sheep_converge_to_one_peasant_l775_775180

-- Define the initial conditions
def peasants : Type := ℕ
def sheep_count (p : peasants) : ℕ := sorry
def total_sheep := 128
def peasant_has_majority_sheep (p : peasants) : Prop := sheep_count p ≥ total_sheep / 2

-- Assuming there are several peasants and defining total sheep rule
axiom sheep_initial (p : peasants): sheep_count p < total_sheep
axiom total_sheep_count : ∀ (p : peasants), sheep_count p + (total_sheep - sheep_count p) = total_sheep

-- Defining the seizure process
constant seizure : list peasants → list peasants

-- Defining the process after 7 seizures
def after_seizures (n : ℕ) : list peasants → Prop := sorry

-- Stating the main theorem
theorem sheep_converge_to_one_peasant :
  ∀ (ps : list peasants), (after_seizures 7 ps) → ∃ p : peasants, sheep_count p = total_sheep :=
sorry

end sheep_converge_to_one_peasant_l775_775180


namespace positive_solution_system_l775_775694

theorem positive_solution_system (x1 x2 x3 x4 x5 : ℝ) (h1 : (x3 + x4 + x5)^5 = 3 * x1)
  (h2 : (x4 + x5 + x1)^5 = 3 * x2) (h3 : (x5 + x1 + x2)^5 = 3 * x3)
  (h4 : (x1 + x2 + x3)^5 = 3 * x4) (h5 : (x2 + x3 + x4)^5 = 3 * x5) :
  x1 > 0 → x2 > 0 → x3 > 0 → x4 > 0 → x5 > 0 →
  x1 = x2 ∧ x2 = x3 ∧ x3 = x4 ∧ x4 = x5 ∧ (x1 = 1/3) :=
by 
  intros hpos1 hpos2 hpos3 hpos4 hpos5
  sorry

end positive_solution_system_l775_775694


namespace find_a_l775_775407

-- Define M and N based on the given conditions
def M := {x : ℝ | x^2 = 2}
def N (a : ℝ) := {x : ℝ | a * x = 1}

-- Statement of the problem
theorem find_a (a : ℝ) : N a ⊆ M → a = 0 ∨ a = 1/√2 ∨ a = -1/√2 := sorry

end find_a_l775_775407


namespace roots_arithmetic_prog_l775_775686

theorem roots_arithmetic_prog (a : ℝ) :
  (∃ r d : ℂ, r - d ≠ r + d ∧
    (root1 root2 root3 : ℂ)
    (root1 * root2 * root3 = a) 
    (root1, root2, root3 : ℂ), 
      root1 + root2 + root3 = 9 ∧
      root1 * root2 + root2 * root3 + root3 * root1 = 42 ∧
      root1 * root2 * root3 = -a) 
    (root1, root2, root3 : ℂ) such that 
      root1 = 3 - √15 * I ∧
      root2 = 3 ∧
      root3 = 3 + √15 * I )  → a = -72 :=
by {
  sorry
}

end roots_arithmetic_prog_l775_775686


namespace find_f_l775_775824

-- Given function f(x)
def f (x : ℝ) := f'1 * x^3 - 2 * x^2 + 3

-- Statement we need to prove
theorem find_f'_1 : ∃ (f'1 : ℝ), (∂ f / ∂ x) 1 = 2 := by
  sorry

end find_f_l775_775824


namespace mark_age_in_5_years_l775_775154

-- Definitions based on the conditions
def Amy_age := 15
def age_difference := 7

-- Statement specifying the age Mark will be in 5 years
theorem mark_age_in_5_years : (Amy_age + age_difference + 5) = 27 := 
by
  sorry

end mark_age_in_5_years_l775_775154


namespace probability_of_forming_triangle_l775_775566

noncomputable def number_of_combinations (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_of_forming_triangle :
  let lengths := [1, 3, 5, 7, 9]
  ∃ (probability : ℚ),
    probability = 3 / 10 ∧
    probability =
      (∑
       (x y z : ℕ) in
       lengths.to_finset.powerset.filter (λ s, s.card = 3) | 
       x + y > z ∧ y + z > x ∧ z + x > y, 1 : ℕ) /
      number_of_combinations 5 3 :=
by
  sorry

end probability_of_forming_triangle_l775_775566


namespace extreme_value_tangent_line_tangent_line_equation_l775_775055
noncomputable def f(x : ℝ) : ℝ := (1 / 2) * x^2 - x - 2 * real.log x + (1 / 2)

theorem extreme_value :
  f 2 = -2 * real.log 2 + (1 / 2) :=
sorry

theorem tangent_line :
  ∃ x0 y0 k : ℝ, y0 = f x0 ∧ k = x0 - 1 - (2 / x0) ∧
  (y0 - 2) / x0 = k ∧ k = -2 :=
sorry

theorem tangent_line_equation :
  ∀ y x : ℝ, y = -2 * (x - 1) ↔ (2 * x + y - 2 = 0) :=
sorry

end extreme_value_tangent_line_tangent_line_equation_l775_775055


namespace joan_exam_time_difference_l775_775121

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l775_775121


namespace height_of_bobby_flag_l775_775648

def area_of_fabric(f1_w : Nat, f1_h : Nat, f2_w : Nat, f2_h : Nat, f3_w : Nat, f3_h : Nat) : Nat :=
  (f1_w * f1_h) + (f2_w * f2_h) + (f3_w * f3_h)

def height_of_flag(area : Nat, length : Nat) : Nat :=
  area / length

theorem height_of_bobby_flag : 
  height_of_flag (area_of_fabric 8 5 10 7 5 5) 15 = 9 := 
by
  sorry

end height_of_bobby_flag_l775_775648


namespace arithmetic_sequence_a8_l775_775761

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_a8 (h : a 7 + a 9 = 8) : a 8 = 4 := 
by 
  -- proof steps would go here
  sorry

end arithmetic_sequence_a8_l775_775761


namespace geometric_sequence_value_l775_775837

-- Define given conditions
variables {a_3 a_7 a_5 : ℝ}

-- The main theorem to prove
theorem geometric_sequence_value : 
  (a_3 * a_7 = 2) ∧ (a_3 + a_7 = -4) ∧ (a_3 > 0) ∧ (a_7 > 0) → a_5 = sqrt 2 :=
by
  -- Use provided proof structure and hypotheses
  sorry

end geometric_sequence_value_l775_775837


namespace operation_equivalence_l775_775243

theorem operation_equivalence :
  (∀ (x : ℝ), (x * (4 / 5) / (2 / 7)) = x * (7 / 5)) :=
by
  sorry

end operation_equivalence_l775_775243


namespace Joana_seq_maxima_l775_775461

theorem Joana_seq_maxima (max_count : ℕ) (most_freq_digits : list ℕ) :
  (∀ (d : ℕ), d ∈ most_freq_digits → d ∈ [4, 5])
  ∧ max_count = 30
  ∧ most_freq_digits = [4, 5]
:=
by
  -- The setup of the problem conditions
  let seq_count (d n : ℕ) :=
    if n < d + 1 then 0 else d + 1
  -- Calculate how many times each digit appears
  let digit_counts : ℕ → ℕ :=
    λ d, (list.range 10).sum (λ n, seq_count d n)
  -- Evaluate the counts for each digit
  let count_list := list.map digit_counts (list.range 10)
  have h_max_count : max_count = list.maximum count_list,
  { sorry },
  have h_digit_4_5 : digit_counts 4 = max_count ∧ digit_counts 5 = max_count,
  { sorry },
  exact ⟨by simp [most_freq_digits, [4, 5], h_digit_4_5], h_max_count, sorry⟩

end Joana_seq_maxima_l775_775461


namespace smallest_positive_period_of_h_l775_775910

-- Definitions of f and g with period 1
axiom f : ℝ → ℝ
axiom g : ℝ → ℝ
axiom T1 : ℝ
axiom T2 : ℝ

-- Given conditions
@[simp] axiom f_periodic : ∀ x, f (x + T1) = f x
@[simp] axiom g_periodic : ∀ x, g (x + T2) = g x
@[simp] axiom T1_eq_one : T1 = 1
@[simp] axiom T2_eq_one : T2 = 1

-- Statement to prove the smallest positive period of h(x) = f(x) + g(x) is 1/k
theorem smallest_positive_period_of_h (k : ℕ) (h : ℝ → ℝ) (hk: k > 0) :
  (∀ x, h (x + 1) = h x) →
  (∀ T > 0, (∀ x, h (x + T) = h x) → (∃ k : ℕ, T = 1 / k)) :=
by sorry

end smallest_positive_period_of_h_l775_775910


namespace value_of_100d_l775_775480

noncomputable def sequence_b : ℕ → ℝ
| 0       := 7 / 10
| (n + 1) := 3 * (sequence_b n)^2 - 2

def condition (d : ℝ) : Prop :=
∀ (n : ℕ), |list.prod (list.map sequence_b (list.range n))| ≤ d / 3^n

theorem value_of_100d : ∃ (d : ℝ), condition d ∧ 100 * d = 29 :=
sorry

end value_of_100d_l775_775480


namespace find_c10_l775_775039

-- Arithmetic sequence properties
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ (d : ℕ), ∀ n : ℕ, a (n + 1) = a n + d

-- Given conditions
def a : ℕ → ℕ := λ n, match n with
  | 1     => 3
  | 2     => 4
  | _     => 1 * (n - 1) + 3 -- General formula a_n = a_1 + (n-1)*d_aa

def b : ℕ → ℕ := λ n, match n with
  | 1     => 6
  | 2     => 9
  | _     => 3 * (n - 1) + 6 -- General formula b_n = b_1 + (n-1)*d_bb

noncomputable def c : ℕ → ℕ :=
  λ n, 3 * n + 6  -- General term of the common sequence c_n

-- Proof Problem
theorem find_c10 : c 10 = 36 :=
by {
    sorry
}

end find_c10_l775_775039


namespace correct_statement_l775_775394

noncomputable def f (a x : ℝ) : ℝ := logBase a x

theorem correct_statement 
  (a x₁ x₂ : ℝ)
  (h₀ : 0 < a) 
  (h₁ : a < 1) 
  (h₂ : f a x₁ > f a x₂) : 
  x₁ < x₂ := 
begin
  sorry
end

end correct_statement_l775_775394


namespace distance_between_cars_after_1_hour_l775_775546

def initial_distance : ℝ := 200
def speed_car1 : ℝ := 60
def speed_car2 : ℝ := 80
def one_hour : ℝ := 1

theorem distance_between_cars_after_1_hour : 
  ∃ (d : ℝ), d ∈ {60, 180, 220, 340} :=
by
  -- We declare the distance in each case for clarity, skip the proof. 
  sorry

end distance_between_cars_after_1_hour_l775_775546


namespace variance_of_X_l775_775091

-- Definitions based on the conditions
def prob_X_0 : ℝ := 0.4
def prob_X_1 : ℝ := 1 - prob_X_0

-- Statement with the given conditions and the question translated into Lean
theorem variance_of_X : 
  let p0 : ℝ := prob_X_0,
      p1 : ℝ := prob_X_1,
      x0 : ℝ := 0,
      x1 : ℝ := 1,
      E_X : ℝ := x0 * p0 + x1 * p1,
      variance_X : ℝ := (x0 - E_X) ^ 2 * p0 + (x1 - E_X) ^ 2 * p1 in
  variance_X = 0.24 :=
by
  sorry

end variance_of_X_l775_775091


namespace minimum_value_of_g_l775_775043

noncomputable def f (x : ℝ) := x^2
noncomputable def f' (x : ℝ) := 2 * x

def g (x m : ℝ) := f x + abs (f' x - m)

theorem minimum_value_of_g (m : ℝ) :
  (∀ x, f x = x^2) ∧
  (∀ x, f' x = 2 * x) ∧
  (f' 1 = 2) →
  ∀ x, 
  g (x : ℝ) (m : ℝ) = 
    if m < -2 then -m - 1
    else if -2 ≤ m ∧ m ≤ 2 then (m^2) / 4
    else m - 1 :=
begin
  intros h x,
  obtain ⟨hf, hf', hf'_1⟩ := h,
  cases le_total m (-2) with h_1 h_1,
  { have h_1_lt : m < -2 := h_1.lt_of_ne (by linarith only [h_1]),
    rw if_pos h_1_lt,
    sorry,
  },
  { cases le_total 2 m with h_2 h_2,
    { have h_3.lt_of_ne : m > 2 := h_2.lt_of_ne (by linarith only [h_2]),
      rw if_neg (not_lt_of_le h_2) (if_pos h_3.lt_of_ne),
      sorry,
    },
    { have h_1_le : -2 ≤ m ∧ m ≤ 2 := ⟨le_of_not_gt h_1, h_2⟩,
      rw if_neg (not_lt_of_le h_2) (if_neg (not_or_of_not_lt h_1_lt) h_1_le),
      sorry,
    }
  },
end

end minimum_value_of_g_l775_775043


namespace average_speed_inequality_l775_775594

theorem average_speed_inequality (a b : ℝ) (h : a < b) (ha : 0 < a) (hb : 0 < b) :
  let v := (2 * a * b) / (a + b) in a < v ∧ v < sqrt (a * b) :=
by
  sorry

end average_speed_inequality_l775_775594


namespace base_4_representation_156_l775_775578

theorem base_4_representation_156 :
  ∃ b3 b2 b1 b0 : ℕ,
    156 = b3 * 4^3 + b2 * 4^2 + b1 * 4^1 + b0 * 4^0 ∧
    b3 = 2 ∧ b2 = 1 ∧ b1 = 3 ∧ b0 = 0 :=
by
  have h1 : 156 = 2 * 4^3 + 28 := by norm_num
  have h2 : 28 = 1 * 4^2 + 12 := by norm_num
  have h3 : 12 = 3 * 4^1 + 0 := by norm_num
  refine ⟨2, 1, 3, 0, _, rfl, rfl, rfl, rfl⟩
  rw [h1, h2, h3]
  norm_num

end base_4_representation_156_l775_775578


namespace find_f5_l775_775201

noncomputable def f : ℝ → ℝ := sorry

axiom additivity : ∀ x y : ℝ, f (x + y) = f x + f y
axiom f4_value : f 4 = 5

theorem find_f5 : f 5 = 25 / 4 :=
by
  -- Proof goes here
  sorry

end find_f5_l775_775201


namespace _l775_775571

noncomputable def point : Type := ℝ × ℝ -- Representing a point as an ordered pair in ℝ²

variables (A B E F G : point) -- The points in the problem
variables (AB : ℝ) (BE AF EA FB AG GB : ℝ) -- The lengths given in the problem
variables (congruent_triangles : ∀ (P Q : point), triangles_congruent P Q A B E F) -- Congruence condition of triangles

-- Conditions
_axiom AB_len : AB = 12
_axiom BE_len : BE = 13
_axiom AF_len : AF = 13
_axiom EA_len : EA = 20
_axiom FB_len : FB = 20
_axiom AG_len : AG = 5
_axiom GB_len : GB = 7
_axiom E_exists : E ≠ F ∧ point_on_opposite_sides E F A B -- E and F lie on opposite sides of line AB
_axiom congruent : triangles_congruent A E B A F B congruent_triangles -- Triangles ABE and ABF are congruent

-- Question: Prove the area of intersection is correct
_theorem area_intersection : triangle_area A B E = 71.28 :=
sorry

end _l775_775571


namespace top_grades_proof_l775_775501

-- Define the basic variables
variables (P1 V1 P2 V2 Pt Vt : ℕ)

-- Define the conditions from part a)
def condition1 : Prop := P1 + V1 = 10
def condition2 : Prop := P1 > V1
def condition3 : Prop := V2 = 3
def condition4 : Prop := P2 = 0
def condition5 : Prop := Vt > Pt
def condition6 : Prop := Pt = P1 + P2
def condition7 : Prop := Vt = V1 + V2

-- Define the conclusion we want to prove
def conclusion : Prop := Pt = 6 ∧ Vt = 7

theorem top_grades_proof 
  (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) 
  (h5 : condition5) (h6 : condition6) (h7 : condition7) : conclusion := by
  sorry

end top_grades_proof_l775_775501


namespace translated_parabola_eq_l775_775226

-- Define the initial parabola
def initial_parabola (x : ℝ) : ℝ := 2 * x^2

-- Define the transformation: translate up by 3 units
def translate_up_3 (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 3

-- Define the transformation: translate right by 2 units
def translate_right_2 (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x - 2)

-- The resulting function after both translations
def resulting_parabola := translate_right_2 (translate_up_3 initial_parabola)

-- The theorem stating that the resulting parabola is 
-- given by the equation y = 2(x-2)^2 + 3.
theorem translated_parabola_eq : resulting_parabola = (λ x, 2 * (x - 2)^2 + 3) :=
sorry

end translated_parabola_eq_l775_775226


namespace find_integer_k_l775_775337

theorem find_integer_k (k : ℤ) : (∃ k : ℤ, (k = 6) ∨ (k = 2) ∨ (k = 0) ∨ (k = -4)) ↔ (∃ k : ℤ, (2 * k^2 + k - 8) % (k - 1) = 0) :=
by
  sorry

end find_integer_k_l775_775337


namespace f_has_three_distinct_roots_l775_775481

theorem f_has_three_distinct_roots (c : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = x^2 - 4 * x + c) :
  (∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ f(f(a)) = 0 ∧ f(f(b)) = 0 ∧ f(f(c)) = 0) ↔ c = 8 :=
by
  sorry

end f_has_three_distinct_roots_l775_775481


namespace count_divisible_by_9_l775_775143

def b (n : ℕ) : ℕ := 
  (List.range (n + 1)).reverse.foldl (λ acc x, acc * 10 + x) 0

theorem count_divisible_by_9 (k : ℕ) (hk : 1 ≤ k ∧ k ≤ 150) : 
  (List.range' 1 150).count (λ k, b k % 9 = 0) = 32 := 
sorry

end count_divisible_by_9_l775_775143


namespace triangle_area_l775_775443

theorem triangle_area (a b : ℝ) (h1 : b = (24 / a)) (h2 : 3 * 4 + a * (12 / a) = 12) : b = 3 / 2 :=
by
  sorry

end triangle_area_l775_775443


namespace all_A_digit_numbers_are_A_minus_1_expressible_l775_775258

def m_expressible (m : ℕ) (n : ℕ) : Prop :=
  ∃ expr : String, (expr.length = m) ∧ (evaluate expr = some n)

def evaluate (expr : String) : Option ℕ := 
  sorry -- Placeholder for the evaluation function

theorem all_A_digit_numbers_are_A_minus_1_expressible :
  ∃ A : ℕ, ∀ n : ℕ, (Nat.digits 10 n).length = A → m_expressible (A - 1) n :=
by
  sorry -- Proof not required

end all_A_digit_numbers_are_A_minus_1_expressible_l775_775258


namespace planes_parallel_l775_775482

variable {m n : Line} {α β : Plane}

axiom m_parallel_alpha : m ∥ α
axiom n_parallel_alpha : n ∥ α
axiom m_parallel_beta : m ∥ β
axiom n_parallel_beta : n ∥ β
axiom m_n_skew : skew m n

theorem planes_parallel : α ∥ β := 
by
  sorry

end planes_parallel_l775_775482


namespace centers_collinear_l775_775389

theorem centers_collinear (k : ℝ) (hk : k ≠ -1) :
    ∀ p : ℝ × ℝ, p = (-k, -2*k-5) → (2*p.1 - p.2 - 5 = 0) :=
by
  sorry

end centers_collinear_l775_775389


namespace range_of_a_l775_775427

noncomputable def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y : ℝ⦄, x ∈ I → y ∈ I → x < y → f x ≤ f y

theorem range_of_a (a : ℝ) :
  is_increasing_on (λ x => x^2 + a * x + 1 / x) (Set.Ioi (1 / 2)) ↔ 3 ≤ a := 
by
  sorry

end range_of_a_l775_775427


namespace rearrangement_finish_before_830_l775_775217

theorem rearrangement_finish_before_830
  (n : ℕ)
  (h : n = 30)
  : ∀ (queue : list (bool × ℕ)), -- the queue is represented as a list of pairs (is_girl, position)
    (∀ t : ℕ, t < n - 1 → 
      (∀ i : ℕ, i < (queue.length - 1) → 
        (queue.nth_le i ‹i < queue.length›).fst = tt → (queue.nth_le (i + 1) ‹i + 1 < queue.length›).fst = ff → 
        (∀ j : ℕ, j < i → (queue.iterate j) «move boy behind girl one step ahead» ← sorry))) →
    (∀ i : ℕ, i < n → (queue.nth_le i (i < queue.length)).fst = false → (queue.nth_le (i + 1) (i + 1 < queue.length)).fst = true) →
    sorted queue :=
begin
  sorry
end

end rearrangement_finish_before_830_l775_775217


namespace joan_exam_time_difference_l775_775120

theorem joan_exam_time_difference :
  ∀ (E_time M_time E_questions M_questions : ℕ),
  E_time = 60 →
  M_time = 90 →
  E_questions = 30 →
  M_questions = 15 →
  (M_time / M_questions) - (E_time / E_questions) = 4 :=
by
  intros E_time M_time E_questions M_questions hE_time hM_time hE_questions hM_questions
  sorry

end joan_exam_time_difference_l775_775120


namespace other_number_is_29_l775_775076

theorem other_number_is_29
    (k : ℕ)
    (some_number : ℕ)
    (h1 : k = 2)
    (h2 : (5 + k) * (5 - k) = some_number - 2^3) :
    some_number = 29 :=
by
  sorry

end other_number_is_29_l775_775076


namespace units_digit_7_pow_6_l775_775254

theorem units_digit_7_pow_6 : (7 ^ 6) % 10 = 9 := by
  sorry

end units_digit_7_pow_6_l775_775254


namespace min_workers_for_profit_l775_775272

-- Conditions
def daily_maintenance_fee : ℝ := 600
def hourly_wage : ℝ := 20
def hours_per_day : ℝ := 8
def widgets_per_hour : ℝ := 3
def price_per_widget : ℝ := 2.80

-- Function to calculate cost per worker per day
def cost_per_worker_per_day := hourly_wage * hours_per_day 
-- Function to calculate total cost for n workers
def total_cost (n : ℕ) : ℝ := daily_maintenance_fee + cost_per_worker_per_day * n

-- Function to calculate total widgets produced per day by one worker
def total_widgets_per_worker_per_day := widgets_per_hour * hours_per_day
-- Function to calculate revenue per worker per day
def revenue_per_worker_per_day := total_widgets_per_worker_per_day * price_per_widget
-- Function to calculate total revenue for n workers
def total_revenue (n : ℕ) : ℝ := revenue_per_worker_per_day * n

-- Proof statement
theorem min_workers_for_profit : ∃ n : ℕ, (total_revenue n > total_cost n) ∧ n = 7 :=
by {
    -- we need to show that 7 workers are sufficient for making profit
    -- this will involve calculating the revenue and cost to ensure the conditions
    -- sorry is placed because actual proof isn't required, only statement
    sorry,
}

end min_workers_for_profit_l775_775272


namespace cubic_has_three_roots_l775_775805

noncomputable def solve_cubic (a b c α : ℚ) (x : ℚ) : Prop :=
  let poly := (Polynomial.X^3 + a * Polynomial.X^2 + b * Polynomial.X + c) in
  Polynomial.eval α poly = 0 ∧
  ∃ (q : Polynomial ℚ),
    poly = Polynomial.X - Polynomial.C α * q ∧
    q.degree = Polynomial.natDegree q - 1 ∧
    ∃ (roots : List ℚ), Polynomial.roots q = roots 

theorem cubic_has_three_roots (a b c α : ℚ) (x : ℚ) : solve_cubic a b c α x :=
by
  sorry

end cubic_has_three_roots_l775_775805


namespace general_term_a_n_sum_of_inverse_bn_l775_775775

-- Definitions and conditions
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  ∀ n, S n = (n + 1) * a n / 2

def first_term (a : ℕ → ℝ) : Prop :=
  a 0 = 1 / 2

def arithmetic_sequence (a S : ℕ → ℝ) : Prop :=
  ∀ n, 2 * a n = S n + 1 / 2

def bn_def (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∀ n, b n = (Real.log 2 (a (2 * n + 1))) * (Real.log 2 (a (2 * n + 3)))

-- Theorem about general term formula for the sequence a_n
theorem general_term_a_n (a S : ℕ → ℝ)
  (hS : sum_first_n_terms a S)
  (hFirst : first_term a)
  (hArith : arithmetic_sequence a S) :
  ∀ n, a n = 2^(n - 2) :=
sorry

-- Theorem about sum of 1 / b_n
theorem sum_of_inverse_bn (a b : ℕ → ℝ)
  (hS : sum_first_n_terms a (λ n, (n + 1) * a n / 2))
  (hFirst : first_term a)
  (hArith : arithmetic_sequence a (λ n, (n + 1) * a n / 2))
  (hbDef : bn_def a b) :
  ∀ n, (∑ i in Finset.range (n+1), 1 / (b i)) = n / (2 * n + 1) :=
sorry

end general_term_a_n_sum_of_inverse_bn_l775_775775


namespace triangle_area_l775_775342

theorem triangle_area :
  let A := (0, 7, 10 : ℝ × ℝ × ℝ)
  let B := (-1, 6, 8 : ℝ × ℝ × ℝ)
  let C := (-4, 9, 6 : ℝ × ℝ × ℝ)
  let AB := (B.1 - A.1, B.2 - A.2, B.3 - A.3)
  let AC := (C.1 - A.1, C.2 - A.2, C.3 - A.3)
  let cross_product := (AB.2 * AC.3 - AB.3 * AC.2, AB.3 * AC.1 - AB.1 * AC.3, AB.1 * AC.2 - AB.2 * AC.1)
  let magnitude := real.sqrt (cross_product.1^2 + cross_product.2^2 + cross_product.3^2)
  (1 / 2 * magnitude) = real.sqrt 41 := 
by
  sorry

end triangle_area_l775_775342


namespace find_c_l775_775006

theorem find_c (c : ℝ) : 
  (let A := 5 in
  let half_area := A / 2 in
  let line_eq := λ x,  3 * (x - c) / (3 - c) in
  let base := 3 - c in
  let height := 3 in
  let triangle_area := (base * height) / 2 in
  let shaded_area := triangle_area - 1 in
  shaded_area = half_area) → 
  c = 2 / 3 := 
begin
  intros h,
  -- Proof omitted for demonstration
  sorry
end

end find_c_l775_775006


namespace average_score_first_10_matches_l775_775192

theorem average_score_first_10_matches (A : ℕ) 
  (h1 : 0 < A) 
  (h2 : 10 * A + 15 * 70 = 25 * 66) : A = 60 :=
by
  sorry

end average_score_first_10_matches_l775_775192


namespace sum_two_digit_squares_ends_with_36_l775_775588

theorem sum_two_digit_squares_ends_with_36 : 
  (∑ n in Finset.filter (λ n : ℕ, 10 ≤ n ∧ n < 100 ∧ n^2 % 100 = 36) (Finset.range 100), n) = 194 :=
by
  sorry

end sum_two_digit_squares_ends_with_36_l775_775588


namespace gcd_18_n_eq_6_l775_775721

theorem gcd_18_n_eq_6 (num_valid_n : Nat) :
  (num_valid_n = (List.range 200).count (λ n, (1 ≤ n ∧ n ≤ 200) ∧ (6 ∣ n) ∧ ¬(9 ∣ n))) →
  num_valid_n = 22 := by
  sorry

end gcd_18_n_eq_6_l775_775721


namespace find_a2_plus_a8_l775_775101

variable {α : Type}
variables (a : ℕ → α)

-- Define the condition for an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → α) [Add α] [Mul α] [OfNat α 5] [OfNat α 450] :=
  ∀ (n m : ℕ), m > 0 → a (n + 1) - a n = a (m + 1) - a m

-- Given sum of terms in the arithmetic sequence
axiom sum_terms (h : is_arithmetic_sequence a) : 
  (a 3 + a 4 + a 5 + a 6 + a 7 = (450 : α))

-- Prove the result
theorem find_a2_plus_a8 (h : is_arithmetic_sequence a) : 
  (a 2 + a 8 = (180 : α)) :=
sorry

end find_a2_plus_a8_l775_775101


namespace min_washes_at_least_4_l775_775999

noncomputable def min_washes (x : ℕ) : Prop :=
  (1/4 : ℝ)^x ≤ 1/100

theorem min_washes_at_least_4 : ∃ x, min_washes x ∧ x = 4 :=
begin
  use 4,
  unfold min_washes,
  norm_num,
  have log_ineq : (4 : ℝ).log ≤ (100 : ℝ).log,
  { apply log_le_log,
    norm_num,
    apply pow_nonneg,
    norm_num },
  exact le_of_log_ineq log_ineq,
end

end min_washes_at_least_4_l775_775999


namespace monotonic_increasing_interval_l775_775203

-- Given definition of the function
def f (x : ℝ) : ℝ := -x^2 + 4*x

-- Statement that needs to be proven
theorem monotonic_increasing_interval : {x : ℝ | ∀ y, f y ≤ f x} = set.Iic 2 :=
sorry

end monotonic_increasing_interval_l775_775203


namespace area_increase_of_square_garden_l775_775285

theorem area_increase_of_square_garden
  (length : ℝ) (width : ℝ)
  (h_length : length = 60)
  (h_width : width = 20) :
  let perimeter := 2 * (length + width)
  let side_length := perimeter / 4
  let initial_area := length * width
  let square_area := side_length ^ 2
  square_area - initial_area = 400 :=
by
  sorry

end area_increase_of_square_garden_l775_775285


namespace angle_equality_l775_775484

variables (A B C M N P Q : Type)
variables [Point A] [Point B] [Point C] [Point M] [Point N] [Point P] [Point Q]

-- Triangle ABC and points on segments AB and AC
variables (AB : Segment A B) (AC : Segment A C)
variables (M_on_AB : M ∈ AB) (N_on_AC : N ∈ AC)

-- MN is parallel to BC
variable (MN_parallel_BC : Parallel (Segment M N) (Segment B C))

-- Intersection of BN and CM
variable (P_def : Intersection (Line B N) (Line C M) = P)

-- Q is the other intersection point of the circumcircles of BMP and CNP
variable (Q_def : Other_intersection (Circumcircle B M P) (Circumcircle C N P) Q)

-- Final goal
theorem angle_equality
  (triangle_cond : Triangle A B C)
  (M_on_AB : M ∈ (Segment A B))
  (N_on_AC : N ∈ (Segment A C))
  (MN_parallel_BC : Parallel (Segment M N) (Segment B C))
  (P_intersection : Intersection (Line B N) (Line C M) = P)
  (Q_intersection_circles : Other_intersection (Circumcircle B M P) (Circumcircle C N P) Q) :
  Angle A B Q = Angle A C P :=
by sorry

end angle_equality_l775_775484


namespace range_of_f_lt_zero_l775_775825

theorem range_of_f_lt_zero (f : ℝ → ℝ)
  (h_even : ∀ x, f(-x) = f(x))
  (h_decreasing : ∀ x y, x < 0 ∧ y < 0 ∧ x ≤ y → f(y) ≤ f(x))
  (h_f2_eq_0 : f(2) = 0) :
  {x : ℝ | f(x) < 0} = set.Ioo (-2) 2 :=
by sorry

end range_of_f_lt_zero_l775_775825


namespace eighth_number_in_series_l775_775437

def sequence (n : ℕ) : ℚ := (-1)^(n+1) * (n+1) / 2^(n+1)

theorem eighth_number_in_series : sequence 7 = 1/32 :=
by sorry

end eighth_number_in_series_l775_775437


namespace blocks_for_sculpture_l775_775638

noncomputable def volume_block := 8 * 3 * 1
noncomputable def radius_cylinder := 3
noncomputable def height_cylinder := 8
noncomputable def volume_cylinder := Real.pi * radius_cylinder^2 * height_cylinder
noncomputable def blocks_needed := Nat.ceil (volume_cylinder / volume_block)

theorem blocks_for_sculpture : blocks_needed = 10 := by
  sorry

end blocks_for_sculpture_l775_775638


namespace max_distance_origin_perpendicular_bisector_l775_775492

noncomputable def circle (O : Point) (r : ℝ) : Set Point :=
  { p | (p.x - O.x) ^ 2 + (p.y - O.y) ^ 2 = r ^ 2 }

noncomputable def ellipse (a b : ℝ) : Set Point :=
  { p | (p.x ^ 2) / (a ^ 2) + (p.y ^ 2) / (b ^ 2) = 1 }

def maximum_distance_perpendicular_bisector (l : Line) : ℝ :=
  -- this definition will be fleshed out in the proof
  sorry

theorem max_distance_origin_perpendicular_bisector :
  ∃ l : Line,
    (tangent_to_circle l (circle ⟨0, 0⟩ 1)) ∧ 
    ∃ A B : Point, 
      (A ≠ B) ∧ 
      (A ∈ ellipse 3 1) ∧ 
      (B ∈ ellipse 3 1) ∧ 
      (A ∈ l) ∧ 
      (B ∈ l) ∧ 
      (maximum_distance_perpendicular_bisector l = 4 / 3) :=
by
  sorry

end max_distance_origin_perpendicular_bisector_l775_775492


namespace trigonometric_identity_l775_775817

variable (a b c : ℝ)
variable (θ : ℝ)

theorem trigonometric_identity (h : (sin θ) ^ 6 / a + (cos θ) ^ 6 / b + (sin θ) ^ 2 * (cos θ) ^ 2 / c = 1 / (a + b + c)) :
  (sin θ) ^ 12 / a ^ 5 + (cos θ) ^ 12 / b ^ 5 + ((sin θ) ^ 2 * (cos θ) ^ 2) ^ 3 / c ^ 5 =
  (a + b + (a * b) ^ 3 / c ^ 5) / (a + b + c) ^ 6 :=
by
  sorry

end trigonometric_identity_l775_775817


namespace evaluate_expression_l775_775321

theorem evaluate_expression : (1 - 1 / (1 - 1 / (1 + 2))) = (-1 / 2) :=
by sorry

end evaluate_expression_l775_775321


namespace graph_pass_through_point_l775_775921

noncomputable def y (m x : ℝ) : ℝ := (m^2 + 2*m - 2) / x

theorem graph_pass_through_point (m : ℝ) :
  (∃ (x y : ℝ), x * y = 6 ∧ y = y m x) →
  ∃ (x y : ℝ), (x, y) = (-2, -3) :=
by
  sorry

end graph_pass_through_point_l775_775921


namespace smallest_positive_period_of_f_is_pi_minimum_value_of_f_is_neg2_f_is_monotonically_increasing_on_0_pi_l775_775699

noncomputable def f (x : ℝ) : ℝ :=
  sin x ^ 4 + 2 * sin x * cos x - cos x ^ 4

theorem smallest_positive_period_of_f_is_pi : ∀ x : ℝ, f (x + π) = f x := sorry

theorem minimum_value_of_f_is_neg2 : ∃ x : ℝ, f x = -2 := sorry

theorem f_is_monotonically_increasing_on_0_pi :
  ∀ x y : ℝ, 0 ≤ x ∧ x < y ∧ y ≤ π → f x ≤ f y := sorry

end smallest_positive_period_of_f_is_pi_minimum_value_of_f_is_neg2_f_is_monotonically_increasing_on_0_pi_l775_775699


namespace count_integers_congruent_to_2_mod_7_up_to_300_l775_775810

theorem count_integers_congruent_to_2_mod_7_up_to_300 : 
  (Finset.card (Finset.filter (λ n : ℕ, n % 7 = 2) (Finset.range 301))) = 43 := 
by
  sorry

end count_integers_congruent_to_2_mod_7_up_to_300_l775_775810


namespace garden_breadth_l775_775429

theorem garden_breadth (P L B : ℕ) (h₁ : P = 950) (h₂ : L = 375) (h₃ : P = 2 * (L + B)) : B = 100 := by
  sorry

end garden_breadth_l775_775429


namespace students_taking_neither_l775_775247

theorem students_taking_neither (total_students music_students art_students both_students : ℕ) 
  (htotal : total_students = 500)
  (hmusic : music_students = 40)
  (hart : art_students = 20)
  (hboth : both_students = 10) : 
  total_students - (music_students + art_students - both_students) = 450 := by
  have h := total_students - (music_students + art_students - both_students)
  rw [htotal, hmusic, hart, hboth] at h
  exact h

end students_taking_neither_l775_775247


namespace solve_system_of_equations_l775_775188

theorem solve_system_of_equations : ∃ x y : ℝ, (5 * x + y = 7) ∧ (6 * x - 2 * y = 18) ∧ x = 2 ∧ y = -3 :=
by {
  existsi (2 : ℝ),
  existsi (-3 : ℝ),
  split, { sorry },
  split, { sorry },
  split; refl,
}

end solve_system_of_equations_l775_775188


namespace pencil_lead_loss_l775_775575

theorem pencil_lead_loss (L r : ℝ) (h : r = L * 1/10):
  ((9/10 * r^3) * (2/3)) / (r^3) = 3/5 := 
by
  sorry

end pencil_lead_loss_l775_775575


namespace m_cubed_plus_m_inv_cubed_l775_775422

theorem m_cubed_plus_m_inv_cubed (m : ℝ) (h : m + 1/m = 10) : m^3 + 1/m^3 + 1 = 971 :=
sorry

end m_cubed_plus_m_inv_cubed_l775_775422


namespace complex_number_quadrant_l775_775383

variables {i : ℂ}
def imaginary_unit := i = complex.I
def z := (1 + i) * i

theorem complex_number_quadrant 
  (h1 : imaginary_unit) 
  (h2 : z = (1 + complex.I) * complex.I) : 
  z.re < 0 ∧ z.im > 0 := 
sorry

end complex_number_quadrant_l775_775383


namespace sum_distances_to_vertices_geq_6r_l775_775472

variables {A B C P : Type} [MetricSpace A B C P]
variables {PA PB PC r : ℝ} {triangle_ABC : Set A B C} 

noncomputable def is_inside (P : Point) (triangle : Set Point) : Prop := -- define when P is inside triangle

theorem sum_distances_to_vertices_geq_6r
  (hA : Point A)
  (hB : Point B)
  (hC : Point C)
  (hP : Point P)
  (hin : is_inside P triangle_ABC)
  (PA : dist P A)
  (PB : dist P B)
  (PC : dist P C)
  (r : ℝ)
  (hr : radius_incircle triangle_ABC = r)
  : PA + PB + PC ≥ 6 * r := 
sorry

end sum_distances_to_vertices_geq_6r_l775_775472


namespace jacket_cost_price_l775_775275

theorem jacket_cost_price :
  (∃ x : ℝ, 500 * 0.7 = x + 50) → (∃ x : ℝ, x = 300) :=
by
  intro hx
  obtain ⟨x, h⟩ := hx
  use x
  linarith
  sorry

end jacket_cost_price_l775_775275


namespace range_of_a_for_decreasing_function_l775_775393

noncomputable def f (x : ℝ) (a : ℝ) := Real.logBase 3 (x^2 + a*x + a + 5)

theorem range_of_a_for_decreasing_function :
  (∀ x, x < 1 → Real.logBase 3 (x^2 + a*x + a + 5) < Real.logBase 3 ((x + h)^2 + a*(x + h) + a + 5))
  → -3 ≤ a ∧ a ≤ -2 :=
by
  intro h_decreasing
  -- Skipping proof
  sorry

end range_of_a_for_decreasing_function_l775_775393


namespace sequence_a1_l775_775913

variable (S : ℕ → ℤ) (a : ℕ → ℤ)

def Sn_formula (n : ℕ) (a₁ : ℤ) : ℤ := (a₁ * (4^n - 1)) / 3

theorem sequence_a1 (h1 : ∀ n : ℕ, S n = Sn_formula n (a 1))
                    (h2 : a 4 = 32) :
  a 1 = 1 / 2 :=
by
  sorry

end sequence_a1_l775_775913


namespace unpainted_area_l775_775569

noncomputable def cross_area_unpainted (w1 w2 : ℝ) (angle : ℝ) : ℝ :=
  let hypotenuse := w1 * Real.sqrt 2 in
  w2 * hypotenuse

theorem unpainted_area (w1 w2 : ℝ) (angle : ℝ) (h : w1 = 5 ∧ w2 = 8 ∧ angle = π / 4) :
  cross_area_unpainted w1 w2 angle = 40 * Real.sqrt 2 :=
by
  rcases h with ⟨h1, h2, h3⟩
  simp only [cross_area_unpainted, h1, h2, h3, Real.pi_div_four, Real.sqrt]
  norm_num
  sorry

end unpainted_area_l775_775569


namespace find_intersection_of_normal_at_A_with_parabola_l775_775890

theorem find_intersection_of_normal_at_A_with_parabola {x : ℝ} {y : ℝ} 
  (A : (ℝ × ℝ)) (hA_on_parabola : A = (2, 4)) (hParabola : y = x^2) :
  let B := (-2.25, 5.0625) in 
  ∃ B : (ℝ × ℝ), (B.2 = B.1^2) ∧ (B.1 = -2.25) ∧ (B.2 = 5.0625) ∧
  (∃ c : ℝ, 
    (y - 4 = -1/4 * (x - 2)) ∧ 
    (y = -1/4 * x + c) ∧ 
    (y = x^2) ∧ 
    (B.1 ≠ 2)) := by sorry

end find_intersection_of_normal_at_A_with_parabola_l775_775890


namespace parabola_directrix_l775_775046

open Real

/--
Given the vertex of the parabola \( C \) is at the origin and the directrix has the equation \( x = 2 \).
1. Prove the equation of the parabola \( C \) is \( y^2 = -8x \).
2. Prove the length of the chord \( AB \) where the line \( l: y = x + 2 \) intersects \( C \).
-/
theorem parabola_directrix (C : ℝ → ℝ) :
  (∀ x y : ℝ, C y = y^2 / (-8 : ℝ)) ∧
  (∃ x1 x2 : ℝ, ∃ y1 y2 : ℝ, (y1 = x1 + 2 ∧ y2 = x2 + 2) ∧ 
    ((y1^2 = -8*x1) ∧ (y2^2 = -8*x2)) ∧
    (chord_length ==== (some_correct_length))) :=
  sorry

end parabola_directrix_l775_775046


namespace correct_answer_l775_775244

-- Conditions from the problem.

def proposition_A_correct : Prop :=
  ∀ (x y : ℝ), (x^2 + y^2 = 0) → (x = 0 ∧ y = 0)

def contrapositive_A_correct : Prop :=
  ∀ (x y : ℝ), (x ≠ 0 ∨ y ≠ 0) → (x^2 + y^2 ≠ 0)

def proposition_B : Prop :=
  ∃ (x0 : ℝ), x0^2 - x0 + 1 ≤ 0

def neg_prop_B : Prop :=
  ∀ (x : ℝ), x^2 - x + 1 > 0

def proposition_C (A B : ℝ) (sin_A sin_B : ℝ → ℝ) : Prop :=
  (sin_A A > sin_B B) ↔ (A > B)

def proposition_D (a b : E) [inner_product_space ℝ E] : Prop :=
  a ⬝ b < 0 → inner_product_space.angle a b = real.pi / 2

-- The proof statement.
theorem correct_answer : ¬proposition_D :=
  sorry

end correct_answer_l775_775244


namespace cans_in_third_bin_l775_775645

noncomputable def num_cans_in_bin (n : ℕ) : ℕ :=
  match n with
  | 1 => 2
  | 2 => 4
  | 3 => 7
  | 4 => 11
  | 5 => 16
  | _ => sorry

theorem cans_in_third_bin :
  num_cans_in_bin 3 = 7 :=
sorry

end cans_in_third_bin_l775_775645


namespace milan_billed_minutes_l775_775718

-- Define the conditions
def monthly_fee : ℝ := 2
def cost_per_minute : ℝ := 0.12
def total_bill : ℝ := 23.36

-- Define the number of minutes based on the above conditions
def minutes := (total_bill - monthly_fee) / cost_per_minute

-- Prove that the number of minutes is 178
theorem milan_billed_minutes : minutes = 178 := by
  -- Proof steps would go here, but as instructed, we use 'sorry' to skip the proof.
  sorry

end milan_billed_minutes_l775_775718


namespace five_f_is_perfect_square_iff_l775_775352

-- Define f(n) as the number of solutions to x + 2y + 5z = n
def f (n : ℕ) : ℕ :=
  Multiset.card { xyz : ℕ × ℕ × ℕ | let (x, y, z) := xyz in x + 2 * y + 5 * z = n }

-- Main theorem
theorem five_f_is_perfect_square_iff (n : ℕ) : 
  ∃k : ℤ, 5 * f n = (k * k) ↔ ∃k : ℤ, (n : ℤ) = 10 * k + 6 :=
sorry

end five_f_is_perfect_square_iff_l775_775352


namespace proof_of_length_QR_l775_775602

def length_QR (edge_length : ℝ) : ℝ :=
  let diagonal_length := real.sqrt (edge_length * edge_length + edge_length * edge_length)
  let QS := diagonal_length / 2
  real.sqrt (QS * QS  + edge_length * edge_length)

theorem proof_of_length_QR :
  length_QR 2 = real.sqrt 6 :=
by
  sorry

end proof_of_length_QR_l775_775602


namespace simplify_and_evaluate_expression_l775_775181

-- Define the condition
def condition (x y : ℝ) := (x - 2) ^ 2 + |y + 1| = 0

-- Define the expression
def expression (x y : ℝ) := 3 * x ^ 2 * y - (2 * x ^ 2 * y - 3 * (2 * x * y - x ^ 2 * y) + 5 * x * y)

-- State the theorem
theorem simplify_and_evaluate_expression (x y : ℝ) (h : condition x y) : expression x y = 6 :=
by
  sorry

end simplify_and_evaluate_expression_l775_775181


namespace arithmetic_sequence_difference_l775_775079

theorem arithmetic_sequence_difference (a b c : ℤ) (d : ℤ)
  (h1 : 9 - 1 = 4 * d)
  (h2 : c - a = 2 * d) :
  c - a = 4 := by sorry

end arithmetic_sequence_difference_l775_775079


namespace seashells_total_l775_775117

theorem seashells_total (joan_seashells jessica_seashells : ℕ)
  (h_joan : joan_seashells = 6)
  (h_jessica : jessica_seashells = 8) :
  joan_seashells + jessica_seashells = 14 :=
by 
  sorry

end seashells_total_l775_775117


namespace sam_total_time_is_correct_l775_775175

noncomputable def total_study_time : ℕ := 
  let science := 60 -- 1 hour in minutes
  let math := 80
  let literature := 40
  let history := 90 -- 1.5 hours in minutes
  let geography := 1800 / 60 -- 1800 seconds to minutes
  let physical_education := 1500 / 60 -- 1500 seconds to minutes
  science + math + literature + history + geography + physical_education

noncomputable def total_break_time : ℕ :=
  let breaks := 5
  let break_time := 15
  breaks * break_time

noncomputable def total_time : ℕ := total_study_time + total_break_time

noncomputable def total_hours : ℚ := total_time / 60

theorem sam_total_time_is_correct : total_hours ≈ 6.67 := 
  by
  -- To complete the proof, it would typically go here
  sorry

end sam_total_time_is_correct_l775_775175


namespace max_val_sqrt_sum_l775_775149

theorem max_val_sqrt_sum (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 8) :
  sqrt (3 * a + 2) + sqrt (3 * b + 2) + sqrt (3 * c + 2) ≤ sqrt 78 :=
sorry

end max_val_sqrt_sum_l775_775149


namespace Tucker_last_number_l775_775980

-- Define the sequence of numbers said by Todd, Tadd, and Tucker
def game_sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n = 2 then 2
  else if n = 3 then 3
  else if n = 4 then 4
  else if n = 5 then 5
  else if n = 6 then 6
  else sorry -- Define recursively for subsequent rounds

-- Condition: The game ends when they reach the number 1000.
def game_end := 1000

-- Define the function to determine the last number said by Tucker
def last_number_said_by_Tucker (end_num : ℕ) : ℕ :=
  -- Assuming this function correctly calculates the last number said by Tucker
  if end_num = game_end then 1000 else sorry

-- Problem statement to prove
theorem Tucker_last_number : last_number_said_by_Tucker game_end = 1000 := by
  sorry

end Tucker_last_number_l775_775980


namespace least_positive_integer_n_reducible_fraction_l775_775345

theorem least_positive_integer_n_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (2 * n - 26) ≠ 0 ∧ (∃ d : ℕ, d > 1 ∧ d ∣ (2 * n - 26) ∧ d ∣ (10 * n + 12)) ∧ 
  (∀ m : ℕ, 0 < m ∧ (m < n) → ((2 * m - 26) = 0 ∨ ∀ d : ℕ, d > 1 → ¬ (d ∣ (2 * m - 26) ∧ d ∣ (10 * m + 12)))) ∧
  n = 49 :=
begin
  sorry
end

end least_positive_integer_n_reducible_fraction_l775_775345


namespace area_ratio_l775_775855

def right_triangle_AB_12_BC_16_angleB_90 := 
  ∃ (A B C D E X : Point) (r : ℝ), 
    is_right_triangle A B C ∧
    A.distance_to B = 12 ∧
    B.distance_to C = 16 ∧
    (∠ ABC = π / 2) ∧
    midpoint D A B ∧
    midpoint E A C ∧
    segments_intersect CD BE X ∧
    circle X 2 inscribed_in_triangle ABC ∧
    ∃ (ratio : ℝ), 
      ratio = 72 * π / sqrt 39712 ∧
      ratio = area_of_circle_center_X_radius_2 / area_of_triangle_BXC

theorem area_ratio:
  right_triangle_AB_12_BC_16_angleB_90 →
  ∃ ratio : ℝ, ratio = 72 * π / sqrt 39712
:= 
by {
  assume h,
  use 72 * π / sqrt 39712,
  sorry
}

end area_ratio_l775_775855


namespace locate_quadrant_of_complex_l775_775086

theorem locate_quadrant_of_complex (z : ℂ) (hz : z = 3 - 1 * complex.I) : 
  point_in_quadrant z 4 :=
by sorry

end locate_quadrant_of_complex_l775_775086


namespace f_l775_775782

-- Define the function f
def f (x : ℝ) := Real.sin x / (Real.sin x + Real.cos x)

-- State the theorem
theorem f'_pi_over_2 : (deriv f) (Real.pi / 2) = 1 := by
  sorry

end f_l775_775782


namespace exists_divisible_by_2021_l775_775479

def concat_numbers (n m : ℕ) : ℕ :=
  -- function to concatenate numbers from n to m
  sorry

theorem exists_divisible_by_2021 :
  ∃ (n m : ℕ), n > m ∧ m ≥ 1 ∧ 2021 ∣ concat_numbers n m :=
by
  sorry

end exists_divisible_by_2021_l775_775479


namespace diagonal_ratio_l775_775842

variables {α : Type*} [add_comm_group α] [vector_space ℝ α]

-- Definitions for the points and vectors
def A : α := (0 : α)
def B (a : ℝ) : α := (2 * a, 0)
def D (a : ℝ) : α := (0, 2 * a * real.sqrt 3)
def C (a : ℝ) : α := (2 * a, 2 * a * real.sqrt 3)

-- Definitions for the diagonals
def AC (a : ℝ) : ℝ := dist A (C a)
def BD (a : ℝ) : ℝ := dist (B a) (D a)

-- Main theorem statement
theorem diagonal_ratio (a : ℝ) :
  AC a / BD a = 2 :=
begin
  sorry
end

end diagonal_ratio_l775_775842


namespace cost_per_kg_cherries_profit_from_selling_cherries_min_selling_price_large_cherries_l775_775615

/-- Given the conditions, find the cost per kilogram of small and large cherries. -/
theorem cost_per_kg_cherries 
    (total_weight : ℕ := 200)
    (total_cost : ℕ := 8000)
    (x : ℕ) -- cost per kilogram of small cherries
    (y : ℕ := x + 20) -- cost per kilogram of large cherries
    (cost_eq : total_weight * x + total_weight * y = total_cost) :
    x = 10 ∧ y = 30 := 
begin
  sorry
end

/-- Calculate the profit earned after selling all the cherries at given prices. -/
theorem profit_from_selling_cherries 
    (total_weight : ℕ := 200)
    (cost_small : ℕ := 10)
    (cost_large : ℕ := 30)
    (price_small : ℕ := 16)
    (price_large : ℕ := 40) :
    (total_weight * (price_large - cost_large)) + (total_weight * (price_small - cost_small)) = 3200 :=
begin
  sorry
end

/-- Determine the minimum selling price per kilogram of large cherries in the second purchase given the conditions. -/
theorem min_selling_price_large_cherries 
    (total_weight : ℕ := 200)
    (cost_small : ℕ := 10)
    (cost_large : ℕ := 30)
    (price_small : ℕ := 16)
    (previous_profit : ℕ := 3200)
    (loss_percent : ℕ := 20) -- 20% loss in small cherries
    (desired_profit_percent : ℕ := 90) -- 90% of previous profit
    (min_price_large : ℕ) :
    (total_weight * min_price_large) + (total_weight * price_small * (100 - loss_percent) / 100) - 8000 ≥ (previous_profit * desired_profit_percent / 100) →
    min_price_large ≥ 41.6 :=
begin
  sorry
end

end cost_per_kg_cherries_profit_from_selling_cherries_min_selling_price_large_cherries_l775_775615


namespace solve_equation_using_factoring_method_l775_775538

theorem solve_equation_using_factoring_method (x : ℝ) :
  (5 * x - 1) ^ 2 = 3 * (5 * x - 1) →
  -- The correct method to solve is factoring
  (∃ y z : ℝ, (5*x-1)*(5*x-1-3)=y ∧ y = 0 ∧ (5*x-1)=z ∧ z=0) :=
begin
  intros h,
  sorry
end

end solve_equation_using_factoring_method_l775_775538


namespace digit_150_in_fraction_l775_775989

-- Define the decimal expansion repeating sequence for the fraction 31/198
def repeat_seq : List Nat := [1, 5, 6, 5, 6, 5]

-- Define a function to get the nth digit of the repeating sequence
def nth_digit (n : Nat) : Nat :=
  repeat_seq.get! ((n - 1) % repeat_seq.length)

-- State the theorem to be proved
theorem digit_150_in_fraction : nth_digit 150 = 5 := 
sorry

end digit_150_in_fraction_l775_775989


namespace rotten_tomatoes_l775_775620

-- Conditions
def weight_per_crate := 20
def num_crates := 3
def total_cost := 330
def selling_price_per_kg := 6
def profit := 12

-- Derived data
def total_weight := num_crates * weight_per_crate
def total_revenue := profit + total_cost
def sold_weight := total_revenue / selling_price_per_kg

-- Proof statement
theorem rotten_tomatoes : total_weight - sold_weight = 3 := by
  sorry

end rotten_tomatoes_l775_775620


namespace peg_placement_unique_l775_775219

noncomputable def triangular_board_pegs := ℕ      -- number of ways to place the pegs

theorem peg_placement_unique :
  let yellow_pegs := 6 in
  let red_pegs := 5 in
  let green_pegs := 4 in
  let blue_pegs := 3 in
  let orange_pegs := 2 in
  let violet_pegs := 1 in
  let total_pegs := yellow_pegs + red_pegs + green_pegs + blue_pegs + orange_pegs + violet_pegs in
  (∀ row: ℕ, row ∈ {1, 2, 3, 4, 5, 6} → ∃! peg_placement : finset ℕ, peg_placement.card = row) ∧
  (∀ col: ℕ, col ∈ {1, 2, 3, 4, 5, 6} → ∃! peg_placement : finset ℕ, peg_placement.card = col) →
  triangular_board_pegs = 1 :=
by
  sorry

end peg_placement_unique_l775_775219


namespace total_children_l775_775550

theorem total_children (n : ℕ) (h₁ : 12 = 0.25 * n * (n + 2)) : n + 2 = 8 := 
by 
  sorry

end total_children_l775_775550


namespace no_partition_equal_product_l775_775516

theorem no_partition_equal_product (n : ℕ) : 
  ¬(∃ A B : finset ℕ, 
      A ∪ B = finset.range' n 18 ∧ 
      A ∩ B = ∅ ∧ 
      A ≠ ∅ ∧ 
      B ≠ ∅ ∧ 
      (∏ a in A, a) = (∏ b in B, b)) := 
sorry

end no_partition_equal_product_l775_775516


namespace sum_of_exponents_2023_l775_775224

theorem sum_of_exponents_2023 :
  ∃ (s : ℕ) (m : Fin s → ℕ) (b : Fin s → ℤ), 
  (∀ i j, i < j → m i > m j) ∧
  (∀ k, b k = 1 ∨ b k = -1) ∧
  (∑ i, b i * 3 ^ (m i) = 2023) ∧
  (∑ i, m i = 22) := 
by 
  sorry

end sum_of_exponents_2023_l775_775224


namespace angles_equal_l775_775450

-- Defining a trapezoid ABCD, and condition that AC equals BC
variable {A B C D M L N : Type}

-- Defining conditions
variable [is_trapezoid ABCD] (AC_eq_BC : AC = BC) (M_mid : midpoint M A B) (L_ext : extension_point L DA)
variable (LM_ext_BD : collinear_three_points L M N ∧ line_of_collinear_extended_to_intersect L M BD N)

-- Statement of the problem
theorem angles_equal (h : ∀ (M_mid : is_midpoint M A B) (L_ext : is_extension_point L DA) 
  (LM_ext_BD : is_intersect_lines L M BD N): ∠ ACL = ∠ BCN) : 
  ∠ ACL = ∠ BCN :=
sorry -- Proof is omitted

end angles_equal_l775_775450


namespace jane_change_l775_775878

theorem jane_change :
  let skirt_cost := 13
  let skirts := 2
  let blouse_cost := 6
  let blouses := 3
  let total_paid := 100
  let total_cost := (skirts * skirt_cost) + (blouses * blouse_cost)
  total_paid - total_cost = 56 :=
by
  sorry

end jane_change_l775_775878


namespace missing_fraction_l775_775967

-- Defining all the given fractions
def f1 : ℚ := 1 / 3
def f2 : ℚ := 1 / 2
def f3 : ℚ := 1 / 5
def f4 : ℚ := 1 / 4
def f5 : ℚ := -9 / 20
def f6 : ℚ := -5 / 6

-- Defining the total sum in decimal form
def total_sum : ℚ := 5 / 6  -- Since 0.8333333333333334 is equivalent to 5/6

-- Defining the sum of the given fractions
def given_sum : ℚ := f1 + f2 + f3 + f4 + f5 + f6

-- The Lean 4 statement to prove the missing fraction
theorem missing_fraction : ∃ x : ℚ, (given_sum + x = total_sum) ∧ x = 5 / 6 :=
by
  use 5 / 6
  constructor
  . sorry
  . rfl

end missing_fraction_l775_775967


namespace min_max_product_l775_775483

-- Given conditions
variables {x y : ℝ} -- Real variables x and y
variable h : 3 * x^2 + 6 * x * y + 4 * y^2 = 2

-- Theorem: The product of the minimum and maximum values of x^2 + 2xy + 3y^2 is 1/3
theorem min_max_product : 
  ∃ m M : ℝ, 
    (∀ z, z = x^2 + 2 * x * y + 3 * y^2 → z ≥ m) ∧ 
    (∀ z, z = x^2 + 2 * x * y + 3 * y^2 → z ≤ M) ∧ 
    m * M = 1 / 3 :=
by 
  sorry

end min_max_product_l775_775483


namespace angles_arithmetic_sequence_min_value_b_l775_775028

variables {α : Type*} [real α]

-- Define the conditions for the first proof
def condition1 (A B C : α) (a b c : α) : Prop := 
  2 * cos B * (c * cos A + a * cos C) = b

def condition2 (A B C : α) : Prop := 
  A + B + C = pi

def condition3 (B : α) : Prop := 
  B = pi / 3

-- Prove that angles form an arithmetic sequence
theorem angles_arithmetic_sequence (A B C : α) (a b c : α) 
  (h1 : condition1 A B C a b c) 
  (h2 : condition2 A B C) 
  (h3 : condition3 B) : 
  A + C = 2 * B :=
sorry

-- Define the conditions for the second proof
def area_condition (a b c B : α) : Prop := 
  (1 / 2) * a * c * sin B = (3 * sqrt 3) / 2

-- Prove the minimum value of b
theorem min_value_b (a b c B : α) 
  (h1 : area_condition a b c B) 
  (h2 : B = pi / 3) : 
  b^2 ≥ 6 :=
sorry

end angles_arithmetic_sequence_min_value_b_l775_775028


namespace triangle_area_l775_775252

/-- Given a triangle with a perimeter of 20 cm and an inradius of 2.5 cm,
prove that its area is 25 cm². -/
theorem triangle_area (perimeter : ℝ) (inradius : ℝ) (area : ℝ)
  (h1 : perimeter = 20) (h2 : inradius = 2.5) :
  area = 25 :=
by
  sorry

end triangle_area_l775_775252


namespace average_marks_of_first_class_l775_775539

theorem average_marks_of_first_class (n1 n2 : ℕ) (avg2 avg_all : ℝ)
  (h_n1 : n1 = 25) (h_n2 : n2 = 40) (h_avg2 : avg2 = 65) (h_avg_all : avg_all = 59.23076923076923) :
  ∃ (A : ℝ), A = 50 :=
by 
  sorry

end average_marks_of_first_class_l775_775539


namespace age_solution_l775_775248

noncomputable def age_problem : Prop :=
  ∃ (A B x : ℕ),
    A = B + 5 ∧
    A + B = 13 ∧
    3 * (A + x) = 4 * (B + x) ∧
    x = 11

theorem age_solution : age_problem :=
  sorry

end age_solution_l775_775248


namespace double_angle_quadrant_l775_775387

noncomputable def is_in_quadrant2 (α : ℝ) : Prop := ∃ k : ℤ, π / 2 + 2 * k * π < α ∧ α < π + 2 * k * π

noncomputable def is_in_quadrant3_or_4 (β : ℝ) : Prop :=
  ∃ k : ℤ, π + 4 * k * π < β ∧ β < 2 * π + 4 * k * π

theorem double_angle_quadrant (α : ℝ) (h : is_in_quadrant2 α) : is_in_quadrant3_or_4 (2 * α) :=
by
  sorry

end double_angle_quadrant_l775_775387


namespace problem_solution_l775_775685

theorem problem_solution (x : ℝ) (hx_pos : 0 < x) :
  x.sqrt * (20 - x).sqrt + (20 * x - x^3).sqrt ≥ 20 ↔ x = 20 ∨ x ≈ 4.14 :=
by
  sorry

end problem_solution_l775_775685


namespace tan_F_eq_sqrt_13_l775_775453

open Real

theorem tan_F_eq_sqrt_13 (D E F : ℝ) 
  (h1 : cot D * cot F = 1 / 3)
  (h2 : cot E * cot F = 1 / 8)
  (h3 : D + E + F = π) :
  tan F = sqrt 13 := 
by
  sorry

end tan_F_eq_sqrt_13_l775_775453


namespace fifty_dips_eq_onehundredtwentyfive_daps_l775_775080

section equivalence

variables (daps dops dips : Type) [HasEq daps] [HasEq dops] [HasEq dips]

-- Conditions
def five_daps_eq_four_dops (daps dops : Type) [HasEq daps] [HasEq dops] : Prop :=
  5 * daps = 4 * dops

def four_dops_eq_ten_dips (dops dips : Type) [HasEq dops] [HasEq dips] : Prop :=
  4 * dops = 10 * dips

-- Theorem to prove
theorem fifty_dips_eq_onehundredtwentyfive_daps
  (daps dops dips : Type) [HasEq daps] [HasEq dops] [HasEq dips]
  (h1 : five_daps_eq_four_dops daps dops)
  (h2 : four_dops_eq_ten_dips dops dips) :
  50 * dips = 125 * daps :=
sorry

end equivalence

end fifty_dips_eq_onehundredtwentyfive_daps_l775_775080


namespace amphibians_frog_count_l775_775847

-- Define the species of each amphibian
inductive Species
| toad
| frog
| salamander

open Species

-- Define amphibians
def Logan : Species := sorry
def Jack : Species := sorry
def Neil : Species := sorry
def Oscar : Species := sorry
def Peter : Species := salamander -- Peter is given as a salamander

-- Conditions as given statements
def logan_statement : Prop := (Neil ≠ Logan)
def jack_statement : Prop := (Oscar = frog)
def neil_statement : Prop := (Jack = frog)
def oscar_statement : Prop := (Logan = toad ∧ Jack = toad ∧ Oscar = toad ∨ Neil = toad)
def peter_statement : Prop := (Logan = Oscar)

-- Proof problem statement
theorem amphibians_frog_count :
  (logan_statement ∧ jack_statement ∧ neil_statement ∧ oscar_statement ∧ peter_statement)
  → (List.count (λ x => x = frog) [Logan, Jack, Neil, Oscar, Peter] = 1) :=
sorry

end amphibians_frog_count_l775_775847


namespace triangle_identity_l775_775515

theorem triangle_identity
  (A B C : ℝ) (a b c : ℝ)
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h1 : A + B + C = π)
  (h2 : a > 0) (h3 : b > 0) (h4 : c > 0) :
  (Real.sin A + Real.sin B + Real.sin C) * (Real.cot A + Real.cot B + Real.cot C) =
  0.5 * (a^2 + b^2 + c^2) * (1/(a * b) + 1/(a * c) + 1/(b * c)) := sorry

end triangle_identity_l775_775515


namespace height_of_bobby_flag_l775_775649

def area_of_fabric(f1_w : Nat, f1_h : Nat, f2_w : Nat, f2_h : Nat, f3_w : Nat, f3_h : Nat) : Nat :=
  (f1_w * f1_h) + (f2_w * f2_h) + (f3_w * f3_h)

def height_of_flag(area : Nat, length : Nat) : Nat :=
  area / length

theorem height_of_bobby_flag : 
  height_of_flag (area_of_fabric 8 5 10 7 5 5) 15 = 9 := 
by
  sorry

end height_of_bobby_flag_l775_775649


namespace geo_seq_a12_equal_96_l775_775447

def is_geometric (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

theorem geo_seq_a12_equal_96
  (a : ℕ → ℝ) (q : ℝ)
  (h0 : 1 < q)
  (h1 : is_geometric a q)
  (h2 : a 3 * a 7 = 72)
  (h3 : a 2 + a 8 = 27) :
  a 12 = 96 :=
sorry

end geo_seq_a12_equal_96_l775_775447


namespace milan_billed_minutes_l775_775713

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) 
  (h1 : monthly_fee = 2) 
  (h2 : cost_per_minute = 0.12) 
  (h3 : total_bill = 23.36) : 
  (total_bill - monthly_fee) / cost_per_minute = 178 := 
by 
  sorry

end milan_billed_minutes_l775_775713


namespace length_of_AC_l775_775853

-- Definitions from the problem
variable (AB BC CD DA : ℝ)
variable (angle_ADC : ℝ)
variable (AC : ℝ)

-- Conditions from the problem
def conditions : Prop :=
  AB = 10 ∧ BC = 10 ∧ CD = 17 ∧ DA = 17 ∧ angle_ADC = 120

-- The mathematically equivalent proof statement
theorem length_of_AC (h : conditions AB BC CD DA angle_ADC) : AC = Real.sqrt 867 := sorry

end length_of_AC_l775_775853


namespace integral_of_odd_function_l775_775042

-- Define the conditions: f is an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

-- Define the problem statement:
theorem integral_of_odd_function (f : ℝ → ℝ) (h_odd : is_odd_function f) :
  ∫ x in 1..3, (f (x - 2) + 1 / x) = Real.log 3 :=
  sorry

end integral_of_odd_function_l775_775042


namespace min_value_sum_of_products_l775_775490

noncomputable def diamond (A1 A2 A3 A4 : ℝ × ℝ) (side_length : ℝ) : Prop :=
  dist A1 A2 = side_length ∧
  dist A2 A3 = side_length ∧
  dist A3 A4 = side_length ∧
  dist A4 A1 = side_length ∧
  ∠ A1 A2 A3 = π / 6 

def rhombus_center (A1 A2 A3 A4 : ℝ × ℝ) : ℝ × ℝ :=
  ((A1.1 + A2.1 + A3.1 + A4.1) / 4, (A1.2 + A2.2 + A3.2) / 4)

theorem min_value_sum_of_products
  (A1 A2 A3 A4 P : ℝ × ℝ) 
  (h1 : diamond A1 A2 A3 A4 1) :
      ∑ 1_le_i_lt_j_le_4, ((P.1 - A1.1) * (P.1 - A2.1) + (P.2 - A1.2) * (P.2 - A2.2)) ≥ -1 :=
sorry

end min_value_sum_of_products_l775_775490


namespace trapezoid_area_l775_775110

-- Definitions of the conditions based on the given problem
variables (A B C D O : Type)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O]
variables (AB : segment A B) (BC : segment B C) (CD : segment C D) (DA : segment D A)
variables (OC : segment O C) (OD : segment O D)
variables (O_center : circle O) (inscribed_in_trapezoid : inscribed_circle_in_trapezoid O_center A B C D)

-- Hypotheses based on the conditions given
axiom angle_DAB_right : ∀ (A B D : Type), is_right_angle (angle A B D)
axiom OC_value : dist O C = 2
axiom OD_value : dist O D = 4

-- Goal: Prove the area of the trapezoid ABCD
theorem trapezoid_area (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space O] 
    (AB : segment A B) (BC : segment B C) (CD : segment C D) (DA : segment D A) 
    (OC : segment O C) (OD : segment O D)
    (O_center : circle O) (inscribed_in_trapezoid : inscribed_circle_in_trapezoid O_center A B C D)
    (angle_DAB_right : is_right_angle (angle D A B))
    (OC_value : dist O C = 2)
    (OD_value : dist O D = 4) :
    area_trapezoid A B C D = 72 / 5 := by
  sorry

end trapezoid_area_l775_775110


namespace area_of_quadrilateral_CKOL_l775_775455

-- Given values and conditions
variables {a b c S : ℝ}
variables (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c)
variables (S_pos : 0 < S)

-- Statement to prove
theorem area_of_quadrilateral_CKOL (area_ABC : real) :
  let CKOL_area := (a * b * S * (a + b + 2 * c)) / ((b + c) * (a + c) * (a + b + c)) in
  CKOL_area = (a * b * S * (a + b + 2 * c)) / ((b + c) * (a + c) * (a + b + c)) :=
by
  sorry

end area_of_quadrilateral_CKOL_l775_775455


namespace ratio_AB_CD_lengths_AB_CD_l775_775256

-- Given conditions as definitions
def ABD_triangle (A B D : Point) : Prop := true  -- In quadrilateral ABCD, a diagonal BD is drawn
def BCD_triangle (B C D : Point) : Prop := true  -- Circles are inscribed in triangles ABD and BCD
def Line_through_B_center_AM_M (A B D M : Point) (AM MD : ℚ) : Prop :=
  (AM = 8/5) ∧ (MD = 12/5)
def Line_through_D_center_BN_N (B C D N : Point) (BN NC : ℚ) : Prop :=
  (BN = 30/11) ∧ (NC = 25/11)

-- Mathematically equivalent proof problems
theorem ratio_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB / CD = 4 / 5 :=
by
  sorry

theorem lengths_AB_CD (A B C D M N : Point) (AM MD BN NC : ℚ) :
  ABD_triangle A B D → 
  BCD_triangle B C D →
  Line_through_B_center_AM_M A B D M AM MD → 
  Line_through_D_center_BN_N B C D N BN NC →
  AB + CD = 9 ∧
  AB - CD = -1 :=
by 
  sorry

end ratio_AB_CD_lengths_AB_CD_l775_775256


namespace smallest_angle_y_l775_775833

theorem smallest_angle_y (a b c : ℝ) (ha : a = 2) (hb : b = 2) (hc : c = 4) : 
  ∃ y, (∀ C, angle_in_degrees a b c C → C < y) ∧ y = 180 := 
by
  sorry

end smallest_angle_y_l775_775833


namespace problem1_problem2_l775_775797

-- Definitions of the sets A and B based on the given conditions
def A : Set ℝ := { x | x^2 - 6 * x + 8 < 0 }
def B (a : ℝ) : Set ℝ := { x | (x - a) * (x - 3 * a) < 0 }

-- Proof statement for problem (1)
theorem problem1 (a : ℝ) : (∀ x, x ∈ A → x ∈ (B a)) ↔ (4 / 3 ≤ a ∧ a ≤ 2) := by
  sorry

-- Proof statement for problem (2)
theorem problem2 (a : ℝ) : (∀ x, (x ∈ A ∧ x ∈ (B a)) ↔ (3 < x ∧ x < 4)) ↔ (a = 3) := by
  sorry

end problem1_problem2_l775_775797


namespace geometric_sequence_proof_sum_terms_sequence_proof_l775_775106

noncomputable def first_term_and_ratio 
    (a_n : ℕ → ℝ)
    (S_n : ℕ → ℝ)
    (S_2n : ℕ → ℝ)
    (n : ℕ) : ℝ × ℝ :=
  (2, 3) -- Representing the answer (a1, q)

noncomputable def sum_terms_sequence
    (A_1 : ℝ)
    (A_n : ℕ → ℝ)
    (c_n : ℕ → ℝ)
    (T_n : ℕ → ℝ)
    (n : ℕ) : ℝ :=
  (sqrt 3 / 3) * (tan ((4 * n - 1) / 12 * π) - 1) - n -- Representing the answer T_n

theorem geometric_sequence_proof 
  (a_n : ℕ → ℝ)
  (S_n : ℕ → ℝ)
  (S_2n : ℕ → ℝ)
  (h1 : a_n(4) = 54)
  (h2 : S_n(4) = 80)
  (h3 : S_2n(8) = 6560) :
  (first_term_and_ratio a_n S_n S_2n 4) = (2, 3) := by
  sorry

theorem sum_terms_sequence_proof 
  (A_1 : ℝ)
  (A_n : ℕ → ℝ)
  (c_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (n : ℕ)
  (h4 : A_1 = π / 4)
  (h5 : ∀ n ≥ 2, A_n(n) - A_n(n-1) = 2 * π / 6)
  (h6 : ∀ n, c_n(n) = tan (A_n(n)) * tan (A_n(n-1))) :
  (sum_terms_sequence A_1 A_n c_n T_n n) = (sqrt 3 / 3 * (tan ((4 * n - 1) / 12 * π) - 1) - n) := by
  sorry

end geometric_sequence_proof_sum_terms_sequence_proof_l775_775106


namespace part_a_part_b_l775_775458

-- Define the set of digits
def digits : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the sum to check for divisibility condition
def sum_cond (xs ys : List ℕ) : Prop :=
  let x_sum := xs.sum
  let y_sum := ys.sum
  y_sum > 0 ∧ x_sum % y_sum = 0

-- Shape a) "Ш"
def is_possible_shape_a (digits : List ℕ) : Prop :=
  ∃ (partition : List (List ℕ)), 
  partition.length = 2 ∧ 
  (∀ x ∈ partition, xs.sum = 36 / ∧ nonempty_partitions [ r.sum | r ∈ ] xs
   |
   ∀ partition, partition.length = 2 → ∃ (x ∈), ∃ (partition_map r partition)

-- Shape b) strips
def is_possible_shape_b (digits : List ℕ) : Prop :=
  ∀ partition, partition.length = 2 → ∃ (xs.sum = 36), ¬sum_cond partition

theorem part_a : is_possible_shape_a digits := sorry
theorem part_b : ¬is_possible_shape_b digits := sorry

end part_a_part_b_l775_775458


namespace analytical_expression_inequality_solution_l775_775384

noncomputable def f (x : ℝ) : ℝ := x / (1 + x^2)

-- Given the conditions
def conditions (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, -1 < x ∧ x < 1 → (f= (λ x, x / (1 + x^2)) ∧ f(-x) = -f(x) ∧ f (1/2) = 2 / 5)

-- Proving the analytical expression of the function
theorem analytical_expression (f : ℝ → ℝ) (h : conditions f) : ∀ x : ℝ, -1 < x ∧ x < 1 → f(x) = x / (1 + x^2) :=
sorry

-- Proving the inequality solution
theorem inequality_solution (f : ℝ → ℝ) (h : conditions f) (hf_increasing : ∀ x y : ℝ, -1 < x ∧ x < y ∧ y < 1 → f(x) < f(y)) :
  { x : ℝ | f(x-1) + f(x) < 0 } = { x : ℝ | 0 < x ∧ x < 1 / 2 } :=
sorry

end analytical_expression_inequality_solution_l775_775384


namespace find_range_of_f_l775_775361

def f (x : ℝ) : ℝ := 3 + 2 * 3^(x+1) - 9^x

theorem find_range_of_f : 
  ∀ x, -1 ≤ x ∧ x ≤ 2 → -24 ≤ f x ∧ f x ≤ 12 :=
sorry

end find_range_of_f_l775_775361


namespace milan_billed_minutes_l775_775714

theorem milan_billed_minutes (monthly_fee : ℝ) (cost_per_minute : ℝ) (total_bill : ℝ) (minutes : ℝ)
  (h1 : monthly_fee = 2)
  (h2 : cost_per_minute = 0.12)
  (h3 : total_bill = 23.36)
  (h4 : total_bill = monthly_fee + cost_per_minute * minutes)
  : minutes = 178 := 
sorry

end milan_billed_minutes_l775_775714


namespace correct_sunset_time_l775_775498

theorem correct_sunset_time 
  (daylight_hours : ℕ)
  (daylight_minutes : ℕ)
  (sunrise_hour : ℕ)
  (sunrise_minute : ℕ)
  (sunset_hour : ℕ)
  (sunset_minute : ℕ)
  (daylight_hours = 14)
  (daylight_minutes = 42)
  (sunrise_hour = 5)
  (sunrise_minute = 35)
  (sunset_hour = 8)
  (sunset_minute = 17) :
  let sunrise_time := sunrise_hour * 60 + sunrise_minute in
  let daylight_time := daylight_hours * 60 + daylight_minutes in
  let calculated_sunset_time := sunrise_time + daylight_time in
  let calculated_sunset_hour := (calculated_sunset_time / 60) % 24 in
  let calculated_sunset_minute := calculated_sunset_time % 60 in
  (calculated_sunset_hour = 20 ∧ calculated_sunset_minute = 17) → 
  (sunset_hour = 8 ∧ sunset_minute = 17) :=
by sorry

end correct_sunset_time_l775_775498


namespace normal_intersects_again_at_B_l775_775888

noncomputable def point := (ℚ, ℚ)

def A : point := (2, 4)

def parabola (x : ℚ) : ℚ := x^2

def is_on_parabola (p : point) : Prop := (parabola p.1) = p.2

def normal_slope (x : ℚ) : ℚ := - (1 / (2 * x))

def normal_line (p : point) (x : ℚ) : ℚ := - (1/(4 * p.1) * (x - p.1)) + p.2

theorem normal_intersects_again_at_B : 
  ∃ B : point,
    is_on_parabola A ∧
    is_on_parabola B ∧
    B ≠ A ∧
    ∀ x, x ≠ 2 → (parabola x) = (normal_line A x) ↔ x = -9/4 :=
  sorry

end normal_intersects_again_at_B_l775_775888


namespace product_of_xyz_is_correct_l775_775077

theorem product_of_xyz_is_correct : 
  ∃ x y z : ℤ, 
    (-3 * x + 4 * y - z = 28) ∧ 
    (3 * x - 2 * y + z = 8) ∧ 
    (x + y - z = 2) ∧ 
    (x * y * z = 2898) :=
by
  sorry

end product_of_xyz_is_correct_l775_775077


namespace f_periodic_l775_775790

variable {R : Type} [OrderedRing R] [TopologicalSpace R] [BorelSpace R]
variable {f : R → R}
variables {a b : R}
variable h1 : ∀ x : R, f x = f (2 * b - x)
variable h2 : ∀ x : R, f (a + x) = -f (a - x)
variable h3 : a ≠ b

theorem f_periodic : ∃ p : R, 0 < p ∧ ∀ x : R, f (x + p) = f x := by
  use 4 * (a - b)
  sorry

end f_periodic_l775_775790


namespace log_decreasing_on_interval_l775_775671

noncomputable def f (x : ℝ) : ℝ := log 2 (x^2 + 2 * x - 3)

theorem log_decreasing_on_interval :
  ∀ x y : ℝ, x ∈ set.Iic (-3) → y ∈ set.Iic (-3) → x < y → f y < f x := 
sorry

end log_decreasing_on_interval_l775_775671


namespace midpoints_collinear_l775_775485

noncomputable def are_collinear (A B C : Point) : Prop :=
  A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y) = 0

theorem midpoints_collinear (A B C D E F O M N : Point)
  (h1 : O = midpoint A B)
  (h2 : M = midpoint C D)
  (h3 : N = midpoint E F)
  (h4 : C ≠ D)
  (h5 : on_semicircle_with_diameter C D A B)
  (h6 : intersects_at (line_through A C) (line_through B D) F)
  (h7 : intersects_at (line_through A D) (line_through B C) E)
  : are_collinear O M N :=
sorry

end midpoints_collinear_l775_775485


namespace apples_rotations_l775_775979

theorem apples_rotations 
  (toby_baseballs : ℕ)
  (toby_baseball_rotations : ℕ)
  (friend_apples : ℕ)
  (total_winner_rotations : ℕ)
  (toby_total_rotations : toby_baseballs * toby_baseball_rotations = 400)
  (winner_total : total_winner_rotations = 404) :
  ∃ (friend_apple_rotations : ℕ), friend_apple_rotations = 81 :=
by {
  let friend_total_rotations := total_winner_rotations - 400,
  have h : friend_total_rotations = 4 := by {
    rw winner_total,
    norm_num,
  },
  let friend_apple_rotations := friend_total_rotations / friend_apples,
  have h2 : friend_apple_rotations = 1 := by {
    rw h,
    norm_num,
  },
  use 81,
  norm_num,
}

end apples_rotations_l775_775979


namespace parallelogram_AB_length_l775_775749

open EuclideanGeometry

/-- Given a parallelogram ABCD such that BD = 2 and 2(AD ⋅ AB) = |BC|^2, 
    then |AB| = 2. --/
theorem parallelogram_AB_length (A B C D : Point)
  (h_parallelogram : is_parallelogram A B C D)
  (h_BD : dist B D = 2)
  (h_AD_AB : 2 * (vector.dot (A - D) (A - B)) = real.norm_sq (B - C)) :
  dist A B = 2 := by
  sorry

end parallelogram_AB_length_l775_775749


namespace base10_to_base12_conversion_153_l775_775313

theorem base10_to_base12_conversion_153 :
  let B := 11
  let base12_representation := "B9"  -- Custom symbolic representation
  ∀ n : ℕ, (n = 153) → 
    (let a := n / 12 in
     let b := n % 12 in
     (a = B ∧ b = 9) →
       base12_representation = "B9") :=
begin
  intros,
  sorry
end

end base10_to_base12_conversion_153_l775_775313


namespace convex_ngon_can_be_divided_l775_775511

open Convex

theorem convex_ngon_can_be_divided (n : ℕ) (h : n ≥ 6) :
  ∃ (S : finset (fin 5 → ℝ)) (S' : finset (fin 5 → ℝ)),
    (∀ s ∈ S, Convex ℝ (finset.image id s)) ∧ 
    (∀ s' ∈ S', Convex ℝ (finset.image id s')) ∧
    (Convex ℝ (⋃₀ (S ∪ S'))) ∧
    (finset.card S + finset.card S' = n) :=
by
  sorry

end convex_ngon_can_be_divided_l775_775511


namespace children_positions_valid_l775_775862

-- Define the positions of the children
inductive Child
| Yan : Child
| Kolya : Child
| Vasya : Child
| Senya : Child

-- Define the position type
def Position := Nat

-- Define the positions of the children
variable (pos: Child -> Position)

-- Initial conditions
axiom senya_right_of_kolya : pos Child.Kolya < pos Child.Senya
axiom kolya_left_hand_to_vasya : pos Child.Kolya + 1 = pos Child.Vasya

-- The proof target
theorem children_positions_valid :
    ∃ (pos : Child -> Position),
      pos Child.Yan = 0 ∧
      pos Child.Kolya = 1 ∧
      pos Child.Vasya = 2 ∧
      pos Child.Senya = 3 := by
  sorry

end children_positions_valid_l775_775862


namespace sum_of_solutions_l775_775489

noncomputable def f : ℕ → ℕ
| 1       := 1
| (n + 1) := if n % 2 = 0 then 2 * f (n / 2 + 1) else 2 * f (n / 2 + 1) - 1

theorem sum_of_solutions 
  : (∑ x in (finset.filter (λ x, f x = 19 ∧ x ≤ 2019) (finset.range 2020)), x) = 1889 :=
by
sorr

end sum_of_solutions_l775_775489


namespace quadratic_inequality_empty_solution_set_l775_775374

theorem quadratic_inequality_empty_solution_set
  (a b c : ℝ)
  (h₁ : a > 0)
  (h₂ : ¬ ∃ x : ℝ, a * x^2 + b * x + c = 0) :
  {x : ℝ | a * x^2 + b * x + c < 0} = ∅ := 
by sorry

end quadratic_inequality_empty_solution_set_l775_775374


namespace cupboard_selling_percentage_l775_775229

theorem cupboard_selling_percentage (CP SP : ℝ) (h1 : CP = 6250) (h2 : SP + 1500 = 6250 * 1.12) :
  ((CP - SP) / CP) * 100 = 12 := by
sorry

end cupboard_selling_percentage_l775_775229


namespace problem_statement_l775_775030

variable {A B C D E F H : Point}
variable {a b c : ℝ}

-- Assume the conditions
variable (h_triangle : Triangle A B C)
variable (h_acute : AcuteTriangle h_triangle)
variable (h_altitudes : AltitudesIntersectAt h_triangle H A D B E C F)
variable (h_sides : Sides h_triangle BC a AC b AB c)

-- Statement to prove
theorem problem_statement : AH * AD + BH * BE + CH * CF = 1/2 * (a^2 + b^2 + c^2) :=
sorry

end problem_statement_l775_775030
