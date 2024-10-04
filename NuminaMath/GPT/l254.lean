import Mathlib

namespace monotonicity_of_f_range_of_a_l254_254256

open Real

noncomputable def f (x a : ℝ) : ℝ := log x + a / x

-- Monotonicity of f(x) within its domain
theorem monotonicity_of_f {a : ℝ} :
  (∀ x > 0, 0 < x → a ≤ 0 → has_deriv_at (λ x, f x a) ((1 - a / x) / x) x ∧ ((1 - a / x) / x > 0)) ∧
  (∀ x > 0, 0 < x < a → a > 0 → has_deriv_at (λ x, f x a) ((1 - a / x) / x) x ∧ ((1 - a / x) / x < 0)) ∧
  (∀ x > 0, x > a → a > 0 → has_deriv_at (λ x, f x a) ((1 - a / x) / x) x ∧ ((1 - a / x) / x > 0)) :=
sorry

-- Range of values for a such that f(x) < 2x² always holds for x in (1/2, +∞)
theorem range_of_a {a : ℝ} :
  (∀ x > 1/2, f x a < 2 * x^2) ↔ a < (1/4 + (1/2) * log 2) :=
sorry

end monotonicity_of_f_range_of_a_l254_254256


namespace intersection_M_N_l254_254657

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254657


namespace aluminum_mass_percentage_correct_l254_254201

variable (mol_mass_Al : ℝ) (mol_mass_S : ℝ) (mol_mass_Al2S3 : ℝ)

-- Given conditions
def molar_mass_Al := 26.98
def molar_mass_S := 32.06
def molar_mass_Al2S3 := 2 * molar_mass_Al + 3 * molar_mass_S

-- The main statement to prove
theorem aluminum_mass_percentage_correct :
  (2 * molar_mass_Al / molar_mass_Al2S3) * 100 ≈ 36 :=
begin
  unfold molar_mass_Al molar_mass_S molar_mass_Al2S3,
  sorry
end

end aluminum_mass_percentage_correct_l254_254201


namespace lateral_surface_area_of_parallelepiped_is_correct_l254_254013

noncomputable def lateral_surface_area (diagonal : ℝ) (angle : ℝ) (base_area : ℝ) : ℝ :=
  let h := diagonal * Real.sin angle
  let s := diagonal * Real.cos angle
  let side1_sq := s ^ 2  -- represents DC^2 + AD^2
  let base_diag_sq := 25  -- already given as 25 from BD^2
  let added := side1_sq + 2 * base_area
  2 * h * Real.sqrt added

theorem lateral_surface_area_of_parallelepiped_is_correct :
  lateral_surface_area 10 (Real.pi / 3) 12 = 70 * Real.sqrt 3 :=
by
  sorry

end lateral_surface_area_of_parallelepiped_is_correct_l254_254013


namespace intersection_of_M_and_N_l254_254745

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254745


namespace triangle_proof_l254_254945

-- Define the key points and triangles involved
variables (A B C D E1 F1 : Point)
variables (triangle_ABC : Triangle ABC)
variables (isosceles_ABE1 : IsoscelesTriangle A B E1)
variables (isosceles_E1CF1 : IsoscelesTriangle E1 C F1)

-- Define the sides and angles based on descriptions in conditions
def sides_parallel (E1 B : Point) (F1 C : Point) : Prop := parallel E1 B F1 C
def angle_Equal (angle1 angle2 : ℝ) : Prop := angle1 = angle2

-- Prove the statement
theorem triangle_proof (congruent_BCE1_DCF1 : Congruent (Triangle B C E1) (Triangle D C F1)) : 
  length D F1 = length B C :=
by 
  have hyp1 : IsoscelesTriangle E1 C F1 := isosceles_E1CF1,
  have hyp2 : Congruent (Triangle B C E1) (Triangle D C F1) := congruent_BCE1_DCF1,
  sorry

end triangle_proof_l254_254945


namespace perfect_cubes_count_l254_254880

def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem perfect_cubes_count : 
  (∀ (n : ℕ), n ≥ 51 ∧ n ≤ 1999 → (is_perfect_cube n) -> 
  (n = 64 ∨ n = 125 ∨ n = 216 ∨ n = 343 ∨ n = 512 ∨ n = 729 ∨ n = 1000 ∨ n = 1331 ∨ n = 1728)) ∧
  (∀ (n : ℕ), n ≥ 50 ∧ n ≤ 2000 → (is_perfect_cube n) -> 
  ((n = 64 ∨ n = 125 ∨ n = 216 ∨ n = 343 ∨ n = 512 ∨ n = 729 ∨ n = 1000 ∨ n = 1331 ∨ n = 1728) -> True)) :=
begin
  sorry
end

end perfect_cubes_count_l254_254880


namespace circumradius_of_right_triangle_l254_254520

theorem circumradius_of_right_triangle (a b c : ℕ) (h : a = 8 ∧ b = 15 ∧ c = 17) : 
  ∃ R : ℝ, R = 8.5 :=
by
  sorry

end circumradius_of_right_triangle_l254_254520


namespace salary_percentage_increase_of_C_over_A_l254_254112

noncomputable def salaryRatio (x : ℕ) := (x, 2 * x, 3 * x)

theorem salary_percentage_increase_of_C_over_A :
  ∀ (x : ℕ), 
  let (Sa, Sb, Sc) := salaryRatio x in
  Sb + Sc = 6000 → 
  ((Sc - Sa) / Sa) * 100 = 200 :=
by 
  intros x h
  let Sa : ℕ := x
  let Sb : ℕ := 2 * x
  let Sc : ℕ := 3 * x
  rw [←nat.cast_add, h]
  have hx : x = 1200 := sorry
  rw [hx] 
  -- Further calculations would follow here, but are omitted
  sorry

end salary_percentage_increase_of_C_over_A_l254_254112


namespace smallest_value_range_l254_254041

theorem smallest_value_range {a : Fin 8 → ℝ}
  (h_sum : (∑ i, a i) = 4/3)
  (h_pos_7 : ∀ i : Fin 8, (∑ j in Finset.erase Finset.univ i, a j) > 0) :
  -8 < a 0 ∧ a 0 ≤ 1/6 :=
sorry

end smallest_value_range_l254_254041


namespace change_calculation_l254_254958

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l254_254958


namespace smallest_value_range_l254_254044

theorem smallest_value_range {a : Fin 8 → ℝ}
  (h_sum : (∑ i, a i) = 4/3)
  (h_pos_7 : ∀ i : Fin 8, (∑ j in Finset.erase Finset.univ i, a j) > 0) :
  -8 < a 0 ∧ a 0 ≤ 1/6 :=
sorry

end smallest_value_range_l254_254044


namespace intersection_of_M_and_N_l254_254640

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254640


namespace intersection_M_N_l254_254669

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254669


namespace proof_MN_parallel_O1O2_l254_254962

-- Variables
variables (A B C D P S T M N O1 O2 : Type)
variables [Quadrilateral A B C D] [IntersectsDiagonals A C B D P]
variables [Circumcircle (Triangle A B P) (Circle ω1 O1)]
variables [Circumcircle (Triangle C D P) (Circle ω2 O2)]
variables [MeetsSegmentAt BC ω1 B S]
variables [MeetsSegmentAt BC ω2 C T]
variables [MidpointOfMinorArcNotIncluding X: Type ≠ B SP: Type M]
variables [MidpointOfMinorArcNotIncluding Y: Type ≠ C TP: Type N]

-- Definitions and Theorem
def MN_parallel_to_O1O2 : Prop :=
  let quadrilateral := Quadrilateral A B C D,
      meet_diagonals := IntersectsDiagonals A C B D P,
      circumcircle_abp  := Circumcircle (Triangle A B P) (Circle ω1 O1),
      circumcircle_cdp  := Circumcircle (Triangle C D P) (Circle ω2 O2),
      meets_ω1 := MeetsSegmentAt BC ω1 B S,
      meets_ω2 := MeetsSegmentAt BC ω2 C T,
      midpoint_m := MidpointOfMinorArcNotIncluding X: Type ≠ B SP: Type M,
      midpoint_n := MidpointOfMinorArcNotIncluding Y: Type ≠ C TP: Type N
  in 
  MN ∥ O1O2

theorem proof_MN_parallel_O1O2 (A B C D P S T M N O1 O2 : Type)
  [Quadrilateral A B C D] [IntersectsDiagonals A C B D P]
  [Circumcircle (Triangle A B P) (Circle ω1 O1)]
  [Circumcircle (Triangle C D P) (Circle ω2 O2)]
  [MeetsSegmentAt BC ω1 B S]
  [MeetsSegmentAt BC ω2 C T]
  [MidpointOfMinorArcNotIncluding X: Type ≠ B SP: Type M]
  [MidpointOfMinorArcNotIncluding Y: Type ≠ C TP: Type N] :
  MN_parallel_to_O1O2 A B C D P S T M N O1 O2 := sorry

end proof_MN_parallel_O1O2_l254_254962


namespace change_proof_l254_254955

-- Definitions of the given conditions
def lee_money : ℕ := 10
def friend_money : ℕ := 8
def chicken_wings_cost : ℕ := 6
def chicken_salad_cost : ℕ := 4
def soda_cost : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Statement of the theorem
theorem change_proof : 
  let total_money : ℕ := lee_money + friend_money,
      meal_cost_before_tax : ℕ := chicken_wings_cost + chicken_salad_cost + num_sodas * soda_cost,
      total_meal_cost : ℕ := meal_cost_before_tax + tax
  in total_money - total_meal_cost = 3 := 
by
  -- We skip the proof, as it's not required per instructions
  sorry

end change_proof_l254_254955


namespace interest_rate_second_type_l254_254550

variable (totalInvestment : ℝ) (interestFirstTypeRate : ℝ) (investmentSecondType : ℝ) (totalInterestRate : ℝ) 
variable [Nontrivial ℝ]

theorem interest_rate_second_type :
    totalInvestment = 100000 ∧
    interestFirstTypeRate = 0.09 ∧
    investmentSecondType = 29999.999999999993 ∧
    totalInterestRate = 9 + 3 / 5 →
    (9.6 * totalInvestment - (interestFirstTypeRate * (totalInvestment - investmentSecondType))) / investmentSecondType = 0.11 :=
by
  sorry

end interest_rate_second_type_l254_254550


namespace positive_divisors_of_5400_multiple_of_5_l254_254877

-- Declare the necessary variables and conditions
theorem positive_divisors_of_5400_multiple_of_5 :
  let n := 5400
  let factorization := [(2, 2), (3, 3), (5, 2)]
  ∀ (a b c: ℕ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2 →
    (a*b*c).count(n) = 24 := 
sorry

end positive_divisors_of_5400_multiple_of_5_l254_254877


namespace M_inter_N_eq_2_4_l254_254822

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254822


namespace correct_calculation_l254_254096

-- Definitions of the conditions
def condition1 : Prop := 3 + Real.sqrt 3 ≠ 3 * Real.sqrt 3
def condition2 : Prop := 2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3
def condition3 : Prop := 2 * Real.sqrt 3 - Real.sqrt 3 ≠ 2
def condition4 : Prop := Real.sqrt 3 + Real.sqrt 2 ≠ Real.sqrt 5

-- Proposition using the conditions to state the correct calculation
theorem correct_calculation (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  2 * Real.sqrt 3 + Real.sqrt 3 = 3 * Real.sqrt 3 :=
by
  exact h2

end correct_calculation_l254_254096


namespace average_infection_num_two_rounds_l254_254893

theorem average_infection_num_two_rounds : 
  ∃ x : ℝ, 2 + 2 * x + (2 + 2 * x) * x = 50 ∧ x = 4 := 
by
  let x := 4
  use x
  split
  · calc
      2 + 2 * x + (2 + 2 * x) * x
      = 2 + 2 * 4 + (2 + 2 * 4) * 4 : by rfl
  · calc
      x = 4 : by rfl
  sorry

end average_infection_num_two_rounds_l254_254893


namespace positive_divisors_multiple_of_5_l254_254870

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end positive_divisors_multiple_of_5_l254_254870


namespace intersection_M_N_l254_254664

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254664


namespace not_all_ten_points_on_one_square_boundary_l254_254936

theorem not_all_ten_points_on_one_square_boundary
    (points : set (euclidean_space ℝ (fin 2))) :
    points.card = 10 →
    (∀ {p q r s : euclidean_space ℝ (fin 2)},
      p ∈ points → q ∈ points → r ∈ points → s ∈ points →
      is_on_boundary_of_square {p, q, r, s}) →
    ¬ (∃ square, ∀ p ∈ points, p ∈ boundary_of square) :=
begin
  sorry
end

end not_all_ten_points_on_one_square_boundary_l254_254936


namespace product_of_chords_l254_254968

theorem product_of_chords (r : ℝ) (h : r = 3) :
    let ω := complex.exp (2 * real.pi * complex.I / 10)
    let D := λ (k : ℕ), r * ω^k
    (∏ i in finset.range 1 5, complex.abs (r * (1 - ω^i))) * (∏ i in finset.range 6 10, complex.abs (r * (1 - ω^i))) = 65610 :=
by
  sorry

end product_of_chords_l254_254968


namespace func_fixed_point_l254_254429

theorem func_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : ∃ x, f x = 3 :=
by
  let f : ℝ → ℝ := λ x, a ^ (x - 2) + 2
  use 2
  unfold f
  calc
    f 2 = a ^ (2 - 2) + 2    : rfl
    ... = a ^ 0 + 2         : by congr; ring
    ... = 1 + 2             : by rw pow_zero; exact add_comm (2 : ℝ) (_)
    ... = 3                 : rfl

end func_fixed_point_l254_254429


namespace prob_negative_two_to_one_l254_254247

open Probability

noncomputable def ξ : Real := sorry

axiom dist_ξ : dist ξ = Normal 1 σ^2
axiom prob_ξ_leq_4 : P ξ ≤ 4 = 0.79

theorem prob_negative_two_to_one :
  P (-2 ≤ ξ ∧ ξ ≤ 1) = 0.29 :=
by sorry

end prob_negative_two_to_one_l254_254247


namespace vowel_soup_sequences_count_l254_254542

theorem vowel_soup_sequences_count :
  let vowels := 5
  let sequence_length := 6
  vowels ^ sequence_length = 15625 :=
by
  sorry

end vowel_soup_sequences_count_l254_254542


namespace sqrt_multiplication_l254_254095

theorem sqrt_multiplication :
  let a := 49 + 121
  let b := 64 - 49
  sqrt a * sqrt b = sqrt 2550 :=
by
  let a := 49 + 121
  let b := 64 - 49
  sorry

end sqrt_multiplication_l254_254095


namespace det_A_eq_26x_plus_24_l254_254582

def A (x : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![x + 2, x, x], ![x, x + 3, x], ![x, x, x + 4]]

theorem det_A_eq_26x_plus_24 (x : ℝ) : A x.det = 26 * x + 24 :=
by
  sorry

end det_A_eq_26x_plus_24_l254_254582


namespace XF_XG_value_l254_254399

-- Define the given conditions
noncomputable def AB := 4
noncomputable def BC := 3
noncomputable def CD := 7
noncomputable def DA := 9

noncomputable def DX (BD : ℚ) := (1 / 3) * BD
noncomputable def BY (BD : ℚ) := (1 / 4) * BD

-- Variables and points in the problem
variables (BD p q : ℚ)
variables (A B C D X Y E F G : Point)

-- Proof statement
theorem XF_XG_value 
(AB_eq : AB = 4) (BC_eq : BC = 3) (CD_eq : CD = 7) (DA_eq : DA = 9)
(DX_eq : DX BD = (1 / 3) * BD) (BY_eq : BY BD = (1 / 4) * BD)
(AC_BD_prod : p * q = 55) :
  XF * XG = (110 / 9) := 
by
  sorry

end XF_XG_value_l254_254399


namespace intersection_of_M_and_N_l254_254798

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254798


namespace probability_binomial_coeff_divisible_by_7_probability_binomial_coeff_divisible_by_12_l254_254483

noncomputable def binomial (n k : ℕ) : ℚ :=
  (nat.factorial n) / ((nat.factorial k) * (nat.factorial (n - k)))

theorem probability_binomial_coeff_divisible_by_7 : 
  (∀ (n : ℕ), n > 7 → ∃ (k : ℚ), k = 1 / 7 ↔ (binomial n 7 % 49 = 0)) :=
sorry

theorem probability_binomial_coeff_divisible_by_12 : 
  (∀ (n : ℕ), n > 7 → ∃ (k : ℚ), k = 91 / 144 ↔ (binomial n 7 % 12 = 0)) :=
sorry

end probability_binomial_coeff_divisible_by_7_probability_binomial_coeff_divisible_by_12_l254_254483


namespace total_paint_area_correct_l254_254144

def length : ℝ := 15
def width : ℝ := 12
def height : ℝ := 8
def window_length : ℝ := 3
def window_width : ℝ := 4
def num_windows : ℕ := 2

-- Define dimensions of the walls
def wall_area_length_height : ℝ := length * height
def wall_area_width_height : ℝ := width * height

-- Define dimensions of the ceiling
def ceiling_area : ℝ := length * width

-- Define window area
def window_area : ℝ := window_length * window_width * num_windows

-- Total wall area to be painted (both inside and outside)
def total_wall_area : ℝ := 2 * (wall_area_length_height + wall_area_width_height)

-- Total area to be painted excluding windows and including ceiling
def total_area_to_be_painted : ℝ := total_wall_area - window_area + ceiling_area

theorem total_paint_area_correct : total_area_to_be_painted = 1020 := by 
  sorry

end total_paint_area_correct_l254_254144


namespace intersection_M_N_l254_254628

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254628


namespace average_increase_l254_254122

variable (A : ℕ) -- The batsman's average before the 17th inning
variable (runs_in_17th_inning : ℕ := 86) -- Runs made in the 17th inning
variable (new_average : ℕ := 38) -- The average after the 17th inning
variable (total_runs_16_innings : ℕ := 16 * A) -- Total runs after 16 innings
variable (total_runs_after_17_innings : ℕ := total_runs_16_innings + runs_in_17th_inning) -- Total runs after 17 innings
variable (total_runs_should_be : ℕ := 17 * new_average) -- Theoretical total runs after 17 innings

theorem average_increase :
  total_runs_after_17_innings = total_runs_should_be → (new_average - A) = 3 :=
by
  sorry

end average_increase_l254_254122


namespace intersection_M_N_l254_254654

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254654


namespace enter_finals_judgment_by_median_l254_254331

-- Definitions
def num_students := 15
def top_advance := 8

-- Conditions
variable (scores : Fin num_students → ℝ)
variable (ming_score : ℝ)
hypothesis H_distinct : Function.Injective scores  -- All scores are different

-- Statement
theorem enter_finals_judgment_by_median (ming_score_le_median: ∃ k, ming_score = (list.sort (· ≤ ·) (list.ofFn scores)).nthLe (num_students / 2) (by norm_num) k) : 
  ming_score > (list.sort (· ≤ ·) (list.ofFn scores)).nthLe (num_students - top_advance) (by norm_num) → ming_score ∈ list.drop (num_students - top_advance) (list.sort (· ≤ ·) (list.ofFn scores)) :=
sorry

end enter_finals_judgment_by_median_l254_254331


namespace log_abs_is_even_l254_254340

noncomputable def f (x : ℝ) : ℝ := Real.log (|x|)

theorem log_abs_is_even : ∀ x : ℝ, f (-x) = f x := by
  intro x
  suffices h : ∀ x : ℝ, |x| = |-x| by
    rw [f, f, h]
  intro x
  exact abs_neg x

end log_abs_is_even_l254_254340


namespace h_at_3_l254_254992

-- Defining the functions f, g, and h
def f (x : ℝ) : ℝ := 3 * x + 6
def g (x : ℝ) : ℝ := Real.sqrt (f x) - 2
def h (x : ℝ) : ℝ := f (g x)

-- The theorem asserts that h(3) = 3 * Real.sqrt 15
theorem h_at_3 : h 3 = 3 * Real.sqrt 15 := by
  sorry

end h_at_3_l254_254992


namespace intersection_M_N_l254_254626

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254626


namespace sum_arithmetic_sequence_l254_254904

-- Define the arithmetic sequence condition and sum of given terms
def arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n d : ℕ, a n = a 1 + (n - 1) * d

def given_sum_condition (a : ℕ → ℕ) : Prop :=
  a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_arithmetic_sequence (a : ℕ → ℕ) (h_arith_seq : arithmetic_sequence a) 
  (h_sum_cond : given_sum_condition a) : a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 := 
by 
  sorry  -- Proof of the theorem

end sum_arithmetic_sequence_l254_254904


namespace probability_of_A_and_B_l254_254438

variable (A B : Prop)
variable (P : Prop → Prop)
variable [DecidablePred P]

axiom p_A : P A = 0.25
axiom p_B : P B = 0.40
axiom p_not_A_union_B : P (¬(A ∨ B)) = 0.55

theorem probability_of_A_and_B :
  P (A ∧ B) = 0.20 :=
by
  sorry

end probability_of_A_and_B_l254_254438


namespace intersection_M_N_l254_254738

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254738


namespace A_minus_B_is_perfect_square_l254_254463

noncomputable def A (k : ℕ) : ℕ :=
  let num7s := List.replicate (2 * k + 1) 7
  let A_str := "1" ++ String.intercalate "" (num7s.map (λ n => n.repr)) ++ "6"
  A_str.to_nat

noncomputable def B (k : ℕ) : ℕ :=
  let num5s := List.replicate k 5
  let B_str := "3" ++ String.intercalate "" (num5s.map (λ n => n.repr)) ++ "2"
  B_str.to_nat

theorem A_minus_B_is_perfect_square (k : ℕ) : 
  ∃ n : ℕ, A k - B k = n^2 :=
  sorry

end A_minus_B_is_perfect_square_l254_254463


namespace min_value_expression_l254_254241

theorem min_value_expression (x1 x2 x3 x4 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) (h4 : 0 < x4) (h_sum : x1 + x2 + x3 + x4 = Real.pi) :
  (2 * (Real.sin x1) ^ 2 + 1 / (Real.sin x1) ^ 2) *
  (2 * (Real.sin x2) ^ 2 + 1 / (Real.sin x2) ^ 2) *
  (2 * (Real.sin x3) ^ 2 + 1 / (Real.sin x3) ^ 2) *
  (2 * (Real.sin x4) ^ 2 + 1 / (Real.sin x4) ^ 2) = 81 := 
sorry

end min_value_expression_l254_254241


namespace best_play_wins_probability_l254_254081

theorem best_play_wins_probability (n : ℕ) :
  let p := (n! * n!) / (2 * n)! in
  1 - p = 1 - (fact n * fact n / fact (2 * n)) :=
sorry

end best_play_wins_probability_l254_254081


namespace intersection_M_N_l254_254733

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254733


namespace tan_alpha_eq_l254_254612

theorem tan_alpha_eq (a h : ℝ) (n : ℕ) (hn_odd : n % 2 = 1) 
  (right_triangle : ∃ A B C : Type, ∃ h : B ≠ C, triangle.right ∆ABC) :
  tan α = 4 * n * h / ((n ^ 2 - 1) * a) :=
sorry

end tan_alpha_eq_l254_254612


namespace incorrect_converse_C_l254_254546

structure parallelogram (P : Type) where
  diagonals_bisect_each_other : Prop
  one_pair_opposite_sides_parallel_and_other_pair_equal : Prop

structure quadrilateral (Q : Type) where
  is_parallelogram : Prop
  has_two_pairs_adjacent_angles_supplementary : Prop
  has_two_pairs_opposite_sides_equal : Prop

theorem incorrect_converse_C 
  (P : Type) [parallelogram P] 
  (Q : Type) [quadrilateral Q]
  (hA : ∀ (P : parallelogram). P.diagonals_bisect_each_other)
  (hB : ∀ (Q : quadrilateral). Q.has_two_pairs_adjacent_angles_supplementary → Q.is_parallelogram)
  (hC : ∀ (P : parallelogram). P.one_pair_opposite_sides_parallel_and_other_pair_equal)
  (hD : ∀ (Q : quadrilateral). Q.has_two_pairs_opposite_sides_equal → Q.is_parallelogram) :
  ¬ (∀ (Q : quadrilateral). Q.one_pair_opposite_sides_parallel_and_other_pair_equal → Q.is_parallelogram) :=
sorry

end incorrect_converse_C_l254_254546


namespace real_life_distance_between_cities_l254_254419

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end real_life_distance_between_cities_l254_254419


namespace sum_first_2016_terms_l254_254616

open BigOperators

-- Definitions based on the problem conditions
def a (n : ℕ) : ℕ := n
def S (n : ℕ) : ℕ := n * (n + 1) / 2  -- Sum of first n natural numbers

-- The problem statement in Lean 4 notation
theorem sum_first_2016_terms : S 5 = 15 → (∃ a1 : ℕ, a 5 = 5 ∧ S 5 = 15 ∧ a 1 = a1) →
  ∑ k in Finset.range 2016, (1 / (a k * a (k + 1))) = (2016 / 2017) := 
begin
  sorry
end

end sum_first_2016_terms_l254_254616


namespace MeatMarket_sales_l254_254387

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end MeatMarket_sales_l254_254387


namespace log_monotonically_decreasing_interval_l254_254854

theorem log_monotonically_decreasing_interval (a : ℝ) (h_a : a > 1) :
  ∃ I, I = set.Iio (-3 : ℝ) ∧ (∀ x ∈ I, 0 < log a (x^2 + 2 * x - 3) ∧ has_deriv_at (λ x, log a (x^2 + 2 * x - 3)) _ x ∧ (∀ y ∈ I, y ≠ x ∧ y < x → (log a (y^2 + 2 * y - 3) < log a (x^2 + 2 * x - 3)))) :=
by
  sorry

end log_monotonically_decreasing_interval_l254_254854


namespace smallest_value_of_a1_conditions_l254_254036

noncomputable theory

variables {a1 a2 a3 a4 a5 a6 a7 a8 : ℝ}

/-- The smallest value of \(a_1\) when the sum of \(a_1, \ldots, a_8\) is \(4/3\) 
    and the sum of any seven of these numbers is positive. -/
theorem smallest_value_of_a1_conditions 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 / 3)
  (h_sum_seven : ∀ i : {j // j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8}, 
                  0 < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 - a i.val) :
  -8 < a1 ∧ a1 ≤ 1 / 6 :=
sorry

end smallest_value_of_a1_conditions_l254_254036


namespace solve_for_x_l254_254191

theorem solve_for_x (x : ℝ) : 16^(x + 2) = 80 + 64 * 16^x ↔ x = 0.25 * Real.log2 (5 / 12) :=
by {
  sorry
}

end solve_for_x_l254_254191


namespace find_largest_smallest_A_l254_254169

theorem find_largest_smallest_A :
  ∃ (B : ℕ), B > 7777777 ∧ B.gcd 36 = 1 ∧
  (let A := 10^7 * (B % 10) + (B / 10) in A = 99999998 ∨ A = 17777779) :=
begin
  sorry
end

end find_largest_smallest_A_l254_254169


namespace max_distinct_numbers_l254_254475

theorem max_distinct_numbers : 
  ∃ (S : Finset ℕ), S.card = 400 ∧ S ⊆ (Finset.range 1000).filter (λ x, 1 ≤ x ∧ x ≤ 1000) ∧ 
    ∀ x y ∈ S, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6) :=
by
  sorry

end max_distinct_numbers_l254_254475


namespace factorization_correct_l254_254488

theorem factorization_correct : 
    (x(x + 1) = x^2 + x → False) ∧
    (x^2 + 2x + 1 = (x + 1)^2) ∧
    (x^2 + xy - 3 = x(x + y - 3) → False) ∧
    (x^2 + 6x + 4 = (x + 3)^2 - 5 → False) :=
by
  sorry

end factorization_correct_l254_254488


namespace optimal_screen_position_l254_254011

open Real

-- Define the scenario with necessary variables and conditions
def maximize_screen_area (O A B : Point) (θ : ℝ) : Prop :=
  let screen_length : ℝ := 4
  in (dist O A = screen_length) ∧ (dist O B = screen_length) ∧ angle O A B = θ

-- The theorem to prove that the screens should form a 45° angle with the adjacent walls
theorem optimal_screen_position (O A B : Point) :
  maximize_screen_area O A B (π / 4) :=
sorry

end optimal_screen_position_l254_254011


namespace geometric_progression_product_l254_254308

variables (b q T U U' : ℝ)
variables (h1 : U = b * (1 - q^5) / (1 - q))
variables (h2 : U' = (q^5 - 1) / (b * (q - 1)))
variables (h3 : T = b^5 * q^10)

theorem geometric_progression_product :
  T = (U * U') ^ 2.5 :=
sorry

end geometric_progression_product_l254_254308


namespace twentieth_special_number_is_28_l254_254435

-- Define what it means for a number to be special.
def is_special (n : Nat) (special_nums : List Nat) : Bool :=
  n = 1 ∨ (n > 1 ∧ Nat.gcd n (special_nums.sum) = 1)

-- Define a function to compute the list of special numbers up to n.
def special_numbers_up_to (n : Nat) : List Nat :=
  List.recOn (List.range n) [] (λ x xs, if is_special x xs then x :: xs else xs)

-- Property to prove: The 20th special number is 28.
theorem twentieth_special_number_is_28 : special_numbers_up_to 45 |>.getD 19 0 = 28 :=
  sorry

end twentieth_special_number_is_28_l254_254435


namespace max_distinct_numbers_l254_254476

theorem max_distinct_numbers : 
  ∃ (S : Finset ℕ), S.card = 400 ∧ S ⊆ (Finset.range 1000).filter (λ x, 1 ≤ x ∧ x ≤ 1000) ∧ 
    ∀ x y ∈ S, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6) :=
by
  sorry

end max_distinct_numbers_l254_254476


namespace max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254467

def max_distinct_numbers (n : ℕ) (k : ℕ) (f : ℕ → ℕ → Prop) : ℕ :=
  sorry

theorem max_num_distinct_from_1_to_1000_no_diff_4_5_6 :
  max_distinct_numbers 1000 4 (λ a b, ¬(a - b = 4 ∨ a - b = 5 ∨ a - b = 6)) = 400 :=
sorry

end max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254467


namespace bob_has_17_pennies_l254_254897

-- Definitions based on the problem conditions
variable (a b : ℕ)
def condition1 : Prop := b + 1 = 4 * (a - 1)
def condition2 : Prop := b - 2 = 2 * (a + 2)

-- The main statement to be proven
theorem bob_has_17_pennies (a b : ℕ) (h1 : condition1 a b) (h2 : condition2 a b) : b = 17 :=
by
  sorry

end bob_has_17_pennies_l254_254897


namespace proof_of_independence_l254_254101

/-- A line passing through the plane of two parallel lines and intersecting one of them also intersects the other. -/
def independent_of_parallel_postulate (statement : String) : Prop :=
  statement = "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other."

theorem proof_of_independence :
  independent_of_parallel_postulate "A line passing through the plane of two parallel lines and intersecting one of them also intersects the other." :=
sorry

end proof_of_independence_l254_254101


namespace smallest_number_bounds_l254_254034

theorem smallest_number_bounds (a : Fin 8 → ℝ)
    (h_sum : (∑ i, a i) = (4 / 3))
    (h_pos : ∀ i, (∑ j in finset.univ.filter (λ j, j ≠ i), a j) > 0) :
    -8 < (finset.univ.arg_min id a).get (finset.univ.nonempty) ∧
    (finset.univ.arg_min id a).get (finset.univ.nonempty) ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l254_254034


namespace composition_result_l254_254974

noncomputable def P (x : ℝ) : ℝ := 3 * real.sqrt x
def Q (x : ℝ) : ℝ := x^2

theorem composition_result : P (Q (P (Q (P (Q 2))))) = 54 :=
by
  sorry

end composition_result_l254_254974


namespace intersection_M_N_l254_254806

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254806


namespace stockholm_to_uppsala_distance_l254_254422

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end stockholm_to_uppsala_distance_l254_254422


namespace arrangement_count_l254_254552

/--
There are 12 different ways to arrange the letters a, a, b, b, c, c 
in three rows and two columns such that each row and each column 
contains different letters.
-/
theorem arrangement_count : 
  ∃ (arrangements : finset (fin 3 → fin 2 → char)), 
  set.card arrangements = 12 ∧ 
  all_unique_rows_and_columns arrangements := 
sorry

/-- 
Helper predicate to check uniqueness constraint in rows and columns
-/
def all_unique_rows_and_columns (arrangements : finset (fin 3 → fin 2 → char)) : Prop :=
  ∀ (matrix ∈ arrangements), 
    (∀ i : fin 3, function.injective (matrix i)) ∧ 
    (∀ j : fin 2, function.injective (fun i => matrix i j))
sorry

end arrangement_count_l254_254552


namespace magnitude_of_sum_l254_254838

variables (a b : EuclideanSpace ℝ (Fin 3))
variables (ha : ‖a‖ = 2) (hb : ‖b‖ = 1) (angle_ab : real.angleBetween a b = 120)

theorem magnitude_of_sum :
  ‖a + 2 • b‖ = 2 :=
sorry

end magnitude_of_sum_l254_254838


namespace intersection_M_N_l254_254671

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254671


namespace two_three_digit_numbers_sum_multiple_252_ratio_2_1_l254_254599

theorem two_three_digit_numbers_sum_multiple_252_ratio_2_1 (n : ℤ)
  (h1 : 1 < n) (h2 : n < 6) :
  ∃ A B : ℤ, A = 168 * n ∧ B = 84 * n ∧ 100 ≤ B ∧ B ≤ 999 ∧ 100 ≤ A ∧ A ≤ 999 ∧ (A + B) % 252 = 0 :=
by
  use 168 * n, 84 * n
  split
  . refl
  split
  . refl
  split
  . linarith
  split
  . linarith
  split
  . linarith
  . linarith
  . simp [mul_add, mul_comm]
  . sorry -- Proof that A + B is divisible by 252

end two_three_digit_numbers_sum_multiple_252_ratio_2_1_l254_254599


namespace parabola_passing_through_origin_l254_254495

def is_parabola_through_origin (a b c : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f x = y ↔ y = ax^2 + bx + c ∧ f 0 = 0 

theorem parabola_passing_through_origin : is_parabola_through_origin 1 0 0 (λ x, x^2) :=
by
  sorry

end parabola_passing_through_origin_l254_254495


namespace max_terms_quadratic_trinomial_l254_254510

theorem max_terms_quadratic_trinomial (a b c : ℝ) :
  let P := λ x, a * x^2 + b * x + c in
  ∃ n1 n2 : ℕ, n1 ≥ 2 ∧ n2 ≥ 2 ∧ n1 ≠ n2 ∧
  P (n1 + 1) = P n1 + P (n1 - 1) ∧ P (n2 + 1) = P n2 + P (n2 - 1) ∧ 
  (∀ n : ℕ, n ≥ 2 → n ≠ n1 → n ≠ n2 → P (n + 1) ≠ P n + P (n - 1)).

end max_terms_quadratic_trinomial_l254_254510


namespace best_play_wins_probability_best_play_wins_with_certainty_l254_254079

-- Define the conditions

variables (n : ℕ)

-- Part (a): Probability that the best play wins
theorem best_play_wins_probability (hn_pos : 0 < n) : 
  1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) = 1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) :=
  by sorry

-- Part (b): With more than two plays, the best play wins with certainty
theorem best_play_wins_with_certainty (s : ℕ) (hs : 2 < s) : 
  1 = 1 :=
  by sorry

end best_play_wins_probability_best_play_wins_with_certainty_l254_254079


namespace intersection_of_M_and_N_l254_254743

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254743


namespace find_valid_A_l254_254166

-- Define condition that B is coprime with 36
def coprime_with_36 (b : ℕ) : Prop :=
  Nat.coprime b 36

-- Define condition that moving the last digit to the first place forms A
def derived_A (b : ℕ) : ℕ :=
  let last_digit := b % 10
  let remaining := b / 10
  10^7 * last_digit + remaining

-- Define the range of B based on the problem statement
def valid_range (b : ℕ) : Prop :=
  b > 7777777 ∧ b < 10^8

-- Define the smallest and largest valid A
def smallest_valid_A (a : ℕ) : Prop :=
  a = 17777779

def largest_valid_A (a : ℕ) : Prop :=
  a = 99999998

-- Proof goal statement
theorem find_valid_A (b : ℕ) (h1 : coprime_with_36 b) (h2 : valid_range b) :
  smallest_valid_A (derived_A b) ∨ largest_valid_A (derived_A b) :=
sorry

end find_valid_A_l254_254166


namespace domain_of_log_function_l254_254423

def function_domain : Set ℝ :=
  {x : ℝ | x < -Real.sqrt 6 ∨ x > Real.sqrt 6}

theorem domain_of_log_function :
  ∀ x : ℝ, (∃ y : ℝ, f x = log 2 y ∧ y = x^2 - 6) ↔ x ∈ function_domain := by
  sorry

end domain_of_log_function_l254_254423


namespace vector_opposite_magnitude_l254_254228

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2)

theorem vector_opposite_magnitude :
  ∀ (x : ℝ), 
    (∃ x, (∃ y, (1, -1) = y ∧ (2, x) = (2, x)) ∧ (x : ℝ, 2) = (x, 2)) → 
    (1 / (x + 1) = x / 2) → 
    x = -2 → 
    magnitude (x, 2) = 2 * real.sqrt 2 :=
begin
  intros x h1 h2 h3,
  sorry
end

end vector_opposite_magnitude_l254_254228


namespace arrangement_count_of_letters_l254_254890

theorem arrangement_count_of_letters : 
  let letters := ['B1', 'B2', 'A1', 'A2', 'A3', 'N1', 'N2'] in 
  (list.permutations letters).length = 5040 :=
by
  let letters := ['B1', 'B2', 'A1', 'A2', 'A3', 'N1', 'N2']
  rw [list.permutations, list.length]
  sorry

end arrangement_count_of_letters_l254_254890


namespace perimeter_eq_20a_l254_254322

variables (A B C H M K : Type) [IsoscelesTriangle A B C] [AltitudeFromAToBC A B C H]
variables [MidpointOfAB M B] [PerpendicularFromM A K]
variables (a : ℝ) (AK : ℝ)

-- Given conditions
axiom AB_eq_BC : AB = BC
axiom AH_eq_MK : AH = MK
axiom AK_eq_a : AK = a

-- Proof statement
theorem perimeter_eq_20a : perimeter A B C = 20 * a :=
by
  -- Hypotheses based on given conditions
  sorry

end perimeter_eq_20a_l254_254322


namespace integer_values_abs_lt_5pi_l254_254869

theorem integer_values_abs_lt_5pi : 
  ∃ n : ℕ, n = 31 ∧ ∀ x : ℤ, |(x : ℝ)| < 5 * Real.pi → x ∈ (Finset.Icc (-15) 15) := 
sorry

end integer_values_abs_lt_5pi_l254_254869


namespace intersection_M_N_l254_254721

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254721


namespace double_luckiness_l254_254600

variable (oats marshmallows : ℕ)
variable (initial_luckiness doubled_luckiness : ℚ)

def luckiness (marshmallows total_pieces : ℕ) : ℚ :=
  marshmallows / total_pieces

theorem double_luckiness (h_oats : oats = 90) (h_marshmallows : marshmallows = 9)
  (h_initial : initial_luckiness = luckiness marshmallows (oats + marshmallows))
  (h_doubled : doubled_luckiness = 2 * initial_luckiness) :
  ∃ x : ℕ, doubled_luckiness = luckiness (marshmallows + x) (oats + marshmallows + x) :=
  sorry

#check double_luckiness

end double_luckiness_l254_254600


namespace sum_of_10_consecutive_terms_l254_254927

theorem sum_of_10_consecutive_terms 
  (a : ℕ → ℕ)
  (a_1 : a 1 = 3)
  (d : ℕ)
  (d_val : d = 2)
  (sum_of_9_terms : ∃ (n m : ℕ), 0 < m ∧ m < 9 ∧ m ∈ Set.Ico 1 9 ∧ 
    (∑ i in Finset.Ico n (n + 10), a (i+1)) - a (n+m+1) = 185) :
  ∑ i in Finset.Ico 0 10, (a (i + 1)) = 200 :=
sorry

end sum_of_10_consecutive_terms_l254_254927


namespace trip_time_is_17_point_4_l254_254459

-- Definitions from the conditions
def totalDistance : ℝ := 120
def carSpeed : ℝ := 25
def walkSpeed : ℝ := 4

def T (d1 : ℝ) : ℝ := d1 / carSpeed + (totalDistance - d1) / walkSpeed

theorem trip_time_is_17_point_4 (d1 d2 : ℝ) :
  (d1 / carSpeed + (totalDistance - d1) / walkSpeed = 
   d1 / carSpeed + d2 / carSpeed + (totalDistance - (d1 - d2)) / carSpeed) ∧
  (d1 / carSpeed + (totalDistance - d1) / walkSpeed = 17.4) := 
begin
  sorry
end

end trip_time_is_17_point_4_l254_254459


namespace line_eqn_m_b_l254_254137

def point := (2, 5 : ℝ)
def slope := -3

theorem line_eqn_m_b 
  (y1 x1 m : ℝ) 
  (h1: m = slope) 
  (h2: point = (x1, y1)) :
  ∃ b : ℝ, y1 = m * x1 + b ∧ m + b = 8 :=
by
  let b := y1 - m * x1
  use b
  split
  sorry
  have hb: b = 11 := by
    sorry
  rw [hb]
  sorry

end line_eqn_m_b_l254_254137


namespace xe_x_monotonic_increasing_l254_254238

def is_monotonic_increasing (f: ℝ → ℝ) (I: set ℝ) : Prop :=
  ∀ x1 x2 ∈ I, x1 ≤ x2 → f x1 ≤ f x2

theorem xe_x_monotonic_increasing : is_monotonic_increasing (λ x, x * Real.exp x) {x : ℝ | -1 ≤ x} :=
sorry

end xe_x_monotonic_increasing_l254_254238


namespace bug_visits_tiles_l254_254531

/-- A rectangular floor is 15 feet wide and 20 feet long, tiled with one-foot square tiles.
 A bug starts at the midpoint of one of the shorter sides and walks in a straight line to the 
 midpoint of the opposite side. Prove that the bug visits 20 tiles, including the first and 
 the last tile. -/
theorem bug_visits_tiles :
  let width := 15
  let length := 20
  let start := (0, 7.5: ℝ)
  let end := (20, 7.5: ℝ)
  let tiles_crossed := 20
  tiles_crossed = 20 := by
  sorry

end bug_visits_tiles_l254_254531


namespace better_fitting_model_l254_254084

theorem better_fitting_model (R_A R_B : ℝ) (hA : R_A = 0.32) (hB : R_B = 0.91) :
  R_B > R_A :=
by
  rw [hA, hB]
  linarith

end better_fitting_model_l254_254084


namespace maximum_numbers_l254_254482

theorem maximum_numbers (S : Finset ℕ) (h1 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 1000)
                       (h2 : ∀ x y ∈ S, x ≠ y → (x - y).natAbs ≠ 4 ∧ (x - y).natAbs ≠ 5 ∧ (x - y).natAbs ≠ 6) :
  S.card ≤ 400 :=
begin
  sorry
end

end maximum_numbers_l254_254482


namespace solution_set_of_inequality_l254_254442

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x^2 - 3 * x - 2 > 0} = {x : ℝ | x < -0.5 ∨ x > 2} := 
sorry

end solution_set_of_inequality_l254_254442


namespace intersection_of_M_and_N_l254_254739

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254739


namespace intersection_M_N_l254_254691

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254691


namespace shirt_price_final_l254_254021

noncomputable def final_price (P : ℝ) := P * 0.9775

theorem shirt_price_final (P : ℝ) :
  let increased_price := P * 1.15 in
  let decreased_price := increased_price * 0.85 in
  decreased_price = P * 0.9775 :=
by
  let increased_price := P * 1.15
  let decreased_price := increased_price * 0.85
  sorry

end shirt_price_final_l254_254021


namespace segment_equivalence_l254_254932

theorem segment_equivalence
  (A B D E F C : Type)
  [linear_ordered_field A]
  [linear_ordered_field B]
  [linear_ordered_field D]
  [linear_ordered_field E]
  [linear_ordered_field F]
  [linear_ordered_field C]
  (segment_AB : E)
  (segment_AD : F)
  (point_E : A)
  (point_F : B)
  (point_C : C)
  (intersect_C : D)
  (h1 : E + point_C = F + intersect_C):
  segment_AB + point_C = segment_AD + intersect_C :=
sorry

end segment_equivalence_l254_254932


namespace intersection_M_N_l254_254687

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254687


namespace intersection_M_N_l254_254782

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254782


namespace exists_consecutive_integers_sum_cube_l254_254578

theorem exists_consecutive_integers_sum_cube :
  ∃ (n : ℤ), ∃ (k : ℤ), 1981 * (n + 990) = k^3 :=
by
  sorry

end exists_consecutive_integers_sum_cube_l254_254578


namespace depth_of_tunnel_l254_254012

theorem depth_of_tunnel (a b area : ℝ) (h := (2 * area) / (a + b)) (ht : a = 15) (hb : b = 5) (ha : area = 400) :
  h = 40 :=
by
  sorry

end depth_of_tunnel_l254_254012


namespace num_common_terms_arith_seq_l254_254193

def is_arithmetic_sequence (a d n : ℕ) : ℕ := a + (n - 1) * d

theorem num_common_terms_arith_seq :
  ∀ (a1 d1 l1 a2 d2 l2 : ℕ),
  a1 = 2 → d1 = 3 → l1 = 2021 →
  a2 = 4 → d2 = 5 → l2 = 2019 →
  (let n_terms1 := (l1 - a1) / d1 + 1 in
  let n_terms2 := (l2 - a2) / d2 + 1 in
  let common := min l1 l2 in
  (common - 1) / (d1 * d2 / gcd d1 d2) = 134) :=
by
  intros a1 d1 l1 a2 d2 l2 ha1 hd1 hl1 ha2 hd2 hl2,
  rw [ha1, hd1, hl1, ha2, hd2, hl2],
  let n_terms1 := (2021 - 2) / 3 + 1,
  let n_terms2 := (2019 - 4) / 5 + 1,
  let common := min 2021 2019,
  show (common - 1) / (3 * 5 / gcd 3 5) = 134,
  calc
    (min 2021 2019 - 1) / (3 * 5 / gcd 3 5) = 2018 / 15 : by simp [min_eq_iff]
                                           ... = 134 : by norm_num

end num_common_terms_arith_seq_l254_254193


namespace price_of_light_bulb_and_motor_l254_254324

theorem price_of_light_bulb_and_motor
  (x : ℝ) (motor_price : ℝ)
  (h1 : x + motor_price = 12)
  (h2 : 10 / x = 2 * 45 / (12 - x)) :
  x = 3 ∧ motor_price = 9 :=
sorry

end price_of_light_bulb_and_motor_l254_254324


namespace product_of_coordinates_of_D_l254_254230

theorem product_of_coordinates_of_D (Mx My Cx Cy Dx Dy : ℝ) (M : (Mx, My) = (4, 8)) (C : (Cx, Cy) = (5, 4)) 
  (midpoint : (Mx, My) = ((Cx + Dx) / 2, (Cy + Dy) / 2)) : (Dx * Dy) = 36 := 
by
  sorry

end product_of_coordinates_of_D_l254_254230


namespace nine_segment_closed_broken_line_impossible_l254_254341

theorem nine_segment_closed_broken_line_impossible :
  ¬(∃ (segments : Fin 9 → (ℝ × ℝ) × (ℝ × ℝ)), 
     (∀ i : Fin 9, ∃! j : Fin 9, i ≠ j ∧ segments_intersect (segments i) (segments j)) ∧ 
     closed_broken_line segments) := 
sorry

-- Definitions for segments_intersect and closed_broken_line are required but assumed to be provided in Mathlib

end nine_segment_closed_broken_line_impossible_l254_254341


namespace books_arrangement_l254_254891

-- All conditions provided in Lean as necessary definitions
def num_arrangements (math_books english_books science_books : ℕ) : ℕ :=
  if math_books = 4 ∧ english_books = 6 ∧ science_books = 2 then
    let arrangements_groups := 2 * 3  -- Number of valid group placements
    let arrangements_math := Nat.factorial math_books
    let arrangements_english := Nat.factorial english_books
    let arrangements_science := Nat.factorial science_books
    arrangements_groups * arrangements_math * arrangements_english * arrangements_science
  else
    0

theorem books_arrangement : num_arrangements 4 6 2 = 207360 :=
by
  sorry

end books_arrangement_l254_254891


namespace incorrect_membership_l254_254099

-- Let's define the sets involved.
def a : Set ℕ := {1}             -- singleton set {a}
def ab : Set (Set ℕ) := {{1}, {2}}  -- set {a, b}

-- Now, the proof statement.
theorem incorrect_membership : ¬ (a ∈ ab) := 
by { sorry }

end incorrect_membership_l254_254099


namespace inverse_of_h_is_j_l254_254466

def h (x : ℝ) : ℝ := 6 - 3 * x

def j (x : ℝ) : ℝ := (6 - x) / 3

theorem inverse_of_h_is_j : ∀ x, h (j x) = x ∧ j (h x) = x := by
  sorry

end inverse_of_h_is_j_l254_254466


namespace cyclic_sums_sine_cosine_l254_254964

theorem cyclic_sums_sine_cosine (α β γ : ℝ) (h : α + β + γ = Real.pi) : 
  (Real.sin (2 * α) + Real.sin (2 * β) + Real.sin (2 * γ)) = 
  2 * (Real.sin α + Real.sin β + Real.sin γ) * 
      (Real.cos α + Real.cos β + Real.cos γ) - 
  2 * (Real.sin α + Real.sin β + Real.sin γ) := 
  sorry

end cyclic_sums_sine_cosine_l254_254964


namespace intersection_M_N_l254_254632

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254632


namespace cans_increment_l254_254303

/--
If there are 9 rows of cans in a triangular display, where each successive row increases 
by a certain number of cans \( x \) compared to the row above it, with the seventh row having 
19 cans, and the total number of cans being fewer than 120, then 
each row has 4 more cans than the row above it.
-/
theorem cans_increment (x : ℕ) : 
  9 * 19 - 16 * x < 120 → x > 51 / 16 → x = 4 :=
by
  intros h1 h2
  sorry

end cans_increment_l254_254303


namespace crayons_birthday_l254_254996

theorem crayons_birthday (total now at_end: Real) (h₁: now = 613) (h₂: at_end = 134.0) : total = now - at_end → total = 479 := 
by
  intros
  rw [h₁, h₂]
  norm_num
  assumption

end crayons_birthday_l254_254996


namespace taxi_fare_for_100_miles_l254_254539

theorem taxi_fare_for_100_miles
  (base_fare : ℝ := 10)
  (proportional_fare : ℝ := 140 / 80)
  (fare_for_80_miles : ℝ := 150)
  (distance_80 : ℝ := 80)
  (distance_100 : ℝ := 100) :
  let additional_fare := proportional_fare * distance_100
  let total_fare_for_100_miles := base_fare + additional_fare
  total_fare_for_100_miles = 185 :=
by
  sorry

end taxi_fare_for_100_miles_l254_254539


namespace intersection_M_N_l254_254692

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254692


namespace number_of_divisors_30_l254_254883

theorem number_of_divisors_30 : 
  ∃ (d : ℕ), d = 2 * 2 * 2 ∧ d = 8 :=
  by sorry

end number_of_divisors_30_l254_254883


namespace smallest_value_range_l254_254045

theorem smallest_value_range {a : Fin 8 → ℝ}
  (h_sum : (∑ i, a i) = 4/3)
  (h_pos_7 : ∀ i : Fin 8, (∑ j in Finset.erase Finset.univ i, a j) > 0) :
  -8 < a 0 ∧ a 0 ≤ 1/6 :=
sorry

end smallest_value_range_l254_254045


namespace solution_set_of_inequality_l254_254028

theorem solution_set_of_inequality (x : ℝ) : (0 < x ∧ x < 1/3) ↔ (1/x > 3) := 
sorry

end solution_set_of_inequality_l254_254028


namespace intersection_M_N_l254_254807

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254807


namespace number_of_prime_divisors_420_l254_254888

theorem number_of_prime_divisors_420 : 
  ∃ (count : ℕ), (∀ (p : ℕ), prime p → p ∣ 420 → p ∈ {2, 3, 5, 7}) ∧ count = 4 := 
by
  sorry

end number_of_prime_divisors_420_l254_254888


namespace intersection_of_M_and_N_l254_254795

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254795


namespace max_distinct_numbers_l254_254471

theorem max_distinct_numbers (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 1000) :
  ∃ (s : set ℕ), (∀ x y ∈ s, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6)) ∧ s.card = 400 :=
by
  sorry

end max_distinct_numbers_l254_254471


namespace johnson_vincent_work_together_l254_254348

theorem johnson_vincent_work_together (work : Type) (time_johnson : ℕ) (time_vincent : ℕ) (combined_time : ℕ) :
  time_johnson = 10 → time_vincent = 40 → combined_time = 8 → 
  (1 / time_johnson + 1 / time_vincent) = 1 / combined_time :=
by
  intros h_johnson h_vincent h_combined
  sorry

end johnson_vincent_work_together_l254_254348


namespace sum_of_acute_angles_l254_254249

theorem sum_of_acute_angles (α β : Real) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
    (h1 : Real.sin α = 2 * Real.sqrt 5 / 5)
    (h2 : Real.sin β = 3 * Real.sqrt 10 / 10) :
    α + β = 3 * Real.pi / 4 :=
sorry

end sum_of_acute_angles_l254_254249


namespace problem1_problem2_l254_254116

-- Problem 1
theorem problem1 : 3 * (Real.sqrt 3 + Real.sqrt 2) - 2 * (Real.sqrt 3 - Real.sqrt 2) = Real.sqrt 3 + 5 * Real.sqrt 2 :=
by
  sorry

-- Problem 2
theorem problem2 : abs (Real.sqrt 3 - Real.sqrt 2) + abs (Real.sqrt 3 - 2) + Real.sqrt 4 = 4 - Real.sqrt 2 :=
by
  sorry

end problem1_problem2_l254_254116


namespace student_D_not_top_student_l254_254900

-- Students rank data and definitions
structure StudentData where
  average : ℝ
  median : ℝ
  variance : Option ℝ := none
  mode : Option ℝ := none

-- Data for each student
def StudentA : StudentData := { average := 2, median := 2 }
def StudentB : StudentData := { average := 2, median := 2, variance := some 0.9 }
def StudentC : StudentData := { median := 2, mode := some 2 }
def StudentD : StudentData := { variance := some 1.1, mode := some 2 }

-- Condition for being a top student
def isTopStudent (data : StudentData) : Prop :=
  data.average ≤ 3 ∧ data.median ≤ 3 ∧ (data.variance.isNone ∨ data.variance ≤ some 1) ∧ (data.mode.isNone ∨ data.mode ≤ some 3)

-- Theorem to prove Student D cannot be a top student
theorem student_D_not_top_student : ¬ isTopStudent StudentD := by
  sorry

end student_D_not_top_student_l254_254900


namespace votes_difference_l254_254321

theorem votes_difference (V : ℝ) (h1 : 0.62 * V = 899) :
  |(0.62 * V) - (0.38 * V)| = 348 :=
by
  -- The solution goes here
  sorry

end votes_difference_l254_254321


namespace profit_percentage_previous_year_l254_254914

-- Declaring variables
variables (R P : ℝ) -- revenues and profits in the previous year
variable (revenues_1999 := 0.8 * R) -- revenues in 1999
variable (profits_1999 := 0.14 * revenues_1999) -- profits in 1999

-- Given condition: profits in 1999 were 112.00000000000001 percent of the profits in the previous year
axiom profits_ratio : 0.112 * R = 1.1200000000000001 * P

-- Prove the profit as a percentage of revenues in the previous year was 10%
theorem profit_percentage_previous_year : (P / R) * 100 = 10 := by
  sorry

end profit_percentage_previous_year_l254_254914


namespace problem_part_1_problem_part_2_l254_254378

universe u

noncomputable theory

variables {α : Type u} [linear_ordered_field α] 
  (a : ℕ → α) (S : ℕ → α)

-- Conditions
def condition_1 (n : ℕ) : Prop := 4 * S n = (a n)^2 + 2 * a n - 3
def condition_2 : Prop := ∀ n, n ∈ {1, 2, 3, 4, 5} → a n > 0
def condition_3 : Prop := ∀ n, n ∈ {1, 2, 3, 4, 5} → a n = λ k, a 1 * (2:α)^(k-1).nat_abs

-- Problem Statement
theorem problem_part_1 (n : ℕ) (hn : n ≥ 5) (h1 : condition_1 a S n) 
  (h3: ∀ m, m ≥ 5 → a m > 0) : (a (n + 1) - a n = a (n + 2) - a (n + 1)) :=
sorry

theorem problem_part_2 (n : ℕ) (h1 : ∀ m, condition_1 a S m) 
  (h3: ∀ m, m ≥ 5 → a m > 0) : S n = if n < 5 then (3/2) * (1 - (-1)^n) else n^2 - 6*n + 8 :=
sorry

end problem_part_1_problem_part_2_l254_254378


namespace find_valid_A_l254_254165

-- Define condition that B is coprime with 36
def coprime_with_36 (b : ℕ) : Prop :=
  Nat.coprime b 36

-- Define condition that moving the last digit to the first place forms A
def derived_A (b : ℕ) : ℕ :=
  let last_digit := b % 10
  let remaining := b / 10
  10^7 * last_digit + remaining

-- Define the range of B based on the problem statement
def valid_range (b : ℕ) : Prop :=
  b > 7777777 ∧ b < 10^8

-- Define the smallest and largest valid A
def smallest_valid_A (a : ℕ) : Prop :=
  a = 17777779

def largest_valid_A (a : ℕ) : Prop :=
  a = 99999998

-- Proof goal statement
theorem find_valid_A (b : ℕ) (h1 : coprime_with_36 b) (h2 : valid_range b) :
  smallest_valid_A (derived_A b) ∨ largest_valid_A (derived_A b) :=
sorry

end find_valid_A_l254_254165


namespace intersection_eq_l254_254702

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254702


namespace value_99_cubed_expansion_l254_254485

theorem value_99_cubed_expansion : 99^3 + 3 * 99^2 + 3 * 99 + 1 = 1_000_000 := 
by
  sorry

end value_99_cubed_expansion_l254_254485


namespace correct_operation_l254_254100

theorem correct_operation :
  (\(\sqrt{2} \times \sqrt{5} = \sqrt{10}\)) \land
  ¬(\(\sqrt{5} - \sqrt{3} = \sqrt{2}\)) \land
  ¬(5\sqrt{3} - \sqrt{3} = 5) \land
  ¬(\(\sqrt{(-3)^2} = -3\)) :=
by {
  sorry
}

end correct_operation_l254_254100


namespace max_distinct_numbers_l254_254477

theorem max_distinct_numbers : 
  ∃ (S : Finset ℕ), S.card = 400 ∧ S ⊆ (Finset.range 1000).filter (λ x, 1 ≤ x ∧ x ≤ 1000) ∧ 
    ∀ x y ∈ S, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6) :=
by
  sorry

end max_distinct_numbers_l254_254477


namespace arrange_tracks_l254_254452

theorem arrange_tracks (n : ℕ) (m : ℕ) : n = 8 → m = 3 → 
  let total_arrangements := 6 * (m.factorial) * ((n - m).factorial)
  in total_arrangements = 4320 
| 8, 3, rfl, rfl := by
  -- Definitions and calculations should be replaced with the appropriate Lean code if providing proof
  sorry

end arrange_tracks_l254_254452


namespace lattice_points_on_hyperbola_l254_254275

theorem lattice_points_on_hyperbola : 
  ∃ n, (∀ x y : ℤ, x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | 
  ∃ a b : ℤ, x = 2 * a + b ∧ y = 2 * a - b}) ∧ n = 250 := 
by {
  sorry
}

end lattice_points_on_hyperbola_l254_254275


namespace find_a_l254_254290

theorem find_a (f : ℝ → ℝ)
  (h : ∀ x : ℝ, x < 2 → a - 3 * x > 0) :
  a = 6 :=
by sorry

end find_a_l254_254290


namespace problem_a_problem_b_problem_c_l254_254844

-- Problem (a)
theorem problem_a (x : ℝ) (k : ℝ) : (∃! (x0 : ℝ), g x0 = 0) → k ∈ (-∞, (3 / 2) + Real.log 2) ∨ k ∈ (3 - 2 * Real.log 2, ∞) :=
by
  let f := λ x : ℝ, Real.log x + 9 / (2 * (x + 1))
  let g := λ x : ℝ, f x - k
  sorry

-- Problem (b)
theorem problem_b (x : ℝ) : 
  x > 1 → (ln x + 2 / (x + 1) > 1) ∧
  (0 < x ∧ x < 1 → ln x + 2 / (x + 1) < 1) ∧
  (x = 1 → ln x + 2 / (x + 1) = 1) :=
by
  let f := λ x : ℝ, ln x + 2 / (x + 1)
  sorry

-- Problem (c)
theorem problem_c (n : ℕ) : 
  0 < n → ln (n + 1) > (Finset.sum (Finset.range n) (λ i, 1 / (2 * i + 3))) :=
by
  let s := Finset.range n
  let sum := Finset.sum s (λ x, 1 / (2 * x + 3))
  sorry

end problem_a_problem_b_problem_c_l254_254844


namespace counterexample_exists_l254_254569

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end counterexample_exists_l254_254569


namespace thirty_five_power_identity_l254_254001

theorem thirty_five_power_identity (m n : ℕ) : 
  let P := 5^m 
  let Q := 7^n 
  35^(m*n) = P^n * Q^m :=
by 
  sorry

end thirty_five_power_identity_l254_254001


namespace prob_KH_then_Ace_l254_254460

noncomputable def probability_KH_then_Ace_drawn_in_sequence : ℚ :=
  let prob_first_card_is_KH := 1 / 52
  let prob_second_card_is_Ace := 4 / 51
  prob_first_card_is_KH * prob_second_card_is_Ace

theorem prob_KH_then_Ace : probability_KH_then_Ace_drawn_in_sequence = 1 / 663 := by
  sorry

end prob_KH_then_Ace_l254_254460


namespace M_inter_N_eq_2_4_l254_254827

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254827


namespace time_to_pass_platform_l254_254104

-- Definitions
def train_length : ℕ := 1400
def platform_length : ℕ := 700
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

-- Prove that the time to pass the platform is 150 seconds
theorem time_to_pass_platform : total_distance / train_speed = 150 :=
by
  sorry

end time_to_pass_platform_l254_254104


namespace probability_of_best_performance_winning_best_performance_wins_more_than_two_l254_254075

def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def choose (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))

theorem probability_of_best_performance_winning (n : ℕ) :
  1 - (choose (2 * n) n) / (factorial (2 * n)) =
  1 - ((factorial n) * (factorial n)) / (factorial (2 * n)) := 
by sorry

theorem best_performance_wins_more_than_two (n s : ℕ) (h : s > 2) : 
  1 = 1 := 
by sorry

end probability_of_best_performance_winning_best_performance_wins_more_than_two_l254_254075


namespace range_f_plus_g_not_all_rationals_l254_254944

variables {f g : ℚ → ℚ}

-- Define strictly monotonically increasing for functions from ℚ to ℚ
def strictly_monotonic (h : ℚ → ℚ) : Prop :=
  ∀ x y : ℚ, x < y → h x < h y

-- Define range being ℚ for functions from ℚ to ℚ
def range_is_all_rationals (h : ℚ → ℚ) : Prop :=
  ∀ q : ℚ, ∃ x : ℚ, h x = q

-- Given conditions in the problem
variable (h_f_strict : strictly_monotonic f)
variable (h_g_strict : strictly_monotonic g)
variable (h_f_range : range_is_all_rationals f)
variable (h_g_range : range_is_all_rationals g)

-- Proof Problem: Show that the range of f + g is not necessarily ℚ
theorem range_f_plus_g_not_all_rationals :
  ¬ range_is_all_rationals (λ x, f x + g x) :=
sorry

end range_f_plus_g_not_all_rationals_l254_254944


namespace intersection_eq_l254_254697

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254697


namespace sum_of_factorization_constants_l254_254014

theorem sum_of_factorization_constants (p q r s t : ℤ) (y : ℤ) :
  (512 * y ^ 3 + 27 = (p * y + q) * (r * y ^ 2 + s * y + t)) →
  p + q + r + s + t = 60 :=
by
  intro h
  sorry

end sum_of_factorization_constants_l254_254014


namespace calculate_difference_of_squares_l254_254561

theorem calculate_difference_of_squares : (640^2 - 360^2) = 280000 := by
  sorry

end calculate_difference_of_squares_l254_254561


namespace sqrt_means_x_ge2_l254_254024

theorem sqrt_means_x_ge2 (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) ↔ (x ≥ 2) :=
sorry

end sqrt_means_x_ge2_l254_254024


namespace intersection_M_N_l254_254761

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254761


namespace intersection_M_N_l254_254809

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254809


namespace correct_propositions_l254_254428

def proposition_1 := false
def proposition_2 := true
def proposition_3 := true
def proposition_4 := false

theorem correct_propositions : ({proposition_1, proposition_2, proposition_3, proposition_4}.filter id) = [true, true] := by
  -- Proof steps go here, but we'll exclude them for now
  sorry

end correct_propositions_l254_254428


namespace problem_statement_l254_254338

open Real
open EuclideanGeometry

noncomputable def isosceles_triangle_AB_AC : Prop :=
  ∃ (A B C D : Point) (AB AC BC BD DA : ℝ),
    Triangle A B C ∧
    Segment AB = Segment AC ∧
    ∠ BAC = 100 ∧
    Bisection D A C B ∧
    BC = BD + DA

theorem problem_statement (A B C D : Point) (AB AC BC BD DA : ℝ) :
  isosceles_triangle_AB_AC A B C D AB AC BC BD DA →
  BC = BD + DA :=
by
  intro h
  sorry

end problem_statement_l254_254338


namespace largest_prime_factor_5f6f_l254_254087

open Nat

theorem largest_prime_factor_5f6f : 
  ∃ p, Prime p ∧ p = 7 ∧ ∀ q, Prime q ∧ q ∣ (fact 5 + fact 6) → q ≤ p := by
  sorry

end largest_prime_factor_5f6f_l254_254087


namespace intersection_M_N_l254_254810

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254810


namespace hamburger_combinations_l254_254866

theorem hamburger_combinations :
  let condiments := 9
  let buns := 2
  let patties := 3
  -- the number of combinations of condiments is 2^condiments
  let condiment_combinations := 2 ^ condiments
  -- total hamburgers is the product of condiment combinations, buns, and patties
  let total_hamburgers := condiment_combinations * buns * patties
  total_hamburgers = 3072 :=
by
  let condiments := 9
  let buns := 2
  let patties := 3
  let condiment_combinations := 2 ^ condiments
  let total_hamburgers := condiment_combinations * buns * patties
  show total_hamburgers = 3072 from sorry

end hamburger_combinations_l254_254866


namespace sqrt_meaningful_real_range_l254_254906

theorem sqrt_meaningful_real_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_meaningful_real_range_l254_254906


namespace timeToFillDrum_l254_254342

-- Definitions for the conditions given in the problem
def rainRate (t : ℝ) : ℝ := 5 * t^2
def internalArea : ℝ := 300
def effectiveDepth : ℝ := 15
def volumeToFill : ℝ := internalArea * effectiveDepth

-- Statement of the theorem to prove
theorem timeToFillDrum : ∃T : ℝ, ∫ (t : ℝ) in 0..T, rainRate t = volumeToFill ∧ |T - 13.98| < 0.01 :=
by
  sorry

end timeToFillDrum_l254_254342


namespace nested_radical_eq_three_l254_254926

theorem nested_radical_eq_three (m : ℝ) (h : 0 < m) : 
  m = sqrt (3 + 2 * sqrt (3 + 2 * sqrt (3 + 2 * sqrt (3 + 2 * sqrt (3 + 2 * ...)))))  :=
by
  have eq : m^2 = 3 + 2 * m := sorry
  have roots := polynomial.roots (polynomial.quadratic (by linarith))
  exact or.elim (roots_property (by linarith)) 
    (λ (h_pos : m = 3), h_pos (by refl)) 
    (λ (h_neg : m = -1), false.elim (by linarith))

end nested_radical_eq_three_l254_254926


namespace intersection_M_N_l254_254775

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254775


namespace log_sum_of_real_coefficients_l254_254981

theorem log_sum_of_real_coefficients :
  let S := ∑ k in finset.range 2009.succ, 
    if even k then (binom 2009 k : ℝ) else 0
  in
  log S / log 2 = 1004 := by sorry

end log_sum_of_real_coefficients_l254_254981


namespace find_k_values_l254_254270

noncomputable def parallel_vectors (k : ℝ) : Prop :=
  (k^2 / k = (k + 1) / 4)

theorem find_k_values (k : ℝ) : parallel_vectors k ↔ (k = 0 ∨ k = 1 / 3) :=
by sorry

end find_k_values_l254_254270


namespace sum_of_reciprocals_l254_254985

variables (n : ℕ) (h_odd : n % 2 = 1) (h_pos : n > 0)
noncomputable def z : ℂ := complex.exp (2 * real.pi * complex.i / n)

theorem sum_of_reciprocals (n > 0) (n % 2 = 1) :
  (∑ i in finset.range n, 1 / (1 + (z n) ^ (i + 1))) = n / 2 :=
sorry

end sum_of_reciprocals_l254_254985


namespace plates_usage_when_parents_join_l254_254382

theorem plates_usage_when_parents_join
  (total_plates : ℕ)
  (plates_per_day_matt_and_son : ℕ)
  (days_matt_and_son : ℕ)
  (days_with_parents : ℕ)
  (total_days_in_week : ℕ)
  (total_plates_needed : total_plates = 38)
  (plates_used_matt_and_son : plates_per_day_matt_and_son = 2)
  (days_matt_and_son_eq : days_matt_and_son = 3)
  (days_with_parents_eq : days_with_parents = 4)
  (total_days_in_week_eq : total_days_in_week = 7)
  (plates_used_when_parents_join : total_plates - plates_per_day_matt_and_son * days_matt_and_son = days_with_parents * 8) :
  true :=
sorry

end plates_usage_when_parents_join_l254_254382


namespace M_inter_N_eq_2_4_l254_254821

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254821


namespace distribute_stops_l254_254300

/--
In a certain city, the network of bus routes is arranged such that any two routes have exactly one common stop, and each route has at least 4 stops. 

Prove that all stops can be distributed between two companies in such a way that each route will have stops belonging to both companies.
-/
theorem distribute_stops (city : Type) (routes : set (set city)) :
  (∀ r1 r2 ∈ routes, r1 ≠ r2 → ∃! s ∈ r1, s ∈ r2) →
  (∀ r ∈ routes, 4 ≤ r.size) →
  ∃ company1 company2 : set city,
    (∀ r ∈ routes, ∃ s1 ∈ company1, ∃ s2 ∈ company2, s1 ∈ r ∧ s2 ∈ r) :=
by
  sorry

end distribute_stops_l254_254300


namespace distinct_license_plates_l254_254136

theorem distinct_license_plates :
  let num_digits := 10
  let num_letters := 26
  let num_digit_positions := 5
  let num_letter_pairs := num_letters * num_letters
  let num_letter_positions := num_digit_positions + 1
  num_digits^num_digit_positions * num_letter_pairs * num_letter_positions = 40560000 := by
  sorry

end distinct_license_plates_l254_254136


namespace midpoints_on_circle_l254_254398

-- Define that we have a triangle with vertices A, B, C.
variables {A B C : Type} [EuclideanGeometry A B C]

-- Define the centers of excircles I_A, I_B, I_C opposite to vertices A, B, C respectively.
noncomputable def I_A := excircle_center A
noncomputable def I_B := excircle_center B
noncomputable def I_C := excircle_center C

-- Define the midpoints of the sides of the triangle I_A I_B I_C.
noncomputable def midpoint_I_B_I_C := midpoint I_B I_C 
noncomputable def midpoint_I_C_I_A := midpoint I_C I_A
noncomputable def midpoint_I_A_I_B := midpoint I_A I_B

-- The main theorem stating that the midpoints of the sides of the triangle formed by the centers of the excircles lie on the same circle.
theorem midpoints_on_circle :
  ∃ (O : Type) [Circumcircle O (midpoint_I_B_I_C, midpoint_I_C_I_A, midpoint_I_A_I_B)], true := 
sorry

end midpoints_on_circle_l254_254398


namespace youngest_brother_age_l254_254408

theorem youngest_brother_age 
  (x : ℤ) 
  (h1 : ∃ (a b c : ℤ), a = x ∧ b = x + 1 ∧ c = x + 2 ∧ a + b + c = 96) : 
  x = 31 :=
by sorry

end youngest_brother_age_l254_254408


namespace log_xy_l254_254280

theorem log_xy (x y : ℝ) (h1 : log (x ^ 2 * y ^ 3) = 2) (h2 : log (x ^ 3 * y ^ 2) = 2) : log (x * y) = 4 / 5 :=
by sorry

end log_xy_l254_254280


namespace independent_variable_range_l254_254933

/-- In the function y = 1 / (x - 2), the range of the independent variable x is all real numbers except 2. -/
theorem independent_variable_range (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 2)) ↔ x ≠ 2 :=
by
  sorry

end independent_variable_range_l254_254933


namespace inequality_iff_l254_254233

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end inequality_iff_l254_254233


namespace intersection_M_N_l254_254653

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254653


namespace intersection_M_N_l254_254625

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254625


namespace intersection_M_N_l254_254768

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254768


namespace limit_S_l254_254222

noncomputable def a : ℕ → ℝ
| 0       := 1
| (n + 1) := -1 / 2 * a n

def S (n : ℕ) : ℝ := ∑ i in Finset.range (n + 1), a i

open Filter Real

theorem limit_S : tendsto S at_top (𝓝 (2 / 3)) :=
sorry

end limit_S_l254_254222


namespace train_speed_is_72_kmph_l254_254154

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 112
noncomputable def crossing_time : ℝ := 11.099112071034318

theorem train_speed_is_72_kmph :
  let total_distance := train_length + bridge_length
  let speed_m_per_s := total_distance / crossing_time
  let speed_kmph := speed_m_per_s * 3.6
  speed_kmph = 72 :=
by
  sorry

end train_speed_is_72_kmph_l254_254154


namespace chemical_X_percentage_l254_254108

-- Let X be the percentage of chemical X in the resulting mixture
-- Let initial_volume be the initial volume of the mixture without added chemical X
-- Let initial_percentage_X be the initial percentage of chemical X in the original mixture
-- Let added_X be the volume of added chemical X
-- Let resulting_volume be the volume of the resulting mixture

def percentage_of_chemical_X (initial_volume : ℝ) (initial_percentage_X : ℝ) (added_X : ℝ) : ℝ :=
  let initial_amount_X := initial_percentage_X * initial_volume
  let total_amount_X := initial_amount_X + added_X
  let resulting_volume := initial_volume + added_X
  (total_amount_X / resulting_volume) * 100

theorem chemical_X_percentage :
  percentage_of_chemical_X 80 0.30 20 = 44 := by
    sorry

end chemical_X_percentage_l254_254108


namespace point_in_third_quadrant_l254_254329

theorem point_in_third_quadrant (x y : ℝ) (h1 : x = -3) (h2 : y = -2) : 
  x < 0 ∧ y < 0 :=
by
  sorry

end point_in_third_quadrant_l254_254329


namespace intersection_of_M_and_N_l254_254740

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254740


namespace quotient_remainder_l254_254595

theorem quotient_remainder (x y : ℕ) (hx : 0 ≤ x) (hy : 0 < y) : 
  ∃ q r : ℕ, q ≥ 0 ∧ 0 ≤ r ∧ r < y ∧ x = q * y + r := by
  sorry

end quotient_remainder_l254_254595


namespace equivalent_statements_l254_254492

variable (P Q : Prop)

theorem equivalent_statements : 
  (P → Q) ↔ ((¬ Q → ¬ P) ∧ (¬ P ∨ Q)) :=
by 
  sorry

end equivalent_statements_l254_254492


namespace restore_vertex_l254_254994

theorem restore_vertex (a b c d e f x y z w u v : ℕ)
    (hx : x = a + f) (hy : y = a + b) (hz : z = b + c)
    (hw : w = c + d) (hu : u = d + e) (hv : v = e + f) :
    ∃ (x' : ℕ), x' = (b + d + f) - (y + z + v) :=
by {
    use (b + d + f) - (y + z + v),
    sorry
}

end restore_vertex_l254_254994


namespace hundredth_digit_of_concatenated_squares_is_nine_l254_254029

/-- The 100th digit in the concatenated sequence of the squares of natural numbers from 1 to 99 is 9. -/
theorem hundredth_digit_of_concatenated_squares_is_nine :
  let digit_at_position (n : Nat) : Nat :=
    let concatenated_squares := 
      String.join (List.map (fun x => toString (x * x)) (List.range' 1 99))
    if h : n < concatenated_squares.length then 
      concatenated_squares.get ⟨n, h⟩.toNat - '0'.toNat
    else
      0
  in digit_at_position 99 = 9 :=
by
  sorry

end hundredth_digit_of_concatenated_squares_is_nine_l254_254029


namespace change_proof_l254_254957

-- Definitions of the given conditions
def lee_money : ℕ := 10
def friend_money : ℕ := 8
def chicken_wings_cost : ℕ := 6
def chicken_salad_cost : ℕ := 4
def soda_cost : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Statement of the theorem
theorem change_proof : 
  let total_money : ℕ := lee_money + friend_money,
      meal_cost_before_tax : ℕ := chicken_wings_cost + chicken_salad_cost + num_sodas * soda_cost,
      total_meal_cost : ℕ := meal_cost_before_tax + tax
  in total_money - total_meal_cost = 3 := 
by
  -- We skip the proof, as it's not required per instructions
  sorry

end change_proof_l254_254957


namespace angle_ABN_possible_values_l254_254998

-- Definitions of the structures and given conditions
variable {Point : Type}

structure Trapezoid (A B C D : Point) : Prop :=
(base_ratio : dist A B = 2 * dist C D)
(congruent_sides : dist A D = dist D C ∧ dist D C = dist C B ∧ dist C B = dist A D)

structure Square (D C N : Point) : Prop :=
(shared_side : dist D C = dist C N)
(is_square  : True) -- Simply assuming the existence of the square for simplicity

variable {A B C D N : Point}

-- Given the conditions, the main theorem to prove
theorem angle_ABN_possible_values (h_trapezoid : Trapezoid A B C D) (h_square : Square D C N) :
  ∃ θ : ℝ, θ = 15 ∨ θ = 75 := sorry

end angle_ABN_possible_values_l254_254998


namespace y_increase_by_18_when_x_increases_by_12_l254_254298

theorem y_increase_by_18_when_x_increases_by_12
  (h_slope : ∀ x y: ℝ, (4 * y = 6 * x) ↔ (3 * y = 2 * x)) :
  ∀ Δx : ℝ, Δx = 12 → ∃ Δy : ℝ, Δy = 18 :=
by
  sorry

end y_increase_by_18_when_x_increases_by_12_l254_254298


namespace length_of_train_l254_254540

-- Declare the natural numbers
variables (L : ℝ)

-- Declare the conditions
def speed_pole := L / 30
def speed_platform := (L + 200) / 45

-- The theorem statement
theorem length_of_train (h : speed_pole L = speed_platform L) : L = 400 := by sorry

end length_of_train_l254_254540


namespace ratio_of_areas_l254_254072

theorem ratio_of_areas 
  (a b c : ℕ) (d e f : ℕ)
  (hABC : a = 6 ∧ b = 8 ∧ c = 10 ∧ a^2 + b^2 = c^2)
  (hDEF : d = 8 ∧ e = 15 ∧ f = 17 ∧ d^2 + e^2 = f^2) :
  (1/2 * a * b) / (1/2 * d * e) = 2 / 5 :=
by
  sorry

end ratio_of_areas_l254_254072


namespace cone_surface_area_ratio_l254_254524

variables (radius slant_height : ℝ)

-- Condition according to the problem
def cone_relationship (r l : ℝ) := 2 * π * r = (1 / 2) * π * l

noncomputable def base_area (r : ℝ) := π * r ^ 2

noncomputable def side_surface_area (l : ℝ) := (1 / 2) * π * l ^ 2

noncomputable def total_surface_area (r l : ℝ) := base_area r + side_surface_area l

noncomputable def ratio_total_to_side (r l : ℝ) := (total_surface_area r l) / (side_surface_area l)

theorem cone_surface_area_ratio : 
  ∀ (r l : ℝ), cone_relationship r l → ratio_total_to_side r l = 5 / 4 :=
by
  intros r l h
  unfold cone_relationship at h
  unfold base_area side_surface_area total_surface_area ratio_total_to_side
  sorry

end cone_surface_area_ratio_l254_254524


namespace parallel_lines_slope_eq_l254_254433

theorem parallel_lines_slope_eq (m : ℝ) : 
  (∀ x y : ℝ, 3 * x + 2 * y - 3 = 0 → 6 * x + m * y + 1 = 0) → m = 4 :=
by
  sorry

end parallel_lines_slope_eq_l254_254433


namespace min_c_value_l254_254999

noncomputable def has_unique_solution (a b c : ℕ) : Prop :=
  ∃ x y : ℤ, 2 * x + y = 2022 ∧ y = abs (x - a) + abs (x - b) + abs (x - c)

noncomputable def min_c_for_unique_solution (a b : ℕ) : ℕ :=
  if h : ∃ (c : ℕ), ∀ c' > c, ¬ has_unique_solution a b c' then
    nat.find h
  else
    0  -- This is a placeholder and should never be hit since there is always a minimum c

theorem min_c_value (a b : ℕ) (h : a < b) : 
  min_c_for_unique_solution a b = 1012 := 
by
  sorry

end min_c_value_l254_254999


namespace difference_of_two_smallest_integers_divisors_l254_254455

theorem difference_of_two_smallest_integers_divisors (n m : ℕ) (h₁ : n > 1) (h₂ : m > 1) 
(h₃ : n % 2 = 1) (h₄ : n % 3 = 1) (h₅ : n % 4 = 1) (h₆ : n % 5 = 1) 
(h₇ : n % 6 = 1) (h₈ : n % 7 = 1) (h₉ : n % 8 = 1) (h₁₀ : n % 9 = 1) 
(h₁₁ : n % 10 = 1) (h₃' : m % 2 = 1) (h₄' : m % 3 = 1) (h₅' : m % 4 = 1) 
(h₆' : m % 5 = 1) (h₇' : m % 6 = 1) (h₈' : m % 7 = 1) (h₉' : m % 8 = 1) 
(h₁₀' : m % 9 = 1) (h₁₁' : m % 10 = 1): m - n = 2520 :=
sorry

end difference_of_two_smallest_integers_divisors_l254_254455


namespace smallest_a1_range_l254_254055

noncomputable def smallest_a1_possible (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = (4 : ℝ) / (3 : ℝ) ∧ 
  (∀ i : Fin₈, (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - a_i) > 0) ∧ 
  (-8 < a_1 ∧ a_1 ≤ 1 / 6)

-- Since we are only required to state the problem, we leave the proof as a "sorry".
theorem smallest_a1_range (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
    (h_sum: a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 4 / 3)
    (h_pos: ∀ i : Fin 8, a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - ([-][i]) > 0):
    smallest_a1_possible a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 :=
  by
  sorry

end smallest_a1_range_l254_254055


namespace factory_product_fractions_l254_254601

theorem factory_product_fractions (a_j a_f b_j b_f : ℕ)
(hj_irreducible : nat.coprime a_j b_j)
(hf_irreducible : nat.coprime a_f b_f)
(hj_transform : (a_j + 2) * b_j = a_j * 2 * b_j)
(hf_transform : (a_f + 2) * b_f = a_f * 2 * b_f)
(hj_gt_one_third : 3 * a_j > b_j)
(hf_gt_one_third : 3 * a_f > b_f)
(hj_higher_than_feb : a_j * b_f > a_f * b_j) :
  a_j = 2 ∧ b_j = 3 ∧ a_f = 2 ∧ b_f = 5 :=
by
  sorry

end factory_product_fractions_l254_254601


namespace maximum_numbers_l254_254481

theorem maximum_numbers (S : Finset ℕ) (h1 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 1000)
                       (h2 : ∀ x y ∈ S, x ≠ y → (x - y).natAbs ≠ 4 ∧ (x - y).natAbs ≠ 5 ∧ (x - y).natAbs ≠ 6) :
  S.card ≤ 400 :=
begin
  sorry
end

end maximum_numbers_l254_254481


namespace intersection_M_N_l254_254770

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254770


namespace r_plus_s_l254_254973

def parabola_equation (x : ℝ) : ℝ := x^2 + 4*x + 4

def line_equation_through_Q_with_slope (m x : ℝ) : ℝ := m * (x - 10) + 16

def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def find_r_and_s : (ℝ × ℝ) :=
let a := 1
let b := 48
let c := 64
let discriminant := quadratic_discriminant a b c
if discriminant < 0
then let r := (-48 - 32 * Real.sqrt 2) / 2
     let s := (-48 + 32 * Real.sqrt 2) / 2
     (r, s)
else (0, 0)

theorem r_plus_s : 
  let (r, s) := find_r_and_s in
  r + s = -48 :=
by
  let (r, s) := find_r_and_s
  sorry

end r_plus_s_l254_254973


namespace perry_more_games_than_phil_l254_254997

theorem perry_more_games_than_phil (dana_wins charlie_wins perry_wins : ℕ) :
  perry_wins = dana_wins + 5 →
  charlie_wins = dana_wins - 2 →
  charlie_wins + 3 = 12 →
  perry_wins - 12 = 4 :=
by
  sorry

end perry_more_games_than_phil_l254_254997


namespace evaluate_x_l254_254583

theorem evaluate_x : ∀ x : ℕ, x = 225 + 2 * 15 * 8 + 64 → x = 529 :=
by
  intro x h,
  sorry

end evaluate_x_l254_254583


namespace sum_of_arithmetic_sequence_l254_254231

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℝ) := ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = ∑ i in Finset.range n, a i

def given_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  a 1 = 1 ∧ S 2 = a 3

-- Prove that the sum of the first n terms is as specified
theorem sum_of_arithmetic_sequence (a : ℕ → ℝ) (S : ℕ → ℝ)
  (ha : arithmetic_sequence a)
  (hs : sum_of_sequence a S)
  (hc : given_conditions a S) :
  ∀ n : ℕ, S n = (1 / 2) * n^2 + (1 / 2) * n :=
by
  sorry

end sum_of_arithmetic_sequence_l254_254231


namespace terminal_angle_quadrant_l254_254056

theorem terminal_angle_quadrant : 
  let angle := -558
  let reduced_angle := angle % 360
  90 < reduced_angle ∧ reduced_angle < 180 →
  SecondQuadrant := 
by 
  intro angle reduced_angle h 
  sorry

end terminal_angle_quadrant_l254_254056


namespace non_transit_cities_connected_l254_254306

-- Definitions to represent the problem setup
def kingdom (V : Type) := (V → V → Prop)  -- Represents roads as relations between cities

-- Conditions
variables {V : Type}
variables (vertices : Finset V)
variables (roads : V → V → Prop)
variables (n_transit n_nontransit : Finset V)
variables (H1 : vertices.card = 39)  -- 39 cities
variables (H2 : ∀ v, (roads v).card ≥ 21)  -- Each city has at least 21 one-way roads
variables (H3 : n_transit.card = 26)  -- 26 transit cities
variables (H4 : n_nontransit.card = 13)  -- 13 non-transit cities
variables (H5 : n_transit ∪ n_nontransit = vertices)  -- All cities are either transit or non-transit
variables (H6 : n_transit ∩ n_nontransit = ∅)  -- Disjoint sets of transit and non-transit cities
variables (H7 : ∀ a b ∈ n_transit, ¬roads a b)  -- Transit cities do not have direct return paths

-- Question to prove:
theorem non_transit_cities_connected :
  ∀ a b ∈ n_nontransit, roads a b :=
by
  sorry

end non_transit_cities_connected_l254_254306


namespace sqrt_means_x_ge2_l254_254025

theorem sqrt_means_x_ge2 (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 2)) ↔ (x ≥ 2) :=
sorry

end sqrt_means_x_ge2_l254_254025


namespace intersection_M_N_l254_254620

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254620


namespace intersection_M_N_l254_254759

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254759


namespace cows_milk_days_l254_254284

variable (y : ℕ)

theorem cows_milk_days :
  (y : ℕ) → (y : ℕ) → (y + 4) : ℕ → (y + 6) : ℕ :=
by
  let daily_production_per_cow := (y + 2) / (y * (y + 3))
  let total_daily_production := (y + 4) * daily_production_per_cow
  let number_of_days := (y + 6) / total_daily_production
  have daily_production_per_cow_eq : daily_production_per_cow = (y + 2) / (y * (y + 3)) := rfl
  have total_daily_production_eq : total_daily_production = ((y + 4) * (y + 2)) / (y * (y + 3)) := rfl
  have number_of_days_eq : number_of_days = (y * (y + 3) * (y + 6)) / ((y + 2) * (y + 4)) := rfl
  exact number_of_days_eq
  sorry

end cows_milk_days_l254_254284


namespace change_calculation_l254_254960

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l254_254960


namespace min_sum_of_factors_9_factorial_l254_254212

noncomputable theory

open Real Nat

theorem min_sum_of_factors_9_factorial :
  ∃ (p q r s : ℕ), 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ p * q * r * s = 9! ∧ p + q + r + s = 133 :=
  sorry

end min_sum_of_factors_9_factorial_l254_254212


namespace meeting_time_at_start_l254_254106

theorem meeting_time_at_start (a_speed_kmph : ℝ) (b_speed_kmph : ℝ) (c_speed_kmph : ℝ) (track_length_m : ℝ) :
  a_speed_kmph = 4 → b_speed_kmph = 6 → c_speed_kmph = 8 → track_length_m = 400 →
  let a_speed_mpm := a_speed_kmph * 1000 / 60 in
  let b_speed_mpm := b_speed_kmph * 1000 / 60 in
  let c_speed_mpm := c_speed_kmph * 1000 / 60 in
  let a_time := track_length_m / a_speed_mpm in
  let b_time := track_length_m / b_speed_mpm in
  let c_time := track_length_m / c_speed_mpm in
  let lcm_time := Int.lcm a_time b_time in
  lcm_time = 12 :=
by
  intros h1 h2 h3 h4
  sorry

end meeting_time_at_start_l254_254106


namespace range_of_m_for_B_subset_A_l254_254861

-- Definitions based on the conditions
def A : set ℝ := {x | 4 ≤ x ∧ x ≤ 8}
def B (m : ℝ) : set ℝ := {x | m + 1 < x ∧ x < 2 * m - 2}

-- The statement we want to prove
theorem range_of_m_for_B_subset_A (m : ℝ) : (∀ (x : ℝ), x ∈ B m → x ∈ A) ↔ m ≤ 5 :=
by
sorry

end range_of_m_for_B_subset_A_l254_254861


namespace find_d_in_line_eq_l254_254935

theorem find_d_in_line_eq (d : ℝ) (h : let x := -d / 3 in let y := -d / 5 in x + y = 16) : d = -30 :=
by
  sorry

end find_d_in_line_eq_l254_254935


namespace factors_of_72_that_are_cubes_l254_254887

theorem factors_of_72_that_are_cubes : 
  ∃ n : ℕ, n = 2 ∧ ∀ d : ℕ, d | 72 → (∃ (x y : ℕ), d = 2^x * 3^y ∧ (x = 0 ∨ x % 3 = 0) ∧ y = 0) := 
sorry

end factors_of_72_that_are_cubes_l254_254887


namespace problem_1_problem_2_problem_3_l254_254865

-- Definitions and conditions
def monomial_degree_condition (a : ℝ) : Prop := 2 + (1 + a) = 5

-- Proof goals
theorem problem_1 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = 9 := sorry
theorem problem_2 (a : ℝ) (h : monomial_degree_condition a) : (a + 1) * (a^2 - a + 1) = 9 := sorry
theorem problem_3 (a : ℝ) (h : monomial_degree_condition a) : a^3 + 1 = (a + 1) * (a^2 - a + 1) := sorry

end problem_1_problem_2_problem_3_l254_254865


namespace part_I_1_part_I_2_part_II_l254_254829

open Set
open Real

def A : Set ℝ := { x | x^2 - 4*x + 3 ≤ 0 }
def B : Set ℝ := { x | log 2 x > 1 }

theorem part_I_1 : A ∩ B = { x | 2 < x ∧ x ≤ 3 } := sorry

theorem part_I_2 : (compl B) ∪ A = { x | x ≤ 3 } := sorry

theorem part_II (a : ℝ) (h : { x | 1 < x ∧ x < a } ⊆ A) : a ≤ 3 := sorry

end part_I_1_part_I_2_part_II_l254_254829


namespace intersection_M_N_l254_254772

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254772


namespace cylinder_volume_ratio_l254_254515

theorem cylinder_volume_ratio :
  let h_A := 8
  let C_A := 5
  let r_A := C_A / (2 * Real.pi)
  let V_A := Real.pi * (r_A^2) * h_A
  let h_B := 5
  let C_B := 8
  let r_B := C_B / (2 * Real.pi)
  let V_B := Real.pi * (r_B^2) * h_B
  V_B / V_A = (8 / 5) :=
by
  sorry

end cylinder_volume_ratio_l254_254515


namespace intersection_M_N_l254_254690

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254690


namespace num_perfect_square_factors_of_2880_l254_254437

theorem num_perfect_square_factors_of_2880 :
  let n := 2880
  let prime_factors := [(2, 6), (3, 2), (5, 1)]
  (number of positive integer factors of n that are perfect squares) = 8 :=
by
  sorry

end num_perfect_square_factors_of_2880_l254_254437


namespace integer_solution_xy_eq_yx_l254_254499

theorem integer_solution_xy_eq_yx (x y : ℕ) (hx : 0 < x) (hy : 0 < y) (hxy : x ≠ y) (e : x < y) :
  x ^ y = y ^ x ↔ (x = 2 ∧ y = 4) :=
sorry

end integer_solution_xy_eq_yx_l254_254499


namespace ratio_3_7_impossible_l254_254496

-- Define the conditions
def total_students_range := {n : ℕ | 31 ≤ n ∧ n ≤ 39}

-- Define the ratio and conditions
def ratio_3_7 : ℤ × ℤ := (3, 7)
def total_parts_3_7 : ℤ := ratio_3_7.1 + ratio_3_7.2

-- Prove that no multiple of the sum of parts lies within the range 31 to 39
theorem ratio_3_7_impossible : ¬ ∃ (k : ℕ), total_parts_3_7 * k ∈ total_students_range := 
by 
  sorry

end ratio_3_7_impossible_l254_254496


namespace percentage_difference_in_gain_is_6_12_l254_254551

/-- Define the conditions -/
def selling_price1 := 220
def selling_price2 := 160
def cost_price := 1200

/-- Define the gains, difference in gain, and percentage difference -/
def gain1 := selling_price1 - cost_price
def gain2 := selling_price2 - cost_price
def difference_in_gain := gain1 - gain2
def percentage_difference_in_gain := (difference_in_gain / (abs gain1).toRat) * 100

/-- State the theorem and approximate the percentage difference -/
theorem percentage_difference_in_gain_is_6_12 :
  percentage_difference_in_gain ≈ 6.12 := 
sorry

end percentage_difference_in_gain_is_6_12_l254_254551


namespace change_calculation_l254_254952

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l254_254952


namespace sqrt_meaningful_iff_l254_254909

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 := 
by 
sry.

end sqrt_meaningful_iff_l254_254909


namespace intersection_M_N_l254_254719

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254719


namespace intersection_M_N_l254_254712

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254712


namespace tangent_line_at_2_m_range_for_three_roots_l254_254251

def f (x : ℝ) : ℝ := 2 * x ^ 3 - 3 * x ^ 2 + 3

theorem tangent_line_at_2 :
  ∃ k b, k = 12 ∧ b = -17 ∧ (∀ x, 12 * x - (k * (x - 2) + f 2) = b) :=
by
  sorry

theorem m_range_for_three_roots :
  {m : ℝ | ∃ x₀ x₁ x₂, x₀ < x₁ ∧ x₁ < x₂ ∧ f x₀ + m = 0 ∧ f x₁ + m = 0 ∧ f x₂ + m = 0} = 
  {m : ℝ | -3 < m ∧ m < -2} :=
by
  sorry

end tangent_line_at_2_m_range_for_three_roots_l254_254251


namespace value_of_f_sin_20_l254_254217

theorem value_of_f_sin_20 (f : ℝ → ℝ) (h : ∀ x, f (cos x) = sin (3 * x)) :
  f (sin (real.pi / 9)) = -1 / 2 :=
sorry

end value_of_f_sin_20_l254_254217


namespace chess_tournament_ratio_l254_254915

theorem chess_tournament_ratio (n : ℕ) (juniors seniors total matches : ℕ) 
  (juniors_ratio seniors_ratio total_ratio : ℕ) :
  juniors = n →
  seniors = 3 * n →
  total = juniors + seniors →
  matches = (total * (total - 1)) / 2 →
  juniors_ratio / seniors_ratio = 3 / 7 →
  total_ratio = juniors_ratio + seniors_ratio →
  juniors = 4 :=
by
  sorry

end chess_tournament_ratio_l254_254915


namespace intersection_M_N_l254_254650

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254650


namespace minimum_value_func_l254_254593

noncomputable def func (x : ℝ) : ℝ :=
  2 * Real.sin (π / 3 - x) - Real.cos (π / 6 + x)

theorem minimum_value_func : ∃ x : ℝ, ∀ y : ℝ, y = func x → y >= -1 :=
sorry

end minimum_value_func_l254_254593


namespace sn_value_l254_254850

open Function

def f (x : ℝ) (b : ℝ) : ℝ := x^2 + b * x

def a (n : ℕ) (b : ℝ) : ℝ := 1 / (f n b)

def S (n : ℕ) (b : ℝ) : ℝ := ∑ i in Finset.range (n + 1), a i b

theorem sn_value (b : ℝ) (h : f 1 b = 2) : S n b = n / (n + 1) := by
  -- Proof goes here
  sorry

end sn_value_l254_254850


namespace problem_1_problem_2_l254_254851

def f (x : ℝ) : ℝ := |x - 1|

theorem problem_1 (x : ℝ) : f(x) + f(x + 4) ≥ 8 ↔ x ≤ -5 ∨ x ≥ 3 :=
by
  sorry

theorem problem_2 (a b : ℝ) (h1 : f(a + 1) < 1) (h2 : f(b + 1) < 1) (h3 : a ≠ 0) :
    (f(a * b) / |a|) > f(b / a) :=
by
  sorry

end problem_1_problem_2_l254_254851


namespace intersection_of_M_and_N_l254_254638

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254638


namespace johns_share_is_1100_l254_254506

def total_amount : ℕ := 6600
def ratio_john : ℕ := 2
def ratio_jose : ℕ := 4
def ratio_binoy : ℕ := 6
def total_parts : ℕ := ratio_john + ratio_jose + ratio_binoy
def value_per_part : ℚ := total_amount / total_parts
def amount_received_by_john : ℚ := value_per_part * ratio_john

theorem johns_share_is_1100 : amount_received_by_john = 1100 := by
  sorry

end johns_share_is_1100_l254_254506


namespace smallest_positive_period_of_f_range_of_f_in_range_l254_254863

-- Define the vectors a and b and the function f(x)
def a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)
def f (x : ℝ) : ℝ := (a x).1 * ((a x).1 - (b x).1) + (a x).2 * ((a x).2 - (b x).2)

-- Prove the smallest positive period of f(x)
theorem smallest_positive_period_of_f : ∃ T > 0, ∀ x, f (x + T) = f x :=
by 
  existsi π
  sorry

-- Prove the range of f(x) for x in [-π/4, π/4]
theorem range_of_f_in_range : ∀ x ∈ Set.Icc (-π/4) (π/4), f x ∈ Set.Icc 0 2 :=
by
  intros x hx
  sorry

end smallest_positive_period_of_f_range_of_f_in_range_l254_254863


namespace triangle_and_square_count_l254_254554

variables (a : ℝ) (S : ℝ) 

-- Given conditions
def small_square_area : ℝ := a^2
def shaded_triangle_area : ℝ := S
def S_def : Prop := S = small_square_area / 4

-- Main statement
theorem triangle_and_square_count (a : ℝ) (S : ℝ) (h : S_def a S) :
  ∃ triangle_count square_count, 
  triangle_count = 20 ∧ square_count = 1 :=
by
  sorry

end triangle_and_square_count_l254_254554


namespace cost_per_adult_meal_l254_254176

-- Definitions and given conditions
def total_people : ℕ := 13
def num_kids : ℕ := 9
def total_cost : ℕ := 28

-- Question translated into a proof statement
theorem cost_per_adult_meal : (total_cost / (total_people - num_kids)) = 7 := 
by
  sorry

end cost_per_adult_meal_l254_254176


namespace smallest_value_of_a1_conditions_l254_254040

noncomputable theory

variables {a1 a2 a3 a4 a5 a6 a7 a8 : ℝ}

/-- The smallest value of \(a_1\) when the sum of \(a_1, \ldots, a_8\) is \(4/3\) 
    and the sum of any seven of these numbers is positive. -/
theorem smallest_value_of_a1_conditions 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 / 3)
  (h_sum_seven : ∀ i : {j // j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8}, 
                  0 < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 - a i.val) :
  -8 < a1 ∧ a1 ≤ 1 / 6 :=
sorry

end smallest_value_of_a1_conditions_l254_254040


namespace intersection_of_M_and_N_l254_254749

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254749


namespace smallest_number_bounds_l254_254033

theorem smallest_number_bounds (a : Fin 8 → ℝ)
    (h_sum : (∑ i, a i) = (4 / 3))
    (h_pos : ∀ i, (∑ j in finset.univ.filter (λ j, j ≠ i), a j) > 0) :
    -8 < (finset.univ.arg_min id a).get (finset.univ.nonempty) ∧
    (finset.univ.arg_min id a).get (finset.univ.nonempty) ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l254_254033


namespace intersection_eq_l254_254703

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254703


namespace round_robin_cycles_count_l254_254316

-- Define the problem setup
variables (Team : Type) [fintype Team] [decidable_eq Team] -- 25 teams
variables (beats : Team → Team → Prop) -- relationship where one team beats another
variable [is_tournament : ∀ t, (finset.univ : finset Team).card = 25 ∧
  ∀ t₁ t₂, t₁ ≠ t₂ → (beats t₁ t₂ ∨ beats t₂ t₁) ∧ ¬(beats t₁ t₂ ∧ beats t₂ t₁) ∧
  (finset.filter (λ t, beats t₁ t) finset.univ).card = 12 ∧ 
  (finset.filter (λ t, beats t t₁) finset.univ).card = 12] -- each team wins 12 and loses 12

-- Define the property for a set of three teams to form a cycle
def cycle3 (A B C : Team) : Prop := beats A B ∧ beats B C ∧ beats C A

-- Statement of the theorem
theorem round_robin_cycles_count :
  (finset.univ : finset (finset Team)).filter 
    (λ S, finset.card S = 3 ∧ ∃ (A B C : Team), {A, B, C} = S ∧ cycle3 A B C).card = 1200 :=
sorry

end round_robin_cycles_count_l254_254316


namespace intersection_M_N_l254_254730

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254730


namespace time_gaps_l254_254913

theorem time_gaps (dist_a dist_b dist_c : ℕ) (time_a time_b time_c : ℕ) :
  dist_a = 130 →
  dist_b = 130 →
  dist_c = 130 →
  time_a = 36 →
  time_b = 45 →
  time_c = 42 →
  (time_b - time_a = 9) ∧ (time_c - time_a = 6) ∧ (time_b - time_c = 3) := by
  intros hdist_a hdist_b hdist_c htime_a htime_b htime_c
  sorry

end time_gaps_l254_254913


namespace gretel_received_15_percent_raise_l254_254272

-- Define Hansel's initial salary
def hansel_initial_salary : ℝ := 30000

-- Define the raise percentage Hansel received
def hansel_raise_percentage : ℝ := 0.10

-- Define the raise amount Hansel received
def hansel_raise_amount : ℝ := hansel_initial_salary * hansel_raise_percentage

-- Define Hansel's new salary
def hansel_new_salary : ℝ := hansel_initial_salary + hansel_raise_amount

-- Define the additional amount Gretel received more than Hansel
def gretel_extra_amount : ℝ := 1500

-- Define Gretel's new salary
def gretel_new_salary : ℝ := hansel_new_salary + gretel_extra_amount

-- Define Gretel's initial salary (equal to Hansel's initial salary)
def gretel_initial_salary : ℝ := hansel_initial_salary

-- Define the raise amount Gretel received
def gretel_raise_amount : ℝ := gretel_new_salary - gretel_initial_salary

-- Define the raise percentage Gretel received
def gretel_raise_percentage : ℝ := (gretel_raise_amount / gretel_initial_salary) * 100

-- Prove that Gretel received a 15% raise
theorem gretel_received_15_percent_raise : gretel_raise_percentage = 15 :=
by
  sorry

end gretel_received_15_percent_raise_l254_254272


namespace minimum_reciprocal_sum_l254_254991

theorem minimum_reciprocal_sum (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i) (h_sum : ∑ i, b i = 1) :
  (∑ i, (1 / b i)) ≥ 225 :=
sorry

end minimum_reciprocal_sum_l254_254991


namespace M_inter_N_eq_2_4_l254_254816

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254816


namespace perfectCubesCount_l254_254881

theorem perfectCubesCount (a b : Nat) (h₁ : 50 < a ∧ a ^ 3 > 50) (h₂ : b ^ 3 < 2000 ∧ b < 2000) :
  let n := b - a + 1
  n = 9 := by
  sorry

end perfectCubesCount_l254_254881


namespace perpendicular_line_expression_value_l254_254609

theorem perpendicular_line_expression_value
  (θ : ℝ)
  (h : ∃ l : ℝ, x - 3 * y + l = 0 ∧ (tan θ = -3)) :
  2 / (3 * sin θ ^ 2 - cos θ ^ 2) = 10 / 13 :=
by
  sorry

end perpendicular_line_expression_value_l254_254609


namespace intersection_M_N_l254_254803

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254803


namespace intersection_M_N_l254_254773

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254773


namespace kaylee_current_age_l254_254294

-- Define the initial conditions
def matt_current_age : ℕ := 5
def kaylee_future_age_in_7_years : ℕ := 3 * matt_current_age

-- Define the main theorem to be proven
theorem kaylee_current_age : ∃ x : ℕ, x + 7 = kaylee_future_age_in_7_years ∧ x = 8 :=
by
  -- Use given conditions to instantiate the future age
  have h1 : kaylee_future_age_in_7_years = 3 * 5 := rfl
  have h2 : 3 * 5 = 15 := rfl
  have h3 : kaylee_future_age_in_7_years = 15 := by rw [h1, h2]
  -- Prove that there exists an x such that x + 7 = 15 and x = 8
  use 8
  split
  . rw [h3]
    norm_num
  . rfl

end kaylee_current_age_l254_254294


namespace find_third_smallest_three_digit_palindromic_prime_l254_254439

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 ∧ m < n → n % m ≠ 0

def second_smallest_three_digit_palindromic_prime : ℕ :=
  131 -- Given in the problem statement

noncomputable def third_smallest_three_digit_palindromic_prime : ℕ :=
  151 -- Answer obtained from the solution

theorem find_third_smallest_three_digit_palindromic_prime :
  ∃ n, is_palindrome n ∧ is_prime n ∧ 100 ≤ n ∧ n < 1000 ∧
  (n ≠ 101) ∧ (n ≠ 131) ∧ (∀ m, is_palindrome m ∧ is_prime m ∧ 100 ≤ m ∧ m < 1000 → second_smallest_three_digit_palindromic_prime < m → m = n) :=
by
  sorry -- This is where the proof would be, but it is not needed as per instructions.

end find_third_smallest_three_digit_palindromic_prime_l254_254439


namespace remainder_2011_2015_mod_17_l254_254091

theorem remainder_2011_2015_mod_17 :
  ((2011 * 2012 * 2013 * 2014 * 2015) % 17) = 7 :=
by
  have h1 : 2011 % 17 = 5 := by sorry
  have h2 : 2012 % 17 = 6 := by sorry
  have h3 : 2013 % 17 = 7 := by sorry
  have h4 : 2014 % 17 = 8 := by sorry
  have h5 : 2015 % 17 = 9 := by sorry
  sorry

end remainder_2011_2015_mod_17_l254_254091


namespace part1_part2_part3_monotonicity_l254_254254

noncomputable def f (a x : ℝ) : ℝ := x^2 + a * Real.log x

noncomputable def f_prime (a x : ℝ) : ℝ := 2 * x + a / x

theorem part1 (a : ℝ) : (f_prime a 1 = 0 ↔ a = -2) := 
by 
  split;
  { intro h,
    linarith [h] }

theorem part2 (a : ℝ) : (∀ x > 1, f_prime a x ≥ 0) ↔ (a ≥ -2) :=
by
  split;
  { intro h,
    specialize h 1 (by linarith), 
    linarith [h, (f_prime a 1).symm] }

noncomputable def g (a x : ℝ) : ℝ := f a x - (a + 2) * x

noncomputable def g_prime (a x : ℝ) : ℝ := (x - 1) * (2 * x - a) / x

theorem part3_monotonicity (a : ℝ) : 
  (0 ≥ a → ∀ x > 0, x ≤ 1 → g_prime a x ≤ 0) ∧ 
  (0 ≥ a → ∀ x > 1, g_prime a x ≥ 0) ∧ 
  (0 < a ∧ a < 2 → ∃ c ∈ (0, 1), ∀ x ∈ (0, c), g_prime a x ≥ 0 ∧ 
     ∀ x ∈ (c, 1), g_prime a x ≤ 0 ∧ ∀ x > 1, g_prime a x ≥ 0) ∧ 
  (a =2 → ∀ x > 0, g_prime a x ≥ 0) ∧ 
  (a > 2 → ∃ c ∈ (1, a / 2), ∀ x ∈ (0, 1), g_prime a x ≥ 0 ∧ 
     ∀ x ∈ (1, c), g_prime a x≤ 0 ∧ ∀ x ∈ (c, +∞), g_prime a x≥ 0) := sorry

end part1_part2_part3_monotonicity_l254_254254


namespace largest_valid_subset_card_l254_254153

-- Define the set and the conditions
def valid_subset (S : set ℕ) : Prop :=
  ∀ x y ∈ S, x ≠ y → x ≠ 4 * y ∧ y ≠ 4 * x

-- Define the set of interest
def set150 : set ℕ := {n | 1 ≤ n ∧ n ≤ 150}

-- The theorem stating the desired property
theorem largest_valid_subset_card : ∃ S ⊆ set150, valid_subset S ∧ S.card = 142 := 
by 
  sorry

end largest_valid_subset_card_l254_254153


namespace intersection_M_N_l254_254769

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254769


namespace probability_of_drawing_white_ball_l254_254220

theorem probability_of_drawing_white_ball (P_A P_B P_C : ℝ) 
    (hA : P_A = 0.4) 
    (hB : P_B = 0.25)
    (hSum : P_A + P_B + P_C = 1) : 
    P_C = 0.35 :=
by
    -- Placeholder for the proof
    sorry

end probability_of_drawing_white_ball_l254_254220


namespace flash_catch_up_distance_l254_254157

variables (v x z k : ℝ)

-- Given conditions
axiom h1 : 1 < x

-- Proof statement
theorem flash_catch_up_distance : 
  (D(v, x, z, k) = (x * (z + v * k) / (x - 1))) :=
by
  sorry

end flash_catch_up_distance_l254_254157


namespace intersection_M_N_l254_254649

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254649


namespace total_students_is_2000_l254_254065

theorem total_students_is_2000
  (S : ℝ) 
  (h1 : 0.10 * S = chess_students) 
  (h2 : 0.50 * chess_students = swimming_students) 
  (h3 : swimming_students = 100) 
  (chess_students swimming_students : ℝ) 
  : S = 2000 := 
by 
  sorry

end total_students_is_2000_l254_254065


namespace largest_prime_factor_of_T_l254_254573

-- Definition of the sequence property
def cyclic_property (seq : List ℕ) : Prop :=
  ∀ i, i < seq.length →
    let a := seq.nthLe i (by linarith)
    let b := seq.nthLe ((i + 1) % seq.length) (by linarith [Nat.mod_lt]) in
      (b % 1000) * 10 + (a / 1000) = b

-- The sequence given in the problem
def sequence : List ℕ := [1254, 2547, 5478, 4781]

-- The theorem statement
theorem largest_prime_factor_of_T :
  cyclic_property sequence → (sequence.sum % 101 = 0) :=
by
  intros h
  sorry

end largest_prime_factor_of_T_l254_254573


namespace intersection_M_N_l254_254725

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254725


namespace a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l254_254236

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l254_254236


namespace hyperbola_equation_l254_254203

noncomputable def hyperbola : Prop :=
  ∃ (a b : ℝ), 
    (2 : ℝ) * a = (3 : ℝ) * b ∧
    ∀ (x y : ℝ), (4 * x^2 - 9 * y^2 = -32) → (x = 1) ∧ (y = 2)

theorem hyperbola_equation (a b : ℝ) :
  (2 * a = 3 * b) ∧ (∀ x y : ℝ, 4 * x^2 - 9 * y^2 = -32 → x = 1 ∧ y = 2) → 
  (9 / 32 * y^2 - x^2 / 8 = 1) :=
by
  sorry

end hyperbola_equation_l254_254203


namespace Prove_T1_l254_254614

def S : Type := {dots: Type} × {lines: Type}

def P1 (S : Type) [Fintype S.dots] [Fintype S.lines] (line : set S.dots) : Prop := (∀ l, is_line l → ∃ s, S.dots.s ∈ l)

def P2 (S : Type) [Fintype S.dots] [Fintype S.lines] : Prop :=
  ∀ (l1 l2 : set S.dots), l1 ≠ l2 → ∃ (d : S.dots), d ∈ l1 ∧ d ∈ l2

def P3 (S : Type) [Fintype S.dots] [Fintype S.lines] : Prop :=
  ∀ (d : S.dots), ∃! (l1 l2 l3 : set S.dots), d ∈ l1 ∧ d ∈ l2 ∧ d ∈ l3

def P4 (S : Type) [Fintype S.dots] [Fintype S.lines] : Prop :=
  Fintype.card S.lines = 5
  
theorem Prove_T1 (S : Type) [Fintype S.dots] [Fintype S.lines] 
  (p1 : P1 S) (p2 : P2 S) (p3 : P3 S) (p4 : P4 S) : Fintype.card S.dots = 10 := 
sorry

end Prove_T1_l254_254614


namespace intersection_of_M_and_N_l254_254637

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254637


namespace cone_water_filled_volume_l254_254127

theorem cone_water_filled_volume (r h : ℝ) (h_pos : 0 < h) :
  let volume_original : ℝ := (1/3) * π * r^2 * h in
  let volume_water : ℝ := (1/3) * π * (2/3 * r)^2 * (2/3 * h) in
  (volume_water / volume_original) = 8 / 27 :=
by
  let volume_original := (1/3) * π * r^2 * h
  let volume_water := (1/3) * π * (2/3 * r)^2 * (2/3 * h)
  sorry

end cone_water_filled_volume_l254_254127


namespace rectangle_area_ratio_correct_l254_254026

noncomputable def rectangle_area_ratio_k (length width d : ℝ) (h1 : length / width = 5 / 2) (h2 : 2 * (length + width) = 42) : ℝ :=
  let area := length * width in
  let d_squared := length ^ 2 + width ^ 2 in
  area / d_squared

theorem rectangle_area_ratio_correct :
  ∃ (length width d : ℝ), length / width = 5 / 2 ∧ 2 * (length + width) = 42 ∧ rectangle_area_ratio_k length width d (by sorry) (by sorry) = 10 / 29 :=
sorry

end rectangle_area_ratio_correct_l254_254026


namespace piecewise_value_l254_254846

def piecewise_function (x : ℕ) : ℕ :=
  if x ≤ 1 then 
    1 - x^2 
  else 
    x^2 + x - 2

theorem piecewise_value : 
  f (1 / f 2) = 15 / 16 :=
  by 
  sorry

end piecewise_value_l254_254846


namespace problem_statement_l254_254000

open Real

theorem problem_statement (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : A ≠ 0)
    (h3 : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
    |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| := sorry

end problem_statement_l254_254000


namespace equation_represents_two_lines_l254_254192

theorem equation_represents_two_lines :
  ∀ x y : ℝ, (x^2 - 50 * y^2 - 16 * x + 64 = 0) ↔ (x = 8 + 5 * real.sqrt 2 * y) ∨ (x = 8 - 5 * real.sqrt 2 * y) :=
by
  sorry

end equation_represents_two_lines_l254_254192


namespace cereal_difference_l254_254059

-- Variables to represent the amounts of cereal in each box
variable (A B C : ℕ)

-- Define the conditions given in the problem
def problem_conditions : Prop :=
  A = 14 ∧
  B = A / 2 ∧
  A + B + C = 33

-- Prove the desired conclusion under these conditions
theorem cereal_difference
  (h : problem_conditions A B C) :
  C - B = 5 :=
sorry

end cereal_difference_l254_254059


namespace factorial_expression_1540_l254_254436

theorem factorial_expression_1540 :
  ∃ a_1 a_2 a_m b_1 b_2 b_n : ℕ,
  a_1 >= a_2 ∧ a_2 >= a_m ∧ b_1 >= b_2 ∧ b_2 >= b_n ∧
  1540 = (fact a_1 * fact a_2 * fact a_m) / (fact b_1 * fact b_2 * fact b_n) ∧
  a_1 >= 11 ∧
  (∀ a_1 b_1 : ℕ, a_1 + b_1 ≥ (11 + 9)) ∧
  |a_1 - b_1| = 2 :=
  sorry

end factorial_expression_1540_l254_254436


namespace avg_age_second_group_l254_254409

theorem avg_age_second_group (avg_age_class : ℕ) (avg_age_first_group : ℕ) (age_15th_student : ℕ) (students_class : ℕ) (students_first_group : ℕ) (students_second_group : ℕ) :
  avg_age_class = 15 →
  avg_age_first_group = 14 →
  age_15th_student = 15 →
  students_class = 15 →
  students_first_group = 7 →
  students_second_group = 7 →
  let total_age_class := students_class * avg_age_class,
      total_age_first_group := students_first_group * avg_age_first_group,
      total_age_combined_groups := total_age_class - age_15th_student,
      total_age_second_group := total_age_combined_groups - total_age_first_group,
      avg_age_second_group := total_age_second_group / students_second_group
  in avg_age_second_group = 16 :=
by
  intros h1 h2 h3 h4 h5 h6,
  let total_age_class := students_class * avg_age_class,
  let total_age_first_group := students_first_group * avg_age_first_group,
  let total_age_combined_groups := total_age_class - age_15th_student,
  let total_age_second_group := total_age_combined_groups - total_age_first_group,
  let avg_age_second_group := total_age_second_group / students_second_group,
  rw [h1, h2, h3, h4, h5, h6] at *,
  exact calc 
    avg_age_second_group
    = (total_age_combined_groups - total_age_first_group) / students_second_group : rfl
    ... = 16 : by sorry

end avg_age_second_group_l254_254409


namespace intersection_M_N_l254_254622

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254622


namespace largest_integral_ratio_l254_254370

theorem largest_integral_ratio (P A : ℕ) (rel_prime_sides : ∃ (a b c : ℕ), gcd a b = 1 ∧ gcd b c = 1 ∧ gcd c a = 1 ∧ a^2 + b^2 = c^2 ∧ P = a + b + c ∧ A = a * b / 2) :
  (∃ (k : ℕ), k = 45 ∧ ∀ l, l < 45 → l ≠ (P^2 / A)) :=
sorry

end largest_integral_ratio_l254_254370


namespace median_mode_correct_average_usage_correct_estimated_students_correct_l254_254921

def usage_data : List (ℕ × ℕ) := [(0, 22), (1, 14), (2, 24), (3, 27), (4, 8), (5, 5)]

theorem median_mode_correct (data : List (ℕ × ℕ)) : 
  ∃ median mode, median = 2 ∧ mode = 3 :=
by
  -- Placeholder for proof
  sorry

theorem average_usage_correct (data : List (ℕ × ℕ)) : 
  ∃ avg, avg = 2 :=
by
  -- Placeholder for proof
  sorry

theorem estimated_students_correct (data : List (ℕ × ℕ)) : 
  ∃ est, est = 600 :=
by
  -- Placeholder for proof
  sorry

#eval median_mode_correct usage_data
#eval average_usage_correct usage_data
#eval estimated_students_correct usage_data

end median_mode_correct_average_usage_correct_estimated_students_correct_l254_254921


namespace tenth_number_drawn_l254_254311

-- Define the conditions
def num_students := 1000
def sample_size := 50
def interval := num_students / sample_size
def first_number_drawn := 15   -- Representing 0015

-- Define the proof problem
theorem tenth_number_drawn : first_number_drawn + interval * 9 = 195 := by
  have h_interval : interval = 20 := by sorry
  calc
    15 + 20 * 9 = 15 + 180 := by rfl
    ... = 195 := by rfl

end tenth_number_drawn_l254_254311


namespace intersection_of_M_and_N_l254_254752

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254752


namespace intersection_M_N_l254_254633

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254633


namespace KE_eq_KF_l254_254986

variables (A B C D E F K : Point)
variables (h1 : Parallelogram A B C D)
variables (h2 : E ∈ Segment C D)
variables (h2' : 2 * ∠ A E B = ∠ A D B + ∠ A C B)
variables (h3 : F ∈ Segment B C)
variables (h3' : 2 * ∠ D F A = ∠ D C A + ∠ D B A)
variables (h4 : IsCircumcenter K (Triangle A B D))

theorem KE_eq_KF (h1 : Parallelogram A B C D)
  (h2 : E ∈ Segment C D)
  (h2' : 2 * ∠ A E B = ∠ A D B + ∠ A C B)
  (h3 : F ∈ Segment B C)
  (h3' : 2 * ∠ D F A = ∠ D C A + ∠ D B A)
  (h4 : IsCircumcenter K (Triangle A B D)) : KE = KF :=
sorry

end KE_eq_KF_l254_254986


namespace water_height_l254_254416

open Real

def height_of_water_in_cone (r h : ℝ) (percentage_full : ℝ) : ℝ :=
  let V_tank := (1 / 3) * π * r^2 * h
  let V_water := percentage_full * V_tank
  let y_cubed := V_water / ((1 / 3) * π * (r^2) * h)
  let y := (y_cubed^(1 / 3))
  h * y

theorem water_height (r h : ℝ) (percentage_full : ℝ) (c d : ℕ) (condition : percentage_full = 0.4) : 
  height_of_water_in_cone r h percentage_full = c * (d : ℝ)^(1/3) → c + d = 32 :=
by
  sorry

end water_height_l254_254416


namespace meat_sales_beyond_plan_l254_254390

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end meat_sales_beyond_plan_l254_254390


namespace intersection_of_M_and_N_l254_254751

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254751


namespace one_in_M_l254_254860

def N := { x : ℕ | true } -- Define the natural numbers ℕ

def M : Set ℕ := { x ∈ N | 1 / (x - 2) ≤ 0 }

theorem one_in_M : 1 ∈ M :=
  sorry

end one_in_M_l254_254860


namespace angle_EMK_is_90_deg_l254_254535

-- Define the right triangle ABC with hypotenuse AB and inscribed in a circle
variables {A B C K N M E O : Type*}

-- Some appropriate definitions and assumptions based on the conditions
def circle {A B C : Type*} (O : Type*) (r : ℝ) : Prop := sorry
def is_right_triangle (A B C : Type*) : Prop := sorry
def is_midpoint (X Y Z : Type*) : Prop := sorry
def is_intersection (R S T : Type*) : Prop := sorry
def are_tangents (A B O : Type*) : Prop := sorry

-- Given Conditions
variables (h1: is_right_triangle A B C)
          (h2: circle O A)
          (h3: is_midpoint K B C)
          (h4: ¬ (A ∈ segment K B C))
          (h5: is_midpoint N A C)
          (h6: is_intersection (ray K N) M O)
          (h7: are_tangents A C E)

theorem angle_EMK_is_90_deg :
  angle E M K = 90 :=
sorry

end angle_EMK_is_90_deg_l254_254535


namespace octal_7324_to_decimal_l254_254148

theorem octal_7324_to_decimal :
  let n := 7 * 8^3 + 3 * 8^2 + 2 * 8^1 + 4 * 8^0
  in n = 2004 :=
by
  sorry

end octal_7324_to_decimal_l254_254148


namespace combinations_correct_l254_254063

-- Define the set of balls
inductive Ball
| r1 | r2 | r3 | b1 | b2 | b3

open Ball

-- List all combinations of selecting 3 balls out of 6 balls
def combinations_of_3_balls : set (set Ball) :=
  {{r1, r2, r3}, {b1, b2, b3}, {r1, r2, b1}, {r1, r2, b2}, {r1, r2, b3},
   {r1, r3, b1}, {r1, r3, b2}, {r1, r3, b3}, {r2, r3, b1}, {r2, r3, b2},
   {r2, r3, b3}, {r1, b1, b2}, {r1, b1, b3}, {r1, b2, b3}, {r2, b1, b2},
   {r2, b1, b3}, {r2, b2, b3}, {r3, b1, b2}, {r3, b2, b3}, {r3, b1, b3}}

theorem combinations_correct :
  combinations_of_3_balls =
    {{r1, r2, r3}, {b1, b2, b3}, {r1, r2, b1}, {r1, r2, b2}, {r1, r2, b3},
     {r1, r3, b1}, {r1, r3, b2}, {r1, r3, b3}, {r2, r3, b1}, {r2, r3, b2},
     {r2, r3, b3}, {r1, b1, b2}, {r1, b1, b3}, {r1, b2, b3}, {r2, b1, b2},
     {r2, b1, b3}, {r2, b2, b3}, {r3, b1, b2}, {r3, b2, b3}, {r3, b1, b3}} :=
sorry

end combinations_correct_l254_254063


namespace intersection_M_N_l254_254724

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254724


namespace intersection_M_N_l254_254714

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254714


namespace num_experimental_setups_l254_254062

-- Let's define the problem conditions first

def elements_A : Type := Fin 6  -- Represents 6 different elements of type A
def elements_B : Type := Fin 4  -- Represents 4 different elements of type B

def even_subsets_A (l : List (Fin 6)) : Prop :=
  l.length % 2 = 0

def subsets_B_at_least_2 (l : List (Fin 4)) : Prop :=
  l.length >= 2

-- The main theorem to prove the number of valid experimental setups is 352
theorem num_experimental_setups :
  ∃ (plans : Set (List elements_A × List elements_B)),
    (∀ p ∈ plans, even_subsets_A p.1 ∧ subsets_B_at_least_2 p.2) ∧
    plans.to_finset.card = 352 :=
sorry

end num_experimental_setups_l254_254062


namespace andrew_calculation_l254_254174

theorem andrew_calculation (x y : ℝ) (hx : x ≠ 0) :
  0.4 * 0.5 * x = 0.2 * 0.3 * y → y = (10 / 3) * x :=
by
  sorry

end andrew_calculation_l254_254174


namespace intersection_M_N_l254_254758

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254758


namespace intersection_of_M_and_N_l254_254750

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254750


namespace powers_of_3_solution_l254_254576

theorem powers_of_3_solution (n : ℕ) :
  (∃ m : ℕ, n = 3 ^ m) ↔ ∀ k : ℤ, ∃ a : ℤ, ↑n ∣ (a ^ 3 + a - k) :=
by
  sorry

end powers_of_3_solution_l254_254576


namespace necklaces_rel_prime_l254_254002

theorem necklaces_rel_prime (n : ℤ) (hn : odd n) (hn_ge : n ≥ 1)
  (beads_A beads_B : List ℤ) (hA_len : beads_A.length = 14) (hB_len : beads_B.length = 19)
  (hUnion : beads_A ++ beads_B = List.range (n) (33) -- range positions for n to n+32
  (hAdjacent : ∀ (i : ℤ), i ∈ beads_A ++ beads_B → i + 1 ∈ beads_A ++ beads_B → gcd i (i + 1) = 1 ∧ 
    i - 1 ∈ beads_A ++ beads_B → gcd i (i - 1) = 1) :
  ∃ (x : List ℤ)(y : List ℤ), (x ++ y = List.range (n) (33)) ∧
    (∀ (i : ℤ), i ∈ x ++ y → i + 1 ∈ x ++ y → gcd i (i + 1) = 1) ∧
    (∀ (i : ℤ), i ∈ x ++ y → i - 1 ∈ x ++ y → gcd i (i - 1) = 1) ∧ 
    (x.length = 14) ∧ (y.length = 19) :=
sorry

end necklaces_rel_prime_l254_254002


namespace triangle_ABC_solve_B_triangle_ABC_range_l254_254292

variable {A B C a b c : ℝ}

-- Triangles' conditions and necessary imports
theorem triangle_ABC_solve_B (h : (sin A - sqrt 2 * sin C)/(sin B + sin C) = (b - c)/a) :
  ∃ B, B = π / 4 :=
by
  sorry

theorem triangle_ABC_range (h : (sin A - sqrt 2 * sin C)/(sin B + sin C) = (b - c)/a) :
  ∀ A, (sqrt 2 * cos A + cos (π - A - π/4)) ∈ (0, 1] :=
by
  sorry

end triangle_ABC_solve_B_triangle_ABC_range_l254_254292


namespace part1_price_light_bulb_motor_part2_minimal_cost_l254_254327

-- Define the conditions
noncomputable def sum_price : ℕ := 12
noncomputable def total_cost_light_bulbs : ℕ := 30
noncomputable def total_cost_motors : ℕ := 45
noncomputable def ratio_light_bulbs_motors : ℕ := 2
noncomputable def total_items : ℕ := 90
noncomputable def max_ratio_light_bulbs_motors : ℕ := 2

-- Statement of the problems
theorem part1_price_light_bulb_motor (x : ℕ) (y : ℕ):
  x + y = sum_price → 
  total_cost_light_bulbs = 30 →
  total_cost_motors = 45 →
  total_cost_light_bulbs / x = ratio_light_bulbs_motors * (total_cost_motors / y) → 
  x = 3 ∧ y = 9 := 
sorry

theorem part2_minimal_cost (m : ℕ) (n : ℕ):
  m + n = total_items →
  m ≤ total_items / max_ratio_light_bulbs_motors →
  let cost := 3 * m + 9 * n in
  (∀ x y, x + y = total_items → x ≤ total_items / max_ratio_light_bulbs_motors → cost ≤ 3 * x + 9 * y) → 
  m = 30 ∧ n = 60 ∧ cost = 630 :=
sorry

end part1_price_light_bulb_motor_part2_minimal_cost_l254_254327


namespace minimum_value_225_l254_254988

noncomputable def min_value_inverse_sum (b : Fin 15 → ℝ) : ℝ := 
  ∑ i, 1 / b i

theorem minimum_value_225 (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i) (h_sum : ∑ i, b i = 1) :
  min_value_inverse_sum b ≥ 225 :=
  sorry

end minimum_value_225_l254_254988


namespace two_rel_prime_exists_l254_254462

theorem two_rel_prime_exists (A : Finset ℕ) (h1 : A.card = 2011) (h2 : ∀ x ∈ A, 1 ≤ x ∧ x ≤ 4020) : 
  ∃ (a b : ℕ), a ∈ A ∧ b ∈ A ∧ a ≠ b ∧ Nat.gcd a b = 1 :=
by
  sorry

end two_rel_prime_exists_l254_254462


namespace pentagon_quadrilateral_containment_l254_254943

variable {Point : Type}
variable [AffineSpace V Point]

def is_convex_pentagon (A B C D E : Point) : Prop :=
  -- Assuming we have a way to define convex pentagon
  sorry

def lies_inside (P : Point) (polygon : Set Point) : Prop :=
  -- Assuming we have a definition to check if a point lies inside a polygon
  sorry

theorem pentagon_quadrilateral_containment 
  (A B C D E M N : Point) 
  (h_convex : is_convex_pentagon A B C D E)
  (h_M : lies_inside M {A, B, C, D, E})
  (h_N : lies_inside N {A, B, C, D, E}) :
  ∃ (P Q R S : Point), {P, Q, R, S} ⊆ {A, B, C, D, E} ∧ 
  lies_inside M {P, Q, R, S} ∧ lies_inside N {P, Q, R, S} :=
sorry

end pentagon_quadrilateral_containment_l254_254943


namespace composition_of_two_symmetries_is_rotation_any_rotation_as_composition_of_symmetries_l254_254500

def intersection_at_point (s1 s2: axis) (O: point) : Prop :=
  s1 ∩ s2 = O

def angle_between_axes (s1 s2: axis) (φ: ℝ) : Prop :=
  angle (s1, s2) = φ

def rotation_about_axis (s1 s2: axis) (O: point) (l: axis) (θ: ℝ) : Prop :=
  is_rotation (composition (symmetry s1) (symmetry s2)) l θ

theorem composition_of_two_symmetries_is_rotation (s1 s2: axis) (O: point) (φ: ℝ) :
  intersection_at_point s1 s2 O → angle_between_axes s1 s2 φ →
  ∃ l θ, rotation_about_axis s1 s2 O l (2 * φ) :=
by
  intros h_intersect h_angle
  sorry

theorem any_rotation_as_composition_of_symmetries (l: axis) (θ: ℝ) :
  ∃ s1 s2 O φ, rotation_about_axis s1 s2 O l θ ∧ intersection_at_point s1 s2 O ∧ angle_between_axes s1 s2 (θ / 2) :=
by
  sorry

end composition_of_two_symmetries_is_rotation_any_rotation_as_composition_of_symmetries_l254_254500


namespace intersection_M_N_l254_254676

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254676


namespace intersection_M_N_l254_254799

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254799


namespace problem_arithmetic_sequence_l254_254606

noncomputable def arithmetic_sequence (a d : ℝ) (n : ℕ) := a + d * (n - 1)

theorem problem_arithmetic_sequence (a d : ℝ) (h₁ : d < 0) (h₂ : (arithmetic_sequence a d 1)^2 = (arithmetic_sequence a d 9)^2):
  (arithmetic_sequence a d 5) = 0 :=
by
  -- This is where the proof would go
  sorry

end problem_arithmetic_sequence_l254_254606


namespace intersection_M_N_l254_254660

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254660


namespace sqrt_meaningful_real_range_l254_254907

theorem sqrt_meaningful_real_range (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 :=
by
  sorry

end sqrt_meaningful_real_range_l254_254907


namespace intersection_M_N_l254_254668

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254668


namespace arc_length_of_sector_l254_254088

noncomputable def arc_length (r : ℝ) (θ : ℝ) : ℝ := (θ / 360) * 2 * real.pi * r

theorem arc_length_of_sector :
  arc_length 15 42 = 3.5 * real.pi :=
by
  sorry

end arc_length_of_sector_l254_254088


namespace find_divisor_l254_254085

theorem find_divisor 
  (dividend : ℤ)
  (quotient : ℤ)
  (remainder : ℤ)
  (divisor : ℤ)
  (h : dividend = (divisor * quotient) + remainder)
  (h_dividend : dividend = 474232)
  (h_quotient : quotient = 594)
  (h_remainder : remainder = -968) :
  divisor = 800 :=
sorry

end find_divisor_l254_254085


namespace joy_quadrilateral_l254_254351

theorem joy_quadrilateral :
  let lengths := (Finset.range 41).filter (λ n, n > 0)
  let used_lengths := {5, 12, 20}
  let remaining_lengths := lengths \ used_lengths
  let possible_lengths := remaining_lengths.filter (λ x, 4 ≤ x ∧ x ≤ 36)
  possible_lengths.card = 30 :=
by
  sorry

end joy_quadrilateral_l254_254351


namespace intersection_M_N_l254_254732

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254732


namespace fill_time_both_pipes_l254_254114

def rate_pipe_A : ℝ := 1 / 10
def rate_pipe_B : ℝ := 1 / 20
def combined_rate : ℝ := rate_pipe_A + rate_pipe_B

theorem fill_time_both_pipes :
  1 / combined_rate = 20 / 3 :=
by
  sorry

end fill_time_both_pipes_l254_254114


namespace smallest_number_bounds_l254_254035

theorem smallest_number_bounds (a : Fin 8 → ℝ)
    (h_sum : (∑ i, a i) = (4 / 3))
    (h_pos : ∀ i, (∑ j in finset.univ.filter (λ j, j ≠ i), a j) > 0) :
    -8 < (finset.univ.arg_min id a).get (finset.univ.nonempty) ∧
    (finset.univ.arg_min id a).get (finset.univ.nonempty) ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l254_254035


namespace lattice_points_hyperbola_count_l254_254276

theorem lattice_points_hyperbola_count : 
  {p : ℤ × ℤ | p.fst^2 - p.snd^2 = 1800^2}.to_finset.card = 150 :=
sorry

end lattice_points_hyperbola_count_l254_254276


namespace simplify_expression_l254_254403

noncomputable def givenExpression : ℝ := 
  abs (-0.01) ^ 2 - (-5 / 8) ^ 0 - 3 ^ (Real.log 2 / Real.log 3) + 
  (Real.log 2) ^ 2 + (Real.log 2) * (Real.log 5) + Real.log 5

theorem simplify_expression : givenExpression = -1.9999 := by
  sorry

end simplify_expression_l254_254403


namespace intersection_M_N_l254_254709

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254709


namespace same_final_sum_l254_254507

variable (n : ℕ)
def consecutive_numbers := List.range 10 |>.map (λ i => n + i)

def petya_pairs (lst : List ℕ) : List (ℕ × ℕ) :=
[(lst[0], lst[1]), (lst[2], lst[3]), (lst[4], lst[5]), (lst[6], lst[7]), (lst[8], lst[9])]

def vasya_pairs (lst : List ℕ) : List (ℕ × ℕ) :=
[(lst[0], lst[3]), (lst[1], lst[2]), (lst[4], lst[7]), (lst[5], lst[6]), (lst[8], lst[9])]

def sum_of_products (pairs : List (ℕ × ℕ)) : ℕ :=
pairs.map (λ (a, b) => a * b) |>.sum

theorem same_final_sum :
    sum_of_products (petya_pairs (consecutive_numbers n)) = sum_of_products (vasya_pairs (consecutive_numbers n)) :=
by
  sorry

end same_final_sum_l254_254507


namespace road_sign_sums_not_all_distinct_l254_254161

-- Defining the problem conditions in Lean
def road_sign_sums (distances : Fin 60 → Fin 60 → ℝ) : Fin 60 → ℝ :=
  λ i => ∑ j in Finset.univ \ {i}, distances i j

-- Main theorem to prove
theorem road_sign_sums_not_all_distinct (distances : Fin 60 → Fin 60 → ℝ) :
  ∃ i j : Fin 60, i ≠ j ∧ road_sign_sums distances i = road_sign_sums distances j :=
sorry

end road_sign_sums_not_all_distinct_l254_254161


namespace total_score_equals_total_count_l254_254509

theorem total_score_equals_total_count 
  (n : ℕ) 
  (a : Fin n → ℕ)
  (b : Fin 100 → ℕ)
  (M : ℕ)
  (N : ℕ)
  (h1 : M = ∑ i in Finset.range 100, (i + 1) * b ⟨i, by linarith⟩)
  (h2 : N = ∑ i in Finset.range 100, b ⟨i, by linarith⟩ )
  : M = N :=
sorry

end total_score_equals_total_count_l254_254509


namespace measure_angle_DQP_l254_254185

/-- If a circle Omega is both the incircle of triangle DEF and the circumcircle of triangle PQR,
with P on EF, Q on DE, and R on DF, and given angle D = 50 degrees, angle E = 70 degrees, 
and angle F = 60 degrees, then the measure of angle DQP is 60 degrees. -/
theorem measure_angle_DQP {D E F P Q R : Point} {Omega : Circle} 
  (h1 : Omega.is_incircle_of_triangle DEF) (h2 : Omega.is_circumcircle_of_triangle PQR)
  (h3 : P ∈ Segment EF) (h4 : Q ∈ Segment DE) (h5 : R ∈ Segment DF)
  (h6 : ∠D = 50) (h7 : ∠E = 70) (h8 : ∠F = 60) : ∠DQP = 60 :=
sorry

end measure_angle_DQP_l254_254185


namespace intersection_of_M_and_N_l254_254644

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254644


namespace number_of_prime_divisors_420_l254_254889

theorem number_of_prime_divisors_420 : 
  ∃ (count : ℕ), (∀ (p : ℕ), prime p → p ∣ 420 → p ∈ {2, 3, 5, 7}) ∧ count = 4 := 
by
  sorry

end number_of_prime_divisors_420_l254_254889


namespace sum_series_l254_254182

theorem sum_series :
  (1 + ∑ k in finset.range(2015), (1:ℝ) / (k + 1) - (1:ℝ) / (k + 2)) = (2015:ℝ) / 1008 :=
by
  sorry

end sum_series_l254_254182


namespace find_tilda_q_l254_254142

noncomputable def is_unruly (p : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), (∀ x, p(x) = x^2 + b*x + c) ∧
  (∃ t u v w : ℝ, t ≠ u ∧ t ≠ v ∧ t ≠ w ∧ u ≠ v ∧ u ≠ w ∧ v ≠ w ∧ 
  p(p(t)) + 1 = 0 ∧ p(p(u)) + 1 = 0 ∧  p(p(v)) + 1 = 0∧  p(p(w)) + 1 = 0)

def is_min_product (p : ℝ → ℝ) : Prop :=
  ∃ c, (∀ (b x: ℝ), p(x) = x^2 + b*x + c) ∧ c = -1

def tilda_q (x : ℝ) : ℝ := 
x^2 - 1

theorem find_tilda_q :
  is_unruly tilda_q ∧ is_min_product tilda_q → tilda_q 1 = 0 :=
  by sorry

end find_tilda_q_l254_254142


namespace max_distinct_numbers_l254_254472

theorem max_distinct_numbers (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 1000) :
  ∃ (s : set ℕ), (∀ x y ∈ s, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6)) ∧ s.card = 400 :=
by
  sorry

end max_distinct_numbers_l254_254472


namespace parabola_area_p_slope_MN_equals_slope_tangent_l254_254264

-- Problem 1
theorem parabola_area_p (C : ℝ → ℝ → Prop) (F : ℝ × ℝ) (p : ℝ) (p_pos : p > 0) (Q R S : ℝ × ℝ) (area_QRS : 4) :
  (C Q.1 Q.2 ∧ C R.1 R.2 ∧ (S.1, -p/2) = S ∧ (Q.1^2 = 2*p*Q.2) ∧ (R.1^2 = 2*p*R.2) ∧ abs ((Q.1 - R.1) * p) = 4) →
  p = 2 :=
by sorry

-- Problem 2
theorem slope_MN_equals_slope_tangent (C: ℝ → ℝ → Prop) (A : ℝ × ℝ) (x0 y0 p: ℝ) (x0_ne_zero: x0 ≠ 0) (M N : ℝ × ℝ) 
      (A1: ℝ × ℝ) (tangent_slope_A1: ℝ) :
  (A = (x0, y0) ∧ A1 = (-x0, y0) ∧ C M.1 M.2 ∧ C N.1 N.2 ∧ tangent_slope_A1 = -x0 / p ∧ 
  (C (x0 + M.1) ((x0 + M.1)^2 / (2*p)) ∧ C (x0 + N.1) ((x0 + N.1)^2 / (2*p))) ∧ 
  (tangent_slope_A1 = -x0 / p ) ∧ 
  (let k_AM := (M.2 - y0) / (M.1 - x0), k_AN := (N.2 - y0) / (N.1 - x0) in 
  k_AM + k_AN = 0) → 
  (let k_MN := (N.2 - M.2) / (N.1 - M.1) in k_MN = tangent_slope_A1) :=
by sorry

end parabola_area_p_slope_MN_equals_slope_tangent_l254_254264


namespace employed_males_percentage_l254_254335

theorem employed_males_percentage (total_population employed employed_as_percent employed_females female_as_percent employed_males employed_males_percentage : ℕ) 
(total_population_eq : total_population = 100)
(employed_eq : employed = employed_as_percent * total_population / 100)
(employed_as_percent_eq : employed_as_percent = 60)
(employed_females_eq : employed_females = female_as_percent * employed / 100)
(female_as_percent_eq : female_as_percent = 25)
(employed_males_eq : employed_males = employed - employed_females)
(employed_males_percentage_eq : employed_males_percentage = employed_males * 100 / total_population) :
employed_males_percentage = 45 :=
sorry

end employed_males_percentage_l254_254335


namespace perfect_cubes_count_l254_254879

def is_perfect_cube (n : ℕ) : Prop := ∃ (k : ℕ), k^3 = n

theorem perfect_cubes_count : 
  (∀ (n : ℕ), n ≥ 51 ∧ n ≤ 1999 → (is_perfect_cube n) -> 
  (n = 64 ∨ n = 125 ∨ n = 216 ∨ n = 343 ∨ n = 512 ∨ n = 729 ∨ n = 1000 ∨ n = 1331 ∨ n = 1728)) ∧
  (∀ (n : ℕ), n ≥ 50 ∧ n ≤ 2000 → (is_perfect_cube n) -> 
  ((n = 64 ∨ n = 125 ∨ n = 216 ∨ n = 343 ∨ n = 512 ∨ n = 729 ∨ n = 1000 ∨ n = 1331 ∨ n = 1728) -> True)) :=
begin
  sorry
end

end perfect_cubes_count_l254_254879


namespace initial_deadlift_weight_l254_254346

theorem initial_deadlift_weight
    (initial_squat : ℕ := 700)
    (initial_bench : ℕ := 400)
    (D : ℕ)
    (squat_loss : ℕ := 30)
    (deadlift_loss : ℕ := 200)
    (new_total : ℕ := 1490) :
    (initial_squat * (100 - squat_loss) / 100) + initial_bench + (D - deadlift_loss) = new_total → D = 800 :=
by
  sorry

end initial_deadlift_weight_l254_254346


namespace max_distinct_numbers_l254_254473

theorem max_distinct_numbers (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 1000) :
  ∃ (s : set ℕ), (∀ x y ∈ s, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6)) ∧ s.card = 400 :=
by
  sorry

end max_distinct_numbers_l254_254473


namespace M_inter_N_eq_2_4_l254_254814

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254814


namespace max_x_plus_y_l254_254373

theorem max_x_plus_y (x y: ℝ) 
  (h1: 4 * x + 3 * y ≤ 9)
  (h2: 3 * x + 5 * y ≤ 10) : 
  x + y ≤ 93 / 44 :=
begin
  sorry
end

end max_x_plus_y_l254_254373


namespace intersection_M_N_l254_254757

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254757


namespace count_sequences_9_terms_l254_254613

theorem count_sequences_9_terms
    (a : Fin 9 → ℤ)
    (h1 : a 0 = 1)
    (h9 : a 8 = 1)
    (hcond : ∀ i : Fin 8, a (i + 1) / a i ∈ ({2, 1, -1/2} : Set ℚ)) :
    ∃ n : ℕ, n = 491 :=
by
    sorry

end count_sequences_9_terms_l254_254613


namespace smallest_number_bounds_l254_254049

theorem smallest_number_bounds (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos : ∀ i, 0 < ∑ j, if j = i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (Finset.univ_nonempty)) a ∧
  (Finset.min' Finset.univ (Finset.univ_nonempty)) a ≤ 1 / 6 :=
begin
  sorry
end

end smallest_number_bounds_l254_254049


namespace kim_morning_routine_time_l254_254356

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end kim_morning_routine_time_l254_254356


namespace intersection_M_N_l254_254630

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254630


namespace intersection_M_N_l254_254619

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254619


namespace sum_valid_student_rows_l254_254918

theorem sum_valid_student_rows (x y : ℕ) (h : x * y = 360) (hx : x ≥ 18) (hy : y ≥ 12) :
    ∑ (d : ℕ) in (finset.filter (λ x, 360 % x = 0 ∧ x ≥ 18 ∧ 360 / x ≥ 12) (finset.range 361)), d = 92 :=
begin
  sorry
end

end sum_valid_student_rows_l254_254918


namespace intersection_M_N_l254_254652

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254652


namespace total_length_of_intervals_l254_254963

def S : set ℝ := {x | 0 ≤ x ∧ x ≤ 2016 * Real.pi ∧ Real.sin x < 3 * Real.sin (x / 3)}

theorem total_length_of_intervals : ∃ l : ℝ, (∀ I ∈ (set.disjoint_intervals S), set.length I = l) ∧ l = 1008 * Real.pi := by
  sorry

end total_length_of_intervals_l254_254963


namespace count_divisors_multiple_of_5_l254_254875

-- Define the conditions as Lean definitions
def prime_factorization (n : ℕ) := 
  n = 2^2 * 3^3 * 5^2

def is_divisor (d : ℕ) (n : ℕ) :=
  d ∣ n

def is_multiple_of_5 (d : ℕ) :=
  ∃ a b c, d = 2^a * 3^b * 5^c ∧ 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2

-- The theorem to be proven
theorem count_divisors_multiple_of_5 (h: prime_factorization 5400) : 
  {d : ℕ | is_divisor d 5400 ∧ is_multiple_of_5 d}.to_finset.card = 24 :=
by {
  sorry -- Proof goes here
}

end count_divisors_multiple_of_5_l254_254875


namespace intersection_of_M_and_N_l254_254797

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254797


namespace intersection_complement_M_N_l254_254862

def U : Set Int := {0, -1, -2, -3, -4}
def M : Set Int := {0, -1, -2}
def N : Set Int := {0, -3, -4}

theorem intersection_complement_M_N :
  (U \ M) ∩ N = {-3, -4} :=
by {
  sorry
}

end intersection_complement_M_N_l254_254862


namespace kevin_hops_six_hops_total_distance_is_485_over_256_l254_254352

theorem kevin_hops_six_hops_total_distance_is_485_over_256 :
  let total_distance (n : ℕ) : ℚ :=
    (match n with
      | 1 => 2 / 2
      | 2 => (2 - 2 / 2) / 4
      | 3 => ((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2
      | 4 => (((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) / 4
      | 5 => ((((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) - ((((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) / 4)) / 2
      | 6 => (((((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) - ((((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) / 4)) - ((((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) - ((((2 - 2 / 2) - (2 - 2 / 2) / 4) - (((2 - 2 / 2) - (2 - 2 / 2) / 4) / 2)) / 4)) / 2)/ 4
      | _ => 0
    ) in (2 - ∑ i in range 1 n, total_distance i) / 4)
  total_distance 6 = 485 / 256 := 
by
  sorry

end kevin_hops_six_hops_total_distance_is_485_over_256_l254_254352


namespace min_box_dimension_sum_l254_254418

theorem min_box_dimension_sum :
  ∃ (a b c : ℕ), (a > 0) ∧ (b > 0) ∧ (c > 0) ∧ (a * b * c = 2310) ∧ (a + b + c = 42) :=
begin
  sorry
end

end min_box_dimension_sum_l254_254418


namespace regular_polygon_sides_l254_254902

theorem regular_polygon_sides (n : ℕ) (h : ∀ (θ : ℝ), θ = 36 → θ = 360 / n) : n = 10 := by
  sorry

end regular_polygon_sides_l254_254902


namespace intersection_M_N_l254_254811

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254811


namespace surrounding_circle_radius_l254_254125

theorem surrounding_circle_radius :
  ∀ (r : ℝ), 
  (∀ (radius_central : ℝ), radius_central = 2 → 
  ∀ (tangent : ∀ (r : ℝ) (c : ℝ),  c = radius_central + r), 
  tangent r r → r = (4 * Real.sqrt 2 + 2) / 7) :=
by
  sorry

end surrounding_circle_radius_l254_254125


namespace intersection_M_N_l254_254655

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254655


namespace cos_4theta_l254_254899

theorem cos_4theta (θ : ℝ) (h : Real.cos θ = 1 / 3) : Real.cos (4 * θ) = 17 / 81 := 
by 
  sorry

end cos_4theta_l254_254899


namespace remainder_of_large_product_mod_17_l254_254093

theorem remainder_of_large_product_mod_17 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end remainder_of_large_product_mod_17_l254_254093


namespace log_identity_l254_254187

open Real

theorem log_identity :
  log 10 8 + 3 * log 10 2 + 4 * log 10 5 + 2 * log 10 3 + log 2 8 = log 10 3.6 + 8 :=
by sorry

end log_identity_l254_254187


namespace geometric_sequence_general_term_l254_254309

theorem geometric_sequence_general_term (q : ℕ) (S : ℕ) (a_n : ℕ → ℕ) :
  q = 4 → S = 21 → (∀ n, a_n = 4^(n-1)) :=
begin
  intros hq hS,
  sorry -- The detailed proof would go here
end

end geometric_sequence_general_term_l254_254309


namespace smallest_n_for_g_iterated_applications_l254_254574

def g (n : ℕ) : ℕ :=
if n % 2 = 1 then n^2 + 1 else n / 2 + 3

theorem smallest_n_for_g_iterated_applications : ∃ k : ℕ, (iterated g k 8) = 7 ∧ (∀ m < 8, ¬ ∃ k : ℕ, (iterated g k m) = 7) :=
sorry

end smallest_n_for_g_iterated_applications_l254_254574


namespace rectangle_area_l254_254394

variable {a b : ℝ}
variable {A B C D E F : Point}

-- A, B, C, D are vertices of the rectangle
-- Assume coordinates for the vertices
axiom A_coord : A = (0, b)
axiom B_coord : B = (a, b)
axiom C_coord : C = (a, 0)
axiom D_coord : D = (0, 0)

-- E is two-thirds the way from C to D
axiom E_coord : E = (2*a/3, 0)

-- F is the intersection of BE and AC
axiom BE_line : ∃ m, ∀ x, (x, m * x + b) ∈ Line B E
axiom AC_line : ∃ n, ∀ x, (x, -b/a * x + b) ∈ Line A C
axiom F_intersection : ∃ F, (F ∈ Line B E) ∧ (F ∈ Line A C)

-- Area of quadrilateral AFED is 36
axiom area_AFED : area_quadrilateral A F E D = 36

-- We need to show that the area of rectangle ABCD is 72
theorem rectangle_area : a * b = 72 := 
sorry

end rectangle_area_l254_254394


namespace intersection_M_N_l254_254674

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254674


namespace intersection_eq_l254_254705

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254705


namespace location_D_meets_condition_l254_254196

-- Definitions
def locA := (average 3 ∧ median 4)
def locB := (average 1 ∧ variance > 0)
def locC := (median 2 ∧ mode 3)
def locD := (average 2 ∧ variance = 3)

-- Condition that needs to be met
def condition_met := ∀ (daily_cases : Finset ℝ), (∀ x ∈ daily_cases, x ≤ 7) ∧ daily_cases.card = 10

-- Locations information
variables {A B C D : Finset ℝ}

-- Stating that location D meets the condition
theorem location_D_meets_condition :
  locD → condition_met D := by
  intros locD
  sorry

end location_D_meets_condition_l254_254196


namespace intersection_M_N_l254_254683

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254683


namespace smallest_number_bounds_l254_254047

theorem smallest_number_bounds (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos : ∀ i, 0 < ∑ j, if j = i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (Finset.univ_nonempty)) a ∧
  (Finset.min' Finset.univ (Finset.univ_nonempty)) a ≤ 1 / 6 :=
begin
  sorry
end

end smallest_number_bounds_l254_254047


namespace cos_sum_zero_sin_cos_sum_one_l254_254181

-- Proof problem for the first question
theorem cos_sum_zero : 
  cos (π / 5) + cos (2 * π / 5) + cos (3 * π / 5) + cos (4 * π / 5) = 0 := 
by 
  sorry

-- Proof problem for the second question
theorem sin_cos_sum_one : 
  sin (420 * (π / 180)) * cos (330 * (π / 180)) + sin (-690 * (π / 180)) * cos (-660 * (π / 180)) = 1 := 
by 
  sorry

end cos_sum_zero_sin_cos_sum_one_l254_254181


namespace lattice_points_on_hyperbola_l254_254274

theorem lattice_points_on_hyperbola : 
  ∃ n, (∀ x y : ℤ, x^2 - y^2 = 1800^2 → (x, y) ∈ {p : ℤ × ℤ | 
  ∃ a b : ℤ, x = 2 * a + b ∧ y = 2 * a - b}) ∧ n = 250 := 
by {
  sorry
}

end lattice_points_on_hyperbola_l254_254274


namespace larger_square_side_length_l254_254441

theorem larger_square_side_length :
  ∃ (a : ℕ), ∃ (b : ℕ), a^2 = b^2 + 2001 ∧ (a = 1001 ∨ a = 335 ∨ a = 55 ∨ a = 49) :=
by
  sorry

end larger_square_side_length_l254_254441


namespace find_largest_smallest_A_l254_254170

theorem find_largest_smallest_A :
  ∃ (B : ℕ), B > 7777777 ∧ B.gcd 36 = 1 ∧
  (let A := 10^7 * (B % 10) + (B / 10) in A = 99999998 ∨ A = 17777779) :=
begin
  sorry
end

end find_largest_smallest_A_l254_254170


namespace intersection_of_M_and_N_l254_254786

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254786


namespace money_left_for_other_expenses_l254_254057

noncomputable def budget : ℝ := 40  -- million
noncomputable def policing_ratio : ℝ := 0.35
noncomputable def education_ratio : ℝ := 0.25
noncomputable def healthcare_ratio : ℝ := 0.15

theorem money_left_for_other_expenses :
  let policing := policing_ratio * budget
  let education := education_ratio * budget
  let healthcare := healthcare_ratio * budget
  let total_spent := policing + education + healthcare in
  budget - total_spent = 10 := by
  sorry

end money_left_for_other_expenses_l254_254057


namespace Albert_more_rocks_than_Joshua_l254_254950

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_rocks_than_Joshua_l254_254950


namespace intersection_M_N_l254_254780

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254780


namespace inequality_iff_l254_254234

theorem inequality_iff (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) : (a > b) ↔ (1/a < 1/b) = false :=
by
  sorry

end inequality_iff_l254_254234


namespace intersection_M_N_l254_254737

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254737


namespace intersection_of_M_and_N_l254_254744

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254744


namespace counts_duel_with_marquises_l254_254310

theorem counts_duel_with_marquises (x y z k : ℕ) (h1 : 3 * x = 2 * y) (h2 : 6 * y = 3 * z)
    (h3 : ∀ c : ℕ, c = x → ∃ m : ℕ, m = k) : k = 6 :=
by
  sorry

end counts_duel_with_marquises_l254_254310


namespace geometric_sequence_product_l254_254339

theorem geometric_sequence_product {a : ℕ → ℝ} 
(h₁ : a 1 = 2) 
(h₂ : a 5 = 8) 
(h_geom : ∀ n, a (n+1) / a n = a (n+2) / a (n+1)) :
a 2 * a 3 * a 4 = 64 := 
sorry

end geometric_sequence_product_l254_254339


namespace aiolis_population_max_capacity_l254_254262

theorem aiolis_population_max_capacity (S : ℕ) (R : ℕ) (P : ℕ → ℕ) (Y : ℕ) : 
  S = 30000 → 
  R = 30 → 
  (∀ n, P (Y + n * R) = P Y * 3 ^ n) → 
  Y = 2000 → 
  (P 2000 = 240) → 
  (S / 1.8 : ℕ) ≤ P 2100 →
  Y + 120 = 2120 :=
by
  intros hS hR hP hY hP2000 hMaxCapacity
  sorry

end aiolis_population_max_capacity_l254_254262


namespace ellipse_area_proof_l254_254244

noncomputable def ellipse_area_proof_problem
    (x y : ℝ)
    (on_ellipse : (x^2 / 25) + (y^2 / 16) = 1)
    (area_triangle : 6 = 3 * |y|)
    (a b c perimeter inscribed_radius circumscribed_radius : ℝ)
    (a_def : a = 5)
    (b_def : b = 4)
    (c_def : c = 3)
    (perimeter_def : perimeter = 16)
    (inscribed_radius_def : inscribed_radius = 3 / 4)
    (circumscribed_radius_def : circumscribed_radius = 73 / 16) : Prop :=
    perimeter_def ∧ inscribed_radius_def ∧ circumscribed_radius_def

theorem ellipse_area_proof
    (x y : ℝ)
    (h_on_ellipse : (x^2 / 25) + (y^2 / 16) = 1)
    (h_area_triangle : 6 = 3 * |y|)
    (a := 5)
    (b := 4)
    (c := 3)
    (perimeter := 16)
    (inscribed_radius := 3 / 4)
    (circumscribed_radius := 73 / 16) : 
    ellipse_area_proof_problem x y h_on_ellipse h_area_triangle a b c perimeter inscribed_radius circumscribed_radius :=
by
    sorry

end ellipse_area_proof_l254_254244


namespace calculate_S_l254_254602

def b (p : ℕ) : ℕ :=
  let k := (Real.sqrt p).toNat
  if abs (k - Real.sqrt p) < 1 / 3 then k else k + 1

def S : ℕ := ∑ p in Finset.range 3000, b (p + 1)

theorem calculate_S : S = 45173 := by
  sorry

end calculate_S_l254_254602


namespace smallest_number_bounds_l254_254032

theorem smallest_number_bounds (a : Fin 8 → ℝ)
    (h_sum : (∑ i, a i) = (4 / 3))
    (h_pos : ∀ i, (∑ j in finset.univ.filter (λ j, j ≠ i), a j) > 0) :
    -8 < (finset.univ.arg_min id a).get (finset.univ.nonempty) ∧
    (finset.univ.arg_min id a).get (finset.univ.nonempty) ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l254_254032


namespace chord_product_l254_254970

def ω := Complex.exp (2 * Real.pi * Complex.I / 10)

def A := (3 : Complex)
def B := (-3 : Complex)
def D (k : ℕ) := 3 * (ω ^ k)

def AD_length (k : ℕ) := Complex.abs (A - D k)
def BD_length (k : ℕ) := Complex.abs (B - D k)

theorem chord_product :
  (AD_length 1) * (AD_length 2) * (AD_length 3) * (AD_length 4) *
  (BD_length 1) * (BD_length 2) * (BD_length 3) * (BD_length 4) = 32805 :=
by
  sorry

end chord_product_l254_254970


namespace seq_value_at_101_l254_254940

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else (some_term_related_to_n) -- Placeholder if the sequence needs to be defined

theorem seq_value_at_101 :
  (∀ n : ℕ, 2 * a n + 1 = 2 * a n + 1) → a 101 = 52 := 
by
  intros h,
  sorry

end seq_value_at_101_l254_254940


namespace intersection_M_N_l254_254672

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254672


namespace vector_combination_l254_254371

theorem vector_combination
  {C D Q : Type}
  [AddCommGroup C] [Module ℝ C]
  [AddCommGroup D] [Module ℝ D]
  (h : ∃ (ratioC ratioD : ℝ), ratioC / ratioD = 3 / 5 ∧ Q = ratioC • C + ratioD • D) :
  ∃ (s v : ℝ), Q = s • C + v • D ∧ (s = 5 / 8 ∧ v = 3 / 8) :=
sorry

end vector_combination_l254_254371


namespace area_difference_correct_l254_254929

noncomputable def pi_estimated : ℝ := 3.14
def diameter_AB : ℝ := 6
def angle_abe : ℝ := 45 * (Real.pi / 180)  -- converting degrees to radians

def radius_O := diameter_AB / 2
def side_square : ℝ := radius_O * Real.sqrt 2
def area_square : ℝ := side_square * side_square
def area_circle : ℝ := pi_estimated * (radius_O * radius_O)
def area_difference : ℝ := area_circle - area_square

theorem area_difference_correct : area_difference = 10.26 := 
by
  sorry

end area_difference_correct_l254_254929


namespace third_place_prize_l254_254350

-- Definitions based on conditions
def num_people := 1 + 7  -- Josh and his 7 friends
def contribution_per_person := 5
def total_pot := num_people * contribution_per_person

def first_place_share_percentage := 0.8
def first_place_share := first_place_share_percentage * total_pot

def remaining_pot := total_pot - first_place_share
def second_third_split := remaining_pot / 2

-- Theorem stating the amount of money third place gets
theorem third_place_prize : second_third_split = 4 :=
by
  sorry

end third_place_prize_l254_254350


namespace intersection_M_N_l254_254726

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254726


namespace solve_math_problem_l254_254987

noncomputable def math_problem : Prop :=
  ∃ (ω α β : ℂ), (ω^5 = 1) ∧ (ω ≠ 1) ∧ (α = ω + ω^2) ∧ (β = ω^3 + ω^4) ∧
  (∀ x : ℂ, x^2 + x + 3 = 0 → x = α ∨ x = β) ∧ (α + β = -1) ∧ (α * β = 3)

theorem solve_math_problem : math_problem := sorry

end solve_math_problem_l254_254987


namespace solve_numRedBalls_l254_254299

-- Condition (1): There are a total of 10 balls in the bag
def totalBalls : ℕ := 10

-- Condition (2): The probability of drawing a black ball is 2/5
-- This means the number of black balls is 4
def numBlackBalls : ℕ := 4

-- Condition (3): The probability of drawing at least 1 white ball when drawing 2 balls is 7/9
def probAtLeastOneWhiteBall : ℚ := 7 / 9

-- The number of red balls in the bag is calculated based on the given conditions
def numRedBalls (totalBalls numBlackBalls : ℕ) (probAtLeastOneWhiteBall : ℚ) : ℕ := 
  let totalWhiteAndRedBalls := totalBalls - numBlackBalls
  let probTwoNonWhiteBalls := 1 - probAtLeastOneWhiteBall
  let comb (n k : ℕ) := Nat.choose n k
  let equation := comb totalWhiteAndRedBalls 2 * comb (totalBalls - 2) 0 / comb totalBalls 2
  if equation = probTwoNonWhiteBalls then totalWhiteAndRedBalls else 0

theorem solve_numRedBalls : numRedBalls totalBalls numBlackBalls probAtLeastOneWhiteBall = 1 := by
  sorry

end solve_numRedBalls_l254_254299


namespace f_always_positive_l254_254490

def f (x : ℝ) : ℝ := x^2 + 3 * x + 4

theorem f_always_positive : ∀ x : ℝ, f x > 0 := 
by 
  sorry

end f_always_positive_l254_254490


namespace general_formula_sum_of_first_n_terms_l254_254834

-- Definition of geometric sequence {a_n} with conditions a1 + a2 = 6 and a1 * a2 = a3
def geometric_seq (a : ℕ → ℝ) := (a 1 + a 2 = 6) ∧ (a 1 * a 2 = a 3)

-- Definition of arithmetic sequence {b_n} with condition S_{2n+1} = b_n * b_{n+1}
def arith_seq (b : ℕ → ℝ) (S : ℕ → ℝ) := (∀ n, S (2 * n + 1) = b n * b (n + 1))

-- General formula for sequence {a_n}
def a_n (n : ℕ) : ℝ := 2^n

-- Sum of first n terms T_n for the sequence {b_n / a_n}
def T_n (n : ℕ) : ℝ := 5 - (2 * n + 5) / (2^n)

-- Theorem to prove the general formula for the sequence {a_n}
theorem general_formula (a : ℕ → ℝ) (h : geometric_seq a) : ∀ n, a n = a_n n :=
begin
  sorry
end

-- Theorem to prove the sum of the first n terms T_n for the sequence {b_n / a_n}
theorem sum_of_first_n_terms (b : ℕ → ℝ) (S : ℕ → ℝ) (h : arith_seq b S) : ∀ n, 
  let c := λn, b n / a_n n in ∑ i in finset.range n, c i = T_n n :=
begin
  sorry
end

end general_formula_sum_of_first_n_terms_l254_254834


namespace problem1_problem2_l254_254119

-- First problem
theorem problem1 (x : ℝ) (h : x + 1/x = 3) : x^2 + 1/x^2 = 7 := 
by sorry

-- Second problem
theorem problem2 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hx : ∃ k, a^x = k ∧ b^y = k ∧ c^z = k) (h_sum : 1/x + 1/y + 1/z = 0) : a * b * c = 1 := 
by sorry

end problem1_problem2_l254_254119


namespace extremum_at_x_eq_neg3_l254_254016

theorem extremum_at_x_eq_neg3 (a : ℝ) :
  let f := λ x : ℝ, x^3 + a * x^2 + 3 * x - 9
  ∃ x : ℝ, f' x = 0 ∧ x = -3 → a = 5 :=
begin
  sorry
end

end extremum_at_x_eq_neg3_l254_254016


namespace inequality_proof_minimum_value_l254_254118

-- Proof statement for question (1)
theorem inequality_proof (a b x y : ℝ) (ht1 : a > 0) (ht2 : b > 0) (ht3 : x > 0) (ht4 : y > 0) :
  (a^2 / x) + (b^2 / y) ≥ ((a + b)^2 / (x + y)) ∧ (x / y = a / b) :=
by
  sorry

-- Proof statement for question (2)
theorem minimum_value (x_0 : ℝ) (h0 : 0 < x_0) (h1 : x_0 < 1/2) :
  let f : ℝ -> ℝ := λ x, 2 / x + 9 / (1 - 2 * x)
  in f x_0 = 25 ↔ x_0 = 1/5 :=
by
  sorry

end inequality_proof_minimum_value_l254_254118


namespace intersection_M_N_l254_254754

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254754


namespace intersection_correct_l254_254831

open Set

def M : Set ℤ := {-1, 3, 5}
def N : Set ℤ := {-1, 0, 1, 2, 3}
def MN_intersection : Set ℤ := {-1, 3}

theorem intersection_correct : M ∩ N = MN_intersection := by
  sorry

end intersection_correct_l254_254831


namespace final_computation_l254_254141

noncomputable def f (x : ℝ) : ℝ := (x - 2) * (x - 5) / 3
noncomputable def g (x : ℝ) : ℝ := -f(x)
noncomputable def k (x : ℝ) : ℝ := f(x - 2)

def c : ℕ := 2  -- Number of intersection points between f(x) and k(x)
def d : ℕ := 0  -- Number of intersection points between f(x) and g(x)

theorem final_computation : (d + 1) * c = 2 :=
by
  -- The proof is not provided and is left as an exercise.
  sorry

end final_computation_l254_254141


namespace intersection_M_N_l254_254666

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254666


namespace intersection_M_N_l254_254777

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254777


namespace sum_xyz_zero_l254_254841

theorem sum_xyz_zero 
  (x y z : ℝ)
  (h1 : x + y = 2 * x + z)
  (h2 : x - 2 * y = 4 * z)
  (h3 : y = 6 * z) : 
  x + y + z = 0 := by
  sorry

end sum_xyz_zero_l254_254841


namespace calculation_A_correct_l254_254097

theorem calculation_A_correct : (-1: ℝ)^4 * (-1: ℝ)^3 = 1 := by
  sorry

end calculation_A_correct_l254_254097


namespace Z_range_l254_254604

theorem Z_range (a b : ℝ) (h : a^2 + 3 * a * b + 9 * b^2 = 4) : 
  ∃ Z : ℝ, Z = a^2 + 9 * b^2 ∧ Z ∈ set.Icc (8 / 3) 8 :=
by
  sorry

end Z_range_l254_254604


namespace find_common_area_l254_254514

noncomputable def region_common_area (rect_width rect_height circle_radius : ℝ) : ℝ :=
  if ((rect_width / 2)^2 + (rect_height / 2)^2 ≤ circle_radius^2) 
  then rect_width * rect_height
  else 2 * (pi * circle_radius^2 / 4)

theorem find_common_area :
  region_common_area 10 (2 * Real.sqrt 3) 3 = (9 * pi / 2) :=
sorry

end find_common_area_l254_254514


namespace domain_of_reciprocal_minus_one_l254_254424

theorem domain_of_reciprocal_minus_one (x : ℝ) (h : x ≠ 1): ¬(y = 1 / (x - 1)) = ∃ x, x ≠ 1 ∧ x ∈ (-∞ : ℝ) ∪ (1 : ℝ) :=
by 
  sorry

end domain_of_reciprocal_minus_one_l254_254424


namespace value_of_fraction_l254_254058

theorem value_of_fraction : (20 + 15) / (30 - 25) = 7 := by
  sorry

end value_of_fraction_l254_254058


namespace infinitely_many_primes_divide_at_least_one_term_infinitely_many_primes_do_not_divide_any_term_l254_254377

-- Definition of the sequence {a_n}
def sequence (a : ℕ → ℕ) : Prop :=
  a 0 = 4 ∧ ∀ n : ℕ, a (n + 1) = (a n)^2 - a n

-- Part 1: Proving infinite primes that divide elements in the sequence
theorem infinitely_many_primes_divide_at_least_one_term (a : ℕ → ℕ) (h : sequence a) : 
  ∃ᶠ p in filter.at_top primes, ∃ n : ℕ, p ∣ a n := 
sorry

-- Part 2: Proving infinite primes that do not divide any term in the sequence
theorem infinitely_many_primes_do_not_divide_any_term (a : ℕ → ℕ) (h : sequence a) : 
  ∃ᶠ p in filter.at_top primes, ∀ n : ℕ, ¬ p ∣ a n :=
sorry

end infinitely_many_primes_divide_at_least_one_term_infinitely_many_primes_do_not_divide_any_term_l254_254377


namespace flower_combinations_l254_254522

theorem flower_combinations (t l : ℕ) (h : 4 * t + 3 * l = 60) : 
  ∃ (t_values : Finset ℕ), (∀ x ∈ t_values, 0 ≤ x ∧ x ≤ 15 ∧ x % 3 = 0) ∧
  t_values.card = 6 :=
sorry

end flower_combinations_l254_254522


namespace man_l254_254138

-- Given conditions
def V_m := 15 - 3.2
def V_c := 3.2
def man's_speed_with_current : Real := 15

-- Required to prove
def man's_speed_against_current := V_m - V_c

theorem man's_speed_against_current_is_correct : man's_speed_against_current = 8.6 := by
  sorry

end man_l254_254138


namespace pow_eq_of_81_27_l254_254278

theorem pow_eq_of_81_27 (y : ℝ) (h : 81^4 = 27^y) : 3^(-y) = (1 / 3^(16/3)) :=
by
  -- Here, one would transform the given conditions and work through the steps
  -- to arrive at the required conclusion, but this is left as a proof to be filled in.
  sorry

end pow_eq_of_81_27_l254_254278


namespace intersection_eq_l254_254694

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254694


namespace minimum_k_for_covering_and_separating_l254_254362

theorem minimum_k_for_covering_and_separating (n : ℕ) : 
  ∃ k : ℕ, (∀ (i j : ℕ), i ≠ j → (∃ A : set (set ℕ), (∀ x, x ∈ X → x ∈ ⋃₀ A) ∧ (∃ B ∈ A, (i ∈ B ∧ j ∉ B) ∨ (i ∉ B ∧ j ∈ B)))) → 
  k = Nat.ceil (Real.log2 (n + 1)) :=
sorry

end minimum_k_for_covering_and_separating_l254_254362


namespace danny_bottle_caps_l254_254572

variable (caps_found : Nat) (caps_existing : Nat)
variable (wrappers_found : Nat) (wrappers_existing : Nat)

theorem danny_bottle_caps:
  caps_found = 58 → caps_existing = 12 →
  wrappers_found = 25 → wrappers_existing = 11 →
  (caps_found + caps_existing) - (wrappers_found + wrappers_existing) = 34 := 
by
  intros h1 h2 h3 h4
  sorry

end danny_bottle_caps_l254_254572


namespace inequality_proof_l254_254395

theorem inequality_proof (a : ℝ) (h : a > 1) (n : ℕ) (hn : n ≥ 1) :
  (∑ i in range (n + 1), a^(2 * i)) / (∑ i in range n, a^(2 * i + 1)) > (n + 1) / n :=
sorry

end inequality_proof_l254_254395


namespace best_play_wins_probability_best_play_wins_with_certainty_l254_254077

-- Define the conditions

variables (n : ℕ)

-- Part (a): Probability that the best play wins
theorem best_play_wins_probability (hn_pos : 0 < n) : 
  1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) = 1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) :=
  by sorry

-- Part (b): With more than two plays, the best play wins with certainty
theorem best_play_wins_with_certainty (s : ℕ) (hs : 2 < s) : 
  1 = 1 :=
  by sorry

end best_play_wins_probability_best_play_wins_with_certainty_l254_254077


namespace transform_sine_half_horizontal_l254_254070

noncomputable def transformed_fn (x : ℝ) : ℝ :=
  sin (2 * x + π / 4)

theorem transform_sine_half_horizontal :
  ∀ x : ℝ, 
    let y := sin (x + π / 4) in
    transformed_fn (x / 2) = sin (x + π / 4) :=
by
  intros x y
  simp [transformed_fn]
  sorry

end transform_sine_half_horizontal_l254_254070


namespace smallest_number_bounds_l254_254031

theorem smallest_number_bounds (a : Fin 8 → ℝ)
    (h_sum : (∑ i, a i) = (4 / 3))
    (h_pos : ∀ i, (∑ j in finset.univ.filter (λ j, j ≠ i), a j) > 0) :
    -8 < (finset.univ.arg_min id a).get (finset.univ.nonempty) ∧
    (finset.univ.arg_min id a).get (finset.univ.nonempty) ≤ 1 / 6 :=
by
  sorry

end smallest_number_bounds_l254_254031


namespace distinct_tower_heights_l254_254064

theorem distinct_tower_heights :
  let total_bricks := 94 in
  let brick_height1 := 19 in
  let brick_height2 := 10 in
  let brick_height3 := 4 in
  number_of_distinct_heights total_bricks brick_height1 brick_height2 brick_height3 = 465 :=
sorry

end distinct_tower_heights_l254_254064


namespace subtraction_of_two_digit_numbers_from_set_is_61_l254_254208

/-- 
  Given the set of numbers {9, 4, 3, 5}, prove that the value of 
  subtracting the smallest two-digit number from the largest two-digit 
  number that can be formed by drawing two different numbers is 61.
-/
theorem subtraction_of_two_digit_numbers_from_set_is_61 :
  let s := {9, 4, 3, 5}
  let largest_two_digit_number := 95  -- maximum two-digit number that can be formed
  let smallest_two_digit_number := 34  -- minimum two-digit number that can be formed
  ∀ (a b : ℕ), a ∈ s → b ∈ s → a ≠ b → 
    (largest_two_digit_number - smallest_two_digit_number) = 61 :=
by {
  let s := {9, 4, 3, 5},
  let largest_two_digit_number := 95,
  let smallest_two_digit_number := 34,
  assume a b,
  assume ha : a ∈ s,
  assume hb : b ∈ s,
  assume hab : a ≠ b,
  sorry
}

end subtraction_of_two_digit_numbers_from_set_is_61_l254_254208


namespace intersection_points_count_l254_254563

open Real

theorem intersection_points_count :
  (∃ (x y : ℝ), ((x - ⌊x⌋)^2 + y^2 = x - ⌊x⌋) ∧ (y = 1/3 * x + 1)) →
  (∃ (n : ℕ), n = 8) :=
by
  -- proof goes here
  sorry

end intersection_points_count_l254_254563


namespace sequence_a_2016_l254_254266

-- Define the sequence a
def a : ℕ → ℤ
| 0     := 1
| 1     := 5
| (n+2) := a (n+1) - a n

-- State the main proof goal
theorem sequence_a_2016 : a 2015 = -4 :=
sorry

end sequence_a_2016_l254_254266


namespace work_efficiency_ratio_l254_254133

theorem work_efficiency_ratio (A B : ℝ) (k : ℝ)
  (h1 : A = k * B)
  (h2 : B = 1 / 27)
  (h3 : A + B = 1 / 9) :
  k = 2 :=
by
  sorry

end work_efficiency_ratio_l254_254133


namespace relationship_f_k_l254_254218

def f (n : ℕ) : ℕ :=
  (Finset.range (2 * n + 1)).sum (λ i, (i + 1) * (i + 1))

theorem relationship_f_k (k : ℕ) : f (k + 1) = f k + (2 * k + 1) ^ 2 + (2 * k + 2) ^ 2 :=
by {
  sorry
}

end relationship_f_k_l254_254218


namespace train_clicks_l254_254022

theorem train_clicks (y : ℝ) :
  let speed_mpm := (50 * y) / 3 in
  let clicks_per_minute := speed_mpm / 25 in
  let time_in_minutes := y / clicks_per_minute in
  let time_in_seconds := time_in_minutes * 60 in
  time_in_seconds = 90 :=
by
  hint
  sorry

end train_clicks_l254_254022


namespace intersection_M_N_l254_254673

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254673


namespace count_divisors_multiple_of_5_l254_254874

-- Define the conditions as Lean definitions
def prime_factorization (n : ℕ) := 
  n = 2^2 * 3^3 * 5^2

def is_divisor (d : ℕ) (n : ℕ) :=
  d ∣ n

def is_multiple_of_5 (d : ℕ) :=
  ∃ a b c, d = 2^a * 3^b * 5^c ∧ 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2

-- The theorem to be proven
theorem count_divisors_multiple_of_5 (h: prime_factorization 5400) : 
  {d : ℕ | is_divisor d 5400 ∧ is_multiple_of_5 d}.to_finset.card = 24 :=
by {
  sorry -- Proof goes here
}

end count_divisors_multiple_of_5_l254_254874


namespace variance_of_2X_plus_4_l254_254265

/-- Definition of the binomial random variable X. -/
def binomial_rv (n : ℕ) (p : ℝ) : ProbabilityMassFunction ℕ :=
  ProbabilityMassFunction.ofDiscreteUniform (Finset.range (n+1)).val
  sorry -- assume we have the construction details here

variables {X : ProbabilityMassFunction ℕ}
variable (hX : X = binomial_rv 6 0.5)

theorem variance_of_2X_plus_4 :
  variance (λ ω, 2 * X ω + 4) = 6 :=
by
  sorry

end variance_of_2X_plus_4_l254_254265


namespace intersection_M_N_l254_254756

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254756


namespace bella_truck_stamps_more_l254_254177

def num_of_truck_stamps (T R : ℕ) : Prop :=
  11 + T + R = 38 ∧ R = T - 13

theorem bella_truck_stamps_more (T R : ℕ) (h : num_of_truck_stamps T R) : T - 11 = 9 := sorry

end bella_truck_stamps_more_l254_254177


namespace num_towers_mod_1000_l254_254564

def T (n : ℕ) : ℕ :=
if n = 1 then 1 else 4 * T (n - 1)

theorem num_towers_mod_1000 (n : ℕ) (h_n : n = 10) : (T n) % 1000 = 144 := by
  sorry

end num_towers_mod_1000_l254_254564


namespace hydrocarbons_percentage_l254_254140

theorem hydrocarbons_percentage 
  (x : ℝ)                              -- percentage of hydrocarbons in the first source
  (source2_percent : ℝ)                -- percentage of hydrocarbons in the second source
  (source2_gallons : ℝ)                -- gallons of crude oil from the second source
  (total_gallons : ℝ)                  -- total gallons of crude oil needed
  (final_percent : ℝ)                  -- percentage of hydrocarbons in the final mixture
  (total_gallons = 50) 
  (source2_gallons = 30) 
  (source2_percent = 0.75) 
  (final_percent = 0.55) 
  : x = 25 :=
sorry

end hydrocarbons_percentage_l254_254140


namespace intersection_eq_l254_254704

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254704


namespace intersection_M_N_l254_254729

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254729


namespace equivalent_knicks_l254_254285

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end equivalent_knicks_l254_254285


namespace sum_of_integers_l254_254417

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 14) (h3 : x * y = 180) :
  x + y = 2 * Int.sqrt 229 :=
sorry

end sum_of_integers_l254_254417


namespace kims_morning_routine_total_time_l254_254354

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end kims_morning_routine_total_time_l254_254354


namespace coloring_the_circle_l254_254117

-- Part 1
def competition_medals_awarded (n m : ℕ) (h1 : n > 1) (h2 : m > 0) 
  (h3 : ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (k + (m - 1 - k) / 7 + 1) = m): 
  n = 6 ∧ m = 36 :=
begin
 sorry
end

-- Part 2
def ways_to_color_circle (n : ℕ) (h1 : n ≥ 2) : ℕ :=
2^n + (-1)^n * 2

theorem coloring_the_circle (n : ℕ) (h1 : n ≥ 2) :
  ways_to_color_circle n = 2^n + (-1)^n * 2 :=
begin
 sorry
end

end coloring_the_circle_l254_254117


namespace intersection_M_N_l254_254723

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254723


namespace intersection_M_N_l254_254663

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254663


namespace max_primes_in_9_consecutive_odds_l254_254162

theorem max_primes_in_9_consecutive_odds : 
  ∃ (s : Fin 9 → ℕ), 
    (∀ i j : Fin 9, i ≠ j → s i ≠ s j) ∧ 
    (∀ i : Fin 9, odd (s i) ∧ 0 < s i) ∧ 
    (∀ i j : Fin 9, i.succ = j → s j = s i + 2) →
    ∃ count : ℕ, count = 7 ∧ (∀ i : Fin 9, prime (s i) → count > 0) := 
sorry

end max_primes_in_9_consecutive_odds_l254_254162


namespace side_length_a_cosine_A_l254_254246

variable (A B C : Real)
variable (a b c : Real)
variable (triangle_inequality : a + b + c = 10)
variable (sine_equation : Real.sin B + Real.sin C = 4 * Real.sin A)
variable (bc_product : b * c = 16)

theorem side_length_a :
  a = 2 :=
  sorry

theorem cosine_A :
  b + c = 8 → 
  a = 2 → 
  b * c = 16 →
  Real.cos A = 7 / 8 :=
  sorry

end side_length_a_cosine_A_l254_254246


namespace triangle_inequality_proof_l254_254983

theorem triangle_inequality_proof (a b c : ℝ) (PA QA PB QB PC QC : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hpa : PA ≥ 0) (hqa : QA ≥ 0) (hpb : PB ≥ 0) (hqb : QB ≥ 0) 
  (hpc : PC ≥ 0) (hqc : QC ≥ 0):
  a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c := 
sorry

end triangle_inequality_proof_l254_254983


namespace part1_part2_l254_254610

-- Definitions and conditions of the problem.
variables {A B C H M N P D E F K X Y Z S T : Point}
variables {O : Circle}
variables {BC CA AB : Line}
variables {DE : Line}
variables {MP MN : Line}

-- Given conditions:
-- 1. Non-isosceles acute triangle ABC with circumcircle O and orthocenter H.
def non_isosceles_acute_triangle (A B C H : Point) (O : Circle) : Prop :=
  triangle_acute A B C ∧ ¬ (isosceles_triangle A B C) ∧ orthocenter A B C H ∧ circumcircle A B C O

-- 2. Midpoints of BC, CA, and AB are M, N, and P.
def midpoints_are_MNP (A B C M N P : Point) : Prop :=
  midpoint M B C ∧ midpoint N C A ∧ midpoint P A B

-- 3. Feet of altitudes from A, B, C are D, E, F.
def feet_of_altitudes (A B C D E F : Point) : Prop :=
  foot_of_altitude A B C D ∧ foot_of_altitude B A C E ∧ foot_of_altitude C A B F

-- 4. K is the reflection of H over BC.
def reflection_K (H K : Point) (BC : Line) : Prop :=
  reflection_over_line H K BC

-- 5. DE intersects MP at X.
def intersects_DE_MP (DE MP : Line) (X : Point) : Prop :=
  line_intersection DE MP X

-- 6. DF intersects MN at Y.
def intersects_DF_MN (DF MN : Line) (Y : Point) : Prop :=
  line_intersection DF MN Y

-- 7. XY intersects the minor arc of circumcircle O at Z.
def intersects_XY_minor_arc (XY : Line) (O : Circle) (Z : Point) : Prop :=
  arc_intersection XY O Z

-- 8. Second intersections of KE, KF with circumcircle O at S, T.
def second_intersections (K E F S T : Point) (O : Circle) : Prop :=
  second_intersection K E O S ∧ second_intersection K F O T

-- The proof statements
-- Part (1): Prove K, Z, E, F are concyclic.
theorem part1 (A B C H M N P D E F K X Y Z : Point) (O : Circle) (BC CA AB DE MP MN XY : Line)
  (h1 : non_isosceles_acute_triangle A B C H O)
  (h2 : midpoints_are_MNP A B C M N P)
  (h3 : feet_of_altitudes A B C D E F)
  (h4 : reflection_K H K BC)
  (h5 : intersects_DE_MP DE MP X)
  (h6 : intersects_DF_MN DF MN Y)
  (h7 : intersects_XY_minor_arc XY O Z) : concyclic K Z E F :=
sorry

-- Part (2): Prove BS, CT, and XY are concurrent.
theorem part2 (A B C H M N P D E F K X Y Z S T : Point) (O : Circle) (BC CA AB DE MP MN XY : Line)
  (h1 : non_isosceles_acute_triangle A B C H O)
  (h2 : midpoints_are_MNP A B C M N P)
  (h3 : feet_of_altitudes A B C D E F)
  (h4 : reflection_K H K BC)
  (h5 : intersects_DE_MP DE MP X)
  (h6 : intersects_DF_MN DF MN Y)
  (h7 : intersects_XY_minor_arc XY O Z)
  (h8 : second_intersections K E F S T O) : concurrent (line_through B S) (line_through C T) XY :=
sorry

end part1_part2_l254_254610


namespace roots_exist_for_all_b_l254_254283

theorem roots_exist_for_all_b : ∀ b : ℝ, ∃ x : ℝ, x^2 + b * x - 20 = 0 :=
by
  intros b
  have discriminant := b^2 + 80
  -- discriminant is non-negative for all real b
  have : discriminant ≥ 0 := by
    sorry
  obtain ⟨x, hx⟩ := exists_quadratic_root this
  exact ⟨x, hx⟩
  
-- helper lemma to state the quadratic formula root existence
lemma exists_quadratic_root (h : ∀ a b c : ℝ, b^2 - 4 * a * c ≥ 0) :
  ∀ a b c : ℝ, ∃ x : ℝ, a * x^2 + b * x + c = 0 :=
by
  intros a b c
  sorry

end roots_exist_for_all_b_l254_254283


namespace company_fund_initial_amount_l254_254434

theorem company_fund_initial_amount (n : ℕ) :
  (70 * n + 75 = 80 * n - 20) →
  (n = 9) →
  (80 * n - 20 = 700) :=
by
  intros h1 h2
  rw [h2] at h1
  linarith

end company_fund_initial_amount_l254_254434


namespace intersection_M_N_l254_254684

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254684


namespace ratio_greater_than_two_ninths_l254_254301

-- Define the conditions
def M : ℕ := 8
def N : ℕ := 36

-- State the theorem
theorem ratio_greater_than_two_ninths : (M : ℚ) / (N : ℚ) > 2 / 9 := 
by {
    -- skipping the proof with sorry
    sorry
}

end ratio_greater_than_two_ninths_l254_254301


namespace intersection_of_M_and_N_l254_254794

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254794


namespace computer_b_hourly_rate_l254_254105

noncomputable def hourly_charge_computer_B : ℝ :=
let
  CA := CB + 0.40 * CB,  -- cost per hour for Computer A
  A := (550 / CB) - 20   -- time for Computer A to do the job
in
  ∀ CB, (CB * (A + 20) = 550) → (1.40 * CB) * A = 550 → CB ≈ 7.86

theorem computer_b_hourly_rate :
  ∀ (CB : ℝ), (CB * ((550 / CB) - 20 + 20) = 550) ∧ (1.40 * CB) * ((550 / CB) - 20) = 550 → CB ≈ 7.86 :=
begin
  sorry
end

end computer_b_hourly_rate_l254_254105


namespace fraction_of_age_l254_254343

theorem fraction_of_age (jane_age_current : ℕ) (years_since_babysit : ℕ) (age_oldest_babysat_current : ℕ) :
  jane_age_current = 32 →
  years_since_babysit = 12 →
  age_oldest_babysat_current = 23 →
  ∃ (f : ℚ), f = 11 / 20 :=
by
  intros
  sorry

end fraction_of_age_l254_254343


namespace intersection_of_M_and_N_l254_254753

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254753


namespace total_skittles_l254_254405

-- Conditions
def groups : ℕ := 77
def skittles_per_group : ℕ := 77

-- Theorem: total number of Skittles in Steven's collection
theorem total_skittles (N : ℕ) (h1 : groups = 77) (h2 : skittles_per_group = 77) : 
  N = groups * skittles_per_group := 
begin
  sorry
end

-- Assert the final result with the calculated value
example : total_skittles 5929 := by
  sorry

end total_skittles_l254_254405


namespace line_circle_intersection_length_l254_254227

theorem line_circle_intersection_length :
  let l := λ (x y : ℝ), 12*x - 5*y - 3 = 0 in
  let circle := λ (x y : ℝ), (x - 3)^2 + (y - 4)^2 = 9 in
  let A B : ℝ × ℝ := sorry in -- Points of intersection
  dist A B = 4 * Real.sqrt 2 :=
sorry

end line_circle_intersection_length_l254_254227


namespace simplify_expression_l254_254094

theorem simplify_expression (w : ℤ) : 
  (-2 * w + 3 - 4 * w + 7 + 6 * w - 5 - 8 * w + 8) = (-8 * w + 13) :=
by {
  sorry
}

end simplify_expression_l254_254094


namespace impossible_score_53_l254_254313

def quizScoring (total_questions correct_answers incorrect_answers unanswered_questions score: ℤ) : Prop :=
  total_questions = 15 ∧
  correct_answers + incorrect_answers + unanswered_questions = 15 ∧
  score = 4 * correct_answers - incorrect_answers ∧
  unanswered_questions ≥ 0 ∧ correct_answers ≥ 0 ∧ incorrect_answers ≥ 0

theorem impossible_score_53 :
  ¬ ∃ (correct_answers incorrect_answers unanswered_questions : ℤ), quizScoring 15 correct_answers incorrect_answers unanswered_questions 53 := 
sorry

end impossible_score_53_l254_254313


namespace sequence_fourth_term_l254_254318

def sequence (n : ℕ) : ℕ :=
  Nat.recOn n 500 (λ _ an, an / 2 + 10)

theorem sequence_fourth_term : sequence 4 = 80 := by
  sorry

end sequence_fourth_term_l254_254318


namespace M_inter_N_eq_2_4_l254_254815

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254815


namespace part_I_part_II_l254_254852

noncomputable def f (x a : ℝ) := |x - 4| + |x - a|

theorem part_I (x : ℝ) : (f x 2 > 10) ↔ (x > 8 ∨ x < -2) :=
by sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≥ 1) ↔ (a ≥ 5 ∨ a ≤ 3) :=
by sorry

end part_I_part_II_l254_254852


namespace center_of_circle_l254_254519

theorem center_of_circle (
  center : ℝ × ℝ
) :
  (∀ (p : ℝ × ℝ), (p.1 * 3 + p.2 * 4 = 24) ∨ (p.1 * 3 + p.2 * 4 = -6) → (dist center p = dist center p)) ∧
  (center.1 * 3 - center.2 = 0)
  → center = (3 / 5, 9 / 5) :=
by
  sorry

end center_of_circle_l254_254519


namespace non_transit_cities_connected_l254_254307

-- Definitions to represent the problem setup
def kingdom (V : Type) := (V → V → Prop)  -- Represents roads as relations between cities

-- Conditions
variables {V : Type}
variables (vertices : Finset V)
variables (roads : V → V → Prop)
variables (n_transit n_nontransit : Finset V)
variables (H1 : vertices.card = 39)  -- 39 cities
variables (H2 : ∀ v, (roads v).card ≥ 21)  -- Each city has at least 21 one-way roads
variables (H3 : n_transit.card = 26)  -- 26 transit cities
variables (H4 : n_nontransit.card = 13)  -- 13 non-transit cities
variables (H5 : n_transit ∪ n_nontransit = vertices)  -- All cities are either transit or non-transit
variables (H6 : n_transit ∩ n_nontransit = ∅)  -- Disjoint sets of transit and non-transit cities
variables (H7 : ∀ a b ∈ n_transit, ¬roads a b)  -- Transit cities do not have direct return paths

-- Question to prove:
theorem non_transit_cities_connected :
  ∀ a b ∈ n_nontransit, roads a b :=
by
  sorry

end non_transit_cities_connected_l254_254307


namespace minimize_folded_area_l254_254615

-- defining the problem as statements in Lean
variables (a M N : ℝ) (M_on_AB : M > 0 ∧ M < a) (N_on_CD : N > 0 ∧ N < a)

-- main theorem statement
theorem minimize_folded_area :
  BM = 5 * a / 8 →
  CN = a / 8 →
  S = 3 * a ^ 2 / 8 := sorry

end minimize_folded_area_l254_254615


namespace brad_trips_to_fill_tank_l254_254178

noncomputable def bucket_volume : ℝ := (4 / 3) * Real.pi * (6:ℝ)^3 

noncomputable def tank_volume : ℝ := (1 / 3) * Real.pi * (8:ℝ)^2 * (20:ℝ)

def num_trips := (tank_volume / bucket_volume).ceil

theorem brad_trips_to_fill_tank : num_trips = 2 :=
by
  -- Calculations performed and validated in the problem steps.
  -- V_bucket = 288 * π and V_tank = 426.67 * π confirmed.
  sorry

end brad_trips_to_fill_tank_l254_254178


namespace probability_no_real_roots_l254_254197

theorem probability_no_real_roots :
  (let pairs := {p : ℤ × ℤ | abs p.1 ≤ 4 ∧ abs p.2 ≤ 4 ∧ p.2 ≠ 0};
       total_pairs := (9:ℕ) * (8:ℕ);
       valid_pairs := {p : ℤ × ℤ | abs p.1 ≤ 4 ∧ abs p.2 ≤ 4 ∧ p.2 ≠ 0 ∧ p.2 > p.1^2 / 4};
       num_valid_pairs := 32) in
  (num_valid_pairs : ℚ) / total_pairs = 4 / 9 :=
by
  -- Proof is omitted
  sorry

end probability_no_real_roots_l254_254197


namespace AM_GM_inequality_example_l254_254219

theorem AM_GM_inequality_example (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) : 
  (a + b) * (a + c) * (b + c) ≥ 8 * a * b * c :=
by
  sorry

end AM_GM_inequality_example_l254_254219


namespace original_perimeter_is_integer_l254_254530

-- Define the conditions
def is_perimeter_integer (width height : ℕ) : Prop :=
  ∃ p : ℕ, p = 2 * (width + height)

-- Define the proof problem
theorem original_perimeter_is_integer 
  (widths heights : Fin 7 → ℕ)
  (h : ∀ i j, is_perimeter_integer (widths i) (heights j)) :
  is_perimeter_integer (array.sum widths) (array.sum heights) :=
sorry

end original_perimeter_is_integer_l254_254530


namespace geometric_sequence_problem_l254_254221

variable {α : Type*} [LinearOrderedField α]

def geometric_sequence (a : ℕ → α) (q : α) : Prop :=
  ∀ n, a (n + 1) = a n * q

def sum_of_first_n_terms (a : ℕ → α) (n : ℕ) : α :=
  (Finset.range n).sum a

theorem geometric_sequence_problem
  (a : ℕ → α) (q : α)
  (h_geo : geometric_sequence a q)
  (h_a2 : a 2 = 2)
  (h_S3 : sum_of_first_n_terms a 3 = 7) :
  (∑ i in Finset.range 5, a i) / (a 3) = 31 / 4 := by
  sorry

end geometric_sequence_problem_l254_254221


namespace complement_of_A_in_U_l254_254268

noncomputable def U : Set ℤ := {-3, -1, 0, 1, 3}

noncomputable def A : Set ℤ := {x | x^2 - 2 * x - 3 = 0}

theorem complement_of_A_in_U : (U \ A) = {-3, 0, 1} :=
by sorry

end complement_of_A_in_U_l254_254268


namespace find_constants_C_D_l254_254598

theorem find_constants_C_D
  (C : ℚ) (D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 → (5 * x - 3) / (x^2 - 5 * x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32 / 9 ∧ D = 13 / 9 :=
by
  sorry

end find_constants_C_D_l254_254598


namespace sum_of_exterior_angles_of_triangle_l254_254098

theorem sum_of_exterior_angles_of_triangle (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
(h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b) : 
  (exterior_angle_sum : ℝ) := 
  360 :=
sorry

end sum_of_exterior_angles_of_triangle_l254_254098


namespace intersection_M_N_l254_254662

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254662


namespace tour_group_size_l254_254450

def adult_price : ℕ := 8
def child_price : ℕ := 3
def total_spent : ℕ := 44

theorem tour_group_size :
  ∃ (x y : ℕ), adult_price * x + child_price * y = total_spent ∧ (x + y = 8 ∨ x + y = 13) :=
by
  sorry

end tour_group_size_l254_254450


namespace circumradius_of_triangle_l254_254336

theorem circumradius_of_triangle
  (b : ℝ) (A : ℝ) (S : ℝ)
  (hb : b = 2)
  (hA : A = 120 * Real.pi / 180) -- converting degrees to radians
  (hS : S = Real.sqrt 3) :
  let c := (2 * S) / (b * Real.sin A),
      a := Real.sqrt (b^2 + c^2 - 2 * b * c * Real.cos A),
      R := a / (2 * Real.sin A) in
  R = 2 :=
by
  sorry

end circumradius_of_triangle_l254_254336


namespace sum_of_octal_numbers_l254_254206

theorem sum_of_octal_numbers : 521₈ + 146₈ = 667₈ :=
by sorry

end sum_of_octal_numbers_l254_254206


namespace M_inter_N_eq_2_4_l254_254828

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254828


namespace convert_base8_to_base7_l254_254571

theorem convert_base8_to_base7 (n : ℕ) (h : n = 5 * 8^2 + 3 * 8^1 + 7 * 8^0) : nat.to_digits 7 (5 * 8^2 + 3 * 8^1 + 7 * 8^0) = [1,1,0,1,1] := by
  sorry

end convert_base8_to_base7_l254_254571


namespace functional_equation_solution_l254_254365

theorem functional_equation_solution (α : ℝ) (hα : α ≠ 0) :
  (∀ f : ℝ → ℝ, (∀ x y : ℝ, f(f(x+y)) = f(x+y) + f(x) * f(y) + α * x * y) → (∀ x : ℝ, f(x) = x) ↔ α = -1) :=
by
  sorry

end functional_equation_solution_l254_254365


namespace problem1_problem2_problem3_problem4_l254_254558

theorem problem1 : 0.175 / 0.25 / 4 = 0.175 := by
  sorry

theorem problem2 : 1.4 * 99 + 1.4 = 140 := by 
  sorry

theorem problem3 : 3.6 / 4 - 1.2 * 6 = -6.3 := by
  sorry

theorem problem4 : (3.2 + 0.16) / 0.8 = 4.2 := by
  sorry

end problem1_problem2_problem3_problem4_l254_254558


namespace intersection_M_N_l254_254727

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254727


namespace weeds_in_rice_l254_254925

-- Define the conditions
def total_weight_of_rice := 1536
def sample_size := 224
def weeds_in_sample := 28

-- State the main proof
theorem weeds_in_rice (total_rice : ℕ) (sample_size : ℕ) (weeds_sample : ℕ) 
  (H1 : total_rice = total_weight_of_rice) (H2 : sample_size = sample_size) (H3 : weeds_sample = weeds_in_sample) :
  total_rice * weeds_sample / sample_size = 192 := 
by
  -- Evidence of calculations and external assumptions, translated initial assumptions into mathematical format
  sorry

end weeds_in_rice_l254_254925


namespace remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l254_254179

theorem remainder_of_x_pow_105_div_x_sq_sub_4x_add_3 :
  ∀ (x : ℤ), (x^105) % (x^2 - 4*x + 3) = (3^105 * (x-1) - (x-2)) / 2 :=
by sorry

end remainder_of_x_pow_105_div_x_sq_sub_4x_add_3_l254_254179


namespace inv_88_mod_89_l254_254588

theorem inv_88_mod_89 : (88 * 88) % 89 = 1 := by
  sorry

end inv_88_mod_89_l254_254588


namespace smallest_n_l254_254596

-- Define the polynomial expression
def expression (x y : ℕ) := x*y - 3*x + 7*y - 21

-- Define the condition for the number of unique terms in the expansion
def number_of_unique_terms (n : ℕ) := (n + 1) * (n + 1)

-- Define the proof statement
theorem smallest_n (n : ℕ) : number_of_unique_terms n >= 1996 ↔ n = 44 := by
  sorry

end smallest_n_l254_254596


namespace solve_inequality_l254_254372

noncomputable def f : ℝ → ℝ := sorry -- Define the function f, assume appropriate details based on the graph

axiom odd_function {f : ℝ → ℝ} (h : ∀ x, f (-x) = -f x): true

theorem solve_inequality : 
  ∀ x, x ∈ Icc (-5:ℝ) 5 → f x < 0 ↔ x ∈ Ioo (-2:ℝ) 0 ∪ Ico (2:ℝ) 5 :=
by
  sorry

end solve_inequality_l254_254372


namespace intersection_M_N_l254_254718

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254718


namespace intersection_M_N_l254_254711

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254711


namespace intersection_M_N_l254_254731

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254731


namespace constant_term_expansion_l254_254242

theorem constant_term_expansion 
  (n : ℕ) 
  (h_n : n = 8) 
  (h_condition : ∀ k : ℕ, (k ≠ 4 → binomial n (k+1) < binomial n 4) ∧ (k = 4 → binomial n (k+1) = binomial n 4)) :
  let r := 6 in
  ((x - (1 / x^(1/3)))^n).coeff 8 = 28 :=
sorry

end constant_term_expansion_l254_254242


namespace sum_paintable_numbers_l254_254565

-- Definitions based on conditions
def is_paintable (h t u : ℕ) : Prop :=
  (∀ n : ℕ, 1 ≤ n → n ≠ h → n ≠ t → n ≠ u → (n - 1) % h = 0 ∨ (n - 2) % t = 0 ∨ (n - 4) % u = 0)
  ∧ 1 < h ∧ 1 < t ∧ 1 < u
  ∧ h ≠ 2

-- The statement to prove
theorem sum_paintable_numbers : 
  (∑ h t u in (finset.filter is_paintable 
                         (finset.range 10).product (finset.range 10).product (finset.range 10)), 
                         100 * h + 10 * t + u) = 767 := 
by 
  sorry

end sum_paintable_numbers_l254_254565


namespace soccer_team_selection_l254_254151

-- Definitions of the problem
def total_members := 16
def utility_exclusion_cond := total_members - 1

-- Lean statement for the proof problem, using the conditions and answer:
theorem soccer_team_selection :
  (utility_exclusion_cond) * (total_members - 1) * (total_members - 2) * (total_members - 3) * (total_members - 4) = 409500 :=
by
  sorry

end soccer_team_selection_l254_254151


namespace maximum_numbers_l254_254479

theorem maximum_numbers (S : Finset ℕ) (h1 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 1000)
                       (h2 : ∀ x y ∈ S, x ≠ y → (x - y).natAbs ≠ 4 ∧ (x - y).natAbs ≠ 5 ∧ (x - y).natAbs ≠ 6) :
  S.card ≤ 400 :=
begin
  sorry
end

end maximum_numbers_l254_254479


namespace right_triangle_area_l254_254534

theorem right_triangle_area :
  ∃ (a b c : ℕ), (c^2 = a^2 + b^2) ∧ (2 * b^2 - 23 * b + 11 = 0) ∧ (a * b / 2 = 330) :=
sorry

end right_triangle_area_l254_254534


namespace find_F_l254_254279

theorem find_F (C : ℝ) (F : ℝ) (h₁ : C = 35) (h₂ : C = 4 / 7 * (F - 40)) : F = 101.25 := by
  sorry

end find_F_l254_254279


namespace counterexample_exists_l254_254568

theorem counterexample_exists : ∃ n : ℕ, n ≥ 2 ∧ ¬ ∃ k : ℕ, 2 ^ 2 ^ n % (2 ^ n - 1) = 4 ^ k := 
by
  sorry

end counterexample_exists_l254_254568


namespace volume_of_tetrahedron_l254_254923

-- Definitions for the problem conditions
def length_PQ := 5 -- cm
def area_PQR := 20 -- cm^2
def area_PQS := 18 -- cm^2
def angle_PQR_PQS := 60 -- degrees

-- The goal is to prove the volume of the tetrahedron
theorem volume_of_tetrahedron :
  let volume := (1 / 3) * area_PQR * (7.2 * (Mathlib.Real.Basic.sin (Mathlib.Real.Basic.ofRat (angle_PQR_PQS * (Mathlib.Real.Basic.pi / 180)))))
  in volume = 41.6 :=
by sorry

end volume_of_tetrahedron_l254_254923


namespace M_inter_N_eq_2_4_l254_254818

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254818


namespace sum_x_coords_Q3_eq_176_l254_254516

def sum_x_coords (x_coords : List ℝ) : ℝ :=
List.sum x_coords

theorem sum_x_coords_Q3_eq_176 (x_coords : List ℝ) 
  (h_len : x_coords.length = 44) 
  (h_sum : sum_x_coords x_coords = 176) : 
  sum_x_coords (List.map (λ i, (x_coords[(i % 44)] + x_coords[((i + 1) % 44)]) / 2) (List.range 44)) = 176 :=
  sorry

end sum_x_coords_Q3_eq_176_l254_254516


namespace gcd_7920_14553_l254_254591

theorem gcd_7920_14553 : Int.gcd 7920 14553 = 11 := by
  sorry

end gcd_7920_14553_l254_254591


namespace polygon_sidedness_l254_254449

-- Define the condition: the sum of the interior angles of the polygon
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

-- Given condition
def given_condition : ℝ := 1260

-- Target proposition to prove
theorem polygon_sidedness (n : ℕ) (h : sum_of_interior_angles n = given_condition) : n = 9 :=
by
  sorry

end polygon_sidedness_l254_254449


namespace parabola_line_intersection_line_eq_when_max_distance_l254_254856

theorem parabola_line_intersection (λ : ℝ) (x1 x2 y1 y2 : ℝ) 
  (A : ℝ × ℝ := (-1,0))
  (P : ℝ × ℝ := (x1, y1))
  (Q : ℝ × ℝ := (x2, y2))
  (h1 : y1^2 = 4*x1) -- Condition from the parabola equation
  (h2 : y2^2 = 4*x2) -- Condition from the parabola equation
  (h3 : x1 + 1 = λ * (x2 + 1)) -- Given condition on coordinates
  (h4 : y1 = λ * y2) -- Given condition on coordinates
  : x1 = λ ∧ x2 = 1/λ :=
sorry

theorem line_eq_when_max_distance (λ : ℝ) (x1 x2 y1 y2 : ℝ) 
  (A : ℝ × ℝ := (-1,0))
  (P : ℝ × ℝ := (x1, y1))
  (Q : ℝ × ℝ := (x2, y2))
  (h1 : y1^2 = 4*x1) -- Condition from the parabola equation
  (h2 : y2^2 = 4*x2) -- Condition from the parabola equation
  (h3 : x1 + 1 = λ * (x2 + 1)) -- Given condition on coordinates
  (h4 : y1 = λ * y2) -- Given condition on coordinates
  (hλ : λ ∈ set.Icc (1/3 : ℝ) (1/2 : ℝ)) -- λ in the given interval
  (hx1 : x1 = λ) -- From previous theorem/application
  (hx2 : x2 = 1/λ) -- From previous theorem/application
  : ∃ m b : ℝ, (m = sqrt 3 ∧ b = sqrt 3 ∧ (∀ x y : ℝ, x*x - 4*x + y*y = 0 ↔ x + sqrt 3 * y + sqrt 3 = 0)) :=
sorry

end parabola_line_intersection_line_eq_when_max_distance_l254_254856


namespace intersection_of_M_and_N_l254_254747

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254747


namespace smallest_a1_range_l254_254053

noncomputable def smallest_a1_possible (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = (4 : ℝ) / (3 : ℝ) ∧ 
  (∀ i : Fin₈, (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - a_i) > 0) ∧ 
  (-8 < a_1 ∧ a_1 ≤ 1 / 6)

-- Since we are only required to state the problem, we leave the proof as a "sorry".
theorem smallest_a1_range (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
    (h_sum: a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 4 / 3)
    (h_pos: ∀ i : Fin 8, a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - ([-][i]) > 0):
    smallest_a1_possible a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 :=
  by
  sorry

end smallest_a1_range_l254_254053


namespace terry_mary_same_combination_l254_254135

noncomputable def probability_same_combination : ℚ :=
  let total_candies := 12 + 8
  let terry_red := (12.choose 2) / (total_candies.choose 2) 
  let remaining_red := 12 - 2
  let remaining_total := total_candies - 2
  let mary_red := (remaining_red.choose 2) / (remaining_total.choose 2)
  let both_red := terry_red * mary_red

  let terry_diff := (12.choose 1 * 8.choose 1) / (total_candies.choose 2)
  let mary_diff_red := (11.choose 1 * 7.choose 1) / (remaining_total.choose 2)
  let both_diff := terry_diff * mary_diff_red
  
  let combined := 2 * both_red + both_diff
  combined

theorem terry_mary_same_combination:
  probability_same_combination = (143 / 269) := by
  sorry

end terry_mary_same_combination_l254_254135


namespace intersection_M_N_l254_254710

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254710


namespace sandy_puppies_count_l254_254400

theorem sandy_puppies_count :
  ∃ n : ℕ,
    n = 9 ∧
    (∃ s₁ ns₁ s₂ ns₂ : ℕ,
      s₁ = 3 ∧ ns₁ = 5 ∧ s₂ = 2 ∧ ns₂ = 2 ∧
      (∃ ns₃ : ℕ, ns₃ = 3 ∧
        ns₁ + ns₂ - ns₃ + s₁ + s₂ = n
      )
    ) :=
begin
  sorry
end

end sandy_puppies_count_l254_254400


namespace plane_intersects_at_least_four_l254_254538

-- Define the space and the sets
variable (X : Type) [plane : ProjectivePlane X]
variable (A B C D E : set X)
variable (h_disjoint : ∀ (S₁ S₂ : set X), S₁ ≠ S₂ → S₁ ∩ S₂ = ∅)
variable (h_nonemptyA : A.nonempty)
variable (h_nonemptyB : B.nonempty)
variable (h_nonemptyC : C.nonempty)
variable (h_nonemptyD : D.nonempty)
variable (h_nonemptyE : E.nonempty)

-- Statement to prove:
theorem plane_intersects_at_least_four :
  ∃ p : Plane X, (p ∩ A).nonempty ∧ (p ∩ B).nonempty ∧ (p ∩ C).nonempty ∧ (p ∩ D).nonempty ∨ (p ∩ E).nonempty := sorry

end plane_intersects_at_least_four_l254_254538


namespace intersection_M_N_l254_254734

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254734


namespace find_divisor_l254_254111

theorem find_divisor (d : ℕ) (quotient : ℕ) (remainder : ℕ) 
  (h1 : 52 = d * quotient + remainder) (h2 : quotient = 16) (h3 : remainder = 4) : 
  d = 3 :=
by 
  -- declare the assumptions
  have h1' : 52 = d * 16 + 4 := by {
    rw [h2, h3] at h1,
    exact h1,
  },
  -- solve the equation 52 = d * 16 + 4
  have h2' : 52 - 4 = d * 16 := by {
    linarith,
  },
  -- simplify 48 = d * 16
  have h3' : 48 = d * 16 := by {
    exact h2',
  },
  -- solve for d
  exact (eq_of_mul_eq_mul_right (nat.zero_lt_succ 15) h3')

end find_divisor_l254_254111


namespace intersection_M_N_l254_254693

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254693


namespace count_arithmetic_sequences_l254_254163

def is_arithmetic_sequence (seq : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, seq (n + 1) - seq n = d

def seq1 : ℕ → ℤ := λ n, 6
def seq2 : ℕ → ℤ := λ n, n - 3
def seq3 : ℕ → ℤ := λ n, 3 * n + 2
def seq4 : ℕ → ℤ := λ n, (n^2 - n) / 2

theorem count_arithmetic_sequences :
  let sequences := [seq1, seq2, seq3, seq4] in
  list.countp is_arithmetic_sequence sequences = 3 :=
by sorry

end count_arithmetic_sequences_l254_254163


namespace intersection_of_M_and_N_l254_254791

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254791


namespace flower_nectar_water_content_l254_254486

/-- Given that to yield 1 kg of honey, 1.6 kg of flower-nectar must be processed,
    and the honey obtained from this nectar contains 20% water,
    prove that the flower-nectar contains 50% water. --/
theorem flower_nectar_water_content :
  (1.6 : ℝ) * (0.2 / 1) = (50 / 100) * (1.6 : ℝ) := by
  sorry

end flower_nectar_water_content_l254_254486


namespace minimum_value_225_l254_254989

noncomputable def min_value_inverse_sum (b : Fin 15 → ℝ) : ℝ := 
  ∑ i, 1 / b i

theorem minimum_value_225 (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i) (h_sum : ∑ i, b i = 1) :
  min_value_inverse_sum b ≥ 225 :=
  sorry

end minimum_value_225_l254_254989


namespace prime_addition_equality_l254_254109

def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

theorem prime_addition_equality (x y : ℕ)
  (hx : is_prime x)
  (hy : is_prime y)
  (hxy : x < y)
  (hsum : x + y = 36) : 4 * x + y = 51 :=
sorry

end prime_addition_equality_l254_254109


namespace triangle_isosceles_right_circumcircle_equation_line_equation_l254_254223

-- Definitions and conditions
def point (ℝ : Type) := (ℝ, ℝ)
def A : point ℝ := (1, 0)
def B : point ℝ := (1, 4)
def C : point ℝ := (3, 2)

-- Problems

-- 1. Prove that triangle ABC is an isosceles right triangle.
theorem triangle_isosceles_right : 
  is_triangle A B C ∧ right_angle (slope AC) (slope BC) ∧ distance A C = distance B C :=
sorry

-- 2. Find the equation of the circumcircle of △ABC.
theorem circumcircle_equation :
  circumcenter A B C = (1, 2) ∧ circumradius A B C = 2 ∧ equation_circumcircle (1, 2) 2 = "x-1)^2 + (y-2)^2 = 4" :=
sorry

-- 3. Given line l passing through (0, 4), find the equation of line l if it intersects the circumcircle forming a chord of length 2√3.
theorem line_equation :
  (passes_through_line (0, 4) l) ∧ (chord_length_intersect_circumcircle l (1, 2) 2 = 2√3) ∧ 
    (equation_line l = "x=0" ∨ equation_line l = "3x + 4y - 16 = 0") :=
sorry

end triangle_isosceles_right_circumcircle_equation_line_equation_l254_254223


namespace expand_and_simplify_product_l254_254585

variable (x : ℝ)

theorem expand_and_simplify_product :
  (x^2 + 3*x - 4) * (x^2 - 5*x + 6) = x^4 - 2*x^3 - 13*x^2 + 38*x - 24 :=
by
  sorry

end expand_and_simplify_product_l254_254585


namespace remainder_2011_2015_mod_17_l254_254090

theorem remainder_2011_2015_mod_17 :
  ((2011 * 2012 * 2013 * 2014 * 2015) % 17) = 7 :=
by
  have h1 : 2011 % 17 = 5 := by sorry
  have h2 : 2012 % 17 = 6 := by sorry
  have h3 : 2013 % 17 = 7 := by sorry
  have h4 : 2014 % 17 = 8 := by sorry
  have h5 : 2015 % 17 = 9 := by sorry
  sorry

end remainder_2011_2015_mod_17_l254_254090


namespace intersection_of_M_and_N_l254_254647

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254647


namespace vector_decomposition_l254_254115

theorem vector_decomposition :
  let x : ℝ × ℝ × ℝ := (6, -1, 7)
  let p : ℝ × ℝ × ℝ := (1, -2, 0)
  let q : ℝ × ℝ × ℝ := (-1, 1, 3)
  let r : ℝ × ℝ × ℝ := (1, 0, 4)
  x = (-p.1 - 3 * q.1 + 4 * r.1, -p.2 - 3 * q.2 + 4 * r.2, -p.3 - 3 * q.3 + 4 * r.3) := 
by
  sorry

end vector_decomposition_l254_254115


namespace area_decreased_by_1_percent_l254_254910

variable (L B : ℝ)

def L_new := 1.10 * L
def B_new := 0.90 * B

def A_original := L * B

theorem area_decreased_by_1_percent : (L_new * B_new) = 0.99 * A_original :=
by
  let A_new := L_new * B_new
  have h1 : A_new = (1.10 * L) * (0.90 * B) := rfl
  rw [h1]
  have h2 : A_new = 1.10 * 0.90 * (L * B) := by ring
  rw [h2]
  have h3 : A_new = 0.99 * A_original := by norm_num
  exact h3

end area_decreased_by_1_percent_l254_254910


namespace largest_expression_l254_254898

noncomputable def a : ℝ := 10^(-2010)

theorem largest_expression : 
  (3 + a < 3 / a) ∧ (3 - a < 3 / a) ∧ (3 * a < 3 / a) ∧ (a / 3 < 3 / a) :=
by
  sorry

end largest_expression_l254_254898


namespace probability_john_dave_paired_l254_254919

theorem probability_john_dave_paired:
  (probability_john_paired_with_dave (participants 24) (random_pairing)) = (1 / 23) := 
sorry

-- Definitions for conditions in a)

def participants := 24

def random_pairing : 24 → (24 → bool) := λ n m, 
  if n ≠ m then true else false

-- Probability calculation

def probability_john_paired_with_dave (n : ℕ) (pairing : ℕ → (ℕ → bool)) : ℚ :=
  if n = 24 then 1 / 23 else 0


end probability_john_dave_paired_l254_254919


namespace other_number_is_25_l254_254004

theorem other_number_is_25 (x y : ℤ) (h1 : 3 * x + 4 * y + 2 * x = 160) (h2 : x = 12 ∨ y = 12) : y = 25 :=
by
  cases h2 with
  | inl hx =>
    rw [hx] at h1
    linarith
  | inr hy =>
    have hy_invalid: x = 112 / 5 := by linarith
    contradiction

end other_number_is_25_l254_254004


namespace intersection_M_N_l254_254661

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254661


namespace intersection_M_N_l254_254651

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254651


namespace tangent_line_f_at_one_l254_254607

noncomputable def g : ℝ → ℝ := sorry
noncomputable def f (x : ℝ) : ℝ := g (2 * x - 1) + x^2

axiom tangent_g_at_one : ∀ x, g 1 = 3 ∧ (deriv g 1 = 2)

theorem tangent_line_f_at_one : 
  let t : ℝ × ℝ := (1, f 1)
  in ∀ x y : ℝ, y - t.2 = (deriv f 1) * (x - t.1) ↔ 6 * x - y - 2 = 0 :=
begin
  sorry
end

end tangent_line_f_at_one_l254_254607


namespace max_volume_prism_l254_254916

theorem max_volume_prism (a b h : ℝ) (V : ℝ) 
  (h1 : a * h + b * h + a * b = 32) : 
  V = a * b * h → V ≤ 128 * Real.sqrt 3 / 3 := 
by
  sorry

end max_volume_prism_l254_254916


namespace find_largest_smallest_A_l254_254168

theorem find_largest_smallest_A :
  ∃ (B : ℕ), B > 7777777 ∧ B.gcd 36 = 1 ∧
  (let A := 10^7 * (B % 10) + (B / 10) in A = 99999998 ∨ A = 17777779) :=
begin
  sorry
end

end find_largest_smallest_A_l254_254168


namespace find_valid_A_l254_254167

-- Define condition that B is coprime with 36
def coprime_with_36 (b : ℕ) : Prop :=
  Nat.coprime b 36

-- Define condition that moving the last digit to the first place forms A
def derived_A (b : ℕ) : ℕ :=
  let last_digit := b % 10
  let remaining := b / 10
  10^7 * last_digit + remaining

-- Define the range of B based on the problem statement
def valid_range (b : ℕ) : Prop :=
  b > 7777777 ∧ b < 10^8

-- Define the smallest and largest valid A
def smallest_valid_A (a : ℕ) : Prop :=
  a = 17777779

def largest_valid_A (a : ℕ) : Prop :=
  a = 99999998

-- Proof goal statement
theorem find_valid_A (b : ℕ) (h1 : coprime_with_36 b) (h2 : valid_range b) :
  smallest_valid_A (derived_A b) ∨ largest_valid_A (derived_A b) :=
sorry

end find_valid_A_l254_254167


namespace simplify_t_l254_254836

noncomputable def t : ℝ := 1 / (1 - real.rpow 2 (1/4 : ℝ))

theorem simplify_t : 
  t = -(1 + real.rpow 2 (1/4 : ℝ)) * (1 + real.sqrt 2) := 
by
-- Proof is omitted
sorry

end simplify_t_l254_254836


namespace proof_problem_l254_254287

noncomputable def problem_statement (x y z : ℝ) (n : ℕ) : Prop :=
  x + y + z = 1 ∧ arctan x + arctan y + arctan z = π / 4 → x^(2*n+1) + y^(2*n+1) + z^(2*n+1) = 1

theorem proof_problem :
  ∀ (x y z : ℝ) (n : ℕ), 0 < n → problem_statement x y z n :=
by
  intros x y z n hn
  sorry

end proof_problem_l254_254287


namespace intersection_of_M_and_N_l254_254639

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254639


namespace circle_equation_solution_l254_254120

theorem circle_equation_solution (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 - 2 * m * x - 2 * m * y + 2 * m^2 + m - 1 = 0) ↔ m < 1 :=
sorry

end circle_equation_solution_l254_254120


namespace estimate_excellent_students_l254_254357

-- Define conditions
def total_students : ℕ := 1500
def sample_size : ℕ := 200
def excellent_students_in_sample : ℕ := 60

-- Define the statement to prove
theorem estimate_excellent_students : 
  (excellent_students_in_sample.toRat / sample_size.toRat) * total_students.toRat = 450 := 
by 
  sorry

end estimate_excellent_students_l254_254357


namespace min_value_of_f1_f2_decreasing_sum_x1_x2_bound_l254_254849

-- Definitions
def f1 (x : ℝ) := - Real.log x + x
def f2 (a b x: ℝ) := a * Real.log x + b * x^2 + x
def f3 (x : ℝ) := Real.log x + x^2 + x

-- Conditions
def condition_2_f1 (h1 : f2 a b 1 = 0) (h2 : (deriv (f2 a b)) 1 = 0) (x : ℝ) : Prop :=
  a = 1 ∧ b = -1 
def condition_3 (x1 x2 : ℝ) (h : f3 x1 + f3 x2 + x1 * x2 = 0) : Prop :=
  x1 > 0 ∧ x2 > 0

-- Theorems to be proven
theorem min_value_of_f1 (x : ℝ) (h : 0 < x) : f1 x ≥ 1 :=
sorry

theorem f2_decreasing (h1 : f2 a b 1 = 0) (h2 : (deriv (f2 a b)) 1 = 0) (x : ℝ) : x > 1 → (deriv (f2 a b)) x < 0 :=
sorry

theorem sum_x1_x2_bound (x1 x2 : ℝ) (h : condition_3 x1 x2 h) : x1 + x2 ≥ (Real.sqrt 5 - 1) / 2 :=
sorry

end min_value_of_f1_f2_decreasing_sum_x1_x2_bound_l254_254849


namespace MeatMarket_sales_l254_254388

theorem MeatMarket_sales :
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  total_sales - planned_sales = 325 :=
by
  let thursday_sales := 210
  let friday_sales := 2 * thursday_sales
  let saturday_sales := 130
  let sunday_sales := saturday_sales / 2
  let total_sales := thursday_sales + friday_sales + saturday_sales + sunday_sales
  let planned_sales := 500
  show total_sales - planned_sales = 325
  sorry

end MeatMarket_sales_l254_254388


namespace intersection_of_M_and_N_l254_254641

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254641


namespace tank_fill_time_l254_254392

theorem tank_fill_time (R1 R2 t_required : ℝ) (hR1: R1 = 1 / 8) (hR2: R2 = 1 / 12) (hT : t_required = 4.8) :
  t_required = 1 / (R1 + R2) :=
by 
  -- Proof goes here
  sorry

end tank_fill_time_l254_254392


namespace squares_equation_example_l254_254587

theorem squares_equation_example :
  ∃ (a b c d e : ℕ), a + b = c * (d - e) ∧ 
                     {a, b, c, d, e} = {1, 2, 3, 4, 5} :=
by
  use [1, 2, 3, 5, 4]
  simp
  sorry

end squares_equation_example_l254_254587


namespace intersection_M_N_l254_254767

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254767


namespace game_win_l254_254984

-- Definitions to encapsulate the conditions
def positive_integer (n : ℕ) := n > 0

def valid_necklace (n : ℕ) (necklace : list char) :=
  length necklace = 6 * n ∧
  (∀ i, necklace.nth i = some 'S' → necklace.nth (i + 1) = some 'T' ∨ necklace.nth (i + 2) = some 'T') ∧
  (∀ i, necklace.nth i = some 'T' → necklace.nth (i + 1) = some 'S' ∨ necklace.nth (i + 2) = some 'S')

def TST_turn (necklace : list char) : Prop :=
  ∃ i, (necklace.nth i = some 'T' ∧ necklace.nth (i + 1) = some 'S' ∧ necklace.nth (i + 2) = some 'T')

def STS_turn (necklace : list char) : Prop :=
  ∃ i, (necklace.nth i = some 'S' ∧ necklace.nth (i + 1) = some 'T' ∧ necklace.nth (i + 2) = some 'S')

theorem game_win (n : ℕ) (necklace : list char)
  (h_pos : positive_integer n)
  (h_valid : valid_necklace n necklace)
  (h_TASTY_first_win : ∀ necklace,
    valid_necklace n necklace →
    ∀ t ≤ 2 * n, t = 2 * n → TST_turn necklace → ∃ necklace', valid_necklace n necklace' ∧ length necklace' = 0 ) :
  ∀ necklace, 
    valid_necklace n necklace →
    ∀ t ≤ 2 * n, t = 2 * n → STS_turn necklace → ∃ necklace', valid_necklace n necklace' ∧ length necklace' = 0 :=
sorry

end game_win_l254_254984


namespace tiles_difference_ninth_eighth_rectangle_l254_254149

theorem tiles_difference_ninth_eighth_rectangle : 
  let width (n : Nat) := 2 * n
  let height (n : Nat) := n
  let tiles (n : Nat) := width n * height n
  tiles 9 - tiles 8 = 34 :=
by
  intro width height tiles
  sorry

end tiles_difference_ninth_eighth_rectangle_l254_254149


namespace minimum_jumps_required_l254_254130

-- Define the conditions and the frog's movement as hypotheses
def valid_jump (p q : (ℤ × ℤ)) : Prop :=
  let ⟨x1, y1⟩ := p
  let ⟨x2, y2⟩ := q
  (x2 - x1)^2 + (y2 - y1)^2 = 25

def starts_at_origin (p : (ℤ × ℤ)) : Prop :=
  p = (0,0)

def ends_at_target (p : (ℤ × ℤ)) : Prop :=
  p = (1,0)

-- Main theorem stating the minimum number of jumps
theorem minimum_jumps_required :
  ∃ (jumps : List (ℤ × ℤ)), 
    List.length jumps = 3 ∧
    starts_at_origin (jumps.head) ∧
    ends_at_target (jumps.last) ∧
    All (valid_jump (jumps.nth)) : sorry

end minimum_jumps_required_l254_254130


namespace best_play_wins_probability_l254_254080

theorem best_play_wins_probability (n : ℕ) :
  let p := (n! * n!) / (2 * n)! in
  1 - p = 1 - (fact n * fact n / fact (2 * n)) :=
sorry

end best_play_wins_probability_l254_254080


namespace parabola_vertex_l254_254447

theorem parabola_vertex (c d : ℝ) (s : Set ℝ)
  (h₁ : s = { x : ℝ | (-∞ ≤ x ∧ x ≤ -5) ∨ (7 ≤ x ∧ x < ∞) })
  (h₂ : ∀ x, x ∈ s ↔ -x^2 + c * x + d ≤ 0) :
  ∃ p : ℝ × ℝ, p = (1, -34) ∧ ∃ y, y = -x^2 + c * x + d :=
sorry

end parabola_vertex_l254_254447


namespace exists_plane_with_congruent_projections_l254_254269

-- Definitions for the conditions
variables (A1 A2 A3 B1 B2 B3 : Point)
variable (congr : Congruent A1 A2 A3 B1 B2 B3)

-- The main theorem statement
theorem exists_plane_with_congruent_projections :
  ∃ (π : Plane), OrthogonalProjectionsCongruentAndOriented A1 A2 A3 B1 B2 B3 π :=
sorry

end exists_plane_with_congruent_projections_l254_254269


namespace Albert_more_rocks_than_Joshua_l254_254949

-- Definitions based on the conditions
def Joshua_rocks : ℕ := 80
def Jose_rocks : ℕ := Joshua_rocks - 14
def Albert_rocks : ℕ := Jose_rocks + 20

-- Statement to prove
theorem Albert_more_rocks_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_rocks_than_Joshua_l254_254949


namespace smallest_a1_range_l254_254054

noncomputable def smallest_a1_possible (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = (4 : ℝ) / (3 : ℝ) ∧ 
  (∀ i : Fin₈, (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - a_i) > 0) ∧ 
  (-8 < a_1 ∧ a_1 ≤ 1 / 6)

-- Since we are only required to state the problem, we leave the proof as a "sorry".
theorem smallest_a1_range (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
    (h_sum: a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 4 / 3)
    (h_pos: ∀ i : Fin 8, a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - ([-][i]) > 0):
    smallest_a1_possible a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 :=
  by
  sorry

end smallest_a1_range_l254_254054


namespace intersection_of_M_and_N_l254_254790

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254790


namespace alice_souvenir_expense_l254_254544

/-- Alice is visiting Japan. She wants to buy a souvenir for 500 yen. If one US dollar 
    is worth 113 yen, how much money, rounded to the nearest hundredth, does she have to 
    spend for the souvenir in USD? -/ 
theorem alice_souvenir_expense :
  let conversion_rate := 113
  let yen_amount := 500
  let amount_in_usd := (yen_amount : ℝ) / (conversion_rate : ℝ)
  let rounded_amount := Float.round (amount_in_usd * 100) / 100 in
  rounded_amount = 4.42 :=
by
  sorry

end alice_souvenir_expense_l254_254544


namespace workman_problem_l254_254134

theorem workman_problem 
  {A B : Type}
  (W : ℕ)
  (RA RB : ℝ)
  (h1 : RA = (1/2) * RB)
  (h2 : RA + RB = W / 14)
  : W / RB = 21 :=
by
  sorry

end workman_problem_l254_254134


namespace part_I_part_II_l254_254845

noncomputable def f (x a : ℝ) : ℝ := |x + 1| - |x - a|

theorem part_I (x : ℝ) : (∃ a : ℝ, a = 1 ∧ f x a < 1) ↔ x < (1/2) :=
sorry

theorem part_II (a : ℝ) : (∀ x : ℝ, f x a ≤ 6) ↔ (a = 5 ∨ a = -7) :=
sorry

end part_I_part_II_l254_254845


namespace percentage_favoring_all_three_l254_254319

variable (A B C A_union_B_union_C Y X : ℝ)

-- Conditions
axiom hA : A = 0.50
axiom hB : B = 0.30
axiom hC : C = 0.20
axiom hA_union_B_union_C : A_union_B_union_C = 0.78
axiom hY : Y = 0.17

-- Question: Prove that the percentage of those asked favoring all three proposals is 5%
theorem percentage_favoring_all_three :
  A = 0.50 → B = 0.30 → C = 0.20 →
  A_union_B_union_C = 0.78 →
  Y = 0.17 →
  X = 0.05 :=
by
  intros
  sorry

end percentage_favoring_all_three_l254_254319


namespace average_increase_fraction_l254_254347

-- First, we define the given conditions:
def incorrect_mark : ℕ := 82
def correct_mark : ℕ := 62
def number_of_students : ℕ := 80

-- We state the theorem to prove that the fraction by which the average marks increased is 1/4. 
theorem average_increase_fraction (incorrect_mark correct_mark : ℕ) (number_of_students : ℕ) :
  (incorrect_mark - correct_mark) / number_of_students = 1 / 4 :=
by
  sorry

end average_increase_fraction_l254_254347


namespace intersection_of_M_and_N_l254_254793

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254793


namespace largest_square_plots_l254_254532

theorem largest_square_plots (width length pathway_material : Nat) (width_eq : width = 30) (length_eq : length = 60) (pathway_material_eq : pathway_material = 2010) : ∃ (n : Nat), n * (2 * n) = 578 := 
by
  sorry

end largest_square_plots_l254_254532


namespace chord_ratio_l254_254461

theorem chord_ratio (EQ GQ HQ FQ : ℝ) (h1 : EQ = 5) (h2 : GQ = 12) (h3 : HQ = 3) (h4 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 12 / 5 := by
  sorry

end chord_ratio_l254_254461


namespace find_A_l254_254465

-- Define the condition that A and B are single-digit numbers
def single_digit (x : Nat) : Prop := x < 10

-- Define the equation 5A8 - B14 = 364
def equation (A B : Nat) : Prop := (500 + 10 * A + 8) - (B * 100 + 14) = 364

-- The main statement to prove
theorem find_A : ∃ A B : Nat, single_digit A ∧ single_digit B ∧ equation A B ∧ A = 7 :=
by
  have hA : single_digit 7 := by norm_num
  have hB : single_digit 2 := by norm_num
  have heq : equation 7 2 := by norm_num
  use 7, 2
  exact ⟨hA, hB, heq, rfl⟩
  sorry

end find_A_l254_254465


namespace number_of_nine_leading_digits_l254_254982

noncomputable def number_of_leading_9s : ℕ :=
  (λ (T : set ℕ), T.count (λ n, (n / 10^(nat.log10 n)) = 9)) { n : ℕ | ∃ k : ℤ, 0 ≤ k ∧ k ≤ 4000 ∧ n = 9 ^ k }

theorem number_of_nine_leading_digits :
  (∃ T : set ℕ, (T = { 9^k | k : ℤ, 0 ≤ k ∧ k ≤ 4000 }) ∧ 
    (∃ n, n ∈ T ∧ (n / 10^(nat.log10 n)) = 9 ∧ nat.digits 10 (9^4000) = 3817 ∧ n = 184 )) :=
  sorry

end number_of_nine_leading_digits_l254_254982


namespace distance_is_two_l254_254840

noncomputable def parabola_focus : ℝ × ℝ := (3, 0)

noncomputable def hyperbola : {x : ℝ × ℝ // (x.1^2 / 5 - x.2^2 / 4 = 1)} :=
⟨(3, 0), by sorry⟩

noncomputable def hyperbola_asymptote (x : ℝ) : ℝ := (2 / √5) * x

noncomputable def distance_from_focus_to_asymptote (focus : ℝ × ℝ) (asymptote_slope : ℝ) : ℝ :=
(abs (asymptote_slope * focus.1 - focus.2) / sqrt (asymptote_slope^2 + 1))

theorem distance_is_two : distance_from_focus_to_asymptote parabola_focus (2 / √5) = 2 :=
by sorry

end distance_is_two_l254_254840


namespace university_theater_ticket_sales_l254_254083

theorem university_theater_ticket_sales (total_tickets : ℕ) (adult_price : ℕ) (senior_price : ℕ) (senior_tickets : ℕ) 
  (h1 : total_tickets = 510) (h2 : adult_price = 21) (h3 : senior_price = 15) (h4 : senior_tickets = 327) : 
  (total_tickets - senior_tickets) * adult_price + senior_tickets * senior_price = 8748 :=
by 
  -- Proof skipped
  sorry

end university_theater_ticket_sales_l254_254083


namespace hyperbola_eccentricity_l254_254258

theorem hyperbola_eccentricity (a b c : ℝ) (h₀ : a > 0) (h₁ : b > 0)
  (h₂ : b = 1)
  (H : ∀ x y : ℝ, (x, y) ≠ (c, 0) ∧ (x, y) ≠ (0, 1) →
    (x = c * 3 / 4 ∧ y = 1 / 4) →
    (x^2 / a^2 - y^2 / b^2 = 1)) :
  (c / a) = real.sqrt 17 / 3 :=
by sorry

end hyperbola_eccentricity_l254_254258


namespace intersection_eq_l254_254695

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254695


namespace isosceles_triangle_of_equal_segments_l254_254431

open EuclideanGeometry

theorem isosceles_triangle_of_equal_segments {A B C O X Y Z P Q : Point} 
  (h_incircle : incircle A B C O X Y Z)
  (h_intersections : intersect_lines BO YZ P ∧ intersect_lines CO YZ Q)
  (h_eq_segments : dist X P = dist X Q) : 
  isosceles A B C :=
sorry

end isosceles_triangle_of_equal_segments_l254_254431


namespace shirley_cases_needed_l254_254401

-- Define the given conditions
def trefoils_boxes := 54
def samoas_boxes := 36
def boxes_per_case := 6

-- The statement to prove
theorem shirley_cases_needed : trefoils_boxes / boxes_per_case >= samoas_boxes / boxes_per_case ∧ 
                               samoas_boxes / boxes_per_case = 6 :=
by
  let n_cases := samoas_boxes / boxes_per_case
  have h1 : trefoils_boxes / boxes_per_case = 9 := sorry
  have h2 : samoas_boxes / boxes_per_case = 6 := sorry
  have h3 : 9 >= 6 := by linarith
  exact ⟨h3, h2⟩


end shirley_cases_needed_l254_254401


namespace change_calculation_l254_254959

-- Definition of amounts and costs
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8
def cost_chicken_wings : ℕ := 6
def cost_chicken_salad : ℕ := 4
def cost_soda : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Main theorem statement
theorem change_calculation
  (total_cost := cost_chicken_wings + cost_chicken_salad + num_sodas * cost_soda + tax)
  (total_amount := lee_amount + friend_amount)
  : total_amount - total_cost = 3 :=
by
  -- Proof steps placeholder
  sorry

end change_calculation_l254_254959


namespace intersection_of_M_and_N_l254_254648

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254648


namespace mode_QQ_Live_median_DingTalk_student_prefer_DingTalk_school_choose_QQ_Live_l254_254580

theorem mode_QQ_Live (ratings_QQ_Live : List ℕ) :
  List.mode ratings_QQ_Live = 3 := by
  sorry

theorem median_DingTalk (ratings_DingTalk : List ℕ) :
  List.median ratings_DingTalk = 4 := by
  sorry

theorem student_prefer_DingTalk (average_DingTalk average_QQ_Live median_DingTalk median_QQ_Live : ℝ) :
  (average_DingTalk > average_QQ_Live) ∨ (median_DingTalk > median_QQ_Live) → 
  List.mean (List.map ℝ.ofNat ratings_DingTalk) > List.mean (List.map ℝ.ofNat ratings_QQ_Live) ∨ 
  List.median (List.map ℝ.ofNat ratings_DingTalk) > List.median (List.map ℝ.ofNat ratings_QQ_Live) :=
  by sorry

theorem school_choose_QQ_Live (avg_student_DingTalk avg_teacher_DingTalk avg_student_QQ_Live avg_teacher_QQ_Live : ℝ) :
  0.4 * avg_student_DingTalk + 0.6 * avg_teacher_DingTalk < 
  0.4 * avg_student_QQ_Live + 0.6 * avg_teacher_QQ_Live := by
  sorry

end mode_QQ_Live_median_DingTalk_student_prefer_DingTalk_school_choose_QQ_Live_l254_254580


namespace find_sin_A_l254_254603

variables (a b c : ℝ) (A B C : ℝ)

theorem find_sin_A (h1 : a = 2) (h2 : b = 3) (h3 : cos B = 4 / 5) :
  sin A = 2 / 5 :=
  sorry

end find_sin_A_l254_254603


namespace find_d_l254_254432

noncomputable theory

open_locale real_inner_product_space

def line (x : ℝ) : ℝ := (4*x - 7) / 5

def point (x t : ℝ) (d : ℝ × ℝ) : ℝ × ℝ := (7 + t * d.1, line (7 + t * d.1))

def distance (p1 p2 : ℝ × ℝ) : ℝ := real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem find_d (d : ℝ × ℝ) :
  (∀ (t x : ℝ), x ≤ 7 → distance (x, line x) (7, 3) = |t| →
    point x t d = (7 + t * d.1, 3 + t * d.2)) →
  d = (-25 / real.sqrt 41, -20 / real.sqrt 41) :=
sorry

end find_d_l254_254432


namespace intersection_of_M_and_N_l254_254741

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254741


namespace intersection_M_N_l254_254736

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254736


namespace num_non_fiction_books_l254_254517

-- Definitions based on the problem conditions
def num_fiction_configurations : ℕ := 24
def total_configurations : ℕ := 36

-- Non-computable definition for factorial
noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

-- Theorem to prove the number of new non-fiction books
theorem num_non_fiction_books (n : ℕ) :
  num_fiction_configurations * factorial n = total_configurations → n = 2 :=
by
  sorry

end num_non_fiction_books_l254_254517


namespace sum_of_digits_of_product_l254_254560

def repeated_digit_string (d : ℕ) (n : ℕ) : ℕ :=
  (n*(10^n - 1))/9

def digit_sum (n : ℕ) : ℕ :=
  n.digits.sum

theorem sum_of_digits_of_product :
  digit_sum (repeated_digit_string 9 100 * repeated_digit_string 5 100) = 1800 := sorry

end sum_of_digits_of_product_l254_254560


namespace intersection_M_N_l254_254774

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254774


namespace sum_first_11_terms_l254_254330

-- Defining the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a (n + 1) = a n + d

noncomputable def a (n : ℕ) : ℤ := sorry -- Define the specific terms of the sequence

-- Conditions from the problem
variables (d : ℤ)
axiom a9 : a 8 = (1/2 : ℚ) * a 11 + 6 -- note: indices start from 0 in Lean, so a₉ is a(8)

-- Defining the sum of the first 11 terms
def S_11 (a : ℕ → ℤ) : ℤ :=
(finset.range 11).sum a

-- Stating the theorem
theorem sum_first_11_terms (a : ℕ → ℤ) (d : ℤ) (ha : arithmetic_sequence a d) (h9 : a 8 = (1/2 : ℚ) * a 11 + 6):
  S_11 a = 132 :=
sorry

end sum_first_11_terms_l254_254330


namespace average_age_of_second_group_is_16_l254_254411

theorem average_age_of_second_group_is_16
  (total_age_15_students : ℕ := 225)
  (total_age_first_group_7_students : ℕ := 98)
  (age_15th_student : ℕ := 15) :
  (total_age_15_students - total_age_first_group_7_students - age_15th_student) / 7 = 16 := 
by
  sorry

end average_age_of_second_group_is_16_l254_254411


namespace value_of_a6_l254_254448

theorem value_of_a6 
  (S : ℕ → ℤ)
  (a : ℕ → ℤ) 
  (hS : ∀ n, S n = 3 * n^2 - 5 * n)
  (ha : ∀ n, n ≥ 2 → a n = S n - S (n - 1)) 
  (h1 : a 1 = S 1):
  a 6 = 28 :=
sorry

end value_of_a6_l254_254448


namespace exists_positive_integer_m_l254_254361

theorem exists_positive_integer_m (n : ℕ) (hn : 0 < n) : ∃ m : ℕ, 0 < m ∧ 7^n ∣ (3^m + 5^m - 1) :=
sorry

end exists_positive_integer_m_l254_254361


namespace num_pos_divisors_30_l254_254886

theorem num_pos_divisors_30 : ∃ n : ℕ, n = 8 ∧ (∀ m : ℕ, m ∣ 30 ↔ m ∈ {1, 2, 3, 5, 6, 10, 15, 30})
 :=
begin
  sorry
end

end num_pos_divisors_30_l254_254886


namespace intersection_M_N_l254_254621

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254621


namespace sets_of_bleachers_l254_254993

def totalFans : ℕ := 2436
def fansPerSet : ℕ := 812

theorem sets_of_bleachers (n : ℕ) (h : totalFans = n * fansPerSet) : n = 3 :=
by {
    sorry
}

end sets_of_bleachers_l254_254993


namespace num_girls_not_playing_soccer_l254_254110

variables (total_students : ℕ) (total_boys : ℕ) (total_soccer_players : ℕ) 
variables (percent_boys_soccer : ℝ)

def total_girls := total_students - total_boys
def boys_playing_soccer := (percent_boys_soccer * total_soccer_players).to_nat
def girls_playing_soccer := total_soccer_players - boys_playing_soccer
def girls_not_playing_soccer := total_girls total_students total_boys - girls_playing_soccer total_soccer_players percent_boys_soccer

theorem num_girls_not_playing_soccer (h_total_students : total_students = 420) 
                                      (h_total_boys : total_boys = 296) 
                                      (h_total_soccer_players : total_soccer_players = 250) 
                                      (h_percent_boys_soccer : percent_boys_soccer = 0.86) :
  girls_not_playing_soccer total_students total_boys total_soccer_players percent_boys_soccer = 89 :=
by {
  rw [h_total_students, h_total_boys, h_total_soccer_players, h_percent_boys_soccer],
  -- Steps leading to the final result:
  sorry
}

end num_girls_not_playing_soccer_l254_254110


namespace root_sum_expression_eq_l254_254978

noncomputable theory
open Classical

theorem root_sum_expression_eq :
  (∀ r s t : ℝ, (r + s + t = 15) → (rs + rt + st = 25) → (rst = 10) → 
  (x^3 - 15*x^2 + 25*x - 10 = 0) →
  (\frac{r}{\frac{1}{r} + st} + \frac{s}{\frac{1}{s} + tr} + \frac{t}{\frac{1}{t} + rs}) = \frac{175}{11}) :=
by sorry

end root_sum_expression_eq_l254_254978


namespace empty_volume_in_cylinder_is_correct_l254_254068

-- conditions
def radius_cone : ℝ := 10
def height_cone : ℝ := 10
def height_cylinder : ℝ := 30
def number_of_cones : ℕ := 3

-- volume formulas
def volume_cylinder : ℝ := π * radius_cone^2 * height_cylinder
def volume_cone : ℝ := (1/3) * π * radius_cone^2 * height_cone
def total_volume_cones : ℝ := volume_cone * number_of_cones

-- volume of empty space
def volume_empty_space : ℝ := volume_cylinder - total_volume_cones

-- target proof statement
theorem empty_volume_in_cylinder_is_correct : volume_empty_space = 2000 * π :=
by sorry

end empty_volume_in_cylinder_is_correct_l254_254068


namespace intersection_M_N_l254_254779

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254779


namespace loss_percent_correct_l254_254501

/-- Definition of cost price and selling price -/
def cost_price : ℝ := 1200
def selling_price : ℝ := 800

/-- Definition of the loss percent -/
def loss_percent := ((cost_price - selling_price) / cost_price) * 100

/-- The proof statement for the loss percent -/
theorem loss_percent_correct : loss_percent = 33.33 :=
by
  -- skipped proofs
  sorry

end loss_percent_correct_l254_254501


namespace count_even_five_digit_numbers_l254_254211

-- Define the set of digits
def digits : Finset ℕ := {1, 2, 3, 4, 5}

-- Function to check if a number is even
def is_even (n : ℕ) : Prop := n % 2 = 0

-- Define the problem statement
theorem count_even_five_digit_numbers : ∃ n : ℕ, 
  (∑ last_digit in digits, is_even last_digit → 
  (∑ first_digit in digits \ {last_digit}, 
  (∑ second_digit in digits \ {last_digit, first_digit},
  (∑ third_digit in digits \ {last_digit, first_digit, second_digit},
  (∑ fourth_digit in digits \ {last_digit, first_digit, second_digit, third_digit}, 
  1)))) = n) ∧ n = 48 := by
{
  sorry -- Proof skipped
}

end count_even_five_digit_numbers_l254_254211


namespace find_a_l254_254440

def vector (α : Type*) := fin 2 → α

variables {α : Type*} [LinearOrderedField α]

def dot_product (u v : vector α) : α :=
  u 0 * v 0 + u 1 * v 1

def norm_squared (v : vector α) : α :=
  dot_product v v

def proj (u v : vector α) : vector α :=
  (dot_product u v / norm_squared v) • v

theorem find_a (a : α) (h : proj ⟨![3, a]⟩ ⟨![-1, 2]⟩ = (-7 / 5 : α) • ⟨![-1, 2]⟩) : a = -2 :=
by sorry

end find_a_l254_254440


namespace midpoint_distance_l254_254980

variables {A B C D E M : Type*}
variables [triangle A B C]
variables (D : midpoint A B)
variables (E : midpoint B C)
variables (M : lies_on M C)

theorem midpoint_distance (h : dist M D < dist A D) : dist M E > dist E C :=
sorry

end midpoint_distance_l254_254980


namespace intersection_M_N_l254_254689

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254689


namespace quadrilateral_parallelogram_proof_l254_254379

-- Definitions
variable {A B C D E F G H : Type} [EuclideanGeometry A B C D E F G H]
variable (AB BC CD DA : ℝ)
variable (EG FH : ℝ)
variable (is_midpoint : ∀ {P Q R S : Type} [EuclideanGeometry P Q R S], Midpoint P Q R S)

-- Conditions as Lean definitions (assuming EuclideanGeometry is a placeholder for required geometric properties and Midpoint is a proper definition)
def quadrilateral_is_parallelogram (A B C D E F G H : Type) [EuclideanGeometry A B C D E F G H] 
  (AB BC CD DA : ℝ) (EG FH : ℝ) (is_midpoint : ∀ {P Q R S : Type} [EuclideanGeometry P Q R S], Midpoint P Q R S :=
  let s := (AB + BC + CD + DA) / 2 in
  EG = (DA + BC) / 2 ∧ FH = (AB + CD) / 2 ∧ (EG + FH = s) → parallelogram A B C D

-- The theorem statement
theorem quadrilateral_parallelogram_proof {A B C D E F G H : Type} [EuclideanGeometry A B C D E F G H]
  (AB BC CD DA : ℝ) (EG FH : ℝ) (is_midpoint : ∀ {P Q R S : Type} [EuclideanGeometry P Q R S], Midpoint P Q R S) :
  let s := (AB + BC + CD + DA) / 2 in
  EG = (DA + BC) / 2 ∧ FH = (AB + CD) / 2 ∧ (EG + FH = s) → parallelogram A B C D :=
by sorry

end quadrilateral_parallelogram_proof_l254_254379


namespace determine_parabola_coefficients_l254_254188

noncomputable def parabola_coefficients (a b c : ℚ) : Prop :=
  ∀ (x y : ℚ), 
      (y = a * x^2 + b * x + c) ∧
      (
        ((4, 5) = (x, y)) ∧
        ((2, 3) = (x, y))
      )

theorem determine_parabola_coefficients :
  parabola_coefficients (-1/2) 4 (-3) :=
by
  sorry

end determine_parabola_coefficients_l254_254188


namespace age_ratio_3_2_l254_254213

/-
Define variables: 
  L : ℕ -- Liam's current age
  M : ℕ -- Mia's current age
  y : ℕ -- number of years until the age ratio is 3:2
-/

theorem age_ratio_3_2 (L M : ℕ) 
  (h1 : L - 4 = 2 * (M - 4)) 
  (h2 : L - 10 = 3 * (M - 10)) 
  (h3 : ∃ y, (L + y) * 2 = (M + y) * 3) : 
  ∃ y, y = 8 :=
by
  sorry

end age_ratio_3_2_l254_254213


namespace right_triangle_side_length_l254_254243

theorem right_triangle_side_length (area : ℝ) (side1 : ℝ) (side2 : ℝ) (h_area : area = 8) (h_side1 : side1 = Real.sqrt 10) (h_area_eq : area = 0.5 * side1 * side2) :
  side2 = 1.6 * Real.sqrt 10 :=
by 
  sorry

end right_triangle_side_length_l254_254243


namespace sum_cos_fourth_l254_254186

theorem sum_cos_fourth :
  let T := ∑ k in Finset.range 181, (Real.cos (k * Real.pi / 180))^4
  in T = 543 / 8 :=
by
  let T := ∑ k in Finset.range 181, (Real.cos (k * Real.pi / 180))^4
  show T = 543 / 8
  sorry

end sum_cos_fourth_l254_254186


namespace cubes_intersected_by_diagonal_l254_254189

theorem cubes_intersected_by_diagonal :
  let l := 120
  let w := 210
  let h := 336
  let gcd_xy := Int.gcd l w
  let gcd_yz := Int.gcd w h
  let gcd_zx := Int.gcd h l
  let gcd_xyz := Int.gcd (Int.gcd l w) h
  l + w + h - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 576 :=
by
  let l := 120
  let w := 210
  let h := 336
  let gcd_xy := Int.gcd l w
  let gcd_yz := Int.gcd w h
  let gcd_zx := Int.gcd h l
  let gcd_xyz := Int.gcd (Int.gcd l w) h
  have h1 : l + w + h - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 120 + 210 + 336 - 30 - 42 - 24 + 6 := 
    sorry -- prove the intermediate steps
  have h2 : 120 + 210 + 336 - 30 - 42 - 24 + 6 = 576 :=
    sorry -- prove the final evaluation
  exact h1.trans h2

end cubes_intersected_by_diagonal_l254_254189


namespace simplify_fraction_l254_254404

theorem simplify_fraction :
  ( (Real.sqrt 10 + Real.sqrt 15) / (Real.sqrt 3 + Real.sqrt 5 - Real.sqrt 2) =
    (2 * Real.sqrt 30 + 5 * Real.sqrt 2 + 11 * Real.sqrt 5 + 5 * Real.sqrt 3) / 6 ) :=
begin
  sorry
end

end simplify_fraction_l254_254404


namespace train_length_l254_254541

open Real

theorem train_length 
  (v : ℝ) -- speed of the train in km/hr
  (t : ℝ) -- time in seconds
  (d : ℝ) -- length of the bridge in meters
  (h_v : v = 36) -- condition 1
  (h_t : t = 50) -- condition 2
  (h_d : d = 140) -- condition 3
  : (v * 1000 / 3600) * t = 360 + 140 := 
sorry

end train_length_l254_254541


namespace intersection_of_M_and_N_l254_254643

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254643


namespace interval_inclusion_and_irrational_coverage_l254_254965

theorem interval_inclusion_and_irrational_coverage
  {m n : ℕ} (hmn : m ≥ n)
  (S : set (ℕ × ℕ))
  (hS : ∀ a b, (a, b) ∈ S ↔ nat.coprime a b ∧ a ≤ m ∧ b ≤ m ∧ a + b > m)
  (I : ℕ × ℕ → set ℝ)
  (hI : ∀ a b, (a, b) ∈ S →
    (∃ u v : ℕ, v ≥ 0 ∧ u ≥ 0 ∧ a * u - b * v = n ∧
      (∀ v', v' < v → ¬(∃ u', a * u' - b * v' = n)) ∧ 
      I (a, b) = set.Ioo (v / a) (u / b)))
  (α : ℝ) (hα : irrational α ∧ 0 < α ∧ α < 1) :
  (∀ (a b : ℕ), (a, b) ∈ S → I (a, b) ⊆ set.Ioo 0 1) ∧
  (∃! (pairs : finset (ℕ × ℕ)), ∀ (a b : ℕ), (a, b) ∈ pairs ↔ (a, b) ∈ S ∧ α ∈ I (a, b) ∧ pairs.card = n) :=
by sorry

end interval_inclusion_and_irrational_coverage_l254_254965


namespace intersection_M_N_l254_254762

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254762


namespace positive_divisors_of_5400_multiple_of_5_l254_254878

-- Declare the necessary variables and conditions
theorem positive_divisors_of_5400_multiple_of_5 :
  let n := 5400
  let factorization := [(2, 2), (3, 3), (5, 2)]
  ∀ (a b c: ℕ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2 →
    (a*b*c).count(n) = 24 := 
sorry

end positive_divisors_of_5400_multiple_of_5_l254_254878


namespace intersection_M_N_l254_254659

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254659


namespace kims_morning_routine_total_time_l254_254353

def time_spent_making_coffee := 5 -- in minutes
def time_spent_per_employee_status_update := 2 -- in minutes
def time_spent_per_employee_payroll_update := 3 -- in minutes
def number_of_employees := 9

theorem kims_morning_routine_total_time :
  time_spent_making_coffee +
  (time_spent_per_employee_status_update + time_spent_per_employee_payroll_update) * number_of_employees = 50 :=
by
  sorry

end kims_morning_routine_total_time_l254_254353


namespace hyperbola_asymptotes_l254_254590

-- Define the hyperbola
def hyperbola_eq (x y : ℝ) : Prop := x^2 - y^2 / 4 = 1

-- Define the equations for the asymptotes
def asymptote_pos (x y : ℝ) : Prop := y = 2 * x
def asymptote_neg (x y : ℝ) : Prop := y = -2 * x

-- State the theorem
theorem hyperbola_asymptotes (x y : ℝ) :
  hyperbola_eq x y → (asymptote_pos x y ∨ asymptote_neg x y) := 
by
  sorry

end hyperbola_asymptotes_l254_254590


namespace boat_speed_in_still_water_l254_254323

theorem boat_speed_in_still_water (b s : ℝ) (h1 : b + s = 11) (h2 : b - s = 5) : b = 8 :=
by 
  sorry

end boat_speed_in_still_water_l254_254323


namespace father_son_fish_problem_l254_254513

variables {F S x : ℕ}

theorem father_son_fish_problem (h1 : F - x = S + x) (h2 : F + x = 2 * (S - x)) : 
  (F - S) / S = 2 / 5 :=
by sorry

end father_son_fish_problem_l254_254513


namespace problem_statement_l254_254240

theorem problem_statement (x : ℝ) (h₀ : x > 0) (n : ℕ) (hn : n > 0) :
  (x + (n^n : ℝ) / x^n) ≥ (n + 1) :=
sorry

end problem_statement_l254_254240


namespace min_value_of_quadratic_l254_254369

theorem min_value_of_quadratic (x y s : ℝ) (h : x + y = s) : 
  ∃ x y, 3 * x^2 + 2 * y^2 = 6 * s^2 / 5 := sorry

end min_value_of_quadratic_l254_254369


namespace complex_quadrant_l254_254289

def fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

theorem complex_quadrant (z : ℂ) (h : (1 + complex.i) * z = complex.abs (complex.mk (real.sqrt 3) 1)) :
  fourth_quadrant z :=
sorry

end complex_quadrant_l254_254289


namespace groupB_ions_can_coexist_l254_254555

-- Define the ions as constants
constant Na_pos : Type
constant Al_3pos : Type
constant SO4_2neg : Type
constant Cl_neg : Type
constant Cu_2pos : Type
constant Br_neg : Type
constant NH4_pos : Type
constant K_pos : Type
constant HCO3_neg : Type
constant NO3_neg : Type
constant Fe_3pos : Type

-- Define conditions for each group 
def groupA_condition : Prop := 
  -- Solution turning red upon adding phenolphthalein (indicating alkalinity)
  contains Na_pos ∧ contains Al_3pos ∧ contains SO4_2neg ∧ contains Cl_neg ∧ alkaline

def groupB_condition : Prop := 
  -- Solution turning red upon adding KSCN (indicating presence of Fe3+)
  contains Na_pos ∧ contains Cu_2pos ∧ contains Br_neg ∧ contains SO4_2neg 

def groupC_condition : Prop := 
  -- Solution with c(H+)/c(OH-) = 10^12 (indicating acidity)
  contains NH4_pos ∧ contains K_pos ∧ contains HCO3_neg ∧ contains NO3_neg ∧ acidic

def groupD_condition : Prop := 
  -- Solution with c(H+) = sqrt(Kw) (indicating neutrality)
  contains K_pos ∧ contains Fe_3pos ∧ contains Cl_neg ∧ contains SO4_2neg ∧ neutral

-- Define a property for ions coexisting in large amounts
def can_coexist_in_large_amounts (ions : Prop) : Prop := sorry

-- The theorem statement
theorem groupB_ions_can_coexist : 
  groupB_condition → can_coexist_in_large_amounts groupB_condition := sorry

end groupB_ions_can_coexist_l254_254555


namespace sqrt_of_36_is_6_l254_254005

-- Define the naturals
def arithmetic_square_root (x : ℕ) : ℕ := Nat.sqrt x

theorem sqrt_of_36_is_6 : arithmetic_square_root 36 = 6 :=
by
  -- The proof goes here, but we use sorry to skip it as per instructions.
  sorry

end sqrt_of_36_is_6_l254_254005


namespace price_of_light_bulb_and_motor_l254_254325

theorem price_of_light_bulb_and_motor
  (x : ℝ) (motor_price : ℝ)
  (h1 : x + motor_price = 12)
  (h2 : 10 / x = 2 * 45 / (12 - x)) :
  x = 3 ∧ motor_price = 9 :=
sorry

end price_of_light_bulb_and_motor_l254_254325


namespace sixtieth_term_of_arithmetic_sequence_l254_254427

theorem sixtieth_term_of_arithmetic_sequence (a1 a15 : ℚ) (d : ℚ) (h1 : a1 = 7) (h2 : a15 = 37)
  (h3 : a15 = a1 + 14 * d) : a1 + 59 * d = 134.5 := by
  sorry

end sixtieth_term_of_arithmetic_sequence_l254_254427


namespace intersection_eq_l254_254696

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254696


namespace river_flow_speed_l254_254315

theorem river_flow_speed (v : ℝ) :
  (6 - v ≠ 0) ∧ (6 + v ≠ 0) ∧ ((48 / (6 - v)) + (48 / (6 + v)) = 18) → v = 2 := 
by
  sorry

end river_flow_speed_l254_254315


namespace elena_earnings_l254_254198

theorem elena_earnings (hours : ℕ) (seq_earnings : list ℕ)
  (h_length : seq_earnings.length = 5)
  (h_seq : seq_earnings = [3, 4, 5, 6, 7]) :
  hours = 47 → (9 * seq_earnings.sum + (seq_earnings.take 2).sum) = 232 :=
begin
  intro h_hours,
  rw h_hours,
  have h1 : seq_earnings.sum = 25,
  { rw h_seq, norm_num },
  have h2 : (seq_earnings.take 2).sum = 7,
  { rw h_seq, norm_num },
  calc 9 * seq_earnings.sum + (seq_earnings.take 2).sum
      = 9 * 25 + 7 : by rw [h1, h2]
  ... = 225 + 7 : by norm_num
  ... = 232 : by norm_num
end

end elena_earnings_l254_254198


namespace a_2018_eq_5_l254_254938

noncomputable def a : ℕ → ℚ
| 1 => -1 / 4
| n + 1 => 1 - 1 / a n

theorem a_2018_eq_5 : a 2018 = 5 := 
by
  -- the proof would go here
  sorry

end a_2018_eq_5_l254_254938


namespace yadav_expense_l254_254505

noncomputable def clothesAndTransportExpense (S : ℝ) : ℝ :=
  0.2 * S

theorem yadav_expense (S : ℝ) (h1 : 0.4 * S - 0.2 * S = 4038) :
  clothesAndTransportExpense S = 4038 :=
by
  -- Given conditions
  have savings : (S - 0.6 * S) * 0.5 = 4038 := by
    calc
      (S - 0.6 * S) * 0.5 = 0.4 * S * 0.5 : by sorry
      ... = 0.2 * S : by sorry
  -- Proof step
  show clothesAndTransportExpense S = 4038 from sorry

end yadav_expense_l254_254505


namespace intersection_M_N_l254_254667

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254667


namespace smallest_value_of_a1_conditions_l254_254037

noncomputable theory

variables {a1 a2 a3 a4 a5 a6 a7 a8 : ℝ}

/-- The smallest value of \(a_1\) when the sum of \(a_1, \ldots, a_8\) is \(4/3\) 
    and the sum of any seven of these numbers is positive. -/
theorem smallest_value_of_a1_conditions 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 / 3)
  (h_sum_seven : ∀ i : {j // j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8}, 
                  0 < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 - a i.val) :
  -8 < a1 ∧ a1 ≤ 1 / 6 :=
sorry

end smallest_value_of_a1_conditions_l254_254037


namespace intersection_M_N_l254_254629

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254629


namespace salesman_total_commission_l254_254147

def total_sales := 14000
def excess_sales := total_sales - 10000
def bonus_commission := 0.03 * excess_sales
def total_commission := 0.09 * total_sales + bonus_commission

theorem salesman_total_commission :
  (bonus_commission = 120) →
  (total_commission = 1380) :=
by
  intros h_bonus_commission
  rw [bonus_commission, show excess_sales = 4000, by sorry]
  rw [show total_sales = 14000, by sorry]
  rw h_bonus_commission
  linarith

end salesman_total_commission_l254_254147


namespace negative_square_inequality_l254_254605

theorem negative_square_inequality (a b : ℝ) (h1 : a < b) (h2 : b < 0) : a^2 > b^2 :=
sorry

end negative_square_inequality_l254_254605


namespace calculate_expressions_l254_254562

theorem calculate_expressions :
  (16^0.75 - 3^0.3 * 3^1.7 + 1.5^0 = 0) ∧
  (Real.log 8 / Real.log 4 - Real.log 3 / Real.log (1/9) - Real.log 4 / Real.log (sqrt 2) + (1/2)^(-1 + Real.log 4 / Real.log (1/2)) = 6) :=
by
  sorry

end calculate_expressions_l254_254562


namespace annual_pension_l254_254145

variable (a b p q k y x : ℝ)

-- Conditions
def condition1 : Prop := x = k * Real.sqrt y
def condition2 : Prop := x + p = k * Real.sqrt (y + a)
def condition3 : Prop := x + q = k * Real.sqrt (y + b)

-- Theorem to prove
theorem annual_pension (h1 : condition1 a b p q k y x)
                       (h2 : condition2 a b p q k y x)
                       (h3 : condition3 a b p q k y x) :
  x = (a * q^2 - b * p^2) / (2 * (b * p - a * q)) :=
sorry

end annual_pension_l254_254145


namespace max_seq_b_at_six_l254_254939

noncomputable def seq_a : ℕ → ℝ 
| 1       := 0
| (n + 2) := seq_a (n + 1) + (2 * (n + 1) - 1)

noncomputable def seq_b (n : ℕ) : ℝ :=
  n * sqrt (seq_a (n + 1) + 1) * (8 / 11) ^ (n - 1)

theorem max_seq_b_at_six :
  ∀ (n : ℕ), seq_b n ≤ seq_b 6 := by sorry

end max_seq_b_at_six_l254_254939


namespace M_inter_N_eq_2_4_l254_254823

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254823


namespace outfit_choices_no_same_color_l254_254892

def num_outfit_choices (shirt_colors pants_colors hat_colors : Finset ℕ) (no_same_color : ∀ s p h, s ≠ p ∧ p ≠ h ∧ s ≠ h) : ℕ := 
  (shirt_colors.card * (pants_colors.card - 1) * (hat_colors.card - 2))

theorem outfit_choices_no_same_color 
  (shirts pants hats : ℕ) 
  (colors : Finset ℕ) 
  (h_eq : shirts = 7) 
  (p_eq : pants = 7) 
  (ha_eq : hats = 7) 
  (c_eq : colors.card = 7)
  (no_same_color : ∀ s p h, s ≠ p ∧ p ≠ h ∧ s ≠ h) :
  num_outfit_choices colors colors colors no_same_color = 210 := by 
  sorry 

end outfit_choices_no_same_color_l254_254892


namespace max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254468

def max_distinct_numbers (n : ℕ) (k : ℕ) (f : ℕ → ℕ → Prop) : ℕ :=
  sorry

theorem max_num_distinct_from_1_to_1000_no_diff_4_5_6 :
  max_distinct_numbers 1000 4 (λ a b, ¬(a - b = 4 ∨ a - b = 5 ∨ a - b = 6)) = 400 :=
sorry

end max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254468


namespace calculate_fabric_per_sleeve_l254_254344

def cost_per_square_foot : ℝ := 3
def total_spent : ℝ := 468
def skirts := 3
def dimensions : ℝ × ℝ := (12, 4)
def shirt_area : ℝ := 2
def skirt_area : ℝ := dimensions.1 * dimensions.2
def total_skirt_area : ℝ := skirt_area * skirts
def total_areas : ℝ := total_skirt_area + shirt_area
def sleeves_total_cost : ℝ := total_spent - (total_areas * cost_per_square_foot)
def sleeves_total_area : ℝ := sleeves_total_cost / cost_per_square_foot
def sleeves_count : ℝ := 2
def area_per_sleeve : ℝ := sleeves_total_area / sleeves_count

theorem calculate_fabric_per_sleeve : area_per_sleeve = 5 := by
  sorry

end calculate_fabric_per_sleeve_l254_254344


namespace minimum_reciprocal_sum_l254_254990

theorem minimum_reciprocal_sum (b : Fin 15 → ℝ) (h_pos : ∀ i, 0 < b i) (h_sum : ∑ i, b i = 1) :
  (∑ i, (1 / b i)) ≥ 225 :=
sorry

end minimum_reciprocal_sum_l254_254990


namespace minimum_value_proof_l254_254239

noncomputable def min_value (x y : ℝ) : ℝ :=
  (x^2 / (x + 2)) + (y^2 / (y + 1))

theorem minimum_value_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x + y = 1) :
  min_value x y = 1 / 4 :=
  sorry

end minimum_value_proof_l254_254239


namespace intersection_M_N_l254_254764

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254764


namespace expand_product_l254_254586

theorem expand_product : (2 : ℝ) * (x + 2) * (x + 3) * (x + 4) = 2 * x^3 + 18 * x^2 + 52 * x + 48 :=
by
  sorry

end expand_product_l254_254586


namespace intersection_M_N_l254_254728

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254728


namespace tan_of_slope_angle_l254_254832

def slope_angle_tangent (α : ℝ) : Prop :=
  tan (α - (π / 4)) = 3

theorem tan_of_slope_angle {α : ℝ} (h : tan α = -2) : slope_angle_tangent α :=
  by
  -- Begin proof (actual proof not included)
  sorry

end tan_of_slope_angle_l254_254832


namespace girls_in_school_l254_254425

noncomputable def num_of_girls (total_students : ℕ) (sampled_students : ℕ) (sampled_diff : ℤ) : ℕ :=
  sorry

theorem girls_in_school :
  let total_students := 1600
  let sampled_students := 200
  let sampled_diff := 10
  num_of_girls total_students sampled_students sampled_diff = 760 :=
  sorry

end girls_in_school_l254_254425


namespace positive_divisors_multiple_of_5_l254_254871

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end positive_divisors_multiple_of_5_l254_254871


namespace tan_subtraction_l254_254214

variable {α β : ℝ}

theorem tan_subtraction (h1 : Real.tan α = 2) (h2 : Real.tan β = -3) : Real.tan (α - β) = -1 := by
  sorry

end tan_subtraction_l254_254214


namespace number_of_people_playing_l254_254102

theorem number_of_people_playing
  (total_points : ℕ)
  (points_per_person : ℕ)
  (total_points_eq : total_points = 18)
  (points_per_person_eq : points_per_person = 2)
  :
  total_points / points_per_person = 9 :=
by
  rw [total_points_eq, points_per_person_eq]
  norm_num

end number_of_people_playing_l254_254102


namespace intersection_M_N_l254_254812

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254812


namespace number_of_divisors_30_l254_254884

theorem number_of_divisors_30 : 
  ∃ (d : ℕ), d = 2 * 2 * 2 ∧ d = 8 :=
  by sorry

end number_of_divisors_30_l254_254884


namespace minimum_distance_QS_ST_TR_correct_l254_254293

open Real

noncomputable def minimum_distance_QS_ST_TR (P Q R : ℝ×ℝ) (angle_PQR : ℝ) (PQ_length PR_length : ℝ) : ℝ :=
  if angle_PQR = 50 ∧ PQ_length = 8 ∧ PR_length = 10 then 16.337 else -1

theorem minimum_distance_QS_ST_TR_correct 
  (P Q R S T : ℝ×ℝ)
  (h1 : ∃ angle_PQR, angle_PQR = 50)
  (h2 : dist P Q = 8)
  (h3 : dist P R = 10)
  (h4 : on_line_segment P Q S)
  (h5 : on_line_segment P R T) :
  minimum_distance_QS_ST_TR P Q R 50 8 10 = 16.337 :=
by
  sorry

end minimum_distance_QS_ST_TR_correct_l254_254293


namespace least_gumballs_to_ensure_four_same_color_l254_254132

theorem least_gumballs_to_ensure_four_same_color :
  let purple := 12
  let orange := 6
  let green := 8
  let yellow := 5
  (∀ picks : Nat, (picks >= 13) → (∃ color, (count color picks) >= 4)) :=
by
  sorry

end least_gumballs_to_ensure_four_same_color_l254_254132


namespace range_of_m_l254_254511

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 - 2 * x - 1 ≥ m^2 - 3 * m) ↔ m ∈ set.Ioo (-(1 : ℝ)) 1 ∪ set.Ioo 2 (⊤) := sorry

end range_of_m_l254_254511


namespace sin_cos_product_l254_254374

theorem sin_cos_product (x : ℝ) (h₁ : 0 < x) (h₂ : x < π / 2) (h₃ : Real.sin x = 3 * Real.cos x) : 
  Real.sin x * Real.cos x = 3 / 10 :=
by
  sorry

end sin_cos_product_l254_254374


namespace at_least_4_stayed_l254_254903

-- We define the number of people and their respective probabilities of staying.
def numPeople : ℕ := 8
def numCertain : ℕ := 5
def numUncertain : ℕ := 3
def probUncertainStay : ℚ := 1 / 3

-- We state the problem formally:
theorem at_least_4_stayed :
  (probUncertainStay ^ 3 * 3 + (probUncertainStay ^ 2 * (2 / 3) * 3) + (probUncertainStay * (2 / 3)^2 * 3)) = 19 / 27 :=
by
  sorry

end at_least_4_stayed_l254_254903


namespace find_AB_length_l254_254917

noncomputable def tan := Real.tan -- We need the noncomputable definition of tangent since we use a noncomputable Real value.

theorem find_AB_length :
  ∀ (ABC : Triangle),
  ABC.is_right_triangle B ∧
  ABC.angle B = 90 ∧
  ABC.angle A = 40 ∧
  ABC.side BC = 12
  → ABC.side AB = 14.3 :=
by
  sorry

end find_AB_length_l254_254917


namespace change_calculation_l254_254954

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l254_254954


namespace intersection_M_N_l254_254678

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254678


namespace intersection_M_N_l254_254800

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254800


namespace min_value_expression_eq_2_l254_254237

theorem min_value_expression_eq_2 (a b m n : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : m > 0) (h4 : n > 0)
  (h5 : a + b = 1) (h6 : m * n = 2) : (am + bn) * (bm + an) ≥ 2 :=
begin
  sorry
end

end min_value_expression_eq_2_l254_254237


namespace garrison_provisions_last_initially_l254_254527

noncomputable def garrison_initial_provisions (x : ℕ) : Prop :=
  ∃ x : ℕ, 2000 * (x - 21) = 3300 * 20 ∧ x = 54

theorem garrison_provisions_last_initially :
  garrison_initial_provisions 54 :=
by
  sorry

end garrison_provisions_last_initially_l254_254527


namespace rectangle_area_from_roots_l254_254015

theorem rectangle_area_from_roots :
  let z := ℂ,
  let p : Polynomial ℂ := Polynomial.C (2 - 8 * Complex.I) + 
                          Polynomial.C (-8 - Complex.I) * Polynomial.X + 
                          Polynomial.C (-4 + 4 * Complex.I) * Polynomial.X^2 + 
                          Polynomial.C 4 * Complex.I * Polynomial.X^3 + 
                          Polynomial.X^4,
  (z.roots p).length = 4 ∧
  (∀ (a b c d : ℂ), 
    z.roots p = [a, b, c, d] → 
    a + b + c + d = -4 * Complex.I ∧
    let O := -Complex.I,
    let area := √((Complex.abs (-O - a)) * (Complex.abs (-O - b))) * 
                (Complex.abs (-O - c) * (Complex.abs (-O - d))),
    area = √17) :=
sorry

end rectangle_area_from_roots_l254_254015


namespace intersection_of_M_and_N_l254_254792

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254792


namespace CM_is_median_l254_254359

theorem CM_is_median 
  (A B C H L M : Point) 
  (triangle_ABC : Triangle A B C)
  (AH : Altitude A H triangle_ABC)
  (BL : AngleBisector B L triangle_ABC)
  (CM : Median C M triangle_ABC)
  (triangle_HLM : Triangle H L M)
  (AH_HLM : Altitude A H triangle_HLM)
  (BL_HLM : AngleBisector B L triangle_HLM) :
  Median C M triangle_HLM := 
sorry

end CM_is_median_l254_254359


namespace unique_geometric_sequence_value_l254_254017

theorem unique_geometric_sequence_value (a : ℝ) (q : ℝ) (h_a_pos : 0 < a)
  (geometric_sequence : ∀ n, a_n n = a * q ^ (n - 1))
  (geometric_sequence_condition : (a + 1) • (a * q + 2) = (a * q + 2) • (a * q^2 + 3)) :
  a = 1 / 3 :=
by
  sorry

end unique_geometric_sequence_value_l254_254017


namespace problem_diamond_value_l254_254282

def diamond (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem problem_diamond_value :
  diamond 3 4 = 36 := 
by
  sorry

end problem_diamond_value_l254_254282


namespace cost_to_fill_sandbox_l254_254946

-- Definitions for conditions
def side_length : ℝ := 3
def volume_per_bag : ℝ := 3
def cost_per_bag : ℝ := 4

-- Theorem statement
theorem cost_to_fill_sandbox : (side_length ^ 3 / volume_per_bag * cost_per_bag) = 36 := by
  sorry

end cost_to_fill_sandbox_l254_254946


namespace people_in_line_after_three_rounds_l254_254556

-- Initial conditions
def initial_people : Int := 30
def people_per_round : Int := 5
def leave_per_round : Int := 10
def priority_per_round : Int := 5
def rounds : Int := 3

theorem people_in_line_after_three_rounds : 
  ∀ (initial_people people_per_round leave_per_round priority_per_round rounds : Int), 
  initial_people = 30 → 
  people_per_round = 5 → 
  leave_per_round = 10 → 
  priority_per_round = 5 → 
  rounds = 3 → 
  let round1 := initial_people - people_per_round - leave_per_round + priority_per_round in
  let round2 := round1 - people_per_round + priority_per_round in
  let round3 := round2 - people_per_round + priority_per_round in
  round3 = 20 := 
by
  intros initial_people people_per_round leave_per_round priority_per_round rounds
  intros h_initial h_pp_round h_leave h_priority h_rounds
  have h_round1 : round1 = initial_people - people_per_round - leave_per_round + priority_per_round := rfl
  have h_round2 : round2 = round1 - people_per_round + priority_per_round := rfl
  have h_round3 : round3 = round2 - people_per_round + priority_per_round := rfl
  rw [h_initial, h_pp_round, h_leave, h_priority, h_rounds] at h_round1 h_round2 h_round3
  sorry

end people_in_line_after_three_rounds_l254_254556


namespace deduce_pi_from_cylinder_volume_l254_254426

theorem deduce_pi_from_cylinder_volume 
  (C h V : ℝ) 
  (Circumference : C = 20) 
  (Height : h = 11)
  (VolumeFormula : V = (1 / 12) * C^2 * h) : 
  pi = 3 :=
by 
  -- Carry out the proof
  sorry

end deduce_pi_from_cylinder_volume_l254_254426


namespace sin_B_value_cos_2A_plus_pi_over_6_value_l254_254942

section TriangleProperties

variables (a b c : ℝ) (A B : ℝ)

-- Define the given conditions in Lean
def condition_a : Prop := a = 8
def condition_b : Prop := b - c = 2
def condition_cosA : Prop := cos A = -1 / 4

-- Theorem for Part 1
theorem sin_B_value (h₁ : condition_a a) (h₂ : condition_b b c) (h₃ : condition_cosA A) :
  sin B = 3 * sqrt 15 / 16 :=
sorry

-- Theorem for Part 2
theorem cos_2A_plus_pi_over_6_value (h₁ : condition_a a) (h₂ : condition_b b c) (h₃ : condition_cosA A) :
  cos (2 * A + π / 6) = (sqrt 15 - 7 * sqrt 3) / 16 :=
sorry

end TriangleProperties

end sin_B_value_cos_2A_plus_pi_over_6_value_l254_254942


namespace sum_of_powers_of_neg_one_l254_254180

theorem sum_of_powers_of_neg_one : 
  (∑ k in Finset.range 2007, (-1)^(k + 1)) = -1 :=
by
  sorry

end sum_of_powers_of_neg_one_l254_254180


namespace inequality_solution_l254_254027

-- Define the expression for the inequality
def expr (x : ℝ) : ℝ := (1 - 2 * x) / ((x - 3) * (2 * x + 1))

-- Define the solution set as a set of real numbers
def solutionSet : Set ℝ := {x | x < - 1 / 2} ∪ {x | 1 / 2 ≤ x ∧ x < 3}

-- State the theorem that proves the solution set of the inequality
theorem inequality_solution : {x : ℝ | expr x ≥ 0} = solutionSet :=
sorry

end inequality_solution_l254_254027


namespace change_proof_l254_254956

-- Definitions of the given conditions
def lee_money : ℕ := 10
def friend_money : ℕ := 8
def chicken_wings_cost : ℕ := 6
def chicken_salad_cost : ℕ := 4
def soda_cost : ℕ := 1
def num_sodas : ℕ := 2
def tax : ℕ := 3

-- Statement of the theorem
theorem change_proof : 
  let total_money : ℕ := lee_money + friend_money,
      meal_cost_before_tax : ℕ := chicken_wings_cost + chicken_salad_cost + num_sodas * soda_cost,
      total_meal_cost : ℕ := meal_cost_before_tax + tax
  in total_money - total_meal_cost = 3 := 
by
  -- We skip the proof, as it's not required per instructions
  sorry

end change_proof_l254_254956


namespace orthic_triangle_has_minimal_perimeter_l254_254224

-- Define the vertices of the triangle and the altitudes.
variables {A B C P Q R : Point}

-- Define the conditions: acute-angled triangle and orthogonality conditions.
variables (h_tri : Triangle ABC)
          (h_acute : AcuteAngled ABC)
          (h_perpAP : Perpendicular AP BC)
          (h_perpBQ : Perpendicular BQ AC)
          (h_perpCR : Perpendicular CR AB)

-- The orthic triangle is the triangle formed by the feet of the altitudes.
def orthic_triangle (ABC : Triangle) := Triangle PQR

-- Statement of the theorem.
theorem orthic_triangle_has_minimal_perimeter :
  ∀ (T : Triangle) (h_inscribed : Inscribed T ABC), 
    Perimeter (orthic_triangle ABC) ≤ Perimeter T :=
by
  sorry

end orthic_triangle_has_minimal_perimeter_l254_254224


namespace avg_age_grandparents_is_64_l254_254128

-- Definitions of conditions
def num_grandparents : ℕ := 2
def num_parents : ℕ := 2
def num_grandchildren : ℕ := 3
def num_family_members : ℕ := num_grandparents + num_parents + num_grandchildren

def avg_age_parents : ℕ := 39
def avg_age_grandchildren : ℕ := 6
def avg_age_family : ℕ := 32

-- Total number of family members
theorem avg_age_grandparents_is_64 (G : ℕ) :
  (num_grandparents * G) + (num_parents * avg_age_parents) + (num_grandchildren * avg_age_grandchildren) = (num_family_members * avg_age_family) →
  G = 64 :=
by
  intro h
  sorry

end avg_age_grandparents_is_64_l254_254128


namespace parabola_intersection_l254_254226

-- Define the given initial conditions and problem
Variables {n m : ℕ}
hypothesis (hn : n ≥ 2)

-- Define the equations and intersection conditions
def parabola (x : ℝ) : ℝ := n * x - 1

def intersect_point (x0 : ℝ) : Prop := 
  let y0 := x0
  (x0 * x0 = n * x0 - 1) ∧ (y0 = x0)

-- The main theorem to prove
theorem parabola_intersection (x0 : ℝ)
  (h_intersect : intersect_point x0)
  (hm_pos : m > 0) :
  ∃ k : ℕ, k ≥ 2 ∧ (x0^m * x0^m = k * x0^m - 1) :=
sorry

end parabola_intersection_l254_254226


namespace translate_sin_3x_pi_4_l254_254924

-- Define the original function f
def f (x : ℝ) : ℝ := Real.sin (3 * x + Real.pi / 4)

-- Define the resulting function g after translation by φ units
def g (x φ : ℝ) : ℝ := Real.sin (3 * (x + φ) + Real.pi / 4)

-- The property that g passes through the origin
theorem translate_sin_3x_pi_4 (φ : ℝ) (hφ : φ > 0) : 
  g 0 φ = 0 ↔ φ = Real.pi / 4 :=
by
  sorry

end translate_sin_3x_pi_4_l254_254924


namespace digit_possibilities_for_divisibility_by_4_l254_254526

theorem digit_possibilities_for_divisibility_by_4 :
  (Finset.filter (λ N, (8620 + 10 * N) % 4 = 0) (Finset.range 10)).card = 3 :=
by
  sorry

end digit_possibilities_for_divisibility_by_4_l254_254526


namespace smallest_m_Rn_eq_l_l254_254380

-- Define the problem parameters and transformation R
noncomputable def l (x : ℝ) : ℝ := (11 / 25) * x
def angle_l1 := Real.pi / 40
def angle_l2 := Real.pi / 72

-- Define the transformation R
def R (θ : ℝ) : ℝ := 2 * angle_l2 - (2 * angle_l1 - θ)

-- Define the iterative transformation R
def Rn (θ : ℝ) (n : ℕ) : ℝ := θ + (7 * (Real.pi / 180) * n)

-- Statement of the theorem
theorem smallest_m_Rn_eq_l :
  ∃ m : ℕ, m > 0 ∧ (Rn (Real.atan (11 / 25)) m) % (2 * Real.pi) = Real.atan (11 / 25) :=
begin
  use 180,
  -- Proof omitted
  sorry
end

end smallest_m_Rn_eq_l_l254_254380


namespace remainder_of_large_product_mod_17_l254_254092

theorem remainder_of_large_product_mod_17 :
  (2011 * 2012 * 2013 * 2014 * 2015) % 17 = 0 := by
  sorry

end remainder_of_large_product_mod_17_l254_254092


namespace real_life_distance_between_cities_l254_254420

variable (map_distance : ℕ)
variable (scale : ℕ)

theorem real_life_distance_between_cities (h1 : map_distance = 45) (h2 : scale = 10) :
  map_distance * scale = 450 :=
sorry

end real_life_distance_between_cities_l254_254420


namespace intersection_of_M_and_N_l254_254635

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254635


namespace intersection_of_M_and_N_l254_254742

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254742


namespace circle_equation_l254_254518

theorem circle_equation (a : ℝ) (h1 : a < 0) :
    (x : ℝ) (y : ℝ) :
    ( (x + 5)^2 + y^2 = 5 ) :=
begin
  -- Conditions
  -- h1: center of the circle at (a, 0) with a < 0
  have center_x := a,
  
  -- h2: radius of the circle is sqrt(5)
  have h2 : real.sqrt 5 = abs a / real.sqrt (1^2 + 2^2),
  
  -- h3: tangent to the line x + 2y = 0 implies abs a = 5
  have h3 : abs a = 5,
  
  -- h4: circle is located on the left side of the y-axis implies a = -5
  have h4 : a = -5,
  
  -- eqn: equation of circle C
  have eqn : (x + 5)^2 + y^2 = 5,
  sorry
end

end circle_equation_l254_254518


namespace distance_P0_P2023_eq_1012_sqrt2_l254_254123

noncomputable def complex_rotation_distance : ℂ :=
  let ω := Complex.exp (Real.pi * Complex.I / 4)
  let path := (Finset.range 2023).sum (λ j, (j + 3) * ω ^ j)
  path + 2

theorem distance_P0_P2023_eq_1012_sqrt2 :
  Complex.abs complex_rotation_distance = 1012 * Real.sqrt 2 :=
sorry

end distance_P0_P2023_eq_1012_sqrt2_l254_254123


namespace sin_R_value_l254_254922

theorem sin_R_value (P Q R : Type) [RightTriangle P Q R (angleQ : Q = 90)] (sin_R cos_R : ℝ)
  (h : 5 * sin_R = 4 * cos_R) : sin_R = 4 * sqrt 41 / 41 :=
by
  sorry

end sin_R_value_l254_254922


namespace non_transit_fully_connected_l254_254304

theorem non_transit_fully_connected (cities : Type) (C : Finset cities) (T : Finset cities) (N : Finset cities)
  (hC_size : C.card = 39)
  (hT_size : T.card = 26)
  (hN_size : N.card = 13)
  (hC_partition : C = T ∪ N)
  (hT_disjoint : T ∩ N = ∅)
  (h_roads : ∀ c ∈ C, ∃ R : Finset cities, (R.card ≥ 21) ∧ (∀ r ∈ R, (r ≠ c) ∧ (r ∈ C)))
  (h_transit : ∀ t ∈ T, ∀ n ∈ N, ¬(∃ r ∈ N, r = t) → (∃ d : Finset cities, d.card ≥ 21 ∧ ∀ r ∈ d, r ≠ t ∧ r ∈ C))
  : ∀ n1 ∈ N, ∀ n2 ∈ N, n1 ≠ n2 → ∃ R : Finset cities, R.card ≥ 1 ∧ ∀ r ∈ R, (r = n2).

end non_transit_fully_connected_l254_254304


namespace postman_speeds_l254_254020

-- Define constants for the problem
def d1 : ℝ := 2 -- distance uphill in km
def d2 : ℝ := 4 -- distance on flat ground in km
def d3 : ℝ := 3 -- distance downhill in km
def time1 : ℝ := 2.267 -- time from A to B in hours
def time2 : ℝ := 2.4 -- time from B to A in hours
def half_time_round_trip : ℝ := 2.317 -- round trip to halfway point in hours

-- Define the speeds
noncomputable def V1 : ℝ := 3 -- speed uphill in km/h
noncomputable def V2 : ℝ := 4 -- speed on flat ground in km/h
noncomputable def V3 : ℝ := 5 -- speed downhill in km/h

-- The mathematically equivalent proof statement
theorem postman_speeds :
  (d1 / V1 + d2 / V2 + d3 / V3 = time1) ∧
  (d3 / V1 + d2 / V2 + d1 / V3 = time2) ∧
  (1 / V1 + 2 / V2 + 1.5 / V3 = half_time_round_trip / 2) :=
by 
  -- Equivalence holds because the speeds satisfy the given conditions
  sorry

end postman_speeds_l254_254020


namespace pentagon_perimeter_l254_254594

-- Problem statement: Given an irregular pentagon with specified side lengths,
-- prove that its perimeter is equal to 52.9 cm.

theorem pentagon_perimeter 
  (a b c d e : ℝ)
  (h1 : a = 5.2)
  (h2 : b = 10.3)
  (h3 : c = 15.8)
  (h4 : d = 8.7)
  (h5 : e = 12.9) 
  : a + b + c + d + e = 52.9 := 
by
  sorry

end pentagon_perimeter_l254_254594


namespace g_value_at_neg_3pi_over_4_l254_254257

noncomputable def f (x : ℝ) := Math.sin (2 * x) - (Real.sqrt 3) * Math.cos (2 * x)
noncomputable def g (x : ℝ) := 2 * Math.sin (2 * x) + 1

theorem g_value_at_neg_3pi_over_4 : g (-3 * Real.pi / 4) = 3 :=
by
  sorry

end g_value_at_neg_3pi_over_4_l254_254257


namespace total_surface_area_of_T_is_correct_l254_254972

open Real

noncomputable def surface_area_of_t (l: ℝ) (a: ℝ) : ℝ :=
  let original_surface_area := 6 * (l^2)
  let removed_triangle_area := 3 * (1 / 2) * (sqrt ((a * a)^2 + (a * a)^2 + (a * a)^2))
  original_surface_area - removed_triangle_area

theorem total_surface_area_of_T_is_correct :
  ∀ (L : ℝ), L = 10 →
  ∀ (A : ℝ), A = 3 →
  surface_area_of_t L A = 600 - 27 * sqrt 2 :=
by
  intros L hL A hA
  subst hL
  subst hA
  rw surface_area_of_t
  sorry

end total_surface_area_of_T_is_correct_l254_254972


namespace meat_sales_beyond_plan_l254_254389

-- Define the constants for each day's sales
def sales_thursday := 210
def sales_friday := 2 * sales_thursday
def sales_saturday := 130
def sales_sunday := sales_saturday / 2
def original_plan := 500

-- Define the total sales
def total_sales := sales_thursday + sales_friday + sales_saturday + sales_sunday

-- Prove that they sold 325kg beyond their original plan
theorem meat_sales_beyond_plan : total_sales - original_plan = 325 :=
by
  -- The proof is not included, so we add sorry to skip the proof
  sorry

end meat_sales_beyond_plan_l254_254389


namespace perimeter_quadrilateral_eq_l254_254089

-- Definitions based on conditions
def EF : ℝ := 15
def HG : ℝ := 6
def FG : ℝ := 20
def is_right_angle (A B C : Type) [HilbertSpace A] : Prop :=
  ∃ (x y z : A), x • y = (1 : ℝ) * z

-- Given parameters from the problem conditions
variables (E F G H : Type) [HilbertSpace E] [HilbertSpace F] [HilbertSpace G] [HilbertSpace H]

axiom EF_perp_FG : is_right_angle E F G
axiom HG_perp_FG : is_right_angle H G F

-- Statement of the theorem
theorem perimeter_quadrilateral_eq :
  (EF + FG + HG + Real.sqrt (9^2 + 20^2)) = 41 + Real.sqrt 481 :=
by
  sorry

end perimeter_quadrilateral_eq_l254_254089


namespace find_YW_length_l254_254977

noncomputable def area_of_triangle (base height : ℝ) : ℝ := 
  (1 / 2) * base * height

theorem find_YW_length (XYZ : Type) [RealTriangle XYZ] 
  (Y Z W : XYZ) 
  (hyp1 : right_angle Y) 
  (hyp2 : diameter_circle_intersects_side YZ XZ W) 
  (hyp3 : area (triangle XYZ) = 195) 
  (hyp4 : side_length XZ = 30) : 
  side_length YW = 13 := 
by 
  sorry

end find_YW_length_l254_254977


namespace kolya_or_leva_l254_254951

theorem kolya_or_leva (k l : ℝ) (hkl : k > 0) (hll : l > 0) : 
  (k > l → ∃ a b c : ℝ, a = l + (2 / 3) * (k - l) ∧ b = (1 / 6) * (k - l) ∧ c = (1 / 6) * (k - l) ∧ a > b + c + l ∧ ¬(a < b + c + a)) ∨ 
  (k ≤ l → ∃ k1 k2 k3 : ℝ, k1 ≥ k2 ∧ k2 ≥ k3 ∧ k = k1 + k2 + k3 ∧ ∃ a' b' c' : ℝ, a' = k1 ∧ b' = (l - k1) / 2 ∧ c' = (l - k1) / 2 ∧ a' + a' > k2 ∧ b' + b' > k3) :=
by sorry

end kolya_or_leva_l254_254951


namespace prime_divisors_constraint_l254_254611

theorem prime_divisors_constraint (p n : ℕ) (hp : p.Prime) (hn : 0 < n)
  (hdiv : p^2 ∣ ∏ k in Finset.range n, (k + 1)^2 + 1) :
  p < 2 * n := 
by
  sorry

end prime_divisors_constraint_l254_254611


namespace smallest_a1_range_l254_254052

noncomputable def smallest_a1_possible (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = (4 : ℝ) / (3 : ℝ) ∧ 
  (∀ i : Fin₈, (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - a_i) > 0) ∧ 
  (-8 < a_1 ∧ a_1 ≤ 1 / 6)

-- Since we are only required to state the problem, we leave the proof as a "sorry".
theorem smallest_a1_range (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
    (h_sum: a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 4 / 3)
    (h_pos: ∀ i : Fin 8, a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - ([-][i]) > 0):
    smallest_a1_possible a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 :=
  by
  sorry

end smallest_a1_range_l254_254052


namespace product_of_factors_equality_l254_254559

theorem product_of_factors_equality :
  (∏ n in Finset.range 11, (1 - (1 / (n + 2)))) = 1 / 12 :=
by
  sorry

end product_of_factors_equality_l254_254559


namespace stockholm_to_uppsala_distance_l254_254421

theorem stockholm_to_uppsala_distance :
  let map_distance_cm : ℝ := 45
  let map_scale_cm_to_km : ℝ := 10
  (map_distance_cm * map_scale_cm_to_km = 450) :=
by
  sorry

end stockholm_to_uppsala_distance_l254_254421


namespace best_play_wins_probability_best_play_wins_with_certainty_l254_254078

-- Define the conditions

variables (n : ℕ)

-- Part (a): Probability that the best play wins
theorem best_play_wins_probability (hn_pos : 0 < n) : 
  1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) = 1 - (Nat.factorial n * Nat.factorial n) / (Nat.factorial (2 * n)) :=
  by sorry

-- Part (b): With more than two plays, the best play wins with certainty
theorem best_play_wins_with_certainty (s : ℕ) (hs : 2 < s) : 
  1 = 1 :=
  by sorry

end best_play_wins_probability_best_play_wins_with_certainty_l254_254078


namespace middle_term_is_correct_l254_254931

noncomputable def middle_term_in_expansion_of_polynomial : Polynomial ℝ :=
  let a := (2 : ℝ)
  let b := (-1 : ℝ)
  let x := Polynomial.X
  (Polynomial.C (-160) * x ^ 3)

theorem middle_term_is_correct :
  (middle_term_in_expansion_of_polynomial) = 
  Polynomial.of_coefficients [(0,-160), (3,1)] :=
  sorry

end middle_term_is_correct_l254_254931


namespace g_of_neg_two_g_of_three_l254_254376

def g (x : ℝ) : ℝ :=
  if x < 0 then 4 * x ^ 2 + 3 else 5 - 3 * x ^ 2

theorem g_of_neg_two : g (-2) = 19 := by
  unfold g
  simp [(by norm_num : -2 < 0)]
  norm_num
  sorry

theorem g_of_three : g 3 = -22 := by
  unfold g
  simp [(by norm_num : 0 ≤ 3)]
  norm_num
  sorry

end g_of_neg_two_g_of_three_l254_254376


namespace lattice_points_hyperbola_count_l254_254277

theorem lattice_points_hyperbola_count : 
  {p : ℤ × ℤ | p.fst^2 - p.snd^2 = 1800^2}.to_finset.card = 150 :=
sorry

end lattice_points_hyperbola_count_l254_254277


namespace sum_of_perpendiculars_constant_l254_254533

theorem sum_of_perpendiculars_constant (r : ℝ) (P : ℝ × ℝ) 
  (h : ∀ (x : ℝ × ℝ), x ∈ inside_pentagon ⟹ is_perpendicular x (sides x)) : 
  ∃ c : ℝ, ∀ (P : ℝ × ℝ), sum_of_perpendiculars P (sides P) = c :=
by
  let n := 5
  let cos36 := real.cos (real.pi / 5)
  exact ⟨5 * r * cos36, sorry⟩

end sum_of_perpendiculars_constant_l254_254533


namespace intersection_of_M_and_N_l254_254788

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254788


namespace intersection_M_N_l254_254760

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254760


namespace orthographic_projection_area_l254_254617

theorem orthographic_projection_area (s : ℝ) (h : s = 1) : 
  let S := (Real.sqrt 3) / 4 
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  S' = (Real.sqrt 6) / 16 :=
by
  let S := (Real.sqrt 3) / 4
  let factor := (Real.sqrt 2) / 2
  let S' := (factor ^ 2) * S
  sorry

end orthographic_projection_area_l254_254617


namespace problem_statement_l254_254966

noncomputable def floor_sum (n : ℕ) (x : ℝ) : ℝ :=
(∑ k in finset.range n.succ, ⌊(k : ℕ) * x⌋) + 
(∑ k in finset.range (⌊n*x⌋).succ, ⌊k / x⌋)

theorem problem_statement
  (n : ℕ) (n_pos : 0 < n) 
  (x : ℝ) (x_pos : 0 < x)
  (h1 : ∀ k : ℕ, 1 ≤ k → k ≤ n → ¬ (k * x).isInt)
  (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ ⌊n * x⌋ → ¬ (k / x).isInt) :
  floor_sum n x = n * ⌊n * x⌋ :=
by {
  sorry
}

end problem_statement_l254_254966


namespace range_of_k_l254_254855

-- Define functions f and g
def f (x : ℝ) : ℝ := (real.exp 2 * x^2 + 1) / x
def g (x : ℝ) : ℝ := (real.exp 2 * x^2) / (real.exp x)

-- Define the condition on k
theorem range_of_k (k : ℝ) (h : 0 < k) :
  (∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → g x1 / k ≤ f x2 / (k + 1)) →
  k ≥ 4 / (2 * real.exp 1 - 4) :=
by
  sorry

end range_of_k_l254_254855


namespace compute_radius_of_circumcircle_l254_254961

variables (ABC : Triangle) (Γ : Circle)
variables (A B C E N A' V : Point)
variables (EV VA' A'N : ℝ)

-- Defining conditions
def conditions := 
  Triangle.isAcute ABC ∧
  Circle.circumcircle ABC Γ ∧
  AngleBisector.intersection (BAC) E ∧
  AngleBisector.intersection (BAC) N ∧
  Point.antipode A' A Γ ∧
  Line.segment (A'A) BC V ∧
  EV = 6 ∧
  VA' = 7 ∧
  A'N = 9

-- Radius of the circumcircle
def radius_of_Γ := Γ.radius

-- The proof statement
theorem compute_radius_of_circumcircle (h : conditions ABC Γ A B C E N A' V EV VA' A'N):
  radius_of_Γ Γ = 15 / 2 :=
by sorry

end compute_radius_of_circumcircle_l254_254961


namespace factor_of_marks_change_l254_254413

theorem factor_of_marks_change {n : ℕ} (initial_avg new_avg : ℝ) (h1 : n = 25) (h2 : initial_avg = 70) (h3 : new_avg = 140) :
  let initial_total := n * initial_avg in
  let new_total := n * new_avg in
  new_total / initial_total = 2 :=
by
  sorry

end factor_of_marks_change_l254_254413


namespace third_place_prize_l254_254349

-- Definitions based on conditions
def num_people := 1 + 7  -- Josh and his 7 friends
def contribution_per_person := 5
def total_pot := num_people * contribution_per_person

def first_place_share_percentage := 0.8
def first_place_share := first_place_share_percentage * total_pot

def remaining_pot := total_pot - first_place_share
def second_third_split := remaining_pot / 2

-- Theorem stating the amount of money third place gets
theorem third_place_prize : second_third_split = 4 :=
by
  sorry

end third_place_prize_l254_254349


namespace midpoint_polar_coordinates_l254_254312

theorem midpoint_polar_coordinates :
  let A := (10 : ℝ, π / 6)
  let B := (10 : ℝ, 11 * π / 6)
  ∃ (r θ : ℝ), 0 < r ∧ 0 ≤ θ ∧ θ < 2 * π ∧ (r, θ) = (10, 0) :=
by
  have dec1 : 0 < 10 := by norm_num
  have dec2 : 0 ≤ 0 := by norm_num
  have dec3 : 0 < 2 * π := by norm_num
  existsi 10
  existsi 0
  split
  · exact dec1
  split
  · exact dec2
  split
  · exact dec3
  · refl
  sorry

end midpoint_polar_coordinates_l254_254312


namespace product_of_chords_l254_254967

theorem product_of_chords (r : ℝ) (h : r = 3) :
    let ω := complex.exp (2 * real.pi * complex.I / 10)
    let D := λ (k : ℕ), r * ω^k
    (∏ i in finset.range 1 5, complex.abs (r * (1 - ω^i))) * (∏ i in finset.range 6 10, complex.abs (r * (1 - ω^i))) = 65610 :=
by
  sorry

end product_of_chords_l254_254967


namespace intersection_M_N_l254_254670

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254670


namespace four_segments_coplanar_l254_254489

open_locale classical

-- Definitions for conditions
variable (P : Type) [plane_geometry P]

variables (A B C D E : P)
variables (a b c : line P)
variables (l1 l2 l3 l4 : segment P)

-- Condition stating that the four line segments are joined end to end
def end_to_end (l1 l2 l3 l4 : segment P) : Prop :=
  endpoint (l1) = initial_point (l2) ∧
  endpoint (l2) = initial_point (l3) ∧
  endpoint (l3) = initial_point (l4)

-- The proof problem statement
theorem four_segments_coplanar
  (h : end_to_end l1 l2 l3 l4) : coplanar {l1, l2, l3, l4} :=
sorry

end four_segments_coplanar_l254_254489


namespace intersection_eq_l254_254699

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254699


namespace problem1_problem2_l254_254183

-- Problem 1
theorem problem1 :
  (1 : ℝ) * (2 * Real.sqrt 12 - (1 / 2) * Real.sqrt 18) - (Real.sqrt 75 - (1 / 4) * Real.sqrt 32)
  = -Real.sqrt 3 - (Real.sqrt 2) / 2 :=
by
  sorry

-- Problem 2
theorem problem2 :
  (2 : ℝ) * (Real.sqrt 5 + 2) * (Real.sqrt 5 - 2) + Real.sqrt 48 / (2 * Real.sqrt (1 / 2)) - Real.sqrt 30 / Real.sqrt 5
  = 1 + Real.sqrt 6 :=
by
  sorry

end problem1_problem2_l254_254183


namespace g_sum_1_2_3_2_l254_254358

def g (a b : ℚ) : ℚ :=
  if a + b ≤ 4 then
    (a * b + a - 3) / (3 * a)
  else
    (a * b + b + 3) / (-3 * b)

theorem g_sum_1_2_3_2 : g 1 2 + g 3 2 = -11 / 6 :=
by sorry

end g_sum_1_2_3_2_l254_254358


namespace max_distance_from_circle_to_line_l254_254971

noncomputable def maximum_distance : ℝ := 3 * Real.sqrt 2

theorem max_distance_from_circle_to_line :
  ∀ (A : ℝ × ℝ), (A.1 - 2)^2 + (A.2 - 2)^2 = 2 → 
  ∃ (d : ℝ), d = 3 * Real.sqrt 2 ∧
    ∀ (P : ℝ × ℝ), (P.1 - A.1)^2 + (P.2 - A.2)^2 = 0 → 
    (| P.1 - P.2 - 4 |) / (Real.sqrt 2) ≤ d :=
begin
  sorry
end

end max_distance_from_circle_to_line_l254_254971


namespace find_coordinates_C_l254_254536

-- Definitions of the given points and condition
def A : ℝ × ℝ := (1, -3)
def B : ℝ × ℝ := (11, 3)

-- BC is half of AB
def BC : ℝ × ℝ := (1 / 2 * (B.1 - A.1), 1 / 2 * (B.2 - A.2))
def C : ℝ × ℝ := (B.1 + BC.1, B.2 + BC.2)

-- Proof statement
theorem find_coordinates_C : C = (16, 6) :=
sorry

end find_coordinates_C_l254_254536


namespace max_sin_cos_l254_254575

theorem max_sin_cos : ∀ x : ℝ, (sin x - real.sqrt 3 * cos x) ≤ 2 ∧ ∃ x : ℝ, (sin x - real.sqrt 3 * cos x) = 2 := by
  sorry

end max_sin_cos_l254_254575


namespace parabola_points_count_l254_254529

theorem parabola_points_count :
  ∃ (points : set (ℤ × ℤ)),
    (∀ p ∈ points, ∃ x y, p = (x, y) ∧ 
      (y - 0 = (x^2 - (6*6 + 2*2))/20)) ∧ -- Parabola condition derived from the focus
    ∀ p ∈ points, |(3 * (p.1) + 2 * (p.2))| ≤ 1200 ∧
    points.finite ∧ 
    points.card = 241 :=
sorry

end parabola_points_count_l254_254529


namespace find_m_l254_254190

theorem find_m :
  ∃ (a : ℕ → ℕ) (m : ℕ), 
    (strict_mono a → (∀ i < m, a i < a (i + 1) )) → m = 140 ∧ 
    ((2^300 + 1) / (2^20 + 1) = (∑ i in Finset.range(m), 2^(a i))) :=
begin
  sorry
end

end find_m_l254_254190


namespace hexagonal_pyramid_ratio_l254_254314

noncomputable def ratio_of_radii {R r : ℝ} (h1 : 2 * r + R) (h2 : R = r * (1 + (Real.sqrt 21)/3)) : ℝ :=
R / r

theorem hexagonal_pyramid_ratio :
  ∀ (R r : ℝ), 
  (h : 2 * r + R = h) ∧ (r > 0) ∧ (R > 0) ∧ (R = r * (1 + (Real.sqrt 21)/3)) →
  ratio_of_radii := sorry

end hexagonal_pyramid_ratio_l254_254314


namespace intersection_M_N_l254_254805

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254805


namespace smallest_value_range_l254_254042

theorem smallest_value_range {a : Fin 8 → ℝ}
  (h_sum : (∑ i, a i) = 4/3)
  (h_pos_7 : ∀ i : Fin 8, (∑ j in Finset.erase Finset.univ i, a j) > 0) :
  -8 < a 0 ∧ a 0 ≤ 1/6 :=
sorry

end smallest_value_range_l254_254042


namespace range_of_f_t_l254_254848

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x) / (Real.exp x) + Real.log x - x

theorem range_of_f_t (a : ℝ) (t : ℝ) 
  (h_unique_critical : ∀ x, f a x = 0 → x = t) : 
  ∃ y : ℝ, y ≥ -2 ∧ ∀ z : ℝ, y = f a t :=
sorry

end range_of_f_t_l254_254848


namespace total_pictures_debby_vacation_l254_254579

noncomputable def initial_pictures : Nat := 250 + 70 + 180 + 90 + 150
def zoo_deletion : Nat := (21 * 250) / 100
def botanical_deletion : Nat := (33 * 70) / 100
def amusement_deletion : Nat := (10 * 150) / 100
def remaining_zoo : Nat := 250 - zoo_deletion
def remaining_botanical : Nat := 70 - botanical_deletion
def remaining_amusement : Nat := 150 - amusement_deletion
def remaining_aquarium : Nat := 180
def remaining_museum : Nat := 90
def additional_museum : Nat := 15
def additional_street_festival : Nat := 12

def total_pictures_after : Nat :=
  remaining_zoo + remaining_botanical + remaining_aquarium +
  (remaining_museum + additional_museum) + remaining_amusement +
  additional_street_festival

theorem total_pictures_debby_vacation : total_pictures_after = 676 :=
by
  have h1 : remaining_zoo = 197 := by calc
    remaining_zoo = 250 - zoo_deletion : rfl
    ... = 250 - 53 : rfl
    ... = 197 : rfl
  have h2 : remaining_botanical = 47 := by calc
    remaining_botanical = 70 - botanical_deletion : rfl
    ... = 70 - 23 : rfl
    ... = 47 : rfl
  have h3 : remaining_amusement = 135 := by calc
    remaining_amusement = 150 - amusement_deletion : rfl
    ... = 150 - 15 : rfl
    ... = 135 : rfl
  have h4 : remaining_museum + additional_museum = 105 := by calc
    remaining_museum + additional_museum = 90 + 15 : rfl
    ... = 105 : rfl
  have total : total_pictures_after = 197 + 47 + 180 + 105 + 135 + 12 := by 
    calc total_pictures_after = remaining_zoo + remaining_botanical + remaining_aquarium + (remaining_museum + additional_museum) + remaining_amusement + additional_street_festival : rfl
    ... = 197 + 47 + 180 + 105 + 135 + 12 : rfl
  exact calc
    total_pictures_after = 197 + 47 + 180 + 105 + 135 + 12 : total
    ... = 676 : by norm_num

end total_pictures_debby_vacation_l254_254579


namespace intersection_M_N_l254_254783

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254783


namespace tangent_intersection_locus_l254_254857

-- Define the problem parameters
def parabola : ℝ → ℝ := λ x, x^2
def ellipse (x y : ℝ) : Prop := (x - 3)^2 + 4 * y^2 = 1

-- Define the focus of the parabola
def focus : ℝ × ℝ := (0, 1 / 4)

-- Define the point A on the ellipse
def point_A (θ : ℝ) : ℝ × ℝ := (3 + Real.cos θ, 1 / 2 * Real.sin θ)

-- Define the locus of the intersection of tangents (track of point M)
def locus (x : ℝ) : Prop := 
  x ≥ (-3 - Real.sqrt 33) / 64 ∧ x ≤ (-3 + Real.sqrt 33) / 64 ∧ y = -1 / 4

-- The Lean statement proving the locus of the intersection points of the tangents
theorem tangent_intersection_locus (x y : ℝ) (θ : ℝ)
  (hA : ellipse (point_A θ).fst (point_A θ).snd) :
  locus x y :=
sorry

end tangent_intersection_locus_l254_254857


namespace ratio_of_x_to_y_l254_254484

theorem ratio_of_x_to_y (x y : ℚ) (h : (8 * x - 5 * y) / (11 * x - 3 * y) = 2 / 7) : 
  x / y = 29 / 34 :=
sorry

end ratio_of_x_to_y_l254_254484


namespace range_a_inequality_l254_254194

theorem range_a_inequality (a : ℝ) : (∀ x : ℝ, (a-2) * x^2 + 4 * (a-2) * x - 4 < 0) ↔ 1 < a ∧ a ≤ 2 :=
by {
    sorry
}

end range_a_inequality_l254_254194


namespace intersection_eq_l254_254700

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254700


namespace a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l254_254235

-- Definitions given in the conditions
variables {a b : ℝ}
variables (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0)

-- Math proof problem in Lean 4
theorem a_gt_b_iff_one_over_a_lt_one_over_b_is_false (a b : ℝ) (a_non_zero : a ≠ 0) (b_non_zero : b ≠ 0) :
  (a > b) ↔ (1 / a < 1 / b) = false :=
sorry

end a_gt_b_iff_one_over_a_lt_one_over_b_is_false_l254_254235


namespace find_a_l254_254847

def f (x : ℝ) : ℝ :=
  if x >= 5 then x^2 - x + 12 else 2^x

theorem find_a (a : ℝ) : f (f a) = 16 → a = 2 := by
  sorry

end find_a_l254_254847


namespace yuko_moves_greater_than_yuri_l254_254103

theorem yuko_moves_greater_than_yuri (X Y : ℕ) : 2 + 4 + 5 + 6 = 17 → 1 + 5 = 6 → X + Y > 11 → 6 + X + Y > 17 := 
by {
    intros h1 h2 h3,
    calc
    6 + X + Y = 6 + (X + Y) : by ring
    ... > 6 + 11 : by linarith
    ... = 17 : by ring
}

end yuko_moves_greater_than_yuri_l254_254103


namespace area_ADE_l254_254337

open Real

variables {A B C D E : Type} [point_data : HasPoints A B C D E]

axiom isosceles_triangle (triangle : ABC) : A ≠ B ∧ B ≠ C ∧ C ≠ A ∧
  (∃ (s : ℝ), s = AB ∧ s = AC)

axiom angle_BAC (angle : ABC) : ∠ BAC = 80

axiom area_ABC (triangle : ABC) : area ABC = 30

axiom trisecting_rays (points : D E) : trisect ∠ BAC D E

theorem area_ADE (triangle : ADE) : area ADE = 10 := 
sorry

end area_ADE_l254_254337


namespace intersection_M_N_l254_254766

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254766


namespace intersection_eq_l254_254706

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254706


namespace intersection_of_M_and_N_l254_254634

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254634


namespace Mateen_garden_area_l254_254458

theorem Mateen_garden_area :
  ∀ (L W : ℝ), (50 * L = 2000) ∧ (20 * (2 * L + 2 * W) = 2000) → (L * W = 400) :=
by
  intros L W h
  -- We have two conditions based on the problem:
  -- 1. Mateen must walk the length 50 times to cover 2000 meters.
  -- 2. Mateen must walk the perimeter 20 times to cover 2000 meters.
  have h1 : 50 * L = 2000 := h.1
  have h2 : 20 * (2 * L + 2 * W) = 2000 := h.2
  -- We can use these conditions to derive the area of the garden
  sorry

end Mateen_garden_area_l254_254458


namespace smallest_integer_solution_l254_254205

open Int

theorem smallest_integer_solution :
  ∃ x : ℤ, (⌊ (x : ℚ) / 8 ⌋ - ⌊ (x : ℚ) / 40 ⌋ + ⌊ (x : ℚ) / 240 ⌋ = 210) ∧ x = 2016 :=
by
  sorry

end smallest_integer_solution_l254_254205


namespace employee_paid_correct_amount_l254_254498

theorem employee_paid_correct_amount:
  ∀ (wholesale_cost : ℝ) (retail_markup_percent : ℝ) (employee_discount_percent : ℝ),
  wholesale_cost = 200 ∧ retail_markup_percent = 20 ∧ employee_discount_percent = 20 →
  let retail_price := wholesale_cost * (1 + retail_markup_percent / 100) in
  let employee_price := retail_price * (1 - employee_discount_percent / 100) in
  employee_price = 192 :=
by
  sorry

end employee_paid_correct_amount_l254_254498


namespace intersection_M_N_l254_254715

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254715


namespace line_passes_through_fixed_point_max_distance_from_center_to_line_line_intersects_circle_l254_254263

-- Define the line equation
def line (k : ℝ) : ℝ → ℝ := λ x, k * x + 3 * k + 1

-- Define the circle equation
def circle (x y : ℝ) : Prop := x^2 + y^2 = 16

-- Prove that the line always passes through the point (-3, 1)
theorem line_passes_through_fixed_point (k : ℝ) : line k (-3) = 1 := by
  -- Proof steps are skipped
  sorry

-- Calculate the maximum distance from the center of the circle to the line
def distance_center_line (k : ℝ) : ℝ :=
  let a := k
  let b := -1
  let c := 3 * k + 1
  abs (c / Real.sqrt (a^2 + b^2))

theorem max_distance_from_center_to_line (k : ℝ) : distance_center_line k = Real.sqrt 10 := by
  -- Proof steps are skipped
  sorry

-- Prove the line intersects with the circle
theorem line_intersects_circle (k : ℝ) : ∃ x y : ℝ, circle x y ∧ (y = line k x) := by
  -- Proof steps are skipped
  sorry

end line_passes_through_fixed_point_max_distance_from_center_to_line_line_intersects_circle_l254_254263


namespace jennifer_money_left_l254_254345

def money_left (initial_amount sandwich_fraction museum_fraction book_fraction : ℚ) : ℚ :=
  initial_amount - (initial_amount * sandwich_fraction + initial_amount * museum_fraction + initial_amount * book_fraction)

theorem jennifer_money_left :
  money_left 150 (1/5) (1/6) (1/2) = 20 := by
  -- Proof goes here
  sorry

end jennifer_money_left_l254_254345


namespace vinegar_solution_example_l254_254288

theorem vinegar_solution_example :
  ∃ x : ℝ, x ≈ 12 ∧ 
  0.3616666666666664 * x = 0.07 * (x + 50) :=
sorry

end vinegar_solution_example_l254_254288


namespace permutations_count_correct_l254_254525

noncomputable def number_of_valid_permutations : Nat :=
  let digits := [1, 1, 3, 3, 6, 8]
  let choose_digit := λ d : Nat => d > 5
  let valid_start_digits := digits.filter choose_digit
  let multiset_permutations (lst : List Nat) : Nat := 
    let counts := lst.foldr (λ x m, m.insert x (m.findD x 0 + 1)) Std.Data.HashMap.empty
    list_permutations (lst.length) (counts.values)
  valid_start_digits.length * multiset_permutations [1, 1, 3, 3, 8]

theorem permutations_count_correct : number_of_valid_permutations = 60 := sorry

end permutations_count_correct_l254_254525


namespace max_min_A_of_coprime_36_and_greater_than_7777777_l254_254173

theorem max_min_A_of_coprime_36_and_greater_than_7777777 
  (A B : ℕ) 
  (h1 : coprime B 36) 
  (h2 : B > 7777777)
  (h3 : A = 10^7 * (B % 10) + (B / 10)) 
  (h4 : (99999999 : ℕ)) 
  : A = 99999998 ∨ A = 17777779 :=
sorry

end max_min_A_of_coprime_36_and_greater_than_7777777_l254_254173


namespace red_bowling_balls_count_l254_254010

theorem red_bowling_balls_count (G R : ℕ) (h1 : G = R + 6) (h2 : R + G = 66) : R = 30 :=
by
  sorry

end red_bowling_balls_count_l254_254010


namespace bricks_needed_l254_254503

-- Define the dimensions of the brick in cm
def brick_length : ℝ := 125
def brick_width : ℝ := 11.25
def brick_height : ℝ := 6

-- Define the dimensions of the wall in cm
def wall_length : ℝ := 800
def wall_height : ℝ := 600
def wall_thickness : ℝ := 22.5

-- Calculate the volume of the brick and the wall
def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_height * wall_thickness

-- Define the number of bricks required
def number_of_bricks := wall_volume / brick_volume

theorem bricks_needed : number_of_bricks ≈ 1280 := 
by 
  simp [number_of_bricks, wall_volume, brick_volume, wall_length, wall_height, wall_thickness, brick_length, brick_width, brick_height]
  sorry

end bricks_needed_l254_254503


namespace students_not_made_the_cut_l254_254067

-- Define the constants for the number of girls, boys, and students called back
def girls := 17
def boys := 32
def called_back := 10

-- Total number of students trying out for the team
def total_try_out := girls + boys

-- Number of students who didn't make the cut
def not_made_the_cut := total_try_out - called_back

-- The theorem to be proved
theorem students_not_made_the_cut : not_made_the_cut = 39 := by
  -- Adding the proof is not required, so we use sorry
  sorry

end students_not_made_the_cut_l254_254067


namespace mischievous_walks_even_l254_254317

open Finset

-- Define the context of the problem

def isFriend (G : SimpleGraph V) (u v : V) : Prop := G.Adj u v

def isMischievousWalk (G : SimpleGraph V) (walk : List V) : Prop :=
  ∃ (n : ℕ), n = 2022 ∧
  (walk.length = 2023) ∧
  (G.degree walk.head % 2 = 1) ∧
  (∀ i, i < 2022 → G.Adj (walk.nth i) (walk.nth (i + 1))) ∧
  (G.degree (walk.last) % 2 = 0)

noncomputable def targetConstruct (G : SimpleGraph V) : ℕ :=
  (universe Fin).sum (λ walk, if isMischievousWalk G walk then 1 else 0)

theorem mischievous_walks_even (G : SimpleGraph V) :
  targetConstruct G % 2 = 0 :=
sorry

end mischievous_walks_even_l254_254317


namespace paint_gallons_l254_254523

theorem paint_gallons (W B : ℕ) (h1 : 5 * B = 8 * W) (h2 : W + B = 6689) : B = 4116 :=
by
  sorry

end paint_gallons_l254_254523


namespace k_value_l254_254504

theorem k_value (k m : ℤ) (h : (m - 8) ∣ (m^2 - k * m - 24)) : k = 5 := by
  have : (m - 8) ∣ (m^2 - 8 * m - 24) := sorry
  sorry

end k_value_l254_254504


namespace diana_garden_area_l254_254195

def number_of_fence_posts := 24
def distance_between_posts := 3
def posts_shorter_side := 4
def posts_longer_side := 6
def short_side_length := (posts_shorter_side - 1) * distance_between_posts
def long_side_length := (posts_longer_side - 1) * distance_between_posts
def garden_area := short_side_length * long_side_length

theorem diana_garden_area : garden_area = 135 := by
  -- Given conditions:
  have h1 : number_of_fence_posts = 24 := rfl
  have h2 : distance_between_posts = 3 := rfl
  have h3 : posts_shorter_side = 4 := rfl
  have h4 : posts_longer_side = 6 := rfl
  have h5 : short_side_length = (posts_shorter_side - 1) * distance_between_posts := rfl
  have h6 : long_side_length = (posts_longer_side - 1) * distance_between_posts := rfl
  have h7 : garden_area = short_side_length * long_side_length := rfl
  
  -- Now prove the required statement:
  rw [h3, h4, h5, h6, h7],
  norm_num,
  sorry

end diana_garden_area_l254_254195


namespace largest_proportion_triple_berry_pie_l254_254209

theorem largest_proportion_triple_berry_pie:
  (∀ (strawberries raspberries certain_fruit : ℕ),
   strawberries + raspberries + certain_fruit = 6 →
   strawberries * 3 = raspberries * 1 →
   raspberries * 3 = certain_fruit * 2 →
   certain_fruit = 3) :=
begin
  intros strawberries raspberries certain_fruit h_sum h_ratio1 h_ratio2,
  sorry
end

end largest_proportion_triple_berry_pie_l254_254209


namespace probability_of_best_performance_winning_best_performance_wins_more_than_two_l254_254076

def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def choose (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))

theorem probability_of_best_performance_winning (n : ℕ) :
  1 - (choose (2 * n) n) / (factorial (2 * n)) =
  1 - ((factorial n) * (factorial n)) / (factorial (2 * n)) := 
by sorry

theorem best_performance_wins_more_than_two (n s : ℕ) (h : s > 2) : 
  1 = 1 := 
by sorry

end probability_of_best_performance_winning_best_performance_wins_more_than_two_l254_254076


namespace train_crosses_bridge_in_time_l254_254867

/-- Define the various constants as given in the problem -/
def length_tr : ℝ := 250
def speed_kmhr : ℝ := 95
def length_br : ℝ := 355

/-- Convert speed from km/hr to m/s -/
noncomputable def speed_ms : ℝ := speed_kmhr * 1000 / 3600

/-- Calculate total distance to clear the bridge -/
def total_distance : ℝ := length_tr + length_br

/-- Calculate time to cross the bridge -/
noncomputable def time_to_cross : ℝ := total_distance / speed_ms

/-- Main theorem statement -/
theorem train_crosses_bridge_in_time : 
  time_to_cross ≈ 22.92 := by
  sorry

end train_crosses_bridge_in_time_l254_254867


namespace probability_of_best_performance_winning_best_performance_wins_more_than_two_l254_254074

def factorial : ℕ → ℕ 
| 0     := 1
| (n+1) := (n+1) * factorial n

noncomputable def choose (n k : ℕ) : ℕ := (factorial n) / ((factorial k) * (factorial (n - k)))

theorem probability_of_best_performance_winning (n : ℕ) :
  1 - (choose (2 * n) n) / (factorial (2 * n)) =
  1 - ((factorial n) * (factorial n)) / (factorial (2 * n)) := 
by sorry

theorem best_performance_wins_more_than_two (n s : ℕ) (h : s > 2) : 
  1 = 1 := 
by sorry

end probability_of_best_performance_winning_best_performance_wins_more_than_two_l254_254074


namespace count_divisors_multiple_of_5_l254_254873

-- Define the conditions as Lean definitions
def prime_factorization (n : ℕ) := 
  n = 2^2 * 3^3 * 5^2

def is_divisor (d : ℕ) (n : ℕ) :=
  d ∣ n

def is_multiple_of_5 (d : ℕ) :=
  ∃ a b c, d = 2^a * 3^b * 5^c ∧ 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2

-- The theorem to be proven
theorem count_divisors_multiple_of_5 (h: prime_factorization 5400) : 
  {d : ℕ | is_divisor d 5400 ∧ is_multiple_of_5 d}.to_finset.card = 24 :=
by {
  sorry -- Proof goes here
}

end count_divisors_multiple_of_5_l254_254873


namespace circle_numbers_equal_l254_254391

variable {α β : ℝ}
variable {k : ℕ} (hk : k ≥ 3)
variable {a : fin k → ℝ}

theorem circle_numbers_equal
  (h1 : ∀ (i : fin k), a (i + 1) % k = α * a i + β * a (i + 2) % k)
  (hα : α ≥ 0)
  (hβ : β ≥ 0)
  (hαβ : α + β = 1) :
  ∃ c : ℝ, ∀ i : fin k, a i = c := 
sorry

end circle_numbers_equal_l254_254391


namespace fixed_point_of_line_l254_254386

theorem fixed_point_of_line (k : ℝ) : ∃ (x y : ℝ), x = -2 ∧ y = -5 ∧ (2 * k + 1) * x + (1 - k) * y + 7 - k = 0 :=
by
  use -2
  use -5
  split <;> try { norm_num }
  sorry

end fixed_point_of_line_l254_254386


namespace intersection_M_N_l254_254722

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254722


namespace smallest_value_of_a1_conditions_l254_254038

noncomputable theory

variables {a1 a2 a3 a4 a5 a6 a7 a8 : ℝ}

/-- The smallest value of \(a_1\) when the sum of \(a_1, \ldots, a_8\) is \(4/3\) 
    and the sum of any seven of these numbers is positive. -/
theorem smallest_value_of_a1_conditions 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 / 3)
  (h_sum_seven : ∀ i : {j // j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8}, 
                  0 < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 - a i.val) :
  -8 < a1 ∧ a1 ≤ 1 / 6 :=
sorry

end smallest_value_of_a1_conditions_l254_254038


namespace travel_methods_l254_254868

theorem travel_methods (bus_services : Nat) (train_services : Nat) (ship_services : Nat) :
  bus_services = 8 → train_services = 3 → ship_services = 2 → 
  bus_services + train_services + ship_services = 13 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  rfl

end travel_methods_l254_254868


namespace intersection_M_N_l254_254804

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254804


namespace ratio_of_average_speeds_l254_254502

-- Define the conditions as constants
def distance_ab : ℕ := 510
def distance_ac : ℕ := 300
def time_eddy : ℕ := 3
def time_freddy : ℕ := 4

-- Define the speeds
def speed_eddy := distance_ab / time_eddy
def speed_freddy := distance_ac / time_freddy

-- The ratio calculation and verification function
def speed_ratio (a b : ℕ) : ℕ × ℕ := (a / Nat.gcd a b, b / Nat.gcd a b)

-- Define the main theorem to be proved
theorem ratio_of_average_speeds : speed_ratio speed_eddy speed_freddy = (34, 15) := by
  sorry

end ratio_of_average_speeds_l254_254502


namespace eggs_in_two_boxes_l254_254060

theorem eggs_in_two_boxes (eggs_per_box : ℕ) (number_of_boxes : ℕ) (total_eggs : ℕ) 
  (h1 : eggs_per_box = 3)
  (h2 : number_of_boxes = 2) :
  total_eggs = eggs_per_box * number_of_boxes :=
sorry

end eggs_in_two_boxes_l254_254060


namespace exists_poly_with_properties_l254_254396

theorem exists_poly_with_properties :
  ∃ᶠ n in at_top, ∃ (p : polynomial ℤ), 
    p.degree = n ∧
    (p.leading_coeff : ℝ) < real.pow 3 n ∧
    (∃ roots : finset ℝ, roots.card = n ∧ ∀ x ∈ roots, x ∈ Ioo 0 1 ∧ (∀ y ∈ roots, x ≠ y → is_root p x))
:= 
  sorry

end exists_poly_with_properties_l254_254396


namespace total_money_spent_l254_254581

/-- Erika, Elizabeth, Emma, and Elsa went shopping on Wednesday.
Emma spent $58.
Erika spent $20 more than Emma.
Elsa spent twice as much as Emma.
Elizabeth spent four times as much as Elsa.
Erika received a 10% discount on what she initially spent.
Elizabeth had to pay a 6% tax on her purchases.
Prove that the total amount of money they spent is $736.04.
-/
theorem total_money_spent :
  let emma_spent := 58
  let erika_initial_spent := emma_spent + 20
  let erika_discount := 0.10 * erika_initial_spent
  let erika_final_spent := erika_initial_spent - erika_discount
  let elsa_spent := 2 * emma_spent
  let elizabeth_initial_spent := 4 * elsa_spent
  let elizabeth_tax := 0.06 * elizabeth_initial_spent
  let elizabeth_final_spent := elizabeth_initial_spent + elizabeth_tax
  let total_spent := emma_spent + erika_final_spent + elsa_spent + elizabeth_final_spent
  total_spent = 736.04 := by
  sorry

end total_money_spent_l254_254581


namespace find_lambda_l254_254920

variables (A B C P : ℝ) (λ : ℝ)
variables (h1 : 0 < λ) (h2 : λ < 1)

theorem find_lambda 
    (h3 : A = λ * B) 
    (h4 : (C - P) * B = (A - P) * (B - A)) 
    : λ = (2 - Real.sqrt 2) / 2 :=
sorry

end find_lambda_l254_254920


namespace maximum_numbers_l254_254480

theorem maximum_numbers (S : Finset ℕ) (h1 : ∀ x ∈ S, 1 ≤ x ∧ x ≤ 1000)
                       (h2 : ∀ x y ∈ S, x ≠ y → (x - y).natAbs ≠ 4 ∧ (x - y).natAbs ≠ 5 ∧ (x - y).natAbs ≠ 6) :
  S.card ≤ 400 :=
begin
  sorry
end

end maximum_numbers_l254_254480


namespace intersection_of_M_and_N_l254_254796

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254796


namespace intersection_M_N_l254_254679

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254679


namespace clock1_runs_10_months_longer_l254_254407

noncomputable def battery_a_charge (C_B : ℝ) := 6 * C_B
noncomputable def clock1_total_charge (C_B : ℝ) := 2 * battery_a_charge C_B
noncomputable def clock2_total_charge (C_B : ℝ) := 2 * C_B
noncomputable def clock2_operating_time := 2
noncomputable def clock1_operating_time (C_B : ℝ) := clock1_total_charge C_B / C_B
noncomputable def operating_time_difference (C_B : ℝ) := clock1_operating_time C_B - clock2_operating_time

theorem clock1_runs_10_months_longer (C_B : ℝ) :
  operating_time_difference C_B = 10 :=
by
  unfold operating_time_difference clock1_operating_time clock2_operating_time clock1_total_charge battery_a_charge
  sorry

end clock1_runs_10_months_longer_l254_254407


namespace decreasing_interval_l254_254200

def t (x : ℝ) : ℝ := 2*x^2 - 3*x + 1 

def log_base_half (t : ℝ) : ℝ := Real.log t / Real.log (1/2)

theorem decreasing_interval (x : ℝ) (hx : t x > 0) :
  ∃ a b : ℝ, (1 : ℝ) < a ∧ b = +∞ ∧ ∀ x ∈ Set.Ioc a b, monotone_decreasing (log_base_half ∘ t x) :=
sorry

end decreasing_interval_l254_254200


namespace find_a9_l254_254928

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- conditions
def is_arithmetic_sequence := ∀ n : ℕ, a (n + 1) = a n + d
def given_condition1 := a 5 + a 7 = 16
def given_condition2 := a 3 = 4

-- theorem
theorem find_a9 (h1 : is_arithmetic_sequence a d) (h2 : given_condition1 a) (h3 : given_condition2 a) :
  a 9 = 12 :=
sorry

end find_a9_l254_254928


namespace smallest_number_bounds_l254_254046

theorem smallest_number_bounds (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos : ∀ i, 0 < ∑ j, if j = i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (Finset.univ_nonempty)) a ∧
  (Finset.min' Finset.univ (Finset.univ_nonempty)) a ≤ 1 / 6 :=
begin
  sorry
end

end smallest_number_bounds_l254_254046


namespace cross_section_area_is_correct_l254_254199

noncomputable def area_of_cross_section
  (side_base : ℝ) (distance_plane : ℝ) : ℝ :=
let a := side_base in
if side_base = sqrt 14 ∧ distance_plane = 1 then
  21 / 4
else
  0  -- placeholder for other cases

theorem cross_section_area_is_correct :
  area_of_cross_section (sqrt 14) 1 = 21 / 4 :=
by sorry

end cross_section_area_is_correct_l254_254199


namespace max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254469

def max_distinct_numbers (n : ℕ) (k : ℕ) (f : ℕ → ℕ → Prop) : ℕ :=
  sorry

theorem max_num_distinct_from_1_to_1000_no_diff_4_5_6 :
  max_distinct_numbers 1000 4 (λ a b, ¬(a - b = 4 ∨ a - b = 5 ∨ a - b = 6)) = 400 :=
sorry

end max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254469


namespace giant_slide_wait_is_15_l254_254184

noncomputable def wait_time_for_giant_slide
  (hours_at_carnival : ℕ) 
  (roller_coaster_wait : ℕ)
  (tilt_a_whirl_wait : ℕ)
  (rides_roller_coaster : ℕ)
  (rides_tilt_a_whirl : ℕ)
  (rides_giant_slide : ℕ) : ℕ :=
  
  (hours_at_carnival * 60 - (roller_coaster_wait * rides_roller_coaster + tilt_a_whirl_wait * rides_tilt_a_whirl)) / rides_giant_slide

theorem giant_slide_wait_is_15 :
  wait_time_for_giant_slide 4 30 60 4 1 4 = 15 := 
sorry

end giant_slide_wait_is_15_l254_254184


namespace intersection_of_M_and_N_l254_254636

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254636


namespace general_term_sum_value_l254_254248

-- Definitions from the problem
def S (n : ℕ) : ℕ := n^2
def a (n : ℕ) : ℕ := if n = 1 then 1 else S n - S (n - 1)

-- Property to be proved for part (I)
theorem general_term (n : ℕ) : a n = 2 * n - 1 := by
  sorry

-- Property to be proved for part (II)
theorem sum_value : 
  (∑ k in Finset.range 2016, 1 / (Real.sqrt (a (k + 1)) + Real.sqrt (a (k + 2)))) = 
  1 / 2 * (Real.sqrt 4033 - 1) := by 
  sorry

end general_term_sum_value_l254_254248


namespace dividend_percentage_shares_l254_254528

theorem dividend_percentage_shares :
  ∀ (purchase_price market_value : ℝ) (interest_rate : ℝ),
  purchase_price = 56 →
  market_value = 42 →
  interest_rate = 0.12 →
  ( (interest_rate * purchase_price) / market_value * 100 = 16) :=
by
  intros purchase_price market_value interest_rate h1 h2 h3
  rw [h1, h2, h3]
  -- Calculations were done in solution
  sorry

end dividend_percentage_shares_l254_254528


namespace M_inter_N_eq_2_4_l254_254820

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254820


namespace part1_price_light_bulb_motor_part2_minimal_cost_l254_254326

-- Define the conditions
noncomputable def sum_price : ℕ := 12
noncomputable def total_cost_light_bulbs : ℕ := 30
noncomputable def total_cost_motors : ℕ := 45
noncomputable def ratio_light_bulbs_motors : ℕ := 2
noncomputable def total_items : ℕ := 90
noncomputable def max_ratio_light_bulbs_motors : ℕ := 2

-- Statement of the problems
theorem part1_price_light_bulb_motor (x : ℕ) (y : ℕ):
  x + y = sum_price → 
  total_cost_light_bulbs = 30 →
  total_cost_motors = 45 →
  total_cost_light_bulbs / x = ratio_light_bulbs_motors * (total_cost_motors / y) → 
  x = 3 ∧ y = 9 := 
sorry

theorem part2_minimal_cost (m : ℕ) (n : ℕ):
  m + n = total_items →
  m ≤ total_items / max_ratio_light_bulbs_motors →
  let cost := 3 * m + 9 * n in
  (∀ x y, x + y = total_items → x ≤ total_items / max_ratio_light_bulbs_motors → cost ≤ 3 * x + 9 * y) → 
  m = 30 ∧ n = 60 ∧ cost = 630 :=
sorry

end part1_price_light_bulb_motor_part2_minimal_cost_l254_254326


namespace solve_equation_l254_254443

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end solve_equation_l254_254443


namespace identify_max_t_l254_254121

-- Definition of the game conditions
def card_game_conditions :=
    ∃ (cards : List ℝ) (n : ℕ), n = 2013 ∧ ∀ (k : ℕ), List.length cards = n ∧
    (∀ (chosen : Finset ℕ), chosen.card = 10 → ∃ (num : ℝ), num ∈ (chosen.image cards.nthLe)) ∧
    (∀ (cards_i cards_j : ℕ), cards_i < n → cards_j < n → cards_i ≠ cards_j → cards.nthLe cards_i ≠ cards.nthLe cards_j)

-- The main theorem determining the maximum positive integer t such that
-- A can identify the numbers on t cards
theorem identify_max_t :
    card_game_conditions →
    ∃ (t : ℕ), t = 1987 := by
    sorry

end identify_max_t_l254_254121


namespace rug_shorter_side_l254_254143

theorem rug_shorter_side (x : ℝ) :
  (64 - x * 7) / 64 = 0.78125 → x = 2 :=
by
  sorry

end rug_shorter_side_l254_254143


namespace largest_integral_value_l254_254592

theorem largest_integral_value (y : ℤ) (h1 : 0 < y) (h2 : (1 : ℚ)/4 < y / 7) (h3 : y / 7 < 7 / 11) : y = 4 :=
sorry

end largest_integral_value_l254_254592


namespace intersection_M_N_l254_254623

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254623


namespace largest_T_inequality_l254_254859

def k (n : ℕ) : ℚ :=
  n * (n + 1) / 2

def S (n : ℕ) : ℚ :=
  let sum_part_m := fun m => (∑ i in finset.range m, 1 / k (i + 1))
  (∑ j in finset.range n, 1 / (1 + sum_part_m (j + 1)))

theorem largest_T_inequality : S 2002 > 1004 := by
  sorry

end largest_T_inequality_l254_254859


namespace rook_traversal_minimum_M_l254_254146

noncomputable def minimum_value_M (n : ℕ) : ℕ :=
  2 * n - 1

theorem rook_traversal_minimum_M (n : ℕ) (traversal : Fin (n * n) → Fin (n * n)) :
  (∀ i, traversal (traversal i) = traversal i) →
  (∀ (i : Fin (n * n)), 
    (traversal i).val < (n * n)) →
  (∀ (i : Fin (n * n)),
    (abs ((traversal (⟨i.1 + 1, sorry⟩)).val - (traversal i).val) ≤ 
    minimum_value_M n) →
    min { d | ∃ i j, (traversal i, traversal j) ∈ 
      {(traversal (⟨i.1 + 1, sorry⟩), traversal i), (traversal (⟨i.1, sorry⟩), traversal (⟨i.1 + 1, sorry⟩))}
    d } = minimum_value_M n :=
sorry

end rook_traversal_minimum_M_l254_254146


namespace root_power_six_eq_512_l254_254557

theorem root_power_six_eq_512 :
  (8^(1 / 2 : ℝ))^6 = 512 := 
by {
  sorry
}

end root_power_six_eq_512_l254_254557


namespace intersection_of_M_and_N_l254_254645

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254645


namespace area_PF1F2_l254_254618

def P : Type := (ℝ × ℝ) -- A point on the plane as a tuple of two reals

def hyperbola (P : ℝ × ℝ) : Prop :=
  P.1^2 - (P.2^2) / 3 = 1

def F1 : P := (-2, 0)
def F2 : P := (2, 0)

def distance (A B : P) : ℝ :=
  ((B.1 - A.1)^2 + (B.2 - A.2)^2).sqrt

def condition (P : P) : Prop :=
  3 * distance P F1 = 4 * distance P F2

theorem area_PF1F2 (P : P) (h1 : hyperbola P) (h2 : condition P) :
  let d1 := distance P F1 in
  let d2 := distance P F2 in
  3 * sqrt 15 = 1/2 * d1 * d2 * (sqrt (1 - ( ((d1^2 + d2^2 - 4^2) / (2 * d1 * d2))^2)) := sorry

end area_PF1F2_l254_254618


namespace intersection_of_M_and_N_l254_254787

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254787


namespace area_of_figM_l254_254363

def figM_area := ∃ (x y a b : ℝ),
  (x - a) ^ 2 + (y - b) ^ 2 ≤ 20 ∧
  a ^ 2 + b ^ 2 ≤ min (8 * a - 4 * b) 20

theorem area_of_figM : 
  let A := 60 * Real.pi - 10 * Real.sqrt 3 in
  ∃ (A : ℝ), A = 60 * Real.pi - 10 * Real.sqrt 3 :=
begin
  sorry
end

end area_of_figM_l254_254363


namespace intersection_eq_l254_254708

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254708


namespace line_intersects_curve_and_length_l254_254937

-- Define the parametric equations of line l
def parametric_line (t : ℝ) : ℝ × ℝ :=
  (-1 + sqrt 2 / 2 * t, sqrt 2 / 2 * t)

-- Define the polar equation of curve C
def polar_equation (ρ θ : ℝ) : Prop :=
  ρ^2 = 12 / (4 * sin θ^2 + 3 * cos θ^2)

-- Define the rectangular equation of curve C
def rectangular_equation (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 3 = 1

-- The main theorem combining all parts
theorem line_intersects_curve_and_length {l : ℝ → ℝ × ℝ}
  (h_l : ∀ t, l t = parametric_line t)
  (h_curve : ∀ ρ θ, polar_equation ρ θ → rectangular_equation (ρ * cos θ) (ρ * sin θ)) :
  (∀ t, snd (l t) = fst (l t) + 1) ∧
  (∀ x y, rectangular_equation x y → polar_equation (sqrt (x^2 + y^2)) (atan2 y x)) ∧
  (∃ t1 t2, t1 ≠ t2 ∧ rectangular_equation (fst (l t1)) (snd (l t1)) ∧ rectangular_equation (fst (l t2)) (snd (l t2)) ∧ abs (t1 - t2) = 24 / 7) :=
by
  intro h_l h_curve
  split
  sorry
  split
  sorry
  use [1, 1] -- placeholder, this should be derived
  split
  sorry
  split
  sorry
  sorry

end line_intersects_curve_and_length_l254_254937


namespace max_min_A_of_coprime_36_and_greater_than_7777777_l254_254172

theorem max_min_A_of_coprime_36_and_greater_than_7777777 
  (A B : ℕ) 
  (h1 : coprime B 36) 
  (h2 : B > 7777777)
  (h3 : A = 10^7 * (B % 10) + (B / 10)) 
  (h4 : (99999999 : ℕ)) 
  : A = 99999998 ∨ A = 17777779 :=
sorry

end max_min_A_of_coprime_36_and_greater_than_7777777_l254_254172


namespace positive_divisors_multiple_of_5_l254_254872

theorem positive_divisors_multiple_of_5 (a b c : ℕ) (h_a : 0 ≤ a ∧ a ≤ 2) (h_b : 0 ≤ b ∧ b ≤ 3) (h_c : 1 ≤ c ∧ c ≤ 2) :
  (a * b * c = 3 * 4 * 2) :=
sorry

end positive_divisors_multiple_of_5_l254_254872


namespace smallest_number_divisible_l254_254113

theorem smallest_number_divisible
  (x : ℕ)
  (h : (x - 2) % 12 = 0 ∧ (x - 2) % 16 = 0 ∧ (x - 2) % 18 = 0 ∧ (x - 2) % 21 = 0 ∧ (x - 2) % 28 = 0) :
  x = 1010 :=
by
  sorry

end smallest_number_divisible_l254_254113


namespace area_of_triangle_l254_254250

-- Define the function to calculate the area of a right isosceles triangle given the side lengths of squares
theorem area_of_triangle (a b c : ℝ) (h1 : a = 10) (h2 : b = 8) (h3 : c = 10) (right_isosceles : true) :
  (1 / 2) * a * c = 50 :=
by
  -- We state the theorem but leave the proof as sorry.
  sorry

end area_of_triangle_l254_254250


namespace smallest_positive_period_range_of_f_on_interval_l254_254864

section
variables {ω : ℝ} {λ : ℝ}
def a (x : ℝ) := (Real.cos (ω * x) - Real.sin (ω * x), Real.sin x)
def b (x : ℝ) := (-Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sqrt 3 * Real.cos (ω * x))
def f (x : ℝ) := (a x).1 * (b x).1 + (a x).2 * (b x).2 + λ

theorem smallest_positive_period 
  (hω : ω > 1/2 ∧ ω < 1)
  (hf_symm : ∀ x : ℝ, f (π + x) = f (π - x)) :
  ∃ T : ℝ, T = 12 * π / 5 ∧ ∀ x : ℝ, f (x + T) = f x := sorry

theorem range_of_f_on_interval
  (hf_at_π_over_4 : f (π / 4) = 0)
  (hω : ω = 5 / 6) :
  ∃ I : set ℝ, I = set.Icc (-1 - Real.sqrt 2) (1 - Real.sqrt 2) ∧ 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 * π / 5 → f x ∈ I := sorry
end

end smallest_positive_period_range_of_f_on_interval_l254_254864


namespace intersection_M_N_l254_254778

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254778


namespace maximum_volume_l254_254152

noncomputable def volume (x : ℝ) : ℝ :=
  (48 - 2*x)^2 * x

theorem maximum_volume :
  (∀ x : ℝ, (0 < x) ∧ (x < 24) → volume x ≤ volume 8) ∧ (volume 8 = 8192) :=
by
  sorry

end maximum_volume_l254_254152


namespace number_satisfies_equation_l254_254995

theorem number_satisfies_equation :
  ∃ x : ℝ, (x^2 + 100 = (x - 20)^2) ∧ x = 7.5 :=
by
  use 7.5
  sorry

end number_satisfies_equation_l254_254995


namespace windmill_area_l254_254175

/--
Suppose we have four identical right-angled triangles arranged in a windmill shape with vertices \(O, A, B, C, D)\).
Let \(E, F, G,\) and \(H\) be midpoints of \(OA, OB, OC,\) and \(OD\) respectively.
Given that the distance \(AC = BD = 20\) cm.
We need to prove that the area of the windmill is equal to 100 square centimeters.
-/
theorem windmill_area (O A B C D E F G H : Point) 
  (AC_len : length AC = 20) (BD_len : length BD = 20)
  (mid_E : midpoint E O A) (mid_F : midpoint F O B) 
  (mid_G : midpoint G O C) (mid_H : midpoint H O D) :
  area_windmill O A B C D E F G H = 100 :=
sorry

end windmill_area_l254_254175


namespace range_of_m_triangle_inequality_l254_254008

noncomputable theory

-- stating that the function f(x)
def f (x : ℝ) := x^2 - 2 * x + 2

-- Given conditions
theorem range_of_m_triangle_inequality :
  ∀ (m : ℝ) (a b c : ℝ),
  (1/3 ≤ a) → (a ≤ m^2 - m + 2) →
  (1/3 ≤ b) → (b ≤ m^2 - m + 2) →
  (1/3 ≤ c) → (c ≤ m^2 - m + 2) →
  a ≠ b → b ≠ c → a ≠ c →
  (f a + f b > f c) ∧ (f a + f c > f b) ∧ (f b + f c > f a) →
  0 ≤ m ∧ m ≤ 1 :=
begin
  sorry
end

end range_of_m_triangle_inequality_l254_254008


namespace increasing_interval_of_f_l254_254019

noncomputable def f : ℝ → ℝ := fun x => Real.exp x - x

theorem increasing_interval_of_f : 
  (fun x => x ∈ {x : ℝ | 0 < x}) = (fun x => f' x = Real.exp x - 1 > 0) :=
by
  sorry

end increasing_interval_of_f_l254_254019


namespace compare_store_costs_l254_254126

-- Define the conditions mathematically
def StoreA_cost (x : ℕ) : ℝ := 5 * x + 125
def StoreB_cost (x : ℕ) : ℝ := 4.5 * x + 135

theorem compare_store_costs (x : ℕ) (h : x ≥ 5) : 
  5 * 15 + 125 = 200 ∧ 4.5 * 15 + 135 = 202.5 ∧ 200 < 202.5 := 
by
  -- Here the theorem states the claims to be proved.
  sorry

end compare_store_costs_l254_254126


namespace fixed_point_l254_254430

noncomputable def func (a : ℝ) (x : ℝ) : ℝ := a^(x-1)

theorem fixed_point (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : func a 1 = 1 :=
by {
  -- We need to prove that func a 1 = 1 for any a > 0 and a ≠ 1
  sorry
}

end fixed_point_l254_254430


namespace num_pos_divisors_30_l254_254885

theorem num_pos_divisors_30 : ∃ n : ℕ, n = 8 ∧ (∀ m : ℕ, m ∣ 30 ↔ m ∈ {1, 2, 3, 5, 6, 10, 15, 30})
 :=
begin
  sorry
end

end num_pos_divisors_30_l254_254885


namespace smallest_n_l254_254267

noncomputable def a : ℕ → ℝ
| 1 => 9
| (n + 1) => (4 - a (n + 1)) / 3

noncomputable def S : ℕ → ℝ
| 1 => a 1
| (n + 1) => S n + a (n + 1)

theorem smallest_n
  (a : ℕ → ℝ)
  (h_recur : ∀ n, n ≥ 1 → 3 * a (n + 1) + a n = 4)
  (h_a1 : a 1 = 9) :
  ∃ n, n ≥ 7 ∧ ∀ m < n, ∃ m < n, ∀ m < n, ∃ m < n, 
  ∀ m < n, ∃ m < n, 
  ∀ m < n, ∃ m < n, 
  ∀ m < n, 
  ∀ m < n, 
  ∀ m < n, 
  ∀ m < n, -- Adding satisfication of
    |S n - n - 6| < 1/125 :=
sorry

end smallest_n_l254_254267


namespace propositions_incorrect_l254_254545

variable {Point : Type}
variable {Line : Type}
variable {Plane : Type}

variable parallel_line : Line -> Line -> Prop
variable parallel_plane : Line -> Plane -> Prop
variable contains : Plane -> Line -> Prop
variable intersection : Plane -> Plane -> Line

theorem propositions_incorrect 
  (l m : Line) (α β : Plane)
  (hm_in_α : contains α m)
  (hl_par_m : parallel_line l m)
  (hl_par_α : parallel_plane l α)
  (hm_par_α : parallel_plane m α)
  (hl_in_β : contains β l)
  (α_int_β_m : intersection α β = m) : 
  ¬(hl_par_m ∧ hm_in_α → parallel_plane l α) ∧
  ¬(hl_par_α ∧ hm_par_α → parallel_line l m) ∧
  ¬(hl_par_m ∧ hm_par_α → parallel_plane l α) ∧
  (hl_par_α ∧ hl_in_β ∧ α_int_β_m → parallel_line l m) := 
sorry

end propositions_incorrect_l254_254545


namespace intersection_M_N_l254_254713

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254713


namespace determine_xy_l254_254835

theorem determine_xy (x y : ℝ) (h : (2 * x - 1) + (1 : ℂ).i = (-y) - (3 - y) * (1 : ℂ).i) : 
  x = -3 / 2 ∧ y = 4 :=
by
  sorry

end determine_xy_l254_254835


namespace Albert_more_than_Joshua_l254_254947

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_than_Joshua_l254_254947


namespace find_q_l254_254894

theorem find_q (p q : ℚ) (h1 : 5 * p + 6 * q = 20) (h2 : 6 * p + 5 * q = 29) : q = -25 / 11 :=
by
  sorry

end find_q_l254_254894


namespace greatest_M_l254_254202

section
variable (a : ℕ → ℝ)

/-- Definition of a positive real sequence -/
def positive_sequence := ∀ n : ℕ, a n > 0

/-- Definition of the condition involving the sum inequality -/
def sum_inequality (M : ℝ) : Prop :=
  ∀ m : ℝ, m < M →
    ∃ n : ℕ, 1 ≤ n ∧ (∑ i in (Finset.range (n + 1)), a i) > m * a n

/-- The theorem statement proving that the greatest M with the above condition is 4. -/
theorem greatest_M : ∀ a : ℕ → ℝ,
  positive_sequence a →
  ∀ M, sum_inequality a M ↔ M = 4 :=
by
  intros
  sorry

end

end greatest_M_l254_254202


namespace min_omega_l254_254853

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * sin (ω * x + φ)

theorem min_omega (ω φ x₀ : ℝ) (h : ω > 0) (hx : f ω φ (x₀ + 2) - f ω φ x₀ = 4) : ω = π / 2 :=
by 
  sorry

end min_omega_l254_254853


namespace bacteria_growth_time_l254_254007
-- Import necessary library

-- Define the conditions
def initial_bacteria_count : ℕ := 100
def final_bacteria_count : ℕ := 102400
def multiplication_factor : ℕ := 4
def multiplication_period_hours : ℕ := 6

-- Define the proof problem
theorem bacteria_growth_time :
  ∃ t : ℕ, t * multiplication_period_hours = 30 ∧ initial_bacteria_count * multiplication_factor^t = final_bacteria_count :=
by
  sorry

end bacteria_growth_time_l254_254007


namespace proof_problem_l254_254406

variable {a : ℕ → ℕ}

-- Given conditions
def is_geometric_sequence (a : ℕ → ℕ) (q > 1) : Prop :=
  ∀ n, a (n + 1) = q * a n

def sum_of_first_three_terms (a : ℕ → ℕ) (S₃ = 7) : Prop :=
  a 1 + a 2 + a 3 = S₃

def forms_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  a 1 + 3 + a 3 + 4 = 6 * a 2

-- Correct answers
def general_term (a : ℕ → ℕ) : Prop :=
  ∀ n, a n = 2 ^ (n - 1)

def bn (a : ℕ → ℕ) (n : ℕ) : ℕ := (n + 1) * (2 ^ n).natAbsLog 2 -- Definition of b_n

def sum_inverse_bn (a : ℕ → ℕ) (n : ℕ) : Prop :=
  (∑ k in finset.range n, (1 / bn a k)) < 1

-- Theorem
theorem proof_problem (q > 1) (S₃ = 7) :
  is_geometric_sequence a q →
  sum_of_first_three_terms a S₃ →
  forms_arithmetic_sequence a →
  general_term a ∧ (∀ n, sum_inverse_bn a n) := by
  sorry

end proof_problem_l254_254406


namespace value_of_f_f_one_fourth_l254_254842

-- Defining the piecewise function f as described
def f (x : ℝ) : ℝ :=
if x > 0 then log x / log 2 else 2 - 2^(-x)

-- The main statement to be proven
theorem value_of_f_f_one_fourth : f(f (1 / 4)) = -2 :=
sorry

end value_of_f_f_one_fourth_l254_254842


namespace M_inter_N_eq_2_4_l254_254824

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254824


namespace polynomial_simplification_l254_254402

/-- The polynomial simplification problem -/
theorem polynomial_simplification :
  (Polynomial.Coeff.1 * Polynomial.X 3 + 4 * Polynomial.X 2 - 7 * Polynomial.X + 11) +
  (-4 * Polynomial.X 4 - Polynomial.X 3 + Polynomial.X 2 + 7 * Polynomial.X + 3) +
  (3 * Polynomial.X 4 - 2 * Polynomial.X 3 + 5 * Polynomial.X - 1) =
  -Polynomial.Coeff.1 * Polynomial.X 4 - 2 * Polynomial.X 3 + 5 * Polynomial.X 2 + 5 * Polynomial.X + 13 := by
  sorry

end polynomial_simplification_l254_254402


namespace caleb_burgers_l254_254107

theorem caleb_burgers
  (S D : ℕ)
  (h1 : 1 * S + 1.5 * D = 68.5)
  (h2 : S + D = 50) :
  D = 37 :=
by
  sorry

end caleb_burgers_l254_254107


namespace hyperbola_and_m_l254_254608

noncomputable def hyperbola_equation : Prop :=
  ∃ (a b : ℝ), a = sqrt 3 ∧ b = 1 ∧ (∀ (x y : ℝ), (x^2 / a^2 - y^2 / b^2 = 1) ↔ (x^2 / 3 - y^2 = 1))

noncomputable def range_of_m : Prop :=
  ∀ (k m : ℝ), k ≠ 0 ∧ m ≠ 0 ∧ (∃ (x y : ℝ), (y = k * x + m) ∧ (x^2 / 3 - y^2 = 1))
    ∧ ((∀ (x1 y1 x2 y2 : ℝ), (y1 = k * x1 + m) ∧ (x1^2 / 3 - y1^2 = 1) ∧ 
        (y2 = k * x2 + m) ∧ (x2^2 / 3 - y2^2 = 1) → 
         let x0 := (x1 + x2) / 2
             y0 := k * x0 + m
         in (y0 = -1 / k)))
    → (m < -1/4 ∨ m > 4)

theorem hyperbola_and_m :
  hyperbola_equation ∧ range_of_m :=
by
  apply And.intro
  {
    use [sqrt 3, 1]
    simp
  }
  {
    unfold range_of_m
    sorry
  }

end hyperbola_and_m_l254_254608


namespace intersection_of_M_and_N_l254_254785

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254785


namespace largest_angle_of_triangle_l254_254155

theorem largest_angle_of_triangle
  (A B C : ℝ)  -- The vertices of the triangle
  (incircle_passes_through_circumcenter : ∀ (O : Point), incircle O ∈ circumcenter O)
  (angle_ABC_is_60 : angle A B C = 60) :
  exists (alpha beta gamma : ℝ), alpha = 83.91 ∧ is_triangle α β γ :=
sorry

end largest_angle_of_triangle_l254_254155


namespace best_play_wins_probability_l254_254082

theorem best_play_wins_probability (n : ℕ) :
  let p := (n! * n!) / (2 * n)! in
  1 - p = 1 - (fact n * fact n / fact (2 * n)) :=
sorry

end best_play_wins_probability_l254_254082


namespace solve_agatha_number_problem_l254_254160

def is_valid_pair (x y : ℕ) : Prop := x = y + 1 ∨ y = x + 1

def valid_pairs_from_subsequences (subseq: list ℕ) : Prop :=
  (subseq.contains 79 ∧ subseq.contains 80 ∧ is_valid_pair 79 80) ∨
  (subseq.contains 19 ∧ subseq.contains 20 ∧ is_valid_pair 19 20)

theorem solve_agatha_number_problem :
  ∃ subseq : list ℕ, subseq ⊆ [1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012, 0123, 1234, 2345, 3456, 4567, 5678, 6789, 7890] ∧ 
  valid_pairs_from_subsequences subseq := sorry

end solve_agatha_number_problem_l254_254160


namespace intersection_eq_l254_254701

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254701


namespace ratio_c_d_l254_254911

theorem ratio_c_d (x y c d : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hd : d ≠ 0) 
  (h1 : 8 * x - 6 * y = c) (h2 : 10 * y - 15 * x = d) : 
  c / d = -4 / 5 :=
by
  sorry

end ratio_c_d_l254_254911


namespace replaced_person_weight_l254_254006

variable (W : ℝ)
variable (avg_weight_increase : ℝ := 2.5)
variable (num_persons : ℕ := 8)
variable (new_person_weight : ℝ := 85)
variable (total_weight_increase : ℝ := avg_weight_increase * num_persons)

theorem replaced_person_weight :
  new_person_weight - W = total_weight_increase → W = 65 :=
by { intro h, rw [total_weight_increase, show new_person_weight = 85, from rfl, show avg_weight_increase = 2.5, from rfl, show num_persons = 8, from rfl] at h, linarith }

end replaced_person_weight_l254_254006


namespace shorter_diagonal_of_trapezoid_l254_254071

theorem shorter_diagonal_of_trapezoid (AB CD AD BC : ℝ) (h1 : AB = 25) (h2 : CD = 15) (h3 : AD = 9) (h4 : BC = 12) (h5 : AB ∥ CD) (h_angleA_obtuse : true) (h_angleB_obtuse : true) : 
  ∃ BD : ℝ, BD = 14 * Real.sqrt 2 := 
by
  sorry

end shorter_diagonal_of_trapezoid_l254_254071


namespace non_transit_fully_connected_l254_254305

theorem non_transit_fully_connected (cities : Type) (C : Finset cities) (T : Finset cities) (N : Finset cities)
  (hC_size : C.card = 39)
  (hT_size : T.card = 26)
  (hN_size : N.card = 13)
  (hC_partition : C = T ∪ N)
  (hT_disjoint : T ∩ N = ∅)
  (h_roads : ∀ c ∈ C, ∃ R : Finset cities, (R.card ≥ 21) ∧ (∀ r ∈ R, (r ≠ c) ∧ (r ∈ C)))
  (h_transit : ∀ t ∈ T, ∀ n ∈ N, ¬(∃ r ∈ N, r = t) → (∃ d : Finset cities, d.card ≥ 21 ∧ ∀ r ∈ d, r ≠ t ∧ r ∈ C))
  : ∀ n1 ∈ N, ∀ n2 ∈ N, n1 ≠ n2 → ∃ R : Finset cities, R.card ≥ 1 ∧ ∀ r ∈ R, (r = n2).

end non_transit_fully_connected_l254_254305


namespace right_triangle_acute_angles_l254_254332

noncomputable def acute_angles (x y : ℝ) (h : real.sqrt 3 * (x + y) = real.sqrt ((x + y) * (x + y))) : 
  Prop :=
let C := real.arctan 2 in
let A := (real.pi / 2) - C in
((C = real.arctan 2) ∧ (A = (real.pi / 2) - real.arctan 2))

theorem right_triangle_acute_angles {x y : ℝ} 
  (h : real.sqrt 3 * (x + y) = real.sqrt ((x + y) * (x + y))) :
  acute_angles x y h :=
by
  sorry

end right_triangle_acute_angles_l254_254332


namespace line_intersection_parallel_l254_254334

-- Definitions of the points and lines in the trapezoid
variables {A B C D S O E G F H : Type}

-- Conditions from the problem
def is_trapezoid (A B C D : Type) : Prop := parallel AB CD
def intersect_at (A B S : Type) : Prop := line AD ∩ line BC = S
def diagonals_intersect_at (A C B D O : Type) : Prop := line AC ∩ line BD = O
def intersects_at (S O A B C D E G F H : Type) : Prop :=
  (line SO ∩ line AB = E) ∧ (line SO ∩ line CD = G) ∧
  ((line O ∥ line AB) ∧ (line O ∩ line AD = F) ∧ (line O ∩ line BC = H))

-- Main theorem to prove
theorem line_intersection_parallel (A B C D S O E G F H : Type)
  (h1 : is_trapezoid A B C D)
  (h2 : intersect_at A B S)
  (h3 : diagonals_intersect_at A C B D O)
  (h4 : intersects_at S O A B C D E G F H):
  parallel (line_intersection E G) (line_intersection F H) AB :=
sorry

end line_intersection_parallel_l254_254334


namespace part1_arithmetic_sequence_part1_a_n_part2_sum_b_n_part3_min_m_l254_254333

-- Definitions for conditions
def sequence_S (n : ℕ) : ℝ := if n = 1 then 1 else sorry -- This will be defined based on the problem
def sequence_Sn (n : ℕ) : ℝ := (\frac{1}{2n-1}) -- This definition holds from the solution

def a_n (n : ℕ) : ℝ := 
  match n with
  | 1 => 1
  | n => -\frac{2}{(2n-1)*(2n-3)}

def b_n (n : ℕ) : ℝ := sequence_Sn n / (2 * n + 1)
def T_n (n : ℕ) : ℝ := finset.sum (finset.range n) (λ k, b_n (k + 1))

-- Part 1 proof statement
theorem part1_arithmetic_sequence (n : ℕ) (hn : n ≥ 2) : 
  ∃ d : ℝ, ∀ m : ℕ, n ≤ m → (1 / sequence_Sn m) = (1 / sequence_Sn n) + d * (m - n) := sorry

theorem part1_a_n (n : ℕ) (hn : n ≥ 2) : a_n n = - (2 / (2 * n - 1) * (2 * n - 3)) := sorry

-- Part 2 proof statement
theorem part2_sum_b_n (n : ℕ) : T_n n = (n / (2 * n + 1)) := sorry

-- Part 3 proof statement
theorem part3_min_m (m : ℕ) : 
  (∃ m : ℕ, m ≥ 10 ∧ ∀ n : ℕ, n > 0 → T_n n < (1 / 4) * (m - 8)) := sorry

end part1_arithmetic_sequence_part1_a_n_part2_sum_b_n_part3_min_m_l254_254333


namespace find_y_l254_254979

theorem find_y (a b y : ℝ) (h1 : s = (3 * a) ^ (2 * b)) (h2 : s = 5 * (a ^ b) * (y ^ b))
  (h3 : 0 < a) (h4 : 0 < b) : 
  y = 9 * a / 5 := by
  sorry

end find_y_l254_254979


namespace purely_imaginary_complex_l254_254905

-- Define the problem statement in Lean 4
theorem purely_imaginary_complex (x : ℝ) (z : ℂ) (h1 : z = (x^2 + x - 2) + (x + 2) * complex.I) 
  (h2 : x^2 + x - 2 = 0) (h3 : x + 2 ≠ 0) : x = 1 :=
sorry -- Proof not required

end purely_imaginary_complex_l254_254905


namespace intersection_M_N_l254_254720

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254720


namespace sales_revenue_nonnegative_l254_254487

def revenue (x : ℝ) : ℝ := -10 * x^2 + 200 * x + 15000

theorem sales_revenue_nonnegative (x : ℝ) (hx : x = 9 ∨ x = 11) : revenue x ≥ 15950 :=
by
  cases hx
  case inl h₁ =>
    sorry -- calculation for x = 9
  case inr h₂ =>
    sorry -- calculation for x = 11

end sales_revenue_nonnegative_l254_254487


namespace solve_equation_l254_254445

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end solve_equation_l254_254445


namespace find_value_l254_254858

theorem find_value 
    (x y : ℝ) 
    (h : 2 * x = Math.log (x + y - 1) + Math.log (x - y - 1) + 4) :
    2015 * x^2 + 2016 * y^3 = 8060 := 
sorry

end find_value_l254_254858


namespace smallest_value_of_a1_conditions_l254_254039

noncomputable theory

variables {a1 a2 a3 a4 a5 a6 a7 a8 : ℝ}

/-- The smallest value of \(a_1\) when the sum of \(a_1, \ldots, a_8\) is \(4/3\) 
    and the sum of any seven of these numbers is positive. -/
theorem smallest_value_of_a1_conditions 
  (h_sum : a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 = 4 / 3)
  (h_sum_seven : ∀ i : {j // j = 1 ∨ j = 2 ∨ j = 3 ∨ j = 4 ∨ j = 5 ∨ j = 6 ∨ j = 7 ∨ j = 8}, 
                  0 < a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 - a i.val) :
  -8 < a1 ∧ a1 ≤ 1 / 6 :=
sorry

end smallest_value_of_a1_conditions_l254_254039


namespace avg_age_second_group_l254_254410

theorem avg_age_second_group (avg_age_class : ℕ) (avg_age_first_group : ℕ) (age_15th_student : ℕ) (students_class : ℕ) (students_first_group : ℕ) (students_second_group : ℕ) :
  avg_age_class = 15 →
  avg_age_first_group = 14 →
  age_15th_student = 15 →
  students_class = 15 →
  students_first_group = 7 →
  students_second_group = 7 →
  let total_age_class := students_class * avg_age_class,
      total_age_first_group := students_first_group * avg_age_first_group,
      total_age_combined_groups := total_age_class - age_15th_student,
      total_age_second_group := total_age_combined_groups - total_age_first_group,
      avg_age_second_group := total_age_second_group / students_second_group
  in avg_age_second_group = 16 :=
by
  intros h1 h2 h3 h4 h5 h6,
  let total_age_class := students_class * avg_age_class,
  let total_age_first_group := students_first_group * avg_age_first_group,
  let total_age_combined_groups := total_age_class - age_15th_student,
  let total_age_second_group := total_age_combined_groups - total_age_first_group,
  let avg_age_second_group := total_age_second_group / students_second_group,
  rw [h1, h2, h3, h4, h5, h6] at *,
  exact calc 
    avg_age_second_group
    = (total_age_combined_groups - total_age_first_group) / students_second_group : rfl
    ... = 16 : by sorry

end avg_age_second_group_l254_254410


namespace more_cans_l254_254158

def cat_packages := 9
def dog_packages := 7
def cans_per_cat_package := 10
def cans_per_dog_package := 5

theorem more_cans := 
  let cat_cans := cat_packages * cans_per_cat_package
  let dog_cans := dog_packages * cans_per_dog_package
  cat_cans - dog_cans = 55 :=
begin
  sorry
end

end more_cans_l254_254158


namespace smaller_angle_between_ne_sw_l254_254521

-- Definitions based on given conditions
def total_degrees_circle : ℕ := 360
def number_of_rays : ℕ := 12
def first_ray_direction : ℕ := 0  -- North corresponds to 0 degrees
def central_angle (total_degrees : ℕ) (num_rays : ℕ) : ℕ := total_degrees / num_rays
def ray_angle (i : ℕ) (angle_per_ray : ℕ) : ℕ := i * angle_per_ray

-- Lean theorem statement
theorem smaller_angle_between_ne_sw :
  let angle_per_ray := central_angle total_degrees_circle number_of_rays in
  let northeast_angle := ray_angle 2 angle_per_ray in
  let southwest_angle := ray_angle 6 angle_per_ray in
  northeast_angle < southwest_angle →
  southwest_angle - northeast_angle = 120 := by
  sorry

end smaller_angle_between_ne_sw_l254_254521


namespace smallest_third_term_is_neg_19_l254_254537

noncomputable def smallest_third_term_geom_prog (d : ℝ) : ℝ :=
  if ((9 + d) ^ 2 = 5 * (35 + 2 * d)) 
  then min (35 + 2 * d) (-19) 
  else (35 + 2 * d)

theorem smallest_third_term_is_neg_19 :
  ∃ d : ℝ, ((9 + d) ^ 2 = 5 * (35 + 2 * d)) ∧ (smallest_third_term_geom_prog d) = -19 :=
begin
  use -12,
  split,
  { sorry, },
  { sorry, }
end

end smallest_third_term_is_neg_19_l254_254537


namespace intersection_M_N_l254_254776

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254776


namespace distinct_four_digit_numbers_with_5_l254_254273

theorem distinct_four_digit_numbers_with_5 
  (digits : Finset ℕ) 
  (four_digits : digits = {1, 2, 3, 4, 5}) :
  (Σ (numbers : Finset (Fin 10000)), 
    ∀ n ∈ numbers, n ∉ (Finset.pmap (λ _, {1, 2, 3, 4}) digits sorry)) =
    96 := 
  sorry

end distinct_four_digit_numbers_with_5_l254_254273


namespace intersection_M_N_l254_254688

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254688


namespace hyperbola_eccentricity_l254_254259

variables {a b c e : ℝ}
-- Given the hyperbola $x^2 / a^2 - y^2 / b^2 = 1$
def hyperbola (x y : ℝ) : Prop := x^2 / a^2 - y^2 / b^2 = 1

-- Given the conditions: $a > 0, b > 0$
variables (a_pos : a > 0) (b_pos : b > 0)

-- The distance c is half the hyperbola's focal distance
def focal_distance : ℝ := sqrt (a^2 + b^2) / 2

-- Circle's equation: $x^2 + y^2 = c^2$
def circle (x y : ℝ) : Prop := x^2 + y^2 = (focal_distance / 2)^2

-- The area of quadrilateral ABCD is $2b^2$
def quadrilateral_area (A B C D : ℝ × ℝ) : Prop :=
  (let (x₁, y₁) := A, (x₂, y₂) := B, (x₃, y₃) := C, (x₄, y₄) := D in 
  abs ((x₁ * y₂ - y₁ * x₂) + (x₂ * y₃ - y₂ * x₃) + (x₃ * y₄ - y₃ * x₄) + (x₄ * y₁ - y₄ * x₁)) / 2 = 2 * b^2)

-- Prove the hyperbola's eccentricity is sqrt(5)
theorem hyperbola_eccentricity : e = sqrt 5 := sorry

end hyperbola_eccentricity_l254_254259


namespace lottery_probability_l254_254381

theorem lottery_probability (x_1 x_2 x_3 x_4 : ℝ) (p : ℝ) (h0 : 0 < p ∧ p < 1) : 
  x_1 = p * x_3 → 
  x_2 = p * x_4 + (1 - p) * x_1 → 
  x_3 = p + (1 - p) * x_2 → 
  x_4 = p + (1 - p) * x_3 → 
  x_2 = 0.19 :=
by
  sorry

end lottery_probability_l254_254381


namespace xiaoming_water_usage_l254_254328

noncomputable def water_usage {yuan_per_cubic_meter_1 : ℝ} {max_cubic: ℕ} 
  {yuan_per_cubic_meter_2 : ℝ} (total_payment: ℝ) : ℝ :=
let first_payment := max_cubic * yuan_per_cubic_meter_1
in if total_payment ≤ first_payment then total_payment / yuan_per_cubic_meter_1
   else max_cubic + (total_payment - first_payment) / yuan_per_cubic_meter_2

theorem xiaoming_water_usage : 
  ∀ (total_payment : ℝ) (max_cubic: ℕ) (yuan_per_cubic_meter_1 yuan_per_cubic_meter_2: ℝ),
    total_payment = 33.6 → max_cubic = 15 → 
    yuan_per_cubic_meter_1 = 1.6 → yuan_per_cubic_meter_2 = 2.4 →
    water_usage total_payment = 19 :=
by
  intros total_payment max_cubic yuan_per_cubic_meter_1 yuan_per_cubic_meter_2 htp hmc hy1 hy2
  simp [htp, hmc, hy1, hy2, water_usage]
  sorry

end xiaoming_water_usage_l254_254328


namespace M_inter_N_eq_2_4_l254_254825

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254825


namespace find_n_l254_254204

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n ≤ 6 ∧ n ≡ -2023 [MOD 7] ∧ n = 6 :=
by
  use 6
  split
  · norm_num
  split
  · norm_num
  split
  · norm_num
  · exact rfl

end find_n_l254_254204


namespace change_calculation_l254_254953

-- Define the initial amounts of Lee and his friend
def lee_amount : ℕ := 10
def friend_amount : ℕ := 8

-- Define the cost of items they ordered
def chicken_wings : ℕ := 6
def chicken_salad : ℕ := 4
def soda : ℕ := 1
def soda_count : ℕ := 2
def tax : ℕ := 3

-- Define the total money they initially had
def total_money : ℕ := lee_amount + friend_amount

-- Define the total cost of the food without tax
def food_cost : ℕ := chicken_wings + chicken_salad + (soda * soda_count)

-- Define the total cost including tax
def total_cost : ℕ := food_cost + tax

-- Define the change they should receive
def change : ℕ := total_money - total_cost

theorem change_calculation : change = 3 := by
  -- Note: Proof here is omitted
  sorry

end change_calculation_l254_254953


namespace smallest_positive_period_abs_sin_l254_254597

def f : ℝ → ℝ := λ x, |sin x|

theorem smallest_positive_period_abs_sin : 
    ∃ T > 0, (∀ x, f (x + T) = f x) ∧
    (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) :=
begin
  use π,
  split,
  { sorry },    -- prove T > 0
  split,
  { sorry },    -- prove ∀ x, f (x + T) = f x
  { sorry }     -- prove ∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T
end

end smallest_positive_period_abs_sin_l254_254597


namespace intersection_M_N_l254_254735

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254735


namespace sqrt_meaningful_iff_l254_254908

theorem sqrt_meaningful_iff (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 1)) ↔ x ≥ 1 := 
by 
sry.

end sqrt_meaningful_iff_l254_254908


namespace min_moves_wren_l254_254360

/-- Definitions for gcd, tuple (p, q), and the least positive integer t meeting the required conditions. -/
def gcd (a b : ℤ) : ℤ := Int.gcd a b

def p_def (m n : ℤ) : ℕ := Nat.find (exists t : ℕ, (2 * m * t ≡ ±1 [MOD n]))

def q_def (m n p : ℤ) : ℤ := min ((2 * m * p - 1) / n) ((2 * m * p + 1) / n)

/-- The smallest number of moves required to travel from (0, 0) to (1, 0) or prove it does not exist. -/
theorem min_moves_wren (m n : ℕ) (h1 : 0 < m) (h2 : 0 < n) : 
  (gcd m n > 1 → ¬ ∃ k, True) ∧ 
  ((Nat.odd m ∧ Nat.odd n) → ¬ ∃ k, True) ∧ 
  ((gcd m n = 1 ∧ Nat.even m ∧ Nat.odd n) → 
    ∃ k, k = max (2 * p_def m n) m + max (q_def m n (p_def m n)) n) :=
by 
  split
  sorry -- The three parts of the proof need to be filled in here.

end min_moves_wren_l254_254360


namespace range_of_f_l254_254023

-- Define the piecewise function
def f (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 1 then 2 * x^2
  else if 1 < x ∧ x < 2 then 2
  else if x ≥ 2 then 3
  else 0 -- this case should never happen, for completeness

-- The range of f
def f_range : set ℝ := {y | ∃ x, f x = y}

-- The expected range
def expected_range : set ℝ := {y | (0 ≤ y ∧ y ≤ 2) ∨ y = 3}

-- The theorem statement
theorem range_of_f : f_range = expected_range :=
by 
  sorry

end range_of_f_l254_254023


namespace students_in_second_class_l254_254454

-- Definitions based on the conditions
def students_first_class : ℕ := 30
def avg_mark_first_class : ℕ := 40
def avg_mark_second_class : ℕ := 80
def combined_avg_mark : ℕ := 65

-- Proposition to prove
theorem students_in_second_class (x : ℕ) 
  (h1 : students_first_class * avg_mark_first_class + x * avg_mark_second_class = (students_first_class + x) * combined_avg_mark) : 
  x = 50 :=
sorry

end students_in_second_class_l254_254454


namespace prism_volume_correct_l254_254414

variable {l α β : ℝ} (hα : α > 0) (hβ : β > 0) (hl : l > 0)

def volume_of_prism : ℝ :=
  (1/8) * l^3 * sin (2 * β) * cos β * cot (α / 2)

theorem prism_volume_correct :
  volume_of_prism l α β = (1/8) * l^3 * sin (2 * β) * cos β * cot (α / 2) := by
  sorry

end prism_volume_correct_l254_254414


namespace tetrahedron_plane_area_eq_sum_l254_254456

-- Definitions of areas and segments
variables {A B C D I P Q R : Type}
variables [HasArea A] [HasArea B] [HasArea C] [HasArea D]
variables [HasSegment AB] [HasSegment AC] [HasSegment AD]
variables [HasSegment AP] [HasSegment AQ] [HasSegment AR]

-- Define face areas and edge ratios within the tetrahedron
variables {S_A S_B S_C S_D : ℝ}

-- Hypotheses (conditions)
variable (h1 : Incenter I A B C D)
variable (h2 : PlaneThroughIncenter I A B P Q R)
variable (h3 : AreaOpposite A S_A)
variable (h4 : AreaOpposite B S_B)
variable (h5 : AreaOpposite C S_C)
variable (h6 : AreaOpposite D S_D)

-- Actual theorem statement
theorem tetrahedron_plane_area_eq_sum :
  S_B * (AB / AP) + S_C * (AC / AQ) + S_D * (AD / AR) = S_A + S_B + S_C + S_D :=
sorry

end tetrahedron_plane_area_eq_sum_l254_254456


namespace pepperoni_coverage_fraction_l254_254549

noncomputable def diameter_pizza : ℝ := 16
noncomputable def diameter_pepperoni : ℝ := diameter_pizza / 8
noncomputable def radius_pepperoni : ℝ := diameter_pepperoni / 2
noncomputable def area_pepperoni : ℝ := π * radius_pepperoni ^ 2
noncomputable def total_area_pepperoni : ℝ := 32 * area_pepperoni
noncomputable def radius_pizza : ℝ := diameter_pizza / 2
noncomputable def area_pizza : ℝ := π * radius_pizza ^ 2
noncomputable def fraction_covered : ℝ := total_area_pepperoni / area_pizza

theorem pepperoni_coverage_fraction :
  fraction_covered = 1 / 2 := by
  sorry

end pepperoni_coverage_fraction_l254_254549


namespace f_minus_one_eq_neg_four_l254_254367

noncomputable def f (x : ℝ) : ℝ :=
if x >= 0 then 2 * x^2 + 2 * x else -(2 * (-x)^2 + 2 * (-x))

theorem f_minus_one_eq_neg_four :
  f (-1) = -4 :=
by {
  -- Given f(x) is odd, applying definition of f for x >= 0 and using b = 0
  have h1 : f (0) = 0 := by simp [f], -- f(0) = 2 * 0^2 + 2 * 0 + 0
  have h2 : f (1) = 2 * 1^2 + 2 * 1 := by simp [f]; simp [if_pos]; norm_num,
  have h3 : f (-1) = -f (1) := by simp [f]; simp [if_neg]; norm_num,
  rw h2 at h3,
  exact h3,
}

end f_minus_one_eq_neg_four_l254_254367


namespace range_of_x_plus_one_over_x_l254_254896

theorem range_of_x_plus_one_over_x (x : ℝ) (h : x < 0) : x + 1/x ≤ -2 := by
  sorry

end range_of_x_plus_one_over_x_l254_254896


namespace KayleeAgeCorrect_l254_254296

-- Define Kaylee's current age
def KayleeCurrentAge (k : ℕ) : Prop :=
  (3 * 5 + (7 - k) = 7)

-- State the theorem
theorem KayleeAgeCorrect : ∃ k : ℕ, KayleeCurrentAge k ∧ k = 8 := 
sorry

end KayleeAgeCorrect_l254_254296


namespace greatest_multiple_of_6_and_5_less_than_1000_l254_254086

theorem greatest_multiple_of_6_and_5_less_than_1000 : 
  ∃ (n : ℕ), n < 1000 ∧ (n % 6 = 0) ∧ (n % 5 = 0) ∧ ∀ m : ℕ, (m < 1000 ∧ m % 6 = 0 ∧ m % 5 = 0) → m ≤ n :=
begin
  let answer := 990,
  use answer,
  split,
  { exact 990 < 1000, },
  split,
  { exact 990 % 6 = 0, },
  split,
  { exact 990 % 5 = 0, },
  { intros m hm,
    cases hm with hmlt1000 hm,
    cases hm with hm_mod6 hm_mod5,
    have : m ≤ 990, sorry,
    exact this, }
end

end greatest_multiple_of_6_and_5_less_than_1000_l254_254086


namespace part_Ⅰ_part_Ⅱ_l254_254261

theorem part_Ⅰ (x m : ℝ) (h : ∀ x, -2 ≤ x ∧ x ≤ 1 → |x+2| + |x - m| ≤ 3) : m = 1 :=
sorry

theorem part_Ⅱ (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 1) : - real.sqrt 6 ≤ a + 2 * b + 3 * c ∧ a + 2 * b + 3 * c ≤ real.sqrt 6 :=
sorry

end part_Ⅰ_part_Ⅱ_l254_254261


namespace max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254470

def max_distinct_numbers (n : ℕ) (k : ℕ) (f : ℕ → ℕ → Prop) : ℕ :=
  sorry

theorem max_num_distinct_from_1_to_1000_no_diff_4_5_6 :
  max_distinct_numbers 1000 4 (λ a b, ¬(a - b = 4 ∨ a - b = 5 ∨ a - b = 6)) = 400 :=
sorry

end max_num_distinct_from_1_to_1000_no_diff_4_5_6_l254_254470


namespace melissa_work_hours_l254_254383

def fabricate_dresses (fabricA_total : ℕ) (fabricA_consumption : ℕ) (fabricA_time : ℕ) (fabricB_total : ℕ) (fabricB_consumption : ℕ) (fabricB_time : ℕ) : ℕ :=
  let dressesA := fabricA_total / fabricA_consumption
  let dressesB := fabricB_total / fabricB_consumption
  (dressesA * fabricA_time) + (dressesB * fabricB_time)

theorem melissa_work_hours :
  fabricate_dresses 40 4 3 28 5 4 = 50 :=
by
  unfold fabricate_dresses
  norm_num
  sorry

end melissa_work_hours_l254_254383


namespace partial_fraction_decomposition_l254_254589

noncomputable theory

def find_coefficients : Prop :=
  ∃ A B : ℚ, 
    (∀ x : ℚ, 
        (x^2 + 2*x - 63) ≠ 0 → 
        (2*x + 4) / (x^2 + 2*x - 63) = A / (x - 7) + B / (x + 9)) ∧
    A = 9/8 ∧
    B = 7/8

theorem partial_fraction_decomposition : find_coefficients := 
sorry

end partial_fraction_decomposition_l254_254589


namespace candy_bar_price_increase_l254_254124

-- Definition of the conditions:
variables {W P : ℝ}  -- Weight in ounces and price in dollars

-- Original effective price per ounce in local currency
def old_effective_price_per_ounce (W P : ℝ) : ℝ := (1.26 * P) / W

-- New effective price per ounce in local currency
def new_effective_price_per_ounce (W P : ℝ) : ℝ := (1.458 * P) / (0.6 * W)

-- The percent increase in effective price per ounce
def percent_increase (old new : ℝ) : ℝ := ((new / old) - 1) * 100

-- The main theorem
theorem candy_bar_price_increase (W P : ℝ) (h1 : W > 0) (h2 : P > 0) :
  percent_increase (old_effective_price_per_ounce W P) (new_effective_price_per_ounce W P) = 92.857 :=
by
  sorry

end candy_bar_price_increase_l254_254124


namespace triangle_area_l254_254073

theorem triangle_area :
  let L1 : ℝ × ℝ → Prop := λ p, p.2 = 3 * p.1 - 6
  let L2 : ℝ × ℝ → Prop := λ p, p.2 = - (1 / 3) * p.1 + 4
  let L3 : ℝ × ℝ → Prop := λ p, p.1 + p.2 = 12
  let P1 : ℝ × ℝ := (3, 3)
  let P2 : ℝ × ℝ := (4.5, 7.5)
  let P3 : ℝ × ℝ := (12, 0)
  let area : ℝ := 1 / 2 * abs (P1.1 * (P2.2 - P3.2) + P2.1 * (P3.2 - P1.2) + P3.1 * (P1.2 - P2.2))
  P1 ∈ L1 ∧
  P1 ∈ L2 ∧
  P2 ∈ (λ p, L1 p ∧ L3 p) ∧
  P3 ∈ (λ p, L2 p ∧ L3 p) ∧
  P1 = (3, 3) ∧
  P2 = (4.5, 7.5) ∧
  P3 = (12, 0) →
  area = 2.25 :=
by
  intros L1 L2 L3 P1 P2 P3 area h
  sorry

end triangle_area_l254_254073


namespace current_short_trees_l254_254453

theorem current_short_trees (S : ℕ) (S_planted : ℕ) (S_total : ℕ) 
  (H1 : S_planted = 105) 
  (H2 : S_total = 217) 
  (H3 : S + S_planted = S_total) :
  S = 112 :=
by
  sorry

end current_short_trees_l254_254453


namespace quadratic_function_properties_l254_254839

variables (a b c k m : ℝ)
def f (x : ℝ) : ℝ := 2 * x^2 - 4 * x + 3
def g (x m : ℝ) : ℝ := 2 * x + 2 * m + 1

theorem quadratic_function_properties :
  (∀ x, f x ≥ 1) ∧ (f 0 = 3) ∧ (f 2 = 3) ∧ (∀ x ∈ set.Icc (-3 : ℝ) (-1 : ℝ), f x > g x m) → m < 5 := 
sorry

end quadratic_function_properties_l254_254839


namespace books_selection_l254_254150

theorem books_selection : 
    ∃ n : ℕ, n = nat.choose 7 4 ∧ n = 35 :=
by {
  use (nat.choose 7 4),
  split,
  { refl },
  { norm_num }
}

end books_selection_l254_254150


namespace intersection_eq_l254_254707

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254707


namespace intersection_M_N_l254_254658

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254658


namespace correct_option_l254_254216

-- Define the given conditions
def a : ℕ := 7^5
def b : ℕ := 5^7

-- State the theorem to be proven
theorem correct_option : a^7 * b^5 = 35^35 := by
  -- insert the proof here
  sorry

end correct_option_l254_254216


namespace kim_morning_routine_time_l254_254355

-- Definitions based on conditions
def minutes_coffee : ℕ := 5
def minutes_status_update_per_employee : ℕ := 2
def minutes_payroll_update_per_employee : ℕ := 3
def num_employees : ℕ := 9

-- Problem statement: Verifying the total morning routine time for Kim
theorem kim_morning_routine_time:
  minutes_coffee + (minutes_status_update_per_employee * num_employees) + 
  (minutes_payroll_update_per_employee * num_employees) = 50 :=
by
  -- Proof can follow here, but is currently skipped
  sorry

end kim_morning_routine_time_l254_254355


namespace remainder_not_power_of_4_l254_254566

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end remainder_not_power_of_4_l254_254566


namespace intersection_of_M_and_N_l254_254784

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254784


namespace Albert_more_than_Joshua_l254_254948

def Joshua_rocks : ℕ := 80

def Jose_rocks : ℕ := Joshua_rocks - 14

def Albert_rocks : ℕ := Jose_rocks + 20

theorem Albert_more_than_Joshua : Albert_rocks - Joshua_rocks = 6 :=
by
  sorry

end Albert_more_than_Joshua_l254_254948


namespace max_distinct_numbers_l254_254474

theorem max_distinct_numbers (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 1000) :
  ∃ (s : set ℕ), (∀ x y ∈ s, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6)) ∧ s.card = 400 :=
by
  sorry

end max_distinct_numbers_l254_254474


namespace average_age_of_second_group_is_16_l254_254412

theorem average_age_of_second_group_is_16
  (total_age_15_students : ℕ := 225)
  (total_age_first_group_7_students : ℕ := 98)
  (age_15th_student : ℕ := 15) :
  (total_age_15_students - total_age_first_group_7_students - age_15th_student) / 7 = 16 := 
by
  sorry

end average_age_of_second_group_is_16_l254_254412


namespace relationship_b_a_c_l254_254245

variable {f : ℝ → ℝ}
variable {g : ℝ → ℝ}
variable {a b c y : ℝ}

-- Given conditions
def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f (x)
def increasing_function (f : ℝ → ℝ) : Prop := ∀ x1 x2, x1 < x2 → f x1 < f x2
def g (x : ℝ) : ℝ := x * f x
def b : ℝ := g (2 ^ 0.8)
def c : ℝ := g 3
def y_condition : Prop := 2 ^ 0.8 < y ∧ y < 3

-- To be proved
theorem relationship_b_a_c (h_odd : odd_function f) (h_increasing : increasing_function f) (h_y_condition : y_condition) :
  b < g y ∧ g y < c := by
  sorry

end relationship_b_a_c_l254_254245


namespace intersection_M_N_l254_254677

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254677


namespace distance_symmetric_point_l254_254941

theorem distance_symmetric_point (A B : ℝ × ℝ × ℝ)
  (hA : A = (2, 3, 1))
  (hB : B = (-2, 3, 1))
  (symmetric : B = (-A.1, A.2, A.3)) :
  dist A B = 4 := 
by sorry

end distance_symmetric_point_l254_254941


namespace intersection_M_N_l254_254813

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254813


namespace intersection_M_N_l254_254682

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254682


namespace intersection_M_N_l254_254675

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254675


namespace intersection_M_N_l254_254765

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254765


namespace find_a_l254_254830

noncomputable def A : Set ℝ := {x | x^2 - x - 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def is_solution (a : ℝ) : Prop := ∀ b, b ∈ B a → b ∈ A

theorem find_a (a : ℝ) : (B a ⊆ A) → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  intro h
  sorry

end find_a_l254_254830


namespace diamond_problem_l254_254210

namespace DiamondProblem

def diamond (a b : ℝ) : ℝ := Real.sqrt (a^2 + b^2)

theorem diamond_problem : diamond (diamond 6 8) (diamond (-8) (-6)) = 10 * Real.sqrt 2 :=
by -- We state the theorem without providing a proof.
sorry

end DiamondProblem

end diamond_problem_l254_254210


namespace elephant_distribution_l254_254393

theorem elephant_distribution (unions nonunions : ℕ) (elephants : ℕ) :
  unions = 28 ∧ nonunions = 37 ∧ (∀ k : ℕ, elephants = 28 * k ∨ elephants = 37 * k) ∧ (∀ k : ℕ, ((28 * k ≤ elephants) ∧ (37 * k ≤ elephants))) → 
  elephants = 2072 :=
by
  sorry

end elephant_distribution_l254_254393


namespace intersection_eq_l254_254698

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}
def intersection : Set ℕ := {x | x ∈ M ∧ x ∈ N}

theorem intersection_eq :
  intersection = {2, 4} := by
  sorry

end intersection_eq_l254_254698


namespace kaylee_current_age_l254_254295

-- Define the initial conditions
def matt_current_age : ℕ := 5
def kaylee_future_age_in_7_years : ℕ := 3 * matt_current_age

-- Define the main theorem to be proven
theorem kaylee_current_age : ∃ x : ℕ, x + 7 = kaylee_future_age_in_7_years ∧ x = 8 :=
by
  -- Use given conditions to instantiate the future age
  have h1 : kaylee_future_age_in_7_years = 3 * 5 := rfl
  have h2 : 3 * 5 = 15 := rfl
  have h3 : kaylee_future_age_in_7_years = 15 := by rw [h1, h2]
  -- Prove that there exists an x such that x + 7 = 15 and x = 8
  use 8
  split
  . rw [h3]
    norm_num
  . rfl

end kaylee_current_age_l254_254295


namespace sequence_stabilizes_mod_3_l254_254508

theorem sequence_stabilizes_mod_3 (a0 : ℕ) (h₀ : a0 > 1) (h_mod : a0 % 3 = 0) :
  ∃ A : ℕ, ∀ᶠ n in at_top, (a_seq a0 n) = A :=
begin
  -- Conditions given
  assume h₀ : a0 > 1,
  assume h_mod : a0 % 3 = 0,
  sorry -- proof is skipped
end

def a_seq : ℕ → ℕ → ℕ
| 0 := λ a0, a0
| (n + 1) := λ a_n, if (nat.sqrt a_n)^2 = a_n then nat.sqrt a_n else a_n + 3

lemma sqrt_int (a : ℕ) : (nat.sqrt a)^2 = a ↔ (∃ (k : ℕ), a = k^2) :=
begin
  sorry -- utility lemma for dealing with integer roots
end

end sequence_stabilizes_mod_3_l254_254508


namespace units_digit_of_2_pow_2012_l254_254451

theorem units_digit_of_2_pow_2012 : Nat.digits 10 (2 ^ 2012) % 10 = 6 :=
by
  sorry

end units_digit_of_2_pow_2012_l254_254451


namespace positive_divisors_of_5400_multiple_of_5_l254_254876

-- Declare the necessary variables and conditions
theorem positive_divisors_of_5400_multiple_of_5 :
  let n := 5400
  let factorization := [(2, 2), (3, 3), (5, 2)]
  ∀ (a b c: ℕ), 0 ≤ a ∧ a ≤ 2 ∧ 0 ≤ b ∧ b ≤ 3 ∧ 1 ≤ c ∧ c ≤ 2 →
    (a*b*c).count(n) = 24 := 
sorry

end positive_divisors_of_5400_multiple_of_5_l254_254876


namespace lines_intersect_condition_l254_254577

theorem lines_intersect_condition (a : ℝ) : 
  (a = 2) ∨ (a = -1) ↔ (ax + 2y ≠ 3) ∧ (x + (a-1)y ≠ 1)  :=
by
  sorry

end lines_intersect_condition_l254_254577


namespace evaluate_expression_l254_254584

theorem evaluate_expression (a b c : ℚ) (h1 : c = b - 8) (h2 : b = a + 3) (h3 : a = 2) 
  (h4 : a + 1 ≠ 0) (h5 : b - 3 ≠ 0) (h6 : c + 5 ≠ 0) :
  (a + 3) / (a + 1) * (b - 1) / (b - 3) * (c + 7) / (c + 5) = 20 / 3 := by
  sorry

end evaluate_expression_l254_254584


namespace AgathaAdditionalAccessories_l254_254159

def AgathaBudget : ℕ := 250
def Frame : ℕ := 85
def FrontWheel : ℕ := 35
def RearWheel : ℕ := 40
def Seat : ℕ := 25
def HandlebarTape : ℕ := 15
def WaterBottleCage : ℕ := 10
def BikeLock : ℕ := 20
def FutureExpenses : ℕ := 10

theorem AgathaAdditionalAccessories :
  AgathaBudget - (Frame + FrontWheel + RearWheel + Seat + HandlebarTape + WaterBottleCage + BikeLock + FutureExpenses) = 10 := by
  sorry

end AgathaAdditionalAccessories_l254_254159


namespace chord_product_l254_254969

def ω := Complex.exp (2 * Real.pi * Complex.I / 10)

def A := (3 : Complex)
def B := (-3 : Complex)
def D (k : ℕ) := 3 * (ω ^ k)

def AD_length (k : ℕ) := Complex.abs (A - D k)
def BD_length (k : ℕ) := Complex.abs (B - D k)

theorem chord_product :
  (AD_length 1) * (AD_length 2) * (AD_length 3) * (AD_length 4) *
  (BD_length 1) * (BD_length 2) * (BD_length 3) * (BD_length 4) = 32805 :=
by
  sorry

end chord_product_l254_254969


namespace pq_perpendicular_to_median_l254_254164

-- Define the geometry problem
variables {A B C E F P Q M : Type}
variables [MetricSpace A] [MetricSpace B] [MetricSpace C]
variables [MetricSpace E] [MetricSpace F] [MetricSpace P] [MetricSpace Q] [MetricSpace M]

-- Define the points and lines based on the given conditions
def is_acute_triangle (ABC : Triangle) : Prop := ABC.isAcute ∧ ABC.edge AB < ABC.edge AC
def feet_of_heights (E F : Point) (ABC : Triangle) : Prop := E = foot ABC B ∧ F = foot ABC C
def tangent_at_a_intersects_bc (P : Point) (A B C : Point) : Prop := tangent_circle_intersect{("A", A)} ("P", P) ("BC", BC)
def parallel_to_bc_through_a (Q : Point) (A B C E F : Point) : Prop := Q ∈ line_parallel A BC ∧ Q ∈ line_intersect E F

-- Main theorem to be proven
theorem pq_perpendicular_to_median
  (ABC : Triangle)
  (E F : Point)
  (P Q M : Point)
  (h1 : is_acute_triangle ABC)
  (h2 : feet_of_heights E F ABC)
  (h3 : tangent_at_a_intersects_bc P A B C)
  (h4 : parallel_to_bc_through_a Q A B C E F)
  (median : Line := median_from_A A B C) :
  perpendicular P Q median :=
begin
  sorry
end

end pq_perpendicular_to_median_l254_254164


namespace M_inter_N_eq_2_4_l254_254817

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254817


namespace find_S_l254_254975

-- Let's define the parameters
variables (R S T c : ℝ)

-- Given conditions
def condition1 := (R = c * (S ^ 2) / T)
def condition2 := (2 = c * (1 ^ 2) / 3)
def condition3 := (18 = 6 * (S ^ 2) / 2)

-- Goal is S = √6 under these conditions
theorem find_S : condition1 R S T c ∧ condition2 R S T c ∧ condition3 R S T c → S = Real.sqrt 6 :=
by
  sorry

end find_S_l254_254975


namespace pictures_hung_in_new_galleries_l254_254543

noncomputable def total_pencils_used : ℕ := 218
noncomputable def pencils_per_picture : ℕ := 5
noncomputable def pencils_per_exhibition : ℕ := 3

noncomputable def pictures_initial : ℕ := 9
noncomputable def galleries_requests : List ℕ := [4, 6, 8, 5, 7, 3, 9]
noncomputable def total_exhibitions : ℕ := 1 + galleries_requests.length

theorem pictures_hung_in_new_galleries :
  let total_pencils_for_signing := total_exhibitions * pencils_per_exhibition
  let total_pencils_for_drawing := total_pencils_used - total_pencils_for_signing
  let total_pictures_drawn := total_pencils_for_drawing / pencils_per_picture
  let pictures_in_new_galleries := total_pictures_drawn - pictures_initial
  pictures_in_new_galleries = 29 :=
by
  sorry

end pictures_hung_in_new_galleries_l254_254543


namespace median_ride_duration_is_190_l254_254030

def ride_durations : List ℕ :=
  [50, 95, 120, 153, 179, 130, 153, 210, 145, 160, 181, 190, 216, 239, 260]

-- Converting minute-based data points to seconds as per the problem conditions
def converted_durations : List ℕ :=
  [50, 95, 120, 130, 153, 153, 160, 179, 181, 190, 210, 145, 216, 239, 260]

def sorted_durations : List ℕ :=
  converted_durations.qsort (· < ·)

def median_rides (n : List ℕ) : ℕ :=
  n[n.length / 2]

theorem median_ride_duration_is_190 :
  median_rides sorted_durations = 190 :=
by
  sorry

end median_ride_duration_is_190_l254_254030


namespace intersection_M_N_l254_254716

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254716


namespace intersection_M_N_l254_254656

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_M_N_l254_254656


namespace solve_exponential_equation_l254_254512

theorem solve_exponential_equation : ∃ x : ℝ, 2^x = 8 ∧ x = 3 :=
by
  sorry

end solve_exponential_equation_l254_254512


namespace intersection_M_N_l254_254808

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254808


namespace intersection_M_N_l254_254801

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254801


namespace intersection_of_M_and_N_l254_254646

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254646


namespace mr_johnson_total_volunteers_l254_254385

theorem mr_johnson_total_volunteers (students_per_class : ℕ) (classes : ℕ) (teachers : ℕ) (additional_volunteers : ℕ) :
  students_per_class = 5 → classes = 6 → teachers = 13 → additional_volunteers = 7 →
  (students_per_class * classes + teachers + additional_volunteers) = 50 :=
by intros; simp [*]

end mr_johnson_total_volunteers_l254_254385


namespace intersection_M_N_l254_254665

def M := {2, 4, 6, 8, 10}
def N := {x : ℤ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254665


namespace correct_propositions_l254_254547

-- Proposition definitions
def prop1 (a b : ℝ) : Prop := (log a 3 > log b 3) → (a > b)
def range_f : Set ℝ := {y | ∃ x, 0 ≤ x ∧ y = x^2 - 2*x + 3}
def prop2 : Prop := range_f = {y | 2 ≤ y}
def prop3 (g : ℝ → ℝ) (a b : ℝ) [Continuous g] : Prop := (g a = g b ∧ g a > 0) → ¬∃ x ∈ Icc a b, g x = 0
def h (x : ℝ) : ℝ := (1 - exp (2*x)) / (exp x)
def prop4 : Prop := (∀ x, h (-x) = -h x) ∧ ∀ x, deriv h x < 0

-- The theorem statement for the proof problem
theorem correct_propositions :
  (prop1 a b) = false ∧ (prop2) = true ∧ (prop3 g a b) = false ∧ (prop4) = true := sorry

end correct_propositions_l254_254547


namespace min_a5_of_geom_seq_l254_254901

-- Definition of geometric sequence positivity and difference condition.
def geom_seq_pos_diff (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (∀ n, a n > 0) ∧ (a 3 - a 1 = 2)

-- The main theorem stating that the minimum value of a_5 is 8.
theorem min_a5_of_geom_seq {a : ℕ → ℝ} {q : ℝ} (h : geom_seq_pos_diff a q) :
  a 5 ≥ 8 :=
sorry

end min_a5_of_geom_seq_l254_254901


namespace multiple_of_three_l254_254232

theorem multiple_of_three (a b : ℤ) : ∃ k : ℤ, (a + b = 3 * k) ∨ (ab = 3 * k) ∨ (a - b = 3 * k) :=
sorry

end multiple_of_three_l254_254232


namespace intersection_M_N_l254_254755

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254755


namespace problem_statement_l254_254281

-- Conditions from the problem
def f (x : ℝ) : ℝ := sin (2 * x + π / 4) + cos (2 * x + π / 4)
def ω := 2
def period := π
def f_zero := sqrt 2

-- Hypothesis: assuming the conditions given in the problem
axiom period_eq : ∀ x, f (x + period) = f x
axiom f_at_zero : f 0 = f_zero

-- Problem: given the conditions, prove that f(x) is decreasing in the interval (0, π/2)
theorem problem_statement :
  (∀ x : ℝ, 0 < x ∧ x < π / 2 → f' x < 0) := sorry

end problem_statement_l254_254281


namespace intersection_M_N_l254_254763

open Set

def M := {2, 4, 6, 8, 10}
def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254763


namespace intersection_M_N_l254_254631

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254631


namespace domain_transform_l254_254255

variable (f : ℝ → ℝ)

theorem domain_transform :
  (∀ x, 1 < f (2^x) < 2) → (∀ x, 4 < x → x < 16 → 1 < f (log x) < 2) := by
  sorry

end domain_transform_l254_254255


namespace intersection_sum_of_coordinates_l254_254930

structure Point where
  x : ℝ
  y : ℝ

def midpoint (A B : Point) : Point :=
  {x := (A.x + B.x) / 2, y := (A.y + B.y) / 2}

def line_eq (P1 P2 : Point) (x : ℝ) : ℝ :=
  let slope := (P1.y - P2.y) / (P1.x - P2.x)
  let y_intercept := P1.y - slope * P1.x
  slope * x + y_intercept

theorem intersection_sum_of_coordinates :
  let P := Point.mk 0 8
  let Q := Point.mk 0 0
  let R := Point.mk 10 0
  let G := midpoint P Q
  let H := midpoint Q R
  G.x = 0 →
  G.y = 4 →
  H.x = 5 →
  H.y = 0 →
  let PH := line_eq P H
  let GQ_x := G.x
  (PH GQ_x) + GQ_x = 8 :=
by
  sorry

end intersection_sum_of_coordinates_l254_254930


namespace vieta_symmetric_functions_l254_254934

-- Defining the problem setting
variables {α : Type*} [Field α]
variables (α₁ α₂ α₃ αₙ : α) (A₁ A₂ A₃ Aₙ : α) (n : ℕ)

-- The polynomial and its roots
noncomputable def polynomial_has_roots : Prop :=
  ∀ (x : α), (x^n + A₁ * x^(n - 1) + A₂ * x^(n - 2) + ... + Aₙ = 0) ↔ (x = α₁) ∨ (x = α₂) ∨ ... ∨ (x = αₙ)

-- The Vieta's theorem conditions
def vieta_conditions :=
  (α₁ + α₂ + α₃ + ... + αₙ = -A₁) ∧
  (α₁ * α₂ + α₁ * α₃ + α₂ * α₃ + ... + αₙ₋₁ * αₙ = A₂) ∧
  (α₁ * α₂ * α₃ + α₁ * α₂ * α₄ + ... + αₙ₋₂ * αₙ₋₁ * αₙ = -A₃) ∧
  (α₁ * α₂ * α₃ * ... * αₙ = (-1)^n * Aₙ)

-- The theorem we need to prove
theorem vieta_symmetric_functions :
  polynomial_has_roots α₁ α₂ α₃ αₙ A₁ A₂ A₃ Aₙ n →
  vieta_conditions α₁ α₂ α₃ αₙ A₁ A₂ A₃ Aₙ n →
  (symmetric_function (α₁ + α₂ + α₃ + ... + αₙ)) ∧
  (symmetric_function (α₁ * α₂ + α₁ * α₃ + α₂ * α₃ + ... + αₙ₋₁ * αₙ)) ∧
  (symmetric_function (α₁ * α₂ * α₃ + α₁ * α₂ * α₄ + ... + αₙ₋₂ * αₙ₋₁ * αₙ)) ∧
  (symmetric_function (α₁ * α₂ * α₃ * ... * αₙ))
:= 
sorry

end vieta_symmetric_functions_l254_254934


namespace triangle_congruence_statements_l254_254491

theorem triangle_congruence_statements :
  ∀ (A B : Type) [OrderedRing A] [OrderedRing B]
    (T1 T2: Triangle A) (T3 T4: Triangle B),
    ¬ (∃ (a1 a2 a3 : A), T1.angles = [a1, a2, a3] ∧ T2.angles = [a1, a2, a3] ∧ congruent T1 T2) ∧
    (∃ (s1 s2 s3 : A), T1.sides = [s1, s2, s3] ∧ T2.sides = [s1, s2, s3] ∧ congruent T1 T2) ∧
    ( (∃ (a1 a2 : A) (s : A), (T1.angles = [a1, a2] ∧ T1.side = s) ∧ (T2.angles = [a1, a2] ∧ T2.side = s) ∧ congruent T1 T2)
    ∨ (∃ (a1 a2 : A) (s : A), (T1.angles = [a1, a2] ∧ T1.side' = s) ∧ (T2.angles = [a1, a2] ∧ T2.side' = s) ∧ congruent T1 T2) ) ∧
    ¬ (∃ (b h : A), T1.base = b ∧ T1.height = h ∧ T2.base = b ∧ T2.height = h ∧ congruent T1 T2) :=
begin
    sorry
end

end triangle_congruence_statements_l254_254491


namespace arithmetic_sequence_ratio_l254_254225

variable (a_1 d : ℝ)

-- Definition of the sum of the first n terms of the arithmetic sequence
def S (n : ℝ) : ℝ := n * a_1 + (n * (n - 1) * d) / 2

-- Given condition
def condition1 : Prop := (S 5) / (S 3) = 3

-- The relation to prove
def result : Prop := (a_1 + 4 * d) / (a_1 + 2 * d) = 17 / 9

-- The main statement
theorem arithmetic_sequence_ratio (h1 : condition1 a_1 d) : result a_1 d := by
  sorry

end arithmetic_sequence_ratio_l254_254225


namespace range_of_m_l254_254252

noncomputable def f (x : ℝ) : ℝ := Real.log (1 + x^4) / Real.log 3
noncomputable def g (x : ℝ) : ℝ := x * f x

theorem range_of_m (m : ℝ) : g (1 - m) < g (2 * m) → m > 1 / 3 :=
  by
  sorry

end range_of_m_l254_254252


namespace solve_equation_l254_254444

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 → (x = 0 ∨ x = 1) :=
by
  intro h
  sorry

end solve_equation_l254_254444


namespace tamika_vs_carlos_probability_l254_254003

theorem tamika_vs_carlos_probability :
  let tamika_sums := {16, 18, 20}
  let carlos_products := {8, 10, 20}
  let favorable_pairs :=
    {(16, 8), (16, 10), (18, 8), (18, 10), (20, 8), (20, 10)}
  let total_pairs :=
    {(16, 8), (16, 10), (16, 20), (18, 8), (18, 10), (18, 20), (20, 8), (20, 10), (20, 20)}
  (favorable_pairs.card : ℚ) / total_pairs.card = 2 / 3 := sorry

end tamika_vs_carlos_probability_l254_254003


namespace total_height_geometric_solid_l254_254553

-- Definitions corresponding to conditions
def radius_cylinder1 : ℝ := 1
def radius_cylinder2 : ℝ := 3
def height_water_surface_figure2 : ℝ := 20
def height_water_surface_figure3 : ℝ := 28

-- The total height of the geometric solid is 29 cm
theorem total_height_geometric_solid :
  ∃ height_total : ℝ,
    (height_water_surface_figure2 + height_total - height_water_surface_figure3) = 29 :=
sorry

end total_height_geometric_solid_l254_254553


namespace speed_of_man_in_still_water_l254_254497

variable (v_m v_s : ℝ)

-- Conditions
def downstream_distance : ℝ := 51
def upstream_distance : ℝ := 18
def time : ℝ := 3

-- Equations based on the conditions
def downstream_speed_eq : Prop := downstream_distance = (v_m + v_s) * time
def upstream_speed_eq : Prop := upstream_distance = (v_m - v_s) * time

-- The theorem to prove
theorem speed_of_man_in_still_water : downstream_speed_eq v_m v_s ∧ upstream_speed_eq v_m v_s → v_m = 11.5 :=
by
  intro h
  sorry

end speed_of_man_in_still_water_l254_254497


namespace max_min_A_of_coprime_36_and_greater_than_7777777_l254_254171

theorem max_min_A_of_coprime_36_and_greater_than_7777777 
  (A B : ℕ) 
  (h1 : coprime B 36) 
  (h2 : B > 7777777)
  (h3 : A = 10^7 * (B % 10) + (B / 10)) 
  (h4 : (99999999 : ℕ)) 
  : A = 99999998 ∨ A = 17777779 :=
sorry

end max_min_A_of_coprime_36_and_greater_than_7777777_l254_254171


namespace wire_length_between_poles_l254_254009

theorem wire_length_between_poles :
  ∀ (d h1 h2 : ℝ), d = 20 ∧ h1 = 8 ∧ h2 = 18 →
  (∃ c : ℝ, c = 10 * Real.sqrt 5 ∧ c^2 = d^2 + (h2 - h1)^2) := by
  intros d h1 h2 H
  obtain ⟨Hd, Hh1, Hh2⟩ := H
  use 10 * Real.sqrt 5
  rw [Hd, Hh1, Hh2]
  field_simp
  rw [pow_two, pow_two, pow_two, Real.sqrt_mul, Real.sqrt_sq_eq_abs, abs_of_pos]
  norm_num
  norm_num
  norm_num
  exact Real.sqrt_nonneg _
  sorry

end wire_length_between_poles_l254_254009


namespace smallest_number_bounds_l254_254050

theorem smallest_number_bounds (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos : ∀ i, 0 < ∑ j, if j = i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (Finset.univ_nonempty)) a ∧
  (Finset.min' Finset.univ (Finset.univ_nonempty)) a ≤ 1 / 6 :=
begin
  sorry
end

end smallest_number_bounds_l254_254050


namespace intersection_M_N_l254_254771

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254771


namespace circle_tangent_locus_l254_254415

theorem circle_tangent_locus (a b : ℝ) :
  (∃ r : ℝ, (a ^ 2 + b ^ 2 = (r + 1) ^ 2) ∧ ((a - 3) ^ 2 + b ^ 2 = (5 - r) ^ 2)) →
  3 * a ^ 2 + 4 * b ^ 2 - 14 * a - 49 = 0 := by
  sorry

end circle_tangent_locus_l254_254415


namespace easter_eggs_problem_l254_254384

noncomputable def mia_rate : ℕ := 24
noncomputable def billy_rate : ℕ := 10
noncomputable def total_hours : ℕ := 5
noncomputable def total_eggs : ℕ := 170

theorem easter_eggs_problem :
  (mia_rate + billy_rate) * total_hours = total_eggs :=
by
  sorry

end easter_eggs_problem_l254_254384


namespace intersection_M_N_l254_254781

def M : Set ℝ := {2, 4, 6, 8, 10}

def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N :
  M ∩ N = {2, 4} :=
by 
  sorry

end intersection_M_N_l254_254781


namespace intersection_M_N_l254_254627

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254627


namespace sequence_sum_l254_254375

theorem sequence_sum (m : ℕ) (y : ℕ → ℕ) 
    (h0 : y 0 = 1) 
    (h1 : y 1 = m)
    (h_rec : ∀ k ≥ 0, y (k + 2) = (m * y (k + 1) - (m + k) * y k) / (k + 1))
    : (∑ k in finset.range (m + 1), y k) = 2^m := 
sorry

end sequence_sum_l254_254375


namespace sales_fifth_month_l254_254131

theorem sales_fifth_month (s1 s2 s3 s4 s6 s5 : ℝ) (target_avg total_sales : ℝ)
  (h1 : s1 = 4000)
  (h2 : s2 = 6524)
  (h3 : s3 = 5689)
  (h4 : s4 = 7230)
  (h6 : s6 = 12557)
  (h_avg : target_avg = 7000)
  (h_total_sales : total_sales = 42000) :
  s5 = 6000 :=
by
  sorry

end sales_fifth_month_l254_254131


namespace equivalent_knicks_l254_254286

theorem equivalent_knicks (knicks knacks knocks : ℕ) (h1 : 5 * knicks = 3 * knacks) (h2 : 4 * knacks = 6 * knocks) :
  36 * knocks = 40 * knicks :=
by
  sorry

end equivalent_knicks_l254_254286


namespace h_at_3_l254_254570

-- Define the function h(x) given the condition
noncomputable def h (x : ℝ) : ℝ := 
  ((x + 1) * (x^2 + 1) * (x^4 + 1) * (x^8 + 1) * (x^16 + 1) * (x^32 + 1)
  * (x^64 + 1) * (x^128 + 1) * (x^256 + 1) * (x^512 + 1) * (x^1024 + 1)
  * (x^2048 + 1) * (x^4096 + 1) * (x^8192 + 1) * (x^16384 + 1) * (x^32768 + 1) 
  * (x^65536 + 1) * (x^131072 + 1) * (x^262144 + 1) * (x^524288 + 1)) - 1) / (x^(2^2009 - 1) - 1)

-- Proof statement that h(3) = 3
theorem h_at_3 : h 3 = 3 := by
  sorry

end h_at_3_l254_254570


namespace complex_problem_l254_254215

theorem complex_problem (a b : ℝ) (h : (⟨a, 3⟩ : ℂ) + ⟨2, -1⟩ = ⟨5, b⟩) : a * b = 6 := by
  sorry

end complex_problem_l254_254215


namespace cosine_of_angle_between_diagonals_l254_254139

open Real

def vector_a : ℝ × ℝ × ℝ := (3, 0, 1)
def vector_b : ℝ × ℝ × ℝ := (1, 2, -1)

noncomputable def cos_theta_between_diagonals (a b : ℝ × ℝ × ℝ) : ℝ :=
  let vector_add := (a.1 + b.1, a.2 + b.2, a.3 + b.3)
  let vector_sub := (b.1 - a.1, b.2 - a.2, b.3 - a.3)
  let dot_product := vector_add.1 * vector_sub.1 + vector_add.2 * vector_sub.2 + vector_add.3 * vector_sub.3
  let magnitude_add := sqrt (vector_add.1^2 + vector_add.2^2 + vector_add.3^2)
  let magnitude_sub := sqrt (vector_sub.1^2 + vector_sub.2^2 + vector_sub.3^2)
  dot_product / (magnitude_add * magnitude_sub)

theorem cosine_of_angle_between_diagonals : cos_theta_between_diagonals vector_a vector_b = -1 / sqrt 15 := by
  sorry

end cosine_of_angle_between_diagonals_l254_254139


namespace smallest_value_range_l254_254043

theorem smallest_value_range {a : Fin 8 → ℝ}
  (h_sum : (∑ i, a i) = 4/3)
  (h_pos_7 : ∀ i : Fin 8, (∑ j in Finset.erase Finset.univ i, a j) > 0) :
  -8 < a 0 ∧ a 0 ≤ 1/6 :=
sorry

end smallest_value_range_l254_254043


namespace intersection_M_N_l254_254680

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254680


namespace sin_alpha_beta_one_l254_254833

theorem sin_alpha_beta_one (α β : ℝ) (h1 : Real.sin α + Real.cos β = 1) (h2 : Real.cos α + Real.sin β = √3) 
  : Real.sin (α + β) = 1 := 
  sorry

end sin_alpha_beta_one_l254_254833


namespace M_inter_N_eq_2_4_l254_254826

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254826


namespace niu_fraction_property_l254_254291

open Nat

-- Given mn <= 2009, where m, n are positive integers and (n/m) is in lowest terms
-- Prove that for adjacent terms in the sequence, m_k n_{k+1} - m_{k+1} n_k = 1.

noncomputable def is_numerator_denom_pair_in_seq (m n : ℕ) : Bool :=
  m > 0 ∧ n > 0 ∧ m * n ≤ 2009

noncomputable def are_sorted_adjacent_in_seq (m_k n_k m_k1 n_k1 : ℕ) : Bool :=
  m_k * n_k1 - m_k1 * n_k = 1

theorem niu_fraction_property :
  ∀ (m_k n_k m_k1 n_k1 : ℕ),
  is_numerator_denom_pair_in_seq m_k n_k →
  is_numerator_denom_pair_in_seq m_k1 n_k1 →
  m_k < m_k1 →
  are_sorted_adjacent_in_seq m_k n_k m_k1 n_k1
:=
sorry

end niu_fraction_property_l254_254291


namespace intersection_cardinality_l254_254229

theorem intersection_cardinality :
  let A := { x : ℤ | x^2 - x - 2 ≤ 0 }
  let B := { x : ℝ | 1 ≤ x }
  #(A ∩ B) = 2 := 
by
  sorry

end intersection_cardinality_l254_254229


namespace dot_product_sum_eq_neg69_l254_254976

open Real

variables (a b c : Vector ℝ 3)

-- Conditions
def norm_a : ‖a‖ = 5 := sorry 
def norm_b : ‖b‖ = 7 := sorry
def norm_c : ‖c‖ = 8 := sorry
def vec_sum_zero : a + b + c = 0 := sorry

-- Theorem to prove
theorem dot_product_sum_eq_neg69 : 
  (a • b) + (a • c) + (b • c) = -69 := 
by {
  -- proof goes here
  sorry
}

end dot_product_sum_eq_neg69_l254_254976


namespace smallest_divisible_by_4_l254_254018

def base_layer := {a b c d e f g h i : ℕ // a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ a ≠ i ∧
                              b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ b ≠ i ∧
                              c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ c ≠ i ∧
                              d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
                              e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
                              f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
                              g ≠ h ∧ g ≠ i ∧
                              h ≠ i }

def top_cube_value (b : base_layer) : ℕ :=
  b.val.a + b.val.c + b.val.g + b.val.i + 2 * (b.val.b + b.val.d + b.val.f + b.val.h) + 4 * b.val.e

theorem smallest_divisible_by_4 :
  ∃ b : base_layer, top_cube_value b = 64 := 
sorry

end smallest_divisible_by_4_l254_254018


namespace M_inter_N_eq_2_4_l254_254819

def M : Set ℕ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem M_inter_N_eq_2_4 : M ∩ (N ∩ Set.univ_ℕ) = {2, 4} :=
by
  sorry

end M_inter_N_eq_2_4_l254_254819


namespace solve_equation_l254_254446

theorem solve_equation (x : ℝ) : (x - 1) * (x + 1) = x - 1 ↔ x = 1 ∨ x = 0 :=
by
  sorry

end solve_equation_l254_254446


namespace problem_1_problem_2_problem_3_l254_254843

noncomputable def f (x : ℝ) (a : ℝ) := (a + log x) / x
noncomputable def f_prime (x : ℝ) (a : ℝ) := (1 - a - log x) / x^2

theorem problem_1 (h : f_prime real.exp a = -1 / (real.exp ^ 2)) : a = 1 := by
  sorry

theorem problem_2 (h : ∃ x : ℝ, x ∈ set.Ioo m (m + 1) ∧ (∀ y : ℝ, y ∈ set.Ioo (0 : ℝ) 1 → f_prime y 1 > 0) ∧ (∀ y : ℝ, y > 1 → f_prime y 1 < 0)) : 0 < m ∧ m < 1 := by
  sorry

theorem problem_3 (x : ℝ) (h : 1 < x) : (f x 1 > 2 / (x + 1)) := by
  sorry

end problem_1_problem_2_problem_3_l254_254843


namespace servant_leave_months_l254_254271

/-- Define the conditions --/
def total_salary (turban_value: ℤ) (annual_cash_salary: ℤ): ℤ :=
  annual_cash_salary + turban_value

def monthly_salary (total_annual_salary: ℤ) : ℚ :=
  total_annual_salary / 12

def leave_amount (turban_value: ℤ) (leave_cash: ℤ): ℤ :=
  leave_cash + turban_value

/-- Define the problem statement --/
theorem servant_leave_months :
  let turban_value := 70 in
  let annual_cash_salary := 90 in
  let leave_cash := 50 in
  let total_annual_salary := total_salary turban_value annual_cash_salary in
  let per_month := monthly_salary total_annual_salary in
  let received_leave_amount := leave_amount turban_value leave_cash in
  (received_leave_amount : ℚ) / per_month ≈ (9 : ℚ) :=
by
  sorry

end servant_leave_months_l254_254271


namespace min_max_value_sum_l254_254366

variable (a b c d e : ℝ)

theorem min_max_value_sum :
  a + b + c + d + e = 10 ∧ a^2 + b^2 + c^2 + d^2 + e^2 = 30 →
  let expr := 5 * (a^3 + b^3 + c^3 + d^3 + e^3) - (a^4 + b^4 + c^4 + d^4 + e^4)
  let m := 42
  let M := 52
  m + M = 94 := sorry

end min_max_value_sum_l254_254366


namespace intersection_M_N_l254_254802

open Set

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := by
  sorry

end intersection_M_N_l254_254802


namespace sum_of_digits_of_n_l254_254066

theorem sum_of_digits_of_n :
  ∃ (n : ℕ), 0 < n ∧ (n + 1)! + (n + 2)! = n! * 630 ∧ (n / 10) + (n % 10) = 5 :=
sorry

end sum_of_digits_of_n_l254_254066


namespace hyperbola_asymptote_l254_254837

theorem hyperbola_asymptote (a : ℝ) (h : y = - (sqrt 3) * x) : (1 / a) = sqrt 3 → a = sqrt 3 / 3 :=
begin
  sorry
end

end hyperbola_asymptote_l254_254837


namespace increase_in_difference_between_strawberries_and_blueberries_l254_254464

theorem increase_in_difference_between_strawberries_and_blueberries :
  ∀ (B S : ℕ), B = 32 → S = B + 12 → (S - B) = 12 :=
by
  intros B S hB hS
  sorry

end increase_in_difference_between_strawberries_and_blueberries_l254_254464


namespace avg_consecutive_odd_primes_composite_l254_254397

theorem avg_consecutive_odd_primes_composite (p p' : ℕ) (h1 : Nat.Prime p) (h2 : Nat.Prime p') 
  (h3 : p < p') (h4 : odd p) (h5 : odd p') (h6 : ∀ n, p < n ∧ n < p' → ¬ Nat.Prime n) 
  : ∃ c, c ∣ (p + p') / 2 ∧ c ≠ 1 ∧ c ≠ (p + p') / 2 :=
by
  sorry

end avg_consecutive_odd_primes_composite_l254_254397


namespace num_sequences_with_zero_l254_254364

/-- Define the function generating the sequence according to the rule. -/
def generate_seq (b : Fin 20 → ℕ) (n : ℕ) : ℕ :=
  if h : n < 3 then b ⟨n, by linarith⟩
  else generate_seq b (n - 1) * (abs (generate_seq b (n - 2) - generate_seq b (n - 3)))

/-- Check if the sequence contains a zero. -/
def contains_zero (b : Fin 20 → ℕ) : Prop :=
  ∃ n, generate_seq b n = 0

/-- Count the valid ordered triples. -/
def count_valid_triples : ℕ :=
  Finset.card {b : Fin 20 → ℕ | contains_zero b}

/-- Main theorem statement. -/
theorem num_sequences_with_zero : count_valid_triples = 780 := by
  sorry

end num_sequences_with_zero_l254_254364


namespace remainder_not_power_of_4_l254_254567

theorem remainder_not_power_of_4 : ∃ n : ℕ, n ≥ 2 ∧ ¬ (∃ k : ℕ, (2^2^n) % (2^n - 1) = 4^k) := sorry

end remainder_not_power_of_4_l254_254567


namespace derivative_at_1_l254_254253

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem derivative_at_1 : deriv f 1 = 1 + Real.cos 1 :=
by
  sorry

end derivative_at_1_l254_254253


namespace probability_sum_30_l254_254156

-- Define the sets representing the faces of the two dice
def die1_faces : Set ℕ := { n | (n ∈ Finset.range 18) ∨ n = 19 }
def die2_faces : Set ℕ := { n | (n ∈ Finset.range 16) ∨ (n ∈ Finset.Icc 17 20) }

-- Define the rolling function for both dice
def valid_pairs : Set (ℕ × ℕ) := { (a, b) | a ∈ die1_faces ∧ b ∈ die2_faces ∧ a + b = 30 }

-- Define the total number of possible outcomes
def total_outcomes : ℕ := 400

-- Prove the probability statement
theorem probability_sum_30 : 
  (valid_pairs.card : ℚ) / total_outcomes = 1 / 100 :=
sorry

end probability_sum_30_l254_254156


namespace smallest_number_bounds_l254_254048

theorem smallest_number_bounds (a : Fin 8 → ℝ)
  (h_sum : ∑ i, a i = 4 / 3)
  (h_pos : ∀ i, 0 < ∑ j, if j = i then 0 else a j) :
  -8 < (Finset.min' Finset.univ (Finset.univ_nonempty)) a ∧
  (Finset.min' Finset.univ (Finset.univ_nonempty)) a ≤ 1 / 6 :=
begin
  sorry
end

end smallest_number_bounds_l254_254048


namespace intersection_M_N_l254_254686

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254686


namespace intersection_of_M_and_N_l254_254746

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254746


namespace perfectCubesCount_l254_254882

theorem perfectCubesCount (a b : Nat) (h₁ : 50 < a ∧ a ^ 3 > 50) (h₂ : b ^ 3 < 2000 ∧ b < 2000) :
  let n := b - a + 1
  n = 9 := by
  sorry

end perfectCubesCount_l254_254882


namespace intersection_M_N_l254_254681

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254681


namespace intersection_of_M_and_N_l254_254642

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := 
  sorry

end intersection_of_M_and_N_l254_254642


namespace intersection_M_N_l254_254685

def M := {2, 4, 6, 8, 10}

def N := {x : ℝ | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254685


namespace points_are_concyclic_l254_254320

variables {A B C H : Type}
variables (M N P Q : Type)
variables [altitude_from_C : Altitude C A B] [altitude_from_B: Altitude B A C]
variables (circle_AB : Circumcircle A B) (circle_AC : Circumcircle A C)
variables {H_M : ∈ circle_AB A B}
variables {H_N : ∈ circle_AB}
variables {H_P : ∈ circle_AC}
variables {H_Q : ∈ circle_AC}

theorem points_are_concyclic
  (h1 : altitude_from_C intersects circle_AB at M)
  (h2 : altitude_from_C intersects circle_AB at N)
  (h3 : altitude_from_B intersects circle_AC at P)
  (h4 : altitude_from_B intersects circle_AC at Q) :
  Concyclic M N P Q :=
by 
  sorry

end points_are_concyclic_l254_254320


namespace find_unit_vector_in_xy_plane_l254_254207

-- Define unit vector in the xy-plane
structure unit_vector (α : Type*) [field α] :=
(x y : α)
(norm_eq_one : x^2 + y^2 = 1)

noncomputable def proof_problem : Prop :=
∃ (u : unit_vector ℝ), 
  (3 * u.x + u.y) / real.sqrt 10 = real.sqrt 3 / 2 ∧
  (u.x + 2 * u.y) / real.sqrt 5 = 1 / real.sqrt 2

theorem find_unit_vector_in_xy_plane : proof_problem :=
sorry

end find_unit_vector_in_xy_plane_l254_254207


namespace intersection_M_N_l254_254717

def M : set ℕ := {2, 4, 6, 8, 10}
def N : set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} := 
sorry

end intersection_M_N_l254_254717


namespace smallest_a1_range_l254_254051

noncomputable def smallest_a1_possible (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) : Prop :=
  a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = (4 : ℝ) / (3 : ℝ) ∧ 
  (∀ i : Fin₈, (a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - a_i) > 0) ∧ 
  (-8 < a_1 ∧ a_1 ≤ 1 / 6)

-- Since we are only required to state the problem, we leave the proof as a "sorry".
theorem smallest_a1_range (a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ)
    (h_sum: a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 = 4 / 3)
    (h_pos: ∀ i : Fin 8, a_1 + a_2 + a_3 + a_4 + a_5 + a_6 + a_7 + a_8 - ([-][i]) > 0):
    smallest_a1_possible a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 :=
  by
  sorry

end smallest_a1_range_l254_254051


namespace vector_division_by_three_l254_254895

def OA : ℝ × ℝ := (2, 8)
def OB : ℝ × ℝ := (-7, 2)
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
noncomputable def scalar_mult (k : ℝ) (u : ℝ × ℝ) : ℝ × ℝ := (k * u.1, k * u.2)

theorem vector_division_by_three :
  scalar_mult (1 / 3) (vector_sub OB OA) = (-3, -2) :=
sorry

end vector_division_by_three_l254_254895


namespace Probability_X_in_Interval_0_4_l254_254302

noncomputable def X : Type := sorry

def normal_distribution (μ σ: ℝ) (σ_pos: σ > 0) :  X → ℝ := sorry
def prob_event : set X → ℝ := sorry

variable (σ : ℝ)
variable (hσ : σ > 0)
variable (P : ℝ)

-- Given X follows a normal distribution N(4, σ^2)
axiom h1 : normal_distribution 4 σ hσ

-- Given the probability of X in interval (0, 8) is 0.6
axiom h2 : prob_event {x : X | 0 < x ∧ x < 8} = 0.6

-- Prove the probability of X in interval (0, 4) is 0.3
theorem Probability_X_in_Interval_0_4 : prob_event {x : X | 0 < x ∧ x < 4} = 0.3 := 
by sorry

end Probability_X_in_Interval_0_4_l254_254302


namespace intersection_M_N_l254_254624

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_M_N : M ∩ N = {2, 4} :=
sorry

end intersection_M_N_l254_254624


namespace intersection_of_M_and_N_l254_254748

-- Define the given sets M and N
def M : Set ℤ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the proof problem
theorem intersection_of_M_and_N : (M : Set ℝ) ∩ N = {2, 4} :=
sorry

end intersection_of_M_and_N_l254_254748


namespace cannot_represent_diagonals_l254_254493

theorem cannot_represent_diagonals (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 7) (h₂ : c = 9) : 
  ¬((a^2 + b^2 > c^2) ∧ (b^2 + c^2 > a^2) ∧ (a^2 + c^2 > b^2)) :=
by
  -- Definitions of diagonals based on Pythagorean theorem
  let d1 := a^2 + b^2
  let d2 := b^2 + c^2
  let d3 := a^2 + c^2
  -- Diagonal squares
  have h_d1 : d1 = 74 := by sorry
  have h_d2 : d2 = 130 := by sorry
  have h_d3 : d3 = 106 := by sorry
  -- External diagonal validation
  finish

end cannot_represent_diagonals_l254_254493


namespace irrational_count_l254_254548

theorem irrational_count :
  let nums := [-(2/3 : ℝ), 0, -(2/3 : ℝ) * Real.sqrt 6, -Real.pi, Real.sqrt 4, Real.cbrt 27] in
  (nums.filter (λ x => ¬ Rational.isRational x)).length = 2 :=
by
  sorry

end irrational_count_l254_254548


namespace KayleeAgeCorrect_l254_254297

-- Define Kaylee's current age
def KayleeCurrentAge (k : ℕ) : Prop :=
  (3 * 5 + (7 - k) = 7)

-- State the theorem
theorem KayleeAgeCorrect : ∃ k : ℕ, KayleeCurrentAge k ∧ k = 8 := 
sorry

end KayleeAgeCorrect_l254_254297


namespace intersection_of_M_and_N_l254_254789

-- Define the sets M and N
def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {2, 4} :=
by
  sorry

end intersection_of_M_and_N_l254_254789


namespace total_genuine_purses_and_handbags_l254_254457

def TirzahPurses : ℕ := 26
def TirzahHandbags : ℕ := 24
def FakePurses : ℕ := TirzahPurses / 2
def FakeHandbags : ℕ := TirzahHandbags / 4
def GenuinePurses : ℕ := TirzahPurses - FakePurses
def GenuineHandbags : ℕ := TirzahHandbags - FakeHandbags

theorem total_genuine_purses_and_handbags : GenuinePurses + GenuineHandbags = 31 := by
  sorry

end total_genuine_purses_and_handbags_l254_254457


namespace max_distinct_numbers_l254_254478

theorem max_distinct_numbers : 
  ∃ (S : Finset ℕ), S.card = 400 ∧ S ⊆ (Finset.range 1000).filter (λ x, 1 ≤ x ∧ x ≤ 1000) ∧ 
    ∀ x y ∈ S, x ≠ y → (abs (x - y) ≠ 4 ∧ abs (x - y) ≠ 5 ∧ abs (x - y) ≠ 6) :=
by
  sorry

end max_distinct_numbers_l254_254478


namespace probability_calculation_l254_254368

noncomputable def probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 : ℝ :=
  let total_interval_length := 100
  let intersection_interval_length := 324 - 312.5
  intersection_interval_length / total_interval_length

theorem probability_calculation : probability_floor_sqrt_x_eq_17_given_floor_sqrt_2x_eq_25 = 23 / 200 := by
  sorry

end probability_calculation_l254_254368


namespace carla_max_correct_answers_l254_254912

-- Define the conditions as premises in the Lean statement.
theorem carla_max_correct_answers (a b c : ℕ) (h1 : a + b + c = 60) (h2 : 5 * a - 2 * c = 150) :
  a ≤ 38 :=
begin
  sorry
end

end carla_max_correct_answers_l254_254912


namespace equal_costs_l254_254069

noncomputable def cost_scheme_1 (x : ℕ) : ℝ := 350 + 5 * x

noncomputable def cost_scheme_2 (x : ℕ) : ℝ := 360 + 4.5 * x

theorem equal_costs (x : ℕ) : cost_scheme_1 x = cost_scheme_2 x ↔ x = 20 := by
  sorry

end equal_costs_l254_254069


namespace total_vegetables_l254_254061

-- Definitions for the conditions in the problem
def cucumbers := 58
def carrots := cucumbers - 24
def tomatoes := cucumbers + 49
def radishes := carrots

-- Statement for the proof problem
theorem total_vegetables :
  cucumbers + carrots + tomatoes + radishes = 233 :=
by sorry

end total_vegetables_l254_254061


namespace eccentricity_hyperbola_l254_254260

theorem eccentricity_hyperbola 
  (a b : ℝ) (h₁ : ∀ x y : ℝ, (y = 2 * x) → (y = (b / a) * x))
  (h₂ : a > 0) (h₃ : b = 2 * a) :
  ∃ e : ℝ, (e = sqrt 5 ∨ e = sqrt 5 / 2) := by
  sorry

end eccentricity_hyperbola_l254_254260


namespace zoe_pop_albums_l254_254494

theorem zoe_pop_albums (total_songs country_albums songs_per_album : ℕ) (h1 : total_songs = 24) (h2 : country_albums = 3) (h3 : songs_per_album = 3) :
  total_songs - (country_albums * songs_per_album) = 15 ↔ (total_songs - (country_albums * songs_per_album)) / songs_per_album = 5 :=
by
  sorry

end zoe_pop_albums_l254_254494


namespace farmer_finished_ahead_l254_254129

def productivity_initial := 120
def daily_area_initial := 2 * productivity_initial
def total_area := 1440
def productivity_increase := 0.25
def productivity_new := productivity_initial + productivity_initial * productivity_increase
def remaining_area := total_area - daily_area_initial
def days_initial := total_area / productivity_initial
def days_new := remaining_area / productivity_new
def days_total := 2 + days_new
def days_ahead := days_initial - days_total

theorem farmer_finished_ahead : days_ahead = 2 := by
  -- computations are automatically handled by Lean's computational engine
  sorry

end farmer_finished_ahead_l254_254129
