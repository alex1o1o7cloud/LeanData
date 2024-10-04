import Mathlib

namespace lavinia_son_age_difference_l138_138944

open Nat

-- Definitions based on conditions
def K_d : ℕ := 12
def L_d : ℕ := K_d / 3
def L_s_initial : ℕ := 2 * K_d
def correct_L_s : ℕ := 29 - 4
def K_s : ℕ := correct_L_s - 3

-- Statement to prove
theorem lavinia_son_age_difference : L_s_initial > L_d ∧ correct_L_s - L_d = 21 :=
by
  have h1 : L_d = K_d / 3 := rfl
  have h2 : L_d = 4 := by norm_num
  have h3 : L_s_initial = 2 * K_d := rfl
  have h4 : correct_L_s = 29 - 4 := rfl
  have h5 : correct_L_s - L_d = 21 := by norm_num
  split
  exact Nat.gt_of_gt_of_ge (by norm_num : 24 > 4) (by norm_num : 25 ≥ 24)
  exact h5

end lavinia_son_age_difference_l138_138944


namespace sapling_height_l138_138762

theorem sapling_height (n : ℕ) : height n = 1.5 + 0.2 * n :=
by
  let h0 := 1.5
  let g := 0.2
  let height := λ n, h0 + g * n
  sorry

end sapling_height_l138_138762


namespace super_knight_tour_impossible_l138_138011

theorem super_knight_tour_impossible :
  ¬ ∃ (tour: Fin 12 × Fin 12 → Fin 144), 
    (∀ v: Fin 12 × Fin 12, ∃ n: Nat, tour v = (n % 12, n / 12)) ∧
    (∀ n: Fin 144, 
      ((tour (Fin.mk (n % 12) (by sorry), Fin.mk (n / 12) (by sorry))) = 
      (Fin.mk ((n + 4) % 12) (by sorry), Fin.mk ((n + 3) % 12) (by sorry))) ∨
      ((tour (Fin.mk (n % 12) (by sorry), Fin.mk (n / 12) (by sorry))) = 
      (Fin.mk ((n + 8) % 12) (by sorry), Fin.mk ((n + 6) % 12) (by sorry))))
       ∧
      (tour (Fin.mk 0 _) = (11, 11) ∧ tour (Fin.mk 11 _) = (0, 0))) :=
  sorry

end super_knight_tour_impossible_l138_138011


namespace mass_percentage_of_Ba_in_mixture_is_correct_l138_138755

theorem mass_percentage_of_Ba_in_mixture_is_correct :
  ∀ (mass_percent_BaI2 mass_percent_BaSO4 mass_percent_BaNO3 : ℝ),
  mass_percent_BaI2 = 30 →
  mass_percent_BaSO4 = 20 →
  mass_percent_BaNO3 = 50 →
  (137.33 / 391.13 * mass_percent_BaI2 / 100 + 
   137.33 / 233.40 * mass_percent_BaSO4 / 100 + 
   137.33 / 261.35 * mass_percent_BaNO3 / 100) * 100 = 48.579 :=
by
  intros mass_percent_BaI2 mass_percent_BaSO4 mass_percent_BaNO3 hB1 hB2 hB3
  have : 137.33 / 391.13 * 30 / 100 + 137.33 / 233.40 * 20 / 100 + 137.33 / 261.35 * 50 / 100 = 0.48579 := by
    sorry
  linarith

end mass_percentage_of_Ba_in_mixture_is_correct_l138_138755


namespace find_smallest_n_l138_138271

noncomputable def sequence (n : ℕ) : ℕ → ℝ
| 0       := 9
| (n + 1) := (4 - sequence n) / 3

def sum_sequence (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ k, sequence k)

def meets_inequality (n : ℕ) : Prop :=
  |sum_sequence n - n - 6| < 1 / 125

theorem find_smallest_n : ∃ n : ℕ, meets_inequality n ∧ ∀ m < n, ¬ meets_inequality m :=
  ⟨7, by sorry, by sorry⟩

end find_smallest_n_l138_138271


namespace correct_statements_l138_138775

/-- Statement 1: The correlation coefficient r is used to measure the strength of the linear
relationship between two variables. The closer |r| is to 1, the weaker the correlation. -/
def statement1 : Prop :=
  ∀ (r : ℝ), (|r| < 1 → abs r < abs (1 - r))

/-- Statement 2: The regression line y = bx + a always passes through the center of the sample points
(𝑥̅, 𝑦̅). -/
def statement2 : Prop :=
  ∀ (x y : ℝ) (a b : ℝ), let x̅ := (x + a) / 2, y̅ := (y + b) / 2 in
  y̅ = b * x̅ + a

/-- Statement 3: The variance of the random error e, denoted as D(e), is used to measure the accuracy
of the forecast. -/
def statement3 : Prop :=
  ∀ (e : ℝ), let D := e ^ 2 in (D e < e)

/-- Statement 4: The coefficient of determination R^2 is used to characterize the effectiveness of
the regression. The smaller R^2 is, the better the model fits. -/
def statement4 : Prop :=
  ∀ (R_squared : ℝ), (R_squared < 1 → abs (1 - R_squared))

/-- Proof that only statements 2 and 3 are correct, while 1 and 4 are incorrect. -/
theorem correct_statements : ¬ statement1 ∧ statement2 ∧ statement3 ∧ ¬ statement4 :=
by
  sorry

end correct_statements_l138_138775


namespace probability_factor_of_36_l138_138637

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138637


namespace digit_in_421st_place_l138_138709

noncomputable def decimal_rep := "368421052631578947"

theorem digit_in_421st_place : 
  ∀ (n : ℕ), n = 421 → 
  (let m := 19, a := 7 in a / m = 0.repeating_decimal sequence_of (decimal_rep, 18) at position n) :=
by 
  sorry

end digit_in_421st_place_l138_138709


namespace hypotenuse_length_l138_138905

theorem hypotenuse_length (a b c : ℝ)
  (h1 : b = a + 10)
  (h2 : a^2 + b^2 + c^2 = 2450)
  (h3 : c^2 = a^2 + b^2) :
  | c - 53.958 | < 1e-3 :=
by
  sorry  -- Proof is omitted

end hypotenuse_length_l138_138905


namespace remaining_sugar_l138_138163

/-- Chelsea has 24 kilos of sugar. She divides them into 4 bags equally.
  Then one of the bags gets torn and half of the sugar falls to the ground.
  How many kilos of sugar remain? --/
theorem remaining_sugar (total_sugar : ℕ) (bags : ℕ) (torn_bag_fraction : ℚ) (initial_per_bag : ℕ) (fallen_sugar : ℕ) :
  total_sugar = 24 →
  bags = 4 →
  (total_sugar / bags) = initial_per_bag →
  initial_per_bag = 6 →
  (initial_per_bag * torn_bag_fraction) = fallen_sugar →
  torn_bag_fraction = 1/2 →
  fallen_sugar = 3 →
  (total_sugar - fallen_sugar) = 21 :=
begin
  intros h_total h_bags h_initial_per_bag_eq h_initial_per_bag h_torn_bag_fraction_eq h_torn_bag_fraction h_fallen_sugar,
  rw [h_total, h_initial_per_bag_eq.symm, h_bags],
  norm_num at *,
  sorry
end

end remaining_sugar_l138_138163


namespace probability_divisor_of_36_is_one_fourth_l138_138468

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138468


namespace inequality_proof_l138_138957

theorem inequality_proof (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x + y > 2) : (1 + x) / y < 2 ∨ (1 + y) / x < 2 :=
sorry

end inequality_proof_l138_138957


namespace probability_factor_of_36_l138_138571

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138571


namespace sequence_sum_l138_138702

theorem sequence_sum :
  1 - 4 + 7 - 10 + 13 - 16 + 19 - 22 + 25 - 28 + 31 - 34 + 37 - 40 + 43 - 46 + 49 - 52 + 55 = 28 :=
by
  sorry

end sequence_sum_l138_138702


namespace incenter_quad_is_rectangle_l138_138385

theorem incenter_quad_is_rectangle {A B C D : Point} (hA : IsInCircle A B C D) 
  (hI_A : is_incenter (Triangle B.C.D) I_A) 
  (hI_B : is_incenter (Triangle A.C.D) I_B) 
  (hI_C : is_incenter (Triangle A.B.D) I_C) 
  (hI_D : is_incenter (Triangle A.B.C) I_D) : 
  is_rectangle (Quadrilateral I_A I_B I_C I_D) :=
sorry

end incenter_quad_is_rectangle_l138_138385


namespace general_term_form_sum_first_n_terms_l138_138056

noncomputable def geom_seq_general_term (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a n = a 0 * q ^ n

axiom positive_terms (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, 0 < a n

theorem general_term_form (a : ℕ → ℝ) (q : ℝ)
  (h_pos : positive_terms a)
  (h1 : 2 * a 1 + 3 * (a 1 * q) = 1)
  (h2 : (a 1 * q ^ 2) ^ 2 = 9 * (a 1 * q) * (a 1 * q ^ 5)) :
  geom_seq_general_term a (1 / 3) :=
sorry

-- Define b_n
def b (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, real.log a (i + 1)

-- Define T_n
def T (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, -1 / b a (i + 1)

theorem sum_first_n_terms (a : ℕ → ℝ)
  (h_gen_term : ∀ n : ℕ, a n = 1 / (3 ^ n)) (n : ℕ) :
  T a n = 2 * n / (n + 1) :=
sorry

end general_term_form_sum_first_n_terms_l138_138056


namespace pyramid_surface_area_l138_138127

noncomputable def total_surface_area_of_pyramid (a b : ℝ) (theta : ℝ) (height : ℝ) : ℝ :=
  let base_area := a * b * Real.sin theta
  let slant_height := Real.sqrt (height ^ 2 + (a / 2) ^ 2)
  let lateral_area := 4 * (1 / 2 * a * slant_height)
  base_area + lateral_area

theorem pyramid_surface_area :
  total_surface_area_of_pyramid 12 14 (Real.pi / 3) 15 = 168 * Real.sqrt 3 + 216 * Real.sqrt 29 :=
by sorry

end pyramid_surface_area_l138_138127


namespace pink_cookies_eq_fifty_l138_138369

-- Define the total number of cookies
def total_cookies : ℕ := 86

-- Define the number of red cookies
def red_cookies : ℕ := 36

-- The property we want to prove
theorem pink_cookies_eq_fifty : (total_cookies - red_cookies = 50) :=
by
  sorry

end pink_cookies_eq_fifty_l138_138369


namespace quasiperfect_is_square_of_odd_l138_138153

def is_quasiperfect (N : ℕ) : Prop :=
  ∑ i in (finset.range N.succ).filter (λ d, N % d = 0), d = 2 * N + 1

theorem quasiperfect_is_square_of_odd (N : ℕ) (h : is_quasiperfect N) : 
  ∃ m : ℕ, N = m * m ∧ odd m :=
sorry

end quasiperfect_is_square_of_odd_l138_138153


namespace probability_factor_of_36_l138_138641

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138641


namespace find_a5_l138_138176

/-- Define the sequence (a_n) with the initial condition a_0 and the recurrence relation. -/
noncomputable def a : ℕ → ℚ
| 0       := 0
| (n + 1) := (10 / 6 : ℚ) * a n + (8 / 6 : ℚ) * Real.sqrt (4^n - a n ^ 2)

/-- The target proposition to be proved -/
theorem find_a5 : a 5 = 32000 / 81 := 
sorry

end find_a5_l138_138176


namespace find_a1_S4_l138_138239

variable (q a_1 a_3 a_4 : ℝ) 
variable (S_2 S_4 : ℝ)
variable (n : ℕ)

-- Definitions of the sequence and sums.
def geometric_sequence (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := a_1 * q^(n-1)

def sum_geometric_sequence (a_1 : ℝ) (q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then n * a_1 else a_1 * (1 - q^n) / (1 - q)

-- given conditions
axiom a3_is_2 : geometric_sequence a_1 q 3 = 2
axiom S4_is_5S2 : sum_geometric_sequence a_1 q 4 = 5 * sum_geometric_sequence a_1 q 2
axiom q_positive : q > 0

theorem find_a1_S4 (h1 : geometric_sequence a_1 q 3 = 2)
                   (h2 : sum_geometric_sequence a_1 q 4 = 5 * sum_geometric_sequence a_1 q 2)
                   (h3 : q > 0) :
                   a_1 = 1/2 ∧ sum_geometric_sequence a_1 q 4 = 15/2 :=
  by sorry

end find_a1_S4_l138_138239


namespace truncated_trigonal_pyramid_circumscribed_sphere_l138_138769

theorem truncated_trigonal_pyramid_circumscribed_sphere
  (h R_1 R_2 : ℝ)
  (O_1 T_1 O_2 T_2 : ℝ)
  (circumscribed : ∃ r : ℝ, h = 2 * r)
  (sphere_touches_lower_base : ∀ P, dist P T_1 = r)
  (sphere_touches_upper_base : ∀ Q, dist Q T_2 = r)
  (dist_O1_T1 : ℝ)
  (dist_O2_T2 : ℝ) :
  R_1 * R_2 * h^2 = (R_1^2 - dist_O1_T1^2) * (R_2^2 - dist_O2_T2^2) :=
sorry

end truncated_trigonal_pyramid_circumscribed_sphere_l138_138769


namespace find_length_LD_l138_138846

variable {Point : Type}
variable (A B C D L K : Point)
variable (KD CL LD : ℝ)
variable (angle : Point → Point → Point → ℝ)

-- Conditions from the problem
axiom square_ABCD : (A = B ∧ B = C ∧ C = D ∧ D = A)
axiom L_on_CD : (L = C ∨ L = D)
axiom K_on_extension_DA : ¬(K = A) ∧ -- K is not equal to A, but on the extension
axiom angle_KBL_90 : angle K B L = 90
axiom KD_19 : KD = 19
axiom CL_6 : CL = 6

-- Proof goal (finding the length of LD equal to 7)
theorem find_length_LD : LD = 7 := sorry

end find_length_LD_l138_138846


namespace probability_factor_36_l138_138518

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138518


namespace highTrendPermutationsCount_l138_138123

def isHighTrend (perm : List ℕ) : Prop :=
  perm.length = 6 ∧ 
  perm.nth 2 = some 3 ∧
  perm.nth 3 = some 4 ∧
  (perm.nth 0).iget + (perm.nth 1).iget < (perm.nth 4).iget + (perm.nth 5).iget

def countHighTrendPermutations : ℕ :=
  (permutations [1, 2, 3, 4, 5, 6]).count isHighTrend

theorem highTrendPermutationsCount : countHighTrendPermutations = 4 :=
sorry

end highTrendPermutationsCount_l138_138123


namespace smaller_tetrahedron_volume_proof_l138_138133

noncomputable def tetrahedron_volume_proof_problem
  (side_length : ℝ)
  (smaller_vol_ratio : ℝ)
  (base_area_big : ℝ)
  (base_area_small : ℝ)
  (height_ratio : ℝ)
  (smaller_height : ℝ)
  (volume_small : ℝ) :=
  (side_length = 2) →
  (smaller_vol_ratio = 1/2) →
  (base_area_big = (√3 / 4) * 2^2) →
  (base_area_small = (√3 / 4) * 1^2) →
  (height_ratio = √(2/3)) →
  (smaller_height = height_ratio / 2) →
  (volume_small = (1/3) * base_area_small * smaller_height) →
  (volume_small = (√2) / 12)

theorem smaller_tetrahedron_volume_proof :
  tetrahedron_volume_proof_problem 2 ((1:ℝ) / 2) ((sqrt 3 / 4) * 4) ((sqrt 3 / 4) * 1) (sqrt (2 / 3)) (sqrt (2 / 3) / 2) ((1:ℝ) / 3 * (sqrt 3 / 4) * (sqrt (2 / 3) / 2)) :=
by
  sorry

end smaller_tetrahedron_volume_proof_l138_138133


namespace reciprocal_of_neg_five_l138_138426

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end reciprocal_of_neg_five_l138_138426


namespace area_and_volume_ratios_l138_138780

def regular_tetrahedron (S : ℝ) (V : ℝ) :=
  ∃ (P A B C : ℝ), P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ B ≠ C ∧ A ≠ C
  ∧ is_tetrahedron P A B C
  ∧ face_area P A B = S 
  ∧ face_area P B C = S
  ∧ face_area P C A = S
  ∧ face_area A B C = S
  ∧ tetrahedron_volume P A B C = V

def polyhedron_G (S : ℝ) (V : ℝ) :=
  regular_tetrahedron S V ∧
  ∃ (G : Type), is_polyhedron G ∧
  ∀ face, face ∈ faces_of G → is_hexagonal face ∧ face.area = S / 9

theorem area_and_volume_ratios (S V : ℝ) 
  (h_tetra : regular_tetrahedron S V)
  (h_polyh : polyhedron_G S V) : 
  (surface_area (regular_tetrahedron S V) / surface_area (polyhedron_G S V) = 9 / 7) ∧ 
  (volume (regular_tetrahedron S V) / volume (polyhedron_G S V) = 27 / 23) :=
by
  sorry

end area_and_volume_ratios_l138_138780


namespace ellen_lunch_calories_l138_138190

-- Definitions of the conditions.
def total_daily_calories : ℝ := 2200
def breakfast_calories : ℝ := 353
def snack_calories : ℝ := 130
def dinner_remaining_calories : ℝ := 832

-- Statement of the proof problem
theorem ellen_lunch_calories : ℝ :=
  total_daily_calories - breakfast_calories - snack_calories - dinner_remaining_calories = 885
  sorry

end ellen_lunch_calories_l138_138190


namespace intersection_traces_ellipse_l138_138244

theorem intersection_traces_ellipse (A B C D O : Point) (AB CD AC BD : ℝ)
  (h_AB_eq_CD : AB = CD) (h_fixed_AB : FixedSegment A B AB) (h_equal_diagonals : AC = BD) :
  ∃ (e : Ellipse), (is_in_ellipse e O) ∧ 
                   (foci e = (A, B)) ∧ 
                   (major_axis_length e = AC) ∧ 
                   (O ≠ endpoint_of_major_axis e) :=
sorry

end intersection_traces_ellipse_l138_138244


namespace probability_factor_36_l138_138556

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138556


namespace handshake_count_l138_138441

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end handshake_count_l138_138441


namespace find_length_AE_l138_138093

theorem find_length_AE (AB BC CD DE AC CE AE : ℕ) 
  (h1 : AB = 2) 
  (h2 : BC = 2) 
  (h3 : CD = 5) 
  (h4 : DE = 7)
  (h5 : AC > 2) 
  (h6 : AC < 4) 
  (h7 : CE > 2) 
  (h8 : CE < 5)
  (h9 : AC ≠ CE)
  (h10 : AC ≠ AE)
  (h11 : CE ≠ AE)
  : AE = 5 :=
sorry

end find_length_AE_l138_138093


namespace probability_factor_36_l138_138517

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138517


namespace revolutions_equal_distance_l138_138751

variable (π : Real) -- Declaring π as a real number for the problem

def circumference (r : Real) : Real :=
  2 * π * r

def revolutions (r₁ r₂ : Real) (revs₁ : Nat) : Nat :=
  let C₁ := circumference r₁
  let C₂ := circumference r₂
  revs₁ * (C₁ / C₂)

theorem revolutions_equal_distance :
  let r₁ := 30
  let r₂ := 10
  let revs₁ := 40
  revolutions π r₁ r₂ revs₁ = 120 :=
by
  sorry

end revolutions_equal_distance_l138_138751


namespace probability_factor_of_36_l138_138576

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138576


namespace pills_left_after_2_weeks_l138_138779

theorem pills_left_after_2_weeks (s1 s2 s3: Nat) (f: Nat) : 
  s1 = 3 * 120 → 
  s2 = 2 * 30 → 
  s3 = s1 + s2 → 
  f = 2 * 7 * 5 → 
  s3 - f = 350 :=
by
  intros hs1 hs2 hs3 hf
  rw [hs1, hs2, hs3, hf]
  calc
    (3 * 120) + (2 * 30) - (2 * 7 * 5) = 420 - 70 := by sorry
    _ = 350 := by sorry

end pills_left_after_2_weeks_l138_138779


namespace question1_question2_l138_138266

noncomputable def f (x b c : ℝ) := x^2 + b * x + c

theorem question1 (b c : ℝ) (h : ∀ x : ℝ, 2 * x + b ≤ f x b c) (x : ℝ) (hx : 0 ≤ x) :
  f x b c ≤ (x + c)^2 :=
sorry

theorem question2 (b c m : ℝ) (h : ∀ b c : ℝ, b ≠ c → f c b b - f b b b ≤ m * (c^2 - b^2)) :
  m ≥ 3/2 :=
sorry

end question1_question2_l138_138266


namespace sum_of_cubes_l138_138356

variable {R : Type} [OrderedRing R] [Field R] [DecidableEq R]

theorem sum_of_cubes (a b c : R) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : (a^3 + 12) / a = (b^3 + 12) / b) (h₅ : (b^3 + 12) / b = (c^3 + 12) / c) :
    a^3 + b^3 + c^3 = -36 := by
  sorry

end sum_of_cubes_l138_138356


namespace min_value_of_z_l138_138249

noncomputable def minimum_value_z (x y : ℝ) : ℝ :=
  real.sqrt ((x - 1)^2 + (y - 1)^2) + real.sqrt (x^2 + (y - 2)^2)
  
theorem min_value_of_z : ∃ (x y : ℝ), minimum_value_z x y = real.sqrt 2 := 
sorry

end min_value_of_z_l138_138249


namespace probability_of_three_tails_one_head_in_four_tosses_l138_138291

noncomputable def probability_three_tails_one_head (n : ℕ) : ℚ :=
  if n = 4 then 1 / 4 else 0

theorem probability_of_three_tails_one_head_in_four_tosses :
  probability_three_tails_one_head 4 = 1 / 4 :=
by sorry

end probability_of_three_tails_one_head_in_four_tosses_l138_138291


namespace tangent_segment_le_one_eighth_perimeter_l138_138185

-- Define the structure for a triangle
structure Triangle :=
(a b c : ℝ)  -- Sides of the triangle
(h_pos_a : 0 < a)
(h_pos_b : 0 < b)
(h_pos_c : 0 < c)
(h_triangle_ineq1 : a + b > c)
(h_triangle_ineq2 : b + c > a)
(h_triangle_ineq3 : c + a > b)

-- Define the semiperimeter and perimeter
def semiperimeter (T : Triangle) : ℝ := (T.a + T.b + T.c) / 2
def perimeter (T : Triangle) : ℝ := 2 * semiperimeter T

-- State the math proof problem as a theorem
theorem tangent_segment_le_one_eighth_perimeter (T : Triangle) :
  ∀ (PQ : ℝ), (PQ = ... ) → PQ ≤ (perimeter T) / 8 := 
sorry

end tangent_segment_le_one_eighth_perimeter_l138_138185


namespace cube_root_25360000_l138_138285

theorem cube_root_25360000 :
  (real.cbrt 25.36 = 2.938) →
  (real.cbrt 253.6 = 6.329) →
  real.cbrt 25360000 = 293.8 :=
by
  intros h1 h2
  sorry

end cube_root_25360000_l138_138285


namespace number_of_real_z5_is_10_l138_138058

theorem number_of_real_z5_is_10 :
  ∃ S : Finset ℂ, (∀ z ∈ S, z ^ 30 = 1 ∧ (z ^ 5).im = 0) ∧ S.card = 10 :=
sorry

end number_of_real_z5_is_10_l138_138058


namespace ticket_cost_l138_138186

noncomputable def calculate_cost (x : ℝ) : ℝ :=
  6 * (1.1 * x) + 5 * (x / 2)

theorem ticket_cost (x : ℝ) (h : 4 * (1.1 * x) + 3 * (x / 2) = 28.80) : 
  calculate_cost x = 44.41 := by
  sorry

end ticket_cost_l138_138186


namespace probability_factor_of_36_l138_138682

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138682


namespace solve_for_P_l138_138022

noncomputable def sqrt16_81 : ℝ := real.rpow 81 (1/16)
noncomputable def cube_root_of_59049 := 27 * real.rpow 3 (1 / 3)

theorem solve_for_P (P : ℝ) (h : real.rpow P (3/4) = 81 * sqrt16_81) : 
  P = cube_root_of_59049 :=
begin
  sorry
end

end solve_for_P_l138_138022


namespace problem_l138_138887

theorem problem (a b c d : ℝ) (h₁ : a + b = 0) (h₂ : c * d = 1) : 
  (5 * a + 5 * b - 7 * c * d) / (-(c * d) ^ 3) = 7 := 
by
  sorry

end problem_l138_138887


namespace sam_paint_cans_l138_138012

theorem sam_paint_cans : 
  ∀ (cans_per_room : ℝ) (initial_cans remaining_cans : ℕ),
    initial_cans * cans_per_room = 40 ∧
    remaining_cans * cans_per_room = 30 ∧
    initial_cans - remaining_cans = 4 →
    remaining_cans = 12 :=
by sorry

end sam_paint_cans_l138_138012


namespace cosine_angle_between_a_and_b_l138_138832

-- Definitions of vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (5, 12)

-- Function to compute the dot product of two vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Function to compute the magnitude of a vector
def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

-- The statement we want to prove
theorem cosine_angle_between_a_and_b :
  (dot_product a b) / (magnitude a * magnitude b) = 63 / 65 := by
  sorry

end cosine_angle_between_a_and_b_l138_138832


namespace chess_tournament_l138_138310

theorem chess_tournament (n games : ℕ) (h1 : n = 20) (h2 : games = 190) :
  games = (n * (n - 1)) / 2 → n - 1 = 19 :=
by 
  intros h
  rw [h1, h2] at h ⊢
  sorry

end chess_tournament_l138_138310


namespace arithmetic_sequence_sum_l138_138157

-- Condition definitions
def a : Int := 3
def d : Int := 2
def a_n : Int := 25
def n : Int := 12

-- Sum formula for an arithmetic sequence proof
theorem arithmetic_sequence_sum :
    let n := 12
    let S_n := (n * (a + a_n)) / 2
    S_n = 168 := by
  sorry

end arithmetic_sequence_sum_l138_138157


namespace pow_expression_eq_l138_138790

theorem pow_expression_eq : (-3)^(4^2) + 2^(3^2) = 43047233 := by
  sorry

end pow_expression_eq_l138_138790


namespace inverse_PropositionA_valid_l138_138081

/-- A quadrilateral with diagonals bisecting each other is a parallelogram. -/
def PropositionA (Q : Type) [Quadrilateral Q] : Prop :=
  Q.bisecting_diagonals → Q.is_parallelogram

/-- The diagonals of a square are equal in length. -/
def PropositionB (Q : Type) [Square Q] : Prop :=
  Q.equal_diagonals

/-- Vertically opposite angles are equal. -/
def PropositionC (A : Type) [Angles A] : Prop :=
  A.vertically_opposite → A.equal

/-- If a = b, then sqrt(a^2) = sqrt(b^2). -/
def PropositionD (a b : ℝ) : Prop :=
  a = b → Real.sqrt (a^2) = Real.sqrt (b^2)

/-- Inverse of Proposition A is valid. -/
theorem inverse_PropositionA_valid (Q : Type) [Quadrilateral Q] :
  PropositionA Q :=
sorry

end inverse_PropositionA_valid_l138_138081


namespace number_of_people_favor_chips_l138_138899

theorem number_of_people_favor_chips (
    total_people : ℕ,
    central_angle_chips : ℕ
) : total_people = 600 → central_angle_chips = 216 → (total_people * central_angle_chips) / 360 = 360 :=
begin
    intros h1 h2,
    rw [h1, h2],
    norm_num,
end

end number_of_people_favor_chips_l138_138899


namespace find_n_l138_138418

noncomputable def arithmeticSequenceTerm (a b : ℝ) (n : ℕ) : ℝ :=
  let A := Real.log a
  let B := Real.log b
  6 * B + (n - 1) * 11 * B

theorem find_n 
  (a b : ℝ) 
  (h1 : Real.log (a^2 * b^4) = 2 * Real.log a + 4 * Real.log b)
  (h2 : Real.log (a^6 * b^11) = 6 * Real.log a + 11 * Real.log b)
  (h3 : Real.log (a^12 * b^20) = 12 * Real.log a + 20 * Real.log b) 
  (h_diff : (6 * Real.log a + 11 * Real.log b) - (2 * Real.log a + 4 * Real.log b) = 
            (12 * Real.log a + 20 * Real.log b) - (6 * Real.log a + 11 * Real.log b))
  : ∃ n : ℕ, arithmeticSequenceTerm a b 15 = Real.log (b^n) ∧ n = 160 :=
by
  use 160
  sorry

end find_n_l138_138418


namespace sum_of_dimensions_l138_138134

theorem sum_of_dimensions
  (X Y Z : ℝ)
  (h1 : X * Y = 24)
  (h2 : X * Z = 48)
  (h3 : Y * Z = 72) :
  X + Y + Z = 22 := 
sorry

end sum_of_dimensions_l138_138134


namespace parabola_directrix_standard_eq_l138_138429

theorem parabola_directrix_standard_eq (y : ℝ) (x : ℝ) : 
  (∃ (p : ℝ), p > 0 ∧ ∀ (P : {P // P ≠ x ∨ P ≠ y}), 
  (y + 1) = p) → x^2 = 4 * y :=
sorry

end parabola_directrix_standard_eq_l138_138429


namespace moles_of_CO2_formed_l138_138217

-- Define the reaction
def reaction (HCl NaHCO3 CO2 : ℕ) : Prop :=
  HCl = NaHCO3 ∧ HCl + NaHCO3 = CO2

-- Given conditions
def given_conditions : Prop :=
  ∃ (HCl NaHCO3 CO2 : ℕ),
    reaction HCl NaHCO3 CO2 ∧ HCl = 3 ∧ NaHCO3 = 3

-- Prove the number of moles of CO2 formed is 3.
theorem moles_of_CO2_formed : given_conditions → ∃ CO2 : ℕ, CO2 = 3 :=
  by
    intros h
    sorry

end moles_of_CO2_formed_l138_138217


namespace possible_values_of_sum_l138_138727

def b_seq (i : ℕ) : ℤ :=
  if i % 3 = 0 then 1 else 0

def a_seq (n : ℕ) : ℤ → Prop :=
  ∃ (a : ℕ → ℤ), 
    (∀ n, b_seq n = (a (n - 1) + a n + a (n + 1)) % 2) ∧ 
    (a 0 = a 60) ∧ 
    (a (-1) = a 59)

theorem possible_values_of_sum : ∀ (a : ℕ → ℤ), 
  (∀ n, b_seq n = (a (n - 1) + a n + a (n + 1)) % 2) → 
  a 0 = a 60 → a (-1) = a 59 →
  ∃ v, v ∈ {0, 3, 5, 6} ∧ v = 4 * a 0 + 2 * a 1 + a 2 := 
by {
  sorry
}

end possible_values_of_sum_l138_138727


namespace reflect_point_across_x_axis_l138_138034

theorem reflect_point_across_x_axis (x y : ℝ) (hx : x = 1) (hy : y = 2) : (x, -y) = (1, -2) :=
by
  rw [hx, hy]
  exact rfl

end reflect_point_across_x_axis_l138_138034


namespace math_problem_l138_138000

theorem math_problem (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end math_problem_l138_138000


namespace taller_tree_height_is_108_l138_138453

variables (H : ℝ)

-- Conditions
def taller_tree_height := H
def shorter_tree_height := H - 18
def ratio_condition := (H - 18) / H = 5 / 6

-- Theorem to prove
theorem taller_tree_height_is_108 (hH : 0 < H) (h_ratio : ratio_condition H) : taller_tree_height H = 108 :=
sorry

end taller_tree_height_is_108_l138_138453


namespace probability_area_not_related_to_shape_l138_138050

-- Define the condition related to geometric models
def geometric_model_def (P : Type) [ProbabilitySpace P] (A : set P) : Prop :=
  ∃ area : ℝ, area_probability A = area

-- Statement of the problem
theorem probability_area_not_related_to_shape {P : Type} [ProbabilitySpace P] (A : set P) :
  (geometric_model_def P A) → ¬shape_related_to_probability (ProbabilitySpace P) A :=
sorry

end probability_area_not_related_to_shape_l138_138050


namespace linear_equation_value_m_l138_138294

theorem linear_equation_value_m (m : ℝ) (h : ∀ x : ℝ, 2 * x^(m - 1) + 3 = 0 → x ≠ 0) : m = 2 :=
sorry

end linear_equation_value_m_l138_138294


namespace local_minimum_at_neg_one_l138_138972

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x + 1

noncomputable def derivative_f (x : ℝ) : ℝ := (x + 1) * Real.exp x

theorem local_minimum_at_neg_one :
  IsLocalMin f (-1) :=
begin
  -- Add your proof here
  sorry
end

end local_minimum_at_neg_one_l138_138972


namespace percentage_gain_eq_ten_percent_l138_138107

-- Definitions for conditions
def cost_price : ℝ := 810 / 0.9
def desired_selling_price : ℝ := 990
def loss_percent : ℝ := 10
def selling_price : ℝ := 810

-- Theorem to be proved
theorem percentage_gain_eq_ten_percent : (desired_selling_price - cost_price) / cost_price * 100 = 10 := by
  sorry

end percentage_gain_eq_ten_percent_l138_138107


namespace probability_factor_of_36_l138_138688

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138688


namespace max_value_of_linear_combination_of_m_n_k_l138_138248

-- The style grants us maximum flexibility for definitions.
theorem max_value_of_linear_combination_of_m_n_k 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (m n k : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ m → a i % 3 = 1)
  (h2 : ∀ i, 1 ≤ i ∧ i ≤ n → b i % 3 = 2)
  (h3 : ∀ i, 1 ≤ i ∧ i ≤ k → c i % 3 = 0)
  (h4 : Function.Injective a)
  (h5 : Function.Injective b)
  (h6 : Function.Injective c)
  (h_distinct : ∀ i j, i ≠ j → a i ≠ b j ∧ a i ≠ c j ∧ b i ≠ c j)
  (h_sum : (Finset.range m).sum a + (Finset.range n).sum b + (Finset.range k).sum c = 2007)
  : 4 * m + 3 * n + 5 * k ≤ 256 := by
  sorry

end max_value_of_linear_combination_of_m_n_k_l138_138248


namespace width_of_first_sheet_paper_l138_138409

theorem width_of_first_sheet_paper :
  ∀ (w : ℝ),
  2 * 11 * w = 2 * 4.5 * 11 + 100 → 
  w = 199 / 22 := 
by
  intro w
  intro h
  sorry

end width_of_first_sheet_paper_l138_138409


namespace probability_factor_of_36_l138_138534

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138534


namespace probability_factor_of_36_is_1_over_4_l138_138507

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138507


namespace third_median_length_l138_138137

theorem third_median_length 
  (m_A m_B : ℝ) -- lengths of the first two medians
  (area : ℝ)   -- area of the triangle
  (h_median_A : m_A = 5) -- the first median is 5 inches
  (h_median_B : m_B = 8) -- the second median is 8 inches
  (h_area : area = 6 * Real.sqrt 15) -- the area of the triangle is 6√15 square inches
  : ∃ m_C : ℝ, m_C = Real.sqrt 31 := -- the length of the third median is √31
sorry

end third_median_length_l138_138137


namespace original_number_is_76_l138_138101

-- Define the original number x and the condition given
def original_number_condition (x : ℝ) : Prop :=
  (3 / 4) * x = x - 19

-- State the theorem that the original number x must be 76 if it satisfies the condition
theorem original_number_is_76 (x : ℝ) (h : original_number_condition x) : x = 76 :=
sorry

end original_number_is_76_l138_138101


namespace max_subset_size_l138_138002

def epsilon (p : ℕ) : ℕ :=
if p % 2 = 0 then 1 else 0

theorem max_subset_size (p : ℕ) (n := 2^p) (A : set (Fin n)) 
  (h : ∀ x ∈ A, (2 * x) % n ∈ A):
  ∃ A_max, A_max = A ∧ ∀ B : set (Fin n), (∀ x ∈ B, (2 * x) % n ∈ B) → B.finite → B.card ≤ (n / 3) + epsilon p :=
sorry

end max_subset_size_l138_138002


namespace probability_divisor_of_36_is_one_fourth_l138_138469

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138469


namespace factor_probability_36_l138_138611

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138611


namespace minimal_subset_exists_l138_138065

-- Define the set of two-digit numbers
def two_digit_numbers : set (ℕ × ℕ) :=
  {(a, b) | a >= 0 ∧ a < 10 ∧ b >= 0 ∧ b < 10}

-- Define the condition of infinite sequence
def infinite_sequence (seq : ℕ → ℕ) : Prop :=
  ∀ n, seq n >= 0 ∧ seq n < 10

-- Define the condition on the subset X
def minimal_hitting_set (X : set (ℕ × ℕ)) : Prop :=
  X ⊆ two_digit_numbers ∧ (∀ seq : ℕ → ℕ, infinite_sequence seq → ∃ n, (seq n, seq (n+1)) ∈ X)

-- The main statement to prove
theorem minimal_subset_exists :
  ∃ X : set (ℕ × ℕ), minimal_hitting_set X ∧ X.size ≤ 10 :=
sorry

end minimal_subset_exists_l138_138065


namespace part_I_solution_set_part_II_solution_range_l138_138263

-- Part I: Defining the function and proving the solution set for m = 3
def f (x m : ℝ) : ℝ := |x + 1| + |m - x|

theorem part_I_solution_set (x : ℝ) :
  (f x 3 ≥ 6) ↔ (x ≤ -2 ∨ x ≥ 4) :=
sorry

-- Part II: Proving the range of values for m such that f(x) ≥ 8 for any real number x
theorem part_II_solution_range (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 8) ↔ (m ≤ -9 ∨ m ≥ 7) :=
sorry

end part_I_solution_set_part_II_solution_range_l138_138263


namespace log_inequality_solution_l138_138079

noncomputable def solution_set (a x : ℝ) : Prop :=
x > 0 ∧ x < 2

theorem log_inequality_solution (a : ℝ) (x : ℝ) (h : a > 1) :
  solution_set a x ↔ log a (4 - x) > - log (1/a) x :=
sorry

end log_inequality_solution_l138_138079


namespace chords_parallel_l138_138729

theorem chords_parallel {A B C D M N : Type} [incircle : Circle A B C D] 
  (h1 : Circle.AB_meets_CD)
  (h2 : M_on_AB : \$M \in AB\$) 
  (h3 : N_on_CD : \$N \in CD\$)
  (h4 : eq1 : AM = AC)
  (h5 : eq2 : DN = DB)
  (h6 : M_not_eq_N : M ≠ N) :
  MN \parallel AD :=
by
  sorry

end chords_parallel_l138_138729


namespace pears_weight_difference_l138_138094

theorem pears_weight_difference (n : ℕ) (pears : fin (2 * n) → ℕ) (X Y : fin (2 * n)) 
(h_X : ∀ i, pears X ≤ pears i) (h_Y : ∀ i, pears i ≤ pears Y) 
(h_sorted_cw : ∀ i j, cw X i Y j → |pears i - pears j| ≤ 1)
(h_sorted_ccw : ∀ i j, ccw X i Y j → |pears i - pears j| ≤ 1) :
  ∀ i j, i ≤ j → j < 2 * n → |pears i - pears j| ≤ 1 :=
sorry

end pears_weight_difference_l138_138094


namespace best_fitting_regression_line_l138_138146

theorem best_fitting_regression_line
  (R2_A : ℝ) (R2_B : ℝ) (R2_C : ℝ) (R2_D : ℝ)
  (h_A : R2_A = 0.27)
  (h_B : R2_B = 0.85)
  (h_C : R2_C = 0.96)
  (h_D : R2_D = 0.5) :
  R2_C = 0.96 :=
by
  -- Proof goes here
  sorry

end best_fitting_regression_line_l138_138146


namespace probability_two_students_same_school_l138_138444

theorem probability_two_students_same_school :
  let students := 3
  let schools := 4
  let n := schools ^ students -- total number of basic events
  let m := 36 -- number of events where exactly two students choose the same school
  p = (m:ℚ) / (n:ℚ) 
  ∧ p = 9 / 16 :=
by
  let students := 3
  let schools := 4
  let n := schools ^ students
  let m := 36
  let p := (m:ℚ) / (n:ℚ)
  have : p = 9 / 16, sorry
  exact ⟨rfl, this⟩

end probability_two_students_same_school_l138_138444


namespace smallest_period_of_f_intervals_of_monotonic_decrease_max_min_value_of_f_on_interval_l138_138363

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.cos x) ^ 2 + (Real.sqrt 3) * (Real.sin (2 * x))

theorem smallest_period_of_f : 
  ∃ T > 0, ∀ x ∈ ℝ, f (x + T) = f x ∧ T = real.pi :=
sorry

theorem intervals_of_monotonic_decrease (k : ℤ) :
  ∀ x, f' x < 0 ↔ (Real.pi / 6 + k * Real.pi ≤ x ∧ x ≤ 2 * Real.pi / 3 + k * Real.pi) :=
sorry

theorem max_min_value_of_f_on_interval : 
  ∃ xₘ xₙ, (xₘ = Real.pi / 6 ∧ xₙ = - Real.pi / 6) ∧ (f xₘ = 3 ∧ f xₙ = 0) :=
sorry

end smallest_period_of_f_intervals_of_monotonic_decrease_max_min_value_of_f_on_interval_l138_138363


namespace isosceles_triangle_leg_length_l138_138151

-- Define the necessary condition for the isosceles triangle
def isosceles_triangle (a b c : ℕ) : Prop :=
  b = c ∧ a + b + c = 16 ∧ a = 4

-- State the theorem we want to prove
theorem isosceles_triangle_leg_length :
  ∃ (b c : ℕ), isosceles_triangle 4 b c ∧ b = 6 :=
by
  -- Formal proof will be provided here
  sorry

end isosceles_triangle_leg_length_l138_138151


namespace probability_divisor_of_36_is_one_fourth_l138_138461

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138461


namespace probability_factor_of_36_l138_138575

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138575


namespace bus_driver_earnings_l138_138108

variables (rate : ℝ) (regular_hours overtime_hours : ℕ) (regular_rate overtime_rate : ℝ)

def calculate_regular_earnings (regular_rate : ℝ) (regular_hours : ℕ) : ℝ :=
  regular_rate * regular_hours

def calculate_overtime_earnings (overtime_rate : ℝ) (overtime_hours : ℕ) : ℝ :=
  overtime_rate * overtime_hours

def total_compensation (regular_rate overtime_rate : ℝ) (regular_hours overtime_hours : ℕ) : ℝ :=
  calculate_regular_earnings regular_rate regular_hours + calculate_overtime_earnings overtime_rate overtime_hours

theorem bus_driver_earnings :
  let regular_rate := 16
  let overtime_rate := regular_rate * 1.75
  let regular_hours := 40
  let total_hours := 44
  let overtime_hours := total_hours - regular_hours
  total_compensation regular_rate overtime_rate regular_hours overtime_hours = 752 :=
by
  sorry

end bus_driver_earnings_l138_138108


namespace carrie_remaining_money_l138_138162

theorem carrie_remaining_money
  (work_hours_per_week : ℕ)
  (hourly_rate : ℕ)
  (weeks_per_month : ℕ)
  (bike_cost : ℕ)
  (sales_tax_rate : ℚ)
  (helmet_cost : ℕ)
  (accessories_cost : ℕ)
  (remaining_money : ℕ)
  (H1 : work_hours_per_week = 35)
  (H2 : hourly_rate = 8)
  (H3 : weeks_per_month = 4)
  (H4 : bike_cost = 400)
  (H5 : sales_tax_rate = 0.06)
  (H6 : helmet_cost = 50)
  (H7 : accessories_cost = 30)
  (H8 : remaining_money = 616) :
  let monthly_earnings : ℕ := work_hours_per_week * hourly_rate * weeks_per_month in
  let sales_tax : ℚ := bike_cost * sales_tax_rate in
  let total_bike_cost : ℚ := bike_cost + sales_tax in
  let helmet_and_accessories_cost : ℕ := helmet_cost + accessories_cost in
  let total_cost : ℚ := total_bike_cost + helmet_and_accessories_cost in
  let actual_remaining_money : ℚ := monthly_earnings - total_cost in
  actual_remaining_money = remaining_money :=
by
  sorry

end carrie_remaining_money_l138_138162


namespace probability_factor_of_36_is_1_over_4_l138_138502

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138502


namespace minimize_sum_abs_diff_l138_138840

variable (P : ℕ → ℝ) (n : ℕ)

-- Define the condition that the points are in non-decreasing order
def are_points_ordered : Prop :=
  ∀ (i : ℕ), (i > 0) ∧ (i < n) → P i ≤ P (i + 1)

-- Define the function that calculates the sum of absolute differences
def sum_abs_diff (P : ℕ → ℝ) (Q : ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, abs (Q - P i)

-- The main statement of the problem
theorem minimize_sum_abs_diff (h_order : are_points_ordered P n) :
  (∃ Q, Q = P ((n + 1) / 2) ∧ sum_abs_diff P Q n = ∑ i in finset.range n, abs (P ((n + 1) / 2) - P i)) ∨
  (∃ Q, (P (n / 2) ≤ Q ∧ Q ≤ P (n / 2 + 1)) ∧ sum_abs_diff P Q n = ∑ i in finset.range n, abs (Q - P i)) := sorry

end minimize_sum_abs_diff_l138_138840


namespace conference_center_people_l138_138112

def capacities : List ℕ := [100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320]
def occupancy_rates : List ℚ := [3/4, 5/6, 2/3, 3/5, 4/9, 11/15, 7/10, 1/2, 5/8, 9/14, 8/15, 17/20]

noncomputable def people_in_rooms : List ℚ := (List.zipWith (*) capacities occupancy_rates)
noncomputable def total_people : ℚ := (people_in_rooms.foldl (.+.) 0).floor

theorem conference_center_people : total_people = 1639 := 
by
  sorry

end conference_center_people_l138_138112


namespace reciprocal_of_neg_5_l138_138424

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end reciprocal_of_neg_5_l138_138424


namespace magnitude_of_vec_sum_l138_138277

noncomputable def vec_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))
noncomputable def vec_sum : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vec_sum : magnitude vec_sum = Real.sqrt 7 := 
by 
  sorry

end magnitude_of_vec_sum_l138_138277


namespace point_transform_l138_138376

theorem point_transform : 
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  P' = (-3, 0) :=
by
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  show P' = (-3, 0)
  sorry

end point_transform_l138_138376


namespace largest_N_satisfying_cond_l138_138994

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138994


namespace quarter_wedge_volume_l138_138746

noncomputable def quarter_wedge_volume_approx (r h : ℝ) : ℝ :=
  let V := π * r^2 * h
  let V_wedge := 1 / 4 * V
  V_wedge

theorem quarter_wedge_volume (r h : ℝ) (hr : r = 5) (hh : h = 10) : 
  quarter_wedge_volume_approx r h ≈ 196 :=
by
  unfold quarter_wedge_volume_approx
  rw [hr, hh]
  norm_num
  rw [Real.pi_approx_eq]
  norm_num
  -- As the proof needs approximation of irrational π, we use a 3.14 approximation.
  apply Real.lt_of_abs_lt
  norm_num
  sorry

end quarter_wedge_volume_l138_138746


namespace total_time_for_seven_flights_l138_138288

theorem total_time_for_seven_flights :
  let a := 15
  let d := 8
  let n := 7
  let l := a + (n - 1) * d
  let S_n := n * (a + l) / 2
  S_n = 273 :=
by
  sorry

end total_time_for_seven_flights_l138_138288


namespace probability_factor_of_36_l138_138696

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138696


namespace Jennifer_more_boxes_l138_138344

-- Definitions based on conditions
def Kim_boxes : ℕ := 54
def Jennifer_boxes : ℕ := 71

-- Proof statement (no actual proof needed, just the statement)
theorem Jennifer_more_boxes : Jennifer_boxes - Kim_boxes = 17 := by
  sorry

end Jennifer_more_boxes_l138_138344


namespace molecular_weight_of_ammonium_bromide_l138_138789

-- Define the atomic weights for the elements.
def atomic_weight_N : ℝ := 14.01
def atomic_weight_H : ℝ := 1.01
def atomic_weight_Br : ℝ := 79.90

-- Define the molecular weight of ammonium bromide based on its formula NH4Br.
def molecular_weight_NH4Br : ℝ := (1 * atomic_weight_N) + (4 * atomic_weight_H) + (1 * atomic_weight_Br)

-- State the theorem that the molecular weight of ammonium bromide is approximately 97.95 g/mol.
theorem molecular_weight_of_ammonium_bromide :
  molecular_weight_NH4Br = 97.95 :=
by
  -- The following line is a placeholder to mark the theorem as correct without proving it.
  sorry

end molecular_weight_of_ammonium_bromide_l138_138789


namespace number_multiple_and_divisor_of_15_l138_138703

theorem number_multiple_and_divisor_of_15 : ∃ n : ℕ, (15 % n = 0) ∧ (n % 15 = 0) ∧ n = 15 := by
  use 15
  have h1 : 15 % 15 = 0 := by norm_num
  have h2 : 15 % 15 = 0 := by norm_num
  exact ⟨h1, h2, rfl⟩
  sorry

end number_multiple_and_divisor_of_15_l138_138703


namespace probability_factor_of_36_l138_138583

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138583


namespace resultingPoint_is_correct_l138_138379

def Point (α : Type _) := (x : α) × (y : α)

def initialPoint : Point Int := (-2, -3)

def moveLeft (p : Point Int) (units : Int) : Point Int :=
  (p.1 - units, p.2)

def moveUp (p : Point Int) (units : Int) : Point Int :=
  (p.1, p.2 + units)

theorem resultingPoint_is_correct : 
  (moveUp (moveLeft initialPoint 1) 3) = (-3, 0) :=
by
  sorry

end resultingPoint_is_correct_l138_138379


namespace find_b_l138_138951

open Real

noncomputable def vector3 := ℝ × ℝ × ℝ

def a : vector3 := (8, -5, -3)
def c : vector3 := (-3, -2, 3)

def collinear (a b c : vector3) : Prop :=
  ∃ t : ℝ, b = (a.1 + t * (c.1 - a.1), a.2 + t * (c.2 - a.2), a.3 + t * (c.3 - a.3))

def dot_product (v1 v2 : vector3) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

def norm (v : vector3) : ℝ :=
  sqrt (v.1^2 + v.2^2 + v.3^2)

def angle_double_condition (a b c : vector3) : Prop :=
  (dot_product a b)^2 = 4 * (dot_product b c)^2

theorem find_b : 
  ∃ b : vector3, collinear a b c ∧ angle_double_condition a b c ∧ b = (-2, 1, 1) :=
  sorry

end find_b_l138_138951


namespace incenter_inside_triangle_XYZ_l138_138434

variable {X Y Z B C A : Type} [OrderedOne X] [OrderedOne Y] [OrderedOne Z]
variable {BC CA AB : Set (OrderedOne X × OrderedOne Y)}
variable {ABC : Type} [Triangle ABC]
variable {I : Incenter ABC}
variable {XYZ : Type} [EquilateralTriangle XYZ]

-- Assuming we have the definitions and properties of incircle, incenter, and equilateral triangles:
axiom incircle_tangent {ω : Incircle ABC} {D E F : Type}
  : TangentPoint D BC ∧ TangentPoint E CA ∧ TangentPoint F AB
axiom vertices_XYZ_on_sides_ABC
  : (X ∈ BC) ∧ (Y ∈ CA) ∧ (Z ∈ AB)

theorem incenter_inside_triangle_XYZ
  : In ∈ XYZ := sorry

end incenter_inside_triangle_XYZ_l138_138434


namespace probability_factor_36_l138_138591

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138591


namespace unique_B_squared_l138_138350

theorem unique_B_squared (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) : 
  ∃! B2 : Matrix (Fin 2) (Fin 2) ℝ, B2 = B * B :=
sorry

end unique_B_squared_l138_138350


namespace probability_factor_of_36_l138_138642

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138642


namespace probability_factor_of_36_l138_138678

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138678


namespace sum_of_two_numbers_l138_138452

theorem sum_of_two_numbers (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h1 : x * y = 12) (h2 : 1 / x = 3 * (1 / y)) : x + y = 8 :=
by
  sorry

end sum_of_two_numbers_l138_138452


namespace banana_cost_l138_138785

theorem banana_cost (pounds: ℕ) (rate: ℕ) (per_pounds: ℕ) : 
 (pounds = 18) → (rate = 3) → (per_pounds = 3) → 
  (pounds / per_pounds * rate = 18) := by
  intros
  sorry

end banana_cost_l138_138785


namespace chelsea_sugar_problem_l138_138165

variable (initial_sugar : ℕ)
variable (num_bags : ℕ)
variable (sugar_lost_fraction : ℕ)

def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (sugar_lost_fraction : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let sugar_lost := sugar_per_bag / sugar_lost_fraction
  let remaining_bags_sugar := (num_bags - 1) * sugar_per_bag
  remaining_bags_sugar + (sugar_per_bag - sugar_lost)

theorem chelsea_sugar_problem : 
  remaining_sugar 24 4 2 = 21 :=
by
  sorry

end chelsea_sugar_problem_l138_138165


namespace conditional_probability_l138_138704

-- Definitions of the events and probabilities given in the conditions
def event_A (red : ℕ) : Prop := red % 3 = 0
def event_B (red blue : ℕ) : Prop := red + blue > 8

-- The actual values of probabilities calculated in the solution
def P_A : ℚ := 1/3
def P_B : ℚ := 1/3
def P_AB : ℚ := 5/36

-- Definition of conditional probability
def P_B_given_A : ℚ := P_AB / P_A

-- The claim we want to prove
theorem conditional_probability :
  P_B_given_A = 5 / 12 :=
sorry

end conditional_probability_l138_138704


namespace area_of_triangle_CPC_l138_138451

-- Points representation
structure Point where
  x : ℝ
  y : ℝ

-- Conditions
def C : Point := ⟨5, 7⟩
def y_intercepts_sum (b1 b2 : ℝ) : Prop := b1 + b2 = 4

-- Definition of area of triangle given the points
def triangle_area (A B C : Point) : ℝ :=
  1 / 2 * abs (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y))

-- Problem Statement
theorem area_of_triangle_CPC'_is_60 (b1 b2 : ℝ) (h_sum : y_intercepts_sum b1 b2) : 
    triangle_area ⟨0, b1⟩ ⟨0, b2⟩ C = 60 :=
  sorry

end area_of_triangle_CPC_l138_138451


namespace total_cookies_baked_l138_138740

theorem total_cookies_baked (num_members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ)
  (h1 : num_members = 100) (h2 : sheets_per_member = 10) (h3 : cookies_per_sheet = 16) :
  num_members * sheets_per_member * cookies_per_sheet = 16000 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end total_cookies_baked_l138_138740


namespace isosceles_trapezoid_area_is_56_l138_138092

def isosceles_trapezoid_area (a b c d : ℝ) :=
  (1 / 2) * (a + b) * c

theorem isosceles_trapezoid_area_is_56 :
  ∀ (a b : ℝ), a = 11 → b = 17 → c = 5 →
  ∃ (h : ℝ),
    (h^2 + (b - a)^2 / 4 = c^2) ∧
    isosceles_trapezoid_area a b h = 56 := by
  sorry

end isosceles_trapezoid_area_is_56_l138_138092


namespace probability_factor_of_36_l138_138681

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138681


namespace sara_red_balloons_l138_138391

variable (total_red_balloons : ℕ) (sandy_red_balloons : ℕ)

theorem sara_red_balloons : total_red_balloons = 55 ∧ sandy_red_balloons = 24 → ∃ sara_red_balloons : ℕ, sara_red_balloons = total_red_balloons - sandy_red_balloons ∧ sara_red_balloons = 31 :=
by
  intro h
  cases h with h_total h_sandy
  use 31
  split
  · rwa [h_total, h_sandy]
  · exact rfl

-- Sara has 31 red balloons, as expected.

end sara_red_balloons_l138_138391


namespace repeating_decimals_product_l138_138815

theorem repeating_decimals_product : (0.06 : ℝ) * (0.3 : ℝ) = (2 / 99 : ℝ) :=
by {
  have h1 : (0.06 : ℝ) = (2 / 33 : ℝ),
  { sorry },
  have h2 : (0.3 : ℝ) = (1 / 3 : ℝ),
  { sorry },
  rw [h1, h2],
  norm_num,
}

end repeating_decimals_product_l138_138815


namespace impossible_to_transport_stones_l138_138337

-- Define the conditions of the problem
def stones : List ℕ := List.range' 370 (468 - 370 + 2 + 1) 2
def truck_capacity : ℕ := 3000
def number_of_trucks : ℕ := 7
def number_of_stones : ℕ := 50

-- Prove that it is impossible to transport the stones using the given trucks
theorem impossible_to_transport_stones :
  stones.length = number_of_stones →
  (∀ weights ∈ stones.sublists, (weights.sum ≤ truck_capacity → List.length weights ≤ number_of_trucks)) → 
  false :=
by
  sorry

end impossible_to_transport_stones_l138_138337


namespace all_empty_squares_red_all_empty_squares_blue_l138_138372

section Chessboard

def board_size := 6

-- Define what it means for rooks to not threaten each other
def non_threatening_rooks (rooks : List (ℕ × ℕ)) : Prop :=
  ∀ i j, i ≠ j → (rooks[i].fst ≠ rooks[j].fst ∧ rooks[i].snd ≠ rooks[j].snd)

-- Define the color rule for a given square
inductive SquareColor
| red
| blue

def square_color (rooks : List (ℕ × ℕ)) (x y : ℕ) : SquareColor :=
  let distances := rooks.map (λ p => (abs (p.fst - x), abs (p.snd - y)))
  if distances.all_same then SquareColor.red else SquareColor.blue

-- Statement of the problems

-- Problem (a)
theorem all_empty_squares_red (rooks : List (ℕ × ℕ)) (h_non_threat : non_threatening_rooks rooks) :
  (∀ x y, (x, y) ∉ rooks → square_color rooks x y = SquareColor.red) :=
  sorry

-- Problem (b)
theorem all_empty_squares_blue (rooks : List (ℕ × ℕ)) (h_non_threat : non_threatening_rooks rooks) :
  (∀ x y, (x, y) ∉ rooks → square_color rooks x y = SquareColor.blue) :=
  sorry

end Chessboard

end all_empty_squares_red_all_empty_squares_blue_l138_138372


namespace probability_factor_of_36_l138_138574

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138574


namespace num_pairs_x_y_l138_138884

theorem num_pairs_x_y :
  {p : ℕ × ℕ // p.1^2 - p.2^2 = 45 ∧ p.1 > 0 ∧ p.2 > 0}.card = 3 :=
sorry

end num_pairs_x_y_l138_138884


namespace sec_product_degrees_l138_138082

theorem sec_product_degrees 
  (p q : ℕ) 
  (hp : p > 1) 
  (hq : q > 1) 
  (h : ∏ k in (Finset.range 36).image (λ k, 5*(k+1)), (sec (k : ℝ) * (sec (k : ℝ))) = p^q) :
  p + q = 38 := 
sorry

end sec_product_degrees_l138_138082


namespace probability_of_odd_m_n_l138_138976

def count_odds (l : List ℕ) : ℕ :=
  List.length (List.filter (λ x => x % 2 = 1) l)

def possible_combinations : ℕ := 7 * 9

def favorable_combinations : ℕ :=
  (count_odds [1, 2, 3, 4, 5, 6, 7]) * (count_odds [1, 2, 3, 4, 5, 6, 7, 8, 9])

theorem probability_of_odd_m_n : (favorable_combinations : ℚ) / (possible_combinations : ℚ) = 20/63 :=
by
  -- Proof to be filled
  sorry

end probability_of_odd_m_n_l138_138976


namespace find_polynomials_l138_138818

-- Define the conditions
def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def poly_integer_coeff (p : ℕ → ℤ) : Prop :=
  ∀ x, ∃ a b : ℤ, p x = a * x + b

-- Define main theorem
theorem find_polynomials (p : ℕ → ℤ) (h_poly : poly_integer_coeff p) :
  (∀ a b : ℕ, is_perfect_square (a + b) → is_perfect_square (p a + p b)) →
  (∃ k : ℤ, ∀ x, p x = k * k * x) ∨ (∃ u : ℤ, ∀ x, p x = 2 * u * u) :=
by
  sorry

end find_polynomials_l138_138818


namespace amount_paid_is_correct_l138_138933

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l138_138933


namespace probability_factor_36_l138_138552

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138552


namespace largest_possible_N_l138_138991

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138991


namespace probability_factor_of_36_l138_138564

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138564


namespace factor_probability_l138_138625

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138625


namespace rowing_distance_l138_138754

def man_rowing_speed_still_water : ℝ := 10
def stream_speed : ℝ := 8
def rowing_time_downstream : ℝ := 5
def effective_speed_downstream : ℝ := man_rowing_speed_still_water + stream_speed

theorem rowing_distance :
  effective_speed_downstream * rowing_time_downstream = 90 := 
by 
  sorry

end rowing_distance_l138_138754


namespace typing_orders_count_l138_138319

theorem typing_orders_count :
  (∑ k in Finset.range 11, binomial 10 k * (k + 1)) = 6144 :=
by sorry

end typing_orders_count_l138_138319


namespace total_dogs_l138_138443

theorem total_dogs (D : ℕ) 
(h1 : 12 = 12)
(h2 : D / 2 = D / 2)
(h3 : D / 4 = D / 4)
(h4 : 10 = 10)
(h_eq : 12 + D / 2 + D / 4 + 10 = D) : 
D = 88 := by
sorry

end total_dogs_l138_138443


namespace minimize_expression_l138_138457

theorem minimize_expression (x : ℝ) : 
  ∃ (m : ℝ), m = 2023 ∧ ∀ x, (x + 2) * (x + 3) * (x + 4) * (x + 5) + 2024 ≥ m :=
sorry

end minimize_expression_l138_138457


namespace cistern_fill_time_with_leak_l138_138745

theorem cistern_fill_time_with_leak :
  ∀ (T_without_leak T_leak T_with_leak : ℝ) (R : ℝ),
  (T_without_leak = 10) →
  (T_leak = 60) →
  (R = 1 / T_without_leak) →
  (T_with_leak = 1 / (R - 1 / T_leak)) →
  T_with_leak - T_without_leak = 2 :=
by
  intros T_without_leak T_leak T_with_leak R
  assume h1 h2 h3 h4
  sorry

end cistern_fill_time_with_leak_l138_138745


namespace cosine_square_plus_alpha_sine_l138_138382

variable (α : ℝ)

theorem cosine_square_plus_alpha_sine (h1 : 0 ≤ α) (h2 : α ≤ Real.pi / 2) : 
  Real.cos α * Real.cos α + α * Real.sin α ≥ 1 :=
sorry

end cosine_square_plus_alpha_sine_l138_138382


namespace solve_system_l138_138025

variables {R : Type*} [Field R]

theorem solve_system 
  (p q u v : R)
  (h1 : p ≠ 0) 
  (h2 : q ≠ 0) 
  (h3 : p ≠ q) 
  (h4 : p + q ≠ 0)
  (h5 : ∣p∣ ≠ ∣q∣)
  (h6 : ∣u∣ ≠ ∣v∣)
  (eq1 : p * u + q * v = 2 * (p^2 - q^2))
  (eq2 : v / (p - q) - u / (p + q) = (p^2 + q^2) / (p * q)) :
  (u = (p^2 - q^2) / p ∧ v = (p^2 - q^2) / q) ∨
  (p = u * v^2 / (v^2 - u^2) ∧ q = u^2 * v / (v^2 - u^2)) ∨
  ∃ w : R, (w = 1 + sqrt 2 ∨ w = 1 - sqrt 2) ∧
  (p = (1/4) * ((1 - w) * u - v) ∧ q = (1/4) * (-u - (1 + w) * v)) :=
sorry

end solve_system_l138_138025


namespace probability_three_tails_one_head_l138_138290

theorem probability_three_tails_one_head :
  let p := (1 : ℝ) / 2 in
  (∃ t1 t2 t3 t4 : bool, (t1 = tt) ∨ (t2 = tt) ∨ (t3 = tt) ∨ (t4 = tt)) → 
  (∑ e in {t | t = tt ∨ t = ff}, p ^ 4) * 4 = 1 / 4 :=
by sorry

end probability_three_tails_one_head_l138_138290


namespace arithmetic_geometric_sequence_x_l138_138030

theorem arithmetic_geometric_sequence_x :
  ∀ (a : ℕ → ℝ), a 1 = -8 → a 2 = -6 →
  (∃ x : ℝ, (-8 + x), (-2 + x), x form_geometric_sequence →
  x = -1) :=
by
  intro a h1 h2
  have d : ℝ := a 2 - a 1
  have a₄ : ℝ := a 1 + 3 * d
  have a₅ : ℝ := a 1 + 4 * d
  use -1
  sorry

end arithmetic_geometric_sequence_x_l138_138030


namespace probability_factor_of_36_l138_138580

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138580


namespace probability_factor_of_36_is_1_over_4_l138_138500

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138500


namespace yan_distance_to_stadium_l138_138715

-- Definition of the variables w (walking speed), x (distance to home), and y (distance to stadium)
variables (w x y : ℝ)

-- Given conditions stated as assumptions
-- Yan's bicycle speed is 5 times his walking speed.
-- Both options (walking directly or going home first and then riding the bicycle) take the same amount of time.
def distances_ratio (w x y : ℝ) : Prop :=
  (5 * y = 6 * x + y)

noncomputable def yan_distance_ratio : ℝ :=
  if distances_ratio w x y then x / y else 0

-- The theorem to prove
theorem yan_distance_to_stadium (w x y : ℝ) (h: distances_ratio w x y) :
  x / y = 2 / 3 :=
by
  sorry

end yan_distance_to_stadium_l138_138715


namespace solve_system_of_equations_l138_138401

def system_solution : Prop := ∃ x y : ℚ, 4 * x - 6 * y = -14 ∧ 8 * x + 3 * y = -15 ∧ x = -11 / 5 ∧ y = 2.6 / 3

theorem solve_system_of_equations : system_solution := sorry

end solve_system_of_equations_l138_138401


namespace probability_factor_of_36_l138_138587

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138587


namespace correct_linear_regression_l138_138256

theorem correct_linear_regression :
  ∀ (x y : Type) [LinearOrder x] [LinearOrder y] [AddGroup x] [AddGroup y]
    (positively_correlated : x → y → Prop)
    (mean_x mean_y : ℝ)
    (A B C D : ℝ → ℝ)
    (A_eqn : ∀ x, A x = 0.4 * x + 2.3)
    (B_eqn : ∀ x, B x = 2 * x - 2.4)
    (C_eqn : ∀ x, C x = -2 * x + 9.5)
    (D_eqn : ∀ x, D x = -0.3 * x + 4.4)
    (mean_x_val : mean_x = 3)
    (mean_y_val : mean_y = 3.5),
    positively_correlated x y →
    (A mean_x = mean_y) →
    ¬ (B mean_x = mean_y) →
    ¬ (C mean_x = mean_y) →
    ¬ (D mean_x = mean_y) →
    A mean_x = mean_y :=
by
  intros x y h1 h2 h3 positively_correlated mean_x mean_y A B C D A_eqn B_eqn C_eqn D_eqn mean_x_val mean_y_val h_pos_corr h_a h_not_b h_not_c h_not_d
  rw [← mean_x_val, ← mean_y_val] at h_not_b h_not_c h_not_d
  exact h_a

end correct_linear_regression_l138_138256


namespace factor_probability_36_l138_138607

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138607


namespace license_plate_palindrome_probability_l138_138366

theorem license_plate_palindrome_probability : 
  let p := 775 
  let q := 67600  
  p + q = 776 :=
by
  let p := 775
  let q := 67600
  show p + q = 776
  sorry

end license_plate_palindrome_probability_l138_138366


namespace probability_three_tails_one_head_l138_138289

theorem probability_three_tails_one_head :
  let p := (1 : ℝ) / 2 in
  (∃ t1 t2 t3 t4 : bool, (t1 = tt) ∨ (t2 = tt) ∨ (t3 = tt) ∨ (t4 = tt)) → 
  (∑ e in {t | t = tt ∨ t = ff}, p ^ 4) * 4 = 1 / 4 :=
by sorry

end probability_three_tails_one_head_l138_138289


namespace hexagon_area_proof_l138_138218

noncomputable def square_vertices : list (ℝ × ℝ) := [(0,0), (40,0), (40,40), (0,40)]

noncomputable def hexagon_vertices : list (ℝ × ℝ) := [(0,0), (15,0), (40,35), (40,40), (30,40), (0,25)]

noncomputable def area_of_hexagon (hexagon : list (ℝ × ℝ)) : ℝ := 
  -- The function implementation is omitted
  -- You need to implement the function to calculate the area using the given vertices
  sorry

theorem hexagon_area_proof : area_of_hexagon hexagon_vertices = 1325 := by 
  sorry

end hexagon_area_proof_l138_138218


namespace probability_factor_of_36_l138_138485

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138485


namespace largest_N_satisfying_cond_l138_138993

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138993


namespace total_cookies_baked_l138_138743

theorem total_cookies_baked (members sheets_per_member cookies_per_sheet : ℕ) (h1 : members = 100) (h2 : sheets_per_member = 10) (h3 : cookies_per_sheet = 16) :
  (members * sheets_per_member * cookies_per_sheet) = 16000 :=
by
  rw [h1, h2, h3]
  norm_num
  -- Additional steps or simplification if necessary
  sorry

end total_cookies_baked_l138_138743


namespace ratio_areas_APD_BPC_constant_l138_138947

variable {A B C D E F P Q X Y : Point}

-- Defining the cyclic quadrilateral ABCD
def isCyclicQuadrilateral (A B C D : Point) : Prop := sorry

-- Defining the ratios given in the problem
def ratioAE_EB_CF_FD (A B E : Point) (C D F : Point) : Prop :=
  (AE/EB = CF/FD)

def ratioPE_PF_AB_CD (P E F : Point) (AB CD : ℝ) : Prop :=
  (PE/PF = AB/CD)

-- Proving the ratio of areas
theorem ratio_areas_APD_BPC_constant (h_cyclic : isCyclicQuadrilateral A B C D)
    (h_ratio1 : ratioAE_EB_CF_FD A B E C D F)
    (h_ratio2 : ratioPE_PF_AB_CD P E F (AB.dist A B) (CD.dist C D)) : 
    ∃ k : ℝ, ratioAreas (triangle A P D) (triangle B P C) = k :=
  sorry

end ratio_areas_APD_BPC_constant_l138_138947


namespace probability_divisor_of_36_l138_138657

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138657


namespace probability_factor_of_36_l138_138533

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138533


namespace sector_area_correct_l138_138893

variable (r : ℝ) (l : ℝ)

def sector_area (r : ℝ) (l : ℝ) : ℝ := (1/2) * l * r

theorem sector_area_correct : 
  r = 20 → 
  l = 8 * Real.pi → 
  sector_area r l = 80 * Real.pi := 
by
  intros h_r h_l
  rw [h_r, h_l]
  unfold sector_area
  norm_num
  ring


end sector_area_correct_l138_138893


namespace positive_difference_median_mode_l138_138460

def data : List ℕ := [27, 28, 29, 29, 29, 30, 30, 30, 31, 31, 
                      42, 43, 45, 46, 48, 51, 51, 51, 52, 53, 
                      61, 64, 65, 68, 69]

def mode (lst : List ℕ) : ℕ :=
  let freq_map := lst.foldl (λ (acc : Std.HashMap ℕ ℕ) n, acc.insert n ((acc.find! n).getD 0 + 1)) Std.HashMap.empty
  freq_map.foldl (λ (max_key max_val) (key val), if val > max_val then (key, val) else (max_key, max_val)) (0, 0) |>.fst

def median (lst : List ℕ) : ℕ :=
  let sorted := lst.qsort (· ≤ ·)
  sorted[(sorted.length / 2)]

theorem positive_difference_median_mode :
  |median data - mode data| = 16 := by
  -- Proof goes here.
  sorry

end positive_difference_median_mode_l138_138460


namespace probability_of_three_tails_one_head_in_four_tosses_l138_138292

noncomputable def probability_three_tails_one_head (n : ℕ) : ℚ :=
  if n = 4 then 1 / 4 else 0

theorem probability_of_three_tails_one_head_in_four_tosses :
  probability_three_tails_one_head 4 = 1 / 4 :=
by sorry

end probability_of_three_tails_one_head_in_four_tosses_l138_138292


namespace factor_probability_36_l138_138614

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138614


namespace total_payment_l138_138938

/--
  Jerry took 8 hours painting the house. 
  The time to fix the kitchen counter was three times longer than painting the house.
  Jerry took 6 hours mowing the lawn.
  Jerry charged $15 per hour of work.
  Prove that the total amount of money Miss Stevie paid Jerry is $570.
-/
theorem total_payment (h_paint: ℕ := 8) (h_counter: ℕ := 3 * h_paint) (h_mow: ℕ := 6) (rate: ℕ := 15) :
  let total_hours := h_paint + h_counter + h_mow
  in total_hours * rate = 570 :=
by
  -- provide proof here
  sorry

end total_payment_l138_138938


namespace minimum_k_for_mutual_criticism_l138_138193

theorem minimum_k_for_mutual_criticism (k : ℕ) (h1 : 15 * k > 105) : k ≥ 8 := by
  sorry

end minimum_k_for_mutual_criticism_l138_138193


namespace find_eccentricity_of_ellipse_l138_138853

noncomputable def ellipseEccentricity (k : ℝ) : ℝ :=
  let a := Real.sqrt (k + 2)
  let b := Real.sqrt (k + 1)
  let c := Real.sqrt (a^2 - b^2)
  c / a

theorem find_eccentricity_of_ellipse (k : ℝ) (h1 : k + 2 = 4) (h2 : Real.sqrt (k + 2) = 2) :
  ellipseEccentricity k = 1 / 2 := by
  sorry

end find_eccentricity_of_ellipse_l138_138853


namespace rate_per_sq_meter_l138_138043

theorem rate_per_sq_meter (b l: ℝ) (cost area rate: ℝ) (hl: l = 20) (hb: l = 3 * b) (hc: cost = 400) (ha: area = l * b) (hr: rate = cost / area) :
  rate = 3 := 
begin
  sorry
end

end rate_per_sq_meter_l138_138043


namespace find_f_double_prime_at_1_l138_138834

noncomputable def f (x : ℝ) : ℝ := x^2 + 3 * x * (f' (1))

theorem find_f_double_prime_at_1 : f'' 1 = 0 := 
sorry

end find_f_double_prime_at_1_l138_138834


namespace probability_factor_of_36_l138_138588

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138588


namespace remainder_N_mod_1000_l138_138948

open Nat

def A : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}

noncomputable def f (x : ℕ) : ℕ := sorry

theorem remainder_N_mod_1000 : 
  (∃ N : ℕ, (∀ x ∈ A, f(f(f(x))) = f(f(f(1))) ∧ N = 8 * (7 + 7^2 + 7^3 + 7^4 + 7^5 + 7^6 + 7^7)) 
  ∧ ((8 * 7 * 137257) % 1000 = 992)) :=
by {
  sorry
}

end remainder_N_mod_1000_l138_138948


namespace length_of_crease_l138_138122

theorem length_of_crease {ABC : Type*} (A B C : ABC) (M : ABC) (h_eq_triangle : equilateral_triangle ABC 6) (h_midpoint : midpoint M B C) : 
  length_of_crease A M = (3 * sqrt 3) / 2 :=
sorry

end length_of_crease_l138_138122


namespace factor_probability_l138_138622

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138622


namespace fibonacci_sum_of_squares_l138_138346

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci (n + 1) + fibonacci n

theorem fibonacci_sum_of_squares (n : ℕ) (hn : n ≥ 1) :
  (Finset.range n).sum (λ i => (fibonacci (i + 1))^2) = fibonacci n * fibonacci (n + 1) :=
sorry

end fibonacci_sum_of_squares_l138_138346


namespace instrument_failure_probability_l138_138308

noncomputable def probability_of_instrument_not_working (m : ℕ) (P : ℝ) : ℝ :=
  1 - (1 - P)^m

theorem instrument_failure_probability (m : ℕ) (P : ℝ) :
  0 ≤ P → P ≤ 1 → probability_of_instrument_not_working m P = 1 - (1 - P)^m :=
by
  intros _ _
  sorry

end instrument_failure_probability_l138_138308


namespace median_books_read_l138_138766

noncomputable def median_number_of_books : ℕ :=
let students_books := [2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5] in
students_books.nth (students_books.length / 2)

theorem median_books_read : median_number_of_books = 3 :=
by {
  -- Consider the list students_books which includes:
  -- 5 students with 2 books,
  -- 6 students with 3 books,
  -- 4 students with 4 books,
  -- 6 students with 5 books,
  -- resulting in 21 total entries.
  -- The (21 + 1) / 2 = 11th element in the ordered list must be the median.
  -- In the list, the 11th element corresponds to the number 3.
  sorry
}

end median_books_read_l138_138766


namespace digit_in_421st_place_l138_138710

theorem digit_in_421st_place (r : ℚ) (rep_seq : list ℕ) (h1 : r = 7 / 19)
  (h2 : rep_seq = [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7])
  (h3 : ∀ n, r * 10^n = fin (19 * 10^(n - 18))) :
    rep_seq.nth (421 % 18) = some 1 :=
by
  sorry

end digit_in_421st_place_l138_138710


namespace reservoir_ratio_l138_138784

noncomputable def T : ℕ := 40  -- Total capacity in million gallons
def W : ℕ := 30              -- Water at the end of the month in million gallons
def N : ℕ := T - 20          -- Normal level in million gallons

theorem reservoir_ratio :
  (N = 20) ∧ (W = 0.75 * T) → (W / N = 1.5) :=
by
  sorry

end reservoir_ratio_l138_138784


namespace probability_factor_of_36_l138_138579

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138579


namespace probability_divisor_of_36_l138_138665

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138665


namespace exists_lucky_integer_within_70_l138_138125

def is_lucky_integer (x : ℕ) : Prop :=
  x % 7 = 0 ∧ (nat.digits 10 x).sum % 7 = 0

theorem exists_lucky_integer_within_70 (n : ℕ) (hn : n > 0) : 
  ∃ ℓ : ℕ, is_lucky_integer ℓ ∧ |n - ℓ| ≤ 70 :=
sorry

end exists_lucky_integer_within_70_l138_138125


namespace battery_charge_to_60_percent_l138_138121

noncomputable def battery_charge_time (initial_charge_percent : ℝ) (initial_time_minutes : ℕ) (additional_time_minutes : ℕ) : ℕ :=
  let rate_per_minute := initial_charge_percent / initial_time_minutes
  let additional_charge_percent := additional_time_minutes * rate_per_minute
  let total_percent := initial_charge_percent + additional_charge_percent
  if total_percent = 60 then
    initial_time_minutes + additional_time_minutes
  else
    sorry

theorem battery_charge_to_60_percent : battery_charge_time 20 60 120 = 180 :=
by
  -- The formal proof will be provided here.
  sorry

end battery_charge_to_60_percent_l138_138121


namespace probability_factor_of_36_l138_138562

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138562


namespace total_population_of_cities_l138_138436

theorem total_population_of_cities (n : ℕ) (avg_pop : ℕ) (pn : (n = 20)) (avg_factor: (avg_pop = (4500 + 5000) / 2)) : 
  n * avg_pop = 95000 := 
by 
  sorry

end total_population_of_cities_l138_138436


namespace find_measure_of_angle3_l138_138855

noncomputable def angle_measure_proof (angle1 angle2 angle3 : ℝ) : Prop :=
  angle1 = 67 + 12 / 60 ∧
  (angle1 + angle2 = 90) ∧
  (angle2 + angle3 = 180) →
  angle3 = 157 + 12 / 60

theorem find_measure_of_angle3 :
  angle_measure_proof 67 12 180 :=
by
  intros angle1 angle2 angle3 h
  cases h with h_angle1 h_rest
  cases h_rest with h_comp h_supp
  have h_angle2 := 90 - h_angle1
  have h_angle3 := 180 - h_angle2
  rw [h_angle2, h_angle3]
  exact sorry

end find_measure_of_angle3_l138_138855


namespace probability_factor_of_36_l138_138640

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138640


namespace hyperbola_right_focus_coordinates_l138_138413

theorem hyperbola_right_focus_coordinates (x y : ℝ) : 
  (x^2 - 2 * y^2 = 1) → (x = sqrt (3 / 2) ∧ y = 0) :=
by
  sorry

end hyperbola_right_focus_coordinates_l138_138413


namespace repeating_decimal_transform_l138_138926

theorem repeating_decimal_transform (n : ℕ) (s : String) (k : ℕ) (m : ℕ)
  (original : s = "2345678") (len : k = 7) (position : n = 2011)
  (effective_position : m = n - 1) (mod_position : m % k = 3) :
  "0.1" ++ s = "0.12345678" :=
sorry

end repeating_decimal_transform_l138_138926


namespace area_equivalence_l138_138029

variables {A B C D E I : Type*} 
variables [OrderedCommGroup A] [OrderedCommGroup B] [OrderedCommGroup C]
variables (sideCA sideCB sideAB : ℝ)

def angle_bisectors_intersect (triangleABC : Type*) (AD BE : Type*) : Prop :=
  -- Define the property that AD and BE are angle bisectors of triangleABC and intersect at I
  sorry

def area (x : Type*) (y : ℝ) : ℝ := 
  -- Map an element of type x to an area measurement
  y

theorem area_equivalence 
  (triangleABC : Type*) (AD BE : Type*) 
  (angle_bisectors_intersect : Prop)
  (h_side_ratio : sideCA * sideCB = sideAB^2) :
  area (triangleABC ∩ {I}) = area (CDIE) := 
sorry

end area_equivalence_l138_138029


namespace sum_of_square_of_geometric_sequence_thm_l138_138329

noncomputable def sum_of_square_of_geometric_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range (n+1), (a i)^2

theorem sum_of_square_of_geometric_sequence_thm (a : ℕ → ℝ) (n : ℕ)
    (h1 : ∀ n, (∑ i in finset.range (n+1), a i) = 2^n - 1)
    (h2 : ∃ q : ℝ, q = 2 ∧ (∀ n, a (n+1) = q * a n)) :
  sum_of_square_of_geometric_sequence a n = (4^n - 1) / 3 :=
sorry

end sum_of_square_of_geometric_sequence_thm_l138_138329


namespace isosceles_triangle_base_angle_l138_138318

theorem isosceles_triangle_base_angle (α : ℕ) (base_angle : ℕ) 
  (hα : α = 40) (hsum : α + 2 * base_angle = 180) : 
  base_angle = 70 :=
sorry

end isosceles_triangle_base_angle_l138_138318


namespace integer_pairs_summing_to_six_l138_138817

theorem integer_pairs_summing_to_six :
  ∃ m n : ℤ, m + n + m * n = 6 ∧ ((m = 0 ∧ n = 6) ∨ (m = 6 ∧ n = 0)) :=
by
  sorry

end integer_pairs_summing_to_six_l138_138817


namespace factor_probability_l138_138632

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138632


namespace dog_bones_l138_138717

theorem dog_bones (initial_bones found_bones : ℕ) (h₁ : initial_bones = 15) (h₂ : found_bones = 8) : initial_bones + found_bones = 23 := by
  sorry

end dog_bones_l138_138717


namespace max_value_of_f_l138_138207

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l138_138207


namespace problem1_problem2_l138_138227

variable {a b : ℝ}

theorem problem1
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : (a + b) * (a^5 + b^5) ≥ 4 := sorry

theorem problem2
  (h1 : 0 < a)
  (h2 : 0 < b)
  (h3 : a^3 + b^3 = 2)
  : a + b ≤ 2 := sorry

end problem1_problem2_l138_138227


namespace cricket_initial_overs_l138_138311

/-- Prove that the number of initial overs played was 10. -/
theorem cricket_initial_overs 
  (target : ℝ) 
  (initial_run_rate : ℝ) 
  (remaining_run_rate : ℝ) 
  (remaining_overs : ℕ)
  (h_target : target = 282)
  (h_initial_run_rate : initial_run_rate = 4.6)
  (h_remaining_run_rate : remaining_run_rate = 5.9)
  (h_remaining_overs : remaining_overs = 40) 
  : ∃ x : ℝ, x = 10 := 
by
  sorry

end cricket_initial_overs_l138_138311


namespace probability_factor_of_36_l138_138480

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138480


namespace symmetric_circle_eq_l138_138822

theorem symmetric_circle_eq :
  (∃ x y : ℝ, (x + 2)^2 + (y - 1)^2 = 4) :=
by
  sorry

end symmetric_circle_eq_l138_138822


namespace probability_factor_36_l138_138603

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138603


namespace parabola_translation_l138_138448

theorem parabola_translation (x y : ℝ) :
  let original_parabola := -x * (x + 2),
      parabola_translated_right := -((x - 1)^2) + 1,
      resulting_parabola := parabola_translated_right - 3
  in resulting_parabola = -((x - 1)^2) - 2 :=
by
  sorry

end parabola_translation_l138_138448


namespace probability_factor_36_l138_138524

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138524


namespace reciprocal_of_neg_five_l138_138425

theorem reciprocal_of_neg_five: 
  ∃ x : ℚ, -5 * x = 1 ∧ x = -1 / 5 := 
sorry

end reciprocal_of_neg_five_l138_138425


namespace men_work_problem_l138_138891

theorem men_work_problem (x : ℕ) (h1 : x * 70 = 40 * 63) : x = 36 := 
by
  sorry

end men_work_problem_l138_138891


namespace probability_factor_of_36_l138_138526

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138526


namespace shortest_side_length_l138_138907

noncomputable def length_of_shortest_side (AE EC radius : ℝ) (h1 : AE = 5) (h2 : EC = 9) (h3 : radius = 5) : ℝ :=
  let s := (AE + EC + AB) / 2 in
  let Δ := sqrt(s * (s - AB) * (s - 2 * x) * (s - 14)) in
  if 5 = Δ / (s - 14) then 14 else sorry

theorem shortest_side_length (AE EC : ℝ) (radius : ℝ) (h1 : AE = 5) (h2 : EC = 9) (h3 : radius = 5) :
  length_of_shortest_side AE EC radius h1 h2 h3 = 14 := sorry

end shortest_side_length_l138_138907


namespace probability_factor_of_36_is_1_over_4_l138_138493

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138493


namespace det_E_l138_138950

def D : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![5, 0], ![0, 5]]

def R : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -1], ![1, 0]]

def E := D ⬝ R

theorem det_E : Matrix.det E = 25 :=
by
  sorry

end det_E_l138_138950


namespace hyperbola_lambda_range_l138_138864

-- Define the condition that the equation \dfrac{x^2}{2+\lambda} - \dfrac{y^2}{1+\lambda} = 1 represents a hyperbola.
def isHyperbola (λ : ℝ) : Prop :=
  (2 + λ) * (1 + λ) > 0

-- Prove that if the equation represents a hyperbola, then λ lies in (-∞, -2) ∪ (-1, +∞).
theorem hyperbola_lambda_range (λ : ℝ) (h : isHyperbola λ) : λ ∈ set.Ioo (-∞) (-2) ∪ set.Ioo (-1) (∞) :=
sorry

end hyperbola_lambda_range_l138_138864


namespace probability_factor_36_l138_138553

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138553


namespace area_quadrilateral_ABCD_l138_138808

-- Definitions based on given conditions
def length_diagonal_BD : ℝ := 50
def offset_AE : ℝ := 10
def offset_CF : ℝ := 8
def angle_ABD (x : ℝ) (h : 0 < x ∧ x < 180) : Prop := True  -- Given angle condition

-- Problem statement in Lean 4
theorem area_quadrilateral_ABCD {x : ℝ} (h : 0 < x ∧ x < 180) :
  let BD := length_diagonal_BD,
      AE := offset_AE,
      CF := offset_CF in
  0.5 * BD * AE + 0.5 * BD * CF = 450 :=
by
  let BD := length_diagonal_BD
  let AE := offset_AE
  let CF := offset_CF
  have BD_nonzero : BD ≠ 0 := by norm_num
  sorry

end area_quadrilateral_ABCD_l138_138808


namespace slower_pipe_filling_time_l138_138091

theorem slower_pipe_filling_time (R : ℝ) (t : ℝ) 
  (h1 : ∀ (R : ℝ), Faster := 4 * R)
  (h2 : (R + 4 * R) * 36 = 1) : 
  t = 180 := 
  sorry

end slower_pipe_filling_time_l138_138091


namespace simplify_fraction_l138_138398

theorem simplify_fraction (h1 : irrational(√5)) (h2 : irrational(√7)) : 
  1 / (1 / (√5 + 2) + 3 / (√7 - 2)) = (√7 - √5) / 2 :=
by
  sorry

end simplify_fraction_l138_138398


namespace smallest_five_digit_congruent_to_2_mod_17_l138_138075

-- Definitions provided by conditions
def is_five_digit (x : ℕ) : Prop := 10000 ≤ x ∧ x < 100000
def is_congruent_to_2_mod_17 (x : ℕ) : Prop := x % 17 = 2

-- Proving the existence of the smallest five-digit integer satisfying the conditions
theorem smallest_five_digit_congruent_to_2_mod_17 : 
  ∃ x : ℕ, is_five_digit x ∧ is_congruent_to_2_mod_17 x ∧ 
  (∀ y : ℕ, is_five_digit y ∧ is_congruent_to_2_mod_17 y → x ≤ y) := 
begin
  use 10013,
  split,
  { -- Check if it's a five digit number
    unfold is_five_digit,
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Check if it's congruent to 2 mod 17
    unfold is_congruent_to_2_mod_17,
    exact by norm_num },
  { -- Prove it is the smallest
    intros y hy,
    have h_congruent : y % 17 = 2 := hy.2,
    have h_five_digit : 10000 ≤ y ∧ y < 100000 := hy.1,
    sorry
  }
end

end smallest_five_digit_congruent_to_2_mod_17_l138_138075


namespace jeremy_distance_proof_l138_138339

noncomputable def distance_to_school 
    (v : ℝ) (t_r t_q t_b : ℝ) : Prop :=
  let d := v * t_r in
  let v_q := v + 25 in
  let v_b := v - 10 in
  d = v_q * t_q ∧ d = v_b * t_b ∧ d = 12.5

theorem jeremy_distance_proof
    (v : ℝ) (t_r t_q t_b : ℝ) 
    (hr1 : t_r = 1/2)
    (hq1 : t_q = 1/4)
    (hb1 : t_b = 2/3)
    (hrush : d = v * t_r)
    (hquiet : d = (v + 25) * t_q)
    (hback : d = (v - 10) * t_b) : 
  distance_to_school v t_r t_q t_b :=
by {
  unfold distance_to_school,
  rw [hrush, hquiet, hback],
  split,
  { sorry },
  { sorry },
  { sorry }
}

end jeremy_distance_proof_l138_138339


namespace friends_new_games_l138_138942

theorem friends_new_games (Katie_new_games : ℕ) (Total_new_games : ℕ) 
  (hK : Katie_new_games = 84) (hT : Total_new_games = 92) : 
  (Total_new_games - Katie_new_games = 8) :=
by {
  rw [hK, hT],
  exact rfl,
}

end friends_new_games_l138_138942


namespace horner_method_V1_at_5_l138_138868

def polynomial (x : ℝ) : ℝ := 4 * x^5 + 2 * x^4 + 3.5 * x^3 - 2.6 * x^2 + 1.7 * x - 0.8

theorem horner_method_V1_at_5 :
  let V0 := 4 in
  let V1 := V0 * 5 + 2 in
  V1 = 22 :=
by
  sorry

end horner_method_V1_at_5_l138_138868


namespace number_of_true_inequalities_l138_138371

noncomputable def question_and_conditions (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hx2a2 : x^2 < a^2) (hy2b2 : y^2 < b^2) : Prop :=
  let ineqs := [ x + y < a + b, x + y^2 < a + b^2, xy < ab, abs (x / y) < abs (a / b) ] in
  (ineqs.filter id).length = 2

-- This is the statement of the problem
theorem number_of_true_inequalities (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hx2a2 : x^2 < a^2) (hy2b2 : y^2 < b^2) : 
  question_and_conditions x y a b hx hy ha hb hx2a2 hy2b2 := 
sorry

end number_of_true_inequalities_l138_138371


namespace roots_with_difference_one_l138_138819

theorem roots_with_difference_one (p : ℝ) :
  (∃ α β γ : ℝ, β = α + 1 ∧ polynomial.map (λ x, x^3 + 3*p*x^2 + (4*p-1)*x + p) =
    polynomial.map (λ x, (x - α)*(x - β)*(x - γ)))
  ↔ p = 0 ∨ p = 6 / 5 ∨ p = 10 / 9 :=
sorry

end roots_with_difference_one_l138_138819


namespace problem_20_l138_138735

open Real

def point := ℝ × ℝ

def A : point := (-1,0)
def B : point := (3,4)
def P : point

-- The line CD is the perpendicular bisector of AB
def is_perpendicular_bisector (L : point → Prop) (A B : point) : Prop :=
  let midpoint := ((fst A + fst B) / 2, (snd A + snd B) / 2)
  ∧
    ∀ (x : point),
      L x → fst x + snd x = 3

def line_equation := { L : point → Prop // is_perpendicular_bisector L A B }

-- Circle P passes through points A and B with diameter |CD| = 4 * sqrt 10
def passes_through (c : point) (r : ℝ) (x : point) : Prop :=
  (fst x - fst c) ^ 2 + (snd x - snd c) ^ 2 = r ^ 2

def circle_equation (P : point) (r : ℝ) : Prop :=
  passes_through P r A ∧ passes_through P r B
  ∧ ∃ (r : ℝ), |P - A| = r

theorem problem_20 :
  ∃ (L : line_equation), ∃ (P : point), ∃ (r : ℝ),
    (fst P + snd P = 3)
    ∧ (
      (fst P - 5) ^ 2 + (snd P + 2) ^ 2 = 40
      ∨ (fst P + 3) ^ 2 + (snd P - 6) ^ 2 = 40
    ) := sorry

end problem_20_l138_138735


namespace find_a2008_l138_138028

theorem find_a2008 (a : Fin 2009 → ℕ)
  (h : ∀ i : Fin 2007, ∃ p1 p2 p3 : ℕ, p1 ≠ p2 ∧ p2 ≠ p3 ∧ p1 ≠ p3 ∧ ∃ (i1 i2 : ℕ), 
    (1 ≤ i1 ∧ i1 ≤ 9) ∧ (1 ≤ i2 ∧ i2 ≤ 9) ∧ 10 * i1 + i2 = 10 * a i + a (i + 1) ∧ is_prime p1 ∧ is_prime p2 ∧ is_prime p3 ∧ p1 * p2 * p3 = 10 * i1 + i2) :
  a 2008 = 6 :=
begin
  sorry
end

end find_a2008_l138_138028


namespace midpoint_m_distance_M_M_l138_138375

-- Defining the points A and B initially
variables (a b c d : ℝ)

-- Defining the initial midpoint M
def initial_midpoint_m : ℝ := (a + c) / 2
def initial_midpoint_n : ℝ := (b + d) / 2

-- Defining the new positions of A and B
def new_A : (ℝ × ℝ) := (a + 5, b + 10)
def new_B : (ℝ × ℝ) := (c - 5, d - 5)

-- Defining the new midpoint M'
def new_midpoint_m' : ℝ := initial_midpoint_m a c
def new_midpoint_n' : ℝ := initial_midpoint_n b d + 2.5

-- Prove the new midpoint M' is as expected
theorem midpoint_m' (a b c d : ℝ) :
  (new_midpoint_m' a c) = (initial_midpoint_m a c) ∧
  (new_midpoint_n' b d) = (initial_midpoint_n b d + 2.5) := by
  sorry

-- Prove the distance between M and M' is 2.5
theorem distance_M_M' (a b c d : ℝ) :
  real.sqrt ((initial_midpoint_m a c - new_midpoint_m' a c) ^ 2 + 
  (initial_midpoint_n b d - new_midpoint_n' b d) ^ 2) = 2.5 := by
  sorry

end midpoint_m_distance_M_M_l138_138375


namespace system_of_equations_implies_quadratic_l138_138182

theorem system_of_equations_implies_quadratic (x y : ℝ) :
  (3 * x^2 + 9 * x + 4 * y + 2 = 0) ∧ (3 * x + y + 4 = 0) → (y^2 + 11 * y - 14 = 0) := by
  sorry

end system_of_equations_implies_quadratic_l138_138182


namespace inheritance_amount_l138_138943

theorem inheritance_amount (x : ℝ)
  (h1 : x ≥ 0)
  (federal_tax_rate : ℝ := 0.25)
  (state_tax_rate : ℝ := 0.15)
  (total_tax_paid : ℝ := 20000) :
  0.3625 * x = total_tax_paid → x ≈ 55172 := by
  sorry

end inheritance_amount_l138_138943


namespace probability_factor_of_36_l138_138557

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138557


namespace probability_factor_36_l138_138604

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138604


namespace lena_calculation_l138_138945

def round_to_nearest_ten (n : ℕ) : ℕ :=
  if n % 10 < 5 then n - n % 10 else n + (10 - n % 10)

theorem lena_calculation :
  round_to_nearest_ten (63 + 2 * 29) = 120 :=
by
  sorry

end lena_calculation_l138_138945


namespace max_value_3x_sub_9x_l138_138209

open Real

theorem max_value_3x_sub_9x : ∃ x : ℝ, 3^x - 9^x ≤ 1/4 ∧ (∀ y : ℝ, 3^y - 9^y ≤ 3^x - 9^x) :=
by
  sorry

end max_value_3x_sub_9x_l138_138209


namespace cartesian_equations_minimum_distance_l138_138915

noncomputable def curve_param (θ : ℝ) : ℝ × ℝ :=
  (cos θ, 2 * sin θ)

def curve_cart_eq (x y : ℝ) : Prop :=
  x^2 + (y^2 / 4) = 1

def line_polar_eq (ρ θ : ℝ) : Prop :=
  2 * ρ * cos θ + sqrt 3 * ρ * sin θ + 11 = 0

def line_cart_eq (x y : ℝ) : Prop :=
  2 * x + sqrt 3 * y + 11 = 0

def distance (x y : ℝ) :=
  abs (2 * x + sqrt 3 * y + 11) / real.sqrt (2^2 + (sqrt 3)^2)

theorem cartesian_equations (θ : ℝ) (ρ : ℝ) :
  curve_cart_eq (cos θ) (2 * sin θ) ∧ line_cart_eq (cos θ) (2 * sin θ) :=
sorry

theorem minimum_distance : 
  ∃ θ : ℝ, dist (cos θ) (2 * sin θ) = sqrt 7 :=
sorry

end cartesian_equations_minimum_distance_l138_138915


namespace smallest_positive_multiple_of_3_4_5_is_60_l138_138077

theorem smallest_positive_multiple_of_3_4_5_is_60 :
  ∃ n : ℕ, n > 0 ∧ (n % 3 = 0) ∧ (n % 4 = 0) ∧ (n % 5 = 0) ∧ n = 60 :=
by
  use 60
  sorry

end smallest_positive_multiple_of_3_4_5_is_60_l138_138077


namespace fixed_points_odd_if_odd_function_exists_even_function_with_odd_fixed_points_l138_138411

-- Define the conditions, the problem involves odd and even functions, and fixed points
def is_fixed_point (f : ℝ → ℝ) (c : ℝ) : Prop := f c = c

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Prove statement 1: If f is odd, then the number of its fixed points is odd
theorem fixed_points_odd_if_odd_function (f : ℝ → ℝ) : is_odd_function f → (∃ n : ℕ, n % 2 = 1 ∧ (set.finite {c : ℝ | is_fixed_point f c}).to_finset.card = n) :=
by
  sorry

-- Prove statement 2: There exists an even function with an odd number of fixed points
theorem exists_even_function_with_odd_fixed_points : ∃ f : ℝ → ℝ, is_even_function f ∧ (∃ n : ℕ, n % 2 = 1 ∧ (set.finite {c : ℝ | is_fixed_point f c}).to_finset.card = n) :=
by
  sorry

end fixed_points_odd_if_odd_function_exists_even_function_with_odd_fixed_points_l138_138411


namespace resistance_parallel_l138_138909

theorem resistance_parallel (x y r : ℝ) (hy : y = 6) (hr : r = 2.4) 
  (h : 1 / r = 1 / x + 1 / y) : x = 4 :=
  sorry

end resistance_parallel_l138_138909


namespace probability_factor_of_36_is_1_over_4_l138_138497

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138497


namespace square_in_base_3_contains_1_l138_138395

theorem square_in_base_3_contains_1 (n : ℤ) : ∃ d : ℕ, d ∈ (n^2).digits 3 ∧ d = 1 :=
  sorry

end square_in_base_3_contains_1_l138_138395


namespace quadratic_unique_solution_l138_138826

theorem quadratic_unique_solution (c : ℝ) (h : c ≠ 0) :
  (∃ b : ℝ, b > 0 ∧ (λ (x : ℝ), x^2 + (b^2 + 1/b^2) * x + c = 0).discriminant = 0) ↔
  (c = (1 + Real.sqrt 2)/2 ∨ c = (1 - Real.sqrt 2)/2) :=
by
  sorry

end quadratic_unique_solution_l138_138826


namespace probability_factor_of_36_l138_138690

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138690


namespace Annika_hiking_rate_exact_l138_138778

-- Define the problem's conditions
variables (distanceEast initiallyHiked distanceFullEast totalDistance totalTime : ℝ)

-- Define the given conditions
def conditions : Prop :=
  distanceEast = 3 ∧
  initiallyHiked = 2.5 ∧
  totalDistance = 2 * distanceEast ∧
  totalTime = 35

-- Define the target rate
def rate (totalTime totalDistance : ℝ) : ℝ :=
  totalTime / totalDistance

-- The proof statement in Lean 4
theorem Annika_hiking_rate_exact (h: conditions) : rate totalTime totalDistance = 35 / 6 :=
by 
  -- You could add the proof here using the 'sorry' keyword for now.
  sorry

#check Annika_hiking_rate_exact

end Annika_hiking_rate_exact_l138_138778


namespace probability_factor_36_l138_138544

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138544


namespace point_transform_l138_138377

theorem point_transform : 
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  P' = (-3, 0) :=
by
  let P := (-2, -3)
  let P' := (P.1 - 1, P.2 + 3)
  show P' = (-3, 0)
  sorry

end point_transform_l138_138377


namespace maximum_sum_of_factors_exists_maximum_sum_of_factors_l138_138331

theorem maximum_sum_of_factors {A B C : ℕ} (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C)
  (h4 : A * B * C = 2023) : A + B + C ≤ 297 :=
sorry

theorem exists_maximum_sum_of_factors : ∃ (A B C : ℕ), A ≠ B ∧ B ≠ C ∧ A ≠ C ∧ A * B * C = 2023 ∧ A + B + C = 297 :=
sorry

end maximum_sum_of_factors_exists_maximum_sum_of_factors_l138_138331


namespace hyperbola_eccentricity_l138_138204

-- Definition of the hyperbola and the eccentricity
theorem hyperbola_eccentricity : 
  ∀ (x y : ℝ), (x^2 / 4 - y^2 / 2 = 1) → ∃ e : ℝ, e = √6 / 2 :=
by
  sorry

end hyperbola_eccentricity_l138_138204


namespace prove_arithmetic_and_find_sum_find_sum_bn_l138_138237

open_locale big_operators

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else sorry -- Placeholder for the expression of a_n

noncomputable def s (n : ℕ) : ℕ := 
sorry -- Placeholder for the sum of first n terms s_n

noncomputable def b (n : ℕ) : ℕ :=
(s n / (2 * n + 1)) + (2^n / s n)

def is_arithmetic_seq (f : ℕ → ℤ) : Prop :=
∀ n : ℕ, f (n + 1) - f n = f 1

theorem prove_arithmetic_and_find_sum :
  is_arithmetic_seq (λ n, 1 / s n) ∧ ∑ k in finset.range n, (1 / s k) = n^2 :=
sorry

theorem find_sum_bn :
  ∑ k in finset.range n, b k = (n / (2 * n + 1) + (2 * n - 3) * 2^(n + 1) - 6) :=
sorry

end prove_arithmetic_and_find_sum_find_sum_bn_l138_138237


namespace intersection_complement_empty_l138_138874

open Set

variable {U : Type} 
noncomputable def U := ({1, 2, 3, 4} : Set ℕ)
noncomputable def A := ({1, 3} : Set ℕ)
noncomputable def B := ({1, 3, 4} : Set ℕ)

theorem intersection_complement_empty : A ∩ (U \ B) = ∅ :=
by
  sorry

end intersection_complement_empty_l138_138874


namespace tangent_line_at_neg2_l138_138797

def f (x : ℝ) (b : ℝ) : ℝ := x^3 - 12 * x + b

theorem tangent_line_at_neg2 (b : ℝ) (h : b = -6) :
  let t := (-2, f (-2) b) in
  (f' x b).eval (t.fst) = 0 ∧ f (t.fst) b = 10 :=
by
  sorry

end tangent_line_at_neg2_l138_138797


namespace stratified_sampling_l138_138224

variable (H M L total_sample : ℕ)
variable (H_fams M_fams L_fams : ℕ)

-- Conditions
def community : Prop := H_fams = 150 ∧ M_fams = 360 ∧ L_fams = 90
def total_population : Prop := H_fams + M_fams + L_fams = 600
def sample_size : Prop := total_sample = 100

-- Statement
theorem stratified_sampling (H_fams M_fams L_fams : ℕ) (total_sample : ℕ)
  (h_com : community H_fams M_fams L_fams)
  (h_total_pop : total_population H_fams M_fams L_fams)
  (h_sample_size : sample_size total_sample)
  : H = 25 ∧ M = 60 ∧ L = 15 :=
by
  sorry

end stratified_sampling_l138_138224


namespace factor_probability_l138_138635

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138635


namespace evaluate_expression_l138_138794

theorem evaluate_expression : 
  |-2| + (1 / 4) - 1 - 4 * Real.cos (Real.pi / 4) + Real.sqrt 8 = 5 / 4 :=
by
  sorry

end evaluate_expression_l138_138794


namespace meal_combinations_l138_138896

theorem meal_combinations (n : ℕ) (m : ℕ) (h1 : n = 12) (h2 : m = 11) : n * m = 132 :=
by
  rw [h1, h2]
  exact rfl

end meal_combinations_l138_138896


namespace probability_at_most_five_diff_digits_l138_138765

theorem probability_at_most_five_diff_digits : 
  let total_sequences := 10^7
  let binom_10_6 := 210
  let binom_10_7 := 120
  let factorial_7 := 5040
  let factorial_7_div_2 := 2520
  let num_sequences_6_diff := binom_10_6 * 6 * factorial_7_div_2
  let num_sequences_7_diff := binom_10_7 * factorial_7
  let num_sequences_at_most_5 := total_sequences - num_sequences_6_diff - num_sequences_7_diff
  let probability := num_sequences_at_most_5 / total_sequences in
  probability = 0.622 := 
by { sorry }

end probability_at_most_five_diff_digits_l138_138765


namespace eval_expression_l138_138433

theorem eval_expression : (20 - 16) * (12 + 8) / 4 = 20 := 
by 
  sorry

end eval_expression_l138_138433


namespace volume_prism_PQR_eq_25_l138_138387

noncomputable def volume_of_prism :=
  let base_area := (1/2) * real.sqrt 5 * real.sqrt 5
  let height := 10
  base_area * height

theorem volume_prism_PQR_eq_25 :
  (let PQ := real.sqrt 5
   let PR := real.sqrt 5
   let height := 10
   volume_of_prism = 25) :=
by 
  sorry

end volume_prism_PQR_eq_25_l138_138387


namespace ilios_population_2060_l138_138156

-- Defining the initial conditions
constant initial_population : ℕ := 100
constant doubling_period : ℕ := 15

-- Function to calculate the population at a given year
noncomputable def population_at_year (initial_population : ℕ) (doubling_period : ℕ) (start_year target_year : ℕ) : ℕ :=
let years := target_year - start_year
    doubling_intervals := years / doubling_period
    remaining_years := years % doubling_period
    growth_factor := (2 : ℚ)^(remaining_years / doubling_period : ℚ) in
nat_ceil (initial_population * 2^doubling_intervals * growth_factor)

-- State the theorem to be proved
theorem ilios_population_2060 : population_at_year initial_population doubling_period 1995 2060 = 1838 :=
begin
  sorry
end

end ilios_population_2060_l138_138156


namespace find_b_l138_138952

noncomputable def f (a x : ℝ) : ℝ := sqrt (abs (x - a)) + sqrt (abs (x + a))

theorem find_b (a b : ℝ) (h_a : a > 0) (h_g_has_zeros : ∃ x₁ x₂ x₃ : ℝ, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧ f a x₁ - b = 0 ∧ f a x₂ - b = 0 ∧ f a x₃ - b = 0) (h_b_is_zero : f a 0 - b = 0) : b = 16 / 5 := 
sorry

end find_b_l138_138952


namespace closest_vector_l138_138830

open scoped Matrix

noncomputable def vector_v (t : ℚ) : Matrix (Fin 3) (Fin 1) ℚ :=
  ![![3 + 5 * t],
    ![-2 + 4 * t],
    ![1 + 2 * t]]

def vector_a : Matrix (Fin 3) (Fin 1) ℚ :=
  ![![-1],
    ![1],
    ![-3]]

def direction_vector : Matrix (Fin 3) (Fin 1) ℚ :=
  ![![5],
    ![4],
    ![2]]

theorem closest_vector (t : ℚ) (h : vector_v t - vector_a ⬝ direction_vector = 0) : t = -16 / 45 :=
sorry

end closest_vector_l138_138830


namespace sum_of_even_numbers_between_1_and_31_l138_138724

theorem sum_of_even_numbers_between_1_and_31 : 
  (∑ k in (Finset.filter (λ x, x % 2 = 0) (Finset.range 32)), k) = 240 := 
by
  sorry

end sum_of_even_numbers_between_1_and_31_l138_138724


namespace intersections_of_square_and_pentagon_l138_138026

theorem intersections_of_square_and_pentagon
  (CASH MONEY : Set Point)
  (circ : Circle)
  (inscribed_in_circle : ∀ x ∈ CASH ∪ MONEY, x ∈ circ)
  (no_shared_vertices : ∀ x ∈ CASH ∩ MONEY, False) :
  ∃ (intersections : ℕ), intersections = 8 :=
by
  sorry

end intersections_of_square_and_pentagon_l138_138026


namespace move_both_horizontal_vertical_l138_138752

variables (M W : ℝ) -- M: Mass of the block, W: Mass of the wedge
variables (v : ℝ) -- v: velocity of the block
variables (h : ℝ) -- h: height of the incline
variables (t : ℝ) -- time
variables (x_b y_b : ℝ → ℝ) -- x_b, y_b: horizontal and vertical position functions of the block over time
variables (x_w : ℝ → ℝ) -- x_w: horizontal position function of the wedge over time

-- Assume initial conditions
axiom initial_conditions : x_b 0 = 0 ∧ y_b 0 = h ∧ x_w 0 = 0

-- Assume the block slides down with horizontal and vertical velocity components
axiom block_motion : ∀ t, y_b t = h - t * v ∧ x_b t = t * v

-- Assume the wedge moves due to reaction forces
axiom wedge_motion : ∀ t, x_w t = - (M / W) * x_b t

-- Define the system's center of mass position functions
def X_cm (t : ℝ) : ℝ := (M * x_b t + W * x_w t) / (M + W)
def Y_cm (t : ℝ) : ℝ := (M * y_b t) / (M + W)

-- The Lean 4 statement we need to prove:
theorem move_both_horizontal_vertical : (X_cm 0 = 0) ∧ (Y_cm 0 = h) ∧ (∀ t, X_cm t ≠ 0 ∧ Y_cm t ≠ h) :=
by
  sorry

end move_both_horizontal_vertical_l138_138752


namespace probability_factor_of_36_l138_138670

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138670


namespace find_tip_percentage_l138_138431

def original_bill : ℝ := 139.00
def per_person_share : ℝ := 30.58
def number_of_people : ℕ := 5

theorem find_tip_percentage (original_bill : ℝ) (per_person_share : ℝ) (number_of_people : ℕ) 
  (total_paid : ℝ := per_person_share * number_of_people) 
  (tip_amount : ℝ := total_paid - original_bill) : 
  (tip_amount / original_bill) * 100 = 10 :=
by
  sorry

end find_tip_percentage_l138_138431


namespace sum_first_60_terms_l138_138427

theorem sum_first_60_terms {a : ℕ → ℤ}
  (h : ∀ n, a (n + 1) + (-1)^n * a n = 2 * n - 1) :
  (Finset.range 60).sum a = 1830 :=
sorry

end sum_first_60_terms_l138_138427


namespace log_comparison_l138_138357

open Real

noncomputable def a := log 4 / log 5  -- a = log_5(4)
noncomputable def b := log 5 / log 3  -- b = log_3(5)
noncomputable def c := log 5 / log 4  -- c = log_4(5)

theorem log_comparison : a < c ∧ c < b := 
by
  sorry

end log_comparison_l138_138357


namespace ratio_of_s_t_l138_138062

theorem ratio_of_s_t (s t : ℚ)
  (h1 : ∀ x : ℚ, (2 * x + 5 = 0) → s = x)
  (h2 : ∀ x : ℚ, (7 * x + 5 = 0) → t = x) :
  (s / t) = 7 / 2 := by
  subst h1
  subst h2
  sorry

end ratio_of_s_t_l138_138062


namespace probability_factor_of_36_l138_138529

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138529


namespace probability_factor_of_36_l138_138676

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138676


namespace conjugate_in_third_quadrant_l138_138325

def z : ℂ := complex.I * (1 + 2 * complex.I)
def z_conjugate : ℂ := complex.conj z

theorem conjugate_in_third_quadrant (z z_conjugate : ℂ) (hz: z = complex.I * (1 + 2 * complex.I)) :
  z_conjugate.re < 0 ∧ z_conjugate.im < 0 :=
by {
  rw [complex.conj, hz],
  sorry
}

end conjugate_in_third_quadrant_l138_138325


namespace depth_of_sea_l138_138750

theorem depth_of_sea (height_above_sea : ℝ) (horizontal_distance : ℝ) : 
  height_above_sea = 40 → horizontal_distance = 84 → x = 68.2 :=
by 
  assume h1 : height_above_sea = 40, 
  assume h2 : horizontal_distance = 84, 
  sorry

end depth_of_sea_l138_138750


namespace perimeter_of_plot_l138_138722

variable (w : ℕ) (l : ℕ)
def cost_per_meter : ℕ := 65
def total_cost : ℕ := 1170

axiom length_is_width_plus_ten : l = w + 10
axiom cost_condition : cost_per_meter * 10 * 2 * (w + length_is_width_plus_ten) = total_cost

theorem perimeter_of_plot : ∃ (P : ℕ), P = 10 * 2 * (w + length_is_width_plus_ten) ∧ P = 180 :=
by
  sorry

end perimeter_of_plot_l138_138722


namespace probability_factor_of_36_l138_138540

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138540


namespace first_offset_length_l138_138203

theorem first_offset_length (diagonal : ℝ) (offset2 : ℝ) (area : ℝ) (h_diagonal : diagonal = 50) (h_offset2 : offset2 = 8) (h_area : area = 450) :
  ∃ offset1 : ℝ, offset1 = 10 :=
by
  sorry

end first_offset_length_l138_138203


namespace find_function_and_interval_find_m_and_sum_of_roots_l138_138234

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi / 3)

theorem find_function_and_interval :
  (∀ x ∈ {-13 * Real.pi / 6, -5 * Real.pi / 3, -7 * Real.pi / 6, -2 * Real.pi / 3, -Real.pi / 6, Real.pi / 3, 5 * Real.pi / 6}, 
    f x = (x match | -13 * Real.pi / 6 => -2 | -5 * Real.pi / 3 => 0 | -7 * Real.pi / 6 => 2 | -2 * Real.pi / 3 => 0 
                      | -Real.pi / 6 => -2 | Real.pi / 3 => 0 | 5 * Real.pi / 6 => 2 | _ => sorry)) ∧ 
  ∀ k ∈ Set.Z, (∀ x ∈ Set.Icc (-(Real.pi / 6) + 2 * k * Real.pi) ((5 * Real.pi / 6) + 2 * k * Real.pi), Real.derivative f x > 0) :=
sorry

theorem find_m_and_sum_of_roots (k : ℝ) :
  (k = 3 ∨ k = -3) →
  (∃ m ∈ ((0, 2) ∩ Set.Ioo (-2, -Real.sqrt (3)) (Real.sqrt (3), 2)),
    let roots := {x ∈ Set.Ioo 0 (4 * Real.pi / 9) | f (k * x) = m} in
    roots.card = 2 ∧
    let x₁ := roots.min' (sorry_proof _),
        x₂ := roots.max' (sorry_proof _) in
    x₁ + x₂ = if k = 3 then 5 * Real.pi / 9 else if k = -3 then (Real.pi / 9 ∨ 7 * Real.pi / 9) else 0) :=
sorry

end find_function_and_interval_find_m_and_sum_of_roots_l138_138234


namespace probability_factor_of_36_l138_138669

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138669


namespace probability_factor_of_36_l138_138645

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138645


namespace probability_factor_of_36_l138_138486

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138486


namespace number_of_integers_with_gcd_21_eq_3_l138_138831

theorem number_of_integers_with_gcd_21_eq_3 : 
  (∃ S : Finset ℕ, (∀ n ∈ S, 1 ≤ n ∧ n ≤ 50 ∧ Nat.gcd 21 n = 3) ∧ S.card = 14) := 
begin
  sorry
end

end number_of_integers_with_gcd_21_eq_3_l138_138831


namespace number_of_digit_combinations_l138_138052

theorem number_of_digit_combinations :
  {n : ℕ // ∃ a b c d e : ℕ, 
    (a * b * c * d * e = 180) ∧ 
    (1 ≤ a ∧ a ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (1 ≤ d ∧ d ≤ 9) ∧ (1 ≤ e ∧ e ≤ 9) ∧
    (n = 360)} := 
by 
sorrr

end number_of_digit_combinations_l138_138052


namespace pencils_in_pack_l138_138941

theorem pencils_in_pack (pencils_per_week : ℕ) (cost_per_pack : ℕ) (total_spent : ℕ) (total_days : ℕ) :
  pencils_per_week = 10 → cost_per_pack = 4 → total_spent = 12 → total_days = 45 → 
  (total_days / 5 * pencils_per_week) / (total_spent / cost_per_pack) = 30 :=
by 
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end pencils_in_pack_l138_138941


namespace sum_inverse_roots_polynomial_eq_l138_138954

/-- 
Given that \( b_1, b_2, \dots, b_{10} \) are the roots of the polynomial 
\( x^{10} + x^9 + \dots + x^2 + x - 105 = 0 \), prove that
\(\sum_{n = 1}^{10} \frac{1}{2 - b_n} = \frac{55}{104}\). 
-/
theorem sum_inverse_roots_polynomial_eq (b : Fin 10 → ℝ) (h : ∀ n, Polynomial.eval b^[n] (Polynomial.monomial 10 1 + Polynomial.monomial 9 1 + Polynomial.monomial 8 1 +
  Polynomial.monomial 7 1 + Polynomial.monomial 6 1 + Polynomial.monomial 5 1 + Polynomial.monomial 4 1 + Polynomial.monomial 3 1 + Polynomial.monomial 2 1 +
  Polynomial.monomial 1 1 - 105) = 0) :
  ∑ n in Finset.finRange 10, (1 / (2 - b n)) = 55 / 104 :=
by
  sorry

end sum_inverse_roots_polynomial_eq_l138_138954


namespace range_of_m_range_of_t_sum_ln_gt_n_minus_2_l138_138854

-- Definitions based on given conditions
def P_on_graph (x : ℝ) (y : ℝ) := y = 1 + Real.log x
def k (x : ℝ) := (1 + Real.log x) / x
def f_derivative (x : ℝ) := -Real.log x / (x^2)

-- Range of m where f(x) has extreme value in (m, m + 1/3) and m > 0
theorem range_of_m : {m : ℝ | ∃ x ∈ Set.interval (m : ℝ) (m + 1/3), f_derivative x = 0} =
  {m : ℝ | (2/3 : ℝ) < m ∧ m < 1} :=
sorry

-- Range of t for which f(x) ≥ t / (x + 1) always holds for x ≥ 1
theorem range_of_t : {t : ℝ | ∀ x ≥ 1, k(x) ≥ t / (x + 1)} = {t : ℝ | t ≤ 2} :=
sorry

-- Prove that sum ln[i(i+1)] > n - 2 for n ∈ ℕ*
theorem sum_ln_gt_n_minus_2 (n : ℕ) (hn : 0 < n) :
  ∑ i in Finset.range n, Real.log (i*(i+1)) > n - 2 :=
sorry

end range_of_m_range_of_t_sum_ln_gt_n_minus_2_l138_138854


namespace factor_probability_l138_138633

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138633


namespace resultingPoint_is_correct_l138_138378

def Point (α : Type _) := (x : α) × (y : α)

def initialPoint : Point Int := (-2, -3)

def moveLeft (p : Point Int) (units : Int) : Point Int :=
  (p.1 - units, p.2)

def moveUp (p : Point Int) (units : Int) : Point Int :=
  (p.1, p.2 + units)

theorem resultingPoint_is_correct : 
  (moveUp (moveLeft initialPoint 1) 3) = (-3, 0) :=
by
  sorry

end resultingPoint_is_correct_l138_138378


namespace probability_factor_of_36_l138_138487

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138487


namespace handshake_count_l138_138440

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end handshake_count_l138_138440


namespace exercise_statement_l138_138798

variable (x y z : ℝ)

def N : Matrix (Fin 3) (Fin 3) ℝ :=
  !![ [0, 3*y, 2*z],
      [2*x, y, -z],
      [2*x, -y, z] ]

theorem exercise_statement
  (h : Nᵀ.mul N = 1) :
  x^2 + y^2 + z^2 = 47 / 120 :=
sorry

end exercise_statement_l138_138798


namespace area_of_triangle_eq_twice_squared_circumradius_multiplied_by_product_of_sines_l138_138015

-- Definitions for the conditions
def angles (α β γ : ℝ) := α + β + γ = π
def circumradius (R : ℝ) := R > 0

-- Lean statement for the proof problem
theorem area_of_triangle_eq_twice_squared_circumradius_multiplied_by_product_of_sines
  (α β γ R : ℝ) 
  (h_angles : angles α β γ)
  (h_circumradius : circumradius R) :
  let S_Δ := (R * R * 2 * (Real.sin α) * (Real.sin β) * (Real.sin γ)) in
  S_Δ = 2 * R^2 * Real.sin α * Real.sin β * Real.sin γ :=
by
  sorry

end area_of_triangle_eq_twice_squared_circumradius_multiplied_by_product_of_sines_l138_138015


namespace num_topping_combinations_l138_138806

-- Define the conditions as constants in Lean
constant cheese_options : ℕ := 3
constant meat_options : ℕ := 4
constant vegetable_options : ℕ := 5
constant pepperoni_option : ℕ := 1 -- Only one option for pepperoni
constant restricted_vegetable_options : ℕ := 1 -- Only one restricted option (peppers)

-- Define the total number of combinations without restrictions
def total_combinations : ℕ := cheese_options * meat_options * vegetable_options

-- Define the number of restricted combinations (pepperoni and peppers)
def restricted_combinations : ℕ := cheese_options * pepperoni_option * restricted_vegetable_options

-- Define the allowed combinations
def allowed_combinations : ℕ := total_combinations - restricted_combinations

-- The theorem stating the problem question and expected answer
theorem num_topping_combinations : allowed_combinations = 57 := by
  sorry

end num_topping_combinations_l138_138806


namespace magnitude_of_vec_sum_l138_138278

noncomputable def vec_a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def vec_b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))
noncomputable def vec_sum : ℝ × ℝ := (vec_a.1 + 2 * vec_b.1, vec_a.2 + 2 * vec_b.2)
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem magnitude_of_vec_sum : magnitude vec_sum = Real.sqrt 7 := 
by 
  sorry

end magnitude_of_vec_sum_l138_138278


namespace projection_of_b_onto_a_l138_138878

variables (a b : ℝ ^ 3) (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hab : dot_product a b = 1)

theorem projection_of_b_onto_a :
  (‖b‖ * (dot_product a b / (‖a‖ * ‖b‖)) * (a / ‖a‖)) = (1/4) • a :=
by sorry

end projection_of_b_onto_a_l138_138878


namespace largest_possible_N_l138_138982

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138982


namespace probability_factor_of_36_l138_138490

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138490


namespace sum_of_areas_l138_138816

-- Define the problem conditions
def radius_of_circle : ℝ := 1.5
def number_of_disks : ℕ := 15
def cos_pi_div_fifteen : ℝ := Real.cos (Real.pi / 15)
def radius_of_small_disk : ℝ := radius_of_circle * cos_pi_div_fifteen

-- Define the areas
def area_of_one_disk : ℝ := Real.pi * radius_of_small_disk ^ 2
def total_area_of_disks : ℝ := number_of_disks * area_of_one_disk

-- Define the form of the answer
def a : ℕ := 28
def b : ℕ := 3
def c : ℕ := 5
def total_area_expected : ℝ := Real.pi * (a - (b * Real.sqrt c))

-- Main theorem to prove
theorem sum_of_areas :
  total_area_of_disks = total_area_expected ∧ (a + b + c) = 36 :=
by
  sorry

end sum_of_areas_l138_138816


namespace smallest_five_digit_congruent_two_mod_seventeen_l138_138073

theorem smallest_five_digit_congruent_two_mod_seventeen : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 ∧ n = 10013 :=
by
  sorry

end smallest_five_digit_congruent_two_mod_seventeen_l138_138073


namespace cube_max_intersection_edges_l138_138013

theorem cube_max_intersection_edges :
  ∀ (P : Plane) (C : Cube),
  (P.intersection C).is_polygon →
  (P.intersection C).edges ≤ 6 :=
sorry

end cube_max_intersection_edges_l138_138013


namespace radius_of_base_of_cone_volume_of_sphere_l138_138862

noncomputable def radius_base_cone : ℝ := 2 * Real.sqrt 3
def slant_height_cone : ℝ := 2 * Real.sqrt 3
def volume_sphere (R : ℝ) : ℝ := (4 / 3) * Real.pi * R^3

theorem radius_of_base_of_cone :
  let r := radius_base_cone / 2 in
  r = Real.sqrt 3 := by
  let r := radius_base_cone / 2
  have h : r = Real.sqrt 3 := by sorry
  exact h

theorem volume_of_sphere :
  let R := slant_height_cone / 2 in
  volume_sphere R = 32 * Real.pi / 3 := by
  let R := slant_height_cone / 2
  have h : volume_sphere R = 32 * Real.pi / 3 := by sorry
  exact h

end radius_of_base_of_cone_volume_of_sphere_l138_138862


namespace probability_divisor_of_36_l138_138666

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138666


namespace factor_probability_l138_138630

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138630


namespace probability_factor_36_l138_138554

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138554


namespace largest_possible_N_l138_138978

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138978


namespace family_boys_girls_l138_138304

theorem family_boys_girls (B G : ℕ) 
  (h1 : B - 1 = G) 
  (h2 : B = 2 * (G - 1)) : 
  B = 4 ∧ G = 3 := 
by {
  sorry
}

end family_boys_girls_l138_138304


namespace intervals_satisfying_inequalities_intervals_not_satisfying_inequalities_l138_138707

noncomputable def satisfies_inequalities (x : ℝ) : Prop :=
(0 ≤ x ∧ x < 360) ∧ (
  (sin (2 * x) > sin x) ∨ 
  (cos (2 * x) > cos x) ∨ 
  (tan (2 * x) > tan x) ∨ 
  (cot (2 * x) > cot x))

theorem intervals_satisfying_inequalities :
  {x | 0 < x ∧ x < 45} ∪
  {x | 90 < x ∧ x < 180} ∪
  {x | 180 < x ∧ x < 240} ∪
  {x | 270 < x ∧ x < 315} ⊆ {x : ℝ | satisfies_inequalities x} := 
sorry

theorem intervals_not_satisfying_inequalities :
  {x | 60 ≤ x ∧ x < 90} ∪
  {x | x = 0} ⊆ {x : ℝ | ¬ satisfies_inequalities x} :=
sorry

end intervals_satisfying_inequalities_intervals_not_satisfying_inequalities_l138_138707


namespace vector_magnitude_l138_138276

noncomputable def a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

theorem vector_magnitude : |(a.1 + 2 * b.1, a.2 + 2 * b.2)| = Real.sqrt 7 :=
by sorry

end vector_magnitude_l138_138276


namespace gcd_euclidean_120_168_gcd_subtraction_459_357_l138_138063

theorem gcd_euclidean_120_168 : Nat.gcd 120 168 = 24 := by
  sorry

theorem gcd_subtraction_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end gcd_euclidean_120_168_gcd_subtraction_459_357_l138_138063


namespace total_pupils_correct_l138_138314

-- Definitions of the conditions
def number_of_girls : ℕ := 308
def number_of_boys : ℕ := 318

-- Definition of the number of pupils
def total_number_of_pupils : ℕ := number_of_girls + number_of_boys

-- The theorem to be proven
theorem total_pupils_correct : total_number_of_pupils = 626 := by
  -- The proof would go here
  sorry

end total_pupils_correct_l138_138314


namespace cyclic_quadrilateral_area_l138_138007

-- Define the necessary parameters
def cyclic_quadrilateral := Type
def side_length : cyclic_quadrilateral → ℝ := sorry
def area (ABCD : cyclic_quadrilateral) := real.sqrt(40) * 3

theorem cyclic_quadrilateral_area {ABCD : cyclic_quadrilateral}
  (h_ab : side_length ABCD = 3)
  (h_bc : side_length ABCD = 4)
  (h_cd : side_length ABCD = 5)
  (h_da : side_length ABCD = 6)
  (h_cyclic : true)   -- Represents that ABCD is cyclic
  : area ABCD = 6 * real.sqrt 10 := 
sorry

#check cyclic_quadrilateral_area -- We add this line to make sure it compiles correctly.

end cyclic_quadrilateral_area_l138_138007


namespace projection_vector_l138_138879

variables {V : Type*} [InnerProductSpace ℝ V]

theorem projection_vector (a b : V) (h₁ : ∥a∥ = 2) (h₂ : ∥b∥ = 3) (h₃ : ⟪a, b⟫ = 1) :
  (inner b a / ∥a∥ ^ 2) • a = (1 / 4) • a :=
by
  sorry

end projection_vector_l138_138879


namespace debt_amount_is_40_l138_138405

theorem debt_amount_is_40 (l n t debt remaining : ℕ) (h_l : l = 6)
  (h_n1 : n = 5 * l) (h_n2 : n = 3 * t) (h_remaining : remaining = 6) 
  (h_share : ∀ x y z : ℕ, x = y ∧ y = z ∧ z = 2) :
  debt = 40 := 
by
  sorry

end debt_amount_is_40_l138_138405


namespace triangle_midpoint_third_l138_138973

-- Define variables for points A, B, C, D, E, F
variables (A B C D E F : Type)

-- Define midpoint property
def is_midpoint (M : Type) (P Q : Type) : Prop :=
  ∃ (x : ℝ), x > 0 ∧ x < 1 ∧ P = x * M + (1 - x) * Q

-- Definitions as per given conditions
variables [h1 : is_midpoint D B C] [h2 : is_midpoint E A D] [h3 : line_intersection BE AC F]

-- Define the main theorem to prove
theorem triangle_midpoint_third : AF = (1 / 3 : ℝ) * AC :=
  sorry

end triangle_midpoint_third_l138_138973


namespace max_value_of_3_pow_x_minus_9_pow_x_l138_138212

open Real

theorem max_value_of_3_pow_x_minus_9_pow_x :
  ∃ (x : ℝ), ∀ y : ℝ, 3^x - 9^x ≤ 3^y - 9^y := 
sorry

end max_value_of_3_pow_x_minus_9_pow_x_l138_138212


namespace digit_in_421st_place_l138_138711

theorem digit_in_421st_place (r : ℚ) (rep_seq : list ℕ) (h1 : r = 7 / 19)
  (h2 : rep_seq = [3, 6, 8, 4, 2, 1, 0, 5, 2, 6, 3, 1, 5, 7, 8, 9, 4, 7])
  (h3 : ∀ n, r * 10^n = fin (19 * 10^(n - 18))) :
    rep_seq.nth (421 % 18) = some 1 :=
by
  sorry

end digit_in_421st_place_l138_138711


namespace largest_possible_N_l138_138987

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138987


namespace minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l138_138861

noncomputable def f (x : ℝ) : ℝ := |x + 1| + |x - 2|

theorem minimum_value_of_f : ∀ x : ℝ, f x ≥ 3 := by
  sorry

theorem minimum_value_achieved : ∃ x : ℝ, f x = 3 := by
  sorry

theorem sum_of_squares_ge_three (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) (h : p + q + r = 3) : p^2 + q^2 + r^2 ≥ 3 := by
  sorry

end minimum_value_of_f_minimum_value_achieved_sum_of_squares_ge_three_l138_138861


namespace cos_B_value_l138_138302

-- Definitions of the sides in terms of their geometric relationship
variable (a b c : ℝ) (A B C : ℝ)
variable (h_geom_seq : b^2 = a * c)
variable (h_c_def : c = 2 * a)
variable (h_cos_rule : cos B = (a^2 + c^2 - b^2) / (2 * a * c))

-- The goal is to prove that cos B = 3/4 given the conditions.
theorem cos_B_value (a b c : ℝ) (h_geom_seq : b^2 = a * c) (h_c_def : c = 2 * a) (h_cos_rule : cos B = (a^2 + c^2 - b^2) / (2 * a * c)) : 
    cos B = 3 / 4 :=
  sorry

end cos_B_value_l138_138302


namespace smallest_possible_a_l138_138404

theorem smallest_possible_a (a b c : ℝ) 
  (h1 : (∀ x, y = a * x ^ 2 + b * x + c ↔ y = a * (x + 1/3) ^ 2 + 5/9))
  (h2 : a > 0)
  (h3 : ∃ n : ℤ, a + b + c = n) : 
  a = 1/4 :=
sorry

end smallest_possible_a_l138_138404


namespace probability_divisor_of_36_l138_138660

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138660


namespace midpoint_is_incenter_l138_138167

variables {A B C P K O1 : Type} [HasIncenter (triangle A B C)]
variables [IsoscelesTriangle A B C] [tangent S (triangle A B C) A P K]

theorem midpoint_is_incenter
  (isosceles_triangle : IsoscelesTriangle A B C)
  (circle_tangent : ∀ {S : Type}, tangent S A B P K)
  (circle_inscribed : ∀ {O : Type}, inscribed (circumcircle (triangle B C)) S)
  (midpoint_point : midpoint O1 P K)
  (mid_point_incenter: HasIncenter (triangle A B C)): 
  incenter O1 (triangle A B C):= 
by 
  sorry

end midpoint_is_incenter_l138_138167


namespace proof_problem_l138_138269

noncomputable def parabola_equation (p : ℝ) (hp : 0 < p) : Prop :=
∃ y₀ : ℝ, |df_dist 2 y₀ 0 p| = 3 ∧ y₀ ^ 2 = 2 * p * 2 ∧ p = 2

noncomputable def triangle_OAB_area : Prop :=
let p := 2 in
let parabola : ℝ -> ℝ := λ x, real.sqrt (4 * x) in
let line := λ x, x - 1 in
solution_ABC := intersection_points parabola line,
let x₁ := solution_ABC.fst in
let x₂ := solution_ABC.snd in
x₁ + x₂ = 6 ∧
let dist_O_line := (real.sqrt 2) / 2 in
let |AB| := x₁ + x₂ + p in
let area := 1 / 2 * |AB| * dist_O_line in
area = 2 * real.sqrt 2

theorem proof_problem : 
  ( ∀ p, p > 0 → parabola_equation p ) ∧
  triangle_OAB_area :=
begin
  split,
  { -- prove parabola equation
    intros p hp,
    sorry, -- equation proof to be filled
  },
  { -- prove triangle area
    sorry, -- area proof to be filled
  }
end

end proof_problem_l138_138269


namespace cost_of_paving_floor_l138_138723

-- Conditions
def length_of_room : ℝ := 8
def width_of_room : ℝ := 4.75
def rate_per_sq_metre : ℝ := 900

-- Statement to prove
theorem cost_of_paving_floor : (length_of_room * width_of_room * rate_per_sq_metre) = 34200 :=
by
  sorry

end cost_of_paving_floor_l138_138723


namespace max_value_of_3_pow_x_minus_9_pow_x_l138_138214

open Real

theorem max_value_of_3_pow_x_minus_9_pow_x :
  ∃ (x : ℝ), ∀ y : ℝ, 3^x - 9^x ≤ 3^y - 9^y := 
sorry

end max_value_of_3_pow_x_minus_9_pow_x_l138_138214


namespace perimeter_hexagon_leq_l138_138386

-- Definitions of the given conditions:
variables {a b c : ℝ} (r s T : ℝ)

-- Given:
-- s is the semiperimeter of the triangle ABC
def semiperimeter := (a + b + c) / 2

-- T is the area of the triangle ABC
def area (a b c : ℝ) : ℝ := T

-- r is the radius of the incircle
noncomputable def inradius (a b c : ℝ) : ℝ := 2 * T / (a + b + c)

-- Problem statement:
theorem perimeter_hexagon_leq (a b c : ℝ) (r s T : ℝ) (h1 : r = 2 * T / (a + b + c)) (h2 : s = (a + b + c) / 2) (h3 : T = (a * b * c) / (s - a) / (s - b) / (s - c)) : 
  2 * ((a + b + c) - (a^2 + b^2 + c^2) / s) ≤ 2 * (a*b + b*c + c*a) / (a + b + c) :=
by
  sorry

end perimeter_hexagon_leq_l138_138386


namespace find_chemistry_marks_l138_138174

theorem find_chemistry_marks 
    (marks_english : ℕ := 70)
    (marks_math : ℕ := 63)
    (marks_physics : ℕ := 80)
    (marks_biology : ℕ := 65)
    (average_marks : ℚ := 68.2) :
    ∃ (marks_chemistry : ℕ), 
      (marks_english + marks_math + marks_physics + marks_biology + marks_chemistry) = 5 * average_marks 
      → marks_chemistry = 63 :=
by
  sorry

end find_chemistry_marks_l138_138174


namespace probability_factor_36_l138_138546

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138546


namespace product_closest_to_l138_138051

theorem product_closest_to :
  let expr := (2.1 * (30.3 + 0.13))
  abs (expr - 63) < min (abs (expr - 55)) (min (abs (expr - 60)) (min (abs (expr - 65)) (abs (expr - 70)))) :=
sorry

end product_closest_to_l138_138051


namespace probability_factor_36_l138_138600

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138600


namespace largest_N_satisfying_cond_l138_138995

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138995


namespace number_of_common_tangents_of_two_circles_l138_138259

theorem number_of_common_tangents_of_two_circles 
  (x y : ℝ)
  (circle1 : x^2 + y^2 = 1)
  (circle2 : x^2 + y^2 - 6 * x - 8 * y + 9 = 0) :
  ∃ n : ℕ, n = 3 :=
by
  sorry

end number_of_common_tangents_of_two_circles_l138_138259


namespace uniform_convergence_l138_138960

-- Define the non-decreasing property and the limit condition for F_n
variable {F : ℕ → ℝ → ℝ}
variable {n : ℕ}
variable {t : ℝ}

-- Conditions
axiom F_n_non_decreasing (n : ℕ) (t₁ t₂ : ℝ) (h₁ : t₁ ∈ set.Icc (0 : ℝ) 1) (h₂ : t₂ ∈ set.Icc (0 : ℝ) 1) :
  t₁ ≤ t₂ → F n t₁ ≤ F n t₂

axiom F_n_converges_to_t (t : ℚ) (ht : t ∈ set.Icc (0 : ℚ) 1) :
  filter.tendsto (λ n, F n t) filter.at_top (nhds t)

-- The Lean 4 proof statement
theorem uniform_convergence (F : ℕ → ℝ → ℝ)
  (h1 : ∀ n, ∀ t₁ t₂ ∈ Icc (0:ℝ) 1, t₁ ≤ t₂ → F n t₁ ≤ F n t₂)
  (h2 : ∀ (t : ℚ) (ht : t ∈ Icc (0:ℚ) 1), filter.tendsto (λ n, F n t) filter.at_top (nhds t)) :
  filter.tendsto (λ n, supr (λ t : ℝ, abs (F n t - t))) filter.at_top (nhds 0) :=
by
  sorry

end uniform_convergence_l138_138960


namespace probability_factor_of_36_is_1_over_4_l138_138504

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138504


namespace probability_factor_36_l138_138589

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138589


namespace probability_factor_of_36_l138_138568

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138568


namespace probability_factor_36_l138_138594

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138594


namespace probability_divisor_of_36_is_one_fourth_l138_138464

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138464


namespace sum_four_cells_zero_l138_138192

theorem sum_four_cells_zero (T : ℕ → ℕ → ℤ) (n m : ℕ)
  (H : ∀ i j, T i j ∈ {-1, 0, 1}) (H0 : (∑ i in Finset.range n, ∑ j in Finset.range m, T i j) = 0)
  (Hn : n = 2005) (Hm : m = 2005) :
  ∃ r1 r2 c1 c2, r1 ≠ r2 ∧ c1 ≠ c2 ∧ T r1 c1 + T r1 c2 + T r2 c1 + T r2 c2 = 0 := 
sorry

end sum_four_cells_zero_l138_138192


namespace machine_X_takes_2_days_longer_l138_138389

-- Define the rates of machines X and Y as variables
variables {W : ℝ}

-- Define the conditions
def rate_X : ℝ := W / 6
def combined_rate_X_Y : ℝ := 5 * W / 12
def rate_Y : ℝ := combined_rate_X_Y - rate_X

-- Define the time it takes for machine Y to produce W widgets
def time_Y : ℝ := W / rate_Y

-- Define the time difference
def time_difference : ℝ := 6 - time_Y

-- The theorem to prove
theorem machine_X_takes_2_days_longer (h1 : rate_X = W / 6)
  (h2 : combined_rate_X_Y = 5 * W / 12) :
  time_difference = 2 :=
by
  -- The proof is omitted for now
  sorry

end machine_X_takes_2_days_longer_l138_138389


namespace quadratic_discriminant_nonnegative_integer_roots_positive_m_l138_138844

theorem quadratic_discriminant_nonnegative (m : ℝ) : 
  let Δ := (3 * m + 2)^2 - 4 * m * 6
  in Δ ≥ 0 :=
by
  let Δ := (3 * m + 2)^2 - 4 * m * 6
  calc
    Δ = (3 * m - 2)^2 : by sorry
    ... ≥ 0 : by nlinarith

theorem integer_roots_positive_m :
  (∀ x : ℝ, (m : ℤ) > 0 → ((m : ℝ) * x^2 - (3 * m + 2) * x + 6 = 0 → x ∈ ℤ)) → (m = 1 ∨ m = 2) :=
by sorry

end quadratic_discriminant_nonnegative_integer_roots_positive_m_l138_138844


namespace range_of_m_l138_138039

theorem range_of_m (m : ℝ) : (∀ x : ℝ, x > 0 → (y = 1 - 3 * m / x) → y > 0) ↔ (m > 1 / 3) :=
sorry

end range_of_m_l138_138039


namespace largest_N_satisfying_cond_l138_138996

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138996


namespace probability_factor_of_36_l138_138698

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138698


namespace probability_factor_of_36_l138_138477

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138477


namespace discount_difference_is_correct_l138_138132

-- Define the successive discounts in percentage
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.15
def discount3 : ℝ := 0.10

-- Define the store's claimed discount
def claimed_discount : ℝ := 0.45

-- Calculate the true discount
def true_discount : ℝ := 1 - ((1 - discount1) * (1 - discount2) * (1 - discount3))

-- Calculate the difference between the true discount and the claimed discount
def discount_difference : ℝ := claimed_discount - true_discount

-- State the theorem to be proved
theorem discount_difference_is_correct : discount_difference = 2.375 / 100 := by
  sorry

end discount_difference_is_correct_l138_138132


namespace iron_conducts_electricity_reasoning_type_l138_138143

variable (Metal : Type) (Iron : Metal)
variable conducts_electricity : Metal → Prop
variable is_metal : Metal → Prop

axiom all_metals_conducts_electricity : ∀ (m : Metal), is_metal m → conducts_electricity m
axiom iron_is_metal : is_metal Iron

theorem iron_conducts_electricity : conducts_electricity Iron := by
  apply all_metals_conducts_electricity Iron
  apply iron_is_metal

/-- The type of reasoning used in this proof is deductive reasoning. -/
theorem reasoning_type : String := "deductive reasoning"

end iron_conducts_electricity_reasoning_type_l138_138143


namespace fraction_simplifies_to_cot2_l138_138396

noncomputable def simplify_fraction (x : ℝ) : ℝ :=
  (1 + Math.sin x + Math.cos x + Math.sin (2 * x)) / 
  (1 + Math.sin x - Math.cos x + Math.cos (2 * x))

theorem fraction_simplifies_to_cot2 (x : ℝ) :
  simplify_fraction x = Math.cot (x / 2) ^ 2 := by
  sorry

end fraction_simplifies_to_cot2_l138_138396


namespace probability_factor_of_36_l138_138694

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138694


namespace probability_factor_of_36_l138_138525

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138525


namespace probability_factor_of_36_is_1_over_4_l138_138503

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138503


namespace segments_equal_l138_138347

variable (ABC : Type) [triangle ABC] -- Let ABC be an acute-angled, non-isosceles triangle
variable (H : Point ABC) (orthocenter : orthocenter H) -- H is the orthocenter of ΔABC
variable (M : Point ABC) (midpointAB : midpoint M = (AB : Line ABC)) -- M is the midpoint of AB
variable (bisector_w : bisector (B : Angle ABC)) (w : Line bisecting (angle : ∠ACB)) -- w is the bisector of ∠ACB
variable (S : Point ABC) (intersectionS : intersecting (perpendicular_bisector AB) w S) -- S is the intersection of the perpendicular bisector of AB with w
variable (F : Point ABC) (footF : foot_of_perpendicular_from F H w) -- F is the foot of the perpendicular from H on w

theorem segments_equal (MS : segment M S) (MF : segment M F) : 
  MS = MF := 
sorry

end segments_equal_l138_138347


namespace incorrect_statements_count_l138_138714

theorem incorrect_statements_count :
  let a b : ℤ
  let statements := [
    "Statement 1: -a is always negative",
    "Statement 2: If |a|=|b|, then a=b",
    "Statement 3: A rational number is either an integer or a fraction",
    "Statement 4: A rational number is either positive or negative"
  ]
  -- Incorrect analyses:
  -- Statement 1: Incorrect since -a is not always negative
  -- Statement 2: Incorrect since |a|=|b| does not imply a=b but a=b or a=-b
  -- Statement 3: Correct since rational numbers can be written as fractions which include integers
  -- Statement 4: Incorrect as rational numbers can also be zero which is neither positive nor negative

  -- There are exactly 3 incorrect statements.
  3 = 
  (
    (if (-a < 0 = false) then 1 else 0) +
    (if (abs a = abs b → a = b = false) then 1 else 0) +
    (if (∀ q r : ℚ, r ≠ 0 → q / r = p / q) then 0 else 1) +
    (if (rational_number p q ∧ p ≠ 0 ∧ q ≠ 0 → ((q / p > 0 ∨ q / p < 0))) then 0 else 1)
  )
  :=
  sorry

end incorrect_statements_count_l138_138714


namespace problem_equivalent_l138_138241

noncomputable def f (x : ℝ) : ℝ := Real.log (x + 2) / Real.log 4 - 1

theorem problem_equivalent : 
  (∀ x : ℝ, f x = f (-x)) →
  (∀ x : ℝ, f x = Real.log (x + 2) / Real.log 4 - 1) →
  {x : ℝ | f (x - 2) > 0} = {x | x < 0 ∨ x > 4} :=
by
  intro h_even h_def
  sorry

end problem_equivalent_l138_138241


namespace f_monotonically_increasing_f_at_alpha_l138_138881

noncomputable section

def vector_a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sqrt 3 * Real.sin x)
def vector_b (x : ℝ) : ℝ × ℝ := (Real.cos x, -2 * Real.cos x)
def f (x : ℝ) : ℝ := (vector_a x).1 * (vector_b x).1 + (vector_a x).2 * (vector_b x).2

def is_monotonically_increasing (x : ℝ) : Prop :=
  ∃ (k : ℤ), k * Real.pi - (2 * Real.pi / 3) ≤ x ∧ x ≤ k * Real.pi - (Real.pi / 6)

-- Proof that f(x) is monotonically increasing in the given intervals
theorem f_monotonically_increasing : 
  ∀ x : ℝ, (is_monotonically_increasing x) → 
  (∃ k : ℤ, ∀ y : ℝ, k * Real.pi - (2 * Real.pi / 3) ≤ y ∧ y ≤ k * Real.pi - (Real.pi / 6) → f(x) ≤ f(y)) :=
by 
  sorry

def alpha : ℝ := Real.arctan (Real.sqrt 2)

-- Proof of the value of f(alpha) given tan(alpha) = sqrt(2)
theorem f_at_alpha : f(alpha) = (2 - 2 * Real.sqrt 6) / 3 :=
by 
  sorry

end f_monotonically_increasing_f_at_alpha_l138_138881


namespace factor_probability_36_l138_138606

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138606


namespace probability_factor_of_36_l138_138569

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138569


namespace largest_N_satisfying_cond_l138_138997

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138997


namespace probability_factor_of_36_l138_138699

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138699


namespace two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l138_138067

variable (n : ℕ) (F : ℕ → ℕ) (p : ℕ)

-- Condition: F_n = 2^{2^n} + 1
def F_n (n : ℕ) : ℕ := 2^(2^n) + 1

-- Assuming n >= 2
def n_ge_two (n : ℕ) : Prop := n ≥ 2

-- Assuming p is a prime factor of F_n
def prime_factor_of_F_n (p : ℕ) (n : ℕ) : Prop := p ∣ (F_n n) ∧ Prime p

-- Part a: 2 is a quadratic residue modulo p
theorem two_quadratic_residue_mod_p (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  ∃ x : ℕ, x^2 ≡ 2 [MOD p] := sorry

-- Part b: p ≡ 1 (mod 2^(n+2))
theorem p_congruent_one_mod_2_pow_n_plus_two (n : ℕ) (p : ℕ) (hn : n_ge_two n) (hp : prime_factor_of_F_n p n) :
  p ≡ 1 [MOD 2^(n+2)] := sorry

end two_quadratic_residue_mod_p_p_congruent_one_mod_2_pow_n_plus_two_l138_138067


namespace probability_factor_36_l138_138516

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138516


namespace part1_part2_l138_138837

-- Define x and y
def x : ℝ := 1 / (3 + 2 * real.sqrt 2)
def y : ℝ := 1 / (3 - 2 * real.sqrt 2)

-- Prove x^2 + y^2 + xy = 35
theorem part1 : x^2 + y^2 + x * y = 35 :=
sorry

-- Define the decimal parts of x and y
def m : ℝ := 3 - 2 * real.sqrt 2
def n : ℝ := 2 * real.sqrt 2 - 2

-- Prove (m+n)^2023 - ∛((m-n)^3) = 0
theorem part2 : (m + n)^2023 - real.cbrt ((m - n)^3) = 0 :=
sorry

end part1_part2_l138_138837


namespace rectangle_area_prob_greater_than_32_l138_138757

theorem rectangle_area_prob_greater_than_32 :
  let AC BC : ℝ := 0
  let AB : ℝ := 12
  ∃ x : ℝ, (0 < x ∧ x < AB) →
  let rect_area (x : ℝ) : ℝ := x * (AB - x)
  let prob := (number of x ∈ Ioo (4:ℝ) 8) / (number of x ∈ Icc (0:ℝ) (12:ℝ))
  prob = (1 / 3)
:= begin
  sorry
end

end rectangle_area_prob_greater_than_32_l138_138757


namespace orthocenter_PQR_l138_138913

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end orthocenter_PQR_l138_138913


namespace statement_parametric_and_polar_equation_l138_138326

/-
Define the parametric equations of the conic curve
-/
def conic_curve (θ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos θ, Real.sqrt 3 * Real.sin θ)

/-
Define the fixed point A
-/
def point_A : ℝ × ℝ := (0, Real.sqrt 3)

/-
Define the focal points F₁ and F₂
-/
def focal_point_F1 : ℝ × ℝ := (-1, 0)
def focal_point_F2 : ℝ × ℝ := (1, 0)

/-
Parametric equation of the line L
-/
noncomputable def line_L (t : ℝ) : ℝ × ℝ :=
  let (cos_30, sin_30) := (Real.sqrt 3 / 2, 1 / 2)
  (-1 + cos_30 * t, sin_30 * t)

/-
Polar equation of the line AF₂
-/
def polar_eq (ρ φ : ℝ) : Prop :=
  (Real.sqrt 3 * ρ * Real.cos φ + ρ * Real.sin φ - Real.sqrt 3 = 0)

/-
Theorem statement combining all parts
-/
theorem parametric_and_polar_equation :
  ∀ t : ℝ, line_L t = (-1 + (Real.sqrt 3 / 2) * t, (1 / 2) * t) ∧
  ∀ ρ φ : ℝ, polar_eq ρ φ ↔ (Real.sqrt 3 * ρ * Real.cos φ + ρ * Real.sin φ - Real.sqrt 3 = 0) := by
  sorry

end statement_parametric_and_polar_equation_l138_138326


namespace probability_factor_36_l138_138515

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138515


namespace sum_of_all_possible_values_of_M_l138_138421

-- Given conditions
-- M * (M - 8) = -7
-- We need to prove that the sum of all possible values of M is 8

theorem sum_of_all_possible_values_of_M : 
  ∃ M1 M2 : ℝ, (M1 * (M1 - 8) = -7) ∧ (M2 * (M2 - 8) = -7) ∧ (M1 + M2 = 8) :=
by
  sorry

end sum_of_all_possible_values_of_M_l138_138421


namespace solve_for_x_l138_138024

theorem solve_for_x : ∃ x : ℝ, 5 * x + 9 * x = 570 - 12 * (x - 5) ∧ x = 315 / 13 :=
by
  sorry

end solve_for_x_l138_138024


namespace quadratic_roots_new_equation_l138_138235

theorem quadratic_roots_new_equation (a b c x1 x2 : ℝ) 
  (h1 : a ≠ 0) 
  (h2 : x1 + x2 = -b / a) 
  (h3 : x1 * x2 = c / a) : 
  ∃ (a' b' c' : ℝ), a' * x^2 + b' * x + c' = 0 ∧ a' = a^2 ∧ b' = 3 * a * b ∧ c' = 2 * b^2 + a * c :=
sorry

end quadratic_roots_new_equation_l138_138235


namespace complex_point_in_fourth_quadrant_l138_138888

theorem complex_point_in_fourth_quadrant (a b : ℝ) :
  (a^2 - 4*a + 5 > 0) ∧ (-b^2 + 2*b - 6 < 0) →
  let x := a^2 - 4*a + 5
  in let y := -b^2 + 2*b - 6
  in (x, y).1 > 0 ∧ (x, y).2 < 0 :=
by
  intros h
  have x_pos : a^2 - 4*a + 5 > 0 := h.1
  have y_neg : -b^2 + 2*b - 6 < 0 := h.2
  exact And.intro x_pos y_neg

end complex_point_in_fourth_quadrant_l138_138888


namespace probability_factor_36_l138_138555

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138555


namespace construction_delay_without_additional_men_l138_138124

-- Definitions for the conditions
def initial_work_force := 100
def additional_work_force := 100
def total_days := 100
def initial_period := 20
def work_done := initial_work_force * total_days
def remaining_work := work_done - (initial_work_force * initial_period)

-- Question as a theorem statement
theorem construction_delay_without_additional_men :
  let days_remaining_work := remaining_work / initial_work_force in
  let total_construction_days := initial_period + days_remaining_work in
  (total_construction_days - total_days) = 80 :=
sorry

end construction_delay_without_additional_men_l138_138124


namespace probability_divisor_of_36_is_one_fourth_l138_138473

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138473


namespace coordinate_of_equidistant_point_l138_138068

def euclidean_distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def point_on_line (x : ℝ) : ℝ × ℝ :=
  (x, -x)

def point_equidistant (p A B : ℝ × ℝ) : Prop :=
  euclidean_distance p A = euclidean_distance p B

theorem coordinate_of_equidistant_point :
  point_on_line (-8) = (-8, 8) ∧ point_equidistant (-8, 8) (-2, 0) (2, 6) :=
by
  sorry

end coordinate_of_equidistant_point_l138_138068


namespace factor_probability_l138_138626

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138626


namespace no_line_with_30_deg_angle_l138_138753

theorem no_line_with_30_deg_angle (l : Line) (p : Plane) (h : angle l p = 45) :
  ∀ l' ∈ p, angle l l' ≠ 30 :=
by
  sorry

end no_line_with_30_deg_angle_l138_138753


namespace exists_unique_perpendicular_l138_138929

theorem exists_unique_perpendicular (l : set Point) (hline : is_line l) (P : Point) (hP : P ∉ l) :
  ∃! m : set Point, is_line m ∧ (∃ Q : Point, Q ∈ m ∧ Q ≠ P ∧ Q ∈ l) ∧ perpendicular m l := 
sorry

end exists_unique_perpendicular_l138_138929


namespace line_intersects_x_axis_at_specified_point_l138_138118

theorem line_intersects_x_axis_at_specified_point :
  ∀ (P Q : ℝ × ℝ), (P = (4, 10) ∧ Q = (-2, 8)) →
  ∃ R : ℝ × ℝ, R = (-26, 0) ∧ (∃ m b : ℝ, m = (Q.snd - P.snd) / (Q.fst - P.fst) ∧ b = P.snd - m * P.fst ∧ R.snd = 0 ∧ R.fst = (0 - b) / m) :=
begin
  intros P Q h,
  cases h with hP hQ,
  use (-26, 0),
  split,
  { refl },
  { use ((Q.snd - P.snd) / (Q.fst - P.fst)),
    use (P.snd - ((Q.snd - P.snd) / (Q.fst - P.fst)) * P.fst),
    have m_def : ((Q.snd - P.snd) / (Q.fst - P.fst)) = (8 - 10) / (-2 - 4), by rw [hP, hQ],
    have b_def : (P.snd - ((Q.snd - P.snd) / (Q.fst - P.fst)) * P.fst) = 10 - ((8 - 10) / (-2 - 4)) * 4, by rw [hP, hQ],
    split,
    { rw m_def, norm_num },
    split,
    { rw b_def, norm_num },
    split,
    { refl },
    { rw [b_def, m_def],
      norm_num } }
end

end line_intersects_x_axis_at_specified_point_l138_138118


namespace milk_removal_replacement_l138_138139

theorem milk_removal_replacement (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 45) :
  (45 - x) * (45 - x) / 45 = 28.8 → x = 9 :=
by
  -- skipping the proof for now
  sorry

end milk_removal_replacement_l138_138139


namespace sum_non_prime_21_to_29_equals_173_l138_138725

open Nat

def is_non_prime (n : ℕ) : Prop :=
  ¬ is_prime n

def sum_non_prime_21_to_29 : ℕ :=
  (21 + 22 + 24 + 25 + 26 + 27 + 28)

theorem sum_non_prime_21_to_29_equals_173 : sum_non_prime_21_to_29 = 173 := by
  sorry

end sum_non_prime_21_to_29_equals_173_l138_138725


namespace hypercube_diagonals_count_l138_138116

def isDiagonal (v1 v2 : ℕ) (edges : set (ℕ × ℕ)) : Prop :=
  v1 ≠ v2 ∧ (v1, v2) ∉ edges ∧ (v2, v1) ∉ edges

def hypercube_diagonals (vertices edges : ℕ → ℕ → Prop) : ℕ :=
  (vertices.card * (vertices.card - 1) / 2) - edges.card

theorem hypercube_diagonals_count (vertices edges : set ℕ) (h_vert : vertices.card = 16) (h_edge : edges.card = 32) :
  hypercube_diagonals vertices edges = 408 := 
by
  sorry

end hypercube_diagonals_count_l138_138116


namespace probability_factor_of_36_l138_138649

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138649


namespace probability_divisor_of_36_is_one_fourth_l138_138463

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138463


namespace domain_of_f_log_l138_138251

theorem domain_of_f_log (f : ℝ → ℝ) :
  (∀ x, (-1 < 2 * x - 1 ∧ 2 * x - 1 ≤ 1) → domain (λ x, f (2 * x - 1)) = set.Icc (-1 : ℝ) 1) →
  domain (λ x, f (Real.logb (1 / 2 : ℝ) x)) = set.Icc (1 / 2 : ℝ) 8 :=
by
  sorry

end domain_of_f_log_l138_138251


namespace sum_of_angles_WYZ_XYZ_l138_138744

theorem sum_of_angles_WYZ_XYZ (W X Y Z : Point) (circumcircle : Circle)
    (angle_WXY : ℝ) (angle_YZW : ℝ)
    (h_WXY : angle_WXY = 50) (h_YZW : angle_YZW = 20)
    (is_circumscribed : IsCircumscribed quadrilateral circumcircle) :
    angle_WYZ + angle_XYZ = 110 :=
by
  sorry

end sum_of_angles_WYZ_XYZ_l138_138744


namespace probability_factor_of_36_l138_138695

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138695


namespace mike_travel_miles_l138_138368

theorem mike_travel_miles
  (toll_fees_mike : ℝ) (toll_fees_annie : ℝ) (mike_start_fee : ℝ) 
  (annie_start_fee : ℝ) (mike_per_mile : ℝ) (annie_per_mile : ℝ) 
  (annie_travel_time : ℝ) (annie_speed : ℝ) (mike_cost : ℝ) 
  (annie_cost : ℝ) 
  (h_mike_cost_eq : mike_cost = mike_start_fee + toll_fees_mike + mike_per_mile * 36)
  (h_annie_cost_eq : annie_cost = annie_start_fee + toll_fees_annie + annie_per_mile * annie_speed * annie_travel_time)
  (h_equal_costs : mike_cost = annie_cost)
  : 36 = 36 :=
by 
  sorry

end mike_travel_miles_l138_138368


namespace value_at_2013_l138_138254

noncomputable def f : ℝ → ℝ := sorry

axiom odd_function : ∀ x : ℝ, f x = -f (-x)
axiom periodic_5 : ∀ x : ℝ, f (x + 5) ≥ f x
axiom periodic_1 : ∀ x : ℝ, f (x + 1) ≤ f x

theorem value_at_2013 : f 2013 = 0 :=
by
  -- Proof goes here
  sorry

end value_at_2013_l138_138254


namespace probability_divisor_of_36_is_one_fourth_l138_138474

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138474


namespace not_multiple_of_121_l138_138381

theorem not_multiple_of_121 (n : ℤ) : ¬ ∃ k : ℤ, n^2 + 2*n + 12 = 121*k := 
sorry

end not_multiple_of_121_l138_138381


namespace factor_probability_l138_138624

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138624


namespace polynomial_no_negative_roots_l138_138384

theorem polynomial_no_negative_roots (x : ℝ) (h : x < 0) : x^4 - 4 * x^3 - 6 * x^2 - 3 * x + 9 ≠ 0 := 
by 
  sorry

end polynomial_no_negative_roots_l138_138384


namespace orthocenter_PQR_is_correct_l138_138910

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end orthocenter_PQR_is_correct_l138_138910


namespace students_per_group_l138_138438

theorem students_per_group (n m : ℕ) (h_n : n = 36) (h_m : m = 9) : 
  (n - m) / 3 = 9 := 
by
  sorry

end students_per_group_l138_138438


namespace ratio_of_ages_l138_138736

theorem ratio_of_ages (S F : Nat) 
  (h1 : F = 3 * S) 
  (h2 : (S + 6) + (F + 6) = 156) : 
  (F + 6) / (S + 6) = 19 / 7 := 
by 
  sorry

end ratio_of_ages_l138_138736


namespace mod_remainder_l138_138828

theorem mod_remainder :
  ((85^70 + 19^32)^16) % 21 = 16 := by
  -- Given conditions
  have h1 : 85^70 % 21 = 1 := sorry
  have h2 : 19^32 % 21 = 4 := sorry
  -- Conclusion
  sorry

end mod_remainder_l138_138828


namespace probability_factor_of_36_l138_138482

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138482


namespace carlos_distance_l138_138791

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem carlos_distance :
  let a := (3, -7)
  let b := (1, 1)
  let c := (-6, 3)
  distance a b + distance b c = 2 * real.sqrt 17 + real.sqrt 53 :=
by 
  let a := (3, -7)
  let b := (1, 1)
  let c := (-6, 3)
  have h1 : distance a b = 2 * real.sqrt 17 := sorry
  have h2 : distance b c = real.sqrt 53 := sorry
  exact calc
    distance a b + distance b c = 2 * real.sqrt 17 + real.sqrt 53 : by rw [h1, h2]

end carlos_distance_l138_138791


namespace probability_factor_36_l138_138513

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138513


namespace probability_factor_of_36_l138_138672

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138672


namespace positive_difference_l138_138055

variable (x y : ℕ)

theorem positive_difference (h1 : x + y = 40) (h2 : 3 * y - 2 * x = 10) : |x - y| = 4 := by
  sorry

end positive_difference_l138_138055


namespace product_of_extremes_l138_138795

-- Define the variables and conditions described in the problem
variables (a1 a2 a3 a4 a5 a6 : ℝ)
variable (m : ℝ) -- the smallest number
variable (M : ℝ) -- the largest number

-- Condition 1: When the largest number is removed, the average decreases by 1
def avg_largest_removed (a1 a2 a3 a4 a5 a6 : ℝ) : Prop :=
  (a1 + a2 + a3 + a4 + a5 + a6 - M) / 5 = ((a1 + a2 + a3 + a4 + a5 + a6) / 6) - 1

-- Condition 2: When the smallest number is removed, the average increases by 1
def avg_smallest_removed (a1 a2 a3 a4 a5 a6 : ℝ) : Prop :=
  (a1 + a2 + a3 + a4 + a5 + a6 - m) / 5 = ((a1 + a2 + a3 + a4 + a5 + a6) / 6) + 1
    
-- Condition 3: When both the largest and smallest numbers are removed, the average of the remaining four numbers is 20
def avg_four_numbers (a1 a2 a3 a4 a5 a6 : ℝ) : Prop :=
  (a1 + a2 + a3 + a4 + a5 + a6 - m - M) / 4 = 20

-- The final statement to prove the product of the smallest and the largest numbers
theorem product_of_extremes (a1 a2 a3 a4 a5 a6 : ℝ) (m : ℝ) (M : ℝ)
  (h1 : avg_largest_removed a1 a2 a3 a4 a5 a6)
  (h2 : avg_smallest_removed a1 a2 a3 a4 a5 a6)
  (h3 : avg_four_numbers a1 a2 a3 a4 a5 a6) :
  m * M = 375 := 
  sorry

end product_of_extremes_l138_138795


namespace discount_percentage_is_25_l138_138343

def piano_cost := 500
def lessons_count := 20
def lesson_price := 40
def total_paid := 1100

def lessons_cost := lessons_count * lesson_price
def total_cost := piano_cost + lessons_cost
def discount_amount := total_cost - total_paid
def discount_percentage := (discount_amount / lessons_cost) * 100

theorem discount_percentage_is_25 : discount_percentage = 25 := by
  sorry

end discount_percentage_is_25_l138_138343


namespace factor_probability_l138_138623

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138623


namespace probability_factor_36_l138_138522

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138522


namespace initial_value_is_76800_l138_138191

noncomputable def find_initial_value
  (after_two_years : ℝ)
  (growth_rate : ℝ)
  (condition : (after_two_years * (growth_rate ^ 2) = 97200)) : ℝ :=
  (after_two_years * 64) / 81

theorem initial_value_is_76800 :
  ∃ (P : ℝ),
    (let growth_rate := 9 / 8 in
     let after_two_years := growth_rate * growth_rate * P in
     after_two_years = 97200) →
    P = 76800 :=
begin
  use 76800,
  intros h,
  calc
    _ = 97200 : h
    ... = ((9.0 / 8.0) ^ 2 * 76800) / 1 : by sorry,
end

end initial_value_is_76800_l138_138191


namespace reciprocal_of_neg_5_l138_138423

theorem reciprocal_of_neg_5 : (∃ r : ℚ, -5 * r = 1) ∧ r = -1 / 5 :=
by sorry

end reciprocal_of_neg_5_l138_138423


namespace probability_divisor_of_36_is_one_fourth_l138_138470

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138470


namespace seq_pos_int_seq_perf_square_l138_138764

def seq (a : ℕ → ℚ) :=
∀ n : ℕ, a (n + 1) = (7 * a n + real.sqrt (45 * (a n)^2 - 36)) / 2

theorem seq_pos_int (a : ℕ → ℚ) (h0 : a 0 = 1) (h : seq a) :
∀ n : ℕ, a n ∈ ℕ :=
sorry

theorem seq_perf_square (a : ℕ → ℚ) (h0 : a 0 = 1) (h : seq a) :
∀ n : ℕ, ∃ k : ℕ, a n * a (n + 1) - 1 = k^2 :=
sorry

end seq_pos_int_seq_perf_square_l138_138764


namespace smallest_n_l138_138152

-- Define the conditions based on the problem statement

variables (r w b g y : ℕ)

-- Define the total number of marbles
def n : ℕ := r + w + b + g + y

-- Define the binomial coefficients for each event
def event_a : ℕ := r.choose 5
def event_b : ℕ := w * r.choose 4
def event_c : ℕ := w * b * r.choose 3
def event_d : ℕ := w * b * g * r.choose 2
def event_e : ℕ := w * b * g * y * r

-- Assuming the events are equally likely
axiom equal_probability : 
  event_a = event_b ∧
  event_a = event_c ∧
  event_a = event_d ∧
  event_a = event_e

-- Aim: Prove that the smallest value of n such that all conditions hold and n is divisible by 7 is 28
theorem smallest_n : ∃ n, n = r + w + b + g + y ∧ n % 7 = 0 ∧ ∀ n' < n, n' ≠ r + w + b + g + y := 
begin
  use 28,
  split,
  sorry, -- provide proof that n = 28 matches the conditions
  split,
  norm_num,
  intros n' hn hnval,
  sorry -- provide proof that there is no smaller n that matches the conditions
end

end smallest_n_l138_138152


namespace allocation_schemes_4_students_3_universities_l138_138782

def num_allocation_schemes (n : ℕ) (k : ℕ) : ℕ :=
if h : k ≤ n then by
  have : ∀ {a b : ℕ}, k = b + a → b * fact a = fact k := sorry
  exact 36
else 0

theorem allocation_schemes_4_students_3_universities : 
  num_allocation_schemes 4 3 = 36 :=
sorry

end allocation_schemes_4_students_3_universities_l138_138782


namespace frame_dimension_ratio_l138_138756

theorem frame_dimension_ratio (W H x : ℕ) (h1 : W = 20) (h2 : H = 30) (h3 : 2 * (W + 2 * x) * (H + 6 * x) - W * H = 2 * (W * H)) :
  (W + 2 * x) / (H + 6 * x) = 1/2 :=
by sorry

end frame_dimension_ratio_l138_138756


namespace projection_of_b_onto_a_l138_138877

variables (a b : ℝ ^ 3) (ha : ‖a‖ = 2) (hb : ‖b‖ = 3) (hab : dot_product a b = 1)

theorem projection_of_b_onto_a :
  (‖b‖ * (dot_product a b / (‖a‖ * ‖b‖)) * (a / ‖a‖)) = (1/4) • a :=
by sorry

end projection_of_b_onto_a_l138_138877


namespace probability_factor_of_36_l138_138647

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138647


namespace no_solution_impl_no_solution_PP_qq_l138_138962

theorem no_solution_impl_no_solution_PP_qq {P Q : ℝ → ℝ}
  (H1 : ∀ x : ℝ, P(Q(x)) = Q(P(x)))
  (H2 : ∀ x : ℝ, P(x) ≠ Q(x)) :
  ∀ x : ℝ, P(P(x)) ≠ Q(Q(x)) :=
by
  sorry

end no_solution_impl_no_solution_PP_qq_l138_138962


namespace syllogism_minor_premise_l138_138148

theorem syllogism_minor_premise:
  (∀ a > 1, ∃ f : ℝ → ℝ, (∀ x y : ℝ, x < y → f x < f y) ∧ (f = λ x, a ^ x)) →
  (∃ g : ℝ → ℝ, (g = λ x, 2 ^ x) ∧ ∀ x y : ℝ, x < y → g x < g y) :=
by sorry

end syllogism_minor_premise_l138_138148


namespace largest_possible_N_l138_138988

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138988


namespace polar_to_cartesian_equation_l138_138924

noncomputable def polar_to_cartesian (rho θ : ℝ) :=
  (rho * cos θ, rho * sin θ)

theorem polar_to_cartesian_equation (ρ θ : ℝ) (h : ρ * sin (θ - π / 4) = √2) :
  let (x, y) := polar_to_cartesian ρ θ in
  x - y + 2 = 0 := 
by sorry

end polar_to_cartesian_equation_l138_138924


namespace cone_lateral_to_base_area_ratio_l138_138892

/-- Given that the lateral surface of a cone is unfolded to create a sector with a central angle of 90°, 
the ratio of the lateral surface area to the base area of the cone is 4:1. -/
theorem cone_lateral_to_base_area_ratio (r : ℝ) (h_r_pos : r > 0) :
  let R := 4 * r,
      A_base := π * r^2,
      A_lateral := 4 * π * r^2
  in 4 * π * r^2 / (π * r^2) = 4 :=
by
  sorry

end cone_lateral_to_base_area_ratio_l138_138892


namespace factor_probability_36_l138_138608

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138608


namespace ratio_MP_over_AB_l138_138449

noncomputable def isosceles_triangle (A B C : Type) [IsLinearOrderedField ℚ]
  [∀ x y : ℚ, decidable (x < y)] : Prop :=
  ∃ (α β γ : ℚ), A = α ∧ B = β ∧ C = γ ∧ α = β

noncomputable def similar_triangles (A B C P Q R : Type) [IsLinearOrderedField ℚ]
  [∀ x y : ℚ, decidable (x < y)] : Prop :=
  ∃ (k : ℚ), P = k * A ∧ Q = k * B ∧ R = k * C

noncomputable def positioned_points (A B C M N P : Type) [IsLinearOrderedField ℚ]
  [∀ x y : ℚ, decidable (x < y)] : Prop :=
  ∃ k₁ k₂ k₃ : ℚ, M = k₁ * A + k₂ * B + k₃ * C

theorem ratio_MP_over_AB
  (A B C M N P : Type) [IsLinearOrderedField ℚ] [∀ x y : ℚ, decidable (x < y)]
  (hABC : isosceles_triangle A B C) (hMNP : isosceles_triangle M N P)
  (hSim : similar_triangles A B C M N P) (hPos : positioned_points A B C M N P)
  (hRatio : (N / B - N) = 2) (hAngle : (A / B / C) = arctan 4) :
  (M / P) / (A / B) = 5 / 6 :=
sorry

end ratio_MP_over_AB_l138_138449


namespace find_a_given_constant_term_l138_138860

theorem find_a_given_constant_term :
  (∃ a : ℝ, ∀ x > 0, let term := ( ∑ r in Finset.range 7, binom 6 r * ((sqrt x)^(6-r) * ((-sqrt a)/x)^r)) in 
    (term = 60) → a = 4) :=
sorry

end find_a_given_constant_term_l138_138860


namespace max_magnitude_z3_minus_3z_minus_2_l138_138859

open Complex

theorem max_magnitude_z3_minus_3z_minus_2 (z : ℂ) (hz : abs z = 1) : 
  ∃ (t : ℝ), (t = 3 * Real.sqrt 3) ∧ ∀ (w : ℂ), abs w = 1 → abs (w^3 - 3*w - 2) ≤ t := 
sorry

end max_magnitude_z3_minus_3z_minus_2_l138_138859


namespace solve_for_A_l138_138085

noncomputable def A (x : ℝ) : ℝ :=
  ((2 * x + 5 + 4 * real.sqrt (2 * x + 1)) ^ (-1 / 2)
  + (2 * x + 5 - 4 * real.sqrt (2 * x + 1)) ^ (-1 / 2))
  / ((2 * x + 5 + 4 * real.sqrt (2 * x + 1)) ^ (-1 / 2)
  - (2 * x + 5 - 4 * real.sqrt (2 * x + 1)) ^ (-1 / 2))

theorem solve_for_A (x : ℝ) (h₁ : -1/2 < x)  :
  if h : x < 3/2 then A x = -2 / real.sqrt (2 * x + 1) 
  else A x = -real.sqrt (2 * x + 1) / 2 :=
sorry -- Proof skipped

end solve_for_A_l138_138085


namespace geometric_sequence_properties_l138_138841

section geometric_sequence

open Real

variables {a : ℕ → ℝ} (q : ℝ) (s : ℕ → ℝ)

-- Conditions
def is_geometric (a : ℕ → ℝ) (q : ℝ) := ∀ n, a (n + 1) = q * a n
def a1_a3_eq_5 := a 1 + a 3 = 5
def s4_eq_15 := s 4 = 15

-- Sum of first n terms of a geometric sequence
def sum_of_first_n_terms := ∀ n, s n = a 0 * (1 - q ^ n) / (1 - q)

-- Prove the required statements
theorem geometric_sequence_properties (h₀ : is_geometric a q) (h₁ : a1_a3_eq_5) (h₂ : s4_eq_15)
  (h₃ : sum_of_first_n_terms s q) :
  (∀ n, a n = 2 ^ (n - 1)) ∧ (∀ n, ∑ i in Finset.range n, 3 * log 2 (a i) = (3 / 2) * n ^ 2 - (3 / 2) * n) :=
begin
  sorry
end

end geometric_sequence

end geometric_sequence_properties_l138_138841


namespace probability_factor_of_36_l138_138639

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138639


namespace inverse_property_l138_138253

-- Given conditions
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)
variable (hf_injective : Function.Injective f)
variable (hf_surjective : Function.Surjective f)
variable (h_inverse : ∀ y : ℝ, f (f_inv y) = y)
variable (hf_property : ∀ x : ℝ, f (-x) + f (x) = 3)

-- The proof goal
theorem inverse_property (x : ℝ) : (f_inv (x - 1) + f_inv (4 - x)) = 0 :=
by
  sorry

end inverse_property_l138_138253


namespace probability_factor_36_l138_138597

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138597


namespace factor_probability_36_l138_138617

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138617


namespace eight_digit_numbers_with_four_nines_l138_138173

theorem eight_digit_numbers_with_four_nines :
  let invalid_count := (Nat.choose 7 4) * 9^3,
      valid_count := (Nat.choose 8 4) * 9^4
  in valid_count - invalid_count = 433755 :=
by
  sorry

end eight_digit_numbers_with_four_nines_l138_138173


namespace odd_function_range_even_function_range_l138_138252

variables {f : ℝ → ℝ} (m : ℝ)

-- Conditions
def is_monotonically_decreasing_on_0_2 (f : ℝ → ℝ) : Prop :=
  ∀ x y ∈ set.Icc 0 2, x ≤ y → f x ≥ f y

def is_odd_on_minus2_2 (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ set.Icc (-2) 2, f (-x) = -f x

def is_even_on_minus2_2 (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ set.Icc (-2) 2, f (-x) = f x

-- (1) Proof statement for odd function
theorem odd_function_range (h_odd : is_odd_on_minus2_2 f)
    (h_mono : is_monotonically_decreasing_on_0_2 f)
    (h_condition : f (1 - m) < f (3 * m)) :
    -2/3 ≤ m ∧ m < 1/4 :=
sorry

-- (2) Proof statement for even function
theorem even_function_range (h_even : is_even_on_minus2_2 f)
    (h_mono : is_monotonically_decreasing_on_0_2 f)
    (h_condition : f (1 - m) < f (3 * m)) :
    -1/2 ≤ m ∧ m < 1/4 :=
sorry

end odd_function_range_even_function_range_l138_138252


namespace enclosed_region_area_rounded_l138_138358

noncomputable def f (x : ℝ) : ℝ := 1 - real.sqrt (1 - x^2)

theorem enclosed_region_area_rounded :
  let area := (real.pi / 2 - 1) in
  real.round (area * 100) / 100 = 0.57 :=
begin
  let f : ℝ → ℝ := λ x, 1 - real.sqrt (1 - x^2),
  sorry
end

end enclosed_region_area_rounded_l138_138358


namespace candies_found_l138_138739

theorem candies_found (initial_candies : ℕ) (n : ℕ) (h1 : initial_candies = 111)
                      (h2 : n = 60) : 
  let after_lunch := 11 * n / 20
  in ∃ (found_candies : ℕ), found_candies = after_lunch / 3 ∧ found_candies = 11 :=
by
  sorry

end candies_found_l138_138739


namespace BM_BN_CM_CN_ratio_l138_138417

variables {A B C M N : Point} (c b : ℝ)
  (h_triangle : Triangle A B C)
  (h_MN_on_BC : M ∈ Line B C ∧ N ∈ Line B C)
  (h_symmetry : SymmetricWithRespectToAngleBisector A M N)

theorem BM_BN_CM_CN_ratio (h_c_eq_AC : c = distance A C)
  (h_b_eq_AB : b = distance A B) :
  distance B M * distance B N / (distance C M * distance C N) = c^2 / b^2 :=
sorry

end BM_BN_CM_CN_ratio_l138_138417


namespace probability_factor_of_36_is_1_over_4_l138_138495

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138495


namespace board_game_return_to_start_l138_138900

theorem board_game_return_to_start : 
  let primes_in_range := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
  let num_primes := 10                 -- number of primes between 2 and 30
  let num_composites := 19             -- number of composites between 2 and 30
  let steps_forward := num_primes * 2  -- total steps forward for primes
  let steps_backward := num_composites * (-3)  -- total steps backward for composites
  let net_steps := steps_forward + steps_backward  -- net steps after 30 moves
  in net_steps = -37
  → 37 steps need to be made to return to the start := sorry

end board_game_return_to_start_l138_138900


namespace probability_factor_of_36_l138_138683

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138683


namespace probability_divisor_of_36_is_one_fourth_l138_138465

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138465


namespace probability_two_white_balls_l138_138117

noncomputable def probability_of_two_white_balls (total_balls white_balls black_balls: ℕ) : ℚ :=
  if white_balls + black_balls = total_balls ∧ total_balls = 15 ∧ white_balls = 7 ∧ black_balls = 8 then
    (white_balls / total_balls) * ((white_balls - 1) / (total_balls - 1))
  else 0

theorem probability_two_white_balls : 
  probability_of_two_white_balls 15 7 8 = 1/5
:= sorry

end probability_two_white_balls_l138_138117


namespace factor_probability_36_l138_138620

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138620


namespace probability_at_least_two_different_fruits_l138_138341

theorem probability_at_least_two_different_fruits :
  let p := (1 - (4 * (1 / 4)^4)) in
  p = 63 / 64 :=
by
  -- Conditions from part a)
  have h1 : Joe_has_four_meals_a_day := sorry,
  have h2 : Each_meal_choice_is_random_with_equal_probabilities := sorry,
  -- Correct answer
  have h3 : p = 63 / 64 := sorry,
  exact h3

end probability_at_least_two_different_fruits_l138_138341


namespace cost_to_fix_all_l138_138930

def num_shirts := 10
def num_pants := 12
def time_per_shirt := 1.5
def time_per_pants := 3.0
def rate_per_hour := 30

theorem cost_to_fix_all : 
  (num_shirts * time_per_shirt + num_pants * time_per_pants) * rate_per_hour = 1530 := 
by
  sorry

end cost_to_fix_all_l138_138930


namespace probability_of_high_quality_second_given_first_l138_138705

def num_high_quality_items : ℕ := 5
def num_defective_items : ℕ := 3
def total_items : ℕ := num_high_quality_items + num_defective_items

-- Define event A
def event_A : ℙ := num_high_quality_items / total_items.to_real

-- Define event B given A
def event_B_given_A : ℙ := (num_high_quality_items - 1) / (total_items - 1).to_real  

-- Define conditional probability P(B|A)
def P_B_given_A : ℙ := event_B_given_A / event_A

-- The correct answer to be proved
theorem probability_of_high_quality_second_given_first :
  P_B_given_A = 4 / 7 := 
by
  sorry

end probability_of_high_quality_second_given_first_l138_138705


namespace seating_arrangement_l138_138914

noncomputable def factorial : ℕ → ℕ
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def totalArrangements (n : ℕ) : ℕ :=
  factorial n

def restrictedArrangements (a b c d others : ℕ) : ℕ :=
  factorial ((a + b + c + d - 1) + others) * factorial (a + b + c + d)

def acceptableArrangements (total restricted : ℕ) : ℕ :=
  total - restricted

theorem seating_arrangement :
  acceptableArrangements (totalArrangements 10) (restrictedArrangements 1 1 1 1 6) = 3507840 := 
sorry

end seating_arrangement_l138_138914


namespace exists_infinite_bisecting_circles_l138_138875

-- Define circle and bisecting condition
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

def bisects (B C : Circle) : Prop :=
  let chord_len := (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2
  (B.center.1 - C.center.1)^2 + (B.center.2 - C.center.2)^2 + C.radius^2 = B.radius^2

-- Define the theorem statement
theorem exists_infinite_bisecting_circles (C1 C2 : Circle) (h : C1.center ≠ C2.center) :
  ∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧
  ∀ (b_center : ℝ × ℝ), (∃ (B : Circle), bisects B C1 ∧ bisects B C2 ∧ B.center = b_center) ↔
  2 * (C2.center.1 - C1.center.1) * b_center.1 + 2 * (C2.center.2 - C1.center.2) * b_center.2 =
  (C2.center.1^2 - C1.center.1^2) + (C2.center.2^2 - C1.center.2^2) + (C2.radius^2 - C1.radius^2) := 
sorry

end exists_infinite_bisecting_circles_l138_138875


namespace find_f_pi_4_value_l138_138265

-- Defining the function f along with the conditions given in the problem.
noncomputable def f (x A φ: ℝ) : ℝ := A * Real.sin (x + φ)

theorem find_f_pi_4_value :
  ∀ (A φ : ℝ), A > 0 ∧ 0 < φ ∧ φ < π ∧ (∀ x, f x A φ ≤ 1) ∧ f (π / 3) A φ = 1 / 2 → 
  f (3 * π / 4) 1 (π / 2) = -Real.sqrt 2 / 2 :=
by 
  intros A φ h,
  have h1 : A = 1 := by sorry,
  have φ_eq : φ = π / 2 := by sorry,
  rw [f, h1, φ_eq],
  exact by sorry


end find_f_pi_4_value_l138_138265


namespace amount_paid_is_correct_l138_138935

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l138_138935


namespace triangle_hypotenuse_l138_138321

noncomputable def hypotenuse_length (BM CN : ℝ) (is_right_triangle : Type)
  (angle_BAC_is_right : is_right_triangle → Prop)
  (medians_perpendicular : Prop)
  (BM_length : ℝ)
  (CN_length : ℝ)
  (BC_length : ℝ) : Prop := 
  BM_length = 30 ∧ CN_length = 40 ∧ angle_BAC_is_right is_right_triangle ∧ medians_perpendicular ∧ BC_length = 100 / 3

theorem triangle_hypotenuse (BM CN : ℝ) : hypotenuse_length BM CN 
  (is_right_triangle := ∃ A B C, is_triangle A B C ∧ ∠ A B C = 90)
  (angle_BAC_is_right := λ is_right_triangle, ∀ A B C, is_right_triangle → ∠ A B C = 90)
  (medians_perpendicular := ∃ B C, is_median_perpendicular B C)
  BM CN (30 : ℝ) (40 : ℝ) (100 / 3 : ℝ) :=
by
  sorry

end triangle_hypotenuse_l138_138321


namespace factor_probability_36_l138_138618

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138618


namespace probability_factor_36_l138_138519

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138519


namespace sum_of_cubes_l138_138355

variable {R : Type} [OrderedRing R] [Field R] [DecidableEq R]

theorem sum_of_cubes (a b c : R) (h₁ : a ≠ b) (h₂ : b ≠ c) (h₃ : c ≠ a)
    (h₄ : (a^3 + 12) / a = (b^3 + 12) / b) (h₅ : (b^3 + 12) / b = (c^3 + 12) / c) :
    a^3 + b^3 + c^3 = -36 := by
  sorry

end sum_of_cubes_l138_138355


namespace line_tangent_to_circle_l138_138894

-- Define the terms and conditions involved
def radius (O : Point) (r : ℝ) : Prop := r = 3
def distance_to_line (O : Point) (l : Line) (d : ℝ) : Prop := d = 3

-- The theorem stating the problem's conclusion given the conditions
theorem line_tangent_to_circle
  (O : Point) (l : Line) (d : ℝ) (r : ℝ)
  (h_radius : radius O r)
  (h_dist : distance_to_line O l d) :
  l.is_tangent_to_circle O r :=
by
  sorry

end line_tangent_to_circle_l138_138894


namespace probability_divisor_of_36_l138_138661

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138661


namespace batsman_total_score_l138_138105

-- We establish our variables and conditions first
variables (T : ℕ) -- total score
variables (boundaries : ℕ := 3) -- number of boundaries
variables (sixes : ℕ := 8) -- number of sixes
variables (boundary_runs_per : ℕ := 4) -- runs per boundary
variables (six_runs_per : ℕ := 6) -- runs per six
variables (running_percentage : ℕ := 50) -- percentage of runs made by running

-- Define the amounts of runs from boundaries and sixes
def runs_from_boundaries := boundaries * boundary_runs_per
def runs_from_sixes := sixes * six_runs_per

-- Main theorem to prove
theorem batsman_total_score :
  T = runs_from_boundaries + runs_from_sixes + T / 2 → T = 120 :=
by
  sorry

end batsman_total_score_l138_138105


namespace coefficient_x10_expansion_eq_36_l138_138032

open Finset

noncomputable def binomial_coeff (n k : ℕ) : ℕ := nat.choose n k

noncomputable def poly1 : ℕ → ℕ := λ n,
  if n = 0 then 1 else if n = 1 then 1 else if n = 2 then 1 else 0

noncomputable def poly2 : ℕ → ℕ := λ n, (-1)^n * binomial_coeff 10 n

def coefficient_of_x10_in_expansion : ℕ :=
  poly1 10 * poly2 0 + poly1 9 * poly2 1 + poly1 8 * poly2 2

theorem coefficient_x10_expansion_eq_36 : coefficient_of_x10_in_expansion = 36 :=
by sorry

end coefficient_x10_expansion_eq_36_l138_138032


namespace number_of_paperback_books_l138_138783

variables (P H : ℕ)

theorem number_of_paperback_books (h1 : H = 4) (h2 : P / 3 + 2 * H = 10) : P = 6 := 
by
  sorry

end number_of_paperback_books_l138_138783


namespace cost_price_is_92_percent_l138_138035

noncomputable def cost_price_percentage_of_selling_price (profit_percentage : ℝ) : ℝ :=
  let CP := (1 / ((profit_percentage / 100) + 1))
  CP * 100

theorem cost_price_is_92_percent (profit_percentage : ℝ) (h : profit_percentage = 8.695652173913043) :
  cost_price_percentage_of_selling_price profit_percentage = 92 :=
by
  rw [h]
  -- now we need to show that cost_price_percentage_of_selling_price 8.695652173913043 = 92
  -- by definition, cost_price_percentage_of_selling_price 8.695652173913043 is:
  -- let CP := 1 / (8.695652173913043 / 100 + 1)
  -- CP * 100 = (1 / (8.695652173913043 / 100 + 1)) * 100
  sorry

end cost_price_is_92_percent_l138_138035


namespace probability_cos_gt_half_l138_138016

open Real
open Set

theorem probability_cos_gt_half (x : ℝ) (hx : x ∈ Icc 0 π) : 
  measure_theory.measure.probability_space.prob 
    ({y | y ∈ Icc 0 π ∧ cos y > (1 / 2)} : Set ℝ) 
    = 1 / 3 :=
sorry

end probability_cos_gt_half_l138_138016


namespace eval_expr_at_sqrt3_minus_3_l138_138399

noncomputable def expr (a : ℝ) : ℝ :=
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2))

theorem eval_expr_at_sqrt3_minus_3 : expr (Real.sqrt 3 - 3) = -Real.sqrt 3 / 6 := 
  by sorry

end eval_expr_at_sqrt3_minus_3_l138_138399


namespace max_plots_l138_138113

-- Field dimensions
def field_length : ℝ := 30
def field_width : ℝ := 45

-- Total fencing available
def total_fencing : ℝ := 2700

-- Pathway dimensions
def pathway_width : ℝ := 5
def pathway_fencing : ℝ := 2 * field_length

-- Total fencing for the pathway
def fencing_for_pathway : ℝ := pathway_fencing
def available_fencing : ℝ := total_fencing - fencing_for_pathway

-- Side length of each square plot
def plot_side : ℝ := 7.5

-- Number of plots along the length of the field
def num_plots_length : ℕ := (field_length / plot_side).to_nat

-- Number of plots along the width of the field
def num_plots_width : ℕ := (field_width / plot_side).to_nat

-- Total number of plots per half
def plots_per_half : ℕ := num_plots_length * num_plots_width

-- Total number of plots
def total_plots : ℕ := 2 * plots_per_half

-- Statement to prove
theorem max_plots : total_plots = 48 :=
by {
  -- Definitions aligned with the problem conditions
  sorry
}

end max_plots_l138_138113


namespace probability_factor_of_36_is_1_over_4_l138_138506

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138506


namespace cone_surface_area_l138_138863

theorem cone_surface_area (l : ℝ) (θ : ℝ) (h_l : l = 2) (h_θ : θ = π / 6) : 
  ∃ A: ℝ, A = 3 * Real.pi :=
by
    use 3 * Real.pi
    sorry

end cone_surface_area_l138_138863


namespace right_triangle_complex_nums_l138_138281

noncomputable def count_nonzero_complex_nums (z : ℂ) : Prop :=
  (z ≠ 0) ∧ (0 + z = z) ∧ ((z + z^6)^2 + z^6^2 = 0)

theorem right_triangle_complex_nums : ∃ (z : ℂ), z ≠ 0 → count_nonzero_complex_nums = 5 :=
by sorry

end right_triangle_complex_nums_l138_138281


namespace max_value_3x_sub_9x_l138_138210

open Real

theorem max_value_3x_sub_9x : ∃ x : ℝ, 3^x - 9^x ≤ 1/4 ∧ (∀ y : ℝ, 3^y - 9^y ≤ 3^x - 9^x) :=
by
  sorry

end max_value_3x_sub_9x_l138_138210


namespace function_properties_l138_138865

-- Define the function f(x)
def f (x : ℝ) : ℝ := Real.cos x ^ 4 + 2 * Real.sin x * Real.cos x - Real.sin x ^ 4

-- Theorem statements encapsulating all the questions
theorem function_properties : 
  (∃ T > 0, ∀ x, f(x + T) = f(x)) ∧                  -- Periodicity: smallest positive period T
  ¬ (∀ x, f(-x) = f(x)) ∧ ¬ (∀ x, f(-x) = -f(x)) ∧   -- Neither odd nor even
  (∀ k : ℤ, ∀ x, x ∈ [-3*Real.pi/8 + k*Real.pi, k*Real.pi + Real.pi/8] →
    f(x + (Real.pi/16)) > f(x)) ∧                     -- Intervals of monotonic increase
  (max_val : ∀ x ∈ [0, Real.pi/2], f x ≤ Real.sqrt 2) ∧ -- Maximum value
  (min_val : ∀ x ∈ [0, Real.pi/2], f x ≥ -1)              -- Minimum value
:= by
  sorry

end function_properties_l138_138865


namespace daily_rental_cost_l138_138737

theorem daily_rental_cost (rental_fee_per_day : ℝ) (mileage_rate : ℝ) (budget : ℝ) (max_miles : ℝ) 
  (h1 : mileage_rate = 0.20) 
  (h2 : budget = 88.0) 
  (h3 : max_miles = 190.0) :
  rental_fee_per_day = 50.0 := 
by
  sorry

end daily_rental_cost_l138_138737


namespace probability_factor_of_36_l138_138536

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138536


namespace probability_factor_36_l138_138541

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138541


namespace max_value_3x_sub_9x_l138_138211

open Real

theorem max_value_3x_sub_9x : ∃ x : ℝ, 3^x - 9^x ≤ 1/4 ∧ (∀ y : ℝ, 3^y - 9^y ≤ 3^x - 9^x) :=
by
  sorry

end max_value_3x_sub_9x_l138_138211


namespace probability_factor_of_36_l138_138697

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138697


namespace geometric_sum_S12_l138_138430

theorem geometric_sum_S12 (a r : ℝ) (h₁ : r ≠ 1) (S4_eq : a * (1 - r^4) / (1 - r) = 24) (S8_eq : a * (1 - r^8) / (1 - r) = 36) : a * (1 - r^12) / (1 - r) = 42 := 
sorry

end geometric_sum_S12_l138_138430


namespace total_reading_materials_l138_138154

theorem total_reading_materials 
  (books_per_shelf : ℕ) (magazines_per_shelf : ℕ) (newspapers_per_shelf : ℕ) (graphic_novels_per_shelf : ℕ) 
  (bookshelves : ℕ)
  (h_books : books_per_shelf = 23) 
  (h_magazines : magazines_per_shelf = 61) 
  (h_newspapers : newspapers_per_shelf = 17) 
  (h_graphic_novels : graphic_novels_per_shelf = 29) 
  (h_bookshelves : bookshelves = 37) : 
  (books_per_shelf * bookshelves + magazines_per_shelf * bookshelves + newspapers_per_shelf * bookshelves + graphic_novels_per_shelf * bookshelves) = 4810 := 
by {
  -- Condition definitions are already given; the proof is omitted here.
  sorry
}

end total_reading_materials_l138_138154


namespace largest_possible_N_l138_138979

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138979


namespace probability_factor_of_36_is_1_over_4_l138_138494

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138494


namespace prove_perpendicular_l138_138349

-- Definitions of points and conditions for the problem 
variables {O A B C D E : Type}
variables [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]

-- Assuming the isosceles triangle with AB = AC
class IsoscelesTriangle (A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
(eq_sides : dist A B = dist A C)

-- Assuming O is the circumcenter of triangle ABC
class Circumcenter (O A B C : EuclideanSpace ℝ (Fin 3)) : Prop :=
(eq_dist : dist O A = dist O B ∧ dist O B = dist O C)

-- D is the midpoint of side AB
class Midpoint (D A B : EuclideanSpace ℝ (Fin 3)) : Prop :=
(midpoint_eq : D = (A + B) / 2)

-- E is the centroid of triangle ACD
class Centroid (E A C D : EuclideanSpace ℝ (Fin 3)) : Prop :=
(centroid_eq : E = (A + C + D) / 3)

-- Definition of perpendicular vectors
def Perpendicular (u v : EuclideanSpace ℝ (Fin 3)) : Prop :=
inner u v = 0

-- Main theorem statement
theorem prove_perpendicular
  {O A B C D E : EuclideanSpace ℝ (Fin 3)}
  [IsoscelesTriangle A B C]
  [Circumcenter O A B C]
  [Midpoint D A B]
  [Centroid E A C D] : Perpendicular (O -ᵥ E) (C -ᵥ D) :=
sorry

end prove_perpendicular_l138_138349


namespace factor_probability_l138_138634

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138634


namespace cannot_reach_60_cents_with_six_coins_l138_138021

def penny_val := 1
def nickel_val := 5
def dime_val := 10
def quarter_val := 25
def total_value (p : ℕ) (n : ℕ) (d : ℕ) (q : ℕ) : ℕ :=
  p * penny_val + n * nickel_val + d * dime_val + q * quarter_val

def total_count (p : ℕ) (n : ℕ) (d : ℕ) (q : ℕ) : ℕ :=
  p + n + 2 * d + q

theorem cannot_reach_60_cents_with_six_coins :
  ∀ p n d q : ℕ, total_count p n d q = 6 → total_value p n d q ≠ 60 :=
by {
  intros,
  sorry
}

end cannot_reach_60_cents_with_six_coins_l138_138021


namespace Events_B_and_C_mutex_l138_138796

-- Definitions of events based on scores
def EventA (score : ℕ) := score ≥ 1 ∧ score ≤ 10
def EventB (score : ℕ) := score > 5 ∧ score ≤ 10
def EventC (score : ℕ) := score > 1 ∧ score < 6
def EventD (score : ℕ) := score > 0 ∧ score < 6

-- Mutually exclusive definition:
def mutually_exclusive (P Q : ℕ → Prop) := ∀ (x : ℕ), ¬ (P x ∧ Q x)

-- The proof statement:
theorem Events_B_and_C_mutex : mutually_exclusive EventB EventC :=
by
  sorry

end Events_B_and_C_mutex_l138_138796


namespace probability_factor_36_l138_138590

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138590


namespace optimal_estimate_l138_138811

theorem optimal_estimate :
  ∃ (E : ℝ), E = 500 ∧ E ∈ set.Icc 0 1000 :=
sorry

end optimal_estimate_l138_138811


namespace cyclic_iff_sum_segments_eq_l138_138969

variable (A B C D E F : Type)

-- Conditions as variables and predicates
variable [ConvexQuadrilateral A B C D]
variable [NoParallelSides A B C D]
variable [Intersection E A B D C]
variable [Intersection F B C A D]
variable [OnSegment A B E]
variable [OnSegment C B F]

-- Main statement
theorem cyclic_iff_sum_segments_eq :
  (CyclicQuadrilateral A B C D) ↔ (SegmentLength E A + SegmentLength A F = SegmentLength E C + SegmentLength C F) := 
sorry

end cyclic_iff_sum_segments_eq_l138_138969


namespace probability_factor_of_36_l138_138531

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138531


namespace all_numbers_positive_l138_138229

theorem all_numbers_positive 
  (nums : Fin 101 → ℝ)
  (h : ∀ (s : Finset (Fin 101)), s.card = 50 → (∑ i in s, nums i) < (∑ i in (Finset.univ \ s), nums i)) :
  ∀ i, 0 < nums i :=
by
  -- Placeholder for the proof
  sorry

end all_numbers_positive_l138_138229


namespace max_value_of_3_pow_x_minus_9_pow_x_l138_138213

open Real

theorem max_value_of_3_pow_x_minus_9_pow_x :
  ∃ (x : ℝ), ∀ y : ℝ, 3^x - 9^x ≤ 3^y - 9^y := 
sorry

end max_value_of_3_pow_x_minus_9_pow_x_l138_138213


namespace factor_probability_36_l138_138613

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138613


namespace probability_divisor_of_36_l138_138655

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138655


namespace find_quadratic_and_extremes_l138_138247

-- Definitions from the problem conditions

def quadratic_function (f : ℝ → ℝ) :=
  ∃ a b c : ℝ, ∀ x : ℝ, f(x) = a * x^2 + b * x + c

def min_value_at (f : ℝ → ℝ) (x₀ : ℝ) (min_val : ℝ) :=
  ∀ x : ℝ, f(x₀) = min_val ∧ (∀ h : ℝ, f(x₀) ≤ f(x₀ + h))

def passes_through_origin (f : ℝ → ℝ) :=
  f(0) = 0

-- The main proof statement

theorem find_quadratic_and_extremes :
  ∃ f : ℝ → ℝ, quadratic_function f ∧ min_value_at f 2 (-4) ∧ passes_through_origin f ∧
  (∀ t ∈ set.Icc (-1 : ℝ) 3, (f t = (t-2)^2 - 4) ∧ 
    (∀ g, g = λ (t : ℝ), (t-2)^2 - 4 → 
      (set.Icc (-1 : ℝ) 3).nonempty → 
        (f (argmin g {t : ℝ | t ∈ set.Icc (-1 : ℝ) 3}) = -4 ∧ 
         f (argmax g {t : ℝ | t ∈ set.Icc (-1 : ℝ) 3}) = 5))) :=
by
  sorry

end find_quadratic_and_extremes_l138_138247


namespace largest_possible_N_l138_138980

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138980


namespace probability_factor_of_36_l138_138535

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138535


namespace probability_divisor_of_36_l138_138664

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138664


namespace larger_number_is_1590_l138_138036

theorem larger_number_is_1590 (L S : ℕ) (h1 : L - S = 1365) (h2 : L = 7 * S + 15) : L = 1590 :=
by
  sorry

end larger_number_is_1590_l138_138036


namespace probability_factor_of_36_l138_138651

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138651


namespace closest_fraction_to_two_thirds_l138_138810

noncomputable def abs_diff (a b : ℚ) : ℚ := abs (a - b)

def fractions := [ (Rat.mk 4 7), (Rat.mk 9 14), (Rat.mk 20 31), (Rat.mk 61 95), (Rat.mk 73 110) ]

theorem closest_fraction_to_two_thirds : 
  let target := Rat.mk 2 3 in 
  let fractions := fractions in 
  let closest := (Rat.mk 73 110) in 
  ∀ (f : ℚ), f ∈ fractions → abs_diff target f ≥ abs_diff target closest := sorry

end closest_fraction_to_two_thirds_l138_138810


namespace height_of_tank_l138_138747

-- Define the radius of the cylinder's base (in cm)
def radius : ℝ := 3

-- Define the half-full volume of the tank (up to height h/2)
def half_full_volume (h : ℝ) : ℝ := π * radius^2 * (h / 2)

-- Define the volume of one sphere with diameter equal to the base diameter of the cylinder
def sphere_volume : ℝ := (4 / 3) * π * (radius^2 * √(radius^2))

-- The combined volume of the two spheres
def combined_sphere_volume : ℝ := 2 * sphere_volume

-- The main theorem to prove: the height of the tank
theorem height_of_tank (h : ℝ) (condition : half_full_volume h = combined_sphere_volume) : h = 16 := 
by
  -- The proof would go here
  sorry

end height_of_tank_l138_138747


namespace orthocenter_PQR_is_correct_l138_138911

def Point := (ℝ × ℝ × ℝ)

def P : Point := (2, 3, 4)
def Q : Point := (6, 4, 2)
def R : Point := (4, 5, 6)

def orthocenter (P Q R : Point) : Point := sorry

theorem orthocenter_PQR_is_correct : orthocenter P Q R = (3 / 2, 13 / 2, 5) :=
sorry

end orthocenter_PQR_is_correct_l138_138911


namespace opposite_angles_equal_l138_138380

-- Defining the setup
variable {A B C D : Type}
variable [LinearOrderedField R] -- Ensure we are working in a linear ordered field, such as the real numbers.
variable (A D B C : R)

-- Conditions
def parallel_lines_1 : Prop := ∀ (x y : R), x ∈ {A, D} → y ∈ {A, D} → x ≠ y → x - y = 0
def parallel_lines_2 : Prop := ∀ (x y : R), x ∈ {B, C} → y ∈ {B, C} → x ≠ y → x - y = 0
def parallel_ab_cd : Prop := ∀ (x y : R), x ∈ {A, B} → y ∈ {C, D} → x ≠ y → x - y = 0

-- Theorem
theorem opposite_angles_equal (A B C D : R) :
  parallel_lines_1 A D ∧ parallel_lines_2 B C ∧ parallel_ab_cd A B C D →
  ∠ABC = ∠ADC ∧ ∠BAD = ∠BCD :=
by
  sorry

end opposite_angles_equal_l138_138380


namespace certain_sale_price_property_l138_138168

-- Define the unit sales list and corresponding properties
def unit_sales : List ℝ := [50, 50, 97, 97, 97, 120, 125, 155, 199, 199, 239]
noncomputable def mean_sale_price := (unit_sales.sum / unit_sales.length)

-- The statement to prove: there's exactly one unit sale greater than the mean but less than a certain sale price
theorem certain_sale_price_property (h_cond : (unit_sales.length = 11) ∧ (unit_sales.sum = 1318)) :
  ∃! x, (x > mean_sale_price) ∧ (x < 120) :=
by 
  sorry

end certain_sale_price_property_l138_138168


namespace denmark_pizza_combinations_l138_138804

theorem denmark_pizza_combinations :
  (let cheese_options := 3
   let meat_options := 4
   let vegetable_options := 5
   let invalid_combinations := 1
   let total_combinations := cheese_options * meat_options * vegetable_options
   let valid_combinations := total_combinations - invalid_combinations
   valid_combinations = 59) :=
by
  sorry

end denmark_pizza_combinations_l138_138804


namespace AA1_BB1_CC1_triangle_l138_138906

variables {A B C A₁ B₁ C₁ : Type*} [metric_space A]
variables [metric_space B] [metric_space C]
variables [metric_space A₁] [metric_space B₁] [metric_space C₁]

def is_excircle_touchpoint (P Q R : Type*) (P₁ : Type*) [metric_space P] [metric_space Q] [metric_space R] [metric_space P₁] : Prop :=
-- Definition that point is an excircle touchpoint, omitted for simplicity
sorry 

def excircle_touchpoints (A B C A₁ B₁ C₁ : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space A₁] [metric_space B₁] [metric_space C₁] : Prop :=
is_excircle_touchpoint A B C A₁ ∧ is_excircle_touchpoint B C A B₁ ∧ is_excircle_touchpoint C A B C₁

theorem AA1_BB1_CC1_triangle (A B C A₁ B₁ C₁ : Type*) [metric_space A] [metric_space B] [metric_space C] [metric_space A₁] [metric_space B₁] [metric_space C₁]
  (h : excircle_touchpoints A B C A₁ B₁ C₁):
  (dist A A₁) + (dist B B₁) > (dist C C₁) ∧ (dist B B₁) + (dist C C₁) > (dist A A₁) ∧ (dist C C₁) + (dist A A₁) > (dist B B₁) :=
sorry

end AA1_BB1_CC1_triangle_l138_138906


namespace solution_of_linear_equation_l138_138713

theorem solution_of_linear_equation (x y : ℝ) (h₁ : x = 4) (h₂ : y = 2) : 2 * x - y = 6 :=
by
  rw [h₁, h₂]
  norm_num
  sorry

end solution_of_linear_equation_l138_138713


namespace average_score_l138_138312

theorem average_score (m n : ℝ) (hm : m ≥ 0) (hn : n ≥ 0) :
  (20 * m + 23 * n) / (20 + 23) = 20 / 43 * m + 23 / 43 * n := sorry

end average_score_l138_138312


namespace matrix_det_evaluation_l138_138812

noncomputable def matrix_det (x y z : ℝ) : ℝ :=
  Matrix.det ![
    ![1,   x,     y,     z],
    ![1, x + y,   y,     z],
    ![1,   x, x + y,     z],
    ![1,   x,     y, x + y + z]
  ]

theorem matrix_det_evaluation (x y z : ℝ) :
  matrix_det x y z = y * x * x + y * y * x :=
by sorry

end matrix_det_evaluation_l138_138812


namespace solve_arccos_eq_l138_138023

theorem solve_arccos_eq (x : ℝ) (h : arccos (4 * x) - arccos (2 * x) = π / 4) : 
  x = 1 / (2 * sqrt (19 - 8 * sqrt 2)) :=
sorry

end solve_arccos_eq_l138_138023


namespace abs_eq_self_nonneg_l138_138419

theorem abs_eq_self_nonneg (x : ℝ) : abs x = x ↔ x ≥ 0 :=
sorry

end abs_eq_self_nonneg_l138_138419


namespace probability_factor_36_l138_138551

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138551


namespace ed_lost_seven_marbles_l138_138189

theorem ed_lost_seven_marbles (D L : ℕ) (h1 : ∃ (Ed_init Tim_init : ℕ), Ed_init = D + 19 ∧ Tim_init = D - 10)
(h2 : ∃ (Ed_final Tim_final : ℕ), Ed_final = D + 19 - L - 4 ∧ Tim_final = D - 10 + 4 + 3)
(h3 : ∀ (Ed_final : ℕ), Ed_final = D + 8)
(h4 : ∀ (Tim_final : ℕ), Tim_final = D):
  L = 7 :=
by
  sorry

end ed_lost_seven_marbles_l138_138189


namespace proj_3_3_on_6_neg2_l138_138338

-- Defining the given vectors
def u : Vector ℝ := ⟨3, 3⟩
def v : Vector ℝ := ⟨6, -2⟩

-- Dot product of two vectors
def dot (a b : Vector ℝ) : ℝ := a.x * b.x + a.y * b.y

-- Projection of vector u onto v
def proj (u v : Vector ℝ) : Vector ℝ :=
  let scalar := (dot u v) / (dot v v)
  ⟨scalar * v.x, scalar * v.y⟩

-- Proving the specific projection case
theorem proj_3_3_on_6_neg2 : proj u v = ⟨1.8, -0.6⟩ :=
  sorry

end proj_3_3_on_6_neg2_l138_138338


namespace complex_sqrt_inequality_l138_138851

theorem complex_sqrt_inequality (n : ℕ) (z : Finₓ n → ℂ) :
  |(Complex.re (Complex.sqrt (∑ i , z i ^ 2)))| ≤ ∑ i , |Complex.re (z i)| := 
sorry

end complex_sqrt_inequality_l138_138851


namespace three_digit_number_is_156_l138_138198

theorem three_digit_number_is_156:
  ∃ (Π B Γ : ℕ), 
  (Π != B) ∧ (Π != Γ) ∧ (B != Γ) ∧ -- Digits are distinct
  (0 ≤ Π ∧ Π < 10) ∧ (0 ≤ B ∧ B < 10) ∧ (0 ≤ Γ ∧ Γ < 10) ∧ -- Digits are in the range 0 to 9
  (100 * Π + 10 * B + Γ = (Π + B + Γ) * ((Π + B + Γ) + 1)) ∧ -- The given condition
  (100 * Π + 10 * B + Γ = 156) -- The required number is 156
:=
begin
  sorry
end

end three_digit_number_is_156_l138_138198


namespace digit_zero_count_in_20_pow_10_l138_138885

theorem digit_zero_count_in_20_pow_10 : 
  ∀ (a b : ℕ) (n k : ℕ), (a = 2) → (b = 10) → (n = 10) → (k = 1024) → (10^n = 10000000000) → (a^n = 1024) → (a * b) ^ n → number_of_zeros ((a^n) * (b^n)) = 11 := 
by
  assume (a b n k : ℕ) (ha : a = 2) (hb : b = 10) (hn : n = 10) (hk : k = 1024) (h10 : 10^n = 10000000000) (h2 : 2^n = 1024),
  sorry

def number_of_zeros : ℕ → ℕ
| 0       := 1
| (n+1) := if n % 10 = 0 then number_of_zeros (n / 10) + 1 else number_of_zeros (n / 10)

end digit_zero_count_in_20_pow_10_l138_138885


namespace negation_abs_val_statement_l138_138046

theorem negation_abs_val_statement (x : ℝ) :
  ¬ (|x| ≤ 3 ∨ |x| > 5) ↔ (|x| > 3 ∧ |x| ≤ 5) :=
by sorry

end negation_abs_val_statement_l138_138046


namespace not_possible_110_cents_l138_138020

-- Define the coin denominations
def coins : List ℕ := [1, 5, 10, 25, 50]

-- Definition to check if a certain amount can be achieved with exactly 6 coins
def canAchieve (coins : List ℕ) (amt : ℕ) (num_coins : ℕ) : Prop :=
  ∃ (coin_counts : List ℕ), coin_counts.length = num_coins ∧
  coin_counts.all (λ x, x ∈ coins) ∧
  coin_counts.sum = amt

-- The theorem to prove
theorem not_possible_110_cents : ¬ canAchieve coins 110 6 :=
by
  sorry

end not_possible_110_cents_l138_138020


namespace profit_percentage_B_l138_138763

-- Definitions based on given conditions
def CP_A : ℝ := 112.5
def profit_A : ℝ := 0.60
def SP_C : ℝ := 225

-- Question: Prove that B's profit percentage is 25%
theorem profit_percentage_B : 
  let SP_B := CP_A + profit_A * CP_A in
  let CP_B := SP_B in
  let profit_B := SP_C - CP_B in
  let profit_percentage_B := (profit_B / CP_B) * 100 in
  profit_percentage_B = 25 :=
by
  sorry

end profit_percentage_B_l138_138763


namespace third_term_arithmetic_sequence_l138_138299

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end third_term_arithmetic_sequence_l138_138299


namespace solve_cubic_equation_l138_138197

theorem solve_cubic_equation (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by sorry

end solve_cubic_equation_l138_138197


namespace probability_factor_of_36_l138_138638

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138638


namespace locus_of_point_P_is_ellipse_l138_138044

noncomputable def dist (P Q : Point) : ℝ := sorry  -- Define distance function

theorem locus_of_point_P_is_ellipse 
  (P F1 F2 : Point) 
  (m : ℝ) 
  (h1 : dist F1 F2 < m) 
  (h2 : dist P F1 + dist P F2 = m) : 
  is_ellipse { P : Point | dist P F1 + dist P F2 = m } :=
sorry

end locus_of_point_P_is_ellipse_l138_138044


namespace math_club_prob_l138_138057

/-- There are 3 math clubs with 6, 9, and 10 students respectively.
Each club has 3 co-presidents. Prove that the probability of selecting exactly two co-presidents
out of three randomly selected members, across randomly chosen clubs, is approximately 0.27977. -/
theorem math_club_prob :
    let clubs := [(6, 3), (9, 3), (10, 3)] in
    let total_prob := (1/3 : ℝ) * (
        (↑(nat.choose 3 2 * nat.choose 3 1) / ↑(nat.choose 6 3)) +
        (↑(nat.choose 3 2 * nat.choose 6 1) / ↑(nat.choose 9 3)) +
        (↑(nat.choose 3 2 * nat.choose 7 1) / ↑(nat.choose 10 3))
    ) in
    |total_prob - 0.27977| < 0.001 :=
by
  -- Add your proof here
  sorry

end math_club_prob_l138_138057


namespace max_value_of_quadratic_function_l138_138070

noncomputable def quadratic_function (x : ℝ) : ℝ := -5*x^2 + 25*x - 15

theorem max_value_of_quadratic_function : ∃ x : ℝ, quadratic_function x = 750 :=
by
-- maximum value
sorry

end max_value_of_quadratic_function_l138_138070


namespace constants_cd_l138_138809

theorem constants_cd (c d : ℚ) (h : ∀ x : ℚ, x > 0 → (c / (2^x - 1) + d / (2^x + 3) = (3 * 2^x + 4) / ((2^x - 1) * (2^x + 3)))) : c - d = 1 / 2 :=
sorry 

end constants_cd_l138_138809


namespace inequality_proof_l138_138003

theorem inequality_proof (x y z : ℝ) (hx : x > -1) (hy : y > -1) (hz : z > -1) : 
    (1 + x^2) / (1 + y + z^2) + (1 + y^2) / (1 + z + x^2) + (1 + z^2) / (1 + x + y^2) ≥ 2 := 
by
  sorry

end inequality_proof_l138_138003


namespace max_triangle_area_l138_138843

-- Definitions derived from the given conditions
def parabola (x y : ℝ) : Prop := y^2 = 6 * x
def points_on_parabola (A B : ℝ × ℝ) : Prop := 
  ∃ x1 y1 x2 y2, x1 ≠ x2 ∧ (x1 + x2 = 4) ∧ A = (x1, y1) ∧ B = (x2, y2) ∧ parabola x1 y1 ∧ parabola x2 y2 

-- The main theorem stating the maximum area of triangle ABC
theorem max_triangle_area (A B C : ℝ × ℝ) (C_on_x_axis : C.2 = 0) :
  points_on_parabola A B →
  (∃ M : ℝ × ℝ, M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
  (∃ C, C = (5, 0)) →
  ∀ t, t ∈ triangle_area A B C →
  t ≤ (14 / 3) * real.sqrt 7 :=
sorry -- proof goes here

end max_triangle_area_l138_138843


namespace continuous_function_solution_l138_138196

theorem continuous_function_solution (f : ℝ → ℝ) (a : ℝ) (h_continuous : Continuous f) (h_pos : 0 < a)
    (h_equation : ∀ x, f x = a^x * f (x / 2)) :
    ∃ C : ℝ, ∀ x, f x = C * a^(2 * x) := 
sorry

end continuous_function_solution_l138_138196


namespace compute_f_of_1_plus_g_of_3_l138_138955

def f (x : ℝ) : ℝ := 3 * x - 4
def g (x : ℝ) : ℝ := x^2 + 1

theorem compute_f_of_1_plus_g_of_3 : f (1 + g 3) = 29 := by 
  sorry

end compute_f_of_1_plus_g_of_3_l138_138955


namespace angle_y_measurement_l138_138216

theorem angle_y_measurement :
  ∀ (A B C D : Type) (degrees : A → ℝ),
    (∠ ABC = 180) → 
    (∠ ABC = 102) → 
    (∠ BAD = 34) → 
    (∠ ADB = 19) → 
    (∠ BAD + ∠ ABD + ∠ ADB = 180) → 
    (∠ y = 68) := by 
  sorry

end angle_y_measurement_l138_138216


namespace obtuse_angles_in_regular_pentagon_l138_138883

theorem obtuse_angles_in_regular_pentagon (n : ℕ) (h_n: n = 5) : 
  let sum_of_interior_angles := 180 * (n - 2) in
  let each_angle := sum_of_interior_angles / n in
  ∀ angle, angle = each_angle → angle > 90 → ∃ k : ℕ, k = 5 :=
by
  sorry

end obtuse_angles_in_regular_pentagon_l138_138883


namespace factor_probability_36_l138_138610

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138610


namespace problem_solution_l138_138925

-- Given conditions
def line (t : ℝ) : (ℝ × ℝ) := (t, - (Real.sqrt 3) * t)

def curve1 (θ : ℝ) : (ℝ × ℝ) := (Real.cos θ, 1 + Real.sin θ)

def curve2 (θ : ℝ) : ℝ := -2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ

-- Polar equation of curve1
def polar_curve1_eq (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.sin θ

-- Rectangular equation of curve2
def rectangular_curve2_eq (x y : ℝ) : Prop :=
  (x + 1)^2 + (y - Real.sqrt 3)^2 = 4

-- Length of |AB|
def length_AB_eq (θ1 θ2 : ℝ) : ℝ :=
  let ρ1 := 2 * Real.sin θ1
  let ρ2 := -2 * Real.cos θ2 + 2 * Real.sqrt 3 * Real.sin θ2
  Real.abs (ρ1 - ρ2)

theorem problem_solution (θ : ℝ) :
  (polar_curve1_eq (curve1 θ).fst θ) ∧
  (rectangular_curve2_eq (curve2 θ) θ) ∧
  (curve1 (2 * Real.pi / 3)).fst = sqrt 3 ∧
  (curve2 (2 * Real.pi / 3)) = 4 ∧
  length_AB_eq (2 * Real.pi / 3) (2 * Real.pi / 3) = 4 - sqrt 3 :=
by sorry

end problem_solution_l138_138925


namespace probability_factor_of_36_l138_138530

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138530


namespace fish_weight_l138_138284

-- Definitions of weights
variable (T B H : ℝ)

-- Given conditions
def cond1 : Prop := T = 9
def cond2 : Prop := H = T + (1/2) * B
def cond3 : Prop := B = H + T

-- Theorem to prove
theorem fish_weight (h1 : cond1 T) (h2 : cond2 T B H) (h3 : cond3 T B H) :
  T + B + H = 72 :=
by
  sorry

end fish_weight_l138_138284


namespace geom_seq_prod_of_terms_l138_138923

theorem geom_seq_prod_of_terms (a : ℕ → ℝ) (r : ℝ) (h_geom : ∀ n, a (n + 1) = r * a n) (h_a5 : a 5 = 2) : a 1 * a 9 = 4 := by
  sorry

end geom_seq_prod_of_terms_l138_138923


namespace problem1_problem2_problem3_l138_138264

noncomputable def f (a : ℝ) (x : ℝ) := (Real.log (x + 1)) / (a * x + 1)

theorem problem1 : (f 1 0 = 0) ∧ (∀ x, Deriv.deriv (f 1) x = (1 - Real.log (x + 1)) / (x + 1)^2) :=
begin
  sorry
end

theorem problem2 (a : ℝ) : (∀ x ∈ Ioo 0 1, Deriv.deriv (f a) x ≥ 0) ↔ (a ∈ Icc (-1) (1 / (2 * Real.log 2 - 1))) :=
begin
  sorry
end

theorem problem3 (x y z : ℝ) (h : x + y + z = 1) : 
  (0 < x) → (0 < y) → (0 < z) → 
  ((3 * x - 1) * Real.log (x + 1) / (x - 1) + 
   (3 * y - 1) * Real.log (y + 1) / (y - 1) + 
   (3 * z - 1) * Real.log (z + 1) / (z - 1)) ≤ 0 :=
begin
  sorry
end

end problem1_problem2_problem3_l138_138264


namespace min_positive_value_l138_138215

theorem min_positive_value :
  ∃ s : list (fin 2022 → ℤ), ((∀ i < 2022, s i = 1 ∨ s i = -1) ∧ list.sum (list.zip_with (*) (list.range 2022) s) = 1) :=
sorry

end min_positive_value_l138_138215


namespace correct_answer_l138_138226

noncomputable def sqrt_2 : ℝ := Real.sqrt 2

def P : Set ℝ := { x | x^2 - 2*x - 3 ≤ 0 }

theorem correct_answer : {sqrt_2} ⊆ P :=
sorry

end correct_answer_l138_138226


namespace vector_properties_l138_138221

variables (a : ℝ^3) (h_a : a ≠ 0)

theorem vector_properties :
  (-3 : ℝ) • a = -(3 • a) ∧
  a - (3 • a) = -2 • a ∧
  ‖a‖ = 1 / 3 * ‖-3 • a‖ :=
by
  sorry

end vector_properties_l138_138221


namespace three_digit_number_is_156_l138_138199

theorem three_digit_number_is_156:
  ∃ (Π B Γ : ℕ), 
  (Π != B) ∧ (Π != Γ) ∧ (B != Γ) ∧ -- Digits are distinct
  (0 ≤ Π ∧ Π < 10) ∧ (0 ≤ B ∧ B < 10) ∧ (0 ≤ Γ ∧ Γ < 10) ∧ -- Digits are in the range 0 to 9
  (100 * Π + 10 * B + Γ = (Π + B + Γ) * ((Π + B + Γ) + 1)) ∧ -- The given condition
  (100 * Π + 10 * B + Γ = 156) -- The required number is 156
:=
begin
  sorry
end

end three_digit_number_is_156_l138_138199


namespace value_of_expression_l138_138890

theorem value_of_expression (x y : ℕ) (h1 : x = 3) (h2 : y = 4) : 
  (x^4 + 3 * y^3 + 10) / 7 = 283 / 7 := by
  sorry

end value_of_expression_l138_138890


namespace area_of_quadrilateral_l138_138172

theorem area_of_quadrilateral (a : ℝ) (ha : a > 0) :
  ∃ A, A = (SetOf (λ (x y : ℝ), (x - a * y)^2 = 9 * a^2 ∧ (a * x + y)^2 = 4 * a^2) ∧ 
         0 ≤ A) ∧ A = (24 * a^2) / (1 + a^2) :=
sorry

end area_of_quadrilateral_l138_138172


namespace function_is_odd_and_decreasing_l138_138147

noncomputable def f : ℝ → ℝ := λ x, -x^3 - x

theorem function_is_odd_and_decreasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ (∀ x y : ℝ, x < y → f x > f y) := by
  sorry

end function_is_odd_and_decreasing_l138_138147


namespace construction_possible_l138_138367

-- Define the vertex O of the right angle
variables {O A B P : Point}

-- Define the conditions
axiom angle_constraint : ∀ (α : ℝ), ∠P A O = 2 * α ∧ ∠P B A = α
axiom distance_constraint : dist O A < dist O B

-- Prove the problem is solvable if and only if A < B
theorem construction_possible (α : ℝ) : ∃ P, ∠P A O = 2 * α ∧ ∠P B A = α ↔ dist O A < dist O B :=
by
  sorry

end construction_possible_l138_138367


namespace domain_of_function_l138_138037

-- Definitions based on conditions
def function_domain (x : ℝ) : Prop := (x > -1) ∧ (x ≠ 1)

-- Prove the domain is the desired set
theorem domain_of_function :
  ∀ x, function_domain x ↔ ((-1 < x ∧ x < 1) ∨ (1 < x)) :=
  by
    sorry

end domain_of_function_l138_138037


namespace log_comparison_l138_138169

theorem log_comparison :
  (Real.log 80 / Real.log 20) < (Real.log 640 / Real.log 80) :=
by
  sorry

end log_comparison_l138_138169


namespace probability_factor_of_36_l138_138491

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138491


namespace equilateral_triangle_area_outside_circle_ratio_l138_138858

noncomputable def equilateral_triangle_area_ratio (r : ℝ) : ℝ :=
  let s := r * (real.sqrt 3) in
  let area_triangle := (real.sqrt 3 / 4) * s^2 in
  let area_circle := real.pi * r^2 in
  (area_triangle - area_circle) / area_triangle

theorem equilateral_triangle_area_outside_circle_ratio (r : ℝ) (h : r > 0) :
  equilateral_triangle_area_ratio r = (4/3 : ℝ) - (4 * real.sqrt 3 * real.pi / 27) :=
by
  -- proof omitted
  sorry

end equilateral_triangle_area_outside_circle_ratio_l138_138858


namespace coefficient_x3_in_expansion_l138_138033

noncomputable def binomial_coeff (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

def polynomial1 : Polynomial ℤ :=
  Polynomial.C 1 - Polynomial.C 1 * Polynomial.X + Polynomial.C 1 * Polynomial.X^2

def polynomial2 : Polynomial ℤ :=
  Polynomial.C 1 + Polynomial.X)^6

def expanded_polynomial : Polynomial ℤ :=
  polynomial1 * polynomial2

theorem coefficient_x3_in_expansion :
  (expanded_polynomial.coeff 3) = 11 :=
sorry

end coefficient_x3_in_expansion_l138_138033


namespace monotonic_decrease_interval_l138_138415

noncomputable def f (x : ℝ) : ℝ := (1 / 2) * x^2 - Real.log x

def derivative_f (x : ℝ) : ℝ := x - 1 / x

theorem monotonic_decrease_interval :
  ∃ (a b : ℝ), (∀ x, a < x ∧ x < b → derivative_f x < 0) ∧ a = 0 ∧ b = 1 := by
  sorry

end monotonic_decrease_interval_l138_138415


namespace total_players_must_be_square_l138_138309

variables (k m : ℕ)
def n : ℕ := k + m

theorem total_players_must_be_square (h: (k*(k-1) / 2) + (m*(m-1) / 2) = k * m) :
  ∃ (s : ℕ), n = s^2 :=
by sorry

end total_players_must_be_square_l138_138309


namespace total_cookies_baked_l138_138742

theorem total_cookies_baked (members sheets_per_member cookies_per_sheet : ℕ) (h1 : members = 100) (h2 : sheets_per_member = 10) (h3 : cookies_per_sheet = 16) :
  (members * sheets_per_member * cookies_per_sheet) = 16000 :=
by
  rw [h1, h2, h3]
  norm_num
  -- Additional steps or simplification if necessary
  sorry

end total_cookies_baked_l138_138742


namespace all_equal_l138_138270

theorem all_equal (a : Fin 100 → ℝ) 
  (h1 : a 0 - 3 * a 1 + 2 * a 2 ≥ 0)
  (h2 : a 1 - 3 * a 2 + 2 * a 3 ≥ 0)
  (h3 : a 2 - 3 * a 3 + 2 * a 4 ≥ 0)
  -- ...
  (h99: a 98 - 3 * a 99 + 2 * a 0 ≥ 0)
  (h100: a 99 - 3 * a 0 + 2 * a 1 ≥ 0) : 
    ∀ i : Fin 100, a i = a 0 := 
by 
  sorry

end all_equal_l138_138270


namespace min_value_of_sum_of_squares_l138_138970

theorem min_value_of_sum_of_squares (a b c : ℕ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) 
  (h : a^2 + b^2 - c = 2022) : 
  a^2 + b^2 + c^2 = 2034 ∧ a = 27 ∧ b = 36 ∧ c = 3 := 
sorry

end min_value_of_sum_of_squares_l138_138970


namespace probability_divisor_of_36_l138_138667

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138667


namespace length_is_56_l138_138090

noncomputable def length_of_plot (b : ℝ) : ℝ := b + 12

theorem length_is_56 (b : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) (h_cost : cost_per_meter = 26.50) (h_total_cost : total_cost = 5300) (h_fencing : 26.50 * (4 * b + 24) = 5300) : length_of_plot b = 56 := 
by 
  sorry

end length_is_56_l138_138090


namespace has_two_distinct_real_roots_parabola_equation_l138_138279

open Real

-- Define the quadratic polynomial
def quad_poly (m : ℝ) (x : ℝ) : ℝ := x^2 - 2 * m * x + m^2 - 4

-- Question 1: Prove that the quadratic equation has two distinct real roots
theorem has_two_distinct_real_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (quad_poly m x₁ = 0) ∧ (quad_poly m x₂ = 0) := by
  sorry

-- Question 2: Prove the equation of the parabola given certain conditions
theorem parabola_equation (m : ℝ) (hx : quad_poly m 0 = 0) : 
  m = 0 ∧ ∀ x : ℝ, quad_poly m x = x^2 - 4 := by
  sorry

end has_two_distinct_real_roots_parabola_equation_l138_138279


namespace exists_point_proportional_distances_l138_138330

theorem exists_point_proportional_distances
  (A B C D L : Point)
  (hABCD : Quadrilateral A B C D)
  (hAB_parallel_CD : Parallel (LineThrough A B) (LineThrough C D))
  (hAD_parallel_BC : Parallel (LineThrough A D) (LineThrough B C)) :
  ∃ L : Point, ProportionalDistances L (LineThrough A B) (LineThrough B C) (LineThrough C D) (LineThrough A D) := 
begin
  sorry
end

end exists_point_proportional_distances_l138_138330


namespace not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l138_138776

def vector_a : ℝ × ℝ := (3, 2)
def vector_vA : ℝ × ℝ := (3, -2)
def vector_vB : ℝ × ℝ := (2, 3)
def vector_vD : ℝ × ℝ := (-3, 2)

def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

theorem not_perpendicular_to_vA : dot_product vector_a vector_vA ≠ 0 := by sorry
theorem not_perpendicular_to_vB : dot_product vector_a vector_vB ≠ 0 := by sorry
theorem not_perpendicular_to_vD : dot_product vector_a vector_vD ≠ 0 := by sorry

end not_perpendicular_to_vA_not_perpendicular_to_vB_not_perpendicular_to_vD_l138_138776


namespace Brians_trip_distance_l138_138788

theorem Brians_trip_distance (miles_per_gallon : ℕ) (gallons_used : ℕ) (distance_traveled : ℕ) 
  (h1 : miles_per_gallon = 20) (h2 : gallons_used = 3) : 
  distance_traveled = 60 :=
by
  sorry

end Brians_trip_distance_l138_138788


namespace positive_number_property_l138_138760

theorem positive_number_property (x : ℝ) (h : x > 0) (hx : (x / 100) * x = 9) : x = 30 := by
  sorry

end positive_number_property_l138_138760


namespace geometry_problem_l138_138959

-- Definitions for the problem
variables {A B C A1 C1 K L M : Point}
variables (hABCacute : acute triangle A B C)
variables (hScaleneABC : scalene triangle A B C)
variables (hAltitudes : is_altitude A A1 B C ∧ is_altitude C C1 A B)
variables (hMidpoints : is_midpoint K A B ∧ is_midpoint L B C ∧ is_midpoint M C A)
variables (hGiven : ∠ C1 M A1 = ∠ A B C)

-- Statement to prove
theorem geometry_problem (hABCacute : acute (triangle A B C))
                         (hScaleneABC : scalene (triangle A B C))
                         (hAltitudes : is_altitude A A1 B C ∧ is_altitude C C1 A B)
                         (hMidpoints : is_midpoint K A B ∧ is_midpoint L B C ∧ is_midpoint M C A)
                         (hGiven : ∠ C1 M A1 = ∠ A B C) :
                         dist C1 K = dist A1 L :=
begin
  sorry
end

end geometry_problem_l138_138959


namespace projection_vector_l138_138880

variables {V : Type*} [InnerProductSpace ℝ V]

theorem projection_vector (a b : V) (h₁ : ∥a∥ = 2) (h₂ : ∥b∥ = 3) (h₃ : ⟪a, b⟫ = 1) :
  (inner b a / ∥a∥ ^ 2) • a = (1 / 4) • a :=
by
  sorry

end projection_vector_l138_138880


namespace isosceles_trapezoid_circle_tangent_l138_138949

theorem isosceles_trapezoid_circle_tangent 
  (x : ℝ) (A B C D : ℝ) (AB CD AD BC : ℝ)
  (isosceles : AB = 80 ∧ CD = 17)
  (sides_equal : AD = x ∧ BC = x)
  (circle_tangent : ∀ M, M ∈ segment[A, B] → M ∈ circle_tangent_AD_BC): 
  x^2 = 1940 := 
sorry

end isosceles_trapezoid_circle_tangent_l138_138949


namespace Sam_has_38_dollars_l138_138390

theorem Sam_has_38_dollars (total_money erica_money sam_money : ℕ) 
  (h1 : total_money = 91)
  (h2 : erica_money = 53) 
  (h3 : total_money = erica_money + sam_money) : 
  sam_money = 38 := 
by 
  sorry

end Sam_has_38_dollars_l138_138390


namespace dividing_by_10_l138_138141

theorem dividing_by_10 (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 :=
by
  sorry

end dividing_by_10_l138_138141


namespace probability_factor_of_36_l138_138488

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138488


namespace find_pairs_l138_138971

noncomputable theory

def A : Set ℕ := {n | 0 ≤ n}

def f : ℕ → ℕ
| 1 => 1
| n => if n % 2 = 0 then 3 * f (n / 2) else 3 * f ((n - 1) / 2) + 1

axiom f_prop1 (n : ℕ) : 3 * f(n) * f(2*n + 1) = f(2*n) * (1 + 3 * f(n))
axiom f_prop2 (n : ℕ) : f(2*n) < 6 * f(n)

theorem find_pairs :
  {(5, 47), (7, 45), (13, 39), (15, 37)} =
  { (k, l) : ℕ × ℕ | f(k) + f(l) = 293 ∧ k < l } :=
by
  sorry

end find_pairs_l138_138971


namespace probability_factor_36_l138_138542

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138542


namespace simplify_fraction_l138_138397

noncomputable def simplify_complex_fraction (a b c d : ℤ) (i : ℂ) (h : i^2 = -1) : ℂ :=
  let num := (a + b * i) * (c - d * i)
  let denom := (c + d * i) * (c - d * i)
  num / denom

theorem simplify_fraction : simplify_complex_fraction 3 (-4) 5 2 (complex.I) (by norm_num[complex.I]) = complex.mk (7/29) (-14/29) :=
sorry

end simplify_fraction_l138_138397


namespace quadrilateral_BFGC_area_l138_138732

-- Define the properties of the square and the geometry involved
def square_side_length : ℝ := 3

-- Define points based on the problem
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (square_side_length, 0)
def C : ℝ × ℝ := (square_side_length, square_side_length)
def D : ℝ × ℝ := (0, square_side_length)
def E : ℝ × ℝ := (2 * square_side_length, 0)
def F : ℝ × ℝ := (square_side_length / 2, 0)
def G : ℝ × ℝ := (0, 0) -- Intersection of FE and AD

-- Define the function to calculate the area of quadrilateral BFGC
noncomputable def area_of_quadrilateral_BFGC : ℝ :=
  let base1 := (B.1 - F.1) in
  let base2 := (C.1 - G.1) in
  let height := square_side_length in
  1/2 * (base1 + base2) * height

-- The main theorem stating the area of quadrilateral BFGC is 6.75 square cm
theorem quadrilateral_BFGC_area :
  area_of_quadrilateral_BFGC = 6.75 :=
by
  unfold area_of_quadrilateral_BFGC
  norm_num
  sorry

end quadrilateral_BFGC_area_l138_138732


namespace product_of_dice_divisible_by_9_l138_138454

-- Define the probability of rolling a number divisible by 3
def prob_roll_div_by_3 : ℚ := 1/6

-- Define the probability of rolling a number not divisible by 3
def prob_roll_not_div_by_3 : ℚ := 2/3

-- Define the probability that the product of numbers rolled on 6 dice is divisible by 9
def prob_product_div_by_9 : ℚ := 449/729

-- Main statement of the problem
theorem product_of_dice_divisible_by_9 :
  (1 - ((prob_roll_not_div_by_3^6) + 
        (6 * prob_roll_div_by_3 * (prob_roll_not_div_by_3^5)) + 
        (15 * (prob_roll_div_by_3^2) * (prob_roll_not_div_by_3^4)))) = prob_product_div_by_9 :=
by {
  sorry
}

end product_of_dice_divisible_by_9_l138_138454


namespace find_q_l138_138171

open Real

theorem find_q (q : ℝ) (a : ℕ → ℝ) (S : ℕ → ℝ) (h₁ : ∀ n : ℕ, n > 0 → a n = q^(n - 1)) (h₂ : ∀ n : ℕ, n > 0 → S n = (1 - q^n) / (1 - q)) (h₃ : tendsto (λ n, (S n) / (a (n + 1))) at_top (𝓝 (1/2)))
: q = 3 :=
begin
  sorry
end

end find_q_l138_138171


namespace units_digit_of_odd_product_between_20_130_l138_138078

-- Define the range and the set of odd integers within the specified range.
def odd_ints_between (a b : ℕ) : Set ℕ := { n | a < n ∧ n < b ∧ n % 2 = 1}

-- Define the product of all elements in a set.
def product (s : Set ℕ) : ℕ := s.toFinset.prod id

-- The main theorem to prove that the units digit of our specific product is 5.
theorem units_digit_of_odd_product_between_20_130 : 
  Nat.unitsDigit (product (odd_ints_between 20 130)) = 5 :=
  sorry

end units_digit_of_odd_product_between_20_130_l138_138078


namespace polynomial_int_values_l138_138383

theorem polynomial_int_values (x : ℤ) :
  let P : ℤ → ℚ := λ x,
    (1/630) * (x:ℚ)^(0 : ℕ) -
    (1/21) * (x:ℚ)^(7 : ℕ) +
    (13/30) * (x:ℚ)^(5 : ℕ) - 
    (82/63) * (x:ℚ)^(3 : ℕ) +
    (32/35) * (x:ℚ)^(1 : ℕ)
  in P x ∈ ℤ :=
by {
  sorry
}

end polynomial_int_values_l138_138383


namespace product_of_two_primes_is_good_l138_138287

-- Definition of a good number
def good_number (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ n = a * b ∧ (∀ c ≥ 2, n ≠ c * (n / c))

theorem product_of_two_primes_is_good (n : ℕ) :
  (∃ p q : ℕ, prime p ∧ prime q ∧ n = p * q) → good_number n :=
by
  sorry

end product_of_two_primes_is_good_l138_138287


namespace third_term_arithmetic_sequence_l138_138298

theorem third_term_arithmetic_sequence (a x : ℝ) 
  (h : 2 * a + 2 * x = 10) : a + x = 5 := 
by
  sorry

end third_term_arithmetic_sequence_l138_138298


namespace perpendicular_vectors_l138_138274

variables (k : ℝ)
def a : ℝ × ℝ := (real.sqrt 3, 0)
def b : ℝ × ℝ := (0, -1)
def c : ℝ × ℝ := (k, real.sqrt 3)
def a_minus_2b : ℝ × ℝ := (a.1 - 2 * b.1, a.2 - 2 * b.2)

theorem perpendicular_vectors : ((a_minus_2b.1 * c.1 + a_minus_2b.2 * c.2) = 0) → k = -2 := by
  sorry

end perpendicular_vectors_l138_138274


namespace probability_factor_of_36_l138_138573

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138573


namespace probability_factor_36_l138_138592

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138592


namespace probability_divisor_of_36_l138_138662

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138662


namespace probability_factor_of_36_l138_138572

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138572


namespace probability_factor_of_36_l138_138680

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138680


namespace houses_in_lawrence_county_l138_138786

theorem houses_in_lawrence_county 
  (houses_before_boom : ℕ := 1426) 
  (houses_built_during_boom : ℕ := 574) 
  : houses_before_boom + houses_built_during_boom = 2000 := 
by 
  sorry

end houses_in_lawrence_county_l138_138786


namespace sum_first_60_natural_numbers_l138_138720

theorem sum_first_60_natural_numbers : (60 * (60 + 1)) / 2 = 1830 := by
  sorry

end sum_first_60_natural_numbers_l138_138720


namespace negation_of_p_l138_138869

variable (x : ℝ)

def proposition_p : Prop := ∀ x > 2, log 2 (x + 4 / x) > 2

theorem negation_of_p : ¬ (∃ x > 2, log 2 (x + 4 / x) ≤ 2) := sorry

end negation_of_p_l138_138869


namespace distance_MK_l138_138041

theorem distance_MK (CK CM : ℝ) (h1 : CK = 9.6) (h2 : CM = 28) : 
  let MK := Real.sqrt (CK^2 + CM^2) in
  MK = 29.6 :=
by
  sorry

end distance_MK_l138_138041


namespace probability_divisor_of_36_l138_138653

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138653


namespace train_crossing_time_l138_138135

/-- Define constants for distance and speed -/
def distance : ℝ := 300
def speed : ℝ := 53.99999999999999

/-- Define the time formula -/
def time (d : ℝ) (s : ℝ) : ℝ := d / s

/-- State that the time it takes approximately equals 5.56 seconds -/
theorem train_crossing_time :
  abs (time distance speed - 5.56) < 0.01 := sorry

end train_crossing_time_l138_138135


namespace initial_food_days_l138_138060

theorem initial_food_days (x : ℕ) (h : 760 * (x - 2) = 3040 * 5) : x = 22 := by
  sorry

end initial_food_days_l138_138060


namespace probability_factor_of_36_l138_138483

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138483


namespace choose_three_sum_multiple_of_three_l138_138792

theorem choose_three_sum_multiple_of_three : 
  ∃ (n : ℕ), n = 42 ∧ (
    ∃ (A B C : Set ℕ), 
    A = {0, 3, 6, 9} ∧ 
    B = {1, 4, 7} ∧ 
    C = {2, 5, 8} ∧ 
    (
      (
        (A.card.choose 3) + (B.card.choose 3) + (C.card.choose 3) + 
        (A.card.choose 1 * B.card.choose 1 * C.card.choose 1) = n
      )
    )
  ) := sorry

end choose_three_sum_multiple_of_three_l138_138792


namespace find_largest_x_l138_138177

theorem find_largest_x : 
  ∃ x : ℝ, (4 * x ^ 3 - 17 * x ^ 2 + x + 10 = 0) ∧ 
           (∀ y : ℝ, 4 * y ^ 3 - 17 * y ^ 2 + y + 10 = 0 → y ≤ x) ∧ 
           x = (25 + Real.sqrt 545) / 8 :=
sorry

end find_largest_x_l138_138177


namespace probability_factor_of_36_l138_138644

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138644


namespace handshake_count_l138_138439

theorem handshake_count :
  let total_people := 5 * 4
  let handshakes_per_person := total_people - 1 - 3
  let total_handshakes_with_double_count := total_people * handshakes_per_person
  let total_handshakes := total_handshakes_with_double_count / 2
  total_handshakes = 160 :=
by
-- We include "sorry" to indicate that the proof is not provided.
sorry

end handshake_count_l138_138439


namespace probability_factor_of_36_l138_138643

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138643


namespace symmetry_of_f_min_value_of_f_l138_138866

-- For any positive integer n, the graph of f(x) is symmetric about the line x = π / 4
theorem symmetry_of_f 
  (n : ℕ) (h : n > 0) : 
  ∀ x : ℝ, f(π / 4 - x) = f(x) :=
sorry

-- When n = 3, the minimum value of f(x) on [0, π / 2] is √2 / 2
theorem min_value_of_f 
  (n : ℕ) (h : n = 3) : 
  ∃ x ∈ Set.Icc (0 : ℝ) (π / 2), f x = sqrt 2 / 2 :=
sorry

-- Define f
def f (n : ℕ) (x : ℝ) : ℝ := (Real.sin x)^n + (Real.cos x)^n

end symmetry_of_f_min_value_of_f_l138_138866


namespace probability_log_floor_difference_l138_138968

def fractional_part (x : ℝ) : ℝ := x - (⌊x⌋ : ℝ)

noncomputable def log_base2 := real.logb 2

theorem probability_log_floor_difference :
  ∀ (y : ℝ), (0 < y ∧ y < 1) →
  ℙ(y, (log_base2 (3 * y) - log_base2 y).floor = 1) = 3/5 := 
by
  sorry

end probability_log_floor_difference_l138_138968


namespace largest_N_satisfying_cond_l138_138998

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138998


namespace length_relation_l138_138407

variables (C1 C2 : Type) [metric_space C1] [metric_space C2]
          (O1 O2 A B E F M N : C1)
          [is_center O1 C1] [is_center O2 C2]
          (intersects : exists (A B : C1), on_circle A C1 ∧ on_circle B C1 ∧ on_circle A C2 ∧ on_circle B C2)
          (intersect_radii_1 : O1 ∈ line_segment (B : C1) F)
          (intersect_radii_2 : O2 ∈ line_segment (B : C1) E)
          (line_parallel : is_parallel B (line_segment (E : C1) F))
          (intersects_at : exists (M N : C1), on_circle M C1 ∧ on_circle N C2)

-- Goal to prove:
theorem length_relation : distance M N = distance A E + distance A F :=
sorry

end length_relation_l138_138407


namespace max_value_is_correct_l138_138964

noncomputable def max_value_inequality 
  (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a^2 + b^2 + c^2 = 1) : ℝ :=
  2 * a * b + 2 * b * c * real.sqrt 2 + 2 * a * c

theorem max_value_is_correct
  (a b c : ℝ) (h₀ : a ≥ 0) (h₁ : b ≥ 0) (h₂ : c ≥ 0) (h₃ : a^2 + b^2 + c^2 = 1) :
  max_value_inequality a b c h₀ h₁ h₂ h₃ ≤ (2 * (1 + real.sqrt 2) / 3) :=
sorry

end max_value_is_correct_l138_138964


namespace probability_factor_of_36_l138_138648

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138648


namespace james_problem_l138_138931

def probability_at_least_two_green_apples (total: ℕ) (red: ℕ) (green: ℕ) (yellow: ℕ) (choices: ℕ) : ℚ :=
  let favorable_outcomes := (Nat.choose green 2) * (Nat.choose (total - green) 1) + (Nat.choose green 3)
  let total_outcomes := Nat.choose total choices
  favorable_outcomes / total_outcomes

theorem james_problem : probability_at_least_two_green_apples 10 5 3 2 3 = 11 / 60 :=
by sorry

end james_problem_l138_138931


namespace probability_factor_of_36_l138_138687

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138687


namespace liam_marble_boxes_l138_138365

theorem liam_marble_boxes (m : ℕ) (h1 : 360 % m = 0) (h2 : m > 1) : 
  (∀ k : ℕ, (m = k) → 360 % k = 0 ∧ (360 / k) % 2 = 0) ↔ m ∈ {2, 6, 10, 18, 30, 90, 4, 12, 20, 36, 60, 180, 8, 24, 40, 72, 120} :=
by
  sorry

end liam_marble_boxes_l138_138365


namespace sum_of_k_values_l138_138895

theorem sum_of_k_values (k : ℤ) (x : ℝ) : 
  (k - 2 * x = 3 * (k - 2) ∧ x ≥ 0) ∧ 
  (x - 2 * (x - 1) ≤ 3) ∧ 
  (2 * k + x) / 3 ≥ x → 
  ∃ k : ℤ, k ∈ [2, 3] ∧ ∑ i in [2, 3], i = 5 := 
by
  sorry

end sum_of_k_values_l138_138895


namespace probability_factor_of_36_l138_138691

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138691


namespace probability_factor_of_36_l138_138538

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138538


namespace angle_MDB_is_ninety_degrees_l138_138332

open Locale Classical
open Triangle

-- Define the problem conditions
def triangle_ABC (A B C : Point) : Prop :=
  AB = 3 * BC ∧
  M = midpoint AB ∧
  is_angle_bisector B D ∠ ABC

-- Define the theorem to be proved
theorem angle_MDB_is_ninety_degrees (A B C D M : Point) :
  triangle_ABC A B C →
  is_midpoint M A B →
  is_angle_bisector D B (∠ ABC) →
  ∠ MDB = 90 :=
by
  sorry

end angle_MDB_is_ninety_degrees_l138_138332


namespace part1_part2_l138_138268

theorem part1 (x : ℝ) : -5 * x^2 + 3 * x + 2 > 0 ↔ -2/5 < x ∧ x < 1 :=
by sorry

theorem part2 (a x : ℝ) (h : 0 < a) :
  (a * x^2 + (a + 3) * x + 3 > 0 ↔
    (
      (0 < a ∧ a < 3 ∧ (x < -3/a ∨ x > -1)) ∨
      (a = 3 ∧ x ≠ -1) ∨
      (a > 3 ∧ (x < -1 ∨ x > -3/a))
    )
  ) :=
by sorry

end part1_part2_l138_138268


namespace evaluate_expression_l138_138159

theorem evaluate_expression :
  (Real.sqrt 5 * 5^(1/2) + 20 / 4 * 3 - 8^(3/2) + 5) = 25 - 16 * Real.sqrt 2 := 
by
  sorry

end evaluate_expression_l138_138159


namespace largest_possible_N_l138_138985

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138985


namespace factor_probability_36_l138_138609

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138609


namespace remainder_of_sum_of_consecutive_days_l138_138829

theorem remainder_of_sum_of_consecutive_days :
  (100045 + 100046 + 100047 + 100048 + 100049 + 100050 + 100051 + 100052) % 5 = 3 :=
by
  sorry

end remainder_of_sum_of_consecutive_days_l138_138829


namespace central_angle_measure_l138_138236

-- Given conditions
def radius : ℝ := 2
def area : ℝ := 4

-- Central angle α
def central_angle : ℝ := 2

-- Theorem statement: The central angle measure is 2 radians
theorem central_angle_measure :
  ∃ α : ℝ, α = central_angle ∧ area = (1/2) * (α * radius) := 
sorry

end central_angle_measure_l138_138236


namespace necessary_but_not_sufficient_l138_138272

open Set

namespace Mathlib

noncomputable def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
noncomputable def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem necessary_but_not_sufficient (a : ℝ) : 
  (a ∈ M → a ∈ N) ∧ ¬(a ∈ N → a ∈ M) :=
by
  sorry

end Mathlib

end necessary_but_not_sufficient_l138_138272


namespace composite_numbers_condition_l138_138145

open Nat

-- Definition of the problem
def proper_divisors (n : ℕ) : List ℕ := (range n).filter (λ d, d > 1 ∧ d ∣ n)

def next_numbers_are_divisors (n m : ℕ) : Prop :=
  proper_divisors n |>.map (λ d, d + 1) |>.all (λ x, x ∣ m ∧ x < m)

-- Conditional statement to be proven
theorem composite_numbers_condition (n : ℕ) :
  (2 < n ∧ n ∣ 2^(log2 n) ∧ ∃ k, 2^k = n) →
  (next_numbers_are_divisors n 9 ∨ next_numbers_are_divisors n 15) →
  n = 4 ∨ n = 8 :=
by
  sorry

end composite_numbers_condition_l138_138145


namespace probability_factor_of_36_l138_138700

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138700


namespace amount_paid_is_correct_l138_138934

-- Define the conditions
def time_painting_house : ℕ := 8
def time_fixing_counter := 3 * time_painting_house
def time_mowing_lawn : ℕ := 6
def hourly_rate : ℕ := 15

-- Define the total time worked
def total_time_worked := time_painting_house + time_fixing_counter + time_mowing_lawn

-- Define the total amount paid
def total_amount_paid := total_time_worked * hourly_rate

-- Formalize the goal
theorem amount_paid_is_correct : total_amount_paid = 570 :=
by
  -- Proof steps to be filled in
  sorry  -- Placeholder for the proof

end amount_paid_is_correct_l138_138934


namespace average_production_last_5_days_l138_138721

theorem average_production_last_5_days
  (avg_first_25_days : ℕ → ℕ → ℕ → ℕ → Prop)
  (avg_monthly : ℕ)
  (total_days : ℕ)
  (days_first_period : ℕ)
  (avg_production_first_period : ℕ)
  (avg_total_monthly : ℕ)
  (days_second_period : ℕ)
  (total_production_five_days : ℕ):
  (days_first_period = 25) →
  (avg_production_first_period = 50) →
  (avg_total_monthly = 48) →
  (total_production_five_days = 190) →
  (days_second_period = 5) →
  avg_first_25_days days_first_period avg_production_first_period 
  (days_first_period * avg_production_first_period) avg_total_monthly ∧
  avg_monthly = avg_total_monthly →
  ((days_first_period + days_second_period) * avg_monthly - 
  days_first_period * avg_production_first_period = total_production_five_days) →
  (total_production_five_days / days_second_period = 38) := sorry

end average_production_last_5_days_l138_138721


namespace total_number_of_ways_to_choose_courses_l138_138128

theorem total_number_of_ways_to_choose_courses :
  (∑ b in finset.range 3, nat.choose 2 b * nat.choose 4 (4 - b)) - nat.choose 4 4 = 14 := 
by {
  have sum_cases := (nat.choose 2 1 * nat.choose 4 3) + (nat.choose 2 2 * nat.choose 4 2),
  have subtract_for_no_b := sum_cases - nat.choose 4 4,
  exact subtract_for_no_b,
  sorry
}

end total_number_of_ways_to_choose_courses_l138_138128


namespace max_value_of_f_l138_138208

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l138_138208


namespace expected_value_N_given_S_2_l138_138716

noncomputable def expected_value_given_slip_two (P_N_n : ℕ → ℝ) : ℝ :=
  ∑' n, n * (2^(-n) / (n * (Real.log 2 - 0.5)))

theorem expected_value_N_given_S_2 (P_N_n : ℕ → ℝ) (h : ∀ n, P_N_n n = 2^(-n)) :
  expected_value_given_slip_two P_N_n = 1 / (2 * Real.log 2 - 1) :=
by
  sorry

end expected_value_N_given_S_2_l138_138716


namespace probability_divisor_of_36_is_one_fourth_l138_138471

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138471


namespace smallest_n_l138_138701

theorem smallest_n (n : ℕ) : 17 * n ≡ 136 [MOD 5] → n = 3 := 
by sorry

end smallest_n_l138_138701


namespace probability_factor_of_36_l138_138578

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138578


namespace factor_probability_36_l138_138615

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138615


namespace min_female_students_participated_l138_138110

theorem min_female_students_participated 
  (male_students : ℕ) (female_students : ℕ) 
  (total_students : ℕ) (participants : ℕ) 
  (male_students = 22) (female_students = 18) 
  (total_students = male_students + female_students) 
  (participants = total_students * 60 / 100) 
  (participants = 24) :
  ∃ (female_participants : ℕ), female_participants ≥ 2 :=
by 
  use 2
  sorry

end min_female_students_participated_l138_138110


namespace power_mod_five_l138_138072

theorem power_mod_five (n : ℕ) (hn : n ≡ 0 [MOD 4]): (3^2000 ≡ 1 [MOD 5]) :=
by 
  sorry

end power_mod_five_l138_138072


namespace probability_divisor_of_36_is_one_fourth_l138_138467

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138467


namespace find_total_time_l138_138184

def painting_problem (t : ℝ) : Prop :=
  let rate_doug := 1 / 6
  let rate_dave := 1 / 8
  let rate_ralph := 1 / 12
  let combined_rate := rate_doug + rate_dave + rate_ralph
  (combined_rate * (t - 1.5) = 1)

theorem find_total_time (t : ℝ) (h : painting_problem t) : Prop :=
  h = (1 / 6 + 1 / 8 + 1 / 12) * (t - 1.5) = 1

#check find_total_time

end find_total_time_l138_138184


namespace cos_double_angle_l138_138916

-- Definition of the terminal condition
def terminal_side_of_angle (α : ℝ) (x y : ℝ) : Prop :=
  (x = 1) ∧ (y = Real.sqrt 3) ∧ (x^2 + y^2 = 1)

-- Prove the required statement
theorem cos_double_angle (α : ℝ) :
  (terminal_side_of_angle α 1 (Real.sqrt 3)) →
  Real.cos (2 * α + Real.pi / 2) = - Real.sqrt 3 / 2 :=
by
  sorry

end cos_double_angle_l138_138916


namespace find_m_value_l138_138359

noncomputable def find_value_of_m : Prop :=
  ∃ (x1 x2 m : ℝ), 
    (x1 + x2 = 3) ∧ 
    (x1 * x2 = m) ∧ 
    (x1 + x2 - x1 * x2 = 1) ∧
    (m = 2)

theorem find_m_value : find_value_of_m :=
begin
  sorry
end

end find_m_value_l138_138359


namespace probability_factor_of_36_l138_138481

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138481


namespace triangle_third_side_bounds_l138_138847

theorem triangle_third_side_bounds (a b : ℝ) (h₁ : a = 7) (h₂ : b = 11) (c : ℝ) :
  5 ≤ c ∧ c ≤ 17 :=
by
  have ha : a > 0 := by linarith
  have hb : b > 0 := by linarith
  have habc₁ : a + b > c := by linarith [h₁, h₂]
  have habc₂ : a + c > b := by linarith [h₁]
  have habc₃ : b + c > a := by linarith [h₂]
  sorry

end triangle_third_side_bounds_l138_138847


namespace quadratic_distinct_roots_l138_138928

theorem quadratic_distinct_roots
  (a b c : ℝ)
  (h1 : 5 * a + 3 * b + 2 * c = 0)
  (h2 : a ≠ 0) :
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1 ^ 2 + b * x1 + c = 0) ∧ (a * x2 ^ 2 + b * x2 + c = 0) :=
by
  sorry

end quadratic_distinct_roots_l138_138928


namespace factor_probability_l138_138627

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138627


namespace factor_probability_36_l138_138612

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138612


namespace probability_factor_of_36_l138_138585

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138585


namespace probability_factor_36_l138_138543

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138543


namespace base_conversion_subtraction_l138_138814

theorem base_conversion_subtraction :
  let n1 := 3 * (9^2) + 2 * (9^1) + 5 * (9^0),
      n2 := 2 * (6^2) + 4 * (6^1) + 3 * (6^0)
  in
  n1 - n2 = 167 :=
by
  let n1 := 3 * (9^2) + 2 * (9^1) + 5 * (9^0)
  let n2 := 2 * (6^2) + 4 * (6^1) + 3 * (6^0)
  have : n1 = 266 := rfl
  have : n2 = 99 := rfl
  show n1 - n2 = 167, by
    rw [this, this]
    exact rfl

end base_conversion_subtraction_l138_138814


namespace coin_distribution_l138_138001

theorem coin_distribution (n : ℕ) (c : ℕ → ℕ) (h1 : n > 3)
  (h2 : ∀ i, 0 ≤ c i) 
  (h3 : ∑ i in finset.range n, c i = n) 
  (h4 : ∑ i in finset.range n, (i + 1) * c (i + 1) % n = (n * (n + 1) / 2) % n) : 
  ∃ (steps : ℕ), ∀ i ∈ finset.range n, (c' : ℕ → ℕ) → 
  (∀ j, c' j = if j = 0 then c j else c (j - 1) - 2 + (c j + 1) + (c (j + 1) + 1)) →
  ((steps > 0 ∧ ∀ i, 0 ≤ c' i ∧ c' i ≤ n - 2) ∧ 
  (∀ sum in finset.range n, c' sum = 1)) :=
sorry

end coin_distribution_l138_138001


namespace most_probable_dissatisfied_expected_dissatisfied_variance_dissatisfied_l138_138095

-- Define our assumptions
def passengers (n : ℕ) := 2 * n
def meal_pref (p : ℝ) := 0.5
def chicken_meals (n : ℕ) := n
def fish_meals (n : ℕ) := n

-- Part (a) statement: The most probable number of dissatisfied passengers is 1.
theorem most_probable_dissatisfied (n : ℕ) : 
  ∃ k : ℕ, k = 1 ∧ (2 * choose (2 * n) (n - k) * (1 / 2 ^ (2 * n))) > ∀ m ≠ k, (2 * choose (2 * n) (n - m) * (1 / 2 ^ (2 * n))) := 
sorry

-- Part (b) statement: The expected number of dissatisfied passengers.
theorem expected_dissatisfied (n : ℕ) : 
  ∃ E : ℝ, E = 0.564 * (real.sqrt n) :=
sorry

-- Part (c) statement: The variance of the number of dissatisfied passengers.
theorem variance_dissatisfied (n : ℕ) : 
  ∃ V : ℝ, V = 0.182 * n :=
sorry

end most_probable_dissatisfied_expected_dissatisfied_variance_dissatisfied_l138_138095


namespace probability_divisor_of_36_is_one_fourth_l138_138472

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138472


namespace pell_solution_unique_l138_138967

theorem pell_solution_unique 
  (x_0 y_0 x y : ℤ) 
  (h_fundamental : x_0^2 - 2003 * y_0^2 = 1)
  (h_pos_x : 0 < x) 
  (h_pos_y : 0 < y)
  (h_prime_div : ∀ p, Prime p → p ∣ x → p ∣ x_0) :
  x^2 - 2003 * y^2 = 1 → (x, y) = (x_0, y_0) :=
sorry

end pell_solution_unique_l138_138967


namespace arithmetic_seq_third_term_l138_138300

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end arithmetic_seq_third_term_l138_138300


namespace find_S_lambda_range_l138_138238

open Nat

-- Define the sequence {a_n} and the sum of the first n terms S_n
def a : ℕ → ℚ
| 0 => 1 / 2
| n + 1 => sorry  -- since a_{n+1} depended on the calculation from S_n

def S : ℕ → ℚ
| 0 => 1 / 2
| n + 1 => S n^2 / (3 * S n + 1)   -- from given conditions

noncomputable def T (n : ℕ) : ℚ :=
  ∑ k in range (n + 1), (2 ^ (k + 1)) * (n + 1) / (k + 1)

theorem find_S (n : ℕ) : 
  S n = 1 / (n + 1) := sorry

theorem lambda_range (λ : ℚ) : 
  (∀ n : ℕ, λ * T n ≤ (n ^ 2 + 9) * 2 ^ (n + 1)) → 
  λ ∈ set.Icc (-∞ : ℚ) 3 := sorry

end find_S_lambda_range_l138_138238


namespace probability_factor_of_36_l138_138563

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138563


namespace forty_percent_more_than_seventyfive_by_fifty_l138_138102

def number : ℝ := 312.5

theorem forty_percent_more_than_seventyfive_by_fifty 
    (x : ℝ) 
    (h : 0.40 * x = 0.75 * 100 + 50) : 
    x = number :=
by
  sorry

end forty_percent_more_than_seventyfive_by_fifty_l138_138102


namespace right_triangle_side_length_l138_138458

theorem right_triangle_side_length (c a b : ℕ) (h1 : c = 5) (h2 : a = 3) (h3 : c^2 = a^2 + b^2) : b = 4 :=
  by
  sorry

end right_triangle_side_length_l138_138458


namespace probability_factor_36_l138_138520

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138520


namespace second_term_binomial_expansion_l138_138054

theorem second_term_binomial_expansion : 
  ∀ (x : ℝ), (x - (2 / x))^6 = 
  (1 : ℝ).choose 0 * x^6 * (-(2 / x))^0 + (6).choose 1 * x^5 * (-(2 / x)) + 
  ∑ i in finset.range 5, (6).choose (i + 2) * x^(6 - (i + 2)) * (-(2 / x))^(i + 2) := 
by
  sorry

end second_term_binomial_expansion_l138_138054


namespace probability_factor_36_l138_138510

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138510


namespace probability_factor_of_36_l138_138685

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138685


namespace incircle_radius_equal_l138_138313

theorem incircle_radius_equal (ABC : Triangle) (M : Point)
  (O : Point) (MO_line : Line)
  (r : ℝ)
  (inscribed_circle : Circle)
  (H : Point)
  (E : Point) :
  is_scalene_triangle ABC →
  is_midpoint M ABC.BC →
  is_incenter O inscribed_circle →
  passes_through M O MO_line →
  intersects_at MO_line (altitude H ABC A) E →
  radius inscribed_circle = r →
  distance A E = r := 
sorry

end incircle_radius_equal_l138_138313


namespace min_value_inverse_sum_of_chord_l138_138293

noncomputable theory
open Real

def radius_of_circle (h k : ℝ) : ℝ := 2

def length_of_chord (a b : ℝ) (h k : ℝ) : ℝ :=
  if (a > 0 ∧ b > 0) then 4 else 0

def minimum_value_of_inverse_sum (a b : ℝ) :=
  if (a > 0 ∧ b > 0) then (3 / 2 + sqrt 2) else 0

theorem min_value_inverse_sum_of_chord 
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0) :
  (∃ h k : ℝ, radius_of_circle h k = 2 ∧ length_of_chord a b h k = 4) →
  (minimum_value_of_inverse_sum a b = 3 / 2 + sqrt 2) :=
by
  intros h k
  sorry

end min_value_inverse_sum_of_chord_l138_138293


namespace number_of_solutions_l138_138282

theorem number_of_solutions : 
  (card {p : ℤ × ℤ × ℤ | |p.1 + p.2.1| + p.2.2 = 21 ∧ p.1 * p.2.1 + |p.2.2| = 99} = 4) :=
by {
  sorry
}

end number_of_solutions_l138_138282


namespace length_of_BC_l138_138870

open Real

theorem length_of_BC (AD AC BD : ℝ) (hAD : AD = 45) (hAC : AC = 20) (hBD : BD = 52) :
  let AB := sqrt (BD^2 - AD^2) in
  let BC := sqrt (AB^2 + AC^2) in
  BC = sqrt 1079 :=
by
  have hBD_sq : BD^2 = 52^2 := sorry
  have hAD_sq : AD^2 = 45^2 := sorry
  have AB_eq : AB = sqrt (52^2 - 45^2) := sorry
  have AB_sq := by
    rw [AB_eq, sqrt_sq]
    sorry
  have AC_sq : AC^2 = 20^2 := sorry
  have BC_def : BC = sqrt (AB^2 + AC^2) := sorry
  have BC_eq : BC = sqrt (679 + 400) := sorry
  rw BC_eq
  norm_num
  exact rfl

end length_of_BC_l138_138870


namespace probability_factor_of_36_is_1_over_4_l138_138501

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138501


namespace arithmetic_sequence_a3_l138_138919

theorem arithmetic_sequence_a3 (a : ℕ → ℝ) (h : a 2 + a 4 = 8) (h_seq : a 2 + a 4 = 2 * a 3) :
  a 3 = 4 :=
by
  sorry

end arithmetic_sequence_a3_l138_138919


namespace bisects_CD_l138_138921

theorem bisects_CD
  (A B C D E P: Type)
  [is_cyclic_quad A B C D]
  (h1: AD^2 + BC^2 = AB^2)
  (h2: E = intersection (diagonals A B C D))
  (h3: P ∈ line A B)
  (h4: ∠ APD = ∠ BPC)
  :
  bisects_PE_CD P E C D :=
sorry

end bisects_CD_l138_138921


namespace common_integer_count_l138_138393

open Set
open Polynomial

/-- Define Set A as the set of integers from 3 to 30 inclusive -/
def SetA : Set ℤ := {i | 3 ≤ i ∧ i ≤ 30}

/-- Define Set B as the set of integers from 10 to 40 inclusive -/
def SetB : Set ℤ := {i | 10 ≤ i ∧ i ≤ 40}

/-- Define the condition polynomial f(i) = i^2 - 5i - 6 -/
def condition (i : ℤ) : Prop := eval i (X^2 - 5 * X - 6) = 0

/-- The number of distinct integers i that belong to both Set A and Set B and satisfy the condition equals 0 -/
theorem common_integer_count : 
  card {i | i ∈ SetA ∧ i ∈ SetB ∧ condition i} = 0 :=
sorry

end common_integer_count_l138_138393


namespace solution_set_xf_gt_zero_l138_138856

variables {f : ℝ → ℝ}

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f(-x) = -f(x)
def is_increasing (f : ℝ → ℝ) (S : set ℝ) : Prop := ∀ x y ∈ S, x < y → f(x) < f(y)

theorem solution_set_xf_gt_zero 
  (h_odd : is_odd f)
  (h_inc_neg : is_increasing f {x | x < 0})
  (h_f_neg3 : f (-3) = 0) :
  {x | x * f(x) > 0} = {x | x < -3} ∪ {x | x > 3} :=
sorry

end solution_set_xf_gt_zero_l138_138856


namespace mutually_exclusive_but_not_complementary_l138_138149

def E1 : Prop := "miss the target"
def E2 : Prop := "hit the target"
def E3 : Prop := "the number of rings hit is greater than 4"
def E4 : Prop := "the number of rings hit is not less than 5"

theorem mutually_exclusive_but_not_complementary :
  (number_of_mutually_exclusive_but_not_complementary E1 E2 E3 E4) = 2 :=
sorry

end mutually_exclusive_but_not_complementary_l138_138149


namespace area_of_square_with_diagonal_20_l138_138455

theorem area_of_square_with_diagonal_20 (d : ℝ) (h : d = 20) : (s : ℝ) (hs : d = s * Real.sqrt 2) (A : ℝ) (hA : A = s * s) : A = 200 :=
begin
  sorry
end

end area_of_square_with_diagonal_20_l138_138455


namespace interval_between_segments_systematic_sampling_l138_138447

theorem interval_between_segments_systematic_sampling 
  (total_students : ℕ) (sample_size : ℕ) 
  (h_total_students : total_students = 1000) 
  (h_sample_size : sample_size = 40):
  total_students / sample_size = 25 :=
by
  sorry

end interval_between_segments_systematic_sampling_l138_138447


namespace probability_divisor_of_36_l138_138663

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138663


namespace probability_factor_of_36_is_1_over_4_l138_138505

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138505


namespace fixed_point_on_line_through_AB_l138_138258

def ellipse (a b : ℝ) := ∀ (x y : ℝ), (x^2) / (a^2) + (y^2) / (b^2) = 1

def point_on_ellipse (x y a b : ℝ) : Prop := (x^2) / (a^2) + (y^2) / (b^2) = 1

def eccentricity_condition (a b : ℝ) : Prop := (a > b) ∧ (b > 0) ∧ ( (sqrt (a^2 - b^2)) / a = (sqrt 2) / 2 )

def intersection_line_through_fixed_point
  (a b : ℝ) (P : ℝ × ℝ) := 
  let line_through_p := λ (k m : ℝ), ∀ x y : ℝ, y = k * x + m in
  ∀ (A B : ℝ × ℝ), 
  A ≠ P ∧ B ≠ P ∧ 
  point_on_ellipse A.1 A.2 a b ∧ 
  point_on_ellipse B.1 B.2 a b ∧
  (line_through_p P.1 P.2).fst A.1 + B.1 = -4 * (A.2 + B.2) / 3  → 
  line_through_p A.1 A.2 = line_through_p (-(2 / 3)) (-(1 / 3))

theorem fixed_point_on_line_through_AB 
  (a b : ℝ) (P : ℝ × ℝ) 
  (h_cond : eccentricity_condition a b) 
  (h_point : P = (-2, 1)) : 
  intersection_line_through_fixed_point a b P :=
sorry

end fixed_point_on_line_through_AB_l138_138258


namespace longest_side_of_triangle_l138_138250

theorem longest_side_of_triangle (a d : ℕ) (h1 : d = 2) (h2 : a - d > 0) (h3 : a + d > 0)
    (h_angle : ∃ C : ℝ, C = 120) 
    (h_arith_seq : ∃ (b c : ℕ), b = a - d ∧ c = a ∧ b + 2 * d = c + d) : 
    a + d = 7 :=
by
  -- The proof will be provided here
  sorry

end longest_side_of_triangle_l138_138250


namespace bretschneiders_theorem_l138_138004

-- Definitions for lengths of edges and dihedral angles
variables (a b : ℝ) (alpha beta : ℝ)

-- Main theorem statement
theorem bretschneiders_theorem (a b : ℝ) (alpha beta : ℝ) :
  ∃ c : ℝ, a^2 + b^2 + 2 * a * b * (Real.cot alpha) * (Real.cot beta) = c :=
sorry

end bretschneiders_theorem_l138_138004


namespace maximizeRevenue_l138_138446

-- part 1
def costPrice : ℕ := 100
def subsidyPerPiece : ℕ := 20
def monthlySalesVolume (x : ℕ) : ℕ := -3 * x + 900
def totalMonthlySubsidy (x : ℕ) : ℕ := monthlySalesVolume x * subsidyPerPiece

example : totalMonthlySubsidy 160 = 8400 := by
  sorry

-- part 2
def profitPerPiece (x : ℕ) : ℕ := x - costPrice + subsidyPerPiece
def totalRevenue (x : ℕ) : ℕ := profitPerPiece x * monthlySalesVolume x

theorem maximizeRevenue (x : ℕ) : ∃ x, x = 190 ∧ totalRevenue x = 36300 := by
  sorry

end maximizeRevenue_l138_138446


namespace largest_possible_N_l138_138999

theorem largest_possible_N (N : ℕ) :
  let divisors := Nat.divisors N
  in (1 ∈ divisors) ∧ (N ∈ divisors) ∧ (divisors.length ≥ 3) ∧ (divisors[divisors.length - 3] = 21 * divisors[1]) → N = 441 := 
by
  sorry

end largest_possible_N_l138_138999


namespace Katie_cupcakes_l138_138222

theorem Katie_cupcakes (initial_cupcakes sold_cupcakes final_cupcakes : ℕ) (h1 : initial_cupcakes = 26) (h2 : sold_cupcakes = 20) (h3 : final_cupcakes = 26) :
  (final_cupcakes - (initial_cupcakes - sold_cupcakes)) = 20 :=
by
  sorry

end Katie_cupcakes_l138_138222


namespace min_value_of_f_l138_138825

def f (x : ℝ) : ℝ := |x - 1| + |x + 4| - 5

theorem min_value_of_f : ∃ (m : ℝ), m = 0 ∧ ∀ x : ℝ, f(x) ≥ m := by
  let m := 0
  exists m
  constructor
  · rfl
  · intro x
    sorry

end min_value_of_f_l138_138825


namespace equation_of_BC_l138_138136

-- Definitions
def point (α : Type) [Field α] := { p : prod α α // True }

def line_equation (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

-- Conditions
def A : ℝ × ℝ := (-4, 2)  -- Vertex A
def median1 (x y : ℝ) : Prop := line_equation 3 (-2) 2 x y  -- Equation of 1st median.
def median2 (x y : ℝ) : Prop := line_equation 3 5 (-12) x y  -- Equation of 2nd median.

noncomputable def equation_of_line_BC (bx by cx cy : ℝ) : Prop :=
  line_equation 2 1 (-8) (bx - cx) (by - cy)

-- Theorem to prove
theorem equation_of_BC (bx by cx cy : ℝ) (h1 : median1 4 0) (h2 : median2 4 0) (h3 : median1 2 4) (h4 : median2 2 4) :
  equation_of_line_BC 2 4 4 0 :=
by
  sorry

end equation_of_BC_l138_138136


namespace compare_numbers_l138_138807

def first_number : ℕ := (2014 ^ (2 ^ 2014)) - 1

def second_number : ℕ := (List.prod (List.map (λ k, 2014 ^ (2 ^ k) + 1) (List.range 2014))) + 1

theorem compare_numbers : first_number / second_number = 2013 := by
  -- Proof goes here
  sorry

end compare_numbers_l138_138807


namespace new_triangle_area_l138_138845

theorem new_triangle_area (a b : ℝ) (x y : ℝ) (hypotenuse : x = a ∧ y = b ∧ x^2 + y^2 = (a + b)^2) : 
    (3  * (1 / 2) * a * b) = (3 / 2) * a * b :=
by
  sorry

end new_triangle_area_l138_138845


namespace probability_factor_of_36_l138_138582

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138582


namespace solve_for_y_l138_138225

theorem solve_for_y (x y : ℝ) (h : 2 * x - y = 6) : y = 2 * x - 6 :=
by
  sorry

end solve_for_y_l138_138225


namespace ms_cole_students_l138_138370

theorem ms_cole_students (S6 S4 S7 : ℕ)
  (h1: S6 = 40)
  (h2: S4 = 4 * S6)
  (h3: S7 = 2 * S4) :
  S6 + S4 + S7 = 520 :=
by
  sorry

end ms_cole_students_l138_138370


namespace integral_solution_l138_138728

noncomputable def integral_problem (x : Real) : Real :=
  ∫ u in 0 .. x, (u^3 + 4*u^2 + 3*u + 2)/((u + 1)^2 * (u^2 + 1))

theorem integral_solution : 
  ∀ (x : Real), integral_problem x = -1/(x + 1) + (1/2) * log (x^2 + 1) + arctan x + C :=
sorry

end integral_solution_l138_138728


namespace factor_probability_l138_138631

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138631


namespace range_of_m_range_of_x_l138_138230

variable {a b m : ℝ}

-- Given conditions
axiom a_pos : a > 0
axiom b_pos : b > 0
axiom sum_eq_one : a + b = 1

-- Problem (I): Prove range of m
theorem range_of_m (h : ab ≤ m) : m ≥ 1 / 4 := by
  sorry

variable {x : ℝ}

-- Problem (II): Prove range of x
theorem range_of_x (h : 4 / a + 1 / b ≥ |2 * x - 1| - |x + 2|) : -2 ≤ x ∧ x ≤ 6 := by
  sorry

end range_of_m_range_of_x_l138_138230


namespace maximize_profit_l138_138408

def cost_price_A (x y : ℕ) := x = y + 20
def cost_sum_eq_200 (x y : ℕ) := x + 2 * y = 200
def linear_function (m n : ℕ) := m = -((1/2) : ℚ) * n + 90
def profit_function (w n : ℕ) : ℚ := (-((1/2) : ℚ) * ((n : ℚ) - 130)^2) + 1250

theorem maximize_profit
  (x y m n : ℕ)
  (hx : cost_price_A x y)
  (hsum : cost_sum_eq_200 x y)
  (hlin : linear_function m n)
  (hmaxn : 80 ≤ n ∧ n ≤ 120)
  : y = 60 ∧ x = 80 ∧ n = 120 ∧ profit_function 120 120 = 1200 := 
sorry

end maximize_profit_l138_138408


namespace probability_factor_36_l138_138512

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138512


namespace compute_ARML_l138_138958

open Real

theorem compute_ARML (A R M L : ℝ) (h_pos : 0 < A ∧ 0 < R ∧ 0 < M ∧ 0 < L)
  (h1 : log 10 (A * L) + log 10 (A * M) = 2)
  (h2 : log 10 (M * L) + log 10 (M * R) = 2)
  (h3 : log 10 (R * A) + log 10 (R * L) = 5) : A * R * M * L = 1000 :=
sorry

end compute_ARML_l138_138958


namespace probability_divisor_of_36_l138_138654

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138654


namespace fourth_number_in_first_set_88_l138_138045

theorem fourth_number_in_first_set_88 (x y : ℝ)
  (h1 : (28 + x + 70 + y + 104) / 5 = 67)
  (h2 : (50 + 62 + 97 + 124 + x) / 5 = 75.6) :
  y = 88 :=
by
  sorry

end fourth_number_in_first_set_88_l138_138045


namespace solve_for_k_l138_138202

theorem solve_for_k (p q : ℝ) (k : ℝ) (hpq : 3 * p^2 + 6 * p + k = 0) (hq : 3 * q^2 + 6 * q + k = 0) 
    (h_diff : |p - q| = (1 / 2) * (p^2 + q^2)) : k = -16 + 12 * Real.sqrt 2 ∨ k = -16 - 12 * Real.sqrt 2 :=
by
  sorry

end solve_for_k_l138_138202


namespace closest_point_on_line_l138_138827

theorem closest_point_on_line :
  ∀ (x y : ℝ), (4, -2) = (4, -2) →
    y = 3 * x - 1 →
    (∃ (p : ℝ × ℝ), p = (-0.5, -2.5) ∧ p = (-0.5, -2.5))
  := by
    -- The proof of the theorem goes here
    sorry

end closest_point_on_line_l138_138827


namespace largest_possible_N_l138_138989

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138989


namespace exists_dividing_line_l138_138838

theorem exists_dividing_line (points : Fin 1988 → ℝ × ℝ)
  (h_ncollinear : ∀ (p1 p2 p3 p4 : Fin 1988), p1 ≠ p2 → p1 ≠ p3 → p1 ≠ p4 → p2 ≠ p3 → p2 ≠ p4 → p3 ≠ p4 →
    ¬ collinear ({points p1, points p2, points p3, points p4} : Set (ℝ × ℝ)))
  (colors : Fin 1988 → Bool)
  (h_blue_count : (Finset.univ.filter (λ p, colors p = tt)).card = 1788)
  (h_red_count : (Finset.univ.filter (λ p, colors p = ff)).card = 200) :
  ∃ l : ℝ × ℝ → Prop, 
    (Finset.filter (λ p, l (points p)) Finset.univ).filter (λ p, colors p = tt).card = 894 ∧
    (Finset.filter (λ p, l (points p)) Finset.univ).filter (λ p, colors p = ff).card = 100 ∧
    (Finset.filter (λ p, ¬ l (points p)) Finset.univ).filter (λ p, colors p = tt).card = 894 ∧
    (Finset.filter (λ p, ¬ l (points p)) Finset.univ).filter (λ p, colors p = ff).card = 100 :=
  sorry

end exists_dividing_line_l138_138838


namespace sine_angle_sum_identity_l138_138097

theorem sine_angle_sum_identity :
  sin 13 * cos 17 + cos 13 * sin 17 = 1 / 2 := by
sor

end sine_angle_sum_identity_l138_138097


namespace shaded_area_of_joined_squares_l138_138781

theorem shaded_area_of_joined_squares:
  ∀ (a b : ℕ) (area_of_shaded : ℝ),
  (a = 6) → (b = 8) → 
  (area_of_shaded = (6 * 6 : ℝ) + (8 * 8 : ℝ) / 2) →
  area_of_shaded = 50.24 := 
by
  intros a b area_of_shaded h1 h2 h3
  -- skipping the proof for now
  sorry

end shaded_area_of_joined_squares_l138_138781


namespace find_x_l138_138286

open Real

theorem find_x 
  (x y : ℝ) 
  (hx_pos : 0 < x)
  (hy_pos : 0 < y) 
  (h_eq : 7 * x^2 + 21 * x * y = 2 * x^3 + 3 * x^2 * y) 
  : x = 7 := 
sorry

end find_x_l138_138286


namespace denmark_pizza_combinations_l138_138803

theorem denmark_pizza_combinations :
  (let cheese_options := 3
   let meat_options := 4
   let vegetable_options := 5
   let invalid_combinations := 1
   let total_combinations := cheese_options * meat_options * vegetable_options
   let valid_combinations := total_combinations - invalid_combinations
   valid_combinations = 59) :=
by
  sorry

end denmark_pizza_combinations_l138_138803


namespace probability_margo_paired_with_irma_l138_138901

theorem probability_margo_paired_with_irma (total_students : ℕ) (friends : ℕ) (other_students : ℕ) (irma : ℕ) :
  total_students = 40 → friends = 5 → other_students = total_students - 1 →
  irma = 1 → (1 : ℚ) / (other_students : ℚ) = 1 / 39 :=
by
  intro h_total_students h_friends h_other_students h_irma
  rw [h_total_students, h_friends, h_other_students, h_irma]
  norm_num

end probability_margo_paired_with_irma_l138_138901


namespace local_maximum_at_e_l138_138040

open Real

noncomputable def f (x : ℝ) : ℝ := (ln x) / x
noncomputable def f_prime (x : ℝ) : ℝ := (1 - ln x) / x^2

theorem local_maximum_at_e :
  is_local_max (λ x : ℝ, (ln x) / x) e :=
by {
  -- Proof would go here
  sorry
}

end local_maximum_at_e_l138_138040


namespace octal_to_binary_conversion_l138_138801

theorem octal_to_binary_conversion :
  ∃ b : ℕ, octal_to_decimal 127 = b ∧ decimal_to_binary b = 1010111 :=
by
  sorry

-- Supporting definitions that capture the concepts used in the problem
def octal_to_decimal (o : ℕ) : ℕ :=
  -- Implement the conversion of an octal number (represented as a natural number) to a decimal number
  sorry

def decimal_to_binary (d : ℕ) : ℕ :=
  -- Implement the conversion of a decimal number to a binary number (represented as a natural number)
  sorry

end octal_to_binary_conversion_l138_138801


namespace correct_statements_l138_138320

def population : ℕ := 240
def sample_size : ℕ := 40

def is_population (n : ℕ) : Prop := n = population
def is_individual (s : Type) : Prop := ∀ (x : s), true
def is_sample (s : Type) (students_measured : s) : Prop := ∃ (subset : s), ∃ (size : ℕ), size = sample_size ∧ size = 40
def is_sample_size (n : ℕ) : Prop := n = sample_size

theorem correct_statements :
  is_population 240 ∧
  is_individual (fin population) ∧
  is_sample (fin population) (fin sample_size) ∧
  is_sample_size 40 := 
by {
  sorry
}

end correct_statements_l138_138320


namespace grace_putting_down_mulch_hours_l138_138280

/-- Grace's earnings conditions and hours calculation in September. -/
theorem grace_putting_down_mulch_hours :
  ∃ h : ℕ, 
    6 * 63 + 11 * 9 + 9 * h = 567 ∧
    h = 10 :=
by
  sorry

end grace_putting_down_mulch_hours_l138_138280


namespace largest_possible_N_l138_138981

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138981


namespace num_permutations_eq_ten_fact_squared_l138_138362

open Nat

-- Define the problem
theorem num_permutations_eq_ten_fact_squared :
  ∃ (x : Fin 20 → Fin 21), 
    (∀ i j : Fin 20, i ≠ j → x i ≠ x j) ∧
    ∑ i : Fin 20, (| x i - ↑i | + | x i + ↑i |) = 620 →
    fintype.card {x : Fin 20 → Fin 21 // ∀ i j : Fin 20, i ≠ j → x i ≠ x j ∧ 
      ∑ i : Fin 20, (| x i - ↑i | + | x i + ↑i |) = 620} = (10!)^2 :=
by
  sorry

end num_permutations_eq_ten_fact_squared_l138_138362


namespace probability_factor_of_36_l138_138677

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138677


namespace find_X_l138_138103

theorem find_X : ∃ X : ℝ, 0.60 * X = 0.30 * 800 + 370 ∧ X = 1016.67 := by
  sorry

end find_X_l138_138103


namespace quadrilateral_traversal_theorem_l138_138119

-- Variables for points A, B, C, D, P, Q, R, S
variable {A B C D P Q R S : Type}

-- Corresponding colinearity condition for the points on segments
variable [collinear A B P]
variable [collinear B C Q]
variable [collinear C D R]
variable [collinear D A S]

theorem quadrilateral_traversal_theorem
    (hAB : collinear A B P)
    (hBC : collinear B C Q)
    (hCD : collinear C D R)
    (hDA : collinear D A S) :
    (segment_ratio A P B) * (segment_ratio B Q C) * (segment_ratio C R D) * (segment_ratio D S A) = 1 := 
by {
  sorry
}

end quadrilateral_traversal_theorem_l138_138119


namespace intervals_of_monotonicity_m_eq_1_value_of_m_l138_138260

noncomputable def f (x m : ℝ) : ℝ := (x^2 + m * x + m) * Real.exp x

theorem intervals_of_monotonicity_m_eq_1 :
  let f1 := λ x : ℝ, f x 1 in
  (∀ x, (x < -2 ∨ -1 < x) → deriv f1 x > 0) ∧
  (∀ x, -2 < x ∧ x < -1 → deriv f1 x < 0) :=
sorry

theorem value_of_m (h₁ : ∀ m, m < 2 → (∃ x, f x m = 10 * Real.exp (-2))) :
  m = -6 :=
sorry

end intervals_of_monotonicity_m_eq_1_value_of_m_l138_138260


namespace F_of_2_not_integer_l138_138787

noncomputable def F (x : ℝ) : ℝ := 
  real.sqrt (abs (x + 2)) + (8 / real.pi) * real.arctan (real.sqrt (abs x))

theorem F_of_2_not_integer : F 2 = 2 + (8 / real.pi) * real.arctan (real.sqrt 2) 
∧ ¬ (∃ (n : ℤ), F 2 = n) := by
  sorry

end F_of_2_not_integer_l138_138787


namespace probability_factor_of_36_is_1_over_4_l138_138498

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138498


namespace func_eq_8117_l138_138364

theorem func_eq_8117 (f : ℤ → ℤ) 
  (h : ∀ x y : ℤ, f(x) + f(y) = f(x+1) + f(y-1))
  (h1 : f(2016) = 6102)
  (h2 : f(6102) = 2016)
  : f(1) = 8117 := 
sorry

end func_eq_8117_l138_138364


namespace factor_probability_l138_138636

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138636


namespace probability_factor_of_36_l138_138577

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138577


namespace compute_combination_product_l138_138170

def combination (n k : ℕ) : ℕ := n.factorial / (k.factorial * (n - k).factorial)

theorem compute_combination_product :
  combination 10 3 * combination 8 3 = 6720 :=
by
  sorry

end compute_combination_product_l138_138170


namespace probability_factor_of_36_l138_138532

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138532


namespace four_digit_number_properties_l138_138195

theorem four_digit_number_properties :
  ∃ (a b c d : ℕ), 
    a + b + c + d = 8 ∧ 
    a = 3 * b ∧ 
    d = 4 * c ∧ 
    1000 * a + 100 * b + 10 * c + d = 6200 :=
by
  sorry

end four_digit_number_properties_l138_138195


namespace length_of_each_train_is_40_l138_138726

-- Definitions and conditions
def speed_faster_train_kmh : ℝ := 44
def speed_slower_train_kmh : ℝ := 36
def time_to_pass_seconds : ℝ := 36

-- Conversion factor from km/hr to m/s
def kmhr_to_ms (speed : ℝ) : ℝ := speed * 1000 / 3600

-- Relative speed
def relative_speed_ms : ℝ := kmhr_to_ms (speed_faster_train_kmh - speed_slower_train_kmh)

-- Distance covered when passing (equals the combined length of both trains)
def distance_covered_m : ℝ := relative_speed_ms * time_to_pass_seconds

-- Length of each train
def length_of_each_train_m : ℝ := distance_covered_m / 2

-- Theorem statement
theorem length_of_each_train_is_40 :
  length_of_each_train_m = 40 := by
  sorry

end length_of_each_train_is_40_l138_138726


namespace general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l138_138918

theorem general_term_of_arithmetic_seq
  (a_n : ℕ → ℕ)
  (S_n : ℕ → ℕ)
  (a_2_eq_3 : a_n 2 = 3)
  (S_4_eq_16 : S_n 4 = 16) :
  (∀ n, a_n n = 2 * n - 1) :=
sorry

theorem sum_of_first_n_terms_b_n
  (a_n : ℕ → ℕ)
  (b_n : ℕ → ℝ)
  (T_n : ℕ → ℝ)
  (general_formula_a_n : ∀ n, a_n n = 2 * n - 1)
  (b_n_definition : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1))) :
  (∀ n, T_n n = n / (2 * n + 1)) :=
sorry

end general_term_of_arithmetic_seq_sum_of_first_n_terms_b_n_l138_138918


namespace orthocenter_PQR_l138_138912

structure Point3D :=
  (x : ℚ)
  (y : ℚ)
  (z : ℚ)

def orthocenter (P Q R : Point3D) : Point3D :=
  sorry

theorem orthocenter_PQR :
  orthocenter ⟨2, 3, 4⟩ ⟨6, 4, 2⟩ ⟨4, 5, 6⟩ = ⟨1/2, 13/2, 15/2⟩ :=
by {
  sorry
}

end orthocenter_PQR_l138_138912


namespace count_distinct_tetrahedrons_l138_138839

-- Define the condition that any three lengths can form a triangle.
variable (a b c d e f : ℝ)
axiom triangle_inequality : ∀ (x y z : ℝ), x + y > z ∧ y + z > x ∧ z + x > y

-- To show that the given lengths of the rods can form 30 distinct tetrahedral frameworks.
theorem count_distinct_tetrahedrons (h_triangle : triangle_inequality a b c ∧ 
                                           triangle_inequality a b d ∧ 
                                           triangle_inequality a b e ∧ 
                                           triangle_inequality a b f ∧
                                           triangle_inequality a c d ∧ 
                                           triangle_inequality a c e ∧ 
                                           triangle_inequality a c f ∧ 
                                           triangle_inequality a d e ∧ 
                                           triangle_inequality a d f ∧
                                           triangle_inequality a e f ∧ 
                                           triangle_inequality b c d ∧ 
                                           triangle_inequality b c e ∧ 
                                           triangle_inequality b c f ∧ 
                                           triangle_inequality b d e ∧ 
                                           triangle_inequality b d f ∧ 
                                           triangle_inequality b e f ∧ 
                                           triangle_inequality c d e ∧ 
                                           triangle_inequality c d f ∧ 
                                           triangle_inequality c e f ∧ 
                                           triangle_inequality d e f) :
  ∃ (n : ℕ), n = 30 :=
by {
  -- Proof is elided.
  sorry
}

end count_distinct_tetrahedrons_l138_138839


namespace line_passing_through_bisects_segment_reflected_ray_equation_l138_138733

-- Problem (1)
theorem line_passing_through_bisects_segment :
  ∃ l : ℝ → ℝ → Prop,
    (∀ x y, l x y ↔ x + 4*y - 4 = 0) ∧
    (∀ x, l 0 1) ∧
    (∀ a, (2*a, 8-2*a) ∈ l) ∧
    (∃ b, (b, 6-2*b) ∈ l ∧ b ∈ { x | x - 3*(8-2*x) + 10 = 0}) :=
begin
  -- This is just the statement, proof is not included
  sorry
end

-- Problem (2)
theorem reflected_ray_equation :
  ∃ M K : ℝ → ℝ → Prop,
    (∀ x y, M x y ↔ x - 2*y + 5 = 0) ∧
    (∀ x y, K x y ↔ 3*x - 2*y + 7 = 0) ∧
    (∃ N P : ℝ → ℝ → Prop,
      (∀ b c, P(-1,2)) ∧
      (∀ b c, N(-5,0)) ∧
      (∀ b c, P(b, c) ↔ c = -32/13 ∧ b = -17/13)) ∧
    (∃ R : ℝ → ℝ → Prop,
      (∀ x y, R x y ↔ 29*x - 2*y + 33 = 0)
      ∧ ( -17/13, 32/13) ∈ R ∧
      ( 0, 1) ∈ R) :=
begin
  -- This is just the statement, proof is not included
  sorry
end

end line_passing_through_bisects_segment_reflected_ray_equation_l138_138733


namespace probability_factor_36_l138_138521

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138521


namespace probability_factor_of_36_l138_138539

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138539


namespace smallest_number_of_students_l138_138772

theorem smallest_number_of_students
  (n : ℕ)
  (h1 : 3 * 90 + (n - 3) * 65 ≤ n * 80)
  (h2 : ∀ k, k ≤ n - 3 → 65 ≤ k)
  (h3 : (3 * 90) + ((n - 3) * 65) / n = 80) : n = 5 :=
sorry

end smallest_number_of_students_l138_138772


namespace largest_possible_N_l138_138986

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138986


namespace probability_divisor_of_36_l138_138656

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138656


namespace polar_eq_circle_given_intersections_l138_138322

-- Definition of the parametric equations for the line and curve
def line_parametric (t : ℝ) : ℝ × ℝ := (t + 1, 2 * t)

def curve_parametric (θ : ℝ) : ℝ × ℝ := (2 * (Real.tan θ)^2, 2 * Real.tan θ)

-- Cartesian equation of the line and curve derived from parametric equations
def line_cartesian (x y : ℝ) : Prop := 2 * x - y - 2 = 0

def curve_cartesian (x y : ℝ) : Prop := y^2 = 2 * x

-- Polar equation of the circle given the intersections of line and curve
def polar_circle_eq (ρ θ : ℝ) : Prop := ρ^2 - (5 / 2) * ρ * Real.cos θ - ρ * Real.sin θ - 1 = 0

-- Coordinates of intersection points dialed manually or calculated
def intersection_points (A B : ℝ × ℝ) : Prop :=
  A = (2, 2) ∧ B = (1 / 2, -1)

-- Lean 4 statement: prove polar equation of the circle matches the derived equation
theorem polar_eq_circle_given_intersections :
  ∀ A B : ℝ × ℝ,
    intersection_points A B →
    ∀ ρ θ : ℝ, polar_circle_eq ρ θ :=
by
  intros A B h_intersections ρ θ
  sorry

end polar_eq_circle_given_intersections_l138_138322


namespace probability_three_green_in_seven_trials_l138_138340

namespace Probability

/-- Probability that Jessy picks exactly three green marbles in 7 trials -/
theorem probability_three_green_in_seven_trials :
  let green_prob := 8.0 / 12.0,
      purple_prob := 4.0 / 12.0,
      comb := Nat.choose 7 3
  in  (comb : ℚ) * (green_prob^3 * purple_prob^4) = 280.0 / 729.0 :=
by
  sorry

end Probability

end probability_three_green_in_seven_trials_l138_138340


namespace factor_probability_l138_138621

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138621


namespace p_iff_q_l138_138267

def f (x a : ℝ) := x * (x - a) * (x - 2)

def p (a : ℝ) := 0 < a ∧ a < 2

def q (a : ℝ) := 
  let f' x := 3 * x^2 - 2 * (a + 2) * x + 2 * a
  f' a < 0

theorem p_iff_q (a : ℝ) : (p a) ↔ (q a) := by
  sorry

end p_iff_q_l138_138267


namespace probability_factor_of_36_l138_138674

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138674


namespace problem_proof_l138_138098

open Real

noncomputable def f : ℝ → ℝ := sorry

-- Conditions: f is monotonically decreasing and odd
axiom h_mono_decr : ∀ x y : ℝ, x < y → f(y) < f(x)
axiom h_odd : ∀ x : ℝ, f(-x) = -f(x)

-- Proof Problem: Prove that -f(-3) < f(-4)
theorem problem_proof : -f(-3) < f(-4) :=
by
  sorry

end problem_proof_l138_138098


namespace probability_factor_36_l138_138602

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138602


namespace cone_height_l138_138706

theorem cone_height (r : ℝ) (h : ℝ) (R : ℝ) (H : ℝ) (slant_height : ℝ) (A : ℝ)
  (h₁ : R = 2)
  (h₂ : slant_height = R)
  (h₃ : 2 * real.pi * r = real.pi * R)
  (h₄ : A = slant_height^2 - r^2)
  (h₅ : h = real.sqrt A):
  h = real.sqrt 3 :=
by
  sorry

end cone_height_l138_138706


namespace wages_days_l138_138087

theorem wages_days (A B : ℝ) (hA : 20 * A = 30 * B) : 
  let D := 20 * A / (A + B) in
  D = 12 :=
  by
  have h : D = 12 ∧ D = 20 * A / (A + B) := sorry
  exact h.left

end wages_days_l138_138087


namespace probability_factor_of_36_l138_138693

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138693


namespace probability_factor_of_36_l138_138561

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138561


namespace arithmetic_formula_sum_geometric_sequence_l138_138848

variable {α : Type*} [ring α]

def arithmetic_sequence (a d : α) : ℕ → α
| 0       := a
| (n + 1) := a + (n + 1) • d

def geometric_sequence (r a : α) : ℕ → α
| 0       := a
| (n + 1) := a * r^(n + 1)

theorem arithmetic_formula :
  ∃ (a d : α), (4 * a + 6 * d = 10) ∧
              let a_seq := arithmetic_sequence a d in
                (a_seq 2)^2 = (a_seq 1) * (a_seq 5) ∧
                ((3 : α) = 1) → ∀n, a_seq n = 3 * n - 5 :=
sorry

theorem sum_geometric_sequence (n : ℕ) :
  let a_seq := arithmetic_sequence (-2) 3 in
  let b_seq := λ n, 2^(a_seq n) in
  let sum_b := (finset.range n).sum (λ k, b_seq k) in
  sum_b = (8^n - 1) / 28 :=
sorry

end arithmetic_formula_sum_geometric_sequence_l138_138848


namespace remainder_3211_div_103_l138_138071

theorem remainder_3211_div_103 :
  3211 % 103 = 18 :=
by
  sorry

end remainder_3211_div_103_l138_138071


namespace probability_factor_of_36_l138_138650

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138650


namespace stationary_points_f_l138_138400

variable {ℝ : Type*}

noncomputable def f (x : ℝ) : ℝ := (x^2 + 1) / (x + 1)

-- Theorem to prove stationary points of f(x) occur at specific x values
theorem stationary_points_f (x : ℝ) :
  (1 - 2 / (x + 1)^2) = 0 ↔ x = Real.sqrt 2 - 1 ∨ x = -Real.sqrt 2 - 1 :=
by
  sorry

end stationary_points_f_l138_138400


namespace _l138_138088

noncomputable def prove_concurrent_lines {α β γ : ℝ} (hαβ : α + β < 180) (hβγ : β + γ < 180) (hγα : γ + α < 180) : Prop :=
  let A, B, C := ℝ, ℝ, ℝ in
  ∃ (A_1 B_1 C_1 : ℝ),
    let A2 B2 C2 := ℝ, ℝ, ℝ in 
    let sin := Real.sin in
    let ABC := triangle A B C α β γ in
    let triangle_cond := ∀ {A B C : PlainGeometry.Point}, ∃ (A_1 B_1 C_1 : PlainGeometry.Point), 
    (PlainGeometry.Angle A A_1 = α) ∧ (PlainGeometry.Angle B B_1 = β) ∧ (PlainGeometry.Angle C C_1 = γ) ∧
    ((PlainGeometry.Angle_sum A B C = α + β + γ) → PlainGeometry.Intersect A B ∧ PlainGeometry.Intersect A_1 B_1 ∧ PlainGeometry.Intersect A B C) ∧
    (PlainGeometry.Angle α + PlainGeometry.Angle β < 180) (PlainGeometry.Angle β + PlainGeometry.Angle γ < 180) (PlainGeometry.Angle γ + PlainGeometry.Angle α < 180) in 
      triangle_cond ∧ 
      let intersect_cond := PlainGeometry.Intersect (PlainGeometry.Line A A_1) (PlainGeometry.Line B B_1) (PlainGeometry.Line C C_1) in 
      ∃ (P : PlainGeometry.Point), 
         intersect_cond A A_1 B B_1 (triangle_cond) P ∧ 
         intersect_cond A B_1 C C_1 (triangle_cond) P 

lemma concurrent_lines_theorem : ∀ {α β γ} (hαβ : α + β < 180) (hβγ : β + γ < 180) (hγα : γ + α < 180),
  prove_concurrent_lines hαβ hβγ hγα :=
sorry

end _l138_138088


namespace discuss_monotonicity_proof_inequality_l138_138261

open Real

noncomputable def f (x a : ℝ) : ℝ := (x^2 - a) * exp (1 - x)
noncomputable def f' (x a : ℝ) : ℝ := (2 * x - x^2 + a) * exp (1 - x)

-- Defining that the function f(x) has two different extreme points
def has_two_extreme_points (a : ℝ) : Prop :=
  a > -1

-- Statement to prove monotonicity (not fully executable without further details)
theorem discuss_monotonicity (a : ℝ) (x : ℝ) (h : has_two_extreme_points a) : 
  -- include necessary statements for monotonicity
  sorry

-- Lean 4 theorem statement encapsulating the given proof problem
theorem proof_inequality (a : ℝ) (x1 x2 : ℝ) (h₀ : has_two_extreme_points a) (h₁ : x1 < x2) (h₂ : 1 - sqrt (1 + a) = x1)
(h₃ : 1 + sqrt (1 + a) = x2) 
: 
  x2 * f x1 a ≤ (2 * exp (1 - x1) / (exp (1 - x1) + 1)) * (f' x1 a - a * (exp (1 - x1) + 1)) :=
begin
  sorry
end

end discuss_monotonicity_proof_inequality_l138_138261


namespace shortest_distance_l138_138348

noncomputable def point_A (u : ℝ) : ℝ × ℝ × ℝ := (u + 1, u + 2, 2 * u + 3)
noncomputable def point_B (v : ℝ) : ℝ × ℝ × ℝ := (2 * v + 2, -v + 4, v)

def distance_squared (A B : ℝ × ℝ × ℝ) : ℝ :=
  (A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2

theorem shortest_distance : ∃ u v : ℝ, distance_squared (point_A u) (point_B v) = 5 :=
sorry

end shortest_distance_l138_138348


namespace unique_cube_coloring_l138_138038

-- Definition of vertices at the bottom of the cube with specific colors
inductive Color 
| Red | Green | Blue | Purple

open Color

def bottom_colors : Fin 4 → Color
| 0 => Red
| 1 => Green
| 2 => Blue
| 3 => Purple

-- Definition of the property that ensures each face of the cube has different colored corners
def all_faces_different_colors (top_colors : Fin 4 → Color) : Prop :=
  (top_colors 0 ≠ Red) ∧ (top_colors 0 ≠ Green) ∧ (top_colors 0 ≠ Blue) ∧
  (top_colors 1 ≠ Green) ∧ (top_colors 1 ≠ Blue) ∧ (top_colors 1 ≠ Purple) ∧
  (top_colors 2 ≠ Red) ∧ (top_colors 2 ≠ Blue) ∧ (top_colors 2 ≠ Purple) ∧
  (top_colors 3 ≠ Red) ∧ (top_colors 3 ≠ Green) ∧ (top_colors 3 ≠ Purple)

-- Prove there is exactly one way to achieve this coloring of the top corners
theorem unique_cube_coloring : ∃! (top_colors : Fin 4 → Color), all_faces_different_colors top_colors :=
sorry

end unique_cube_coloring_l138_138038


namespace rectangle_width_decrease_l138_138416

theorem rectangle_width_decrease
  (L W : ℝ) 
  (h_area : (1.4 * L) * (W / 1.4) = L * W)
  (h_perimeter : 2 * (1.4 * L) + 2 * (W / 1.4) = 2 * L + 2 * W) :
  ((W - (W / 1.4)) / W) * 100 ≈ 28.57 :=
begin
  sorry -- proof is not required
end

end rectangle_width_decrease_l138_138416


namespace unique_digit_sum_l138_138327

theorem unique_digit_sum (X Y M Z F : ℕ) (H1 : X ≠ 0) (H2 : Y ≠ 0) (H3 : M ≠ 0) (H4 : Z ≠ 0) (H5 : F ≠ 0)
  (H6 : X ≠ Y) (H7 : X ≠ M) (H8 : X ≠ Z) (H9 : X ≠ F)
  (H10 : Y ≠ M) (H11 : Y ≠ Z) (H12 : Y ≠ F)
  (H13 : M ≠ Z) (H14 : M ≠ F)
  (H15 : Z ≠ F)
  (H16 : 10 * X + Y ≠ 0) (H17 : 10 * M + Z ≠ 0)
  (H18 : 111 * F = (10 * X + Y) * (10 * M + Z)) :
  X + Y + M + Z + F = 28 := by
  sorry

end unique_digit_sum_l138_138327


namespace tournament_schedule_count_l138_138187
   
   -- Definitions for the players of the two schools
   inductive EastwoodPlayer | A | B | C | D
   inductive WestviewPlayer | W | X | Y | Z

   -- Definition to state that each player plays every other player exactly once
   def playsEachOtherExactlyOnce :=
     ∀ (e : EastwoodPlayer) (w : WestviewPlayer), (∃ (round : ℕ), round ∈ {1, 2, 3, 4})

   -- Problem statement in Lean 4
   theorem tournament_schedule_count :
     (numberOfWays : ℕ) 
       ∧ numberOfWays = 4! * (4!)^4
       ∧ numberOfWays = 24 * (24^4) 
       ∧ numberOfWays = 7962624 := by
     sorry
   
end tournament_schedule_count_l138_138187


namespace probability_factor_36_l138_138514

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138514


namespace probability_factor_of_36_l138_138673

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138673


namespace cone_prism_volume_ratio_l138_138126

theorem cone_prism_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) :
  (volume_cone r h / volume_prism r h) = (Real.pi / 18) :=
by
  let volume_cone := (1/3) * Real.pi * r^2 * h
  let volume_prism := (2 * r) * (3 * r) * h
  let ratio := volume_cone / volume_prism
  have : volume_cone = ((1/3) * Real.pi * r^2 * h) := by simp
  have : volume_prism = ((6 * r^2) * h) := by simp
  have : ratio = (volume_cone / volume_prism) := by simp
  sorry

end cone_prism_volume_ratio_l138_138126


namespace probability_factor_of_36_l138_138478

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138478


namespace probability_of_monochromatic_triangle_l138_138903

noncomputable def problem_statement : ℕ → ℕ → ℕ → real := sorry

theorem probability_of_monochromatic_triangle:
  ∀ (pentagon_sides diagonals red_diagonals : ℕ), 
  pentagon_sides = 5 → 
  diagonals = 5 → 
  red_diagonals = 3 → 
  problem_statement pentagon_sides diagonals red_diagonals = 0.9979 :=
begin
  sorry
end

end probability_of_monochromatic_triangle_l138_138903


namespace false_statement_E_l138_138061

theorem false_statement_E
  (A B C : Type)
  (a b c : ℝ)
  (ha_gt_hb : a > b)
  (hb_gt_hc : b > c)
  (AB BC : ℝ)
  (hAB : AB = a - b → True)
  (hBC : BC = b + c → True)
  (hABC : AB + BC > a + b + c → True)
  (hAC : AB + BC > a - c → True) : False := sorry

end false_statement_E_l138_138061


namespace num_topping_combinations_l138_138805

-- Define the conditions as constants in Lean
constant cheese_options : ℕ := 3
constant meat_options : ℕ := 4
constant vegetable_options : ℕ := 5
constant pepperoni_option : ℕ := 1 -- Only one option for pepperoni
constant restricted_vegetable_options : ℕ := 1 -- Only one restricted option (peppers)

-- Define the total number of combinations without restrictions
def total_combinations : ℕ := cheese_options * meat_options * vegetable_options

-- Define the number of restricted combinations (pepperoni and peppers)
def restricted_combinations : ℕ := cheese_options * pepperoni_option * restricted_vegetable_options

-- Define the allowed combinations
def allowed_combinations : ℕ := total_combinations - restricted_combinations

-- The theorem stating the problem question and expected answer
theorem num_topping_combinations : allowed_combinations = 57 := by
  sorry

end num_topping_combinations_l138_138805


namespace probability_factor_of_36_l138_138489

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138489


namespace total_handshakes_l138_138889

-- Define the problem conditions
def num_boys : ℕ := 8

-- Calculate the number of handshakes using combinations
def num_handshakes : ℕ := Nat.choose num_boys 2

-- The statement to prove
theorem total_handshakes (n : ℕ) (h1 : n = num_boys) : num_handshakes = 28 :=
by
  rw [h1]
  sorry

end total_handshakes_l138_138889


namespace max_constant_inequality_l138_138963

theorem max_constant_inequality
  (a b c : ℝ) (h1 : a ≤ b) (h2 : b < c) (h3 : a^2 + b^2 = c^2) :
  ∃ M : ℝ, (∀ a b c, a ≤ b → b < c → a^2 + b^2 = c^2 → 
           (1 / a + 1 / b + 1 / c ≥ M / (a + b + c))) ∧ M = 5 + 3 * Real.sqrt 2 :=
begin
  use 5 + 3 * Real.sqrt 2,
  sorry
end

end max_constant_inequality_l138_138963


namespace probability_divisor_of_36_is_one_fourth_l138_138466

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138466


namespace number_of_pairs_l138_138296

noncomputable def are_same_graphs (f g : ℝ × ℝ → Prop) : Prop :=
  ∀ (x y : ℝ), f (x, y) ↔ g (x, y)

theorem number_of_pairs (b c : ℝ) :
  (∀ (x y : ℝ), (3 * x + b * y + c = 0) ↔ (c * x - 2 * y + 12 = 0)) →
  (∃ (pairs : set (ℝ × ℝ)), pairs = {(b, c) | (b = -1 ∧ c = 6) ∨ (b = 1 ∧ c  = -6)} ∧ pairs.card = 2) :=
by {
  intro H,
  -- Further proof steps would go here
  sorry
}

end number_of_pairs_l138_138296


namespace f_log2_9_eq_8_over_9_l138_138233

noncomputable def f : ℝ → ℝ :=
  sorry -- Placeholder for the actual function definition based on conditions

theorem f_log2_9_eq_8_over_9 :
  (∀ x : ℝ, f(x + 1) = 1 / f(x)) →
  (∀ x : ℝ, 0 < x ∧ x ≤ 1 → f(x) = 2 ^ x) →
  f(log 9 / log 2) = 8 / 9 :=
by
  intros h1 h2
  sorry

end f_log2_9_eq_8_over_9_l138_138233


namespace M_inter_N_M_union_not_N_l138_138873

def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}
def N : Set ℝ := {x | x > 0}

theorem M_inter_N :
  M ∩ N = {x | 0 < x ∧ x ≤ 3} := 
sorry

theorem M_union_not_N :
  M ∪ {x | x ≤ 0} = {x | x ≤ 3} := 
sorry

end M_inter_N_M_union_not_N_l138_138873


namespace sum_of_primes_no_solution_congruence_l138_138180

theorem sum_of_primes_no_solution_congruence :
  (∑ p in {p : ℕ | p.Prime ∧ ∀ x, ¬(4 * (6 * x + 1) ≡ 7 [MOD p])}, p) = 5 :=
sorry

end sum_of_primes_no_solution_congruence_l138_138180


namespace sales_percentage_l138_138306

theorem sales_percentage {markers notebooks : ℕ} (h_total : markers + notebooks ≤ 100) :
  100 - markers - notebooks = 36 :=
by
  -- Given conditions
  have markers_eq : markers = 42 := by sorry
  have notebooks_eq : notebooks = 22 := by sorry
  -- Substituting and proving
  rw [markers_eq, notebooks_eq]
  norm_num

end sales_percentage_l138_138306


namespace new_students_weights_correct_l138_138080

-- Definitions of the initial conditions
def initial_student_count : ℕ := 29
def initial_avg_weight : ℚ := 28
def total_initial_weight := initial_student_count * initial_avg_weight
def new_student_counts : List ℕ := [30, 31, 32, 33]
def new_avg_weights : List ℚ := [27.2, 27.8, 27.6, 28]

-- Weights of the four new students
def W1 : ℚ := 4
def W2 : ℚ := 45.8
def W3 : ℚ := 21.4
def W4 : ℚ := 40.8

-- The proof statement
theorem new_students_weights_correct :
  total_initial_weight = 812 ∧
  W1 = 4 ∧
  W2 = 45.8 ∧
  W3 = 21.4 ∧
  W4 = 40.8 ∧
  (total_initial_weight + W1) = 816 ∧
  (total_initial_weight + W1) / new_student_counts.head! = new_avg_weights.head! ∧
  (total_initial_weight + W1 + W2) = 861.8 ∧
  (total_initial_weight + W1 + W2) / new_student_counts.tail.head! = new_avg_weights.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3) = 883.2 ∧
  (total_initial_weight + W1 + W2 + W3) / new_student_counts.tail.tail.head! = new_avg_weights.tail.tail.head! ∧
  (total_initial_weight + W1 + W2 + W3 + W4) = 924 ∧
  (total_initial_weight + W1 + W2 + W3 + W4) / new_student_counts.tail.tail.tail.head! = new_avg_weights.tail.tail.tail.head! :=
by
  sorry

end new_students_weights_correct_l138_138080


namespace black_piece_probability_l138_138435

-- Definitions based on conditions
def total_pieces : ℕ := 10 + 5
def black_pieces : ℕ := 10

-- Probability calculation
def probability_black : ℚ := black_pieces / total_pieces

-- Statement to prove
theorem black_piece_probability : probability_black = 2/3 := by
  sorry -- proof to be filled in later

end black_piece_probability_l138_138435


namespace probability_factor_of_36_is_1_over_4_l138_138496

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138496


namespace probability_factor_of_36_l138_138484

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138484


namespace Joey_age_is_six_l138_138939

theorem Joey_age_is_six (ages: Finset ℕ) (a1 a2 a3 a4 : ℕ) (h1: ages = {4, 6, 8, 10})
  (h2: a1 + a2 = 14 ∨ a2 + a3 = 14 ∨ a3 + a4 = 14) (h3: a1 > 7 ∨ a2 > 7 ∨ a3 > 7 ∨ a4 > 7)
  (h4: (6 ∈ ages ∧ a1 ∈ ages) ∨ (6 ∈ ages ∧ a2 ∈ ages) ∨ 
      (6 ∈ ages ∧ a3 ∈ ages) ∨ (6 ∈ ages ∧ a4 ∈ ages)): 
  (a1 = 6 ∨ a2 = 6 ∨ a3 = 6 ∨ a4 = 6) :=
by
  sorry

end Joey_age_is_six_l138_138939


namespace find_cost_of_two_enchiladas_and_five_tacos_l138_138009

noncomputable def cost_of_two_enchiladas_and_five_tacos (e t : ℝ) : ℝ :=
  2 * e + 5 * t

theorem find_cost_of_two_enchiladas_and_five_tacos (e t : ℝ):
  (e + 4 * t = 3.50) → (4 * e + t = 4.20) → cost_of_two_enchiladas_and_five_tacos e t = 5.04 :=
by
  intro h1 h2
  sorry

end find_cost_of_two_enchiladas_and_five_tacos_l138_138009


namespace distance_A_B_l138_138342

variable (D : ℝ) -- Define the distance between point A and point B as a real number

def john_max_speed : ℝ := 5  -- John's maximum rowing speed in still water
def time_total : ℝ := 5  -- Total time for the round trip
def stream_speed_AB : ℝ := 1  -- Stream speed from A to B
def stream_speed_BA : ℝ := 2  -- Stream speed from B to A
def row_speed_AB : ℝ := 0.9 * john_max_speed  -- John's rowing speed from A to B as 90% of his maximum speed
def row_speed_BA : ℝ := 0.8 * john_max_speed  -- John's rowing speed from B to A as 80% of his maximum speed

-- Effective speed calculations
def effective_speed_AB : ℝ := row_speed_AB + stream_speed_AB
def effective_speed_BA : ℝ := row_speed_BA - stream_speed_BA

-- Time calculations
def time_AB : ℝ := D / effective_speed_AB
def time_BA : ℝ := D / effective_speed_BA

-- The core proof statement
theorem distance_A_B :
  time_AB + time_BA = time_total → D ≈ 4.23 :=
by
  sorry

end distance_A_B_l138_138342


namespace unique_colored_pencils_l138_138392

open Set

-- Conditions
variables (S J A : Finset ℕ) -- Sets representing unique colors by Serenity, Jordan, and Alex
variable (n : ℕ) -- The correct answer we want to prove

-- Given conditions
axiom Serenity_colors : S.card = 24
axiom Jordan_colors : J.card = 36
axiom Alex_colors : A.card = 30

axiom Serenity_Jordan_overlap : (S ∩ J).card = 8
axiom Serenity_Alex_overlap : (S ∩ A).card = 5
axiom Jordan_Alex_overlap : (J ∩ A).card = 10
axiom Triple_overlap : (S ∩ J ∩ A).card = 3

-- Question to prove
theorem unique_colored_pencils :
  S ∪ J ∪ A.card = 73 :=
by
  sorry

end unique_colored_pencils_l138_138392


namespace factor_probability_36_l138_138619

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138619


namespace probability_factor_of_36_l138_138528

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138528


namespace probability_factor_36_l138_138547

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138547


namespace solve_system_eq_l138_138977

theorem solve_system_eq (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hxy : x + y ≠ 0) 
  (hyz : y + z ≠ 0) (hzx : z + x ≠ 0) :
  (xy / (x + y) = 1 / 3) ∧ (yz / (y + z) = 1 / 4) ∧ (zx / (z + x) = 1 / 5) →
  (x = 1 / 2) ∧ (y = 1) ∧ (z = 1 / 3) :=
  sorry

end solve_system_eq_l138_138977


namespace largest_solution_l138_138205

noncomputable def floor : ℝ → ℤ := λ x, Int.floor x
noncomputable def fractional_part : ℝ → ℝ := λ x, x - Int.floor x

theorem largest_solution (x : ℝ) (h1 : floor x = 7 + 150 * fractional_part x)
  (h2 : 0 ≤ fractional_part x) (h3 : fractional_part x < 1) : x = 156.9933 :=
sorry

end largest_solution_l138_138205


namespace probability_factor_36_l138_138523

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138523


namespace no_positive_solutions_exist_l138_138183

theorem no_positive_solutions_exist :
  ∀ (a b c d : ℝ), 0 < a → 0 < b → 0 < c → 0 < d →
  (a * d + b = c) →
  (sqrt a * sqrt d + sqrt b = sqrt c) →
  false :=
by
  intros a b c d ha hb hc hd h1 h2
  sorry

end no_positive_solutions_exist_l138_138183


namespace probability_factor_of_36_l138_138527

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138527


namespace probability_factor_of_36_l138_138558

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138558


namespace probability_factor_of_36_l138_138559

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138559


namespace carla_order_total_l138_138161

def base_cost : ℝ := 7.50
def coupon_discount : ℝ := 2.50
def senior_discount_rate : ℝ := 0.20
def swap_charge : ℝ := 1.00

theorem carla_order_total :
  let subtotal := base_cost - coupon_discount in
  let senior_discount := subtotal * senior_discount_rate in
  let discounted_total := subtotal - senior_discount in
  let final_total := discounted_total + swap_charge in
  final_total = 5.00 := by
  sorry

end carla_order_total_l138_138161


namespace parabola_standard_equation_l138_138048

theorem parabola_standard_equation :
  ∃ m : ℝ, (∀ x y : ℝ, (x^2 = 2 * m * y ↔ (0, -6) ∈ ({p | 3 * p.1 - 4 * p.2 - 24 = 0}))) → 
  (x^2 = -24 * y) := 
by {
  sorry
}

end parabola_standard_equation_l138_138048


namespace triangle_equilateral_of_conditions_l138_138334

variables {A B C : ℕ} {a b c : ℕ}

/-- Given the conditions on the sides and angles of triangle ABC, show that the triangle is equilateral. -/
theorem triangle_equilateral_of_conditions
  (h1 : b^2 + c^2 - a^2 = b * c)
  (h2 : 2 * cos B * sin C = sin A) :
  (A = π / 3 ∧ b = c) → A = B ∧ B = C :=
sorry

end triangle_equilateral_of_conditions_l138_138334


namespace original_wire_length_l138_138140

theorem original_wire_length (S L : ℝ) (h1: S = 30) (h2: S = (3 / 5) * L) : S + L = 80 := by 
  sorry

end original_wire_length_l138_138140


namespace train_ticket_product_l138_138403

theorem train_ticket_product
  (a b c d e : ℕ)
  (h1 : b = a + 1)
  (h2 : c = a + 2)
  (h3 : d = a + 3)
  (h4 : e = a + 4)
  (h_sum : a + b + c + d + e = 120) :
  a * b * c * d * e = 7893600 :=
sorry

end train_ticket_product_l138_138403


namespace general_formulas_sequences_find_T_n_exist_pairs_mn_l138_138974
noncomputable def a (n : ℕ) : ℝ := n
noncomputable def b (n : ℕ) : ℝ := (1 / 2) * (1 / 3)^(n - 1)
noncomputable def T (n : ℕ) : ℝ := (3 / 4) - (1 / (4 * 3^(n - 1))) - (n / (2 * 3^n))

theorem general_formulas_sequences :
  ∀ n : ℕ, a(3) + a(6) = a(9) ∧ 
            a(5) + a(7)^2 = 6*a(9) ∧ 
            a(n) = n ∧ 
            b(n) = (1 / 2) * (1 / 3)^(n - 1) := by 
  sorry

theorem find_T_n :
  ∀ n : ℕ, T(n) = (3 / 4) - (1 / (4 * 3^(n - 1))) - (n / (2 * 3^n)) := by 
  sorry

theorem exist_pairs_mn :
  ∃ (m n : ℕ), T(n) = (a(m+1) / (2 * a(m))) := by 
  sorry

end general_formulas_sequences_find_T_n_exist_pairs_mn_l138_138974


namespace probability_factor_of_36_l138_138492

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138492


namespace area_of_rectangular_room_l138_138718

/--
A square carpet with an area of 169 m² must have 2 meters cut off one of its edges in order to be a perfect fit for a rectangular room.
Prove that the area of the rectangular room is 143 m².
-/
theorem area_of_rectangular_room (a : ℕ) (s : ℕ) (cut : ℕ) (new_a : ℕ) : 
  a = 169 ∧ s = Int.sqrt 169 ∧ cut = 2 ∧ new_a = s * (s - cut) → new_a = 143 :=
begin
  sorry
end

end area_of_rectangular_room_l138_138718


namespace total_ages_four_years_ago_l138_138432

-- Declaring variables for the ages of Amar, Akbar, Anthony, and Alex
variables (A B C X : ℕ)

-- Conditions given in the problem
def condition1 : Prop := A + B + C + X = 88
def condition2 : Prop := (A - 4) + (B - 4) + (C - 4) = 66
def condition3 : Prop := A = 2 * X
def condition4 : Prop := B = A - 3

-- The theorem we need to prove, stating that the total ages of Amar, Akbar, Anthony, and Alex four years ago was 72
theorem total_ages_four_years_ago (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : 
  (A - 4) + (B - 4) + (C - 4) + (X - 4) = 72 := 
sorry

end total_ages_four_years_ago_l138_138432


namespace remaining_sugar_l138_138164

/-- Chelsea has 24 kilos of sugar. She divides them into 4 bags equally.
  Then one of the bags gets torn and half of the sugar falls to the ground.
  How many kilos of sugar remain? --/
theorem remaining_sugar (total_sugar : ℕ) (bags : ℕ) (torn_bag_fraction : ℚ) (initial_per_bag : ℕ) (fallen_sugar : ℕ) :
  total_sugar = 24 →
  bags = 4 →
  (total_sugar / bags) = initial_per_bag →
  initial_per_bag = 6 →
  (initial_per_bag * torn_bag_fraction) = fallen_sugar →
  torn_bag_fraction = 1/2 →
  fallen_sugar = 3 →
  (total_sugar - fallen_sugar) = 21 :=
begin
  intros h_total h_bags h_initial_per_bag_eq h_initial_per_bag h_torn_bag_fraction_eq h_torn_bag_fraction h_fallen_sugar,
  rw [h_total, h_initial_per_bag_eq.symm, h_bags],
  norm_num at *,
  sorry
end

end remaining_sugar_l138_138164


namespace part1_range_part2_range_l138_138833

noncomputable def f1 (x : ℝ) : ℝ := Real.sin (x + Real.pi / 3)

theorem part1_range (h : ∀ x, 0 ≤ x ∧ x ≤ Real.pi) : 
  ∃ y, f1 y ≥ -Real.sqrt 3 / 2 ∧ f1 y ≤ 1 := sorry

noncomputable def f2 (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 3)

theorem part2_range (a : ℝ) (h : ∀ k, x = -Real.pi / 6 + k * Real.pi / 2 ∧ 3 = ∑ k, (f2 k = 0)) : 
  a ∈ Set.Ioc (7 * Real.pi / 3) (17 * Real.pi / 6) := sorry

end part1_range_part2_range_l138_138833


namespace range_of_m_l138_138857

variable (f : ℝ → ℝ) (m : ℝ)

-- Given conditions
def condition1 := ∀ x, f (-x) = -f x -- f(x) is an odd function
def condition2 := ∀ x, f (x + 3) = f x -- f(x) has a minimum positive period of 3
def condition3 := f 2015 > 1 -- f(2015) > 1
def condition4 := f 1 = (2 * m + 3) / (m - 1) -- f(1) = (2m + 3) / (m - 1)

-- We aim to prove that -2/3 < m < 1 given these conditions.
theorem range_of_m : condition1 f → condition2 f → condition3 f → condition4 f m → -2 / 3 < m ∧ m < 1 := by
  intros
  sorry

end range_of_m_l138_138857


namespace find_value_of_expression_l138_138882

theorem find_value_of_expression (x y : ℝ) (h1 : |x| = 2) (h2 : |y| = 3) (h3 : x / y < 0) :
  (2 * x - y = 7) ∨ (2 * x - y = -7) :=
by
  sorry

end find_value_of_expression_l138_138882


namespace find_concentration_of_second_mixture_l138_138748

noncomputable def concentration_of_second_mixture (total_volume : ℝ) (final_percent : ℝ) (pure_antifreeze : ℝ) (pure_antifreeze_amount : ℝ) : ℝ :=
  let remaining_volume := total_volume - pure_antifreeze_amount
  let final_pure_amount := final_percent * total_volume
  let required_pure_antifreeze := final_pure_amount - pure_antifreeze
  (required_pure_antifreeze / remaining_volume) * 100

theorem find_concentration_of_second_mixture :
  concentration_of_second_mixture 55 0.20 6.11 6.11 = 10 :=
by
  simp [concentration_of_second_mixture]
  sorry

end find_concentration_of_second_mixture_l138_138748


namespace goals_in_fifth_match_l138_138115

variable (G : ℕ)
variable (total_goals : ℕ := 8)
variable (increase : ℚ := 0.1)
variable (matches_before : ℚ := 4)
variable (matches_after : ℚ := 5)

-- Define the average goals per match before the fifth match
def avg_goals_before (goals_before : ℚ) : ℚ := goals_before / matches_before

-- Define the average goals per match after the fifth match
def avg_goals_after (goals_after : ℚ) := goals_after / matches_after

-- Define the number of goals scored before the fifth match
def goals_before_fifth_match := total_goals - G

-- Define the proof problem statement
theorem goals_in_fifth_match 
  (h : avg_goals_before (goals_before_fifth_match) + increase = avg_goals_after total_goals) :
  G = 2 := sorry

end goals_in_fifth_match_l138_138115


namespace find_smallest_n_l138_138953

noncomputable def b (n : ℕ) : ℝ :=
  match n with
  | 0       => sin (Real.pi / 30) ^ 2
  | n + 1   => 4 * (b n) * (1 - (b n))

theorem find_smallest_n
  (h : ∀ (n : ℕ), b_0 = sin (Real.pi / 30) ^ 2 ∧ ∀ n ≥ 0, b (n + 1) = 4 * (b n) * (1 - (b n))) :
  ∃ (n : ℕ), n > 0 ∧ b n = b 0 ∧ n = 15 :=
begin
  sorry
end

end find_smallest_n_l138_138953


namespace largest_possible_N_l138_138984

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138984


namespace perpendicular_of_triangle_conditions_l138_138303

noncomputable def triangle (A B C : Type) [inhabited A] [innhabited B] [inhabited C] :=  
(A ≠ B) ∧ (A ≠ C) ∧ (B ≠ C)

theorem perpendicular_of_triangle_conditions
    {A B C M D E : Type}
    [inhabited A]
    [inhabited B]
    [inhabited C]
    [inhabited M]
    [inhabited D]
    [inhabited E]
    (triangle_ABC : triangle A B C)
    (h1 : AB > AC)
    (median_AM : M = median A B C)
    (angle_bisector_AD : D = angle_bisector ∠A)
    (E_on_AM : E ∈ inline AM)
    (ED_parallel_AC : ED ∥ AC)
  : EC ⊥ AD := 
begin
  sorry
end

end perpendicular_of_triangle_conditions_l138_138303


namespace arithmetic_seq_third_term_l138_138301

theorem arithmetic_seq_third_term
  (a d : ℝ)
  (h : a + (a + 2 * d) = 10) :
  a + d = 5 := by
  sorry

end arithmetic_seq_third_term_l138_138301


namespace area_of_ABCD_not_integer_l138_138904

theorem area_of_ABCD_not_integer (AB CD : ℕ) (h : AB * CD ≠ n^2) :
  (1/2 * (AB + CD) * BC : ℝ).denom ≠ 1  := 
by 
  sorry

example : area_of_ABCD_not_integer 4 2 := by
  exact 32

example : area_of_ABCD_not_integer 6 3 := by
  exact 18

example : area_of_ABCD_not_integer 8 4 := by
  exact 32

example : area_of_ABCD_not_integer 10 5 := by
  exact 50

example : area_of_ABCD_not_integer 12 6 := by
  exact 72

end area_of_ABCD_not_integer_l138_138904


namespace probability_factor_of_36_is_1_over_4_l138_138508

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138508


namespace probability_factor_of_36_l138_138570

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138570


namespace probability_divisor_of_36_is_one_fourth_l138_138475

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138475


namespace calculate_expression_l138_138160

theorem calculate_expression :
  (-3)^2 - Real.sqrt 4 + (1/2)^(-1) = 9 :=
by
  sorry

end calculate_expression_l138_138160


namespace num_valid_arrangements_l138_138014

-- Definitions of the four balls and three boxes
inductive Ball : Type
| one
| two
| three
| four

inductive Box : Type
| A
| B
| C

-- A function that checks whether a list of boxes fulfills the condition of each box containing at least one ball
def valid_assignment (assignment : List (List Ball)) : Prop :=
  ∀ box, box ∈ assignment → box ≠ []

-- Function that checks if balls 1 and 2 are not in the same box
def no_1_and_2_together (assignment : List (List Ball)) : Prop :=
  ∀ box, ¬ (Ball.one ∈ box ∧ Ball.two ∈ box)

-- Function to count valid assignments
def count_valid_assignments : ℕ := sorry

-- Theorem stating that the number of valid arrangements is 30
theorem num_valid_arrangements : count_valid_assignments = 30 := sorry

end num_valid_arrangements_l138_138014


namespace train_length_is_correct_l138_138767

-- Definitions of speeds and time
def speedTrain_kmph := 100
def speedMotorbike_kmph := 64
def overtakingTime_s := 20

-- Calculate speeds in m/s
def speedTrain_mps := speedTrain_kmph * 1000 / 3600
def speedMotorbike_mps := speedMotorbike_kmph * 1000 / 3600

-- Calculate relative speed
def relativeSpeed_mps := speedTrain_mps - speedMotorbike_mps

-- Calculate the length of the train
def length_of_train := relativeSpeed_mps * overtakingTime_s

-- Theorem: Verifying the length of the train is 200 meters
theorem train_length_is_correct : length_of_train = 200 := by
  -- Sorry placeholder for proof
  sorry

end train_length_is_correct_l138_138767


namespace product_of_distances_eq_square_of_center_distance_l138_138374

-- Define the geometric setup
variables {Γ : Type*} [metric_space Γ] 

-- Circle center O, radius r
variables {O : Γ} {r : ℝ} (circle : Metric.sphere O r)

-- Points on parallel tangents to the circle
variables {A B : Γ} (is_tangent_A : Metric.tangent_line O A) (is_tangent_B : Metric.tangent_line O B)

-- Intersection point of the second tangents
variables {M : Γ} (second_tangents_intersect : Metric.second_tangent_point O A B M)

-- Distance functions
def dist := Metric.dist

theorem product_of_distances_eq_square_of_center_distance :
  dist A M * dist B M = (dist O M) ^ 2 :=
sorry

end product_of_distances_eq_square_of_center_distance_l138_138374


namespace largest_N_satisfying_cond_l138_138992

theorem largest_N_satisfying_cond :
  ∃ N : ℕ, (∀ d1 d2 d3 d4 d5 d6 d7 d8 d9 : ℕ, 
  list.in_list [1, d1, d2, d3, d4, d5, d6, d7, N],
  d8 = 21 * d2 ∧ N = 441) :=
sorry

end largest_N_satisfying_cond_l138_138992


namespace zero_point_in_interval_l138_138823

open Real

noncomputable def f (x : ℝ) : ℝ :=
  log (x + 1) / log 2 - 2 / x

theorem zero_point_in_interval :
  f 1 < 0 ∧ f 2 > 0 → ∃ x ∈ Ioo 1 2, f x = 0 :=
by
  have f1 : f 1 < 0 := sorry
  have f2 : f 2 > 0 := sorry
  sorry

end zero_point_in_interval_l138_138823


namespace task1_max_min_in_interval_task2_satisfies_inequality_l138_138262

noncomputable def f (x : ℝ) : ℝ := x^2 - (3 - 1)*x - 3^2

theorem task1_max_min_in_interval : 
  ∀ x ∈ set.Icc 0 2, f x ≥ -4 ∧ f x ≤ -3 :=
by 
  sorry

theorem task2_satisfies_inequality :
  ∀ (a : ℝ), (a < 0) ∧ (∀ x : ℝ, - (cos x)^2 + (a-1)*(cos x) + a^2 ≥ 0) → (a ≤ -2) :=
by 
  sorry

end task1_max_min_in_interval_task2_satisfies_inequality_l138_138262


namespace proof_ellipse_equation_proof_no_such_line_exists_l138_138240

noncomputable def ellipse_equation : Prop :=
  ∀ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ b < a ∧ c = a * (√2 / 2) ∧ (1/a^2 + 1/(2*b^2) = 1) →
      ((1 / a^2) + (1 / (2 * b^2)) = 1) → 
      (a ^ 2 = 2 ∧ b ^ 2 = 1) ∧ 
      ∀ (x y : ℝ), (x = 1 ∧ y = √2 / 2) → 
      (1 / 2 * x^2 + y^2 = 1)

noncomputable def no_such_line_exists : Prop :=
  ∀ (k : ℝ), 
    (k ^ 2 > 1/2) →
      ∀ (P Q : ℝ × ℝ), 
        P = (x1, y1) ∧ Q = (x2, y2) ∧ 
        ∀ (OP OQ A2 B : ℝ × ℝ), 
          OP = (x1 + x2, y1 + y2) ∧ OQ = (x1 + x2, y1 + y2) ∧ 
          A2 = (√2, 0) ∧ B = (0, 1) →
            (¬ collinear_vector (OP + OQ) (A2 - B))

theorem proof_ellipse_equation : ellipse_equation := 
by 
  sorry

theorem proof_no_such_line_exists : no_such_line_exists := 
by 
  sorry

end proof_ellipse_equation_proof_no_such_line_exists_l138_138240


namespace village_population_rate_decrease_l138_138066

/--
Village X has a population of 78,000, which is decreasing at a certain rate \( R \) per year.
Village Y has a population of 42,000, which is increasing at the rate of 800 per year.
In 18 years, the population of the two villages will be equal.
We aim to prove that the rate of decrease in population per year for Village X is 1200.
-/
theorem village_population_rate_decrease (R : ℝ) 
  (hx : 78000 - 18 * R = 42000 + 18 * 800) : 
  R = 1200 :=
by
  sorry

end village_population_rate_decrease_l138_138066


namespace price_per_packet_of_corn_chips_l138_138414

theorem price_per_packet_of_corn_chips
  (price_chips : ℕ)
  (num_chips : ℕ)
  (total_budget : ℕ)
  (num_corn_chips : ℕ)
  (price_corn_chips : ℝ)
  (price_chips = 2)
  (num_chips = 15)
  (total_budget = 45)
  (num_corn_chips = 10)
  (total_spent_on_chips : ℕ := price_chips * num_chips)
  (remaining_budget : ℕ := total_budget - total_spent_on_chips) :
  remaining_budget / num_corn_chips = price_corn_chips → price_corn_chips = 1.5 := 
by
  sorry

end price_per_packet_of_corn_chips_l138_138414


namespace probability_divisor_of_36_l138_138658

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138658


namespace probability_factor_of_36_l138_138537

def is_factor (d n : ℕ) : Prop := n % d = 0

theorem probability_factor_of_36 : 
  (∑ i in finset.range (36+1), if is_factor i 36 then (1:ℚ) else 0) / 36 = 1 / 4 := by
sorry

end probability_factor_of_36_l138_138537


namespace convert_octal_127_to_binary_l138_138800

def octal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 1 => 3  -- 001 in binary
  | 2 => 2  -- 010 in binary
  | 7 => 7  -- 111 in binary
  | _ => 0  -- No other digits are used in this example

theorem convert_octal_127_to_binary :
  octal_to_binary 1 * 1000000 + octal_to_binary 2 * 1000 + octal_to_binary 7 = 1010111 :=
by
  -- Proof would go here
  sorry

end convert_octal_127_to_binary_l138_138800


namespace relationship_of_a_b_c_l138_138243

theorem relationship_of_a_b_c
  (a b c : ℝ)
  (ha : (1 / 3)^a = 2)
  (hb : log 3 b = 1 / 2)
  (hc : c^(-3) = 2) :
  a < c ∧ c < b :=
sorry

end relationship_of_a_b_c_l138_138243


namespace small_cube_volume_l138_138131

theorem small_cube_volume (edge_length : ℝ) (d : ℝ) (s : ℝ) (V : ℝ) 
  (h1 : edge_length = 12)
  (h2 : d = edge_length)
  (h3 : d = s * real.sqrt 3)
  (h4 : V = s^3) : 
  V = 192 * real.sqrt 3 := 
sorry

end small_cube_volume_l138_138131


namespace circumcenter_condition_l138_138324

-- We define an acute triangle ABC with BC < CA < AB and certain perpendiculars.
variable {A B C D E F M N : Type}
variable [euclidean_geometry A B C D E F M N]

-- The conditions
axiom acute_triangle (ABC : triangle) (h1 : acute_triangle ABC)
axiom BC_less_CA : BC < CA
axiom CA_less_AB : CA < AB
axiom perp_AD_BC : perpendicular AD BC D
axiom perp_BE_AC : perpendicular BE AC E
axiom perp_CF_AB : perpendicular CF AB F
axiom FM_parallel_DE : parallel FM DE
axiom FM_intersect_BC_at_M : intersect FM BC M
axiom angle_bisector_MFE_intersects_DE_at_N : angle_bisector MFE DE N

-- The to be proved statement
theorem circumcenter_condition :
  circumcenter F DMN ↔ circumcenter B FMN := by
  sorry

end circumcenter_condition_l138_138324


namespace largest_nat_satisfies_conditions_l138_138824

theorem largest_nat_satisfies_conditions :
  (∃ (x : ℕ), x = 180625 ∧ 
  ¬ (x % 10 = 0) ∧ 
  (∀ (d : ℤ), d ∈ (digits 10 x).drop 1 → (∃ (y : ℕ), y = nat.pred x × d ∧ x % y = 0))) :=
begin
  use 180625,
  split,
  { refl },
  split,
  { norm_num }, -- last digit is not zero
  { intros d hd,
    sorry } -- digit removal and divisibility proof
end

end largest_nat_satisfies_conditions_l138_138824


namespace vegetable_garden_total_l138_138138

noncomputable def total_vegetables (potatoes cucumbers tomatoes peppers carrots : ℕ) : ℕ :=
  potatoes + cucumbers + tomatoes + peppers + carrots

theorem vegetable_garden_total :
  let potatoes := 1200 in
  let cucumbers := potatoes - 160 in
  let tomatoes := 4 * cucumbers in
  let product := cucumbers * tomatoes in
  let peppers := Int.to_nat (Int.sqrt product) in
  let combined_total := cucumbers + tomatoes in
  let carrots := Int.to_nat (Real.to_nnreal (combined_total * 120 / 100)) in
  total_vegetables potatoes cucumbers tomatoes peppers carrots = 14720 :=
by
  sorry

end vegetable_garden_total_l138_138138


namespace Tim_spent_correct_amount_l138_138445

def mealCost : ℝ := 60.50

def tipPercentage : ℝ := 0.20
def stateTaxPercentage : ℝ := 0.05
def cityTaxPercentage : ℝ := 0.03
def surchargePercentage : ℝ := 0.015

-- Function to calculate total cost given the conditions
noncomputable def totalCost : ℝ :=
  let tip := tipPercentage * mealCost
  let stateTax := 0.03.roundup (stateTaxPercentage * mealCost * 100) / 100
  let cityTax := 0.03.roundup (cityTaxPercentage * mealCost * 100) / 100
  let subtotalBeforeSurcharge := mealCost + stateTax + cityTax
  let surcharge := 0.03.roundup (surchargePercentage * subtotalBeforeSurcharge * 100) / 100
  mealCost + tip + stateTax + cityTax + surcharge

theorem Tim_spent_correct_amount :
  totalCost = 78.43 :=
by
  sorry

end Tim_spent_correct_amount_l138_138445


namespace andrew_made_35_sandwiches_l138_138777

-- Define the number of friends and sandwiches per friend
def num_friends : ℕ := 7
def sandwiches_per_friend : ℕ := 5

-- Define the total number of sandwiches and prove it equals 35
def total_sandwiches : ℕ := num_friends * sandwiches_per_friend

theorem andrew_made_35_sandwiches : total_sandwiches = 35 := by
  sorry

end andrew_made_35_sandwiches_l138_138777


namespace total_cookies_baked_l138_138741

theorem total_cookies_baked (num_members : ℕ) (sheets_per_member : ℕ) (cookies_per_sheet : ℕ)
  (h1 : num_members = 100) (h2 : sheets_per_member = 10) (h3 : cookies_per_sheet = 16) :
  num_members * sheets_per_member * cookies_per_sheet = 16000 :=
by
  rw [h1, h2, h3]
  norm_num
  sorry

end total_cookies_baked_l138_138741


namespace probability_factor_of_36_l138_138565

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138565


namespace total_payment_l138_138937

/--
  Jerry took 8 hours painting the house. 
  The time to fix the kitchen counter was three times longer than painting the house.
  Jerry took 6 hours mowing the lawn.
  Jerry charged $15 per hour of work.
  Prove that the total amount of money Miss Stevie paid Jerry is $570.
-/
theorem total_payment (h_paint: ℕ := 8) (h_counter: ℕ := 3 * h_paint) (h_mow: ℕ := 6) (rate: ℕ := 15) :
  let total_hours := h_paint + h_counter + h_mow
  in total_hours * rate = 570 :=
by
  -- provide proof here
  sorry

end total_payment_l138_138937


namespace probability_factor_of_36_l138_138686

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138686


namespace smallest_five_digit_congruent_to_2_mod_17_l138_138076

-- Definitions provided by conditions
def is_five_digit (x : ℕ) : Prop := 10000 ≤ x ∧ x < 100000
def is_congruent_to_2_mod_17 (x : ℕ) : Prop := x % 17 = 2

-- Proving the existence of the smallest five-digit integer satisfying the conditions
theorem smallest_five_digit_congruent_to_2_mod_17 : 
  ∃ x : ℕ, is_five_digit x ∧ is_congruent_to_2_mod_17 x ∧ 
  (∀ y : ℕ, is_five_digit y ∧ is_congruent_to_2_mod_17 y → x ≤ y) := 
begin
  use 10013,
  split,
  { -- Check if it's a five digit number
    unfold is_five_digit,
    exact ⟨by norm_num, by norm_num⟩ },
  split,
  { -- Check if it's congruent to 2 mod 17
    unfold is_congruent_to_2_mod_17,
    exact by norm_num },
  { -- Prove it is the smallest
    intros y hy,
    have h_congruent : y % 17 = 2 := hy.2,
    have h_five_digit : 10000 ≤ y ∧ y < 100000 := hy.1,
    sorry
  }
end

end smallest_five_digit_congruent_to_2_mod_17_l138_138076


namespace average_weight_l138_138305

theorem average_weight (w : ℕ) : 
  (64 < w ∧ w ≤ 67) → w = 66 :=
by sorry

end average_weight_l138_138305


namespace largest_of_five_even_integers_l138_138158

-- Definitions for the conditions
def sum_of_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def sum_of_first_30_even_integers : ℕ :=
  2 * sum_of_first_n_integers 30

def five_consecutive_even_integers_sum (n : ℕ) :=
  (n - 8) + (n - 6) + (n - 4) + (n - 2) + n 

-- The theorem we need to prove
theorem largest_of_five_even_integers : ∃ n : ℕ, 
  five_consecutive_even_integers_sum n = sum_of_first_30_even_integers ∧
  n = 190 :=
begin
  sorry,
end

end largest_of_five_even_integers_l138_138158


namespace slope_of_line_through_focus_l138_138842

theorem slope_of_line_through_focus
  (ecc : ℝ) (hyp_eq : ∀ (x y a b : ℝ), 0 < a → 0 < b → ecc = 2 → (x^2 / a^2 - y^2 / b^2 = 1))
  (focus_eq : ∀ (x y : ℝ), y^2 = 8*x → focus_eq (2, 0))
  (line_inter : ∀ (F2 P Q : ℝ × ℝ), line_through F2 P Q)
  (perp_cond : ∀ (P Q F1 : ℝ × ℝ), orthogonal (P - F1) (Q - F1))
  : ∃ k : ℝ, k = √(7) / 3 ∨ k = -√(7) / 3 :=
sorry

end slope_of_line_through_focus_l138_138842


namespace cloth_sold_l138_138129

theorem cloth_sold (total_sell_price : ℤ) (loss_per_meter : ℤ) (cost_price_per_meter : ℤ) (x : ℤ) 
    (h1 : total_sell_price = 18000) 
    (h2 : loss_per_meter = 5) 
    (h3 : cost_price_per_meter = 50) 
    (h4 : (cost_price_per_meter - loss_per_meter) * x = total_sell_price) : 
    x = 400 :=
by
  sorry

end cloth_sold_l138_138129


namespace cost_comparison_for_30_pens_l138_138083

def cost_store_a (x : ℕ) : ℝ :=
  if x > 10 then 0.9 * x + 6
  else 1.5 * x

def cost_store_b (x : ℕ) : ℝ :=
  1.2 * x

theorem cost_comparison_for_30_pens :
  cost_store_a 30 < cost_store_b 30 :=
by
  have store_a_cost : cost_store_a 30 = 0.9 * 30 + 6 := by rfl
  have store_b_cost : cost_store_b 30 = 1.2 * 30 := by rfl
  rw [store_a_cost, store_b_cost]
  sorry

end cost_comparison_for_30_pens_l138_138083


namespace radius_of_circle_is_13_l138_138410

def diameter : ℝ := 26
def radius (d : ℝ) : ℝ := d / 2

theorem radius_of_circle_is_13 :
  radius diameter = 13 := by
  sorry

end radius_of_circle_is_13_l138_138410


namespace determine_angle_B_l138_138333

variable (A B C : Type) [metric_space B]

noncomputable def angle_B (BC AC : ℝ) (angle_A : ℝ) : ℝ :=
  let angle_B := 1/6 * real.pi;
  if BC = 3 ∧ AC = real.sqrt 3 ∧ angle_A = real.pi / 3 then angle_B else 0

theorem determine_angle_B (BC AC : ℝ) (angle_A : ℝ) (h1 : BC = 3) (h2 : AC = real.sqrt 3) (h3 : angle_A = real.pi / 3) :
  angle_B BC AC angle_A = real.pi / 6 :=
begin
  simp [angle_B, h1, h2, h3],
  sorry
end

end determine_angle_B_l138_138333


namespace probability_factor_of_36_l138_138679

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138679


namespace polygon_interior_angle_sum_l138_138758

-- Definition of each interior angle in degrees
def interior_angle_deg (n : ℕ) : ℝ := 108

-- Sum of the interior angles for an n-sided polygon
def sum_of_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

theorem polygon_interior_angle_sum (h : interior_angle_deg 5 = 108) : 
  sum_of_interior_angles 5 = 540 :=
begin
  sorry
end

end polygon_interior_angle_sum_l138_138758


namespace reflect_across_x_axis_l138_138917

theorem reflect_across_x_axis (x y : ℝ) (A : ℝ × ℝ) (hA : A = (-5, -2)) :
  (x, y) = (-5, 2) :=
begin
  sorry
end

end reflect_across_x_axis_l138_138917


namespace ten_sided_game_winner_twelve_sided_game_winner_l138_138719

-- Proof Problem (a): 10-sided Polygon
theorem ten_sided_game_winner (n : ℕ) (h : n = 10) 
  (alternating_colors : ∀ i ∈ (Finset.range n), even i ↔ color i = black)
  (game_rule : ∀ segments : Finset (Fin n × Fin n),
               ∀ s ∈ segments,
               ∀ t ∈ segments,
               s ≠ t → disjoint s t)
  (optimal_play : ∀ strategy : (Fin n × Fin n) → Prop, 
                   player_strategy strategy n (1%2 = 1)) :
  player_wins optimal_play n (2%2 = 0) :=
sorry

-- Proof Problem (b): 12-sided Polygon
theorem twelve_sided_game_winner (n : ℕ) (h : n = 12) 
  (alternating_colors : ∀ i ∈ (Finset.range n), even i ↔ color i = black)
  (game_rule : ∀ segments : Finset (Fin n × Fin n),
               ∀ s ∈ segments,
               ∀ t ∈ segments,
               s ≠ t → disjoint s t)
  (optimal_play : ∀ strategy : (Fin n × Fin n) → Prop, 
                   player_strategy strategy n (0%2 = 0)) :
  player_wins optimal_play n (1%2 = 0) :=
sorry

end ten_sided_game_winner_twelve_sided_game_winner_l138_138719


namespace three_horsemen_single_overtake_point_ten_horsemen_single_overtake_point_thirty_three_horsemen_single_overtake_point_l138_138089

/-- Given three horsemen moving counterclockwise on a circular road with distinct constant speeds, 
    they can ride indefinitely such that there is only one point on the road where they overtake each other. -/
theorem three_horsemen_single_overtake_point 
    (u v w : ℝ) (huv : u ≠ v) (huw : u ≠ w) (hvw : v ≠ w) :
    ∃ (p : ℝ), ∀ t : ℝ, (u * t) % p = (v * t) % p = (w * t) % p :=
sorry

/-- Given ten horsemen moving counterclockwise on a circular road with distinct constant speeds, 
    they can ride indefinitely such that there is only one point on the road where they overtake each other. -/
theorem ten_horsemen_single_overtake_point 
    (speeds : Fin (10) → ℝ) 
    (distinct_speeds : ∀ (i j : Fin 10), i ≠ j → speeds i ≠ speeds j) :
    ∃ (p : ℝ), ∀ t : ℝ, ∀ i : Fin 10, (speeds i * t) % p = (speeds 0 * t) % p :=
sorry

/-- Given thirty-three horsemen moving counterclockwise on a circular road with distinct constant speeds, 
    they can ride indefinitely such that there is only one point on the road where they overtake each other. -/
theorem thirty_three_horsemen_single_overtake_point 
    (speeds : Fin (33) → ℝ) 
    (distinct_speeds : ∀ (i j : Fin 33), i ≠ j → speeds i ≠ speeds j) :
    ∃ (p : ℝ), ∀ t : ℝ, ∀ i : Fin 33, (speeds i * t) % p = (speeds 0 * t) % p :=
sorry

end three_horsemen_single_overtake_point_ten_horsemen_single_overtake_point_thirty_three_horsemen_single_overtake_point_l138_138089


namespace greatest_possible_length_l138_138456

-- Definitions of the given lengths in meters
def length1_m : ℝ := 7290
def length2_m : ℝ := 12425
def length3_m : ℝ := 321.75

-- Convert lengths to centimeters
def length1_cm : ℝ := length1_m * 100
def length2_cm : ℝ := length2_m * 100
def length3_cm : ℝ := length3_m * 100

-- Definition of the greatest common divisor in centimeters
def gcd_cm : ℝ := Real.gcd (Real.gcd length1_cm length2_cm) length3_cm

-- Proof statement to be proven true
theorem greatest_possible_length : gcd_cm = 225 := sorry

end greatest_possible_length_l138_138456


namespace opposite_numbers_option_A_l138_138774

def is_opposite (a b : ℝ) : Prop := a = -b

theorem opposite_numbers_option_A :
  is_opposite (-real.sqrt 9) (real.cbrt 27) ∧
  ¬is_opposite (real.cbrt (-8)) (-real.cbrt 8) ∧
  ¬is_opposite (real.abs (-real.sqrt 2)) (real.sqrt 2) ∧
  ¬is_opposite (real.sqrt 2) (real.cbrt (-8)) :=
by
  sorry

end opposite_numbers_option_A_l138_138774


namespace simplest_fraction_sum_l138_138047

theorem simplest_fraction_sum (c d : ℕ) (h1 : 0.325 = (c:ℚ)/d) (h2 : Int.gcd c d = 1) : c + d = 53 :=
by sorry

end simplest_fraction_sum_l138_138047


namespace factor_probability_36_l138_138605

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138605


namespace find_c_interval_l138_138114

def quadratic_function (c : ℝ) (x : ℝ) : ℝ := x^2 - c * x + c

def fixed_points (f : ℝ → ℝ) : Set ℝ :=
  { x | f x = x }

def four_distinct_fixed_points (f : ℝ → ℝ) : Prop :=
  (fixed_points f).card = 4

def value_of_c (c : ℝ) : Prop :=
  let f := quadratic_function c
  four_distinct_fixed_points (f ∘ f)

theorem find_c_interval :
  ∀ c : ℝ, value_of_c c ↔ (c < -1 ∨ c > 3) :=
sorry

end find_c_interval_l138_138114


namespace planes_PKF_TDH_no_intersect_along_PT_l138_138898

noncomputable def quadrilateral_pyramid (M A B C D : Type) : Prop :=
  (∃ (ABCD : Type), parallelogram ABCD)

noncomputable def point_not_in_planes (T M C D A B : Type) : Prop :=
  ¬ (T ∈ plane M C D) ∧ ¬ (T ∈ plane A B C)
  
noncomputable def planes_do_not_intersect_along_PT (P K F T D H : Type) : Prop :=
  (∀ (PT : line P T), ¬ (plane P K F) ∩ (plane T D H) = PT)

theorem planes_PKF_TDH_no_intersect_along_PT
  (M A B C D T P K F H : Type)
  (h1 : quadrilateral_pyramid M A B C D)
  (h2 : point_not_in_planes T M C D A B)
  : planes_do_not_intersect_along_PT P K F T D H :=
sorry

end planes_PKF_TDH_no_intersect_along_PT_l138_138898


namespace chelsea_sugar_problem_l138_138166

variable (initial_sugar : ℕ)
variable (num_bags : ℕ)
variable (sugar_lost_fraction : ℕ)

def remaining_sugar (initial_sugar : ℕ) (num_bags : ℕ) (sugar_lost_fraction : ℕ) : ℕ :=
  let sugar_per_bag := initial_sugar / num_bags
  let sugar_lost := sugar_per_bag / sugar_lost_fraction
  let remaining_bags_sugar := (num_bags - 1) * sugar_per_bag
  remaining_bags_sugar + (sugar_per_bag - sugar_lost)

theorem chelsea_sugar_problem : 
  remaining_sugar 24 4 2 = 21 :=
by
  sorry

end chelsea_sugar_problem_l138_138166


namespace conditional_probability_l138_138106

-- Definitions and conditions
def total_products := 4
def first_class_products := 3
def second_class_products := 1
def draws := 2

-- Events definition
def event_A (draw_seq : List ℕ) : Prop := 
  draw_seq.length = 2 ∧ draw_seq.head? = some 1

def event_B (draw_seq : List ℕ) : Prop := 
  draw_seq.length = 2 ∧ draw_seq.tail.head? = some 1

-- Desired probability statement
theorem conditional_probability : 
  ∀ (draw_seq : List ℕ), event_A draw_seq → event_B draw_seq → 
  (probability (event_B | event_A) = 2/3) :=
begin
  -- Placeholder for the proof
  sorry
end

end conditional_probability_l138_138106


namespace set_intersection_l138_138273

variable (U : Set Int)
variable (A B : Set Int)
variable (CU B : Set Int)

-- Conditions
def UnivSet : U = {-1, 0, 1, 2, 3}
def SetA : A = {0, 1, 2}
def SetB : B = {2, 3}
def CompB : CU B = {-1, 0, 1}

-- Proposition to prove
theorem set_intersection : (A ∩ (CU B)) = {0, 1} :=
by 
  have h1 : U = {-1, 0, 1, 2, 3} := UnivSet;
  have h2 : A = {0, 1, 2} := SetA;
  have h3 : B = {2, 3} := SetB;
  have h4 : CU B = {-1, 0, 1} := CompB;
  sorry

end set_intersection_l138_138273


namespace numerator_when_x_fraction_y_l138_138897

theorem numerator_when_x_fraction_y 
  (x y a : ℝ)
  (h1: x + y = -10)
  (h2: x = a / y)
  (h3: x ^ 2 + y ^ 2 = 50) :
  a = ∣25∣ := 
sorry

end numerator_when_x_fraction_y_l138_138897


namespace handshake_count_l138_138442

-- Define the conditions
def num_companies : ℕ := 5
def reps_per_company : ℕ := 4
def total_people : ℕ := num_companies * reps_per_company
def handshakes_per_person : ℕ := total_people - 1 - (reps_per_company - 1)

-- Define the theorem to prove
theorem handshake_count : 
  total_people * (total_people - 1 - (reps_per_company - 1)) / 2 = 160 := 
by
  -- Here should be the proof, but it is omitted
  sorry

end handshake_count_l138_138442


namespace range_of_m_l138_138835

-- Definitions
def f (x : ℝ) : ℝ := x^2

def g (x : ℝ) (m : ℝ) : ℝ := (1 / 2)^x - m

-- The Lean theorem statement
theorem range_of_m (m : ℝ) :
  (∀ x1 ∈ Set.Icc 0 2, ∃ x2 ∈ Set.Icc 1 2, f x1 ≥ g x2 m) → m ≥ 1 / 4 :=
by
  intros h
  -- Proof is skipped
  sorry

end range_of_m_l138_138835


namespace distinct_real_numbers_cubed_sum_l138_138353

theorem distinct_real_numbers_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_eq : ∀ x ∈ {a, b, c}, (x^3 + 12) / x = (a^3 + 12) / a) : 
  a^3 + b^3 + c^3 = -36 :=
by
  sorry

end distinct_real_numbers_cubed_sum_l138_138353


namespace maximize_profit_l138_138179

theorem maximize_profit 
  (cost_per_product : ℝ)
  (initial_price : ℝ)
  (initial_sales : ℝ)
  (price_increase_effect : ℝ)
  (daily_sales_decrease : ℝ)
  (max_profit_price : ℝ)
  (max_profit : ℝ)
  :
  cost_per_product = 8 ∧ initial_price = 10 ∧ initial_sales = 100 ∧ price_increase_effect = 1 ∧ daily_sales_decrease = 10 → 
  max_profit_price = 14 ∧
  max_profit = 360 :=
by 
  intro h
  have h_cost := h.1
  have h_initial_price := h.2.1
  have h_initial_sales := h.2.2.1
  have h_price_increase_effect := h.2.2.2.1
  have h_daily_sales_decrease := h.2.2.2.2
  sorry

end maximize_profit_l138_138179


namespace Alyssa_spent_on_toys_l138_138773

theorem Alyssa_spent_on_toys (price_football price_marbles total_spent : ℝ) 
  (h1 : price_football = 5.71)
  (h2 : price_marbles = 6.59)
  (h3 : total_spent = 5.71 + 6.59) : 
  total_spent = 12.30 := 
by 
  rw [h1, h2] at h3
  exact h3

end Alyssa_spent_on_toys_l138_138773


namespace calculateSurfaceArea_l138_138406

noncomputable def totalSurfaceArea (r : ℝ) : ℝ :=
  let hemisphereCurvedArea := 2 * Real.pi * r^2
  let cylinderLateralArea := 2 * Real.pi * r * r
  hemisphereCurvedArea + cylinderLateralArea

theorem calculateSurfaceArea :
  ∃ r : ℝ, (Real.pi * r^2 = 144 * Real.pi) ∧ totalSurfaceArea r = 576 * Real.pi :=
by
  exists 12
  constructor
  . sorry -- Proof that 144π = π*12^2 can be shown
  . sorry -- Proof that 576π = 288π + 288π can be shown

end calculateSurfaceArea_l138_138406


namespace probability_factor_36_l138_138596

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138596


namespace question_1_question_2_l138_138232

noncomputable def f (m x : ℝ) : ℝ := (m * x - 1) * Real.exp x - x ^ 2
noncomputable def f_prime (m x : ℝ) : ℝ := m * Real.exp(x) * (1 + x) - Real.exp(x) - 2 * x

theorem question_1 (m : ℝ) (h : f_prime m 1 = Real.exp 1 - 2) :
  (∀ x, f_prime 1 x > 0 ↔ x < 0 ∨ x > Real.log 2) ∧
  (∀ x, f_prime 1 x < 0 ↔ 0 < x ∧ x < Real.log 2) :=
sorry

theorem question_2 (m : ℝ) :
  (∀ x, f 1 x < -x^2 + 1*x - m ↔ (Int.exists_succ' (λ x, f 1 x < -x^2 + x - m) ∧ Int.exists_pred' (λ x, f 1 x < -x^2 + x - m))) ↔
  (Real.exp 2 / (2 * Real.exp 2 - 1) ≤ m ∧ m < 1) :=
sorry

end question_1_question_2_l138_138232


namespace probability_factor_of_36_l138_138567

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138567


namespace simplify_complex_fraction_l138_138018

theorem simplify_complex_fraction :
  (⟨(3 : ℂ), 5⟩ / ⟨-2, 7⟩) = (⟨29 / 53, -31 / 53⟩ : ℂ) :=
sorry

end simplify_complex_fraction_l138_138018


namespace probability_factor_of_36_l138_138671

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138671


namespace probability_divisor_of_36_is_one_fourth_l138_138462

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138462


namespace candy_box_original_price_l138_138219

theorem candy_box_original_price (P : ℝ) (h1 : 1.25 * P = 20) : P = 16 :=
sorry

end candy_box_original_price_l138_138219


namespace largest_possible_N_l138_138983

theorem largest_possible_N : 
  ∃ (N : ℕ), (∀ (d : ℕ), d ∣ N → d = 1 ∨ d = N ∨ ∃ (k : ℕ), k * d = N)
    ∧ (∃ (p q r : ℕ), 1 < p ∧ p < q ∧ q < r ∧ q * 21 = r 
      ∧ [1, p, q, r, N / p, N / q, N / r, N] = multiset.sort has_dvd.dvd [1, p, q, N / q, (N / r), N / p, r, N])
    ∧ N = 441 := sorry

end largest_possible_N_l138_138983


namespace determinant_of_cross_product_matrix_l138_138351

noncomputable theory

variables (u v w : ℝ^3)

def E : ℝ := (u ⬝ (v × w))

def A : Matrix (Fin 3) (Fin 3) ℝ :=
  ![u × v, v × w, w × u]

theorem determinant_of_cross_product_matrix :
  Matrix.det A = E^2 :=
  sorry

end determinant_of_cross_product_matrix_l138_138351


namespace seating_arrangements_count_l138_138142

inductive Person : Type
| Alice | Bob | Carla | Derek | Eric | Fiona

open Person

def isNextTo (p1 p2 : Person) (arrangement : List Person) : Prop :=
  ∃ i, i < arrangement.length - 1 ∧ (arrangement.get i = p1 ∧ arrangement.get (i + 1) = p2 ∨ 
                                     arrangement.get i = p2 ∧ arrangement.get (i + 1) = p1)

def satisfiesConditions (arrangement : List Person) : Prop :=
  ¬isNextTo Alice Bob arrangement ∧ 
  ¬isNextTo Alice Carla arrangement ∧ 
  ¬isNextTo Derek Eric arrangement ∧ 
  arrangement.get 0 ≠ Fiona ∧ arrangement.get 5 ≠ Fiona

theorem seating_arrangements_count : 
  ∃ (arrangements : List (List Person)),
    arrangements.length = 16 ∧
    ∀ arrangement ∈ arrangements, arrangement.length = 6 ∧ satisfiesConditions arrangement :=
sorry

end seating_arrangements_count_l138_138142


namespace number_of_intersections_l138_138059

theorem number_of_intersections (k : ℕ) (hk : k ≥ 4) : 
  (number_of_segments_intersections k) = (k.choose 4) :=
sorry

end number_of_intersections_l138_138059


namespace distinct_real_numbers_cubed_sum_l138_138354

theorem distinct_real_numbers_cubed_sum (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (h_eq : ∀ x ∈ {a, b, c}, (x^3 + 12) / x = (a^3 + 12) / a) : 
  a^3 + b^3 + c^3 = -36 :=
by
  sorry

end distinct_real_numbers_cubed_sum_l138_138354


namespace cost_of_shorts_l138_138932

-- Define the given conditions and quantities
def initial_money : ℕ := 50
def jerseys_cost : ℕ := 5 * 2
def basketball_cost : ℕ := 18
def remaining_money : ℕ := 14

-- The total amount spent
def total_spent : ℕ := initial_money - remaining_money

-- The total cost of the jerseys and basketball
def jerseys_basketball_cost : ℕ := jerseys_cost + basketball_cost

-- The cost of the shorts
def shorts_cost : ℕ := total_spent - jerseys_basketball_cost

theorem cost_of_shorts : shorts_cost = 8 := sorry

end cost_of_shorts_l138_138932


namespace part_a_part_b_l138_138966

variables {n m : ℕ}
variables (A : fin m → finset (fin n))
variable (h1A : ∀ i, (A i).card = 3)
variable (h2A : ∀ i j, i < j → (A i) ∩ (A j)).card ≤ 1

theorem part_a (h1A : ∀ i, (A i).card = 3) (h2A : ∀ i j, i < j → (A i ∩ A j).card ≤ 1) :
  m ≤ n * (n - 1) / 6 :=
sorry

theorem part_b (nn3 : n ≥ 3) :
  ∃ (A : fin (n.choose 3) → finset (fin n)),
    (∀ i, (A i).card = 3) ∧
    (∀ i j, i < j → (A i ∩ A j).card ≤ 1) ∧
    (n.choose 3 ≥ (n - 1) * (n - 2) / 6) :=
sorry

end part_a_part_b_l138_138966


namespace probability_factor_of_36_l138_138652

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138652


namespace octal_to_binary_conversion_l138_138802

theorem octal_to_binary_conversion :
  ∃ b : ℕ, octal_to_decimal 127 = b ∧ decimal_to_binary b = 1010111 :=
by
  sorry

-- Supporting definitions that capture the concepts used in the problem
def octal_to_decimal (o : ℕ) : ℕ :=
  -- Implement the conversion of an octal number (represented as a natural number) to a decimal number
  sorry

def decimal_to_binary (d : ℕ) : ℕ :=
  -- Implement the conversion of a decimal number to a binary number (represented as a natural number)
  sorry

end octal_to_binary_conversion_l138_138802


namespace geometric_sequence_sum_l138_138922

-- Definition of a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

-- Main statement to prove
theorem geometric_sequence_sum (a : ℕ → ℝ) (q : ℝ)
  (h1 : is_geometric_sequence a q)
  (h2 : a 1 + a 2 = 40)
  (h3 : a 3 + a 4 = 60) :
  a 7 + a 8 = 135 :=
sorry

end geometric_sequence_sum_l138_138922


namespace wrapping_paper_area_l138_138111

theorem wrapping_paper_area (w h : ℝ) : 
  let box_length := 2 * w in
  let box_width := w in
  let box_height := h in
  let wrapping_paper_area := 2 * w^2 + 3 * w * h in
  wrapping_paper_area = 2 * w^2 + 3 * w * h :=
by
  let box_length := 2 * w
  let box_width := w
  let box_height := h
  let wrapping_paper_area := 2 * w^2 + 3 * w * h
  sorry

end wrapping_paper_area_l138_138111


namespace number_of_starting_positions_l138_138352

variables (x y : ℝ)

def hyperbola (x y : ℝ) := y^2 - 4 * x^2 = 4

def line (x_n x : ℝ) := 2 * x - 2 * x_n

theorem number_of_starting_positions :
  (∃ x₀ : ℝ, ∀ n, let x_n := (n % 2 = 0 : ℝ) in P (x_n, 0) ∧
    vertical_projection (intersection (line x_n x) (hyperbola x y)) = y
    ∧ P_n = P_512) →
    2 :=
sorry

end number_of_starting_positions_l138_138352


namespace geometric_sequence_common_ratio_l138_138328

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (q : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * q)
  (h_a1 : a 1 = 1/2) 
  (h_a4 : a 4 = -4) : 
  q = -2 := 
by
  sorry

end geometric_sequence_common_ratio_l138_138328


namespace seventh_number_is_177_l138_138144

def digits_are_positive_integers (n: ℕ) : Prop :=
  ∀ d ∈ (nat.digits 10 n), d > 0

def sum_of_digits (n: ℕ) : ℕ :=
  (nat.digits 10 n).sum

def satisfies_conditions (n: ℕ) : Prop :=
  digits_are_positive_integers n ∧ sum_of_digits n = 15

def nth_number (n: ℕ) : ℕ :=
  nat.find (λ m, satisfies_conditions m ∧ list.sorted (nat.lt) (nat.digits 10 m) ∧ m = n)

theorem seventh_number_is_177 : nth_number 7 = 177 :=
by
  sorry

end seventh_number_is_177_l138_138144


namespace factor_probability_l138_138629

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138629


namespace part1_part2_part3_l138_138852

-- Definition of a (k, m)-sequence
def is_km_sequence (k m : ℕ) (a : ℕ → ℕ) : Prop :=
  (∀ n, 1 ≤ a n ∧ a n ≤ k) ∧
  ∀ (s : Finset ℕ), s.card = m → 
  ∀ (perm : List ℕ), (∀ x ∈ perm, x ∈ s) → ∃ t : ℕ → ℕ, Subseq t (a ∘ perm.to_finset.to_list)

-- Example Sequences A1 and A2
def A1 := [1, 2, 3, 1, 2, 3, 1, 2, 3]
def A2 := [1, 2, 3, 2, 1, 3, 1]

-- Prove A1 is a (3,3)-sequence and A2 is not
theorem part1 (h1 : is_km_sequence 3 3 A1) (h2 : ¬ is_km_sequence 3 3 A2) : 
  True := sorry

-- Prove that G(k,2) = 2k - 1
theorem part2 (k : ℕ) (hk : 2 ≤ k) : 
  G(k,2) = 2k-1 := sorry

-- Prove that G(4,4) = 12
theorem part3 : 
  G(4,4) = 12 := sorry

end part1_part2_part3_l138_138852


namespace area_of_inscribed_rectangle_l138_138109

open Real

theorem area_of_inscribed_rectangle (r l w : ℝ) (h_radius : r = 7) 
  (h_ratio : l / w = 3) (h_width : w = 2 * r) : l * w = 588 :=
by
  sorry

end area_of_inscribed_rectangle_l138_138109


namespace probability_factor_36_l138_138601

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138601


namespace find_three_digit_number_l138_138201

theorem find_three_digit_number :
  ∃ (Π В Γ : ℕ), (Π ≠ В ∧ В ≠ Γ ∧ Π ≠ Γ) 
               ∧ (0 ≤ Π ∧ Π ≤ 9) 
               ∧ (0 ≤ В ∧ В ≤ 9) 
               ∧ (0 ≤ Γ ∧ Γ ≤ 9) 
               ∧ (100 * Π + 10 * В + Γ = (Π + В + Γ) * ((Π + В + Γ) + 1))
               ∧ (100 * Π + 10 * В + Γ = 156) :=
by {
  sorry
}

end find_three_digit_number_l138_138201


namespace probability_factor_of_36_l138_138581

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138581


namespace Kishore_misc_expense_l138_138771

theorem Kishore_misc_expense:
  let savings := 2400
  let percent_saved := 0.10
  let rent := 5000
  let milk := 1500
  let groceries := 4500
  let education := 2500
  let petrol := 2000
  let total_salary := savings / percent_saved 
  let total_spent := rent + milk + groceries + education + petrol
  total_salary - (total_spent + savings) = 6100 := 
by
  sorry

end Kishore_misc_expense_l138_138771


namespace distribution_schemes_l138_138738

theorem distribution_schemes :
  let employees := 8 in
  let translators := 2 in
  let rest_employees := employees - translators in
  let department_size := employees / 2 in
  let ways_to_assign_rest_employees := Nat.choose rest_employees department_size in
  let ways_to_assign_translators := 2 in
  (ways_to_assign_rest_employees * ways_to_assign_translators) = 40 :=
by
  let employees := 8
  let translators := 2
  let rest_employees := employees - translators
  let department_size := employees / 2
  let ways_to_assign_rest_employees := Nat.choose rest_employees department_size
  let ways_to_assign_translators := 2
  have h : ways_to_assign_rest_employees = 20 := by sorry
  rw h
  have h2 : (20 * ways_to_assign_translators) = 40 := by norm_num
  exact h2

end distribution_schemes_l138_138738


namespace ultratown_run_difference_l138_138307

/-- In Ultratown, the streets are all 25 feet wide, 
and the blocks they enclose are rectangular with lengths of 500 feet and widths of 300 feet. 
Hannah runs around the block on the longer 500-foot side of the street, 
while Harry runs on the opposite, outward side of the street. 
Prove that Harry runs 200 more feet than Hannah does for every lap around the block.
-/ 
theorem ultratown_run_difference :
  let street_width : ℕ := 25
  let inner_length : ℕ := 500
  let inner_width : ℕ := 300
  let outer_length := inner_length + 2 * street_width
  let outer_width := inner_width + 2 * street_width
  let inner_perimeter := 2 * (inner_length + inner_width)
  let outer_perimeter := 2 * (outer_length + outer_width)
  (outer_perimeter - inner_perimeter) = 200 :=
by
  sorry

end ultratown_run_difference_l138_138307


namespace probability_factor_of_36_l138_138566

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138566


namespace trapezoid_property_l138_138100

open EuclideanGeometry

variables {A B C D K M N : Point}

noncomputable def midpoint (P Q : Point) : Point := sorry

theorem trapezoid_property 
  (h1 : parallel A D B C) 
  (h2 : dist A D = dist D B) 
  (h3 : dist D B = dist D C)
  (h4 : angle B C D = 72) 
  (h5 : dist A D = dist A K) 
  (h6 : K ≠ D) 
  (h7 : midpoint C D = M) 
  (h8 : line_through A M N ∧ line_through B D N): 
  dist B K = dist N D :=
sorry

end trapezoid_property_l138_138100


namespace problem_statement_l138_138220

variable {α : Type} [LinearOrder α] [TopologicalSpace α] [TopologicalAddGroup α]
variable {f : α → α} (a b : α)

theorem problem_statement (h_diff : Differentiable ℝ f)
  (h_order : a > b ∧ b > 1)
  (h_cond : ∀ x, (x - 1) * (f' x : ℝ) ≥ 0) : 
  f(a) + f(b) ≥ 2 * f(1) := 
by 
  sorry

end problem_statement_l138_138220


namespace max_value_of_f_l138_138206

noncomputable def f (x : ℝ) : ℝ := 3^x - 9^x

theorem max_value_of_f : ∃ x : ℝ, f x = 1 / 4 := sorry

end max_value_of_f_l138_138206


namespace probability_of_foci_on_x_axis_l138_138412

def C_equation (m n : ℕ) := (∃ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1)
def Foci_on_x_axis (m n : ℕ) := m > n

theorem probability_of_foci_on_x_axis
    (m n : ℕ)
    (hmn_range : m ∈ {1, 2, 3, 4, 5, 6} ∧ n ∈ {1, 2, 3, 4, 5, 6})
    (A : Prop)
    (hA : A ↔ Foci_on_x_axis m n) :
  (∃ A, (∃! m n, C_equation m n ∧ A) ∧ A) →
  (fraction_of_success : ℕ := 
    (∑ k in finset.range 7, finset.card (finset.Icc 1 (k-1))))
   / 36 = 5 / 12 :=
by sorry

end probability_of_foci_on_x_axis_l138_138412


namespace handshakes_exchanged_l138_138394

/-- Seven couples are at a conference. Each person shakes hands exactly once with everyone else except their spouse and the last person they shook hands with. The total number of handshakes exchanged is 77. -/
theorem handshakes_exchanged (people : Finset ℕ) (couples : Finset (Finset ℕ)) :
  people.card = 14 →
  (∀ c ∈ couples, c.card = 2) →
  (∃ t ∈ finset.pairs people, ∀ p ∈ t, ∀ q ∈ couples.filter (λ c, p ∉ c), q.card - 1 ≥ 0) →
  E (handshake_count people couples) = 77 :=
sorry

end handshakes_exchanged_l138_138394


namespace Petya_never_loses_in_simple_state_Vasya_always_wins_in_complex_state_l138_138908

variables {V : Type} [fintype V] [decidable_eq V]
variables (G : simple_graph V) [decidable_rel G.adj]
variables (startr : V → V → Prop)

def simple_state : Prop := ∀ u v, ∃! p : G.walk u v, p.is_path
def complex_state : Prop := ∃ c : G.walk, c.is_cycle

noncomputable def Petya_strategy_simple (h: simple_state G) := sorry
noncomputable def Vasya_strategy_complex (h: complex_state G) := sorry

theorem Petya_never_loses_in_simple_state (G: simple_graph V) :
  simple_state G → noncomputable (Petya_strategy_simple h).never_loses :=
  sorry

theorem Vasya_always_wins_in_complex_state (G: simple_graph V) :
  complex_state G → noncomputable (Vasya_strategy_complex h).always_wins :=
  sorry

end Petya_never_loses_in_simple_state_Vasya_always_wins_in_complex_state_l138_138908


namespace optimal_production_transformers_l138_138770

def transformers_optimal_production : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
λ (x y p thrA thrB ix iy),
  5 * thrA + 3 * thrB ≤ ix ∧
  3 * thrA + 2 * thrB ≤ iy ∧
  x * thrA = 1 ∧ y * thrB = 149 ∧
  p = 12 * thrA + 10 * thrB ∧
  12 * 1 + 10 * 149 = 1502

theorem optimal_production_transformers :
  transformers_optimal_production 5 3 1502 1 149 481 301 :=
by
  sorry

end optimal_production_transformers_l138_138770


namespace gcd_inequality_l138_138850

theorem gcd_inequality (a b c : ℕ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) :
  nat.gcd (a * b + 1) (nat.gcd (a * c + 1) (b * c + 1)) ≤ (a + b + c) / 3 :=
  sorry

end gcd_inequality_l138_138850


namespace trig_identity_l138_138084

theorem trig_identity :
  sin^2 (π / 8) + cos^2 (3 * π / 8) + sin^2 (5 * π / 8) + cos^2 (7 * π / 8) = 2 :=
by
  -- Proof omitted, using 'sorry' as placeholder.
  sorry

end trig_identity_l138_138084


namespace triangle_angle_calculation_l138_138315

theorem triangle_angle_calculation {a b c : ℝ} (h : (a + b + c) * (a + b - c) = 4 * a * b) :
  ∀ C : ℝ, (C = 0 ∨ C = 360) ↔ (∀ C, c^2 = a^2 + b^2 - 2 * a * b * real.cos C) :=
by sorry

end triangle_angle_calculation_l138_138315


namespace probability_factor_of_36_l138_138584

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138584


namespace probability_divisor_of_36_l138_138659

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138659


namespace maximization_of_function_l138_138459

def f (t : ℝ) : ℝ := ((2^t - 5 * t) * t) / 4^t

theorem maximization_of_function :
  ∃ t_max : ℝ, ∀ t : ℝ, f t ≤ f t_max ∧ f t_max = 1 / 20 :=
sorry

end maximization_of_function_l138_138459


namespace probability_factor_36_l138_138509

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138509


namespace binary_operation_correct_l138_138155

theorem binary_operation_correct:
  let b1 := 0b110010
  let b2 := 0b1101
  let b3 := 0b101 
  in ((b1 * b2) / b3) = 0b11110100 := by
  sorry

end binary_operation_correct_l138_138155


namespace emails_left_in_inbox_l138_138027

theorem emails_left_in_inbox
  (initial_emails : ℕ := 500)
  (trash_percentage : ℕ := 50) -- 50%
  (work_percentage : ℕ := 40) -- 40%
  (personal_percentage : ℕ := 25) -- 25%
  (miscellaneous_percentage : ℕ := 10) -- 10%
  (rounding : ℕ → ℕ) -- Placeholder for rounding function
  : initial_emails > 0 → 
    let half_emails := initial_emails / 2;
    let remaining_after_trash := initial_emails - half_emails;
    let to_work := work_percentage * remaining_after_trash / 100;
    let remaining_after_work := remaining_after_trash - to_work;
    let to_personal := rounding (personal_percentage * remaining_after_work / 100);
    let remaining_after_personal := remaining_after_work - to_personal;
    let to_miscellaneous := rounding (miscellaneous_percentage * remaining_after_personal / 100);
    let final_remaining := remaining_after_personal - to_miscellaneous
    in final_remaining = 102 :=
  sorry

end emails_left_in_inbox_l138_138027


namespace concurrency_La_Lb_Lc_l138_138946

open EuclideanGeometry

noncomputable def triangle {α : Type*} [ordered_field α]:
  Type := {A B C : pt α}

noncomputable def incenter {α : Type*} [ordered_field α] 
  (Δ : triangle α): pt α := sorry -- definition of incenter

noncomputable def excenter {α : Type*} [ordered_field α] 
  (Δ : triangle α) (A' : pt α): pt α := sorry -- definition of excenter for a vertex A'

noncomputable def orthocenter {α : Type*} [ordered_field α] 
  (A B C : pt α): pt α := sorry -- definition of orthocenter for any triangle

noncomputable def define_La {α : Type*} [ordered_field α]
  (A B C I Ia : pt α): line α := 
line_through (orthocenter I B C) (orthocenter Ia B C)

theorem concurrency_La_Lb_Lc {α : Type*} [ordered_field α]
  (A B C I Ia Ib Ic : pt α)
  (h_incenter : I = incenter ⟨A, B, C⟩)
  (h_Ia : Ia = excenter ⟨A, B, C⟩ A)
  (h_Ib : Ib = excenter ⟨A, B, C⟩ B)
  (h_Ic : Ic = excenter ⟨A, B, C⟩ C)
  (L_a := define_La A B C I Ia)
  (L_b := define_La B C A I Ib)
  (L_c := define_La C A B I Ic):
  concurrent L_a L_b L_c :=
sorry -- proof


end concurrency_La_Lb_Lc_l138_138946


namespace solve_semi_integer_eq_l138_138975

def is_semi_integer (x : ℝ) : Prop := ∃ (k : ℤ), x = k / 2

def semi_integer_part (x : ℝ) : ℝ := (floor (2 * x)) / 2

theorem solve_semi_integer_eq (x : ℝ) (h1 : is_semi_integer x) (h2 : x^2 + 2 * semi_integer_part x = 6) :
  x = sqrt 3 ∨ x = -sqrt 14 := 
sorry

end solve_semi_integer_eq_l138_138975


namespace bombarded_percentage_l138_138316

theorem bombarded_percentage (P : ℕ) (final_population : ℕ) (x : ℝ) 
  (hP : P = 3800) (h_final : final_population = 2907) :
  P - (x / 100) * P - 0.15 * (P - (x / 100) * P) = final_population → 
  x ≈ 10 :=
by {
  intros,
  sorry
}

end bombarded_percentage_l138_138316


namespace reciprocal_of_neg3_l138_138422

theorem reciprocal_of_neg3 : 1 / (-3 : ℝ) = - (1 / 3) :=
by
  sorry

end reciprocal_of_neg3_l138_138422


namespace smallest_value_of_y1_y2_y3_sum_l138_138361

noncomputable def y_problem := 
  ∃ (y1 y2 y3 : ℝ), 
  (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)

theorem smallest_value_of_y1_y2_y3_sum :
  (∃ (y1 y2 y3 : ℝ), 0 < y1 ∧ 0 < y2 ∧ 0 < y3 ∧ (y1 + 3 * y2 + 5 * y3 = 120) 
  ∧ (y1^2 + y2^2 + y3^2 = 720 / 7)) :=
by 
  sorry

end smallest_value_of_y1_y2_y3_sum_l138_138361


namespace problem_l138_138228

variable (m : ℝ)

def p (m : ℝ) : Prop := m ≤ 2
def q (m : ℝ) : Prop := 0 < m ∧ m < 1

theorem problem (hpq : ¬ (p m ∧ q m)) (hlpq : p m ∨ q m) : m ≤ 0 ∨ (1 ≤ m ∧ m ≤ 2) := 
sorry

end problem_l138_138228


namespace sum_union_eq_34_l138_138871

open Finset

def A : Finset ℕ := {2, 0, 1, 9}
def B : Finset ℕ := { x | ∃ a ∈ A, x = 2 * a }

noncomputable def union_sum : ℕ := (A ∪ B).sum id

theorem sum_union_eq_34 : union_sum = 34 := by
  sorry

end sum_union_eq_34_l138_138871


namespace main_results_l138_138255

-- Arithmetic sequence a_n
def a_n (n : ℕ) : ℕ := n

-- Geometric sequence b_n
def b_n (n : ℕ) : ℕ := 2 ^ n

-- Given conditions
theorem main_results :
  (∀ n, a_n n = n) ∧
  (∀ n, b_n n = 2 ^ n) ∧
  (∀ m, a_n 4 = 4 * (b_n 1) - (b_n 2)) ∧
  (∀ m, (a_n 2) + (a_n 3) = (a_n 5)) ∧
  (∀ m, (b_n 3) = (a_n 3) + (a_n 5)) ∧
  (∀ n, let c_n := (b_n (2 * n)) / ((b_n (2 * n) - 1) * (b_n (2 * n + 2) - 1)) in
    let S_n := finset.range n.sum (λ k, c_n k) in
    S_n = 1 / 3 * (1 / 3 - 1 / (2 ^ (2 * n + 2) - 1))) ∧
  (∀ n, finset.prod (finset.range n) (λ k, (a_n (2 * k - 1)) * (b_n k) / (a_n (2 * k + 1))) = (n + 1) * 2^(finset.sum (finset.range n) (λ k, k + 1))) ∧
  (∀ n, let d_1 := 1 in
    let d_n := λ n, if 2^k < n < 2^(k+1) then 1 else b_n k in
    finset.range (2 ^ n).sum (λ i, (a_n i) * (d_n i)) = 11 / 6 * 4^n - 3 / 2 * 2^n + 2 / 3) :=
sorry

end main_results_l138_138255


namespace infinitely_many_nonpositive_terms_l138_138242

noncomputable def sequence (a1 : ℝ) : ℕ → ℝ
| 0     := a1
| (n+1) := if sequence n = 0 then 0 else (1/2) * (sequence n - 1 / (sequence n))

theorem infinitely_many_nonpositive_terms (a1 : ℝ) :
  ∃ infinitely_many n, sequence a1 n ≤ 0 :=
sorry

end infinitely_many_nonpositive_terms_l138_138242


namespace machine_B_production_time_l138_138373

noncomputable def minutes_in_a_day : ℕ := 1440
noncomputable def production_time_machine_A : ℕ := 4
noncomputable def items_produced_by_A : ℕ := minutes_in_a_day / production_time_machine_A
noncomputable def percentage_increase : ℚ := 1.25
noncomputable def items_produced_by_B : ℕ := items_produced_by_A / percentage_increase

theorem machine_B_production_time : items_produced_by_B = 288 → ∃ (T_B : ℕ), T_B = 5 :=
by
  intro h
  have : T_B = minutes_in_a_day / items_produced_by_B := sorry
  existsi T_B
  exact h

end machine_B_production_time_l138_138373


namespace find_n_l138_138069

theorem find_n : ∃ n : ℤ, 0 ≤ n ∧ n < 107 ∧ 103 * n ≡ 56 [MOD 107] ∧ n ≡ 85 [MOD 107] :=
by {
  use 85,
  sorry
}

end find_n_l138_138069


namespace T_4_value_l138_138246

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ (n : ℕ), a (n + 1) = a n * q

def satisfies_conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  (a 1 * a 2 = 2 * a 0) ∧
  ((a 3 + 2 * a 6) / 2 = 5 / 4)

def S (n : ℕ) (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  a 0 * (1 - q^n) / (1 - q)

def T (n : ℕ) (a : ℕ → ℝ) (q : ℝ) : ℝ :=
  ∑ i in finset.range (n + 1), S i a q

theorem T_4_value (a : ℕ → ℝ) (q : ℝ) (h1 : geometric_sequence a)
  (h2 : satisfies_conditions a q) : T 4 a q = 196 :=
by
  sorry

end T_4_value_l138_138246


namespace range_of_a_l138_138836

variable (a : ℝ)

def p : Prop := ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x ^ 2 - a ≥ 0
def q : Prop := ∃ x : ℝ, x ^ 2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h₁ : p a) (h₂ : q a) : a ≤ -2 ∨ a = 1 := 
sorry

end range_of_a_l138_138836


namespace find_ab_cd_l138_138360

variables (a b c d : ℝ)

def special_eq (x : ℝ) := 
  (min (20 * x + 19) (19 * x + 20) = (a * x + b) - abs (c * x + d))

theorem find_ab_cd (h : ∀ x : ℝ, special_eq a b c d x) :
  a * b + c * d = 380 := 
sorry

end find_ab_cd_l138_138360


namespace proof_problem_l138_138297

variable {a b : ℝ}

-- Assuming the given set equality as a condition, we need to use it properly in Lean
axiom set_equality : ({1, a, b / a} : set ℝ) = {0, a^2, a + b}

-- The theorem to prove
theorem proof_problem (h : set_equality) : a^2015 + b^2016 = -1 :=
sorry

end proof_problem_l138_138297


namespace remaining_income_after_expenses_l138_138120

theorem remaining_income_after_expenses :
  (100 - (42 + 18 + 12)) * (1 - 0.3) - ((1 - 0.3) * 0.25) * (1 - 0.15) * (1 - 0.06) = 13.9825 := 
by 
  -- The intermediate calculations from the problem can be used to verify step by step
  calc (100 - (42 + 18 + 12)) * (1 - 0.3) = 28 * 0.7 : by {norm_num}
  ... = 19.6 
  sorry

end remaining_income_after_expenses_l138_138120


namespace marcella_shoes_lost_l138_138008

theorem marcella_shoes_lost (pairs_initial : ℕ) (pairs_left_max : ℕ) (individuals_initial : ℕ) (individuals_left_max : ℕ) (pairs_initial_eq : pairs_initial = 25) (pairs_left_max_eq : pairs_left_max = 20) (individuals_initial_eq : individuals_initial = pairs_initial * 2) (individuals_left_max_eq : individuals_left_max = pairs_left_max * 2) : (individuals_initial - individuals_left_max) = 10 := 
by
  sorry

end marcella_shoes_lost_l138_138008


namespace vector_n_and_length_l138_138876

def vector_m := (1, 1)
def vector_a := (1, 0)
def vector_b (x : ℝ) := (Real.cos x, Real.sin x)

theorem vector_n_and_length
  (n : ℝ × ℝ)
  (h1 : ∠ n = (3 * Real.pi / 4))
  (h2 : vector_m.1 * n.1 + vector_m.2 * n.2 = -1)
  (h3 : vector_a.1 * n.1 + vector_a.2 * n.2 = 0) :
  n = (0, -1) ∧ 0 ≤ Real.sqrt (2 * (1 - Real.sin n.1)) ∧ Real.sqrt (2 * (1 - Real.sin n.1)) ≤ 2 := 
sorry

end vector_n_and_length_l138_138876


namespace triangle_perimeter_l138_138042

noncomputable def triangle_perimeter_proof : Prop :=
  let r := 15
  let DS := 18
  let SE := 22
  let DE := DS + SE
  let DF := DS
  let EF := SE
  let y := 30.0 / 57.0
  let perimeter := DE + DF + EF + 2 * y
  perimeter = 90.474

theorem triangle_perimeter : triangle_perimeter_proof := 
  by
    -- Definitions
    let r := 15
    let DS := 18
    let SE := 22
    let DE := DS + SE
    let DF := DS
    let EF := SE
    let y := 30.0 / 57.0
    let perimeter := DE + DF + EF + 2 * y
    -- Proof of the perimeter
    show perimeter = 90.474
    sorry

end triangle_perimeter_l138_138042


namespace largest_olympic_not_exceeding_2015_l138_138759

def is_olympic (n : ℕ) : Prop :=
  ∃ f : ℝ → ℝ, ∃ a b c : ℤ, (f = λ x, a * x^2 + b * x + c) ∧ f (f (Real.sqrt n)) = 0

theorem largest_olympic_not_exceeding_2015 : ∃ n : ℕ, n ≤ 2015 ∧ is_olympic n ∧ ∀ m : ℕ, m ≤ 2015 ∧ is_olympic m → m ≤ n :=
sorry

end largest_olympic_not_exceeding_2015_l138_138759


namespace jamie_fathers_age_twice_when_l138_138010

theorem jamie_fathers_age_twice_when (x : ℕ) :
  ∃ x : ℕ, let jamie_age := 10 in let father_age := 5 * jamie_age in
  let future_jamie_age := jamie_age + x in let future_father_age := father_age + x in
  future_father_age = 2 * future_jamie_age ∧ 2010 + x = 2040 :=
by {
  sorry
}

end jamie_fathers_age_twice_when_l138_138010


namespace length_of_each_piece_l138_138283

theorem length_of_each_piece (rod_length : ℝ) (num_pieces : ℕ) (h₁ : rod_length = 42.5) (h₂ : num_pieces = 50) : (rod_length / num_pieces * 100) = 85 := 
by 
  sorry

end length_of_each_piece_l138_138283


namespace probability_factor_36_l138_138511

theorem probability_factor_36 : (∃ n : ℕ, n ≤ 36 ∧ ∃ k, 36 = k * n) → (9 / 36 = 1 / 4) :=
begin
  sorry
end

end probability_factor_36_l138_138511


namespace simplify_expression_l138_138813

theorem simplify_expression :
  (∃ (a b c d e f : ℝ), 
    a = (7)^(1/4) ∧ 
    b = (3)^(1/3) ∧ 
    c = (7)^(1/2) ∧ 
    d = (3)^(1/6) ∧ 
    e = (a / b) / (c / d) ∧ 
    f = ((1 / 7)^(1/4)) * ((1 / 3)^(1/6))
    → e = f) :=
by {
  sorry
}

end simplify_expression_l138_138813


namespace probability_factor_36_l138_138545

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138545


namespace solve_triangle_problem_l138_138336

noncomputable def triangle_problem (A B C a b c : ℝ) (h1 : b^2 = a * c)
(h2 : sqrt 3 * b * sin C = c * cos B + c) : Prop :=
B = π / 3 ∧ (1 / tan A) + (1 / tan C) = 2 * sqrt 3 / 3

theorem solve_triangle_problem (A B C a b c : ℝ) (h1 : b^2 = a * c)
(h2 : sqrt 3 * b * sin C = c * cos B + c) : triangle_problem A B C a b c h1 h2 :=
sorry

end solve_triangle_problem_l138_138336


namespace total_payment_l138_138936

/--
  Jerry took 8 hours painting the house. 
  The time to fix the kitchen counter was three times longer than painting the house.
  Jerry took 6 hours mowing the lawn.
  Jerry charged $15 per hour of work.
  Prove that the total amount of money Miss Stevie paid Jerry is $570.
-/
theorem total_payment (h_paint: ℕ := 8) (h_counter: ℕ := 3 * h_paint) (h_mow: ℕ := 6) (rate: ℕ := 15) :
  let total_hours := h_paint + h_counter + h_mow
  in total_hours * rate = 570 :=
by
  -- provide proof here
  sorry

end total_payment_l138_138936


namespace common_ratio_geometric_sequence_l138_138749

theorem common_ratio_geometric_sequence (a b c d : ℤ) (h1 : a = 10) (h2 : b = -20) (h3 : c = 40) (h4 : d = -80) :
    b / a = -2 ∧ c = b * -2 ∧ d = c * -2 := by
  sorry

end common_ratio_geometric_sequence_l138_138749


namespace problem1_correct_solution_problem2_correct_solution_l138_138867

noncomputable def g (x a : ℝ) : ℝ := |x| + 2 * |x + 2 - a|

/-- 
    Prove that the set {x | -2/3 ≤ x ≤ 2} satisfies g(x) ≤ 4 when a = 3 
--/
theorem problem1_correct_solution (x : ℝ) : g x 3 ≤ 4 ↔ -2/3 ≤ x ∧ x ≤ 2 :=
by
  sorry

noncomputable def f (x a : ℝ) : ℝ := g (x - 2) a

/-- 
    Prove that the range of a such that f(x) ≥ 1 for all x ∈ ℝ 
    is a ≤ 1 or a ≥ 3
--/
theorem problem2_correct_solution (a : ℝ) : 
  (∀ x : ℝ, f x a ≥ 1) ↔ a ≤ 1 ∨ a ≥ 3 :=
by
  sorry

end problem1_correct_solution_problem2_correct_solution_l138_138867


namespace geometric_series_remainder_500_l138_138178

theorem geometric_series_remainder_500 :
  ∑ i in Finset.range 1001, 3 ^ i % 500 = 1 :=
by
  -- Sum of geometric series formula
  sorry

end geometric_series_remainder_500_l138_138178


namespace probability_factor_of_36_l138_138586

def is_factor_of (n d : ℕ) : Prop := d % n = 0

theorem probability_factor_of_36 : 
  let total := 36
  let factors_of_36 := { n | n ≤ 36 ∧ is_factor_of n 36 }
  (factors_of_36.to_finset.card : ℚ) / total = 1 / 4 := 
by
  sorry

end probability_factor_of_36_l138_138586


namespace probability_factor_36_l138_138550

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138550


namespace probability_factor_36_l138_138598

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138598


namespace digit_in_421st_place_l138_138708

noncomputable def decimal_rep := "368421052631578947"

theorem digit_in_421st_place : 
  ∀ (n : ℕ), n = 421 → 
  (let m := 19, a := 7 in a / m = 0.repeating_decimal sequence_of (decimal_rep, 18) at position n) :=
by 
  sorry

end digit_in_421st_place_l138_138708


namespace two_m_plus_three_b_l138_138181

noncomputable def m : ℚ := (-(3/2) - (1/2)) / (2 - (-1))

noncomputable def b : ℚ := (1/2) - m * (-1)

theorem two_m_plus_three_b :
  2 * m + 3 * b = -11 / 6 :=
by
  sorry

end two_m_plus_three_b_l138_138181


namespace jacob_find_more_l138_138188

theorem jacob_find_more :
  let initial_shells := 2
  let ed_limpet_shells := 7
  let ed_oyster_shells := 2
  let ed_conch_shells := 4
  let total_shells := 30
  let ed_shells := ed_limpet_shells + ed_oyster_shells + ed_conch_shells + initial_shells
  let jacob_shells := total_shells - ed_shells
  (jacob_shells - ed_limpet_shells - ed_oyster_shells - ed_conch_shells = 2) := 
by 
  sorry

end jacob_find_more_l138_138188


namespace probability_factor_36_l138_138595

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138595


namespace canvas_decreased_by_40_percent_l138_138150

noncomputable def canvas_decrease (P C : ℝ) (x d : ℝ) : Prop :=
  (P = 4 * C) ∧
  ((P - 0.60 * P) + (C - (x / 100) * C) = (1 - d / 100) * (P + C)) ∧
  (d = 55.99999999999999)

theorem canvas_decreased_by_40_percent (P C : ℝ) (x d : ℝ) 
  (h : canvas_decrease P C x d) : x = 40 :=
by
  sorry

end canvas_decreased_by_40_percent_l138_138150


namespace probability_factor_of_36_l138_138689

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138689


namespace probability_factor_36_l138_138599

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138599


namespace convert_octal_127_to_binary_l138_138799

def octal_to_binary (n : ℕ) : ℕ :=
  match n with
  | 1 => 3  -- 001 in binary
  | 2 => 2  -- 010 in binary
  | 7 => 7  -- 111 in binary
  | _ => 0  -- No other digits are used in this example

theorem convert_octal_127_to_binary :
  octal_to_binary 1 * 1000000 + octal_to_binary 2 * 1000 + octal_to_binary 7 = 1010111 :=
by
  -- Proof would go here
  sorry

end convert_octal_127_to_binary_l138_138799


namespace area_of_T_shaped_region_l138_138099

theorem area_of_T_shaped_region :
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  (ABCD_area - (EFHG_area + EFGI_area + EFCD_area)) = 24 :=
by
  let ABCD_area : ℝ := 48
  let EFHG_area : ℝ := 4
  let EFGI_area : ℝ := 8
  let EFCD_area : ℝ := 12
  exact sorry

end area_of_T_shaped_region_l138_138099


namespace sum_of_reciprocal_squares_lt_two_l138_138096

theorem sum_of_reciprocal_squares_lt_two (n : ℕ) : 
  (∑ k in Finset.range (n + 1), (1 : ℝ) / (k + 1) ^ 2) < 2 :=
sorry

end sum_of_reciprocal_squares_lt_two_l138_138096


namespace modular_inverse_l138_138245

theorem modular_inverse :
  (24 * 22) % 53 = 1 :=
by
  have h1 : (24 * -29) % 53 = (53 * 0 - 29 * 24) % 53 := by sorry
  have h2 : (24 * -29) % 53 = (-29 * 24) % 53 := by sorry
  have h3 : (-29 * 24) % 53 = (-29 % 53 * 24 % 53 % 53) := by sorry
  have h4 : -29 % 53 = 53 - 24 := by sorry
  have h5 : (53 - 29) % 53 = (22 * 22) % 53 := by sorry
  have h6 : (22 * 22) % 53 = (24 * 22) % 53 := by sorry
  have h7 : (24 * 22) % 53 = 1 := by sorry
  exact h7

end modular_inverse_l138_138245


namespace levels_for_blocks_l138_138940

theorem levels_for_blocks (S : ℕ → ℕ) (n : ℕ) (h1 : S n = n * (n + 1)) (h2 : S 10 = 110) : n = 10 :=
by {
  sorry
}

end levels_for_blocks_l138_138940


namespace probability_factor_of_36_l138_138560

theorem probability_factor_of_36 :
  (∃ n ∈ finset.range 37, (∃ k : ℕ, 36 = k * n)) → 9 / 36 = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138560


namespace infinite_multiples_in_partition_l138_138428

theorem infinite_multiples_in_partition (S : Set (Set ℕ))
  (hS : ∀ n : ℕ, ∃ s ∈ S, ∃ M : ℕ, ∀ m ∈ finset.range M, (n * m) ∉ s ∨ ∃ k : ℕ, n * m = k * n) :
  ∃ s ∈ S, ∀ n : ℕ, set_infinite {m ∈ s | ∃ k : ℕ, m = k * n} :=
by
  sorry

end infinite_multiples_in_partition_l138_138428


namespace Q_one_div_Q_neg_one_eq_one_l138_138420

noncomputable def f (x : ℝ) : ℝ := x^2023 + 19 * x^2022 + 1

-- Assume the distinct roots r₁, ..., r₂₀₂₃ of f(x)
axiom roots : ℕ → ℝ
axiom roots_are_distinct : ∀ (i j : ℕ), i ≠ j → roots i ≠ roots j
axiom roots_zeroes : ∀ j, f (roots j) = 0

noncomputable def Q (z : ℝ) : ℝ :=
  ∏ j in (finset.range 2023), (z - (roots j - 1 / roots j))

-- Prove that the value of Q(1) / Q(-1) is 1
theorem Q_one_div_Q_neg_one_eq_one :
  (Q 1) / (Q (-1)) = 1 :=
sorry

end Q_one_div_Q_neg_one_eq_one_l138_138420


namespace batsman_average_46_innings_l138_138104

variable (A : ℕ) (highest_score : ℕ) (lowest_score : ℕ) (average_excl : ℕ)
variable (n_innings n_without_highest_lowest : ℕ)

theorem batsman_average_46_innings
  (h_diff: highest_score - lowest_score = 190)
  (h_avg_excl: average_excl = 58)
  (h_highest: highest_score = 199)
  (h_innings: n_innings = 46)
  (h_innings_excl: n_without_highest_lowest = 44) :
  A = (44 * 58 + 199 + 9) / 46 := by
  sorry

end batsman_average_46_innings_l138_138104


namespace triangle_inequality_AMB_l138_138335

variables {A B C D E M : Type}

noncomputable def median (A B : Type) : Type := sorry

-- Given data
axiom medians_intersect_at_M : ∀ (ABC : Type) (D E M : Type), 
  median ABC D → median ABC E → (\( M \)) = true

axiom angle_AMB_right_or_acute : ∀ (A B C M : Type),
  (∠AMB = 90 ∨ ∠AMB < 90) → true

-- Main proof statement
theorem triangle_inequality_AMB (A B C D E M : Type)
  (h1 : median A B D)
  (h2 : median A C E)
  (h3 : (∠AMB = 90 ∨ ∠AMB < 90)) :
  (A + B > 3 * A B) :=
begin
  sorry
end

end triangle_inequality_AMB_l138_138335


namespace factor_probability_36_l138_138616

-- Definitions based on conditions
def is_factor (n d : ℕ) : Prop := d ∣ n
def num_factors (n : ℕ) : ℕ := (Finset.range (n + 1)).filter (is_factor n).card

def probability_factor (n : ℕ) : ℚ := num_factors n / n

-- Problem statement
theorem factor_probability_36 : probability_factor 36 = 1 / 4 :=
  by sorry

end factor_probability_36_l138_138616


namespace root_approximation_l138_138064

noncomputable def f (x : ℝ) : ℝ := 3^x - x - 4

theorem root_approximation :
  ∃ ξ : ℝ, |ξ - 1.56| < 0.01 ∧ f(1.5625) > 0 ∧ f(1.5562) < 0 :=
by
  have f_1_6000 := 0.200 : by sorry
  have f_1_5875 := 0.133 : by sorry
  have f_1_5750 := 0.067 : by sorry
  have f_1_5625 := 0.003 : by sorry
  have f_1_5562 := -0.029 : by sorry
  have f_1_5500 := -0.060 : by sorry
  sorry

end root_approximation_l138_138064


namespace vector_magnitude_l138_138275

noncomputable def a : ℝ × ℝ := (Real.cos (5 * Real.pi / 180), Real.sin (5 * Real.pi / 180))
noncomputable def b : ℝ × ℝ := (Real.cos (65 * Real.pi / 180), Real.sin (65 * Real.pi / 180))

theorem vector_magnitude : |(a.1 + 2 * b.1, a.2 + 2 * b.2)| = Real.sqrt 7 :=
by sorry

end vector_magnitude_l138_138275


namespace sequence_an_correct_l138_138006

noncomputable def sequence_an (n : ℕ) : ℕ :=
  let S : ℕ → ℕ := λ n, 2^n - 1 -- Sum of the first n terms as given for the sequence
  in if n = 0 then 0 else S n - S (n - 1)

theorem sequence_an_correct (n : ℕ) (hn : n > 0) : sequence_an n = 2^(n-1) :=
by
  sorry

end sequence_an_correct_l138_138006


namespace disk_contains_origin_l138_138053

theorem disk_contains_origin
  {x1 x4 y1 y2 : ℝ}
  {x2 x3 y3 y4 : ℝ}
  (h1 : x1 > 0) (h2 : x4 > 0) (h3 : y1 > 0) (h4 : y2 > 0)
  (h5 : x2 < 0) (h6 : x3 < 0) (h7 : y3 < 0) (h8 : y4 < 0)
  (h9 : ∀ i ∈ {1, 2, 3, 4},
    let xi := if i = 1 then x1 else if i = 2 then x2 else if i = 3 then x3 else x4
    in let yi := if i = 1 then y1 else if i = 2 then y2 else if i = 3 then y3 else y4
    in (xi - a) ^ 2 + (yi - b) ^ 2 ≤ c ^ 2)
  : a ^ 2 + b ^ 2 ≤ c ^ 2 := 
sorry

end disk_contains_origin_l138_138053


namespace factor_probability_l138_138628

theorem factor_probability : 
  let S := { n : ℕ | n > 0 ∧ n ≤ 36 } in
  let factor_count := (multiplicity 2 36 + 1) * (multiplicity 3 36 + 1) in
  let total_count := 36 in
  (factor_count : ℚ) / total_count = (1 : ℚ) / 4 := 
by
  sorry

end factor_probability_l138_138628


namespace simplify_division_l138_138019

noncomputable def simplify_expression (m : ℝ) : ℝ :=
  (m^2 - 3 * m + 1) / m + 1

noncomputable def divisor_expression (m : ℝ) : ℝ :=
  (m^2 - 1) / m

theorem simplify_division (m : ℝ) (hm1 : m ≠ 0) (hm2 : m ≠ 1) (hm3 : m ≠ -1) :
  (simplify_expression m) / (divisor_expression m) = (m - 1) / (m + 1) :=
by {
  sorry
}

end simplify_division_l138_138019


namespace batsman_average_after_17th_inning_l138_138086

theorem batsman_average_after_17th_inning :
  ∃ x : ℤ, (63 + (16 * x) = 17 * (x + 3)) ∧ (x + 3 = 17) :=
by
  sorry

end batsman_average_after_17th_inning_l138_138086


namespace largest_possible_N_l138_138990

theorem largest_possible_N (N : ℕ) (divisors : List ℕ) 
  (h1 : divisors = divisors.filter (λ d, N % d = 0)) 
  (h2 : divisors.length > 2) 
  (h3 : List.nth divisors (divisors.length - 3) = some (21 * (List.nth! divisors 1))) :
  N = 441 :=
sorry

end largest_possible_N_l138_138990


namespace derivative_of_hyperbolic_fn_l138_138821

noncomputable def hyperbolic_ln_th_ch_sh (x : ℝ) : ℝ :=
  -((1 / 2) * Real.log (Real.tanh (x / 2))) - (Real.cosh x / (2 * (Real.sinh x) ^ 2))

theorem derivative_of_hyperbolic_fn (x : ℝ) : 
  deriv (λ x, hyperbolic_ln_th_ch_sh x) x = 1 / (Real.sinh x) ^ 3 :=
by
  sorry

end derivative_of_hyperbolic_fn_l138_138821


namespace polynomial_nonfactorable_l138_138965

open Polynomial

theorem polynomial_nonfactorable 
  (n : ℕ) 
  (a : Fin n → ℤ) 
  (h_distinct : Function.Injective a)
  (h_pos : 0 < n) :
  ¬ ∃ p q : Polynomial ℤ, p.degree ≥ 1 ∧ q.degree ≥ 1 ∧ 
  (∏ i in Finset.range n, (X - C (a ⟨i, Nat.lt_succ_iff.mpr (Finset.mem_range.mp (Finset.mem_univ _))⟩))) - 1 = p * q :=
sorry

end polynomial_nonfactorable_l138_138965


namespace probability_factor_of_36_l138_138646

theorem probability_factor_of_36 : 
  (finset.card (finset.filter (λ x : ℕ, 36 % x = 0) (finset.range (36 + 1))) : ℚ) / 36 = 1 / 4 := 
by 
sorry

end probability_factor_of_36_l138_138646


namespace cuboid_volume_l138_138231

theorem cuboid_volume (x y z : ℝ)
  (h1 : 2 * (x + y) = 20)
  (h2 : 2 * (y + z) = 32)
  (h3 : 2 * (x + z) = 28) : x * y * z = 240 := 
by
  sorry

end cuboid_volume_l138_138231


namespace smallest_five_digit_congruent_two_mod_seventeen_l138_138074

theorem smallest_five_digit_congruent_two_mod_seventeen : 
  ∃ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ n % 17 = 2 ∧ n = 10013 :=
by
  sorry

end smallest_five_digit_congruent_two_mod_seventeen_l138_138074


namespace absolute_value_inequality_l138_138402

theorem absolute_value_inequality (x : ℝ) : ¬ (|x - 3| + |x + 4| < 6) :=
sorry

end absolute_value_inequality_l138_138402


namespace find_three_digit_number_l138_138200

theorem find_three_digit_number :
  ∃ (Π В Γ : ℕ), (Π ≠ В ∧ В ≠ Γ ∧ Π ≠ Γ) 
               ∧ (0 ≤ Π ∧ Π ≤ 9) 
               ∧ (0 ≤ В ∧ В ≤ 9) 
               ∧ (0 ≤ Γ ∧ Γ ≤ 9) 
               ∧ (100 * Π + 10 * В + Γ = (Π + В + Γ) * ((Π + В + Γ) + 1))
               ∧ (100 * Π + 10 * В + Γ = 156) :=
by {
  sorry
}

end find_three_digit_number_l138_138200


namespace solve_quadratic_complex_find_imaginary_z_l138_138731

-- Problem 1: solving the quadratic equation in the complex field
theorem solve_quadratic_complex (x : ℂ) : x^2 - 6*x + 13 = 0 ↔ x = 3 + 2*complex.I ∨ x = 3 - 2*complex.I :=
sorry

-- Problem 2: proving that the complex number z is purely imaginary
theorem find_imaginary_z (a : ℝ) (z : ℂ) : z = (1 + complex.I) * (a + 2 * complex.I) → z.im ≠ 0 → z = 4 * complex.I :=
sorry

end solve_quadratic_complex_find_imaginary_z_l138_138731


namespace calc_value_l138_138956

def f (x : ℤ) : ℤ := x^2 + 5 * x + 4
def g (x : ℤ) : ℤ := 2 * x - 3

theorem calc_value :
  f (g (-3)) - 2 * g (f 2) = -26 := by
  sorry

end calc_value_l138_138956


namespace right_triangle_area_perimeter_l138_138902

theorem right_triangle_area_perimeter (a b : ℕ) (h₁ : a = 36) (h₂ : b = 48) : 
  (1/2) * (a * b) = 864 ∧ a + b + Nat.sqrt (a * a + b * b) = 144 := by
  sorry

end right_triangle_area_perimeter_l138_138902


namespace find_m_values_l138_138872

noncomputable def A : Set ℝ := {x | x^2 - 3 * x + 2 = 0}
noncomputable def B (m : ℝ) : Set ℝ := {x | x^2 - m * x + 2 = 0}
def range_m : Set ℝ := {m | (B m) ⊆ A}

theorem find_m_values : range_m = {m : ℝ | (-sqrt 8 < m ∧ m < sqrt 8) ∨ m = 3} :=
by
  sorry

end find_m_values_l138_138872


namespace sum_F_sqrt_inverses_l138_138175

def closest_integer (x : ℝ) : ℤ := Int.round x

def F (x : ℝ) : ℤ := closest_integer x

theorem sum_F_sqrt_inverses :
  (∑ i in finset.range 100, (1 / F (real.sqrt i.succ))) = 19 :=
sorry

end sum_F_sqrt_inverses_l138_138175


namespace probability_factor_36_l138_138548

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138548


namespace min_basket_count_l138_138223

theorem min_basket_count (students : ℕ) (total_baskets : ℕ) (h1 : students = 4) (h2 : total_baskets = 30) : 
  ∃ k, k ≥ 8 ∧ k ∈ set_of (λ n, ∃ m : ℕ, m ≤ students ∧ total_baskets = n * students + m) := by
  sorry

end min_basket_count_l138_138223


namespace triangles_perimeter_impossible_l138_138450

theorem triangles_perimeter_impossible :
  ∀ x, x = 58 ∨ x = 64 ∨ x = 70 ∨ x = 76 ∨ x = 82 →
  25 + 20 > 25 ∧
  25 + x > 20 ∧
  20 + x > 25 ∧
  ¬(25 + 20 + x = 82) :=
by
  intro x H
  have h1 : 25 + 20 > x → x < 45 := sorry
  have h2 : 25 + x > 20 → x > 5 := sorry
  have h3 : 20 + x > 25 → x > 5 := sorry
  have h4 : (5 < x ∧ x < 45) → (50 < 25 + 20 + x ∧ 25 + 20 + x < 90) := sorry
  have h5 : 25 + 20 + x ≠ 82 → true := sorry
  exact h1 H.1 ∧ h2 H.2 ∧ h3 H.3 ∧ h4 (and.intro h2 h3) ∧ h5 sorry

end triangles_perimeter_impossible_l138_138450


namespace village_male_population_l138_138130

theorem village_male_population (total_population parts male_parts : ℕ) (h1 : total_population = 600) (h2 : parts = 4) (h3 : male_parts = 2) :
  male_parts * (total_population / parts) = 300 :=
by
  -- We are stating the problem as per the given conditions
  sorry

end village_male_population_l138_138130


namespace hexagon_perimeter_l138_138049

theorem hexagon_perimeter (side_length : ℕ) (num_sides : ℕ) (perimeter : ℕ) 
  (h1 : num_sides = 6)
  (h2 : side_length = 7)
  (h3 : perimeter = side_length * num_sides) : perimeter = 42 := by
  sorry

end hexagon_perimeter_l138_138049


namespace altitude_is_6_units_l138_138317

-- Defining the points and lengths
structure Point :=
(x : ℝ)
(y : ℝ)

def AB (A B : Point) : ℝ := 10
def AC (A C : Point) : ℝ := 10
def BC (B C : Point) : ℝ := 16

noncomputable def midpoint (B C : Point) : Point :=
{x := (B.x + C.x) / 2, y := (B.y + C.y) / 2}

-- The length of altitude AD from A to D
noncomputable def altitude_length (A D : Point) (h : D = midpoint B C) : ℝ := 
sqrt (AB A B ^ 2 - (BC B C / 2) ^ 2)

theorem altitude_is_6_units (A B C D : Point) (hAB_AC : AB A B = AC A C) 
(hAB_10 : AB A B = 10) (hAC_10 : AC A C = 10) (hBC_16 : BC B C = 16) 
(hD_mid : D = midpoint B C) : altitude_length A D hD_mid = 6 := 
sorry

end altitude_is_6_units_l138_138317


namespace arithmetic_sequence_sum_l138_138920

-- Define the arithmetic sequence and the given conditions
def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Define the specific values for the sequence a_1 = 2 and a_2 + a_3 = 13
variables {a : ℕ → ℤ} (d : ℤ)
axiom h1 : a 1 = 2
axiom h2 : a 2 + a 3 = 13

-- Conclude the value of a_4 + a_5 + a_6
theorem arithmetic_sequence_sum : a 4 + a 5 + a 6 = 42 :=
sorry

end arithmetic_sequence_sum_l138_138920


namespace probability_factor_of_36_l138_138675

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138675


namespace probability_divisor_of_36_is_one_fourth_l138_138476

noncomputable def probability_divisor_of_36 : ℚ :=
  let total_divisors := 9 in
  let total_integers := 36 in
  total_divisors / total_integers

theorem probability_divisor_of_36_is_one_fourth :
  probability_divisor_of_36 = 1 / 4 :=
by
  sorry

end probability_divisor_of_36_is_one_fourth_l138_138476


namespace seating_arrangements_l138_138437

theorem seating_arrangements (n : ℕ) (people : ℕ) (middle_seat : ℕ) (total_arrangements : ℕ) :
  n = 11 → people = 2 → middle_seat = 6 → total_arrangements = 84 →
  (∃ (arrangements : ℕ), arrangements = total_arrangements ∧
    ∀ (p1 p2 : ℕ), p1 ≠ middle_seat ∧ p2 ≠ middle_seat ∧ 
    p1 ≠ p2 ∧ abs (p1 - p2) > 1) :=
by
  intros hn hpeople hmiddle htotal
  use total_arrangements
  split
  { exact htotal },
  { sorry }

end seating_arrangements_l138_138437


namespace probability_factor_36_l138_138549

theorem probability_factor_36 : 
  (∃ n : ℕ, 1 ≤ n ∧ n ≤ 36 ∧ (∀ k : ℕ, k ∣ n ↔ k ∣ 36) → (n ≠ 0 → (∃ p : ℚ, p = 1 / 4))) :=
begin
  sorry
end

end probability_factor_36_l138_138549


namespace probability_factor_36_l138_138593

def is_factor (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

theorem probability_factor_36 : 
  (set.univ.filter (λ n : ℕ, n > 0 ∧ n ≤ 36 ∧ is_factor 36 n)).card.toRat / 36 = 1 / 4 := 
by
  sorry

end probability_factor_36_l138_138593


namespace remainder_of_1998_to_10_mod_10k_l138_138194

theorem remainder_of_1998_to_10_mod_10k : 
  let x := 1998
  let y := 10^4
  x^10 % y = 1024 := 
by
  let x := 1998
  let y := 10^4
  sorry

end remainder_of_1998_to_10_mod_10k_l138_138194


namespace probability_factor_of_36_l138_138684

theorem probability_factor_of_36 :
  let N := 36
  let S := {n : ℕ | n > 0 ∧ n ≤ N}
  let E := {n : ℕ | n ∈ S ∧ N % n = 0}
  (E.card : ℚ) / S.card = 1 / 4 :=
by
  sorry

end probability_factor_of_36_l138_138684


namespace equal_real_roots_of_quadratic_l138_138295

theorem equal_real_roots_of_quadratic (a : ℝ) :
  (∃ x : ℝ, x^2 - a * x + 1 = 0 ∧ (λ z, z = x) = (λ z, z = x)) →
  (a = 2 ∨ a = -2) :=
by
  sorry

end equal_real_roots_of_quadratic_l138_138295


namespace probability_factor_of_36_l138_138692

theorem probability_factor_of_36 : 
  let pos_ints := { n : ℕ | 1 ≤ n ∧ n ≤ 36 }
  let num_divisors := { d : ℕ | d ∣ 36 }
  (num_divisors.to_finset.card : ℚ) / (pos_ints.to_finset.card : ℚ) = 1 / 4 :=
by {
  sorry
}

end probability_factor_of_36_l138_138692


namespace vehicles_travelled_last_year_l138_138345

theorem vehicles_travelled_last_year (V : ℕ) : 
  (∀ (x : ℕ), (96 : ℕ) * (V / 100000000) = 2880) → V = 3000000000 := 
by 
  sorry

end vehicles_travelled_last_year_l138_138345


namespace probability_factor_of_36_l138_138479

theorem probability_factor_of_36 : (∃ p : ℚ, p = 1 / 4 ∧
  let n := 36 in
  let factors := {d | d ∣ n ∧ d > 0 ∧ d ≤ n} in
  let total_pos_ints := 36 in
  let num_factors := set.card factors in
  p = num_factors / total_pos_ints) :=
begin
  sorry
end

end probability_factor_of_36_l138_138479


namespace product_less_than_50_l138_138927

def product_expr (n : ℕ) : ℝ :=
∏ k in finset.range n, (1 + 1 / (2 * (k + 1) : ℝ))

theorem product_less_than_50 : product_expr 1009 < 50 := 
sorry

end product_less_than_50_l138_138927


namespace arith_seq_a1_eq_15_l138_138257

variable {a : ℕ → ℤ} (a_seq : ∀ n, a n = a 1 + (n-1) * d)
variable {a_4 : ℤ} (h4 : a 4 = 9)
variable {a_8 : ℤ} (h8 : a 8 = -a 9)

theorem arith_seq_a1_eq_15 (a_seq : ∀ n, a n = a 1 + (n-1) * d) (h4 : a 4 = 9) (h8 : a 8 = -a 9) : a 1 = 15 :=
by
  -- Proof should go here
  sorry

end arith_seq_a1_eq_15_l138_138257


namespace compare_fx_values_l138_138005

noncomputable def f (a b x : ℝ) := real.log (abs (x + b)) / real.log a

theorem compare_fx_values (a b : ℝ) (h1 : ∀ x : ℝ, f a b x = f a b (-x)) 
                         (h2 : ∀ x y : ℝ, 0 < x ∧ x < y → f a b x < f a b y) :
  f a b (b - 2) < f a b (a + 1) := sorry

end compare_fx_values_l138_138005


namespace pies_and_leftover_apples_l138_138017

theorem pies_and_leftover_apples 
  (apples : ℕ) 
  (h : apples = 55) 
  (h1 : 15/3 = 5) :
  (apples / 5 = 11) ∧ (apples - 11 * 5 = 0) :=
by
  sorry

end pies_and_leftover_apples_l138_138017


namespace rectangular_solid_volume_l138_138761

theorem rectangular_solid_volume 
  (a b c : ℝ) 
  (h1 : a * b = 18) 
  (h2 : b * c = 50) 
  (h3 : a * c = 45) : 
  a * b * c = 150 * Real.sqrt 3 := 
by 
  sorry

end rectangular_solid_volume_l138_138761


namespace cat_finishes_food_on_saturday_l138_138388

noncomputable def daily_morning_consumption : ℚ := 1/4
noncomputable def daily_evening_consumption : ℚ := 1/6
noncomputable def daily_total_consumption : ℚ := daily_morning_consumption + daily_evening_consumption

noncomputable def initial_total_cans : ℚ := 6

noncomputable def days_to_finish (daily_consumption total_cans : ℚ) : ℕ :=
  nat_ceil (total_cans / daily_consumption)

theorem cat_finishes_food_on_saturday :
  days_to_finish daily_total_consumption initial_total_cans = 6 :=
by
  sorry

end cat_finishes_food_on_saturday_l138_138388


namespace area_sum_l138_138961

variables {A B C O H : Point}

-- Conditions: O is the circumcenter and H is the orthocenter of acute triangle ABC 
-- The Acute_angled_triangle O is the circumcenter and H is the orthocenter
def acute_triangle (A B C : Point) : Prop := isAcuteTriangle A B C

def is_circumcenter (O A B C : Point) : Prop := circumcenter O A B C

def is_orthocenter (H A B C : Point) : Prop := orthocenter H A B C

-- Main theorem
theorem area_sum (A B C O H : Point) 
  (hAcute : acute_triangle A B C)
  (hCircum : is_circumcenter O A B C)
  (hOrtho : is_orthocenter H A B C)
  : area_triangle A O H = area_triangle B O H + area_triangle C O H :=
by
  sorry

end area_sum_l138_138961


namespace valid_distributions_count_l138_138768

-- Definitions extracted from the conditions.
def triangular_array (n : ℕ) : ℕ × ℕ := (n, n * (n + 1) / 2)

def top_square_value (x : Fin 12 → ℕ) : ℕ :=
  ∑ k in Finset.range 12, Nat.choose 11 k * x k

def is_multiple_of_5 (n : ℕ) : Prop := n % 5 = 0

-- The theorem to prove
theorem valid_distributions_count :
  ∃ (f : Finset (Fin 12 → Fin 2)), 
    (∀ (x ∈ f), is_multiple_of_5 (top_square_value (λ i, x i))) ∧
    f.card = 1280 :=
sorry

end valid_distributions_count_l138_138768


namespace initial_sum_is_correct_l138_138820

def compoundInterest (P : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  P * (1 + r / n) ^ (n * t)

def find_initial_sum_of_money (CI : ℝ) (r : ℝ) (n : ℕ) (t : ℝ) : ℝ :=
  CI / ((1 + r / n) ^ (n * t) - 1)

theorem initial_sum_is_correct :
  let CI := 1289.0625
  let r := 0.25
  let n := 4
  let t := 0.5
  find_initial_sum_of_money CI r n t = 10000 :=
by
  sorry

end initial_sum_is_correct_l138_138820


namespace probability_factor_of_36_is_1_over_4_l138_138499

def totalPositiveIntegers := 36

def isDivisor (n: Nat) : Prop := ∃ (a b: Nat), (0 ≤ a ∧ a ≤ 2) ∧ (0 ≤ b ∧ b ≤ 2) ∧ (n = 2^a * 3^b)

def numDivisors := 9
def probability := (numDivisors : ℚ) / (totalPositiveIntegers : ℚ)

theorem probability_factor_of_36_is_1_over_4 :
  probability = (1 / 4 : ℚ) :=
sorry

end probability_factor_of_36_is_1_over_4_l138_138499


namespace geom_seq_gen_term_l138_138734

-- Defining the sequence according to the given recurrence relation
def a : ℕ → ℚ
| 0     := 0 -- assuming the sequence starts from index 1, curiously defining for n=0
| (n+1) := 1 / 2 * a n + 1 / 3

-- Question (1): Prove that the sequence {a_n - 2/3} is a geometric sequence with ratio 1/2
theorem geom_seq {n : ℕ} (hn : a n ≠ 2 / 3) : a (n + 1) - 2 / 3 = 1 / 2 * (a n - 2 / 3) := by 
  sorry

-- Question (2): Prove the general term when a_1 = 7 / 6
theorem gen_term (h : a 1 = 7 / 6) : ∀ n, a n = 2 / 3 + (1 / 2) ^ n := by 
  sorry

end geom_seq_gen_term_l138_138734


namespace average_weight_increase_l138_138031

theorem average_weight_increase (A : ℝ) : 
  let total_weight_before : ℝ := 8 * A in
  let weight_increase : ℝ := 55 - 35 in
  let total_weight_after : ℝ := total_weight_before + weight_increase in
  let new_average : ℝ := total_weight_after / 8 in
  new_average - A = 2.5 :=
by
  let total_weight_before := 8 * A
  let weight_increase := 55 - 35
  let total_weight_after := total_weight_before + weight_increase
  let new_average := total_weight_after / 8
  sorry

end average_weight_increase_l138_138031


namespace correct_relationship_l138_138712

theorem correct_relationship :
  ∃ (A B C D : Prop), 
    (A ↔ ¬ (\sin 11 - \sin 168 > 0)) ∧
    (B ↔ ¬ (\sin 194 < \cos 160)) ∧  
    (C ↔ ¬ (\tan (-π / 5) < \tan (-3 * π / 7))) ∧ 
    (D ↔ (\cos (-15 * π / 8) > \cos (14 * π / 9))) ∧ 
    (D = True) :=
by
  sorry

end correct_relationship_l138_138712


namespace probability_divisor_of_36_l138_138668

theorem probability_divisor_of_36 :
  (∃ (n : ℕ) (h₁ : 1 ≤ n ∧ n ≤ 36), (36 % n = 0) → (1 / 4 : ℝ)) :=
by 
  sorry

end probability_divisor_of_36_l138_138668


namespace angle_FAG_is_60_l138_138849

-- Define points and triangles
structure Point :=
(x : ℝ) (y : ℝ)

structure Triangle :=
(A B C : Point)

def equilateral_triangle (T : Triangle) : Prop :=
  dist T.A T.B = dist T.B T.C ∧ dist T.B T.C = dist T.C T.A

structure Square :=
(B C F G : Point)

def is_square (S : Square) : Prop :=
  dist S.B S.C = dist S.C S.F ∧ dist S.C S.F = dist S.F S.G ∧ dist S.F S.G = dist S.G S.B ∧
  angle S.B S.C S.F = 90 ∧ angle S.C S.F S.G = 90 ∧ angle S.F S.G S.B = 90 ∧ angle S.G S.B S.C = 90

def extension_points (A B C F G : Point) (T : Triangle) (S : Square) : Prop :=
  S.B = T.B ∧ S.C = T.C ∧ 
  dist T.B F = dist T.A T.B ∧
  dist T.C G = dist T.A T.C ∧
  -- Assure F and G are on the extensions of AB and AC respectively
  collinear [T.A, T.B, F] ∧ 
  collinear [T.A, T.C, G]

noncomputable def measure_angle_FAG (A B C F G : Point) : ℝ :=
  -- Function to compute the measure of angle FAG
  sorry

theorem angle_FAG_is_60 (A B C F G : Point) (T : Triangle) (S : Square) 
  (h1 : equilateral_triangle T) (h2 : is_square S) 
  (h3 : extension_points A B C F G T S) : 
  measure_angle_FAG A B C F G = 60 :=
  by sorry

end angle_FAG_is_60_l138_138849


namespace reflect_origin_l138_138323

theorem reflect_origin (x y : ℝ) (h₁ : x = 4) (h₂ : y = -3) : 
  (-x, -y) = (-4, 3) :=
by {
  sorry
}

end reflect_origin_l138_138323


namespace calculate_x_inv_cubed_l138_138886

noncomputable def log5 := Math.log 5
noncomputable def log4 (y: ℝ) := (Math.log y) / (Math.log 4)
noncomputable def log2 (z: ℝ) := (Math.log z) / (Math.log 2)

theorem calculate_x_inv_cubed (x : ℝ) 
  (h : log5 (log4 (log2 x)) = 1) : 
  x^(-1/3) = 2^(-1024/3) := 
by 
  sorry

end calculate_x_inv_cubed_l138_138886


namespace fraction_equals_half_l138_138793

def numerator : ℤ := 1 - 2 + 4 - 8 + 16 - 32 + 64
def denominator : ℤ := 2 - 4 + 8 - 16 + 32 - 64 + 128

theorem fraction_equals_half : (numerator : ℚ) / (denominator : ℚ) = 1 / 2 :=
by
  sorry

end fraction_equals_half_l138_138793


namespace part_a_tournament_n_6_part_b_tournament_n_8_l138_138730

def add_mod_n (x y n : ℕ) : ℕ :=
if x + y > n then (x + y) - n else x + y

def tournament_round (teams : List ℕ) (round : ℕ) (n : ℕ) : List (ℕ × ℕ) :=
teams.map (λ x => if x = 0 then (x, x) else ((add_mod_n x round (n - 1)), (add_mod_n x round (n - 1))))

def tournament_schedule (teams : List ℕ) (rounds : ℕ) (n : ℕ) : List (List (ℕ × ℕ)) :=
(List.range rounds).map (λ r => tournament_round teams (r + 1) n)

def initial_round_6 := [(1, 0), (2, 5), (3, 4)]
def initial_round_8 := [(1, 0), (2, 7), (3, 6), (4, 5)]

theorem part_a_tournament_n_6 :
  tournament_schedule initial_round_6 5 6 = [
    [(1, 0), (2, 5), (3, 4)],
    [(2, 0), (3, 1), (4, 5)],
    [(3, 0), (4, 2), (5, 1)],
    [(4, 0), (5, 3), (1, 2)],
    [(5, 0), (1, 4), (2, 3)]
  ] :=
sorry

theorem part_b_tournament_n_8 :
  tournament_schedule initial_round_8 7 8 = [
    [(1, 0), (2, 7), (3, 6), (4, 5)],
    [(2, 0), (3, 1), (4, 7), (5, 6)],
    [(3, 0), (4, 2), (5, 1), (6, 7)],
    [(4, 0), (5, 3), (6, 2), (7, 1)],
    [(5, 0), (6, 4), (7, 3), (1, 2)],
    [(6, 0), (7, 5), (1, 4), (2, 3)],
    [(7, 0), (1, 6), (2, 5), (3, 4)]
  ] :=
sorry

end part_a_tournament_n_6_part_b_tournament_n_8_l138_138730
