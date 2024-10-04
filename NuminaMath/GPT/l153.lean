import Mathlib

namespace ax5_plus_by5_l153_153811

-- Declare real numbers a, b, x, y
variables (a b x y : ℝ)

theorem ax5_plus_by5 (h1 : a * x + b * y = 3)
                     (h2 : a * x^2 + b * y^2 = 7)
                     (h3 : a * x^3 + b * y^3 = 6)
                     (h4 : a * x^4 + b * y^4 = 42) :
                     a * x^5 + b * y^5 = 20 := 
sorry

end ax5_plus_by5_l153_153811


namespace probability_of_letter_in_mathematics_l153_153442

open Classical

/-- The probability that a letter randomly chosen from the English alphabet is in the word "MATHEMATICS". -/
theorem probability_of_letter_in_mathematics : 
  let english_alphabet := {c : Char | 'a' ≤ c ∧ c ≤ 'z' ∨ 'A' ≤ c ∧ c ≤ 'Z'},
      mathematics_letters := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'} in
  (mathematics_letters.card.toRat / english_alphabet.card.toRat) = (4 / 13) := 
by
  let english_alphabet_size := 26
  let mathematics_unique_letters := {'M', 'A', 'T', 'H', 'E', 'I', 'C', 'S'}
  have h1 : mathematics_unique_letters.card = 8 := rfl
  rw [Rat.div_eq_div, Rat.eq_div_iff_mul_eq, Nat.cast_mul, Nat.cast_mul]
  simp [english_alphabet_size, h1]
  norm_num
  sorry

end probability_of_letter_in_mathematics_l153_153442


namespace residue_mod_17_l153_153713

theorem residue_mod_17 : (230 * 15 - 20 * 9 + 5) % 17 = 0 :=
  by
  sorry

end residue_mod_17_l153_153713


namespace chi_squared_confidence_l153_153268

theorem chi_squared_confidence (K_squared : ℝ) :
  (99.5 / 100 : ℝ) = 0.995 → (K_squared ≥ 7.879) :=
sorry

end chi_squared_confidence_l153_153268


namespace convert_2000mm_to_inches_l153_153604

def mmToInches (x : ℕ) : ℝ := x / 25.4

noncomputable def round (x : ℝ) (precision : ℝ) : ℝ :=
  Real.floor (x / precision + 0.5) * precision

theorem convert_2000mm_to_inches :
  round (mmToInches 2000) 0.01 = 78.74 := by
  sorry

end convert_2000mm_to_inches_l153_153604


namespace exists_integer_quotient_2012_l153_153752

def is_perfect_square_divisor (n d : ℕ) : Prop :=
  ∃ x : ℕ, d = x * x ∧ d ∣ n

def is_perfect_cube_divisor (n d : ℕ) : Prop :=
  ∃ x : ℕ, d = x * x * x ∧ d ∣ n

def f (n : ℕ) : ℕ :=
  ∃ dm : List ℕ, (∀ d ∈ dm, is_perfect_square_divisor n d) ∧ 
  dm.length = (nat.factorization n).to_finset.sum (λ p, nat.floor (nat.factorization n p / 2) + 1)

def g (n : ℕ) : ℕ :=
  ∃ dm : List ℕ, (∀ d ∈ dm, is_perfect_cube_divisor n d) ∧ 
  dm.length = (nat.factorization n).to_finset.sum (λ p, nat.floor (nat.factorization n p / 3) + 1)

theorem exists_integer_quotient_2012 : ∃ (n : ℕ), f(n) / g(n) = 2012 :=
sorry

end exists_integer_quotient_2012_l153_153752


namespace find_m_l153_153931

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end find_m_l153_153931


namespace first_term_of_geometric_series_l153_153692

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l153_153692


namespace expected_value_and_variance_of_2xi_plus_1_l153_153787

variables (ξ : ℝ) -- Define the random variable \xi

-- Conditions given in the problem
def normal_distribution : Prop := (E ξ = 3) ∧ (D ξ = 4)

-- Statement to prove
theorem expected_value_and_variance_of_2xi_plus_1 (h : normal_distribution ξ) :
  E(2 * ξ + 1) = 7 ∧ D(2 * ξ + 1) = 16 :=
sorry

end expected_value_and_variance_of_2xi_plus_1_l153_153787


namespace infinite_n_such_that_p_eq_nr_l153_153191

theorem infinite_n_such_that_p_eq_nr :
  ∃ (n : ℕ), ∃ (p r : ℕ), (a b c : ℕ) 
  (p = (a + b + c) / 2) (r = A / p) (A = p * r) (A = √(p * (p - a) * (p - b) * (p - c))),
  ∀ (p r : ℕ), ∃ (n : ℕ), p = n * r := sorry

end infinite_n_such_that_p_eq_nr_l153_153191


namespace largest_four_digit_number_mod_l153_153237

theorem largest_four_digit_number_mod (n : ℕ) : 
  (n < 10000) → 
  (n % 11 = 2) → 
  (n % 7 = 4) → 
  n ≤ 9973 :=
by
  sorry

end largest_four_digit_number_mod_l153_153237


namespace max_difference_exists_l153_153659

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l153_153659


namespace remainder_when_7n_divided_by_4_l153_153263

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l153_153263


namespace proof_hyperbola_eccentricity_l153_153484

-- Definitions based on the given conditions

variable (a b : ℝ) -- Parameters of the hyperbola
variable (c : ℝ) -- Focus of the hyperbola, located at (c, 0)
variable (m n : ℝ) -- Coordinates of the point P on the hyperbola

-- Assume the hyperbola definition and conditions
def hyperbola (x y : ℝ) := (x^2 / a^2) - (y^2 / b^2) = 1

-- Focus F and point P such that the midpoint condition holds
def focus := (c, 0)
def point_P := (m, n)
def midpoint_PF := (0, b)

-- Statement to prove that eccentricity of the hyperbola is sqrt(5)
theorem proof_hyperbola_eccentricity :
  c^2 / a^2 = 5 → sqrt (c^2 / a^2) = ∥√5∥ :=
by
  -- Variables required for proof
  assume h : c^2 / a^2 = 5
  
  -- Proof placeholder
  sorry

end proof_hyperbola_eccentricity_l153_153484


namespace solve_for_y_l153_153902

theorem solve_for_y :
  ∃ y : ℝ, (1 / 8)^(3 * y + 6) = 32^(y + 4) ↔ y = -19 / 7 :=
by
  sorry

end solve_for_y_l153_153902


namespace pascal_triangle_47_rows_l153_153087

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153087


namespace total_chocolate_bars_l153_153634

theorem total_chocolate_bars (small_boxes : ℕ) (bars_per_box : ℕ) 
  (h1 : small_boxes = 17) (h2 : bars_per_box = 26) 
  : small_boxes * bars_per_box = 442 :=
by sorry

end total_chocolate_bars_l153_153634


namespace geometric_series_product_l153_153724

theorem geometric_series_product 
: (∑' n : ℕ, (1:ℝ) * ((1:ℝ)/3)^n) * (∑' n : ℕ, (1:ℝ) * (-(1:ℝ)/3)^n)
 = ∑' n : ℕ, (1:ℝ) / 9^n := 
sorry

end geometric_series_product_l153_153724


namespace tile_side_length_l153_153172

theorem tile_side_length (height length : ℝ) (num_tiles : ℕ) (area_per_tile_sqft : ℝ) (area_per_tile_sqinch : ℝ) (side_length_inches : ℝ)
  (h1 : height = 10)
  (h2 : length = 15)
  (h3 : num_tiles = 21600)
  (h4 : area_per_tile_sqft = (height * length) / num_tiles)
  (h5 : area_per_tile_sqinch = area_per_tile_sqft * 144)
  (h6 : side_length_inches = real.sqrt area_per_tile_sqinch)
  : side_length_inches = 1 :=
by
  sorry

end tile_side_length_l153_153172


namespace woman_worked_days_l153_153656

theorem woman_worked_days :
  ∃ (W I : ℕ), (W + I = 25) ∧ (20 * W - 5 * I = 450) ∧ W = 23 := by
  sorry

end woman_worked_days_l153_153656


namespace distance_D_AB_proof_radius_circumscribed_circle_ADC_proof_l153_153129

-- Definition of conditions in terms of Lean constructs
variable (ABC : IsoscelesTriangle)
variable (base_AC : Real) (angle_ABC : Real)
variable (D : Point) (BC : Real)
variable (area_ADC_to_area_ABC : Ratio)

-- Assigning values to variables
def is_isosceles : Prop := Isosceles ABC
def base_AC_eq_one : Prop := base_AC = 1
def angle_ABC_def : Prop := angle_ABC = 2 * arctan(1 / 2)
def lies_on_BC : Prop := OnLine D BC
def area_relation : Prop := area_ABC = 4 * area_ADC

-- Questions related to distance from D to AB and radius of circumscribed circle around ADC
theorem distance_D_AB_proof : 
  is_isosceles ∧ base_AC_eq_one ∧ angle_ABC_def ∧ lies_on_BC ∧ area_relation → 
  distance_from_D_to_AB D AB = 3 / (2 * sqrt(5)) := sorry

theorem radius_circumscribed_circle_ADC_proof : 
  is_isosceles ∧ base_AC_eq_one ∧ angle_ABC_def ∧ lies_on_BC ∧ area_relation → 
  circumscribed_circle_radius ADC = sqrt(265) / 32 := sorry

end distance_D_AB_proof_radius_circumscribed_circle_ADC_proof_l153_153129


namespace general_term_sum_formula_l153_153025

variable {n : ℕ}
variable {a b q : ℤ}
variable {a_n b_n : ℕ → ℤ}
variable {S_n : ℕ → ℤ}

-- Given conditions
def geom_seq (a_n : ℕ → ℤ) (q : ℤ) := ∀ n, a_n (n + 1) = q * a_n n

def cond1 := geom_seq a_n q
def cond2 := a_n 1 + a_n 2 + a_n 3 = -6
def cond3 := a_n 1 * a_n 2 * a_n 3 = 64
def cond4 := |q| > 1

-- First Part: Proving the formula for a_n
theorem general_term (cond1 : cond1) (cond2 : cond2) (cond3 : cond3) (cond4 : cond4) :
  ∀ n, a_n n = (-2)^n := 
sorry

-- Second Part: Formula for the sum of the first n terms of {b_n}
def b_n (n : ℕ) := (2 * n + 1) * a_n n

theorem sum_formula (cond1 : cond1) (cond2 : cond2) (cond3 : cond3) (cond4 : cond4) :
  (S_n n = ∑ i in finset.range n, b_n i) → 
  (S_n n = - (10 / 9) - ((6 * n + 5) * (-2)^(n + 1) / 9)) := 
sorry

end general_term_sum_formula_l153_153025


namespace complex_number_power_l153_153768

theorem complex_number_power :
  let i : ℂ := complex.I in
  ( (1 + i) / (1 - i) )^2013 = i :=
by
  sorry

end complex_number_power_l153_153768


namespace pascal_triangle_contains_prime_l153_153058

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153058


namespace square_of_binomial_l153_153588

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l153_153588


namespace solve_equation_l153_153196

theorem solve_equation (x : ℝ) (h : x ≠ 1) : 
  1 / (x - 1) + 1 = 3 / (2 * x - 2) ↔ x = 3 / 2 := by
  sorry

end solve_equation_l153_153196


namespace angle_EKF_135_l153_153133

-- Define the context and points involved
variables {A B C D E F G K : Type} [has_inner A] [has_inner B] [has_inner C] [has_inner D]
variables [has_inner E] [has_inner F] [has_inner G] [has_inner K]
variables [inhabited A] [inhabited B] [inhabited C] [inhabited D] [inhabited E] [inhabited F]
variables [inhabited G] [inhabited K]

-- Define that ABCD is a square
def is_square (A B C D : Type) := 
  ∀ p : Type, p ∈ {angle A B C = angle B C D} ∧ p ∈ {length A B = length B C}

-- Define the conditions of the problem in Lean 4
variable (squareABCD : is_square A B C D)
variable (E_on_BC : E ∈ segment B C)
variable (F_on_CD : F ∈ segment C D)
variable (perp_FG_AE_passes_G : line F G ⊥ line A E ∧ G = intersection (line A E) (diagonal B D))
variable (K_on_FG_AK_eq_EF : K ∈ line_segment F G ∧ length A K = length E F)

-- Define the proof goal in Lean 4
theorem angle_EKF_135 :
  angle E K F = 135 :=
sorry

end angle_EKF_135_l153_153133


namespace fraction_of_employees_laid_off_l153_153942

theorem fraction_of_employees_laid_off
    (total_employees : ℕ)
    (salary_per_employee : ℕ)
    (total_payment_after_layoffs : ℕ)
    (h1 : total_employees = 450)
    (h2 : salary_per_employee = 2000)
    (h3 : total_payment_after_layoffs = 600000) :
    (total_employees * salary_per_employee - total_payment_after_layoffs) / (total_employees * salary_per_employee) = 1 / 3 := 
by
    sorry

end fraction_of_employees_laid_off_l153_153942


namespace cosine_value_of_angle_between_vectors_l153_153765

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (1, 3)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

noncomputable def cosine_angle (u v : ℝ × ℝ) : ℝ :=
  dot_product u v / (magnitude u * magnitude v)

theorem cosine_value_of_angle_between_vectors :
  cosine_angle a b = 7 * Real.sqrt 2 / 10 :=
by
  sorry

end cosine_value_of_angle_between_vectors_l153_153765


namespace average_math_chemistry_l153_153941

variables (M P C : ℕ)

axiom h1 : M + P = 60
axiom h2 : C = P + 20

theorem average_math_chemistry : (M + C) / 2 = 40 :=
by
  sorry

end average_math_chemistry_l153_153941


namespace prism_volume_l153_153306

noncomputable def volume_of_prism (PQ PR θ: ℝ) (tanθ: ℝ) (sinθ: ℝ) : ℝ :=
  let QR := Real.sqrt (PQ^2 + PR^2) in
  let area_base := (1 / 2) * PQ * PR in
  let cosθ := Real.sqrt (1 - (sinθ ^ 2)) in
  let h := tanθ * PQ in
  area_base * h

theorem prism_volume (PQ PR h θ tanθ sinθ: ℝ)
  (h₀: PQ = Real.sqrt 5)
  (h₁: PR = Real.sqrt 5)
  (h₂: ∠(PQR) = π / 2)
  (h₃: tanθ = h / (Real.sqrt 5))
  (h₄: sinθ = 3 / 5) :
  volume_of_prism PQ PR θ tanθ sinθ = (15 * Real.sqrt 5) / 8 :=
sorry

end prism_volume_l153_153306


namespace quadratic_roots_correct_l153_153220

theorem quadratic_roots_correct (x : ℝ) : (x^2 = 2 * x) ↔ (x = 0 ∨ x = 2) := 
by
  sorry

end quadratic_roots_correct_l153_153220


namespace FG_half_AB_l153_153480

variables (A B C U D E F G : Type)
variables [RightTriangle ABC] [Circumcenter ABC U] [PointsOnSides AC BC D E] [AngleEUD90 E U D]
variables [ProjectionsOnAB D E F G]

theorem FG_half_AB :
  segment_length_FG F G = segment_length_AB A B / 2 := sorry

end FG_half_AB_l153_153480


namespace minimum_value_integral_l153_153157

theorem minimum_value_integral (a : ℝ) : (∃ a : ℝ, ∫ x in 0..1, |a*x - x^3| = 1/8) :=
sorry

end minimum_value_integral_l153_153157


namespace kangaroo_can_jump_exact_200_in_30_jumps_l153_153633

/-!
  A kangaroo can jump:
  - 3 meters using its left leg
  - 5 meters using its right leg
  - 7 meters using both legs
  - -3 meters backward
  We need to prove that the kangaroo can jump exactly 200 meters in 30 jumps.
 -/

theorem kangaroo_can_jump_exact_200_in_30_jumps :
  ∃ (n3 n5 n7 nm3 : ℕ),
    (n3 + n5 + n7 + nm3 = 30) ∧
    (3 * n3 + 5 * n5 + 7 * n7 - 3 * nm3 = 200) :=
sorry

end kangaroo_can_jump_exact_200_in_30_jumps_l153_153633


namespace frisbee_total_distance_correct_l153_153331

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end frisbee_total_distance_correct_l153_153331


namespace joggers_difference_l153_153322

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end joggers_difference_l153_153322


namespace pins_after_month_correct_l153_153513

variable (pins_per_day : ℕ)
variable (pins_per_week : ℕ)
variable (num_members : ℕ)
variable (initial_pins : ℕ)
variable (days_in_month : ℕ)
variable (weeks_in_month : ℝ)

def total_pins_after_month :=
  initial_pins + (num_members * pins_per_day * days_in_month) - (num_members * pins_per_week * weeks_in_month).toNat

theorem pins_after_month_correct :
  pins_per_day = 10 →
  pins_per_week = 5 →
  num_members = 20 →
  initial_pins = 1000 →
  days_in_month = 30 →
  weeks_in_month = 30 / 7 →
  total_pins_after_month pins_per_day pins_per_week num_members initial_pins days_in_month weeks_in_month = 6571 :=
by
  intros h1 h2 h3 h4 h5 h6
  unfold total_pins_after_month
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end pins_after_month_correct_l153_153513


namespace fractions_sum_correct_l153_153335

noncomputable def fractions_sum : ℝ := (3 / 20) + (5 / 200) + (7 / 2000) + 5

theorem fractions_sum_correct : fractions_sum = 5.1785 :=
by
  sorry

end fractions_sum_correct_l153_153335


namespace hazel_speed_l153_153820

theorem hazel_speed :
  (Hazel running at constant speed)
  → (∀ d : ℝ, d = 17.7 ∧ ∀ t : ℝ, t = 5 → (Hazel covers d kms in t minutes))
  → (∀ km_m : ℝ, km_m = 1000 → (∀ min_s : ℝ, min_s = 60 → (Hazel's speed = 59 m/s))) :=
by
  sorry

end hazel_speed_l153_153820


namespace geometric_sequence_problem_l153_153846

theorem geometric_sequence_problem 
  (a : ℕ → ℝ)
  (h_geo : ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n)
  (h1 : a 7 * a 11 = 6)
  (h2 : a 4 + a 14 = 5) :
  ∃ x : ℝ, x = 2 / 3 ∨ x = 3 / 2 := by
  sorry

end geometric_sequence_problem_l153_153846


namespace trajectory_of_P_l153_153226

-- Define points P, A, and B in a 2D plane
variable {P A B : EuclideanSpace ℝ (Fin 2)}

-- Define the condition that the sum of the distances from P to A and P to B equals the distance between A and B
def sum_of_distances_condition (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P A + dist P B = dist A B

-- Main theorem statement: If P satisfies the above condition, then P lies on the line segment AB
theorem trajectory_of_P (P A B : EuclideanSpace ℝ (Fin 2)) (h : sum_of_distances_condition P A B) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • A + (1 - t) • B :=
  sorry

end trajectory_of_P_l153_153226


namespace equation_of_line_l153_153725

-- Define the parametric equations
def x_param (t : ℝ) : ℝ := 3 * t + 6
def y_param (t : ℝ) : ℝ := 5 * t - 7

-- Define the main theorem to prove
theorem equation_of_line :
  ∃ m b : ℝ, ∀ (x y t : ℝ), x_param t = x → y_param t = y → y = m * x + b ∧ m = 5 / 3 ∧ b = -17 :=
by 
  simp [x_param, y_param, mul_div_assoc, sub_eq_add_neg, ← div_sub, add_assoc]
  sorry

end equation_of_line_l153_153725


namespace train_speed_kmh_l153_153653

-- Definitions based on the conditions
variables (L V : ℝ)
variable (h1 : L = 10 * V)
variable (h2 : L + 600 = 30 * V)

-- The proof statement, no solution steps, just the conclusion
theorem train_speed_kmh : (V * 3.6) = 108 :=
by
  sorry

end train_speed_kmh_l153_153653


namespace rachel_envelopes_first_hour_l153_153896

theorem rachel_envelopes_first_hour (total_envelopes : ℕ) (hours : ℕ) (e2 : ℕ) (e_per_hour : ℕ) :
  total_envelopes = 1500 → hours = 8 → e2 = 141 → e_per_hour = 204 →
  ∃ e1 : ℕ, e1 = 135 :=
by
  sorry

end rachel_envelopes_first_hour_l153_153896


namespace average_class_a_average_class_b_expectation_of_X_l153_153348

noncomputable section

def selfStudyTimesClassA : List ℝ := [8, 13, 28, 32, 39]
def selfStudyTimesClassB : List ℝ := [12, 25, 26, 28, 31]
def threshold : ℝ := 26
def average (l : List ℝ) : ℝ := l.sum / l.length

def averageClassA := average selfStudyTimesClassA
def averageClassB := average selfStudyTimesClassB

theorem average_class_a : averageClassA = 24 := by
  sorry

theorem average_class_b : averageClassB = 24.4 := by
  sorry

def binom (n k : ℕ) : ℕ := Nat.choose n k
def probabilityX (x : ℕ) : ℝ := 
  match x with
  | 0 => binom 2 2 * binom 3 2 / (binom 5 2 * binom 5 2)
  | 1 => (binom 2 1 * binom 3 1 * binom 3 2 + binom 2 2 * binom 3 1 * binom 2 1) / (binom 5 2 * binom 5 2)
  | 2 => (binom 2 1 * binom 3 1 * binom 3 1 * binom 2 1 + binom 3 2 * binom 3 2 + binom 2 2 * binom 2 2) / (binom 5 2 * binom 5 2)
  | 3 => (binom 3 2 * binom 3 1 * binom 2 1 + binom 2 1 * binom 3 1 * binom 2 2) / (binom 5 2 * binom 5 2)
  | 4 => binom 3 2 * binom 2 2 / (binom 5 2 * binom 5 2)
  | _ => 0

def expectationX : ℝ := ∑ x in Finset.range 5, x * probabilityX x

theorem expectation_of_X : expectationX = 2 := by
  sorry

end average_class_a_average_class_b_expectation_of_X_l153_153348


namespace initial_percentage_decrease_l153_153935

theorem initial_percentage_decrease (P x : ℝ) (h1 : 0 < P) (h2 : 0 ≤ x) (h3 : x ≤ 100) :
  ((P - (x / 100) * P) * 1.50 = P * 1.20) → x = 20 :=
by
  sorry

end initial_percentage_decrease_l153_153935


namespace unique_cells_l153_153194

variable (A : Type) 
variable (is_cell : set A → Prop)

-- Definition of C_a
def C_a (a : A) : set A :=
  { b : A | ∃ (n : ℕ) (c : fin (n + 1) → A), c 0 = a ∧ c n = b }

-- The main theorem to prove
theorem unique_cells (a : A) : ∃! C : set A, a ∈ C ∧ is_cell C :=
by
  -- The proof would go here
  sorry

end unique_cells_l153_153194


namespace correct_proposition_l153_153411

-- Definitions representing the given propositions
def propositionA : Prop := ∀ (R^2 : ℝ), R^2 = 0.80 → (contributionRate R^2 = 80%)
def propositionB : Prop := ∀ (var1 var2 : Type) (table : Matrix var1 var2),
  (is2x2ContingencyTable table) → (largerProductDifferenceOfDiagonalEntries table → unrelated var1 var2)
def propositionC : Prop := ∀ (R^2 : ℝ) (residuals : ℝ),
  (regressionEffect R^2 → smaller_R2 → largerSumOfSquaresOfResiduals residuals) → betterModelFit
def propositionD : Prop := ∀ (r : ℝ),
  (linearCorrelationCoefficient r → (abs r = 1) → strongerLinearCorrelation)

-- The theorem asserting that proposition D is correct given all conditions
theorem correct_proposition : propositionD :=
by sorry

end correct_proposition_l153_153411


namespace pascal_triangle_47_l153_153094

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153094


namespace jill_arrives_before_jack_l153_153475

theorem jill_arrives_before_jack : 
  ∀ (distance speed_jill speed_jack : ℝ), 
  distance = 3 → 
  speed_jill = 12 → 
  speed_jack = 3 → 
  (distance / speed_jack - distance / speed_jill) * 60 = 45 :=
by
  intros distance speed_jill speed_jack h1 h2 h3
  rw [h1, h2, h3]
  simp
  norm_num
  sorry

end jill_arrives_before_jack_l153_153475


namespace square_of_binomial_eq_100_l153_153578

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l153_153578


namespace JoggerDifference_l153_153319

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end JoggerDifference_l153_153319


namespace max_difference_evens_l153_153681

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l153_153681


namespace parabola_directrix_l153_153354

theorem parabola_directrix (x y : ℝ) (h_eqn : y = -3 * x^2 + 6 * x - 5) :
  y = -23 / 12 :=
sorry

end parabola_directrix_l153_153354


namespace length_I1I2_l153_153876

theorem length_I1I2 :
  ∀ (ABC : Triangle) (A B C : Point) (O : Point) (R : ℝ) (D E : Point) (I1 I2 : Point),
  isosceles_right_triangle A B C ∧
  circumcircle_diameter ABC O R 40 ∧
  points_on_arc_not_containing_A B C A D E ∧
  AD_and_AE_trisect_angle_BAC A B C D E 90 ∧
  I1_is_incenter_of_triangle_ABE A B E I1 ∧
  I2_is_incenter_of_triangle_ACD A C D I2 →
  dist (I1, I2) = 20 - sqrt(6) + sqrt(2) ∧
  let a := 20 in
  let b := 1 in
  let c := 0 in
  let d := -1 in
  a + b + c + d = 20 := 
begin
  sorry,
end

end length_I1I2_l153_153876


namespace remainder_when_7n_divided_by_4_l153_153262

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l153_153262


namespace range_of_x_min_max_f_min_value_at_2_max_value_at_1_l153_153407

open Real

noncomputable def f (x : ℝ) : ℝ :=
  (log x / log 2 - 1) * (log x / log 2 - 2)

theorem range_of_x (x : ℝ) (h : 9^x - 4 * 3^(x+1) + 27 ≤ 0) :
  1 ≤ x ∧ x ≤ 2 :=
sorry

theorem min_max_f (x : ℝ) (h : 1 ≤ x ∧ x ≤ 2) :
  0 ≤ f x ∧ f x ≤ 2 :=
sorry

theorem min_value_at_2 : f 2 = 0 := sorry

theorem max_value_at_1 : f 1 = 2 := sorry

end range_of_x_min_max_f_min_value_at_2_max_value_at_1_l153_153407


namespace remainder_when_7n_divided_by_4_l153_153266

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l153_153266


namespace pascal_triangle_contains_prime_l153_153057

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153057


namespace sign_up_combinations_l153_153953

theorem sign_up_combinations (num_students : ℕ) (num_groups : ℕ)
  (students : Fin num_students) (groups : Fin num_groups)
  (num_students = 3) (num_groups = 4) :
  (num_groups ^ num_students) = 4 ^ 3 := 
by
  sorry

end sign_up_combinations_l153_153953


namespace max_sqrt_expr_l153_153391

variable {x y z : ℝ}

noncomputable def f (x y z : ℝ) : ℝ := Real.sqrt x + Real.sqrt (2 * y) + Real.sqrt (3 * z)

theorem max_sqrt_expr (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z = 2) : 
  f x y z ≤ 2 * Real.sqrt 3 := by
  sorry

end max_sqrt_expr_l153_153391


namespace largest_divisible_number_l153_153105

theorem largest_divisible_number : 
  ∀ (numbers : Finset ℕ), 
  numbers = {5, 6, 7, 8, 9} → 
  ∃ (selected_numbers : Finset ℕ), 
  selected_numbers ⊆ numbers ∧ 
  selected_numbers.card = 4 ∧ 
  (∃ (n : ℕ), 
    (∀ (d ∈ selected_numbers), d ∈ numbers) ∧ 
    n = selected_numbers.sum * 10 + selected_numbers.min' sorry ∧ 
    n ≥ 1000 ∧
    n < 10000 ∧
    n % 3 = 0 ∧ 
    n % 5 = 0 ∧ 
    n % 7 = 0 ∧ 
    (∀ m, 
      (∀ (d ∈ selected_numbers), d ∈ numbers) ∧ 
      m = selected_numbers.sum * 10 + selected_numbers.min' sorry ∧ 
      m ≥ 1000 ∧
      m < 10000 ∧
      m % 3 = 0 ∧ 
      m % 5 = 0 ∧ 
      m % 7 = 0 → 
      m ≤ n
    ) 
  ) :=
by
  sorry

end largest_divisible_number_l153_153105


namespace problem_statement_l153_153894

noncomputable def point (X : Type) := X
def k_coeffs (k : list ℝ) := k
def A_coeffs (A : list (point ℝ × point ℝ)) := A
def condition_sum (k : list ℝ) := k.sum

def satisfies_equation (X : point ℝ) (k : list ℝ) (A : list (point ℝ × point ℝ)) (c : ℝ) : Prop :=
∑ (i : ℕ) in list.finRange k.length, 
  ((k[i] * ((X.1 - A[i].1)^2 + (X.2 - A[i].2)^2))) = c

theorem problem_statement (k : list ℝ) (A : list (point ℝ × point ℝ)) (c : ℝ) (X : point ℝ) :
  (condition_sum k ≠ 0 → satisfies_equation X k A c → (circle X ∨ X = ∅)) ∧
  (condition_sum k = 0 → satisfies_equation X k A c → (line X ∨ plane X ∨ X = ∅)) :=
sorry

end problem_statement_l153_153894


namespace inverse_piecewise_correct_l153_153925

def piecewise_function_inverse (x : ℝ) : ℝ :=
  if x >= 0 then 2*x else -x^2

def piecewise_inverse_function (y : ℝ) : ℝ :=
  if y >= 0 then y/2 else -Real.sqrt (-y)

theorem inverse_piecewise_correct :
  ∀ y : ℝ, piecewise_function_inverse (piecewise_inverse_function y) = y :=
by
  intros y
  sorry

end inverse_piecewise_correct_l153_153925


namespace range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l153_153803

noncomputable def f (x : ℝ) : ℝ := Real.log (Real.exp x + Real.exp (-x)) + x^2

theorem range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3 :
  {x : ℝ | f (2 * x) > f (x + 3)} = {x : ℝ | x < -1 ∨ x > 3} :=
by
  sorry

end range_of_x_for_which_f_of_2x_greater_than_f_of_x_plus_3_l153_153803


namespace largest_diff_even_digits_l153_153666

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l153_153666


namespace part_I_part_II_l153_153793

noncomputable def f (x : ℝ) : ℝ :=
  if x < -1 then (1/2)^x else x^2 + 3 * x

theorem part_I (x : ℝ) : f x < 4 ↔ -2 < x ∧ x < 1 :=
by
  sorry

theorem part_II (m : ℝ) : (∀ x ∈ Ioo 0 2, f x ≥ m * x - 2) ↔ m ≤ 3 + 2 * Real.sqrt 2 :=
by
  sorry

end part_I_part_II_l153_153793


namespace train_crossing_time_l153_153652

/-- The time taken for a train of length 250 meters traveling at 
    a speed of 70 km/hr to cross a stationary object is approximately 12.86 seconds. -/
theorem train_crossing_time :
  let length_of_train := 250 -- in meters
  let speed_in_kmh := 70 -- in km/hr
  let speed_in_ms := (speed_in_kmh * 1000) / 3600 -- convert to m/s
  let time_taken := length_of_train / speed_in_ms -- in seconds
  abs(time_taken - 12.86) < 0.01 := 
by
  sorry

end train_crossing_time_l153_153652


namespace no_all_blue_possible_l153_153505

-- Define initial counts of chameleons
def initial_red : ℕ := 25
def initial_green : ℕ := 12
def initial_blue : ℕ := 8

-- Define the invariant condition
def invariant (r g : ℕ) : Prop := (r - g) % 3 = 1

-- Define the main theorem statement
theorem no_all_blue_possible : ¬∃ r g, r = 0 ∧ g = 0 ∧ invariant r g :=
by {
  sorry
}

end no_all_blue_possible_l153_153505


namespace sector_area_l153_153641

def central_angle := 120 -- in degrees
def radius := 3 -- in units

theorem sector_area (n : ℕ) (R : ℕ) (h₁ : n = central_angle) (h₂ : R = radius) :
  (n * R^2 * Real.pi / 360) = 3 * Real.pi :=
by
  sorry

end sector_area_l153_153641


namespace max_difference_evens_l153_153680

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l153_153680


namespace min_value_ineq_l153_153399

theorem min_value_ineq (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + y = 1) : 
  (∀ z : ℝ, z = (4 / x + 1 / y) → z ≥ 9) :=
by
  sorry

end min_value_ineq_l153_153399


namespace scientific_notation_00625_l153_153352

/-- Definition of scientific notation: a * 10^n where 1 ≤ a < 10 and n is an integer -/
def is_scientific_notation (a : ℝ) (n : ℤ) (b : ℝ) : Prop :=
1 ≤ a ∧ a < 10 ∧ b = a * 10^n

/-- The number 0.00625 can be expressed as the scientific notation 6.25 * 10^(-3) -/
theorem scientific_notation_00625 : ∃ a n, is_scientific_notation a n 0.00625 ∧ a = 6.25 ∧ n = -3 :=
by
  use 6.25
  use -3
  split
  . unfold is_scientific_notation
    split
    . exact le_of_lt (by norm_num)
    split
    . norm_num
    . norm_num
  split
  . norm_num
  . norm_num


end scientific_notation_00625_l153_153352


namespace find_a_l153_153872

variable (a : ℝ)

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ a }
def B : Set ℝ := { y | ∃ x ∈ A a, y = x + 1 }
def C : Set ℝ := { y | ∃ x ∈ A a, y = x^2 }

theorem find_a (h1: nonempty A) (h2: B = C) : a = 0 ∨ a = (1 + Real.sqrt 5) / 2 := 
sorry

end find_a_l153_153872


namespace probability_heads_in_12_flips_l153_153997

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153997


namespace prime_count_30_40_l153_153046

theorem prime_count_30_40 : 
  (finset.filter nat.prime {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}).card = 2 := 
by
  sorry

end prime_count_30_40_l153_153046


namespace B_visited_LakeQinghai_l153_153144

-- Definitions for the locations
inductive Place : Type
| LakeQinghai
| HundredMileRapeseedFlowerSea
| TeaCardSkyMirror

open Place

-- Structure to represent a person and the places they have visited
structure Person :=
  (visited : set Place)
  (visited_more_than : Person → Prop)

-- Definitions of person A, B, C with their visitation statuses
def A : Person := {
  visited := {LakeQinghai}, -- A has not visited Hundred Mile Rapeseed Flower Sea and hence can be assumed to have visited Lake Qinghai if it visited more than B
  visited_more_than := λ B, (B.visited = {LakeQinghai})
}

def B : Person := {
  visited := 
  -- Using an axiom that could be derived from discussion
  sorry,
  visited_more_than := sorry
}

def C : Person := {
  visited := {LakeQinghai},
  visited_more_than := sorry
}

-- Condition that all three have visited the same place
axiom C_condition : C.visited = A.visited ∧ C.visited = B.visited 

-- Theorem: B visited Lake Qinghai
theorem B_visited_LakeQinghai : (B.visited = {LakeQinghai}) :=
by
  -- Since B_condition is strictly derived from the problem definition
  sorry

end B_visited_LakeQinghai_l153_153144


namespace sum_of_first_nine_coprime_numbers_eq_100_l153_153516

theorem sum_of_first_nine_coprime_numbers_eq_100 :
  ∃ (a b c d e f g h i : ℕ),
    (nat.gcd a b = 1) ∧ (nat.gcd a c = 1) ∧ (nat.gcd b c = 1) ∧ 
    (nat.gcd a d = 1) ∧ (nat.gcd b d = 1) ∧ (nat.gcd c d = 1) ∧ 
    (nat.gcd a e = 1) ∧ (nat.gcd b e = 1) ∧ (nat.gcd c e = 1) ∧ 
    (nat.gcd d e = 1) ∧ (nat.gcd a f = 1) ∧ (nat.gcd b f = 1) ∧ 
    (nat.gcd c f = 1) ∧ (nat.gcd d f = 1) ∧ (nat.gcd e f = 1) ∧ 
    (nat.gcd a g = 1) ∧ (nat.gcd b g = 1) ∧ (nat.gcd c g = 1) ∧ 
    (nat.gcd d g = 1) ∧ (nat.gcd e g = 1) ∧ (nat.gcd f g = 1) ∧ 
    (nat.gcd a h = 1) ∧ (nat.gcd b h = 1) ∧ (nat.gcd c h = 1) ∧ 
    (nat.gcd d h = 1) ∧ (nat.gcd e h = 1) ∧ (nat.gcd f h = 1) ∧ 
    (nat.gcd g h = 1) ∧ (nat.gcd a i = 1) ∧ (nat.gcd b i = 1) ∧ 
    (nat.gcd c i = 1) ∧ (nat.gcd d i = 1) ∧ (nat.gcd e i = 1) ∧ 
    (nat.gcd f i = 1) ∧ (nat.gcd g i = 1) ∧ (nat.gcd h i = 1) ∧ 
    (a + b + c + d + e + f + g + h + i = 100) :=
begin
  use [2, 3, 5, 7, 11, 13, 17, 19, 23],
  -- verifying gcd conditions
  split, repeat {exact dec_trivial},
  -- verifying sum condition
  exact dec_trivial
end

end sum_of_first_nine_coprime_numbers_eq_100_l153_153516


namespace range_of_a_l153_153003

noncomputable theory
open Real

def f (x a : ℝ) : ℝ := (x-2)^2 * exp x + a * exp (-x)
def g (x a : ℝ) : ℝ := 2 * a * abs (x-2)

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f x a = g x a → (1 < a ∧ a < (exp 2 / (2 * exp 1 - 1)))) ↔
  (∀ x : ℝ, f x a = g x a → (a ∈ (Ioo 1 (exp 2 / (2 * exp 1 - 1))))) :=
begin
  sorry,
end

end range_of_a_l153_153003


namespace cows_to_add_l153_153883

theorem cows_to_add (initial_cows initial_pigs initial_goats : ℕ) 
  (added_pigs added_goats total_animals final_animals : ℕ) 
  (h_initial_cows : initial_cows = 2) 
  (h_initial_pigs : initial_pigs = 3) 
  (h_initial_goats : initial_goats = 6) 
  (h_added_pigs : added_pigs = 5) 
  (h_added_goats : added_goats = 2) 
  (h_final_animals : final_animals = 21)
  (h_total_animals : total_animals = initial_cows + initial_pigs + initial_goats) : 
  initial_cows + added_pigs + added_goats + ?n = final_animals := 
begin
  sorry
end

end cows_to_add_l153_153883


namespace angle_ratio_2B_angle_ratio_3B_l153_153891

-- Problem 1:
theorem angle_ratio_2B (A B a b c : ℝ) (h1 : A = 2 * B) : a^2 = b * (b + c) :=
by 
sor

-- Problem 2:
theorem angle_ratio_3B (A B a b c : ℝ) (h1 : A = 3 * B) : c^2 = (1 / b) * (a - b) * (a^2 - b^2) :=
by 
sorry

end angle_ratio_2B_angle_ratio_3B_l153_153891


namespace speed_ratio_of_runners_l153_153503

theorem speed_ratio_of_runners (v_A v_B : ℝ) (c : ℝ)
  (h1 : 0 < v_A ∧ 0 < v_B) -- They run at constant, but different speeds
  (h2 : (v_B / v_A) = (2 / 3)) -- Distance relationship from meeting points
  : v_B / v_A = 2 :=
by
  sorry

end speed_ratio_of_runners_l153_153503


namespace find_f_8_l153_153786

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f x
axiom periodicity : ∀ x : ℝ, f (x + 6) = f x
axiom function_on_interval : ∀ x : ℝ, -3 < x ∧ x < 0 → f x = 2 * x - 5

theorem find_f_8 : f 8 = -9 :=
by
  sorry

end find_f_8_l153_153786


namespace pins_after_month_correct_l153_153512

variable (pins_per_day : ℕ)
variable (pins_per_week : ℕ)
variable (num_members : ℕ)
variable (initial_pins : ℕ)
variable (days_in_month : ℕ)
variable (weeks_in_month : ℝ)

def total_pins_after_month :=
  initial_pins + (num_members * pins_per_day * days_in_month) - (num_members * pins_per_week * weeks_in_month).toNat

theorem pins_after_month_correct :
  pins_per_day = 10 →
  pins_per_week = 5 →
  num_members = 20 →
  initial_pins = 1000 →
  days_in_month = 30 →
  weeks_in_month = 30 / 7 →
  total_pins_after_month pins_per_day pins_per_week num_members initial_pins days_in_month weeks_in_month = 6571 :=
by
  intros h1 h2 h3 h4 h5 h6
  unfold total_pins_after_month
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end pins_after_month_correct_l153_153512


namespace exists_polyhedron_with_equal_edges_and_diagonals_l153_153347

inductive Polyhedron (V E F : ℕ) : Type
| mk : (is_convex : Prop) → (num_vertices : ℕ) → (num_edges : ℕ) → (num_faces : ℕ) → Polyhedron V E F

def is_convex (P : Polyhedron) : Prop := 
  -- This should contain the formal definition of convexity
  sorry 

def number_of_diagonals (P : Polyhedron) : ℕ :=
  (P.num_vertices * (P.num_vertices - 3)) / 2

def number_of_edges (P : Polyhedron) : ℕ :=
  P.num_edges 

theorem exists_polyhedron_with_equal_edges_and_diagonals : 
  ∃ (P : Polyhedron), is_convex P ∧ number_of_edges P = number_of_diagonals P :=
sorry

end exists_polyhedron_with_equal_edges_and_diagonals_l153_153347


namespace max_stories_on_odd_pages_l153_153284

theorem max_stories_on_odd_pages 
    (stories : Fin 30 -> Fin 31) 
    (h_unique : Function.Injective stories) 
    (h_bounds : ∀ i, stories i < 31)
    : ∃ n, n = 23 ∧ ∃ f : Fin n -> Fin 30, ∀ j, f j % 2 = 1 := 
sorry

end max_stories_on_odd_pages_l153_153284


namespace arithmetic_sequence_an_12_l153_153137

theorem arithmetic_sequence_an_12 {a : ℕ → ℝ} (h_arith : ∀ n, a (n + 1) - a n = a 1 - a 0)
  (h_a3 : a 3 = 9)
  (h_a6 : a 6 = 15) :
  a 12 = 27 := 
sorry

end arithmetic_sequence_an_12_l153_153137


namespace min_value_g_l153_153795

def f (x k : ℝ) := k * x^2 + (2 * k - 1) * x + k
def g (x k : ℝ) := Real.logb 2 (x + k)

theorem min_value_g 
  (k : ℝ) 
  (h : f 0 k = 7) 
  (hx : ∀ x, 9 ≤ x → g x k ≥ g 9 k) : 
  g 9 7 = 4 :=
by
  have h₁ : k = 7 := 
    by
      rw [f] at h
      simp at h
      exact h.symm
  
  rw [g, h₁]
  simp
  sorry

end min_value_g_l153_153795


namespace circle_radius_eq_sqrt2_l153_153203

theorem circle_radius_eq_sqrt2 (a : ℝ) (h : (∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 = 0 ↔ ∀ y : ℝ, (x - 2) ^ 2 + y ^ 2 = 2)) : 
    (∃ r : ℝ, r = sqrt 2) :=
begin
  use sqrt 2,
  sorry
end

end circle_radius_eq_sqrt2_l153_153203


namespace white_balls_count_l153_153623

def total_balls := 60
def green_balls := 18
def yellow_balls := 2
def red_balls := 15
def purple_balls := 3
def neither_red_nor_purple_prob := 0.7

theorem white_balls_count : 
  let W := total_balls * neither_red_nor_purple_prob - green_balls - yellow_balls
  W = 24 := 
by sorry

end white_balls_count_l153_153623


namespace pascal_row_contains_prime_47_l153_153051

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153051


namespace coeff_of_x_inv_3_is_21_l153_153451

noncomputable def coeff : ℤ := 
  let n := 7 in
  let r := 6 in
  Nat.choose n r * (-1) ^ r * 3 ^ (n - r)

theorem coeff_of_x_inv_3_is_21 :
  coeff = 21 :=
by sorry

end coeff_of_x_inv_3_is_21_l153_153451


namespace find_x_l153_153616

noncomputable def x := 97

theorem find_x (h : (sqrt x + sqrt 486) / sqrt 54 = 4.340259786868312) : x = 97 := 
by
  -- Proof goes here
  sorry

end find_x_l153_153616


namespace passion_fruit_crates_l153_153154

-- Step d): Lean 4 statement
theorem passion_fruit_crates (total_crates grapes_crates mangoes_crates : ℕ) (h1 : total_crates = 50) (h2 : grapes_crates = 13) (h3 : mangoes_crates = 20) : total_crates - (grapes_crates + mangoes_crates) = 17 :=
by
  rw [h1, h2, h3]
  norm_num
  -- The full proof would normally go here, but is omitted as instructed
  sorry

end passion_fruit_crates_l153_153154


namespace desk_height_proof_l153_153441

noncomputable def height_of_desk (h_chair h_total_chair h_total_desk : ℕ) : ℕ :=
  let h_dongmin := h_total_chair - h_chair in
  h_total_desk - h_dongmin

theorem desk_height_proof : height_of_desk 537 1900 2325 = 962 := by
  sorry

end desk_height_proof_l153_153441


namespace arithmetic_geometric_sequence_S6_l153_153777

variables (S : ℕ → ℕ)

-- Definitions of conditions from a)
def S2 := S 2 = 3
def S4 := S 4 = 15

-- Main proof statement
theorem arithmetic_geometric_sequence_S6 (S : ℕ → ℕ) (h1 : S 2 = 3) (h2 : S 4 = 15) :
  S 6 = 63 :=
sorry

end arithmetic_geometric_sequence_S6_l153_153777


namespace max_value_of_f_angle_C_in_triangle_l153_153002

noncomputable def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x + 2 * cos x ^ 2 - 1

theorem max_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≤ 2) ∧ (∀ k : ℤ, x = k * π + π / 6 → f x = 2) :=
sorry

theorem angle_C_in_triangle (a b c : ℝ) (A C : ℝ)
  (ha : a = 1) (hb : b = sqrt 3) (hA : f A = 2)
  (hC1 : a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * cos A)
  :
  C = π / 3 :=
sorry

end max_value_of_f_angle_C_in_triangle_l153_153002


namespace elements_with_8_as_leftmost_digit_l153_153159

def S := {k : ℕ | 0 ≤ k ∧ k ≤ 3000}

axiom digits_8_3000 : 8 ^ 3000 = 2713

axiom first_digit_8_3000 : (8 ^ 3000).digitAt 0 = 8

theorem elements_with_8_as_leftmost_digit : 
  {k ∈ S | (8 ^ k).digitAt 0 = 8}.card = 288 := 
by
  sorry

end elements_with_8_as_leftmost_digit_l153_153159


namespace largest_diff_even_digits_l153_153664

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l153_153664


namespace number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l153_153625

-- Define the conditions
def first_batch_cost : ℝ := 13200
def second_batch_cost : ℝ := 28800
def unit_price_difference : ℝ := 10
def discount_rate : ℝ := 0.8
def profit_margin : ℝ := 1.25
def last_batch_count : ℕ := 50

-- Define the theorem for the first part
theorem number_of_shirts_in_first_batch (x : ℕ) (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x)) : x = 120 :=
sorry

-- Define the theorem for the second part
theorem minimum_selling_price_per_shirt (x : ℕ) (y : ℝ)
  (h₁ : first_batch_cost / x + unit_price_difference = second_batch_cost / (2 * x))
  (h₂ : x = 120)
  (h₃ : (3 * x - last_batch_count) * y + last_batch_count * discount_rate * y ≥ (first_batch_cost + second_batch_cost) * profit_margin) : y ≥ 150 :=
sorry

end number_of_shirts_in_first_batch_minimum_selling_price_per_shirt_l153_153625


namespace limit_of_derivative_l153_153910

noncomputable def f (x : ℝ) : ℝ := Real.exp x / x

theorem limit_of_derivative (h : ∀ n, ∃ g, has_deriv_at g (f^(n+1)) x) :
  tendsto (λ n, f^(2 * n) 1 / (2 * n)!) at_top (𝓝 1) :=
sorry

end limit_of_derivative_l153_153910


namespace range_of_a_l153_153403

noncomputable def f : ℝ → ℝ := sorry

variables (a : ℝ)
variable (is_even : ∀ x : ℝ, f (x) = f (-x)) -- f is even
variable (monotonic_incr : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y) -- f is monotonically increasing in [0, +∞)

theorem range_of_a
  (h : f (Real.log a / Real.log 2) + f (Real.log (1/a) / Real.log 2) ≤ 2 * f 1) : 
  1 / 2 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l153_153403


namespace problem_solution_l153_153748

theorem problem_solution (a b c : ℕ) (h1 : a = 2014 - 2) (h2 : b = 2014 + 2) (h3 : c = 2014) :
  a * b - c^2 = -4 := 
sorry

end problem_solution_l153_153748


namespace optimal_bus_station_location_l153_153922

-- Given seven factories located on both sides of a highway and connected at points B, C, D, E, F
variable {A1 A2 A3 A4 A5 A6 A7 : Type}

-- Distance calculations along the highway and connecting roads
variable {B C D E F P : ℝ}

-- Function that calculates the total travel distance from bus station S to all factories
def total_travel_distance (S : ℝ) (u1 u2 u3 u4 u5 u6 u7 : ℝ) : ℝ :=
  u1 + u2 + u3 + u4 + u5 + u6 + u7 +

theorem optimal_bus_station_location 
  (h_initial : ∀ S : ℝ, total_travel_distance D A1 A2 A3 A4 A5 A6 A7 ≤ total_travel_distance S A1 A2 A3 A4 A5 A6 A7)
  (h_new_factory : ∀ S ∈ set.Icc D E, total_travel_distance S A1 A2 A3 A4 A5 A6 A7 + dist S P ≤ total_travel_distance S A1 A2 A3 A4 A5 A6 A7 + dist S P) :
  D = optimal_location_init ∧ ∀ S ∈ set.Icc D E, S = optimal_location_new :=
sorry

end optimal_bus_station_location_l153_153922


namespace probability_same_color_l153_153620

/-- 
  A bag contains 5 green balls and 8 white balls. If two balls are drawn simultaneously, 
  the probability that both balls are the same colour is 19/39.
-/
theorem probability_same_color (total_balls green_balls white_balls : ℕ) :
  total_balls = 13 → green_balls = 5 → white_balls = 8 →
  (5 / 13) * (4 / 12) + (8 / 13) * (7 / 12) = 19 / 39 :=
by
  intros total_balls_eq green_balls_eq white_balls_eq
  rw [total_balls_eq, green_balls_eq, white_balls_eq]
  norm_num
  sorry

end probability_same_color_l153_153620


namespace absolute_value_h_l153_153939

theorem absolute_value_h (h : ℝ) :
  (∃ r s : ℝ, r^2 + s^2 = 17 ∧ r + s = 4*h ∧ r * s = -5) → |h| = sqrt(7) / 4 :=
by sorry

end absolute_value_h_l153_153939


namespace sum_of_x_is_4_l153_153240

noncomputable def mean (a b c d e : ℝ) : ℝ := (a + b + c + d + e) / 5

def median (a b c d e : ℝ) : ℝ :=
  let l := [a, b, c, d, e].qsort (· ≤ ·)
  l.get! 2

theorem sum_of_x_is_4 :
  ∑ x in {x : ℝ | median 3 7 11 22 x = mean 3 7 11 22 x}.toFinset = 4 :=
by
  sorry

end sum_of_x_is_4_l153_153240


namespace fixed_monthly_costs_l153_153291

theorem fixed_monthly_costs
  (cost_per_component : ℕ) (shipping_cost : ℕ) 
  (num_components : ℕ) (selling_price : ℚ)
  (F : ℚ) :
  cost_per_component = 80 →
  shipping_cost = 6 →
  num_components = 150 →
  selling_price = 196.67 →
  F = (num_components * selling_price) - (num_components * (cost_per_component + shipping_cost)) →
  F = 16600.5 :=
by
  intros
  sorry

end fixed_monthly_costs_l153_153291


namespace correct_conclusions_l153_153030

theorem correct_conclusions (x : ℝ)
  (am bm : ℝ)
  (l1 l2 : ℝ)
  {f g : ℝ → ℝ}
  (h1 : ∀ x, ¬ (∃ x : ℝ, x^2 - x > 0) → x^2 - x ≤ 0)
  (h2 : ¬ (∀ am bm, ∃ m : ℝ, m ≠ 0 → am < bm → am * m^2 < bm * m^2 ))
  (h3 : ∀ a b : ℝ, (a + 2 * b ≠ 0) → a / b ≠ -2)
  (h4 : ∀ x : ℝ, f (-x) = -f x → g (-x) = g x → x > 0 → f' x > 0 → g' x > 0 → x < 0 → f' x > g' x) :
  (1 = 1) ∧ (2 = 2) := 
by
  sorry

end correct_conclusions_l153_153030


namespace possible_values_of_a_l153_153029

theorem possible_values_of_a (a x : ℤ) (h_eq : ax + 3 = 4x + 1) (hx_pos : x > 0) :
  a = 2 ∨ a = 3 :=
by {
  have h1 : (a - 4) * x = -2, from sorry, -- obtained by rearranging the original equation
  have h2 : x = (-2) / (a - 4), from sorry, -- expressing x
  have h3 : (-2) / (a - 4) > 0, from sorry, -- condition for x to be positive
  have h4 : a - 4 is a divisor of -2, from sorry, -- a - 4 must be one of the divisors of -2
  have h5 : a < 4, from sorry, -- condition derived from x being a positive integer and -2 being negative
  finish, -- concluding a = 2 or a = 3 from the derived conditions
  sorry
}

end possible_values_of_a_l153_153029


namespace union_of_A_B_complement_intersection_l153_153426

def U : Set ℝ := Set.univ

def A : Set ℝ := { x | -x^2 + 2*x + 15 ≤ 0 }

def B : Set ℝ := { x | |x - 5| < 1 }

theorem union_of_A_B :
  A ∪ B = { x | x ≤ -3 ∨ x > 4 } :=
by
  sorry

theorem complement_intersection :
  (U \ A) ∩ B = { x | 4 < x ∧ x < 5 } :=
by
  sorry

end union_of_A_B_complement_intersection_l153_153426


namespace simplify_expression_l153_153899

variable (q : ℝ)

theorem simplify_expression : ((6 * q + 2) - 3 * q * 5) * 4 + (5 - 2 / 4) * (7 * q - 14) = -4.5 * q - 55 :=
by sorry

end simplify_expression_l153_153899


namespace employee_pay_l153_153611

variable (X Y : ℝ)

theorem employee_pay (h1: X + Y = 572) (h2: X = 1.2 * Y) : Y = 260 :=
by
  sorry

end employee_pay_l153_153611


namespace distance_between_point_and_line_l153_153742

-- Define the given points
def a : ℝ × ℝ × ℝ := (2, 0, 3)
def p1 : ℝ × ℝ × ℝ := (1, 3, 2)
def p2 : ℝ × ℝ × ℝ := (2, 0, 5)

-- Define the direction vector of the line
def direction_vector : ℝ × ℝ × ℝ := 
  (p2.1 - p1.1, p2.2 - p1.2, p2.3 - p1.3)

-- Define the line parameterization as a function of t
def line (t : ℝ) : ℝ × ℝ × ℝ := 
  (1 + t, 3 - 3 * t, 2 + 3 * t)

-- Calculate the vector from a to a point on the line
def vector_to_line (t : ℝ) : ℝ × ℝ × ℝ := 
  (line t).1 - a.1, (line t).2 - a.2, (line t).3 - a.3

-- Define the function to calculate the dot product
def dot_product (u v : ℝ × ℝ × ℝ) : ℝ := 
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

-- Define the distance function
noncomputable def distance : ℝ := 
  Real.sqrt ((12/19)^2 + (22/19)^2 + (26/19)^2)

-- The theorem to prove
theorem distance_between_point_and_line : 
  distance = Real.sqrt (1060 / 361) :=
sorry

end distance_between_point_and_line_l153_153742


namespace pascal_row_contains_prime_47_l153_153047

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153047


namespace percentage_increase_decrease_exceeds_original_l153_153438

open Real

theorem percentage_increase_decrease_exceeds_original (p q M : ℝ) (hp : 0 < p) (hq1 : 0 < q) (hq2 : q < 100) (hM : 0 < M) :
  (M * (1 + p / 100) * (1 - q / 100) > M) ↔ (p > (100 * q) / (100 - q)) :=
by
  sorry

end percentage_increase_decrease_exceeds_original_l153_153438


namespace max_stamps_l153_153836

theorem max_stamps (price_per_stamp total_cents : ℕ) (h_price : price_per_stamp = 45) (h_total : total_cents = 4500) : 
  ∃ n : ℕ, n = 100 ∧ 45 * n ≤ 4500 :=
by {
  use 100,
  split,
  { 
    refl,
  },
  { 
    rw [h_price, h_total],
    exact le_refl 4500,
  }
}

end max_stamps_l153_153836


namespace angle_is_90_degrees_l153_153436

variable (θ : ℝ)
def a : ℝ × ℝ × ℝ := (Real.sin θ, 2, Real.cos θ)
def b : ℝ × ℝ × ℝ := (Real.sqrt 2 * Real.cos θ, -Real.sqrt 2 * Real.sin θ, Real.sqrt 3)

noncomputable def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def vec_add (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 + v.1, u.2 + v.2, u.3 + v.3)

def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

axiom angle_between_vec90 : 
  dot_product (vec_add (a θ) (b θ)) (vec_sub (a θ) (b θ)) = 0

theorem angle_is_90_degrees :
  ∃ (θ : ℝ), let a := (Real.sin θ, 2, Real.cos θ) in
  let b := (Real.sqrt 2 * Real.cos θ, -Real.sqrt 2 * Real.sin θ, Real.sqrt 3) in
  dot_product (vec_add a b) (vec_sub a b) = 0 :=
sorry

end angle_is_90_degrees_l153_153436


namespace remainder_when_7n_divided_by_4_l153_153251

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l153_153251


namespace geometric_seq_sum_l153_153789

theorem geometric_seq_sum (a : ℕ → ℝ) (S : ℕ → ℝ) 
  (h_geom : ∀ n, a (n + 1) = a n * (-1)) 
  (h_a3 : a 3 = 3) 
  (h_sum_cond : a 2016 + a 2017 = 0) : 
  S 101 = 3 := 
by
  sorry

end geometric_seq_sum_l153_153789


namespace regression_correlation_relation_l153_153112

variable (b r : ℝ)

theorem regression_correlation_relation (h : b = 0) : r = 0 := 
sorry

end regression_correlation_relation_l153_153112


namespace inequalities_not_simultaneous_l153_153166

theorem inequalities_not_simultaneous (a b c d : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : 0 < d)
  (ineq1 : a + b < c + d) (ineq2 : (a + b) * (c + d) < a * b + c * d) (ineq3 : (a + b) * c * d < (c + d) * a * b) :
  false := 
sorry

end inequalities_not_simultaneous_l153_153166


namespace abs_eq_neg_imp_nonpos_l153_153829

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_imp_nonpos_l153_153829


namespace simplify_expression_l153_153519

theorem simplify_expression (x : ℝ) (h : x ≠ 1) :
  (x^2 + x) / (x^2 - 2*x + 1) / ((x + 1) / (x - 1)) = x / (x - 1) :=
by
  sorry

end simplify_expression_l153_153519


namespace avg_age_increase_l153_153915

theorem avg_age_increase 
    (student_count : ℕ) (avg_student_age : ℕ) (teacher_age : ℕ) (new_count : ℕ) (new_avg_age : ℕ) (age_increase : ℕ)
    (hc1 : student_count = 23)
    (hc2 : avg_student_age = 22)
    (hc3 : teacher_age = 46)
    (hc4 : new_count = student_count + 1)
    (hc5 : new_avg_age = ((avg_student_age * student_count + teacher_age) / new_count))
    (hc6 : age_increase = new_avg_age - avg_student_age) :
  age_increase = 1 := 
sorry

end avg_age_increase_l153_153915


namespace probability_at_least_9_heads_l153_153976

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153976


namespace prime_numbers_between_30_and_40_l153_153044

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ finset.range (n-2).succ, m + 2 ≤ n → ¬ (n % (m + 2) = 0)

theorem prime_numbers_between_30_and_40 : 
  finset.card (finset.filter is_prime (finset.Ico 31 40)) = 2 :=
by
  sorry

end prime_numbers_between_30_and_40_l153_153044


namespace pins_after_one_month_l153_153514

theorem pins_after_one_month
  (daily_pins_per_member : ℕ)
  (deletion_rate_per_week_per_person : ℕ)
  (num_members : ℕ)
  (initial_pins : ℕ)
  (days_in_month : ℕ)
  (weeks_in_month : ℕ) :
  (daily_pins_per_member = 10) →
  (deletion_rate_per_week_per_person = 5) →
  (num_members = 20) →
  (initial_pins = 1000) →
  (days_in_month = 30) →
  (weeks_in_month = 4) →
  let daily_addition := daily_pins_per_member * num_members,
      monthly_addition := daily_addition * days_in_month,
      total_initial_and_new := initial_pins + monthly_addition,
      weekly_deletion := deletion_rate_per_week_per_person * num_members,
      monthly_deletion := weekly_deletion * weeks_in_month,
      final_pins := total_initial_and_new - monthly_deletion
  in final_pins = 6600 :=
by
  intros h1 h2 h3 h4 h5 h6
  have daily_addition := daily_pins_per_member * num_members
  have monthly_addition := daily_addition * days_in_month
  have total_initial_and_new := initial_pins + monthly_addition
  have weekly_deletion := deletion_rate_per_week_per_person * num_members
  have monthly_deletion := weekly_deletion * weeks_in_month
  have final_pins := total_initial_and_new - monthly_deletion
  sorry

end pins_after_one_month_l153_153514


namespace coin_flip_heads_probability_l153_153965

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153965


namespace find_area_EPQ_l153_153913

noncomputable theory
open_locale classical

structure Trapezoid (A B C D E P Q F : Type) :=
(area : ℝ)
(E_properties : E = intersection (extend AD) (extend BC))
(midpoint_PQ : is_midpoint P BC ∧ is_midpoint Q AD)
(ratio : EF / FC = EP / EQ = 1 / 3)

def area_triangle (A B C : Type) := ℝ -- Placeholder for triangle area function

theorem find_area_EPQ {A B C D E P Q F : Type} [trapezoid : Trapezoid A B C D E P Q F] :
  area_triangle E P F = 3 / 32 :=
sorry

end find_area_EPQ_l153_153913


namespace heartsuit_sum_l153_153104

def heartsuit (x : ℝ) : ℝ := (x + x^2 + x^3) / 3

theorem heartsuit_sum : heartsuit 1 + heartsuit 2 + heartsuit 3 = 56 / 3 :=
by
  sorry

end heartsuit_sum_l153_153104


namespace trigonometric_inequality_l153_153869

theorem trigonometric_inequality :
  let a := (1 / 2) * Real.cos (6 * Real.pi / 180) - (Real.sqrt 3 / 2) * Real.sin (6 * Real.pi / 180)
  let b := (2 * Real.tan (13 * Real.pi / 180)) / (1 + (Real.tan (13 * Real.pi / 180)) ^ 2)
  let c := Real.sqrt ((1 - Real.cos (50 * Real.pi / 180)) / 2)
  in a < c ∧ c < b :=
by
  sorry

end trigonometric_inequality_l153_153869


namespace square_of_binomial_l153_153580

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l153_153580


namespace height_of_trapezoid_l153_153458

theorem height_of_trapezoid 
  (BD AC midsegment_height : ℝ) 
  (h_diagonals : BD = 6) 
  (h_diagonals2 : AC = 8) 
  (h_midsegment : midsegment_height = 5) : 
  ∃ (height : ℝ), height = 4.8 :=
by 
  exists 4.8 
  sorry

end height_of_trapezoid_l153_153458


namespace missing_digit_divisible_by_11_l153_153350

theorem missing_digit_divisible_by_11 (A : ℕ) (h : 1 ≤ A ∧ A ≤ 9) (div_11 : (100 + 10 * A + 2) % 11 = 0) : A = 3 :=
sorry

end missing_digit_divisible_by_11_l153_153350


namespace product_of_sisters_and_brothers_l153_153122

/-- 
In a family, Emma has 4 sisters and 6 brothers. Eric, her brother, 
has some number of sisters (S) and some number of brothers (B). 
Prove that the product of S and B is 30.
-/
theorem product_of_sisters_and_brothers (S B : ℕ) (h1 : S = 5) (h2 : B = 6) : S * B = 30 := by
  rw [h1, h2]
  exact Nat.mul_eq_mul_right (show 5 * 6 = 30 from rfl)

end product_of_sisters_and_brothers_l153_153122


namespace hyperbola_eccentricity_l153_153410

variable (a b c : ℝ)
variable (ha : a > 0) (hb : b > 0)
variable (c_def : c = real.sqrt (a^2 + b^2))

theorem hyperbola_eccentricity :
  (∃ x y : ℝ, 
    (x + c = 2*c) ∧ 
    (y = (2 * real.sqrt 3 * c) / 3) ∧ 
    (x^2 / a^2 - y^2 / b^2 = 1) ∧
    (c = real.sqrt (a^2 + b^2)) ∧
    (e = c / a)) → 
  e = real.sqrt 3 :=
by
  sorry

end hyperbola_eccentricity_l153_153410


namespace correct_statement_C_l153_153271

theorem correct_statement_C (α β : ℝ) (h : α + β = 180) :
  let bisector := λθ, θ / 2
  ∀ P, P ∈ bisector α → dist P (line α) = dist P (line β) := 
  sorry

end correct_statement_C_l153_153271


namespace pascal_triangle_contains_47_l153_153066

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153066


namespace total_distance_travelled_l153_153297

theorem total_distance_travelled
  (still_water_speed : ℝ)
  (river_speed : ℝ)
  (round_trip_time : ℝ)
  (effective_upstream_speed : ℝ)
  (effective_downstream_speed : ℝ)
  (distance_to_place : ℝ)
  (t1 t2 : ℝ)
  (t1_eq : t1 = distance_to_place / effective_upstream_speed)
  (t2_eq : t2 = distance_to_place / effective_downstream_speed)
  (total_time_eq : t1 + t2 = round_trip_time) :
  still_water_speed = 6 → river_speed = 3 → round_trip_time = 1 → 
  effective_upstream_speed = still_water_speed - river_speed → 
  effective_downstream_speed = still_water_speed + river_speed → 
  distance_to_place = 2.25 →
  2 * distance_to_place = 4.5 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [←h6]
  norm_num
  -- sorry

end total_distance_travelled_l153_153297


namespace max_difference_evens_l153_153678

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l153_153678


namespace initial_breads_count_l153_153280

theorem initial_breads_count :
  ∃ (X : ℕ), ((((X / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2) / 2 - 1 / 2 = 3 ∧ X = 127 :=
by sorry

end initial_breads_count_l153_153280


namespace arithmetic_seq_sum_l153_153136

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h : a 3 + a 4 + a 5 + a 6 + a 7 = 25) :
  (∑ i in finset.range 9, a i) = 45 :=
begin
  -- Proof is omitted
  sorry
end

end arithmetic_seq_sum_l153_153136


namespace tip_percentage_l153_153150

def julie_food_cost : ℝ := 10
def letitia_food_cost : ℝ := 20
def anton_food_cost : ℝ := 30
def julie_tip : ℝ := 4
def letitia_tip : ℝ := 4
def anton_tip : ℝ := 4

theorem tip_percentage : 
  (julie_tip + letitia_tip + anton_tip) / (julie_food_cost + letitia_food_cost + anton_food_cost) * 100 = 20 :=
by
  sorry

end tip_percentage_l153_153150


namespace find_max_difference_l153_153684

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l153_153684


namespace find_p_l153_153435

theorem find_p (p : ℕ) : 64^5 = 8^p → p = 10 :=
by
  intro h
  sorry

end find_p_l153_153435


namespace first_player_wins_starting_60_l153_153539

theorem first_player_wins_starting_60:
  (∀ n : ℕ, ∃ div, div ∣ n ∧ n > 0 → ∃ m, m = n - div ∧ m = 0 → false)→
  (60 % 2 = 0) → 
  (∃ strategy, strategy(60) = true):=
by
  -- Define what it means for the first player to have a winning strategy.
  let winning_strategy := λ n: ℕ, ∃ m, m = n - (find_important_divisor n) ∧ m: odd

  -- We know the first player is in a winning position starting from 60.
  have h₁ : winning_strategy 60,
  from sorry

  exact h₁

end first_player_wins_starting_60_l153_153539


namespace a_2018_equals_2_pow_2018_minus_1_l153_153775

-- Define the sequence {a_n} and the sum {S_n}
noncomputable def a : ℕ → ℤ 
noncomputable def S : ℕ → ℤ

-- Given the conditions
axiom sum_of_terms (n : ℕ) : 3 * S n = 2 * a n - 3 * n

-- Define the proof problem
theorem a_2018_equals_2_pow_2018_minus_1 : a 2018 = 2^2018 - 1 := 
by
  sorry

end a_2018_equals_2_pow_2018_minus_1_l153_153775


namespace angle_KAL_is_45_degrees_l153_153188

-- Define the square ABCD and points K, L on its sides BC and CD
variables {A B C D K L : Point}
variables (h_sq : square A B C D) (h_K_on_BC : K ∈ seg B C) (h_L_on_CD : L ∈ seg C D)
variables (h_angle_eq : ∠ A K B = ∠ A K L)

-- Prove that ∠ K A L = 45 degrees
theorem angle_KAL_is_45_degrees : ∠ K A L = 45 := 
by sorry

end angle_KAL_is_45_degrees_l153_153188


namespace plane_through_three_points_l153_153726

section
variable {α : Type} [LinearOrderedField α] {p1 p2 p3 : EuclideanSpace α}

/- 
Definition: A plane is determined by points in space.
Two cases:
1. If three points are collinear, an infinite number of planes can be constructed through them.
2. If three points are not collinear, only one plane can be determined through them.

We need to prove:
Given any three points in space, the number of planes that can be constructed through these points is either only one or an infinite number.
-/

theorem plane_through_three_points (p1 p2 p3 : EuclideanSpace α) :
  let collinear (p1 p2 p3 : EuclideanSpace α) := ∃ (l : Line α), p1 ∈ l ∧ p2 ∈ l ∧ p3 ∈ l
  in (collinear p1 p2 p3) ∨ (∃! (plane : EuclideanPlane α), p1 ∈ plane ∧ p2 ∈ plane ∧ p3 ∈ plane) :=
sorry
end

end plane_through_three_points_l153_153726


namespace sara_total_payment_l153_153193

structure DecorationCosts where
  balloons: ℝ
  tablecloths: ℝ
  streamers: ℝ
  banners: ℝ
  confetti: ℝ
  change_received: ℝ

noncomputable def total_cost (c : DecorationCosts) : ℝ :=
  c.balloons + c.tablecloths + c.streamers + c.banners + c.confetti

noncomputable def amount_given (c : DecorationCosts) : ℝ :=
  total_cost c + c.change_received

theorem sara_total_payment : 
  ∀ (costs : DecorationCosts), 
    costs = ⟨3.50, 18.25, 9.10, 14.65, 7.40, 6.38⟩ →
    amount_given costs = 59.28 :=
by
  intros
  sorry

end sara_total_payment_l153_153193


namespace probability_heads_in_12_flips_l153_153999

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153999


namespace pages_written_per_month_l153_153524

theorem pages_written_per_month 
  (d : ℕ) (days_in_month : ℕ) (letters_freq : ℕ) (time_per_letter : ℕ) 
  (time_per_page : ℕ) (long_letter_time_ratio : ℕ) (long_letter_time : ℕ) :
  d = 3 →
  days_in_month = 30 →
  letters_freq = 10 →
  time_per_letter = 20 →
  time_per_page = 10 →
  long_letter_time_ratio = 2 →
  long_letter_time = 80 →
  (days_in_month / letters_freq * time_per_letter / time_per_page) +
  (long_letter_time / (time_per_page * long_letter_time_ratio)) = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6, h7]
  simp
  exact rfl

#eval pages_written_per_month

end pages_written_per_month_l153_153524


namespace percentage_error_in_area_l153_153602

theorem percentage_error_in_area (S : ℝ) :
  let measured_side := 1.17 * S
  let actual_area := S^2
  let calculated_area := measured_side^2
  let error := ((calculated_area - actual_area) / actual_area) * 100
  error ≈ 36.89 := by
  sorry

end percentage_error_in_area_l153_153602


namespace unique_positive_integer_l153_153723

def S (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 2) * 2^(k + 2))

theorem unique_positive_integer (n : ℕ) (h : S n = 2^(n + 11)) : n = 1025 :=
sorry

end unique_positive_integer_l153_153723


namespace value_of_k_l153_153585

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l153_153585


namespace air_conditioner_sales_l153_153197

/-- Represent the conditions -/
def conditions (x y m : ℕ) : Prop :=
  (3 * x + 5 * y = 23500) ∧
  (4 * x + 10 * y = 42000) ∧
  (x = 2500) ∧
  (y = 3200) ∧
  (700 * (50 - m) + 800 * m ≥ 38000)

/-- Prove that the unit selling prices of models A and B are 2500 yuan and 3200 yuan respectively,
    and at least 30 units of model B need to be purchased for a profit of at least 38000 yuan,
    given the conditions. -/
theorem air_conditioner_sales :
  ∃ (x y m : ℕ), conditions x y m ∧ m ≥ 30 := by
  sorry

end air_conditioner_sales_l153_153197


namespace problem_15300_l153_153727

theorem problem_15300 :
  let S := {1, 2, ..., 100}
  let chosen_numbers := (S × S × S)
  let D := chosen_numbers.1
  let K := chosen_numbers.2.1
  let M := chosen_numbers.2.2
  let prob1 := |D - K| < |K - M|
  let prob2 := |D - K| > |K - M|
  let prob_eq := |D - K| = |K - M|
  let total_prob := 1
  let prob_i := fraction of chosen_numbers such that prob_i
  let m n : ℕ
  let h_coprime : m.gcd n = 1
  in (149 * 100 + 400 = 15300) := by
    sorry

end problem_15300_l153_153727


namespace probability_sleep_ge_7h_l153_153443

noncomputable def probability_average_sleep_time (x y : ℝ) : ℝ :=
if x ≥ 6 ∧ x ≤ 9 ∧ y ≥ 6 ∧ y ≤ 9 ∧ x + y ≥ 14 then 1 else 0

noncomputable def probability_event_ge_7hours : ℝ :=
∫ x in 6..9, ∫ y in 6..9, probability_average_sleep_time x y

theorem probability_sleep_ge_7h : probability_event_ge_7hours = (7/9) :=
sorry

end probability_sleep_ge_7h_l153_153443


namespace actual_average_height_l153_153536

theorem actual_average_height 
  (incorrect_avg_height : ℝ)
  (num_students : ℕ)
  (incorrect_height : ℝ)
  (correct_height : ℝ)
  (actual_avg_height : ℝ) :
  incorrect_avg_height = 175 →
  num_students = 20 →
  incorrect_height = 151 →
  correct_height = 111 →
  actual_avg_height = 173 :=
by
  sorry

end actual_average_height_l153_153536


namespace sequence_general_term_l153_153019

variable {R : Type*} [CommRing R]

theorem sequence_general_term (f : R → R)
  (h_nonzero : ∃ x, f x ≠ 0)
  (h_feq : ∀ x y, f (x * y) = x * f y + y * f x)
  (a : ℕ+ → R) 
  (h_a : ∀ n, a (n : ℕ) = f (3^n))
  (h_a1 : a 1 = 3) :
  ∀ n : ℕ+, a n = (n : ℕ) * 3^n := 
by
  sorry

end sequence_general_term_l153_153019


namespace pascal_triangle_contains_prime_l153_153055

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153055


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153075

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153075


namespace chengdu_gdp_scientific_notation_l153_153141

theorem chengdu_gdp_scientific_notation :
  15000 = 1.5 * 10^4 :=
sorry

end chengdu_gdp_scientific_notation_l153_153141


namespace geometric_sequence_a4_l153_153805

variable (a : ℕ → ℝ) (q : ℝ)

-- Conditions
def condition1 : Prop := 3 * a 5 = a 6
def condition2 : Prop := a 2 = 1

-- Question
def question : Prop := a 4 = 9

theorem geometric_sequence_a4 (h1 : condition1 a) (h2 : condition2 a) : question a :=
sorry

end geometric_sequence_a4_l153_153805


namespace cylinder_surface_l153_153492

open EuclideanGeometry

-- Define the points A, B, and C as non-collinear
variables (A B C : Point)
variable (h : ℝ)
variable (P : Point)

-- Define the projections of P onto the lines BC, CA, and AB
variable (P_proj : Point)
axiom proj_on_plane : P_proj ∈ PlaneABC A B C

axiom non_collinear : ¬Collinear A B C
axiom dist_to_plane_not_exceed_h : dist_to_plane P (PlaneABC A B C) ≤ h

-- Define the condition that the projections of P on lines BC, CA, and AB lie on a single straight line
axiom projections_on_line : SameLine (Proj P A B) (Proj P B C) (Proj P C A)

theorem cylinder_surface :
  {P : Point | dist_to_plane P (PlaneABC A B C) ≤ h ∧ projections_on_line (Proj P A B) (Proj P B C) (Proj P C A)}
  = {P : Point | CylinderSurface P A B C h} :=
sorry

end cylinder_surface_l153_153492


namespace pascal_triangle_contains_prime_l153_153056

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153056


namespace number_of_zeros_correct_l153_153161

def sequence_condition (a : Fin 50 → ℤ) : Prop :=
  ∀ i, a i ∈ {-1, 0, 1}

def sum_condition (a : Fin 50 → ℤ) : Prop :=
  ∑ i, a i = 9

def sum_of_squares_condition (a : Fin 50 → ℤ) : Prop :=
  ∑ i, (a i + 1) * (a i + 1) = 107

def count_zeros (a : Fin 50 → ℤ) : ℕ :=
  (Finset.univ.filter (λ i, a i = 0)).card

theorem number_of_zeros_correct (a : Fin 50 → ℤ) :
  sequence_condition a →
  sum_condition a →
  sum_of_squares_condition a →
  count_zeros a = 11 :=
by
  intros hseq hsum hsquares
  -- proof goes here
  sorry

end number_of_zeros_correct_l153_153161


namespace area_is_1_l153_153017

noncomputable def isosceles_right_triangle_area
  (a b : Type*) [InnerProductSpace ℝ a b]
  : \(\triangle ABC\) : Type*
  := sorry

theorem area_is_1
  (a b : ℝ × ℝ)
  (θ : ℝ)
  (h₁ : \(\triangle ABC\; is\;isosceles\;right\;triangle\) with \(\angle A = 90^\circ\))
  (h₂ : \(\overrightarrow{AB} = a + b\))
  (h₃ : \(\overrightarrow{AC} = a - b\))
  (h₄ : \(a = (\cos \theta, \sin \theta)\))
  : area \(\triangle ABC\) = 1 := sorry

end area_is_1_l153_153017


namespace length_of_AB_proof_l153_153807

noncomputable def length_of_AB (x y : ℝ) (θ : ℝ) : ℝ :=
  if h : (∃ A B : ℝ × ℝ, A = (2 - 3 * Real.cos θ, 1 + 3 * Real.sin θ) ∧
                           B = (2 - 3 * Real.cos θ, 1 + 3 * Real.sin θ) ∧
                           (2 - 3 * Real.cos θ) + 2 * (1 + 3 * Real.sin θ) - 4 = 0 ∧
                           (2 - 3 * Real.cos θ) + 2 * (1 + 3 * Real.sin θ) - 4 = 0) then 6 else 0

theorem length_of_AB_proof :
  ∀ θ, ((∃ A B : ℝ × ℝ, A = (2 - 3 * Real.cos θ, 1 + 3 * Real.sin θ) ∧
                          B = (2 - 3 * Real.cos θ, 1 + 3 * Real.sin θ) ∧
                          (2 - 3 * Real.cos θ) + 2 * (1 + 3 * Real.sin θ) - 4 = 0 ∧
                          (2 - 3 * Real.cos θ) + 2 * (1 + 3 * Real.sin θ) - 4 = 0) → length_of_AB 2 1 θ = 6) :=
by
  -- Include a proof using the necessary Lean tactics
  sorry

end length_of_AB_proof_l153_153807


namespace distance_between_cities_l153_153562

noncomputable def speed_a : ℝ := 1 / 10
noncomputable def speed_b : ℝ := 1 / 15
noncomputable def time_to_meet : ℝ := 6
noncomputable def distance_diff : ℝ := 12

theorem distance_between_cities : 
  (time_to_meet * (speed_a + speed_b) = 60) →
  time_to_meet * speed_a - time_to_meet * speed_b = distance_diff →
  time_to_meet * (speed_a + speed_b) = 60 :=
by
  intros h1 h2
  sorry

end distance_between_cities_l153_153562


namespace possible_angles_of_triangle_l153_153866

theorem possible_angles_of_triangle
  (ABC : Type) [NormedAddCommGroup ABC] [InnerProductSpace ℝ ABC]
  (O H A B C : ABC)
  (h_circumcenter : ∀ P : ABC, dist O P = dist O A ∨ dist O P = dist O B ∨ dist O P = dist O C)
  (h_orthocenter : is_orthocenter_of_triangle H A B C)
  (h_eq_dist : dist B O = dist B H) :
  ∠A B C = 60 ∨ ∠A B C = 120 :=
by
  sorry

end possible_angles_of_triangle_l153_153866


namespace simplify_expression_l153_153900

theorem simplify_expression (x : ℝ) (h : x^2 + 2 * x - 6 = 0) : 
  ((x - 1) / (x - 3) - (x + 1) / x) / ((x^2 + 3 * x) / (x^2 - 6 * x + 9)) = -1/2 := 
by
  sorry

end simplify_expression_l153_153900


namespace P_collinear_with_circumcenters_l153_153844

open EuclideanGeometry

noncomputable def problem_statement : Prop :=
  ∀ (A B C D E P : Point)
  (h1 : ConvexPentagon A B C D E)
  (h2 : Intersect BD CE P)
  (h3 : ∠PAD = ∠ACB)
  (h4 : ∠CAP = ∠EDA),
  Collinear P (Circumcenter (Triangle A B C)) (Circumcenter (Triangle A D E))

-- We Define Points A, B, C, D, E, and P
def A : Point := Point.mk 0 0
def B : Point := Point.mk 1 0
def C : Point := Point.mk 0 1
def D : Point := Point.mk 1 1
def E : Point := Point.mk 0.5 0.5
def P : Point := Point.mk 0.5 0.5

-- Theorem stating point P lies on the line connecting the circumcenters of ΔABC and ΔADE
theorem P_collinear_with_circumcenters :
  (problem_statement A B C D E P) :=
begin
  sorry
end

end P_collinear_with_circumcenters_l153_153844


namespace determine_a_b_l153_153414

variable (a b : ℝ)
def f (x : ℝ) : ℝ := real.log x - a * x^2 - b * x
def f_prime (x : ℝ) : ℝ := (1 / x) - 2 * a * x - b

theorem determine_a_b 
  (h_extreme : f a b 1 = 0) 
  (h_deriv : f_prime a b 1 = 0) : 
  a = 1 ∧ b = -1 := by
  sorry

end determine_a_b_l153_153414


namespace pascal_row_contains_prime_47_l153_153053

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153053


namespace volume_of_rotated_region_eq_3pi_over_10_l153_153714

open Real

noncomputable def volume_of_solid : ℝ :=
  let y1 := λ x : ℝ, x^2
  let y2 := λ x : ℝ, x^(1/2)
  π * (∫ x in 0..1, y1 x^2) - π * (∫ x in 0..1, y2 x^4)

theorem volume_of_rotated_region_eq_3pi_over_10 : volume_of_solid = (3 * π) / 10 :=
  sorry

end volume_of_rotated_region_eq_3pi_over_10_l153_153714


namespace ratio_of_side_lengths_l153_153691

theorem ratio_of_side_lengths (t p : ℕ) (h1 : 3 * t = 30) (h2 : 5 * p = 30) : t / p = 5 / 3 :=
by
  sorry

end ratio_of_side_lengths_l153_153691


namespace students_not_enrolled_in_either_l153_153120

-- Definitions based on conditions
def total_students : ℕ := 120
def french_students : ℕ := 65
def german_students : ℕ := 50
def both_courses_students : ℕ := 25

-- The proof statement
theorem students_not_enrolled_in_either : total_students - (french_students + german_students - both_courses_students) = 30 := by
  sorry

end students_not_enrolled_in_either_l153_153120


namespace concyclic_iff_angle_eq_l153_153851

variables {A B C D E P Q M : Type} [Geometry A B C D E P Q M]

/-- In triangle ABC, M is the midpoint of AB, P is an 
    interior point of triangle ABC, and Q is the reflection of P across M.
    Let D be the intersection point of AP with BC and 
    E be the intersection point of BP with AC. Then A, B, D, and E 
    are concyclic if and only if ∠ACP = ∠QCB. -/
theorem concyclic_iff_angle_eq :
  (midpoint M A B) → 
  (interior P A B C) →
  (reflection P Q M) → 
  (intersect_linecircle A P B C D) →
  (intersect_linecircle B P A C E) →
  (concyclic A B D E ↔ ∠ A C P = ∠ Q C B) := 
begin
  -- Proof omitted
  sorry
end

end concyclic_iff_angle_eq_l153_153851


namespace max_regions_two_convex_polygons_l153_153563

theorem max_regions_two_convex_polygons (M N : ℕ) (hM : M > N) :
    ∃ R, R = 2 * N + 2 := 
sorry

end max_regions_two_convex_polygons_l153_153563


namespace max_difference_evens_l153_153676

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l153_153676


namespace victor_final_amount_is_874_l153_153564

noncomputable def final_amount (initial_rubles : ℝ) (usd_rate : ℝ) (annual_rate : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let principal := initial_rubles / usd_rate
  let quarterly_rate := 1 + annual_rate / n.to_real
  principal * quarterly_rate ^ (n * t).to_real

theorem victor_final_amount_is_874 :
  final_amount 45000 56.60 0.047 4 2 ≈ 874 :=
by
  sorry

end victor_final_amount_is_874_l153_153564


namespace elmwood_earnings_l153_153195

theorem elmwood_earnings (a b c d e f total_payment : ℝ) (students_Elmwood days_Elmwood total_student_days : ℕ) :
  a = 6 * 2 →
  b = 5 * 6 →
  c = 7 * 8 →
  d = 3 * 4 →
  e = 8 * 3 →
  f = 4 * 7 →
  total_payment = 1494 →
  students_Elmwood = 8 →
  days_Elmwood = 3 →
  total_student_days = a + b + c + d + e + f →
  let daily_wage := total_payment / total_student_days in  
  let elmwood_student_days := students_Elmwood * days_Elmwood in  
  let elmwood_earning := daily_wage * elmwood_student_days in  
  elmwood_earning = 221.33 := 
begin
  sorry
end

end elmwood_earnings_l153_153195


namespace minimum_value_quadratic_expression_l153_153356

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end minimum_value_quadratic_expression_l153_153356


namespace niles_win_value_l153_153711

-- Define the problem conditions
def billie_die : List ℕ := [1, 2, 3, 4, 5, 6]
def niles_die : List ℕ := [4, 4, 4, 5, 5, 5]

-- Define the probability that Niles wins given the conditions
noncomputable def probability_niles_wins : ℚ :=
  ((3 / 6) * (3 / 6)) + ((3 / 6) * (4 / 6))

-- Statement of the theorem to prove
theorem niles_win_value :
  let p := 7
  let q := 12
  7 * p + 11 * q = 181 := by
  sorry

end niles_win_value_l153_153711


namespace sin_double_angle_l153_153098

theorem sin_double_angle (θ : ℝ) (h : Real.tan θ + 1 / Real.tan θ = 4) : Real.sin (2 * θ) = 1 / 2 :=
by
  sorry

end sin_double_angle_l153_153098


namespace sum_of_abc_l153_153219

theorem sum_of_abc (h : (∃ a b c : ℕ, a * real.sqrt b = 5 * real.sqrt 5 ∧ c = 7 ∧ a + b + c = 37)) : 
  ∃ a b c : ℕ, 
  a * real.sqrt b / c = (250 / 98)^(1/2) ∧
  a + b + c = 37 :=
by
  sorry

end sum_of_abc_l153_153219


namespace part_I_part_II_l153_153417

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  x * (log x + a * x + 1) - a * x + 1

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ :=
  log x + 2 * a * x + 2 - a

theorem part_I (a : ℝ) : (∀ x : ℝ, 1 ≤ x → f_prime x a ≤ 0) → a ≤ -2 :=
by sorry

theorem part_II (a : ℝ) : (∃ x : ℝ, f x a = 2 ∧ f_prime 1 a = 0) → a = -2 :=
by sorry

end part_I_part_II_l153_153417


namespace pascal_triangle_47_l153_153092

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153092


namespace circle_distance_formula_l153_153917

variable (R r d t m x y : ℝ)

-- Conditions
def cond1 := true -- Placeholder for the assumption of existence of the circles
def cond2 := x = (R * t) / d
def cond3 := y = (r * t) / d
def cond4 := x + y = m
def cond5 := t^2 = m^2 + 4 * x * y
def cond6 := d^2 = (R + r)^2 + t^2

-- Proof statement
theorem circle_distance_formula (h1 : cond1) (h2 : cond2) (h3 : cond3) (h4 : cond4) (h5 : cond5) (h6 : cond6) :
  d^2 = (R + r)^2 + 4 * R * r :=
by
  sorry

end circle_distance_formula_l153_153917


namespace polar_to_cartesian_l153_153037

theorem polar_to_cartesian (ρ θ : ℝ) (h : ρ = 2 * Real.cos θ) :
  (λ x y : ℝ, x^2 + y^2 - 2 * x = 0) (ρ * Real.cos θ) (ρ * Real.sin θ) :=
by sorry

end polar_to_cartesian_l153_153037


namespace thirteen_factorial_mod_seventeen_l153_153366

-- Define factorial function
def factorial : ℕ → ℕ
| 0 => 1
| (n+1) => (n+1) * factorial n

-- Wilson's theorem as a lemma
lemma wilson_theorem {p : ℕ} (hp : Nat.Prime p) : (p - 1)! ≡ -1 [MOD p] :=
  sorry

-- Main theorem statement
theorem thirteen_factorial_mod_seventeen : factorial 13 % 17 = 9 :=
  by
  -- Use Wilson's theorem for p = 17
  have wilson := wilson_theorem Nat.prime_seventeen
  have fact16 := factorial 16
  have step1 : fact16 % 17 = 16! % 17 := sorry
  have step2 : 16! % 17 = (16 * 15 * 14 * factorial 13) % 17 := sorry
  have step3 : (16 * 15 * 14) % 17 = -6 % 17 := sorry
  have step4 : -6 % 17 = 11 := sorry
  have step5 : (11 * factorial 13) % 17 ≡ -1 [MOD 17] by
    -- detailed steps to compute this
    sorry
  have step6 : factorial 13 ≡ 9 [MOD 17] by
    -- detailed steps to prove this equivalence
    sorry
  exact sorry -- needed to complete the theorem

end thirteen_factorial_mod_seventeen_l153_153366


namespace not_prime_sum_abcd_not_prime_sum_square_abcd_l153_153517

theorem not_prime_sum_abcd
  (a b c d : ℕ) (h : a * b = c * d) : ¬ prime (a + b + c + d) :=
sorry

theorem not_prime_sum_square_abcd
  (a b c d : ℕ) (h : a * b = c * d) : ¬ prime (a^2 + b^2 + c^2 + d^2) :=
sorry

end not_prime_sum_abcd_not_prime_sum_square_abcd_l153_153517


namespace average_age_7_people_l153_153534

-- Define the conditions
def average_age_8_people : ℝ := 35
def number_of_people_initially : ℕ := 8
def age_of_leaving_person : ℝ := 22

-- Calculate the average age of the remaining 7 people
theorem average_age_7_people :
  let total_initial_age := average_age_8_people * number_of_people_initially in
  let total_new_age := total_initial_age - age_of_leaving_person in
  let number_of_people_remaining := 7 in
  (total_new_age / number_of_people_remaining) = 258 / 7 :=
by sorry

end average_age_7_people_l153_153534


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153984

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153984


namespace symmetric_about_origin_l153_153688

theorem symmetric_about_origin :
  (∀ x : ℝ, - x ≠ 0 → -x = sin x) :=
by sorry

end symmetric_about_origin_l153_153688


namespace abs_neg_five_halves_l153_153911

theorem abs_neg_five_halves : abs (-5 / 2) = 5 / 2 := by
  sorry

end abs_neg_five_halves_l153_153911


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153992

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153992


namespace road_system_possible_a_road_system_impossible_b_l153_153467

-- Define a city as an element of a finite type of 16 elements
inductive City : Type
| city0 | city1 | city2 | city3 | city4 | city5 | city6 | city7
| city8 | city9 | city10 | city11 | city12 | city13 | city14 | city15

open City

-- Define the road system as a set of pairs of cities
def RoadSystem := set (City × City)

-- Define the problem for Part (a)
theorem road_system_possible_a 
  (C : finset City) (hC : C.card = 16) (R : RoadSystem)
  (hR₁ : ∀ c ∈ C, finset.filter (λ p : City × City, p.1 = c ∨ p.2 = c) R).card ≤ 5)
  (hR₂ : ∀ c1 c2 ∈ C, c1 ≠ c2 → ∃ p ∈ R, (p.1 = c1 ∧ p.2 = c2) ∨ (∃ c3 ∈ C, c3 ≠ c1 ∧ c3 ≠ c2 ∧ (p.1 = c1 ∧ p.2 = c3 ∨ p.1 = c3 ∧ p.2 = c2)))
  : ∃ R' : RoadSystem, ∀ c ∈ C, finset.filter (λ p : City × City, p.1 = c ∨ p.2 = c) R').card ≤ 5 ∧ ∀ c1 c2 ∈ C, c1 ≠ c2 → ∃ p ∈ R', (p.1 = c1 ∧ p.2 = c2) ∨ (∃ c3 ∈ C, c3 ≠ c1 ∧ c3 ≠ c2 ∧ (p.1 = c1 ∧ p.2 = c3 ∨ p.1 = c3 ∧ p.2 = c2))
:= sorry

-- Define the problem for Part (b)
theorem road_system_impossible_b 
  (C : finset City) (hC : C.card = 16) (R : RoadSystem)
  (hR₁ : ∀ c ∈ C, finset.filter (λ p : City × City, p.1 = c ∨ p.2 = c) R).card ≤ 4)
  (hR₂ : ∀ c1 c2 ∈ C, c1 ≠ c2 → ∃ p ∈ R, (p.1 = c1 ∧ p.2 = c2) ∨ (∃ c3 ∈ C, c3 ≠ c1 ∧ c3 ≠ c2 ∧ (p.1 = c1 ∧ p.2 = c3 ∨ p.1 = c3 ∧ p.2 = c2)))
  : ¬ (∃ R' : RoadSystem, ∀ c ∈ C, finset.filter (λ p : City × City, p.1 = c ∨ p.2 = c) R').card ≤ 4 ∧ ∀ c1 c2 ∈ C, c1 ≠ c2 → ∃ p ∈ R', (p.1 = c1 ∧ p.2 = c2) ∨ (∃ c3 ∈ C, c3 ≠ c1 ∧ c3 ≠ c2 ∧ (p.1 = c1 ∧ p.2 = c3 ∨ p.1 = c3 ∧ p.2 = c2)))
:= sorry

end road_system_possible_a_road_system_impossible_b_l153_153467


namespace guess_grandfathers_age_l153_153274

/-- Definition of the age difference problem -/
def age_difference_criteria(g_age x_age : ℕ) : Prop :=
  g_age > 7 * x_age ∧
  (g_age - x_age) % 6 = 0 ∧ 
  (g_age - x_age) % 5 = 0 ∧
  (g_age - x_age) % 4 = 0

theorem guess_grandfathers_age (g_age : ℕ) : g_age = 69 :=
by
  let x_age := g_age - 60 in
  have h1 : age_difference_criteria g_age x_age,
  {
    sorry -- fill in proof steps
  },
  exact h1

end guess_grandfathers_age_l153_153274


namespace min_value_solution_l153_153490

noncomputable def minimum_value (p q r s t u : ℝ) (h_pos : 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u) (h_sum : p + q + r + s + t + u = 10) : ℝ :=
  Inf {x | ∃ (p q r s t u : ℝ), 0 < p ∧ 0 < q ∧ 0 < r ∧ 0 < s ∧ 0 < t ∧ 0 < u ∧ 
  p + q + r + s + t + u = 10 ∧ (2 / p) + (3 / q) + (5 / r) + (7 / s) + (11 / t) + (13 / u) = x}

theorem min_value_solution : minimum_value = 23.875 :=
  by sorry

end min_value_solution_l153_153490


namespace smallest_number_last_four_digits_l153_153874

def is_divisible (n d : ℕ) : Prop := ∃ k : ℕ, n = k * d

def valid_digits (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ (nat.digits 10 n) → d = 5 ∨ d = 9

def contains_5_and_9 (n : ℕ) : Prop :=
  ∃ (l : List ℕ), l = (nat.digits 10 n) ∧ 5 ∈ l ∧ 9 ∈ l

def last_four_digits (n : ℕ) : ℕ :=
  n % 10000

theorem smallest_number_last_four_digits : 
  ∃ n : ℕ, 
    is_divisible n 5 ∧ 
    is_divisible n 9 ∧ 
    valid_digits n ∧ 
    contains_5_and_9 n ∧ 
    (∀ m : ℕ, 
      is_divisible m 5 ∧ 
      is_divisible m 9 ∧ 
      valid_digits m ∧ 
      contains_5_and_9 m →
      n ≤ m) ∧ 
    last_four_digits n = 9995 :=
sorry

end smallest_number_last_four_digits_l153_153874


namespace smallest_balanced_number_l153_153650

theorem smallest_balanced_number :
  ∃ (a b c : ℕ), 
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  100 * a + 10 * b + c = 
  (10 * a + b) + (10 * b + a) + (10 * a + c) + (10 * c + a) + (10 * b + c) + (10 * c + b) ∧ 
  100 * a + 10 * b + c = 132 :=
sorry

end smallest_balanced_number_l153_153650


namespace part1_part2_l153_153393

noncomputable def A (x : ℝ) : Prop := x < 0 ∨ x > 2
noncomputable def B (a x : ℝ) : Prop := a ≤ x ∧ x ≤ 3 - 2 * a

-- Part (1)
theorem part1 (a : ℝ) : 
  (∀ x : ℝ, A x ∨ B a x) ↔ (a ≤ 0) := 
sorry

-- Part (2)
theorem part2 (a : ℝ) : 
  (∀ x : ℝ, B a x → (0 ≤ x ∧ x ≤ 2)) ↔ (1 / 2 ≤ a) :=
sorry

end part1_part2_l153_153393


namespace problem_l153_153815

open Set

def M : Set ℝ := { x | -1 ≤ x ∧ x < 3 }
def N : Set ℝ := { x | x < 0 }
def complement_N : Set ℝ := { x | x ≥ 0 }

theorem problem : M ∩ complement_N = { x | 0 ≤ x ∧ x < 3 } :=
by
  sorry

end problem_l153_153815


namespace problem_l153_153808

variables {R : Type*} [field R]

def N (x y z : R) : Matrix (fin 3) (fin 3) R :=
  !![0, x, y;
    z, 0, -x;
    z, y, 0]

noncomputable def N_transpose (x y z : R) : Matrix (fin 3) (fin 3) R :=
  Matrix.transpose (N x y z)

theorem problem (x y z : ℝ) (h : N_transpose x y z ⬝ N x y z = 1) :
  x^2 + y^2 + z^2 = 3 / 2 :=
by sorry

end problem_l153_153808


namespace inequality_true_l153_153022

noncomputable def f : ℝ → ℝ := sorry

def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x ≤ b ∧ a ≤ y ∧ y ≤ b ∧ x < y → f x < f y

variables (h_even : even_function f) (h_increasing : increasing_on f 0 5)

theorem inequality_true : f 4 > f (-real.pi) ∧ f (-real.pi) > f 3 :=
by {
  sorry
}

end inequality_true_l153_153022


namespace work_done_in_isothermal_is_equal_to_heat_added_l153_153182

-- First, define the ideal gas and process conditions
def n : ℕ := 1                      -- Number of moles (one mole)
def W₁ : ℝ := 30                    -- Work done in the first process (Joules)
def R : ℝ := 8.314                  -- Ideal gas constant (J/(mol·K))
def P : ℝ := 101325                -- An arbitrary pressure for the ideal gas (Pa)

-- State the first and second law of thermodynamics conditions
def ∆V : ℝ := 0.000295              -- Change in volume, derived arbitrarily, not provided in problem
def ∆T : ℝ := W₁ / P                -- Change of temperature calculated from the ideal gas law relation for isobaric process

def Q₁ : ℝ := W₁ + (3 / 2) * n * R * ∆T  -- Heat added in the first process

-- State the equality for the second process
theorem work_done_in_isothermal_is_equal_to_heat_added : Q₁ = 75 →  W₂ = 75 :=
by
  sorry

end work_done_in_isothermal_is_equal_to_heat_added_l153_153182


namespace triangle_ec_length_l153_153398

theorem triangle_ec_length (A B C D E : Type) [metric_space A] [metric_space B] [metric_space C]
  [metric_space D] [metric_space E] (angle_ABC : ℝ) (BC_length : ℝ) (BD_perp_AC : Prop)
  (CE_perp_AB : Prop) (DBC_angle : ℝ) (ECB_angle : ℝ)
  (h1 : angle_ABC = 60) 
  (h2 : BC_length = 10)
  (h3 : BD_perp_AC)
  (h4 : CE_perp_AB)
  (h5 : DBC_angle = 4 * ECB_angle) :
  ∃ (a b c : ℝ), a = 5 ∧ b = 1 ∧ c = 0 ∧ (a + b + c) = 6 :=
begin
  sorry
end

end triangle_ec_length_l153_153398


namespace probability_at_least_9_heads_l153_153981

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153981


namespace method_of_completing_square_reflects_transformation_l153_153612

noncomputable def quadratic_completion_reflects_transformation (a b c : ℝ) (h : a ≠ 0) : Prop :=
  let quadratic_eq := λ x : ℝ, a * x^2 + b * x + c = 0
  let transformed_eq := λ x m n : ℝ, (x + m)^2 = n ∧ n ≥ 0
  ∀ x m n, quadratic_eq x → transformed_eq x m n

-- To be proved later
theorem method_of_completing_square_reflects_transformation (a b c : ℝ) (h : a ≠ 0) :
  quadratic_completion_reflects_transformation a b c h :=
sorry

end method_of_completing_square_reflects_transformation_l153_153612


namespace simplify_sqrt_expression_l153_153518

-- Define the conditions
def expr1 := Real.sqrt (5 * 3) * Real.sqrt (3^5 * 5^2)
def result := 135 * Real.sqrt 5

-- The theorem to prove the question is correct
theorem simplify_sqrt_expression : expr1 = result :=
by
  -- Skipping the actual proof with 'sorry'
  sorry

end simplify_sqrt_expression_l153_153518


namespace centroid_locus_circle_l153_153771

theorem centroid_locus_circle 
  (A B C : Point)
  (hne : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (n_equi : ¬ (∃ R, dist A B = R ∧ dist B C = R ∧ dist C A = R))
  : ∃ L : Point, 
      ∀ (A' B' C' : Point),
        (equilateral A' B' C') → 
        (collinear A B' C') →
        (collinear A' B C') → 
        (collinear A' B' C) → 
        circle (centroid A' B' C') L := 
sorry

end centroid_locus_circle_l153_153771


namespace rectangular_prism_area_l153_153847

theorem rectangular_prism_area
  (AB AD AA1 : ℝ)
  (h_AB : AB = 6)
  (h_AD : AD = 4)
  (h_AA1 : AA1 = 3)
  (V1 V2 V3 : ℝ)
  (h_V1 : V1 = 12)
  (h_V2 : V2 = 48)
  (h_V3 : V3 = 12)
  (h_ratio : V1 / V2 = 1 / 4 ∧ V1 = V3) :
  let A1E := real.sqrt (2^2 + 3^2),
      A1D1 := AD in
  (A1E * A1D1) = 4 * real.sqrt 13 := by
  sorry

end rectangular_prism_area_l153_153847


namespace tangent_lines_equal_length_l153_153704

open EuclideanGeometry

theorem tangent_lines_equal_length
  (Γ1 Γ2 : Circle) 
  (A B C D E F M G : Point) 
  (tangent1 : Tangent Γ1 A) 
  (tangent2 : Tangent Γ2 B) 
  (internal_tangent1 : Tangent Γ1 C) 
  (internal_tangent2 : Tangent Γ2 D)
  (intsec_E : IntersectsLines (line AC) (line BD) E)
  (F_on_Γ1 : OnCircle F Γ1)
  (tangent_F_Γ1 : TangentFromPoint F Γ1)
  (M_on_perp_bisector_EF : OnPerpBisector M (segment EF))
  (tangent_G_Γ2_from_M : TangentFromPoint M Γ2 G) :
  distance M F = distance M G := by
  sorry

end tangent_lines_equal_length_l153_153704


namespace probability_heads_at_least_9_l153_153969

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153969


namespace max_diff_six_digit_even_numbers_l153_153670

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l153_153670


namespace prove_sides_of_inscribed_rectangle_l153_153640

noncomputable def sides_of_inscribed_rectangle (sides_of_triangle : ℝ × ℝ × ℝ) (perimeter_of_rectangle : ℝ) : ℝ × ℝ :=
  let (a, b, c) := sides_of_triangle
  in (72 / 13, 84 / 13) -- 72/13 = 5 + 7/13, 84/13 = 6 + 6/13

theorem prove_sides_of_inscribed_rectangle :
  sides_of_inscribed_rectangle (21, 10, 17) 24 = (72 / 13, 84 / 13) :=
  sorry

end prove_sides_of_inscribed_rectangle_l153_153640


namespace smallest_number_divisible_conditions_l153_153299

theorem smallest_number_divisible_conditions :
  ∃ N : ℕ, 
  (∃ a b : ℕ, N = 10 * a + b ∧ b < 10 ∧ (a % 10 = 0) ∧ (a / 10) % 20 = 0) ∧
  (∃ k : ℕ, ∃ M : ℕ, N = k * 10^(Nat.log10 N) + M ∧ M % 21 = 0) ∧
  (Nat.digit_at (Nat.log10 N - 1) N ≠ 0) ∧
  (∀ N', 
    ( (∃ a' b' : ℕ, N' = 10 * a' + b' ∧ b' < 10 ∧ (a' % 10 = 0) ∧ (a' / 10) % 20 = 0) ∧
      (∃ k' M' : ℕ, N' = k' * 10^(Nat.log10 N') + M' ∧ M' % 21 = 0) ∧
      (Nat.digit_at (Nat.log10 N' - 1) N' ≠ 0) ) → N ≤ N'
  ) ∧ N = 1609 :=
sorry

end smallest_number_divisible_conditions_l153_153299


namespace z_real_iff_z_complex_iff_z_purely_imaginary_iff_z_third_quadrant_iff_l153_153760

-- Definitions for the problem
def z (m : ℝ) : ℂ := (m^2 - 5 * m + 6 : ℝ) + (m^2 - 3 * m : ℝ)*I

-- (1) z is real if and only if m = 0 or m = 3
theorem z_real_iff (m : ℝ) : (z m).im = 0 ↔ m = 0 ∨ m = 3 := 
sorry

-- (2) z is complex if and only if m ≠ 0 and m ≠ 3
theorem z_complex_iff (m : ℝ) : (z m).im ≠ 0 ↔ m ≠ 0 ∧ m ≠ 3 := 
sorry

-- (3) z is purely imaginary if and only if m = 2
theorem z_purely_imaginary_iff (m : ℝ) : (z m).re = 0 ∧ (z m).im ≠ 0 ↔ m = 2 := 
sorry

-- (4) z is in the third quadrant if and only if 2 < m < 3
theorem z_third_quadrant_iff (m : ℝ) : (z m).re < 0 ∧ (z m).im < 0 ↔ 2 < m ∧ m < 3 := 
sorry

end z_real_iff_z_complex_iff_z_purely_imaginary_iff_z_third_quadrant_iff_l153_153760


namespace complement_intersection_M_N_l153_153425

def M (x : ℝ) : Prop := ∃ y : ℝ, y = sqrt (3 * x - 1)
def N (x : ℝ) : Prop := ∃ y : ℝ, y = log (x - 2 * x^2)

theorem complement_intersection_M_N :
  (∁ (set_of M ∩ set_of N)) = (set.Iio (1 / 3) ∪ set.Ici (1 / 2)) :=
by
  sorry

end complement_intersection_M_N_l153_153425


namespace general_formula_smallest_positive_integer_l153_153379

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  (∀ n, a (n + 1) = a n * 2) ∧ (∀ m n, m ≤ n → a m ≤ a n)

def satisfies_conditions (a : ℕ → ℝ) :=
  a 2 + a 3 + a 4 = 28 ∧ a 3 + 2 = (a 2 + a 4) / 2

def b (a : ℕ → ℝ) (n : ℕ) : ℝ := a n * (Real.log (a n) / Real.log (1 / 2))

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  ∑ i in finset.range n, b a (i + 1)

theorem general_formula (a : ℕ → ℝ) (h : geometric_sequence a) (hc : satisfies_conditions a) :
  ∀ n, a n = 2 ^ n :=
sorry

theorem smallest_positive_integer (a : ℕ → ℝ) (h : geometric_sequence a) (hc : satisfies_conditions a) :
  (∃ n : ℕ, S a n + n * 2^(n+1) > 50 ∧ ∀ m < n, S a m + m * 2^(m+1) ≤ 50) ∧
  ∃ n, n = 5 :=
sorry

end general_formula_smallest_positive_integer_l153_153379


namespace math_problem_l153_153189

theorem math_problem (x y z : ℤ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : (x + y * Complex.I)^2 - 46 * Complex.I = z) :
  x + y + z = 552 :=
by
  sorry

end math_problem_l153_153189


namespace CousinReadingTime_l153_153500

-- Define key variables and conditions
variables (CousinReadsFast : ℕ) (MyReadingTimeHrs : ℕ)

-- Given conditions based on problem statement
def MyReadingTimeMins := MyReadingTimeHrs * 60
def cousinSpeedFactor := 5
def lengthMultiplier := 1.5

-- Define cousin's reading time in minutes
def CousinReadingTimeMins : ℕ := ((MyReadingTimeMins * lengthMultiplier) / cousinSpeedFactor).toNat

-- State the theorem to be proven
theorem CousinReadingTime : CousinReadsFast = CousinReadingTimeMins := by
  sorry

#eval CousinReadingTime 54 3  -- Evaluates the theorem with given conditions

end CousinReadingTime_l153_153500


namespace expansion_solution_l153_153140

theorem expansion_solution (n : ℕ) (S : ℕ) (h₁ : S = 64) (h₂ : (\(\sqrt{x} + \frac{1}{x}\))^n) :
  n = 6 ∧ constant_term ((\(\sqrt{x} + \frac{1}{x}\))^n) = 15 := 
by
  sorry

end expansion_solution_l153_153140


namespace triangle_side_AC_l153_153470

theorem triangle_side_AC (B : Real) (BC AB : Real) (AC : Real) (hB : B = 30 * Real.pi / 180) (hBC : BC = 2) (hAB : AB = Real.sqrt 3) : AC = 1 :=
by
  sorry

end triangle_side_AC_l153_153470


namespace part1_part2_l153_153349

noncomputable def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

theorem part1 (a b : ℝ) (h₀ : a = 1) (h₁ : b = 2) :
  {x : ℝ | f x a b ≤ 5} = {x : ℝ | -3 ≤ x ∧ x ≤ 2} :=
by
  sorry

theorem part2 (a b : ℝ) (h_min_value : ∀ x : ℝ, f x a b ≥ 3) :
  a + b = 3 → (a > 0 ∧ b > 0) →
  (∃ a b : ℝ, a = b ∧ a + b = 3 ∧ (a = b → f x a b = 3)) →
  (∀ a b : ℝ, (a^2/b + b^2/a) ≥ 3) :=
by
  sorry

end part1_part2_l153_153349


namespace age_ratio_five_years_later_l153_153433

theorem age_ratio_five_years_later (my_age : ℕ) (son_age : ℕ) (h1 : my_age = 45) (h2 : son_age = 15) :
  (my_age + 5) / gcd (my_age + 5) (son_age + 5) = 5 ∧ (son_age + 5) / gcd (my_age + 5) (son_age + 5) = 2 :=
by
  sorry

end age_ratio_five_years_later_l153_153433


namespace additional_amount_needed_l153_153148

-- Definitions of the conditions
def shampoo_cost : ℝ := 10.00
def conditioner_cost : ℝ := 10.00
def lotion_cost : ℝ := 6.00
def lotions_count : ℕ := 3
def free_shipping_threshold : ℝ := 50.00

-- Calculating the total amount spent
def total_spent : ℝ :=
  shampoo_cost + conditioner_cost + lotions_count * lotion_cost

-- Required statement for the proof
theorem additional_amount_needed : 
  total_spent + 12.00 = free_shipping_threshold :=
by 
  -- Proof will be here
  sorry

end additional_amount_needed_l153_153148


namespace elastic_collision_ball_speed_l153_153298

open Real

noncomputable def final_ball_speed (v_car v_ball : ℝ) : ℝ :=
  let relative_speed := v_ball + v_car
  relative_speed + v_car

theorem elastic_collision_ball_speed :
  let v_car := 5
  let v_ball := 6
  final_ball_speed v_car v_ball = 16 := 
by
  sorry

end elastic_collision_ball_speed_l153_153298


namespace pascal_triangle_contains_47_l153_153065

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153065


namespace sum_of_digits_of_2010_l153_153498

noncomputable def sum_of_base6_digits (n : ℕ) : ℕ :=
  (n.digits 6).sum

theorem sum_of_digits_of_2010 : sum_of_base6_digits 2010 = 10 := by
  sorry

end sum_of_digits_of_2010_l153_153498


namespace range_of_k_l153_153796

noncomputable def f : ℝ → ℝ
| x => if x > 2 then 2^x - 4 else real.sqrt (-x^2 + 2*x)

noncomputable def F (f : ℝ → ℝ) (k : ℝ) : ℝ → ℝ := 
  λ x, f x - k * x - 3 * k

theorem range_of_k (F : ℝ → ℝ) : 
  (∀ k, (∃ a b c, F(a) = 0 ∧ F(b) = 0 ∧ F(c) = 0) → 0 < k ∧ k < (real.sqrt 15) / 15) :=
begin
  sorry
end

end range_of_k_l153_153796


namespace value_of_expression_l153_153832

theorem value_of_expression (p q : ℚ) (h : p / q = 4 / 5) :
    11 / 7 + (2 * q - p) / (2 * q + p) = 2 :=
sorry

end value_of_expression_l153_153832


namespace probability_of_Z_l153_153126

namespace ProbabilityProof

def P_X : ℚ := 1 / 4
def P_Y : ℚ := 1 / 8
def P_X_or_Y_or_Z : ℚ := 0.4583333333333333

theorem probability_of_Z :
  ∃ P_Z : ℚ, P_Z = 0.0833333333333333 ∧ 
  P_X_or_Y_or_Z = P_X + P_Y + P_Z :=
by
  sorry

end ProbabilityProof

end probability_of_Z_l153_153126


namespace remainder_7n_mod_4_l153_153256

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l153_153256


namespace part1_part2_l153_153880

open BigOperators

namespace MathProof

def C (n k : ℕ) : ℕ := nat.choose n k

def P (n m : ℕ) : ℚ :=
  ∑ k in finset.range (n + 1), (-1 : ℚ) ^ k * C n k * (m : ℚ) / (m + k)

def Q (n m : ℕ) : ℕ := C (n + m) m

theorem part1 (n : ℕ) (h_n : 0 < n) : P n 1 * Q n 1 = 1 := sorry

theorem part2 (n m : ℕ) (h_n : 0 < n) (h_m : 0 < m) : P n m * Q n m = 1 := sorry

end MathProof

end part1_part2_l153_153880


namespace find_max_difference_l153_153683

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l153_153683


namespace square_of_binomial_l153_153590

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l153_153590


namespace prime_numbers_between_30_and_40_l153_153043

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m ∈ finset.range (n-2).succ, m + 2 ≤ n → ¬ (n % (m + 2) = 0)

theorem prime_numbers_between_30_and_40 : 
  finset.card (finset.filter is_prime (finset.Ico 31 40)) = 2 :=
by
  sorry

end prime_numbers_between_30_and_40_l153_153043


namespace age_of_15th_student_l153_153607

theorem age_of_15th_student 
  (avg_age_15_students : ℕ) 
  (total_students : ℕ) 
  (avg_age_5_students : ℕ) 
  (num_5_students : ℕ) 
  (avg_age_9_students : ℕ) 
  (num_9_students : ℕ) 
  (h_total_students : total_students = 15) 
  (h_avg_age_15_students : avg_age_15_students = 15) 
  (h_num_5_students : num_5_students = 5)
  (h_avg_age_5_students : avg_age_5_students = 14)
  (h_num_9_students : num_9_students = 9)
  (h_avg_age_9_students : avg_age_9_students = 16) 
  : let total_age_15_students := total_students * avg_age_15_students in
    let total_age_5_students := num_5_students * avg_age_5_students in
    let total_age_9_students := num_9_students * avg_age_9_students in
    let age_15th_student := total_age_15_students - (total_age_5_students + total_age_9_students) in
    age_15th_student = 11 := 
by 
  sorry

end age_of_15th_student_l153_153607


namespace vector_length_l153_153818

variables {a b : ℝ^3}
noncomputable def angle_a_b := (Real.pi / 6)

def norm_a : ℝ := 2
def norm_b : ℝ := Real.sqrt 3

theorem vector_length (h1 : Real.angle a b = angle_a_b)
  (h2 : ∥a∥ = norm_a) (h3 : ∥b∥ = norm_b) : ∥a - 2 • b∥ = 2 :=
sorry

end vector_length_l153_153818


namespace pascals_triangle_contains_47_once_l153_153074

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153074


namespace sufficient_but_not_necessary_condition_l153_153035

def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) :=
∀ x y ∈ I, x ≤ y → f x ≤ f y

def sequence_is_increasing (a : ℕ → ℝ) :=
∀ n : ℕ, a n ≤ a (n + 1)

theorem sufficient_but_not_necessary_condition (f : ℝ → ℝ) (a : ℕ → ℝ)
  (hf : ∀ n : ℕ, a n = f n) :
  (is_increasing_on f (set.Ici 1) →
    sequence_is_increasing a) ∧
  (¬ (sequence_is_increasing a → is_increasing_on f (set.Ici 1))) :=
sorry

end sufficient_but_not_necessary_condition_l153_153035


namespace percent_within_one_sd_of_mean_l153_153286

/- Define the mean and standard deviation -/
variables (m d : ℝ)

/- Define the distribution condition and the symmetry property -/
def is_symmetric_about_mean (f : ℝ → ℝ) (m : ℝ) : Prop :=
  ∀ x, f (m + x) = f (m - x)

def percent_less_than (f : ℝ → ℝ) (x : ℝ) : ℝ :=
  ∫ (-∞) (x) f

/- Main theorem -/
theorem percent_within_one_sd_of_mean
  (f : ℝ → ℝ)
  (h_symmetry : is_symmetric_about_mean f m)
  (h_84 : percent_less_than f (m + d) = 0.84) :
  percent_less_than f (m + d) - percent_less_than f (m - d) = 0.68 :=
by
  sorry

end percent_within_one_sd_of_mean_l153_153286


namespace probability_exactly_three_prime_numbers_on_six_dice_l153_153710

theorem probability_exactly_three_prime_numbers_on_six_dice :
  (probability (λ (dice_config : Fin 6 → Fin 12), 
    (Fintype.card (Set.filter (λ n, n ∈ {2, 3, 5, 7, 11}) (Finset.image dice_config Finset.univ)) = 3)) 
  = (857500 / 2985984 : ℚ)) :=
sorry

end probability_exactly_three_prime_numbers_on_six_dice_l153_153710


namespace plan1_more_cost_effective_than_plan2_l153_153341

variable (x : ℝ)

def plan1_cost (x : ℝ) : ℝ :=
  36 + 0.1 * x

def plan2_cost (x : ℝ) : ℝ :=
  0.6 * x

theorem plan1_more_cost_effective_than_plan2 (h : x > 72) : 
  plan1_cost x < plan2_cost x :=
by
  sorry

end plan1_more_cost_effective_than_plan2_l153_153341


namespace coin_flip_heads_probability_l153_153963

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153963


namespace _l153_153550

noncomputable theorem system_solution (x y : ℤ) (h1 : 2 * x - 3 * y = 5) (h2 : 4 * x - y = 5) : 
  x = 1 ∧ y = -1 := by
  sorry

end _l153_153550


namespace simplify_distance_sum_eq_l153_153223

theorem simplify_distance_sum_eq : 
  (∃ x y : ℝ, (real.sqrt ((x - 4)^2 + y^2) + real.sqrt ((x + 4)^2 + y^2) = 10) ↔
  (∃ x y : ℝ, (x^2 / 25 + y^2 / 9 = 1)) :=
by
  sorry

end simplify_distance_sum_eq_l153_153223


namespace num_girls_on_trip_l153_153631

/-- Given the conditions: 
  * Three adults each eating 3 eggs.
  * Ten boys each eating one more egg than each girl.
  * A total of 36 eggs.
  Prove that there are 7 girls on the trip. -/
theorem num_girls_on_trip (adults boys girls eggs : ℕ) 
  (H1 : adults = 3)
  (H2 : boys = 10)
  (H3 : eggs = 36)
  (H4 : ∀ g, (girls * g) + (boys * (g + 1)) + (adults * 3) = eggs)
  (H5 : ∀ g, g = 1) :
  girls = 7 :=
by
  sorry

end num_girls_on_trip_l153_153631


namespace find_value_of_expression_l153_153021

-- Given condition
variables (m : ℝ)
hypothesis (hm : m^2 - 2 * m - 7 = 0)

-- Goal
theorem find_value_of_expression : m^2 - 2 * m + 1 = 8 :=
sorry

end find_value_of_expression_l153_153021


namespace coin_flip_heads_probability_l153_153964

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153964


namespace common_tangent_midpoint_l153_153384

theorem common_tangent_midpoint
  (A B C D: Point)
  (s: ℝ)
  (sq: square A B C D s)
  (Oa Ob: Point)
  (Ta Tb: Point)
  (da db: ℝ)
  (circle_at_A: circle_at Oa Ta A da)
  (circle_at_B: circle_at Ob Tb B db)
  (Hdiam: da + db = s) :
  is_midpoint (common_tangent Oa Ob Ta Tb) AB :=
sorry

end common_tangent_midpoint_l153_153384


namespace f_zero_f_three_f_even_f_increasing_range_a_l153_153416

noncomputable def f : ℝ → ℝ := sorry

axiom f_mul (x y : ℝ) : f(x * y) = f(x) * f(y)
axiom f_neg1 : f(-1) = 1
axiom f_27 : f(27) = 9
axiom f_interval (x : ℝ) (h : 0 ≤ x ∧ x < 1) : 0 ≤ f(x) ∧ f(x) < 1

theorem f_zero : f(0) = 0 :=
sorry

theorem f_three : f(3) = real.cbrt 9 :=
sorry

theorem f_even (x : ℝ) : f(-x) = f(x) :=
sorry

theorem f_increasing (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x < y) : f(x) < f(y) :=
sorry

theorem range_a (a : ℝ) (h : 0 ≤ a) (ha : f(a + 1) ≤ 39) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end f_zero_f_three_f_even_f_increasing_range_a_l153_153416


namespace option_A_correct_option_D_correct_l153_153402

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

axiom f_neg_x_plus_1_add_f_x_minus_1_eq_zero : ∀ x : ℝ, f (-x + 1) + f (x - 1) = 0
axiom g_4_minus_x_eq_f_neg_2_plus_x_add_1 : ∀ x : ℝ, g (4 - x) = f (-2 + x) + 1
axiom f_eq_sin_pi_div_2_x : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → f x = Real.sin (Real.pi / 2 * x)
axiom f_x_plus_2_eq_neg_2_f_x : ∀ x : ℝ, x ≥ 2 → f (x + 2) = -2 * f x

theorem option_A_correct : ∀ n : ℕ, f (2 * n) = 0 := sorry

theorem option_D_correct : ∑ n in Finset.range (2023 + 1), g n = 2024 - (1 + 2^1011) / 3 := sorry

end option_A_correct_option_D_correct_l153_153402


namespace zero_in_interval_l153_153531

noncomputable def f (x : ℝ) : ℝ := Real.log x - 3 / x

theorem zero_in_interval : ∃ (c : ℝ), 2 < c ∧ c < 3 ∧ f c = 0 :=
by
  have f2_lt_0 : f 2 < 0 := by sorry
  have f3_gt_0 : f 3 > 0 := by sorry
  exact IntermediateValueTheorem f 2 3 f2_lt_0 f3_gt_0

end zero_in_interval_l153_153531


namespace trigonometric_identity_l153_153440

theorem trigonometric_identity (α : ℝ) (h : tan (α + π / 4) = -3) :
  (sin α + 2 * cos α) / (sin α - cos α) = 4 :=
by
  sorry

end trigonometric_identity_l153_153440


namespace passion_fruit_crates_l153_153152

def total_crates : ℕ := 50
def crates_of_grapes : ℕ := 13
def crates_of_mangoes : ℕ := 20
def crates_of_fruits_sold := crates_of_grapes + crates_of_mangoes

theorem passion_fruit_crates (p : ℕ) : p = total_crates - crates_of_fruits_sold :=
by
  have h1 : crates_of_fruits_sold = 13 + 20 := by rfl
  have h2 : p = 50 - 33 := by rw [h1]; rfl
  show p = 17 from h2

end passion_fruit_crates_l153_153152


namespace total_distance_traveled_l153_153328

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end total_distance_traveled_l153_153328


namespace solve_triangle_l153_153115

noncomputable def angle_A := 45
noncomputable def angle_B := 60
noncomputable def side_a := Real.sqrt 2

theorem solve_triangle {A B : ℕ} {a b : Real}
    (hA : A = angle_A)
    (hB : B = angle_B)
    (ha : a = side_a) :
    b = Real.sqrt 3 := 
by sorry

end solve_triangle_l153_153115


namespace standard_equation_of_ellipse_triangle_area_l153_153024

-- Given conditions
def major_axis_length : ℝ := 10
def f1_coordinates : Prod ℝ ℝ := (3, 0)
def f2_coordinates : Prod ℝ ℝ := (-3, 0)

-- Define constants for calculations
def a : ℝ := major_axis_length / 2
def c : ℝ := (f1_coordinates.1).abs

-- Calculate b from c^2 = a^2 - b^2
def b : ℝ := Real.sqrt (a^2 - c^2)

-- Define Standard Equation Proof
theorem standard_equation_of_ellipse :
    (a = 5) →
    (c = 3) →
    b = 4 →
    ∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 :=
sorry

-- Define Triangle Area Proof
def P : Prod ℝ ℝ := (0, b)

theorem triangle_area :
    ∀ (f1 f2 : Prod ℝ ℝ),
    (f1 = f1_coordinates) →
    (f2 = f2_coordinates) →
    (b = 4) →
    let height := b * 2 in
    let base := 2 * c in
    let area := (1 / 2) * base * height in
    area = 12 :=
sorry

end standard_equation_of_ellipse_triangle_area_l153_153024


namespace coin_flip_heads_probability_l153_153958

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153958


namespace average_math_score_first_year_students_l153_153119

theorem average_math_score_first_year_students 
  (total_male_students : ℕ) (total_female_students : ℕ)
  (sample_size : ℕ) (avg_score_male : ℕ) (avg_score_female : ℕ)
  (male_sample_size female_sample_size : ℕ)
  (weighted_avg : ℚ) :
  total_male_students = 300 → 
  total_female_students = 200 →
  sample_size = 60 → 
  avg_score_male = 110 →
  avg_score_female = 100 →
  male_sample_size = (3 * sample_size) / 5 →
  female_sample_size = (2 * sample_size) / 5 →
  weighted_avg = (male_sample_size * avg_score_male + female_sample_size * avg_score_female : ℕ) / sample_size → 
  weighted_avg = 106 := 
by
  sorry

end average_math_score_first_year_students_l153_153119


namespace refrigerator_feels_like_right_prism_l153_153270

-- Define the objects and their geometric shapes
inductive Object 
| Refrigerator 
| Basketball 
| Shuttlecock 
| Thermos 

def shape : Object → Type
| Object.Refrigerator := {P:Type}

def isRightPrism : Type → Prop := 
    λ P : Type, ∃ (b1 b2: P), (b1 = b2) ∧ (∃ lateral_face : P, true) 

-- Theorem: Refrigerator gives the feeling of a right prism
theorem refrigerator_feels_like_right_prism : isRightPrism (shape Object.Refrigerator) := 
    sorry

end refrigerator_feels_like_right_prism_l153_153270


namespace expected_heads_after_four_tosses_l153_153857
open Probability

noncomputable def P_heads_one_toss : ℚ := 1 / 3
noncomputable def P_tails_one_toss : ℚ := 2 / 3

theorem expected_heads_after_four_tosses (n : ℕ) (h100 : n = 100) : 
  let P_heads_up_to_four_tosses := 
    P_heads_one_toss + 
    P_tails_one_toss * P_heads_one_toss + 
    P_tails_one_toss^2 * P_heads_one_toss + 
    P_tails_one_toss^3 * P_heads_one_toss
  in 
  (n : ℚ) * P_heads_up_to_four_tosses = 80 := 
by
  sorry

end expected_heads_after_four_tosses_l153_153857


namespace Elisa_paint_total_is_correct_l153_153179

def day1 (square_feet : ℕ) : Prop :=
  square_feet = 30

def day2 (monday_paint : ℕ) (tuesday_paint : ℕ) : Prop :=
  tuesday_paint = 2 * monday_paint

def day3 (monday_paint : ℕ) (wednesday_paint : ℕ) : Prop :=
  wednesday_paint = monday_paint / 2

def total_paint (monday_paint tuesday_paint wednesday_paint total_paint : ℕ) : Prop :=
  total_paint = monday_paint + tuesday_paint + wednesday_paint

theorem Elisa_paint_total_is_correct :
  ∃ (monday_paint tuesday_paint wednesday_paint total_paint : ℕ),
    day1 monday_paint ∧
    day2 monday_paint tuesday_paint ∧
    day3 monday_paint wednesday_paint ∧
    total_paint monday_paint tuesday_paint wednesday_paint total_paint ∧
    total_paint = 105 :=
by
  sorry

end Elisa_paint_total_is_correct_l153_153179


namespace polynomial_remainder_l153_153867

theorem polynomial_remainder :
  ∀ Q : ℚ[X],
  (Q.eval 18 = 20) →
  (Q.eval 12 = 10) →
  ∃ (c d : ℚ), 
    (Q = (λ R, R * (X - 12) * (X - 18) + c * X + d) ∧
      c = 5/3 ∧
      d = -10) :=
by
  intros Q hQ18 hQ12
  use [(5 / 3), (-10)]
  split
  {
    sorry
  }
  {
    constructor
    {
      refl
    }
    {
      refl
    }
  }

end polynomial_remainder_l153_153867


namespace ticket_distribution_count_l153_153556

theorem ticket_distribution_count :
  let A := 2
  let B := 2
  let C := 1
  let D := 1
  let total_tickets := A + B + C + D
  ∃ (num_dist : ℕ), num_dist = 180 :=
by {
  sorry
}

end ticket_distribution_count_l153_153556


namespace arithmetic_sequence_properties_l153_153776

noncomputable def a_n (n : ℕ) : ℕ := 2 * n + 1
noncomputable def b_n (n : ℕ) : ℕ := a_n (2^n)
noncomputable def S (n : ℕ) : ℕ := n * (2 * a_n 1 + (n - 1) * 2) / 2
noncomputable def T (n : ℕ) : ℕ := (List.range n).sum b_n

theorem arithmetic_sequence_properties :
  let a1 := a_n 1
  let d := 2
  S 3 + S 5 = 50 ∧ (a1 + 3 * d)^2 = a1 * (a1 + 12 * d) ∧ d ≠ 0 →
  (∀ n, a_n n = 2 * n + 1) ∧
  (∀ n, T n = 2^(n+2) + n - 4) :=
by
  intros a1 d h
  sorry

end arithmetic_sequence_properties_l153_153776


namespace polynomial_not_factored_l153_153511

noncomputable def distinct_integers : Type := sorry

theorem polynomial_not_factored (a : list ℤ) (h_distinct : a.nodup) :
    ∀ (p q : polynomial ℤ), 
    (∏ i in a, (X - C i)^2 + 1 = p * q → false) :=
sorry

end polynomial_not_factored_l153_153511


namespace probability_of_draw_l153_153235

-- Let P be the probability of the game ending in a draw.
-- Let PA be the probability of Player A winning.

def PA_not_losing := 0.8
def PB_not_losing := 0.7

theorem probability_of_draw : ¬ (1 - PA_not_losing + PB_not_losing ≠ 1.5) → PA_not_losing + (1 - PB_not_losing) = 1.5 → PB_not_losing + 0.5 = 1 := by
  intros
  sorry

end probability_of_draw_l153_153235


namespace mean_of_added_numbers_l153_153914

noncomputable def mean (a : List ℚ) : ℚ :=
  (a.sum) / (a.length)

theorem mean_of_added_numbers 
  (sum_eight_numbers : ℚ)
  (sum_eleven_numbers : ℚ)
  (x y z : ℚ)
  (h_eight : sum_eight_numbers = 8 * 72)
  (h_eleven : sum_eleven_numbers = 11 * 85)
  (h_sum_added : x + y + z = sum_eleven_numbers - sum_eight_numbers) :
  (x + y + z) / 3 = 119 + 2/3 := 
sorry

end mean_of_added_numbers_l153_153914


namespace order_of_a_b_c_l153_153540

noncomputable def f : ℝ → ℝ := sorry

theorem order_of_a_b_c :
  (∀ x : ℝ, x < 0 → f(x) + x * deriv f x < 0) →
  (∀ x : ℝ, f(x) = -f(-x)) →
  let a := 3^0.3 * f(3^0.3) in
  let b := Real.log 3 / Real.log π * f(Real.log 3 / Real.log π) in
  let c := -2 * f(-2) in
  c > a ∧ a > b :=
by
  intros h1 h2
  let a := 3^0.3 * f(3^0.3)
  let b := Real.log 3 / Real.log π * f(Real.log 3 / Real.log π)
  let c := -2 * f(-2)
  sorry

end order_of_a_b_c_l153_153540


namespace rational_function_properties_l153_153923

theorem rational_function_properties (p q : ℚ → ℚ) (h₁ : ∀ x, p(3) = 2)
  (h₂ : ∀ x, q(3) = 4) 
  (h₃ : ∃ c d : ℚ, q(x) = c * (x + 2) * (x - 1) * x ∧ p(x) = d * x * (x - 1) 
    ∧ p(x) / q(x) has_horizontal_asymptote 0 
    ∧ p(x) / q(x) has_vertical_asymptotes [x = -2, x = 1]
    ∧ has_hole p q x = 0) :
  p(x) + q(x) = (2 / 15) * x^3 + (17 / 15) * x^2 - (32 / 15) * x := by 
sorry

end rational_function_properties_l153_153923


namespace find_number_l153_153283

theorem find_number (x : ℤ) (h : 300 + 8 * x = 340) : x = 5 := by
  sorry

end find_number_l153_153283


namespace exists_bounding_lines_l153_153630

theorem exists_bounding_lines (L : Finset (Set Point))
  (h_card : L.card = 2006)
  (h_no_parallel : ∀ l1 l2 ∈ L, l1 ≠ l2 → ∀ p1 p2 : Point, p1 ∈ l1 → p2 ∈ l2 → ¬(p1 = p2))
  (h_no_tripoint : ∀ l1 l2 l3 ∈ L, (∀ p1 : Point, p1 ∈ l1 → p1 ∈ l2 → p1 ∈ l3 → False))
  : ∃ l l' ∈ L, (∀ p ∈ L, p ≠ l → ∀ q, q ∈ p → q ∈ l' → (∀ r, r ∈ p → r ∈ l → ¬(q = r))) ∧ ¬(∀ p ∈ L, p ≠ l' → ∀ q, q ∈ p → q ∈ l → (∀ r, r ∈ p → r ∈ l' → ¬(q = r))) := 
sorry

end exists_bounding_lines_l153_153630


namespace largest_diff_even_digits_l153_153665

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l153_153665


namespace chocolate_candy_cost_l153_153285

theorem chocolate_candy_cost 
  (candy_per_box : ℕ := 30)
  (num_candies : ℕ := 900)
  (cost_per_box : ℝ := 7.50)
  (discount_threshold : ℕ := 500)
  (discount_rate : ℝ := 0.10) :
  (num_candies > discount_threshold * candy_per_box) →
  (let num_boxes := num_candies / candy_per_box in
   let discounted_cost_per_box := cost_per_box * (1 - discount_rate) in
   let total_cost := num_boxes * discounted_cost_per_box in
   total_cost = 202.50) :=
by
  sorry

end chocolate_candy_cost_l153_153285


namespace thomas_total_blocks_l153_153558

-- Definitions according to the conditions
def a1 : Nat := 7
def a2 : Nat := a1 + 3
def a3 : Nat := a2 - 6
def a4 : Nat := a3 + 10
def a5 : Nat := 2 * a2

-- The total number of blocks
def total_blocks : Nat := a1 + a2 + a3 + a4 + a5

-- The proof statement
theorem thomas_total_blocks :
  total_blocks = 55 := 
sorry

end thomas_total_blocks_l153_153558


namespace smallest_integer_neither_prime_nor_cube_with_even_prime_factors_all_gt_60_l153_153575

theorem smallest_integer_neither_prime_nor_cube_with_even_prime_factors_all_gt_60 :
  ∃ n : ℕ,
    n = 3721 ∧
    ¬ Prime n ∧
    ∀ k : ℕ, k^3 ≠ n ∧
    even (PrimeFactors.card n) ∧
    ∀ p : ℕ, p ∈ PrimeFactors n → p > 60 :=
by
  sorry

end smallest_integer_neither_prime_nor_cube_with_even_prime_factors_all_gt_60_l153_153575


namespace sum_second_largest_smallest_l153_153228

theorem sum_second_largest_smallest (a b c : ℕ) (order_cond : a < b ∧ b < c) : a + b = 21 :=
by
  -- Following the correct answer based on the provided conditions:
  -- 10, 11, and 12 with their ordering, we have the smallest a and the second largest b.
  sorry

end sum_second_largest_smallest_l153_153228


namespace lambda_range_l153_153448

noncomputable def inequality_condition (a b c λ : ℝ) : Prop :=
  (a > b) ∧ (b > c) ∧ (1 / (a - b) + 1 / (b - c) + λ / (c - a) > 0)

theorem lambda_range (λ : ℝ) :
  (∀ a b c : ℝ, inequality_condition a b c λ) ↔ (λ < 4) :=
by sorry

end lambda_range_l153_153448


namespace pascals_triangle_contains_47_once_l153_153069

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153069


namespace max_value_of_k_l153_153372

theorem max_value_of_k:
  ∃ (k : ℕ), 
  (∀ (a b : ℕ → ℕ) (h : ∀ i, a i < b i) (no_share : ∀ i j, i ≠ j → (a i ≠ a j ∧ a i ≠ b j ∧ b i ≠ a j ∧ b i ≠ b j)) (distinct_sums : ∀ i j, i ≠ j → a i + b i ≠ a j + b j) (sum_limit : ∀ i, a i + b i ≤ 3011), 
    k ≤ 3011 ∧ k = 1204) := sorry

end max_value_of_k_l153_153372


namespace remainder_of_7n_mod_4_l153_153244

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l153_153244


namespace find_circle_radius_l153_153559

-- Lean statement for the given problem.

theorem find_circle_radius :
  (∃ (A B C : Point) (circle : Circle), 
    A = (2, 0) ∧ B = (8, 0) ∧ C = (5, 5) ∧ 
    is_inscribed_in_triangle circle (triangle.mk A B C) ∧ 
    (∃ (P Q R S : Point),
      lies_on_segment P A C ∧ lies_on_segment Q B C ∧
      lies_on_circle R circle ∧ lies_on_circle S circle ∧
      is_square (square.mk P Q R S) 5)) → 
  (radius circle = 5) :=
by
  sorry

end find_circle_radius_l153_153559


namespace find_max_difference_l153_153682

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l153_153682


namespace trabant_mixture_density_l153_153230

theorem trabant_mixture_density : 
  ∀ (V : ℝ), (ρ_gasoline ρ_oil : ℝ), ρ_gasoline = 700 → ρ_oil = 900 → 
  let V_oil := V / 40 in
  let V' := V + V_oil in
  let M := V * ρ_gasoline + V_oil * ρ_oil in
  let ρ' := M / V' in
  ρ' = 704.9 :=
by
  intros V ρ_gasoline ρ_oil h1 h2
  let V_oil := V / 40
  let V' := V + V_oil
  let M := V * ρ_gasoline + V_oil * ρ_oil
  let ρ' := M / V'
  have h3 : ρ_gasoline = 700 := h1.symm
  have h4 : ρ_oil = 900 := h2.symm
  rw [h3, h4] at *
  have M_calc : M = 700 * V + (V / 40) * 900 := rfl
  rw M_calc
  have M_simp : M = 722.5 * V := by ring
  rw M_simp
  have V'_calc: V' = (41 / 40) * V := by field_simp [V']
  rw V'_calc
  have ρ'_calc: ρ' = 722.5 / (41 / 40) := by field_simp [ρ']
  rw ρ'_calc
  norm_num
  exact rfl

end trabant_mixture_density_l153_153230


namespace ratio_xavier_to_katie_is_3_to_1_l153_153273

-- Definitions for the conditions
def miles_cole_runs : ℕ := 7
def miles_katie_runs (C : ℕ) : ℕ := 4 * C
def miles_xavier_runs : ℕ := 84

-- Lean theorem stating that the ratio of Xavier's miles to Katie's miles is 3:1
theorem ratio_xavier_to_katie_is_3_to_1 :
  let C := miles_cole_runs,
      K := miles_katie_runs C,
      X := miles_xavier_runs in
  X / K = 3 :=
by
  let C := miles_cole_runs
  have hK : C = 7 := rfl
  let K := miles_katie_runs C
  have hK_def : K = 28 := by rw [hK, miles_katie_runs]
  let X := miles_xavier_runs
  have hX : X = 84 := rfl
  calc
    X / K = 84 / 28 : by rw [hX, hK_def]
        ... = 3 : by norm_num

end ratio_xavier_to_katie_is_3_to_1_l153_153273


namespace exists_mathematician_with_high_average_friends_l153_153124

open Finset

variable {V : Type} [Fintype V] [DecidableEq V] (G : SimpleGraph V) [Nonempty V]

theorem exists_mathematician_with_high_average_friends (h1 : ∀ u : V, 1 ≤ G.degree u) : 
  ∃ u : V, (∑ v in G.neighbors u, G.degree v) / G.degree u ≥ (∑ u in Finset.univ, G.degree u) / Fintype.card V :=
sorry

end exists_mathematician_with_high_average_friends_l153_153124


namespace trapezoid_area_relation_l153_153385

variables (A B C D O M : Type) 
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] 
variables [AffineSpace ℝ O] [AffineSpace ℝ M]
variables (BC AD : Line ℝ) 
variables [Parallel BC AD]
variables (S_AOM S_COD S_ABCD : ℝ)

theorem trapezoid_area_relation
  (trapezoid : Trapezoid A B C D)
  (inscribed_circle : InscribedCircle O A B C D)
  (BO_intersects_AD_at_M : IntersectsAt M BO AD)
  (area_relation : S_AOM + S_COD = ½ * S_ABCD) :
  S_AOM + S_COD = ½ * S_ABCD :=
sorry

end trapezoid_area_relation_l153_153385


namespace omega_range_l153_153413

noncomputable def f (ω x : ℝ) : ℝ := 3 + 2 * sin (ω * x) * cos (ω * x) - 2 * sqrt 3 * (cos (ω * x))^2

theorem omega_range (ω : ℝ) : 
  (∀ x ∈ Ioo π (2 * π), derivative (f ω) x ≠ 0) ∧ ω > 0 → ω ∈ Ioo 0 (5/24) ∪ Icc (5/12) (11/24) :=
sorry

end omega_range_l153_153413


namespace find_unknown_number_l153_153749

theorem find_unknown_number (x : ℝ) (h : (28 + 48 / x) * x = 1980) : x = 69 :=
sorry

end find_unknown_number_l153_153749


namespace find_a_and_b_l153_153205

theorem find_a_and_b (a b : ℝ) (f : ℝ → ℝ) :
  (∀ x, f x = x^3 - a * x^2 - b * x + a^2) →
  f 1 = 10 →
  deriv f 1 = 0 →
  (a = -4 ∧ b = 11) :=
by
  intros hf hf1 hderiv
  sorry

end find_a_and_b_l153_153205


namespace lcm_ge_ten_times_first_l153_153377

open Nat

theorem lcm_ge_ten_times_first {a : ℕ → ℕ} (h : ∀ i j, i < j → a i < a j) :
    ∀ (a1 : ℕ), a1 = a 0 → (∀ i, i < 10 → a i ∈ ℕ) →
    lcm (finset.range 10).val.map a ≥ 10 * a 0 := 
by
  sorry

end lcm_ge_ten_times_first_l153_153377


namespace joggers_difference_l153_153321

-- Define the conditions as per the problem statement
variables (Tyson Alexander Christopher : ℕ)
variable (H1 : Alexander = Tyson + 22)
variable (H2 : Christopher = 20 * Tyson)
variable (H3 : Christopher = 80)

-- The theorem statement to prove Christopher bought 54 more joggers than Alexander
theorem joggers_difference : (Christopher - Alexander) = 54 :=
  sorry

end joggers_difference_l153_153321


namespace triangle_right_if_angle_difference_l153_153147

noncomputable def is_right_triangle (A B C : ℝ) : Prop := 
  A = 90

theorem triangle_right_if_angle_difference (A B C : ℝ) (h : A - B = C) (sum_angles : A + B + C = 180) :
  is_right_triangle A B C :=
  sorry

end triangle_right_if_angle_difference_l153_153147


namespace count_integers_Q_le_zero_l153_153728

def Q (x : ℤ) : ℤ := ∏ k in (finset.range 50).map (function.embedding.subtype _), (x - (k + 1) ^ 2)

def num_ints_le_zero_Q : ℤ :=
  1300

theorem count_integers_Q_le_zero :
  (finset.filter (λ n : ℤ, Q n ≤ 0) (finset.Icc (-((50) ^ 2)) ((50) ^ 2))).card = num_ints_le_zero_Q :=
  sorry

end count_integers_Q_le_zero_l153_153728


namespace minimum_length_intersection_l153_153236

def length (a b : ℝ) : ℝ := b - a

def M (m : ℝ) : Set ℝ := { x | m ≤ x ∧ x ≤ m + 2/3 }
def N (n : ℝ) : Set ℝ := { x | n - 1/2 ≤ x ∧ x ≤ n }

def IntervalSet : Set ℝ := { x | 0 ≤ x ∧ x ≤ 1 }

theorem minimum_length_intersection (m n : ℝ) (hM : M m ⊆ IntervalSet) (hN : N n ⊆ IntervalSet) :
  length (max m (n - 1/2)) (min (m + 2/3) n) = 1/6 :=
by
  sorry

end minimum_length_intersection_l153_153236


namespace part1_part2_l153_153814

universe u
open Set Real

variable (A B C : Set ℝ) (U : Set ℝ) (a : ℝ)

def A := {x : ℝ | 2 ≤ x ∧ x ≤ 6}
def B := {x : ℝ | -1 < x ∧ x < 5}
def C := {x : ℝ | x < a}
def U := @Set.univ ℝ

theorem part1 : A ∩ (U \ B) = {x : ℝ | 5 ≤ x ∧ x ≤ 6} := by sorry

theorem part2 (h : A ∩ C ≠ ∅) : a ∈ Ioi 2 := by sorry

end part1_part2_l153_153814


namespace longest_side_similar_triangle_l153_153927

-- Define the original triangle's sides
def side_a := 8
def side_b := 10
def side_c := 12

-- Define the perimeter of the similar triangle
def similar_triangle_perimeter := 150

-- Define the area scaling factor
def area_scaling_factor := 3

-- Using Heron's formula to calculate the area
noncomputable def heron_area (a b c : ℕ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

-- Define the area of the original triangle
noncomputable def original_triangle_area : ℝ :=
  heron_area side_a side_b side_c

-- Define the area of the similar triangle
noncomputable def similar_triangle_area : ℝ :=
  area_scaling_factor * original_triangle_area

-- Proving that the longest side of the similar triangle is 60 cm
theorem longest_side_similar_triangle :
  ∃ (x : ℝ), 8 * x = 150 / 30 ∧ 12 * (150 / 30) = 60 :=
by
  -- Calculation steps would follow to prove x = 5 and the longest side equals to 60 cm.
  sorry

end longest_side_similar_triangle_l153_153927


namespace base7_perfect_square_xy5z_l153_153107

theorem base7_perfect_square_xy5z (n : ℕ) (x y z : ℕ) (hx : x ≠ 0) (hn : n = 343 * x + 49 * y + 35 + z) (hsq : ∃ m : ℕ, n = m * m) : z = 1 ∨ z = 6 :=
sorry

end base7_perfect_square_xy5z_l153_153107


namespace area_triangle_MNP_l153_153954

/-- 
Given two circles centered at O and P with radii 5 and 6 respectively,
where circle O passes through P, and the intersection points of the circles
are M and N, the area of triangle MNP can be expressed as a fraction in simplest
form as a/b. Prove that a + b = 12.
-/
theorem area_triangle_MNP (O P M N : Point) :
  dist O P = 5 ∧ dist O M = 5 ∧ dist O N = 5 ∧ dist P M = 6 ∧ dist P N = 6 ∧
  dist M N = 2 * Real.sqrt(11) → 
  (∃ (a b : ℕ), (triangle_area P M N = a / b) ∧ (a + b = 12)) :=
sorry

end area_triangle_MNP_l153_153954


namespace cone_volume_ratio_l153_153343

theorem cone_volume_ratio (rC hC rD hD : ℝ) (h_rC : rC = 10) (h_hC : hC = 20) (h_rD : rD = 20) (h_hD : hD = 10) :
  ((1/3) * π * rC^2 * hC) / ((1/3) * π * rD^2 * hD) = 1/2 :=
by 
  sorry

end cone_volume_ratio_l153_153343


namespace sum_x_coordinates_of_intersection_l153_153918

def g : ℝ → ℝ
| x := if x < -3 then -3 * x - 4
  else if x < -1 then -x - 1
  else if x < 1 then 3 * x
  else if x < 3 then -x + 4
  else if x < 4 then x - 1
  else 2 * x - 5

theorem sum_x_coordinates_of_intersection : 
  let intersections := ([
    (if -3 < -2 ∧ -2 < -1 then some (-1.5) else none), 
    (if -1 < 1 ∧ 1 < 3 then some 1 else none)
  ].filter_map (λ x_opt, x_opt)) in
  intersections.sum = -0.5
:= sorry

end sum_x_coordinates_of_intersection_l153_153918


namespace range_of_a_l153_153794

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x - Real.log x

theorem range_of_a {a : ℝ} :
  (∀ x > 1, f a x > 1) → a ∈ Set.Ici 1 := by
  sorry

end range_of_a_l153_153794


namespace distance_between_complex_points_l153_153466

open Complex

noncomputable def z1 : ℂ := 8 + 5 * Complex.i
noncomputable def z2 : ℂ := 4 + 2 * Complex.i

theorem distance_between_complex_points :
  Complex.abs (z1 - z2) = 5 :=
by
  sorry

end distance_between_complex_points_l153_153466


namespace function_is_monotonically_increasing_l153_153538

theorem function_is_monotonically_increasing (a : ℝ) :
  (∀ x : ℝ, (x^2 + 2*x + a) ≥ 0) ↔ (1 ≤ a) := 
sorry

end function_is_monotonically_increasing_l153_153538


namespace pascal_triangle_47_rows_l153_153086

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153086


namespace pascals_triangle_contains_47_once_l153_153070

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153070


namespace coeff_x2y3_in_expansion_eq_neg40_l153_153537

theorem coeff_x2y3_in_expansion_eq_neg40 : 
  (∃ (c : ℤ), coeff (binomial_expansion (2 * x - y) 5) (x^2 * y^3) = c) → c = -40 :=
by
  sorry

end coeff_x2y3_in_expansion_eq_neg40_l153_153537


namespace marbles_leftover_l153_153593

theorem marbles_leftover (g j : ℕ) (hg : g % 8 = 5) (hj : j % 8 = 6) :
  ((g + 5 + j) % 8) = 0 :=
by
  sorry

end marbles_leftover_l153_153593


namespace mrs_heine_dogs_treats_l153_153499

theorem mrs_heine_dogs_treats (heart_biscuits_per_dog puppy_boots_per_dog total_items : ℕ)
  (h_biscuits : heart_biscuits_per_dog = 5)
  (h_boots : puppy_boots_per_dog = 1)
  (total : total_items = 12) :
  (total_items / (heart_biscuits_per_dog + puppy_boots_per_dog)) = 2 :=
by
  sorry

end mrs_heine_dogs_treats_l153_153499


namespace goods_train_speed_l153_153599

theorem goods_train_speed (speed_mans_train_kmph : ℚ) (time_pass_seconds : ℚ) (length_goods_train_m : ℚ) : 
  (speed_mans_train_kmph = 50) → 
  (time_pass_seconds = 9) → 
  (length_goods_train_m = 280) →
  let speed_mans_train_mps := speed_mans_train_kmph * 1000 / 3600 in
  let relative_speed_mps := length_goods_train_m / time_pass_seconds in
  let speed_goods_train_mps := relative_speed_mps - speed_mans_train_mps in
  let speed_goods_train_kmph := speed_goods_train_mps * 3600 / 1000 in
  speed_goods_train_kmph = 62 := 
begin
  intros h1 h2 h3,
  have h4 : speed_mans_train_mps = 50 * 1000 / 3600,
    sorry,
  have h5 : relative_speed_mps = 280 / 9,
    sorry,
  have h6 : speed_goods_train_mps = 280 / 9 - 50 * 1000 / 3600,
    sorry,
  have h7 : speed_goods_train_kmph = (280 / 9 - 50 * 1000 / 3600) * 3600 / 1000,
    sorry,
  show speed_goods_train_kmph = 62,
    sorry
end

end goods_train_speed_l153_153599


namespace inv_64_mod_97_l153_153015

theorem inv_64_mod_97 (h : 8⁻¹ ≡ 85 [MOD 97]) : 64⁻¹ ≡ 47 [MOD 97] :=
begin
  sorry
end

end inv_64_mod_97_l153_153015


namespace find_f_2008_l153_153826

open Function

noncomputable def f : ℝ → ℝ := sorry

theorem find_f_2008 (hf₀ : f 0 = 1)
  (h₁ : ∀ x : ℝ, f(x + 4) ≤ f(x) + 4)
  (h₂ : ∀ x : ℝ, f(x + 2) ≥ f(x) + 2) :
  f 2008 = 2009 :=
sorry

end find_f_2008_l153_153826


namespace isosceles_triangle_vertex_angle_l153_153460

theorem isosceles_triangle_vertex_angle (α : ℝ) (β : ℝ) (γ : ℝ)
  (h1 : α = β)
  (h2: α = 70) 
  (h3 : α + β + γ = 180) : 
  γ = 40 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l153_153460


namespace find_x_l153_153598

def floor (x : ℝ) := int.floor x

theorem find_x (x : ℝ) : (floor (x - 1 / 2) = 3 * x - 5) → x = 2 := by
  sorry

end find_x_l153_153598


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153078

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153078


namespace alcohol_water_ratio_l153_153233

theorem alcohol_water_ratio
  (V p q : ℝ)
  (hV : V > 0)
  (hp : p > 0)
  (hq : q > 0) :
  let total_alcohol := 3 * V * (p / (p + 1)) + V * (q / (q + 1))
  let total_water := 3 * V * (1 / (p + 1)) + V * (1 / (q + 1))
  total_alcohol / total_water = (3 * p * (q + 1) + q * (p + 1)) / (3 * (q + 1) + (p + 1)) :=
sorry

end alcohol_water_ratio_l153_153233


namespace value_of_k_l153_153586

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l153_153586


namespace lucille_cents_left_l153_153496

def lucille_earnings (weeds_flower_bed weeds_vegetable_patch weeds_grass : ℕ) (cent_per_weed soda_cost : ℕ) : ℕ :=
  let weeds_pulled := weeds_flower_bed + weeds_vegetable_patch + weeds_grass / 2 in
  let total_earnings := weeds_pulled * cent_per_weed in
  total_earnings - soda_cost

theorem lucille_cents_left :
  let weeds_flower_bed := 11
  let weeds_vegetable_patch := 14
  let weeds_grass := 32
  let cent_per_weed := 6
  let soda_cost := 99
  lucille_earnings weeds_flower_bed weeds_vegetable_patch weeds_grass cent_per_weed soda_cost = 147 := by
  sorry

end lucille_cents_left_l153_153496


namespace sum_of_ages_l153_153882

theorem sum_of_ages (M S G : ℕ)
  (h1 : M = 2 * S)
  (h2 : S = 2 * G)
  (h3 : G = 20) :
  M + S + G = 140 :=
sorry

end sum_of_ages_l153_153882


namespace coefficient_a_for_factor_l153_153241

noncomputable def P (a : ℚ) (x : ℚ) : ℚ := x^3 + 2 * x^2 + a * x + 20

theorem coefficient_a_for_factor (a : ℚ) :
  (∀ x : ℚ, (x - 3) ∣ P a x) → a = -65/3 :=
by
  sorry

end coefficient_a_for_factor_l153_153241


namespace smallest_possible_degree_of_polynomial_l153_153528

-- Define the given roots
def root1 : ℝ := 2 - sqrt 5
def root2 : ℝ := 4 + sqrt 10
def root3 : ℝ := 14 - 2 * sqrt 7
def root4 : ℝ := - sqrt 2

-- Define the smallest degree
def smallest_degree := 8

-- Main statement asserting the smallest possible degree of the polynomial
theorem smallest_possible_degree_of_polynomial :
  ∀ (p : Polynomial ℚ),
    (p ≠ 0) →
    (Polynomial.aeval root1 p = 0) →
    (Polynomial.aeval root2 p = 0) →
    (Polynomial.aeval root3 p = 0) →
    (Polynomial.aeval root4 p = 0) →
    (∃ (deg : ℕ), Polynomial.degree p = Polynomial.natDegree p ∧ deg ≥ smallest_degree) := 
begin
  sorry
end

end smallest_possible_degree_of_polynomial_l153_153528


namespace pascals_triangle_contains_47_once_l153_153071

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153071


namespace algebraic_expression_independence_l153_153452

theorem algebraic_expression_independence (a b : ℝ) (h : ∀ x : ℝ, (x^2 + a*x - (b*x^2 - x - 3)) = 3) : a - b = -2 :=
by
  sorry

end algebraic_expression_independence_l153_153452


namespace find_a1_b1_l153_153643

def seq (n : ℕ) : ℂ :=
  match n with
  | 0     => a + b * complex.I
  | (n+1) => (√3 + complex.I) * seq n

theorem find_a1_b1 (a b : ℝ) :
  seq 150 = 4 + 0 * complex.I →
  a + b = 4 / 2^149 := by
  sorry

end find_a1_b1_l153_153643


namespace determine_winner_l153_153229

inductive Player
| first
| second

def move (left : ℕ → ℕ) (pos : ℕ) : bool :=
  if pos ≥ 1 ∧ pos ≤ 3 then true
  else if pos <= 14 ∧ pos > 3 then left pos ∧ left (pos - 1) ∧ left (pos - 2) ∧ left (pos - 3)
  else false

def winning_position (pos : ℕ) : Player :=
  if pos = 4 ∨ pos = 8 ∨ pos = 12 then Player.second
  else Player.first

theorem determine_winner : ∀ (pos : ℕ), (pos ≤ 14 ∧ pos ≥ 0) → (winning_position pos = Player.second) ↔ (pos = 4 ∨ pos = 8 ∨ pos = 12) :=
by
  intros pos h
  unfold winning_position
  split_ifs with h1; exact h1
  split_ifs with h2; split; intro h3;
  simp at *; try { contradiction }
  sorry

end determine_winner_l153_153229


namespace coin_flip_heads_probability_l153_153960

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153960


namespace probability_of_selecting_quarter_l153_153296

theorem probability_of_selecting_quarter 
  (value_quarters value_nickels value_pennies total_value : ℚ)
  (coin_value_quarter coin_value_nickel coin_value_penny : ℚ) 
  (h1 : value_quarters = 10)
  (h2 : value_nickels = 10)
  (h3 : value_pennies = 10)
  (h4 : coin_value_quarter = 0.25)
  (h5 : coin_value_nickel = 0.05)
  (h6 : coin_value_penny = 0.01)
  (total_coins : ℚ) 
  (h7 : total_coins = (value_quarters / coin_value_quarter) + (value_nickels / coin_value_nickel) + (value_pennies / coin_value_penny)) : 
  (value_quarters / coin_value_quarter) / total_coins = 1 / 31 :=
by
  sorry

end probability_of_selecting_quarter_l153_153296


namespace pascal_triangle_contains_47_l153_153063

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153063


namespace problem_statement_l153_153016

theorem problem_statement (a : ℕ) (h : 0 < a) :
  (sqrt (8 + (8 / a)) = 8 * sqrt (8 / a)) → a = 63 :=
by sorry

end problem_statement_l153_153016


namespace remainder_when_7n_divided_by_4_l153_153247

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l153_153247


namespace pq_plus_p_plus_q_eq_1_l153_153875

-- Define the polynomial
def poly (x : ℝ) : ℝ := x^4 - 6 * x - 1

-- Prove the target statement
theorem pq_plus_p_plus_q_eq_1 (p q : ℝ) (hpq : poly p = 0) (hq : poly q = 0) :
  p * q + p + q = 1 := by
  sorry

end pq_plus_p_plus_q_eq_1_l153_153875


namespace acute_angle_between_planes_l153_153463

noncomputable def right_angled_triangle (A B C : Type) := sorry -- definition of a right-angled triangle

def hypotenuse_lies_on_plane (ABC : Type) (α : Type) := sorry -- definition where hypotenuse lies on plane α
def vertex_outside_plane (C : Type) (α : Type) := sorry -- definition where vertex C is outside plane α
def angle_with_plane (l : Type) (α : Type) (θ : Real) := sorry -- definition for angle of line l with plane α

theorem acute_angle_between_planes (A B C α : Type) 
  [right_angled_triangle A B C]
  [hypotenuse_lies_on_plane (A, B, C) α]
  [vertex_outside_plane C α]
  (hAC : angle_with_plane A α 30)
  (hBC : angle_with_plane B α 45) : 
  ∃ θ, θ = 60 :=
begin
  sorry
end

end acute_angle_between_planes_l153_153463


namespace coin_payment_difference_l153_153477

/-- John needs to pay exactly 30 cents using 5-cent coins, 10-cent coins, and a 20-cent coin. -/
def min_coins (total : ℕ) : ℕ :=
if total ≥ 20 then 1 + (total - 20) / 10 else total / 5

/-- John needs to pay exactly 30 cents using 5-cent coins, 10-cent coins, and a 20-cent coin. -/
def max_coins (total : ℕ) : ℕ :=
total / 5

theorem coin_payment_difference :
  let total := 30 in
  max_coins total - min_coins total = 4 :=
by
  let total := 30
  have h_min_coins : min_coins total = 2 := by sorry
  have h_max_coins : max_coins total = 6 := by sorry
  rw [h_min_coins, h_max_coins]
  norm_num

end coin_payment_difference_l153_153477


namespace find_annual_income_l153_153225

noncomputable def annual_income (q : ℝ) (A : ℝ) : ℝ :=
  let T := 0.01 * q * 35000 + 0.01 * (q + 3) * (50000 - 35000) + 0.01 * (q + 5) * (A - 50000)
  (q + 0.75) / 100 * A

theorem find_annual_income (q : ℝ) (A : ℝ) :
  A = 48235 → 
  (∀ A > 50000, annual_income q A = (q + 0.75) / 100 * A) :=
begin
  intro h,
  sorry,
end

end find_annual_income_l153_153225


namespace monotonic_decreasing_interval_of_f_l153_153213

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonic_decreasing_interval_of_f :
    ((∀ (x : ℝ), x > 0 → f' x = Real.log x + 1) →
     (∀ (x : ℝ), x ∈ Ioo 0 (1 / Real.exp 1) → f' x < 0) →
     (∃ (I : Set ℝ), I = Ioo 0 (1 / Real.exp 1) ∧
      (∀ (x y : ℝ), x ∈ I → y ∈ I → x < y → f x > f y))) :=
begin
    sorry
end

end monotonic_decreasing_interval_of_f_l153_153213


namespace find_b_l153_153373

theorem find_b (b : ℚ) (m : ℚ) 
  (h1 : x^2 + b*x + 1/6 = (x + m)^2 + 1/18) 
  (h2 : b < 0) : 
  b = -2/3 := 
sorry

end find_b_l153_153373


namespace inequality_proof_l153_153190

variable (f : ℕ → ℕ → ℕ)

theorem inequality_proof :
  f 1 6 * f 2 5 * f 3 4 + f 1 5 * f 2 4 * f 3 6 + f 1 4 * f 2 6 * f 3 5 ≥
  f 1 6 * f 2 4 * f 3 5 + f 1 5 * f 2 6 * f 3 4 + f 1 4 * f 2 5 * f 3 6 :=
by sorry

end inequality_proof_l153_153190


namespace keychain_arrangements_l153_153131

-- Conditions modeled as parameters
def keys := {house, car, office, key1, key2, key3}

-- Conditions and Problem Statement as Lean Theorem Statement
theorem keychain_arrangements :
  ∀ (arrangements : finset (setoid {l : list keys // l.nodup})),
    (∀ a, a ∈ arrangements ↔ 
      (∃ l, a = ⟦l⟧ ∧ l.nodup ∧
      ((l.rotate 1 = l ∨ l.rotate 1 = list.reverse l) ))) ∧
    (∃ house_pos car_pos office_pos : ℕ,
      l.nth house_pos = house ∧
      ((l.nth ((house_pos + 1) % 6) = car ∧ l.nth ((house_pos + 5) % 6) ≠ office) ∨
       (l.nth ((house_pos + 5) % 6) = car ∧ l.nth ((house_pos + 1) % 6) ≠ office) )):
  arrangements.card = 48 := 
begin
  sorry
end

end keychain_arrangements_l153_153131


namespace sum_digits_increment_l153_153868

def sum_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

theorem sum_digits_increment (n : ℕ) (h : sum_digits n = 1365) : 
  sum_digits (n + 1) = 1360 :=
by
  sorry

end sum_digits_increment_l153_153868


namespace a_cubed_divisible_l153_153831

theorem a_cubed_divisible {a : ℤ} (h1 : 60 ≤ a) (h2 : a^3 ∣ 216000) : a = 60 :=
by {
   sorry
}

end a_cubed_divisible_l153_153831


namespace A_gains_2700_dollars_l153_153175

-- Define the initial worth of the home
def initial_worth (A : Type) [AddMonoid A] := 15000

-- Define the sell price of the house after a 20% profit
def sell_price (A : Type) [AddMonoid A] [HasSmul ℝ A] (x : A) := (1.2 : ℝ) • x

-- Define the buyback price after a 15% loss
def buyback_price (A : Type) [AddMonoid A] [HasSmul ℝ A] (x : A) := (0.85 : ℝ) • x

-- Assert the initial worth of the home
def initial_worth_A : ℕ := initial_worth ℕ

-- Assert the sell price from Mr. A to Mr. B
def sell_price_A_to_B : ℕ := sell_price ℕ initial_worth_A

-- Assert the buyback price from Mr. B to Mr. A
def buyback_price_B_to_A : ℕ := buyback_price ℕ sell_price_A_to_B

-- Define the net gain by Mr. A
def net_gain (A : Type) [AddMonoid A] [AddGroup A] (x y : A) := x - y

-- Calculate net gain when sold at sell_price_A_to_B and bought back at buyback_price_B_to_A
def net_gain_A (A : Type) [AddMonoid A] [AddGroup A] [AddCommGroup A] :=
  net_gain A sell_price_A_to_B buyback_price_B_to_A

-- The Lean statement
theorem A_gains_2700_dollars : net_gain_A ℤ = 2700 := by
  sorry

end A_gains_2700_dollars_l153_153175


namespace product_of_fractions_l153_153712

theorem product_of_fractions :
  (Finset.prod (Finset.range 835) (λ n, (n + 1 : ℚ) / (n + 2))) = (1 / 836 : ℚ) := 
sorry

end product_of_fractions_l153_153712


namespace pow_eq_of_pow_sub_eq_l153_153099

theorem pow_eq_of_pow_sub_eq (a : ℝ) (m n : ℕ) (h1 : a^m = 6) (h2 : a^(m-n) = 2) : a^n = 3 := 
by
  sorry

end pow_eq_of_pow_sub_eq_l153_153099


namespace pascal_triangle_47_l153_153090

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153090


namespace value_of_expression_l153_153609

def numerator : ℝ := 0.625 * 0.0729 * 28.9
def denominator : ℝ := 0.0017 * 0.025 * 8.1
def expression : ℝ := numerator / denominator

theorem value_of_expression :
  expression ≈ 3779.999275 := sorry

end value_of_expression_l153_153609


namespace problem_solution_l153_153434

theorem problem_solution (b : ℝ) (i : ℂ) (h : i^2 = -1) (h_cond : (2 - i) * (4 * i) = 4 + b * i) : 
  b = 8 := 
by 
  sorry

end problem_solution_l153_153434


namespace unique_positive_integer_l153_153722

def S (n : ℕ) : ℕ :=
  (finset.range n).sum (λ k, (k + 2) * 2^(k + 2))

theorem unique_positive_integer (n : ℕ) (h : S n = 2^(n + 11)) : n = 1025 :=
sorry

end unique_positive_integer_l153_153722


namespace inequality_holds_for_a_l153_153542

theorem inequality_holds_for_a (a : ℝ) : (∀ x ∈ set.Icc 0 2, exp x - x > a * x) → a < real.exp 1 - 1 :=
by
  assume h : ∀ x ∈ set.Icc 0 2, exp x - x > a * x
  sorry

end inequality_holds_for_a_l153_153542


namespace collinear_A_T_X_l153_153165

variables {A B C D T X : Point} (incircle : circle) (triangle : triangle)
           (hX : X = some_defined_point)
           (hT : T = diametrically_opposite_point D incircle)

theorem collinear_A_T_X : collinear {A, T, X} :=
by
  sorry

end collinear_A_T_X_l153_153165


namespace JoggerDifference_l153_153320

theorem JoggerDifference (tyson_joggers alexander_joggers christopher_joggers : ℕ)
  (h1 : christopher_joggers = 20 * tyson_joggers)
  (h2 : christopher_joggers = 80)
  (h3 : alexander_joggers = tyson_joggers + 22) :
  christopher_joggers - alexander_joggers = 54 := by
  sorry

end JoggerDifference_l153_153320


namespace ff_of_10_eq_2_l153_153797

noncomputable def f : ℝ → ℝ
| x => if x ≤ 1 then x^2 + 1 else Real.log x

theorem ff_of_10_eq_2 : f (f 10) = 2 :=
by
  sorry

end ff_of_10_eq_2_l153_153797


namespace lines_perpendicular_to_same_plane_are_parallel_l153_153871

-- Definitions related to lines and planes
variable (Line Plane : Type)
variable m n : Line
variable α : Plane

-- The perpendicular relation between lines and planes
variable (perp : Line → Plane → Prop)
-- The parallel relation between lines
variable (parallel : Line → Line → Prop)

-- The statement of the problem
theorem lines_perpendicular_to_same_plane_are_parallel :
  perp m α ∧ perp n α → parallel m n :=
by
  sorry

end lines_perpendicular_to_same_plane_are_parallel_l153_153871


namespace remainder_7n_mod_4_l153_153253

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l153_153253


namespace minimum_value_expression_l153_153358

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end minimum_value_expression_l153_153358


namespace eq_solution_count_l153_153729

theorem eq_solution_count :
  (∃ s : Finset ℕ, s = Finset.filter (λ x, x ≠ 4 ∧ x ≠ 16 ∧ x ≠ 36) (Finset.range 51) ∧ s.card = 47) :=
by 
  sorry

end eq_solution_count_l153_153729


namespace studentsScoringAbove130_l153_153111

noncomputable def normalDistributionAboveThreshold : ℝ → ℝ → ℝ → ℝ := sorry -- Assume a function P(X > a | X ~ N(mu, sigma^2))

theorem studentsScoringAbove130 :
  ∀ (mu sigma : ℝ) (n : ℕ),
  sigma > 0 →
  P (mu - sigma < X ∧ X ≤ mu + sigma) = 0.6826 →
  P (mu - 2 * sigma < X ∧ X ≤ mu + 2 * sigma) = 0.9544 →
  P (mu - 3 * sigma < X ∧ X ≤ mu + 3 * sigma) = 0.9974 →
  mu = 120 →
  sigma^2 = 100 →
  n = 48 →
  let probability_above_130 := normalDistributionAboveThreshold 120 10 130 in
  let expected_students := probability_above_130 * n in
  round(expected_students) = 8 :=
by
  sorry

end studentsScoringAbove130_l153_153111


namespace prime_factor_bound_l153_153221

def seq (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | 1 => 2
  | n + 2 => (a n+1)^2 + (List.prod (List.map a (List.range n+1)))^2

noncomputable def a : ℕ → ℕ := seq a

theorem prime_factor_bound (k : ℕ) (p : ℕ) (hpos : 0 < k) (hp : Prime p) (hpk : p ∣ (a k)) :
  p > 4 * (k - 1) :=
by
  sorry

end prime_factor_bound_l153_153221


namespace probability_at_least_9_heads_l153_153975

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153975


namespace total_paint_correct_l153_153177

def monday_paint : ℕ := 30

def tuesday_paint : ℕ := 2 * monday_ppaint

def wednesday_paint : ℕ := monday_paint / 2

def total_paint : ℕ := monday_paint + tuesday_paint + wednesday_paint

theorem total_paint_correct : total_paint = 105 := by
  sorry

end total_paint_correct_l153_153177


namespace tangent_lines_parallel_to_4x_minus_1_l153_153940

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 2

theorem tangent_lines_parallel_to_4x_minus_1 :
  ∃ (a b : ℝ), (f a = b ∧ 3 * a^2 + 1 = 4) → (b = 4 * a - 4 ∨ b = 4 * a) :=
by
  sorry

end tangent_lines_parallel_to_4x_minus_1_l153_153940


namespace vector_magnitude_square_magnitude_of_vectors_l153_153408

noncomputable def length_of_diagonal (s : ℝ) : ℝ :=
  Real.sqrt (s^2 + s^2)

theorem vector_magnitude_square (s : ℝ) (h : s = 1) :
  √((s * s) + (s * s)) = √2 :=
by
  rw [h, one_mul, one_mul, add_self_eq_mul_two, Real.sqrt_mul (by norm_num : 0 ≤ (2)) (by norm_num : 0 ≤ 1)],
  norm_num

theorem magnitude_of_vectors (s : ℝ) (h : s = 1) :
  let a := length_of_diagonal s in 2 * a = 2 * Real.sqrt 2 :=
by
  rw [h],
  have h1 := vector_magnitude_square s h,
  rw [length_of_diagonal, h1,
      Real.sqrt_two_mul_one],
  norm_num

end vector_magnitude_square_magnitude_of_vectors_l153_153408


namespace centerville_remaining_budget_l153_153944

def centerville_budget (total_budget : ℝ) : Prop :=
  (0.15 * total_budget = 3000) ∧
  ((total_budget - 3000 - (0.24 * total_budget)) = 12200)

theorem centerville_remaining_budget : 
  ∃ (total_budget : ℝ), centerville_budget total_budget :=
by
  use 20000
  unfold centerville_budget
  split
  · simp
  · ring
  sorry -- Finish the proof accordingly

end centerville_remaining_budget_l153_153944


namespace g_is_odd_l153_153473

def g (x : ℝ) : ℝ := (7^x - 1) / (7^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x :=
by
  intro x
  have h1 : 7^(-x) = 1 / 7^x := by sorry  -- We use properties of exponents
  calc
    g (-x) = (7^(-x) - 1) / (7^(-x) + 1) : by sorry
        ... = (1 / 7^x - 1) / (1 / 7^x + 1) : by rw [h1]
        ... = (-(7^x - 1)) / (7^x + 1) : by sorry
        ... = -((7^x - 1) / (7^x + 1)) : by sorry
        ... = - g x : by sorry

end g_is_odd_l153_153473


namespace bottles_not_placed_in_crate_l153_153478

-- Defining the constants based on the conditions
def bottles_per_crate : Nat := 12
def total_bottles : Nat := 130
def crates : Nat := 10

-- Theorem statement based on the question and the correct answer
theorem bottles_not_placed_in_crate :
  total_bottles - (bottles_per_crate * crates) = 10 :=
by
  -- Proof will be here
  sorry

end bottles_not_placed_in_crate_l153_153478


namespace square_of_binomial_l153_153591

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l153_153591


namespace cosine_function_symmetry_l153_153033

theorem cosine_function_symmetry (φ : ℝ) (h1 : φ ∈ Ioo 0 (π / 2))
  (hf : ∀ x, 2 * sin x * sin (x + 3 * φ) = -2 * sin (-x) * sin (-x + 3 * φ)) :
  ∃ k : ℤ, ∀ x, cos (2 * x - φ) = cos (2 * (x - (k * π / 2 + π / 12))) :=
by sorry

end cosine_function_symmetry_l153_153033


namespace trig_identity_l153_153554

theorem trig_identity :
  (sin (15 * (Real.pi / 180)) * sin (30 * (Real.pi / 180)) * sin (75 * (Real.pi / 180))) = 1 / 8 :=
by
  sorry

end trig_identity_l153_153554


namespace find_a_l153_153813

theorem find_a (a : ℝ) (A B : Set ℝ)
    (hA : A = {a^2, a + 1, -3})
    (hB : B = {a - 3, 2 * a - 1, a^2 + 1}) 
    (h : A ∩ B = {-3}) : a = -1 := by
  sorry

end find_a_l153_153813


namespace airplane_children_l153_153951

theorem airplane_children (total_passengers men women children : ℕ) 
    (h1 : total_passengers = 80) 
    (h2 : men = women) 
    (h3 : men = 30) 
    (h4 : total_passengers = men + women + children) : 
    children = 20 := 
by
    -- We need to show that the number of children is 20.
    sorry

end airplane_children_l153_153951


namespace non_real_root_modulus_gt_one_over_n_root_l153_153481

theorem non_real_root_modulus_gt_one_over_n_root (a : ℝ) (n : ℕ) (hn : 2 ≤ n) (z : ℂ)
  (hz : ¬ (z.im = 0)) (hroot : z ^ (n + 1) - z ^ 2 + a * z + 1 = 0) : abs z > 1 / real.nroot (↑n) :=
sorry

end non_real_root_modulus_gt_one_over_n_root_l153_153481


namespace least_number_to_subtract_from_724946_l153_153603

def divisible_by_10 (n : ℕ) : Prop :=
  n % 10 = 0

theorem least_number_to_subtract_from_724946 :
  ∃ k : ℕ, k = 6 ∧ divisible_by_10 (724946 - k) :=
by
  sorry

end least_number_to_subtract_from_724946_l153_153603


namespace alex_current_height_l153_153318

theorem alex_current_height 
  (required_height : ℕ := 54) 
  (natural_growth_per_month : ℚ := 1/3) 
  (hanging_growth_per_hour : ℚ := 1/12)
  (hours_hanging_per_month : ℚ := 2)
  (months_in_a_year : ℕ := 12) :
  let natural_growth_per_year := natural_growth_per_month * months_in_a_year
  let hanging_growth_per_year := hanging_growth_per_hour * hours_hanging_per_month * months_in_a_year
  let total_growth_per_year := natural_growth_per_year + hanging_growth_per_year
  let current_height := required_height - total_growth_per_year in
  current_height = 48 := by
  sorry

end alex_current_height_l153_153318


namespace race_victory_l153_153117

variable (distance : ℕ := 200)
variable (timeA : ℕ := 18)
variable (timeA_beats_B_by : ℕ := 7)

theorem race_victory : ∃ meters_beats_B : ℕ, meters_beats_B = 56 :=
by
  let speedA := distance / timeA
  let timeB := timeA + timeA_beats_B_by
  let speedB := distance / timeB
  let distanceB := speedB * timeA
  let meters_beats_B := distance - distanceB
  use meters_beats_B
  sorry

end race_victory_l153_153117


namespace pet_store_dogs_l153_153637

-- Define the given conditions as Lean definitions
def initial_dogs : ℕ := 2
def sunday_dogs : ℕ := 5
def monday_dogs : ℕ := 3

-- Define the total dogs calculation to use in the theorem
def total_dogs : ℕ := initial_dogs + sunday_dogs + monday_dogs

-- State the theorem
theorem pet_store_dogs : total_dogs = 10 := 
by
  -- Placeholder for the proof
  sorry

end pet_store_dogs_l153_153637


namespace sum_is_31_l153_153014

-- Definitions based on given conditions
def condition1 (t : ℝ) : Prop := (1 + Real.sin t) * (1 + Real.cos t) = 9 / 4
def condition2 (t : ℝ) (m n k : ℕ) : Prop := 
  (1 - Real.sin t) * (1 - Real.cos t) = m / n - Real.sqrt k
def positive_integers (m n k : ℕ) : Prop := 0 < m ∧ 0 < n ∧ 0 < k
def relatively_prime (m n : ℕ) : Prop := Nat.coprime m n

-- The target proposition to prove
theorem sum_is_31 (t : ℝ) (m n k : ℕ) :
  condition1 t →
  condition2 t m n k →
  positive_integers m n k →
  relatively_prime m n →
  k + m + n = 31 :=
begin
  sorry
end

end sum_is_31_l153_153014


namespace sum_of_interior_angles_l153_153142

theorem sum_of_interior_angles (ABC ABD : Type) [RegularPolygon ABC 8] [RegularPolygon ABD 3] :
  (interior_angle 8) + (interior_angle 3) = 195 :=
by
  sorry

end sum_of_interior_angles_l153_153142


namespace whitney_greatest_sets_l153_153272

-- Define the conditions: Whitney has 4 T-shirts and 20 buttons.
def num_tshirts := 4
def num_buttons := 20

-- The problem statement: Prove that the greatest number of sets Whitney can make is 4.
theorem whitney_greatest_sets : Nat.gcd num_tshirts num_buttons = 4 := by
  sorry

end whitney_greatest_sets_l153_153272


namespace coin_flip_heads_probability_l153_153961

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153961


namespace find_z_plus_one_over_y_l153_153198

variables (x y z : ℝ)

-- axiomatize the given conditions as definitions
def positive (a : ℝ) := a > 0
def C1 := positive x ∧ positive y ∧ positive z
def C2 := x * y * z = 1
def C3 := x + 1 / z = 4
def C4 := y + 1 / x = 30

-- theorem statement with given conditions and the assertion to be proved
theorem find_z_plus_one_over_y (h1 : C1) (h2 : C2) (h3 : C3) (h4 : C4) : z + 1 / y = 36 / 119 :=
by sorry

end find_z_plus_one_over_y_l153_153198


namespace first_term_of_geometric_series_l153_153694

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l153_153694


namespace AI_KD_meet_on_Gamma_l153_153168

variables {A B C I D K : Point}
variables {Γ : Circumcircle A B C}
variables (AB_AC : AB > AC)
variables (incenter_I : Incenter A B C I)
variables (contact_D : ContactPointIncircle D B C)
variables (K_on_Gamma : OnCircumcircle K A B C)
variables (right_angle_AKI : Angle AKI = 90)

theorem AI_KD_meet_on_Gamma (hAB_AC: AB > AC) (hI: Incenter A B C I)
  (hD: ContactPointIncircle D B C) (hK: OnCircumcircle K A B C)
  (hAKI: Angle AKI = 90) : IntersectsOnCircumcircle AI KD A B C := 
sorry

end AI_KD_meet_on_Gamma_l153_153168


namespace remainder_7n_mod_4_l153_153261

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l153_153261


namespace incorrect_conclusion_l153_153000

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

axiom a_gt_b_gt_0 : a > b ∧ b > 0
axiom ab_eq_10 : a * b = 10

theorem incorrect_conclusion : 
  ¬ (∃ a b : ℝ, a > b ∧ b > 0 ∧ a * b = 10 ∧ (∃ h : (log a / log b > 1), True)) := sorry

end incorrect_conclusion_l153_153000


namespace at_least_12_lyamziks_rowed_l153_153561

-- Define the lyamziks, their weights, and constraints
def LyamzikWeight1 : ℕ := 7
def LyamzikWeight2 : ℕ := 14
def LyamzikWeight3 : ℕ := 21
def LyamzikWeight4 : ℕ := 28
def totalLyamziks : ℕ := LyamzikWeight1 + LyamzikWeight2 + LyamzikWeight3 + LyamzikWeight4
def boatCapacity : ℕ := 10
def maxRowsPerLyamzik : ℕ := 2

-- Question to prove
theorem at_least_12_lyamziks_rowed : totalLyamziks ≥ 12 :=
  by sorry


end at_least_12_lyamziks_rowed_l153_153561


namespace range_my_function_l153_153239

noncomputable def my_function (x : ℝ) := (x^2 + 4 * x + 3) / (x + 2)

theorem range_my_function : 
  Set.range my_function = Set.univ := 
sorry

end range_my_function_l153_153239


namespace pascal_row_contains_prime_47_l153_153049

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153049


namespace triangle_max_area_l153_153387

noncomputable def max_area_triangle (a b c : ℝ) := 
  (1/2) * b * c * Real.sin (Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))

theorem triangle_max_area 
  (a b c : ℝ)
  (h_a : a = Real.sqrt 2)
  (h_bc : b^2 - c^2 = 6)
  : max_area_triangle a b c = Real.sqrt 2 :=
begin
  sorry
end

end triangle_max_area_l153_153387


namespace zack_travel_countries_l153_153275

theorem zack_travel_countries (G J P Z : ℕ) 
  (hG : G = 6)
  (hJ : J = G / 2)
  (hP : P = 3 * J)
  (hZ : Z = 2 * P) :
  Z = 18 := by
  sorry

end zack_travel_countries_l153_153275


namespace inequality_change_l153_153825

theorem inequality_change (a b : ℝ) (h : a < b) : -2 * a > -2 * b :=
sorry

end inequality_change_l153_153825


namespace exists_n_f_over_g_eq_2012_l153_153755

def is_perfect_square (d : ℕ) : Prop :=
  ∃ k : ℕ, k * k = d

def is_perfect_cube (d : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = d

def f (n : ℕ) : ℕ :=
  (finset.range n).filter (λ d, d ∣ n ∧ is_perfect_square d).card

def g (n : ℕ) : ℕ :=
  (finset.range n).filter (λ d, d ∣ n ∧ is_perfect_cube d).card

theorem exists_n_f_over_g_eq_2012 :
  ∃ n : ℕ, f n / g n = 2012 :=
sorry

end exists_n_f_over_g_eq_2012_l153_153755


namespace lemma2_l153_153420

noncomputable def f (x a b : ℝ) := |x + a| - |x - b|

lemma lemma1 {x : ℝ} : f x 1 2 > 2 ↔ x > 3 / 2 := 
sorry

theorem lemma2 {a b : ℝ} (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : ∀ x : ℝ, f x a b ≤ 3):
  1 / a + 2 / b = (1 / 3) * (3 + 2 * Real.sqrt 2) := 
sorry

end lemma2_l153_153420


namespace ellipse_equation_l153_153508

theorem ellipse_equation (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : ∃ (P : ℝ × ℝ), P = (0, -1) ∧ P.2^2 = b^2) 
  (h4 : ∃ (C2 : ℝ → ℝ → Prop), (∀ x y : ℝ, C2 x y ↔ x^2 + y^2 = 4) ∧ 2 * a = 4) :
  (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1) :=
by
  sorry

end ellipse_equation_l153_153508


namespace factor_transformation_option_C_l153_153204

theorem factor_transformation_option_C (y : ℝ) : 
  4 * y^2 - 4 * y + 1 = (2 * y - 1)^2 :=
sorry

end factor_transformation_option_C_l153_153204


namespace robin_made_more_cupcakes_l153_153369

theorem robin_made_more_cupcakes (initial final sold made: ℕ)
  (h1 : initial = 42)
  (h2 : sold = 22)
  (h3 : final = 59)
  (h4 : initial - sold + made = final) :
  made = 39 :=
  sorry

end robin_made_more_cupcakes_l153_153369


namespace solve_quadratic_1_solve_quadratic_2_l153_153522

theorem solve_quadratic_1 : 
  ∃ x1 x2 : ℝ, 
    2 * x1^2 + 4 * x1 - 1 = 0 ∧ 
    2 * x2^2 + 4 * x2 - 1 = 0 ∧ 
    x1 = -1 - sqrt 6 / 2 ∧ 
    x2 = -1 + sqrt 6 / 2 := 
sorry

theorem solve_quadratic_2 : 
  ∃ x1 x2 : ℝ, 
    (4 * (2 * x1 - 1)^2 = 9 * (x1 + 4)^2 ∧ 
    4 * (2 * x2 - 1)^2 = 9 * (x2 + 4)^2) ∧ 
    x1 = -8 / 11 ∧ 
    x2 = 16 / 5 := 
sorry

end solve_quadratic_1_solve_quadratic_2_l153_153522


namespace power_function_properties_l153_153110

-- Define the function and the necessary conditions
def is_symmetric_about_y_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(-x) = f(x)

def has_no_points_in_common_with_x_axis (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f(x) ≠ 0

-- Define the specific function
def power_function (x : ℝ) : ℝ := x ^ -2

-- State the theorem
theorem power_function_properties :
  is_symmetric_about_y_axis power_function ∧ has_no_points_in_common_with_x_axis power_function :=
by
  sorry

end power_function_properties_l153_153110


namespace jim_total_payment_l153_153861

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end jim_total_payment_l153_153861


namespace range_of_a_bisection_method_solution_l153_153032

noncomputable def f (a x : ℝ) : ℝ := a * x^3 - 2 * a * x + 3 * a - 4

theorem range_of_a (a : ℝ) (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f a x = 0) :
  (12 * (27 - 4 * Real.sqrt 6) / 211 ≤ a) ∧ (a ≤ 12 * (27 + 4 * Real.sqrt 6) / 211) :=
sorry

theorem bisection_method_solution (h : ∃ x : ℝ, -1 < x ∧ x < 1 ∧ f (32 / 17) x = 0) :
  ∃ x : ℝ, (0 < x) ∧ (x < 1) ∧ (|f (32 / 17) x| < 0.1) :=
sorry

end range_of_a_bisection_method_solution_l153_153032


namespace circle_intersection_proof_l153_153627

noncomputable def geometricSetup (k : circle) (B D C A E F: Point) (r : ℝ) (h1 : diameter k B D) 
    (h2 : smaller_circle_centered_at_B B r C A)
    (h3 : perpendicular C BD E)
    (h4 : circle_diameter_AD A D F)
    (h5 : perpendicular B BD F) : Prop :=
  BE = BF

theorem circle_intersection_proof (k : circle) (B D C A E F: Point) (r : ℝ)
  (h1 : diameter k B D) (h2 : smaller_circle_centered_at_B B r C A)
  (h3 : perpendicular C BD E) (h4 : circle_diameter_AD A D F)
  (h5 : perpendicular B BD F) :
  geometricSetup k B D C A E F r h1 h2 h3 h4 h5 :=
sorry

end circle_intersection_proof_l153_153627


namespace article_final_price_l153_153211

theorem article_final_price (list_price : ℝ) (first_discount : ℝ) (second_discount : ℝ) : 
  first_discount = 0.1 → 
  second_discount = 0.01999999999999997 → 
  list_price = 70 → 
  ∃ final_price, final_price = 61.74 := 
by {
  sorry
}

end article_final_price_l153_153211


namespace rectangle_diagonal_in_triangle_l153_153778

def is_isosceles_right_triangle (A B C : Type) (AB AC BC : ℝ) : Prop :=
  AB = AC ∧ AB^2 + AC^2 = BC^2

def calc_diagonal_length (EG EH : ℝ) : ℝ :=
  real.sqrt (EG^2 + EH^2)

theorem rectangle_diagonal_in_triangle
  (A B C E F G H : Type)
  (AB AC BC EG EH EF : ℝ)
  (isosceles_triangle : is_isosceles_right_triangle A B C AB AC BC)
  (EG_val : EG = 6) (EH_val : EH = 8) :
  EF = 10 :=
by {
  have diag := calc_diagonal_length EG EH,
  simp [calc_diagonal_length, EG_val, EH_val] at diag,
  have EF_val := @real.sqrt_unique 100 EF 10,
  simp,
  sorry
}

end rectangle_diagonal_in_triangle_l153_153778


namespace num_irrational_equals_two_l153_153214

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem num_irrational_equals_two :
  let nums := [real.sqrt 2, 1 / 4, real.pi, real.cbrt 8, -22 / 7, 32 / 99] in
  list.countp is_irrational nums = 2 :=
by
  -- Definitions of the provided numbers
  have h1 : is_irrational (real.sqrt 2), from sorry,
  have h2 : ¬is_irrational (1 / 4), from sorry,
  have h3 : is_irrational (real.pi), from sorry,
  have h4 : ¬is_irrational (real.cbrt 8), from sorry,
  have h5 : ¬is_irrational (-22 / 7), from sorry,
  have h6 : ¬is_irrational (32 / 99), from sorry,
  have irrationals := [real.sqrt 2, real.pi],
  have rationals := [1 / 4, real.cbrt 8, -22 / 7, 32 / 99],
  have nums := irrationals ++ rationals,
  show list.countp is_irrational nums = 2, from sorry

end num_irrational_equals_two_l153_153214


namespace determine_a_l153_153345

variables {f : ℝ → ℝ}

def is_even (f : ℝ → ℝ) (domain : Set ℝ) : Prop := 
  ∀ x ∈ domain, f (-x) = f x

def is_decreasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀⦃x y⦄, x ∈ I → y ∈ I → x < y → f y < f x

def condition (a : ℝ) : Prop :=
  f (1 - a) - f (1 - a^2) < 0

def range_of_a (a : ℝ) : Prop :=
  (0 < a ∧ a < 1) ∨ (1 < a ∧ a < Real.sqrt 3)

theorem determine_a (f : ℝ → ℝ) (h1 : is_even f (Ioo (-2) 2)) 
  (h2 : is_decreasing f (Ioo (-2) 0)) 
  (h3 : ∀ a, a ∈ Ioo (-2) 2 → condition a) :
  ∀ a, a ∈ Ioo (-2) 2 → range_of_a a :=
by sorry

end determine_a_l153_153345


namespace area_of_trapezoid_ABCD_l153_153654

noncomputable def trapezoid : Type := sorry

structure is_trapezoid (ABCD : trapezoid) : Prop :=
(parallel : ∃ AB CD, AB ∥ CD)
(intersect_at_M : ∃ M, ∃ AC BD, AC ∩ BD = {M} ∧ divides_into_four_parts AC BD)

structure trapezoid_triangle_areas (ABCD : trapezoid) (M : point) : Prop :=
(AMD_area : area (triangle AMD) = 8)
(DCM_area : area (triangle DCM) = 4)

theorem area_of_trapezoid_ABCD (ABCD : trapezoid) (M : point)
  (h₁ : is_trapezoid ABCD)
  (h₂ : trapezoid_triangle_areas ABCD M) :
  area ABCD = 36 :=
sorry

end area_of_trapezoid_ABCD_l153_153654


namespace cos_neg_2theta_l153_153824

theorem cos_neg_2theta (θ : ℝ) (h : Real.sin (Real.pi / 2 + θ) = 3 / 5) : Real.cos (-2 * θ) = -7 / 25 := 
by
  sorry

end cos_neg_2theta_l153_153824


namespace pavan_travel_time_l153_153186

theorem pavan_travel_time :
  ∃ T : ℝ, T = 15 ∧ (∀ D : ℝ, D = 409.0909090909091 →
    let time1 := (D / 2) / 30 in
    let time2 := (D / 2) / 25 in
    T = time1 + time2) :=
by
  use 15
  intros D hD
  rw hD
  let time1 := (409.0909090909091 / 2) / 30
  let time2 := (409.0909090909091 / 2) / 25
  have ht1 : time1 = 6.818181818181818 := by norm_num
  have ht2 : time2 = 8.181818181818182 := by norm_num
  rw [ht1, ht2]
  norm_num
  sorry

end pavan_travel_time_l153_153186


namespace period_f_decreasing_interval_f_minimum_value_f_l153_153767

open Real

def f (x : ℝ) : ℝ := (1 / 2) * cos x ^ 2 - (1 / 2) * sin x ^ 2 + 1 - sqrt 3 * sin x * cos x

theorem period_f : (∀ x, f (x + π) = f x) := 
sorry

theorem decreasing_interval_f : ∀ k : ℤ, ∀ x ∈ [k * π - π / 6, k * π + π / 3], f' x < 0 := 
sorry

theorem minimum_value_f : ∀ x ∈ [0, π / 2], f x ≥ 0 ∧ (f x = 0 ↔ x = π / 3) := 
sorry

end period_f_decreasing_interval_f_minimum_value_f_l153_153767


namespace max_marks_l153_153174

theorem max_marks (M S : ℕ) :
  (267 + 45 = 312) ∧ (312 = (45 * M) / 100) ∧ (292 + 38 = 330) ∧ (330 = (50 * S) / 100) →
  (M + S = 1354) :=
by
  sorry

end max_marks_l153_153174


namespace a_square_plus_one_over_a_square_l153_153018

theorem a_square_plus_one_over_a_square (a : ℝ) (h : a + 1/a = 5) : a^2 + 1/a^2 = 23 :=
by 
  sorry

end a_square_plus_one_over_a_square_l153_153018


namespace effective_market_value_after_three_years_l153_153544

def initialValue : ℝ := 8000
def depreciationRates : List ℝ := [0.30, 0.20, 0.15]
def maintenanceCosts : List ℝ := [500, 800, 1000]
def inflationRate : ℝ := 0.05

/-- Effective market value after three years given the conditions. --/
theorem effective_market_value_after_three_years :
  let year1Value := (initialValue * (1 - depreciationRates.head!)) - maintenanceCosts.head!
  let year1Inflated := year1Value * (1 + inflationRate)
  let year2Value := (year1Inflated * (1 - depreciationRates.nth! 1)) - maintenanceCosts.nth! 1
  let year2Inflated := year2Value * (1 + inflationRate)
  let year3Value := (year2Inflated * (1 - depreciationRates.nth! 2)) - maintenanceCosts.nth! 2
  let year3Inflated := year3Value * (1 + inflationRate)
  year3Inflated ≈ 2214.94 :=
sorry

end effective_market_value_after_three_years_l153_153544


namespace gwen_money_remaining_l153_153363

def gwen_money (initial : ℝ) (spent1 : ℝ) (earned : ℝ) (spent2 : ℝ) : ℝ :=
  initial - spent1 + earned - spent2

theorem gwen_money_remaining :
  gwen_money 5 3.25 1.5 0.7 = 2.55 :=
by
  sorry

end gwen_money_remaining_l153_153363


namespace math_quiz_scores_stability_l153_153455

theorem math_quiz_scores_stability :
  let avgA := (90 + 82 + 88 + 96 + 94) / 5
  let avgB := (94 + 86 + 88 + 90 + 92) / 5
  let varA := ((90 - avgA) ^ 2 + (82 - avgA) ^ 2 + (88 - avgA) ^ 2 + (96 - avgA) ^ 2 + (94 - avgA) ^ 2) / 5
  let varB := ((94 - avgB) ^ 2 + (86 - avgB) ^ 2 + (88 - avgB) ^ 2 + (90 - avgB) ^ 2 + (92 - avgB) ^ 2) / 5
  avgA = avgB ∧ varB < varA :=
by
  sorry

end math_quiz_scores_stability_l153_153455


namespace value_of_m_l153_153444

theorem value_of_m :
  (∃ m : ℤ, 2^2000 - 3 * 2^1998 + 5 * 2^1996 - 2^1995 = m * 2^1995) →
  ∃ m : ℤ, m = 17 :=
begin
  sorry,
end

end value_of_m_l153_153444


namespace factor_chain_properties_l153_153167

noncomputable def T (x : ℕ) : ℕ :=
sorry  -- Placeholder for the definition of T(x)

noncomputable def R (x : ℕ) : ℕ :=
sorry  -- Placeholder for the definition of R(x)

theorem factor_chain_properties (k m n : ℕ) :
  let x := 5^k * 31^m * 1990^n in
  T(x) = 3n + k + m ∧
  R(x) = (Nat.factorial (3n + k + m)) / ((Nat.factorial n)^2 * (Nat.factorial m) * (Nat.factorial (k + n))) :=
by
  sorry

end factor_chain_properties_l153_153167


namespace isosceles_triangle_area_ratio_l153_153461

theorem isosceles_triangle_area_ratio (x α : ℝ) (h0 : 0 < x) (hα : 0 < α ∧ α < π / 2) :
  let R := (x / (2 * sin α)) in
  let S_triangle := (x * x * sin (2 * α)) / 2 in
  let S_circle := π * R ^ 2 in
  (S_triangle / S_circle) = (2 * sin (2 * α) * sin α ^ 2) / π := by
  sorry

end isosceles_triangle_area_ratio_l153_153461


namespace root_in_interval_l153_153924

open Function

noncomputable def f (x : ℝ) : ℝ := 3^x + x - 2

theorem root_in_interval : ∃ ξ ∈ Ioo 0 1, f ξ = 0 := 
by {
  sorry
}

end root_in_interval_l153_153924


namespace probability_jerry_same_color_l153_153632

theorem probability_jerry_same_color :
  ∀ (a b c : ℕ), (a + b + c = 97) ∧ 
                 (12 * (a * (a-1) + b * (b-1) + c * (c-1)) = 5 * 97 * 96) 
                 → (a^2 + b^2 + c^2 = 3977) 
                 → (a^2 + b^2 + c^2) / (97^2) = 41 / 97 := 
by {
  intros a b c h1 h2 h3,
  have h_eq : a^2 + b^2 + c^2 = 3977, from h3,
  rw h_eq,
  norm_num,
}

end probability_jerry_same_color_l153_153632


namespace remainder_when_7n_divided_by_4_l153_153264

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l153_153264


namespace continuous_stripe_probability_is_3_16_l153_153734

-- Define the stripe orientation enumeration
inductive StripeOrientation
| diagonal
| straight

-- Define the face enumeration
inductive Face
| front
| back
| left
| right
| top
| bottom

-- Total number of stripe combinations (2^6 for each face having 2 orientations)
def total_combinations : ℕ := 2^6

-- Number of combinations for continuous stripes along length, width, and height
def length_combinations : ℕ := 2^2 -- 4 combinations
def width_combinations : ℕ := 2^2  -- 4 combinations
def height_combinations : ℕ := 2^2 -- 4 combinations

-- Total number of continuous stripe combinations across all dimensions
def total_continuous_stripe_combinations : ℕ :=
  length_combinations + width_combinations + height_combinations

-- Probability calculation
def continuous_stripe_probability : ℚ :=
  total_continuous_stripe_combinations / total_combinations

-- Final theorem statement
theorem continuous_stripe_probability_is_3_16 :
  continuous_stripe_probability = 3 / 16 :=
by
  sorry

end continuous_stripe_probability_is_3_16_l153_153734


namespace determine_principal_amount_l153_153311

-- given conditions: 
-- original simple interest
def SI (P R : ℝ) : ℝ := (P * R * 5) / 100

-- new simple interest with rate 5% higher
def SI' (P R : ℝ) : ℝ := (P * (R + 5) * 5) / 100

-- interest difference condition
def condition (P R : ℝ) : Prop := SI' P R = SI P R + 250

-- statement to prove
theorem determine_principal_amount (R : ℝ) : 
  ∃ P : ℝ, condition P R ∧ P = 1000 := 
by
  sorry

end determine_principal_amount_l153_153311


namespace distribute_books_l153_153841

theorem distribute_books :
    ∃ (n : ℕ), n = 220 ∧ (∃ (students books : ℕ), students = 12 ∧ books = 3 ∧ binomial students books = 220) := 
begin
  sorry
end

end distribute_books_l153_153841


namespace purely_imaginary_z_z_in_third_quadrant_l153_153378

-- Definitions based on conditions
def z (m : ℝ) : ℂ := ((1+complex.I)*m^2 - m*complex.I - 1 - 2*complex.I)

-- Prove (Ⅰ): For what value of the real number \( m \) is the complex number \( z \) purely imaginary?
theorem purely_imaginary_z (m : ℝ) : (z m).re = 0 ↔ m = 1 := by
  sorry

-- Prove (Ⅱ): If the complex number \( z \) corresponds to a point in the third quadrant on the complex plane, find the range of values for the real number \( m \).
theorem z_in_third_quadrant (m : ℝ) : ((z m).re < 0) ∧ ((z m).im < 0) ↔ -1 < m ∧ m < 1 := by
  sorry

end purely_imaginary_z_z_in_third_quadrant_l153_153378


namespace ivan_receives_amount_l153_153853

def initial_deposit : ℕ := 100000

def compensation_limit : ℕ := 1400000

def insurable_event : Prop := true

def bank_participant_in_insurance : Prop := true

theorem ivan_receives_amount (d : ℕ) (bank_insured : Prop) (insurable : Prop) (comp_limit : ℕ) :
  d = initial_deposit →
  bank_insured →
  insurable →
  comp_limit = compensation_limit →
  (d ≤ comp_limit) →
  (∃ accrued_interest : ℕ, (d + accrued_interest) ≤ comp_limit) → 
  d + (some accrued_interest) ≤ compensation_limit :=
by
  intros h_initial h_insured h_insurable h_limit h_d_le_limit h_exists_interest,
  sorry

end ivan_receives_amount_l153_153853


namespace perpendicular_a_b_l153_153040

def a : ℝ × ℝ := (1, 3)
def b (m : ℝ) : ℝ × ℝ := (-2, m)

noncomputable def dot_product (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

theorem perpendicular_a_b (m : ℝ) (h : dot_product a (a.1 + 2 * b m.1, a.2 + 2 * b m.2) = 0) : m = -1 :=
sorry

end perpendicular_a_b_l153_153040


namespace max_difference_exists_l153_153661

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l153_153661


namespace total_cost_correct_l153_153292

variables (gravel_cost_per_ton : ℝ) (gravel_tons : ℝ)
variables (sand_cost_per_ton : ℝ) (sand_tons : ℝ)
variables (cement_cost_per_ton : ℝ) (cement_tons : ℝ)

noncomputable def total_cost : ℝ :=
  (gravel_cost_per_ton * gravel_tons) + (sand_cost_per_ton * sand_tons) + (cement_cost_per_ton * cement_tons)

theorem total_cost_correct :
  gravel_cost_per_ton = 30.5 → gravel_tons = 5.91 →
  sand_cost_per_ton = 40.5 → sand_tons = 8.11 →
  cement_cost_per_ton = 55.6 → cement_tons = 4.35 →
  total_cost gravel_cost_per_ton gravel_tons sand_cost_per_ton sand_tons cement_cost_per_ton cement_tons = 750.57 :=
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3, h4, h5, h6]
  norm_num
  sorry

end total_cost_correct_l153_153292


namespace probability_at_least_9_heads_l153_153980

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153980


namespace orthocentric_parallelepiped_isohedral_parallelepiped_l153_153234

structure Tetrahedron :=
(A B C D : ℝ × ℝ × ℝ)

structure Parallelepiped :=
(A B C D E F G H : ℝ × ℝ × ℝ)

def is_orthocentric (t : Tetrahedron) : Prop := sorry -- define orthocentric property

def is_isohedral (t : Tetrahedron) : Prop := sorry -- define isohedral property

def forms_parallelepiped (t : Tetrahedron) (p : Parallelepiped) : Prop := sorry -- define the condition where parallelepiped is formed 

def all_edges_equal (p : Parallelepiped) : Prop := sorry -- define property where all edges are equal

def is_rectangular (p : Parallelepiped) : Prop := sorry -- define property where parallelepiped is rectangular

theorem orthocentric_parallelepiped (t : Tetrahedron) (p : Parallelepiped) : 
  is_orthocentric(t) → forms_parallelepiped(t, p) → all_edges_equal(p) := 
sorry

theorem isohedral_parallelepiped (t : Tetrahedron) (p : Parallelepiped) : 
  is_isohedral(t) → forms_parallelepiped(t, p) → is_rectangular(p) := 
sorry

end orthocentric_parallelepiped_isohedral_parallelepiped_l153_153234


namespace general_formula_for_a_no_geometric_sequence_l153_153774

noncomputable theory

-- Define the sequence {a_n} based on the given conditions
def a (n : ℕ) : ℕ :=
  if n = 0 then 0 else n

-- Prove the general formula for the sequence {a_n}
theorem general_formula_for_a (n : ℕ) (hn : n ≥ 1) : 
  a n = n := 
sorry

-- Define b_n as ln(a_n)
def b (n : ℕ) : ℝ :=
  if n = 0 then 0 else Real.log (a n)

-- Prove there does not exist k ≥ 2 such that b_k, b_{k+1}, b_{k+2} form a geometric sequence
theorem no_geometric_sequence (k : ℕ) (hk : k ≥ 2) : 
  ¬(b k * b (k + 2) = b (k + 1) * b (k + 1)) := 
sorry

end general_formula_for_a_no_geometric_sequence_l153_153774


namespace work_done_in_isothermal_is_equal_to_heat_added_l153_153183

-- First, define the ideal gas and process conditions
def n : ℕ := 1                      -- Number of moles (one mole)
def W₁ : ℝ := 30                    -- Work done in the first process (Joules)
def R : ℝ := 8.314                  -- Ideal gas constant (J/(mol·K))
def P : ℝ := 101325                -- An arbitrary pressure for the ideal gas (Pa)

-- State the first and second law of thermodynamics conditions
def ∆V : ℝ := 0.000295              -- Change in volume, derived arbitrarily, not provided in problem
def ∆T : ℝ := W₁ / P                -- Change of temperature calculated from the ideal gas law relation for isobaric process

def Q₁ : ℝ := W₁ + (3 / 2) * n * R * ∆T  -- Heat added in the first process

-- State the equality for the second process
theorem work_done_in_isothermal_is_equal_to_heat_added : Q₁ = 75 →  W₂ = 75 :=
by
  sorry

end work_done_in_isothermal_is_equal_to_heat_added_l153_153183


namespace part1_part2_l153_153419

theorem part1 (a : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x + |x - 1| ≥ 2) → (a ≤ 0 ∨ a ≥ 4) :=
by sorry

theorem part2 (a : ℝ) (f : ℝ → ℝ) (h_a : a < 2) (h_f : ∀ x, f x = |2 * x - a| + |x - 1|) :
  (∀ x, f x ≥ a - 1) → (a = 4 / 3) :=
by sorry

end part1_part2_l153_153419


namespace integral_x_sub_sin_x_l153_153351

open Real

theorem integral_x_sub_sin_x :
  ∫x in 0..π, (x - sin x) = (π^2) / 2 - 2 :=
by
  sorry

end integral_x_sub_sin_x_l153_153351


namespace work_done_in_isothermal_process_l153_153184

-- One mole of an ideal monatomic gas (n=1)
-- Work done in the first isobaric process (W1)
def n : ℕ := 1
def W1 : ℝ := 30

-- The problem states the gas receives the same heat in the second process (Q2 = Q1)
-- We need to prove that the work done in the second (isothermal) process W2 is 75 J
theorem work_done_in_isothermal_process :
  (W2 = 75) :=
by
  -- Hypothesis: heat added in the first process (Q1) is equal to the heat added in the second process (Q2)
  let Q1 := W1 + (3 / 2) * n * R * ΔT
  let Q2 := Q1
  -- Since the second process is isothermal, the work done in the second process W2 is exactly Q2
  let W2 := Q2
  -- Therefore, W2 = 75 holds under the given conditions
  sorry

end work_done_in_isothermal_process_l153_153184


namespace margo_travel_distance_l153_153171

/-- Let t_bike be the time in hours Margo spends biking,
      t_walk be the time in hours Margo spends walking,
      avg_speed be the average speed in miles per hour for the entire trip,
      and total_distance be the total distance in miles Margo traveled. -/
def t_bike : ℝ := 15 / 60  -- 15 minutes in hours
def t_walk : ℝ := 25 / 60  -- 25 minutes in hours
def avg_speed : ℝ := 6     -- average speed in miles per hour

/--
  Given the above definitions, prove that the total distance Margo traveled
  is equal to 4 miles.
-/
theorem margo_travel_distance : avg_speed * (t_bike + t_walk) = 4 :=
by sorry

end margo_travel_distance_l153_153171


namespace ratio_side_lengths_sum_l153_153217

theorem ratio_side_lengths_sum :
  let area_ratio := (250 : ℝ) / 98
  let side_length_ratio := Real.sqrt area_ratio
  ∃ a b c : ℕ, side_length_ratio = a * Real.sqrt b / c ∧ a + b + c = 17 :=
by
  let area_ratio := (250 : ℝ) / 98
  let side_length_ratio := Real.sqrt area_ratio
  use 5, 5, 7
  split
  {
    sorry -- Proof that side_length_ratio = 5 * Real.sqrt 5 / 7
  }
  {
    refl -- Proof that 5 + 5 + 7 = 17
  }

end ratio_side_lengths_sum_l153_153217


namespace d_alembert_theorem_l153_153427

-- Definition of centers and radii conditions
variables {O₁ O₂ O₃ : Point} -- Centers of the circles
variables {r₁ r₂ r₃ : ℝ} -- Radii of the circles

-- Conditions: the centers are not collinear and radii are different
axiom centers_not_collinear : ¬ collinear O₁ O₂ O₃
axiom radii_distinct : r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₂ ≠ r₃

-- Homothety centers
axiom homothety_centers : exists S₁ S₁' S₂ S₂' S₃ S₃', 
  (homothety_center O₁ O₂ r₁ r₂ S₁ ∧ 
   homothety_center O₂ O₁ r₂ r₁ S₁' ∧ 
   homothety_center O₁ O₃ r₁ r₃ S₂ ∧ 
   homothety_center O₃ O₁ r₃ r₁ S₂' ∧ 
   homothety_center O₂ O₃ r₂ r₃ S₃ ∧ 
   homothety_center O₃ O₂ r₃ r₂ S₃')

-- Proof that homothety centers satisfy D'Alembert's theorem
theorem d_alembert_theorem : 
  ∃ S₁ S₁' S₂ S₂' S₃ S₃', 
  (homothety_center O₁ O₂ r₁ r₂ S₁ ∧ 
   homothety_center O₂ O₁ r₂ r₁ S₁' ∧ 
   homothety_center O₁ O₃ r₁ r₃ S₂ ∧ 
   homothety_center O₃ O₁ r₃ r₁ S₂' ∧ 
   homothety_center O₂ O₃ r₂ r₃ S₃ ∧ 
   homothety_center O₃ O₂ r₃ r₂ S₃') ∧
  (collinear S₁ S₂ S₃ ∧ 
   collinear S₁ S₂' S₃' ∧ 
   collinear S₂ S₁' S₃ ∧ 
   collinear S₃ S₁' S₂') :=
sorry

end d_alembert_theorem_l153_153427


namespace train_speed_in_km_per_hr_l153_153618

-- Definitions based on the conditions
def train_length : ℝ := 415
def tunnel_length : ℝ := 285
def time_crossing_tunnel : ℝ := 40

-- The proof statement
theorem train_speed_in_km_per_hr :
  let total_distance := train_length + tunnel_length in
  let speed_m_per_s := total_distance / time_crossing_tunnel in
  let speed_km_per_hr := speed_m_per_s * 3.6 in
  speed_km_per_hr = 63 :=
by
  -- include sorry to skip proof, as per instructions
  sorry

end train_speed_in_km_per_hr_l153_153618


namespace birds_in_store_l153_153301

/-- 
A pet store had a total of 180 animals, consisting of birds, dogs, and cats. 
Among the birds, 64 talked, and 13 didn't. If there were 40 dogs in the store 
and the number of birds that talked was four times the number of cats, 
prove that there were 124 birds in total.
-/
theorem birds_in_store (total_animals : ℕ) (talking_birds : ℕ) (non_talking_birds : ℕ) 
  (dogs : ℕ) (cats : ℕ) 
  (h1 : total_animals = 180)
  (h2 : talking_birds = 64)
  (h3 : non_talking_birds = 13)
  (h4 : dogs = 40)
  (h5 : talking_birds = 4 * cats) : 
  talking_birds + non_talking_birds + dogs + cats = 180 ∧ 
  talking_birds + non_talking_birds = 124 :=
by
  -- We are skipping the proof itself and focusing on the theorem statement
  sorry

end birds_in_store_l153_153301


namespace circle_equation_and_intersections_l153_153005

theorem circle_equation_and_intersections :
  (∃ C : ℝ × ℝ, ∃ r : ℝ, ∃ m : ℝ, 
      C = (m, 0) ∧
      r = 5 ∧
      (∀ x y : ℝ, line_eq x y = 0 → chord_length = 2 * sqrt(17)) ∧
      (line_eq x y = x - y + 3) ∧
      chord_length = abs(m + 3) / sqrt(2) ∧
      abs(m + 3) = sqrt(25 - 17) ∧ m > 0 ∧ m = 1 ∧
      circle_eq x y = (x - 1)^2 + y^2 = 25 ∧

      -- For Line 'ax - y + 5 = 0'
      ( ∃ a : ℝ, 
          ((abs(a + 5) / sqrt(a^2 + 1)) < 5) ∧ (12 * a^2 - 5 * a > 0) ∧ 
          (a < 0 ∨ a > 5 / 12)) ∧

      -- For symmetry condition with respect to line passing through (-2, 4)
      ( ∃ a : ℝ, 
          ((∃ P : ℝ × ℝ, P = (-2, 4)) ∧ 
          (line_eq PA PB) = 0 ∧ 
          (PC ⟂ AB) ∧ 
          ((a < 0 ∨ a > 5 / 12) ∧ (a * (- 4 / 3) = - 1))) ∧
          a = 3 / 4)
  )
 := sorry

end circle_equation_and_intersections_l153_153005


namespace find_incorrect_number_l153_153201

-- Define conditions
variables (incorrect_avg correct_avg : ℕ) (count : ℕ) (incorrect_read correct_read : ℕ)
variables (incorrect_sum correct_sum diff : ℕ)

-- Given conditions
def given_conditions : Prop := 
  incorrect_avg = 16 ∧ 
  correct_avg = 18 ∧ 
  count = 10 ∧ 
  correct_read = 46 ∧ 
  incorrect_sum = incorrect_avg * count ∧ 
  correct_sum = correct_avg * count ∧ 
  diff = correct_sum - incorrect_sum

-- The final goal (correct answer)
def final_goal : Prop :=
  ∃ (x: ℕ), correct_read - diff = x ∧ x = 26

-- Main theorem statement
theorem find_incorrect_number (h : given_conditions) : final_goal :=
by
  -- proof steps would go here
  sorry

end find_incorrect_number_l153_153201


namespace larger_ball_radius_l153_153548

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

theorem larger_ball_radius :
  let small_ball_radius := 2
  let small_ball_volume := volume_of_sphere small_ball_radius
  let total_volume_small_balls := 6 * small_ball_volume
  let large_ball_volume := total_volume_small_balls
  let R := (large_ball_volume * 3 / (4 * Real.pi))^(1/3)
  in R = Real.cbrt 48 :=
by
  sorry

end larger_ball_radius_l153_153548


namespace suitcase_lock_possibilities_l153_153648

theorem suitcase_lock_possibilities :
  let digits := Finset.range 10
  let possibilities := (Finset.card digits) * (Finset.card (digits.erase 0)) * 
                       (Finset.card (digits.erase 0).erase 1) * 
                       (Finset.card (digits.erase 0).erase 1).erase 2 in
  possibilities = 5040 :=
by
  sorry

end suitcase_lock_possibilities_l153_153648


namespace problem_part1_problem_part2_l153_153781

variable {θ m : ℝ}
variable {h₀ : θ ∈ Ioo 0 (Real.pi / 2)}
variable {h₁ : Real.sin θ + Real.cos θ = (Real.sqrt 3 + 1) / 2}
variable {h₂ : Real.sin θ * Real.cos θ = m / 2}

theorem problem_part1 :
  (Real.sin θ / (1 - 1 / Real.tan θ) + Real.cos θ / (1 - Real.tan θ)) = (Real.sqrt 3 + 1) / 2 :=
sorry

theorem problem_part2 :
  m = Real.sqrt 3 / 2 ∧ (θ = Real.pi / 6 ∨ θ = Real.pi / 3) :=
sorry

end problem_part1_problem_part2_l153_153781


namespace least_number_divisible_l153_153926

theorem least_number_divisible (n : ℕ) :
  ((∀ d ∈ [24, 32, 36, 54, 72, 81, 100], (n + 21) % d = 0) ↔ n = 64779) :=
sorry

end least_number_divisible_l153_153926


namespace cakes_baker_made_initially_l153_153709

theorem cakes_baker_made_initially (x : ℕ) (h1 : x - 75 + 76 = 111) : x = 110 :=
by
  sorry

end cakes_baker_made_initially_l153_153709


namespace equilateral_triangle_rot_area_l153_153114

theorem equilateral_triangle_rot_area (a : ℝ) (h : (3 * 3.14 - 4) ≠ 0) : 
    let S : ℝ := 18 in
    ∃ a, (S = 24.39 * 8 / (3 * 3.14 - 4)) →
    S = 18 :=
by
  intros
  let S := (sqrt 3 / 4) * (a ^ 2)
  have h1 : (3 * 3.14 - 4) ≠ 0 := sorry
  let a := sqrt (24.39 * 8 / (3 * 3.14 - 4))
  have h2 : sqrt ((3 * 3.14 - 4) / 4) * sqrt a ^ 2 = 18 := sorry
  show S = 18 from sorry

end equilateral_triangle_rot_area_l153_153114


namespace largest_cos_a_l153_153487

theorem largest_cos_a (a b c : ℝ) (h1 : Real.sin a = Real.cot b) (h2 : Real.sin b = Real.cot c) (h3 : Real.sin c = Real.cot a) : 
  Real.cos a ≤ Real.sqrt ((3 - Real.sqrt 5) / 2) := 
sorry

end largest_cos_a_l153_153487


namespace pascal_triangle_47_rows_l153_153085

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153085


namespace largest_diff_even_digits_l153_153668

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l153_153668


namespace sphere_surface_area_l153_153180

-- Define the conditions
def points_on_sphere (A B C : Type) := 
  ∃ (AB BC AC : Real), AB = 6 ∧ BC = 8 ∧ AC = 10

-- Define the distance condition
def distance_condition (R : Real) := 
  ∃ (d : Real), d = R / 2

-- Define the main theorem
theorem sphere_surface_area 
  (A B C : Type) 
  (h_points : points_on_sphere A B C) 
  (h_distance : ∃ R : Real, distance_condition R) : 
  4 * Real.pi * (10 / 3 * Real.sqrt 3) ^ 2 = 400 / 3 * Real.pi := 
by 
  sorry

end sphere_surface_area_l153_153180


namespace max_difference_evens_l153_153677

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l153_153677


namespace difference_in_interest_rates_l153_153312

variable (R D : ℝ)
variable (sum time : ℝ) (extra_profit : ℝ)

-- Given conditions
def initial_sum := 2500
def investment_time := 5
def additional_interest := 250
def original_interest := sum * R * time / 100
def higher_interest := sum * (R + D) * time / 100

-- Lean 4 statement
theorem difference_in_interest_rates :
  sum = initial_sum →
  time = investment_time →
  extra_profit = additional_interest →
  higher_interest - original_interest = extra_profit →
  D = 0.4 := by
  sorry

end difference_in_interest_rates_l153_153312


namespace convex_hull_regular_ngons_l153_153885

theorem convex_hull_regular_ngons (n : ℕ) (hn : n ≥ 3)
    (polys : list (fin n → ℝ × ℝ))
    (hregular : ∀ p ∈ polys, regular_polygon n p)
    (convex_hull_vertices : fin n → ℝ × ℝ → Prop) :
    ∀ (m : ℕ), convex_hull_vertices = convex_hull (⋃ p ∈ polys, vertices p) →
    m = finset.card (vertices_of_convex_hull convex_hull_vertices) →
    m ≥ n :=
by
  sorry

end convex_hull_regular_ngons_l153_153885


namespace regain_original_wage_l153_153445

-- Define the initial conditions and parameters.
variable {W : ℝ}

-- Define what it means to have a 30% wage cut
def new_wage (W : ℝ) : ℝ := 0.7 * W

-- Define the required raise percentage to regain the original pay.
def required_raise_percentage (W : ℝ) : ℝ := (W / (new_wage W) - 1) * 100

-- Theorem stating the required raise to regain the original wage is 42.857%.
theorem regain_original_wage (W : ℝ) : required_raise_percentage W = 42.857 :=
by
  sorry

end regain_original_wage_l153_153445


namespace close_200_cities_l153_153456

structure Graph :=
(vertices : ℕ)
(edges : ℕ)
(adj : vertices → set vertices)
(degree : ∀ v : ℕ, (adj v).card = 3)
(connected : ∀ v1 v2 : ℕ, ∃ path : list ℕ, (path.head = v1) ∧ (path.last = v2) ∧ (∀ (i : ℕ) (h : i < path.length - 1), (path.nth_le i h) ∈ (adj (path.nth_le (i + 1) (by linarith)))))

theorem close_200_cities (G : Graph) (h : G.vertices = 1998) :
  ∃ (closed : finset ℕ), closed.card = 200 ∧
  (∀ v ∈ closed, ∀ w ∈ closed, w ∉ G.adj v) ∧
  (∀ v ∉ closed, ∀ u ∉ closed, ∃ path : list ℕ, (path.head = v) ∧ (path.last = u) ∧ (∀ (i : ℕ) (h : i < path.length - 1), (path.nth_le i h) ∉ closed)) :=
sorry

end close_200_cities_l153_153456


namespace geometric_series_first_term_l153_153697

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l153_153697


namespace line_y_intercept_l153_153310

theorem line_y_intercept (x1 y1 x2 y2 : ℝ) (h1 : x1 = 2) (h2 : y1 = -3) (h3 : x2 = 6) (h4 : y2 = 9) :
  ∃ b : ℝ, b = -9 := 
by
  sorry

end line_y_intercept_l153_153310


namespace square_of_binomial_l153_153581

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l153_153581


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153080

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153080


namespace probability_of_train_present_l153_153651

noncomputable def probability_train_present (train_arrival susan_arrival : ℝ) : ℝ :=
if train_arrival <= 90 then
  if susan_arrival <= train_arrival + 30 then
    1
  else
    0
else
  if susan_arrival <= 120 then
    if susan_arrival >= train_arrival then
      1
    else
      0
  else
    0

theorem probability_of_train_present : 
  (∫ t in 0..120, ∫ s in 0..120, probability_train_present t s) / 14400 = 7 / 32 :=
by
  sorry

end probability_of_train_present_l153_153651


namespace renumber_points_acute_right_angle_l153_153013

/-- 
  Given points A and points B on the plane, 
  prove that the points B_i can be renumbered 
  such that for all i, j, 
  the angle between vectors 
  (A_i - A_j) and (B_i - B_j) is either acute or right.
-/
theorem renumber_points_acute_right_angle 
  (n : ℕ)
  (A B : fin n → ℝ × ℝ) :
  ∃ (perm : equiv.perm (fin n)), 
    ∀ i j : fin n,
    0 ≤ ((A i).1 - (A j).1) * ((B (perm i)).1 - (B (perm j)).1) + 
        ((A i).2 - (A j).2) * ((B (perm i)).2 - (B (perm j)).2) := 
sorry

end renumber_points_acute_right_angle_l153_153013


namespace intersection_value_l153_153465

-- Definitions from conditions
def line_parametric (t : ℝ) (α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

def curve_polar (θ : ℝ) (p : ℝ) : ℝ :=
  p / (1 - Real.cos θ)

-- Proving the required value
theorem intersection_value (t : ℝ) (α : ℝ) (p : ℝ)
  (hα : 0 < α ∧ α < π) (hp : 0 < p) :
  let OA := p / (1 - Real.cos α),
      OB := p / (1 + Real.cos α)
  in 1 / OA + 1 / OB = 2 / p :=
by
  -- Sorry to indicate where the proof should go but is not necessary for this task
  sorry

end intersection_value_l153_153465


namespace solution_count_l153_153731

noncomputable def equation (x : ℝ) : ℝ := 3 * real.sin x ^ 3 - 10 * real.sin x ^ 2 + 3 * real.sin x

def in_range (x : ℝ) : Prop := 0 ≤ x ∧ x ≤ 2 * real.pi

theorem solution_count : 
  {x | in_range x ∧ equation x = 0}.card = 7 := 
sorry

end solution_count_l153_153731


namespace perimeter_non_shaded_region_l153_153533

variables (area_shaded total_area : ℝ)
variables (d1 d2 d3 d4 d5 : ℝ)
variables (angles_right : Prop)

-- Conditions
def conditions : Prop :=
  area_shaded = 120 ∧
  total_area = (10 * 12) + (3 * 5) ∧
  angles_right ∧
  d1 = 3 ∧ d2 = 5 ∧ d3 = 2 ∧ d4 = 12 ∧ d5 = 10

-- Question proving that the perimeter of the non-shaded region is 16 inches
theorem perimeter_non_shaded_region (h : conditions) : 
  2 * (d1 + d2) = 16 :=
by { sorry }

end perimeter_non_shaded_region_l153_153533


namespace pascal_triangle_47_rows_l153_153084

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153084


namespace value_of_z_l153_153102

theorem value_of_z {x y z : ℤ} (h1 : x = 2) (h2 : y = x^2 - 5) (h3 : z = y^2 - 5) : z = -4 := by
  sorry

end value_of_z_l153_153102


namespace pascal_row_contains_prime_47_l153_153052

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153052


namespace pyramid_volume_is_sqrt3_l153_153381

noncomputable def volume_of_pyramid := 
  let base_area : ℝ := 2 * Real.sqrt 3
  let angle_ABC : ℝ := 60
  let BC := 2
  let EC := BC
  let FB := BC / 2
  let height : ℝ := Real.sqrt 3
  let pyramid_volume := 1/3 * EC * FB * height
  pyramid_volume

theorem pyramid_volume_is_sqrt3 : volume_of_pyramid = Real.sqrt 3 :=
by sorry

end pyramid_volume_is_sqrt3_l153_153381


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153991

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153991


namespace expansion_binomial_coeff_no_constant_term_l153_153023

theorem expansion_binomial_coeff (n : ℕ) (x : ℝ) :
  (2 * (n.choose 2) = (n.choose 1) + (n.choose 3)) →
  n = 7 :=
by
  sorry

theorem no_constant_term (x : ℝ) :
  let n := 7 in
  ∀ r : ℕ, ((n - 2 * r) / 2 : ℝ) ≠ 0 :=
by
  sorry

end expansion_binomial_coeff_no_constant_term_l153_153023


namespace tangent_line_at_origin_is_y_eq_x_l153_153327

noncomputable def f (x a : ℝ) : ℝ := x^3 + (a-1)*x^2 + a*x

theorem tangent_line_at_origin_is_y_eq_x (a : ℝ) (h : ∀ x : ℝ, f (-x) a = -f x a) :
  ∃ m, m = 1 ∧ ∀ x : ℝ, y : ℝ, y = m * x → y = x :=
by
  sorry

end tangent_line_at_origin_is_y_eq_x_l153_153327


namespace pascal_triangle_47_l153_153093

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153093


namespace probability_heads_at_least_9_l153_153970

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153970


namespace exists_centrally_symmetric_inscribed_convex_hexagon_l153_153446

-- Definition of a convex polygon with vertices
def convex_polygon (W : Type) : Prop := sorry

-- Definition of the unit area condition
def has_unit_area (W : Type) : Prop := sorry

-- Definition of being centrally symmetric
def is_centrally_symmetric (V : Type) : Prop := sorry

-- Definition of being inscribed
def is_inscribed_polygon (V W : Type) : Prop := sorry

-- Definition of a convex hexagon
def convex_hexagon (V : Type) : Prop := sorry

-- Main theorem statement
theorem exists_centrally_symmetric_inscribed_convex_hexagon (W : Type) 
  (hW_convex : convex_polygon W) (hW_area : has_unit_area W) : 
  ∃ V : Type, convex_hexagon V ∧ is_centrally_symmetric V ∧ is_inscribed_polygon V W ∧ sorry :=
  sorry

end exists_centrally_symmetric_inscribed_convex_hexagon_l153_153446


namespace tan_theta_minus_pi_over_4_l153_153782

theorem tan_theta_minus_pi_over_4 (θ : Real) (k : ℤ)
  (h1 : - (π / 2) + (2 * k * π) < θ)
  (h2 : θ < 2 * k * π)
  (h3 : Real.sin (θ + π / 4) = 3 / 5) :
  Real.tan (θ - π / 4) = -4 / 3 :=
sorry

end tan_theta_minus_pi_over_4_l153_153782


namespace tank_emptying_time_l153_153278

theorem tank_emptying_time (fill_without_leak fill_with_leak : ℝ) (h1 : fill_without_leak = 7) (h2 : fill_with_leak = 8) : 
  let R := 1 / fill_without_leak
  let L := R - 1 / fill_with_leak
  let emptying_time := 1 / L
  emptying_time = 56 :=
by
  sorry

end tank_emptying_time_l153_153278


namespace sugar_packs_l153_153546

variable (totalSugar : ℕ) (packWeight : ℕ) (sugarLeft : ℕ)

noncomputable def numberOfPacks (totalSugar packWeight sugarLeft : ℕ) : ℕ :=
  (totalSugar - sugarLeft) / packWeight

theorem sugar_packs : numberOfPacks 3020 250 20 = 12 := by
  sorry

end sugar_packs_l153_153546


namespace pascal_triangle_47_rows_l153_153088

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153088


namespace triangle_congruence_l153_153764

-- Define the vertices of the regular decagon
inductive Vertex : Type
| A : Vertex
| B : Vertex
| C : Vertex
| D : Vertex
| E : Vertex
| F : Vertex
| G : Vertex
| H : Vertex
| I : Vertex
| J : Vertex

open Vertex

-- Define the function to color the vertices red or blue
inductive Color : Type
| red : Color
| blue : Color

open Color

def colorOf : Vertex → Color
| A := red -- Assume 5 vertices are red
| B := red
| C := red
| D := red
| E := red
| F := blue -- The rest are blue
| G := blue
| H := blue
| I := blue
| J := blue

-- Main statement to prove
theorem triangle_congruence : 
  ∃ (red_triangle blue_triangle : (Vertex × Vertex × Vertex)),
  (colorOf red_triangle.1 = red ∧ colorOf red_triangle.2 = red ∧ colorOf red_triangle.3 = red) ∧
  (colorOf blue_triangle.1 = blue ∧ colorOf blue_triangle.2 = blue ∧ colorOf blue_triangle.3 = blue) ∧
  (is_congruent red_triangle blue_triangle) :=
sorry

end triangle_congruence_l153_153764


namespace gwen_total_books_l153_153042

theorem gwen_total_books
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (mystery_shelves_count : mystery_shelves = 3)
  (picture_shelves_count : picture_shelves = 5)
  (each_shelf_books : books_per_shelf = 9) :
  (mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf) = 72 := by
  sorry

end gwen_total_books_l153_153042


namespace circle_line_distance_difference_l153_153741

theorem circle_line_distance_difference :
  let circle := ∀ (x y : ℝ), x^2 + y^2 - 4 * x - 4 * y + 5 = 0
  let line := ∀ (x y : ℝ), x + y - 9 = 0
  let center := (2, 2)
  let radius := real.sqrt 3
  let d_center_line := abs (1 * (2 : ℝ) + 1 * (2 : ℝ) - 9) / real.sqrt 2
  let D_max := d_center_line + radius
  let D_min := d_center_line - radius
  D_max - D_min = 2 * real.sqrt 3 :=
by
  sorry

end circle_line_distance_difference_l153_153741


namespace remainder_when_b_divided_by_29_l153_153873

theorem remainder_when_b_divided_by_29 :
  ∃ b : ℕ, (b ≡ (13⁻¹ + 17⁻¹ + 19⁻¹)⁻¹ [MOD 29]) ∧ b % 29 = 2 :=
sorry

end remainder_when_b_divided_by_29_l153_153873


namespace general_term_l153_153382

noncomputable def a (n : ℕ) : ℕ :=
if n = 1 then 1 else 3 * 4^(n - 2)

def S (n : ℕ) : ℕ := ∑ i in Finset.range n, a (i+1)

theorem general_term (n : ℕ) : a n = if n = 1 then 1 else 3 * 4^(n - 2) :=
sorry

end general_term_l153_153382


namespace range_of_m_l153_153806

theorem range_of_m 
  (θ : ℝ) 
  (h : ∀ θ : ℝ, (λ m : ℝ, m^2 + (Real.cos θ)^2 * m - 5 * m + 4 * (Real.sin θ)^2) m ≥ 0) :
  ∀ m : ℝ, (m ≥ 4 ∨ m ≤ 0) :=
by
  sorry

end range_of_m_l153_153806


namespace min_distance_between_points_l153_153833

-- Define the exponential function
def exp (x : ℝ) : ℝ := Real.exp x

-- Define the natural logarithm function
def ln (x : ℝ) : ℝ := Real.log x

-- Define a point structure
structure Point where
  x : ℝ
  y : ℝ

-- Define a point P on the graph of y = e^x
def P (x : ℝ) : Point :=
  { x := x, y := exp x }

-- Define a point Q on the graph of y = ln x
def Q (x : ℝ) : Point :=
  { x := x, y := ln x }

-- Define the minimum distance between points P and Q
def min_distance : ℝ :=
  sqrt 2

-- The theorem statement
theorem min_distance_between_points :
  (∀ x : ℝ, ∀ y : ℝ, (P x).x = x → (P x).y = exp x → (Q y).x = y → (Q y).y = ln y → 
   dist (P x) (Q (exp x)) = sqrt 2) := 
sorry

end min_distance_between_points_l153_153833


namespace sum_of_products_l153_153541

theorem sum_of_products (n : ℕ) (h : n > 0) : 
  ∑ i in Finset.range n, i * (i+1) = (1 / 3 : ℚ) * n * (n + 1) * (n + 2) := 
by
  sorry

end sum_of_products_l153_153541


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153989

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153989


namespace probability_heads_at_least_9_l153_153973

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153973


namespace center_of_circle_l153_153784

-- Given conditions
variables {a b : ℝ}
def C : ℝ×ℝ → Prop := λ (x y : ℝ), x^2 + y^2 + a*x - 2*y + b = 0
def symmetric_point (P : ℝ×ℝ) := (λ (P : ℝ×ℝ), ((P.fst + P.snd - 1), (1 - P.fst + P.snd))) P

-- Main statement to prove
theorem center_of_circle :
  C 2 1 ∧ C (symmetric_point (2, 1)).fst (symmetric_point (2, 1)).snd → (0, 1) = ( -a / 2, 1) :=
by
  assume h
  sorry

end center_of_circle_l153_153784


namespace Q_finishes_in_6_hours_l153_153507

def Q_time_to_finish_job (T_Q : ℝ) : Prop :=
  let P_rate := 1 / 3
  let Q_rate := 1 / T_Q
  let work_together_2hr := 2 * (P_rate + Q_rate)
  let P_alone_work_40min := (2 / 3) * P_rate
  work_together_2hr + P_alone_work_40min = 1

theorem Q_finishes_in_6_hours : Q_time_to_finish_job 6 :=
  sorry -- Proof skipped

end Q_finishes_in_6_hours_l153_153507


namespace winning_strategy_for_pawns_l153_153622

def wiit_or_siti_wins (n : ℕ) : Prop :=
  (∃ k : ℕ, n = 3 * k + 2) ∨ (∃ k : ℕ, n ≠ 3 * k + 2)

theorem winning_strategy_for_pawns (n : ℕ) : wiit_or_siti_wins n :=
sorry

end winning_strategy_for_pawns_l153_153622


namespace total_surface_area_cylinder_in_cone_in_sphere_l153_153628

variables (R : ℝ) (α : ℝ)
def total_surface_area (r : ℝ) : ℝ := 2 * real.pi * r^2 + 2 * real.pi * r * (2 * r)

theorem total_surface_area_cylinder_in_cone_in_sphere :
  let r := R * real.sin (2 * α) / (1 + 2 * real.tan α) in
  total_surface_area r = 6 * real.pi * (R^2) * (real.sin (2 * α) ^ 2) / (1 + 2 * real.tan α)^2 :=
sorry

end total_surface_area_cylinder_in_cone_in_sphere_l153_153628


namespace p_n_q_m_coprime_l153_153222

/-- Define the sequences p_n and q_n according to given recurrence relations -/
def p : ℕ → ℕ
| 1     := 1
| (n+1) := 2 * (q n)^2 - (p n)^2

def q : ℕ → ℕ
| 1     := 1
| (n+1) := 2 * (q n)^2 + (p n)^2

/-- The main theorem statement -/
theorem p_n_q_m_coprime (m n : ℕ) : gcd (p n) (q m) = 1 := 
sorry

end p_n_q_m_coprime_l153_153222


namespace calc_two_i_pow_four_l153_153337

theorem calc_two_i_pow_four : 2 * (complex.I ^ 4) = 2 := 
by {
    have h1: complex.I ^ 4 = 1 := by sorry,
    rw h1,
    norm_num,
}

end calc_two_i_pow_four_l153_153337


namespace angle_bisector_length_l153_153146

noncomputable def length_of_angle_bisector_PQR (PQ PR : ℝ) (cos_P : ℝ) : ℝ :=
  if PQ = 5 ∧ PR = 8 ∧ cos_P = 1/9 then 8 * real.sqrt 649 / 39 else 0

theorem angle_bisector_length (PQ PR : ℝ) (cos_P : ℝ)
  (hPQ : PQ = 5) (hPR : PR = 8) (hCos_P : cos_P = 1/9) :
  length_of_angle_bisector_PQR PQ PR cos_P = 8 * real.sqrt 649 / 39 := 
by {
  simp [length_of_angle_bisector_PQR, hPQ, hPR, hCos_P],
  sorry
}

end angle_bisector_length_l153_153146


namespace sector_area_l153_153011

-- Define the conditions for the problem
def arc_length : ℝ := Real.pi
def central_angle : ℝ := Real.pi / 4

-- Statement for proving the area of the sector is 2*pi square cm
theorem sector_area (L : ℝ) (θ : ℝ) (hL : L = arc_length) (hθ : θ = central_angle) :
  (1 / 2) * (L / θ) ^ 2 * θ = 2 * Real.pi := 
by 
sor

end sector_area_l153_153011


namespace area_triangle_AMB_l153_153933

def parabola (x : ℝ) : ℝ := x^2 + 2*x + 3

def point_A : ℝ × ℝ := (0, parabola 0)

def rotated_parabola (x : ℝ) : ℝ := -(x + 1)^2 + 2

def point_B : ℝ × ℝ := (0, rotated_parabola 0)

def vertex_M : ℝ × ℝ := (-1, 2)

def area_of_triangle (A B M : ℝ × ℝ) : ℝ :=
  0.5 * (A.2 - M.2) * (M.1 - B.1)

theorem area_triangle_AMB : area_of_triangle point_A point_B vertex_M = 1 :=
  sorry

end area_triangle_AMB_l153_153933


namespace probability_heads_in_12_flips_l153_153994

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153994


namespace parabola_properties_l153_153772

theorem parabola_properties 
  (p m : ℝ) (f : ℝ × ℝ) (h_p : p = 4)
  (h_parabola : ∀ x y : ℝ, y^2 = 2 * p * x)
  (h_focus : f = (2, 0))
  (h_point_on_parabola : ∃ m : ℝ, ∀ x y : ℝ, y^2 = 2 * p * x → (3, m))
  (h_distance_focus : ∃ d : ℝ, d = 5 ∧ dist (3, m) f = d)
  (a b midpoint_ab : ℝ × ℝ) (h_midpoint : (midpoint_ab = (a + b) / 2))
  (h_y_midpoint : midpoint_ab.2 = -1) :
  (∀ x y : ℝ, y^2 = 8 * x) ∧ (∃ l : ℝ → ℝ, l = fun x => -4 * (x - 2) ∧ ∀ x y : ℝ, x * 4 + y - 8 = 0) := 
sorry

end parabola_properties_l153_153772


namespace intersection_distance_l153_153543

theorem intersection_distance:
  let f := λ x : ℝ, 2 * Math.sin (x + Real.pi / 4) * Math.cos (x - Real.pi / 4)
  let line := (1 / 2 : ℝ)
  ∃ x2 x4 : ℝ, 
    ((f x2 = line) ∧ (f x4 = line) ∧ (0 < x2) ∧ (x2 < x4) ∧ 
    ∀ x < x2, f x ≠ line ∧ 
    ∀ x, (x2 < x ∧ x < x4) → (f x = line → x = x2 ∨ x = x4)) ∧ 
    |x2 - x4| = Real.pi :=
sorry

end intersection_distance_l153_153543


namespace Bryan_offer_l153_153888

theorem Bryan_offer (x : ℝ) : 
    let total_records := 200
    let sammy_offer := total_records * 4
    let bryan_half_records := total_records / 2
    let bryan_offer_interested := bryan_half_records * x
    let bryan_offer_not_interested := bryan_half_records * 1
    let bryan_total_offer := bryan_offer_interested + bryan_offer_not_interested
    let profit_difference := sammy_offer - bryan_total_offer
    profit_difference = 100 -> x = 6 :=
begin
    sorry
end

end Bryan_offer_l153_153888


namespace minimum_norm_diff_l153_153041

variables (t : ℝ)
variables (a b : ℝ × ℝ × ℝ)
#check a - b

def vector_a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 3)
def vector_b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)
def vector_diff (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 2 * t - 1, 3) - (2, t, t)

def norm (v : ℝ × ℝ × ℝ) := Real.sqrt (v.1 * v.1 + v.2 * v.2 + v.3 * v.3)

theorem minimum_norm_diff :
  ∃ t : ℝ, norm (vector_diff t) = 2 * Real.sqrt 2 :=
sorry

end minimum_norm_diff_l153_153041


namespace sum_of_abc_l153_153218

theorem sum_of_abc (h : (∃ a b c : ℕ, a * real.sqrt b = 5 * real.sqrt 5 ∧ c = 7 ∧ a + b + c = 37)) : 
  ∃ a b c : ℕ, 
  a * real.sqrt b / c = (250 / 98)^(1/2) ∧
  a + b + c = 37 :=
by
  sorry

end sum_of_abc_l153_153218


namespace second_term_is_correct_l153_153551

noncomputable def arithmetic_sequence_second_term (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) : ℤ :=
  a + d

theorem second_term_is_correct (a d : ℤ) (h1 : a + 9 * d = 15) (h2 : a + 10 * d = 18) :
  arithmetic_sequence_second_term a d h1 h2 = -9 :=
sorry

end second_term_is_correct_l153_153551


namespace pascal_triangle_contains_prime_l153_153060

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153060


namespace pascal_triangle_47_l153_153091

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153091


namespace length_of_EF_in_parallelogram_l153_153462

theorem length_of_EF_in_parallelogram (A B C D E F : Point)
  (h1 : Parallelogram A B C D)
  (h2 : ∠DAB = 60)
  (h3 : distance A B = 73)
  (h4 : distance B C = 88)
  (h5 : Line A D intersects Line C D at E)
  (h6 : Line B E is the angle bisector of ∠ABC and intersects Line A D at E)
  (h7 : Line B E intersects the extension of Line C D at F) :
  distance E F = 15 := 
sorry

end length_of_EF_in_parallelogram_l153_153462


namespace square_of_length_MN_l153_153132

-- Define the given polar points
def pointA : ℝ × ℝ := (4, Real.pi / 100)
def pointB : ℝ × ℝ := (8, 51 * Real.pi / 100)

-- Define the function to compute the polar coordinate midpoint
def polarMidpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Define the rectangular coordinates conversion from polar coordinates
def toRectangular (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * Real.cos p.2, p.1 * Real.sin p.2)

-- Define the midpoint of the segment AB in rectangular coordinates
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

-- Calculate N, the polar coordinate midpoint
def pointN : ℝ × ℝ := polarMidpoint pointA pointB

-- Convert the coordinates of N to rectangular coordinates
def rectN : ℝ × ℝ := toRectangular pointN

-- Calculate M, the rectangular coordinate midpoint
def rectM : ℝ × ℝ := midpoint (toRectangular pointA) (toRectangular pointB)

-- Define the squared distance between two points
def squaredDistance (p1 p2 : ℝ × ℝ) : ℝ :=
  (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2

-- State the problem to prove the square of the length of segment MN is 56 - 36 sqrt(2)
theorem square_of_length_MN :
  squaredDistance rectM rectN = 56 - 36 * Real.sqrt 2 := by
  sorry

end square_of_length_MN_l153_153132


namespace orthocenter_constructible_l153_153471

noncomputable def mark_orthocenter_of_triangle (A B C : Point) 
  (operations : List (Point → Point → Line)) 
  (can_draw_parallel : ∀ (l : Line), ∃ (l_parallel : Line), parallel l l_parallel ∧ is_distance_1 l l_parallel) : Prop :=
  ∃ H : Point, is_orthocenter H A B C

-- Assuming the necessary definitions and axioms have been established elsewhere
axiom non_collinear (A B C : Point) : ¬ collinear A B C

theorem orthocenter_constructible (A B C : Point) 
  (h_non_collinear : ¬ collinear A B C) 
  (operations : List (Point → Point → Line)) 
  (can_draw_parallel : ∀ (l : Line), ∃ (l_parallel : Line), parallel l l_parallel ∧ is_distance_1 l l_parallel) 
  (can_mark_arbitrary_point : ∀ (P : Point), ∃ (Q : Point), true)
  (can_mark_on_line : ∀ (l : Line), ∃ (P : Point), P ∈ l)
  (can_draw_line : ∀ (P Q : Point), ∃ (l : Line), l.contains P ∧ l.contains Q)
  (can_mark_intersection : ∀ (l₁ l₂ : Line), ¬ parallel l₁ l₂ → ∃ (P : Point), P ∈ l₁ ∧ P ∈ l₂) 
  : mark_orthocenter_of_triangle A B C operations can_draw_parallel :=
sorry

end orthocenter_constructible_l153_153471


namespace fill_tank_with_leak_l153_153303

theorem fill_tank_with_leak (P L T : ℝ) 
  (hP : P = 1 / 2)  -- Rate of the pump
  (hL : L = 1 / 6)  -- Rate of the leak
  (hT : T = 3)  -- Time taken to fill the tank with the leak
  : 1 / (P - L) = T := 
by
  sorry

end fill_tank_with_leak_l153_153303


namespace adam_total_weight_l153_153317

noncomputable def total_weight (monday_kilos : ℝ) (tuesday_multiplier : ℝ) (percentage_increase_wednesday : ℝ) 
  (kilo_to_pounds : ℝ) (kilo_to_grams : ℝ) : ℝ :=
let tuesday_kilos := monday_kilos * tuesday_multiplier in
let tuesday_pounds := tuesday_kilos * kilo_to_pounds in
let wednesday_pounds := tuesday_pounds * (percentage_increase_wednesday / 100 + 1) in
let wednesday_kilos := wednesday_pounds / kilo_to_pounds in
monday_kilos + tuesday_kilos + wednesday_kilos

theorem adam_total_weight :
  total_weight 15.5 3.2 105 2.2 1000 = 117.18 :=
by
  sorry

end adam_total_weight_l153_153317


namespace probability_same_color_l153_153950

-- Define the total number of plates
def totalPlates : ℕ := 6 + 5 + 3

-- Define the number of red plates, blue plates, and green plates
def redPlates : ℕ := 6
def bluePlates : ℕ := 5
def greenPlates : ℕ := 3

-- Define the total number of ways to choose 3 plates from 14
def totalWaysChoose3 : ℕ := Nat.choose totalPlates 3

-- Define the number of ways to choose 3 red plates, 3 blue plates, and 3 green plates
def redWaysChoose3 : ℕ := Nat.choose redPlates 3
def blueWaysChoose3 : ℕ := Nat.choose bluePlates 3
def greenWaysChoose3 : ℕ := Nat.choose greenPlates 3

-- Calculate the total number of favorable combinations (all plates being the same color)
def favorableCombinations : ℕ := redWaysChoose3 + blueWaysChoose3 + greenWaysChoose3

-- State the theorem: the probability that all plates are of the same color.
theorem probability_same_color : (favorableCombinations : ℚ) / (totalWaysChoose3 : ℚ) = 31 / 364 := by sorry

end probability_same_color_l153_153950


namespace shorter_piece_length_correct_l153_153619

problem wire_length (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : Prop :=
  total_length = 60 → 
  ratio = (2 / 5) → 
  let longer_length := (5 / 2) * shorter_length in
  total_length = shorter_length + longer_length → 
  shorter_length = 120 / 7

theorem shorter_piece_length_correct : problem wire_length 60 (2 / 5) (120 / 7) :=
by {
  intro h1,
  intro h2,
  let longer_length := (5 / 2) * (120 / 7),
  have h3 : 60 = (120 / 7) + longer_length, from sorry,
  exact h3
}

end shorter_piece_length_correct_l153_153619


namespace value_of_b_l153_153109

theorem value_of_b (b : ℝ) : 
  (∃ (x1 x2 : ℝ), x1 ≠ x2 ∧ (x1^3 - b*x1^2 + 1/2 = 0) ∧ (x2^3 - b*x2^2 + 1/2 = 0)) → b = 3/2 :=
by
  sorry

end value_of_b_l153_153109


namespace possible_values_interval_l153_153160

noncomputable def set_of_possible_values (a b c d : ℝ) (h_cond : a + b + c = -d) (h_d_nonzero : d ≠ 0) : Set ℝ := 
  {x | ∃ a b c : ℝ, a + b + c = -d ∧ x = ab + ac + bc}

theorem possible_values_interval (a b c d : ℝ) (h_cond : a + b + c = -d) (h_d_nonzero : d ≠ 0) :
  set_of_possible_values a b c d h_cond h_d_nonzero = Set.Icc 0 (d^2 / 2) :=
sorry

end possible_values_interval_l153_153160


namespace cross_section_area_parallel_to_base_l153_153773

/-- Given a regular triangular prism S-ABC with height SO = 3 and base side length of 6,
    a perpendicular is drawn from point A to the opposite face SBC, with the foot of the
    perpendicular denoted as O'. A point P is chosen on AO' such that AP / PQ = 8,
    prove the area of the cross-section passing through point P and parallel to the base. -/
theorem cross_section_area_parallel_to_base (S A B C O O' P Q : Point)
  (h_prism : is_regular_triangular_prism S A B C O)
  (h_SO : height (S, O) = 3)
  (h_base_side : side_length (A, B) = 6)
  (h_perpendicular : is_perpendicular_from_to A O' S B C)
  (h_partition : ∃ Q, on_line_segment AO' P ∧ AP / PQ = 8) :
  area (cross_section_pass_through P parallel_to (A B C)) = sqrt 3 :=
sorry

end cross_section_area_parallel_to_base_l153_153773


namespace remainder_of_7n_mod_4_l153_153242

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l153_153242


namespace find_a_solutions_l153_153361

theorem find_a_solutions (a : ℝ) : (log 5 (a^2 - 20 * a) = 3) → (a = 25 ∨ a = -5) := 
by
  sorry

end find_a_solutions_l153_153361


namespace exists_integer_quotient_2012_l153_153753

def is_perfect_square_divisor (n d : ℕ) : Prop :=
  ∃ x : ℕ, d = x * x ∧ d ∣ n

def is_perfect_cube_divisor (n d : ℕ) : Prop :=
  ∃ x : ℕ, d = x * x * x ∧ d ∣ n

def f (n : ℕ) : ℕ :=
  ∃ dm : List ℕ, (∀ d ∈ dm, is_perfect_square_divisor n d) ∧ 
  dm.length = (nat.factorization n).to_finset.sum (λ p, nat.floor (nat.factorization n p / 2) + 1)

def g (n : ℕ) : ℕ :=
  ∃ dm : List ℕ, (∀ d ∈ dm, is_perfect_cube_divisor n d) ∧ 
  dm.length = (nat.factorization n).to_finset.sum (λ p, nat.floor (nat.factorization n p / 3) + 1)

theorem exists_integer_quotient_2012 : ∃ (n : ℕ), f(n) / g(n) = 2012 :=
sorry

end exists_integer_quotient_2012_l153_153753


namespace find_b_l153_153750

/-- Given the distance between the parallel lines l₁ : x - y = 0
  and l₂ : x - y + b = 0 is √2, prove that b = 2 or b = -2. --/
theorem find_b (b : ℝ) (h : ∀ (x y : ℝ), (x - y = 0) → ∀ (x' y' : ℝ), (x' - y' + b = 0) → (|b| / Real.sqrt 2 = Real.sqrt 2)) :
  b = 2 ∨ b = -2 :=
sorry

end find_b_l153_153750


namespace translation_graph_symmetric_eq_exp_l153_153231

theorem translation_graph_symmetric_eq_exp (f : ℝ → ℝ) :
  (∀ x, f(x - 1) = exp x) → (∀ x, f x = log (x + 1)) :=
by
  intro h
  funext
  sorry

end translation_graph_symmetric_eq_exp_l153_153231


namespace slope_of_line_l153_153936

-- Definition of the line equation in slope-intercept form
def line_eq (x : ℝ) : ℝ := -5 * x + 9

-- Statement: The slope of the line y = -5x + 9 is -5
theorem slope_of_line : (∀ x : ℝ, ∃ m b : ℝ, line_eq x = m * x + b ∧ m = -5) :=
by
  -- proof goes here
  sorry

end slope_of_line_l153_153936


namespace probability_heads_at_least_9_l153_153971

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153971


namespace second_train_speed_l153_153921

theorem second_train_speed :
  ∀ (distance_AB time_first_train_start time_second_train_start time_meet speed_first_train : ℕ),
    distance_AB = 330 →
    speed_first_train = 60 →
    time_first_train_start = 8 →
    time_second_train_start = 9 →
    time_meet = 11 →
    let distance_first_train := (time_meet - time_first_train_start) * speed_first_train,
        distance_second_train := (time_meet - time_second_train_start) * v in
    distance_first_train + distance_second_train = distance_AB →
    v = 75 :=
by
  assume distance_AB time_first_train_start time_second_train_start time_meet speed_first_train h1 h2 h3 h4 h5,
  let distance_first_train := (time_meet - time_first_train_start) * speed_first_train,
      distance_second_train := (time_meet - time_second_train_start) * v in sorry

end second_train_speed_l153_153921


namespace fixed_monthly_costs_l153_153290

theorem fixed_monthly_costs
  (production_cost_per_component : ℕ)
  (shipping_cost_per_component : ℕ)
  (components_per_month : ℕ)
  (lowest_price_per_component : ℕ)
  (total_revenue : ℕ)
  (total_variable_cost : ℕ)
  (F : ℕ) :
  production_cost_per_component = 80 →
  shipping_cost_per_component = 5 →
  components_per_month = 150 →
  lowest_price_per_component = 195 →
  total_variable_cost = components_per_month * (production_cost_per_component + shipping_cost_per_component) →
  total_revenue = components_per_month * lowest_price_per_component →
  total_revenue = total_variable_cost + F →
  F = 16500 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end fixed_monthly_costs_l153_153290


namespace probability_at_least_9_heads_l153_153978

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153978


namespace shooting_to_practice_ratio_l153_153155

variable (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ)
variable (runningWeightliftingRatio : ℕ)

axiom practiceTime_def : practiceTime = 2 * 60 -- converting 2 hours to minutes
axiom weightliftingTime_def : weightliftingTime = 20
axiom runningWeightliftingRatio_def : runningWeightliftingRatio = 2
axiom runningTime_def : runningTime = runningWeightliftingRatio * weightliftingTime
axiom shootingTime_def : shootingTime = practiceTime - (runningTime + weightliftingTime)

theorem shooting_to_practice_ratio (practiceTime minutes weightliftingTime runningTime shootingTime : ℕ) 
                                   (runningWeightliftingRatio : ℕ) :
  practiceTime = 120 →
  weightliftingTime = 20 →
  runningWeightliftingRatio = 2 →
  runningTime = runningWeightliftingRatio * weightliftingTime →
  shootingTime = practiceTime - (runningTime + weightliftingTime) →
  (shootingTime : ℚ) / practiceTime = 1 / 2 :=
by sorry

end shooting_to_practice_ratio_l153_153155


namespace remainder_of_7n_mod_4_l153_153243

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l153_153243


namespace impossible_piles_of_three_l153_153181

theorem impossible_piles_of_three (n : ℕ) (h1 : n = 1001)
  (h2 : ∀ p : ℕ, p > 1 → ∃ a b : ℕ, a + b = p - 1 ∧ a ≤ b) : 
  ¬ (∃ piles : List ℕ, ∀ pile ∈ piles, pile = 3 ∧ (piles.sum = n + piles.length)) :=
by
  sorry

end impossible_piles_of_three_l153_153181


namespace complement_of_P_l153_153113

def U : Set ℤ := {-1, 0, 1, 2}
def P : Set ℤ := {x | x^2 < 2}

theorem complement_of_P :
  (U \ P) = {2} :=
by
  sorry

end complement_of_P_l153_153113


namespace lucille_cents_left_l153_153494

-- Definitions based on conditions
def weeds_in_flower_bed : ℕ := 11
def weeds_in_vegetable_patch : ℕ := 14
def weeds_in_grass : ℕ := 32
def cents_per_weed : ℕ := 6
def soda_cost : ℕ := 99

-- Main statement to prove
theorem lucille_cents_left : 
  let total_weeds_pulled := weeds_in_flower_bed + weeds_in_vegetable_patch + weeds_in_grass / 2
  let total_earnings := total_weeds_pulled * cents_per_weed
  let cents_left := total_earnings - soda_cost
  in cents_left = 147 := 
by {
  -- Definitions for computing intermediate results
  def total_weeds_pulled := weeds_in_flower_bed + weeds_in_vegetable_patch + weeds_in_grass / 2
  def total_earnings := total_weeds_pulled * cents_per_weed
  def cents_left := total_earnings - soda_cost
  -- Specify the exact values
  have h1 : total_weeds_pulled = 41 := by simp [weeds_in_flower_bed, weeds_in_vegetable_patch, weeds_in_grass]
  have h2 : total_earnings = 246 := by rw [h1]; simp [cents_per_weed]
  have h3 : cents_left = 147 := by rw [h2]; simp [soda_cost]
  exact h3
}

end lucille_cents_left_l153_153494


namespace equal_angles_locus_l153_153210

noncomputable def locus_of_points 
  (A B C M : Point) (α : Plane) (O : Point) (circumcenter : Point) : Set Point :=
{M | 
  let O := orthogonal_projection M α in
  (∠ (M, A) α = ∠ (M, B) α) ∧ 
  (∠ (M, A) α = ∠ (M, C) α) ∧ 
  (O = circumcenter.triangle_circumcenter A B C) ∧ 
  ∃ l : Line, 
    (l := Line.mk O perpendicular α) ∧ 
    M ∈ l
}

theorem equal_angles_locus (A B C : Point) (α : Plane): 
  ∃ l : Line, 
  let circumcenter := circumcenter.triangle_circumcenter A B C in 
  ∀ (M : Point),
    (∠ (M, A) α = ∠ (M, B) α) ∧ 
    (∠ (M, A) α = ∠ (M, C) α) →
    (M ∈ l) ↔ (l = Line.mk circumcenter perpendicular α) :=
sorry

end equal_angles_locus_l153_153210


namespace slope_of_line_l153_153573

theorem slope_of_line (x₁ y₁ x₂ y₂ : ℝ) (h₁ : x₁ = 1) (h₂ : y₁ = 3) (h₃ : x₂ = 4) (h₄ : y₂ = -6) : 
  (y₂ - y₁) / (x₂ - x₁) = -3 := by
  sorry

end slope_of_line_l153_153573


namespace square_of_binomial_l153_153582

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l153_153582


namespace purely_imaginary_number_l153_153027

noncomputable def complex_number (m : ℝ) : ℂ :=
  m * (m - 1) + (m - 1) * complex.I

theorem purely_imaginary_number (m : ℝ) (hz : m * (m - 1) = 0) (hm : m ≠ 1) :
  1 / (complex_number m) = complex.I :=
  sorry

end purely_imaginary_number_l153_153027


namespace average_speed_l153_153600

variable (x : ℝ)

-- Condition 1: The train covered x km at 70 kmph
def time1 (x : ℝ) : ℝ := x / 70

-- Condition 2: The train covered another 2x km at 20 kmph
def time2 (x : ℝ) : ℝ := (2 * x) / 20

-- Condition 3: Total distance covered is 3x km
def total_distance (x : ℝ) : ℝ := 3 * x

-- Defining the total time taken for the entire journey
def total_time (x : ℝ) : ℝ := time1 x + time2 x

-- The goal is to prove the average speed
theorem average_speed (x : ℝ) : total_distance x / total_time x = 26.25 :=
by
  sorry

end average_speed_l153_153600


namespace find_radius_of_cylinder_l153_153549

noncomputable def radius_of_cylinder (side_base : ℝ) (lateral_edge : ℝ) (dist : ℝ) : ℝ :=
  let R := sqrt 7 / 4
  R

theorem find_radius_of_cylinder :
  ∀ (side_base lateral_edge dist : ℝ), 
    (side_base = 1) → (lateral_edge = 1 / sqrt 3) → (dist = 1 / 4) →
    radius_of_cylinder side_base lateral_edge dist = (sqrt 7) / 4 :=
by
  intros side_base lateral_edge dist
  intros h1 h2 h3
  rw [h1, h2, h3]
  unfold radius_of_cylinder
  sorry

end find_radius_of_cylinder_l153_153549


namespace least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l153_153238

def least_subtrahend (n m : ℕ) (k : ℕ) : Prop :=
  (n - k) % m = 0 ∧ ∀ k' : ℕ, k' < k → (n - k') % m ≠ 0

theorem least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22 :
  least_subtrahend 102932847 25 22 :=
sorry

end least_number_subtracted_from_102932847_to_be_divisible_by_25_is_22_l153_153238


namespace part1_part2_l153_153397

variable {x : ℝ}

theorem part1 (t : ℤ) (f : ℝ → ℝ) (h1 : f x = x^(-t^2 + 2*t + 3))
  (h2 : ∀ x > 0, x ∈ (0 : ℝ, +∞) → f x = f (-x))  -- Even function
  (h3 : ∀ x > 0, x < y → f x ≤ f y) :  -- Monotonically increasing
  f x = x^4 :=
sorry

theorem part2 (a : ℝ) (g : ℝ → ℝ) (h4 : ∀ a > 0, a ≠ 1)
  (h5 : ∀ x ∈ [2, 4], g x = log a (a * sqrt (x^4) - x))
  (h6 : ∀ x ∈ [2, 4], g x is_monotonically_decreasing) :
  1/2 < a ∧ a < 1 :=
sorry

end part1_part2_l153_153397


namespace value_of_k_l153_153584

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l153_153584


namespace max_area_region_T_l153_153371

-- Given conditions
def circle1_radius := 2
def circle2_radius := 4
def circle3_radius := 6
def circle4_radius := 8

-- The radii of circles and their arrangement constraints
def circles (r: ℕ) := r ∈ {circle1_radius, circle2_radius, circle3_radius, circle4_radius}

-- Define the areas of the individual circles
def circle_area (r : ℕ) : ℝ := π * r^2

-- Region T is the sum of the areas of the four circles without overlap
def max_area_T : ℝ := (circle_area circle1_radius) + (circle_area circle2_radius) + (circle_area circle3_radius) + (circle_area circle4_radius)

-- The theorem to prove
theorem max_area_region_T : max_area_T = 120 * π := by
  sorry

end max_area_region_T_l153_153371


namespace bob_distance_regular_hexagon_l153_153636

theorem bob_distance_regular_hexagon (s : ℝ) (d : ℝ) (h1 : s = 3) (h2 : d = 8)
  (h3 : true) : sqrt ((5.5)^2 + (5*sqrt(3)/2)^2) = 7 := by
sorry

end bob_distance_regular_hexagon_l153_153636


namespace first_term_of_geometric_series_l153_153693

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/4) (h2 : S = 40) (h3 : S = a / (1 - r)) : 
  a = 30 :=
by
  -- Proof to be provided
  sorry

end first_term_of_geometric_series_l153_153693


namespace fraction_members_absent_l153_153130

variable (p : ℕ) -- Number of persons in the office
variable (W : ℝ) -- Total work amount
variable (x : ℝ) -- Fraction of members absent

theorem fraction_members_absent (h : W / (p * (1 - x)) = W / p + W / (6 * p)) : x = 1 / 7 :=
by
  sorry

end fraction_members_absent_l153_153130


namespace max_difference_evens_l153_153679

def even_digits_only (n : Nat) : Prop :=
  ∀ i, i < 6 → n.digitVal i % 2 = 0

def odd_digit_exists_between (a b : Nat) : Prop :=
  ∀ n, a < n → n < b → ∃ i, i < 6 ∧ n.digitVal i % 2 = 1

theorem max_difference_evens :
  ∃ a b : Nat, (even_digits_only a) ∧ (even_digits_only b) ∧
    (odd_digit_exists_between a b) ∧ b - a = 111112 := sorry

end max_difference_evens_l153_153679


namespace multiply_by_nine_l153_153267

theorem multiply_by_nine (x : ℝ) (h : 9 * x = 36) : x = 4 :=
sorry

end multiply_by_nine_l153_153267


namespace triple_g_eq_nineteen_l153_153879

def g (n : ℕ) : ℕ :=
  if n < 3 then n^2 + 3 else 2 * n + 1

theorem triple_g_eq_nineteen : g (g (g 1)) = 19 := by
  sorry

end triple_g_eq_nineteen_l153_153879


namespace smallest_perfect_square_divisible_252_l153_153224

theorem smallest_perfect_square_divisible_252 : 
  ∃ n : ℕ, (252 ∣ n) ∧ is_square n ∧ (∀ m : ℕ, (252 ∣ m) ∧ is_square m → n ≤ m) :=
begin
  use 1764,
  split,
  { -- Proof that 252 divides 1764
    sorry,
  },
  split,
  { -- Proof that 1764 is a perfect square
    sorry,
  },
  { -- Proof that 1764 is the smallest number satisfying above conditions
    sorry,
  }
end

end smallest_perfect_square_divisible_252_l153_153224


namespace arithmetic_sequence_sum_l153_153843

noncomputable def S (n : ℕ) : ℤ :=
  n * (-2012) + n * (n - 1) / 2 * (1 : ℤ)

theorem arithmetic_sequence_sum :
  (S 2012) / 2012 - (S 10) / 10 = 2002 → S 2017 = 2017 :=
by
  sorry

end arithmetic_sequence_sum_l153_153843


namespace exists_colored_triangle_l153_153735

theorem exists_colored_triangle (color : ℝ × ℝ → bool) :
  (∀ x y : ℝ × ℝ, x ≠ y → color x ≠ color y) →
  ∃ (A B C : ℝ × ℝ), color A = color B ∧ color B = color C ∧
  dist A B = 1 ∧
  ∃ θ α β : ℝ, θ = (1/7) * π ∧ α = (2/7) * π ∧ β = (4/7) * π ∧
  (angle B A C = θ ∧ angle A B C = α ∧ angle A C B = β) :=
by sorry

end exists_colored_triangle_l153_153735


namespace square_of_binomial_l153_153583

theorem square_of_binomial (k : ℝ) : (∃ a : ℝ, x^2 - 20 * x + k = (x - a)^2) → k = 100 :=
by {
  sorry
}

end square_of_binomial_l153_153583


namespace ivan_receives_amount_l153_153854

def initial_deposit : ℕ := 100000

def compensation_limit : ℕ := 1400000

def insurable_event : Prop := true

def bank_participant_in_insurance : Prop := true

theorem ivan_receives_amount (d : ℕ) (bank_insured : Prop) (insurable : Prop) (comp_limit : ℕ) :
  d = initial_deposit →
  bank_insured →
  insurable →
  comp_limit = compensation_limit →
  (d ≤ comp_limit) →
  (∃ accrued_interest : ℕ, (d + accrued_interest) ≤ comp_limit) → 
  d + (some accrued_interest) ≤ compensation_limit :=
by
  intros h_initial h_insured h_insurable h_limit h_d_le_limit h_exists_interest,
  sorry

end ivan_receives_amount_l153_153854


namespace incorrect_propositions_l153_153364

variable (A B C a b c : ℝ)

def is_isosceles (A B C : ℝ) : Prop :=
A = B ∨ B = C ∨ C = A

def is_obtuse (A B C : ℝ) : Prop :=
A > π / 2 ∨ B > π / 2 ∨ C > π / 2

theorem incorrect_propositions :
  ¬( (tan A / tan B = a^2 / b^2 → is_isosceles A B C) ∧ 
     ( (b^2 + c^2 - a^2) / (a^2 + c^2 - b^2) = b^2 / a^2 → is_isosceles A B C)) := 
sorry

end incorrect_propositions_l153_153364


namespace true_absolute_error_l153_153360

theorem true_absolute_error (a a₀ : ℝ) (ha : a = 246) (ha₀ : a₀ = 245.2) : abs (a - a₀) = 0.8 :=
by
  rw [ha, ha₀]
  norm_num
  sorry

end true_absolute_error_l153_153360


namespace lies_count_l153_153560

theorem lies_count (c : ℕ → ℕ) (truthful : ∃ i, c i = i)
  (h₁ : c 0 = 1)
  (h₂ : c 1 = 2)
  (h₃ : c 2 = 3)
  (h₄ : c 3 = 4)
  (h₅ : c 4 = 5)
  (h₆ : c 5 = 6)
  (h₇ : c 6 = 7)
  (h₈ : c 7 = 8)
  (h₉ : c 8 = 9)
  (h₁₀ : c 9 = 10)
  (h₁₁ : c 10 = 11)
  (h₁₂ : c 11 = 12) :
  (∑ i in finset.range 12, if c i ≠ i then 1 else 0) = 12 :=
sorry

end lies_count_l153_153560


namespace probability_at_least_9_heads_l153_153983

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153983


namespace radon_nikodym_inv_exists_l153_153169

open MeasureTheory

variables {Ω : Type*} {𝓕 : MeasurableSpace Ω} (λ μ : Measure Ω)
variables (f : Ω → ℝ)
variable (c : ℝ)

noncomputable def radon_nikodym_derivative_inv (hf : Integrable f μ) (hμ : μ (setOf (λ ω, f ω = 0)) = 0) : Ω → ℝ :=
  λ ω, if f ω ≠ 0 then 1 / f ω else c

theorem radon_nikodym_inv_exists 
  (hf : f = (λ μ.toMeasurable λ).rnDeriv μ.toMeasurable)
  (hμ : μ (setOf (λ ω, f ω = 0)) = 0) :
  μ.withDensity (radon_nikodym_derivative_inv λ μ f c hf hμ) = λ :=
sorry

end radon_nikodym_inv_exists_l153_153169


namespace median_is_3_l153_153947

-- We are defining the list of number of children per family.
def num_children : List ℕ := [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5]

-- We know that the total number of families is 15.
def num_families : ℕ := 15

-- The median function would return the middle value in an ordered list.
def median (l : List ℕ) : ℕ :=
  let sorted_l := l.qsort (≤)
  sorted_l.get! ((sorted_l.length + 1) / 2 - 1)

-- Define our theorem to state that the median of num_children is 3.
theorem median_is_3 : median num_children = 3 := by
  -- In the statement part, 'sorry' is used to indicate that the proof is omitted.
  sorry

end median_is_3_l153_153947


namespace find_distinct_numbers_l153_153751

theorem find_distinct_numbers :
  ∃ (x y : ℕ), 
    x ≠ y ∧ 
    (∃ k₁, x^2 + y^2 = k₁^3) ∧ 
    (∃ k₂, x^3 + y^3 = k₂^2) :=
by {
  use 625,
  use 1250,
  split,
  { -- x ≠ y
    exact ne_of_lt (by norm_num),
  },
  split,
  { -- ∃ k₁, 625^2 + 1250^2 = k₁^3
    use 125,
    norm_num,
  },
  { -- ∃ k₂, 625^3 + 1250^3 = k₂^2
    use 46875,
    norm_num,
  },
}

end find_distinct_numbers_l153_153751


namespace remainder_of_7n_mod_4_l153_153246

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l153_153246


namespace square_of_binomial_eq_100_l153_153577

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l153_153577


namespace min_value_y_l153_153798

/-- The given function f(x) = ln(x) + mx where m is a constant -/
def f (x m : ℝ) : ℝ := log x + m * x

/-- The function g(x) = f(x) + 1/2 * x^2 -/
def g (x m : ℝ) : ℝ := f x m + 1/2 * x^2

/-- The function h(x) = 2 * log(x) - a * x - x^2 -/
def h (x a : ℝ) : ℝ := 2 * log x - a * x - x^2

/-- The derivative of h(x) is h'(x) = 2 / x - a - 2 * x -/
def h' (x a : ℝ) : ℝ := 2 / x - a - 2 * x

noncomputable def y (x1 x2 a : ℝ) : ℝ := (x1 - x2) * h' ((x1 + x2) / 2) a

theorem min_value_y (m : ℝ) (x1 x2 : ℝ) (h_cond : m ≤ 3 * Real.sqrt 2 / 2)
  (x1_lt_x2 : x1 < x2) (ext_pts_h: ∀ a, (h x1 a = 0 ∧ h x2 a = 0)) :
  ∃ a : ℝ, y x1 x2 a = -4/3 + 2 * log 2 :=
by {
  sorry -- Proof is not required
}

end min_value_y_l153_153798


namespace probability_heads_in_12_flips_l153_153996

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153996


namespace number_of_terms_in_expansion_l153_153333

def first_factor : List Char := ['x', 'y']
def second_factor : List Char := ['u', 'v', 'w', 'z', 's']

theorem number_of_terms_in_expansion :
  first_factor.length * second_factor.length = 10 :=
by
  -- Lean expects a proof here, but the problem statement specifies to use sorry to skip the proof.
  sorry

end number_of_terms_in_expansion_l153_153333


namespace carmen_burning_candles_l153_153338

theorem carmen_burning_candles (candle_hours_per_night: ℕ) (nights_per_candle: ℕ) (candles_used: ℕ) (total_nights: ℕ) : 
  candle_hours_per_night = 2 →
  nights_per_candle = 8 / candle_hours_per_night →
  candles_used = 6 →
  total_nights = candles_used * (nights_per_candle / candle_hours_per_night) →
  total_nights = 24 :=
by
  intros h1 h2 h3 h4
  sorry

end carmen_burning_candles_l153_153338


namespace projection_locus_l153_153039

variables (A B C P P' : Point)
variables (ABP ABC A_1B_1C_1: Triangle)

def orthogonal_projection (P : Point) (ABC : Triangle) : Point := sorry
def height_in_tetrahedron (ABC : Triangle) (P : Point) : Real := sorry
def smallest_height (ABCP : Tetrahedron) : Real := sorry
def interior_and_boundary_triangle (A_1B_1C_1 : Triangle) : Set Point := sorry

theorem projection_locus
  (h_ABC : is_triangle ABC)
  (h_ABP : is_triangle ABP)
  (h_projection : P' = orthogonal_projection P ABC)
  (h_smallest_height : height_in_tetrahedron ABC P = smallest_height (tetrahedron ABC P)) :
  P' ∈ interior_and_boundary_triangle A_1B_1C_1 := 
sorry

end projection_locus_l153_153039


namespace x_intercept_of_line_l2_l153_153928

theorem x_intercept_of_line_l2 :
  ∀ (l1 l2 : ℝ → ℝ),
  (∀ x y, 2 * x - y + 3 = 0 → l1 x = y) →
  (∀ x y, 2 * x - y - 6 = 0 → l2 x = y) →
  l1 0 = 6 →
  l2 0 = -6 →
  l2 3 = 0 :=
by
  sorry

end x_intercept_of_line_l2_l153_153928


namespace min_value_M_l153_153359

theorem min_value_M : 
  ∃ M : ℝ, M = 9 * real.sqrt 2 / 32 ∧ 
  ∀ a b c : ℝ, 
  |a * b * (a^2 - b^2) + b * c * (b^2 - c^2) + c * a * (c^2 - a^2)| ≤ M * (a^2 + b^2 + c^2)^2 := 
by
  let M := 9 * real.sqrt 2 / 32
  use M
  split
  { exact rfl }
  { intros a b c
    sorry }

end min_value_M_l153_153359


namespace largest_divisor_of_expression_l153_153103

theorem largest_divisor_of_expression (x : ℤ) (h : x % 2 = 1) : 
  324 ∣ (12 * x + 3) * (12 * x + 9) * (6 * x + 6) :=
sorry

end largest_divisor_of_expression_l153_153103


namespace perfect_square_trinomial_implies_value_of_a_l153_153208

theorem perfect_square_trinomial_implies_value_of_a (a : ℝ) :
  (∃ (b : ℝ), (∃ (x : ℝ), (x^2 - ax + 9 = 0) ∧ (x + b)^2 = x^2 - ax + 9)) ↔ a = 6 ∨ a = -6 :=
by
  sorry

end perfect_square_trinomial_implies_value_of_a_l153_153208


namespace exist_sequence_of_points_l153_153004

theorem exist_sequence_of_points (n m : ℕ) (h : m ≤ (n - 1) / 2) 
    (points : set (ℝ × ℝ)) (line_segments : set ((ℝ × ℝ) × (ℝ × ℝ))) 
    (h_card_points : points.card = n) 
    (h_card_segments : line_segments.card ≤ m * n): 
    ∃ (V : fin (m + 1) → (ℝ × ℝ)), 
      ∀ i : fin m, ((V i), (V (i + 1))) ∈ line_segments ∧ V i ≠ V (i + 1) := 
sorry

end exist_sequence_of_points_l153_153004


namespace max_triangle_side_sum_l153_153889

-- Defining the problem approximately where set N is the number 1 to 8 with all different numbers being vertexes.
theorem max_triangle_side_sum (a b c d e f g h : ℕ) 
  (h_sum : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ f ≠ g ∧ f ≠ h ∧ g ≠ h) 
  (h_range : {a, b, c, d, e, f, g, h} = {1, 2, 3, 4, 5, 6, 7, 8}):
  ∃ sum, (∀ (sum1 sum2 sum3 : ℕ), sum1 = a + b + d ∧ sum2 = b + c + e ∧ sum3 = d + e + f ∧ 
  sum1 = sum2 ∧ sum2 = sum3 ∧ sum1 = sum) ∧ sum = 19 :=
sorry

end max_triangle_side_sum_l153_153889


namespace abs_inequality_solution_l153_153938

theorem abs_inequality_solution (x : ℝ) :
  abs (2 * x - 5) ≤ 7 ↔ -1 ≤ x ∧ x ≤ 6 :=
sorry

end abs_inequality_solution_l153_153938


namespace find_first_offset_l153_153739

theorem find_first_offset
  (area : ℝ)
  (diagonal : ℝ)
  (offset2 : ℝ)
  (first_offset : ℝ)
  (h_area : area = 225)
  (h_diagonal : diagonal = 30)
  (h_offset2 : offset2 = 6)
  (h_formula : area = (diagonal * (first_offset + offset2)) / 2)
  : first_offset = 9 := by
  sorry

end find_first_offset_l153_153739


namespace cylinder_slicing_area_l153_153629

theorem cylinder_slicing_area :
  ∃ (d e f : ℤ), 
  (∃ (radius height: ℝ) (P Q : ℝ × ℝ), 
      radius = 8 ∧ height = 10 ∧
      (arc PQ).measure = 150 ∧
      sliced_plane PQ ∧
      painted PQ radius height ∧
      area_of_face PQ radius height = d * Real.pi + e * Real.sqrt f ∧
      (f ∉ { p * p | p : ℤ })) ∧
  d + e + f = 149 :=
sorry

end cylinder_slicing_area_l153_153629


namespace number_of_sets_l153_153545

theorem number_of_sets : 
  ∃ (count : ℕ), count = 
    (finset {A : finset ℕ // {1, 2}.to_finset ⊆ A ∧ A ⊆ {1, 2, 3, 4}.to_finset ∧ A ≠ {1, 2, 3, 4}.to_finset}).card :=
  sorry

end number_of_sets_l153_153545


namespace handshakes_at_event_l153_153705

theorem handshakes_at_event 
  (num_couples : ℕ) 
  (num_people : ℕ) 
  (num_handshakes_men : ℕ) 
  (num_handshakes_men_women : ℕ) 
  (total_handshakes : ℕ) 
  (cond1 : num_couples = 15) 
  (cond2 : num_people = 2 * num_couples) 
  (cond3 : num_handshakes_men = (num_couples * (num_couples - 1)) / 2) 
  (cond4 : num_handshakes_men_women = num_couples * (num_couples - 1)) 
  (cond5 : total_handshakes = num_handshakes_men + num_handshakes_men_women) : 
  total_handshakes = 315 := 
by sorry

end handshakes_at_event_l153_153705


namespace find_height_of_cone_l153_153401

noncomputable def height_of_cone (S1 S2 V : ℝ) : ℝ :=
  let h := 3 in
  if S1 = 4 * Real.pi ∧ S2 = 9 * Real.pi ∧ V = 19 * Real.pi then h else 0

theorem find_height_of_cone :
  ∀ (h S1 S2 V : ℝ), S1 = 4 * Real.pi → S2 = 9 * Real.pi → V = 19 * Real.pi →
  V = (1 / 3) * h * (S1 + (Real.sqrt (S1 * S2)) + S2) →
  h = 3 :=
begin
  intros h S1 S2 V H1 H2 H3 H4,
  sorry
end

end find_height_of_cone_l153_153401


namespace time_to_complete_together_l153_153529

theorem time_to_complete_together (sylvia_time carla_time combined_time : ℕ) (h_sylvia : sylvia_time = 45) (h_carla : carla_time = 30) :
  let sylvia_rate := 1 / (sylvia_time : ℚ)
  let carla_rate := 1 / (carla_time : ℚ)
  let combined_rate := sylvia_rate + carla_rate
  let time_to_complete := 1 / combined_rate
  time_to_complete = (combined_time : ℚ) :=
by
  sorry

end time_to_complete_together_l153_153529


namespace solve_system_l153_153906

noncomputable def system_solution (C1 C2 C3 : ℝ) : (ℝ → ℝ) × (ℝ → ℝ) × (ℝ → ℝ) := 
  (λ t, C1 * Real.exp (2 * t) + C2 * Real.exp (3 * t) + C3 * Real.exp (6 * t),
   λ t, C2 * Real.exp (3 * t) - 2 * C3 * Real.exp (6 * t),
   λ t, -C1 * Real.exp (2 * t) + C2 * Real.exp (3 * t) + C3 * Real.exp (6 * t))

theorem solve_system :
  ∃ (C1 C2 C3 : ℝ), 
  ∀ (t : ℝ),
  let (x, y, z) := system_solution C1 C2 C3 in
  (x t, y t, z t) = (C1 * Real.exp (2 * t) + C2 * Real.exp (3 * t) + C3 * Real.exp (6 * t),
                     C2 * Real.exp (3 * t) - 2 * C3 * Real.exp (6 * t),
                     -C1 * Real.exp (2 * t) + C2 * Real.exp (3 * t) + C3 * Real.exp (6 * t)) ∧
  ∃ (x y z : ℝ → ℝ), 
  (∀ t, (dx/dt := 3 * x t - y t + z t) ∧
       (dy/dt := -x t + 5 * y t - z t) ∧
       (dz/dt := x t - y t + 3 * z t))
sorry

end solve_system_l153_153906


namespace largest_divisor_of_expression_expression_divisible_by_12_l153_153346

theorem largest_divisor_of_expression (n : ℤ) : nat.gcd ((n^3 + 3*n^2 - n - 3).natAbs) 12 = 12 :=
by sorry

/-- For all integers n, the expression n^3 + 3n^2 - n - 3 is divisible by 12. -/
theorem expression_divisible_by_12 (n : ℤ) : 12 ∣ (n^3 + 3*n^2 - n - 3) :=
by sorry

end largest_divisor_of_expression_expression_divisible_by_12_l153_153346


namespace least_possible_value_of_s_l153_153822

theorem least_possible_value_of_s (a b : ℤ) 
(h : a^3 + b^3 - 60 * a * b * (a + b) ≥ 2012) : 
∃ a b, a^3 + b^3 - 60 * a * b * (a + b) = 2015 :=
by sorry

end least_possible_value_of_s_l153_153822


namespace solve_for_x_l153_153903

theorem solve_for_x (x : ℚ) (h : (2 * x + 18) / (x - 6) = (2 * x - 4) / (x + 10)) : x = -26 / 9 :=
sorry

end solve_for_x_l153_153903


namespace angle_between_a_b_is_120_l153_153779

noncomputable def vector_a : ℝ × ℝ := sorry
noncomputable def vector_b : ℝ × ℝ := sorry
noncomputable def vector_c : ℝ × ℝ := sorry

-- Assuming the magnitudes of the vectors
axiom length_a : real.norm (vector_a) = 1
axiom length_b : real.norm (vector_b) = 1
axiom length_c : real.norm (vector_c) = 1

-- Condition that vectors sum to c
axiom a_b_sum_c : vector_a + vector_b = vector_c

-- Definitions to help with the proof
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def cos_angle (u v : ℝ × ℝ) : ℝ := dot_product u v / (real.norm u * real.norm v)

def angle (u v : ℝ × ℝ) : ℝ := real.arccos (cos_angle u v)

-- Theorem to be proven
theorem angle_between_a_b_is_120 : angle vector_a vector_b = real.pi / 3 * 2 :=
by 
  sorry

end angle_between_a_b_is_120_l153_153779


namespace geometric_sequence_sufficiency_geometric_sequence_not_necessary_l153_153007

theorem geometric_sequence_sufficiency (a : ℕ → ℝ) (q : ℝ) (a1_pos : a 1 > 0) (q_gt_one : q > 1) 
        (geometric_seq : ∀ n : ℕ, a (n + 1) = a 1 * q ^ n) :
        ∀ n : ℕ, a (n + 2) > a (n + 1) :=
by
  intro n
  have h : a (n + 2) = a 1 * q ^ (n + 1) := geometric_seq (n + 1)
  have h' : a (n + 1) = a 1 * q ^ n := geometric_seq n
  have q_power : q ^ (n + 1) > q ^ n := 
    by 
    apply pow_lt_pow_of_lt_left 
    { exact q_gt_one }
    { exact nat.succ_pos' n }
  linarith [h, h', a1_pos]

theorem geometric_sequence_not_necessary (a : ℕ → ℝ) (a1_non_pos : a 1 ≤ 0 ∨ ¬ (q > 1)) 
        (geometric_seq : ∀ n : ℕ, a (n + 1) > a n) :
        ¬ (∀ m : ℕ, a (m + 2) = a 1 * q ^ (m + 1)) :=
sorry

end geometric_sequence_sufficiency_geometric_sequence_not_necessary_l153_153007


namespace pirates_on_schooner_l153_153638

def pirate_problem (N : ℝ) : Prop :=
  let total_pirates       := N
  let non_participants    := 10
  let participants        := total_pirates - non_participants
  let lost_arm            := 0.54 * participants
  let lost_arm_and_leg    := 0.34 * participants
  let lost_leg            := (2 / 3) * total_pirates
  -- The number of pirates who lost only a leg can be calculated.
  let lost_only_leg       := lost_leg - lost_arm_and_leg
  -- The equation that needs to be satisfied
  lost_leg = lost_arm_and_leg + lost_only_leg

theorem pirates_on_schooner : ∃ N : ℝ, N > 10 ∧ pirate_problem N :=
sorry

end pirates_on_schooner_l153_153638


namespace C_task_dig_l153_153952

def heights := {A, B, C : Type}
def task := {dig, fertilize, water : Type}

variables (tallest shortest : heights)
variables (A B C : heights)
variables (dig_task fertilize_task water_task : task)

axiom height_distinct : A ≠ B ∧ A ≠ C ∧ B ≠ C
axiom task_distinct : dig_task ≠ fertilize_task ∧ dig_task ≠ water_task ∧ fertilize_task ≠ water_task

axiom A_not_tallest : A ≠ tallest
axiom tallest_not_water : tallest ≠ C -- If C is tallest, then C did not water
axiom shortest_fertilize : fertilize_task = shortest
axiom B_not_shortest : B ≠ shortest
axiom B_not_dig : dig_task ≠ B

theorem C_task_dig : dig_task = C :=
begin
  sorry
end

end C_task_dig_l153_153952


namespace certain_number_exists_l153_153572

theorem certain_number_exists :
  ∃ x : ℤ, 55 * x % 7 = 6 ∧ x % 7 = 1 := by
  sorry

end certain_number_exists_l153_153572


namespace passion_fruit_crates_l153_153153

-- Step d): Lean 4 statement
theorem passion_fruit_crates (total_crates grapes_crates mangoes_crates : ℕ) (h1 : total_crates = 50) (h2 : grapes_crates = 13) (h3 : mangoes_crates = 20) : total_crates - (grapes_crates + mangoes_crates) = 17 :=
by
  rw [h1, h2, h3]
  norm_num
  -- The full proof would normally go here, but is omitted as instructed
  sorry

end passion_fruit_crates_l153_153153


namespace remainder_7n_mod_4_l153_153259

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l153_153259


namespace inequality_solution_set_l153_153404

noncomputable def f : ℝ → ℝ := sorry
noncomputable def f' : ℝ → ℝ := sorry
def y (x : ℝ) := f x - 1

-- Assume f is differentiable and other conditions
axiom f_differentiable : differentiable ℝ f
axiom f_greater_than_f' : ∀ x : ℝ, f x > f' x
axiom y_is_odd : ∀ x : ℝ, y x = -y (-x)

theorem inequality_solution_set :
  { x : ℝ | f x < real.exp x } = set.Ioi 0 :=
by
  sorry

end inequality_solution_set_l153_153404


namespace remainder_when_7n_divided_by_4_l153_153249

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l153_153249


namespace unique_integer_n_l153_153720

theorem unique_integer_n :
  ∃! (n : ℕ), 0 < n ∧ (∑ i in Finset.range n, (i + 2) * 2^(i + 2)) = 2^(n + 11) ∧ n = 1025 :=
begin
  sorry
end

end unique_integer_n_l153_153720


namespace volume_ratio_2_l153_153293

noncomputable def volumeCylinder (r h : ℝ) : ℝ := π * r ^ 2 * h
noncomputable def volumeCone (r h : ℝ) : ℝ := (1/3) * π * r ^ 2 * h
noncomputable def volumeSphere (r : ℝ) : ℝ := (4/3) * π * r ^ 3

theorem volume_ratio_2 {r h : ℝ} (cylinder_volume : volumeCylinder r h = 128 * π) (same_radius : h = r) :
  (volumeSphere r) / (volumeCone r h) = 2 :=
by
  sorry

end volume_ratio_2_l153_153293


namespace cos_alpha_l153_153790

-- Definition of the point through which the terminal side of the angle passes
def point : ℝ × ℝ := (3, -4)

-- Definition of radius r obtained from the Pythagorean theorem
def radius : ℝ := Real.sqrt (3^2 + (-4)^2)

-- Statement to prove
theorem cos_alpha : Real.cos (atan2 (-4) 3) = 3 / radius :=
by
  -- Proof omitted
  sorry

end cos_alpha_l153_153790


namespace Elisa_paint_total_is_correct_l153_153178

def day1 (square_feet : ℕ) : Prop :=
  square_feet = 30

def day2 (monday_paint : ℕ) (tuesday_paint : ℕ) : Prop :=
  tuesday_paint = 2 * monday_paint

def day3 (monday_paint : ℕ) (wednesday_paint : ℕ) : Prop :=
  wednesday_paint = monday_paint / 2

def total_paint (monday_paint tuesday_paint wednesday_paint total_paint : ℕ) : Prop :=
  total_paint = monday_paint + tuesday_paint + wednesday_paint

theorem Elisa_paint_total_is_correct :
  ∃ (monday_paint tuesday_paint wednesday_paint total_paint : ℕ),
    day1 monday_paint ∧
    day2 monday_paint tuesday_paint ∧
    day3 monday_paint wednesday_paint ∧
    total_paint monday_paint tuesday_paint wednesday_paint total_paint ∧
    total_paint = 105 :=
by
  sorry

end Elisa_paint_total_is_correct_l153_153178


namespace identify_element_in_CCl4_l153_153743

noncomputable def molar_mass_C : ℚ := 12.01
noncomputable def molar_mass_Cl : ℚ := 35.45

theorem identify_element_in_CCl4 (mass_percentage : ℚ) (compounds : String) 
(h₁ : compounds = "CCl4")
(h₂ : mass_percentage = 7.89) : "C" :=
sorry

end identify_element_in_CCl4_l153_153743


namespace sum_of_numbers_is_103_l153_153762

theorem sum_of_numbers_is_103 (a b c d : ℤ) 
  (h1 : a + b = 7) 
  (h2 : b + c = 15) 
  (h3 : c + d = 43) 
  (h4 : a + c = 47) 
  (h5 : (a + d ≥ 47) 
  (h6 : b + d ≥ 47)) : 
  a + b + c + d = 103 :=
sorry

end sum_of_numbers_is_103_l153_153762


namespace find_max_difference_l153_153687

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l153_153687


namespace line_through_circumcenter_and_incenter_intersects_sides_l153_153690

variable {α : Type*} [EuclideanGeometry α]

def acute_triangle_has_unequal_sides (A B C O I : α) : Prop :=
  acute_triangle A B C ∧ 
  ¬(A = B ∨ B = C ∨ C = A) ∧
  (dist A B > dist B C) ∧ 
  (dist B C > dist C A) ∧ 
  is_circumcenter O A B C ∧ 
  is_incenter I A B C

theorem line_through_circumcenter_and_incenter_intersects_sides 
  {A B C O I : α} 
  (h : acute_triangle_has_unequal_sides A B C O I) :
  (∃ P, lies_on_line O I P ∧ lies_on_segment P A B) ∧
  (∃ Q, lies_on_line O I Q ∧ lies_on_segment Q A C) :=
sorry

end line_through_circumcenter_and_incenter_intersects_sides_l153_153690


namespace charlyn_visible_area_l153_153340

-- Define the constants
def s : ℝ := 6
def r : ℝ := 1.5

-- Define the areas to be calculated
def area_viewed_inside_square (s r : ℝ) : ℝ := s^2 - (s - 2 * r)^2
def area_rectangles (s r : ℝ) : ℝ := 4 * (s * r)
def area_quarter_circles (r : ℝ) : ℝ := 4 * (π * r^2 / 4)

-- Total visible area
def total_visible_area (s r : ℝ) : ℝ := area_viewed_inside_square s r + area_rectangles s r + area_quarter_circles r

-- Round the area to the nearest whole number
def rounded_total_visible_area (area : ℝ) : ℕ := Int.toNat (Int.round area)

-- The theorem to prove
theorem charlyn_visible_area : rounded_total_visible_area (total_visible_area s r) = 70 := by
  sorry

end charlyn_visible_area_l153_153340


namespace derivative_at_pi_over_four_l153_153034

noncomputable def f (x : ℝ) : ℝ := f' (π / 2) * sin x + cos x

theorem derivative_at_pi_over_four : f' (π / 4) = -Real.sqrt 2 :=
by
  -- place the necessary assumptions here
  sorry

end derivative_at_pi_over_four_l153_153034


namespace probability_three_balls_form_arithmetic_sequence_l153_153621

open Finset

def balls : Finset ℕ := {1, 2, 3, 4, 6}

noncomputable def combinations := (balls.choose 3).filter (λ s, ∃ a b c, s = {a, b, c} ∧ (2 * b = a + c ∨ 2 * a = b + c))

theorem probability_three_balls_form_arithmetic_sequence :
  (combinations.card : ℚ) / (balls.choose 3).card = 3 / 10 :=
by sorry

end probability_three_balls_form_arithmetic_sequence_l153_153621


namespace difference_increased_decreased_l153_153920

theorem difference_increased_decreased (n : ℕ) (h : n = 40) : 
  let increased := n + (25 * n) / 100
      decreased := n - (30 * n) / 100
  in increased - decreased = 22 :=
by
  sorry

end difference_increased_decreased_l153_153920


namespace line_intersects_circle_l153_153406

theorem line_intersects_circle (m : ℝ) (h : m^2 > 3/4) :
  ∃ x y : ℝ, x^2 + y^2 = 1 ∧ y = 2 * m * x + sqrt 3 :=
sorry

end line_intersects_circle_l153_153406


namespace graph_triples_l153_153295

theorem graph_triples (V : Finset ℕ) (hV : V.card = 30)
  (h_edges : ∀ v ∈ V, ∃ (E : Finset (Finset ℕ)), E.card = 6 ∧ ∀ e ∈ E, e ⊆ V ∧ v ∉ e ∧ e.card = 2):
  ∃ m, m = 1990 :=
by
  sorry

end graph_triples_l153_153295


namespace more_books_than_movies_l153_153948

noncomputable def max_books : ℕ := 20
noncomputable def max_movies : ℕ := 12

theorem more_books_than_movies (read_books watched_movies : ℕ) (h_books : read_books ≤ max_books) 
    (h_movies : watched_movies ≤ max_movies) : read_books - watched_movies ≤ 8 := 
begin
  sorry
end

end more_books_than_movies_l153_153948


namespace triangle_ABCsqrt2_ratio_l153_153386

variable {α : Type} [linearOrderedField α] [hasSin α] [hasCos α] 

theorem triangle_ABCsqrt2_ratio 
  (a b c A B C : α) 
  (h1 : 2 * b * sin (2 * A) = 3 * a * sin B) 
  (h2 : c = 2 * b) 
  (h3 : cos A = (b^2 + c^2 - a^2) / (2 * b * c)) :
  a / b = sqrt 2 := 
sorry

end triangle_ABCsqrt2_ratio_l153_153386


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153986

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153986


namespace proof_problem1_proof_problem2_l153_153715

noncomputable def problem1 : ℝ :=
  (Real.pi ^ 0) + (2 ^ -2) * (9 / 4) ^ (1 / 2)

theorem proof_problem1 : problem1 = 11 / 8 :=
by
  sorry

noncomputable def problem2 : ℝ :=
  2 * Real.logb 5 10 + Real.logb 5 0.25

theorem proof_problem2 : problem2 = 2 :=
by
  sorry

end proof_problem1_proof_problem2_l153_153715


namespace necessary_implies_sufficient_l153_153439

theorem necessary_implies_sufficient (p q : Prop) (h : p → q) : p → q :=
by {
  exact h,
  sorry
}

end necessary_implies_sufficient_l153_153439


namespace _l153_153106

/-- This theorem states that if the GCD of 8580 and 330 is diminished by 12, the result is 318. -/
example : (Int.gcd 8580 330) - 12 = 318 :=
by
  sorry

end _l153_153106


namespace tank_a_is_60_percent_of_tank_b_l153_153279

noncomputable def volume_of_cylinder (r h : ℝ) : ℝ := π * r^2 * h

def tank_a_height := 10
def tank_a_circumference := 6
def tank_b_height := 6
def tank_b_circumference := 10

def radius_from_circumference (C : ℝ) : ℝ := C / (2 * π)

noncomputable def tank_a_volume : ℝ :=
  volume_of_cylinder (radius_from_circumference tank_a_circumference) tank_a_height

noncomputable def tank_b_volume : ℝ :=
  volume_of_cylinder (radius_from_circumference tank_b_circumference) tank_b_height

theorem tank_a_is_60_percent_of_tank_b :
  (tank_a_volume / tank_b_volume) * 100 = 60 := by
  sorry

end tank_a_is_60_percent_of_tank_b_l153_153279


namespace beetle_avg_speed_correct_l153_153325

-- Definitions and conditions
def ant_smooth_dist : ℝ := 1 -- 1000 meters = 1 kilometer
def ant_time : ℝ := 0.5 -- 30 minutes = 0.5 hours
def forest_speed_reduction : ℝ := 0.75 -- 75% speed on forest terrain
def beetle_smooth_dist_reduction : ℝ := 0.1 -- 10% less distance on smooth terrain
def beetle_forest_speed_percent : ℝ := 0.9 -- 90% speed on forest terrain

def ant_speed_smooth := ant_smooth_dist / ant_time
def ant_speed_forest := ant_speed_smooth * forest_speed_reduction
def beetle_smooth_dist := ant_smooth_dist * (1 - beetle_smooth_dist_reduction)
def beetle_speed_smooth := beetle_smooth_dist / ant_time
def beetle_speed_forest := beetle_speed_smooth * beetle_forest_speed_percent
def beetle_avg_speed := (beetle_speed_smooth + beetle_speed_forest) / 2

-- Proof statement
theorem beetle_avg_speed_correct : beetle_avg_speed = 1.71 := by
  sorry

end beetle_avg_speed_correct_l153_153325


namespace monomials_like_terms_sum_l153_153450

-- Define the conditions given in the problem
variable (m n : ℕ)
axiom exponents_match_x : 5 = m
axiom exponents_match_y : 2 * n = 4

-- The proof statement
theorem monomials_like_terms_sum : m + 2 * n = 9 :=
by
  have m_val : m = 5 := exponents_match_x
  have n_val : n = 2 := (Nat.mul_right_inj' (by norm_num : 2 ≠ 0)).mp exponents_match_y
  calc
    m + 2 * n = 5 + 2 * n := by rw [m_val]
    ...      = 5 + 2 * 2 := by rw [n_val]
    ...      = 9         := by norm_num

end monomials_like_terms_sum_l153_153450


namespace sum_binomial_coeffs_conditions_l153_153788

theorem sum_binomial_coeffs_conditions
  (n : ℕ) :
  (∑ k in finset.range (n+1), if even k then binomial n k else 0 = 128) →
  (n = 8 ∧
   ∑ k in finset.range (n+1), binomial n k * (x - 2) ^ (n - k) = (1 : ℤ) ∧
   ∑ k in finset.range (n+1), binomial n k = 256) :=
by
  sorry

end sum_binomial_coeffs_conditions_l153_153788


namespace remainder_7n_mod_4_l153_153257

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l153_153257


namespace find_t_l153_153370

theorem find_t : 
  ∃ t : ℝ, (∃ x : ℝ, 3 * x^2 + 15 * x + t = 0 ∧ x = (-15 - sqrt 145) / 6) ↔ t = 20 / 3 :=
by
  sorry

end find_t_l153_153370


namespace coefficient_x3_in_x_mul_x_minus_1_pow_5_l153_153730

theorem coefficient_x3_in_x_mul_x_minus_1_pow_5 :
  (coefficient (x * (x - 1)^5) 3) = -10 :=
sorry

end coefficient_x3_in_x_mul_x_minus_1_pow_5_l153_153730


namespace lcm_of_4_6_10_18_l153_153571

theorem lcm_of_4_6_10_18 : Nat.lcm (Nat.lcm 4 6) (Nat.lcm 10 18) = 180 := by
  sorry

end lcm_of_4_6_10_18_l153_153571


namespace math_problem_l153_153569

theorem math_problem :
  8 / 4 - 3^2 + 4 * 2 + (Nat.factorial 5) = 121 :=
by
  sorry

end math_problem_l153_153569


namespace left_handed_ratio_l153_153277

theorem left_handed_ratio 
    (red_participants blue_participants : ℕ)
    (ratio_red_blue : red_participants = 7 * blue_participants / 3)
    (left_handed_red : ℚ)
    (left_handed_blue : ℚ) 
    (one_third_red : left_handed_red = 1 / 3) 
    (two_thirds_blue : left_handed_blue = 2 / 3) : 
    (7 / 3) * (1 / 3) + (2 / 3) * (blue_participants) = 13 / 30 * 10 * blue_participants :=
begin
  sorry
end

end left_handed_ratio_l153_153277


namespace min_value_of_f_l153_153802

definition f (x : ℝ) (h : x > 0) : ℝ := (x^2 + 1) / (2 * x)

theorem min_value_of_f :
  (∀ x : ℝ, x > 0 → f (x + 2016) ⟨x + 2016 > 0⟩ ≥ 1) ∧ 
  (∃ x : ℝ, x > 0 ∧ f (x + 2016) ⟨x + 2016 > 0⟩ = 1) :=
begin
  sorry
end

end min_value_of_f_l153_153802


namespace probability_of_defective_product_is_0_032_l153_153326

-- Defining the events and their probabilities
def P_H1 : ℝ := 0.30
def P_H2 : ℝ := 0.25
def P_H3 : ℝ := 0.45

-- Defining the probabilities of defects given each production line
def P_A_given_H1 : ℝ := 0.03
def P_A_given_H2 : ℝ := 0.02
def P_A_given_H3 : ℝ := 0.04

-- Summing up the total probabilities
def P_A : ℝ :=
  P_H1 * P_A_given_H1 +
  P_H2 * P_A_given_H2 +
  P_H3 * P_A_given_H3

-- The statement to be proven
theorem probability_of_defective_product_is_0_032 :
  P_A = 0.032 :=
by
  -- Proof would go here
  sorry

end probability_of_defective_product_is_0_032_l153_153326


namespace pascal_triangle_contains_prime_l153_153059

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153059


namespace factor_2210_two_digit_count_l153_153096

theorem factor_2210_two_digit_count :
  ∃ (n : ℕ), n = 2 ∧
  ∃ (a b c d : ℕ),
    10 ≤ a ∧ a < 100 ∧
    10 ≤ b ∧ b < 100 ∧
    10 ≤ c ∧ c < 100 ∧
    10 ≤ d ∧ d < 100 ∧
    a * b = 2210 ∧
    c * d = 2210 ∧
    (a, b) ≠ (c, d) ∧
    (a, b) ≠ (d, c) :=
begin
  sorry
end

end factor_2210_two_digit_count_l153_153096


namespace number_of_b_for_line_passing_through_parabola_vertex_l153_153367

theorem number_of_b_for_line_passing_through_parabola_vertex :
  (∀ b : ℝ, ∃ y : ℝ, y = 2 * 0 + b ∧ y = 0^2 + b^2) → ∃ b_vals : set ℝ, b_vals = {0, 1} ∧ b_vals.card = 2 :=
by
  sorry

end number_of_b_for_line_passing_through_parabola_vertex_l153_153367


namespace find_A_coordinates_l153_153135

-- Given conditions
variable (B : (ℝ × ℝ)) (hB1 : B = (1, 2))

-- Definitions to translate problem conditions into Lean
def symmetric_y (P B : ℝ × ℝ) : Prop :=
  P.1 = -B.1 ∧ P.2 = B.2

def symmetric_x (A P : ℝ × ℝ) : Prop :=
  A.1 = P.1 ∧ A.2 = -P.2

-- Theorem statement
theorem find_A_coordinates (A P B : ℝ × ℝ) (hB1 : B = (1, 2))
    (h_symm_y: symmetric_y P B) (h_symm_x: symmetric_x A P) : 
    A = (-1, -2) :=
by
  sorry

end find_A_coordinates_l153_153135


namespace ratio_of_men_to_women_l153_153555

theorem ratio_of_men_to_women
  (M W : ℕ)
  (h1 : W = M + 6)
  (h2 : M + W = 16) :
  M * 11 = 5 * W :=
by
    -- We can explicitly construct the necessary proof here, but according to instructions we add sorry to bypass for now
    sorry

end ratio_of_men_to_women_l153_153555


namespace circles_intersect_parametric_to_standard_polar_to_rectangular_l153_153842

def circle_C1_parametric (α : ℝ) : ℝ × ℝ :=
  (2 * Real.cos α, 2 + 2 * Real.sin α)

def circle_C1_standard (x y : ℝ) : Prop :=
  x^2 + (y - 2)^2 = 4

def circle_C2_polar (θ : ℝ) : ℝ :=
  2 * Real.sqrt 2 * Real.cos (θ + Real.pi / 4)

def circle_C2_rectangular (x y : ℝ) : Prop :=
  (x - 1)^2 + (y + 1)^2 = 2

def distance_sq (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  (x₁ - x₂)^2 + (y₁ - y₂)^2

def radius_c1 : ℝ := 2
def radius_c2 : ℝ := Real.sqrt 2

def centers_distance_sq : ℝ := distance_sq 0 2 1 (-1)

theorem circles_intersect : centers_distance_sq < (radius_c1 + radius_c2)^2 :=
  sorry

theorem parametric_to_standard (α : ℝ) : 
  ∃ x y : ℝ, circle_C1_parametric α = (x, y) ∧ circle_C1_standard x y :=
  sorry

theorem polar_to_rectangular (θ : ℝ) : 
  ∃ x y : ℝ, (circle_C2_polar θ = Real.sqrt (x^2 + y^2)) ∧ (circle_C2_rectangular x y) :=
  sorry

end circles_intersect_parametric_to_standard_polar_to_rectangular_l153_153842


namespace work_done_in_isothermal_process_l153_153185

-- One mole of an ideal monatomic gas (n=1)
-- Work done in the first isobaric process (W1)
def n : ℕ := 1
def W1 : ℝ := 30

-- The problem states the gas receives the same heat in the second process (Q2 = Q1)
-- We need to prove that the work done in the second (isothermal) process W2 is 75 J
theorem work_done_in_isothermal_process :
  (W2 = 75) :=
by
  -- Hypothesis: heat added in the first process (Q1) is equal to the heat added in the second process (Q2)
  let Q1 := W1 + (3 / 2) * n * R * ΔT
  let Q2 := Q1
  -- Since the second process is isothermal, the work done in the second process W2 is exactly Q2
  let W2 := Q2
  -- Therefore, W2 = 75 holds under the given conditions
  sorry

end work_done_in_isothermal_process_l153_153185


namespace sequence_sum_l153_153010

theorem sequence_sum (a : ℕ → ℝ) (h_seq : ∀ n, a (n + 1) = (1/3) * a n) (h_a4a5 : a 4 + a 5 = 4) :
    a 2 + a 3 = 36 :=
    sorry

end sequence_sum_l153_153010


namespace bullet_speed_l153_153594

theorem bullet_speed (a l : ℝ) (h1 : a = 5 * 10^5) (h2 : l = 0.81) :
  sqrt (2 * a * l) = 9 * 10^2 :=
by
  sorry

end bullet_speed_l153_153594


namespace abs_eq_neg_imp_nonpos_l153_153830

theorem abs_eq_neg_imp_nonpos (a : ℝ) (h : |a| = -a) : a ≤ 0 :=
sorry

end abs_eq_neg_imp_nonpos_l153_153830


namespace intersecting_lines_value_k_l153_153493

theorem intersecting_lines_value_k :
  (∀ x, y, (y = -2 * x + 3) → (y = (k * x) + 9) → (x = 6) → (y = -9)) → (k = -3) :=
by
  intro h
  have h1 := h 6 (-9)
  have h2 : (-9 = -2 * 6 + 3), by linarith
  simp at h2
  have h3 : (-9 = 6 * k + 9), by apply h1; linarith
  linarith
  sorry

end intersecting_lines_value_k_l153_153493


namespace circle_area_l153_153453

theorem circle_area (r : ℝ) (h : 3 * (1 / (2 * π * r)) = r) : 
  π * r^2 = 3 / 2 :=
by
  sorry

end circle_area_l153_153453


namespace solution_set_l153_153412

-- Define the given function f
def f (m : ℝ) (x : ℝ) : ℝ := Real.log (4^x + 1) / Real.log 2 + m * x

-- State the quadratic equation in terms of t
def quadratic_eqn (m t : ℝ) : ℝ := 2 * t^2 - 2 * t + 4 / m - 4

-- Prove that m must be in the interval (8/7, 1] for exactly two real solutions in t
theorem solution_set (m : ℝ) : 
  (m > 8 / 7 ∧ m ≤ 1) ↔ 
  (∃! t1 t2 : ℝ, 
     2 * t1^2 - 2 * t1 + 4 / m - 4 = 0 ∧ 
     2 * t2^2 - 2 * t2 + 4 / m - 4 = 0 ∧ 
     t1 ≠ t2 ∧ 
     0 ≤ t1 ∧ t1 ≤ 3 / 2 ∧ 
     0 ≤ t2 ∧ t2 ≤ 3 / 2) := 
by 
  sorry

end solution_set_l153_153412


namespace pages_written_per_month_l153_153525

theorem pages_written_per_month 
  (d : ℕ) (days_in_month : ℕ) (letters_freq : ℕ) (time_per_letter : ℕ) 
  (time_per_page : ℕ) (long_letter_time_ratio : ℕ) (long_letter_time : ℕ) :
  d = 3 →
  days_in_month = 30 →
  letters_freq = 10 →
  time_per_letter = 20 →
  time_per_page = 10 →
  long_letter_time_ratio = 2 →
  long_letter_time = 80 →
  (days_in_month / letters_freq * time_per_letter / time_per_page) +
  (long_letter_time / (time_per_page * long_letter_time_ratio)) = 24 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h2, h3, h4, h5, h6, h7]
  simp
  exact rfl

#eval pages_written_per_month

end pages_written_per_month_l153_153525


namespace problem1_problem2_l153_153904

-- Define the conditions and the target proofs based on identified questions and answers

-- Problem 1
theorem problem1 (x : ℚ) : 
  9 * (x - 2)^2 ≤ 25 ↔ x = 11 / 3 ∨ x = 1 / 3 :=
sorry

-- Problem 2
theorem problem2 (x y : ℚ) :
  (x + 1) / 3 = 2 * y ∧ 2 * (x + 1) - y = 11 ↔ x = 5 ∧ y = 1 :=
sorry

end problem1_problem2_l153_153904


namespace geometric_series_first_term_l153_153695

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l153_153695


namespace sqrt_value_l153_153592

theorem sqrt_value (a : ℝ) (h : a = -2) : √(2 - a) = 2 :=
by
  sorry

end sqrt_value_l153_153592


namespace remainder_7n_mod_4_l153_153258

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l153_153258


namespace f_g_2_eq_1_l153_153100

def f (x : ℝ) : ℝ := x^2
def g (x : ℝ) : ℝ := -2 * x + 5

theorem f_g_2_eq_1 : f (g 2) = 1 :=
by
  sorry

end f_g_2_eq_1_l153_153100


namespace distinct_roots_implies_m_greater_than_half_find_m_given_condition_l153_153930

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end distinct_roots_implies_m_greater_than_half_find_m_given_condition_l153_153930


namespace max_difference_exists_l153_153660

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l153_153660


namespace remainder_when_7n_divided_by_4_l153_153248

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l153_153248


namespace pascals_triangle_contains_47_once_l153_153073

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153073


namespace find_constant_c_l153_153302

theorem find_constant_c 
  (v u g : ℝ) 
  (h_vg : g ≠ 0) :
  ∃ (c : ℝ), 
    (∀ (θ : ℝ), 
      0 ≤ θ ∧ θ ≤ real.pi → 
      let t := (v * real.sin θ + u) / g in
      let x := v * t * real.cos θ in
      let y := (v * t * real.sin θ + u * t) - (1 / 2) * g * t^2 in  
      (∀ (highest_point : ℝ × ℝ), 
        highest_point = (x, y) → 
        let A := (v^2 + u^2) / (2 * g) in
        let B := (v^2 + u^2) / (4 * g) in
        ∃ (area : ℝ), area = real.pi * A * B)) ∧
    c = real.pi / 8 :=
begin
  sorry
end

end find_constant_c_l153_153302


namespace first_term_geometric_series_l153_153700

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l153_153700


namespace remainder_7n_mod_4_l153_153255

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l153_153255


namespace first_term_geometric_series_l153_153698

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l153_153698


namespace total_distance_cars_l153_153716

theorem total_distance_cars :
  ∀ (fuel_dist_ratio_A fuel_dist_ratio_B : ℝ×ℝ) (fuel_used_A fuel_used_B : ℝ),
  fuel_dist_ratio_A = (4, 7) →
  fuel_dist_ratio_B = (3, 5) →
  fuel_used_A = 44 →
  fuel_used_B = 27 →
  let distance_A := (fuel_used_A * (fuel_dist_ratio_A.2 / fuel_dist_ratio_A.1))
  in let distance_B := (fuel_used_B * (fuel_dist_ratio_B.2 / fuel_dist_ratio_B.1))
  in distance_A + distance_B = 122 :=
by
  intros fuel_dist_ratio_A fuel_dist_ratio_B fuel_used_A fuel_used_B h1 h2 h3 h4
  have distance_A := (fuel_used_A * (fuel_dist_ratio_A.2 / fuel_dist_ratio_A.1))
  have distance_B := (fuel_used_B * (fuel_dist_ratio_B.2 / fuel_dist_ratio_B.1))
  show distance_A + distance_B = 122
  sorry

end total_distance_cars_l153_153716


namespace frisbee_total_distance_correct_l153_153330

-- Define the conditions
def bess_distance_per_throw : ℕ := 20 * 2 -- 20 meters out and 20 meters back = 40 meters
def bess_number_of_throws : ℕ := 4
def holly_distance_per_throw : ℕ := 8
def holly_number_of_throws : ℕ := 5

-- Calculate total distances
def bess_total_distance : ℕ := bess_distance_per_throw * bess_number_of_throws
def holly_total_distance : ℕ := holly_distance_per_throw * holly_number_of_throws
def total_distance : ℕ := bess_total_distance + holly_total_distance

-- The proof statement
theorem frisbee_total_distance_correct :
  total_distance = 200 :=
by
  -- proof goes here (we use sorry to skip the proof)
  sorry

end frisbee_total_distance_correct_l153_153330


namespace minimum_value_expression_l153_153357

theorem minimum_value_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x ^ 2 + 4 * x * y + 2 * y ^ 2 - 6 * x + 8 * y + 9 ≥ -10 :=
by
  sorry

end minimum_value_expression_l153_153357


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153079

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153079


namespace possible_combinations_of_scores_l153_153125

theorem possible_combinations_of_scores 
    (scores : Set ℕ := {0, 3, 5})
    (total_scores : ℕ := 32)
    (teams : ℕ := 3)
    : (∃ (number_of_combinations : ℕ), number_of_combinations = 255) := by
  sorry

end possible_combinations_of_scores_l153_153125


namespace ratio_men_to_women_on_team_l153_153610

theorem ratio_men_to_women_on_team (M W : ℕ) 
  (h1 : W = M + 6) 
  (h2 : M + W = 24) : 
  M / W = 3 / 5 := 
by 
  sorry

end ratio_men_to_women_on_team_l153_153610


namespace systematic_sampling_sequence_l153_153121

theorem systematic_sampling_sequence :
  ∀ (s : List Nat), (0 < s.head) ∧ (s.head ≤ 60) ∧ 
    (List.length s = 5) ∧ (s.head = 4) → 
    (s = [4, 16, 28, 40, 52]) :=
begin
  sorry
end

end systematic_sampling_sequence_l153_153121


namespace square_of_binomial_eq_100_l153_153579

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l153_153579


namespace value_of_k_l153_153587

theorem value_of_k (k : ℕ) : (∃ b : ℕ, x^2 - 20 * x + k = (x + b)^2) → k = 100 := by
  sorry

end value_of_k_l153_153587


namespace clock_hands_meeting_duration_l153_153657

noncomputable def angle_between_clock_hands (h m : ℝ) : ℝ :=
  abs ((30 * h + m / 2) - (6 * m) % 360)

theorem clock_hands_meeting_duration : 
  ∃ n m : ℝ, 0 <= n ∧ n < m ∧ m < 60 ∧ angle_between_clock_hands 5 n = 120 ∧ angle_between_clock_hands 5 m = 120 ∧ m - n = 44 :=
sorry

end clock_hands_meeting_duration_l153_153657


namespace usamo_pages_sum_l153_153200

def sum_of_pages (a : Fin 6 → ℕ) : ℕ :=
  ∑ k, a k

theorem usamo_pages_sum (a : Fin 6 → ℕ) :
  (∑ k, (a k + 1 : ℕ)) / 2 = 2017 → sum_of_pages a = 4028 := by
  intro h
  sorry

end usamo_pages_sum_l153_153200


namespace smallest_n_identity_matrix_l153_153746

noncomputable def rotation_45_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (Real.pi / 4), -Real.sin (Real.pi / 4)],
    ![Real.sin (Real.pi / 4), Real.cos (Real.pi / 4)]
  ]

theorem smallest_n_identity_matrix : ∃ n : ℕ, n > 0 ∧ (rotation_45_matrix ^ n = 1) ∧ ∀ m : ℕ, m > 0 → (rotation_45_matrix ^ m = 1 → n ≤ m) := sorry

end smallest_n_identity_matrix_l153_153746


namespace find_m_l153_153932

theorem find_m (m : ℝ) (x1 x2 : ℝ) 
  (h_eq : x1 ^ 2 - 4 * x1 - 2 * m + 5 = 0)
  (h_distinct : x1 ≠ x2)
  (h_product_sum_eq : x1 * x2 + x1 + x2 = m ^ 2 + 6) : 
  m = 1 ∧ m > 1/2 :=
sorry

end find_m_l153_153932


namespace systematic_sampling_interval_l153_153464

-- Define the problem statement
theorem systematic_sampling_interval (N n : ℕ) (h1 : N ≥ n) : 
  (N != 0 ∧ n != 0) → ∀ k : ℕ, k = N / n → interval_for_segmentation = ⌊N / n⌋ := 
by
  sorry

end systematic_sampling_interval_l153_153464


namespace inequality_hold_l153_153437

theorem inequality_hold {a b : ℝ} (h : a < b) : -3 * a > -3 * b :=
sorry

end inequality_hold_l153_153437


namespace max_difference_exists_l153_153663

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l153_153663


namespace domain_of_f_l153_153031

def f (x : ℝ) : ℝ := Real.log (1 - x)

theorem domain_of_f :
  (Set.Ioc 0 1) = {x | ∃ y, f x = y ∧ y < 0} :=
by
  sorry

end domain_of_f_l153_153031


namespace smallest_5_digit_integer_congruent_21_mod_30_l153_153574

theorem smallest_5_digit_integer_congruent_21_mod_30 : 
  ∃ n : ℕ, n ≥ 10000 ∧ n < 100000 ∧ n % 30 = 21 ∧ n = 10011 :=
by
  use 10011
  split; only [apply and.intro]
  · show 10000 ≤ 10011
    sorry
  · split; only [apply and.intro]
    · show 10011 < 100000
      sorry
    · split; only [apply and.intro] 
      · show 10011 % 30 = 21
        sorry
      · show 10011 = 10011
        sorry

end smallest_5_digit_integer_congruent_21_mod_30_l153_153574


namespace intersection_of_A_and_B_l153_153424

theorem intersection_of_A_and_B :
  let A := {0, 1, 2, 3, 4}
  let B := {x | ∃ n ∈ A, x = 2 * n}
  A ∩ B = {0, 2, 4} :=
by
  sorry

end intersection_of_A_and_B_l153_153424


namespace total_students_in_lab_l153_153626

def total_workstations : Nat := 16
def workstations_for_2_students : Nat := 10
def students_per_workstation_2 : Nat := 2
def students_per_workstation_3 : Nat := 3

theorem total_students_in_lab :
  let workstations_with_2_students := workstations_for_2_students
  let workstations_with_3_students := total_workstations - workstations_for_2_students
  let students_in_2_student_workstations := workstations_with_2_students * students_per_workstation_2
  let students_in_3_student_workstations := workstations_with_3_students * students_per_workstation_3
  students_in_2_student_workstations + students_in_3_student_workstations = 38 :=
by
  sorry

end total_students_in_lab_l153_153626


namespace last_four_digits_of_5_pow_9000_l153_153597

theorem last_four_digits_of_5_pow_9000 (h : 5^300 ≡ 1 [MOD 1250]) : 
  5^9000 ≡ 1 [MOD 1250] :=
sorry

end last_four_digits_of_5_pow_9000_l153_153597


namespace unique_solution_triplet_l153_153738

theorem unique_solution_triplet :
  ∃! (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (x^y + y^x = z^y ∧ x^y + 2012 = y^(z+1)) ∧ (x = 6 ∧ y = 2 ∧ z = 10) := 
by {
  sorry
}

end unique_solution_triplet_l153_153738


namespace centerville_remaining_budget_l153_153943

def centerville_budget (total_budget : ℝ) : Prop :=
  (0.15 * total_budget = 3000) ∧
  ((total_budget - 3000 - (0.24 * total_budget)) = 12200)

theorem centerville_remaining_budget : 
  ∃ (total_budget : ℝ), centerville_budget total_budget :=
by
  use 20000
  unfold centerville_budget
  split
  · simp
  · ring
  sorry -- Finish the proof accordingly

end centerville_remaining_budget_l153_153943


namespace vertical_asymptotes_count_l153_153821

noncomputable def f (x : ℝ) : ℝ := (x - 2) / (x^2 + 4 * x - 12)

theorem vertical_asymptotes_count : 
  {x : ℝ | ∃ (l : ℝ), tendsto f (𝓝 x) (𝓝 l) = false}.count = 1 :=
sorry

end vertical_asymptotes_count_l153_153821


namespace funding_problem_l153_153287

def funding_allocation (F : ℕ) : ℕ :=
  if F = 204600000 then 0
  else 1 + funding_allocation (F / 2 - 10000)

theorem funding_problem
  (F : ℕ)
  (h : funding_allocation F = 0) :
  F = 204600000 :=
sorry

end funding_problem_l153_153287


namespace max_value_expression_l153_153008

-- Define the function d(a, b) in Lean
def d (p : ℕ) [hp : Fact (Nat.Prime p)] (a b : ℕ) : ℕ :=
  ∑ c in Finset.range p, if 1 ≤ c ∧ c < p ∧ (ac % p ≤ p/3 ∧ bc % p ≤ p/3) then 1 else 0

noncomputable def value_expression (p : ℕ) [hp : Fact (Nat.Prime p)] (xs : Fin p → ℝ) : ℝ :=
  let sum1 := ∑ a in Finset.range (p - 1), ∑ b in Finset.range (p - 1), (d p a b) * ((xs a) + 1) * ((xs b) + 1)
  let sum2 := ∑ a in Finset.range (p - 1), ∑ b in Finset.range (p - 1), (d p a b) * (xs a) * (xs b)
  (Real.sqrt sum1) - (Real.sqrt sum2)

theorem max_value_expression (p : ℕ) [hp : Fact (Nat.Prime p)] (xs : Fin p → ℝ) :
  value_expression p xs = Real.sqrt (p - 1) :=
sorry

end max_value_expression_l153_153008


namespace polygon_triangle_existence_l153_153282

theorem polygon_triangle_existence (n : ℕ) (h₁ : n > 1)
  (h₂ : ∀ (k₁ k₂ : ℕ), k₁ ≠ k₂ → (4 ≤ k₁) → (4 ≤ k₂) → k₁ ≠ k₂) :
  ∃ k, k = 3 :=
by
  sorry

end polygon_triangle_existence_l153_153282


namespace megan_seashells_l153_153173

theorem megan_seashells :
  ∀ (initial_seashells : ℕ) (additional_seashells : ℕ),
  initial_seashells = 19 → additional_seashells = 6 →
  initial_seashells + additional_seashells = 25 :=
by
  intros initial_seashells additional_seashells h_initial h_additional
  rw [h_initial, h_additional]
  rfl

end megan_seashells_l153_153173


namespace union_M_N_l153_153038

-- Definitions for the sets M and N
def M : Set ℝ := { x | x^2 = x }
def N : Set ℝ := { x | Real.log x / Real.log 2 ≤ 0 }

-- Proof problem statement
theorem union_M_N : M ∪ N = Set.Icc 0 1 := by
  sorry

end union_M_N_l153_153038


namespace no_adjacent_people_stand_probability_l153_153362

theorem no_adjacent_people_stand_probability :
  let outcomes := 2^5,
  let favorable_outcomes := 11 in
  (favorable_outcomes / outcomes : ℚ) = 11 / 32 :=
by
  -- Problem statement and conditions
  have outcomes_def : outcomes = 32 := by norm_num,
  have favorable_outcomes_def : favorable_outcomes = 11 := by norm_num,
  sorry

end no_adjacent_people_stand_probability_l153_153362


namespace ball_distribution_possible_l153_153949

theorem ball_distribution_possible 
  (colors : Finₓ 20 → Nat)
  (h_each_color_has_10_balls : ∀ i, 10 ≤ colors i)
  (h_total_800_balls : ∑ i, colors i = 800)
  (students : Finₓ 20 → Nat)
  (h_each_student_same_number_of_balls : ∀ i, students i = 40) :
  ∃ boxes : Finₓ 80 → Nat, 
    (∀ i, 10 ≤ boxes i ∧  ∃ color, boxes i ≤ colors color) ∧ 
    (∑ i, boxes i = 800) ∧ 
    (∃ f : Finₓ 80 → Finₓ 20, ∀ j, ∑ i in (Finset.univ.filter (λ k, f k = j)), boxes i = students j) :=
sorry

end ball_distribution_possible_l153_153949


namespace minimize_perimeter_l153_153158

def point (x : ℝ) (y : ℝ) : ℝ × ℝ := (x, y)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def perimeter (A B C : ℝ × ℝ) : ℝ :=
  distance A B + distance A C + distance B C

theorem minimize_perimeter :
  let A := point 7 3
  let B := point (-2) 3
  let C (k : ℝ) := point 0 k
  (∀ k : ℝ, k = 3 → perimeter A B (C k) = 18) :=
sorry

end minimize_perimeter_l153_153158


namespace part1_part2_l153_153801

open Real

def f (x : ℝ) : ℝ := abs (x + 1) + abs (x - 3)

theorem part1 (a : ℝ) : (∃ x : ℝ, f x < a) ↔ (a > 4) :=
by sorry

theorem part2 (a b : ℝ) (h : ∀ x : ℝ, f x < a ↔ x ∈ set.Ioo b (7 / 2)) : a + b = 3.5 :=
by sorry

end part1_part2_l153_153801


namespace min_n_for_conditions_l153_153769

open Real

theorem min_n_for_conditions
  (n : ℕ)
  (x : Fin n → ℝ)
  (h_abs_lt : ∀ i, |x i| < 1)
  (h_sum_ge : ∑ i, |x i| ≥ 19 + |∑ i, x i|) :
  n ≥ 20 :=
sorry

end min_n_for_conditions_l153_153769


namespace find_lambda_l153_153817

-- Define vector addition and subtraction
def vector_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)
def vector_sub (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 - v2.1, v1.2 - v2.2)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Define vectors m and n
def m (λ : ℝ) : ℝ × ℝ := (λ + 1, 1)
def n (λ : ℝ) : ℝ × ℝ := (λ + 2, 2)

-- State the theorem to be proven
theorem find_lambda (λ : ℝ) : dot_product (vector_add (m λ) (n λ)) (vector_sub (m λ) (n λ)) = 0 → λ = -3 := by
  sorry

end find_lambda_l153_153817


namespace jill_arrives_30_minutes_before_jack_l153_153212

theorem jill_arrives_30_minutes_before_jack
    (d : ℝ) (s_jill : ℝ) (s_jack : ℝ) (t_diff : ℝ)
    (h_d : d = 2)
    (h_s_jill : s_jill = 12)
    (h_s_jack : s_jack = 3)
    (h_t_diff : t_diff = 30) :
    ((d / s_jack) * 60 - (d / s_jill) * 60) = t_diff :=
by
  sorry

end jill_arrives_30_minutes_before_jack_l153_153212


namespace ellipse_foci_xaxis_range_A_G_N_collinear_l153_153028

noncomputable def curve (m : ℝ) : ℝ → ℝ → Prop :=
  λ x y, (5 - m) * x^2 + (m - 2) * y^2 = 8

theorem ellipse_foci_xaxis_range (m : ℝ) : (5 - m > 0) ∧ (m - 2 > 0) ∧ ((5 - m) / (m - 2) > 0) → (7 / 2) < m ∧ m < 5 :=
sorry

theorem A_G_N_collinear
    (xₘ xₙ k : ℝ)
    (A : ℝ × ℝ)
    (G : ℝ × ℝ)
    (N : ℝ × ℝ)
    (m = 4)
    (curve 4 x y)
    (y = k*x + 4)
    (yₘ = k*xₘ + 4)
    (yₙ = k*xₙ + 4)
    (B : ℝ × ℝ)
    (A = (0, 2))
    (B = (0, -2))
    (M = (xₘ, yₘ))
    (N = (xₙ, yₙ))
    (xₙ ≠ xₘ)
    (G = (x, 1))
    := collinear ([A, G, N] : List (ℝ × ℝ)) :=
sorry

end ellipse_foci_xaxis_range_A_G_N_collinear_l153_153028


namespace charles_paper_count_l153_153717

theorem charles_paper_count :
  let papers_today := 6
  let papers_before_work := 6
  let papers_after_work := 6
  let papers_left := 2
  let total_pictures := papers_today + papers_before_work + papers_after_work
  let used_papers := total_pictures
  let initial_papers := used_papers + papers_left
  in
  initial_papers = 20 := 
by 
  sorry

end charles_paper_count_l153_153717


namespace total_paint_correct_l153_153176

def monday_paint : ℕ := 30

def tuesday_paint : ℕ := 2 * monday_ppaint

def wednesday_paint : ℕ := monday_paint / 2

def total_paint : ℕ := monday_paint + tuesday_paint + wednesday_paint

theorem total_paint_correct : total_paint = 105 := by
  sorry

end total_paint_correct_l153_153176


namespace total_pages_written_is_24_l153_153527

def normal_letter_interval := 3
def time_per_normal_letter := 20
def time_per_page := 10
def additional_time_factor := 2
def time_spent_long_letter := 80
def days_in_month := 30

def normal_letters_written := days_in_month / normal_letter_interval
def pages_per_normal_letter := time_per_normal_letter / time_per_page
def total_pages_normal_letters := normal_letters_written * pages_per_normal_letter

def time_per_page_long_letter := additional_time_factor * time_per_page
def pages_long_letter := time_spent_long_letter / time_per_page_long_letter

def total_pages_written := total_pages_normal_letters + pages_long_letter

theorem total_pages_written_is_24 : total_pages_written = 24 := by
  sorry

end total_pages_written_is_24_l153_153527


namespace matrix_inverse_identity_l153_153449

theorem matrix_inverse_identity 
  (A : Matrix n n ℝ) 
  (hA_inv : invertible A) 
  (h : (A - 3 • (1 : Matrix n n ℝ)) * (A - 5 • (1 : Matrix n n ℝ)) = 0) :
  A + 9 • (A⁻¹) = A + 4.8 • (1 : Matrix n n ℝ) :=
by sorry

end matrix_inverse_identity_l153_153449


namespace find_parallel_line_l153_153835

def direction_vector (x y z : ℝ) := (x, y, z)
def normal_vector (x y z : ℝ) := (x, y, z)

def dot_product (u v : (ℝ × ℝ × ℝ)) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

theorem find_parallel_line :
  (dot_product (direction_vector 1 (-1) 3) (normal_vector 0 3 1)) = 0 :=
by
  sorry

end find_parallel_line_l153_153835


namespace diagramD_non_eulerian_l153_153269

-- Given conditions
def hasEulerianPath (G : SimpleGraph) : Prop :=
  let odd_degree_vertices := {v | G.degree v % 2 = 1}
  odd_degree_vertices.card = 0 ∨ odd_degree_vertices.card = 2

-- Given information for our specific diagrams (We create a specific name DiagramD to be explicit)
variable (DiagramD : SimpleGraph)

-- The theorem to prove
theorem diagramD_non_eulerian (h : {v | DiagramD.degree v % 2 = 1}.card ≠ 0 ∧ {v | DiagramD.degree v % 2 = 1}.card ≠ 2) : ¬ hasEulerianPath DiagramD :=
by sorry

end diagramD_non_eulerian_l153_153269


namespace maximum_norm_c_l153_153395

open Real

variables {a b c : ℝ → ℝ} -- Assuming vectors are functions from ℝ to ℝ for simplicity.
-- Add specific assumptions about unit vectors and given condition
-- In Lean, vectors are represented differently, but assume simplistic real-valued function representation for simplicity here

-- Definition of unit vectors
def is_unit_vector (v : ℝ → ℝ) : Prop := ∥v∥ = 1

-- The condition given in the problem
def satisfies_condition (a b c : ℝ → ℝ) : Prop := 
  ∥c - (a + b)∥ = ∥a - b∥

-- The main statement that needs to be proven
theorem maximum_norm_c (a b c : ℝ → ℝ) 
  (ha : is_unit_vector a) 
  (hb : is_unit_vector b) 
  (hc : satisfies_condition a b c) : 
  ∥c∥ ≤ 2 * sqrt 2 :=
sorry

end maximum_norm_c_l153_153395


namespace probability_at_least_9_heads_l153_153982

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153982


namespace max_ab_ineq_l153_153206

theorem max_ab_ineq {a b : ℝ} (ha : 0 < a) (hb : 0 < b) (h : a + 4 * b = 1) : ab ≤ 1 / 16 :=
by
  apply (am_gm_eq _ _ h)
  sorry

-- This statement declares that given positive real numbers a and b
-- such that a + 4b = 1, the maximum value of ab is 1/16.

end max_ab_ineq_l153_153206


namespace area_square_l153_153209

-- Define the conditions
variables (l r s : ℝ)
variable (breadth : ℝ := 10)
variable (area_rect : ℝ := 180)

-- Given conditions
def length_is_two_fifths_radius : Prop := l = (2/5) * r
def radius_is_side_square : Prop := r = s
def area_of_rectangle : Prop := area_rect = l * breadth

-- The theorem statement
theorem area_square (h1 : length_is_two_fifths_radius l r)
                    (h2 : radius_is_side_square r s)
                    (h3 : area_of_rectangle l breadth area_rect) :
  s^2 = 2025 :=
by
  sorry

end area_square_l153_153209


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153987

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153987


namespace total_distance_traveled_l153_153329

-- Definitions of conditions
def bess_throw_distance : ℕ := 20
def bess_throws : ℕ := 4
def holly_throw_distance : ℕ := 8
def holly_throws : ℕ := 5
def bess_effective_throw_distance : ℕ := 2 * bess_throw_distance

-- Theorem statement
theorem total_distance_traveled :
  (bess_throws * bess_effective_throw_distance + holly_throws * holly_throw_distance) = 200 := 
  by sorry

end total_distance_traveled_l153_153329


namespace car_safety_l153_153232

theorem car_safety (V_A V_B V_C : ℕ) (d_AB d_AC : ℕ) (h1: V_B = 50) (h2: V_C = 80) (h3: d_AB = 50) (h4: d_AC = 200):
  V_A <= 83 ∨ V_A <= 84 := 
by 
  have h5 : 50 * (V_A + V_C) <= 200 * (V_A - V_B),
    from sorry,
  have h6 : 150 * V_A >= (50 * V_C + 200 * V_B),
    from sorry,
  have h7 : V_A >= (50 * 80 + 200 * 50) / 150,
    from sorry,
  have h8: V_A <= 83 + 1,
  from sorry,
  exact or.inr h8

end car_safety_l153_153232


namespace LCM_quotient_l153_153485

-- Define M as the least common multiple of integers from 12 to 25
def LCM_12_25 : ℕ := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 
                       (Nat.lcm 12 13) 14) 15) 16) 17) (Nat.lcm (Nat.lcm 
                       (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 18 19) 20) 21) 22) 23) 24)

-- Define N as the least common multiple of LCM_12_25, 36, 38, 40, 42, 44, 45
def N : ℕ := Nat.lcm LCM_12_25 (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 36 38) 40) 42) (Nat.lcm 44 45))

-- Prove that N / LCM_12_25 = 1
theorem LCM_quotient : N / LCM_12_25 = 1 := by
    sorry

end LCM_quotient_l153_153485


namespace jim_total_payment_l153_153860

def lamp_cost : ℕ := 7
def bulb_cost : ℕ := lamp_cost - 4
def num_lamps : ℕ := 2
def num_bulbs : ℕ := 6

def total_cost : ℕ := (num_lamps * lamp_cost) + (num_bulbs * bulb_cost)

theorem jim_total_payment : total_cost = 32 := by
  sorry

end jim_total_payment_l153_153860


namespace smallest_sum_xyz_l153_153552

theorem smallest_sum_xyz (x y z : ℕ) (h : x * y * z = 40320) : x + y + z ≥ 103 :=
sorry

end smallest_sum_xyz_l153_153552


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153077

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153077


namespace laura_owe_l153_153606

noncomputable def principal : ℝ := 35
noncomputable def annual_rate : ℝ := 0.06
noncomputable def time_period : ℝ := 1
noncomputable def interest (P R T : ℝ) : ℝ := P * R * T
noncomputable def total_owed (P I : ℝ) : ℝ := P + I

theorem laura_owe (P R T : ℝ): P = 35 → R = 0.06 → T = 1 → total_owed P (interest P R T) = 37.10 :=
by {
  intros,
  sorry
}

end laura_owe_l153_153606


namespace solution_l153_153006

/-
Define the piecewise function f.
-/
def f (x : Real) : Real :=
  if x ≥ 1 then x^2 - 1 else x - 2

/-
The main theorem to prove: 
Proving that if f(f(a)) = 3, then a = sqrt(3).
-/
theorem solution (a : Real) (h : f (f a) = 3) : a = Real.sqrt 3 :=
  sorry

end solution_l153_153006


namespace find_max_difference_l153_153685

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l153_153685


namespace geometric_series_first_term_l153_153696

theorem geometric_series_first_term 
  (S : ℝ) (r : ℝ) (a : ℝ)
  (h_sum : S = 40) (h_ratio : r = 1/4) :
  S = a / (1 - r) → a = 30 := by
  sorry

end geometric_series_first_term_l153_153696


namespace general_formula_for_an_comparison_of_9T2n_and_Qn_l153_153163

-- Given conditions
def f1 (x : ℝ) : ℝ := 2 / (1 + x)

def f (n : ℕ) : ℝ → ℝ
| 0 => f1 
| (k+1) => f1 (f k)

def a (n : ℕ) : ℝ :=
(f n 0 - 1) / (f n 0 + 2)

def T (n : ℕ) : ℝ :=
Finset.sum (Finset.range (2 * n)) (λ i => (i + 1) * a (i + 1))

def Q (n : ℕ) : ℝ :=
(4 * n^2 + n) / (4 * n^2 + 4 * n + 1)

-- Proving that the sequence is as described and comparing sizes
theorem general_formula_for_an (n : ℕ) : 
a n = ¼ * (-½)^(n-1) :=
sorry

theorem comparison_of_9T2n_and_Qn (n : ℕ) : 
(if n ≥ 3 then 9 * T n > Q n else 9 * T n < Q n) :=
sorry

end general_formula_for_an_comparison_of_9T2n_and_Qn_l153_153163


namespace Kyle_throws_farther_l153_153707

theorem Kyle_throws_farther (Parker_distance : ℕ) (Grant_ratio : ℚ) (Kyle_ratio : ℚ) (Grant_distance : ℚ) (Kyle_distance : ℚ) :
  Parker_distance = 16 → 
  Grant_ratio = 0.25 → 
  Kyle_ratio = 2 → 
  Grant_distance = Parker_distance + Parker_distance * Grant_ratio → 
  Kyle_distance = Kyle_ratio * Grant_distance → 
  Kyle_distance - Parker_distance = 24 :=
by
  intros hp hg hk hg_dist hk_dist
  subst hp
  subst hg
  subst hk
  subst hg_dist
  subst hk_dist
  -- The proof steps are omitted
  sorry

end Kyle_throws_farther_l153_153707


namespace length_of_platform_l153_153313

-- Define the given conditions
def train_length : ℕ := 1020  -- in meters
def train_speed : ℕ := 102    -- in kmph
def crossing_time : ℕ := 50   -- in seconds

-- Define the correct answer
def platform_length : ℕ := 3230   -- expected result in meters

-- Prove that given conditions imply the correct answer for the platform length.
theorem length_of_platform :
  let speed_m_per_s := train_speed * 1000 / 3600,                  -- convert kmph to m/s
      distance_covered := speed_m_per_s * crossing_time in         -- calculate total distance covered by train in 50 s
  distance_covered - train_length = platform_length := sorry

end length_of_platform_l153_153313


namespace opposite_fraction_l153_153553

def opposite (x : ℚ) : ℚ := -x

theorem opposite_fraction :
  opposite (1 / 2023) = - (1 / 2023) :=
by
  simp [opposite]
  sorry

end opposite_fraction_l153_153553


namespace line_always_passes_fixed_point_l153_153502

theorem line_always_passes_fixed_point (k : ℝ) : 
  ∃ fixed_point : ℝ × ℝ, ∀ k : ℝ, let line : ℝ × ℝ × ℝ := (2k - 1, -(k + 3), k - 11) in 
  fixed_point = (2, 3) ∧ (line.1 * fixed_point.1 + line.2 * fixed_point.2 + line.3 = 0) :=
sorry

end line_always_passes_fixed_point_l153_153502


namespace victor_final_usd_l153_153566

variable (initial_rubles : ℝ) 
variable (term_years : ℕ) 
variable (annual_rate : ℝ) 
variable (buy_rate : ℝ) 
variable (sell_rate : ℝ)

def rubles_to_usd (rubles : ℝ) (rate : ℝ) : ℝ :=
  rubles / rate

def compound (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem victor_final_usd :
  initial_rubles = 45000 →
  term_years = 2 →
  annual_rate = 0.047 →
  buy_rate = 59.60 →
  sell_rate = 56.60 →
  let P := rubles_to_usd initial_rubles sell_rate in
  round (compound P annual_rate 4 term_years) = 874 :=
by
  intros h1 h2 h3 h4 h5
  let P := rubles_to_usd initial_rubles sell_rate
  sorry

end victor_final_usd_l153_153566


namespace geometric_sequence_general_formula_first_n_terms_sum_l153_153845

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^n

def S_n (n : ℕ) : ℕ :=
  if n = 1 then 2 else 2^(n + 1) - 4 * n + 2

theorem geometric_sequence_general_formula :
  ∃ q c : ℕ, (∀ n : ℕ, a n = c * q^n) ∧ (a 1 = 2) ∧ (a 2 + 1 - a 3 = 1) ∧ (a 4 = 8 * a 1) :=
sorry

theorem first_n_terms_sum (n : ℕ) :
  S_n n = (∑ i in finset.range n, abs (a i - 4)) :=
sorry

end geometric_sequence_general_formula_first_n_terms_sum_l153_153845


namespace sample_mean_y_l153_153809

theorem sample_mean_y {x y : Type} 
  (reg_eq : ∀ x, y = 0.6 * x - 0.5)
  (mean_x : y = 5) : 
  y = 2.5 := 
by
  sorry

end sample_mean_y_l153_153809


namespace square_of_binomial_eq_100_l153_153576

-- Given conditions
def is_square_of_binomial (p : ℝ[X]) : Prop :=
  ∃ b : ℝ, p = (X + C b)^2

-- The equivalent proof problem statement
theorem square_of_binomial_eq_100 (k : ℝ) :
  is_square_of_binomial (X^2 - 20 * X + C k) → k = 100 :=
by
  sorry

end square_of_binomial_eq_100_l153_153576


namespace first_term_geometric_series_l153_153699

theorem first_term_geometric_series (r a S : ℝ) (h1 : r = 1 / 4) (h2 : S = 40) (h3 : S = a / (1 - r)) : a = 30 :=
by
  sorry

end first_term_geometric_series_l153_153699


namespace cos_sum_zero_l153_153334

noncomputable def cos_sum : ℂ :=
  Real.cos (Real.pi / 15) + Real.cos (4 * Real.pi / 15) + Real.cos (7 * Real.pi / 15) + Real.cos (10 * Real.pi / 15)

theorem cos_sum_zero : cos_sum = 0 := by
  sorry

end cos_sum_zero_l153_153334


namespace probability_heads_in_12_flips_l153_153995

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153995


namespace inequality_bound_l153_153892

noncomputable def p : ℝ := sorry
noncomputable def q : ℝ := sorry

theorem inequality_bound
  (p q : ℝ)
  (h : p > 0 ∧ q > 0)
  (h₁ : 1/p + 1/q = 1) :
  1/(p * sqrt p) + 1/(q * sqrt q) < 1 := sorry

end inequality_bound_l153_153892


namespace total_infections_second_wave_l153_153732

noncomputable def f (x : ℝ) := 300 * Real.exp (0.1 * x)

theorem total_infections_second_wave : (∫ x in 0..14, f x) = 9150 :=
by
  sorry

end total_infections_second_wave_l153_153732


namespace exists_triangle_with_angle_leq_45_l153_153763

-- Define the type for points in a plane
structure Point where
  x : ℝ
  y : ℝ

-- Predicate that specifies that no three points are collinear
def no_three_collinear (a b c d : Point) : Prop := 
  (a.x * (b.y - c.y) + b.x * (c.y - a.y) + c.x * (a.y - b.y) ≠ 0) ∧ 
  (a.x * (b.y - d.y) + b.x * (d.y - a.y) + d.x * (a.y - b.y) ≠ 0) ∧ 
  (a.x * (c.y - d.y) + c.x * (d.y - a.y) + d.x * (a.y - c.y) ≠ 0) ∧ 
  (b.x * (c.y - d.y) + c.x * (d.y - b.y) + d.x * (b.y - c.y) ≠ 0)

-- Theorem: Among four non-collinear points on a plane, there exists a triangle with an angle not exceeding 45 degrees.
theorem exists_triangle_with_angle_leq_45 (a b c d : Point) (h : no_three_collinear a b c d) : 
  ∃ (p q r : Point), 
    p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ 
    (p = a ∨ p = b ∨ p = c ∨ p = d) ∧ 
    (q = a ∨ q = b ∨ q = c ∨ q = d) ∧ 
    (r = a ∨ r = b ∨ r = c ∨ r = d) ∧ 
    (∃ θ : ℝ, θ ≤ 45 ∧ is_angle_of_triangle p q r θ) :=
sorry

end exists_triangle_with_angle_leq_45_l153_153763


namespace cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l153_153689

theorem cannot_be_expressed_as_difference_of_squares (a b : ℤ) (h : 2006 = a^2 - b^2) : False := sorry

theorem can_be_expressed_as_difference_of_squares_2004 : ∃ (a b : ℤ), 2004 = a^2 - b^2 := by
  use 502, 500
  norm_num

theorem can_be_expressed_as_difference_of_squares_2005 : ∃ (a b : ℤ), 2005 = a^2 - b^2 := by
  use 1003, 1002
  norm_num

theorem can_be_expressed_as_difference_of_squares_2007 : ∃ (a b : ℤ), 2007 = a^2 - b^2 := by
  use 1004, 1003
  norm_num

end cannot_be_expressed_as_difference_of_squares_can_be_expressed_as_difference_of_squares_2004_can_be_expressed_as_difference_of_squares_2005_can_be_expressed_as_difference_of_squares_2007_l153_153689


namespace ivan_compensation_l153_153856

noncomputable def initial_deposit : ℝ := 100000
noncomputable def insurance_limit : ℝ := 1400000
noncomputable def insured_event := true
noncomputable def within_limit (d : ℝ) : Prop := d ≤ insurance_limit

theorem ivan_compensation (d : ℝ) (h₁ : d = initial_deposit) (h₂ : insured_event) (h₃ : within_limit d) : 
  ∃ c : ℝ, c = d + accrued_interest d :=
sorry

end ivan_compensation_l153_153856


namespace area_ABC_is_35_l153_153570

open Point

-- Define points A, B, and C
def A : Point := ⟨ -2, 3 ⟩
def B : Point := ⟨ 8, 3 ⟩
def C : Point := ⟨ 6, -4 ⟩

-- Definition for the area of triangle ABC
def areaOfTriangle (A B C : Point) : ℝ :=
  0.5 * (B.x - A.x) * (A.y - C.y)

-- Theorem stating the area of triangle ABC is 35 square units
theorem area_ABC_is_35 :
  areaOfTriangle A B C = 35 := by
  sorry

end area_ABC_is_35_l153_153570


namespace cost_of_lamps_and_bulbs_l153_153863

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end cost_of_lamps_and_bulbs_l153_153863


namespace problem_solution_l153_153423

noncomputable def a (n : ℕ) : ℝ := (1/3)^n
noncomputable def S (n : ℕ) : ℕ → ℝ
| 0 => 0
| (m+1) => S m + a (m+1)

noncomputable def b (n : ℕ) : ℝ := Real.logb (3 : ℝ) (a (2*n - 1))
noncomputable def c (n : ℕ) : ℝ :=
  let bn := b n
  let bnp1 := b (n + 1)
  (4 * n^2) / (bn * bnp1)

noncomputable def T (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, c (i + 1)

theorem problem_solution (n : ℕ) (h₁ : a n = (1/3)^n) (h₂ : b n = 2*n - 1):
  T 2016 <= 2016 + 2016 / 4033 ∧ (⌊T 2016⌋ : ℝ) = 2016 :=
sorry

end problem_solution_l153_153423


namespace preimage_of_3_2_eq_l153_153405

noncomputable def f (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1 * p.2, p.1 + p.2)

theorem preimage_of_3_2_eq (x y : ℝ) :
  f (x, y) = (-3, 2) ↔ (x = 3 ∧ y = -1) ∨ (x = -1 ∧ y = 3) :=
by
  sorry

end preimage_of_3_2_eq_l153_153405


namespace solve_for_q_l153_153702

theorem solve_for_q (t h q : ℝ) (h_eq : h = -14 * (t - 3)^2 + q) (h_5_eq : h = 94) (t_5_eq : t = 3 + 2) : q = 150 :=
by
  sorry

end solve_for_q_l153_153702


namespace passion_fruit_crates_l153_153151

def total_crates : ℕ := 50
def crates_of_grapes : ℕ := 13
def crates_of_mangoes : ℕ := 20
def crates_of_fruits_sold := crates_of_grapes + crates_of_mangoes

theorem passion_fruit_crates (p : ℕ) : p = total_crates - crates_of_fruits_sold :=
by
  have h1 : crates_of_fruits_sold = 13 + 20 := by rfl
  have h2 : p = 50 - 33 := by rw [h1]; rfl
  show p = 17 from h2

end passion_fruit_crates_l153_153151


namespace quadratic_real_roots_l153_153368

theorem quadratic_real_roots (m : ℝ) : 
  (∃ x : ℝ, x^2 - 2*x + m = 0) → m ≤ 1 :=
by sorry

example : ¬(1.2 ≤ 1) :=
by norm_num

end quadratic_real_roots_l153_153368


namespace number_of_parents_l153_153708

theorem number_of_parents (n m : ℕ) 
  (h1 : n + m = 31) 
  (h2 : 15 + m = n) 
  : n = 23 := 
by 
  sorry

end number_of_parents_l153_153708


namespace coin_flip_heads_probability_l153_153957

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153957


namespace part_I_min_part_I_max_part_II_monotonicity_part_III_range_of_a_l153_153799

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x + (a + 1) / 2 * x^2 + 1

theorem part_I_min (a : ℝ) (h : a = -1/2) :
  ∀ x ∈ Set.Icc (1/Real.exp 1) Real.exp 1, Real.exp 1, f a x has minimum value = 5 / 4 := sorry

theorem part_I_max (a : ℝ) (h : a = -1/2) :
  ∀ x ∈ Set.Icc (1 / Real.exp 1) Real.exp 1, f a x has maximum value = 1 / 2 + Real.exp 2 / 4 := sorry

theorem part_II_monotonicity (a x : ℝ) :
  (a ≤ -1 → ∀ x > 0, deriv (f a) x < 0) ∧
  (a ≥ 0 → ∀ x > 0, deriv (f a) x > 0) ∧
  (-1 < a < 0 → ∀ x < Real.sqrt (-a / (a+1)), deriv (f a) x < 0) ∧
  (-1 < a < 0 → ∀ x ≥ Real.sqrt (-a / (a+1)), deriv (f a) x > 0) := sorry

theorem part_III_range_of_a (a : ℝ) (h : -1 < a < 0) :
  ∀ x > 0, f a x > 1 + a / 2 * Real.log (-a) → (1 / Real.exp 1 - 1 < a ∧ a < 0) := sorry

end part_I_min_part_I_max_part_II_monotonicity_part_III_range_of_a_l153_153799


namespace probability_red_then_black_l153_153647

theorem probability_red_then_black :
  let number_of_cards := 52
  let number_of_red_cards := 26
  let number_of_black_cards := 26
  let probability_red_first := (number_of_red_cards : ℝ) / number_of_cards
  let probability_black_given_red := (number_of_black_cards : ℝ) / (number_of_cards - 1)
  probability_red_first * probability_black_given_red = (13 / 51 : ℝ) :=
by
  let number_of_cards := 52
  let number_of_red_cards := 26
  let number_of_black_cards := 26
  let probability_red_first := (number_of_red_cards : ℝ) / number_of_cards
  let probability_black_given_red := (number_of_black_cards : ℝ) / (number_of_cards - 1)
  calc
    probability_red_first * probability_black_given_red
        = (26 / 52 : ℝ) * (26 / 51 : ℝ) : by sorry
    ... = 13 / 51 : by sorry

end probability_red_then_black_l153_153647


namespace find_n_l153_153430

noncomputable def problem_statement (m n : ℤ) : Prop :=
  (∀ x : ℝ, x^2 - (m + 2) * x + (m - 2) = 0 → ∃ x1 x2 : ℝ, x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 * x2 < 0 ∧ x1 > |x2|) ∧
  (∃ r1 r2 : ℚ, r1 * r2 = 2 ∧ m * (r1 * r1 + r2 * r2) = (n - 2) * (r1 + r2) + m^2 - 3)

theorem find_n (m : ℤ) (hm : -2 < m ∧ m < 2) : 
  problem_statement m 5 ∨ problem_statement m (-1) :=
sorry

end find_n_l153_153430


namespace largest_diff_even_digits_l153_153669

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l153_153669


namespace profit_percentage_l153_153276

theorem profit_percentage (selling_price profit : ℝ) (h1 : selling_price = 900) (h2 : profit = 300) : 
  (profit / (selling_price - profit)) * 100 = 50 :=
by
  sorry

end profit_percentage_l153_153276


namespace shape_volume_and_surface_area_l153_153642

-- Defining the radii
def AB : ℝ := 20
def R₁ : ℝ := AB / 2
def R₂ : ℝ := R₁ / 2

-- Given the volumes of spheres based on their radii
def volume_sphere (r : ℝ) : ℝ := (4 / 3) * Real.pi * r ^ 3

-- Given the surface area of spheres based on their radii
def surface_area_sphere (r : ℝ) : ℝ := 4 * Real.pi * r ^ 2

-- Volumes of the larger and smaller parts
def V₁ := volume_sphere R₁
def V₂ := volume_sphere R₂

-- Surface areas of the larger and smaller parts
def S₁ := surface_area_sphere R₁
def S₂ := surface_area_sphere R₂

-- The resulting volumes and surface areas after subtraction and addition
def final_volume := V₁ - 2 * V₂
def final_surface_area := S₁ + 2 * S₂

theorem shape_volume_and_surface_area :
  final_surface_area = 600 * Real.pi ∧ final_volume = 1000 * Real.pi :=
by
  sorry

end shape_volume_and_surface_area_l153_153642


namespace max_difference_exists_l153_153658

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l153_153658


namespace sum_max_min_val_of_distances_l153_153012

noncomputable def distance_sq (P Q : ℝ × ℝ) : ℝ :=
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2

theorem sum_max_min_val_of_distances :
  let A := (-2, -2)
  let B := (-2, 6)
  let C := (4, -2)
  let circle := { P : ℝ × ℝ | P.1^2 + P.2^2 = 4 }
  let PA_sq := λ P, distance_sq P A
  let PB_sq := λ P, distance_sq P B
  let PC_sq := λ P, distance_sq P C
  let sum_sq := λ P, PA_sq P + PB_sq P + PC_sq P
  (let max_val := (λ f S, Real.Sup (f '' S)) in
   let min_val := (λ f S, Real.Inf (f '' S)) in
   max_val sum_sq circle + min_val sum_sq circle = 160) :=
by
  sorry

end sum_max_min_val_of_distances_l153_153012


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153081

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153081


namespace largest_among_abc_l153_153374

theorem largest_among_abc
  (x : ℝ) 
  (hx : 0 < x) 
  (hx1 : x < 1)
  (a : ℝ)
  (ha : a = 2 * Real.sqrt x )
  (b : ℝ)
  (hb : b = 1 + x)
  (c : ℝ)
  (hc : c = 1 / (1 - x)) 
  : a < b ∧ b < c :=
by
  sorry

end largest_among_abc_l153_153374


namespace hyperbola_eccentricity_proof_l153_153421

noncomputable def hyperbola_eccentricity (a : ℝ) (h_pos: a > 0) : ℝ :=
  let c := Real.sqrt (a^2 + 1) in
  c / a

theorem hyperbola_eccentricity_proof :
  let a := 1 / Real.tan (Real.pi / 6) in
  hyperbola_eccentricity a (by norm_num [Real.tan_pos_pi_div_two]) = 2 * Real.sqrt 3 / 3 :=
sorry

end hyperbola_eccentricity_proof_l153_153421


namespace sum_e_f_l153_153605

noncomputable def sum_of_possible_values (e f : ℚ) :=
  2 * abs (2 - e) = 5 ∧ abs (3 * e + f) = 7

theorem sum_e_f : ∃ (e f : ℚ), sum_of_possible_values e f ∧ 
  (e + (3 * e + f = 7 ⟹ f = 7 - 3 * e)) ∧ -- equation for e and f pairs
  (e = -0.5 ∨ e = 4.5) ∧ -- possible values for e
  (((-0.5 + 8.5) + (4.5 - 6.5)) = 6) := -- calculated sum
sorry

end sum_e_f_l153_153605


namespace queen_can_traverse_one_color_l153_153733

variable (Chessboard : Type)
variable [Fintype Chessboard]
variable (color : Chessboard → Bool) -- true for blue, false for red
variable (queen_move : Chessboard → Chessboard → Bool) -- denotes valid queen moves

def traversable (color : Chessboard → Bool) :=
  ∃ path : List Chessboard,
    ∀ i : ℕ, i < path.length - 1 → queen_move (path.nthLe i sorry) (path.nthLe (i + 1) sorry) ∧
    ∀ j, path.nthLe j sorry ∈ color ∨ path.nthLe j sorry ∉ color

theorem queen_can_traverse_one_color :
  ∃ col, traversable color col :=
sorry

end queen_can_traverse_one_color_l153_153733


namespace option_A_correct_option_D_correct_l153_153192

-- Define the gesture and rules of the game
inductive Gesture
| rock : Gesture
| paper : Gesture
| scissors : Gesture

open Gesture

def beats : Gesture → Gesture → Prop
| rock, scissors := True
| scissors, paper := True
| paper, rock := True
| _, _ := False

def draws : Gesture → Gesture → Prop
| rock, rock := True
| scissors, scissors := True
| paper, paper := True
| _, _ := False

def neither_wins (a b : Gesture) : Prop :=
  draws a b

-- Probability of neither player winning for one game
def prob_neither_wins_one_game : ℚ := 1 / 3

-- Probability calculation for exactly 2 wins from 3 games
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

def prob_win_two_out_of_three : ℚ :=
  (binom 3 2) * (1 / 3)^2 * (2 / 3) + (binom 3 3) * (1 / 3)^3

-- Proposed correct statements
theorem option_A_correct :
  (∀ (a b : Gesture), (neither_wins a b) ↔ ((a = b)) ) →
  prob_neither_wins_one_game = 1 / 3 := sorry

theorem option_D_correct : 
  prob_win_two_out_of_three = 7 / 27 := sorry

end option_A_correct_option_D_correct_l153_153192


namespace dollars_spent_on_books_l153_153097

theorem dollars_spent_on_books (total_allowance : ℕ) (games_ratio snacks_ratio toys_ratio : ℚ) : 
  total_allowance = 45 →
  games_ratio = 2/9 →
  snacks_ratio = 1/3 →
  toys_ratio = 1/5 →
  let dollars_spent_on_books := total_allowance - (games_ratio * total_allowance).to_nat - (snacks_ratio * total_allowance).to_nat - (toys_ratio * total_allowance).to_nat in
  dollars_spent_on_books = 11 :=
by sorry

end dollars_spent_on_books_l153_153097


namespace find_marks_in_english_l153_153344

theorem find_marks_in_english 
    (avg : ℕ) (math_marks : ℕ) (physics_marks : ℕ) (chemistry_marks : ℕ) (biology_marks : ℕ) (total_subjects : ℕ)
    (avg_eq : avg = 78) 
    (math_eq : math_marks = 65) 
    (physics_eq : physics_marks = 82) 
    (chemistry_eq : chemistry_marks = 67) 
    (biology_eq : biology_marks = 85) 
    (subjects_eq : total_subjects = 5) : 
    math_marks + physics_marks + chemistry_marks + biology_marks + E = 78 * 5 → 
    E = 91 :=
by sorry

end find_marks_in_english_l153_153344


namespace roman_total_worth_l153_153897

-- Define the initial worth of gold coins
def initial_worth : ℝ := 20

-- Define the number of gold coins sold
def coins_sold : ℕ := 3

-- Define the number of gold coins remaining
def coins_left : ℕ := 2

-- Define the total number of gold coins initially
def total_coins : ℕ := coins_sold + coins_left

-- Define the worth of each gold coin
def coin_value : ℝ := initial_worth / total_coins

-- Define the money received from selling the gold coins
def money_received : ℝ := coins_sold * coin_value

-- Define the worth of the remaining gold coins
def remaining_worth : ℝ := coins_left * coin_value

-- Define the total worth after selling
def total_worth : ℝ := money_received + remaining_worth

-- Prove the total_worth is equal to the initial_worth
theorem roman_total_worth : total_worth = initial_worth := sorry

end roman_total_worth_l153_153897


namespace angle_between_vectors_l153_153036

variables {a b : EuclideanSpace ℝ (Fin 2)}

noncomputable def norm (v : EuclideanSpace ℝ (Fin 2)) : ℝ := Real.sqrt (v.dotProduct v)

theorem angle_between_vectors : 
  (a + b).dotProduct a = 5 ∧ norm a = 2 ∧ norm b = 1 → ∃ θ : ℝ, θ = (real.pi / 3) :=
by
  sorry

end angle_between_vectors_l153_153036


namespace new_persons_joined_l153_153535

theorem new_persons_joined (initial_avg_age new_avg_age initial_total new_avg_age_total final_avg_age final_total : ℝ) 
  (n_initial n_new : ℕ) 
  (h1 : initial_avg_age = 16)
  (h2 : n_initial = 20)
  (h3 : new_avg_age = 15)
  (h4 : final_avg_age = 15.5)
  (h5 : initial_total = initial_avg_age * n_initial)
  (h6 : new_avg_age_total = new_avg_age * (n_new : ℝ))
  (h7 : final_total = initial_total + new_avg_age_total)
  (h8 : final_total = final_avg_age * (n_initial + n_new)) 
  : n_new = 20 :=
by
  sorry

end new_persons_joined_l153_153535


namespace contradiction_method_conditions_l153_153143

theorem contradiction_method_conditions :
  (using_judgments_contrary_to_conclusion ∧ using_conditions_of_original_proposition ∧ using_axioms_theorems_definitions) =
  (needed_conditions_method_of_contradiction) :=
sorry

end contradiction_method_conditions_l153_153143


namespace election_winner_percentage_l153_153128

def votes : List ℕ := [4571, 9892, 17653, 3217, 15135, 11338, 8629]

def votes_max : ℕ := List.maximum votes

def votes_total : ℕ := List.sum votes

def winning_percentage : ℝ := (votes_max.toReal / votes_total.toReal) * 100

theorem election_winner_percentage :
  abs (winning_percentage - 24.03) < 0.01 :=
by
  sorry

end election_winner_percentage_l153_153128


namespace triangle_area_OAB_l153_153469

theorem triangle_area_OAB (t ρ θ : ℝ) : 
  (∃ t : ℝ, (x = t^2 ∧ y = 2 * t)) →  
  (∃ ρ θ : ℝ, 2 * ρ * sin(π / 3 - θ) = √3) →
  let A : ℝ × ℝ := (t^2, 2 * t) in
  let B : ℝ × ℝ := (t^2, 2 * t) in
  ∀ O : ℝ × ℝ, O = (0, 0) →
  let area : ℝ := 1/2 * 1 * (2 * √3 + 2 * √3 / 3) in
  area = 4 * √3 / 3 :=
by
  sorry

end triangle_area_OAB_l153_153469


namespace f_at_2_l153_153804

noncomputable def f (x : ℝ) : ℝ := Real.log x + x^2 * (f' 1)
noncomputable def f' (x : ℝ) : ℝ := (1 / x) + 2 * x * (f' 1)

theorem f_at_2 : f 2 = Real.log 2 - 4 :=
by
  have h_f'1 : f' 1 = -1 :=
    by
      calc f' 1 = (1 / 1) + 2 * 1 * f'(1) : by sorry
           ... = 1 + 2 * f'(1)           : by sorry
           ... = 0                       : by sorry
           ... = -1                      : by sorry
  show f 2 = Real.log 2 - 4 from
    calc
      f 2 = Real.log 2 + 2^2 * (-1) : by sorry
       ... = Real.log 2 - 4         : by sorry

end f_at_2_l153_153804


namespace functions_with_inverses_l153_153819

def graphA (x : ℝ) : ℝ := if x < -2 then -x - 1 else if x < 0 then 2*x + 3 else 3 - x
def graphB (x : ℝ) : Option ℝ := 
  if x < -3 then some (x + 1)
  else if x < 2 then none 
  else some (x - 2)
def graphC (x : ℝ) : ℝ := -x
def graphD (x : ℝ) : Option ℝ := 
  let theta := x + 20
  if theta ≤ 210 then some (4 * Real.cos theta) 
  else none
def graphE (x : ℝ) : ℝ :=
  x^3/30 + x^2/10 - x/3 + 1

theorem functions_with_inverses :
  (∃ f_inv : ℝ → Option ℝ, ∀ x, graphB x = some (f_inv x)) ∧
  (∃ f_inv : ℝ → ℝ, ∀ x, graphC (f_inv x) = x) ∧
  (∃ f_inv : ℝ → Option ℝ, ∀ x, graphD x = some (f_inv x)) :=
by
  sorry -- Here we need to provide the actual proof which shows the existence of these inverses.

end functions_with_inverses_l153_153819


namespace number_of_seeds_per_row_l153_153339

-- Define the conditions as variables
def rows : ℕ := 6
def total_potatoes : ℕ := 54
def seeds_per_row : ℕ := 9

-- State the theorem
theorem number_of_seeds_per_row :
  total_potatoes / rows = seeds_per_row :=
by
-- We ignore the proof here, it will be provided later
sorry

end number_of_seeds_per_row_l153_153339


namespace truck_travel_distance_l153_153655

open_locale real

theorem truck_travel_distance (b t : ℝ) (ht : t ≠ 0) :
  let rate := b / 4,
      time := 5 * 60, -- 5 minutes in seconds
      distance_in_feet := (rate / t) * time,
      distance_in_yards := distance_in_feet / 3
  in distance_in_yards = 25 * b / t := 
sorry

end truck_travel_distance_l153_153655


namespace exists_n_f_over_g_eq_2012_l153_153754

def is_perfect_square (d : ℕ) : Prop :=
  ∃ k : ℕ, k * k = d

def is_perfect_cube (d : ℕ) : Prop :=
  ∃ k : ℕ, k * k * k = d

def f (n : ℕ) : ℕ :=
  (finset.range n).filter (λ d, d ∣ n ∧ is_perfect_square d).card

def g (n : ℕ) : ℕ :=
  (finset.range n).filter (λ d, d ∣ n ∧ is_perfect_cube d).card

theorem exists_n_f_over_g_eq_2012 :
  ∃ n : ℕ, f n / g n = 2012 :=
sorry

end exists_n_f_over_g_eq_2012_l153_153754


namespace max_area_triangle_l153_153145

open Real

theorem max_area_triangle (a b : ℝ) (C : ℝ) (h₁ : a + b = 4) (h₂ : C = π / 6) : 
  (1 : ℝ) ≥ (1 / 2 * a * b * sin (π / 6)) := 
by 
  sorry

end max_area_triangle_l153_153145


namespace one_cow_one_bag_in_34_days_l153_153838

-- Definitions: 34 cows eat 34 bags in 34 days, each cow eats one bag in those 34 days.
def cows : Nat := 34
def bags : Nat := 34
def days : Nat := 34

-- Hypothesis: each cow eats one bag in 34 days.
def one_bag_days (c : Nat) (b : Nat) : Nat := days

-- Theorem: One cow will eat one bag of husk in 34 days.
theorem one_cow_one_bag_in_34_days : one_bag_days 1 1 = 34 := sorry

end one_cow_one_bag_in_34_days_l153_153838


namespace rhombus_other_diagonal_l153_153919

theorem rhombus_other_diagonal (d1 : ℝ) (area : ℝ) (d2 : ℝ) 
  (h1 : d1 = 50) 
  (h2 : area = 625) 
  (h3 : area = (d1 * d2) / 2) : 
  d2 = 25 :=
by
  sorry

end rhombus_other_diagonal_l153_153919


namespace at_least_one_free_node_l153_153646

-- Definitions
structure Grid :=
  (cells : ℕ)
  (dimension : cells = 100)
  (nodes : set (ℕ × ℕ) := { (i, j) | i ∈ {0, ..., 100} ∧ j ∈ {0, ..., 100} })
  (broken_lines : set (finset (ℕ × ℕ)))

-- Assumptions
def assumptions (g : Grid) : Prop :=
  ∀ l ∈ g.broken_lines, 
    (l.card ≥ 2) ∧ 
    (∀ (x y : ℕ × ℕ), (x ∈ l ∧ y ∈ l) → (x ≠ y)) ∧
    (∀ (x : ℕ × ℕ), x ∈ l → x.1 = 0 ∨ x.1 = 100 ∨ x.2 = 0 ∨ x.2 = 100)

-- Theorem
theorem at_least_one_free_node (g : Grid) (h : assumptions g) : 
  ∃ n ∈ g.nodes, (n ≠ (0, 0)) ∧ (n ≠ (0, 100)) ∧ (n ≠ (100, 0)) ∧ (n ≠ (100, 100)) ∧ 
  (∀ l ∈ g.broken_lines, n ∉ l) :=
sorry

end at_least_one_free_node_l153_153646


namespace students_in_donnelly_class_l153_153895

-- Conditions
def initial_cupcakes : ℕ := 40
def cupcakes_to_delmont_class : ℕ := 18
def cupcakes_to_staff : ℕ := 4
def leftover_cupcakes : ℕ := 2

-- Question: How many students are in Mrs. Donnelly's class?
theorem students_in_donnelly_class : 
  let cupcakes_given_to_students := initial_cupcakes - (cupcakes_to_delmont_class + cupcakes_to_staff)
  let cupcakes_given_to_donnelly_class := cupcakes_given_to_students - leftover_cupcakes
  cupcakes_given_to_donnelly_class = 16 :=
by
  sorry

end students_in_donnelly_class_l153_153895


namespace minimum_area_l153_153916

variables (T A B C D : Type) (a : ℝ)
variables [real_plane T A B C D]
variables (alpha beta : ℝ)

-- Conditions
def base_is_rectangle : Prop := is_rectangle A B C D
def height_is_TA : Prop := height_of_pyramid_is_TA T A
def TC_inclined_angle : Prop := inclined_angle_to_plane TC (30 : ℝ)
def plane_TC_parallel_BD : Prop := plane_passing_through_TC_parallel_to_BD TC (BD : ℝ)
def distance_plane_BD : Prop := distance_plane_is TC (BD : ℝ) a

-- Question/Answer
def minimum_area_cross_section_AC : ℝ := (4 * sqrt 3 * a^2) / (sqrt 5)

theorem minimum_area (h1 : base_is_rectangle A B C D)
  (h2 : height_is_TA T A)
  (h3 : TC_inclined_angle TC (30 : ℝ))
  (h4 : plane_TC_parallel_BD TC (BD : ℝ))
  (h5 : distance_plane_BD TC (BD : ℝ) a) :
  minimum_area_cross_section_AC a = (4 * sqrt 3 * a^2) / (sqrt 5) :=
sorry

end minimum_area_l153_153916


namespace pins_after_one_month_l153_153515

theorem pins_after_one_month
  (daily_pins_per_member : ℕ)
  (deletion_rate_per_week_per_person : ℕ)
  (num_members : ℕ)
  (initial_pins : ℕ)
  (days_in_month : ℕ)
  (weeks_in_month : ℕ) :
  (daily_pins_per_member = 10) →
  (deletion_rate_per_week_per_person = 5) →
  (num_members = 20) →
  (initial_pins = 1000) →
  (days_in_month = 30) →
  (weeks_in_month = 4) →
  let daily_addition := daily_pins_per_member * num_members,
      monthly_addition := daily_addition * days_in_month,
      total_initial_and_new := initial_pins + monthly_addition,
      weekly_deletion := deletion_rate_per_week_per_person * num_members,
      monthly_deletion := weekly_deletion * weeks_in_month,
      final_pins := total_initial_and_new - monthly_deletion
  in final_pins = 6600 :=
by
  intros h1 h2 h3 h4 h5 h6
  have daily_addition := daily_pins_per_member * num_members
  have monthly_addition := daily_addition * days_in_month
  have total_initial_and_new := initial_pins + monthly_addition
  have weekly_deletion := deletion_rate_per_week_per_person * num_members
  have monthly_deletion := weekly_deletion * weeks_in_month
  have final_pins := total_initial_and_new - monthly_deletion
  sorry

end pins_after_one_month_l153_153515


namespace range_of_f_on_interval_inequality_f_log_on_interval_l153_153792

-- Define the function f(x)
def f (x : ℝ) : ℝ := (Real.log x / Real.log 2 - 2) * (Real.log x / (Real.log 4) - 1/2)

-- Part 1: The range of f(x) when x ∈ [2, 4] is [-1/8, 0]
theorem range_of_f_on_interval :
  ∀ x, 2 ≤ x ∧ x ≤ 4 → f 2 ≤ f x ∧ f x ≤ f 4 :=
begin
  sorry
end

-- Part 2: f(x) ≥ m * Real.log x / Real.log 2 for x ∈ [4, 16] implies m ≤ 0
theorem inequality_f_log_on_interval :
  ∀ m, (∀ x, 4 ≤ x ∧ x ≤ 16 → f x ≥ m * (Real.log x / Real.log 2)) →
       m ≤ 0 :=
begin
  sorry
end

end range_of_f_on_interval_inequality_f_log_on_interval_l153_153792


namespace coin_flip_heads_probability_l153_153959

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153959


namespace midpoints_of_segments_l153_153468

theorem midpoints_of_segments
  (S_ABD S_BCD S_ABC AM_over_AC CN_over_CD : ℝ)
  (areas_ratio : S_ABD / S_BCD = 3 / 4 ∧ S_BCD / S_ABC = 4 / 1)
  (collinear_B_M_N : collinear ℝ {B, M, N})
  (AM_AC : AM_over_AC = r)
  (CN_CD : CN_over_CD = r)
  (r_value : r = 1 / 2) :
  AM_over_AC = 1 / 2 ∧ CN_over_CD = 1 / 2 :=
by
  sorry

end midpoints_of_segments_l153_153468


namespace lucille_cents_left_l153_153495

-- Definitions based on conditions
def weeds_in_flower_bed : ℕ := 11
def weeds_in_vegetable_patch : ℕ := 14
def weeds_in_grass : ℕ := 32
def cents_per_weed : ℕ := 6
def soda_cost : ℕ := 99

-- Main statement to prove
theorem lucille_cents_left : 
  let total_weeds_pulled := weeds_in_flower_bed + weeds_in_vegetable_patch + weeds_in_grass / 2
  let total_earnings := total_weeds_pulled * cents_per_weed
  let cents_left := total_earnings - soda_cost
  in cents_left = 147 := 
by {
  -- Definitions for computing intermediate results
  def total_weeds_pulled := weeds_in_flower_bed + weeds_in_vegetable_patch + weeds_in_grass / 2
  def total_earnings := total_weeds_pulled * cents_per_weed
  def cents_left := total_earnings - soda_cost
  -- Specify the exact values
  have h1 : total_weeds_pulled = 41 := by simp [weeds_in_flower_bed, weeds_in_vegetable_patch, weeds_in_grass]
  have h2 : total_earnings = 246 := by rw [h1]; simp [cents_per_weed]
  have h3 : cents_left = 147 := by rw [h2]; simp [soda_cost]
  exact h3
}

end lucille_cents_left_l153_153495


namespace explicit_formula_and_monotonicity_l153_153415

def f (x : ℝ) (a b : ℝ) := x^3 + a * x^2 - 9 * x + b

theorem explicit_formula_and_monotonicity (a b : ℝ) 
  (h₁ : f 0 a b = 2) 
  (h₂ : f' a 1 = 0) :
  f x 3 2 = x^3 + 3 * x^2 - 9 * x + 2 ∧ 
  (∀ x, x < -3 → monotone_incr (f x 3 2)) ∧
  (∀ x, 1 < x → monotone_incr (f x 3 2)) ∧
  (∀ x, -3 < x ∧ x < 1 → monotone_decr (f x 3 2)) := 
sorry

end explicit_formula_and_monotonicity_l153_153415


namespace minimum_cuts_for_11_sided_polygons_l153_153886

theorem minimum_cuts_for_11_sided_polygons (k : ℕ) :
  (∀ k, (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4)) ∧ (252 ≤ (k + 1)) ∧ (4 * k + 4 ≥ 11 * 252 + 3 * (k + 1 - 252))
  ∧ (11 * 252 + 3 * (k + 1 - 252) ≤ 4 * k + 4) → (k ≥ 2012) ∧ (k = 2015) := 
sorry

end minimum_cuts_for_11_sided_polygons_l153_153886


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153990

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153990


namespace distinct_roots_implies_m_greater_than_half_find_m_given_condition_l153_153929

-- Define the quadratic equation with a free parameter m
def quadratic_eq (x : ℝ) (m : ℝ) : Prop :=
  x^2 - 4 * x - 2 * m + 5 = 0

-- Prove that if the quadratic equation has distinct roots, then m > 1/2
theorem distinct_roots_implies_m_greater_than_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂) →
  m > 1 / 2 :=
by
  sorry

-- Given that x₁ and x₂ satisfy both the quadratic equation and the sum-product condition, find the value of m
theorem find_m_given_condition (m : ℝ) (x₁ x₂ : ℝ) :
  quadratic_eq x₁ m ∧ quadratic_eq x₂ m ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ + x₁ + x₂ = m^2 + 6) → 
  m = 1 :=
by
  sorry

end distinct_roots_implies_m_greater_than_half_find_m_given_condition_l153_153929


namespace not_enrolled_eq_80_l153_153127

variable (total_students : ℕ)
variable (french_students : ℕ)
variable (german_students : ℕ)
variable (spanish_students : ℕ)
variable (french_and_german : ℕ)
variable (german_and_spanish : ℕ)
variable (spanish_and_french : ℕ)
variable (all_three : ℕ)

noncomputable def students_not_enrolled_in_any_language 
  (total_students french_students german_students spanish_students french_and_german german_and_spanish spanish_and_french all_three : ℕ) : ℕ :=
  total_students - (french_students + german_students + spanish_students - french_and_german - german_and_spanish - spanish_and_french + all_three)

theorem not_enrolled_eq_80 : 
  students_not_enrolled_in_any_language 180 60 50 35 20 15 10 5 = 80 :=
  by
    unfold students_not_enrolled_in_any_language
    simp
    sorry

end not_enrolled_eq_80_l153_153127


namespace find_BC_distance_l153_153852

-- Definitions of constants as per problem conditions
def ACB_angle : ℝ := 120
def AC_distance : ℝ := 2
def AB_distance : ℝ := 3

-- The theorem to prove the distance BC
theorem find_BC_distance (BC : ℝ) (h : AC_distance * AC_distance + (BC * BC) - 2 * AC_distance * BC * Real.cos (ACB_angle * Real.pi / 180) = AB_distance * AB_distance) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end find_BC_distance_l153_153852


namespace find_nsatisfy_l153_153756

-- Define the function S(n) that denotes the sum of the digits of n
def S (n : ℕ) : ℕ := n.digits 10 |>.sum

-- State the main theorem
theorem find_nsatisfy {n : ℕ} : n = 2 * (S n)^2 → n = 50 ∨ n = 162 ∨ n = 392 ∨ n = 648 := 
sorry

end find_nsatisfy_l153_153756


namespace prism_minimized_height_l153_153848

-- Define the radius of the hemisphere
def radius_hemisphere := 1

-- Define the height of the prism in such a way that we need to prove it is minimized
def height_prism_minimized := 2 * Real.sqrt 2

-- Prove the statement that height of prism when volume is minimized is 2 * sqrt(2) given the conditions.
theorem prism_minimized_height : 
  ∀ (radius_hemisphere: ℝ), 
  (∀ (base_coincides: Prop), ∀ (lateral_faces_tangent: Prop), radius_hemisphere = 1 → base_coincides → lateral_faces_tangent → 
  ∃ h: ℝ, h = 2 * Real.sqrt 2) :=
by
  intro radius_hemisphere
  intros base_coincides lateral_faces_tangent radius_eq base_cond lateral_cond
  use 2 * Real.sqrt 2
  sorry

end prism_minimized_height_l153_153848


namespace minimum_value_quadratic_expression_l153_153355

noncomputable def quadratic_expression (x y : ℝ) : ℝ :=
  3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 8 * y + 9

theorem minimum_value_quadratic_expression :
  ∃ (x y : ℝ), quadratic_expression x y = -15 ∧
    ∀ (a b : ℝ), quadratic_expression a b ≥ -15 :=
by sorry

end minimum_value_quadratic_expression_l153_153355


namespace line_through_lattice_points_l153_153635

theorem line_through_lattice_points : ∃ p : ℤ × ℤ, 
  let y_of_x := λ x, (3 / 5 : ℚ) * x in
  p ≠ (0,0) ∧ p ≠ (5,3) ∧ p.2 = y_of_x p.1 ∧ p.1 = 10 ∧ p.2 = 6 :=
by
  sorry

end line_through_lattice_points_l153_153635


namespace sodium_chloride_formed_l153_153432

-- Define the conditions
def chlorine_moles := 1
def sodium_hydroxide_moles := 2
def sodium_hypochlorite_moles := 1
def water_moles := 1

-- Prove the number of moles of sodium chloride formed
theorem sodium_chloride_formed :
  (chlorine_moles = 1) →
  (sodium_hydroxide_moles = 2) →
  (sodium_hypochlorite_moles = 1) →
  (water_moles = 1) →
  ∃ (NaCl_moles : ℕ), NaCl_moles = 1 :=
by
  intros
  use 1
  sorry

end sodium_chloride_formed_l153_153432


namespace min_positive_n_constant_term_l153_153108

theorem min_positive_n_constant_term : ∃ (n : ℕ), (∀ r, 2 * n = 5 * r → r ∈ ℕ) ∧ (2 * n ≠ 0) ∧ n = 5 :=
by sorry

end min_positive_n_constant_term_l153_153108


namespace isabella_hourly_rate_l153_153474

def isabella_hours_per_day : ℕ := 5
def isabella_days_per_week : ℕ := 6
def isabella_weeks : ℕ := 7
def isabella_total_earnings : ℕ := 1050

theorem isabella_hourly_rate :
  (isabella_hours_per_day * isabella_days_per_week * isabella_weeks) * x = isabella_total_earnings → x = 5 := by
  sorry

end isabella_hourly_rate_l153_153474


namespace probability_at_least_9_heads_l153_153977

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153977


namespace min_dot_product_of_vectors_at_fixed_point_l153_153139

noncomputable def point := ℝ × ℝ

def on_ellipse (x y : ℝ) : Prop := 
  (x^2) / 36 + (y^2) / 9 = 1

def dot_product (p q : point) : ℝ := 
  p.1 * q.1 + p.2 * q.2

def vector_magnitude_squared (p : point) : ℝ := 
  p.1^2 + p.2^2

def KM (M : point) : point := 
  (M.1 - 2, M.2)

def NM (N M : point) : point := 
  (M.1 - N.1, M.2 - N.2)

def fixed_point_K : point := 
  (2, 0)

theorem min_dot_product_of_vectors_at_fixed_point (M N : point) 
  (hM_on_ellipse : on_ellipse M.1 M.2) 
  (hN_on_ellipse : on_ellipse N.1 N.2) 
  (h_orthogonal : dot_product (KM M) (KM N) = 0) : 
  ∃ (α : ℝ), dot_product (KM M) (NM N M) = 23 / 3 :=
sorry

end min_dot_product_of_vectors_at_fixed_point_l153_153139


namespace ratio_side_lengths_sum_l153_153216

theorem ratio_side_lengths_sum :
  let area_ratio := (250 : ℝ) / 98
  let side_length_ratio := Real.sqrt area_ratio
  ∃ a b c : ℕ, side_length_ratio = a * Real.sqrt b / c ∧ a + b + c = 17 :=
by
  let area_ratio := (250 : ℝ) / 98
  let side_length_ratio := Real.sqrt area_ratio
  use 5, 5, 7
  split
  {
    sorry -- Proof that side_length_ratio = 5 * Real.sqrt 5 / 7
  }
  {
    refl -- Proof that 5 + 5 + 7 = 17
  }

end ratio_side_lengths_sum_l153_153216


namespace number_of_squares_with_two_same_color_vertices_is_even_l153_153907

noncomputable def is_even (n : ℕ) : Prop :=
  ∃ (k : ℕ), n = 2 * k

def color := ℕ → bool -- bool representing red and blue

theorem number_of_squares_with_two_same_color_vertices_is_even:
  ∀ (n : ℕ) (coloring : color), 
    (n > 0) ∧
    (coloring 0 = tt) ∧ (coloring 1 = ff) ∧
    (coloring (n^2 - 1) = tt) ∧ (coloring (n^2 - n) = ff) →
    ∃ (k : ℕ), ∑ i in range (n^2), (count_squares_with_same_color_vertices i coloring) = 2 * k :=
begin
  sorry
end

noncomputable def count_squares_with_same_color_vertices (i : ℕ) (coloring : color) : ℕ :=
  if vertices (i) coloring = 2 * tt || vertices (i) coloring = 2 * ff then 1 else 0

def vertices (i : ℕ) (coloring : color) : ℕ := 
  -- function to count vertices with same color for the i-th square
  sorry

end number_of_squares_with_two_same_color_vertices_is_even_l153_153907


namespace rooks_non_attacking_l153_153482

theorem rooks_non_attacking (n : ℕ) (h_n : n ≥ 2) :
  (∃ (f : fin n → fin n), function.injective f ∧
   ∀ pos in fin n, rook_movable f pos) → even n :=
sorry

-- Define the rook placement and movement
def rook_movable {n : ℕ} (f : fin n → fin n) (pos : fin n) : Prop :=
  let rook_pos := f pos in
  ∃ (adj : fin n), adj ≠ rook_pos ∧ adjacent rook_pos adj ∧
  ∀ other_pos in fin n, other_pos ≠ rook_pos → f other_pos ≠ adj

-- Adjacent squares on an n x n board
def adjacent {n : ℕ} (a b : fin n) : Prop :=
  (a.1 = b.1 ∧ (a.2 = b.2 + 1 ∨ a.2 = b.2 - 1)) ∨
  (a.2 = b.2 ∧ (a.1 = b.1 + 1 ∨ a.1 = b.1 - 1))

end rooks_non_attacking_l153_153482


namespace find_ab_find_c_range_l153_153418

noncomputable def f (x : ℝ) (a b c : ℝ) : ℝ := x^3 + a * x^2 + b * x + c
noncomputable def f_prime (x : ℝ) (a b : ℝ) : ℝ := 3 * x^2 + 2 * a * x + b

theorem find_ab (a b c : ℝ) :
  (f_prime 1 a b = 0) ∧ (f_prime (-2/3) a b = 0) → 
  (a = -1/2) ∧ (b = -2) :=
by sorry

theorem find_c_range (a b c : ℝ) : 
  (a = -1/2) ∧ (b = -2) →
  (∀ x : ℝ, x ∈ set.Icc (-1) 2 → f x a b c = 2 * c → 
    (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ f x1 a b c = 2 * c ∧ 
      f x2 a b c = 2 * c ∧ f x3 a b c = 2 * c)) →
  (1/2 ≤ c ∧ c < 22/27) :=
by sorry

end find_ab_find_c_range_l153_153418


namespace pascals_triangle_contains_47_once_l153_153068

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153068


namespace sum_of_solutions_sin_cos_eq_one_l153_153747

theorem sum_of_solutions_sin_cos_eq_one :
  (∑ x in { x | 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.sin x + Real.cos x = 1 }, x) = 5 * Real.pi / 2 :=
by
  sorry

end sum_of_solutions_sin_cos_eq_one_l153_153747


namespace anna_original_number_l153_153703

theorem anna_original_number :
  ∃ x : ℚ, (let result := 5 * ((3 * x) + 15) in result = 200) → x = 25 / 3 :=
by
  sorry

end anna_original_number_l153_153703


namespace probability_heads_in_12_flips_l153_153993

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153993


namespace victor_final_amount_is_874_l153_153565

noncomputable def final_amount (initial_rubles : ℝ) (usd_rate : ℝ) (annual_rate : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  let principal := initial_rubles / usd_rate
  let quarterly_rate := 1 + annual_rate / n.to_real
  principal * quarterly_rate ^ (n * t).to_real

theorem victor_final_amount_is_874 :
  final_amount 45000 56.60 0.047 4 2 ≈ 874 :=
by
  sorry

end victor_final_amount_is_874_l153_153565


namespace problem1_problem2_l153_153138

def arithmetic_sequence (a : ℕ → ℤ) (a1 a3 : ℤ) : Prop :=
  a 1 = a1 ∧ a 3 = a3 ∧ ∃ d, ∀ n, a (n + 1) = a n + d 

theorem problem1 
  (a : ℕ → ℤ) 
  (S : ℕ → ℤ) 
  (a1 a3 : ℤ)
  (h_arith : arithmetic_sequence a 8 4)
  (h_S : ∀ n, S n = n * (a 1 + a n) / 2) 
  (max_S : ∀ n, S n ≤ 20) :
  (max_S 4 ∧ max_S 5) ∧ (S 4 = 20 ∧ S 5 = 20) :=
sorry

def harmonic_sequence (a : ℕ → ℤ) (b : ℕ → ℚ) : Prop :=
  ∀ n, b n = 1 / n * (12 - a n)

theorem problem2 
  (a : ℕ → ℤ)
  (b T : ℕ → ℚ)
  (h_arith : arithmetic_sequence a 8 4)
  (h_b : harmonic_sequence a b)
  (h_T : ∀ n, T n = ∑ i in finset.range n, b (i + 1)) :
  ∀ n, T n = n / (2 * (n + 1)) :=
sorry

end problem1_problem2_l153_153138


namespace complex_value_of_z_l153_153834

theorem complex_value_of_z (z : ℂ) (h : (3 - z) * complex.i = 2 * complex.i) : z = 3 + 2 * complex.i :=
sorry

end complex_value_of_z_l153_153834


namespace max_k_value_l153_153376

noncomputable theory

-- Statement of the problem in Lean 4
theorem max_k_value (x y k : ℝ) (hx : 0 < x) (hy : 0 < y) (hk : 0 < k) 
  (h : 5 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x)) : 
  k = (-1 + Real.sqrt 13) / 2 :=
sorry

end max_k_value_l153_153376


namespace hyperbola_focal_length_l153_153134

/--
In the Cartesian coordinate system \( xOy \),
let the focal length of the hyperbola \( \frac{x^{2}}{2m^{2}} - \frac{y^{2}}{3m} = 1 \) be 6.
Prove that the set of all real numbers \( m \) that satisfy this condition is {3/2}.
-/
theorem hyperbola_focal_length (m : ℝ) (h1 : 2 * m^2 > 0) (h2 : 3 * m > 0) (h3 : 2 * m^2 + 3 * m = 9) :
  m = 3 / 2 :=
sorry

end hyperbola_focal_length_l153_153134


namespace unique_integer_n_l153_153721

theorem unique_integer_n :
  ∃! (n : ℕ), 0 < n ∧ (∑ i in Finset.range n, (i + 2) * 2^(i + 2)) = 2^(n + 11) ∧ n = 1025 :=
begin
  sorry
end

end unique_integer_n_l153_153721


namespace cylinder_radius_in_cone_l153_153305

theorem cylinder_radius_in_cone :
  ∀ (r : ℚ) (d h : ℚ),
  d = 8 →
  h = 10 →
  3 * r = (h - 3 * r) / (d / 2) * r →
  r = 20 / 11 :=
by
  intros r d h
  assume h_diameter : d = 8
  assume h_altitude : h = 10
  assume h_condition : 3 * r = (h - 3 * r) / (d / 2) * r
  sorry

end cylinder_radius_in_cone_l153_153305


namespace expectation_variance_comparison_l153_153780

variable {p1 p2 : ℝ}
variable {ξ1 ξ2 : ℝ}

theorem expectation_variance_comparison
  (h_p1 : 0 < p1)
  (h_p2 : p1 < p2)
  (h_p3 : p2 < 1 / 2)
  (h_ξ1 : ξ1 = p1)
  (h_ξ2 : ξ2 = p2):
  (ξ1 < ξ2) ∧ (ξ1 * (1 - ξ1) < ξ2 * (1 - ξ2)) := by
  sorry

end expectation_variance_comparison_l153_153780


namespace megatek_manufacturing_percentage_l153_153199

-- Define the given conditions
def sector_deg : ℝ := 18
def full_circle_deg : ℝ := 360

-- Define the problem as a theorem statement in Lean
theorem megatek_manufacturing_percentage : 
  (sector_deg / full_circle_deg) * 100 = 5 := 
sorry

end megatek_manufacturing_percentage_l153_153199


namespace solve_sqrt_equation_l153_153905

theorem solve_sqrt_equation (x : ℝ) : 
  (sqrt (4 * x^2 + 4 * x + 1) - sqrt (4 * x^2 - 12 * x + 9) = 4) → 
  (x ≥ 3 / 2) :=
by 
  sorry

end solve_sqrt_equation_l153_153905


namespace find_k_l153_153307

noncomputable def average_distance (h : ℝ) (g : ℝ) : ℝ :=
  let T := sqrt (2 * h / g) in
  (1 / T) * ∫ t in 0..T, (1/2 * g * t^2)

theorem find_k (h : ℝ) (g : ℝ) (h_pos : 0 < h) (g_pos : 0 < g) : average_distance h g = (1 / 3) * h :=
by
  have T := sqrt (2 * h / g)
  have d := (1 / 2) * g * (λ t, t^2)
  have I := (∫ t in 0..T, d t)
  have avg_dist := (1 / T) * I
  have result := (1 / 3) * h
  sorry

end find_k_l153_153307


namespace centroid_traces_circle_l153_153428

-- Definitions of the geometric elements
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A B C : Point

-- Midpoint definition
def midpoint (P Q : Point) : Point :=
  { x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2 }

-- Centroid definition
def centroid (T : Triangle) : Point :=
  { x := (T.A.x + T.B.x + T.C.x) / 3, y := (T.A.y + T.B.y + T.C.y) / 3 }

-- Given conditions
variable (A B : Point)
def is_fixed_base (T : Triangle) : Prop :=
  T.A = A ∧ T.B = B

variable (M : Point)
def is_midpointAB (A B M : Point) : Prop :=
  M = midpoint A B

-- Statement to prove
theorem centroid_traces_circle
  (T : Triangle) (r : ℝ)
  (h_base : is_fixed_base T)
  (h_midpoint : is_midpointAB T.A T.B M)
  (h_C_circle : ∀ C : Point, dist M C = r → T.C = C):
  ∃ (G : Point) (r' : ℝ), G = centroid T ∧ dist M G = (2/3) * r := 
sorry

end centroid_traces_circle_l153_153428


namespace max_diff_six_digit_even_numbers_l153_153672

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l153_153672


namespace square_of_binomial_l153_153589

theorem square_of_binomial (k : ℝ) : (∃ b : ℝ, x^2 - 20 * x + k = (x + b)^2) -> k = 100 :=
sorry

end square_of_binomial_l153_153589


namespace pascal_triangle_47_rows_l153_153082

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153082


namespace cost_of_lamps_and_bulbs_l153_153862

theorem cost_of_lamps_and_bulbs : 
    let lamp_cost := 7
    let bulb_cost := lamp_cost - 4
    let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
    total_cost = 32 := by
  let lamp_cost := 7
  let bulb_cost := lamp_cost - 4
  let total_cost := (lamp_cost * 2) + (bulb_cost * 6)
  sorry

end cost_of_lamps_and_bulbs_l153_153862


namespace exists_sum_pair_l153_153383

theorem exists_sum_pair (n : ℕ) (a b : List ℕ) (h₁ : ∀ x ∈ a, x < n) (h₂ : ∀ y ∈ b, y < n) 
  (h₃ : List.Nodup a) (h₄ : List.Nodup b) (h₅ : a.length + b.length ≥ n) : ∃ x ∈ a, ∃ y ∈ b, x + y = n := by
  sorry

end exists_sum_pair_l153_153383


namespace pascal_triangle_47_rows_l153_153083

theorem pascal_triangle_47_rows :
  ∃! n, ∀ k, k ≠ 47 → binom k 47 ≠ 47 :=
sorry

end pascal_triangle_47_rows_l153_153083


namespace Pat_height_l153_153887

noncomputable def Pat_first_day_depth := 40 -- in cm
noncomputable def Mat_second_day_depth := 3 * Pat_first_day_depth -- Mat digs 3 times the depth on the second day
noncomputable def Pat_third_day_depth := Mat_second_day_depth - Pat_first_day_depth -- Pat digs the same amount on the third day
noncomputable def Total_depth_after_third_day := Mat_second_day_depth + Pat_third_day_depth -- Total depth after third day's digging
noncomputable def Depth_above_Pat_head := 50 -- The depth above Pat's head

theorem Pat_height : Total_depth_after_third_day - Depth_above_Pat_head = 150 := by
  sorry

end Pat_height_l153_153887


namespace part1_part2_l153_153800

variables (a b k : ℝ)
def f (x : ℝ) : ℝ := a * x^2 + b * x + 1
def g (x : ℝ) : ℝ := f a b x - k * x

-- Conditions for part 1
def cond1 := a - b + 1 = 0
def cond2 := -b / (2 * a) = -1

-- Conditions for part 2
def cond3 := ∀ x, x ∈ set.Icc (-2 : ℝ) 2 → g a b k x ∈ set.Icc (0 : ℝ) (2 : ℝ)

-- Part 1: Determine the Analytical Expression of f(x)
theorem part1 (h1 : cond1) (h2 : cond2) : f a b = λ x, x^2 + 2 * x + 1 := 
sorry

-- Part 2: Determine the Range of the Real Number k
theorem part2 (h1 : cond1) (h2 : cond2) (h3 : cond3) : k ≥ 6 ∨ k ≤ -2 :=
sorry

end part1_part2_l153_153800


namespace max_diff_six_digit_even_numbers_l153_153673

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l153_153673


namespace find_b_l153_153365

-- Define the set and its conditions
variables (x b : ℕ)
variables (S : Set ℕ) (median mean : ℚ)

-- The set of positive integers
def S := {x, x + 2, x + b, x + 7, x + 37}

-- The median of the set
def median := x + b

-- The mean of the set
def mean := (x + (x + 2) + (x + b) + (x + 7) + (x + 37)) / 5

-- Condition that the mean is 6 greater than the median
def condition : Prop := mean = median + 6

-- Proving b = 4 under the given conditions
theorem find_b (h : condition) : b = 4 :=
by
  sorry

end find_b_l153_153365


namespace sum_of_common_terms_l153_153816

theorem sum_of_common_terms :
  let seq1 := (λ n : ℕ, 2 + 4*n) in
  let seq2 := (λ n : ℕ, 2 + 6*n) in
  let common_term k := seq1 k = seq2 k in
  let common_terms := (λ n, 2 + 12*n) in
  let n_common := 17 in -- There are 18 terms including the term for n=0
  (∑ i in finset.range (n_common + 1), common_terms i) = 1872 :=
by
  -- Begin proof
  sorry

end sum_of_common_terms_l153_153816


namespace sum_reciprocal_crossing_bound_l153_153294

open Finset

variables {V : Type} [Fintype V] (E : Finset (Sym2 V))

-- Let G be a finite graph on n vertices
variable (n : ℕ) (hG : E.card = n)

-- ∀ e ∈ E, let χ(e) be the number of edges that cross over edge e
variable (χ : E → ℕ)

/-- Prove that the sum of 1/(χ(e) + 1) over all edges e is at most 3n - 6 -/
theorem sum_reciprocal_crossing_bound : 
  (∑ e in E, 1 / (χ e + 1 : ℝ)) ≤ (3 * n - 6 : ℝ) :=
sorry

end sum_reciprocal_crossing_bound_l153_153294


namespace pascal_row_contains_prime_47_l153_153048

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153048


namespace solution_l153_153520

noncomputable def proof_problem (x : ℝ) : Prop :=
  arcsin x + arcsin (1 - x) = arccos x

theorem solution (x : ℝ) (hx : x ∈ set.Icc (-1:ℝ) 1): proof_problem x ↔ (x = 0 ∨ x = 1/2) :=
  sorry

end solution_l153_153520


namespace solve_for_x_l153_153901

theorem solve_for_x : ∃ x : ℝ, 16^x * 16^x * 16^(2 * x) = 64^6 ∧ x = 3 :=
by
  sorry

end solve_for_x_l153_153901


namespace find_value_of_b_cosC_plus_c_cosB_l153_153116

variable {A B C : Type} -- The angles
variable {a b c : ℝ} -- The sides of the triangle (opposite to the angles A, B, and C respectively)

theorem find_value_of_b_cosC_plus_c_cosB
  (h1 : a = 17)
  (h2 : a^2 = b^2 + c^2 - 2 * b * c * Math.cos(A))
  (h3 : b^2 = a^2 + c^2 - 2 * a * c * Math.cos(B))
  (h4 : c^2 = a^2 + b^2 - 2 * a * b * Math.cos(C)) :
  b * Math.cos(C) + c * Math.cos(B) = 17 := by
  sorry

end find_value_of_b_cosC_plus_c_cosB_l153_153116


namespace second_fisherman_more_fish_l153_153837

-- Define the given conditions
def days_in_season : ℕ := 213
def rate_first_fisherman : ℕ := 3
def rate_second_fisherman_phase_1 : ℕ := 1
def rate_second_fisherman_phase_2 : ℕ := 2
def rate_second_fisherman_phase_3 : ℕ := 4
def days_phase_1 : ℕ := 30
def days_phase_2 : ℕ := 60
def days_phase_3 : ℕ := days_in_season - (days_phase_1 + days_phase_2)

-- Define the total number of fish caught by each fisherman
def total_fish_first_fisherman : ℕ := rate_first_fisherman * days_in_season
def total_fish_second_fisherman : ℕ := 
  (rate_second_fisherman_phase_1 * days_phase_1) + 
  (rate_second_fisherman_phase_2 * days_phase_2) + 
  (rate_second_fisherman_phase_3 * days_phase_3)

-- Define the theorem statement
theorem second_fisherman_more_fish : 
  total_fish_second_fisherman = total_fish_first_fisherman + 3 := by sorry

end second_fisherman_more_fish_l153_153837


namespace min_value_of_expression_l153_153766

variable (a b c : ℝ)
variable (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
variable (h_eq : a * (a + b + c) + b * c = 4 - 2 * Real.sqrt 3)

theorem min_value_of_expression :
  2 * a + b + c ≥ 2 * Real.sqrt 3 - 2 := by
  sorry

end min_value_of_expression_l153_153766


namespace heights_parallel_l153_153156

variables (A B C H A1 B1 A2 B2 : Type) [triangle ABC]
  (height_AA1 : height A A1)
  (height_BB1 : height B B1)
  (intersect_H : intersect AA1 BB1 H)
  (height_A1A2 : height A1 A2)
  (height_B1B2 : height B1 B2)

theorem heights_parallel :
  A_2B_2 ∥ AB :=
sorry

end heights_parallel_l153_153156


namespace meeting_lantern_l153_153323

theorem meeting_lantern :
  let num_lanterns := 400
  let intervals := num_lanterns - 1
  let alla_start := 1
  let boris_start := 400
  let alla_at := 55
  let boris_at := 321
  let alla_intervals_covered := alla_at - alla_start
  let boris_intervals_covered := boris_start - boris_at
  let total_intervals_covered := alla_intervals_covered + boris_intervals_covered
  let total_intervals := intervals
  let proportion := ¬total_intervals_covered / total_intervals
  (alla_intervals_covered = 54) ∧ (boris_intervals_covered = 79) ∧
  (total_intervals_covered = 133) ∧ (total_intervals = 399) ∧
  (proportion = 1 / 3) → ∃ (meet_lantern : ℕ), meet_lantern = 163 :=
by
  sorry

end meeting_lantern_l153_153323


namespace factor_expression_l153_153353

theorem factor_expression (c : ℝ) : 180 * c ^ 2 + 36 * c = 36 * c * (5 * c + 1) := 
by
  sorry

end factor_expression_l153_153353


namespace sum_of_three_numbers_l153_153595

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35) 
  (h2 : b + c = 57) 
  (h3 : c + a = 62) : 
  a + b + c = 77 :=
by
  sorry

end sum_of_three_numbers_l153_153595


namespace min_value_of_2x_plus_4y_l153_153380

noncomputable def minimum_value (x y : ℝ) : ℝ := 2^x + 4^y

theorem min_value_of_2x_plus_4y (x y : ℝ) (h : x + 2 * y = 3) : minimum_value x y = 4 * Real.sqrt 2 :=
by
  sorry

end min_value_of_2x_plus_4y_l153_153380


namespace train_speed_correct_l153_153314

noncomputable def train_length : ℝ := 110
noncomputable def bridge_length : ℝ := 170
noncomputable def time_to_cross_bridge : ℝ := 13.998880089592832
noncomputable def total_distance : ℝ := train_length + bridge_length
noncomputable def speed_of_train : ℝ := total_distance / time_to_cross_bridge
noncomputable def expected_speed : ℝ := 20.0014286607

theorem train_speed_correct :
  speed_of_train ≈ expected_speed :=
sorry

end train_speed_correct_l153_153314


namespace find_f_l153_153488

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := sorry

axiom f_0 : f 0 = 0
axiom f_xy (x y : ℝ) : f (x * y) = f ((x^2 + y^2) / 2) + 3 * (x - y)^2

-- Theorem to be proved
theorem find_f (x : ℝ) : f x = -6 * x + 3 :=
by sorry -- proof goes here

end find_f_l153_153488


namespace triangle_cosine_identity_l153_153912

variables (α β γ : ℝ) (R r s : ℝ)

theorem triangle_cosine_identity
  (h_perimeter: 2 * s = 2 * s)
  (h_angles: α + β + γ = π)
  (h_radius: r = r) 
  (h_circumradius: R = R):
  4 * R^2 * cos α * cos β * cos γ = s^2 - (r + 2 * R)^2 :=
sorry

end triangle_cosine_identity_l153_153912


namespace oil_bottles_l153_153309

/-- Given that a store owner repacked his oils into 200 mL bottles and he had 4 liters of oil, 
prove that the number of bottles he made is 20. -/
theorem oil_bottles (oil_volume_l : ℕ) (bottle_volume_ml : ℕ) (units_conversion : ℕ) : 
    (oil_volume_l * units_conversion / bottle_volume_ml) = 20 :=
by
  -- Define the given conditions
  have h1 : oil_volume_l = 4 := by sorry
  have h2 : bottle_volume_ml = 200 := by sorry
  have h3 : units_conversion = 1000 := by sorry
  -- Use these conditions to establish the proof
  rw [h1, h2, h3]
  -- Calculate the left-hand side to prove it equals 20
  have calculation : 4 * 1000 / 200 = 20 := by sorry
  exact calculation

end oil_bottles_l153_153309


namespace pascal_triangle_47_l153_153095

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153095


namespace number_of_lines_l153_153884

theorem number_of_lines (S : ℕ) (n : ℕ)
  (h_condition_1 : ∀ i, i ∈ (list.range n) → S = 15 + (S - 15))
  (h_condition_2 : ∀ i, i ∈ (list.range n) → 15 ≤ S)
  : (S = 16) ∨ (S = 18) ∨ (S = 20) ∨ (S = 30) := 
by
  sorry

end number_of_lines_l153_153884


namespace larger_square_uncovered_area_l153_153308

theorem larger_square_uncovered_area :
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  (area_larger - area_smaller) = 84 :=
by
  let side_length_larger := 10
  let side_length_smaller := 4
  let area_larger := side_length_larger ^ 2
  let area_smaller := side_length_smaller ^ 2
  sorry

end larger_square_uncovered_area_l153_153308


namespace total_pages_written_is_24_l153_153526

def normal_letter_interval := 3
def time_per_normal_letter := 20
def time_per_page := 10
def additional_time_factor := 2
def time_spent_long_letter := 80
def days_in_month := 30

def normal_letters_written := days_in_month / normal_letter_interval
def pages_per_normal_letter := time_per_normal_letter / time_per_page
def total_pages_normal_letters := normal_letters_written * pages_per_normal_letter

def time_per_page_long_letter := additional_time_factor * time_per_page
def pages_long_letter := time_spent_long_letter / time_per_page_long_letter

def total_pages_written := total_pages_normal_letters + pages_long_letter

theorem total_pages_written_is_24 : total_pages_written = 24 := by
  sorry

end total_pages_written_is_24_l153_153526


namespace center_square_side_length_is_approximately_89_l153_153532

noncomputable def side_length_center_square (side_length_large_square : ℝ) (ratio_area_L_shape : ℝ) : ℝ :=
  let total_area := side_length_large_square^2
  let total_area_L_shapes := 4 * ratio_area_L_shape * total_area
  let center_square_area := total_area - total_area_L_shapes
  real.sqrt center_square_area

theorem center_square_side_length_is_approximately_89 :
  (side_length_center_square 120 (5/18)) ≈ 89 := 
sorry

end center_square_side_length_is_approximately_89_l153_153532


namespace bob_distance_from_start_l153_153300

theorem bob_distance_from_start (side_length : ℝ) (walk_distance : ℝ) :
  side_length = 3 → walk_distance = 7 → (euclidean_distance (7 *cos (π / 3)) (7  * sin  (7 * cos (π / 3)) ) = 2 * sqrt 3 := 
begin
sorry
end

end bob_distance_from_start_l153_153300


namespace prime_sum_max_l153_153392

theorem prime_sum_max (p q : ℕ) (hp : p.prime) (hq : q.prime) (h : ∃ r : ℕ, p^2 + 3*p*q + q^2 = r^2) : p + q ≤ 10 :=
sorry

end prime_sum_max_l153_153392


namespace abs_eq_neg_iff_non_positive_l153_153827

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end abs_eq_neg_iff_non_positive_l153_153827


namespace infinite_prime_divisors_l153_153877

theorem infinite_prime_divisors (a b c d : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (habcd : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
: ∃ᶠ p in filter.at_top (nat.filter (nat.prime)), ∃ n : ℕ, p ∣ a * c^n + b * d^n :=
sorry

end infinite_prime_divisors_l153_153877


namespace f_800_l153_153162

-- Definitions of hypothesis from conditions given
def f : ℕ → ℤ := sorry
axiom f_mul (x y : ℕ) : f (x * y) = f x + f y
axiom f_10 : f 10 = 10
axiom f_40 : f 40 = 18

-- Proof problem statement: prove that f(800) = 32
theorem f_800 : f 800 = 32 := 
by
  sorry

end f_800_l153_153162


namespace raghu_investment_l153_153281

-- Conditions given in the problem
variable (R T V : ℝ)
variable h1 : T = 0.90 * R  -- Trishul's investment is 10% less than Raghu's
variable h2 : V = 0.99 * R  -- Vishal's investment is 10% more than Trishul's
variable h3 : R + T + V = 7225  -- Total investment sum

-- The theorem we want to prove
theorem raghu_investment : R = 2500 :=
by
  sorry

end raghu_investment_l153_153281


namespace number_of_possible_committees_l153_153706

variable (university : Type) (department : university → Type)
variable (Professors : department → Type)
variable (Man : Professors → Prop) (Woman : Professors → Prop)
variable [DecidablePred Man] [DecidablePred Woman]

-- Assume there are exactly 3 male and 3 female professors in each department
variable (men_in_dept : ∀ d : university, Fin 3) 
variable (women_in_dept : ∀ d : university, Fin 3)

-- Assume the condition that each department is represented
variable (committee_condition : ∀ d : university, ∃ p1 p2 p3 p4 : Professors d, Man p1 ∧ Man p2 ∧ Woman p3 ∧ Woman p4)

-- The problem statement: Find the number of possible committees
noncomputable def possible_committees : Nat :=
  if h : ∀ d : university, ∃ m1 m2 : Professors d, Man m1 ∧ Man m2 ∧ 
     ∃ w1 w2 : Professors d, Woman w1 ∧ Woman w2 then Sorry else 0

theorem number_of_possible_committees (u : university) :
  ∀ (committee : Fin 8 → Professors u), possible_committees university department Professors =
  59049 :=
sorry

end number_of_possible_committees_l153_153706


namespace remainder_7n_mod_4_l153_153252

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l153_153252


namespace line_l_passes_through_fixed_point_intersecting_lines_find_k_l153_153422

-- Define the lines
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0
def line_l1 (x y : ℝ) : Prop := 2 * x + 3 * y + 8 = 0
def line_l2 (x y : ℝ) : Prop := x - y - 1 = 0

-- 1. Prove line l passes through the point (-2, 1)
theorem line_l_passes_through_fixed_point (k : ℝ) :
  line_l k (-2) 1 :=
by sorry

-- 2. Given lines l, l1, and l2 intersect at a single point, find k
theorem intersecting_lines_find_k (k : ℝ) :
  (∃ x y : ℝ, line_l k x y ∧ line_l1 x y ∧ line_l2 x y) ↔ k = -3 :=
by sorry

end line_l_passes_through_fixed_point_intersecting_lines_find_k_l153_153422


namespace collinear_sequence_specific_distance_l153_153510

-- Define the sequence and the conditions
variables {P : Type*} [metric_space P] [normed_add_comm_group P] [normed_space ℝ P]
variables (A : ℕ → P) (radius : ℝ) (h_radius : radius = 1)

-- Conditions
variable (h_nondiameter : ∀ n, ¬ (A n = A (n + 2)))
variable (h_circumcenter : ∀ n, A n = circumcenter (A (n-1)) (A (n-2)) (A (n-3)))

-- Theorems to prove
theorem collinear_sequence :
  ∀ (k: ℕ), (∃ (l : list ℕ), ∀ n ∈ l, n = 1 + 4*k) →
  collinear {A (1 + 4*k) : k ∈ ℕ} := by
  sorry

theorem specific_distance : 
  ∀ (x : ℝ), 
  (x = dist (A 1) (A 3)) → 
  (x = dist (A 3) (A 5)) →
  ∀ n >= 1001, 
  ∀ m >= 2001,
  ∃ (k : ℤ), (dist (A 1) (A 1001)) / (dist (A 1001) (A 2001)) = k^500 ∧
  (x = sqrt 2) := by
  sorry

end collinear_sequence_specific_distance_l153_153510


namespace abs_eq_neg_iff_non_positive_l153_153828

theorem abs_eq_neg_iff_non_positive (a : ℝ) : |a| = -a → a ≤ 0 :=
by
  intro h
  sorry

end abs_eq_neg_iff_non_positive_l153_153828


namespace sugar_amount_l153_153304

noncomputable def cups_required (total_cups: ℚ) (fraction: ℚ) : ℚ := 
  total_cups * fraction

def mixed_number (num : ℚ) : ℚ × ℚ :=
  let whole := num.toInt
  let frac := num - whole
  (whole, frac)

theorem sugar_amount (s : ℚ) : 
  let total_cups : ℚ := 5 + 3/4
  let required_cups : ℚ := 1 + 11/12 in
  s = total_cups →
  cups_required s (1/3) = required_cups :=
begin
  intros h,
  rw h,
  norm_num,
  rw [←fraction_def 23 12, mixed_number_of_int]
  sorry
end

end sugar_amount_l153_153304


namespace prob_drawing_2_or_3_white_balls_prob_drawing_at_least_1_white_ball_prob_drawing_at_least_1_black_ball_l153_153118

open Finset

-- Define the conditions
def white_balls := 5
def black_balls := 4
def total_balls := white_balls + black_balls
def drawn_balls := 3

-- Define combinations
def choose (n k : ℕ) : ℕ := (nat.choose n k).to_nat

-- Define the events
def event_1 := choose white_balls 2 * choose black_balls 1
def event_2 := choose white_balls 3
def total_event := choose total_balls drawn_balls

-- Define probabilities
def prob_event_1 := (event_1 + event_2) / total_event
def prob_event_2 := 1 - (choose black_balls 3 / total_event)
def prob_event_3 := 1 - (choose white_balls 3 / total_event)

-- Proof problem statements
theorem prob_drawing_2_or_3_white_balls : prob_event_1 = 25 / 42 := by
  sorry

theorem prob_drawing_at_least_1_white_ball : prob_event_2 = 20 / 21 := by
  sorry

theorem prob_drawing_at_least_1_black_ball : prob_event_3 = 37 / 42 := by
  sorry

end prob_drawing_2_or_3_white_balls_prob_drawing_at_least_1_white_ball_prob_drawing_at_least_1_black_ball_l153_153118


namespace solve_abs_quadratic_eq_l153_153521

theorem solve_abs_quadratic_eq (x : ℝ) (h : |2 * x + 4| = 1 - 3 * x + x ^ 2) :
    x = (5 + Real.sqrt 37) / 2 ∨ x = (5 - Real.sqrt 37) / 2 := by
  sorry

end solve_abs_quadratic_eq_l153_153521


namespace population_proof_l153_153608

noncomputable def population_after_two_years (P_initial : ℤ) (increase_percent decrease_percent : ℚ) : ℤ :=
  let P_end_first_year := P_initial + (P_initial * increase_percent).toInt
  let P_end_second_year := P_end_first_year - (P_end_first_year * decrease_percent).toInt
  P_end_second_year

theorem population_proof :
  population_after_two_years 415600 0.25 0.30 = 363650 := 
by 
  sorry

end population_proof_l153_153608


namespace max_diff_six_digit_even_numbers_l153_153674

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l153_153674


namespace distance_from_dorm_to_city_l153_153850

theorem distance_from_dorm_to_city (D : ℝ) (h1 : D = (1/4)*D + (1/2)*D + 10 ) : D = 40 :=
sorry

end distance_from_dorm_to_city_l153_153850


namespace parallelogram_diagonals_intersect_at_midpoint_l153_153509

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem parallelogram_diagonals_intersect_at_midpoint :
  midpoint (2, -3) (10, 9) = (6, 3) :=
by sorry

end parallelogram_diagonals_intersect_at_midpoint_l153_153509


namespace K_time_is_10_hours_l153_153615

-- Define the speeds of K and M, where x is K's speed
noncomputable def speed_K (x : ℕ) : ℕ := x
noncomputable def speed_M (x : ℕ) : ℕ := x - 1

-- Define the time taken by K and M to travel 40 miles
noncomputable def time_K (x : ℕ) : ℕ := 40 / speed_K x
noncomputable def time_M (x : ℕ) : ℕ := 40 / speed_M x

-- The condition that K takes 40 minutes (2/3 hour) less than M to travel 40 miles
axiom time_difference_condition (x : ℕ) : time_M x - time_K x = 2 / 3

-- The statement to be proven: K's time for the distance is 10 hours
theorem K_time_is_10_hours (x : ℕ) (h : time_difference_condition x) : time_K x = 10 :=
  sorry

end K_time_is_10_hours_l153_153615


namespace Wendy_did_not_recycle_2_bags_l153_153568

theorem Wendy_did_not_recycle_2_bags (points_per_bag : ℕ) (total_bags : ℕ) (points_earned : ℕ) (did_not_recycle : ℕ) : 
  points_per_bag = 5 → 
  total_bags = 11 → 
  points_earned = 45 → 
  5 * (11 - did_not_recycle) = 45 → 
  did_not_recycle = 2 :=
by
  intros h_points_per_bag h_total_bags h_points_earned h_equation
  sorry

end Wendy_did_not_recycle_2_bags_l153_153568


namespace defective_rate_20_percent_l153_153026

open_locale big_operators

noncomputable def defective_rate (n : ℕ) : ℝ := n / 10.0

theorem defective_rate_20_percent (p : ℝ) (ξ : ℝ)
  (h1 : p = 16/45)
  (h2 : ξ = 1)
  (h3 : ∀ n ≤ 4, P(ξ = n) ≤ p) : defective_rate 2 = 0.2 :=
by
  sorry

end defective_rate_20_percent_l153_153026


namespace line_equation_l153_153617

theorem line_equation (x y : ℝ) (m : ℝ) (h1 : (1, 2) = (x, y)) (h2 : m = 3) :
  y = 3 * x - 1 :=
by
  sorry

end line_equation_l153_153617


namespace number_of_valid_six_digit_numbers_l153_153761

noncomputable def total_valid_numbers : ℕ :=
  let digits : Finset ℕ := {1, 2, 3, 4, 5, 6}
  let is_valid_permutation (perm : List ℕ) :=
    (perm.length = 6) ∧
    (perm.nodup) ∧
    (∀ i, i < perm.length - 1 → (if perm.nth_le i sorry % 2 = 1 then perm.nth_le (i + 1) sorry % 2 = 0 else True)) ∧
    (perm.nth_le 3 sorry ≠ 4)
  in (Finset.permutations_of_multiset digits.val).countP is_valid_permutation

theorem number_of_valid_six_digit_numbers : total_valid_numbers = 120 := 
  sorry

end number_of_valid_six_digit_numbers_l153_153761


namespace num_geography_books_l153_153227

theorem num_geography_books
  (total_books : ℕ)
  (history_books : ℕ)
  (math_books : ℕ)
  (h1 : total_books = 100)
  (h2 : history_books = 32)
  (h3 : math_books = 43) :
  total_books - history_books - math_books = 25 :=
by
  sorry

end num_geography_books_l153_153227


namespace pascal_triangle_contains_47_l153_153067

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153067


namespace A_eq_B_l153_153878

noncomputable def A (a : ℕ) : set ℕ :=
  {k | ∃ x y : ℤ, x > int.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

noncomputable def B (a : ℕ) : set ℕ :=
  {k | ∃ x y : ℤ, 0 ≤ x ∧ x < int.sqrt a ∧ k = (x^2 - a) / (x^2 - y^2)}

theorem A_eq_B (a : ℕ) (h : ¬(∃ (m : ℕ), a = m * m)) : A a = B a := 
by 
  sorry

end A_eq_B_l153_153878


namespace centerville_budget_l153_153945

/-- Define the annual budget. --/
def annual_budget (library_spending : Real) : Real :=
  library_spending / 0.15

/-- Define the spending on public parks. --/
def parks_spending (total_budget : Real) : Real :=
  0.24 * total_budget

/-- Calculate the amount left of the annual budget. --/
def amount_left (total_budget library_spending parks_spending : Real) : Real :=
  total_budget - (library_spending + parks_spending)

/-- The town of Centerville spends 15% of its annual budget on its public library. 
    Centerville spent $3,000 on its public library and 24% on the public parks. 
    We prove that the amount left of the annual budget is $12,200. --/
theorem centerville_budget (library_spending : Real) (h_library : library_spending = 3000) :
  amount_left (annual_budget library_spending) library_spending (parks_spending (annual_budget library_spending)) = 12200 :=
by
  -- Since no proof is required, we use sorry here.
  sorry

end centerville_budget_l153_153945


namespace find_m_l153_153770

-- Define the points and slope condition
def A : ℝ × ℝ := (2, 4)
def B (m : ℝ) : ℝ × ℝ := (1, m)

-- Define the inclination angle condition's implication into the slope of the line
def inclination_angle (angle : ℝ) : Prop :=
  ∀ (A B : ℝ × ℝ), angle = 45 → (B.1 - A.1 ≠ 0) → ((B.2 - A.2) / (B.1 - A.1) = 1)

-- Given conditions, prove that m = 3
theorem find_m (m : ℝ) (h : inclination_angle 45 (2, 4) (1, m)) : m = 3 :=
by
  sorry

end find_m_l153_153770


namespace coin_flip_heads_probability_l153_153962

theorem coin_flip_heads_probability :
  (∑ k in (finset.range 4).map (λ i, 9 + i), nat.choose 12 k : ℚ) / 4096 = 299 / 4096 :=
by sorry

end coin_flip_heads_probability_l153_153962


namespace part1_fixed_point_part1_vertices_on_parabola_part2_larger_root_range_l153_153865

variables (a : ℝ)

-- Part (1) Fixed Point
theorem part1_fixed_point : ∀ a : ℝ, ∃ (x y : ℝ), (x = 2) ∧ (y = 9) ∧ (y = x^2 + (a + 2) * x - 2 * a + 1) := 
by
  intros a 
  use 2
  use 9
  split
  exact eq.refl 2
  split
  exact eq.refl 9
  calc
    9 = 2^2 + (a + 2) * 2 - 2 * a + 1 := by ring
sorry

-- Part (1) Vertices on a Parabola
theorem part1_vertices_on_parabola : ∀ a : ℝ, ∃ (x : ℝ), x = - (1 / 4) * a^2 - 3 * a :=
by
  intros a
  use - (1 / 4) * a^2 - 3 * a
  ring
sorry

-- Part (2) Larger Root Range
theorem part2_larger_root_range : ∀ a : ℝ, (a > 0 ∨ a < -12) → ∃ b : ℝ, (b = max ((-(a + 2) + real.sqrt (a^2 + 12 * a)) / 2) ((-(a + 2) - real.sqrt (a^2 + 12 * a)) / 2)) → (b > -1 ∨ b > 5) :=
by
  intros a h
  have discriminant_positive : a^2 + 12 * a > 0 := 
    by 
      cases h
      case or.inl { linarith only [real.sqrt_pos.mpr] }
      case or.inr { linarith only [real.sqrt_lt_zero.mpr] }
  use max ((-(a + 2) + real.sqrt (a^2 + 12 * a)) / 2) ((-(a + 2) - real.sqrt (a^2 + 12 * a)) / 2)
  intro root_eq
  have root_behaviour := root_eq
  linarith only [discriminant_positive]
sorry

end part1_fixed_point_part1_vertices_on_parabola_part2_larger_root_range_l153_153865


namespace blocks_differs_in_exactly_two_ways_correct_l153_153288

structure Block where
  material : Bool       -- material: false for plastic, true for wood
  size : Fin 3          -- sizes: 0 for small, 1 for medium, 2 for large
  color : Fin 4         -- colors: 0 for blue, 1 for green, 2 for red, 3 for yellow
  shape : Fin 4         -- shapes: 0 for circle, 1 for hexagon, 2 for square, 3 for triangle
  finish : Bool         -- finish: false for glossy, true for matte

def originalBlock : Block :=
  { material := false, size := 1, color := 2, shape := 0, finish := false }

def differsInExactlyTwoWays (b1 b2 : Block) : Bool :=
  (if b1.material ≠ b2.material then 1 else 0) +
  (if b1.size ≠ b2.size then 1 else 0) +
  (if b1.color ≠ b2.color then 1 else 0) +
  (if b1.shape ≠ b2.shape then 1 else 0) +
  (if b1.finish ≠ b2.finish then 1 else 0) == 2

def countBlocksDifferingInTwoWays : Nat :=
  let allBlocks := List.product
                  (List.product
                    (List.product
                      (List.product
                        [false, true]
                        ([0, 1, 2] : List (Fin 3)))
                      ([0, 1, 2, 3] : List (Fin 4)))
                    ([0, 1, 2, 3] : List (Fin 4)))
                  [false, true]
  (allBlocks.filter
    (λ b => differsInExactlyTwoWays originalBlock
                { material := b.1.1.1.1, size := b.1.1.1.2, color := b.1.1.2, shape := b.1.2, finish := b.2 })).length

theorem blocks_differs_in_exactly_two_ways_correct :
  countBlocksDifferingInTwoWays = 51 :=
  by
    sorry

end blocks_differs_in_exactly_two_ways_correct_l153_153288


namespace ratio_of_spinsters_to_cats_l153_153547

theorem ratio_of_spinsters_to_cats (S C : ℕ) (hS : S = 12) (hC : C = S + 42) : S / gcd S C = 2 ∧ C / gcd S C = 9 :=
by
  -- skip proof (use sorry)
  sorry

end ratio_of_spinsters_to_cats_l153_153547


namespace smallest_number_in_set_l153_153324

noncomputable def negative_sqrt2 := -2 * Real.sqrt 2

theorem smallest_number_in_set : 
  ∀ (a : ℝ), a ∈ ({1, 0, negative_sqrt2, -3} : set ℝ) → -3 ≤ a :=
by
  intros a h
  simp [negative_sqrt2] at h
  fin_cases h with h1 h0 hn2sqrt2 h3
  · linarith
  · linarith
  · rw h, have h := real.sqrt_pos.2 (show (0 : ℝ) < 2, by linarith)
    linarith
  · linarith
  sorry

end smallest_number_in_set_l153_153324


namespace remainder_of_7n_mod_4_l153_153245

theorem remainder_of_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by 
  sorry

end remainder_of_7n_mod_4_l153_153245


namespace find_y_l153_153849

-- Definitions
def point (x y z : ℝ) := (x, y, z : ℝ × ℝ × ℝ)

def distance (p q : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2 + (p.3 - q.3) ^ 2)

-- Points definition
def P : ℝ × ℝ × ℝ := point 1 0 2
def Q : ℝ × ℝ × ℝ := point 1 (-3) 1

-- Given M is on the y-axis
def M (y : ℝ) : ℝ × ℝ × ℝ := point 0 y 0

-- Problem Statement
theorem find_y (y : ℝ) : distance (M y) P = distance (M y) Q → y = -1 := by
  sorry

end find_y_l153_153849


namespace absentees_count_l153_153908

theorem absentees_count 
  (yesterday_students : ℕ)
  (total_registered : ℕ)
  (attendance_percent : ℝ)
  (students_attended_today : ℕ) 
  (absent_students : ℕ) : 
    yesterday_students = 70 ∧ 
    total_registered = 156 ∧ 
    attendance_percent = 0.1 ∧ 
    students_attended_today = Nat.floor (2 * 70 * (1 - 0.1)) ∧
    absent_students = total_registered - students_attended_today 
  → absent_students = 30 :=
begin
  sorry
end

end absentees_count_l153_153908


namespace pascals_triangle_contains_47_once_l153_153072

theorem pascals_triangle_contains_47_once (n : ℕ) : 
  (∃ k, k ≤ n ∧ Nat.choose n k = 47) ↔ n = 47 := by
  sorry

end pascals_triangle_contains_47_once_l153_153072


namespace camel_cannot_end_adjacent_l153_153613

-- Define the size of the board
def hex_board (m : ℕ) := 3 * m^2 - 3 * m + 1

-- Define a field type (color can only be populated based on proof context)
inductive Field : Type
| black : Field
| white : Field

-- A move on the board (the fields are represented as colored, but the actual positions are not modeled here)
noncomputable def move (start : Field) (moves : ℕ) : Field :=
if (moves % 3 == 0) then start else if (start = Field.black) then Field.white else Field.black

-- The main theorem stating the impossibility
theorem camel_cannot_end_adjacent (m : ℕ) (start : Field) :
  (move start (hex_board m - 1) = start) → false := 
by
  sorry

end camel_cannot_end_adjacent_l153_153613


namespace remainder_when_7n_divided_by_4_l153_153250

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
  sorry

end remainder_when_7n_divided_by_4_l153_153250


namespace least_possible_lcm_l153_153207

-- Definitions of the least common multiples given the conditions
variable (a b c : ℕ)
variable (h₁ : Nat.lcm a b = 20)
variable (h₂ : Nat.lcm b c = 28)

-- The goal is to prove the least possible value of lcm(a, c) given the conditions
theorem least_possible_lcm (a b c : ℕ) (h₁ : Nat.lcm a b = 20) (h₂ : Nat.lcm b c = 28) : Nat.lcm a c = 35 :=
by
  sorry

end least_possible_lcm_l153_153207


namespace probability_heads_at_least_9_l153_153974

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153974


namespace find_max_difference_l153_153686

theorem find_max_difference :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a ≤ 999999) ∧
    (100000 ≤ b ∧ b ≤ 999999) ∧
    (∀ d : ℕ, d ∈ List.digits a → d % 2 = 0) ∧
    (∀ d : ℕ, d ∈ List.digits b → d % 2 = 0) ∧
    (a < b) ∧
    (∀ c : ℕ, a < c ∧ c < b → ∃ d : ℕ, d ∈ List.digits c ∧ d % 2 = 1) ∧
    b - a = 111112 := sorry

end find_max_difference_l153_153686


namespace van_distance_l153_153315

theorem van_distance (h_initial_time : ∀ D : ℝ, D / 6 = D / 42 * (9 / 42)) : 
  ∀ D : ℝ, D = 42 * 9 → D = 378 :=
by 
  assume D hD,
  rw [<- hD],
  norm_num

end van_distance_l153_153315


namespace back_wheel_revolutions_l153_153504

theorem back_wheel_revolutions
  (r_front : ℝ) (r_back : ℝ) (revolutions_front : ℕ)
  (h1 : r_front = 3) (h2 : r_back = 0.5) (h3 : revolutions_front = 50) :
  (2 * r_front * real.pi * revolutions_front) / (2 * r_back * real.pi) = 300 :=
by
  sorry

end back_wheel_revolutions_l153_153504


namespace roots_of_polynomial_sum_of_fourth_powers_l153_153489

theorem roots_of_polynomial_sum_of_fourth_powers (p q r s : ℂ) 
  (h1 : (polynomial.C p * polynomial.C q * polynomial.C r * polynomial.C s).roots = {p, q, r, s} ∧
        (polynomial.X ^ 4 - polynomial.C 1 * polynomial.X ^ 3 + polynomial.C 1 * polynomial.X ^ 2 - polynomial.C 3 * polynomial.X + polynomial.C 3) = 0) :
  p^4 + q^4 + r^4 + s^4 = 5 := 
begin
  sorry
end

end roots_of_polynomial_sum_of_fourth_powers_l153_153489


namespace polynomial_divisibility_l153_153390

theorem polynomial_divisibility (a b : ℕ) (ha : a > 0) (hb : b > 0) (h_cond : a ≥ 2 * b) : 
  ∃ (P : Polynomial ℕ), P.degree ≥ 1 ∧ (∀ i : ℕ, i ∈ P.coeffs → i < b) ∧ P.eval b ∣ P.eval a := 
sorry

end polynomial_divisibility_l153_153390


namespace regular_pentagon_l153_153881

open Complex

theorem regular_pentagon (z1 z2 z3 z4 z5 : ℂ) 
  (h_abs : ∀ i ∈ [z1, z2, z3, z4, z5], |i| = 1)
  (h_sum : z1 + z2 + z3 + z4 + z5 = 0)
  (h_squaresum : z1^2 + z2^2 + z3^2 + z4^2 + z5^2 = 0) 
  : ∃ θ (θ0 θ1 θ2 θ3 θ4 : ℂ), θ ≠ 0 ∧
    (θ0 = z1 ∧ θ1 = z2 ∧ θ2 = z3 ∧ θ3 = z4 ∧ θ4 = z5) ∧
    (θ0 = θ * exp(0 * (2 * π * Complex.I / 5))) ∧
    (θ1 = θ * exp(1 * (2 * π * Complex.I / 5))) ∧
    (θ2 = θ * exp(2 * (2 * π * Complex.I / 5))) ∧
    (θ3 = θ * exp(3 * (2 * π * Complex.I / 5))) ∧
    (θ4 = θ * exp(4 * (2 * π * Complex.I / 5))) :=
sorry

end regular_pentagon_l153_153881


namespace group_b_more_stable_l153_153624

theorem group_b_more_stable (
  variance_A : ℝ,
  variance_B : ℝ
) 
  (hA : variance_A = 36)
  (hB : variance_B = 30) : 
  variance_B < variance_A :=
by 
  -- This sorry is a placeholder to indicate the steps to the proof are not included here.
  sorry

end group_b_more_stable_l153_153624


namespace complement_intersection_l153_153812

open Set

def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 3, 4}
def B : Set ℕ := {2, 4}

theorem complement_intersection :
  ((U \ A) ∩ B) = {2} :=
sorry

end complement_intersection_l153_153812


namespace probability_heads_in_12_flips_l153_153998

noncomputable def probability_at_least_9_heads_flips (n : ℕ) (k : ℕ) : ℚ :=
  ∑ i in finset.range (k + 1), (nat.choose n i) / (2^n : ℚ)

theorem probability_heads_in_12_flips :
  probability_at_least_9_heads_flips 12 9 = 299 / 4096 :=
by
  sorry

end probability_heads_in_12_flips_l153_153998


namespace find_m_l153_153870

theorem find_m :
  ∃ (m : ℤ) (d e f g : ℤ), 
    (g(x) = d * x^3 + e * x^2 + f * x + g) ∧
    (g(1) = 0) ∧ 
    (70 < g(5) ∧ g(5) < 80) ∧
    (120 < g(6) ∧ g(6) < 130) ∧
    (10000 * m < g(50) ∧ g(50) < 10000 * (m + 1)) ∧
    m = 12 :=
by {
  sorry
}

end find_m_l153_153870


namespace detergent_box_ranking_l153_153289

section DetergentBoxRanking

  -- Define the basics
  variables (c_S c_M c_L c_XL q_S q_M q_L q_XL : ℝ)
  
  -- Problem conditions
  def medium_cost_more_than_small := c_M = 1.6 * c_S
  def medium_quantity_less_than_extralarge := q_M = 0.7 * q_XL
  def large_quantity_twice_small := q_L = 2 * q_S
  def large_cost_more_than_medium := c_L = 1.4 * c_M
  def extralarge_quantity_more_than_large := q_XL = 1.5 * q_L
  def extralarge_cost_more_than_large := c_XL = 1.25 * c_L

  -- Define cost per ounce for each size
  def cost_per_ounce_small := c_S / q_S
  def cost_per_ounce_medium := c_M / q_M
  def cost_per_ounce_large := c_L / q_L
  def cost_per_ounce_extralarge := c_XL / q_XL

  -- Prove that the ranking from best to worst buy is MXLS
  theorem detergent_box_ranking :
    (cost_per_ounce_medium < cost_per_ounce_extralarge) ∧ 
    (cost_per_ounce_extralarge < cost_per_ounce_small) ∧ 
    (cost_per_ounce_small < cost_per_ounce_large) :=
  sorry

end DetergentBoxRanking

end detergent_box_ranking_l153_153289


namespace gcd_inequality_l153_153491

theorem gcd_inequality (n : ℕ) (a : ℕ → ℕ) (P : ℕ) (hn : odd n) (ha : ∀ i : fin n, 0 < a i)
  (hP : P = finset.prod finset.univ (λ i : fin n, a i)) :
  let g := nat.gcd (a 0) (nat.gcd (a 1) (nat.gcd (a 2) ... (nat.gcd (a (n-1))))) in
  nat.gcd (a 0 ^ n + P) (nat.gcd (a 1 ^ n + P) (nat.gcd (a 2 ^ n + P) ... (nat.gcd (a (n-1) ^ n + P))) ≤ 2 * g ^ n :=
sorry

end gcd_inequality_l153_153491


namespace kevin_total_distance_hopped_l153_153859

theorem kevin_total_distance_hopped : 
  let distance := 2
  let fraction := 1/4 
  [distance_after_first_hop, distance_after_second_hop, distance_after_third_hop, distance_after_fourth_hop] = 
  [fraction * distance, 
  fraction * (3/4 * distance), 
  fraction * (3/4 * (3/4 * distance)), 
  fraction * (3/4 * (3/4 * (3/4 * distance)))] in
  distance_after_first_hop + distance_after_second_hop + distance_after_third_hop + distance_after_fourth_hop = 175/128 := 
by
  let distance := 2
  let fraction := 1/4 
  let distance_after_first_hop := fraction * distance
  let distance_after_second_hop := fraction * (3/4 * distance)
  let distance_after_third_hop := fraction * (3/4 * (3/4 * distance))
  let distance_after_fourth_hop := fraction * (3/4 * (3/4 * (3/4 * distance)))
  have step1 : distance_after_first_hop = 1/2 := by norm_num
  have step2 : distance_after_second_hop = 3/8 := by norm_num
  have step3 : distance_after_third_hop = 9/32 := by norm_num
  have step4 : distance_after_fourth_hop = 27/128 := by norm_num
  have total_distance := step1 + step2 + step3 + step4
  have steps_equal : (1/2) + (3/8) + (9/32) + (27/128) = 175/128 := by norm_num
  exact steps_equal

end kevin_total_distance_hopped_l153_153859


namespace nancy_initial_files_correct_l153_153501

-- Definitions based on the problem conditions
def initial_files (deleted_files : ℕ) (folder_count : ℕ) (files_per_folder : ℕ) : ℕ :=
  (folder_count * files_per_folder) + deleted_files

-- The proof statement
theorem nancy_initial_files_correct :
  initial_files 31 7 7 = 80 :=
by
  sorry

end nancy_initial_files_correct_l153_153501


namespace sequence_sum_to_2008_l153_153336

def sequence (n : ℕ) : ℤ :=
  if n % 4 == 0 then (n / 4 + 1) * 4 - 3
  else if n % 4 == 1 then -(n / 4 + 1) * 4 + 1
  else if n % 4 == 2 then -(n / 4 + 1) * 4 + 2
  else (n / 4 + 1) * 4

theorem sequence_sum_to_2008 : 
  ∑ i in Finset.range 2008, sequence (i + 1) = 0 :=
sorry

end sequence_sum_to_2008_l153_153336


namespace sum_of_fraction_l153_153388

-- Define the arithmetic sequence conditions
def arithmetic_seq (a : ℕ → ℚ) (S : ℕ → ℚ) (a_9_eq : a 9 = (1 / 2) * (a 12) + 6) (a_2_eq : a 2 = 4) :=
  ∃ (a1 d : ℚ), (∀ n, a n = a1 + (n - 1) * d) ∧ (∀ n, S n = n * (a1 + (n - 1) / 2 * d))

-- Define the sum of the first 10 terms of the sequence 1/S_n
def sum_of_first_10_terms (S : ℕ → ℚ) := (∑ n in Finset.range 10, 1 / S (n + 1))

-- The main theorem that we need to prove
theorem sum_of_fraction (a S : ℕ → ℚ)
  (a_9_eq : a 9 = (1 / 2) * (a 12) + 6)
  (a_2_eq : a 2 = 4)
  (ha_sn : arithmetic_seq a S a_9_eq a_2_eq)
  : sum_of_first_10_terms S = 10 / 11 :=
by sorry

end sum_of_fraction_l153_153388


namespace largest_possible_set_size_l153_153864

theorem largest_possible_set_size
  (S : Set ℝ) (hS : S.Finite) 
  (cond : ∀ x y z : ℝ, x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z →
             (x + y ∈ S ∨ x + z ∈ S ∨ y + z ∈ S)) :
  S.toFinset.card ≤ 7 :=
sorry

end largest_possible_set_size_l153_153864


namespace min_b_over_a_l153_153429

noncomputable theory
open Real

variables (m : ℝ) (h : m > 0)
def x_A := 2^(-m)
def x_B := 2^(m)
def x_C := 2^(- (8 / (2 * m + 1)))
def x_D := 2^((8 / (2 * m + 1)))

def a := abs (x_A m - x_C m)
def b := abs (x_B m - x_D m)

theorem min_b_over_a (m : ℝ) (h : m > 0) : 
  ∃ m, 2^(m + 8 / (2 * m + 1)) = 8 * sqrt(2) :=
begin
  -- This proof is omitted
  sorry
end

end min_b_over_a_l153_153429


namespace pascal_triangle_contains_47_l153_153064

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153064


namespace range_of_a_l153_153810

theorem range_of_a (a : ℝ) : 
  (∀ θ ∈ Ioo 0 (π / 2), a ≤ (1 / Real.sin θ) + (1 / Real.cos θ)) → a ≤ 2 * Real.sqrt 2 :=
by
  intros h
  -- Full proof required here
  sorry

end range_of_a_l153_153810


namespace min_faces_to_repaint_l153_153909

theorem min_faces_to_repaint (n_cubes : ℕ) (faces_per_cube : ℕ) (initial_white_faces : ℕ)
  (repainted_black_faces : ℕ) (target_cube_faces : ℕ) :
  n_cubes = 8 →
  faces_per_cube = 6 →
  initial_white_faces = 48 →
  repainted_black_faces ≥ 2 →
  target_cube_faces = 6 →
  ∃ (repaint_faces : ℕ), repaint_faces = 2 ∧
  (∃ (arrangement_blackface : ℕ), arrangement_blackface ∈ finset.range (8 * 6) ∧
   ∀ (arrangement : finset (finset (ℕ × ℕ × ℕ))),
     finset.card arrangement = 8 → 
     (∃ (outer_face_black : ℕ), outer_face_black ∈ finset.range 6 ∧ repaint_faces ≤ 2)) :=
begin
  sorry,
end

end min_faces_to_repaint_l153_153909


namespace pascal_triangle_contains_prime_l153_153054

theorem pascal_triangle_contains_prime :
  ∃! n, ∃ k, (0 ≤ k ∧ k ≤ n) ∧ (nat.prime 47) ∧ nat.choose n k = 47 :=
begin
  sorry
end

end pascal_triangle_contains_prime_l153_153054


namespace find_matrix_n_l153_153744

theorem find_matrix_n :
  ∃ (N : Matrix (Fin 3) (Fin 3) ℤ),
    (N ⬝ (Matrix.vecCons 1 0 0 0 0 0 0 0 0 0) = (Matrix.vecCons 4 6 (-16) 0 0 0 0 0 0 0)) ∧
    (N ⬝ (Matrix.vecCons 0 1 0 0 0 0 0 0 0 0) = (Matrix.vecCons 0 10 (-4) 0 0 0 0 0 0 0)) ∧
    (N ⬝ (Matrix.vecCons 0 0 1 0 0 0 0 0 0 0) = (Matrix.vecCons 14 (-2) 8 0 0 0 0 0 0 0)) ∧
    (N = (Matrix.of_vector
      (Vector.of_array
        #[#[4, 0, 14],
          #[6, 10, -2],
          #[-16, -4, 8]]
        sorry : _()))

end find_matrix_n_l153_153744


namespace repeating_and_symmetric_l153_153893

def sequence (n : ℕ) : ℕ := (n - 1) * n

def last_digit (n : ℕ) : ℕ := n % 10

theorem repeating_and_symmetric (n : ℕ) : 
  (last_digit (sequence (n + 5)) = last_digit (sequence n)) ∧
  (∀ a b, a + b = 5 → last_digit (sequence a) = last_digit (sequence b)) :=
by
  sorry

end repeating_and_symmetric_l153_153893


namespace jericho_money_problem_l153_153149

theorem jericho_money_problem
  (j_owes_annika : ℤ) (j_owes_manny_half : j_owes_manny_half = j_owes_annika / 2)
  (j_left_after_debts : ℤ) (initial_money : ℤ)
  (total_debts : ℤ) (twice_initial_money : ℤ) :
  j_owes_annika = 14 → 
  j_owes_manny_half = 7 → 
  j_left_after_debts = 9 → 
  total_debts = j_owes_annika + j_owes_manny_half →
  initial_money = j_left_after_debts + total_debts →
  twice_initial_money = 2 * initial_money →
  twice_initial_money = 60 := 
by {
  intros h_annika h_manny h_left h_debts h_initial h_twice,
  sorry
}

end jericho_money_problem_l153_153149


namespace shaded_area_l153_153454

theorem shaded_area (r : ℝ) (T : ℝ) (h1 : r = 4) (h2 : ∃ A B O, ∠ A O B = 90) 
    (I J K : Point) (h3 : I = midpoint O B) (h4 : K ∈ semicircle I B)
    (h5 : parallel I J A O) : 
    T = 5 - 2 * sqrt 3 :=
begin
    -- sorry to be filled
    sorry
end

end shaded_area_l153_153454


namespace remainder_7n_mod_4_l153_153260

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := by
  sorry

end remainder_7n_mod_4_l153_153260


namespace range_of_a_l153_153486

variable (a : ℝ)
variable (f : ℝ → ℝ)

-- Conditions
def isOddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

def fWhenNegative (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x < 0 → f x = 9 * x + a^2 / x + 7

def fNonNegativeCondition (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, x ≥ 0 → f x ≥ a + 1

-- Theorem to prove
theorem range_of_a (odd_f : isOddFunction f) (f_neg : fWhenNegative f a) 
  (nonneg_cond : fNonNegativeCondition f a) : 
  a ≤ -8 / 7 :=
by
  sorry

end range_of_a_l153_153486


namespace add_ab_equals_four_l153_153783

theorem add_ab_equals_four (a b : ℝ) (h₁ : a * (a - 4) = 5) (h₂ : b * (b - 4) = 5) (h₃ : a ≠ b) : a + b = 4 :=
by
  sorry

end add_ab_equals_four_l153_153783


namespace largest_diff_even_digits_l153_153667

theorem largest_diff_even_digits (a b : ℕ) (ha : 100000 ≤ a) (hb : b ≤ 999998) (h6a : a < 1000000) (h6b : b < 1000000)
  (h_all_even_digits_a : ∀ d ∈ Nat.digits 10 a, d % 2 = 0)
  (h_all_even_digits_b : ∀ d ∈ Nat.digits 10 b, d % 2 = 0)
  (h_between_contains_odd : ∀ x, a < x → x < b → ∃ d ∈ Nat.digits 10 x, d % 2 = 1) : b - a = 111112 :=
sorry

end largest_diff_even_digits_l153_153667


namespace balls_color_equality_l153_153530

theorem balls_color_equality (r g b: ℕ) (h1: r + g + b = 20) (h2: b ≥ 7) (h3: r ≥ 4) (h4: b = 2 * g) : 
  r = b ∨ r = g :=
by
  sorry

end balls_color_equality_l153_153530


namespace variance_product_l153_153890

variables (X Y : ℝ → ℝ) -- We assume X and Y to be random variables.
variable [measure_space ℝ]
variable {μ : measure ℝ} -- We will use measure theory to formalize probability.
variable (m n : ℝ) -- Means of X and Y respectively.

-- Definitions for the means
def mean_X := ∫ x, X x ∂μ
def mean_Y := ∫ y, Y y ∂μ

-- Variance definitions
def variance (Z : ℝ → ℝ) := ∫ z, (Z z - ∫ x, Z x ∂μ)^2 ∂μ

-- Conditions
axiom X_Y_independent : independent X Y
axiom mean_X_def : mean_X = m
axiom mean_Y_def : mean_Y = n

theorem variance_product :
  variance (λ ω, X ω * Y ω) = variance X * variance Y + n^2 * variance X + m^2 * variance Y :=
sorry

end variance_product_l153_153890


namespace evaluate_operations_l153_153758

def spadesuit (x y : ℝ) := (x + y) * (x - y)
def heartsuit (x y : ℝ) := x ^ 2 - y ^ 2

theorem evaluate_operations : spadesuit 5 (heartsuit 3 2) = 0 :=
by
  sorry

end evaluate_operations_l153_153758


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153988

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153988


namespace perimeter_bound_l153_153472

variable {n : ℕ} (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ)

noncomputable def vector_sum (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ) : ℝ × ℝ :=
  Finset.univ.sum (λ i, (A i).1 - O.1, (A i).2 - O.2)

noncomputable def perimeter (A : Fin n → ℝ × ℝ) : ℝ :=
  Finset.univ.sum (λ i, (A i).1 - (A ((i + 1) % n)).1, (A i).2 - (A ((i + 1) % n)).2).norm

noncomputable def total_distance (A : Fin n → ℝ × ℝ) (O : ℝ × ℝ) : ℝ :=
  Finset.univ.sum (λ i, (A i).1 - O.1, (A i).2 - O.2).norm

theorem perimeter_bound :
  (vector_sum A O = (0, 0)) →
  (∃ d : ℝ, d = total_distance A O) →
  if n % 2 = 0 then perimeter A ≥ (4 * d / n)
  else perimeter A ≥ (4 * d * n / (n^2 - 1)) :=
  by sorry

end perimeter_bound_l153_153472


namespace volume_of_sphere_inscribed_in_cube_of_edge_8_l153_153645

noncomputable def volume_of_inscribed_sphere (edge_length : ℝ) : ℝ := 
  (4 / 3) * Real.pi * (edge_length / 2) ^ 3

theorem volume_of_sphere_inscribed_in_cube_of_edge_8 :
  volume_of_inscribed_sphere 8 = (256 / 3) * Real.pi :=
by
  sorry

end volume_of_sphere_inscribed_in_cube_of_edge_8_l153_153645


namespace total_distance_covered_l153_153316

noncomputable def radius : ℝ := 0.242
noncomputable def circumference : ℝ := 2 * Real.pi * radius
noncomputable def number_of_revolutions : ℕ := 500
noncomputable def total_distance : ℝ := circumference * number_of_revolutions

theorem total_distance_covered :
  total_distance = 760 :=
by
  -- sorry Re-enable this line for the solver to automatically skip the proof 
  sorry

end total_distance_covered_l153_153316


namespace eccentricity_range_l153_153202

variable {a b c e : ℝ}

-- Conditions given in the problem as assumptions:
variable (h1 : a > b > 0)
variable (h2 : e = c / a)
variable (h3 : b^2 = a^2 - c^2)

-- The proof statement:
theorem eccentricity_range
  (h4 : b < b/2 + c)
  (h5 : b/2 + c < a) :
  sqrt(5)/5 < e ∧ e < 3/5 :=
sorry

end eccentricity_range_l153_153202


namespace max_difference_exists_l153_153662

theorem max_difference_exists :
  ∃ a b : ℕ, 
    (100000 ≤ a ∧ a < 1000000) ∧ (100000 ≤ b ∧ b < 1000000) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 a)) → d % 2 = 0) ∧ 
    (∀ d, d ∈ (List.ofFn (Nat.digits 10 b)) → d % 2 = 0) ∧ 
    (∃ n, a < n ∧ n < b ∧ (∃ d, d ∈ (List.ofFn (Nat.digits 10 n)) ∧ d % 2 = 1)) ∧ 
    (b - a = 111112) := 
sorry

end max_difference_exists_l153_153662


namespace leah_earned_initially_l153_153479

noncomputable def initial_money (x : ℝ) : Prop :=
  let amount_after_milkshake := (6 / 7) * x
  let amount_left_wallet := (3 / 7) * x
  amount_left_wallet = 12

theorem leah_earned_initially (x : ℝ) (h : initial_money x) : x = 28 :=
by
  sorry

end leah_earned_initially_l153_153479


namespace pascal_triangle_contains_47_l153_153061

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153061


namespace constant_term_in_expansion_l153_153447

-- Define the terms involved in the problem.
def term1 (x : ℝ) : ℝ := x - real.sqrt x
def term2 (x : ℝ) : ℝ := real.cbrt x + x^(-2)

-- The main theorem to be stated.
theorem constant_term_in_expansion (n : ℕ) (x : ℝ) : 
  ∃ (n : ℕ), 
  (∀ x : ℝ, (term1 x)^2 * (term2 x)^n = c) → n = 4 :=
sorry

end constant_term_in_expansion_l153_153447


namespace ivan_compensation_l153_153855

noncomputable def initial_deposit : ℝ := 100000
noncomputable def insurance_limit : ℝ := 1400000
noncomputable def insured_event := true
noncomputable def within_limit (d : ℝ) : Prop := d ≤ insurance_limit

theorem ivan_compensation (d : ℝ) (h₁ : d = initial_deposit) (h₂ : insured_event) (h₃ : within_limit d) : 
  ∃ c : ℝ, c = d + accrued_interest d :=
sorry

end ivan_compensation_l153_153855


namespace find_n_l153_153215

def digit_sum (n : ℕ) : ℕ :=
-- This function needs a proper definition for the digit sum, we leave it as sorry for this example.
sorry

def num_sevens (n : ℕ) : ℕ :=
7 * (10^n - 1) / 9

def product (n : ℕ) : ℕ :=
8 * num_sevens n

theorem find_n (n : ℕ) : digit_sum (product n) = 800 ↔ n = 788 :=
sorry

end find_n_l153_153215


namespace daps_equivalent_to_dips_l153_153101

theorem daps_equivalent_to_dips (daps dops dips : ℕ) 
  (h1 : 4 * daps = 3 * dops) 
  (h2 : 2 * dops = 7 * dips) :
  35 * dips = 20 * daps :=
by
  sorry

end daps_equivalent_to_dips_l153_153101


namespace correct_options_l153_153400

-- Definition of the ellipse and points
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1^2 / 4 + P.2^2 / 2 = 1)

def F1 : ℝ × ℝ := (-√2, 0)
def F2 : ℝ × ℝ := (√2, 0)
def M : ℝ × ℝ := (0, 2)

-- Calculate distance between points
def distance (A B : ℝ × ℝ) : ℝ :=
  ( (A.1 - B.1)^2 + (A.2 - B.2)^2 )^0.5

-- Conditions
axiom P_on_ellipse (P : ℝ × ℝ) : is_on_ellipse P

-- Proof statement
theorem correct_options :
  ∀ (P : ℝ × ℝ), is_on_ellipse P →
    (distance P F1 + distance P F2 = 4) ∧
    (∀ P : ℝ × ℝ, is_on_ellipse P → ∃ diff : ℝ, diff = 2*√2 ∧ distance P F1 - distance P F2 ≤ diff) ∧
    (¬∃ P : ℝ × ℝ, is_on_ellipse P ∧ ∠ (F1 - P) (F2 - P) = 120) ∧
    (∃ P : ℝ × ℝ, is_on_ellipse P ∧ distance M P = 2 + √2) :=
by
  sorry

end correct_options_l153_153400


namespace coeff_x2_correct_l153_153740

def coeff_x2_expr : ℤ :=
  let expr := 5 * (λ x : ℤ, x^2 - 2 * x^3) - 3 * (λ x : ℤ, 2 * x^2 - 3 * x + 4 * x^4) + 2 * (λ x : ℤ, 5 * x^3 - 3 * x^2)
  in (-7 : ℤ)

theorem coeff_x2_correct : coeff_x2_expr = (-7 : ℤ) :=
by
    -- Simple proof placeholder
    sorry


end coeff_x2_correct_l153_153740


namespace pascal_triangle_47_l153_153089

theorem pascal_triangle_47 (n : ℕ) (h_prime : Nat.prime 47) : 
  (∃ k : ℕ, k ≤ n ∧ binomial n k = 47) ↔ n = 47 :=
by
  sorry

end pascal_triangle_47_l153_153089


namespace max_rectangle_area_l153_153934

-- Definitions based on conditions
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ¬ is_prime n
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

theorem max_rectangle_area
  (l w : ℕ)
  (h_perim : perimeter l w = 50)
  (h_prime : is_prime l)
  (h_composite : is_composite w) :
  l * w = 156 :=
sorry

end max_rectangle_area_l153_153934


namespace pascal_row_contains_prime_47_l153_153050

theorem pascal_row_contains_prime_47 :
  ∃! (n : ℕ), ∃! (k : ℕ), 47 = Nat.choose n k := by
  sorry

end pascal_row_contains_prime_47_l153_153050


namespace probability_katie_greater_allie_l153_153858

theorem probability_katie_greater_allie :
  let die_faces := Set.univ : Set (Fin 6)
  let coin_faces := {0, 1} : Set (Fin 2)
  let head_value := 4
  let tail_value := 2
  let coin_value (c : Fin 2) : ℕ := if c = 0 then tail_value else head_value
  let allie_scores := {coin_value c1 * coin_value c2 | c1 c2 ∈ coin_faces }
  let katie_scores := {d1.val + d2.val | d1 d2 ∈ die_faces}
  let favorable_outcomes := (katie_scores.product katie_scores).card { p : ℕ × ℕ // p.1 > p.2 }
  let total_outcomes := die_faces.card * die_faces.card * coin_faces.card * coin_faces.card
  friendly_outcomes / total_outcomes = 25 / 72 :=
by sorry

end probability_katie_greater_allie_l153_153858


namespace line_eq_hyperbola_circle_l153_153759

theorem line_eq_hyperbola_circle :
  let hyperbola_eq := (λ x y : ℝ, x^2 / 3 - y^2 = 1)
  let circle_eq := (λ x y : ℝ, x^2 + (y + 2)^2 = 9)
  let right_focus := (2, 0)
  let circle_center := (0, -2)
  (∃ l : ℝ × ℝ → ℝ × ℝ → Prop, l right_focus circle_center → ∀ x y, l x y = x - y - 2) :=
begin
  sorry
end

end line_eq_hyperbola_circle_l153_153759


namespace compute_operation_l153_153757

def operation_and (x : ℝ) := 10 - x
def operation_and_prefix (x : ℝ) := x - 10

theorem compute_operation (x : ℝ) : operation_and_prefix (operation_and 15) = -15 :=
by
  sorry

end compute_operation_l153_153757


namespace ordering_of_a_b_c_l153_153001

noncomputable def a : ℝ := 2 ^ 0.6
noncomputable def b : ℝ := Real.log 0.6 -- Using Real.log for natural logarithm
noncomputable def c : ℝ := Real.log 0.4 -- Using Real.log for natural logarithm

theorem ordering_of_a_b_c : c < b ∧ b < a := by
  -- Prove that c < b < a
  sorry

end ordering_of_a_b_c_l153_153001


namespace probability_heads_at_least_9_l153_153967

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153967


namespace john_can_see_jane_for_45_minutes_l153_153476

theorem john_can_see_jane_for_45_minutes :
  ∀ (john_speed : ℝ) (jane_speed : ℝ) (initial_distance : ℝ) (final_distance : ℝ),
  john_speed = 7 →
  jane_speed = 3 →
  initial_distance = 1 →
  final_distance = 2 →
  (initial_distance / (john_speed - jane_speed) + final_distance / (john_speed - jane_speed)) * 60 = 45 :=
by
  intros john_speed jane_speed initial_distance final_distance
  sorry

end john_can_see_jane_for_45_minutes_l153_153476


namespace remainder_when_7n_divided_by_4_l153_153265

theorem remainder_when_7n_divided_by_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 := 
by
  sorry

end remainder_when_7n_divided_by_4_l153_153265


namespace selling_price_correct_l153_153644

noncomputable def cost_price : ℝ := 90.91

noncomputable def profit_rate : ℝ := 0.10

noncomputable def profit : ℝ := profit_rate * cost_price

noncomputable def selling_price : ℝ := cost_price + profit

theorem selling_price_correct : selling_price = 100.00 := by
  sorry

end selling_price_correct_l153_153644


namespace water_depth_in_upright_tank_height_8_2_l153_153639

def radius_of_cylinder (diameter : ℝ) : ℝ :=
diameter / 2

def circular_segment_area (angle : ℝ) (radius : ℝ) : ℝ :=
(angle / 360) * real.pi * radius^2 - 0.5 * radius^2 * real.sin (real.to_radians angle)

def total_circle_area (radius : ℝ) : ℝ :=
real.pi * radius^2

def water_fraction (segment_area : ℝ) (circle_area : ℝ) : ℝ :=
segment_area / circle_area

def water_depth_upright (fraction : ℝ) (height : ℝ) : ℝ :=
fraction * height

theorem water_depth_in_upright_tank_height_8_2 :
  let radius := radius_of_cylinder 10 in
  let segment_area := circular_segment_area 156.92 radius in
  let circle_area := total_circle_area radius in
  let fraction := water_fraction segment_area circle_area in
  water_depth_upright fraction 20 = 8.2 :=
sorry

end water_depth_in_upright_tank_height_8_2_l153_153639


namespace probability_condition_l153_153791

section

-- Definitions of the functions
def f1 (x : ℝ) : ℝ := -3 * x + 2
def f2 (x : ℝ) : ℝ := 1 / x
def f3 (x : ℝ) : ℝ := -x^2 + 1

-- Property that checks if a function y decreases as x increases when x < -1
def decreases_as_x_increases (f : ℝ → ℝ) : Prop :=
  ∀ ⦃x1 x2 : ℝ⦄, x1 < x2 → x1 < -1 → x2 < -1 → f x1 > f x2

-- Probability calculation statement
theorem probability_condition : 
  let functions := [f1, f2, f3] in
  (↑(functions.count (decreases_as_x_increases)) : ℝ) / functions.length = 2 / 3 := by
  sorry

end

end probability_condition_l153_153791


namespace max_diff_six_digit_even_numbers_l153_153671

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l153_153671


namespace vector_magnitude_square_magnitude_of_vectors_l153_153409

noncomputable def length_of_diagonal (s : ℝ) : ℝ :=
  Real.sqrt (s^2 + s^2)

theorem vector_magnitude_square (s : ℝ) (h : s = 1) :
  √((s * s) + (s * s)) = √2 :=
by
  rw [h, one_mul, one_mul, add_self_eq_mul_two, Real.sqrt_mul (by norm_num : 0 ≤ (2)) (by norm_num : 0 ≤ 1)],
  norm_num

theorem magnitude_of_vectors (s : ℝ) (h : s = 1) :
  let a := length_of_diagonal s in 2 * a = 2 * Real.sqrt 2 :=
by
  rw [h],
  have h1 := vector_magnitude_square s h,
  rw [length_of_diagonal, h1,
      Real.sqrt_two_mul_one],
  norm_num

end vector_magnitude_square_magnitude_of_vectors_l153_153409


namespace candle_lighting_time_l153_153955

theorem candle_lighting_time :
  ∀ (length : ℝ) (t : ℕ),
  (length / 3) * (3 - (144 / 60 + t / 60)) = 2 * (length / 4) * (4 - (144 / 60 + t / 60))
  → t = 36 :=
begin
  sorry
end

end candle_lighting_time_l153_153955


namespace prime_count_30_40_l153_153045

theorem prime_count_30_40 : 
  (finset.filter nat.prime {30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}).card = 2 := 
by
  sorry

end prime_count_30_40_l153_153045


namespace pascal_triangle_47_number_of_rows_containing_47_l153_153076

/-- Prove that the number 47 appears only once in Pascal's Triangle, specifically in the 47th row -/
theorem pascal_triangle_47 :
  ∀ n, 47 ≤ n →  ∃ k, binomial n k = 47 → n = 47 := 
begin
  sorry
end

-- Conclusion: The number 47 appears in exactly 1 row of Pascal's Triangle: the 47th row
theorem number_of_rows_containing_47 : 
  (Finset.filter (λ n : ℕ, ∃ k : ℕ, binomial n k = 47 ) (Finset.range 48)).card = 1 :=
begin
  sorry
end

end pascal_triangle_47_number_of_rows_containing_47_l153_153076


namespace probability_result_l153_153898

-- Conditions: definitions based on given problem
def dice_rolls_initial (triplet_value : ℕ) : ℕ := 7 -- Rolling 7 dice initially
def triplet (triplet_value : ℕ) := 3 -- Triplet value indicates 3 dice show the same number
def remaining_dice := 4 -- The remaining 4 dice to re-roll

-- Define the number of total outcomes for the re-rolled 4 dice
def total_outcomes : ℕ := 6^remaining_dice

-- Define the number of outcomes where none of the re-rolled dice match the triplet value
def non_matching_outcomes (triplet_value : ℕ) : ℕ := 5^remaining_dice

-- Define the number of outcomes where at least one of the re-rolled dice matches the triplet value
def at_least_one_matching (triplet_value : ℕ) : ℕ := total_outcomes - non_matching_outcomes triplet_value

-- Define the number of outcomes where all re-rolled dice are the same
def all_same_outcomes : ℕ := 6

-- Define the number of overlap outcomes for all dice showing the same number
def overlap (triplet_value : ℕ) : ℕ := 1

-- Define the number of successful outcomes
def successful_outcomes (triplet_value : ℕ) : ℕ := 
  at_least_one_matching triplet_value + all_same_outcomes - overlap triplet_value

-- Define the probability of at least four out of seven dice showing the same value
def probability (triplet_value : ℕ) : ℚ :=
  (successful_outcomes triplet_value : ℚ) / (total_outcomes : ℚ)

-- Proving the main theorem
theorem probability_result (triplet_value : ℕ) (h_triplet_valid : triplet_value ∈ {1, 2, 3, 4, 5, 6}) : 
  probability triplet_value = 169/324 :=
by
  sorry

end probability_result_l153_153898


namespace percentage_of_male_students_solved_l153_153123

variable (M F : ℝ)
variable (M_25 F_25 : ℝ)
variable (prob_less_25 : ℝ)

-- Conditions from the problem
def graduation_class_conditions (M F M_25 F_25 prob_less_25 : ℝ) : Prop :=
  M + F = 100 ∧
  M_25 = 0.50 * M ∧
  F_25 = 0.30 * F ∧
  (1 - 0.50) * M + (1 - 0.30) * F = prob_less_25 * 100

-- Theorem to prove
theorem percentage_of_male_students_solved (M F : ℝ) (M_25 F_25 prob_less_25 : ℝ) :
  graduation_class_conditions M F M_25 F_25 prob_less_25 → prob_less_25 = 0.62 → M = 40 :=
by
  sorry

end percentage_of_male_students_solved_l153_153123


namespace eight_digit_numbers_with_product_64827_l153_153745

theorem eight_digit_numbers_with_product_64827 : 
  -- Define the condition that the number has eight digits and their product is 64827
  ∃ (digits : Fin 8 → ℕ), 
    (∏ i, digits i) = 64827 ∧ 
    (∀ i, 0 < digits i ∧ digits i < 10) → 
    (number_of_such_numbers = 1120) :=
by
  sorry

end eight_digit_numbers_with_product_64827_l153_153745


namespace minimal_sum_dist_trapezoid_l153_153389

structure Trapezoid (A B C D P L : Type) :=
  (isosceles : A = D)
  (on_circle : A = P)
  (intersect_base : B = C)

variables {A B C D P L E F : Type}

theorem minimal_sum_dist_trapezoid (T : Trapezoid A B C D P L) :
  let E := midpoint A B, F := midpoint C D in
  ∀ Q : Type, Q ∈ line PL → 
  dist E Q + dist F Q ≤ dist E L + dist F L :=
sorry

end minimal_sum_dist_trapezoid_l153_153389


namespace max_diff_six_digit_even_numbers_l153_153675

-- Definitions for six-digit numbers with all digits even
def is_6_digit_even (n : ℕ) : Prop :=
  n >= 100000 ∧ n < 1000000 ∧ (∀ (d : ℕ), d < 6 → (n / 10^d) % 10 % 2 = 0)

def contains_odd_digit (n : ℕ) : Prop :=
  ∃ (d : ℕ), d < 6 ∧ (n / 10^d) % 10 % 2 = 1

-- The main theorem
theorem max_diff_six_digit_even_numbers (a b : ℕ) 
  (ha : is_6_digit_even a) 
  (hb : is_6_digit_even b)
  (h_cond : ∀ n : ℕ, a < n ∧ n < b → contains_odd_digit n) 
  : b - a = 111112 :=
sorry

end max_diff_six_digit_even_numbers_l153_153675


namespace total_sum_is_427_l153_153649

-- Let A, B, C be real numbers representing the shares of A, B, and C respectively
def share_A (C : ℝ) : ℝ := (7 * C) / 3
def share_B (C : ℝ) : ℝ := (7 * C) / 4
def share_C : ℝ := 83.99999999999999 -- C is given as approximately 84

-- Define the total sum to be divided
def total_sum (C : ℝ) : ℝ := share_A C + share_B C + C

-- Theorem stating that the sum of shares equals approximately 427
theorem total_sum_is_427 : total_sum share_C ≈ 427 := sorry

end total_sum_is_427_l153_153649


namespace valid_triple_count_l153_153839

-- Defining the structure of the problem
universe u
noncomputable theory

def Team := Fin 31 -- 31 teams

-- Dummy win-loss function for teams
def wins (t1 t2 : Team) : Prop := sorry -- Needs to be filled appropriately

-- Tournament condition definitions
def round_robin_tournament (teams : Set Team) : Prop :=
  ∀ (t : Team), 
    (∃ (win_set : Finset Team), win_set.card = 15 ∧ 
      (∀ (t' ∈ win_set), wins t t')) ∧
    (∃ (lose_set : Finset Team), lose_set.card = 15 ∧ 
      (∀ (t' ∈ lose_set), wins t' t))

-- The theorem to prove the number of valid triples
theorem valid_triple_count :
  (∃ (teams : Set Team), round_robin_tournament teams ∧ 
   ∑ t1 in teams.to_finset, ∑ t2 in teams.to_finset, ∑ t3 in teams.to_finset, 
     if wins t1 t2 ∧ wins t2 t3 ∧ wins t3 t1 then 1 else 0) = 1240 :=
-- Proof is skipped
sorry

end valid_triple_count_l153_153839


namespace probability_heads_at_least_9_l153_153968

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153968


namespace rectangle_perimeter_gt_16_l153_153009

theorem rectangle_perimeter_gt_16 (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0)
  (h_area_gt_perim : a * b > 2 * (a + b)) : 2 * (a + b) > 16 :=
by
  sorry

end rectangle_perimeter_gt_16_l153_153009


namespace min_value_func_l153_153170

/-- Let the function y = x^2 - 2x for x in [-2, a].
    Prove that the minimum value of the function is g(a) = a^2 - 2a -/
theorem min_value_func (a : ℝ) (h : ∀ x, x ∈ set.Icc (-2 : ℝ) a → x ^ 2 - 2 * x ≥ a ^ 2 - 2 * a) :
  ∀ x, x ∈ set.Icc (-2 : ℝ) a → x ^ 2 - 2 * x ≥ a ^ 2 - 2 * a :=
sorry

end min_value_func_l153_153170


namespace integral_along_parabola_eq_four_l153_153956

-- Definition of the vector field components
def P (x y : ℝ) : ℝ := 2 * x * y
def Q (x y : ℝ) : ℝ := x^2

-- The condition of the parabola
def parabola (x : ℝ) : ℝ := (x^2) / 4

-- The line integral calculation using the given conditions
theorem integral_along_parabola_eq_four :
  ∫ x in 0..2, P x (parabola x) * (D x) + Q x (parabola x) * (D (parabola x)) = 4 :=
by 
  sorry

end integral_along_parabola_eq_four_l153_153956


namespace ln_inequality_solution_set_l153_153937

theorem ln_inequality_solution_set (x e : ℝ) (h₁ : e > 0) : ln(x - e) < 1 ↔ e < x ∧ x < 2 * e := by
  sorry

end ln_inequality_solution_set_l153_153937


namespace monotonicity_range_of_a_sum_greater_than_two_l153_153375

noncomputable def f (a x : ℝ) : ℝ := log x - a * x + 1

theorem monotonicity (a : ℝ) (h : 0 < a) : 
  ∀ x : ℝ, 0 < x → (f a)' x = (1 / x) - a :=
sorry

theorem range_of_a (a : ℝ) (x1 x2 : ℝ) (h : x1 < x2) 
  (hx1 : f a x1 = 0) (hx2 : f a x2 = 0) : 
  0 < a ∧ a < 1 :=
sorry

theorem sum_greater_than_two (a x1 x2 : ℝ) 
  (h : 0 < a ∧ a < 1) (hx1 : f a x1 = 0) 
  (hx2 : f a x2 = 0) (hx1x2 : x1 < x2) : 
  x1 + x2 > 2 :=
sorry

end monotonicity_range_of_a_sum_greater_than_two_l153_153375


namespace matrix_multiplication_correct_l153_153719

def A : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![2, 0, -1], ![0, 3, -2], ![-2, 3, 2]]

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![1, -1, 0], ![2, 0, -1], ![3, 0, 0]]

theorem matrix_multiplication_correct :
  A.mul B = ![![ -1, -2, 0], ![0, 0, -3], ![10, 2, -3]] :=
  sorry

end matrix_multiplication_correct_l153_153719


namespace geometric_seq_sin_log_eq_l153_153457

variable {a : ℕ → ℝ}
variable (h_geometric_seq : ∀ n, a (n + 1) = a n * r) -- stating geometric sequence with common ratio r
variable (h_pos : ∀ n, a n > 0) -- all elements are positive
variable (h_condition : a 3 * a 4 * a 5 = 3 ^ Real.pi)

theorem geometric_seq_sin_log_eq : 
  sin (log 3 (a 1) + log 3 (a 2) + log 3 (a 3) + log 3 (a 4) + log 3 (a 5) + log 3 (a 6) + log 3 (a 7)) = (3.sqrt) / 2 :=
sorry

end geometric_seq_sin_log_eq_l153_153457


namespace lucille_cents_left_l153_153497

def lucille_earnings (weeds_flower_bed weeds_vegetable_patch weeds_grass : ℕ) (cent_per_weed soda_cost : ℕ) : ℕ :=
  let weeds_pulled := weeds_flower_bed + weeds_vegetable_patch + weeds_grass / 2 in
  let total_earnings := weeds_pulled * cent_per_weed in
  total_earnings - soda_cost

theorem lucille_cents_left :
  let weeds_flower_bed := 11
  let weeds_vegetable_patch := 14
  let weeds_grass := 32
  let cent_per_weed := 6
  let soda_cost := 99
  lucille_earnings weeds_flower_bed weeds_vegetable_patch weeds_grass cent_per_weed soda_cost = 147 := by
  sorry

end lucille_cents_left_l153_153497


namespace centerville_budget_l153_153946

/-- Define the annual budget. --/
def annual_budget (library_spending : Real) : Real :=
  library_spending / 0.15

/-- Define the spending on public parks. --/
def parks_spending (total_budget : Real) : Real :=
  0.24 * total_budget

/-- Calculate the amount left of the annual budget. --/
def amount_left (total_budget library_spending parks_spending : Real) : Real :=
  total_budget - (library_spending + parks_spending)

/-- The town of Centerville spends 15% of its annual budget on its public library. 
    Centerville spent $3,000 on its public library and 24% on the public parks. 
    We prove that the amount left of the annual budget is $12,200. --/
theorem centerville_budget (library_spending : Real) (h_library : library_spending = 3000) :
  amount_left (annual_budget library_spending) library_spending (parks_spending (annual_budget library_spending)) = 12200 :=
by
  -- Since no proof is required, we use sorry here.
  sorry

end centerville_budget_l153_153946


namespace solve_inequality_l153_153523

theorem solve_inequality (a : ℝ) : 
  (a > 0 → {x : ℝ | x < -a / 4 ∨ x > a / 3 } = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a = 0 → {x : ℝ | x ≠ 0} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) ∧ 
  (a < 0 → {x : ℝ | x < a / 3 ∨ x > -a / 4} = {x : ℝ | 12 * x^2 - a * x - a^2 > 0}) :=
sorry

end solve_inequality_l153_153523


namespace first_copy_machine_copies_per_minute_l153_153506

theorem first_copy_machine_copies_per_minute (x : ℕ) :
  (∀ (c1 c2 : ℕ), c2 = 10 → (40 * (c1 + c2) = 1000)) → x = 15 :=
by
  intro h
  have : 40 * (x + 10) = 1000 := h x 10 rfl
  have : (x + 10) = 25 := by
    rw [mul_comm] at this
    exact (nat.div_eq_of_eq_mul_left (by norm_num) this.symm).symm
  linarith

end first_copy_machine_copies_per_minute_l153_153506


namespace part_a_part_b_l153_153601

theorem part_a (x : ℝ) (n : ℕ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) (hn_pos : 0 < n) :
  Real.log x < n * (x ^ (1 / n) - 1) ∧ n * (x ^ (1 / n) - 1) < (x ^ (1 / n)) * Real.log x := sorry

theorem part_b (x : ℝ) (hx_pos : 0 < x) (hx_ne_one : x ≠ 1) :
  (Real.log x) = (Real.log x) := sorry

end part_a_part_b_l153_153601


namespace functional_equation_solution_l153_153737

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) = x * f x - y * f y) →
  ∃ m b : ℝ, ∀ t : ℝ, f t = m * t + b :=
by
  intro h
  sorry

end functional_equation_solution_l153_153737


namespace find_missing_number_l153_153736

theorem find_missing_number (some_number : ℕ) : 
  (∏ n in (finset.range 90).filter (λ n, n + 10 ≠ some_number), (1 - (1 / (n + 10 : ℝ))) * (1 - 1 / some_number)) = 0.09 ↔ some_number = 10 := 
sorry

end find_missing_number_l153_153736


namespace greatest_percentage_difference_month_is_may_l153_153431

def percentage_difference (a b c : ℕ) : ℚ :=
  ((max (max a b) c - min (min a b) c).toRat / (min (min a b) c).toRat) * 100

def sales_data := [ (5, 4, 6), -- January
                    (6, 4, 5), -- February
                    (5, 5, 5), -- March
                    (4, 6, 4), -- April
                    (3, 4, 7)  -- May
                  ]

noncomputable def highest_percentage_difference_month : ℕ :=
  sales_data.enum.map (λ (i, (d, b, f)), (i + 1, percentage_difference d b f))
                     .maxBy (λ x, x.snd)
                     |>.fst

theorem greatest_percentage_difference_month_is_may :
  highest_percentage_difference_month = 5 :=
sorry

end greatest_percentage_difference_month_is_may_l153_153431


namespace determinant_of_triangle_angles_zero_l153_153483

theorem determinant_of_triangle_angles_zero {A B C : ℝ} 
  (h : A + B + C = π) : 
  det ![
    [cos A ^ 2, tan A, 1],
    [cos B ^ 2, tan B, 1],
    [cos C ^ 2, tan C, 1]
  ] = 0 :=
sorry

end determinant_of_triangle_angles_zero_l153_153483


namespace tan_2pi_minus_alpha_l153_153394

noncomputable def log₈ (x : ℝ) : ℝ := Real.log x / Real.log 8

theorem tan_2pi_minus_alpha (α : ℝ) 
  (h1 : Real.cos ((3 / 2) * Real.pi + α) = log₈ (1 / 4))
  (h2 : α ∈ set.Ioo (-Real.pi / 2) 0) : 
  Real.tan (2 * Real.pi - α) = 2 * Real.sqrt 5 / 5 := 
by
  sorry

end tan_2pi_minus_alpha_l153_153394


namespace find_m_l153_153020

theorem find_m (m : ℕ) (h_pos : m > 0) (h1 : Nat.lcm 36 m = 108) (h2 : Nat.lcm 45 m = 180) : m = 72 := 
by 
  sorry

end find_m_l153_153020


namespace victor_final_usd_l153_153567

variable (initial_rubles : ℝ) 
variable (term_years : ℕ) 
variable (annual_rate : ℝ) 
variable (buy_rate : ℝ) 
variable (sell_rate : ℝ)

def rubles_to_usd (rubles : ℝ) (rate : ℝ) : ℝ :=
  rubles / rate

def compound (P : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  P * ((1 + r / n) ^ (n * t))

theorem victor_final_usd :
  initial_rubles = 45000 →
  term_years = 2 →
  annual_rate = 0.047 →
  buy_rate = 59.60 →
  sell_rate = 56.60 →
  let P := rubles_to_usd initial_rubles sell_rate in
  round (compound P annual_rate 4 term_years) = 874 :=
by
  intros h1 h2 h3 h4 h5
  let P := rubles_to_usd initial_rubles sell_rate
  sorry

end victor_final_usd_l153_153567


namespace infinite_sum_computation_l153_153342

theorem infinite_sum_computation : 
  ∑' n : ℕ, (3 * (n + 1) + 2) / (n * (n + 1) * (n + 3)) = 10 / 3 :=
by sorry

end infinite_sum_computation_l153_153342


namespace phillip_math_percentage_right_l153_153187

theorem phillip_math_percentage_right (total_math_questions : ℕ) (total_english_questions : ℕ)
  (english_percentage_right : ℕ) (total_questions_right : ℕ) :
  total_math_questions = 40 → total_english_questions = 50 → english_percentage_right = 98 →
  total_questions_right = 79 → 
  ((total_questions_right - (english_percentage_right * total_english_questions / 100)) * 100 / total_math_questions = 75) :=
by
  intros h1 h2 h3 h4
  simp [h1, h2, h3, h4]
  sorry

end phillip_math_percentage_right_l153_153187


namespace min_value_of_z_l153_153396

theorem min_value_of_z (a x y : ℝ) (h1 : a > 0) (h2 : x ≥ 1) (h3 : x + y ≤ 3) (h4 : y ≥ a * (x - 3)) :
  (∃ (x y : ℝ), 2 * x + y = 1) → a = 1 / 2 :=
by {
  sorry
}

end min_value_of_z_l153_153396


namespace sin_expression_eq_sqrt_expression_l153_153614

theorem sin_expression_eq_sqrt_expression (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, a i ∈ {-1, 1}) :
  2 * sin ((∑ i in Finset.range n, ((Finset.prod (Finset.range (i + 1)) (λ j, a ⟨j, sorry⟩)) / 2^i) * (π / 4))) = 
  a 0 * sqrt (2 + (Finset.foldr (λ a b, a * sqrt (2 + b)) 0 (Finset.range (n - 1)))) := sorry

end sin_expression_eq_sqrt_expression_l153_153614


namespace friend_c_spent_26_l153_153596

theorem friend_c_spent_26 :
  let you_spent := 12
  let friend_a_spent := you_spent + 4
  let friend_b_spent := friend_a_spent - 3
  let friend_c_spent := friend_b_spent * 2
  friend_c_spent = 26 :=
by
  sorry

end friend_c_spent_26_l153_153596


namespace calculate_expression_l153_153332

theorem calculate_expression : 
  (3^1 + 3^0 + 3^(-1)) / (3^(-2) + 3^(-3) + 3^(-4)) = 27 := 
  sorry

end calculate_expression_l153_153332


namespace isosceles_triangle_angles_l153_153701

noncomputable def is_acos(x : ℝ) := x >= 0 ∧ x <= 1

theorem isosceles_triangle_angles (x : ℝ) (hx : ∃ t : ℝ, t > 0 ∧ t < 90 ∧ x = t * (π / 180)) 
  (h_sinx : is_acos (sin x)) (h_sin5x : is_acos (sin (5 * x))) :
  (sin x = sin (15 * (π / 180)) ∨ sin x = sin (45 * (π / 180))) :=
sorry

end isosceles_triangle_angles_l153_153701


namespace sample_mean_variance_l153_153785

variable (x1 x2 x3 : ℝ)
variable (x̄ : ℝ)
variable (σ² : ℝ)

theorem sample_mean_variance :
  (x̄ = 40) →
  (σ² = 1) →
  (x̄ = (x1 + x2 + x3) / 3) →
  let y1 := x1 + x̄ in
  let y2 := x2 + x̄ in
  let y3 := x3 + x̄ in
  ((y1 + y2 + y3) / 3 = 80) ∧ (σ² = 1) :=
by
  sorry

end sample_mean_variance_l153_153785


namespace max_digitally_symmetric_plates_l153_153840

theorem max_digitally_symmetric_plates : 
  let digitally_symmetric (plate : List ℕ) :=
    (plate.length = 5) ∧ (plate.head = some 8 ∨ plate.head = some 9) ∧ 
    (plate.nth 1 = plate.nth 3) ∧ (plate.nth 2 = plate.nth 2)
  in
  (∃ plates : List (List ℕ), ∀ plate ∈ plates, digitally_symmetric plate ∧ plates.length = 200) :=
sorry

end max_digitally_symmetric_plates_l153_153840


namespace probability_at_least_9_heads_l153_153979

theorem probability_at_least_9_heads (n k : ℕ) (hn : n = 12) (hfair : k = 2^12) : 
  ∑ i in finset.range 4, nat.choose 12 (i + 9) = 299 → (299 : ℚ) / 4096 = 299 / 4096 := 
by
  -- Total outcomes
  have h_total : 2^12 = k := by
    sorry

  -- Favorable outcomes
  have h_heads_9 : nat.choose 12 9 = 220 := by
    sorry

  have h_heads_10 : nat.choose 12 10 = 66 := by
    sorry

  have h_heads_11 : nat.choose 12 11 = 12 := by
    sorry
  
  have h_heads_12 : nat.choose 12 12 = 1 := by
    sorry

  have h_sum : 220 + 66 + 12 + 1 = 299 := by
    sorry

  intro h_proof
  rw [h_proof]
  norm_num
  rw ←rat.cast_add
  rw ←rat.cast_mul
  rw rat.cast_inj
  sorry

end probability_at_least_9_heads_l153_153979


namespace area_of_triangle_ABC_l153_153718

noncomputable theory

def Point := (ℝ × ℝ)

variables (A B C : Point)

-- Conditions on the distances between the centers of the circles
def distance (p1 p2 : Point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

-- Coordinates derived from the problem
variable (hA : A = (5, 2))
variable (hB : B = (0, 3))
variable (hC : C = (0, -4))

-- Tangency conditions and relations between the circles' distances
variable (hAB : distance A B = 5)
variable (hBC : distance B C = 7)

-- The area calculation for the triangle given the coordinates
def triangle_area (p1 p2 p3 : Point) : ℝ :=
  0.5 * abs (p1.1 * (p2.2 - p3.2) + p2.1 * (p3.2 - p1.2) + p3.1 * (p1.2 - p2.2))

theorem area_of_triangle_ABC :
    triangle_area A B C = 17.5 :=
  sorry

end area_of_triangle_ABC_l153_153718


namespace congruent_triangles_l153_153459

-- Definitions representing the sides of the triangles
variables {x y : ℝ}

-- Conditions: The triangles are congruent, and the sides correspond as described
def triangle1 := (2, 5, x)
def triangle2 := (y, 2, 6)

-- Theorem stating that under these conditions, x + y = 11
theorem congruent_triangles (congruent : triangle1 = triangle2) : x + y = 11 := 
sorry

end congruent_triangles_l153_153459


namespace probability_heads_at_least_9_l153_153972

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153972


namespace pascal_triangle_contains_47_l153_153062

theorem pascal_triangle_contains_47 :
  ∃! n : ℕ, ∃ k : ℕ, pascal n k = 47 ∧ n = 47 := 
sorry

end pascal_triangle_contains_47_l153_153062


namespace probability_heads_ge_9_in_12_flips_is_correct_l153_153985

/- Define the probability of getting at least 9 heads in 12 fair coin flips -/
noncomputable def prob_heads_ge_9_if_flipped_12_times : ℚ :=
  ((Nat.choose 12 9) + (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)) / (2 ^ 12)

/- State the theorem -/
theorem probability_heads_ge_9_in_12_flips_is_correct :
  prob_heads_ge_9_if_flipped_12_times = 299 / 4096 :=
sorry

end probability_heads_ge_9_in_12_flips_is_correct_l153_153985


namespace probability_heads_at_least_9_l153_153966

open Nat

noncomputable def num_outcomes : ℕ := 2 ^ 12

noncomputable def binom : ℕ → ℕ → ℕ := Nat.choose

noncomputable def favorable_outcomes : ℕ := binom 12 9 + binom 12 10 + binom 12 11 + binom 12 12

noncomputable def probability_of_at_least_9_heads : ℚ := favorable_outcomes / num_outcomes

theorem probability_heads_at_least_9 : probability_of_at_least_9_heads = 299 / 4096 := by
  sorry

end probability_heads_at_least_9_l153_153966


namespace valid_group_count_l153_153557

theorem valid_group_count (n : ℕ) (h_n : n = 4043) :
  (Nat.choose n 3) - n * (Nat.choose 2021 2) = 
  (Nat.choose 4043 3) - 4043 * (Nat.choose 2021 2) :=
by
  rw h_n
  simp
  sorry

end valid_group_count_l153_153557


namespace find_a_l153_153823

theorem find_a (a : ℝ) (h : 3 ∈ {a, a^2 - 2 * a}) : a = -1 :=
by
  sorry

end find_a_l153_153823


namespace sum_of_digits_of_largest_n_is_16_l153_153164

-- Define the primes less than 20
def primes_less_than_20 : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19]

-- Define the condition that a number is prime
def is_prime (p : ℕ) : Prop := p.prime

-- Fix variables for the two distinct primes less than 20
variables (d e : ℕ)
(Hd : d ∈ primes_less_than_20)
(He : e ∈ primes_less_than_20)
(hde : d ≠ e)

-- Define the number n
def n : ℕ := d * e * (e^2 + 10 * d)

-- Define the largest_n as the maximum n satisfying the conditions
def largest_n : ℕ :=
  max (list.map (λ d, max (list.map (λ e, if is_prime (e^2 + 10 * d) then d * e * (e^2 + 10 * d) else 0) primes_less_than_20)) primes_less_than_20).maximum

-- Define the sum of the digits function
def sum_digits (n : ℕ) : ℕ := (n.to_string.data.map (λ c, c.to_nat - '0'.to_nat)).sum

-- The theorem statement
theorem sum_of_digits_of_largest_n_is_16 :
  sum_digits largest_n = 16 :=
by sorry

end sum_of_digits_of_largest_n_is_16_l153_153164


namespace remainder_7n_mod_4_l153_153254

theorem remainder_7n_mod_4 (n : ℤ) (h : n % 4 = 3) : (7 * n) % 4 = 1 :=
by
  sorry

end remainder_7n_mod_4_l153_153254
