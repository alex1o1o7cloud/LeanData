import Mathlib

namespace XY_equidistant_from_M_l662_662582

open Geometry

-- Assume the points and lines in the problem
variable {A B C H A_1 C_1 X Y M : Point}

-- Conditions given in the problem
variable (triangleABC : Triangle A B C)
variable (altitudeA : Line A A_1)
variable (altitudeC : Line C C_1)
variable (H_is_intersect : Intersection altitudeA altitudeC H)
variable (line_parallel_A1C1 : Line H parallelTo Line A_1 C_1)
variable (circumcircleAHC1 : Circle A H C_1)
variable (circumcircleCHA1 : Circle C H A_1)
variable (X_intersect_circleAHC1 : Intersection circumcircleAHC1 line_parallel_A1C1 X)
variable (Y_intersect_circleCHA1 : Intersection circumcircleCHA1 line_parallel_A1C1 Y)

-- Midpoint definition
variable (M_is_midpoint : Midpoint M B H)

-- The theorem to prove
theorem XY_equidistant_from_M :
  dist M X = dist M Y :=
sorry

end XY_equidistant_from_M_l662_662582


namespace trigonometric_identity_l662_662699

theorem trigonometric_identity :
  cos (-17 / 4 * Real.pi) - sin (-17 / 4 * Real.pi) = Real.sqrt 2 := 
by
  sorry

end trigonometric_identity_l662_662699


namespace normal_pdf_and_cdf_l662_662276

noncomputable def M (X : Type) := 3
noncomputable def D (X : Type) := 4
noncomputable def σ (X : Type) := Real.sqrt (D X)

noncomputable def normal_pdf (x : ℝ) (μ σ : ℝ) : ℝ :=
  (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * σ^2))

noncomputable def normal_cdf (x : ℝ) (μ σ : ℝ) : ℝ :=
  (1 / 2) + (Mathlib.ℕ.pos.inv 2 * Mathlib.ℕ.sub 3 μ)

theorem normal_pdf_and_cdf :
  ∀ (x : ℝ), normal_pdf x 3 2 = (1 / (2 * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - 3)^2 / 8) ∧
             normal_cdf x 3 2 = (1 / 2) + (Mathlib.ℕ.pos.inv (2 * (Mathlib.ℕ.sub 3 (M X))))
:= by
  sorry

end normal_pdf_and_cdf_l662_662276


namespace Lizzy_total_after_loan_returns_l662_662913

theorem Lizzy_total_after_loan_returns : 
  let initial_amount := 50
  let alice_loan := 25 
  let alice_interest_rate := 0.15
  let bob_loan := 20
  let bob_interest_rate := 0.20
  let alice_interest := alice_loan * alice_interest_rate
  let bob_interest := bob_loan * bob_interest_rate
  let total_alice := alice_loan + alice_interest
  let total_bob := bob_loan + bob_interest
  let total_amount := total_alice + total_bob
  total_amount = 52.75 :=
by
  sorry

end Lizzy_total_after_loan_returns_l662_662913


namespace complex_magnitude_problem_l662_662545

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l662_662545


namespace cyclic_quad_unknown_side_l662_662569

noncomputable def area (a b c s: ℝ) : ℝ :=
  0.5 * a * b * (Real.sin c)

theorem cyclic_quad_unknown_side (AB BC CD AC : ℝ) (H: AB * BC = AC * CD) :
  AB = 5 ∨ AB = 8 ∨ AB = 10 ∧
  BC = 5 ∨ BC = 8 ∨ BC = 10 ∧
  CD = 5 ∨ CD = 8 ∨ CD = 10 →
  AC = 4 ∨ AC = 6.25 ∨ AC = 16 := 
by
  intros h1
  rw [area]
  sorry

end cyclic_quad_unknown_side_l662_662569


namespace probability_king_of_diamonds_top_two_l662_662312

-- Definitions based on the conditions
def total_cards : ℕ := 54
def king_of_diamonds : ℕ := 1
def jokers : ℕ := 2

-- The main theorem statement proving the probability
theorem probability_king_of_diamonds_top_two :
  let prob := (king_of_diamonds / total_cards) + ((total_cards - 1) / total_cards * king_of_diamonds / (total_cards - 1))
  prob = 1 / 27 :=
by
  sorry

end probability_king_of_diamonds_top_two_l662_662312


namespace total_dolls_l662_662443

def sisters_dolls : ℝ := 8.5

def hannahs_dolls : ℝ := 5.5 * sisters_dolls

theorem total_dolls : hannahs_dolls + sisters_dolls = 55.25 :=
by
  -- Proof is omitted
  sorry

end total_dolls_l662_662443


namespace initial_total_packs_l662_662496

def initial_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  total_packs = regular_packs + unusual_packs + excellent_packs

def ratio_packs (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) : Prop :=
  3 * (regular_packs + unusual_packs + excellent_packs) = 3 * regular_packs + 4 * unusual_packs + 6 * excellent_packs

def new_ratios (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  2 * (new_regular_packs) + 5 * (new_unusual_packs) + 8 * (new_excellent_packs) = regular_packs + unusual_packs + excellent_packs + 8 * (regular_packs)

def pack_changes (initial_regular_packs : ℕ) (initial_unusual_packs : ℕ) (initial_excellent_packs : ℕ) (new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) : Prop :=
  initial_excellent_packs <= new_excellent_packs + 80 ∧ initial_regular_packs - new_regular_packs ≤ 10

theorem initial_total_packs (total_packs : ℕ) (regular_packs : ℕ) (unusual_packs : ℕ) (excellent_packs : ℕ) 
(new_regular_packs : ℕ) (new_unusual_packs : ℕ) (new_excellent_packs : ℕ) :
  initial_packs total_packs regular_packs unusual_packs excellent_packs ∧
  ratio_packs regular_packs unusual_packs excellent_packs ∧ 
  new_ratios regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs ∧ 
  pack_changes regular_packs unusual_packs excellent_packs new_regular_packs new_unusual_packs new_excellent_packs 
  → total_packs = 260 := 
sorry

end initial_total_packs_l662_662496


namespace museum_earnings_from_nyc_college_students_l662_662851

def visitors := 200
def nyc_residents_fraction := 1 / 2
def college_students_fraction := 0.30
def ticket_price := 4

theorem museum_earnings_from_nyc_college_students : 
  ((visitors * nyc_residents_fraction * college_students_fraction) * ticket_price) = 120 := 
by 
  sorry

end museum_earnings_from_nyc_college_students_l662_662851


namespace cone_surface_area_l662_662425

theorem cone_surface_area {h : ℝ} {A_base : ℝ} (h_eq : h = 4) (A_base_eq : A_base = 9 * Real.pi) :
  let r := Real.sqrt (A_base / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  let lateral_area := Real.pi * r * l
  let total_surface_area := lateral_area + A_base
  total_surface_area = 24 * Real.pi :=
by
  sorry

end cone_surface_area_l662_662425


namespace min_sum_first_n_terms_l662_662104

noncomputable def Sn (n : ℕ) : ℝ :=
  let a1 := -11 -- Derived from the solution steps where a1 = -11
  let d := 2
  let a := λ n, a1 + (n - 1) * d
  (n * (a 1 + a n)) / 2

theorem min_sum_first_n_terms : ∀ (n : ℕ),
  Sn 6 ≤ Sn n := by
  sorry

end min_sum_first_n_terms_l662_662104


namespace magnitude_inverse_sum_eq_l662_662538

noncomputable def complex_magnitude_inverse_sum (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
|1/z + 1/w|

theorem magnitude_inverse_sum_eq (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_magnitude_inverse_sum z w hz hw hzw = 3 / 8 :=
by sorry

end magnitude_inverse_sum_eq_l662_662538


namespace lcm_of_primes_l662_662378

theorem lcm_of_primes :
  ¬ (is_prime 1223 ∧ is_prime 1399 ∧ is_prime 2687) →
  nat.lcm 1223 (nat.lcm 1399 2687) = 4583641741 :=
by
  intros h
  sorry

end lcm_of_primes_l662_662378


namespace probability_at_least_one_two_l662_662655

theorem probability_at_least_one_two (a b c : ℕ) (h₁ : a + b = c) 
  (h₂ : 1 ≤ a ∧ a ≤ 6) (h₃ : 1 ≤ b ∧ b ≤ 6) (h₄ : 1 ≤ c ∧ c ≤ 6) : 
  (∃ a b c, a + b = c ∧ 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 
    (a = 2 ∨ b = 2 ∨ c = 2)) 
  → (15 : ℚ) = 15 
  ∧ (8 : ℚ / 15) = 8 / 15 := sorry

end probability_at_least_one_two_l662_662655


namespace solve_equation_l662_662232

theorem solve_equation : ∀ x : ℝ, 2 * x - 4 * x = 0 → x = 0 := 
by
  assume x : ℝ
  assume h : 2 * x - 4 * x = 0
  sorry

end solve_equation_l662_662232


namespace common_number_in_sequence_l662_662846

theorem common_number_in_sequence 
  (a b c d e f g h i j : ℕ) 
  (h1 : (a + b + c + d + e) / 5 = 4) 
  (h2 : (f + g + h + i + j) / 5 = 9)
  (h3 : (a + b + c + d + e + f + g + h + i + j) / 10 = 7)
  (h4 : e = f) :
  e = 5 :=
by
  sorry

end common_number_in_sequence_l662_662846


namespace variance_calculation_l662_662304

noncomputable def factory_qualification_rate : ℝ := 0.98
noncomputable def number_of_pieces_selected : ℕ := 10
noncomputable def variance_of_qualified_products : ℝ := number_of_pieces_selected * factory_qualification_rate * (1 - factory_qualification_rate)

theorem variance_calculation : variance_of_qualified_products = 0.196 := by
  -- Proof omitted for brevity
  sorry

end variance_calculation_l662_662304


namespace base7_difference_l662_662714

theorem base7_difference :
  let x := 1 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 1 in
  let y := 6 * 7^2 + 5 * 7^1 + 2 * 1 in
  let diff := x - y in
  term_to_base7 diff = 2 * 7^2 + 5 * 7^1 + 2 * 1 :=
by
  let x := 1 * 7^3 + 2 * 7^2 + 3 * 7^1 + 4 * 1
  let y := 6 * 7^2 + 5 * 7^1 + 2 * 1
  let diff := x - y
  let term_to_base7 := λ (n : ℕ), 2 * 7^2 + 5 * 7^1 + 2 * 1 -- This function would convert from decimal to base 7
  sorry

end base7_difference_l662_662714


namespace min_value_f_l662_662980

def f (x : ℝ) : ℝ := 2 * Real.sin (2 * x + Real.pi / 6)

theorem min_value_f : 
  (∀ x ∈ Set.Icc (0 : ℝ) (Real.pi / 12), f(x) ≥ 1) ∧ (∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 12), f(x) = 1) :=
by
  sorry

end min_value_f_l662_662980


namespace prove_angle_A_prove_cos_pi_plus_B_l662_662500

-- Define the conditions as hypotheses
variable {A B C : ℝ}
variable {a b c : ℝ}

-- Mathematical equivalent proof problem
theorem prove_angle_A (h1 : c = a * real.cos B) : A = real.pi / 2 := 
sorry

theorem prove_cos_pi_plus_B (h1 : c = a * real.cos B) (h2 : real.sin C = 1 / 3) : real.cos (real.pi + B) = -1 / 3 := 
sorry

end prove_angle_A_prove_cos_pi_plus_B_l662_662500


namespace sum_of_squares_area_l662_662621

-- Define the right triangle BAE with the given properties
structure triangle (A B E : Type) :=
(angleEAB : angle A = 90°)
(BE_len : BE = 12)
(angleBAE : angle B = 30°)

-- Define squares ABCD and AEFG using the lengths from triangle BAE
noncomputable def area_square_ABCD (AB : ℝ) := AB ^ 2
noncomputable def area_square_AEFG (AE : ℝ) := AE ^ 2

-- Define the problem statement in Lean
theorem sum_of_squares_area (A B E : Type) [triangle A B E] : 
  ∃ (AB AE : ℝ), BE_len B E = 12 ∧ angleBAE B E = 30°
  ∧ area_square_ABCD AB + area_square_AEFG AE = 144 := 
sorry

end sum_of_squares_area_l662_662621


namespace prove_odd_function_l662_662049

theorem prove_odd_function (f : ℝ → ℝ) (h : ∀ x, f(x+1) + f(-x+1) = 2) : ∀ x, f(x+1) - 1 = - (f(-x+1) - 1) :=
by
  intro x
  sorry

end prove_odd_function_l662_662049


namespace area_minimum_triangle_BQC_l662_662497

noncomputable def cos_angle_bac (AB CA BC : ℝ) : ℝ :=
  (AB^2 + CA^2 - BC^2) / (2 * AB * CA)

def minimum_area_triangle_BQC (AB BC CA : ℝ) : ℝ :=
  let E := BC / 2 in -- E is the midpoint of BC for minimization
  let QE := E in   -- Distance from midpoint to line for minimum area
  0.5 * BC * QE

theorem area_minimum_triangle_BQC (AB BC CA : ℝ) (h_AB : AB = 12) (h_BC : BC = 15) (h_CA : CA = 17) :
  minimum_area_triangle_BQC AB BC CA = 56.25 :=
by
  sorry

end area_minimum_triangle_BQC_l662_662497


namespace intersecting_count_l662_662352

noncomputable def gcd (a b : ℕ) : ℕ := if b = 0 then a else gcd b (a % b)

theorem intersecting_count (m n : ℕ) :
  let gcd := gcd 803 608
  let slope := 608 / 803
  let line := λ x : ℚ, (608/803) * x
  -- conditions
  (gcd = 1) ∧
  (∀ x y, (x = 803 → y = line x) ∧ (y = 608)) →
  -- question and answer
  m + n = 1411 :=
sorry

end intersecting_count_l662_662352


namespace award_distribution_l662_662949

/-- Six different awards are to be given to four students. 
    Each student will receive at least one award.
    One specific student must receive at least two awards.
    Prove that the number of different ways to distribute the awards is exactly 3000. -/
theorem award_distribution : 
  (∃ f : fin 6 → fin 4, 
    (∀ (i : fin 4), 1 ≤ finset.card (finset.filter (λ x, f x = i) finset.univ)) ∧ 
    (∀ (j : fin 4), ∃ k : fin 6, f k = j → ∃ l : fin 6, f l = j ∧ k ≠ l)) ∧ 
  finset.card (finset.filter (λ g,  (all_f g ∧ specific_student_at_least_two g)) (finset.univ)): finset.card = 3000 :=
begin
  sorry
end

end award_distribution_l662_662949


namespace hyperbola_equation_eccentricity_directrix_l662_662060

theorem hyperbola_equation_eccentricity_directrix (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
  (eccentricity_hyperbola : c / a = sqrt 3)
  (directrix_parabola_passes_focus : -3/2 = -c)
  : (a = sqrt 3 ∧ b = sqrt (c^2 - a^2) ∧ c = 3) → ( ∃ a b : ℝ, (a = sqrt 3) ∧ (b = sqrt 6) ∧ (a > 0) ∧ (b > 0) ∧ (∀ x y : ℝ, (a ≠ 0 ∧ b ≠ 0) → (x ^ 2 / 3 - y ^ 2 / 6 = 1))) :=
by
  intros ⟨h4, h5, h6⟩
  use sqrt 3, sqrt 6
  simp
  split
  rintro rfl rfl
  sorry

end hyperbola_equation_eccentricity_directrix_l662_662060


namespace cos_value_in_second_quadrant_l662_662456

theorem cos_value_in_second_quadrant {B : ℝ} (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = - (12 / 13) :=
sorry

end cos_value_in_second_quadrant_l662_662456


namespace number_of_evens_from_60_to_80_l662_662835

variable (y : ℕ)

def is_even (n : ℕ) : Prop := n % 2 = 0

def is_in_range (n : ℕ) : Prop := 60 ≤ n ∧ n ≤ 80

def evens_from_60_to_80 : Finset ℕ := (Finset.filter (λ n, is_even n ∧ is_in_range n) (Finset.range 81))

theorem number_of_evens_from_60_to_80 (hy : y = evens_from_60_to_80.card) : y = 11 :=
by
  sorry

end number_of_evens_from_60_to_80_l662_662835


namespace remainder_when_dividing_150_l662_662388

theorem remainder_when_dividing_150 (k : ℕ) (hk1 : k > 0) (hk2 : 80 % k^2 = 8) : 150 % k = 6 :=
by
  sorry

end remainder_when_dividing_150_l662_662388


namespace complex_magnitude_theorem_l662_662535

theorem complex_magnitude_theorem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  ∣(1 / z) + (1 / w)∣ = 3 / 8 := by
  sorry

end complex_magnitude_theorem_l662_662535


namespace science_major_men_percentage_l662_662617

variable (total_students : ℕ) (women_percentage : ℕ) (science_major_women_percentage : ℕ)
          (nonscience_major_percentage : ℕ) (men_percentage : ℕ)

-- Assumption
axiom h1 : women_percentage = 100 - men_percentage
axiom h2 : science_major_women_percentage = 30
axiom h3 : nonscience_major_percentage = 60
axiom h4 : men_percentage = 40

theorem science_major_men_percentage 
  (h1 : women_percentage = 100 - men_percentage)
  (h2 : science_major_women_percentage = 30)
  (h3 : nonscience_major_percentage = 60)
  (h4 : men_percentage = 40) :
  25 ∗ total_students = (total_students * science_major_women_percentage / 30) / men_percentage :=
sorry

end science_major_men_percentage_l662_662617


namespace ratio_divisor_to_remainder_l662_662841

theorem ratio_divisor_to_remainder (R D Q : ℕ) (hR : R = 46) (hD : D = 10 * Q) (hdvd : 5290 = D * Q + R) :
  D / R = 5 :=
by
  sorry

end ratio_divisor_to_remainder_l662_662841


namespace find_pairs_l662_662709

def isDivisible (m n : ℕ) : Prop := ∃ k : ℕ, m = k * n
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def satisfiesConditions (a b : ℕ) : Prop :=
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  isPrime (a + 6 * b + 2)) ∨
  (isDivisible (a^2 + 6 * a + 8) b ∧
  (a^2 + a * b - 6 * b^2 - 15 * b - 9 = 0) ∧
  ¬ (a + 2 * b + 2) % 4 = 0 ∧
  ¬ isPrime (a + 6 * b + 2))

theorem find_pairs (a b : ℕ) :
  (a = 5 ∧ b = 1) ∨ 
  (a = 17 ∧ b = 7) → 
  satisfiesConditions a b :=
by
  -- Proof to be completed
  sorry

end find_pairs_l662_662709


namespace sufficient_condition_decreasing_necessary_condition_decreasing_main_theorem_l662_662736

def function_f (a x : ℝ) : ℝ := a * x^2 + 2 * (a - 1) * x + 2
def derivative_f (a x : ℝ) : ℝ := 2 * a * x + 2 * (a - 1)

theorem sufficient_condition_decreasing (a : ℝ) (h : 0 < a ∧ a ≤ 1 / 5) :
  ∀ x ≤ 4, derivative_f a x ≤ 0 :=
sorry

theorem necessary_condition_decreasing :
  (∃ a : ℝ, (0 < a ∧ a ≤ 1 / 5) ∧ ∀ x ≤ 4, derivative_f a x ≤ 0) →
  ∀ a : ℝ, ∀ x ≤ 4, derivative_f a x ≤ 0 :=
sorry

theorem main_theorem :
  (∀ a, (0 < a ∧ a ≤ 1 / 5) → ∀ x ≤ 4, derivative_f a x ≤ 0) →
  (A : Prop)
  (A = "Sufficient but not necessary condition" ∧ ¬ (∀ a, ∀ x ≤ 4, derivative_f a x ≤ 0))
:=
sorry

end sufficient_condition_decreasing_necessary_condition_decreasing_main_theorem_l662_662736


namespace solve_for_m_l662_662394

theorem solve_for_m (m n : ℤ) (h1 : 8 * (2^m)^n = 64) (h2 : |n| = 1) : m = 3 ∨ m = -3 :=
by
  sorry

end solve_for_m_l662_662394


namespace fred_games_last_year_l662_662392

def total_games : Nat := 47
def games_this_year : Nat := 36

def games_last_year (total games games this year : Nat) : Nat := total_games - games_this_year

theorem fred_games_last_year : games_last_year total_games games_this_year = 11 :=
by
  sorry

end fred_games_last_year_l662_662392


namespace parabola_focus_l662_662967

theorem parabola_focus (x y : ℝ) (p : ℝ) (h_eq : x^2 = 8 * y) (h_form : x^2 = 4 * p * y) : 
  p = 2 ∧ y = (x^2 / 8) ∧ (0, p) = (0, 2) :=
by
  sorry

end parabola_focus_l662_662967


namespace partition_Xn_into_disjoint_subsets_l662_662160

variable (n : ℕ) (X_n : Finset (Fin (n * (n + 1) / 2)))
variable (k : ℕ)

-- Basic conditions from the problem
def condition_1 : Prop := n > 3
def condition_2 : Prop := k = ⌊n * (n + 1) / 6⌋
def condition_3 : Prop := X_n.card = n * (n + 1) / 2
def condition_4 : Prop := 
  ∃ (B R W : Finset (Fin (n * (n + 1) / 2))),
  B.card = k ∧ R.card = k ∧ W.card = (X_n.card - 2 * k) ∧
  B ∪ R ∪ W = X_n

-- Declare the main theorem
theorem partition_Xn_into_disjoint_subsets 
  (h1 : condition_1 n) (h2 : condition_2 n k) (h3 : condition_3 n X_n) (h4 : condition_4 n X_n k) :
  ∃ (A : Fin n → Finset (X_n)),
  (∀ m, (A m).card = m + 1 ∧ (∃ c : Color, ∀ x ∈ A m, x.color = c)) ∧
  (¬∃ i j, i ≠ j ∧ (A i ∩ A j ≠ ∅)) :=
sorry

end partition_Xn_into_disjoint_subsets_l662_662160


namespace part_I_f_inequality_solution_set_part_II_max_value_ab_2bc_l662_662005

def f (x : ℝ) : ℝ := |x - 1| - |x + 1|

theorem part_I_f_inequality_solution_set :
  {x : ℝ | f(x) < |x| - 1} = {x : ℝ | x < -3} ∪ {x : ℝ | x > 1 / 3} :=
sorry

theorem part_II_max_value_ab_2bc (a b c : ℝ) (h : a^2 + 2c^2 + 3b^2 = 3 / 2) :
  ab + 2bc ≤ 3 / 4 :=
sorry

end part_I_f_inequality_solution_set_part_II_max_value_ab_2bc_l662_662005


namespace product_of_real_values_k_l662_662019

theorem product_of_real_values_k (k : ℝ) :
  (∀ x : ℝ, x ≠ 0 → (1 / (3 * x) = (k - 2 * x) / 5)) ↔
  (k = real.sqrt (40 / 3) ∨ k = -real.sqrt (40 / 3)) →
  (real.sqrt (40 / 3) * -real.sqrt (40 / 3) = -40 / 3) :=
by sorry

end product_of_real_values_k_l662_662019


namespace number_of_intersections_max_distance_line_to_curve_l662_662492

-- Definitions of given equations in polar coordinates
def curve_C (ρ θ : ℝ) : Prop :=
  ρ ^ 2 = 3 / (2 - cos(2 * θ))

def line_l (ρ θ : ℝ) : Prop :=
  ρ * cos(θ + π / 4) = sqrt 2

-- Number of intersection points of the line l and curve C
theorem number_of_intersections : ∃! (ρ θ : ℝ), curve_C ρ θ ∧ line_l ρ θ := by
  sorry

-- Maximum distance between a line m through a point on curve C parallel to line l and line l
theorem max_distance_line_to_curve : ∀ θ : ℝ, 
  let P := (sqrt 3 * cos θ, sin θ) in
  ∃ d : ℝ, 
    (d = abs ((2 * cos (θ + π / 6) - 2) / sqrt 2)) ∧
    (d = 2 * sqrt 2) := by
  sorry

end number_of_intersections_max_distance_line_to_curve_l662_662492


namespace rounding_4_36_to_nearest_tenth_l662_662189

theorem rounding_4_36_to_nearest_tenth : (round (10 * 4.36) / 10) = 4.4 :=
by
  sorry

end rounding_4_36_to_nearest_tenth_l662_662189


namespace find_projection_q_l662_662272

open Matrix

-- Define vector a and b
def vec_a : Matrix (Fin 2) (Fin 1) ℚ := ![![3], ![2]]
def vec_b : Matrix (Fin 2) (Fin 1) ℚ := ![![2], ![5]]

-- Define the resulting vector q
def vec_q : Matrix (Fin 2) (Fin 1) ℚ := ![![33 / 10], ![11 / 10]]

-- The proof statement
theorem find_projection_q : 
  ∃ (u : Matrix (Fin 2) (Fin 1) ℚ), 
  let q := 
    (Matrix.proj u (Matrix.vec_uniform 2 1 ξ) vec_a : Matrix (Fin 2) (Fin 1) ℚ) in
  q = vec_q ∧ 
  (Matrix.proj u (Matrix.vec_uniform 2 1 ξ) vec_b : Matrix (Fin 2) (Fin 1) ℚ) = vec_q := 
sorry

end find_projection_q_l662_662272


namespace students_need_to_walk_distance_l662_662581

-- Define distance variables and the relationships
def teacher_initial_distance : ℝ := 235
def xiao_ma_initial_distance : ℝ := 87
def xiao_lu_initial_distance : ℝ := 59
def xiao_zhou_initial_distance : ℝ := 26
def speed_ratio : ℝ := 1.5

-- Prove the distance x students need to walk
theorem students_need_to_walk_distance (x : ℝ) :
  teacher_initial_distance - speed_ratio * x =
  (xiao_ma_initial_distance - x) + (xiao_lu_initial_distance - x) + (xiao_zhou_initial_distance - x) →
  x = 42 :=
by
  sorry

end students_need_to_walk_distance_l662_662581


namespace integer_sequence_count_l662_662605

theorem integer_sequence_count (a₀ : ℕ) (step : ℕ → ℕ) (n : ℕ) 
  (h₀ : a₀ = 5184)
  (h_step : ∀ k, k < n → step k = (a₀ / 4^k))
  (h_stop : a₀ = (4 ^ (n - 1)) * 81) :
  n = 4 := 
sorry

end integer_sequence_count_l662_662605


namespace number_of_possible_ceil_values_l662_662807

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l662_662807


namespace number_of_vertices_with_odd_degree_even_l662_662936

open Finset

variable {V : Type} [Fintype V] (E : Finset (V × V))

def degree (v : V) : ℕ := (E.filter (λ e, e.1 = v ∨ e.2 = v)).card

noncomputable def vertices_with_odd_degree : Finset V :=
  E.to_finset.filter (λ v, degree E v % 2 = 1)

theorem number_of_vertices_with_odd_degree_even : 
  vertices_with_odd_degree E).card % 2 = 0 := sorry

end number_of_vertices_with_odd_degree_even_l662_662936


namespace equal_magnitude_necessary_condition_l662_662313

variable (u v : ℝ^3)

theorem equal_magnitude_necessary_condition (hu : u ≠ 0) (hv : v ≠ 0) (h : u = v) : ‖u‖ = ‖v‖ :=
by sorry

end equal_magnitude_necessary_condition_l662_662313


namespace total_chairs_taken_l662_662520

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l662_662520


namespace total_chairs_taken_l662_662513

theorem total_chairs_taken (students trips chairs_per_trip : ℕ) (h_students : students = 5) (h_trips : trips = 10) (h_chairs_per_trip : chairs_per_trip = 5) : 
  students * trips * chairs_per_trip = 250 := by
  rw [h_students, h_trips, h_chairs_per_trip]
  norm_num

end total_chairs_taken_l662_662513


namespace largest_A_when_quotient_remainder_equal_l662_662723

theorem largest_A_when_quotient_remainder_equal :
  ∃ A B C : ℕ, (A = 8 * B + C) ∧ (B = C) ∧ (0 ≤ C ∧ C < 8) ∧ (A = 63) :=
by {
  existsi (63 : ℕ),
  existsi (7 : ℕ),
  existsi (7 : ℕ),
  split,
  { reflexivity },
  { split,
    { reflexivity },
    { split,
      { split,
        { exact zero_le 7 },
        { linarith }},
      { reflexivity }}}
}

end largest_A_when_quotient_remainder_equal_l662_662723


namespace train_crossing_time_l662_662669

theorem train_crossing_time :
  ∀ (train_length bridge_length speed_kmh : ℕ),
    train_length = 100 →
    bridge_length = 300 →
    speed_kmh = 120 →
    (train_length + bridge_length) / ((speed_kmh * (1000 / 3600)) : ℝ) = 12 :=
by
  intros train_length bridge_length speed_kmh h_train_length h_bridge_length h_speed_kmh
  have h_dist : train_length + bridge_length = 400 := by
    rw [h_train_length, h_bridge_length]
    norm_num

  have h_speed : speed_kmh * (1000 / 3600 : ℝ) = 33.33 := by
    rw [h_speed_kmh]
    norm_num
    exact h_speed

  sorry

end train_crossing_time_l662_662669


namespace find_x_for_purely_imaginary_l662_662048

theorem find_x_for_purely_imaginary (x : ℝ) (h1 : x > 0) (h2 : (x - complex.I)^2.im = 0) : x = 1 := 
sorry

end find_x_for_purely_imaginary_l662_662048


namespace series_sum_gt_13_over_24_l662_662182

theorem series_sum_gt_13_over_24 (n : ℕ) (h : n > 2) : 
  (\sum_{i in finset.range (n+1) (2*n + 1)} (1 / i : ℝ)) > (13 / 24 : ℝ) :=
sorry

end series_sum_gt_13_over_24_l662_662182


namespace calculator_sum_is_3_l662_662843

-- Definitions of the problem variables and conditions
def initial_values := (2, 0, -2)
def num_participants := 60

-- A function implementing the operations described: 
-- 1. Taking the square root of the number on the calculator showing 2.
-- 2. Squaring the number on the calculator showing 0.
-- 3. Multiplying by -1 for the number on the calculator showing -2.
def operations (values : ℤ × ℤ × ℤ) : ℤ × ℤ × ℤ :=
  (int.sqrt values.1, values.2 ^ 2, (-1) * values.3)

-- A function to simulate the passing of calculators through all participants.
def pass_calculators (values : ℤ × ℤ × ℤ) (p : ℕ) : ℤ × ℤ × ℤ :=
  nat.iterate operations p values

-- Calculate the end result after all participants have performed the operations.
def end_result (values : ℤ × ℤ × ℤ) : ℤ :=
  let final_values := pass_calculators values num_participants
  in final_values.1 + final_values.2 + int_pow final_values.3 60

-- The theorem we need to prove
theorem calculator_sum_is_3 : end_result initial_values = 3 :=
  sorry

end calculator_sum_is_3_l662_662843


namespace average_speed_triathlon_l662_662465

theorem average_speed_triathlon :
  let swimming_distance := 1.5
  let biking_distance := 3
  let running_distance := 2
  let swimming_speed := 2
  let biking_speed := 25
  let running_speed := 8

  let t_s := swimming_distance / swimming_speed
  let t_b := biking_distance / biking_speed
  let t_r := running_distance / running_speed
  let total_time := t_s + t_b + t_r

  let total_distance := swimming_distance + biking_distance + running_distance
  let average_speed := total_distance / total_time

  average_speed = 5.8 :=
  by
    sorry

end average_speed_triathlon_l662_662465


namespace ball_probability_l662_662110

theorem ball_probability (n : ℕ) (h : (n : ℚ) / (n + 2) = 1 / 3) : n = 1 :=
sorry

end ball_probability_l662_662110


namespace find_angle_EBC_l662_662741

noncomputable def cyclic_quadrilateral (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E] :=
  ∃ (circle : Circle A), circle.contains B ∧ circle.contains C ∧ circle.contains D

theorem find_angle_EBC (A B C D E : Type) [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]
  (h1 : cyclic_quadrilateral A B C D E)
  (h2 : ∃ angle_BAD : ℝ, angle_BAD = 92)
  (h3 : ∃ angle_ADC : ℝ, angle_ADC = 68)
  :
  ∃ angle_EBC : ℝ, angle_EBC = 68
  := by
  sorry

end find_angle_EBC_l662_662741


namespace sin_pi_plus_alpha_l662_662044

open Real

-- Define the given conditions
variable (α : ℝ) (hα1 : sin (π / 2 + α) = 3 / 5) (hα2 : 0 < α ∧ α < π / 2)

-- The theorem statement that must be proved
theorem sin_pi_plus_alpha : sin (π + α) = -4 / 5 :=
by
  sorry

end sin_pi_plus_alpha_l662_662044


namespace f_2007_l662_662162

def A : Set ℚ := {x : ℚ | x ≠ 0 ∧ x ≠ 1}

noncomputable def f : A → ℝ := sorry

theorem f_2007 :
  (∀ x : ℚ, x ∈ A → f ⟨x, sorry⟩ + f ⟨1 - (1/x), sorry⟩ = Real.log (|x|)) →
  f ⟨2007, sorry⟩ = Real.log (|2007|) :=
sorry

end f_2007_l662_662162


namespace closest_fraction_to_medals_won_l662_662344

theorem closest_fraction_to_medals_won :
  let won_ratio : ℚ := 35 / 225
  let choices : List ℚ := [1/5, 1/6, 1/7, 1/8, 1/9]
  (closest : ℚ) = 1 / 6 → 
  (closest_in_choices : closest ∈ choices) →
  ∀ choice ∈ choices, abs ((7 / 45) - (1 / 6)) ≤ abs ((7 / 45) - choice) :=
by
  let won_ratio := 7 / 45
  let choices := [1/5, 1/6, 1/7, 1/8, 1/9]
  let closest := 1 / 6
  have closest_in_choices : closest ∈ choices := sorry
  intro choice h_choice_in_choices
  sorry

end closest_fraction_to_medals_won_l662_662344


namespace fraction_difference_l662_662451

theorem fraction_difference (x y : ℝ) (h : x - y = 3 * x * y) : (1 / x) - (1 / y) = -3 :=
by
  sorry

end fraction_difference_l662_662451


namespace divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l662_662600

theorem divisibility_of_3_pow_p_minus_2_pow_p_minus_1 (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) : 
  (3^p - 2^p - 1) % (42 * p) = 0 := 
by
  sorry

end divisibility_of_3_pow_p_minus_2_pow_p_minus_1_l662_662600


namespace angle_AED_45_l662_662107

-- Defining the given setup for triangle and points
variables (A B C E D : Type) [real_space : real.vector_space ℝ] [point : real.vector_space.point ℝ]
variables (line_segment : point → point → set point)
variables (∠ : point → point → point → ℝ)

-- Define the points in the problem
axiom A_on_AB : E ∈ line_segment A B
axiom D_on_BC : D ∈ line_segment B C
axiom BD_eq_DE_eq_EC : distance B D = distance D E ∧ distance D E = distance E C
axiom angle_BDE_90 : ∠ B D E = 90

-- The angle to be proven
theorem angle_AED_45 : ∠ A E D = 45 :=
sorry

end angle_AED_45_l662_662107


namespace hexagon_diagonals_intersect_at_point_l662_662180

-- Assuming necessary definitions for Hexagon and perpendicular properties
section Hexagon

variables (Q1 Q2 Q3 Q4 Q5 Q6 P : Point)
variables (q1 q2 q3 q4 q5 q6 : Line)
variables (R1 R2 R3 R4 R5 R6 O : Point)
variables (k k' : Circle)
variables (mid14 mid25 mid36 : Point)

-- Given conditions
def is_perpendicular_hexagon (Q1 Q2 Q3 Q4 Q5 Q6 : Point) : Prop :=
  are_perpendicular Q1 Q2 Q2 Q3 ∧
  are_perpendicular Q2 Q3 Q3 Q4 ∧
  are_perpendicular Q3 Q4 Q4 Q5 ∧
  are_perpendicular Q4 Q5 Q5 Q6 ∧
  are_perpendicular Q5 Q6 Q6 Q1 ∧
  are_perpendicular Q6 Q1 Q1 Q2

def perpendiculars_on_lines (Q1 Q2 Q3 Q4 Q5 Q6 P q1 q2 q3 q4 q5 q6 : Point × Point → Prop) : Prop :=
  are_perpendiculars [ (Q1, P, q1), (Q2, P, q2), (Q3, P, q3), (Q4, P, q4), (Q5, P, q5), (Q6, P, q6) ]

def intersections_on_circle (R1 R2 R3 R4 R5 R6 k : Point × Circle → Prop) : Prop :=
  on_circle R1 k ∧ on_circle R2 k ∧ on_circle R3 k ∧ on_circle R4 k ∧ on_circle R5 k ∧ on_circle R6 k

def diameters_of_circle (R1 R2 R3 R4 R5 R6 k : Point × Circle → Prop) : Prop :=
  is_diameter R1 R4 k ∧ is_diameter R2 R5 k ∧ is_diameter R3 R6 k

def center_and_midpoints_on_circle (O P k' : Point × Circle → Prop ∧ Point × Point → Prop) : Prop :=
  on_circle O k' ∧ on_circle (midpoint Q1 Q4) k' ∧ on_circle (midpoint Q2 Q5) k' ∧ on_circle (midpoint Q3 Q6) k' ∧ is_diameter O P k'

-- Lean statement for the proof
theorem hexagon_diagonals_intersect_at_point
  (h1 : is_perpendicular_hexagon Q1 Q2 Q3 Q4 Q5 Q6)
  (h2 : perpendiculars_on_lines Q1 Q2 Q3 Q4 Q5 Q6 P q1 q2 q3 q4 q5 q6)
  (h3 : intersections_on_circle R1 R2 R3 R4 R5 R6 k)
  (h4 : diameters_of_circle R1 R4 R2 R5 R3 R6 k)
  (h5 : center_and_midpoints_on_circle O P k' (midpoint Q1 Q4) (midpoint Q2 Q5) (midpoint Q3 Q6))
  : intersects_at_point [Q1 Q4, Q2 Q5, Q3 Q6] P :=
sorry

end Hexagon

end hexagon_diagonals_intersect_at_point_l662_662180


namespace red_houses_after_painters_l662_662243

theorem red_houses_after_painters : 
  let n := 100 in 
  let painters := 50 in 
  let red_painted (house : ℕ) : Bool := 
    let largest_odd_divisor := 
      List.range (house + 1) 
        |>.reverse
        |>.filter (fun k => k % 2 = 1 ∧ house % k = 0)
        |>.headD 1
    in 
    exists k, largest_odd_divisor = 4 * k + 1
  in 
  (List.range n |>.filter red_painted).length = 52 :=
begin
  sorry
end

end red_houses_after_painters_l662_662243


namespace set_B_has_4_elements_l662_662747

theorem set_B_has_4_elements (A B : Set ℕ) (hA : A = {1, 2}) (hUnion : A ∪ B = {1, 2}) :
  (∃ n, n = 4 ∧ fintype.card {x | x ∈ B} = n) :=
by {
  sorry -- Proof is skipped
}

end set_B_has_4_elements_l662_662747


namespace ceil_x_pow_2_values_l662_662814

theorem ceil_x_pow_2_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ n, n = 29 ∧ (∀ y, ceil (y^2) = ⌈x^2⌉ → 196 < y^2 ∧ y^2 ≤ 225) :=
sorry

end ceil_x_pow_2_values_l662_662814


namespace max_distance_l662_662771

noncomputable def polar_to_cartesian (rho theta : ℝ) : ℝ × ℝ := 
  (rho * Real.cos theta, rho * Real.sin theta)

noncomputable def curve_C (p : ℝ × ℝ) : Prop := 
  let x := p.1 
  let y := p.2 
  x^2 + y^2 - 2*y = 0

noncomputable def line_l (t : ℝ) : ℝ × ℝ := 
  (-3/5 * t + 2, 4/5 * t)

def x_axis_intersection (l : ℝ → ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := l 0 
  (x, 0)

noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem max_distance {M : ℝ × ℝ} {N : ℝ × ℝ}
  (curve_c : (ℝ × ℝ) → Prop)
  (line_l : ℝ → ℝ × ℝ)
  (h1 : curve_c = curve_C)
  (h2 : line_l = line_l)
  (M_def : x_axis_intersection line_l = M)
  (hNP : curve_c N) :
  distance M N ≤ Real.sqrt 5 + 1 :=
sorry

end max_distance_l662_662771


namespace product_of_cubes_l662_662690

theorem product_of_cubes :
  ( (2^3 - 1) / (2^3 + 1) * (3^3 - 1) / (3^3 + 1) * (4^3 - 1) / (4^3 + 1) * 
    (5^3 - 1) / (5^3 + 1) * (6^3 - 1) / (6^3 + 1) * (7^3 - 1) / (7^3 + 1) 
  ) = 57 / 72 := 
by
  sorry

end product_of_cubes_l662_662690


namespace largest_lambda_inequality_l662_662013

theorem largest_lambda_inequality (a b c d e : ℝ) (h_a : 0 ≤ a) (h_b : 0 ≤ b) (h_c : 0 ≤ c) (h_d : 0 ≤ d) (h_e : 0 ≤ e) :
  a^2 + 2 * b^2 + 2 * c^2 + d^2 + e^2 ≥ a * b + (sqrt 45 / 12) * b * c + c * d + 2 * d * e :=
by
  sorry

end largest_lambda_inequality_l662_662013


namespace cruiser_safe_path_exists_l662_662928

-- Define a simple type for points using coordinates.
structure Point where 
  x : ℝ 
  y : ℝ

-- Define the condition for a path avoiding mines.
def avoidsMines (p1 p2 : Point) (mines : list (Point × Point)) : Prop :=
  -- This function checks if a path from p1 to p2 does not intersect any mine.
  ∀ mine in mines, ¬intersects p1 p2 mine.fst mine.snd

-- Assume we have a function to check if two segments intersect.
noncomputable def intersects : Point → Point → Point → Point → Prop := sorry

-- Define the problem statement in Lean.
theorem cruiser_safe_path_exists (A B : Point) (mines : list (Point × Point)) :
  ∃ C : Point, avoidsMines A C mines ∧ avoidsMines C B mines := by
  sorry

end cruiser_safe_path_exists_l662_662928


namespace roots_purely_imaginary_l662_662941

def is_imaginary (z : ℂ) : Prop := z.re = 0

theorem roots_purely_imaginary (k : ℂ) (h : k.re < 0 ∧ k.im = 0) : 
  let discriminant := -9 + 40 * k
  ∀ z : ℂ, is_root (λ z, 10 * z^2 - 3 * complex.I * z - k) z → is_imaginary z := 
by 
  simp only [is_root, complex.I]
  sorry

end roots_purely_imaginary_l662_662941


namespace train_length_eq_1800_l662_662598

theorem train_length_eq_1800 (speed_kmh : ℕ) (time_sec : ℕ) (distance : ℕ) (L : ℕ)
  (h_speed : speed_kmh = 216)
  (h_time : time_sec = 60)
  (h_distance : distance = 60 * time_sec)
  (h_total_distance : distance = 2 * L) :
  L = 1800 := by
  sorry

end train_length_eq_1800_l662_662598


namespace sum_of_two_lowest_test_scores_l662_662950

theorem sum_of_two_lowest_test_scores (scores : List ℕ) (h_len : scores.length = 6)
  (h_mean : (scores.sum : ℕ) / 6 = 85) 
  (h_median : (scores.sorted.nth 2 + scores.sorted.nth 3) / 2 = 88)
  (h_mode : ∃ x, (list.count x scores) > 1 ∧ x = 89) :
  (scores.sorted.head + (scores.sorted.tail.head)) = 166 := 
  sorry

end sum_of_two_lowest_test_scores_l662_662950


namespace water_depth_is_60_l662_662335

def Ron_height : ℕ := 12
def depth_of_water (h_R : ℕ) : ℕ := 5 * h_R

theorem water_depth_is_60 : depth_of_water Ron_height = 60 :=
by
  sorry

end water_depth_is_60_l662_662335


namespace solve_for_m_l662_662028

theorem solve_for_m (m : ℝ) : ∃ x y : ℝ, y = 3 * m * x + 5 ∧ y = (3 * m - 2) * x + 7 :=
by
  use 1, 3 * m + 5
  simp
  rw [eq_comm]
  exact ⟨rfl, by linarith⟩

end solve_for_m_l662_662028


namespace probability_of_odd_and_divisible_by_3_l662_662206

theorem probability_of_odd_and_divisible_by_3 :
  let digits := [2, 3, 5, 7, 9],
      sum_digits := 26,
      divisible_by_3 := sum_digits % 3 = 0
  in sum_digits % 3 ≠ 0 → 0 = 0 :=
by
  intro h
  sorry

end probability_of_odd_and_divisible_by_3_l662_662206


namespace total_baseball_cards_l662_662561

-- Define the number of baseball cards each person has
def mary_cards : ℕ := 15
def sam_cards : ℕ := 15
def keith_cards : ℕ := 15
def alyssa_cards : ℕ := 15
def john_cards : ℕ := 12
def sarah_cards : ℕ := 18
def emma_cards : ℕ := 10

-- The total number of baseball cards they have
theorem total_baseball_cards :
  mary_cards + sam_cards + keith_cards + alyssa_cards + john_cards + sarah_cards + emma_cards = 100 :=
by
  sorry

end total_baseball_cards_l662_662561


namespace find_f_2021_l662_662594

def f (x : ℝ) : ℝ := sorry

theorem find_f_2021 (h : ∀ a b : ℝ, f ((a + 2 * b) / 3) = (f a + 2 * f b) / 3)
    (h1 : f 1 = 5) (h4 : f 4 = 2) : f 2021 = -2015 :=
by
  sorry

end find_f_2021_l662_662594


namespace smallest_integer_y_l662_662260

theorem smallest_integer_y (y : ℤ) (h : 7 - 3 * y < 20) : ∃ (y : ℤ), y = -4 :=
by
  sorry

end smallest_integer_y_l662_662260


namespace power_product_is_100_l662_662627

theorem power_product_is_100 :
  (10^0.6) * (10^0.4) * (10^0.3) * (10^0.2) * (10^0.5) = 100 :=
by
  sorry

end power_product_is_100_l662_662627


namespace compute_f_1_g_3_l662_662894

def f (x : ℝ) := 3 * x - 5
def g (x : ℝ) := x + 1

theorem compute_f_1_g_3 : f (1 + g 3) = 10 := by
  sorry

end compute_f_1_g_3_l662_662894


namespace area_of_triangle_is_correct_l662_662488

-- Define the conditions using Lean definitions
variables {a b c : ℝ} -- sides opposite to angles A, B, and C
variables {α β γ : ℝ} -- angles at A, B, and C
variables {O A B C : EuclideanSpace} -- points in Euclidean Space

-- Given conditions
def condition_1 := (b = 4)
def condition_2 := (4 * real.sqrt 6 * a * real.sin(2 * γ) = 3 * (a^2 + b^2 - c^2) * real.sin β)
def condition_3 := (2 * O + B + C = 0)
def condition_4 := (real.cos (∠ C A O) = 1 / 4)

-- Proof goal
theorem area_of_triangle_is_correct :
  condition_1 → condition_2 → condition_3 → condition_4 →
  (let s := (1/2 * euclidean.norm (A - B) * euclidean.norm (A - C) * real.sin (∠ B A C))
  in s = 2 * real.sqrt 15) :=
by
  intros
  -- Proof is omitted
  sorry

end area_of_triangle_is_correct_l662_662488


namespace ceil_x_squared_values_count_l662_662811

open Real

theorem ceil_x_squared_values_count (x : ℝ) (h : ceil x = 15) : 
  ∃ n : ℕ, n = 29 ∧ ∃ a b : ℕ, a ≤ b ∧ (∀ (m : ℕ), a ≤ m ∧ m ≤ b → (ceil (x^2) = m)) := 
by
  sorry

end ceil_x_squared_values_count_l662_662811


namespace shaded_area_l662_662991

variable (F G H I J K L M : Type) [fin: Finite F] [decF: DecidableEq F]
variable (FK GL HM IJ KG : ℝ)
variable (area : ℝ)

-- Given conditions
def square_area (s : ℝ) : Prop :=
  s * s = 80

def equal_segments (a b c d : ℝ) : Prop :=
  a = b ∧ b = c ∧ c = d

def segment_ratio (a b : ℝ) : Prop :=
  a = 3 * b

-- Prove that the area of the shaded region is 25
theorem shaded_area :
  ∃ (s l jk : ℝ), square_area s ∧ equal_segments FK GL HM IJ ∧ segment_ratio FK KG ∧ 
    (s = 4 * l) ∧ (16 * l * l = 80) ∧ (jk * jk = 50) ∧ (25 = 1/2 * 50) :=
begin
  sorry
end

end shaded_area_l662_662991


namespace quadratic_eq_zero_l662_662098

theorem quadratic_eq_zero (x a b : ℝ) (h : x = a ∨ x = b) : x^2 - (a + b) * x + a * b = 0 :=
by sorry

end quadratic_eq_zero_l662_662098


namespace arithmetic_sequence_problem_l662_662743

variable {ℕ : Type} [comm_semiring ℕ]

-- Conditions
def a_3 : ℕ := 7
def S_3 : ℕ := 12

-- Expected answers
def a_n (n : ℕ) : ℕ := 3 * n - 2
def S_n (n : ℕ) : ℕ := 3 * (n * n) / 2 - n / 2

-- Prove that the sequence general term and sum of the first n terms meet the conditions
theorem arithmetic_sequence_problem :
  (a_3 = 7) ∧ (S_3 = 12) →
  (∀ n : ℕ, a_n n = 3 * n - 2) ∧ (∀ n : ℕ, S_n n = 3 * n * n / 2 - n / 2) :=
sorry

end arithmetic_sequence_problem_l662_662743


namespace find_r_l662_662153

noncomputable def f (x r : ℝ) := (x - (r + 2)) * (x - (r + 4)) * (x - a)
noncomputable def g (x r : ℝ) := (x - (r + 3)) * (x - (r + 5)) * (x - b)

theorem find_r (a b : ℝ) (f g : ℝ → ℝ) (r : ℝ) : 
  (∀ x : ℝ, f x = (x - (r + 2)) * (x - (r + 4)) * (x - a) ∧ g x = (x - (r + 3)) * (x - (r + 5)) * (x - b)) ∧
  (∀ x : ℝ, f x - g x = 2 * r + 1) →
  r = 1/4 :=
by
  sorry

end find_r_l662_662153


namespace problem_I_problem_II_l662_662746

theorem problem_I (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h_sum : a + b + c = 3) :
  sqrt a + sqrt b + sqrt c ≤ 3 :=
sorry

theorem problem_II (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a + b + c = 3) (h_ab : c = a * b) :
  c ≤ 1 :=
sorry

end problem_I_problem_II_l662_662746


namespace probability_equal_even_odd_sum_gt_20_l662_662004

-- Definitions based on conditions:
def Die: Type := Fin 8  -- A die has 8 faces, represented by the finite type Fin 8

-- Probability of even number on a die
def even (d : Die) : Prop := (d.val % 2 = 1)  -- Adjusted to fit the Lean definition of even (1-indexed)
def odd (d : Die) : Prop := (d.val % 2 = 0)  -- Adjusted to fit the Lean definition of odd (1-indexed)

-- Main problem statement
theorem probability_equal_even_odd_sum_gt_20 : 
  ∑ (d1 d2 d3 d4 : Die) in (finset.filter (λ x, even x = tt) (finset.univ : finset Die)).toFinset, 
  d1.val + d2.val + d3.val + d4.val > 20 -> 
  ∑ (d1 d2 d3 d4 : Die) in (finset.filter (λ x, odd x = tt) (finset.univ : finset Die)).toFinset = 4 ->
  ∑ (d1 d2 d3 d4 : Die) in (finset.univ : finset Die) = 8 ->
  ∑ (d1 d2 d3 d4 : Die) (h : even (d1) ∧ even (d2) ∧ even (d3) ∧ even (d4)), 1 / (Finset.card (Finset.filter even (Finset.univ))) 
  = 35 / 256 :=
sorry

end probability_equal_even_odd_sum_gt_20_l662_662004


namespace trajectory_of_A_l662_662050

def B : ℝ × ℝ := (-5, 0)
def C : ℝ × ℝ := (5, 0)

def sin_B : ℝ := sorry
def sin_C : ℝ := sorry
def sin_A : ℝ := sorry

axiom sin_relation : sin_B - sin_C = (3/5) * sin_A

theorem trajectory_of_A :
  ∃ x y : ℝ, (x^2 / 9) - (y^2 / 16) = 1 ∧ x < -3 :=
sorry

end trajectory_of_A_l662_662050


namespace pick_4_balls_with_at_least_one_red_and_one_white_l662_662611

namespace PingPongBalls

-- Definitions from conditions in step a)
def balls := {red, white, yellow, blue, black} -- with red appearing twice, and white four times
def red_balls := 2
def white_balls := 4
def total_balls := 9
def different_balls := {yellow, blue, black}

-- The goal is to prove that there are 11 ways to pick 4 balls with at least one red and one white ball from these 9 balls
theorem pick_4_balls_with_at_least_one_red_and_one_white :
  ∃ n : ℕ, n = 11 ∧ ∀ (selection : finset balls),
   selection.card = 4 ∧ (∃ red : red_balls > 0, ∃ white : white_balls > 0, true) →
      n = 11 :=
sorry

end PingPongBalls

end pick_4_balls_with_at_least_one_red_and_one_white_l662_662611


namespace cone_height_l662_662752

theorem cone_height (radius_sector : ℝ) (central_angle : ℝ) (l : ℝ) (r : ℝ) (h : ℝ) :
  radius_sector = 3 →
  central_angle = 120 →
  l = 3 →
  2 * real.pi * r = (2 * real.pi / 3) * 3 →
  h = real.sqrt (l * l - r * r) →
  h = real.sqrt 6.75 :=
by
  intros radiusSectorEq centralAngleEq lEq rEq hEq
  rw [radiusSectorEq, centralAngleEq, lEq, ←hEq] at *
  have r_val : r = 1.5 := by
    rw [mul_div_cancel' (3 * 2 * real.pi) ((3 : ℝ) * 2), two_mul, mul_assoc, ←mul_assoc,
         inv_mul_cancel_right' (two_ne_zero : (2 : ℝ) ≠ 0), two_mul]
    linarith
  rw r_val at hEq
  sorry

end cone_height_l662_662752


namespace cone_lateral_area_l662_662065

theorem cone_lateral_area (S A B : Point) (cos_theta : ℝ) (alpha : ℝ) (area_SAB : ℝ) :
  cos_theta = 7 / 8 →
  alpha = π / 4 →
  area_SAB = 5 * sqrt 15 →
  ∃ r l : ℝ, (sqrt 2 * r = l) ∧ 
             (l * sqrt 15 / 8 * 2 = area_SAB) ∧ 
             (π * r * l = 40 * sqrt 2 * π) := sorry

end cone_lateral_area_l662_662065


namespace gcd_lcm_product_24_54_l662_662381

theorem gcd_lcm_product_24_54 :
  let a := 24 in
  let b := 54 in
  let gcd_ab := Int.gcd a b in
  let lcm_ab := Int.lcm a b in
  gcd_ab * lcm_ab = a * b := by
  let a := 24
  let b := 54
  have gcd_ab : Int.gcd a b = 6 := by
    rw [Int.gcd_eq_right_iff_dvd.mpr (dvd.intro 9 rfl)]
    
  have lcm_ab : Int.lcm a b = 216 := by
    sorry -- We simply add sorry here for the sake of completeness

  show gcd_ab * lcm_ab = a * b
  rw [gcd_ab, lcm_ab]
  norm_num

end gcd_lcm_product_24_54_l662_662381


namespace proof_abc_identity_l662_662177

variable {a b c : ℝ}

theorem proof_abc_identity
  (h_ne_a : a ≠ 1) (h_ne_na : a ≠ -1)
  (h_ne_b : b ≠ 1) (h_ne_nb : b ≠ -1)
  (h_ne_c : c ≠ 1) (h_ne_nc : c ≠ -1)
  (habc : a * b + b * c + c * a = 1) :
  a / (1 - a ^ 2) + b / (1 - b ^ 2) + c / (1 - c ^ 2) = (4 * a * b * c) / (1 - a ^ 2) / (1 - b ^ 2) / (1 - c ^ 2) :=
by 
  sorry

end proof_abc_identity_l662_662177


namespace find_natural_pairs_l662_662375

theorem find_natural_pairs (m n : ℕ) :
  (n * (n - 1) * (n - 2) * (n - 3) = m * (m - 1)) ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 3 ∧ m = 1) :=
by sorry

end find_natural_pairs_l662_662375


namespace toby_deleted_nine_bad_shots_l662_662619

theorem toby_deleted_nine_bad_shots 
  (x : ℕ)
  (h1 : 63 > x)
  (h2 : (63 - x) + 15 - 3 = 84)
  : x = 9 :=
by
  sorry

end toby_deleted_nine_bad_shots_l662_662619


namespace sum_g_eq_1000_l662_662533

def g (x : ℝ) : ℝ := 4 / (16^x + 4)

theorem sum_g_eq_1000 :
  (∑ k in Finset.range 2000, g ((k + 1) / 2001)) = 1000 := 
sorry

end sum_g_eq_1000_l662_662533


namespace price_of_second_tea_l662_662502

theorem price_of_second_tea (price_first_tea : ℝ) (mixture_price : ℝ) (required_ratio : ℝ) (price_second_tea : ℝ) :
  price_first_tea = 62 → mixture_price = 64.5 → required_ratio = 3 → price_second_tea = 65.33 :=
by
  intros h1 h2 h3
  sorry

end price_of_second_tea_l662_662502


namespace avg_cost_is_12_cents_l662_662659

noncomputable def avg_cost_per_pencil 
    (price_per_package : ℝ)
    (num_pencils : ℕ)
    (shipping_cost : ℝ)
    (discount_rate : ℝ) : ℝ :=
  let price_after_discount := price_per_package - (discount_rate * price_per_package)
  let total_cost := price_after_discount + shipping_cost
  let total_cost_cents := total_cost * 100
  total_cost_cents / num_pencils

theorem avg_cost_is_12_cents :
  avg_cost_per_pencil 29.70 300 8.50 0.10 = 12 := 
by {
  sorry
}

end avg_cost_is_12_cents_l662_662659


namespace g_in_M_l662_662149

def M : set (ℝ → ℝ) :=
  {f | ∀ x1 x2 : ℝ, |x1| ≤ 1 → |x2| ≤ 1 → |f x1 - f x2| ≤ 4 * |x1 - x2|}

def g (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem g_in_M : g ∈ M :=
sorry

end g_in_M_l662_662149


namespace evaluate_f_2x_l662_662731

def f (x : ℝ) : ℝ := x^2 - 1

theorem evaluate_f_2x (x : ℝ) : f (2 * x) = 4 * x^2 - 1 :=
by
  sorry

end evaluate_f_2x_l662_662731


namespace sin_X_correct_l662_662479

variables {X Y Z : Type} [real_matrix X Y Z]

-- Define the lengths of the sides
def triangle_90 (XY XZ : ℝ) : Prop :=
  XY = 15 ∧ XZ = 5 ∧ (XY^2 + XZ^2 = 15^2 + 5^2)

-- Define the sine function based on the given variables.
def sin_X (XY XZ : ℝ) : ℝ :=
  XZ / XY

theorem sin_X_correct (XY XZ : ℝ) (h: triangle_90 XY XZ) : 
  sin_X XY XZ = 1 / 3 :=
by
  sorry

end sin_X_correct_l662_662479


namespace inequality_solution_real_l662_662954

theorem inequality_solution_real (x : ℝ) :
  (x + 1) * (2 - x) < 4 ↔ true :=
by
  sorry

end inequality_solution_real_l662_662954


namespace range_of_y0_l662_662661

theorem range_of_y0 (y0 : Real) (M : Real × Real) (O : Real × Real → Prop)
  (hM : M = (Real.sqrt 3, y0))
  (hO : ∀ (P : Real × Real), O P ↔ (P.1^2 + P.2^2 = 1))
  (hAngle : ∀ N : Real × Real, (N ∈ tangent_points M O) → angle O M N ≥ π / 6) :
  -1 ≤ y0 ∧ y0 ≤ 1 :=
sorry

end range_of_y0_l662_662661


namespace triangle_area_l662_662256

-- Define the vertices of the triangle
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (6, 1)
def C : ℝ × ℝ := (3, -4)

-- State that the area of the triangle is 12.5 square units
theorem triangle_area :
  let base := 6 - 1
  let height := 1 - -4
  (1 / 2) * base * height = 12.5 := by
  sorry

end triangle_area_l662_662256


namespace chord_lengths_less_than_l662_662462

theorem chord_lengths_less_than (r k : ℝ) (chords : list ℝ) 
  (h_r : r = 1)
  (h_diameter_intersect : ∀ (d : ℝ), d ∈ diameters -> (diameter_intersections d chords).length ≤ k) :  
  sum chords < π * k := 
sorry

end chord_lengths_less_than_l662_662462


namespace simplify_expression_l662_662946

variables {m : ℝ}

theorem simplify_expression (m : ℝ) : 
  ((1 / (3 * m)) ^ (-3) * (2 * m) ^ 4) = 432 * m ^ 7 :=
by
  sorry

end simplify_expression_l662_662946


namespace ratio_of_boys_l662_662111

theorem ratio_of_boys (p : ℚ) (h : p = (3/5) * (1 - p)) : p = 3 / 8 := by
  sorry

end ratio_of_boys_l662_662111


namespace hyperbola_example_l662_662011

noncomputable def hyperbola_properties : Prop :=
  let a_squared := 6
  let b_squared := 3
  let c_squared := a_squared + b_squared
  let a := Real.sqrt a_squared
  let b := Real.sqrt b_squared
  let c := Real.sqrt c_squared in
  (a_squared = 6) ∧ (b_squared = 3) ∧ (c_squared = 9) ∧
  (a = Real.sqrt 6) ∧ (b = Real.sqrt 3) ∧ (c = 3) ∧
  (∀ x y, (x^2 / a_squared - y^2 / b_squared = 1) →
    ∀ (A1 A2 F1 F2 : Prod ℝ ℝ),
      (A1 = (-a, 0)) ∧ (A2 = (a, 0)) ∧
      (F1 = (-c, 0)) ∧ (F2 = (c, 0)) ∧
      (Real.sqrt a_squared = Real.sqrt 6) ∧
      (Real.sqrt b_squared = Real.sqrt 3) ∧
      (c / a = Real.sqrt 6 / 2) ∧
      (y = (Real.sqrt 2 / 2) * x ∨ y = -(Real.sqrt 2 / 2) * x))

theorem hyperbola_example : hyperbola_properties :=
by
  sorry

end hyperbola_example_l662_662011


namespace area_cross_section_of_prism_l662_662203

theorem area_cross_section_of_prism :
  let side_length := 2,
      base_angle := 30,
      prism_height := 1,
      plane_angle := 60 in
  let cross_section_area := (4 * Real.sqrt 3) / 3 in
  ∃ (A B : ℝ),
  A = side_length ∧
  B = base_angle ∧
  prism_height = 1 ∧
  plane_angle = 60 ∧
  cross_section_area = (4 * Real.sqrt 3) / 3 :=
begin
  sorry
end

end area_cross_section_of_prism_l662_662203


namespace find_number_l662_662296

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end find_number_l662_662296


namespace total_votes_polled_l662_662120

theorem total_votes_polled (V : ℕ) (h1 : ∃ c : ℕ, c = 0.75 * V) (h2 : ∃ m : ℕ, m = 1500 ) :
  V = 3000 :=
sorry

end total_votes_polled_l662_662120


namespace factorization_correct_l662_662707

theorem factorization_correct (x : ℝ) : 
  98 * x^7 - 266 * x^13 = 14 * x^7 * (7 - 19 * x^6) :=
by
  sorry

end factorization_correct_l662_662707


namespace amount_saved_l662_662876

-- Initial conditions as definitions
def initial_amount : ℕ := 6000
def cost_ballpoint_pen : ℕ := 3200
def cost_eraser : ℕ := 1000
def cost_candy : ℕ := 500

-- Mathematical equivalent proof problem as a Lean theorem statement
theorem amount_saved : initial_amount - (cost_ballpoint_pen + cost_eraser + cost_candy) = 1300 := 
by 
  -- Proof is omitted
  sorry

end amount_saved_l662_662876


namespace find_distance_l662_662002

theorem find_distance :
  ∀ (Eddy_distance Eddy_time Freddy_time : ℝ) (avg_speed_ratio : ℝ),
    Eddy_distance = 600 →
    Eddy_time = 3 →
    Freddy_time = 4 →
    avg_speed_ratio = 2.2222222222222223 →
    let V_Eddy := Eddy_distance / Eddy_time in
    ∃ (D_AC : ℝ), V_Eddy / (D_AC / Freddy_time) = avg_speed_ratio ∧ D_AC = 360 :=
by
  intros Eddy_distance Eddy_time Freddy_time avg_speed_ratio h1 h2 h3 h4 V_Eddy
  use 360
  rw [h1, h2, h3, h4]
  field_simp
  sorry

end find_distance_l662_662002


namespace remaining_grass_area_l662_662652

theorem remaining_grass_area (d : ℝ) (w : ℝ) (π : ℝ) : d = 20 → w = 4 → 
  ∃ A : ℝ, A = π * (d / 2) ^ 2 ∧ A = 100 * π :=
begin
  intros,
  use π * (d / 2) ^ 2,
  split,
  { refl, },
  { sorry, }
end

end remaining_grass_area_l662_662652


namespace real_root_bound_l662_662320

noncomputable def P (x : ℝ) (n : ℕ) (ns : List ℕ) : ℝ :=
  1 + x^2 + x^5 + ns.foldr (λ n acc => x^n + acc) 0 + x^2008

theorem real_root_bound (n1 n2 : ℕ) (ns : List ℕ) (x : ℝ) :
  5 < n1 →
  List.Chain (λ a b => a < b) n1 (n2 :: ns) →
  n2 < 2008 →
  P x n1 (n2 :: ns) = 0 →
  x ≤ (1 - Real.sqrt 5) / 2 :=
sorry

end real_root_bound_l662_662320


namespace distance_travelled_downstream_l662_662234

-- Definitions and Conditions
def speed_boat_still_water := 12 -- in km/hr
def rate_of_current := 4 -- in km/hr
def time_travel_minutes := 18 -- in minutes

-- Required conversion and effective speed calculations
def effective_speed_downstream := speed_boat_still_water + rate_of_current -- in km/hr
def time_travel_hours := time_travel_minutes / 60.0 -- converting minutes to hours

-- Desired proof statement
theorem distance_travelled_downstream : 
  effective_speed_downstream * time_travel_hours = 4.8 := 
by
  -- proof goes here
  sorry

end distance_travelled_downstream_l662_662234


namespace parallelogram_sides_l662_662315

theorem parallelogram_sides (x y : ℝ) (h₁ : 4 * x + 1 = 11) (h₂ : 10 * y - 3 = 5) : x + y = 3.3 :=
sorry

end parallelogram_sides_l662_662315


namespace intersection_points_of_C1_and_C2_parametric_line_PQ_values_l662_662491

-- Definition and proof problem for intersection points of C1 and C2
theorem intersection_points_of_C1_and_C2 (ρ θ: ℝ) (C1 C2: ℝ → Prop) :
  C1 ρ = 4 * sin θ ∧ C2 ρ = ρ * cos (θ - π / 4) = 2 * sqrt 2 →
  (∃ (ρ1 θ1 ρ2 θ2: ℝ),
    (ρ1 = 4 ∧ θ1 = π / 2) ∧
    (ρ2 = 2 * sqrt 2 ∧ θ2 = π / 4)) :=
by
  sorry

-- Definition and proof problem for values of a and b
theorem parametric_line_PQ_values (a b: ℝ) :
  let x := t^3 + a
  let y := b/2 * t^3 + 1
  exist xP yP xQ yQ: ℝ,
    line_PQ xP yP xQ yQ = t^3 + (a, b) →
    a = -1 ∧ b = 2 :=
by
  sorry

end intersection_points_of_C1_and_C2_parametric_line_PQ_values_l662_662491


namespace MNOP_is_rhombus_l662_662143

open EuclideanGeometry

variables {P A B M N C D O : Point}
variables {ω : Circle}

-- Conditions
axiom tangent_points (hP : ¬(P ∈ ω)) (A B ∈ ω) (tangent_PA : -- proof that PA is tangent to ω at A) (tangent_PB : -- proof that PB is tangent to ω at B) : PA = PB

-- Definitions of midpoints
axiom midpoint_AP (M : midpoint A P M)
axiom midpoint_AB (N : midpoint A B N)

-- Conditions on points C, D and O
axiom collinear_MN_C (hMNC : collinear {M, N, C})
axiom hNC : N lies between M and C
axiom point_D (hD : C ∈ ω ∧ line PC intersects ω again at D)
axiom intersect_ND_PB (hO : Nd intersects PB at O)

-- Theorem
theorem MNOP_is_rhombus : rhombus M N O P :=
sorry

end MNOP_is_rhombus_l662_662143


namespace tan_identity_at_30_degrees_l662_662799

theorem tan_identity_at_30_degrees :
  let A := 30
  let B := 30
  let deg_to_rad := pi / 180
  let tan := fun x : ℝ => Real.tan (x * deg_to_rad)
  (1 + tan A) * (1 + tan B) = (4 + 2 * Real.sqrt 3) / 3 := by
  sorry

end tan_identity_at_30_degrees_l662_662799


namespace fraction_blue_mushrooms_white_spots_l662_662345

theorem fraction_blue_mushrooms_white_spots (x : ℝ) :
  let red_mushrooms : ℕ := 12
      brown_mushrooms : ℕ := 6
      blue_mushrooms : ℕ := 6
      red_white_spots := (2 / 3) * red_mushrooms
      brown_white_spots := brown_mushrooms
      blue_white_spots := x * blue_mushrooms
      total_white_spots := red_white_spots + brown_white_spots + blue_white_spots
  in total_white_spots = 17 → x = 1 / 2 :=
by
  intros
  sorry

end fraction_blue_mushrooms_white_spots_l662_662345


namespace line_standard_equation_circle_cartesian_equation_line_intersects_circle_l662_662704

open real

-- Definitions from conditions
def parametric_line (t : ℝ) : ℝ × ℝ := (t, 1 + 2 * t)
def polar_circle (θ : ℝ) : ℝ := 2 * sqrt 2 * sin (θ + π / 4)

-- Statement to prove the standard equation of line l
theorem line_standard_equation (x y t : ℝ) (h1 : x = t) (h2 : y = 1 + 2 * t) :
  y - 2 * x - 1 = 0 :=
sorry

-- Statement to prove the Cartesian equation of circle C
theorem circle_cartesian_equation (x y : ℝ) (ρ θ : ℝ) 
  (h1 : x = ρ * cos θ) (h2 : y = ρ * sin θ) 
  (h3 : ρ = 2 * sqrt 2 * sin (θ + π / 4)) :
  (x - 1) ^ 2 + (y - 1) ^ 2 = 2 :=
sorry

-- Statement to prove that line l and circle C intersect
theorem line_intersects_circle :
  let d := abs (1 - 2 - 1) / sqrt 5 in
  d < sqrt 2 :=
sorry

end line_standard_equation_circle_cartesian_equation_line_intersects_circle_l662_662704


namespace part1_part2_l662_662133

theorem part1 (a b : ℝ) (h_a : a = 4) (h_b : b = 4 * Real.sqrt 2) (h_sin_cos : Real.sin (Real.pi / 4) - Real.cos (Real.pi / 4) = 0) : 
  ∃ c : ℝ, c = 4 ∧ a^2 = b^2 + c^2 - 2 * b * c * Real.cos (Real.pi / 4) :=
begin
  use 4,
  split,
  { refl },
  { sorry }
end

theorem part2 (a : ℝ) (h_a : a = 4) : 
  ∃ m : ℝ, 4 < m ∧ m < 4 * Real.sqrt 2 ∧ 
  ∃ t1 t2 : Triangle, 
    t1.side1 = a ∧ t1.angleA = Real.pi / 4 ∧ t1.side2 = m ∧
    t2.side1 = a ∧ t2.angleA = Real.pi / 4 ∧ t2.side2 = m ∧ 
    t1 ≠ t2 :=
begin
  use 5,
  split,
  { linarith [Real.sqrt 2_pos], },
  split,
  { linarith [Real.sqrt 2_pos], },
  { use Triangle.mk 4 (Real.pi / 4) 5, -- Assuming Triangle is a valid structure
    use Triangle.mk 4 (Real.pi / 4) 6, -- Placeholder for different triangle
    split, -- conditions for triangle 1
    { refl },
    split,
    { exact Real.pi_div_four_pos },
    split,
    { refl },
    split, -- conditions for triangle 2
    { refl },
    split,
    { exact Real.pi_div_four_pos },
    split,
    { refl },
    { exact absurd rfl (by linarith) }, -- t1 ≠ t2 via trivial different third side length
  }
end

end part1_part2_l662_662133


namespace solve_for_x_l662_662193

theorem solve_for_x :
  (∃ x : ℝ, (2^(4*x + 2)) * ((2^3)^(2*x + 7)) = (2^4)^(3*x + 10) ∧ x = -17/2) :=
by
  sorry

end solve_for_x_l662_662193


namespace common_distance_l662_662174

open Real

noncomputable def point_distance (a : ℝ) : ℝ :=
  let x := a / 2 in
  let y := a * 5 / 8 in
  sqrt (x^2 + y^2)

theorem common_distance (a : ℝ) : 
  (0 < a) → point_distance a = 5 * a / 8 :=
  sorry

end common_distance_l662_662174


namespace triangle_ABC_area_l662_662864

theorem triangle_ABC_area (area_WXYZ : ℕ) (side_small_squares : ℕ) (isosceles_AB_AC : AB = AC)
  (A_folds_to_O : folds A BC O) : 
  area_WXYZ = 64 ∧ side_small_squares = 2 → 
  area_triangle ABC = 16 :=
by {
  sorry
}

end triangle_ABC_area_l662_662864


namespace minimize_distance_l662_662246

theorem minimize_distance (A B C D : ℝ) (h : A < B) (h1 : B < C) (h2 : C < D) :
  ∃ P ∈ set.Icc B C, 
  ∀ Q : ℝ, P ∈ set.Icc B C → dist Q A + dist Q B + dist Q C + dist Q D ≥ dist P A + dist P B + dist P C + dist P D :=
sorry

end minimize_distance_l662_662246


namespace prove_circle_problem_l662_662411

def point_on_circle (M : ℝ × ℝ) : Prop := 
  let (x, y) := M 
  in x^2 + y^2 - 4*x - 14*y + 45 = 0

def P_on_circle (a : ℝ) : Prop :=
  (a, a+1) ∈ {M : ℝ × ℝ | point_on_circle M}

def length_PQ (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  in Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def slope_PQ (P Q : ℝ × ℝ) : ℝ :=
  let (x1, y1) := P
  let (x2, y2) := Q
  in (y2 - y1) / (x2 - x1)

noncomputable def max_min_length_MQ (M Q C : ℝ × ℝ) (r : ℝ) : ℝ × ℝ :=
  let (x1, y1) := M
  let (x2, y2) := Q
  let (x3, y3) := C
  in (Real.sqrt ((x2 - x3)^2 + (y2 - y3)^2) + r, Real.sqrt ((x2 - x3)^2 + (y2 - y3)^2) - r)

noncomputable def max_min_slope_MQ (C Q : ℝ × ℝ) (radius : ℝ) : ℝ × ℝ :=
  (2 + Real.sqrt 3, 2 - Real.sqrt 3)

theorem prove_circle_problem :
  let C := (2, 7)
  let r := 2 * Real.sqrt 2
  let Q := (-2, 3)
  ∀ M : ℝ × ℝ, point_on_circle M →
  ∃ a b : ℝ, P_on_circle a ∧ length_PQ (a, a+1) Q = 2 * Real.sqrt 10 ∧ slope_PQ (a, a+1) Q = (1 / 3) ∧
  max_min_length_MQ M Q C r = (6 * Real.sqrt 2, 2 * Real.sqrt 2) ∧
  max_min_slope_MQ C Q r = (2 + Real.sqrt 3, 2 - Real.sqrt 3) :=
by
  sorry

end prove_circle_problem_l662_662411


namespace ellipse_hyperbola_eccentricity_sum_eq_two_l662_662738

theorem ellipse_hyperbola_eccentricity_sum_eq_two
  (F1 F2 : ℝ)
  (e1 e2 : ℝ)
  (A : ℝ × ℝ)
  (h1 : ∠ (F1, A, F2) = 90°) 
  (h2 : ∃ a, ∃ a', |A.1 - F1| + |A.1 - F2| = 2 * a ∧ |A.1 - F1| - |A.1 - F2| = 2 * a') :
  (1 / e1^2 + 1 / e2^2 = 2) := 
sorry

end ellipse_hyperbola_eccentricity_sum_eq_two_l662_662738


namespace cos_diff_formula_triangle_right_angle_l662_662940

-- Part (1)
theorem cos_diff_formula (α β A B : ℝ)
  (h1 : α + β = A)
  (h2 : α - β = B)
  (h3 : α = (A + B) / 2)
  (h4 : β = (A - B) / 2)
  : cos A - cos B = -2 * sin ((A + B) / 2) * sin ((A - B) / 2) :=
  sorry

-- Part (2)
theorem triangle_right_angle (A B C : ℝ)
  (h_angles : A + B + C = π)  -- Interior angles of a triangle sum to π
  (h_cos_eq : cos (2 * A) - cos (2 * B) = 1 - cos (2 * C))
  : (B = π / 2 ∨ A = π / 2 ∨ C = π / 2) :=
  sorry

end cos_diff_formula_triangle_right_angle_l662_662940


namespace problem_1_problem_2_l662_662071

-- Problem 1
theorem problem_1: (f : ℝ → ℝ := λ x => sin x + cos (x + π / 6)) 
  (ω : ℝ := 1):
  f (π / 3) = sqrt 3 / 2 :=
by 
  sorry

-- Problem 2
theorem problem_2: (f : ℝ → ℝ := λ x => sin (2 * x + π / 3)) 
  (ω : ℝ := 2):
  ∃ x ∈ set.Icc (0 : ℝ) (π / 4), 
  x = π / 12 ∧ ∀ y ∈ set.Icc (0 : ℝ) (π / 4), f y ≤ f x :=
by 
  sorry

end problem_1_problem_2_l662_662071


namespace water_depth_upright_l662_662325

noncomputable def cylinder_tank_volume (r h w_d : ℝ) : ℝ :=
  let A := r^2 * real.arccos((r-w_d)/r) - (r-w_d)*real.sqrt(2*r*w_d-w_d^2)
  A * h

noncomputable def height_when_upright (V r : ℝ) : ℝ :=
  V / (real.pi * r^2)

theorem water_depth_upright :
  let r := 3
  let h := 10
  let w_d := 2
  let V := (9 * real.arccos ((r - (r - w_d)) / r) - (r - (r - w_d)) * real.sqrt (2 * r * (r - w_d) - (r - w_d)^2)) * h in 
  height_when_upright V r = (10 * (9 * real.arccos (2 / 3) - 2 * real.sqrt 5)) / (9 * real.pi) :=
by
  sorry

end water_depth_upright_l662_662325


namespace good_set_cardinality_l662_662422

theorem good_set_cardinality (p : ℕ) [fact (nat.prime p)] (F : set ℕ) (A : set ℕ) (hF : F = {0, 1, .. p - 1})
  (hA : ∀ (a b : ℕ), a ∈ A → b ∈ A → (a * b + 1) % p ∈ A) : 
  ∃ (n : ℕ), n = 1 ∨ n = p ∧ (∀ (x : ℕ), x ∈ A → x ∈ F) ∧ n = (A.to_finset.card) := 
sorry

end good_set_cardinality_l662_662422


namespace smallest_N_expansion_terms_l662_662720

theorem smallest_N_expansion_terms :
  (∀ N : ℕ, (∑ (x y z w u t : ℕ), (x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 ∧ u > 0 ∧ x + y + z + w + u + t = N) → 3432)
  → N = 16) :=
by {
  sorry
}

end smallest_N_expansion_terms_l662_662720


namespace travel_between_grandparents_and_brother_l662_662138

variable (Days : Type) [AddGroup Days]

structure KellyTrip where
  first_day_travel : Days
  grandparents_days : Days
  next_day_travel : Days
  brother_days : Days
  two_days_travel_to_sister : Days
  sister_days : Days
  two_days_travel_home : Days

def Kelly : KellyTrip Days := {
  first_day_travel := 1,
  grandparents_days := 5,
  next_day_travel := 1,
  brother_days := 5,
  two_days_travel_to_sister := 2,
  sister_days := 5,
  two_days_travel_home := 2
}

theorem travel_between_grandparents_and_brother (K : KellyTrip Days) :
  K.next_day_travel = 1 := by
  sorry

end travel_between_grandparents_and_brother_l662_662138


namespace Misha_can_place_rooks_l662_662918

def rook_placement_possible : Prop :=
  ∃ (positions : Finset (Fin 100 × Fin 100)), 
    positions.card = 100 ∧ 
    (∀ p1 p2 ∈ positions, p1 = p2 ∨ (p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2)) ∧ 
    (positions.card <= 199)

theorem Misha_can_place_rooks : rook_placement_possible :=
sorry

end Misha_can_place_rooks_l662_662918


namespace octagon_area_l662_662328

theorem octagon_area (a b : ℝ)
  (h1 : 4 * a = 8 * b)
  (h2 : a^2 = 16) :
  2 * (1 + real.sqrt 2) * b^2 = 8 * (1 + real.sqrt 2) :=
by 
  sorry

end octagon_area_l662_662328


namespace min_folds_paper_exceeds_moon_distance_l662_662387

theorem min_folds_paper_exceeds_moon_distance (n : ℕ) : 
  0.0384 * 2 ^ (n-1) ≥ 384000000 → n ≥ 41 :=
sorry

end min_folds_paper_exceeds_moon_distance_l662_662387


namespace coplanar_condition_l662_662150

noncomputable def find_m (A B C D : Type) [vector_space : vector_space ℝ] 
  (vector_oa vector_ob vector_oc vector_od : vector_space)
  (h : 4 • vector_oa - 3 • vector_ob + 6 • vector_oc + m • vector_od = 0) : ℝ := by
  sorry

theorem coplanar_condition (A B C D : Type) [vector_space : vector_space ℝ] 
  (vector_oa vector_ob vector_oc vector_od : vector_space)
  (h : 4 • vector_oa - 3 • vector_ob + 6 • vector_oc + m • vector_od = 0) : m = -7 := by
  sorry

end coplanar_condition_l662_662150


namespace f_increasing_range_of_a_l662_662419

-- Defining the function f with the given properties
axiom f : ℝ → ℝ

-- Defining the conditions for f
noncomputable def f_defined : ∀ x, 0 < x → ∃ y, f(x) = y := sorry
axiom f_property1 : ∀ x y : ℝ, 0 < x → 0 < y → f(x * y) = f(x) + f(y)
axiom f_property2 : ∀ x : ℝ, 1 < x → 0 < f(x)
axiom f_value : f(3) = 1

-- Proving that f is an increasing function
theorem f_increasing : ∀ x1 x2 : ℝ, 0 < x1 → x1 < x2 → f(x1) < f(x2) := sorry

-- Defining sets A and B
def A : set ℝ := { x | f(x) > f(x - 1) + 2 }
def B (a : ℝ) : set ℝ := { x | f((a + 1) * x - 1 / (x + 1)) > 0 }

-- Proving the range of a
theorem range_of_a : ∀ a : ℝ, (A ∩ B a = ∅) → a ≤ 16 / 9 := sorry

end f_increasing_range_of_a_l662_662419


namespace ambiguous_times_l662_662317

theorem ambiguous_times (h m : ℝ) : 
  (∃ k l : ℕ, 0 ≤ k ∧ k < 12 ∧ 0 ≤ l ∧ l < 12 ∧ 
              (12 * h = k * 360 + m) ∧ 
              (12 * m = l * 360 + h) ∧
              k ≠ l) → 
  (∃ n : ℕ, n = 132) := 
sorry

end ambiguous_times_l662_662317


namespace count_integer_values_l662_662235

theorem count_integer_values (x : ℝ) (h1 : 3 < sqrt x) (h2 : sqrt x < 4) : 
  ∃ n : ℕ, n = 6 := 
sorry

end count_integer_values_l662_662235


namespace problem_statement_l662_662895

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 2 then real.exp (x - 1) else real.log (x*x - 1) / real.log 3

theorem problem_statement : f (f (real.sqrt 10)) = real.exp 1 := 
sorry

end problem_statement_l662_662895


namespace q_at_2_is_41_l662_662551

-- Given conditions
variables {a b : ℤ}
def q (x : ℤ) : ℤ := x^2 + a * x + b

-- given polynomial expressions
def p1 (x : ℤ) : ℤ := 2 * x^4 + 8 * x^2 + 50
def p2 (x : ℤ) : ℤ := 6 * x^4 + 12 * x^2 + 56 * x + 20

-- q(x) needs to be a factor of both p1(x) and p2(x)
def is_factor (q p : ℤ → ℤ) : Prop := ∀ x, q(x) ∣ p(x)

-- proving q(2) = 41 under the given conditions
theorem q_at_2_is_41 (hq1 : is_factor q p1) (hq2 : is_factor q p2) : q 2 = 41 :=
begin
  sorry
end

end q_at_2_is_41_l662_662551


namespace product_event_classification_l662_662338
open ProbabilityTheory

def numProducts : ℕ := 200
def numFirstGradeProducts : ℕ := 192
def numSecondGradeProducts : ℕ := 8
def numSelectedProducts : ℕ := 9

def event1 := ("selecting 9 out of these 200 products, all of them are first-grade")
def event2 := ("selecting 9 out of these 200 products, all of them are second-grade")
def event3 := ("selecting 9 out of these 200 products, not all of them are first-grade")
def event4 := ("selecting 9 out of these 200 products, the number of products that are not first-grade is less than 100")

theorem product_event_classification :
  (is_random_event event1) ∧ 
  (is_impossible_event event2) ∧ 
  (is_random_event event3) ∧ 
  (is_certain_event event4) :=
sorry

end product_event_classification_l662_662338


namespace probability_eventA_l662_662929

-- Define the sample space S
def S : Set (ℕ × ℕ) := {p | 1 ≤ p.1 ∧ p.1 ≤ 5 ∧ 1 ≤ p.2 ∧ p.2 ≤ 5}

-- Define the event A representing the sum being 6
def eventA (p : ℕ × ℕ) : Prop := p.1 + p.2 = 6

-- Define the probability P
noncomputable def P (e : Set (ℕ × ℕ)) : ℚ :=
  ↑(e.toFinset.card) / ↑(S.toFinset.card)

-- Axiom to convert infinite sets to finite ones for this example - simplifying finite set assumption
axiom fin_S : S.finite

theorem probability_eventA : P ({p | eventA p} ∩ S) = 1 / 5 :=
by
  let S_fin := fin_S.toFinset
  have : S_fin.card = 25 := sorry
  let eventA_fin := ({p ∈ S | eventA p}).toFinset
  have : eventA_fin.card = 5 := sorry
  rw [P, this, this]
  -- Perform the calculation
  norm_num
  sorry

end probability_eventA_l662_662929


namespace lines_concurrent_l662_662528

variables {Γ Γ₁ Γ₂ Γ₃ : Type*}
variables [circle Γ] [circle Γ₁] [circle Γ₂] [circle Γ₃]
variables {A B C P Q R : point}

-- Define the conditions
def externally_tangent_pairwise (Γ₁ Γ₂ Γ₃ : circle) : Prop :=
  tangent_externally Γ₁ Γ₂ ∧ tangent_externally Γ₂ Γ₃ ∧ tangent_externally Γ₃ Γ₁

def internally_tangent_common (Γ Γ₁ Γ₂ Γ₃ : circle) (A B C : point) : Prop :=
  tangent_internally Γ Γ₁ A ∧ tangent_internally Γ Γ₂ B ∧ tangent_internally Γ Γ₃ C

-- The proof problem statement
theorem lines_concurrent
  (h₁ : externally_tangent_pairwise Γ₁ Γ₂ Γ₃)
  (h₂ : P = point_of_tangency Γ₂ Γ₃)
  (h₃ : Q = point_of_tangency Γ₃ Γ₁)
  (h₄ : R = point_of_tangency Γ₁ Γ₂)
  (h₅ : internally_tangent_common Γ Γ₁ Γ₂ Γ₃ A B C) :
  concurrent (line_through A P) (line_through B Q) (line_through C R) :=
sorry

end lines_concurrent_l662_662528


namespace solution_set_of_inequality_l662_662231

theorem solution_set_of_inequality (x : ℝ) :
  (x + 1) * (2 - x) < 0 ↔ (x > 2 ∨ x < -1) :=
by
  sorry

end solution_set_of_inequality_l662_662231


namespace ellipse_equation_triangle_area_range_l662_662040

def ellipse (x y a b : Real) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def parabola_eq (y x: Real) : Prop := y^2 = 4 * x

theorem ellipse_equation : 
  ∀ (x y a b : Real), 
  a = 2 ∧ b = √3 ∧ ellipse x y a b ∧ (1 / 2 = c / a) → (x^2 / 4 + y^2 / 3 = 1) := 
by 
  intros; 
  sorry

theorem triangle_area_range : 
  ∀ (λ t : Real), 
  (2 ≤ λ ∧ λ ≤ 3) ∧ (t^2 = (λ + 1)^2 / (4 * λ)) → (√2 ≤ √(16 * t^2 - 16) ∧ √(16 * t^2 - 16) ≤ 4 * √3 / 3) :=
by
  intros; 
  sorry

end ellipse_equation_triangle_area_range_l662_662040


namespace bus_driver_regular_rate_l662_662300

theorem bus_driver_regular_rate 
  (total_hours : ℝ) 
  (regular_hours : ℝ) 
  (overtime_multiplier : ℝ) 
  (total_compensation : ℝ) :
  total_hours = 63.62 →
  regular_hours = 40 →
  overtime_multiplier = 1.75 →
  total_compensation = 976 →
  ∃ R : ℝ, R ≈ 12 :=
by
  intros h_total_hours h_regular_hours h_overtime_multiplier h_total_compensation
  let overtime_hours := total_hours - regular_hours
  have h_overtime_hours : overtime_hours = 23.62 := sorry
  let overtime_rate := overtime_multiplier * R
  let eqn := total_compensation = (regular_hours * R) + (overtime_hours * overtime_rate)
  have h_eqn : 976 = 40 * R + 23.62 * 1.75 * R := by
    rw [h_total_hours, h_regular_hours, h_overtime_multiplier, h_total_compensation]
    sorry
  have h_R : R ≈ 12 := by
    sorry
  exact ⟨R, h_R⟩

end bus_driver_regular_rate_l662_662300


namespace tangent_parallel_divide_l662_662978

noncomputable theory
open_locale classical

variables {Ω : Type*} [nonempty Ω] [metric_space Ω] {A B C D P T B' D' : Ω} {ω : set Ω}

-- Definitions based on identified conditions
def is_tangent (l : set Ω) (c : set Ω) (p : Ω) := line p ∈ l ∧ ∀ q ∈ c, q ≠ p → dist p q = dist p c

def similar_triangles (∆ABC ∆DEF : set Ω) :=
    ∀ (A B C D E F : Ω), ∃ k : ℝ, AB1 = k * DE ∧ BC1 = k * EF ∧ CA1 = k * FD

axiom circle_points (A B C D : Ω) : A, B, C, D ∈ ω := sorry

axiom tangents_intersect (l₁ l₂ : set Ω) (B D P : Ω) (c : set Ω) : 
    is_tangent l₁ c B ∧ is_tangent l₂ c D → P ∈ l₁ ∧ P ∈ l₂ := sorry

axiom line_parallel_intersection (P T A C : Ω) (BD T' : set Ω) :
    ∃ l : set Ω, l ∥ BD ∧ T ∈ l ∧ ∀ x, x ∈ l → x ∈ A * C := sorry

-- Problem in Lean statement
theorem tangent_parallel_divide (A B C D P T B' D' : Ω) (ω : set Ω)
    (h_pts : A ∈ ω ∧ B ∈ ω ∧ C ∈ ω ∧ D ∈ ω)
    (h_tangents : is_tangent BD ω B ∧ is_tangent BD ω D)
    (h_PA : P ∈ BD)
    (h_AC : ∀ x, x ∈ A * C → T ∈ x ∧ T ∈ BD ∧ x ∥ BD)
    (h_int : ∀ x, x ∈ BD ∧ x ∈ A * C → T ∈ x ∧ ∃ {B' D'}, B' ∈ x ∧ D' ∈ {[x]})
    : ∃ (T : Ω), ∀ {B' D' : set Ω},
     T ∈ A * C ∧ BD ∥ ω ∧ D' ∈ AC ∧ B' ∈ AC → B' ∈ BD ∧ D' ∈ BD ∧ ω :=
sorry

end tangent_parallel_divide_l662_662978


namespace point_on_y_axis_l662_662481

theorem point_on_y_axis (a : ℝ) :
  (a + 2 = 0) -> a = -2 :=
by
  intro h
  sorry

end point_on_y_axis_l662_662481


namespace triangle_ABC_area_l662_662866

theorem triangle_ABC_area :
  let WXYZ_area := 64 
  let WXYZ_side := 8
  let small_square_side := 2
  let O := WXYZ_side / 2
  let BC := WXYZ_side - 2 * small_square_side
  let AM := O
  is_isosceles (triangle ABC) AB AC → 
  point_coincides_after_fold (triangle ABC) A O BC → 
  area_of_triangle ABC = 8 := by
  sorry

end triangle_ABC_area_l662_662866


namespace dot_product_l662_662427

open Real EuclideanSpace

variables {V : Type*} [EuclideanSpace V]
variables (a b : V)

theorem dot_product (h1 : ‖a‖ = sqrt 3)
                    (h2 : ‖b‖ = 2)
                    (h3 : ‖b‖ * real.cos (inner_product_angle a b) = 1) :
                    inner a b = sqrt 3 :=
by sorry

end dot_product_l662_662427


namespace find_28th_term_l662_662606

-- Define the sequence as described in the problem
def sequence (n : ℕ) : ℚ :=
  -- Calculate the smallest m such that 1 + 2 + ... + m >= n
  let m := Nat.find (λ m, (m * (m + 1)) / 2 ≥ n) in
  -- Return 1 / 2^m
  1 / 2 ^ m

-- Statement of the problem
theorem find_28th_term : sequence 28 = 1 / 2 ^ 7 :=
by {
  sorry
}

end find_28th_term_l662_662606


namespace museum_college_students_income_l662_662855

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l662_662855


namespace root_of_equation_l662_662719

theorem root_of_equation : 
  ∀ x : ℝ, x ≠ 3 → x ≠ -3 → (18 / (x^2 - 9) - 3 / (x - 3) = 2) → (x = -4.5) :=
by sorry

end root_of_equation_l662_662719


namespace general_term_an_sum_Tn_formula_l662_662062

-- Step 1: Define Sn as n + 3/2 * a_n
def Sn (n : ℕ) (a : ℕ → ℝ) : ℝ := n + 3/2 * a n

-- Step 2: Define the condition for Sn
def condition_Sn (n : ℕ) (a : ℕ → ℝ) : Prop := Sn n a = n + 3/2 * a n

-- Step 3: Define bn as an arithmetic sequence with given terms
def bn (n b_2 b_20 : ℕ) : ℕ := b_2 + (n - 2) * ((b_20 - b_2) / (20 - 2))

-- Step 4: Define the sum of bn / (an-1)
def sum_Tn (n : ℕ) (b a : ℕ → ℕ) : ℕ → ℝ 
    | 0 => 0
    | (n + 1) => (bn (n + 1) (b 2) (b 20)) / (a (n + 1) - 1) + sum_Tn n b a

theorem general_term_an (a : ℕ → ℝ) (h : ∀ n, condition_Sn n a) :
  ∀ n, a n = 1 - 3^n :=
sorry

theorem sum_Tn_formula (b : ℕ → ℕ) (a : ℕ → ℕ) (b2 b20 : ℕ)
  (hb : bn 2 b2 b20 = a 2) (hb' : bn 20 b2 b20 = a 4) :
  ∀ n, sum_Tn n b a = 3 - (2*n + 3) / 3^n :=
sorry

end general_term_an_sum_Tn_formula_l662_662062


namespace value_of_s_4_l662_662636

noncomputable def s (n : ℕ) : ℕ :=
  let squares := List.range n |>.map (λ x => (x + 1) * (x + 1))
  squares.foldl (λ acc sq => acc * 10^Nat.log10 (sq) + sq) 0

theorem value_of_s_4 : s 4 = 14916 := 
by
  sorry

end value_of_s_4_l662_662636


namespace largest_lambda_inequality_l662_662377

theorem largest_lambda_inequality :
  (∃ λ : ℝ, (∀ a b c d : ℝ, 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → a^2 + b^2 + c^2 + d^2 ≥ a * b^2 + λ * b^2 * c + c^2 * d) ∧ λ = 2) :=
sorry

end largest_lambda_inequality_l662_662377


namespace area_under_curve_l662_662200

-- Definition of the function
def f (x : ℝ) : ℝ := 3 * x^2

-- Statement about the definite integral giving the area under the curve f(x) between x = 1 and x = 2
theorem area_under_curve :
  ∫ x in 1..2, f x = 7 :=
by
  sorry

end area_under_curve_l662_662200


namespace count_valid_two_digit_integers_l662_662440

variable (d1 d2 : ℕ)
variable (digits : Finset ℕ := {1, 3, 5, 7, 8, 9})

def valid_digit_pair (d1 d2 : ℕ) : Prop :=
  d1 ∈ digits ∧ d2 ∈ digits ∧ d1 ≠ d2 ∧ ((d1 + d2) % 2 = 0)

theorem count_valid_two_digit_integers : 
  (digits.filter (λ d1, ∃ d2, valid_digit_pair d1 d2)).card = 20 :=
sorry

#eval count_valid_two_digit_integers

end count_valid_two_digit_integers_l662_662440


namespace sin_x_negative_sqrt_l662_662030

noncomputable theory

/-- Given x in (-π/2, 0) and cos(2x) = a, prove that sin(x) = -sqrt((1 - a) / 2) -/
theorem sin_x_negative_sqrt (x a : ℝ) (h₁ : x ∈ set.Ioo (-π / 2) 0) (h₂ : cos (2 * x) = a) :
  sin x = -real.sqrt ((1 - a) / 2) :=
sorry

end sin_x_negative_sqrt_l662_662030


namespace max_difference_of_segment_sums_l662_662366

theorem max_difference_of_segment_sums :
  ∀ (x : Fin 100 → ℝ), 
  ∑ i : Fin 100, x i ^ 2 = 1 → 
  let k : ℝ := ∑ (i : Fin 100) (j : Fin 100) (h : i < j ∧ (i + j) % 2 = 0), x i * x j,
      s : ℝ := ∑ (i : Fin 100) (j : Fin 100) (h : i < j ∧ (i + j) % 2 = 1), x i * x j
  in k - s ≤ 1 / 2 := sorry

end max_difference_of_segment_sums_l662_662366


namespace tan_half_angle_product_l662_662091

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end tan_half_angle_product_l662_662091


namespace area_of_circle_is_six_pi_l662_662770

noncomputable def circleArea (a : ℝ) : ℝ :=
  let radius := Real.sqrt (a^2 - 1)
  π * radius^2

theorem area_of_circle_is_six_pi (a : ℝ) (h1 : ∃ A B, ∃ y : ℝ, y = ax ∧ 
  (x^2 + y^2 - 2ax - 2y + 2 = 0) ∧
  (∃ C, C = (a, 1) ∧ 
    y = ax ∧
    triangle_equilateral C A B)) : 
  circleArea a = 6 * π := 
sorry

end area_of_circle_is_six_pi_l662_662770


namespace mark_paired_with_mike_prob_l662_662170

def total_students := 16
def other_students := 15
def prob_pairing (mark: Nat) (mike: Nat) : ℚ := 1 / other_students

theorem mark_paired_with_mike_prob : prob_pairing 1 2 = 1 / 15 := 
sorry

end mark_paired_with_mike_prob_l662_662170


namespace monochromatic_triangle_l662_662882

noncomputable def a_n (n : ℕ) : ℕ :=
  1 + n.factorial * (Finset.sum (Finset.range (n + 1)) (λ i => 1 / i.factorial))

theorem monochromatic_triangle (n : ℕ) (hn : 0 < n) :
  ∀ n : ℕ, ∃ points : Finset (ℕ × ℕ), 
    points.card = a_n n ∧ 
    (∀ p1 p2 p3 ∈ points, ¬Collinear ℝ ([p1, p2, p3] : list (ℕ × ℕ))) ∧ 
    (∃ (coloring : (points × points) → Fin n), has_monochromatic_triangle points coloring) :=
by 
  sorry

end monochromatic_triangle_l662_662882


namespace angle_CRT_l662_662132

theorem angle_CRT (h₁ : ∀ C A T : ℝ, ∠ACT = ∠ATC)
                (h₂ : ∠CAT = 36)
                (h₃ : ∀ T R : ℝ, bisect ∠ATC ∠CTR ∠CRT) :
                ∠CRT = 72 :=
by sorry

end angle_CRT_l662_662132


namespace sum_first_11_terms_l662_662237

-- Definitions for the arithmetic sequence and sum of terms
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n + 1) * (a 0 + a n) / 2

-- Given conditions
axiom a_3_plus_a_9 (a : ℕ → ℝ) : a 3 + a 9 = 16

-- To prove: S_11 = 88
theorem sum_first_11_terms (a : ℕ → ℝ) (h1 : a 3 + a 9 = 16) :
  sum_of_first_n_terms a 10 = 88 := 
sorry

end sum_first_11_terms_l662_662237


namespace eccentricity_of_hyperbola_l662_662389

noncomputable def distancePtoDirectrix (a e : ℝ) : Prop := 2 * a / (6 - e)

theorem eccentricity_of_hyperbola 
  (a b : ℝ) (h₀ : a > b) (h₁ : b > 0) 
  (h₂ : ∀ P : ℝ, P = 6 * (distancePtoDirectrix a (real.sqrt (a^2 + b^2) / a)) ) 
  (e : ℝ) 
  (h₃ : e = (real.sqrt (a^2 + b^2) / a)) 
  (h₄ : e < 6) 
  : (1 < e ∧ e ≤ 2) ∨ (3 ≤ e ∧ e < 6) := 
sorry

end eccentricity_of_hyperbola_l662_662389


namespace tomatoes_picked_l662_662305

theorem tomatoes_picked (original_tomatoes left_tomatoes picked_tomatoes : ℕ)
  (h1 : original_tomatoes = 97)
  (h2 : left_tomatoes = 14)
  (h3 : picked_tomatoes = original_tomatoes - left_tomatoes) :
  picked_tomatoes = 83 :=
by sorry

end tomatoes_picked_l662_662305


namespace smallest_n_for_pairwise_coprime_l662_662899

def is_pairwise_coprime (lst : List ℕ) : Prop := ∀ i j, i ≠ j → Nat.coprime (lst.nthLe i sorry) (lst.nthLe j sorry)

def subset_with_n_elements_contains_pairwise_coprime (n : ℕ) (S : Finset ℕ) : Prop := 
  ∀ T ⊆ S, T.card = n → ∃ lst : List ℕ, lst.to_finset = T ∧ is_pairwise_coprime lst

theorem smallest_n_for_pairwise_coprime (S : Finset ℕ) (h : S = Finset.range 281) : 
  (∀ n, subset_with_n_elements_contains_pairwise_coprime n S → 217 ≤ n) ∧ 
  subset_with_n_elements_contains_pairwise_coprime 217 S :=
by 
  sorry

end smallest_n_for_pairwise_coprime_l662_662899


namespace race_length_l662_662678

variables (x : ℝ) (t t_star : ℝ)

-- Define conditions
def anne_completion_time := t
def bronwyn_completion_distance := x - 15
def bronwyn_completion_time := bronwyn_completion_distance / t
def carl_completion_distance_anne := x - 35
def carl_completion_time_anne := carl_completion_distance_anne / t
def carl_completion_distance_bronwyn := x - 22
def carl_completion_time_bronwyn := carl_completion_distance_bronwyn / (t_star)

-- Define ratio conditions
def ratio_condition_anne_bronwyn : Prop := carl_completion_time_anne / bronwyn_completion_time = (carl_completion_distance_anne) / (bronwyn_completion_distance)
def ratio_condition_bronwyn_carl : Prop := carl_completion_time_bronwyn / bronwyn_completion_time = (13 / 15)

-- Theorem to prove
theorem race_length : ratio_condition_anne_bronwyn ∧ ratio_condition_bronwyn_carl → x = 165 :=
by
  sorry

end race_length_l662_662678


namespace solve_trig_equation_l662_662951
open Real

-- Define the original equation
def original_equation (x : ℝ) : Prop :=
  (1 / 2) * abs (cos (2 * x) + (1 / 2)) = (sin (3 * x))^2 - (sin x) * (sin (3 * x))

-- Define the correct solution set 
def solution_set (x : ℝ) : Prop :=
  ∃ k : ℤ, x = (π / 6) + (k * (π / 2)) ∨ x = -(π / 6) + (k * (π / 2))

-- The theorem we need to prove
theorem solve_trig_equation : ∀ x : ℝ, original_equation x ↔ solution_set x :=
by sorry

end solve_trig_equation_l662_662951


namespace at_least_three_babies_speak_l662_662830

structure BabySpeaking (n : ℕ) where
  prob_speak : ℝ
  prob_not_speak : ℝ
  total_babies : ℕ

noncomputable def prob_at_least_three_speak (b : BabySpeaking 6) : ℝ :=
  1 - (prob_none_speak b + prob_one_speak b + prob_two_speak b)
where
  prob_none_speak (b : BabySpeaking 6) : ℝ :=
    b.prob_not_speak ^ b.total_babies
  prob_one_speak (b : BabySpeaking 6) : ℝ :=
    (Nat.choose b.total_babies 1 : ℝ) * b.prob_speak * (b.prob_not_speak)^(b.total_babies - 1)
  prob_two_speak (b : BabySpeaking 6) : ℝ :=
    (Nat.choose b.total_babies 2 : ℝ) * (b.prob_speak^2) * (b.prob_not_speak^(b.total_babies - 2))

theorem at_least_three_babies_speak (b : BabySpeaking 6) (h : b.prob_speak = 1/3) (h' : b.prob_not_speak = 2/3) : 
  prob_at_least_three_speak b = 353 / 729 := by
  let p_none := prob_none_speak b
  let p_one := prob_one_speak b
  let p_two := prob_two_speak b
  let p_fewer_than_three := p_none + p_one + p_two
  have h1 : p_none = (2/3)^6 := sorry
  have h2 : p_one = (Nat.choose 6 1 : ℝ) * (1/3) * (2/3)^5 := sorry
  have h3 : p_two = (Nat.choose 6 2 : ℝ) * (1/3)^2 * (2/3)^4 := sorry
  have h4 : p_fewer_than_three = ((64 : ℝ) / 729) + 6 * (1/3) * (32/243) + 15 * (1/9) * (16/81) := sorry
  have h5 : p_fewer_than_three = 376 / 729 := sorry
  have h6 : prob_at_least_three_speak b = 1 - 376 / 729 := by
    exact sorry
  show prob_at_least_three_speak b = 353 / 729 from sorry

end at_least_three_babies_speak_l662_662830


namespace shortest_chord_length_l662_662079

noncomputable def line (m : ℝ) : ℝ → ℝ → Prop :=
λ x y, 2 * m * x - y - 8 * m - 3 = 0

noncomputable def circle : ℝ → ℝ → Prop :=
λ x y, x^2 + y^2 - 6 * x + 12 * y + 20 = 0

theorem shortest_chord_length (m : ℝ) : 
  (∃ x y, line m x y ∧ circle x y) ∧ (length_chord line circle = 2 * real.sqrt 15) :=
sorry

end shortest_chord_length_l662_662079


namespace sum_consecutive_integers_l662_662578

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_consecutive_integers_l662_662578


namespace line_contains_point_iff_l662_662390

theorem line_contains_point_iff (k : ℝ) :
  (\frac{3}{4} - 3 * k * (\frac{1}{3}) = 7 * (-4)) ↔ (k = 28.75) :=
by sorry

end line_contains_point_iff_l662_662390


namespace sufficientMonotonicityCondition_l662_662744

noncomputable def isMonotonicIncreasing (f : ℝ → ℝ) (D : Set ℝ) : Prop :=
  ∀ x ∈ D, ∀ y ∈ D, x ≤ y → f x ≤ f y

noncomputable def sufficientCondition (f : ℝ → ℝ) (a : ℝ) (D : Set ℝ) : Prop :=
  (∀ x ∈ D, (3/2) * a * Real.sqrt x - 1 + (a + 1) / (2 * Real.sqrt x) ≥ 0)
  ↔ a > (-3 - Real.sqrt 21) / 6

theorem sufficientMonotonicityCondition :
  ∀ (a : ℝ) (D : Set ℝ), sufficientCondition (λ x, a * x^(3/2) - x + (a + 1) * x^(1/2) - a) a D :=
by
  intros a D
  sorry

end sufficientMonotonicityCondition_l662_662744


namespace emily_spending_l662_662368

theorem emily_spending (X Y : ℝ) 
  (h1 : (X + 2*X + 3*X + 12*X) = Y) : 
  X = Y / 18 := 
by
  sorry

end emily_spending_l662_662368


namespace polar_distance_l662_662859

theorem polar_distance 
  (P : ℝ × ℝ) (C : ℝ × ℝ) (r : ℝ) 
  (hP_polar : P = (2, π / 3)) 
  (hC_polar : ∀ θ, C = (1, 0) ∧ r = 1 ∧ ρ = 2 * cos θ) :
  dist (P.to_cartesian) C = sqrt 3 :=
by
  -- Definitions of conversion from polar to Cartesian coordinates:
  let P_cart := (P.1 * cos P.2, P.1 * sin P.2)
  have hP_cart : P_cart = (1, sqrt 3), sorry
  -- Definitions of distance calculation:
  let dist := (x : ℝ) (y : ℝ) (x' : ℝ) (y' : ℝ) => sqrt ((x - x') ^ 2 + (y - y') ^ 2)
  -- Proving the distance:
  have h_dist : dist 1 (sqrt 3) 1 0 = sqrt 3, sorry
  exact h_dist

end polar_distance_l662_662859


namespace equation_of_perpendicular_line_l662_662052

-- Define the slope of the given line y = 2/3 * x
def slope_of_y_eq_2_over_3_x : ℝ := 2 / 3

-- Define the condition that the line l passes through the point (-1, 2)
def passes_through_point (a b : ℝ) (x y : ℝ) := a * x + b * y + c = 0

-- Define the condition that line l is perpendicular to the given line
def perpendicular_slope (m₁ m₂ : ℝ) := m₁ * m₂ = -1

-- Proving the equation of the line l is 3x + 2y - 1 = 0
theorem equation_of_perpendicular_line
  (m : ℝ) (x₁ y₁ : ℝ) (p₁ q₁ : ℝ)
  (h₁ : passes_through_point 3 2 (-1) 2)
  (h₂ : perpendicular_slope (-3 / 2) slope_of_y_eq_2_over_3_x) :
  3 * x₁ + 2 * y₁ - 1 = 0 :=
sorry

end equation_of_perpendicular_line_l662_662052


namespace factoring_correct_l662_662211

-- Definitions corresponding to the problem conditions
def optionA (a : ℝ) : Prop := a^2 - 5*a - 6 = (a - 6) * (a + 1)
def optionB (a x b c : ℝ) : Prop := a*x + b*x + c = (a + b)*x + c
def optionC (a b : ℝ) : Prop := (a + b)^2 = a^2 + 2*a*b + b^2
def optionD (a b : ℝ) : Prop := (a + b)*(a - b) = a^2 - b^2

-- The main theorem that proves option A is the correct answer
theorem factoring_correct : optionA a := by
  sorry

end factoring_correct_l662_662211


namespace two_digit_product_l662_662220

theorem two_digit_product (x y : ℕ) (h₁ : 10 ≤ x) (h₂ : x < 100) (h₃ : 10 ≤ y) (h₄ : y < 100) (h₅ : x * y = 4320) :
  (x = 60 ∧ y = 72) ∨ (x = 72 ∧ y = 60) :=
sorry

end two_digit_product_l662_662220


namespace rice_flour_weights_l662_662297

variables (r f : ℝ)

theorem rice_flour_weights :
  (8 * r + 6 * f = 550) ∧ (4 * r + 7 * f = 375) → (r = 50) ∧ (f = 25) :=
by
  intro h
  sorry

end rice_flour_weights_l662_662297


namespace allWhiteSquaresAreOne_l662_662838

-- Define a chessboard size
def n : ℕ := 1983
def m : ℕ := 1984

-- Define the type for a chessboard cell: white or black
inductive CellType
| White : CellType
| Black : CellType

-- Define a function that returns the cell type based on position
def cellType (i j : ℕ) : CellType :=
  if (i + j) % 2 == 0 then CellType.White else CellType.Black

-- Define the type for the value in a white cell
def ValueInWhiteCell : Type := ℤ

-- Define that each white cell is filled with either 1 or -1
def valueOfWhiteCell (i j : ℕ) : ValueInWhiteCell :=
  if cellType i j = CellType.White then
    1 -- Define possible values (1 or -1)
  else 
    0 -- Black cells have no defined value for our condition

-- The key condition: 
-- For every black square, the product of the adjacent white squares is 1.
axiom blackCellAdjacentProduct (i j : ℕ) (h: cellType i j = CellType.Black) :
  (∏ (di dj : ℤ) in [(1,0), (-1,0), (0,1), (0,-1)], valueOfWhiteCell (i + di.toNat) (j + dj.toNat)) = 1

-- The theorem to prove: all numbers in the white squares are 1
theorem allWhiteSquaresAreOne :
  ∀ (i j : ℕ), cellType i j = CellType.White → valueOfWhiteCell i j = 1 :=
sorry

end allWhiteSquaresAreOne_l662_662838


namespace even_numbers_in_table_l662_662173

theorem even_numbers_in_table
  (a b c : ℕ)
  (h1 : nat.coprime a b)
  (h2 : nat.coprime b c)
  (h3 : nat.coprime c a)
  (table : ℕ → ℕ → ℤ)
  (h_even_sums : ∀ i j, ( ∑ x in finset.range a, ∑ y in finset.range a, table (i * a + x) (j * a + y) ) % 2 = 0
     ∧ ( ∑ x in finset.range b, ∑ y in finset.range b, table (i * b + x) (j * b + y) ) % 2 = 0
     ∧ ( ∑ x in finset.range c, ∑ y in finset.range c, table (i * c + x) (j * c + y) ) % 2 = 0)
  : ∀ i j, table i j % 2 = 0 :=
sorry

end even_numbers_in_table_l662_662173


namespace smallest_d_exists_and_is_optimal_l662_662386

/-- Define the notion of n, a positive integer greater than 1 -/
def is_valid_n (n : ℕ) : Prop :=
  n > 1

/-- Define d-coverable as per the conditions given -/
def is_d_coverable (n : ℕ) (d : ℕ) (S : finset ℕ) : Prop :=
  ∃ P : polynomial ℤ, P.nat_degree ≤ d ∧ 
    (S = (finset.Ico 0 n).filter (λ x, ∃ k : ℤ, (x : ℤ) ≡ P.eval k [MOD n]))

/-- Prove that the smallest d exists for n = 4 or n is a prime, with d = 3 for n = 4 
and d = n - 1 for prime n -/
theorem smallest_d_exists_and_is_optimal (n : ℕ) (h : is_valid_n n) :
  (n = 4 ∨ (nat.prime n)) →
  ∃ d : ℕ, ∀ S : finset ℕ, S.nonempty → (S ⊆ (finset.Ico 0 n)) → is_d_coverable n d S ∧
    (∀ d' : ℕ, d' < d → ¬ (∀ S : finset ℕ, S.nonempty → (S ⊆ (finset.Ico 0 n)) → is_d_coverable n d' S)) :=
by {
  sorry
}

end smallest_d_exists_and_is_optimal_l662_662386


namespace price_difference_correct_l662_662912

noncomputable def original_price_80_percent (P : ℝ) := 0.80 * P
noncomputable def down_payment (new_price : ℝ) := 0.25 * new_price
noncomputable def remaining_amount (new_price : ℝ) := 0.75 * new_price
noncomputable def price_difference (new_price old_price : ℝ) := new_price - old_price

theorem price_difference_correct :
  ∀ (P : ℝ),
  let new_price := 30000 in
  let down_payment_amount := down_payment new_price in
  let remaining_to_finance := remaining_amount new_price in
  remaining_to_finance = 22500 →
  0.80 * P + 4000 = remaining_to_finance →
  P = 23125 →
  price_difference new_price P = 6875 :=
by
  intros,
  sorry

end price_difference_correct_l662_662912


namespace constant_in_diamond_area_l662_662209

theorem constant_in_diamond_area :
  ∃ C : ℝ, (∀ x y : ℝ, |x / 5| + |y / 5| = C → region_area x y = 200) ↔ C = 2 :=
by 
  sorry

-- Definition of the area of the diamond-shaped region
def region_area (x y : ℝ) : ℝ :=
  let d1 := 10 * (abs ((y / 5)))
  let d2 := 10 * (abs ((x / 5)))
  (d1 * d2) / 2

end constant_in_diamond_area_l662_662209


namespace roof_area_l662_662225

-- Definitions of the roof's dimensions based on the given conditions.
def length (w : ℝ) := 4 * w
def width (w : ℝ) := w
def difference (l w : ℝ) := l - w
def area (l w : ℝ) := l * w

-- The proof problem: Given the conditions, prove the area is 576 square feet.
theorem roof_area : ∀ w : ℝ, (length w) - (width w) = 36 → area (length w) (width w) = 576 := by
  intro w
  intro h_diff
  sorry

end roof_area_l662_662225


namespace inscribed_circle_distance_l662_662530

def right_triangle (X Y Z : ℝ × ℝ) : Prop :=
  ∃ x y z : ℝ, X = (0, 0) ∧ Y = (x, 0) ∧ Z = (0, y) ∧ x^2 + y^2 = z^2

def inscribed_circle (T : Type) [topological_space T] (t : T) :=
  ∀ (p : T), t.distance p = T.radius

def construct_MN {T : Type} [topological_space T] 
  (C1 : T) (L1 L2 : subtype T) : Prop :=
  ∃ M N : T, collinear M N ∧ perpendicular_to L1 L2 ∧ tangent_to C1 (line_through M N)

def construct_AB {T : Type} [topological_space T] 
  (C1 : T) (L1 L2 : subtype T) : Prop :=
  ∃ A B : T, collinear A B ∧ perpendicular_to L1 L2 ∧ tangent_to C1 (line_through A B)

theorem inscribed_circle_distance
  (X Y Z : ℝ × ℝ)
  (h1 : right_triangle X Y Z)
  (hMZ : construct_MN (inscribed_circle Z) X Z)
  (hYB : construct_AB (inscribed_circle Z) Y Z)
  : ∃ p : ℝ, p = 988.45 :=
sorry

end inscribed_circle_distance_l662_662530


namespace arithmetic_sequence_problem_l662_662484

noncomputable def a_n (n : ℕ) : ℚ := 1 + (n - 1) / 2

noncomputable def S_n (n : ℕ) : ℚ := n * (n + 3) / 4

theorem arithmetic_sequence_problem :
  -- Given
  (∀ n, ∃ d, a_n n = a_1 + (n - 1) * d) →
  (a_n 7 = 4) →
  (a_n 19 = 2 * a_n 9) →
  -- Prove
  (∀ n, a_n n = (n + 1) / 2) ∧ (∀ n, S_n n = n * (n + 3) / 4) :=
by
  sorry

end arithmetic_sequence_problem_l662_662484


namespace cyclic_quadrilateral_fourth_side_lengths_l662_662222

noncomputable def possible_fourth_sides (BC CD DA : ℝ) : ℝ → Prop :=
  λ x, (x = 2 ∨ x = 5 ∨ x = 10 ∨ x = 14.4)

theorem cyclic_quadrilateral_fourth_side_lengths (AB BC CD DA : ℝ) (h_cyclic : ∃ (O : Point), is_cyclic_quad O A B C D) 
    (h_areas : triangle_area A B C = triangle_area A C D) 
    (h_sides : BC = 5 ∧ CD = 6 ∧ DA = 12)
    (possible_sides : ∃ (x : ℝ), possible_fourth_sides BC CD DA x ∧ x = AB) : 
  possible_fourth_sides BC CD DA AB :=
by
  sorry

end cyclic_quadrilateral_fourth_side_lengths_l662_662222


namespace distance_focus_line_l662_662740

theorem distance_focus_line (p : ℝ) (h : p > 0) (h1 : p / 2 = 4) :
  ∃ (L : ℝ → ℝ), (L(-1) = 0) ∧ 
                  (∀ x y, y = L x → x^2 = 2*p*y → 
                  (x*x - 2*p*y - 2*p*L(x) + 2*p*(-1) = 0)) ∧
                  (let delta := 256*(L(-1))^2 + 64*(L(-1)) in
                   delta = 0 → (p := 8) ∧ 
                   (x := 0) ∧ 
                   (y := 4) ∧ 
                   (d1 := abs(x + 1)) ∧ 
                   (d2 := abs(4 - (-1/4 + 1/4))) ∧ 
                   (d3 := real.sqrt((x - 1)^2 + (4 - 1/4 * (-1 - 1))^2)) ∧ 
                   (d1 = 1 ∨ d2 = 4 ∨ d3 = real.sqrt(17))) := sorry

end distance_focus_line_l662_662740


namespace smallest_b_for_composite_l662_662698

theorem smallest_b_for_composite (x : ℤ) : 
  ∃ b : ℕ, b > 0 ∧ Even b ∧ (∀ x : ℤ, ¬ Prime (x^4 + b^2)) ∧ b = 16 := 
by 
  sorry

end smallest_b_for_composite_l662_662698


namespace total_points_scored_l662_662172

theorem total_points_scored (m1 m2 m3 m4 m5 m6 j1 j2 j3 j4 j5 j6 : ℕ) :
  m1 = 5 → j1 = m1 + 2 →
  m2 = 7 → j2 = m2 - 3 →
  m3 = 10 → j3 = m3 / 2 →
  m4 = 12 → j4 = m4 * 2 →
  m5 = 6 → j5 = m5 →
  j6 = 8 → m6 = j6 + 4 →
  m1 + m2 + m3 + m4 + m5 + m6 + j1 + j2 + j3 + j4 + j5 + j6 = 106 :=
by
  intros
  sorry

end total_points_scored_l662_662172


namespace geometric_series_inequality_l662_662622

theorem geometric_series_inequality (n : ℕ) (h : n ≥ 8) : 
  (1 + ∑ i in finset.range(n - 1), (1 / (2^i : ℝ))) > (127 / 64) :=
by 
  sorry

end geometric_series_inequality_l662_662622


namespace soccer_game_attendance_difference_l662_662673

theorem soccer_game_attendance_difference :
  (abs (55000 - 40000 * 0.92).toInt + 999) / 1000 * 1000 = 18000 :=
by 
  let s_min := 40000 * 0.92
  let s_max := 40000 * 1.08
  let b := 55000
  let largest_possible_difference := max (abs (b - s_min)).toInt (abs (b - s_max)).toInt
  have h : largest_possible_difference = (abs (b - s_min)).toInt from sorry
  exact h.trans (by norm_num)

end soccer_game_attendance_difference_l662_662673


namespace max_value_of_a_l662_662102

theorem max_value_of_a (a : ℝ) (h : ∀ (x : ℝ), x > 1 → a - x + Real.log(x * (x + 1)) ≤ 0) :
  a ≤ (1 + Real.sqrt(3)) / 2 - Real.log((3 / 2) + Real.sqrt(3)) :=
by
  sorry

end max_value_of_a_l662_662102


namespace number_of_true_false_questions_is_six_l662_662959

variable (x : ℕ)
variable (num_true_false num_free_response num_multiple_choice total_problems : ℕ)

axiom problem_conditions :
  (num_free_response = x + 7) ∧ 
  (num_multiple_choice = 2 * (x + 7)) ∧ 
  (total_problems = 45) ∧ 
  (total_problems = x + num_free_response + num_multiple_choice)

theorem number_of_true_false_questions_is_six (h : problem_conditions x num_true_false num_free_response num_multiple_choice total_problems) : 
  x = 6 :=
  sorry

end number_of_true_false_questions_is_six_l662_662959


namespace domain_of_function_interval_l662_662592

noncomputable def domain_of_function (x : ℝ) : Set ℝ := 
  {x | ∃ y, y = 1 / (Real.sqrt (Real.logb 0.5 (4 * x - 3)))}

theorem domain_of_function_interval (x : ℝ) :
  domain_of_function x = Ioo (3 / 4) 1 :=
by
  sorry

end domain_of_function_interval_l662_662592


namespace number_of_even_factors_of_m_l662_662445

def m : ℕ := 2^3 * 3^2 * 5^1

noncomputable def number_of_even_factors (n : ℕ) : ℕ :=
  (n.factors.count 2 > 0).to_nat * (3 * 2)  -- From solution, even factors depend on a (3 values) × b (3 values) × c (2 values)

theorem number_of_even_factors_of_m :
  number_of_even_factors m = 18 :=
by
  sorry

end number_of_even_factors_of_m_l662_662445


namespace CSRT_is_rhombus_and_area_equal_l662_662870

noncomputable def parallelogram (A B C D : Point) : Prop :=
  -- Definition for a parallelogram

noncomputable def perpendicular (P Q R : Point) : Prop :=
  -- Definition for perpendicular points

theorem CSRT_is_rhombus_and_area_equal (A B C D P Q R S T : Point)
  (hParallelogram : parallelogram A B C D)
  (hPerpendicular1 : perpendicular C P Q)
  (hIntersection1 : intersects P A B)
  (hIntersection2 : intersects Q A D)
  (hPerpendicular2 : perpendicular R S T)
  (hSemicircle : semicircle P Q R) :
  -- Prove that CSRT is a rhombus
  -- Prove that area of CSRT equals area of ABCD :=
sorry

end CSRT_is_rhombus_and_area_equal_l662_662870


namespace double_x_value_l662_662093

theorem double_x_value (x : ℝ) (h : x / 2 = 32) : 2 * x = 128 := by
  sorry

end double_x_value_l662_662093


namespace no_integer_solution_for_system_l662_662697

theorem no_integer_solution_for_system :
  (¬ ∃ x y : ℤ, 18 * x + 27 * y = 21 ∧ 27 * x + 18 * y = 69) :=
by
  sorry

end no_integer_solution_for_system_l662_662697


namespace concatenated_prime_impossible_l662_662579

theorem concatenated_prime_impossible:
  ∀ (l : List ℕ), l = List.range (2124 - 2024 + 1) (λ i => i + 2024) →
  (∀ p : ℕ, p = List.foldr (λ a b, a * 10^4 + b) 0 l → Prime p) →
  False :=
by
  sorry

end concatenated_prime_impossible_l662_662579


namespace least_number_of_roots_l662_662657

noncomputable def f : ℝ → ℝ := sorry

lemma symmetric_at_3 : ∀ x, f (3 + x) = f (3 - x) := sorry
lemma symmetric_at_8 : ∀ x, f (8 + x) = f (8 - x) := sorry
lemma at_zero : f 0 = 2 := sorry

theorem least_number_of_roots : 
  ∃ n ≥ 0, ∀ x ∈ Icc (-1010 : ℝ) 1010, f(x) = 2 ↔ (∃ k : ℤ, x = 10 * k) ∧ n = 203 := 
sorry

end least_number_of_roots_l662_662657


namespace amy_balloons_l662_662505

theorem amy_balloons (james_balloons amy_balloons : ℕ) (h1 : james_balloons = 1222) (h2 : james_balloons = amy_balloons + 709) : amy_balloons = 513 :=
by
  sorry

end amy_balloons_l662_662505


namespace total_brownies_l662_662570

theorem total_brownies (brought_to_school left_at_home : ℕ) (h1 : brought_to_school = 16) (h2 : left_at_home = 24) : 
  brought_to_school + left_at_home = 40 := 
by 
  sorry

end total_brownies_l662_662570


namespace power_inequality_l662_662145

theorem power_inequality (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hcb : c ≥ b) : 
  a^b * (a + b)^c > c^b * a^c := 
sorry

end power_inequality_l662_662145


namespace pure_imaginary_value_l662_662552

theorem pure_imaginary_value (m : ℝ) (i : ℂ) (h_imaginary_unit : i * i = -1)
    (h : m + (10 / (3 + i)) = -imaginary_part of pure complex number * i) :
    m = -3 :=
begin
  sorry
end

end pure_imaginary_value_l662_662552


namespace domain_of_f_l662_662212

theorem domain_of_f {f : ℝ → ℝ} (h1 : ∀ x ∈ ℝ, x ≠ 0 → (1/x) ∈ ℝ) (h2 : ∀ x ∈ ℝ, x ≠ 0 → f(x) + f(1/x) = x) :
  ∀ x ∈ ℝ, (x = -1 ∨ x = 1) ↔ ∃ D, D = {-1, 1} ∧ ∀ x, x ∈ D → x ∈ ℝ ∧ (1/x) ∈ ℝ :=
sorry

end domain_of_f_l662_662212


namespace circumscribed_circle_diameter_l662_662457

-- Define the inputs: side length and angle of the triangle
def sideLength : ℝ := 14
def angle : ℝ := (Real.pi / 4)  -- 45 degrees in radians

-- Calculate the diameter using the extended law of sines
noncomputable def diameter (a : ℝ) (A : ℝ) : ℝ := a / Real.sin A

-- Statement of the theorem we need to prove
theorem circumscribed_circle_diameter :
  diameter sideLength angle = 14 * Real.sqrt 2 :=
by
  sorry

end circumscribed_circle_diameter_l662_662457


namespace value_of_g_800_l662_662532

variable (g : ℝ → ℝ)
variable (h : ∀ (x y : ℝ), 0 < x → 0 < y → g(x * y) = g(x) * y)
variable (h200 : g 200 = 4)

theorem value_of_g_800 : g 800 = 16 :=
by
  sorry

end value_of_g_800_l662_662532


namespace isosceles_triangle_circumradius_l662_662287

noncomputable def circumscribed_radius (a b : ℝ) : ℝ :=
  (a + Real.sqrt (a^2 - 4 * b))^2 / (4 * a * Real.sqrt (a^2 - 4 * b))

theorem isosceles_triangle_circumradius (a b : ℝ) :
  (∀ x : ℝ, x^2 - a * x + b = 0 → x = (a + Real.sqrt (a^2 - 4 * b)) / 2 ∨ x = (a - Real.sqrt (a^2 - 4 * b)) / 2) →
  ∃ R : ℝ, R = circumscribed_radius a b :=
by
  intro h
  use circumscribed_radius a b
  sorry

end isosceles_triangle_circumradius_l662_662287


namespace amina_wins_is_21_over_32_l662_662674

/--
Amina and Bert alternate turns tossing a fair coin. Amina goes first and each player takes three turns.
The first player to toss a tail wins. If neither Amina nor Bert tosses a tail, then neither wins.
Prove that the probability that Amina wins is \( \frac{21}{32} \).
-/
def amina_wins_probability : ℚ :=
  let p_first_turn := 1 / 2
  let p_second_turn := (1 / 2) ^ 3
  let p_third_turn := (1 / 2) ^ 5
  p_first_turn + p_second_turn + p_third_turn

theorem amina_wins_is_21_over_32 :
  amina_wins_probability = 21 / 32 :=
sorry

end amina_wins_is_21_over_32_l662_662674


namespace probability_of_covering_black_region_l662_662607

noncomputable def probability_covering_black_region : ℚ :=
  (78 + 5 * Real.pi + 12 * Real.sqrt 2) / 64

theorem probability_of_covering_black_region :
  ∃ P : ℚ, P = probability_covering_black_region :=
by
  use (78 + 5 * Real.pi + 12 * Real.sqrt 2) / 64
  sorry

end probability_of_covering_black_region_l662_662607


namespace gcd_891_810_l662_662979

theorem gcd_891_810 : Nat.gcd 891 810 = 81 := 
by
  sorry

end gcd_891_810_l662_662979


namespace length_of_other_train_is_correct_l662_662279

noncomputable def length_of_second_train 
  (length_first_train : ℝ) 
  (speed_first_train : ℝ) 
  (speed_second_train : ℝ) 
  (time_to_cross : ℝ) 
  : ℝ := 
  let speed_first_train_m_s := speed_first_train * (1000 / 3600)
  let speed_second_train_m_s := speed_second_train * (1000 / 3600)
  let relative_speed := speed_first_train_m_s + speed_second_train_m_s
  let total_distance := relative_speed * time_to_cross
  total_distance - length_first_train

theorem length_of_other_train_is_correct :
  length_of_second_train 250 120 80 9 = 249.95 :=
by
  unfold length_of_second_train
  simp
  sorry

end length_of_other_train_is_correct_l662_662279


namespace ratio_of_circumference_to_area_l662_662651

def radius := 10
def circumference (r : ℝ) := 2 * Real.pi * r
def area (r : ℝ) := Real.pi * r^2

theorem ratio_of_circumference_to_area : 
  (circumference radius) / (area radius) = 1 / 5 := 
by
  sorry

end ratio_of_circumference_to_area_l662_662651


namespace mean_of_all_numbers_l662_662216

theorem mean_of_all_numbers
  (a : Fin 7 → ℝ)
  (b : Fin 8 → ℝ)
  (h1 : (∑ i, a i) / 7 = 15)
  (h2 : (∑ i, b i) / 8 = 30) :
  (∑ i, a i + ∑ i, b i) / 15 = 23 := sorry

end mean_of_all_numbers_l662_662216


namespace deductive_reasoning_is_option_A_l662_662275

-- Define the types of reasoning.
inductive ReasoningType
| Deductive
| Analogical
| Inductive

-- Define the options provided in the problem.
def OptionA : ReasoningType := ReasoningType.Deductive
def OptionB : ReasoningType := ReasoningType.Analogical
def OptionC : ReasoningType := ReasoningType.Inductive
def OptionD : ReasoningType := ReasoningType.Inductive

-- Statement to prove that Option A is Deductive reasoning.
theorem deductive_reasoning_is_option_A : OptionA = ReasoningType.Deductive := by
  -- proof
  sorry

end deductive_reasoning_is_option_A_l662_662275


namespace percentage_given_away_l662_662694

theorem percentage_given_away
  (initial_bottles : ℕ)
  (drank_percentage : ℝ)
  (remaining_percentage : ℝ)
  (gave_away : ℝ):
  initial_bottles = 3 →
  drank_percentage = 0.90 →
  remaining_percentage = 0.70 →
  gave_away = initial_bottles - (drank_percentage * 1 + remaining_percentage) →
  (gave_away / 2) / 1 * 100 = 70 :=
by
  intros
  sorry

end percentage_given_away_l662_662694


namespace maximize_probability_l662_662083

def A : Set ℕ := {1, 2}
def B : Set ℕ := {1, 2, 3}
def points : Set (ℕ × ℕ) := { (a, b) | a ∈ A ∧ b ∈ B }
def C_n (n : ℕ) : Set (ℕ × ℕ) := { (a, b) | a + b = n }

theorem maximize_probability : 
  ∀ n : ℕ, n = 3 ∨ n = 4 ↔ 
  (∀ m : ℕ, (m = n → (m = 3 ∨ m = 4)) ∧ 
  (∀ p : ℕ × ℕ, p ∈ points → (p ∈ C_n m → m = 3 ∨ m = 4)) :=
sorry

end maximize_probability_l662_662083


namespace evaluate_expression_l662_662164

noncomputable def a : ℝ := Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def b : ℝ := -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 6
noncomputable def c : ℝ := Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6
noncomputable def d : ℝ := -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 6

theorem evaluate_expression :
  ( (1 / a) + (1 / b) + (1 / c) + (1 / d) ) ^ 2 = 96 / 529 :=
by
  sorry

end evaluate_expression_l662_662164


namespace sum_f_2023_l662_662751

/-- Given a function g(x) that is both a quadratic function and a power function, 
    and f(x) = x^3 / (g(x) + 1) + 1, the sum of f evaluated at each integer from 
    -2023 to 2023 equals 4047. -/
theorem sum_f_2023 (g : ℝ → ℝ) (f : ℝ → ℝ) 
  (hg_quad : ∃ a : ℝ, ∀ x : ℝ, g(x) = a * x^2) 
  (hf : ∀ x : ℝ, f(x) = x^3 / (g(x) + 1) + 1) : 
  (finset.sum (finset.range (2 * 2023 + 1)) (λ i, f (i - 2023))) = 4047 :=
sorry

end sum_f_2023_l662_662751


namespace excircle_midpoint_congruence_l662_662340

open EuclideanGeometry

-- Definitions and conditions based on problem
variables {A B C P Q R : Point}

-- Assume necessary geometric structures and conditions
def triangle (A B C : Point) : Prop := ¬ Collinear A B C

def excircle_touches (A B C P Q R : Point) : Prop :=
  -- excircle touches AB at P, extension of AC at Q, and extension of BC at R
  -- Define using appropriate geometric properties
  sorry

def midpoint (X Y M : Point) := dist X M = dist Y M

def circumcircle_contains (A B C X : Point) : Prop :=
  -- Define circumcircle of triangle ABC containing point X
  sorry

-- Variables for midpoints
variable {M N : Point}

-- The resulting Lean 4 theorem statement
theorem excircle_midpoint_congruence
  (h_triangle : triangle A B C)
  (h_excircle : excircle_touches A B C P Q R)
  (h_midpoint_N : midpoint P Q N)
  (h_on_circum_N : circumcircle_contains A B C N) :
  ∃ M, midpoint P R M ∧ circumcircle_contains A B C M :=
sorry

end excircle_midpoint_congruence_l662_662340


namespace angle_at_3_15_l662_662789

theorem angle_at_3_15 : 
  let minute_hand_angle := 90.0
  let hour_at_3 := 90.0
  let hour_hand_at_3_15 := hour_at_3 + 7.5
  minute_hand_angle == 90.0 ∧ hour_hand_at_3_15 == 97.5 →
  abs (hour_hand_at_3_15 - minute_hand_angle) = 7.5 :=
by
  sorry

end angle_at_3_15_l662_662789


namespace smaller_angle_at_315_l662_662784

def full_circle_degrees : ℝ := 360
def hours_on_clock : ℕ := 12
def degrees_per_hour : ℝ := full_circle_degrees / hours_on_clock
def minute_position_at_315 : ℝ := 3 * degrees_per_hour
def hour_position_at_315 : ℝ := 3 * degrees_per_hour + degrees_per_hour / 4

theorem smaller_angle_at_315 :
  minute_position_at_315 = 90 → 
  hour_position_at_315 = 3 * degrees_per_hour + degrees_per_hour / 4 → 
  abs (hour_position_at_315 - minute_position_at_315) = 7.5 :=
by 
  intro h_minute h_hour 
  rw [h_minute, h_hour]
  sorry

end smaller_angle_at_315_l662_662784


namespace tan_x_is_correct_l662_662775

-- Define the vectors a and b as given in the conditions
def vector_a (x : ℝ) : ℝ × ℝ := (Real.sin x, Real.cos x)
def vector_b : ℝ × ℝ := (2, -3)

-- Define the condition that vector_a is parallel to vector_b
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

-- The main theorem to prove
theorem tan_x_is_correct (x : ℝ) (h : are_parallel (vector_a x) vector_b) :
  Real.tan x = -2/3 :=
by
  sorry

end tan_x_is_correct_l662_662775


namespace cricket_player_innings_l662_662302

theorem cricket_player_innings (n : ℕ) (T : ℕ) 
  (h1 : T = n * 48) 
  (h2 : T + 178 = (n + 1) * 58) : 
  n = 12 :=
by
  sorry

end cricket_player_innings_l662_662302


namespace smallest_positive_integer_l662_662265

theorem smallest_positive_integer (n : ℕ) (h : n > 0) : (sqrt n - sqrt (n - 1 : ℕ)) < 0.05 ↔ n = 101 := by
  sorry

end smallest_positive_integer_l662_662265


namespace convex_quadrilateral_ratio_l662_662839

theorem convex_quadrilateral_ratio (A B C D : Type) [convex_quadrilateral A B C D]
  (H1 : angle_eq B D)
  (H2 : CD = 4 * BC)
  (H3 : angle_bisector A passes_through_midpoint (midpoint CD)) :
  ratio AD AB = 2 / 3 := by
  sorry

end convex_quadrilateral_ratio_l662_662839


namespace problem_part1_problem_part2_l662_662115

-- Vectors and their conditions
variables {V : Type*} [inner_product_space ℝ V]
variables (a b c d : V)
variable (m n : ℝ)

-- Define the conditions
def conditions (a b c d : V) (m n : ℝ) : Prop :=
  (a + b + c + d = 0) ∧
  (⟪a, b⟫ = m) ∧ (⟪b, c⟫ = m) ∧
  (⟪c, d⟫ = n) ∧ (⟪d, a⟫ = n)

-- Proof statement for part 1
theorem problem_part1 (h : conditions a b c d m m) : 
  (is_rectangle a b c d) :=
sorry

-- Proof statement for part 2
theorem problem_part2 (h : conditions a b c d m n) (hmn : m ≠ n) : 
  (is_isosceles_trapezoid a b c d) :=
sorry

end problem_part1_problem_part2_l662_662115


namespace sin_A_value_of_triangle_l662_662037

theorem sin_A_value_of_triangle 
  (a b : ℝ) (A B C : ℝ) (h_triangle : a = 2) (h_b : b = 3) (h_tanB : Real.tan B = 3) :
  Real.sin A = Real.sqrt 10 / 5 :=
sorry

end sin_A_value_of_triangle_l662_662037


namespace sum_of_roots_is_twelve_l662_662593

noncomputable def g : ℝ → ℝ := sorry
axiom symmetry_property (x : ℝ) : g(3 + x) = g(3 - x)
axiom four_distinct_roots : ∃ a b c d : ℝ, (g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0) ∧ 
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)

theorem sum_of_roots_is_twelve :
  (∃a b c d : ℝ, g a = 0 ∧ g b = 0 ∧ g c = 0 ∧ g d = 0 ∧ 
  (a = 3 + y ∧ b = 3 - y ∧ c = 3 + z ∧ d = 3 - z) ∧ 
  y ≠ z ∧ y ≠ 0 ∧ z ≠ 0) → a + b + c + d = 12 :=
by
  sorry

end sum_of_roots_is_twelve_l662_662593


namespace problem_solution_l662_662722

variables {R : Type} [LinearOrder R]

def M (x y : R) : R := max x y
def m (x y : R) : R := min x y

theorem problem_solution (p q r s t : R) (h : p < q) (h1 : q < r) (h2 : r < s) (h3 : s < t) :
  M (M p (m q r)) (m s (M p t)) = q :=
by
  sorry

end problem_solution_l662_662722


namespace angle_of_inclination_l662_662759

theorem angle_of_inclination (m : ℝ) (h : m = -1) : 
  ∃ α : ℝ, α = 3 * Real.pi / 4 := 
sorry

end angle_of_inclination_l662_662759


namespace cubics_product_equals_1_over_1003_l662_662612

theorem cubics_product_equals_1_over_1003
  (x_1 y_1 x_2 y_2 x_3 y_3 : ℝ)
  (h1 : x_1^3 - 3 * x_1 * y_1^2 = 2007)
  (h2 : y_1^3 - 3 * x_1^2 * y_1 = 2006)
  (h3 : x_2^3 - 3 * x_2 * y_2^2 = 2007)
  (h4 : y_2^3 - 3 * x_2^2 * y_2 = 2006)
  (h5 : x_3^3 - 3 * x_3 * y_3^2 = 2007)
  (h6 : y_3^3 - 3 * x_3^2 * y_3 = 2006) :
  (1 - x_1 / y_1) * (1 - x_2 / y_2) * (1 - x_3 / y_3) = 1 / 1003 :=
by
  sorry

end cubics_product_equals_1_over_1003_l662_662612


namespace round_4_36_to_nearest_tenth_l662_662187

-- Definitions based on the conditions
def number_to_round : ℝ := 4.36
def round_half_up (x : ℝ) : ℝ := (↑(Int(10 * x) + if 10 * x - Int(10 * x) >= 0.5 then 1 else 0)) / 10

-- The assertion to prove
theorem round_4_36_to_nearest_tenth : round_half_up number_to_round = 4.4 :=
by
  -- Proof will go here
  sorry

end round_4_36_to_nearest_tenth_l662_662187


namespace magnitude_vector_sum_l662_662413

variables (OA OB OC : ℝ^3)
variables (hOA : |OA| = 1) (hOB : |OB| = 1) (hOC : |OC| = 1)
variables (h1 : (OA • OB) = 1/2)
variables (h2 : (OA • OC) = 1/2)
variables (h3 : (OB • OC) = 1/2)

theorem magnitude_vector_sum : |OA + OB + OC| = real.sqrt 6 :=
sorry

end magnitude_vector_sum_l662_662413


namespace find_real_numbers_l662_662009

theorem find_real_numbers (x1 x2 x3 x4 : ℝ) :
  x1 + x2 * x3 * x4 = 2 →
  x2 + x1 * x3 * x4 = 2 →
  x3 + x1 * x2 * x4 = 2 →
  x4 + x1 * x2 * x3 = 2 →
  (x1 = 1 ∧ x2 = 1 ∧ x3 = 1 ∧ x4 = 1) ∨ 
  (x1 = -1 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = 3) ∨
  (x1 = -1 ∧ x2 = -1 ∧ x3 = 3 ∧ x4 = -1) ∨
  (x1 = -1 ∧ x2 = 3 ∧ x3 = -1 ∧ x4 = -1) ∨
  (x1 = 3 ∧ x2 = -1 ∧ x3 = -1 ∧ x4 = -1) :=
by sorry

end find_real_numbers_l662_662009


namespace largest_set_of_triangles_is_9_l662_662664

def is_valid_triangle (a b c : ℕ) : Prop :=
  a < 7 ∧ b < 7 ∧ c < 7 ∧ (b + c > a) ∧ (a + c > b) ∧ (a + b > c) ∧
  ¬(a * 2 <= b) ∧ ¬(b * 2 <= c) ∧ ¬(c * 2 <= a)

def unique_up_to_similarity (T : set (ℕ × ℕ × ℕ)) : Prop :=
  ∀ (x y : ℕ × ℕ × ℕ), x ∈ T → y ∈ T → x ≠ y → 
  let (a1, b1, c1) := x in
  let (a2, b2, c2) := y in
  (a1 ≠ k * a2 ∨ b1 ≠ k * b2 ∨ c1 ≠ k * c2) → (k : ℕ)

def maximum_number_of_valid_triangles (n : ℕ) : Prop :=
  ∃ (S : set (ℕ × ℕ × ℕ)), (∀ (x : ℕ × ℕ × ℕ), x ∈ S → is_valid_triangle x.1 x.2 x.3) ∧ 
  unique_up_to_similarity S ∧ S.card = n

theorem largest_set_of_triangles_is_9 : maximum_number_of_valid_triangles 9 :=
sorry

end largest_set_of_triangles_is_9_l662_662664


namespace inverse_sum_l662_662896

def g (x : ℝ) : ℝ :=
  if x ≤ 2 then 3 - x else 3 * x - x ^ 2

theorem inverse_sum : g⁻¹ (-4) + g⁻¹ 0 + g⁻¹ 5 = 5 := 
by sorry

end inverse_sum_l662_662896


namespace angle_ABD_l662_662844

theorem angle_ABD (A B C D E F : Type)
  (quadrilateral : Prop)
  (angle_ABC : ℝ)
  (angle_BDE : ℝ)
  (angle_BDF : ℝ)
  (h1 : quadrilateral)
  (h2 : angle_ABC = 120)
  (h3 : angle_BDE = 30)
  (h4 : angle_BDF = 28) :
  (180 - angle_ABC = 60) :=
by
  sorry

end angle_ABD_l662_662844


namespace rowing_time_one_hour_l662_662658

noncomputable def total_time_to_travel (Vm Vr distance : ℝ) : ℝ :=
  let upstream_speed := Vm - Vr
  let downstream_speed := Vm + Vr
  let one_way_distance := distance / 2
  let time_upstream := one_way_distance / upstream_speed
  let time_downstream := one_way_distance / downstream_speed
  time_upstream + time_downstream

theorem rowing_time_one_hour : 
  total_time_to_travel 8 1.8 7.595 = 1 := 
sorry

end rowing_time_one_hour_l662_662658


namespace magnitude_inverse_sum_eq_l662_662539

noncomputable def complex_magnitude_inverse_sum (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
|1/z + 1/w|

theorem magnitude_inverse_sum_eq (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_magnitude_inverse_sum z w hz hw hzw = 3 / 8 :=
by sorry

end magnitude_inverse_sum_eq_l662_662539


namespace find_f_neg_100_l662_662159

noncomputable def f (x : ℝ) : ℝ :=
if 0 ≤ x ∧ x < 7 then log (9 - x) / log 2 else 1 -- default value for non-specified range

axiom f_property : ∀ x : ℝ, f (x + 3) * f (x - 4) = -1

theorem find_f_neg_100 : f (-100) = -1 / 2 :=
by
  sorry

end find_f_neg_100_l662_662159


namespace vector_collinear_k_l662_662414

theorem vector_collinear_k {a b : Vector} (hna : a ≠ 0) (hnb : b ≠ 0) (hanc : ∃ c : ℝ, a = c • b) :
  ∃ k : ℝ, (8 • a - k • b) = l • (-k • a + b) → k = 2 * real.sqrt 2 ∨ k = -2 * real.sqrt 2 :=
sorry

end vector_collinear_k_l662_662414


namespace solve_crease_length_l662_662660

noncomputable def lengthCrease (a b c e g : ℝ) : ℝ := 
  let distance : ℝ := real.sqrt ((e - 0)^2 + (g - 6)^2)
  distance

theorem solve_crease_length :
  ∃ creaseLength : ℝ, creaseLength = real.sqrt 7.336806 :=
begin
  use (real.sqrt 7.336806),
  sorry
end

end solve_crease_length_l662_662660


namespace equivalent_functions_l662_662274

theorem equivalent_functions :
  (∀ x: ℝ, f x = real.sqrt ((1 + x) / (1 - x))) ∧
  (∀ t: ℝ, g t = real.sqrt ((1 + t) / (1 - t))) ∧
  ( domain(f) = domain(g) )
  → (f = g) :=
sorry

end equivalent_functions_l662_662274


namespace eccentricity_of_ellipse_l662_662208

theorem eccentricity_of_ellipse :
  ∀ (x y : ℝ), (x^2) / 25 + (y^2) / 16 = 1 → 
  (∃ (e : ℝ), e = 3 / 5) :=
by
  sorry

end eccentricity_of_ellipse_l662_662208


namespace point_D_in_fourth_quadrant_l662_662628

def is_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

def point_A : ℝ × ℝ := (1, 2)
def point_B : ℝ × ℝ := (-1, -2)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (1, -2)

theorem point_D_in_fourth_quadrant : is_in_fourth_quadrant (point_D.1) (point_D.2) :=
by
  sorry

end point_D_in_fourth_quadrant_l662_662628


namespace eval_expr_l662_662161

def max_power_of_2_factor_200 : ℕ := 3
def max_power_of_5_factor_200 : ℕ := 2
def expr : ℝ := (1 / 3) ^ (max_power_of_5_factor_200 - max_power_of_2_factor_200)

theorem eval_expr : expr = 3 := by
  sorry

end eval_expr_l662_662161


namespace cyclic_quadrilateral_fourth_side_lengths_l662_662221

noncomputable def possible_fourth_sides (BC CD DA : ℝ) : ℝ → Prop :=
  λ x, (x = 2 ∨ x = 5 ∨ x = 10 ∨ x = 14.4)

theorem cyclic_quadrilateral_fourth_side_lengths (AB BC CD DA : ℝ) (h_cyclic : ∃ (O : Point), is_cyclic_quad O A B C D) 
    (h_areas : triangle_area A B C = triangle_area A C D) 
    (h_sides : BC = 5 ∧ CD = 6 ∧ DA = 12)
    (possible_sides : ∃ (x : ℝ), possible_fourth_sides BC CD DA x ∧ x = AB) : 
  possible_fourth_sides BC CD DA AB :=
by
  sorry

end cyclic_quadrilateral_fourth_side_lengths_l662_662221


namespace f_1000_eq_999_l662_662355

-- Definitions for f and the conditions given
def f : ℕ+ → ℕ+ -- ℕ+ is positive integers

axiom f_f (n : ℕ+) : f(f(n)) = 3 * n
axiom f_4n2 (n : ℕ+) : f(4 * n + 2) = 4 * n + 3

-- Theorem to prove
theorem f_1000_eq_999 : f (1000) = 999 := by
  sorry

end f_1000_eq_999_l662_662355


namespace increasing_condition_min_value_a_eq_one_l662_662450

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (-1/3) * x^3 + (1/2) * x^2 + 2 * x

theorem increasing_condition (a : ℝ) : 
  (∀ x > 2/3, ((x - 1/2)^2 + 1/4 + 2 * a > 0)) → a > -1/8 :=
by
  intro h
  sorry 

theorem min_value_a_eq_one : 
  ∀ x, 1 ≤ x ∧ x ≤ 4 → (-1/3)*4^3 + (1/2)*4^2 + 2 * 4 = (-16/3) :=
by
  sorry

end increasing_condition_min_value_a_eq_one_l662_662450


namespace problem_statement_l662_662423

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ x : ℝ, f(x + 3) = -f(x)) (h2 : f 4 = -2) : f 2011 = 2 :=
sorry

end problem_statement_l662_662423


namespace relationship_among_a_b_c_l662_662604

noncomputable def a : ℝ := 0.4^2
noncomputable def b : ℝ := Real.log 0.4 / Real.log 2  -- log base 2
noncomputable def c : ℝ := 2^0.4

theorem relationship_among_a_b_c : b < a ∧ a < c :=
by
  sorry

end relationship_among_a_b_c_l662_662604


namespace tetrahedron_is_regular_regular_tetrahedron_has_spheres_l662_662402

-- Given a tetrahedron SABC
variables {S A B C : Type}

-- 5 spheres tangent to edges of a tetrahedron
variable exists_spheres : ∃ (Ω₁ Ω₂ Ω₃ Ω₄ Ω₅ : Sphere), 
  (∀ (P Q R K L M N : Type), 
    (P = edge SA ∧ Q = edge SB ∧ R = edge SC ∧ K = edge AB ∧ L = edge BC ∧ M = edge CA) ∨
    (P = extension SA ∧ Q = extension SB ∧ R = extension SC ∧ K = edge AB ∧ L = edge BC ∧ M = edge CA))

-- The first theorem: SABC is a regular tetrahedron
theorem tetrahedron_is_regular (h : exists_spheres) : is_regular_tetrahedron S A B C := sorry

-- The second theorem: every regular tetrahedron has exactly 5 such spheres
theorem regular_tetrahedron_has_spheres (h : is_regular_tetrahedron S A B C) : 
  ∃ (Ω₁ Ω₂ Ω₃ Ω₄ Ω₅ : Sphere), 
    (∀ (P Q R K L M N : Type), 
      (P = edge SA ∧ Q = edge SB ∧ R = edge SC ∧ K = edge AB ∧ L = edge BC ∧ M = edge CA) ∨ 
      (P = extension SA ∧ Q = extension SB ∧ R = extension SC ∧ K = edge AB ∧ L = edge BC ∧ M = edge CA)) := sorry

end tetrahedron_is_regular_regular_tetrahedron_has_spheres_l662_662402


namespace equalize_candies_l662_662245

theorem equalize_candies :
  ∀ (candies : List ℕ),
    (candies = [7, 8, 9, 11, 20] →
    ¬ (∃ (moves : List (ℕ × ℕ × ℕ)) (h_moves_len : moves.length = 2), 
        redistribute candies moves = [11, 11, 11, 11, 11]) ∧
    (∃ (moves : List (ℕ × ℕ × ℕ)) (h_moves_len : moves.length = 3), 
        redistribute candies moves = [11, 11, 11, 11, 11])) :=
by
  sorry

def redistribute (candies : List ℕ) (moves : List (ℕ × ℕ × ℕ)) : List ℕ :=
  sorry

end equalize_candies_l662_662245


namespace tan_is_odd_function_l662_662359

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x ∈ ℝ, f (-x) = -f (x)

def tan_function (x : ℝ) := Real.tan x

theorem tan_is_odd_function (k : ℤ) :
  ∀ x ∈ set.Ioo (-Real.pi / 2 + k * Real.pi) (Real.pi / 2 + k * Real.pi), is_odd_function tan_function :=
sorry

end tan_is_odd_function_l662_662359


namespace sequence_term_value_l662_662166

/-- Let {a_n} and {b_n} be arithmetic sequences, a_1 = 25, b_1 = 75, and a_2 + b_2 = 100. -/
def arithmetic_seq (a b : ℕ → ℕ) : Prop :=
  ∃ d₁ d₂ : ℕ, ∀ n : ℕ, a (n + 1) = a n + d₁ ∧ b (n + 1) = b n + d₂

theorem sequence_term_value (a b : ℕ → ℕ)
  (h1 : a 1 = 25)
  (h2 : b 1 = 75)
  (h3 : a 2 + b 2 = 100)
  (h4 : arithmetic_seq a b) :
  a 37 + b 37 = 100 :=
begin
  sorry
end

end sequence_term_value_l662_662166


namespace hyperbola_eccentricity_l662_662128

def hyperbola_equation (a b : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ (∃ x y : ℝ, (x ^ 2) / (a ^ 2) - (y ^ 2) / (b ^ 2) = 1)

def circle_intersects_asymptotes (a b : ℝ) (c : ℝ) : Prop :=
  a = b ∧ a = sqrt (2 * c)

theorem hyperbola_eccentricity (a b c e : ℝ) :
  hyperbola_equation a b →
  circle_intersects_asymptotes a b c →
  e = (c / a) →
  e = sqrt 2 := 
by 
  intros h_conditions h_intersects h_eccentricity
  sorry

end hyperbola_eccentricity_l662_662128


namespace second_candidate_votes_l662_662118

theorem second_candidate_votes (total_votes : ℕ) (first_candidate_percentage : ℝ) 
    (total_votes_condition : total_votes = 1200) (first_percentage_condition : first_candidate_percentage = 0.80) 
    : ℕ :=
  let second_candidate_percentage := 1 - first_candidate_percentage in
  let second_candidate_votes := (second_candidate_percentage * total_votes : ℝ).toNat in
  second_candidate_votes

example : second_candidate_votes 1200 0.80 1200 rfl = 240 := by
  unfold second_candidate_votes
  sorry

end second_candidate_votes_l662_662118


namespace find_x_l662_662903

def g (x : ℝ) : ℝ := 5 * x - 10
def g_inv (x : ℝ) : ℝ := (x + 10) / 5

theorem find_x : ∃ x : ℝ, g x = g_inv x ∧ x = 5 / 2 := by
  sorry

end find_x_l662_662903


namespace number_of_true_false_questions_is_six_l662_662958

variable (x : ℕ)
variable (num_true_false num_free_response num_multiple_choice total_problems : ℕ)

axiom problem_conditions :
  (num_free_response = x + 7) ∧ 
  (num_multiple_choice = 2 * (x + 7)) ∧ 
  (total_problems = 45) ∧ 
  (total_problems = x + num_free_response + num_multiple_choice)

theorem number_of_true_false_questions_is_six (h : problem_conditions x num_true_false num_free_response num_multiple_choice total_problems) : 
  x = 6 :=
  sorry

end number_of_true_false_questions_is_six_l662_662958


namespace min_L_shaped_trios_l662_662259

-- Definition of a 5x5 grid
def grid : Type := ℕ × ℕ

-- Conditions to define a non-overlapping "L-shaped" trio in the grid (just a placeholder for the type)
def L_shaped_trio (g : grid) : Prop := sorry 

-- Definition of what it means for "L-shaped" trios to not overlap (placeholder for now)
def non_overlapping (t1 t2 : L_shaped_trio grid) : Prop := sorry

-- Minimal number of non-overlapping "L-shaped" trios on a 5x5 grid is 4
theorem min_L_shaped_trios : ∃ t1 t2 t3 t4 : L_shaped_trio grid,
  non_overlapping t1 t2 ∧ non_overlapping t1 t3 ∧ non_overlapping t1 t4 ∧
  non_overlapping t2 t3 ∧ non_overlapping t2 t4 ∧
  non_overlapping t3 t4 ∧
  ∀ t : L_shaped_trio grid,
    ¬(non_overlapping t t1 ∧ non_overlapping t t2 ∧ non_overlapping t t3 ∧ non_overlapping t t4) :=
sorry

end min_L_shaped_trios_l662_662259


namespace coefficient_x_110_equals_neg30_l662_662010

-- Define the polynomial as the product of terms (x^i - i) for i from 1 to 15.
def polynomial := ∏ i in Finset.range 15, (X ^ (i + 1) - (i + 1))

-- Declare that the expansion of this polynomial has a term with x^110
noncomputable def coefficient_of_x_110_in_expansion : ℤ :=
  polynomial.coeff 110

-- Prove that the coefficient of x^110 in the expansion is equal to -30
theorem coefficient_x_110_equals_neg30 : coefficient_of_x_110_in_expansion = -30 :=
by
  sorry

end coefficient_x_110_equals_neg30_l662_662010


namespace greatest_length_of_each_piece_l662_662169

def rope_lengths : list ℕ := [48, 72, 96, 120]

theorem greatest_length_of_each_piece (h1: ∀ x ∈ rope_lengths, x % 24 = 0) : (48.gcd 72).gcd (96.gcd 120) = 24 :=
by
  sorry

end greatest_length_of_each_piece_l662_662169


namespace plums_total_count_l662_662917

theorem plums_total_count : 
    let melanie_plums := 4
    let dan_plums := 9
    let sally_plums := 3
in melanie_plums + dan_plums + sally_plums = 16 := 
by
  simp [melanie_plums, dan_plums, sally_plums]
  sorry

end plums_total_count_l662_662917


namespace true_false_questions_count_l662_662957

noncomputable def number_of_true_false_questions (T F M : ℕ) : Prop :=
  T + F + M = 45 ∧ M = 2 * F ∧ F = T + 7

theorem true_false_questions_count : ∃ T F M : ℕ, number_of_true_false_questions T F M ∧ T = 6 :=
by
  sorry

end true_false_questions_count_l662_662957


namespace isosceles_triangle_angle_l662_662103

noncomputable def isosceles_triangle_vertex_angle (a : ℝ) : Prop :=
  ∃ v : ℝ, v = 70 ∨ v = 40

theorem isosceles_triangle_angle {a : ℝ} (h : a = 70) : isosceles_triangle_vertex_angle a :=
begin
  sorry
end

end isosceles_triangle_angle_l662_662103


namespace range_of_x_for_f_gt_1_l662_662148

theorem range_of_x_for_f_gt_1 :
  (∀ x : ℝ, f(x) = log ((2 / (1 - x)) - 1) → f(x) > 1) → 
  (∀ x : ℝ, f(x) > 1 ↔ (9 / 11 < x ∧ x < 1)) :=
begin
  sorry
end

end range_of_x_for_f_gt_1_l662_662148


namespace simplify_f_evaluate_f_with_condition_l662_662087

noncomputable def f (α : ℝ) : ℝ := (2 * (Real.cos α)^2 - Real.sin (2 * α)) / (2 * (Real.cos α - Real.sin α))

theorem simplify_f (α : ℝ) : 
  f(α) = Real.cos α := by
  sorry

theorem evaluate_f_with_condition (α : ℝ) (h1 : α ∈ (π / 2, π)) (h2 : Real.sin α = 3 / 5) : 
  f(α + π / 6) = -(4 * Real.sqrt 3 + 3) / 10 := by
  sorry

end simplify_f_evaluate_f_with_condition_l662_662087


namespace rounding_4_36_to_nearest_tenth_l662_662188

theorem rounding_4_36_to_nearest_tenth : (round (10 * 4.36) / 10) = 4.4 :=
by
  sorry

end rounding_4_36_to_nearest_tenth_l662_662188


namespace circle_equation_l662_662398

noncomputable def chord_length_y_axis := 2
noncomputable def arc_ratio_x_axis := 3 / 1
noncomputable def distance_center_line := sqrt(5) / 5

theorem circle_equation (a b r : ℝ) (h1 : chord_length_y_axis = 2)
  (h2 : arc_ratio_x_axis = 3 / 1) 
  (h3 : distance_center_line = sqrt(5) / 5)
  (ha : 2 * b^2 - a^2 = 1)
  (hb1 : a - 2 * b = 1)
  (hb2 : a - 2 * b = -1)
  : ((x + 1)^2 + (y + 1)^2 = 2) ∨ ((x - 1)^2 + (y - 1)^2 = 2) := sorry

end circle_equation_l662_662398


namespace extreme_value_at_zero_monotonically_decreasing_on_interval_l662_662433

-- Define the function f(x) = (3*x^2 + m*x) / exp(x)
def f (x m : ℝ) : ℝ := (3 * x^2 + m * x) / Real.exp x

-- 1. Show that f(x) has an extreme value at x = 0 if and only if m = 0
theorem extreme_value_at_zero (m : ℝ) : 
  (∀ x, (deriv (λ x, (3 * x^2 + m * x) / Real.exp x)) x = 0 → x = 0) ↔ m = 0 :=
begin
    sorry
end

-- 2. Show that f(x) is monotonically decreasing on [3, +∞) if -3 ≤ m ≤ 2
theorem monotonically_decreasing_on_interval (m : ℝ) : 
  -3 ≤ m ∧ m ≤ 2 ↔ ∀ x ∈ set.Ici (3 : ℝ), (deriv (λ x, (3 * x^2 + m * x) / Real.exp x)) x ≤ 0 :=
begin
    sorry
end

end extreme_value_at_zero_monotonically_decreasing_on_interval_l662_662433


namespace probability_at_least_3_speak_l662_662833

noncomputable def probability_speaks (n : ℕ) (p : ℚ) : ℚ :=
  ∑ k in finset.range (n + 1), if k < 3 then (nat.choose n k) * (p^k) * ((1 - p)^(n-k)) else 0

theorem probability_at_least_3_speak :
  let p := (1 : ℚ) / 3,
      n := 6 in
  1 - probability_speaks n p = 233 / 729 :=
by {
  let p := 1/3,
  let n := 6,
  have h₀ : ∑ k in finset.range (n + 1), if k < 3 then (nat.choose n k) * (p^k) * ((1 - p)^(n-k)) else 0 = 496 / 729 := sorry,
  rw [probability_speaks, h₀],
  norm_num,
}

end probability_at_least_3_speak_l662_662833


namespace degree_of_g_l662_662152

open Polynomial

def f : Polynomial ℝ := -7 * X^4 + 3 * X^3 + X - 5

theorem degree_of_g (g : Polynomial ℝ)
  (hdeg : (f + g).degree = 2) :
  g.degree = 4 :=
sorry

end degree_of_g_l662_662152


namespace koby_boxes_l662_662879

theorem koby_boxes (x : ℕ) (sparklers_per_box : ℕ := 3) (whistlers_per_box : ℕ := 5) 
    (cherie_sparklers : ℕ := 8) (cherie_whistlers : ℕ := 9) (total_fireworks : ℕ := 33) : 
    (sparklers_per_box * x + cherie_sparklers) + (whistlers_per_box * x + cherie_whistlers) = total_fireworks → x = 2 :=
by
  sorry

end koby_boxes_l662_662879


namespace exponent_evaluation_l662_662885

theorem exponent_evaluation {a b : ℕ} (h₁ : 2 ^ a ∣ 200) (h₂ : ¬ (2 ^ (a + 1) ∣ 200))
                           (h₃ : 5 ^ b ∣ 200) (h₄ : ¬ (5 ^ (b + 1) ∣ 200)) :
  (1 / 3) ^ (b - a) = 3 :=
by sorry

end exponent_evaluation_l662_662885


namespace problem_statement_l662_662765

def f (x : ℝ) : ℝ := 3 * Real.sin (x - π / 3) + 1

noncomputable def is_monotonic_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y ∈ I, x < y → f x ≤ f y

theorem problem_statement :
  (∀ x : ℝ, f x = 3 * Real.sin (x - π / 3) + 1) ∧
  (∀ k : ℤ, is_monotonic_increasing f {x | 2 * k * π - π / 6 ≤ x ∧ x ≤ 2 * k * π + 5 * π / 6}) ∧
  (∀ m : ℝ, (∃ x ∈ Set.Icc (π / 6) (7 * π / 6), f x = m + 1) ↔ m ∈ Set.Icc (-3/2 : ℝ) 3) :=
by
  sorry

end problem_statement_l662_662765


namespace conjugate_of_i_2015_l662_662897

-- Definitions based on conditions
def imaginary_unit : ℂ := Complex.I
def pow_two : imaginary_unit ^ 2 = -1 := by simp [Complex.I_pow_two]
def pow_four : imaginary_unit ^ 4 = 1 := by simp [Complex.I_pow_four]

-- Proving the main statement
theorem conjugate_of_i_2015 : Complex.conj (imaginary_unit ^ 2015) = imaginary_unit :=
by sorry

end conjugate_of_i_2015_l662_662897


namespace taco_truck_income_l662_662667

-- Definitions based on conditions
variables {x y : ℕ} -- cost of soft and hard shell tacos are natural numbers
variables {n : ℕ}  -- number of additional customers is a natural number

-- Main theorem/proof problem
theorem taco_truck_income (x y n : ℕ) : 
  let family_income := 4 * y + 3 * x in
  let additional_customers_income := 2 * n * x in
  let total_income := family_income + additional_customers_income in
  total_income = 4 * y + 3 * x + 2 * n * x :=
by
  sorry

end taco_truck_income_l662_662667


namespace remainder_of_poly_division_l662_662020

def poly_div_remainder (f g : ℚ[X]) : Prop :=
  ∃ q r : ℚ[X], f = g * q + r ∧ r.degree < g.degree

theorem remainder_of_poly_division :
  poly_div_remainder (3 * X^5 - 2 * X^3 + 5 * X - 8) (X^2 + 3 * X + 2) (84 * X + 70) :=
by {
  use start_poly_div_remainder
  sorry
}

end remainder_of_poly_division_l662_662020


namespace remainder_y_div_13_l662_662269

def x (k : ℤ) : ℤ := 159 * k + 37
def y (x : ℤ) : ℤ := 5 * x^2 + 18 * x + 22

theorem remainder_y_div_13 (k : ℤ) : (y (x k)) % 13 = 8 := by
  sorry

end remainder_y_div_13_l662_662269


namespace integer_solution_pair_l662_662708

theorem integer_solution_pair (x y : ℤ) (h : x^2 + x * y = y^2) : (x = 0 ∧ y = 0) :=
by
  sorry

end integer_solution_pair_l662_662708


namespace total_number_of_players_l662_662121

theorem total_number_of_players (n : ℕ) (h1 : n > 7) 
  (h2 : (4 * (n * (n - 1)) / 3 + 56 = (n + 8) * (n + 7) / 2)) : n + 8 = 50 :=
by
  sorry

end total_number_of_players_l662_662121


namespace solve_equation_l662_662952

theorem solve_equation (x : ℝ) (h : x = 5) :
  (3 * x - 5) / (x^2 - 7 * x + 12) + (5 * x - 1) / (x^2 - 5 * x + 6) = (8 * x - 13) / (x^2 - 6 * x + 8) := 
  by 
  rw [h]
  sorry

end solve_equation_l662_662952


namespace log_function_odd_symmetry_origin_l662_662213

theorem log_function_odd_symmetry_origin :
  ∀ x : ℝ, x ∈ Set.Ioo (-2 : ℝ) 2 →
    (log 2 ((2 - x) / (2 + x)) = - log 2 ((2 + x) / (2 - x))) ∧
    (y = log 2 ((2 - x) / (2 + x)) → y = - y) :=
by sorry

end log_function_odd_symmetry_origin_l662_662213


namespace range_of_a_l662_662761

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) - cos (2 * x)

theorem range_of_a (a : ℝ) (h₀ : 0 < a) 
  (h₁ : ∀ x : ℝ, 0 ≤ x ∧ x ≤ a / 3 → f' x > 0)
  (h₂ : ∀ x : ℝ, 2 * a ≤ x ∧ x ≤ 4 * π / 3 → f' x > 0) :
  a ∈ set.Icc (5 * π / 12) π :=
sorry

end range_of_a_l662_662761


namespace ceil_x_pow_2_values_l662_662817

theorem ceil_x_pow_2_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ n, n = 29 ∧ (∀ y, ceil (y^2) = ⌈x^2⌉ → 196 < y^2 ∧ y^2 ≤ 225) :=
sorry

end ceil_x_pow_2_values_l662_662817


namespace area_of_right_triangle_integers_l662_662935

theorem area_of_right_triangle_integers (a b c : ℤ) (h : a^2 + b^2 = c^2) :
  ∃ (A : ℤ), A = (a * b) / 2 := 
sorry

end area_of_right_triangle_integers_l662_662935


namespace ellipse_equation_correct_hyperbola_equation_correct_l662_662713

noncomputable def c_ellipse : ℝ := real.sqrt (9 - 4)
noncomputable def a_hyperbola : ℝ := real.sqrt 25
noncomputable def b_hyperbola : ℝ := real.sqrt (a_hyperbola^2 - c_ellipse^2)

def passes_through (curve: ℝ → ℝ → Prop) (pt: ℝ × ℝ) : Prop :=
  curve pt.1 pt.2

def is_focus_of (curve: ℝ → ℝ → Prop) (focus: ℝ × ℝ) : Prop :=
  ∃ a b : ℝ, (curve = λ x y, x^2 / a^2 + y^2 / b^2 = 1) ∧
    (focus = (real.sqrt (a^2 - b^2), 0))

def has_asymptotes (curve: ℝ → ℝ → Prop) (slope1 slope2: ℝ) : Prop :=
  ∃ a b : ℝ, (curve = λ x y, y^2 / b^2 - x^2 / a^2 = 1) ∧
    (slope1 = b / a ∧ slope2 = -b / a)

def ellipse_25_20 : ℝ → ℝ → Prop :=
  λ x y, x^2 / 25 + y^2 / 20 = 1

def hyperbola_3_12 : ℝ → ℝ → Prop :=
  λ y x, y^2 / 3 - x^2 / 12 = 1

theorem ellipse_equation_correct :
  is_focus_of ellipse_25_20 (c_ellipse, 0) ∧
  passes_through ellipse_25_20 (- real.sqrt 5, 4) :=
sorry

theorem hyperbola_equation_correct :
  has_asymptotes hyperbola_3_12 (1/2) (-1/2) ∧ 
  passes_through hyperbola_3_12 (2, 2) :=
sorry

end ellipse_equation_correct_hyperbola_equation_correct_l662_662713


namespace egor_last_payment_l662_662003

theorem egor_last_payment (a b c d : ℕ) (h_sum : a + b + c + d = 28)
  (h1 : b ≥ 2 * a) (h2 : c ≥ 2 * b) (h3 : d ≥ 2 * c) : d = 18 := by
  sorry

end egor_last_payment_l662_662003


namespace volleyball_match_prob_A_win_l662_662473

-- Definitions of given probabilities and conditions
def rally_scoring_system := true
def first_to_25_wins := true
def tie_at_24_24_continues_until_lead_by_2 := true
def prob_team_A_serves_win : ℚ := 2/3
def prob_team_B_serves_win : ℚ := 2/5
def outcomes_independent := true
def score_22_22_team_A_serves := true

-- The problem to prove
theorem volleyball_match_prob_A_win :
  rally_scoring_system ∧
  first_to_25_wins ∧
  tie_at_24_24_continues_until_lead_by_2 ∧
  prob_team_A_serves_win = 2/3 ∧
  prob_team_B_serves_win = 2/5 ∧
  outcomes_independent ∧
  score_22_22_team_A_serves →
  (prob_team_A_serves_win ^ 3 + (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win ^ 2 + prob_team_A_serves_win * (1 - prob_team_A_serves_win) * prob_team_B_serves_win * prob_team_A_serves_win + prob_team_A_serves_win ^ 2 * (1 - prob_team_A_serves_win) * prob_team_B_serves_win) = 64/135 :=
by
  sorry

end volleyball_match_prob_A_win_l662_662473


namespace value_of_S_2016_l662_662482

variable (a d : ℤ)
variable (S : ℕ → ℤ)

-- Definitions of conditions
def a_1 := -2014
def sum_2012 := S 2012
def sum_10 := S 10
def S_n (n : ℕ) : ℤ := n * a_1 + (n * (n - 1) / 2) * d

-- Given conditions
axiom S_condition : (sum_2012 / 2012) - (sum_10 / 10) = 2002
axiom S_def : ∀ n : ℕ, S n = S_n n

-- The theorem to be proved
theorem value_of_S_2016 : S 2016 = 2016 := by
  sorry

end value_of_S_2016_l662_662482


namespace angle_between_hands_at_3_15_l662_662780

theorem angle_between_hands_at_3_15 :
  let hour_angle_at_3 := 3 * 30
  let hour_hand_move_rate := 0.5
  let minute_angle := 15 * 6
  let hour_angle_at_3_15 := hour_angle_at_3 + 15 * hour_hand_move_rate
  abs (hour_angle_at_3_15 - minute_angle) = 7.5 := 
by
  sorry

end angle_between_hands_at_3_15_l662_662780


namespace ratio_of_areas_l662_662650

theorem ratio_of_areas (w : ℝ) (h : ℝ) (H1 : h = 1.5 * w) : 
  let initial_rectangle_area := w * h,
      smaller_circle_area := π * (sqrt (13) / 4 * w) ^ 2,
      ratio := smaller_circle_area / initial_rectangle_area
  in ratio = 13 * π / 24 :=
sorry

end ratio_of_areas_l662_662650


namespace find_AB_distance_l662_662124

noncomputable def l1_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos α, t * Real.sin α)

noncomputable def l2_parametric (t α : ℝ) : ℝ × ℝ :=
  (t * Real.cos (α + π / 4), t * Real.sin (α + π / 4))

noncomputable def curve_C_polar (θ : ℝ) : ℝ :=
  4 * Real.cos θ

theorem find_AB_distance (α : ℝ) (hα : 0 < α ∧ α < 3 * π / 4) :
  let A := l1_parametric (curve_C_polar α) α,
      B := l2_parametric (curve_C_polar (α + π/4)) (α + π/4) in
  Real.dist (A.1, A.2) (B.1, B.2) = 2 * Real.sqrt 2 := sorry

end find_AB_distance_l662_662124


namespace nth_monomial_equation_l662_662663

theorem nth_monomial_equation (n : ℕ) : 
  (∀ (n : ℕ), monomial n = (-1)^(n+1) * 2 * n * x^n) := 
by sorry

end nth_monomial_equation_l662_662663


namespace monotonic_on_interval_even_function_minimum_value_cases_l662_662431

variable (f : ℝ → ℝ) (a x : ℝ)

-- Define the function f(x) = x^2 + 2ax
def func (a x : ℝ) : ℝ := x^2 + 2 * a * x

-- Prove if f(x) = x^2 + 2ax is monotonic on [-5, 5], then a <= -5 or a >= 5
theorem monotonic_on_interval (h : monotonic_on (func a) (set.Icc (-5) 5)) : a <= -5 ∨ a >= 5 := 
sorry

-- Prove if y = f(x) - 2x is even, then a = 1
theorem even_function (h : ∀ x, func a x - 2 * x = func a (-x) - 2 * (-x)) : a = 1 := 
sorry

-- Maximum and minimum values of f(x) for a = 1 on [-5, 5]
lemma max_min_values_a_eq_1 : 
  let f := func 1 in 
  max_on f (set.Icc (-5) 5) f 5 = 35 ∧ min_on f (set.Icc (-5) 5) f (-1) = -1 := 
sorry

-- Minimum value of f(x) for x ∈ [-5, 5] given different ranges of a
theorem minimum_value_cases : 
  (∀ a, (a >= 5 → (min_on (func a) (set.Icc (-5) 5) (func a (-5)) = 25 - 10 * a)) 
    ∧ (a <= -5 → (min_on (func a) (set.Icc 5 (-5)) (func a 5) = 25 + 10 * a))
    ∧ (-5 < a ∧ a < 5 → (min_on (func a) (set.Icc (-5) 5) (func a (-a)) = -a^2))) :=
sorry

end monotonic_on_interval_even_function_minimum_value_cases_l662_662431


namespace locus_of_vertex_P_l662_662406

noncomputable def M : ℝ × ℝ := (0, 5)
noncomputable def N : ℝ × ℝ := (0, -5)
noncomputable def perimeter : ℝ := 36

theorem locus_of_vertex_P : ∃ (P : ℝ × ℝ), 
  (∃ (a b : ℝ), a = 13 ∧ b = 12 ∧ P ≠ (0,0) ∧
  (a^2 = b^2 + 5^2) ∧ 
  (perimeter = 2 * a + (5 - (-5))) ∧ 
  ((P.1)^2 / 144 + (P.2)^2 / 169 = 1)) :=
sorry

end locus_of_vertex_P_l662_662406


namespace spongebob_price_l662_662955

variable (x : ℝ)

theorem spongebob_price (h : 30 * x + 12 * 1.5 = 78) : x = 2 :=
by
  -- Given condition: 30 * x + 12 * 1.5 = 78
  sorry

end spongebob_price_l662_662955


namespace conjugate_of_z_l662_662824

theorem conjugate_of_z (z : ℂ) (h : z * (1 + complex.I) = 2 + 4 * complex.I) : conj z = 3 - complex.I :=
sorry

end conjugate_of_z_l662_662824


namespace sum_sequence_S_l662_662772

def S (n : ℕ) : ℤ :=
  ∑ k in Finset.range n, ((-1 : ℤ) ^ (k + 1)) * (4 * (k + 1) - 3)

theorem sum_sequence_S :
  S 15 + S 22 - S 31 = -76 := by
  sorry

end sum_sequence_S_l662_662772


namespace PA_dot_PB_l662_662158

noncomputable def P (x0 : ℝ) : ℝ × ℝ := (x0, x0 + (2 / x0))
noncomputable def A (x0 : ℝ) : ℝ × ℝ := (x0 + (1 / x0), x0 + (1 / x0))
noncomputable def B (x0 : ℝ) : ℝ × ℝ := (0, x0 + (2 / x0))

theorem PA_dot_PB (x0 : ℝ) (hx0 : x0 > 0) : 
  let PA := (x0 + 1/x0 - x0, x0 + 1/x0 - (x0 + 2/x0))
  let PB := (0 - x0, (x0 + 2/x0) - x0)
  inner_product PA PB = -1 :=
by
  sorry

end PA_dot_PB_l662_662158


namespace moon_carbon_percentage_l662_662599

theorem moon_carbon_percentage :
  let moon_weight := 250
  let iron_percentage := 0.5
  let mars_weight := 2 * moon_weight
  let mars_other_elements := 150
  let moon_other_elements := mars_other_elements / 2
  let moon_iron := iron_percentage * moon_weight
  let remaining_weight := moon_weight - (moon_other_elements + moon_iron)
  let carbon_percentage := (remaining_weight / moon_weight) * 100
  carbon_percentage = 20 :=
by
  let moon_weight := 250
  let iron_percentage := 0.5
  let mars_weight := 2 * moon_weight
  let mars_other_elements := 150
  let moon_other_elements := mars_other_elements / 2
  let moon_iron := iron_percentage * moon_weight
  let remaining_weight := moon_weight - (moon_other_elements + moon_iron)
  let carbon_percentage := (remaining_weight / moon_weight) * 100
  show carbon_percentage = 20, from sorry

end moon_carbon_percentage_l662_662599


namespace usual_time_is_75_l662_662253

variable (T : ℕ) -- let T be the usual time in minutes

theorem usual_time_is_75 (h1 : (6 * T) / 5 = T + 15) : T = 75 :=
by
  sorry

end usual_time_is_75_l662_662253


namespace find_f_l662_662396

noncomputable def f (x : ℝ) : ℝ := x^3 - 2 * (deriv f 1) * x

theorem find_f'_one : deriv f 1 = 1 :=
sorry

end find_f_l662_662396


namespace probability_of_freezing_l662_662299

def fair_die : set ℕ := {1, 2, 3, 4, 5, 6}

def is_valid_roll (roll : ℕ × ℕ) : Prop :=
  roll.1 ∈ fair_die ∧ roll.2 ∈ fair_die

def successful_outcomes : set (ℕ × ℕ) :=
  {(2, 6), (3, 5), (4, 4), (5, 3), (6, 2)}

theorem probability_of_freezing : 
  (∑' outcome in {outcome | is_valid_roll outcome ∧ outcome ∈ successful_outcomes}, 1) = (5 / 36) :=
by
  sorry

end probability_of_freezing_l662_662299


namespace number_of_elements_in_S_l662_662774

open Set

noncomputable def setP : Set ℝ := { x | x^2 - 5 * x + 4 ≥ 0 }

noncomputable def setQ : Set ℕ := { n | n^2 - 5 * n + 4 ≤ 0 ∧ n > 0 }

def setS (S : Set ℕ) := (S ∩ setP.to_nat = {1, 4}) ∧ (S ∩ setQ = S)

theorem number_of_elements_in_S (S : Set ℕ) (hS : setS S) : 
  (2 ≤ S.card ∧ S.card ≤ 4) :=
sorry

end number_of_elements_in_S_l662_662774


namespace width_of_canal_at_bottom_l662_662969

theorem width_of_canal_at_bottom (h : Real) (b : Real) : 
  (A = 1/2 * (top_width + b) * d) ∧ 
  (A = 840) ∧ 
  (top_width = 12) ∧ 
  (d = 84) 
  → b = 8 := 
by
  intros
  sorry

end width_of_canal_at_bottom_l662_662969


namespace probability_at_least_3_speak_l662_662832

noncomputable def probability_speaks (n : ℕ) (p : ℚ) : ℚ :=
  ∑ k in finset.range (n + 1), if k < 3 then (nat.choose n k) * (p^k) * ((1 - p)^(n-k)) else 0

theorem probability_at_least_3_speak :
  let p := (1 : ℚ) / 3,
      n := 6 in
  1 - probability_speaks n p = 233 / 729 :=
by {
  let p := 1/3,
  let n := 6,
  have h₀ : ∑ k in finset.range (n + 1), if k < 3 then (nat.choose n k) * (p^k) * ((1 - p)^(n-k)) else 0 = 496 / 729 := sorry,
  rw [probability_speaks, h₀],
  norm_num,
}

end probability_at_least_3_speak_l662_662832


namespace number_of_possible_ceil_values_l662_662806

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l662_662806


namespace intersection_sets_l662_662911

def universal_set : Set ℝ := Set.univ
def set_A : Set ℝ := {x | (x + 2) * (x - 5) < 0}
def set_B : Set ℝ := {x | -3 < x ∧ x < 4}

theorem intersection_sets (x : ℝ) : 
  (x ∈ set_A ∩ set_B) ↔ (-2 < x ∧ x < 4) :=
by sorry

end intersection_sets_l662_662911


namespace angle_B_plus_angle_D_105_l662_662800

theorem angle_B_plus_angle_D_105
(angle_A : ℝ) (angle_AFG angle_AGF : ℝ)
(h1 : angle_A = 30)
(h2 : angle_AFG = angle_AGF)
: angle_B + angle_D = 105 := sorry

end angle_B_plus_angle_D_105_l662_662800


namespace problem_solution_l662_662072

-- Definitions of given functions
def f (x : ℝ) (a : ℝ) : ℝ := a * x ^ 3 + 3 * x ^ 2 - 6 * a * x - 11
def g (x : ℝ) : ℝ := 3 * x ^ 2 + 6 * x + 12
def m (x : ℝ) (k : ℝ) : ℝ := k * x + 9

-- Main proof statement
theorem problem_solution (a k : ℝ) :
  (∃ a : ℝ, f'(-1, a) = 0 ∧ a = -2) ∧ (∃ k : ℝ, ∀ x : ℝ, is_tangent f g m k x ∧ k = 0)
  :=
sorry 

-- Helper definition for the derivative (specialized for use in this problem)
def f' (x : ℝ) (a : ℝ) : ℝ := 3 * a * x ^ 2 + 6 * x - 6 * a

-- Placeholder for the tangent relationship, to be concretely defined
def is_tangent (f g : ℝ → ℝ) (m : ℝ → ℝ) (k x : ℝ) : Prop :=
  sorry

end problem_solution_l662_662072


namespace multiply_powers_l662_662686

theorem multiply_powers (a : ℝ) : (a^3) * (a^3) = a^6 := by
  sorry

end multiply_powers_l662_662686


namespace complex_magnitude_problem_l662_662543

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l662_662543


namespace sum_consecutive_integers_l662_662577

theorem sum_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_consecutive_integers_l662_662577


namespace locus_points_are_parallel_to_KL_l662_662229

-- Definitions and setup
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Quadrilateral :=
  (A B C D : Point)
  (convex: True)  -- Assume convexity (placeholder for the actual convexity condition)
  (area : ℝ)  -- Total area S

def triangle_area (A B C : Point) : ℝ := sorry -- Placeholder function for area computation

def locus_condition (Q : Quadrilateral) (X : Point) : Prop :=
  let A := Q.A
  let B := Q.B
  let C := Q.C
  let D := Q.D
  let S := Q.area in
  triangle_area A B X + triangle_area C D X = S / 2

-- Theorem
theorem locus_points_are_parallel_to_KL (Q : Quadrilateral) :
  ∃ L K line_segment, (∀ X, X inside Q → locus_condition Q X → X ∈ line_segment) ∧ (line_segment ∥ KL) :=
sorry

end locus_points_are_parallel_to_KL_l662_662229


namespace exponent_evaluation_l662_662886

theorem exponent_evaluation {a b : ℕ} (h₁ : 2 ^ a ∣ 200) (h₂ : ¬ (2 ^ (a + 1) ∣ 200))
                           (h₃ : 5 ^ b ∣ 200) (h₄ : ¬ (5 ^ (b + 1) ∣ 200)) :
  (1 / 3) ^ (b - a) = 3 :=
by sorry

end exponent_evaluation_l662_662886


namespace multiple_of_first_number_is_nine_l662_662565

theorem multiple_of_first_number_is_nine : 
  ∃ m : ℝ, let x := 4.2 in let y := x + 2 in let z := y + 2 in m * x = 2 * z + 2 * y + 9 ∧ m = 9 :=
begin
  sorry
end

end multiple_of_first_number_is_nine_l662_662565


namespace tan_alpha_and_expression_values_l662_662416

noncomputable def tan (x : ℝ) : ℝ := Real.sin x / Real.cos x

theorem tan_alpha_and_expression_values 
  (α β : ℝ)
  (h1 : tan(α + β) = 2) 
  (h2 : tan(π - β) = (3 / 2)) :
  tan α = - (7 / 4) ∧ (Real.sin (π / 2 + α) - Real.sin (π + α)) / (Real.cos α + 2 * Real.sin α) = (3 / 10) :=
begin
  sorry
end

end tan_alpha_and_expression_values_l662_662416


namespace volume_of_prism_l662_662585

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 48) (h3 : b * c = 72) : a * b * c = 168 :=
by
  sorry

end volume_of_prism_l662_662585


namespace trigonometric_identity_l662_662278

theorem trigonometric_identity (α : ℝ) :
  4.46 * (cot (2 * α - π))^2 / (1 + (tan (3 * π / 2 - 2 * α))^2) - 3 * (cos (5 * π / 2 - 2 * α))^2 = 
  4 * sin (π / 6 - 2 * α) * sin (π / 6 + 2 * α) := 
by sorry

end trigonometric_identity_l662_662278


namespace part1_part2_max_part2_max_achieved_l662_662086

noncomputable
def a (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x)

noncomputable
def b (x : ℝ) : ℝ × ℝ := (Real.cos x, Real.sin x)

def parallel (v1 v2 : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

def f (x : ℝ) : ℝ :=
  let ax := a x
  let bx := b x
  (ax.1 + bx.1) * bx.1 + (ax.2 + bx.2) * bx.2

theorem part1 (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) (h_par : parallel (a x) (b x)) :
  x = Real.pi / 6 := by
sorry

theorem part2_max (x : ℝ) (h : 0 < x ∧ x < Real.pi / 2) :
  f x ≤ 5 / 2 := by
sorry

theorem part2_max_achieved :
  ∃ x : ℝ, 0 < x ∧ x < Real.pi / 2 ∧ f x = 5 / 2 := by
sorry

end part1_part2_max_part2_max_achieved_l662_662086


namespace arithmetic_sequence_sum_l662_662483

variable {α : Type} [LinearOrderedField α]

noncomputable def a_n (a1 d n : α) := a1 + (n - 1) * d

theorem arithmetic_sequence_sum (a1 d : α) (h1 : a_n a1 d 3 * a_n a1 d 11 = 5)
  (h2 : a_n a1 d 3 + a_n a1 d 11 = 3) : a_n a1 d 5 + a_n a1 d 6 + a_n a1 d 10 = 9 / 2 :=
by
  sorry

end arithmetic_sequence_sum_l662_662483


namespace sin_law_proportion_of_triangle_l662_662501

theorem sin_law_proportion_of_triangle (A B C : ℝ) (a b c : ℝ) (hA : A = 60) (ha : a = real.sqrt 3) 
(h_sin_A : real.sin (A * real.pi / 180) = real.sqrt 3 / 2) 
(h_law_of_sines : a / real.sin (A * real.pi / 180) = 2) 
: (a + b) / (real.sin (A * real.pi / 180) + real.sin (B * real.pi / 180)) = 2 :=
sorry

end sin_law_proportion_of_triangle_l662_662501


namespace pyramids_from_cuboid_l662_662624

-- Define the vertices of a cuboid
def vertices_of_cuboid : ℕ := 8

-- Define the edges of a cuboid
def edges_of_cuboid : ℕ := 12

-- Define the faces of a cuboid
def faces_of_cuboid : ℕ := 6

-- Define the combinatoric calculation
def combinations (n k : ℕ) : ℕ := (n.choose k)

-- Define the total number of tetrahedrons formed
def total_tetrahedrons : ℕ := combinations 7 3 - faces_of_cuboid * combinations 4 3

-- Define the expected result
def expected_tetrahedrons : ℕ := 106

-- The theorem statement to prove that the total number of tetrahedrons is 106
theorem pyramids_from_cuboid : total_tetrahedrons = expected_tetrahedrons :=
by
  sorry

end pyramids_from_cuboid_l662_662624


namespace range_of_a_l662_662745

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x
noncomputable def g (x a : ℝ) : ℝ := Real.exp x - 2 * a * x + 2

theorem range_of_a (a : ℝ) :
  (∀ x1 x2 ∈ set.Icc (0 : ℝ) 2, f x1 ≤ g x2 a) ↔ a ≤ Real.exp 1 / 2 :=
sorry

end range_of_a_l662_662745


namespace range_x_add_y_l662_662090

theorem range_x_add_y (x y : ℝ) (h : 2^x + 2^y = 1) : x + y ∈ Iic (-2) :=
sorry

end range_x_add_y_l662_662090


namespace min_value_of_f_on_interval_l662_662981

noncomputable def f (x : ℝ) : ℝ := (1 / x) - 2 * x

theorem min_value_of_f_on_interval :
  ∃ m : ℝ, is_glb (set.range (λ x, f x)) m ∧ m = -7 / 2 :=
by
  let I := set.Icc (1 : ℝ) (2 : ℝ)
  refine ⟨-7 / 2, _, rfl⟩
  sorry

end min_value_of_f_on_interval_l662_662981


namespace distance_between_J_and_Y_l662_662351

def distance_between_stations (t1 t2: ℝ) (sA sB sC: ℝ) : ℝ :=
  let t := (t2 - t1) / (sB - sC)
  let d := (sA + sB) * t
  d

theorem distance_between_J_and_Y : 
  let t1 := 20.0 / 60.0
  let t2 := 80.0 / 60.0
  let sA := 90.0
  let sB := 80.0
  let sC := 60.0
  distance_between_stations t1 t2 sA sB sC = 425 :=
by
  sorry

end distance_between_J_and_Y_l662_662351


namespace sequence_divisibility_l662_662227

theorem sequence_divisibility :
  ∃ h : ℕ, h = 6 ∧
  (∀ n : ℕ, 1998 ∣ (a (n + h) - a n))
  :=
begin
  -- Sequence definition
  let a : ℕ → ℤ 
      | 0 := 20
      | 1 := 100
      | (n+2) := 4 * a (n+1) + 5 * a n + 20,
  use 6,
  split,
  { refl, },
  { intros n,
    sorry,
  },
end

end sequence_divisibility_l662_662227


namespace find_point_P_line_AB_fixed_point_l662_662081

open Real

-- Define the parabola y^2 = 4x
def is_on_parabola (x y : ℝ) : Prop := y^2 = 4 * x

-- Define the area condition
def area_triangle_POQ (x₀ y₀ : ℝ) : Prop := abs(x₀) * y₀ = 2

-- Define the coordinates of the fixed point for line AB
def fixed_point (x y : ℝ) : Prop := x = 0 ∧ y = -2

theorem find_point_P (x₀ y₀ : ℝ) (h_pos : y₀ > 0) (h_parabola : is_on_parabola x₀ y₀) (h_area : area_triangle_POQ x₀ y₀) :
  x₀ = 1 ∧ y₀ = 2 :=
by
  sorry
  
theorem line_AB_fixed_point (x₀ y₀ : ℝ) (xA yA xB yB : ℝ) (hP : (x₀, y₀) = (1, 2)) (h_parabola_A : is_on_parabola xA yA) (h_parabola_B : is_on_parabola xB yB)
  (k1 k2 : ℝ) (h_slope_product : k1 * k2 = 4) (h_k1 : k1 = (yA - y₀) / (xA - x₀)) (h_k2 : k2 = (yB - y₀) / (xB - x₀)) :
  ∃ (x y : ℝ), fixed_point x y :=
by
  sorry

end find_point_P_line_AB_fixed_point_l662_662081


namespace defeated_candidate_percentage_l662_662119

theorem defeated_candidate_percentage (total_polled_votes invalid_votes : ℕ) (defeat_margin : ℕ) 
  (valid_votes := total_polled_votes - invalid_votes) 
  (D := (valid_votes - defeat_margin) / 2) : 
  (D : ℝ) / valid_votes * 100 ≈ 45.03 :=
by
  sorry

end defeated_candidate_percentage_l662_662119


namespace tiles_needed_to_cover_floor_l662_662662

theorem tiles_needed_to_cover_floor:
  (floor_length floor_width tile_length tile_width : ℝ)
  (floor_area tile_area number_of_tiles : ℝ)
  (floor_length = 10) 
  (floor_width = 14) 
  (tile_length = 1 / 2) 
  (tile_width = 2 / 3) 
  (floor_area = floor_length * floor_width) 
  (tile_area = tile_length * tile_width) 
  (number_of_tiles = floor_area / tile_area):
  number_of_tiles = 420 :=
by
  -- The proof will go here
  sorry

end tiles_needed_to_cover_floor_l662_662662


namespace largest_lambda_inequality_l662_662014

theorem largest_lambda_inequality :
  ∃ λ : ℝ, λ = 3 / 2 ∧ ∀ (a b c d : ℝ), a^2 + b^2 + c^2 + d^2 ≥ a * b + λ * b * c + c * d :=
begin
  sorry
end

end largest_lambda_inequality_l662_662014


namespace nadia_pies_l662_662920

variables (T R B S : ℕ)

theorem nadia_pies (h₁: R = T / 2) 
                   (h₂: B = R - 14) 
                   (h₃: S = (R + B) / 2) 
                   (h₄: T = R + B + S) :
                   R = 21 ∧ B = 7 ∧ S = 14 := 
  sorry

end nadia_pies_l662_662920


namespace not_function_age_height_l662_662631

def radius_circumference_function (r : ℝ) : ℝ := 2 * π * r
def angle_sine_function (θ : ℝ) : ℝ := Real.sin θ
def sides_interior_angles_function (n : ℕ) : ℝ := (n - 2) * 180

def age_height_relation (age : ℕ) (height : ℝ) : Prop := True

theorem not_function_age_height : 
  ¬ ∃ f : ℕ → ℝ, ∀ a h, age_height_relation a h → f a = h :=
sorry

end not_function_age_height_l662_662631


namespace gauss_range_l662_662393

def f (x : ℝ) : ℝ :=
  x^2 / (|x| + 1)

def gauss_function (x : ℝ) : ℤ :=
  Int.floor (f x)

def Q : Set ℤ :=
  { n | ∃ x, gauss_function x = n }

def B : Set ℤ := {0, 2}
def C : Set ℤ := {1, 2}
def D : Set ℤ := {1, 2, 3}

theorem gauss_range :
  Q = Set.range (λ n, n) ∧ 
  B ⊆ Q ∧ 
  C ⊆ Q ∧ 
  D ⊆ Q :=
by
  sorry

end gauss_range_l662_662393


namespace minimum_value_condition_l662_662077

theorem minimum_value_condition (a b : ℝ) (m n : ℝ):
  a = -2 → b = -1 → 
  (∀ x : ℝ, (x + 2) / (x + 1) < 0 → a < x ∧ x < b) → 
  (∃ (a b : ℝ), a = -2 ∧ b = -1) →
  (∀ (A : ℝ × ℝ), A = (a, b) → a = -2 ∧ b = -1 → m * n > 0 → 2 * m + n = 1) →
  (m > 0 ∧ n > 0) → 
  (∃ (val : ℝ), val = 9 ∧ val = 2 / m + 1 / n) :=
by
  -- Placeholders for the conditions
  intro ha hb _
  intro _ hab_eq
  intro _ line_eq
  intro mn_pos
  use 9
  split
  case intro h1 =>
    -- This assert means we prove the true value directly
    sorry
  case intro h2 =>
    -- This assert means we prove the equivalence directly
    sorry

end minimum_value_condition_l662_662077


namespace sqrt_eq_two_minus_a_l662_662268

theorem sqrt_eq_two_minus_a (a : ℝ) (h : a < 2) : (⟦(a-2)^(4:ℝ)⟧^(4:ℝ)) = 2 - a :=
by
  sorry

end sqrt_eq_two_minus_a_l662_662268


namespace centroid_of_interior_point_of_acute_angled_triangle_l662_662140

variables {A B C P D E F : Type} [InnerProductSpace ℝ A]
variables (AP : A × ℝ) (BP : B × ℝ) (CP : C × ℝ)
variables (D : Type) (E : Type) (F : Type)
variables [AddGroup D] [AddGroup E] [AddGroup F]
variables [MulAction ℝ D] [MulAction ℝ E] [MulAction ℝ F]
variables [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F]

def is_interior_point (P : A) (ABC : AffineSpace ℝ) : Prop := sorry
def is_acute_angled_triangle (ABC : AffineSpace ℝ) : Prop := sorry
def are_similar_triangles (DEF ABC : Type) : Prop := sorry
def is_centroid (P : A) (ABC : AffineSpace ℝ) : Prop := sorry

theorem centroid_of_interior_point_of_acute_angled_triangle
  (ABC : AffineSpace ℝ)
  (h1 : is_interior_point P ABC)
  (h2 : is_acute_angled_triangle ABC)
  (h3 : are_similar_triangles DEF ABC) :
  is_centroid P ABC :=
sorry

end centroid_of_interior_point_of_acute_angled_triangle_l662_662140


namespace goods_train_speed_l662_662309

theorem goods_train_speed (V_man : ℝ) (L_g : ℝ) (T : ℝ) : 
  V_man = 20 → L_g = 280 → T = 9 → 
  let V_g := (0.28 * 3600 / T) - V_man in 
  V_g = 92 := by
  intros hV_man hL_g hT
  rw [hL_g, hT, hV_man]
  let V_relative := 0.28 * 3600 / 9
  have hV_relative : V_relative = 112 := by
    norm_num
  rw [←hV_relative]
  let V_g := 112 - 20
  suffices : V_g = 92
  exact this
  norm_num

end goods_train_speed_l662_662309


namespace ceil_x_pow_2_values_l662_662815

theorem ceil_x_pow_2_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ n, n = 29 ∧ (∀ y, ceil (y^2) = ⌈x^2⌉ → 196 < y^2 ∧ y^2 ≤ 225) :=
sorry

end ceil_x_pow_2_values_l662_662815


namespace area_ABC_eq_2m_l662_662131

variables {ABC : Type} [triangle ABC] (I : incenter ABC) (D E F : ABC → ABC → ABC)
variables (r m : ℝ)

def area_triangle (ABC : Type) [triangle ABC] : ℝ := sorry

theorem area_ABC_eq_2m (h₁ : lines_AI_BI_CI_intersect_sides ABC I D E F)
                      (h₂ : inradius ABC I = r)
                      (h₃ : area_DEF ABC D E F = m) :
  area_triangle ABC = 2 * m := sorry

end area_ABC_eq_2m_l662_662131


namespace induction_case_n_eq_1_l662_662176

variable {α : ℝ}
variable {k : ℤ}
variable (h : α ≠ k * Real.pi)

theorem induction_case_n_eq_1 (h : α ≠ k * Real.pi) : 
  (1 / 2 + ∑ i in (Finset.range 1).image (λn, (2 * (n + 1) - 1) * α), Real.cos i) = 1 / 2 + Real.cos α :=
by sorry

end induction_case_n_eq_1_l662_662176


namespace weight_of_B_l662_662168

theorem weight_of_B (A B C : ℝ)
(h1 : (A + B + C) / 3 = 45)
(h2 : (A + B) / 2 = 40)
(h3 : (B + C) / 2 = 41)
(h4 : 2 * A = 3 * B ∧ 5 * C = 3 * B)
(h5 : A + B + C = 144) :
B = 43.2 :=
sorry

end weight_of_B_l662_662168


namespace alpha_value_l662_662084

def vector_oa : ℝ × ℝ × ℝ := (1, -7, 8)
def vector_ob : ℝ × ℝ × ℝ := (0, 14, 16)
def vector_c (α : ℝ) : ℝ × ℝ × ℝ := (Real.sqrt 2, (1/7) * Real.sin α, (1/8) * Real.cos α)

-- Conditions
axiom alpha_in_interval (α : ℝ) : 0 < α ∧ α < π

-- Question:
-- If vector_c is perpendicular to the plane formed by vector_oa and vector_ob, then α = 3π/4
theorem alpha_value (α : ℝ) 
  (h1 : vector_oa.1 * (Real.sqrt 2) + vector_oa.2 * ((1 / 7) * Real.sin α) + vector_oa.3 * ((1 / 8) * Real.cos α) = 0)
  (h2 : vector_ob.1 * (Real.sqrt 2) + vector_ob.2 * ((1 / 7) * Real.sin α) + vector_ob.3 * ((1 / 8) * Real.cos α) = 0)
  (h3 : alpha_in_interval α) : α = 3 * Real.pi / 4 := 
sorry

end alpha_value_l662_662084


namespace cost_of_10_pens_l662_662564

theorem cost_of_10_pens (cost_per_pen : ℕ) (number_of_pens : ℕ) (h1 : cost_per_pen = 2) (h2 : number_of_pens = 10) : 
  number_of_pens * cost_per_pen = 20 := 
by
  rw [h1, h2]
  exact rfl

end cost_of_10_pens_l662_662564


namespace ceil_square_count_ceil_x_eq_15_l662_662802

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l662_662802


namespace optimal_years_minimize_cost_l662_662976

noncomputable def initial_cost : ℝ := 150000
noncomputable def annual_expenses (n : ℕ) : ℝ := 15000 * n
noncomputable def maintenance_cost (n : ℕ) : ℝ := (n * (3000 + 3000 * n)) / 2
noncomputable def total_cost (n : ℕ) : ℝ := initial_cost + annual_expenses n + maintenance_cost n
noncomputable def average_annual_cost (n : ℕ) : ℝ := total_cost n / n

theorem optimal_years_minimize_cost : ∀ n : ℕ, n = 10 ↔ average_annual_cost 10 ≤ average_annual_cost n :=
by sorry

end optimal_years_minimize_cost_l662_662976


namespace angle_at_3_15_l662_662788

theorem angle_at_3_15 : 
  let minute_hand_angle := 90.0
  let hour_at_3 := 90.0
  let hour_hand_at_3_15 := hour_at_3 + 7.5
  minute_hand_angle == 90.0 ∧ hour_hand_at_3_15 == 97.5 →
  abs (hour_hand_at_3_15 - minute_hand_angle) = 7.5 :=
by
  sorry

end angle_at_3_15_l662_662788


namespace max_f_l662_662017

def f (x : ℝ) : ℝ := (1 / 3) * x^3 - 4 * x + 4

theorem max_f :
  ∃ x ∈ set.Icc (-3 : ℝ) (3 : ℝ), ∀ y ∈ set.Icc (-3 : ℝ) (3 : ℝ), f y ≤ f x ∧ f x = 28 / 3 :=
begin
  sorry
end

end max_f_l662_662017


namespace f_of_neg_l662_662732

-- Definitions
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f(-x) = f(x)

noncomputable def f : ℝ → ℝ :=
  λ x, if x > 0 then x^2 + x else x^2 - x  -- Temporary placeholder

-- Theorem statement
theorem f_of_neg {f : ℝ → ℝ}
  (h_even : even_function f)
  (h_pos : ∀ x, 0 < x → f(x) = x^2 + x) :
  ∀ x, x < 0 → f(x) = x^2 - x :=
by
  intro x hx
  sorry

end f_of_neg_l662_662732


namespace rhombus_construction_exists_l662_662334

-- Let's define the elements as per the conditions
variables {P Q : Point} -- two given points
variables {l1 l2 : Line} -- two given parallel lines
variables {m : ℝ} -- distance between the lines

-- Define the condition that l1 and l2 are parallel and m is the distance between them
axiom parallel (l1 l2 : Line) : Prop
axiom distance_between (l1 l2 : Line) : ℝ → Prop

-- Define the concept of constructing a rhombus passing through given points and lines
def construct_rhombus (P Q : Point) (l1 l2 : Line) (m : ℝ) : Prop :=
  ∃ (A B C D : Point), 
    (A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) ∧ 
    (distance A B = distance B C ∧ distance B C = distance C D ∧ distance C D = distance D A) ∧
    (is_on_line A l1 ∧ is_on_line B l1 ∧ is_on_line C l2 ∧ is_on_line D l2) ∧ 
    (is_on_tangent P A m ∧ is_on_tangent P B m ∧ is_on_tangent Q C m ∧ is_on_tangent Q D m) 

-- The main statement to prove
theorem rhombus_construction_exists 
    (h_parallel : parallel l1 l2) 
    (h_dist : distance_between l1 l2 m) 
    (h_points : P ≠ Q) :
    construct_rhombus P Q l1 l2 m :=
sorry

end rhombus_construction_exists_l662_662334


namespace area_of_45_45_90_triangle_l662_662326

theorem area_of_45_45_90_triangle (h : ℝ) (h_eq : h = 8 * Real.sqrt 2) : 
  ∃ (A : ℝ), A = 32 := 
by
  sorry

end area_of_45_45_90_triangle_l662_662326


namespace vec_v_satisfies_l662_662695

open Matrix

def A := ![![0, 2], ![4, 0]] : Matrix (Fin 2) (Fin 2) ℚ
def v := ![0, 47 / 665] : Fin 2 → ℚ
def I2 := (1 : Matrix (Fin 2) (Fin 2) ℚ)

theorem vec_v_satisfies :
  (A^6 + 2 * A^4 + 3 * A^2 + I2) ⬝ v = ![0, 47] := 
  sorry

end vec_v_satisfies_l662_662695


namespace tangent_slope_ln_passing_origin_eq_inv_e_l662_662066

noncomputable def curve := λ x : ℝ, Real.log x

theorem tangent_slope_ln_passing_origin_eq_inv_e :
  ∃ a : ℝ, (curve a) = Real.log a ∧ 
            ((λ x : ℝ, (curve a) + (1/a) * (x - a)) 0 = 0) ∧ 
            (1/a = 1/Real.exp 1) :=
begin
  sorry
end

end tangent_slope_ln_passing_origin_eq_inv_e_l662_662066


namespace triangle_segment_ratio_l662_662971

theorem triangle_segment_ratio
  (A B C D : Type)
  [is_triangle A B C]
  [on_segment D B C]
  (BD CD : ℕ)
  (h_parallel_lines_divide_AD : parallel_lines_divide_equal_segments A D 4)
  (h_ratio_shaded_unshaded : area_ratio_shaded_unshaded (shaded_region A D) (unshaded_region A D) = 49 / 33)
  (rel_prime : nat.coprime BD CD) :
  BD + CD = 40 := 
  sorry

end triangle_segment_ratio_l662_662971


namespace system_solution_l662_662710

theorem system_solution (x₁ x₂ x₃ x₄ x₅ : ℝ) (h₁ : 0 < x₁) (h₂ : 0 < x₂) (h₃ : 0 < x₃) (h₄ : 0 < x₄) (h₅ : 0 < x₅)
  (h₆ : x₁ + x₂ = x₃^2) (h₇ : x₃ + x₄ = x₅^2) (h₈ : x₄ + x₅ = x₁^2) (h₉ : x₅ + x₁ = x₂^2) :
  x₁ = 2 ∧ x₂ = 2 ∧ x₃ = 2 ∧ x₄ = 2 ∧ x₅ = 2 := 
by 
  sorry

end system_solution_l662_662710


namespace cyclic_quadrilateral_fourth_side_l662_662223

theorem cyclic_quadrilateral_fourth_side (AB BC CD DA : ℝ) 
  (hBC : BC = 5) (hCD : CD = 6) (hDA : DA = 12)
  (h1 : ∃ (AC : ℝ), 2 * area ABC AC hBC = 2 * area ACD AC hDA) :
  AB ∈ {2, 5, 10, 14.4} :=
sorry

end cyclic_quadrilateral_fourth_side_l662_662223


namespace probability_of_above_parabola_probability_of_above_parabola_fraction_l662_662960

def is_above_parabola (a b c : ℕ) : Prop :=
  b > (a^3 + c) / (1 + a^2)

theorem probability_of_above_parabola :
  (∑ a in finset.range 9, ∑ c in finset.range 9, ∑ b in finset.range 9, if is_above_parabola (a+1) (b+1) (c+1) then 1 else 0 : ℕ) = 200 :=
sorry

theorem probability_of_above_parabola_fraction :
  (200 : ℚ) / 729 = 200 / 729 :=
sorry

end probability_of_above_parabola_probability_of_above_parabola_fraction_l662_662960


namespace function_graph_proof_l662_662826

noncomputable def function_graph_condition (a b: ℝ) :=
  (∀ x : ℝ, a > 0 ∧ a ≠ 1 → (a = 1 ∨ a > 1 ∧ b > 0 → 
    ∃ y : ℝ, y = a^x - (b + 1) ∧ 
    (y > 0 ∨ (a ≠ 1 ∧ b > 0 ∧ x > 0 ∧ y < 0 ) ∨ (a > 1 ∧ b > 0 ∧ x < 0 ∧ y < 0))))

theorem function_graph_proof (a b: ℝ) :
  (∃ x y : ℝ, y = a^x - (b + 1) ∧ 
    (y > 0 ∨ (a ≠ 1 ∧ b > 0 ∧ x > 0 ∧ y < 0 ) ∨ (a > 1 ∧ b > 0 ∧ x < 0 ∧ y < 0))) ↔ (a > 1 ∧ b > 0) := 
begin
  sorry
end

end function_graph_proof_l662_662826


namespace max_value_2ab_plus_2bc_sqrt2_l662_662163

theorem max_value_2ab_plus_2bc_sqrt2 (a b c : ℝ) (h1 : a ≥ 0) (h2 : b ≥ 0) (h3 : c ≥ 0) (h4 : a^2 + b^2 + c^2 = 1) :
  2 * a * b + 2 * b * c * Real.sqrt 2 ≤ Real.sqrt 3 :=
sorry

end max_value_2ab_plus_2bc_sqrt2_l662_662163


namespace initial_percentage_females_l662_662610

noncomputable def initial_percentage (E F : ℕ) : ℚ :=
  (F : ℚ) / (E : ℚ) * 100

theorem initial_percentage_females (E F : ℕ)
  (h1 : E + 26 = 312)
  (h2 : F = Int.floor (0.55 * 312)) :
  initial_percentage E F ≈ 60.14 := by
  sorry

end initial_percentage_females_l662_662610


namespace f_2024_possible_values_l662_662656

-- Define a cardinal set.
def is_cardinal (S : Set ℕ) : Prop :=
  S.finite ∧ S.card = S.card

-- Define the function f and state the necessary properties.
variable (f : ℕ → ℕ)

-- Define the property that f preserves cardinality.
def cardinal_preserving (f : ℕ → ℕ) : Prop :=
  ∀ S : Set ℕ, is_cardinal S → is_cardinal (f '' S)

theorem f_2024_possible_values
  (f : ℕ → ℕ)
  (h_f_card_preserving : cardinal_preserving f)
  : f 2024 ∈ {1, 2, 2024} :=
sorry

end f_2024_possible_values_l662_662656


namespace solve_nat_pairs_l662_662637

theorem solve_nat_pairs (n m : ℕ) :
  m^2 + 2 * 3^n = m * (2^(n+1) - 1) ↔ (n = 3 ∧ m = 6) ∨ (n = 3 ∧ m = 9) :=
by sorry

end solve_nat_pairs_l662_662637


namespace stable_state_exists_l662_662242

-- Definition of the problem
theorem stable_state_exists 
(N : ℕ) (N_ge_3 : N ≥ 3) (letters : Fin N → Fin 3) 
(perform_operation : ∀ (letters : Fin N → Fin 3), Fin N → Fin 3)
(stable : ∀ (letters : Fin N → Fin 3), Prop)
(initial_state : Fin N → Fin 3):
  ∃ (state : Fin N → Fin 3), (∀ i, perform_operation state i = state i) ∧ stable state :=
sorry

end stable_state_exists_l662_662242


namespace ratio_DS_BC_volume_cone_l662_662925

-- Part (a)
theorem ratio_DS_BC (S A B C D K : Point)
  (h_regular_pyramid : IsRegularPyramid S A B C D)
  (h_on_edge_SA : K ∈ Segment S A)
  (h_ratios : AK / KS = 1 / 4)
  (h_cone_apex : IsApexOfRightCircularCone K (Triangle B C D)) :
  DS / BC = 2 / sqrt(3) :=
sorry

-- Part (b)
theorem volume_cone (S A B C D K : Point)
  (h_regular_pyramid : IsRegularPyramid S A B C D)
  (h_on_edge_SA : K ∈ Segment S A)
  (h_ratios : AK / KS = 1 / 4)
  (h_cone_apex : IsApexOfRightCircularCone K (Triangle B C D))
  (h_pyramid_height : PyramidHeight S A B C D = 5) :
  VolumeOfCone K (CircleThrough B C D) = 64 * π / sqrt(15) :=
sorry

end ratio_DS_BC_volume_cone_l662_662925


namespace cube_faces_sum_l662_662601

theorem cube_faces_sum (n m k l p q : ℕ) 
  (h1 : n + m + k + l + p + q = 81)
  (h2 : m = n + 1) (h3 : k = m + 1) (h4 : l = k + 1) (h5 : p = l + 1) (h6 : q = p + 1)
  (h7 : ∃ a b c, {n, m} = {a, b} ∧ {k, l} = {b, c} ∧ {p, q} = {c, a})
  : n + m + k + l + p + q = 81 := 
by 
  sorry

end cube_faces_sum_l662_662601


namespace correct_differentiation_operations_l662_662629

noncomputable def A : Prop :=
  deriv (λ x : ℝ, sin (x / 2)) = λ x : ℝ, cos (x / 2)

noncomputable def B : Prop :=
  deriv (λ x : ℝ, x / (x + 1) - real.exp x) = λ x : ℝ, 1 / (x + 1)^2 - real.exp x * real.log 2

noncomputable def C : Prop :=
  deriv (λ x : ℝ, x^2 / real.exp x) = λ x : ℝ, (2 * x - x^2) / real.exp x

noncomputable def D : Prop :=
  deriv (λ x : ℝ, x^2 * cos x) = λ x : ℝ, 2 * x * cos x + x^2 * sin x

theorem correct_differentiation_operations (hA : ¬A) (hB : B) (hC : C) (hD : ¬D) : B ∧ C :=
by { exact ⟨hB, hC⟩ }

end correct_differentiation_operations_l662_662629


namespace museum_college_students_income_l662_662853

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l662_662853


namespace distance_between_intersections_l662_662691

noncomputable def cube_vertices : list (ℝ × ℝ × ℝ) := [
  (0,0,0), (0,0,6), (0,6,0), (0,6,6), 
  (6,0,0), (6,0,6), (6,6,0), (6,6,6)
]

def P : ℝ × ℝ × ℝ := (0, 3, 0)
def Q : ℝ × ℝ × ℝ := (2, 0, 0)
def R : ℝ × ℝ × ℝ := (2, 6, 6)

theorem distance_between_intersections : 
  ∃ S T : ℝ × ℝ × ℝ, 
  S ∈ cube_vertices ∧ T ∈ cube_vertices ∧ 
  S ≠ T ∧ 
  (∃ (a b c : ℝ), a * S.1 + b * S.2 + c * S.3 = -6 ∧ a * T.1 + b * T.2 + c * T.3 = -6) ∧
  dist S T = 2 * real.sqrt 13 :=
by sorry

end distance_between_intersections_l662_662691


namespace simplest_quadratic_radical_l662_662630

theorem simplest_quadratic_radical (x : ℝ) (hx : x ≠ 0) : 
    is_simplest_quadratic_radical (sqrt 3) [sqrt 3, sqrt (1 / x), sqrt 12, sqrt (1 / 2)] :=
sorry

def is_simplest_quadratic_radical (r : ℝ) (lst : List ℝ) : Prop :=
    r = sqrt 3 ∧ ∀ x ∈ lst, (r = sqrt 3) ⊑ ((x ≠ sqrt 3) ∧ (simplify x = r))

def simplify (r : ℝ) : ℝ := match r with
    | sqrt 3 => sqrt 3
    | (sqrt (1 / x)) => sqrt ((1:ℝ) / x)
    | (sqrt 12) => 2 * sqrt 3
    | (sqrt (1 / 2)) => sqrt (1 / 2)
    | _ => r

end simplest_quadratic_radical_l662_662630


namespace log_function_range_l662_662983

noncomputable def range_of_log_function : set ℝ :=
{y | ∃ x : ℝ, y = real.log2 (x^2 + 2)}

theorem log_function_range : range_of_log_function = {y | 1 < y} :=
by
  sorry

end log_function_range_l662_662983


namespace tangent_line_intercepts_ellipse_l662_662032

/-- Let O be the circle defined by x^2 + y^2 = 1, and let l be the line y = kx + b with b > 0,
which is tangent to the circle and intersects the ellipse x^2 / 2 + y^2 = 1 at two distinct points
A and B. Let f(k) = sqrt(k^2 + 1). We will:
1. Prove that b = f(k).
2. Given OA • OB = 2/3, prove that the equations of l are y = x + sqrt(2) or y = -x + sqrt(2). -/
theorem tangent_line_intercepts_ellipse 
  (k b : ℝ) (h_tangent : b = Real.sqrt (k^2 + 1)) (b_pos : b > 0) 
  (h_intercepts : ∃ A B : ℝ × ℝ, 
    let O := λ x y : ℝ => x^2 + y^2 == 1 in
    let l := λ x y : ℝ => y == k * x + b in
    ∀ A B, (A = A ∧ B = B) → x^2 / 2 + y^2 = 1)
  (dot_prod_condition : ∃ A B : ℝ × ℝ, 
    let OA := λ A : ℝ × ℝ => A.fst^2 + A.snd^2 == 1 in
    let OB := λ B : ℝ × ℝ => B.fst^2 + B.snd^2 == 1 in
    OA ** OB = 2 / 3):
  b = Real.sqrt (k^2 + 1) ∧ 
  (∀ A B : ℝ × ℝ, 
    let l1 := λ x y : ℝ => y == x + Real.sqrt 2 in
    let l2 := λ x y : ℝ => y == -x + Real.sqrt 2 in
    tangent_line_intercepts_ellipse k b) ∨ 
    tangent_line_intercepts_ellipse k (Real.sqrt 2) :=
sorry

end tangent_line_intercepts_ellipse_l662_662032


namespace polynomial_form_l662_662156

variable {n : ℕ} (a_n a_{n-1} a_1 a_0 : ℝ)
variable (P : ℝ → ℝ)
variable (roots : Fin n → ℝ)

def polynomial_has_n_real_roots_le_neg1 (P : ℝ → ℝ) (roots : Fin n → ℝ) : Prop :=
  ∀ i, roots i ≤ -1 ∧ P = λ x, a_n * (roots i + x)

def polynomial_coeff_relation (a_n a_{n-1} a_1 a_0 : ℝ) : Prop :=
  a_0^2 + a_1 * a_n = a_n^2 + a_0 * a_{n-1}

theorem polynomial_form (P : ℝ → ℝ) (roots : Fin n → ℝ) (a_n a_{n-1} a_1 a_0 : ℝ)
    (h1 : polynomial_has_n_real_roots_le_neg1 P roots)
    (h2 : polynomial_coeff_relation a_n a_{n-1} a_1 a_0) :
  ∃ β : ℝ, β ≥ 1 ∧ a_n ≠ 0 ∧
    P = λ x, a_n * (x + 1)^(n-1) * (x + β) :=
sorry

end polynomial_form_l662_662156


namespace sum_of_chords_less_than_pi_k_l662_662460

-- Definitions from problem conditions:
def radius : ℝ := 1
def k : ℕ -- a natural number representing the maximum chords a diameter intersects

-- Theorem to be proven
theorem sum_of_chords_less_than_pi_k (k : ℕ) (chords : list ℝ) (h_intersect : ∀ (d : ℝ), list.count (λ chord, chord_intersects_on_diameter d chord) chords ≤ k) :
    list.sum chords < Real.pi * k := 
sorry

-- Assume a function exists to determine if a chord intersects on a given diameter
def chord_intersects_on_diameter (d : ℝ) (chord : ℝ) : Prop := sorry

end sum_of_chords_less_than_pi_k_l662_662460


namespace rubber_duck_cost_l662_662204

theorem rubber_duck_cost 
  (price_large : ℕ)
  (num_regular : ℕ)
  (num_large : ℕ)
  (total_revenue : ℕ)
  (h1 : price_large = 5)
  (h2 : num_regular = 221)
  (h3 : num_large = 185)
  (h4 : total_revenue = 1588) :
  ∃ (cost_regular : ℕ), (num_regular * cost_regular + num_large * price_large = total_revenue) ∧ cost_regular = 3 :=
by
  exists 3
  sorry

end rubber_duck_cost_l662_662204


namespace third_stick_not_possible_l662_662862

theorem third_stick_not_possible (a b c d : ℕ) (h1 : a = 8) (h2 : b = 7) (h3 : c = 13) (h4 : d = 15) :
  ∀ x, (x = d) → ¬ (a + b > x ∧ |a - b| < x) :=
by {
  intros,
  subst h1 h2 h4,
  sorry,
}

end third_stick_not_possible_l662_662862


namespace Bilbo_can_find_adjacent_gnomes_l662_662646

theorem Bilbo_can_find_adjacent_gnomes (n : ℕ) (gnomes : Finset ℕ) (q : ℕ → ℕ → ℕ)
  (H1 : n = 99)
  (H2 : ∀ (i j : ℕ), i ≠ j → q i j ≤ 48 ∧ q i j = d i j) :
  ∃ (i j : ℕ), i ≠ j ∧ d i j = 0 :=
by
  sorry

end Bilbo_can_find_adjacent_gnomes_l662_662646


namespace negation_equivalence_l662_662270

-- Define the angles in a triangle as three real numbers
def is_triangle (a b c : ℝ) : Prop :=
  a + b + c = 180 ∧ a > 0 ∧ b > 0 ∧ c > 0

-- Define the proposition
def at_least_one_angle_not_greater_than_60 (a b c : ℝ) : Prop :=
  a ≤ 60 ∨ b ≤ 60 ∨ c ≤ 60

-- Negate the proposition
def all_angles_greater_than_60 (a b c : ℝ) : Prop :=
  a > 60 ∧ b > 60 ∧ c > 60

-- Prove that the negation of the proposition is equivalent
theorem negation_equivalence (a b c : ℝ) (h_triangle : is_triangle a b c) :
  ¬ at_least_one_angle_not_greater_than_60 a b c ↔ all_angles_greater_than_60 a b c :=
by
  sorry

end negation_equivalence_l662_662270


namespace length_of_second_train_l662_662333

/-
  Given:
  - l₁ : Length of the first train in meters
  - v₁ : Speed of the first train in km/h
  - v₂ : Speed of the second train in km/h
  - t : Time to cross the second train in seconds

  Prove:
  - l₂ : Length of the second train in meters = 299.9560035197185 meters
-/

variable (l₁ : ℝ) (v₁ : ℝ) (v₂ : ℝ) (t : ℝ) (l₂ : ℝ)

theorem length_of_second_train
  (h₁ : l₁ = 250)
  (h₂ : v₁ = 72)
  (h₃ : v₂ = 36)
  (h₄ : t = 54.995600351971845)
  (h_result : l₂ = 299.9560035197185) :
  (v₁ * 1000 / 3600 - v₂ * 1000 / 3600) * t - l₁ = l₂ := by
  sorry

end length_of_second_train_l662_662333


namespace museum_revenue_from_college_students_l662_662857

/-!
In one day, 200 people visit The Metropolitan Museum of Art in New York City. Half of the visitors are residents of New York City. 
Of the NYC residents, 30% are college students. If the cost of a college student ticket is $4, we need to prove that 
the museum gets $120 from college students that are residents of NYC.
-/

theorem museum_revenue_from_college_students :
  let total_visitors := 200
  let residents_nyc := total_visitors / 2
  let college_students_percentage := 30 / 100
  let college_students := residents_nyc * college_students_percentage
  let ticket_cost := 4
  residents_nyc = 100 ∧ 
  college_students = 30 ∧ 
  ticket_cost * college_students = 120 := 
by
  sorry

end museum_revenue_from_college_students_l662_662857


namespace angle_between_medians_of_tetrahedron_l662_662376

variables (a : ℝ)
def A := (0, 0, 0)
def B := (a * sqrt 3 / 2, a / 2, 0)
def C := (a * sqrt 3 / 2, -a / 2, 0)
def D := (a * sqrt 3 / 3, 0, a * sqrt (2 / 3))
def K := (a * sqrt 3 / 2, 0, 0)
def M := (a * sqrt 3 / 4, 0, a * sqrt (2 / 3) / 2)

def vec_sub (u v : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (u.1 - v.1, u.2 - v.2, u.3 - v.3)

def dot_product (u v : ℝ × ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2 + u.3 * v.3

def magnitude (u : ℝ × ℝ × ℝ) : ℝ :=
  sqrt (u.1^2 + u.2^2 + u.3^2)

def AK := vec_sub K A
def DM := vec_sub M D

noncomputable def angle_between_vectors (u v : ℝ × ℝ × ℝ) : ℝ :=
  real.arccos (dot_product u v / (magnitude u * magnitude v))

theorem angle_between_medians_of_tetrahedron :
  angle_between_vectors AK DM = real.arccos (1 / 6) := 
sorry

end angle_between_medians_of_tetrahedron_l662_662376


namespace main_theorem_l662_662449

-- The condition
def condition (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, f (2^x - 1) = 4^x - 1

-- The property we need to prove
def proves (f : ℝ → ℝ) : Prop :=
∀ x : ℝ, -1 ≤ x → f x = x^2 + 2*x

-- The main theorem connecting the condition to the desired property
theorem main_theorem (f : ℝ → ℝ) (h : condition f) : proves f :=
sorry

end main_theorem_l662_662449


namespace find_value_l662_662493

-- Given points A(a, 1), B(2, b), and C(3, 4).
variables (a b : ℝ)

-- Given condition from the problem
def condition : Prop := (3 * a + 4 = 6 + 4 * b)

-- The target is to find 3a - 4b
def target : ℝ := 3 * a - 4 * b

theorem find_value (h : condition a b) : target a b = 2 := 
by sorry

end find_value_l662_662493


namespace unique_integer_satisfying_conditions_l662_662289

theorem unique_integer_satisfying_conditions (P : ℕ) (h1 : ∑ d in (Finset.filter (λ x, P % x = 0) (Finset.range (P + 1)), d) = 2 * P) 
  (h2 : ∏ d in (Finset.filter (λ x, P % x = 0) (Finset.range (P + 1)), d) = P * P) : 
  P = 6 := 
by 
  sorry

end unique_integer_satisfying_conditions_l662_662289


namespace sum_of_chords_less_than_pi_k_l662_662461

-- Definitions from problem conditions:
def radius : ℝ := 1
def k : ℕ -- a natural number representing the maximum chords a diameter intersects

-- Theorem to be proven
theorem sum_of_chords_less_than_pi_k (k : ℕ) (chords : list ℝ) (h_intersect : ∀ (d : ℝ), list.count (λ chord, chord_intersects_on_diameter d chord) chords ≤ k) :
    list.sum chords < Real.pi * k := 
sorry

-- Assume a function exists to determine if a chord intersects on a given diameter
def chord_intersects_on_diameter (d : ℝ) (chord : ℝ) : Prop := sorry

end sum_of_chords_less_than_pi_k_l662_662461


namespace cannot_determine_good_carrots_l662_662563

theorem cannot_determine_good_carrots (O M T : ℕ) (hO : O = 20) (hM : M = 14) (hT : T = O + M) : ¬∃ good_carrots : ℕ, good_carrots = ? := 
by 
  sorry

end cannot_determine_good_carrots_l662_662563


namespace ABCD_eq_one_l662_662357

def A : ℝ := Real.sqrt 3008 + Real.sqrt 3009
def B : ℝ := -Real.sqrt 3008 - Real.sqrt 3009
def C : ℝ := Real.sqrt 3008 - Real.sqrt 3009
def D : ℝ := Real.sqrt 3009 - Real.sqrt 3008

theorem ABCD_eq_one : A * B * C * D = 1 := by
  sorry

end ABCD_eq_one_l662_662357


namespace crabapple_sequences_l662_662681

/--
Prove that the number of different sequences of crabapple recipients in a week is 14641,
given the conditions:
- there are 11 students,
- the class meets four times a week,
- the same student can be picked more than once.
-/
theorem crabapple_sequences :
  let num_students := 11
  let num_meetings := 4
  num_students ^ num_meetings = 14641 :=
by
  let num_students := 11
  let num_meetings := 4
  have h : num_students ^ num_meetings = 11 ^ 4 := rfl
  rw h
  exact rfl

end crabapple_sequences_l662_662681


namespace angle_between_hands_at_3_15_l662_662779

theorem angle_between_hands_at_3_15 :
  let hour_angle_at_3 := 3 * 30
  let hour_hand_move_rate := 0.5
  let minute_angle := 15 * 6
  let hour_angle_at_3_15 := hour_angle_at_3 + 15 * hour_hand_move_rate
  abs (hour_angle_at_3_15 - minute_angle) = 7.5 := 
by
  sorry

end angle_between_hands_at_3_15_l662_662779


namespace inequality_proof_l662_662397

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) :
  x * (x - z) ^ 2 + y * (y - z) ^ 2 ≥ (x - z) * (y - z) * (x + y - z) :=
by
  sorry

end inequality_proof_l662_662397


namespace Darnel_sprinted_further_l662_662354

-- Define the distances sprinted and jogged
def sprinted : ℝ := 0.88
def jogged : ℝ := 0.75

-- State the theorem to prove the main question
theorem Darnel_sprinted_further : sprinted - jogged = 0.13 :=
by
  sorry

end Darnel_sprinted_further_l662_662354


namespace xiao_ming_runs_distance_l662_662613

theorem xiao_ming_runs_distance 
  (num_trees : ℕ) 
  (first_tree : ℕ) 
  (last_tree : ℕ) 
  (distance_between_trees : ℕ) 
  (gap_count : ℕ) 
  (total_distance : ℕ)
  (h1 : num_trees = 200) 
  (h2 : first_tree = 1) 
  (h3 : last_tree = 200) 
  (h4 : distance_between_trees = 6) 
  (h5 : gap_count = last_tree - first_tree)
  (h6 : total_distance = gap_count * distance_between_trees) :
  total_distance = 1194 :=
sorry

end xiao_ming_runs_distance_l662_662613


namespace vertex_of_quadratic_l662_662924

theorem vertex_of_quadratic :
  ∀ x : ℝ, ∃ v : ℝ × ℝ, (v.1 = 0 ∧ v.2 = -48) ∧ y = 24 * x^2 - 48 := 
begin
  intros x,
  use (0, -48),
  split,
  { split; refl },
  { sorry }
end

end vertex_of_quadratic_l662_662924


namespace largest_n_unit_square_l662_662863

/-- A set of 2000 points is called good if each point lies in the coordinate bounds
    0 ≤ x_i ≤ 83 and 0 ≤ y_i ≤ 83 for i = 1, 2, ..., 2000 and x_i ≠ x_j when i ≠ j.
    The task is to find the largest positive integer n such that, for any good set,
    there exists a unit square containing exactly n points on its interior or its boundary. -/
theorem largest_n_unit_square : ∃ n : ℕ, n = 25 ∧
  ∀ (pts : Fin 2000 → (ℝ × ℝ)),
    (∀ i, 0 ≤ pts i.1 ∧ pts i.1 ≤ 83) ∧
    (∀ i, 0 ≤ pts i.2 ∧ pts i.2 ≤ 83) ∧
    (∀ i j, i ≠ j → pts i.1 ≠ pts j.1) →
  ∃ (sq_x sq_y: ℝ), sq_x ≥ 0 ∧ sq_x ≤ 82 ∧ sq_y ≥ 0 ∧ sq_y ≤ 82 ∧
    (∃ k, k = 25 ∧ 
      (∃ (idx : Finset (Fin 2000)), idx.card = k ∧
        (∀ a ∈ idx, (sq_x ≤ (pts a).1 ∧ (pts a).1 < sq_x + 1) ∧ (sq_y ≤ (pts a).2 ∧ (pts a).2 < sq_y + 1)))) :=
by
  sorry

end largest_n_unit_square_l662_662863


namespace line_inclination_angle_l662_662608

theorem line_inclination_angle :
  ∀ (x y : ℝ), (x + real.sqrt 3 * y - 1 = 0) → (∃ θ, θ = real.arctan (-real.sqrt 3 / 3) ∧ θ = 150) :=
by
  assume x y h_line,
  sorry 
  -- Replace sorry with the actual proof steps.

end line_inclination_angle_l662_662608


namespace vinegar_evaporation_rate_l662_662679

def percentage_vinegar_evaporates_each_year (x : ℕ) : Prop :=
  let initial_vinegar : ℕ := 100
  let vinegar_left_after_first_year : ℕ := initial_vinegar - x
  let vinegar_left_after_two_years : ℕ := vinegar_left_after_first_year * (100 - x) / 100
  vinegar_left_after_two_years = 64

theorem vinegar_evaporation_rate :
  ∃ x : ℕ, percentage_vinegar_evaporates_each_year x ∧ x = 20 :=
by
  sorry

end vinegar_evaporation_rate_l662_662679


namespace area_trapezoid_EFBA_l662_662525

def Rectangle (A B C D : Type) (area : ℝ) : Prop := 
  ∃ (l w : ℝ), l * w = area

def PointsOnSides (A D B C E F : Type) (AE BF : ℝ) : Prop :=
  (AE = 2) ∧ (BF = 2)

theorem area_trapezoid_EFBA :
  ∀ (A B C D E F : Type),
    Rectangle A B C D 20 →
    PointsOnSides A D B C E F 2 →
    ∃ (area : ℝ), area = 6 :=
by
  intros A B C D E F hrect hps
  -- Proof steps would go here
  sorry  -- Proof is intentionally left as sorry

#check area_trapezoid_EFBA

end area_trapezoid_EFBA_l662_662525


namespace sum_of_first_n_terms_l662_662025

noncomputable def arithmetic_sequence (n : ℕ) : ℕ → ℤ := sorry

theorem sum_of_first_n_terms (a : ℕ → ℤ) (d : ℤ) (h1 : d = 2) (h2 : ∀ n : ℕ, a (n + 1) = a n + d) (h3 : (a 1 + 6) ^ 2 = (a 1 + 2) * (a 1 + 14)) :
  (∀ n : ℕ, (range n).sum (λ i, a i) = n^2 + n) :=
sorry

end sum_of_first_n_terms_l662_662025


namespace volume_of_quadrangular_pyramid_eq_l662_662343

noncomputable def volume_of_quadrangular_pyramid (V : ℝ) 
    (AP_eq_C1Q : ℝ) -- Condition: AP = C1Q
    (points_on_edges : true) : ℝ := -- Points P and Q on the edges AA1 and OC1

  let B_APQC_volume := V / 3 in
  B_APQC_volume

theorem volume_of_quadrangular_pyramid_eq (V : ℝ) 
    (AP_eq_C1Q : ℝ) -- Condition: AP = C1Q
    (points_on_edges : true) : volume_of_quadrangular_pyramid V AP_eq_C1Q points_on_edges = V / 3 :=
by
  sorry

end volume_of_quadrangular_pyramid_eq_l662_662343


namespace volume_of_solid_of_revolution_l662_662942

theorem volume_of_solid_of_revolution :
  let parabola := λ x, -x^2 + 4,
      ellipse := λ x y, 16 * x^2 + 25 * y^2 = 400,
      volume := 29.96 * Real.pi
  in
  ∃ (V : ℝ), V = volume :=
begin
  sorry
end

end volume_of_solid_of_revolution_l662_662942


namespace sum_of_seven_consecutive_integers_l662_662576

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l662_662576


namespace sum_of_numbers_l662_662266

theorem sum_of_numbers : 
  12345 + 23451 + 34512 + 45123 + 51234 = 166665 := 
sorry

end sum_of_numbers_l662_662266


namespace find_f_l662_662902

-- Given conditions and definitions
def f (n : ℕ+) : ℕ+ := sorry

axiom gcd_condition : Nat.gcd (f 1) (Nat.gcd (f 2) (Nat.gcd (f 3) ...)) = 1
axiom large_n_condition {a b n : ℕ+} : ∃ N, ∀ n > N, f n ≠ 1
axiom divisibility_condition {a b n : ℕ+} : ∃ N, ∀ n > N, f(a)^n ∣ (f(a + b))^(a^(n-1)) - (f(b))^(a^(n-1))

-- The theorem to prove
theorem find_f : ∀ (x : ℕ+), f x = x := sorry

end find_f_l662_662902


namespace james_lifting_heavy_after_39_days_l662_662874

noncomputable def JamesInjuryHealingTime : Nat := 3
noncomputable def HealingTimeFactor : Nat := 5
noncomputable def WaitingTimeAfterHealing : Nat := 3
noncomputable def AdditionalWaitingTimeWeeks : Nat := 3

theorem james_lifting_heavy_after_39_days :
  let healing_time := JamesInjuryHealingTime * HealingTimeFactor
  let total_time_before_workout := healing_time + WaitingTimeAfterHealing
  let additional_waiting_time_days := AdditionalWaitingTimeWeeks * 7
  let total_time_before_lifting_heavy := total_time_before_workout + additional_waiting_time_days
  total_time_before_lifting_heavy = 39 := by
  sorry

end james_lifting_heavy_after_39_days_l662_662874


namespace sum_abs_sin_pi_l662_662901

theorem sum_abs_sin_pi (n : ℕ) (h : n = 2010) : 
  ∑ k in finset.range (n + 1) \{0}, |Real.sin (k * π)| = 0 :=
by
  sorry

end sum_abs_sin_pi_l662_662901


namespace transformed_area_l662_662527

variable (T : Type) [MeasureTheory.MeasureSpace T] [MeasureTheory.Measure T] 

def area (s : T) : ℝ :=
  MeasureTheory.MeasureTheory.Measure.of R sorry

noncomputable def transformation_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
![
  ![3, 2],
  ![1, 4]
]

noncomputable def det_transformation_matrix : ℝ :=
  Matrix.det transformation_matrix

theorem transformed_area (area_T : ℝ) (h_area_T : area T = 9) : 
  area (Matrix.mvMul transformation_matrix T) = 90 :=
by
  simp [det_transformation_matrix]
  sorry

end transformed_area_l662_662527


namespace percent_of_projected_revenue_is_50_l662_662171

-- Define the revenues and percentages
variable (R : ℝ)
def projected_revenue := 1.40 * R
def actual_revenue := 0.70 * R

-- Define the percentage of actual revenue to projected revenue
def percentage_of_actual_to_projected := (actual_revenue / projected_revenue) * 100

-- Prove that the actual revenue is 50% of the projected revenue
theorem percent_of_projected_revenue_is_50 :
  percentage_of_actual_to_projected R = 50 :=
by
  -- Insert the proof here
  sorry

end percent_of_projected_revenue_is_50_l662_662171


namespace compound_interest_example_l662_662509

-- Define the compound interest problem conditions
def P : ℝ := 10000
def r : ℝ := 0.0396
def t : ℝ := 2
def A : ℝ := 10815.83

-- Prove that n = 2 satisfies the compound interest formula
theorem compound_interest_example : 
  ∃ (n : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ n = 2 :=
by
  use 2
  sorry

end compound_interest_example_l662_662509


namespace find_m_l662_662441

-- We define the universal set U, the set A with an unknown m, and the complement of A in U.
def U : Set ℕ := {1, 2, 3}
def A (m : ℕ) : Set ℕ := {1, m}
def complement_U_A (m : ℕ) : Set ℕ := U \ A m

-- The main theorem where we need to prove m = 3 given the conditions.
theorem find_m (m : ℕ) (hU : U = {1, 2, 3})
  (hA : ∀ m, A m = {1, m})
  (h_complement : complement_U_A m = {2}) : m = 3 := sorry

end find_m_l662_662441


namespace triangle_A1B1C1_equilateral_l662_662873

variable {A B C : Type*} [InnerProductSpace ℝ (fin 3)]

-- Define points A, B, C
variables (A B C B1 C1 A1 : (fin 3))

-- Condition: \(\angle C + \angle B = 120^\circ\)
axiom ang_C_plus_ang_B : ∡C + ∡B = 120

-- Condition: In triangle ABC, altitudes BB1 and CC1 and the median AA1 are drawn
axiom altitude_BB1 : altitude A B B1
axiom altitude_CC1 : altitude A C C1
axiom median_AA1 : median A B C A1

-- Prove that triangle A1B1C1 is equilateral
theorem triangle_A1B1C1_equilateral : is_equilateral (triangle A1 B1 C1) := 
sorry

end triangle_A1B1C1_equilateral_l662_662873


namespace cot_diff_of_median_angle_l662_662498

theorem cot_diff_of_median_angle (A B C D : ℝ) (h1 : is_midpoint D B C)
  (h2 : ∠AD BC = 45) : |cot ∠B - cot ∠C| = 2 := 
sorry

end cot_diff_of_median_angle_l662_662498


namespace range_f_l662_662384

noncomputable def f (x : ℝ) : ℝ := Real.arcsin x + Real.arccos x + Real.log (1 + x)

theorem range_f : 
  (set_of (λ y, ∃ x : ℝ, -1 < x ∧ x ≤ 1 ∧ y = f x)) = Iic (Real.pi / 2 + Real.log 2) :=
by
  sorry

end range_f_l662_662384


namespace train_six_circuits_time_l662_662311

-- Definitions based on conditions
def time_per_circuit_minutes : ℕ := 1
def time_per_circuit_seconds : ℕ := 11
def circuits : ℕ := 6

-- Problem statement
theorem train_six_circuits_time :
  let total_time_per_circuit := time_per_circuit_minutes * 60 + time_per_circuit_seconds in
  let total_time_seconds := circuits * total_time_per_circuit in
  let minutes := total_time_seconds / 60 in
  let seconds := total_time_seconds % 60 in
  minutes = 7 ∧ seconds = 6 :=
by
  sorry

end train_six_circuits_time_l662_662311


namespace tan_alpha_condition_l662_662748

theorem tan_alpha_condition
  (α : ℝ) 
  (h1 : 0 < α ∧ α < π / 2)
  (h2 : sin α + sin (α + π / 3) + sin (α + 2 * π / 3) = sqrt 3) :
  tan α = sqrt 3 := 
  sorry

end tan_alpha_condition_l662_662748


namespace probability_of_common_books_l662_662919

-- Definitions based on conditions
def total_ways_4_books (n : ℕ) (k : ℕ) : ℕ :=
  nat.choose n k

def favorable_outcomes (n k : ℕ) (common_books : ℕ) :=
  nat.choose n common_books * nat.choose (n - common_books) k * nat.choose (n - common_books - (k - common_books)) k

def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

-- Given the conditions of the problem
def problem_statement : Prop :=
  let total := (total_ways_4_books 12 4) * (total_ways_4_books 12 4) in
  let favorable := favorable_outcomes 12 4 2 in
  probability favorable total = 36 / 105

-- The proof is omitted with sorry
theorem probability_of_common_books :
  problem_statement :=
sorry

end probability_of_common_books_l662_662919


namespace complex_magnitude_theorem_l662_662536

theorem complex_magnitude_theorem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  ∣(1 / z) + (1 / w)∣ = 3 / 8 := by
  sorry

end complex_magnitude_theorem_l662_662536


namespace euler_formula_for_convex_polyhedra_l662_662860

theorem euler_formula_for_convex_polyhedra (F V E : ℕ) (h : convex F V E) : V + F - E = 2 := 
sorry

end euler_formula_for_convex_polyhedra_l662_662860


namespace find_fifth_term_l662_662609

noncomputable def geometric_sequence_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : ℝ :=
  a * r^4

theorem find_fifth_term (a r : ℝ) (h₁ : a * r^2 = 16) (h₂ : a * r^6 = 2) : geometric_sequence_fifth_term a r h₁ h₂ = 2 := sorry

end find_fifth_term_l662_662609


namespace no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l662_662701

theorem no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0 :
  ¬ ∃ n : ℕ, (n^2 + 6 * n + 2019) % 100 = 0 :=
sorry

end no_natural_number_such_that_n_sq_plus_6n_plus_2019_mod_100_eq_0_l662_662701


namespace num_mystery_shelves_l662_662184

def num_books_per_shelf : ℕ := 9
def num_picture_shelves : ℕ := 2
def total_books : ℕ := 72
def num_books_from_picture_shelves : ℕ := num_picture_shelves * num_books_per_shelf
def num_books_from_mystery_shelves : ℕ := total_books - num_books_from_picture_shelves

theorem num_mystery_shelves :
  num_books_from_mystery_shelves / num_books_per_shelf = 6 := by
sorry

end num_mystery_shelves_l662_662184


namespace smallest_x_l662_662263

theorem smallest_x (x : ℕ) (h900 : 900 = 2^2 * 3^2 * 5^2) (h1152 : 1152 = 2^7 * 3^2) : 
  (900 * x) % 1152 = 0 ↔ x = 32 := 
by
  sorry

end smallest_x_l662_662263


namespace gcd_u_n_u_n_plus_3_l662_662987

def sequence (u : ℕ → ℕ) : Prop :=
  u 1 = 1 ∧
  u 2 = 1 ∧ 
  ∀ n, n ≥ 3 → u n = u (n - 1) + 2 * u (n - 2)

theorem gcd_u_n_u_n_plus_3 (u : ℕ → ℕ) (h : sequence u) (n : ℕ) : 
  gcd (u n) (u (n + 3)) = if n % 3 = 0 then 3 else 1 :=
sorry

end gcd_u_n_u_n_plus_3_l662_662987


namespace compare_squares_l662_662730

theorem compare_squares (a : ℝ) : (a + 1)^2 > a^2 + 2 * a := by
  -- the proof would go here, but we skip it according to the instruction
  sorry

end compare_squares_l662_662730


namespace number_power_eq_l662_662301

theorem number_power_eq (x : ℕ) (h : x^10 = 16^5) : x = 4 :=
by {
  -- Add supporting calculations here if needed
  sorry
}

end number_power_eq_l662_662301


namespace isosceles_right_triangle_inequality_l662_662888

variable {a x : ℝ}
variable {AP PC BP : ℝ → ℝ}

noncomputable def P (x : ℝ) : Prop :=  0 ≤ x ∧ x ≤ a
noncomputable def AP (x : ℝ) : ℝ := x
noncomputable def PC (x : ℝ) : ℝ := a - x
noncomputable def s (x : ℝ) : ℝ := (AP x) ^ 2 + (PC x)^ 2
noncomputable def BP (x : ℝ) : ℝ := (a^2 + (AP x)^2)^(1/2)

theorem isosceles_right_triangle_inequality (a : ℝ) (h : ∀ x, P x → s x < 2 * (BP x)^2) :
  ∀ P : ℝ, P a → s P < 2 * (BP P) ^ 2 :=
by 
  intro P
  sorry

end isosceles_right_triangle_inequality_l662_662888


namespace square_root_rounding_impossible_l662_662615

open Nat

theorem square_root_rounding_impossible (a d : ℕ) (h_d : d ≠ 0) :
  ¬(∀ n, ∃ k : ℤ, round (sqrt (a + n * d : ℕ)) = k * sign d) :=
sorry

end square_root_rounding_impossible_l662_662615


namespace students_mistakes_l662_662472

theorem students_mistakes (x y z : ℕ) (h : x + y + z = 333) (ht : 4 * y + 6 * z ≤ 1000) :
  z ≤ x :=
begin
  sorry
end

end students_mistakes_l662_662472


namespace number_of_valid_alphas_l662_662891

-- Defining the set of values for alpha
def alpha_set : set ℝ := {-3, -2, -1, -1/2, 1/2, 1, 2, 3}

-- Defining a function to check if y = x^alpha is an odd function
def is_odd_function (α : ℝ) : Prop :=
  ∀ x : ℝ, x > 0 → (x^α = -((-x)^α))

-- Defining a function to check if y = x^alpha is monotonically decreasing on (0, +∞)
def is_monotonically_decreasing (α : ℝ) : Prop :=
  ∀ x y : ℝ, 0 < x → x < y → (y^α < x^α)

-- The main theorem to prove
theorem number_of_valid_alphas : (finset.card (finset.filter (λ α, is_odd_function α ∧ is_monotonically_decreasing α) (finset.filter (λ x, x ∈ alpha_set) (finset.range (-4..4))))) = 2 :=
by
  sorry

end number_of_valid_alphas_l662_662891


namespace num_mappings_is_seven_l662_662439

noncomputable def num_mappings (M N : Type) [Fintype M] [Fintype N] (f : M → N) : ℕ :=
  Fintype.card {f : M → N // ∑ m in (Finset.univ : Finset M), f m = 0}

def problem_conditions : Prop :=
  {a, b, c : ℕ} ∧ {-1, 0, 1 : ℤ} ∧ (∀ f : ℕ → ℤ, ∑ x in {a, b, c}, f x = 0) 

theorem num_mappings_is_seven :
  num_mappings {a, b, c} {-1, 0, 1} (λ x, if x = a then -1 else if x = b then 0 else 1) = 7 :=
sorry

end num_mappings_is_seven_l662_662439


namespace hex_minus_20F_to_dec_l662_662476

/-- Define a function to convert a single hexadecimal digit to its decimal equivalent. -/
def hex_digit_to_dec : Char → ℕ 
| '0' := 0
| '1' := 1
| '2' := 2
| '3' := 3
| '4' := 4
| '5' := 5
| '6' := 6
| '7' := 7
| '8' := 8
| '9' := 9
| 'A' := 10
| 'B' := 11
| 'C' := 12
| 'D' := 13
| 'E' := 14
| 'F' := 15
| _ := 0  -- Assuming valid input for simplicity, invalid characters map to 0

/-- Convert a hexadecimal string to its decimal equivalent. -/
def hex_to_dec : String → ℤ
| ⟨[], h⟩ := 0
| ⟨c :: cs, h⟩ := 
  let n := hex_digit_to_dec c.toUpper in
  n * (16 : ℤ) ^ cs.length + (hex_to_dec ⟨cs, _⟩)

/-- Proof that the decimal equivalent of the hexadecimal number "-20F" is -527. -/
theorem hex_minus_20F_to_dec : hex_to_dec "-20F" = -527 := 
by {
  -- This proof would normally follow the steps of the manual conversion process,
  -- breaking down the string, converting each part, and combining the results.
  sorry
}

end hex_minus_20F_to_dec_l662_662476


namespace power_factor_200_l662_662883

theorem power_factor_200 :
  (let a := 3 in let b := 2 in (1 / 3)^(b - a) = 3) :=
by
  -- assume a and b definitions
  let a := 3
  let b := 2
  -- main statement
  show (1 / 3) ^ (b - a) = 3
  -- we skip the proof
  sorry

end power_factor_200_l662_662883


namespace triangle_area_correct_l662_662255

noncomputable def triangle_area_formed_by_lines : ℝ :=
  let p1 := (6, 7) in
  let p2 := (-6, 7) in
  let p3 := (0, 1) in
  1/2 * abs ((6 * 7 + (-6) * 1 + 0 * 7) - (7 * (-6) + 1 * 0 + 7 * 6))

theorem triangle_area_correct :
  triangle_area_formed_by_lines = 36 := by
  sorry

end triangle_area_correct_l662_662255


namespace imaginary_part_of_z_is_neg_one_l662_662596

noncomputable def z : ℂ := (3 + complex.I) / (1 + complex.I)

theorem imaginary_part_of_z_is_neg_one : z.im = -1 :=
by
  sorry

end imaginary_part_of_z_is_neg_one_l662_662596


namespace sum_a_n_l662_662036

theorem sum_a_n (a : ℕ → ℕ) :
  a 1 = 0 →
  a 2 = 2 →
  (∀ n ≥ 2, a (n+1) + a n = 2^n) →
  (∑ n in finset.range 2018, a (n+1)) = (2 / 3) * (4^1009 - 1) := 
by
  sorry

end sum_a_n_l662_662036


namespace parabola_b_value_l662_662218

noncomputable def parabola_properties (a b c h : ℝ) : Prop :=
  (∀ x : ℝ, a * (x - h) ^ 2 + 2 * h = a * x ^ 2 + b * x + c) ∧ -- Vertex form condition
  c = -3 * h ∧ -- y-intercept condition
  h ≠ 0        -- h non-zero condition

theorem parabola_b_value (a b c h : ℝ) (hc : parabola_properties a b c h) : b = 10 := 
by
  cases hc with vertex_form_rest hc_non_zero
  sorry

end parabola_b_value_l662_662218


namespace triangle_angle_division_l662_662927

theorem triangle_angle_division (α β γ : ℝ) (B C A' B' C' : Point) 
  (ABC_is_triangle : Triangle B C A') 
  (isosceles1 : IsoscelesTriangle B C A' α) 
  (isosceles2 : IsoscelesTriangle A' C B' β) 
  (isosceles3 : IsoscelesTriangle A' B C' γ) 
  (angle_sum : α + β + γ = 2 * Real.pi) : 
  angles_of_triangle (triangle A' B' C') = (α / 2, β / 2, γ / 2) := 
sorry

end triangle_angle_division_l662_662927


namespace pirate_treasure_l662_662307

theorem pirate_treasure (N : ℕ) (h1 : ∀ k : ℕ, k < 15 → 15 ∣ (N * (15 - k - 1) ^ (14 - k))) :
  N = 208080 :=
begin
  sorry
end

#print axioms pirate_treasure

end pirate_treasure_l662_662307


namespace largest_number_4597_l662_662640

def swap_adjacent_digits (n : ℕ) : ℕ :=
  sorry

def max_number_after_two_swaps_subtract_100 (n : ℕ) : ℕ :=
  -- logic to perform up to two adjacent digit swaps and subtract 100
  sorry

theorem largest_number_4597 : max_number_after_two_swaps_subtract_100 4597 = 4659 :=
  sorry

end largest_number_4597_l662_662640


namespace range_of_a_l662_662434

noncomputable def f (a x : ℝ) : ℝ := (sqrt (3 - a * x)) / (a - 1)

theorem range_of_a (a : ℝ) (h₀ : a ≠ 1) (h₁ : ∀ x ∈ Ioc 0 1, f a x < f a x.succ) :
  a ∈ Iio 0 ∪ Ioc 1 3 := sorry

end range_of_a_l662_662434


namespace angle_between_hands_at_3_15_l662_662778

theorem angle_between_hands_at_3_15 :
  let hour_angle_at_3 := 3 * 30
  let hour_hand_move_rate := 0.5
  let minute_angle := 15 * 6
  let hour_angle_at_3_15 := hour_angle_at_3 + 15 * hour_hand_move_rate
  abs (hour_angle_at_3_15 - minute_angle) = 7.5 := 
by
  sorry

end angle_between_hands_at_3_15_l662_662778


namespace temperature_increase_l662_662869

variable (T_morning T_afternoon : ℝ)

theorem temperature_increase : 
  (T_morning = -3) → (T_afternoon = 5) → (T_afternoon - T_morning = 8) :=
by
intros h1 h2
rw [h1, h2]
sorry

end temperature_increase_l662_662869


namespace remaining_sweets_in_packet_l662_662547

theorem remaining_sweets_in_packet 
  (C : ℕ) (S : ℕ) (P : ℕ) (R : ℕ) (L : ℕ)
  (HC : C = 30) (HS : S = 100) (HP : P = 60) (HR : R = 25) (HL : L = 150) 
  : (C - (2 * C / 5) - ((C - P / 4) / 3)) 
  + (S - (S / 4)) 
  + (P - (3 * P / 5)) 
  + ((max 0 (R - (3 * R / 2))))
  + (L - (3 * (S / 4) / 2)) = 232 :=
by
  sorry

end remaining_sweets_in_packet_l662_662547


namespace range_of_a_for_extrema_l662_662069

noncomputable def f (x a : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + x

def discriminant (a : ℝ) : ℝ := (2 * a)^2 - 4 * 1 * 1

theorem range_of_a_for_extrema (a : ℝ) : 
  (∀ x : ℝ, f x a has one_maximum_and_one_minimum_value) ↔ (a < -1 ∨ a > 1) := 
sorry

end range_of_a_for_extrema_l662_662069


namespace problem_statement_l662_662904

variable {m n : ℕ}
variable {T S : Finset ℕ}

-- Condition: m > n ≥ 2
def conditions : Prop := m > n ∧ n ≥ 2 

-- Condition: S = {1, 2, ..., m}
def set_S : Finset ℕ := Finset.range (m + 1)

-- Condition: T ⊆ S and any two numbers in T cannot both divide any number in S
def valid_set (T : Finset ℕ) (set_S : Finset ℕ) : Prop :=
  (T ⊆ set_S) ∧ ∀ a b ∈ T, a ≠ b → ∀ s ∈ set_S, ¬ (a ∣ s ∧ b ∣ s)

-- Main statement to prove
theorem problem_statement (h1 : conditions) (h2 : valid_set T set_S) :
  (∑ x in T, (1 : ℚ) / x) < (↑m + ↑n) / ↑m := 
sorry

end problem_statement_l662_662904


namespace false_proposition_l662_662933

variables {x y : ℝ}

def p (x y : ℝ) : Prop := (sin x > sin y) → (x > y)
def q (x y : ℝ) : Prop := x^2 + y^2 ≥ 2 * x * y

theorem false_proposition :
  ∃ x y : ℝ, ¬ (p x y ∧ q x y) :=
begin
  use [π / 2, π],
  simp [p, q, sin_pi, sin_le_iff, sin_ge_iff],
  split,
  { intros h, exfalso,
    exact not_lt_of_gt (by norm_num : sin (π / 2) < sin π) h, },
  { calc
    (π / 2)^2 + π^2 ≥ 2 * (π / 2) * π : by norm_num }
end

end false_proposition_l662_662933


namespace minimum_triangle_area_l662_662125

-- Define points and distances
variable (P A M N : Point)
variable (PD AD : ℝ)

-- Define given conditions
variable (tan_MAN : ℝ)
variable (inside_angle : Point → Point → Point → Prop)
variable (distance : Point → Point → ℝ)

-- Define triangle area calculation
def triangle_area (A B C : Point) : ℝ := sorry  -- Assume we have calculated triangle area

-- Assert our conditions
axiom h1 : inside_angle P A M N
axiom h2 : tan_MAN = 3
axiom h3 : distance P (line A N) = 12
axiom h4 : distance A D = 30

-- Prove the minimum area of triangle ABC
theorem minimum_triangle_area :
  ∃ B C : Point, inside_angle P A M N ∧
    distance P (line A N) = 12 ∧ 
    distance A D = 30 ∧ 
    triangle_area A B C = 624 :=
by
  sorry

end minimum_triangle_area_l662_662125


namespace smallest_y_exists_l662_662314

def x : ℕ := 11 * 36 * 54

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

theorem smallest_y_exists : ∃ y : ℕ, is_perfect_square (x * y) ∧ (∀ z : ℕ, is_perfect_square (x * z) → y ≤ z) :=
begin
  use 66,
  split,
  { sorry, -- Prove that 11 * 36 * 54 * 66 is a perfect square
  },
  { intros z hz,
    sorry, -- Prove that 66 is the smallest such number
  }
end

end smallest_y_exists_l662_662314


namespace range_of_m_l662_662408

noncomputable def proposition_p (m : ℝ) : Prop :=
∀ x : ℝ, x^2 + m * x + 1 ≥ 0

noncomputable def proposition_q (m : ℝ) : Prop :=
∀ x : ℝ, (8 * x + 4 * (m - 1)) ≥ 0

def conditions (m : ℝ) : Prop :=
(proposition_p m ∨ proposition_q m) ∧ ¬(proposition_p m ∧ proposition_q m)

theorem range_of_m (m : ℝ) : 
  conditions m → ( -2 ≤ m ∧ m < 1 ) ∨ m > 2 :=
by
  intros h
  sorry

end range_of_m_l662_662408


namespace correct_propositions_l662_662571

noncomputable def f (x : ℝ) : ℝ :=
  Real.cos (2 * x - Real.pi / 3) + Real.cos (2 * x + Real.pi / 6)

def prop1 (y : ℝ) : Prop :=
  ∀ x, y = f x → y ≤ Real.sqrt 2

def prop2 (y : ℝ) : Prop :=
  ∀ x, y = f x → ∃ T > 0, f (x + T) = f x ∧ ∀ t > 0, f (x + t) = f x → t = T

def prop3 (y : ℝ) : Prop :=
  ∀ x, x > Real.pi / 24 ∧ x < 13 * Real.pi / 24 → f' x < 0

def prop4 (y₁ y₂ : ℝ) : Prop :=
  ∀ x, y₁ = Real.sqrt 2 * Real.cos (2 * x) → y₂ = f x → 
        y₁ = Real.sqrt 2 * Real.cos (2 * (x + Real.pi / 24))

theorem correct_propositions : 
  ∀ y₁ y₂ : ℝ, prop1 (f y₁) ∧ prop2 (f y₁) ∧ prop3 (f y₁) ∧ ¬ prop4 y₁ y₂ :=
by 
  sorry

end correct_propositions_l662_662571


namespace find_m_n_l662_662486

variables {R : Type*} [RealField R] (OA OB OC : EuclideanSpace R) (m n : R)

-- Define the conditions
def norm_OA : ∥OA∥ = 1 := sorry
def norm_OB : ∥OB∥ = 2 := sorry
def norm_OC : ∥OC∥ = sqrt 3 := sorry
def tan_angle_AOC : Real.tan (angle_between OA OC) = 3 := sorry
def angle_BOC : angle_between OB OC = Real.pi / 3 := sorry

-- Prove that (m, n) satisfies OC = m * OA + n * OB
theorem find_m_n : OC = (1 / 2 * (1 + 3 * sqrt 3)) • OA + (sqrt 3 / (4 * sqrt 10)) • OB := sorry

end find_m_n_l662_662486


namespace sequence_fifth_term_l662_662871

theorem sequence_fifth_term :
  let a : ℕ → ℤ := λ n, if n = 1 then 2 else if n = 2 then 5 else
    (a (n-2) + a (n-1))
  in a 5 = 19 :=
by
  let a : ℕ → ℤ := λ n, if n = 1 then 2 else if n = 2 then 5 else
    (a (n-2) + a (n-1))
  sorry

end sequence_fifth_term_l662_662871


namespace equivalent_integral_function_count_l662_662046

def f1 (x : ℝ) : ℝ := 2 * |x|
def g1 (x : ℝ) : ℝ := x + 1

def f2 (x : ℝ) : ℝ := sin x
def g2 (x : ℝ) : ℝ := cos x

def f3 (x : ℝ) : ℝ := sqrt (1 - x^2)
def g3 (x : ℝ) : ℝ := (3 / 4) * real.pi * x^2

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f(x)

theorem equivalent_integral_function_count :
  (∫ x in -1..1, f1 x) = (∫ x in -1..1, g1 x) →
  (∫ x in -1..1, f2 x) ≠ (∫ x in -1..1, g2 x) →
  (∫ x in -1..1, f3 x) = (∫ x in -1..1, g3 x) →
  (∀ f g : ℝ → ℝ, odd_function f → odd_function g → 
    (∫ x in -1..1, f x) = (∫ x in -1..1, g x)) →
  3 :=
by 
  sorry

end equivalent_integral_function_count_l662_662046


namespace triangle_condition_isosceles_l662_662499

variables {V : Type*} [inner_product_space ℝ V]
variables (A B C : V)

def is_isosceles (A B C : V) : Prop :=
  dist A B = dist A C

theorem triangle_condition_isosceles
  (h : (⟪(B - A) / ∥B - A∥ + (C - A) / ∥C - A∥, B - C⟫ ℝ) = 0) : is_isosceles A B C :=
by sorry

end triangle_condition_isosceles_l662_662499


namespace solve_problem_l662_662067

noncomputable def f (x : ℝ) : ℝ := x * Real.sin x + Real.cos x + x^2

def problem_statement : set ℝ := { x | f (Real.log x) + f (Real.log (1 / x)) < 2 * f 1 }

def solution_set : set ℝ := { x | 1 / Real.exp 1 < x ∧ x < Real.exp 1 }

theorem solve_problem : problem_statement = solution_set :=
by
  sorry

end solve_problem_l662_662067


namespace sum_total_ages_correct_l662_662996

variable father_age : ℕ
variable son_age : ℕ

def sum_total_ages (father_age son_age : ℕ) : ℕ :=
  father_age + son_age

theorem sum_total_ages_correct :
  sum_total_ages 37 18 = 55 :=
by
  unfold sum_total_ages
  sorry

end sum_total_ages_correct_l662_662996


namespace john_paid_same_as_maria_l662_662915

theorem john_paid_same_as_maria :
  let num_slices := 10
  let plain_cost := 10
  let olive_extra_cost := 3
  let pizza_cost := plain_cost + olive_extra_cost
  let cost_per_slice := pizza_cost / num_slices
  -- John ate all 3 olive slices and 2 plain slices
  let john_olive_slices := 3
  let john_plain_slices := 2
  let john_total_slices := john_olive_slices + john_plain_slices
  let john_cost := (john_olive_slices * cost_per_slice) + (john_plain_slices * cost_per_slice)
  -- Maria ate the remaining plain slices
  let maria_plain_slices := num_slices - john_total_slices
  let maria_cost := maria_plain_slices * cost_per_slice
  john_cost = maria_cost :=
by 
  -- Definitions
  let num_slices := 10
  let plain_cost := 10
  let olive_extra_cost := 3
  let pizza_cost := plain_cost + olive_extra_cost
  let cost_per_slice := pizza_cost / num_slices
  
  -- Calculations for John
  let john_olive_slices := 3
  let john_plain_slices := 2
  let john_total_slices := john_olive_slices + john_plain_slices
  let john_cost := (john_olive_slices * cost_per_slice) + (john_plain_slices * cost_per_slice)
  
  -- Calculations for Maria
  let maria_plain_slices := num_slices - john_total_slices
  let maria_cost := maria_plain_slices * cost_per_slice

  -- Proving equality
  have h1 : pizza_cost = 13 := by linarith
  have h2 : cost_per_slice = 1.3 := by linarith
  
  have h3 : john_cost = 6.5 :=
    by calc
      john_cost = (3 * 1.3) + (2 * 1.3) : by rw [john_olive_slices, john_plain_slices, cost_per_slice]
            ... = 3.9 + 2.6 : by norm_num
            ... = 6.5 : by norm_num
  
  have h4 : maria_cost = 6.5 := 
    by calc
      maria_cost = (num_slices - (john_olive_slices + john_plain_slices)) * 1.3 : by rw [num_slices, john_olive_slices, john_plain_slices, cost_per_slice]
              ... = 5 * 1.3 : by norm_num
              ... = 6.5 : by norm_num
  
  show john_cost = maria_cost, from eq.trans h3.symm h4.symm
  
  exact eq.symm h4.symm

end john_paid_same_as_maria_l662_662915


namespace circle_properties_l662_662906

noncomputable def circle_center_and_radius (x y : ℝ) : ℝ × ℝ × ℝ :=
  let eq1 := x^2 - 4 * y - 18
  let eq2 := -y^2 + 6 * x + 26
  let lhs := x^2 - 6 * x + y^2 - 4 * y
  let rhs := 44
  let center_x := 3
  let center_y := 2
  let radius := Real.sqrt 57
  let target := 5 + radius
  (center_x, center_y, target)

theorem circle_properties
  (x y : ℝ) :
  let (a, b, r) := circle_center_and_radius x y 
  a + b + r = 5 + Real.sqrt 57 :=
by
  sorry

end circle_properties_l662_662906


namespace partition_natural_numbers_l662_662638

open Set

theorem partition_natural_numbers :
  ∃ (P : ℕ → Fin 100), (∀ a b c : ℕ, a + 99 * b = c → ∃ i : Fin 100, (P a = i ∧ P c = i) ∨ (P a = i ∧ P b = i) ∨ (P b = i ∧ P c = i)) ∧ 
  (∀ i : Fin 100, ∃ n : ℕ, P n = i) :=
begin
  sorry
end

end partition_natural_numbers_l662_662638


namespace perpendicular_OP_AE_l662_662113

noncomputable def cyclic_pentagon (A B C D E : Point) (O : Circle) : Prop :=
inscribed_pentagon A B C D E O

noncomputable def intersect_at (A B C D : Point) (F : Point) : Prop :=
line A D ∩ line B E = {F}

noncomputable def extension_meets_circle (C F P : Point) (O : Circle) : Prop :=
∃ P' : Point, (P' ≠ F ∧ P' ≠ C) ∧ P' ∈ CirclePoints O ∧ line C F ∩ CirclePoints O = {F, P'}

noncomputable def segment_relation (A B C D E : Point) : Prop :=
segment_length A B * segment_length C D = segment_length B C * segment_length E D

theorem perpendicular_OP_AE (A B C D E F P : Point) (O : Circle) :
  cyclic_pentagon A B C D E O →
  intersect_at A B C D F →
  extension_meets_circle C F P O →
  segment_relation A B C D E →
  perpendicular (radius O P) (line A E) :=
sorry

end perpendicular_OP_AE_l662_662113


namespace six_digit_no_zero_divisible_by_10_l662_662834

noncomputable def probability_divisible_by_10 (digits : set ℕ) (number_len : ℕ) : ℝ :=
  if ∃ d, d ∈ digits ∧ d = 0 then 1
  else 0

theorem six_digit_no_zero_divisible_by_10 (digits : set ℕ) 
(digit_condition: digits = {1, 2, 3, 4, 5, 8}) 
(number_len : ℕ)
(is_six_digits : number_len = 6) :
  probability_divisible_by_10 digits number_len = 0 :=
begin
  sorry
end

end six_digit_no_zero_divisible_by_10_l662_662834


namespace coeff_x_squared_l662_662358

-- Define the two polynomials as given in the conditions
def poly1 : Polynomial ℝ := 3 * X^3 - 4 * X^2 + 5 * X - 2
def poly2 : Polynomial ℝ := 2 * X^2 - 3 * X + 4

-- Formulate the proof problem as a Lean theorem statement
theorem coeff_x_squared :
  (poly1 * poly2).coeff 2 = -35 :=
by
  sorry

end coeff_x_squared_l662_662358


namespace log_8_80_eq_l662_662094

variable (x y : ℝ)

-- Given conditions
def log_10_2 := x
def log_10_5 := y

-- Proving the question
theorem log_8_80_eq : log_base 8 80 = (4 * x + y) / (3 * x) :=
by 
  sorry

end log_8_80_eq_l662_662094


namespace xy_value_l662_662251

theorem xy_value {x y : ℝ} (h1 : x - y = 6) (h2 : x^3 - y^3 = 162) : x * y = 21 :=
by
  sorry

end xy_value_l662_662251


namespace simple_interest_rate_l662_662635

theorem simple_interest_rate (P : ℝ) (R : ℝ) (T : ℝ) (SI : ℝ) :
  (T = 20) →
  (SI = P) →
  (SI = P * R * T / 100) →
  R = 5 :=
by
  sorry

end simple_interest_rate_l662_662635


namespace fraction_sum_is_integer_l662_662568

theorem fraction_sum_is_integer (n : ℤ) : 
  ∃ k : ℤ, (n / 3 + (n^2) / 2 + (n^3) / 6) = k := 
sorry

end fraction_sum_is_integer_l662_662568


namespace smaller_angle_at_3_15_l662_662793

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l662_662793


namespace jane_sandwich_count_l662_662648

noncomputable def total_sandwiches : ℕ := 5 * 7 * 4

noncomputable def turkey_swiss_reduction : ℕ := 5 * 1 * 1

noncomputable def salami_bread_reduction : ℕ := 5 * 1 * 4

noncomputable def correct_sandwich_count : ℕ := 115

theorem jane_sandwich_count : total_sandwiches - turkey_swiss_reduction - salami_bread_reduction = correct_sandwich_count :=
by
  sorry

end jane_sandwich_count_l662_662648


namespace complex_magnitude_problem_l662_662542

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l662_662542


namespace goods_train_length_l662_662671

noncomputable def length_of_goods_train (speed_woman_train speed_goods_train : ℝ) (time_pass : ℝ) : ℝ :=
  let relative_speed_kmph := speed_woman_train + speed_goods_train
  let relative_speed_mps := (relative_speed_kmph * 1000) / 3600
  in relative_speed_mps * time_pass

theorem goods_train_length :
  length_of_goods_train 25 142.986561075114 3 = 38.932074757434 :=
  by sorry

end goods_train_length_l662_662671


namespace math_problem_l662_662418

noncomputable def f (a b : ℝ) (x : ℝ) := a * x ^ 2 + b * x

theorem math_problem
  (a b : ℝ)
  (h₀ : a ≠ 0)
  (h₁ : f a b 2 = 0)
  (h₂ : ∀ x : ℝ, f a b x - x = 0 → (f a b 1 = 1)) :
  (f a b = λ x, - (1 / 2) * (x ^ 2) + x) ∧
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → (0 ≤ f a b x ∧ f a b x ≤ 1 / 2)) ∧
  ∀ x : ℝ, (f a b x - f a b (-x) = 2 * x) ∧ (∀ y : ℝ, f a b (-y) = -f a b y) :=
by
  sorry

end math_problem_l662_662418


namespace ellipse_C2_equation_line_AB_equation_l662_662041

noncomputable def eccentricity (a b : ℝ) : ℝ := sqrt (1 - (b^2 / a^2))

variables (O : Point) (A B : Point)
variables (x_A y_A x_B y_B : ℝ)

-- Definition of ellipse C1
def ellipse_C1 (x y : ℝ) : Prop := (x^2 / 4) + y^2 = 1

-- Definition of ellipse C2 with the same eccentricity as C1 and its major axis is the minor axis of C1
def ellipse_C2 (x y : ℝ) (e : ℝ) : Prop :=
  let a := 4 in
  let b := 2 in
  e = eccentricity a b ∧ (y^2 / 16) + (x^2 / 4) = 1

-- Definition for collinearity given OB = 2*OA
def collinear (O A B : Point) : Prop :=
  let vec OA := (x_A - O.x, y_A - O.y) in
  let vec OB := (2 * (x_A - O.x), 2 * (y_A - O.y)) in
  (x_B, y_B) = vec OB ∧ (x_A, y_A) = vec OA

-- The final Lean statements
theorem ellipse_C2_equation (x y : ℝ) (e : ℝ) (h_e : e = sqrt (3) / 2) :
  ellipse_C2 x y e → (y^2 / 16) + (x^2 / 4) = 1 := sorry

theorem line_AB_equation (k : ℝ) (h_k : k = 1 ∨ k = -1) :
  ∃ k, (B.y = k * B.x) := sorry

-- These are merely the statements. The proofs are omitted.

end ellipse_C2_equation_line_AB_equation_l662_662041


namespace sum_of_coefficients_l662_662360

noncomputable def sum_all_coefficients_eq : Prop :=
  ∀ (a : ℝ), (binomial_coeff 5 4 * a + binomial_coeff 5 2 = 15) →
    (a + 1) * (1 + 1)^5 = 64

theorem sum_of_coefficients (a : ℝ) (h : binomial_coeff 5 4 * a + binomial_coeff 5 2 = 15) : 
  sum_all_coefficients_eq :=
by
  sorry

end sum_of_coefficients_l662_662360


namespace four_digit_number_exists_l662_662494

theorem four_digit_number_exists :
  ∃ (x1 x2 y1 y2 : ℕ), (x1 > 0) ∧ (x2 > 0) ∧ (y1 > 0) ∧ (y2 > 0) ∧
                       (x2 * y2 - x1 * y1 = 67) ∧ (x2 > y2) ∧ (x1 < y1) ∧
                       (x1 * 10^3 + x2 * 10^2 + y2 * 10 + y1 = 1985) := sorry

end four_digit_number_exists_l662_662494


namespace quadratic_root_value_of_b_l662_662051

theorem quadratic_root_value_of_b :
  (∃ r1 r2 : ℝ, 2 * r1^2 + b * r1 - 20 = 0 ∧ r1 = -5 ∧ r1 * r2 = -10 ∧ r1 + r2 = -b / 2) → b = 6 :=
by
  intro h
  obtain ⟨r1, r2, h_eq1, h_r1, h_prod, h_sum⟩ := h
  sorry

end quadratic_root_value_of_b_l662_662051


namespace arithmetic_sequence_difference_l662_662448

theorem arithmetic_sequence_difference 
  (a b c : ℝ) 
  (h1: 2 + (7 / 4) = a)
  (h2: 2 + 2 * (7 / 4) = b)
  (h3: 2 + 3 * (7 / 4) = c)
  (h4: 2 + 4 * (7 / 4) = 9):
  c - a = 3.5 :=
by sorry

end arithmetic_sequence_difference_l662_662448


namespace range_of_k_l662_662756

theorem range_of_k (k : ℝ) (hE : ∀ x y : ℝ, (x^2 / 2 + y^2 / k = 1)) (hp : ¬(0 < k ∧ k < 2)) (hq : sqrt 2 < sqrt (1 - k / 2) ∧ sqrt (1 - k / 2) < sqrt 3) :
  -4 < k ∧ k < -2 :=
by {
  sorry
}

end range_of_k_l662_662756


namespace minimum_value_l662_662459

theorem minimum_value(a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1/2) :
  (2 / a + 3 / b) ≥ 5 + 2 * Real.sqrt 6 :=
sorry

end minimum_value_l662_662459


namespace find_x_values_l662_662711

noncomputable def condition (x : ℝ) : Prop :=
  1 / (x * (x + 2)) - 1 / ((x + 1) * (x + 3)) < 1 / 4

theorem find_x_values : 
  {x : ℝ | condition  x} = {x : ℝ | x < -3} ∪ {x : ℝ | -1 < x ∧ x < 0} :=
by sorry

end find_x_values_l662_662711


namespace parallel_lines_slopes_l662_662061

theorem parallel_lines_slopes (k : ℝ) :
  (∀ x y : ℝ, x + (1 + k) * y = 2 - k → k * x + 2 * y + 8 = 0 → k = 1) :=
by
  intro h1 h2
  -- We can see that there should be specifics here about how the conditions lead to k = 1
  sorry

end parallel_lines_slopes_l662_662061


namespace find_distance_to_x_axis_l662_662420

noncomputable def hyperbola_asymptote_distance 
  (l : ℝ → ℝ)
  (P : ℝ × ℝ)
  (F₁ F₂ : ℝ × ℝ)
  (hyperbola : ℝ × ℝ → Prop)
  (PF₁_dot_PF₂_zero : Prop)
  (distance_from_x_axis : ℝ) : Prop :=
  (∀ x y, hyperbola (x, y) ↔ (x^2 / 2) - (y^2 / 4) = 1) ∧
  (∀ x, l x = √2 * x) ∧
  P ∈ set_of (λ x : ℝ × ℝ, x.2 = l x.1) ∧
  F₁ = (-(√6), 0) ∧
  F₂ = (√6, 0) ∧
  PF₁_dot_PF₂_zero = (let PF₁ := (P.1 + √6, P.2) 
                          PF₂ := (P.1 - √6, P.2)
                      in PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2 = 0) ∧
  distance_from_x_axis = 2

theorem find_distance_to_x_axis :
  ∀ (l : ℝ → ℝ)
    (P : ℝ × ℝ)
    (F₁ F₂ : ℝ × ℝ)
    (hyperbola : ℝ × ℝ → Prop)
    (PF₁_dot_PF₂_zero : Prop)
    (distance_from_x_axis : ℝ),
  hyperbola_asymptote_distance l P F₁ F₂ hyperbola PF₁_dot_PF₂_zero distance_from_x_axis → 
  distance_from_x_axis = 2 :=
by
  sorry

end find_distance_to_x_axis_l662_662420


namespace population_net_increase_in_one_day_l662_662845

/-- Definition for birth rate per two seconds and death rate per two seconds --/
def birth_rate_per_two_seconds : ℕ := 7
def death_rate_per_two_seconds : ℕ := 2

/-- Calculate the net increase per second and then for one day --/
theorem population_net_increase_in_one_day :
  let birth_rate_per_second := (birth_rate_per_two_seconds : ℝ) / 2
      death_rate_per_second := (death_rate_per_two_seconds : ℝ) / 2
      net_increase_per_second := birth_rate_per_second - death_rate_per_second
      seconds_in_one_day := 86400
  in net_increase_per_second * seconds_in_one_day = 216000 :=
by
  sorry

end population_net_increase_in_one_day_l662_662845


namespace single_duplicate_numbers_eq_28_l662_662454

def is_single_duplicate_number (n : ℕ) : Prop :=
  (100 ≤ n ∧ n ≤ 200) ∧
  (((n / 100) = (n % 100 / 10) ∧ (n % 100 / 10) ≠ (n % 10)) ∨
   ((n / 100) = (n % 10) ∧ (n % 100 / 10) ≠ (n % 10)) ∨
   ((n % 100 / 10) = (n % 10) ∧ (n / 100) ≠ (n % 100 / 10)))

def count_single_duplicate_numbers : ℕ :=
  (finset.filter is_single_duplicate_number (finset.range 201)).card

theorem single_duplicate_numbers_eq_28 :
  count_single_duplicate_numbers = 28 :=
by
  sorry

end single_duplicate_numbers_eq_28_l662_662454


namespace sum_possible_values_sum_of_values_math_problem_l662_662165

theorem sum_possible_values (x : ℝ) (h : 3 * (x - 4)^2 = (2 * x - 6) * (x + 2)) : x = 10 ∨ x = 6 :=
  sorry

theorem sum_of_values (x : ℝ) (h : 3 * (x - 4)^2 = (2 * x - 6) * (x + 2)) : x = 10 ∨ x = 6 → (x = 10 ∨ x = 6) → 10 + 6 = 16 :=
  sorry

theorem math_problem (x : ℝ) (h : 3 * (x - 4)^2 = (2 * x - 6) * (x + 2)) : 10 + 6 = 16 :=
  sum_of_values x h (sum_possible_values x h)

end sum_possible_values_sum_of_values_math_problem_l662_662165


namespace parity_of_f_l662_662825

def f (x : ℝ) : ℝ := x^3 - x

theorem parity_of_f : ∀ x, f (-x) = -f x := by
  intro x
  calc
    f (-x) = (-x)^3 - (-x) : by rfl
    ...   = -x^3 + x       : by rfl
    ...   = - (x^3 - x)    : by rfl
    ...   = - f x          : by rfl

end parity_of_f_l662_662825


namespace chemistry_textbook_weight_l662_662511

theorem chemistry_textbook_weight (G C : ℝ) 
  (h1 : G = 0.625) 
  (h2 : C = G + 6.5) : 
  C = 7.125 := 
by 
  sorry

end chemistry_textbook_weight_l662_662511


namespace geometry_problem_l662_662175

noncomputable def problem_statement (O A B C P Q : Point) : Prop :=
  ∃ (r : ℝ), (r = 7 ∧ 
  circle_centered_at O r A ∧ circle_centered_at O r B ∧ circle_centered_at O r C ∧
  perpendicular_bisector_of_line_meets AB P AND BC ∧ extension_meets AC Q ∧
  OP * OQ = 49)

theorem geometry_problem (O A B C P Q : Point) : problem_statement O A B C P Q := sorry

end geometry_problem_l662_662175


namespace hyperbola_equation_l662_662076

theorem hyperbola_equation 
  (a b c : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0)
  (eccentricity_sqrt_3 : c / a = Real.sqrt 3) 
  (parabola_focus : (3,0)) 
  (focus_hyperbola : (3,0))
  (b_squared : b^2 = c^2 - a^2) : 
  (∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1) → (x^2 / 3 - y^2 / 6 = 1)) :=
begin
  sorry
end

end hyperbola_equation_l662_662076


namespace find_circle_equation_l662_662053

noncomputable def symmetric_point (p : ℝ × ℝ) (l : ℝ × ℝ → Prop) : ℝ × ℝ :=
⟨2 * (1 - p.2 - 1) / 2 - 1, -p.1⟩

noncomputable def distance_to_line (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
(abs (a * p.1 + b * p.2 + c)) / sqrt (a^2 + b^2)

theorem find_circle_equation : 
  ∃ (a b r : ℝ), (∀ c : ℝ, c = 2) ∧ ∀ p : ℝ × ℝ, 
  let M := (1, -1) in let l := (fun z => z.1 - z.2 + 1 = 0) in
  let C := (a, b) in symmetric_point M l = C ∧
  distance_to_line (C.1, C.2) 1 -1 1 = r ∧
  (x + a)^2 + (y + b)^2 = r^2 := 
by 
  let a := -2
  let b := 2
  let r := sqrt((3/2:ℝ)^2)
  -- define terms
  use a, b, r
  -- define proof conditions
  sorry

end find_circle_equation_l662_662053


namespace find_x_value_l662_662717

-- Define the condition
def equation (x : ℝ) : Prop := log 5 (x - 3) + log (√5) (x^3 - 3) + log (1 / 5) (x - 3) = 4

theorem find_x_value (x : ℝ) (h : equation x) (hx_pos : 0 < x) : x = real.cbrt 28 :=
sorry

end find_x_value_l662_662717


namespace tan_sum_identity_l662_662064

theorem tan_sum_identity (α : ℝ) (h : tan α = 2) : tan (α + π / 4) = -3 :=
by sorry

end tan_sum_identity_l662_662064


namespace redistribute_cheese_l662_662391

variables (M T_Th T_F T_Wh T_G : ℝ)

-- Conditions
axiom Thin_fat : T_Th = T_F - 20
axiom White_gray : T_Wh = T_G - 8
axiom White_quarter : T_Wh = M / 4
axiom Gray_cut : T_G' = T_G - 8 -- T_G' is new weight of Gray's slice after cutting 8
axiom Fat_cut : T_F' = T_F - 20 -- T_F' is new weight of Fat's slice after cutting 20

-- Goal: 28 grams should be divided such that all slices are equal
lemma equal_weights (H : T_Wh = T_G') (H1 : T_F' + 14 = T_Th + 14 ) : 
  T_F + 14 = T_Th + 14 := 
by
  rw [←Thin_fat, ←Fat_cut]
  sorry

-- Overall proof objective
theorem redistribute_cheese : (T_F' + 14 = T_Th + 14) → (T_Wh = T_G') → 
  (T_Wh = T_Th + 14) ∧ (T_Wh = T_F + 14) :=
by 
  intro H H1
  split
  all_goals 
    rw [←equal_weights] at *
    assumption
  sorry

end redistribute_cheese_l662_662391


namespace find_m_pure_imaginary_l662_662155

theorem find_m_pure_imaginary (m : ℝ) (h : m^2 + m - 2 + (m^2 - 1) * I = (0 : ℝ) + (m^2 - 1) * I) :
  m = -2 :=
by {
  sorry
}

end find_m_pure_imaginary_l662_662155


namespace transform_f_to_g_l662_662070

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 2 * Real.sin x * Real.cos x
noncomputable def g (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem transform_f_to_g :
  ∀ x : ℝ, g x = f (x + (π / 8)) :=
by
  sorry

end transform_f_to_g_l662_662070


namespace cyclic_quadrilateral_de_eq_ac_bd_over_ab_bc_l662_662042

open EuclideanGeometry

theorem cyclic_quadrilateral_de_eq_ac_bd_over_ab_bc
  (A B C D E : Point)
  (h_cyclic : cyclic_quad A B C D)
  (hE_on_DA : E ∈ segment D A)
  (h_angle_eq : 2 * ∠ E B D = ∠ A B C) :
  DE = (AC * BD) / (AB + BC) :=
sorry

end cyclic_quadrilateral_de_eq_ac_bd_over_ab_bc_l662_662042


namespace find_y_value_l662_662023

theorem find_y_value (y : ℕ) (h1 : y ≤ 150)
  (h2 : (45 + 76 + 123 + y + y + y) / 6 = 2 * y) :
  y = 27 :=
sorry

end find_y_value_l662_662023


namespace difference_of_squares_l662_662370

theorem difference_of_squares (n : ℤ) : 4 - n^2 = (2 + n) * (2 - n) := 
by
  -- Proof goes here
  sorry

end difference_of_squares_l662_662370


namespace sum_of_seven_consecutive_integers_l662_662575

theorem sum_of_seven_consecutive_integers (n : ℤ) :
  n + (n + 1) + (n + 2) + (n + 3) + (n + 4) + (n + 5) + (n + 6) = 7 * n + 21 :=
by
  sorry

end sum_of_seven_consecutive_integers_l662_662575


namespace find_B_age_l662_662283

variable (a b c : ℕ)

def problem_conditions : Prop :=
  a = b + 2 ∧ b = 2 * c ∧ a + b + c = 22

theorem find_B_age (h : problem_conditions a b c) : b = 8 :=
by {
  sorry
}

end find_B_age_l662_662283


namespace range_of_m_l662_662407

open Real

-- Defining conditions as propositions
def p (m : ℝ) : Prop := ∀ x : ℝ, x^2 - m*x + 1 ≠ 0
def q (m : ℝ) : Prop := m > 1
def p_or_q (m : ℝ) : Prop := p m ∨ q m
def p_and_q (m : ℝ) : Prop := p m ∧ q m

-- Mathematically equivalent proof problem
theorem range_of_m (m : ℝ) (H1 : p_or_q m) (H2 : ¬p_and_q m) : -2 < m ∧ m ≤ 1 ∨ 2 ≤ m :=
by
  sorry

end range_of_m_l662_662407


namespace arithmetic_expression_l662_662347

theorem arithmetic_expression : 8 / 4 + 5 * 2 ^ 2 - (3 + 7) = 12 := by
  sorry

end arithmetic_expression_l662_662347


namespace triangle_is_isosceles_or_right_l662_662108

-- Definitions for the angles in a triangle
def is_triangle (A B C : ℝ) : Prop := A + B + C = π

-- Definitions for isosceles and right triangles
def is_isosceles_triangle (A B C : ℝ) : Prop :=
  (A = B) ∨ (B = C) ∨ (A = C)

def is_right_triangle (A B C : ℝ) : Prop :=
  (A = π / 2) ∨ (B = π / 2) ∨ (C = π / 2)

-- Theorem statement
theorem triangle_is_isosceles_or_right (A B C : ℝ) :
  is_triangle A B C →
  sin (A + B - C) = sin (A - B + C) →
  is_isosceles_triangle A B C ∨ is_right_triangle A B C :=
begin
  sorry
end

end triangle_is_isosceles_or_right_l662_662108


namespace graph_shift_symmetry_l662_662944

noncomputable def f (x : ℝ) : ℝ :=
  real.cos (2 * x + (2 * real.pi / 3))

theorem graph_shift_symmetry :
  ∀ x : ℝ, f (x) = f (-x - real.pi / 3) := by
  sorry

end graph_shift_symmetry_l662_662944


namespace magnitude_m_l662_662442

open Real

variables (a b m : ℝ × ℝ)
variable (theta : ℝ)
variable (x y : ℝ)

-- Definitions of the conditions
def angle_condition : Prop := theta = 2 * Real.pi / 3
def magnitude_a : Prop := norm (a) = 1
def magnitude_b : Prop := norm (b) = 2
def dot_product_ma : Prop := a.1 * m.1 + a.2 * m.2 = 1
def dot_product_mb : Prop := b.1 * m.1 + b.2 * m.2 = 1

-- Given the conditions, prove that the magnitude of m is sqrt(21) / 3
theorem magnitude_m (h1 : angle_condition) (h2 : magnitude_a) (h3 : magnitude_b) (h4 : dot_product_ma) (h5 : dot_product_mb) : 
  norm (m) = sqrt(21) / 3 :=
by
  sorry

end magnitude_m_l662_662442


namespace smallest_x_mul_900_multiple_of_1152_l662_662261

theorem smallest_x_mul_900_multiple_of_1152 : 
  ∃ x : ℕ, (x > 0) ∧ (900 * x) % 1152 = 0 ∧ ∀ y : ℕ, (y > 0) ∧ (900 * y) % 1152 = 0 → y ≥ x := 
begin
  use 32,
  split,
  { exact nat.one_pos, }, -- the positive condition
  split,
  { change (900 * 32) % 1152 = 0, -- 32 satisfies the multiple condition
    norm_num, 
  },
  { intros y hy, -- minimality condition
    cases hy with hy_pos hy_dvd,
    have : 1152 ∣ 900 * 32 := by {
        change 900 * 32 % 1152 = 0,
        norm_num,
    },
    obtain ⟨k, hk⟩ := exists_eq_mul_left_of_dvd this,
    change 1152 * k < 1152 * 32,
    refine le_of_dvd (mul_pos (@nat.one_pos _) hy_pos) _,
    exact ⟨32, rfl⟩,
  },
end

end smallest_x_mul_900_multiple_of_1152_l662_662261


namespace probability_at_least_3_speak_l662_662831

noncomputable def probability_speaks (n : ℕ) (p : ℚ) : ℚ :=
  ∑ k in finset.range (n + 1), if k < 3 then (nat.choose n k) * (p^k) * ((1 - p)^(n-k)) else 0

theorem probability_at_least_3_speak :
  let p := (1 : ℚ) / 3,
      n := 6 in
  1 - probability_speaks n p = 233 / 729 :=
by {
  let p := 1/3,
  let n := 6,
  have h₀ : ∑ k in finset.range (n + 1), if k < 3 then (nat.choose n k) * (p^k) * ((1 - p)^(n-k)) else 0 = 496 / 729 := sorry,
  rw [probability_speaks, h₀],
  norm_num,
}

end probability_at_least_3_speak_l662_662831


namespace determine_chris_age_l662_662586

theorem determine_chris_age (a b c : ℚ)
  (h1 : (a + b + c) / 3 = 10)
  (h2 : c - 5 = 2 * a)
  (h3 : b + 4 = (3 / 4) * (a + 4)) :
  c = 283 / 15 :=
by
  sorry

end determine_chris_age_l662_662586


namespace a_30_eq_sqrt3_l662_662438

noncomputable def a : ℕ → ℝ
| 0       := 0
| (n + 1) := (a n - real.sqrt 3) / (real.sqrt 3 * a n + 1)

theorem a_30_eq_sqrt3 : a 30 = real.sqrt 3 :=
sorry

end a_30_eq_sqrt3_l662_662438


namespace conjugate_of_complex_fraction_l662_662589

theorem conjugate_of_complex_fraction : 
  ∀ (i : ℂ), i ^ 2 = -1 → complex.conj (2 / ((1 - i) * i)) = 1 + i :=
by
  intro i h
  -- Skipping the actual steps of the proof
  apply sorry

end conjugate_of_complex_fraction_l662_662589


namespace angle_AA1_BD1_angle_BD1_DC1_angle_AD1_DC1_l662_662739

structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

structure Line where
  p1 : Point
  p2 : Point

def length (l : Line) : ℝ :=
  Real.sqrt ((l.p2.x - l.p1.x)^2 + (l.p2.y - l.p1.y)^2 + (l.p2.z - l.p1.z)^2)

def dot_product (l1 l2 : Line) : ℝ :=
  (l1.p2.x - l1.p1.x) * (l2.p2.x - l2.p1.x) +
  (l1.p2.y - l1.p1.y) * (l2.p2.y - l2.p1.y) +
  (l1.p2.z - l1.p1.z) * (l2.p2.z - l2.p1.z)

def angle_between_lines (l1 l2 : Line) : ℝ :=
  Real.arccos ((dot_product l1 l2) / (length l1 * length l2))

def cube_edge_length (l : ℝ) : Prop :=
  l > 0

variable (a : ℝ)

def pointA : Point := { x := 0, y := 0, z := 0 }
def pointB : Point := { x := a, y := 0, z := 0 }
def pointC : Point := { x := a, y := a, z := 0 }
def pointD : Point := { x := 0, y := a, z := 0 }
def pointA1 : Point := { x := 0, y := 0, z := a }
def pointB1 : Point := { x := a, y := 0, z := a }
def pointC1 : Point := { x := a, y := a, z := a }
def pointD1 : Point := { x := 0, y := a, z := a }

def lineAA1 : Line := { p1 := pointA, p2 := pointA1 }
def lineBD1 : Line := { p1 := pointB, p2 := pointD1 }
def lineDC1 : Line := { p1 := pointD, p2 := pointC1 }
def lineAD1 : Line := { p1 := pointA, p2 := pointD1 }

theorem angle_AA1_BD1 :
  angle_between_lines lineAA1 lineBD1 = Real.arccos (1 / Real.sqrt 3) :=
sorry

theorem angle_BD1_DC1 :
  angle_between_lines lineBD1 lineDC1 = Real.pi / 2 :=
sorry

theorem angle_AD1_DC1 :
  angle_between_lines lineAD1 lineDC1 = Real.pi / 3 :=
sorry

end angle_AA1_BD1_angle_BD1_DC1_angle_AD1_DC1_l662_662739


namespace diagonal_of_square_with_perimeter_40_ft_l662_662015

-- Define the perimeter of the square
def perimeter : Float := 40

-- Define the side length of the square
def side_length (P : Float) : Float := P / 4

-- Define the length of the diagonal using the Pythagorean theorem
def diagonal_length (s : Float) : Float := Real.sqrt (s ^ 2 + s ^ 2)

-- The specific proof statement we want to prove
theorem diagonal_of_square_with_perimeter_40_ft :
  diagonal_length (side_length perimeter) = 10 * Real.sqrt 2 :=
by
  sorry

end diagonal_of_square_with_perimeter_40_ft_l662_662015


namespace stairway_standing_arrangements_l662_662029

theorem stairway_standing_arrangements :
  let A B C D : ℕ := 4 in
  let steps : ℕ := 7 in
  let max_people_per_step : ℕ := 3 in
  let total_arrangements : ℕ := 2394 in
  (A = 4 ∧ steps = 7 ∧ max_people_per_step = 3) → total_arrangements = 2394 := 
by
  sorry

end stairway_standing_arrangements_l662_662029


namespace largest_number_4597_l662_662639

def swap_adjacent_digits (n : ℕ) : ℕ :=
  sorry

def max_number_after_two_swaps_subtract_100 (n : ℕ) : ℕ :=
  -- logic to perform up to two adjacent digit swaps and subtract 100
  sorry

theorem largest_number_4597 : max_number_after_two_swaps_subtract_100 4597 = 4659 :=
  sorry

end largest_number_4597_l662_662639


namespace find_number_l662_662295

theorem find_number (x : ℝ) (h : 3034 - (x / 20.04) = 2984) : x = 1002 :=
by
  sorry

end find_number_l662_662295


namespace K6_edge_coloring_l662_662001

-- Define the complete graph K_6
def K6 : SimpleGraph (Fin 6) := CompleteGraph (Fin 6)

-- State that there exists a 5-edge-coloring for K_6 such that
-- each vertex has edges of all different colors
theorem K6_edge_coloring :
  ∃ (c : K6.Edge → Fin 5), 
    ∀ (v : Fin 6), 
      ∀ (e1 e2 : K6.Edge), e1 ∈ K6.edgeSet v → e2 ∈ K6.edgeSet v → 
        e1 ≠ e2 → (c e1 ≠ c e2) :=
sorry

end K6_edge_coloring_l662_662001


namespace european_fraction_is_one_fourth_l662_662452

-- Define the total number of passengers
def P : ℕ := 108

-- Define the fractions and the number of passengers from each continent
def northAmerica := (1 / 12) * P
def africa := (1 / 9) * P
def asia := (1 / 6) * P
def otherContinents := 42

-- Define the total number of non-European passengers
def totalNonEuropean := northAmerica + africa + asia + otherContinents

-- Define the number of European passengers
def european := P - totalNonEuropean

-- Define the fraction of European passengers
def europeanFraction := european / P

-- Prove that the fraction of European passengers is 1/4
theorem european_fraction_is_one_fourth : europeanFraction = 1 / 4 := 
by
  unfold europeanFraction european totalNonEuropean northAmerica africa asia P
  sorry

end european_fraction_is_one_fourth_l662_662452


namespace fraction_denominator_l662_662649

theorem fraction_denominator (d : ℕ) (h1 : 325 / d = 0.125)
  (h2 : ∀ n : ℕ, 0 < n → 81 % 3 = 0 → (show digit of (325 / d) at 81 = 5)) : d = 8 :=
by sorry

end fraction_denominator_l662_662649


namespace distance_to_work_eq_10_l662_662875

-- Define the conditions
def T : ℝ := 12  -- Tank capacity
def frac_left : ℝ := 2 / 3  -- Fraction of the tank left
def E : ℝ := 5  -- Car efficiency in miles per gallon
def frac_used : ℝ := 1 - frac_left  -- Fraction of the tank used

-- Translating the problem: prove that distance_to_work = 10
theorem distance_to_work_eq_10 : 
  (T * frac_used * E) / 2 = 10 := 
by
  sorry  -- Proof is omitted, as per the instructions.

end distance_to_work_eq_10_l662_662875


namespace sector_area_l662_662827

theorem sector_area (r : ℝ) (α : ℝ) (h1 : 2 * r + α * r = 16) (h2 : α = 2) :
  1 / 2 * α * r^2 = 16 :=
by
  sorry

end sector_area_l662_662827


namespace div_add_fraction_l662_662625

theorem div_add_fraction : (3 / 7) / 4 + 2 = 59 / 28 :=
by
  sorry

end div_add_fraction_l662_662625


namespace GPA_at_least_3_25_l662_662185

def Probability (A B C : ℝ) : ℝ := (A + B + C) / 3

theorem GPA_at_least_3_25 :
  let englishA := (1 / 3 : ℝ)
  let englishB := (1 / 3 : ℝ)
  let englishC := (1 / 3 : ℝ)
  let historyB := (1 / 2 : ℝ)
  let historyC := (1 / 2 : ℝ)
  let math_points := 4
  let science_points := 4
  let total_needed_points := 13
  let total_subjects := 4
  let gpa_of_3_25 := 13 / 4
  let math_and_science_points := math_points + science_points
  let remaining_points_needed := total_needed_points - math_and_science_points
  let english_history_combinations := englishA * 0 + englishB * historyB + englishB * historyB
  ∃ P, P = (englishB * historyB + englishB * historyB) → P = 1 / 3 :=
  sorry

end GPA_at_least_3_25_l662_662185


namespace frank_earnings_weed_eating_l662_662727

variable W : ℝ

theorem frank_earnings_weed_eating (h : 5 + W = 63) : W = 58 :=
by
  sorry

end frank_earnings_weed_eating_l662_662727


namespace round_4_36_to_nearest_tenth_l662_662186

-- Definitions based on the conditions
def number_to_round : ℝ := 4.36
def round_half_up (x : ℝ) : ℝ := (↑(Int(10 * x) + if 10 * x - Int(10 * x) >= 0.5 then 1 else 0)) / 10

-- The assertion to prove
theorem round_4_36_to_nearest_tenth : round_half_up number_to_round = 4.4 :=
by
  -- Proof will go here
  sorry

end round_4_36_to_nearest_tenth_l662_662186


namespace perpendicular_line_plane_l662_662734

variable (l m : Type) [Line l] [Line m] [Plane a]
variable (Perpendicular : l ⊥ m -> Prop)
variable (Subset : m ⊆ a -> Prop)
variable (PerpendicularPlane : l ⊥ a -> Prop)

theorem perpendicular_line_plane (h1 : PerpendicularPlane l a) (h2 : Subset m a) : Perpendicular l m :=
sorry

end perpendicular_line_plane_l662_662734


namespace common_tangent_line_minimum_m_l662_662059

noncomputable def f (x : ℝ) := Real.exp x

-- Part 1
theorem common_tangent_line 
  (x0 : ℝ)
  (hroot : Real.exp x0 = (x0 + 1) / (x0 - 1)) 
  : ∃ l : ℝ → ℝ, 
    (∀ x, l x = Real.exp x0 * (x - x0) + Real.exp x0) ∧ 
    (∀ x, l x = Real.exp x0 * x - x0 - 1) := sorry

-- Part 2
noncomputable def fi (i n : ℕ) (x : ℝ) := Real.exp ((i : ℝ) / (n : ℝ))^x

theorem minimum_m 
  (n : ℕ)
  (x : ℝ)
  (hn : 2 ≤ n) 
  (hx : -1 ≤ x ∧ x ≤ 1) 
  : (∏ i in Finset.range (n-1), fi i n x) ≥ Real.exp (-1 / 2) := sorry

end common_tangent_line_minimum_m_l662_662059


namespace no_four_digit_number_exists_l662_662700

theorem no_four_digit_number_exists :
  ¬ ∃ (a b : ℕ), a ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                 b ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} ∧ 
                 (10 + a + b = a * b) := 
by 
  sorry

end no_four_digit_number_exists_l662_662700


namespace Libby_initial_quarters_l662_662555

theorem Libby_initial_quarters (payed_dollars : ℝ) (quarters_left : ℕ) (quarter_value : ℝ)
  (h1 : payed_dollars = 35)
  (h2 : quarters_left = 20)
  (h3 : quarter_value = 0.25)
  : ∃ q : ℕ, q = (payed_dollars / quarter_value).to_nat + quarters_left := 
sorry

end Libby_initial_quarters_l662_662555


namespace uniqueFlavors_l662_662410

-- Definitions for the conditions
def numRedCandies : ℕ := 6
def numGreenCandies : ℕ := 4
def numBlueCandies : ℕ := 5

-- Condition stating each flavor must use at least two candies and no more than two colors
def validCombination (x y z : ℕ) : Prop :=
  (x = 0 ∨ y = 0 ∨ z = 0) ∧ (x + y ≥ 2 ∨ x + z ≥ 2 ∨ y + z ≥ 2)

-- The main theorem statement
theorem uniqueFlavors : 
  ∃ n : ℕ, n = 30 ∧ 
  (∀ x y z : ℕ, validCombination x y z → (x ≤ numRedCandies) ∧ (y ≤ numGreenCandies) ∧ (z ≤ numBlueCandies)) :=
sorry

end uniqueFlavors_l662_662410


namespace solve_congruences_l662_662373

theorem solve_congruences :
  ∃ (x : ℕ), x ≡ 15 [MOD 17] ∧ x ≡ 11 [MOD 13] ∧ x ≡ 3 [MOD 10] :=
  ∃ x, x = 1103 ∧ (x % 17 = 15) ∧ (x % 13 = 11) ∧ (x % 10 = 3) := 
begin
  use 1103,
  split, refl,
  split,
  exact rfl,
  split,
  exact rfl,
  exact rfl,
sorry

end solve_congruences_l662_662373


namespace olesya_game_winning_conditions_l662_662250

-- Definition for the game conditions
def game_conditions (n : ℕ) : Prop :=
  n > 1 ∧ (∀ k, 1 ≤ k ∧ k ≤ 2 * n → ∃ w : ℕ, distinct_pieces k w)

-- Function to check if Olesya wins given n
def olesya_wins (n : ℕ) : Prop :=
  (n ≠ 2 ∧ n ≠ 4)

-- Theorem statement for the problem
theorem olesya_game_winning_conditions (n : ℕ) (h : game_conditions n) : olesya_wins n :=
  by
    -- Insert proof here
    sorry

end olesya_game_winning_conditions_l662_662250


namespace inequality_proof_l662_662881

open Real

theorem inequality_proof
  (a b c x y z : ℝ) 
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) 
  (hx_cond : 1 / x + 1 / y + 1 / z = 1) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a ^ x + b ^ y + c ^ z ≥ (4 * a * b * c * x * y * z) / (x + y + z - 3) ^ 2 :=
by
  sorry

end inequality_proof_l662_662881


namespace smaller_angle_at_3_15_l662_662792

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l662_662792


namespace complex_magnitude_theorem_l662_662534

theorem complex_magnitude_theorem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  ∣(1 / z) + (1 / w)∣ = 3 / 8 := by
  sorry

end complex_magnitude_theorem_l662_662534


namespace susan_age_in_5_years_l662_662645

variable (J N S X : ℕ)

-- Conditions
axiom h1 : J - 8 = 2 * (N - 8)
axiom h2 : J + X = 37
axiom h3 : S = N - 3

-- Theorem statement
theorem susan_age_in_5_years : S + 5 = N + 2 :=
by sorry

end susan_age_in_5_years_l662_662645


namespace find_value_of_expression_l662_662531

-- Define the constants a, b, and c
variables (a b c : ℝ) 

-- Define the function representing the inequality condition
def inequality (x : ℝ) : ℝ := (x - a) * (x - b) / (x - c)

-- Conditions on the intervals for the inequality
def condition1 : Prop := ∀ x : ℝ, x < -6 → inequality a b c x ≥ 0
def condition2 : Prop := ∀ x : ℝ, 20 ≤ x ∧ x ≤ 23 → inequality a b c x ≥ 0
def condition3 : Prop := a < b

-- The main theorem to be proven
theorem find_value_of_expression (h1 : condition1 a b c) (h2 : condition2 a b c) (h3 : condition3 a b) : 
  a + 2 * b + 3 * c = 48 :=
sorry

end find_value_of_expression_l662_662531


namespace solve_for_x_l662_662194

theorem solve_for_x (x : ℝ) : 3^(2 * x) = sqrt 243 → x = 5 / 4 := by
  sorry

end solve_for_x_l662_662194


namespace museum_earnings_from_nyc_college_students_l662_662852

def visitors := 200
def nyc_residents_fraction := 1 / 2
def college_students_fraction := 0.30
def ticket_price := 4

theorem museum_earnings_from_nyc_college_students : 
  ((visitors * nyc_residents_fraction * college_students_fraction) * ticket_price) = 120 := 
by 
  sorry

end museum_earnings_from_nyc_college_students_l662_662852


namespace sum_of_coeff_without_y_l662_662994

theorem sum_of_coeff_without_y (x y : ℝ) : 
  (finset.sum (finset.range 4) (λ k, (nat.choose 3 k) * 3^(3 - k))) = 64 := 
by 
  sorry

end sum_of_coeff_without_y_l662_662994


namespace distinguishable_large_triangles_l662_662247

noncomputable def number_of_distinguishable_large_triangles : Nat :=
  let same_color := 8
  let two_same_one_diff := (Nat.choose 8 2) * 2
  let all_diff := Nat.choose 8 3
  let total_sets := same_color + two_same_one_diff + all_diff
  let center_choices := 8
  center_choices * total_sets

theorem distinguishable_large_triangles : number_of_distinguishable_large_triangles = 960 :=
by
  -- Calculations for each part
  have h_same := 8
  have h_two_same_one_diff := (Nat.choose 8 2) * 2
  have h_all_diff := Nat.choose 8 3
  have h_total_sets := h_same + h_two_same_one_diff + h_all_diff
  have h_center_choices := 8
  have h_total_configs := h_center_choices * h_total_sets
  -- Final total
  calc
    number_of_distinguishable_large_triangles
    = h_total_configs : by rfl
    = 960 : by norm_num

end distinguishable_large_triangles_l662_662247


namespace mary_total_income_l662_662560

noncomputable def pay_rates := (regular_rate : ℝ) × (overtime_rate1 : ℝ) × (overtime_rate2 : ℝ)
noncomputable def earnings := (hours_worked : ℕ) × pay_rates × (bonus : ℝ) × (dues : ℝ)

noncomputable def calculate_total_income (e : earnings) : ℝ :=
  let (hours_worked, (regular_rate, overtime_rate1, overtime_rate2), bonus, dues) := e in
  let regular_pay := regular_rate * 20 in
  let overtime_pay1 := (regular_rate * 1.25) * 20 in
  let overtime_pay2 := (regular_rate * 1.5) * (hours_worked - 40) in
  let total_earnings := regular_pay + overtime_pay1 + overtime_pay2 + bonus in
  total_earnings - dues

theorem mary_total_income :
  calculate_total_income (60, (8, 10, 12), 100, 50) = 650 :=
by
  sorry

end mary_total_income_l662_662560


namespace larger_number_is_25_l662_662285

-- Let x and y be real numbers, with x being the larger number
variables (x y : ℝ)

-- The sum of the two numbers is 45
axiom sum_eq_45 : x + y = 45

-- The difference of the two numbers is 5
axiom diff_eq_5 : x - y = 5

-- We need to prove that the larger number x is 25
theorem larger_number_is_25 : x = 25 :=
by
  sorry

end larger_number_is_25_l662_662285


namespace div_pow_eq_l662_662689

theorem div_pow_eq : 23^11 / 23^5 = 148035889 := by
  sorry

end div_pow_eq_l662_662689


namespace b5_plus_b9_l662_662750

variable {a : ℕ → ℕ} -- Geometric sequence
variable {b : ℕ → ℕ} -- Arithmetic sequence

axiom geom_progression {r x y : ℕ} : a x = a 1 * r^(x - 1) ∧ a y = a 1 * r^(y - 1)
axiom arith_progression {d x y : ℕ} : b x = b 1 + d * (x - 1) ∧ b y = b 1 + d * (y - 1)

axiom a3a11_equals_4a7 : a 3 * a 11 = 4 * a 7
axiom a7_equals_b7 : a 7 = b 7

theorem b5_plus_b9 : b 5 + b 9 = 8 := by
  apply sorry

end b5_plus_b9_l662_662750


namespace angle_XOY_half_AOB_l662_662215

variables {P A B O X Y Z : Type}
variables [Circle O] -- A representation of circle with center O
variables [is_tangent OA PA] [is_tangent OB PB] -- Tangents PA, PB at A, B

-- PA and PB are tangent to the circle at A and B respectively and intersect at P.
-- A third tangent XY intersects PA at X and PB at Y.
-- We need to prove that the angle XOY is always half of the angle AOB regardless of the choice of XY.

theorem angle_XOY_half_AOB
  (h1 : tangent A (circle O))
  (h2 : tangent B (circle O))
  (h3 : tangent Z (circle O))
  (H : Z = PA ∩ PB) : 
  angle X O Y = (1 / 2) * angle A O B :=
begin
  sorry
end

end angle_XOY_half_AOB_l662_662215


namespace y_value_l662_662760

theorem y_value (x : ℕ) (h : x = 60) : 
  let y :=
    if x ≤ 50 then
      0.5 * x
    else
      25 + 0.6 * (x - 50)
  in y = 31 :=
by
  sorry

end y_value_l662_662760


namespace symmetry_center_coordinates_l662_662430

variable (ω : ℝ) (φ : ℝ)

def f (x : ℝ) := Real.sin (ω * x + φ)

theorem symmetry_center_coordinates :
  (ω > 0) →
  |φ| < Real.pi / 2 →
  (∀ x, f x = f (x + 4 * Real.pi)) →
  f (Real.pi / 3) = 1 →
  ∃ k : ℤ, (2 * k * Real.pi - 2 * Real.pi / 3, 0) = (-2 * Real.pi / 3, 0) := 
by
  intros hω hφ h_period h_val
  use 0  -- This corresponds to k = 0
  simp
  sorry

end symmetry_center_coordinates_l662_662430


namespace no_five_coprime_two_digit_composites_l662_662000

/-- 
  Prove that there do not exist five two-digit composite 
  numbers such that each pair of them is coprime, under 
  the conditions that each composite number must be made 
  up of the primes 2, 3, 5, and 7.
-/
theorem no_five_coprime_two_digit_composites :
  ¬∃ (a b c d e : ℕ),
    10 ≤ a ∧ a < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ a → p ∣ a) ∧
    10 ≤ b ∧ b < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ b → p ∣ b) ∧
    10 ≤ c ∧ c < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ c → p ∣ c) ∧
    10 ≤ d ∧ d < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ d → p ∣ d) ∧
    10 ≤ e ∧ e < 100 ∧ (∀ p ∈ [2, 3, 5, 7], p ∣ e → p ∣ e) ∧
    ∀ (x y : ℕ), (x ∈ [a, b, c, d, e] ∧ y ∈ [a, b, c, d, e] ∧ x ≠ y) → Nat.gcd x y = 1 :=
by
  sorry

end no_five_coprime_two_digit_composites_l662_662000


namespace area_of_triangle_AOB_l662_662057

noncomputable def triangle_area (z1 z2 : ℂ) : ℝ :=
  0.5 * (complex.abs z1) * (complex.abs z2) * (complex.argument z1 - complex.argument z2).sin

theorem area_of_triangle_AOB 
  (z1 z2 : ℂ) 
  (h1 : complex.abs z2 = 4) 
  (h2 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0):
  triangle_area z1 z2 = 2 * real.sqrt 3 :=
sorry

end area_of_triangle_AOB_l662_662057


namespace order_of_means_l662_662724

noncomputable def A (a b : ℝ) : ℝ := (2 * a + 2 * b) / 3
noncomputable def G (a b : ℝ) : ℝ := Real.sqrt(a * b) - 1
noncomputable def H (a b : ℝ) : ℝ := (6 * a * b) / (a + b + 1)

theorem order_of_means (a b : ℝ)
  (h₁ : a > 0) (h₂ : b > 0) (h₃ : a ≠ b) : A a b > G a b ∧ G a b > H a b :=
by
  sorry

end order_of_means_l662_662724


namespace complex_magnitude_problem_l662_662544

open Complex

theorem complex_magnitude_problem (z w : ℂ) (hz : Complex.abs z = 2) (hw : Complex.abs w = 4) (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := 
by
  sorry

end complex_magnitude_problem_l662_662544


namespace area_inequality_l662_662546
-- Import the necessary library

-- Define the required conditions in Lean
variables (A B C D O E F G H : Point) 
variables [convex_quadrilateral A B C D] (O_inside : inside_quadrilateral O A B C D)
variables (parallels : parallel_through_point O (line BC) (line AB) (line DA) (line CD) E F G H)

-- The statement to be proven
theorem area_inequality : 
  (sqrt (non_directed_area A H O E) + sqrt (non_directed_area C F O G)) <= sqrt (non_directed_area A B C D) :=
begin
  sorry -- Proof goes here
end

end area_inequality_l662_662546


namespace find_original_cost_l662_662205

def original_cost (cost_reduction : ℝ) (decreased_cost : ℝ) : ℝ := 
  decreased_cost / cost_reduction

theorem find_original_cost :
  original_cost 0.50 100 = 200 :=
by 
  -- This is where the proof would go
  sorry

end find_original_cost_l662_662205


namespace cross_product_magnitude_l662_662590

section
variables (a b : EuclideanSpace ℝ 3) -- Assuming a and b are vectors in 3-dimensional Euclidean space
variables (a_mag : ℝ) (b_mag : ℝ) (a_dot_b : ℝ)

-- We define the given conditions
def a_magnitude := |a| = 2
def b_magnitude := |b| = 5
def a_dot_b_condition := a ⬝ b = -6

-- We state the theorem to prove the magnitude of the cross product
theorem cross_product_magnitude : a_magnitude → b_magnitude → a_dot_b_condition → |a × b| = 8 := by
  intros h1 h2 h3
  sorry
end

end cross_product_magnitude_l662_662590


namespace product_xy_approx_l662_662819

-- Define the constants and equations from the problem statement
def a : Real := (1 / 3) * 500
def x : Real := a / 0.50
def sqrt_y : Real := a / 0.40
def y : Real := sqrt_y^2

-- The theorem statement proving the desired result
theorem product_xy_approx : x * y ≈ 57916685.6 := by
  have a_val : a = 166.67 := by sorry
  have x_val : x = 333.34 := by sorry
  have y_val : y = 173750.056 := by sorry
  calc
    x * y = 333.34 * 173750.056 : by sorry
    _ ≈ 57916685.6 : by sorry

end product_xy_approx_l662_662819


namespace average_speed_return_trip_l662_662284

theorem average_speed_return_trip
  (d₁ d₂ : ℝ) (s₁ s₂ : ℝ) (t_total : ℝ)
  (h₁ : d₁ = 16) (h₂ : d₂ = 16) 
  (h₃ : s₁ = 8) (h₄ : s₂ = 10) 
  (h₅ : t_total = 6.8) :
  let t₁ := d₁ / s₁ in
  let t₂ := d₂ / s₂ in
  let t_first_half := t₁ + t₂ in
  let t_return := t_total - t_first_half in
  let d_total := d₁ + d₂ in
  t_return = 3.2 → d_total = 32 → (d_total / t_return) = 10 :=
by continue

end average_speed_return_trip_l662_662284


namespace ellipse_a_b_solution_equation_of_ellipse_C_area_triangle_OMN_l662_662758

-- Define constants and conditions
def ellipse_eq (a b x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1
def focus := (sqrt 3, 0)
def e := sqrt 3 / 2
def a := 2
def b := 1
def equation_of_ellipse := (x : ℝ) (y : ℝ) := ellipse_eq 2 1 x y

-- Define the line equation
def line_through_focus (x : ℝ) := x - sqrt 3

-- Prove that a and b are derived from conditions
theorem ellipse_a_b_solution : a = 2 ∧ b = 1 := sorry

-- Prove that the equation of ellipse C is as given
theorem equation_of_ellipse_C : ∀ x y, ellipse_eq 2 1 x y := sorry

-- Prove the area of triangle OMN given the conditions
theorem area_triangle_OMN : ∀ M N : ℝ × ℝ, 
  let x1 := (8 * sqrt 3 / 5)
  let x2 := (8 / 5)
  let distance := (sqrt 6 / 2)
  let len_MN := (8 / 5)
  in 
  (|MN| * distance) = 2 * sqrt 6 / 5 := sorry

end ellipse_a_b_solution_equation_of_ellipse_C_area_triangle_OMN_l662_662758


namespace total_chairs_calculation_l662_662516

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l662_662516


namespace customers_one_bought_exactly_one_painting_l662_662620

variable (total_customers customers_two customers_four total_paintings : ℕ)

-- Defining the given conditions
def total_customers := 20
def customers_two := 4
def customers_four := 4
def total_paintings := 36

-- Representing the number of paintings sold by specific groups
def paintings_from_two : ℕ := customers_two * 2
def paintings_from_four : ℕ := customers_four * 4

-- Define the number of customers who bought one painting each
def customers_one (total_customers customers_two customers_four total_paintings : ℕ) : ℕ :=
  total_paintings - (paintings_from_two + paintings_from_four)

-- Creating the main theorem statement
theorem customers_one_bought_exactly_one_painting :
  customers_one total_customers customers_two customers_four total_paintings = 12 :=
by trivial -- The proof is trivial based on the calculations in the problem.

end customers_one_bought_exactly_one_painting_l662_662620


namespace line_properties_l662_662990

-- Let's define the given condition and the proof we aim to derive.
theorem line_properties (x y : ℝ) (h : x + sqrt 3 * y + 1 = 0) :
  (y = (-1 / sqrt 3) * x - 1 / sqrt 3 → ∃ x, x | x = -1) :=
by
  sorry


end line_properties_l662_662990


namespace seq_eventually_zero_l662_662290

noncomputable def f (k x : ℝ) : ℝ :=
  if x ≤ k then 0
  else 1 - (Real.sqrt (k * x) + Real.sqrt ((1 - k) * (1 - x))) ^ 2

noncomputable def seq_f (f : ℝ → ℝ) : ℕ → ℝ → ℝ
| 0, x := x
| (n + 1), x := f (seq_f f n x)

theorem seq_eventually_zero (k : ℝ) (h : 0 < k ∧ k < 1) :
  ∃ N : ℕ, ∀ n ≥ N, seq_f (f k) n 1 = 0 :=
sorry

end seq_eventually_zero_l662_662290


namespace common_chord_intercepted_length_l662_662965

theorem common_chord_intercepted_length :
  let C1 := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }
  let C2 := { p : ℝ × ℝ | p.1^2 + p.2^2 - 2 * p.1 - 2 * p.2 + 1 = 0 }
  let C3 := { p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 25 / 4 }
  length_of_chord_intercepted_by_C3 := λ (h_line : set (ℝ × ℝ)), -- Placeholder for the set representing the common chord
    let d := abs (1 + 1 - 1) / real.sqrt 2 in
    2 * real.sqrt ((25 / 4) - (d^2)) = real.sqrt 23 → -- The length formula applied
  sorry

end common_chord_intercepted_length_l662_662965


namespace angle_bao_proof_l662_662868

noncomputable def angle_bao : ℝ := sorry -- angle BAO in degrees

theorem angle_bao_proof 
    (CD_is_diameter : true)
    (A_on_extension_DC_beyond_C : true)
    (E_on_semicircle : true)
    (B_is_intersection_AE_semicircle : B ≠ E)
    (AB_eq_OE : AB = OE)
    (angle_EOD_30_degrees : EOD = 30) : 
    angle_bao = 7.5 :=
sorry

end angle_bao_proof_l662_662868


namespace ceil_square_count_ceil_x_eq_15_l662_662804

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l662_662804


namespace min_value_of_function_l662_662214

-- Define the function f
def f (x : ℝ) := 3 * x^2 - 6 * x + 9

-- State the theorem about the minimum value of the function.
theorem min_value_of_function : ∀ x : ℝ, f x ≥ 6 := by
  sorry

end min_value_of_function_l662_662214


namespace coloring_ways_l662_662626

-- Define the problem conditions in Lean
def chessboard := Array (Array (Option Bool))
def color (chessboard : chessboard) : Prop :=
  ∀ i, count (chessboard[i]) = 2 red ∧ 2 blue ∧
  ∀ j, count (column j chessboard) = 2 red ∧ 2 blue

-- The main theorem: The number of valid ways to color the board
theorem coloring_ways {chessboard : chessboard} : color chessboard -> 
  ∃! n, n = 90 :=
sorry

end coloring_ways_l662_662626


namespace value_of_angle_A_ratio_of_areas_ABD_and_ACD_l662_662134

open Real

def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  ∃ (A B C : ℝ), 
    (A + B + C = π) ∧
    (a = b * cos (A - π / 6) / sin B) ∧
    (b * cos C = c * cos B)

def point_D_on_BC (B C D : ℝ) (a b c : ℝ) : Prop :=
  ∃ (d : ℝ), (0 ≤ d) ∧ (d ≤ b) ∧ (a = b * cos (A - π / 6) / sin B)
  
def cos_BAD : ℝ := 4 / 5

theorem value_of_angle_A {A B C : ℝ} {a b c : ℝ}
  (h1 : triangle_ABC A B C a b c) :
  A = π / 3 :=
by
  sorry

theorem ratio_of_areas_ABD_and_ACD {A B C D : ℝ} {a b c : ℝ}
  (h1 : triangle_ABC A B C a b c) (h2 : point_D_on_BC B C D a b c)
  (h3 : cos BAD = 4 / 5) :
  (area_ABD / area_ACD = 8 * sqrt 3 + 6 / 13 :=
by
  sorry

end value_of_angle_A_ratio_of_areas_ABD_and_ACD_l662_662134


namespace f_neg_l662_662424

-- Definitions for the proof problem
def f (x : ℝ) : ℝ := if x > 0 then x^2 - 2*x + 3 else -(x^2 - 2*(-x) + 3)

lemma odd_function_f (x : ℝ) : f (-x) = -f(x) := by
  sorry

-- Goal: Show that f(x) for x < 0 is equivalent to -x^2 - 2x - 3
theorem f_neg (x : ℝ) (h : x < 0) : f(x) = -x^2 - 2x - 3 :=
by
  have hx_pos : -x > 0 := by linarith
  have h_f_neg_x : f (-x) = x^2 + 2*x + 3 := by
    unfold f
    rw if_pos hx_pos
  have h_odd : f (-x) = -f x := odd_function_f x
  rw [h_odd, <-neg_eq_iff_neg_eq] at h_f_neg_x
  simp only [neg_add, neg_neg] at h_f_neg_x
  rw ←h_f_neg_x
  sorry

end f_neg_l662_662424


namespace museum_college_students_income_l662_662854

theorem museum_college_students_income:
  let visitors := 200
  let nyc_residents := visitors / 2
  let college_students_rate := 30 / 100
  let cost_ticket := 4
  let nyc_college_students := nyc_residents * college_students_rate
  let total_income := nyc_college_students * cost_ticket
  total_income = 120 :=
by
  sorry

end museum_college_students_income_l662_662854


namespace number_of_zeros_of_f_l662_662089

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 6

theorem number_of_zeros_of_f : ∃! x : ℝ, 0 < x ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l662_662089


namespace max_regions_divided_l662_662469

theorem max_regions_divided (n m : ℕ) (h_n : n = 10) (h_m : m = 4) (h_m_le_n : m ≤ n) : 
  ∃ r : ℕ, r = 50 :=
by
  have non_parallel_lines := n - m
  have regions_non_parallel := (non_parallel_lines * (non_parallel_lines + 1)) / 2 + 1
  have regions_parallel := m * non_parallel_lines + m
  have total_regions := regions_non_parallel + regions_parallel
  use total_regions
  sorry

end max_regions_divided_l662_662469


namespace interstellar_object_distance_l662_662316

def annual_distance := 9.834 * 10^11
def years_interval := 50
def speed_factor := 2
def total_years := 150
def expected_distance := 3.4718 * 10^14

theorem interstellar_object_distance :
  let first_period := 50 * annual_distance
  let second_period := 50 * (speed_factor * annual_distance)
  let third_period := 50 * (speed_factor^2 * annual_distance)
  first_period + second_period + third_period = expected_distance := by sorry

end interstellar_object_distance_l662_662316


namespace ellipse_standard_equation_constant_value_of_PA_PB_l662_662403

-- Definitions for the problem conditions
structure Ellipse (a b : ℝ) :=
  (equation : ℝ × ℝ → Prop)
  (eccentricity : ℝ)
  (is_ellipse : ∀ p : ℝ × ℝ, equation p ↔ (p.1 ^ 2 / a ^ 2) + (p.2 ^ 2 / b ^ 2) = 1)
  (a_gt_b : a > b)
  (b_gt_zero : b > 0)
  (eccentricity_eq : eccentricity = Real.sqrt 3 / 2)

def chord_through_focus (C : Ellipse 2 1) (focus : ℝ × ℝ) : Prop := 
  ∃ y : ℝ, C.equation ⟨- Real.sqrt 3, y⟩ ∧ y = 1 / 2

def point_on_major_axis (m : ℝ) : Prop :=
  -2 ≤ m ∧ m ≤ 2

def line_through_point_slope (P : ℝ × ℝ) : ℝ × ℝ → Prop :=
  λ A, A.2 = 1 / 2 * (A.1 - P.1)

def intersection_with_ellipse (l : ℝ × ℝ → Prop) (C : Ellipse 2 1) : Set (ℝ × ℝ) :=
  { p | l p ∧ C.equation p }

-- The main theorem statements
theorem ellipse_standard_equation (C : Ellipse 2 1)
  (H1 : chord_through_focus C (⟨- Real.sqrt 3, 0⟩)) :
  C.equation = λ p, (p.1 ^ 2 / 4) + (p.2 ^ 2) = 1 := sorry

theorem constant_value_of_PA_PB (m : ℝ) 
  (H : point_on_major_axis m)
  (P : ℝ × ℝ := (m, 0))
  (l : ℝ × ℝ → Prop := line_through_point_slope P)
  (C : Ellipse 2 1)
  (H1 : chord_through_focus C (⟨- Real.sqrt 3, 0⟩))
  (A B : ℝ × ℝ)
  (HA : C.equation A ∧ l A)
  (HB : C.equation B ∧ l B) :
  (dist P A) ^ 2 + (dist P B) ^ 2 = 5 := sorry

end ellipse_standard_equation_constant_value_of_PA_PB_l662_662403


namespace find_pairs_of_positive_integers_l662_662007

theorem find_pairs_of_positive_integers (n m : ℕ) (hn : 0 < n) (hm : 0 < m) : 
  3 * 2^m + 1 = n^2 ↔ (n = 7 ∧ m = 4) ∨ (n = 5 ∧ m = 3) :=
sorry

end find_pairs_of_positive_integers_l662_662007


namespace percentage_increase_in_rate_l662_662666

variables (R x : ℝ)
variables (S : ℝ) -- sum of 400

def original_interest := (S * R * 10) / 100
def increased_interest := (S * (R + x) * 10) / 100

theorem percentage_increase_in_rate
    (hS : S = 400)
    (h_increase : increased_interest S R x - original_interest S R = 200) : x = 50 :=
sorry

end percentage_increase_in_rate_l662_662666


namespace additional_cards_l662_662327

theorem additional_cards (total_cards : ℕ) (num_decks : ℕ) (cards_per_deck : ℕ) 
  (h1 : total_cards = 319) (h2 : num_decks = 6) (h3 : cards_per_deck = 52) : 
  319 - 6 * 52 = 7 := 
by
  sorry

end additional_cards_l662_662327


namespace fractional_linear_unique_l662_662085

-- Define the distinctness requirement
def distinct {α : Type} [DecidableEq α] (x1 x2 x3 : α) : Prop :=
x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3

-- Lean statement for the equivalence proof problem
theorem fractional_linear_unique 
  {α β : Type} [LinearOrderedField α] [LinearOrderedField β]
  (x1 x2 x3 : α) (y1 y2 y3 : β)
  (h_distinct_x : distinct x1 x2 x3)
  (h_distinct_y : distinct y1 y2 y3) :
  ∃! (f : α → β), ∃ (a b c d : α), 
    (ad - bc ≠ 0) ∧ (∀ x, f x = (a * x + b) / (c * x + d)) 
      ∧ (f x1 = y1) ∧ (f x2 = y2) ∧ (f x3 = y3) :=
sorry

end fractional_linear_unique_l662_662085


namespace prob_odd_and_two_div_by_3_l662_662021

def range_size := 902
def num_odds := 451
def num_div_by_3 := 300
def num_non_div_by_3 := range_size - num_div_by_3

def prob_of_odd := (num_odds : ℝ) / range_size
def prob_of_three_div := (num_div_by_3 : ℝ) / range_size
def prob_of_not_div_by_3 := (num_non_div_by_3 : ℝ) / range_size

theorem prob_odd_and_two_div_by_3 (p : ℝ) : p < 1 / 32 :=
  let comb := (num_odds * (num_odds - 1) * (num_odds - 2) * (num_odds - 3) * (num_odds - 4) / (range_size * (range_size - 1) * (range_size - 2) * (range_size - 3) * (range_size - 4))) * 
              ((num_div_by_3 : ℝ) / num_odds)^2 * 
              ((num_non_div_by_3 : ℝ) / (num_odds - 2))^3
  in comb < 1 / 32 := sorry

end prob_odd_and_two_div_by_3_l662_662021


namespace number_of_coprime_integers_l662_662794

-- Main Theorem Statement
theorem number_of_coprime_integers (count : ℕ) :
  count = ∑ n in Finset.range 1000, if (n + 1) % 2 = 0 then 1 else 0 := 
  sorry

end number_of_coprime_integers_l662_662794


namespace minimum_value_expr_l662_662379

noncomputable def expr (x : ℝ) : ℝ := 9 * x + 3 / (x ^ 3)

theorem minimum_value_expr : (∀ x : ℝ, x > 0 → expr x ≥ 12) ∧ (∃ x : ℝ, x > 0 ∧ expr x = 12) :=
by
  sorry

end minimum_value_expr_l662_662379


namespace triangle_ABC_area_l662_662867

theorem triangle_ABC_area :
  let WXYZ_area := 64 
  let WXYZ_side := 8
  let small_square_side := 2
  let O := WXYZ_side / 2
  let BC := WXYZ_side - 2 * small_square_side
  let AM := O
  is_isosceles (triangle ABC) AB AC → 
  point_coincides_after_fold (triangle ABC) A O BC → 
  area_of_triangle ABC = 8 := by
  sorry

end triangle_ABC_area_l662_662867


namespace value_g_tan_squared_l662_662024

theorem value_g_tan_squared (t : ℝ) (g : ℝ → ℝ) :
  (∀ (x : ℝ), x ≠ 0 → x ≠ 1 → g (x / (x - 1)) = 1 / x) → 
  (0 ≤ t ∧ t ≤ π / 2) → 
  g (tan t ^ 2) = tan t ^ 2 - (tan t ^ 4) :=
by
  intro h_def h_t_range
  sorry

end value_g_tan_squared_l662_662024


namespace remainder_of_1999_pow_11_mod_8_l662_662985

theorem remainder_of_1999_pow_11_mod_8 :
  (1999 ^ 11) % 8 = 7 :=
  sorry

end remainder_of_1999_pow_11_mod_8_l662_662985


namespace negation_of_existence_l662_662217

theorem negation_of_existence (x : ℝ) (hx : 0 < x) : ¬ (∃ x_0 : ℝ, 0 < x_0 ∧ Real.log x_0 = x_0 - 1) 
  → ∀ x : ℝ, 0 < x → Real.log x ≠ x - 1 :=
by sorry

end negation_of_existence_l662_662217


namespace ratio_ravi_kiran_l662_662984

-- Definitions for the conditions
def ratio_money_ravi_giri := 6 / 7
def money_ravi := 36
def money_kiran := 105

-- The proof problem
theorem ratio_ravi_kiran : (money_ravi : ℕ) / money_kiran = 12 / 35 := 
by 
  sorry

end ratio_ravi_kiran_l662_662984


namespace parabola_directrix_l662_662039

-- Definitions for the conditions given in the problem
def is_ellipse (x y b : ℝ) : Prop :=
  x^2 + (y^2 / b^2) = 1 ∧ 0 < b ∧ b < 1

def is_parabola_focus (c : ℝ) (x y : ℝ) : Prop :=
  y^2 = 4 * c * x

def intersection_point (P : ℝ × ℝ) (c b : ℝ) : Prop :=
  let (x, y) := P in
  is_ellipse x y b ∧ is_parabola_focus c x y

def angle_condition (P F₁ F₂ : ℝ × ℝ) : Prop :=
  let (px, py) := P in
  let (f1x, f1y) := F₁ in
  let (f2x, f2y) := F₂ in
  ∠(P, F₁, F₂) = 45

-- Proof statement
theorem parabola_directrix (b : ℝ) (P F₁ F₂ : ℝ × ℝ) :
  ∀ c,
  is_ellipse (P.1) (P.2) b →
  intersection_point P c b →
  angle_condition P F₁ F₂ →
  ∃ d, d = 1 - real.sqrt 2 := 
sorry

end parabola_directrix_l662_662039


namespace exponent_property_l662_662798

theorem exponent_property (m n : ℝ) (h1 : 2^m = 5) (h2 : 2^n = 6) : 2^(m + n) = 30 := 
by
  sorry

end exponent_property_l662_662798


namespace female_population_city_l662_662282

theorem female_population_city :
  ∀ (total_population migrants_percentage rural_migrants_percentage 
     local_female_percentage rural_female_percentage urban_female_percentage 
     migrants locals rural_migrants urban_migrants local_females rural_females urban_females total_females : ℝ),
  total_population = 728400 →
  migrants_percentage = 0.35 →
  rural_migrants_percentage = 0.20 →
  local_female_percentage = 0.48 →
  rural_female_percentage = 0.30 →
  urban_female_percentage = 0.40 →
  migrants = migrants_percentage * total_population →
  locals = total_population - migrants →
  rural_migrants = rural_migrants_percentage * migrants →
  urban_migrants = migrants - rural_migrants →
  local_females = local_female_percentage * locals →
  rural_females = rural_female_percentage * rural_migrants →
  urban_females = urban_female_percentage * urban_migrants →
  total_females = local_females + rural_females + urban_females →
  total_females = 324118 :=
by 
  intros 
    _ _ _ _ _ _ 
    _ _ _ _ _ _ _ _ _ 
    htotal_population 
    hmigrants_percentage 
    hrural_migrants_percentage 
    hlocal_female_percentage 
    hrural_female_percentage 
    hurban_female_percentage 
    hmigrants 
    hlocals 
    hrural_migrants 
    hurban_migrants 
    hlocal_females 
    hrural_females 
    hurban_females 
    htotal_females; 
  rw [htotal_population, hmigrants_percentage, hrural_migrants_percentage, 
      hlocal_female_percentage, hrural_female_percentage, hurban_female_percentage] at *; 
  sorry

end female_population_city_l662_662282


namespace f_at_2_l662_662768

noncomputable def f (x : ℝ) (a b : ℝ) := a * Real.log x + b / x + x
noncomputable def g (x : ℝ) (a b : ℝ) := (a / x) - (b / (x ^ 2)) + 1

theorem f_at_2 (a b : ℝ) (ha : g 1 a b = 0) (hb : g 3 a b = 0) : f 2 a b = 1 / 2 - 4 * Real.log 2 :=
by
  sorry

end f_at_2_l662_662768


namespace added_amount_correct_l662_662777

theorem added_amount_correct (n x : ℕ) (h1 : n = 20) (h2 : 1/2 * n + x = 15) :
  x = 5 :=
by
  sorry

end added_amount_correct_l662_662777


namespace friend_gives_30_l662_662022

noncomputable def total_earnings := 10 + 30 + 50 + 40 + 70

noncomputable def equal_share := total_earnings / 5

noncomputable def contribution_of_highest_earner := 70

noncomputable def amount_to_give := contribution_of_highest_earner - equal_share

theorem friend_gives_30 : amount_to_give = 30 := by
  sorry

end friend_gives_30_l662_662022


namespace max_value_ratio_l662_662135

theorem max_value_ratio (A B C D : Point) (a b c : ℝ) 
  (h_AD_perp_BC : perpendicular AD BC) 
  (h_AD_eq_BC : AD = BC = a) 
  (h_max_value : ∀ b c, ∃ k (h_k : k = \(\frac{b}{c} + \frac{c}{b}\)), k ≤ \(\frac{3}{2} \sqrt{2}\)):
  ∃ b c, f = \(\frac{3}{2} \sqrt{2}\) := 
sorry

end max_value_ratio_l662_662135


namespace sum_of_four_integers_l662_662728

noncomputable def originalSum (a b c d : ℤ) :=
  (a + b + c + d)

theorem sum_of_four_integers
  (a b c d : ℤ)
  (h1 : (a + b + c) / 3 + d = 8)
  (h2 : (a + b + d) / 3 + c = 12)
  (h3 : (a + c + d) / 3 + b = 32 / 3)
  (h4 : (b + c + d) / 3 + a = 28 / 3) :
  originalSum a b c d = 30 :=
sorry

end sum_of_four_integers_l662_662728


namespace find_number_l662_662293

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end find_number_l662_662293


namespace cos_sum_to_product_identity_l662_662974

theorem cos_sum_to_product_identity :
  ∃ (a b c d e : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e),
    let sum : ℝ → ℝ := λ x, cos x + cos (4 * x) + cos (6 * x) + cos (9 * x) + cos (11 * x),
        product : ℝ → ℝ := λ x, a * cos (b * x) * cos (c * x) * cos (d * x) * cos (e * x) in
    (∀ x, sum x = product x) ∧ (a + b + c + d + e = 15) := sorry

end cos_sum_to_product_identity_l662_662974


namespace cyclic_quadrilateral_fourth_side_l662_662224

theorem cyclic_quadrilateral_fourth_side (AB BC CD DA : ℝ) 
  (hBC : BC = 5) (hCD : CD = 6) (hDA : DA = 12)
  (h1 : ∃ (AC : ℝ), 2 * area ABC AC hBC = 2 * area ACD AC hDA) :
  AB ∈ {2, 5, 10, 14.4} :=
sorry

end cyclic_quadrilateral_fourth_side_l662_662224


namespace number_of_non_congruent_triangles_l662_662322

-- Define the points
structure Point :=
  (x : ℚ) (y : ℚ)

-- Given points coordinates
def A : Point := ⟨0, 0⟩
def B : Point := ⟨1, 0⟩
def C : Point := ⟨2, 0⟩
def D : Point := ⟨0, 1⟩
def E : Point := ⟨1, 1⟩
def F : Point := ⟨2, 1⟩
def G : Point := ⟨0.5, 2⟩
def H : Point := ⟨1.5, 2⟩
def I : Point := ⟨2.5, 2⟩

-- Prove that there are 18 non-congruent triangles
theorem number_of_non_congruent_triangles : 
  (∃ (triangles : set (set Point)), 
   (∀ t ∈ triangles, t.card = 3) ∧ 
   (∀ t1 t2 ∈ triangles, t1 ≠ t2 → ¬is_congruent t1 t2) ∧ 
   triangles.card = 18) :=
sorry

end number_of_non_congruent_triangles_l662_662322


namespace cannot_obtain_triangle_B_l662_662495

-- Define the triangles as types
inductive Triangle
| A : Triangle
| B : Triangle
| C : Triangle
| D : Triangle
| Shaded : Triangle  -- The given shaded triangle on the right

-- Define transformations: rotation and translation only
inductive Transformation
| Rotate : Transformation
| Translate : Transformation

-- Define a function that checks if the transformation of the shaded triangle is equal to another triangle
def can_obtain_by_rotating_or_translating : Triangle → Triangle → Prop
| Triangle.Shaded Triangle.A := true
| Triangle.Shaded Triangle.B := false
| Triangle.Shaded Triangle.C := true
| Triangle.Shaded Triangle.D := true
| _ _ := false

-- Statement: proving Triangle.B cannot be obtained by the allowed transformations
theorem cannot_obtain_triangle_B :
  ¬ can_obtain_by_rotating_or_translating Triangle.Shaded Triangle.B :=
by sorry

end cannot_obtain_triangle_B_l662_662495


namespace sum_of_first_8_terms_l662_662105

theorem sum_of_first_8_terms (seq : ℕ → ℝ) (q : ℝ) (h_q : q = 2) 
  (h_sum_first_4 : seq 0 + seq 1 + seq 2 + seq 3 = 1) 
  (h_geom : ∀ n, seq (n + 1) = q * seq n) : 
  seq 0 + seq 1 + seq 2 + seq 3 + seq 4 + seq 5 + seq 6 + seq 7 = 17 := 
sorry

end sum_of_first_8_terms_l662_662105


namespace complex_magnitude_theorem_l662_662537

theorem complex_magnitude_theorem (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  ∣(1 / z) + (1 / w)∣ = 3 / 8 := by
  sorry

end complex_magnitude_theorem_l662_662537


namespace option_C_correct_l662_662027

def f (x : ℝ) : ℝ := 3 + Real.cos (2 * x)

theorem option_C_correct : ∃ x ∈ Ioo 0 (3 * Real.pi), f x = 4 :=
sorry

end option_C_correct_l662_662027


namespace probability_co_presidents_l662_662117

open_locale big_operators

theorem probability_co_presidents (n1 n2 n3 p1 p2 p3 : ℕ) (h_n1 : n1 = 6) (h_n2 : n2 = 9) (h_n3 : n3 = 10)
  (h_p1 : p1 = 2) (h_p2 : p2 = 3) (h_p3 : p3 = 2) :
  (1 / 3 : ℚ) * ((↑(nat.choose p1 2 * nat.choose (n1 - p1) 2) / ↑(nat.choose n1 4)) +
    (↑(nat.choose p2 2 * nat.choose (n2 - p2) 2) / ↑(nat.choose n2 4)) +
    (↑(nat.choose p3 2 * nat.choose (n3 - p3) 2) / ↑(nat.choose n3 4))) = 11 / 42 :=
by
  sorry

end probability_co_presidents_l662_662117


namespace fraction_exponentiation_l662_662684

theorem fraction_exponentiation : (3/4 : ℚ)^3 = 27/64 := by
  sorry

end fraction_exponentiation_l662_662684


namespace find_polynomial_A_and_y_l662_662063

noncomputable theory

open Polynomial

theorem find_polynomial_A_and_y (x y : ℝ) (A B : Polynomial ℝ) 
  (h1 : A + B = (C * x^2 * y + 2 * x * y + 5))
  (h2 : B = (3 * x^2 * y - 5 * x * y + x + 7)) :
  (A = 9 * x^2 * y + 7 * x * y - x - 2) ∧ (y = 2 / 11) :=
by
  sorry

end find_polynomial_A_and_y_l662_662063


namespace candles_problem_l662_662350

theorem candles_problem (total_candles : ℕ) (yellow_candles : ℕ) (blue_candles : ℕ) : (total_candles - yellow_candles - blue_candles) = 14 :=
by
  -- Given values
  let total_candles := 79
  let yellow_candles := 27
  let blue_candles := 38
  -- We need to prove 79 - 27 - 38 = 14
  show 79 - 27 - 38 = 14
  sorry

end candles_problem_l662_662350


namespace typist_original_salary_l662_662226

theorem typist_original_salary (S : ℝ) (h1 : S * 1.10 * 0.95 * 1.07 * 0.97 * 0.92 + 500 = 2090) : S ≈ 1625.63 :=
by
  sorry

end typist_original_salary_l662_662226


namespace train_crossing_time_l662_662446

noncomputable def train_time_to_cross
  (length1 : ℕ) (speed1_kmh : ℕ) (length2 : ℕ) (speed2_kmh : ℕ) : ℕ :=
let speed1_ms := (speed1_kmh * 1000 / 3600) in -- converting km/h to m/s
let speed2_ms := (speed2_kmh * 1000 / 3600) in -- converting km/h to m/s
let relative_speed := speed1_ms - speed2_ms in
let total_length := length1 + length2 in
total_length / relative_speed

theorem train_crossing_time
  (length1 : ℕ) (speed1_kmh : ℕ) (length2 : ℕ) (speed2_kmh : ℕ)
  (h_length1 : length1 = 380)
  (h_speed1_kmh : speed1_kmh = 72)
  (h_length2 : length2 = 540)
  (h_speed2_kmh : speed2_kmh = 36) :
  train_time_to_cross length1 speed1_kmh length2 speed2_kmh = 92 :=
by {
  unfold train_time_to_cross,
  rw [h_length1, h_speed1_kmh, h_length2, h_speed2_kmh],
  norm_num, -- this simplifies the arithmetic expressions
  sorry, -- this skips the proof
}

end train_crossing_time_l662_662446


namespace first_pair_cost_l662_662922

def price_of_first_pair (P : ℝ) : Prop :=
  let second_pair := 1.5 * P in
  let total_price := P + second_pair in
  total_price = 55

theorem first_pair_cost : ∃ P : ℝ, price_of_first_pair P ∧ P = 22 :=
by
  use 22
  unfold price_of_first_pair
  dsimp
  split
  calc
    P + second_pair = 22 + 1.5 * 22 : by norm_num
    ... = 22 + 33 : by norm_num
    ... = 55 : by norm_num
  done

end first_pair_cost_l662_662922


namespace shen_final_position_is_west_3_km_shen_total_distance_is_55_km_shen_total_earnings_is_130_yuan_l662_662362

-- Assume the distances travelled in each batch
def distances : List ℤ := [+8, -6, +3, -7, +8, +4, -9, -4, +3, -3]

-- Calculate the final position from the starting point
def final_position : ℤ := distances.sum

-- Calculate the total distance driven
def total_distance : ℤ := (distances.map Int.natAbs).sum

-- Fare calculation constants
def starting_price : ℕ := 8
def base_distance : ℕ := 3
def per_km_rate : ℕ := 2

-- Fare calculation function for a single trip
def fare (distance : ℤ) : ℕ :=
  let travelled := distance.natAbs
  in if travelled <= base_distance then
       starting_price
     else
       starting_price + per_km_rate * (travelled - base_distance)

-- Calculate the total earnings from all trips
def total_earnings : ℕ :=
  (distances.map fare).sum

theorem shen_final_position_is_west_3_km :
  final_position = -3 :=
by sorry

theorem shen_total_distance_is_55_km :
  total_distance = 55 :=
by sorry

theorem shen_total_earnings_is_130_yuan :
  total_earnings = 130 :=
by sorry

end shen_final_position_is_west_3_km_shen_total_distance_is_55_km_shen_total_earnings_is_130_yuan_l662_662362


namespace proof_g_3x_eq_3gx_l662_662453

def g (x : ℝ) : ℝ := (↑(x + 5)/5)^(1/3)

theorem proof_g_3x_eq_3gx (x : ℝ) (h : x = -65 / 12) : 
  g (3 * x) = 3 * g x :=
by
  sorry

end proof_g_3x_eq_3gx_l662_662453


namespace functional_equation_solution_l662_662006

noncomputable def f (x : ℝ) : ℝ := sorry

theorem functional_equation_solution : 
  (∀ x y : ℝ, f (⌊x⌋ * y) = f x * ⌊f y⌋) 
  → ∃ c : ℝ, (c = 0 ∨ (1 ≤ c ∧ c < 2)) ∧ (∀ x : ℝ, f x = c) :=
by
  intro h
  sorry

end functional_equation_solution_l662_662006


namespace position_2005_is_1_l662_662614

def repeating_sequence (n : Nat) : Nat :=
  let pattern := [1, 2, 3, 4, 3, 2]
  pattern[(n % 6)]

theorem position_2005_is_1 : repeating_sequence 2005 = 1 := by
  sorry

end position_2005_is_1_l662_662614


namespace range_of_f_on_interval_l662_662399

noncomputable def f: ℝ → ℝ := sorry

axiom f_additive : ∀ x y : ℝ, f(x + y) = f(x) + f(y)
axiom f_positive : ∀ x : ℝ, x > 0 → f(x) > 0
axiom f_neg_one : f(-1) = -2

theorem range_of_f_on_interval : 
  set.range (λ x, f x) (set.Icc (-2 : ℝ) 1) = set.Icc (-4 : ℝ) 2 := sorry

end range_of_f_on_interval_l662_662399


namespace john_plot_trees_l662_662877

theorem john_plot_trees :
  ∃ (number_of_columns : ℕ), 
    3 * number_of_columns = (30 / 0.5) / 5 ∧ number_of_columns = 4 :=
by
  sorry

end john_plot_trees_l662_662877


namespace smallest_x_l662_662264

theorem smallest_x (x : ℕ) (h900 : 900 = 2^2 * 3^2 * 5^2) (h1152 : 1152 = 2^7 * 3^2) : 
  (900 * x) % 1152 = 0 ↔ x = 32 := 
by
  sorry

end smallest_x_l662_662264


namespace find_k_l662_662035

-- Definitions based on conditions
def sequence : ℕ → ℝ
noncomputable def sequence a_n (a: ℕ → ℝ) : Prop :=
  ∀ n, a n - a (n + 1) = (a n * a (n + 1)) / 2^(n - 1)

def initial_condition (a: ℕ → ℝ) : Prop :=
  a 2 = -1

-- Mathematical statement
theorem find_k (a: ℕ → ℝ) : (sequence a ∧ initial_condition a) → (∃ k : ℕ, k = 12 ∧ a k = 16 * a 8) :=
begin
  sorry
end

end find_k_l662_662035


namespace eat_both_veg_nonveg_l662_662114

theorem eat_both_veg_nonveg (total_veg only_veg : ℕ) (h1 : total_veg = 31) (h2 : only_veg = 19) :
  (total_veg - only_veg) = 12 :=
by
  have h3 : total_veg - only_veg = 31 - 19 := by rw [h1, h2]
  exact h3

end eat_both_veg_nonveg_l662_662114


namespace polynomial_remainder_l662_662718

theorem polynomial_remainder (x : ℂ) (hx : x^5 = 1) :
  (x^25 + x^20 + x^15 + x^10 + x^5 + 1) % (x^5 - 1) = 6 :=
by
  -- Proof will go here
  sorry

end polynomial_remainder_l662_662718


namespace find_radii_of_circles_l662_662248

-- Definitions based on the provided conditions
variables {r a : ℝ} (r_pos : r > 0) (a_pos : a > 0) (r_neq_a : r ≠ a)

-- Lean definition to formalize the problem
def radii_of_circles (x y : ℝ) : Prop :=
  x = (a * r) / (a - r) ∧ y = (a^2 * r) / ((a - r)^2)

-- Statement with the conditions and the desired outcome
theorem find_radii_of_circles (x y : ℝ) (r_pos : r > 0) (a_pos : a > 0) (r_neq_a : r ≠ a)
  (large_through_medium : dist (0:ℝ) (y:ℝ) = y)
  (medium_through_small : dist (0:ℝ) (x:ℝ) = x) 
  : radii_of_circles r a x y := 
  sorry -- proof omitted

end find_radii_of_circles_l662_662248


namespace pairs_satisfying_condition_l662_662523

theorem pairs_satisfying_condition (x y : ℤ) (h : x + y ≠ 0) :
  (x^2 + y^2)/(x + y) = 10 ↔ (x, y) = (12, 6) ∨ (x, y) = (-2, 6) ∨ (x, y) = (12, 4) ∨ (x, y) = (-2, 4) ∨ (x, y) = (10, 10) ∨ (x, y) = (0, 10) ∨ (x, y) = (10, 0) :=
sorry

end pairs_satisfying_condition_l662_662523


namespace centers_form_rectangle_l662_662880

-- Define the problem conditions
variables {C C1 C2 C3 C4 : Circle} {O O1 O2 O3 O4 : Point}

-- Assume conditions about the circles and their tangencies
axiom C_radius : radius(C) = 2
axiom C1_radius : radius(C1) = 1
axiom C2_radius : radius(C2) = 1
axiom C1_C2_tangent : tangent(C1, C2)
axiom C1_inside_C : inside(C1, C)
axiom C2_inside_C : inside(C2, C)
axiom C3_tangent_C : tangent(C3, C)
axiom C3_tangent_C1 : tangent(C3, C1)
axiom C3_tangent_C2 : tangent(C3, C2)
axiom C3_inside_C : inside(C3, C)
axiom C4_tangent_C : tangent(C4, C)
axiom C4_tangent_C1 : tangent(C4, C1)
axiom C4_tangent_C3 : tangent(C4, C3)
axiom C4_inside_C : inside(C4, C)

-- State the proof goal
theorem centers_form_rectangle 
  (C1_C2_tangent : tangent(C1, C2)) 
  (C1_inside_C : inside(C1, C))
  (C2_inside_C : inside(C2, C))
  (C3_tangent_C : tangent(C3, C))
  (C3_tangent_C1 : tangent(C3, C1))
  (C3_tangent_C2 : tangent(C3, C2))
  (C3_inside_C : inside(C3, C))
  (C4_tangent_C : tangent(C4, C))
  (C4_tangent_C1 : tangent(C4, C1))
  (C4_tangent_C3 : tangent(C4, C3))
  (C4_inside_C : inside(C4, C)) :
  is_rectangle(O, O1, O3, O4) :=
sorry

end centers_form_rectangle_l662_662880


namespace find_two_digit_number_l662_662670

def tens_digit (n: ℕ) := n / 10
def unit_digit (n: ℕ) := n % 10
def is_required_number (n: ℕ) : Prop :=
  tens_digit n + 2 = unit_digit n ∧ n < 30 ∧ 10 ≤ n

theorem find_two_digit_number (n : ℕ) :
  is_required_number n → n = 13 ∨ n = 24 :=
by
  -- Proof placeholder
  sorry

end find_two_digit_number_l662_662670


namespace museum_earnings_from_nyc_college_students_l662_662850

def visitors := 200
def nyc_residents_fraction := 1 / 2
def college_students_fraction := 0.30
def ticket_price := 4

theorem museum_earnings_from_nyc_college_students : 
  ((visitors * nyc_residents_fraction * college_students_fraction) * ticket_price) = 120 := 
by 
  sorry

end museum_earnings_from_nyc_college_students_l662_662850


namespace minimum_socks_for_pairs_l662_662840

theorem minimum_socks_for_pairs (n m : ℕ) (h_n : n = 10) (h_m : m = 5) : (2 * n + m - 1) = 24 :=
begin
  -- Proof will show that under the given conditions, the minimum number of socks needed is 24
  rw [h_n, h_m],
  norm_num,
end

end minimum_socks_for_pairs_l662_662840


namespace option_a_is_correct_l662_662273

theorem option_a_is_correct : 
  ∀ (x ∈ ({-3, -2, -1, 0} : set ℤ)), x < -2 ↔ x = -3 :=
by sorry

end option_a_is_correct_l662_662273


namespace find_zeros_graph_above_implies_b_gt_2_solve_inequality_l662_662073

def f (a b x : ℝ) : ℝ := a * x ^ 2 + (2 * a + 1) * x + b

theorem find_zeros (x : ℝ) : (f 1 (-4) x = 0) ↔ (x = -4 ∨ x = 1) :=
sorry

theorem graph_above_implies_b_gt_2 (a b : ℝ) (H : ∀ x, f(a b x) > x + 2) : b > 2 :=
sorry

theorem solve_inequality (a : ℝ) (x : ℝ) :
  (a < 0 → x < -2 ∨ x > -1 / a) ∧
  (a = 0 → x < -2) ∧
  (0 < a ∧ a < 1/2 → -1 / a < x ∧ x < -2) ∧
  (a = 1/2 → False) ∧
  (a > 1/2 → -2 < x ∧ x < -1 / a) :=
sorry

end find_zeros_graph_above_implies_b_gt_2_solve_inequality_l662_662073


namespace range_of_a_l662_662733

-- Define the piecewise function f 
noncomputable def f (a : ℝ) : ℝ → ℝ :=
λ x, if x < 1 then (2 - a) * x + 1 else a ^ x

-- Statement of the theorem to be proven
theorem range_of_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≠ x2 → (f a x1 - f a x2) / (x1 - x2) > 0) → (3 / 2 ≤ a ∧ a < 2) :=
begin
  sorry
end

end range_of_a_l662_662733


namespace minimum_k_value_l662_662047

theorem minimum_k_value (a b k : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∀ a b, (1 / a + 1 / b + k / (a + b)) ≥ 0) : k ≥ -4 :=
sorry

end minimum_k_value_l662_662047


namespace fran_red_macaroons_baked_l662_662726

theorem fran_red_macaroons_baked (R : ℕ) :
  ∃ (R : ℕ), 
    let G := 40 in 
    let green_macaroons_eaten := 15 in 
    let red_macaroons_eaten := 2 * green_macaroons_eaten in
    let remaining_macaroons := 45 in
    let remaining_green_macaroons := G - green_macaroons_eaten in
    let remaining_red_macaroons := remaining_macaroons - remaining_green_macaroons in
    (R = red_macaroons_eaten + remaining_red_macaroons) → R = 50 :=
by
  sorry

end fran_red_macaroons_baked_l662_662726


namespace minimum_value_is_46_l662_662018

def minimum_value_expression (a b c : ℕ) : ℕ :=
  ⌊(8 * (a + b)) / c⌋ + ⌊(8 * (a + c)) / b⌋ + ⌊(8 * (b + c)) / a⌋

theorem minimum_value_is_46 : ∀ a b c : ℕ, 46 = minimum_value_expression a b c :=
sorry

end minimum_value_is_46_l662_662018


namespace solution_l662_662763

def f (x : ℝ) : ℝ := 
  if x < 2 then 
    -2^x 
  else 
    Real.log (x^2 - 1) / Real.log 3

theorem solution (a : ℝ) (h : f a = 1) : a = 2 :=
begin
  -- We need to prove this, but it's skipped with sorry
  sorry
end

end solution_l662_662763


namespace three_digit_numbers_property_l662_662444

theorem three_digit_numbers_property : 
  let valid_numbers := 
    (finset.univ : finset (fin 10)).filter (λ d3, 1 ≤ d3 ∧ d3 ≤ 9).bind
      (λ d3, 
        (finset.univ : finset (fin 10)).filter (λ d2, d2 ≥ 3 * d3).bind
          (λ d2, 
            (finset.univ : finset (fin 10)).filter (λ d1, d1 ≥ 2 * d2).image 
              (λ d1, (d3, d2, d1)))).card = 6 := 
by sorry

end three_digit_numbers_property_l662_662444


namespace simplify_expression_l662_662573

theorem simplify_expression :
  ( (32:ℝ)^(1/3) - (sqrt (17/4)) )^2 = ((8 * (2:ℝ)^(1/3) - (sqrt 17))^2)/4 := by
  sorry

end simplify_expression_l662_662573


namespace wire_around_field_l662_662016

theorem wire_around_field (A L : ℕ) (hA : A = 69696) (hL : L = 15840) : L / (4 * (Nat.sqrt A)) = 15 :=
by
  sorry

end wire_around_field_l662_662016


namespace cost_of_grapes_and_watermelon_l662_662580

noncomputable def price_of_oranges : ℝ := 4.8

theorem cost_of_grapes_and_watermelon :
  ∃ (o g w f : ℝ), 
    o + g + w + f = 24 ∧ 
    f = 3 * o ∧ 
    w = o - 2 * g ∧ 
    g + w = 4.8 :=
begin
  existsi price_of_oranges,
  existsi (5 * price_of_oranges - 24),
  existsi (price_of_oranges - 2 * (5 * price_of_oranges - 24)),
  existsi (3 * price_of_oranges),
  simp [price_of_oranges],
  sorry,
end

end cost_of_grapes_and_watermelon_l662_662580


namespace negative_integer_solution_l662_662183

theorem negative_integer_solution (x : ℤ) (h : 3 * x + 13 ≥ 0) : x = -1 :=
by
  sorry

end negative_integer_solution_l662_662183


namespace max_difference_of_products_in_100_gon_l662_662363

theorem max_difference_of_products_in_100_gon : 
  ∃ (x : Fin 100 → ℝ), (∑ i, x i ^ 2 = 1) ∧ 
  (
    let k := ∑ i j, if (i < j) ∧ ((i + j) % 2 = 0) then x i * x j else 0
    let s := ∑ i j, if (i < j) ∧ ((i + j) % 2 ≠ 0) then x i * x j else 0
    k - s ≤ 1 / 2
  ) :=
sorry

end max_difference_of_products_in_100_gon_l662_662363


namespace problem_statement_l662_662766

open Real

-- Definitions of the functions and their conditions
def f (x : ℝ) : ℝ := ln x - x + 1
def g (a x : ℝ) : ℝ := a * x * exp x - 4 * x

-- Stating the main theorem
theorem problem_statement (a : ℝ) (h_pos : 0 < a) (x : ℝ) (hx_pos : 0 < x) :
  (∀ x, 0 < x ∧ x < 1 → deriv f x > 0) ∧
  (∀ x, 1 < x → deriv f x < 0) ∧
  (g a x - 2 * f x ≥ 2 * (ln a - ln 2)) :=
sorry

end problem_statement_l662_662766


namespace cost_of_other_disc_l662_662665

theorem cost_of_other_disc (x : ℝ) (total_spent : ℝ) (num_discs : ℕ) (num_850_discs : ℕ) (price_850 : ℝ) 
    (total_cost : total_spent = 93) (num_bought : num_discs = 10) (num_850 : num_850_discs = 6) (price_per_850 : price_850 = 8.50) 
    (total_cost_850 : num_850_discs * price_850 = 51) (remaining_discs_cost : total_spent - 51 = 42) (remaining_discs : num_discs - num_850_discs = 4) :
    total_spent = num_850_discs * price_850 + (num_discs - num_850_discs) * x → x = 10.50 :=
by
  sorry

end cost_of_other_disc_l662_662665


namespace volume_ratio_l662_662146

theorem volume_ratio (k a : ℝ) (hk : 0 < k) (ha : 0 < a) :
  let V1 := π * ∫ x in 0..a, (x / (x + k)) ^ 2
      V2 := 2 * π * ∫ x in 0..a, x * (a / (a + k) - x / (x + k))
  in V2 / V1 = k :=
by
  let V1 := π * ∫ x in 0..a, (x / (x + k)) ^ 2
  let V2 := 2 * π * ∫ x in 0..a, x * (a / (a + k) - x / (x + k))
  sorry

end volume_ratio_l662_662146


namespace circle_equation_through_points_l662_662715

theorem circle_equation_through_points 
  (M N : ℝ × ℝ)
  (hM : M = (5, 2))
  (hN : N = (3, 2))
  (hk : ∃ k : ℝ, (M.1 + N.1) / 2 = k ∧ (M.2 + N.2) / 2 = (2 * k - 3))
  : (∃ h : ℝ, ∀ x y: ℝ, (x - 4) ^ 2 + (y - 5) ^ 2 = h) ∧ (∃ r : ℝ, r = 10) := 
sorry

end circle_equation_through_points_l662_662715


namespace work_completion_l662_662633

theorem work_completion (W : ℕ) (a_rate b_rate combined_rate : ℕ) 
  (h1: combined_rate = W/8) 
  (h2: a_rate = W/12) 
  (h3: combined_rate = a_rate + b_rate) 
  : combined_rate = W/8 :=
by
  sorry

end work_completion_l662_662633


namespace geometry_problem_l662_662992

noncomputable def radius (R : ℝ) := R

def side_length_pentagon (R : ℝ) : ℝ :=
  (radius R) / 2 * real.sqrt (10 - 2 * real.sqrt 5)

def side_length_decagon (R : ℝ) : ℝ :=
  (radius R) / 2 * (real.sqrt 5 - 1)

def side_length_hexagon (R : ℝ) : ℝ :=
  radius R

theorem geometry_problem (R : ℝ) :
  (side_length_pentagon R)^2 = (side_length_decagon R)^2 + (side_length_hexagon R)^2 :=
by
  sorry

end geometry_problem_l662_662992


namespace simplify_polynomial_l662_662192

theorem simplify_polynomial (q : ℚ) :
  (4 * q^3 - 7 * q^2 + 3 * q - 2) + (5 * q^2 - 9 * q + 8) = 4 * q^3 - 2 * q^2 - 6 * q + 6 := 
by 
  sorry

end simplify_polynomial_l662_662192


namespace length_PT_l662_662478

-- Define the geometric properties and given conditions
structure Pentagon :=
(Q R S T P K L : Type)
(QR RS ST : ℝ)
(angleT : ℝ)
(angleQ angleR angleS : ℝ)

-- Introduce a noncomputable instance of the pentagon with given conditions
noncomputable def given_pentagon : Pentagon :=
{
  QR := 3,
  RS := 3,
  ST := 3,
  angleT := 90,
  angleQ := 120,
  angleR := 120,
  angleS := 120,
  Q := ℝ,
  R := ℝ,
  S := ℝ,
  T := ℝ,
  P := ℝ,
  K := ℝ,
  L := ℝ,
}

-- Define the statement to be proven with the correct answer
theorem length_PT (PQRST: Pentagon) : 
  PQRST.Q != PQRST.T -> 
  (exists (PT : ℝ), PT = 6 + 3 * Real.sqrt 3) -> 
  (let c := 6,
  let d := 3 in c + d = 9) :=
by
  sorry

end length_PT_l662_662478


namespace tangent_line_condition_l662_662074

theorem tangent_line_condition (x₀ : ℝ) (h₀ : x₀ > 0) (hx₀ : sqrt 3 < x₀) (hx₀' : x₀ < 2) :
  ∃ m : ℝ, 0 < m ∧ m < 1 ∧ ((1/2) * x₀^2) - log (1/x₀) - 1 = 0 ∧
    (∀ x : ℝ, x > 1 → (1/2) * x^2 - log x - 1 > 0) :=
by
  sorry

end tangent_line_condition_l662_662074


namespace binomial_coeff_sum_l662_662031

theorem binomial_coeff_sum {a a_1 a_2 a_3 a_4 a_5 a_6 a_7 : ℤ}
  (h : (1 - x)^7 = a + a_1 * x + a_2 * x^2 + a_3 * x^3 + a_4 * x^4 + a_5 * x^5 + a_6 * x^6 + a_7 * x^7) :
  |a| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| + |a_7| = 128 :=
by
  sorry

end binomial_coeff_sum_l662_662031


namespace sum_of_cubes_roots_theorem_l662_662685

noncomputable def math_problem (x : ℝ) : Prop :=
  (x^(1/3) * x + 4 * x - 9 * x^(1/3) + 2 = 0)

noncomputable def sum_of_cubes_roots (f : ℝ → Prop) : ℝ :=
  let roots := {x : ℝ | f x} in
  ∑ x in roots, (x^(3))

theorem sum_of_cubes_roots_theorem :
  ∀ (roots : set ℝ), (∀ x ∈ roots, x ≥ 0) →
  (∀ x ∈ roots, math_problem x) →
  sum_of_cubes_roots math_problem = k :=
sorry

end sum_of_cubes_roots_theorem_l662_662685


namespace closest_multiple_of_15_to_2023_is_2025_l662_662267

theorem closest_multiple_of_15_to_2023_is_2025 (n : ℤ) (h : 15 * n = 2025) : 
  ∀ m : ℤ, abs (2023 - 2025) ≤ abs (2023 - 15 * m) :=
by
  exact sorry

end closest_multiple_of_15_to_2023_is_2025_l662_662267


namespace magnitude_inverse_sum_eq_l662_662540

noncomputable def complex_magnitude_inverse_sum (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
|1/z + 1/w|

theorem magnitude_inverse_sum_eq (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_magnitude_inverse_sum z w hz hw hzw = 3 / 8 :=
by sorry

end magnitude_inverse_sum_eq_l662_662540


namespace plane_line_infinite_intersection_l662_662632

variable {Point Line Plane : Type}
variable (line_in_plane : Line → Plane → Prop)
variable (infinite_common_points : Line → Plane → Prop)
variable (line_in_space : Line)
variable (plane_in_space : Plane)

-- Assuming the property that if a line is in a plane, then they have infinite common points.
axiom line_plane_infinite_common_points :
  line_in_plane line_in_space plane_in_space → infinite_common_points line_in_space plane_in_space

theorem plane_line_infinite_intersection :
  line_in_plane line_in_space plane_in_space → infinite_common_points line_in_space plane_in_space :=
by
  apply line_plane_infinite_common_points
  sorry

end plane_line_infinite_intersection_l662_662632


namespace part1_values_part2_decreasing_part3_range_l662_662157

variables {f : ℝ → ℝ}

axiom f_cond1 : ∀ {x y : ℝ}, x > 0 → y > 0 → f (x * y) = f x + f y
axiom f_cond2 : ∀ {x : ℝ}, x > 1 → f x < 0
axiom f_cond3 : f 3 = -1

-- Part 1: Proving the values of f(1) and f(1/9)
theorem part1_values :
  f 1 = 0 ∧ f (1/9) = 2 := sorry

-- Part 2: Proving the function f(x) is decreasing on ℝ⁺
theorem part2_decreasing (x1 x2 : ℝ) (hx1 : x1 > 0) (hx2 : x2 > 0) (h : x1 < x2) :
  f x2 < f x1 := sorry

-- Part 3: Find range of values for x such that f(x) + f(2-x) < 2
theorem part3_range (x : ℝ) (hx : 0 < x ∧ x < 2) (h : f x + f (2 - x) < 2) :
  1 - (2 * real.sqrt 2) / 3 < x ∧ x < 1 + (2 * real.sqrt 2) / 3 := sorry

end part1_values_part2_decreasing_part3_range_l662_662157


namespace domain_of_sqrt_l662_662207

def domain_of_function := {x : ℝ | 1 ≤ x ∧ x ≤ 2}

theorem domain_of_sqrt : 
  ∀ (x : ℝ), (∃ y : ℝ, y = (sqrt (x - 1)) + (sqrt (2 - x))) ↔ x ∈ domain_of_function :=
by 
  sorry

end domain_of_sqrt_l662_662207


namespace exists_parallel_line_dividing_triangle_l662_662471

theorem exists_parallel_line_dividing_triangle (A B C O : Point) (hBC : O ∈ segment B C) :
  ∃ l, parallel l (line_through A O) ∧ divides_triangle A B C l :=
sorry

end exists_parallel_line_dividing_triangle_l662_662471


namespace chord_lengths_less_than_l662_662463

theorem chord_lengths_less_than (r k : ℝ) (chords : list ℝ) 
  (h_r : r = 1)
  (h_diameter_intersect : ∀ (d : ℝ), d ∈ diameters -> (diameter_intersections d chords).length ≤ k) :  
  sum chords < π * k := 
sorry

end chord_lengths_less_than_l662_662463


namespace factorization_property_l662_662210

theorem factorization_property (a b : ℤ) (h1 : 25 * x ^ 2 - 160 * x - 144 = (5 * x + a) * (5 * x + b)) 
    (h2 : a + b = -32) (h3 : a * b = -144) : 
    a + 2 * b = -68 := 
sorry

end factorization_property_l662_662210


namespace length_of_MN_eq_5_sqrt_10_div_3_l662_662485

theorem length_of_MN_eq_5_sqrt_10_div_3 
  (A : ℝ × ℝ) 
  (B : ℝ × ℝ)
  (C : ℝ × ℝ)
  (D : ℝ × ℝ)
  (M : ℝ × ℝ)
  (N : ℝ × ℝ)
  (hyp_A : A = (1, 3))
  (hyp_B : B = (25 / 3, 5 / 3))
  (hyp_C : C = (22 / 3, 14 / 3))
  (hyp_eq_edges : (dist (0, 0) M = dist M N) ∧ (dist M N = dist N B))
  (hyp_D : D = (5 / 2, 15 / 2))
  (hyp_M : M = (5 / 3, 5)) :
  dist M N = 5 * Real.sqrt 10 / 3 :=
sorry

end length_of_MN_eq_5_sqrt_10_div_3_l662_662485


namespace problem_statement_l662_662753

noncomputable def pole_origin (ρ θ: ℝ) : ℝ × ℝ := 
  let x := ρ * Real.cos θ
  let y := ρ * Real.sin θ
  (x, y)

noncomputable def polar_curve (ρ : ℝ) (θ : ℝ) : ℝ := 
  ρ = 2 * (Real.cos θ + Real.sin θ)

noncomputable def rectangular_curve (x y : ℝ) : Prop := 
  (x-1)^2 + (y-1)^2 = 2

noncomputable def line_l (t: ℝ) : ℝ × ℝ :=
  ( (Real.sqrt 2 / 2) * t, 1 + (Real.sqrt 2 / 2) * t )

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem problem_statement :
  (∀ ρ θ, polar_curve ρ θ → ∃ x y, rectangular_curve x y) ∧
  (∀ t, let E := (0, 1)
   let (x1, y1) := line_l t 
   let A := (x1, y1)
   let B := ( (Real.sqrt 2 / 2) * (-t), 1 + (Real.sqrt 2 / 2) * (-t) )
   rectangular_curve x1 y1 →
   distance E A ≠ 0 →
   distance E B ≠ 0 →
   (1 / distance E A) + (1 / distance E B) = Real.sqrt 6) :=
sorry

end problem_statement_l662_662753


namespace cows_black_more_than_half_l662_662943

theorem cows_black_more_than_half (t b : ℕ) (h1 : t = 18) (h2 : t - 4 = b) : b - t / 2 = 5 :=
by
  sorry

end cows_black_more_than_half_l662_662943


namespace daniel_age_l662_662693

def isAgeSet (s : Set ℕ) : Prop :=
  s = {4, 6, 8, 10, 12, 14}

def sumTo18 (s : Set ℕ) : Prop :=
  ∃ (a b : ℕ), a ∈ s ∧ b ∈ s ∧ a + b = 18 ∧ a ≠ b

def youngerThan11 (s : Set ℕ) : Prop :=
  ∀ (a : ℕ), a ∈ s → a < 11

def staysHome (DanielAge : ℕ) (s : Set ℕ) : Prop :=
  6 ∈ s ∧ DanielAge ∈ s

theorem daniel_age :
  ∀ (ages : Set ℕ) (DanielAge : ℕ),
    isAgeSet ages →
    (∃ s, sumTo18 s ∧ s ⊆ ages) →
    (∃ s, youngerThan11 s ∧ s ⊆ ages ∧ 6 ∉ s) →
    staysHome DanielAge ages →
    DanielAge = 12 :=
by
  intros ages DanielAge isAgeSetAges sumTo18Ages youngerThan11Ages staysHomeDaniel
  sorry

end daniel_age_l662_662693


namespace longest_cyclic_non_intersecting_route_l662_662923

-- We define the context for the problem: a 999 x 999 grid and the constraints on a limp rook.
def cyclic_non_intersecting_route {m n : ℕ} (board : Array (Array Bool)) (start_pos : (ℕ × ℕ)) : Prop :=
  -- Function to check if a given route is cyclic and non-intersecting.
  sorry -- Full implementation would include these detailed conditions

theorem longest_cyclic_non_intersecting_route:
  let n := 999 in
  ∀ (board : Array (Array Bool)),
  ∃ m : ℕ, 
  cyclic_non_intersecting_route board (n, n) ∧
  m = 996000 :=
sorry

end longest_cyclic_non_intersecting_route_l662_662923


namespace positive_integers_exist_l662_662712

theorem positive_integers_exist (a b c : ℕ) (h1 : a = 28) (h2 : b = 7) (h3 : c = 14) :
    2 * real.sqrt (cbrt 7 - cbrt 6) = cbrt a - cbrt b + cbrt c ∧ a + b + c = 49 := 
by
  sorry

end positive_integers_exist_l662_662712


namespace tan_arithmetic_sequence_l662_662754

theorem tan_arithmetic_sequence {a : ℕ → ℝ}
  (h_arith : ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + n * d)
  (h_sum : a 1 + a 7 + a 13 = Real.pi) :
  Real.tan (a 2 + a 12) = - Real.sqrt 3 :=
sorry

end tan_arithmetic_sequence_l662_662754


namespace line_through_two_points_l662_662973

theorem line_through_two_points :
  ∃ (m b : ℝ), (∀ (x y : ℝ), (x, y) = (1, 3) ∨ (x, y) = (3, 7) → y = m * x + b) ∧ (m + b = 3) := by
{ sorry }

end line_through_two_points_l662_662973


namespace log_base_inequality_l662_662101

theorem log_base_inequality (a : ℝ) (h : Real.logBase a (3 / 5) < 1) : (0 < a ∧ a < 3 / 5) ∨ (1 < a) :=
by 
  sorry

end log_base_inequality_l662_662101


namespace divide_friends_among_teams_l662_662797

theorem divide_friends_among_teams :
  ∃ ways : Nat, ways = 3 ^ 7 ∧ ways = 2187 :=
by
  use 2187
  split
  · sorry
  · rfl

end divide_friends_among_teams_l662_662797


namespace night_crew_worker_fraction_l662_662680

noncomputable def box_fraction_day : ℝ := 5/7

theorem night_crew_worker_fraction
  (D N : ℝ) -- Number of workers in day and night crew
  (B : ℝ)  -- Number of boxes each worker in the day crew loads
  (H1 : ∀ day_boxes_loaded : ℝ, day_boxes_loaded = D * B)
  (H2 : ∀ night_boxes_loaded : ℝ, night_boxes_loaded = N * (B / 2))
  (H3 : (D * B) / ((D * B) + (N * (B / 2))) = box_fraction_day) :
  N / D = 4/5 := 
sorry

end night_crew_worker_fraction_l662_662680


namespace complex_exponential_sum_angle_l662_662349

theorem complex_exponential_sum_angle :
  ∃ r : ℝ, r ≥ 0 ∧ (e^(Complex.I * 11 * Real.pi / 60) + 
                     e^(Complex.I * 21 * Real.pi / 60) + 
                     e^(Complex.I * 31 * Real.pi / 60) + 
                     e^(Complex.I * 41 * Real.pi / 60) + 
                     e^(Complex.I * 51 * Real.pi / 60) = r * Complex.exp (Complex.I * 31 * Real.pi / 60)) := 
by
  sorry

end complex_exponential_sum_angle_l662_662349


namespace ceil_x_squared_values_count_l662_662810

open Real

theorem ceil_x_squared_values_count (x : ℝ) (h : ceil x = 15) : 
  ∃ n : ℕ, n = 29 ∧ ∃ a b : ℕ, a ≤ b ∧ (∀ (m : ℕ), a ≤ m ∧ m ≤ b → (ceil (x^2) = m)) := 
by
  sorry

end ceil_x_squared_values_count_l662_662810


namespace all_special_integers_l662_662142

def is_common_divisor (N d : ℕ) : Prop :=
  1 < d ∧ d < N ∧ N % d = 0

def is_special_integer (N : ℕ) : Prop :=
  ∃ d1 d2, is_common_divisor N d1 ∧ is_common_divisor N d2 ∧ d1 ≠ d2 ∧ 
  ∀ d1 d2 : ℕ, is_common_divisor N d1 → is_common_divisor N d2 → 
  N % (abs (d1 - d2)) = 0

theorem all_special_integers :
  ∀ N : ℕ, is_special_integer N ↔ (N = 6 ∨ N = 8 ∨ N = 12) := 
sorry

end all_special_integers_l662_662142


namespace average_difference_per_day_is_8_l662_662595

theorem average_difference_per_day_is_8 (differences : List ℤ) (h_len : differences.length = 7) (h_diffs : differences = [15, -5, 25, -10, 15, 5, 10]) : 
  (differences.sum / 7 : ℤ) = 8 := 
by
  have h_sum : differences.sum = 55 := by
    rw h_diffs
    norm_num
  exact (Int.ediv_eq_of_eq_mul_left (by norm_num : (7 : ℤ) ≠ 0) (by norm_num : 7 * 8 = 55)).mp h_sum.symm

end average_difference_per_day_is_8_l662_662595


namespace collinear_points_condition_l662_662938

theorem collinear_points_condition {z1 z2 z3 : ℂ} :
  (∃ (a b : ℂ), z2 = z1 + a * (z3 - z1) ∧ b = 1 - a ∧ 0 ≤ a ∧ a ≤ 1 ∧ 0 ≤ b ∧ b ≤ 1) ↔
  (∃ k : ℂ, has_real_part (z3 - z1) / (z2 - z1) = k ∨ (z2 - z1) = 0) ∨
  (real_part (of_real (conj z1 *  z2) + conj z2 *  z3 + conj z3 * z1) = 0 ∨ (z1 = z2 ∨ z2 = z3 ∨ z1 = z3)) :=
sorry

end collinear_points_condition_l662_662938


namespace find_number_l662_662294

theorem find_number (x : ℝ) (h : 3034 - x / 20.04 = 2984) : x = 1002 :=
sorry

end find_number_l662_662294


namespace ratio_of_areas_of_triangle_and_parallelogram_l662_662477

open Real function EuclideanGeometry

variables {A B C D M N : Point}
variable [EuclideanSpace ℝ (E : Type)]

-- Definitions and property conditions
def parallelogram (A B C D M N : Point) := parallelogram A B C D
def on_side (M : Point) (A B : Line) := PointOnSegment M AB
def AB_eq_3AM (A B M : Point) := dist A B = 3 * dist A M
def intersection_point (A D C M N : Line) := N = intersection AC DM

-- Proof problem: Given the conditions, find ratio of areas
theorem ratio_of_areas_of_triangle_and_parallelogram 
  (H1 : parallelogram A B C D) 
  (H2 : on_side M A B) 
  (H3 : AB_eq_3AM A B M) 
  (H4 : intersection_point A D C M N):
  area △ A M N / area (parallelogram A B C D) = 1 / 12 :=
sorry

end ratio_of_areas_of_triangle_and_parallelogram_l662_662477


namespace evaluate_expression_at_3_l662_662369

theorem evaluate_expression_at_3 : (3^3)^(3^3) = 27^27 := by
  sorry

end evaluate_expression_at_3_l662_662369


namespace pastries_sold_is_correct_l662_662682

-- Definitions of the conditions
def initial_pastries : ℕ := 56
def remaining_pastries : ℕ := 27

-- Statement of the theorem
theorem pastries_sold_is_correct : initial_pastries - remaining_pastries = 29 :=
by
  sorry

end pastries_sold_is_correct_l662_662682


namespace rockets_win_in_7_games_probability_l662_662961

theorem rockets_win_in_7_games_probability :
  (∃ p_b : ℚ, p_b = 3 / 5 ∧ 
  (∃ n : ℕ, n = 4 ∧ 
  (∃ t : ℕ, t = 7 ∧ 
  (∃ k : ℕ, k = 6 ∧ 
  (p_b * ((nat.choose k (t - n)) * ( ( (2 / 5 : ℚ) ^ (n - t)) * ((p_b) ^ (n - t)) ))) = 8640 / 78125 ))))) := 
begin
  sorry
end

end rockets_win_in_7_games_probability_l662_662961


namespace total_chairs_taken_l662_662518

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l662_662518


namespace trigonometric_identity_l662_662737

theorem trigonometric_identity (α : ℝ)
 (h : Real.sin (α / 2) - 2 * Real.cos (α / 2) = 1) :
  (1 + Real.sin α + Real.cos α) / (1 + Real.sin α - Real.cos α) = 3 / 4 := 
sorry

end trigonometric_identity_l662_662737


namespace sum_of_digits_correct_l662_662643

theorem sum_of_digits_correct :
  ∃ a b c : ℕ,
    (1 + 7 + 3 + a) % 9 = 0 ∧
    (1 + 3 - (7 + b)) % 11 = 0 ∧
    (c % 2 = 0) ∧
    ((1 + 7 + 3 + c) % 3 = 0) ∧
    (a + b + c = 19) :=
sorry

end sum_of_digits_correct_l662_662643


namespace M_subset_N_l662_662409

open Set

noncomputable def M : Set ℚ := {x | ∃ k : ℤ, x = k / 2 + 1 / 4}
noncomputable def N : Set ℚ := {x | ∃ k : ℤ, x = k / 4 + 1 / 2}

theorem M_subset_N : M ⊆ N := 
sorry

end M_subset_N_l662_662409


namespace transform_sin_wave_l662_662191

theorem transform_sin_wave (x : ℝ) (ω φ : ℝ) : 
  (∃ ω φ, (ω > 0) ∧ (|φ| < π / 2) ∧ 
         (∀ x, sin x = sin (ω * x + φ)) :=
∃ (ω = 1 / 2) (φ = -π / 6), 
   (ω > 0) ∧ (|φ| < π / 2) ∧ (∀ x, sin x = sin ((1 / 2) * x - π / 6)) := 
sorry

end transform_sin_wave_l662_662191


namespace staircase_perimeter_l662_662129

theorem staircase_perimeter (height : ℝ) (base : ℝ) (area_staircase : ℝ) (n_congruent_sides : ℕ) 
  (side_length : ℝ) 
  (h_total_area : base * height - n_congruent_sides * side_length^2 = area_staircase)
  (h_base : base = 10)
  (h_n_congruent_sides : n_congruent_sides = 12)
  (h_side_length : side_length = 1)
  : 10 + height + 12 * side_length - (n_congruent_sides - 1) * side_length = 19.4 :=
by 
  have h : height = 8.4 := by sorry
  simp only [h, h_side_length, h_n_congruent_sides]
  norm_num
  sorry

end staircase_perimeter_l662_662129


namespace at_least_one_not_less_than_100_l662_662909

-- Defining the original propositions
def p : Prop := ∀ (A_score : ℕ), A_score ≥ 100
def q : Prop := ∀ (B_score : ℕ), B_score < 100

-- Assertion to be proved in Lean
theorem at_least_one_not_less_than_100 (h1 : p) (h2 : q) : p ∨ ¬q := 
sorry

end at_least_one_not_less_than_100_l662_662909


namespace range_of_f_l662_662553

def g (x : ℝ) : ℝ := x^2 - 2

def f (x : ℝ) : ℝ :=
  if x < g x then g x + x + 4 else g x - x

theorem range_of_f : set.range f = {y : ℝ | y ∈ set.Icc (-2.25) 0 ∪ set.Ioo 2 ⊤} :=
by
  sorry

end range_of_f_l662_662553


namespace average_std_dev_qualified_prob_wang_qiang_l662_662993

variables {μ σ : ℝ} {students : ℕ}
variables {scores : ℕ → ℝ}
variables {groupA_scores : Fin 24 → ℝ}
variables {groupB_scores : Fin 16 → ℝ}
variables (μ σ students groupA_scores groupB_scores)

-- Given conditions
def groupA_mean : ℝ := 70
def groupB_mean : ℝ := 80
def groupA_std_dev : ℝ := 4
def groupB_std_dev : ℝ := 6
def groupA : Fin 24 → ℝ := groupA_scores
def groupB : Fin 16 → ℝ := groupB_scores

-- Definitions for group properties
def mean (scores : List ℝ) : ℝ := (scores.sum) / (scores.length)
def variance (scores : List ℝ) : ℝ := (scores.map (λ x, (x - mean scores) ^ 2)).sum / scores.length
def std_dev (scores : List ℝ) : ℕ := real.sqrt (variance scores)

-- Translate average and standard deviation calculation
theorem average_std_dev :
  mean ((List.ofFn groupA) ++ (List.ofFn groupB)) = 74 ∧
  std_dev ((List.ofFn groupA) ++ (List.ofFn groupB)) ≈ 7.75 :=
begin
  sorry
end

-- Normal distribution properties and qualification evaluation
def normal_qualification (cutoff : ℝ) (p_threshold : ℝ) : Prop :=
  P (λ x, x < cutoff) + P (λ x, x > 1000 - cutoff) < p_threshold

theorem qualified : normal_qualification 60 0.05 :=
begin
  sorry
end

-- Probability that Wang Qiang wins first 3 games and earns 3 points
variables {p_win p_lose : ℕ}

def matchup_probability (win : ℕ) (lose : ℕ) : ℝ :=
  (nat.choose (4 + lose - 1) 3 * ((2/3)^3) * ((1/3)^lose) * (2/3))

theorem prob_wang_qiang : matchup_probability 3 1 = (2/25) :=
begin
  sorry
end

end average_std_dev_qualified_prob_wang_qiang_l662_662993


namespace math_problem_correct_answers_l662_662675

theorem math_problem_correct_answers :
  (∀ k : ℝ, ¬ (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ x1 * log x1 - 1/2 * x1^2 + k = 0 ∧ x2 * log x2 - 1/2 * x2^2 + k = 0)) ∧
  (∀ a b c : ℝ, a^2 + b^2 = 2 * c^2 → ∀ C : ℝ, cos C = (a^2 + b^2 - c^2) / (2 * a * b) → C ≤ π / 3) ∧
  (¬ (∀ x : ℝ, (x ≠ n * π + π / 2) → (x ≠ n * π) → 1/2 * log ((1 - cos x) / (1 + cos x)) = log (tan (x / 2)))) ∧
  (∀ a b : ℝ, a > b → b > 0 → ∀ P : ℝ × ℝ, (P ≠ (-a, 0)) ∧ (P ≠ (a, 0)) → 
    let m := P.1, n := P.2 in
    (n / (m + a)) * (n / (m - a)) = -(b^2 / a^2)) :=
sorry

end math_problem_correct_answers_l662_662675


namespace interior_angles_sum_l662_662470

theorem interior_angles_sum (n : ℕ) (h : ∀ (k : ℕ), k = n → 60 * n = 360) : 
  180 * (n - 2) = 720 :=
by
  sorry

end interior_angles_sum_l662_662470


namespace proof_stops_with_two_pizzas_l662_662602

/-- The number of stops with orders of two pizzas. -/
def stops_with_two_pizzas : ℕ := 2

theorem proof_stops_with_two_pizzas
  (total_pizzas : ℕ)
  (single_stops : ℕ)
  (two_pizza_stops : ℕ)
  (average_time : ℕ)
  (total_time : ℕ)
  (h1 : total_pizzas = 12)
  (h2 : two_pizza_stops * 2 + single_stops = total_pizzas)
  (h3 : total_time = 40)
  (h4 : average_time = 4)
  (h5 : two_pizza_stops + single_stops = total_time / average_time) :
  two_pizza_stops = stops_with_two_pizzas := 
sorry

end proof_stops_with_two_pizzas_l662_662602


namespace square_segments_inequality_l662_662926

-- Define the structure of a square
structure Square (V : Type*) :=
(A B C D : V)
(is_square : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A ) -- Simplified for illustration

-- Define the points K, L, M, N on the sides of the square
structure PointsOnSquare (V : Type*) (sq : Square V) :=
(K L M N : V)
(on_sides : true) -- Simplified for illustration

-- Define the center of the square
def center (V : Type*) {sq : Square V} (h : sq) : V := 
  sorry

-- Define projection properties
def projection_properties (V : Type*) [linear_order V] 
  {sq : Square V} (O : V) (X : V) : Prop :=
  let AC := sorry in  -- diagonal length, simplified
  let BD := sorry in  -- diagonal length, simplified
  sorry

-- Theorem statement
theorem square_segments_inequality (V : Type*) [linear_order V]
  (sq : Square V) (points : PointsOnSquare V sq) 
  (O : V) (hO : O = center sq)
  (proj_prop : ∀ X : V, on_boundary X sq → projection_properties O X) :
  KL + LM + MN + NK ≥ 2 * AC :=
sorry

end square_segments_inequality_l662_662926


namespace range_of_a_l662_662769

def sum_part (n : ℕ) : ℝ := 
  ∑ i in range n, (1 : ℝ) / (i * (i + 1))

theorem range_of_a (a : ℝ) (h : ∀ n : ℕ, sum_part n > real.log (a - 1) / real.log 2 + a - 7 / 2) : 
  1 < a ∧ a < 3 :=
sorry

end range_of_a_l662_662769


namespace area_of_quadrilateral_STUV_l662_662123

-- Define the rectangle $JKLM$
structure Rectangle (J K L M : ℝ × ℝ) :=
(JK : dist J K = 6)
(JL : dist J L = 4)
(perpendicular : ∀ {A B C : ℝ × ℝ}, dist A B = dist B C → dist A C = √((dist A B)^2 + (dist B C)^2))

-- Define midpoints $N$, $P$, $Q$, and $R$
def midpoint (A B : ℝ × ℝ) : ℝ × ℝ := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

-- Given points J, K, L, M
variables (J K L M : ℝ × ℝ)

-- Define bisected points
def N := midpoint J K
def P := midpoint K J   -- same as N
def Q := midpoint L M
def R := midpoint M L   -- same as Q

-- Define midpoints S, T, U, V of the formed quadrilateral
def S := midpoint J N
def T := midpoint K P
def U := midpoint L Q
def V := midpoint M R

-- Proof problem: area of the quadrilateral
theorem area_of_quadrilateral_STUV (hJ_def : J = (0, 4)) (hK_def : K = (6, 4)) (hL_def : L = (0, 0)) (hM_def : M = (6, 0)) :
  let S := midpoint J N,
      T := midpoint K P,
      U := midpoint L Q,
      V := midpoint M R,
      parallelogram := rectangle J K L M in
  parallelogram.area = 6 := sorry

end area_of_quadrilateral_STUV_l662_662123


namespace mean_median_difference_l662_662238

def rolls := [150, 230, 160, 190, 210, 180]

noncomputable def mean (l : List ℕ) : ℝ :=
  (l.sum : ℝ) / l.length

noncomputable def median (l : List ℕ) : ℝ :=
  let sorted := l.qsort (· < ·)
  if sorted.length % 2 = 0 then
    let mid := sorted.length / 2
    (sorted.get! (mid - 1) + sorted.get! mid : ℝ) / 2
  else
    sorted.get! (sorted.length / 2)

theorem mean_median_difference : abs (mean rolls - median rolls) = 1.67 := sorry

end mean_median_difference_l662_662238


namespace parabola_focus_distance_l662_662426

theorem parabola_focus_distance (p : ℝ) 
  (hp : p > 0) 
  (min_dist : ∀ (x y : ℝ), y^2 = 2 * p * x → min_dist_focus : (x - 2)^2 + y^2 = 4) 
  (l : ℝ → Prop) 
  (hl : l (0, 1) ∧ ∀ (x y : ℝ), y^2 = 2 * p * x → (∃! y, l (x, y))) :
  ∃ d ∈ (∅insert 1 (∅insert 2 (∅insert (sqrt 5) ∅))), distance focus l = d :=
by
  sorry

end parabola_focus_distance_l662_662426


namespace sequence_inequality_l662_662291

variable (a : ℕ → ℕ) (b : ℕ → ℕ)
variable (x y : ℕ → ℕ)

/-- Given that a is a non-decreasing, unbounded sequence of non-negative integers, 
    and b n denotes the number of terms in the sequence that do not exceed n,
    prove that (sum x 0 m) * (sum y 0 n) ≥ (m + 1) * (n + 1),
    where x_i is the number of occurrences of i in the sequence a. -/
theorem sequence_inequality 
  (H1 : ∀ i j, i ≤ j → a i ≤ a j)
  (H2 : ∀ n, ∃ m, a m > n) 
  (H3 : ∀ n, b n = ∑ k in finset.range n.succ, if a k ≤ n then 1 else 0) 
  (H4 : ∀ i, x i = ∑ k in finset.range i.succ, if a k = i then 1 else 0)
  (H5 : ∀ j, y j = b j.succ - b j) :
  (finset.range m.succ).sum x * (finset.range n.succ).sum y ≥ (m + 1) * (n + 1) :=
sorry

end sequence_inequality_l662_662291


namespace convex_hexagon_area_triangle_convex_octagon_area_triangle_l662_662281

-- Definitions and statements for the given problems.

-- 1. Hexagon problem statement
theorem convex_hexagon_area_triangle (S : ℝ) (h : hexagon) (hS : h.area = S) : 
  ∃ diag : h.diagonal, ∃ tri : h.triangle diag, tri.area ≤ S / 6 :=
sorry

-- 2. Octagon problem statement
theorem convex_octagon_area_triangle (S : ℝ) (o : octagon) (oS : o.area = S) : 
  ∃ diag : o.diagonal, ∃ tri : o.triangle diag, tri.area ≤ S / 8 :=
sorry

end convex_hexagon_area_triangle_convex_octagon_area_triangle_l662_662281


namespace magnitude_eq_sqrt_13_l662_662725

theorem magnitude_eq_sqrt_13 (t : ℝ) (ht : |Complex.mk (-4) t| = 2 * Real.sqrt 13) : t = 6 := by
  sorry

end magnitude_eq_sqrt_13_l662_662725


namespace side_of_rhombus_l662_662324

variable (d : ℝ) (K : ℝ) 

-- Conditions
def shorter_diagonal := d
def longer_diagonal := 3 * d
def area_rhombus := K = (1 / 2) * d * (3 * d)

-- Proof Statement
theorem side_of_rhombus (h1 : K = (3 / 2) * d^2) : (∃ s : ℝ, s = Real.sqrt (5 * K / 3)) := 
  sorry

end side_of_rhombus_l662_662324


namespace problem1_problem2_l662_662435

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := m * x - m / x
noncomputable def g (x : ℝ) : ℝ := 3 * Real.log x

theorem problem1 :
  let m := 4
  let tangent_line (x : ℝ) : ℝ := 5 * x - 4
  tangent_line 2 = f m 2 :=
by 
  let m := 4
  let f := f m
  let tangent_line (x : ℝ) := 5 * x - 4
  have : f 2 = 6 := sorry
  show tangent_line 2 = 6 from sorry

theorem problem2 :
  (∀ x : ℝ, 1 < x ∧ x ≤ Real.sqrt Real.exp → f m x - g x < 3) → 
    m < (9 * Real.sqrt Real.exp / (2 * (Real.exp - 1))) :=
by 
  intro h
  apply sorry

end problem1_problem2_l662_662435


namespace museum_revenue_from_college_students_l662_662858

/-!
In one day, 200 people visit The Metropolitan Museum of Art in New York City. Half of the visitors are residents of New York City. 
Of the NYC residents, 30% are college students. If the cost of a college student ticket is $4, we need to prove that 
the museum gets $120 from college students that are residents of NYC.
-/

theorem museum_revenue_from_college_students :
  let total_visitors := 200
  let residents_nyc := total_visitors / 2
  let college_students_percentage := 30 / 100
  let college_students := residents_nyc * college_students_percentage
  let ticket_cost := 4
  residents_nyc = 100 ∧ 
  college_students = 30 ∧ 
  ticket_cost * college_students = 120 := 
by
  sorry

end museum_revenue_from_college_students_l662_662858


namespace min_reciprocal_sum_l662_662907

theorem min_reciprocal_sum (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 3) : 
  1 / x + 1 / y + 1 / z ≥ 9 :=
begin
  sorry
end

end min_reciprocal_sum_l662_662907


namespace coloring_exists_l662_662908

section ProofProblem

-- Define the set S and the condition that no three points are collinear
variable {α : Type} [RealPlane α]
variable (S : Finset α) 
  (hS : S.card = 2004) 
  (no_three_collinear : ∀ (p q r : α), p ∈ S → q ∈ S → r ∈ S → p ≠ q → q ≠ r → r ≠ p → ¬ collinear p q r)

-- Define the set of all lines determined by pairs of points from S
def L (S : Finset α) : Finset (Line α) :=
  S.pairs Image (λ (p q : α), Line.mk p q)

-- Define the function n(q, r) as the number of lines separating q from r
def n (q r : α) : ℕ :=
  (L S).card (λ ℓ, separates ℓ q r)

-- Prove the existence of at most two-color coloring satisfying the given conditions
theorem coloring_exists :
  ∃ (coloring : S → Fin 2), 
    ∀ {p q : α} (hp : p ∈ S) (hq : q ∈ S), 
    (n p q).odd ↔ coloring p = coloring q :=
begin
  sorry
end

end ProofProblem

end coloring_exists_l662_662908


namespace pq_eqv_l662_662641

theorem pq_eqv (p q : Prop) : 
  ((¬ p ∧ ¬ q) ∧ (p ∨ q)) ↔ ((p ∧ ¬ q) ∨ (¬ p ∧ q)) :=
by
  sorry

end pq_eqv_l662_662641


namespace martin_speed_l662_662559

theorem martin_speed (distance : ℝ) (time : ℝ) (h₁ : distance = 12) (h₂ : time = 6) : (distance / time = 2) :=
by 
  -- Note: The proof is not required as per instructions, so we use 'sorry'
  sorry

end martin_speed_l662_662559


namespace jeremy_school_distance_l662_662508

theorem jeremy_school_distance (d : ℝ) (v : ℝ) :
  (d = v * 0.5) ∧
  (d = (v + 15) * 0.3) ∧
  (d = (v - 10) * (2 / 3)) →
  d = 15 :=
by 
  sorry

end jeremy_school_distance_l662_662508


namespace number_of_people_got_off_at_third_stop_l662_662587

-- Definitions for each stop
def initial_passengers : ℕ := 0
def passengers_after_first_stop : ℕ := initial_passengers + 7
def passengers_after_second_stop : ℕ := passengers_after_first_stop - 3 + 5
def passengers_after_third_stop (x : ℕ) : ℕ := passengers_after_second_stop - x + 4

-- Final condition stating there are 11 passengers after the third stop
def final_passengers : ℕ := 11

-- Proof goal
theorem number_of_people_got_off_at_third_stop (x : ℕ) :
  passengers_after_third_stop x = final_passengers → x = 2 :=
by
  -- proof goes here
  sorry

end number_of_people_got_off_at_third_stop_l662_662587


namespace fraction_min_sum_l662_662900

theorem fraction_min_sum (a b : ℕ) (h_pos : 0 < a ∧ 0 < b) 
  (h_ineq : 45 * b < 110 * a ∧ 110 * a < 50 * b) :
  a = 3 ∧ b = 7 :=
sorry

end fraction_min_sum_l662_662900


namespace gumball_count_to_ensure_five_same_color_l662_662308

theorem gumball_count_to_ensure_five_same_color : 
  ∀ (red white blue green : ℕ), 
    red = 10 → white = 8 → blue = 9 → green = 7 → 
    (∃ n, n = 17 ∧ 
          ∀ (picked : ℕ → ℕ), 
            (picked red ≤ 10 ∧ picked white ≤ 8 ∧ picked blue ≤ 9 ∧ picked green ≤ 7) → 
            (∀ c, picked c ≤ 4 → ∃ c', picked c' ≥ 5)) :=
by
  intros red white blue green h_red h_white h_blue h_green,
  use 17,
  split,
  { refl },
  { intros picked h_picked h_count,
    sorry }

end gumball_count_to_ensure_five_same_color_l662_662308


namespace midpoint_trajectory_l662_662998

theorem midpoint_trajectory (x y : ℝ) (x0 y0 : ℝ)
  (h_circle : x0^2 + y0^2 = 4)
  (h_tangent : x0 * x + y0 * y = 4)
  (h_x0 : x0 = 2 / x)
  (h_y0 : y0 = 2 / y) :
  x^2 * y^2 = x^2 + y^2 :=
sorry

end midpoint_trajectory_l662_662998


namespace digits_product_impossible_l662_662982

theorem digits_product_impossible (N : ℕ) :
  (∀ d ∈ digits 10 N, d ≠ 0) →
  (list.prod (digits 10 N) = 20) →
  ¬(list.prod (digits 10 (N+1)) = 35) :=
by
  sorry

end digits_product_impossible_l662_662982


namespace number_of_satisfying_a1_l662_662893

theorem number_of_satisfying_a1 (n : ℕ) :
  (λ m, m ≤ 3000 ∧ ∃ k, m = 4 * k + 3).count (λ m, True) = 750 :=
by sorry

end number_of_satisfying_a1_l662_662893


namespace problem_solution_l662_662696

noncomputable def is_solution (a b : ℤ) :=
  let F : ℕ → ℤ := λ n, if n = 0 then 0
                         else if n == 1 then 1
                         else F (n - 1) + F (n - 2) in
  (2584 * a + 1597 * b = 0) ∧ (1597 * a + 987 * b = -1)

theorem problem_solution : ∃ a b : ℤ, is_solution a b ∧ a = -987 ∧ b = 2584 :=
by
  use [-987, 2584]
  unfold is_solution
  split
  all_goals { sorry }

end problem_solution_l662_662696


namespace cylinder_volume_is_correct_l662_662436

-- Define the sphere diameter and the cylinder's height
def sphere_diameter := 2
def cylinder_height := 1

-- Define the radius of the sphere
def sphere_radius := sphere_diameter / 2

-- Define the radius of the cylinder base given the radius of the sphere
def cylinder_base_radius := real.sqrt (cylinder_height^2 - (sphere_radius^2) / 4)

-- Define the volume of the cylinder
def cylinder_volume := π * cylinder_base_radius^2 * cylinder_height

-- Prove the calculated volume equals the given value
theorem cylinder_volume_is_correct : cylinder_volume = (3 * π) / 4 := by
  sorry

end cylinder_volume_is_correct_l662_662436


namespace Borgnine_total_legs_l662_662683

def numChimps := 12
def numLions := 8
def numLizards := 5
def numTarantulas := 125

def chimpLegsEach := 2
def lionLegsEach := 4
def lizardLegsEach := 4
def tarantulaLegsEach := 8

def legsSeen := numChimps * chimpLegsEach +
                numLions * lionLegsEach +
                numLizards * lizardLegsEach

def legsToSee := numTarantulas * tarantulaLegsEach

def totalLegs := legsSeen + legsToSee

theorem Borgnine_total_legs : totalLegs = 1076 := by
  sorry

end Borgnine_total_legs_l662_662683


namespace mars_moon_cost_share_l662_662196

theorem mars_moon_cost_share :
  let total_cost := 40 * 10^9 -- total cost in dollars
  let num_people := 200 * 10^6 -- number of people sharing the cost
  (total_cost / num_people) = 200 := by
  sorry

end mars_moon_cost_share_l662_662196


namespace midpoints_on_circumcircle_l662_662583

theorem midpoints_on_circumcircle (A B C : Point) (hA_col : collinear {A, B, C})
  (hB_perp : perpendicular (altitude B A C) (side B A))
  (hC_perp : perpendicular (altitude C A B) (side C A))
  (hBisectorA_B : intersects (angle_bisector A) (altitude B A C) B₁)
  (hBisectorA_C : intersects (angle_bisector A) (altitude C A B) C₁)
  (hMid_B₁C₁_perp : perpendicular_bisector B₁ C₁ A₀)
  (hMid_B₁C₁_external : perpendicular_bisector B₁' C₁' A₀')
  (hCirc : circumcircle A B C) :
  on_circumcircle A₀ (circumcircle A B C) ∧ on_circumcircle A₀' (circumcircle A B C) :=
sorry

end midpoints_on_circumcircle_l662_662583


namespace bill_grass_stains_l662_662346

-- Definition of conditions
def soaking_time (G : ℕ) : ℕ := 4 * G + 7

-- Main theorem
theorem bill_grass_stains (G : ℕ) : soaking_time G = 19 → G = 3 :=
begin
  sorry
end

end bill_grass_stains_l662_662346


namespace focus_parabola_4x2_y_distance_point_to_plane_hyperbola_equation_cos_alpha_beta_ratio_l662_662288

-- Proof 1.
theorem focus_parabola_4x2_y : (0, 1 / 16) = focus (4 * x^2 = y) :=
sorry

-- Proof 2.
theorem distance_point_to_plane : sqrt 14 = distance_point_plane (0, 1, 3) (plane_eq 1 2 3 3) :=
sorry

-- Proof 3.
theorem hyperbola_equation : x ^ 2 / 4 - y ^ 2 / 2 = 1 = equation_hyperbola_asymptotes_passing_point (2, 0) (x^2 / 2 - y^2 = 1) :=
sorry

-- Proof 4.
theorem cos_alpha_beta_ratio : 7 = cos_alpha_plus_beta_div_cos_alpha_minus_beta ellipse_properties given_inclinations :=
sorry

end focus_parabola_4x2_y_distance_point_to_plane_hyperbola_equation_cos_alpha_beta_ratio_l662_662288


namespace ceil_square_count_ceil_x_eq_15_l662_662805

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l662_662805


namespace largest_constant_inequality_l662_662012

theorem largest_constant_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) : 
  ∃ (m : ℝ), (∀ a b c d : ℝ, a > 0 → b > 0 → c > 0 → d > 0 →
  sqrt (a / (b + c + d)) + sqrt (b / (a + c + d)) + sqrt (c / (a + b + d)) + sqrt (d / (a + b + c)) > m) ∧ m = 2 :=
by
  use 2
  sorry

end largest_constant_inequality_l662_662012


namespace Pradeep_marks_l662_662932

variable (T : ℕ) (P : ℕ) (F : ℕ)

def passing_marks := P * T / 100

theorem Pradeep_marks (hT : T = 925) (hP : P = 20) (hF : F = 25) :
  (passing_marks P T) - F = 160 :=
by
  sorry

end Pradeep_marks_l662_662932


namespace irrational_trio_l662_662241

theorem irrational_trio : ∀ (a b c d e f : ℝ), 
  (irrational a) ∧ (irrational b) ∧ (irrational c) ∧ (irrational d) ∧ (irrational e) ∧ (irrational f) →
  ∃ (x y z : ℝ), (x + y) + (y + z) + (z + x) ∧ irrational x ∧ irrational y ∧ irrational z := 
begin
  sorry
end

end irrational_trio_l662_662241


namespace probability_al_multiple_of_bill_bill_multiple_of_cal_even_sum_l662_662336

theorem probability_al_multiple_of_bill_bill_multiple_of_cal_even_sum :
  let nums := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12} in
  finite (nums \times nums \times nums) → 
  ∀ (A B C : ℕ) (hA : A ∈ nums) (hB : B ∈ nums) (hC : C ∈ nums),
  ¬(A = B ∨ B = C ∨ A = C) → 
  A % B = 0 → 
  B % C = 0 → 
  (A + B + C) % 2 = 0 → 
  let valid_assignments := (filter (λ (abc : ℕ × ℕ × ℕ), 
    let (a, b, c) := abc in
    a % b = 0 ∧ b % c = 0 ∧ (a + b + c) % 2 = 0) (nums × nums × nums)).to_finset in
  (valid_assignments.card : ℚ) / (12 * 11 * 10 : ℚ) = 2 / 110 :=
sorry

end probability_al_multiple_of_bill_bill_multiple_of_cal_even_sum_l662_662336


namespace no_real_roots_for_quadratic_l662_662988

theorem no_real_roots_for_quadratic :
  let a := 2
  let b := -5
  let c := 6
  let Δ : ℝ := b^2 - 4 * a * c
  Δ < 0 := 
by {
  -- Definition of Δ
  let Δ := b^2 - 4 * a * c,
  -- Calculating Δ
  calc Δ = (-5)^2 - 4 * 2 * 6 : by sorry -- 25 - 48
     ... = 25 - 48 : by sorry  -- simplification
     ... = -23 : by sorry, -- final result
  -- Establishing Δ < 0
  show Δ < 0, from sorry
}

end no_real_roots_for_quadratic_l662_662988


namespace triangle_area_l662_662755

theorem triangle_area (a b c : ℝ) (h₁ : a = 5) (h₂ : b = 12) (h₃ : c = 13) (h₄ : a * a + b * b = c * c) :
  (1/2) * a * b = 30 :=
by
  sorry

end triangle_area_l662_662755


namespace percentage_decrease_in_length_l662_662668

variables (L B : ℝ)

-- Original area
def A : ℝ := L * B

-- Conditions
def B' : ℝ := 0.8 * B -- New breadth after 20% reduction
def A' : ℝ := 0.72 * A L B -- New area after 28% reduction

-- New length to be proven:
def L' : ℝ := 0.9 * L

theorem percentage_decrease_in_length (h1 : B' B = 0.8 * B)
  (h2 : A' (A L B) = 0.72 * (A L B)) : L' L = 0.9 * L :=
by sorry

end percentage_decrease_in_length_l662_662668


namespace solve_complex_problem_l662_662395

-- Define the problem
def complex_sum_eq_two (a b : ℝ) (i : ℂ) : Prop :=
  a + b = 2

-- Define the conditions
def conditions (a b : ℝ) (i : ℂ) : Prop :=
  a + b * i = (1 - i) * (2 + i)

-- State the theorem
theorem solve_complex_problem (a b : ℝ) (i : ℂ) (h : conditions a b i) : complex_sum_eq_two a b i :=
by
  sorry -- Proof goes here

end solve_complex_problem_l662_662395


namespace problem_statement_l662_662934

variable {S R p a b c : ℝ}
variable (τ τ_a τ_b τ_c : ℝ)

theorem problem_statement
  (h1: S = τ * p)
  (h2: S = τ_a * (p - a))
  (h3: S = τ_b * (p - b))
  (h4: S = τ_c * (p - c))
  (h5: τ = S / p)
  (h6: τ_a = S / (p - a))
  (h7: τ_b = S / (p - b))
  (h8: τ_c = S / (p - c))
  (h9: abc / S = 4 * R) :
  1 / τ^3 - 1 / τ_a^3 - 1 / τ_b^3 - 1 / τ_c^3 = 12 * R / S^2 :=
  sorry

end problem_statement_l662_662934


namespace average_of_first_45_results_l662_662201

theorem average_of_first_45_results
  (A : ℝ)
  (h1 : (45 + 25 : ℝ) = 70)
  (h2 : (25 : ℝ) * 45 = 1125)
  (h3 : (70 : ℝ) * 32.142857142857146 = 2250)
  (h4 : ∀ x y z : ℝ, 45 * x + y = z → x = 25) :
  A = 25 :=
by
  sorry

end average_of_first_45_results_l662_662201


namespace triangle_AB_length_l662_662716

theorem triangle_AB_length (AC BC : ℝ) (hAC : AC = 8 * Real.sqrt 2) (hBC : BC = 8 * Real.sqrt 2) (angle_C : ℝ = 45) :
  ∃ (AB : ℝ), AB = 16 := by
    sorry

end triangle_AB_length_l662_662716


namespace part1_part2_l662_662872

-- Definitions from the conditions:
variables {A B C D : Type*} [EuclideanSpace ℝ A] [EuclideanSpace ℝ B] [EuclideanSpace ℝ C] [EuclideanSpace ℝ D]
variables {BD CD AC BC AB AD : ℝ}
variable (α : ℝ) -- α is ∠BAC
variable (β : ℝ) -- β is ∠DAC

-- Given conditions
hypothesis h1 : BD = 2 * CD
hypothesis h2 : α + β = π

-- Proof to be provided
theorem part1 : AB = 3 * AD :=
sorry

-- Additional condition for part 2
hypothesis h3 : BC = 3 * AC

-- Proof to be provided
theorem part2 : cos α = - (sqrt 6) / 6 :=
sorry

end part1_part2_l662_662872


namespace least_possible_area_of_square_l662_662230

theorem least_possible_area_of_square (s : ℝ) (h₁ : 4.5 ≤ s) (h₂ : s < 5.5) : 
  s * s ≥ 20.25 :=
sorry

end least_possible_area_of_square_l662_662230


namespace last_day_of_week_is_wednesday_l662_662199

/-- A definition indicating that the 24th of the month is a Wednesday -/
def day_of_24th : DayOfWeek := DayOfWeek.wednesday

/-- A definition indicating the number of days in the month -/
def days_in_month : ℕ := 31

/-- The function that determines the day of the week given a starting day and a number of days -/
def day_of_week_n_days_later (start_day : DayOfWeek) (n : ℕ) : DayOfWeek :=
  (start_day.toNat + n) % 7

/-- The main theorem stating the last day of the month is a Wednesday -/
theorem last_day_of_week_is_wednesday :
  day_of_week_n_days_later day_of_24th (days_in_month - 24) = DayOfWeek.wednesday :=
by
  sorry

end last_day_of_week_is_wednesday_l662_662199


namespace total_chairs_taken_l662_662519

def num_students : ℕ := 5
def chairs_per_trip : ℕ := 5
def num_trips : ℕ := 10

theorem total_chairs_taken :
  (num_students * chairs_per_trip * num_trips) = 250 :=
by
  sorry

end total_chairs_taken_l662_662519


namespace breadth_of_rectangular_plot_l662_662962

theorem breadth_of_rectangular_plot (b : ℝ) (A : ℝ) (l : ℝ)
  (h1 : A = 20 * b)
  (h2 : l = b + 10)
  (h3 : A = l * b) : b = 10 := by
  sorry

end breadth_of_rectangular_plot_l662_662962


namespace probability_first_hearts_second_ace_correct_l662_662249

noncomputable def probability_first_hearts_second_ace : ℚ :=
  let total_cards := 104
  let total_aces := 8 -- 4 aces per deck, 2 decks
  let hearts_count := 2 * 13 -- 13 hearts per deck, 2 decks
  let ace_of_hearts_count := 2

  -- Case 1: the first is an ace of hearts
  let prob_first_ace_of_hearts := (ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_ace_of_hearts := (total_aces - 1 : ℚ) / (total_cards - 1)

  -- Case 2: the first is a hearts but not an ace
  let prob_first_hearts_not_ace := (hearts_count - ace_of_hearts_count : ℚ) / total_cards
  let prob_second_ace_given_first_hearts_not_ace := total_aces / (total_cards - 1)

  -- Combined probability
  (prob_first_ace_of_hearts * prob_second_ace_given_first_ace_of_hearts) +
  (prob_first_hearts_not_ace * prob_second_ace_given_first_hearts_not_ace)

theorem probability_first_hearts_second_ace_correct : 
  probability_first_hearts_second_ace = 7 / 453 := 
sorry

end probability_first_hearts_second_ace_correct_l662_662249


namespace construct_plane_with_dihedral_angle_l662_662618

variables {Point : Type} {Line : Type} {Plane : Type} [HasIntersection Point Line] [HasIntersection Point Plane]

def is_dihedral_angle (β : Plane) (α : Plane) (ϕ : ℝ) : Prop :=
  ∃ P M P' : Point, P ∈ β ∧ P ∈ α ∧ 
                    M ∈ β ∧ M ∉ α ∧ 
                    P' ∈ α ∧ P' ≠ P ∧ 
                    angle P M P' = ϕ

theorem construct_plane_with_dihedral_angle 
  (l : Line) (α : Plane) (ϕ : ℝ) (h_angle : ϕ < π / 2) : 
  ∃ β₁ β₂ : Plane, β₁ ≠ β₂ ∧ is_dihedral_angle β₁ α ϕ ∧ is_dihedral_angle β₂ α ϕ :=
begin
  sorry
end

end construct_plane_with_dihedral_angle_l662_662618


namespace min_length_other_sides_l662_662977

theorem min_length_other_sides {a b c d : ℝ} (h₁ : a = 1) (h₂ : c = 1)
    (h₃ : a * b + a * d + c * b + c * d ≥ 5)
    (h₄ : b ≥ 2) :
    ∃ (x y : ℝ), x = 3 + 2*Real.sqrt 2 ∧ a * b + a * d + c * b + c * d = x :=
begin
  sorry
end

end min_length_other_sides_l662_662977


namespace integral_abs_ln_sin_le_pi_cubed_over_12_l662_662566

noncomputable def integral_of_abs_ln_sin := ∫ x in 0..π, |Real.log (Real.sin x)|

theorem integral_abs_ln_sin_le_pi_cubed_over_12 : integral_of_abs_ln_sin ≤ π^3 / 12 := sorry

end integral_abs_ln_sin_le_pi_cubed_over_12_l662_662566


namespace find_train_length_l662_662332

def train_speed_kmph : ℝ := 60
def time_to_cross_seconds : ℝ := 18.598512119030477
def bridge_length_meters : ℝ := 200

def train_length_meters : ℝ := 110.000201951774

theorem find_train_length :
  let speed_mps := train_speed_kmph * 1000 / 3600,
      total_distance := speed_mps * time_to_cross_seconds
  in total_distance - bridge_length_meters = train_length_meters :=
by
  -- Proof omitted
  sorry

end find_train_length_l662_662332


namespace ceil_x_squared_values_count_l662_662812

open Real

theorem ceil_x_squared_values_count (x : ℝ) (h : ceil x = 15) : 
  ∃ n : ℕ, n = 29 ∧ ∃ a b : ℕ, a ≤ b ∧ (∀ (m : ℕ), a ≤ m ∧ m ≤ b → (ceil (x^2) = m)) := 
by
  sorry

end ceil_x_squared_values_count_l662_662812


namespace triangle_proof_l662_662109

variable (A : ℝ) (a b c d : ℝ)
variable (triangle_ABC : Type*)
variable (D : triangle_ABC)
variable [BC DC : Prop] [AD : Prop]

theorem triangle_proof : BC = DC → AD = d → 
                         c + d = 2 * b * cos A ∧ 
                         c * d = b^2 - a^2 := by
  sorry

end triangle_proof_l662_662109


namespace smallest_ratio_AB_CD_l662_662548

theorem smallest_ratio_AB_CD 
  (A B C D K : Point)
  (h1 : cyclic A B C D) 
  (h2 : K ∈ line AB)
  (h3 : bisects (line BD) (segment KC))
  (h4 : bisects (line AC) (segment KD))
  : (|AB / CD|) ≥ 2 :=
sorry

end smallest_ratio_AB_CD_l662_662548


namespace tetrahedron_edge_length_l662_662989

theorem tetrahedron_edge_length (a b c d e f rs pq : ℝ)
  (ha : a = 8) (hb : b = 14) (hc : c = 19) (hd : d = 28) (he : e = 37) (hf : f = 42)
  (hpq : pq = 42) :
  rs = 19 :=
by
  -- Proof here
  sorry

-- Definitions of the given edges to aid in interpretation
noncomputable def PQR_edges := {8, 14, 19, 28, 37, 42}
noncomputable def PQ := 42

end tetrahedron_edge_length_l662_662989


namespace taro_vlad_played_30_rounds_l662_662197

variables (total_points : ℝ) (taro_points : ℝ) (vlad_points : ℝ)

def vlad_points_equals_64 (v : ℝ) : Prop := v = 64
def taro_points_formula (t p : ℝ) : Prop := t = (3 / 5) * p - 4
def total_points_formula (t v p : ℝ) : Prop := p = t + v

def number_of_rounds (p : ℝ) (points_per_win : ℝ) : ℝ := p / points_per_win

theorem taro_vlad_played_30_rounds 
    (points_per_win : ℝ) 
    (H_points_win : points_per_win = 5)
    (H_vlad : vlad_points_equals_64 vlad_points)
    (H_taro : taro_points_formula taro_points total_points)
    (H_total : total_points_formula taro_points vlad_points total_points) :
  number_of_rounds total_points points_per_win = 30 := 
by 
  sorry

end taro_vlad_played_30_rounds_l662_662197


namespace modulus_complex_number_l662_662822

-- Define the conditions for the problem
variables {i a b : ℝ}

-- Define the imaginary unit i and the complex condition
def imaginary_unit := Complex.I
def condition := (a - 2 * imaginary_unit) * imaginary_unit = b - imaginary_unit

-- Main theorem statement to prove
theorem modulus_complex_number (ha : a = -1) (hb : b = 2) (h_condition : condition) :
  Complex.abs (Complex.mk a b) = Real.sqrt 5 :=
by 
  sorry

end modulus_complex_number_l662_662822


namespace angle_at_3_15_l662_662786

theorem angle_at_3_15 : 
  let minute_hand_angle := 90.0
  let hour_at_3 := 90.0
  let hour_hand_at_3_15 := hour_at_3 + 7.5
  minute_hand_angle == 90.0 ∧ hour_hand_at_3_15 == 97.5 →
  abs (hour_hand_at_3_15 - minute_hand_angle) = 7.5 :=
by
  sorry

end angle_at_3_15_l662_662786


namespace largest_number_with_sum_12_l662_662258

/-
  Prove that the largest number, all of whose digits are 1 or 2,
  and whose digits add up to 12, is 222222.
-/

theorem largest_number_with_sum_12
  (n : ℕ)
  (digits : list ℕ)
  (h1 : ∀ (d : ℕ), d ∈ digits → d = 1 ∨ d = 2)
  (h2 : digits.sum = 12) :
  digits = [2, 2, 2, 2, 2, 2] :=
sorry

end largest_number_with_sum_12_l662_662258


namespace ptolemy_theorem_l662_662836

noncomputable def is_triangle (A B C : Point) : Prop :=
  collinear A B C = false

theorem ptolemy_theorem
  (A B C P D E F : Point)
  (h_triangle : is_triangle A B C)
  (h_largest_angle_B : ∃ X, X ∈ segment A B)
  (h_smallest_angle_C : ∃ Y, Y ∈ segment A B)
  (h_pd_pe_relation : (∃ P, P D + P E = PA + PB + PC + PF))
  : PD + PE = PA + PB + PC + PF :=
by
  sorry

end ptolemy_theorem_l662_662836


namespace solve_sys_eqns_l662_662953

def sys_eqns_solution (x y : ℝ) : Prop :=
  y^2 = x^3 - 3*x^2 + 2*x ∧ x^2 = y^3 - 3*y^2 + 2*y

theorem solve_sys_eqns :
  ∃ (x y : ℝ),
  (sys_eqns_solution x y ∧
  ((x = 0 ∧ y = 0) ∨
  (x = 2 - Real.sqrt 2 ∧ y = 2 - Real.sqrt 2) ∨
  (x = 2 + Real.sqrt 2 ∧ y = 2 + Real.sqrt 2))) :=
by
  sorry

end solve_sys_eqns_l662_662953


namespace angle_between_hands_at_3_15_l662_662781

theorem angle_between_hands_at_3_15 :
  let hour_angle_at_3 := 3 * 30
  let hour_hand_move_rate := 0.5
  let minute_angle := 15 * 6
  let hour_angle_at_3_15 := hour_angle_at_3 + 15 * hour_hand_move_rate
  abs (hour_angle_at_3_15 - minute_angle) = 7.5 := 
by
  sorry

end angle_between_hands_at_3_15_l662_662781


namespace prod_ineq_min_value_l662_662151

theorem prod_ineq_min_value (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a^2 + 4*a + 1) * (b^2 + 4*b + 1) * (c^2 + 4*c + 1) / (a * b * c) ≥ 216 := by
  sorry

end prod_ineq_min_value_l662_662151


namespace evaluate_expression_l662_662706

theorem evaluate_expression (b : ℕ) (h : b = 4) : (b ^ b - b * (b - 1) ^ b) ^ b = 21381376 := by
  sorry

end evaluate_expression_l662_662706


namespace find_a2007_plus_a2008_l662_662749

noncomputable def a : ℕ → ℝ
def q : ℝ := sorry

-- Assume that a_n is a geometric sequence with common ratio q
axiom geo_seq (n : ℕ) : a (n + 1) = a n * q
-- Assume q > 1
axiom q_gt_1 : q > 1
-- Assume that a_{2005} and a_{2006} are the roots of the equation 4x^2 - 8x + 3 = 0
axiom roots_eqn : ∀ {a2005 a2006 : ℝ}, (a 2005 = a2005) ∧ (a 2006 = a2006) → (4 * a2005^2 - 8 * a2005 + 3 = 0) ∧ (4 * a2006^2 - 8 * a2006 + 3 = 0)

theorem find_a2007_plus_a2008 : a 2007 + a 2008 = 18 :=
sorry

end find_a2007_plus_a2008_l662_662749


namespace integer_solutions_count_l662_662292

-- Define the conditions and the goal 
theorem integer_solutions_count : 
  let S := { (x, y) : ℤ × ℤ | 2 * x + y = 13 ∧ |y| ≤ 13 } 
  in S.toFinset.card = 14 := 
by 
  -- sorry is used to skip the proof 
  sorry

end integer_solutions_count_l662_662292


namespace quadratic_roots_f12_l662_662588

noncomputable def f (i : ℕ) : polynomial ℝ :=
  polynomial.C 1 * polynomial.X^2 + polynomial.C (b i) * polynomial.X + polynomial.C (c i)

def b (i : ℕ) : ℝ :=
  if i = 1 then -1
  else 2 ^ (i - 1)

def c (i : ℕ) : ℝ :=
  -32 * b i - 1024

theorem quadratic_roots_f12 :
  polynomial.roots (f 12).to_finsupp.to_multiset = {2016, 32} :=
sorry

end quadratic_roots_f12_l662_662588


namespace nicky_total_run_time_l662_662921

-- Define the conditions and needed statements
def nicky_speed : ℝ := 3.5
def cristina_speed : ℝ := 6
def head_start : ℝ := 25

-- Define the expression to find the time Cristina catches up to Nicky
def cristina_catchup_time : ℝ := 87.5 / (cristina_speed - nicky_speed)

-- Calculate the total time Nicky runs
def total_time_nicky_runs : ℝ := cristina_catchup_time + head_start

theorem nicky_total_run_time :
  total_time_nicky_runs = 60 :=
by
  have dist_nicky_head_start : ℝ := nicky_speed * head_start
  have dist_nicky_catchup : ℝ := dist_nicky_head_start + nicky_speed * cristina_catchup_time
  have dist_cristina_catchup : ℝ := cristina_speed * cristina_catchup_time
  have catchup_eq : dist_cristina_catchup = dist_nicky_catchup by sorry
  
  show total_time_nicky_runs = 60 from
  calc
    total_time_nicky_runs
        = cristina_catchup_time + head_start : rfl
    ... = 35 + 25 : by sorry
    ... = 60 : by sorry

end nicky_total_run_time_l662_662921


namespace remainder_div_40_l662_662100

theorem remainder_div_40 (a : ℤ) : 
  let n := 40 * a + 2 in
  (n^2 - 3 * n + 5) % 40 = 3 := by
  sorry

end remainder_div_40_l662_662100


namespace orthocenter_of_triangle_l662_662603

noncomputable def orthocenter (A B C : Point) : Point := sorry

variables (S A B C O : Point)
variables [plane_geom : PlaneGeometry ℝ]

open PlaneGeometry

axiom pyramid_condition_1 : S ≠ A
axiom pyramid_condition_2 : S ≠ B
axiom pyramid_condition_3 : S ≠ C

axiom perpendicular_SA_SB : IsPerpendicular (S, A) (S, B)
axiom perpendicular_SA_SC : IsPerpendicular (S, A) (S, C)
axiom perpendicular_SB_SC : IsPerpendicular (S, B) (S, C)
axiom SO_is_altitude : ∀ P, P ∈ LineThrough S O → 
  IsPerpendicular (S, P) (PlaneContainingPoints S A B C) 

theorem orthocenter_of_triangle :
  IsOrthocenter O A B C :=
  sorry

end orthocenter_of_triangle_l662_662603


namespace sequence_values_l662_662339

theorem sequence_values (a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) 
    (h_arith : 2 + (a - 2) = a + (b - a)) (h_geom : a * a = b * (9 / b)) : a = 4 ∧ b = 6 :=
by
  -- insert proof here
  sorry

end sequence_values_l662_662339


namespace no_winning_strategy_for_A_l662_662975

structure GameBoard :=
(left_part : Set ℕ) -- Squares in the left part
(right_part : Set ℕ) -- Squares in the right part
(segments : Set (ℕ × ℕ)) -- Segments connecting squares from different parts
(move_able : ∀ {a b : ℕ}, (a ∈ left_part ∨ a ∈ right_part) → (b ∈ left_part ∨ b ∈ right_part) → (a, b) ∈ segments ∨ (b, a) ∈ segments)

def initial_positions (board : GameBoard) :=
  ∃ l₀ r₀ : ℕ, l₀ ∈ board.left_part ∧ r₀ ∈ board.right_part

def move (board : GameBoard) (l r : ℕ) : Prop :=
  ∃ m₁ m₂ : ℕ, (m₁ = l ∧ (l, m₂) ∈ board.segments ∧ m₂ ≠ r) ∨ (m₂ = r ∧ (r, m₁) ∈ board.segments ∧ m₁ ≠ l)

theorem no_winning_strategy_for_A : 
  ∀ (board : GameBoard) (l₀ r₀ : ℕ), 
    l₀ ∈ board.left_part ∧ r₀ ∈ board.right_part ∧
    initial_positions board →
    ¬ ∃ strategy_for_A, -- some putative strategy for Basha
        (∀ board : GameBoard, ∀ l₀ r₀ : ℕ, 
          (¬ move board l₀ r₀) → 
          Basha_has_winning_strategy board l₀ r₀ strategy_for_A) := 
sorry

end no_winning_strategy_for_A_l662_662975


namespace largest_cube_side_length_in_sphere_containing_cuboid_l662_662303

theorem largest_cube_side_length_in_sphere_containing_cuboid :
  let a := 22
  let b := 2
  let c := 10
  ∃ s : ℝ, (s * real.sqrt 3 = real.sqrt (a^2 + b^2 + c^2)) ∧ s = 14 := by
sorry

end largest_cube_side_length_in_sphere_containing_cuboid_l662_662303


namespace minimum_n_is_65_l662_662489

def connected (a b : ℕ) : Prop := sorry -- Define connection relation appropriately

def gcd_relatively_prime (a b n : ℕ) : Prop := Nat.gcd (a^2 + b^2) n = 1
def gcd_common_divisor_greater_than_one (a b n : ℕ) : Prop := Nat.gcd (a^2 + b^2) n > 1

noncomputable def smallest_valid_n : ℕ :=
  if h : ∃ n : ℕ, ∀ (a b : ℕ),
    (¬ connected a b → gcd_relatively_prime a b n) ∧
    (connected a b → gcd_common_divisor_greater_than_one a b n)
  then classical.choose h
  else 0

theorem minimum_n_is_65 : smallest_valid_n = 65 :=
by
  sorry

end minimum_n_is_65_l662_662489


namespace smallest_positive_integer_l662_662361

theorem smallest_positive_integer
    (n : ℕ)
    (h : ∀ (a : Fin n → ℤ), ∃ (i j : Fin n), i ≠ j ∧ (2009 ∣ (a i + a j) ∨ 2009 ∣ (a i - a j))) : n = 1006 := by
  -- Proof is required here
  sorry

end smallest_positive_integer_l662_662361


namespace total_travel_time_is_correct_l662_662999

-- Conditions as definitions
def total_distance : ℕ := 200
def initial_fraction : ℚ := 1 / 4
def initial_time : ℚ := 1 -- in hours
def lunch_time : ℚ := 1 -- in hours
def remaining_fraction : ℚ := 1 / 2
def pit_stop_time : ℚ := 0.5 -- in hours
def speed_increase : ℚ := 10

-- Derived/Calculated values needed for the problem statement
def initial_distance : ℚ := initial_fraction * total_distance
def initial_speed : ℚ := initial_distance / initial_time
def remaining_distance : ℚ := total_distance - initial_distance
def half_remaining_distance : ℚ := remaining_fraction * remaining_distance
def second_drive_time : ℚ := half_remaining_distance / initial_speed
def last_distance : ℚ := remaining_distance - half_remaining_distance
def last_speed : ℚ := initial_speed + speed_increase
def last_drive_time : ℚ := last_distance / last_speed

-- Total time calculation
def total_time : ℚ :=
  initial_time + lunch_time + second_drive_time + pit_stop_time + last_drive_time

-- Lean theorem statement
theorem total_travel_time_is_correct : total_time = 5.25 :=
  sorry

end total_travel_time_is_correct_l662_662999


namespace degree_measure_angle_BCD_correct_l662_662122

open Real

noncomputable def degree_measure_angle_BCD (circle O : Type) (EB DC AB F D C B E : O) : ℝ :=
  if h₁ : ∃ (O : Type) (EB DC AB F D C B E : O), -- Circle O with points E, B, D, C, A, F lying on it.
    (∃ E B, EB = line(E,B) ∧ is_diameter(EB)) ∧          -- EB is a diameter
    (∃ D C, is_line(DC) ∧ DC ∥ EB) ∧                    -- DC is a line parallel to EB
    (∃ A B F E, AB ∩ circle = {A,F} ∧ AB ∥ ED) ∧        -- AB intersects the circle at points A and F and is parallel to ED
    (∃ α β : ℝ, angle(A,F,B) = α ∧ angle(A,B,F) = β ∧   -- Given angle ratio AFB:ABF = 3:2
      3*α = 2*β) ∧
    (∃ B C D, ∠BCD = 1/2*(∠EB - ∠EDF))                   -- Compute angle BCD
  then 72
  else 0

/-- Prove that the degree measure of angle BCD is 72 degrees under given conditions -/
theorem degree_measure_angle_BCD_correct
  (circle O : Type) (EB DC AB F D C B E : O)
  (is_diameter : is_diameter EB)
  (DC_parallel_EB : is_parallel DC EB)
  (AB_intersect_F_ED_parallel : is_parallel AB ED ∧ AB_intersects_circle_at AB F O)
  (angle_ratios : ∃ α β : ℝ, ∠A(F)B = 3*α ∧ ∠A(B)F = 2*α) :
  degree_measure_angle_BCD circle O EB DC AB F D C B E = 72 := by
  sorry

end degree_measure_angle_BCD_correct_l662_662122


namespace area_bounded_by_f_and_x_axis_l662_662348

def f (x : ℝ) : ℝ := x^2 * cos x
def lower_bound (x : ℝ) : ℝ := 0
def upper_limit := (Real.pi) / 2

theorem area_bounded_by_f_and_x_axis :
  (∫ x in 0..upper_limit, f x) = (Real.pi^2 / 4) - 2 :=
by
  sorry

end area_bounded_by_f_and_x_axis_l662_662348


namespace sin_cos_sixth_sum_l662_662529

noncomputable def theta : ℝ := 25 * Real.pi / 180  -- Convert degrees to radians

theorem sin_cos_sixth_sum :
  (Real.tan theta = 1 / 6) →
  Real.sin theta ^ 6 + Real.cos theta ^ 6 = 11 / 12 :=
begin
  intro h,
  sorry
end

end sin_cos_sixth_sum_l662_662529


namespace area_of_triangle_AOB_l662_662058

noncomputable def triangle_area (z1 z2 : ℂ) : ℝ :=
  0.5 * (complex.abs z1) * (complex.abs z2) * (complex.argument z1 - complex.argument z2).sin

theorem area_of_triangle_AOB 
  (z1 z2 : ℂ) 
  (h1 : complex.abs z2 = 4) 
  (h2 : 4 * z1^2 - 2 * z1 * z2 + z2^2 = 0):
  triangle_area z1 z2 = 2 * real.sqrt 3 :=
sorry

end area_of_triangle_AOB_l662_662058


namespace collinear_points_l662_662342

variables {A B C D P Q R E F : Type*}
variables [InCircle A B C D] (AB C)[LineIntersection A B P C D] (AD BC)[LineIntersection A D Q B C] (AC BD)[LineIntersection A C R B D] (CircleTangent Q E F)

theorem collinear_points
  (quad_condition: quadrilateral A B C D)
  (extension_AB_C: intersect_extension A B P C D)
  (extension_AD_BC: intersect_extension A D Q B C)
  (diagonals_intersect_R: intersect_extension A C R B D)
  (tangents_from_Q: tangent Q E F) :
  collinear P F R E := by
  sorry

end collinear_points_l662_662342


namespace crazy_silly_school_movie_count_l662_662244

theorem crazy_silly_school_movie_count
  (books : ℕ) (read_books : ℕ) (watched_movies : ℕ) (diff_books_movies : ℕ)
  (total_books : books = 8) 
  (read_movie_count : watched_movies = 19)
  (read_book_count : read_books = 16)
  (book_movie_diff : watched_movies = read_books + diff_books_movies)
  (diff_value : diff_books_movies = 3) :
  ∃ M, M ≥ 19 :=
by
  sorry

end crazy_silly_school_movie_count_l662_662244


namespace min_value_of_quadratic_l662_662082

noncomputable def quadratic_min_value (x : ℕ) : ℝ :=
  3 * (x : ℝ)^2 - 12 * x + 800

theorem min_value_of_quadratic : (∀ x : ℕ, quadratic_min_value x ≥ 788) ∧ (quadratic_min_value 2 = 788) :=
by
  sorry

end min_value_of_quadratic_l662_662082


namespace count_valid_permutations_l662_662623

def digits : List ℕ := [1, 2, 3, 4, 5, 6]

def is_valid_permutation (l : List ℕ) : Prop :=
  l ∈ digits.permutations ∧
  List.indexOf l 1 < List.indexOf l 3

theorem count_valid_permutations : (List.filter is_valid_permutation digits.permutations).length = 360 := 
  sorry

end count_valid_permutations_l662_662623


namespace congruent_faces_of_pyramid_l662_662567

noncomputable def triangular_pyramid (T : Type*) := sorry

variables {A B C D : Type*}
variable [triangular_pyramid A]

def equal_dihedral_angles (T : triangular_pyramid A) : Prop := sorry

theorem congruent_faces_of_pyramid (T : triangular_pyramid A)
  (h : equal_dihedral_angles T) :
  -- Question: The faces of the triangular pyramid are congruent
  sorry :=
begin
  -- proof is skipped and hence replaced with sorry
  sorry
end

end congruent_faces_of_pyramid_l662_662567


namespace largest_number_with_5_in_tens_l662_662257

theorem largest_number_with_5_in_tens:
  ∀ (a b c d : ℕ), 
  ({a, b, c, d} = {4, 6, 5, 0} ∧ 0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9) →
  ∃ (n : ℕ), 
  (n = a * 100 + b * 10 + c ∧ b = 5 ∧ 
  n = 654) :=
by
  sorry

end largest_number_with_5_in_tens_l662_662257


namespace perimeter_of_rectangle_l662_662236

theorem perimeter_of_rectangle (s : ℝ) (h1 : 4 * s = 160) : 2 * (s + s / 4) = 100 :=
by
  sorry

end perimeter_of_rectangle_l662_662236


namespace lloyd_total_hours_worked_l662_662914

-- Definitions of the given conditions
variables (regular_hours : ℝ) (regular_rate : ℝ) (excess_rate_multiplier : ℝ) (total_earnings : ℝ)

-- Given conditions
def lloyd_regular_hours : Prop := regular_hours = 7.5
def lloyd_regular_rate : Prop := regular_rate = 3.5
def lloyd_excess_rate : Prop := excess_rate_multiplier = 1.5
def lloyd_total_earnings : Prop := total_earnings = 42

-- The proof problem to be stated: Prove that Lloyd worked 10.5 hours
theorem lloyd_total_hours_worked (h1 : lloyd_regular_hours) (h2 : lloyd_regular_rate) (h3 : lloyd_excess_rate) (h4 : lloyd_total_earnings) :
  regular_hours + (total_earnings - (regular_hours * regular_rate)) / (regular_rate * excess_rate_multiplier) = 10.5 := sorry

end lloyd_total_hours_worked_l662_662914


namespace price_increase_for_restoration_l662_662219

theorem price_increase_for_restoration (P : ℝ) (hP : P > 0) (reduction : ℝ) (hreduction : reduction = 0.8) :
  let new_price := reduction * P in
  let factor := P / new_price in
  (factor - 1) * 100 = 25 :=
by
  sorry

end price_increase_for_restoration_l662_662219


namespace product_gcd_lcm_eq_1296_l662_662383

theorem product_gcd_lcm_eq_1296 : (Int.gcd 24 54) * (Int.lcm 24 54) = 1296 := by
  sorry

end product_gcd_lcm_eq_1296_l662_662383


namespace function_b_is_even_and_monotonically_increasing_l662_662676

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def is_monotonically_increasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ ⦃a b : ℝ⦄, a ∈ s → b ∈ s → a < b → f a ≤ f b

def f (x : ℝ) : ℝ := abs x + 1

theorem function_b_is_even_and_monotonically_increasing :
  is_even_function f ∧ is_monotonically_increasing_on f (Set.Ioi 0) :=
by
  sorry

end function_b_is_even_and_monotonically_increasing_l662_662676


namespace solution_set_l662_662068

variable {f : ℝ → ℝ}

def decreasing_iff_derivative_neg (f : ℝ → ℝ) :=
∀ x, deriv f x < 0

noncomputable def phi (f : ℝ → ℝ) := λ x, f x - x / 2 - 1 / 2

theorem solution_set (hf1 : f 1 = 1) (hderiv : ∀ x, deriv f x < 1 / 2) :
  ∀ x, f x < x / 2 + 1 / 2 ↔ x > 1 :=
by
  have hphi : ∀ x, deriv (phi f) x = deriv f x - 1 / 2 := by
    sorry -- the derivative computation for phi should be placed here

  have hphi_neg : decreasing_iff_derivative_neg (phi f) := by
    intro x
    rw [hphi]
    exact hderiv x

  have hphi_1 : phi f 1 = 0 := by
    simp [hf1, phi]

  intro x
  split
  · intro hx
    have := hphi_neg x
    sorry -- here we would use the decreasing property of phi to show x > 1

  · intro hx
    have := hphi_neg x
    sorry -- here we would show the equivalence in the other direction

end solution_set_l662_662068


namespace joao_claudia_scores_l662_662878

theorem joao_claudia_scores (joao_score claudia_score total_score : ℕ) 
  (h1 : claudia_score = joao_score + 13)
  (h2 : total_score = joao_score + claudia_score)
  (h3 : 100 ≤ total_score ∧ total_score < 200) :
  joao_score = 68 ∧ claudia_score = 81 := by
  sorry

end joao_claudia_scores_l662_662878


namespace Meena_cookies_left_l662_662562

def cookies_initial := 5 * 12
def cookies_sold_to_teacher := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

def cookies_left := cookies_initial - cookies_sold_to_teacher - cookies_bought_by_brock - cookies_bought_by_katy

theorem Meena_cookies_left : cookies_left = 15 := 
by 
  -- steps to be proven here
  sorry

end Meena_cookies_left_l662_662562


namespace percentage_failed_english_l662_662847

-- Definitions
variables (total_students F_H F_BE P_BE F_E : ℝ)
-- Given conditions
def condition1 : F_H = 25 := sorry
def condition2 : F_BE = 40 := sorry
def condition3 : P_BE = 80 := sorry
-- The complement condition derived
def failed_at_least_one_subject := 100 - P_BE

-- The target statement to prove
theorem percentage_failed_english :
  F_H + F_E - F_BE = failed_at_least_one_subject → F_E = 35 :=
begin
  -- Using given conditions
  rw condition1,
  rw condition2,
  rw condition3,
  -- The inclusion-exclusion principle applied
  intros h,
  -- Solving for F_E
  linarith,
end

end percentage_failed_english_l662_662847


namespace no_such_finite_set_l662_662179

variable {n : ℕ} (hn : n > 3)

/- Definition of finite set G and conditions -/
structure G (n : ℕ) :=
  (vectors : Finset (ℝ × ℝ))
  (non_parallel : ∀ v1 v2 ∈ vectors, v1 ≠ v2 → ∃ k, v1 ≠ k • v2)
  (card_gt_2n : vectors.card > 2 * n)
  (property1 : ∀ S : Finset (ℝ × ℝ), S.card = n →
    ∃ T : Finset (ℝ × ℝ), T.card = n - 1 ∧ S ∪ T ⊆ vectors ∧ (S ∪ T).sum = 0)
  (property2 : ∀ S : Finset (ℝ × ℝ), S.card = n →
    ∃ T : Finset (ℝ × ℝ), T.card = n ∧ S ∪ T ⊆ vectors ∧ (S ∪ T).sum = 0)

/- The actual theorem statement -/
theorem no_such_finite_set :
  ¬ ∃ G : G n, true :=
by
  intro h
  cases h with g _ 
  reject sorry

end no_such_finite_set_l662_662179


namespace smaller_angle_at_3_15_l662_662790

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l662_662790


namespace D_l662_662898

open_locale classical

variables (B D C A P E F D' : Type*)
variables [incidence_structure B D C]
variables [incidence_structure A B C D P E F D']

-- Hypothesize the fixed point conditions
axiom BC_line : collinear B C D
axiom D_segment : D ∈ [B, C]

-- Hypothesize the mobile points and their conditions
axiom A_outside : ¬collinear B C A
axiom P_on_AD : ∃ (d : incidence_structure), ∈ [d]

-- Define the intersections
def E := AC ∩ BP
def F := AB ∩ CP
def D' := FE ∩ BC

-- Main statement to prove
theorem D'_fixed : D' ∈ fixed_points B C D :=
sorry

end D_l662_662898


namespace difference_between_possible_values_of_x_l662_662820

noncomputable def difference_of_roots (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) : ℝ :=
  let sol1 := 11  -- First root
  let sol2 := -11 -- Second root
  sol1 - sol2

theorem difference_between_possible_values_of_x (x : ℝ) (h : (x + 3)^2 / (3 * x + 65) = 2) :
  difference_of_roots x h = 22 :=
sorry

end difference_between_possible_values_of_x_l662_662820


namespace ceil_x_squared_values_count_l662_662813

open Real

theorem ceil_x_squared_values_count (x : ℝ) (h : ceil x = 15) : 
  ∃ n : ℕ, n = 29 ∧ ∃ a b : ℕ, a ≤ b ∧ (∀ (m : ℕ), a ≤ m ∧ m ≤ b → (ceil (x^2) = m)) := 
by
  sorry

end ceil_x_squared_values_count_l662_662813


namespace polynomial_number_l662_662910

def polynomial_count (n : ℕ) (a : ℕ → ℤ) : ℕ :=
  if (n + ∑ i in Finset.range n.succ, (int.nat_abs (a i)) = 3) ∧ (a 0 > 0) then 1 else 0

theorem polynomial_number : ∑ n in Finset.range 3, ∑ a in Finset.range (3 - n + 1), 
  polynomial_count n (fun i => if i = 0 then a else 0) = 5 :=
sorry

end polynomial_number_l662_662910


namespace true_false_questions_count_l662_662956

noncomputable def number_of_true_false_questions (T F M : ℕ) : Prop :=
  T + F + M = 45 ∧ M = 2 * F ∧ F = T + 7

theorem true_false_questions_count : ∃ T F M : ℕ, number_of_true_false_questions T F M ∧ T = 6 :=
by
  sorry

end true_false_questions_count_l662_662956


namespace museum_revenue_from_college_students_l662_662856

/-!
In one day, 200 people visit The Metropolitan Museum of Art in New York City. Half of the visitors are residents of New York City. 
Of the NYC residents, 30% are college students. If the cost of a college student ticket is $4, we need to prove that 
the museum gets $120 from college students that are residents of NYC.
-/

theorem museum_revenue_from_college_students :
  let total_visitors := 200
  let residents_nyc := total_visitors / 2
  let college_students_percentage := 30 / 100
  let college_students := residents_nyc * college_students_percentage
  let ticket_cost := 4
  residents_nyc = 100 ∧ 
  college_students = 30 ∧ 
  ticket_cost * college_students = 120 := 
by
  sorry

end museum_revenue_from_college_students_l662_662856


namespace part_I_part_II_l662_662890

def S_n (n : ℕ) : ℕ := sorry
def a_n (n : ℕ) : ℕ := sorry

theorem part_I (n : ℕ) (h1 : 2 * S_n n = 3^n + 3) :
  a_n n = if n = 1 then 3 else 3^(n-1) :=
sorry

theorem part_II (n : ℕ) (h1 : a_n 1 = 1) (h2 : ∀ n : ℕ, a_n (n + 1) - a_n n = 2^n) :
  S_n n = 2^(n + 1) - n - 2 :=
sorry

end part_I_part_II_l662_662890


namespace temperature_on_friday_is_35_l662_662202

variables (M T W Th F : ℤ)

def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem temperature_on_friday_is_35
  (h1 : (M + T + W + Th) / 4 = 48)
  (h2 : (T + W + Th + F) / 4 = 46)
  (h3 : M = 43)
  (h4 : is_odd M)
  (h5 : is_odd T)
  (h6 : is_odd W)
  (h7 : is_odd Th)
  (h8 : is_odd F) : 
  F = 35 :=
sorry

end temperature_on_friday_is_35_l662_662202


namespace smallest_integer_m_l662_662147

theorem smallest_integer_m {r : ℕ} (h : r > 0) : 
  ∃ m, (∀ (A : Finset (Finset ℕ)), (∀ i j, i ≠ j → (A i ∩ A j) = ∅) →
    (Finset.bUnion A id = Finset.range (m + 1)) →
    ∃ a b ∈ A.some, (1 ≤ (b / a)) ∧ ((b / a) ≤ (1 + (1 / 2022)))) ∧
    (∀ n, n < m → ¬ (∀ (A' : Finset (Finset ℕ)), (∀ i j, i ≠ j → (A' i ∩ A' j) = ∅) →
    (Finset.bUnion A' id = Finset.range (n + 1)) →
    ∃ a b ∈ A'.some, (1 ≤ (b / a)) ∧ ((b / a) ≤ (1 + (1 / 2022)))) :=
begin
  sorry
end

end smallest_integer_m_l662_662147


namespace trevor_eggs_from_blanche_l662_662504

theorem trevor_eggs_from_blanche :
  let gertrude_eggs := 4
  let nancy_eggs := 2
  let martha_eggs := 2
  let dropped_eggs := 2
  let left_eggs := 9
  let total_eggs_before_dropping := left_eggs + dropped_eggs
  let known_eggs := gertrude_eggs + nancy_eggs + martha_eggs
  ∃ blanche_eggs : ℕ, total_eggs_before_dropping = known_eggs + blanche_eggs ∧ blanche_eggs = 3 := 
begin
  let gertrude_eggs := 4,
  let nancy_eggs := 2,
  let martha_eggs := 2,
  let dropped_eggs := 2,
  let left_eggs := 9,
  let total_eggs_before_dropping := left_eggs + dropped_eggs,
  let known_eggs := gertrude_eggs + nancy_eggs + martha_eggs,
  use (total_eggs_before_dropping - known_eggs),
  split,
  repeat { sorry }
end

end trevor_eggs_from_blanche_l662_662504


namespace pregnant_female_cows_count_l662_662198

-- Definitions for conditions
def total_cows : ℕ := 44
def percentage_female : ℝ := 0.50
def percentage_pregnant_females : ℝ := 0.50

-- Main theorem statement
theorem pregnant_female_cows_count : 
  (total_cows * percentage_female * percentage_pregnant_females).to_nat = 11 :=
by
  sorry

end pregnant_female_cows_count_l662_662198


namespace most_consistent_player_l662_662286

section ConsistentPerformance

variables (σA σB σC σD : ℝ)
variables (σA_eq : σA = 0.023)
variables (σB_eq : σB = 0.018)
variables (σC_eq : σC = 0.020)
variables (σD_eq : σD = 0.021)

theorem most_consistent_player : σB < σC ∧ σB < σD ∧ σB < σA :=
by 
  rw [σA_eq, σB_eq, σC_eq, σD_eq]
  sorry

end ConsistentPerformance

end most_consistent_player_l662_662286


namespace min_odd_squares_in_grid_rectangle_l662_662306

theorem min_odd_squares_in_grid_rectangle (w h : ℕ) (H_w : w = 629) (H_h : h = 630) : (∃ n : ℕ, min_odd_squares w h n ∧ n = 2) :=
by
  sorry

end min_odd_squares_in_grid_rectangle_l662_662306


namespace total_chairs_calculation_l662_662517

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l662_662517


namespace contractor_laborers_l662_662280

theorem contractor_laborers (x : ℕ) (h : 9 * x = 15 * (x - 6)) : x = 15 :=
by
  sorry

end contractor_laborers_l662_662280


namespace polynomial_even_evaluation_count_l662_662487

theorem polynomial_even_evaluation_count (f : ℝ → ℝ) (n : ℕ) (hf : ∀ x, polynomial.eval (x : polynomial ℝ) ≤ n) :
  (even n → (∃ evs : fin n → ℝ, ∀ (x : ℝ), (f x = f (-x)) → x ∈ evs)) ∧
  (odd n → (∃ evs : fin (n + 1) → ℝ, ∀ (x : ℝ), (f x = f (-x)) → x ∈ evs)) :=
sorry

end polynomial_even_evaluation_count_l662_662487


namespace dot_product_vec_a_vec_b_l662_662776

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 1)

def dot_product (a b : ℝ × ℝ) : ℝ := a.1 * b.1 + a.2 * b.2

theorem dot_product_vec_a_vec_b : dot_product vec_a vec_b = 1 := by
  sorry

end dot_product_vec_a_vec_b_l662_662776


namespace alex_pen_difference_l662_662337

theorem alex_pen_difference 
  (alex_initial_pens : Nat) 
  (doubling_rate : Nat) 
  (weeks : Nat) 
  (jane_pens_month : Nat) :
  alex_initial_pens = 4 →
  doubling_rate = 2 →
  weeks = 4 →
  jane_pens_month = 16 →
  (alex_initial_pens * doubling_rate ^ weeks) - jane_pens_month = 16 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  simp
  sorry

end alex_pen_difference_l662_662337


namespace number_of_possible_ceil_values_l662_662809

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l662_662809


namespace trapezoid_area_is_8_l662_662130

noncomputable def trapezoid_area
  (AB CD : ℝ)        -- lengths of the bases
  (h : ℝ)            -- height (distance between the bases)
  : ℝ :=
  0.5 * (AB + CD) * h

theorem trapezoid_area_is_8 
  (AB CD : ℝ) 
  (h : ℝ) 
  (K M : ℝ) 
  (height_condition : h = 2)
  (AB_condition : AB = 5)
  (CD_condition : CD = 3)
  (K_midpoint : K = AB / 2) 
  (M_midpoint : M = CD / 2)
  : trapezoid_area AB CD h = 8 :=
by
  rw [trapezoid_area, AB_condition, CD_condition, height_condition]
  norm_num

end trapezoid_area_is_8_l662_662130


namespace max_difference_of_segment_sums_l662_662365

theorem max_difference_of_segment_sums :
  ∀ (x : Fin 100 → ℝ), 
  ∑ i : Fin 100, x i ^ 2 = 1 → 
  let k : ℝ := ∑ (i : Fin 100) (j : Fin 100) (h : i < j ∧ (i + j) % 2 = 0), x i * x j,
      s : ℝ := ∑ (i : Fin 100) (j : Fin 100) (h : i < j ∧ (i + j) % 2 = 1), x i * x j
  in k - s ≤ 1 / 2 := sorry

end max_difference_of_segment_sums_l662_662365


namespace white_fraction_of_surface_area_l662_662653

theorem white_fraction_of_surface_area :
  let side_length := 4
  let total_surface_area := 6 * side_length ^ 2
  let num_black_faces := 3 * 8 + 16
  let num_white_faces := total_surface_area - num_black_faces
  let fraction_white := num_white_faces / total_surface_area
  fraction_white = 7 / 12 :=
by {
  let side_length := 4
  let total_surface_area := 6 * side_length ^ 2
  let num_black_faces := 3 * 8 + 16
  let num_white_faces := total_surface_area - num_black_faces
  let fraction_white := num_white_faces / total_surface_area
  have h1 : fraction_white = 56 / 96 := by {
    simp [total_surface_area, num_black_faces],
    norm_num,
  }
  norm_num at h1,
  exact h1,
}

end white_fraction_of_surface_area_l662_662653


namespace perp_from_tangent_point_passes_through_incenter_l662_662930

theorem perp_from_tangent_point_passes_through_incenter
  (A B C D X : Point)
  (hD_on_AB : lies_on D AB)
  (circle_ADC_touches_CD : touches_internally (inscribed_circle ADC) (circumcircle ACD))
  (circle_BDC_touches_CD : touches_internally (inscribed_circle BDC) (circumcircle BCD))
  (common_tangent_CD : touches (inscribed_circle ADC) CD = touches (inscribed_circle BDC) CD at X) :
  passes_through (perpendicular X AB) (incenter A B C) :=
sorry

end perp_from_tangent_point_passes_through_incenter_l662_662930


namespace minimum_value_y_is_2_l662_662818

noncomputable def minimum_value_y (x : ℝ) : ℝ :=
  x + (1 / x)

theorem minimum_value_y_is_2 (x : ℝ) (hx : 0 < x) : 
  (∀ y, y = minimum_value_y x → y ≥ 2) :=
by
  sorry

end minimum_value_y_is_2_l662_662818


namespace max_difference_of_products_in_100_gon_l662_662364

theorem max_difference_of_products_in_100_gon : 
  ∃ (x : Fin 100 → ℝ), (∑ i, x i ^ 2 = 1) ∧ 
  (
    let k := ∑ i j, if (i < j) ∧ ((i + j) % 2 = 0) then x i * x j else 0
    let s := ∑ i j, if (i < j) ∧ ((i + j) % 2 ≠ 0) then x i * x j else 0
    k - s ≤ 1 / 2
  ) :=
sorry

end max_difference_of_products_in_100_gon_l662_662364


namespace regression_eq_l662_662271

variable (x : ℝ)

def y (k a : ℝ) :=
  real.exp (k * x + a)

def z (y : ℝ) :=
  real.log y

/- The given linear regression equation -/
def hat_z :=
  0.25 * x - 2.58

/- Prove the regression equation for the model -/
theorem regression_eq (k a : ℝ) (h : ∀ x, z (y k a) = 0.25 * x - 2.58) :
  y 0.25 (-2.58) = real.exp (0.25 * x - 2.58) :=
by 
  unfold y
  unfold z at h
  sorry -- proof is omitted

end regression_eq_l662_662271


namespace number_of_possible_ceil_values_l662_662808

theorem number_of_possible_ceil_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  (∃ (n : ℕ), 196 < x^2 ∧ x^2 ≤ 225 → n = 29) := by
sorry

end number_of_possible_ceil_values_l662_662808


namespace polygon_perimeter_l662_662319

theorem polygon_perimeter (hexagon_side_length : ℕ) (polygon_sides_occupied : ℕ) (additional_side : ℕ)  
  (h1 : hexagon_side_length = 8) (h2 : polygon_sides_occupied = 3) (h3 : additional_side = hexagon_side_length): 
  3 * hexagon_side_length + additional_side = 32 := 
by {
  rw [h1, h2, h3],
  norm_num,
  }


end polygon_perimeter_l662_662319


namespace periodic_sum_constant_l662_662254

noncomputable def is_periodic (f : ℝ → ℝ) (a : ℝ) : Prop :=
a ≠ 0 ∧ ∀ x : ℝ, f (a + x) = f x

theorem periodic_sum_constant (f g : ℝ → ℝ) (a b : ℝ)
  (ha : a ≠ 0) (hb : b ≠ 0) (hfa : is_periodic f a) (hgb : is_periodic g b)
  (harational : ∃ m n : ℤ, (a : ℝ) = m / n) (hbirrational : ¬ ∃ m n : ℤ, (b : ℝ) = m / n) :
  (∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, (f + g) (c + x) = (f + g) x) →
  (∀ x : ℝ, f x = f 0) ∨ (∀ x : ℝ, g x = g 0) :=
sorry

end periodic_sum_constant_l662_662254


namespace madeline_rent_l662_662556

noncomputable def groceries : ℝ := 400
noncomputable def medical_expenses : ℝ := 200
noncomputable def utilities : ℝ := 60
noncomputable def emergency_savings : ℝ := 200
noncomputable def hourly_wage : ℝ := 15
noncomputable def hours_worked : ℕ := 138
noncomputable def total_expenses_and_savings : ℝ := groceries + medical_expenses + utilities + emergency_savings
noncomputable def total_earnings : ℝ := hourly_wage * hours_worked

theorem madeline_rent : total_earnings - total_expenses_and_savings = 1210 := by
  sorry

end madeline_rent_l662_662556


namespace ratio_of_cream_l662_662705

def initial_coffee := 18
def cup_capacity := 22
def Emily_drank := 3
def Emily_added_cream := 4
def Ethan_added_cream := 4
def Ethan_drank := 3

noncomputable def cream_in_Emily := Emily_added_cream

noncomputable def cream_remaining_in_Ethan :=
  Ethan_added_cream - (Ethan_added_cream * Ethan_drank / (initial_coffee + Ethan_added_cream))

noncomputable def resulting_ratio := cream_in_Emily / cream_remaining_in_Ethan

theorem ratio_of_cream :
  resulting_ratio = 200 / 173 :=
by
  sorry

end ratio_of_cream_l662_662705


namespace angle_ratio_l662_662466

-- Define the problem setup with conditions.
variables (EFGH : Type) [parallelogram EFGH] 
          (E F G H P : EFGH)
          (h1 : diagonals_intersect E F G H P) 
          (phi : ℝ) 
          (h2 : angle GEF = 3 * phi)
          (h3 : angle GFH = 3 * phi)

-- State the theorem.
theorem angle_ratio (s : ℝ) (h4 : angle EGH = s * angle EPO) : 
  s = 11 / 14 :=
sorry

end angle_ratio_l662_662466


namespace mistaken_divisor_is_12_l662_662842

theorem mistaken_divisor_is_12 (dividend : ℕ) (mistaken_divisor : ℕ) (correct_divisor : ℕ) 
  (mistaken_quotient : ℕ) (correct_quotient : ℕ) (remainder : ℕ) :
  remainder = 0 ∧ correct_divisor = 21 ∧ mistaken_quotient = 42 ∧ correct_quotient = 24 ∧ 
  dividend = mistaken_quotient * mistaken_divisor ∧ dividend = correct_quotient * correct_divisor →
  mistaken_divisor = 12 :=
by 
  sorry

end mistaken_divisor_is_12_l662_662842


namespace min_value_frac_2sqrt6_over_3_l662_662033

noncomputable def min_value_frac (a b e : ℝ) (h : a > 0 ∧ b > 0) (ha : b/a = Real.sqrt 3) (he : e = 2) : ℝ :=
  a^2 + e / b

theorem min_value_frac_2sqrt6_over_3 :
  ∃ a b e : ℝ, a > 0 ∧ b > 0 ∧ (b/a = Real.sqrt 3) ∧ (e = 2) ∧ min_value_frac a b e ⟨by sorry, by sorry⟩ HA HE = 2 * Real.sqrt 6 / 3 :=
sorry

end min_value_frac_2sqrt6_over_3_l662_662033


namespace exists_non_parallel_diagonal_l662_662521

variable (n : ℕ) (h : n > 0)

structure ConvexPolygon (n : ℕ) :=
(vertices : Fin (2 * n) → Fin (2 * n)) -- A type representing our vertices indexed by natural numbers
(convex : Prop)                        -- A property ensuring the polygon is convex

theorem exists_non_parallel_diagonal (P : ConvexPolygon n) (h : n > 0) :
  ∃ (i j : Fin (2 * n)), (i ≠ j) ∧ (¬ ∃ (k : Fin (2 * n)), k ≠ i ∧ k ≠ j ∧ P.vertices i P.vertices j // P.vertices i P.vertices j).parallel_to P.vertices k P.vertices (k + 1) :=
sorry

end exists_non_parallel_diagonal_l662_662521


namespace trigonometric_quadrant_l662_662801

theorem trigonometric_quadrant (θ : ℝ) (h1 : Real.sin θ > Real.cos θ) (h2 : Real.sin θ * Real.cos θ < 0) : 
  (θ > π / 2) ∧ (θ < π) :=
by
  sorry

end trigonometric_quadrant_l662_662801


namespace intersection_of_curves_l662_662757

noncomputable def curve1 (t : ℝ) : ℝ × ℝ :=
  ⟨1 + (sqrt 2 / 2) * t, (sqrt 2 / 2) * t⟩

noncomputable def curve2 (ρ : ℝ) (θ : ℝ) : Prop :=
  1 = (ρ^2 * cos θ^2) / 2 + ρ^2 * sin θ^2

noncomputable def M := (1 : ℝ, 0 : ℝ)

theorem intersection_of_curves 
    (t : ℝ)
    (A B : ℝ × ℝ)
    (hA1 : A ∈ (λ t, curve1 t) '' set.univ)
    (hA2 : A ∈ (λ (ρ θ), curve2 ρ θ) '' set.univ × set.univ)
    (hB1 : B ∈ (λ t, curve1 t) '' set.univ)
    (hB2 : B ∈ (λ (ρ θ), curve2 ρ θ) '' set.univ × set.univ)
    (hA_ne_B : A ≠ B) : 
  (dist M A * dist M B / dist A B) = sqrt 2 / 4 :=
sorry

end intersection_of_curves_l662_662757


namespace at_least_three_babies_speak_l662_662828

structure BabySpeaking (n : ℕ) where
  prob_speak : ℝ
  prob_not_speak : ℝ
  total_babies : ℕ

noncomputable def prob_at_least_three_speak (b : BabySpeaking 6) : ℝ :=
  1 - (prob_none_speak b + prob_one_speak b + prob_two_speak b)
where
  prob_none_speak (b : BabySpeaking 6) : ℝ :=
    b.prob_not_speak ^ b.total_babies
  prob_one_speak (b : BabySpeaking 6) : ℝ :=
    (Nat.choose b.total_babies 1 : ℝ) * b.prob_speak * (b.prob_not_speak)^(b.total_babies - 1)
  prob_two_speak (b : BabySpeaking 6) : ℝ :=
    (Nat.choose b.total_babies 2 : ℝ) * (b.prob_speak^2) * (b.prob_not_speak^(b.total_babies - 2))

theorem at_least_three_babies_speak (b : BabySpeaking 6) (h : b.prob_speak = 1/3) (h' : b.prob_not_speak = 2/3) : 
  prob_at_least_three_speak b = 353 / 729 := by
  let p_none := prob_none_speak b
  let p_one := prob_one_speak b
  let p_two := prob_two_speak b
  let p_fewer_than_three := p_none + p_one + p_two
  have h1 : p_none = (2/3)^6 := sorry
  have h2 : p_one = (Nat.choose 6 1 : ℝ) * (1/3) * (2/3)^5 := sorry
  have h3 : p_two = (Nat.choose 6 2 : ℝ) * (1/3)^2 * (2/3)^4 := sorry
  have h4 : p_fewer_than_three = ((64 : ℝ) / 729) + 6 * (1/3) * (32/243) + 15 * (1/9) * (16/81) := sorry
  have h5 : p_fewer_than_three = 376 / 729 := sorry
  have h6 : prob_at_least_three_speak b = 1 - 376 / 729 := by
    exact sorry
  show prob_at_least_three_speak b = 353 / 729 from sorry

end at_least_three_babies_speak_l662_662828


namespace monic_polynomial_roots_scaled_l662_662905

noncomputable def p (x : ℝ) : ℝ := x^3 - 5 * x^2 + 12

theorem monic_polynomial_roots_scaled (r1 r2 r3 : ℝ) 
  (h_roots : p r1 = 0 ∧ p r2 = 0 ∧ p r3 = 0) :
  ∃ (q : ℝ → ℝ), (∀ x, q x = x^3 - 15 * x^2 + 324) ∧
    (q (3 * r1) = 0 ∧ q (3 * r2) = 0 ∧ q (3 * r3) = 0) :=
by {
  let q : ℝ → ℝ := λ x, x^3 - 15 * x^2 + 324,
  use q,
  split,
  { intro x, simp [q] },
  { simp, sorry }
}

end monic_polynomial_roots_scaled_l662_662905


namespace ceil_square_count_ceil_x_eq_15_l662_662803

theorem ceil_square_count_ceil_x_eq_15 : 
  ∀ (x : ℝ), ( ⌈x⌉ = 15 ) → ∃ n : ℕ, n = 29 ∧ ∀ k : ℕ, k = ⌈x^2⌉ → 197 ≤ k ∧ k ≤ 225 :=
sorry

end ceil_square_count_ceil_x_eq_15_l662_662803


namespace volume_prism_l662_662034

-- Definitions for the geometrical structure and given conditions

structure Point :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

-- Assume rhombus ABCD with given base angles
structure Rhombus :=
(A B C D : Point)
(angle_ABC : ℝ)

-- Assume the prism with base as a rhombus and point M at certain distances
structure Prism :=
(P A B C D M : Point)
(rhombus_base : Rhombus)
(angle_face_base : ℝ)
(distance_M : ℝ)

-- The theorem to prove
theorem volume_prism (P A B C D M : Point)
    (rhombus_base : Rhombus A B C D 60)
    (angle_face_base : angle_face_base = 60)
    (distance_M_from_base_and_face : ∀ (face : list Point), face ∈ [[P,A,B,D], [P,B,C,D], [P,A,C,D]] → distance M face = 1) :
    volume_prism (P A B C D M) = 8 * real.sqrt 3 :=
sorry

end volume_prism_l662_662034


namespace complex_mul_real_a_eq_neg1_l662_662823

theorem complex_mul_real_a_eq_neg1 {a : ℝ} : 
  (1 + complex.i) * (1 + a * complex.i) ∈ set.range (coe : ℝ → ℂ) ↔ a = -1 := 
by
  sorry

end complex_mul_real_a_eq_neg1_l662_662823


namespace selling_price_correct_l662_662557

/-- Condition: Cost price (CP) of the article is Rs. 540. -/
def CP : ℝ := 540

/-- Condition: Markup percentage above the cost price is 15%. -/
def MarkupPercentage : ℝ := 15

/-- Condition: Discount percentage on the marked price is 26.40901771336554%. -/
def DiscountPercentage : ℝ := 26.40901771336554

/-- Definition: Function to calculate the marked price (MP). -/
def markedPrice (cp : ℝ) (markup : ℝ) : ℝ :=
  cp + (markup / 100) * cp

/-- Definition: Function to calculate the selling price (SP). -/
def sellingPrice (mp : ℝ) (discount : ℝ) : ℝ :=
  mp - (discount / 100) * mp

/-- Problem Statement: The selling price (SP) of the article after the discount is applied.
Prove that the selling price is approximately Rs. 456.98. -/
theorem selling_price_correct (h : CP = 540) (h1 : MarkupPercentage = 15) (h2 : DiscountPercentage = 26.40901771336554) :
  sellingPrice (markedPrice CP MarkupPercentage) DiscountPercentage = 456.98 := by
  sorry

end selling_price_correct_l662_662557


namespace probability_calculation_l662_662644

noncomputable def probability_of_three_one_digit_three_two_digit : ℚ :=
  let dice_rolls := finset.range 6
  let one_digit_prob := (9 : ℚ) / 20
  let two_digit_prob := (11 : ℚ) / 20
  let prob := (one_digit_prob ^ 3) * (two_digit_prob ^ 3)
  let combinations := nat.choose 6 3
  (combinations : ℚ) * prob

theorem probability_calculation :
  probability_of_three_one_digit_three_two_digit = 485264 / 1600000 := by
  sorry

end probability_calculation_l662_662644


namespace complex_quadrant_l662_662729

theorem complex_quadrant (a b : ℝ) (z : ℂ) (h1 : a - complex.I = (2 + b * complex.I).conj) (h2 : z = (a + b * complex.I) ^ 2) :
  0 < z.re ∧ 0 < z.im := 
sorry

end complex_quadrant_l662_662729


namespace abe_age_is_22_l662_662995

-- Define the conditions of the problem
def abe_age_condition (A : ℕ) : Prop := A + (A - 7) = 37

-- State the theorem
theorem abe_age_is_22 : ∃ A : ℕ, abe_age_condition A ∧ A = 22 :=
by
  sorry

end abe_age_is_22_l662_662995


namespace angle_at_3_15_l662_662787

theorem angle_at_3_15 : 
  let minute_hand_angle := 90.0
  let hour_at_3 := 90.0
  let hour_hand_at_3_15 := hour_at_3 + 7.5
  minute_hand_angle == 90.0 ∧ hour_hand_at_3_15 == 97.5 →
  abs (hour_hand_at_3_15 - minute_hand_angle) = 7.5 :=
by
  sorry

end angle_at_3_15_l662_662787


namespace circle_intersects_y_axis_side_origin_l662_662458

theorem circle_intersects_y_axis_side_origin (D E F : ℝ)
  (h_eq : ∀ y : ℝ, x^2 + y^2 + D * x + E * y + F = 0)
  (h_intersect : ∃ y1 y2 : ℝ, y1 ≠ y2 ∧ h_eq 0 y1 ∧ h_eq 0 y2 ∧ y1 * y2 < 0) :
  F < 0 :=
by
  sorry

end circle_intersects_y_axis_side_origin_l662_662458


namespace sum_series_base6_l662_662372

theorem sum_series_base6 : 
  let sum_series := (∑ k in range 56, (k + 1 : ℕ)) in
  sum_series_base6.to_base_6 = "2500" :=
by
  sorry

end sum_series_base6_l662_662372


namespace volume_of_rotated_solid_l662_662228

theorem volume_of_rotated_solid : 
  let first_cylinder_volume := (real.pi * 6^2 * 1) in
  let second_cylinder_volume := (real.pi * 3^2 * 4) in
  first_cylinder_volume + second_cylinder_volume = 72 * real.pi :=
by
  sorry

end volume_of_rotated_solid_l662_662228


namespace triangle_semicircle_l662_662353

noncomputable def triangle_semicircle_ratio : ℝ :=
  let AB := 8
  let BC := 6
  let CA := 2 * Real.sqrt 7
  let radius_AB := AB / 2
  let radius_BC := BC / 2
  let radius_CA := CA / 2
  let area_semicircle_AB := (1 / 2) * Real.pi * radius_AB ^ 2
  let area_semicircle_BC := (1 / 2) * Real.pi * radius_BC ^ 2
  let area_semicircle_CA := (1 / 2) * Real.pi * radius_CA ^ 2
  let area_triangle := AB * BC / 2
  let total_shaded_area := (area_semicircle_AB + area_semicircle_BC + area_semicircle_CA) - area_triangle
  let area_circle_CA := Real.pi * (radius_CA ^ 2)
  total_shaded_area / area_circle_CA

theorem triangle_semicircle : triangle_semicircle_ratio = 2 - (12 * Real.sqrt 3) / (7 * Real.pi) := by
  sorry

end triangle_semicircle_l662_662353


namespace bacteria_doubling_time_l662_662964

theorem bacteria_doubling_time (initial_count final_count : ℕ) (hours_per_doubling : ℕ) (k : ℕ) :
  initial_count * 2^k = final_count ∧ hours_per_doubling = 3 ∧ initial_count = 800 ∧ final_count = 51200 → k * hours_per_doubling = 18 :=
begin
  sorry
end

end bacteria_doubling_time_l662_662964


namespace percent_decrease_is_4_l662_662986

-- Define the revenues in 1995, 1996, and 1997 based on given conditions
def revenue_1995 (R : ℝ) : ℝ := R
def revenue_1996 (R : ℝ) : ℝ := 1.2 * R
def revenue_1997 (R : ℝ) : ℝ := revenue_1996 R * 0.8

def percent_decrease (R : ℝ) : ℝ := 
  ((revenue_1995 R - revenue_1997 R) / revenue_1995 R) * 100

-- Conditionally prove the percent decrease in revenue is 4% from 1995 to 1997 when x is 20.
theorem percent_decrease_is_4 (x : ℝ) (h : x = 20) : percent_decrease x = 4 :=
by
  unfold percent_decrease revenue_1995 revenue_1997 revenue_1996
  rw [h]
  unfold revenue_1997 revenue_1996
  sorry

end percent_decrease_is_4_l662_662986


namespace pizza_slices_with_both_toppings_l662_662318

theorem pizza_slices_with_both_toppings :
  ∃ n : ℕ, n = 6 ∧
          (∀ slices : ℕ, slices = 16 → 
          (∀ pepperoni : ℕ, pepperoni = 8 →
          (∀ mushrooms : ℕ, mushrooms = 14 →
          (∀ with_both : ℕ, with_both = n →
          (∀ only_pepperoni : ℕ, only_pepperoni = pepperoni - with_both →
          (∀ only_mushrooms : ℕ, only_mushrooms = mushrooms - with_both →
          only_pepperoni + only_mushrooms + with_both = slices))))))

end pizza_slices_with_both_toppings_l662_662318


namespace tan_half_angle_product_l662_662092

theorem tan_half_angle_product (a b : ℝ) 
  (h : 7 * (Real.cos a + Real.cos b) + 6 * (Real.cos a * Real.cos b + 1) = 0) :
  ∃ x : ℝ, x = Real.tan (a / 2) * Real.tan (b / 2) ∧ (x = Real.sqrt (26 / 7) ∨ x = -Real.sqrt (26 / 7)) :=
by
  sorry

end tan_half_angle_product_l662_662092


namespace sqrt_90000_eq_300_l662_662947

theorem sqrt_90000_eq_300 : Real.sqrt 90000 = 300 := by
  sorry

end sqrt_90000_eq_300_l662_662947


namespace correct_statements_l662_662026

def floor (x : ℝ) : ℤ := ⌊x⌋

def f (x : ℝ) : ℤ :=
  floor ((x + 1) / 3) - floor (x / 3)

def g (x : ℝ) : ℝ :=
  f x - Real.cos (π * x)

theorem correct_statements :
  (∀ x, f (x + 3) = f x) ∧ -- Statement 1: f(x) is periodic with period 3
  (∀ y ∈ set.range f, y = 0 ∨ y = 1) -- Statement 3: The range of f(x) is {0, 1}
:= sorry

end correct_statements_l662_662026


namespace total_chairs_taken_l662_662512

theorem total_chairs_taken (students trips chairs_per_trip : ℕ) (h_students : students = 5) (h_trips : trips = 10) (h_chairs_per_trip : chairs_per_trip = 5) : 
  students * trips * chairs_per_trip = 250 := by
  rw [h_students, h_trips, h_chairs_per_trip]
  norm_num

end total_chairs_taken_l662_662512


namespace circle_general_eq_l662_662721
noncomputable def center_line (x : ℝ) := -4 * x
def tangent_line (x : ℝ) := 1 - x

def is_circle (center : ℝ × ℝ) (radius : ℝ) :=
  ∃ (x y : ℝ), (x - center.1)^2 + (y - center.2)^2 = radius^2

def is_on_line (p : ℝ × ℝ) := (p.2 = center_line p.1)

def is_tangent_at_p (center : ℝ × ℝ) (p : ℝ × ℝ) (r : ℝ) :=
  is_circle center r ∧ p.2 = tangent_line p.1 ∧ (center.1 - p.1)^2 + (center.2 - p.2)^2 = r^2

theorem circle_general_eq :
  ∀ (center : ℝ × ℝ), is_on_line center →
  ∀ r, is_tangent_at_p center (3, -2) r →
  ∀ x y : ℝ, (x - center.1)^2 + (y - center.2)^2 = r^2 →
  x^2 + y^2 - 2 * x + 8 * y + 9 = 0 := by
  sorry

end circle_general_eq_l662_662721


namespace quadratic_roots_l662_662045

noncomputable def theta : ℝ := sorry

def g (x : ℝ) : ℝ := x^4 - 7 * x^2 + 1

def is_root_g (t : ℝ) : Prop := g(t) = 0

noncomputable def h (x : ℝ) : ℚ := sorry

def is_quadratic (h : ℝ → ℚ) : Prop := 
  ∃ (a b c : ℚ), (a ≠ 0) ∧ (∀ x, h(x) = a * x^2 + b * x + c)

def h_condition (h : ℝ → ℚ) (θ : ℝ) : Prop := h(θ^2 - 1) = 0

def roots_h (h : ℝ → ℚ) : set ℝ := {x : ℝ | h(x) = 0}

theorem quadratic_roots (t : ℝ) (h : ℝ → ℚ)
  (ht : is_root_g t) 
  (hq : is_quadratic h) 
  (hc : h_condition h t) : 
  roots_h h = { (5 + 3 * real.sqrt 5) / 2, (5 - 3 * real.sqrt 5) / 2 } :=
sorry

end quadratic_roots_l662_662045


namespace part1_part2_l662_662762

noncomputable def f (x a : ℝ) : ℝ := x ^ 2 + (a - 3) * x - 3 * a

theorem part1 (x : ℝ) : {x : ℝ | f x 5 > 0} = set.Union (set.Ioo (-∞) (-5)) (set.Ioo 3 ∞) := sorry

theorem part2 (x a : ℝ) :
  (if a = -3 then
    {x : ℝ | f x a > 0} = set.Union (set.Ioo (-∞) 3) (set.Ioo 3 ∞)
  else if a > -3 then
    {x : ℝ | f x a > 0} = set.Union (set.Ioo (-∞) (-a)) (set.Ioo 3 ∞)
  else
    {x : ℝ | f x a > 0} = set.Union (set.Ioo (-∞) 3) (set.Ioo (-a) ∞)
  ) := sorry

end part1_part2_l662_662762


namespace power_function_log_sum_l662_662075

noncomputable def f (x : ℝ) : ℝ := x ^ (1 / 2)

theorem power_function_log_sum :
  (f (1 / 2) = (Real.sqrt 2 / 2)) →
  log (f 2) + log (f 5) = (1 / 2 : ℝ) :=
by
  sorry

end power_function_log_sum_l662_662075


namespace triangle_area_AOB_l662_662056

noncomputable def area_of_triangle (z1 z2 : ℂ) : ℝ :=
  (1 / 2) * abs (z1 * z2 * complex.sin (real.pi / 3))

theorem triangle_area_AOB (z1 z2 : ℂ)
  (h1 : abs z2 = 4)
  (h2 : 4 * z1 ^ 2 - 2 * z1 * z2 + z2 ^ 2 = 0) :
  area_of_triangle z1 z2 = 2 * real.sqrt 3 := 
by {
  sorry
}

end triangle_area_AOB_l662_662056


namespace smallest_n_l662_662702

theorem smallest_n :
  ∃ n : ℕ, n = 10 ∧ (n * (n + 1) > 100 ∧ ∀ m : ℕ, m < n → m * (m + 1) ≤ 100) := by
  sorry

end smallest_n_l662_662702


namespace max_obtuse_angles_in_quadrilateral_l662_662116

theorem max_obtuse_angles_in_quadrilateral (a b c d : ℝ) 
  (h₁ : a + b + c + d = 360)
  (h₂ : 90 < a)
  (h₃ : 90 < b)
  (h₄ : 90 < c) :
  90 > d :=
sorry

end max_obtuse_angles_in_quadrilateral_l662_662116


namespace percentage_increase_l662_662821

-- Define the initial weekly salary of Sharon, and the target salaries.
variables (S P : ℝ) 

-- Set the known conditions
def condition1 := S * 1.20 = 600
def condition2 := S = 500
def condition3 := (1 + P / 100) * S = 575

-- The target condition to prove
theorem percentage_increase :
  condition1 → condition2 → condition3 → P = 15 :=
by
  intros h1 h2 h3
  -- Skip proof for now
  sorry

end percentage_increase_l662_662821


namespace Carla_is_2_years_older_than_Karen_l662_662837

-- Define the current age of Karen.
def Karen_age : ℕ := 2

-- Define the current age of Frank given that in 5 years he will be 36 years old.
def Frank_age : ℕ := 36 - 5

-- Define the current age of Ty given that Frank will be 3 times his age in 5 years.
def Ty_age : ℕ := 36 / 3

-- Define Carla's current age given that Ty is currently 4 years more than two times Carla's age.
def Carla_age : ℕ := (Ty_age - 4) / 2

-- Define the difference in age between Carla and Karen.
def Carla_Karen_age_diff : ℕ := Carla_age - Karen_age

-- The statement to be proven.
theorem Carla_is_2_years_older_than_Karen : Carla_Karen_age_diff = 2 := by
  -- The proof is not required, so we use sorry.
  sorry

end Carla_is_2_years_older_than_Karen_l662_662837


namespace ceil_x_pow_2_values_l662_662816

theorem ceil_x_pow_2_values (x : ℝ) (h : ⌈x⌉ = 15) : 
  ∃ n, n = 29 ∧ (∀ y, ceil (y^2) = ⌈x^2⌉ → 196 < y^2 ∧ y^2 ≤ 225) :=
sorry

end ceil_x_pow_2_values_l662_662816


namespace range_of_a_l662_662043

theorem range_of_a (a : ℝ) : (∀ x : ℝ, a * x^2 + a * x + 1 > 0 ∨ ((∃ x : ℝ, x^2 - x + a = 0) ∧ ¬ (∀ x : ℝ, a * x^2 + a * x + 1 > 0 ∧ ∃ x : ℝ, x^2 - x + a = 0))):
  a < 0 ∨ (1 / 4 < a ∧ a < 4) :=
by
  sorry

end range_of_a_l662_662043


namespace vertical_shift_equation_l662_662106

theorem vertical_shift_equation (x : ℝ) : 
  let original_parabola := (λ x : ℝ, 2 * x ^ 2)
  let shifted_parabola := (λ x : ℝ, 2 * x ^ 2 - 4)
  ∀ x : ℝ, shifted_parabola x = original_parabola x - 4 := 
by 
  let original_parabola := (λ x : ℝ, 2 * x ^ 2)
  let shifted_parabola := (λ x : ℝ, 2 * x ^ 2 - 4)
  intros
  simp 
  sorry

end vertical_shift_equation_l662_662106


namespace volleyball_tournament_l662_662474

theorem volleyball_tournament (n m : ℕ) 
  (h1: 73 % 2 = 1) -- condition: 73 teams
  (h2: ∀ x, 0 < x ∧ x < 73 → x * (73 - x) = 2628) -- condition: total matches played = 2628 (includes each game resulting in one win)
  (h3: ∀ x, x * n + (73 - x) * m = 2628) -- condition: total number of wins accounted for by each grouping
  : n = m :=
begin
  -- Proof steps not included as stated in the requirements
  sorry,
end

end volleyball_tournament_l662_662474


namespace janice_total_hours_worked_l662_662506

-- Declare the conditions as definitions
def hourly_rate_first_40_hours : ℝ := 10
def hourly_rate_overtime : ℝ := 15
def first_40_hours : ℕ := 40
def total_pay : ℝ := 700

-- Define the main theorem
theorem janice_total_hours_worked (H : ℕ) (O : ℕ) : 
  H = first_40_hours + O ∧ (hourly_rate_first_40_hours * first_40_hours + hourly_rate_overtime * O = total_pay) → H = 60 :=
by
  sorry

end janice_total_hours_worked_l662_662506


namespace at_least_three_babies_speak_l662_662829

structure BabySpeaking (n : ℕ) where
  prob_speak : ℝ
  prob_not_speak : ℝ
  total_babies : ℕ

noncomputable def prob_at_least_three_speak (b : BabySpeaking 6) : ℝ :=
  1 - (prob_none_speak b + prob_one_speak b + prob_two_speak b)
where
  prob_none_speak (b : BabySpeaking 6) : ℝ :=
    b.prob_not_speak ^ b.total_babies
  prob_one_speak (b : BabySpeaking 6) : ℝ :=
    (Nat.choose b.total_babies 1 : ℝ) * b.prob_speak * (b.prob_not_speak)^(b.total_babies - 1)
  prob_two_speak (b : BabySpeaking 6) : ℝ :=
    (Nat.choose b.total_babies 2 : ℝ) * (b.prob_speak^2) * (b.prob_not_speak^(b.total_babies - 2))

theorem at_least_three_babies_speak (b : BabySpeaking 6) (h : b.prob_speak = 1/3) (h' : b.prob_not_speak = 2/3) : 
  prob_at_least_three_speak b = 353 / 729 := by
  let p_none := prob_none_speak b
  let p_one := prob_one_speak b
  let p_two := prob_two_speak b
  let p_fewer_than_three := p_none + p_one + p_two
  have h1 : p_none = (2/3)^6 := sorry
  have h2 : p_one = (Nat.choose 6 1 : ℝ) * (1/3) * (2/3)^5 := sorry
  have h3 : p_two = (Nat.choose 6 2 : ℝ) * (1/3)^2 * (2/3)^4 := sorry
  have h4 : p_fewer_than_three = ((64 : ℝ) / 729) + 6 * (1/3) * (32/243) + 15 * (1/9) * (16/81) := sorry
  have h5 : p_fewer_than_three = 376 / 729 := sorry
  have h6 : prob_at_least_three_speak b = 1 - 376 / 729 := by
    exact sorry
  show prob_at_least_three_speak b = 353 / 729 from sorry

end at_least_three_babies_speak_l662_662829


namespace eigenvalues_of_matrix_A_l662_662078

noncomputable def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![ -1/4,  3/4],
    ![ 1/2, -1/2 ]]

theorem eigenvalues_of_matrix_A :
  let A : Matrix (Fin 2) (Fin 2) ℚ := Matrix.inverse A_inv
  eigenvalues A = set_of (-1, 4) :=
by
  sorry

end eigenvalues_of_matrix_A_l662_662078


namespace rhombus_area_equivalence_l662_662329

noncomputable def area_of_rhombus (side_length : ℝ) : ℝ :=
  let height := (sqrt 3 / 2) * side_length
  let d1 := side_length
  let d2 := height
  (1 / 2) * d1 * d2

theorem rhombus_area_equivalence :
  area_of_rhombus 4 = 4 * sqrt 3 :=
by sorry

end rhombus_area_equivalence_l662_662329


namespace castle_impossible_traverse_l662_662298

theorem castle_impossible_traverse (m n : ℕ) (central_pool : Prop) (H : m = 7 ∧ n = 9 ∧ central_pool) :
  ¬∃ path : List (ℕ × ℕ), (∀ room ∈ path, room.1 ≤ m ∧ room.2 ≤ n) ∧ 
    path.nodup ∧ (∀ i ∈ (List.init path), List.nth_le path i _ ↔ List.nth_le path (i+1) _) ∧
    (4, 5) ∉ path ∧ List.length path = (m * n - 1) :=
begin
  sorry
end

end castle_impossible_traverse_l662_662298


namespace females_dont_listen_correct_l662_662997

/-- Number of males who listen to the station -/
def males_listen : ℕ := 45

/-- Number of females who don't listen to the station -/
def females_dont_listen : ℕ := 87

/-- Total number of people who listen to the station -/
def total_listen : ℕ := 120

/-- Total number of people who don't listen to the station -/
def total_dont_listen : ℕ := 135

/-- Number of females surveyed based on the problem description -/
def total_females_surveyed (total_peoples_total : ℕ) (males_dont_listen : ℕ) : ℕ := 
  total_peoples_total - (males_listen + males_dont_listen)

/-- Number of females who listen to the station -/
def females_listen (total_females : ℕ) : ℕ := total_females - females_dont_listen

/-- Proof that the number of females who do not listen to the station is 87 -/
theorem females_dont_listen_correct 
  (total_peoples_total : ℕ)
  (males_dont_listen : ℕ)
  (total_females := total_females_surveyed total_peoples_total males_dont_listen)
  (females_listen := females_listen total_females) :
  females_dont_listen = 87 :=
sorry

end females_dont_listen_correct_l662_662997


namespace abs_triangle_inequality_abs_diff_inequality_abs_nested_inequality_l662_662181

theorem abs_triangle_inequality (x y : ℝ) : |x + y| ≤ |x| + |y| :=
by
  sorry

theorem abs_diff_inequality (x y : ℝ) : |x - y| ≥ |x| - |y| :=
by
  sorry

theorem abs_nested_inequality (x y : ℝ) : |x - y| ≥ | |x| - |y| | :=
by
  sorry

end abs_triangle_inequality_abs_diff_inequality_abs_nested_inequality_l662_662181


namespace find_S_four_twelve_sixteen_l662_662239

variable {R : Type*} [Field R]
variable {V : Type*} [AddCommGroup V] [Module R V]
variable (S : V → V)
variable (a b : R) (u v : V)

-- Given conditions
def cond1 : Prop := ∀ (a : R) (u v : V), S (a • u + b • v) = a • S u + b • S v
def cond2 : Prop := ∀ (u v : V), S (u × v) = S u × S v
def cond3 : Prop := S ⟨8, 8, 4⟩ = ⟨5, 0, 10⟩
def cond4 : Prop := S ⟨-8, 4, 8⟩ = ⟨5, 10, 0⟩

-- Prove the required transformation
theorem find_S_four_twelve_sixteen :
  cond1 S → cond2 S → cond3 S → cond4 S →
  S ⟨4, 12, 16⟩ = ⟨20, 0, 20⟩ :=
  by intros; sorry

end find_S_four_twelve_sixteen_l662_662239


namespace melanie_coins_and_amount_l662_662916

def Melanie_initial_coins := (19, 12, 8) -- (dimes, nickels, quarters)
def Melanie_dad_coins := (39, 22, 15)
def Melanie_sister_coins := (15, 7, 12)
def Melanie_mother_coins := (25, 10, 0)
def Melanie_grandmother_coins := (0, 30, 3)

def Total_coins := (98, 81, 38) -- (dimes, nickels, quarters)

noncomputable def Dimes_spent := Int.floor(0.10 * 98)
noncomputable def Quarters_spent := Int.floor(0.25 * 38)

def Dimes_left := 98 - Dimes_spent
def Nickels_left := 81
def Quarters_left := 38 - Quarters_spent

noncomputable def Total_dollars := (Dimes_left * 0.10) + (Nickels_left * 0.05) + (Quarters_left * 0.25)

theorem melanie_coins_and_amount :
  Melanie_initial_coins = (19, 12, 8) ∧ 
  Melanie_dad_coins = (39, 22, 15) ∧ 
  Melanie_sister_coins = (15, 7, 12) ∧ 
  Melanie_mother_coins = (25, 10, 0) ∧ 
  Melanie_grandmother_coins = (0, 30, 3) →
  Total_coins = (19 + 39 + 15 + 25, 12 + 22 + 7 + 10 + 30, 8 + 15 + 12 + 3) ∧
  Dimes_left = 89 ∧ 
  Nickels_left = 81 ∧ 
  Quarters_left = 29 ∧ 
  Total_dollars = 20.20 :=
by
  intros _
  simp [Melanie_initial_coins, Melanie_dad_coins, Melanie_sister_coins, Melanie_mother_coins, Melanie_grandmother_coins, Total_coins, Dimes_spent, Quarters_spent, Dimes_left, Nickels_left, Quarters_left, Total_dollars]
  sorry

end melanie_coins_and_amount_l662_662916


namespace highest_score_is_174_l662_662963

theorem highest_score_is_174
  (avg_40_innings : ℝ)
  (highest_exceeds_lowest : ℝ)
  (avg_excl_two : ℝ)
  (total_runs_40 : ℝ)
  (total_runs_38 : ℝ)
  (sum_H_L : ℝ)
  (new_avg_38 : ℝ)
  (H : ℝ)
  (L : ℝ)
  (H_eq_L_plus_172 : H = L + 172)
  (total_runs_40_eq : total_runs_40 = 40 * avg_40_innings)
  (total_runs_38_eq : total_runs_38 = 38 * new_avg_38)
  (sum_H_L_eq : sum_H_L = total_runs_40 - total_runs_38)
  (new_avg_eq : new_avg_38 = avg_40_innings - 2)
  (sum_H_L_val : sum_H_L = 176)
  (avg_40_val : avg_40_innings = 50) :
  H = 174 :=
sorry

end highest_score_is_174_l662_662963


namespace trigonometric_identity_l662_662415

theorem trigonometric_identity (α : ℝ) (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (3 * Real.cos α + 3 * Real.sin α) = 2 / 3 :=
by
  sorry

end trigonometric_identity_l662_662415


namespace problem_1_problem_2_l662_662773

section SetOperations

open Set

variables { ℝ : Type* }
noncomputable theory

-- Definitions for conditions
def A := {x : ℝ | ∃ y, y = log (x^2 - 2 * x - 3) / log 2 ∧ y ∈ Real}
def B (m : ℝ) := {x : ℝ | m - 1 ≤ x ∧ x ≤ 2 * m + 1}

-- Proof (1)
theorem problem_1 (m : ℝ) (h : m = -1) :
  A ∩ B m = {x : ℝ | -2 ≤ x ∧ x < -1} ∧ (ℝ ∖ A) ∪ B m = Icc (-2 : ℝ) 3 :=
sorry

-- Proof (2)
theorem problem_2 (h : A ⊆ (ℝ ∖ B 0)) :
  0 ≤ 0 ∧ 0 ≤ 1 :=
sorry

end SetOperations

end problem_1_problem_2_l662_662773


namespace ladder_distance_l662_662584

theorem ladder_distance (θ : ℝ) (H : θ = 60 * (π / 180)) (ladder_length : ℝ) (L : ladder_length = 9.2) : 
  ∃ d : ℝ, d = 4.6 ∧ cos θ = d / ladder_length :=
by 
  -- Given that θ is 60 degrees and ladder_length is 9.2,
  -- We need to show that d is 4.6
  use 4.6
  split
  · exact rfl
  · rw [H, L]
    norm_num
    sorry

end ladder_distance_l662_662584


namespace smaller_angle_at_315_l662_662785

def full_circle_degrees : ℝ := 360
def hours_on_clock : ℕ := 12
def degrees_per_hour : ℝ := full_circle_degrees / hours_on_clock
def minute_position_at_315 : ℝ := 3 * degrees_per_hour
def hour_position_at_315 : ℝ := 3 * degrees_per_hour + degrees_per_hour / 4

theorem smaller_angle_at_315 :
  minute_position_at_315 = 90 → 
  hour_position_at_315 = 3 * degrees_per_hour + degrees_per_hour / 4 → 
  abs (hour_position_at_315 - minute_position_at_315) = 7.5 :=
by 
  intro h_minute h_hour 
  rw [h_minute, h_hour]
  sorry

end smaller_angle_at_315_l662_662785


namespace standard_eq_line_l_cartesian_eq_curve_C_distance_PA_PB_l662_662467
noncomputable def parametric_eq (t : ℝ) : ℝ × ℝ :=
  (-1 + (real.sqrt 2 / 2) * t, (real.sqrt 2 / 2) * t)

def polar_eq (θ : ℝ) : ℝ :=
  6 * real.cos θ

theorem standard_eq_line_l : ∀ (t : ℝ),
  let (x, y) := parametric_eq t in
  x - y + 1 = 0 :=
sorry

theorem cartesian_eq_curve_C : ∀ (ρ θ : ℝ),
  ρ = polar_eq θ →
  let x := ρ * real.cos θ
  let y := ρ * real.sin θ in
  (x - 3)^2 + y^2 = 9 :=
sorry

theorem distance_PA_PB : ∀ (t : ℝ) (P A B : ℝ × ℝ),
  P = (-1, 0) →
  let (x_A, y_A) := parametric_eq t
  let (x_B, y_B) := parametric_eq (-t) in
  A = (x_A, y_A) →
  B = (x_B, y_B) →
  abs (dist P A) + abs (dist P B) = 4 * real.sqrt 2 :=
sorry

end standard_eq_line_l_cartesian_eq_curve_C_distance_PA_PB_l662_662467


namespace incorrect_option_C_l662_662233

theorem incorrect_option_C (y : ℝ → ℝ)
  (h₀ : y 0 = 331)
  (h₅ : y 5 = 334)
  (h₁₀ : y 10 = 337)
  (h₁₅ : y 15 = 340)
  (h₂₀ : y 20 = 343) :
  ¬ (y 15 = 343) := 
by
  -- Assume for contradiction that y 15 = 343
  intro h
  -- Derive contradiction from given data
  have : 340 = 343 := by
    rw [h₁₅, h]
  -- This is a contradiction since 340 ≠ 343
  exact ne_of_lt (by linarith) this

#check incorrect_option_C

end incorrect_option_C_l662_662233


namespace sin_double_angle_l662_662417

variable (θ : ℝ)

-- Given condition: tan(θ) = -3/5
def tan_theta : Prop := Real.tan θ = -3/5

-- Target to prove: sin(2θ) = -15/17
theorem sin_double_angle : tan_theta θ → Real.sin (2*θ) = -15/17 :=
by
  sorry

end sin_double_angle_l662_662417


namespace concyclic_points_D_N_K_E_l662_662549

-- Define the conditions for the problem
variables {A B C D E F K M N T : Point}
variables (ω : Circle)
variables [cyclic_quadrilateral : CyclicQuadrilateral B C E D]
variables [meet_CB_ED_at_A : Meet CB ED A]
variables [DF_parallel_BC : Parallel (LineThrough D F) (LineThrough B C)]
variables [F_on_ω : OnCircle F ω]
variables [meet_AF_T_on_ω : Meet (Segment A F) T]
variables [meet_ET_BC_at_M : Meet (LineThrough E T) (LineThrough B C) M]
variables [K_midpoint_BC : Midpoint K B C]
variables [N_reflection_A_about_M : Reflection N A M]

-- State the theorem to prove D, N, K, E are concyclic
theorem concyclic_points_D_N_K_E
  (h1 : CyclicQuadrilateral B C E D)
  (h2 : Meet CB ED A)
  (h3 : Parallel (LineThrough D F) (LineThrough B C))
  (h4 : OnCircle F ω)
  (h5 : Meet (Segment A F) T)
  (h6 : Meet (LineThrough E T) (LineThrough B C) M)
  (h7 : Midpoint K B C)
  (h8 : Reflection N A M) : 
  Concyclic D N K E := 
sorry

end concyclic_points_D_N_K_E_l662_662549


namespace urn_final_state_l662_662136

structure Urn :=
(white : ℕ)
(black : ℕ)

def operation1 (u : Urn) : Urn :=
⟨u.white, u.black - 1⟩

def operation2 (u : Urn) : Urn :=
⟨u.white + 1, u.black - 2⟩

def operation3 (u : Urn) : Urn :=
⟨u.white - 1, u.black⟩

def operation4 (u : Urn) : Urn :=
⟨u.white - 3, u.black + 1⟩

theorem urn_final_state : 
  ∃ (f : Urn → Urn), 
  (f ⟨120, 80⟩ = ⟨2, 1⟩) ∧
  (∀ u, f u = operation1 u ∨ f u = operation2 u ∨ f u = operation3 u ∨ f u = operation4 u) :=
begin
  sorry
end

end urn_final_state_l662_662136


namespace count_primes_in_list_l662_662088

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

def list_of_numbers : List ℕ := [1, 12, 123, 1234, 12345, 123456]

theorem count_primes_in_list : list_of_numbers.filter is_prime = [] :=
by
  sorry

end count_primes_in_list_l662_662088


namespace percentage_of_black_cats_is_25_l662_662464

theorem percentage_of_black_cats_is_25 :
  let total_cats := 16 in
  let white_cats := 2 in
  let grey_cats := 10 in
  let black_cats := total_cats - white_cats - grey_cats in
  let percentage_black_cats := (black_cats / total_cats : ℝ) * 100 in
  percentage_black_cats = 25 :=
by
  sorry

end percentage_of_black_cats_is_25_l662_662464


namespace only_solution_is_two_l662_662374

theorem only_solution_is_two :
  ∀ n : ℕ, (Nat.Prime (n^n + 1) ∧ Nat.Prime ((2*n)^(2*n) + 1)) → n = 2 :=
by
  sorry

end only_solution_is_two_l662_662374


namespace smallest_x_mul_900_multiple_of_1152_l662_662262

theorem smallest_x_mul_900_multiple_of_1152 : 
  ∃ x : ℕ, (x > 0) ∧ (900 * x) % 1152 = 0 ∧ ∀ y : ℕ, (y > 0) ∧ (900 * y) % 1152 = 0 → y ≥ x := 
begin
  use 32,
  split,
  { exact nat.one_pos, }, -- the positive condition
  split,
  { change (900 * 32) % 1152 = 0, -- 32 satisfies the multiple condition
    norm_num, 
  },
  { intros y hy, -- minimality condition
    cases hy with hy_pos hy_dvd,
    have : 1152 ∣ 900 * 32 := by {
        change 900 * 32 % 1152 = 0,
        norm_num,
    },
    obtain ⟨k, hk⟩ := exists_eq_mul_left_of_dvd this,
    change 1152 * k < 1152 * 32,
    refine le_of_dvd (mul_pos (@nat.one_pos _) hy_pos) _,
    exact ⟨32, rfl⟩,
  },
end

end smallest_x_mul_900_multiple_of_1152_l662_662262


namespace infinitely_many_n_divisible_by_5_and_13_l662_662937

theorem infinitely_many_n_divisible_by_5_and_13 :
  ∀ k : ℤ, (let n := 65 * k + 4 in (4 * n^2 + 1) % 65 = 0) :=
by 
  sorry

end infinitely_many_n_divisible_by_5_and_13_l662_662937


namespace circle_properties_l662_662490

noncomputable theory

def polar_point := (ρ : ℝ, θ : ℝ)

def polar_circle (C : ℝ × ℝ → Prop) :=
  ∃ (P : ℝ × ℝ), P = (sqrt 2, π / 4) ∧ C (sqrt 2) (π / 4)

def circle_center := { C | ∃ O, O = (1, 0) }

def polar_line := { L | ∃ θ ρ, θ = π / 3 ∧ ρ ∈ ℝ }

theorem circle_properties : (∃ C, polar_circle C) → (∃ C, circle_center C) →
  (C (ρ θ : ℝ), polar_circle ⟨1, π / 4⟩) ∧ 
  (polar_center (intersects polar_axis $\polar_line$ (θ - π / 3) = -sqrt 3 / 2)) →
  (2 ⋅ cos θ = ρ) ∧ 
  (polar_chord_length (Line (polar_circle_center ∘ intersects) (Line θ = π / 3)) → length = 1 := 
  by sorry

end circle_properties_l662_662490


namespace remove_and_replace_volume_needed_l662_662447

-- Define the initial conditions
def initial_volume : ℝ := 80
def initial_concentration : ℝ := 0.20
def final_volume : ℝ := 80
def final_concentration : ℝ := 0.40

-- Define the acid volumes based on initial conditions
def initial_acid_volume : ℝ := initial_volume * initial_concentration
def final_acid_volume : ℝ := final_volume * final_concentration

-- The Lean 4 theorem to state and prove
theorem remove_and_replace_volume_needed :
  ∃ x : ℝ, (initial_acid_volume - x * initial_concentration + x = final_acid_volume) ∧ x = 20 :=
by
  -- Initial acid volume
  have h1 : initial_acid_volume = 16 := by sorry
  -- Final acid volume
  have h2 : final_acid_volume = 32 := by sorry
  -- Set up and solve the equation
  have h3 : 16 - x * 0.20 + x = 32 := by sorry
  use (16 / 0.80)
  have h4 : 16 / 0.80 = 20 := by sorry
  exact ⟨rfl, h4⟩

end remove_and_replace_volume_needed_l662_662447


namespace arithmetic_and_geometric_sums_l662_662038

-- Definition of the Arithmetic sequence
def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ := a₁ + (n - 1) * d

-- Given conditions
axiom a₁ : ℕ := 1
axiom S₁₀₀ : ℕ := 10000

-- Define the sum of the first n terms of an arithmetic sequence
def arithmetic_sum (a₁ : ℕ) (aₙ : ℕ) (n : ℕ) : ℕ := (n * (a₁ + aₙ)) / 2

-- General term of the Arithmetic sequence
axiom general_term (d n : ℕ) : ℕ := 2 * n - 1

-- Sequence bn
def b_sequence (n : ℕ) : ℕ := 2 ^ (general_term d n + 1)

-- Sum of geometric series
def geometric_sum (a r n : ℕ) : ℕ := a * (1 - r^n) / (1 - r)

-- Main proof statement
theorem arithmetic_and_geometric_sums :
  let a_n := general_term 2 in
  let b_n := b_sequence in
  let S_n := geometric_sum 4 4 n in
  S₁₀₀ = 10000 → -- given condition
  S_n = 4*(4^n - 1) / 3 := 
by
  sorry

end arithmetic_and_geometric_sums_l662_662038


namespace smallest_angle_mul_6006_eq_154440_l662_662341

noncomputable def smallest_angle_mul_result (ABC : Type) [IsoscelesTriangle ABC] : ℝ :=
  let α := (triangleAngleA ABC : ℝ)
  let β := 180 - 2 * α
  let smallest_angle := if (β < min (2 * α) (180 / 7)) then min (2 * α) (180 / 7) else β
  smallest_angle * 6006

theorem smallest_angle_mul_6006_eq_154440 (ABC : Type) [h : IsoscelesTriangle ABC] : 
  smallest_angle_mul_result ABC = 154440 :=
  by 
    sorry

end smallest_angle_mul_6006_eq_154440_l662_662341


namespace problem_statement_l662_662468

-- Define the point P and other necessary calculations
def point_P : ℝ × ℝ := (1, 2)

-- Define the radius r of the circle
noncomputable def r (P : ℝ × ℝ) : ℝ := Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

-- Define sin and cos using the coordinates of point P
noncomputable def sin_alpha (P : ℝ × ℝ) : ℝ := P.2 / r P
noncomputable def cos_alpha (P : ℝ × ℝ) : ℝ := P.1 / r P

-- The first problem: to prove tan α = 2
def tan_alpha (P : ℝ × ℝ) : ℝ := P.2 / P.1

-- The second problem: to prove the given expression equals 4/3
noncomputable def given_expression (P : ℝ × ℝ) : ℝ := 
  (sin_alpha P + 2 * cos_alpha P) / (2 * sin_alpha P - cos_alpha P)

-- Problem: Prove the two statements
theorem problem_statement : 
  (tan_alpha point_P = 2) ∧ (given_expression point_P = 4 / 3) := 
by 
  sorry

end problem_statement_l662_662468


namespace triangle_area_AOB_l662_662055

noncomputable def area_of_triangle (z1 z2 : ℂ) : ℝ :=
  (1 / 2) * abs (z1 * z2 * complex.sin (real.pi / 3))

theorem triangle_area_AOB (z1 z2 : ℂ)
  (h1 : abs z2 = 4)
  (h2 : 4 * z1 ^ 2 - 2 * z1 * z2 + z2 ^ 2 = 0) :
  area_of_triangle z1 z2 = 2 * real.sqrt 3 := 
by {
  sorry
}

end triangle_area_AOB_l662_662055


namespace range_of_phi_l662_662429

theorem range_of_phi (φ : ℝ) :
  (∀ x y, (π/5 < x ∧ x < 5*π/8) ∧ (π/5 < y ∧ y < 5*π/8) → 
    (x < y → -2 * sin (2 * x + φ) < -2 * sin (2 * y + φ))) ∧ 
    (abs φ < π)  →
  (π/10 ≤ φ ∧ φ ≤ π/4) :=
sorry

end range_of_phi_l662_662429


namespace solve_ineq_l662_662574

theorem solve_ineq (x : ℝ) (hx1 : -1 < x) (hx2 : x < 1) (hx3 : x ≠ 0) (hx4 : x ≠ -1) (hx5 : x ≠ 1/4) : 
x ∈ Icc (-1/2) 0 ∪ Ioc 0 (1/5) ∪ Icc (1/4) (1/2) :=
sorry

end solve_ineq_l662_662574


namespace ratio_of_segments_is_one_five_l662_662970

noncomputable def ratio_of_segments (p q : ℕ) : Prop :=
  p < q ∧ Nat.coprime p q ∧
  (∃ (x1 x2 x3 : ℝ), x1 = 60 ∧ x2 = 120 ∧ x3 = 420 ∧
   sin x1 = sin 60 ∧ sin x2 = sin 60 ∧ sin x3 = sin 60 ∧
   (p : ℝ) / (q : ℝ) = (x2 - x1) / (x3 - x2))

theorem ratio_of_segments_is_one_five : ratio_of_segments 1 5 := by
  sorry

end ratio_of_segments_is_one_five_l662_662970


namespace ab_value_l662_662931

theorem ab_value (A B C D E : Point) (x : ℝ)
  (h1 : lies_on_line [A, B, C, D])
  (h2 : dist A B = x)
  (h3 : dist C D = x)
  (h4 : dist B C = 8)
  (h5 : ∀ {P}, ¬ lies_on_line [P, E])
  (h6 : dist B E = 8)
  (h7 : dist C E = 8)
  (h8 : (2 * sqrt (x^2 + 8 * x + 64) + 2 * x + 8) = 3 * (8 + 8 + 8)) :
  x = 40 / 3 :=
by
  sorry

end ab_value_l662_662931


namespace sum_of_x0_values_tangent_to_circle_l662_662972

theorem sum_of_x0_values_tangent_to_circle :
  let circle_eq (x y : ℝ) := (x - 4)^2 + y^2
  let elliptic_curve_eq (x y : ℝ) := y^2 = x^3 + 1
  let is_tangent (x y : ℝ) := circle_eq x y = r^2 ∧ elliptic_curve_eq x y = y^2 ∧
                               ((diff circle_eq) x y 0) = (diff elliptic_curve_eq) x y 0
  let sum_x0 := ∑ x0 in {x | ∃ y0, is_tangent x0 y0}, x0
  in sum_x0 = 1 / 3 :=
by sorry

end sum_of_x0_values_tangent_to_circle_l662_662972


namespace visible_steps_on_escalator_l662_662672

variable (steps_visible : ℕ) -- The number of steps visible on the escalator
variable (al_steps : ℕ := 150) -- Al walks down 150 steps
variable (bob_steps : ℕ := 75) -- Bob walks up 75 steps
variable (al_speed : ℕ := 3) -- Al's walking speed
variable (bob_speed : ℕ := 1) -- Bob's walking speed
variable (escalator_speed : ℚ) -- The speed of the escalator

theorem visible_steps_on_escalator : steps_visible = 120 :=
by
  -- Define times taken by Al and Bob
  let al_time := al_steps / al_speed
  let bob_time := bob_steps / bob_speed

  -- Define effective speeds considering escalator speed 'escalator_speed'
  let al_effective_speed := al_speed - escalator_speed
  let bob_effective_speed := bob_speed + escalator_speed

  -- Calculate the total steps walked if the escalator was stopped (same total steps)
  have al_total_steps := al_effective_speed * al_time
  have bob_total_steps := bob_effective_speed * bob_time

  -- Set up the equation
  have eq := al_total_steps = bob_total_steps

  -- Substitute and solve for escalator_speed
  sorry

end visible_steps_on_escalator_l662_662672


namespace fibonacci_property_l662_662144

open Nat

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci (n + 1) + fibonacci n

noncomputable def polynomial : ℕ → ℝ := sorry -- Placeholder for the polynomial of degree 1006

theorem fibonacci_property (f : ℕ → ℝ) (h_poly_deg : nat_degree f = 1006)
    (h_fib_eq : ∀ k, 1008 ≤ k ∧ k ≤ 2014 → f k = fibonacci k) :
    233 ∣ f 2015 + 1 :=
sorry

end fibonacci_property_l662_662144


namespace shares_difference_l662_662677

-- conditions: the ratio is 3:7:12, and the difference between q and r's share is Rs. 3000
theorem shares_difference (x : ℕ) (h : 12 * x - 7 * x = 3000) : 7 * x - 3 * x = 2400 :=
by
  -- simply skip the proof since it's not required in the prompt
  sorry

end shares_difference_l662_662677


namespace problem_conditions_l662_662356

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x
def is_increasing_function (f : ℝ → ℝ) : Prop := ∀ x y, x < y → f x < f y

variables {f g : ℝ → ℝ} {a b : ℝ}

theorem problem_conditions (h_odd_f : is_odd_function f) (h_increasing_f : is_increasing_function f) 
(h_even_g : is_even_function g) (h_coincide : ∀ x, x ≥ 0 → g x = f x) (h_pos : a > b) (h_pos_b : b > 0) : 
  (f b - f (-a) > g a - g (-b)) ∧ (f a - f (-b) < g b - g (-a)) :=
begin
  -- proof here
  sorry
end

end problem_conditions_l662_662356


namespace parabola_sum_l662_662647

theorem parabola_sum (a b c : ℝ)
  (h_vertex : ∀ x : ℝ, -3 = (-b / (2*a)))
  (h_pass_point : ∀ x : ℝ, y = ax^2 + bx + c) :
  let y := a * (1:ℝ)^2 + b * (1:ℝ) + c in
  y = (6:ℝ) →
  a + b + c = 6 :=
by
  sorry

end parabola_sum_l662_662647


namespace ratio_is_three_to_one_l662_662310

-- Define the number of households surveyed
def households_surveyed : ℕ := 240

-- Define the number of households that used neither brand A nor brand B soap
def neither_brand : ℕ := 80

-- Define the number of households that used only brand A soap
def only_brand_A : ℕ := 60

-- Define the number of households that used both brand A and brand B soap
def both_brands : ℕ := 25

-- Define a function to calculate the number of households that used only brand B soap
def only_brand_B (total : ℕ) (neither : ℕ) (only_A : ℕ) (both : ℕ) : ℕ := total - neither - only_A - both

-- Calculate the number of households that used only brand B soap
def households_only_brand_B : ℕ := only_brand_B households_surveyed neither_brand only_brand_A both_brands

-- Define the ratio of households that used only brand B soap to households that used both brands
def ratio_only_brand_B_to_both (only_B : ℕ) (both : ℕ) : ℕ := only_B / both

-- Prove the ratio is 3:1
theorem ratio_is_three_to_one : ratio_only_brand_B_to_both households_only_brand_B both_brands = 3 := by
  dsimp [households_only_brand_B, only_brand_B, ratio_only_brand_B_to_both, households_surveyed, neither_brand, only_brand_A, both_brands]
  sorry

end ratio_is_three_to_one_l662_662310


namespace smaller_angle_at_315_l662_662783

def full_circle_degrees : ℝ := 360
def hours_on_clock : ℕ := 12
def degrees_per_hour : ℝ := full_circle_degrees / hours_on_clock
def minute_position_at_315 : ℝ := 3 * degrees_per_hour
def hour_position_at_315 : ℝ := 3 * degrees_per_hour + degrees_per_hour / 4

theorem smaller_angle_at_315 :
  minute_position_at_315 = 90 → 
  hour_position_at_315 = 3 * degrees_per_hour + degrees_per_hour / 4 → 
  abs (hour_position_at_315 - minute_position_at_315) = 7.5 :=
by 
  intro h_minute h_hour 
  rw [h_minute, h_hour]
  sorry

end smaller_angle_at_315_l662_662783


namespace geometric_sequence_common_ratio_l662_662126

theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ)
    (h1 : a 2 = a 1 * q)
    (h2 : a 5 = a 1 * q ^ 4)
    (h3 : a 2 = 8)
    (h4 : a 5 = 64) :
    q = 2 := 
sorry

end geometric_sequence_common_ratio_l662_662126


namespace total_chairs_taken_l662_662514

theorem total_chairs_taken (students trips chairs_per_trip : ℕ) (h_students : students = 5) (h_trips : trips = 10) (h_chairs_per_trip : chairs_per_trip = 5) : 
  students * trips * chairs_per_trip = 250 := by
  rw [h_students, h_trips, h_chairs_per_trip]
  norm_num

end total_chairs_taken_l662_662514


namespace pascals_triangle_sum_l662_662795

theorem pascals_triangle_sum : (∑ n in Finset.range 30, n + 1) = 465 := by
  sorry

end pascals_triangle_sum_l662_662795


namespace isosceles_triangle_perimeter_l662_662848

-- An auxiliary definition to specify that the triangle is isosceles
def is_isosceles (a b c : ℕ) :=
  a = b ∨ b = c ∨ c = a

-- The main theorem statement
theorem isosceles_triangle_perimeter (a b : ℕ) (h₁ : a = 3) (h₂ : b = 6) (h₃ : is_isosceles a b 6): a + b + 6 = 15 :=
by
  -- the proof would go here
  sorry

end isosceles_triangle_perimeter_l662_662848


namespace a_n_formula_T_n_formula_l662_662167

-- Definitions for the sequence
def a : ℕ → ℕ
| 0       := 0
| 1       := 1
| (n + 1) := 2 * (finset.sum (finset.range n) a) + 1

-- Definitions for the sequence of absolute differences
def b (n : ℕ) : ℕ := |a n - n - 2|

-- Sum of the first n terms function
def S (n : ℕ) : ℕ := finset.sum (finset.range n) a

-- Sum of the first n terms of b_n
def T : ℕ → ℕ
| 0       := 0
| 1       := 2
| n@(m+2) := 3 + (9 * (1 - 3^(m-1)) / (1 - 3)) - ((m - 1) * (m + 7)) / 2

/-- Prove the given theorem -/
theorem a_n_formula (n : ℕ) : a n = 3^(n-1) := sorry

theorem T_n_formula : ∀ (n : ℕ), T n = (if n = 1 then 2 else (3^n - n^2 - 5 * n + 11) / 2) := sorry

end a_n_formula_T_n_formula_l662_662167


namespace triangle_ABC_area_l662_662865

theorem triangle_ABC_area (area_WXYZ : ℕ) (side_small_squares : ℕ) (isosceles_AB_AC : AB = AC)
  (A_folds_to_O : folds A BC O) : 
  area_WXYZ = 64 ∧ side_small_squares = 2 → 
  area_triangle ABC = 16 :=
by {
  sorry
}

end triangle_ABC_area_l662_662865


namespace not_perfect_square_l662_662178

theorem not_perfect_square (p : ℕ) (hp : Nat.Prime p) : ¬ ∃ t : ℕ, 7 * p + 3^p - 4 = t^2 :=
sorry

end not_perfect_square_l662_662178


namespace range_of_possible_slopes_l662_662437

open Real

def hyperbola (x y : ℝ) : Prop := (x^2) / 4 - (y^2) / 12 = 1

def right_focus (a b : ℝ) := (a^2 + b^2)^(1/2)

-- Range of possible slopes for line passing through the right focus
theorem range_of_possible_slopes :
  ∃ F : ℝ × ℝ, right_focus F.1 F.2 ∧ (∀ (m : ℝ), (F.2 = m * F.1) → m ∈ set.Icc (-sqrt (3 : ℝ)) (sqrt (3 : ℝ))) :=
sorry

end range_of_possible_slopes_l662_662437


namespace smaller_angle_at_315_l662_662782

def full_circle_degrees : ℝ := 360
def hours_on_clock : ℕ := 12
def degrees_per_hour : ℝ := full_circle_degrees / hours_on_clock
def minute_position_at_315 : ℝ := 3 * degrees_per_hour
def hour_position_at_315 : ℝ := 3 * degrees_per_hour + degrees_per_hour / 4

theorem smaller_angle_at_315 :
  minute_position_at_315 = 90 → 
  hour_position_at_315 = 3 * degrees_per_hour + degrees_per_hour / 4 → 
  abs (hour_position_at_315 - minute_position_at_315) = 7.5 :=
by 
  intro h_minute h_hour 
  rw [h_minute, h_hour]
  sorry

end smaller_angle_at_315_l662_662782


namespace sum_of_odd_numbers_group_l662_662945

theorem sum_of_odd_numbers_group (n : ℕ) : 
  let odd_series := λ (k : ℕ), 2 * k + 1
  in let sum_of_n_terms := ∑ i in finset.range n, odd_series (i + finset.range.sum.to_nat (i))
  in sum_of_n_terms = n^3 := 
sorry

end sum_of_odd_numbers_group_l662_662945


namespace common_difference_arithmetic_sequence_l662_662861

theorem common_difference_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ)
  (h1 : a 5 = 10) (h2 : a 12 = 31) : d = 3 :=
by
  sorry

end common_difference_arithmetic_sequence_l662_662861


namespace variances_equal_thirtieth_percentile_y_l662_662742

noncomputable def sample_data_x (i : ℕ) (h : 1 ≤ i ∧ i ≤ 10) : ℝ := 2 * i

def sample_data_y (i : ℕ) (h : 1 ≤ i ∧ i ≤ 10) : ℝ := sample_data_x i h - 20

def mean (data : ℕ → ℝ) (n : ℕ) : ℝ :=
  (∑ i in finset.range (n + 1), data (i + 1) sorry) / n

def variance (data : ℕ → ℝ) (n : ℕ) : ℝ :=
  let mean_val := mean data n in
  (∑ i in finset.range (n + 1), ((data (i + 1) sorry) - mean_val) ^ 2) / n

def percentile (data : ℕ → ℝ) (n : ℕ) (p : ℝ) : ℝ := 
  let sorted_data := list.sort (≤) [data (i + 1) sorry | i in finset.range (n + 1)] in
  (sorted_data.nth ((n * p).toNat) + sorted_data.nth ((n * p).toNat + 1)) / 2

theorem variances_equal : variance sample_data_x 10 = variance sample_data_y 10 :=
  sorry

theorem thirtieth_percentile_y : percentile sample_data_y 10 0.3 = -13 :=
  sorry

end variances_equal_thirtieth_percentile_y_l662_662742


namespace math_proof_problem_part1_math_proof_problem_part2_l662_662127

-- Geometry setup
variables {A B C D : Type} [EuclideanPlane A] [EuclideanPlane B] [EuclideanPlane C] [EuclideanPlane D]
noncomputable def sin_angle_ADB (AB : ℝ) (angle_ADC : ℝ) (angle_A : ℝ) (cos_angle_ABD : ℝ) (area_BCD : ℝ) : Prop :=
  AB = 6 ∧ angle_ADC = Real.pi / 2 ∧ angle_A = Real.pi / 3 ∧ cos_angle_ABD = -1 / 7 ∧
  area_BCD = 39 * Real.sqrt 3 ∧ 
  sin (angle_A + angle_ABD) = 3 * Real.sqrt 3 / 14

noncomputable def length_BC (AB : ℝ) (angle_ADC : ℝ) (angle_A : ℝ) (cos_angle_ABD : ℝ) (area_BCD : ℝ) (BD : ℝ) : Prop :=
  AB = 6 ∧ angle_ADC = Real.pi / 2 ∧ angle_A = Real.pi / 3 ∧ cos_angle_ABD = -1 / 7 ∧ 
  area_BCD = 39 * Real.sqrt 3 ∧ BD = 14 ∧
  BC = 14

theorem math_proof_problem_part1 :
  sin_angle_ADB 6 (Real.pi / 2) (Real.pi / 3) (-1 / 7) (39 * Real.sqrt 3) :=
by sorry

theorem math_proof_problem_part2 :
  length_BC 6 (Real.pi / 2) (Real.pi / 3) (-1 / 7) (39 * Real.sqrt 3) 14 :=
by sorry

end math_proof_problem_part1_math_proof_problem_part2_l662_662127


namespace part_one_part_two_l662_662367

-- Part 1:
-- Define the function f
def f (x : ℝ) : ℝ := abs (2 * x - 3) + abs (2 * x + 2)

-- Define the inequality problem
theorem part_one (x : ℝ) : f x < x + 5 ↔ 0 < x ∧ x < 2 :=
by sorry

-- Part 2:
-- Define the condition for part 2
theorem part_two (a : ℝ) : (∀ x : ℝ, f x > a + 4 / a) ↔ (a ∈ Set.Ioo 1 4 ∨ a < 0) :=
by sorry

end part_one_part_two_l662_662367


namespace cos_value_in_second_quadrant_l662_662455

theorem cos_value_in_second_quadrant {B : ℝ} (h1 : π / 2 < B ∧ B < π) (h2 : Real.sin B = 5 / 13) : 
  Real.cos B = - (12 / 13) :=
sorry

end cos_value_in_second_quadrant_l662_662455


namespace sin_double_angle_eq_ratio_l662_662421

theorem sin_double_angle_eq_ratio
  (α : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hcos : cos (α + π / 6) = 4 / 5) :
  sin (2 * α + π / 3) = 24 / 25 := by
  sorry

end sin_double_angle_eq_ratio_l662_662421


namespace total_flowers_in_vase_l662_662385

-- Conditions as definitions
def num_roses : ℕ := 5
def num_lilies : ℕ := 2

-- Theorem statement
theorem total_flowers_in_vase : num_roses + num_lilies = 7 :=
by
  sorry

end total_flowers_in_vase_l662_662385


namespace SA_value_l662_662412

-- Definitions of the points S, A, B, and C on the sphere O
variables (S A B C : Point)
-- Definitions to express perpendicularity and distances
variables 
  (plane_ABC : Plane)
  (O : Sphere)
  (SA : ℝ) (AB : ℝ) (BC : ℝ)

-- Given conditions
noncomputable def given_conditions : Prop :=
  (dist S A = SA) ∧ (dist A B = 1) ∧ (dist B C = sqrt 2) ∧
  (is_on_sphere S O ∧ is_on_sphere A O ∧ is_on_sphere B O ∧ is_on_sphere C O) ∧
  (perpendicular SA plane_ABC) ∧ (perpendicular AB BC) ∧
  (surface_area O = 4 * real.pi)

-- The assertion we need to prove
theorem SA_value (h : given_conditions S A B C plane_ABC O SA AB BC) : SA = 1 := 
by sorry

end SA_value_l662_662412


namespace product_gcd_lcm_eq_1296_l662_662382

theorem product_gcd_lcm_eq_1296 : (Int.gcd 24 54) * (Int.lcm 24 54) = 1296 := by
  sorry

end product_gcd_lcm_eq_1296_l662_662382


namespace find_vector_b_l662_662428

def vector (α : Type*) := α × α

variables (α : Type*) [Real α]

def collinear (a b : vector α) : Prop :=
  ∃ (λ : α), b = (λ * a.1, λ * a.2)

def magnitude (v : vector α) : α :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (v1 v2 : vector α) : α :=
  v1.1 * v2.1 + v1.2 * v2.2

def is_acute (v1 v2 : vector α) : Prop :=
  dot_product v1 v2 > 0

constant a : vector ℝ := (-4, 3)
constant c : vector ℝ := (1, 1)
constant b : vector ℝ := (8, -6)

theorem find_vector_b (b : vector ℝ)
  (h1 : collinear a b)
  (h2 : magnitude b = 10)
  (h3 : is_acute b c) :
  b = (8, -6) :=
sorry

end find_vector_b_l662_662428


namespace sequence_general_formula_sum_of_squares_inequality_l662_662401

def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1
  else if n = 2 then 1 / 4
  else (n - 2) * sequence (n - 1) / (n - 1 - sequence (n - 1))

theorem sequence_general_formula (n : ℕ) (h : n > 0) : 
  sequence n = 1 / (3 * n - 2) :=
by
  sorry

theorem sum_of_squares_inequality (n : ℕ) (h : n > 0) :
  ∑ k in finset.range (n + 1), (sequence k)^2 < 7 / 6 :=
by
  sorry

end sequence_general_formula_sum_of_squares_inequality_l662_662401


namespace only_valid_n_l662_662008

theorem only_valid_n (n : ℕ) (h : n > 2) : 
  (n! ∣ ∏ (p q : ℕ) in (finset.Icc 2 n).filter (λ p, nat.prime p) × (finset.Icc 2 n).filter (λ q, nat.prime q), if p < q then p + q else 1) ↔ n = 7 :=
sorry

end only_valid_n_l662_662008


namespace ellipse_properties_line_passing_fixed_point_max_area_triangle_l662_662404

noncomputable def ellipse_params 
  (a b c : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ 2 * a = 8 ∧ a^2 = b^2 + c^2 ∧ c = 2

noncomputable def ellipse_eq 
  (a b : ℝ) : Prop :=
  a = 4 ∧ b = 2 * Real.sqrt 3

theorem ellipse_properties
  (a b c : ℝ)
  (params : ellipse_params a b c) :
  ellipse_eq a b ∧ c = 2 :=
sorry

noncomputable def fixed_point (E : Point) : Prop := 
  E.x = 5 ∧ E.y = 0

theorem line_passing_fixed_point 
  (a b : ℝ)
  (h: ∀ (A B : Point) (D : Point), LET (l: Line), line_passing_through_focus A B ∧ line_perpendicular_to_l A D l ∧ D.x = 8 → line_intersecting_fixed_point B D (Point.mk 5 0)) :
  fixed_point (Point.mk 5 0) :=
sorry

theorem max_area_triangle
  (O B D : Point)
  (h: ∀ (S : Triangle O B D) (Area_max: max_triangle_area_triangle S), max_area S = 15) : 
  max_area_triangle (Triangle.mk O B D) = 15 :=
sorry

end ellipse_properties_line_passing_fixed_point_max_area_triangle_l662_662404


namespace area_inequality_l662_662112

variable (N : Type) [convex N] (A B C D A' B' C' D' A1 B1 C1 D1 : N)
variable (midpoint_of : N → N → N) (intersection_of : N → N → N → N → N)
variable (T T1 : ℝ)

axiom quadrilateral_midpoints :
  midpoint_of B C = A' ∧ midpoint_of C D = B' ∧ midpoint_of D A = C' ∧ midpoint_of A B = D'

axiom lines_intersection :
  intersection_of A A' B B' = A1 ∧ intersection_of B B' C C' = B1 ∧
  intersection_of C C' D D' = C1 ∧ intersection_of D D' A A' = D1

axiom area_definitions :
  area A1 B1 C1 D1 = T1 ∧ area N = T

theorem area_inequality : 5 * T1 ≤ T ∧ T ≤ 6 * T1 := by
  sorry

end area_inequality_l662_662112


namespace lines_intersect_lines_parallel_lines_coincident_l662_662405

-- Define line equations
def l1 (m x y : ℝ) := (m + 2) * x + (m + 3) * y - 5 = 0
def l2 (m x y : ℝ) := 6 * x + (2 * m - 1) * y - 5 = 0

-- Prove conditions for intersection
theorem lines_intersect (m : ℝ) : ¬(m = -5 / 2 ∨ m = 4) ↔
  ∃ x y : ℝ, l1 m x y ∧ l2 m x y := sorry

-- Prove conditions for parallel lines
theorem lines_parallel (m : ℝ) : m = -5 / 2 ↔
  ∀ x y : ℝ, l1 m x y ∧ l2 m x y → l1 m x y → l2 m x y := sorry

-- Prove conditions for coincident lines
theorem lines_coincident (m : ℝ) : m = 4 ↔
  ∀ x y : ℝ, l1 m x y ↔ l2 m x y := sorry

end lines_intersect_lines_parallel_lines_coincident_l662_662405


namespace Randy_pictures_l662_662939

theorem Randy_pictures :
  ∃ R : ℕ, (let P := 8 in
            let Q := P + 20 in
            let S := R + 15 in
            R + P + Q + S = 75) ∧ R = 12 := 
by
  sorry

end Randy_pictures_l662_662939


namespace minimum_amount_for_class1_minimum_amount_for_grade_l662_662330

-- Define the conditions
def retail_price : ℕ → ℝ
| n := if n < 12 then 0.30 * n else if n <= 120 then (n / 12) * 3.00 else (n / 12) * 2.70 + (n % 12) * 0.30

-- Problem 1: Prove the minimum amount to pay for 57 notebooks is 14.7 yuan
theorem minimum_amount_for_class1 : retail_price 57 = 14.7 := 
sorry

-- Problem 2: Prove the minimum amount to pay for 227 notebooks is 51.30 yuan
theorem minimum_amount_for_grade : retail_price 227 = 51.30 := 
sorry

end minimum_amount_for_class1_minimum_amount_for_grade_l662_662330


namespace time_to_pass_tree_is_24_seconds_l662_662634

/-- Define the given values -/
def length_of_train : ℝ := 420 -- meters
def speed_kmph : ℝ := 63 -- km/hr
def conversion_factor : ℝ := 1000 / 3600 -- to convert from km/hr to m/s
def speed_mps : ℝ := speed_kmph * conversion_factor -- convert speed to m/s

/-- Define the time to pass tree based on the length and speed -/
def time_to_pass_tree : ℝ := length_of_train / speed_mps

/-- The theorem stating that the time to pass the tree is 24 seconds -/
theorem time_to_pass_tree_is_24_seconds : time_to_pass_tree = 24 := by
  -- placeholder for the proof
  sorry

end time_to_pass_tree_is_24_seconds_l662_662634


namespace derivative_f_eq_l662_662591

noncomputable def f (x : ℝ) : ℝ := (Real.exp (2 * x)) / x

theorem derivative_f_eq :
  (deriv f) = fun x ↦ ((2 * x - 1) * (Real.exp (2 * x))) / (x ^ 2) := by
  sorry

end derivative_f_eq_l662_662591


namespace cole_average_speed_return_l662_662687

theorem cole_average_speed_return :
  ∀ (speed_to_work time_to_work total_time : ℝ),
    speed_to_work = 60 →
    time_to_work = 72 / 60 →
    total_time = 2 →
    (let distance := speed_to_work * time_to_work in
     let time_return := total_time - time_to_work in
     distance / time_return = 90) :=
by
  intros speed_to_work time_to_work total_time h_speed_to_work h_time_to_work h_total_time
  let distance := speed_to_work * time_to_work
  let time_return := total_time - time_to_work
  have : distance / time_return = 90 := sorry
  exact this

end cole_average_speed_return_l662_662687


namespace billboard_perimeter_l662_662323

theorem billboard_perimeter (A W : ℝ) (hA : A = 117) (hW : W = 9) : 
  ∃ (P : ℝ), P = 44 := 
by
  let L := A / W
  let P := 2 * L + 2 * W
  have hL : L = 13 := by
    calc
      L = A / W : rfl
      _ = 117 / 9 : by rw [hA, hW]
      _ = 13 : by norm_num
  have hP : P = 44 := by
    calc
      P = 2 * L + 2 * W : rfl
      _ = 2 * 13 + 2 * 9 : by rw [hL, hW]
      _ = 44 : by norm_num
  use P
  exact hP

end billboard_perimeter_l662_662323


namespace symmetric_point_coordinates_l662_662480

-- Definition of symmetry in the Cartesian coordinate system
def is_symmetrical_about_origin (A A' : ℝ × ℝ) : Prop :=
  A'.1 = -A.1 ∧ A'.2 = -A.2

-- Given point A and its symmetric property to find point A'
theorem symmetric_point_coordinates (A A' : ℝ × ℝ)
  (hA : A = (1, -2))
  (h_symm : is_symmetrical_about_origin A A') :
  A' = (-1, 2) :=
by
  sorry -- Proof to be filled in (not required as per the instructions)

end symmetric_point_coordinates_l662_662480


namespace exists_infinite_discontinuous_sequence_l662_662321

def is_discontinuous (n : ℕ) : Prop :=
  ∀ (divisors : List ℕ), 
  (divisors = (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0)) →
  (∀ ⦃i : ℕ⦄, i < divisors.length → 
    ∃ j, 1 ≤ j ∧ j < divisors.length ∧ divisors.get j > (List.take j divisors).sum + j)

theorem exists_infinite_discontinuous_sequence :
  ∃ᶠ n in at_top, ∀ k : ℕ, 0 ≤ k → k ≤ 2019 → is_discontinuous (n + k) :=
sorry

end exists_infinite_discontinuous_sequence_l662_662321


namespace height_difference_l662_662099

noncomputable def percentage_increase (b a : ℕ) : ℕ :=
  ((b - a) * 100) / a

theorem height_difference (b : ℕ) (h_pos: 0 < b):
  let a := b - (25 * b) / 100 in
  percentage_increase b a = 33 := 
begin
  sorry
end

end height_difference_l662_662099


namespace max_distance_from_earth_to_sun_l662_662503

-- Assume the semi-major axis 'a' and semi-minor axis 'b' specified in the problem.
def semi_major_axis : ℝ := 1.5 * 10^8
def semi_minor_axis : ℝ := 3 * 10^6

-- Define the theorem stating the maximum distance from the Earth to the Sun.
theorem max_distance_from_earth_to_sun :
  let a := semi_major_axis
  let b := semi_minor_axis
  a + b = 1.53 * 10^8 :=
by
  -- Proof will be completed
  sorry

end max_distance_from_earth_to_sun_l662_662503


namespace rem_a_b_c_div_9_l662_662892

noncomputable def proof_problem : ℕ → ℕ → ℕ → ℕ
| a, b, c :=
let abc_cong_1 := (a * b * c) % 9 = 1 in
let four_c_cong_5 := (4 * c) % 9 = 5 in
let seven_b_cong_4_plus_b := (7 * b) % 9 = (4 + b) % 9 in
if h1 : a ∈ {1,2,3,4,5,6,7,8,9} ∧ b ∈ {1,2,3,4,5,6,7,8,9} ∧ c ∈ {1,2,3,4,5,6,7,8,9} then
  if h2 : abc_cong_1 ∧ four_c_cong_5 ∧ seven_b_cong_4_plus_b then
    (a + b + c) % 9 
  else 
    sorry
else 
  sorry

theorem rem_a_b_c_div_9 (a b c : ℕ) (h1 : a ∈ {1,2,3,4,5,6,7,8,9}) (h2 : b ∈ {1,2,3,4,5,6,7,8,9}) (h3 : c ∈ {1,2,3,4,5,6,7,8,9})
(habc : (a * b * c) % 9 = 1) 
(h4c : (4 * c) % 9 = 5)
(h7b : (7 * b) % 9 = (4 + b) % 9) :
(a + b + c) % 9 = 8 :=
by
  sorry

end rem_a_b_c_div_9_l662_662892


namespace cos_sum_nonneg_one_l662_662524

theorem cos_sum_nonneg_one (x y z : ℝ) (h : x + y + z = 0) : abs (Real.cos x) + abs (Real.cos y) + abs (Real.cos z) ≥ 1 := 
by {
  sorry
}

end cos_sum_nonneg_one_l662_662524


namespace elect_male_voters_voted_for_Sobel_correct_l662_662475

variable (N : ℕ) -- total number of voters
variable (P_S P_M P_{L_F} : ℚ) -- proportions as rational numbers

-- Given conditions
variables (hP_S : P_S = 7/10) -- 70% voters voted for Sobel
variables (hP_M : P_M = 6/10) -- 60% voters are male
variables (hP_{L_F} : P_{L_F} = 35/100) -- 35% female voters voted for Lange

noncomputable def percentage_male_voters_voted_for_Sobel (N : ℕ) 
  (P_S P_M P_{L_F} : ℚ) : ℚ :=
let total_voters := N in
let total_voted_Sobel := P_S * N in
let total_voted_Lange := N - total_voted_Sobel in
let total_male := P_M * N in
let total_female := N - total_male in
let female_voted_Lange := P_{L_F} * total_female in
let male_voted_Lange := total_voted_Lange - female_voted_Lange in
let male_voted_Sobel := total_male - male_voted_Lange in
(male_voted_Sobel / total_male) * 100

theorem elect_male_voters_voted_for_Sobel_correct :
  percentage_male_voters_voted_for_Sobel N P_S P_M P_{L_F} = 73.33 :=
by
  sorry

end elect_male_voters_voted_for_Sobel_correct_l662_662475


namespace selected_representatives_correct_l662_662331

-- Definitions used directly from the conditions
def total_students : ℕ := 247
def num_representatives : ℕ := 4
def start_position : ℕ × ℕ := (4, 9)
def random_numbers_table (row col : ℕ) : ℕ := 
  [[ 3, 47, 43, 73, 86, 36, 96, 47, 36, 61, 46, 98, 63, 71, 62, 33, 26, 16, 80, 45, 60, 11, 14, 10, 95], 
   [97, 74, 24, 67, 62, 42, 81, 14, 57, 20, 42, 53, 32, 37, 32, 27, 07, 36, 07, 51, 24, 51, 79, 89, 73], 
   [16, 76, 62, 27, 66, 56, 50, 26, 71, 07, 32, 90, 79, 78, 53, 13, 55, 38, 58, 59, 88, 97, 54, 14, 10], 
   [12, 56, 85, 99, 26, 96, 96, 68, 27, 31, 05, 03, 72, 93, 15, 57, 12, 10, 14, 21, 88, 26, 49, 81, 76], 
   [55, 59, 56, 35, 64, 38, 54, 82, 46, 22, 31, 62, 43, 09, 90, 06, 18, 44, 32, 53, 23, 83, 01, 30, 30]]

-- Correct Answer based on the problem
def selected_numbers : list ℕ := [50, 121, 14, 218]

-- Lean theorem statement
theorem selected_representatives_correct :
  ∃ selected : list ℕ, selected.length = num_representatives ∧ selected = selected_numbers := by
  sorry

end selected_representatives_correct_l662_662331


namespace triangle_areas_inequality_l662_662889

variable (A B C P A1 B1 C1 : Type) 
variable (S S1 S2 S3 : ℝ)
variable [is_point A]
variable [is_point B]
variable [is_point C]
variable [is_point P]
variable [is_on_line AP]
variable [is_on_line BP]
variable [is_on_line CP]
variable [is_on_line A1]
variable [is_on_line B1]
variable [is_on_line C1]

theorem triangle_areas_inequality 
  (hP_inside_triangle : is_inside_triangle P A B C)
  (h_intersections : 
    intersects (line_through A P) (side BC, A1) ∧ 
    intersects (line_through B P) (side CA, B1) ∧ 
    intersects (line_through C P) (side AB, C1))
  (h_areas_inequality : S1 ≤ S2 ∧ S2 ≤ S3)
  (h_S : S = area A1 B1 C1) :
  sqrt(S1 * S2) ≤ S ∧ S ≤ sqrt(S2 * S3) :=
  sorry

end triangle_areas_inequality_l662_662889


namespace factorize_polynomial_l662_662371

theorem factorize_polynomial (x : ℝ) :
  x^4 + 2 * x^3 - 9 * x^2 - 2 * x + 8 = (x + 4) * (x - 2) * (x + 1) * (x - 1) :=
sorry

end factorize_polynomial_l662_662371


namespace polynomial_evaluation_l662_662654

theorem polynomial_evaluation :
  ∃ p : ℝ → ℝ, (∀ n : ℕ, 1 ≤ n ∧ n ≤ 5 → p n = 1 / n^3) ∧ (p 6 = 0) :=
begin
  sorry
end

end polynomial_evaluation_l662_662654


namespace quadratic_func_inequality_l662_662767

theorem quadratic_func_inequality (c : ℝ) (f : ℝ → ℝ) 
  (h_def : ∀ x, f x = x^2 + 4 * x + c)
  (h_increasing : ∀ x y, x ≤ y → -2 ≤ x → f x ≤ f y) :
  f 1 > f 0 ∧ f 0 > f (-2) :=
by
  sorry

end quadratic_func_inequality_l662_662767


namespace joseph_pairs_adj_l662_662510

def adjacent_pairs (n : ℕ) (p : ℕ → ℕ → Prop) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ a ∈ {1, 2, ..., n} ∧ b ∈ {1, 2, ..., n} ∧ p a b

def diff_by_1_or_2 (a b : ℕ) : Prop :=
  abs (a - b) = 1 ∨ abs (a - b) = 2

theorem joseph_pairs_adj :
  adjacent_pairs 12 (λ a b, diff_by_1_or_2 a b) :=
by
  -- Proof omitted
  sorry

end joseph_pairs_adj_l662_662510


namespace problem_statement_l662_662764

noncomputable def f (x a b : ℝ) : ℝ := (1/3) * x^3 - a * x^2 + (a^2 - 1) * x + b

theorem problem_statement 
  (a b : ℝ)
  (h_extremum : (f 1 a b).derivative' 1 = 0)
  (h_tangent : let t := (λ y, -(1:ℝ) + 3 - y in t (f 1 a b) = 0 ∧ (1 + f 1 a b - 3 = 0))
  :
  (a = 0 ∨ a = 2) ∧
  let f_on_interval := (λ x, f x a b) in
  (f_on_interval (-2) ≤ f_on_interval 4 ∧ f_on_interval (-2) = -4) ∧
  (f_on_interval 2 ≥ f_on_interval 0 ∧ f_on_interval 4 = 8) := 
sorry

end problem_statement_l662_662764


namespace find_area_of_region_l662_662400

def point_area_problem :=
  ∃ (x y : ℝ), 2 * x + y ≤ 2 ∧ x ≥ 0 ∧ (x + sqrt (x^2 + 1)) * (y + sqrt (y^2 + 1)) ≥ 1

theorem find_area_of_region : 
  point_area_problem → 
  (∃ (area : ℝ), area = 2) :=
sorry

end find_area_of_region_l662_662400


namespace complex_number_imaginary_axis_l662_662968

theorem complex_number_imaginary_axis (a : ℝ) : (a^2 - 2 * a = 0) → (a = 0 ∨ a = 2) :=
by
  sorry

end complex_number_imaginary_axis_l662_662968


namespace tangent_segments_l662_662240

theorem tangent_segments (n : ℕ) (h : n = 2017) 
  (h1 : ∀ (i j : ℕ), i ≠ j → ¬ ∃ p : ℝ × ℝ, (p ∈ circles i) ∧ (p ∈ circles j)) 
  (h2 : ∀ (i j k : ℕ), i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ ∃ ℓ : line, (ℓ is tangent to i) ∧ (ℓ is tangent to j) ∧ (ℓ is tangent to k))
  (h3 : ∀ (i j : ℕ) (l : line), i ≠ j → (l is tangent to i) ∧ (l is tangent to j) → ∀ k ≠ i ∧ k ≠ j, (l is not intersecting k)) :
  ∃ m : ℕ, m = 3 * (2017 - 1) := by
  sorry

end tangent_segments_l662_662240


namespace trapezoid_triangle_area_l662_662550

theorem trapezoid_triangle_area (AB CD : ℝ) (area_BEA : ℝ) (h1 : AB = 6) (h2 : CD = 8) 
  (h3 : area_BEA = 60) : 
  ∃ area_BAD : ℝ, area_BAD = 20 :=
by 
  use 20
  sorry

end trapezoid_triangle_area_l662_662550


namespace quadrilateral_parallelogram_l662_662966

def parallelogram (A B C D : Type) [geometry.quadrilateral A B C D] : Prop :=
  geometry.parallel A B C D ∧ geometry.equal_angles C A

theorem quadrilateral_parallelogram (A B C D : Type) [geometry.quadrilateral A B C D]
  (h1 : geometry.parallel A B C D)
  (h2 : geometry.equal_angles C A) :
  parallelogram A B C D :=
sorry

end quadrilateral_parallelogram_l662_662966


namespace math_problem_l662_662097

variable {x a b : ℝ}

theorem math_problem (h1 : x < a) (h2 : a < 0) (h3 : b = -a) : x^2 > b^2 ∧ b^2 > 0 :=
by {
  sorry
}

end math_problem_l662_662097


namespace solve_for_product_l662_662277

theorem solve_for_product (a b c d : ℚ) (h1 : 3 * a + 4 * b + 6 * c + 8 * d = 48)
                          (h2 : 4 * (d + c) = b) 
                          (h3 : 4 * b + 2 * c = a) 
                          (h4 : c - 2 = d) : 
                          a * b * c * d = -1032192 / 1874161 := 
by 
  sorry

end solve_for_product_l662_662277


namespace compute_expression_l662_662688

theorem compute_expression : (3 + 9)^2 + (3^2 + 9^2) = 234 := by
  sorry

end compute_expression_l662_662688


namespace area_ratio_of_triangles_l662_662887

theorem area_ratio_of_triangles (x : ℝ) (h1 : x > 0) : 
  let AB := x in
  let AC := x in
  let BC := 2 * x in

  let BB' := 2 * x in
  let CC' := 2 * x in
  let CD := 2 * x in

  let AB' := AB + BB' in
  let AC' := AC + CC' in
  let BD := BC + CD in

  let area_ABC := (1 / 2) * BC * ((x * real.sqrt 3) / 2) in
  let area_ABD := (1 / 2) * BD * (3 * x * real.sqrt 3 / 2) in

  area_ABD / area_ABC = 6 :=
begin
  sorry
end

end area_ratio_of_triangles_l662_662887


namespace simplify_trig_expression_l662_662948

theorem simplify_trig_expression (α : ℝ) :
  (tan (2 * real.pi + α) / (tan (α + real.pi) - cos (-α) + sin (real.pi / 2 - α))) = 1 :=
by
  -- sorry is used to skip the proof
  sorry

#reduce simplify_trig_expression

end simplify_trig_expression_l662_662948


namespace determine_constants_l662_662432

def f (x : ℝ) : ℝ :=
  if x ≥ -3 ∧ x ≤ 0 then -2 - x
  else if 0 ≤ x ∧ x ≤ 2 then sqrt (4 - (x - 2)^2) - 2
  else if 2 ≤ x ∧ x ≤ 3 then 2 * (x - 2)
  else 0  -- undefined outside these intervals for simplicity

def g (a b c : ℝ) (x : ℝ) : ℝ :=
  a * f (b * x) + c

theorem determine_constants :
  let a := 2
  let b := (1 : ℝ) / 3
  let c := -3
  ∀ x, g a b c x = 2 * f (x / 3) - 3 :=
by
  intros a b c x
  let a := 2
  let b := (1 : ℝ) / 3
  let c := -3
  sorry

end determine_constants_l662_662432


namespace sum_of_numbers_on_circle_l662_662558

theorem sum_of_numbers_on_circle (n : ℕ) : 
  let S_0 := 2 in
  S n = 2 * 3^n :=
by
  sorry

end sum_of_numbers_on_circle_l662_662558


namespace red_more_than_purple_cars_l662_662507

variable (R G : ℕ)

-- Conditions
def condition1 := G = 4 * R
def condition2 := R > 47
def condition3 := G + R + 47 = 312
def condition4 := 47 = 47

theorem red_more_than_purple_cars :
  condition1 → condition2 → condition3 → G - 47 = 6 :=
by
  sorry

end red_more_than_purple_cars_l662_662507


namespace find_y_given_x_eq_0_l662_662195

theorem find_y_given_x_eq_0 (t : ℚ) (x y : ℚ) (h1 : x = 3 - 2 * t) (h2 : y = 3 * t + 6) (h3 : x = 0) : 
  y = 21 / 2 :=
by
  sorry

end find_y_given_x_eq_0_l662_662195


namespace ant_routes_on_circle_l662_662703

def binom : ℕ → ℕ → ℕ
| n 0       := 1
| 0 k       := 0
| (n+1) (k+1) := binom n k + binom n (k+1)

def num_routes (n : ℕ) : ℕ :=
binom 8 n * n * 2^(n-2)

def total_routes : ℕ :=
∑ k in finset.range 9, if k ≥ 2 then num_routes k else 0

theorem ant_routes_on_circle : total_routes = 8744 := 
by {
    sorry
}

end ant_routes_on_circle_l662_662703


namespace solve_problem_l662_662154

def g1 (x : ℝ) : ℝ :=
  (3 / 2) - (5 / (4 * x + 2))

def gn : ℕ → ℝ → ℝ
| 1 := g1
| n := λ x, g1 (gn (n - 1) x)

theorem solve_problem : ∃ x : ℝ, gn 1002 x = x + 2 :=
begin
  use -1/2,
  -- Proof omitted, this is the solution directly from the provided steps
  sorry
end

end solve_problem_l662_662154


namespace smaller_angle_at_3_15_l662_662791

theorem smaller_angle_at_3_15 
  (hours_on_clock : ℕ := 12) 
  (degree_per_hour : ℝ := 360 / hours_on_clock) 
  (minute_hand_position : ℝ := 3) 
  (hour_progress_per_minute : ℝ := 1 / 60 * degree_per_hour) : 
  ∃ angle : ℝ, angle = 7.5 := by
  let hour_hand_position := 3 + (15 * hour_progress_per_minute)
  let angle_diff := abs (minute_hand_position * degree_per_hour - hour_hand_position)
  let smaller_angle := if angle_diff > 180 then 360 - angle_diff else angle_diff
  use smaller_angle
  sorry

end smaller_angle_at_3_15_l662_662791


namespace soda_cost_90_cents_l662_662554

theorem soda_cost_90_cents
  (b s : ℕ)
  (h1 : 3 * b + 2 * s = 360)
  (h2 : 2 * b + 4 * s = 480) :
  s = 90 :=
by
  sorry

end soda_cost_90_cents_l662_662554


namespace magnitude_inverse_sum_eq_l662_662541

noncomputable def complex_magnitude_inverse_sum (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) : ℝ :=
|1/z + 1/w|

theorem magnitude_inverse_sum_eq (z w : ℂ) (hz : |z| = 2) (hw : |w| = 4) (hzw : |z + w| = 3) :
  complex_magnitude_inverse_sum z w hz hw hzw = 3 / 8 :=
by sorry

end magnitude_inverse_sum_eq_l662_662541


namespace gcd_lcm_product_24_54_l662_662380

theorem gcd_lcm_product_24_54 :
  let a := 24 in
  let b := 54 in
  let gcd_ab := Int.gcd a b in
  let lcm_ab := Int.lcm a b in
  gcd_ab * lcm_ab = a * b := by
  let a := 24
  let b := 54
  have gcd_ab : Int.gcd a b = 6 := by
    rw [Int.gcd_eq_right_iff_dvd.mpr (dvd.intro 9 rfl)]
    
  have lcm_ab : Int.lcm a b = 216 := by
    sorry -- We simply add sorry here for the sake of completeness

  show gcd_ab * lcm_ab = a * b
  rw [gcd_ab, lcm_ab]
  norm_num

end gcd_lcm_product_24_54_l662_662380


namespace Cheryl_total_material_used_l662_662597

theorem Cheryl_total_material_used :
  let material1 := (5 : ℚ) / 11
  let material2 := (2 : ℚ) / 3
  let total_purchased := material1 + material2
  let material_left := (25 : ℚ) / 55
  let material_used := total_purchased - material_left
  material_used = 22 / 33 := by
  sorry

end Cheryl_total_material_used_l662_662597


namespace minimum_trucks_needed_l662_662572

theorem minimum_trucks_needed 
  (total_weight : ℕ) (box_weight: ℕ) (truck_capacity: ℕ) (min_trucks: ℕ)
  (h_total_weight : total_weight = 10)
  (h_box_weight_le : ∀ (w : ℕ), w <= box_weight → w <= 1)
  (h_truck_capacity : truck_capacity = 3)
  (h_min_trucks : min_trucks = 5) : 
  min_trucks >= (total_weight / truck_capacity) :=
sorry

end minimum_trucks_needed_l662_662572


namespace total_chairs_calculation_l662_662515

theorem total_chairs_calculation
  (chairs_per_trip : ℕ)
  (trips_per_student : ℕ)
  (total_students : ℕ)
  (h1 : chairs_per_trip = 5)
  (h2 : trips_per_student = 10)
  (h3 : total_students = 5) :
  total_students * (chairs_per_trip * trips_per_student) = 250 :=
by
  sorry

end total_chairs_calculation_l662_662515


namespace value_of_k_l662_662095

variable (m n k b : ℝ)

theorem value_of_k (h₁ : log 10 m = 2 * b - log 10 n) (h₂ : m = n ^ 2 * k) : 
    k = (10 ^ b) ^ 2 / n ^ 3 :=
by
  sorry

end value_of_k_l662_662095


namespace exists_prime_among_15_numbers_l662_662642

theorem exists_prime_among_15_numbers 
    (integers : Fin 15 → ℕ)
    (h1 : ∀ i, 1 < integers i)
    (h2 : ∀ i, integers i < 1998)
    (h3 : ∀ i j, i ≠ j → Nat.gcd (integers i) (integers j) = 1) :
    ∃ i, Nat.Prime (integers i) :=
by
  sorry

end exists_prime_among_15_numbers_l662_662642


namespace part1_part2_part3_l662_662252

def digits := {0, 1, 2, 3, 4, 5}

def is_odd (n : ℕ) : Prop := n % 2 = 1
def no_repeated_digits (n : ℕ) : Prop :=
  let digits := List.ofNatDigits 10 n in
  digits.length = digits.erase_dup.length

def six_digit_number (n : ℕ) : Prop :=
  100000 ≤ n ∧ n < 1000000

def valid_six_digit_number (n : ℕ) : Prop :=
  digits.card = 6 ∧ no_repeated_digits n

def ends_with_odd_digit (n : ℕ) : Prop :=
  List.getLast (List.ofNatDigits 10 n) ∈ {1, 3, 5}

def ends_with_5 (n : ℕ) : Prop :=
  List.getLast (List.ofNatDigits 10 n) = 5

def digits_not_adjacent (d1 d2 : ℕ) (n : ℕ) : Prop :=
  let ds := List.ofNatDigits 10 n in
  ¬(∃ i, ds.lookup idx i = d1 ∧ ds.get? (i+1) = some d2)

theorem part1 :
  ∃ (N : ℕ), N = 288 ∧
  ∀ n : ℕ, six_digit_number n → valid_six_digit_number n → ends_with_odd_digit n ↔ n = N :=
by sorry

theorem part2 :
  ∃ (N : ℕ), N = 504 ∧
  ∀ n : ℕ, six_digit_number n → valid_six_digit_number n → ¬ends_with_5 n ↔ n = N :=
by sorry

theorem part3 :
  ∃ (N : ℕ), N = 408 ∧
  ∀ n : ℕ, six_digit_number n → valid_six_digit_number n → digits_not_adjacent 1 2 n ↔ n = N :=
by sorry

end part1_part2_part3_l662_662252


namespace midpoint_y_coordinate_l662_662080

theorem midpoint_y_coordinate (a : ℝ) 
  (h₀ : 0 < a) 
  (h₁ : a < π / 2) 
  (h₂ : abs (sin a - cos a) = 1 / 5) : 
  (sin a + cos a) / 2 = 7 / 10 := 
sorry

end midpoint_y_coordinate_l662_662080


namespace remainder_when_7n_div_by_3_l662_662096

theorem remainder_when_7n_div_by_3 (n : ℤ) (h : n % 3 = 2) : (7 * n) % 3 = 2 := 
sorry

end remainder_when_7n_div_by_3_l662_662096


namespace find_enclosed_area_l662_662141

noncomputable def sqrt_2 := Real.sqrt 2
noncomputable def pi_val := Real.pi

def B : ℝ × ℝ := (20, 14)
def C : ℝ × ℝ := (18, 0)

def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

def radius (d : ℝ) : ℝ := d / 2

def area (r : ℝ) : ℝ := pi_val * r * r

def enclosed_area : ℝ :=
  area (radius (distance B C))

def floor_val (x : ℝ) : ℝ := Real.floor x

theorem find_enclosed_area :
  floor_val enclosed_area = 157 :=
by
  unfold enclosed_area
  unfold area
  unfold radius
  unfold distance
  unfold B
  unfold C
  unfold sqrt_2
  unfold pi_val
  unfold floor_val
  sorry

end find_enclosed_area_l662_662141


namespace midpoint_uniqueness_l662_662190

-- Define a finite set of points in the plane
axiom S : Finset (ℝ × ℝ)

-- Define what it means for P to be the midpoint of a segment
def is_midpoint (P A A' : ℝ × ℝ) : Prop :=
  P.1 = (A.1 + A'.1) / 2 ∧ P.2 = (A.2 + A'.2) / 2

-- Statement of the problem
theorem midpoint_uniqueness (P Q : ℝ × ℝ) :
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint P A A') →
  (∀ A ∈ S, ∃ A' ∈ S, is_midpoint Q A A') →
  P = Q :=
sorry

end midpoint_uniqueness_l662_662190


namespace power_factor_200_l662_662884

theorem power_factor_200 :
  (let a := 3 in let b := 2 in (1 / 3)^(b - a) = 3) :=
by
  -- assume a and b definitions
  let a := 3
  let b := 2
  -- main statement
  show (1 / 3) ^ (b - a) = 3
  -- we skip the proof
  sorry

end power_factor_200_l662_662884


namespace sequence_limit_infinity_l662_662522

/-- τ(n) denotes the number of positive integer divisors of a positive integer n. -/
def tau (n : ℕ) : ℕ := if n = 0 then 0 else (List.range (n+1)).count (λ d => n % d = 0)

/-- Define the sequence a_n based on P(X) and tau(n). -/
def seq (P : ℕ → ℤ) (n : ℕ) : ℕ :=
  if P n > 0 then Int.gcd (P n).natAbs (tau (P n).natAbs) else 0

/-- Prove that the sequence a₁, a₂, ... has limit infinity for any polynomial P(X) with integer coefficients. -/
theorem sequence_limit_infinity (P : ℕ → ℤ) :
  ∀ k : ℕ, ∃ N : ℕ, ∀ n ≥ N, seq P n ≠ k := 
sorry

end sequence_limit_infinity_l662_662522


namespace z2_condition_l662_662054

def z1 : ℂ := 2 - complex.i
def z2 (a : ℝ) : ℂ := a + 2 * complex.i

theorem z2_condition (a : ℝ) : ∃ a, (z2 a) := ∃ a, (2 * z2 a).im = 0 := sorry

end z2_condition_l662_662054


namespace students_not_making_cut_l662_662616

theorem students_not_making_cut (girls boys called_back total students_not_making_cut : ℕ)
  (h1 : girls = 39)
  (h2 : boys = 4)
  (h3 : called_back = 26)
  (h4 : total = girls + boys)
  (h5 : students_not_making_cut = total - called_back) : students_not_making_cut = 17 :=
by {
  rw [h1, h2] at h4,
  rw [h3] at h5,
  rw h4 at h5,
  exact h5,
}

end students_not_making_cut_l662_662616


namespace max_value_of_z_l662_662735

open Complex Real

theorem max_value_of_z (z : ℂ) (h : |z - (2 + 2*I)| = 1) : |z + 2 - I| ≤ sqrt 17 + 1 :=
sorry

end max_value_of_z_l662_662735


namespace remainder_of_sum_of_n_l662_662526

def isPerfectSquare (x : ℤ) : Prop := ∃ m : ℤ, m * m = x

theorem remainder_of_sum_of_n 
  (S : ℤ) 
  (hSum : S = ∑ n in Finset.filter (λ n, isPerfectSquare (n^2 + 14 * n - 1595)) (Finset.range 1000), n) :
  S % 1000 = 405 :=
by
  sorry

end remainder_of_sum_of_n_l662_662526


namespace count_numbers_leaving_remainder_one_l662_662796

theorem count_numbers_leaving_remainder_one :
  (finset.filter (λ n : ℕ, n % 3 = 1) (finset.range 51)).card = 17 :=
sorry

end count_numbers_leaving_remainder_one_l662_662796


namespace intersection_AB_CD_l662_662849

-- Points in coordinate space
def A : ℝ × ℝ × ℝ := (3, -2, 5)
def B : ℝ × ℝ × ℝ := (13, -12, 10)
def C : ℝ × ℝ × ℝ := (-2, 5, -8)
def D : ℝ × ℝ × ℝ := (3, -1, 12)

-- Required intersection point
def intersection_point : ℝ × ℝ × ℝ := (-1/11, 1/11, 15/11)

-- Theorem stating the intersection point of lines AB and CD
theorem intersection_AB_CD : 
  ∃ t s : ℝ, 
    (A.1 + t * (B.1 - A.1) = intersection_point.1) ∧ 
    (A.2 + t * (B.2 - A.2) = intersection_point.2) ∧ 
    (A.3 + t * (B.3 - A.3) = intersection_point.3) ∧ 
    (C.1 + s * (D.1 - C.1) = intersection_point.1) ∧ 
    (C.2 + s * (D.2 - C.2) = intersection_point.2) ∧ 
    (C.3 + s * (D.3 - C.3) = intersection_point.3) := 
by
  sorry

end intersection_AB_CD_l662_662849


namespace functions_with_inverses_l662_662692

-- Definitions of the functions
def is_linear (f : ℝ → ℝ) : Prop :=
∀ x y : ℝ, f (x + y) = f x + f y ∧ ∀ c : ℝ, f (c * x) = c * f x

def is_circle (f : ℝ → ℝ) : Prop :=
∃ a b r : ℝ, r > 0 ∧ f = λ x, sqrt (r^2 - (x - a)^2) + b

def is_upsidedown_parabola (f : ℝ → ℝ) : Prop :=
∃ a b : ℝ, f = λ x, -a * x^2 + b

def is_horizontal_line (f : ℝ → ℝ) : Prop :=
∃ c : ℝ, ∀ x : ℝ, f x = c

-- Statement to prove: Determine which functions have inverses
theorem functions_with_inverses (F G H I : ℝ → ℝ) :
is_linear F → is_circle G → is_upsidedown_parabola H → is_horizontal_line I →
(has_inverse F) ∧ (¬ (has_inverse G)) ∧ (has_inverse H) ∧ (¬ (has_inverse I)) := 
by
  sorry

end functions_with_inverses_l662_662692


namespace john_walking_speed_l662_662137

theorem john_walking_speed (x : ℝ) (hx : x > 0) (h1 : ∀ d: ℝ, d = 3) 
  (h2 : ∀ t₁ t₂: ℝ, t₁ = 3/x ∧ t₂ = 3/(2*x) + 0.25) :
  x = 6 :=
by
  have t₁ := 3/x
  have t₂ := 3/(2*x) + 0.25
  rw [t₁, t₂]
  sorry

end john_walking_speed_l662_662137


namespace geese_among_non_swans_is_40_l662_662139

def percentGeese : ℝ := 30
def percentSwans : ℝ := 25
def percentHerons : ℝ := 10
def percentDucks : ℝ := 35

def nonSwans : ℝ := 100 - percentSwans
def geeseAmongNonSwans : ℝ := (percentGeese / nonSwans) * 100

theorem geese_among_non_swans_is_40 :
  geeseAmongNonSwans = 40 :=
by sorry

end geese_among_non_swans_is_40_l662_662139
