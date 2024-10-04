import Mathlib

namespace p_twice_q_in_future_years_l587_587102

-- We define the ages of p and q
def p_current_age : ℕ := 33
def q_current_age : ℕ := 11

-- Third condition that is redundant given the values we already defined
def age_relation : Prop := (p_current_age = 3 * q_current_age)

-- Number of years in the future when p will be twice as old as q
def future_years_when_twice : ℕ := 11

-- Prove that in future_years_when_twice years, p will be twice as old as q
theorem p_twice_q_in_future_years :
  ∀ t : ℕ, t = future_years_when_twice → (p_current_age + t = 2 * (q_current_age + t)) := by
  sorry

end p_twice_q_in_future_years_l587_587102


namespace susan_initial_amount_l587_587405

theorem susan_initial_amount :
  ∃ S: ℝ, (S - (1/5 * S + 1/4 * S + 120) = 1200) → S = 2400 :=
by
  sorry

end susan_initial_amount_l587_587405


namespace remainder_of_polynomial_division_l587_587575

theorem remainder_of_polynomial_division :
  ∀ (x : ℝ), 
    let p := x^75,
        q := (x + 1)^4 in
    ∃ r : polynomial ℝ,
      degree r < degree q ∧ p = q * (p / q) + r :=
    ∃ r : polynomial ℝ,
      r = -963175 * (X + 1)^3 + 2775 * (X + 1)^2 + 75 * (X + 1) + 1 :=
by sorry

end remainder_of_polynomial_division_l587_587575


namespace matrices_inverses_l587_587549

def matrixA (a b c d e : ℝ) := 
  ![
    [a, 1, 0, b],
    [0, 3, 2, 0],
    [c, 4, d, 5],
    [6, 0, 7, e]
  ]

def matrixB (f g h : ℝ) := 
  ![
    [-7, f, 0, -15],
    [g, -20, h, 0],
    [0, 2, 5, 0],
    [3, 0, 8, 6]
  ]

theorem matrices_inverses (a b c d e f g h : ℝ) 
  (ha : matrixA a b c d e ⬝ matrixB f g h = (1 : Matrix (Fin 4) (Fin 4) ℝ)) :
  a + b + c + d + e + f + g + h = 27 := 
by 
  -- This is where the actual proof would be written.
  sorry

end matrices_inverses_l587_587549


namespace intersection_is_correct_l587_587611

open Set

def A : Set ℝ := { x | x^2 ≤ 1 }

def B : Set ℝ := { x | 2 / x ≥ 1 }

theorem intersection_is_correct : (A ∩ B) = { x | 0 < x ∧ x ≤ 1 } := 
by sorry

end intersection_is_correct_l587_587611


namespace bisect_PQ_by_OM_l587_587274

def ellipse (x y a b : ℝ) : Prop := ((x^2) / (a^2) + (y^2) / (b^2) = 1)
def is_eccentricity (a c : ℝ) : Prop := (c / a = 1 / 2)
def point_on_ellipse (x y a b : ℝ) : Prop := ellipse x y a b

noncomputable def midpoint (x1 y1 x2 y2 : ℝ) : ℝ × ℝ := ((x1 + x2) / 2, (y1 + y2) / 2)
def line_eq (k x1 y1 x : ℝ) : ℝ := k * (x - x1) + y1
def vertical_line (x : ℝ) := (x = 4)
def perpendicular_lines (l1 l2 : ℝ → ℝ) := ∀ x, l2 x = (-1) / (l1 x)
def bisects (P Q M O : ℝ × ℝ) : Prop := O = midpoint P.1 P.2 Q.1 Q.2

theorem bisect_PQ_by_OM : 
  ∀ (a b c : ℝ) (O P Q M : ℝ × ℝ),
    a > b → b > 0 →
    is_eccentricity a c →
    point_on_ellipse 1 (3/2) a b →
    ∃ F : ℝ × ℝ, (perpendicular_lines (λ x, P) (λ x, M)) →
    vertical_line M.1 →
    bisects P Q M O :=
sorry

end bisect_PQ_by_OM_l587_587274


namespace tanQ_tanR_l587_587342

theorem tanQ_tanR (P Q R : Point) (H S : Point) (PS QR : Line) :
  is_triangle P Q R →
  is_altitude PS P QR →
  is_orthocenter H P QR →
  divides H PS into_sections (5, 20) →
  tan_angle (angle P Q R) * tan_angle (angle P R Q) = 25 / 16 :=
begin
  sorry
end

end tanQ_tanR_l587_587342


namespace obtuse_triangle_sum_range_l587_587667

variable (a b c : ℝ)

theorem obtuse_triangle_sum_range (h1 : b^2 + c^2 - a^2 = b * c)
                                   (h2 : a = (Real.sqrt 3) / 2)
                                   (h3 : (b * c) * (Real.cos (Real.pi - Real.arccos ((b^2 + c^2 - a^2) / (2 * b * c)))) < 0) :
    (b + c) ∈ Set.Ioo ((Real.sqrt 3) / 2) (3 / 2) :=
sorry

end obtuse_triangle_sum_range_l587_587667


namespace BC_together_finish_in_2_hours_l587_587855

def time_to_finish_by (work_rate: ℝ) (total_work: ℝ) : ℝ :=
  total_work / work_rate

noncomputable def individual_work (hours: ℝ) : ℝ :=
  1 / hours

noncomputable def combined_work (hours: ℝ) : ℝ :=
  1 / hours

theorem BC_together_finish_in_2_hours :
  let A_work_rate : ℝ := individual_work 4,
      B_work_rate : ℝ := individual_work 4,
      AC_combined_work_rate : ℝ := combined_work 2 in
  time_to_finish_by (B_work_rate + (AC_combined_work_rate - A_work_rate)) 1 = 2 :=
by
  let A_work_rate := individual_work 4
  let B_work_rate := individual_work 4
  let AC_combined_work_rate := combined_work 2
  let C_work_rate := AC_combined_work_rate - A_work_rate
  have BC_combined_work_rate : ℝ := B_work_rate + C_work_rate
  show time_to_finish_by BC_combined_work_rate 1 = 2 from sorry

end BC_together_finish_in_2_hours_l587_587855


namespace _l587_587286

noncomputable def hyperbola_angle_theorem 
  (a b c : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0)
  (h_hyperbola : ∀ x y, (x * x) / (a * a) - (y * y) / (b * b) = 1)
  (h_eccentricity : c = 2 * a) 
  (h_relation : c * c = a * a + b * b) 
  (M F A B : ℝ × ℝ) 
  (hM : M = (-a, 0)) 
  (hF : F = (2 * a, 0))
  (hA : A = (2 * a, b))
  (hB : B = (2 * a, -b)) :
  angle A M B = π / 2 :=
by
  sorry

end _l587_587286


namespace arithmetic_sequence_b_sequence_sum_l587_587377

variable (a_n : ℕ → ℕ) (b_n : ℕ → ℝ) (S_n : ℕ → ℝ) (d : ℕ)

def common_difference := ∀ n : ℕ, a_n (n + 1) - a_n n = d
def first_term_condition := 2 * a_n 1 = d
def sequence_condition := ∀ n : ℕ, 2 * a_n n = a_n (2 * n) - 1

def sequence_formula := ∀ n : ℕ, a_n n = 2 * n - 1
def b_sequence := ∀ n : ℕ, b_n n = (a_n n + 1) / 2^(n + 1)
def b_sum := ∀ n : ℕ, S_n n = ∑ i in range (n + 1), b_n i

theorem arithmetic_sequence (h1 : common_difference a_n d)
                            (h2 : first_term_condition a_n d)
                            (h3 : sequence_condition a_n) :
                            sequence_formula a_n :=
by sorry

theorem b_sequence_sum (h1 : sequence_formula a_n)
                       (h2 : b_sequence a_n b_n) :
                       ∀ n : ℕ, S_n n = 2 - (2 + n) / 2^n :=
by sorry

end arithmetic_sequence_b_sequence_sum_l587_587377


namespace seating_arrangements_l587_587768

open BigOperators

noncomputable def combinations (n k : ℕ) : ℕ :=
  Nat.choose n k

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

theorem seating_arrangements : 
  let chosenPeople := 10.choose 8,
      units := factorial 6,
      AliceBobWays := 2 
  in chosenPeople * units * AliceBobWays = 64800 := 
by
  sorry

end seating_arrangements_l587_587768


namespace binomial_expansion_integer_exponents_l587_587563

theorem binomial_expansion_integer_exponents {n : ℕ} 
  (h_arith_seq: 2 * (1/2: ℚ) * n = 1 + (1/8: ℚ) * n * (n - 1)) :
  n = 8 → (∃ r, r ∈ {0, 4, 8} ∧ (2 * n - 3 * r) % 3 = 0) :=
by
  sorry

end binomial_expansion_integer_exponents_l587_587563


namespace total_matches_won_l587_587326

-- Define the conditions
def matches_in_first_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat
def matches_in_second_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat

-- The main proof statement that we need to prove
theorem total_matches_won (total1 total2 : ℕ) (win_rate1 win_rate2 : ℚ) :
  matches_in_first_period total1 win_rate1 + matches_in_second_period total2 win_rate2 = 110 :=
by
  sorry

end total_matches_won_l587_587326


namespace square_of_area_of_X1X2X3X4_l587_587696

-- Definitions and conditions
def rectangle (A B C D : ℝ × ℝ) :=
  A = (0, 0) ∧
  B = (0, 6) ∧
  C = (6*sqrt 3, 6) ∧
  D = (6*sqrt 3, 0) ∧
  dist A B = 6 ∧
  dist B C = 6 * sqrt 3 ∧
  dist C D = 6 ∧
  dist D A = 6 * sqrt 3

def semicircle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧
  p.2 ≥ center.2

def intersect_at (p : ℝ × ℝ) (ω1 ω2 : ℝ × ℝ → Prop) : Prop :=
  ω1 p ∧ ω2 p

-- Problem statement
theorem square_of_area_of_X1X2X3X4
  (A B C D : ℝ × ℝ)
  (h_rect : rectangle A B C D)
  (ω1 ω2 ω3 ω4 : (ℝ × ℝ) → Prop)
  (hω1 : ∀ p, ω1 p ↔ semicircle (0, 3) 3 p)
  (hω2 : ∀ p, ω2 p ↔ semicircle (3*sqrt 3, 6) (3*sqrt 3) p)
  (hω3 : ∀ p, ω3 p ↔ semicircle (3*sqrt 3, 0) (3*sqrt 3) p)
  (hω4 : ∀ p, ω4 p ↔ semicircle (6*sqrt 3, 3) 3 p)
  (X1 X2 X3 X4 : ℝ × ℝ)
  (hX1 : intersect_at X1 ω1 ω2)
  (hX2 : intersect_at X2 ω2 ω3)
  (hX3 : intersect_at X3 ω3 ω4)
  (hX4 : intersect_at X4 ω4 ω1) :
  (9 * sqrt 3)^2 = 243 := by
  sorry

end square_of_area_of_X1X2X3X4_l587_587696


namespace non_negative_integer_solutions_of_inequality_system_l587_587032

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end non_negative_integer_solutions_of_inequality_system_l587_587032


namespace hours_not_raining_l587_587587

theorem hours_not_raining 
  (start_time end_time rain_hours total_hours hours_not_raining : ℕ)
  (h1 : start_time = 9)
  (h2 : end_time = 17) -- 5 pm is 17:00 in 24-hour format
  (h3 : rain_hours = 2)
  (h4 : total_hours = end_time - start_time)
  (h5 : total_hours = 8)
  (h6 : hours_not_raining = total_hours - rain_hours) :
  hours_not_raining = 6 :=
by
  rw [h5, h3] at h6
  exact h6.symm

end hours_not_raining_l587_587587


namespace product_identity_l587_587475

theorem product_identity :
  (1 + 1 / Nat.factorial 1) * (1 + 1 / Nat.factorial 2) * (1 + 1 / Nat.factorial 3) *
  (1 + 1 / Nat.factorial 4) * (1 + 1 / Nat.factorial 5) * (1 + 1 / Nat.factorial 6) *
  (1 + 1 / Nat.factorial 7) = 5041 / 5040 := sorry

end product_identity_l587_587475


namespace all_numbers_equal_l587_587359

theorem all_numbers_equal (n : ℕ) (h_pos_n : 0 < n) (x : Fin (1369 ^ n) → ℚ)
    (h_property : ∀ i : Fin (1369 ^ n), 
      ∃ s : Finset (Fin (1369 ^ n)), s.card = 1368 ∧ i ∉ s ∧ 
        (∃ f : Fin (1368).succ → Finset ℚ, 
          (∀ j, j < Fin (1368).succ → (s.image (λ k, if k < i then x k else x (k + 1)).f j)).card = 1368 ∧ 
          ∀ j₁ j₂, j₁ < Fin (1368).succ → j₂ < Fin (1368).succ → 
            ∏ j in f j₁, j = ∏ j in f j₂, j)) :
  ∀ i j, x i = x j := sorry

end all_numbers_equal_l587_587359


namespace cube_volume_space_diagonal_l587_587781

-- Definitions of the given condition
def is_space_diagonal (s d : ℝ) : Prop := d = s * real.sqrt 3

-- Main goal: proving the volume of the cube given the condition
theorem cube_volume_space_diagonal 
  (s d : ℝ) (h : is_space_diagonal s d) (h_d : d = 5 * real.sqrt 3) :
  s ^ 3 = 125 :=
by 
  have sqrt_eq : s = 5 := by sorry
  exact sorry

end cube_volume_space_diagonal_l587_587781


namespace only_odd_digit_squared_n_l587_587568

def is_odd_digit (d : ℕ) : Prop :=
  d = 1 ∨ d = 3 ∨ d = 5 ∨ d = 7 ∨ d = 9

def has_only_odd_digits (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ n.digits 10 → is_odd_digit d

theorem only_odd_digit_squared_n (n : ℕ) :
  0 < n ∧ has_only_odd_digits (n * n) ↔ n = 1 ∨ n = 3 :=
sorry

end only_odd_digit_squared_n_l587_587568


namespace R_and_D_per_increase_l587_587538

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end R_and_D_per_increase_l587_587538


namespace quadrilateral_angle_sum_l587_587847

theorem quadrilateral_angle_sum (A B C D M : Point) [ConvexQuadrilateral A B C D] 
(midpoint_M : Midpoint M A C) 
(angle_cond : ∠ M C B = ∠ C M D ∧ ∠ M B A = ∠ M B C - ∠ M D C) :
  distance A D = distance D C + distance A B :=
sorry

end quadrilateral_angle_sum_l587_587847


namespace slope_angle_45_degrees_l587_587793

-- Definition of the given line equation
def line_eqn (x y : ℝ) : Prop := (x / 2019) - (y / 2019) = 1

-- Goal: Prove the slope angle is 45°
theorem slope_angle_45_degrees (x y : ℝ) (h : line_eqn x y) : 
  ∃ θ : ℝ, θ = 45 ∧ slope_angle x y θ := sorry

end slope_angle_45_degrees_l587_587793


namespace domain_of_function_l587_587571

def domain_sqrt_log : Set ℝ :=
  {x | (2 - x ≥ 0) ∧ ((2 * x - 1) / (3 - x) > 0)}

theorem domain_of_function :
  domain_sqrt_log = {x | (1/2 < x) ∧ (x ≤ 2)} :=
by
  sorry

end domain_of_function_l587_587571


namespace sequence_product_equality_l587_587190

theorem sequence_product_equality :
  (∀ n, a n = 2 ^ (9 - n)) →
  (∀ n, T n = (∏ i in finset.range n, a (i + 1))) →
  T 5 = T 12 :=
by
  intros h₁ h₂
  sorry

end sequence_product_equality_l587_587190


namespace number_of_three_digit_whole_numbers_with_digit_sum_24_l587_587234

theorem number_of_three_digit_whole_numbers_with_digit_sum_24 : 
  (finset.filter (λ n, (let a := n / 100 in 
                        let b := (n % 100) / 10 in 
                        let c := n % 10 in 
                        a + b + c = 24) 
                  (finset.Icc 100 999)).card = 4 := 
begin
  sorry
end

end number_of_three_digit_whole_numbers_with_digit_sum_24_l587_587234


namespace students_without_an_A_l587_587664

theorem students_without_an_A :
  ∀ (total_students : ℕ) (history_A : ℕ) (math_A : ℕ) (computing_A : ℕ)
    (math_and_history_A : ℕ) (history_and_computing_A : ℕ)
    (math_and_computing_A : ℕ) (all_three_A : ℕ),
  total_students = 40 →
  history_A = 10 →
  math_A = 18 →
  computing_A = 9 →
  math_and_history_A = 5 →
  history_and_computing_A = 3 →
  math_and_computing_A = 4 →
  all_three_A = 2 →
  total_students - (history_A + math_A + computing_A - math_and_history_A - history_and_computing_A - math_and_computing_A + all_three_A) = 13 :=
by
  intros total_students history_A math_A computing_A math_and_history_A history_and_computing_A math_and_computing_A all_three_A 
         ht_total_students ht_history_A ht_math_A ht_computing_A ht_math_and_history_A ht_history_and_computing_A ht_math_and_computing_A ht_all_three_A
  sorry

end students_without_an_A_l587_587664


namespace sum_b_l587_587602

-- Given a sequence {a_n} with the sum of the first n terms S_n = (3/2) * n^2 + (5/2) * n
def S (n : ℕ) : ℚ := 3 / 2 * n^2 + 5 / 2 * n

-- Defining a_n
def a (n : ℕ) : ℚ := 3 * n + 1

-- Defining b_n in terms of a_n and a_{n+1}
def b (n : ℕ) : ℚ := 1 / (a n * a (n + 1))

-- The sum of the first n terms of {b_n}, denoted as T_n
def T (n : ℕ) : ℚ := (∑ i in Finset.range n, b i)

-- Statement to prove: T_n = n / (12n + 16)
theorem sum_b {n : ℕ} : T n = n / (12 * n + 16) :=
sorry

end sum_b_l587_587602


namespace triangle_ac_bd_squared_l587_587905

noncomputable theory

open Real

-- Define the conditions
variables {O A B C D E : Point}
variables (circle_center_O : is_center O circle)
variables (chord_AB : is_chord A B circle)
variables (chord_CD : is_chord C D circle)
variables (perpendicular_AB_CD : is_perpendicular (line_through A B) (line_through C D))
variables (diameter_AE : is_diameter A E circle)
variables (AE_length : distance A E = d)

-- The main statement to prove
theorem triangle_ac_bd_squared (AC BD d : ℝ) (AC_squared : distance A C ^ 2)
  (BD_squared : distance B D ^ 2) :
  AC ^ 2 + BD ^ 2 = d ^ 2 :=
by
  sorry

end triangle_ac_bd_squared_l587_587905


namespace average_speed_is_40_l587_587743

-- Define the total distance
def total_distance : ℝ := 640

-- Define the distance for the first half
def first_half_distance : ℝ := total_distance / 2

-- Define the average speed for the first half
def first_half_speed : ℝ := 80

-- Define the time taken for the first half
def first_half_time : ℝ := first_half_distance / first_half_speed

-- Define the multiplicative factor for time increase in the second half
def time_increase_factor : ℝ := 3

-- Define the time taken for the second half
def second_half_time : ℝ := first_half_time * time_increase_factor

-- Define the total time for the trip
def total_time : ℝ := first_half_time + second_half_time

-- Define the calculated average speed for the entire trip
def calculated_average_speed : ℝ := total_distance / total_time

-- State the theorem that the average speed for the entire trip is 40 miles per hour
theorem average_speed_is_40 : calculated_average_speed = 40 :=
by
  sorry

end average_speed_is_40_l587_587743


namespace angle_A_is_120_degrees_l587_587317

theorem angle_A_is_120_degrees (a b c : ℝ) (h : a^2 = b^2 + c^2 + b * c) : 
  ∠ (A : ℝ) = 120 :=
sorry

end angle_A_is_120_degrees_l587_587317


namespace sin_α_minus_pi_over_12_l587_587720

variable (α : ℝ)

-- We are given that α is an acute angle
axiom α_is_acute : 0 < α ∧ α < π / 2

-- We are given that cos(α + π/6) = 3/5
axiom cos_α_plus_pi_over_6 : Real.cos (α + π / 6) = 3 / 5

-- We need to show that sin(α - π/12) = √2 / 10
theorem sin_α_minus_pi_over_12 :
  Real.sin (α - π / 12) = √2 / 10 := by
  -- Place proof here
  sorry

end sin_α_minus_pi_over_12_l587_587720


namespace count_pos_int_multiple_6_lcm_gcd_l587_587297

open Nat

theorem count_pos_int_multiple_6_lcm_gcd : 
  let six_fact := fact 6
  let twelve_fact := fact 12
  ∃ (n : ℕ), n > 0 ∧ (n % 6 = 0) ∧ (Nat.lcm six_fact n = 6 * Nat.gcd twelve_fact n) ∧ (Finset.filter (λ n : ℕ, n > 0 ∧ (n % 6 = 0) ∧ (Nat.lcm six_fact n = 6 * Nat.gcd twelve_fact n)) (Finset.range 479001601)).card = 180 := 
  by sorry

end count_pos_int_multiple_6_lcm_gcd_l587_587297


namespace wang_trip_duration_xiao_travel_times_l587_587751

variables (start_fee : ℝ) (time_fee_per_min : ℝ) (mileage_fee_per_km : ℝ) (long_distance_fee_per_km : ℝ)

-- Conditions
def billing_rules := 
  start_fee = 12 ∧ 
  time_fee_per_min = 0.5 ∧ 
  mileage_fee_per_km = 2.0 ∧ 
  long_distance_fee_per_km = 1.0

-- Proof for Mr. Wang's trip duration
theorem wang_trip_duration
  (x : ℝ) 
  (total_fare : ℝ)
  (distance : ℝ) 
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km) : 
  total_fare = 69.5 ∧ distance = 20 → 0.5 * x = 12.5 :=
by 
  sorry

-- Proof for Xiao Hong's and Xiao Lan's travel times
theorem xiao_travel_times 
  (x : ℝ) 
  (travel_time_multiplier : ℝ)
  (distance_hong : ℝ)
  (distance_lan : ℝ)
  (equal_fares : Prop)
  (h : billing_rules start_fee time_fee_per_min mileage_fee_per_km long_distance_fee_per_km)
  (p1 : distance_hong = 14 ∧ distance_lan = 16 ∧ travel_time_multiplier = 1.5) :
  equal_fares → 0.25 * x = 5 :=
by 
  sorry

end wang_trip_duration_xiao_travel_times_l587_587751


namespace series_convergence_l587_587238

theorem series_convergence (c : ℕ → ℝ) (z : ℂ) (h_pos : ∀ k, c k > 0)
  (h_sum : summable (λ k, c k / k)) : summable (λ n, (c 1) * (c 2) * ... * (c n) * z^n) :=
sorry

end series_convergence_l587_587238


namespace percentage_of_skirts_in_hamper_l587_587477

theorem percentage_of_skirts_in_hamper (blouses skirts slacks : ℕ) (pct_blouses_pct_slacks_wash_needed : ℚ) :
  blouses = 12 → skirts = 6 → slacks = 8 → 
  0.75 * blouses = 9 → 0.25 * slacks = 2 → 
  let clothes_in_washer := 14 in 
  (clothes_in_washer - (0.75 * blouses + 0.25 * slacks)) = 3 → 
  (3 / skirts) * 100 = 50 := 
by {
  intro h_blouses h_skirts h_slacks h_pct_blouses h_pct_slacks h_wash_needed,
  simp only [h_blouses, h_skirts, h_slacks, h_pct_blouses, h_pct_slacks] at *,
  have h_total := h_wash_needed,
  simp only [h_pct_blouses, h_pct_slacks, h_wash_needed] at h_total,
  have h_perc_skirts := show  (3 / 6) * 100 = 50, by norm_num,
  exact h_perc_skirts,
}

end percentage_of_skirts_in_hamper_l587_587477


namespace min_questionnaires_to_mail_l587_587311

theorem min_questionnaires_to_mail (response_rate : ℝ) (required_responses : ℕ) (Q : ℕ) : 
  response_rate = 0.70 →
  required_responses = 300 →
  Q ≥ (required_responses / response_rate).ceil →
  Q = 429 :=
by
  intros h1 h2 h3
  -- Proof goes here
  sorry

end min_questionnaires_to_mail_l587_587311


namespace general_term_sequence_l587_587654

theorem general_term_sequence (n : ℕ) (h₁ : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :
  (if n = 1 then 1/2
   else if n = 2 then -1/3
   else if n = 3 then 1/4
   else -1/5) = ( -1: ℤ)^(n + 1) / (n + 1 : ℤ) := 
by {
  cases h₁; 
  simp [h₁]; 
  norm_num,
}

end general_term_sequence_l587_587654


namespace simplify_ratio_l587_587223

noncomputable def c_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), k^2 / (Nat.choose n k : ℚ)
noncomputable def d_n (n : ℕ) : ℚ := ∑ k in Finset.range (n + 1), k / (Nat.choose n k : ℚ)

theorem simplify_ratio {n : ℕ} (hn : 0 < n) : 
  (c_n n) / (d_n n) = n :=
sorry

end simplify_ratio_l587_587223


namespace line_equation_through_origin_and_circle_chord_length_l587_587114

theorem line_equation_through_origin_and_circle_chord_length 
  (x y : ℝ) 
  (h : x^2 + y^2 - 2 * x - 4 * y + 4 = 0) 
  (chord_length : ℝ) 
  (h_chord : chord_length = 2) 
  : 2 * x - y = 0 := 
sorry

end line_equation_through_origin_and_circle_chord_length_l587_587114


namespace dave_more_than_derek_l587_587913

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l587_587913


namespace sqrt_simplify_l587_587651

theorem sqrt_simplify (a b x : ℝ) (h : a < b) (hx1 : x + b ≥ 0) (hx2 : x + a ≤ 0) :
  Real.sqrt (-(x + a)^3 * (x + b)) = -(x + a) * (Real.sqrt (-(x + a) * (x + b))) :=
by
  sorry

end sqrt_simplify_l587_587651


namespace max_value_f_l587_587054

/-- Define the function f(x) -/
def f (x : ℝ) : ℝ := (Real.cos x) ^ 3 + (Real.sin x) ^ 2 - Real.cos x

/-- Define the interval condition on the domain of the function -/
def valid_range (x : ℝ) : Prop := -1 <= Real.cos x ∧ Real.cos x <= 1

/-- Prove that the maximum value of f(x) in the valid_range is 32/27 -/
theorem max_value_f : ∃ x, valid_range x ∧ f x = 32 / 27 :=
by
  sorry

end max_value_f_l587_587054


namespace continuity_at_8_l587_587757

theorem continuity_at_8 :
  ∀ ε > 0, ∃ δ > 0, (∀ x, |x - 8| < δ → |(5 * x^2 + 5) - (5 * 8^2 + 5)| < ε) := by
  intro ε hε
  use ε / 85
  split
  { linarith }
  intros x hx
  calc
    |(5 * x^2 + 5) - (5 * 8^2 + 5)|
      = |5 * (x^2 - 64)| : by ring
  ... = 5 * |x^2 - 64| : by rw abs_mul
  ... < 5 * (ε / 5) : by { apply mul_lt_mul_of_pos_left _ (by norm_num : 5 > 0), linarith }
  ... = ε : by field_simp

end continuity_at_8_l587_587757


namespace solve_inequalities_l587_587035

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end solve_inequalities_l587_587035


namespace bn_geometric_l587_587598

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = q * a n

def sum_of_three_consecutive (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a (3*n - 2) + a (3*n - 1) + a (3*n)

theorem bn_geometric (hq : q ≠ 1) (ha : geometric_sequence a q):
  let b : ℕ → ℝ := fun n => sum_of_three_consecutive a n in
  geometric_sequence b (q^3) :=
sorry

end bn_geometric_l587_587598


namespace monotonic_intervals_maximum_value_on_interval_l587_587276

-- Define the function f(x)
def f (x : ℝ) : ℝ := ln x - x^2 + x + 2

-- Define the derivative of the function f
def f' (x : ℝ) : ℝ := (1 / x) - 2 * x + 1

-- Conditions and statements encapsulated in Lean

-- 1. Prove intervals of monotonicity
theorem monotonic_intervals : 
  (∀ x, 0 < x ∧ x < 1 → f' x > 0) ∧ (∀ x, x > 1 → f' x < 0) := sorry

-- 2. Prove maximum value on the interval (0, a] for a > 0
theorem maximum_value_on_interval (a : ℝ) (ha : a > 0) : 
  (f a = ln a - a^2 + a + 2 ∨ f a = 2) ∧ 
  ((0 < a ∧ a ≤ 1 → f a = ln a - a^2 + a + 2) ∧ 
  (a > 1 → f a = 2)) := sorry

end monotonic_intervals_maximum_value_on_interval_l587_587276


namespace jogging_time_l587_587835

-- Definitions
def distance_covered : Real := 100  -- meters
def time_taken : Real := 4  -- minutes
def lap_distance : Real := 400  -- meters
def num_laps : Real := 2
def jogging_rate : Real := distance_covered / time_taken  -- meters per minute
def total_distance : Real := lap_distance * num_laps

-- Theorem statement
theorem jogging_time :
  total_distance / jogging_rate = 32 :=
by 
  sorry

end jogging_time_l587_587835


namespace arman_age_in_years_l587_587169

theorem arman_age_in_years (A S y : ℕ) (h1: A = 6 * S) (h2: S = 2 + 4) (h3: A + y = 40) : y = 4 :=
sorry

end arman_age_in_years_l587_587169


namespace platform_length_l587_587123

theorem platform_length (train_length : ℕ) (time_cross_platform : ℕ) (time_cross_pole : ℕ) (train_speed : ℕ) (L : ℕ)
  (h1 : train_length = 500) 
  (h2 : time_cross_platform = 65) 
  (h3 : time_cross_pole = 25) 
  (h4 : train_speed = train_length / time_cross_pole)
  (h5 : train_speed = (train_length + L) / time_cross_platform) :
  L = 800 := 
sorry

end platform_length_l587_587123


namespace average_rate_640_miles_trip_l587_587731

theorem average_rate_640_miles_trip 
  (total_distance : ℕ) 
  (first_half_distance : ℕ) 
  (first_half_rate : ℕ) 
  (second_half_time_multiplier : ℕ) 
  (first_half_time : ℕ := first_half_distance / first_half_rate)
  (second_half_time : ℕ := second_half_time_multiplier * first_half_time)
  (total_time : ℕ := first_half_time + second_half_time)
  (average_rate : ℕ := total_distance / total_time) : 
  total_distance = 640 ∧ 
  first_half_distance = 320 ∧ 
  first_half_rate = 80 ∧ 
  second_half_time_multiplier = 3 → 
  average_rate = 40 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  rw [h1, h2, h3, h4] at *
  have h5 : first_half_time = 320 / 80 := rfl
  have h6 : second_half_time = 3 * (320 / 80) := rfl
  have h7 : total_time = (320 / 80) + 3 * (320 / 80) := rfl
  have h8 : average_rate = 640 / (4 + 12) := rfl
  have h9 : average_rate = 640 / 16 := rfl
  have average_rate_correct : average_rate = 40 := rfl
  exact average_rate_correct

end average_rate_640_miles_trip_l587_587731


namespace mike_avg_speed_l587_587741

/-
  Given conditions:
  * total distance d = 640 miles
  * half distance h = 320 miles
  * first half average rate r1 = 80 mph
  * time for first half t1 = h / r1 = 4 hours
  * second half time t2 = 3 * t1 = 12 hours
  * total time tt = t1 + t2 = 16 hours
  * total distance d = 640 miles
  * average rate for entire trip should be (d/tt) = 40 mph.
  
  The goal is to prove that the average rate for the entire trip is 40 mph.
-/
theorem mike_avg_speed:
  let d := 640 in
  let h := 320 in
  let r1 := 80 in
  let t1 := h / r1 in
  let t2 := 3 * t1 in
  let tt := t1 + t2 in
  let avg_rate := d / tt in
  avg_rate = 40 := by
  sorry

end mike_avg_speed_l587_587741


namespace circle_area_from_circumference_l587_587045

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end circle_area_from_circumference_l587_587045


namespace midpoints_collinear_or_equilateral_l587_587810

-- Definitions:
structure Point :=
(x : ℝ)
(y : ℝ)

structure EquilateralTriangle :=
(A B C : Point)

-- Congruence definition for equilateral triangles
def congruent (T1 T2 : EquilateralTriangle) : Prop :=
  -- Placeholder for congruence condition which includes side and angle equality
   sorry

-- Midpoint definition
def midpoint (P Q : Point) : Point :=
  {x := (P.x + Q.x) / 2, y := (P.y + Q.y) / 2}

-- Collinearity definition for three points
def collinear (P Q R : Point) : Prop :=
  (Q.y - P.y) * (R.x - Q.x) = (R.y - Q.y) * (Q.x - P.x)

-- Equilateral triangle formed by three points
def equilateral_triangle (P Q R : Point) : Prop :=
  let d1 := (P.x - Q.x)^2 + (P.y - Q.y)^2 in
  let d2 := (Q.x - R.x)^2 + (Q.y - R.y)^2 in
  let d3 := (R.x - P.x)^2 + (R.y - P.y)^2 in
  d1 = d2 ∧ d2 = d3

-- The main theorem
theorem midpoints_collinear_or_equilateral {T1 T2 : EquilateralTriangle}
  (h_congruent : congruent T1 T2) :
  let A0 := midpoint T1.A T2.A,
      B0 := midpoint T1.B T2.B,
      C0 := midpoint T1.C T2.C in
  collinear A0 B0 C0 ∨ equilateral_triangle A0 B0 C0 :=
sorry

end midpoints_collinear_or_equilateral_l587_587810


namespace floor_S_value_l587_587364

theorem floor_S_value
  (a b c d : ℝ)
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (h_ab_squared : a^2 + b^2 = 1458)
  (h_cd_squared : c^2 + d^2 = 1458)
  (h_ac_product : a * c = 1156)
  (h_bd_product : b * d = 1156) :
  (⌊a + b + c + d⌋ = 77) := 
sorry

end floor_S_value_l587_587364


namespace geometric_sequence_ratio_l587_587974

theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ)
  (hq_pos : 0 < q)
  (h_geom : ∀ n, a (n + 1) = a n * q) 
  (h_arith : 2 * (1/2) * a 2 = 3 * a 0 + 2 * a 1) :
  (a 10 + a 12) / (a 7 + a 9) = 27 :=
sorry

end geometric_sequence_ratio_l587_587974


namespace population_increase_l587_587423

theorem population_increase (birth_rate : ℝ) (death_rate : ℝ) (initial_population : ℝ) :
  initial_population = 1000 →
  birth_rate = 32 / 1000 →
  death_rate = 11 / 1000 →
  ((birth_rate - death_rate) / initial_population) * 100 = 2.1 :=
by
  sorry

end population_increase_l587_587423


namespace angle_QPS_l587_587808

-- Definitions of the points and angles
variables (P Q R S : Point)
variables (angle : Point → Point → Point → ℝ)

-- Conditions about the isosceles triangles and angles
variables (isosceles_PQR : PQ = QR)
variables (isosceles_PRS : PR = RS)
variables (R_inside_PQS : ¬(R ∈ convex_hull ℝ {P, Q, S}))
variables (angle_PQR : angle P Q R = 50)
variables (angle_PRS : angle P R S = 120)

-- The theorem we want to prove
theorem angle_QPS : angle Q P S = 35 :=
sorry -- Proof goes here

end angle_QPS_l587_587808


namespace max_value_is_seven_l587_587657

theorem max_value_is_seven (a : ℕ) (hpos : 0 < a) :
    ∃ x, ∃ (y : ℝ), y = x + real.sqrt (13 - 2 * a * x) ∧
        (∀ b, (b : ℝ) = x + real.sqrt (13 - 2 * a * x) → y ≥ b) ∧ (∃ n : ℕ, n = y) → y = 7 :=
by
  sorry

end max_value_is_seven_l587_587657


namespace boys_girls_lim_nth_root_l587_587919

-- Define the main structures and functions needed for the problem.
noncomputable def boys_girls_probability (n : ℕ) : ℝ :=
  let total_set := {1, 2, 3, 4, 5}
  let prob_cond := λ k, 
    (k! * ∑ i in finset.range k, (-1:ℤ)^i * nat.choose k i * (k - i)^n : ℝ) 
    * ((5 - k:ℝ) / 5)^n / 5^n
  ∑ k in finset.range 1 5, prob_cond k

noncomputable def limit_of_nth_root_probability : ℝ :=
  (6:ℝ) / 25

-- Prove that as n approaches infinity, the nth root of boys_girls_probability approaches the limit.
theorem boys_girls_lim_nth_root (n : ℕ) : 
  let p_n := boys_girls_probability n
  let limit := limit_of_nth_root_probability
  (real.sqrt (p_n ^ (1 / (n:ℝ)))) = limit := sorry

end boys_girls_lim_nth_root_l587_587919


namespace no_one_is_always_largest_l587_587036

theorem no_one_is_always_largest (a b c d : ℝ) :
  a - 2 = b + 3 ∧ a - 2 = c * 2 ∧ a - 2 = d + 5 →
  ∀ x, (x = a ∨ x = b ∨ x = c ∨ x = d) → (x ≤ c ∨ x ≤ a) :=
by
  -- The proof requires assuming the conditions and showing that no variable is always the largest.
  intro h cond
  sorry

end no_one_is_always_largest_l587_587036


namespace number_of_three_digit_whole_numbers_with_digit_sum_24_l587_587236

theorem number_of_three_digit_whole_numbers_with_digit_sum_24 : 
  (finset.filter (λ n, (let a := n / 100 in 
                        let b := (n % 100) / 10 in 
                        let c := n % 10 in 
                        a + b + c = 24) 
                  (finset.Icc 100 999)).card = 4 := 
begin
  sorry
end

end number_of_three_digit_whole_numbers_with_digit_sum_24_l587_587236


namespace friend3_possible_games_l587_587804

-- Define the number of games each friend played
def friend1_games : ℕ := 25
def friend2_games : ℕ := 17

-- Define the proof problem
theorem friend3_possible_games (n : ℕ) :
  (n = 34 → ∃ x y z : ℕ, x + z = friend1_games ∧ y + z = friend2_games ∧ x + y = n) ∧
  (n = 35 → ¬ ∃ x y z : ℕ, x + z = friend1_games ∧ y + z = friend2_games ∧ x + y = n) ∧
  (n = 56 → ¬ ∃ x y z : ℕ, x + z = friend1_games ∧ y + z = friend2_games ∧ x + y = n) :=
  by {
    split,
    { intro h, rw h, sorry }, -- Prove for n = 34
    { split,
      { intro h, rw h, sorry }, -- Prove for n = 35
      { intro h, rw h, sorry }  -- Prove for n = 56
    }
  }

end friend3_possible_games_l587_587804


namespace smallest_positive_m_for_integral_solutions_l587_587826

theorem smallest_positive_m_for_integral_solutions :
  ∃ m : ℕ, (∀ x : ℚ, (10 * x^2 - (m : ℚ) * x + 180 = 0 → x ∈ ℤ)) ∧ m = 90 :=
by
  sorry

end smallest_positive_m_for_integral_solutions_l587_587826


namespace bisect_AM_l587_587409

-- Definitions of points and reflection condition
structure Point where
  x : ℝ
  y : ℝ

def reflection (p q : Point) : Point :=
  ⟨2 * q.x - p.x, 2 * q.y - p.y⟩

-- Definitions of the square ABCD with side length 4 and its center
def A : Point := ⟨0, 0⟩
def B : Point := ⟨4, 0⟩
def C : Point := ⟨4, 4⟩
def D : Point := ⟨0, 4⟩
def M : Point := ⟨2, 2⟩

-- Define E as the reflection of M with respect to C
def E : Point := reflection M C

-- Intersection of the line AM with circumcircle of triangle BDE
noncomputable def circumcircle (p1 p2 p3 : Point) : set Point := sorry

noncomputable def lineAM : set Point := {pt : Point | pt.y = pt.x}

noncomputable def intersection (circle : set Point) (line : set Point) : Point := sorry

-- Prove that the intersection point S bisects AM
theorem bisect_AM :
  let S := intersection (circumcircle B D E) lineAM in
  (S.x, S.y) = (1, 1) :=
by
  sorry

end bisect_AM_l587_587409


namespace R_and_D_per_increase_l587_587537

def R_and_D_t : ℝ := 3013.94
def Delta_APL_t2 : ℝ := 3.29

theorem R_and_D_per_increase :
  R_and_D_t / Delta_APL_t2 = 916 := by
  sorry

end R_and_D_per_increase_l587_587537


namespace probability_even_sum_l587_587797

noncomputable def balls : Set ℕ := {1, 2, 3, 4}

def pairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  {p | p.1 ∈ s ∧ p.2 ∈ s ∧ p.1 < p.2}

def is_even (n : ℕ) : Prop :=
  n % 2 = 0

def even_sum_pairs (s : Set ℕ) : Set (ℕ × ℕ) :=
  {p | p ∈ pairs s ∧ is_even (p.1 + p.2)}

theorem probability_even_sum :
    (even_sum_pairs balls).card = 2 ∧ (pairs balls).card = 6 →
    (even_sum_pairs balls).card / (pairs balls).card = 1 / 3 :=
by
  intros h
  sorry

end probability_even_sum_l587_587797


namespace sum_of_primitive_roots_mod_11_l587_587471

theorem sum_of_primitive_roots_mod_11 : 
  (∑ x in {1, 2, 3, 4, 5, 6, 7, 8}, if is_primitive_root 11 x then x else 0) = 23 :=
by sorry

end sum_of_primitive_roots_mod_11_l587_587471


namespace cos_pi_half_sin_eq_sin_pi_half_cos_has_5_solutions_l587_587298

open Real

theorem cos_pi_half_sin_eq_sin_pi_half_cos_has_5_solutions :
  ∀ (a b : ℝ), (∀ x ∈ Icc (0 : ℝ) (2 * π), cos (a * sin x) = sin (b * cos x)) ↔ (a = b ∧ ∃ n ∈ (finset.range 6).erase 0, ∀ x ∈ Icc (0 : ℝ) (2 * π), cos (a * sin (n * x / 5)) = sin (a * cos (n * x / 5))) :=
begin
  sorry
end

end cos_pi_half_sin_eq_sin_pi_half_cos_has_5_solutions_l587_587298


namespace expression_divisibility_l587_587016

theorem expression_divisibility (x y : ℤ) (k_1 k_2 : ℤ) (h1 : 2 * x + 3 * y = 17 * k_1) :
    ∃ k_2 : ℤ, 9 * x + 5 * y = 17 * k_2 :=
by
  sorry

end expression_divisibility_l587_587016


namespace find_f_at_5_l587_587050

noncomputable def f : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : f(x - y) = f(x) * f(y)
axiom f_nonzero (x : ℝ) : f(x) ≠ 0
axiom f_at_2 : f(2) = 1

theorem find_f_at_5 : f(5) = 1 ∨ f(5) = -1 := by
  sorry

end find_f_at_5_l587_587050


namespace smallest_prime_divisor_sum_odd_powers_l587_587461

theorem smallest_prime_divisor_sum_odd_powers :
  (∃ p : ℕ, prime p ∧ p ∣ (3^15 + 11^21) ∧ p = 2) :=
by
  have h1 : 3^15 % 2 = 1 := by sorry
  have h2 : 11^21 % 2 = 1 := by sorry
  have h3 : (3^15 + 11^21) % 2 = 0 := by
    rw [← Nat.add_mod, h1, h2]
    exact Nat.mod_add_mod 1 1 2
  use 2
  constructor
  · exact Nat.prime_two
  · rw [Nat.dvd_iff_mod_eq_zero, h3] 
  · rfl

end smallest_prime_divisor_sum_odd_powers_l587_587461


namespace mass_percentage_O_in_Al2O3_l587_587941

-- Definitions for the conditions
def atomic_mass_Al : ℝ := 26.98
def atomic_mass_O : ℝ := 16.00
def molar_mass_Al2O3 : ℝ := (2 * atomic_mass_Al) + (3 * atomic_mass_O)
def mass_O_in_Al2O3 : ℝ := 3 * atomic_mass_O

-- Theorem statement
theorem mass_percentage_O_in_Al2O3 :
  (mass_O_in_Al2O3 / molar_mass_Al2O3) * 100 ≈ 47.07 :=
sorry

end mass_percentage_O_in_Al2O3_l587_587941


namespace domain_of_f_eq_l587_587776

open Set

def f (x : ℝ) : ℝ := (Real.sqrt (x + 2)) / (x - 1)

noncomputable def domain_f : Set ℝ :=
  {x | x ≥ -2 ∧ x ≠ 1}

theorem domain_of_f_eq :
  ∀ x, x ∈ domain_f ↔ (x ≥ -2 ∧ x ≠ 1) :=
by
  intro x
  rw [domain_f]
  exact Iff.rfl

#check domain_of_f_eq

end domain_of_f_eq_l587_587776


namespace find_d_l587_587658

-- Define the quadratic equation with given constants
def quadratic (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the roots given in the problem
def root1 : ℝ := (-14 + Real.sqrt 14) / 4
def root2 : ℝ := (-14 - Real.sqrt 14) / 4

-- State the problem: If the quadratic equation 2x^2 + 14x + d = 0 has roots root1 and root2, then d must be 91/4
theorem find_d (d : ℝ) (h : ∀ x : ℝ, quadratic 2 14 d x = 0 ↔ (x = root1 ∨ x = root2)) : d = 91 / 4 :=
sorry

end find_d_l587_587658


namespace no_57_numbers_without_sum_100_l587_587105

theorem no_57_numbers_without_sum_100 : 
  ¬ ∃ (s : Finset ℕ), s.card = 57 ∧ 
  (∀ (a ∈ s) (b ∈ s), a + b ≠ 100) ∧ 
  (∀ (a ∈ s), 10 ≤ a ∧ a ≤ 99) :=
by 
  sorry

end no_57_numbers_without_sum_100_l587_587105


namespace arithmetic_sequence_sum_l587_587780

variable (a : ℕ → ℤ)
variable (d : ℤ)

-- Define the conditions
def a_5 := a 5
def a_6 := a 6
def a_7 := a 7

axiom cond1 : a_5 = 11
axiom cond2 : a_6 = 17
axiom cond3 : a_7 = 23

noncomputable def sum_first_four_terms : ℤ :=
  a 1 + a 2 + a 3 + a 4

theorem arithmetic_sequence_sum :
  a_5 = 11 → a_6 = 17 → a_7 = 23 → sum_first_four_terms a = -16 :=
by
  intros h5 h6 h7
  sorry

end arithmetic_sequence_sum_l587_587780


namespace general_formula_sum_of_reciprocals_l587_587999

-- Define the sequence
def a : ℕ → ℕ
| 0     := 1
| (n+1) := a n + n + 1

-- Part 1: Prove the general formula for a_n
theorem general_formula (n : ℕ) : a n = n * (n + 1) / 2 := 
  sorry

-- Part 2: Prove the sum of the first n terms of the sequence {1 / a_n}
theorem sum_of_reciprocals (n : ℕ) : (∑ k in finset.range n, 2 / (k * (k+1))) = (2 * ↑n) / (↑n + 1) := 
  sorry

end general_formula_sum_of_reciprocals_l587_587999


namespace caterpillar_length_difference_l587_587747

def orange_length : ℝ := 1.17
def green_length : ℝ := 3
def blue_length : ℝ := 2.38
def yellow_length : ℝ := 4.29
def red_length : ℝ := 2.94

theorem caterpillar_length_difference :
  let lengths := [orange_length, green_length, blue_length, yellow_length, red_length]
  in (list.maximum lengths).get_or_else 0 - (list.minimum lengths).get_or_else 0 = 3.12 :=
by
  sorry

end caterpillar_length_difference_l587_587747


namespace inscribed_square_side_length_l587_587020

theorem inscribed_square_side_length :
  ∀ (A B C W X Y Z : ℝ), 
    let AB := dist A B,
        BC := dist B C,
        AC := dist A C in
    AB = 3 ∧ BC = 4 ∧ AC = 5 ∧
    W ∈ seg A B ∧ X ∈ seg A C ∧ Y ∈ seg A C ∧ Z ∈ seg B C →
    ∃ s : ℝ, s = 60 / 37 := by
  intros A B C W X Y Z h,
  sorry

end inscribed_square_side_length_l587_587020


namespace volume_of_parabola_solid_volume_of_ellipse_solid_volume_of_parabola_line_solid_volume_of_parametric_curve_solid_volume_of_shifted_parabola_solid_l587_587544

-- 1. Volume of solid bounded by \( y^2 = 2px \) and \( x = a \) around the \( O x \)-axis.
theorem volume_of_parabola_solid (p a : ℝ) : ∫ x in 0..a, 2 * p * x = π * p * a^2 := 
sorry

-- 2. Volume of solid bounded by ellipse \(\frac{x^2}{a^2} + \frac{y^2}{b^2} = 1\) around the \( O y \)-axis.
theorem volume_of_ellipse_solid (a b : ℝ) : ∫ y in -b..b, a^2 * (1 - (y^2 / b^2)) = (4 / 3) * π * a^2 * b :=
sorry

-- 3. Volume of solid bounded by \( 2y = x^2 \) and \( 2x + 2y - 3 = 0 \) around the \( O x \)-axis.
theorem volume_of_parabola_line_solid (a : ℝ) : 
  let f := λ x, (2*x - 3) / 2
  ∫ x in 0..a, (x^2 / 2) - ((2*x - 3) / 2)^2 = _ := 
sorry

-- 4. Volume of solid bounded by parameteric curve \( x = a \cos^3 t \), \( y = a \sin^3 t \) around the \( O x \)-axis.
theorem volume_of_parametric_curve_solid (a : ℝ) : 
  ∫ t in 0..π, (a * sin t ^ 3)^2 * (a * cos t ^ 3)' = _ := 
sorry

-- 5. Volume of solid bounded by \( y = 4 - x^2 \) and \( y = 0 \) around the line \( x = 3 \).
theorem volume_of_shifted_parabola_solid : 
  ∫ x in -2..2, (4 - x^2)^2 (x + 3 / a) dx = _ :=
sorry

end volume_of_parabola_solid_volume_of_ellipse_solid_volume_of_parabola_line_solid_volume_of_parametric_curve_solid_volume_of_shifted_parabola_solid_l587_587544


namespace rational_count_is_four_l587_587524

theorem rational_count_is_four :
  let options := [-2, Real.sqrt 4, (Real.sqrt 2) / 2, 3.14, 22 / 3] in
  let rational_count := options.countp (λ x, ∃ n m : ℤ, m ≠ 0 ∧ x = n / m) in
  rational_count = 4 :=
by
  let options := [-2, Real.sqrt 4, (Real.sqrt 2) / 2, 3.14, 22 / 3]
  have h1 : ∃ n m : ℤ, m ≠ 0 ∧ -2 = n / m := ⟨-2, 1, by norm_num, by norm_num⟩
  have h2 : ∃ n m : ℤ, m ≠ 0 ∧ Real.sqrt 4 = n / m := ⟨2, 1, by norm_num, by norm_num⟩
  have h3 : ¬ ∃ n m : ℤ, m ≠ 0 ∧ (Real.sqrt 2 / 2) = n / m := sorry
  have h4 : ∃ n m : ℤ, m ≠ 0 ∧ 3.14 = n / m := ⟨314, 100, by norm_num, by norm_num⟩
  have h5 : ∃ n m : ℤ, m ≠ 0 ∧ (22 / 3) = n / m := ⟨22, 3, by norm_num, by norm_num⟩
  have rational_options := [h1, h2, h3, h4, h5].countp (λ x, ∃ n m : ℤ, m ≠ 0 ∧ x = n / m)
  show rational_count = 4 from 
  begin
    simp only [countp] at rational_options,
    exact rational_options
  end, 
  sorry 

end rational_count_is_four_l587_587524


namespace unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l587_587194

theorem unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16 
  (b : ℝ) : 
  (∃ (x : ℝ), bx^2 + 7*x + 4 = 0 ∧ ∀ (x' : ℝ), bx^2 + 7*x' + 4 ≠ 0) ↔ b = 49 / 16 :=
by
  sorry

end unique_intersection_y_eq_bx2_5x_2_y_eq_neg2x_neg2_iff_b_eq_49_div_16_l587_587194


namespace total_digits_first_1500_even_integers_l587_587541

theorem total_digits_first_1500_even_integers : 
  let even_integers := list.range' 2 1500 \;
  let counts := [
    (list.filter (λ n, n < 10) even_integers).length, 
    (list.filter (λ n, 10 ≤ n ∧ n < 100) even_integers).length * 2, 
    (list.filter (λ n, 100 ≤ n ∧ n < 1000) even_integers).length * 3, 
    (list.filter (λ n, 1000 ≤ n) even_integers).length * 4] \;
  even_integers.sum_mtl counts = 5448 :=
by
  sorry

end total_digits_first_1500_even_integers_l587_587541


namespace problem_l587_587700

-- Definitions based on conditions
variables {a b : ℝ} {f : ℝ → ℝ}
variables {λ : ℝ} (h₁ : a ≤ b) (h₂ : f ∈ continuous_differentiable_on ℝ a b) (h₃ : f a = 0) (h₄ : λ > 0)
variables (h₅ : ∀ x ∈ set.Icc a b, |(derivative f x)| ≤ λ * |f x|)

-- Statement of the theorem
theorem problem (a b : ℝ) (f : ℝ → ℝ) (λ : ℝ) 
  (h₁ : a ≤ b) (h₂ : f ∈ continuous_differentiable_on ℝ a b) (h₃ : f a = 0) 
  (h₄ : λ > 0) (h₅ : ∀ x ∈ set.Icc a b, |(derivative f x)| ≤ λ * |f x|) : ∀ x ∈ set.Icc a b, f x = 0 :=
sorry

end problem_l587_587700


namespace remainder_of_division_l587_587089

theorem remainder_of_division :
  Nat.mod 4536 32 = 24 :=
sorry

end remainder_of_division_l587_587089


namespace mean_of_combined_set_l587_587055

theorem mean_of_combined_set
  (mean1 : ℕ → ℕ → ℕ) -- definition for calculating the mean
  (n1 n2 n3 : ℕ) -- size of the sets
  (m1 m2 m3 : ℕ) -- means of the sets
  (h1 : mean1 7 15 = m1)
  (h2 : mean1 8 30 = m2)
  (h3 : mean1 5 18 = m3)
  (h_mean1 :  m1 = 15)
  (h_mean2 :  m2 = 30)
  (h_mean3 :  m3 = 18) :
  (mean1 (n1 + n2 + n3) ((7 * 15) + (8 * 30) + (5 * 18)) = 21.75) :=
by
  -- sorry is used to skip the proof
  sorry

end mean_of_combined_set_l587_587055


namespace Paul_homework_hours_l587_587558

-- Conditions
def total_weeknights : ℕ := 5
def practice_nights : ℕ := 2
def homework_avg_hours_per_night : ℕ := 3

-- Proof statement
theorem Paul_homework_hours : 
  let nights_available := total_weeknights - practice_nights in 
  let total_homework_weeknights := nights_available * homework_avg_hours_per_night in 
  total_homework_weeknights = 9 :=
by
  let nights_available := total_weeknights - practice_nights
  let total_homework_weeknights := nights_available * homework_avg_hours_per_night
  sorry

end Paul_homework_hours_l587_587558


namespace max_abs_z5_l587_587252

theorem max_abs_z5 (z1 z2 z3 z4 z5 : ℂ)
  (h1 : abs z1 ≤ 1)
  (h2 : abs z2 ≤ 1)
  (h3 : abs (2 * z3 - (z1 + z2)) ≤ abs (z1 - z2))
  (h4 : abs (2 * z4 - (z1 + z2)) ≤ abs (z1 - z2))
  (h5 : abs (2 * z5 - (z3 + z4)) ≤ abs (z3 - z4)) :
  abs z5 ≤ real.sqrt 3 := sorry

end max_abs_z5_l587_587252


namespace inverse_of_f_l587_587085

-- Define the function f
def f (x : ℝ) : ℝ := 7 - 8 * x

-- Define the proposed inverse function g
def g (x : ℝ) : ℝ := (7 - x) / 8

-- State the theorem that g is the inverse of f
theorem inverse_of_f : ∀ x, f (g x) = x ∧ g (f x) = x :=
by
  intros
  sorry

end inverse_of_f_l587_587085


namespace prob_xy_minus_x_minus_y_divisible_by_3_l587_587446

theorem prob_xy_minus_x_minus_y_divisible_by_3 :
  let S := { x : ℕ | 1 ≤ x ∧ x ≤ 12 }
  ∃ (x y : ℕ), x ≠ y ∧ x ∈ S ∧ y ∈ S →
  (xy - x - y) % 3 = 0 → 
  Pr { (xy - x - y) % 3 = 0 | x ≠ y ∧ x ∈ S ∧ y ∈ S } = 19 / 33 :=
sorry

end prob_xy_minus_x_minus_y_divisible_by_3_l587_587446


namespace parabola_vertex_y_intercept_l587_587947

theorem parabola_vertex_y_intercept :
  let a := 3
  let b := -6
  let c := 2
  let y := λ x : ℝ, a * x^2 + b * x + c
  (∃ x y : ℝ, x = -b / (2 * a) ∧ y = a * x ^ 2 + b * x + c ∧ x = 1 ∧ y = -1) ∧
  (∃ y : ℝ, y = a * (0 : ℝ)^2 + b * 0 + c ∧ y = 2) :=
by {
  sorry
}

end parabola_vertex_y_intercept_l587_587947


namespace grasshopper_returns_to_start_l587_587772

-- Define the point O
variable (O : Point)

-- Define the circles S_i and points X_i on S_i
variables {n : ℕ} (S : Fin n → Circle) (X : Fin n → Point)

-- Define the rotational homothety transformation P_i
variable (P : Fin n → (Point → Point))

-- Assume each Si passes through point O
def circles_pass_through_O := ∀ i, O ∈ S i

-- Assume the grasshopper's jumps are according to the specified lines
def hops_definition := ∀ i, (X (i + 1)) = (P i) (X i)

-- Prove that after n jumps, the grasshopper returns to the starting point
theorem grasshopper_returns_to_start :
  circles_pass_through_O O S →
  hops_definition S X P →
  ((P (n-1)) ∘ (P (n-2)) ∘ ... ∘ (P 0)) (X 0) = X 0 :=
by
  intro h1 h2
  sorry

end grasshopper_returns_to_start_l587_587772


namespace calculate_value_l587_587164

theorem calculate_value (N p q m n : ℕ) 
  (H₁ : p + q = N - 1) 
  (H₂ : m + n = N) : 
  (p - m) + (q - n) = -1 :=
by
  sorry

end calculate_value_l587_587164


namespace smallest_positive_m_for_integral_solutions_l587_587825

theorem smallest_positive_m_for_integral_solutions :
  ∃ m : ℕ, (∀ x : ℚ, (10 * x^2 - (m : ℚ) * x + 180 = 0 → x ∈ ℤ)) ∧ m = 90 :=
by
  sorry

end smallest_positive_m_for_integral_solutions_l587_587825


namespace monotonic_increasing_on_interval_range_on_interval_l587_587991

-- Proof Problem 1: Monotonicity on (2, +∞)
theorem monotonic_increasing_on_interval (f : ℝ → ℝ) (m : ℝ) (h : f = λ x, x + m / x) :
  (2 < x₁ ∧ 2 < x₂ ∧ x₁ < x₂) → f x₁ < f x₂ :=
by sorry

-- Proof Problem 2: Range on [5/2, 10/3]
theorem range_on_interval (f : ℝ → ℝ) (m : ℝ) (h : f = λ x, x + m / x) :
  5/2 ≤ x₁ ∧ x₁ ≤ 10/3 → f x₁ ∈ set.Icc (41 / 10) (68 / 15) :=
by sorry

end monotonic_increasing_on_interval_range_on_interval_l587_587991


namespace number_of_whole_numbers_between_sqrt2_and_3e_is_7_l587_587644

noncomputable def number_of_whole_numbers_between_sqrt2_and_3e : ℕ :=
  let sqrt2 : ℝ := Real.sqrt 2
  let e : ℝ := Real.exp 1
  let small_int := Nat.ceil sqrt2 -- This is 2
  let large_int := Nat.floor (3 * e) -- This is 8
  large_int - small_int + 1 -- The number of integers between small_int and large_int (inclusive)

theorem number_of_whole_numbers_between_sqrt2_and_3e_is_7 :
  number_of_whole_numbers_between_sqrt2_and_3e = 7 := by
  sorry

end number_of_whole_numbers_between_sqrt2_and_3e_is_7_l587_587644


namespace percentage_first_acid_solution_l587_587503

theorem percentage_first_acid_solution:
  ∃ P : ℝ, ((P / 100) * 420 + (30 / 100) * 210 = (50 / 100) * 630) ∧ P = 60 :=
by
  use 60
  field_simp
  linarith
  sorry

end percentage_first_acid_solution_l587_587503


namespace sum_reverse_contains_even_l587_587120

theorem sum_reverse_contains_even (N : ℕ) (h_len : (N.to_digits.to_list.length = 17)) :
  ∃ d ∈ (N + N.reverse_digits).to_digits, even d :=
sorry

end sum_reverse_contains_even_l587_587120


namespace jelly_beans_remaining_l587_587072

theorem jelly_beans_remaining :
  let initial_jelly_beans := 8000
  let num_first_group := 6
  let num_last_group := 4
  let last_group_took := 400
  let first_group_took := 2 * last_group_took
  let last_group_total := last_group_took * num_last_group
  let first_group_total := first_group_took * num_first_group
  let remaining_jelly_beans := initial_jelly_beans - (first_group_total + last_group_total)
  remaining_jelly_beans = 1600 :=
by {
  -- Define the initial number of jelly beans
  let initial_jelly_beans := 8000
  -- Number of people in first and last groups
  let num_first_group := 6
  let num_last_group := 4
  -- Jelly beans taken by last group and first group per person
  let last_group_took := 400
  let first_group_took := 2 * last_group_took
  -- Calculate total jelly beans taken by last group and first group
  let last_group_total := last_group_took * num_last_group
  let first_group_total := first_group_took * num_first_group
  -- Jelly beans remaining
  let remaining_jelly_beans := initial_jelly_beans - (first_group_total + last_group_total)
  -- Proof of the theorem
  show remaining_jelly_beans = 1600, from sorry
}

end jelly_beans_remaining_l587_587072


namespace ghk_equilateral_l587_587891

noncomputable def is_equilateral_triangle (G H K : Point) : Prop :=
  dist G H = dist H K ∧ dist H K = dist K G

noncomputable def midpoint (P Q : Point) : Point :=
  Point.mk ((P.x + Q.x) / 2) ((P.y + Q.y) / 2)

theorem ghk_equilateral (O A B C D E F G H K : Point) (r : ℝ)
  (h1 : dist A B = r) (h2 : dist C D = r) (h3 : dist E F = r)
  (h4 : G = midpoint B C) (h5 : H = midpoint D E) (h6 : K = midpoint F A)
  (h7 : is_regular_hexagon A B C D E F O r) : is_equilateral_triangle G H K := sorry

end ghk_equilateral_l587_587891


namespace marcy_multiple_tickets_l587_587806

theorem marcy_multiple_tickets (m : ℕ) : 
  (26 + (m * 26 - 6) = 150) → m = 5 :=
by
  intro h
  sorry

end marcy_multiple_tickets_l587_587806


namespace monotonically_increasing_range_of_m_l587_587316

theorem monotonically_increasing_range_of_m (f : ℝ → ℝ)
  (h : ∀ x ∈ Icc (1 : ℝ) (2 : ℝ), f x = x^2 - 4 * x + m * Real.log x) :
  (∀ x ∈ Icc (1 : ℝ) (2 : ℝ), (2 * x^2 - 4 * x + m) / x ≥ 0) ↔ m ∈ Set.Ici 2 :=
by
  sorry

end monotonically_increasing_range_of_m_l587_587316


namespace grid_rectangle_diagonals_two_corners_l587_587136

-- Define a grid rectangle
structure GridRectangle (m n : ℕ) :=
(dominoes : set (set (ℕ × ℕ)))
(diagonals : set ((ℕ × ℕ) × (ℕ × ℕ)))
(domino_condition : ∀ d ∈ dominoes, ∃ a b, d = {a, b} ∧ |a.1 - b.1| + |a.2 - b.2| = 1)
(diagonal_condition : ∀ d ∈ dominoes, ∃ a b, (a, b) ∈ diagonals ∧ (b, a) ∉ diagonals ∧ (a ∉ {0, 0, m-1, 0, 0, n-1, m-1, n-1} ∨ b ∈ {m-1, 0, 0, n-1}))

-- Define the theorem to be proven
theorem grid_rectangle_diagonals_two_corners (m n : ℕ) (R : GridRectangle m n) :
  ∃! (corners : set (ℕ × ℕ)), corners = {(0, 0), (0, n-1), (m-1, 0), (m-1, n-1)} ∧
    (∃ a b ∈ R.diagonals, a ∈ corners ∧ b ∈ corners ∧ a ≠ b ∧ ∀ c ∈ R.diagonals, c ∉ corners) :=
sorry

end grid_rectangle_diagonals_two_corners_l587_587136


namespace find_yz_plane_intersection_radius_l587_587879

def sphere_center : ℝ × ℝ × ℝ := (3, -2, -10)
def xy_plane_circle_center : ℝ × ℝ × ℝ := (3, -2, 0)
def yz_plane_circle_center : ℝ × ℝ × ℝ := (0, -2, -10)
def xy_plane_circle_radius : ℝ := 2
def sphere_radius : ℝ := real.sqrt ((3-5)^2 + (-2+2)^2 + (-10-0)^2)
def yz_plane_circle_radius : ℝ := real.sqrt (sphere_radius ^ 2 - 3 ^ 2)

theorem find_yz_plane_intersection_radius : yz_plane_circle_radius = real.sqrt 95 :=
by
  sorry

end find_yz_plane_intersection_radius_l587_587879


namespace grandfather_7_times_older_after_8_years_l587_587865

theorem grandfather_7_times_older_after_8_years :
  ∃ x : ℕ, ∀ (g_age ng_age : ℕ), 50 < g_age ∧ g_age < 90 ∧ g_age = 31 * ng_age → g_age + x = 7 * (ng_age + x) → x = 8 :=
by
  sorry

end grandfather_7_times_older_after_8_years_l587_587865


namespace proof_problem_l587_587135

noncomputable def f : ℝ → ℝ :=
sorry -- Definition of f is not provided

theorem proof_problem (f_mono : ∀ x₁ x₂ : ℝ, (x₁ - x₂) * (f x₁ - f x₂) > 0) : f (-2) < f (1) ∧ f (1) < f (3) :=
by
  have h₁ : -2 < 1 :=
    by linarith
  have h₂ : 1 < 3 :=
    by linarith
  have f_mono_inc : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ :=
    by intros x₁ x₂ h
       have h_prod := f_mono x₁ x₂
       have h_diff : x₁ - x₂ < 0 :=
         by linarith
       have pos_neg : (f x₁ - f x₂) < 0 :=
         by linarith
       exact lt_trans h_diff pos_neg
  exact ⟨f_mono_inc (-2) 1 h₁, f_mono_inc 1 3 h₂⟩

end proof_problem_l587_587135


namespace fraction_red_after_tripling_l587_587662

-- Define the initial conditions
def initial_fraction_blue : ℚ := 4 / 7
def initial_fraction_red : ℚ := 1 - initial_fraction_blue
def triple_red_fraction (initial_red : ℚ) : ℚ := 3 * initial_red

-- Theorem statement
theorem fraction_red_after_tripling :
  let x := 1 -- Any number since it will cancel out
  let initial_red_marble := initial_fraction_red * x
  let total_marble := x
  let new_red_marble := triple_red_fraction initial_red_marble
  let new_total_marble := initial_fraction_blue * x + new_red_marble
  (new_red_marble / new_total_marble) = 9 / 13 :=
by
  sorry

end fraction_red_after_tripling_l587_587662


namespace chocolate_bar_weight_l587_587887

theorem chocolate_bar_weight :
  let square_weight := 6
  let triangles_count := 16
  let squares_count := 32
  let triangle_weight := square_weight / 2
  let total_square_weight := squares_count * square_weight
  let total_triangles_weight := triangles_count * triangle_weight
  total_square_weight + total_triangles_weight = 240 := 
by
  sorry

end chocolate_bar_weight_l587_587887


namespace largest_three_digit_perfect_square_and_cube_l587_587087

theorem largest_three_digit_perfect_square_and_cube :
  ∃ (n : ℕ), (100 ≤ n ∧ n ≤ 999) ∧ (∃ (a : ℕ), n = a^6) ∧ ∀ (m : ℕ), ((100 ≤ m ∧ m ≤ 999) ∧ (∃ (b : ℕ), m = b^6)) → m ≤ n := 
by 
  sorry

end largest_three_digit_perfect_square_and_cube_l587_587087


namespace weight_of_4_moles_of_Al2S3_correct_l587_587820

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06

def molecular_weight_Al2S3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_S)

def weight_of_4_moles_of_Al2S3 : ℝ := 4 * molecular_weight_Al2S3

theorem weight_of_4_moles_of_Al2S3_correct :
  weight_of_4_moles_of_Al2S3 = 600.56 :=
by
  have hAl2S3: molecular_weight_Al2S3 = 150.14 := by
    rw [molecular_weight_Al2S3, atomic_weight_Al, atomic_weight_S]
    norm_num
  rw [weight_of_4_moles_of_Al2S3, hAl2S3]
  norm_num
  sorry

end weight_of_4_moles_of_Al2S3_correct_l587_587820


namespace solve_triangle_l587_587685

noncomputable def triangle_sides_and_midpoint_distance
  (α β γ : ℝ)
  (b c : ℝ) :
  (a : ℝ) × (ad : ℝ) :=
  let a := b^2 + c^2 - 2 * b * c * Real.cos 120 in
  let ad := Real.sqrt (b^2 + c^2 - 2 * b * c * (-1/2)) / 2 in
  (a, ad)

theorem solve_triangle
  (A : ℝ)
  (b c : ℝ)
  (hA : A = 120*(Real.pi / 180))
  (hb : b = 3)
  (hc : c = 5) :
  ∃ a ad, triangle_sides_and_midpoint_distance A b c = (a, ad) ∧ a = 7 ∧ ad = Real.sqrt 19 / 2 :=
by
  sorry

end solve_triangle_l587_587685


namespace minimize_distance_sum_l587_587607

variable (A B C D : Point)

theorem minimize_distance_sum (A B C D : Point) :
  ∃ O : Point, 
    (convex_quadrilateral A B C D → O = intersection_of_diagonals A B C D) ∧
    (inside_triangle A B C D → ∃ (X : Point), X = point_inside_triangle A B C D) ∧
    (collinear_points A B C D → ∀ O : Point, O ∈ segment C D → sum_distances O A B C D = AB + CD) :=
sorry

end minimize_distance_sum_l587_587607


namespace log_max_value_eq_two_l587_587652

noncomputable def log_max_value (a b : ℝ) (h1 : a ≥ b) (h2 : b > Real.exp 1) : ℝ :=
  Real.log a (a ^ 2 / b) + Real.log b (b ^ 2 / a)

theorem log_max_value_eq_two (a b : ℝ) (h1 : a ≥ b) (h2 : b > Real.exp 1) : log_max_value a b h1 h2 = 2 :=
  sorry

end log_max_value_eq_two_l587_587652


namespace pentagon_area_50_l587_587393

def point := (ℝ × ℝ)

structure Pentagon :=
(A B C D E : point)

def area_rectangle (p1 p2 p3 p4 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
let ⟨x4, y4⟩ := p4 in
abs((x3 - x1) * (y2 - y1))

def area_triangle (p1 p2 p3 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

def y_coordinate_C (pent: Pentagon) : ℝ :=
let ⟨_, _, pC, _, _⟩ := pent in
pC.2

theorem pentagon_area_50 (h : ℝ) :
  let A := (0, 0) in
  let B := (0, 5) in
  let C := (3, h) in
  let D := (6, 5) in
  let E := (6, 0) in
  let rect_area := area_rectangle A B D E in
  let tri_area := area_triangle B C D in
  rect_area + tri_area = 50 :=
by
  sorry

end pentagon_area_50_l587_587393


namespace length_of_CP_l587_587766

open Real

theorem length_of_CP {A B C D P Q : ℝ × ℝ}
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 4))
  (hD : D = (0, 4))
  (hP : ∃ a : ℝ, 0 ≤ a ∧ a ≤ 4 ∧ P = (a, 0))
  (hQ : ∃ b : ℝ, 0 ≤ b ∧ b ≤ 4 ∧ Q = (b, 0))
  (hdivide : area (triangle C B P) = area (triangle C P Q) := (16 / 3)): 
  dist C P = 4 * sqrt 10 / 3 :=
sorry

end length_of_CP_l587_587766


namespace bob_wins_l587_587115

-- Define the grid structure and conditions
structure Grid :=
(rows cols : ℕ)
(cells : Fin (rows * cols) → ℚ)

def initial_grid : Grid :=
{ rows := 6,
  cols := 6,
  cells := λ _, 0 }

def is_black (g : Grid) (r : Fin g.rows) (c : Fin g.cols) : Prop :=
∀ c', g.cells (⟨r.1 * g.cols + c.1, sorry⟩) ≥ g.cells (⟨r.1 * g.cols + c'.1, sorry⟩)

def has_path (g : Grid) : Prop :=
∃ (path : List (Fin g.rows × Fin g.cols)),
  path.head = (⟨0, sorry⟩, ⟨0, sorry⟩) ∧ 
  path.last = (⟨g.rows - 1, sorry⟩, ⟨g.cols - 1, sorry⟩) ∧
  ∀ p ∈ path, is_black g p.1 p.2 ∧
  ∀ (p1 p2 : Fin g.rows × Fin g.cols), p1 ∈ path ∧ p2 ∈ path → adjacent p1 p2

def adjacent (p1 p2 : Fin 6 × Fin 6) : Prop :=
(p1.1.val + 1 = p2.1.val ∧ p1.2 = p2.2) ∨ 
(p1.1 = p2.1 ∧ p1.2.val + 1 = p2.2.val)

theorem bob_wins : ∃ (strategy : Grid → (Fin 6 × Fin 6) → ℚ), ∀ (g : Grid) (move : Fin 6 × Fin 6),
  (¬has_path (strategy g move)) :=
sorry

end bob_wins_l587_587115


namespace max_cookie_price_l587_587351

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end max_cookie_price_l587_587351


namespace greatest_gcd_of_6Tn_and_n_minus_1_l587_587224

theorem greatest_gcd_of_6Tn_and_n_minus_1 (n : ℕ) (hn : n > 0) :
  let T_n := (n * (n + 1)) / 2 in
  let g := Nat.gcd (3 * n ^ 2 + 3 * n) (n - 1) in
  g ≤ 3 :=
by
  sorry

end greatest_gcd_of_6Tn_and_n_minus_1_l587_587224


namespace alice_min_speed_l587_587774

theorem alice_min_speed
  (distance : ℝ)
  (bob_speed : ℝ)
  (alice_delay : ℝ)
  (bob_time : ℝ)
  (alice_time : ℝ)
  (alice_speed : ℝ) :
  distance = 30 ∧ 
  bob_speed = 40 ∧ 
  alice_delay = 0.5 ∧ 
  bob_time = distance / bob_speed ∧
  alice_time = bob_time - alice_delay ∧
  alice_speed = distance / alice_time
  → alice_speed > 60 :=
begin
  intros,
  sorry
end

end alice_min_speed_l587_587774


namespace distinct_pairs_count_l587_587984

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end distinct_pairs_count_l587_587984


namespace average_of_roots_l587_587871

-- Conditions: Definition of the quadratic equation with real solutions
def quadratic_eq (a b x : ℝ) : Prop :=
  a ≠ 0 ∧ (x^2 - (4*x) + (b/a)) = 0

-- Problem Statement: Prove the average of the roots is 2
theorem average_of_roots (a b : ℝ) (h : quadratic_eq a b) : 
  (∀ x₁ x₂ : ℝ, (x₁ + x₂ = 4) → ((x₁ + x₂) / 2) = 2) := 
by
  sorry

end average_of_roots_l587_587871


namespace basketball_games_won_difference_l587_587127

theorem basketball_games_won_difference :
  ∀ (total_games games_won games_lost difference_won_lost : ℕ),
  total_games = 62 →
  games_won = 45 →
  games_lost = 17 →
  difference_won_lost = games_won - games_lost →
  difference_won_lost = 28 :=
by
  intros total_games games_won games_lost difference_won_lost
  intros h_total h_won h_lost h_diff
  rw [h_won, h_lost] at h_diff
  exact h_diff

end basketball_games_won_difference_l587_587127


namespace number_of_cows_l587_587444

def land_cost : ℕ := 30 * 20
def house_cost : ℕ := 120000
def chicken_cost : ℕ := 100 * 5
def installation_cost : ℕ := 6 * 100
def equipment_cost : ℕ := 6000
def total_cost : ℕ := 147700

theorem number_of_cows : 
  (total_cost - (land_cost + house_cost + chicken_cost + installation_cost + equipment_cost)) / 1000 = 20 := by
  sorry

end number_of_cows_l587_587444


namespace shaded_area_correct_l587_587943

def square_area := 40 * 40

def triangle1_area := (1/2) * (40 - 15) * 30

def triangle2_area := (1/2) * 30 * 20

def unshaded_area := triangle1_area + triangle2_area

def shaded_area := square_area - unshaded_area

theorem shaded_area_correct :
  shaded_area = 925 :=
by
  -- Definitions
  let square_area := (40 : ℝ) * 40
  let triangle1_area := (1 / 2) * (40 - 15) * 30
  let triangle2_area := (1 / 2) * 30 * 20
  let unshaded_area := triangle1_area + triangle2_area
  let shaded_area := square_area - unshaded_area
  
  -- Proof to show shaded_area = 925
  calc
  shaded_area = 1600 - 675 : by sorry
           ... = 925 : by sorry

end shaded_area_correct_l587_587943


namespace Dylan_needs_two_trays_l587_587200

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end Dylan_needs_two_trays_l587_587200


namespace speed_of_man_l587_587851

theorem speed_of_man :
  let L := 500 -- Length of the train in meters
  let t := 29.997600191984642 -- Time in seconds
  let V_train_kmh := 63 -- Speed of train in km/hr
  let V_train := (63 * 1000) / 3600 -- Speed of train converted to m/s
  let V_relative := L / t -- Relative speed of train w.r.t man
  
  V_train - V_relative = 0.833 := by
  sorry

end speed_of_man_l587_587851


namespace chris_initial_donuts_l587_587543

theorem chris_initial_donuts (D : ℝ) (H1 : D * 0.90 - 4 = 23) : D = 30 := 
by
sorry

end chris_initial_donuts_l587_587543


namespace triangle_inequality_satisfied_l587_587717

open Real

theorem triangle_inequality_satisfied
  (n : ℕ) (h1 : n ≥ 3)
  (t : Fin n → ℝ) (h2 : ∀ i, t i > 0)
  (h3 : n^2 + 1 > (Finset.univ.sum t) * (Finset.univ.sum (λ i, 1 / t i)))
  :
  ∀ i j k : Fin n, i < j → j < k → (t i + t j > t k ∧ t i + t k > t j ∧ t j + t k > t i) :=
by
  sorry

end triangle_inequality_satisfied_l587_587717


namespace average_rate_of_trip_l587_587736

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l587_587736


namespace ellipse_focus_area_l587_587257

theorem ellipse_focus_area {a b : ℝ} (h : a > b) (hb : b > 0) (P : ℝ × ℝ)
  (hP_on_ellipse : P.1^2 / a^2 + P.2^2 / b^2 = 1) (angle_P : ∠ (F1 P F2) = 60)
  (area_P : area (triangle F1 P F2) = 3 * Real.sqrt 3) : b = 3 := by
sorry

end ellipse_focus_area_l587_587257


namespace flagpole_problem_proof_l587_587067

open Nat

theorem flagpole_problem_proof :
  ∃ N : ℕ,
    let blue_flags := 10 in
    let green_flags := 9 in
    let flagpoles := 2 in
    let total_flags := blue_flags + green_flags in
    -- N is the number of distinguishable arrangements satisfying conditions
    -- Condition: No two green flags on either pole are adjacent
    -- Condition: Each pole must have at least one flag
    -- Using all flags: total_flags = 19
    (N % 1000 = 310) ∧
    (N = 2310) := by
  sorry

end flagpole_problem_proof_l587_587067


namespace expected_value_is_58_l587_587848

-- Definition of the fairness and conditions of the 8-sided die
def outcomes := {1, 2, 3, 4, 5, 6, 7, 8}
def is_even (n : ℕ) : Prop := n % 2 = 0
def even_winnings (n : ℕ) : ℕ := n ^ 2
def odd_loss : ℤ := -1

-- Definition of the probabilities
def prob (n : ℕ) : ℚ := 1 / 8

-- Calculation of the expected value
def expected_value := 
  let evens := [2, 4, 6, 8]
  let odds := [1, 3, 5, 7]
  ((evens.map even_winnings).sum * (1 / 2) : ℤ) + ((odds.length : ℕ) * odd_loss * (1 / 2))

-- Theorem statement
theorem expected_value_is_58 : expected_value = 58 := by
  -- Placeholder for the proof
  sorry

end expected_value_is_58_l587_587848


namespace last_letter_of_86th_permutation_is_E_l587_587187

def permutation_86_last_letter_correct : String :=
  let letters := ['A', 'E', 'H', 'M', 'S']
  let perms := List.permutations letters
  let sorted_perms := perms.sort
  sorted_perms.get! 85 last == 'E'

theorem last_letter_of_86th_permutation_is_E
  (letters : List Char)
  (h_letters : letters = ['A', 'H', 'S', 'M', 'E']) :
  (List.permutations letters).sort.get! 85 |>.last = 'E' :=
  by
  -- Proof should follow here but it is omitted
  sorry

end last_letter_of_86th_permutation_is_E_l587_587187


namespace percent_reduction_l587_587057

def original_price : ℕ := 500
def reduction_amount : ℕ := 400

theorem percent_reduction : (reduction_amount * 100) / original_price = 80 := by
  sorry

end percent_reduction_l587_587057


namespace sequence_formula_l587_587681

noncomputable def a : ℕ → ℚ
| 0      := 0    -- Convention to handle 0 case naturally, though we start from 1
| 1      := 1
| (n+2) := a (n+1) / (1 + a (n+1))

theorem sequence_formula (n : ℕ) (hn : n > 0) : a n = 1 / n :=
by
  induction n with d hd
  · exact absurd hn (Nat.not_lt_zero 0) -- Handles the contradiction for n = 0
  · cases d
    · simp [a] -- Handles base case n = 1
    · suffices a (d.succ.succ) = 1 / (d.succ.succ) by exact this -- Inductive hypothesis
      simp [a, hd]
      field_simp
      sorry -- Placeholder for manual proof steps

end sequence_formula_l587_587681


namespace infinite_solution_pairs_l587_587191

theorem infinite_solution_pairs (a b : ℝ) (h : a ≠ 0) (h : b ≠ 0) : 
  ∃ (S : set (ℝ × ℝ)), set.countable S = false ∧ ∀ (p ∈ S), (a+b)^2 = 2*a*b + 1 :=
by 
  sorry

end infinite_solution_pairs_l587_587191


namespace lyssa_fewer_correct_l587_587727

-- Define the total number of items in the exam
def total_items : ℕ := 75

-- Define the number of mistakes made by Lyssa
def lyssa_mistakes : ℕ := total_items * 20 / 100  -- 20% of 75

-- Define the number of correct answers by Lyssa
def lyssa_correct : ℕ := total_items - lyssa_mistakes

-- Define the number of mistakes made by Precious
def precious_mistakes : ℕ := 12

-- Define the number of correct answers by Precious
def precious_correct : ℕ := total_items - precious_mistakes

-- Statement to prove Lyssa got 3 fewer correct answers than Precious
theorem lyssa_fewer_correct : (precious_correct - lyssa_correct) = 3 := by
  sorry

end lyssa_fewer_correct_l587_587727


namespace ordered_quadruples_fraction_l587_587368

-- Define the problem as the number of ordered quadruples (x1, x2, x3, x4) of positive odd integers such that their sum is 98.
noncomputable def countOrderedQuadruples : ℕ :=
  sorry -- This would be where the calculation of the number happens.

-- Main theorem to state the problem and the expected result.
theorem ordered_quadruples_fraction :
  ∑ (x : Fin 4 → ℕ) in {x : (Fin 4 → ℕ) | ∀ i, x i % 2 = 1 ∧ x i > 0 ∧ ∑ i, x i = 98}.card / 100 = 196 :=
sorry

end ordered_quadruples_fraction_l587_587368


namespace january_31_is_friday_l587_587557

theorem january_31_is_friday (h : ∀ (d : ℕ), (d % 7 = 0 → d = 1)) : ∀ d, (d = 31) → (d % 7 = 3) :=
by
  sorry

end january_31_is_friday_l587_587557


namespace domain_of_transformed_function_l587_587775

theorem domain_of_transformed_function (f : ℝ → ℝ) (h : ∀ x, 0 ≤ x ∧ x ≤ 2 → True) :
  ∀ x, -1 ≤ x ∧ x ≤ 1 → True :=
sorry

end domain_of_transformed_function_l587_587775


namespace enumerate_set_A_l587_587291

open Set Nat

theorem enumerate_set_A :
  {x ∈ ℕ | ∃ k ∈ ℤ, 4 = k * (x - 3)} = {1, 2, 4, 5, 7} :=
by
  sorry

end enumerate_set_A_l587_587291


namespace rent_percentage_this_year_l587_587355

variable (E : ℝ) -- Earnings last year

-- Conditions
def last_year_rent_fraction : ℝ := 0.20
def earnings_growth_rate : ℝ := 1.35
def this_year_rent_multiple : ℝ := 2.025

-- Define expressions for rent last year and earnings this year
def rent_last_year (E : ℝ) : ℝ := last_year_rent_fraction * E
def earnings_this_year (E : ℝ) : ℝ := earnings_growth_rate * E
def rent_this_year (E : ℝ) : ℝ := this_year_rent_multiple * rent_last_year E

theorem rent_percentage_this_year
  (h1 : E > 0)
  : let P_percent := (100 * (rent_this_year E) / (earnings_this_year E)) in
  P_percent = 30 := sorry

end rent_percentage_this_year_l587_587355


namespace points_per_win_is_5_l587_587039

-- Definitions based on conditions
def rounds_played : ℕ := 30
def vlad_points : ℕ := 64
def taro_points (T : ℕ) : ℕ := (3 * T) / 5 - 4
def total_points (T : ℕ) : ℕ := taro_points T + vlad_points

-- Theorem statement to prove the number of points per win
theorem points_per_win_is_5 (T : ℕ) (H : total_points T = T) : T / rounds_played = 5 := sorry

end points_per_win_is_5_l587_587039


namespace volume_of_tetrahedron_l587_587929

theorem volume_of_tetrahedron (A B C D : Type) 
    (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle : ℝ)
    (h1 : area_ABC = 150)
    (h2 : area_BCD = 90)
    (h3 : BC = 10)
    (h4 : angle = π / 4) :
    ∃ V : ℝ, V = 450 * real.sqrt 2 :=
begin
  sorry
end

end volume_of_tetrahedron_l587_587929


namespace betty_initial_marbles_l587_587162

theorem betty_initial_marbles (B : ℝ) (h1 : 0.40 * B = 24) : B = 60 :=
by
  sorry

end betty_initial_marbles_l587_587162


namespace apple_street_length_l587_587168

theorem apple_street_length :
  ∀ (n : ℕ) (d : ℕ), 
    (n = 15) → (d = 200) → 
    (∃ l : ℝ, (l = ((n + 1) * d) / 1000) ∧ l = 3.2) :=
by
  intros
  sorry

end apple_street_length_l587_587168


namespace maximum_x_minus_y_l587_587708

theorem maximum_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end maximum_x_minus_y_l587_587708


namespace oranges_count_l587_587383

noncomputable def initial_oranges (O : ℕ) : Prop :=
  let apples := 14
  let blueberries := 6
  let remaining_fruits := 26
  13 + (O - 1) + 5 = remaining_fruits

theorem oranges_count (O : ℕ) (h : initial_oranges O) : O = 9 :=
by
  have eq : 13 + (O - 1) + 5 = 26 := h
  -- Simplify the equation to find O
  sorry

end oranges_count_l587_587383


namespace total_amount_received_correct_l587_587350

variable (total_won : ℝ) (fraction : ℝ) (students : ℕ)
variable (portion_per_student : ℝ := total_won * fraction)
variable (total_given : ℝ := portion_per_student * students)

theorem total_amount_received_correct :
  total_won = 555850 →
  fraction = 3 / 10000 →
  students = 500 →
  total_given = 833775 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end total_amount_received_correct_l587_587350


namespace simplify_expression_l587_587647

variable (x y z : ℝ)

theorem simplify_expression (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (hne : y - z / x ≠ 0) : 
  (x - z / y) / (y - z / x) = x / y := 
by 
  sorry

end simplify_expression_l587_587647


namespace unique_positive_integers_pqr_l587_587718

noncomputable def y : ℝ := Real.sqrt ((Real.sqrt 61) / 2 + 5 / 2)

lemma problem_condition (p q r : ℕ) (py : ℝ) :
  py = y^100
  ∧ py = 2 * (y^98)
  ∧ py = 16 * (y^96)
  ∧ py = 13 * (y^94)
  ∧ py = - y^50
  ∧ py = ↑p * y^46
  ∧ py = ↑q * y^44
  ∧ py = ↑r * y^40 :=
sorry

theorem unique_positive_integers_pqr : 
  ∃! (p q r : ℕ), 
    p = 37 ∧ q = 47 ∧ r = 298 ∧ 
    y^100 = 2 * y^98 + 16 * y^96 + 13 * y^94 - y^50 + ↑p * y^46 + ↑q * y^44 + ↑r * y^40 :=
sorry

end unique_positive_integers_pqr_l587_587718


namespace first_player_guarantee_win_l587_587798

theorem first_player_guarantee_win (k : ℕ) (h_k : k = 7 ∨ k = 10) :
  ∃ (C : fin (200 * 199 / 2) → fin k), ∀ (P : fin 200 → fin k), 
  ∃ (a b : fin 200), a < b ∧ P a = P b ∧ C (pair a b) = P a :=
by 
  sorry

end first_player_guarantee_win_l587_587798


namespace max_angels_l587_587434

theorem max_angels {n : ℕ} (h₁ : ∀ w, 1 ≤ w ∧ w ≤ 2015 → (w ∈ {a // is_angel a} ∨ w ∈ {d // is_demon d}))
  (h₂ : ∀ a, a ∈ {a // is_angel a} → tells_truth a)
  (h₃ : ∀ d, d ∈ {d // is_demon d} → (tells_truth d ∨ tells_lies d))
  (h₄ : tells_truth woman_1 → count_angel = 1)
  (h₅ : tells_truth woman_3 → count_angel = 3)
  (h₆ : tells_truth woman_2013 → count_angel = 2013)
  (h₇ : tells_truth woman_2 → count_demon = 2)
  (h₈ : tells_truth woman_4 → count_demon = 4)
  (h₉ : tells_truth woman_2014 → count_demon = 2014) :
    count_angel ≤ 3 :=
  sorry

end max_angels_l587_587434


namespace count_three_digit_sum_24_l587_587233

theorem count_three_digit_sum_24 : 
  let count := (λ n : ℕ, let a := n / 100, b := (n / 10) % 10, c := n % 10 in
                        a + b + c = 24 ∧ 100 ≤ n ∧ n < 1000) in
  (finset.range 1000).filter count = 8 :=
sorry

end count_three_digit_sum_24_l587_587233


namespace tan_B_equals_4_tan_A_find_a_value_l587_587334

section TriangleProblem

variables {A B C : ℝ} (a b c : ℝ)
hypotheses
  (h_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_equation : b * sin C * cos A = 4 * c * sin A * cos B)
  (h_tan : tan (A + B) = -3)
  (h_c : c = 3)
  (h_b : b = 5)

theorem tan_B_equals_4_tan_A : tan B = 4 * tan A :=
sorry

theorem find_a_value : a = sqrt 10 :=
sorry

end TriangleProblem

end tan_B_equals_4_tan_A_find_a_value_l587_587334


namespace tan_theta_and_expression_l587_587981

theorem tan_theta_and_expression (θ : ℝ) (h : ∃ (k : ℝ), (4, -3) = (4 * cos k, 4 * sin k)) :
  (tan θ = -3 / 4) ∧ 
  (sin (θ + real.pi / 2) + cos θ) / (sin θ - cos (θ - real.pi)) = 8 :=
by
  sorry

end tan_theta_and_expression_l587_587981


namespace find_a2_l587_587243

open Classical

variable {a_n : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) :=
  ∀ n m : ℕ, a (n + m) = a n * q ^ m

theorem find_a2 (h1 : geometric_sequence a_n q)
                (h2 : a_n 7 = 1 / 4)
                (h3 : a_n 3 * a_n 5 = 4 * (a_n 4 - 1)) :
  a_n 2 = 8 :=
sorry

end find_a2_l587_587243


namespace second_car_speed_l587_587445

-- Definition of the problem parameters
def highway_length := 500 -- highway length in miles
def speed_car1 := 40      -- speed of the first car in mph
def meeting_time := 5     -- time when the cars meet in hours

-- Statement of the problem
theorem second_car_speed :
  ∀ (v : ℝ), 
  (highway_length - speed_car1 * meeting_time) / meeting_time = v → v = 60 := 
by
  intro v
  assume h
  calc
    v = (highway_length - speed_car1 * meeting_time) / meeting_time : by {assumption}
     ... = 60 : by {ring}

end second_car_speed_l587_587445


namespace bricks_needed_l587_587486

def brick_length := 25 -- cm
def brick_width := 11.25 -- cm
def brick_height := 6 -- cm

def wall_length := 900 -- cm
def wall_height := 600 -- cm
def wall_thickness := 22.5 -- cm

def brick_volume := brick_length * brick_width * brick_height
def wall_volume := wall_length * wall_height * wall_thickness

theorem bricks_needed : wall_volume / brick_volume = 7200 :=
by
  sorry

end bricks_needed_l587_587486


namespace angle_baf_eq_angle_dae_l587_587675

variables {A B C D E G F : Type}
variables [IsConvexQuad A B C D] [BE_extends_to_E B E G AC]
          [DG_extends_to_F D G CB]

-- Define a property for convex quadrilateral
class IsConvexQuad (A B C D : Type) :=
  (convex_quad: A B C D)
  (dac_bisect_bad: B C ∠_ A D)

-- Define the extension property of point E on CD
class BE_extends_to_E (B E G AC : Type) :=
  (extend_on_CD: E ∈ ray (C, D))
  (intersect_AC: G ∈ (B ↔ E) ∩ AC)

-- Define the extension property of DG to intersect CB at F
class DG_extends_to_F (D G CB : Type) :=
  (extend_on_CB: F ∈ (ext_end D ↔ G))

-- The verified theorem
theorem angle_baf_eq_angle_dae :
  ∀ {A B C D E G F : Type} [IsConvexQuad A B C D] [BE_extends_to_E B E G AC]
    [DG_extends_to_F D G CB],
  ∠ B A F = ∠ D A E :=
by sorry

end angle_baf_eq_angle_dae_l587_587675


namespace certain_event_l587_587165

/-- Among the following events, the one that is a certain event is --/
def event_A : Prop := ∀ day : Daylight, weather : Weather, cloudy day weather → raining day weather
def event_B : Prop := ∃ coin : Coin, fair coin → head_up coin
def event_C : Prop := ∀ person : Person, gender person = male → height person ≥ height (any_girl person)
def event_D : Prop := ∀ liquid_1 liquid_2 : Liquid, (liquid_1 = oil ∧ liquid_2 = water) → floats_on liquid_1 liquid_2

theorem certain_event :
  event_A ∨ event_B ∨ event_C ∨ event_D → event_D :=
by
  sorry

end certain_event_l587_587165


namespace distinct_ordered_pairs_l587_587985

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end distinct_ordered_pairs_l587_587985


namespace florist_sold_roses_l587_587862

theorem florist_sold_roses (x : ℕ) (h1 : 5 - x + 34 = 36) : x = 3 :=
by sorry

end florist_sold_roses_l587_587862


namespace Dave_has_more_money_than_Derek_l587_587908

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l587_587908


namespace find_temp_M_l587_587041

section TemperatureProof

variables (M T W Th F : ℕ)

-- Conditions
def avg_temp_MTWT := (M + T + W + Th) / 4 = 48
def avg_temp_TWThF := (T + W + Th + F) / 4 = 40
def temp_F := F = 10

-- Proof
theorem find_temp_M (h1 : avg_temp_MTWT M T W Th)
                    (h2 : avg_temp_TWThF T W Th F)
                    (h3 : temp_F F)
                    : M = 42 :=
sorry

end TemperatureProof

end find_temp_M_l587_587041


namespace distance_range_proof_return_fare_proof_l587_587319

variables (x : ℝ) -- distance from Sanfan Middle School to the examination center
variables (fare_start : ℝ := 13) (fare_short_trip_all : ℝ := 14) -- base fares
variables (additional_rate : ℝ := 2.30) (surcharge_pct : ℝ := 0.5)
          (advanced_reservation_fee : ℝ := 6) (short_notice_reservation_fee : ℝ := 5)

-- Distance walked: 0.6 km, resulting in fare 43 yuan reduced to 37 yuan when deducted advanced reservation fee
-- Distance walked: 0.6 km, resulting in fare 42 yuan reduced to 37 yuan when deducted short notice reservation fee

axiom estimated_fare_first : 43 - advanced_reservation_fee = 37
axiom estimated_fare_second : 42 - short_notice_reservation_fee = 37

def fare_calculation (distance : ℝ) : ℝ :=
  if distance <= 3 then 
    fare_short_trip_all 
  else if distance <= 15 then 
    fare_short_trip_all + additional_rate * (distance - 3)
  else
    let base := fare_short_trip_all + additional_rate * (15 - 3)
    in base + (additional_rate * surcharge_pct * (distance - 15))

-- Prove the following:
theorem distance_range_proof (h1 : 37 ≤ estimated_fare_first) (h2 : 37 ≤ estimated_fare_second)
      (h3 : fare_short_trip_all + additional_rate * (x - 3) = estimated_fare_first)
      (h4 : fare_short_trip_all + additional_rate * (x - 3) = estimated_fare_second) : 
      12.6 < x ∧ x ≤ 13 :=
by {
  sorry
}

theorem return_fare_proof (distance_to_exam_center : ℝ) (x : ℝ) (h1 : 12.6 < x) (h2 : x ≤ 13) : 
      fare_calculation ((2 * x) - 4) = 66 :=
by {
  sorry
}

end distance_range_proof_return_fare_proof_l587_587319


namespace sum_of_remaining_squares_l587_587661

variable {a b c d e x S : ℤ}

def isMagicSquare (a b c d e x : ℤ) : Prop :=
  let S := 4 + 7 + b
  (4 + 7 + b = S) ∧
  (4 + x + a = S) ∧
  (e + x + b = S) ∧
  (7 + x + d = S) ∧
  (c + x + 2018 = S) ∧
  (a + 2018 + b = S)

theorem sum_of_remaining_squares (a b c d e x S : ℤ)
  (h : isMagicSquare a b c d e x) :
  (4 + 7 + c + d + e + x + a + b + 2018 - (4 + 7 + 2018)) = -11042.5 :=
by
  sorry

end sum_of_remaining_squares_l587_587661


namespace f_sum_zero_l587_587553

-- Define the function and conditions
def f (x : ℝ) : ℝ := sorry

axiom f_neg : ∀ (x : ℝ), f (-x) = -f (x + 4)
axiom x1_x2_sum : ∀ (x1 x2 : ℝ), x1 + x2 = 4 → f(x1) + f(x2) = 0

-- The theorem stating the problem
theorem f_sum_zero (x1 x2 : ℝ) (h : x1 + x2 = 4) : f(x1) + f(x2) = 0 :=
by
  exact x1_x2_sum x1 x2 h

end f_sum_zero_l587_587553


namespace circumference_tank_A_is_11_l587_587406

-- Definitions of the given conditions
def heightA := 10
def heightB := 11
def circumferenceB := 10
def volumeRelation := 1.1000000000000001 -- 110.00000000000001 in decimal representation

-- Question: What is the circumference of tank A?
theorem circumference_tank_A_is_11 :
    let rA := circumferenceB / (2 * Real.pi)
    let rB := 10 / (2 * Real.pi)
    let volumeA := Real.pi * (rA^2) * heightA
    let volumeB := Real.pi * (rB^2) * heightB
    volumeA = volumeRelation * volumeB →
    (∃ C_A, ∃ CA : Real, CA = 11) :=
by
  sorry

end circumference_tank_A_is_11_l587_587406


namespace diameter_of_Gamma_l587_587807

theorem diameter_of_Gamma:
  ∀ (A B C O : Type)
    (d_AB d_BC d_AC : ℝ)
    (circumcenter : A → B → C → O)
    (tangent_circle_surrounds : O → Prop),
  (d_AB = 4) →
  (d_BC = 6) →
  (d_AC = 5) →
  (tangent_circle_surrounds circumcenter) →
  ∃ diam : ℝ, diam = (256 * Real.sqrt 7) / 17 :=
by
  sorry

end diameter_of_Gamma_l587_587807


namespace S_odd_zero_S_even_product_l587_587577

variable (x : ℝ) (g : ℕ → ℕ → ℝ → ℝ)

def S (l : ℕ) : ℝ :=
  ∑ i in finset.range (l + 1), (-1)^i * g i (l - i) x

theorem S_odd_zero (n : ℕ) : S x g (2 * n + 1) = 0 := 
sorry

theorem S_even_product (n : ℕ) : S x g (2 * n) = ∏ i in finset.range n, (1 - x^(2*i + 1)) := 
sorry

end S_odd_zero_S_even_product_l587_587577


namespace percentage_increase_after_lawnmower_l587_587896

-- Definitions from conditions
def initial_daily_yards := 8
def weekly_yards_after_lawnmower := 84
def days_in_week := 7

-- Problem statement
theorem percentage_increase_after_lawnmower : 
  ((weekly_yards_after_lawnmower / days_in_week - initial_daily_yards) / initial_daily_yards) * 100 = 50 := 
by 
  sorry

end percentage_increase_after_lawnmower_l587_587896


namespace max_xyz_eq_one_l587_587712

noncomputable def max_xyz (x y z : ℝ) : ℝ :=
  if h_cond : 0 < x ∧ 0 < y ∧ 0 < z ∧ (x * y + z ^ 2 = (x + z) * (y + z)) ∧ (x + y + z = 3) then
    x * y * z
  else
    0

theorem max_xyz_eq_one : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → 
  (x * y + z ^ 2 = (x + z) * (y + z)) → (x + y + z = 3) → max_xyz x y z ≤ 1 :=
by
  intros x y z hx hy hz h1 h2
  -- Proof is omitted here
  sorry

end max_xyz_eq_one_l587_587712


namespace find_t_l587_587567

theorem find_t :
  ∃ t : ℕ, 10 ≤ t ∧ t < 100 ∧ 13 * t % 100 = 52 ∧ t = 44 :=
by
  sorry

end find_t_l587_587567


namespace volume_of_tetrahedron_equals_450_sqrt_2_l587_587924

-- Given conditions
variables {A B C D : Point}
variables (areaABC areaBCD : ℝ) (BC : ℝ) (angleABC_BCD : ℝ)

-- The specific values for the conditions
axiom h_areaABC : areaABC = 150
axiom h_areaBCD : areaBCD = 90
axiom h_BC : BC = 10
axiom h_angleABC_BCD : angleABC_BCD = π / 4  -- 45 degrees in radians

-- Definition of the volume to be proven
def volume_tetrahedron (A B C D : Point) : ℝ :=
  (1 / 3) * areaABC * (18 * real.sin angleABC_BCD)

-- Final proof statement
theorem volume_of_tetrahedron_equals_450_sqrt_2 :
  volume_tetrahedron A B C D = 450 * real.sqrt 2 :=
by 
  -- Preliminary setup, add the relevant properties and results
  sorry

end volume_of_tetrahedron_equals_450_sqrt_2_l587_587924


namespace solve_for_x_l587_587839

theorem solve_for_x (x : ℝ) (h : 4 / (1 + 3 / x) = 1) : x = 1 :=
sorry

end solve_for_x_l587_587839


namespace find_number_l587_587508

def incorrect_multiplication (x : ℕ) : ℕ := 394 * x
def correct_multiplication (x : ℕ) : ℕ := 493 * x
def difference (x : ℕ) : ℕ := correct_multiplication x - incorrect_multiplication x
def expected_difference : ℕ := 78426

theorem find_number (x : ℕ) (h : difference x = expected_difference) : x = 792 := by
  sorry

end find_number_l587_587508


namespace complex_number_in_second_quadrant_l587_587340

def complex_quadrant (z : ℂ) : ℕ :=
if z.re > 0 then
  if z.im > 0 then 1 else 4
else
  if z.im > 0 then 2 else 3

theorem complex_number_in_second_quadrant (z : ℂ) (hz : z = -1 + 2 * complex.I) : complex_quadrant z = 2 :=
by
  rw [hz]
  simp [complex_quadrant, complex.re, complex.im]
  sorry

end complex_number_in_second_quadrant_l587_587340


namespace find_exponent_l587_587650

theorem find_exponent (x : ℝ) (h1 : 7^x = 2) (h2 : 7^(4 * x + 2) = 784) : x = 1 / 4 :=
by
  sorry

end find_exponent_l587_587650


namespace sum_primitive_roots_mod_11_l587_587473

def is_primitive_root (a n : ℕ) : Prop :=
  ∀ b < n, ∃ k < n, a^k % n = b

theorem sum_primitive_roots_mod_11 :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  s.sum (λ x, if is_primitive_root x 11 then x else 0) = 23 :=
by
  sorry

end sum_primitive_roots_mod_11_l587_587473


namespace exists_satisfying_m_l587_587849

noncomputable def check_lines (m : ℝ) : Prop :=
  let y1 := 1
  let y2 := 0
  let p1 := (0, 1)
  let p2 := (-1, 0)
  let parallel_line1 := fun (x : ℝ) => m * x + y1
  let parallel_line2 := fun (x : ℝ) => m * (x + 1)
  let perp_line1 := fun (x : ℝ) => (-1 / m) * (x - 1)
  let perp_line2 := fun (x : ℝ) => (-1 / m) * x
  let inter1 := (1 - m) / (m^2 + 1)
  let inter1_y := (m + 1) / (m^2 + 1)
  let inter2 := (1 - m^2) / (m^2 + 1)
  let inter2_y := (2 * m) / (m^2 + 1)
  inter1 ≠ ∞ ∧ inter2 ≠ ∞ ∧ (m = 0 ∨ m = 2)

theorem exists_satisfying_m : ∃ m : ℝ, check_lines m :=
begin
  sorry
end

end exists_satisfying_m_l587_587849


namespace mass_percentage_C_l587_587915

theorem mass_percentage_C (
  CaCO3_molar_mass : ℝ,
  CO_molar_mass : ℝ,
  CaCO3_mass_frac : ℝ,
  CO_mass_frac : ℝ,
  sample_mass : ℝ,
  C_mass_in_CaCO3 : ℝ,
  C_mass_in_CO : ℝ
) : 
  CaCO3_molar_mass = 100.09 ∧
  CO_molar_mass = 28.01 ∧
  CaCO3_mass_frac = 0.8 ∧
  CO_mass_frac = 0.2 ∧
  sample_mass = 100 ∧
  C_mass_in_CaCO3 = 0.12 ∧
  C_mass_in_CO = 0.4288 →
  ((C_mass_in_CaCO3 * (CaCO3_mass_frac * sample_mass) + C_mass_in_CO * (CO_mass_frac * sample_mass)) / sample_mass) * 100 = 18.176 :=
by
  intros h
  sorry

end mass_percentage_C_l587_587915


namespace john_must_deliver_1063_pizzas_l587_587349

-- Declare all the given conditions
def car_cost : ℕ := 8000
def maintenance_cost : ℕ := 500
def pizza_income (p : ℕ) : ℕ := 12 * p
def gas_cost (p : ℕ) : ℕ := 4 * p

-- Define the function that returns the net earnings
def net_earnings (p : ℕ) := pizza_income p - gas_cost p

-- Define the total expenses
def total_expenses : ℕ := car_cost + maintenance_cost

-- Define the minimum number of pizzas John must deliver
def minimum_pizzas (p : ℕ) : Prop := net_earnings p ≥ total_expenses

-- State the theorem that needs to be proved
theorem john_must_deliver_1063_pizzas : minimum_pizzas 1063 := by
  sorry

end john_must_deliver_1063_pizzas_l587_587349


namespace min_value_of_expression_l587_587214

theorem min_value_of_expression : ∃ x : ℝ, (∀ x : ℝ, sqrt (x^2 + (1 + x)^2) + sqrt ((1 + x)^2 + (1 - x)^2) ≥ sqrt 5) ∧ sqrt (x^2 + (1 + x)^2) + sqrt ((1 + x)^2 + (1 - x)^2) = sqrt 5 :=
by
  -- Placeholder proof
  sorry

end min_value_of_expression_l587_587214


namespace planting_methods_count_l587_587008

-- Define the types of crops and plots as finite sets
def Crop : Type := Fin 3  -- 3 types of crops
def Plot : Type := Fin 5  -- 5 plots

-- Define the condition that adjacent plots cannot plant the same crop
def valid_planting (planting: Plot → Crop) : Prop :=
  ∀ (i : Plot), i.val < 4 → planting ⟨i.val⟩ ≠ planting ⟨i.val + 1⟩

-- Define the problem: number of valid plantings
def planting_methods : Nat :=
  (Finset.univ.filter valid_planting).card

-- The theorem: There are 42 different valid planting methods
theorem planting_methods_count : planting_methods = 42 := by
  sorry

end planting_methods_count_l587_587008


namespace max_perimeter_right_triangle_l587_587682

theorem max_perimeter_right_triangle (a b : ℝ) (h₁ : a^2 + b^2 = 25) :
  (a + b + 5) ≤ 5 + 5 * Real.sqrt 2 :=
by
  sorry

end max_perimeter_right_triangle_l587_587682


namespace sum_of_c_eq_3_power_2023_minus_2_l587_587271

noncomputable def geometric_seq (a1 q : ℕ) (n : ℕ) :=
  a1 * q^(n - 1)

noncomputable def arithmetic_seq (b1 d : ℕ) (n : ℕ) :=
  b1 + (n - 1) * d

theorem sum_of_c_eq_3_power_2023_minus_2
  (a1 q d : ℕ)
  (h1 : a1 = 1)
  (h2 : q = 3)
  (h3 : d = 2)
  (h4 : ∀ n, (∑ k in finset.range n, (c k) / (geometric_seq a1 q (k + 1)) = arithmetic_seq b1 d n))
  :
  (∑ k in finset.range 2023, c k) = 3^2023 - 2 :=
sorry

end sum_of_c_eq_3_power_2023_minus_2_l587_587271


namespace minnie_more_than_week_l587_587906

-- Define the variables and conditions
variable (M : ℕ) -- number of horses Minnie mounts per day
variable (mickey_daily : ℕ) -- number of horses Mickey mounts per day

axiom mickey_daily_formula : mickey_daily = 2 * M - 6
axiom mickey_total_per_week : mickey_daily * 7 = 98
axiom days_in_week : 7 = 7

-- Theorem: Minnie mounts 3 more horses per day than there are days in a week
theorem minnie_more_than_week (M : ℕ) 
  (h1 : mickey_daily = 2 * M - 6)
  (h2 : mickey_daily * 7 = 98)
  (h3 : 7 = 7) :
  M - 7 = 3 := 
sorry

end minnie_more_than_week_l587_587906


namespace ratio_CP_PE_l587_587660

-- Definitions based on conditions
variables (A B C D E P : Type)

def triangle_conditions (C D B A E P: Type) :=
  (∃ CD DB : ℚ, CD = 2 ∧ DB = 1) ∧
  (∃ AE EB : ℚ, AE = 4 ∧ EB = 1) ∧
  (∃ CP PE : ℚ, P = CE ∩ AD)

theorem ratio_CP_PE (C D B A E P: Type) (h: triangle_conditions C D B A E P) : 
  let r := (CP / PE) in 
  r = 4 / 3 :=
sorry

end ratio_CP_PE_l587_587660


namespace perimeter_after_adding_tiles_l587_587385

-- Definition of the initial configuration
def initial_perimeter := 16

-- Definition of the number of additional tiles
def additional_tiles := 3

-- Statement of the problem: to prove that the new perimeter is 22
theorem perimeter_after_adding_tiles : initial_perimeter + 2 * additional_tiles = 22 := 
by 
  -- The number initially added each side exposed would increase the perimeter incremented by 6
  -- You can also assume the boundary conditions for the shared sides reducing.
  sorry

end perimeter_after_adding_tiles_l587_587385


namespace gen_formula_arithmetic_seq_sum_maximizes_at_5_l587_587246

def arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a n = a 1 + (n - 1) * d

variables (an : ℕ → ℤ) (Sn : ℕ → ℤ)
variable (d : ℤ)

theorem gen_formula_arithmetic_seq (h1 : an 3 = 5) (h2 : an 10 = -9) :
  ∀ n, an n = 11 - 2 * n :=
sorry

theorem sum_maximizes_at_5 (h_seq : ∀ n, an n = 11 - 2 * n) :
  ∀ n, Sn n = (n * 10 - n^2) → (∃ n, ∀ k, Sn n ≥ Sn k) :=
sorry

end gen_formula_arithmetic_seq_sum_maximizes_at_5_l587_587246


namespace customer_C_weight_l587_587151

def weights : List ℕ := [22, 25, 28, 31, 34, 36, 38, 40, 45]

-- Definitions for customer A and B such that customer A's total weight equals twice of customer B's total weight
variable {A B : List ℕ}

-- Condition on weights distribution
def valid_distribution (A B : List ℕ) : Prop :=
  (A.sum = 2 * B.sum) ∧ (A ++ B).sum + 38 = 299

-- Prove the weight of the bag received by customer C
theorem customer_C_weight :
  ∃ (C : ℕ), C ∈ weights ∧ C = 38 := by
  sorry

end customer_C_weight_l587_587151


namespace function_domain_l587_587425

theorem function_domain (x : ℝ) : (y = sqrt x / (x - 1)) → (x ≥ 0 ∧ x ≠ 1) :=
by
  -- to be proved
  sorry

end function_domain_l587_587425


namespace range_equality_of_f_and_f_f_l587_587279

noncomputable def f (x a : ℝ) := x * Real.log x - x + 2 * a

theorem range_equality_of_f_and_f_f (a : ℝ) :
  (∀ x : ℝ, 0 < x → 1 < f x a) ∧ (∀ x : ℝ, 0 < x → f x a ≤ 1) →
  (∃ I : Set ℝ, (Set.range (λ x => f x a) = I) ∧ (Set.range (λ x => f (f x a) a) = I)) → 
  (1/2 < a ∧ a ≤ 1) :=
by 
  sorry

end range_equality_of_f_and_f_f_l587_587279


namespace additional_grassy_area_l587_587519

theorem additional_grassy_area (r1 r2 : ℝ) (r1_pos : r1 = 10) (r2_pos : r2 = 35) : 
  let A1 := π * r1^2
  let A2 := π * r2^2
  (A2 - A1) = 1125 * π :=
by 
  sorry

end additional_grassy_area_l587_587519


namespace possible_divide_square_into_non_congruent_isosceles_triangles_possible_divide_equilateral_triangle_into_non_congruent_isosceles_triangles_l587_587484

-- Definition of a square
structure Square (α : Type) := (vertices : (Fin 4) → Point α)
-- Definition of an equilateral triangle
structure EquilateralTriangle (α : Type) := (vertices : (Fin 3) → Point α)
-- Definition of an isosceles triangle
structure IsoscelesTriangle (α : Type) := 
  (vertices : (Fin 3) → Point α)
  (sides : (Fin 3) → Length α)
  (isosceles_property : (sides 0 = sides 1) ∨ (sides 1 = sides 2) ∨ (sides 2 = sides 0))

-- Proof problem that a square can be divided into 4 non-congruent isosceles triangles
theorem possible_divide_square_into_non_congruent_isosceles_triangles (α : Type) [ncs : NonemptyCharSpace α]:
  ∃ (divide : Square α → Fin 4 → IsoscelesTriangle α), 
  ((∀ (i j : Fin 4), i ≠ j → ¬ congruent (divide i) (divide j))) := 
sorry

-- Proof problem that an equilateral triangle can be divided into 4 non-congruent isosceles triangles
theorem possible_divide_equilateral_triangle_into_non_congruent_isosceles_triangles (α : Type) [ncs : NonemptyCharSpace α]:
  ∃ (divide : EquilateralTriangle α → Fin 4 → IsoscelesTriangle α), 
  ((∀ (i j : Fin 4), i ≠ j → ¬ congruent (divide i) (divide j))) := 
sorry

end possible_divide_square_into_non_congruent_isosceles_triangles_possible_divide_equilateral_triangle_into_non_congruent_isosceles_triangles_l587_587484


namespace sam_won_total_matches_l587_587324

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end sam_won_total_matches_l587_587324


namespace g_of_50_eq_zero_l587_587051

theorem g_of_50_eq_zero (g : ℝ → ℝ) (h : ∀ x y : ℝ, 0 < x → 0 < y → x * g y - 3 * y * g x = g (x / y)) : g 50 = 0 :=
sorry

end g_of_50_eq_zero_l587_587051


namespace range_of_a_minimum_value_of_g_l587_587994

noncomputable theory
open Real

-- Definitions and Conditions
def f (x : ℝ) (a : ℝ) := a * x - log x
def F (x : ℝ) (a : ℝ) := exp x + a * x
def g (x : ℝ) (a : ℝ) := x * exp (a * x - 1) - 2 * a * x + f x a

-- Conditions
variable {x : ℝ}
variable {a : ℝ}
variable (h_x_pos : 0 < x)
variable (h_a_neg : a < 0)
variable (h_monotonic_f_F : (∀ {x}, 0 < x ∧ x < log 3 → (f x a).monotonic_on (0, log 3) ↔ (F x a).monotonic_on (0, log 3)))
variable (h_a_range : a ∈ Iic (-1 / exp 2))

-- Theorem Statements
theorem range_of_a (h_monotonic_f_F : (∀ {x}, 0 < x ∧ x < log 3 → (f x a).monotonic_on (0, log 3) ↔ (F x a).monotonic_on (0, log 3))) : a ≤ -3 :=
  sorry

theorem minimum_value_of_g (h_a_range : a ∈ Iic (-1 / exp 2)) : ∃ M, M = 0 ∧ ∀ x, g x a = M :=
  sorry

end range_of_a_minimum_value_of_g_l587_587994


namespace total_distributions_l587_587533

theorem total_distributions :
  let x : Fin 7 → ℕ := λ i, 
    ∃ m n : ℕ, 
      (∀ i, i < 4 → 7 ∣ x i) ∧
      (∀ j, j ≥ 4 → j < 7 → 13 ∣ x j) ∧
      (x 0 + x 1 + x 2 + x 3 = 7 * m) ∧
      (x 4 + x 5 + x 6 = 13 * n) ∧
      (7 * m + 13 * n = 270) in
  ∑' x_i in {x | ∃ m n, 
                (∀ i, i < 4 → 7 ∣ x i) ∧
                (∀ j, j ≥ 4 → j < 7 → 13 ∣ x j) ∧
                (x 0 + x 1 + x 2 + x 3 = 7 * m) ∧
                (x 4 + x 5 + x 6 = 13 * n) ∧
                (7 * m + 13 * n = 270)},
    1 = 42244 :=
begin
  sorry
end

end total_distributions_l587_587533


namespace num_intersection_points_l587_587679

theorem num_intersection_points :
  set.countable {x : ℝ | 0 ≤ x ∧ x < 2 * Real.pi ∧
    Real.sin (x + Real.pi / 3) = 1 / 2}.finite.card = 2 := by
sorry

end num_intersection_points_l587_587679


namespace sum_of_sequence_l587_587618

noncomputable def f (x : ℝ) : ℝ := x ^ 2 - 2 * x
noncomputable def f' := (deriv f)

def a (n : ℕ) : ℝ := f' n

def S (n : ℕ) : ℝ := ∑ i in Finset.range n, a (i + 1)

theorem sum_of_sequence (n : ℕ) : S n = n^2 - n := by
  sorry

end sum_of_sequence_l587_587618


namespace circle_tangent_and_AB_distance_l587_587633

noncomputable section

-- Definitions based on the problem conditions
def parabola := {p : ℝ × ℝ | p.2 ^ 2 = 4 * (p.1 - 1)}

def vertex : ℝ × ℝ := (1, 0)

def center : ℝ × ℝ := (-1, 0)

def circle (p : ℝ × ℝ) := (p.1 + 1) ^ 2 + p.2 ^ 2 = 1

def tangent_points (p : ℝ × ℝ) := 
  let x0 := p.1
  let y0 := p.2 in
  y0 ^ 2 = 4 * (x0 - 1)

def distance_AB (x0 y0 : ℝ) : ℝ :=
  sqrt ((2 * y0 / (x0 + 2)) ^ 2 + 4 * (x0 / (x0 + 2)))

-- Theorem statement
theorem circle_tangent_and_AB_distance :
  (∀ p : ℝ × ℝ, p ∈ parabola → circle vertex ∧ 
    ∀ x0 y0 : ℝ, y0 ^ 2 = 4 * (x0 - 1) → 
      (distance_AB x0 y0) ∈ Set.Icc (2 * sqrt 3 / 3) (sqrt 39 / 3)) :=
sorry

end circle_tangent_and_AB_distance_l587_587633


namespace complement_complement_l587_587302

theorem complement_complement (alpha : ℝ) (h : alpha = 35) : (90 - (90 - alpha)) = 35 := by
  -- proof goes here, but we write sorry to skip it
  sorry

end complement_complement_l587_587302


namespace sum_quotient_remainder_l587_587439

theorem sum_quotient_remainder :
  let n := 23 * 17 + 19 in
  let new_n := n * 10 in
  let q := new_n / 23 in
  let r := new_n % 23 in
  q + r = 184 :=
by
  sorry

end sum_quotient_remainder_l587_587439


namespace balls_in_one_box_l587_587456

-- Define the problem setting with 100 boxes and 100 balls.
def balls_boxes_problem :=
  ∃ (f : Fin 100 → ℕ),
  (∀ k (S : Finset (Fin 100)), S.card = k → k < 100 → (S.sum f ≠ k)) →
  (∃ i : Fin 100, f i = 100 ∧ ∀ j : Fin 100, j ≠ i → f j = 0)

-- State the theorem to conclude that all balls are in one single box.
theorem balls_in_one_box : balls_boxes_problem :=
begin
  sorry
end

end balls_in_one_box_l587_587456


namespace max_min_sum_eq_six_l587_587612

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  3 + (a^x - 1) / (a^x + 1) + x * Real.cos x

theorem max_min_sum_eq_six {a : ℝ} (h₀ : 0 < a) (h₁ : a ≠ 1) :
  let I := Set.Icc (-1 : ℝ) 1,
      M := Real.Sup (Set.image (f a) I),
      N := Real.Inf (Set.image (f a) I) in
  M + N = 6 :=
by
  sorry

end max_min_sum_eq_six_l587_587612


namespace grooming_adjustment_time_l587_587688

theorem grooming_adjustment_time :
  ∃ t_adjustment : ℕ, t_adjustment = 20 ∧
  let t_clip := 10,
      t_clean := 90,
      t_shampoo := 5 * 60,
      t_total := 640,
      t_groom := (18 * t_clip) + (2 * t_clean) + t_shampoo in
  t_total + t_adjustment = t_groom :=
by
  existsi 20
  split
  · refl
  · let t_clip := 10
    let t_clean := 90
    let t_shampoo := 5 * 60
    let t_total := 640
    let t_groom := (18 * t_clip) + (2 * t_clean) + t_shampoo
    have : t_groom = 660 := by norm_num
    rw this
    norm_num

end grooming_adjustment_time_l587_587688


namespace smallest_prime_divisor_of_sum_l587_587469

theorem smallest_prime_divisor_of_sum (a b : ℕ) 
  (h₁ : a = 3 ^ 15) 
  (h₂ : b = 11 ^ 21) 
  (h₃ : odd a) 
  (h₄ : odd b) : 
  nat.prime_divisors (a + b) = [2] := 
by
  sorry

end smallest_prime_divisor_of_sum_l587_587469


namespace minimum_cans_required_l587_587128

theorem minimum_cans_required (h : ∀ (a b : ℕ), a * b = 192 → a = 12 ∨ a = -12) 
  (h_can_size : ∀ (c : ℕ), c = 16) 
  (h_gallon : ∀ (g : ℕ), g = 128) : 
  ∃ n : ℕ, 1.5 * h_gallon 1 * h_can_size 1 ≤ n * h_can_size 1 ∧ n = 12 :=
by
  sorry

end minimum_cans_required_l587_587128


namespace sum_of_geometric_seq_not_geometric_l587_587705

noncomputable def is_geometric_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
∀ n : ℕ, a (n + 1) = a n * q

noncomputable def geometric_sum (a : ℕ → ℝ) (n : ℕ) : ℝ :=
finset.sum (finset.range n) a

noncomputable def is_not_geometric_sum_seq (S : ℕ → ℝ) : Prop :=
¬ ∃ r : ℝ, ∀ n : ℕ, S (n + 1) = S n * r

theorem sum_of_geometric_seq_not_geometric (a : ℕ → ℝ) (q : ℝ) (hq : q ≠ 0)
  (h : is_geometric_seq a q) :
  is_not_geometric_sum_seq (geometric_sum a) :=
sorry

end sum_of_geometric_seq_not_geometric_l587_587705


namespace trigonometric_sum_l587_587594

theorem trigonometric_sum (θ : ℝ) (h_tan_θ : Real.tan θ = 5 / 12) (h_range : π ≤ θ ∧ θ ≤ 3 * π / 2) : 
  Real.cos θ + Real.sin θ = -17 / 13 :=
by
  sorry

end trigonometric_sum_l587_587594


namespace find_BD_l587_587363

noncomputable def triangle_BD (AC area : ℝ) : ℝ :=
  let BD := 2 * area / AC in
  BD

theorem find_BD : 
  ∀ {A B C D : Type} {AC : ℝ} {area : ℝ}, 
    (AC = 20) → (area = 200) → 
    (triangle_BD AC area = 20) := by
  intros A B C D AC area h1 h2
  unfold triangle_BD
  rw [h1, h2]
  norm_num
  rfl
  done

end find_BD_l587_587363


namespace circumcircle_contains_F_l587_587244

open Euclidean Geometry

variable {P : Type*} [EuclideanGeometry P]

-- Definitions of line v, point F, and points E1, E2, E3
variables (v : Line P) (F : P) (E1 E2 E3 : P)

-- Conditions stating E1, E2, E3 are equidistant from F and line v
def equidistant (E P: P) (line : Line P) := dist E P = dist_point_line E line

axiom equidistant_E1 : equidistant E1 F v
axiom equidistant_E2 : equidistant E2 F v
axiom equidistant_E3 : equidistant E3 F v

-- Definitions of the lines e1, e2, e3 being the angle bisectors as described
def angle_bisector (E P : P) (line : Line P) : Line P := sorry -- Definition of angle bisector

-- Given definitions of e_i
def e1 : Line P := angle_bisector E1 F v
def e2 : Line P := angle_bisector E2 F v
def e3 : Line P := angle_bisector E3 F v

-- The statement to prove
theorem circumcircle_contains_F : 
  ∃ (circumcircle : Circle P), F ∈ circumcircle ∧ onCircumcircle (triangle e1 e2 e3) circumcircle :=
sorry

end circumcircle_contains_F_l587_587244


namespace sum_of_squares_of_solutions_l587_587220

theorem sum_of_squares_of_solutions (s_1 s_2 : ℝ) :
  (s_1 + s_2 = 10) →
  (s_1 * s_2 = 7) →
  (s_1^2 + s_2^2 = 86) :=
begin
  sorry
end

end sum_of_squares_of_solutions_l587_587220


namespace total_amount_correct_l587_587384

/-- Meghan has the following cash denominations: -/
def num_100_bills : ℕ := 2
def num_50_bills : ℕ := 5
def num_10_bills : ℕ := 10

/-- Value of each denomination: -/
def value_100_bill : ℕ := 100
def value_50_bill : ℕ := 50
def value_10_bill : ℕ := 10

/-- Meghan's total amount of money: -/
def total_amount : ℕ :=
  (num_100_bills * value_100_bill) +
  (num_50_bills * value_50_bill) +
  (num_10_bills * value_10_bill)

/-- The proof: -/
theorem total_amount_correct : total_amount = 550 :=
by
  -- sorry for now
  sorry

end total_amount_correct_l587_587384


namespace ratio_sqrt5_over_5_l587_587335

noncomputable def radius_ratio (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) : ℝ :=
a / b

theorem ratio_sqrt5_over_5 (a b : ℝ) (h : π * b^2 - π * a^2 = 4 * π * a^2) :
  radius_ratio a b h = 1 / Real.sqrt 5 := 
sorry

end ratio_sqrt5_over_5_l587_587335


namespace problem1_correct_problem2_correct_l587_587180

noncomputable def problem1 : Real :=
  sqrt 3 * (sqrt 3 - sqrt 2) + sqrt 6

theorem problem1_correct : problem1 = 3 := by
  sorry

variables (a : Real)

noncomputable def problem2 : Real :=
  2 * sqrt (12 * a) + sqrt (6 * a^2) + sqrt (2 * a)

theorem problem2_correct : problem2 = 4 * sqrt (3 * a) + sqrt 6 * a + sqrt (2 * a) := by
  sorry

end problem1_correct_problem2_correct_l587_587180


namespace wave_number_count_l587_587495

def is_wave_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∧ digits.length = 5 
  ∧ (digits.nth 1 > digits.nth 0) ∧ (digits.nth 1 > digits.nth 2)
  ∧ (digits.nth 3 > digits.nth 2) ∧ (digits.nth 3 > digits.nth 4)

def count_wave_numbers (ds : list ℕ) : ℕ :=
  list.countp (λ n, is_wave_number n ∧ n.digits 10 \in ds.permutations) (list.range 100000)

theorem wave_number_count : count_wave_numbers [0, 1, 2, 3, 4, 5, 6, 7] = 721 := 
  sorry

end wave_number_count_l587_587495


namespace gcd_266_209_l587_587453

-- Definitions based on conditions
def a : ℕ := 266
def b : ℕ := 209

-- Theorem stating the GCD of a and b
theorem gcd_266_209 : Nat.gcd a b = 19 :=
by {
  -- Declare the specific integers as conditions
  let a := 266
  let b := 209
  -- Use the Euclidean algorithm (steps within the proof are not required)
  -- State that the conclusion is the GCD of a and b 
  sorry
}

end gcd_266_209_l587_587453


namespace sum_of_palindromic_primes_less_than_70_l587_587578

def is_prime (n : ℕ) : Prop := Nat.Prime n

def reverse_digits (n : ℕ) : ℕ :=
  let digits := n.digits 10
  digits.foldl (λ acc d => acc * 10 + d) 0

def is_palindromic_prime (n : ℕ) : Prop :=
  is_prime n ∧ is_prime (reverse_digits n)

theorem sum_of_palindromic_primes_less_than_70 :
  let palindromic_primes := [11, 13, 31, 37]
  (∀ p ∈ palindromic_primes, is_palindromic_prime p ∧ p < 70) →
  palindromic_primes.sum = 92 :=
by
  sorry

end sum_of_palindromic_primes_less_than_70_l587_587578


namespace acetic_acid_molecular_weight_is_correct_l587_587539

def molecular_weight_acetic_acid : ℝ :=
  let carbon_weight := 12.01
  let hydrogen_weight := 1.008
  let oxygen_weight := 16.00
  let num_carbons := 2
  let num_hydrogens := 4
  let num_oxygens := 2
  num_carbons * carbon_weight + num_hydrogens * hydrogen_weight + num_oxygens * oxygen_weight

theorem acetic_acid_molecular_weight_is_correct : molecular_weight_acetic_acid = 60.052 :=
by 
  unfold molecular_weight_acetic_acid
  sorry

end acetic_acid_molecular_weight_is_correct_l587_587539


namespace transistor_count_2010_l587_587886

theorem transistor_count_2010 : 
  let initial_transistors := 500000 in
  let tripling_period := 2 in
  let years_passed := 2010 - 1990 in
  let tripling_times := years_passed / tripling_period in
  initial_transistors * 3^tripling_times = 29524500000 := 
by
  sorry

end transistor_count_2010_l587_587886


namespace min_value_of_c_l587_587609

theorem min_value_of_c (a b c : ℕ) 
  (h_pos_a : 0 < a) 
  (h_pos_b : 0 < b) 
  (h_pos_c : 0 < c)
  (h_ineq1 : a < b) 
  (h_ineq2 : b < 2 * b) 
  (h_ineq3 : 2 * b < c)
  (h_unique_sol : ∃ x : ℝ, 3 * x + (|x - a| + |x - b| + |x - (2 * b)| + |x - c|) = 3000) :
  c = 502 := sorry

end min_value_of_c_l587_587609


namespace smallest_k_multiple_of_360_l587_587215

theorem smallest_k_multiple_of_360 :
  ∃ k : ℕ, k > 0 ∧ (1^2 + 2^2 + 3^2 + ... + k^2) % 360 = 0 ∧
           (∀ m : ℕ, m > 0 → (1^2 + 2^2 + 3^2 + ... + m^2) % 360 = 0 → k ≤ m) :=
by
  sorry

end smallest_k_multiple_of_360_l587_587215


namespace perpendicular_lines_a_eq_2_l587_587656

/-- Given two lines, ax + 2y + 2 = 0 and x - y - 2 = 0, prove that if these lines are perpendicular, then a = 2. -/
theorem perpendicular_lines_a_eq_2 {a : ℝ} :
  (∃ a, (a ≠ 0)) → (∃ x y, ((ax + 2*y + 2 = 0) ∧ (x - y - 2 = 0)) → - (a / 2) * 1 = -1) → a = 2 :=
by
  sorry

end perpendicular_lines_a_eq_2_l587_587656


namespace probability_of_same_suit_or_number_but_not_both_l587_587523

def same_suit_or_number_but_not_both : Prop :=
  let total_outcomes := 52 * 52
  let prob_same_suit := 12 / 51
  let prob_same_number := 3 / 51
  let prob_same_suit_and_number := 1 / 51
  (prob_same_suit + prob_same_number - 2 * prob_same_suit_and_number) = 15 / 52

theorem probability_of_same_suit_or_number_but_not_both :
  same_suit_or_number_but_not_both :=
by sorry

end probability_of_same_suit_or_number_but_not_both_l587_587523


namespace find_a_b_and_tangent_l587_587284

noncomputable def f (x a b : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem find_a_b_and_tangent :
  (∃ (a b : ℝ), 
    (f' : ℝ → ℝ := λ x, 3 * x^2 - 3 * a) ∧ 
    (f'(-1) = 0) ∧
    (f(-1) a b = 1)) →
  ∃ (a b : ℝ), 
    a = 1 ∧ 
    b = -1 ∧
    let g (x : ℝ) := f x 1 (-1) + real.exp (2 * x - 1) 
    in ∃ (k : ℝ), 
        k = g' 1 ∧
        let g_val_1 := g(1) 
        in 2 * real.exp 1 * x - y - real.exp 1 - 3 = 0 :=
by
  sorry

end find_a_b_and_tangent_l587_587284


namespace smallest_integer_value_l587_587459

theorem smallest_integer_value (x : ℤ) (h : (x^2 - 4*x + 15) / (x - 5) ∈ ℤ) : x = 6 :=
by
  sorry

end smallest_integer_value_l587_587459


namespace max_among_l587_587711

theorem max_among (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ (1 / (2 * real.sqrt 3)) :=
  sorry

end max_among_l587_587711


namespace possible_values_f2001_l587_587427

noncomputable def f : ℕ → ℝ := sorry

lemma functional_equation (a b d : ℕ) (h₁ : 1 < a) (h₂ : 1 < b) (h₃ : d = Nat.gcd a b) :
  f (a * b) = f d * (f (a / d) + f (b / d)) :=
sorry

theorem possible_values_f2001 :
  f 2001 = 0 ∨ f 2001 = 1 / 2 :=
sorry

end possible_values_f2001_l587_587427


namespace divides_p_minus_1_l587_587715

open Nat

variables (a : ℕ → ℕ) (N : ℕ) (p x : ℕ)

def recurrence_seq := ∀ n, a 0 = 2 ∧ (a (n + 1) = 2 * (a n)^2 - 1)

def prime_divisor (N p : ℕ) := Prime p ∧ ∃ n, n ≥ 1 ∧ a N % p = 0

def exists_x (x p : ℕ) := ∃ x, x^2 % p = 3

theorem divides_p_minus_1 (h1 : recurrence_seq a)
  (h2 : prime_divisor N p) (h3 : exists_x x p) : 2^(N + 2) ∣ (p - 1) := 
  sorry

end divides_p_minus_1_l587_587715


namespace condition_A_condition_B_condition_C_condition_D_l587_587314

noncomputable def z (a b : ℝ) : ℂ := complex.mk a b

theorem condition_A (z : ℂ) : z * (complex.I - 1) = complex.I ^ 2023 - 1 → ¬ (complex.re z < 0 ∧ complex.im z > 0) := 
by sorry

theorem condition_B (z : ℂ) : complex.re z = 0 → complex.re z = 0 := 
by sorry  

theorem condition_C (z : ℂ) : |z| ≤ 3 → ¬ π * 3 ^ 2 = 6 * π := 
by sorry  

theorem condition_D (a b : ℝ) (h : a^2 + b^2 = 1) : 
  let z := complex.mk a b in 
  z + (1 / z) ∈ ℝ :=
by sorry

end condition_A_condition_B_condition_C_condition_D_l587_587314


namespace explicit_formula_increasing_function_range_of_t_l587_587989

-- Define function f with given conditions
def f (x : ℝ) : ℝ := (∃ (a b : ℝ), a * x + b / (1 + x^2)) ∧ f 1 = 1 / 2 ∧ f is odd function on (-1, 1)

-- Define the domain
def domain (x : ℝ) := -1 < x ∧ x < 1

-- Statement (1): Explicit formula
theorem explicit_formula : ∀ x, f x = (x / (1 + x^2)) :=
sorry

-- Statement (2): Increasing function
theorem increasing_function (x1 x2 : ℝ) (h1 : domain x1) (h2 : domain x2) (h : x1 < x2): f x1 < f x2 :=
sorry

-- Statement (3): Range of t
theorem range_of_t (t : ℝ) (h : f (2 * t - 1) + f (t - 1) < 0): 0 < t ∧ t < 2 / 3 :=
sorry

end explicit_formula_increasing_function_range_of_t_l587_587989


namespace function_symmetry_l587_587415

theorem function_symmetry (ω : ℝ) (φ : ℝ) (h₀ : ω > 0) (h₁ : |φ| < π/2) (h₂ : ∀ x, sin (ω * (x + π)) = sin (ω * x)) (h₃ : ∀ x, sin (ω * (x - π/6) + φ) = -sin (ω * (x - π/6) - φ))
    : ∃ k : ℤ, (∀ x : ℝ, sin (ω * x + φ) = sin (ω * (x + k * π/2 + π/12))) :=
by
  sorry

end function_symmetry_l587_587415


namespace average_price_of_towels_l587_587885

theorem average_price_of_towels 
    (price1 : ℕ) (qty1 : ℕ) (price2 : ℕ) (qty2 : ℕ) (price3 : ℕ) (qty3 : ℕ) 
    (total_price : ℕ) (total_qty : ℕ) (average_price : ℕ) :
    price1 = 100 → 
    qty1 = 3 → 
    price2 = 150 → 
    qty2 = 5 → 
    price3 = 400 → 
    qty3 = 2 → 
    total_price = price1 * qty1 + price2 * qty2 + price3 * qty3 → 
    total_qty = qty1 + qty2 + qty3 → 
    average_price = total_price / total_qty → 
    average_price = 185 :=
by
  intros hprice1 hqty1 hprice2 hqty2 hprice3 hqty3 htotal_price htotal_qty havg_price,
  sorry

end average_price_of_towels_l587_587885


namespace cos_double_angle_l587_587268

-- Define the point M and initial conditions for theta
variable (θ : ℝ)
variable (M : ℝ × ℝ)
variable (nonneg_half_axis : θ = 0)
variable (terminal_side : M = (-3, 4))
variable (r : ℝ := (M.1^2 + M.2^2).sqrt)

-- Prove that cos 2θ = -7/25 given the conditions
theorem cos_double_angle :
  θ = 0 → M = (-3, 4) →  r = 5 → cos (2 * θ) = -7 / 25 := by
  sorry

end cos_double_angle_l587_587268


namespace sum_of_segments_eq_hypotenuse_l587_587964

variable {P : Type*} [EuclideanPlane P] {A B C K L : P}

def is_right_triangle (A B C : P) : Prop :=
  ∃ (a b c : ℝ), a ^ 2 + b ^ 2 = c ^ 2 ∧ 
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0) ∧ 
  line_thru A B ≠ line_thru B C ∧
  line_thru A C ≠ line_thru B C

def is_angle_bisector (A B C K : P) : Prop :=
  ∃ (K : P), angle A B K = angle K B C

def is_circumcircle_intersect (A K B L : P) : Prop :=
  on_circle (circumcircle A K B) L ∧ L ≠ B ∧
  on_line L B C

theorem sum_of_segments_eq_hypotenuse
  (hABC : is_right_triangle A B C)
  (hBisector_BK : is_angle_bisector A B C K)
  (hCircumcircle : is_circumcircle_intersect A K B L) :
  distance C B + distance C L = distance A B := 
by
  sorry

end sum_of_segments_eq_hypotenuse_l587_587964


namespace asparagus_spears_needed_l587_587897

def BridgetteGuests : Nat := 84
def AlexGuests : Nat := (2 * BridgetteGuests) / 3
def TotalGuests : Nat := BridgetteGuests + AlexGuests
def ExtraPlates : Nat := 10
def TotalPlates : Nat := TotalGuests + ExtraPlates
def VegetarianPercent : Nat := 20
def LargePortionPercent : Nat := 10
def VegetarianMeals : Nat := (VegetarianPercent * TotalGuests) / 100
def LargePortionMeals : Nat := (LargePortionPercent * TotalGuests) / 100
def RegularMeals : Nat := TotalGuests - (VegetarianMeals + LargePortionMeals)
def AsparagusPerRegularMeal : Nat := 8
def AsparagusPerVegetarianMeal : Nat := 6
def AsparagusPerLargePortionMeal : Nat := 12

theorem asparagus_spears_needed : 
  RegularMeals * AsparagusPerRegularMeal + 
  VegetarianMeals * AsparagusPerVegetarianMeal + 
  LargePortionMeals * AsparagusPerLargePortionMeal = 1120 := by
  sorry

end asparagus_spears_needed_l587_587897


namespace solve_for_a_l587_587966

theorem solve_for_a (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : 13 ∣ 51^2016 - a) : a = 1 :=
by {
  sorry
}

end solve_for_a_l587_587966


namespace athletes_selection_correct_possible_outcomes_correct_probability_of_event_A_l587_587802

-- Definitions based on the problem conditions
def athletes_in_association_A := 27
def athletes_in_association_B := 9
def athletes_in_association_C := 18
def total_athletes := athletes_in_association_A + athletes_in_association_B + athletes_in_association_C
def selected_athletes := 6
def sampling_proportion := selected_athletes / total_athletes

def number_from_A := (sampling_proportion * athletes_in_association_A).to_nat
def number_from_B := (sampling_proportion * athletes_in_association_B).to_nat
def number_from_C := (sampling_proportion * athletes_in_association_C).to_nat

-- List of athletes
def A₁ := "A₁"
def A₂ := "A₂"
def A₃ := "A₃"
def A₄ := "A₄"
def A₅ := "A₅"
def A₆ := "A₆"

def all_athletes := [A₁, A₂, A₃, A₄, A₅, A₆]

-- Possible outcomes of selecting 2 from 6
def possible_outcomes := { 
  (A₁, A₂), (A₁, A₃), (A₁, A₄), (A₁, A₅), (A₁, A₆), 
  (A₂, A₃), (A₂, A₄), (A₂, A₅), (A₂, A₆), 
  (A₃, A₄), (A₃, A₅), (A₃, A₆), 
  (A₄, A₅), (A₄, A₆), (A₅, A₆) 
}

-- Event A: at least one of A₅ or A₆ is selected
def event_A := { (A₁, A₅), (A₁, A₆), (A₂, A₅), (A₂, A₆), 
                 (A₃, A₅), (A₃, A₆), (A₄, A₅), (A₄, A₆), (A₅, A₆) }

def probability_event_A := event_A.size / possible_outcomes.size

-- Theorem statements 
theorem athletes_selection_correct : 
  number_from_A + number_from_B + number_from_C = selected_athletes := 
  sorry

theorem possible_outcomes_correct :
  ∀ (outcome : set (string × string)), outcome \in possible_outcomes ↔ true := 
  sorry

theorem probability_of_event_A : 
  probability_event_A = 3 / 5 := 
  sorry

end athletes_selection_correct_possible_outcomes_correct_probability_of_event_A_l587_587802


namespace min_value_zero_implies_a_two_inequality_f_le_f_prime_l587_587278

def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1 - x^2) / x^2

-- 1. Prove that if the minimum value of f(x) is 0, then a = 2.
theorem min_value_zero_implies_a_two (a : ℝ) :
  (∃ x : ℝ, f(a, x) = 0 ∧ ∀ y : ℝ, f(a, y) ≥ f(a, x)) → a = 2 :=
by
  sorry

-- 2. Prove that when a = 2, f(x) ≤ f'(x) holds for all x ∈ [1, 2].
theorem inequality_f_le_f_prime (x : ℝ) :
  a = 2 ∧ 1 ≤ x ∧ x ≤ 2 → f(2, x) ≤ (deriv (f 2)) x :=
by
  sorry

end min_value_zero_implies_a_two_inequality_f_le_f_prime_l587_587278


namespace larger_number_is_37_point_435_l587_587432

theorem larger_number_is_37_point_435 (x y : ℝ) (h1 : x + y = 40) (h2 : x * y = 96) (h3 : x > y) : x = 37.435 :=
by
  sorry

end larger_number_is_37_point_435_l587_587432


namespace projection_of_right_angled_triangle_l587_587790

/-- Given a right-angled triangle ABC and its projection onto a plane α where the right-angle side AB 
is parallel to α, the projection of the triangle (denoted as A_1B_1C_1) is still a right-angled 
triangle. -/
theorem projection_of_right_angled_triangle (A B C A_1 B_1 C_1 : Type*) [Plane α] 
  (right_angled_triangle : is_right_angled_triangle A B C)
  (AB_parallel_to_alpha : parallel AB α)
  (projection : proj_onto_plane α ABC = triangle A_1 B_1 C_1) :
  is_right_angled_triangle A_1 B_1 C_1 := 
sorry

end projection_of_right_angled_triangle_l587_587790


namespace largest_possible_sum_of_digits_l587_587304

theorem largest_possible_sum_of_digits
  (a b c : Nat)
  (h1 : a < 10)
  (h2 : b < 10)
  (h3 : c < 10)
  (h4 : ∃ y, 0 < y ∧ y ≤ 7 ∧ 0.abc = 1 / y)
  (h5 : 0.abc = abc / 900) :
  a + b + c = 9 :=
by
  sorry

end largest_possible_sum_of_digits_l587_587304


namespace incorrect_option_B_l587_587259

variables (α β γ : Plane) (m n : Line)

-- Define the conditions
variables (perp : Line → Plane → Prop) (parallel : Plane → Plane → Prop)

-- Condition definitions
variables (h1 : ∀ m n : Line, m ≠ n) 
variables (h2 : ∀ α β γ : Plane, α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
variables (h3 : perp m α)
variables (h4 : perp n β)
variables (h5 : parallel α β)

-- Define the hypotheses for Option B
variables (h6 : perp α β)
variables (h7 : perp β γ)

-- Our goal
theorem incorrect_option_B : ¬(parallel α γ) := by
  sorry

end incorrect_option_B_l587_587259


namespace probability_sum_not_less_than_three_l587_587239

theorem probability_sum_not_less_than_three:
  let S := {0, 1, 2, 3}
  let total_choices := { (a, b) | a ∈ S ∧ b ∈ S ∧ a ≠ b }
  let favorable_choices := { (a, b) | a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ a + b ≥ 3 }
  (favorable_choices.size / total_choices.size : ℚ) = 2 / 3 := 
by
  sorry

end probability_sum_not_less_than_three_l587_587239


namespace pos_solution_sum_l587_587412

theorem pos_solution_sum (c d : ℕ) (h_c_pos : 0 < c) (h_d_pos : 0 < d) :
  (∃ x : ℝ, x ^ 2 + 16 * x = 100 ∧ x = Real.sqrt c - d) → c + d = 172 :=
by
  intro h
  sorry

end pos_solution_sum_l587_587412


namespace can_lids_per_box_l587_587159

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l587_587159


namespace number_of_subsets_of_A_l587_587290

def A : Set ℕ := {1, 2, 3}

theorem number_of_subsets_of_A :
  ∃ n : ℕ, n = 8 ∧ ∀ (s : Set ℕ), s ⊆ A → s ∈ Finset.powerset ⟦set.to_finset A⟧ :=
sorry

end number_of_subsets_of_A_l587_587290


namespace loss_percentage_on_refrigerator_is_3_l587_587017

-- Definitions of conditions
def CP_refrigerator : ℕ := 15000
def CP_mobile : ℕ := 8000
def Profit_mobile : ℚ := 10 / 100
def Overall_profit : ℕ := 350

-- The statement to prove
theorem loss_percentage_on_refrigerator_is_3 :
  let SP_mobile := CP_mobile + (Profit_mobile * CP_mobile)
  let CP_total := CP_refrigerator + CP_mobile
  let SP_total := CP_total + Overall_profit
  let SP_refrigerator := SP_total - SP_mobile
  let Loss_refrigerator := CP_refrigerator - SP_refrigerator
  let L := (Loss_refrigerator / CP_refrigerator.to_rat) * 100 in
  L = 3 := by
  sorry

end loss_percentage_on_refrigerator_is_3_l587_587017


namespace jenny_investment_l587_587346

theorem jenny_investment :
  ∃ (m r : ℝ), m + r = 240000 ∧ r = 6 * m ∧ r = 205714.29 :=
by
  sorry

end jenny_investment_l587_587346


namespace solve_equation_l587_587402

noncomputable def unique_solution (x : ℝ) : Prop :=
  2 * x * Real.log x + x - 1 = 0 → x = 1

-- Statement of our theorem
theorem solve_equation (x : ℝ) (h : 0 < x) : unique_solution x := sorry

end solve_equation_l587_587402


namespace wire_length_between_poles_l587_587042

theorem wire_length_between_poles
  (d : ℝ) -- Distance between the bottoms of the poles
  (h1 : ℝ) -- Height of the first pole
  (h2 : ℝ) -- Height of the second pole
  (H : d = 8)
  (H1 : h1 = 10)
  (H2 : h2 = 4) :
  let vertical_diff := h1 - h2 in
  let wire_length := real.sqrt (vertical_diff^2 + d^2) in
  wire_length = 10 :=
by
  sorry

end wire_length_between_poles_l587_587042


namespace one_third_of_sugar_l587_587516

def mixed_to_improper (a b c : ℕ) : ℚ :=
  a + b / c

theorem one_third_of_sugar :
  let cups := mixed_to_improper 5 2 3 in
  (1 / 3) * cups = 1 + 8 / 9 := 
by
  sorry

end one_third_of_sugar_l587_587516


namespace find_three_numbers_l587_587074

theorem find_three_numbers :
  ∃ (a₁ a₄ a₂₅ : ℕ), a₁ + a₄ + a₂₅ = 114 ∧
    ( ∃ r ≠ 1, a₄ = a₁ * r ∧ a₂₅ = a₄ * r * r ) ∧
    ( ∃ d, a₄ = a₁ + 3 * d ∧ a₂₅ = a₁ + 24 * d ) ∧
    a₁ = 2 ∧ a₄ = 14 ∧ a₂₅ = 98 :=
by
  sorry

end find_three_numbers_l587_587074


namespace inequality_solution_sets_l587_587403

variable (a x : ℝ)

theorem inequality_solution_sets:
    ({x | 12 * x^2 - a * x > a^2} =
        if a > 0 then {x | x < -a/4} ∪ {x | x > a/3}
        else if a = 0 then {x | x ≠ 0}
        else {x | x < a/3} ∪ {x | x > -a/4}) :=
by sorry

end inequality_solution_sets_l587_587403


namespace arthur_spent_fraction_l587_587529

-- Define the initial amount and the remaining amount
def initial_amount : ℕ := 200
def remaining_amount : ℕ := 40

-- Define the fraction representing the amount Arthur spent 
def fraction_spent (I R : ℕ) : ℝ :=
  ((I - R) : ℝ) / (I : ℝ)

-- The theorem stating the problem
theorem arthur_spent_fraction (I R : ℕ) 
(hI : I = initial_amount)
(hR : R = remaining_amount) :
  fraction_spent I R = 4 / 5 :=
by
  -- This is where the proof would go, but we add sorry to skip the proof.
  sorry

end arthur_spent_fraction_l587_587529


namespace complex_roots_sum_condition_l587_587617

theorem complex_roots_sum_condition 
  (z1 z2 : ℂ) 
  (h1 : ∀ z, z ^ 2 + z + 1 = 0) 
  (h2 : z1 ^ 2 + z1 + 1 = 0)
  (h3 : z2 ^ 2 + z2 + 1 = 0) : 
  (z2 / (z1 + 1)) + (z1 / (z2 + 1)) = -2 := 
 sorry

end complex_roots_sum_condition_l587_587617


namespace smallest_prime_divisor_sum_odd_powers_l587_587462

theorem smallest_prime_divisor_sum_odd_powers :
  (∃ p : ℕ, prime p ∧ p ∣ (3^15 + 11^21) ∧ p = 2) :=
by
  have h1 : 3^15 % 2 = 1 := by sorry
  have h2 : 11^21 % 2 = 1 := by sorry
  have h3 : (3^15 + 11^21) % 2 = 0 := by
    rw [← Nat.add_mod, h1, h2]
    exact Nat.mod_add_mod 1 1 2
  use 2
  constructor
  · exact Nat.prime_two
  · rw [Nat.dvd_iff_mod_eq_zero, h3] 
  · rfl

end smallest_prime_divisor_sum_odd_powers_l587_587462


namespace thomas_jefferson_handshakes_l587_587894

theorem thomas_jefferson_handshakes
  (num_couples : ℕ)
  (num_special_guest : ℕ)
  (total_people : ℕ)
  (handshakes_men : ℕ)
  (handshakes_men_women : ℕ)
  (handshakes_guest : ℕ)
  (total_handshakes : ℕ)
  (h1 : num_couples = 15)
  (h2 : num_special_guest = 1)
  (h3 : total_people = (2 * num_couples) + num_special_guest)
  (h4 : handshakes_men = nat.choose num_couples 2)
  (h5 : handshakes_men_women = num_couples * (num_couples * 2 - 1))
  (h6 : handshakes_guest = total_people - 1)
  (h7 : total_handshakes = handshakes_men + handshakes_men_women + handshakes_guest) :
  total_handshakes = 345 :=
by
  sorry

end thomas_jefferson_handshakes_l587_587894


namespace equilateral_triangle_side_length_l587_587527

theorem equilateral_triangle_side_length (sum_perimeters : ℕ) (s : ℕ) 
  (H1 : sum_perimeters = 270) 
  (H2 : (3 * s) + (3 * (s / 2)) + (3 * (s / 4)) + (3 * (s / 8)) + ... = sum_perimeters) : 
  s = 45 := 
by 
  sorry

end equilateral_triangle_side_length_l587_587527


namespace distinct_integer_roots_iff_l587_587933

theorem distinct_integer_roots_iff (a : ℤ) :
  (∃ x y : ℤ, x ≠ y ∧ 2 * x^2 - a * x + 2 * a = 0 ∧ 2 * y^2 - a * y + 2 * a = 0) ↔ a = -2 ∨ a = 18 :=
by
  sorry

end distinct_integer_roots_iff_l587_587933


namespace mike_avg_speed_l587_587740

/-
  Given conditions:
  * total distance d = 640 miles
  * half distance h = 320 miles
  * first half average rate r1 = 80 mph
  * time for first half t1 = h / r1 = 4 hours
  * second half time t2 = 3 * t1 = 12 hours
  * total time tt = t1 + t2 = 16 hours
  * total distance d = 640 miles
  * average rate for entire trip should be (d/tt) = 40 mph.
  
  The goal is to prove that the average rate for the entire trip is 40 mph.
-/
theorem mike_avg_speed:
  let d := 640 in
  let h := 320 in
  let r1 := 80 in
  let t1 := h / r1 in
  let t2 := 3 * t1 in
  let tt := t1 + t2 in
  let avg_rate := d / tt in
  avg_rate = 40 := by
  sorry

end mike_avg_speed_l587_587740


namespace initial_water_amount_l587_587500

-- Defining the initial amount of water W as a variable.
variable (W : ℝ)

-- Given conditions
def daily_evaporation := 0.007 -- ounces
def days := 50
def evaporation_percentage := 3.5 / 100 -- which is 3.5000000000000004 percent as a fraction

-- Defining the total evaporation over the period (50 days)
def total_evaporation := daily_evaporation * days

-- The equation relating the total evaporation to the initial amount of water
def evaporation_equation := evaporation_percentage * W = total_evaporation

-- The proposition to prove
theorem initial_water_amount:
  evaporation_equation → W = 10 :=
by
  sorry

end initial_water_amount_l587_587500


namespace range_of_quadratic_function_l587_587916

variable (x : ℝ)
def quadratic_function (x : ℝ) : ℝ := x^2 - 2 * x + 2

theorem range_of_quadratic_function :
  (∀ y : ℝ, (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y = quadratic_function x) ↔ (1 ≤ y ∧ y ≤ 5)) :=
by
  sorry

end range_of_quadratic_function_l587_587916


namespace base_eight_conversion_l587_587083

def base_eight_to_base_ten (n : ℕ) : ℕ := 
  (1 * 8^3) + (4 * 8^2) + (2 * 8^1) + (3 * 8^0)

theorem base_eight_conversion : base_eight_to_base_ten 1423 = 787 :=
by 
  unfold base_eight_to_base_ten
  calc
    (1 * 8^3) + (4 * 8^2) + (2 * 8^1) + (3 * 8^0)
        = 512 + 256 + 16 + 3 : by sorry
    ... = 787 : by sorry

end base_eight_conversion_l587_587083


namespace smallest_s_for_347_l587_587047

open Nat

theorem smallest_s_for_347 (r s : ℕ) (hr_pos : 0 < r) (hs_pos : 0 < s) 
  (h_rel_prime : Nat.gcd r s = 1) (h_r_lt_s : r < s) 
  (h_contains_347 : ∃ k : ℕ, ∃ y : ℕ, 10 ^ k * r - s * y = 347): 
  s = 653 := 
by sorry

end smallest_s_for_347_l587_587047


namespace range_of_vector_magnitude_l587_587308

variable {V : Type} [NormedAddCommGroup V]

theorem range_of_vector_magnitude
  (A B C : V)
  (h_AB : ‖A - B‖ = 8)
  (h_AC : ‖A - C‖ = 5) :
  3 ≤ ‖B - C‖ ∧ ‖B - C‖ ≤ 13 :=
sorry

end range_of_vector_magnitude_l587_587308


namespace runner_distance_l587_587329

theorem runner_distance (track_length race_length : ℕ) (A_speed B_speed C_speed : ℚ)
  (h1 : track_length = 400) (h2 : race_length = 800)
  (h3 : A_speed = 1) (h4 : B_speed = 8 / 7) (h5 : C_speed = 6 / 7) :
  ∃ distance_from_finish : ℚ, distance_from_finish = 200 :=
by {
  -- We are not required to provide the actual proof steps, just setting up the definitions and initial statements for the proof.
  sorry
}

end runner_distance_l587_587329


namespace projection_of_c_onto_b_l587_587303

open Real

noncomputable def vector_projection (a b : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := a.1 * b.1 + a.2 * b.2
  let magnitude_b := sqrt (b.1^2 + b.2^2)
  let scalar := dot_product / magnitude_b
  (scalar * b.1 / magnitude_b, scalar * b.2 / magnitude_b)

theorem projection_of_c_onto_b :
  let a := (2, 3)
  let b := (-4, 7)
  let c := (-a.1, -a.2)
  vector_projection c b = (-sqrt 65 / 5, -sqrt 65 / 5) :=
by sorry

end projection_of_c_onto_b_l587_587303


namespace fill_tank_time_20_minutes_l587_587447

open_locale classical

def time_to_fill_tank : ℝ :=
  let R1 := 1 / 18 in
  let R2 := 1 / 60 in
  let R3 := -1 / 45 in
  let combined_rate := R1 + R2 + R3 in
  1 / combined_rate

theorem fill_tank_time_20_minutes : time_to_fill_tank = 20 := by
  sorry

end fill_tank_time_20_minutes_l587_587447


namespace superior_sequences_count_l587_587245

noncomputable def number_of_superior_sequences (n : ℕ) : ℕ :=
  Nat.choose (2 * n + 1) (n + 1) * 2^n

theorem superior_sequences_count (n : ℕ) (h : 2 ≤ n) 
  (x : Fin (n + 1) → ℤ)
  (h1 : ∀ i, 0 ≤ i ∧ i ≤ n → |x i| ≤ n)
  (h2 : ∀ i j, 0 ≤ i ∧ i < j ∧ j ≤ n → x i ≠ x j)
  (h3 : ∀ (i j k : Nat), 0 ≤ i ∧ i < j ∧ j < k ∧ k ≤ n → 
    max (|x k - x i|) (|x k - x j|) = 
    (|x i - x j| + |x j - x k| + |x k - x i|) / 2) :
  number_of_superior_sequences n = Nat.choose (2 * n + 1) (n + 1) * 2^n :=
sorry

end superior_sequences_count_l587_587245


namespace max_complete_bouquets_l587_587815

-- Definitions based on conditions
def total_roses := 20
def total_lilies := 15
def total_daisies := 10

def wilted_roses := 12
def wilted_lilies := 8
def wilted_daisies := 5

def roses_per_bouquet := 3
def lilies_per_bouquet := 2
def daisies_per_bouquet := 1

-- Calculation of remaining flowers
def remaining_roses := total_roses - wilted_roses
def remaining_lilies := total_lilies - wilted_lilies
def remaining_daisies := total_daisies - wilted_daisies

-- Proof statement
theorem max_complete_bouquets : 
  min
    (remaining_roses / roses_per_bouquet)
    (min (remaining_lilies / lilies_per_bouquet) (remaining_daisies / daisies_per_bouquet)) = 2 :=
by
  sorry

end max_complete_bouquets_l587_587815


namespace angle_BDC_of_quadrilateral_ABCD_l587_587546

theorem angle_BDC_of_quadrilateral_ABCD
  (A B C D : Type)
  [Real.angle A B C = 50]
  [Real.angle D = 120]
  [Real.angle BAC = 40] :
  Real.angle BDC = 190 :=
by
  sorry

end angle_BDC_of_quadrilateral_ABCD_l587_587546


namespace complex_problem_proof_l587_587272

open Complex

noncomputable def z : ℂ := (1 - I)^2 + 1 + 3 * I

theorem complex_problem_proof : z = 1 + I ∧ abs (z - 2 * I) = Real.sqrt 2 ∧ (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := 
by
  have h1 : z = (1 - I)^2 + 1 + 3 * I := rfl
  have h2 : z = 1 + I := sorry
  have h3 : abs (z - 2 * I) = Real.sqrt 2 := sorry
  have h4 : (∀ a b : ℝ, (z^2 + a + b = 1 - I) → (a = -3 ∧ b = 4)) := sorry
  exact ⟨h2, h3, h4⟩

end complex_problem_proof_l587_587272


namespace max_n_l587_587942

noncomputable def prod := 160 * 170 * 180 * 190

theorem max_n : ∃ n : ℕ, n = 30499 ∧ n^2 ≤ prod := by
  sorry

end max_n_l587_587942


namespace geometric_sequence_ratio_q3_l587_587596

theorem geometric_sequence_ratio_q3 (a : ℕ → ℝ) (q : ℝ) (h : q ≠ 1) (h_geom : ∀ n, a (n + 1) = q * a n) :
  let b := λ n, a (3*n - 2) + a (3*n - 1) + a (3*n)
  in ∀ n, b (n + 1) = q^3 * b n :=
by sorry

end geometric_sequence_ratio_q3_l587_587596


namespace Minnie_vs_Penny_time_difference_l587_587001

theorem Minnie_vs_Penny_time_difference:
  let distance_uphill := 5
  let distance_sandy := 10
  let distance_downhill := 20
  let minnie_speed_uphill := 6
  let minnie_speed_sandy := 15
  let minnie_speed_downhill := 32
  let penny_speed_uphill := 12
  let penny_speed_sandy := 20
  let penny_speed_downhill := 35
  let minnie_time_uphill := distance_uphill / minnie_speed_uphill
  let minnie_time_sandy := distance_sandy / minnie_speed_sandy
  let minnie_time_downhill := distance_downhill / minnie_speed_downhill
  let penny_time_uphill := distance_uphill / penny_speed_uphill
  let penny_time_sandy := distance_sandy / penny_speed_sandy
  let penny_time_downhill := distance_downhill / penny_speed_downhill
  let minnie_total_time := minnie_time_uphill + minnie_time_sandy + minnie_time_downhill
  let penny_total_time := penny_time_uphill + penny_time_sandy + penny_time_downhill
  let minnie_total_time_minutes := minnie_total_time * 60
  let penny_total_time_minutes := penny_total_time * 60
  let time_difference := minnie_total_time_minutes - penny_total_time_minutes
  in time_difference = 88 := by
  sorry

end Minnie_vs_Penny_time_difference_l587_587001


namespace complex_expression_l587_587207

theorem complex_expression :
  (8 : ℂ) - 5 * complex.I + 3 * ((2 : ℂ) - 4 * complex.I) = 14 - 17 * complex.I :=
by
  sorry

end complex_expression_l587_587207


namespace equation_of_line_slope_of_line_l587_587626

-- Definitions and conditions
def circle_eq := ∀ (x y : ℝ), x^2 + (y - 1)^2 = 5
def point_p := (1, 1)
def slope_angle_theta := real.pi / 4
def chord_length_ab := real.sqrt 17

-- Problem (1): Prove equation of the line
theorem equation_of_line (x y : ℝ) (h1 : point_p ∈ (λ p, fst p = 1 ∧ snd p = 1))
  (h2 : slope_angle_theta = real.pi / 4)
  (h3 : slope := 1) :
  x - y = 0 :=
by
  sorry

-- Problem (2): Prove the slope of the line
theorem slope_of_line (m : ℝ) (h1 : point_p ∈ (λ p, fst p = 1 ∧ snd p = 1))
  (h2 : ∃ (x y : ℝ), circle_eq x y)
  (h3 : ∃ (A B : ℝ × ℝ), segment_length_ab = chord_length_ab)
  (h4 : distance_from_chord_to_center := real.sqrt 3 / 2) :
  m = real.sqrt 3 ∨ m = -real.sqrt 3 :=
by
  sorry

end equation_of_line_slope_of_line_l587_587626


namespace product_is_integer_l587_587401

theorem product_is_integer (n : ℕ) : ∃ (k : ℤ), ∏ i in Finset.range (n + 1), (4 - (2 / (i + 1) : ℝ)) = k :=
by
  sorry

end product_is_integer_l587_587401


namespace calculate_angle_BAP_l587_587371

variable {A B C E D P : Type}
variables (triangle_ABC : Triangle A B C)
variables (angle_BAC : angle A B C = 60)
variables (point_E_on_BC : B = C → False)
variables (angle_rel : ∀ {γ : ℝ}, γ * 2 = angle A C)

variables (circ_AEC : CyclicQuadrilateral A E C)
variables (point_D_on_AB_circ_AEC : ∃ (D : Type), InCircle D A E C ∧ D ≠ A)
variables (circ_DBE : CyclicQuadrilateral D B E)
variables (point_P_on_CD_circ_DBE : ∃ (P : Type), InCircle P D E B ∧ P ≠ D)

theorem calculate_angle_BAP
  (h1 : InTriangle A B C)
  (h2 : ∠ A B C = 60)
  (h3 : 2 * ∠ B A E = ∠ A C B)
  (h4 : ∃ D, Intersects (Circumcircle A E C) (Line A B) ∧ D ≠ A)
  (h5 : ∃ P, Intersects (Circumcircle D B E) (Line C D) ∧ P ≠ D) :
  ∠ B A P = 30 :=
begin
  sorry
end

end calculate_angle_BAP_l587_587371


namespace longer_train_length_l587_587812

noncomputable def length_of_longer_train 
  (speed_train1 : ℝ) (speed_train2 : ℝ) (length_shorter_train : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := (speed_train1 + speed_train2) * (5 / 18)
  in (relative_speed * time_to_cross) - length_shorter_train

theorem longer_train_length 
  (speed_train1 : ℝ) (speed_train2 : ℝ) (length_shorter_train : ℝ) (time_to_cross : ℝ) :
  speed_train1 = 60 → speed_train2 = 40 →
  length_shorter_train = 120 → time_to_cross = 10.07919366450684 →
  length_of_longer_train speed_train1 speed_train2 length_shorter_train time_to_cross = 157.8220467912412 := by
  intros h1 h2 h3 h4
  simp [length_of_longer_train, h1, h2, h3, h4]
  sorry

end longer_train_length_l587_587812


namespace PQ_perpendicular_AC_l587_587332

/-- Definitions for geometry problem on given convex quadrilateral -/
variables {A B C D E F M P X Y Q : Type}
variable (points_eq : A ≠ B ∧ B ≠ C ∧ C ≠ D ∧ D ≠ A) -- Declare points A, B, C, D are different
variable (convex : convex_quadrilateral A B C D) -- A, B, C, D form a convex quadrilateral
variable (angle_cond : ∠ABC = ∠ADC ∧ ∠ABC < 90) -- ∠ABC = ∠ADC < 90°
variable (bisectors : bisect_angle ABC E ∧ bisect_angle ADC F) -- E, F bisect respective angles
variable (intersection_pe : E ∈ line_through A C) -- E lies on AC
variable (intersection_pf : F ∈ line_through A C) -- F lies on AC
variable (int_angle_bisector : P = angle_bisector_intersection ABC ADC) -- P is intersection of angle bisectors
variable (midpoint_m : M = midpoint A C) -- M is midpoint of AC
variable (circum_circle : Γ = circum_circle (triangle B P D)) -- Γ is the circumcircle of ΔBPD
variable (intersection_circle : X = circle_second_intersection Γ B M ∧ Y = circle_second_intersection Γ D M) -- X, Y are second intersections
variable (line_intersections : Q = intersection (line_through X E) (line_through Y F)) -- Q is intersection of XE and YF

/-- The main theorem stating that PQ is perpendicular to AC -/
theorem PQ_perpendicular_AC : perp PQ AC :=
sorry -- Proof to be provided here

end PQ_perpendicular_AC_l587_587332


namespace integral_value_of_binomial_expansion_l587_587266

theorem integral_value_of_binomial_expansion :
  (binomial_expansion_condition : (C(9, 3) * (1 / (2 * (-1)))^3) = -21/2) →
  ∫ x in 1..e, (x - (1 / x)) = (e^2 - 3) / 2 :=
sorry

end integral_value_of_binomial_expansion_l587_587266


namespace hexagon_angles_l587_587331

theorem hexagon_angles (a e : ℝ) (h1 : a = e - 60) (h2 : 4 * a + 2 * e = 720) :
  e = 160 :=
by
  sorry

end hexagon_angles_l587_587331


namespace find_number_l587_587867

-- Definitions and conditions for the problem
def N_div_7 (N R_1 : ℕ) : ℕ := (N / 7) * 7 + R_1
def N_div_11 (N R_2 : ℕ) : ℕ := (N / 11) * 11 + R_2
def N_div_13 (N R_3 : ℕ) : ℕ := (N / 13) * 13 + R_3

theorem find_number 
  (N a b c R_1 R_2 R_3 : ℕ) 
  (hN7 : N = 7 * a + R_1)
  (hN11 : N = 11 * b + R_2)
  (hN13 : N = 13 * c + R_3)
  (hQ : a + b + c = 21)
  (hR : R_1 + R_2 + R_3 = 21)
  (hR1_lt : R_1 < 7)
  (hR2_lt : R_2 < 11)
  (hR3_lt : R_3 < 13) : 
  N = 74 :=
sorry

end find_number_l587_587867


namespace quadratic_inequality_solution_l587_587792

theorem quadratic_inequality_solution :
  { m : ℝ // ∀ x : ℝ, m * x^2 - 6 * m * x + 5 * m + 1 > 0 } = { m : ℝ // 0 ≤ m ∧ m < 1/4 } :=
sorry

end quadratic_inequality_solution_l587_587792


namespace power_of_i_2014_l587_587261

-- Define the imaginary unit i 
def i : ℂ := complex.I  -- complex.I is predefined as the imaginary unit in Lean

-- State the problem as a theorem
theorem power_of_i_2014 : (i ^ 2014) = -1 := by
  sorry

end power_of_i_2014_l587_587261


namespace prove_condition_count_l587_587889

theorem prove_condition_count :
  ∃ (count : ℕ), count = 3 ∧ (
    (∀ (a b : ℝ), ab > 0 → (a * b > 0 → (b / a + a / b ≥ 2) = true)) ∧
    (∀ (a b : ℝ), ab > 0 → (a > 0 ∧ b > 0 → (b / a + a / b ≥ 2) = true)) ∧
    (∀ (a b : ℝ), ab > 0 → (a < 0 ∧ b < 0 → (b / a + a / b ≥ 2) = true))
  ) :=
begin
  -- Proof is omitted
  sorry
end

end prove_condition_count_l587_587889


namespace dentist_age_considered_years_ago_l587_587005

theorem dentist_age_considered_years_ago (A : ℕ) (X : ℕ) (H1 : A = 32) (H2 : (1/6 : ℚ) * (A - X) = (1/10 : ℚ) * (A + 8)) : X = 8 :=
sorry

end dentist_age_considered_years_ago_l587_587005


namespace find_a_l587_587785

-- Define the logarithmic function f
def f (a x : ℝ) : ℝ := log a (x + 1)

-- State the problem in Lean 4
theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x ∈ Icc (1 / 2 : ℝ) 1, f a x ≥ 1) ∧ (∃ x ∈ Icc (1 / 2 : ℝ) 1, f a x = 1) → a = 3 / 2 :=
sorry

end find_a_l587_587785


namespace determine_b_value_l587_587195

theorem determine_b_value (b : Real) (h : 0 < b) : (1 / Real.log 5 b + 1 / Real.log 6 b + 1 / Real.log 7 b = 1) ↔ (b = 210) := 
by
  sorry

end determine_b_value_l587_587195


namespace not_possible_choice_B_l587_587639

def seven_shapes : Finset (Finset (Fin 17)) := sorry

def can_form_figures (f : Finset (Finset (Fin 17))) : Prop :=
∀ x ∈ f, ∃ (n : ℕ), n = 17 ∧ (x.card = n)

def choice_B : Finset (Fin (17 + 1)) := sorry

theorem not_possible_choice_B :
  ¬ can_form_figures (insert choice_B seven_shapes) :=
sorry

end not_possible_choice_B_l587_587639


namespace area_of_S_solution_l587_587874

noncomputable def area_of_S : Real :=
  let H_center : Complex := 0
  let distance_between_opposite_sides : Real := 1
  let sides_parallel_to_imaginary_axis : Bool := true
  let R : Set Complex := {z : Complex | ∀ h ∈ H, abs (z - H_center) > abs (h - H_center)}
  let S : Set Complex := {1 / z | z ∈ R}
  3 * Real.sqrt 3 + 2 * Real.pi
  
theorem area_of_S_solution : area_of_S = 3 * Real.sqrt 3 + 2 * Real.pi := by
  dsimp [area_of_S]
  sorry -- Proof will be required here when implementing the solution steps.

end area_of_S_solution_l587_587874


namespace sleep_duration_7pm_to_9am_l587_587854

theorem sleep_duration_7pm_to_9am :
  ∀ (bed_time alarm_time : ℕ), bed_time = 19 → alarm_time = 9 → (∃ sleep_duration : ℕ, alarm_time - bed_time = sleep_duration) → sleep_duration = 2 :=
begin
  intros bed_time alarm_time bed_time_is_7pm alarm_time_is_9am exists_duration,
  rw bed_time_is_7pm at *,
  rw alarm_time_is_9am at *,
  let sleep_duration := alarm_time + 12 - bed_time,
  existsi sleep_duration,
  show sleep_duration = 2,
  sorry -- Proof is omitted
end

-- This statement sets the conditions and uses them to state that the sleep duration should be 2 hours.

end sleep_duration_7pm_to_9am_l587_587854


namespace value_of_f_at_13_over_2_l587_587968

noncomputable def f (x : ℝ) : ℝ := sorry

theorem value_of_f_at_13_over_2
  (h1 : ∀ x : ℝ , f (-x) = -f (x))
  (h2 : ∀ x : ℝ , f (x - 2) = f (x + 2))
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f (x) = -x^2) :
  f (13 / 2) = 9 / 4 :=
sorry

end value_of_f_at_13_over_2_l587_587968


namespace y_coordinate_of_C_l587_587388

theorem y_coordinate_of_C (h : ℝ) (H : ∀ (C : ℝ), C = h) :
  let A := (0, 0)
      B := (0, 5)
      C := (3, h)
      D := (6, 5)
      E := (6, 0)
  -- Assuming the area of the pentagon is 50
  let area_square_ABDE := 25
      area_triangle_BCD := 25
  -- Assuming the height of triangle BCD
  let height_triangle_BCD := h - 5
      base_triangle_BCD := 6
      area_BCD := (1/2) * base_triangle_BCD * height_triangle_BCD in
  area_square_ABDE + area_triangle_BCD = 50 →
  area_BCD = area_triangle_BCD →
  h = 40 / 3 :=
by intros h H A B C D E area_square_ABDE area_triangle_BCD height_triangle_BCD base_triangle_BCD area_BCD;
   sorry

end y_coordinate_of_C_l587_587388


namespace find_a_plus_b_l587_587566

theorem find_a_plus_b (a b : ℝ) : (3 = 1/3 * 1 + a) → (1 = 1/3 * 3 + b) → a + b = 8/3 :=
by
  intros h1 h2
  sorry

end find_a_plus_b_l587_587566


namespace train_cross_signal_pole_in_18_seconds_l587_587121

noncomputable def train_length : ℝ := 300
noncomputable def platform_length : ℝ := 550
noncomputable def crossing_time_platform : ℝ := 51
noncomputable def signal_pole_crossing_time : ℝ := 18

theorem train_cross_signal_pole_in_18_seconds (t l_p t_p t_s : ℝ)
    (h1 : t = train_length)
    (h2 : l_p = platform_length)
    (h3 : t_p = crossing_time_platform)
    (h4 : t_s = signal_pole_crossing_time) : 
    (t + l_p) / t_p = train_length / signal_pole_crossing_time :=
by
  unfold train_length platform_length crossing_time_platform signal_pole_crossing_time at *
  -- proof will go here
  sorry

end train_cross_signal_pole_in_18_seconds_l587_587121


namespace even_odd_mul_is_odd_l587_587975

-- Definitions of even and odd functions
def even (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def odd (g : ℝ → ℝ) := ∀ x, g (-x) = -g x

-- Prove that f is even, g is odd implies f * g is odd
theorem even_odd_mul_is_odd (f g : ℝ → ℝ) (hf : even f) (hg : odd g) :
  odd (λ x, f x * g x) :=
by
  sorry

end even_odd_mul_is_odd_l587_587975


namespace find_y_when_x_is_6_l587_587065

-- Define the variables and their conditions
variables (x y : ℝ)

-- Noncomputable because we have no explicit function definitions for now
noncomputable def sum_eq (x y : ℝ) := x + y = 30
noncomputable def diff_eq (x y : ℝ) := x - y = 6
noncomputable def inv_prop (x y : ℝ) := x * y = 36 * 6

-- The theorem to prove the desired result
theorem find_y_when_x_is_6 (x y : ℝ) 
  (h_sum : sum_eq x y) (h_diff : diff_eq x y) (h_inv_prop : inv_prop x y) 
  (hx : x = 6) : y = 36 :=
begin
  sorry
end

end find_y_when_x_is_6_l587_587065


namespace inequality_holds_for_all_x_in_interval_l587_587075

variable (a x : ℝ)

theorem inequality_holds_for_all_x_in_interval :
  (∀ x ∈ set.Icc 1 12, x^2 + 25 + abs (x^3 - 5 * x^2) ≥ a * x) → a ≤ 10 :=
by
  intro h
  -- Step-wise proof would be here, involving the checking that the inequality is bounded correctly.
  sorry

end inequality_holds_for_all_x_in_interval_l587_587075


namespace number_of_blocks_needed_l587_587687

-- Define the dimensions of the fort
def fort_length : ℕ := 20
def fort_width : ℕ := 15
def fort_height : ℕ := 8

-- Define the thickness of the walls and the floor
def wall_thickness : ℕ := 2
def floor_thickness : ℕ := 1

-- Define the original volume of the fort
def V_original : ℕ := fort_length * fort_width * fort_height

-- Define the interior dimensions of the fort considering the thickness of the walls and floor
def interior_length : ℕ := fort_length - 2 * wall_thickness
def interior_width : ℕ := fort_width - 2 * wall_thickness
def interior_height : ℕ := fort_height - floor_thickness

-- Define the volume of the interior space
def V_interior : ℕ := interior_length * interior_width * interior_height

-- Statement to prove: number of blocks needed equals 1168
theorem number_of_blocks_needed : V_original - V_interior = 1168 := 
by 
  sorry

end number_of_blocks_needed_l587_587687


namespace exists_constants_l587_587192

theorem exists_constants :
  ∃ (a b c : ℝ), a • (1 : ℝ) • ℕ × 4 + b • (3 : ℝ) • (-1) + c • (0) • (1) = (5 : ℝ) + (1 : ℝ) :=
sorry

end exists_constants_l587_587192


namespace soccer_league_fraction_female_proof_l587_587006

variable (m f : ℝ)

def soccer_league_fraction_female : Prop :=
  let males_last_year := m
  let females_last_year := f
  let males_this_year := 1.05 * m
  let females_this_year := 1.2 * f
  let total_this_year := 1.1 * (m + f)
  (1.05 * m + 1.2 * f = 1.1 * (m + f)) → ((0.6 * m) / (1.65 * m) = 4 / 11)

theorem soccer_league_fraction_female_proof (m f : ℝ) : soccer_league_fraction_female m f :=
by {
  sorry
}

end soccer_league_fraction_female_proof_l587_587006


namespace volume_Omega₂_eq_7_l587_587683

-- Definition of the region Omega_2
def Omega₂ : Set (ℝ × ℝ × ℝ) :=
{ p | p.1^2 + p.2^2 + p.2.snd.snd^2 ≤ 1 }

-- Definition of the floor function
def floor (x : ℝ) : ℤ := x.to_int

-- The theorem to prove the volume of the region Ω₂ is 7.
theorem volume_Omega₂_eq_7 : volume { p : ℝ × ℝ × ℝ | (floor p.1)^2 + (floor p.2)^2 + (floor p.3)^2 ≤ 1} = 7 := sorry

end volume_Omega₂_eq_7_l587_587683


namespace rhombus_area_l587_587599

def area_of_rhombus (d1 d2 : ℝ) : ℝ :=
  (d1 * d2) / 2

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 14) (h_d2 : d2 = 22) : area_of_rhombus d1 d2 = 154 := by
  rw [h_d1, h_d2]
  have : area_of_rhombus 14 22 = 154 := by sorry
  exact this

end rhombus_area_l587_587599


namespace part1_solution_part2_solution_l587_587454

section part1

/-- Assumption: Digits allowed -/
def digits : Finset ℕ := {0, 1, 2, 3, 4, 5}

/-- Assumption: Set of even digits -/
def even_digits : Finset ℕ := {0, 2, 4}

/-- Predicate to check if a number is a 5-digit number -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n < 100000

/-- Predicate to check if a number is even -/
def is_even (n : ℕ) : Prop := n % 2 = 0

theorem part1_solution : 
  ∃ (count : ℕ), count = 
    (digits.erase 0).card * (digits.card ^ 3) * even_digits.card ∧ count = 3240 :=
by
  sorry

end part1

section part2

/-- 
  Predicate to check if a number is divisible by 5 
  and has no repeating digits, and the hundreds place 
  is not 3. 
-/
def is_divisible_by_5_no_repeats_hundreds_not_3 (n : ℕ) : Prop := 
  n % 5 = 0 ∧ 
  (λ digits, digits.nodup ∧ digits.length = 5 ∧ digits.nth 2 ≠ some 3) (list.of_digits n)

theorem part2_solution :
  ∃ (count : ℕ), count = 
    (set.univ.filter is_divisible_by_5_no_repeats_hundreds_not_3).card ∧ count = 54 :=
by
  sorry

end part2

end part1_solution_part2_solution_l587_587454


namespace exists_alpha_cos_neg_half_l587_587491

theorem exists_alpha_cos_neg_half :
  ∃ α, ∀ n : ℕ, n ≥ 1 → cos (2^n * α) = -1/2 :=
by
  sorry

end exists_alpha_cos_neg_half_l587_587491


namespace f_periodic_2_f_at_five_half_l587_587366

def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 0 then -4 * x^2 + 2
  else if 0 ≤ x ∧ x < 1 then x
  else 0  -- Default value for outside the primary interval, not used in proof

theorem f_periodic_2 (x : ℝ) : f (x + 2) = f x :=
begin
  -- Definition of periodicity
  sorry
end

theorem f_at_five_half : f (5 / 2) = 1 / 2 :=
begin
  have hx : 5 / 2 = 1 / 2 + 2,
  { norm_num },
  calc
    f (5 / 2) = f (1 / 2 + 2)      : by rw hx
          ... = f (1 / 2)          : by rw f_periodic_2 (1 / 2)
          ... = 1 / 2              : by { unfold f, split_ifs, norm_num }
end

end f_periodic_2_f_at_five_half_l587_587366


namespace expected_value_of_max_stick_l587_587437

-- Define the set of bamboo sticks
def bamboo_sticks : set ℕ := {1, 2, 3, 4, 5}

-- Define a probability function based on binomial coefficients
def probability_max (n : ℕ) : ℝ :=
  if n = 5 then 6 / 10
  else if n = 4 then 3 / 10
  else if n = 3 then 1 / 10
  else 0

-- Define the expected value computation for the max of three selected sticks
def expected_value_max : ℝ :=
  ∑ n in {3, 4, 5}, n * probability_max n

-- The theorem to be proved
theorem expected_value_of_max_stick : expected_value_max = 4.5 :=
by
  sorry

end expected_value_of_max_stick_l587_587437


namespace max_ratio_is_one_l587_587556

-- Define the circle and its properties
def circle_eq (x y : ℝ) := x^2 + y^2 = 16

-- Define points A, B, C, D with integer coordinates on the circle
def point_on_circle (x y : ℝ) : Prop :=
  circle_eq x y ∧ ∃ (i j : ℤ), ↑i = x ∧ ↑j = y

-- Define irrational distances
def irrational_distance (p1 p2 : ℝ × ℝ) : Prop :=
  let d := (p1.1 - p2.1)^2 + (p1.2 - p2.2)^2 in ∃ n : ℕ, d = (n^2 : ℝ) ∧ n ≠ 4

-- Define distinct points
def distinct_points (p1 p2 p3 p4 : ℝ × ℝ) : Prop :=
  p1 ≠ p2 ∧ p1 ≠ p3 ∧ p1 ≠ p4 ∧ p2 ≠ p3 ∧ p2 ≠ p4 ∧ p3 ≠ p4

-- Define the distances
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the maximum ratio of AB/CD as 1
theorem max_ratio_is_one (A B C D : ℝ × ℝ)
  (hA : point_on_circle A.1 A.2)
  (hB : point_on_circle B.1 B.2)
  (hC : point_on_circle C.1 C.2)
  (hD : point_on_circle D.1 D.2)
  (h_distinct : distinct_points A B C D)
  (h_irrational_AB : irrational_distance A B)
  (h_irrational_CD : irrational_distance C D) :
  ∀ (r : ℝ), r = distance A B / distance C D → r ≤ 1 :=
sorry

end max_ratio_is_one_l587_587556


namespace area_of_bounded_figure_l587_587356

noncomputable def area_bounded_by_curves : ℝ :=
  ∫ y in 0..2, ( ( (2*y - 2) / (2*y - 1) ) - (2 + 2*y) )

theorem area_of_bounded_figure :
  ∀ (x y t : ℝ),
    (t^3 - (2*y - 1)*t^2 - t + 2*y - 1 = 0) →
    (x*t - 2*t + 2*y = 0) →
    x ≥ 0 →
    y ≤ 2 →
    ∫ y in 0..2, ( ( (2*y - 2) / (2*y - 1) ) - (2 + 2*y) ) = area_bounded_by_curves :=
by
  intros x y t h1 h2 h3 h4
  sorry

end area_of_bounded_figure_l587_587356


namespace part_I_part_II_l587_587795

-- Sum of the first n terms
def S (n : ℕ) : ℕ := 2^(n + 1) - 2

-- First part: general formula for the sequence {a_n}
def a (n : ℕ) : ℕ := 2^n

-- Second part: sequence {b_n} and sum of the first n terms T(n)
def b (n : ℕ) : ℚ := 2^n / (n * (n + 1) * 2^n)
def T (n : ℕ) : ℚ := (n : ℚ) / (n + 1)

theorem part_I : ∀ n : ℕ, n > 0 → (a n = 2^n) := by
  sorry

theorem part_II : ∀ n : ℕ, n > 0 → (T n = ∑ i in Finset.range n, b (i + 1)) := by
  sorry

end part_I_part_II_l587_587795


namespace simplify_expression_l587_587763

variable (x y : ℤ)

theorem simplify_expression : 
  (15 * x + 45 * y) + (7 * x + 18 * y) - (6 * x + 35 * y) = 16 * x + 28 * y :=
by
  sorry

end simplify_expression_l587_587763


namespace max_intersection_value_l587_587380

noncomputable def max_intersection_size (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) : ℕ :=
  if ((A.card = 2019) ∧ (B.card = 2019) ∧ (A ∩ B ∩ C).card = 2018)
  then (A ∩ B ∩ C).card 
  else 0

theorem max_intersection_value (A B C : Finset ℕ) (h1 : (A.card = 2019) ∧ (B.card = 2019)) 
  (h2 : (2 ^ A.card + 2 ^ B.card + 2 ^ C.card = 2 ^ (A ∪ B ∪ C).card)) :
  max_intersection_size A B C h1 h2 = 2018 :=
sorry

end max_intersection_value_l587_587380


namespace union_of_A_B_integers_l587_587610

open Set

variable (A : Set ℤ) (B : Set ℕ)

theorem union_of_A_B_integers :
  A = {-1, 0, 1} → 
  B = {x : ℕ | x < 1} → 
  A ∪ (B : Set ℤ) = {-1, 0, 1} :=
by
  intros hA hB
  rw [hA, hB]
  sorry

end union_of_A_B_integers_l587_587610


namespace gain_percentage_is_correct_l587_587842

-- Define the cost price and selling price of one pen.
variables (C S : ℝ)

-- Define the given condition: the cost price of 20 pens is equal to the selling price of 12 pens.
def condition : Prop :=
  20 * C = 12 * S

-- Define the gain percentage formula.
def gain_percentage (C S : ℝ) : ℝ :=
  ((S - C) / C) * 100

-- Prove that given the condition, the gain percentage is 66.67%.
theorem gain_percentage_is_correct (C S : ℝ) (h : condition C S) : gain_percentage C S = 66.67 :=
by
  -- Assume 20C = 12S
  have h1 : 20 * C = 12 * S := h
  -- Solve for S in terms of C
  have h2 : S = (5 * C) / 3 := by sorry
  -- Substitute S in the gain percentage formula and simplify
  have h3 : gain_percentage C ((5 * C) / 3) = 66.67 := by
    calc
      gain_percentage C ((5 * C) / 3)
        = (((5 * C / 3) - C) / C) * 100 : by sorry
    ... = (2/3) * 100 : by sorry
    ... = 66.67 : by norm_num
  exact h3

end gain_percentage_is_correct_l587_587842


namespace y_coordinate_of_C_l587_587387

theorem y_coordinate_of_C (h : ℝ) (H : ∀ (C : ℝ), C = h) :
  let A := (0, 0)
      B := (0, 5)
      C := (3, h)
      D := (6, 5)
      E := (6, 0)
  -- Assuming the area of the pentagon is 50
  let area_square_ABDE := 25
      area_triangle_BCD := 25
  -- Assuming the height of triangle BCD
  let height_triangle_BCD := h - 5
      base_triangle_BCD := 6
      area_BCD := (1/2) * base_triangle_BCD * height_triangle_BCD in
  area_square_ABDE + area_triangle_BCD = 50 →
  area_BCD = area_triangle_BCD →
  h = 40 / 3 :=
by intros h H A B C D E area_square_ABDE area_triangle_BCD height_triangle_BCD base_triangle_BCD area_BCD;
   sorry

end y_coordinate_of_C_l587_587387


namespace no_valid_base_l587_587914

theorem no_valid_base (b : ℤ) (n : ℤ) : b^2 + 2*b + 2 ≠ n^2 := by
  sorry

end no_valid_base_l587_587914


namespace investment_at_6_percent_l587_587145

variables (x y : ℝ)

-- Conditions from the problem
def total_investment : Prop := x + y = 15000
def total_interest : Prop := 0.06 * x + 0.075 * y = 1023

-- Conclusion to prove
def invest_6_percent (x : ℝ) : Prop := x = 6800

theorem investment_at_6_percent (h1 : total_investment x y) (h2 : total_interest x y) : invest_6_percent x :=
by
  sorry

end investment_at_6_percent_l587_587145


namespace eq_p0_l587_587374

theorem eq_p0 : ∃ (p : Polynomial ℝ), degree p = 7 ∧
  (∀ n, n ∈ Finset.range 8 → p.eval (3 ^ n) = 1 / (3 ^ n)) ∧
  p.eval 0 = 19682 / 6561 :=
begin
  sorry,
end

end eq_p0_l587_587374


namespace cubic_equation_roots_sum_log_l587_587193

theorem cubic_equation_roots_sum_log (a b : ℝ) (p q r : ℝ) :
  27 * p^3 + 7 * a * p^2 + 6 * b * p + 3 * a = 0 ∧
  27 * q^3 + 7 * a * q^2 + 6 * b * q + 3 * a = 0 ∧
  27 * r^3 + 7 * a * r^2 + 6 * b * r + 3 * a = 0 ∧
  p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
  p > 0 ∧ q > 0 ∧ r > 0 ∧
  log 3 p + log 3 q + log 3 r = 5 →
  a = -2187 :=
by
  sorry

end cubic_equation_roots_sum_log_l587_587193


namespace edge_length_of_square_base_l587_587038

theorem edge_length_of_square_base 
  (r h : ℝ)
  (hemisphere_on_pyramid_base : r = 3)
  (pyramid_height : h = 4) 
  (hemisphere_tangent_to_faces: ∀ (face : ℕ), face ∈ {1, 2, 3, 4}) 
  : ∃ s : ℝ, s = sqrt 14 :=
by
  sorry

end edge_length_of_square_base_l587_587038


namespace avg_age_all_l587_587040

-- Define the conditions
def avg_age_seventh_graders (n₁ : Nat) (a₁ : Nat) : Prop :=
  n₁ = 40 ∧ a₁ = 13

def avg_age_parents (n₂ : Nat) (a₂ : Nat) : Prop :=
  n₂ = 50 ∧ a₂ = 40

-- Define the problem to prove
def avg_age_combined (n₁ n₂ a₁ a₂ : Nat) : Prop :=
  (n₁ * a₁ + n₂ * a₂) / (n₁ + n₂) = 28

-- The main theorem
theorem avg_age_all (n₁ n₂ a₁ a₂ : Nat):
  avg_age_seventh_graders n₁ a₁ → avg_age_parents n₂ a₂ → avg_age_combined n₁ n₂ a₁ a₂ :=
by 
  intros h1 h2
  sorry

end avg_age_all_l587_587040


namespace find_ice_cream_cost_l587_587163

def cost_of_ice_cream (total_paid cost_chapati cost_rice cost_vegetable : ℕ) (n_chapatis n_rice n_vegetables n_ice_cream : ℕ) : ℕ :=
  (total_paid - (n_chapatis * cost_chapati + n_rice * cost_rice + n_vegetables * cost_vegetable)) / n_ice_cream

theorem find_ice_cream_cost :
  let total_paid := 1051
  let cost_chapati := 6
  let cost_rice := 45
  let cost_vegetable := 70
  let n_chapatis := 16
  let n_rice := 5
  let n_vegetables := 7
  let n_ice_cream := 6
  cost_of_ice_cream total_paid cost_chapati cost_rice cost_vegetable n_chapatis n_rice n_vegetables n_ice_cream = 40 :=
by
  sorry

end find_ice_cream_cost_l587_587163


namespace least_4_changes_distinct_sums_l587_587547

def original_matrix : matrix (fin 4) (fin 4) ℕ :=
  ![
    ![1, 2, 3, 4],
    ![2, 3, 4, 1],
    ![3, 4, 1, 2],
    ![4, 1, 2, 3]
  ]

theorem least_4_changes_distinct_sums (A : matrix (fin 4) (fin 4) ℕ) :
  (A = original_matrix) → 
  ∃ k ≤ 4, ∀ B, B ≠ A → (∃ u v, row_sum B u ≠ row_sum B v ∧ col_sum B u ≠ col_sum B v) :=
sorry

end least_4_changes_distinct_sums_l587_587547


namespace x4_minus_x1_is_correct_l587_587952

noncomputable def x4_minus_x1 (f g : ℝ → ℝ) (x1 x2 x3 x4 : ℝ) : ℝ :=
  if h₁ : quadratic f ∧ quadratic g ∧ g = λ x, f (60 - x) ∧
            graph_contains_vertex g (vertex_of f) ∧
            increasing_order [x1, x2, x3, x4] ∧ (x3 - x2 = 90) then
      90 + 90 * real.sqrt 2
  else
    0 -- arbitrary value if conditions aren't met

-- Statement that should be proved
theorem x4_minus_x1_is_correct (f g : ℝ → ℝ) (x1 x2 x3 x4 : ℝ) 
  (H : quadratic f) (H2 : quadratic g) (H3 : g = λ x, f (60 - x))
  (H4 : graph_contains_vertex g (vertex_of f)) (H5 : increasing_order [x1, x2, x3, x4])
  (H6 : x3 - x2 = 90) : 
  x4_minus_x1 f g x1 x2 x3 x4 = 90 + 90 * real.sqrt 2 :=
by sorry

end x4_minus_x1_is_correct_l587_587952


namespace proof_f_ab_l587_587260

def f (x : ℝ) (a : ℝ) : ℝ := 2^x + a / 2^x
def g (x : ℝ) (b : ℝ) : ℝ := b * x - Real.log2 (4^x + 1)

theorem proof_f_ab (a b : ℝ) (h1 : ∀ x : ℝ, f (-x) a + f x a = 0)
  (h2 : ∀ x : ℝ, g x b = g (-x) b)
  (h_a : a = -1) (h_b : b = 1) : f (a * b) a = -3/2 :=
by 
  sorry

end proof_f_ab_l587_587260


namespace tan_ratio_l587_587263

theorem tan_ratio (α β : ℝ) (h : (sin (α + β)) / (sin (α - β)) = 3) : (tan α) / (tan β) = 2 :=
sorry

end tan_ratio_l587_587263


namespace orthocenter_ABC_l587_587997

structure Point2D :=
  (x : ℝ)
  (y : ℝ)

def A : Point2D := ⟨5, -1⟩
def B : Point2D := ⟨4, -8⟩
def C : Point2D := ⟨-4, -4⟩

def isOrthocenter (H : Point2D) (A B C : Point2D) : Prop := sorry  -- Define this properly according to the geometric properties in actual formalization.

theorem orthocenter_ABC : ∃ H : Point2D, isOrthocenter H A B C ∧ H = ⟨3, -5⟩ := 
by 
  sorry  -- Proof omitted

end orthocenter_ABC_l587_587997


namespace num_allocation_schemes_l587_587422

theorem num_allocation_schemes : 
  let C(n k : ℕ) := n.choose k
  let A(n k : ℕ) := nat.factorial n / nat.factorial (n - k)
  in (C 6 2 * C 4 2 * C 2 1 * C 1 1) / (A 2 2 * A 2 2) * A 4 4 = 1080 :=
by
  let C := λ (n k : ℕ), n.choose k
  let A := λ (n k : ℕ), nat.factorial n / nat.factorial (n - k)
  have h1 : C 6 2 = 15 := by sorry
  have h2 : C 4 2 = 6 := by sorry
  have h3 : C 2 1 = 2 := by sorry
  have h4 : C 1 1 = 1 := by sorry
  have h5 : A 2 2 = 2 := by sorry
  have h6 : A 4 4 = 24 := by sorry
  calc (C 6 2 * C 4 2 * C 2 1 * C 1 1) / (A 2 2 * A 2 2) * A 4 4 = (15 * 6 * 2 * 1) / (2 * 2) * 24 : by sorry
  ... = (180) / (4) * 24 : by sorry
  ... = 45 * 24 : by sorry
  ... = 1080 : by sorry

end num_allocation_schemes_l587_587422


namespace max_cookie_price_l587_587352

theorem max_cookie_price (k p : ℕ) :
  8 * k + 3 * p < 200 →
  4 * k + 5 * p > 150 →
  k ≤ 19 :=
sorry

end max_cookie_price_l587_587352


namespace condition_B_necessary_but_not_sufficient_l587_587376

theorem condition_B_necessary_but_not_sufficient : 
  ∀ x, 0 < x ∧ x < 5 → (|x - 2| < 3 ∧ ¬(|x - 2| < 3 → 0 < x ∧ x < 5)) :=
by
  intro x
  sorry

end condition_B_necessary_but_not_sufficient_l587_587376


namespace imaginary_part_of_z_l587_587315

theorem imaginary_part_of_z (z : ℂ) (h : z + 3 - 4 * complex.i = 1) : complex.im z = 4 :=
sorry

end imaginary_part_of_z_l587_587315


namespace max_value_of_f_l587_587365

noncomputable def f (x a : ℝ) : ℝ := - (1/3) * x ^ 3 + (1/2) * x ^ 2 + 2 * a * x

theorem max_value_of_f (a : ℝ) (h0 : 0 < a) (h1 : a < 2)
  (h2 : ∀ x, 1 ≤ x → x ≤ 4 → f x a ≥ f 4 a)
  (h3 : f 4 a = -16 / 3) :
  f 2 a = 10 / 3 :=
sorry

end max_value_of_f_l587_587365


namespace probability_correct_pass_through_C_D_l587_587333

noncomputable def probability_pass_through_C_D 
  (P_E: ℝ) (P_S: ℝ) (ways_A_to_C: ℕ) (ways_C_to_D: ℕ) (ways_D_to_B: ℕ) 
  (total_ways_A_to_B: ℕ) : ℝ :=
  let path_probability := P_E ^ 5 * P_S ^ 4 
      passing_path_count := ways_A_to_C * ways_C_to_D * ways_D_to_B 
      total_path_count := total_ways_A_to_B in
  (passing_path_count / total_path_count) * path_probability

theorem probability_correct_pass_through_C_D : 
  probability_pass_through_C_D (2/3) (1/3) 4 2 3 126 = 8 / 1890 :=
by
  sorry

end probability_correct_pass_through_C_D_l587_587333


namespace find_bucket1_capacity_l587_587497

-- Conditions defined with variables and specific constraints
variables (x : ℝ)
variables (total_volume : ℝ)
variables (bucket1_count : ℝ) (bucket2_count : ℝ)
variables (bucket2_capacity : ℝ)

-- Assigning known values to the variables
def bucket1_count := 12
def bucket2_count := 108
def bucket2_capacity := 9

-- Defining the total volume equations
def total_volume_case1 := bucket1_count * x
def total_volume_case2 := bucket2_count * bucket2_capacity

-- Theorem to prove x = 81 given the conditions
theorem find_bucket1_capacity (h : total_volume_case1 = total_volume_case2) : x = 81 :=
by {
  -- Place the proof steps here
  sorry
}

end find_bucket1_capacity_l587_587497


namespace minimum_value_of_MN_is_5_over_4_l587_587419

noncomputable def minimum_value_of_MN : ℝ :=
  let a := λ p : ℝ × ℝ, p.2
  let M_x1 := λ a : ℝ, (a - 4) / 4
  let N_x2 := λ a : ℝ, {
    x : ℝ // a = 3*x + log x
  }
  let MN := λ a : ℝ, {
    p : N_x2 a // a = 4 * M_x1 a + 4
  } in
  infi (λ p : N_x2 p.2, abs (p.1 - (M_x1 p.2)))

theorem minimum_value_of_MN_is_5_over_4 : minimum_value_of_MN = 5 / 4 :=
sorry

end minimum_value_of_MN_is_5_over_4_l587_587419


namespace intersection_complement_R_M_and_N_l587_587064

open Set

def universalSet := ℝ
def M := {x : ℝ | -2 ≤ x ∧ x ≤ 2}
def complementR (S : Set ℝ) := {x : ℝ | x ∉ S}
def N := {x : ℝ | x < 1}

theorem intersection_complement_R_M_and_N:
  (complementR M ∩ N) = {x : ℝ | x < -2} := by
  sorry

end intersection_complement_R_M_and_N_l587_587064


namespace minimum_distance_from_P_to_tangent_line_l587_587418

-- Define the circle x^2 + y^2 = 4
def circle1 (x y : ℝ) : Prop :=
  x^2 + y^2 = 4

-- Define the tangent line to the circle1 at the point (-1, sqrt(3))
def tangent_line (x y : ℝ) : Prop :=
  x - sqrt 3 * y + 4 = 0

-- Define the moving circle x^2 - 4x + y^2 + 3 = 0
def moving_circle (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 + 3 = 0

-- Define the center of the moving circle
def moving_circle_center : ℝ × ℝ :=
  (2, 0)

-- Define the minimum distance we need to prove
def minimum_distance (d : ℝ) : Prop :=
  d = 2

-- The theorem statement
theorem minimum_distance_from_P_to_tangent_line :
  ∃ (P : ℝ × ℝ), moving_circle P.1 P.2 ∧ minimum_distance 
  (abs ((moving_circle_center.1 - P.1) + (moving_circle_center.2 - P.2)) / sqrt (1 + 3)) :=
sorry

end minimum_distance_from_P_to_tangent_line_l587_587418


namespace point_reflection_l587_587672

-- Definition of point and reflection over x-axis
def P : ℝ × ℝ := (-2, 3)

def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Statement to prove
theorem point_reflection : reflect_x_axis P = (-2, -3) :=
by
  -- Proof goes here
  sorry

end point_reflection_l587_587672


namespace part1_intersection_part2_range_l587_587637

noncomputable def A := {x : ℝ | (1/32) ≤ 2^(-x) ∧ 2^(-x) ≤ 4}
noncomputable def B (m : ℝ) := {x : ℝ | x^2 + 2*m*x < 3*m^2}

theorem part1_intersection (x : ℝ) : (A x) ∧ (B 2 x) ↔ -2 ≤ x ∧ x < 2 :=
by sorry

theorem part2_range (m : ℝ) (Hm : 0 < m) : (∀ x, (A x) → (B m x)) ↔ m ≤ 2/3 :=
by sorry

end part1_intersection_part2_range_l587_587637


namespace volume_of_tetrahedron_l587_587925

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end volume_of_tetrahedron_l587_587925


namespace pentagon_area_50_l587_587395

def point := (ℝ × ℝ)

structure Pentagon :=
(A B C D E : point)

def area_rectangle (p1 p2 p3 p4 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
let ⟨x4, y4⟩ := p4 in
abs((x3 - x1) * (y2 - y1))

def area_triangle (p1 p2 p3 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

def y_coordinate_C (pent: Pentagon) : ℝ :=
let ⟨_, _, pC, _, _⟩ := pent in
pC.2

theorem pentagon_area_50 (h : ℝ) :
  let A := (0, 0) in
  let B := (0, 5) in
  let C := (3, h) in
  let D := (6, 5) in
  let E := (6, 0) in
  let rect_area := area_rectangle A B D E in
  let tri_area := area_triangle B C D in
  rect_area + tri_area = 50 :=
by
  sorry

end pentagon_area_50_l587_587395


namespace circle_area_from_circumference_l587_587044

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l587_587044


namespace average_speed_is_40_l587_587745

-- Define the total distance
def total_distance : ℝ := 640

-- Define the distance for the first half
def first_half_distance : ℝ := total_distance / 2

-- Define the average speed for the first half
def first_half_speed : ℝ := 80

-- Define the time taken for the first half
def first_half_time : ℝ := first_half_distance / first_half_speed

-- Define the multiplicative factor for time increase in the second half
def time_increase_factor : ℝ := 3

-- Define the time taken for the second half
def second_half_time : ℝ := first_half_time * time_increase_factor

-- Define the total time for the trip
def total_time : ℝ := first_half_time + second_half_time

-- Define the calculated average speed for the entire trip
def calculated_average_speed : ℝ := total_distance / total_time

-- State the theorem that the average speed for the entire trip is 40 miles per hour
theorem average_speed_is_40 : calculated_average_speed = 40 :=
by
  sorry

end average_speed_is_40_l587_587745


namespace solve_equation_l587_587216

theorem solve_equation :
  {x : ℝ | (15 * x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 54 } = {12, -3, -3 + real.sqrt 33, -3 - real.sqrt 33} :=
by
  sorry

end solve_equation_l587_587216


namespace computers_built_per_month_l587_587859

theorem computers_built_per_month (days_in_month : ℕ) (hours_per_day : ℕ) (computers_per_interval : ℚ) (intervals_per_hour : ℕ)
    (h_days : days_in_month = 28) (h_hours : hours_per_day = 24) (h_computers : computers_per_interval = 2.25) (h_intervals : intervals_per_hour = 2) :
    days_in_month * hours_per_day * intervals_per_hour * computers_per_interval = 3024 :=
by
  -- We would give the proof here, but it's omitted as per instructions.
  sorry

end computers_built_per_month_l587_587859


namespace determine_d_value_l587_587918

noncomputable def Q (d : ℚ) (x : ℚ) : ℚ := x^3 + 3 * x^2 + d * x + 8

theorem determine_d_value (d : ℚ) : x - 3 ∣ Q d x → d = -62 / 3 := by
  sorry

end determine_d_value_l587_587918


namespace distinct_ints_sum_to_4r_l587_587699

theorem distinct_ints_sum_to_4r 
  (a b c d r : ℤ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (h_root : (r - a) * (r - b) * (r - c) * (r - d) = 4) : 
  4 * r = a + b + c + d := 
by sorry

end distinct_ints_sum_to_4r_l587_587699


namespace point_in_third_quadrant_coordinates_l587_587009

theorem point_in_third_quadrant_coordinates :
  ∀ (P : ℝ × ℝ), (P.1 < 0) ∧ (P.2 < 0) ∧ (|P.2| = 2) ∧ (|P.1| = 3) -> P = (-3, -2) :=
by
  intros P h
  sorry

end point_in_third_quadrant_coordinates_l587_587009


namespace fraction_addition_l587_587096

variable (d : ℝ)

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := 
sorry

end fraction_addition_l587_587096


namespace age_of_new_teacher_l587_587769

theorem age_of_new_teacher (sum_of_20_teachers : ℕ)
  (avg_age_20_teachers : ℕ)
  (total_teachers_after_new_teacher : ℕ)
  (new_avg_age_after_new_teacher : ℕ)
  (h1 : sum_of_20_teachers = 20 * 49)
  (h2 : avg_age_20_teachers = 49)
  (h3 : total_teachers_after_new_teacher = 21)
  (h4 : new_avg_age_after_new_teacher = 48) :
  ∃ (x : ℕ), x = 28 :=
by
  sorry

end age_of_new_teacher_l587_587769


namespace factorization_of_polynomial_l587_587428

theorem factorization_of_polynomial (x : ℝ) : 2 * x^2 - 12 * x + 18 = 2 * (x - 3)^2 := by
  sorry

end factorization_of_polynomial_l587_587428


namespace gcd_ab_l587_587817

def a : ℕ := 130^2 + 215^2 + 310^2
def b : ℕ := 131^2 + 216^2 + 309^2

theorem gcd_ab : Nat.gcd a b = 1 := by
  sorry

end gcd_ab_l587_587817


namespace first_player_wins_l587_587112

noncomputable def initial_clock_position : ℕ := 12
noncomputable def first_player_moves : List (ℕ → ℕ) :=
  [λ p => (p + 2) % 12, λ p => (p + 3) % 12, λ p => if (p = 11) then 1 else (p + 2) % 12, λ p => if (p = 1) then (p + 3) % 12 else (p + 2) % 12]
noncomputable def second_player_moves : List (ℕ → ℕ) :=
  [λ p => 5, λ p => if (p = 8) then 11 else if (p = 1) then (p + 2) % 12 else (p + 3) % 12]

theorem first_player_wins :
  ∀ pos: ℕ, (pos = initial_clock_position) →
  let pos1 := (first_player_moves.head) pos
  let pos2 := (second_player_moves.head) pos1
  let pos3 := (first_player_moves.tail.head) pos2
  let pos4 := (second_player_moves.tail.head) pos3
  let pos5 := (first_player_moves.tail.tail.head) pos4
  (pos5 ≤ 12) → (pos6: ℕ) → (pos6 = 6) :=
by
  intro pos hpos pos1 pos2 pos3 pos4 pos5 hcond
  sorry

end first_player_wins_l587_587112


namespace no_perfect_square_in_sequence_l587_587593

noncomputable def x : ℕ → ℤ
| 0     := 1
| 1     := 3
| (n+2) := 6 * x (n + 1) - x n

theorem no_perfect_square_in_sequence : ¬ ∃ (n : ℕ), ∃ (k : ℤ), x n = k^2 :=
by sorry

end no_perfect_square_in_sequence_l587_587593


namespace find_positive_n_l587_587574

theorem find_positive_n (n x : ℝ) (h : 16 * x ^ 2 + n * x + 4 = 0) : n = 16 :=
by
  sorry

end find_positive_n_l587_587574


namespace moment_of_inertia_rectangle_l587_587176

variables (ρ a b : ℝ)

-- Moment of inertia of a rectangle relative to a vertex
def I_vertex_rect := ρ * (a * b) / 3 * (a^2 + b^2)

-- Moment of inertia of a rectangle relative to the centroid (intersection of diagonals)
def I_centroid_rect := ρ * (a * b) / 12 * (a^2 + b^2)

theorem moment_of_inertia_rectangle (ρ a b : ℝ) :
  I_vertex_rect ρ a b = ρ * (a * b) / 3 * (a^2 + b^2) ∧
  I_centroid_rect ρ a b = ρ * (a * b) / 12 * (a^2 + b^2) :=
by sorry

end moment_of_inertia_rectangle_l587_587176


namespace postage_fee_420g_l587_587777

-- Definitions based on the conditions
def initial_cost : ℝ := 0.7
def additional_cost : ℝ := 0.4
def weight : ℝ := 420

-- Calculation based on the conditions
def rounding_up_fraction (x : ℝ) : ℝ := x.ceil

noncomputable def total_cost (w : ℝ) : ℝ :=
initial_cost + rounding_up_fraction ((w - 100) / 100) * additional_cost

-- The proof problem statement
theorem postage_fee_420g : total_cost weight = 2.3 :=
by
  sorry

end postage_fee_420g_l587_587777


namespace divisible_expressions_l587_587013

theorem divisible_expressions (x y : ℤ) (n : ℤ) 
  (h : 2 * x + 3 * y = 17 * n) : 17 ∣ (9 * x + 5 * y) :=
begin
  -- proof goes here
  sorry
end

end divisible_expressions_l587_587013


namespace silvia_trip_shorter_l587_587690

theorem silvia_trip_shorter (j s : ℝ) (jerry_distance : j = 2) (silvia_distance : s = Real.sqrt 2) : 
  ((j - s) / j) * 100 ≈ 30 := by
  sorry

end silvia_trip_shorter_l587_587690


namespace range_of_m_l587_587635

theorem range_of_m (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) →
  (1 < m) :=
by
  sorry

end range_of_m_l587_587635


namespace magnitude_a_add_b_l587_587972

noncomputable def vector_magnitude (v : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt (v.1^2 + v.2^2 + v.3^2)

variables (a b : ℝ × ℝ × ℝ)

axiom norm_a : vector_magnitude a = 1
axiom norm_b : vector_magnitude b = 1
axiom norm_a_sub_b : vector_magnitude (a - b) = 1

theorem magnitude_a_add_b (a b : ℝ × ℝ × ℝ)
  (h1 : vector_magnitude a = 1)
  (h2 : vector_magnitude b = 1)
  (h3 : vector_magnitude (a - b) = 1) :
  vector_magnitude (a + b) = real.sqrt 3 :=
sorry

end magnitude_a_add_b_l587_587972


namespace production_days_l587_587950

theorem production_days (n : ℕ) (P : ℕ)
  (h1 : P = 40 * n)
  (h2 : (P + 90) / (n + 1) = 45) :
  n = 9 :=
by
  sorry

end production_days_l587_587950


namespace probability_A_more_than_BC_l587_587203

-- Definitions for the conditions
def team := Type
def play_game (t1 t2 : team) : Prop := sorry  -- Each team plays each other exactly once
def no_ties (t1 t2 : team) : Prop := sorry  -- Each game has a winner and loser
def chance_of_winning : ℕ → Prop := sorry  -- Each team has a 50% chance of winning any game
def independent_outcomes : Prop := sorry  -- The outcomes of the games are independent
def points (t : team) : ℕ := sorry  -- Each winner earns a point, each loser gets 0 points

-- Specific games mentioned
def A B C : team := sorry
def first_game : play_game A B := sorry  -- In the first game, Team A beats Team B
def second_game : play_game A C := sorry  -- In the second game, Team A beats Team C

theorem probability_A_more_than_BC : team → ℕ :=
  assume A,
  assume B,
  assume C,
  -- Given conditions
  play_game A B →
  play_game A C →
  no_ties A B →
  no_ties A C →
  chance_of_winning 50 →
  independent_outcomes →
  points A > points B + points C →
  -- Expected answer
  625 + 1024 = 1649 :=
sorry

end probability_A_more_than_BC_l587_587203


namespace arithmetic_sequence_sum_nine_l587_587264

variable {a : ℕ → ℤ} -- Define a_n sequence as a function from ℕ to ℤ

-- Define the conditions
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ (d : ℤ), ∀ n m, a (n + m) = a n + m * d

def fifth_term_is_two (a : ℕ → ℤ) : Prop :=
  a 5 = 2

-- Lean statement to prove the sum of the first 9 terms
theorem arithmetic_sequence_sum_nine (a : ℕ → ℤ)
  (h1 : is_arithmetic_sequence a)
  (h2 : fifth_term_is_two a) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9 :=
sorry

end arithmetic_sequence_sum_nine_l587_587264


namespace required_flying_hours_l587_587762

theorem required_flying_hours (day_flying : ℕ) (night_flying : ℕ) (cross_country_flying : ℕ) 
  (hours_per_month : ℕ) (months : ℕ) (total_hours_required : ℕ) : 
  day_flying = 50 → 
  night_flying = 9 → 
  cross_country_flying = 121 → 
  hours_per_month = 220 → 
  months = 6 → 
  total_hours_required = (hours_per_month * months) - (day_flying + night_flying + cross_country_flying) → 
  total_hours_required = 1140 :=
by {
  intro h1,
  intro h2,
  intro h3,
  intro h4,
  intro h5,
  intro h6,
  rw [h1, h2, h3, h4, h5] at h6,
  exact h6
}

end required_flying_hours_l587_587762


namespace find_line_eqn_line_equation_l587_587138

-- Define the necessary structures
structure Point :=
  (x : ℝ)
  (y : ℝ)

structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define the midpoint condition
def midpoint (A B M : Point) : Prop :=
  M.x = (A.x + B.x) / 2 ∧ M.y = (A.y + B.y) / 2

-- Define the ellipse equation condition
def on_ellipse (P : Point) : Prop :=
  (P.x^2 / 25) + (P.y^2 / 16) = 1

-- Define a point lying on a line
def on_line (P : Point) (l : Line) : Prop :=
  P.y = l.slope * P.x + l.intercept

-- Theorem statement
theorem find_line_eqn (M A B : Point) (l : Line)
  (hM : M.x = 1 ∧ M.y = 2)
  (h_mid : midpoint A B M)
  (hA_ellipse : on_ellipse A)
  (hB_ellipse : on_ellipse B)
  (h_line : ∀ P, on_line P l ↔ P = A ∨ P = B ∨ P = M):
  l.slope = -8/25 ∧ l.intercept = 58/25 := 
sorry

-- Define the line equation
def line_eqn (l : Line) : Prop :=
  ∀ (x y : ℝ), y = l.slope * x + l.intercept ↔ 8 * x + 25 * y = 58

-- Final problem statement
theorem line_equation (M A B : Point) (l : Line)
  (hM : M.x = 1 ∧ M.y = 2)
  (h_mid : midpoint A B M)
  (hA_ellipse : on_ellipse A)
  (hB_ellipse : on_ellipse B)
  (h_line_eqn : l.slope = -8/25 ∧ l.intercept = 58/25):
  line_eqn l :=
sorry

end find_line_eqn_line_equation_l587_587138


namespace divisible_expressions_l587_587014

theorem divisible_expressions (x y : ℤ) (n : ℤ) 
  (h : 2 * x + 3 * y = 17 * n) : 17 ∣ (9 * x + 5 * y) :=
begin
  -- proof goes here
  sorry
end

end divisible_expressions_l587_587014


namespace mike_avg_speed_l587_587742

/-
  Given conditions:
  * total distance d = 640 miles
  * half distance h = 320 miles
  * first half average rate r1 = 80 mph
  * time for first half t1 = h / r1 = 4 hours
  * second half time t2 = 3 * t1 = 12 hours
  * total time tt = t1 + t2 = 16 hours
  * total distance d = 640 miles
  * average rate for entire trip should be (d/tt) = 40 mph.
  
  The goal is to prove that the average rate for the entire trip is 40 mph.
-/
theorem mike_avg_speed:
  let d := 640 in
  let h := 320 in
  let r1 := 80 in
  let t1 := h / r1 in
  let t2 := 3 * t1 in
  let tt := t1 + t2 in
  let avg_rate := d / tt in
  avg_rate = 40 := by
  sorry

end mike_avg_speed_l587_587742


namespace bailey_rawhide_bones_l587_587895

variable (dog_treats : ℕ) (chew_toys : ℕ) (total_items : ℕ)
variable (credit_cards : ℕ) (items_per_card : ℕ)

theorem bailey_rawhide_bones :
  (dog_treats = 8) →
  (chew_toys = 2) →
  (credit_cards = 4) →
  (items_per_card = 5) →
  (total_items = credit_cards * items_per_card) →
  (total_items - (dog_treats + chew_toys) = 10) :=
by
  intros
  sorry

end bailey_rawhide_bones_l587_587895


namespace jessica_routes_count_l587_587182

def line := Type

def valid_route_count (p q r s t u : line) : ℕ := 9 + 36 + 36

theorem jessica_routes_count (p q r s t u : line) :
  valid_route_count p q r s t u = 81 :=
by
  sorry

end jessica_routes_count_l587_587182


namespace roots_quadratic_l587_587706

theorem roots_quadratic (a b : ℝ) (h₁ : a + b = 6) (h₂ : a * b = 8) :
  a^2 + a^5 * b^3 + a^3 * b^5 + b^2 = 10260 :=
by
  sorry

end roots_quadratic_l587_587706


namespace finite_permutations_l587_587398

def areConsecutive (n : ℕ) (l : List ℤ) := (l.length = n ∧ ∀ i < n, l[(i + 1) % n] = l[(i + 2) % n])

noncomputable def canSwap (a b c d : ℤ) := ((a - d) * (b - c) < 0)

theorem finite_permutations (n : ℕ) (l : List ℤ) :
  (areConsecutive n l) →
  (∀ a b c d, (a, b, c, d) ∈ l) →
  (∀ i < n, canSwap (l[i]) (l[(i + 1) % n]) (l[(i + 2) % n]) (l[(i + 3) % n])) →
  ∃ k, k < n ∧ ∀ m > k, ¬ (canSwap (l[m]) (l[(m + 1) % n]) (l[(m + 2) % n]) (l[(m + 3) % n])) :=
sorry

end finite_permutations_l587_587398


namespace number_of_seats_l587_587877

noncomputable theory

-- Define the given constants and conditions
def fill_percentage : ℝ := 0.80
def ticket_price : ℝ := 30
def performances : ℕ := 3
def total_revenue : ℝ := 28800

-- Define the goal statement to prove that the number of seats is 400
theorem number_of_seats (S : ℝ) (h1 : fill_percentage * S * ticket_price * (performances : ℝ) = total_revenue) : S = 400 :=
by
  sorry

end number_of_seats_l587_587877


namespace rectangle_length_l587_587490

theorem rectangle_length (b l : ℝ) 
  (h1 : l = 2 * b)
  (h2 : (l - 5) * (b + 5) = l * b + 75) : l = 40 := by
  sorry

end rectangle_length_l587_587490


namespace nicholas_bottle_caps_l587_587002

theorem nicholas_bottle_caps (initial : ℕ) (additional : ℕ) (final : ℕ) (h1 : initial = 8) (h2 : additional = 85) :
  final = 93 :=
by
  sorry

end nicholas_bottle_caps_l587_587002


namespace median_of_list_l587_587088

def is_median (l : List ℕ) (m : ℕ) : Prop :=
  let l_sorted := l.sort
  l.length % 2 = 0 ∧ m = (l_sorted[l.length / 2 - 1] + l_sorted[l.length / 2]) / 2

theorem median_of_list :
  is_median (List.range 3030 ++ List.map (λ n, n^3) (List.range 3030)) 30305 :=
  sorry

end median_of_list_l587_587088


namespace ratio_of_terms_l587_587062

theorem ratio_of_terms (a_n b_n : ℕ → ℕ) (S_n T_n : ℕ → ℕ) :
  (∀ n, S_n n = (n * (2 * a_n n - (n - 1))) / 2) → 
  (∀ n, T_n n = (n * (2 * b_n n - (n - 1))) / 2) → 
  (∀ n, S_n n / T_n n = (n + 3) / (2 * n + 1)) → 
  S_n 6 / T_n 6 = 14 / 23 :=
by
  sorry

end ratio_of_terms_l587_587062


namespace average_first_two_l587_587771

theorem average_first_two (a b c d e f : ℝ)
  (h1 : (a + b + c + d + e + f) = 16.8)
  (h2 : (c + d) = 4.6)
  (h3 : (e + f) = 7.4) : 
  (a + b) / 2 = 2.4 :=
by
  sorry

end average_first_two_l587_587771


namespace no_such_function_exists_l587_587714

theorem no_such_function_exists :
  ¬ ∃ (f : ℝ → ℝ), (∃ M > 0, ∀ x : ℝ, -M ≤ f x ∧ f x ≤ M) ∧
                    (f 1 = 1) ∧
                    (∀ x : ℝ, x ≠ 0 → f (x + 1 / x^2) = f x + (f (1 / x))^2) :=
by
  sorry

end no_such_function_exists_l587_587714


namespace light_bulb_switch_dependency_l587_587853

universe u

variable {K : Type u} [Fintype K] [DecidableEq K]

/-- Represents the state of the switches and light bulbs, where the state is a boolean (on/off) -/
structure State :=
  (switch_state : K → Bool)
  (light_state  : K → Bool)

namespace BlackBox

/-- Hypothesis: flipping a switch exactly changes the state of one light bulb -/
def toggle_switch (s : K) (state : State) : State :=
  {switch_state := fun x => if x = s then !state.switch_state x else state.switch_state x,
   light_state  := fun x => if x = s then !state.light_state x else state.light_state x}

/-- Hypothesis: The state of the display uniquely determines the state of the control panel -/
axiom unique_state_determination : ∀ s1 s2 : State, s1.light_state = s2.light_state → s1.switch_state = s2.switch_state

theorem light_bulb_switch_dependency :
  ∀ (i : K), ∃ (j : K), (∀ state : State, state.light_state i = !toggle_switch j state.light_state i) :=
by
  sorry

end BlackBox

end light_bulb_switch_dependency_l587_587853


namespace circle_area_from_circumference_l587_587043

theorem circle_area_from_circumference (C : ℝ) (A : ℝ) (hC : C = 36) (hCircumference : ∀ r, C = 2 * Real.pi * r) (hAreaFormula : ∀ r, A = Real.pi * r^2) :
  A = 324 / Real.pi :=
by
  sorry

end circle_area_from_circumference_l587_587043


namespace strategy2_is_better_final_cost_strategy2_correct_l587_587153

def initial_cost : ℝ := 12000

def strategy1_discount : ℝ := 
  let after_first_discount := initial_cost * 0.70
  let after_second_discount := after_first_discount * 0.85
  let after_third_discount := after_second_discount * 0.95
  after_third_discount

def strategy2_discount : ℝ := 
  let after_first_discount := initial_cost * 0.55
  let after_second_discount := after_first_discount * 0.90
  let after_third_discount := after_second_discount * 0.90
  let final_cost := after_third_discount + 150
  final_cost

theorem strategy2_is_better : strategy2_discount < strategy1_discount :=
by {
  sorry -- proof goes here
}

theorem final_cost_strategy2_correct : strategy2_discount = 5496 :=
by {
  sorry -- proof goes here
}

end strategy2_is_better_final_cost_strategy2_correct_l587_587153


namespace find_x_for_equation_l587_587458

theorem find_x_for_equation : ∃ x : ℝ, (1 / 2) + ((2 / 3) * x + 4) - (8 / 16) = 4.25 ↔ x = 0.375 := 
by
  sorry

end find_x_for_equation_l587_587458


namespace jean_initial_stuffies_l587_587689

variable (S : ℕ) (h1 : S * 2 / 3 / 4 = 10)

theorem jean_initial_stuffies : S = 60 :=
by
  sorry

end jean_initial_stuffies_l587_587689


namespace equilateral_if_P_inside_right_angled_if_P_outside_l587_587514

theorem equilateral_if_P_inside (A B C P : Point) 
  (h1 : inside_triangle P A B C) 
  (h2 : area_triangle P A B = area_triangle P B C ∧ area_triangle P B C = area_triangle P C A) 
  (h3 : perimeter_triangle P A B = perimeter_triangle P B C ∧ perimeter_triangle P B C = perimeter_triangle P C A) : 
  is_equilateral_triangle A B C :=
sorry

theorem right_angled_if_P_outside (A B C P : Point) 
  (h1 : ¬inside_triangle P A B C) 
  (h2 : area_triangle P A B = area_triangle P B C ∧ area_triangle P B C = area_triangle P C A) 
  (h3 : perimeter_triangle P A B = perimeter_triangle P B C ∧ perimeter_triangle P B C = perimeter_triangle P C A) : 
  is_right_angled_triangle A B C :=
sorry

end equilateral_if_P_inside_right_angled_if_P_outside_l587_587514


namespace minimum_boxes_cost_300_muffins_l587_587126

theorem minimum_boxes_cost_300_muffins :
  ∃ (L_used M_used S_used : ℕ), 
    L_used + M_used + S_used = 28 ∧ 
    (L_used = 10 ∧ M_used = 15 ∧ S_used = 3) ∧ 
    (L_used * 15 + M_used * 9 + S_used * 5 = 300) ∧ 
    (L_used * 5 + M_used * 3 + S_used * 2 = 101) ∧ 
    (L_used ≤ 10 ∧ M_used ≤ 15 ∧ S_used ≤ 25) :=
by
  -- The proof is omitted (theorem statement only).
  sorry

end minimum_boxes_cost_300_muffins_l587_587126


namespace circle_properties_l587_587623

noncomputable def circle_standard_equation (x y : ℝ) : Prop :=
  (x + 1)^2 + (y + 1)^2 = 1

theorem circle_properties
  (ρ θ : ℝ)
  (h : ρ^2 + 2 * real.sqrt 2 * ρ * real.sin (θ + π / 4) + 1 = 0) :
  (∃ x y : ℝ, circle_standard_equation x y) ∧ 
  (∀ P : ℝ × ℝ, P ∈ (λ α : ℝ, ( -1 + real.cos α, -1 + real.sin α )) → P.1 * P.2 ≤ (3 / 2 + real.sqrt 2)) :=
sorry

end circle_properties_l587_587623


namespace goods_train_speed_l587_587864

theorem goods_train_speed :
  ∀ (length_train length_platform time : ℝ),
    length_train = 250.0416 →
    length_platform = 270 →
    time = 26 →
    (length_train + length_platform) / time = 20 :=
by
  intros length_train length_platform time H_train H_platform H_time
  rw [H_train, H_platform, H_time]
  norm_num
  sorry

end goods_train_speed_l587_587864


namespace max_min_of_f_l587_587830

noncomputable def f (x : ℝ) : ℝ := 
  Real.sin (2 * Real.pi + x) + 
  Real.sqrt 3 * Real.cos (2 * Real.pi - x) -
  Real.sin (2013 * Real.pi + Real.pi / 6)

theorem max_min_of_f : 
  - (Real.pi / 2) ≤ x ∧ x ≤ Real.pi / 2 →
  (-1 / 2) ≤ f x ∧ f x ≤ 5 / 2 :=
sorry

end max_min_of_f_l587_587830


namespace xy_computation_l587_587448

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end xy_computation_l587_587448


namespace bugs_meet_again_at_point_P_l587_587078

theorem bugs_meet_again_at_point_P :
  let r1 := 3 -- radius of the smaller circle in inches
  let r2 := 7 -- radius of the larger circle in inches
  let v1 := 4 * Real.pi -- speed of the bug on the smaller circle in inches per minute
  let v2 := 6 * Real.pi -- speed of the bug on the larger circle in inches per minute
  let C1 := 2 * r1 * Real.pi -- circumference of the smaller circle
  let C2 := 2 * r2 * Real.pi -- circumference of the larger circle
  let t1 := C1 / v1 -- time for the bug on the smaller circle to complete one lap
  let t2 := C2 / v2 -- time for the bug on the larger circle to complete one lap
  ∃ t : ℝ, t > 0 ∧ (t / t1).isNat ∧ (t / t2).isNat ∧ t = 21 := sorry

end bugs_meet_again_at_point_P_l587_587078


namespace modulus_of_complex_number_l587_587493

section
variable (z : ℂ)
variable (i : ℂ := Complex.I)

-- Given Condition
local notation "given" := z = 4 * i / (1 - i)

-- Proof Problem
theorem modulus_of_complex_number : given → Complex.abs z = 2 * Real.sqrt 2 :=
by
  sorry
end

end modulus_of_complex_number_l587_587493


namespace maximize_volume_l587_587791

def volume (x : ℝ) : ℝ := x^2 * ((60 - x) / 2)

theorem maximize_volume : ∃ x ∈ set.Ioo 0 60, (∀ y ∈ set.Ioo 0 60, volume y ≤ volume x) ∧ x = 40 :=
by
  sorry

end maximize_volume_l587_587791


namespace triangle_XYZ_l587_587684

theorem triangle_XYZ (XZ : ℝ) (h_XZ : XZ = 18) (right_angle_Z : true) (angle_Y : ∠Y = 60) : XY = 36 :=
by sorry

end triangle_XYZ_l587_587684


namespace expression_divisibility_l587_587015

theorem expression_divisibility (x y : ℤ) (k_1 k_2 : ℤ) (h1 : 2 * x + 3 * y = 17 * k_1) :
    ∃ k_2 : ℤ, 9 * x + 5 * y = 17 * k_2 :=
by
  sorry

end expression_divisibility_l587_587015


namespace sticks_can_be_paired_l587_587801

noncomputable def possibleToPairSticks : Prop :=
  ∀ (r1 r2 r3 b1 b2 b3 b4 b5 : ℕ),
    r1 + r2 + r3 = 30 ∧
    b1 + b2 + b3 + b4 + b5 = 30 ∧
    r1 ≠ r2 ∧ r2 ≠ r3 ∧ r1 ≠ r3 ∧
    b1 ≠ b2 ∧ b2 ≠ b3 ∧ b3 ≠ b4 ∧ b4 ≠ b5 ∧ b1 ≠ b3 ∧ b1 ≠ b4 ∧ b1 ≠ b5 ∧ b2 ≠ b4 ∧ b2 ≠ b5 ∧ b3 ≠ b5 →   
    ∃ (cutting_strategy : list (ℕ × ℕ)),
      ∀ (r_i : ℕ), r_i ∈ list [r1, r2, r3] →
      ∃ (b_i : ℕ), b_i ∈ list [b1, b2, b3, b4, b5] ∧ 
        (r_i = b_i ∨ (∃ (cut_length : ℕ), cut_length < b_i ∧ (r_i = cut_length)))

theorem sticks_can_be_paired : possibleToPairSticks := 
by
  -- Proof goes here.
  sorry

end sticks_can_be_paired_l587_587801


namespace percentage_hindus_l587_587665

-- Conditions 
def total_boys : ℕ := 850
def percentage_muslims : ℝ := 0.44
def percentage_sikhs : ℝ := 0.10
def boys_other_communities : ℕ := 272

-- Question and proof statement
theorem percentage_hindus (total_boys : ℕ) (percentage_muslims percentage_sikhs : ℝ) (boys_other_communities : ℕ) : 
  (total_boys = 850) →
  (percentage_muslims = 0.44) →
  (percentage_sikhs = 0.10) →
  (boys_other_communities = 272) →
  ((850 - (374 + 85 + 272)) / 850) * 100 = 14 := 
by
  intros
  sorry

end percentage_hindus_l587_587665


namespace tan_x_tan_y_relation_l587_587299

/-- If 
  (sin x / cos y) + (sin y / cos x) = 2 
  and 
  (cos x / sin y) + (cos y / sin x) = 3, 
  then 
  (tan x / tan y) + (tan y / tan x) = 16 / 3.
 -/
theorem tan_x_tan_y_relation (x y : ℝ)
  (h1 : (Real.sin x / Real.cos y) + (Real.sin y / Real.cos x) = 2)
  (h2 : (Real.cos x / Real.sin y) + (Real.cos y / Real.sin x) = 3) :
  (Real.tan x / Real.tan y) + (Real.tan y / Real.tan x) = 16 / 3 :=
sorry

end tan_x_tan_y_relation_l587_587299


namespace functional_eq_solution_l587_587209

theorem functional_eq_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c := 
by {
  intro h,
  sorry
}

end functional_eq_solution_l587_587209


namespace symmetric_circle_eq_l587_587778

theorem symmetric_circle_eq (x y : ℝ) :
   let A := (1, 2)
   let initial_circle_center := (2, 1)
   let radius := 1
   let symmetric_circle_center := (0, 3)
 in (x-2)^2 + (y-1)^2 = radius^2 →
    (x - symmetric_circle_center.1)^2 + (y - symmetric_circle_center.2)^2 = radius^2 := 
by sorry

end symmetric_circle_eq_l587_587778


namespace find_real_number_l587_587944

theorem find_real_number (a b : ℤ) (h : a > 0 ∧ b > 0) :
  let R := ((a + 1*I)^4 - (127 * I)).re in
  R = 176 ∨ R = 436 ∨ R = 60706 :=
by
  sorry

end find_real_number_l587_587944


namespace sequence_a_general_formula_proof_sequence_b_sum_proof_l587_587674

noncomputable def a_seq_general_formula (n : ℕ) : ℕ :=
  n + 2

noncomputable def b_seq_formula (n : ℕ) : ℕ :=
  n * 2 ^ (a_seq_general_formula n - 2)

noncomputable def T_n (n : ℕ) : ℕ :=
  (n - 1) * 2 ^ (n + 1) + 2

theorem sequence_a_general_formula_proof :
  ∀ (a : ℕ → ℕ), 
  (a 2 = 4) →
  (a 1 + a 2 + a 3 + a 4 = 18) →
  (∀ n, a n = n + 2) :=
sorry

theorem sequence_b_sum_proof :
  ∀ (b : ℕ → ℕ),
  (∀ n, b n = n * 2 ^ (a_seq_general_formula n - 2)) →
  ∀ n, (finset.range n).sum (λ k, b (k + 1)) = T_n n :=
sorry

end sequence_a_general_formula_proof_sequence_b_sum_proof_l587_587674


namespace concession_stand_l587_587483

noncomputable def totalItemsSold (hot_dog_cost soda_cost total_revenue hot_dogs_sold: ℝ) : ℕ :=
  let hot_dog_revenue := hot_dog_cost * hot_dogs_sold
  let soda_revenue := total_revenue - hot_dog_revenue
  let sodas_sold := soda_revenue / soda_cost
  (hot_dogs_sold + sodas_sold)

theorem concession_stand (hot_dog_cost soda_cost total_revenue hot_dogs_sold: ℝ) : 
  hot_dog_cost = 1.5 ∧ soda_cost = 0.5 ∧ total_revenue = 78.5 ∧ hot_dogs_sold = 35 → 
  totalItemsSold hot_dog_cost soda_cost total_revenue hot_dogs_sold = 87 := by
  sorry

end concession_stand_l587_587483


namespace f_compose_self_at_4_eq_1_f_monotonically_decreasing_on_interval_l587_587631

noncomputable def f : ℝ → ℝ :=
  λ x, if x ≤ 2 then -x^2 + 2*x else Real.log x / Real.log 2 - 1

theorem f_compose_self_at_4_eq_1 :
  f (f 4) = 1 :=
sorry

theorem f_monotonically_decreasing_on_interval :
  ∀ x y, 1 ≤ x ∧ x ≤ y ∧ y ≤ 2 → f(y) ≤ f(x) :=
sorry

end f_compose_self_at_4_eq_1_f_monotonically_decreasing_on_interval_l587_587631


namespace percentage_of_green_ducks_l587_587101

def smaller_pond_ducks : ℕ := 30
def larger_pond_ducks : ℕ := 50
def green_percentage_smaller_pond : ℝ := 20 / 100
def green_percentage_larger_pond : ℝ := 12 / 100

theorem percentage_of_green_ducks (total_ducks number_of_green_ducks : ℝ) (h1 : total_ducks = smaller_pond_ducks + larger_pond_ducks)
  (h2 : number_of_green_ducks = green_percentage_smaller_pond * smaller_pond_ducks + green_percentage_larger_pond * larger_pond_ducks) :
  (number_of_green_ducks / total_ducks) * 100 = 15 :=
by
  sorry

end percentage_of_green_ducks_l587_587101


namespace sum_points_distance_l587_587585

theorem sum_points_distance : 
  (∑ n in Finset.range (2013+1) \ {0}, 
   let x := 1 / n
   let y := 1 / (n + 1)
   real.abs (x - y)) = 2013 / 2014 := 
by
  sorry

end sum_points_distance_l587_587585


namespace parallel_planes_imply_l587_587969

variable {Point Line Plane : Type}

-- Definitions of parallelism and perpendicularity between lines and planes
variables {parallel_perpendicular : Line → Plane → Prop}
variables {parallel_lines : Line → Line → Prop}
variables {parallel_planes : Plane → Plane → Prop}

-- Given conditions
variable {m n : Line}
variable {α β : Plane}

-- Conditions
axiom m_parallel_n : parallel_lines m n
axiom m_perpendicular_α : parallel_perpendicular m α
axiom n_perpendicular_β : parallel_perpendicular n β

-- The statement to be proven
theorem parallel_planes_imply (m_parallel_n : parallel_lines m n)
  (m_perpendicular_α : parallel_perpendicular m α)
  (n_perpendicular_β : parallel_perpendicular n β) :
  parallel_planes α β :=
sorry

end parallel_planes_imply_l587_587969


namespace point_on_graph_l587_587481

def lies_on_graph (x y : ℝ) (f : ℝ → ℝ) : Prop :=
  y = f x

theorem point_on_graph :
  lies_on_graph (-2) 0 (λ x => (1 / 2) * x + 1) :=
by
  sorry

end point_on_graph_l587_587481


namespace count_three_digit_sum_24_l587_587231

theorem count_three_digit_sum_24 : 
  let count := (λ n : ℕ, let a := n / 100, b := (n / 10) % 10, c := n % 10 in
                        a + b + c = 24 ∧ 100 ≤ n ∧ n < 1000) in
  (finset.range 1000).filter count = 8 :=
sorry

end count_three_digit_sum_24_l587_587231


namespace football_team_selection_l587_587752

theorem football_team_selection :
  let team_members : ℕ := 12
  let offensive_lineman_choices : ℕ := 4
  let tight_end_choices : ℕ := 2
  let players_left_after_offensive : ℕ := team_members - 1
  let players_left_after_tightend : ℕ := players_left_after_offensive - 1
  let quarterback_choices : ℕ := players_left_after_tightend
  let players_left_after_quarterback : ℕ := quarterback_choices - 1
  let running_back_choices : ℕ := players_left_after_quarterback
  let players_left_after_runningback : ℕ := running_back_choices - 1
  let wide_receiver_choices : ℕ := players_left_after_runningback
  offensive_lineman_choices * tight_end_choices * 
  quarterback_choices * running_back_choices * 
  wide_receiver_choices = 5760 := 
by 
  sorry

end football_team_selection_l587_587752


namespace train_crossing_time_l587_587295

def train_length : ℝ := 375
def bridge_length : ℝ := 1250
def train_speed_kmph : ℝ := 90

-- Convert speed from kmph to m/s
def train_speed_mps : ℝ := (train_speed_kmph * 1000) / 3600

-- Calculate total distance
def total_distance : ℝ := train_length + bridge_length

-- Calculate time
def crossing_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

theorem train_crossing_time : crossing_time total_distance train_speed_mps = 65 :=
by
  -- The proof is omitted, only the statement is required.
  sorry

end train_crossing_time_l587_587295


namespace arithmetic_sequence_problem_l587_587339

theorem arithmetic_sequence_problem (a : Nat → Int) (d a1 : Int)
  (h1 : ∀ n, a n = a1 + (n - 1) * d) 
  (h2 : a 1 + 3 * a 8 = 1560) :
  2 * a 9 - a 10 = 507 :=
sorry

end arithmetic_sequence_problem_l587_587339


namespace pentagon_area_l587_587390

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end pentagon_area_l587_587390


namespace tangent_line_at_point_l587_587779

theorem tangent_line_at_point (x y : ℝ) (h : y = Real.exp x) (t : x = 2) :
  y = Real.exp 2 * x - 2 * Real.exp 2 :=
by sorry

end tangent_line_at_point_l587_587779


namespace num_ordered_pairs_solutions_l587_587583

theorem num_ordered_pairs_solutions :
  ∃ (n : ℕ), n = 18 ∧
    (∀ (a b : ℝ), (∃ x y : ℤ , a * (x : ℝ) + b * (y : ℝ) = 1 ∧ (x * x + y * y = 50))) :=
sorry

end num_ordered_pairs_solutions_l587_587583


namespace find_f_m_l587_587281

-- Definitions based on the conditions
def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x + 3

axiom condition (m a : ℝ) : f (-m) a = 1

-- The statement to be proven
theorem find_f_m (m a : ℝ) (hm : f (-m) a = 1) : f m a = 5 := 
by sorry

end find_f_m_l587_587281


namespace train_speed_is_70_km_h_l587_587881

-- Definitions based on the given conditions
def train_length : ℝ := 180
def platform_length : ℝ := 208.92
def time_to_cross : ℝ := 20

-- The total distance covered by the train
def total_distance : ℝ := train_length + platform_length

-- The speed of the train in m/s
def speed_m_s : ℝ := total_distance / time_to_cross

-- Conversion factor from m/s to km/h
def m_s_to_km_h (speed : ℝ) : ℝ := speed * 3.6

-- The final statement we need to prove
theorem train_speed_is_70_km_h : m_s_to_km_h speed_m_s = 70.0056 := by
  sorry

end train_speed_is_70_km_h_l587_587881


namespace minimum_liars_100_people_circle_l587_587130

noncomputable def minimal_liars (n : ℕ) : ℕ := 34

theorem minimum_liars_100_people_circle : ∀ n : ℕ, n = 100 →
  ∃ m : ℕ, m ≤ n ∧ m = minimal_liars n :=
by
  intros n hn
  rw hn
  use 34
  split
  -- Proof that 34 is less than or equal to 100 is obvious and skipped.
  sorry
  -- Proof that 34 is the minimum number of liars follows from reasoning given in the problem statement.
  sorry

end minimum_liars_100_people_circle_l587_587130


namespace problem_solution_l587_587540

theorem problem_solution :
  (19 * 19 - 12 * 12) / ((19 / 12) - (12 / 19)) = 228 :=
by sorry

end problem_solution_l587_587540


namespace tree_height_end_of_third_year_l587_587883

theorem tree_height_end_of_third_year (h : ℝ) : 
    (∃ h0 h3 h6 : ℝ, 
      h3 = h0 * 3^3 ∧ 
      h6 = h3 * 2^3 ∧ 
      h6 = 1458) → h3 = 182.25 :=
by sorry

end tree_height_end_of_third_year_l587_587883


namespace area_of_rectangle_l587_587669

theorem area_of_rectangle (A B C D O : Type) [rectangle ABCD] 
  (intersect_ac_bd_at_O : intersect AC BD = O)
  (equilateral_triangle_AOB : equilateral △AOB)
  (AB_eq_10 : AB = 10) : 
  area ABCD = 100 * Real.sqrt 3 :=
by
  sorry

end area_of_rectangle_l587_587669


namespace train_speed_in_kmph_l587_587154

noncomputable def length_of_train : ℝ := 110
noncomputable def length_of_bridge : ℝ := 132
noncomputable def time_to_cross_bridge : ℝ := 24.198064154867613

theorem train_speed_in_kmph :
  let total_distance := length_of_train + length_of_bridge in
  let speed_in_mps := total_distance / time_to_cross_bridge in
  let speed_in_kmph := speed_in_mps * 3.6 in
  speed_in_kmph = 36 :=
by
  sorry

end train_speed_in_kmph_l587_587154


namespace average_rate_640_miles_trip_l587_587733

theorem average_rate_640_miles_trip 
  (total_distance : ℕ) 
  (first_half_distance : ℕ) 
  (first_half_rate : ℕ) 
  (second_half_time_multiplier : ℕ) 
  (first_half_time : ℕ := first_half_distance / first_half_rate)
  (second_half_time : ℕ := second_half_time_multiplier * first_half_time)
  (total_time : ℕ := first_half_time + second_half_time)
  (average_rate : ℕ := total_distance / total_time) : 
  total_distance = 640 ∧ 
  first_half_distance = 320 ∧ 
  first_half_rate = 80 ∧ 
  second_half_time_multiplier = 3 → 
  average_rate = 40 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  rw [h1, h2, h3, h4] at *
  have h5 : first_half_time = 320 / 80 := rfl
  have h6 : second_half_time = 3 * (320 / 80) := rfl
  have h7 : total_time = (320 / 80) + 3 * (320 / 80) := rfl
  have h8 : average_rate = 640 / (4 + 12) := rfl
  have h9 : average_rate = 640 / 16 := rfl
  have average_rate_correct : average_rate = 40 := rfl
  exact average_rate_correct

end average_rate_640_miles_trip_l587_587733


namespace sum_first_2016_terms_of_cn_l587_587949

def a (n : ℕ) : ℕ := n

def S (n : ℕ) : ℕ := n * (n + 1) / 2

def c (n : ℕ) : ℚ := (-1) ^ n * ((2 * a n + 1) / (2 * S n))

def sum_of_c (N : ℕ) : ℚ :=
  finset.sum (finset.range (N + 1)) c

theorem sum_first_2016_terms_of_cn :
  sum_of_c 2016 = -2016 / 2017 :=
sorry

end sum_first_2016_terms_of_cn_l587_587949


namespace rearrange_and_compute_average_l587_587509

open Function

theorem rearrange_and_compute_average :
  let nums := [-3, 0, 5, 8, 11, 13]
  let largest := 13
  let smallest := -3
  let medians := [5, 8]
  ∃ arrangement : List ℤ,
    (arrangement.length = nums.length) ∧
    (arrangement.perm nums) ∧
    (¬ (arrangement.head? = some largest)) ∧
    largest ∈ arrangement.take 4 ∧
    (¬ (arrangement.reverse.head? = some smallest)) ∧
    smallest ∈ arrangement.drop (arrangement.length - 4) ∧
    (¬ (arrangement.head? = some (medians.head)) ∧ (¬ (arrangement.reverse.head? = some (medians.head)))) ∧
    (¬ (arrangement.head? = some (medians.tail.head?))) ∧ (¬ (arrangement.reverse.head=? some (medians.tail.head?))) →
    let first := arrangement.head?
    let last := arrangement.reverse.head?
    ∃ first last,
      (first = some 11) ∧ (last = some 0) ∧
      (first + last) / 2 = 5.5 :=
  sorry

end rearrange_and_compute_average_l587_587509


namespace find_x_collinear_l587_587642

theorem find_x_collinear (x : ℝ) (a b : ℝ × ℝ) (h_a : a = (2, 1)) (h_b : b = (x, -1)) 
  (h_collinear : ∃ k : ℝ, (a.1 - b.1, a.2 - b.2) = (k * b.1, k * b.2)) : x = -2 :=
by 
  -- the proof would go here
  sorry

end find_x_collinear_l587_587642


namespace prism_lateral_surface_area_l587_587143

def rhombus_diagonals_lateral_surface_area (d1 d2 height : ℝ) : ℝ :=
  let side_len := (Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2))
  4 * side_len * height

theorem prism_lateral_surface_area :
  rhombus_diagonals_lateral_surface_area 9 15 5 = 160 :=
by
  sorry

end prism_lateral_surface_area_l587_587143


namespace average_speed_is_40_l587_587744

-- Define the total distance
def total_distance : ℝ := 640

-- Define the distance for the first half
def first_half_distance : ℝ := total_distance / 2

-- Define the average speed for the first half
def first_half_speed : ℝ := 80

-- Define the time taken for the first half
def first_half_time : ℝ := first_half_distance / first_half_speed

-- Define the multiplicative factor for time increase in the second half
def time_increase_factor : ℝ := 3

-- Define the time taken for the second half
def second_half_time : ℝ := first_half_time * time_increase_factor

-- Define the total time for the trip
def total_time : ℝ := first_half_time + second_half_time

-- Define the calculated average speed for the entire trip
def calculated_average_speed : ℝ := total_distance / total_time

-- State the theorem that the average speed for the entire trip is 40 miles per hour
theorem average_speed_is_40 : calculated_average_speed = 40 :=
by
  sorry

end average_speed_is_40_l587_587744


namespace number_of_paths_l587_587545

theorem number_of_paths {width height path_length : ℕ} (h_width : width = 6) (h_height : height = 5) (h_path_length : path_length = 10) :
  (nat.choose path_length height) = 210 :=
by
  sorry

end number_of_paths_l587_587545


namespace smallest_prime_divisor_of_sum_l587_587466

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end smallest_prime_divisor_of_sum_l587_587466


namespace total_players_l587_587496

def kabaddi (K : ℕ) (Kho_only : ℕ) (Both : ℕ) : ℕ :=
  K - Both + Kho_only + Both

theorem total_players (K : ℕ) (Kho_only : ℕ) (Both : ℕ)
  (hK : K = 10)
  (hKho_only : Kho_only = 35)
  (hBoth : Both = 5) :
  kabaddi K Kho_only Both = 45 :=
by
  rw [hK, hKho_only, hBoth]
  unfold kabaddi
  norm_num

end total_players_l587_587496


namespace cyrus_first_day_pages_l587_587907

-- Define the conditions as hypotheses.
variables (x : ℕ) -- Number of pages written on the first day.
variables (h1 : 500 - 315 = 185) -- Total pages written so far.
variables (h2 : x + 2 * x + 4 * x + 10 = 185) -- Sum of pages written in four days equals total pages written so far.

-- State the theorem that proves the solution.
theorem cyrus_first_day_pages : x = 25 :=
by
  have H : 7 * x + 10 = 185 := h2
  have H1 : 7 * x = 175 := by
    rw add_comm at H
    exact (nat.sub_eq_of_eq_add (eq.symm H))
  
  exact (nat.div_eq_of_eq_mul_right (by norm_num) H1)

end cyrus_first_day_pages_l587_587907


namespace largest_roots_bounds_l587_587550

theorem largest_roots_bounds
  (a_3 a_2 a_1 a_0 : ℝ)
  (h1 : |a_3| < 3) (h2 : |a_2| < 3)
  (h3 : |a_1| < 3) (h4 : |a_0| < 3) :
  ∃ r s : ℝ, (|r - 4| < 0.5) ∧ (|s + 2| < 0.5) ∧ 
             (∃ x : ℝ, x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 = 0 ∧ x = r) ∧
             (∃ x : ℝ, x^4 + a_3*x^3 + a_2*x^2 + a_1*x + a_0 = 0 ∧ x = s) :=
begin
  sorry
end

end largest_roots_bounds_l587_587550


namespace problem_statement_l587_587379

noncomputable def f (m x : ℝ) : ℝ := -2 * m + 2 * m * sin (x + (3 * Real.pi / 2)) - 2 * cos (x - (Real.pi / 2)) ^ 2 + 1

noncomputable def h (m : ℝ) : ℝ :=
  if m > 2 then -4 * m + 1
  else if 0 ≤ m ∧ m ≤ 2 then - (m ^ 2 / 2) - 2 * m - 1
  else -2 * m - 1

theorem problem_statement :
  ∀ (m : ℝ), h m = 1 / 2 → m = -3 / 4 ∧ (∀ x ∈ Icc (-Real.pi / 2) 0, f (-3 / 4) x ≤ 4) :=
begin
  sorry
end

end problem_statement_l587_587379


namespace celsius_equals_fahrenheit_l587_587750

-- Define the temperature scales.
def celsius_to_fahrenheit (T_C : ℝ) : ℝ := 1.8 * T_C + 32

-- The Lean statement for the problem.
theorem celsius_equals_fahrenheit : ∃ (T : ℝ), T = celsius_to_fahrenheit T ↔ T = -40 :=
by
  sorry -- Proof is not required, just the statement.

end celsius_equals_fahrenheit_l587_587750


namespace non_negative_integer_solutions_of_inequality_system_l587_587033

theorem non_negative_integer_solutions_of_inequality_system :
  (∀ x : ℚ, 3 * (x - 1) < 5 * x + 1 → (x - 1) / 2 ≥ 2 * x - 4 → (x = 0 ∨ x = 1 ∨ x = 2)) :=
by
  sorry

end non_negative_integer_solutions_of_inequality_system_l587_587033


namespace infinitely_many_bisectors_locus_of_centers_of_bisectors_l587_587294

-- Given non-concentric circles C and C' with centers O and O', radii r and r'
variables {C C' : Circle} {O O' : Point} {r r' : ℝ}
  (hC : C.center = O ∧ C.radius = r)
  (hC' : C'.center = O' ∧ C'.radius = r')
  (h_non_concentric : O ≠ O')

-- Definition: A circle bisects another circle if their common chord is a diameter of the other circle
def bisects (C C' : Circle) : Prop :=
  ∃ (P : Point),
    ∀ (A B : Point), C.contains(A) ∧ C.contains(B) ∧ line_through(A, B).is_diameter_of(C') → 
      ∃ (k : Circle), k.center = P ∧ k.bisects(C) ∧ k.bisects(C')

-- Hypothesis: Non-concentric circles C and C' are given
-- Show: There are infinitely many circles that bisect both C and C'
theorem infinitely_many_bisectors :
  ∃ (P : Point), bisects C C' :=
sorry

-- Locus of centers of bisecting circles is the reflection of the radical axis in the perpendicular bisector of OO'
theorem locus_of_centers_of_bisectors :
  locus_of_centers C C' = reflect_of_radical_axis_perpendicular_bisector O O' :=
sorry

end infinitely_many_bisectors_locus_of_centers_of_bisectors_l587_587294


namespace max_entanglements_l587_587860

theorem max_entanglements (a b : ℕ) (h1 : a < b) (h2 : a < 1000) (h3 : b < 1000) :
  ∃ n ≤ 9, ∀ k, k ≤ n → ∃ a' b' : ℕ, (b' - a' = b - a - 2^k) :=
by sorry

end max_entanglements_l587_587860


namespace log_equation_solution_l587_587570

theorem log_equation_solution (x : ℝ) (h_pos : x > 0) (h_ne3 : x ≠ 3) (h_ne4 : x ≠ 4) :
  (log (x^2) (x^2 - 7 * x + 12)) + (log (x^2) (x^2 / (x - 3))) + (log (x^2) (x^2 / (x - 4))) = 2 → x = 5 :=
by
  sorry

end log_equation_solution_l587_587570


namespace value_of_f2005_l587_587716

def f (x : ℚ) : ℚ := (1 + x) / (1 - 3 * x)

def f_n (n : ℕ) : (ℚ → ℚ) 
| 0       := id
| (n + 1) := f ∘ (f_n n)

theorem value_of_f2005 (x : ℚ) (h : x = 4.7) : f_n 2005 x = 37 / 57 := by
  sorry

end value_of_f2005_l587_587716


namespace georgie_entry_exit_ways_l587_587510

statement

-- Define the number of windows in the mansion
def number_of_windows : ℕ := 8

-- Define the question we want to prove:
-- The number of ways Georgie can enter through one window and exit through a different window is 56.
theorem georgie_entry_exit_ways : (number_of_windows * (number_of_windows - 1) = 56) :=
by { sorry }

end georgie_entry_exit_ways_l587_587510


namespace container_capacity_l587_587118

theorem container_capacity :
  ∃ (M : ℝ), (9 / 25 : ℝ) ^ (1 / 6) = (M - 15) / M ∧
  M ≈ 49 :=
by
  sorry

end container_capacity_l587_587118


namespace volume_ratio_l587_587082

theorem volume_ratio (A B C : ℝ) 
  (h1 : A = (B + C) / 4)
  (h2 : B = (C + A) / 6) : 
  C / (A + B) = 23 / 12 :=
sorry

end volume_ratio_l587_587082


namespace total_carrots_l587_587024

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l587_587024


namespace find_cans_lids_l587_587158

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l587_587158


namespace concurrency_of_median_circle_intersections_l587_587073

theorem concurrency_of_median_circle_intersections
  (A B C : EuclideanGeometry.Point)
  (M1 M2 M3 : EuclideanGeometry.Point)
  (MA MB MC : EuclideanGeometry.Line)
  (circ1 circ2 circ3 : EuclideanGeometry.Circle) :
  EuclideanGeometry.is_median A M1 B C ∧
  EuclideanGeometry.is_median B M2 A C ∧
  EuclideanGeometry.is_median C M3 A B ∧
  EuclideanGeometry.is_circle_diameter circ1 A B ∧
  EuclideanGeometry.is_circle_diameter circ2 B C ∧
  EuclideanGeometry.is_circle_diameter circ3 C A ∧
  EuclideanGeometry.intersects_pairwise circ1 circ2 circ3 →
  let C1 := EuclideanGeometry.intersection_point circ1 circ2
  let A1 := EuclideanGeometry.intersection_point circ2 circ3
  let B1 := EuclideanGeometry.intersection_point circ3 circ1 
  in EuclideanGeometry.are_concurrent (EuclideanGeometry.line_through A A1)
                                      (EuclideanGeometry.line_through B B1)
                                      (EuclideanGeometry.line_through C C1) := 
begin
  sorry
end

end concurrency_of_median_circle_intersections_l587_587073


namespace simplify_expr1_simplify_expr2_l587_587030

variable {a b : ℝ} -- Assume a and b are arbitrary real numbers

-- Part 1: Prove that 2a - [-3b - 3(3a - b)] = 11a
theorem simplify_expr1 : (2 * a - (-3 * b - 3 * (3 * a - b))) = 11 * a :=
by
  sorry

-- Part 2: Prove that 12ab^2 - [7a^2b - (ab^2 - 3a^2b)] = 13ab^2 - 10a^2b
theorem simplify_expr2 : (12 * a * b^2 - (7 * a^2 * b - (a * b^2 - 3 * a^2 * b))) = (13 * a * b^2 - 10 * a^2 * b) :=
by
  sorry

end simplify_expr1_simplify_expr2_l587_587030


namespace imo_inequality_l587_587703

variables {α : Type*}
variables (a b c d1 d2 d3 : ℝ)
variables (P : EuclideanGeometry.Point α) (A B C : EuclideanGeometry.Point α)

-- definitions of the distances from point P to the sides of the triangle
def distance_to_BC := d1
def distance_to_CA := d2
def distance_to_AB := d3

-- side lengths of the triangle
def side_BC := a
def side_CA := b
def side_AB := c

-- area of the triangle ABC
def area_triangle_ABC := (1 / 2) * (a * d1 + b * d2 + c * d3)

-- statement of the problem to be proven
theorem imo_inequality (h1 : distance_to_BC = d1) (h2 : distance_to_CA = d2) (h3 : distance_to_AB = d3)
  (h4 : side_BC = a) (h5 : side_CA = b) (h6 : side_AB = c) :
  (a / d1) + (b / d2) + (c / d3) ≥ ((a + b + c) * (a + b + c)) / (2 * area_triangle_ABC) :=
by sorry

end imo_inequality_l587_587703


namespace R_and_D_expenditure_l587_587535

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end R_and_D_expenditure_l587_587535


namespace max_cookie_price_l587_587354

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end max_cookie_price_l587_587354


namespace polynomial_degree_expansion_l587_587898

def p (x : ℝ) := 2*x^5 + 3*x^3 + x - 14
def q (x : ℝ) := 3*x^10 - 9*x^7 + 9*x^5 + 30
def r (x : ℝ) := x^3 + 5

theorem polynomial_degree_expansion : 
  degree ((p(x) * q(x)) - (r(x)^6)) = 18 :=
sorry

end polynomial_degree_expansion_l587_587898


namespace problem_statement_l587_587179

noncomputable def z1 := complex.mk 3 1
noncomputable def z2 := complex.mk 5 1
noncomputable def z3 := complex.mk 7 1
noncomputable def z4 := complex.mk 8 1

theorem problem_statement :
  real.arctan (1 / 3) + real.arctan (1 / 5) + real.arcsin (1 / real.sqrt 50) + real.arcsin (1 / real.sqrt 65) = real.pi / 4 :=
by
  sorry

end problem_statement_l587_587179


namespace xiaoyue_average_speed_l587_587097

theorem xiaoyue_average_speed :
  ∃ (x : ℕ), 
    let xiaofang_speed := 1.2 * x,
        xiaoyue_time := 20 / x,
        xiaofang_time := 18 / xiaofang_speed in
    xiaoyue_time - xiaofang_time = 1 / 10 ∧ x = 50 :=
by
  sorry

end xiaoyue_average_speed_l587_587097


namespace tangent_line_eq_l587_587414

open Real

theorem tangent_line_eq (x : ℝ) (h1 : x = 1) : 
  let y := 2 * log x in 
  has_deriv_at (λ x, 2 * log x) (2 / x) x →
  y = 2 * x - 2 :=
by
  intro hx deriv
  -- Solution should be manually proven here.
  sorry

end tangent_line_eq_l587_587414


namespace zeros_ordering_l587_587655

def f (x : ℝ) : ℝ := x - x^(1 / 2)
def g (x : ℝ) : ℝ := x + Real.exp x
def h (x : ℝ) : ℝ := x + Real.log x

theorem zeros_ordering {x1 x2 x3 : ℝ} 
  (hx1 : f x1 = 0) 
  (hx2 : g x2 = 0) 
  (hx3 : h x3 = 0) : 
  x2 < x3 ∧ x3 < x1 :=
sorry

end zeros_ordering_l587_587655


namespace find_numbers_l587_587868

theorem find_numbers (x y a : ℕ) (h1 : x = 6 * y - a) (h2 : x + y = 38) : 7 * x = 228 - a → y = 38 - x :=
by
  sorry

end find_numbers_l587_587868


namespace best_fitting_model_l587_587091

theorem best_fitting_model :
  let R2_M1 := 0.98
  let R2_M2 := 0.80
  let R2_M3 := 0.50
  let R2_M4 := 0.25
  let best_model (R2: ℝ) (R2_values: List ℝ) := R2 = List.maximum (R2_values)
  best_model R2_M1 [R2_M1, R2_M2, R2_M3, R2_M4] :=
sory

end best_fitting_model_l587_587091


namespace product_secant_pq_values_l587_587482

-- Defining the angles to work with in degrees
def deg_to_rad (d : ℝ) : ℝ := d * (Real.pi / 180)

-- Question: Compute the product of secant squares
theorem product_secant : 
  (∏ k in Finset.range 30, Real.sec (deg_to_rad (3 * (k + 1))) ^ 2) = 1 := sorry

-- Conclude the values of p and q and their sum
theorem pq_values :
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ p ^ q = 1 ∧ p + q = 2 := sorry

end product_secant_pq_values_l587_587482


namespace range_of_m_l587_587962

theorem range_of_m (a m : ℝ) (h_a_neg : a < 0) (y1 y2 : ℝ)
  (hA : y1 = a * m^2 - 4 * a * m)
  (hB : y2 = 4 * a * m^2 - 8 * a * m)
  (hA_above : y1 > -3 * a)
  (hB_above : y2 > -3 * a)
  (hy1_gt_y2 : y1 > y2) :
  4 / 3 < m ∧ m < 3 / 2 :=
sorry

end range_of_m_l587_587962


namespace rectangle_width_change_l587_587816

theorem rectangle_width_change 
  (L W : ℝ) 
  (hL : 1.3 * L - L = 0.30 * L) 
  (hA : 1.09 * (L * W) = 1.3 * L * W * (1 - x / 100)) : 
  x ≈ 16.15 :=
by {
  sorry
}

end rectangle_width_change_l587_587816


namespace roots_of_quadratic_eq_form_l587_587917

theorem roots_of_quadratic_eq_form (a b c m n p : ℕ) (h : a = 3) (h2 : b = -8) (h3 : c = 2) (h4 : p = 3) (h5 : m = 4):
  (∀ x, x = (m + sqrt n) / p ∨ x = (m - sqrt n) / p) ∧ gcd m n p = 1 → n = 10 :=
by
  intro h_eq
  sorry

end roots_of_quadratic_eq_form_l587_587917


namespace smallest_product_of_non_real_zeros_l587_587416

def quartic_polynomial (a b c d x : ℝ) : ℝ :=
  x ^ 4 + a * x ^ 3 + b * x ^ 2 + c * x + d

theorem smallest_product_of_non_real_zeros
  (a b c d : ℝ)
  (h1 : quartic_polynomial a b c d (-1) > 4)
  (h2 : quartic_polynomial a b c d 0 > 4)
  (h3 : quartic_polynomial a b c d 1 > 2)
  (h4₁ : ∃ x ∈ (Ioo 1 2), quartic_polynomial a b c d x = 0)
  (h4₂ : ∃ x ∈ (Ioo 3 4), quartic_polynomial a b c d x = 0) :
  ∃ p : ℝ, (p < 2) ∧ 
    (∀ v ∈ {quartic_polynomial a b c d (-1), quartic_polynomial a b c d 0, 
            ∑ i in (finset.range 4), coeff (quartic_polynomial a b c d i), 
            ∑ i in (finset.Icc 1 4), root (quartic_polynomial a b c d i)}, 
        p ≤ v)
  sorry

end smallest_product_of_non_real_zeros_l587_587416


namespace angle_between_AD_and_BC_l587_587965

variables (A B C D E F : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]

noncomputable def point : Type := ℝ × ℝ × ℝ

noncomputable def A : point := (0, 0, 0)
noncomputable def D : point := (1, 0, 0)
noncomputable def B : point := (0, 1, 0)
noncomputable def C : point := (1, 1, 0)

-- Given Conditions
axiom AD_eq_BC : dist A D = 1 ∧ dist B C = 1
axiom BE_EA_ratio : ∃ E : point, dist B E = 2 * dist E A
axiom CF_FD_ratio : ∃ F : point, dist C F = 2 * dist F D
axiom EF_value : ∃ a : ℝ, a > 0 ∧ dist E F = a

-- Proof Problem Statement
theorem angle_between_AD_and_BC (a : ℝ) (ha : a > 0) :
  ∃ θ : ℝ, θ = real.arccos ((5 - 9 * a^2) / 4) ∨ θ = real.pi - real.arccos ((5 - 9 * a^2) / 4) := 
sorry

end angle_between_AD_and_BC_l587_587965


namespace maximum_x_minus_y_l587_587709

theorem maximum_x_minus_y (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ 2 * Real.sqrt 3 / 3 := by
  sorry

end maximum_x_minus_y_l587_587709


namespace range_of_z_l587_587615

theorem range_of_z {x y : ℝ} 
  (h1 : x - y ≤ 2)
  (h2 : x + 2 * y ≥ 5)
  (h3 : y ≤ 2) : 
  (⅓ ≤ (y + 1) / (x + 1)) ∧ ((y + 1) / (x + 1) ≤ 3 / 2) :=
sorry

end range_of_z_l587_587615


namespace total_matches_won_l587_587328

-- Define the conditions
def matches_in_first_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat
def matches_in_second_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat

-- The main proof statement that we need to prove
theorem total_matches_won (total1 total2 : ℕ) (win_rate1 win_rate2 : ℚ) :
  matches_in_first_period total1 win_rate1 + matches_in_second_period total2 win_rate2 = 110 :=
by
  sorry

end total_matches_won_l587_587328


namespace problem_statement_l587_587336

open BigOperators

-- Defining the arithmetic sequence
def a (n : ℕ) : ℕ := n - 1

-- Defining the sequence b_n
def b (n : ℕ) : ℕ :=
if n % 2 = 1 then
  a n + 1
else
  2 ^ a n

-- Defining T_2n as the sum of the first 2n terms of b
def T (n : ℕ) : ℕ :=
(∑ i in Finset.range n, b (2 * i + 1)) +
(∑ i in Finset.range n, b (2 * i + 2))

-- The theorem to be proven
theorem problem_statement (n : ℕ) : 
  a 2 * (a 4 + 1) = a 3 ^ 2 ∧
  T n = n^2 + (2^(2*n+1) - 2) / 3 :=
by
  sorry

end problem_statement_l587_587336


namespace P_is_incenter_CDE_l587_587551

-- Assumptions and Definitions based on the conditions
variables {A B C P D E : Type} 
variables [triangle : Type] (ABC : triangle)
-- Defining angle condition
variable [has_angle_greater_than_right : ∀ T, has_angle T A B C (some_angle_condition)]
-- Defining circumradius
variable (r : some_radius)
-- Defining point P properties
variable (P_on_AB : P ∈ segment AB)
variable (PB_eq_PC : distance P B = distance P C)
variable (PA_eq_r : distance P A = r)
-- Defining circumcircle and perpendicular bisector intersection points
variable (Gamma : circle)
variable (D_on_Gamma : D ∈ circumcircle ABC Gamma)
variable (E_on_Gamma : E ∈ circumcircle ABC Gamma)
variable (P_bisector_intersects : perpendicular_bisector PB ∩ Gamma = {D, E})

-- Proposition to prove P is the incenter of triangle CDE
theorem P_is_incenter_CDE : is_incenter P (triangle C D E) :=
sorry

end P_is_incenter_CDE_l587_587551


namespace exists_unique_xy_l587_587012

theorem exists_unique_xy (n : ℕ) : ∃! (x y : ℕ), n = ((x + y)^2 + 3*x + y) / 2 :=
sorry

end exists_unique_xy_l587_587012


namespace total_matches_won_l587_587322

-- Condition definitions
def matches1 := 100
def win_percentage1 := 0.5
def matches2 := 100
def win_percentage2 := 0.6

-- Theorem statement
theorem total_matches_won : matches1 * win_percentage1 + matches2 * win_percentage2 = 110 :=
by
  sorry

end total_matches_won_l587_587322


namespace number_of_elements_in_P7_maximum_n_for_sparse_partition_l587_587222

-- Definition of I_n
def I (n : ℕ) : set ℕ := { m | 1 ≤ m ∧ m ≤ n }

-- Definition of P_n
def P (n : ℕ) : set ℚ := { q | ∃ m k : ℕ, m ∈ I n ∧ k ∈ I n ∧ q = m / real.sqrt k }

-- Problem (1)
theorem number_of_elements_in_P7 : ∀ n, n = 7 → set.card (P n) = 46 :=
by 
  intro n hn
  rw hn
  sorry

-- Definition of sparse set
def is_sparse_set (s : set ℚ) : Prop :=
  ∀ a b ∈ s, ¬ ∃ k : ℕ, k*k = a + b

-- Problem (2)
theorem maximum_n_for_sparse_partition : 
  ∃ n, (∀ s1 s2 : set ℚ, s1 ∪ s2 = P n → s1 ∩ s2 = ∅ → is_sparse_set s1 → is_sparse_set s2 → s1 ∪ s2 = P n) ↔ n = 14 :=
by
  use 14
  sorry

end number_of_elements_in_P7_maximum_n_for_sparse_partition_l587_587222


namespace other_divisor_l587_587940

theorem other_divisor (x : ℕ) (h1 : 261 % 37 = 2) (h2 : 261 % x = 2) (h3 : 259 = 261 - 2) :
  ∃ x : ℕ, 259 % 37 = 0 ∧ 259 % x = 0 ∧ x = 7 :=
by
  sorry

end other_divisor_l587_587940


namespace part_a_concurrent_lines_part_b_concurrent_lines_l587_587701

theorem part_a_concurrent_lines
  (n : ℕ) (A : Fin n → ℝ × ℝ)
  (hA : ∀ i j : Fin n, i ≠ j → A i ≠ A j)
  (hPolygon : ∀ i : Fin n, dist (A i) (A (i + 1) % n) = dist (A 0) (A 1))
  (X : Fin n → ℝ × ℝ)
  (hX_intersections : ∀ i : Fin n, is_intersection (A (i + n - 2) % n) (A (i + n - 1) % n) 
                                      (A i) (A (i + 1) % n) (X i))
  (Γ : ℝ × ℝ → Prop)
  (hΓ_X : ∀ i : Fin n, Γ (X i))
  (ω : Fin n → (ℝ × ℝ) → Prop)
  (T : Fin n → ℝ × ℝ)
  (hTangency_ω_Γ : ∀ i, is_tangent (ω i (X i)) (Γ (T i)))
  :
  ∃ T_0 : ℝ × ℝ, ∀ i, collinear (X i) (T i) T_0 :=
sorry

theorem part_b_concurrent_lines
  (n : ℕ) (A : Fin n → ℝ × ℝ)
  (hA : ∀ i j : Fin n, i ≠ j → A i ≠ A j)
  (hPolygon : ∀ i : Fin n, dist (A i) (A (i + 1) % n) = dist (A 0) (A 1))
  (X : Fin n → ℝ × ℝ)
  (hX_intersections : ∀ i : Fin n, is_intersection (A (i + n - 2) % n) (A (i + n - 1) % n) 
                                      (A i) (A (i + 1) % n) (X i))
  (Γ : ℝ × ℝ → Prop)
  (hΓ_X : ∀ i : Fin n, Γ (X i))
  (Ω : Fin n → (ℝ × ℝ) → Prop)
  (S : Fin n → ℝ × ℝ)
  (hTangency_Ω_Γ : ∀ i, is_tangent (Ω i (X i)) (Γ (S i)))
  :
  ∃ S_0 : ℝ × ℝ, ∀ i, collinear (X i) (S i) S_0 :=
sorry

end part_a_concurrent_lines_part_b_concurrent_lines_l587_587701


namespace correct_statements_arithmetic_seq_l587_587225

/-- For an arithmetic sequence {a_n} with a1 > 0 and common difference d ≠ 0, 
    the correct statements among options A, B, C, and D are B and C. -/
theorem correct_statements_arithmetic_seq (a : ℕ → ℤ) (S : ℕ → ℤ) (d : ℤ) (h_seq : ∀ n, a (n + 1) = a n + d) 
  (h_sum : ∀ n, S n = (n * (a 1 + a n)) / 2) (h_a1_pos : a 1 > 0) (h_d_ne_0 : d ≠ 0) : 
  (S 5 = S 9 → 
   S 7 = (10 * a 4) / 2) ∧ 
  (S 6 > S 7 → S 7 > S 8) := 
sorry

end correct_statements_arithmetic_seq_l587_587225


namespace find_original_speed_l587_587140

theorem find_original_speed :
  ∃ x : ℝ, (x > 0) ∧ (100 = x * (2 + 0.5 + (100 - 2*x) / (1.6*x))) ∧ x = 30 :=
begin
  use 30,
  split,
  -- We need to show x > 0
  { exact lt_of_lt_of_le zero_lt_two (le_of_eq rfl), },
  split,
  -- We need to show the equation holds for x = 30
  { norm_num,
    field_simp,
    norm_num,
  },
  -- Finally x has to be 30 as stated
  refl,
end

end find_original_speed_l587_587140


namespace leftmost_three_nonzero_digits_of_arrangements_l587_587608

def choose (n k : ℕ) : ℕ := nat.choose n k

-- Define the main problem statement in Lean
def number_of_possible_six_ring_arrangements : ℕ :=
  (choose 9 6) * 6.factorial * (choose 9 3)

theorem leftmost_three_nonzero_digits_of_arrangements (m : ℕ) 
  (h_m : m = number_of_possible_six_ring_arrangements) : m.digits.take (3).reverse == [5, 0, 8] :=
sorry

end leftmost_three_nonzero_digits_of_arrangements_l587_587608


namespace jelly_bean_remaining_l587_587069

theorem jelly_bean_remaining (J : ℕ) (P : ℕ) (taken_last_4_each : ℕ) (taken_first_each : ℕ) 
 (taken_last_total : ℕ) (taken_first_total : ℕ) (taken_total : ℕ) (remaining : ℕ) :
  J = 8000 →
  P = 10 →
  taken_last_4_each = 400 →
  taken_first_each = 2 * taken_last_4_each →
  taken_last_total = 4 * taken_last_4_each →
  taken_first_total = 6 * taken_first_each →
  taken_total = taken_last_total + taken_first_total →
  remaining = J - taken_total →
  remaining = 1600 :=
by
  intros
  sorry  

end jelly_bean_remaining_l587_587069


namespace remaining_perimeter_l587_587521

-- Definitions based on conditions
noncomputable def GH : ℝ := 2
noncomputable def HI : ℝ := 2
noncomputable def GI : ℝ := Real.sqrt (GH^2 + HI^2)
noncomputable def side_JKL : ℝ := 5
noncomputable def JI : ℝ := side_JKL - GH
noncomputable def IK : ℝ := side_JKL - HI
noncomputable def JK : ℝ := side_JKL

-- Problem statement in Lean 4
theorem remaining_perimeter :
  JI + IK + JK = 11 :=
by
  sorry

end remaining_perimeter_l587_587521


namespace find_m_range_l587_587670

noncomputable def problem_statement (m : ℝ) : Prop :=
  ∃ k : ℝ, 
    let l : ℝ → ℝ := fun x => k * x + m in
    let tangent_condition := (1 + k^2 = m^2) in
    tangent_condition ∧ 
    ∃ P : ℝ × ℝ, 
      let A := (-2, 2) in
      let B := (2, 6) in
      let (a, b) := P in
      (a^2 + b^2 - 8 * b + 12 = 0) ∧ 
      ((-2 - a) * (2 - a) + (2 - b) * (6 - b) = -4) ∧
      (abs(b - m) / sqrt(1 + k^2) = 1)

theorem find_m_range (m : ℝ) : problem_statement m ↔ (m > 1 ∨ m < -2):=
  sorry

end find_m_range_l587_587670


namespace true_proposition_l587_587998

-- Conditions from the problem
def p := ∀ x > 0, Log.log (x + 1) > 0
def q := ∀ a b, a > b → a ^ 2 > b ^ 2

-- Statement to be proven
theorem true_proposition : p ∧ ¬q :=
by
  -- Sorry is used here to indicate the proof is omitted
  sorry

end true_proposition_l587_587998


namespace find_shift_left_even_function_l587_587996

def det (a1 a2 a3 a4 : ℝ) : ℝ := a1 * a4 - a2 * a3

def f (x : ℝ) : ℝ := det (sqrt 3) (sin x) 1 (cos x)

def g (x n : ℝ) : ℝ := f (x + n)

theorem find_shift_left_even_function (n : ℝ) (h : n > 0) :
  (∀ x, g x n = g (-x) n) ↔ n = 5 * π / 6 :=
sorry

end find_shift_left_even_function_l587_587996


namespace find_m_value_l587_587648

noncomputable def is_direct_proportion_function (m : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x

theorem find_m_value (m : ℝ) (hk : ∃ k : ℝ, k ≠ 0 ∧ ∀ x : ℝ, (m - 1) * x ^ (2 - m^2) = k * x) : m = -1 :=
by
  sorry

end find_m_value_l587_587648


namespace seq_proof_l587_587247

namespace SequenceProblem

-- Define arithmetic sequence {a_n} with given conditions
def arithmetic_seq (a : ℕ → ℤ) : Prop := 
  (a 1 + a 2 + a 3 + a 4 = 60) ∧ 
  (a 2 + a 4 = 34)

-- Define geometric sequence {b_n} with given conditions
def geometric_seq (b : ℕ → ℤ) : Prop := 
  (b 1 + b 2 + b 3 + b 4 = 120) ∧ 
  (b 2 + b 4 = 90)

-- nth term of arithmetic sequence
def a_n (n : ℕ) : ℤ := 4 * n + 5

-- nth term of geometric sequence
def b_n (n : ℕ) : ℤ := 3^n

-- defining c_n as the product
def c_n (n : ℕ) : ℤ := a_n n * b_n n 

-- Sum of the first n terms of c_n
noncomputable def S_n (n : ℕ) : ℤ := 
  ((4 * n + 3) * (3 ^ (n + 1)) - 9) / 2

-- Theorem stating the equality to be proved
theorem seq_proof : ∀ a b, 
  arithmetic_seq a → geometric_seq b → 
  (∀ n, a_n n = a n) → (∀ n, b_n n = b n) → 
  (∀ n, c_n n = a_n n * b_n n) → 
  (∀ n, S_n n = ∑ i in finset.range n, c_n (i + 1)) :=
by
  sorry

end SequenceProblem

end seq_proof_l587_587247


namespace XYZ_total_length_correct_l587_587507

def diagonal_length (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

def total_length_XYZ : ℝ :=
  let X_diagonals := 2 * diagonal_length 0 0 2 2
  let Y_vertical := 2
  let Y_diagonal := diagonal_length 3 2 4 3
  let Z_horizontals := 6
  let Z_vertical := 2
  X_diagonals + Y_vertical + Y_diagonal + Z_horizontals + Z_vertical

theorem XYZ_total_length_correct :
  total_length_XYZ = 10 + 5 * Real.sqrt 2 := by
  sorry

end XYZ_total_length_correct_l587_587507


namespace max_single_player_salary_l587_587876

theorem max_single_player_salary 
    (num_players : ℕ)
    (min_salary : ℕ)
    (total_budget : ℕ)
    (h_players : num_players = 18)
    (h_min_salary : min_salary = 20000)
    (h_total_budget : total_budget = 600000) :
    ∃ x, 17 * min_salary + x = total_budget ∧ x = 260000 :=
by {
    rw [h_players, h_min_salary, h_total_budget],
    use 260000,
    split,
    { norm_num, },
    { refl, }
}

end max_single_player_salary_l587_587876


namespace initial_mean_of_observations_l587_587420

theorem initial_mean_of_observations (M : ℝ) (h1 : 50 * M + 30 = 50 * 40.66) : M = 40.06 := 
sorry

end initial_mean_of_observations_l587_587420


namespace Stuart_reward_points_l587_587530

theorem Stuart_reward_points (reward_points_per_unit : ℝ) (spending : ℝ) (unit_amount : ℝ) : 
  reward_points_per_unit = 5 → 
  spending = 200 → 
  unit_amount = 25 → 
  (spending / unit_amount) * reward_points_per_unit = 40 :=
by 
  intros h_points h_spending h_unit
  sorry

end Stuart_reward_points_l587_587530


namespace minimum_sum_of_areas_l587_587552

theorem minimum_sum_of_areas (x y : ℝ) (hx : x + y = 16) (hx_nonneg : 0 ≤ x) (hy_nonneg : 0 ≤ y) : 
  (x ^ 2 / 16 + y ^ 2 / 16) / 4 ≥ 8 :=
  sorry

end minimum_sum_of_areas_l587_587552


namespace new_ratio_l587_587811

variable (a b : ℕ)
variable (h_ratio : a = 72 ∧ b = 192)
variable (h_original_ratio : (a / b) = (3 / 8))

theorem new_ratio (h1 : a = 72) (h2 : b = 192) : 
  (a - 24) / (b - 24) = 1 / 3.5 := by
sorry

end new_ratio_l587_587811


namespace xyz_div_by_27_l587_587783

theorem xyz_div_by_27 (x y z : ℤ) (h : (x - y) * (y - z) * (z - x) = x + y + z) :
  27 ∣ (x + y + z) :=
sorry

end xyz_div_by_27_l587_587783


namespace area_shaded_region_l587_587141

-- Define the conditions and parameters of the problem
def side_length_hexagon : ℝ := 8
def radius_sector : ℝ := 4
def angle_sector : ℝ := 60
def total_hexagon_area : ℝ := 6 * (√3 / 4 * side_length_hexagon^2)
def total_sector_area : ℝ := 6 * ((angle_sector / 360) * π * radius_sector^2)

-- Statement of the problem as a proof
theorem area_shaded_region :
  total_hexagon_area - total_sector_area = 96 * √3 - 16 * π := by
  sorry

end area_shaded_region_l587_587141


namespace weight_of_11m_rebar_l587_587579

theorem weight_of_11m_rebar (w5m : ℝ) (l5m : ℝ) (l11m : ℝ) 
  (h_w5m : w5m = 15.3) (h_l5m : l5m = 5) (h_l11m : l11m = 11) : 
  (w5m / l5m) * l11m = 33.66 := 
by {
  sorry
}

end weight_of_11m_rebar_l587_587579


namespace area_of_triangle_l587_587341

def base : ℝ := 10
def height : ℝ := 3

theorem area_of_triangle : (1 / 2) * base * height = 15 := 
by sorry

end area_of_triangle_l587_587341


namespace interval_where_f_is_decreasing_range_of_f_on_interval_l587_587632

noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x) - 4 * cos (x) ^ 2 + 2

theorem interval_where_f_is_decreasing (k : ℤ) :
  ∀ x, (k * π + 3 * π / 8 ≤ x ∧ x ≤ k * π + 7 * π / 8) → deriv (f x) ≤ 0 := sorry

theorem range_of_f_on_interval :
  ∀ x, (3 * π / 4 ≤ x ∧ x ≤ π) → f x ∈ set.Icc (-2 * real.sqrt 2) (-2) := sorry

end interval_where_f_is_decreasing_range_of_f_on_interval_l587_587632


namespace cube_sphere_section_areas_l587_587960

theorem cube_sphere_section_areas (a : ℝ) :
  let R := (a * Real.sqrt 3) / 2 in
  (∀ (i : Fin 12), area_of_bicorn a R = (π * a^2 * (2 - Real.sqrt 3)) / 4) ∧
  (∀ (i : Fin 6), area_of_quadrilateral a R = (π * a^2 * (Real.sqrt 3 - 1)) / 2) :=
by
  sorry

def area_of_bicorn (a R : ℝ) : ℝ := 
  -- The function definition for the area of a bicorn would go here
  sorry

def area_of_quadrilateral (a R : ℝ) : ℝ := 
  -- The function definition for the area of a quadrilateral would go here
  sorry

end cube_sphere_section_areas_l587_587960


namespace sum_infinite_series_l587_587208

theorem sum_infinite_series :
  ∑' n, (n^3 + n^2 + n - 1) / (n + 3)! = 1 / 3 := by
  sorry

end sum_infinite_series_l587_587208


namespace find_x_l587_587515

noncomputable def x : ℝ := 20

def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := x / 100 * 150 - 20 = 10

theorem find_x (x : ℝ) : condition1 x ∧ condition2 x ↔ x = 20 :=
by
  sorry

end find_x_l587_587515


namespace cos_of_sum_eq_one_l587_587592

theorem cos_of_sum_eq_one
  (x y : ℝ)
  (a : ℝ)
  (h1 : x ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h2 : y ∈ Set.Icc (-Real.pi / 4) (Real.pi / 4))
  (h3 : x^3 + Real.sin x - 2 * a = 0)
  (h4 : 4 * y^3 + Real.sin y * Real.cos y + a = 0) :
  Real.cos (x + 2 * y) = 1 := 
by
  sorry

end cos_of_sum_eq_one_l587_587592


namespace correct_option_l587_587499

def bag : Type := { balls : Nat // balls = 3 }

def drawn_balls (b : bag) : Type := { draw : Nat // draw = 2 }

def event_P (d : drawn_balls bag) : Prop := sorry  -- Definition for both drawn balls are black.

def event_Q (d : drawn_balls bag) : Prop := sorry  -- Definition for both drawn balls are white.

def event_R (d : drawn_balls bag) : Prop := sorry  -- Definition for at least one of the drawn balls is black.

theorem correct_option (b : bag) (d : drawn_balls b) : (event_Q d ∧ event_R d) → False :=
sorry
/

end correct_option_l587_587499


namespace total_matches_won_l587_587320

-- Condition definitions
def matches1 := 100
def win_percentage1 := 0.5
def matches2 := 100
def win_percentage2 := 0.6

-- Theorem statement
theorem total_matches_won : matches1 * win_percentage1 + matches2 * win_percentage2 = 110 :=
by
  sorry

end total_matches_won_l587_587320


namespace highway_mileage_l587_587386

theorem highway_mileage (distance_highways : ℕ) (distance_city : ℕ) (mileage_city : ℕ)
  (total_gas : ℝ) (H : ℝ) : 
  distance_highways = 210 → 
  distance_city = 54 → 
  mileage_city = 18 → 
  total_gas = 9 → 
  H = distance_highways / (total_gas - (distance_city / mileage_city)) → 
  H = 35 := 
by
  intros d_highways_eq d_city_eq m_city_eq t_gas_eq H_eq
  have gas_used_city := (distance_city / mileage_city : ℝ)
  have gas_used_city_eq : gas_used_city = 3 := by 
    rw [d_city_eq, m_city_eq]
    norm_num
  have gas_used_highways := (total_gas - gas_used_city)
  have gas_used_highways_eq : gas_used_highways = 6 := by 
    rw [t_gas_eq, gas_used_city_eq]
    norm_num
  have H := (distance_highways / gas_used_highways : ℝ)
  have H_eq := by 
    rw [d_highways_eq, gas_used_highways_eq]
    norm_num
  assumption

end highway_mileage_l587_587386


namespace complex_magnitude_product_l587_587561

theorem complex_magnitude_product :
  abs ((⟨5, -3⟩ : ℂ) * (⟨7, 24⟩ : ℂ)) = 25 * Real.sqrt 34 := by
  sorry

end complex_magnitude_product_l587_587561


namespace combine_sqrt_three_l587_587888

theorem combine_sqrt_three (x : ℕ) (hx8 : x = 8) (hx12 : x = 12) (hx18 : x = 18) : 
  sqrt (4 * 3) = 2 * sqrt 3 ∧ (sqrt x = sqrt 12 → sqrt x = 2 * sqrt 3) :=
by
  sorry

end combine_sqrt_three_l587_587888


namespace option_D_valid_l587_587640

variables (Line : Type) (Plane : Type)

def parallel (l₁ l₂ : Line) : Prop := sorry
def perpendicular (l : Line) (p : Plane) : Prop := sorry
def contains (p : Plane) (l : Line) : Prop := sorry

variables m n : Line
variables α β : Plane

theorem option_D_valid : (perpendicular m α) → (parallel m n) → (parallel α β) → (perpendicular n β) :=
by
  intros h1 h2 h3
  exact sorry

end option_D_valid_l587_587640


namespace count_circle_divisors_l587_587857

def circle_divisors (R : ℕ) : ℕ :=
  let divisors := {d ∈ Nat.divisors R | d < R}.card
  divisors

theorem count_circle_divisors :
  circle_divisors 150 = 11 :=
by
  sorry

end count_circle_divisors_l587_587857


namespace set_union_example_l587_587636

variable (A B : Set ℝ)

theorem set_union_example :
  A = {x | -2 < x ∧ x ≤ 1} ∧ B = {x | -1 ≤ x ∧ x < 2} →
  (A ∪ B) = {x | -2 < x ∧ x < 2} := 
by
  sorry

end set_union_example_l587_587636


namespace volume_of_given_cone_l587_587110

noncomputable def volume_of_cone (r h : ℝ) : ℝ := (1 / 3) * π * r^2 * h

-- Define the existing conditions
def AB : ℝ := 4
def SO : ℝ := 3
def OH : ℝ := AB / 2 -- Half of the side AB since O lies at the center

-- Calculate the height SH using Pythagorean theorem
def SH : ℝ := real.sqrt (SO ^ 2 + OH ^ 2)

-- Define point projections based on problem statement conditions
def AR : ℝ := SH
def AT : ℝ := SH / 2

-- Conditions for point F on edge AD
def AD : ℝ := sqrt (AB^2 + SO^2)
def AF : ℝ := (3 / 5) * AD

-- Radius and height of the cone based on calculations
def r : ℝ := 7
def h : ℝ := 7 * (1 / 6) * SH

-- This is the target theorem statement.
theorem volume_of_given_cone : volume_of_cone r h = (343 / 18) * π * real.sqrt 13 := by
  sorry

end volume_of_given_cone_l587_587110


namespace max_area_perpendicular_l587_587485

theorem max_area_perpendicular (a b θ : ℝ) (ha : 0 < a) (hb : 0 < b) (hθ : 0 ≤ θ ∧ θ ≤ 2 * Real.pi) : 
  ∃ θ_max, θ_max = Real.pi / 2 ∧ (∀ θ, 0 ≤ θ ∧ θ ≤ 2 * Real.pi → 
  (0 < Real.sin θ → (1 / 2) * a * b * Real.sin θ ≤ (1 / 2) * a * b * 1)) :=
sorry

end max_area_perpendicular_l587_587485


namespace triangle_area_l587_587686

theorem triangle_area
  (A B C K L : Type)
  (BC AC AL BK : ℝ) 
  (hBC : BC = 15)
  (hAC : AC = 12)
  (hAL : AL = 6)
  (hBK : BK = 9)
  (hAltitude : ∃ (AK : ℝ), AK > 0 ∧ ∀ (P : Type), P ∈ A → (P ∈ (K, C) → ∃ (Q : Type), Q ∈ (P, B) ∧ dist A Q = dist A P ∧ (AK^2 + PK^2 = PA^2) ∧ (AK^2 + CK^2 = A^2)))
  : ∃ AK : ℝ, AK = 3 * real.sqrt 6 ∧ (1/2 * AK * BC) = 45 / 2 * real.sqrt 6 := sorry

end triangle_area_l587_587686


namespace solution_set_f_ge_1_minimum_value_f_x_in_interval_l587_587993

noncomputable def f (x : ℝ) := (x^2 - 4 * x + 5) / (x - 1)

theorem solution_set_f_ge_1 :
  {x : ℝ | f x ≥ 1} = (Set.Ioo 1 2 ∪ Set.Icc 3 ⊤) :=
sorry

theorem minimum_value_f_x_in_interval :
  (∃ x ∈ Set.Ioi 1, f x = 2 * Real.sqrt 2 - 2) ∧ (∀ x ∈ Set.Ioi 1, f x ≥ 2 * Real.sqrt 2 - 2) :=
 sorry

end solution_set_f_ge_1_minimum_value_f_x_in_interval_l587_587993


namespace tetrahedron_probability_divisible_by_4_l587_587142

-- Definitions for the probability and conditions given problem a)
def probability_favorable_event : ℚ := 13 / 16

theorem tetrahedron_probability_divisible_by_4 :
  ∀ (tetrahedrons : Fin 4 → Fin 4), -- array of 4 tetrahedrons each labeled with 1, 2, 3, 4
  (∃ faces : Fin 4 → Fin 4, -- faces showing 1, 2, 3, or 4
    let product := (List.ofFn faces).prod in
    product % 4 = 0) →
  probability_favorable_event = 13 / 16 := 
sorry

end tetrahedron_probability_divisible_by_4_l587_587142


namespace equal_roots_h_l587_587978

theorem equal_roots_h (h : ℝ) : (∀ x : ℝ, 3 * x^2 - 4 * x + (h / 3) = 0) -> h = 4 :=
by 
  sorry

end equal_roots_h_l587_587978


namespace third_term_of_sequence_l587_587979

def sequence (n : ℕ) : ℚ :=
  if n = 1 then 1 else 
  let rec compute (m : ℕ) (a_m : ℚ) : ℚ :=
    if m = 1 then a_m else compute (m - 1) (a_m + a_m / 2 + 1 / (2 * (m - 1)))
  in
  compute n 0

theorem third_term_of_sequence :
  sequence 3 = 3 / 4 :=
by
  sorry

end third_term_of_sequence_l587_587979


namespace maximum_selection_l587_587819

def isValid (n m : ℕ) : Prop :=
  ¬ ((n + m) % (n - m) = 0 ∨ (n + m) % (m - n) = 0)

def subset_condition (S : set ℕ) : Prop :=
  ∀ {n m : ℕ}, n ∈ S → m ∈ S → n ≠ m → isValid n m

def max_subset_size (s : set ℕ) (N : ℕ) : Prop :=
  s ⊆ {x | x ∈ finset.range (N + 1)} ∧ subset_condition s ∧ finset.card s = N

theorem maximum_selection (N : ℕ) :
  ∀ (S : set ℕ), S ⊆ {x | x ∈ finset.range 1964} ∧ subset_condition S → finset.card S ≤ 655 :=
begin
  sorry
end

example : ∃ S : set ℕ, S ⊆ {x | x ∈ finset.range 1964} ∧ subset_condition S ∧ finset.card S = 655 :=
begin
  use {x | ∃ k : ℕ, k ≤ 654 ∧ x = 3 * k + 1},
  split,
  { intros x hx,
    rcases hx with ⟨k, hk1, rfl⟩,
    simp [nat.le_add_right],
  },
  split,
  { intros n m hn hm h,
    rcases hn with ⟨kn, hkn, rfl⟩,
    rcases hm with ⟨km, hkm, rfl⟩,
    unfold isValid,
    cases nat.eq_or_lt_of_le (nat.le_add_right 1 (3 * kn - 3 * km)) with h1 h2,
    { simp [h1] },
    { contradiction },
  },
  { simp,
    exact finset.card_eq_of_injective _ (λ x hx y hy, begin
        exact finset.ext.mp (finset.range_card_eq hx hy),
    end)
  }
end

end maximum_selection_l587_587819


namespace evaluate_expression_l587_587548

theorem evaluate_expression :
  ((gcd 54 42 |> lcm 36) * (gcd 78 66 |> gcd 90) + (lcm 108 72 |> gcd 66 |> gcd 84)) = 24624 := by
  sorry

end evaluate_expression_l587_587548


namespace count_three_digit_numbers_sum_24_l587_587230

theorem count_three_digit_numbers_sum_24 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 24}.card = 4 := 
by
  sorry

end count_three_digit_numbers_sum_24_l587_587230


namespace xy_value_l587_587451

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end xy_value_l587_587451


namespace correct_factorization_A_l587_587092

-- Define the polynomial expressions
def expression_A : Prop :=
  (x : ℝ) → x^2 - x - 6 = (x + 2) * (x - 3)

def expression_B : Prop :=
  (x : ℝ) → x^2 - 1 = x * (x - 1 / x)

def expression_C : Prop :=
  (x y : ℝ) → 7 * x^2 * y^5 = x * y * 7 * x * y^4

def expression_D : Prop :=
  (x : ℝ) → x^2 + 4 * x + 4 = x * (x + 4) + 4

-- The correct factorization from left to right
theorem correct_factorization_A : expression_A := 
by 
  -- Proof omitted
  sorry

end correct_factorization_A_l587_587092


namespace number_of_elements_B_l587_587313

def A : Set ℤ := {-1, 0, 1, 2, 3}
def B : Set ℤ := {x ∈ A | 1 - x ∉ A}

theorem number_of_elements_B : Set.card B = 1 := 
sorry

end number_of_elements_B_l587_587313


namespace volume_of_tetrahedron_l587_587927

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end volume_of_tetrahedron_l587_587927


namespace two_digit_integer_divides_491_remainder_59_l587_587833

theorem two_digit_integer_divides_491_remainder_59 :
  ∃ n Q : ℕ, (n = 10 * x + y) ∧ (0 < x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) ∧ (491 = n * Q + 59) ∧ (n = 72) :=
by
  sorry

end two_digit_integer_divides_491_remainder_59_l587_587833


namespace ordered_pairs_72_l587_587584

/-- 
Given the system of equations ax + by = 1 and x^2 + y^2 = 25,
prove that there are 72 ordered pairs (a, b) such that the system 
has at least one solution where (x, y) is an integer pair.
-/
theorem ordered_pairs_72 :
  ∃ (pairs : Finset (ℝ × ℝ)), pairs.card = 72 ∧
  ∀ (a b : ℝ), (a, b) ∈ pairs → 
  ∃ (x y : ℤ), ax + by = 1 ∧ (x^2 + y^2 = 25) := sorry

end ordered_pairs_72_l587_587584


namespace evaluate_expression_l587_587183

-- Definitions corresponding to the conditions in the problem
def expr1 : ℕ → ℕ → ℕ :=
  λ n m, (1 + n) * (1 + n / 2) * (1 + n / 3) * ... * (1 + n / m)

def expr2 : ℕ → ℕ → ℕ :=
  λ n m, (1 + n) * (1 + n / 2) * (1 + n / 3) * ... * (1 + n / m)

-- The problem statement that needs to be proved
theorem evaluate_expression :
  expr1 16 18 / expr2 18 16 = 1 :=
by
  sorry

end evaluate_expression_l587_587183


namespace average_cost_per_marker_rounded_l587_587150

noncomputable def total_cost_dollars := 25.50 + 8.50
noncomputable def total_cost_cents := total_cost_dollars * 100
noncomputable def number_of_markers := 300
noncomputable def average_cost_per_marker_cents := total_cost_cents / number_of_markers

theorem average_cost_per_marker_rounded :
  round average_cost_per_marker_cents = 11 :=
  by
    sorry

end average_cost_per_marker_rounded_l587_587150


namespace functional_ineq_fxy_l587_587591

section
  variable (a : ℝ) (f : ℝ → ℝ)
  variable (h_a_pos : a > 0)
  variable (h_continuous : ∀ ε > 0, ∀ x ∈ set.Ioi 0, ∃ δ > 0, ∀ y ∈ set.Ioi 0, abs (x - y) < δ → abs (f x - f y) < ε) 
  variable (h_fa : f a = 1)
  variable (h_ineq : ∀ x y : ℝ, x > 0 → y > 0 → f(x) * f(y) + f(a / x) * f(a / y) ≤ 2 * f(x * y))

  theorem functional_ineq_fxy (x y : ℝ) (hx_pos : x > 0) (hy_pos : y > 0) : 
    f(x) * f(y) ≤ f(x * y) := 
  by 
    sorry
end

end functional_ineq_fxy_l587_587591


namespace circle_area_from_circumference_l587_587046

theorem circle_area_from_circumference (r : ℝ) (π : ℝ) (h1 : 2 * π * r = 36) : (π * (r^2) = 324 / π) := by
  sorry

end circle_area_from_circumference_l587_587046


namespace garland_bulbs_l587_587107

variable (bulb : ℕ → Prop)

def is_yellow (n : ℕ) : Prop := bulb n = true
def is_blue (n : ℕ) : Prop := bulb n = false

theorem garland_bulbs :
  (is_yellow 3) ∧ (is_yellow 4) ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 96 → (∃ i j, n + i ≤ n + j ∧ i < j ∧ is_yellow (n + i) ∧ is_yellow (n + j) ∧
             ∀ k, k ≠ n + i ∧ k ≠ n + j ∧ n ≤ k ∧ k < n + 5 -> is_blue k)) ∧
  (bulb 97 = false) ∧ (bulb 98 = true) ∧ (bulb 99 = true) ∧ (bulb 100 = false) :=
sorry

end garland_bulbs_l587_587107


namespace bulb_colors_at_positions_97_to_100_l587_587108

def bulb : Type := ℕ → string

variable (b : bulb)

-- Definitions of given conditions
def c1 := ∀ n, n ∈ {3, 4} → b n = "Yellow"
def c2 := ∀ n, (finset.range 5).sum (λ i, if b (n + i) = "Yellow" then 1 else 0) = 2
def c3 := ∀ n, (finset.range 5).sum (λ i, if b (n + i) = "Blue" then 1 else 0) = 3

-- Combining all conditions
def conditions := c1 ∧ c2 ∧ c3

-- Theorem proving the final output
theorem bulb_colors_at_positions_97_to_100 (h : conditions) :
  b 97 = "Blue" ∧ b 98 = "Yellow" ∧ b 99 = "Yellow" ∧ b 100 = "Blue" :=
sorry

end bulb_colors_at_positions_97_to_100_l587_587108


namespace carolyn_sum_of_removals_l587_587900

-- Define the initial list
def initial_list := [1, 2, 3, 4, 5, 6, 7, 8]

-- Define the game rules based on the conditions

/-- Carolyn and Paul's game starting with integers from 1 to 8
    1. Carolyn always goes first.
    2. Carolyn and Paul alternate turns.
    3. On Carolyn's turn, she removes a number with at least one positive divisor or at least one proper multiple remaining in the list.
    4. On Paul's turn, he removes all the positive divisors of the number Carolyn has just removed.
    5. If Carolyn can't remove any more numbers, Paul removes the remaining numbers.
-/
def game (n : ℕ) := 
  n = 8 ∧ ∀ (turns : list ℕ), 
    -- Carolyn's removals
    (turns.head = 3) ∧  
    -- Second turn, Paul removes divisors of 3
    (turns[1] = 1) ∧ 
    -- Next, Carolyn can remove numbers with positive divisors or multiples
    -- Only 4, 5, 6, and 8 satisfy this condition after 3 and 1 are removed
    (turns[2] = 4 ∨ turns[2] = 5 ∨ turns[2] = 6 ∨ turns[2] = 8) ∧ 
    -- Example sequence leading to the sum 21 is 3, 4, 6, 8
    (∑ i in [turns.head, turns[2], turns[4], turns[6]], i) = 21

/-- Prove that Carolyn's removals sum to 21 given the game rules and her first removal being 3. -/
theorem carolyn_sum_of_removals : 
  ∃ (turns : list ℕ), game 8 → 
  (turns.head + turns[2] + turns[4] + turns[6] = 21) :=
sorry

end carolyn_sum_of_removals_l587_587900


namespace all_numbers_same_color_l587_587358

theorem all_numbers_same_color (n k : ℕ) (M : Finset ℕ) (color : ℕ → Prop) 
  (h1 : n > 0) (h2 : 0 < k) (h3 : k < n) (gcd_nk : Nat.gcd n k = 1)
  (hM : M = Finset.range n \ {0}) 
  (h_color : ∀ i ∈ M, (color i ↔ color (n - i))) 
  (h_diff : ∀ i ∈ M, i ≠ k → (color i ↔ color (Nat.abs (i - k)))) :
  ∀ i j ∈ M, color i = color j := 
sorry

end all_numbers_same_color_l587_587358


namespace arman_age_in_years_l587_587170

theorem arman_age_in_years (A S y : ℕ) (h1: A = 6 * S) (h2: S = 2 + 4) (h3: A + y = 40) : y = 4 :=
sorry

end arman_age_in_years_l587_587170


namespace food_needed_for_vacation_l587_587204

-- Define the conditions
def daily_food_per_dog := 250 -- in grams
def number_of_dogs := 4
def number_of_days := 14

-- Define the proof problem
theorem food_needed_for_vacation :
  (daily_food_per_dog * number_of_dogs * number_of_days / 1000) = 14 :=
by
  sorry

end food_needed_for_vacation_l587_587204


namespace find_p_value_l587_587977

noncomputable def poly (p : ℝ) : Polynomial ℝ := 2 * X^3 - 7 * X^2 + 7 * X + p

theorem find_p_value  :
  ∃ (p : ℝ), p = -2 ∧ (∃ (x k : ℝ), x ≠ 0 ∧ k ≠ 0 ∧ k ≠ 1 ∧ k ≠ -1 ∧
  (poly p).roots = Multiset.sort ℝ.le [x, k*x, k*k*x]) :=
sorry

end find_p_value_l587_587977


namespace pentagon_area_l587_587391

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end pentagon_area_l587_587391


namespace exist_k_stones_no_same_row_col_l587_587902

def Grid (m n : ℕ) := Fin m × Fin n

def stones (m n : ℕ) := Finset (Grid m n)

def removable_rows_columns (m n : ℕ) (k : ℕ) (S : stones m n) : Prop := 
  ∃ (rows : Finset (Fin m)) (cols : Finset (Fin n)),
    rows.card + cols.card = k ∧
    ∀ (r : Fin m) (c : Fin n), (r ∈ rows ∨ c ∈ cols) → (r, c) ∉ S

theorem exist_k_stones_no_same_row_col (m n : ℕ) (S : stones m n) (k : ℕ)
  (h : removable_rows_columns m n k S) :
  ∃ (T : Finset (Grid m n)), T.card = k ∧
  ∀ (p₁ p₂ : (Grid m n)), p₁ ∈ T → p₂ ∈ T → 
    (p₁.1 ≠ p₂.1 ∧ p₁.2 ≠ p₂.2 ∨ p₁ = p₂) :=
begin
  sorry
end

end exist_k_stones_no_same_row_col_l587_587902


namespace marco_older_than_twice_marie_l587_587729

variable (M m x : ℕ)

def marie_age : ℕ := 12
def sum_of_ages : ℕ := 37

theorem marco_older_than_twice_marie :
  m = marie_age → (M = 2 * m + x) → (M + m = sum_of_ages) → x = 1 :=
by
  intros h1 h2 h3
  rw [h1] at h2 h3
  sorry

end marco_older_than_twice_marie_l587_587729


namespace closest_to_sqrt13_is_4_l587_587166

theorem closest_to_sqrt13_is_4 : 
  ∀ a : ℕ, (a ∈ {2, 3, 4, 5} → abs ((a : ℝ) - real.sqrt 13) = min (abs (2 - real.sqrt 13)) (min (abs (3 - real.sqrt 13)) (min (abs (4 - real.sqrt 13)) (abs (5 - real.sqrt 13)))) → a = 4) :=
by
  intro a
  intro h1
  sorry

end closest_to_sqrt13_is_4_l587_587166


namespace youngest_oldest_sum_l587_587760

variable {b : Fin 6 → ℕ}

theorem youngest_oldest_sum (avg_age: 10) (median_age: 9) (sum_ages: ∑ i, b i = 60) 
  (median_cond: b 2 + b 3 = 18) : b 0 + b 5 = 18 :=
sorry

end youngest_oldest_sum_l587_587760


namespace compare_abc_l587_587955

def a : ℝ := 5 ^ 1.2
def b : ℝ := Real.log 6 / Real.log 0.2
def c : ℝ := 2 ^ 1.2

theorem compare_abc : a > c ∧ c > b := by
  sorry

end compare_abc_l587_587955


namespace sequence_sum_formula_l587_587430

open Nat

def sequence (n : ℕ) : ℚ := (2 * n - 1) + 1 / 2 ^ n

def sum_sequence (n : ℕ) : ℚ := (Finset.range n).sum (λ k => sequence (k + 1))

theorem sequence_sum_formula (n : ℕ) : sum_sequence n = n ^ 2 - 1 / 2 ^ n + 1 :=
by
  sorry

end sequence_sum_formula_l587_587430


namespace angle_is_20_l587_587980

theorem angle_is_20 (x : ℝ) (h : 180 - x = 2 * (90 - x) + 20) : x = 20 :=
by
  sorry

end angle_is_20_l587_587980


namespace sum_of_odd_power_coeffs_l587_587678

theorem sum_of_odd_power_coeffs (f : ℝ -> ℝ) :
  (∀ x : ℝ, f x = (1-x)^11) →
  (∃ c0 c1 c2 c3 c4 c5 c6 c7 c8 c9 c10 c11 : ℝ,
    f x = c0 + c1 * x + c2 * x^2 + c3 * x^3 + c4 * x^4 + c5 * x^5 + c6 * x^6 +
           c7 * x^7 + c8 * x^8 + c9 * x^9 + c10 * x^10 + c11 * x^11) →
  let a1 := c1, a3 := c3, a5 := c5, a7 := c7, a9 := c9, a11 := c11 in
  a1 + a3 + a5 + a7 + a9 + a11 = -2^10 :=
begin
  intros h1 h2,
  cases h2 with c0 h2,
  cases h2 with c1 h2,
  cases h2 with c2 h2,
  cases h2 with c3 h2,
  cases h2 with c4 h2,
  cases h2 with c5 h2,
  cases h2 with c6 h2,
  cases h2 with c7 h2,
  cases h2 with c8 h2,
  cases h2 with c9 h2,
  cases h2 with c10 h2,
  cases h2 with c11 h2,
  sorry
end

end sum_of_odd_power_coeffs_l587_587678


namespace divisible_by_6_l587_587489

theorem divisible_by_6 (n : ℕ) : 6 ∣ ((n - 1) * n * (n^3 + 1)) := sorry

end divisible_by_6_l587_587489


namespace count_solutions_congruence_l587_587262

theorem count_solutions_congruence (x : ℕ) (h1 : 0 < x ∧ x < 50) (h2 : x + 7 ≡ 45 [MOD 22]) : ∃ x1 x2, (x1 ≠ x2) ∧ (0 < x1 ∧ x1 < 50) ∧ (0 < x2 ∧ x2 < 50) ∧ (x1 + 7 ≡ 45 [MOD 22]) ∧ (x2 + 7 ≡ 45 [MOD 22]) ∧ (∀ y, (0 < y ∧ y < 50) ∧ (y + 7 ≡ 45 [MOD 22]) → (y = x1 ∨ y = x2)) :=
by {
  sorry
}

end count_solutions_congruence_l587_587262


namespace calculate_bankers_discount_l587_587424

noncomputable def present_worth : ℝ := 800
noncomputable def true_discount : ℝ := 36
noncomputable def face_value : ℝ := present_worth + true_discount
noncomputable def bankers_discount : ℝ := (face_value * true_discount) / (face_value - true_discount)

theorem calculate_bankers_discount :
  bankers_discount = 37.62 := 
sorry

end calculate_bankers_discount_l587_587424


namespace sam_won_total_matches_l587_587325

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end sam_won_total_matches_l587_587325


namespace jelly_bean_remaining_l587_587070

theorem jelly_bean_remaining (J : ℕ) (P : ℕ) (taken_last_4_each : ℕ) (taken_first_each : ℕ) 
 (taken_last_total : ℕ) (taken_first_total : ℕ) (taken_total : ℕ) (remaining : ℕ) :
  J = 8000 →
  P = 10 →
  taken_last_4_each = 400 →
  taken_first_each = 2 * taken_last_4_each →
  taken_last_total = 4 * taken_last_4_each →
  taken_first_total = 6 * taken_first_each →
  taken_total = taken_last_total + taken_first_total →
  remaining = J - taken_total →
  remaining = 1600 :=
by
  intros
  sorry  

end jelly_bean_remaining_l587_587070


namespace T_positive_l587_587258

theorem T_positive 
  (α : ℝ) 
  (h : ∀ k : ℤ, α ≠ k * (Real.pi / 2)) : 
  (sin α + tan α) / (cos α + cot α) > 0 := 
sorry

end T_positive_l587_587258


namespace y_coordinate_of_C_l587_587389

theorem y_coordinate_of_C (h : ℝ) (H : ∀ (C : ℝ), C = h) :
  let A := (0, 0)
      B := (0, 5)
      C := (3, h)
      D := (6, 5)
      E := (6, 0)
  -- Assuming the area of the pentagon is 50
  let area_square_ABDE := 25
      area_triangle_BCD := 25
  -- Assuming the height of triangle BCD
  let height_triangle_BCD := h - 5
      base_triangle_BCD := 6
      area_BCD := (1/2) * base_triangle_BCD * height_triangle_BCD in
  area_square_ABDE + area_triangle_BCD = 50 →
  area_BCD = area_triangle_BCD →
  h = 40 / 3 :=
by intros h H A B C D E area_square_ABDE area_triangle_BCD height_triangle_BCD base_triangle_BCD area_BCD;
   sorry

end y_coordinate_of_C_l587_587389


namespace option_C_incorrect_l587_587381

def sequence (a : ℕ → ℤ) := ∀ n : ℕ, a (n + 2) = 2 * a (n + 1) - a n

def partial_sum (a : ℕ → ℤ) (n : ℕ) : ℤ := ∑ i in range n, a i

variables (a : ℕ → ℤ)
variables (h_seq : sequence a)
variables (h_S5_S6 : partial_sum a 5 < partial_sum a 6)
variables (h_S6_S7_S8 : partial_sum a 6 = partial_sum a 7 ∧ partial_sum a 7 > partial_sum a 8)

theorem option_C_incorrect : ¬ (partial_sum a 9 > partial_sum a 5) :=
by
  sorry

end option_C_incorrect_l587_587381


namespace cost_of_7_enchiladas_and_6_tacos_l587_587396

theorem cost_of_7_enchiladas_and_6_tacos (e t : ℝ) 
  (h₁ : 4 * e + 5 * t = 5.00) 
  (h₂ : 6 * e + 3 * t = 5.40) : 
  7 * e + 6 * t = 7.47 := 
sorry

end cost_of_7_enchiladas_and_6_tacos_l587_587396


namespace allocation_methods_l587_587059

def staff := {A, B, C, D, E, F}

def condition_one (g1 g2 : set staff) : Prop :=
(A ∈ g1 ∧ B ∈ g1) ∨ (A ∈ g2 ∧ B ∈ g2)

def condition_two (g1 g2: set staff) : Prop :=
2 ≤ g1.card ∧ 2 ≤ g2.card

theorem allocation_methods : 
  ∃ (g1 g2 : set staff), condition_one g1 g2 ∧ condition_two g1 g2 ∧ (g1 ∪ g2 = staff ∧ g1 ≠ g2) → 22 := 
by sorry

end allocation_methods_l587_587059


namespace bulb_colors_at_positions_97_to_100_l587_587109

def bulb : Type := ℕ → string

variable (b : bulb)

-- Definitions of given conditions
def c1 := ∀ n, n ∈ {3, 4} → b n = "Yellow"
def c2 := ∀ n, (finset.range 5).sum (λ i, if b (n + i) = "Yellow" then 1 else 0) = 2
def c3 := ∀ n, (finset.range 5).sum (λ i, if b (n + i) = "Blue" then 1 else 0) = 3

-- Combining all conditions
def conditions := c1 ∧ c2 ∧ c3

-- Theorem proving the final output
theorem bulb_colors_at_positions_97_to_100 (h : conditions) :
  b 97 = "Blue" ∧ b 98 = "Yellow" ∧ b 99 = "Yellow" ∧ b 100 = "Blue" :=
sorry

end bulb_colors_at_positions_97_to_100_l587_587109


namespace garland_bulbs_l587_587106

variable (bulb : ℕ → Prop)

def is_yellow (n : ℕ) : Prop := bulb n = true
def is_blue (n : ℕ) : Prop := bulb n = false

theorem garland_bulbs :
  (is_yellow 3) ∧ (is_yellow 4) ∧
  (∀ n : ℕ, n ≥ 1 ∧ n ≤ 96 → (∃ i j, n + i ≤ n + j ∧ i < j ∧ is_yellow (n + i) ∧ is_yellow (n + j) ∧
             ∀ k, k ≠ n + i ∧ k ≠ n + j ∧ n ≤ k ∧ k < n + 5 -> is_blue k)) ∧
  (bulb 97 = false) ∧ (bulb 98 = true) ∧ (bulb 99 = true) ∧ (bulb 100 = false) :=
sorry

end garland_bulbs_l587_587106


namespace rotated_square_height_l587_587443

noncomputable def height_of_B (side_length : ℝ) (rotation_angle : ℝ) : ℝ :=
  let diagonal := side_length * Real.sqrt 2
  let vertical_component := diagonal * Real.sin rotation_angle
  vertical_component

theorem rotated_square_height :
  height_of_B 1 (Real.pi / 6) = Real.sqrt 2 / 2 :=
by
  sorry

end rotated_square_height_l587_587443


namespace sum_consecutive_even_integers_l587_587474

theorem sum_consecutive_even_integers (n : ℕ) (h : 2 * n + 4 = 156) : 
  n + (n + 2) + (n + 4) = 234 := 
by
  sorry

end sum_consecutive_even_integers_l587_587474


namespace machine_N_fraction_l587_587841
noncomputable theory

variables (x T_t T_n T_o : ℝ)
-- Conditions
axiom time_T : T_t = (3 / 4) * T_n
axiom time_O : T_o = (3 / 2) * T_n

-- Proof statement
theorem machine_N_fraction : 
  let R_t := x / T_t,
      R_n := x / T_n,
      R_o := x / T_o,
      R_total := R_t + R_n + R_o in
  (R_n / R_total) = (1 / 3) :=
by sorry

end machine_N_fraction_l587_587841


namespace sum_of_cubes_parity_l587_587227

-- Define the set of cubes from 1^3 to 2010^3
def cubes := { x | ∃ n : ℕ, 1 ≤ n ∧ n ≤ 2010 ∧ x = n^3 }

-- The number of even cubes is 1005
def even_cubes_count := ∃ evens : Finset ℕ, evens.card = 1005 ∧ ∀ x ∈ evens, x % 2 = 0

-- The number of odd cubes is 1005
def odd_cubes_count := ∃ odds : Finset ℕ, odds.card = 1005 ∧ ∀ x ∈ odds, x % 2 = 1

-- The main theorem statement
theorem sum_of_cubes_parity : even_cubes_count ∧ odd_cubes_count → (∀ f : ℕ → ℤ, (∑ i in Finset.range 2010, if (i + 1 % 2 = 0) then f ((i + 1)^3) else f (-(i + 1)^3)) % 2 = 1) :=
by
  intros h
  sorry

end sum_of_cubes_parity_l587_587227


namespace average_weighted_score_l587_587125

theorem average_weighted_score
  (score1 score2 score3 : ℕ)
  (weight1 weight2 weight3 : ℕ)
  (h_scores : score1 = 90 ∧ score2 = 85 ∧ score3 = 80)
  (h_weights : weight1 = 5 ∧ weight2 = 2 ∧ weight3 = 3) :
  (weight1 * score1 + weight2 * score2 + weight3 * score3) / (weight1 + weight2 + weight3) = 86 := 
by
  sorry

end average_weighted_score_l587_587125


namespace magnitude_of_product_l587_587560

-- Variables and conditions definition
def z1 : ℂ := 5 - 3 * complex.I
def z2 : ℂ := 7 + 24 * complex.I

-- The theorem statement
theorem magnitude_of_product (z1 z2 : ℂ) (h1 : z1 = 5 - 3 * complex.I) (h2 : z2 = 7 + 24 * complex.I) : 
  |z1 * z2| = 25 * (√34) :=
by
  rw [h1, h2]
  -- Sorry for not providing the complete proof
  sorry

end magnitude_of_product_l587_587560


namespace total_number_of_ways_l587_587856

theorem total_number_of_ways (num_courses_A : ℕ) (num_courses_B : ℕ) (h1 : num_courses_A = 4) (h2 : num_courses_B = 3) :
  num_courses_A + num_courses_B = 7 :=
by {
  rw [h1, h2],
  exact Nat.add_eq_zero_iff.mpr (Nat.add_eq_succ_iff.mpr (by refl)),
}

end total_number_of_ways_l587_587856


namespace bet_not_fair_l587_587492

noncomputable def expected_gain_A (A_bet B_bet : ℕ) (prob_A_win prob_B_win : ℚ) : ℚ :=
  prob_A_win * A_bet - prob_B_win * B_bet

theorem bet_not_fair (A_bet : ℕ := 10) (B_bet : ℕ := 8) : 
  let prob_A_win := 7 / 12,
      prob_B_win := 5 / 12
  in expected_gain_A A_bet B_bet prob_A_win prob_B_win ≠ 0 ∧ 
     (∃ (new_B_bet : ℚ), expected_gain_A A_bet new_B_bet prob_A_win prob_B_win = 0 ∧ new_B_bet = 50 / 7) := 
begin
  sorry
end

end bet_not_fair_l587_587492


namespace value_2_std_dev_less_than_mean_l587_587104

-- Define the mean and standard deviation as constants
def mean : ℝ := 14.5
def std_dev : ℝ := 1.5

-- State the theorem (problem)
theorem value_2_std_dev_less_than_mean : (mean - 2 * std_dev) = 11.5 := by
  sorry

end value_2_std_dev_less_than_mean_l587_587104


namespace volume_of_tetrahedron_l587_587928

theorem volume_of_tetrahedron (A B C D : Type) 
    (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle : ℝ)
    (h1 : area_ABC = 150)
    (h2 : area_BCD = 90)
    (h3 : BC = 10)
    (h4 : angle = π / 4) :
    ∃ V : ℝ, V = 450 * real.sqrt 2 :=
begin
  sorry
end

end volume_of_tetrahedron_l587_587928


namespace find_a_equiv_l587_587565

theorem find_a_equiv (a x : ℝ) (h : ∀ x, (a * x^2 + 20 * x + 25) = (2 * x + 5) * (2 * x + 5)) : a = 4 :=
by
  sorry

end find_a_equiv_l587_587565


namespace polynomial_coeff_sum_l587_587240

theorem polynomial_coeff_sum (a_0 a_1 : ℚ) (a_2 a_3 ... a_2017: ℚ) :
  (a_0 + a_1 + a_2 + ... + a_2017 = ∑ i in finset.range 2018) :
    ∀ x : ℚ,
    (x^2 - 3) * (2 * x + 3) ^ 2015 =
    a_0 + a_1 * (x + 2) + a_2 * (x + 2)^2 + ... + a_2017 * (x + 2) ^ 2017) 
    → (a_0 = -1)
    → a_1 + a_2 + ... + a_2017 = -1 :=
by
  sorry

end polynomial_coeff_sum_l587_587240


namespace simplified_expression_correct_l587_587028

noncomputable def simplify_expression (x : ℝ) : ℝ :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 9) * (x^2 - 6*x + 8) / (x^2 - 8*x + 15)

theorem simplified_expression_correct (x : ℝ) :
  simplify_expression x = ((x - 1) * (x - 2) * (x - 4)) / ((x - 3) * (x - 5)) :=
  sorry

end simplified_expression_correct_l587_587028


namespace volume_ratio_of_centroid_tetrahedron_l587_587893

noncomputable def centroid (points : List (ℝ × ℝ × ℝ)) : ℝ × ℝ × ℝ :=
  let (x_sum, y_sum, z_sum) := points.foldr (λ (p : ℝ × ℝ × ℝ) (a : ℝ × ℝ × ℝ) =>
    (a.1 + p.1, a.2 + p.2, a.3 + p.3)) (0, 0, 0)
  (x_sum / points.length, y_sum / points.length, z_sum / points.length)

noncomputable def volume (A B C D : ℝ × ℝ × ℝ) : ℝ :=
  let ax := A.1 - D.1; ay := A.2 - D.2; az := A.3 - D.3
  let bx := B.1 - D.1; by := B.2 - D.2; bz := B.3 - D.3
  let cx := C.1 - D.1; cy := C.2 - D.2; cz := C.3 - D.3
  (ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)) / 6

theorem volume_ratio_of_centroid_tetrahedron
  (A B C D P : ℝ × ℝ × ℝ)
  (P_interior : -- condition that P is an interior point of ABCD, which we will assume as a placeholder here): 
  volume (centroid [P, A, B, C]) (centroid [P, B, C, D]) (centroid [P, C, D, A]) (centroid [P, D, A, B]) =
  volume A B C D / 64 :=
sorry

end volume_ratio_of_centroid_tetrahedron_l587_587893


namespace determine_m_of_monotonically_increasing_function_l587_587721

theorem determine_m_of_monotonically_increasing_function 
  (m n : ℝ)
  (h : ∀ x, 12 * x ^ 2 + 2 * m * x + (m - 3) ≥ 0) :
  m = 6 := 
by 
  sorry

end determine_m_of_monotonically_increasing_function_l587_587721


namespace minimum_g_a_l587_587267

noncomputable def f (x a : ℝ) : ℝ := x ^ 2 + 2 * a * x + 3

noncomputable def g (a : ℝ) : ℝ := 3 * a ^ 2 + 2 * a

theorem minimum_g_a : ∀ a : ℝ, a ≤ -1 → g a = 3 * a ^ 2 + 2 * a → g a ≥ 1 := by
  sorry

end minimum_g_a_l587_587267


namespace nested_square_root_eq_two_l587_587037

theorem nested_square_root_eq_two : ∃ y : ℝ, y = 2 ∧ y = real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + real.sqrt (2 + ⬝⬝⬝))))) := 
sorry

end nested_square_root_eq_two_l587_587037


namespace max_value_2ab_plus_2ac_sqrt3_l587_587375

variable (a b c : ℝ)
variable (h1 : a^2 + b^2 + c^2 = 1)
variable (h2 : 0 ≤ a)
variable (h3 : 0 ≤ b)
variable (h4 : 0 ≤ c)

theorem max_value_2ab_plus_2ac_sqrt3 : 2 * a * b + 2 * a * c * Real.sqrt 3 ≤ 1 := by
  sorry

end max_value_2ab_plus_2ac_sqrt3_l587_587375


namespace sequence_square_terms_l587_587289

theorem sequence_square_terms (k : ℤ) (y : ℕ → ℤ) 
  (h1 : y 1 = 1)
  (h2 : y 2 = 1)
  (h3 : ∀ n ≥ 1, y (n + 2) = (4 * k - 5) * y (n + 1) - y n + 4 - 2 * k) :
  (∀ n, ∃ m : ℤ, y n = m ^ 2) ↔ k = 3 :=
by sorry

end sequence_square_terms_l587_587289


namespace parabola_tangent_sum_l587_587800

theorem parabola_tangent_sum (m n : ℕ) (hmn_coprime : Nat.gcd m n = 1)
    (h_tangent : ∃ (k : ℝ), ∀ (x y : ℝ), y = 4 * x^2 ↔ x = y^2 + (m / n)) :
    m + n = 19 :=
by
  sorry

end parabola_tangent_sum_l587_587800


namespace part_I_part_II_l587_587275

noncomputable def f (ω x : ℝ) : ℝ :=
  cos (ω * x) * sin (ω * x - π / 3) + sqrt 3 * (cos (ω * x))^2 - sqrt 3 / 4

-- Part (I)
theorem part_I (hω₀ : ω > 0) :
  (∃ k : ℤ, ∀ x : ℝ, f 1 x = f 1 (x - k * (π / 2))) ∧
  (dist (π / 12 + k * (π / 2)) (π / 12) = π / 4) :=
sorry

-- Part (II)
theorem part_II (A C a b : ℝ) (hA : 0 < A ∧ A < π) (hC : sin C = 1/3) (ha : a = sqrt 3) (hfA : f 1 A = sqrt 3 / 4) :
  A = π / 6 ∧ b = (3 + 2 * sqrt 6) / 3 :=
sorry

end part_I_part_II_l587_587275


namespace intersection_result_complement_union_result_l587_587382

open Set

def A : Set ℝ := {x | -1 < x ∧ x < 2}
def B : Set ℝ := {x | x > 0}

theorem intersection_result : A ∩ B = {x | 0 < x ∧ x < 2} :=
by
  sorry

theorem complement_union_result : (compl B) ∪ A = {x | x < 2} :=
by
  sorry

end intersection_result_complement_union_result_l587_587382


namespace smallest_m_for_integral_solutions_l587_587824

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), 10 * x^2 - m * x + 180 = 0 → x ∈ ℤ) ∧ m = 90 :=
by
  sorry

end smallest_m_for_integral_solutions_l587_587824


namespace molecular_weight_of_compound_l587_587821

noncomputable def atomic_weight_C : ℝ := 12.01
noncomputable def atomic_weight_H : ℝ := 1.008
noncomputable def atomic_weight_O : ℝ := 16.00

def molecular_weight (num_C num_H num_O : ℕ) : ℝ :=
  num_C * atomic_weight_C + num_H * atomic_weight_H + num_O * atomic_weight_O

theorem molecular_weight_of_compound :
  molecular_weight 7 6 2 = 122.118 :=
by
  -- sorry will allow us to skip the proof
  sorry

end molecular_weight_of_compound_l587_587821


namespace sequence_integer_and_divisibility_l587_587011

-- Given conditions
def a (n : ℕ) : ℤ := (1 / (2 * Real.sqrt 3 : ℚ) * ((2 + Real.sqrt 3 : ℝ)^n - (2 - Real.sqrt 3 : ℝ)^n)).toInt

-- Statement of the theorem proving the question == answer given conditions
theorem sequence_integer_and_divisibility (n : ℕ) :
  (∀ n : ℕ, ∃ k : ℤ, a n = k) ∧ (3 ∣ a n ↔ 3 ∣ n) :=
by
  sorry

end sequence_integer_and_divisibility_l587_587011


namespace actual_distance_map_l587_587004

theorem actual_distance_map (scale : ℕ) (map_distance : ℕ) (actual_distance_km : ℕ) (h1 : scale = 500000) (h2 : map_distance = 4) :
  actual_distance_km = 20 :=
by
  -- definitions and assumptions
  let actual_distance_cm := map_distance * scale
  have cm_to_km_conversion : actual_distance_km = actual_distance_cm / 100000 := sorry
  -- calculation
  have actual_distance_sol : actual_distance_cm = 4 * 500000 := sorry
  have actual_distance_eq : actual_distance_km = (4 * 500000) / 100000 := sorry
  -- final answer
  have answer_correct : actual_distance_km = 20 := sorry
  exact answer_correct

end actual_distance_map_l587_587004


namespace growth_rate_l587_587410

variable (x : ℝ)

def initial_investment : ℝ := 500
def expected_investment : ℝ := 720

theorem growth_rate (x : ℝ) (h : 500 * (1 + x)^2 = 720) : x = 0.2 :=
by
  sorry

end growth_rate_l587_587410


namespace jerry_removed_figures_l587_587347

-- Definitions based on conditions
def initialFigures : ℕ := 3
def addedFigures : ℕ := 4
def currentFigures : ℕ := 6

-- Total figures after adding
def totalFigures := initialFigures + addedFigures

-- Proof statement defining how many figures were removed
theorem jerry_removed_figures : (totalFigures - currentFigures) = 1 := by
  sorry

end jerry_removed_figures_l587_587347


namespace sequence_arith_l587_587060

theorem sequence_arith {a : ℕ → ℕ} (h_initial : a 2 = 2) (h_recursive : ∀ n ≥ 2, a (n + 1) = a n + 1) :
  ∀ n ≥ 2, a n = n :=
by
  sorry

end sequence_arith_l587_587060


namespace roxy_daily_water_intake_l587_587796

theorem roxy_daily_water_intake 
  (theo_daily : ℕ) (mason_daily : ℕ) (total_weekly : ℕ) 
  (theo_daily = 8) (mason_daily = 7)
  (total_weekly = 168) : 
  ∃ (roxy_daily : ℕ), roxy_daily = 9 := 
by
  sorry

end roxy_daily_water_intake_l587_587796


namespace evaluate_expression_l587_587360

def P (x : ℝ) : ℝ := 3 * x^(1/3)
def Q (x : ℝ) : ℝ := x^3

theorem evaluate_expression : P (Q (P (Q (P (Q 4))))) = 108 :=
by
  -- First apply the nested functions step by step 
  have h1 : P (Q 4) = 12 := by
    unfold P Q
    simp
    -- using the property of cube and cube root
    norm_num
  have h2 : P (Q (P (Q 4))) = 36 := by
    rw [h1]
    unfold P Q
    norm_num
  have h3 : P (Q (P (Q (P (Q 4))))) = 108 := by
    rw [h2]
    unfold P Q
    norm_num

  -- Apply the final result to the theorem
  exact h3

end evaluate_expression_l587_587360


namespace solve_equation_l587_587218

theorem solve_equation : 
  ∀ x : ℝ, 
  (((15 * x - x^2) / (x + 2)) * (x + (15 - x) / (x + 2)) = 54) → (x = 9 ∨ x = -1) :=
by
  sorry

end solve_equation_l587_587218


namespace john_bought_2_packs_of_gum_l587_587691

-- Given conditions
def cost_of_candy_bar : ℝ := 1.5
def number_of_candy_bars : ℕ := 3
def total_money_paid : ℝ := 6
def cost_of_gum : ℝ := cost_of_candy_bar / 2

-- Assertion we want to prove
theorem john_bought_2_packs_of_gum (G : ℕ) (cost_of_gum : ℝ) (total_money_spent_on_gum : ℝ) 
  (total_money_spent_on_candy : ℝ) (total_money_paid : ℝ) :
  total_money_paid = total_money_spent_on_candy + total_money_spent_on_gum ∧
  total_money_spent_on_candy = number_of_candy_bars * cost_of_candy_bar ∧
  total_money_spent_on_gum = G * cost_of_gum ∧
  cost_of_gum = cost_of_candy_bar / 2 ∧
  total_money_paid = 6 ∧
  cost_of_candy_bar = 1.5 ∧
  number_of_candy_bars = 3
  → G = 2 :=
by
  sorry

end john_bought_2_packs_of_gum_l587_587691


namespace correct_statements_l587_587581

variables {R : Type*} [linear_ordered_field R] (f : R → R)

def is_odd (f : R → R) : Prop :=
  ∀ x, f (-x) = -f x

def is_symmetric_about_point_A (f : R → R) : Prop :=
  ∀ x, f (x-1) = f (2-x)  -- Symmetry about point A(1,0).

def condition_one (f : R → R) (h_oddf : is_odd f) : Prop :=
  is_symmetric_about_point_A f

def condition_two (f : R → R) : Prop :=
  ∀ x, f (x - 1) = f (x + 1)

def result_two (f : R → R) (h_sym : condition_two f) : Prop :=
  ∀ x, f (x) = f (x + 2)

def graphs_symmetric_about_line (f : R → R) : Prop :=
  ∀ x, f (x + 1) = f (1 - x)

def condition_four (f : R → R) : Prop :=
  f (x + 1) = f (1 - x) ∧ f (x + 3) = f (3 - x)

def period_of_4 (f : R → R) : Prop :=
  ∀ x, f (x) = f (x + 4)

def statement_1 := condition_one f (is_odd f)
def statement_2 := result_two f (condition_two f)
def statement_3 := graphs_symmetric_about_line f
def statement_4 := period_of_4 f

theorem correct_statements (h_oddf : is_odd f) (h_sym : condition_two f) (h_cond4 : condition_four f) :
  statement_1 ∧ statement_2 ∧ statement_4 :=
by {
  sorry -- The detailed proof would go here.
}

end correct_statements_l587_587581


namespace find_y_l587_587646

theorem find_y (x y : ℤ) (h1 : x - y = 8) (h2 : x + y = 14) : y = 3 := 
by sorry

end find_y_l587_587646


namespace largest_distance_l587_587818

-- Define the points and radii
def point1 : ℝ × ℝ × ℝ := (3, -7, 10)
def point2 : ℝ × ℝ × ℝ := (-15, 20, -10)
def radius1 : ℝ := 15
def radius2 : ℝ := 40

-- Define the distance formula
def dist (p1 p2 : ℝ × ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2 + (p1.3 - p2.3) ^ 2)

-- Define the largest possible distance
theorem largest_distance :
  radius1 + dist point1 point2 + radius2 = 55 + Real.sqrt 1453 := by
  sorry

end largest_distance_l587_587818


namespace ordered_triples_count_l587_587875

theorem ordered_triples_count :
  let b := 1995 in 
  (∃ (a c : ℕ), a ≤ b ∧ b ≤ c ∧ a * c = b^2) →
  (set.count { (a, c) | a ≤ b ∧ b ≤ c ∧ a * c = b^2 }) = 40 :=
sorry

end ordered_triples_count_l587_587875


namespace lines_parallel_distance_l587_587079

theorem lines_parallel_distance (a b : ℝ)
  (h_parallel : ∀ x y : ℝ, ax + 2 * y + b = 0 → (a - 1) * x + y + b = 0)
  (h_distance : ∀ x y : ℝ, (|b| / real.sqrt 8) = real.sqrt 2 / 2) :
  a * b = 4 ∨ a * b = -4 :=
sorry

end lines_parallel_distance_l587_587079


namespace intervals_of_monotonicity_number_of_zeros_of_f_when_k_gt_0_l587_587723

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (x - 1) * Real.exp x - (k / 2) * x^2

theorem intervals_of_monotonicity (k : ℝ) : 
  (k ≤ 0 → monotone_on (λ x, f k x) {x | x > 0}) ∧ 
  (0 < k ∧ k < 1 → 
    (monotone_on (λ x, f k x) {x | x < Real.log k} ∧ monotone_on (λ x, f k x) {x | x > 0}) ∧ 
    antitone_on (λ x, f k x) {x | Real.log k ≤ x ∧ x ≤ 0}) ∧ 
  (k = 1 → monotone_on (λ x, f k x) {x})) ∧ 
  (k > 1 → 
    (monotone_on (λ x, f k x) {x | x < 0} ∧ monotone_on (λ x, f k x) {x | x > Real.log k}) ∧ 
    antitone_on (λ x, f k x) {x | 0 ≤ x ∧ x ≤ Real.log k}) := 
sorry

theorem number_of_zeros_of_f_when_k_gt_0 (k : ℝ) (hk : k > 0) : 
  ∃! x : ℝ, f k x = 0 := 
sorry

end intervals_of_monotonicity_number_of_zeros_of_f_when_k_gt_0_l587_587723


namespace smallest_prime_divisor_of_sum_l587_587468

theorem smallest_prime_divisor_of_sum (a b : ℕ) 
  (h₁ : a = 3 ^ 15) 
  (h₂ : b = 11 ^ 21) 
  (h₃ : odd a) 
  (h₄ : odd b) : 
  nat.prime_divisors (a + b) = [2] := 
by
  sorry

end smallest_prime_divisor_of_sum_l587_587468


namespace find_m_l587_587146

variable (seq : List Int)
variable {m : Int}

def mode (l : List Int) : Int :=
  l.groupBy id |>.maxBy (·.length) |>.headI! |>.head!

def median (l : List Int) : Int :=
  let sorted := l.qsort (≤)
  if h : sorted.length % 2 = 1 then
    sorted.get ⟨sorted.length / 2, Nat.div_lt_self (sorted.length % 2).pos dec_trivial⟩
  else
    0  -- In a real scenario, we would handle even-lengthed lists differently, but our conditions specify an odd length.

def mean (l : List Int) : Float :=
  (l.sum.toFloat) / (l.length.toFloat)

theorem find_m :
  mode seq = 32 ∧ mean seq = 22 ∧ List.minimum seq = some 10 ∧ median seq = m ∧
  (median (seq.map (λ x → if x = m then m + 10 else x)) = m + 10 ∧ mean (seq.map (λ x → if x = m then m + 10 else x)) = 24) ∧
  median (seq.map (λ x → if x = m then m - 8 else x)) = m - 4 →
  m = 20 := by
sorry

end find_m_l587_587146


namespace height_of_tank_B_l587_587767

noncomputable def height_tank_A : ℝ := 5
noncomputable def circumference_tank_A : ℝ := 4
noncomputable def circumference_tank_B : ℝ := 10
noncomputable def capacity_ratio : ℝ := 0.10000000000000002

theorem height_of_tank_B {h_B : ℝ} 
  (h_tank_A : height_tank_A = 5)
  (c_tank_A : circumference_tank_A = 4)
  (c_tank_B : circumference_tank_B = 10)
  (capacity_percentage : capacity_ratio = 0.10000000000000002)
  (V_A : ℝ := π * (2 / π)^2 * height_tank_A)
  (V_B : ℝ := π * (5 / π)^2 * h_B)
  (capacity_relation : V_A = capacity_ratio * V_B) :
  h_B = 8 :=
sorry

end height_of_tank_B_l587_587767


namespace number_of_valid_numbers_l587_587296

-- Definitions based on the given conditions
def valid_a (a : ℕ) : Prop := a ∈ {3, 4, 5}
def valid_d (d : ℕ) : Prop := d ∈ {0, 4, 8}
def valid_bc_pair (b c : ℕ) : Prop := 3 ≤ b ∧ b < c ∧ c ≤ 6

-- Statement to prove the total number of valid four-digit numbers N
theorem number_of_valid_numbers : 
  let count_a := 3 in
  let count_d := 3 in
  let count_bc_pairs := 6 in
  count_a * count_d * count_bc_pairs = 54 :=
by
  sorry

end number_of_valid_numbers_l587_587296


namespace angle_FCH_eq_angle_GDH_l587_587892

variable (P Q : Circle) (A B C D E F G H : Point)
variables (hPQ_intersect : P ∩ Q = {A, B})
variables (hCD_cdtangentP : tangent C D P)
variables (hCD_cdtangentQ : tangent C D Q)
variables (hE_extBA : collinear E B A)
variables (hF_onEC : E ∈ EC)
variables (hG_onED : E ∈ ED)
variables (hH_bisect: bisects_angle (A, H, F) (A, H, G))
variables (hH_onFG : collinear H F G)

theorem angle_FCH_eq_angle_GDH:
  ∠ (F, C, H) = ∠ (G, D, H) := 
sorry

end angle_FCH_eq_angle_GDH_l587_587892


namespace volume_of_tetrahedron_l587_587926

theorem volume_of_tetrahedron 
(angle_ABC_BCD : Real := 45 * Real.pi / 180)
(area_ABC : Real := 150)
(area_BCD : Real := 90)
(length_BC : Real := 10) :
  let h := 2 * area_BCD / length_BC
  let height_perpendicular := h * Real.sin angle_ABC_BCD
  let volume := (1 / 3 : Real) * area_ABC * height_perpendicular
  volume = 450 * Real.sqrt 2 :=
by
  sorry

end volume_of_tetrahedron_l587_587926


namespace find_a_for_minimum_value_l587_587628

noncomputable def f (x a : ℝ) : ℝ := Real.exp x + x^2 + (3 * a + 2) * x

theorem find_a_for_minimum_value :
  (∃ x ∈ Ioo (-1:ℝ) (0:ℝ), ∀ y ∈ Ioo (-1:ℝ) (0:ℝ), (f y a ≥ f x a)) ↔ -1 < a ∧ a < -1 / (3 * Real.exp 1) :=
by
  sorry

end find_a_for_minimum_value_l587_587628


namespace change_difference_l587_587531

-- Define initial and final distributions
def initial_percentages := {yes := 40, no := 40, unsure := 20}
def final_percentages := {yes := 60, no := 20, unsure := 20}

-- Define the difference in percentages of students changing their responses
def min_change := 20
def max_change := 60

-- Prove that the difference between maximum and minimum possible changes is 40%
theorem change_difference : (max_change - min_change) = 40 :=
by
  -- include some steps if needed
  sorry

end change_difference_l587_587531


namespace tetrahedra_volume_relationship_l587_587788

-- Definitions based on the problem conditions
def Tet (A B C S : Type*) := (A B C S : Type*)

variables {A B C S : Type*}

-- Midpoints of the base edges
def midpoint (x y : Type*) := (x + y) / 2

-- Centroids of the faces
def centroid_face (a b c : Type*) := (a + b + c) / 3

-- Centroid of the tetrahedron
def centroid_tetrahedron (a b c s : Type*) := (a + b + c + s) / 4

-- Length ratio property
def length_ratio (x y : Type*) := (x / y) = 1 / 3

-- Volume ratio
def volume (a b c s : Type*) := abs (a * b * c * s) / 6

theorem tetrahedra_volume_relationship :
  ∃ (A B C S : Type*), 
  ∀ A' B' C' A1 B1 C1 S1, 
    midpoint A B = A' ∧ midpoint B C = B' ∧ midpoint C A = C' →
    centroid_face A B S = A1 ∧ centroid_face B C S = B1 ∧ centroid_face C A S = C1 →
    centroid_tetrahedron A B C S = S1 →
    length_ratio A'S A'A ∧ length_ratio B'S B'B ∧ length_ratio C'S C'C →
    volume S A B C = 729 * volume S1 A1 B1 C1 :=
sorry

end tetrahedra_volume_relationship_l587_587788


namespace height_of_shorter_tree_l587_587063

theorem height_of_shorter_tree (H h : ℝ) (h_difference : H = h + 20) (ratio : h / H = 5 / 7) : h = 50 := 
by
  sorry

end height_of_shorter_tree_l587_587063


namespace ellipse_equation_l587_587937

open Real

theorem ellipse_equation (x y : ℝ) (h₁ : (- sqrt 15) = x) (h₂ : (5 / 2) = y)
  (h₃ : ∃ (a b : ℝ), (a > b) ∧ (b > 0) ∧ (a^2 = b^2 + 5) 
  ∧ b^2 = 20 ∧ a^2 = 25) :
  (x^2 / 20 + y^2 / 25 = 1) :=
sorry

end ellipse_equation_l587_587937


namespace dilation_image_l587_587048

theorem dilation_image :
  let z_0 := (1 : ℂ) + 2 * I
  let k := (2 : ℂ)
  let z_1 := (3 : ℂ) + I
  let z := z_0 + k * (z_1 - z_0)
  z = 5 :=
by
  sorry

end dilation_image_l587_587048


namespace intersection_segment_length_is_correct_l587_587129

noncomputable def length_of_intersection_segment (rect_length : ℝ) (rect_width : ℝ) := 
  let r := min (rect_length / 2) (rect_width / 2)
  let (x1, y1) := (1, 0)
  let (x2, y2) := (-3/5, -4/5)
  real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

theorem intersection_segment_length_is_correct :
  length_of_intersection_segment 4 2 = (4 * real.sqrt 5) / 5 :=
by
  sorry

end intersection_segment_length_is_correct_l587_587129


namespace total_matches_won_l587_587321

-- Condition definitions
def matches1 := 100
def win_percentage1 := 0.5
def matches2 := 100
def win_percentage2 := 0.6

-- Theorem statement
theorem total_matches_won : matches1 * win_percentage1 + matches2 * win_percentage2 = 110 :=
by
  sorry

end total_matches_won_l587_587321


namespace percent_reduction_price_of_apples_l587_587517

theorem percent_reduction_price_of_apples :
  ∃ (P R : ℝ) (P_reduced_per_dozen R : ℝ),
  (R = 3 ∧
  (P_reduced_per_dozen * P = 40 ∧
  P = 80 / 8) ∧
  (64 / 12 = P_reduced_per_dozen - P)) →
  ((P - R) / P * 100 = 40) :=
begin
  sorry
end

end percent_reduction_price_of_apples_l587_587517


namespace cat_finishes_food_by_next_wednesday_l587_587021

theorem cat_finishes_food_by_next_wednesday:
  (morning_consumption: ℚ) (evening_consumption: ℚ)
  (initial_cans: ℕ) (half_can: ℚ) (final_day: ℕ):
  morning_consumption = 1/4 ->
  evening_consumption = 1/6 ->
  initial_cans = 10 ->
  half_can = 1/2 ->
  final_day = 8 ->  -- Tuesday is represented by day 8
  (∃ total_days: ℚ, total_days = 2 ∨ (
    ∀ days_after_tuesday: ℚ,
    days_after_tuesday = total_days - 2 →
    (days_after_tuesday + 2 * (morning_consumption + evening_consumption) +
    days_after_tuesday * (morning_consumption + evening_consumption) = initial_cans + half_can → 
    total_days = final_day))) :=
by
  intros morning_consumption evening_consumption initial_cans half_can final_day
  intro h1 h2 h3 h4 h5
  sorry

end cat_finishes_food_by_next_wednesday_l587_587021


namespace find_total_students_l587_587117

variables (x X : ℕ)
variables (x_percent_students : ℕ) (total_students : ℕ)
variables (boys_fraction : ℝ)

-- Provided Conditions
axiom a1 : x_percent_students = 120
axiom a2 : boys_fraction = 0.30
axiom a3 : total_students = X

-- The theorem we need to prove
theorem find_total_students (a1 : 120 = x_percent_students) 
                            (a2 : boys_fraction = 0.30) 
                            (a3 : total_students = X) : 
  120 = (x / 100) * (boys_fraction * total_students) :=
sorry

end find_total_students_l587_587117


namespace distinct_ordered_pairs_l587_587986

theorem distinct_ordered_pairs (a b : ℕ) (h : a + b = 40) (ha : a > 0) (hb : b > 0) :
  ∃ (pairs : Finset (ℕ × ℕ)), pairs.card = 39 ∧ ∀ p ∈ pairs, p.1 + p.2 = 40 := 
sorry

end distinct_ordered_pairs_l587_587986


namespace greater_solution_of_quadratic_l587_587084

theorem greater_solution_of_quadratic :
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ (x₁^2 - 5 * x₁ - 84 = 0) ∧ (x₂^2 - 5 * x₂ - 84 = 0) ∧ (max x₁ x₂ = 12) :=
by
  sorry

end greater_solution_of_quadratic_l587_587084


namespace find_point_C_l587_587936

theorem find_point_C :
  ∃ z : ℚ, let C := (0, 0, z) in
  dist (C : ℚ × ℚ × ℚ) (-4, 1, 7) = dist (C : ℚ × ℚ × ℚ) (3, 5, -2) ∧ z = 14 / 9 :=
begin
  sorry
end

end find_point_C_l587_587936


namespace johanna_loses_half_turtles_l587_587753

theorem johanna_loses_half_turtles
  (owen_turtles_initial : ℕ)
  (johanna_turtles_fewer : ℕ)
  (owen_turtles_after_month : ℕ)
  (owen_turtles_final : ℕ)
  (johanna_donates_rest_to_owen : ℚ → ℚ)
  (x : ℚ)
  (hx1 : owen_turtles_initial = 21)
  (hx2 : johanna_turtles_fewer = 5)
  (hx3 : owen_turtles_after_month = owen_turtles_initial * 2)
  (hx4 : owen_turtles_final = owen_turtles_after_month + johanna_donates_rest_to_owen (1 - x))
  (hx5 : owen_turtles_final = 50) :
  x = 1 / 2 :=
by
  sorry

end johanna_loses_half_turtles_l587_587753


namespace volume_of_tetrahedron_equals_450_sqrt_2_l587_587922

-- Given conditions
variables {A B C D : Point}
variables (areaABC areaBCD : ℝ) (BC : ℝ) (angleABC_BCD : ℝ)

-- The specific values for the conditions
axiom h_areaABC : areaABC = 150
axiom h_areaBCD : areaBCD = 90
axiom h_BC : BC = 10
axiom h_angleABC_BCD : angleABC_BCD = π / 4  -- 45 degrees in radians

-- Definition of the volume to be proven
def volume_tetrahedron (A B C D : Point) : ℝ :=
  (1 / 3) * areaABC * (18 * real.sin angleABC_BCD)

-- Final proof statement
theorem volume_of_tetrahedron_equals_450_sqrt_2 :
  volume_tetrahedron A B C D = 450 * real.sqrt 2 :=
by 
  -- Preliminary setup, add the relevant properties and results
  sorry

end volume_of_tetrahedron_equals_450_sqrt_2_l587_587922


namespace greatest_n_Tn_perfect_square_l587_587226

def h (x : ℕ) : ℕ :=
  if h: x % 4 = 0 then
    let maxPower := (Nat.find (λ p, x / 4^p % 4 ≠ 0) - 1)
    4^(maxPower + 1)
  else
    0

noncomputable def T_n (n : ℕ) : ℕ :=
  ∑ k in Finset.range (2^(n-1)), h(4*(k+1))

theorem greatest_n_Tn_perfect_square (h : ∀ n < 500, T_n n = 4^(n-1) * (n+1)) : ∃ n < 500, T_n n = 143 :=
  sorry -- Proof will be filled in later

end greatest_n_Tn_perfect_square_l587_587226


namespace solution_y_alcohol_percentage_l587_587031

theorem solution_y_alcohol_percentage :
  (let Px := 10
       Vx := 50
       Vy := 150
       Pf := 25
       Vf := Vx + Vy
       Af := (Pf * Vf) / 100
       Ax := (Px * Vx) / 100
       Ay := (Af - Ax)
       Py := (Ay * 100) / Vy
   in Py) = 30 :=
begin
  sorry
end

end solution_y_alcohol_percentage_l587_587031


namespace midpoint_equidistant_l587_587604

variables {A B C D P Q R S T : Type*} [geometry] [circle_inscribed_quadrilateral A B C D]
variables [segment AB P] [segment BC Q] [ray DA R] [ray DC S]
variables [perpendicular_line BD P Q R S] [PR_eq_QS PR QS]

theorem midpoint_equidistant (h_midpoint : midpoint PQ T) (h_PR_QS : PR = QS) :
    dist T A = dist T C :=
sorry

end midpoint_equidistant_l587_587604


namespace tangent_lines_equal_intercepts_l587_587726

theorem tangent_lines_equal_intercepts : 
  let center := (0, -5)
      radius := √3
      circle_eq := (λ x y, x^2 + (y+5)^2 = 3)
  in ∃ count : ℕ, 
      count = 4 ∧ 
      (∀ (m b : ℝ), (∀ x y : ℝ, line_eq : (λ x y m b, y = m * x + b)) ∧
      ((∃ x y : ℝ, line_eq x y ∧ circle_eq x y) ∧  
      (∀ x_int y_int : ℝ, x_int = y_int ∧ 
      (∃ l : line_eq, l ∩ circle = 2)))
      sorry

end tangent_lines_equal_intercepts_l587_587726


namespace smallest_m_for_integral_solutions_l587_587823

theorem smallest_m_for_integral_solutions :
  ∃ (m : ℕ), (∀ (x : ℤ), 10 * x^2 - m * x + 180 = 0 → x ∈ ℤ) ∧ m = 90 :=
by
  sorry

end smallest_m_for_integral_solutions_l587_587823


namespace probability_correct_l587_587920

def elenaNameLength : Nat := 5
def markNameLength : Nat := 4
def juliaNameLength : Nat := 5
def totalCards : Nat := elenaNameLength + markNameLength + juliaNameLength

-- Without replacement, drawing three cards from 14 cards randomly
def probabilityThreeDifferentSources : ℚ := 
  (elenaNameLength / totalCards) * (markNameLength / (totalCards - 1)) * (juliaNameLength / (totalCards - 2))

def totalPermutations : Nat := 6  -- EMJ, EJM, MEJ, MJE, JEM, JME

def requiredProbability : ℚ := totalPermutations * probabilityThreeDifferentSources

theorem probability_correct :
  requiredProbability = 25 / 91 := by
  sorry

end probability_correct_l587_587920


namespace participated_in_both_l587_587663

-- Define the conditions
def total_students := 40
def math_competition := 31
def physics_competition := 20
def not_participating := 8

-- Define number of students participated in both competitions
def both_competitions := 59 - total_students

-- Theorem statement
theorem participated_in_both : both_competitions = 19 := 
sorry

end participated_in_both_l587_587663


namespace range_of_a_for_decreasing_function_l587_587992

theorem range_of_a_for_decreasing_function :
  ∀ (a : ℝ), (∀ x : ℝ, x ≤ 4 → deriv (λ x, x^2 + 2 * (a - 1) * x + 2) x ≤ 0) ↔ a ≤ -3 :=
by
  sorry

end range_of_a_for_decreasing_function_l587_587992


namespace extreme_value_f_range_of_a_l587_587253

variables {a x : ℝ}

def f (x : ℝ) (a : ℝ) := log x - (a - 1) * x + 1
def g (x : ℝ) := x * (exp x - 1)

theorem extreme_value_f :
  ∀ a : ℝ,
  (a ≤ 1 → ∀ x > 0, ¬∃ (c : ℝ), is_local_max (f x a) c ∧ c > 0) ∧
  (a > 1 → ∃ x₀ : ℝ, x₀ = 1 / (a - 1) ∧ is_local_max (f x a) x₀ ∧ f x₀ a = -log (a - 1)) :=
sorry

theorem range_of_a :
  (∀ x > 0, g x ≥ f x a → a ∈ [1, +∞)) :=
sorry

end extreme_value_f_range_of_a_l587_587253


namespace sales_tax_is_8_percent_l587_587007

-- Define the conditions
def total_before_tax : ℝ := 150
def total_with_tax : ℝ := 162

-- Define the relationship to find the sales tax percentage
noncomputable def sales_tax_percent (before_tax after_tax : ℝ) : ℝ :=
  ((after_tax - before_tax) / before_tax) * 100

-- State the theorem to prove the sales tax percentage is 8%
theorem sales_tax_is_8_percent :
  sales_tax_percent total_before_tax total_with_tax = 8 :=
by
  -- skipping the proof
  sorry

end sales_tax_is_8_percent_l587_587007


namespace finite_permutations_l587_587397

def areConsecutive (n : ℕ) (l : List ℤ) := (l.length = n ∧ ∀ i < n, l[(i + 1) % n] = l[(i + 2) % n])

noncomputable def canSwap (a b c d : ℤ) := ((a - d) * (b - c) < 0)

theorem finite_permutations (n : ℕ) (l : List ℤ) :
  (areConsecutive n l) →
  (∀ a b c d, (a, b, c, d) ∈ l) →
  (∀ i < n, canSwap (l[i]) (l[(i + 1) % n]) (l[(i + 2) % n]) (l[(i + 3) % n])) →
  ∃ k, k < n ∧ ∀ m > k, ¬ (canSwap (l[m]) (l[(m + 1) % n]) (l[(m + 2) % n]) (l[(m + 3) % n])) :=
sorry

end finite_permutations_l587_587397


namespace find_S6_l587_587248

variable (a : ℕ → ℤ)

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, ∃ d : ℤ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  ∑ k in Finset.range (n + 1), a k

theorem find_S6 (a : ℕ → ℤ) (ha_seq : arithmetic_sequence a)
  (h : a 3 + a 4 = 4) :
  sum_first_n_terms a 6 = 12 :=
sorry

end find_S6_l587_587248


namespace sum_s_squares_611_l587_587369

noncomputable theory

def r_domain := {-2, -1, 0, 1, 3}
def r_range := {-1, 0, 3, 4, 6}
def s_domain := {0, 1, 2, 3, 4, 5}

def s (x : ℕ) := x^2 + x + 1

def r_range_intersection_s_domain : Set ℕ := r_range ∩ s_domain

def s_values := r_range_intersection_s_domain.map s

def sum_of_squares := s_values.foldl (λ acc x => acc + x^2) 0

theorem sum_s_squares_611 : sum_of_squares = 611 :=
by
  sorry

end sum_s_squares_611_l587_587369


namespace cube_path_length_k_l587_587861

-- Defining the problem in Lean 4
def cube_path_length (edge_length : ℝ) (turns : ℕ) : ℝ :=
  let radius := edge_length / 2
  let quarter_turn_path := (1 / 4) * (2 * π * radius)
  turns * quarter_turn_path

theorem cube_path_length_k (edge_length : ℝ) (turns : ℕ) (k : ℝ) :
  edge_length = 2 → turns = 4 → cube_path_length edge_length turns = k * π → k = 2 :=
by
  intros h1 h2 h3
  have h4 : cube_path_length 2 4 = 2 * π := by sorry
  have h5 : 2 * π = k * π := by rw [h3, h1, h2]
  have h6 : k = 2 := by sorry
  exact h6

end cube_path_length_k_l587_587861


namespace find_k_l587_587641

variable (a b c : ℝ × ℝ)
variable (k : ℝ)

def is_collinear (k : ℝ) (a b c : ℝ × ℝ) : Prop :=
  let u := (k * a.1 + b.1, k * a.2 + b.2)
  ∃ λ : ℝ, c = (λ * u.1, λ * u.2)

theorem find_k (h : a = (1, 2) ∧ b = (2, 3) ∧ c = (4, -7)) :
  ∃ k, is_collinear k a b c ∧ k = -26/15 :=
sorry

end find_k_l587_587641


namespace part1_part2_l587_587494

-- Part 1: Proving that the product of an odd function and an even function is an odd function.
theorem part1 (f g : ℝ → ℝ) (D : set ℝ) 
  (hf_odd: ∀ x ∈ D, f (-x) = -f x)
  (hg_even: ∀ x ∈ D, g (-x) = g x):
  ∀ x ∈ D, (λ x, f x * g x) (-x) = - (λ x, f x * g x) x :=
by
  sorry

-- Part 2: Finding the expression of the function for x < 0.
theorem part2 (f : ℝ → ℝ) 
  (hf_odd: ∀ x, f (-x) = -f x) 
  (hf_nonneg: ∀ x, 0 ≤ x → f x = x^2) : 
  ∀ x, x < 0 → f x = -x^2 :=
by
  sorry

end part1_part2_l587_587494


namespace max_num_subsets_l587_587713

open Set Finset

-- Define the properties of M
def M : Finset ℕ := {x | 1 ≤ x ∧ x ≤ 20}.toFinset

-- Define the theorem for the maximum value of n with the given conditions
theorem max_num_subsets (n : ℕ) (A : Fin n (Finset ℕ)) 
  (hn : ∀ i j, i ≠ j → |(A i ∩ A j).card| ≤ 2) :
  n ≤ 1350 := sorry

end max_num_subsets_l587_587713


namespace sum_of_corners_is_10_l587_587124

-- Define the Go board to be an 18x18 grid of real numbers
def GoBoard : Type := Array (Array ℝ)

-- Define the main condition of the problem: the sum of any 2x2 sub-grid is always 10
def sum_2x2_is_10 (board : GoBoard) : Prop :=
  ∀ i j, i < 17 → j < 17 → board[i][j] + board[i][j+1] + board[i+1][j] + board[i+1][j+1] = 10

-- Define the objective to prove: the sum of the numbers in the 4 corner squares of the 18x18 board is 10
theorem sum_of_corners_is_10 (board : GoBoard) (h : sum_2x2_is_10 board) : 
  board[0][0] + board[0][17] + board[17][0] + board[17][17] = 10 := 
by 
  sorry

end sum_of_corners_is_10_l587_587124


namespace min_moves_to_chessboard_like_l587_587852

-- Define the grid size and the initial setup
def grid_size : ℕ := 5
def initial_grid : list (list bool) := list.repeat (list.repeat tt grid_size) grid_size -- all cells are white (true)

-- Define a function to check if a grid is chessboard like
def is_chessboard_like (grid : list (list bool)) : Prop :=
  (∀ i j : ℕ, i < grid_size → j < grid_size →
    ((i + j) % 2 = 0 → grid.nthLe i (by linarith) = grid.nthLe i (by linarith))
  ∧ ((i + j) % 2 = 1 → grid.nthLe i (by linarith) = bnot (grid.nthLe i (by linarith))))

-- Define a function to change the colors of two neighboring cells
def change_neighbours (grid : list (list bool)) (i j : ℕ) (ni nj : ℕ) : list (list bool) :=
  grid.modifyNth i (λ row, row.modifyNth j bnot)
      .modifyNth ni (λ row, row.modifyNth nj bnot)

-- The statement of the proof problem
theorem min_moves_to_chessboard_like : ∃ min_moves, min_moves = 12 ∧
(∃ (steps : ℕ) (grids : list (list (list bool))), steps = min_moves ∧
  grids.length = steps ∧
  (∀ (k : ℕ) (h : k < steps), change_neighbours (grids.nth_le k h) = grids.nth_le (k+1) sorry) ∧
  is_chessboard_like (grids.nth_le steps sorry)) :=
  sorry

end min_moves_to_chessboard_like_l587_587852


namespace sum_of_prime_factors_is_prime_l587_587058

/-- Define the specific number in question -/
def num := 30030

/-- List the prime factors of the number -/
def prime_factors := [2, 3, 5, 7, 11, 13]

/-- Sum of the prime factors -/
def sum_prime_factors := prime_factors.sum

theorem sum_of_prime_factors_is_prime :
  sum_prime_factors = 41 ∧ Prime 41 := 
by
  -- The conditions are encapsulated in the definitions above
  -- Now, establish the required proof goal using these conditions
  sorry

end sum_of_prime_factors_is_prime_l587_587058


namespace estimate_profit_l587_587749

noncomputable def numYellowBalls := 3
noncomputable def numWhiteBalls := 3
noncomputable def totalBalls := 6

noncomputable def eventE := "Drawing 3 yellow balls"
noncomputable def eventF := "Drawing 2 yellow balls and 1 white ball"
noncomputable def eventG := "Drawing 3 balls of the same color"

def probability_eventE : Real :=
  1 / 20  -- P(E) = 0.05

def probability_eventF : Real :=
  9 / 20  -- P(F) = 0.45

def probability_eventG : Real :=
  2 / 20  -- P(G) = 0.1

def daily_draws : Nat := 80
def days_in_month : Nat := 30

noncomputable def expected_daily_profit : Real :=
  (72 * 2) - (8 * 10)  -- Earn 64 per day

noncomputable def expected_monthly_profit : Real :=
  expected_daily_profit * days_in_month  -- Earn 1920 in a month

theorem estimate_profit :
  probability_eventE = 0.05 ∧
  probability_eventF = 0.45 ∧
  expected_monthly_profit = 1920 :=
by
  exact And.intro
    (by simp [probability_eventE])
    (by simp [probability_eventF, expected_monthly_profit]); sorry

end estimate_profit_l587_587749


namespace area_of_triangle_OAB_is_4_l587_587241

-- Define the given vectors.
def vec_a : ℝ × ℝ := (2 * Real.cos (2 * Real.pi / 3), 2 * Real.sin (2 * Real.pi / 3))
axiom vec_b : ℝ × ℝ

-- Define vector operations.
def vector_sub (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 - v.1, u.2 - v.2)
def vector_add (u v : ℝ × ℝ) : ℝ × ℝ := (u.1 + v.1, u.2 + v.2)

-- Calculate OA and OB based on given vectors.
def vec_OA : ℝ × ℝ := vector_sub vec_a vec_b
def vec_OB : ℝ × ℝ := vector_add vec_a vec_b

-- Utilize triangle properties and known conditions.
axiom is_isosceles_right_triangle : (vec_OA, vec_OB, vec_a) ∈ {triangle | 
  (triangle.1 ⊥ triangle.2) ∧ (‖triangle.1‖ = ‖triangle.2‖) ∧ 
  (triangle.3 = vec_a) ∧ (triangle.1 = vec_OA) ∧ (triangle.2 = vec_OB)
}

-- Prove the area of triangle OAB
theorem area_of_triangle_OAB_is_4 : 
  1 / 2 * ‖vec_OA‖ * ‖vec_OB‖ = 4 :=
sorry

end area_of_triangle_OAB_is_4_l587_587241


namespace dave_tickets_l587_587173

-- Definitions based on given conditions
def initial_tickets : ℕ := 25
def spent_tickets : ℕ := 22
def additional_tickets : ℕ := 15

-- Proof statement to demonstrate Dave would have 18 tickets
theorem dave_tickets : initial_tickets - spent_tickets + additional_tickets = 18 := by
  sorry

end dave_tickets_l587_587173


namespace arithmetic_expression_eval_l587_587457

-- Condition definitions
def base9_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 2468 := 8 * 9^0 + 6 * 9^1 + 4 * 9^2 + 2 * 9^3
  | 7890 := 0 * 9^0 + 9 * 9^1 + 8 * 9^2 + 7 * 9^3
  | _ => 0  -- We only care about these two numbers for this problem

def base4_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 101 := 1 * 4^0 + 0 * 4^1 + 1 * 4^2
  | _ => 0  -- We only care about this number for this problem

def base8_to_base10 (n : ℕ) : ℕ :=
  match n with
  | 3456 := 6 * 8^0 + 5 * 8^1 + 4 * 8^2 + 3 * 8^3
  | _ => 0  -- We only care about this number for this problem

-- Final statement to prove
theorem arithmetic_expression_eval :
  ((base9_to_base10 2468) / (base4_to_base10 101)) - (base8_to_base10 3456) + (base9_to_base10 7890) = 5102 :=
by
  sorry

end arithmetic_expression_eval_l587_587457


namespace sequence_has_M_property_transformed_M_property_example_1_transformed_M_property_example_2_sequence_has_transformed_M_property_l587_587948

noncomputable def sequence_with_M_property (a : ℕ → ℕ) : Prop :=
  ∀ m : ℕ, ∃ k : ℕ, a m + m = k * k

inductive transformed_M_property (a b : ℕ → ℕ) (n : ℕ) : Prop
  | mk (hp1 : ∀ i : ℕ, i < n → ∃ j : ℕ, b i = a j)
       (hp2 : sequence_with_M_property b) : transformed_M_property

theorem sequence_has_M_property (a : ℕ → ℕ) (Sn : ℕ → ℕ) (h1 : ∀ n, Sn n = n * (n * n - 1) / 3) :
  sequence_with_M_property a :=
sorry

theorem transformed_M_property_example_1 : transformed_M_property (λ n, n + 1) (λ n, [3, 2, 1, 5, 4].nth n) 5 :=
sorry

theorem transformed_M_property_example_2 : ¬ (transformed_M_property (λ n, n + 1) (λ n, n + 1) 11) :=
sorry

theorem sequence_has_transformed_M_property (n m : ℕ) (h1 : 12 ≤ n ∧ n ≤ m^2) :
  transformed_M_property (λ n, n + 1) (λ n, (m^2 - (n + 1))?) n →
  transformed_M_property (λ n, n + 1) (λ n, (m^2 + 1 + n)?) ((m + 1) ^ 2) :=
sorry

end sequence_has_M_property_transformed_M_property_example_1_transformed_M_property_example_2_sequence_has_transformed_M_property_l587_587948


namespace repeat_block_of_fraction_l587_587212

noncomputable def repend_check : String :=
  "235294"

theorem repeat_block_of_fraction :
  ∀ (a b : ℕ), b = 17 ∧ a = 4 → (dec_rep_repetend a b 6) = repend_check :=
by
  intros a b h
  have h₁ : a = 4 := h.left
  have h₂ : b = 17 := h.right
  rw [h₁, h₂]
  sorry  -- Proof is omitted according to instructions

end repeat_block_of_fraction_l587_587212


namespace find_general_term_range_of_Tn_l587_587600

noncomputable def a_sequence (n : ℕ) : ℝ :=
  if n = 0 then 0 else (1 / 3) ^ n

noncomputable def S_n (n : ℕ) : ℝ :=
  (1 / 2) * (1 - a_sequence n)

noncomputable def f (x : ℝ) : ℝ :=
  Real.logBase 1/3 x

noncomputable def b_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, f (a_sequence (i + 1)))

noncomputable def T_n (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, 1 / b_n (i + 1))

theorem find_general_term :
  ∀ (n : ℕ) (hn : n ≥ 1), a_sequence n = (1 / 3) ^ n :=
begin
  sorry
end

theorem range_of_Tn :
  ∀ (n : ℕ) (hn : n ≥ 1), 1 ≤ T_n n ∧ T_n n < 2 :=
begin
  sorry
end

end find_general_term_range_of_Tn_l587_587600


namespace line_circle_equations_l587_587213

theorem line_circle_equations (F : ℝ → ℝ) (A FB : ℝ)
  (hF : F 0 = 0)
  (hAFB : A + FB)
  (t1 t : ℝ)
  (ht1 : t1 + t = real.sqrt 3 ∧ t1 * t = 8) :
  (- real.sqrt 3 * y1 = 0) ∧ ((x - 2) ^ 2 + y ^ 2 = 22) ∧ (abs (t1 - t) * abs (F 0) = 0) :=
by sorry

end line_circle_equations_l587_587213


namespace znayka_can_form_polynomials_l587_587837

structure Quadratic (α : Type _) :=
(p q : α)

def roots {α : Type _} [LinearOrderedField α] (quad : Quadratic α) : set α :=
  let Δ := quad.p ^ 2 - 4 * quad.q in
  if Δ > 0 then
    let sqrt_Δ := sqrt Δ in
    { ((-quad.p + sqrt_Δ) / 2), ((-quad.p - sqrt_Δ) / 2) }
  else if Δ = 0 then
    { -quad.p / 2 }
  else
    ∅

def Quadratic.can_form_polynomials (znayka neznayka : set ℝ) : bool :=
  let nums := (znayka ∪ neznayka).to_list in
  if nums.length ≠ 20 then false
  else
    let quadratics := (finset.product (finset.of_list nums) (finset.of_list nums)).map (λ ⟨p, q⟩ => Quadratic.mk p q) in
    let root_sets := quadratics.map (λ quad => roots quad) in
    (finset.fold (λ acc roots => acc ∪ roots) ∅ root_sets).to_list.length = 11

theorem znayka_can_form_polynomials : ∃ z n : set ℝ, (z.length = 10 ∧ n.length = 10)
  ∧ Quadratic.can_form_polynomials z n = tt :=
sorry

end znayka_can_form_polynomials_l587_587837


namespace math_books_more_than_science_books_l587_587433

theorem math_books_more_than_science_books : 
  ∀ (total_books reading_books math_books history_books science_books : ℕ),
    total_books = 10 →
    reading_books = (2 * total_books) / 5 →
    math_books = (3 * total_books) / 10 →
    history_books = 1 →
    science_books = total_books - (reading_books + math_books + history_books) →
    (math_books - science_books = 1) :=
by
  intros total_books reading_books math_books history_books science_books
  assume h1 h2 h3 h4 h5
  sorry

end math_books_more_than_science_books_l587_587433


namespace ABCE_perimeter_l587_587676

noncomputable def perimeter_of_ABCD (AE : ℝ) (angle_AEB angle_BEC angle_CED : ℝ) 
  (h_angle_AEB : angle_AEB = 60) (h_angle_BEC : angle_BEC = 60) (h_angle_CED : angle_CED = 60) 
  (is_right_angled_ABE : true) (is_right_angled_BCE : true) 
  (is_right_angled_CDE : true) : ℝ := 
  let AB := AE * (Real.sqrt 3 / 2)
  let BE := AE * (1 / 2)
  let BC := BE * (Real.sqrt 3 / 2)
  let CE := BE * (1 / 2)
  let CD := CE * (Real.sqrt 3 / 4)
  let DE := CE * (1 / 2)
  let DA := DE + AE
  AB + BC + CD + DA

theorem ABCE_perimeter : perimeter_of_ABCD 30 60 60 60 sorry sorry sorry =
  (60 * Real.sqrt 3 + 135) / 4 := by
  sorry

end ABCE_perimeter_l587_587676


namespace train_speed_identification_l587_587882

-- Define the conditions
def train_length : ℕ := 300
def crossing_time : ℕ := 30

-- Define the speed calculation
def calculate_speed (distance : ℕ) (time : ℕ) : ℕ := distance / time

-- The target theorem stating the speed of the train
theorem train_speed_identification : calculate_speed train_length crossing_time = 10 := 
by 
  sorry

end train_speed_identification_l587_587882


namespace exponent_fraction_simplification_l587_587829

theorem exponent_fraction_simplification :
  (2 ^ 2020 + 2 ^ 2016) / (2 ^ 2020 - 2 ^ 2016) = 17 / 15 :=
by
  sorry

end exponent_fraction_simplification_l587_587829


namespace find_cans_lids_l587_587156

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l587_587156


namespace trays_needed_to_fill_ice_cubes_l587_587202

-- Define the initial conditions
def ice_cubes_in_glass : Nat := 8
def multiplier_for_pitcher : Nat := 2
def spaces_per_tray : Nat := 12

-- Define the total ice cubes used
def total_ice_cubes_used : Nat := ice_cubes_in_glass + multiplier_for_pitcher * ice_cubes_in_glass

-- State the Lean theorem to be proven: The number of trays needed
theorem trays_needed_to_fill_ice_cubes : 
  total_ice_cubes_used / spaces_per_tray = 2 :=
  by 
  sorry

end trays_needed_to_fill_ice_cubes_l587_587202


namespace max_cookie_price_l587_587353

theorem max_cookie_price :
  ∃ k p : ℕ, 
    (8 * k + 3 * p < 200) ∧ 
    (4 * k + 5 * p > 150) ∧
    (∀ k' p' : ℕ, (8 * k' + 3 * p' < 200) ∧ (4 * k' + 5 * p' > 150) → k' ≤ 19) :=
sorry

end max_cookie_price_l587_587353


namespace complex_product_conjugate_l587_587982

noncomputable def z : ℂ := (√3 + I) / (1 - √3 * I) ^ 2

theorem complex_product_conjugate : z * conj(z) = 1 / 4 := by
  sorry

end complex_product_conjugate_l587_587982


namespace candy_peanut_butter_is_192_l587_587441

/-
   Define the conditions and the statement to be proved.
   The definitions follow directly from the problem's conditions.
-/
def candy_problem : Prop :=
  ∃ (peanut_butter_jar grape_jar banana_jar coconut_jar : ℕ),
    banana_jar = 43 ∧
    grape_jar = banana_jar + 5 ∧
    peanut_butter_jar = 4 * grape_jar ∧
    coconut_jar = 2 * banana_jar - 10 ∧
    peanut_butter_jar = 192
  -- The tuple (question, conditions, correct answer) is translated into this lemma

theorem candy_peanut_butter_is_192 : candy_problem :=
  by
    -- Skipping the actual proof as requested
    sorry

end candy_peanut_butter_is_192_l587_587441


namespace find_omega_and_intervals_l587_587990

noncomputable def f (x ω : ℝ) : ℝ :=
  sin (2 * ω * x - (π / 6)) - 4 * (sin (ω * x))^2 + 2

def is_periodic (f : ℝ → ℝ) (P : ℝ) : Prop :=
  ∀ x : ℝ, f (x + P) = f x

theorem find_omega_and_intervals (ω : ℝ) (hω : ω > 0)
  (h_periodic : is_periodic (f · ω) π) :
  ω = 1 ∧ (∀ x ∈ Icc 0 (3 * π / 4), 
           ∃ I : set ℝ, 
           (I = Icc 0 (π / 12) ∨ I = Icc (7 * π / 12) (3 * π / 4)) ∧ 
           ∀ x y ∈ I, x < y → (f x ω) < (f y ω)) :=
by
  sorry

end find_omega_and_intervals_l587_587990


namespace stratified_sampling_boys_l587_587144

theorem stratified_sampling_boys 
  (total_boys : ℕ) (total_girls : ℕ) (selected_students : ℕ) 
  (total_boys = 48) (total_girls = 36) (selected_students = 21):
  let total_students := total_boys + total_girls in
  let sampling_ratio := selected_students / total_students in
  (total_boys * (sampling_ratio : ℚ)) = 12 :=
by
  sorry

end stratified_sampling_boys_l587_587144


namespace P_cannot_be_written_as_product_l587_587634

/-- Polynomials and related structures definitions -/
noncomputable def P (x : ℤ) (n : ℕ) : ℤ := (x^2 - 7*x + 6)^(2*n) + 13

theorem P_cannot_be_written_as_product (n : ℕ) (h : n > 0) :
  ¬ ∃ (f : ℕ → ℤ[X]) (hf : ∀ i, degree (f i) > 0), degree (f 0) + degree (f 1) + ... + degree (f n) = degree (P X n) ∧ P X n = ∏ i in (finset.range (n+1)), f i :=
sorry

end P_cannot_be_written_as_product_l587_587634


namespace combined_weight_loss_difference_l587_587532

noncomputable def weight_loss_barbi : ℕ := 70.8
noncomputable def weight_loss_luca : ℕ := 156
noncomputable def weight_loss_kim : ℕ := 132

theorem combined_weight_loss_difference :
  (weight_loss_luca + weight_loss_kim) - weight_loss_barbi = 217.2 :=
by
  sorry

end combined_weight_loss_difference_l587_587532


namespace min_value_frac_l587_587963

noncomputable def tetrahedron_edge_length : ℝ := real.sqrt (6)

def distance_to_planes (a b : ℝ) : Prop :=
  a + b = 2

theorem min_value_frac (a b : ℝ) (h_tetrahedron : tetrahedron_edge_length = real.sqrt 6)
  (h_point : P ∈ line_segment A B ∧ P ≠ A ∧ P ≠ B)
  (h_dist : distance_to_planes a b) :
  ∃ x : ℝ, x = 4/a + 1/b ∧ x ≥ 9/2 :=
sorry

end min_value_frac_l587_587963


namespace third_order_central_moment_sum_l587_587362

open Probability

variables {Ω : Type*} {X X1 X2 : Ω → ℝ}
variable [MeasureSpace Ω]

def third_central_moment (X : Ω → ℝ) : ℝ :=
  moment_prop.central_moment X 3

theorem third_order_central_moment_sum (hX : X = λ ω, X1 ω + X2 ω) 
    (h_indep : independent X1 X2)
    (h_mu3_X1 : third_central_moment X1 = μ3_1)
    (h_mu3_X2 : third_central_moment X2 = μ3_2) :
  third_central_moment X = μ3_1 + μ3_2 :=
sorry

end third_order_central_moment_sum_l587_587362


namespace bus_passenger_count_l587_587787

theorem bus_passenger_count (n n' : ℕ) 
  (h1 : ∀ i : ℕ, i ≠ 0 → let p := λ k : ℕ, if k = 0 then n else if k = 1 then n' else p (k-1) + p (k-2)
          in  22 * p 8 + 33 * p 9 = 55) 
  (h2 : let p := λ k : ℕ, if k = 0 then n else if k = 1 then n' else p (k-1) + p (k-2)
          in ∀ i : ℕ, 5 * p i + 7 * p (i + 1) = 9 * p (i+3)) 
  : (9 * n + 12 * n' = 21) :=
sorry

end bus_passenger_count_l587_587787


namespace problem_statement_l587_587080

/-
  Define conditions for m and n
-/

def is_two_divisors (x : Nat) : Prop :=
  (∀ d ∈ (List.range (x+1)).tail, x % d = 0 → (d = 1 ∨ d = x))

def is_prime (p : Nat) : Prop := is_two_divisors p

def three_divisors (x : Nat) : Prop :=
  ∃ p : Nat, is_prime p ∧ x = p^2

def largest_square_less_than (limit : Nat) : Nat :=
  List.last (List.filter (λ x, x < limit) (List.map (λ p, p^2) 
    (List.filter is_prime (List.range limit.tail))), 0)

def m := 2

def n := largest_square_less_than 200

theorem problem_statement : m + n = 171 :=
by {
  have m_def : m = 2 := rfl,
  rw m_def,
  have n_def : n = 169 := sorry, -- Prove that n = 169
  rw n_def,
  norm_num
}

end problem_statement_l587_587080


namespace iphone_cost_l587_587442

variables (earring_cost scarf_cost : ℕ) (num_earrings num_scarves total_value : ℕ)

-- Given conditions
def condition_earring_cost : earring_cost = 6000 := rfl
def condition_scarf_cost : scarf_cost = 1500 := rfl
def condition_num_earrings : num_earrings = 2 := rfl
def condition_num_scarves : num_scarves = 4 := rfl
def condition_total_value : total_value = 20000 := rfl

-- Calculating total cost of known items
def total_known : ℕ := (num_earrings * earring_cost) + (num_scarves * scarf_cost)

-- The statement to prove
theorem iphone_cost :
  (total_value - total_known) = 2000 :=
by
  -- The proof is omitted, but must be provided in practice
  sorry

end iphone_cost_l587_587442


namespace xy_value_l587_587450

theorem xy_value (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : x * y = 21 := 
by sorry

end xy_value_l587_587450


namespace gain_percent_correct_l587_587506

noncomputable def cycleCP : ℝ := 900
noncomputable def cycleSP : ℝ := 1180
noncomputable def gainPercent : ℝ := (cycleSP - cycleCP) / cycleCP * 100

theorem gain_percent_correct :
  gainPercent = 31.11 := by
  sorry

end gain_percent_correct_l587_587506


namespace find_a5_l587_587361

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}

-- Conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d
def S_n (a : ℕ → ℝ) (n : ℕ) := (n * (a 1 + a n)) / 2

-- Given conditions
axiom S2_eq_S6 : S 2 = S 6
axiom a4_eq_1 : a 4 = 1

-- The proof goal is to find a_5
theorem find_a5 (h_arith : arithmetic_sequence a d) (h_S2_S6 : S 2 = S 6) (h_a4 : a 4 = 1) : 
  a 5 = -1 := 
sorry

end find_a5_l587_587361


namespace find_a_l587_587344

def is_centroid (A B C P : Point) : Prop :=
  -- Definition for point P being the centroid of triangle ABC
  sorry

def cos_2A_sin_pi_A_identity (A : ℝ) : Prop :=
  cos 2 * A + (4 + sqrt 3) * sin (pi - A) = 2 * sqrt 3 + 1

def centroid_length (A B C P : Point) : ℝ :=
  -- AP length when P is the centroid of triangle ABC
  sorry

theorem find_a (A B C P : Point) (a b c : ℝ) (√3 : ℝ) :
  b = 2 →
  cos_2A_sin_pi_A_identity A →
  is_centroid A B C P →
  AP = 2 * sqrt 7 / 3 →
  a = 2 * sqrt 3 ∨ a = 2 * sqrt 13 :=
by
  sorry

end find_a_l587_587344


namespace polynomial_roots_inequalities_l587_587625

theorem polynomial_roots_inequalities
  (p q r s : ℝ)
  (roots : {α : Fin 4 → ℝ // ∀ i, 0 < α i})
  (h_eq : ∀ x, x^4 + p * x^3 + q * x^2 + r * x + s = 0):
  (pr_ineq : p * r - 16 * s ≥ 0) ∧ 
  (q2_ineq : q * q - 36 * s ≥ 0) := 
by
  sorry

end polynomial_roots_inequalities_l587_587625


namespace arithmetic_progression_solution_l587_587932

theorem arithmetic_progression_solution (n : ℤ) :
  ∃ a b c d : ℤ, 
    (d = c + 1 ∧ nat_prime (d - c + 1)) ∧ 
    (a + b^2 + c^3 = d^2 * b) ∧ 
    (a, b, c, d) = (n, n + 1, n + 2, n + 3) := 
by {
  sorry
}

end arithmetic_progression_solution_l587_587932


namespace finite_swaps_in_circle_l587_587400

theorem finite_swaps_in_circle (n : ℕ) (a : Fin n → ℤ) :
  (∃ (m : ℕ), ∀ (i : Fin n), (a i - a (i + 3)) * (a (i + 1) - a (i + 2)) < 0 → m > 0 ∧ ∀ (j : ℕ), j > m → (∃ k l, swap_cond a k l))
  where swap_cond (a : Fin n → ℤ) (k l : Fin n) := (a k, a l) := (a l, a k) :=
sorry

end finite_swaps_in_circle_l587_587400


namespace triangle_area_l587_587155

theorem triangle_area (P Q R S : Type) (PR_len : ℝ) 
  (h_angle_PRQ : ∠P Q R = 45)
  (h_right_angle : ∠P Q R = 90)
  (h_altitude : length(QS) = 2) :
  area(ΔPQR) = 2 * sqrt(2) :=
sorry

end triangle_area_l587_587155


namespace sum_is_neg5_opposites_correct_l587_587580

def numbers : List ℤ := [-5, -2, | -2 |, 0]

def sum_of_numbers : ℤ := numbers.sum

theorem sum_is_neg5 : sum_of_numbers = -5 := by
  sorry

def opposite (x : ℤ) : ℤ := -x

theorem opposites_correct :
  opposite (-5) = 5 ∧ 
  opposite (-2) = 2 ∧ 
  opposite (| -2 |) = -2 ∧ 
  opposite 0 = 0 :=
by
  sorry

end sum_is_neg5_opposites_correct_l587_587580


namespace sarees_original_price_l587_587429

theorem sarees_original_price (P : ℝ) (h : 0.90 * P * 0.95 = 342) : P = 400 :=
by
  sorry

end sarees_original_price_l587_587429


namespace transform_curve_l587_587680

theorem transform_curve (λ μ : ℝ) (hλ : λ > 0) (hμ : μ > 0) :
  (∀ x y, 2 * sin (3 * x) = y ↔ sin (λ * x) = μ * y) ↔ (λ = 3 ∧ μ = 1 / 2) :=
by
  sorry

end transform_curve_l587_587680


namespace telephone_connection_count_l587_587098

theorem telephone_connection_count :
  let n := 7 in
  let k := 6 in
  (n * k) / 2 = 21 :=
by
  let n := 7
  let k := 6
  have h : (n * k) / 2 = (7 * 6) / 2 := by rfl
  rw h
  simp
  norm_num
  sorry

end telephone_connection_count_l587_587098


namespace circle_center_l587_587934

theorem circle_center 
    (x y : ℝ)
    (h : 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) : 
    (1, -2) = ((-(h - 4 * (x - 1)^2 - 4 * (y + 2)^2 + 20)) / 4, 
                (-(h - 4 * (x - 1)^2 - 4 * (y + 2)^2 + 20)) / 4 ) :=
sorry

end circle_center_l587_587934


namespace complex_conjugate_of_z_l587_587614

-- Given conditions
def i := Complex.I
def z : Complex := (2 * i ^ 3) / (1 - i)

-- Problem statement: Prove that the complex conjugate of z is equal to 1 + i.
theorem complex_conjugate_of_z :
  Complex.conj(z) = 1 + i := by
  sorry

end complex_conjugate_of_z_l587_587614


namespace perpendicularity_theorem_l587_587970

-- Definitions for lines and planes
variables {m : Type} {α β : Type} -- Assuming m to be a line and α, β to be planes.

-- Assume spatial relationships
variable [line m]
variable [plane α]
variable [plane β]

-- Assume the required conditions
variable (h1 : m ⊥ α)
variable (h2 : α ∥ β)

-- The theorem we need to prove
theorem perpendicularity_theorem : m ⊥ β := sorry

end perpendicularity_theorem_l587_587970


namespace number_of_distinct_sentences_l587_587643

noncomputable def count_distinct_sentences (phrase : String) : Nat :=
  let I_options := 3 -- absent, partially present, fully present
  let II_options := 2 -- absent, present
  let IV_options := 2 -- incomplete or absent
  let III_mandatory := 1 -- always present
  (III_mandatory * IV_options * I_options * II_options) - 1 -- subtract the original sentence

theorem number_of_distinct_sentences :
  count_distinct_sentences "ранним утром на рыбалку улыбающийся Игорь мчался босиком" = 23 :=
by
  sorry

end number_of_distinct_sentences_l587_587643


namespace solve_double_burgers_l587_587181

theorem solve_double_burgers (S D : ℕ) (h1 : S + D = 50) (h2 : 1 * S + 3 / 2 * D = 70.50) : D = 41 :=
by
  sorry

end solve_double_burgers_l587_587181


namespace product_of_roots_l587_587976

-- Let x₁ and x₂ be roots of the quadratic equation x^2 + x - 1 = 0
theorem product_of_roots (x₁ x₂ : ℝ) (h₁ : x₁^2 + x₁ - 1 = 0) (h₂ : x₂^2 + x₂ - 1 = 0) :
  x₁ * x₂ = -1 :=
sorry

end product_of_roots_l587_587976


namespace nails_no_three_collinear_l587_587542

-- Let's denote the 8x8 chessboard as an 8x8 grid of cells

-- Define a type for positions on the chessboard
def Position := (ℕ × ℕ)

-- Condition: 16 nails should be placed in such a way that no three are collinear. 
-- Let's create an inductive type to capture these conditions

def no_three_collinear (nails : List Position) : Prop :=
  ∀ (p1 p2 p3 : Position), p1 ∈ nails → p2 ∈ nails → p3 ∈ nails → 
  (p1.1 = p2.1 ∧ p2.1 = p3.1) → False ∧
  (p1.2 = p2.2 ∧ p2.2 = p3.2) → False ∧
  (p1.1 - p1.2 = p2.1 - p2.2 ∧ p2.1 - p2.2 = p3.1 - p3.2) → False

-- The main statement to prove
theorem nails_no_three_collinear :
  ∃ nails : List Position, List.length nails = 16 ∧ no_three_collinear nails :=
sorry

end nails_no_three_collinear_l587_587542


namespace R_and_D_expenditure_l587_587536

theorem R_and_D_expenditure (R_D_t : ℝ) (Delta_APL_t_plus_2 : ℝ) (ratio : ℝ) :
  R_D_t = 3013.94 → Delta_APL_t_plus_2 = 3.29 → ratio = 916 →
  R_D_t / Delta_APL_t_plus_2 = ratio :=
by
  intros hR hD hRto
  rw [hR, hD, hRto]
  sorry

end R_and_D_expenditure_l587_587536


namespace quadratic_roots_inverse_sum_l587_587904

theorem quadratic_roots_inverse_sum :
  let r s : ℚ in 
  r and s are the roots of the quadratic equation \(2x^2 + 3x - 5 = 0\)
  (r + s = -3/2) ∧ (r * s = -5/2) 
   → (\frac{1}{r^2} + \frac{1}{s^2} = \frac{29}{25}) := 
sorry

end quadratic_roots_inverse_sum_l587_587904


namespace cylinder_volume_ratio_l587_587850

theorem cylinder_volume_ratio (h1 h2 r1 r2 V1 V2 : ℝ)
  (h1_eq : h1 = 9)
  (h2_eq : h2 = 6)
  (circumference1_eq : 2 * π * r1 = 6)
  (circumference2_eq : 2 * π * r2 = 9)
  (V1_eq : V1 = π * r1^2 * h1)
  (V2_eq : V2 = π * r2^2 * h2)
  (V1_calculated : V1 = 81 / π)
  (V2_calculated : V2 = 243 / (4 * π)) :
  (max V1 V2) / (min V1 V2) = 3 / 4 :=
by
  sorry

end cylinder_volume_ratio_l587_587850


namespace second_smallest_odd_1_to_10_l587_587576

noncomputable def second_smallest_odd (n : ℕ) : ℕ :=
  let odd_numbers := List.filter (λ x, x % 2 = 1) (List.range n)
  odd_numbers.getU (by decide : 1 < odd_numbers.length)

theorem second_smallest_odd_1_to_10 : second_smallest_odd 11 = 3 := by
  sorry

end second_smallest_odd_1_to_10_l587_587576


namespace circle_area_of_equilateral_triangle_l587_587526

theorem circle_area_of_equilateral_triangle :
  ∀ (DEF : Triangle) (s : ℝ) (r : ℝ),
  DEF.is_equilateral s →
  s = 4 * Real.sqrt 3 →
  r = s / Real.sqrt 3 →
  ∃ (A : ℝ), A = π * r^2 ∧ A = 16 * π := by
  sorry

end circle_area_of_equilateral_triangle_l587_587526


namespace prime_sum_is_prime_l587_587789

def prime : ℕ → Prop := sorry 

theorem prime_sum_is_prime (A B : ℕ) (hA : prime A) (hB : prime B) (hAB : prime (A - B)) (hABB : prime (A - B - B)) : prime (A + B + (A - B) + (A - B - B)) :=
sorry

end prime_sum_is_prime_l587_587789


namespace total_carrots_l587_587025

theorem total_carrots (sally_carrots fred_carrots : ℕ) (h1 : sally_carrots = 6) (h2 : fred_carrots = 4) : sally_carrots + fred_carrots = 10 := by
  sorry

end total_carrots_l587_587025


namespace neg_all_cups_full_l587_587056

variable (x : Type) (cup : x → Prop) (full : x → Prop)

theorem neg_all_cups_full :
  ¬ (∀ x, cup x → full x) = ∃ x, cup x ∧ ¬ full x := by
sorry

end neg_all_cups_full_l587_587056


namespace polynomial_evaluation_l587_587357

theorem polynomial_evaluation 
  {n : ℕ} 
  (a : Fin n.succ → ℤ) 
  (hai : ∀ i, 0 ≤ a i ∧ a i < 3) 
  (hQsqrt3 : ∑ i in Finset.range n.succ, a i * (3:ℤ) ^ (i / 2) * (sqrt 3) ^ (i % 2) = 20 + 17 * sqrt 3) :
  (∑ i in Finset.range n.succ, a i * 2 ^ i) = 86 := 
sorry

end polynomial_evaluation_l587_587357


namespace average_rate_of_trip_l587_587738

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l587_587738


namespace smallest_positive_period_and_range_specific_trigonometric_value_l587_587277

-- Definition of the function
def f (x : ℝ) := sqrt 3 * sin x + cos x + 1

-- Statement of the proof
theorem smallest_positive_period_and_range :
  (∀ x, f (x + 2 * π) = f x) ∧ (set.range f = set.Icc (-1) 3) :=
by
  sorry

theorem specific_trigonometric_value (α : ℝ) (hα : π < α ∧ α < 3 * π / 2) (h : f(α - π / 6) = 1 / 3) :
  (cos (2 * α)) / (1 + cos (2 * α) - sin (2 * α)) = (4 + sqrt 2) / 8 :=
by
  sorry

end smallest_positive_period_and_range_specific_trigonometric_value_l587_587277


namespace root_of_quadratic_l587_587312

theorem root_of_quadratic (a : ℝ) (ha : a ≠ 1) (hroot : (a-1) * 1^2 - a * 1 + a^2 = 0) : a = -1 := by
  sorry

end root_of_quadratic_l587_587312


namespace additional_savings_if_purchase_together_l587_587149

theorem additional_savings_if_purchase_together :
  let price_per_window := 100
  let windows_each_offer := 4
  let free_each_offer := 1
  let dave_windows := 7
  let doug_windows := 8

  let cost_without_offer (windows : Nat) := windows * price_per_window
  let cost_with_offer (windows : Nat) := 
    if windows % (windows_each_offer + free_each_offer) = 0 then
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window
    else
      (windows / (windows_each_offer + free_each_offer)) * windows_each_offer * price_per_window 
      + (windows % (windows_each_offer + free_each_offer)) * price_per_window

  (cost_without_offer (dave_windows + doug_windows) 
  - cost_with_offer (dave_windows + doug_windows)) 
  - ((cost_without_offer dave_windows - cost_with_offer dave_windows)
  + (cost_without_offer doug_windows - cost_with_offer doug_windows)) = price_per_window := 
  sorry

end additional_savings_if_purchase_together_l587_587149


namespace equal_cost_distribution_impossible_l587_587504

theorem equal_cost_distribution_impossible (n m : ℕ) (h₁ : n > 1) (h₂ : m > 1) :
  ¬ (∃ (x : ℕ), 4 * x + (n * m - 4 * x) = n * m ∧
                 x.count 0 + x.count 10 + x.count 30 + x.count 40 = 4 ∧
                 ∀ y, y ∈ x.count → (y = 0 ∨ y = 10 ∨ y = 30 ∨ y = 40 ∨ y = 20)) :=
by sorry

end equal_cost_distribution_impossible_l587_587504


namespace can_lids_per_box_l587_587161

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l587_587161


namespace simplify_expression_l587_587029

variable (x y : ℕ)
variable (h_x : x = 5)
variable (h_y : y = 2)

theorem simplify_expression : (10 * x^2 * y^3) / (15 * x * y^2) = 20 / 3 := by
  sorry

end simplify_expression_l587_587029


namespace coefficients_sum_l587_587256

theorem coefficients_sum (a0 a1 a2 a3 a4 a5 : ℤ) :
  ∀ x : ℤ, (x - 2)^5 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 + a4 * (x + 1)^4 + a5 * (x + 1)^5 →
  a1 + a2 + a3 + a4 + a5 = 211 :=
by
  intro x h
  let H0 := h 0
  let Hneg1 := h (-1)
  have H_eq := H0 - Hneg1
  sorry

end coefficients_sum_l587_587256


namespace number_of_boys_in_school_l587_587431

theorem number_of_boys_in_school :
  ∃ (B : ℕ), (∃ (G : ℕ), B + G = 100 ∧ G = (B * 100) / (B + G) ∧ B = 50) :=
by {
  use 50,
  use 50,
  sorry,
}

end number_of_boys_in_school_l587_587431


namespace max_among_l587_587710

theorem max_among (x y : ℝ) (h : 3 * (x^2 + y^2) = x + y) : x - y ≤ (1 / (2 * real.sqrt 3)) :=
  sorry

end max_among_l587_587710


namespace g_inv_undefined_at_one_l587_587305

-- Define the function g
def g (x : ℝ) : ℝ := (x - 5) / (x - 6)

-- Define the inverse function g_inv
noncomputable def g_inv (x : ℝ) : ℝ := (5 - 6 * x) / (1 - x)

-- Prove that the inverse function g_inv is undefined at x = 1
theorem g_inv_undefined_at_one : ∀ x, x = 1 → ¬(∃ y, g_inv x = y) :=
by {
  intros x hx,
  rw hx,
  unfold g_inv,
  use sorry,
}

end g_inv_undefined_at_one_l587_587305


namespace area_of_inscribed_square_l587_587309

theorem area_of_inscribed_square (D : ℝ) (h : D = 10) : 
  ∃ A : ℝ, A = 50 :=
by
  sorry

end area_of_inscribed_square_l587_587309


namespace carrots_total_l587_587023

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end carrots_total_l587_587023


namespace trig_equation_solution_l587_587099

open Real

theorem trig_equation_solution (x : ℝ) (k n : ℤ) :
  (sin (2 * x)) ^ 4 + (sin (2 * x)) ^ 3 * (cos (2 * x)) -
  8 * (sin (2 * x)) * (cos (2 * x)) ^ 3 - 8 * (cos (2 * x)) ^ 4 = 0 ↔
  (∃ k : ℤ, x = -π / 8 + (π * k) / 2) ∨ 
  (∃ n : ℤ, x = (1 / 2) * arctan 2 + (π * n) / 2) := sorry

end trig_equation_solution_l587_587099


namespace simplify_expression_l587_587019

theorem simplify_expression (x : ℝ) (h1 : x ≠ 0) (h2 : x ≠ 1) (h3 : x ≠ 2) (h4 : x ≠ 3) :
  ((x - 5) / (x - 3) - ((x^2 + 2 * x + 1) / (x^2 + x)) / ((x + 1) / (x - 2)) = 
  -6 / (x^2 - 3 * x)) :=
by
  sorry

end simplify_expression_l587_587019


namespace reachability_in_connected_equitable_graph_l587_587049

-- Definitions and conditions as per the problem statement
variables {V : Type} [fintype V] 

structure digraph (V : Type) :=
  (edges : V → V → Prop)
  (symm : ∀ {x y : V}, edges x y → ¬edges y x)

structure equitable_graph (G : digraph V) :=
  (in_degree_eq_out_degree : ∀ v : V, (card {u : V // G.edges u v}) = (card {u : V // G.edges v u}))

-- Main theorem statement
theorem reachability_in_connected_equitable_graph
  {G : digraph V}
  (h_connected : ∀ u v : V, u ≠ v → (∃ path : list V, u ∈ path ∧ v ∈ path ∧ ∀ w ∈ path, ∃ (x y : V), G.edges x y))
  (h_equitable : equitable_graph G) :
  ∀ u v : V, ∃ path : list V, u ∈ path ∧ v ∈ path ∧ ∀ (w z : V), w ∈ path → z ∈ path → (∃ (a b : V), G.edges a b ∧ (a = w ∨ a = z) ∧ (b = w ∨ b = z)) :=
by sorry

end reachability_in_connected_equitable_graph_l587_587049


namespace percent_defective_shipped_l587_587487

-- Conditions given in the problem
def percent_defective (percent_total_defective: ℝ) : Prop := percent_total_defective = 0.08
def percent_shipped_defective (percent_defective_shipped: ℝ) : Prop := percent_defective_shipped = 0.04

-- The main theorem we want to prove
theorem percent_defective_shipped (percent_total_defective percent_defective_shipped : ℝ) 
  (h1 : percent_defective percent_total_defective) (h2 : percent_shipped_defective percent_defective_shipped) : 
  (percent_total_defective * percent_defective_shipped * 100) = 0.32 :=
by
  sorry

end percent_defective_shipped_l587_587487


namespace minimal_k_l587_587697

noncomputable def f (M : Set ℕ) (A : Set ℕ) : Set ℕ :=
  {x ∈ M | Nat.Odd (A.count_dvd x)}

theorem minimal_k (M : Finset ℕ) (hM : M.card = 2017)
    (k : ℕ) :
  (∀ (A : Set ℕ), ∃ color : (Set ℕ → ℕ), 
    (∀ A, color A ∈ Finset.fin_range k n) ∧ (∀ A B, A ≠ B → (f M A) = B → color A ≠ color B)) →
    k = 2 := sorry

end minimal_k_l587_587697


namespace length_of_smaller_cube_edge_is_5_l587_587132

-- Given conditions
def stacked_cube_composed_of_smaller_cubes (n: ℕ) (a: ℕ) : Prop := a * a * a = n

def volume_of_larger_cube (l: ℝ) (v: ℝ) : Prop := l ^ 3 = v

-- Problem statement: Prove that the length of one edge of the smaller cube is 5 cm
theorem length_of_smaller_cube_edge_is_5 :
  ∃ s: ℝ, stacked_cube_composed_of_smaller_cubes 8 2 ∧ volume_of_larger_cube (2*s) 1000 ∧ s = 5 :=
  sorry

end length_of_smaller_cube_edge_is_5_l587_587132


namespace average_rate_of_trip_l587_587737

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l587_587737


namespace length_of_second_train_correct_l587_587813

noncomputable def length_of_second_train
  (speed_first_train_kmph : ℝ)
  (speed_second_train_kmph : ℝ)
  (length_first_train_m : ℝ)
  (time_to_clear_s : ℝ) : ℝ :=
  let relative_speed_mps := (speed_first_train_kmph + speed_second_train_kmph) * 1000 / 3600
  let combined_length_m := relative_speed_mps * time_to_clear_s
  in combined_length_m - length_first_train_m

theorem length_of_second_train_correct :
  length_of_second_train 40 50 111 11.039116870650348 ≈ 164.98 :=
  by sorry

end length_of_second_train_correct_l587_587813


namespace total_cost_l587_587076

noncomputable def item_cost_with_tax_and_discount 
    (initial_cost : ℕ) (discount_rate : ℕ) (tax_rate : ℕ) : ℕ :=
let discounted_cost := initial_cost * (100 - discount_rate) / 100 in
let tax := discounted_cost * tax_rate / 100 in
discounted_cost + tax

theorem total_cost :
    let eggs : ℕ := 3 * 50
    let milk : ℕ := 2 * 300
    let bread : ℕ := 4 * 125
    let eggs_total := item_cost_with_tax_and_discount eggs 10 5 in
    let milk_total := item_cost_with_tax_and_discount milk 5 5 in
    let bread_total := bread * 102 / 100 in
    eggs_total + milk_total + bread_total = 1251 :=
by
    sorry

end total_cost_l587_587076


namespace train_speed_kmph_l587_587137

noncomputable def jogger_speed_kmph : ℝ := 9
noncomputable def distance_ahead_m : ℝ := 240
noncomputable def train_length_m : ℝ := 120
noncomputable def passing_time_s : ℝ := 35.99712023038157

theorem train_speed_kmph : (train_length_m + distance_ahead_m) / passing_time_s * 3.6 = 36 :=
by
  -- Define the total distance the train needs to cover
  let total_distance := train_length_m + distance_ahead_m
  -- Calculate the speed in m/s
  let speed_mps := total_distance / passing_time_s
  -- Convert to kmph
  let speed_kmph := speed_mps * 3.6
  -- Assert the speed in kmph
  exact (by linarith : speed_kmph = 36)

end train_speed_kmph_l587_587137


namespace true_proposition_l587_587094

-- Define the propositions
def prop_A : Prop := ∀ (α β γ : ℝ), α + β + γ = 180 → α ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 90} ∧ β ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 90} ∧ γ ∈ {x : ℝ | 0 ≤ x ∧ x ≤ 90}
def prop_B : Prop := ∀ (α : ℝ), (α = 0 ∨ α = 180) → sin α = 0 ∧ tan α = 0
def prop_C : Prop := ∀ (α β : ℝ), terminal_side α = terminal_side β → α = β
def prop_D : Prop := ∀ (α : ℝ), 90 < α ∧ α < 180 → obtuse α

-- Define the main theorem to prove Proposition B is true.
theorem true_proposition : prop_B :=
by
  sorry

end true_proposition_l587_587094


namespace length_more_than_breadth_l587_587784

theorem length_more_than_breadth (length breadth : ℕ) 
  (h₁ : length = 66) 
  (h₂ : 2 * (length + breadth) = (5300 / 26.5) * 2)
  : length - breadth = 32 :=
sorry

end length_more_than_breadth_l587_587784


namespace can_perform_basic_operations_with_star_l587_587502

noncomputable def star (a b : ℝ) : ℝ := 1 - a / b

theorem can_perform_basic_operations_with_star :
  (∀ a b : ℝ, b ≠ 0 → (star a b) star 1 = a / b) ∧
  (∀ a b : ℝ, b ≠ 0 → let inv_b := (star 1 b) star 1 in (star a inv_b) star 1 = a * b) ∧
  (∀ a b : ℝ, let b_star_a := star b a in b_star_a + a = a - b) ∧
  (∀ a b : ℝ, let neg_b := star b 0 + 0 in a - neg_b = a + b) :=
  sorry

end can_perform_basic_operations_with_star_l587_587502


namespace count_three_digit_numbers_sum_24_l587_587228

theorem count_three_digit_numbers_sum_24 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 24}.card = 4 := 
by
  sorry

end count_three_digit_numbers_sum_24_l587_587228


namespace angle_sum_B_D_l587_587301

theorem angle_sum_B_D (A B D F G : Point) (angle_A angle_AFG : Angle) :
  angle_A = 30 ∧ ∠AFG = ∠AGF → ∠B + ∠D = 75 :=
by
  sorry

end angle_sum_B_D_l587_587301


namespace possible_values_of_a_l587_587283

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≤ 1 then x^2 - a*x + 5 else a / x

theorem possible_values_of_a (a : ℝ) :
  (∀ x y : ℝ, x ≤ y → f a x ≥ f a y) ↔ (2 ≤ a ∧ a ≤ 3) :=
by
  sorry

end possible_values_of_a_l587_587283


namespace aluminum_sulfide_molecular_weight_l587_587822

theorem aluminum_sulfide_molecular_weight (mw_4_moles: ℝ) (h: mw_4_moles = 600) : 
  (∃ mw: ℝ, mw = 150) :=
by 
  use 150
  have : mw_4_moles / 4 = 150 := by linarith
  rw [h, this]
  trivial

end aluminum_sulfide_molecular_weight_l587_587822


namespace no_six_clique_guarantee_l587_587511

noncomputable def delegate_knowledge_problem : Prop :=
  ∃ (delegate : Type) [fintype delegate] [decidable_eq delegate] (knows : delegate → delegate → Prop),
    fintype.card delegate = 500 ∧
    (∀ (a b : delegate), knows a b → knows b a) ∧
    (∀ (a : delegate), fintype.card { b : delegate // knows a b } = 400) ∧
    ¬ ∃ (subset : finset delegate), subset.card = 6 ∧ (∀ a b ∈ subset, knows a b)

theorem no_six_clique_guarantee : delegate_knowledge_problem := sorry

end no_six_clique_guarantee_l587_587511


namespace xy_computation_l587_587449

theorem xy_computation (x y : ℝ) (h1 : x + y = 10) (h2 : x^3 + y^3 = 370) : 
  x * y = 21 := by
  sorry

end xy_computation_l587_587449


namespace point_above_line_range_l587_587671

theorem point_above_line_range (a : ℝ) :
  (2 * a - (-1) + 1 < 0) ↔ a < -1 :=
by
  sorry

end point_above_line_range_l587_587671


namespace finite_swaps_in_circle_l587_587399

theorem finite_swaps_in_circle (n : ℕ) (a : Fin n → ℤ) :
  (∃ (m : ℕ), ∀ (i : Fin n), (a i - a (i + 3)) * (a (i + 1) - a (i + 2)) < 0 → m > 0 ∧ ∀ (j : ℕ), j > m → (∃ k l, swap_cond a k l))
  where swap_cond (a : Fin n → ℤ) (k l : Fin n) := (a k, a l) := (a l, a k) :=
sorry

end finite_swaps_in_circle_l587_587399


namespace Dave_has_more_money_than_Derek_l587_587909

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l587_587909


namespace quadrilateral_condition_angle_equal_l587_587113

-- Part 1: Quadrilateral with sides a, b, c, d and a parallel to b

variables (a b c d : ℝ)

theorem quadrilateral_condition (h1 : a > b) (h2 : a > c) (h3 : a > d) :
  (b + c + d ≥ a) ∧ (a + d ≥ b + c) ∧ (a + c ≥ b + d) ↔
  ∃ (A B C D : ℝ), (A, B, C, D) where
  -- assume the construction details are handled, we care about the condition only
sorry

-- Part 2: Angle Relationship in Triangle with Altitude

variables {A B C H D E F : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space H] [metric_space D] [metric_space E] [metric_space F]

theorem angle_equal (hacute : ∃ (A B C : Type*), ∠A < π/2 ∧ ∠B < π/2 ∧ ∠C < π/2)
  (Halt : H = foot (altitude A B C))
  (DonAH : D ∈ (segment A H))
  (BDmeetsAC : (BD ∩ AC) = ∅ → intersection point E)
  (CDmeetsAB : (CD ∩ AB) = ∅ → intersection point F) :
  ∠AHE = ∠AHF :=
sorry

end quadrilateral_condition_angle_equal_l587_587113


namespace count_non_monotonic_ω_l587_587242

def is_not_monotonic_in_interval (ω : ℕ) (I : Set ℝ) : Prop :=
  ∃ x : ℝ, x ∈ I ∧ ∃ k : ℤ, x = π / (2 * ω) + k * π / ω

theorem count_non_monotonic_ω :
  (∃ (ωs : Finset ℕ), (∀ ω ∈ ωs, ω ∈ (Set.Icc 1 15 : Set ℕ) ∧ is_not_monotonic_in_interval ω (Set.Icc (π / 4) (π / 3))) ∧ ωs.card = 8) :=
sorry

end count_non_monotonic_ω_l587_587242


namespace angle_bisector_theorem_l587_587343

noncomputable def triangle_ABC (A B C : Type) := Prop

noncomputable def points_on_segments (A C B : Type) (M N : Type) := Prop

noncomputable def angle_bisector (A C B D : Type) := Prop

theorem angle_bisector_theorem
    (A B C : Type) 
    (M N D : Type) 
    (triangle : triangle_ABC A B C) 
    (pointsM : points_on_segments A C B M) 
    (pointsN : points_on_segments A C B N) 
    (bisector : angle_bisector A C B D) 
    (h₁ : ∠ ACB = 50) 
    (h₂ : D = (arc_intersection_centers M N)) 
    : ∠ ACD = 25 :=
begin
  sorry
end

end angle_bisector_theorem_l587_587343


namespace expected_total_rain_l587_587884

theorem expected_total_rain :
  let p_sun := 0.30
  let p_rain5 := 0.30
  let p_rain12 := 0.40
  let rain_sun := 0
  let rain_rain5 := 5
  let rain_rain12 := 12
  let days := 6
  let E_rain := p_sun * rain_sun + p_rain5 * rain_rain5 + p_rain12 * rain_rain12
  E_rain * days = 37.8 :=
by
  -- Proof omitted
  sorry

end expected_total_rain_l587_587884


namespace CD_parallel_EF_theorem_l587_587189

noncomputable def CD_parallel_EF (Γ₁ Γ₂ : Set Point) (A D B C E F : Point) (d₁ d₂ : Line) : Prop :=
∃ (A D B C E F : Point) (d₁ d₂ : Line), 
  ((A ∈ (Γ₁ ∩ Γ₂)) ∧ (D ∈ (Γ₁ ∩ Γ₂)) ∧
  (A ∈ d₁) ∧ (B ∈ d₂) ∧
  (C ∈ ((Γ₁ ∩ d₁) \ {A})) ∧ 
  (E ∈ ((Γ₂ ∩ d₁) \ {A})) ∧
  (D ∈ ((Γ₁ ∩ d₂) \ {A})) ∧
  (F ∈ ((Γ₂ ∩ d₂) \ {A})) ∧
  Parallel CD EF)

-- Placeholder defs for Point, Line, etc.
def Point := ℝ × ℝ  -- This is just a placeholder. Adjust as needed.
def Line := Set Point

-- Placeholder definition of intersection
def intersection (s₁ s₂ : Set Point) : Set Point := {p | p ∈ s₁ ∧ p ∈ s₂}

-- Placeholder definition of parallel lines
def Parallel (l₁ l₂ : Line) : Prop := sorry

-- Defining the problem in Lean
theorem CD_parallel_EF_theorem (Γ₁ Γ₂ : Set Point) (A D B C E F : Point) (d₁ d₂ : Line) :
  ((A ∈ (Γ₁ ∩ Γ₂)) ∧ (D ∈ (Γ₁ ∩ Γ₂)) ∧
  (A ∈ d₁) ∧ (B ∈ d₂) ∧
  (C ∈ ((Γ₁ ∩ d₁) \ {A})) ∧ 
  (E ∈ ((Γ₂ ∩ d₁) \ {A})) ∧
  (D ∈ ((Γ₁ ∩ d₂) \ {A})) ∧
  (F ∈ ((Γ₂ ∩ d₂) \ {A}))) →
  Parallel (Line_through C D) (Line_through E F) := by
  sorry

end CD_parallel_EF_theorem_l587_587189


namespace parallel_lines_slope_l587_587622

theorem parallel_lines_slope (m : ℝ) :
  let l1 := (m - 2) * x - 3 * y - 1 = 0,
      l2 := m * x + (m + 2) * y + 1 = 0 in
  (∀ (x y : ℝ), l1 = 0 → l2 = 0) → m = -4 :=
by
  sorry

end parallel_lines_slope_l587_587622


namespace inclination_angle_range_l587_587478

theorem inclination_angle_range 
  (k : ℝ) (h : -real_sqrt 3 ≤ k ∧ k ≤ real_sqrt 3 / 3) :
  ∃ (α : ℝ), α ∈ [0, real.pi / 6] ∪ [2 * real.pi / 3, real.pi) ∧ 
             tan α = k := sorry

end inclination_angle_range_l587_587478


namespace problem1_problem2_l587_587318

variable {A B C a b c : ℝ}
variable {S : ℝ}

-- Problem 1
theorem problem1 (cos_A : cos A = 4 / 5) : 
  sin (B + C) / 2 ^ 2 + cos (2 * A) = 59 / 50 := sorry

-- Problem 2
theorem problem2 (cos_A : cos A = 4 / 5)
  (b_eq : b = 2) (S_eq : S = 3) : 
  let sin_A := sqrt (1 - (cos_A ^ 2)) in
  let c := 5 in  -- Derived from area calculation
  a = sqrt (b ^ 2 + c ^ 2 - 2 * b * c * cos A) := sorry

end problem1_problem2_l587_587318


namespace find_r_l587_587307

theorem find_r (r : ℝ) (h₁ : 0 < r) (h₂ : ∀ x y : ℝ, (x - y = r → x^2 + y^2 = r → False)) : r = 2 :=
sorry

end find_r_l587_587307


namespace find_x0_l587_587273

noncomputable def f (x w : ℝ) := Real.sin (w * x) + Real.sqrt 3 * Real.cos (w * x)

theorem find_x0 (x₀ : ℝ) (w : ℝ) (h_w_pos : w > 0) (dist_adj_sym_axes : w = 2)  (h_symmetry : 2 * x₀ + Real.pi / 3 = Int.cast ((Int.of_nat n) * Real.pi)) (h_x₀_range : 0 ≤ x₀ ∧ x₀ ≤ Real.pi / 2) : 
  x₀ = Real.pi / 3 :=
sorry

end find_x0_l587_587273


namespace eval_fraction_l587_587921

theorem eval_fraction : (5^(-2) * 3^0) / 5^(-4) = 25 := by
  -- Placeholder for the proof
  sorry

end eval_fraction_l587_587921


namespace unique_markings_count_l587_587100

-- Definitions and conditions
def total_markings := {0, 1/3, 2/3, 1, 1/4, 1/2, 3/4, 1}

-- Statement of the problem
theorem unique_markings_count : total_markings.to_finset.card = 7 :=
by
  sorry

end unique_markings_count_l587_587100


namespace ship_distance_from_Y_l587_587147

-- Define the points and radii involved
variables (A B C X Y : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space X] [metric_space Y]
variables {r s : ℝ} -- r and s are radii of the semicircles
-- Include assumptions for distances and properties of paths
variables
  (h1 : ∀ (P : A), dist P X = r) -- Distance from any point on path AB to X
  (h2 : ∀ (P : B), dist P Y = s) -- Distance from any point on path BY to Y
  (h3 : ∀ (t : ℝ), 0 ≤ t → t ≤ 1 → ∃ (P : Y), dist P Y = s + t * (dist Y C - s)) -- Distance on straight line YC increases linearly

-- Define the goal as a theorem
theorem ship_distance_from_Y : 
  (∀ P ∈ ABO, dist P Y = r ∧ ∀ Q ∈ BYO, dist Q Y = s ∧ ∀ u ∈ YCO, dist u Y > s) → true :=
sorry

end ship_distance_from_Y_l587_587147


namespace solve_inequalities_l587_587034

/-- Solve the inequality system and find all non-negative integer solutions. -/
theorem solve_inequalities :
  { x : ℤ | 0 ≤ x ∧ 3 * (x - 1) < 5 * x + 1 ∧ (x - 1) / 2 ≥ 2 * x - 4 } = {0, 1, 2} :=
by
  sorry

end solve_inequalities_l587_587034


namespace sqrt_x_div_sqrt_y_l587_587564

theorem sqrt_x_div_sqrt_y (x y : ℝ)
  (h : ( ( (2/3)^2 + (1/6)^2 ) / ( (1/2)^2 + (1/7)^2 ) ) = 28 * x / (25 * y)) :
  (Real.sqrt x) / (Real.sqrt y) = 5 / 2 :=
sorry

end sqrt_x_div_sqrt_y_l587_587564


namespace sum_odd_integers_13_to_51_l587_587828

theorem sum_odd_integers_13_to_51: 
  (finset.range 20).sum (λ k, 13 + 2 * k) = 640 :=
by
  sorry

end sum_odd_integers_13_to_51_l587_587828


namespace dot_product_vec_a_vec_b_l587_587958

def vec_a : ℝ × ℝ := (-1, 2)
def vec_b : ℝ × ℝ := (1, 2)

theorem dot_product_vec_a_vec_b : vec_a.1 * vec_b.1 + vec_a.2 * vec_b.2 = 3 := by
  sorry

end dot_product_vec_a_vec_b_l587_587958


namespace find_h_k_a_b_l587_587702

/-- Define the foci -/
def F1 := (0, 2)
def F2 := (6, 2)

/-- Define the distance condition -/
def dist_condition (P : ℝ × ℝ) : Prop := 
  let dist (A B : ℝ × ℝ) := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  dist P F1 + dist P F2 = 8

/-- Define the ellipse -/
def ellipse_eq (x y h k a b : ℝ) : Prop := 
  (x - h)^2 / a^2 + (y - k)^2 / b^2 = 1

/-- State the problem as a theorem -/
theorem find_h_k_a_b :
  ∃ h k a b, F1 = (0, 2) ∧ F2 = (6, 2) ∧
              (∀ P, dist_condition P) ∧
              ellipse_eq (F1.1 + F2.1) / 2 (F1.2 + F2.2) / 2 h k a b ∧
              h + k + a + b = 9 + Real.sqrt 7 := by
  exists (3 : ℝ)
  exists (2 : ℝ)
  exists (4 : ℝ)
  exists (Real.sqrt 7)
  simp
  sorry

end find_h_k_a_b_l587_587702


namespace solid_with_isosceles_views_is_tetrahedron_l587_587659

/-- If the three views (front, top, and side) of a solid are all isosceles triangles, 
    then the solid could be a tetrahedron. -/
theorem solid_with_isosceles_views_is_tetrahedron (solid : Type) 
  (front_view_is_isosceles : isosceles_triangle_solid_front solid)
  (top_view_is_isosceles : isosceles_triangle_solid_top solid) 
  (side_view_is_isosceles : isosceles_triangle_solid_side solid) :
  solid_is_tetrahedron solid :=
sorry

end solid_with_isosceles_views_is_tetrahedron_l587_587659


namespace license_plate_combinations_l587_587174

theorem license_plate_combinations :
  ∃ (n : ℕ), 
    (n = ((26.choose 2) * 24 * (Nat.factorial 5 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 1)) * (10 * 9 * 8)))
    ∧ n = 5644800 :=
by
  sorry

end license_plate_combinations_l587_587174


namespace carrots_total_l587_587022

def carrots_grown_by_sally := 6
def carrots_grown_by_fred := 4
def total_carrots := carrots_grown_by_sally + carrots_grown_by_fred

theorem carrots_total : total_carrots = 10 := 
by 
  sorry  -- proof to be filled in

end carrots_total_l587_587022


namespace range_of_a_l587_587831

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → x^2 - 2*x + a < 0) ↔ a ≤ 0 :=
by sorry

end range_of_a_l587_587831


namespace time_for_A_to_complete_race_l587_587840

-- Define the main variables and conditions.
variable (v : ℝ) -- Speed of A
variable (t : ℝ) -- Time taken by A
variable (d_A : ℝ := 1000) -- Distance A runs (1000 meters)
variable (d_B : ℝ := 960) -- Distance B runs in the same time
variable (extra_time : ℝ := 8) -- Extra time B takes

-- Define the key constraints from the problem
def speed_A := d_A / t
def speed_B := d_B / (t + extra_time)

-- State the theorem under the given conditions
theorem time_for_A_to_complete_race : speed_A = speed_B → t = 200 :=
by
  sorry

end time_for_A_to_complete_race_l587_587840


namespace part1_part2_l587_587638

noncomputable def A (a : ℝ) : Set ℝ := {x : ℝ | 0 < a * x - 1 ∧ a * x - 1 ≤ 5}
def B : Set ℝ := {x : ℝ | -1/2 < x ∧ x ≤ 2}

-- Part (Ⅰ)
theorem part1 (a : ℝ) (ha : a = 1) : A a ∪ B = {x : ℝ | -1/2 < x ∧ x ≤ 6} := sorry

-- Part (Ⅱ)
theorem part2 (a : ℝ) (ha : A a ∩ B = ∅) (h₀ : 0 ≤ a) : a ∈ Icc 0 (1/2) := sorry

end part1_part2_l587_587638


namespace minimum_odd_count_l587_587805

def is_odd (n : ℤ) : Prop := n % 2 ≠ 0

theorem minimum_odd_count (a b c d e f : ℤ)
  (h1 : a + b + c = 30)
  (h2 : d + e = 18)
  (h3 : f = 13) :
  ∃ s t u v w x : ℕ, 
  (s = if is_odd a then 1 else 0) ∧ 
  (t = if is_odd b then 1 else 0) ∧ 
  (u = if is_odd c then 1 else 0) ∧ 
  (v = if is_odd d then 1 else 0) ∧ 
  (w = if is_odd e then 1 else 0) ∧ 
  (x = if is_odd f then 1 else 0) ∧ 
  s + t + u + v + w + x ≥ 3 :=
sorry

end minimum_odd_count_l587_587805


namespace employee_pays_204_l587_587518

-- Definitions based on conditions
def wholesale_cost : ℝ := 200
def markup_percent : ℝ := 0.20
def discount_percent : ℝ := 0.15

def retail_price := wholesale_cost * (1 + markup_percent)
def employee_payment := retail_price * (1 - discount_percent)

-- Theorem with the expected result
theorem employee_pays_204 : employee_payment = 204 := by
  -- Proof not required, we add sorry to avoid the proof details
  sorry

end employee_pays_204_l587_587518


namespace julia_spent_on_animals_l587_587692

theorem julia_spent_on_animals 
  (total_weekly_cost : ℕ)
  (weekly_cost_rabbit : ℕ)
  (weeks_rabbit : ℕ)
  (weeks_parrot : ℕ) :
  total_weekly_cost = 30 →
  weekly_cost_rabbit = 12 →
  weeks_rabbit = 5 →
  weeks_parrot = 3 →
  total_weekly_cost * weeks_parrot - weekly_cost_rabbit * weeks_parrot + weekly_cost_rabbit * weeks_rabbit = 114 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  linarith,
end

end julia_spent_on_animals_l587_587692


namespace f_neg1_eq_32_f_even_when_a_1_l587_587759

-- Definitions of the function and conditions
def f (x : ℝ) (a : ℝ) : ℝ :=
  x * (1 / (2^x - 1) + a / 2)

-- Given conditions:
axiom f_even : ∀ x : ℝ, f x a = f (-x) a
axiom f_value1 : f 1 a = 3 / 2

-- Prove 1:
theorem f_neg1_eq_32 : f (-1) a = 3 / 2 :=
by sorry

-- Prove 2 with specific a value:
theorem f_even_when_a_1 {x : ℝ} (hx : x ≠ 0) : 
  f x 1 = f (-x) 1 :=
by sorry

end f_neg1_eq_32_f_even_when_a_1_l587_587759


namespace caging_ways_l587_587498

def animals : List Char := ['A', 'B', 'C', 'D', 'E', 'F']

def cages : List Nat := [1, 2, 3, 4, 5, 6]

-- Condition: Cages 1 to 4 are too small for animals A, B, C, and D.
def small_cages : List Nat := [1, 2, 3, 4]

-- Condition: Cages 5 and 6 can contain animals A, B, C, and D.
def large_cages : List Nat := [5, 6]

theorem caging_ways : 
  ∀ (A B C D : Char) (E F : Char), 
  A ∈ animals → B ∈ animals → C ∈ animals → D ∈ animals → 
  E ∈ animals → F ∈ animals → 
  E ≠ A ∧ E ≠ B ∧ E ≠ C ∧ E ≠ D →
  F ≠ A ∧ F ≠ B ∧ F ≠ C ∧ F ≠ D → 
  E ≠ F → 
  4 * 3 * 2 = 24 :=    -- There are 4 choices for the first animal in a large cage and 3 for the second
by {
  repeat { intro },
  sorry  -- Proof not required as per instructions
}

end caging_ways_l587_587498


namespace skating_average_l587_587167

variable (minutesPerDay1 minutesPerDay2 : Nat)
variable (days1 days2 totalDays requiredAverage : Nat)

theorem skating_average :
  minutesPerDay1 = 80 →
  days1 = 6 →
  minutesPerDay2 = 100 →
  days2 = 2 →
  totalDays = 9 →
  requiredAverage = 95 →
  (minutesPerDay1 * days1 + minutesPerDay2 * days2 + x) / totalDays = requiredAverage →
  x = 175 :=
by
  intro h1 h2 h3 h4 h5 h6 h7
  sorry

end skating_average_l587_587167


namespace sally_trip_saving_l587_587026

def SallyTrip : Type :=
  { saved_amount : Nat := 28
  , parking_cost : Nat := 10
  , entrance_cost : Nat := 55
  , meal_cost : Nat := 25
  , souvenir_cost : Nat := 40
  , hotel_cost : Nat := 80
  , distance : Nat := 165
  , car_mpg : Nat := 30
  , gas_price : Nat := 3 }

def total_cost (trip : SallyTrip) : Nat :=
  trip.parking_cost + trip.entrance_cost + trip.meal_cost + trip.souvenir_cost + trip.hotel_cost +
  (2 * (trip.distance / trip.car_mpg) * trip.gas_price)

theorem sally_trip_saving (trip : SallyTrip) : total_cost trip - trip.saved_amount = 215 := by
  sorry

end sally_trip_saving_l587_587026


namespace find_possible_values_of_a_l587_587255

noncomputable def P : Set ℝ := {x | x^2 + x - 6 = 0}
noncomputable def Q (a : ℝ) : Set ℝ := {x | a * x + 1 = 0}

theorem find_possible_values_of_a (a : ℝ) (h : Q a ⊆ P) :
  a = 0 ∨ a = -1/2 ∨ a = 1/3 := by
  sorry

end find_possible_values_of_a_l587_587255


namespace Bryan_books_total_l587_587534

theorem Bryan_books_total (books_per_shelf : ℕ) (num_shelves : ℕ) :
  books_per_shelf = 56 → num_shelves = 9 → books_per_shelf * num_shelves = 504 :=
by
  intros h1 h2
  rw [h1, h2]
  calc
    56 * 9 = 504 := by norm_num

end Bryan_books_total_l587_587534


namespace distinct_remainders_l587_587372

theorem distinct_remainders (n : ℕ) (hn : n % 2 = 1) :
  ∃ (a b : ℕ → ℕ), 
    (∀ i, 1 ≤ i ∧ i ≤ n → a i = 3 * i ∧ b i = 3 * i + 1) ∧ 
    (∀ k, 0 < k ∧ k < n → 
      (∀ i, 1 ≤ i ∧ i ≤ n → ∀ j, 1 ≤ j ∧ j ≤ n → 
        (i ≠ j → 
          a i + a j ≠ a j + a i ∧
          a i + b j ≠ b j + a i ∧
          b i + b j ≠ b j + b (i + k) [MOD 3 * n]))) := 
sorry

end distinct_remainders_l587_587372


namespace magnitude_of_product_l587_587559

-- Variables and conditions definition
def z1 : ℂ := 5 - 3 * complex.I
def z2 : ℂ := 7 + 24 * complex.I

-- The theorem statement
theorem magnitude_of_product (z1 z2 : ℂ) (h1 : z1 = 5 - 3 * complex.I) (h2 : z2 = 7 + 24 * complex.I) : 
  |z1 * z2| = 25 * (√34) :=
by
  rw [h1, h2]
  -- Sorry for not providing the complete proof
  sorry

end magnitude_of_product_l587_587559


namespace math_problem_l587_587282

noncomputable def f (x : ℝ) := |Real.exp x - 1|

theorem math_problem (x1 x2 : ℝ) (h1 : x1 < 0) (h2 : x2 > 0)
  (h3 : - Real.exp x1 * Real.exp x2 = -1) :
  (x1 + x2 = 0) ∧
  (0 < (Real.exp x2 + Real.exp x1 - 2) / (x2 - x1)) ∧
  (0 < Real.exp x1 ∧ Real.exp x1 < 1) :=
by
  sorry

end math_problem_l587_587282


namespace minimum_value_of_c_l587_587756

-- Defining the conditions:
def conditions (a b c : ℕ) : Prop :=
a < b ∧ b < c ∧ ∀ x : ℕ, ∃! (y : ℕ), (2 * x + y = 2023 ∧ y = |x - a| + |x - b| + |x - c|)

-- Statement of the proof problem:
theorem minimum_value_of_c (a b c : ℕ) (h : conditions a b c) : c = 1012 :=
sorry

end minimum_value_of_c_l587_587756


namespace floor_sum_arith_sequence_l587_587185

theorem floor_sum_arith_sequence :
  (∑ i in Finset.range 209, (Nat.floor (1 + 0.5 * i))) = 11025 := by
  sorry

end floor_sum_arith_sequence_l587_587185


namespace count_three_digit_numbers_sum_24_l587_587229

theorem count_three_digit_numbers_sum_24 : 
  {n : ℕ // 100 ≤ n ∧ n < 1000 ∧ (n.digits 10).sum = 24}.card = 4 := 
by
  sorry

end count_three_digit_numbers_sum_24_l587_587229


namespace expression_value_l587_587196

theorem expression_value :
  2 - (-3) - 4 - (-5) - 6 - (-7) * 2 = 14 :=
by sorry

end expression_value_l587_587196


namespace smallest_prime_divisor_of_sum_l587_587465

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end smallest_prime_divisor_of_sum_l587_587465


namespace sum_and_average_of_primes_between_20_and_40_l587_587827

def is_prime (n : Nat) : Prop :=
  n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def primes_in_range (start end_ : Nat) : List Nat :=
  List.filter is_prime (List.range' start (end_ - start + 1))

theorem sum_and_average_of_primes_between_20_and_40 :
  let primes := primes_in_range 20 40
  primes = [23, 29, 31, 37] ∧
  List.sum primes = 120 ∧
  List.sum primes / primes.length = 30 := by
  let primes := primes_in_range 20 40
  have h1 : primes = [23, 29, 31, 37] := sorry
  have h2 : List.sum primes = 120 := sorry
  have h3 : List.sum primes / primes.length = 30 := sorry
  exact ⟨h1, h2, h3⟩

end sum_and_average_of_primes_between_20_and_40_l587_587827


namespace dave_more_than_derek_l587_587912

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l587_587912


namespace angle_ABC_of_regular_octagon_and_square_l587_587880

theorem angle_ABC_of_regular_octagon_and_square 
  (octagon : Polygon (Fin 8) ℝ)
  (h_regular : RegularPolygon octagon)
  (square : Polygon (Fin 4) ℝ)
  (h_square_outward : ConstructedSquareOutward octagon 1 side.square)
  (B : Point ℝ)
  (h_diagonal_intersection : IsDiagonalIntersection octagon B)
  : MeasureOfAngle octagon square B = 22.5 :=
by sorry

end angle_ABC_of_regular_octagon_and_square_l587_587880


namespace metal_beams_per_panel_l587_587858

theorem metal_beams_per_panel (panels sheets_per_panel rods_per_sheet rods_needed beams_per_panel rods_per_beam : ℕ)
    (h1 : panels = 10)
    (h2 : sheets_per_panel = 3)
    (h3 : rods_per_sheet = 10)
    (h4 : rods_needed = 380)
    (h5 : rods_per_beam = 4)
    (h6 : beams_per_panel = 2) :
    (panels * sheets_per_panel * rods_per_sheet + panels * beams_per_panel * rods_per_beam = rods_needed) :=
by
  sorry

end metal_beams_per_panel_l587_587858


namespace quadratic_decreasing_interval_l587_587287

theorem quadratic_decreasing_interval (b c : ℝ) 
  (h1 : (1 : ℝ)^2 + b * 1 + c = 0) 
  (h2 : (3 : ℝ)^2 + b * 3 + c = 0) : 
  ∀ x, x < 2 → (y = x^2 + b * x + c) < y :=
sorry

end quadratic_decreasing_interval_l587_587287


namespace solve_equation_l587_587217

theorem solve_equation :
  {x : ℝ | (15 * x - x^2)/(x + 2) * (x + (15 - x)/(x + 2)) = 54 } = {12, -3, -3 + real.sqrt 33, -3 - real.sqrt 33} :=
by
  sorry

end solve_equation_l587_587217


namespace abc_divides_sum_exp21_l587_587698

theorem abc_divides_sum_exp21
  (a b c : ℕ)
  (ha : a > 0)
  (hb : b > 0)
  (hc : c > 0)
  (hab : a ∣ b^4)
  (hbc : b ∣ c^4)
  (hca : c ∣ a^4)
  : abc ∣ (a + b + c)^21 :=
by
sorry

end abc_divides_sum_exp21_l587_587698


namespace count_three_digit_sum_24_l587_587232

theorem count_three_digit_sum_24 : 
  let count := (λ n : ℕ, let a := n / 100, b := (n / 10) % 10, c := n % 10 in
                        a + b + c = 24 ∧ 100 ≤ n ∧ n < 1000) in
  (finset.range 1000).filter count = 8 :=
sorry

end count_three_digit_sum_24_l587_587232


namespace distance_from_left_focus_to_line_l587_587269

theorem distance_from_left_focus_to_line : 
  let ellipse_eq := ∀ x y : ℝ, x^2 / 4 + y^2 / 3 = 1 in
  let upper_vertex := (0, Real.sqrt 3) in
  let right_focus := (1, 0) in
  let line_eq := ∀ x y : ℝ, Real.sqrt 3 * x + y - Real.sqrt 3 = 0 in
  let left_focus := (-1, 0) in
  ∀ (x y : ℝ), 
  ellipse_eq x y → 
  upper_vertex = (0, Real.sqrt 3) →
  right_focus = (1, 0) → 
  line_eq x y →
  left_focus = (-1, 0) →
  distance_from_point_to_line left_focus line_eq = Real.sqrt 3 :=
sorry

end distance_from_left_focus_to_line_l587_587269


namespace range_of_a_l587_587601

theorem range_of_a 
  (a : ℕ) 
  (an : ℕ → ℕ)
  (Sn : ℕ → ℕ)
  (h1 : a_1 = a)
  (h2 : ∀ n : ℕ, n ≥ 2 → Sn n + Sn (n - 1) = 4 * n^2)
  (h3 : ∀ n : ℕ, an n < an (n + 1)) : 
  3 < a ∧ a < 5 :=
by
  sorry

end range_of_a_l587_587601


namespace max_XYZ_plus_terms_l587_587719

theorem max_XYZ_plus_terms {X Y Z : ℕ} (h : X + Y + Z = 15) :
  X * Y * Z + X * Y + Y * Z + Z * X ≤ 200 :=
sorry

end max_XYZ_plus_terms_l587_587719


namespace sasha_hometown_name_l587_587455

theorem sasha_hometown_name :
  ∃ (sasha_hometown : String), 
  (∃ (vadik_last_column : String), vadik_last_column = "ВКСАМО") →
  (∃ (sasha_transformed : String), sasha_transformed = "мТТЛАРАЕкис") →
  (∃ (sasha_starts_with : Char), sasha_starts_with = 'с') →
  sasha_hometown = "СТЕРЛИТАМАК" :=
by
  sorry

end sasha_hometown_name_l587_587455


namespace find_cans_lids_l587_587157

-- Define the given conditions
def total_lids (x : ℕ) : ℕ := 14 + 3 * x

-- Define the proof problem
theorem find_cans_lids (x : ℕ) (h : total_lids x = 53) : x = 13 :=
sorry

end find_cans_lids_l587_587157


namespace min_distance_M_to_y_axis_l587_587411

-- Definitions
def parabola (x y : ℝ) : Prop := y^2 = x
def length_fixed (A B : ℝ × ℝ) : Prop := real.dist A B = 3

-- Theorem statement
theorem min_distance_M_to_y_axis {A B : ℝ × ℝ} 
  (hA : parabola A.1 A.2) 
  (hB : parabola B.1 B.2) 
  (h_length : length_fixed A B) : 
  (\(x_mid := (A.1 + B.1)/2),
  |x_mid| = 5/4) :=
  sorry

end min_distance_M_to_y_axis_l587_587411


namespace sum_of_q_p_values_l587_587373

def p (x : ℤ) : ℤ := |x| + 1
def q (x : ℤ) : ℤ := -|x - 1|

theorem sum_of_q_p_values :
  (List.sum (List.map (λ x, q (p x)) [-5, -4, -3, -2, -1, 0, 1, 2, 3])) = -21 := 
  by
  sorry

end sum_of_q_p_values_l587_587373


namespace volume_of_water_overflow_l587_587863

-- Definitions based on given conditions
def mass_of_ice : ℝ := 50
def density_of_fresh_ice : ℝ := 0.9
def density_of_salt_ice : ℝ := 0.95
def density_of_fresh_water : ℝ := 1
def density_of_salt_water : ℝ := 1.03

-- Theorem statement corresponding to the problem
theorem volume_of_water_overflow
  (m : ℝ := mass_of_ice) 
  (rho_n : ℝ := density_of_fresh_ice) 
  (rho_c : ℝ := density_of_salt_ice) 
  (rho_fw : ℝ := density_of_fresh_water) 
  (rho_sw : ℝ := density_of_salt_water) :
  ∃ (ΔV : ℝ), ΔV = 2.63 :=
by
  sorry

end volume_of_water_overflow_l587_587863


namespace hired_waiters_l587_587172

theorem hired_waiters (W H : Nat) (hcooks : Nat := 9) 
                      (initial_ratio : 3 * W = 11 * hcooks)
                      (new_ratio : 9 = 5 * (W + H)) 
                      (original_waiters : W = 33) 
                      : H = 12 :=
by
  sorry

end hired_waiters_l587_587172


namespace jelly_beans_remaining_l587_587071

theorem jelly_beans_remaining :
  let initial_jelly_beans := 8000
  let num_first_group := 6
  let num_last_group := 4
  let last_group_took := 400
  let first_group_took := 2 * last_group_took
  let last_group_total := last_group_took * num_last_group
  let first_group_total := first_group_took * num_first_group
  let remaining_jelly_beans := initial_jelly_beans - (first_group_total + last_group_total)
  remaining_jelly_beans = 1600 :=
by {
  -- Define the initial number of jelly beans
  let initial_jelly_beans := 8000
  -- Number of people in first and last groups
  let num_first_group := 6
  let num_last_group := 4
  -- Jelly beans taken by last group and first group per person
  let last_group_took := 400
  let first_group_took := 2 * last_group_took
  -- Calculate total jelly beans taken by last group and first group
  let last_group_total := last_group_took * num_last_group
  let first_group_total := first_group_took * num_first_group
  -- Jelly beans remaining
  let remaining_jelly_beans := initial_jelly_beans - (first_group_total + last_group_total)
  -- Proof of the theorem
  show remaining_jelly_beans = 1600, from sorry
}

end jelly_beans_remaining_l587_587071


namespace Matias_sales_l587_587000

def books_sold (Tuesday Wednesday Thursday : Nat) : Prop :=
  Tuesday = 7 ∧ 
  Wednesday = 3 * Tuesday ∧ 
  Thursday = 3 * Wednesday ∧ 
  Tuesday + Wednesday + Thursday = 91

theorem Matias_sales
  (Tuesday Wednesday Thursday : Nat) :
  books_sold Tuesday Wednesday Thursday := by
  sorry

end Matias_sales_l587_587000


namespace fedya_stick_problem_l587_587111

theorem fedya_stick_problem (a b c : ℝ) (h : a + b + c > 0) :
  (∃ t > 1, t^3 = t^2 + t + 1) →
  (∃ a b c : ℝ, 
    (c = max a b c) → 
    (¬ (a + b > c ∧ a + c > b ∧ b + c > a)) → 
    (a + b ≠ 0 ∧ a + c ≠ 0 ∧ b + c ≠ 0) → 
    ∀ n : ℕ, ∃ a b c : ℝ, 
      (c = max a b c) ∧ 
      (¬ (a + b > c ∧ a + c > b ∧ b + c > a)) ∧ 
      (a + b ≠ 0 ∧ a + c ≠ 0 ∧ b + c ≠ 0)) :=
sorry

end fedya_stick_problem_l587_587111


namespace volume_cone_not_equal_cylinder_minimal_k_value_l587_587959

-- Define the given values: radius of the base of the cone, height of the cone, and radius of the inscribed sphere.
variables (R m r : ℝ)

-- Calculate the volumes of the cone and the cylinder.
def V1 := (1 / 3) * π * R^2 * m
def V2 := 2 * π * r^3

-- Define the math proof statements for Lean 4
theorem volume_cone_not_equal_cylinder : V1 R m ≠ V2 r :=
sorry

theorem minimal_k_value : ∃ k, k = (4 / 3) ∧ (V1 R m = k * V2 r) :=
sorry

end volume_cone_not_equal_cylinder_minimal_k_value_l587_587959


namespace cannot_form_set_of_elderly_people_l587_587890

noncomputable def can_form_set (A : Type*) (P : A → Prop) : Prop := ∃ S : set A, ∀ x, P x → x ∈ S

def positive_numbers := {x : ℝ // 0 < x}

def non_zero_real_numbers := {x : ℝ // x ≠ 0}

def four_great_inventions := {x : string // x = "compass" ∨ x = "gunpowder" ∨ x = "papermaking" ∨ x = "printing"}

constant elderly_person : Type 

constant is_elderly : elderly_person → Prop

theorem cannot_form_set_of_elderly_people :
  ¬ can_form_set elderly_person is_elderly :=
sorry

end cannot_form_set_of_elderly_people_l587_587890


namespace minimum_value_of_function_l587_587572

theorem minimum_value_of_function :
  ∃ x : ℝ, (sin x)^2 + (cos x)^2 = 1 ∧ 
           ∀ y : ℝ, y = (4 / (cos x)^2) + (9 / (sin x)^2) → y ≥ 25 :=
by
  sorry

end minimum_value_of_function_l587_587572


namespace number_of_unmarked_trees_l587_587748

noncomputable def isMarkedOnWayTo : ℕ → Prop :=
λ n, n % 5 = 1

noncomputable def isMarkedOnWayBack : ℕ → Prop :=
λ n, n % 8 = 1

noncomputable def unmarkedTrees (totalTrees : ℕ) :=
let marked_trees := Finset.filter isMarkedOnWayTo (Finset.range (totalTrees + 1)) ∪ Finset.filter isMarkedOnWayBack (Finset.range (totalTrees + 1))
in totalTrees - marked_trees.card

theorem number_of_unmarked_trees :
  unmarkedTrees 200 = 140 :=
by sorry

end number_of_unmarked_trees_l587_587748


namespace geometric_sequence_ratio_q3_l587_587595

theorem geometric_sequence_ratio_q3 (a : ℕ → ℝ) (q : ℝ) (h : q ≠ 1) (h_geom : ∀ n, a (n + 1) = q * a n) :
  let b := λ n, a (3*n - 2) + a (3*n - 1) + a (3*n)
  in ∀ n, b (n + 1) = q^3 * b n :=
by sorry

end geometric_sequence_ratio_q3_l587_587595


namespace part1_part2_ge3_part2_between2and3_l587_587630

def f (a b : ℝ) (x : ℝ) : ℝ :=
  |a * x - 2| + b * Real.log x

-- Problem (1)
theorem part1 (b : ℝ) (h₁ : b ≥ 2) : 
  ∀ (x > 0), f 1 b x = f 1 b x :=
sorry

-- Problem (2)
theorem part2_ge3 (a b : ℝ) (h₁ : a ≥ 3) (h₂ : b = 1) :
  ∃! x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 1 / x :=
sorry

theorem part2_between2and3 (a b : ℝ) (h₁ : 2 ≤ a ∧ a < 3) (h₂ : b = 1) :
  ¬ ∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ f a b x = 1 / x :=
sorry

end part1_part2_ge3_part2_between2and3_l587_587630


namespace part_a_part_b_part_c_l587_587501

-- Part (a)
theorem part_a (equally_likely_births : True) (two_children : True) : 
  let outcomes := [("boy", "boy"), ("boy", "girl"), ("girl", "boy"), ("girl", "girl")]
  let equal_probability := 1 / 4
  let favorable_outcomes := [("boy", "girl"), ("girl", "boy")]
  (list.length favorable_outcomes / list.length outcomes) = 1 / 2 := 
begin
  sorry
end

-- Part (b)
theorem part_b (equally_likely_births : True) (two_children : True) (one_child_is_boy : True) :
  let remaining_outcomes := [("boy", "boy"), ("boy", "girl"), ("girl", "boy")]
  let equal_probability := 1 / 3
  let favorable_outcomes := [("boy", "girl"), ("girl", "boy")]
  (list.length favorable_outcomes / list.length remaining_outcomes) = 2 / 3 :=
begin
  sorry
end

-- Part (c)
theorem part_c (equally_likely_births : True) (two_children : True) (one_boy_born_monday : True) :
  let total_outcomes := (6 + 7 + 7) -- 13 unique scenarios
  let favorable_outcomes := 13
  favorable_outcomes / total_outcomes = 13 / 27 := 
begin
  sorry
end

end part_a_part_b_part_c_l587_587501


namespace correct_statement_l587_587095

-- Definitions based on conditions
def input_stmt_1 : Prop := "x=3"
def input_stmt_2 : Prop := "INPUT \"A, B, C\"; a, b, c"
def output_stmt : Prop := "PRINT A+B=C"
def assign_stmt : Prop := "3=A"

-- Question: Which of the following is the correct statement? Prove that option "②" is correct.
theorem correct_statement : input_stmt_2 ∧ ¬input_stmt_1 ∧ ¬output_stmt ∧ ¬assign_stmt :=
by 
  -- Following the Lean 4 syntax, clear out the proof steps.
  sorry

end correct_statement_l587_587095


namespace sum_of_digits_of_x_l587_587119

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits = digits.reverse

theorem sum_of_digits_of_x (x : ℕ) (h1 : is_palindrome x) (h2 : is_palindrome (x + 10)) (h3 : 100 ≤ x ∧ x ≤ 999) : 
  x.digits 10.sum = 19 :=
begin
  sorry
end

end sum_of_digits_of_x_l587_587119


namespace smallest_number_l587_587460

theorem smallest_number (x : ℕ) (h1 : (x + 7) % 8 = 0) (h2 : (x + 7) % 11 = 0) (h3 : (x + 7) % 24 = 0) : x = 257 :=
sorry

end smallest_number_l587_587460


namespace valid_a_intervals_and_length_l587_587210

variable (a : ℝ)
def f (x : ℝ) := a * x^2 - 4 * a * x + 1

-- Statement of the problem to prove
theorem valid_a_intervals_and_length :
  (∀ x ∈ Icc (0 : ℝ) 4, |f a x| ≤ 3) →
  ((a ∈ Icc (-(1 : ℝ) / 2) 0 ∧ a ≠ 0) ∨ (a ∈ Icc (0 : ℝ) 1)) ∧
  (interval_length (Icc (-(1 : ℝ) / 2) 0) + interval_length (Icc (0) 1) = 1.5) :=
by
  sorry

namespace Mathlib
open Set

def interval_length (I : Set ℝ) : ℝ :=
  if h : ∃ a b, I = Icc a b then classical.some h.2 - classical.some h else 0

end valid_a_intervals_and_length_l587_587210


namespace difference_of_squirrels_and_nuts_l587_587436

-- Definitions
def number_of_squirrels : ℕ := 4
def number_of_nuts : ℕ := 2

-- Theorem statement with conditions and conclusion
theorem difference_of_squirrels_and_nuts : number_of_squirrels - number_of_nuts = 2 := by
  sorry

end difference_of_squirrels_and_nuts_l587_587436


namespace sum_of_primitive_roots_mod_11_l587_587470

theorem sum_of_primitive_roots_mod_11 : 
  (∑ x in {1, 2, 3, 4, 5, 6, 7, 8}, if is_primitive_root 11 x then x else 0) = 23 :=
by sorry

end sum_of_primitive_roots_mod_11_l587_587470


namespace circle_parabola_intersect_l587_587175

theorem circle_parabola_intersect (a : ℝ) :
  (∀ (x y : ℝ), x^2 + (y - 1)^2 = 1 ∧ y = a * x^2 → (x ≠ 0 ∨ y ≠ 0)) ↔ a > 1 / 2 :=
by
  sorry

end circle_parabola_intersect_l587_587175


namespace partitionable_triples_l587_587957

theorem partitionable_triples (n : ℕ) (h : n ∈ Finset.range 3910 ∩ Finset.Ico 3900 3910) :
  (n = 3900 ∨ n = 3903) ↔ 
  ∃ (triples : Finset (Finset ℕ)), (∀ t ∈ triples, ∃ a b c, t = {a, b, c} ∧ (a + b = c ∨ b + c = a ∨ c + a = b)) ∧ 
  (∪ t ∈ triples, t) = Finset.range (n + 1) :=
sorry

end partitionable_triples_l587_587957


namespace increasing_interval_f_l587_587421

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem increasing_interval_f :
  ∀ x, (2 < x) → (∃ ε > 0, ∀ δ > 0, δ < ε → f (x + δ) ≥ f x) :=
by
  sorry

end increasing_interval_f_l587_587421


namespace part1_part2_l587_587677

-- Defining the geometric configuration
structure Config :=
  (A B C D E: Point)
  (circle : Circle)
  (inscribed : inscribed_in_circle A B C D circle) -- Quadrilateral inscribed in the circle
  (E_on_AC : on_line E A C)
  (relation : AB * CD = AD * BC)

-- Part 1: Prove that \(\angle ABE = \angle DBC\) given \(E\) is the midpoint of \(AC\)
theorem part1 (cfg : Config) (midpoint_E : midpoint cfg.A cfg.C cfg.E) : 
  angle cfg.A cfg.B cfg.E = angle cfg.D cfg.B cfg.C := 
sorry

-- Part 2: Prove that \(E\) is the midpoint of \(AC\) given \(\angle ABE = \angle DBC\)
theorem part2 (cfg : Config) (angle_eq : angle cfg.A cfg.B cfg.E = angle cfg.D cfg.B cfg.C) :
  midpoint cfg.A cfg.C cfg.E := 
sorry

end part1_part2_l587_587677


namespace friends_choose_rooms_l587_587866

theorem friends_choose_rooms :
  let rooms := 5
  let friends := 5
  hotelsPossibleConfigurations rooms friends = 2220 :=
by
  sorry

end friends_choose_rooms_l587_587866


namespace cone_surface_area_eq_l587_587946

noncomputable theory

-- Define the central angle and the radius of the sector
def theta : ℝ := π / 2
def R : ℝ := 1

-- Define the area of the circular sector
def sector_area : ℝ := π * R^2 * (theta / (2 * π))

-- Define the radius of the base of the cone
def r : ℝ := 1 / 4

-- Define the area of the base of the cone
def base_area : ℝ := π * r^2

-- Define the total surface area of the cone
def cone_total_surface_area : ℝ := sector_area + base_area

-- Provide the theorem statement to prove the total surface area of the cone
theorem cone_surface_area_eq : cone_total_surface_area = 5 * π / 16 :=
by
  sorry

end cone_surface_area_eq_l587_587946


namespace joeys_age_next_multiple_l587_587348

-- Definitions of the conditions and problem setup
def joey_age (chloe_age : ℕ) : ℕ := chloe_age + 2
def max_age : ℕ := 2
def is_prime (n : ℕ) : Prop := Nat.Prime n

-- Sum of digits function
def sum_of_digits (n : ℕ) : ℕ := n.digits 10 |>.sum

-- Main Lean statement
theorem joeys_age_next_multiple (chloe_age : ℕ) (H1 : is_prime chloe_age)
  (H2 : ∀ n : ℕ, (joey_age chloe_age + n) % (max_age + n) = 0)
  (H3 : ∀ i : ℕ, i < 11 → is_prime (chloe_age + i))
  : sum_of_digits (joey_age chloe_age + 1) = 5 :=
  sorry

end joeys_age_next_multiple_l587_587348


namespace identity_holds_for_all_a_b_l587_587653

theorem identity_holds_for_all_a_b (a b : ℝ) :
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  x^2 + y^2 = 2 * (z^2 + t^2) :=
by {
  let x := a + 5 * b
  let y := 5 * a - b
  let z := 3 * a - 2 * b
  let t := 2 * a + 3 * b
  sorry
}

end identity_holds_for_all_a_b_l587_587653


namespace math_problem_l587_587590

theorem math_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 9 * x + y - x * y = 0) : 
  ((9 * x + y) * (9 / y + 1 / x) = x * y) ∧ ¬ ((x / 9) + y = 10) ∧ 
  ((x + y = 16) ↔ (x = 4 ∧ y = 12)) ∧ 
  ((x * y = 36) ↔ (x = 2 ∧ y = 18)) :=
by {
  sorry
}

end math_problem_l587_587590


namespace square_garden_dimensions_and_area_increase_l587_587872

def original_length : ℝ := 60
def original_width : ℝ := 20

def original_area : ℝ := original_length * original_width
def original_perimeter : ℝ := 2 * (original_length + original_width)

theorem square_garden_dimensions_and_area_increase
    (L : ℝ := 60) (W : ℝ := 20)
    (orig_area : ℝ := L * W)
    (orig_perimeter : ℝ := 2 * (L + W))
    (square_side_length : ℝ := orig_perimeter / 4)
    (new_area : ℝ := square_side_length * square_side_length)
    (area_increase : ℝ := new_area - orig_area) :
    square_side_length = 40 ∧ area_increase = 400 :=
by {sorry}

end square_garden_dimensions_and_area_increase_l587_587872


namespace range_alpha_sub_beta_l587_587645

theorem range_alpha_sub_beta (α β : ℝ) (h₁ : -π/2 < α) (h₂ : α < β) (h₃ : β < π/2) : -π < α - β ∧ α - β < 0 := by
  sorry

end range_alpha_sub_beta_l587_587645


namespace lateral_surface_area_cylinder_l587_587761

-- Define the problem conditions
def side_length : ℝ := 1
def rotation_axis := side_length -- The axis lies on one of the sides.

-- Define the derived quantity
def lateral_surface_area (side_length : ℝ) : ℝ := side_length * 2 * Real.pi * side_length

-- State the theorem that we want to prove
theorem lateral_surface_area_cylinder :
  lateral_surface_area side_length = 2 * Real.pi :=
sorry

end lateral_surface_area_cylinder_l587_587761


namespace planted_fraction_l587_587931

theorem planted_fraction (a b : ℕ) (hypotenuse : ℚ) (distance_to_hypotenuse : ℚ) (x : ℚ)
  (h_triangle : a = 5 ∧ b = 12 ∧ hypotenuse = 13)
  (h_distance : distance_to_hypotenuse = 3)
  (h_x : x = 39 / 17)
  (h_square_area : x^2 = 1521 / 289)
  (total_area : ℚ) (planted_area : ℚ)
  (h_total_area : total_area = 30)
  (h_planted_area : planted_area = 7179 / 289) :
  planted_area / total_area = 2393 / 2890 :=
by
  sorry

end planted_fraction_l587_587931


namespace emily_small_gardens_l587_587205

theorem emily_small_gardens 
  (total_seeds : ℕ)
  (seeds_in_big_garden : ℕ)
  (seeds_per_small_garden : ℕ)
  (remaining_seeds := total_seeds - seeds_in_big_garden)
  (number_of_small_gardens := remaining_seeds / seeds_per_small_garden) :
  total_seeds = 41 → seeds_in_big_garden = 29 → seeds_per_small_garden = 4 → number_of_small_gardens = 3 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end emily_small_gardens_l587_587205


namespace largest_number_in_set_l587_587525

theorem largest_number_in_set :
  let s := {3, -7, 0, (1/9 : ℚ)} in
  ∃ x ∈ s, ∀ y ∈ s, x >= y ∧ x = 3 := by
  sorry

end largest_number_in_set_l587_587525


namespace sufficient_but_not_necessary_l587_587834

theorem sufficient_but_not_necessary (x y : ℝ) : 
  (x ≥ 2 ∧ y ≥ 2) → x + y ≥ 4 ∧ (¬ (x + y ≥ 4 → x ≥ 2 ∧ y ≥ 2)) :=
by
  sorry

end sufficient_but_not_necessary_l587_587834


namespace volume_of_tetrahedron_l587_587930

theorem volume_of_tetrahedron (A B C D : Type) 
    (area_ABC : ℝ) (area_BCD : ℝ) (BC : ℝ) (angle : ℝ)
    (h1 : area_ABC = 150)
    (h2 : area_BCD = 90)
    (h3 : BC = 10)
    (h4 : angle = π / 4) :
    ∃ V : ℝ, V = 450 * real.sqrt 2 :=
begin
  sorry
end

end volume_of_tetrahedron_l587_587930


namespace manuscript_fee_correct_l587_587786

noncomputable def manuscript_fee (tax_paid : ℕ) : ℕ :=
if tax_paid = 420 then 3800 else 0

theorem manuscript_fee_correct (tax_paid : ℕ) (fee : ℕ) :
  tax_paid = 420 ∧ (fee ≤ 800 → tax_paid = 0) ∧
  (800 < fee ∧ fee ≤ 4000 → tax_paid = 0.14 * (fee - 800)) ∧
  (4000 < fee → tax_paid = 0.11 * fee) → fee = 3800 :=
by
  intros h
  cases h with h_tax_paid h_conditions
  cases h_conditions with h_condition1 h_conditions2
  cases h_conditions2 with h_condition2 h_condition3
  rw h_tax_paid at *
  have : fee > 800,
  { intro contra,
    have h_tax := h_condition1 contra,
    rw h_tax_paid at h_tax,
    linarith, },
  have : fee ≤ 4000 ∨ fee > 4000 := le_or_gt fee 4000,
  cases this,
  { have h_fee_4000 := h_condition2 (and.intro (by linarith) this),
    have : 0.14 * (fee - 800) = 420 := h_fee_4000.symm,
    field_simp at this,
    linarith, },
  { have h_fee_gt_4000 := h_condition3 this,
    have : 0.11 * fee = 420 := h_fee_gt_4000.symm,
    field_simp at this,
    linarith, }
  sorry -- This will be filled in with the actual proof later.

end manuscript_fee_correct_l587_587786


namespace trajectory_of_P_is_ellipse_exists_k_such_that_cd_passes_origin_l587_587251

-- Define the initial conditions
def circleA_eq := ∀ (x y : ℝ), (x + Real.sqrt 2)^2 + y^2 = 12
def pointB : ℝ × ℝ := (Real.sqrt 2, 0)
def is_tangent_internally (P : ℝ × ℝ) (r : ℝ) :=
  let A : ℝ × ℝ := (-Real.sqrt 2, 0) in
  let PA := Real.sqrt ((P.1 + Real.sqrt 2)^2 + P.2^2) in
  let PB := Real.sqrt ((P.1 - Real.sqrt 2)^2 + P.2^2) in
  PA + PB = 2 * Real.sqrt 3 ∧ PB = r

-- Prove that the trajectory of the center of circle P is an ellipse
theorem trajectory_of_P_is_ellipse :
  ∀ (x y : ℝ), ((∃ r : ℝ, is_tangent_internally (x, y) r) ∧ 
                 ((x - pointB.1)^2 + y^2 = r^2)) ↔ (x^2 / 3 + y^2 = 1) := sorry

-- Second part of the problem: Prove the existence of certain k
theorem exists_k_such_that_cd_passes_origin :
  ∃ (k : ℝ), ∀ (x₁ y₁ x₂ y₂ : ℝ),
    ((∃ x₁ x₂ : ℝ, 
      (1 + 3*k^2) * x₁^2 + 12*k*x₁ + 9 = 0 ∧
      y₁ = k*x₁ + 2 ∧
      y₂ = k*x₂ + 2 ∧
      y₁*y₂ + x₁*x₂ = 0) ∧ x₁ + x₂ = -12*k / (1 + 3*k^2) ∧ x₁ * x₂ = 9 / (1 + 3*k^2))
    ↔ (k = Real.sqrt 39 / 3 ∨ k = -Real.sqrt 39 / 3) := sorry

end trajectory_of_P_is_ellipse_exists_k_such_that_cd_passes_origin_l587_587251


namespace general_formula_a_n_sum_T_n_l587_587725

-- Define the initial conditions for the arithmetic sequence {a_n}
variables (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom a2 : ∀ (n : ℕ), a (2 * n) = 2 * a n + 1
axiom S4_eq_4S2 : S 4 = 4 * S 2

-- Prove the formula for the arithmetic sequence {a_n}
theorem general_formula_a_n : ∀ (n : ℕ), a n = 2 * n - 1 :=
sorry

-- Define the initial conditions for the sequence {b_n}
variables (b : ℕ → ℝ) (T : ℕ → ℝ)
axiom sum_condition : ∀ (n : ℕ), (finset.range n).sum (λ i, b (i + 1) / a (i + 1)) = 1 - 1 / (2^n)
axiom b_values : ∀ (n : ℕ), if n = 1 then b 1 = 1 / 2 else b n = (2 * n - 1) / (2^n)

-- Prove the sum of the first n terms of {b_n}
theorem sum_T_n : ∀ (n : ℕ), T n = 3 - (2 * n + 3) / (2^n) :=
sorry

end general_formula_a_n_sum_T_n_l587_587725


namespace number_of_three_digit_whole_numbers_with_digit_sum_24_l587_587235

theorem number_of_three_digit_whole_numbers_with_digit_sum_24 : 
  (finset.filter (λ n, (let a := n / 100 in 
                        let b := (n % 100) / 10 in 
                        let c := n % 10 in 
                        a + b + c = 24) 
                  (finset.Icc 100 999)).card = 4 := 
begin
  sorry
end

end number_of_three_digit_whole_numbers_with_digit_sum_24_l587_587235


namespace find_number_l587_587869

theorem find_number (x : ℤ) (h : 16 * x = 32) : x = 2 :=
sorry

end find_number_l587_587869


namespace cement_percentage_final_mixture_l587_587836

noncomputable def final_cement_percentage (total_weight : ℝ) (high_cement_weight : ℝ) (high_cement_percentage : ℝ) (low_cement_percentage : ℝ) : ℝ :=
  let high_cement_amount := high_ccement_weight * high_cement_percentage
  let low_cement_weight := total_weight - high_cement_weight
  let low_cement_amount := low_cement_weight * low_cement_percentage
  let total_cement_amount := high_cement_amount + low_cement_amount
  (total_cement_amount / total_weight) * 100

theorem cement_percentage_final_mixture :
  final_cement_percentage 10 7 0.8 0.2 = 62 := by
  sorry

end cement_percentage_final_mixture_l587_587836


namespace incenter_coincides_l587_587010

-- Definitions for our triangles and points
variables {A B C A1 B1 C1 I : Type} [InnerProductSpace ℝ A] [InnerProductSpace ℝ B] [InnerProductSpace ℝ C]
variables (A B C : Point) (A1 B1 C1 I: Point)

-- Define the conditions: A1, B1, C1 are points on BC, CA, AB respectively 
-- and AB1 - AC1 = CA1 - CB1 = BC1 - BA1
def on_side (P : Point) (X Y : Point) : Prop := ∃ (a b : ℝ), a + b = 1 ∧ P = a • X + b • Y
def condition1 : Prop :=
  on_side A1 B C ∧ on_side B1 C A ∧ on_side C1 A B ∧
  A.distance B1 - A.distance C1 = C.distance A1 - C.distance B1 ∧
  C.distance A1 - C.distance B1 = B.distance C1 - B.distance A1

-- Circumcenters of triangles
def circumcenter (A B C : Point) : Point := sorry 

def O_A := circumcenter A B1 C1
def O_B := circumcenter A1 B C1
def O_C := circumcenter A1 B1 C

-- Definition of incenter (just a placeholder)
def incenter (A B C : Point) : Point := sorry

-- Main theorem
theorem incenter_coincides (h : condition1 A B C A1 B1 C1): 
  incenter O_A O_B O_C = incenter A B C :=
by
  sorry

end incenter_coincides_l587_587010


namespace problem1_problem2_l587_587370

variable (A : Finset α) (n m : ℕ)
variables (A_i : Fin n → Finset α)
variables [DecidableEq α]

def disjoint_pairwise (f : Fin n → Finset α) : Prop :=
  ∀ (i j : Fin n), i ≠ j → Disjoint (f i) (f j)

theorem problem1 (hdisjoint : disjoint_pairwise A_i)
    (hsubset : ∀ i, A_i i ⊆ A) (hcard : ∀ i, (A_i i).card = m):
  (∑ i in Finset.univ, 1 / (Fintype.card (Finset α) / (A_i i).card.choose)) ≤ 1 := sorry

theorem problem2 (hdisjoint : disjoint_pairwise A_i)
    (hsubset : ∀ i, A_i i ⊆ A) (hcard : ∀ i, (A_i i).card = m):
  (∑ i in Finset.univ, (Finset α).card.choose (A_i i).card) ≥ m^2 := sorry

end problem1_problem2_l587_587370


namespace min_rounds_condition_l587_587254

theorem min_rounds_condition (m : ℕ) (h : m ≥ 17) : 
  ∃ n, (n = m - 1) ∧ 
  ∀ (contestants : Fin 2m → Fin 2m → Prop), 
    (∀ i j, i ≠ j → exists_round i j (contestants_rounds m)) → -- each pair has played once over 2m-1 rounds
    (∀ (i j k l : Fin 2m), 
      ((contestants i j ∧ contestants k l) ∨ (contestants i k ∧ contestants j l) ∨ (contestants i l ∧ contestants j k)) ∧ 
      (∀ (x y : Fin 4), x ≠ y → 
        (¬contestants x y ∨ (contestants x y ∧ ∃ c₁ c₂, c₁ ≠ x ∧ c₁ ≠ y ∧ c₂ ≠ x ∧ c₂ ≠ y ∧ contestants c₁ c₂)))) := 
by admit

end min_rounds_condition_l587_587254


namespace largest_n_satisfying_conditions_l587_587939

def is_odd (n : ℤ) : Prop := n % 2 = 1

def satisfies_conditions (n : ℤ) : Prop :=
  n > 10 ∧ ∀ k : ℤ, k^2 >= 2 ∧ k^2 ≤ n / 2 → is_odd (n % k^2)

theorem largest_n_satisfying_conditions :
  ∃ n : ℤ, satisfies_conditions n ∧
    ∀ m : ℤ, satisfies_conditions m → m ≤ n := 
    ∃ n : ℤ, satisfies_conditions n ∧ n = 505 :=
begin
  sorry
end

end largest_n_satisfying_conditions_l587_587939


namespace ratio_BD_BO_l587_587755

-- Define the points and circle
variables {O A C : Type} [euclidean_geometry O A C]
variable {B : Type} -- point on the tangent
variables [tangent O A B] [tangent O C B]

-- Define the angles
variables (angle_BAC : ℝ) (h : angle_BAC = 100)

-- Define the properties of the isosceles triangle
variables (isosceles_triangle : triangle O B A)
variables (angle_OAB : ∀ O A B : Type, [tangent O A B] → 90)
variables (angle_OCB : ∀ O C B : Type, [tangent O C B] → 90)

-- Define the intersection and the distances
variables (D : point) (intersects_BO : circle O ∣ D)
variable {BO : ℝ} -- length of BO
variable {DO : ℝ} -- length of DO
variables (BD : ℝ) (h_length : BD = BO - DO)

-- The proof statement
theorem ratio_BD_BO (angle_BOC : ∀ O A C : Type, center_angle O A C = 80) :
  \(\frac{BD}{BO} = 1 - \sin(40^\circ)\) :=
by
  sorry

end ratio_BD_BO_l587_587755


namespace line_bisects_l587_587845

theorem line_bisects :
  ∀ (A B C K A₀ M : Point)
    (ω : Circle)
    (ω' : Circle),
  triangle ABC →
  incircle ABC ω →
  tangency ω BC K →
  reflection_circle ω A ω' →
  tangent_to_circle BA₀ ω' →
  tangent_to_circle CA₀ ω' →
  midpoint BC M →
  bisects_line_segment AM K A₀ :=
begin
  sorry
end

end line_bisects_l587_587845


namespace triangle_area_calculation_l587_587408

def base : ℝ := 14
def height : ℝ := 14 / 2 - 0.8 -- Height calculation in meters
def area (b h : ℝ) : ℝ := b * h / 2 -- Area calculation for the triangle

theorem triangle_area_calculation :
  area base height = 43.4 :=
by
  unfold base height area
  norm_num -- If necessary
  sorry

end triangle_area_calculation_l587_587408


namespace ratio_addition_l587_587844

theorem ratio_addition (x : ℤ) (h : 4 + x = 3 * (15 + x) / 4): x = 29 :=
by
  sorry

end ratio_addition_l587_587844


namespace directrix_parabola_l587_587413

theorem directrix_parabola (a : ℝ) (h : a ≠ 0) : 
  ∃ y : ℝ, y = - (1 / (4 * a)) :=
by
  use - (1 / (4 * a))
  sorry

end directrix_parabola_l587_587413


namespace min_distance_sum_l587_587265

theorem min_distance_sum (P : ℝ × ℝ) (hP : P.snd ^ 2 = 2 * P.fst) (D : ℝ × ℝ) (hD : D = (2, (3 / 2) * Real.sqrt 3)) :
  ∃ T : ℝ, T = (P.snd ^ 2 = 2 * P.fst → ∀ (F : ℝ × ℝ), F = (1 / 2, 0) → (dist P D + dist P (0, P.snd)) = 5 / 2) := sorry

end min_distance_sum_l587_587265


namespace tiling_problem_l587_587081

theorem tiling_problem (n : ℕ) : 
  (∃ (k : ℕ), k > 1 ∧ n = 4 * k) 
  ↔ (∃ (L_tile T_tile : ℕ), n * n = 3 * L_tile + 4 * T_tile) :=
by
  sorry

end tiling_problem_l587_587081


namespace divide_group_among_boats_l587_587066
noncomputable def number_of_ways_divide_group 
  (boatA_capacity : ℕ) 
  (boatB_capacity : ℕ) 
  (boatC_capacity : ℕ) 
  (num_adults : ℕ) 
  (num_children : ℕ) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : ℕ := 
    sorry

theorem divide_group_among_boats 
  (boatA_capacity : ℕ := 3) 
  (boatB_capacity : ℕ := 2) 
  (boatC_capacity : ℕ := 1) 
  (num_adults : ℕ := 2) 
  (num_children : ℕ := 2) 
  (constraint : ∀ {boat : ℕ}, boat > 1 → num_children ≥ 1 → num_adults ≥ 1) : 
  number_of_ways_divide_group boatA_capacity boatB_capacity boatC_capacity num_adults num_children constraint = 8 := 
sorry

end divide_group_among_boats_l587_587066


namespace f_of_4_f_of_8_inequality_solution_l587_587967

namespace MathProblem

-- Conditions
def is_increasing (f : ℝ → ℝ) : Prop := ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → x < y → f x < f y
def functional_equation (f : ℝ → ℝ) : Prop := ∀ ⦃x y : ℝ⦄, 0 < x → 0 < y → f(x * y) = f(x) + f(y)

-- Problem statement
def increasing_function := 
  ∃ f : ℝ → ℝ,
    is_increasing f ∧
    functional_equation f ∧
    f 2 = 1

-- Prove that f(4) = 2
theorem f_of_4 : increasing_function → ∀ f, is_increasing f → functional_equation f → f 2 = 1 → f 4 = 2 :=
by sorry

-- Prove that f(8) = 3
theorem f_of_8 : increasing_function → ∀ f, is_increasing f → functional_equation f → f 2 = 1 → f 4 = 2 → f 8 = 3 :=
by sorry

-- Prove the inequality solution set
theorem inequality_solution (f : ℝ → ℝ) (h_inc : is_increasing f) (h_eq : functional_equation f) (h_f2 : f 2 = 1) (h_f8 : f 8 = 3) :
  {x : ℝ | f x - f (x - 2) > 3} = {x : ℝ | 2 < x ∧ x < 16 / 7} :=
by sorry

end MathProblem

end f_of_4_f_of_8_inequality_solution_l587_587967


namespace line_tangent_to_circle_passing_through_A_l587_587961

theorem line_tangent_to_circle_passing_through_A :
  ∃ l : ℝ → ℝ, (∀ p ∈ l, p = (1,0)) ∧ (∀ C : set ℝ, ((x - 3)^2 + (y - 4)^2 = 4) → (∃ q1 q2 : set ℝ, (q1 = {1}) ∨ (q2 = {3*x - 4*y - 3 = 0})) :=
begin
  sorry
end

end line_tangent_to_circle_passing_through_A_l587_587961


namespace cos_angle_FHG_correct_l587_587666

open_locale real

noncomputable def cos_angle_FHG (A B C F G H : ℝ × ℝ) : ℝ :=
  let FH := real.sqrt ((H.1 - F.1)^2 + (H.2 - F.2)^2),
      GH := real.sqrt ((H.1 - G.1)^2 + (H.2 - G.2)^2),
      FG := real.sqrt ((G.1 - F.1)^2 + (G.2 - F.2)^2) in
  (FH^2 + GH^2 - FG^2) / (2 * FH * GH)

theorem cos_angle_FHG_correct :
  let A := (0, 6 * real.sqrt 3),
      B := (0, 0),
      C := (12, 0),
      F := (0, 4 * real.sqrt 3),
      G := (0, 2 * real.sqrt 3),
      H := (6, 3 * real.sqrt 3) in
  cos_angle_FHG A B C F G H = 33 / 39 :=
by sorry

end cos_angle_FHG_correct_l587_587666


namespace smallest_prime_divisor_sum_odd_powers_l587_587463

theorem smallest_prime_divisor_sum_odd_powers :
  (∃ p : ℕ, prime p ∧ p ∣ (3^15 + 11^21) ∧ p = 2) :=
by
  have h1 : 3^15 % 2 = 1 := by sorry
  have h2 : 11^21 % 2 = 1 := by sorry
  have h3 : (3^15 + 11^21) % 2 = 0 := by
    rw [← Nat.add_mod, h1, h2]
    exact Nat.mod_add_mod 1 1 2
  use 2
  constructor
  · exact Nat.prime_two
  · rw [Nat.dvd_iff_mod_eq_zero, h3] 
  · rfl

end smallest_prime_divisor_sum_odd_powers_l587_587463


namespace player_A_prize_received_event_A_not_low_probability_l587_587809

-- Condition Definitions
def k : ℕ := 4
def m : ℕ := 2
def n : ℕ := 1
def p : ℚ := 2 / 3
def a : ℚ := 243

-- Part 1: Player A's Prize
theorem player_A_prize_received :
  (a * (p * p + 3 * p * (1 - p) * p + 3 * (1 - p) * p * p + (1 - p) * (1 - p) * p * p)) = 216 := sorry

-- Part 2: Probability of Event A with Low Probability Conditions
def low_probability_event (prob : ℚ) : Prop := prob < 0.05

-- Probability that player B wins the entire prize
def event_A_probability (p : ℚ) : ℚ :=
  (1 - p) ^ 3 + 3 * p * (1 - p) ^ 3

theorem event_A_not_low_probability (p : ℚ) (hp : p ≥ 3 / 4) :
  ¬ low_probability_event (event_A_probability p) := sorry

end player_A_prize_received_event_A_not_low_probability_l587_587809


namespace arithmetic_sequence_general_term_sum_of_b_seq_l587_587603

noncomputable def a_seq (n : ℕ) := 3 ^ (n - 1)

def b_seq (n : ℕ) := (a_seq n) * (Real.log 2 (a_seq n))

def sum_seq (f : ℕ → ℝ) (n : ℕ) : ℝ := ∑ i in Finset.range n, f (i + 1)

theorem arithmetic_sequence_general_term :
  -- Given conditions
  ∀ (S : ℕ → ℝ) (q : ℝ), q ≠ 1 → 
  (a_seq 1 + a_seq 3 = S 4 / S 2) → 
  ((a_seq 1 - 1), (a_seq 2 - 1), (a_seq 3 - 1) form an arithmetic sequence) →
  -- Prove general term
  (∀ n : ℕ, a_seq n = 3 ^ (n - 1)) :=
sorry

theorem sum_of_b_seq :
  -- Given conditions
  ∀ (n : ℕ), n > 0 → 
  -- Prove sum of b_seq
  (sum_seq b_seq n = (2 * n - 3) * 3 ^ n / 4 + 3 / 4) :=
sorry

end arithmetic_sequence_general_term_sum_of_b_seq_l587_587603


namespace sum_of_numbers_l587_587090

theorem sum_of_numbers : 1324 + 2431 + 3142 + 4213 + 1234 = 12344 := sorry

end sum_of_numbers_l587_587090


namespace remainder_div_x_minus_2_l587_587832

noncomputable def q (x : ℝ) (A B C : ℝ) : ℝ := A * x^6 + B * x^4 + C * x^2 + 10

theorem remainder_div_x_minus_2 (A B C : ℝ) (h : q 2 A B C = 20) : q (-2) A B C = 20 :=
by sorry

end remainder_div_x_minus_2_l587_587832


namespace five_c_plus_seven_d_l587_587061

def is_two_digit_prime (n : ℕ) : Prop := 
  nat.prime n ∧ n ≥ 10 ∧ n < 100

theorem five_c_plus_seven_d :
  let primes := [11, 13, 17, 19]
  (∀ x ∈ primes, is_two_digit_prime x) →
  ∃ c d : ℕ, c < d ∧
  (c = 32 ∧ d = 36) ∧
  (let sum1 := 5 * c + 7 * d
  in sum1 = 412) :=
by
  sorry

end five_c_plus_seven_d_l587_587061


namespace tournament_least_difference_l587_587330

theorem tournament_least_difference
  (n m p : Nat)
  (h : 2 * n + 5 * m + 7 * p = 200) :
  ∃ t : Nat, t = |n + m - p| ∧ t = 26 :=
by
  sorry

end tournament_least_difference_l587_587330


namespace monotonic_and_minimum_value_of_f_l587_587280

def f (x a : ℝ) : ℝ := x^2 + 2*a*x + 2

theorem monotonic_and_minimum_value_of_f:
  (∀ x, x ∈ Icc (-5 : ℝ) (5 : ℝ) →
    ( ∃ a, (a ≥ 5 ∨ a ≤ -5) → 
      (∀ x ∈ Icc (-5 : ℝ) (5 : ℝ), 
        f x a ≥ f (-5) a ∨ f x a ≥ f (-a) a ∨ f x a ≥ f (5) a )) )
  ∧
  ( ∃ a,
    min (f (-5) a) (min (f (-a) a) (f (5) a)) = 
      if a ≥ 5 then 27 - 10 * a
      else if -5 ≤ a ∧ a < 5 then 2 - a^2
      else 27 + 10 * a ) := sorry

end monotonic_and_minimum_value_of_f_l587_587280


namespace lattice_points_color_count_eq_l587_587250

noncomputable def lattice_points_coloring_ways (n : ℕ) (h : n ≥ 2) : ℕ :=
  3^n * (2^(n+1) - 1)^(n-1)

-- Theorem stating the number of lattice points coloring ways given the conditions
theorem lattice_points_color_count_eq :
  ∀ n : ℕ, n ≥ 2 →
  lattice_points_coloring_ways n (by assumption) = 3^n * (2^(n+1) - 1)^(n-1) :=
  by sorry

end lattice_points_color_count_eq_l587_587250


namespace exist_sequences_l587_587188

def sequence_a (a : ℕ → ℤ) : Prop :=
  a 0 = 4 ∧ a 1 = 22 ∧ ∀ n ≥ 2, a n = 6 * a (n - 1) - a (n - 2)

theorem exist_sequences (a : ℕ → ℤ) (x y : ℕ → ℤ) :
  sequence_a a → (∀ n, 0 < x n ∧ 0 < y n) →
  (∀ n, a n = (y n ^ 2 + 7) / (x n - y n)) :=
by
  intro h_seq_a h_pos
  sorry

end exist_sequences_l587_587188


namespace radius_of_intersection_l587_587522

noncomputable def sphere_radius := 2 * Real.sqrt 17

theorem radius_of_intersection (s : ℝ) 
  (h1 : (3:ℝ)=(3:ℝ)) (h2 : (5:ℝ)=(5:ℝ)) (h3 : (0-3:ℝ)^2 + (5-5:ℝ)^2 + (s-(-8+8))^2 = sphere_radius^2) :
  s = Real.sqrt 59 :=
by
  sorry

end radius_of_intersection_l587_587522


namespace system_has_two_distinct_solutions_for_valid_a_l587_587569

noncomputable def log_eq (x y a : ℝ) : Prop := 
  Real.log (a * x + 4 * a) / Real.log (abs (x + 3)) = 
  2 * Real.log (x + y) / Real.log (abs (x + 3))

noncomputable def original_system (x y a : ℝ) : Prop :=
  log_eq x y a ∧ (x + 1 + Real.sqrt (x^2 + 2 * x + y - 4) = 0)

noncomputable def valid_range (a : ℝ) : Prop := 
  (4 < a ∧ a < 4.5) ∨ (4.5 < a ∧ a ≤ 16 / 3)

theorem system_has_two_distinct_solutions_for_valid_a (a : ℝ) :
  valid_range a → 
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ original_system x₁ 5 a ∧ original_system x₂ 5 a ∧ (-5 < x₁ ∧ x₁ ≤ -1) ∧ (-5 < x₂ ∧ x₂ ≤ -1) := 
sorry

end system_has_two_distinct_solutions_for_valid_a_l587_587569


namespace average_of_sample_data_l587_587288

variables (x : Fin 10 → ℝ) (a b : ℝ)

theorem average_of_sample_data (h1 : (x 0 + x 1 + x 2) / 3 = a)
                              (h2 : (x 3 + x 4 + x 5 + x 6 + x 7 + x 8 + x 9) / 7 = b) :
  (Finset.univ.sum (λ i, x i) / 10) = (3 * a + 7 * b) / 10 :=
by
  sorry

end average_of_sample_data_l587_587288


namespace scarves_per_box_l587_587758

theorem scarves_per_box (boxes mittens_per_box total_clothing : ℕ) (h1 : boxes = 7) (h2 : mittens_per_box = 4) (h3 : total_clothing = 49) : 
  let total_mittens := boxes * mittens_per_box in
  let total_scarves := total_clothing - total_mittens in
  total_scarves / boxes = 3 :=
by
  sorry

end scarves_per_box_l587_587758


namespace pentagon_area_50_l587_587394

def point := (ℝ × ℝ)

structure Pentagon :=
(A B C D E : point)

def area_rectangle (p1 p2 p3 p4 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
let ⟨x4, y4⟩ := p4 in
abs((x3 - x1) * (y2 - y1))

def area_triangle (p1 p2 p3 : point) : ℝ :=
let ⟨x1, y1⟩ := p1 in
let ⟨x2, y2⟩ := p2 in
let ⟨x3, y3⟩ := p3 in
abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2)

def y_coordinate_C (pent: Pentagon) : ℝ :=
let ⟨_, _, pC, _, _⟩ := pent in
pC.2

theorem pentagon_area_50 (h : ℝ) :
  let A := (0, 0) in
  let B := (0, 5) in
  let C := (3, h) in
  let D := (6, 5) in
  let E := (6, 0) in
  let rect_area := area_rectangle A B D E in
  let tri_area := area_triangle B C D in
  rect_area + tri_area = 50 :=
by
  sorry

end pentagon_area_50_l587_587394


namespace original_number_equals_3408_l587_587649

theorem original_number_equals_3408 :
  let x := 213 * 16 in x = 3408 :=
by
  sorry

end original_number_equals_3408_l587_587649


namespace average_rate_640_miles_trip_l587_587734

theorem average_rate_640_miles_trip 
  (total_distance : ℕ) 
  (first_half_distance : ℕ) 
  (first_half_rate : ℕ) 
  (second_half_time_multiplier : ℕ) 
  (first_half_time : ℕ := first_half_distance / first_half_rate)
  (second_half_time : ℕ := second_half_time_multiplier * first_half_time)
  (total_time : ℕ := first_half_time + second_half_time)
  (average_rate : ℕ := total_distance / total_time) : 
  total_distance = 640 ∧ 
  first_half_distance = 320 ∧ 
  first_half_rate = 80 ∧ 
  second_half_time_multiplier = 3 → 
  average_rate = 40 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  rw [h1, h2, h3, h4] at *
  have h5 : first_half_time = 320 / 80 := rfl
  have h6 : second_half_time = 3 * (320 / 80) := rfl
  have h7 : total_time = (320 / 80) + 3 * (320 / 80) := rfl
  have h8 : average_rate = 640 / (4 + 12) := rfl
  have h9 : average_rate = 640 / 16 := rfl
  have average_rate_correct : average_rate = 40 := rfl
  exact average_rate_correct

end average_rate_640_miles_trip_l587_587734


namespace total_matches_won_l587_587327

-- Define the conditions
def matches_in_first_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat
def matches_in_second_period (total: ℕ) (win_rate: ℚ) : ℕ := (total * win_rate).toNat

-- The main proof statement that we need to prove
theorem total_matches_won (total1 total2 : ℕ) (win_rate1 win_rate2 : ℚ) :
  matches_in_first_period total1 win_rate1 + matches_in_second_period total2 win_rate2 = 110 :=
by
  sorry

end total_matches_won_l587_587327


namespace value_range_of_x_l587_587270

noncomputable def f (x : ℝ) : ℝ := 4^x - 3 * 2^x + 3

theorem value_range_of_x 
  (h : set.range f = set.Icc (1 : ℝ) (7 : ℝ)) : 
  ∀ x : ℝ, x ∈ set.Iic 0 ∪ set.Icc 1 2 :=
sorry

end value_range_of_x_l587_587270


namespace sum_primitive_roots_mod_11_l587_587472

def is_primitive_root (a n : ℕ) : Prop :=
  ∀ b < n, ∃ k < n, a^k % n = b

theorem sum_primitive_roots_mod_11 :
  let s := {1, 2, 3, 4, 5, 6, 7, 8}
  s.sum (λ x, if is_primitive_root x 11 then x else 0) = 23 :=
by
  sorry

end sum_primitive_roots_mod_11_l587_587472


namespace dave_more_than_derek_l587_587911

def derek_initial : ℕ := 40
def derek_spent_on_self1 : ℕ := 14
def derek_spent_on_dad : ℕ := 11
def derek_spent_on_self2 : ℕ := 5

def dave_initial : ℕ := 50
def dave_spent_on_mom : ℕ := 7

def derek_remaining : ℕ := derek_initial - (derek_spent_on_self1 + derek_spent_on_dad + derek_spent_on_self2)
def dave_remaining : ℕ := dave_initial - dave_spent_on_mom

theorem dave_more_than_derek : dave_remaining - derek_remaining = 33 :=
by
  -- The proof goes here
  sorry

end dave_more_than_derek_l587_587911


namespace prob_one_boy_one_girl_l587_587171

-- Defining the probabilities of birth
def prob_boy := 2 / 3
def prob_girl := 1 / 3

-- Calculating the probability of all boys
def prob_all_boys := prob_boy ^ 4

-- Calculating the probability of all girls
def prob_all_girls := prob_girl ^ 4

-- Calculating the probability of having at least one boy and one girl
def prob_at_least_one_boy_and_one_girl := 1 - (prob_all_boys + prob_all_girls)

-- Proof statement
theorem prob_one_boy_one_girl : prob_at_least_one_boy_and_one_girl = 64 / 81 :=
by sorry

end prob_one_boy_one_girl_l587_587171


namespace Faye_age_correct_l587_587197

def ages (C D E F G : ℕ) : Prop :=
  D = E - 2 ∧
  C = E + 3 ∧
  F = C - 1 ∧
  D = 16 ∧
  G = D - 5

theorem Faye_age_correct (C D E F G : ℕ) (h : ages C D E F G) : F = 20 :=
by {
  sorry
}

end Faye_age_correct_l587_587197


namespace sam_won_total_matches_l587_587323

/-- Sam's first 100 matches and he won 50% of them -/
def first_100_matches : ℕ := 100

/-- Sam won 50% of his first 100 matches -/
def win_rate_first : ℕ := 50

/-- Sam's next 100 matches and he won 60% of them -/
def next_100_matches : ℕ := 100

/-- Sam won 60% of his next 100 matches -/
def win_rate_next : ℕ := 60

/-- The total number of matches Sam won -/
def total_matches_won (first_100_matches: ℕ) (win_rate_first: ℕ) (next_100_matches: ℕ) (win_rate_next: ℕ) : ℕ :=
  (first_100_matches * win_rate_first) / 100 + (next_100_matches * win_rate_next) / 100

theorem sam_won_total_matches :
  total_matches_won first_100_matches win_rate_first next_100_matches win_rate_next = 110 :=
by
  sorry

end sam_won_total_matches_l587_587323


namespace calculation_identity_l587_587178

theorem calculation_identity :
  (3.14 - 1)^0 * (-1 / 4)^(-2) = 16 := by
  sorry

end calculation_identity_l587_587178


namespace pentagon_area_l587_587392

/-- This Lean statement represents the problem of finding the y-coordinate of vertex C
    in a pentagon with given vertex positions and specific area constraint. -/
theorem pentagon_area (y : ℝ) 
  (h_sym : true) -- The pentagon ABCDE has a vertical line of symmetry
  (h_A : (0, 0) = (0, 0)) -- A(0,0)
  (h_B : (0, 5) = (0, 5)) -- B(0, 5)
  (h_C : (3, y) = (3, y)) -- C(3, y)
  (h_D : (6, 5) = (6, 5)) -- D(6, 5)
  (h_E : (6, 0) = (6, 0)) -- E(6, 0)
  (h_area : 50 = 50) -- The total area of the pentagon is 50 square units
  : y = 35 / 3 :=
sorry

end pentagon_area_l587_587392


namespace find_g_neg1_l587_587629

noncomputable def f (g : ℝ → ℝ) : ℝ → ℝ :=
λ x, if x ≥ 0 then x^2 + 2*x else g x

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f x

theorem find_g_neg1 (g : ℝ → ℝ) (h_odd : is_odd (f g)) : g (-1) = -3 :=
by
  sorry

end find_g_neg1_l587_587629


namespace find_x2_plus_y2_l587_587300

theorem find_x2_plus_y2 (x y : ℝ) (h1 : x * y = 12) (h2 : x^2 * y + x * y^2 + x + y = 99) : 
  x^2 + y^2 = 5745 / 169 := 
sorry

end find_x2_plus_y2_l587_587300


namespace trays_needed_to_fill_ice_cubes_l587_587201

-- Define the initial conditions
def ice_cubes_in_glass : Nat := 8
def multiplier_for_pitcher : Nat := 2
def spaces_per_tray : Nat := 12

-- Define the total ice cubes used
def total_ice_cubes_used : Nat := ice_cubes_in_glass + multiplier_for_pitcher * ice_cubes_in_glass

-- State the Lean theorem to be proven: The number of trays needed
theorem trays_needed_to_fill_ice_cubes : 
  total_ice_cubes_used / spaces_per_tray = 2 :=
  by 
  sorry

end trays_needed_to_fill_ice_cubes_l587_587201


namespace val_4_at_6_l587_587310

def at_op (a b : ℤ) : ℤ := 2 * a - 4 * b

theorem val_4_at_6 : at_op 4 6 = -16 := by
  sorry

end val_4_at_6_l587_587310


namespace ratio_BG_GE_l587_587973

namespace GeometryProblem

-- Definitions of the points and conditions given in the problem
constant A B C D E F G : Type
constant circumcircle_ABC : set A
constant circumcircle_ABD : set A

-- Point D is the intersection of tangents from A and B to the circumcircle of triangle ABC
axiom D_def : D ∈ circumcircle_ABC

-- Circumcircle of triangle ABD intersects AC at E and BC at F
axiom E_def : E ∈ circumcircle_ABD ∧ E ∈ line AC
axiom F_def : F ∈ circumcircle_ABD ∧ F ∈ segment BC

-- CD and BE intersect at point G
axiom G_def : G ∈ line (CD) ∧ G ∈ line (BE)

-- Given ratio: BC / BF = 2
axiom ratio_BC_BF : BC / BF = 2

-- Prove that BG / GE = 2
theorem ratio_BG_GE : BG / GE = 2 := sorry

end GeometryProblem

end ratio_BG_GE_l587_587973


namespace volume_of_tetrahedron_equals_450_sqrt_2_l587_587923

-- Given conditions
variables {A B C D : Point}
variables (areaABC areaBCD : ℝ) (BC : ℝ) (angleABC_BCD : ℝ)

-- The specific values for the conditions
axiom h_areaABC : areaABC = 150
axiom h_areaBCD : areaBCD = 90
axiom h_BC : BC = 10
axiom h_angleABC_BCD : angleABC_BCD = π / 4  -- 45 degrees in radians

-- Definition of the volume to be proven
def volume_tetrahedron (A B C D : Point) : ℝ :=
  (1 / 3) * areaABC * (18 * real.sin angleABC_BCD)

-- Final proof statement
theorem volume_of_tetrahedron_equals_450_sqrt_2 :
  volume_tetrahedron A B C D = 450 * real.sqrt 2 :=
by 
  -- Preliminary setup, add the relevant properties and results
  sorry

end volume_of_tetrahedron_equals_450_sqrt_2_l587_587923


namespace sum_of_roots_eq_9_div_4_l587_587555

-- Define the values for the coefficients
def a : ℝ := -48
def b : ℝ := 108
def c : ℝ := -27

-- Define the quadratic equation and the function that represents the sum of the roots
def quadratic_eq (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Statement of the problem: Prove the sum of the roots of the quadratic equation equals 9/4
theorem sum_of_roots_eq_9_div_4 : 
  (∀ x y : ℝ, quadratic_eq x = 0 → quadratic_eq y = 0 → x ≠ y → x + y = - (b/a)) → - (b / a) = 9 / 4 :=
by
  sorry

end sum_of_roots_eq_9_div_4_l587_587555


namespace ratio_of_profits_l587_587426

-- Define the conditions and main statement
theorem ratio_of_profits (x : ℕ) (P_investment_ratio Q_investment_ratio : ℕ)
  (P_time_period Q_time_period : ℕ) 
  (h1 : P_investment_ratio = 7)
  (h2 : Q_investment_ratio = 5)
  (h3 : P_time_period = 5)
  (h4 : Q_time_period = 9) : 
  (7 * P_time_period) / (5 * Q_time_period) = 7 / 9 :=
by
  have hP := calc
    (P_investment_ratio * P_time_period) = (7 * 5) : by rw [h1, h3]
  have hQ := calc
    (Q_investment_ratio * Q_time_period) = (5 * 9) : by rw [h2, h4]
  have h_ratio := calc
    (7 * 5) / (5 * 9) = 35 / 45 : by rw [hP, hQ]
  have h_simplify := calc
    35 / 45 = 7 / 9 : by norm_num
  exact h_simplify

end ratio_of_profits_l587_587426


namespace fraction_of_products_inspected_jane_l587_587488

theorem fraction_of_products_inspected_jane 
  (P : ℝ) 
  (J : ℝ) 
  (John_rejection_rate : ℝ) 
  (Jane_rejection_rate : ℝ)
  (Total_rejection_rate : ℝ) 
  (hJohn : John_rejection_rate = 0.005) 
  (hJane : Jane_rejection_rate = 0.008) 
  (hTotal : Total_rejection_rate = 0.0075) 
  : J = 5 / 6 := by
{
  sorry
}

end fraction_of_products_inspected_jane_l587_587488


namespace proof_a_eq_b_pow_n_l587_587606

theorem proof_a_eq_b_pow_n
  (a b n : ℕ)
  (h : ∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) :
  a = b^n := 
by sorry

end proof_a_eq_b_pow_n_l587_587606


namespace least_k_inequality_l587_587554

theorem least_k_inequality :
  ∃ k : ℝ, (∀ a b c : ℝ, 
    ((2 * a / (a - b)) ^ 2 + (2 * b / (b - c)) ^ 2 + (2 * c / (c - a)) ^ 2 + k 
    ≥ 4 * (2 * a / (a - b) + 2 * b / (b - c) + 2 * c / (c - a)))) ∧ k = 8 :=
by
  sorry  -- proof is omitted

end least_k_inequality_l587_587554


namespace max_min_distance_on_ellipse_to_line_l587_587673

noncomputable def ellipse_parametric (theta : ℝ) : ℝ × ℝ :=
(√3 * Real.cos theta, Real.sin theta)

noncomputable def line_polar (theta : ℝ) (rho : ℝ) : Prop :=
2 * rho * Real.cos (theta + π / 3) = 3 * √6

noncomputable def distance (p : ℝ × ℝ) (a b c : ℝ) : ℝ :=
(abs (a * p.1 + b * p.2 + c)) / sqrt (a^2 + b^2)

theorem max_min_distance_on_ellipse_to_line :
  let d := distance in
  (∀ theta, d (ellipse_parametric theta) 1 (-√3) (-3 * √6) ≤ 2 * √6)
  ∧ (∀ theta, d (ellipse_parametric theta) 1 (-√3) (-3 * √6) ≥ √6) :=
sorry

end max_min_distance_on_ellipse_to_line_l587_587673


namespace compare_expressions_l587_587589

theorem compare_expressions (a b : ℝ) : (a + 3) * (a - 5) < (a + 2) * (a - 4) :=
by {
  sorry
}

end compare_expressions_l587_587589


namespace intersection_correct_l587_587293

def M : set ℕ := {1, 2, 3}
def N : set ℕ := {2, 3, 4}

theorem intersection_correct : M ∩ N = {2, 3} :=
sorry

end intersection_correct_l587_587293


namespace smallest_prime_divisor_of_sum_l587_587467

theorem smallest_prime_divisor_of_sum (a b : ℕ) 
  (h₁ : a = 3 ^ 15) 
  (h₂ : b = 11 ^ 21) 
  (h₃ : odd a) 
  (h₄ : odd b) : 
  nat.prime_divisors (a + b) = [2] := 
by
  sorry

end smallest_prime_divisor_of_sum_l587_587467


namespace maximize_volume_l587_587148

open Real

noncomputable def volume (x : ℝ) := x * (6 - 2 * x)^2

theorem maximize_volume : ∃ x ∈ Ioo (0 : ℝ) (3 : ℝ), (∀ y ∈ Ioo (0 : ℝ) (3 : ℝ), volume y ≤ volume x) ∧ x = 1 :=
by
  sorry

end maximize_volume_l587_587148


namespace find_constant_l587_587417

noncomputable def expr (x C : ℝ) : ℝ :=
  (x - 1) * (x - 3) * (x - 4) * (x - 6) + C

theorem find_constant :
  (∀ x : ℝ, expr x (-0.5625) ≥ 1) → expr 3.5 (-0.5625) = 1 :=
by
  sorry

end find_constant_l587_587417


namespace parallel_lines_m_eq_neg4_l587_587620

theorem parallel_lines_m_eq_neg4 (m : ℝ) (h1 : (m-2) ≠ -m) 
  (h2 : (m-2) / 3 = -m / (m + 2)) : m = -4 :=
sorry

end parallel_lines_m_eq_neg4_l587_587620


namespace option_d_same_function_l587_587093

open Real

def f (x: ℝ) : ℝ := (x - 1)^0
def g (x: ℝ) : ℝ := 1 / (x - 1)^0

theorem option_d_same_function : ∀ x: ℝ, x ≠ 1 → f x = g x := by
  intro x hx
  -- specific steps to show f(x) = g(x) are omitted here
  sorry

end option_d_same_function_l587_587093


namespace value_of_3_pow_m_minus_n_value_of_9_pow_m_mul_27_pow_n_l587_587953

variable (m n : ℝ)
variable (h1 : 3^m = 2)
variable (h2 : 3^n = 5)

theorem value_of_3_pow_m_minus_n : 3^(m - n) = 2 / 5 := 
by sorry

theorem value_of_9_pow_m_mul_27_pow_n : 9^m * 27^n = 500 := 
by sorry

end value_of_3_pow_m_minus_n_value_of_9_pow_m_mul_27_pow_n_l587_587953


namespace broth_for_third_l587_587878

theorem broth_for_third (b : ℚ) (h : b = 6 + 3/4) : b / 3 = 2 + 1/4 := by
  sorry

end broth_for_third_l587_587878


namespace correct_propositions_l587_587438

theorem correct_propositions :
  (¬ ∃ x : ℝ, x < 0 ∧ ∀ y : ℝ, y < 0 → y ≤ x) ∧
  (¬ ∃ x : ℤ, ∀ y : ℤ, y ≥ x) ∧
  (∃ x : ℤ, x = -1 ∧ ∀ y : ℤ, y < 0 → y ≤ x) ∧
  (∃ x : ℤ, x = 1 ∧ ∀ y : ℤ, y > 0 → y ≥ x) :=
by {
  -- This is the theorem statement as requested,
  -- proofs for the individual statements have been omitted.
  sorry
}

end correct_propositions_l587_587438


namespace vector_parallel_and_on_line_l587_587903

noncomputable def is_point_on_line (x y t : ℝ) : Prop :=
  x = 5 * t + 3 ∧ y = 2 * t + 4

noncomputable def is_parallel (a b c d : ℝ) : Prop :=
  ∃ k : ℝ, a = k * c ∧ b = k * d

theorem vector_parallel_and_on_line :
  ∃ (a b t : ℝ), 
      (a = (5 * t + 3) - 1) ∧ (b = (2 * t + 4) - 1) ∧ 
      is_parallel a b 3 2 ∧ is_point_on_line (5 * t + 3) (2 * t + 4) t := 
by
  use (33 / 4), (11 / 2), (5 / 4)
  sorry

end vector_parallel_and_on_line_l587_587903


namespace stock_price_end_of_third_year_l587_587198

def first_year_price (initial_price : ℝ) (first_year_increase : ℝ) : ℝ :=
  initial_price + (initial_price * first_year_increase)

def second_year_price (price_end_first : ℝ) (second_year_decrease : ℝ) : ℝ :=
  price_end_first - (price_end_first * second_year_decrease)

def third_year_price (price_end_second : ℝ) (third_year_increase : ℝ) : ℝ :=
  price_end_second + (price_end_second * third_year_increase)

theorem stock_price_end_of_third_year :
  ∀ (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) (third_year_increase : ℝ),
    initial_price = 150 →
    first_year_increase = 0.5 →
    second_year_decrease = 0.3 →
    third_year_increase = 0.2 →
    third_year_price (second_year_price (first_year_price initial_price first_year_increase) second_year_decrease) third_year_increase = 189 :=
by
  intros initial_price first_year_increase second_year_decrease third_year_increase
  sorry

end stock_price_end_of_third_year_l587_587198


namespace problem_a_problem_b_l587_587838

-- a) Petya and Vasya each thought of three natural numbers. Petya wrote on the board
-- the greatest common divisor (GCD) of every pair of his numbers.
-- Vasya wrote on the board the least common multiple (LCM) of every pair of his numbers.
-- It turned out that Petya and Vasya wrote the same numbers on the board
-- (possibly in a different order). Prove that all the numbers written on the board are equal.
theorem problem_a (a b c : ℕ) (h : multiset.map (λ (x : ℕ × ℕ), gcd x.fst x.snd) [(a, b), (b, c), (a, c)] = multiset.map (λ (x : ℕ × ℕ), lcm x.fst x.snd) [(a, b), (b, c), (a, c)]) :
  a = b ∧ b = c :=
sorry

-- b) Will the statement from the previous problem remain true if Petya and Vasya initially thought of four natural numbers each?
theorem problem_b (a b c d : ℕ) (h : multiset.map (λ (x : ℕ × ℕ), gcd x.fst x.snd) [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)] = multiset.map (λ (x : ℕ × ℕ), lcm x.fst x.snd) [(a, b), (a, c), (a, d), (b, c), (b, d), (c, d)]) :
  False :=
begin
  -- Counterexample: Let a = 6, b = 10, c = 15, d = 30.
  let a := 6,
  let b := 10,
  let c := 15,
  let d := 30,
  
  -- Verify gcd pairs
  have gcd_pairs := [gcd 6 10, gcd 6 15, gcd 6 30, gcd 10 15, gcd 10 30, gcd 15 30],
  have gcd_pairs_values : gcd_pairs = [2, 3, 6, 5, 10, 15], by sorry,

  -- Verify lcm pairs for another set, e.g. 1, 2, 3, 5
  let e := 1,
  let f := 2,
  let g := 3,
  let h := 5,
  have lcm_pairs := [lcm 1 2, lcm 1 3, lcm 1 5, lcm 2 3, lcm 2 5, lcm 3 5],
  have lcm_pairs_values : lcm_pairs = [2, 3, 5, 6, 10, 15], by sorry,
  
  -- Since both pairs are the same, contradiction is reached.
  exact sorry,
end

end problem_a_problem_b_l587_587838


namespace count_valid_functions_l587_587573

theorem count_valid_functions : 
  {f : ℝ → ℝ // ∃ a b c : ℝ, (∀ x, f x = a * x^2 + b * x + c) ∧ (∀ x, f x * f (-x) = f (x^2)) }.card = 8 := 
sorry

end count_valid_functions_l587_587573


namespace function_C_is_even_l587_587479

theorem function_C_is_even : ∀ x : ℝ, 2 * (-x)^2 - 1 = 2 * x^2 - 1 :=
by
  intro x
  sorry

end function_C_is_even_l587_587479


namespace julia_total_food_cost_l587_587694

-- Definitions based on conditions
def weekly_total_cost : ℕ := 30
def rabbit_weeks : ℕ := 5
def rabbit_food_cost : ℕ := 12
def parrot_weeks : ℕ := 3
def parrot_food_cost : ℕ := weekly_total_cost - rabbit_food_cost

-- Proof statement
theorem julia_total_food_cost : 
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost = 114 := 
by 
  sorry

end julia_total_food_cost_l587_587694


namespace problem_statement_l587_587052

-- Define aₙ as given
def a (n : ℕ) : ℕ := 4 * n - 1

-- Define Sₙ as the sum of the first n terms of the sequence {aₙ}
def S (n : ℕ) : ℕ := n * (3 + 2 * n - 2)

-- Define bₙ as Sₙ / n
def b (n : ℕ) : ℕ := S n / n

-- Define the sum Tₙ of the first n terms of the sequence {2ⁿ * bₙ}
def T (n : ℕ) : ℕ := ∑ k in finset.range n, (2 ^ (k + 1)) * b (k + 1)

-- Theorem to prove
theorem problem_statement (n : ℕ) : T n = 2^(n+2) * n - 2^(n+2) + 2 := 
by
  sorry

end problem_statement_l587_587052


namespace line_tangent_to_parabola_l587_587053

theorem line_tangent_to_parabola (c : ℝ) : 
  (∀ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x → 16 - 16 * c = 0) → c = 1 :=
by
  intros h
  sorry

end line_tangent_to_parabola_l587_587053


namespace train_seat_count_l587_587435

theorem train_seat_count (x : ℕ) (h1 : ∀ n, n = 4) (h2 : ∀ y, y = 10) (h3 : ∀ z, z = 3) (h4 : ∀ w, w = 420) :
  4 * 3 * (x + 10) = 420 → x = 25 := 
by {
  intro h,
  rw mul_assoc at h,
  simp [h1, h2, h3, h4] at h,
  linarith,
  sorry
}

end train_seat_count_l587_587435


namespace new_belt_time_eq_15_l587_587528

/-- The work rate of the old conveyor belt, which takes 21 hours to move one day's coal output. -/
def old_belt_rate : ℝ := 1 / 21

/-- The combined work rate of both conveyors, moving one day's coal output in 8.75 hours. -/
def combined_rate : ℝ := 1 / 8.75

/-- The work rate of the new conveyor belt, which takes x hours to move one day's coal output. -/
def new_belt_rate (x : ℝ) : ℝ := 1 / x

/-- Given the work rates of the old belt and the combined rate, determine the
    time it would take the new conveyor belt to move one day's coal output. -/
theorem new_belt_time_eq_15 : ∃ x : ℝ, old_belt_rate + new_belt_rate x = combined_rate ∧ x = 15 := 
by 
  sorry

end new_belt_time_eq_15_l587_587528


namespace inequality_solution_set_range_of_a_l587_587378

theorem inequality_solution_set (x : ℝ) :
  (|x - 1| + |x + 1| ≥ 3) ↔ (x ≤ -3 / 2 ∨ x ≥ 3 / 2) :=
begin
  sorry
end

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, |x - 1| + |x - a| ≥ 3) ↔ (a ≤ -2 ∨ a ≥ 4) :=
begin
  sorry
end

end inequality_solution_set_range_of_a_l587_587378


namespace larger_number_is_42_l587_587843

theorem larger_number_is_42 (x y : ℕ) (h1 : x + y = 77) (h2 : 5 * x = 6 * y) : x = 42 :=
by
  sorry

end larger_number_is_42_l587_587843


namespace function_periodic_l587_587237

variable {X : Type*} -- Domain Type
variable (f : X → ℝ) -- The function f from the domain X to real numbers
variable (a : ℝ) -- Fixed value a

noncomputable def periodic_at_4a (f : X → ℝ) (a : ℝ) : Prop :=
  ∀ x : X, f(x + 4 * a) = f(x)

theorem function_periodic (f : X → ℝ) (a : ℝ)
  (h : ∀ x : X, f(x + a) = (1 + f(x)) / (1 - f(x))) :
  periodic_at_4a f a := 
by
  sorry

end function_periodic_l587_587237


namespace find_n_modulo_l587_587938

theorem find_n_modulo :
  ∃ (n : ℕ), 0 <= n ∧ n <= 9 ∧ n ≡ -2839 [MOD 10] ∧ n = 1 :=
by
  sorry

end find_n_modulo_l587_587938


namespace distinct_pairs_count_l587_587983

theorem distinct_pairs_count : 
  (∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, ∃ (a b : ℕ), 1 ≤ a ∧ 1 ≤ b ∧ a + b = 40 ∧ p = (a, b)) ∧ s.card = 39) := sorry

end distinct_pairs_count_l587_587983


namespace distance_BC_l587_587520

variable (AC AB : ℝ) (angleACB : ℝ)
  (hAC : AC = 2)
  (hAB : AB = 3)
  (hAngle : angleACB = 120)

theorem distance_BC (BC : ℝ) : BC = Real.sqrt 6 - 1 :=
by
  sorry

end distance_BC_l587_587520


namespace average_speed_is_40_l587_587746

-- Define the total distance
def total_distance : ℝ := 640

-- Define the distance for the first half
def first_half_distance : ℝ := total_distance / 2

-- Define the average speed for the first half
def first_half_speed : ℝ := 80

-- Define the time taken for the first half
def first_half_time : ℝ := first_half_distance / first_half_speed

-- Define the multiplicative factor for time increase in the second half
def time_increase_factor : ℝ := 3

-- Define the time taken for the second half
def second_half_time : ℝ := first_half_time * time_increase_factor

-- Define the total time for the trip
def total_time : ℝ := first_half_time + second_half_time

-- Define the calculated average speed for the entire trip
def calculated_average_speed : ℝ := total_distance / total_time

-- State the theorem that the average speed for the entire trip is 40 miles per hour
theorem average_speed_is_40 : calculated_average_speed = 40 :=
by
  sorry

end average_speed_is_40_l587_587746


namespace median_number_of_children_l587_587782

noncomputable def median_value_in_class : ℕ → ℕ
| 15 := 3
| _ := 0

theorem median_number_of_children (n : ℕ) (h : n = 15) :
  median_value_in_class n = 3 :=
by {
  rw h,
  exact rfl,
}

end median_number_of_children_l587_587782


namespace ellipse_properties_l587_587249

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 3) = 1

-- Define points and conditions
def F := (1 : ℝ, 0 : ℝ)
def A := (2 : ℝ, 0 : ℝ)
def O := (0 : ℝ, 0 : ℝ)
def H := (0 : ℝ, 4 / 3 : ℝ)

-- Define the line l passing through point A
def line_l (x y : ℝ) (k : ℝ) : Prop := y = k * (x - 2)

-- Define the condition related to the focus and eccentricity
def condition1 (FA OF OA : ℝ) (e : ℝ) : Prop := (FA / OF) + (FA / OA) = 3 * e

-- Define the slopes to be checked
def slopes (k : ℝ) : Prop := k = -9 / 2 ∨ k = 1 / 2

-- Prove that the given slopes satisfy the conditions
theorem ellipse_properties : 
  (∃ k : ℝ, slopes k ∧ (∀ x y : ℝ, ellipse x y → line_l x y k)) → 
  (∃ FA OF OA e : ℝ, condition1 FA OF OA e) := 
by 
  sorry

end ellipse_properties_l587_587249


namespace bn_geometric_l587_587597

variables {a : ℕ → ℝ} {q : ℝ}

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n+1) = q * a n

def sum_of_three_consecutive (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  a (3*n - 2) + a (3*n - 1) + a (3*n)

theorem bn_geometric (hq : q ≠ 1) (ha : geometric_sequence a q):
  let b : ℕ → ℝ := fun n => sum_of_three_consecutive a n in
  geometric_sequence b (q^3) :=
sorry

end bn_geometric_l587_587597


namespace minimize_sum_of_squares_of_roots_l587_587951

noncomputable def quadratic_eqn (p : ℝ) (x : ℝ) : ℝ :=
  p * x^2 + (p^2 + p) * x - 3 * p^2 + 2 * p

def sum_of_squares_of_roots (p : ℝ) (x1 x2 : ℝ) : ℝ :=
  let S := x1 + x2 in
  let P := x1 * x2 in
  S^2 - 2 * P

theorem minimize_sum_of_squares_of_roots :
  ∀ (p x1 x2 : ℝ), p ≠ 0 →
  (quadratic_eqn p x1 = 0 ∧ quadratic_eqn p x2 = 0) →
  (sum_of_squares_of_roots p x1 x2 ≈ 1.10) := 
by sorry

end minimize_sum_of_squares_of_roots_l587_587951


namespace sum_of_extreme_numbers_is_10_l587_587027

variable (blocks : Set ℕ) 
variable (h_blocks : blocks = {1, 3, 7, 9})

theorem sum_of_extreme_numbers_is_10 : (blocks.max' (by decide) + blocks.min' (by decide)) = 10 := by
  sorry

end sum_of_extreme_numbers_is_10_l587_587027


namespace smallest_blocks_to_hide_hooks_l587_587152

/-- A block is a structure with exactly one hook and five slots --/
structure Block where
  hook : Bool
  slots : Nat := 5

/-- A function to determine visibility of hooks in a given arrangement of blocks --/
def all_hooks_hidden (blocks : List Block) : Bool :=
  -- Defining this function would require a proof based on the placement logic
  sorry

/-- The main proof problem for the number of blocks required to hide all hooks--/
theorem smallest_blocks_to_hide_hooks (n : Nat) (blocks : List Block) :
  (∀ i, i < n → (blocks[i] : Block).hook = false) → n ≥ 4 :=
  sorry

end smallest_blocks_to_hide_hooks_l587_587152


namespace two_numbers_correct_l587_587134

-- Definitions based on the conditions
def N₁ := 26829
def N₂ := 41463

-- Property 1 (derived condition): 9N1 + 10N2 = 180000
def property1 : 9 * N₁ + 10 * N₂ = 180000 := by
  sorry

-- Property 2 (derived condition): 10N1 - 11N2 = 360000
def property2 : 10 * N₁ - 11 * N₂ = 360000 := by
  sorry

-- Notification that the main theorem holds based on properties
theorem two_numbers_correct : N₁ = 26829 ∧ N₂ = 41463 := by
  apply And.intro
  . exact property1
  . exact property2
  sorry

end two_numbers_correct_l587_587134


namespace extremum_tangent_monotonic_l587_587988

variables {a b : ℝ}


/-- Rational function for given conditions and results.
  Given the function f(x) = (1 / 3) * x ^ 3 - a * x ^ 2 + (a ^ 2 - 1) * x + b
  1. If x = 1 is an extremum of f(x), then a = 2
  2. If the tangent line to the graph of y = f(x) at the point (1, f(1)) is x + y - 3 = 0,
     then the maximum value of f(x) on the interval [-2, 4] is 8
  3. When a ≠ 0, if f(x) is not monotonic in the interval (-1, 1),
     then the range of values for a is (-2, 0) ∪ (0, 2)
 -/
theorem extremum_tangent_monotonic
  (h_extremum : ∀ x, x = 1 → (derivative (f : ℝ → ℝ) x) = 0)
  (h_tangent : ∀ (pt : ℝ × ℝ), pt = (1, f 1) → line_equation pt = x + y - 3)
  (h_non_monotonic : ¬monotonic_on f (-1, 1))
  : a = 2 ∧ (f (-2) ≤ f (4) ∧ f (4) = 8) ∧ (a ∈ (-2, 0) ∪ (0, 2)) :=
sorry

end extremum_tangent_monotonic_l587_587988


namespace cost_flying_X_to_Y_l587_587754

def distance_XY : ℝ := 4500 -- Distance from X to Y in km
def cost_per_km_flying : ℝ := 0.12 -- Cost per km for flying in dollars
def booking_fee_flying : ℝ := 120 -- Booking fee for flying in dollars

theorem cost_flying_X_to_Y : 
    distance_XY * cost_per_km_flying + booking_fee_flying = 660 := by
  sorry

end cost_flying_X_to_Y_l587_587754


namespace find_y_l587_587588

variable {x y z a b c : ℝ}

def question_condition_1 := (x * y) / (x + y) = a
def question_condition_2 := (x * z) / (x + z) = b
def question_condition_3 := (y * z) / (y + z) = c

theorem find_y (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  question_condition_1 → question_condition_2 → question_condition_3 →
  y = (2 * a * b * c) / (b * c + a * c - a * b) :=
by
  intros
  sorry

end find_y_l587_587588


namespace subset_points_no_three_collinear_l587_587901

theorem subset_points_no_three_collinear :
  ∃ S : Finset (EuclideanSpace ℝ (Fin 2)), S.card = 63 ∧ 
  (∀ (A B C : Point), A ∈ S → B ∈ S → C ∈ S → ¬Collinear A B C) :=
begin
  sorry
end

end subset_points_no_three_collinear_l587_587901


namespace solution_set_of_inequality_l587_587722

noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := x^2 * f x

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_diff : ∀ x < 0, differentiable_at ℝ f x)
  (h_ineq : ∀ x < 0, 2 * f x + x * (derivative f x) > x^2) :
  {x : ℝ | (x + 2016)^2 * f (x + 2016) - 4 * f (-2) > 0} = set.Iic (-2018) :=
by {
  sorry
}

end solution_set_of_inequality_l587_587722


namespace average_rate_of_trip_l587_587735

theorem average_rate_of_trip (d : ℝ) (r1 : ℝ) (t1 : ℝ) (r_total : ℝ) :
  d = 640 →
  r1 = 80 →
  t1 = (320 / r1) →
  t2 = 3 * t1 →
  r_total = d / (t1 + t2) →
  r_total = 40 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end average_rate_of_trip_l587_587735


namespace perimeter_increase_l587_587814

-- Define the convex n-gon and the outward shift of sides
def convex_polygon (n : ℕ) (n_ge_3 : n ≥ 3) := Type -- represents a convex n-gon

-- Define the shifted convex polygon
noncomputable def shifted_polygon (P : convex_polygon n n_ge_3) := Type -- represents the shifted polygon

-- Define the perimeters of the original and shifted polygons
def perimeter (P : convex_polygon n n_ge_3) : ℝ := sorry -- Compute the perimeter of the polygon

def shifted_perimeter (P : convex_polygon n n_ge_3) : ℝ :=
  let P_shifted := shifted_polygon P
  (perimeter P_shifted : ℝ) -- Compute the perimeter of the shifted polygon

-- Main theorem
theorem perimeter_increase (P : convex_polygon n n_ge_3) (shift : ℝ) (shift_eq_5 : shift = 5):
  shifted_perimeter P - perimeter P > 30 :=
sorry

end perimeter_increase_l587_587814


namespace rectangle_perimeter_l587_587077

def sides_of_triangle : ℝ × ℝ × ℝ := (7, 10, 13)
def length_of_rectangle : ℝ := 7

noncomputable def semi_perimeter (a b c : ℝ) : ℝ := (a + b + c) / 2

noncomputable def area_of_triangle_using_herons_formula (a b c : ℝ) : ℝ :=
  let s := semi_perimeter a b c
  sqrt (s * (s - a) * (s - b) * (s - c))

noncomputable def area_of_triangle : ℝ :=
  area_of_triangle_using_herons_formula 7 10 13

noncomputable def width_of_rectangle (area length : ℝ) : ℝ :=
  area / length

noncomputable def perimeter_of_rectangle (length width : ℝ) : ℝ :=
  2 * (length + width)

def calculated_perimeter : ℝ :=
  perimeter_of_rectangle length_of_rectangle (width_of_rectangle area_of_triangle length_of_rectangle)

theorem rectangle_perimeter :
  calculated_perimeter = 14 + (40 * sqrt 3) / 7 :=
by 
  -- This is where the proof would go.
  sorry

end rectangle_perimeter_l587_587077


namespace prime_gt_three_divisible_l587_587306

open Nat

theorem prime_gt_three_divisible (p : ℕ) (hp : Prime p) (hgt : p > 3) : 30 ∣ (p^2 - 1) := by
  sorry

end prime_gt_three_divisible_l587_587306


namespace solve_equation_l587_587219

theorem solve_equation : 
  ∀ x : ℝ, 
  (((15 * x - x^2) / (x + 2)) * (x + (15 - x) / (x + 2)) = 54) → (x = 9 ∨ x = -1) :=
by
  sorry

end solve_equation_l587_587219


namespace smaller_square_area_proof_l587_587512

-- Definitions used from the problem conditions
def larger_square_area : ℕ := 144

def midpoint (a b : ℝ) : ℝ := (a + b) / 2

-- Define the concept of the side length of a square given its area
def side_length_of_square (area : ℝ) : ℝ := Real.sqrt area

-- Calculate the side length of the larger square
def s := side_length_of_square (larger_square_area)

-- Calculate the side length of the smaller square
def smaller_square_side := Real.sqrt (2 * (s/2)^2)

-- Calculate the area of the smaller square
def smaller_square_area := smaller_square_side ^ 2

-- The final theorem proving the area of the smaller square is 72 square units
theorem smaller_square_area_proof :
  smaller_square_area = 72 := sorry

end smaller_square_area_proof_l587_587512


namespace loss_percentage_l587_587513

variable (CP SP : ℕ) -- declare the variables for cost price and selling price

theorem loss_percentage (hCP : CP = 1400) (hSP : SP = 1190) : 
  ((CP - SP) / CP * 100) = 15 := by
sorry

end loss_percentage_l587_587513


namespace parallel_lines_slope_l587_587621

theorem parallel_lines_slope (m : ℝ) :
  let l1 := (m - 2) * x - 3 * y - 1 = 0,
      l2 := m * x + (m + 2) * y + 1 = 0 in
  (∀ (x y : ℝ), l1 = 0 → l2 = 0) → m = -4 :=
by
  sorry

end parallel_lines_slope_l587_587621


namespace smallest_prime_divisor_of_sum_l587_587464

theorem smallest_prime_divisor_of_sum : ∃ p : ℕ, Prime p ∧ p = 2 ∧ p ∣ (3 ^ 15 + 11 ^ 21) :=
by
  sorry

end smallest_prime_divisor_of_sum_l587_587464


namespace repeat_block_of_fraction_l587_587211

noncomputable def repend_check : String :=
  "235294"

theorem repeat_block_of_fraction :
  ∀ (a b : ℕ), b = 17 ∧ a = 4 → (dec_rep_repetend a b 6) = repend_check :=
by
  intros a b h
  have h₁ : a = 4 := h.left
  have h₂ : b = 17 := h.right
  rw [h₁, h₂]
  sorry  -- Proof is omitted according to instructions

end repeat_block_of_fraction_l587_587211


namespace measure_any_amount_l587_587068

theorem measure_any_amount (a : ℤ) (h : -1562 ≤ a ∧ a ≤ 1562) :
  ∃ (b c d e f : ℤ), b ∈ {-2, -1, 0, 1, 2} ∧ c ∈ {-2, -1, 0, 1, 2} ∧ 
  d ∈ {-2, -1, 0, 1, 2} ∧ e ∈ {-2, -1, 0, 1, 2} ∧ f ∈ {-2, -1, 0, 1, 2} ∧ 
  a = 625 * b + 125 * c + 25 * d + 5 * e + f :=
sorry

end measure_any_amount_l587_587068


namespace arithmetic_mean_of_set_l587_587404

theorem arithmetic_mean_of_set (n : ℕ) (h : n > 2) : 
  let numbers := [1 - 1 / n, 1 + 1 / n] ++ list.repeat 1 (n - 2) in
  let total_sum := (list.sum (1 - 1 / n :: 1 + 1 / n :: (list.repeat 1 (n - 2)))) in
  (total_sum / n) = 1 := by
  sorry

end arithmetic_mean_of_set_l587_587404


namespace T_description_l587_587704

-- Definitions of conditions
def T (x y : ℝ) : Prop :=
  (x + 3 = 4 ∧ y ≤ 9) ∨
  (y - 5 = 4 ∧ x ≤ 1) ∨
  (x + 3 = y - 5 ∧ x ≥ 1)

-- The problem statement in Lean: Prove that T describes three rays with a common point (1, 9)
theorem T_description :
  ∀ x y, T x y ↔ 
    ((x = 1 ∧ y ≤ 9) ∨
     (x ≤ 1 ∧ y = 9) ∨
     (x ≥ 1 ∧ y = x + 8)) :=
by sorry

end T_description_l587_587704


namespace sum_of_reciprocal_roots_l587_587724

theorem sum_of_reciprocal_roots (r s α β : ℝ) (h1 : 7 * r^2 - 8 * r + 6 = 0) (h2 : 7 * s^2 - 8 * s + 6 = 0) (h3 : α = 1 / r) (h4 : β = 1 / s) :
  α + β = 4 / 3 := 
sorry

end sum_of_reciprocal_roots_l587_587724


namespace find_possible_values_a_1_l587_587440

noncomputable def sequence (a : ℕ) (p : ℕ → ℕ) : ℕ → ℕ
| 0     := a
| (n+1) := sequence n - p n + (sequence n) / (p n)

lemma smallest_prime_divisor (a_n : ℕ) : ℕ :=
  nat.find (nat.prime_divisors a_n).exists_mem

theorem find_possible_values_a_1 (a : ℕ) :
  (∀ n : ℕ, 37 ∣ (sequence a smallest_prime_divisor n)) → 
  a = 37 * 37 :=
begin
  sorry
end

end find_possible_values_a_1_l587_587440


namespace part1_part2_part3_l587_587018

open Real 

-- Definitions used in the conditions
def simplification_property (n : ℕ) : Prop :=
  ∀ (n ∈ {2, 3, 5}),
  (1 / (sqrt n + sqrt (n - 1))) = (sqrt n - sqrt (n - 1))

-- Statements to be proven
theorem part1 : (1 / (sqrt 3 + 2)) = (2 - sqrt 3) := sorry

theorem part2 (n : ℕ) : (1 / (sqrt (n + 1) + sqrt n)) = (sqrt (n + 1) - sqrt n) := sorry

theorem part3 : ∑ k in finset.range 2022, (1 / (sqrt (k + 2) + sqrt (k + 1))) = (sqrt 2023 - 1) := sorry

end part1_part2_part3_l587_587018


namespace boat_speed_in_still_water_l587_587668

theorem boat_speed_in_still_water (B S : ℕ) (h1 : B + S = 13) (h2 : B - S = 5) : B = 9 :=
by
  sorry

end boat_speed_in_still_water_l587_587668


namespace actual_revenue_percentage_l587_587730

def last_year_revenue (R : ℝ) := R
def projected_revenue (R : ℝ) := 1.25 * R
def actual_revenue (R : ℝ) := 0.75 * R

theorem actual_revenue_percentage (R : ℝ) : 
  (actual_revenue R / projected_revenue R) * 100 = 60 :=
by
  sorry

end actual_revenue_percentage_l587_587730


namespace problem1_problem2_l587_587177

variable (m n x y : ℝ)

theorem problem1 : 4 * m * n^3 * (2 * m^2 - (3 / 4) * m * n^2) = 8 * m^3 * n^3 - 3 * m^2 * n^5 := sorry

theorem problem2 : (x - 6 * y^2) * (3 * x^3 + y) = 3 * x^4 + x * y - 18 * x^3 * y^2 - 6 * y^3 := sorry

end problem1_problem2_l587_587177


namespace square_area_is_4802_l587_587407

-- Condition: the length of the diagonal of the square is 98 meters.
def diagonal (d : ℝ) := d = 98

-- Goal: Prove that the area of the square field is 4802 square meters.
theorem square_area_is_4802 (d : ℝ) (h : diagonal d) : ∃ (A : ℝ), A = 4802 := 
by sorry

end square_area_is_4802_l587_587407


namespace mike_avg_speed_l587_587739

/-
  Given conditions:
  * total distance d = 640 miles
  * half distance h = 320 miles
  * first half average rate r1 = 80 mph
  * time for first half t1 = h / r1 = 4 hours
  * second half time t2 = 3 * t1 = 12 hours
  * total time tt = t1 + t2 = 16 hours
  * total distance d = 640 miles
  * average rate for entire trip should be (d/tt) = 40 mph.
  
  The goal is to prove that the average rate for the entire trip is 40 mph.
-/
theorem mike_avg_speed:
  let d := 640 in
  let h := 320 in
  let r1 := 80 in
  let t1 := h / r1 in
  let t2 := 3 * t1 in
  let tt := t1 + t2 in
  let avg_rate := d / tt in
  avg_rate = 40 := by
  sorry

end mike_avg_speed_l587_587739


namespace decomposition_l587_587846

def vec : Type := ℝ × ℝ × ℝ

def x : vec := (-9, 5, 5)
def p : vec := (4, 1, 1)
def q : vec := (2, 0, -3)
def r : vec := (-1, 2, 1)

def scalar_mul (a : ℝ) (v : vec) : vec :=
  (a * v.1, a * v.2, a * v.3)

def vec_add (v1 v2 : vec) : vec :=
  (v1.1 + v2.1, v1.2 + v2.2, v1.3 + v2.3)

def vec_lin_comb (a b c : ℝ) (v1 v2 v3 : vec) : vec :=
  vec_add (scalar_mul a v1) (vec_add (scalar_mul b v2) (scalar_mul c v3))

theorem decomposition : x = vec_lin_comb (-1) (-1) 3 p q r := by
  sorry

end decomposition_l587_587846


namespace complex_magnitude_product_l587_587562

theorem complex_magnitude_product :
  abs ((⟨5, -3⟩ : ℂ) * (⟨7, 24⟩ : ℂ)) = 25 * Real.sqrt 34 := by
  sorry

end complex_magnitude_product_l587_587562


namespace julia_total_food_cost_l587_587695

-- Definitions based on conditions
def weekly_total_cost : ℕ := 30
def rabbit_weeks : ℕ := 5
def rabbit_food_cost : ℕ := 12
def parrot_weeks : ℕ := 3
def parrot_food_cost : ℕ := weekly_total_cost - rabbit_food_cost

-- Proof statement
theorem julia_total_food_cost : 
  rabbit_weeks * rabbit_food_cost + parrot_weeks * parrot_food_cost = 114 := 
by 
  sorry

end julia_total_food_cost_l587_587695


namespace largest_minus_smallest_l587_587086

def largest_six_digit_number (d1 d2 d3 d4 d5 d6 : ℕ) : ℕ :=
  100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

def smallest_six_digit_number (d1 d2 d3 d4 d5 d6 : ℕ) : ℕ :=
  100000 * d1 + 10000 * d2 + 1000 * d3 + 100 * d4 + 10 * d5 + d6

def digits_set := {1, 4, 0}

axiom digits_used_twice (n : ℕ) : 
  (nat.digits 10 n).count 0 = 2 ∧ (nat.digits 10 n).count 1 = 2 ∧ (nat.digits 10 n).count 4 = 2

theorem largest_minus_smallest : 
  ∃ n1 n2 : ℕ, 
  (nat.digits 10 n1).count 0 = 2 ∧ (nat.digits 10 n1).count 1 = 2 ∧ (nat.digits 10 n1).count 4 = 2 ∧
  (nat.digits 10 n2).count 0 = 2 ∧ (nat.digits 10 n2).count 1 = 2 ∧ (nat.digits 10 n2).count 4 = 2 ∧
  n1 - n2 = 340956 :=
begin
  sorry
end

end largest_minus_smallest_l587_587086


namespace evaluate_m_l587_587206

theorem evaluate_m :
  ∀ m : ℝ, (243:ℝ)^(1/5) = 3^m → m = 1 :=
by
  intro m
  sorry

end evaluate_m_l587_587206


namespace pillar_volume_l587_587505

/-- Defining the radius of the cylindrical pillar -/
def radius : ℝ := 2

/-- Defining the height of the cylindrical pillar -/
def height : ℝ := 8

/-- Volume formula for a right circular cylinder with the given radius and height -/
def volume : ℝ := π * radius^2 * height

/-- Proven fact that the volume of the pillar is 32π cubic feet given the dimensions and fit constraints -/
theorem pillar_volume : volume = 32 * π := by
  sorry

end pillar_volume_l587_587505


namespace hyperbola_equation_l587_587995

section QuadrilateralHyperbola

variable (b : ℝ) (x y : ℝ)

-- Assume the given hyperbola equation, the equation of the circle, and the asymptote equations
def hyperbola : Prop := (x^2 / 4) - (y^2 / (b^2)) = 1
def circle : Prop := (x^2 + y^2) = 4
def asymptotes : Prop := (y = (b / 2) * x) ∨ (y = -(b / 2) * x)
def area_condition : Prop := 2 * abs x * (b * abs x / 2) = 2 * b (quadrilateral area is 2b)

-- Prove the required hyperbola equation
theorem hyperbola_equation : hyperbola b (λ b : ℝ, sqrt 12) :=
by
  sorry

end QuadrilateralHyperbola

end hyperbola_equation_l587_587995


namespace proof_find_C_proof_find_cos_A_l587_587954

noncomputable def find_C {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : Prop :=
  ∃ (C : ℝ), 0 < C ∧ C < Real.pi ∧ C = Real.pi / 6

noncomputable def find_cos_A {a b c : ℝ} {B : ℝ} (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : Prop :=
  ∃ (A : ℝ), Real.cos A = (Real.sqrt 5 - 2 * Real.sqrt 3) / 6

theorem proof_find_C (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) : find_C h1 :=
  sorry

theorem proof_find_cos_A (a b c B : ℝ) (h1 : 2 * c * (Real.cos B) = 2 * a - Real.sqrt 3 * b) (h2 : Real.cos B = 2 / 3) : find_cos_A h1 h2 :=
  sorry

end proof_find_C_proof_find_cos_A_l587_587954


namespace sin_alpha_plus_pi_l587_587616

theorem sin_alpha_plus_pi (α : ℝ) (h1 : α ∈ (π / 2, π)) (h2 : Real.tan α = -3 / 4) :
  Real.sin (α + π) = -3 / 5 := sorry

end sin_alpha_plus_pi_l587_587616


namespace unknown_number_l587_587476

theorem unknown_number (x : ℝ) (h : 7^8 - 6/x + 9^3 + 3 + 12 = 95) : x = 1 / 960908.333 :=
sorry

end unknown_number_l587_587476


namespace find_other_x_intercept_l587_587586

theorem find_other_x_intercept (a b c : ℝ) (h_vertex : ∀ x, x = 2 → y = -3) (h_x_intercept : ∀ x, x = 5 → y = 0) : 
  ∃ x, x = -1 ∧ y = 0 := 
sorry

end find_other_x_intercept_l587_587586


namespace find_f_7_l587_587707

noncomputable def f (a b c d x : ℝ) : ℝ :=
  a * x^8 + b * x^7 + c * x^3 + d * x - 6

theorem find_f_7 (a b c d : ℝ) (h : f a b c d (-7) = 10) :
  f a b c d 7 = 11529580 * a - 22 :=
sorry

end find_f_7_l587_587707


namespace find_lawn_length_l587_587873

theorem find_lawn_length
  (width_lawn : ℕ)
  (road_width : ℕ)
  (cost_total : ℕ)
  (cost_per_sqm : ℕ)
  (total_area_roads : ℕ)
  (area_roads_length : ℕ)
  (area_roads_breadth : ℕ)
  (length_lawn : ℕ) :
  width_lawn = 60 →
  road_width = 10 →
  cost_total = 3600 →
  cost_per_sqm = 3 →
  total_area_roads = cost_total / cost_per_sqm →
  area_roads_length = road_width * length_lawn →
  area_roads_breadth = road_width * (width_lawn - road_width) →
  total_area_roads = area_roads_length + area_roads_breadth →
  length_lawn = 70 :=
by
  intros h_width_lawn h_road_width h_cost_total h_cost_per_sqm h_total_area_roads h_area_roads_length h_area_roads_breadth h_total_area_roads_eq
  sorry

end find_lawn_length_l587_587873


namespace num_of_cute_integers_l587_587122

def is_cute (n : ℕ) : Prop :=
  ∃ (digits : list ℕ), 
    digits.perm (list.range 5).map (λ x, x + 1) ∧
    (∀ k : ℕ, (1 ≤ k ∧ k ≤ 5) → (nat.of_digits 10 (digits.take k)) % k = 0)

theorem num_of_cute_integers : 
  finset.card {n | n < 100000 ∧ is_cute n}.to_finset = 2 :=
sorry

end num_of_cute_integers_l587_587122


namespace sum_of_reciprocal_squares_lt_odd_ratio_l587_587003

theorem sum_of_reciprocal_squares_lt_odd_ratio (n : ℕ) (hn : n ≥ 2) :
  (1 + ∑ i in finset.range n, 1 / ((i+2) ^ 2 : ℝ)) < (2 * n - 1 : ℝ) / n :=
sorry

end sum_of_reciprocal_squares_lt_odd_ratio_l587_587003


namespace average_lifespan_of_sampled_units_euqls_1013_l587_587131

theorem average_lifespan_of_sampled_units_euqls_1013
  (n1 n2 n3 : ℕ)
  (l1 l2 l3 : ℕ)
  (total_units : ℕ)
  (sampled_units : ℕ)
  (h1 : n1 = 1)
  (h2 : n2 = 2)
  (h3 : n3 = 1)
  (htotal : total_units = 100)
  (l1_value : l1 = 980)
  (l2_value : l2 = 1020)
  (l3_value : l3 = 1032)
  (sampled_value : sampled_units = 100) :
  (25 * l1 + 50 * l2 + 25 * l3) / sampled_units = 1013 := 
by {
  rw [l1_value, l2_value, l3_value],
  -- This is equivalent to proving the weighted average formula
  sorry
}

end average_lifespan_of_sampled_units_euqls_1013_l587_587131


namespace sufficient_but_not_necessary_condition_l587_587956

theorem sufficient_but_not_necessary_condition 
  (i : ℂ) (a b : ℝ) (h_i : i = complex.I) :
  ((a = 1 ∧ b = 1) → ((a + b * i)^2 = 2 * i)) ∧
  ¬ ((a + b * i)^2 = 2 * i → (a = 1 ∧ b = 1)) :=
by sorry

end sufficient_but_not_necessary_condition_l587_587956


namespace arc_length_of_parametric_curve_l587_587899

noncomputable def arc_length (x y : ℝ → ℝ) (t1 t2 : ℝ) := ∫ t in t1..t2, Real.sqrt ((deriv x t) ^ 2 + (deriv y t) ^ 2)

def parametric_curve_x (t : ℝ) : ℝ := 4 * (2 * Real.cos t - Real.cos (2 * t))
def parametric_curve_y (t : ℝ) : ℝ := 4 * (2 * Real.sin t - Real.sin (2 * t))

-- Define the theorem stating that the arc length of the given parametric curve from t = 0 to t = π is 32.
theorem arc_length_of_parametric_curve :
  arc_length parametric_curve_x parametric_curve_y 0 π = 32 := sorry

end arc_length_of_parametric_curve_l587_587899


namespace motorboat_travel_time_l587_587139

variable (A B : Type) (t r p : ℝ) (s : ℝ := r + 2)

-- Conditions
def condition1 : Prop := A = A ∧ B = B
def condition2 : Prop := s = r + 2
def condition3 : Prop := ∃ c : ℝ, p = c -- motorboat's speed is constant
def condition4 : Prop := ∀ (t₁ : ℝ), (t₁ ≥ 0 → t₁ ≤ t → p * t₁ = B) -- motorboat reaches B and turns
def condition5 : Prop := t = 6

-- The final statement to prove
theorem motorboat_travel_time (h1 : condition1) (h2 : condition2) (h3 : condition3) (h4 : condition4) : condition5 :=
sorry

end motorboat_travel_time_l587_587139


namespace find_a_value_l587_587338

theorem find_a_value 
  (a : ℝ) 
  (P : ℝ × ℝ) 
  (circle_eq : ∀ x y : ℝ, x^2 + y^2 - 2 * a * x + 2 * y - 1 = 0)
  (M N : ℝ × ℝ)
  (tangent_condition : (N.snd - M.snd) / (N.fst - M.fst) + (M.fst + N.fst - 2) / (M.snd + N.snd) = 0) : 
  a = 3 ∨ a = -2 := 
sorry

end find_a_value_l587_587338


namespace arithmetic_geometric_sum_l587_587773

theorem arithmetic_geometric_sum (a : ℕ → ℕ) (n : ℕ)
  (h_arith : ∀ k, a (k + 1) - a k = 2)
  (h_geom : (a 3)^2 = a 1 * a 7) :
  (∑ i in Finset.range n, a i) = n * (n + 1) := by
  sorry

end arithmetic_geometric_sum_l587_587773


namespace binom_150_149_eq_150_l587_587184

theorem binom_150_149_eq_150 : Nat.binomial 150 149 = 150 := 
by 
  -- Property of binomial coefficients
  have h₁ : Nat.binomial 150 149 = Nat.binomial 150 (150 - 149) := Nat.choose_symm
  -- Simplifying using the given properties
  have h₂ : Nat.binomial 150 1 = 150 := Nat.choose_one_right
  -- Combining results
  rw [h₁, h₂]
  exact h₂

end binom_150_149_eq_150_l587_587184


namespace median_duration_is_127_point_5_l587_587794

def times := [45, 55, 65, 75, 80, 90, 100, 110, 120, 130, 125, 130, 140, 145, 150, 155, 180, 192, 200, 205, 210, 220, 230]

/-- Prove that the median duration of the times list is 127.5 seconds. -/
theorem median_duration_is_127_point_5 :
  let sorted_times := times.qsort (· <= ·)
  let tenth := sorted_times.nth 9 -- 10th element (0-based index)
  let eleventh := sorted_times.nth 10 -- 11th element (0-based index)
  (tenth + eleventh) / 2 = 127.5 :=
sorry

end median_duration_is_127_point_5_l587_587794


namespace simplify_expression_l587_587764

theorem simplify_expression : 1 - (1 / (2 + Real.sqrt 5)) + (1 / (2 - Real.sqrt 5)) = 1 - 2 * Real.sqrt 5 :=
by
  sorry

end simplify_expression_l587_587764


namespace price_new_container_l587_587133
noncomputable def originalVolume (d₁ h₁ : ℝ) : ℝ :=
  π * (d₁ / 2) ^ 2 * h₁

noncomputable def newVolume (d₂ h₂ : ℝ) : ℝ :=
  π * (d₂ / 2) ^ 2 * h₂

noncomputable def price (originalPrice : ℝ) (volumeRatio : ℝ) : ℝ :=
  originalPrice * volumeRatio

theorem price_new_container (d₁ h₁ d₂ h₂ originalPrice newPrice : ℝ)
  (originalVolume_eq : originalVolume d₁ h₁ = π * (d₁ / 2) ^ 2 * h₁)
  (newVolume_eq : newVolume d₂ h₂ = π * (d₂ / 2) ^ 2 * h₂)
  (price_eq : newPrice = price originalPrice (newVolume d₂ h₂ / originalVolume d₁ h₁))
  (d₁ := 5)
  (h₁ := 6)
  (d₂ := 10)
  (h₂ := 8)
  (originalPrice := 1.5)
  (newPrice := 8) :
  price_eq → newPrice = 8 := 
sorry

end price_new_container_l587_587133


namespace trigonometric_values_l587_587613

variable (α : ℝ)

theorem trigonometric_values (h : Real.cos (3 * Real.pi + α) = 3 / 5) :
  Real.cos α = -3 / 5 ∧
  Real.cos (Real.pi + α) = 3 / 5 ∧
  Real.sin (3 * Real.pi / 2 - α) = -3 / 5 :=
by
  sorry

end trigonometric_values_l587_587613


namespace worms_stolen_correct_l587_587728

-- Given conditions translated into Lean statements
def num_babies : ℕ := 6
def worms_per_baby_per_day : ℕ := 3
def papa_bird_worms : ℕ := 9
def mama_bird_initial_worms : ℕ := 13
def additional_worms_needed : ℕ := 34

-- From the conditions, determine the total number of worms needed for 3 days
def total_worms_needed : ℕ := worms_per_baby_per_day * num_babies * 3

-- Calculate how many worms they will have after catching additional worms
def total_worms_after_catching_more : ℕ := papa_bird_worms + mama_bird_initial_worms + additional_worms_needed

-- Amount suspected to be stolen
def worms_stolen : ℕ := total_worms_after_catching_more - total_worms_needed

theorem worms_stolen_correct : worms_stolen = 2 :=
by sorry

end worms_stolen_correct_l587_587728


namespace f_is_odd_l587_587345

noncomputable def f (x : ℝ) : ℝ := log (x + sqrt (1 + x^2))

theorem f_is_odd : ∀ (x : ℝ), f (-x) = - f x := 
by
  sorry

end f_is_odd_l587_587345


namespace average_rate_640_miles_trip_l587_587732

theorem average_rate_640_miles_trip 
  (total_distance : ℕ) 
  (first_half_distance : ℕ) 
  (first_half_rate : ℕ) 
  (second_half_time_multiplier : ℕ) 
  (first_half_time : ℕ := first_half_distance / first_half_rate)
  (second_half_time : ℕ := second_half_time_multiplier * first_half_time)
  (total_time : ℕ := first_half_time + second_half_time)
  (average_rate : ℕ := total_distance / total_time) : 
  total_distance = 640 ∧ 
  first_half_distance = 320 ∧ 
  first_half_rate = 80 ∧ 
  second_half_time_multiplier = 3 → 
  average_rate = 40 :=
by
  intros h
  obtain ⟨h1, h2, h3, h4⟩ := h
  rw [h1, h2, h3, h4] at *
  have h5 : first_half_time = 320 / 80 := rfl
  have h6 : second_half_time = 3 * (320 / 80) := rfl
  have h7 : total_time = (320 / 80) + 3 * (320 / 80) := rfl
  have h8 : average_rate = 640 / (4 + 12) := rfl
  have h9 : average_rate = 640 / 16 := rfl
  have average_rate_correct : average_rate = 40 := rfl
  exact average_rate_correct

end average_rate_640_miles_trip_l587_587732


namespace solve_system_l587_587765

theorem solve_system (x1 x2 x3 : ℝ) :
  (x1 - 2 * x2 + 3 * x3 = 5) ∧ 
  (2 * x1 + 3 * x2 - x3 = 7) ∧ 
  (3 * x1 + x2 + 2 * x3 = 12) 
  ↔ (x1, x2, x3) = (7 - 5 * x3, 1 - x3, x3) :=
by
  sorry

end solve_system_l587_587765


namespace find_f_of_2_l587_587285

def f (x : ℝ) : ℝ :=
  if x > 2 then 6 - x else 2 ^ x

theorem find_f_of_2 : f (4 - 2) = 2 := by
  sorry

end find_f_of_2_l587_587285


namespace monotonic_intervals_a1_min_a_no_zeros_in_interval_l587_587627

section
variable (a : ℝ)

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (2 - a) * (x - 1) - 2 * Real.log x

-- Part 1: Monotonic intervals when a = 1
theorem monotonic_intervals_a1 :
  let fa1 := (λ x : ℝ, x - 1 - 2 * Real.log x)
  ∀ x : ℝ, (x ∈ Ioo 0 2 → D fa1 x < 0) ∧ (2 ≤ x → D fa1 x > 0) :=
sorry

-- Part 2: Minimum value of a for no zeros in (0, 1/2)
theorem min_a_no_zeros_in_interval :
  (∀ x ∈ Ioo 0 (1/2), f a x ≠ 0) ↔ a ≥ 2 - 4 * Real.log 2 :=
sorry
end

end monotonic_intervals_a1_min_a_no_zeros_in_interval_l587_587627


namespace find_angle_B_l587_587337

theorem find_angle_B 
    (AB CD : ℝ → ℝ → Prop)
    (parallel : ∀ (A B C D: ℝ), AB A B → CD C D → AB.parallel CD) 
    (angle_A angle_B angle_C angle_D : ℝ)
    (h1 : angle_A = 3 * angle_D)
    (h2 : angle_C = 4 * angle_B)
    (supplementary_BC : angle_B + angle_C = 180)
    (supplementary_AD : angle_A + angle_D = 180) :
    angle_B = 36 :=
by
  sorry

end find_angle_B_l587_587337


namespace sum_reciprocals_roots_l587_587945

theorem sum_reciprocals_roots :
  (∃ p q : ℝ, p + q = 10 ∧ p * q = 3) →
  (∃ p q : ℝ, p ≠ 0 ∧ q ≠ 0 → (1 / p) + (1 / q) = 10 / 3) :=
by
  sorry

end sum_reciprocals_roots_l587_587945


namespace problem_statement_l587_587367

variables (m n : Line) (alpha beta : Plane)

-- Conditions
axiom different_lines : m ≠ n
axiom different_planes : alpha ≠ beta

-- Condition check: if m || n and m ⊥ alpha, then n ⊥ alpha
theorem problem_statement (h1 : m ∥ n) (h2 : m ⊥ alpha) : n ⊥ alpha := sorry

end problem_statement_l587_587367


namespace sequence_solution_l587_587221

theorem sequence_solution (n : ℕ) (k : ℕ) (h1 : 1 ≤ k ∧ k ≤ n) :
  let x : ℕ → ℚ := λ k, if k = 0 then 0 else if k = 1 then 1 else
  (λ c x_k x_k1, (c * x_k1 - (n - (k - 1)) * x_k) / (k - 1 + 1)) (n - 1) (x (k - 2)) (x (k - 1))
  in x k = (nat.choose (n - 1) (k - 1)) := 
sorry

end sequence_solution_l587_587221


namespace handshakes_at_gathering_l587_587799

theorem handshakes_at_gathering (num_couples : ℕ) (total_people : ℕ) (handshakes_without_restrictions : ℕ)
  (handshakes_not_occurring : ℕ) (handshakes_with_restrictions : ℕ) : 
  num_couples = 8 → total_people = 16 →
  handshakes_without_restrictions = (total_people * (total_people - 1)) / 2 →
  handshakes_not_occurring = num_couples →
  handshakes_with_restrictions = handshakes_without_restrictions - handshakes_not_occurring →
  handshakes_with_restrictions = 112 := 
by
  intros hnc htp hwr hno hwr'
  rw [hnc, htp, hwr, hno, hwr']
  norm_num
  rfl

end handshakes_at_gathering_l587_587799


namespace suresh_borrowed_amount_l587_587103

theorem suresh_borrowed_amount 
  (P: ℝ)
  (i1 i2 i3: ℝ)
  (t1 t2 t3: ℝ)
  (total_interest: ℝ)
  (h1 : i1 = 0.12) 
  (h2 : t1 = 3)
  (h3 : i2 = 0.09)
  (h4 : t2 = 5)
  (h5 : i3 = 0.13)
  (h6 : t3 = 3)
  (h_total : total_interest = 8160) 
  (h_interest_eq : total_interest = P * i1 * t1 + P * i2 * t2 + P * i3 * t3)
  : P = 6800 :=
by
  sorry

end suresh_borrowed_amount_l587_587103


namespace julia_spent_on_animals_l587_587693

theorem julia_spent_on_animals 
  (total_weekly_cost : ℕ)
  (weekly_cost_rabbit : ℕ)
  (weeks_rabbit : ℕ)
  (weeks_parrot : ℕ) :
  total_weekly_cost = 30 →
  weekly_cost_rabbit = 12 →
  weeks_rabbit = 5 →
  weeks_parrot = 3 →
  total_weekly_cost * weeks_parrot - weekly_cost_rabbit * weeks_parrot + weekly_cost_rabbit * weeks_rabbit = 114 :=
begin
  intros h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  linarith,
end

end julia_spent_on_animals_l587_587693


namespace functions_pair_B_same_l587_587480

theorem functions_pair_B_same (x : ℝ) (h : 0 < x) : 
  (∀ x > 0, (x^(1/2))^2 / x = 1) ∧ (∀ x > 0, x / (x^(1/2))^2 = 1) :=
by {
  sorry, -- Proof is not required
}

end functions_pair_B_same_l587_587480


namespace Dave_has_more_money_than_Derek_l587_587910

def Derek_initial := 40
def Derek_expense1 := 14
def Derek_expense2 := 11
def Derek_expense3 := 5
def Derek_remaining := Derek_initial - Derek_expense1 - Derek_expense2 - Derek_expense3

def Dave_initial := 50
def Dave_expense := 7
def Dave_remaining := Dave_initial - Dave_expense

def money_difference := Dave_remaining - Derek_remaining

theorem Dave_has_more_money_than_Derek : money_difference = 33 := by sorry

end Dave_has_more_money_than_Derek_l587_587910


namespace parallel_lines_m_eq_neg4_l587_587619

theorem parallel_lines_m_eq_neg4 (m : ℝ) (h1 : (m-2) ≠ -m) 
  (h2 : (m-2) / 3 = -m / (m + 2)) : m = -4 :=
sorry

end parallel_lines_m_eq_neg4_l587_587619


namespace ratio_of_faster_to_slower_l587_587452

def length_train : ℝ := 150 -- length of each train in meters
def crossing_time : ℝ := 8 -- time taken to cross each other in seconds
def faster_train_speed_kmh : ℝ := 90 -- speed of the faster train in km/h
def faster_train_speed_ms : ℝ := 90 * (1000 / 3600) -- speed of the faster train in m/s
def total_cross_distance : ℝ := 2 * length_train -- total distance covered when crossing in meters
def relative_speed : ℝ := total_cross_distance / crossing_time -- relative speed in m/s
def slower_train_speed : ℝ := relative_speed - faster_train_speed_ms -- speed of the slower train in m/s
def speed_ratio : ℝ := faster_train_speed_ms / slower_train_speed -- ratio of faster train speed to slower train speed

theorem ratio_of_faster_to_slower : speed_ratio = 2 := sorry

end ratio_of_faster_to_slower_l587_587452


namespace find_6th_number_l587_587770

theorem find_6th_number (A : ℕ → ℝ) (h_avg_11 : (∑ i in finset.range 11, A i) / 11 = 60)
  (h_avg_first_6 : (∑ i in finset.range 6, A i) / 6 = 98)
  (h_avg_last_6 : (∑ i in finset.range 11 \ (finset.range 5), A i) / 6 = 65) : 
  A 5 = 159 := by
sorry

end find_6th_number_l587_587770


namespace total_fishermen_count_l587_587803

theorem total_fishermen_count (F T F1 F2 : ℕ) (hT : T = 10000) (hF1 : F1 = 19 * 400) (hF2 : F2 = 2400) (hTotal : F1 + F2 = T) : F = 20 :=
by
  sorry

end total_fishermen_count_l587_587803


namespace minimum_monochromatic_triangles_l587_587605

theorem minimum_monochromatic_triangles 
  (n : ℕ) 
  (points : Fin n → ℝ × ℝ)
  (colors : Fin n → Bool) 
  (h_no_collinear : ∀ i j k : Fin n, i ≠ j → j ≠ k → k ≠ i → ¬ collinear (points i) (points j) (points k)) :
  (∃ (T₁ T₂ : Finset (Fin n)), T₁.card = 3 ∧ T₂.card = 3 ∧ monochromatic T₁ colors ∧ monochromatic T₂ colors ∧ T₁ ≠ T₂) → n ≥ 8 :=
begin
  sorry
end

def collinear (a b c : (ℝ × ℝ)) : Prop :=
  (b.1 - a.1) * (c.2 - a.2) = (c.1 - a.1) * (b.2 - a.2)

def monochromatic (T : Finset (Fin n)) (colors : Fin n → Bool) : Prop :=
  ∃ col, ∀ v ∈ T, colors v = col

end minimum_monochromatic_triangles_l587_587605


namespace range_of_a_l587_587582

theorem range_of_a (x a : ℝ) (h : ∀ x : ℝ, x^2 - 2 * x + 5 ≥ a^2 - 3 * a) : -1 ≤ a ∧ a ≤ 4 :=
sorry

end range_of_a_l587_587582


namespace circle_center_l587_587935

theorem circle_center 
    (x y : ℝ)
    (h : 4 * x^2 - 8 * x + 4 * y^2 + 16 * y + 20 = 0) : 
    (1, -2) = ((-(h - 4 * (x - 1)^2 - 4 * (y + 2)^2 + 20)) / 4, 
                (-(h - 4 * (x - 1)^2 - 4 * (y + 2)^2 + 20)) / 4 ) :=
sorry

end circle_center_l587_587935


namespace problem_l587_587987

def f (x : ℝ) := 5 * x^3

theorem problem : f 2012 + f (-2012) = 0 := 
by
  sorry

end problem_l587_587987


namespace intersection_S_T_l587_587292

def S := { x : ℝ | (x - 3) / (x - 6) ≤ 0 }
def T := { 2, 3, 4, 5, 6 }

theorem intersection_S_T : S ∩ T = { 3, 4, 5 } :=
by sorry

end intersection_S_T_l587_587292


namespace pet_store_customers_buy_different_pets_l587_587870

theorem pet_store_customers_buy_different_pets :
  let puppies := 20
  let kittens := 10
  let hamsters := 12
  let rabbits := 5
  let customers := 4
  (puppies * kittens * hamsters * rabbits * Nat.factorial customers = 288000) := 
by
  sorry

end pet_store_customers_buy_different_pets_l587_587870


namespace mean_of_data_l587_587624

variable (x : Fin 10 → ℝ)

def variance (x : Fin 10 → ℝ) : ℝ :=
  let μ := (∑ i, x i) / 10
  (∑ i, (x i - μ)^2) / 10

def shifted_square_sum (x : Fin 10 → ℝ) : ℝ :=
  ∑ i, (x i - 2)^2

theorem mean_of_data (h1 : variance x = 2) (h2 : shifted_square_sum x = 110) :
  (∑ i, x i) / 10 = -1 ∨ (∑ i, x i) / 10 = 5 :=
sorry

end mean_of_data_l587_587624


namespace determinant_example_l587_587186

theorem determinant_example : 
  let A := ![![7, -2], ![-3, 5]] in 
  Matrix.det A = 29 := 
by 
  sorry

end determinant_example_l587_587186


namespace Dylan_needs_two_trays_l587_587199

noncomputable def ice_cubes_glass : ℕ := 8
noncomputable def ice_cubes_pitcher : ℕ := 2 * ice_cubes_glass
noncomputable def tray_capacity : ℕ := 12
noncomputable def total_ice_cubes_used : ℕ := ice_cubes_glass + ice_cubes_pitcher
noncomputable def number_of_trays : ℕ := total_ice_cubes_used / tray_capacity

theorem Dylan_needs_two_trays : number_of_trays = 2 := by
  sorry

end Dylan_needs_two_trays_l587_587199


namespace camel_cost_l587_587116

variables {C H O E G Z : ℕ} 

-- conditions
axiom h1 : 10 * C = 24 * H
axiom h2 : 16 * H = 4 * O
axiom h3 : 6 * O = 4 * E
axiom h4 : 3 * E = 15 * G
axiom h5 : 8 * G = 20 * Z
axiom h6 : 12 * E = 180000

-- goal
theorem camel_cost : C = 6000 :=
by sorry

end camel_cost_l587_587116


namespace can_lids_per_box_l587_587160

/-- Aaron initially has 14 can lids, and after adding can lids from 3 boxes,
he has a total of 53 can lids. How many can lids are in each box? -/
theorem can_lids_per_box (initial : ℕ) (total : ℕ) (boxes : ℕ) (h₀ : initial = 14) (h₁ : total = 53) (h₂ : boxes = 3) :
  (total - initial) / boxes = 13 :=
by
  sorry

end can_lids_per_box_l587_587160


namespace sum_of_powers_mod_p_squared_l587_587971

theorem sum_of_powers_mod_p_squared (p : ℕ) [Fact (nat.prime p)] (hp_odd : p % 2 = 1) :
  (∑ k in Finset.range p, k ^ (2 * p - 1)) % (p ^ 2) = (p * (p + 1) / 2) % (p ^ 2) := by
  sorry

end sum_of_powers_mod_p_squared_l587_587971
