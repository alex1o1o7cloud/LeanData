import Mathlib

namespace NUMINAMATH_GPT_product_equivalence_l355_35587

theorem product_equivalence 
  (a b c d e f : ℝ) 
  (h1 : a + b + c + d + e + f = 0) 
  (h2 : a^3 + b^3 + c^3 + d^3 + e^3 + f^3 = 0) : 
  (a + c) * (a + d) * (a + e) * (a + f) = (b + c) * (b + d) * (b + e) * (b + f) :=
by
  sorry

end NUMINAMATH_GPT_product_equivalence_l355_35587


namespace NUMINAMATH_GPT_relationship_y1_y2_l355_35510

theorem relationship_y1_y2 (y1 y2 : ℝ) 
  (h1 : y1 = 3 / -1) 
  (h2 : y2 = 3 / -3) : 
  y1 < y2 :=
by
  sorry

end NUMINAMATH_GPT_relationship_y1_y2_l355_35510


namespace NUMINAMATH_GPT_dot_product_ABC_l355_35582

open Real

noncomputable def a : ℝ := 5
noncomputable def b : ℝ := 6
noncomputable def angleC : ℝ := π / 6  -- 30 degrees in radians

theorem dot_product_ABC :
  let CB := a
  let CA := b
  let angle_between := π - angleC  -- 150 degrees in radians
  let cos_angle := - (sqrt 3) / 2  -- cos(150 degrees)
  ∃ (dot_product : ℝ), dot_product = CB * CA * cos_angle :=
by
  have CB := a
  have CA := b
  have angle_between := π - angleC
  have cos_angle := - (sqrt 3) / 2
  use CB * CA * cos_angle
  sorry

end NUMINAMATH_GPT_dot_product_ABC_l355_35582


namespace NUMINAMATH_GPT_largest_multiple_of_18_with_digits_9_or_0_l355_35518

theorem largest_multiple_of_18_with_digits_9_or_0 :
  ∃ (n : ℕ), (n = 9990) ∧ (n % 18 = 0) ∧ (∀ d ∈ (n.digits 10), d = 9 ∨ d = 0) ∧ (n / 18 = 555) :=
by
  sorry

end NUMINAMATH_GPT_largest_multiple_of_18_with_digits_9_or_0_l355_35518


namespace NUMINAMATH_GPT_count_4_digit_numbers_with_conditions_l355_35568

def num_valid_numbers : Nat :=
  432

-- Statement declaring the proposition to be proved
theorem count_4_digit_numbers_with_conditions :
  (count_valid_numbers == 432) :=
sorry

end NUMINAMATH_GPT_count_4_digit_numbers_with_conditions_l355_35568


namespace NUMINAMATH_GPT_math_problem_l355_35580

theorem math_problem (n : ℤ) : 12 ∣ (n^2 * (n^2 - 1)) := 
by
  sorry

end NUMINAMATH_GPT_math_problem_l355_35580


namespace NUMINAMATH_GPT_sufficient_not_necessary_condition_l355_35512

theorem sufficient_not_necessary_condition (x y : ℝ) (h1 : x ≥ 1) (h2 : y ≥ 2) : 
  x + y ≥ 3 ∧ (¬ (∀ x y : ℝ, x + y ≥ 3 → x ≥ 1 ∧ y ≥ 2)) := 
by {
  sorry -- The actual proof goes here.
}

end NUMINAMATH_GPT_sufficient_not_necessary_condition_l355_35512


namespace NUMINAMATH_GPT_Jamie_minimum_4th_quarter_score_l355_35548

theorem Jamie_minimum_4th_quarter_score (q1 q2 q3 : ℤ) (avg : ℤ) (minimum_score : ℤ) :
  q1 = 84 → q2 = 80 → q3 = 83 → avg = 85 → minimum_score = 93 → 4 * avg - (q1 + q2 + q3) = minimum_score :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3, h4, h5]
  sorry

end NUMINAMATH_GPT_Jamie_minimum_4th_quarter_score_l355_35548


namespace NUMINAMATH_GPT_systematic_sampling_works_l355_35552

def missiles : List ℕ := List.range' 1 60 

-- Define the systematic sampling function
def systematic_sampling (start interval n : ℕ) : List ℕ :=
  List.range' 0 n |>.map (λ i => start + i * interval)

-- Stating the proof problem.
theorem systematic_sampling_works :
  systematic_sampling 5 12 5 = [5, 17, 29, 41, 53] :=
sorry

end NUMINAMATH_GPT_systematic_sampling_works_l355_35552


namespace NUMINAMATH_GPT_trains_cross_time_l355_35546

noncomputable def timeToCrossEachOther (L : ℝ) (T1 : ℝ) (T2 : ℝ) : ℝ :=
  let V1 := L / T1
  let V2 := L / T2
  let Vr := V1 + V2
  let totalDistance := L + L
  totalDistance / Vr

theorem trains_cross_time (L T1 T2 : ℝ) (hL : L = 120) (hT1 : T1 = 10) (hT2 : T2 = 15) :
  timeToCrossEachOther L T1 T2 = 12 :=
by
  simp [timeToCrossEachOther, hL, hT1, hT2]
  sorry

end NUMINAMATH_GPT_trains_cross_time_l355_35546


namespace NUMINAMATH_GPT_correct_statements_l355_35542

theorem correct_statements (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 2 * b = 1) :
  (∀ b, a = 1 - 2 * b → a^2 + b^2 ≥ 1/5) ∧
  (∀ a b, a + 2 * b = 1 → ab ≤ 1/8) ∧
  (∀ a b, a + 2 * b = 1 → 3 + 2 * Real.sqrt 2 ≤ (1 / a + 1 / b)) :=
by
  sorry

end NUMINAMATH_GPT_correct_statements_l355_35542


namespace NUMINAMATH_GPT_g_25_eq_zero_l355_35517

noncomputable def g : ℝ → ℝ := sorry

axiom g_def (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : x^2 * g y - y^2 * g x = g (x^2 / y^2)

theorem g_25_eq_zero : g 25 = 0 := by
  sorry

end NUMINAMATH_GPT_g_25_eq_zero_l355_35517


namespace NUMINAMATH_GPT_smallest_angle_l355_35550

noncomputable def smallest_angle_in_triangle (a b c : ℝ) : ℝ :=
  if h : 0 <= a ∧ 0 <= b ∧ 0 <= c ∧ a + b > c ∧ a + c > b ∧ b + c > a then
    Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))
  else
    0

theorem smallest_angle (a b c : ℝ) (h₁ : a = 4) (h₂ : b = 3) (h₃ : c = 2) :
  smallest_angle_in_triangle a b c = Real.arccos (7 / 8) :=
sorry

end NUMINAMATH_GPT_smallest_angle_l355_35550


namespace NUMINAMATH_GPT_interest_calculation_l355_35590

theorem interest_calculation (P : ℝ) (r : ℝ) (CI SI : ℝ → ℝ) (n : ℝ) :
  P = 1300 →
  r = 0.10 →
  (CI n - SI n = 13) →
  (CI n = P * (1 + r)^n - P) →
  (SI n = P * r * n) →
  (1.10 ^ n - 1 - 0.10 * n = 0.01) →
  n = 2 :=
by
  intro P_eq r_eq diff_eq CI_def SI_def equation
  -- Sorry, this is just a placeholder. The proof is omitted.
  sorry

end NUMINAMATH_GPT_interest_calculation_l355_35590


namespace NUMINAMATH_GPT_club_members_l355_35547

variable (x : ℕ)

theorem club_members (h1 : 2 * x + 5 = x + 15) : x = 10 := by
  sorry

end NUMINAMATH_GPT_club_members_l355_35547


namespace NUMINAMATH_GPT_fraction_identity_l355_35513

theorem fraction_identity (a b : ℝ) (h : (1/a + 1/b) / (1/a - 1/b) = 1009) : (a + b) / (a - b) = -1009 :=
by
  sorry

end NUMINAMATH_GPT_fraction_identity_l355_35513


namespace NUMINAMATH_GPT_pq_implications_l355_35555

theorem pq_implications (p q : Prop) (hpq_or : p ∨ q) (hpq_and : p ∧ q) : p ∧ q :=
by
  sorry

end NUMINAMATH_GPT_pq_implications_l355_35555


namespace NUMINAMATH_GPT_num_two_digit_numbers_l355_35529

-- Define the set of given digits
def digits : Finset ℕ := {0, 2, 5}

-- Define the function that counts the number of valid two-digit numbers
def count_two_digit_numbers (d : Finset ℕ) : ℕ :=
  (d.erase 0).card * (d.card - 1)

theorem num_two_digit_numbers : count_two_digit_numbers digits = 4 :=
by {
  -- sorry placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_num_two_digit_numbers_l355_35529


namespace NUMINAMATH_GPT_pony_wait_time_l355_35570

-- Definitions of the conditions
def cycle_time_monster_A : ℕ := 2 + 1 -- hours (2 awake, 1 rest)
def cycle_time_monster_B : ℕ := 3 + 2 -- hours (3 awake, 2 rest)

-- The theorem to prove the correct answer
theorem pony_wait_time :
  Nat.lcm cycle_time_monster_A cycle_time_monster_B = 15 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_pony_wait_time_l355_35570


namespace NUMINAMATH_GPT_find_x_perpendicular_l355_35571

/-- Given vectors a = ⟨-1, 2⟩ and b = ⟨1, x⟩, if a is perpendicular to (a + 2 * b),
    then x = -3/4. -/
theorem find_x_perpendicular
  (x : ℝ)
  (a : ℝ × ℝ := (-1, 2))
  (b : ℝ × ℝ := (1, x))
  (h : (a.1 * (a.1 + 2 * b.1) + a.2 * (a.2 + 2 * b.2) = 0)) :
  x = -3 / 4 :=
sorry

end NUMINAMATH_GPT_find_x_perpendicular_l355_35571


namespace NUMINAMATH_GPT_Euclid_Middle_School_AMC8_contest_l355_35585

theorem Euclid_Middle_School_AMC8_contest (students_Germain students_Newton students_Young : ℕ)
       (hG : students_Germain = 11) 
       (hN : students_Newton = 8) 
       (hY : students_Young = 9) : 
       students_Germain + students_Newton + students_Young = 28 :=
by
  sorry

end NUMINAMATH_GPT_Euclid_Middle_School_AMC8_contest_l355_35585


namespace NUMINAMATH_GPT_solve_cubic_equation_l355_35559

theorem solve_cubic_equation :
  ∀ x : ℝ, (x^3 + 2 * (x + 1)^3 + (x + 2)^3 = (x + 4)^3) → x = 3 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_cubic_equation_l355_35559


namespace NUMINAMATH_GPT_original_selling_price_l355_35598

-- Definitions based on the conditions
def original_price : ℝ := 933.33

-- Given conditions
def discount_rate : ℝ := 0.40
def price_after_discount : ℝ := 560.0

-- Lean theorem statement to prove that original selling price (x) is equal to 933.33
theorem original_selling_price (x : ℝ) 
  (h1 : x * (1 - discount_rate) = price_after_discount) : 
  x = original_price :=
  sorry

end NUMINAMATH_GPT_original_selling_price_l355_35598


namespace NUMINAMATH_GPT_maximum_ab_l355_35574

theorem maximum_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 3*a + 8*b = 48) : ab ≤ 24 :=
by
  sorry

end NUMINAMATH_GPT_maximum_ab_l355_35574


namespace NUMINAMATH_GPT_triangle_area_290_l355_35560

theorem triangle_area_290 
  (P Q R : ℝ × ℝ)
  (h1 : (R.1 - P.1) * (R.1 - Q.1) + (R.2 - P.2) * (R.2 - Q.2) = 0) -- Right triangle condition
  (h2 : (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 50^2) -- Length of hypotenuse PQ
  (h3 : ∀ x: ℝ, (x, x - 2) = P) -- Median through P
  (h4 : ∀ x: ℝ, (x, 3 * x + 3) = Q) -- Median through Q
  :
  ∃ (area : ℝ), area = 290 := 
sorry

end NUMINAMATH_GPT_triangle_area_290_l355_35560


namespace NUMINAMATH_GPT_max_true_statements_l355_35539

theorem max_true_statements 
  (a b : ℝ) 
  (cond1 : a > 0) 
  (cond2 : b > 0) : 
  ( 
    ( (1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( (1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ (a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
    ∨ 
    ( ¬(1 / a > 1 / b) ∧ ¬(a^2 < b^2) 
      ∧ (a > b) ∧ (a > 0) ∧ (b > 0) ) 
  ) 
→ 
  (true ∧ true ∧ true ∧ true → 4 = 4) :=
sorry

end NUMINAMATH_GPT_max_true_statements_l355_35539


namespace NUMINAMATH_GPT_cheryl_material_left_l355_35531

def square_yards_left (bought1 bought2 used : ℚ) : ℚ :=
  bought1 + bought2 - used

theorem cheryl_material_left :
  square_yards_left (4/19) (2/13) (0.21052631578947367 : ℚ) = (0.15384615384615385 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_cheryl_material_left_l355_35531


namespace NUMINAMATH_GPT_tim_balloon_count_l355_35563

theorem tim_balloon_count (Dan_balloons : ℕ) (h1 : Dan_balloons = 59) (Tim_balloons : ℕ) (h2 : Tim_balloons = 11 * Dan_balloons) : Tim_balloons = 649 :=
sorry

end NUMINAMATH_GPT_tim_balloon_count_l355_35563


namespace NUMINAMATH_GPT_log_expression_simplify_l355_35507

variable (x y : ℝ)

theorem log_expression_simplify (hx : 0 < x) (hx' : x ≠ 1) (hy : 0 < y) (hy' : y ≠ 1) :
  (Real.log x^2 / Real.log y^4) * 
  (Real.log y^3 / Real.log x^3) * 
  (Real.log x^4 / Real.log y^5) * 
  (Real.log y^5 / Real.log x^2) * 
  (Real.log x^3 / Real.log y^3) = (1 / 3) * Real.log x / Real.log y := 
sorry

end NUMINAMATH_GPT_log_expression_simplify_l355_35507


namespace NUMINAMATH_GPT_polynomial_divisibility_l355_35549

theorem polynomial_divisibility (a b : ℕ) (h1 : a > 0) (h2 : b > 0) 
    (h3 : (a + b^3) % (a^2 + 3 * a * b + 3 * b^2 - 1) = 0) : 
    ∃ k : ℕ, k ≥ 1 ∧ (a^2 + 3 * a * b + 3 * b^2 - 1) % k^3 = 0 :=
    sorry

end NUMINAMATH_GPT_polynomial_divisibility_l355_35549


namespace NUMINAMATH_GPT_find_coefficients_l355_35541

theorem find_coefficients (a b p q : ℝ) :
    (∀ x : ℝ, (2 * x - 1) ^ 20 - (a * x + b) ^ 20 = (x^2 + p * x + q) ^ 10) →
    a = -2 * b ∧ (b = 1 ∨ b = -1) ∧ p = -1 ∧ q = 1 / 4 :=
by 
    sorry

end NUMINAMATH_GPT_find_coefficients_l355_35541


namespace NUMINAMATH_GPT_number_of_members_greater_than_median_l355_35581

theorem number_of_members_greater_than_median (n : ℕ) (median : ℕ) (avg_age : ℕ) (youngest : ℕ) (oldest : ℕ) :
  n = 100 ∧ avg_age = 21 ∧ youngest = 1 ∧ oldest = 70 →
  ∃ k, k = 50 :=
by
  sorry

end NUMINAMATH_GPT_number_of_members_greater_than_median_l355_35581


namespace NUMINAMATH_GPT_volume_of_box_ground_area_of_box_l355_35508

-- Given conditions
variable (l w h : ℕ)
variable (hl : l = 20)
variable (hw : w = 15)
variable (hh : h = 5)

-- Define volume and ground area
def volume (l w h : ℕ) : ℕ := l * w * h
def ground_area (l w : ℕ) : ℕ := l * w

-- Theorem to prove the correct volume
theorem volume_of_box : volume l w h = 1500 := by
  rw [hl, hw, hh]
  sorry

-- Theorem to prove the correct ground area
theorem ground_area_of_box : ground_area l w = 300 := by
  rw [hl, hw]
  sorry

end NUMINAMATH_GPT_volume_of_box_ground_area_of_box_l355_35508


namespace NUMINAMATH_GPT_line_equation_through_two_points_l355_35533

noncomputable def LineEquation (x0 y0 x1 y1 x y : ℝ) : Prop :=
  (x1 ≠ x0) → (y1 ≠ y0) → 
  (y - y0) / (y1 - y0) = (x - x0) / (x1 - x0)

theorem line_equation_through_two_points 
  (x0 y0 x1 y1 : ℝ) 
  (h₁ : x1 ≠ x0) 
  (h₂ : y1 ≠ y0) : 
  ∀ (x y : ℝ), LineEquation x0 y0 x1 y1 x y :=  
by
  sorry

end NUMINAMATH_GPT_line_equation_through_two_points_l355_35533


namespace NUMINAMATH_GPT_incorrect_statement_A_l355_35551

-- Definitions based on conditions
def equilibrium_shifts (condition: Type) : Prop := sorry
def value_K_changes (condition: Type) : Prop := sorry

-- The incorrect statement definition
def statement_A (condition: Type) : Prop := equilibrium_shifts condition → value_K_changes condition

-- The final theorem stating that 'statement_A' is incorrect
theorem incorrect_statement_A (condition: Type) : ¬ statement_A condition :=
sorry

end NUMINAMATH_GPT_incorrect_statement_A_l355_35551


namespace NUMINAMATH_GPT_geometric_series_sum_l355_35538

/-- 
The series is given as 1/2^2 + 1/2^3 + 1/2^4 + 1/2^5 + 1/2^6 + 1/2^7 + 1/2^8.
First term a = 1/4 and common ratio r = 1/2 and number of terms n = 7. 
The sum should be 127/256.
-/
theorem geometric_series_sum :
  let a := 1 / 4
  let r := 1 / 2
  let n := 7
  let S := (a * (1 - r^n)) / (1 - r)
  S = 127 / 256 :=
by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l355_35538


namespace NUMINAMATH_GPT_sum_of_remainders_l355_35502

theorem sum_of_remainders (n : ℤ) (h : n % 18 = 11) :
  (n % 2 + n % 9) = 3 :=
sorry

end NUMINAMATH_GPT_sum_of_remainders_l355_35502


namespace NUMINAMATH_GPT_solve_N_l355_35596

noncomputable def N (a b c d : ℝ) := (a + b) / c - d

theorem solve_N : 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  N a b c d = -1 :=
by 
  let a := (Real.sqrt (Real.sqrt 6 + 3))
  let b := (Real.sqrt (Real.sqrt 6 - 3))
  let c := (Real.sqrt (Real.sqrt 6 + 2))
  let d := (Real.sqrt (4 - 2 * Real.sqrt 3))
  let n := N a b c d
  sorry

end NUMINAMATH_GPT_solve_N_l355_35596


namespace NUMINAMATH_GPT_area_convex_quadrilateral_l355_35506

theorem area_convex_quadrilateral (x y : ℝ) :
  (x^2 + y^2 = 73 ∧ x * y = 24) →
  -- You can place a formal statement specifying the four vertices here if needed
  ∃ a b c d : ℝ × ℝ,
  a.1^2 + a.2^2 = 73 ∧
  a.1 * a.2 = 24 ∧
  b.1^2 + b.2^2 = 73 ∧
  b.1 * b.2 = 24 ∧
  c.1^2 + c.2^2 = 73 ∧
  c.1 * c.2 = 24 ∧
  d.1^2 + d.2^2 = 73 ∧
  d.1 * d.2 = 24 ∧
  -- Ensure the quadrilateral forms a rectangle (additional conditions here)
  -- Compute the side lengths and area
  -- Specify finally the area and prove it equals 110
  True :=
sorry

end NUMINAMATH_GPT_area_convex_quadrilateral_l355_35506


namespace NUMINAMATH_GPT_circle_Q_radius_l355_35503

theorem circle_Q_radius
  (radius_P : ℝ := 2)
  (radius_S : ℝ := 4)
  (u v : ℝ)
  (h1: (2 + v)^2 = (2 + u)^2 + v^2)
  (h2: (4 - v)^2 = u^2 + v^2)
  (h3: v = u + u^2 / 2)
  (h4: v = 2 - u^2 / 4) :
  v = 16 / 9 :=
by
  /- Proof goes here. -/
  sorry

end NUMINAMATH_GPT_circle_Q_radius_l355_35503


namespace NUMINAMATH_GPT_express_f12_in_terms_of_a_l355_35522

variable {f : ℝ → ℝ}
variable {a : ℝ}
variable (f_add : ∀ x y : ℝ, f (x + y) = f x + f y)
variable (f_neg_three : f (-3) = a)

theorem express_f12_in_terms_of_a : f 12 = -4 * a := sorry

end NUMINAMATH_GPT_express_f12_in_terms_of_a_l355_35522


namespace NUMINAMATH_GPT_isosceles_right_triangle_hypotenuse_length_l355_35535

theorem isosceles_right_triangle_hypotenuse_length (A B C : ℝ) (h1 : (A = 0) ∧ (B = 0) ∧ (C = 1)) (h2 : AC = 5) (h3 : BC = 5) : 
  AB = 5 * Real.sqrt 2 := 
sorry

end NUMINAMATH_GPT_isosceles_right_triangle_hypotenuse_length_l355_35535


namespace NUMINAMATH_GPT_log_comparisons_l355_35583

noncomputable def a := Real.log 3 / Real.log 2
noncomputable def b := Real.log 3 / (2 * Real.log 2)
noncomputable def c := 1 / 2

theorem log_comparisons : c < b ∧ b < a := 
by
  sorry

end NUMINAMATH_GPT_log_comparisons_l355_35583


namespace NUMINAMATH_GPT_power_mod_zero_problem_solution_l355_35566

theorem power_mod_zero (n : ℕ) (h : n ≥ 2) : 2 ^ n % 4 = 0 :=
  sorry

theorem problem_solution : 2 ^ 300 % 4 = 0 :=
  power_mod_zero 300 (by norm_num)

end NUMINAMATH_GPT_power_mod_zero_problem_solution_l355_35566


namespace NUMINAMATH_GPT_max_consecutive_semi_primes_l355_35516

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def is_semi_prime (n : ℕ) : Prop := 
  n > 25 ∧ ∃ (p q : ℕ), is_prime p ∧ is_prime q ∧ p ≠ q ∧ n = p + q

theorem max_consecutive_semi_primes : ∃ (N : ℕ), N = 5 ∧
  ∀ (a b : ℕ), (a > 25) ∧ (b = a + 4) → 
  (∀ n, a ≤ n ∧ n ≤ b → is_semi_prime n) ↔ N = 5 := sorry

end NUMINAMATH_GPT_max_consecutive_semi_primes_l355_35516


namespace NUMINAMATH_GPT_train_speed_is_correct_l355_35553

-- Conditions
def train_length := 190.0152  -- in meters
def crossing_time := 17.1     -- in seconds

-- Convert units
def train_length_km := train_length / 1000  -- in kilometers
def crossing_time_hr := crossing_time / 3600  -- in hours

-- Statement of the proof problem
theorem train_speed_is_correct :
  (train_length_km / crossing_time_hr) = 40 :=
sorry

end NUMINAMATH_GPT_train_speed_is_correct_l355_35553


namespace NUMINAMATH_GPT_european_customer_savings_l355_35565

noncomputable def popcorn_cost : ℝ := 8 - 3
noncomputable def drink_cost : ℝ := popcorn_cost + 1
noncomputable def candy_cost : ℝ := drink_cost / 2

noncomputable def discounted_popcorn_cost : ℝ := popcorn_cost * (1 - 0.15)
noncomputable def discounted_candy_cost : ℝ := candy_cost * (1 - 0.1)

noncomputable def total_normal_cost : ℝ := 8 + discounted_popcorn_cost + drink_cost + discounted_candy_cost
noncomputable def deal_price : ℝ := 20
noncomputable def savings_in_dollars : ℝ := total_normal_cost - deal_price

noncomputable def exchange_rate : ℝ := 0.85
noncomputable def savings_in_euros : ℝ := savings_in_dollars * exchange_rate

theorem european_customer_savings : savings_in_euros = 0.81 := by
  sorry

end NUMINAMATH_GPT_european_customer_savings_l355_35565


namespace NUMINAMATH_GPT_WorldCup_group_stage_matches_l355_35576

theorem WorldCup_group_stage_matches
  (teams : ℕ)
  (groups : ℕ)
  (teams_per_group : ℕ)
  (matches_per_group : ℕ)
  (total_matches : ℕ) :
  teams = 32 ∧ 
  groups = 8 ∧ 
  teams_per_group = 4 ∧ 
  matches_per_group = teams_per_group * (teams_per_group - 1) / 2 ∧ 
  total_matches = matches_per_group * groups →
  total_matches = 48 :=
by 
  -- sorry lets Lean skip the proof.
  sorry

end NUMINAMATH_GPT_WorldCup_group_stage_matches_l355_35576


namespace NUMINAMATH_GPT_find_a_8_l355_35505

-- Define the arithmetic sequence and its sum formula.
def arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n, a (n + 1) = a n + d

-- Define the sum of the first 'n' terms in the arithmetic sequence.
def sum_of_first_n_terms (S : ℕ → ℕ) (a : ℕ → ℕ) (n : ℕ) :=
  S n = n * (a 1 + a n) / 2

-- Given conditions
def S_15_eq_90 (S : ℕ → ℕ) : Prop := S 15 = 90

-- Prove that a_8 is 6
theorem find_a_8 (S : ℕ → ℕ) (a : ℕ → ℕ) (d : ℕ)
  (h1 : arithmetic_sequence a d) (h2 : sum_of_first_n_terms S a 15)
  (h3 : S_15_eq_90 S) : a 8 = 6 :=
sorry

end NUMINAMATH_GPT_find_a_8_l355_35505


namespace NUMINAMATH_GPT_abs_inequality_solution_l355_35561

theorem abs_inequality_solution {x : ℝ} (h : |x + 1| < 5) : -6 < x ∧ x < 4 :=
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l355_35561


namespace NUMINAMATH_GPT_greatest_integer_value_of_x_l355_35504

theorem greatest_integer_value_of_x :
  ∃ x : ℤ, (3 * |2 * x + 1| + 10 > 28) ∧ (∀ y : ℤ, 3 * |2 * y + 1| + 10 > 28 → y ≤ x) :=
sorry

end NUMINAMATH_GPT_greatest_integer_value_of_x_l355_35504


namespace NUMINAMATH_GPT_kara_total_water_intake_l355_35544

-- Define dosages and water intake per tablet
def medicationA_doses_per_day := 3
def medicationB_doses_per_day := 4
def medicationC_doses_per_day := 2
def medicationD_doses_per_day := 1

def water_per_tablet_A := 4
def water_per_tablet_B := 5
def water_per_tablet_C := 6
def water_per_tablet_D := 8

-- Compute weekly water intake
def weekly_water_intake_medication (doses_per_day water_per_tablet : ℕ) (days : ℕ) : ℕ :=
  doses_per_day * water_per_tablet * days

-- Total water intake for two weeks if instructions are followed perfectly
def total_water_no_errors :=
  2 * (weekly_water_intake_medication medicationA_doses_per_day water_per_tablet_A 7 +
       weekly_water_intake_medication medicationB_doses_per_day water_per_tablet_B 7 +
       weekly_water_intake_medication medicationC_doses_per_day water_per_tablet_C 7 +
       weekly_water_intake_medication medicationD_doses_per_day water_per_tablet_D 7)

-- Missed doses in second week
def missed_water_second_week :=
  3 * water_per_tablet_A +
  2 * water_per_tablet_B +
  2 * water_per_tablet_C +
  1 * water_per_tablet_D

-- Total water actually drunk over two weeks
def total_water_real :=
  total_water_no_errors - missed_water_second_week

-- Proof statement
theorem kara_total_water_intake :
  total_water_real = 686 :=
by
  sorry

end NUMINAMATH_GPT_kara_total_water_intake_l355_35544


namespace NUMINAMATH_GPT_Hari_joined_after_5_months_l355_35595

noncomputable def Praveen_investment_per_year : ℝ := 3360 * 12
noncomputable def Hari_investment_for_given_months (x : ℝ) : ℝ := 8640 * (12 - x)

theorem Hari_joined_after_5_months (x : ℝ) (h : Praveen_investment_per_year / Hari_investment_for_given_months x = 2 / 3) : x = 5 :=
by
  sorry

end NUMINAMATH_GPT_Hari_joined_after_5_months_l355_35595


namespace NUMINAMATH_GPT_parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l355_35527

-- Definitions for the problem conditions
def parabola_symmetry_axis := "coordinate axis"
def parabola_vertex := (0, 0)
def directrix_equation := "x = -1"
def intersects_at_two_points (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) := (l P.1 = P.2) ∧ (l Q.1 = Q.2) ∧ (P ≠ Q)

-- Main theorem statements
theorem parabola_standard_equation : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") → 
  ∃ p, 0 < p ∧ ∀ y x, y^2 = 4 * p * x := 
  sorry

theorem oa_dot_ob_value (l : ℝ → ℝ) (focus : ℝ × ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  l focus.1 = focus.2 → 
  (P.1 * Q.1 + P.2 * Q.2 = -3) := 
  sorry

theorem line_passes_fixed_point (l : ℝ → ℝ) (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  (parabola_symmetry_axis = "coordinate axis") ∧ 
  (parabola_vertex = (0, 0)) ∧ 
  (directrix_equation = "x = -1") ∧ 
  intersects_at_two_points l P Q ∧ 
  (P.1 * Q.1 + P.2 * Q.2 = -4) → 
  ∃ fp, fp = (2,0) := 
  sorry

end NUMINAMATH_GPT_parabola_standard_equation_oa_dot_ob_value_line_passes_fixed_point_l355_35527


namespace NUMINAMATH_GPT_shifted_parabola_equation_l355_35528

-- Define the original parabola function
def original_parabola (x : ℝ) : ℝ := -2 * x^2

-- Define the shifted parabola function
def shifted_parabola (x : ℝ) : ℝ := -2 * (x + 1)^2 + 3

-- Proposition to prove that the given parabola equation is correct after transformations
theorem shifted_parabola_equation : 
  ∀ x : ℝ, shifted_parabola x = -2 * (x + 1)^2 + 3 :=
by
  sorry

end NUMINAMATH_GPT_shifted_parabola_equation_l355_35528


namespace NUMINAMATH_GPT_unique_fish_total_l355_35558

-- Define the conditions as stated in the problem
def Micah_fish : ℕ := 7
def Kenneth_fish : ℕ := 3 * Micah_fish
def Matthias_fish : ℕ := Kenneth_fish - 15
def combined_fish : ℕ := Micah_fish + Kenneth_fish + Matthias_fish
def Gabrielle_fish : ℕ := 2 * combined_fish

def shared_fish_Micah_Matthias : ℕ := 4
def shared_fish_Kenneth_Gabrielle : ℕ := 6

-- Define the total unique fish computation
def total_unique_fish : ℕ := (Micah_fish + Kenneth_fish + Matthias_fish + Gabrielle_fish) - (shared_fish_Micah_Matthias + shared_fish_Kenneth_Gabrielle)

-- State the theorem
theorem unique_fish_total : total_unique_fish = 92 := by
  -- Proof omitted
  sorry

end NUMINAMATH_GPT_unique_fish_total_l355_35558


namespace NUMINAMATH_GPT_evaluate_powers_of_i_l355_35591

theorem evaluate_powers_of_i :
  (Complex.I ^ 50) + (Complex.I ^ 105) = -1 + Complex.I :=
by 
  sorry

end NUMINAMATH_GPT_evaluate_powers_of_i_l355_35591


namespace NUMINAMATH_GPT_cocktail_cost_per_litre_is_accurate_l355_35569

noncomputable def mixed_fruit_juice_cost_per_litre : ℝ := 262.85
noncomputable def acai_berry_juice_cost_per_litre : ℝ := 3104.35
noncomputable def mixed_fruit_juice_litres : ℝ := 35
noncomputable def acai_berry_juice_litres : ℝ := 23.333333333333336

noncomputable def cocktail_total_cost : ℝ := 
  (mixed_fruit_juice_cost_per_litre * mixed_fruit_juice_litres) +
  (acai_berry_juice_cost_per_litre * acai_berry_juice_litres)

noncomputable def cocktail_total_volume : ℝ := 
  mixed_fruit_juice_litres + acai_berry_juice_litres

noncomputable def cocktail_cost_per_litre : ℝ := 
  cocktail_total_cost / cocktail_total_volume

theorem cocktail_cost_per_litre_is_accurate : 
  abs (cocktail_cost_per_litre - 1399.99) < 0.01 := by
  sorry

end NUMINAMATH_GPT_cocktail_cost_per_litre_is_accurate_l355_35569


namespace NUMINAMATH_GPT_michael_exceeds_suresh_l355_35556

theorem michael_exceeds_suresh (P M S : ℝ) 
  (h_total : P + M + S = 2400)
  (h_p_m_ratio : P / 5 = M / 7)
  (h_m_s_ratio : M / 3 = S / 2) : M - S = 336 :=
by
  sorry

end NUMINAMATH_GPT_michael_exceeds_suresh_l355_35556


namespace NUMINAMATH_GPT_number_of_birds_is_20_l355_35589

-- Define the given conditions
def distance_jim_disney : ℕ := 50
def distance_disney_london : ℕ := 60
def total_travel_distance : ℕ := 2200

-- Define the number of birds
def num_birds (B : ℕ) : Prop :=
  (distance_jim_disney + distance_disney_london) * B = total_travel_distance

-- The theorem stating the number of birds
theorem number_of_birds_is_20 : num_birds 20 :=
by
  unfold num_birds
  sorry

end NUMINAMATH_GPT_number_of_birds_is_20_l355_35589


namespace NUMINAMATH_GPT_mean_of_first_set_is_67_l355_35562

theorem mean_of_first_set_is_67 (x : ℝ) 
  (h : (50 + 62 + 97 + 124 + x) / 5 = 75.6) : 
  (28 + x + 70 + 88 + 104) / 5 = 67 := 
by
  sorry

end NUMINAMATH_GPT_mean_of_first_set_is_67_l355_35562


namespace NUMINAMATH_GPT_arithmetic_sequence_ratio_l355_35515

theorem arithmetic_sequence_ratio
  (x y a1 a2 a3 b1 b2 b3 b4 : ℝ)
  (h1 : x ≠ y)
  (h2 : a1 = x + (1 * (a2 - a1)))
  (h3 : a2 = x + (2 * (a2 - a1)))
  (h4 : a3 = x + (3 * (a2 - a1)))
  (h5 : y = x + (4 * (a2 - a1)))
  (h6 : x = x)
  (h7 : b2 = x + (1 * (b3 - x)))
  (h8 : b3 = x + (2 * (b3 - x)))
  (h9 : y = x + (3 * (b3 - x)))
  (h10 : b4 = x + (4 * (b3 - x))) :
  (b4 - b3) / (a2 - a1) = 8 / 3 := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_ratio_l355_35515


namespace NUMINAMATH_GPT_largest_c_value_l355_35526

theorem largest_c_value (c : ℝ) :
  (∃ x : ℝ, x^2 + 5 * x + c = -3) → c ≤ 13 / 4 :=
sorry

end NUMINAMATH_GPT_largest_c_value_l355_35526


namespace NUMINAMATH_GPT_family_members_l355_35511

theorem family_members (N : ℕ) (income : ℕ → ℕ) (average_income : ℕ) :
  average_income = 10000 ∧
  income 0 = 8000 ∧
  income 1 = 15000 ∧
  income 2 = 6000 ∧
  income 3 = 11000 ∧
  (income 0 + income 1 + income 2 + income 3) = 4 * average_income →
  N = 4 :=
by {
  sorry
}

end NUMINAMATH_GPT_family_members_l355_35511


namespace NUMINAMATH_GPT_min_stamps_value_l355_35575

theorem min_stamps_value (x y : ℕ) (hx : 5 * x + 7 * y = 74) : x + y = 12 :=
by
  sorry

end NUMINAMATH_GPT_min_stamps_value_l355_35575


namespace NUMINAMATH_GPT_find_f_at_six_l355_35584

theorem find_f_at_six (f : ℝ → ℝ) (h : ∀ x : ℝ, f (4 * x - 2) = x^2 - x + 2) : f 6 = 3.75 :=
by
  sorry

end NUMINAMATH_GPT_find_f_at_six_l355_35584


namespace NUMINAMATH_GPT_calc_exponent_result_l355_35572

theorem calc_exponent_result (m : ℝ) : (2 * m^2)^3 = 8 * m^6 := 
by
  sorry

end NUMINAMATH_GPT_calc_exponent_result_l355_35572


namespace NUMINAMATH_GPT_side_length_of_square_l355_35557

theorem side_length_of_square (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C]
  (a b c : ℝ) (h_leg1 : a = 12) (h_leg2 : b = 9) (h_right : c^2 = a^2 + b^2) :
  ∃ s : ℝ, s = 45/8 :=
by 
  -- Given the right triangle with legs 12 cm and 9 cm, the length of the side of the square is 45/8 cm
  let s := 45/8
  use s
  sorry

end NUMINAMATH_GPT_side_length_of_square_l355_35557


namespace NUMINAMATH_GPT_trajectory_moving_point_l355_35573

theorem trajectory_moving_point (x y : ℝ) (h₁ : x ≠ 1) (h₂ : x ≠ -1) :
  (y / (x + 1)) * (y / (x - 1)) = -1 ↔ x^2 + y^2 = 1 := by
  sorry

end NUMINAMATH_GPT_trajectory_moving_point_l355_35573


namespace NUMINAMATH_GPT_distribute_stickers_l355_35525

-- Definitions based on conditions
def stickers : ℕ := 10
def sheets : ℕ := 5

-- Theorem stating the equivalence of distributing the stickers onto sheets
theorem distribute_stickers :
  (Nat.choose (stickers + sheets - 1) (sheets - 1)) = 1001 :=
by 
  -- Here is where the proof would go, but we skip it with sorry for the purpose of this task
  sorry

end NUMINAMATH_GPT_distribute_stickers_l355_35525


namespace NUMINAMATH_GPT_length_60_more_than_breadth_l355_35597

noncomputable def length_more_than_breadth (cost_per_meter : ℝ) (total_cost : ℝ) (length : ℝ) : Prop :=
  ∃ (breadth : ℝ) (x : ℝ), 
    length = breadth + x ∧
    2 * length + 2 * breadth = total_cost / cost_per_meter ∧
    x = length - breadth ∧
    x = 60

theorem length_60_more_than_breadth : length_more_than_breadth 26.5 5300 80 :=
by
  sorry

end NUMINAMATH_GPT_length_60_more_than_breadth_l355_35597


namespace NUMINAMATH_GPT_distance_to_nearest_edge_of_picture_l355_35532

def wall_width : ℕ := 26
def picture_width : ℕ := 4
def distance_from_end (wall picture : ℕ) : ℕ := (wall - picture) / 2

theorem distance_to_nearest_edge_of_picture :
  distance_from_end wall_width picture_width = 11 :=
sorry

end NUMINAMATH_GPT_distance_to_nearest_edge_of_picture_l355_35532


namespace NUMINAMATH_GPT_angle_x_value_l355_35536

theorem angle_x_value 
  (AB CD : Prop) -- AB and CD are straight lines
  (angle_AXB angle_AXZ angle_BXY angle_CYX : ℝ) -- Given angles in the problem
  (h1 : AB) (h2 : CD)
  (h3 : angle_AXB = 180)
  (h4 : angle_AXZ = 60)
  (h5 : angle_BXY = 50)
  (h6 : angle_CYX = 120) : 
  ∃ x : ℝ, x = 50 := by
sorry

end NUMINAMATH_GPT_angle_x_value_l355_35536


namespace NUMINAMATH_GPT_compare_subtract_one_l355_35540

theorem compare_subtract_one (a b : ℝ) (h : a < b) : a - 1 < b - 1 :=
sorry

end NUMINAMATH_GPT_compare_subtract_one_l355_35540


namespace NUMINAMATH_GPT_magnitude_difference_l355_35588

noncomputable def vector_a : ℝ × ℝ := (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180))
noncomputable def vector_b : ℝ × ℝ := (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))

theorem magnitude_difference (a b : ℝ × ℝ) 
  (ha : a = (Real.cos (10 * Real.pi / 180), Real.sin (10 * Real.pi / 180)))
  (hb : b = (Real.cos (70 * Real.pi / 180), Real.sin (70 * Real.pi / 180))) :
  (Real.sqrt ((a.1 - 2 * b.1)^2 + (a.2 - 2 * b.2)^2)) = Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_difference_l355_35588


namespace NUMINAMATH_GPT_impossibility_of_equal_sum_selection_l355_35520

theorem impossibility_of_equal_sum_selection :
  ¬ ∃ (selected non_selected : Fin 10 → ℕ),
    (∀ i, selected i = 1 ∨ selected i = 36 ∨ selected i = 2 ∨ selected i = 35 ∨ 
              selected i = 3 ∨ selected i = 34 ∨ selected i = 4 ∨ selected i = 33 ∨ 
              selected i = 5 ∨ selected i = 32 ∨ selected i = 6 ∨ selected i = 31 ∨ 
              selected i = 7 ∨ selected i = 30 ∨ selected i = 8 ∨ selected i = 29 ∨ 
              selected i = 9 ∨ selected i = 28 ∨ selected i = 10 ∨ selected i = 27) ∧ 
    (∀ i, non_selected i = 1 ∨ non_selected i = 36 ∨ non_selected i = 2 ∨ non_selected i = 35 ∨ 
              non_selected i = 3 ∨ non_selected i = 34 ∨ non_selected i = 4 ∨ non_selected i = 33 ∨ 
              non_selected i = 5 ∨ non_selected i = 32 ∨ non_selected i = 6 ∨ non_selected i = 31 ∨ 
              non_selected i = 7 ∨ non_selected i = 30 ∨ non_selected i = 8 ∨ non_selected i = 29 ∨ 
              non_selected i = 9 ∨ non_selected i = 28 ∨ non_selected i = 10 ∨ non_selected i = 27) ∧ 
    (selected ≠ non_selected) ∧ 
    (Finset.univ.sum selected = Finset.univ.sum non_selected) :=
sorry

end NUMINAMATH_GPT_impossibility_of_equal_sum_selection_l355_35520


namespace NUMINAMATH_GPT_number_that_multiplies_x_l355_35534

variables (n x y : ℝ)

theorem number_that_multiplies_x :
  n * x = 3 * y → 
  x * y ≠ 0 → 
  (1 / 5 * x) / (1 / 6 * y) = 0.72 →
  n = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_number_that_multiplies_x_l355_35534


namespace NUMINAMATH_GPT_factor_theorem_for_Q_l355_35500

variable (d : ℝ) -- d is a real number

def Q (x : ℝ) : ℝ := x^3 + 3 * x^2 + d * x + 20

theorem factor_theorem_for_Q :
  (x : ℝ) → (Q x = 0) → (x = 4) → d = -33 :=
by
  intro x Q4 hx
  sorry

end NUMINAMATH_GPT_factor_theorem_for_Q_l355_35500


namespace NUMINAMATH_GPT_max_difference_in_volume_l355_35567

noncomputable def computed_volume (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def max_possible_volume (length width height : ℕ) (error : ℕ) : ℕ :=
  (length + error) * (width + error) * (height + error)

theorem max_difference_in_volume :
  ∀ (length width height error : ℕ), length = 150 → width = 150 → height = 225 → error = 1 → 
  max_possible_volume length width height error - computed_volume length width height = 90726 :=
by
  intros length width height error h_length h_width h_height h_error
  rw [h_length, h_width, h_height, h_error]
  simp only [computed_volume, max_possible_volume]
  -- Intermediate calculations
  sorry

end NUMINAMATH_GPT_max_difference_in_volume_l355_35567


namespace NUMINAMATH_GPT_relationship_between_A_and_B_l355_35564

noncomputable def f (x : ℝ) : ℝ := x^2

def A : Set ℝ := {x | f x = x}

def B : Set ℝ := {x | f (f x) = x}

theorem relationship_between_A_and_B : A ∩ B = A :=
by sorry

end NUMINAMATH_GPT_relationship_between_A_and_B_l355_35564


namespace NUMINAMATH_GPT_smallest_n_for_divisibility_property_l355_35543

theorem smallest_n_for_divisibility_property :
  ∃ n : ℕ, 0 < n ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → n^2 + n % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ n ∧ n^2 + n % k ≠ 0) ∧ 
  ∀ m : ℕ, 0 < m ∧ m < n → ¬ ((∀ k : ℕ, 1 ≤ k ∧ k ≤ m → m^2 + m % k = 0) ∧ 
  (∃ k : ℕ, 1 ≤ k ∧ k ≤ m ∧ m^2 + m % k ≠ 0)) := sorry

end NUMINAMATH_GPT_smallest_n_for_divisibility_property_l355_35543


namespace NUMINAMATH_GPT_james_eats_three_l355_35578

variables {p : ℕ} {f : ℕ} {j : ℕ}

-- The initial number of pizza slices
def initial_slices : ℕ := 8

-- The number of slices his friend eats
def friend_slices : ℕ := 2

-- The number of slices left after his friend eats
def remaining_slices : ℕ := initial_slices - friend_slices

-- The number of slices James eats
def james_slices : ℕ := remaining_slices / 2

-- The theorem to prove James eats 3 slices
theorem james_eats_three : james_slices = 3 :=
by
  sorry

end NUMINAMATH_GPT_james_eats_three_l355_35578


namespace NUMINAMATH_GPT_remainder_when_13_plus_x_divided_by_26_l355_35521

theorem remainder_when_13_plus_x_divided_by_26 (x : ℕ) (h1 : 9 * x % 26 = 1) : (13 + x) % 26 = 16 := 
by sorry

end NUMINAMATH_GPT_remainder_when_13_plus_x_divided_by_26_l355_35521


namespace NUMINAMATH_GPT_equation_of_circle_center_0_4_passing_through_3_0_l355_35509

noncomputable def circle_radius (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1) ^ 2 + (y2 - y1) ^ 2)

theorem equation_of_circle_center_0_4_passing_through_3_0 :
  ∃ (r : ℝ), (r = circle_radius 0 4 3 0) ∧ (r = 5) ∧ ((x y : ℝ) → ((x - 0) ^ 2 + (y - 4) ^ 2 = r ^ 2) ↔ (x ^ 2 + (y - 4) ^ 2 = 25)) :=
by
  sorry

end NUMINAMATH_GPT_equation_of_circle_center_0_4_passing_through_3_0_l355_35509


namespace NUMINAMATH_GPT_domain_f_log2_x_to_domain_f_x_l355_35530

variable {f : ℝ → ℝ}

-- Condition: The domain of y = f(log₂ x) is [1/2, 4]
def domain_f_log2_x : Set ℝ := Set.Icc (1 / 2) 4

-- Proof statement
theorem domain_f_log2_x_to_domain_f_x
  (h : ∀ x, x ∈ domain_f_log2_x → f (Real.log x / Real.log 2) = f x) :
  Set.Icc (-1) 2 = {x : ℝ | ∃ y ∈ domain_f_log2_x, Real.log y / Real.log 2 = x} :=
by
  sorry

end NUMINAMATH_GPT_domain_f_log2_x_to_domain_f_x_l355_35530


namespace NUMINAMATH_GPT_sector_radius_l355_35537

theorem sector_radius (α S r : ℝ) (h1 : α = 3/4 * Real.pi) (h2 : S = 3/2 * Real.pi) :
  S = 1/2 * r^2 * α → r = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_radius_l355_35537


namespace NUMINAMATH_GPT_topic_preference_order_l355_35519

noncomputable def astronomy_fraction := (8 : ℚ) / 21
noncomputable def botany_fraction := (5 : ℚ) / 14
noncomputable def chemistry_fraction := (9 : ℚ) / 28

theorem topic_preference_order :
  (astronomy_fraction > botany_fraction) ∧ (botany_fraction > chemistry_fraction) :=
by
  sorry

end NUMINAMATH_GPT_topic_preference_order_l355_35519


namespace NUMINAMATH_GPT_perpendicular_lines_condition_l355_35594

theorem perpendicular_lines_condition (a : ℝ) : 
  (∀ x y : ℝ, x + y = 0 ∧ x - ay = 0 → x = 0) ↔ (a = 1) := 
sorry

end NUMINAMATH_GPT_perpendicular_lines_condition_l355_35594


namespace NUMINAMATH_GPT_magnitude_of_complex_l355_35592

noncomputable def z : ℂ := (2 / 3 : ℝ) - (4 / 5 : ℝ) * Complex.I

theorem magnitude_of_complex :
  Complex.abs z = (2 * Real.sqrt 61) / 15 :=
by
  sorry

end NUMINAMATH_GPT_magnitude_of_complex_l355_35592


namespace NUMINAMATH_GPT_mark_peters_pond_depth_l355_35523

theorem mark_peters_pond_depth :
  let mark_depth := 19
  let peter_depth := 5
  let three_times_peter_depth := 3 * peter_depth
  mark_depth - three_times_peter_depth = 4 :=
by
  sorry

end NUMINAMATH_GPT_mark_peters_pond_depth_l355_35523


namespace NUMINAMATH_GPT_fraction_simplification_l355_35501

theorem fraction_simplification : (145^2 - 121^2) / 24 = 266 := by
  sorry

end NUMINAMATH_GPT_fraction_simplification_l355_35501


namespace NUMINAMATH_GPT_distance_between_adjacent_parallel_lines_l355_35577

noncomputable def distance_between_lines (r d : ℝ) : ℝ :=
  (49 * r^2 - 49 * 600.25 - (49 / 4) * d^2) / (1 - 49 / 4)

theorem distance_between_adjacent_parallel_lines :
  ∃ d : ℝ, ∀ (r : ℝ), 
    (r^2 = 506.25 + (1 / 4) * d^2 ∧ r^2 = 600.25 + (49 / 4) * d^2) →
    d = 2.8 :=
sorry

end NUMINAMATH_GPT_distance_between_adjacent_parallel_lines_l355_35577


namespace NUMINAMATH_GPT_contrapositive_inequality_l355_35579

theorem contrapositive_inequality (x : ℝ) :
  ((x + 2) * (x - 3) > 0) → (x < -2 ∨ x > 0) :=
by
  sorry

end NUMINAMATH_GPT_contrapositive_inequality_l355_35579


namespace NUMINAMATH_GPT_tina_money_left_l355_35586

theorem tina_money_left :
  let june_savings := 27
  let july_savings := 14
  let august_savings := 21
  let books_spending := 5
  let shoes_spending := 17
  june_savings + july_savings + august_savings - (books_spending + shoes_spending) = 40 :=
by
  sorry

end NUMINAMATH_GPT_tina_money_left_l355_35586


namespace NUMINAMATH_GPT_probability_colors_match_l355_35524

noncomputable def prob_abe_shows_blue : ℚ := 2 / 4
noncomputable def prob_bob_shows_blue : ℚ := 3 / 6
noncomputable def prob_abe_shows_green : ℚ := 2 / 4
noncomputable def prob_bob_shows_green : ℚ := 1 / 6

noncomputable def prob_same_color : ℚ :=
  (prob_abe_shows_blue * prob_bob_shows_blue) + (prob_abe_shows_green * prob_bob_shows_green)

theorem probability_colors_match : prob_same_color = 1 / 3 :=
by
  sorry

end NUMINAMATH_GPT_probability_colors_match_l355_35524


namespace NUMINAMATH_GPT_greatest_multiple_of_5_and_6_lt_1000_l355_35514

theorem greatest_multiple_of_5_and_6_lt_1000 : 
  ∃ n, n % 5 = 0 ∧ n % 6 = 0 ∧ n < 1000 ∧ (∀ m, m % 5 = 0 ∧ m % 6 = 0 ∧ m < 1000 → m ≤ n) :=
  sorry

end NUMINAMATH_GPT_greatest_multiple_of_5_and_6_lt_1000_l355_35514


namespace NUMINAMATH_GPT_factorize_expression_l355_35599

theorem factorize_expression : ∀ x : ℝ, 2 * x^2 - 4 * x = 2 * x * (x - 2) :=
by
  intro x
  sorry

end NUMINAMATH_GPT_factorize_expression_l355_35599


namespace NUMINAMATH_GPT_vector_AD_length_l355_35593

open Real EuclideanSpace

noncomputable def problem_statement
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC : ℝ) (AD : ℝ) : Prop :=
  angle_mn = π / 6 ∧ 
  norm_m = sqrt 3 ∧ 
  norm_n = 2 ∧ 
  AB = 2 * m + 2 * n ∧ 
  AC = 2 * m - 6 * n ∧ 
  AD = 2 * m - 2 * n ∧
  sqrt ((AD) * (AD)) = 2

theorem vector_AD_length 
  (m n : ℝ) (angle_mn : ℝ) (norm_m : ℝ) (norm_n : ℝ) (AB AC AD : ℝ) :
  problem_statement m n angle_mn norm_m norm_n AB AC AD :=
by
  unfold problem_statement
  sorry

end NUMINAMATH_GPT_vector_AD_length_l355_35593


namespace NUMINAMATH_GPT_percent_neither_filler_nor_cheese_l355_35545

-- Define the given conditions as constants
def total_weight : ℕ := 200
def filler_weight : ℕ := 40
def cheese_weight : ℕ := 30

-- Definition of the remaining weight that is neither filler nor cheese
def neither_weight : ℕ := total_weight - filler_weight - cheese_weight

-- Calculation of the percentage of the burger that is neither filler nor cheese
def percentage_neither : ℚ := (neither_weight : ℚ) / (total_weight : ℚ) * 100

-- The theorem to prove
theorem percent_neither_filler_nor_cheese :
  percentage_neither = 65 := by
  sorry

end NUMINAMATH_GPT_percent_neither_filler_nor_cheese_l355_35545


namespace NUMINAMATH_GPT_chuck_total_playable_area_l355_35554

noncomputable def chuck_roaming_area (shed_length shed_width leash_length : ℝ) : ℝ :=
  let larger_arc_area := (3 / 4) * Real.pi * leash_length ^ 2
  let additional_sector_area := (1 / 4) * Real.pi * (leash_length - shed_length) ^ 2
  larger_arc_area + additional_sector_area

theorem chuck_total_playable_area :
  chuck_roaming_area 3 4 5 = 19 * Real.pi :=
  by
  sorry

end NUMINAMATH_GPT_chuck_total_playable_area_l355_35554
