import Mathlib

namespace binom_18_4_eq_3060_l205_205802

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l205_205802


namespace tangent_circle_radius_l205_205985

theorem tangent_circle_radius (r1 r2 d : ℝ) (h1 : r2 = 2) (h2 : d = 5) (tangent : abs (r1 - r2) = d ∨ r1 + r2 = d) :
  r1 = 3 ∨ r1 = 7 :=
by
  sorry

end tangent_circle_radius_l205_205985


namespace richmond_more_than_victoria_l205_205963

-- Defining the population of Beacon
def beacon_people : ℕ := 500

-- Defining the population of Victoria based on Beacon's population
def victoria_people : ℕ := 4 * beacon_people

-- Defining the population of Richmond
def richmond_people : ℕ := 3000

-- The proof problem: calculating the difference
theorem richmond_more_than_victoria : richmond_people - victoria_people = 1000 := by
  -- The statement of the theorem
  sorry

end richmond_more_than_victoria_l205_205963


namespace find_some_multiplier_l205_205063

theorem find_some_multiplier (m : ℕ) :
  (422 + 404)^2 - (m * 422 * 404) = 324 ↔ m = 4 :=
by
  sorry

end find_some_multiplier_l205_205063


namespace domain_of_sqrt_div_sqrt_l205_205904

theorem domain_of_sqrt_div_sqrt (x : ℝ) : (3 ≤ x ∧ x < 7) ↔ (∃ f, f = (λ x, (√(x - 3)) / (√(7 - x))) ∧ 3 ≤ x ∧ x < 7) := 
by 
  sorry

end domain_of_sqrt_div_sqrt_l205_205904


namespace largest_prime_number_largest_composite_number_l205_205631

-- Definitions of prime and composite
def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := n > 1 ∧ ∃ m, m ∣ n ∧ m ≠ 1 ∧ m ≠ n

-- Largest prime and composite numbers less than 20
def largest_prime_less_than_20 := 19
def largest_composite_less_than_20 := 18

theorem largest_prime_number : 
  largest_prime_less_than_20 = 19 ∧ is_prime 19 ∧ 
  (∀ n : ℕ, n < 20 → is_prime n → n < 19) := 
by sorry

theorem largest_composite_number : 
  largest_composite_less_than_20 = 18 ∧ is_composite 18 ∧ 
  (∀ n : ℕ, n < 20 → is_composite n → n < 18) := 
by sorry

end largest_prime_number_largest_composite_number_l205_205631


namespace chocolate_flavor_sales_l205_205778

-- Define the total number of cups sold
def total_cups : ℕ := 50

-- Define the fraction of winter melon flavor sales
def winter_melon_fraction : ℚ := 2 / 5

-- Define the fraction of Okinawa flavor sales
def okinawa_fraction : ℚ := 3 / 10

-- Proof statement
theorem chocolate_flavor_sales : 
  (total_cups - (winter_melon_fraction * total_cups).toInt - (okinawa_fraction * total_cups).toInt) = 15 := 
  by 
  sorry

end chocolate_flavor_sales_l205_205778


namespace matrix_eigenvalue_neg7_l205_205528

theorem matrix_eigenvalue_neg7 (M : Matrix (Fin 2) (Fin 2) ℝ) :
  (∀ (v : Fin 2 → ℝ), M.mulVec v = -7 • v) →
  M = !![-7, 0; 0, -7] :=
by
  intro h
  -- proof goes here
  sorry

end matrix_eigenvalue_neg7_l205_205528


namespace inclination_angle_range_l205_205931

theorem inclination_angle_range :
  (∃ l : ℝ → ℝ, (∃ θ, ∀ x, l x = (Real.tan θ) * (x - 3)) ∧
  (∃ x y, (x - 1)^2 + y^2 = 1 ∧ y = l x)) →
   ∃ θ : ℝ, θ ∈ Set.Icc 0 (Real.pi / 6) ∪ Set.Icc (5 * Real.pi / 6) Real.pi :=
begin
  sorry
end

end inclination_angle_range_l205_205931


namespace mary_age_proof_l205_205097

theorem mary_age_proof (suzy_age_now : ℕ) (H1 : suzy_age_now = 20) (H2 : ∀ (years : ℕ), years = 4 → (suzy_age_now + years) = 2 * (mary_age + years)) : mary_age = 8 :=
by
  sorry

end mary_age_proof_l205_205097


namespace joann_lollipop_wednesday_l205_205413

variable (a : ℕ) (d : ℕ) (n : ℕ)

def joann_lollipop_count (a d n : ℕ) : ℕ :=
  a + d * n

theorem joann_lollipop_wednesday :
  let a := 4
  let d := 3
  let total_days := 7
  let target_total := 133
  ∀ (monday tuesday wednesday thursday friday saturday sunday : ℕ),
    monday = a ∧
    tuesday = a + d ∧
    wednesday = a + 2 * d ∧
    thursday = a + 3 * d ∧
    friday = a + 4 * d ∧
    saturday = a + 5 * d ∧
    sunday = a + 6 * d ∧
    (monday + tuesday + wednesday + thursday + friday + saturday + sunday = target_total) →
    wednesday = 10 :=
by
  sorry

end joann_lollipop_wednesday_l205_205413


namespace contribution_amount_l205_205131

theorem contribution_amount (x : ℝ) (S : ℝ) :
  (S = 10 * x) ∧ (S = 15 * (x - 100)) → x = 300 :=
by
  sorry

end contribution_amount_l205_205131


namespace find_k_correct_l205_205337

noncomputable def find_k (x : ℝ) (k : ℝ) : Prop :=
  ∑ (x : ℝ) in { x | x ≥ 0 ∧ (sqrt x * (x + 12) = 17 * x - k) }, x = 256 → k = 90

theorem find_k_correct : ∀ (k : ℝ), find_k x k := 
by
  sorry

end find_k_correct_l205_205337


namespace no_intersection_points_l205_205673

-- Define the absolute value functions
def f1 (x : ℝ) : ℝ := abs (3 * x + 6)
def f2 (x : ℝ) : ℝ := -abs (4 * x - 3)

-- State the theorem
theorem no_intersection_points : ∀ x y : ℝ, f1 x = y ∧ f2 x = y → false := by
  sorry

end no_intersection_points_l205_205673


namespace avg_children_in_families_with_children_l205_205879

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205879


namespace average_children_in_families_with_children_l205_205858

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205858


namespace integer_satisfying_values_l205_205018

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l205_205018


namespace num_int_values_x_l205_205006

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l205_205006


namespace expansion_terms_count_l205_205924

-- Define the number of terms in the first polynomial
def first_polynomial_terms : ℕ := 3

-- Define the number of terms in the second polynomial
def second_polynomial_terms : ℕ := 4

-- Prove that the number of terms in the expansion is 12
theorem expansion_terms_count : first_polynomial_terms * second_polynomial_terms = 12 :=
by
  sorry

end expansion_terms_count_l205_205924


namespace average_children_families_with_children_is_3_point_8_l205_205851

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205851


namespace negation_of_quadratic_inequality_l205_205134

-- Definitions
def quadratic_inequality (a : ℝ) : Prop := ∃ x : ℝ, x * x + a * x + 1 < 0

-- Theorem statement
theorem negation_of_quadratic_inequality (a : ℝ) : ¬ (quadratic_inequality a) ↔ ∀ x : ℝ, x * x + a * x + 1 ≥ 0 :=
by sorry

end negation_of_quadratic_inequality_l205_205134


namespace rationalize_denominator_l205_205586

theorem rationalize_denominator : 
  let x := (1 : ℝ)
  let y := (3 : ℝ)
  let z := real.cbrt 3
  let w := real.cbrt 27
  (w = 3) →
  x / (z + w) = real.cbrt (9) / (3 * (real.cbrt (9) + 1)) := 
by
  intros _ h
  rw [h]
  sorry

end rationalize_denominator_l205_205586


namespace sqrt_x_minus_1_meaningful_l205_205749

theorem sqrt_x_minus_1_meaningful (x : ℝ) : (x - 1 ≥ 0) ↔ (x ≥ 1) := by
  sorry

end sqrt_x_minus_1_meaningful_l205_205749


namespace division_by_fraction_l205_205311

theorem division_by_fraction (a b : ℝ) (hb : b ≠ 0) : a / (1 / b) = a * b :=
by {
  sorry
}

example : 12 / (1 / 6) = 72 :=
by {
  exact division_by_fraction 12 6 (by norm_num),
}

end division_by_fraction_l205_205311


namespace compare_abc_l205_205911

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l205_205911


namespace denomination_of_remaining_coins_l205_205604

/-
There are 324 coins total.
The total value of the coins is Rs. 70.
There are 220 coins of 20 paise each.
Find the denomination of the remaining coins.
-/

def total_coins := 324
def total_value := 7000 -- Rs. 70 converted into paise
def num_20_paise_coins := 220
def value_20_paise_coin := 20
  
theorem denomination_of_remaining_coins :
  let total_remaining_value := total_value - (num_20_paise_coins * value_20_paise_coin)
  let num_remaining_coins := total_coins - num_20_paise_coins
  num_remaining_coins > 0 →
  total_remaining_value / num_remaining_coins = 25 :=
by
  sorry

end denomination_of_remaining_coins_l205_205604


namespace johnny_distance_walked_l205_205289

theorem johnny_distance_walked
  (dist_q_to_y : ℕ) (matthew_rate : ℕ) (johnny_rate : ℕ) (time_diff : ℕ) (johnny_walked : ℕ):
  dist_q_to_y = 45 →
  matthew_rate = 3 →
  johnny_rate = 4 →
  time_diff = 1 →
  (∃ t: ℕ, johnny_walked = johnny_rate * t 
            ∧ dist_q_to_y = matthew_rate * (t + time_diff) + johnny_walked) →
  johnny_walked = 24 := by
  sorry

end johnny_distance_walked_l205_205289


namespace factorial_product_square_l205_205788

theorem factorial_product_square (n : ℕ) (m : ℕ) (h₁ : n = 5) (h₂ : m = 4) :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 :=
by
  have f5 : Nat.factorial 5 = 120 := by norm_num
  have f4 : Nat.factorial 4 = 24 := by norm_num
  rw [Nat.factorial_eq_factorial h₁, Nat.factorial_eq_factorial h₂, f5, f4]
  norm_num
  simp
  sorry

end factorial_product_square_l205_205788


namespace bridge_length_l205_205723

/-- The length of the bridge that a train 110 meters long and traveling at 45 km/hr can cross in 30 seconds is 265 meters. -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (cross_time_sec : ℝ) (bridge_length : ℝ) :
  train_length = 110 ∧ train_speed_kmh = 45 ∧ cross_time_sec = 30 ∧ bridge_length = 265 → 
  (train_speed_kmh * (1000 / 3600) * cross_time_sec - train_length = bridge_length) :=
by
  sorry

end bridge_length_l205_205723


namespace factorial_product_square_l205_205787

theorem factorial_product_square (n : ℕ) (m : ℕ) (h₁ : n = 5) (h₂ : m = 4) :
  (Real.sqrt (Nat.factorial 5 * Nat.factorial 4))^2 = 2880 :=
by
  have f5 : Nat.factorial 5 = 120 := by norm_num
  have f4 : Nat.factorial 4 = 24 := by norm_num
  rw [Nat.factorial_eq_factorial h₁, Nat.factorial_eq_factorial h₂, f5, f4]
  norm_num
  simp
  sorry

end factorial_product_square_l205_205787


namespace complex_expression_evaluation_l205_205182

theorem complex_expression_evaluation (i : ℂ) (h1 : i^(4 : ℤ) = 1) (h2 : i^(1 : ℤ) = i)
   (h3 : i^(2 : ℤ) = -1) (h4 : i^(3 : ℤ) = -i) (h5 : i^(0 : ℤ) = 1) : 
   i^(245 : ℤ) + i^(246 : ℤ) + i^(247 : ℤ) + i^(248 : ℤ) + i^(249 : ℤ) = i :=
by
  sorry

end complex_expression_evaluation_l205_205182


namespace triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l205_205404

-- Define the triangle type
structure Triangle :=
(SideA : ℝ)
(SideB : ℝ)
(SideC : ℝ)
(AngleA : ℝ)
(AngleB : ℝ)
(AngleC : ℝ)
(h1 : SideA > 0)
(h2 : SideB > 0)
(h3 : SideC > 0)
(h4 : AngleA + AngleB + AngleC = 180)

-- Define what it means for two triangles to have three equal angles
def have_equal_angles (T1 T2 : Triangle) : Prop :=
(T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- Define what it means for two triangles to have two equal sides
def have_two_equal_sides (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB) ∨
(T1.SideA = T2.SideA ∧ T1.SideC = T2.SideC) ∨
(T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC)

-- Define what it means for two triangles to be congruent
def congruent (T1 T2 : Triangle) : Prop :=
(T1.SideA = T2.SideA ∧ T1.SideB = T2.SideB ∧ T1.SideC = T2.SideC ∧
 T1.AngleA = T2.AngleA ∧ T1.AngleB = T2.AngleB ∧ T1.AngleC = T2.AngleC)

-- The final theorem
theorem triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent 
  (T1 T2 : Triangle) 
  (h_angles : have_equal_angles T1 T2)
  (h_sides : have_two_equal_sides T1 T2) : ¬ congruent T1 T2 :=
sorry

end triangles_with_equal_angles_and_two_equal_sides_not_necessarily_congruent_l205_205404


namespace ratio_female_to_male_l205_205553

theorem ratio_female_to_male (total_members : ℕ) (female_members : ℕ) (male_members : ℕ) 
  (h1 : total_members = 18) (h2 : female_members = 12) (h3 : male_members = total_members - female_members) : 
  (female_members : ℚ) / (male_members : ℚ) = 2 := 
by 
  sorry

end ratio_female_to_male_l205_205553


namespace simplify_and_rationalize_l205_205590

noncomputable def expression := 
  (Real.sqrt 8 / Real.sqrt 3) * 
  (Real.sqrt 25 / Real.sqrt 30) * 
  (Real.sqrt 16 / Real.sqrt 21)

theorem simplify_and_rationalize :
  expression = 4 * Real.sqrt 14 / 63 :=
by
  sorry

end simplify_and_rationalize_l205_205590


namespace inequality_comparison_l205_205913

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end inequality_comparison_l205_205913


namespace reflection_y_axis_matrix_l205_205341

theorem reflection_y_axis_matrix :
  ∀ (A : Matrix (Fin 2) (Fin 2) ℝ), 
    (A ⬝ ![![1, 0], ![0, 1]] = ![![1, 0], ![0, 1]]) →
    (A ⬝ ![1, 0] = ![-1, 0]) →
    (A ⬝ ![0, 1] = ![0, 1]) →
    A = ![![ -1, 0], ![0, 1]] :=
by
  intros A hA hA1 hA2
  sorry

end reflection_y_axis_matrix_l205_205341


namespace avg_children_in_families_with_children_l205_205836

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205836


namespace cyclic_ABCD_l205_205575

variable {Point : Type}
variable {Angle LineCircle : Type → Type}
variable {cyclicQuadrilateral : List (Point) → Prop}
variable {convexQuadrilateral : List (Point) → Prop}
variable {lineSegment : Point → Point → LineCircle Point}
variable {onSegment : Point → LineCircle Point → Prop}
variable {angle : Point → Point → Point → Angle Point}

theorem cyclic_ABCD (A B C D P Q E : Point)
  (h1 : convexQuadrilateral [A, B, C, D])
  (h2 : cyclicQuadrilateral [P, Q, D, A])
  (h3 : cyclicQuadrilateral [Q, P, B, C])
  (h4 : onSegment E (lineSegment P Q))
  (h5 : angle P A E = angle Q D E)
  (h6 : angle P B E = angle Q C E) :
  cyclicQuadrilateral [A, B, C, D] :=
  sorry

end cyclic_ABCD_l205_205575


namespace solution_set_range_l205_205967

theorem solution_set_range (x : ℝ) : 
  (2 * |x - 10| + 3 * |x - 20| ≤ 35) ↔ (9 ≤ x ∧ x ≤ 23) :=
sorry

end solution_set_range_l205_205967


namespace find_a_l205_205549

theorem find_a (x y a : ℝ) (h1 : 4 * x + y = 8) (h2 : 3 * x - 4 * y = 5) (h3 : a * x - 3 * y = 23) : 
  a = 12.141 :=
by
  sorry

end find_a_l205_205549


namespace average_height_40_girls_l205_205224

/-- Given conditions for a class of 50 students, where the average height of 40 girls is H,
    the average height of the remaining 10 girls is 167 cm, and the average height of the whole
    class is 168.6 cm, prove that the average height H of the 40 girls is 169 cm. -/
theorem average_height_40_girls (H : ℝ)
  (h1 : 0 < H)
  (h2 : (40 * H + 10 * 167) = 50 * 168.6) :
  H = 169 :=
by
  sorry

end average_height_40_girls_l205_205224


namespace root_quadratic_eq_k_value_l205_205761

theorem root_quadratic_eq_k_value (k : ℤ) :
  (∃ x : ℤ, x = 5 ∧ 2 * x ^ 2 + 3 * x - k = 0) → k = 65 :=
by
  sorry

end root_quadratic_eq_k_value_l205_205761


namespace spencer_total_jumps_l205_205474

noncomputable def jumps_per_minute : ℕ := 4
noncomputable def minutes_per_session : ℕ := 10
noncomputable def sessions_per_day : ℕ := 2
noncomputable def days : ℕ := 5

theorem spencer_total_jumps : 
  (jumps_per_minute * minutes_per_session) * (sessions_per_day * days) = 400 :=
by
  sorry

end spencer_total_jumps_l205_205474


namespace m_value_for_power_function_l205_205935

theorem m_value_for_power_function (m : ℝ) :
  (3 * m - 1 = 1) → (m = 2 / 3) :=
by
  sorry

end m_value_for_power_function_l205_205935


namespace fg_of_2_eq_81_l205_205086

def f (x : ℝ) : ℝ := x ^ 2
def g (x : ℝ) : ℝ := x ^ 2 + 2 * x + 1

theorem fg_of_2_eq_81 : f (g 2) = 81 := by
  sorry

end fg_of_2_eq_81_l205_205086


namespace rotation_matrix_150_degrees_l205_205369

theorem rotation_matrix_150_degrees : 
  let θ := Real.pi * (150 / 180)
  let c := Real.cos θ
  let s := Real.sin θ
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![
    ![c, -s],
    ![s, c]
  ] ∧ M = ![
    ![-(Real.sqrt 3 / 2), -(1 / 2)],
    ![(1 / 2), -(Real.sqrt 3 / 2)]
  ] :=
by
  let θ := Real.pi * (150 / 180)
  let c := Real.cos θ
  let s := Real.sin θ
  exists (![
    ![c, -s],
    ![s, c]
  ])
  split
  · -- Showing M is the rotation matrix form
    simp [c, s]
    sorry
  · -- showing that it matches the filled-in values
    simp
    sorry

end rotation_matrix_150_degrees_l205_205369


namespace different_values_of_t_l205_205724

-- Define the conditions on the numbers
variables (p q r s t : ℕ)

-- Define the constraints: p, q, r, s, and t are distinct single-digit numbers
def valid_single_digit (x : ℕ) := x > 0 ∧ x < 10
def distinct_single_digits (p q r s t : ℕ) := 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧
  q ≠ r ∧ q ≠ s ∧ q ≠ t ∧
  r ≠ s ∧ r ≠ t ∧
  s ≠ t

-- Define the relationships given in the problem
def conditions (p q r s t : ℕ) :=
  valid_single_digit p ∧
  valid_single_digit q ∧
  valid_single_digit r ∧
  valid_single_digit s ∧
  valid_single_digit t ∧
  distinct_single_digits p q r s t ∧
  p - q = r ∧
  r - s = t

-- Theorem to be proven
theorem different_values_of_t : 
  ∃! (count : ℕ), count = 6 ∧ (∃ p q r s t, conditions p q r s t) := 
sorry

end different_values_of_t_l205_205724


namespace pine_sample_count_l205_205988

variable (total_saplings : ℕ)
variable (pine_saplings : ℕ)
variable (sample_size : ℕ)

theorem pine_sample_count (h1 : total_saplings = 30000) (h2 : pine_saplings = 4000) (h3 : sample_size = 150) :
  pine_saplings * sample_size / total_saplings = 20 := 
sorry

end pine_sample_count_l205_205988


namespace M_inter_N_l205_205211

def M : Set ℝ := { x | -1 ≤ x ∧ x ≤ 1 }
noncomputable def N : Set ℝ := { x | ∃ y, y = Real.sqrt x + Real.log (1 - x) }

theorem M_inter_N : M ∩ N = {x | 0 ≤ x ∧ x < 1} := by
  sorry

end M_inter_N_l205_205211


namespace binary_addition_l205_205782

-- Define the binary numbers as natural numbers
def b1 : ℕ := 0b101  -- 101_2
def b2 : ℕ := 0b11   -- 11_2
def b3 : ℕ := 0b1100 -- 1100_2
def b4 : ℕ := 0b11101 -- 11101_2
def sum_b : ℕ := 0b110001 -- 110001_2

theorem binary_addition :
  b1 + b2 + b3 + b4 = sum_b := 
by
  sorry

end binary_addition_l205_205782


namespace parabola_equation_l205_205272

-- Define the conditions of the problem
def parabola_vertex := (0, 0)
def parabola_focus_x_axis := true
def line_eq (x y : ℝ) : Prop := x = y
def midpoint_of_AB (x1 y1 x2 y2 mx my: ℝ) : Prop := (mx, my) = ((x1 + x2) / 2, (y1 + y2) / 2)
def point_P := (1, 1)

theorem parabola_equation (A B : ℝ × ℝ) :
  (parabola_vertex = (0, 0)) →
  (parabola_focus_x_axis) →
  (line_eq A.1 A.2) →
  (line_eq B.1 B.2) →
  midpoint_of_AB A.1 A.2 B.1 B.2 point_P.1 point_P.2 →
  A = (0, 0) ∨ B = (0, 0) →
  B = A ∨ A = (0, 0) → B = (2, 2) →
  ∃ a, ∀ x y, y^2 = a * x → a = 2 :=
sorry

end parabola_equation_l205_205272


namespace candy_distribution_powers_of_two_l205_205052

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def children_receive_candies (f : ℕ → ℕ) (n : ℕ) : Prop :=
  ∀ x : ℕ, ∃ k : ℕ, (k * (k + 1) / 2) % n = x

theorem candy_distribution_powers_of_two (n : ℕ) (hn : is_power_of_two n) :
  children_receive_candies (λ x, x * (x + 1) / 2 % n) n :=
sorry

end candy_distribution_powers_of_two_l205_205052


namespace total_green_marbles_l205_205589

-- Conditions
def Sara_green_marbles : ℕ := 3
def Tom_green_marbles : ℕ := 4

-- Problem statement: proving the total number of green marbles
theorem total_green_marbles : Sara_green_marbles + Tom_green_marbles = 7 := by
  sorry

end total_green_marbles_l205_205589


namespace slope_range_l205_205542

theorem slope_range (a : ℝ) (ha : a ∈ Set.Icc (Real.pi / 4) (Real.pi / 2)) :
  ∃ k : ℝ, k = Real.tan a ∧ k ∈ Set.Ici 1 :=
by {
  sorry
}

end slope_range_l205_205542


namespace integer_satisfying_values_l205_205017

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l205_205017


namespace evaluate_product_l205_205055

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_product : 
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * (3 - w^6) *
  (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * (3 - w^11) * (3 - w^12) = 2657205 :=
by 
  sorry

end evaluate_product_l205_205055


namespace problem_R_l205_205572

noncomputable def R (g S h : ℝ) : ℝ := g * S + h

theorem problem_R {g h : ℝ} (h_h : h = 6 - 4 * g) :
  R g 14 h = 56 :=
by
  sorry

end problem_R_l205_205572


namespace _l205_205281

section BoxProblem

open Nat

def volume_box (l w h : ℕ) : ℕ := l * w * h
def volume_block (l w h : ℕ) : ℕ := l * w * h

def can_fit_blocks (box_l box_w box_h block_l block_w block_h n_blocks : ℕ) : Prop :=
  (volume_box box_l box_w box_h) = (n_blocks * volume_block block_l block_w block_h)

example : can_fit_blocks 4 3 3 3 2 1 6 :=
by
  -- calculation that proves the theorem goes here, but no need to provide proof steps
  sorry

end BoxProblem

end _l205_205281


namespace jim_makes_60_dollars_l205_205946

-- Definitions based on the problem conditions
def average_weight_per_rock : ℝ := 1.5
def price_per_pound : ℝ := 4
def number_of_rocks : ℕ := 10

-- Problem statement
theorem jim_makes_60_dollars :
  (average_weight_per_rock * number_of_rocks) * price_per_pound = 60 := by
  sorry

end jim_makes_60_dollars_l205_205946


namespace system_of_equations_solutions_l205_205592

theorem system_of_equations_solutions (x y z : ℝ) :
  (x^2 - y^2 + z = 27 / (x * y)) ∧ 
  (y^2 - z^2 + x = 27 / (y * z)) ∧ 
  (z^2 - x^2 + y = 27 / (z * x)) ↔ 
  (x = 3 ∧ y = 3 ∧ z = 3) ∨
  (x = -3 ∧ y = -3 ∧ z = 3) ∨
  (x = -3 ∧ y = 3 ∧ z = -3) ∨
  (x = 3 ∧ y = -3 ∧ z = -3) :=
by 
  sorry

end system_of_equations_solutions_l205_205592


namespace labourer_total_payment_l205_205624

/--
A labourer was engaged for 25 days on the condition that for every day he works, he will be paid Rs. 2 and for every day he is absent, he will be fined 50 p. He was absent for 5 days. Prove that the total amount he received in the end is Rs. 37.50.
-/
theorem labourer_total_payment :
  let total_days := 25
  let daily_wage := 2.0
  let absent_days := 5
  let fine_per_absent_day := 0.5
  let worked_days := total_days - absent_days
  let total_earnings := worked_days * daily_wage
  let total_fine := absent_days * fine_per_absent_day
  let total_received := total_earnings - total_fine
  total_received = 37.5 :=
by
  sorry

end labourer_total_payment_l205_205624


namespace find_time_l205_205919

variables (V V_0 S g C : ℝ) (t : ℝ)

-- Given conditions.
axiom eq1 : V = 2 * g * t + V_0
axiom eq2 : S = (1 / 3) * g * t^2 + V_0 * t + C * t^3

-- The statement to prove.
theorem find_time : t = (V - V_0) / (2 * g) :=
sorry

end find_time_l205_205919


namespace rationalize_denominator_proof_l205_205581

def rationalize_denominator (cbrt : ℝ → ℝ) (a : ℝ) :=
  cbrt a = a^(1/3)

theorem rationalize_denominator_proof : 
  (rationalize_denominator (λ x, x ^ (1/3)) 27) →
  (rationalize_denominator (λ x, x ^ (1/3)) 9) →
  (1 / (3 ^ (1 / 3) + 3) = 9 ^ (1 / 3) / (3 + 9 * 3 ^ (1 / 3))) :=
by
  sorry

end rationalize_denominator_proof_l205_205581


namespace arithmetic_sequence_common_difference_l205_205684

theorem arithmetic_sequence_common_difference
    (a : ℕ → ℝ)
    (h1 : a 2 + a 3 = 9)
    (h2 : a 4 + a 5 = 21)
    (h3 : ∀ n, a (n + 1) = a n + d) : d = 3 :=
        sorry

end arithmetic_sequence_common_difference_l205_205684


namespace isosceles_triangle_perimeter_l205_205383

def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ c = a

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def perimeter (a b c : ℝ) : ℝ :=
  a + b + c

theorem isosceles_triangle_perimeter (a b c : ℝ) (h1 : is_isosceles a b c) (h2 : is_triangle a b c) (h3 : a = 4 ∨ a = 9) (h4 : b = 4 ∨ b = 9) :
  perimeter a b c = 22 :=
  sorry

end isosceles_triangle_perimeter_l205_205383


namespace mean_of_remaining_four_numbers_l205_205442

theorem mean_of_remaining_four_numbers (a b c d : ℝ) (h: (a + b + c + d + 105) / 5 = 90) :
  (a + b + c + d) / 4 = 86.25 :=
by
  sorry

end mean_of_remaining_four_numbers_l205_205442


namespace intersection_A_B_l205_205210

def A : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (x^2 / 4 + 3 * y^2 / 4 = 1) }
def B : Set (ℝ × ℝ) := { p | ∃ x y, p = (x, y) ∧ (y = x^2) }

theorem intersection_A_B :
  {x : ℝ | 0 ≤ x ∧ x ≤ 2} = 
  {x : ℝ | ∃ y : ℝ, ((x, y) ∈ A ∧ (x, y) ∈ B)} :=
by
  sorry

end intersection_A_B_l205_205210


namespace pool_capacity_l205_205040

theorem pool_capacity (C : ℝ) (initial_water : ℝ) :
  0.85 * C - 0.70 * C = 300 → C = 2000 :=
by
  intro h
  sorry

end pool_capacity_l205_205040


namespace total_videos_watched_l205_205468

variable (Ekon Uma Kelsey : ℕ)

theorem total_videos_watched
  (hKelsey : Kelsey = 160)
  (hKelsey_Ekon : Kelsey = Ekon + 43)
  (hEkon_Uma : Ekon = Uma - 17) :
  Kelsey + Ekon + Uma = 411 := by
  sorry

end total_videos_watched_l205_205468


namespace arrange_PERSEVERANCE_l205_205643

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def count_permutations (total : ℕ) (counts : List ℕ) : ℕ :=
  factorial total / (counts.map factorial).foldl (*) 1

def total_letters := 12
def e_count := 3
def r_count := 2
def n_count := 2
def word_counts := [e_count, r_count, n_count]

theorem arrange_PERSEVERANCE : count_permutations total_letters word_counts = 19958400 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end arrange_PERSEVERANCE_l205_205643


namespace lily_petals_l205_205818

theorem lily_petals (L : ℕ) (h1 : 8 * L + 15 = 63) : L = 6 :=
by sorry

end lily_petals_l205_205818


namespace percentage_students_with_same_grade_l205_205682

def total_students : ℕ := 50
def students_with_same_grade : ℕ := 3 + 6 + 8 + 2 + 1

theorem percentage_students_with_same_grade :
  (students_with_same_grade / total_students : ℚ) * 100 = 40 :=
by
  sorry

end percentage_students_with_same_grade_l205_205682


namespace find_value_l205_205079

theorem find_value (x y : ℝ) (h1 : 3 * x + y = 5) (h2 : x + 3 * y = 8) : 5 * x^2 + 11 * x * y + 5 * y^2 = 89 :=
by
  sorry

end find_value_l205_205079


namespace tank_capacity_l205_205262

theorem tank_capacity (x : ℝ) (h₁ : 0.40 * x = 60) : x = 150 :=
by
  -- a suitable proof would go here
  -- since we are only interested in the statement, we place sorry in place of the proof
  sorry

end tank_capacity_l205_205262


namespace four_people_pairing_l205_205758

theorem four_people_pairing
    (persons : Fin 4 → Type)
    (common_language : ∀ (i j : Fin 4), Prop)
    (communicable : ∀ (i j k : Fin 4), common_language i j ∨ common_language j k ∨ common_language k i)
    : ∃ (i j : Fin 4) (k l : Fin 4), i ≠ j ∧ k ≠ l ∧ common_language i j ∧ common_language k l := 
sorry

end four_people_pairing_l205_205758


namespace binary_multiplication_correct_l205_205373

-- Define binary numbers as strings to directly use them in Lean
def binary_num1 : String := "1111"
def binary_num2 : String := "111"

-- Define a function to convert binary strings to natural numbers
def binary_to_nat (s : String) : Nat :=
  s.foldl (fun acc c => acc * 2 + (if c = '1' then 1 else 0)) 0

-- Define the target multiplication result
def binary_product_correct : Nat :=
  binary_to_nat "1001111"

theorem binary_multiplication_correct :
  binary_to_nat binary_num1 * binary_to_nat binary_num2 = binary_product_correct :=
by
  sorry

end binary_multiplication_correct_l205_205373


namespace baker_new_cakes_l205_205502

theorem baker_new_cakes :
  ∀ (initial_bought new_bought sold final : ℕ),
  initial_bought = 173 →
  sold = 86 →
  final = 190 →
  final = initial_bought + new_bought - sold →
  new_bought = 103 :=
by
  intros initial_bought new_bought sold final H_initial H_sold H_final H_eq
  sorry

end baker_new_cakes_l205_205502


namespace average_children_families_with_children_is_3_point_8_l205_205846

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205846


namespace expression_value_l205_205507

noncomputable def givenExpression : ℝ :=
  -2^2 + Real.sqrt 8 - 3 + 1/3

theorem expression_value : givenExpression = -20/3 + 2 * Real.sqrt 2 := 
by
  sorry

end expression_value_l205_205507


namespace remaining_water_after_45_days_l205_205989

def initial_water : ℝ := 500
def daily_loss : ℝ := 1.2
def days : ℝ := 45

theorem remaining_water_after_45_days :
  initial_water - daily_loss * days = 446 := by
  sorry

end remaining_water_after_45_days_l205_205989


namespace integer_root_of_quadratic_eq_l205_205571

theorem integer_root_of_quadratic_eq (m : ℤ) (hm : ∃ x : ℤ, m * x^2 + 2 * (m - 5) * x + (m - 4) = 0) : m = -4 ∨ m = 4 ∨ m = -16 :=
sorry

end integer_root_of_quadratic_eq_l205_205571


namespace nathan_ate_100_gumballs_l205_205547

/-- Define the number of gumballs per package. -/
def gumballs_per_package : ℝ := 5.0

/-- Define the number of packages Nathan ate. -/
def number_of_packages : ℝ := 20.0

/-- Define the total number of gumballs Nathan ate. -/
def total_gumballs : ℝ := number_of_packages * gumballs_per_package

/-- Prove that Nathan ate 100.0 gumballs. -/
theorem nathan_ate_100_gumballs : total_gumballs = 100.0 :=
sorry

end nathan_ate_100_gumballs_l205_205547


namespace binomial_equality_l205_205811

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l205_205811


namespace books_loaned_out_l205_205148

/-- 
Given:
- There are 75 books in a special collection at the beginning of the month.
- By the end of the month, 70 percent of books that were loaned out are returned.
- There are 60 books in the special collection at the end of the month.
Prove:
- The number of books loaned out during the month is 50.
-/
theorem books_loaned_out (x : ℝ) (h1 : 75 - 0.3 * x = 60) : x = 50 :=
by
  sorry

end books_loaned_out_l205_205148


namespace count_integer_values_satisfying_condition_l205_205011

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l205_205011


namespace motorboat_distance_l205_205299

variable (S v u : ℝ)
variable (V_m : ℝ := 2 * v + u)  -- Velocity of motorboat downstream
variable (V_b : ℝ := 3 * v - u)  -- Velocity of boat upstream

theorem motorboat_distance :
  ( L = (161 / 225) * S ∨ L = (176 / 225) * S) :=
by
  sorry

end motorboat_distance_l205_205299


namespace derivative_value_at_pi_over_12_l205_205397

open Real

theorem derivative_value_at_pi_over_12 :
  let f (x : ℝ) := cos (2 * x + π / 3)
  deriv f (π / 12) = -2 :=
by
  let f (x : ℝ) := cos (2 * x + π / 3)
  sorry

end derivative_value_at_pi_over_12_l205_205397


namespace find_constants_l205_205238

def equation1 (x p q : ℝ) : Prop := (x + p) * (x + q) * (x + 5) = 0
def equation2 (x p q : ℝ) : Prop := (x + 2 * p) * (x + 2) * (x + 3) = 0

def valid_roots1 (p q : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ equation1 x₁ p q ∧ equation1 x₂ p q ∧
  x₁ = -5 ∨ x₁ = -q ∨ x₁ = -p

def valid_roots2 (p q : ℝ) : Prop :=
  ∃ x₃ x₄ : ℝ, x₃ ≠ x₄ ∧ equation2 x₃ p q ∧ equation2 x₄ p q ∧
  (x₃ = -2 * p ∨ x₃ = -2 ∨ x₃ = -3)

theorem find_constants (p q : ℝ) (h1 : valid_roots1 p q) (h2 : valid_roots2 p q) : 100 * p + q = 502 :=
by
  sorry

end find_constants_l205_205238


namespace sum_ratios_eq_l205_205106

-- Define points A, B, C, D, E, and G as well as their relationships
variables {A B C D E G : Type}

-- Given conditions
axiom BD_2DC : ∀ {BD DC : ℝ}, BD = 2 * DC
axiom AE_3EB : ∀ {AE EB : ℝ}, AE = 3 * EB
axiom AG_2GD : ∀ {AG GD : ℝ}, AG = 2 * GD

-- Mass assumptions for the given problem
noncomputable def mC := 1
noncomputable def mB := 2
noncomputable def mD := mB + 2 * mC  -- mD = B's mass + 2*C's mass
noncomputable def mA := 1
noncomputable def mE := 3 * mA + mB  -- mE = 3A's mass + B's mass
noncomputable def mG := 2 * mA + mD  -- mG = 2A's mass + D's mass

-- Ratios defined according to the problem statement
noncomputable def ratio1 := (1 : ℝ) / mE
noncomputable def ratio2 := mD / mA
noncomputable def ratio3 := mD / mG

-- The Lean theorem to state the problem and correct answer
theorem sum_ratios_eq : ratio1 + ratio2 + ratio3 = (73 / 15 : ℝ) :=
by
  unfold ratio1 ratio2 ratio3
  sorry

end sum_ratios_eq_l205_205106


namespace binom_eq_sum_l205_205538

theorem binom_eq_sum (x : ℕ) : (∃ x : ℕ, Nat.choose 7 x = 21) ∧ Nat.choose 7 x = Nat.choose 6 5 + Nat.choose 6 4 :=
by
  sorry

end binom_eq_sum_l205_205538


namespace quadratic_real_roots_iff_range_k_quadratic_real_roots_specific_value_k_l205_205661

theorem quadratic_real_roots_iff_range_k (k : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 - 4 * x1 + k + 1 = 0 ∧ x2^2 - 4 * x2 + k + 1 = 0 ∧ x1 ≠ x2) ↔ k ≤ 3 :=
by
  sorry

theorem quadratic_real_roots_specific_value_k (k : ℝ) (x1 x2 : ℝ) :
  x1^2 - 4 * x1 + k + 1 = 0 →
  x2^2 - 4 * x2 + k + 1 = 0 →
  x1 ≠ x2 →
  (3 / x1 + 3 / x2 = x1 * x2 - 4) →
  k = -3 :=
by
  sorry

end quadratic_real_roots_iff_range_k_quadratic_real_roots_specific_value_k_l205_205661


namespace red_balls_count_l205_205767

theorem red_balls_count (r y b : ℕ) (total_balls : ℕ := 15) (prob_neither_red : ℚ := 2/7) :
    y + b = total_balls - r → (15 - r) * (14 - r) = 60 → r = 5 :=
by
  intros h1 h2
  sorry

end red_balls_count_l205_205767


namespace team_selection_ways_l205_205122

theorem team_selection_ways :
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  choose boys team_size_boys * choose girls team_size_girls = 103950 :=
by
  let boys := 10
  let girls := 12
  let team_size_boys := 4
  let team_size_girls := 4
  let choose (n k : ℕ) : ℕ := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))
  sorry

end team_selection_ways_l205_205122


namespace function_satisfies_conditions_l205_205237

noncomputable def f (x : ℝ) := 1 / 2 - sin x ^ 2

theorem function_satisfies_conditions (x : ℝ) :
  (|f x + cos x ^ 2| ≤ 3 / 4) ∧ (|f x - sin x ^ 2| ≤ 1 / 4) :=
by
  sorry

end function_satisfies_conditions_l205_205237


namespace binom_18_4_l205_205808

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l205_205808


namespace least_whole_number_subtracted_l205_205615

theorem least_whole_number_subtracted {x : ℕ} (h : 6 > x ∧ 7 > x) :
  (6 - x) / (7 - x : ℝ) < 16 / 21 -> x = 3 :=
by
  intros
  sorry

end least_whole_number_subtracted_l205_205615


namespace ratio_of_times_l205_205984

theorem ratio_of_times (A_work_time B_combined_rate : ℕ) 
  (h1 : A_work_time = 6) 
  (h2 : (1 / (1 / A_work_time + 1 / (B_combined_rate / 2))) = 2) :
  (B_combined_rate : ℝ) / A_work_time = 1 / 2 :=
by
  -- below we add the proof part which we will skip for now with sorry.
  sorry

end ratio_of_times_l205_205984


namespace express_q_as_polynomial_l205_205258

def q (x : ℝ) : ℝ := x^3 + 4

theorem express_q_as_polynomial (x : ℝ) : 
  q x + (2 * x^6 + x^5 + 4 * x^4 + 6 * x^2) = (5 * x^4 + 10 * x^3 - x^2 + 8 * x + 15) → 
  q x = -2 * x^6 - x^5 + x^4 + 10 * x^3 - 7 * x^2 + 8 * x + 15 := by
  sorry

end express_q_as_polynomial_l205_205258


namespace rectangle_area_l205_205150

theorem rectangle_area (length diagonal : ℝ) (h_length : length = 16) (h_diagonal : diagonal = 20) : 
  ∃ width : ℝ, (length * width = 192) :=
by 
  sorry

end rectangle_area_l205_205150


namespace valuable_files_count_l205_205785

theorem valuable_files_count 
    (initial_files : ℕ) 
    (deleted_fraction_initial : ℚ) 
    (additional_files : ℕ) 
    (irrelevant_fraction_additional : ℚ) 
    (h1 : initial_files = 800) 
    (h2 : deleted_fraction_initial = (70:ℚ) / 100)
    (h3 : additional_files = 400)
    (h4 : irrelevant_fraction_additional = (3:ℚ) / 5) : 
    (initial_files - ⌊deleted_fraction_initial * initial_files⌋ + additional_files - ⌊irrelevant_fraction_additional * additional_files⌋) = 400 :=
by sorry

end valuable_files_count_l205_205785


namespace pattern_generalization_l205_205123

theorem pattern_generalization (n : ℕ) (h : 0 < n) : n * (n + 2) + 1 = (n + 1) ^ 2 :=
by
  -- TODO: The proof will be filled in later
  sorry

end pattern_generalization_l205_205123


namespace acute_triangle_angle_C_acute_triangle_sum_ab_l205_205095

open Real

theorem acute_triangle_angle_C
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_acute : a^2 + b^2 > c^2)
  (eq1 : sqrt 3 * a = 2 * c * sin (1/2 * Real.pi))
  (h_area : c = sqrt 7)
  (h_area2 : 1 / 2 * a * b * sin (1/3 * Real.pi) = 3 * sqrt 3 / 2) :
  (C : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ → ℝ)
   := sorry

theorem acute_triangle_sum_ab
  (a b c : ℝ)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_acute : a^2 + b^2 > c^2)
  (h_area : c = sqrt 7)
  (h_area2 : 1 / 2 * a * b * sin (1/3 * Real.pi) = 3 * sqrt 3 / 2)
  (C_eq : C = 1/3 * Real.pi) :
  a + b = 5 := sorry

end acute_triangle_angle_C_acute_triangle_sum_ab_l205_205095


namespace integer_values_satisfying_sqrt_inequality_l205_205016

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l205_205016


namespace part1_part2_l205_205093

-- Define the triangle with sides a, b, c and the properties given.
variable (a b c : ℝ) (A B C : ℝ)
variable (A_ne_zero : A ≠ 0)
variable (b_cos_C a_cos_A c_cos_B : ℝ)

-- Given conditions
variable (h1 : b_cos_C = b * Real.cos C)
variable (h2 : a_cos_A = a * Real.cos A)
variable (h3 : c_cos_B = c * Real.cos B)
variable (h_seq : b_cos_C + c_cos_B = 2 * a_cos_A)
variable (A_plus_B_plus_C_eq_pi : A + B + C = Real.pi)

-- Part 1
theorem part1 : (A = Real.pi / 3) :=
by sorry

-- Part 2 with additional conditions
variable (h_a : a = 3 * Real.sqrt 2)
variable (h_bc_sum : b + c = 6)

theorem part2 : (|Real.sqrt (b ^ 2 + c ^ 2 - b * c)| = Real.sqrt 30) :=
by sorry

end part1_part2_l205_205093


namespace businessman_earnings_l205_205136

theorem businessman_earnings : 
  let P : ℝ := 1000
  let day1_stock := 1000 / P
  let day2_stock := 1000 / (P * 1.1)
  let day3_stock := 1000 / (P * 1.1^2)
  let value_on_day4 stock := stock * (P * 1.1^3)
  let total_earnings := value_on_day4 day1_stock + value_on_day4 day2_stock + value_on_day4 day3_stock
  total_earnings = 3641 := sorry

end businessman_earnings_l205_205136


namespace largest_constant_inequality_l205_205650

theorem largest_constant_inequality :
  ∃ C, (∀ x y z : ℝ, x^2 + y^2 + z^3 + 1 ≥ C * (x + y + z)) ∧ (C = Real.sqrt 2) :=
sorry

end largest_constant_inequality_l205_205650


namespace arrange_PERSEVERANCE_l205_205644

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def count_permutations (total : ℕ) (counts : List ℕ) : ℕ :=
  factorial total / (counts.map factorial).foldl (*) 1

def total_letters := 12
def e_count := 3
def r_count := 2
def n_count := 2
def word_counts := [e_count, r_count, n_count]

theorem arrange_PERSEVERANCE : count_permutations total_letters word_counts = 19958400 :=
by
  -- This is where the proof would go, but we'll use sorry to skip it
  sorry

end arrange_PERSEVERANCE_l205_205644


namespace a1_a9_sum_l205_205394

noncomputable def arithmetic_sequence (a: ℕ → ℝ) : Prop :=
∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem a1_a9_sum
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3_a7_roots : (a 3 = 3 ∧ a 7 = -1) ∨ (a 3 = -1 ∧ a 7 = 3)) :
  a 1 + a 9 = 2 :=
by
  sorry

end a1_a9_sum_l205_205394


namespace sam_new_books_not_signed_l205_205174

noncomputable def num_books_adventure := 13
noncomputable def num_books_mystery := 17
noncomputable def num_books_scifi := 25
noncomputable def num_books_nonfiction := 10
noncomputable def num_books_comics := 5
noncomputable def num_books_total := num_books_adventure + num_books_mystery + num_books_scifi + num_books_nonfiction + num_books_comics

noncomputable def num_books_used := 42
noncomputable def num_books_signed := 10
noncomputable def num_books_borrowed := 3
noncomputable def num_books_lost := 4

noncomputable def num_books_new := num_books_total - num_books_used
noncomputable def num_books_new_not_signed := num_books_new - num_books_signed
noncomputable def num_books_final := num_books_new_not_signed - num_books_lost

theorem sam_new_books_not_signed : num_books_final = 14 :=
by
  sorry

end sam_new_books_not_signed_l205_205174


namespace average_children_in_families_with_children_l205_205841

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205841


namespace total_people_present_l205_205735

def parents : ℕ := 105
def pupils : ℕ := 698
def total_people (parents pupils : ℕ) : ℕ := parents + pupils

theorem total_people_present : total_people parents pupils = 803 :=
by
  sorry

end total_people_present_l205_205735


namespace average_children_in_families_with_children_l205_205852

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205852


namespace value_of_a4_l205_205103

open Nat

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ ∀ n, a (n + 1) = 2 * a n + 1

theorem value_of_a4 (a : ℕ → ℕ) (h : sequence a) : a 4 = 23 :=
by
  -- Proof to be provided or implemented
  sorry

end value_of_a4_l205_205103


namespace remainder_of_expression_l205_205062

theorem remainder_of_expression :
  (8 * 7^19 + 1^19) % 9 = 3 :=
  by
    sorry

end remainder_of_expression_l205_205062


namespace sufficient_but_not_necessary_l205_205068

theorem sufficient_but_not_necessary {a b : ℝ} (h : a > b ∧ b > 0) : 
  a^2 > b^2 ∧ (¬ (a^2 > b^2 → a > b ∧ b > 0)) :=
by 
  sorry

end sufficient_but_not_necessary_l205_205068


namespace arman_hourly_rate_increase_l205_205947

theorem arman_hourly_rate_increase :
  let last_week_hours := 35
  let last_week_rate := 10
  let this_week_hours := 40
  let total_payment := 770
  let last_week_earnings := last_week_hours * last_week_rate
  let this_week_earnings := total_payment - last_week_earnings
  let this_week_rate := this_week_earnings / this_week_hours
  let rate_increase := this_week_rate - last_week_rate
  rate_increase = 0.50 :=
by {
  sorry
}

end arman_hourly_rate_increase_l205_205947


namespace lawn_length_is_70_l205_205164

-- Definitions for conditions
def width_of_lawn : ℕ := 50
def road_width : ℕ := 10
def cost_of_roads : ℕ := 3600
def cost_per_sqm : ℕ := 3

-- Proof problem
theorem lawn_length_is_70 :
  ∃ L : ℕ, 10 * L + 10 * width_of_lawn = cost_of_roads / cost_per_sqm ∧ L = 70 := by
  sorry

end lawn_length_is_70_l205_205164


namespace Tyrone_total_money_is_13_l205_205470

-- Definitions of the conditions
def Tyrone_has_two_1_dollar_bills := 2 * 1 -- $2
def Tyrone_has_one_5_dollar_bill := 1 * 5 -- $5
def Tyrone_has_13_quarters_in_dollars := 13 * 0.25 -- $3.25
def Tyrone_has_20_dimes_in_dollars := 20 * 0.10 -- $2.00
def Tyrone_has_8_nickels_in_dollars := 8 * 0.05 -- $0.40
def Tyrone_has_35_pennies_in_dollars := 35 * 0.01 -- $0.35

-- Total value calculation
def total_bills := Tyrone_has_two_1_dollar_bills + Tyrone_has_one_5_dollar_bill
def total_coins := Tyrone_has_13_quarters_in_dollars + Tyrone_has_20_dimes_in_dollars + Tyrone_has_8_nickels_in_dollars + Tyrone_has_35_pennies_in_dollars
def total_money := total_bills + total_coins

-- The theorem to prove
theorem Tyrone_total_money_is_13 : total_money = 13 := by
  sorry  -- proof goes here

end Tyrone_total_money_is_13_l205_205470


namespace angle_BAC_l205_205100

theorem angle_BAC (A B C D : Type*) (AD BD CD : ℝ) (angle_BCA : ℝ) 
  (h_AD_BD : AD = BD) (h_BD_CD : BD = CD) (h_angle_BCA : angle_BCA = 40) :
  ∃ angle_BAC : ℝ, angle_BAC = 110 := 
sorry

end angle_BAC_l205_205100


namespace find_x_for_g_l205_205399

noncomputable def g (x : ℝ) : ℝ := (↑((x + 5)/6))^(1/3)

theorem find_x_for_g :
  ∃ x : ℝ, g (3 * x) = 3 * g x ∧ x = -65 / 12 :=
by
  sorry

end find_x_for_g_l205_205399


namespace rationalize_denominator_l205_205584

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l205_205584


namespace car_speed_is_80_l205_205485

theorem car_speed_is_80 : ∃ v : ℝ, (1 / v * 3600 = 45) ∧ (v = 80) :=
by
  sorry

end car_speed_is_80_l205_205485


namespace range_of_m_l205_205657

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0) ∨ ((1 / 2) * m > 1) ↔ ((m > 4) ∧ ¬(∀ x : ℝ, x^2 + m * x + m / 2 + 2 ≥ 0)) :=
sorry

end range_of_m_l205_205657


namespace find_value_b_l205_205266

-- Define the problem-specific elements
noncomputable def is_line_eqn (y x : ℝ) : Prop := y = 4 - 2 * x

theorem find_value_b (b : ℝ) (h₀ : b > 0) (h₁ : b < 2)
  (hP : ∀ y, is_line_eqn y 0 → y = 4)
  (hS : ∀ y, is_line_eqn y 2 → y = 0)
  (h_ratio : ∀ Q R S O P,
    Q = (2, 0) ∧ R = (2, 0) ∧ S = (2, 0) ∧ P = (0, 4) ∧ O = (0, 0) →
    4 / 9 = 4 / ((Q.1 - O.1) * (Q.1 - O.1)) →
    (Q.1 - O.1) / (P.2 - O.2) = 2 / 3) :
  b = 2 :=
sorry

end find_value_b_l205_205266


namespace max_area_BPC_l205_205559

noncomputable def triangle_area_max (AB BC CA : ℝ) (D : ℝ) : ℝ :=
  if h₁ : AB = 13 ∧ BC = 15 ∧ CA = 14 then
    112.5 - 56.25 * Real.sqrt 3
  else 0

theorem max_area_BPC : triangle_area_max 13 15 14 D = 112.5 - 56.25 * Real.sqrt 3 := by
  sorry

end max_area_BPC_l205_205559


namespace connor_total_cost_l205_205322

def ticket_cost : ℕ := 10
def combo_meal_cost : ℕ := 11
def candy_cost : ℕ := 2.5

def total_cost : ℕ := ticket_cost + ticket_cost + combo_meal_cost + candy_cost + candy_cost

theorem connor_total_cost : total_cost = 36 := 
by sorry

end connor_total_cost_l205_205322


namespace fractional_part_of_water_after_replacements_l205_205483

theorem fractional_part_of_water_after_replacements :
  let initial_volume : ℚ := 25
  let removed_volume : ℚ := 5
  let antifreeze_added : ℚ := 5
  let replacement_fraction : ℚ := (initial_volume - removed_volume) / initial_volume
  (replacement_fraction ^ 5 = (1024 / 3125)) :=
by
  let initial_volume : ℚ := 25
  let removed_volume : ℚ := 5
  let antifreeze_added : ℚ := 5
  let replacement_fraction : ℚ := (initial_volume - removed_volume) / initial_volume
  show (replacement_fraction ^ 5 = (1024 / 3125))
  sorry

end fractional_part_of_water_after_replacements_l205_205483


namespace surrounding_circle_area_l205_205487

theorem surrounding_circle_area (R : ℝ) : 
  (∃ r : ℝ, r = R * (1 + Real.sqrt 2) ∧ ∃ S : ℝ, S = π * r^2) → 
  π * R^2 * (3 + 2 * Real.sqrt 2) = π * (R * (1 + Real.sqrt 2))^2 :=
by
  sorry

end surrounding_circle_area_l205_205487


namespace EmilySixthQuizScore_l205_205819

theorem EmilySixthQuizScore (x : ℕ) : 
  let scores := [85, 92, 88, 90, 93]
  let total_scores_with_x := scores.sum + x
  let desired_average := 91
  total_scores_with_x = 6 * desired_average → x = 98 := by
  sorry

end EmilySixthQuizScore_l205_205819


namespace line_parabola_intersection_l205_205964

theorem line_parabola_intersection (k : ℝ) : 
    (∀ l p: ℝ → ℝ, l = (fun x => k * x + 1) ∧ p = (fun x => 4 * x ^ 2) → 
        (∃ x, l x = p x) ∧ (∀ x1 x2, l x1 = p x1 ∧ l x2 = p x2 → x1 = x2) 
    ↔ k = 0 ∨ k = 1) :=
sorry

end line_parabola_intersection_l205_205964


namespace find_point_A_coordinates_l205_205229

theorem find_point_A_coordinates (A B C : ℝ × ℝ)
  (hB : B = (1, 2)) (hC : C = (3, 4))
  (trans_left : ∃ l : ℝ, A = (B.1 + l, B.2))
  (trans_up : ∃ u : ℝ, A = (C.1, C.2 - u)) :
  A = (3, 2) := 
sorry

end find_point_A_coordinates_l205_205229


namespace ratio_HP_HA_l205_205561

-- Given Definitions
variables (A B C P Q H : Type)
variables (h1 : Triangle A B C) (h2 : AcuteTriangle A B C) (h3 : P ≠ Q)
variables (h4 : FootOfAltitudeFrom A H B C) (h5 : OnExtendedLine P A B) (h6 : OnExtendedLine Q A C)
variables (h7 : HP = HQ) (h8 : CyclicQuadrilateral B C P Q)

-- Required Ratio
theorem ratio_HP_HA : HP = HA := sorry

end ratio_HP_HA_l205_205561


namespace average_children_l205_205868

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205868


namespace negation_is_all_odd_or_at_least_two_even_l205_205598

-- Define natural numbers a, b, and c.
variables {a b c : ℕ}

-- Define a predicate is_even which checks if a number is even.
def is_even (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k

-- Define the statement that exactly one of the natural numbers a, b, and c is even.
def exactly_one_even (a b c : ℕ) : Prop :=
  (is_even a ∨ is_even b ∨ is_even c) ∧
  ¬ (is_even a ∧ is_even b) ∧
  ¬ (is_even a ∧ is_even c) ∧
  ¬ (is_even b ∧ is_even c)

-- Define the negation of the statement that exactly one of the natural numbers a, b, and c is even.
def negation_of_exactly_one_even (a b c : ℕ) : Prop :=
  ¬ exactly_one_even a b c

-- State that the negation of exactly one even number among a, b, c is equivalent to all being odd or at least two being even.
theorem negation_is_all_odd_or_at_least_two_even :
  negation_of_exactly_one_even a b c ↔ (¬ is_even a ∧ ¬ is_even b ∧ ¬ is_even c) ∨ (is_even a ∧ is_even b) ∨ (is_even a ∧ is_even c) ∨ (is_even b ∧ is_even c) :=
sorry

end negation_is_all_odd_or_at_least_two_even_l205_205598


namespace franks_earnings_l205_205907

/-- Frank's earnings problem statement -/
theorem franks_earnings 
  (total_hours : ℕ) (days : ℕ) (regular_pay_rate : ℝ) (overtime_pay_rate : ℝ)
  (hours_first_day : ℕ) (overtime_first_day : ℕ)
  (hours_second_day : ℕ) (hours_third_day : ℕ)
  (hours_fourth_day : ℕ) (overtime_fourth_day : ℕ)
  (regular_hours_per_day : ℕ) :
  total_hours = 32 →
  days = 4 →
  regular_pay_rate = 15 →
  overtime_pay_rate = 22.50 →
  hours_first_day = 12 →
  overtime_first_day = 4 →
  hours_second_day = 8 →
  hours_third_day = 8 →
  hours_fourth_day = 12 →
  overtime_fourth_day = 4 →
  regular_hours_per_day = 8 →
  (32 * regular_pay_rate + 8 * overtime_pay_rate) = 660 := 
by 
  intros 
  sorry

end franks_earnings_l205_205907


namespace avg_children_in_families_with_children_l205_205831

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205831


namespace unique_B_cube_l205_205115

open Matrix

noncomputable def B : Matrix (Fin 2) (Fin 2) ℝ := !![p, q; r, s]

theorem unique_B_cube (B : Matrix (Fin 2) (Fin 2) ℝ) (hB : B^4 = 0) :
  ∃! (C : Matrix (Fin 2) (Fin 2) ℝ), C = B^3 :=
sorry

end unique_B_cube_l205_205115


namespace calories_per_one_bar_l205_205605

variable (total_calories : ℕ) (num_bars : ℕ)
variable (calories_per_bar : ℕ)

-- Given conditions
axiom total_calories_given : total_calories = 15
axiom num_bars_given : num_bars = 5

-- Mathematical equivalent proof problem
theorem calories_per_one_bar :
  total_calories / num_bars = calories_per_bar →
  calories_per_bar = 3 :=
by
  sorry

end calories_per_one_bar_l205_205605


namespace avg_children_in_families_with_children_l205_205834

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205834


namespace proposition_equivalence_l205_205918

-- Definition of propositions p and q
variables (p q : Prop)

-- Statement of the problem in Lean 4
theorem proposition_equivalence :
  (p ∨ q) → ¬(p ∧ q) ↔ (¬((p ∨ q) → ¬(p ∧ q)) ∧ ¬(¬(p ∧ q) → (p ∨ q))) :=
sorry

end proposition_equivalence_l205_205918


namespace difference_in_circumferences_l205_205688

def r_inner : ℝ := 25
def r_outer : ℝ := r_inner + 15

theorem difference_in_circumferences : 2 * Real.pi * r_outer - 2 * Real.pi * r_inner = 30 * Real.pi := by
  sorry

end difference_in_circumferences_l205_205688


namespace apps_added_eq_sixty_l205_205176

-- Definitions derived from the problem conditions
def initial_apps : ℕ := 50
def removed_apps : ℕ := 10
def final_apps : ℕ := 100

-- Intermediate calculation based on the problem
def apps_after_removal : ℕ := initial_apps - removed_apps

-- The main theorem stating the mathematically equivalent proof problem
theorem apps_added_eq_sixty : final_apps - apps_after_removal = 60 :=
by
  sorry

end apps_added_eq_sixty_l205_205176


namespace base6_number_divisibility_l205_205720

/-- 
Given that 45x2 in base 6 converted to its decimal equivalent is 6x + 1046,
and it is divisible by 19. Prove that x = 5 given that x is a base-6 digit.
-/
theorem base6_number_divisibility (x : ℕ) (h1 : 0 ≤ x ∧ x ≤ 5) (h2 : (6 * x + 1046) % 19 = 0) : x = 5 :=
sorry

end base6_number_divisibility_l205_205720


namespace least_subtraction_divisible_by13_l205_205480

theorem least_subtraction_divisible_by13 (n : ℕ) (h : n = 427398) : ∃ k : ℕ, k = 2 ∧ (n - k) % 13 = 0 := by
  sorry

end least_subtraction_divisible_by13_l205_205480


namespace solve_triangle_l205_205410

-- Define the given conditions
variable (A : ℝ) (AC AB : ℝ)
-- Angle A in the triangle
-- Side lengths AC and AB

noncomputable def conditions :=
  AC = 2 ∧ AB = 3 ∧ sin A + cos A = sqrt 2 / 2

-- Define the theorem to prove       
theorem solve_triangle : conditions A AC AB → 
  tan A = -(2 + sqrt 3) ∧ (∃ B C, area (triangle B C) = 3 / 4 * (sqrt 6 + sqrt 2)) :=
by
  sorry

end solve_triangle_l205_205410


namespace proof_P_A_proof_P_B_proof_independence_C_D_l205_205618

open Finset

def cards : Finset ℕ := {1, 2, 3, 4}

def combinations_3 : Finset (Finset ℕ) := cards.powerset.filter (λ s, s.card = 3)
def event_A : Finset (Finset ℕ) := combinations_3.filter (λ s, 7 < s.sum id)

def P_A : ℚ := event_A.card / combinations_3.card

def outcomes_with_replacement : ℕ × ℕ → Finset (ℕ × ℕ) := λ _, 
  { (x, y) | x ∈ cards ∧ y ∈ cards }

def event_B : Finset (ℕ × ℕ) := outcomes_with_replacement (0,0).filter (λ p, 6 < p.fst + p.snd)

def P_B : ℚ := event_B.card / (cards.card * cards.card)

def combinations_2 : Finset (Finset ℕ) := cards.powerset.filter (λ s, s.card = 2)
def event_C : Finset (Finset ℕ) := combinations_2.filter (λ s, s.sum id % 3 = 0)
def event_D : Finset (Finset ℕ) := combinations_2.filter (λ s, s.prod id % 4 = 0)

def P_C : ℚ := event_C.card / combinations_2.card
def P_D : ℚ := event_D.card / combinations_2.card
def P_C_inter_D : ℚ := (event_C ∩ event_D).card / combinations_2.card

theorem proof_P_A : P_A = 1/2 := sorry
theorem proof_P_B : P_B = 3/16 := sorry
theorem proof_independence_C_D : P_C_inter_D = P_C * P_D := sorry

end proof_P_A_proof_P_B_proof_independence_C_D_l205_205618


namespace distinct_fractions_count_sum_of_distinct_fractions_l205_205993

-- Define the conditions as a set of pairs
def domino_tiles : Set (ℕ × ℕ) := 
  { (a, b) | a ≤ 6 ∧ b ≤ 6 ∧ a ≤ b } \ { (0, 0) }

-- Define the fractions
def domino_fractions : Set ℚ := 
  { (a, b) | (a, b) ∈ domino_tiles }.image (λ (p : ℕ × ℕ), p.1 / p.2)

-- Part (a): Number of distinct values
theorem distinct_fractions_count : domino_fractions.to_finset.card = 13 :=
  sorry

-- Part (b): Sum of distinct values
theorem sum_of_distinct_fractions : (domino_fractions.to_finset : Set ℚ).Sum nat.cast = 13 / 2 :=
  sorry

end distinct_fractions_count_sum_of_distinct_fractions_l205_205993


namespace outer_circle_radius_l205_205132

theorem outer_circle_radius 
  (r₁ : ℝ) (r₂ : ℝ)
  (h₁ : r₁ = 1)
  (h₂ : r₂ = (1 + 2 * (Real.sin (Real.pi / 10)) / (1 - Real.sin (Real.pi / 10))))
  : r₂ = (1 + Real.sin (Real.pi / 10)) / (1 - Real.sin (Real.pi / 10)) :=
  by
  rw [h₁, h₂]
  sorry

end outer_circle_radius_l205_205132


namespace soap_box_height_l205_205161

theorem soap_box_height
  (carton_length carton_width carton_height : ℕ)
  (soap_length soap_width h : ℕ)
  (max_soap_boxes : ℕ)
  (h_carton_dim : carton_length = 30)
  (h_carton_width : carton_width = 42)
  (h_carton_height : carton_height = 60)
  (h_soap_length : soap_length = 7)
  (h_soap_width : soap_width = 6)
  (h_max_soap_boxes : max_soap_boxes = 360) :
  h = 1 :=
by
  sorry

end soap_box_height_l205_205161


namespace reflection_y_axis_is_A_l205_205340

def reflection_y_matrix := matrix (fin 2) (fin 2) ℤ

theorem reflection_y_axis_is_A :
  ∃ (A : reflection_y_matrix), 
  (A ⬝ (λ i j, if j = 0 then ![1, 0] else ![0, 1])) = (λ i j, if j = 0 then ![-1, 0] else ![0, 1]) :=
sorry

end reflection_y_axis_is_A_l205_205340


namespace average_children_in_families_with_children_l205_205864

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205864


namespace bob_pennies_l205_205218

theorem bob_pennies (a b : ℕ) 
  (h1 : b + 1 = 4 * (a - 1)) 
  (h2 : b - 1 = 3 * (a + 1)) : 
  b = 31 :=
by
  sorry

end bob_pennies_l205_205218


namespace largest_and_smallest_A_exists_l205_205292

theorem largest_and_smallest_A_exists (B B1 B2 : ℕ) (A_max A_min : ℕ) :
  -- Conditions: B > 666666666, B coprime with 24, and A obtained by moving the last digit to the first position
  B > 666666666 ∧ Nat.coprime B 24 ∧ 
  A_max = 10^8 * (B1 % 10) + B1 / 10 ∧ 
  A_min = 10^8 * (B2 % 10) + B2 / 10 ∧ 
  -- Values of B1 and B2 satisfying conditions
  B1 = 999999989 ∧ B2 = 666666671
  -- Largest and smallest A values
  ⊢ A_max = 999999998 ∧ A_min = 166666667 :=
sorry

end largest_and_smallest_A_exists_l205_205292


namespace ellipse_properties_l205_205059

noncomputable def a_square : ℝ := 2
noncomputable def b_square : ℝ := 9 / 8
noncomputable def c_square : ℝ := a_square - b_square
noncomputable def c : ℝ := Real.sqrt c_square
noncomputable def distance_between_foci : ℝ := 2 * c
noncomputable def eccentricity : ℝ := c / Real.sqrt a_square

theorem ellipse_properties :
  (distance_between_foci = Real.sqrt 14) ∧ (eccentricity = Real.sqrt 7 / 4) := by
  sorry

end ellipse_properties_l205_205059


namespace ratio_n_over_p_l205_205966

-- Definitions and conditions from the problem
variables {m n p : ℝ}

-- The quadratic equation x^2 + mx + n = 0 has roots that are thrice those of x^2 + px + m = 0.
-- None of m, n, and p is zero.

-- Prove that n / p = 27 given these conditions.
theorem ratio_n_over_p (hmn0 : m ≠ 0) (hn : n = 9 * m) (hp : p = m / 3):
  n / p = 27 :=
  by
    sorry -- Formal proof will go here.

end ratio_n_over_p_l205_205966


namespace one_sixth_time_l205_205157

-- Conditions
def total_kids : ℕ := 40
def kids_less_than_6_minutes : ℕ := total_kids * 10 / 100
def kids_less_than_8_minutes : ℕ := 3 * kids_less_than_6_minutes
def remaining_kids : ℕ := total_kids - (kids_less_than_6_minutes + kids_less_than_8_minutes)
def kids_more_than_certain_minutes : ℕ := 4
def one_sixth_remaining_kids : ℕ := remaining_kids / 6

-- Statement to prove the equivalence
theorem one_sixth_time :
  one_sixth_remaining_kids = kids_more_than_certain_minutes := 
sorry

end one_sixth_time_l205_205157


namespace sqrt_factorial_squared_l205_205791

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l205_205791


namespace ratio_of_m_l205_205951

theorem ratio_of_m (a b m m1 m2 : ℚ) 
  (h1 : a^2 - 2*a + (3/m) = 0)
  (h2 : a + b = 2 - 2/m)
  (h3 : a * b = 3/m)
  (h4 : (a/b) + (b/a) = 3/2) 
  (h5 : 8 * m^2 - 31 * m + 8 = 0)
  (h6 : m1 + m2 = 31/8)
  (h7 : m1 * m2 = 1) :
  (m1/m2) + (m2/m1) = 833/64 :=
sorry

end ratio_of_m_l205_205951


namespace product_of_powers_l205_205135

theorem product_of_powers :
  ((-1 : Int)^3) * ((-2 : Int)^2) = -4 := by
  sorry

end product_of_powers_l205_205135


namespace integral_exp_eq_e_sub_1_l205_205730

open intervalIntegral

noncomputable def integral_exp_from_0_to_1 : ℝ :=
  ∫ x in 0..1, exp x

theorem integral_exp_eq_e_sub_1 : integral_exp_from_0_to_1 = Real.exp 1 - 1 :=
by
  sorry

end integral_exp_eq_e_sub_1_l205_205730


namespace unique_partition_no_primes_l205_205532

open Set

def C_oplus_C (C : Set ℕ) : Set ℕ :=
  {z | ∃ x y, x ∈ C ∧ y ∈ C ∧ x ≠ y ∧ z = x + y}

def is_partition (A B : Set ℕ) : Prop :=
  (A ∪ B = univ) ∧ (A ∩ B = ∅)

theorem unique_partition_no_primes (A B : Set ℕ) :
  (is_partition A B) ∧ (∀ x ∈ C_oplus_C A, ¬Nat.Prime x) ∧ (∀ x ∈ C_oplus_C B, ¬Nat.Prime x) ↔ 
    (A = { n | n % 2 = 1 }) ∧ (B = { n | n % 2 = 0 }) :=
sorry

end unique_partition_no_primes_l205_205532


namespace recyclable_cans_and_bottles_collected_l205_205138

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end recyclable_cans_and_bottles_collected_l205_205138


namespace find_m_l205_205669

noncomputable def vec_add (a b : ℝ × ℝ) : ℝ × ℝ :=
  (a.1 + b.1, a.2 + b.2)

noncomputable def dot_product (a b : ℝ × ℝ) : ℝ :=
  a.1 * b.1 + a.2 * b.2

theorem find_m (m : ℝ) (h : dot_product (vec_add (-1, 2) (m, 1)) (-1, 2) = 0) : m = 7 :=
  by 
  sorry

end find_m_l205_205669


namespace eval_expression_l205_205649

theorem eval_expression (a : ℕ) (h : a = 2) : a^3 * a^6 = 512 := by
  sorry

end eval_expression_l205_205649


namespace inequality_solution_l205_205961

theorem inequality_solution (x : ℝ) :
  (x / (x^2 - 4) ≥ 0) ↔ (x ∈ Set.Iio (-2) ∪ Set.Ico 0 2) :=
by sorry

end inequality_solution_l205_205961


namespace cat_ratio_l205_205425

theorem cat_ratio (jacob_cats annie_cats melanie_cats : ℕ)
  (H1 : jacob_cats = 90)
  (H2 : annie_cats = jacob_cats / 3)
  (H3 : melanie_cats = 60) :
  melanie_cats / annie_cats = 2 := 
  by
  sorry

end cat_ratio_l205_205425


namespace no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l205_205328

theorem no_integer_solution_2_to_2x_minus_3_to_2y_eq_58
  (x y : ℕ)
  (h1 : 2 ^ (2 * x) - 3 ^ (2 * y) = 58) : false :=
by
  sorry

end no_integer_solution_2_to_2x_minus_3_to_2y_eq_58_l205_205328


namespace least_divisor_for_perfect_square_l205_205144

theorem least_divisor_for_perfect_square : 
  ∃ d : ℕ, (∀ n : ℕ, n > 0 → 16800 / d = n * n) ∧ d = 21 := 
sorry

end least_divisor_for_perfect_square_l205_205144


namespace power_of_six_evaluation_l205_205820

noncomputable def example_expr : ℝ := (6 : ℝ)^(1/4) / (6 : ℝ)^(1/6)

theorem power_of_six_evaluation : example_expr = (6 : ℝ)^(1/12) := 
by
  sorry

end power_of_six_evaluation_l205_205820


namespace union_of_intervals_l205_205933

theorem union_of_intervals :
  let M := {x : ℝ | x^2 - 3 * x - 4 ≤ 0}
  let N := {x : ℝ | x^2 - 16 ≤ 0}
  M ∪ N = {x : ℝ | -4 ≤ x ∧ x ≤ 4} :=
by
  sorry

end union_of_intervals_l205_205933


namespace perimeter_of_figure_l205_205472

theorem perimeter_of_figure (a b c d : ℕ) (p : ℕ) (h1 : a = 6) (h2 : b = 3) (h3 : c = 2) (h4 : d = 4) (h5 : p = a * b + c * d) : p = 26 :=
by
  sorry

end perimeter_of_figure_l205_205472


namespace complex_number_purely_imaginary_l205_205679

theorem complex_number_purely_imaginary (a : ℝ) (i : ℂ) (h₁ : (a^2 - a - 2 : ℝ) = 0) (h₂ : (a + 1 ≠ 0)) : a = 2 := 
by {
  sorry
}

end complex_number_purely_imaginary_l205_205679


namespace natasha_dimes_l205_205428

theorem natasha_dimes (n : ℕ) :
  100 < n ∧ n < 200 ∧
  n % 3 = 2 ∧
  n % 4 = 2 ∧
  n % 5 = 2 ∧
  n % 7 = 2 ↔ n = 182 := by
sorry

end natasha_dimes_l205_205428


namespace min_p_plus_q_l205_205259

-- Define the conditions
variables {p q : ℕ}

-- Problem statement in Lean 4
theorem min_p_plus_q (h₁ : p > 0) (h₂ : q > 0) (h₃ : 108 * p = q^3) : p + q = 8 :=
sorry

end min_p_plus_q_l205_205259


namespace cistern_fill_time_l205_205024

-- Define the rates at which pipes p, q, and r can fill/drain the cistern.
def rate_p := 1/10
def rate_q := 1/15
def rate_r := -1/30

-- Define the time pipes p and q are open together.
def time_pq_open := 4

-- Define the remaining fraction of the cistern to be filled after 4 minutes.
def filled_cistern_after_4_minutes : ℚ := (rate_p + rate_q) * time_pq_open
def remaining_cistern : ℚ := 1 - filled_cistern_after_4_minutes

-- Define the combined rate of pipes q and r.
def combined_rate_q_r := rate_q + rate_r

-- Prove that the time it takes to fill the remaining cistern at the combined rate is 10 minutes.
theorem cistern_fill_time : 
  remaining_cistern / combined_rate_q_r = 10 := 
by 
  sorry

end cistern_fill_time_l205_205024


namespace expression_D_is_odd_l205_205476

namespace ProofProblem

def is_odd (n : ℤ) : Prop :=
  ∃ k : ℤ, n = 2 * k + 1

theorem expression_D_is_odd :
  is_odd (3 + 5 + 1) :=
by
  sorry

end ProofProblem

end expression_D_is_odd_l205_205476


namespace Nancy_money_in_dollars_l205_205956

-- Condition: Nancy has saved 1 dozen quarters
def dozen : ℕ := 12

-- Condition: Each quarter is worth 25 cents
def value_of_quarter : ℕ := 25

-- Condition: 100 cents is equal to 1 dollar
def cents_per_dollar : ℕ := 100

-- Proving that Nancy has 3 dollars
theorem Nancy_money_in_dollars :
  (dozen * value_of_quarter) / cents_per_dollar = 3 := by
  sorry

end Nancy_money_in_dollars_l205_205956


namespace rotation_matrix_150_deg_correct_l205_205365

open Real
open Matrix

noncomputable def rotation_matrix_150_deg : Matrix (Fin 2) (Fin 2) ℝ :=
  let cos150 := -sqrt 3 / 2
  let sin150 := 1 / 2
  ![![cos150, -sin150], ![sin150, cos150]]

theorem rotation_matrix_150_deg_correct :
  rotation_matrix_150_deg = ![![-(sqrt 3 / 2), -1 / 2], ![1 / 2, -(sqrt 3 / 2)]] :=
by
  sorry

end rotation_matrix_150_deg_correct_l205_205365


namespace avg_children_in_families_with_children_l205_205835

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205835


namespace trumpet_cost_l205_205690

/-
  Conditions:
  1. Cost of the music tool: $9.98
  2. Cost of the song book: $4.14
  3. Total amount Joan spent at the music store: $163.28

  Prove that the cost of the trumpet is $149.16
-/

theorem trumpet_cost :
  let c_mt := 9.98
  let c_sb := 4.14
  let t_sp := 163.28
  let c_trumpet := t_sp - (c_mt + c_sb)
  c_trumpet = 149.16 :=
by
  sorry

end trumpet_cost_l205_205690


namespace coat_total_selling_price_l205_205488

theorem coat_total_selling_price :
  let original_price := 120
  let discount_percent := 30
  let tax_percent := 8
  let discount_amount := (discount_percent / 100) * original_price
  let sale_price := original_price - discount_amount
  let tax_amount := (tax_percent / 100) * sale_price
  let total_selling_price := sale_price + tax_amount
  total_selling_price = 90.72 :=
by
  sorry

end coat_total_selling_price_l205_205488


namespace total_money_given_by_father_is_100_l205_205672

-- Define the costs and quantities given in the problem statement.
def cost_per_sharpener := 5
def cost_per_notebook := 5
def cost_per_eraser := 4
def money_spent_on_highlighters := 30

def heaven_sharpeners := 2
def heaven_notebooks := 4
def brother_erasers := 10

-- Calculate the total amount of money given by their father.
def total_money_given : ℕ :=
  heaven_sharpeners * cost_per_sharpener +
  heaven_notebooks * cost_per_notebook +
  brother_erasers * cost_per_eraser +
  money_spent_on_highlighters

-- Lean statement to prove
theorem total_money_given_by_father_is_100 :
  total_money_given = 100 := by
  sorry

end total_money_given_by_father_is_100_l205_205672


namespace math_problem_l205_205065

theorem math_problem (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x - 1) * (deriv f x) ≥ 0) : 
  f 0 + f 2 ≥ 2 * f 1 :=
sorry

end math_problem_l205_205065


namespace sales_tax_difference_l205_205034

-- Definitions for the conditions
def item_price : ℝ := 50
def tax_rate1 : ℝ := 0.075
def tax_rate2 : ℝ := 0.05

-- Calculations based on the conditions
def tax1 := item_price * tax_rate1
def tax2 := item_price * tax_rate2

-- The proof statement
theorem sales_tax_difference :
  tax1 - tax2 = 1.25 :=
by
  sorry

end sales_tax_difference_l205_205034


namespace isosceles_triangle_base_angle_l205_205226

/-- In an isosceles triangle, if one angle is 110 degrees, then each base angle measures 35 degrees. -/
theorem isosceles_triangle_base_angle (α β γ : ℝ) (h1 : α + β + γ = 180)
  (h2 : α = β ∨ α = γ ∨ β = γ) (h3 : α = 110 ∨ β = 110 ∨ γ = 110) :
  β = 35 ∨ γ = 35 :=
sorry

end isosceles_triangle_base_angle_l205_205226


namespace balloon_ratio_l205_205783

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end balloon_ratio_l205_205783


namespace intersection_A_B_l205_205917

def A (x : ℝ) : Prop := x^2 - 3 * x < 0
def B (x : ℝ) : Prop := x > 2

theorem intersection_A_B : {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | 2 < x ∧ x < 3} :=
by
  sorry

end intersection_A_B_l205_205917


namespace find_g_l205_205887

variable (x : ℝ)

-- Given condition
def given_condition (g : ℝ → ℝ) : Prop :=
  5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5

-- Goal
def goal (g : ℝ → ℝ) : Prop :=
  g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3

-- The statement combining given condition and goal to prove
theorem find_g (g : ℝ → ℝ) (h : given_condition x g) : goal x g :=
by
  sorry

end find_g_l205_205887


namespace rotation_matrix_150_eq_l205_205344

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ], ![sin θ, cos θ]]

noncomputable def angle_150 : ℝ := 150 * (π / 180) -- Convert 150 degrees to radians
  
theorem rotation_matrix_150_eq :
  rotation_matrix angle_150 = ![![-(sqrt 3 / 2), -1 / 2], ![1 / 2, -(sqrt 3 / 2)]] :=
by
  simp [rotation_matrix, angle_150]
  sorry

end rotation_matrix_150_eq_l205_205344


namespace Shekar_science_marks_l205_205435

-- Define Shekar's known marks
def math_marks : ℕ := 76
def social_studies_marks : ℕ := 82
def english_marks : ℕ := 47
def biology_marks : ℕ := 85

-- Define the average mark and the number of subjects
def average_mark : ℕ := 71
def number_of_subjects : ℕ := 5

-- Define Shekar's unknown mark in Science
def science_marks : ℕ := sorry  -- We expect to prove science_marks = 65

-- State the theorem to be proved
theorem Shekar_science_marks :
  average_mark * number_of_subjects = math_marks + science_marks + social_studies_marks + english_marks + biology_marks →
  science_marks = 65 :=
by sorry

end Shekar_science_marks_l205_205435


namespace integer_roots_of_quadratic_l205_205185

theorem integer_roots_of_quadratic (a : ℤ) : 
  (∃ x : ℤ , x^2 + a * x + a = 0) ↔ (a = 0 ∨ a = 4) := 
sorry

end integer_roots_of_quadratic_l205_205185


namespace avg_children_with_kids_l205_205827

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205827


namespace tyrone_money_l205_205469

def bill_value (count : ℕ) (val : ℝ) : ℝ :=
  count * val

def total_value : ℝ :=
  bill_value 2 1 + bill_value 1 5 + bill_value 13 0.25 + bill_value 20 0.10 + bill_value 8 0.05 + bill_value 35 0.01

theorem tyrone_money : total_value = 13 := by 
  sorry

end tyrone_money_l205_205469


namespace quadratic_inequality_solution_set_l205_205221

theorem quadratic_inequality_solution_set (a b : ℝ) (h : ∀ x, (1 < x ∧ x < 2) ↔ x^2 + a * x + b < 0) : b = 2 :=
sorry

end quadratic_inequality_solution_set_l205_205221


namespace one_and_two_thirds_eq_36_l205_205251

theorem one_and_two_thirds_eq_36 (x : ℝ) (h : (5 / 3) * x = 36) : x = 21.6 :=
sorry

end one_and_two_thirds_eq_36_l205_205251


namespace find_a_b_sum_l205_205674

theorem find_a_b_sum (a b : ℕ) (h : a^2 - b^4 = 2009) : a + b = 47 :=
sorry

end find_a_b_sum_l205_205674


namespace rotation_matrix_150_l205_205353

-- Define the rotation matrix
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![
    Real.cos θ, -Real.sin θ;
    Real.sin θ, Real.cos θ
  ]

-- Goal: Prove the matrix for 150 degrees rotation
theorem rotation_matrix_150 : 
  rotation_matrix (150 * Real.pi / 180) = !![
    -Real.sqrt 3 / 2, -1 / 2;
    1 / 2, -Real.sqrt 3 / 2
  ] :=
by
  sorry

end rotation_matrix_150_l205_205353


namespace binomial_equality_l205_205812

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l205_205812


namespace max_sum_of_factors_l205_205230

theorem max_sum_of_factors (A B C : ℕ) (h1 : A ≠ B) (h2 : B ≠ C) (h3 : A ≠ C) (h4 : 0 < A) (h5 : 0 < B) (h6 : 0 < C) (h7 : A * B * C = 3003) :
  A + B + C ≤ 45 :=
sorry

end max_sum_of_factors_l205_205230


namespace unique_students_total_l205_205307

variables (euclid_students raman_students pythagoras_students overlap_3 : ℕ)

def total_students (E R P O : ℕ) : ℕ := E + R + P - O

theorem unique_students_total (hE : euclid_students = 12) 
                              (hR : raman_students = 10) 
                              (hP : pythagoras_students = 15) 
                              (hO : overlap_3 = 3) : 
    total_students euclid_students raman_students pythagoras_students overlap_3 = 34 :=
by
    sorry

end unique_students_total_l205_205307


namespace lateral_area_of_given_cone_l205_205075

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end lateral_area_of_given_cone_l205_205075


namespace solve_x_floor_x_eq_72_l205_205651

theorem solve_x_floor_x_eq_72 : ∃ x : ℝ, 0 < x ∧ x * (⌊x⌋) = 72 ∧ x = 9 :=
by
  sorry

end solve_x_floor_x_eq_72_l205_205651


namespace rotation_matrix_150_l205_205364

def rotation_matrix (theta : ℝ) : Matrix ℝ := 
  ![
    ![Real.cos theta, -Real.sin theta], 
    ![Real.sin theta, Real.cos theta]
  ]

theorem rotation_matrix_150 :
  rotation_matrix (5 * Real.pi / 6) = 
  ![
    ![-Real.sqrt 3 / 2, -1 / 2], 
    ![1 / 2, -Real.sqrt 3 / 2]
  ] := by
  sorry

end rotation_matrix_150_l205_205364


namespace reflection_problem_l205_205263

theorem reflection_problem 
  (m b : ℝ)
  (h : ∀ (P Q : ℝ × ℝ), 
        P = (2,2) ∧ Q = (8,4) → 
        ∃ mid : ℝ × ℝ, 
        mid = ((P.fst + Q.fst) / 2, (P.snd + Q.snd) / 2) ∧ 
        ∃ m' : ℝ, m' ≠ 0 ∧ P.snd - m' * P.fst = Q.snd - m' * Q.fst) :
  m + b = 15 := 
sorry

end reflection_problem_l205_205263


namespace rotation_matrix_150_degrees_l205_205348

theorem rotation_matrix_150_degrees :
  ∃ (R : Matrix (Fin 2) (Fin 2) ℝ),
    R = Matrix.ofFn
      (λ i j, match (i, j) with
              | (0, 0) => -(Real.sqrt 3) / 2
              | (0, 1) => -1 / 2
              | (1, 0) => 1 / 2
              | (1, 1) => -(Real.sqrt 3) / 2
              | _ => 0) :=
begin
  sorry
end

end rotation_matrix_150_degrees_l205_205348


namespace sum_of_sequence_l205_205915

noncomputable def sequence_sum (n : ℕ) : ℤ :=
  6 * 2^n - (n + 6)

theorem sum_of_sequence (a S : ℕ → ℤ) (n : ℕ) :
  a 1 = 5 →
  (∀ n : ℕ, 1 ≤ n → S (n + 1) = 2 * S n + n + 5) →
  S n = sequence_sum n :=
by sorry

end sum_of_sequence_l205_205915


namespace distance_between_parallel_lines_l205_205736

theorem distance_between_parallel_lines (r d : ℝ) :
  let c₁ := 36
  let c₂ := 36
  let c₃ := 40
  let expr1 := (324 : ℝ) + (1 / 4) * d^2
  let expr2 := (400 : ℝ) + d^2
  let radius_eq1 := r^2 = expr1
  let radius_eq2 := r^2 = expr2
  radius_eq1 ∧ radius_eq2 → d = Real.sqrt (304 / 3) :=
by
  sorry

end distance_between_parallel_lines_l205_205736


namespace binom_18_4_eq_3060_l205_205805

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l205_205805


namespace seq_a2010_l205_205104

-- Definitions and conditions
def seq (a : ℕ → ℕ) : Prop :=
  a 1 = 2 ∧ 
  a 2 = 3 ∧ 
  ∀ n ≥ 2, a (n + 1) = (a n * a (n - 1)) % 10

-- Proof statement
theorem seq_a2010 {a : ℕ → ℕ} (h : seq a) : a 2010 = 4 := 
  sorry

end seq_a2010_l205_205104


namespace no_intersection_pair_C_l205_205523

theorem no_intersection_pair_C :
  let y1 := fun x : ℝ => x
  let y2 := fun x : ℝ => x - 3
  ∀ x : ℝ, y1 x ≠ y2 x :=
by
  sorry

end no_intersection_pair_C_l205_205523


namespace original_radius_new_perimeter_l205_205977

variable (r : ℝ)

theorem original_radius_new_perimeter (h : (π * (r + 5)^2 = 4 * π * r^2)) :
  r = 5 ∧ 2 * π * (r + 5) = 20 * π :=
by
  sorry

end original_radius_new_perimeter_l205_205977


namespace cos_double_angle_l205_205197

theorem cos_double_angle (α : ℝ) (h : Real.tan α = 1 / 2) : Real.cos (2 * α) = 3 / 5 :=
by sorry

end cos_double_angle_l205_205197


namespace integral_1_eq_pi_integral_2_eq_pi_div_4e2_l205_205331

noncomputable def integral_1 : ℝ := ∫ x in (1:ℝ)..3, (1 / real.sqrt ((x - 1) * (3 - x)))
noncomputable def integral_2 : ℝ := ∫ x in (1:ℝ)..∞, (1 / (real.exp (x + 1) + real.exp (3 - x)))

theorem integral_1_eq_pi : integral_1 = real.pi := by
  sorry

theorem integral_2_eq_pi_div_4e2 : integral_2 = real.pi / (4 * real.exp 2) := by
  sorry

end integral_1_eq_pi_integral_2_eq_pi_div_4e2_l205_205331


namespace line_through_point_parallel_to_given_line_l205_205722

theorem line_through_point_parallel_to_given_line 
  (x y : ℝ) 
  (h₁ : (x, y) = (1, -4)) 
  (h₂ : ∀ m : ℝ, 2 * 1 + 3 * (-4) + m = 0 → m = 10)
  : 2 * x + 3 * y + 10 = 0 :=
sorry

end line_through_point_parallel_to_given_line_l205_205722


namespace olivia_change_received_l205_205125

theorem olivia_change_received 
    (cost_per_basketball_card : ℕ)
    (basketball_card_count : ℕ)
    (cost_per_baseball_card : ℕ)
    (baseball_card_count : ℕ)
    (bill_amount : ℕ) :
    basketball_card_count = 2 → 
    cost_per_basketball_card = 3 → 
    baseball_card_count = 5 →
    cost_per_baseball_card = 4 →
    bill_amount = 50 →
    bill_amount - (basketball_card_count * cost_per_basketball_card + baseball_card_count * cost_per_baseball_card) = 24 := 
by {
    intros h1 h2 h3 h4 h5,
    rw [h1, h2, h3, h4, h5],
    norm_num,
}

-- Adding a placeholder proof:
-- by sorry

end olivia_change_received_l205_205125


namespace g_inv_f_7_l205_205398

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry
noncomputable def f_inv : ℝ → ℝ := sorry
noncomputable def g_inv : ℝ → ℝ := sorry

axiom f_inv_g (x : ℝ) : f_inv (g x) = x^3 - 1
axiom g_exists_inv : ∀ y : ℝ, ∃ x : ℝ, g x = y

theorem g_inv_f_7 : g_inv (f 7) = 2 :=
by
  sorry

end g_inv_f_7_l205_205398


namespace probability_qualified_from_A_is_correct_l205_205102

-- Given conditions:
def p_A : ℝ := 0.7
def pass_A : ℝ := 0.95

-- Define what we need to prove:
def qualified_from_A : ℝ := p_A * pass_A

-- Theorem statement
theorem probability_qualified_from_A_is_correct :
  qualified_from_A = 0.665 :=
by
  sorry

end probability_qualified_from_A_is_correct_l205_205102


namespace average_children_in_families_with_children_l205_205839

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205839


namespace evaluate_expression_l205_205056

noncomputable def w : ℂ := complex.exp (2 * real.pi * complex.I / 13)

theorem evaluate_expression :
  (3 - w) * (3 - w^2) * (3 - w^3) * (3 - w^4) * (3 - w^5) * 
  (3 - w^6) * (3 - w^7) * (3 - w^8) * (3 - w^9) * (3 - w^10) * 
  (3 - w^11) * (3 - w^12) = 797161 :=
begin
  sorry
end

end evaluate_expression_l205_205056


namespace polynomial_value_at_n_plus_1_l205_205660

theorem polynomial_value_at_n_plus_1 
  (f : ℕ → ℝ) 
  (n : ℕ)
  (hdeg : ∃ m, m = n) 
  (hvalues : ∀ k (hk : k ≤ n), f k = k / (k + 1)) : 
  f (n + 1) = (n + 1 + (-1) ^ (n + 1)) / (n + 2) := 
by
  sorry

end polynomial_value_at_n_plus_1_l205_205660


namespace leak_empties_cistern_in_24_hours_l205_205285

noncomputable def cistern_fill_rate_without_leak : ℝ := 1 / 8
noncomputable def cistern_fill_rate_with_leak : ℝ := 1 / 12

theorem leak_empties_cistern_in_24_hours :
  (1 / (cistern_fill_rate_without_leak - cistern_fill_rate_with_leak)) = 24 :=
by
  sorry

end leak_empties_cistern_in_24_hours_l205_205285


namespace average_children_l205_205867

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205867


namespace average_children_in_families_with_children_l205_205854

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205854


namespace roots_quadratic_sum_squares_l205_205551

theorem roots_quadratic_sum_squares :
  (∃ a b : ℝ, (∀ x : ℝ, x^2 - 4 * x + 4 = 0 → (x = a ∨ x = b)) ∧ a^2 + b^2 = 8) :=
by
  sorry

end roots_quadratic_sum_squares_l205_205551


namespace number_of_rooms_l205_205957

theorem number_of_rooms (x : ℕ) (h1 : ∀ n, 6 * (n - 1) = 5 * n + 4) : x = 10 :=
sorry

end number_of_rooms_l205_205957


namespace range_log_div_pow3_div3_l205_205908

noncomputable def log_div (x y : ℝ) : ℝ := Real.log (x / y)
noncomputable def log_div_pow3 (x y : ℝ) : ℝ := Real.log (x^3 / y^(1/2))
noncomputable def log_div_pow3_div3 (x y : ℝ) : ℝ := Real.log (x^3 / (3 * y))

theorem range_log_div_pow3_div3 
  (x y : ℝ) 
  (h1 : 1 ≤ log_div x y ∧ log_div x y ≤ 2)
  (h2 : 2 ≤ log_div_pow3 x y ∧ log_div_pow3 x y ≤ 3) 
  : Real.log (x^3 / (3 * y)) ∈ Set.Icc (26/15 : ℝ) 3 :=
sorry

end range_log_div_pow3_div3_l205_205908


namespace possible_values_expression_l205_205071

theorem possible_values_expression 
  (a b : ℝ) 
  (h₁ : a^2 = 16) 
  (h₂ : |b| = 3) 
  (h₃ : ab < 0) : 
  (a - b)^2 + a * b^2 = 85 ∨ (a - b)^2 + a * b^2 = 13 := 
by 
  sorry

end possible_values_expression_l205_205071


namespace find_K_l205_205235

theorem find_K (surface_area_cube : ℝ) (volume_sphere : ℝ) (r : ℝ) (K : ℝ) 
  (cube_side_length : ℝ) (surface_area_sphere_eq : surface_area_cube = 4 * Real.pi * (r ^ 2))
  (volume_sphere_eq : volume_sphere = (4 / 3) * Real.pi * (r ^ 3)) 
  (surface_area_cube_eq : surface_area_cube = 6 * (cube_side_length ^ 2)) 
  (volume_sphere_form : volume_sphere = (K * Real.sqrt 6) / Real.sqrt Real.pi) :
  K = 8 :=
by
  sorry

end find_K_l205_205235


namespace odometer_problem_l205_205954

theorem odometer_problem
    (x a b c : ℕ)
    (h_dist : 60 * x = (100 * b + 10 * c + a) - (100 * a + 10 * b + c))
    (h_b_ge_1 : b ≥ 1)
    (h_sum_le_9 : a + b + c ≤ 9) :
    a^2 + b^2 + c^2 = 29 :=
sorry

end odometer_problem_l205_205954


namespace sqrt_12_same_type_sqrt_3_l205_205499

-- We define that two square roots are of the same type if one is a multiple of the other
def same_type (a b : ℝ) : Prop := ∃ k : ℝ, b = k * a

-- We need to show that sqrt(12) is of the same type as sqrt(3), and check options
theorem sqrt_12_same_type_sqrt_3 : same_type (Real.sqrt 3) (Real.sqrt 12) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 8) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 18) ∧
  ¬ same_type (Real.sqrt 3) (Real.sqrt 6) :=
by
  sorry -- Proof is omitted


end sqrt_12_same_type_sqrt_3_l205_205499


namespace prism_coloring_1995_prism_coloring_1996_l205_205107

def prism_coloring_possible (n : ℕ) : Prop :=
  ∃ (color : ℕ → ℕ),
    (∀ i, 1 ≤ color i ∧ color i ≤ 3) ∧ -- Each color is within bounds
    (∀ i, color i ≠ color ((i + 1) % n)) ∧ -- Colors on each face must be different
    (n % 3 = 0 ∨ n ≠ 1996) -- Condition for coloring

theorem prism_coloring_1995 : prism_coloring_possible 1995 :=
sorry

theorem prism_coloring_1996 : ¬prism_coloring_possible 1996 :=
sorry

end prism_coloring_1995_prism_coloring_1996_l205_205107


namespace paint_fraction_second_week_l205_205691

theorem paint_fraction_second_week
  (total_paint : ℕ)
  (first_week_fraction : ℚ)
  (total_used : ℕ)
  (paint_first_week : ℕ)
  (remaining_paint : ℕ)
  (paint_second_week : ℕ)
  (fraction_second_week : ℚ) :
  total_paint = 360 →
  first_week_fraction = 1/4 →
  total_used = 225 →
  paint_first_week = first_week_fraction * total_paint →
  remaining_paint = total_paint - paint_first_week →
  paint_second_week = total_used - paint_first_week →
  fraction_second_week = paint_second_week / remaining_paint →
  fraction_second_week = 1/2 :=
by
  sorry

end paint_fraction_second_week_l205_205691


namespace rotation_matrix_150_l205_205362

def rotation_matrix (theta : ℝ) : Matrix ℝ := 
  ![
    ![Real.cos theta, -Real.sin theta], 
    ![Real.sin theta, Real.cos theta]
  ]

theorem rotation_matrix_150 :
  rotation_matrix (5 * Real.pi / 6) = 
  ![
    ![-Real.sqrt 3 / 2, -1 / 2], 
    ![1 / 2, -Real.sqrt 3 / 2]
  ] := by
  sorry

end rotation_matrix_150_l205_205362


namespace divisibility_l205_205254

def Q (X : ℤ) := (X - 1) ^ 3

def P_n (n : ℕ) (X : ℤ) : ℤ :=
  n * X ^ (n + 2) - (n + 2) * X ^ (n + 1) + (n + 2) * X - n

theorem divisibility (n : ℕ) (h : n > 0) : ∀ X : ℤ, Q X ∣ P_n n X :=
by
  sorry

end divisibility_l205_205254


namespace bryan_more_than_ben_l205_205503

theorem bryan_more_than_ben :
  let Bryan_candies := 50
  let Ben_candies := 20
  Bryan_candies - Ben_candies = 30 :=
by
  let Bryan_candies := 50
  let Ben_candies := 20
  sorry

end bryan_more_than_ben_l205_205503


namespace avg_children_in_families_with_children_l205_205878

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205878


namespace solution_set_inequality_l205_205402

theorem solution_set_inequality (m : ℝ) : (∀ x : ℝ, m * x^2 - m * x + 2 > 0) ↔ m ∈ Set.Ico 0 8 := by
  sorry

end solution_set_inequality_l205_205402


namespace sqrt_of_16_l205_205027

theorem sqrt_of_16 : Real.sqrt 16 = 4 :=
by
  sorry

end sqrt_of_16_l205_205027


namespace candy_distribution_l205_205648

theorem candy_distribution (n : ℕ) (h : n ≥ 2) :
  (∀ i : ℕ, i < n → ∃ k : ℕ, ((k * (k + 1)) / 2) % n = i) ↔ ∃ k : ℕ, n = 2 ^ k :=
by
  sorry

end candy_distribution_l205_205648


namespace numeric_puzzle_AB_eq_B_pow_V_l205_205897

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l205_205897


namespace Willy_more_crayons_l205_205753

theorem Willy_more_crayons (Willy Lucy : ℕ) (h1 : Willy = 1400) (h2 : Lucy = 290) : (Willy - Lucy) = 1110 :=
by
  -- proof goes here
  sorry

end Willy_more_crayons_l205_205753


namespace original_number_l205_205554

theorem original_number (N m a b c : ℕ) (hN : N = 3306) 
  (h_eq : 3306 + m = 222 * (a + b + c)) 
  (hm_digits : m = 100 * a + 10 * b + c) 
  (h1 : a + b + c = 15) 
  (h2 : ∃ (a b c : ℕ), a + b + c = 15 ∧ 100 * a + 10 * b + c = 78): 
  100 * a + 10 * b + c = 753 := 
by sorry

end original_number_l205_205554


namespace system_solution_l205_205755

theorem system_solution (x y z : ℚ) 
  (h1 : x + y + x * y = 19) 
  (h2 : y + z + y * z = 11) 
  (h3 : z + x + z * x = 14) :
    (x = 4 ∧ y = 3 ∧ z = 2) ∨ (x = -6 ∧ y = -5 ∧ z = -4) :=
by
  sorry

end system_solution_l205_205755


namespace binom_18_4_l205_205795

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l205_205795


namespace functional_equation_solution_exists_l205_205336

theorem functional_equation_solution_exists (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ c : ℝ, ∀ x : ℝ, f x = x + c :=
by
  intro h
  sorry

end functional_equation_solution_exists_l205_205336


namespace rotation_matrix_150_l205_205354

-- Define the rotation matrix
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![
    Real.cos θ, -Real.sin θ;
    Real.sin θ, Real.cos θ
  ]

-- Goal: Prove the matrix for 150 degrees rotation
theorem rotation_matrix_150 : 
  rotation_matrix (150 * Real.pi / 180) = !![
    -Real.sqrt 3 / 2, -1 / 2;
    1 / 2, -Real.sqrt 3 / 2
  ] :=
by
  sorry

end rotation_matrix_150_l205_205354


namespace smallest_K_exists_l205_205950

theorem smallest_K_exists (S : Finset ℕ) (h_S : S = (Finset.range 51).erase 0) :
  ∃ K, ∀ (T : Finset ℕ), T ⊆ S ∧ T.card = K → 
  ∃ a b, a ∈ T ∧ b ∈ T ∧ a ≠ b ∧ (a + b) ∣ (a * b) ∧ K = 39 :=
by
  use 39
  sorry

end smallest_K_exists_l205_205950


namespace surface_area_is_33_l205_205333

structure TShape where
  vertical_cubes : ℕ -- Number of cubes in the vertical line
  horizontal_cubes : ℕ -- Number of cubes in the horizontal line
  intersection_point : ℕ -- Intersection point in the vertical line
  
def surface_area (t : TShape) : ℕ :=
  let top_and_bottom := 9 + 9
  let side_vertical := (3 + 4) -- 3 for the top cube, 1 each for the other 4 cubes
  let side_horizontal := (4 - 1) * 2 -- each of 4 left and right minus intersection twice
  let intersection := 2
  top_and_bottom + side_vertical + side_horizontal + intersection

theorem surface_area_is_33 (t : TShape) (h1 : t.vertical_cubes = 5) (h2 : t.horizontal_cubes = 5) (h3 : t.intersection_point = 3) : 
  surface_area t = 33 := by
  sorry

end surface_area_is_33_l205_205333


namespace time_to_fill_partial_bucket_l205_205936

-- Definitions for the conditions
def time_to_fill_full_bucket : ℝ := 135
def r := 2 / 3

-- The time to fill 2/3 of the bucket should be proven as 90
theorem time_to_fill_partial_bucket : time_to_fill_full_bucket * r = 90 := 
by 
  -- Prove that 90 is the correct time to fill two-thirds of the bucket
  sorry

end time_to_fill_partial_bucket_l205_205936


namespace initial_volume_of_solution_is_six_l205_205770

theorem initial_volume_of_solution_is_six
  (V : ℝ)
  (h1 : 0.30 * V + 2.4 = 0.50 * (V + 2.4)) :
  V = 6 :=
by
  sorry

end initial_volume_of_solution_is_six_l205_205770


namespace ratio_sums_is_five_sixths_l205_205927

theorem ratio_sums_is_five_sixths
  (a b c x y z : ℝ)
  (h_positive_a : a > 0) (h_positive_b : b > 0) (h_positive_c : c > 0)
  (h_positive_x : x > 0) (h_positive_y : y > 0) (h_positive_z : z > 0)
  (h₁ : a^2 + b^2 + c^2 = 25)
  (h₂ : x^2 + y^2 + z^2 = 36)
  (h₃ : a * x + b * y + c * z = 30) :
  (a + b + c) / (x + y + z) = (5 / 6) :=
sorry

end ratio_sums_is_five_sixths_l205_205927


namespace no_integer_roots_l205_205327

  theorem no_integer_roots : ∀ x : ℤ, x^3 - 4 * x^2 - 4 * x + 24 ≠ 0 :=
  by
    sorry
  
end no_integer_roots_l205_205327


namespace average_children_in_families_with_children_l205_205838

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205838


namespace coin_flip_probability_l205_205088

theorem coin_flip_probability : 
  ∀ (prob_tails : ℚ) (seq : List (Bool × ℚ)),
    prob_tails = 1/2 →
    seq = [(true, 1/2), (true, 1/2), (false, 1/2), (false, 1/2)] →
    (seq.map Prod.snd).prod = 0.0625 :=
by 
  intros prob_tails seq htails hseq 
  sorry

end coin_flip_probability_l205_205088


namespace percent_absent_l205_205959

-- Conditions
def num_students := 120
def num_boys := 72
def num_girls := 48
def frac_boys_absent := 1 / 8
def frac_girls_absent := 1 / 4

-- Theorem statement
theorem percent_absent : 
  ( (frac_boys_absent * num_boys + frac_girls_absent * num_girls) / num_students ) * 100 = 17.5 :=
by
  sorry

end percent_absent_l205_205959


namespace average_children_in_families_with_children_l205_205853

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205853


namespace least_four_digit_perfect_square_and_fourth_power_l205_205974

theorem least_four_digit_perfect_square_and_fourth_power : 
    ∃ (n : ℕ), (1000 ≤ n) ∧ (n < 10000) ∧ (∃ a : ℕ, n = a^2) ∧ (∃ b : ℕ, n = b^4) ∧ 
    (∀ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ a : ℕ, m = a^2) ∧ (∃ b : ℕ, m = b^4) → n ≤ m) ∧ n = 6561 :=
by
  sorry

end least_four_digit_perfect_square_and_fourth_power_l205_205974


namespace find_x_l205_205942

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end find_x_l205_205942


namespace cube_cut_problem_l205_205999

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l205_205999


namespace total_tickets_sales_l205_205025

theorem total_tickets_sales:
    let student_ticket_price := 6
    let adult_ticket_price := 8
    let number_of_students := 20
    let number_of_adults := 12
    number_of_students * student_ticket_price + number_of_adults * adult_ticket_price = 216 :=
by
    intros
    sorry

end total_tickets_sales_l205_205025


namespace water_added_l205_205172

-- Definitions and constants based on conditions
def initial_volume : ℝ := 80
def initial_jasmine_percentage : ℝ := 0.10
def jasmine_added : ℝ := 5
def final_jasmine_percentage : ℝ := 0.13

-- Problem statement
theorem water_added (W : ℝ) :
  (initial_volume * initial_jasmine_percentage + jasmine_added) / (initial_volume + jasmine_added + W) = final_jasmine_percentage → 
  W = 15 :=
by
  sorry

end water_added_l205_205172


namespace rotation_matrix_150_degrees_l205_205358

open Real

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [cos θ, -sin θ],
    [sin θ, cos θ]
  ]

theorem rotation_matrix_150_degrees :
  rotation_matrix (150 * π / 180) = ![
    [-sqrt 3 / 2, -1 / 2],
    [ 1 / 2, -sqrt 3 / 2]
  ] :=
by
  sorry

end rotation_matrix_150_degrees_l205_205358


namespace eval_at_5_l205_205695

def g (x : ℝ) : ℝ := 3 * x^4 - 8 * x^3 + 15 * x^2 - 10 * x - 75

theorem eval_at_5 : g 5 = 1125 := by
  sorry

end eval_at_5_l205_205695


namespace supplement_of_complement_of_75_degree_angle_l205_205280

def angle : ℕ := 75
def complement_angle (a : ℕ) := 90 - a
def supplement_angle (a : ℕ) := 180 - a

theorem supplement_of_complement_of_75_degree_angle : supplement_angle (complement_angle angle) = 165 :=
by
  sorry

end supplement_of_complement_of_75_degree_angle_l205_205280


namespace function_decreasing_odd_function_m_zero_l205_205080

-- First part: Prove that the function is decreasing
theorem function_decreasing (m : ℝ) (x1 x2 : ℝ) (h : x1 < x2) :
    let f := fun x => -2 * x + m
    f x1 > f x2 :=
by
    sorry

-- Second part: Find the value of m when the function is odd
theorem odd_function_m_zero (m : ℝ) :
    (∀ x : ℝ, let f := fun x => -2 * x + m
              f (-x) = -f x) → m = 0 :=
by
    sorry

end function_decreasing_odd_function_m_zero_l205_205080


namespace minimum_weights_l205_205916

variable {α : Type} [LinearOrderedField α]

theorem minimum_weights (weights : Finset α)
  (h_unique : weights.card = 5)
  (h_balanced : ∀ {x y : α}, x ∈ weights → y ∈ weights → x ≠ y →
    ∃ a b : α, a ∈ weights ∧ b ∈ weights ∧ x + y = a + b) :
  ∃ (n : ℕ), n = 13 ∧ ∀ S : Finset α, S.card = n ∧
    (∀ {x y : α}, x ∈ S → y ∈ S → x ≠ y → ∃ a b : α, a ∈ S ∧ b ∈ S ∧ x + y = a + b) :=
by
  sorry

end minimum_weights_l205_205916


namespace jessies_original_weight_l205_205108

theorem jessies_original_weight (current_weight weight_lost original_weight : ℕ) 
  (h_current: current_weight = 27) (h_lost: weight_lost = 101) 
  (h_original: original_weight = current_weight + weight_lost) : 
  original_weight = 128 :=
by
  rw [h_current, h_lost] at h_original
  exact h_original

end jessies_original_weight_l205_205108


namespace no_solution_if_and_only_if_zero_l205_205548

theorem no_solution_if_and_only_if_zero (n : ℝ) :
  ¬(∃ (x y z : ℝ), 2 * n * x + y = 2 ∧ 3 * n * y + z = 3 ∧ x + 2 * n * z = 2) ↔ n = 0 := 
  by
  sorry

end no_solution_if_and_only_if_zero_l205_205548


namespace smallest_n_transform_l205_205423

open Real

noncomputable def line1_angle : ℝ := π / 30
noncomputable def line2_angle : ℝ := π / 40
noncomputable def line_slope : ℝ := 2 / 45
noncomputable def transform_angle (theta : ℝ) (n : ℕ) : ℝ := theta + n * (7 * π / 120)

theorem smallest_n_transform (theta : ℝ) (n : ℕ) (m : ℕ)
  (h_line1 : line1_angle = π / 30)
  (h_line2 : line2_angle = π / 40)
  (h_slope : tan theta = line_slope)
  (h_transform : transform_angle theta n = theta + m * 2 * π) :
  n = 120 := 
sorry

end smallest_n_transform_l205_205423


namespace perfect_square_quadratic_l205_205659

theorem perfect_square_quadratic (a : ℝ) :
  ∃ (b : ℝ), (x : ℝ) → (x^2 - ax + 16) = (x + b)^2 ∨ (x^2 - ax + 16) = (x - b)^2 → a = 8 ∨ a = -8 :=
by
  sorry

end perfect_square_quadratic_l205_205659


namespace solution_opposite_numbers_l205_205043

theorem solution_opposite_numbers (x y : ℤ) (h1 : 2 * x + 3 * y - 4 = 0) (h2 : x = -y) : x = -4 ∧ y = 4 :=
by
  sorry

end solution_opposite_numbers_l205_205043


namespace remainder_24_2377_mod_15_l205_205975

theorem remainder_24_2377_mod_15 :
  24^2377 % 15 = 9 :=
sorry

end remainder_24_2377_mod_15_l205_205975


namespace austin_pairs_of_shoes_l205_205501

theorem austin_pairs_of_shoes (S : ℕ) :
  0.45 * (S : ℝ) + 11 = S → S / 2 = 10 :=
by
  sorry

end austin_pairs_of_shoes_l205_205501


namespace g_inv_3_l205_205461

-- Define the function g and its inverse g_inv based on the provided table.
def g : ℕ → ℕ
| 1 := 4
| 2 := 3
| 3 := 1
| 4 := 5
| 5 := 2
| _ := 0  -- arbitrary definition for other values

def g_inv : ℕ → ℕ
| 4 := 1
| 3 := 2
| 1 := 3
| 5 := 4
| 2 := 5
| _ := 0  -- arbitrary definition for other values

-- The theorem to prove the inverse property based on the given conditions
theorem g_inv_3 : g_inv (g_inv (g_inv 3)) = 4 :=
by
  -- Proof skipped using sorry
  sorry

end g_inv_3_l205_205461


namespace prime_p_is_2_l205_205275

theorem prime_p_is_2 (p q r : ℕ) 
  (hp : Prime p) (hq : Prime q) (hr : Prime r) 
  (h_sum : p + q = r) (h_lt : p < q) : 
  p = 2 :=
sorry

end prime_p_is_2_l205_205275


namespace cone_volume_l205_205183

theorem cone_volume (S r : ℝ) : 
  ∃ V : ℝ, V = (1 / 3) * S * r :=
by
  sorry

end cone_volume_l205_205183


namespace random_vars_independent_l205_205420

variable {Ω : Type*} {m n : ℕ}
variable {ξ : Fin m → Ω → ℝ} {η : Fin n → Ω → ℝ}
variable {μ : Measure Ω}

def independent_random_variables (ξ : Fin m → Ω → ℝ) : Prop :=
  ∀ i j, i ≠ j → IndependentFun (ξ i) (ξ j) μ 

theorem random_vars_independent
  (h1 : independent_random_variables ξ)
  (h2 : independent_random_variables η)
  (h3 : IndependentFun (λ ω, (λ i, ξ i ω)) (λ ω, (λ j, η j ω)) μ) :
  independent_random_variables (λ k, if h : k.val < m then ξ ⟨k, h⟩ else η ⟨k.val - m, Nat.sub_lt m (by linarith [ne_of_lt h1, h2])⟩) :=
sorry

end random_vars_independent_l205_205420


namespace men_build_fountain_l205_205677

theorem men_build_fountain (m1 m2 : ℕ) (l1 l2 d1 d2 : ℕ) (work_rate : ℚ)
  (h1 : m1 * d1 = l1 * work_rate)
  (h2 : work_rate = 56 / (20 * 7))
  (h3 : l1 = 56)
  (h4 : l2 = 42)
  (h5 : m1 = 20)
  (h6 : m2 = 35)
  (h7 : d1 = 7)
  : d2 = 3 :=
sorry

end men_build_fountain_l205_205677


namespace annual_income_calculation_l205_205756

noncomputable def annual_income (investment : ℝ) (price_per_share : ℝ) (dividend_rate : ℝ) (face_value : ℝ) : ℝ :=
  let number_of_shares := investment / price_per_share
  number_of_shares * face_value * dividend_rate

theorem annual_income_calculation :
  annual_income 4455 8.25 0.12 10 = 648 :=
by
  sorry

end annual_income_calculation_l205_205756


namespace expected_waiting_time_l205_205513

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l205_205513


namespace rotation_matrix_150_degrees_l205_205361

theorem rotation_matrix_150_degrees :
  let θ := 150 * Real.pi / 180
  let cos150 := Real.cos θ
  let sin150 := Real.sin θ
  (cos150, sin150) = (-Real.sqrt 3 / 2, 1 / 2) →
  (Matrix.vecCons (Matrix.vecCons (-Real.sqrt 3 / 2) (-1 / 2) Matrix.nil) (Matrix.vecCons (1 / 2) (-Real.sqrt 3 / 2) Matrix.nil) 
    = Matrix.vecCons (Matrix.vecCons (-Real.sqrt 3 / 2) (-1 / 2) Matrix.nil) (Matrix.vecCons (1 / 2) (-Real.sqrt 3 / 2) Matrix.nil)) := by
  sorry

end rotation_matrix_150_degrees_l205_205361


namespace suitcase_lock_settings_l205_205629

-- Define the number of settings for each dial choice considering the conditions
noncomputable def first_digit_choices : ℕ := 9
noncomputable def second_digit_choices : ℕ := 9
noncomputable def third_digit_choices : ℕ := 8
noncomputable def fourth_digit_choices : ℕ := 7

-- Theorem to prove the total number of different settings
theorem suitcase_lock_settings : first_digit_choices * second_digit_choices * third_digit_choices * fourth_digit_choices = 4536 :=
by sorry

end suitcase_lock_settings_l205_205629


namespace store_profit_l205_205168

theorem store_profit 
  (cost_per_item : ℕ)
  (selling_price_decrease : ℕ → ℕ)
  (profit : ℤ)
  (x : ℕ) :
  cost_per_item = 40 →
  (∀ x, selling_price_decrease x = 150 - 5 * (x - 50)) →
  profit = 1500 →
  (((x = 50 ∧ selling_price_decrease 50 = 150) ∨ (x = 70 ∧ selling_price_decrease 70 = 50)) ↔ (x = 50 ∨ x = 70) ∧ profit = 1500) :=
by
  sorry

end store_profit_l205_205168


namespace total_spent_on_date_l205_205320

-- Constants representing costs
def ticket_cost : ℝ := 10.00
def combo_meal_cost : ℝ := 11.00
def candy_cost : ℝ := 2.50

-- Numbers of items to buy
def num_tickets : ℝ := 2
def num_candies : ℝ := 2

-- Total cost calculation
def total_cost : ℝ := (ticket_cost * num_tickets) + (candy_cost * num_candies) + combo_meal_cost

-- Prove that the total cost is $36.00
theorem total_spent_on_date : total_cost = 36.00 := by
  sorry

end total_spent_on_date_l205_205320


namespace arrangement_count_PERSEVERANCE_l205_205645

theorem arrangement_count_PERSEVERANCE : 
  let count := 12!
  let repeat_E := 3!
  let repeat_R := 2!
  count / (repeat_E * repeat_R) = 39916800 :=
by
  sorry

end arrangement_count_PERSEVERANCE_l205_205645


namespace solution_set_of_inequality_l205_205243

theorem solution_set_of_inequality (f : ℝ → ℝ)
  (h_tangent : ∀ x₀ y₀, y₀ = f x₀ → (∀ x, f x = y₀ + (3*x₀^2 - 6*x₀)*(x - x₀)))
  (h_at_3 : f 3 = 0) :
  {x : ℝ | ((x - 1) / f x) ≥ 0} = {x : ℝ | x < 0} ∪ {x : ℝ | 0 < x ∧ x ≤ 1} ∪ {x : ℝ | x > 3} :=
sorry

end solution_set_of_inequality_l205_205243


namespace waiters_hired_l205_205633

theorem waiters_hired (W H : ℕ) (h1 : 3 * W = 90) (h2 : 3 * (W + H) = 126) : H = 12 :=
sorry

end waiters_hired_l205_205633


namespace factorize_expression_l205_205184

variable (a : ℝ) -- assuming a is a real number

theorem factorize_expression (a : ℝ) : a^2 + 3 * a = a * (a + 3) :=
by
  -- proof goes here
  sorry

end factorize_expression_l205_205184


namespace initial_cakes_count_l205_205064

theorem initial_cakes_count (f : ℕ) (a b : ℕ) 
  (condition1 : f = 5)
  (condition2 : ∀ i, i ∈ Finset.range f → a = 4)
  (condition3 : ∀ i, i ∈ Finset.range f → b = 20 / 2)
  (condition4 : f * a = 2 * b) : 
  b = 40 := 
by
  sorry

end initial_cakes_count_l205_205064


namespace trajectory_of_center_of_moving_circle_l205_205037

theorem trajectory_of_center_of_moving_circle (M : ℝ × ℝ) :
  (∀ (M : ℝ × ℝ), (∃ r > 0, ((M.2 - 2) = r ∧ dist (M.1, M.2) (0, -3) = r + 1))) →
  (∃ y : ℝ, (M.1)^2 = -12 * y) :=
by
  sorry

end trajectory_of_center_of_moving_circle_l205_205037


namespace cubic_equation_root_sum_l205_205697

theorem cubic_equation_root_sum (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p * q + p * r + q * r = 11) (h3 : p * q * r = 6) :
  (p * q / r + p * r / q + q * r / p) = 49 / 6 := sorry

end cubic_equation_root_sum_l205_205697


namespace AB_eq_B_exp_V_l205_205891

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l205_205891


namespace find_x_l205_205129

-- conditions
variable (k : ℝ)
variable (x : ℝ)
variable (y : ℝ)
variable (z : ℝ)

-- proportional relationship
def proportional_relationship (k x y z : ℝ) : Prop := 
  x = (k * y^2) / z

-- initial conditions
def initial_conditions (k : ℝ) : Prop := 
  proportional_relationship k 6 1 3

-- prove x = 24 when y = 2 and z = 3 under given conditions
theorem find_x (k : ℝ) (h : initial_conditions k) : 
  proportional_relationship k 24 2 3 :=
sorry

end find_x_l205_205129


namespace find_expression_value_l205_205072

theorem find_expression_value (x : ℝ) (h : x^2 - 5*x = 14) : 
  (x-1)*(2*x-1) - (x+1)^2 + 1 = 15 := 
by 
  sorry

end find_expression_value_l205_205072


namespace inverse_of_g_compose_three_l205_205464

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Assuming g(x) is defined only for x in {1, 2, 3, 4, 5}

noncomputable def g_inv (y : ℕ) : ℕ :=
  match y with
  | 4 => 1
  | 3 => 2
  | 1 => 3
  | 5 => 4
  | 2 => 5
  | _ => 0  -- Assuming g_inv(y) is defined only for y in {1, 3, 1, 5, 2}

theorem inverse_of_g_compose_three : g_inv (g_inv (g_inv 3)) = 4 := by
  sorry

end inverse_of_g_compose_three_l205_205464


namespace rotation_matrix_150_eq_l205_205346

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ], ![sin θ, cos θ]]

noncomputable def angle_150 : ℝ := 150 * (π / 180) -- Convert 150 degrees to radians
  
theorem rotation_matrix_150_eq :
  rotation_matrix angle_150 = ![![-(sqrt 3 / 2), -1 / 2], ![1 / 2, -(sqrt 3 / 2)]] :=
by
  simp [rotation_matrix, angle_150]
  sorry

end rotation_matrix_150_eq_l205_205346


namespace sufficient_but_not_necessary_l205_205444

theorem sufficient_but_not_necessary (a b : ℝ) :
  (a > 2 ∧ b > 2) → (a + b > 4 ∧ a * b > 4) ∧ ¬((a + b > 4 ∧ a * b > 4) → (a > 2 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_l205_205444


namespace find_q_value_l205_205980

theorem find_q_value (q : ℚ) (x y : ℚ) (hx : x = 5 - q) (hy : y = 3*q - 1) : x = 3*y → q = 4/5 :=
by
  sorry

end find_q_value_l205_205980


namespace new_total_lines_is_240_l205_205578

-- Define the original number of lines, the increase, and the percentage increase
variables (L : ℝ) (increase : ℝ := 110) (percentage_increase : ℝ := 84.61538461538461 / 100)

-- The statement to prove
theorem new_total_lines_is_240 (h : increase = percentage_increase * L) : L + increase = 240 := sorry

end new_total_lines_is_240_l205_205578


namespace average_children_in_families_with_children_l205_205840

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205840


namespace quadratic_roots_sum_product_l205_205727

theorem quadratic_roots_sum_product (m n : ℝ) (h1 : m / 2 = 10) (h2 : n / 2 = 24) : m + n = 68 :=
by
  sorry

end quadratic_roots_sum_product_l205_205727


namespace no_eleven_points_achieve_any_score_l205_205685

theorem no_eleven_points (x y : ℕ) : 3 * x + 7 * y ≠ 11 := 
sorry

theorem achieve_any_score (S : ℕ) (h : S ≥ 12) : ∃ (x y : ℕ), 3 * x + 7 * y = S :=
sorry

end no_eleven_points_achieve_any_score_l205_205685


namespace distance_covered_l205_205169

/-- 
Given the following conditions:
1. The speed of Abhay (A) is 5 km/h.
2. The time taken by Abhay to cover a distance is 2 hours more than the time taken by Sameer.
3. If Abhay doubles his speed, then he would take 1 hour less than Sameer.
Prove that the distance (D) they are covering is 30 kilometers.
-/
theorem distance_covered (D S : ℝ) (A : ℝ) (hA : A = 5) 
  (h1 : D / A = D / S + 2) 
  (h2 : D / (2 * A) = D / S - 1) : 
  D = 30 := by
    sorry

end distance_covered_l205_205169


namespace commute_time_late_l205_205775

theorem commute_time_late (S : ℝ) (T : ℝ) (T' : ℝ) (H1 : T = 1) (H2 : T' = (4/3)) :
  T' - T = 20 / 60 :=
by
  sorry

end commute_time_late_l205_205775


namespace average_difference_is_7_l205_205597

/-- The differences between Mia's and Liam's study times for each day in one week -/
def daily_differences : List ℤ := [15, -5, 25, 0, -15, 20, 10]

/-- The number of days in a week -/
def number_of_days : ℕ := 7

/-- The total difference over the week -/
def total_difference : ℤ := daily_differences.sum

/-- The average difference per day -/
def average_difference_per_day : ℚ := total_difference / number_of_days

theorem average_difference_is_7 : average_difference_per_day = 7 := by 
  sorry

end average_difference_is_7_l205_205597


namespace hiking_rate_l205_205612

theorem hiking_rate (rate_uphill: ℝ) (time_total: ℝ) (time_uphill: ℝ) (rate_downhill: ℝ) 
  (h1: rate_uphill = 4) (h2: time_total = 3) (h3: time_uphill = 1.2) : rate_downhill = 4.8 / (time_total - time_uphill) :=
by
  sorry

end hiking_rate_l205_205612


namespace fixed_monthly_charge_l205_205479

variables (F C_J : ℝ)

-- Conditions
def january_bill := F + C_J = 46
def february_bill := F + 2 * C_J = 76

-- The proof goal
theorem fixed_monthly_charge
  (h_jan : january_bill F C_J)
  (h_feb : february_bill F C_J)
  (h_calls : C_J = 30) : F = 16 :=
by sorry

end fixed_monthly_charge_l205_205479


namespace number_of_winning_scores_l205_205094

-- Define the problem conditions
variable (n : ℕ) (team1_scores team2_scores : Finset ℕ)

-- Define the total number of runners
def total_runners := 12

-- Define the sum of placements
def sum_placements : ℕ := (total_runners * (total_runners + 1)) / 2

-- Define the threshold for the winning score
def winning_threshold : ℕ := sum_placements / 2

-- Define the minimum score for a team
def min_score : ℕ := 1 + 2 + 3 + 4 + 5 + 6

-- Prove that the number of different possible winning scores is 19
theorem number_of_winning_scores : 
  Finset.card (Finset.range (winning_threshold + 1) \ Finset.range min_score) = 19 :=
by
  sorry -- Proof to be filled in

end number_of_winning_scores_l205_205094


namespace avg_children_with_kids_l205_205828

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205828


namespace average_children_in_families_with_children_l205_205855

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205855


namespace expected_waiting_time_for_first_bite_l205_205515

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l205_205515


namespace binomial_expansion_a5_l205_205675

theorem binomial_expansion_a5 (x : ℝ) 
  (a_0 a_1 a_2 a_3 a_4 a_5 a_6 a_7 a_8 : ℝ) 
  (h : (x - 1) ^ 8 = a_0 + a_1 * (1 + x) + a_2 * (1 + x) ^ 2 + a_3 * (1 + x) ^ 3 + a_4 * (1 + x) ^ 4 + a_5 * (1 + x) ^ 5 + a_6 * (1 + x) ^ 6 + a_7 * (1 + x) ^ 7 + a_8 * (1 + x) ^ 8) : 
  a_5 = -448 := 
sorry

end binomial_expansion_a5_l205_205675


namespace probability_two_crack_code_code_more_likely_cracked_l205_205274

noncomputable def probability_two_people_crack_code : ℚ :=
  let P_A1 := 1/5
  let P_A2 := 1/4
  let P_A3 := 1/3
  let P_not_A3 := 1 - P_A3
  let P_not_A2 := 1 - P_A2
  let P_not_A1 := 1 - P_A1
  P_A1 * P_A2 * P_not_A3 + 
  P_A1 * P_not_A2 * P_A3 + 
  P_not_A1 * P_A2 * P_A3

theorem probability_two_crack_code :
  probability_two_people_crack_code = 3/20 :=
sorry

noncomputable def probability_code_not_cracked : ℚ :=
  let P_A1 := 1/5
  let P_A2 := 1/4
  let P_A3 := 1/3
  let P_not_A3 := 1 - P_A3
  let P_not_A2 := 1 - P_A2
  let P_not_A1 := 1 - P_A1
  P_not_A1 * P_not_A2 * P_not_A3

noncomputable def probability_code_cracked : ℚ :=
  1 - probability_code_not_cracked

theorem code_more_likely_cracked :
  probability_code_cracked > probability_code_not_cracked :=
sorry

end probability_two_crack_code_code_more_likely_cracked_l205_205274


namespace tree_growth_rate_consistency_l205_205706

theorem tree_growth_rate_consistency (a b : ℝ) :
  (a + b) / 2 = 0.15 ∧ (1 + a) * (1 + b) = 0.90 → ∃ a b : ℝ, (a + b) / 2 = 0.15 ∧ (1 + a) * (1 + b) = 0.90 := by
  sorry

end tree_growth_rate_consistency_l205_205706


namespace angle_sum_around_point_l205_205145

theorem angle_sum_around_point {x : ℝ} (h : 2 * x + 210 = 360) : x = 75 :=
by
  sorry

end angle_sum_around_point_l205_205145


namespace integral_of_2x_plus_e_to_x_l205_205179

theorem integral_of_2x_plus_e_to_x :
  ∫ x in 0..1, (2 * x + Real.exp x) = Real.exp 1 :=
by
  sorry

end integral_of_2x_plus_e_to_x_l205_205179


namespace sale_in_second_month_l205_205774

theorem sale_in_second_month 
  (m1 m2 m3 m4 m5 m6 : ℕ) 
  (h1: m1 = 6335) 
  (h2: m3 = 6855) 
  (h3: m4 = 7230) 
  (h4: m5 = 6562) 
  (h5: m6 = 5091)
  (average: (m1 + m2 + m3 + m4 + m5 + m6) / 6 = 6500) : 
  m2 = 6927 :=
sorry

end sale_in_second_month_l205_205774


namespace ratio_of_expenditures_l205_205726

theorem ratio_of_expenditures 
  (income_Uma : ℕ) (income_Bala : ℕ) (expenditure_Uma : ℕ) (expenditure_Bala : ℕ)
  (h_ratio_incomes : income_Uma / income_Bala = 4 / 3)
  (h_savings_Uma : income_Uma - expenditure_Uma = 5000)
  (h_savings_Bala : income_Bala - expenditure_Bala = 5000)
  (h_income_Uma : income_Uma = 20000) :
  expenditure_Uma / expenditure_Bala = 3 / 2 :=
sorry

end ratio_of_expenditures_l205_205726


namespace value_of_p_l205_205450

noncomputable def third_term (x y : ℝ) := 45 * x^8 * y^2
noncomputable def fourth_term (x y : ℝ) := 120 * x^7 * y^3

theorem value_of_p (p q : ℝ) (h1 : third_term p q = fourth_term p q) (h2 : p + 2 * q = 1) (h3 : 0 < p) (h4 : 0 < q) : p = 4 / 7 :=
by
  have h : third_term p q = 45 * p^8 * q^2 := rfl
  have h' : fourth_term p q = 120 * p^7 * q^3 := rfl
  rw [h, h'] at h1
  sorry

end value_of_p_l205_205450


namespace pencils_and_notebooks_cost_l205_205021

theorem pencils_and_notebooks_cost
    (p n : ℝ)
    (h1 : 8 * p + 10 * n = 5.36)
    (h2 : 12 * (p - 0.05) + 5 * n = 4.05) :
    15 * (p - 0.05) + 12 * n = 7.01 := 
sorry

end pencils_and_notebooks_cost_l205_205021


namespace remainder_13_pow_2031_mod_100_l205_205473

theorem remainder_13_pow_2031_mod_100 : (13^2031) % 100 = 17 :=
by sorry

end remainder_13_pow_2031_mod_100_l205_205473


namespace max_value_inequality_am_gm_inequality_l205_205664

-- Given conditions and goals as Lean statements
theorem max_value_inequality (x : ℝ) : (|x - 1| + |x - 2| ≥ 1) := sorry

theorem am_gm_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h : (1/a) + (1/(2*b)) + (1/(3*c)) = 1) : (a + 2*b + 3*c) ≥ 9 := sorry

end max_value_inequality_am_gm_inequality_l205_205664


namespace least_number_subtracted_l205_205282

theorem least_number_subtracted (n m1 m2 m3 r : ℕ) (h_n : n = 642) (h_m1 : m1 = 11) (h_m2 : m2 = 13) (h_m3 : m3 = 17) (h_r : r = 4) :
  ∃ x : ℕ, (n - x) % m1 = r ∧ (n - x) % m2 = r ∧ (n - x) % m3 = r ∧ n - x = 638 :=
sorry

end least_number_subtracted_l205_205282


namespace least_small_barrels_l205_205781

theorem least_small_barrels (total_oil : ℕ) (large_barrel : ℕ) (small_barrel : ℕ) (L S : ℕ)
  (h1 : total_oil = 745) (h2 : large_barrel = 11) (h3 : small_barrel = 7)
  (h4 : 11 * L + 7 * S = 745) (h5 : total_oil - 11 * L = 7 * S) : S = 1 :=
by
  sorry

end least_small_barrels_l205_205781


namespace find_D_l205_205252

-- We define the points E and F
def E : ℝ × ℝ := (-3, -2)
def F : ℝ × ℝ := (5, 10)

-- Definition of point D with the given conditions
def D : ℝ × ℝ := (3, 7)

-- We state the main theorem to prove that D is such that ED = 2 * DF given E and F
theorem find_D (D : ℝ × ℝ) (ED_DF_relation : dist E D = 2 * dist D F) : D = (3, 7) :=
sorry

end find_D_l205_205252


namespace identify_stolen_treasure_l205_205031

-- Define the magic square arrangement
def magic_square (bags : ℕ → ℕ) :=
  bags 0 + bags 1 + bags 2 = 15 ∧
  bags 3 + bags 4 + bags 5 = 15 ∧
  bags 6 + bags 7 + bags 8 = 15 ∧
  bags 0 + bags 3 + bags 6 = 15 ∧
  bags 1 + bags 4 + bags 7 = 15 ∧
  bags 2 + bags 5 + bags 8 = 15 ∧
  bags 0 + bags 4 + bags 8 = 15 ∧
  bags 2 + bags 4 + bags 6 = 15

-- Define the stolen treasure detection function
def stolen_treasure (bags : ℕ → ℕ) : Prop :=
  ∃ altered_bag_idx : ℕ, (bags altered_bag_idx ≠ altered_bag_idx + 1)

-- The main theorem
theorem identify_stolen_treasure (bags : ℕ → ℕ) (h_magic_square : magic_square bags) : ∃ altered_bag_idx : ℕ, stolen_treasure bags :=
sorry

end identify_stolen_treasure_l205_205031


namespace line_slope_intercept_l205_205447

theorem line_slope_intercept (a b : ℝ) 
  (h1 : (7 : ℝ) = a * 3 + b) 
  (h2 : (13 : ℝ) = a * (9/2) + b) : 
  a - b = 9 := 
sorry

end line_slope_intercept_l205_205447


namespace greatest_integer_difference_l205_205209

theorem greatest_integer_difference (x y : ℤ) (h1 : 5 < x ∧ x < 8) (h2 : 8 < y ∧ y < 13)
  (h3 : x % 3 = 0) (h4 : y % 3 = 0) : y - x = 6 :=
sorry

end greatest_integer_difference_l205_205209


namespace weight_of_new_person_l205_205719

theorem weight_of_new_person 
  (average_weight_first_20 : ℕ → ℕ → ℕ)
  (new_average_weight : ℕ → ℕ → ℕ) 
  (total_weight_21 : ℕ): 
  (average_weight_first_20 1200 20 = 60) → 
  (new_average_weight (1200 + total_weight_21) 21 = 55) → 
  total_weight_21 = 55 := 
by 
  intros 
  sorry

end weight_of_new_person_l205_205719


namespace convert_speed_72_kmph_to_mps_l205_205332

theorem convert_speed_72_kmph_to_mps :
  let kmph := 72
  let factor_km_to_m := 1000
  let factor_hr_to_s := 3600
  (kmph * factor_km_to_m) / factor_hr_to_s = 20 := by
  -- (72 kmph * (1000 meters / 1 kilometer)) / (3600 seconds / 1 hour) = 20 meters per second
  sorry

end convert_speed_72_kmph_to_mps_l205_205332


namespace binomial_equality_l205_205813

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l205_205813


namespace suitcase_lock_combinations_l205_205630

-- Define the conditions of the problem as Lean definitions.
def first_digit_possibilities : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9}
def remaining_digits (used: Finset ℕ) : Finset ℕ :=
  {0, 1, 2, 3, 4, 5, 6, 7, 8, 9} \ used

-- The actual proof problem
theorem suitcase_lock_combinations : 
  ∃ combinations : ℕ,
    combinations = 9 * 9 * 8 * 7 ∧ combinations = 4536 :=
by
  use 4536
  split
  ·
    simp
    norm_num
  ·
    rfl

end suitcase_lock_combinations_l205_205630


namespace ones_digit_of_73_pow_355_l205_205905

theorem ones_digit_of_73_pow_355 : (73 ^ 355) % 10 = 7 := 
  sorry

end ones_digit_of_73_pow_355_l205_205905


namespace percentage_increase_in_spending_l205_205091

variables (P Q : ℝ)
-- Conditions
def price_increase (P : ℝ) := 1.25 * P
def quantity_decrease (Q : ℝ) := 0.88 * Q

-- Mathemtically equivalent proof problem in Lean:
theorem percentage_increase_in_spending (P Q : ℝ) : 
  (price_increase P) * (quantity_decrease Q) / (P * Q) = 1.10 :=
by
  sorry

end percentage_increase_in_spending_l205_205091


namespace not_divisible_by_pow_two_l205_205253

theorem not_divisible_by_pow_two (n : ℕ) (h : n > 1) : ¬ (2^n ∣ (3^n + 1)) :=
by
  sorry

end not_divisible_by_pow_two_l205_205253


namespace book_pages_l205_205032

theorem book_pages (P : ℕ) 
  (h1 : P / 2 + 11 + (P - (P / 2 + 11)) / 2 = 19)
  (h2 : P - (P / 2 + 11) = 2 * 19) : 
  P = 98 :=
by
  sorry

end book_pages_l205_205032


namespace sandy_books_from_second_shop_l205_205588

noncomputable def books_from_second_shop (books_first: ℕ) (cost_first: ℕ) (cost_second: ℕ) (avg_price: ℕ): ℕ :=
  let total_cost := cost_first + cost_second
  let total_books := books_first + (total_cost / avg_price) - books_first
  total_cost / avg_price - books_first

theorem sandy_books_from_second_shop :
  books_from_second_shop 65 1380 900 19 = 55 :=
by
  sorry

end sandy_books_from_second_shop_l205_205588


namespace metal_sheets_per_panel_l205_205297

-- Define the given conditions
def num_panels : ℕ := 10
def rods_per_sheet : ℕ := 10
def rods_per_beam : ℕ := 4
def beams_per_panel : ℕ := 2
def total_rods_needed : ℕ := 380

-- Question translated to Lean statement
theorem metal_sheets_per_panel (S : ℕ) (h : 10 * (10 * S + 8) = 380) : S = 3 := 
  sorry

end metal_sheets_per_panel_l205_205297


namespace joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l205_205938

-- Defining the weights of Joe's lifts
variable (J1 J2 : ℕ)

-- Conditions for Joe
def joe_conditions : Prop :=
  (J1 + J2 = 900) ∧ (2 * J1 = J2 + 300)

-- Defining the weights of Mike's lifts
variable (M1 M2 : ℕ)

-- Conditions for Mike  
def mike_conditions : Prop :=
  (M1 + M2 = 1100) ∧ (M2 = M1 + 200)

-- Defining the weights of Lisa's lifts
variable (L1 L2 : ℕ)

-- Conditions for Lisa  
def lisa_conditions : Prop :=
  (L1 + L2 = 1000) ∧ (L1 = 3 * L2)

-- Proof statements
theorem joe_first_lift_is_400 (h : joe_conditions J1 J2) : J1 = 400 :=
by
  sorry

theorem mike_first_lift_is_450 (h : mike_conditions M1 M2) : M1 = 450 :=
by
  sorry

theorem lisa_second_lift_is_250 (h : lisa_conditions L1 L2) : L2 = 250 :=
by
  sorry

end joe_first_lift_is_400_mike_first_lift_is_450_lisa_second_lift_is_250_l205_205938


namespace digit_sum_eq_21_l205_205175

theorem digit_sum_eq_21 (A B C D: ℕ) (h1: A ≠ 0) 
    (h2: (A * 10 + B) * 100 + (C * 10 + D) = (C * 10 + D)^2 - (A * 10 + B)^2) 
    (hA: A < 10) (hB: B < 10) (hC: C < 10) (hD: D < 10) : 
    A + B + C + D = 21 :=
by 
  sorry

end digit_sum_eq_21_l205_205175


namespace members_of_groups_l205_205441

variable {x y : ℕ}

theorem members_of_groups (h1 : x = y + 10) (h2 : x - 1 = 2 * (y + 1)) :
  x = 17 ∧ y = 7 :=
by
  sorry

end members_of_groups_l205_205441


namespace probability_of_at_least_10_heads_l205_205746

open ProbabilityTheory

noncomputable def probability_at_least_10_heads_in_12_flips : ℚ :=
  let total_outcomes := (2 : ℕ) ^ 12 in
  let ways_10_heads := Nat.choose 12 10 in
  let ways_11_heads := Nat.choose 12 11 in
  let ways_12_heads := Nat.choose 12 12 in
  let heads_ways := ways_10_heads + ways_11_heads + ways_12_heads in
  (heads_ways : ℚ) / (total_outcomes : ℚ)

theorem probability_of_at_least_10_heads :
  probability_at_least_10_heads_in_12_flips = 79 / 4096 := sorry

end probability_of_at_least_10_heads_l205_205746


namespace sum_of_reciprocals_of_roots_l205_205906

theorem sum_of_reciprocals_of_roots {r1 r2 : ℚ} (h1 : r1 + r2 = 15) (h2 : r1 * r2 = 6) :
  (1 / r1 + 1 / r2) = 5 / 2 := 
by sorry

end sum_of_reciprocals_of_roots_l205_205906


namespace slices_per_person_eq_three_l205_205137

variables (num_people : ℕ) (slices_per_pizza : ℕ) (num_pizzas : ℕ)

theorem slices_per_person_eq_three (h1 : num_people = 18) (h2 : slices_per_pizza = 9) (h3 : num_pizzas = 6) : 
  (num_pizzas * slices_per_pizza) / num_people = 3 :=
sorry

end slices_per_person_eq_three_l205_205137


namespace probability_S4_positive_l205_205620

open ProbabilityTheory

-- Definitions
def fair_coin (n : ℕ) : Distribution (ℕ → bool) :=
  λ f, ∀ i < n, P(f i = tt) = 1 / 2 ∧ P(f i = ff) = 1 / 2

def a (f : ℕ → bool) (n : ℕ) : ℤ :=
  if f n then 1 else -1

noncomputable def S (f : ℕ → bool) (n : ℕ) : ℤ :=
  ∑ i in finset.range n, a f i

-- The proposition to prove
theorem probability_S4_positive : 
  (∑' (f : ℕ → bool), if S f 4 > 0 then 1 else 0) / (∑' (f : ℕ → bool), 1) = 5 / 16 :=
sorry

end probability_S4_positive_l205_205620


namespace correct_expression_after_removing_parentheses_l205_205475

variable (a b c : ℝ)

theorem correct_expression_after_removing_parentheses :
  -2 * (a + b - 3 * c) = -2 * a - 2 * b + 6 * c :=
sorry

end correct_expression_after_removing_parentheses_l205_205475


namespace initial_skittles_geq_16_l205_205245

variable (S : ℕ) -- S represents the total number of Skittles Lillian had initially
variable (L : ℕ) -- L represents the number of Skittles Lillian kept as leftovers

theorem initial_skittles_geq_16 (h1 : S = 8 * 2 + L) : S ≥ 16 :=
by
  sorry

end initial_skittles_geq_16_l205_205245


namespace vanessa_savings_remaining_l205_205471

-- Conditions
def initial_investment : ℝ := 50000
def annual_interest_rate : ℝ := 0.035
def investment_duration : ℕ := 3
def conversion_rate : ℝ := 0.85
def cost_per_toy : ℝ := 75

-- Given the above conditions, prove the remaining amount in euros after buying as many toys as possible is 16.9125
theorem vanessa_savings_remaining
  (P : ℝ := initial_investment)
  (r : ℝ := annual_interest_rate)
  (t : ℕ := investment_duration)
  (c : ℝ := conversion_rate)
  (e : ℝ := cost_per_toy) :
  (((P * (1 + r)^t) * c) - (e * (⌊(P * (1 + r)^3 * 0.85) / e⌋))) = 16.9125 :=
sorry

end vanessa_savings_remaining_l205_205471


namespace expected_value_of_winnings_l205_205120

def is_prime (n : ℕ) : Prop :=
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_perfect_square (n : ℕ) : Prop :=
  n = 1 ∨ n = 4

def winnings (n : ℕ) : ℤ :=
  if is_prime n then n
  else if is_perfect_square n then -n
  else 0

def expected_value : ℚ :=
  (1 / 2 : ℚ) * (2 + 3 + 5 + 7) - (1 / 4 : ℚ) * (1 + 4) + (1 / 4 : ℚ) * 0

theorem expected_value_of_winnings : expected_value = 29 / 4 := 
  sorry

end expected_value_of_winnings_l205_205120


namespace calculate_total_area_l205_205519

theorem calculate_total_area :
  let height1 := 7
  let width1 := 6
  let width2 := 4
  let height2 := 5
  let height3 := 1
  let width3 := 2
  let width4 := 5
  let height4 := 6
  let area1 := width1 * height1
  let area2 := width2 * height2
  let area3 := height3 * width3
  let area4 := width4 * height4
  area1 + area2 + area3 + area4 = 94 := by
  sorry

end calculate_total_area_l205_205519


namespace count_ns_divisible_by_5_l205_205191

open Nat

theorem count_ns_divisible_by_5 : 
  let f (n : ℕ) := 2 * n^5 + 2 * n^4 + 3 * n^2 + 3 
  ∃ (N : ℕ), N = 19 ∧ 
  (∀ (n : ℕ), 2 ≤ n ∧ n ≤ 100 → f n % 5 = 0 → 
  (∃ (m : ℕ), 1 ≤ m ∧ m ≤ 19 ∧ n = 5 * m + 1)) :=
by
  sorry

end count_ns_divisible_by_5_l205_205191


namespace division_by_fraction_l205_205310

theorem division_by_fraction (a b : ℝ) (hb : b ≠ 0) : a / (1 / b) = a * b :=
by {
  sorry
}

example : 12 / (1 / 6) = 72 :=
by {
  exact division_by_fraction 12 6 (by norm_num),
}

end division_by_fraction_l205_205310


namespace chocolate_flavored_cups_sold_l205_205776

-- Define total sales and fractions
def total_cups_sold : ℕ := 50
def fraction_winter_melon : ℚ := 2 / 5
def fraction_okinawa : ℚ := 3 / 10
def fraction_chocolate : ℚ := 1 - (fraction_winter_melon + fraction_okinawa)

-- Define the number of chocolate-flavored cups sold
def num_chocolate_cups_sold : ℕ := 50 - (50 * 2 / 5 + 50 * 3 / 10)

-- Main theorem statement
theorem chocolate_flavored_cups_sold : num_chocolate_cups_sold = 15 := 
by 
  -- The proof would go here, but we use 'sorry' to skip it
  sorry

end chocolate_flavored_cups_sold_l205_205776


namespace train_length_is_120_l205_205995

noncomputable def length_of_train (speed_kmh : ℝ) (time_s : ℝ) (bridge_length_m : ℝ) : ℝ :=
  let speed_ms := (speed_kmh * 1000) / 3600
  let total_distance := speed_ms * time_s
  total_distance - bridge_length_m

theorem train_length_is_120 :
  length_of_train 70 13.884603517432893 150 = 120 :=
by
  sorry

end train_length_is_120_l205_205995


namespace binom_18_4_eq_3060_l205_205798

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l205_205798


namespace count_integer_values_satisfying_condition_l205_205012

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l205_205012


namespace find_function_l205_205239

noncomputable def f (x : ℝ) : ℝ := sorry 

theorem find_function (f : ℝ → ℝ)
  (cond : ∀ x y z : ℝ, x + y + z = 0 → f (x^3) + (f y)^3 + (f z)^3 = 3 * x * y * z) :
  ∀ x : ℝ, f x = x :=
by
  sorry

end find_function_l205_205239


namespace exactly_one_correct_proposition_l205_205696

variables (l1 l2 : Line) (alpha : Plane)

-- Definitions for the conditions
def perpendicular_lines (l1 l2 : Line) : Prop := -- definition of perpendicular lines
sorry

def perpendicular_to_plane (l : Line) (alpha : Plane) : Prop := -- definition of line perpendicular to plane
sorry

def line_in_plane (l : Line) (alpha : Plane) : Prop := -- definition of line in a plane
sorry

-- Problem statement
theorem exactly_one_correct_proposition 
  (h1 : perpendicular_lines l1 l2) 
  (h2 : perpendicular_to_plane l1 alpha) 
  (h3 : line_in_plane l2 alpha) : 
  (¬(perpendicular_lines l1 l2 ∧ perpendicular_to_plane l1 alpha → line_in_plane l2 alpha) ∧
   ¬(perpendicular_lines l1 l2 ∧ line_in_plane l2 alpha → perpendicular_to_plane l1 alpha) ∧
   (perpendicular_to_plane l1 alpha ∧ line_in_plane l2 alpha → perpendicular_lines l1 l2)) :=
sorry

end exactly_one_correct_proposition_l205_205696


namespace sale_in_fifth_month_l205_205490

theorem sale_in_fifth_month 
  (sale_month_1 : ℕ) (sale_month_2 : ℕ) (sale_month_3 : ℕ) (sale_month_4 : ℕ) 
  (sale_month_6 : ℕ) (average_sale : ℕ) 
  (h1 : sale_month_1 = 5266) (h2 : sale_month_2 = 5744) (h3 : sale_month_3 = 5864) 
  (h4 : sale_month_4 = 6122) (h6 : sale_month_6 = 4916) (h_avg : average_sale = 5750) :
  ∃ sale_month_5, sale_month_5 = 6588 :=
by
  sorry

end sale_in_fifth_month_l205_205490


namespace isosceles_triangle_base_length_l205_205721

theorem isosceles_triangle_base_length (a b : ℕ) (h1 : a = 7) (h2 : b + 2 * a = 25) : b = 11 := by
  sorry

end isosceles_triangle_base_length_l205_205721


namespace find_x_value_l205_205766

theorem find_x_value :
  ∃ x : ℝ, (75 * x + (18 + 12) * 6 / 4 - 11 * 8 = 2734) ∧ (x = 37.03) :=
by {
  sorry
}

end find_x_value_l205_205766


namespace stereographic_projection_reflection_l205_205126

noncomputable def sphere : Type := sorry
noncomputable def point_on_sphere (P : sphere) : Prop := sorry
noncomputable def reflection_on_sphere (P P' : sphere) (e : sphere) : Prop := sorry
noncomputable def arbitrary_point (E : sphere) (P P' : sphere) : Prop := E ≠ P ∧ E ≠ P'
noncomputable def tangent_plane (E : sphere) : Type := sorry
noncomputable def stereographic_projection (E : sphere) (δ : Type) : sphere → sorry := sorry
noncomputable def circle_on_plane (e : sphere) (E : sphere) (δ : Type) : Type := sorry
noncomputable def inversion_in_circle (P P' : sphere) (e_1 : Type) : Prop := sorry

theorem stereographic_projection_reflection (P P' E : sphere) (e : sphere) (δ : Type) (e_1 : Type) :
  point_on_sphere P ∧
  reflection_on_sphere P P' e ∧
  arbitrary_point E P P' ∧
  circle_on_plane e E δ = e_1 →
  inversion_in_circle P P' e_1 :=
sorry

end stereographic_projection_reflection_l205_205126


namespace triangle_is_right_l205_205687

variable {n : ℕ}

theorem triangle_is_right 
  (h1 : n > 1) 
  (h2 : a = 2 * n) 
  (h3 : b = n^2 - 1) 
  (h4 : c = n^2 + 1)
  : a^2 + b^2 = c^2 := 
by
  -- skipping the proof
  sorry

end triangle_is_right_l205_205687


namespace arithmetic_evaluation_l205_205636

theorem arithmetic_evaluation : (10 - 9^2 + 8 * 7 + 6^2 - 5 * 4 + 3 - 2^3) = -4 :=
by
  sorry

end arithmetic_evaluation_l205_205636


namespace min_sum_of_dimensions_l205_205446

theorem min_sum_of_dimensions (a b c : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 2310) :
  a + b + c = 42 :=
sorry

end min_sum_of_dimensions_l205_205446


namespace gdp_scientific_notation_l205_205558

theorem gdp_scientific_notation :
  (121 * 10^12 : ℝ) = 1.21 * 10^14 := by
  sorry

end gdp_scientific_notation_l205_205558


namespace find_values_l205_205521

theorem find_values (x y z : ℝ) :
  (x + y + z = 1) →
  (x^2 * y + y^2 * z + z^2 * x = x * y^2 + y * z^2 + z * x^2) →
  (x^3 + y^2 + z = y^3 + z^2 + x) →
  ( (x = 1/3 ∧ y = 1/3 ∧ z = 1/3) ∨ 
    (x = 0 ∧ y = 0 ∧ z = 1) ∨
    (x = 2/3 ∧ y = -1/3 ∧ z = 2/3) ∨
    (x = 0 ∧ y = 1 ∧ z = 0) ∨
    (x = 1 ∧ y = 0 ∧ z = 0) ∨
    (x = -1 ∧ y = 1 ∧ z = 1) ) := 
sorry

end find_values_l205_205521


namespace comm_delegates_room_pairing_l205_205295

theorem comm_delegates_room_pairing :
  (∃ (delegates : Fin 1000 → Type) (can_communicate : Type → Type → Prop),
    (∀ (a b c : Fin 1000), ∃ x y : Fin 1000, x ≠ y ∧ can_communicate x y) →
    ∃ (pairs : list (Type × Type)), 
      (∀ (p : Type × Type), p ∈ pairs → can_communicate p.1 p.2) ∧ 
      list.length pairs = 500) :=
sorry

end comm_delegates_room_pairing_l205_205295


namespace range_of_a_l205_205668

def p (a x : ℝ) : Prop := a * x^2 + a * x - 1 < 0
def q (a : ℝ) : Prop := (3 / (a - 1)) + 1 < 0

theorem range_of_a (a : ℝ) :
  ¬ (∀ x, p a x ∨ q a) → a ≤ -4 ∨ 1 ≤ a :=
by sorry

end range_of_a_l205_205668


namespace ants_total_l205_205304

namespace Ants

-- Defining the number of ants each child finds based on the given conditions
def Abe_ants := 4
def Beth_ants := Abe_ants + Abe_ants
def CeCe_ants := 3 * Abe_ants
def Duke_ants := Abe_ants / 2
def Emily_ants := Abe_ants + (3 * Abe_ants / 4)
def Frances_ants := 2 * CeCe_ants

-- The total number of ants found by the six children
def total_ants := Abe_ants + Beth_ants + CeCe_ants + Duke_ants + Emily_ants + Frances_ants

-- The statement to prove
theorem ants_total: total_ants = 57 := by
  sorry

end Ants

end ants_total_l205_205304


namespace num_int_values_x_l205_205007

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l205_205007


namespace value_of_y_at_3_l205_205943

-- Define the function
def f (x : ℕ) : ℕ := 2 * x^2 + 1

-- Prove that when x = 3, y = 19
theorem value_of_y_at_3 : f 3 = 19 :=
by
  -- Provide the definition and conditions
  let x := 3
  let y := f x
  have h : y = 2 * x^2 + 1 := rfl
  -- State the actual proof could go here
  sorry

end value_of_y_at_3_l205_205943


namespace max_two_terms_eq_one_l205_205127

theorem max_two_terms_eq_one (a b c x y z : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c) (h4 : x ≠ y) (h5 : y ≠ z) (h6 : x ≠ z) :
  ∀ (P : ℕ → ℝ), -- Define P(i) as given expressions
  ((P 1 = a * x + b * y + c * z) ∧
   (P 2 = a * x + b * z + c * y) ∧
   (P 3 = a * y + b * x + c * z) ∧
   (P 4 = a * y + b * z + c * x) ∧
   (P 5 = a * z + b * x + c * y) ∧
   (P 6 = a * z + b * y + c * x)) →
  (P 1 = 1 ∨ P 2 = 1 ∨ P 3 = 1 ∨ P 4 = 1 ∨ P 5 = 1 ∨ P 6 = 1) →
  (∃ i j, i ≠ j ∧ P i = 1 ∧ P j = 1) →
  ¬(∃ i j k, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ P i = 1 ∧ P j = 1 ∧ P k = 1) :=
sorry

end max_two_terms_eq_one_l205_205127


namespace avg_of_7_consecutive_integers_l205_205421

theorem avg_of_7_consecutive_integers (c d : ℝ) (h : d = (c + (c + 1) + (c + 2) + (c + 3) + (c + 4) + (c + 5)) / 6) :
  ((d + (d + 1) + (d + 2) + (d + 3) + (d + 4) + (d + 5) + (d + 6)) / 7) = c + 5.5 := by
  sorry

end avg_of_7_consecutive_integers_l205_205421


namespace evaluate_expression_l205_205939

variables {a b c d e : ℝ}

theorem evaluate_expression (a b c d e : ℝ) : a * b^c - d + e = a * (b^c - (d + e)) :=
by
  sorry

end evaluate_expression_l205_205939


namespace expenditure_on_house_rent_l205_205309

theorem expenditure_on_house_rent (I : ℝ) (H1 : 0.30 * I = 300) : 0.20 * (I - 0.30 * I) = 140 :=
by
  -- Skip the proof, the statement is sufficient at this stage.
  sorry

end expenditure_on_house_rent_l205_205309


namespace candidate_p_wage_difference_l205_205981

theorem candidate_p_wage_difference
  (P Q : ℝ)    -- Candidate p's hourly wage is P, Candidate q's hourly wage is Q
  (H : ℝ)      -- Candidate p's working hours
  (total_payment : ℝ)
  (wage_ratio : P = 1.5 * Q)  -- Candidate p is paid 50% more per hour than candidate q
  (hours_diff : Q * (H + 10) = total_payment)  -- Candidate q's total payment equation
  (candidate_q_payment : Q * (H + 10) = 480)   -- total payment for candidate q
  (candidate_p_payment : 1.5 * Q * H = 480)    -- total payment for candidate p
  : P - Q = 8 := sorry

end candidate_p_wage_difference_l205_205981


namespace cost_per_candy_bar_l205_205566

-- Define the conditions as hypotheses
variables (candy_bars_total : ℕ) (candy_bars_paid_by_dave : ℕ) (amount_paid_by_john : ℝ)
-- Assume the given values
axiom total_candy_bars : candy_bars_total = 20
axiom candy_bars_by_dave : candy_bars_paid_by_dave = 6
axiom paid_by_john : amount_paid_by_john = 21

-- Define the proof problem
theorem cost_per_candy_bar :
  (amount_paid_by_john / (candy_bars_total - candy_bars_paid_by_dave) = 1.50) :=
by
  sorry

end cost_per_candy_bar_l205_205566


namespace find_a2010_l205_205105

noncomputable def seq (n : ℕ) : ℕ :=
if n = 1 then 2 else if n = 2 then 3 else
  (seq (n - 1) * seq (n - 2)) % 10

theorem find_a2010 : seq 2010 = 4 :=
sorry

end find_a2010_l205_205105


namespace avg_children_in_families_with_children_l205_205837

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205837


namespace number_of_dogs_per_box_l205_205177

-- Definition of the problem
def num_boxes : ℕ := 7
def total_dogs : ℕ := 28

-- Statement of the theorem to prove
theorem number_of_dogs_per_box (x : ℕ) (h : num_boxes * x = total_dogs) : x = 4 :=
by
  sorry

end number_of_dogs_per_box_l205_205177


namespace average_children_in_families_with_children_l205_205860

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205860


namespace binom_18_4_l205_205797

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l205_205797


namespace max_value_abc_l205_205698

theorem max_value_abc (a b c : ℝ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
(h_sum : a + b + c = 3) : 
  a^2 * b^3 * c^4 ≤ 2048 / 19683 :=
sorry

end max_value_abc_l205_205698


namespace value_of_expression_l205_205200

theorem value_of_expression (a b c d m : ℝ) (h1 : a = -b) (h2 : a ≠ 0) (h3 : c * d = 1) (h4 : |m| = 3) :
  m^2 - (-1) + |a + b| - c * d * m = 7 ∨ m^2 - (-1) + |a + b| - c * d * m = 13 :=
by
  sorry

end value_of_expression_l205_205200


namespace expected_waiting_time_correct_l205_205509

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l205_205509


namespace build_wall_time_l205_205233

theorem build_wall_time {d : ℝ} : 
  (15 * 1 + 3 * 2) * 3 = 63 ∧ 
  (25 * 1 + 5 * 2) * d = 63 → 
  d = 1.8 := 
by 
  sorry

end build_wall_time_l205_205233


namespace total_value_is_correct_l205_205731

-- Define the conditions from the problem
def totalCoins : Nat := 324
def twentyPaiseCoins : Nat := 220
def twentyPaiseValue : Nat := 20
def twentyFivePaiseValue : Nat := 25
def paiseToRupees : Nat := 100

-- Calculate the number of 25 paise coins
def twentyFivePaiseCoins : Nat := totalCoins - twentyPaiseCoins

-- Calculate the total value of 20 paise and 25 paise coins in paise
def totalValueInPaise : Nat :=
  (twentyPaiseCoins * twentyPaiseValue) + 
  (twentyFivePaiseCoins * twentyFivePaiseValue)

-- Convert the total value from paise to rupees
def totalValueInRupees : Nat := totalValueInPaise / paiseToRupees

-- The theorem to be proved
theorem total_value_is_correct : totalValueInRupees = 70 := by
  sorry

end total_value_is_correct_l205_205731


namespace house_painting_cost_l205_205110

theorem house_painting_cost :
  let judson_contrib := 500.0
  let kenny_contrib_euros := judson_contrib * 1.2 / 1.1
  let camilo_contrib_pounds := (kenny_contrib_euros * 1.1 + 200.0) / 1.3
  let camilo_contrib_usd := camilo_contrib_pounds * 1.3
  judson_contrib + kenny_contrib_euros * 1.1 + camilo_contrib_usd = 2020.0 := 
by {
  sorry
}

end house_painting_cost_l205_205110


namespace tan_alpha_sub_60_l205_205539

theorem tan_alpha_sub_60 
  (alpha : ℝ) 
  (h : Real.tan alpha = 4 * Real.sin (420 * Real.pi / 180)) : 
  Real.tan (alpha - 60 * Real.pi / 180) = (Real.sqrt 3) / 7 :=
by sorry

end tan_alpha_sub_60_l205_205539


namespace find_coordinates_of_P_l205_205932

-- Define the conditions
variable (x y : ℝ)
def in_second_quadrant := x < 0 ∧ y > 0
def distance_to_x_axis := abs y = 7
def distance_to_y_axis := abs x = 3

-- Define the statement to be proved in Lean 4
theorem find_coordinates_of_P :
  in_second_quadrant x y ∧ distance_to_x_axis y ∧ distance_to_y_axis x → (x, y) = (-3, 7) :=
by
  sorry

end find_coordinates_of_P_l205_205932


namespace Yvettes_final_bill_l205_205563

namespace IceCreamShop

def sundae_price_Alicia : Real := 7.50
def sundae_price_Brant : Real := 10.00
def sundae_price_Josh : Real := 8.50
def sundae_price_Yvette : Real := 9.00
def tip_rate : Real := 0.20

theorem Yvettes_final_bill :
  let total_cost := sundae_price_Alicia + sundae_price_Brant + sundae_price_Josh + sundae_price_Yvette
  let tip := tip_rate * total_cost
  let final_bill := total_cost + tip
  final_bill = 42.00 :=
by
  -- calculations are skipped here
  sorry

end IceCreamShop

end Yvettes_final_bill_l205_205563


namespace avg_children_with_kids_l205_205825

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205825


namespace palindrome_probability_divisible_by_7_l205_205301

-- Define the conditions
def is_four_digit_palindrome (n : ℕ) : Prop :=
  ∃ (a b : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0 ≤ b ∧ b ≤ 9) ∧ n = 1001 * a + 110 * b

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

-- Define the proof problem
theorem palindrome_probability_divisible_by_7 : 
  (∃ (n : ℕ), is_four_digit_palindrome n ∧ is_divisible_by_7 n) →
  ∃ p : ℚ, p = 1/5 :=
sorry

end palindrome_probability_divisible_by_7_l205_205301


namespace mail_handling_in_six_months_l205_205455

theorem mail_handling_in_six_months (daily_letters daily_packages days_per_month months : ℕ) :
  daily_letters = 60 →
  daily_packages = 20 →
  days_per_month = 30 →
  months = 6 →
  (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  -- Skipping the proof
  sorry

end mail_handling_in_six_months_l205_205455


namespace prove_min_max_A_l205_205290

theorem prove_min_max_A : 
  ∃ (A_max A_min : ℕ), 
  (∃ B : ℕ, 
    A_max = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧
    B % 10 = 9) ∧ 
  (∃ B : ℕ, 
    A_min = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧ 
    B % 10 = 1) ∧ 
  A_max = 999999998 ∧ 
  A_min = 166666667 := sorry

end prove_min_max_A_l205_205290


namespace hyperbola_equation_l205_205208

theorem hyperbola_equation
  (a b : ℝ) 
  (a_pos : a > 0) 
  (b_pos : b > 0) 
  (focus_at_five : a^2 + b^2 = 25) 
  (asymptote_ratio : b / a = 3 / 4) :
  (a = 4 ∧ b = 3 ∧ ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1) ↔ ( ∀ x y : ℝ, x^2 / 16 - y^2 / 9 = 1 ):=
sorry 

end hyperbola_equation_l205_205208


namespace problem_1_problem_2_l205_205545

def f (x a : ℝ) : ℝ := abs (2 * x - a) + abs (2 * x + 3)
def g (x : ℝ) : ℝ := abs (2 * x - 3) + 2

theorem problem_1 (x : ℝ) :
  abs (g x) < 5 → 0 < x ∧ x < 3 :=
sorry

theorem problem_2 (a : ℝ) :
  (∀ x1 : ℝ, ∃ x2 : ℝ, f x1 a = g x2) →
  (a ≥ -1 ∨ a ≤ -5) :=
sorry

end problem_1_problem_2_l205_205545


namespace sequence_property_l205_205920

def sequence_conditions (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 3 ∧
  ∀ n ≥ 3, S n + S (n - 2) = 2 * S (n - 1) + n

theorem sequence_property (a : ℕ → ℕ) (S : ℕ → ℕ) (h : sequence_conditions a S) : 
  ∀ n ≥ 3, a n = a (n - 1) + n :=
  sorry

end sequence_property_l205_205920


namespace inequality_comparison_l205_205912

theorem inequality_comparison 
  (a : ℝ) (b : ℝ) (c : ℝ) 
  (h₁ : a = (1 / Real.log 3 / Real.log 2))
  (h₂ : b = Real.exp 0.5)
  (h₃ : c = Real.log 2) :
  b > c ∧ c > a := 
by
  sorry

end inequality_comparison_l205_205912


namespace infinite_castle_hall_unique_l205_205305

theorem infinite_castle_hall_unique :
  (∀ (n : ℕ), ∃ hall : ℕ, ∀ m : ℕ, ((m = 2 * n + 1) ∨ (m = 3 * n + 1)) → hall = m) →
  ∀ (hall1 hall2 : ℕ), hall1 = hall2 :=
by
  sorry

end infinite_castle_hall_unique_l205_205305


namespace find_a_of_min_value_of_f_l205_205388

noncomputable def f (a x : ℝ) : ℝ := 4 * Real.sin (2 * x) + 3 * Real.cos (2 * x) + 2 * a * Real.sin x + 4 * a * Real.cos x

theorem find_a_of_min_value_of_f :
  (∃ a : ℝ, (∀ x : ℝ, f a x ≥ -6) ∧ (∃ x : ℝ, f a x = -6)) → (a = Real.sqrt 2 ∨ a = -Real.sqrt 2) :=
by
  sorry

end find_a_of_min_value_of_f_l205_205388


namespace polynomial_solution_l205_205335

theorem polynomial_solution (P : Polynomial ℝ) (h1 : P.eval 0 = 0) (h2 : ∀ x : ℝ, P.eval (x^2 + 1) = (P.eval x)^2 + 1) : 
  ∀ x : ℝ, P.eval x = x :=
by
  sorry

end polynomial_solution_l205_205335


namespace proof_f_values_l205_205952

def f (x : ℤ) : ℤ :=
  if x < 0 then
    2 * x + 7
  else
    x^2 - 2

theorem proof_f_values :
  f (-2) = 3 ∧ f (3) = 7 :=
by
  sorry

end proof_f_values_l205_205952


namespace line_not_tangent_if_only_one_common_point_l205_205448

theorem line_not_tangent_if_only_one_common_point (l p : ℝ) :
  (∃ y, y^2 = 2 * p * l) ∧ ¬ (∃ x : ℝ, y = l ∧ y^2 = 2 * p * x) := 
  sorry

end line_not_tangent_if_only_one_common_point_l205_205448


namespace avg_children_in_families_with_children_l205_205873

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205873


namespace percentage_of_Y_salary_l205_205278

variable (X Y : ℝ)
variable (total_salary Y_salary : ℝ)
variable (P : ℝ)

theorem percentage_of_Y_salary :
  total_salary = 638 ∧ Y_salary = 290 ∧ X = (P / 100) * Y_salary → P = 120 := by
  sorry

end percentage_of_Y_salary_l205_205278


namespace eval_fraction_l205_205822

theorem eval_fraction (a b : ℕ) : (40 : ℝ) = 2^3 * 5 → (10 : ℝ) = 2 * 5 → (40^56 / 10^28) = 160^28 :=
by 
  sorry

end eval_fraction_l205_205822


namespace change_is_24_l205_205124

-- Define the prices and quantities
def price_basketball_card : ℕ := 3
def price_baseball_card : ℕ := 4
def num_basketball_cards : ℕ := 2
def num_baseball_cards : ℕ := 5
def money_paid : ℕ := 50

-- Define the total cost
def total_cost : ℕ := (num_basketball_cards * price_basketball_card) + (num_baseball_cards * price_baseball_card)

-- Define the change received
def change_received : ℕ := money_paid - total_cost

-- Prove that the change received is $24
theorem change_is_24 : change_received = 24 := by
  -- the proof will go here
  sorry

end change_is_24_l205_205124


namespace max_xy_value_l205_205658

theorem max_xy_value {x y : ℝ} (h : 2 * x + y = 1) : ∃ z, z = x * y ∧ z = 1 / 8 :=
by sorry

end max_xy_value_l205_205658


namespace average_children_in_families_with_children_l205_205842

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205842


namespace reflect_over_y_axis_l205_205342

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℝ :=
  !![-1, 0;
      0, 1]

def v1 : Matrix (Fin 2) (Fin 1) ℝ := 
  !![1; 
      0]

def v2 : Matrix (Fin 2) (Fin 1) ℝ := 
  !![0; 
      1]

def reflectY1 : Matrix (Fin 2) (Fin 1) ℝ := 
  !![-1; 
      0]

def reflectY2 : Matrix (Fin 2) (Fin 1) ℝ := 
  !![0; 
      1]

theorem reflect_over_y_axis :
  (A ⬝ v1 = reflectY1) ∧ (A ⬝ v2 = reflectY2) := 
  sorry

end reflect_over_y_axis_l205_205342


namespace find_A_n_find_d1_d2_zero_l205_205199

-- Defining the arithmetic sequences {a_n} and {b_n} with common differences d1 and d2 respectively
variables (a b : ℕ → ℤ)
variables (d1 d2 : ℤ)

-- Conditions on the sequences
axiom a_n_arith : ∀ n, a (n + 1) = a n + d1
axiom b_n_arith : ∀ n, b (n + 1) = b n + d2

-- Definitions of A_n and B_n
def A_n (n : ℕ) : ℤ := a n + b n
def B_n (n : ℕ) : ℤ := a n * b n

-- Given initial conditions
axiom A_1 : A_n a b 1 = 1
axiom A_2 : A_n a b 2 = 3

-- Prove that A_n = 2n - 1
theorem find_A_n : ∀ n, A_n a b n = 2 * n - 1 :=
by sorry

-- Condition that B_n is an arithmetic sequence
axiom B_n_arith : ∀ n, B_n a b (n + 1) - B_n a b n = B_n a b 1 - B_n a b 0

-- Prove that d1 * d2 = 0
theorem find_d1_d2_zero : d1 * d2 = 0 :=
by sorry

end find_A_n_find_d1_d2_zero_l205_205199


namespace average_children_in_families_with_children_l205_205856

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205856


namespace contribution_required_l205_205611

-- Definitions corresponding to the problem statement
def total_amount : ℝ := 2000
def number_of_friends : ℝ := 7
def your_contribution_factor : ℝ := 2

-- Prove that the amount each friend needs to raise is approximately 222.22
theorem contribution_required (x : ℝ) 
  (h : 9 * x = total_amount) :
  x = 2000 / 9 := 
  by sorry

end contribution_required_l205_205611


namespace sally_initial_peaches_l205_205710

section
variables 
  (peaches_after : ℕ)
  (peaches_picked : ℕ)
  (initial_peaches : ℕ)

theorem sally_initial_peaches 
    (h1 : peaches_picked = 42)
    (h2 : peaches_after = 55)
    (h3 : peaches_after = initial_peaches + peaches_picked) : 
    initial_peaches = 13 := 
by 
  sorry
end

end sally_initial_peaches_l205_205710


namespace find_quadruples_l205_205526

def quadrupleSolution (a b c d : ℝ): Prop :=
  (a * (b + c) = b * (c + d) ∧ b * (c + d) = c * (d + a) ∧ c * (d + a) = d * (a + b))

def isSolution (a b c d : ℝ): Prop :=
  (a = 1 ∧ b = 0 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 1 ∧ c = 0 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 1 ∧ d = 0) ∨
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 1) ∨
  (a = 1 ∧ b = 1 ∧ c = 1 ∧ d = 1) ∨
  (a = 1 ∧ b = -1 ∧ c = 1 ∧ d = -1) ∨
  (a = 1 ∧ b = -1 + Real.sqrt 2 ∧ c = -1 ∧ d = 1 - Real.sqrt 2) ∨
  (a = 1 ∧ b = -1 - Real.sqrt 2 ∧ c = -1 ∧ d = 1 + Real.sqrt 2)

theorem find_quadruples (a b c d : ℝ) :
  quadrupleSolution a b c d ↔ isSolution a b c d :=
sorry

end find_quadruples_l205_205526


namespace avg_children_in_families_with_children_l205_205874

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205874


namespace gcd_sixPn_n_minus_2_l205_205376

def nthSquarePyramidalNumber (n : ℕ) : ℕ :=
  (n * (n + 1) * (2 * n + 1)) / 6

def sixPn (n : ℕ) : ℕ :=
  n * (n + 1) * (2 * n + 1)

theorem gcd_sixPn_n_minus_2 (n : ℕ) (h_pos : 0 < n) : Int.gcd (sixPn n) (n - 2) ≤ 12 :=
by
  sorry

end gcd_sixPn_n_minus_2_l205_205376


namespace ambiguous_dates_count_l205_205026

theorem ambiguous_dates_count : 
  ∃ n : ℕ, n = 132 ∧ ∀ d m : ℕ, 1 ≤ d ∧ d ≤ 31 ∧ 1 ≤ m ∧ m ≤ 12 →
  ((d ≥ 1 ∧ d ≤ 12 ∧ m ≥ 1 ∧ m ≤ 12) → n = 132)
  :=
by 
  let ambiguous_days := 12 * 12
  let non_ambiguous_days := 12
  let total_ambiguous := ambiguous_days - non_ambiguous_days
  use total_ambiguous
  sorry

end ambiguous_dates_count_l205_205026


namespace find_principal_l205_205763

-- Define the conditions
def interest_rate : ℝ := 0.05
def time_period : ℕ := 10
def interest_less_than_principal : ℝ := 3100

-- Define the principal
def principal : ℝ := 6200

-- The theorem statement
theorem find_principal :
  ∃ P : ℝ, P - interest_less_than_principal = P * interest_rate * time_period ∧ P = principal :=
by
  sorry

end find_principal_l205_205763


namespace reb_min_biking_speed_l205_205709

theorem reb_min_biking_speed (driving_time_minutes driving_speed driving_distance biking_distance_minutes biking_reduction_percentage biking_distance_hours : ℕ) 
  (driving_time_eqn: driving_time_minutes = 45) 
  (driving_speed_eqn: driving_speed = 40) 
  (driving_distance_eqn: driving_distance = driving_speed * driving_time_minutes / 60)
  (biking_reduction_percentage_eqn: biking_reduction_percentage = 20)
  (biking_distance_eqn: biking_distance = driving_distance * (100 - biking_reduction_percentage) / 100)
  (biking_distance_hours_eqn: biking_distance_minutes = 120)
  (biking_hours_eqn: biking_distance_hours = biking_distance_minutes / 60)
  : (biking_distance / biking_distance_hours) ≥ 12 := 
by
  sorry

end reb_min_biking_speed_l205_205709


namespace ramsey_theorem_six_people_l205_205708

theorem ramsey_theorem_six_people (S : Finset Person)
  (hS: S.card = 6)
  (R : Person → Person → Prop): 
  (∃ (has_relation : Person → Person → Prop), 
    ∀ A B : Person, A ≠ B → R A B ∨ ¬ R A B) →
  (∃ (T : Finset Person), T.card = 3 ∧ 
    ((∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → R x y) ∨ 
     (∀ x y : Person, x ∈ T → y ∈ T → x ≠ y → ¬ R x y))) :=
by
  sorry

end ramsey_theorem_six_people_l205_205708


namespace cube_root_squared_l205_205146

noncomputable def solve_for_x (x : ℝ) : Prop :=
  (x^(1/3))^2 = 81 → x = 729

theorem cube_root_squared (x : ℝ) :
  solve_for_x x :=
by
  sorry

end cube_root_squared_l205_205146


namespace find_x_for_given_y_l205_205603

theorem find_x_for_given_y (x y : ℝ) (h_pos : 0 < x ∧ 0 < y) (h_initial : x = 2 ∧ y = 8) (h_inverse : (2 ^ 3) * 8 = 128) :
  y = 1728 → x = (1 / (13.5) ^ (1 / 3)) :=
by
  sorry

end find_x_for_given_y_l205_205603


namespace largest_inscribed_triangle_area_l205_205508

theorem largest_inscribed_triangle_area 
  (radius : ℝ) 
  (diameter : ℝ)
  (base : ℝ)
  (height : ℝ) 
  (area : ℝ)
  (h1 : radius = 10)
  (h2 : diameter = 2 * radius)
  (h3 : base = diameter)
  (h4 : height = radius) 
  (h5 : area = (1/2) * base * height) : 
  area  = 100 :=
by 
  have h_area := (1/2) * 20 * 10
  sorry

end largest_inscribed_triangle_area_l205_205508


namespace percent_of_whole_is_fifty_l205_205158

theorem percent_of_whole_is_fifty (part whole : ℝ) (h1 : part = 180) (h2 : whole = 360) : 
  ((part / whole) * 100) = 50 := 
by 
  rw [h1, h2] 
  sorry

end percent_of_whole_is_fifty_l205_205158


namespace remainder_when_divided_by_44_l205_205779

theorem remainder_when_divided_by_44 (N Q R : ℕ) :
  (N = 44 * 432 + R) ∧ (N = 39 * Q + 15) → R = 0 :=
by
  sorry

end remainder_when_divided_by_44_l205_205779


namespace ones_digit_of_largest_power_of_3_dividing_factorial_l205_205190

theorem ones_digit_of_largest_power_of_3_dividing_factorial (n : ℕ) (h : 27 = 3^3) : 
  (fun x => x % 10) (3^13) = 3 := by
  sorry

end ones_digit_of_largest_power_of_3_dividing_factorial_l205_205190


namespace wood_length_equation_l205_205683

theorem wood_length_equation (x : ℝ) :
  (1 / 2) * (x + 4.5) = x - 1 :=
sorry

end wood_length_equation_l205_205683


namespace average_children_in_families_with_children_l205_205883

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205883


namespace min_value_fraction_l205_205529

theorem min_value_fraction (x : ℝ) (h : x > 9) : 
  (∃ y, y > 9 ∧ (∀ z, z > 9 → y ≤ (z^3 / (z - 9)))) ∧ (∀ z, z > 9 → (∃ w, w > 9 ∧ z^3 / (z - 9) = 325)) := 
  sorry

end min_value_fraction_l205_205529


namespace gear_revolutions_l205_205638

variable (r_p : ℝ) 

theorem gear_revolutions (h1 : 40 * (1 / 6) = r_p * (1 / 6) + 5) : r_p = 10 := 
by
  sorry

end gear_revolutions_l205_205638


namespace binom_18_4_l205_205807

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l205_205807


namespace arnold_danny_age_l205_205500

theorem arnold_danny_age (x : ℕ) (h : (x + 1) * (x + 1) = x * x + 13) : x = 6 :=
by {
  sorry
}

end arnold_danny_age_l205_205500


namespace find_x_plus_y_l205_205928

theorem find_x_plus_y (x y : ℝ) (h1 : |x| - x + y = 13) (h2 : x - |y| + y = 7) : x + y = 20 := 
by
  sorry

end find_x_plus_y_l205_205928


namespace find_number_l205_205153

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 11) : x = 5.5 :=
by
  sorry

end find_number_l205_205153


namespace heart_op_ratio_l205_205666

def heart_op (n m : ℕ) : ℕ := n^3 * m^2

theorem heart_op_ratio : heart_op 3 5 / heart_op 5 3 = 5 / 9 := 
by 
  sorry

end heart_op_ratio_l205_205666


namespace option_C_qualified_l205_205041

-- Define the acceptable range
def lower_bound : ℝ := 25 - 0.2
def upper_bound : ℝ := 25 + 0.2

-- Define the option to be checked
def option_C : ℝ := 25.1

-- The theorem stating that option C is within the acceptable range
theorem option_C_qualified : lower_bound ≤ option_C ∧ option_C ≤ upper_bound := 
by 
  sorry

end option_C_qualified_l205_205041


namespace problem1_problem2_problem3_l205_205195

-- Definition of the polynomial expansion
def poly (x : ℝ) := (1 - 2*x)^7

-- Definitions capturing the conditions directly
def a_0 := 1
def sum_a_1_to_a_7 := -2
def sum_a_1_3_5_7 := -1094
def sum_abs_a_0_to_a_7 := 2187

-- Lean statements for the proof problems
theorem problem1 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = sum_a_1_to_a_7 :=
sorry

theorem problem2 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  a 1 + a 3 + a 5 + a 7 = sum_a_1_3_5_7 :=
sorry

theorem problem3 (x : ℝ) (a : Fin 8 → ℝ) (h : poly x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6 + a 7 * x^7) :
  abs (a 0) + abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) + abs (a 6) + abs (a 7) = sum_abs_a_0_to_a_7 :=
sorry

end problem1_problem2_problem3_l205_205195


namespace integer_satisfying_values_l205_205019

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l205_205019


namespace telephone_number_A_value_l205_205494

theorem telephone_number_A_value :
  ∃ A B C D E F G H I J : ℕ,
    A > B ∧ B > C ∧
    D > E ∧ E > F ∧
    G > H ∧ H > I ∧ I > J ∧
    (D = E + 1) ∧ (E = F + 1) ∧
    G + H + I + J = 20 ∧
    A + B + C = 15 ∧
    A = 8 := sorry

end telephone_number_A_value_l205_205494


namespace no_real_solutions_l205_205213

theorem no_real_solutions (x : ℝ) : (x - 3 * x + 7)^2 + 1 ≠ -|x| :=
by
  -- The statement of the theorem is sufficient; the proof is not needed as per indicated instructions.
  sorry

end no_real_solutions_l205_205213


namespace fencing_required_l205_205628

theorem fencing_required (L W : ℕ) (hL : L = 30) (hArea : L * W = 720) : L + 2 * W = 78 :=
by
  sorry

end fencing_required_l205_205628


namespace chocolate_milk_tea_sales_l205_205777

theorem chocolate_milk_tea_sales (total_sales : ℕ) (winter_melon_ratio : ℚ) (okinawa_ratio : ℚ) :
  total_sales = 50 →
  winter_melon_ratio = 2 / 5 →
  okinawa_ratio = 3 / 10 →
  ∃ (chocolate_sales : ℕ), chocolate_sales = total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio ∧ chocolate_sales = 15 :=
by
  intro h1 h2 h3
  use (total_sales - total_sales * winter_melon_ratio - total_sales * okinawa_ratio).to_nat
  split
  · simp [h1, h2, h3]
  · exact sorry

end chocolate_milk_tea_sales_l205_205777


namespace total_spent_on_date_l205_205319

-- Constants representing costs
def ticket_cost : ℝ := 10.00
def combo_meal_cost : ℝ := 11.00
def candy_cost : ℝ := 2.50

-- Numbers of items to buy
def num_tickets : ℝ := 2
def num_candies : ℝ := 2

-- Total cost calculation
def total_cost : ℝ := (ticket_cost * num_tickets) + (candy_cost * num_candies) + combo_meal_cost

-- Prove that the total cost is $36.00
theorem total_spent_on_date : total_cost = 36.00 := by
  sorry

end total_spent_on_date_l205_205319


namespace both_selected_prob_l205_205277

def ram_prob : ℚ := 6 / 7
def ravi_prob : ℚ := 1 / 5

theorem both_selected_prob : ram_prob * ravi_prob = 6 / 35 := 
by
  sorry

end both_selected_prob_l205_205277


namespace a_not_multiple_of_5_l205_205203

theorem a_not_multiple_of_5 (a : ℤ) (h : a % 5 ≠ 0) : (a^4 + 4) % 5 = 0 :=
sorry

end a_not_multiple_of_5_l205_205203


namespace max_value_of_a_plus_b_l205_205228

theorem max_value_of_a_plus_b (a b : ℝ) (h₁ : a^2 + b^2 = 25) (h₂ : a ≤ 3) (h₃ : b ≥ 3) :
  a + b ≤ 7 :=
sorry

end max_value_of_a_plus_b_l205_205228


namespace calculate_angles_and_side_l205_205029

theorem calculate_angles_and_side (a b B : ℝ) (h_a : a = Real.sqrt 3) (h_b : b = Real.sqrt 2) (h_B : B = 45) :
  ∃ A C c, (A = 60 ∧ C = 75 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨ (A = 120 ∧ C = 15 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2) :=
by sorry

end calculate_angles_and_side_l205_205029


namespace sufficient_but_not_necessary_condition_subset_condition_l205_205422

open Set

variable (a : ℝ)
def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -1-2*a ≤ x ∧ x ≤ a-2}

theorem sufficient_but_not_necessary_condition (H : ∃ x ∈ A, x ∉ B a) : a ≥ 7 := sorry

theorem subset_condition (H : B a ⊆ A) : a < 1/3 := sorry

end sufficient_but_not_necessary_condition_subset_condition_l205_205422


namespace grisha_cross_coloring_l205_205193

open Nat

theorem grisha_cross_coloring :
  let grid_size := 40
  let cutout_rect_width := 36
  let cutout_rect_height := 37
  let total_cells := grid_size * grid_size
  let cutout_cells := cutout_rect_width * cutout_rect_height
  let remaining_cells := total_cells - cutout_cells
  let cross_cells := 5
  -- the result we need to prove is 113
  (remaining_cells - cross_cells - ((cutout_rect_width + cutout_rect_height - 1) - 1)) = 113 := by
  sorry

end grisha_cross_coloring_l205_205193


namespace find_n_l205_205188

theorem find_n (n : ℤ) (h1 : 1 ≤ n) (h2 : n ≤ 9) (h3 : n % 10 = -245 % 10) : n = 5 := 
  sorry

end find_n_l205_205188


namespace probability_fourth_ball_black_l205_205033

theorem probability_fourth_ball_black :
  let total_balls := 6
  let red_balls := 3
  let black_balls := 3
  let prob_black_first_draw := black_balls / total_balls
  (prob_black_first_draw = 1 / 2) ->
  (prob_black_first_draw = (black_balls / total_balls)) ->
  (black_balls / total_balls = 1 / 2) ->
  1 / 2 = 1 / 2 :=
by
  intros
  sorry

end probability_fourth_ball_black_l205_205033


namespace integer_solutions_count_l205_205003

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l205_205003


namespace mother_daughter_age_relation_l205_205023

theorem mother_daughter_age_relation (x : ℕ) (hc1 : 43 - x = 5 * (11 - x)) : x = 3 := 
sorry

end mother_daughter_age_relation_l205_205023


namespace expected_waiting_time_first_bite_l205_205517

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l205_205517


namespace integer_satisfying_values_l205_205020

theorem integer_satisfying_values (x : ℝ) :
  4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5 → 3 :=
by
  sorry

end integer_satisfying_values_l205_205020


namespace problem_statement_l205_205662

noncomputable def f (m x : ℝ) := (m-1) * Real.log x + m * x^2 + 1

theorem problem_statement (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ → x₂ > 0 → f m x₁ - f m x₂ > 2 * (x₁ - x₂)) ↔ 
  m ≥ (1 + Real.sqrt 3) / 2 :=
sorry

end problem_statement_l205_205662


namespace mr_william_land_percentage_l205_205823

def total_tax_collected : ℝ := 3840
def mr_william_tax_paid : ℝ := 480
def expected_percentage : ℝ := 12.5

theorem mr_william_land_percentage :
  (mr_william_tax_paid / total_tax_collected) * 100 = expected_percentage := 
sorry

end mr_william_land_percentage_l205_205823


namespace integer_values_satisfying_sqrt_inequality_l205_205015

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l205_205015


namespace reflectionYMatrixCorrect_l205_205343

open Matrix

-- Definitions for the basis vectors
def e1 : Matrix (Fin 2) (Fin 1) ℝ := ![![1], ![0]]
def e2 : Matrix (Fin 2) (Fin 1) ℝ := ![![0], ![1]]

-- Definition for the transformation matrix we need to find
noncomputable def reflectionYMatrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), 0], ![0, (1 : ℝ)]]

-- Statement of the theorem
theorem reflectionYMatrixCorrect :
  ∀ (x y : ℝ), reflectionYMatrix.mulVec ![x, y] = ![-x, y] := by
  sorry

end reflectionYMatrixCorrect_l205_205343


namespace cost_per_piece_l205_205424

-- Definitions based on the problem conditions
def total_cost : ℕ := 80         -- Total cost is $80
def num_pizzas : ℕ := 4          -- Luigi bought 4 pizzas
def pieces_per_pizza : ℕ := 5    -- Each pizza was cut into 5 pieces

-- Main theorem statement proving the cost per piece
theorem cost_per_piece :
  (total_cost / (num_pizzas * pieces_per_pizza)) = 4 :=
by
  sorry

end cost_per_piece_l205_205424


namespace expression_evaluation_l205_205396

theorem expression_evaluation (a b c d : ℝ) 
  (h₁ : a + b = 0) 
  (h₂ : c * d = 1) : 
  (a + b)^2 - 3 * (c * d)^4 = -3 := 
by
  -- Proof steps are omitted, as only the statement is required.
  sorry

end expression_evaluation_l205_205396


namespace correct_parameterizations_of_line_l205_205324

theorem correct_parameterizations_of_line :
  ∀ (t : ℝ),
    (∀ (x y : ℝ), ((x = 5/3) ∧ (y = 0) ∨ (x = 0) ∧ (y = -5) ∨ (x = -5/3) ∧ (y = 0) ∨ 
                   (x = 1) ∧ (y = -2) ∨ (x = -2) ∧ (y = -11)) → 
                   y = 3 * x - 5) ∧
    (∀ (a b : ℝ), ((a = 1) ∧ (b = 3) ∨ (a = 3) ∧ (b = 1) ∨ (a = -1) ∧ (b = -3) ∨
                   (a = 1/3) ∧ (b = 1)) → 
                   b = 3 * a) →
    -- Check only Options D and E
    ((x = 1) → (y = -2) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) ∨
    ((x = -2) → (y = -11) → (a = 1/3) → (b = 1) → y = 3 * x - 5 ∧ b = 3 * a) :=
by
  sorry

end correct_parameterizations_of_line_l205_205324


namespace simplify_expr_l205_205712

variable (a b : ℤ)

theorem simplify_expr :
  (22 * a + 60 * b) + (10 * a + 29 * b) - (9 * a + 50 * b) = 23 * a + 39 * b :=
by
  sorry

end simplify_expr_l205_205712


namespace distance_to_second_museum_l205_205689

theorem distance_to_second_museum (d x : ℕ) (h1 : d = 5) (h2 : 2 * d + 2 * x = 40) : x = 15 :=
by
  sorry

end distance_to_second_museum_l205_205689


namespace part_I_part_II_l205_205416

variable {a b c : ℝ}
variable (habc : a ∈ Set.Ioi 0)
variable (hbbc : b ∈ Set.Ioi 0)
variable (hcbc : c ∈ Set.Ioi 0)
variable (h_sum : a + b + c = 1)

theorem part_I : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 :=
by
  sorry

theorem part_II : (a^2 + c^2) / b + (b^2 + a^2) / c + (c^2 + b^2) / a ≥ 2 :=
by
  sorry

end part_I_part_II_l205_205416


namespace largest_d_l205_205078

variable (a b c d : ℤ)

def condition : Prop := a + 2 = b - 1 ∧ a + 2 = c + 3 ∧ a + 2 = d - 4

theorem largest_d (h : condition a b c d) : d > a ∧ d > b ∧ d > c :=
by
  -- Assuming the condition holds, we need to prove d > a, d > b, and d > c
  sorry

end largest_d_l205_205078


namespace factorial_sqrt_square_l205_205790

theorem factorial_sqrt_square (n : ℕ) : (nat.succ 4)! * 4! = 2880 := by 
  sorry

end factorial_sqrt_square_l205_205790


namespace panda_bamboo_digestion_l205_205579

theorem panda_bamboo_digestion (h : 16 = 0.40 * x) : x = 40 :=
by sorry

end panda_bamboo_digestion_l205_205579


namespace ratio_of_pentagon_to_rectangle_l205_205039

theorem ratio_of_pentagon_to_rectangle (p l : ℕ) 
  (h1 : 5 * p = 30) (h2 : 2 * l + 2 * 5 = 30) : 
  p / l = 3 / 5 :=
by {
  sorry 
}

end ratio_of_pentagon_to_rectangle_l205_205039


namespace avg_children_in_families_with_children_l205_205875

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205875


namespace input_value_for_output_16_l205_205227

theorem input_value_for_output_16 (x : ℝ) (y : ℝ) (h1 : x < 0 → y = (x + 1)^2) (h2 : x ≥ 0 → y = (x - 1)^2) (h3 : y = 16) : x = 5 ∨ x = -5 := by
  sorry

end input_value_for_output_16_l205_205227


namespace AB_eq_B_exp_V_l205_205889

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l205_205889


namespace tan_A_area_triangle_ABC_l205_205411
open Real

-- Define the given conditions
def conditions (A : ℝ) (AC AB : ℝ) : Prop :=
  (sin A + cos A = sqrt 2 / 2) ∧ (AC = 2) ∧ (AB = 3)

-- State the first proof problem for tan A
theorem tan_A (A : ℝ) (hcond : conditions A 2 3) : tan A = -(2 + sqrt 3) := 
by 
  -- sorry for the proof placeholder
  sorry

-- State the second proof problem for the area of triangle ABC
theorem area_triangle_ABC (A B C : ℝ) (C_eq : C = 90) 
  (hcond : conditions A 2 3)
  (hBC : BC = sqrt ((AC^2) + (AB^2) - 2 * AC * AB * cos B)) : 
  (1/2) * AC * AB * sin A = (3 / 4) * (sqrt 6 + sqrt 2) := 
by 
  -- sorry for the proof placeholder
  sorry

end tan_A_area_triangle_ABC_l205_205411


namespace work_completion_days_l205_205028

theorem work_completion_days (D_a : ℝ) (R_a R_b : ℝ)
  (h1 : R_a = 1 / D_a)
  (h2 : R_b = 1 / (1.5 * D_a))
  (h3 : R_a = 1.5 * R_b)
  (h4 : 1 / 18 = R_a + R_b) : D_a = 30 := 
by
  sorry

end work_completion_days_l205_205028


namespace totalStudents_correct_l205_205595

-- Defining the initial number of classes, students per class, and new classes
def initialClasses : ℕ := 15
def studentsPerClass : ℕ := 20
def newClasses : ℕ := 5

-- Prove that the total number of students is 400
theorem totalStudents_correct : 
  initialClasses * studentsPerClass + newClasses * studentsPerClass = 400 := by
  sorry

end totalStudents_correct_l205_205595


namespace fraction_of_job_B_completes_l205_205478

theorem fraction_of_job_B_completes (hA : Nat) (hB : Nat) (t : Nat)
  (hA_time : hA = 6) (hB_time : hB = 3) (t_A_work : t = 1)
  : (1 - t_A_work / hA) / (1 / hA + 1 / hB) * 1 / hB = 5 / 9 :=
by
  sorry

end fraction_of_job_B_completes_l205_205478


namespace average_children_l205_205872

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205872


namespace brianna_marbles_lost_l205_205045

theorem brianna_marbles_lost
  (total_marbles : ℕ)
  (remaining_marbles : ℕ)
  (L : ℕ)
  (gave_away : ℕ)
  (dog_ate : ℚ)
  (h1 : total_marbles = 24)
  (h2 : remaining_marbles = 10)
  (h3 : gave_away = 2 * L)
  (h4 : dog_ate = L / 2)
  (h5 : total_marbles - remaining_marbles = L + gave_away + dog_ate) : L = 4 := 
by
  sorry

end brianna_marbles_lost_l205_205045


namespace find_r_l205_205085

theorem find_r (k r : ℝ) (h1 : 5 = k * 3^r) (h2 : 45 = k * 9^r) : r = Real.log 9 / Real.log 3 := by
  sorry

end find_r_l205_205085


namespace division_by_fraction_l205_205313

theorem division_by_fraction :
  (12 : ℝ) / (1 / 6) = 72 :=
by
  sorry

end division_by_fraction_l205_205313


namespace post_office_mail_in_six_months_l205_205457

/-- The number of pieces of mail the post office receives per day -/
def mail_per_day : ℕ := 60 + 20

/-- The number of days in six months, assuming each month has 30 days -/
def days_in_six_months : ℕ := 6 * 30

/-- The total number of pieces of mail handled in six months -/
def total_mail_in_six_months : ℕ := mail_per_day * days_in_six_months

/-- The post office handles 14400 pieces of mail in six months -/
theorem post_office_mail_in_six_months : total_mail_in_six_months = 14400 := by
  sorry

end post_office_mail_in_six_months_l205_205457


namespace olivia_probability_l205_205431

noncomputable def total_outcomes (n m : ℕ) : ℕ := Nat.choose n m

noncomputable def favorable_outcomes : ℕ :=
  let choose_three_colors := total_outcomes 4 3
  let choose_one_for_pair := total_outcomes 3 1
  let choose_socks :=
    (total_outcomes 3 2) * (total_outcomes 3 1) * (total_outcomes 3 1)
  choose_three_colors * choose_one_for_pair * choose_socks

def probability (n m : ℕ) : ℚ := n / m

theorem olivia_probability :
  probability favorable_outcomes (total_outcomes 12 5) = 9 / 22 :=
by
  sorry

end olivia_probability_l205_205431


namespace mary_age_l205_205096

theorem mary_age (M : ℕ) (h1 : ∀ t : ℕ, t = 4 → 24 = 2 * (M + t)) (h2 : 20 = 20) : M = 8 :=
by {
  have t_eq_4 := h1 4 rfl,
  norm_num at t_eq_4,
  linarith,
}

end mary_age_l205_205096


namespace Connor_spends_36_dollars_l205_205318

def tickets_cost := 10.00
def combo_meal_cost := 11.00
def candy_cost := 2.50
def total_cost := tickets_cost * 2 + combo_meal_cost + candy_cost * 2

theorem Connor_spends_36_dollars : total_cost = 36.00 := 
by 
  sorry

end Connor_spends_36_dollars_l205_205318


namespace number_of_boundaries_l205_205617

theorem number_of_boundaries 
  (total_runs : ℕ) 
  (number_of_sixes : ℕ) 
  (percentage_runs_by_running : ℝ) 
  (runs_per_six : ℕ) 
  (runs_per_boundary : ℕ)
  (h_total_runs : total_runs = 125)
  (h_number_of_sixes : number_of_sixes = 5)
  (h_percentage_runs_by_running : percentage_runs_by_running = 0.60)
  (h_runs_per_six : runs_per_six = 6)
  (h_runs_per_boundary : runs_per_boundary = 4) :
  (total_runs - percentage_runs_by_running * total_runs - number_of_sixes * runs_per_six) / runs_per_boundary = 5 := by 
  sorry

end number_of_boundaries_l205_205617


namespace desired_percentage_of_alcohol_l205_205255

def solution_x_alcohol_by_volume : ℝ := 0.10
def solution_y_alcohol_by_volume : ℝ := 0.30
def volume_solution_x : ℝ := 200
def volume_solution_y : ℝ := 600

theorem desired_percentage_of_alcohol :
  ((solution_x_alcohol_by_volume * volume_solution_x + solution_y_alcohol_by_volume * volume_solution_y) / 
  (volume_solution_x + volume_solution_y)) * 100 = 25 := 
sorry

end desired_percentage_of_alcohol_l205_205255


namespace age_in_1930_l205_205038

/-- A person's age at the time of their death (y) was one 31st of their birth year,
and we want to prove the person's age in 1930 (x). -/
theorem age_in_1930 (x y : ℕ) (h : 31 * y + x = 1930) (hx : 0 < x) (hxy : x < y) :
  x = 39 :=
sorry

end age_in_1930_l205_205038


namespace sqrt_floor_eq_l205_205286

theorem sqrt_floor_eq (n : ℤ) (h : n ≥ 0) : 
  (⌊Real.sqrt n + Real.sqrt (n + 2)⌋) = ⌊Real.sqrt (4 * n + 1)⌋ :=
sorry

end sqrt_floor_eq_l205_205286


namespace apple_and_cherry_pies_total_l205_205486

-- Given conditions state that:
def apple_pies : ℕ := 6
def cherry_pies : ℕ := 5

-- We aim to prove that the total number of apple and cherry pies is 11.
theorem apple_and_cherry_pies_total : apple_pies + cherry_pies = 11 := by
  sorry

end apple_and_cherry_pies_total_l205_205486


namespace average_children_in_families_with_children_l205_205882

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205882


namespace solution_l205_205439

theorem solution (y : ℚ) (h : (1/3 : ℚ) + 1/y = 7/9) : y = 9/4 :=
by
  sorry

end solution_l205_205439


namespace time_for_Harish_to_paint_alone_l205_205083

theorem time_for_Harish_to_paint_alone (H : ℝ) (h1 : H > 0) (h2 :  (1 / 6 + 1 / H) = 1 / 2 ) : H = 3 :=
sorry

end time_for_Harish_to_paint_alone_l205_205083


namespace cube_root_expression_l205_205786

theorem cube_root_expression (N : ℝ) (h : N > 1) : 
    (N^(1/3)^(1/3)^(1/3)^(1/3)) = N^(40/81) :=
sorry

end cube_root_expression_l205_205786


namespace men_became_absent_l205_205623

theorem men_became_absent (original_men planned_days actual_days : ℕ) (h1 : original_men = 48) (h2 : planned_days = 15) (h3 : actual_days = 18) :
  ∃ x : ℕ, 48 * 15 = (48 - x) * 18 ∧ x = 8 :=
by
  sorry

end men_became_absent_l205_205623


namespace kittens_given_is_two_l205_205234

-- Definitions of the conditions
def original_kittens : Nat := 8
def current_kittens : Nat := 6

-- Statement of the proof problem
theorem kittens_given_is_two : (original_kittens - current_kittens) = 2 := 
by
  sorry

end kittens_given_is_two_l205_205234


namespace prove_min_max_A_l205_205291

theorem prove_min_max_A : 
  ∃ (A_max A_min : ℕ), 
  (∃ B : ℕ, 
    A_max = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧
    B % 10 = 9) ∧ 
  (∃ B : ℕ, 
    A_min = 10^8 * (B % 10) + (B / 10) ∧ 
    B.gcd 24 = 1 ∧ 
    B > 666666666 ∧ 
    B % 10 = 1) ∧ 
  A_max = 999999998 ∧ 
  A_min = 166666667 := sorry

end prove_min_max_A_l205_205291


namespace pentagon_diagonals_l205_205392

def number_of_sides_pentagon : ℕ := 5
def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem pentagon_diagonals : number_of_diagonals number_of_sides_pentagon = 5 := by
  sorry

end pentagon_diagonals_l205_205392


namespace find_length_of_AC_l205_205686

-- Define the conditions as hypotheses
variables {A B C : Type} [metric_space A] [normed_group A]
variables (AB BC : ℝ) (B_angle_deg : ℝ)
def conditions := (AB = 3) ∧ (BC = 4) ∧ (B_angle_deg = 60)

-- Define the Cosine Rule
def cosine_rule_ac := ∀ (AB BC : ℝ) (B_angle : ℝ), 
  (AB = 3 → BC = 4 → B_angle = real.pi * 60 / 180 → 
  (real.sqrt (AB^2 + BC^2 - 2 * AB * BC * real.cos B_angle) = real.sqrt 13))

-- Statement to be proved
theorem find_length_of_AC : conditions AB BC B_angle_deg → cosine_rule_ac AB BC (real.pi * 60 / 180) := by
  intros h,
  rcases h with ⟨hab, hbc, hangle⟩,
  sorry

end find_length_of_AC_l205_205686


namespace find_m_l205_205219

noncomputable def hex_to_dec (m : ℕ) : ℕ :=
  3 * 6^4 + m * 6^3 + 5 * 6^2 + 2

theorem find_m (m : ℕ) : hex_to_dec m = 4934 ↔ m = 4 := 
by
  sorry

end find_m_l205_205219


namespace problem_statement_l205_205099

-- Define the arithmetic sequence and the conditions
noncomputable def a : ℕ → ℝ := sorry
axiom a_arith_seq : ∃ d : ℝ, ∀ n m : ℕ, a (n + m) = a n + m • d
axiom condition : a 4 + a 10 + a 16 = 30

-- State the theorem
theorem problem_statement : a 18 - 2 * a 14 = -10 :=
sorry

end problem_statement_l205_205099


namespace percentage_students_camping_trip_l205_205760

theorem percentage_students_camping_trip 
  (total_students : ℝ)
  (camping_trip_with_more_than_100 : ℝ) 
  (camping_trip_without_more_than_100_ratio : ℝ) :
  camping_trip_with_more_than_100 / (camping_trip_with_more_than_100 / 0.25) = 0.8 :=
by
  sorry

end percentage_students_camping_trip_l205_205760


namespace downstream_speed_l205_205773

-- Define the speed of the fish in still water
def V_s : ℝ := 45

-- Define the speed of the fish going upstream
def V_u : ℝ := 35

-- Define the speed of the stream
def V_r : ℝ := V_s - V_u

-- Define the speed of the fish going downstream
def V_d : ℝ := V_s + V_r

-- The theorem to be proved
theorem downstream_speed : V_d = 55 := by
  sorry

end downstream_speed_l205_205773


namespace raptors_points_l205_205489

theorem raptors_points (x y z : ℕ) (h1 : x + y + z = 48) (h2 : x - y = 18) :
  (z = 0 → y = 15) ∧
  (z = 12 → y = 9) ∧
  (z = 18 → y = 6) ∧
  (z = 30 → y = 0) :=
by sorry

end raptors_points_l205_205489


namespace line_divides_circle_l205_205090

theorem line_divides_circle (k m : ℝ) :
  (∀ x y : ℝ, y = x - 1 → x^2 + y^2 + k*x + m*y - 4 = 0 → m - k = 2) :=
sorry

end line_divides_circle_l205_205090


namespace units_digit_factorial_sum_l205_205531

theorem units_digit_factorial_sum :
  (1! + 2! + 3! + 4! + ∑ n in Finset.range (2011 - 5), (5 + n)! % 10) % 10 = 3 :=
by
  -- We will handle the details of the proof here.
  sorry

end units_digit_factorial_sum_l205_205531


namespace inequality_k_m_l205_205573

theorem inequality_k_m (k m : ℕ) (hk : 0 < k) (hm : 0 < m) (hkm : k > m) (hdiv : (k^3 - m^3) ∣ k * m * (k^2 - m^2)) :
  (k - m)^3 > 3 * k * m := 
by sorry

end inequality_k_m_l205_205573


namespace n_greater_than_sqrt_p_sub_1_l205_205570

theorem n_greater_than_sqrt_p_sub_1 {p n : ℕ} (hp : Nat.Prime p) (hn : n ≥ 2) (hdiv : p ∣ (n^6 - 1)) : n > Nat.sqrt p - 1 := 
by
  sorry

end n_greater_than_sqrt_p_sub_1_l205_205570


namespace avg_children_in_families_with_children_l205_205876

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205876


namespace rotation_matrix_150_degrees_l205_205356

open Real

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [cos θ, -sin θ],
    [sin θ, cos θ]
  ]

theorem rotation_matrix_150_degrees :
  rotation_matrix (150 * π / 180) = ![
    [-sqrt 3 / 2, -1 / 2],
    [ 1 / 2, -sqrt 3 / 2]
  ] :=
by
  sorry

end rotation_matrix_150_degrees_l205_205356


namespace probability_both_asian_selected_probability_A1_but_not_B1_selected_l205_205495

noncomputable def choose (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem probability_both_asian_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  asian_ways / total_ways = 1 / 5 := by
  let total_ways := choose 6 2
  let asian_ways := choose 3 2
  sorry

theorem probability_A1_but_not_B1_selected (A1 A2 A3 B1 B2 B3 : Prop) :
  let total_ways := 9
  let valid_ways := 2
  valid_ways / total_ways = 2 / 9 := by
  let total_ways := 9
  let valid_ways := 2
  sorry

end probability_both_asian_selected_probability_A1_but_not_B1_selected_l205_205495


namespace first_issue_pages_l205_205142

-- Define the conditions
def total_pages := 220
def pages_third_issue (x : ℕ) := x + 4

-- Statement of the problem
theorem first_issue_pages (x : ℕ) (hx : 3 * x + 4 = total_pages) : x = 72 :=
sorry

end first_issue_pages_l205_205142


namespace arctan_combination_l205_205372

noncomputable def find_m : ℕ :=
  133

theorem arctan_combination :
  (Real.arctan (1/7) + Real.arctan (1/8) + Real.arctan (1/9) + Real.arctan (1/find_m)) = (Real.pi / 4) :=
by
  sorry

end arctan_combination_l205_205372


namespace distance_between_A_and_mrs_A_l205_205703

-- Define the initial conditions
def speed_mr_A : ℝ := 30 -- Mr. A's speed in kmph
def speed_mrs_A : ℝ := 10 -- Mrs. A's speed in kmph
def speed_bee : ℝ := 60 -- The bee's speed in kmph
def distance_bee_traveled : ℝ := 180 -- Distance traveled by the bee in km

-- Define the proven statement
theorem distance_between_A_and_mrs_A : 
  distance_bee_traveled / speed_bee * (speed_mr_A + speed_mrs_A) = 120 := 
by 
  sorry

end distance_between_A_and_mrs_A_l205_205703


namespace size_relationship_l205_205070

noncomputable def a : ℝ := 1 + Real.sqrt 7
noncomputable def b : ℝ := Real.sqrt 3 + Real.sqrt 5
noncomputable def c : ℝ := 4

theorem size_relationship : a < b ∧ b < c := by
  sorry

end size_relationship_l205_205070


namespace division_by_fraction_l205_205312

theorem division_by_fraction :
  (12 : ℝ) / (1 / 6) = 72 :=
by
  sorry

end division_by_fraction_l205_205312


namespace rotation_matrix_150_l205_205355

-- Define the rotation matrix
def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![
    Real.cos θ, -Real.sin θ;
    Real.sin θ, Real.cos θ
  ]

-- Goal: Prove the matrix for 150 degrees rotation
theorem rotation_matrix_150 : 
  rotation_matrix (150 * Real.pi / 180) = !![
    -Real.sqrt 3 / 2, -1 / 2;
    1 / 2, -Real.sqrt 3 / 2
  ] :=
by
  sorry

end rotation_matrix_150_l205_205355


namespace infinitely_many_triples_of_integers_l205_205436

theorem infinitely_many_triples_of_integers (k : ℕ) :
  ∃ (x y z : ℕ), (x > 0 ∧ y > 0 ∧ z > 0) ∧
                  (x^999 + y^1000 = z^1001) :=
by
  sorry

end infinitely_many_triples_of_integers_l205_205436


namespace segment_area_l205_205505

theorem segment_area (d : ℝ) (θ : ℝ) (r := d / 2)
  (A_triangle := (1 / 2) * r^2 * Real.sin (θ * Real.pi / 180))
  (A_sector := (θ / 360) * Real.pi * r^2) :
  θ = 60 →
  d = 10 →
  A_sector - A_triangle = (100 * Real.pi - 75 * Real.sqrt 3) / 24 :=
by
  sorry

end segment_area_l205_205505


namespace simplify_expression_l205_205711

variable (b c : ℝ)

theorem simplify_expression :
  (1 : ℝ) * (-2 * b) * (3 * b^2) * (-4 * c^3) * (5 * c^4) = -120 * b^3 * c^7 :=
by sorry

end simplify_expression_l205_205711


namespace rationalize_denominator_l205_205583

noncomputable def cube_root (x : ℝ) := x^(1/3)

theorem rationalize_denominator (a b : ℝ) (h : cube_root 27 = 3) : 
  1 / (cube_root 3 + cube_root 27) = (3 - cube_root 3) / (9 - 3 * cube_root 3) :=
by
  sorry

end rationalize_denominator_l205_205583


namespace rhind_papyrus_smallest_portion_l205_205261

theorem rhind_papyrus_smallest_portion :
  ∀ (a1 d : ℚ),
    5 * a1 + (5 * 4 / 2) * d = 10 ∧
    (3 * a1 + 9 * d) / 7 = a1 + (a1 + d) →
    a1 = 1 / 6 :=
by sorry

end rhind_papyrus_smallest_portion_l205_205261


namespace expected_waiting_time_for_first_bite_l205_205516

noncomputable def average_waiting_time (λ : ℝ) : ℝ := 1 / λ

theorem expected_waiting_time_for_first_bite (bites_first_rod : ℝ) (bites_second_rod : ℝ) (total_time_minutes : ℝ) (total_time_seconds : ℝ) :
  bites_first_rod = 5 → 
  bites_second_rod = 1 → 
  total_time_minutes = 5 → 
  total_time_seconds = 300 → 
  average_waiting_time (bites_first_rod + bites_second_rod) * total_time_seconds = 50 :=
begin
  intros,
  sorry
end

end expected_waiting_time_for_first_bite_l205_205516


namespace hyperbola_constants_l205_205704

theorem hyperbola_constants (h k a c b : ℝ) : 
  h = -3 ∧ k = 1 ∧ a = 2 ∧ c = 5 ∧ b = Real.sqrt 21 → 
  h + k + a + b = 0 + Real.sqrt 21 :=
by
  intro hka
  sorry

end hyperbola_constants_l205_205704


namespace average_children_in_families_with_children_l205_205880

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205880


namespace original_price_of_cycle_l205_205625

variable (P : ℝ)

theorem original_price_of_cycle (h : 0.92 * P = 1610) : P = 1750 :=
sorry

end original_price_of_cycle_l205_205625


namespace expected_waiting_time_correct_l205_205510

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end expected_waiting_time_correct_l205_205510


namespace rational_roots_iff_a_eq_b_l205_205111

theorem rational_roots_iff_a_eq_b (a b : ℤ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℚ, x^2 + (a + b)^2 * x + 4 * a * b = 1) ↔ a = b :=
by
  sorry

end rational_roots_iff_a_eq_b_l205_205111


namespace find_abcd_l205_205186

theorem find_abcd 
    (a b c d : ℕ) 
    (h : 5^a + 6^b + 7^c + 11^d = 1999) : 
    (a, b, c, d) = (4, 2, 1, 3) :=
by
    sorry

end find_abcd_l205_205186


namespace fans_received_all_offers_l205_205308

theorem fans_received_all_offers :
  let hotdog_freq := 90
  let soda_freq := 45
  let popcorn_freq := 60
  let stadium_capacity := 4500
  let lcm_freq := Nat.lcm (Nat.lcm hotdog_freq soda_freq) popcorn_freq
  (stadium_capacity / lcm_freq) = 25 :=
by
  sorry

end fans_received_all_offers_l205_205308


namespace total_books_l205_205606

-- Define the number of books Tim has
def TimBooks : ℕ := 44

-- Define the number of books Sam has
def SamBooks : ℕ := 52

-- Statement to prove that the total number of books is 96
theorem total_books : TimBooks + SamBooks = 96 := by
  sorry

end total_books_l205_205606


namespace find_x_l205_205941

theorem find_x :
    ∃ x : ℚ, (1/7 + 7/x = 15/x + 1/15) ∧ x = 105 := by
  sorry

end find_x_l205_205941


namespace average_children_in_families_with_children_l205_205859

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205859


namespace intersection_of_A_and_B_l205_205536

def setA (x : ℝ) : Prop := x^2 < 4
def setB : Set ℝ := {0, 1}

theorem intersection_of_A_and_B :
  {x : ℝ | setA x} ∩ setB = setB := by
  sorry

end intersection_of_A_and_B_l205_205536


namespace nina_age_l205_205576

theorem nina_age : ∀ (M L A N : ℕ), 
  (M = L - 5) → 
  (L = A + 6) → 
  (N = A + 2) → 
  (M = 16) → 
  N = 17 :=
by
  intros M L A N h1 h2 h3 h4
  sorry

end nina_age_l205_205576


namespace fraction_sum_eq_one_l205_205693

variables {a b c x y z : ℝ}

-- Conditions
axiom h1 : 11 * x + b * y + c * z = 0
axiom h2 : a * x + 24 * y + c * z = 0
axiom h3 : a * x + b * y + 41 * z = 0
axiom h4 : a ≠ 11
axiom h5 : x ≠ 0

-- Theorem Statement
theorem fraction_sum_eq_one : 
  a/(a - 11) + b/(b - 24) + c/(c - 41) = 1 :=
by sorry

end fraction_sum_eq_one_l205_205693


namespace binom_18_4_eq_3060_l205_205800

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l205_205800


namespace find_the_number_l205_205715

theorem find_the_number 
  (x y n : ℤ)
  (h : 19 * (x + y) + 17 = 19 * (-x + y) - n)
  (hx : x = 1) :
  n = -55 :=
by
  sorry

end find_the_number_l205_205715


namespace suff_and_necc_l205_205537

variable (x : ℝ)

def A : Set ℝ := { x | x > 2 }
def B : Set ℝ := { x | x < 0 }
def C : Set ℝ := { x | x * (x - 2) > 0 }

theorem suff_and_necc : (x ∈ (A ∪ B)) ↔ (x ∈ C) := by
  sorry

end suff_and_necc_l205_205537


namespace function_increasing_on_R_l205_205914

noncomputable def f (x : ℝ) (a : ℝ) : ℝ :=
  if x < 1 then (3 - a) * x + 1 else a^x

theorem function_increasing_on_R (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ a ≤ f x₂ a) ↔ (2 ≤ a ∧ a < 3) :=
by
  sorry

end function_increasing_on_R_l205_205914


namespace range_of_a_intersection_l205_205654

theorem range_of_a_intersection (a : ℝ) : 
  (∀ k : ℝ, ∃ x y : ℝ, y = k * x - 2 * k + 2 ∧ y = a * x^2 - 2 * a * x - 3 * a) ↔ (a ≤ -2/3 ∨ a > 0) := by
  sorry

end range_of_a_intersection_l205_205654


namespace average_children_families_with_children_is_3_point_8_l205_205850

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205850


namespace rotation_matrix_150_l205_205351

noncomputable def cos_150 : ℝ := -real.cos (real.pi / 6)
noncomputable def sin_150 : ℝ := real.sin (real.pi / 6)

theorem rotation_matrix_150 : 
  ∀ θ : ℝ, θ = 5 * real.pi / 6 → 
  (matrix (fin 2) (fin 2) ℝ) := 
begin
  assume θ hθ,
  rw hθ,
  exact matrix.cons_vec_cons
    (matrix.cons_vec_cons cos_150 (-sin_150))
    (matrix.cons_vec_cons sin_150 cos_150),
  sorry
end

end rotation_matrix_150_l205_205351


namespace correct_equation_for_tournament_l205_205408

theorem correct_equation_for_tournament (x : ℕ) (h : x * (x - 1) / 2 = 28) : True :=
sorry

end correct_equation_for_tournament_l205_205408


namespace find_y_l205_205653

theorem find_y (y : ℝ) (h : (17.28 / 12) / (3.6 * y) = 2) : y = 0.2 :=
by {
  sorry
}

end find_y_l205_205653


namespace max_A_l205_205377

noncomputable def A (x y : ℝ) : ℝ :=
  x^4 * y + x * y^4 + x^3 * y + x * y^3 + x^2 * y + x * y^2

theorem max_A (x y : ℝ) (h : x + y = 1) : A x y ≤ 7 / 16 :=
sorry

end max_A_l205_205377


namespace average_upstream_speed_l205_205969

/--
There are three boats moving down a river. Boat A moves downstream at a speed of 1 km in 4 minutes 
and upstream at a speed of 1 km in 8 minutes. Boat B moves downstream at a speed of 1 km in 
5 minutes and upstream at a speed of 1 km in 11 minutes. Boat C moves downstream at a speed of 
1 km in 6 minutes and upstream at a speed of 1 km in 10 minutes. Prove that the average speed 
of the boats against the current is 6.32 km/h.
-/
theorem average_upstream_speed :
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  average_speed = 6.32 :=
by
  let speed_A_upstream := 1 / (8 / 60 : ℝ)
  let speed_B_upstream := 1 / (11 / 60 : ℝ)
  let speed_C_upstream := 1 / (10 / 60 : ℝ)
  let average_speed := (speed_A_upstream + speed_B_upstream + speed_C_upstream) / 3
  sorry

end average_upstream_speed_l205_205969


namespace rotation_matrix_150_degrees_l205_205368

theorem rotation_matrix_150_degrees : 
  let θ := Real.pi * (150 / 180)
  let c := Real.cos θ
  let s := Real.sin θ
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![
    ![c, -s],
    ![s, c]
  ] ∧ M = ![
    ![-(Real.sqrt 3 / 2), -(1 / 2)],
    ![(1 / 2), -(Real.sqrt 3 / 2)]
  ] :=
by
  let θ := Real.pi * (150 / 180)
  let c := Real.cos θ
  let s := Real.sin θ
  exists (![
    ![c, -s],
    ![s, c]
  ])
  split
  · -- Showing M is the rotation matrix form
    simp [c, s]
    sorry
  · -- showing that it matches the filled-in values
    simp
    sorry

end rotation_matrix_150_degrees_l205_205368


namespace intersection_a_b_l205_205073

-- Definitions of sets A and B
def A : Set ℝ := {x | -2 < x ∧ x ≤ 2}
def B : Set ℝ := {-2, -1, 0}

-- The proof problem
theorem intersection_a_b : A ∩ B = {-1, 0} :=
by
  sorry

end intersection_a_b_l205_205073


namespace find_number_l205_205482

theorem find_number (x : ℤ) (h : (7 * (x + 10) / 5) - 5 = 44) : x = 25 :=
sorry

end find_number_l205_205482


namespace binom_18_4_l205_205794

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l205_205794


namespace population_ratio_l205_205149

-- Definitions
def population_z (Z : ℕ) : ℕ := Z
def population_y (Z : ℕ) : ℕ := 2 * population_z Z
def population_x (Z : ℕ) : ℕ := 6 * population_y Z

-- Theorem stating the ratio
theorem population_ratio (Z : ℕ) : (population_x Z) / (population_z Z) = 12 :=
  by 
  unfold population_x population_y population_z
  sorry

end population_ratio_l205_205149


namespace all_girls_probability_l205_205160

-- Definition of the problem conditions
def probability_of_girl : ℚ := 1 / 2
def events_independent (P1 P2 P3 : ℚ) : Prop := P1 * P2 = P1 ∧ P2 * P3 = P2

-- The statement to prove
theorem all_girls_probability :
  events_independent probability_of_girl probability_of_girl probability_of_girl →
  (probability_of_girl * probability_of_girl * probability_of_girl) = 1 / 8 := 
by
  intros h
  sorry

end all_girls_probability_l205_205160


namespace subtraction_of_negatives_l205_205635

theorem subtraction_of_negatives : (-7) - (-5) = -2 := 
by {
  -- sorry replaces the actual proof steps.
  sorry
}

end subtraction_of_negatives_l205_205635


namespace avg_children_with_kids_l205_205824

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205824


namespace tetrahedron_volume_l205_205968

noncomputable def volume_of_tetrahedron (A B C O : Point) (r : ℝ) :=
  1 / 3 * (Real.sqrt (3) / 4 * 2^2 * Real.sqrt 11)

theorem tetrahedron_volume 
  (A B C O : Point)
  (side_length : ℝ)
  (surface_area : ℝ)
  (radius : ℝ)
  (h : ℝ)
  (radius_eq : radius = Real.sqrt (37 / 3))
  (side_length_eq : side_length = 2)
  (surface_area_eq : surface_area = (4 * Real.pi * radius^2))
  (sphere_surface_area_eq : surface_area = 148 * Real.pi / 3)
  (height_eq : h^2 = radius^2 - (2 / 3 * 2 * Real.sqrt 3 / 2)^2)
  (height_value_eq : h = Real.sqrt 11) :
  volume_of_tetrahedron A B C O radius = Real.sqrt 33 / 3 := sorry

end tetrahedron_volume_l205_205968


namespace first_group_men_l205_205593

theorem first_group_men (x : ℕ) (days1 days2 : ℝ) (men2 : ℕ) (h1 : days1 = 25) (h2 : days2 = 17.5) (h3 : men2 = 20) (h4 : x * days1 = men2 * days2) : x = 14 := 
by
  sorry

end first_group_men_l205_205593


namespace students_came_to_school_l205_205734

theorem students_came_to_school (F M T A : ℕ) 
    (hF : F = 658)
    (hM : M = F - 38)
    (hA : A = 17)
    (hT : T = M + F - A) :
    T = 1261 := by 
sorry

end students_came_to_school_l205_205734


namespace equivalent_expression_l205_205607

theorem equivalent_expression :
  (5+3) * (5^2 + 3^2) * (5^4 + 3^4) * (5^8 + 3^8) * (5^16 + 3^16) * 
  (5^32 + 3^32) * (5^64 + 3^64) = 5^128 - 3^128 := 
  sorry

end equivalent_expression_l205_205607


namespace vehicle_A_must_pass_B_before_B_collides_with_C_l205_205739

theorem vehicle_A_must_pass_B_before_B_collides_with_C
  (V_A : ℝ) -- speed of vehicle A in mph
  (V_B : ℝ := 40) -- speed of vehicle B in mph
  (V_C : ℝ := 65) -- speed of vehicle C in mph
  (distance_AB : ℝ := 100) -- distance between A and B in ft
  (distance_BC : ℝ := 250) -- initial distance between B and C in ft
  : (V_A > (100 * 65 - 150 * 40) / 250) :=
by {
  sorry
}

end vehicle_A_must_pass_B_before_B_collides_with_C_l205_205739


namespace reciprocal_inverse_proportional_l205_205737

variable {x y k c : ℝ}

-- Given condition: x * y = k
axiom inverse_proportional (h : x * y = k) : ∃ c, (1/x) * (1/y) = c

theorem reciprocal_inverse_proportional (h : x * y = k) :
  ∃ c, (1/x) * (1/y) = c :=
inverse_proportional h

end reciprocal_inverse_proportional_l205_205737


namespace min_value_x2_y2_z2_l205_205216

theorem min_value_x2_y2_z2 (x y z : ℝ) (h : 2 * x + 3 * y + 4 * z = 11) : 
  x^2 + y^2 + z^2 ≥ 121 / 29 :=
sorry

end min_value_x2_y2_z2_l205_205216


namespace parallel_DM_AO_l205_205414

open EuclideanGeometry

-- Definitions and assumptions
variables {A B C D E F M O : Point}
variables [Triangle ABC]
variables [h1 : AB ≠ AC]
variables [h2 : Midpoint D B C]
variables [h3 : Projection E D AB]
variables [h4 : Projection F D AC]
variables [h5 : Midpoint M E F]
variables [h6 : Circumcenter O A B C]

-- The statement to be proven
theorem parallel_DM_AO : Parallel (Line D M) (Line A O) :=
by sorry -- proofs omitted

end parallel_DM_AO_l205_205414


namespace arith_seq_sum_first_four_terms_l205_205202

noncomputable def sum_first_four_terms_arith_seq (a1 : ℤ) (d : ℤ) : ℤ :=
  4 * a1 + 6 * d

theorem arith_seq_sum_first_four_terms (a1 a3 : ℤ) 
  (h1 : a3 = a1 + 2 * 3)
  (h2 : a1 + a3 = 8) 
  (d : ℤ := 3) :
  sum_first_four_terms_arith_seq a1 d = 22 := by
  unfold sum_first_four_terms_arith_seq
  sorry

end arith_seq_sum_first_four_terms_l205_205202


namespace circle_radius_c_eq_32_l205_205178

theorem circle_radius_c_eq_32 :
  ∃ c : ℝ, (∀ x y : ℝ, x^2 - 8*x + y^2 + 10*y + c = 0 ↔ (x-4)^2 + (y+5)^2 = 9) :=
by
  use 32
  sorry

end circle_radius_c_eq_32_l205_205178


namespace simplify_expr_l205_205713

variable (a b : ℤ)

theorem simplify_expr :
  (22 * a + 60 * b) + (10 * a + 29 * b) - (9 * a + 50 * b) = 23 * a + 39 * b :=
by
  sorry

end simplify_expr_l205_205713


namespace rotation_matrix_150_deg_correct_l205_205367

open Real
open Matrix

noncomputable def rotation_matrix_150_deg : Matrix (Fin 2) (Fin 2) ℝ :=
  let cos150 := -sqrt 3 / 2
  let sin150 := 1 / 2
  ![![cos150, -sin150], ![sin150, cos150]]

theorem rotation_matrix_150_deg_correct :
  rotation_matrix_150_deg = ![![-(sqrt 3 / 2), -1 / 2], ![1 / 2, -(sqrt 3 / 2)]] :=
by
  sorry

end rotation_matrix_150_deg_correct_l205_205367


namespace claire_photos_l205_205246

-- Define the number of photos taken by Claire, Lisa, and Robert
variables (C L R : ℕ)

-- Conditions based on the problem
def Lisa_photos (C : ℕ) := 3 * C
def Robert_photos (C : ℕ) := C + 24

-- Prove that C = 12 given the conditions
theorem claire_photos : 
  (L = Lisa_photos C) ∧ (R = Robert_photos C) ∧ (L = R) → C = 12 := 
by
  sorry

end claire_photos_l205_205246


namespace shares_of_c_l205_205757

theorem shares_of_c (a b c : ℝ) (h1 : 3 * a = 4 * b) (h2 : 4 * b = 7 * c) (h3 : a + b + c = 427): 
  c = 84 :=
by {
  sorry
}

end shares_of_c_l205_205757


namespace probability_heads_at_least_10_out_of_12_l205_205744

theorem probability_heads_at_least_10_out_of_12 (n m : Nat) (hn : n = 12) (hm : m = 10):
  let total_outcomes := 2^n
  let ways_10 := Nat.choose n m
  let ways_11 := Nat.choose n (m + 1)
  let ways_12 := Nat.choose n (m + 2)
  let successful_outcomes := ways_10 + ways_11 + ways_12
  total_outcomes = 4096 →
  successful_outcomes = 79 →
  (successful_outcomes : ℚ) / total_outcomes = 79 / 4096 :=
by
  sorry

end probability_heads_at_least_10_out_of_12_l205_205744


namespace mail_in_six_months_l205_205453

/-- The post office receives 60 letters and 20 packages per day. Each month has 30 days. -/
def daily_letters := 60
def daily_packages := 20
def days_per_month := 30
def months := 6

/-- Prove that the post office handles 14400 pieces of mail in six months. -/
theorem mail_in_six_months : (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  sorry

end mail_in_six_months_l205_205453


namespace Connor_spends_36_dollars_l205_205317

def tickets_cost := 10.00
def combo_meal_cost := 11.00
def candy_cost := 2.50
def total_cost := tickets_cost * 2 + combo_meal_cost + candy_cost * 2

theorem Connor_spends_36_dollars : total_cost = 36.00 := 
by 
  sorry

end Connor_spends_36_dollars_l205_205317


namespace mail_in_six_months_l205_205451

/-- The post office receives 60 letters and 20 packages per day. Each month has 30 days. -/
def daily_letters := 60
def daily_packages := 20
def days_per_month := 30
def months := 6

/-- Prove that the post office handles 14400 pieces of mail in six months. -/
theorem mail_in_six_months : (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  sorry

end mail_in_six_months_l205_205451


namespace real_solution_four_unknowns_l205_205888

theorem real_solution_four_unknowns (x y z t : ℝ) :
  x^2 + y^2 + z^2 + t^2 = x * (y + z + t) ↔ (x = 0 ∧ y = 0 ∧ z = 0 ∧ t = 0) :=
by
  sorry

end real_solution_four_unknowns_l205_205888


namespace sum_squares_of_real_solutions_problem_sum_of_squares_l205_205530

theorem sum_squares_of_real_solutions (x : ℝ) (hx : x ^ 64 = 2 ^ 64) :
  x = 2 ∨ x = -2 :=
begin
  by_cases x = 2,
  { left,
    assumption, },
  { right,
    apply (neg_eq_iff_neg_eq).1,
    rw [← pow_eq_one_iff_mod_two_eq_zero] at hx,
    exact hx.symm, }
end

theorem problem_sum_of_squares :
  ∑ x in {x | x ^ 64 = 2 ^ 64}.to_finset, x^2 = 8 :=
begin
  apply finset.sum_const_nat,
  sorry,
end

end sum_squares_of_real_solutions_problem_sum_of_squares_l205_205530


namespace polynomial_evaluation_at_8_l205_205764

def P (x : ℝ) : ℝ := x^3 + 2*x^2 + x - 1

theorem polynomial_evaluation_at_8 : P 8 = 647 :=
by sorry

end polynomial_evaluation_at_8_l205_205764


namespace faster_speed_l205_205626

theorem faster_speed (D : ℝ) (v : ℝ) (h₁ : D = 33.333333333333336) 
                      (h₂ : 10 * (D + 20) = v * D) : v = 16 :=
by
  sorry

end faster_speed_l205_205626


namespace selling_price_correct_l205_205567

/-- Define the initial cost of the gaming PC. -/
def initial_pc_cost : ℝ := 1200

/-- Define the cost of the new video card. -/
def new_video_card_cost : ℝ := 500

/-- Define the total spending after selling the old card. -/
def total_spending : ℝ := 1400

/-- Define the selling price of the old card -/
def selling_price_of_old_card : ℝ := (initial_pc_cost + new_video_card_cost) - total_spending

/-- Prove that John sold the old card for $300. -/
theorem selling_price_correct : selling_price_of_old_card = 300 := by
  sorry

end selling_price_correct_l205_205567


namespace max_rock_value_l205_205316

/-- Carl discovers a cave with three types of rocks:
    - 6-pound rocks worth $16 each,
    - 3-pound rocks worth $9 each,
    - 2-pound rocks worth $3 each.
    There are at least 15 of each type.
    He can carry a maximum of 20 pounds and no more than 5 rocks in total.
    Prove that the maximum value, in dollars, of the rocks he can carry is $52. -/
theorem max_rock_value :
  ∃ (max_value: ℕ),
  (∀ (c6 c3 c2: ℕ),
    (c6 + c3 + c2 ≤ 5) ∧
    (6 * c6 + 3 * c3 + 2 * c2 ≤ 20) →
    max_value ≥ 16 * c6 + 9 * c3 + 3 * c2) ∧
  max_value = 52 :=
by
  sorry

end max_rock_value_l205_205316


namespace orchids_to_roses_ratio_l205_205699

noncomputable def total_centerpieces : ℕ := 6
noncomputable def roses_per_centerpiece : ℕ := 8
noncomputable def lilies_per_centerpiece : ℕ := 6
noncomputable def total_budget : ℕ := 2700
noncomputable def cost_per_flower : ℕ := 15
noncomputable def total_flowers : ℕ := total_budget / cost_per_flower

noncomputable def total_roses : ℕ := total_centerpieces * roses_per_centerpiece
noncomputable def total_lilies : ℕ := total_centerpieces * lilies_per_centerpiece
noncomputable def total_roses_and_lilies : ℕ := total_roses + total_lilies
noncomputable def total_orchids : ℕ := total_flowers - total_roses_and_lilies
noncomputable def orchids_per_centerpiece : ℕ := total_orchids / total_centerpieces

theorem orchids_to_roses_ratio : orchids_per_centerpiece / roses_per_centerpiece = 2 :=
by
  sorry

end orchids_to_roses_ratio_l205_205699


namespace Murtha_pebbles_problem_l205_205955

theorem Murtha_pebbles_problem : 
  let a := 3
  let d := 3
  let n := 18
  let a_n := a + (n - 1) * d
  let S_n := n / 2 * (a + a_n)
  S_n = 513 :=
by
  sorry

end Murtha_pebbles_problem_l205_205955


namespace percentage_BCM_hens_l205_205772

theorem percentage_BCM_hens (total_chickens : ℕ) (BCM_percentage : ℝ) (BCM_hens : ℕ) : 
  total_chickens = 100 → BCM_percentage = 0.20 → BCM_hens = 16 →
  ((BCM_hens : ℝ) / (total_chickens * BCM_percentage)) * 100 = 80 :=
by
  sorry

end percentage_BCM_hens_l205_205772


namespace length_decreased_by_l205_205265

noncomputable def length_decrease_proof : Prop :=
  let length := 33.333333333333336
  let breadth := length / 2
  let new_length := length - 2.833333333333336
  let new_breadth := breadth + 4
  let original_area := length * breadth
  let new_area := new_length * new_breadth
  (new_area = original_area + 75) ↔ (new_length = length - 2.833333333333336)

theorem length_decreased_by : length_decrease_proof := sorry

end length_decreased_by_l205_205265


namespace coin_draw_probability_l205_205159

theorem coin_draw_probability :
  let coins := [ (3, 25), (5, 10), (7, 5) ]
  let total_ways := Nat.choose 15 8
  let successful_outcomes := 
    Nat.choose 3 3 * Nat.choose 5 5 +
    Nat.choose 3 2 * Nat.choose 5 4 * Nat.choose 7 2
  (successful_outcomes.toRat / total_ways.toRat) = 316 / 6435 := 
by {
  sorry
}

end coin_draw_probability_l205_205159


namespace inverse_of_composite_l205_205466

-- Define the function g
def g (x : ℕ) : ℕ :=
  if x = 1 then 4 else
  if x = 2 then 3 else
  if x = 3 then 1 else
  if x = 4 then 5 else
  if x = 5 then 2 else
  0  -- g is not defined for values other than 1 to 5

-- Define the inverse g_inv
def g_inv (x : ℕ) : ℕ :=
  if x = 4 then 1 else
  if x = 3 then 2 else
  if x = 1 then 3 else
  if x = 5 then 4 else
  if x = 2 then 5 else
  0  -- g_inv is not defined for values other than 1 to 5

theorem inverse_of_composite :
  g_inv (g_inv (g_inv 3)) = 4 :=
by
  sorry

end inverse_of_composite_l205_205466


namespace rotation_matrix_150_deg_correct_l205_205366

open Real
open Matrix

noncomputable def rotation_matrix_150_deg : Matrix (Fin 2) (Fin 2) ℝ :=
  let cos150 := -sqrt 3 / 2
  let sin150 := 1 / 2
  ![![cos150, -sin150], ![sin150, cos150]]

theorem rotation_matrix_150_deg_correct :
  rotation_matrix_150_deg = ![![-(sqrt 3 / 2), -1 / 2], ![1 / 2, -(sqrt 3 / 2)]] :=
by
  sorry

end rotation_matrix_150_deg_correct_l205_205366


namespace initial_kids_l205_205733

theorem initial_kids {N : ℕ} (h1 : 1 / 2 * N = N / 2) (h2 : 1 / 2 * (N / 2) = N / 4) (h3 : N / 4 = 5) : N = 20 :=
by
  sorry

end initial_kids_l205_205733


namespace integer_solutions_count_l205_205002

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l205_205002


namespace at_least_two_equal_l205_205380

theorem at_least_two_equal (x y z : ℝ) (h1 : x * y + z = y * z + x) (h2 : y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l205_205380


namespace square_side_length_on_hexagon_l205_205303

noncomputable def side_length_of_square (s : ℝ) : Prop :=
  let hexagon_side := 1
  let internal_angle := 120
  ((s * (1 + 1 / Real.sqrt 3)) = 2) → s = (3 - Real.sqrt 3)

theorem square_side_length_on_hexagon : ∃ s : ℝ, side_length_of_square s :=
by
  use 3 - Real.sqrt 3
  -- Proof to be provided
  sorry

end square_side_length_on_hexagon_l205_205303


namespace total_students_in_class_l205_205681

theorem total_students_in_class 
  (hockey_players : ℕ)
  (basketball_players : ℕ)
  (neither_players : ℕ)
  (both_players : ℕ)
  (hockey_players_eq : hockey_players = 15)
  (basketball_players_eq : basketball_players = 16)
  (neither_players_eq : neither_players = 4)
  (both_players_eq : both_players = 10) :
  hockey_players + basketball_players - both_players + neither_players = 25 := 
by 
  sorry

end total_students_in_class_l205_205681


namespace even_function_increasing_on_negative_half_l205_205242

variable (f : ℝ → ℝ)
variable (x1 x2 : ℝ)

theorem even_function_increasing_on_negative_half (h1 : ∀ x, f (-x) = f x)
                                                  (h2 : ∀ a b : ℝ, a < b → b < 0 → f a < f b)
                                                  (h3 : x1 < 0 ∧ 0 < x2) (h4 : x1 + x2 > 0) 
                                                  : f (- x1) > f (x2) :=
by
  sorry

end even_function_increasing_on_negative_half_l205_205242


namespace find_angle_B_l205_205405

theorem find_angle_B 
  (a b : ℝ) (A B : ℝ) 
  (ha : a = 2 * Real.sqrt 2) 
  (hb : b = 2)
  (hA : A = Real.pi / 4) -- 45 degrees in radians
  (h_triangle : ∃ c, a^2 + b^2 - 2*a*b*Real.cos A = c^2 ∧ a^2 * Real.sin 45 = b^2 * Real.sin B) :
  B = Real.pi / 6 := -- 30 degrees in radians
sorry

end find_angle_B_l205_205405


namespace matrix_B3_is_zero_unique_l205_205116

theorem matrix_B3_is_zero_unique (B : Matrix (Fin 2) (Fin 2) ℝ) (h : B^4 = 0) :
  ∃! (B3 : Matrix (Fin 2) (Fin 2) ℝ), B3 = B^3 ∧ B3 = 0 := sorry

end matrix_B3_is_zero_unique_l205_205116


namespace average_children_in_families_with_children_l205_205843

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205843


namespace minimum_buses_required_l205_205432

-- Condition definitions
def one_way_trip_time : ℕ := 50
def stop_time : ℕ := 10
def departure_interval : ℕ := 6

-- Total round trip time
def total_round_trip_time : ℕ := 2 * one_way_trip_time + 2 * stop_time

-- The total number of buses needed to ensure the bus departs every departure_interval minutes
-- from both stations A and B.
theorem minimum_buses_required : 
  (total_round_trip_time / departure_interval) = 20 := by
  sorry

end minimum_buses_required_l205_205432


namespace count_integer_values_satisfying_condition_l205_205010

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l205_205010


namespace simplify_expression_l205_205438

theorem simplify_expression (x : ℝ) : 
  (x^2 + 2 * x + 3) / 4 + (3 * x - 5) / 6 = (3 * x^2 + 12 * x - 1) / 12 := 
by
  sorry

end simplify_expression_l205_205438


namespace boys_assigned_l205_205022

theorem boys_assigned (B G : ℕ) (h1 : B + G = 18) (h2 : B = G - 2) : B = 8 :=
sorry

end boys_assigned_l205_205022


namespace speed_increase_71_6_percent_l205_205930

theorem speed_increase_71_6_percent (S : ℝ) (hS : 0 < S) : 
    let S₁ := S * 1.30
    let S₂ := S₁ * 1.10
    let S₃ := S₂ * 1.20
    (S₃ - S) / S * 100 = 71.6 :=
by
  let S₁ := S * 1.30
  let S₂ := S₁ * 1.10
  let S₃ := S₂ * 1.20
  sorry

end speed_increase_71_6_percent_l205_205930


namespace average_children_in_families_with_children_l205_205861

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205861


namespace negative_product_implies_negatives_l205_205401

theorem negative_product_implies_negatives (a b c : ℚ) (h : a * b * c < 0) :
  (∃ n : ℕ, n = 1 ∨ n = 3 ∧ (n = 1 ↔ (a < 0 ∧ b > 0 ∧ c > 0 ∨ a > 0 ∧ b < 0 ∧ c > 0 ∨ a > 0 ∧ b > 0 ∧ c < 0)) ∨ 
                                n = 3 ∧ (n = 3 ↔ (a < 0 ∧ b < 0 ∧ c < 0 ∨ a < 0 ∧ b < 0 ∧ c > 0 ∨ a < 0 ∧ b > 0 ∧ c < 0 ∨ a > 0 ∧ b < 0 ∧ c < 0))) :=
  sorry

end negative_product_implies_negatives_l205_205401


namespace find_matrix_l205_205527

theorem find_matrix (M : Matrix (Fin 2) (Fin 2) ℝ)
  (h : ∀ v : Matrix (Fin 2) (Fin 1) ℝ, M.mul_vec v = (-7 : ℝ) • v) :
  M = !![-7, 0; 0, -7] :=
by
  sorry

end find_matrix_l205_205527


namespace hyperbola_asymptotes_and_parabola_l205_205443

-- Definitions for hyperbola and parabola
noncomputable def hyperbola (x y : ℝ) : Prop := (x^2 / 9) - (y^2 / 16) = 1
noncomputable def focus_of_hyperbola (focus : ℝ × ℝ) : Prop := focus = (5, 0)
noncomputable def asymptote_of_hyperbola (y x : ℝ) : Prop := y = (4 / 3) * x ∨ y = - (4 / 3) * x
noncomputable def parabola (y x p : ℝ) : Prop := y^2 = 2 * p * x

-- Main statement
theorem hyperbola_asymptotes_and_parabola :
  (∀ x y, hyperbola x y → asymptote_of_hyperbola y x) ∧
  (∀ y x, focus_of_hyperbola (5, 0) → parabola y x 10) :=
by
  -- To be proved
  sorry

end hyperbola_asymptotes_and_parabola_l205_205443


namespace average_children_families_with_children_is_3_point_8_l205_205849

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205849


namespace sum_of_square_roots_l205_205314

theorem sum_of_square_roots : 
  (Real.sqrt 1 + Real.sqrt (1 + 3) + Real.sqrt (1 + 3 + 5) + 
  Real.sqrt (1 + 3 + 5 + 7) + Real.sqrt (1 + 3 + 5 + 7 + 9) + 
  Real.sqrt (1 + 3 + 5 + 7 + 9 + 11)) = 21 :=
by
  -- Proof here
  sorry

end sum_of_square_roots_l205_205314


namespace post_office_mail_in_six_months_l205_205458

/-- The number of pieces of mail the post office receives per day -/
def mail_per_day : ℕ := 60 + 20

/-- The number of days in six months, assuming each month has 30 days -/
def days_in_six_months : ℕ := 6 * 30

/-- The total number of pieces of mail handled in six months -/
def total_mail_in_six_months : ℕ := mail_per_day * days_in_six_months

/-- The post office handles 14400 pieces of mail in six months -/
theorem post_office_mail_in_six_months : total_mail_in_six_months = 14400 := by
  sorry

end post_office_mail_in_six_months_l205_205458


namespace geese_ratio_l205_205587

/-- Define the problem conditions --/

def lily_ducks := 20
def lily_geese := 10

def rayden_ducks : ℕ := 3 * lily_ducks
def total_lily_animals := lily_ducks + lily_geese
def total_rayden_animals := total_lily_animals + 70
def rayden_geese := total_rayden_animals - rayden_ducks

/-- Prove the desired ratio of the number of geese Rayden bought to the number of geese Lily bought --/
theorem geese_ratio : rayden_geese / lily_geese = 4 :=
sorry

end geese_ratio_l205_205587


namespace interval_where_f_decreasing_minimum_value_of_a_l205_205663

open Real

noncomputable def f (x : ℝ) : ℝ := log x - x^2 + x
noncomputable def h (a x : ℝ) : ℝ := (a - 1) * x^2 + 2 * a * x - 1

theorem interval_where_f_decreasing :
  {x : ℝ | 1 < x} = {x : ℝ | deriv f x < 0} :=
by sorry

theorem minimum_value_of_a (a : ℤ) (ha : ∀ x : ℝ, 0 < x → (a - 1) * x^2 + 2 * a * x - 1 ≥ log x - x^2 + x) :
  a ≥ 1 :=
by sorry

end interval_where_f_decreasing_minimum_value_of_a_l205_205663


namespace gabby_additional_money_needed_l205_205067

theorem gabby_additional_money_needed
  (cost_makeup : ℕ := 65)
  (cost_skincare : ℕ := 45)
  (cost_hair_tool : ℕ := 55)
  (initial_savings : ℕ := 35)
  (money_from_mom : ℕ := 20)
  (money_from_dad : ℕ := 30)
  (money_from_chores : ℕ := 25) :
  (cost_makeup + cost_skincare + cost_hair_tool) - (initial_savings + money_from_mom + money_from_dad + money_from_chores) = 55 := 
by
  sorry

end gabby_additional_money_needed_l205_205067


namespace mike_worked_four_hours_l205_205562

-- Define the time to perform each task in minutes
def time_wash_car : ℕ := 10
def time_change_oil : ℕ := 15
def time_change_tires : ℕ := 30

-- Define the number of tasks Mike performed
def num_wash_cars : ℕ := 9
def num_change_oil : ℕ := 6
def num_change_tires : ℕ := 2

-- Define the total minutes Mike worked
def total_minutes_worked : ℕ :=
  (num_wash_cars * time_wash_car) +
  (num_change_oil * time_change_oil) +
  (num_change_tires * time_change_tires)

-- Define the conversion from minutes to hours
def total_hours_worked : ℕ := total_minutes_worked / 60

-- Formalize the proof statement
theorem mike_worked_four_hours :
  total_hours_worked = 4 :=
by
  sorry

end mike_worked_four_hours_l205_205562


namespace average_children_in_families_with_children_l205_205865

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205865


namespace sum_of_exterior_angles_of_triangle_l205_205976

theorem sum_of_exterior_angles_of_triangle
  {α β γ α' β' γ' : ℝ} 
  (h1 : α + β + γ = 180)
  (h2 : α + α' = 180)
  (h3 : β + β' = 180)
  (h4 : γ + γ' = 180) :
  α' + β' + γ' = 360 := 
by 
sorry

end sum_of_exterior_angles_of_triangle_l205_205976


namespace trig_expression_value_l205_205909

theorem trig_expression_value (θ : Real) (h1 : θ > Real.pi) (h2 : θ < 3 * Real.pi / 2) (h3 : Real.tan (2 * θ) = 3 / 4) :
  (2 * Real.cos (θ / 2) ^ 2 + Real.sin θ - 1) / (Real.sqrt 2 * Real.cos (θ + Real.pi / 4)) = 2 := by
  sorry

end trig_expression_value_l205_205909


namespace perimeter_of_triangle_l205_205152

-- Define the average length of the sides of the triangle
def average_length (a b c : ℕ) : ℕ := (a + b + c) / 3

-- Define the perimeter of the triangle
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- The theorem we want to prove
theorem perimeter_of_triangle {a b c : ℕ} (h_avg : average_length a b c = 12) : perimeter a b c = 36 :=
sorry

end perimeter_of_triangle_l205_205152


namespace find_f_l205_205639

theorem find_f 
  (h_vertex : ∃ (d e f : ℝ), ∀ x, y = d * (x - 3)^2 - 5 ∧ y = d * x^2 + e * x + f)
  (h_point : y = d * (4 - 3)^2 - 5) 
  (h_value : y = -3) :
  ∃ f, f = 13 :=
sorry

end find_f_l205_205639


namespace degrees_to_radians_90_l205_205814

theorem degrees_to_radians_90 : (90 : ℝ) * (Real.pi / 180) = (Real.pi / 2) :=
by
  sorry

end degrees_to_radians_90_l205_205814


namespace solid_is_frustum_l205_205680

-- Definitions for views
def front_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def side_view_is_isosceles_trapezoid (S : Type) : Prop := sorry
def top_view_is_concentric_circles (S : Type) : Prop := sorry

-- Define the target solid as a frustum
def is_frustum (S : Type) : Prop := sorry

-- The theorem statement
theorem solid_is_frustum
  (S : Type) 
  (h1 : front_view_is_isosceles_trapezoid S)
  (h2 : side_view_is_isosceles_trapezoid S)
  (h3 : top_view_is_concentric_circles S) :
  is_frustum S :=
sorry

end solid_is_frustum_l205_205680


namespace simplify_fraction_l205_205437

theorem simplify_fraction (x : ℤ) :
  (2 * x - 3) / 4 + (3 * x + 5) / 5 - (x - 1) / 2 = (12 * x + 15) / 20 :=
by sorry

end simplify_fraction_l205_205437


namespace huangs_tax_is_65_yuan_l205_205042

noncomputable def monthly_salary : ℝ := 2900
noncomputable def tax_free_portion : ℝ := 2000
noncomputable def tax_rate_5_percent : ℝ := 0.05
noncomputable def tax_rate_10_percent : ℝ := 0.10

noncomputable def taxable_income_amount (income : ℝ) (exemption : ℝ) : ℝ := income - exemption

noncomputable def personal_income_tax (income : ℝ) : ℝ :=
  let taxable_income := taxable_income_amount income tax_free_portion
  if taxable_income ≤ 500 then
    taxable_income * tax_rate_5_percent
  else
    (500 * tax_rate_5_percent) + ((taxable_income - 500) * tax_rate_10_percent)

theorem huangs_tax_is_65_yuan : personal_income_tax monthly_salary = 65 :=
by
  sorry

end huangs_tax_is_65_yuan_l205_205042


namespace simplify_expression_l205_205793

variable (a : ℤ)

theorem simplify_expression : (-2 * a) ^ 3 * a ^ 3 + (-3 * a ^ 3) ^ 2 = a ^ 6 :=
by sorry

end simplify_expression_l205_205793


namespace average_children_families_with_children_is_3_point_8_l205_205845

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205845


namespace part1_part2_l205_205081

noncomputable def f (x k : ℝ) : ℝ := (x ^ 2 + k * x + 1) / (x ^ 2 + 1)

theorem part1 (k : ℝ) (h : k = -4) : ∃ x > 0, f x k = -1 :=
  by sorry -- Proof goes here

theorem part2 (k : ℝ) : (∀ (x1 x2 x3 : ℝ), (0 < x1) → (0 < x2) → (0 < x3) → 
  ∃ a b c, a = f x1 k ∧ b = f x2 k ∧ c = f x3 k ∧ 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) ↔ (-1 ≤ k ∧ k ≤ 2) :=
  by sorry -- Proof goes here

end part1_part2_l205_205081


namespace at_least_two_equal_l205_205379

theorem at_least_two_equal (x y z : ℝ) (h1 : x * y + z = y * z + x) (h2 : y * z + x = z * x + y) : 
  x = y ∨ y = z ∨ z = x := 
sorry

end at_least_two_equal_l205_205379


namespace average_children_in_families_with_children_l205_205884

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205884


namespace second_part_of_ratio_l205_205769

-- Define the conditions
def ratio_percent := 20
def first_part := 4

-- Define the proof statement using the conditions
theorem second_part_of_ratio (ratio_percent : ℕ) (first_part : ℕ) : 
  ∃ second_part : ℕ, (first_part * 100) = ratio_percent * second_part :=
by
  -- Let the second part be 20 and verify the condition
  use 20
  -- Clear the proof (details are not required)
  sorry

end second_part_of_ratio_l205_205769


namespace total_time_in_range_l205_205481

-- Definitions for the problem conditions
def section1 := 240 -- km
def section2 := 300 -- km
def section3 := 400 -- km

def speed1 := 40 -- km/h
def speed2 := 75 -- km/h
def speed3 := 80 -- km/h

-- The time it takes to cover a section at a certain speed
def time (distance : ℕ) (speed : ℕ) : ℕ := distance / speed

-- Total time to cover all sections with different speed assignments
def total_time (s1 s2 s3 v1 v2 v3 : ℕ) : ℕ :=
  time s1 v1 + time s2 v2 + time s3 v3

-- Prove that the total time is within the range [15, 17]
theorem total_time_in_range :
  (total_time section1 section2 section3 speed3 speed2 speed1 = 15) ∧
  (total_time section1 section2 section3 speed1 speed2 speed3 = 17) →
  ∃ (T : ℕ), 15 ≤ T ∧ T ≤ 17 :=
by
  intro h
  sorry

end total_time_in_range_l205_205481


namespace polynomial_identity_l205_205596

theorem polynomial_identity (a b c : ℝ) : 
  a * (b - c)^3 + b * (c - a)^3 + c * (a - b)^3 = 
  (a - b) * (b - c) * (c - a) * (a + b + c) :=
sorry

end polynomial_identity_l205_205596


namespace percentage_of_total_population_absent_l205_205958

def total_students : ℕ := 120
def boys : ℕ := 72
def girls : ℕ := 48
def boys_absent_fraction : ℚ := 1/8
def girls_absent_fraction : ℚ := 1/4

theorem percentage_of_total_population_absent : 
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students * 100 = 17.5 :=
by
  sorry

end percentage_of_total_population_absent_l205_205958


namespace max_value_5x_minus_25x_l205_205060

noncomputable def max_value_of_expression : ℝ :=
  (1 / 4 : ℝ)

theorem max_value_5x_minus_25x :
  ∃ x : ℝ, ∀ y : ℝ, y = 5^x → (5^y - 25^y) ≤ max_value_of_expression :=
sorry

end max_value_5x_minus_25x_l205_205060


namespace find_cos_value_l205_205393

theorem find_cos_value (α : Real) 
  (h : Real.cos (Real.pi / 8 - α) = 1 / 6) : 
  Real.cos (3 * Real.pi / 4 + 2 * α) = 17 / 18 :=
by
  sorry

end find_cos_value_l205_205393


namespace complex_sum_eighth_power_l205_205240

noncomputable def compute_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : ℂ :=
  ζ1^8 + ζ2^8 + ζ3^8

theorem complex_sum_eighth_power 
(ζ1 ζ2 ζ3 : ℂ) 
(h1 : ζ1 + ζ2 + ζ3 = 2) 
(h2 : ζ1^2 + ζ2^2 + ζ3^2 = 5) 
(h3 : ζ1^3 + ζ2^3 + ζ3^3 = 8) : 
  compute_sum_eighth_power ζ1 ζ2 ζ3 h1 h2 h3 = 451.625 :=
sorry

end complex_sum_eighth_power_l205_205240


namespace ratio_of_wealth_l205_205815

theorem ratio_of_wealth (P W : ℝ) (hP : P > 0) (hW : W > 0) : 
  let wX := (0.40 * W) / (0.20 * P)
  let wY := (0.30 * W) / (0.10 * P)
  (wX / wY) = 2 / 3 := 
by
  sorry

end ratio_of_wealth_l205_205815


namespace power_equality_l205_205147

-- Definitions based on conditions
def nine := 3^2

-- Theorem stating the given mathematical problem
theorem power_equality : nine^4 = 3^8 := by
  sorry

end power_equality_l205_205147


namespace sector_area_proof_l205_205074

-- Define variables for the central angle, arc length, and derived radius
variables (θ L : ℝ) (r A: ℝ)

-- Define the conditions given in the problem
def central_angle_condition : Prop := θ = 2
def arc_length_condition : Prop := L = 4
def radius_condition : Prop := r = L / θ

-- Define the formula for the area of the sector
def area_of_sector_condition : Prop := A = (1 / 2) * r^2 * θ

-- The theorem that needs to be proved
theorem sector_area_proof :
  central_angle_condition θ ∧ arc_length_condition L ∧ radius_condition θ L r ∧ area_of_sector_condition r θ A → A = 4 :=
by
  sorry

end sector_area_proof_l205_205074


namespace recyclable_cans_and_bottles_collected_l205_205139

-- Define the conditions in Lean
def people_at_picnic : ℕ := 90
def soda_cans : ℕ := 50
def plastic_bottles_sparkling_water : ℕ := 50
def glass_bottles_juice : ℕ := 50
def guests_drank_soda : ℕ := people_at_picnic / 2
def guests_drank_sparkling_water : ℕ := people_at_picnic / 3
def juice_consumed : ℕ := (glass_bottles_juice * 4) / 5

-- The theorem statement
theorem recyclable_cans_and_bottles_collected :
  (soda_cans + guests_drank_sparkling_water + juice_consumed) = 120 :=
by
  sorry

end recyclable_cans_and_bottles_collected_l205_205139


namespace evaluate_expression_l205_205057

theorem evaluate_expression (a : ℚ) (h : a = 3/2) : 
  ((5 * a^2 - 13 * a + 4) * (2 * a - 3)) = 0 := by
  sorry

end evaluate_expression_l205_205057


namespace worker_efficiency_l205_205996

theorem worker_efficiency (W_p W_q : ℚ) 
  (h1 : W_p = 1 / 24) 
  (h2 : W_p + W_q = 1 / 14) :
  (W_p - W_q) / W_q * 100 = 40 :=
by
  sorry

end worker_efficiency_l205_205996


namespace sister_weight_difference_is_12_l205_205784

-- Define Antonio's weight
def antonio_weight : ℕ := 50

-- Define the combined weight of Antonio and his sister
def combined_weight : ℕ := 88

-- Define the weight of Antonio's sister
def sister_weight : ℕ := combined_weight - antonio_weight

-- Define the weight difference
def weight_difference : ℕ := antonio_weight - sister_weight

-- Theorem statement to prove the weight difference is 12 kg
theorem sister_weight_difference_is_12 : weight_difference = 12 := by
  sorry

end sister_weight_difference_is_12_l205_205784


namespace proof_problem_l205_205552

variables (Books : Type) (Available : Books -> Prop)

def all_books_available : Prop := ∀ b : Books, Available b
def some_books_not_available : Prop := ∃ b : Books, ¬ Available b
def not_all_books_available : Prop := ¬ all_books_available Books Available

theorem proof_problem (h : ¬ all_books_available Books Available) : 
  some_books_not_available Books Available ∧ not_all_books_available Books Available :=
by 
  sorry

end proof_problem_l205_205552


namespace hamburgers_served_l205_205780

def hamburgers_made : ℕ := 9
def hamburgers_leftover : ℕ := 6

theorem hamburgers_served : ∀ (total : ℕ) (left : ℕ), total = hamburgers_made → left = hamburgers_leftover → total - left = 3 := 
by
  intros total left h_total h_left
  rw [h_total, h_left]
  rfl

end hamburgers_served_l205_205780


namespace machines_working_together_l205_205140

theorem machines_working_together (x : ℝ) :
  let R_time := x + 4
  let Q_time := x + 9
  let P_time := x + 12
  (1 / P_time + 1 / Q_time + 1 / R_time) = 1 / x ↔ x = 1 := 
by
  sorry

end machines_working_together_l205_205140


namespace numerical_puzzle_solution_l205_205899

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l205_205899


namespace mary_change_received_l205_205247

def cost_of_adult_ticket : ℝ := 2
def cost_of_child_ticket : ℝ := 1
def discount_first_child : ℝ := 0.5
def discount_second_child : ℝ := 0.75
def discount_third_child : ℝ := 1
def sales_tax_rate : ℝ := 0.08
def amount_paid : ℝ := 20

def total_ticket_cost_before_tax : ℝ :=
  cost_of_adult_ticket + (cost_of_child_ticket * discount_first_child) + 
  (cost_of_child_ticket * discount_second_child) + (cost_of_child_ticket * discount_third_child)

def sales_tax : ℝ :=
  total_ticket_cost_before_tax * sales_tax_rate

def total_ticket_cost_with_tax : ℝ :=
  total_ticket_cost_before_tax + sales_tax

def change_received : ℝ :=
  amount_paid - total_ticket_cost_with_tax

theorem mary_change_received :
  change_received = 15.41 :=
by
  sorry

end mary_change_received_l205_205247


namespace fraction_of_menu_safely_eaten_l205_205329

-- Given conditions
def VegetarianDishes := 6
def GlutenContainingVegetarianDishes := 5
def TotalDishes := 3 * VegetarianDishes

-- Derived information
def GlutenFreeVegetarianDishes := VegetarianDishes - GlutenContainingVegetarianDishes

-- Question: What fraction of the menu can Sarah safely eat?
theorem fraction_of_menu_safely_eaten : 
  (GlutenFreeVegetarianDishes / TotalDishes) = 1 / 18 :=
by
  sorry

end fraction_of_menu_safely_eaten_l205_205329


namespace average_children_in_families_with_children_l205_205881

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205881


namespace find_a_l205_205544

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 2

theorem find_a (a : ℝ) (h : deriv (f a) (-1) = 4) : a = 10 / 3 :=
by {
  sorry
}

end find_a_l205_205544


namespace distance_between_foci_of_ellipse_l205_205747

theorem distance_between_foci_of_ellipse :
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  distance = 2 * Real.sqrt 61 :=
by
  let F1 := (4, -3)
  let F2 := (-6, 9)
  let distance := Real.sqrt ( ((4 - (-6))^2) + ((-3 - 9)^2) )
  sorry

end distance_between_foci_of_ellipse_l205_205747


namespace polar_to_rectangular_l205_205050

open Real
open Real.Angle

theorem polar_to_rectangular (r θ : ℝ) (h_r : r = 4) (h_θ : θ = 5 * pi / 3) :
    (r * cos θ, r * sin θ) = (2, -2 * sqrt 3) := by
  rw [h_r, h_θ]
  have h_cos : cos (5 * pi / 3) = 1 / 2 := sorry
  have h_sin : sin (5 * pi / 3) = -sqrt 3 / 2 := sorry
  rw [h_cos, h_sin]
  norm_num
  split
  norm_num
  ring
  norm_num
  ring

end polar_to_rectangular_l205_205050


namespace AB_eq_B_exp_V_l205_205890

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l205_205890


namespace vector_dot_product_l205_205391

open Matrix

section VectorDotProduct

variables (A : ℝ × ℝ) (B : ℝ × ℝ) (C : ℝ × ℝ)
variables (E : ℝ × ℝ) (F : ℝ × ℝ)

def vector_sub (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 - Q.1, P.2 - Q.2)
def vector_add (P Q : ℝ × ℝ) : ℝ × ℝ := (P.1 + Q.1, P.2 + Q.2)
def scalar_mul (k : ℝ) (P : ℝ × ℝ) : ℝ × ℝ := (k * P.1, k * P.2)
def dot_product (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

axiom A_coord : A = (1, 2)
axiom B_coord : B = (2, -1)
axiom C_coord : C = (2, 2)
axiom E_is_trisection : vector_add (vector_sub B A) (scalar_mul (1/3) (vector_sub C B)) = E
axiom F_is_trisection : vector_add (vector_sub B A) (scalar_mul (2/3) (vector_sub C B)) = F

theorem vector_dot_product : dot_product (vector_sub E A) (vector_sub F A) = 3 := by
  sorry

end VectorDotProduct

end vector_dot_product_l205_205391


namespace mean_of_second_set_l205_205220

theorem mean_of_second_set (x : ℝ)
  (H1 : (28 + x + 70 + 88 + 104) / 5 = 67) :
  (50 + 62 + 97 + 124 + x) / 5 = 75.6 :=
sorry

end mean_of_second_set_l205_205220


namespace total_number_of_matches_l205_205225

-- Define the total number of teams
def numberOfTeams : ℕ := 10

-- Define the number of matches each team competes against each other team
def matchesPerPair : ℕ := 4

-- Calculate the total number of unique matches
def calculateUniqueMatches (teams : ℕ) : ℕ :=
  (teams * (teams - 1)) / 2

-- Main statement to be proved
theorem total_number_of_matches : calculateUniqueMatches numberOfTeams * matchesPerPair = 180 := by
  -- Placeholder for the proof
  sorry

end total_number_of_matches_l205_205225


namespace minimum_cost_l205_205430

theorem minimum_cost (price_pen_A price_pen_B price_notebook_A price_notebook_B : ℕ) 
  (discount_B : ℚ) (num_pens num_notebooks : ℕ)
  (h_price_pen : price_pen_A = 10) (h_price_notebook : price_notebook_A = 2)
  (h_discount : discount_B = 0.9) (h_num_pens : num_pens = 4) (h_num_notebooks : num_notebooks = 24) :
  ∃ (min_cost : ℕ), min_cost = 76 :=
by
  -- The conditions should be used here to construct the min_cost
  sorry

end minimum_cost_l205_205430


namespace compare_abc_l205_205910

noncomputable def a : ℝ := 1 / Real.log 3 / Real.log 2
noncomputable def b : ℝ := Real.exp 0.5
noncomputable def c : ℝ := Real.log 2

theorem compare_abc : b > c ∧ c > a := by
  sorry

end compare_abc_l205_205910


namespace calculate_expression_l205_205504

theorem calculate_expression :
  (2^3 * 3 * 5) + (18 / 2) = 129 := by
  -- Proof skipped
  sorry

end calculate_expression_l205_205504


namespace binom_18_4_eq_3060_l205_205799

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l205_205799


namespace total_wheels_at_park_l205_205173

-- Define the problem based on the given conditions
def num_bicycles : ℕ := 6
def num_tricycles : ℕ := 15
def wheels_per_bicycle : ℕ := 2
def wheels_per_tricycle : ℕ := 3

-- Statement to prove the total number of wheels is 57
theorem total_wheels_at_park : (num_bicycles * wheels_per_bicycle + num_tricycles * wheels_per_tricycle) = 57 := by
  -- This will be filled in with the proof.
  sorry

end total_wheels_at_park_l205_205173


namespace find_other_divisor_l205_205189

theorem find_other_divisor (x : ℕ) (h : x ≠ 35) (h1 : 386 % 35 = 1) (h2 : 386 % x = 1) : x = 11 :=
sorry

end find_other_divisor_l205_205189


namespace AB_eq_B_exp_V_l205_205893

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l205_205893


namespace rotation_matrix_150_eq_l205_205345

open Real Matrix

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![cos θ, -sin θ], ![sin θ, cos θ]]

noncomputable def angle_150 : ℝ := 150 * (π / 180) -- Convert 150 degrees to radians
  
theorem rotation_matrix_150_eq :
  rotation_matrix angle_150 = ![![-(sqrt 3 / 2), -1 / 2], ![1 / 2, -(sqrt 3 / 2)]] :=
by
  simp [rotation_matrix, angle_150]
  sorry

end rotation_matrix_150_eq_l205_205345


namespace find_ordered_pairs_l205_205492

theorem find_ordered_pairs (a b x : ℕ) (h1 : b > a) (h2 : a + b = 15) (h3 : (a - 2 * x) * (b - 2 * x) = 2 * a * b / 3) :
  (a, b) = (8, 7) :=
by
  sorry

end find_ordered_pairs_l205_205492


namespace relationship_between_ys_l205_205089

theorem relationship_between_ys :
  ∀ (y1 y2 y3 : ℝ),
    (y1 = - (6 / (-2))) ∧ (y2 = - (6 / (-1))) ∧ (y3 = - (6 / 3)) →
    y2 > y1 ∧ y1 > y3 :=
by sorry

end relationship_between_ys_l205_205089


namespace max_value_of_expression_l205_205418

-- Define the real numbers p, q, r and the conditions
variables {p q r : ℝ}

-- Define the main goal
theorem max_value_of_expression 
(h : 9 * p^2 + 4 * q^2 + 25 * r^2 = 4) : 
  (5 * p + 3 * q + 10 * r) ≤ (10 * Real.sqrt 13 / 3) :=
sorry

end max_value_of_expression_l205_205418


namespace total_area_equals_total_frequency_l205_205271

-- Definition of frequency and frequency distribution histogram
def frequency_distribution_histogram (frequencies : List ℕ) := ∀ i, (i < frequencies.length) → ℕ

-- Definition that the total area of the small rectangles is the sum of the frequencies
def total_area_of_rectangles (frequencies : List ℕ) : ℕ := frequencies.sum

-- Theorem stating the equivalence
theorem total_area_equals_total_frequency (frequencies : List ℕ) :
  total_area_of_rectangles frequencies = frequencies.sum := 
by
  sorry

end total_area_equals_total_frequency_l205_205271


namespace point_K_is_intersection_of_diagonals_l205_205231

variable {K A B C D : Type}

/-- A quadrilateral is circumscribed if there exists a circle within which all four vertices lie. -/
noncomputable def is_circumscribed (A B C D : Type) : Prop :=
sorry

/-- Distances from point K to the sides of the quadrilateral ABCD are proportional to the lengths of those sides. -/
noncomputable def proportional_distances (K A B C D : Type) : Prop :=
sorry

/-- A point is the intersection point of the diagonals AC and BD of quadrilateral ABCD. -/
noncomputable def intersection_point_of_diagonals (K A C B D : Type) : Prop :=
sorry

theorem point_K_is_intersection_of_diagonals 
  (K A B C D : Type) 
  (circumQ : is_circumscribed A B C D) 
  (propDist : proportional_distances K A B C D) 
  : intersection_point_of_diagonals K A C B D :=
sorry

end point_K_is_intersection_of_diagonals_l205_205231


namespace statement_two_even_function_statement_four_minimum_value_l205_205192

open Real

noncomputable def f (x : ℝ) : ℝ := exp (sin x) + exp (cos x)

-- Statement 2: $y=f(x+\frac{\pi}{4})$ is an even function
theorem statement_two_even_function :
  ∀ x, f(x + π / 4) = f(-x + π / 4) :=
by
  sorry

-- Statement 4: The minimum value of $f(x)$ is $2e^{-\frac{\sqrt{2}}{2}}$
theorem statement_four_minimum_value :
  ∃ m, (∀ x, f x ≥ m) ∧ (∃ x, f x = 2 * exp(-sqrt 2 / 2)) :=
by
  sorry

end statement_two_even_function_statement_four_minimum_value_l205_205192


namespace rectangle_exists_l205_205717

theorem rectangle_exists (n : ℕ) (h_n : 0 < n)
  (marked : Finset (Fin n × Fin n))
  (h_marked : marked.card ≥ n * (Real.sqrt n + 0.5)) :
  ∃ (r1 r2 : Fin n) (c1 c2 : Fin n), r1 ≠ r2 ∧ c1 ≠ c2 ∧ 
    ((r1, c1) ∈ marked ∧ (r1, c2) ∈ marked ∧ (r2, c1) ∈ marked ∧ (r2, c2) ∈ marked) :=
  sorry

end rectangle_exists_l205_205717


namespace positive_root_of_real_root_l205_205141

theorem positive_root_of_real_root (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h1 : b^2 - 4*a*c ≥ 0) (h2 : c^2 - 4*b*a ≥ 0) (h3 : a^2 - 4*c*b ≥ 0) : 
  ∀ (p q r : ℝ), (p = a ∧ q = b ∧ r = c) ∨ (p = b ∧ q = c ∧ r = a) ∨ (p = c ∧ q = a ∧ r = b) →
  (∃ x : ℝ, x > 0 ∧ p*x^2 + q*x + r = 0) :=
by 
  sorry

end positive_root_of_real_root_l205_205141


namespace largest_integer_of_five_with_product_12_l205_205725

theorem largest_integer_of_five_with_product_12 (a b c d e : ℤ) (h : a * b * c * d * e = 12) (h_diff : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ d ∧ b ≠ e ∧ c ≠ e) : 
  max a (max b (max c (max d e))) = 3 :=
sorry

end largest_integer_of_five_with_product_12_l205_205725


namespace total_opponents_runs_l205_205991

theorem total_opponents_runs (team_scores : List ℕ) (opponent_scores : List ℕ) :
  team_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] →
  ∃ lost_games won_games opponent_lost_scores opponent_won_scores,
    lost_games = [1, 3, 5, 7, 9, 11] ∧
    won_games = [2, 4, 6, 8, 10, 12] ∧
    (∀ (t : ℕ), t ∈ lost_games → ∃ o : ℕ, o = t + 1 ∧ o ∈ opponent_scores) ∧
    (∀ (t : ℕ), t ∈ won_games → ∃ o : ℕ, o = t / 2 ∧ o ∈ opponent_scores) ∧
    opponent_scores = opponent_lost_scores ++ opponent_won_scores ∧
    opponent_lost_scores = [2, 4, 6, 8, 10, 12] ∧
    opponent_won_scores = [1, 2, 3, 4, 5, 6] →
  opponent_scores.sum = 63 :=
by
  sorry

end total_opponents_runs_l205_205991


namespace num_int_values_x_l205_205008

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l205_205008


namespace problem1_problem2_problem3_problem4_l205_205047

theorem problem1 : (-23 + 13 - 12) = -22 := 
by sorry

theorem problem2 : ((-2)^3 / 4 + 3 * (-5)) = -17 := 
by sorry

theorem problem3 : (-24 * (1/2 - 3/4 - 1/8)) = 9 := 
by sorry

theorem problem4 : ((2 - 7) / 5^2 + (-1)^2023 * (1/10)) = -3/10 := 
by sorry

end problem1_problem2_problem3_problem4_l205_205047


namespace find_bottle_caps_l205_205326

variable (B : ℕ) -- Number of bottle caps Danny found at the park.

-- Conditions
variable (current_wrappers : ℕ := 67) -- Danny has 67 wrappers in his collection now.
variable (current_bottle_caps : ℕ := 35) -- Danny has 35 bottle caps in his collection now.
variable (found_wrappers : ℕ := 18) -- Danny found 18 wrappers at the park.
variable (more_wrappers_than_bottle_caps : ℕ := 32) -- Danny has 32 more wrappers than bottle caps.

-- Given the conditions, prove that Danny found 18 bottle caps at the park.
theorem find_bottle_caps (h1 : current_wrappers = current_bottle_caps + more_wrappers_than_bottle_caps)
                         (h2 : current_bottle_caps - B + found_wrappers = current_wrappers - more_wrappers_than_bottle_caps - B) :
  B = 18 :=
by
  sorry

end find_bottle_caps_l205_205326


namespace area_of_common_region_l205_205493

theorem area_of_common_region (β : ℝ) (h1 : 0 < β ∧ β < π / 2) (h2 : Real.cos β = 3 / 5) :
  ∃ (area : ℝ), area = 4 / 9 := 
by 
  sorry

end area_of_common_region_l205_205493


namespace dealer_gross_profit_l205_205621

noncomputable def computeGrossProfit (purchasePrice initialMarkupRate discountRate salesTaxRate: ℝ) : ℝ :=
  let initialSellingPrice := purchasePrice / (1 - initialMarkupRate)
  let discount := discountRate * initialSellingPrice
  let discountedPrice := initialSellingPrice - discount
  let salesTax := salesTaxRate * discountedPrice
  let finalSellingPrice := discountedPrice + salesTax
  finalSellingPrice - purchasePrice - discount

theorem dealer_gross_profit 
  (purchasePrice : ℝ)
  (initialMarkupRate : ℝ)
  (discountRate : ℝ)
  (salesTaxRate : ℝ) 
  (grossProfit : ℝ) :
  purchasePrice = 150 →
  initialMarkupRate = 0.25 →
  discountRate = 0.10 →
  salesTaxRate = 0.05 →
  grossProfit = 19 →
  computeGrossProfit purchasePrice initialMarkupRate discountRate salesTaxRate = grossProfit :=
  by
    intros hp hm hd hs hg
    rw [hp, hm, hd, hs, hg]
    rw [computeGrossProfit]
    sorry

end dealer_gross_profit_l205_205621


namespace find_side_PR_of_PQR_l205_205945

open Real

noncomputable def triangle_PQR (PQ PM PH PR : ℝ) : Prop :=
  let HQ := sqrt (PQ^2 - PH^2)
  let MH := sqrt (PM^2 - PH^2)
  let MQ := MH - HQ
  let RH := HQ + 2 * MQ
  PR = sqrt (PH^2 + RH^2)

theorem find_side_PR_of_PQR (PQ PM PH : ℝ) (h_PQ : PQ = 3) (h_PM : PM = sqrt 14) (h_PH : PH = sqrt 5) (h_angle : ∀ QPR PRQ : ℝ, QPR + PRQ < 90) : 
  triangle_PQR PQ PM PH (sqrt 21) :=
by
  rw [h_PQ, h_PM, h_PH]
  exact sorry

end find_side_PR_of_PQR_l205_205945


namespace route_down_distance_l205_205613

noncomputable def rate_up : ℝ := 3
noncomputable def time_up : ℝ := 2
noncomputable def time_down : ℝ := 2
noncomputable def rate_down := 1.5 * rate_up

theorem route_down_distance : rate_down * time_down = 9 := by
  sorry

end route_down_distance_l205_205613


namespace numerical_puzzle_solution_l205_205900

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l205_205900


namespace probability_heads_at_least_10_in_12_flips_l205_205741

theorem probability_heads_at_least_10_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 79 / 4096 := by
  sorry

end probability_heads_at_least_10_in_12_flips_l205_205741


namespace triangle_altitude_l205_205962

theorem triangle_altitude {A b h : ℝ} (hA : A = 720) (hb : b = 40) (hArea : A = 1 / 2 * b * h) : h = 36 :=
by
  sorry

end triangle_altitude_l205_205962


namespace age_problem_l205_205678

theorem age_problem 
  (P R J M : ℕ)
  (h1 : P = 1 / 2 * R)
  (h2 : R = J + 7)
  (h3 : J + 12 = 3 * P)
  (h4 : M = J + 17)
  (h5 : M = 2 * R + 4) : 
  P = 5 ∧ R = 10 ∧ J = 3 ∧ M = 24 :=
by sorry

end age_problem_l205_205678


namespace day_of_week_after_n_days_l205_205048

theorem day_of_week_after_n_days (birthday : ℕ) (n : ℕ) (day_of_week : ℕ) :
  birthday = 4 → (n % 7) = 2 → day_of_week = 6 :=
by sorry

end day_of_week_after_n_days_l205_205048


namespace total_cups_needed_l205_205268

-- Define the known conditions
def ratio_butter : ℕ := 2
def ratio_flour : ℕ := 3
def ratio_sugar : ℕ := 5
def total_sugar_in_cups : ℕ := 10

-- Define the parts-to-cups conversion
def cup_per_part := total_sugar_in_cups / ratio_sugar

-- Define the amounts of each ingredient in cups
def butter_in_cups := ratio_butter * cup_per_part
def flour_in_cups := ratio_flour * cup_per_part
def sugar_in_cups := ratio_sugar * cup_per_part

-- Define the total number of cups
def total_cups := butter_in_cups + flour_in_cups + sugar_in_cups

-- Theorem to prove
theorem total_cups_needed : total_cups = 20 := by
  sorry

end total_cups_needed_l205_205268


namespace ab_value_l205_205098

theorem ab_value 
  (a b : ℝ) 
  (hx : 2 = b + 1) 
  (hy : a = -3) : 
  a * b = -3 :=
by
  sorry

end ab_value_l205_205098


namespace average_waiting_time_for_first_bite_l205_205512

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l205_205512


namespace avg_children_with_kids_l205_205826

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205826


namespace smallest_c_l205_205389

variable {f : ℝ → ℝ}

def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (f 0 = 0) ∧ (f 1 = 1) ∧ (∀ x, 0 ≤ x ∧ x ≤ 1 → 0 ≤ f x) ∧ (∀ x1 x2, 0 ≤ x1 ∧ 0 ≤ x2 ∧ x1 + x2 ≤ 1 → f (x1 + x2) ≥ f x1 + f x2)

theorem smallest_c (f : ℝ → ℝ) (h : satisfies_conditions f) : (∀ x, 0 ≤ x ∧ x ≤ 1 → f x ≤ 2 * x) ∧ (∀ c, c < 2 → ∃ x, 0 < x ∧ x ≤ 1 ∧ ¬ (f x ≤ c * x)) :=
by
  sorry

end smallest_c_l205_205389


namespace johns_initial_money_l205_205109

/-- John's initial money given that he gives 3/8 to his mother and 3/10 to his father,
and he has $65 left after giving away the money. Prove that he initially had $200. -/
theorem johns_initial_money 
  (M : ℕ)
  (h_left : (M : ℚ) - (3 / 8) * M - (3 / 10) * M = 65) :
  M = 200 :=
sorry

end johns_initial_money_l205_205109


namespace number_of_terms_in_expansion_is_12_l205_205923

-- Define the polynomials
def p (x y z : ℕ) := x + y + z
def q (u v w x : ℕ) := u + v + w + x

-- Define the number of terms in a polynomial as a function.
def numberOfTerms (poly : Polynomial ℕ) : ℕ :=
  poly.degree + 1

-- Prove the number of terms in expansion of (x + y + z)(u + v + w + x) is 12.
theorem number_of_terms_in_expansion_is_12 (x y z u v w : ℕ) :
  numberOfTerms (p x y z * q u v w x) = 12 := by
  sorry

end number_of_terms_in_expansion_is_12_l205_205923


namespace rotation_matrix_150_degrees_l205_205359

theorem rotation_matrix_150_degrees :
  let θ := 150 * Real.pi / 180
  let cos150 := Real.cos θ
  let sin150 := Real.sin θ
  (cos150, sin150) = (-Real.sqrt 3 / 2, 1 / 2) →
  (Matrix.vecCons (Matrix.vecCons (-Real.sqrt 3 / 2) (-1 / 2) Matrix.nil) (Matrix.vecCons (1 / 2) (-Real.sqrt 3 / 2) Matrix.nil) 
    = Matrix.vecCons (Matrix.vecCons (-Real.sqrt 3 / 2) (-1 / 2) Matrix.nil) (Matrix.vecCons (1 / 2) (-Real.sqrt 3 / 2) Matrix.nil)) := by
  sorry

end rotation_matrix_150_degrees_l205_205359


namespace sqrt_eq_cond_l205_205117

theorem sqrt_eq_cond (a b c : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ a) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
    (not_perfect_square_a : ¬(∃ n : ℕ, n * n = a)) (not_perfect_square_b : ¬(∃ n : ℕ, n * n = b))
    (not_perfect_square_c : ¬(∃ n : ℕ, n * n = c)) :
    (Real.sqrt a + Real.sqrt b = Real.sqrt c) →
    (2 * Real.sqrt (a * b) = c - (a + b) ∧ (∃ k : ℕ, a * b = k * k)) :=
sorry

end sqrt_eq_cond_l205_205117


namespace complement_set_P_l205_205204

open Set

theorem complement_set_P (P : Set ℝ) (hP : P = {x : ℝ | x ≥ 1}) : Pᶜ = {x : ℝ | x < 1} :=
sorry

end complement_set_P_l205_205204


namespace proof_probability_second_science_given_first_arts_l205_205937

noncomputable def probability_second_science_given_first_arts : ℚ :=
  let total_questions := 5
  let science_questions := 3
  let arts_questions := 2

  -- Event A: drawing an arts question in the first draw.
  let P_A := arts_questions / total_questions

  -- Event AB: drawing an arts question in the first draw and a science question in the second draw.
  let P_AB := (arts_questions / total_questions) * (science_questions / (total_questions - 1))

  -- Conditional probability P(B|A): drawing a science question in the second draw given drawing an arts question in the first draw.
  P_AB / P_A

theorem proof_probability_second_science_given_first_arts :
  probability_second_science_given_first_arts = 3 / 4 :=
by
  -- Lean does not include the proof in the statement as required.
  sorry

end proof_probability_second_science_given_first_arts_l205_205937


namespace children_on_bus_after_events_l205_205982

-- Definition of the given problem parameters
def initial_children : Nat := 21
def got_off : Nat := 10
def got_on : Nat := 5

-- The theorem we want to prove
theorem children_on_bus_after_events : initial_children - got_off + got_on = 16 :=
by
  -- This is where the proof would go, but we leave it as sorry for now
  sorry

end children_on_bus_after_events_l205_205982


namespace impossible_odd_n_m_even_sum_l205_205949

theorem impossible_odd_n_m_even_sum (n m : ℤ) (h : (n^2 + m^2 + n*m) % 2 = 0) : ¬ (n % 2 = 1 ∧ m % 2 = 1) :=
by sorry

end impossible_odd_n_m_even_sum_l205_205949


namespace card_arrangements_l205_205732

open Finset

/-- The number of different arrangements of six letter cards
    (A, B, C, D, E, F) arranged in a row such that A is at the left end 
    and F at the right end is 24. -/
theorem card_arrangements : 
  let letters := {'A', 'B', 'C', 'D', 'E', 'F'} in 
  ∃ arrangements : Finset (Finset Char), 
    ∃ left := 'A', ∃ right := 'F', let remaining := erase (erase letters left) right in
    arrangements = (permutations remaining) →
    card arrangements = 24 := sorry

end card_arrangements_l205_205732


namespace evaluate_expression_l205_205054

noncomputable def w : ℂ := Complex.exp (2 * Real.pi * Complex.I / 13)

theorem evaluate_expression : 
  let p := ∏ i in Finset.range 12, (3 - w ^ (i + 1))
  p = 2391483 := 
by sorry

end evaluate_expression_l205_205054


namespace find_m_l205_205113

variable {y m : ℝ} -- define variables y and m in the reals

-- define the logarithmic conditions
axiom log8_5_eq_y : log 8 5 = y
axiom log2_125_eq_my : log 2 125 = m * y

-- state the theorem to prove m equals 9
theorem find_m (log8_5_eq_y : log 8 5 = y) (log2_125_eq_my : log 2 125 = m * y) : m = 9 := by
  sorry

end find_m_l205_205113


namespace part1_max_min_part2_cos_value_l205_205386

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (Real.pi * x + Real.pi / 6)

theorem part1_max_min (x : ℝ) (hx : -1/2 ≤ x ∧ x ≤ 1/2) : 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = 2) ∧ 
  (∃ xₘ, (xₘ ∈ Set.Icc (-1/2) (1/2)) ∧ f xₘ = -Real.sqrt 3) :=
sorry

theorem part2_cos_value (α : ℝ) (h : f (α / (2 * Real.pi)) = 1/4) : 
  Real.cos (2 * Real.pi / 3 - α) = -31/32 :=
sorry

end part1_max_min_part2_cos_value_l205_205386


namespace lcm_gcd_product_12_15_l205_205046

theorem lcm_gcd_product_12_15 : 
  let a := 12
  let b := 15
  lcm a b * gcd a b = 180 :=
by
  sorry

end lcm_gcd_product_12_15_l205_205046


namespace john_alone_finishes_in_48_days_l205_205288

theorem john_alone_finishes_in_48_days (J R : ℝ) (h1 : J + R = 1 / 24)
  (h2 : 16 * (J + R) = 16 / 24) (h3 : ∀ T : ℝ, J * T = 1 → T = 48) : 
  (J = 1 / 48) → (∀ T : ℝ, J * T = 1 → T = 48) :=
by
  intro hJohn
  sorry

end john_alone_finishes_in_48_days_l205_205288


namespace inequality_solution_sets_l205_205921

theorem inequality_solution_sets (a b m : ℝ) (h_sol_set : ∀ x, x^2 - a * x - 2 > 0 ↔ x < -1 ∨ x > b) (hb : b > -1) (hm : m > -1 / 2) :
  a = 1 ∧ b = 2 ∧ 
  (if m > 0 then ∀ x, (x < -1/m ∨ x > 2) ↔ (mx + 1) * (x - 2) > 0 
   else if m = 0 then ∀ x, x > 2 ↔ (mx + 1) * (x - 2) > 0 
   else ∀ x, (2 < x ∧ x < -1/m) ↔ (mx + 1) * (x - 2) > 0) :=
by
  sorry

end inequality_solution_sets_l205_205921


namespace bin_expected_value_l205_205983

theorem bin_expected_value (m : ℕ) (h : (21 - 4 * m) / (7 + m) = 1) : m = 3 := 
by {
  sorry
}

end bin_expected_value_l205_205983


namespace rotation_matrix_150_degrees_l205_205347

theorem rotation_matrix_150_degrees :
  ∃ (R : Matrix (Fin 2) (Fin 2) ℝ),
    R = Matrix.ofFn
      (λ i j, match (i, j) with
              | (0, 0) => -(Real.sqrt 3) / 2
              | (0, 1) => -1 / 2
              | (1, 0) => 1 / 2
              | (1, 1) => -(Real.sqrt 3) / 2
              | _ => 0) :=
begin
  sorry
end

end rotation_matrix_150_degrees_l205_205347


namespace average_children_in_families_with_children_l205_205862

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205862


namespace problem1_problem2_problem3_problem4_l205_205616

theorem problem1 : (5 / 16) - (3 / 16) + (7 / 16) = 9 / 16 := by
  sorry

theorem problem2 : (3 / 12) - (4 / 12) + (6 / 12) = 5 / 12 := by
  sorry

theorem problem3 : 64 + 27 + 81 + 36 + 173 + 219 + 136 = 736 := by
  sorry

theorem problem4 : (2 : ℚ) - (8 / 9) - (1 / 9) + (1 + 98 / 99) = 2 + 98 / 99 := by
  sorry

end problem1_problem2_problem3_problem4_l205_205616


namespace M_k_max_l205_205375

noncomputable def J_k (k : ℕ) : ℕ := 5^(k+3) * 2^(k+3) + 648

def M (k : ℕ) : ℕ := 
  if k < 3 then k + 3
  else 3

theorem M_k_max (k : ℕ) : M k = 3 :=
by sorry

end M_k_max_l205_205375


namespace integer_solution_system_eq_det_l205_205118

theorem integer_solution_system_eq_det (a b c d : ℤ) 
  (h : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) : 
  a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by
  sorry

end integer_solution_system_eq_det_l205_205118


namespace tony_running_speed_l205_205705

theorem tony_running_speed :
  (∀ R : ℝ, (4 / 2 * 60) + 2 * ((4 / R) * 60) = 168 → R = 10) :=
sorry

end tony_running_speed_l205_205705


namespace total_weight_of_packages_l205_205426

theorem total_weight_of_packages (x y z w : ℕ) (h1 : x + y + z = 150) (h2 : y + z + w = 160) (h3 : z + w + x = 170) :
  x + y + z + w = 160 :=
by sorry

end total_weight_of_packages_l205_205426


namespace original_price_of_shoes_l205_205752

theorem original_price_of_shoes (
  initial_amount : ℝ := 74
) (sweater_cost : ℝ := 9) (tshirt_cost : ℝ := 11) 
  (final_amount_after_refund : ℝ := 51)
  (refund_percentage : ℝ := 0.90)
  (S : ℝ) :
  (initial_amount - sweater_cost - tshirt_cost - S + refund_percentage * S = final_amount_after_refund) -> 
  S = 30 := 
by
  intros h
  sorry

end original_price_of_shoes_l205_205752


namespace expected_value_linear_combination_l205_205196

variable (ξ η : ℝ)
variable (E : ℝ → ℝ)
axiom E_lin (a b : ℝ) (X Y : ℝ) : E (a * X + b * Y) = a * E X + b * E Y

axiom E_ξ : E ξ = 10
axiom E_η : E η = 3

theorem expected_value_linear_combination : E (3 * ξ + 5 * η) = 45 := by
  sorry

end expected_value_linear_combination_l205_205196


namespace inequality_holds_l205_205655

theorem inequality_holds (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 :=
by {
  sorry
}

end inequality_holds_l205_205655


namespace article_usage_correct_l205_205970

def blank1 := "a"
def blank2 := ""  -- Representing "不填" (no article) as an empty string for simplicity

theorem article_usage_correct :
  (blank1 = "a" ∧ blank2 = "") :=
by
  sorry

end article_usage_correct_l205_205970


namespace problem_l205_205520

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) :
  (f (x + 2) + f x = 0) →
  (∀ x, f (-(x - 1)) = -f (x - 1)) →
  (
    (∀ e, ¬(e > 0 ∧ ∀ x, f (x + e) = f x)) ∧
    (∀ x, f (x + 1) = f (-x + 1)) ∧
    (¬(∀ x, f x = f (-x)))
  ) :=
by
  sorry

end problem_l205_205520


namespace div_30_prime_ge_7_l205_205642

theorem div_30_prime_ge_7 (p : ℕ) (hp_prime : Nat.Prime p) (hp_ge_7 : p ≥ 7) : 30 ∣ (p^2 - 1) := 
sorry

end div_30_prime_ge_7_l205_205642


namespace binom_18_4_eq_3060_l205_205803

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l205_205803


namespace sam_weight_l205_205738

theorem sam_weight (Tyler Sam Peter : ℕ) : 
  (Peter = 65) →
  (Peter = Tyler / 2) →
  (Tyler = Sam + 25) →
  Sam = 105 :=
  by
  intros hPeter1 hPeter2 hTyler
  sorry

end sam_weight_l205_205738


namespace regular_polygon_sides_l205_205143

theorem regular_polygon_sides (A B C : Type) [Inhabited A] [Inhabited B] [Inhabited C]
  (angle_A angle_B angle_C : ℝ)
  (is_circle_inscribed_triangle : angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A ∧ angle_B + angle_C + angle_A = 180)
  (n : ℕ)
  (is_regular_polygon : B = C ∧ angle_B = 3 * angle_A ∧ angle_C = 3 * angle_A) :
  n = 9 := sorry

end regular_polygon_sides_l205_205143


namespace total_telephone_bill_second_month_l205_205330

variable (F C : ℝ)

-- Elvin's total telephone bill for January is 40 dollars
axiom january_bill : F + C = 40

-- The charge for calls in the second month is twice the charge for calls in January
axiom second_month_call_charge : ∃ C2, C2 = 2 * C

-- Proof that the total telephone bill for the second month is 40 + C
theorem total_telephone_bill_second_month : 
  ∃ S, S = F + 2 * C ∧ S = 40 + C :=
sorry

end total_telephone_bill_second_month_l205_205330


namespace factor_expression_l205_205525

theorem factor_expression (x : ℝ) : 
  5 * x * (x - 2) + 9 * (x - 2) - 4 * (x - 2) = 5 * (x - 2) * (x + 1) :=
by
  -- proof goes here
  sorry

end factor_expression_l205_205525


namespace perpendicular_line_through_point_l205_205665

theorem perpendicular_line_through_point (m t : ℝ) (h : 2 * m^2 + m + t = 0) :
  m = 1 → t = -3 → (∀ x y : ℝ, m^2 * x + m * y + t = 0 ↔ x + y - 3 = 0) :=
by
  intros hm ht
  subst hm
  subst ht
  sorry

end perpendicular_line_through_point_l205_205665


namespace integer_values_satisfying_sqrt_inequality_l205_205014

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l205_205014


namespace ratio_area_shaded_triangle_l205_205409

variables (PQ PX QR QY YR : ℝ)
variables {A : ℝ}

def midpoint_QR (QR QY YR : ℝ) : Prop := QR = QY + YR ∧ QY = YR

def fraction_PQ_PX (PQ PX : ℝ) : Prop := PX = (3 / 4) * PQ

noncomputable def area_square (PQ : ℝ) : ℝ := PQ * PQ

noncomputable def area_triangle (base height : ℝ) : ℝ := (1 / 2) * base * height

theorem ratio_area_shaded_triangle
  (PQ PX QR QY YR : ℝ)
  (h_mid : midpoint_QR QR QY YR)
  (h_frac : fraction_PQ_PX PQ PX)
  (hQY_QR2 : QY = QR / 2)
  (hYR_QR2 : YR = QR / 2) :
  A = 5 / 16 :=
sorry

end ratio_area_shaded_triangle_l205_205409


namespace avg_children_with_kids_l205_205830

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205830


namespace binom_18_4_l205_205806

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l205_205806


namespace perpendicular_condition_l205_205294

-- Definitions of lines
def line1 (x y : ℝ) : Prop := x + y = 0
def line2 (x y : ℝ) (a : ℝ) : Prop := x - a * y = 0

-- Theorem: Prove that a = 1 is a necessary and sufficient condition for the lines
-- line1 and line2 to be perpendicular.
theorem perpendicular_condition (a : ℝ) : 
  (∀ x y : ℝ, line1 x y → line2 x y a) ↔ (a = 1) :=
sorry

end perpendicular_condition_l205_205294


namespace average_children_families_with_children_is_3_point_8_l205_205847

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205847


namespace solve_for_x_l205_205817

theorem solve_for_x (x : ℝ) : (1 / (x + 3) + 3 * x / (x + 3) - 5 / (x + 3) = 2) → x = 10 :=
by
  sorry

end solve_for_x_l205_205817


namespace part1_part2_l205_205610

noncomputable def probability_A_receives_one_red_envelope : ℚ :=
  sorry

theorem part1 (P_A1 : ℚ) (P_not_A1 : ℚ) (P_A2 : ℚ) (P_not_A2 : ℚ) :
  P_A1 = 1/3 ∧ P_not_A1 = 2/3 ∧ P_A2 = 1/3 ∧ P_not_A2 = 2/3 →
  probability_A_receives_one_red_envelope = 4/9 :=
sorry

noncomputable def probability_B_receives_at_least_10_yuan : ℚ :=
  sorry

theorem part2 (P_B1 : ℚ) (P_not_B1 : ℚ) (P_B2 : ℚ) (P_not_B2 : ℚ) (P_B3 : ℚ) (P_not_B3 : ℚ) :
  P_B1 = 1/3 ∧ P_not_B1 = 2/3 ∧ P_B2 = 1/3 ∧ P_not_B2 = 2/3 ∧ P_B3 = 1/3 ∧ P_not_B3 = 2/3 →
  probability_B_receives_at_least_10_yuan = 11/27 :=
sorry

end part1_part2_l205_205610


namespace numeric_puzzle_AB_eq_B_pow_V_l205_205896

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l205_205896


namespace base_n_representation_l205_205569

theorem base_n_representation 
  (n : ℕ) 
  (hn : n > 0)
  (a b c : ℕ) 
  (ha : 0 ≤ a ∧ a < n)
  (hb : 0 ≤ b ∧ b < n) 
  (hc : 0 ≤ c ∧ c < n) 
  (h_digits_sum : a + b + c = 24)
  (h_base_repr : 1998 = a * n^2 + b * n + c) 
  : n = 15 ∨ n = 22 ∨ n = 43 :=
sorry

end base_n_representation_l205_205569


namespace union_sets_l205_205953

noncomputable def A : Set ℝ := {x | (x + 1) * (x - 2) < 0}
noncomputable def B : Set ℝ := {x | 1 < x ∧ x ≤ 3}
noncomputable def C : Set ℝ := {x | -1 < x ∧ x ≤ 3}

theorem union_sets (A : Set ℝ) (B : Set ℝ) : (A ∪ B = C) := by
  sorry

end union_sets_l205_205953


namespace general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l205_205384

section ArithmeticSequence

-- Given conditions
def a1 : Int := 13
def a4 : Int := 7
def d : Int := (a4 - a1) / 3

-- General formula for a_n
def a_n (n : Int) : Int := a1 + (n - 1) * d

-- Sum of the first n terms S_n
def S_n (n : Int) : Int := n * (a1 + a_n n) / 2

-- Maximum value of S_n and corresponding term
def S_max : Int := 49
def n_max_S : Int := 7

-- Sum of the absolute values of the first n terms T_n
def T_n (n : Int) : Int :=
  if n ≤ 7 then n^2 + 12 * n
  else 98 - 12 * n - n^2

-- Statements to prove
theorem general_formula (n : Int) : a_n n = 15 - 2 * n := sorry

theorem sum_of_first_n_terms (n : Int) : S_n n = 14 * n - n^2 := sorry

theorem max_sum_of_S_n : (S_n n_max_S = S_max) := sorry

theorem sum_of_absolute_values (n : Int) : T_n n = 
  if n ≤ 7 then n^2 + 12 * n else 98 - 12 * n - n^2 := sorry

end ArithmeticSequence

end general_formula_sum_of_first_n_terms_max_sum_of_S_n_sum_of_absolute_values_l205_205384


namespace mean_of_set_l205_205371

theorem mean_of_set (m : ℝ) (h : m + 7 = 12) :
  (m + (m + 6) + (m + 7) + (m + 11) + (m + 18)) / 5 = 13.4 :=
by sorry

end mean_of_set_l205_205371


namespace mod_x_squared_l205_205925

theorem mod_x_squared :
  (∃ x : ℤ, 5 * x ≡ 9 [ZMOD 26] ∧ 4 * x ≡ 15 [ZMOD 26]) →
  ∃ y : ℤ, y ≡ 10 [ZMOD 26] :=
by
  intro h
  rcases h with ⟨x, h₁, h₂⟩
  exists x^2
  sorry

end mod_x_squared_l205_205925


namespace no_unique_sums_on_cube_l205_205232

open Nat

def vertices := Fin 8
def edges : Finset (vertices × vertices) := 
  { ((0 : vertices), (1 : vertices)), ((0 : vertices), (3 : vertices)), ((0 : vertices), (4 : vertices)),
    ((1 : vertices), (2 : vertices)), ((1 : vertices), (5 : vertices)), ((2 : vertices), (3 : vertices)),
    ((2 : vertices), (6 : vertices)), ((3 : vertices), (7 : vertices)), ((4 : vertices), (5 : vertices)),
    ((4 : vertices), (7 : vertices)), ((5 : vertices), (6 : vertices)), ((6 : vertices), (7 : vertices)) }

theorem no_unique_sums_on_cube :
  ¬ ∃ (f : vertices → ℕ), (∀ i j : vertices,
    (i, j) ∈ edges → 1 ≤ f i ∧ f i ≤ 8 ∧ 1 ≤ f j ∧ f j ≤ 8) ∧
  (∀ (i₁ j₁ i₂ j₂ : vertices), 
    (i₁, j₁) ∈ edges → (i₂, j₂) ∈ edges → (i₁ ≠ i₂ ∨ j₁ ≠ j₂) → (f i₁ + f j₁ ≠ f i₂ + f j₂)) :=
sorry

end no_unique_sums_on_cube_l205_205232


namespace find_the_number_l205_205162

theorem find_the_number (x : ℕ) : (220040 = (x + 445) * (2 * (x - 445)) + 40) → x = 555 :=
by
  intro h
  sorry

end find_the_number_l205_205162


namespace average_children_in_families_with_children_l205_205844

theorem average_children_in_families_with_children
  (n : ℕ)
  (c_avg : ℕ)
  (c_no_children : ℕ)
  (total_children : ℕ)
  (families_with_children : ℕ)
  (avg_children_families_with_children : ℚ) :
  n = 15 →
  c_avg = 3 →
  c_no_children = 3 →
  total_children = n * c_avg →
  families_with_children = n - c_no_children →
  avg_children_families_with_children = total_children / families_with_children →
  avg_children_families_with_children = 3.8 :=
by
  intros
  sorry

end average_children_in_families_with_children_l205_205844


namespace numeric_puzzle_AB_eq_B_pow_V_l205_205895

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l205_205895


namespace candy_bars_given_to_sister_first_time_l205_205171

theorem candy_bars_given_to_sister_first_time (x : ℕ) :
  (7 - x) + 30 - 4 * x = 22 → x = 3 :=
by
  sorry

end candy_bars_given_to_sister_first_time_l205_205171


namespace overtime_percentage_increase_l205_205768

-- Define the conditions.
def regular_rate : ℝ := 16
def regular_hours : ℕ := 40
def total_compensation : ℝ := 1116
def total_hours_worked : ℕ := 57
def overtime_hours : ℕ := total_hours_worked - regular_hours

-- Define the question and the answer as a proof problem.
theorem overtime_percentage_increase :
  let regular_earnings := regular_rate * regular_hours
  let overtime_earnings := total_compensation - regular_earnings
  let overtime_rate := overtime_earnings / overtime_hours
  overtime_rate > regular_rate →
  ((overtime_rate - regular_rate) / regular_rate) * 100 = 75 := 
by
  sorry

end overtime_percentage_increase_l205_205768


namespace common_ratio_common_difference_l205_205385

noncomputable def common_ratio_q {a b : ℕ → ℝ} (d : ℝ) (q : ℝ) :=
  (∀ n, b (n+1) = q * b n) ∧ (a 2 = -1) ∧ (a 1 < a 2) ∧ 
  (b 1 = (a 1)^2) ∧ (b 2 = (a 2)^2) ∧ (b 3 = (a 3)^2) ∧ 
  (∀ n, a (n+1) = a n + d)

theorem common_ratio
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  q = 3 - 2 * (2:ℝ).sqrt :=
sorry

theorem common_difference
  {a b : ℕ → ℝ} {d : ℝ}
  (h_arith : ∀ n, a (n + 1) = a n + d) (h_nonzero : d ≠ 0)
  (h_geom : ∀ n, b (n + 1) = (b 1^(1/2)) ^ (2 ^ n))
  (h_b1 : b 1 = (a 1) ^ 2) (h_b2 : b 2 = (a 2) ^ 2)
  (h_b3 : b 3 = (a 3) ^ 2) (h_a2 : a 2 = -1) (h_a1a2 : a 1 < a 2) :
  d = (2 : ℝ).sqrt :=
sorry

end common_ratio_common_difference_l205_205385


namespace problem_solution_l205_205207

noncomputable def f (x : ℝ) : ℝ := (Real.sin x) ^ 4 + (Real.cos x) ^ 4

theorem problem_solution (x1 x2 : ℝ) 
  (hx1 : x1 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (hx2 : x2 ∈ Set.Icc (-(Real.pi / 4)) (Real.pi / 4)) 
  (h : f x1 < f x2) : x1^2 > x2^2 := 
sorry

end problem_solution_l205_205207


namespace binom_18_4_eq_3060_l205_205801

theorem binom_18_4_eq_3060 : Nat.choose 18 4 = 3060 := by
  sorry

end binom_18_4_eq_3060_l205_205801


namespace angle_A_minimum_a_l205_205560

variable {α : Type} [LinearOrderedField α]

-- Part 1: Prove A = π / 3 given the specific equation in triangle ABC
theorem angle_A (a b c : α) (cos : α → α)
  (h : b^2 * c * cos c + c^2 * b * cos b = a * b^2 + a * c^2 - a^3) :
  ∃ A : α, A = π / 3 :=
sorry

-- Part 2: Prove the minimum value of a is 1 when b + c = 2
theorem minimum_a (a b c : α) (h : b + c = 2) :
  ∃ a : α, a = 1 :=
sorry

end angle_A_minimum_a_l205_205560


namespace rotation_matrix_150_degrees_l205_205370

theorem rotation_matrix_150_degrees : 
  let θ := Real.pi * (150 / 180)
  let c := Real.cos θ
  let s := Real.sin θ
  ∃ M : Matrix (Fin 2) (Fin 2) ℝ, M = ![
    ![c, -s],
    ![s, c]
  ] ∧ M = ![
    ![-(Real.sqrt 3 / 2), -(1 / 2)],
    ![(1 / 2), -(Real.sqrt 3 / 2)]
  ] :=
by
  let θ := Real.pi * (150 / 180)
  let c := Real.cos θ
  let s := Real.sin θ
  exists (![
    ![c, -s],
    ![s, c]
  ])
  split
  · -- Showing M is the rotation matrix form
    simp [c, s]
    sorry
  · -- showing that it matches the filled-in values
    simp
    sorry

end rotation_matrix_150_degrees_l205_205370


namespace find_a_add_b_l205_205087

theorem find_a_add_b (a b : ℝ) 
  (h1 : ∀ (x : ℝ), y = a + b / (x^2 + 1))
  (h2 : (y = 3) → (x = 1)) 
  (h3 : (y = 2) → (x = 0)) : a + b = 2 :=
by
  sorry

end find_a_add_b_l205_205087


namespace problem_part1_problem_part2_l205_205543

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  2 * (Real.sin (ω * x)) * (Real.cos (ω * x)) - 2 * Real.sqrt 3 * (Real.cos (ω * x))^2 + Real.sqrt 3

theorem problem_part1 (ω : ℝ) (k : ℤ) (x : ℝ) (hω : ω > 0) (hx : k * Real.pi - Real.pi / 12 ≤ x ∧ x ≤ k * Real.pi + 5 * Real.pi / 12) :
  monotone_on (f ω) (set.Icc (k * Real.pi - Real.pi / 12) (k * Real.pi + 5 * Real.pi / 12)) := sorry

theorem problem_part2 (A B C a b c : ℝ) (hC : 0 < C ∧ C < π / 2)
  (hfC : f 1 C = Real.sqrt 3) (hc : c = 3) (hB : Real.sin B = 2 * Real.sin A) :
  let area := 1 / 2 * a * b * Real.sin C in area = 3 * Real.sqrt 3 / 2 := sorry

end problem_part1_problem_part2_l205_205543


namespace inv_matrix_A_l205_205546

def A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ -2, 1 ],
     ![ (3/2 : ℚ), -1/2 ] ]

def A_inv : Matrix (Fin 2) (Fin 2) ℚ :=
  ![ ![ 1, 2 ],
     ![ 3, 4 ] ]

theorem inv_matrix_A : A⁻¹ = A_inv := by
  sorry

end inv_matrix_A_l205_205546


namespace theorem_1_valid_theorem_6_valid_l205_205751

theorem theorem_1_valid (a b : ℤ) (h1 : a % 7 = 0) (h2 : b % 7 = 0) : (a + b) % 7 = 0 :=
by sorry

theorem theorem_6_valid (a b : ℤ) (h : (a + b) % 7 ≠ 0) : a % 7 ≠ 0 ∨ b % 7 ≠ 0 :=
by sorry

end theorem_1_valid_theorem_6_valid_l205_205751


namespace num_distinct_values_for_sum_l205_205460

theorem num_distinct_values_for_sum (x y z : ℝ) 
  (h : (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0) :
  ∃ s : Finset ℝ, 
  (∀ x y z, (x^2 - 9)^2 + (y^2 - 4)^2 + (z^2 - 1)^2 = 0 → (x + y + z) ∈ s) ∧ 
  s.card = 7 :=
by sorry

end num_distinct_values_for_sum_l205_205460


namespace find_line_eq_l205_205273

theorem find_line_eq (m b k : ℝ) (h1 : (2, 7) ∈ ⋃ x, {(x, m * x + b)}) (h2 : ∀ k, abs ((k^2 + 4 * k + 3) - (m * k + b)) = 4) (h3 : b ≠ 0) : (m = 10) ∧ (b = -13) := by
  sorry

end find_line_eq_l205_205273


namespace polygon_sides_l205_205403

def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180
def sum_exterior_angles : ℝ := 360

theorem polygon_sides (n : ℕ) (h : 1/4 * sum_interior_angles n - sum_exterior_angles = 90) : n = 12 := 
by
  -- sorry to skip the proof
  sorry

end polygon_sides_l205_205403


namespace total_amount_shared_l205_205986

-- Define the variables
variables (a b c : ℕ)

-- Define the conditions
axiom condition1 : a = (1 / 3 : ℝ) * (b + c)
axiom condition2 : b = (2 / 7 : ℝ) * (a + c)
axiom condition3 : a = b + 15

-- The proof statement
theorem total_amount_shared : a + b + c = 540 :=
by
  -- We assume these axioms are declared and noncontradictory
  sorry

end total_amount_shared_l205_205986


namespace integer_values_satisfying_sqrt_inequality_l205_205013

theorem integer_values_satisfying_sqrt_inequality :
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  {x : ℤ | x ∈ ({n : ℤ | (S n)}.subtype)}.card = 3 :=
by
  let S := {x : ℝ | 4 < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < 5}
  let TIntS := {n : ℤ | n ∈ {x : ℤ | ∃ x' : ℝ, x = int.floor x' ∧ x' ∈ S}}
  have h : TIntS = {6, 7, 8} := sorry
  exact fintype.card_eq.mpr ⟨6, by sorry⟩ (by sorry)

end integer_values_satisfying_sqrt_inequality_l205_205013


namespace publishing_company_break_even_l205_205167

theorem publishing_company_break_even : 
  ∀ (F V P : ℝ) (x : ℝ), F = 35630 ∧ V = 11.50 ∧ P = 20.25 →
  (P * x = F + V * x) → x = 4074 :=
by
  intros F V P x h_eq h_rev
  sorry

end publishing_company_break_even_l205_205167


namespace any_positive_integer_can_be_expressed_l205_205580

theorem any_positive_integer_can_be_expressed 
  (N : ℕ) (hN : 0 < N) : 
  ∃ (p q u v : ℤ), N = p * q + u * v ∧ (u - v = 2 * (p - q)) := 
sorry

end any_positive_integer_can_be_expressed_l205_205580


namespace distance_midpoint_AB_to_y_axis_l205_205449

def parabola := { p : ℝ × ℝ // p.2^2 = 4 * p.1 }

variable (A B : parabola)
variable (x1 x2 : ℝ)
variable (y1 y2 : ℝ)

open scoped Classical

noncomputable def midpoint_x (x1 x2 : ℝ) : ℝ :=
  (x1 + x2) / 2

theorem distance_midpoint_AB_to_y_axis 
  (h1 : x1 + x2 = 3) 
  (hA : A.val = (x1, y1))
  (hB : B.val = (x2, y2)) : 
  midpoint_x x1 x2 = 3 / 2 := 
by
  sorry

end distance_midpoint_AB_to_y_axis_l205_205449


namespace rotation_matrix_150_degrees_l205_205360

theorem rotation_matrix_150_degrees :
  let θ := 150 * Real.pi / 180
  let cos150 := Real.cos θ
  let sin150 := Real.sin θ
  (cos150, sin150) = (-Real.sqrt 3 / 2, 1 / 2) →
  (Matrix.vecCons (Matrix.vecCons (-Real.sqrt 3 / 2) (-1 / 2) Matrix.nil) (Matrix.vecCons (1 / 2) (-Real.sqrt 3 / 2) Matrix.nil) 
    = Matrix.vecCons (Matrix.vecCons (-Real.sqrt 3 / 2) (-1 / 2) Matrix.nil) (Matrix.vecCons (1 / 2) (-Real.sqrt 3 / 2) Matrix.nil)) := by
  sorry

end rotation_matrix_150_degrees_l205_205360


namespace part1_infinite_n_part2_no_solutions_l205_205323

-- Definitions for part (1)
theorem part1_infinite_n (n : ℕ) (x y z t : ℕ) :
  (∃ n, x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

-- Definitions for part (2)
theorem part2_no_solutions (n k m x y z t : ℕ) :
  n = 4 ^ k * (8 * m + 7) → ¬(x ^ 2 + y ^ 2 + z ^ 2 + t ^ 2 - n * x * y * z * t - n = 0) :=
  sorry

end part1_infinite_n_part2_no_solutions_l205_205323


namespace expected_waiting_time_l205_205514

/-- Consider a 5-minute interval. There are 5 bites on the first rod 
and 1 bite on the second rod in this interval. Therefore, the total average 
number of bites on both rods during these 5 minutes is 6. The expected waiting 
time for the first bite is 50 seconds. -/
theorem expected_waiting_time
    (average_bites_first_rod : ℝ)
    (average_bites_second_rod : ℝ)
    (total_interval_minutes : ℝ)
    (expected_waiting_time_seconds : ℝ) :
    average_bites_first_rod = 5 ∧
    average_bites_second_rod = 1 ∧
    total_interval_minutes = 5 →
    expected_waiting_time_seconds = 50 :=
by
  sorry

end expected_waiting_time_l205_205514


namespace total_cost_is_46_8_l205_205035

def price_pork : ℝ := 6
def price_chicken : ℝ := price_pork - 2
def price_beef : ℝ := price_chicken + 4
def price_lamb : ℝ := price_pork + 3

def quantity_chicken : ℝ := 3.5
def quantity_pork : ℝ := 1.2
def quantity_beef : ℝ := 2.3
def quantity_lamb : ℝ := 0.8

def total_cost : ℝ :=
    (quantity_chicken * price_chicken) +
    (quantity_pork * price_pork) +
    (quantity_beef * price_beef) +
    (quantity_lamb * price_lamb)

theorem total_cost_is_46_8 : total_cost = 46.8 :=
by
  sorry

end total_cost_is_46_8_l205_205035


namespace first_two_digits_of_1666_l205_205101

/-- Lean 4 statement for the given problem -/
theorem first_two_digits_of_1666 (y k : ℕ) (H_nonzero_k : k ≠ 0) (H_nonzero_y : y ≠ 0) (H_y_six : y = 6) :
  (1666 / 100) = 16 := by
  sorry

end first_two_digits_of_1666_l205_205101


namespace math_problem_l205_205395

variables (a b c d m : ℝ)

theorem math_problem 
  (h1 : a = -b)            -- condition 1: a and b are opposite numbers
  (h2 : c * d = 1)         -- condition 2: c and d are reciprocal numbers
  (h3 : |m| = 1) :         -- condition 3: absolute value of m is 1
  (a + b) * c * d - 2009 * m = -2009 ∨ (a + b) * c * d - 2009 * m = 2009 :=
sorry

end math_problem_l205_205395


namespace parabola_focus_l205_205445

theorem parabola_focus (x y : ℝ) : (y^2 = -8 * x) → (x, y) = (-2, 0) :=
by
  sorry

end parabola_focus_l205_205445


namespace soda_preference_l205_205222

theorem soda_preference (total_surveyed : ℕ) (angle_soda_sector : ℕ) (h_total_surveyed : total_surveyed = 540) (h_angle_soda_sector : angle_soda_sector = 270) :
  let fraction_soda := angle_soda_sector / 360
  let people_soda := fraction_soda * total_surveyed
  people_soda = 405 :=
by
  sorry

end soda_preference_l205_205222


namespace isosceles_triangle_perimeter_l205_205602

theorem isosceles_triangle_perimeter (a b : ℕ) (h1 : a = 9) (h2 : b = 4) (h3 : b < a + a) : a + a + b = 22 := by
  sorry

end isosceles_triangle_perimeter_l205_205602


namespace carson_giant_slide_rides_l205_205637

theorem carson_giant_slide_rides :
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  -- Convert hours to minutes
  let total_minutes := total_hours * 60
  -- Calculate total wait time for roller coaster
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  -- Calculate total wait time for tilt-a-whirl
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  -- Calculate total wait time for roller coaster and tilt-a-whirl
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  -- Calculate remaining time
  let remaining_time := total_minutes - total_wait
  -- Calculate how many times Carson can ride the giant slide
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  giant_slide_rides = 4 := by
  let total_hours := 4
  let roller_coaster_wait_time := 30
  let roller_coaster_rides := 4
  let tilt_a_whirl_wait_time := 60
  let tilt_a_whirl_rides := 1
  let giant_slide_wait_time := 15
  let total_minutes := total_hours * 60
  let roller_coaster_total_wait := roller_coaster_wait_time * roller_coaster_rides
  let tilt_a_whirl_total_wait := tilt_a_whirl_wait_time * tilt_a_whirl_rides
  let total_wait := roller_coaster_total_wait + tilt_a_whirl_total_wait
  let remaining_time := total_minutes - total_wait
  let giant_slide_rides := remaining_time / giant_slide_wait_time
  show giant_slide_rides = 4
  sorry

end carson_giant_slide_rides_l205_205637


namespace cricket_player_average_l205_205987

theorem cricket_player_average
  (A : ℕ)
  (h1 : 8 * A + 96 = 9 * (A + 8)) :
  A = 24 :=
by
  sorry

end cricket_player_average_l205_205987


namespace mail_handling_in_six_months_l205_205454

theorem mail_handling_in_six_months (daily_letters daily_packages days_per_month months : ℕ) :
  daily_letters = 60 →
  daily_packages = 20 →
  days_per_month = 30 →
  months = 6 →
  (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  -- Skipping the proof
  sorry

end mail_handling_in_six_months_l205_205454


namespace quadratic_has_two_distinct_real_roots_l205_205378

-- Definitions of the conditions
def a : ℝ := 1
def b (k : ℝ) : ℝ := -3 * k
def c : ℝ := -2

-- Definition of the discriminant function
def discriminant (k : ℝ) : ℝ := (b k) ^ 2 - 4 * a * c

-- Logical statement to be proved
theorem quadratic_has_two_distinct_real_roots (k : ℝ) : discriminant k > 0 :=
by
  unfold discriminant
  unfold b a c
  simp
  sorry

end quadratic_has_two_distinct_real_roots_l205_205378


namespace exists_non_deg_triangle_in_sets_l205_205429

-- Definitions used directly from conditions in a)
def non_deg_triangle (a b c : ℕ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Main theorem statement
theorem exists_non_deg_triangle_in_sets (S : Fin 100 → Set ℕ) (h_disjoint : ∀ i j : Fin 100, i ≠ j → Disjoint (S i) (S j))
  (h_union : (⋃ i, S i) = {x | 1 ≤ x ∧ x ≤ 400}) :
  ∃ i : Fin 100, ∃ a b c : ℕ, a ∈ S i ∧ b ∈ S i ∧ c ∈ S i ∧ non_deg_triangle a b c := sorry

end exists_non_deg_triangle_in_sets_l205_205429


namespace day_crew_fraction_l205_205287

theorem day_crew_fraction (D W : ℝ) (h1 : D > 0) (h2 : W > 0) :
  (D * W / (D * W + (3 / 4 * D * 1 / 2 * W)) = 8 / 11) :=
by
  sorry

end day_crew_fraction_l205_205287


namespace tara_road_trip_cost_l205_205260

theorem tara_road_trip_cost :
  let tank_capacity := 12
  let price1 := 3
  let price2 := 3.50
  let price3 := 4
  let price4 := 4.50
  (price1 * tank_capacity) + (price2 * tank_capacity) + (price3 * tank_capacity) + (price4 * tank_capacity) = 180 :=
by
  sorry

end tara_road_trip_cost_l205_205260


namespace minjun_current_height_l205_205248

variable (initial_height : ℝ) (growth_last_year : ℝ) (growth_this_year : ℝ)

theorem minjun_current_height
  (h_initial : initial_height = 1.1)
  (h_growth_last_year : growth_last_year = 0.2)
  (h_growth_this_year : growth_this_year = 0.1) :
  initial_height + growth_last_year + growth_this_year = 1.4 :=
by
  sorry

end minjun_current_height_l205_205248


namespace sandy_walks_before_meet_l205_205434

/-
Sandy leaves her home and walks toward Ed's house.
Two hours later, Ed leaves his home and walks toward Sandy's house.
The distance between their homes is 52 kilometers.
Sandy's walking speed is 6 km/h.
Ed's walking speed is 4 km/h.
Prove that Sandy will walk 36 kilometers before she meets Ed.
-/

theorem sandy_walks_before_meet
    (distance_between_homes : ℕ)
    (sandy_speed ed_speed : ℕ)
    (sandy_start_time ed_start_time : ℕ)
    (time_to_meet : ℕ) :
  distance_between_homes = 52 →
  sandy_speed = 6 →
  ed_speed = 4 →
  sandy_start_time = 2 →
  ed_start_time = 0 →
  time_to_meet = 4 →
  (sandy_start_time * sandy_speed + time_to_meet * sandy_speed) = 36 := 
by
  sorry

end sandy_walks_before_meet_l205_205434


namespace rate_of_current_l205_205000

theorem rate_of_current (c : ℝ) : 
  (∀ t : ℝ, t = 0.4 → ∀ d : ℝ, d = 9.6 → ∀ b : ℝ, b = 20 →
  d = (b + c) * t → c = 4) :=
sorry

end rate_of_current_l205_205000


namespace presidency_meeting_ways_l205_205990

def numWaysToChooseRepresentatives : ℕ :=
  4 * (Nat.choose 5 3) * (5 * 5 * 5)

theorem presidency_meeting_ways : numWaysToChooseRepresentatives = 5000 :=
by
  unfold numWaysToChooseRepresentatives
  rw [Nat.choose_eq_factorial_div_factorial (le_refl 2)]
  sorry

end presidency_meeting_ways_l205_205990


namespace algebra_ineq_l205_205670

theorem algebra_ineq (a b c : ℝ) (a_pos : 0 < a) (b_pos : 0 < b) (c_pos : 0 < c) (h : a * b + b * c + c * a = 1) : a + b + c ≥ 2 := 
by sorry

end algebra_ineq_l205_205670


namespace ship_length_in_steps_l205_205180

theorem ship_length_in_steps (E S L : ℝ) (H1 : L + 300 * S = 300 * E) (H2 : L - 60 * S = 60 * E) :
  L = 100 * E :=
by sorry

end ship_length_in_steps_l205_205180


namespace num_two_digit_math_representation_l205_205944

-- Define the problem space
def unique_digits (n : ℕ) : Prop := 
  n >= 1 ∧ n <= 9

-- Representation of the characters' assignment
def representation (x y z w : ℕ) : Prop :=
  unique_digits x ∧ unique_digits y ∧ unique_digits z ∧ unique_digits w ∧
  x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w ∧ 
  x = z ∧ 3 * (10 * y + z) = 10 * w + x

-- The main theorem to prove
theorem num_two_digit_math_representation : 
  ∃ x y z w, representation x y z w :=
sorry

end num_two_digit_math_representation_l205_205944


namespace sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l205_205315

-- Problem 1
theorem sqrt_18_mul_sqrt_6 : (Real.sqrt 18 * Real.sqrt 6 = 6 * Real.sqrt 3) :=
sorry

-- Problem 2
theorem sqrt_8_sub_sqrt_2_add_2_sqrt_half : (Real.sqrt 8 - Real.sqrt 2 + 2 * Real.sqrt (1 / 2) = 3 * Real.sqrt 2) :=
sorry

-- Problem 3
theorem sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3 : (Real.sqrt 12 * (Real.sqrt 9 / 3) / (Real.sqrt 3 / 3) = 6) :=
sorry

-- Problem 4
theorem sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5 : ((Real.sqrt 7 + Real.sqrt 5) * (Real.sqrt 7 - Real.sqrt 5) = 2) :=
sorry

end sqrt_18_mul_sqrt_6_sqrt_8_sub_sqrt_2_add_2_sqrt_half_sqrt_12_mul_sqrt_9_div_3_div_sqrt_3_div_3_sqrt_7_add_sqrt_5_mul_sqrt_7_sub_sqrt_5_l205_205315


namespace least_students_with_brown_eyes_and_lunch_box_l205_205406

variable (U : Finset ℕ) (B L : Finset ℕ)
variables (hU : U.card = 25) (hB : B.card = 15) (hL : L.card = 18)

theorem least_students_with_brown_eyes_and_lunch_box : 
  (B ∩ L).card ≥ 8 := by
  sorry

end least_students_with_brown_eyes_and_lunch_box_l205_205406


namespace area_ratio_equilateral_triangle_extension_l205_205415

variable (s : ℝ)

theorem area_ratio_equilateral_triangle_extension :
  (let A := (0, 0)
   let B := (s, 0)
   let C := (s / 2, s * (Real.sqrt 3 / 2))
   let A' := (0, -4 * s * (Real.sqrt 3 / 2))
   let B' := (3 * s, 0)
   let C' := (s / 2, s * (Real.sqrt 3 / 2) + 3 * s * (Real.sqrt 3 / 2))
   let area_ABC := (Real.sqrt 3 / 4) * s^2
   let area_A'B'C' := (Real.sqrt 3 / 4) * 60 * s^2
   area_A'B'C' / area_ABC = 60) :=
sorry

end area_ratio_equilateral_triangle_extension_l205_205415


namespace animal_arrangement_count_l205_205718

theorem animal_arrangement_count :
  (3! * 5! * 1! * 6! = 518400) := by
  sorry

end animal_arrangement_count_l205_205718


namespace problem_solution_l205_205030

-- Definitions from the problem conditions
def sum_of_marked_angles (a : ℝ) : Prop := a = 900

def polygon_interior_angles (a b : ℝ) : Prop := a = (b - 2) * 180

def exponential_relationship (b c : ℝ) : Prop := 8^b = c^21

def logarithmic_relationship (c d : ℝ) : Prop := c = Real.logb d 81

-- Prove the questions equal the answers given conditions
theorem problem_solution (a b c d : ℝ) (h1 : sum_of_marked_angles a) 
    (h2 : polygon_interior_angles a b) 
    (h3 : exponential_relationship b c)
    (h4 : logarithmic_relationship c d) : 
    a = 900 ∧ b = 7 ∧ c = 2 ∧ d = 9 := 
begin
    sorry
end

end problem_solution_l205_205030


namespace solve_for_x_l205_205714

theorem solve_for_x (x : ℝ) (h : (x + 6) / (x - 3) = 4) : x = 6 :=
by
  sorry

end solve_for_x_l205_205714


namespace number_of_space_diagonals_l205_205619

-- Define the conditions of the polyhedron
def vertices : Nat := 30
def edges : Nat := 72
def faces := [
  (30, 3),  -- 30 triangular faces
  (10, 4),  -- 10 quadrilateral faces
  (4, 5)    -- 4 pentagonal faces
]

-- Calculate the total number of line segments (combinations of 2 vertices)
def total_line_segments := Nat.choose vertices 2

-- Calculate the number of face diagonals
def face_diagonals : Nat :=
  faces.foldl (fun acc (count, sides) =>
    acc + count * (sides * (sides - 3) / 2)
  ) 0

-- The final statement to prove the number of space diagonals
theorem number_of_space_diagonals : 
  total_line_segments - edges - face_diagonals = 323 := by
  -- Calculate the total line segments
  let total_line_segments := 30 * 29 / 2
  -- Calculate the face diagonals based on type of faces
  let face_diagonals := 0 + 20 + 20
  -- So, the total number of space diagonals is
  show total_line_segments - 72 - face_diagonals = 323
  sorry

end number_of_space_diagonals_l205_205619


namespace triangular_number_30_sum_of_first_30_triangular_numbers_l205_205634

theorem triangular_number_30 
  (T : ℕ → ℕ)
  (hT : ∀ n : ℕ, T n = n * (n + 1) / 2) : 
  T 30 = 465 :=
by
  -- Skipping proof with sorry
  sorry

theorem sum_of_first_30_triangular_numbers 
  (S : ℕ → ℕ)
  (hS : ∀ n : ℕ, S n = n * (n + 1) * (n + 2) / 6) : 
  S 30 = 4960 :=
by
  -- Skipping proof with sorry
  sorry

end triangular_number_30_sum_of_first_30_triangular_numbers_l205_205634


namespace count_integer_values_satisfying_condition_l205_205009

theorem count_integer_values_satisfying_condition :
  ∃ (n : ℕ), n = 3 ∧ ∀ (x : ℤ), (4 : ℝ) < real.sqrt (3 * x) ∧ real.sqrt (3 * x) < (5 : ℝ) → x ∈ {6, 7, 8} := 
by sorry

end count_integer_values_satisfying_condition_l205_205009


namespace numerical_puzzle_solution_l205_205901

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l205_205901


namespace find_t_l205_205419

variable {x y z w t : ℝ}

theorem find_t (hx : x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w)
               (hpos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w)
               (hxy : x + 1/y = t)
               (hyz : y + 1/z = t)
               (hzw : z + 1/w = t)
               (hwx : w + 1/x = t) : 
               t = Real.sqrt 2 :=
by
  sorry

end find_t_l205_205419


namespace winning_candidate_votes_percentage_l205_205555

theorem winning_candidate_votes_percentage (P : ℝ) 
    (majority : P/100 * 6000 - (6000 - P/100 * 6000) = 1200) : 
    P = 60 := 
by 
  sorry

end winning_candidate_votes_percentage_l205_205555


namespace problems_per_worksheet_l205_205496

theorem problems_per_worksheet (P : ℕ) (graded : ℕ) (remaining : ℕ) (total_worksheets : ℕ) (total_problems_remaining : ℕ) :
    graded = 5 →
    total_worksheets = 9 →
    total_problems_remaining = 16 →
    remaining = total_worksheets - graded →
    4 * P = total_problems_remaining →
    P = 4 :=
by
  intros h_graded h_worksheets h_problems h_remaining h_equation
  sorry

end problems_per_worksheet_l205_205496


namespace probability_palindrome_divisible_by_7_l205_205302

def is_palindrome (n : ℕ) : Prop :=
  let s := n.to_string in
  s = s.reverse

def is_divisible_by_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem probability_palindrome_divisible_by_7 : 
  (∃ (a : ℕ) (b : ℕ), 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ is_palindrome (1001 * a + 110 * b) ∧ is_divisible_by_7 (1001 * a + 110 * b)) →
  (∃ (a' b' : ℕ), 1 ≤ a' ∧ a' ≤ 9 ∧ 0 ≤ b' ∧ b' ≤ 9) →
  (18 : ℚ) / 90 = 1 / 5 :=
sorry

end probability_palindrome_divisible_by_7_l205_205302


namespace numerical_puzzle_solution_l205_205903

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l205_205903


namespace probability_same_flavor_l205_205044

theorem probability_same_flavor (num_flavors : ℕ) (num_bags : ℕ) (h1 : num_flavors = 4) (h2 : num_bags = 2) :
  let total_outcomes := num_flavors ^ num_bags
  let favorable_outcomes := num_flavors
  (favorable_outcomes : ℚ) / (total_outcomes : ℚ) = 1 / 4 :=
by
  sorry

end probability_same_flavor_l205_205044


namespace factorial_sqrt_square_l205_205789

theorem factorial_sqrt_square (n : ℕ) : (nat.succ 4)! * 4! = 2880 := by 
  sorry

end factorial_sqrt_square_l205_205789


namespace wade_total_spent_l205_205973

def sandwich_cost : ℕ := 6
def drink_cost : ℕ := 4
def num_sandwiches : ℕ := 3
def num_drinks : ℕ := 2

def total_cost : ℕ :=
  (num_sandwiches * sandwich_cost) + (num_drinks * drink_cost)

theorem wade_total_spent : total_cost = 26 := by
  sorry

end wade_total_spent_l205_205973


namespace problem1_l205_205154

theorem problem1 : sqrt 18 - sqrt 8 - sqrt 2 = 0 := 
by 
  have h₁ : sqrt 18 = 3 * sqrt 2 := sorry
  have h₂ : sqrt 8 = 2 * sqrt 2 := sorry
  rw [h₁, h₂]
  sorry

end problem1_l205_205154


namespace largest_prime_factor_of_1729_is_19_l205_205522

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def largest_prime_factor (n : ℕ) (p : ℕ) := is_prime p ∧ p ∣ n ∧ ∀ q : ℕ, is_prime q ∧ q ∣ n → q ≤ p

theorem largest_prime_factor_of_1729_is_19 : largest_prime_factor 1729 19 :=
by
  sorry

end largest_prime_factor_of_1729_is_19_l205_205522


namespace avg_children_in_families_with_children_l205_205833

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205833


namespace average_children_l205_205869

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205869


namespace find_omega_range_l205_205387

noncomputable def function_conditions (ω : ℝ) : Prop :=
  ∀ x, (x ∈ set.Ioo (π / 6) (π / 4)) → (f x = sin (ω * x + π / 4)) ∧
    (∃ a b, set.Ioo (π / 6) (π / 4) ⊆ set.set_of (λ y, y = a ∨ y = b) ∧
           ∀ z, (f z = 0 ∧ z ≠ a ∧ z ≠ b) ∨ (f z = 1 / 2 ∧ (z = a ∨ z = b)))

noncomputable def correct_range : set ℝ :=
  set.Ioo 25 (51 / 2) ∪ set.Icc (69 / 2) 35

theorem find_omega_range :
  ∀ (ω : ℝ), (ω > 0) → function_conditions ω → ω ∈ correct_range :=
sorry

end find_omega_range_l205_205387


namespace real_roots_of_system_l205_205647

theorem real_roots_of_system :
  { (x, y) : ℝ × ℝ | (x + y)^4 = 6 * x^2 * y^2 - 215 ∧ x * y * (x^2 + y^2) = -78 } =
  { (3, -2), (-2, 3), (-3, 2), (2, -3) } :=
by 
  sorry

end real_roots_of_system_l205_205647


namespace douglas_votes_in_county_X_l205_205556

theorem douglas_votes_in_county_X (V : ℝ) :
  (0.64 * (2 * V + V) - 0.4000000000000002 * V) / (2 * V) * 100 = 76 := by
sorry

end douglas_votes_in_county_X_l205_205556


namespace sequence_recurrence_l205_205128

theorem sequence_recurrence (v : ℕ → ℝ) (h_rec : ∀ n, v (n + 2) = 3 * v (n + 1) + 2 * v n) 
    (h_v3 : v 3 = 8) (h_v6 : v 6 = 245) : v 5 = 70 :=
sorry

end sequence_recurrence_l205_205128


namespace equation_of_parallel_line_l205_205187

theorem equation_of_parallel_line {x y : ℝ} :
  (∃ b : ℝ, ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 + b = 0)) ↔ 
  (∃ b : ℝ, b = -2 ∧ ∀ (P : ℝ × ℝ), P = (1, 0) → (2 * P.1 + P.2 - 2 = 0)) := 
by 
  sorry

end equation_of_parallel_line_l205_205187


namespace number_of_10_people_rows_l205_205334

theorem number_of_10_people_rows (x r : ℕ) (h1 : r = 54) (h2 : ∀ i : ℕ, i * 9 + x * 10 = 54) : x = 0 :=
by
  sorry

end number_of_10_people_rows_l205_205334


namespace sarah_can_make_max_servings_l205_205627

-- Definitions based on the conditions of the problem
def servings_from_bananas (bananas : ℕ) : ℕ := (bananas * 8) / 3
def servings_from_strawberries (cups_strawberries : ℕ) : ℕ := (cups_strawberries * 8) / 2
def servings_from_yogurt (cups_yogurt : ℕ) : ℕ := cups_yogurt * 8
def servings_from_milk (cups_milk : ℕ) : ℕ := (cups_milk * 8) / 4

-- Given Sarah's stock
def sarahs_bananas : ℕ := 10
def sarahs_strawberries : ℕ := 5
def sarahs_yogurt : ℕ := 3
def sarahs_milk : ℕ := 10

-- The maximum servings calculation
def max_servings : ℕ := 
  min (servings_from_bananas sarahs_bananas)
      (min (servings_from_strawberries sarahs_strawberries)
           (min (servings_from_yogurt sarahs_yogurt)
                (servings_from_milk sarahs_milk)))

-- The theorem to be proved
theorem sarah_can_make_max_servings : max_servings = 20 :=
by
  sorry

end sarah_can_make_max_servings_l205_205627


namespace average_children_in_families_with_children_l205_205857

theorem average_children_in_families_with_children :
  let total_families := 15
  let average_children_per_family := 3
  let childless_families := 3
  let total_children := total_families * average_children_per_family
  let families_with_children := total_families - childless_families
  let average_children_per_family_with_children := total_children / families_with_children
  average_children_per_family_with_children = 3.8 /- here 3.8 represents the decimal number 3.8 -/ := 
by
  sorry

end average_children_in_families_with_children_l205_205857


namespace average_children_in_families_with_children_l205_205885

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205885


namespace shara_age_l205_205412

-- Definitions derived from conditions
variables (S : ℕ) (J : ℕ)

-- Jaymee's age is twice Shara's age plus 2
def jaymee_age_relation : Prop := J = 2 * S + 2

-- Jaymee's age is given as 22
def jaymee_age : Prop := J = 22

-- The proof problem to prove Shara's age equals 10
theorem shara_age (h1 : jaymee_age_relation S J) (h2 : jaymee_age J) : S = 10 :=
by 
  sorry

end shara_age_l205_205412


namespace intersection_of_A_and_B_l205_205390

-- Define the sets A and B
def setA : Set ℝ := { x | -1 < x ∧ x ≤ 4 }
def setB : Set ℝ := { x | 2 < x ∧ x ≤ 5 }

-- The intersection of sets A and B
def intersectAB : Set ℝ := { x | 2 < x ∧ x ≤ 4 }

-- The theorem statement to be proved
theorem intersection_of_A_and_B : ∀ x, x ∈ setA ∩ setB ↔ x ∈ intersectAB := by
  sorry

end intersection_of_A_and_B_l205_205390


namespace partition_count_l205_205568

noncomputable def count_partition (n : ℕ) : ℕ :=
  -- Function that counts the number of ways to partition n as per the given conditions
  n

theorem partition_count (n : ℕ) (h : n > 0) :
  count_partition n = n :=
sorry

end partition_count_l205_205568


namespace parallel_lines_chords_distance_l205_205381

theorem parallel_lines_chords_distance
  (r d : ℝ)
  (h1 : ∀ (P Q : ℝ), P = Q + d / 2 → Q = P - d / 2)
  (h2 : ∀ (A B : ℝ), A = B + 3 * d / 2 → B = A - 3 * d / 2)
  (chords : ∀ (l1 l2 l3 l4 : ℝ), (l1 = 40 ∧ l2 = 40 ∧ l3 = 36 ∧ l4 = 36)) :
  d = 1.46 :=
sorry

end parallel_lines_chords_distance_l205_205381


namespace reflection_y_axis_matrix_correct_l205_205339

def reflect_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]]

theorem reflection_y_axis_matrix_correct :
  reflect_y_axis_matrix = ![![(-1 : ℝ), (0 : ℝ)], ![(0 : ℝ), (1 : ℝ)]] :=
by
  sorry

end reflection_y_axis_matrix_correct_l205_205339


namespace exists_large_absolute_value_solutions_l205_205051

theorem exists_large_absolute_value_solutions : 
  ∃ (x1 x2 y1 y2 y3 y4 : ℤ), 
    x1 + x2 = y1 + y2 + y3 + y4 ∧ 
    x1^2 + x2^2 = y1^2 + y2^2 + y3^2 + y4^2 ∧ 
    x1^3 + x2^3 = y1^3 + y2^3 + y3^3 + y4^3 ∧ 
    abs x1 > 2020 ∧ abs x2 > 2020 ∧ abs y1 > 2020 ∧ abs y2 > 2020 ∧ abs y3 > 2020 ∧ abs y4 > 2020 :=
  by
  sorry

end exists_large_absolute_value_solutions_l205_205051


namespace fraction_sum_identity_l205_205267

variable (a b c : ℝ)

theorem fraction_sum_identity (h1 : a + b + c = 0) (h2 : a / b + b / c + c / a = 100) : 
  b / a + c / b + a / c = -103 :=
by {
  -- Proof goes here
  sorry
}

end fraction_sum_identity_l205_205267


namespace min_tangent_length_is_4_l205_205934

noncomputable def min_tangent_length (a b : ℝ) :=
  let pc := Real.sqrt ((a + 1)^2 + (b - 2)^2)
  let r := Real.sqrt 2
  Real.sqrt (pc^2 - r^2)

theorem min_tangent_length_is_4 :
  ∀ b : ℝ, let a := b + 3 in
  2 * (b + 1)^2 + 16 = (min_tangent_length (b + 3) b)^2 → 
  (min_tangent_length (b + 3) b) = 4 :=
by
  intros b a ha
  sorry

end min_tangent_length_is_4_l205_205934


namespace expected_occur_two_consecutive_zeros_in_33_bits_string_l205_205535

theorem expected_occur_two_consecutive_zeros_in_33_bits_string :
  ∀ (s : string), s.length = 33 →
  (∀ i j, (string.get⟨i, h1⟩ = '0' ∧ string.get⟨i + 1, h2⟩ = '0')) →
  expected_value (occur_two_consecutive_zeros s) = 8 :=
by
  sorry

end expected_occur_two_consecutive_zeros_in_33_bits_string_l205_205535


namespace rationalize_denominator_proof_l205_205582

def rationalize_denominator (cbrt : ℝ → ℝ) (a : ℝ) :=
  cbrt a = a^(1/3)

theorem rationalize_denominator_proof : 
  (rationalize_denominator (λ x, x ^ (1/3)) 27) →
  (rationalize_denominator (λ x, x ^ (1/3)) 9) →
  (1 / (3 ^ (1 / 3) + 3) = 9 ^ (1 / 3) / (3 + 9 * 3 ^ (1 / 3))) :=
by
  sorry

end rationalize_denominator_proof_l205_205582


namespace product_correct_l205_205652

/-- Define the number and the digit we're interested in -/
def num : ℕ := 564823
def digit : ℕ := 4

/-- Define a function to calculate the local value of the digit 4 in the number 564823 -/
def local_value (n : ℕ) (d : ℕ) := if d = 4 then 40000 else 0

/-- Define a function to calculate the absolute value, although it is trivial for natural numbers -/
def abs_value (d : ℕ) := d

/-- Define the product of local value and absolute value of 4 in 564823 -/
def product := local_value num digit * abs_value digit

/-- Theorem stating that the product is as specified in the problem -/
theorem product_correct : product = 160000 :=
by
  sorry

end product_correct_l205_205652


namespace fg_equals_seven_l205_205217

def g (x : ℤ) : ℤ := x * x
def f (x : ℤ) : ℤ := 2 * x - 1

theorem fg_equals_seven : f (g 2) = 7 := by
  sorry

end fg_equals_seven_l205_205217


namespace average_children_in_families_with_children_l205_205863

theorem average_children_in_families_with_children :
  (15 * 3 = 45) ∧ (15 - 3 = 12) →
  (45 / (15 - 3) = 3.75) →
  (Float.round 3.75) = 3.8 :=
by
  intros h1 h2
  sorry

end average_children_in_families_with_children_l205_205863


namespace not_even_nor_odd_l205_205119

def f (x : ℝ) : ℝ := x^2

theorem not_even_nor_odd (x : ℝ) (h₁ : -1 < x) (h₂ : x ≤ 1) : ¬(∀ y, f y = f (-y)) ∧ ¬(∀ y, f y = -f (-y)) :=
by
  sorry

end not_even_nor_odd_l205_205119


namespace Clever_not_Green_l205_205564

variables {Lizard : Type}
variables [DecidableEq Lizard] (Clever Green CanJump CanSwim : Lizard → Prop)

theorem Clever_not_Green (h1 : ∀ x, Clever x → CanJump x)
                        (h2 : ∀ x, Green x → ¬ CanSwim x)
                        (h3 : ∀ x, ¬ CanSwim x → ¬ CanJump x) :
  ∀ x, Clever x → ¬ Green x :=
by
  intro x hClever hGreen
  apply h3 x
  apply h2 x hGreen
  exact h1 x hClever

end Clever_not_Green_l205_205564


namespace xyz_mod_3_l205_205325

theorem xyz_mod_3 {x y z : ℕ} (hx : x = 3) (hy : y = 3) (hz : z = 2) : 
  (x^2 + y^2 + z^2) % 3 = 1 := by
  sorry

end xyz_mod_3_l205_205325


namespace rotation_matrix_150_degrees_l205_205349

theorem rotation_matrix_150_degrees :
  ∃ (R : Matrix (Fin 2) (Fin 2) ℝ),
    R = Matrix.ofFn
      (λ i j, match (i, j) with
              | (0, 0) => -(Real.sqrt 3) / 2
              | (0, 1) => -1 / 2
              | (1, 0) => 1 / 2
              | (1, 1) => -(Real.sqrt 3) / 2
              | _ => 0) :=
begin
  sorry
end

end rotation_matrix_150_degrees_l205_205349


namespace arithmetic_geometric_condition_l205_205049

-- Define the arithmetic sequence
noncomputable def arithmetic_seq (a₁ d : ℕ) (n : ℕ) : ℕ := a₁ + (n-1) * d

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def sum_arith_seq (a₁ d n : ℕ) : ℕ := n * a₁ + (n * (n-1) / 2) * d

-- Given conditions and required proofs
theorem arithmetic_geometric_condition {d a₁ : ℕ} (h : d ≠ 0) (S₃ : sum_arith_seq a₁ d 3 = 9)
  (geometric_seq : (arithmetic_seq a₁ d 5)^2 = (arithmetic_seq a₁ d 3) * (arithmetic_seq a₁ d 8)) :
  d = 1 ∧ ∀ n, sum_arith_seq 2 1 n = (n^2 + 3 * n) / 2 :=
by
  sorry

end arithmetic_geometric_condition_l205_205049


namespace opposite_of_2023_l205_205600

def opposite (n : Int) : Int := -n

theorem opposite_of_2023 : opposite 2023 = -2023 := by
  sorry

end opposite_of_2023_l205_205600


namespace part1_part2_l205_205206

noncomputable def f (x a : ℝ) : ℝ := |x - 2 * a| + |x - 3 * a|

theorem part1 (a : ℝ) (h_min : ∃ x, f x a = 2) : |a| = 2 := by
  sorry

theorem part2 (m : ℝ)
  (h_condition : ∀ x : ℝ, ∃ a : ℝ, -2 ≤ a ∧ a ≤ 2 ∧ (m^2 - |m| - f x a) < 0) :
  -1 < m ∧ m < 2 := by
  sorry

end part1_part2_l205_205206


namespace rotation_matrix_150_l205_205350

noncomputable def cos_150 : ℝ := -real.cos (real.pi / 6)
noncomputable def sin_150 : ℝ := real.sin (real.pi / 6)

theorem rotation_matrix_150 : 
  ∀ θ : ℝ, θ = 5 * real.pi / 6 → 
  (matrix (fin 2) (fin 2) ℝ) := 
begin
  assume θ hθ,
  rw hθ,
  exact matrix.cons_vec_cons
    (matrix.cons_vec_cons cos_150 (-sin_150))
    (matrix.cons_vec_cons sin_150 cos_150),
  sorry
end

end rotation_matrix_150_l205_205350


namespace area_of_small_parallelograms_l205_205601

theorem area_of_small_parallelograms (m n : ℕ) (h_m : 0 < m) (h_n : 0 < n) :
  (1 : ℝ) / (m * n : ℝ) = 1 / (m * n) :=
by
  sorry

end area_of_small_parallelograms_l205_205601


namespace numeric_puzzle_AB_eq_B_pow_V_l205_205898

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l205_205898


namespace interest_earned_l205_205609

noncomputable def simple_interest (P R T : ℚ) : ℚ :=
  P * R * T

noncomputable def T_years : ℚ :=
  5 + (8 / 12) + (12 / 365)

def principal : ℚ := 30000
def rate : ℚ := 23.7 / 100

theorem interest_earned :
  simple_interest principal rate T_years = 40524 := by
  sorry

end interest_earned_l205_205609


namespace average_children_l205_205866

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205866


namespace sin_double_angle_identity_l205_205382

noncomputable def given_tan_alpha (α : ℝ) : Prop := 
  Real.tan α = 1/2

theorem sin_double_angle_identity (α : ℝ) (h : given_tan_alpha α) : 
  Real.sin (2 * α) = 4 / 5 := 
sorry

end sin_double_angle_identity_l205_205382


namespace reflection_over_y_axis_correct_l205_205338

noncomputable def reflection_over_y_axis_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![ -1, 0],
    ![ 0, 1]]

theorem reflection_over_y_axis_correct (x y : ℝ) : 
  let p := (x, y)
  let p' := (-x, y)
  let A := reflection_over_y_axis_matrix 
  p' = A.mul_vec ![x, y] :=
by
  sorry

end reflection_over_y_axis_correct_l205_205338


namespace average_children_families_with_children_is_3_point_8_l205_205848

-- Define the main conditions
variables (total_families : ℕ) (average_children : ℕ) (childless_families : ℕ)
variable (total_children : ℕ)

axiom families_condition : total_families = 15
axiom average_children_condition : average_children = 3
axiom childless_families_condition : childless_families = 3
axiom total_children_condition : total_children = total_families * average_children

-- Definition for the average number of children in families with children
noncomputable def average_children_with_children_families : ℕ := total_children / (total_families - childless_families)

-- Theorem to prove
theorem average_children_families_with_children_is_3_point_8 :
  average_children_with_children_families total_families average_children childless_families total_children = 4 :=
by
  rw [families_condition, average_children_condition, childless_families_condition, total_children_condition]
  norm_num
  rw [div_eq_of_eq_mul _]
  norm_num
  sorry -- steps to show rounding of 3.75 to 3.8 can be written here if needed

end average_children_families_with_children_is_3_point_8_l205_205848


namespace no_solutions_ordered_triples_l205_205084

theorem no_solutions_ordered_triples :
  ¬ ∃ (x y z : ℤ), 
    x^2 - 4 * x * y + 3 * y^2 - z^2 = 25 ∧
    -x^2 + 5 * y * z + 3 * z^2 = 55 ∧
    x^2 + 2 * x * y + 9 * z^2 = 150 :=
by
  sorry

end no_solutions_ordered_triples_l205_205084


namespace value_of_fraction_l205_205748

theorem value_of_fraction : (121^2 - 112^2) / 9 = 233 := by
  -- use the difference of squares property
  sorry

end value_of_fraction_l205_205748


namespace roger_first_bag_correct_l205_205960

noncomputable def sandra_total_pieces : ℕ := 2 * 6
noncomputable def roger_total_pieces : ℕ := sandra_total_pieces + 2
noncomputable def roger_known_bag_pieces : ℕ := 3
noncomputable def roger_first_bag_pieces : ℕ := 11

theorem roger_first_bag_correct :
  roger_total_pieces - roger_known_bag_pieces = roger_first_bag_pieces := 
  by sorry

end roger_first_bag_correct_l205_205960


namespace base_six_to_ten_2154_l205_205992

def convert_base_six_to_ten (n : ℕ) : ℕ :=
  2 * 6^3 + 1 * 6^2 + 5 * 6^1 + 4 * 6^0

theorem base_six_to_ten_2154 :
  convert_base_six_to_ten 2154 = 502 :=
by
  sorry

end base_six_to_ten_2154_l205_205992


namespace probability_heads_at_least_10_out_of_12_l205_205743

theorem probability_heads_at_least_10_out_of_12 (n m : Nat) (hn : n = 12) (hm : m = 10):
  let total_outcomes := 2^n
  let ways_10 := Nat.choose n m
  let ways_11 := Nat.choose n (m + 1)
  let ways_12 := Nat.choose n (m + 2)
  let successful_outcomes := ways_10 + ways_11 + ways_12
  total_outcomes = 4096 →
  successful_outcomes = 79 →
  (successful_outcomes : ℚ) / total_outcomes = 79 / 4096 :=
by
  sorry

end probability_heads_at_least_10_out_of_12_l205_205743


namespace license_plate_difference_l205_205640

theorem license_plate_difference : 
    let alpha_plates := 26^4 * 10^4
    let beta_plates := 26^3 * 10^4
    alpha_plates - beta_plates = 10^4 * 26^3 * 25 := 
by sorry

end license_plate_difference_l205_205640


namespace value_of_b_l205_205279

theorem value_of_b (a b : ℤ) (h1 : 3 * a + 2 = 2) (h2 : b - a = 3) : b = 3 := 
by
  sorry

end value_of_b_l205_205279


namespace find_missing_dimension_of_carton_l205_205622

-- Definition of given dimensions and conditions
def carton_length : ℕ := 25
def carton_width : ℕ := 48
def soap_length : ℕ := 8
def soap_width : ℕ := 6
def soap_height : ℕ := 5
def max_soap_boxes : ℕ := 300
def soap_volume : ℕ := soap_length * soap_width * soap_height
def total_carton_volume : ℕ := max_soap_boxes * soap_volume

-- The main statement to prove
theorem find_missing_dimension_of_carton (h : ℕ) (volume_eq : carton_length * carton_width * h = total_carton_volume) : h = 60 :=
sorry

end find_missing_dimension_of_carton_l205_205622


namespace range_of_f_l205_205283

open Set

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log x / Real.log 3

theorem range_of_f :
  ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 3 → 2 ≤ f x ∧ f x ≤ 3 :=
by
  intro x hx
  sorry

end range_of_f_l205_205283


namespace extra_amount_spent_on_shoes_l205_205979

theorem extra_amount_spent_on_shoes (total_cost shirt_cost shoes_cost: ℝ) 
  (h1: total_cost = 300) (h2: shirt_cost = 97) 
  (h3: shoes_cost > 2 * shirt_cost)
  (h4: shirt_cost + shoes_cost = total_cost): 
  shoes_cost - 2 * shirt_cost = 9 :=
by
  sorry

end extra_amount_spent_on_shoes_l205_205979


namespace smallest_eraser_packs_needed_l205_205754

def yazmin_packs_condition (pencils_packs erasers_packs pencils_per_pack erasers_per_pack : ℕ) : Prop :=
  pencils_packs * pencils_per_pack = erasers_packs * erasers_per_pack

theorem smallest_eraser_packs_needed (pencils_per_pack erasers_per_pack : ℕ) (h_pencils_5 : pencils_per_pack = 5) (h_erasers_7 : erasers_per_pack = 7) : ∃ erasers_packs, yazmin_packs_condition 7 erasers_packs pencils_per_pack erasers_per_pack ∧ erasers_packs = 5 :=
by
  sorry

end smallest_eraser_packs_needed_l205_205754


namespace caricatures_sold_on_sunday_l205_205671

def caricature_price : ℕ := 20
def saturday_sales : ℕ := 24
def total_earnings : ℕ := 800

theorem caricatures_sold_on_sunday :
  (total_earnings - saturday_sales * caricature_price) / caricature_price = 16 :=
by
  sorry  -- Proof goes here

end caricatures_sold_on_sunday_l205_205671


namespace no_divide_five_to_n_minus_three_to_n_l205_205417

theorem no_divide_five_to_n_minus_three_to_n (n : ℕ) (h : n ≥ 1) : ¬ (2 ^ n + 65 ∣ 5 ^ n - 3 ^ n) :=
by
  sorry

end no_divide_five_to_n_minus_three_to_n_l205_205417


namespace largest_and_smallest_A_exists_l205_205293

theorem largest_and_smallest_A_exists (B B1 B2 : ℕ) (A_max A_min : ℕ) :
  -- Conditions: B > 666666666, B coprime with 24, and A obtained by moving the last digit to the first position
  B > 666666666 ∧ Nat.coprime B 24 ∧ 
  A_max = 10^8 * (B1 % 10) + B1 / 10 ∧ 
  A_min = 10^8 * (B2 % 10) + B2 / 10 ∧ 
  -- Values of B1 and B2 satisfying conditions
  B1 = 999999989 ∧ B2 = 666666671
  -- Largest and smallest A values
  ⊢ A_max = 999999998 ∧ A_min = 166666667 :=
sorry

end largest_and_smallest_A_exists_l205_205293


namespace TriangleRHS_solution_l205_205940

noncomputable def TriangleRHS (PQ PR : ℝ) :=
  let Q := (PQ, 0)
  let P := (0, 0)
  let R := (0, PR)
  let RQ := real.sqrt (PQ^2 + PR^2)
  let M := ((PQ + 0) / 2, (0 + R) / 2)
  let L := (PQ, 1.5*real.sqrt(3))
  let PF := 0.825 * real.sqrt(3)
  ∃ F: ℝ × ℝ, (L.1 ≠ M.1 ∧ L.2 ≠ M.2 ∧ L.1 = M.1 ∧ L.2 = PF)

variable {PQ PR : ℝ}

theorem TriangleRHS_solution : PQ = 3 → PR = 3 * real.sqrt 3 → 
  ∃ F : ℝ × ℝ, 0.825 * real.sqrt 3 := by
  intros hPQ hPR

  sorry

end TriangleRHS_solution_l205_205940


namespace rotation_matrix_150_degrees_l205_205357

open Real

noncomputable def rotation_matrix (θ : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    [cos θ, -sin θ],
    [sin θ, cos θ]
  ]

theorem rotation_matrix_150_degrees :
  rotation_matrix (150 * π / 180) = ![
    [-sqrt 3 / 2, -1 / 2],
    [ 1 / 2, -sqrt 3 / 2]
  ] :=
by
  sorry

end rotation_matrix_150_degrees_l205_205357


namespace inverse_of_composite_l205_205465

-- Define the function g
def g (x : ℕ) : ℕ :=
  if x = 1 then 4 else
  if x = 2 then 3 else
  if x = 3 then 1 else
  if x = 4 then 5 else
  if x = 5 then 2 else
  0  -- g is not defined for values other than 1 to 5

-- Define the inverse g_inv
def g_inv (x : ℕ) : ℕ :=
  if x = 4 then 1 else
  if x = 3 then 2 else
  if x = 1 then 3 else
  if x = 5 then 4 else
  if x = 2 then 5 else
  0  -- g_inv is not defined for values other than 1 to 5

theorem inverse_of_composite :
  g_inv (g_inv (g_inv 3)) = 4 :=
by
  sorry

end inverse_of_composite_l205_205465


namespace expand_expression_l205_205058

variable (x : ℝ)

theorem expand_expression : (9 * x + 4) * (2 * x ^ 2) = 18 * x ^ 3 + 8 * x ^ 2 :=
by sorry

end expand_expression_l205_205058


namespace pq_work_together_in_10_days_l205_205151

theorem pq_work_together_in_10_days 
  (p q r : ℝ)
  (hq : 1/q = 1/28)
  (hr : 1/r = 1/35)
  (hp : 1/p = 1/q + 1/r) :
  1/p + 1/q = 1/10 :=
by sorry

end pq_work_together_in_10_days_l205_205151


namespace value_of_x_squared_plus_inverse_squared_l205_205929

theorem value_of_x_squared_plus_inverse_squared (x : ℝ) (hx : x + 1 / x = 8) : x^2 + 1 / x^2 = 62 := 
sorry

end value_of_x_squared_plus_inverse_squared_l205_205929


namespace sqrt_factorial_squared_l205_205792

theorem sqrt_factorial_squared :
  (Real.sqrt ((Nat.factorial 5) * (Nat.factorial 4))) ^ 2 = 2880 :=
by sorry

end sqrt_factorial_squared_l205_205792


namespace leadership_configurations_l205_205300

theorem leadership_configurations (members chiefA_chiefsB chiefA_inferiors chiefB_inferiors: ℕ) :
  members = 12 →
  chiefA_chiefsB = 2 →
  chiefA_inferiors = 3 →
  chiefB_inferiors = 2 →
  (members * (members - 1) * (members - 2) * (Nat.choose (members - 3) chiefA_inferiors) * (Nat.choose (members - 3 - chiefA_inferiors) chiefB_inferiors)) = 1663200 :=
by
  intro h1 h2 h3 h4
  sorry

end leadership_configurations_l205_205300


namespace book_arrangement_count_l205_205215

theorem book_arrangement_count : 
  let math_books := 4
  let english_books := 4
  let groups := 1
  1 * math_books.factorial * english_books.factorial = 576 := by
  sorry

end book_arrangement_count_l205_205215


namespace probability_heads_at_least_10_in_12_flips_l205_205742

theorem probability_heads_at_least_10_in_12_flips :
  let total_outcomes := 2^12
  let favorable_outcomes := (Nat.choose 12 10) + (Nat.choose 12 11) + (Nat.choose 12 12)
  let probability := (favorable_outcomes : ℚ) / (total_outcomes : ℚ)
  probability = 79 / 4096 := by
  sorry

end probability_heads_at_least_10_in_12_flips_l205_205742


namespace unique_non_overtaken_city_l205_205467

structure City :=
(size_left : ℕ)
(size_right : ℕ)

def canOvertake (A B : City) : Prop :=
  A.size_right > B.size_left 

theorem unique_non_overtaken_city (n : ℕ) (H : n > 0) (cities : Fin n → City) : 
  ∃! i : Fin n, ∀ j : Fin n, ¬ canOvertake (cities j) (cities i) :=
by
  sorry

end unique_non_overtaken_city_l205_205467


namespace employee_Y_base_pay_l205_205066

theorem employee_Y_base_pay (P : ℝ) (h1 : 1.2 * P + P * 1.1 + P * 1.08 + P = P * 4.38)
                            (h2 : 2 * 1.5 * 1.2 * P = 3.6 * P)
                            (h3 : P * 4.38 + 100 + 3.6 * P = 1800) :
  P = 213.03 :=
by
  sorry

end employee_Y_base_pay_l205_205066


namespace coordinates_of_P_respect_to_symmetric_y_axis_l205_205557

-- Definition of points in the Cartesian coordinate system
structure Point where
  x : ℝ
  y : ℝ

def symmetric_x_axis (p : Point) : Point :=
  { p with y := -p.y }

def symmetric_y_axis (p : Point) : Point :=
  { p with x := -p.x }

-- The given condition
def P_with_respect_to_symmetric_x_axis := Point.mk (-1) 2

-- The problem statement
theorem coordinates_of_P_respect_to_symmetric_y_axis :
    symmetric_y_axis (symmetric_x_axis P_with_respect_to_symmetric_x_axis) = Point.mk 1 (-2) :=
by
  sorry

end coordinates_of_P_respect_to_symmetric_y_axis_l205_205557


namespace connor_total_cost_l205_205321

def ticket_cost : ℕ := 10
def combo_meal_cost : ℕ := 11
def candy_cost : ℕ := 2.5

def total_cost : ℕ := ticket_cost + ticket_cost + combo_meal_cost + candy_cost + candy_cost

theorem connor_total_cost : total_cost = 36 := 
by sorry

end connor_total_cost_l205_205321


namespace numeric_puzzle_AB_eq_B_pow_V_l205_205894

theorem numeric_puzzle_AB_eq_B_pow_V 
  (A B V : ℕ)
  (h_A_different_digits : A ≠ B ∧ A ≠ V ∧ B ≠ V)
  (h_AB_two_digits : 10 ≤ 10 * A + B ∧ 10 * A + B < 100) :
  (10 * A + B = B^V) ↔ 
  (10 * A + B = 32 ∨ 10 * A + B = 36 ∨ 10 * A + B = 64) :=
sorry

end numeric_puzzle_AB_eq_B_pow_V_l205_205894


namespace post_office_mail_in_six_months_l205_205459

/-- The number of pieces of mail the post office receives per day -/
def mail_per_day : ℕ := 60 + 20

/-- The number of days in six months, assuming each month has 30 days -/
def days_in_six_months : ℕ := 6 * 30

/-- The total number of pieces of mail handled in six months -/
def total_mail_in_six_months : ℕ := mail_per_day * days_in_six_months

/-- The post office handles 14400 pieces of mail in six months -/
theorem post_office_mail_in_six_months : total_mail_in_six_months = 14400 := by
  sorry

end post_office_mail_in_six_months_l205_205459


namespace total_area_of_histogram_l205_205270

section FrequencyDistributionHistogram

variable {n : ℕ} {w : ℝ} {f : Fin n → ℝ}

theorem total_area_of_histogram (h_w : 0 < w)
  (h_bins : ∀ i, 0 ≤ f i) :
  let A_i := λ i, f i * w in
  let A_total := Finset.univ.sum (λ i, A_i i) in
  A_total = w * Finset.univ.sum f :=
by
  sorry

end FrequencyDistributionHistogram

end total_area_of_histogram_l205_205270


namespace minimum_value_8_l205_205061

noncomputable def minimum_value (x : ℝ) : ℝ :=
  3 * x + 2 / x^5 + 3 / x

theorem minimum_value_8 (x : ℝ) (hx : x > 0) :
  ∃ y : ℝ, (∀ z > 0, minimum_value z ≥ y) ∧ (y = 8) :=
by
  sorry

end minimum_value_8_l205_205061


namespace average_children_l205_205871

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205871


namespace rationalize_denominator_l205_205585

theorem rationalize_denominator : 
  let x := (1 : ℝ)
  let y := (3 : ℝ)
  let z := real.cbrt 3
  let w := real.cbrt 27
  (w = 3) →
  x / (z + w) = real.cbrt (9) / (3 * (real.cbrt (9) + 1)) := 
by
  intros _ h
  rw [h]
  sorry

end rationalize_denominator_l205_205585


namespace jacqueline_candy_multiple_l205_205534

theorem jacqueline_candy_multiple :
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  (jackie_candy / total_candy = 10) :=
by
  let fred_candy := 12
  let uncle_bob_candy := fred_candy + 6
  let total_candy := fred_candy + uncle_bob_candy
  let jackie_candy := 120 / 0.40
  show _ = _
  sorry

end jacqueline_candy_multiple_l205_205534


namespace sphere_radius_eq_l205_205728

theorem sphere_radius_eq (h d : ℝ) (r_cylinder : ℝ) (r : ℝ) (pi : ℝ) 
  (h_eq : h = 14) (d_eq : d = 14) (r_cylinder_eq : r_cylinder = d / 2) :
  4 * pi * r^2 = 2 * pi * r_cylinder * h → r = 7 := by
  sorry

end sphere_radius_eq_l205_205728


namespace average_waiting_time_for_first_bite_l205_205511

/-- 
Let S be a period of 5 minutes (300 seconds).
- We have an average of 5 bites in 300 seconds on the first fishing rod.
- We have an average of 1 bite in 300 seconds on the second fishing rod.
- The total average number of bites on both rods during this period is 6 bites.
The bites occur independently and follow a Poisson process.

We aim to prove that the waiting time for the first bite, given these conditions, is 
expected to be 50 seconds.
-/
theorem average_waiting_time_for_first_bite :
  let S := 300 -- 5 minutes in seconds
  -- The average number of bites on the first and second rod in period S.
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  -- The rate parameter λ for the Poisson process is total_avg_bites / S.
  let λ := total_avg_bites / S
  -- The average waiting time for the first bite.
  1 / λ = 50 :=
by
  let S := 300
  let avg_bites1 := 5
  let avg_bites2 := 1
  let total_avg_bites := avg_bites1 + avg_bites2
  let λ := total_avg_bites / S
  -- convert λ to seconds to ensure unit consistency
  have hλ: λ = 6 / 300 := rfl
  -- The expected waiting time for the first bite is 1 / λ
  have h_waiting_time: 1 / λ = 300 / 6 := by
    rw [hλ, one_div, div_div_eq_mul]
    norm_num
  exact h_waiting_time

end average_waiting_time_for_first_bite_l205_205511


namespace determine_OQ_l205_205374

theorem determine_OQ (l m n p O A B C D Q : ℝ) (h0 : O = 0)
  (hA : A = l) (hB : B = m) (hC : C = n) (hD : D = p)
  (hQ : l ≤ Q ∧ Q ≤ m)
  (h_ratio : (|C - Q| / |Q - D|) = (|B - Q| / |Q - A|)) :
  Q = (l + m) / 2 :=
sorry

end determine_OQ_l205_205374


namespace area_of_region_l205_205608

-- Define the equation as a predicate
def region (x y : ℝ) : Prop := x^2 + y^2 + 6*x = 2*y + 10

-- The proof statement
theorem area_of_region : (∃ (x y : ℝ), region x y) → ∃ A : ℝ, A = 20 * Real.pi :=
by 
  sorry

end area_of_region_l205_205608


namespace ann_susan_age_sum_l205_205632

theorem ann_susan_age_sum (ann_age : ℕ) (susan_age : ℕ) (h1 : ann_age = 16) (h2 : ann_age = susan_age + 5) : ann_age + susan_age = 27 :=
by
  sorry

end ann_susan_age_sum_l205_205632


namespace Will_worked_on_Tuesday_l205_205284

variable (HourlyWage MondayHours TotalEarnings : ℝ)

-- Given conditions
def Wage : ℝ := 8
def Monday_worked_hours : ℝ := 8
def Total_two_days_earnings : ℝ := 80

theorem Will_worked_on_Tuesday (HourlyWage_eq : HourlyWage = Wage)
  (MondayHours_eq : MondayHours = Monday_worked_hours)
  (TotalEarnings_eq : TotalEarnings = Total_two_days_earnings) :
  let MondayEarnings := MondayHours * HourlyWage
  let TuesdayEarnings := TotalEarnings - MondayEarnings
  let TuesdayHours := TuesdayEarnings / HourlyWage
  TuesdayHours = 2 :=
by
  sorry

end Will_worked_on_Tuesday_l205_205284


namespace simplify_fraction_l205_205591

theorem simplify_fraction (m : ℝ) (h₁: m ≠ 0) (h₂: m ≠ 1): (m - 1) / m / ((m - 1) / (m * m)) = m := by
  sorry

end simplify_fraction_l205_205591


namespace equilateral_triangle_ab_l205_205965

noncomputable def a : ℝ := 25 * Real.sqrt 3
noncomputable def b : ℝ := 5 * Real.sqrt 3

theorem equilateral_triangle_ab
  (a_val : a = 25 * Real.sqrt 3)
  (b_val : b = 5 * Real.sqrt 3)
  (h1 : Complex.abs (a + 15 * Complex.I) = 25)
  (h2 : Complex.abs (b + 45 * Complex.I) = 45)
  (h3 : Complex.abs ((a - b) + (15 - 45) * Complex.I) = 30) :
  a * b = 375 := 
sorry

end equilateral_triangle_ab_l205_205965


namespace multiplication_equivalence_l205_205765

theorem multiplication_equivalence :
    44 * 22 = 88 * 11 :=
by
  sorry

end multiplication_equivalence_l205_205765


namespace binom_18_4_eq_3060_l205_205804

theorem binom_18_4_eq_3060 : nat.choose 18 4 = 3060 := sorry

end binom_18_4_eq_3060_l205_205804


namespace solution_to_diff_eq_l205_205740

def y (x C : ℝ) : ℝ := x^2 + x + C

theorem solution_to_diff_eq (C : ℝ) : ∀ x : ℝ, 
  (dy = (2 * x + 1) * dx) :=
by
  sorry

end solution_to_diff_eq_l205_205740


namespace deductive_reasoning_correct_l205_205750

theorem deductive_reasoning_correct :
  (∀ (s : ℕ), s = 3 ↔
    (s == 1 → DeductiveReasoningGeneralToSpecific ∧
     s == 2 → alwaysCorrect ∧
     s == 3 → InFormOfSyllogism ∧
     s == 4 → ConclusionDependsOnPremisesAndForm)) :=
sorry

end deductive_reasoning_correct_l205_205750


namespace num_int_values_x_l205_205005

theorem num_int_values_x (x : ℕ) :
  (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) → ∃ n : ℕ, n = 3 :=
by
  sorry

end num_int_values_x_l205_205005


namespace jill_peaches_l205_205565

-- Definitions based on conditions in a
def Steven_has_peaches : ℕ := 19
def Steven_more_than_Jill : ℕ := 13

-- Statement to prove Jill's peaches
theorem jill_peaches : (Steven_has_peaches - Steven_more_than_Jill = 6) :=
by
  sorry

end jill_peaches_l205_205565


namespace arrangement_count_PERSEVERANCE_l205_205646

theorem arrangement_count_PERSEVERANCE : 
  let count := 12!
  let repeat_E := 3!
  let repeat_R := 2!
  count / (repeat_E * repeat_R) = 39916800 :=
by
  sorry

end arrangement_count_PERSEVERANCE_l205_205646


namespace cookies_difference_l205_205440

theorem cookies_difference :
  let bags := 9
  let boxes := 8
  let cookies_per_bag := 7
  let cookies_per_box := 12
  8 * 12 - 9 * 7 = 33 := 
by
  sorry

end cookies_difference_l205_205440


namespace common_number_of_two_sets_l205_205971

theorem common_number_of_two_sets (a b c d e f g : ℚ) :
  (a + b + c + d) / 4 = 5 →
  (d + e + f + g) / 4 = 8 →
  (a + b + c + d + e + f + g) / 7 = 46 / 7 →
  d = 6 :=
by
  intros h₁ h₂ h₃
  sorry

end common_number_of_two_sets_l205_205971


namespace positive_x_condition_l205_205491

theorem positive_x_condition (x : ℝ) (h : x > 0 ∧ (0.01 * x * x = 9)) : x = 30 :=
sorry

end positive_x_condition_l205_205491


namespace avg_children_with_kids_l205_205829

theorem avg_children_with_kids 
  (num_families total_families childless_families : ℕ)
  (avg_children_per_family : ℚ)
  (H_total_families : total_families = 15)
  (H_avg_children_per_family : avg_children_per_family = 3)
  (H_childless_families : childless_families = 3)
  (H_num_families : num_families = total_families - childless_families) 
  : (45 / num_families).round = 4 := 
by
  -- Prove that the average is 3.8 rounded up to the nearest tenth
  sorry

end avg_children_with_kids_l205_205829


namespace gcd_lcm_sum_l205_205506

-- Definitions
def gcd_42_70 := Nat.gcd 42 70
def lcm_8_32 := Nat.lcm 8 32

-- Theorem statement
theorem gcd_lcm_sum : gcd_42_70 + lcm_8_32 = 46 := by
  sorry

end gcd_lcm_sum_l205_205506


namespace cube_inequality_contradiction_l205_205433

variable {x y : ℝ}

theorem cube_inequality_contradiction (h : x < y) (hne : x^3 ≥ y^3) : false :=
by 
  sorry

end cube_inequality_contradiction_l205_205433


namespace rotation_matrix_150_l205_205352

noncomputable def cos_150 : ℝ := -real.cos (real.pi / 6)
noncomputable def sin_150 : ℝ := real.sin (real.pi / 6)

theorem rotation_matrix_150 : 
  ∀ θ : ℝ, θ = 5 * real.pi / 6 → 
  (matrix (fin 2) (fin 2) ℝ) := 
begin
  assume θ hθ,
  rw hθ,
  exact matrix.cons_vec_cons
    (matrix.cons_vec_cons cos_150 (-sin_150))
    (matrix.cons_vec_cons sin_150 cos_150),
  sorry
end

end rotation_matrix_150_l205_205352


namespace erika_walked_distance_l205_205181

/-- Erika traveled to visit her cousin. She started on a scooter at an average speed of 
22 kilometers per hour. After completing three-fifths of the distance, the scooter's battery died, 
and she walked the rest of the way at 4 kilometers per hour. The total time it took her to reach her cousin's 
house was 2 hours. How far, in kilometers rounded to the nearest tenth, did Erika walk? -/
theorem erika_walked_distance (d : ℝ) (h1 : d > 0)
  (h2 : (3 / 5 * d) / 22 + (2 / 5 * d) / 4 = 2) : 
  (2 / 5 * d) = 6.3 :=
sorry

end erika_walked_distance_l205_205181


namespace proof_problem_l205_205205

theorem proof_problem
  (x y : ℚ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 16) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 3280 / 9 :=
sorry

end proof_problem_l205_205205


namespace last_digit_square_of_second_l205_205407

def digit1 := 1
def digit2 := 3
def digit3 := 4
def digit4 := 9

theorem last_digit_square_of_second :
  digit4 = digit2 ^ 2 :=
by
  -- Conditions
  have h1 : digit1 = digit2 / 3 := by sorry
  have h2 : digit3 = digit1 + digit2 := by sorry
  sorry

end last_digit_square_of_second_l205_205407


namespace cycle_selling_price_l205_205771

theorem cycle_selling_price
  (cost_price : ℝ)
  (gain_percentage : ℝ)
  (profit : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 930)
  (h2 : gain_percentage = 30.107526881720432)
  (h3 : profit = (gain_percentage / 100) * cost_price)
  (h4 : selling_price = cost_price + profit)
  : selling_price = 1210 := 
sorry

end cycle_selling_price_l205_205771


namespace concert_attendance_difference_l205_205577

/-- Define the number of people attending the first concert. -/
def first_concert_attendance : ℕ := 65899

/-- Define the number of people attending the second concert. -/
def second_concert_attendance : ℕ := 66018

/-- The proof statement that the difference in attendance between the second and first concert is 119. -/
theorem concert_attendance_difference :
  (second_concert_attendance - first_concert_attendance = 119) := by
  sorry

end concert_attendance_difference_l205_205577


namespace duration_of_each_movie_l205_205036

-- define the conditions
def num_screens : ℕ := 6
def hours_open : ℕ := 8
def num_movies : ℕ := 24

-- define the total screening time
def total_screening_time : ℕ := num_screens * hours_open

-- define the expected duration of each movie
def movie_duration : ℕ := total_screening_time / num_movies

-- state the theorem
theorem duration_of_each_movie : movie_duration = 2 := by sorry

end duration_of_each_movie_l205_205036


namespace geometric_sequence_general_term_and_sum_l205_205077

theorem geometric_sequence_general_term_and_sum (a : ℕ → ℝ) (b : ℕ → ℝ) (T : ℕ → ℝ) 
  (h₁ : ∀ n, a n = 2 ^ n)
  (h₂ : ∀ n, b n = 2 * n - 1)
  : (∀ n, T n = 6 + (2 * n - 3) * 2 ^ (n + 1)) :=
by {
  sorry
}

end geometric_sequence_general_term_and_sum_l205_205077


namespace numerical_puzzle_solution_l205_205902

theorem numerical_puzzle_solution (A B V : ℕ) (h_diff_digits : A ≠ B) (h_two_digit : 10 ≤ A * 10 + B ∧ A * 10 + B < 100) :
  (A * 10 + B = B^V) → (A = 3 ∧ B = 2 ∧ V = 5) ∨ (A = 3 ∧ B = 6 ∧ V = 2) ∨ (A = 6 ∧ B = 4 ∧ V = 3) :=
sorry

end numerical_puzzle_solution_l205_205902


namespace solve_inequality_l205_205269

theorem solve_inequality (x : ℝ) : 2 * x + 4 > 0 ↔ x > -2 := sorry

end solve_inequality_l205_205269


namespace total_number_of_squares_is_13_l205_205214

-- Define the vertices of the region
def region_condition (x y : ℕ) : Prop :=
  y ≤ x ∧ y ≤ 4 ∧ x ≤ 4

-- Define the type of squares whose vertices have integer coordinates
def square (n : ℕ) (x y : ℕ) : Prop :=
  region_condition x y ∧ region_condition (x - n) y ∧ 
  region_condition x (y - n) ∧ region_condition (x - n) (y - n)

-- Count the number of squares of each size within the region
def number_of_squares (size : ℕ) : ℕ :=
  match size with
  | 1 => 10 -- number of 1x1 squares
  | 2 => 3  -- number of 2x2 squares
  | _ => 0  -- there are no larger squares in this context

-- Prove the total number of squares is 13
theorem total_number_of_squares_is_13 : number_of_squares 1 + number_of_squares 2 = 13 :=
by
  sorry

end total_number_of_squares_is_13_l205_205214


namespace mail_handling_in_six_months_l205_205456

theorem mail_handling_in_six_months (daily_letters daily_packages days_per_month months : ℕ) :
  daily_letters = 60 →
  daily_packages = 20 →
  days_per_month = 30 →
  months = 6 →
  (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  -- Skipping the proof
  sorry

end mail_handling_in_six_months_l205_205456


namespace infinite_geometric_sum_example_l205_205541

noncomputable def infinite_geometric_sum (a₁ q : ℝ) : ℝ :=
a₁ / (1 - q)

theorem infinite_geometric_sum_example :
  infinite_geometric_sum 18 (-1/2) = 12 := by
  sorry

end infinite_geometric_sum_example_l205_205541


namespace asparagus_cost_correct_l205_205701

def cost_asparagus (total_start: Int) (total_left: Int) (cost_bananas: Int) (cost_pears: Int) (cost_chicken: Int) : Int := 
  total_start - total_left - cost_bananas - cost_pears - cost_chicken

theorem asparagus_cost_correct :
  cost_asparagus 55 28 8 2 11 = 6 :=
by
  sorry

end asparagus_cost_correct_l205_205701


namespace inverse_of_g_compose_three_l205_205463

def g (x : ℕ) : ℕ :=
  match x with
  | 1 => 4
  | 2 => 3
  | 3 => 1
  | 4 => 5
  | 5 => 2
  | _ => 0  -- Assuming g(x) is defined only for x in {1, 2, 3, 4, 5}

noncomputable def g_inv (y : ℕ) : ℕ :=
  match y with
  | 4 => 1
  | 3 => 2
  | 1 => 3
  | 5 => 4
  | 2 => 5
  | _ => 0  -- Assuming g_inv(y) is defined only for y in {1, 3, 1, 5, 2}

theorem inverse_of_g_compose_three : g_inv (g_inv (g_inv 3)) = 4 := by
  sorry

end inverse_of_g_compose_three_l205_205463


namespace log2_125_eq_9y_l205_205114

theorem log2_125_eq_9y (y : ℝ) (h : Real.log 5 / Real.log 8 = y) : Real.log 125 / Real.log 2 = 9 * y :=
by
  sorry

end log2_125_eq_9y_l205_205114


namespace log_eq_res_l205_205112

theorem log_eq_res (y m : ℝ) (h₁ : real.log 5 / real.log 8 = y) (h₂ : real.log 125 / real.log 2 = m * y) : m = 9 := 
sorry

end log_eq_res_l205_205112


namespace sin_cos_value_l205_205198

theorem sin_cos_value (x : ℝ) (h : Real.cos x - 3 * Real.sin x = 2) :
  (3 * Real.sin x + Real.cos x = 0) ∨ (3 * Real.sin x + Real.cos x = -4) :=
sorry

end sin_cos_value_l205_205198


namespace initial_holes_count_additional_holes_needed_l205_205700

-- Defining the conditions as variables
def circumference : ℕ := 400
def initial_interval : ℕ := 50
def new_interval : ℕ := 40

-- Defining the problems

-- Problem 1: Calculate the number of holes for the initial interval
theorem initial_holes_count (circumference : ℕ) (initial_interval : ℕ) : 
  circumference % initial_interval = 0 → 
  circumference / initial_interval = 8 := 
sorry

-- Problem 2: Calculate the additional holes needed
theorem additional_holes_needed (circumference : ℕ) (initial_interval : ℕ) 
  (new_interval : ℕ) (lcm_interval : ℕ) :
  lcm new_interval initial_interval = lcm_interval →
  circumference % new_interval = 0 →
  circumference / new_interval - 
  (circumference / lcm_interval) = 8 :=
sorry

end initial_holes_count_additional_holes_needed_l205_205700


namespace expected_waiting_time_first_bite_l205_205518

-- Definitions and conditions as per the problem
def poisson_rate := 6  -- lambda value, bites per 5 minutes
def interval_minutes := 5
def interval_seconds := interval_minutes * 60
def expected_waiting_time_seconds := interval_seconds / poisson_rate

-- The theorem we want to prove
theorem expected_waiting_time_first_bite :
  expected_waiting_time_seconds = 50 := 
by
  let x := interval_seconds / poisson_rate
  have h : interval_seconds = 300 := by norm_num; rfl
  have h2 : x = 50 := by rw [h, interval_seconds]; norm_num
  exact h2

end expected_waiting_time_first_bite_l205_205518


namespace solve_fractional_equation_l205_205256

theorem solve_fractional_equation (x : ℝ) (h : (4 * x^2 - 3 * x + 2) / (x + 2) = 4 * x - 3) : 
  x = 1 :=
sorry

end solve_fractional_equation_l205_205256


namespace binomial_equality_l205_205810

theorem binomial_equality : (Nat.choose 18 4) = 3060 := by
  sorry

end binomial_equality_l205_205810


namespace find_original_volume_l205_205997

theorem find_original_volume
  (V : ℝ)
  (h1 : V - (3 / 4) * V = (1 / 4) * V)
  (h2 : (1 / 4) * V - (3 / 4) * ((1 / 4) * V) = (1 / 16) * V)
  (h3 : (1 / 16) * V = 0.2) :
  V = 3.2 :=
by 
  -- Proof skipped, as the assistant is instructed to provide only the statement 
  sorry

end find_original_volume_l205_205997


namespace minimum_value_of_objective_function_l205_205244

theorem minimum_value_of_objective_function :
  ∃ (x y : ℝ), x - y + 2 ≥ 0 ∧ 2 * x + 3 * y - 6 ≥ 0 ∧ 3 * x + 2 * y - 9 ≤ 0 ∧ (∀ (x' y' : ℝ), x' - y' + 2 ≥ 0 ∧ 2 * x' + 3 * y' - 6 ≥ 0 ∧ 3 * x' + 2 * y' - 9 ≤ 0 → 2 * x + 5 * y ≤ 2 * x' + 5 * y') ∧ 2 * x + 5 * y = 6 :=
sorry

end minimum_value_of_objective_function_l205_205244


namespace probability_edge_within_five_hops_l205_205194

noncomputable def GabbyGrid := fin 4 × fin 4

noncomputable def is_interior (pos : GabbyGrid) : Prop :=
  match pos with
  | (⟨2, h1⟩, ⟨2, h2⟩) => true
  | (⟨2, h1⟩, ⟨3, h2⟩) => true
  | (⟨3, h1⟩, ⟨2, h2⟩) => true
  | (⟨3, h1⟩, ⟨3, h2⟩) => true
  | _ => false

noncomputable def is_edge (pos : GabbyGrid) : Prop :=
  match pos with
  | (⟨0, _⟩, _) => true
  | (⟨3, _⟩, _) => true
  | (_, ⟨0, _⟩) => true
  | (_, ⟨3, _⟩) => true
  | _ => false

noncomputable def probability_gabby_reaches_edge : ℚ :=
  27 / 64

theorem probability_edge_within_five_hops :
  ∀ (start : GabbyGrid), is_interior start →
  (gabby_hops : fin 5 → GabbyGrid → GabbyGrid) →
  finset.univ.filter (λ start, is_edge (gabby_hops 4 start)).card.to_rat / finset.univ.card.to_rat = probability_gabby_reaches_edge :=
sorry

end probability_edge_within_five_hops_l205_205194


namespace initial_stock_decaf_percentage_l205_205298

variable (x : ℝ)
variable (initialStock newStock totalStock initialDecaf newDecaf totalDecaf: ℝ)

theorem initial_stock_decaf_percentage :
  initialStock = 400 ->
  newStock = 100 ->
  totalStock = 500 ->
  initialDecaf = initialStock * x / 100 ->
  newDecaf = newStock * 60 / 100 ->
  totalDecaf = 180 ->
  initialDecaf + newDecaf = totalDecaf ->
  x = 30 := by
  intros h₁ h₂ h₃ h₄ h₅ h₆ h₇
  sorry

end initial_stock_decaf_percentage_l205_205298


namespace joan_total_money_l205_205978

-- Define the number of each type of coin found
def dimes_jacket : ℕ := 15
def dimes_shorts : ℕ := 4
def nickels_shorts : ℕ := 7
def quarters_jeans : ℕ := 12
def pennies_jeans : ℕ := 2
def nickels_backpack : ℕ := 8
def pennies_backpack : ℕ := 23

-- Calculate the total number of each type of coin
def total_dimes : ℕ := dimes_jacket + dimes_shorts
def total_nickels : ℕ := nickels_shorts + nickels_backpack
def total_quarters : ℕ := quarters_jeans
def total_pennies : ℕ := pennies_jeans + pennies_backpack

-- Calculate the total value of each type of coin
def value_dimes : ℝ := total_dimes * 0.10
def value_nickels : ℝ := total_nickels * 0.05
def value_quarters : ℝ := total_quarters * 0.25
def value_pennies : ℝ := total_pennies * 0.01

-- Calculate the total amount of money found
def total_money : ℝ := value_dimes + value_nickels + value_quarters + value_pennies

-- Proof statement
theorem joan_total_money : total_money = 5.90 := by
  sorry

end joan_total_money_l205_205978


namespace average_children_in_families_with_children_l205_205886

-- Definitions of the conditions
def total_families : Nat := 15
def average_children_per_family : ℕ := 3
def childless_families : Nat := 3
def total_children : ℕ := total_families * average_children_per_family
def families_with_children : ℕ := total_families - childless_families

-- Theorem statement
theorem average_children_in_families_with_children :
  (total_children.toFloat / families_with_children.toFloat).round = 3.8 :=
by
  sorry

end average_children_in_families_with_children_l205_205886


namespace celer_tanks_dimensions_l205_205250

theorem celer_tanks_dimensions :
  ∃ (a v : ℕ), 
    (a * a * v = 200) ∧
    (2 * a ^ 3 + 50 = 300) ∧
    (a = 5) ∧
    (v = 8) :=
sorry

end celer_tanks_dimensions_l205_205250


namespace solve_x_l205_205212

variable (x : ℝ)

def vector_a := (2, 1)
def vector_b := (1, x)

def vectors_parallel : Prop :=
  let a_plus_b := (2 + 1, 1 + x)
  let a_minus_b := (2 - 1, 1 - x)
  a_plus_b.1 * a_minus_b.2 = a_plus_b.2 * a_minus_b.1

theorem solve_x (hx : vectors_parallel x) : x = 1/2 := by
  sorry

end solve_x_l205_205212


namespace probability_of_at_least_10_heads_l205_205745

open ProbabilityTheory

noncomputable def probability_at_least_10_heads_in_12_flips : ℚ :=
  let total_outcomes := (2 : ℕ) ^ 12 in
  let ways_10_heads := Nat.choose 12 10 in
  let ways_11_heads := Nat.choose 12 11 in
  let ways_12_heads := Nat.choose 12 12 in
  let heads_ways := ways_10_heads + ways_11_heads + ways_12_heads in
  (heads_ways : ℚ) / (total_outcomes : ℚ)

theorem probability_of_at_least_10_heads :
  probability_at_least_10_heads_in_12_flips = 79 / 4096 := sorry

end probability_of_at_least_10_heads_l205_205745


namespace g_inv_3_l205_205462

-- Define the function g and its inverse g_inv based on the provided table.
def g : ℕ → ℕ
| 1 := 4
| 2 := 3
| 3 := 1
| 4 := 5
| 5 := 2
| _ := 0  -- arbitrary definition for other values

def g_inv : ℕ → ℕ
| 4 := 1
| 3 := 2
| 1 := 3
| 5 := 4
| 2 := 5
| _ := 0  -- arbitrary definition for other values

-- The theorem to prove the inverse property based on the given conditions
theorem g_inv_3 : g_inv (g_inv (g_inv 3)) = 4 :=
by
  -- Proof skipped using sorry
  sorry

end g_inv_3_l205_205462


namespace work_completion_days_l205_205296

theorem work_completion_days (A_time : ℝ) (A_efficiency : ℝ) (B_time : ℝ) (B_efficiency : ℝ) (C_time : ℝ) (C_efficiency : ℝ) :
  A_time = 60 → A_efficiency = 1.5 → B_time = 20 → B_efficiency = 1 → C_time = 30 → C_efficiency = 0.75 → 
  (1 / (A_efficiency / A_time + B_efficiency / B_time + C_efficiency / C_time)) = 10 := 
by
  intros A_time_eq A_efficiency_eq B_time_eq B_efficiency_eq C_time_eq C_efficiency_eq
  rw [A_time_eq, A_efficiency_eq, B_time_eq, B_efficiency_eq, C_time_eq, C_efficiency_eq]
  -- Proof omitted
  sorry

end work_completion_days_l205_205296


namespace vector_dot_product_calculation_l205_205926

theorem vector_dot_product_calculation : 
  let a := (2, 3, -1)
  let b := (2, 0, 3)
  let c := (0, 2, 2)
  (2 * (2 + 0) + 3 * (0 + 2) + -1 * (3 + 2)) = 5 := 
by
  sorry

end vector_dot_product_calculation_l205_205926


namespace ticket_ratio_proof_l205_205130

-- Define the initial number of tickets Tate has.
def initial_tate_tickets : ℕ := 32

-- Define the additional tickets Tate buys.
def additional_tickets : ℕ := 2

-- Define the total tickets they have together.
def combined_tickets : ℕ := 51

-- Calculate Tate's total number of tickets after buying more tickets.
def total_tate_tickets := initial_tate_tickets + additional_tickets

-- Define the number of tickets Peyton has.
def peyton_tickets := combined_tickets - total_tate_tickets

-- Define the ratio of Peyton's tickets to Tate's tickets.
def tickets_ratio := peyton_tickets / total_tate_tickets

theorem ticket_ratio_proof : tickets_ratio = 1 / 2 :=
by
  unfold tickets_ratio peyton_tickets total_tate_tickets initial_tate_tickets additional_tickets
  norm_num
  sorry

end ticket_ratio_proof_l205_205130


namespace part_a_l205_205759

theorem part_a (x : ℝ) : (6 - x) / x = 3 / 6 → x = 4 := by
  sorry

end part_a_l205_205759


namespace negation_of_prop_l205_205599

theorem negation_of_prop (P : Prop) :
  (¬ ∀ x > 0, x - 1 ≥ Real.log x) ↔ ∃ x > 0, x - 1 < Real.log x :=
by
  sorry

end negation_of_prop_l205_205599


namespace payment_to_Y_is_227_27_l205_205972

-- Define the conditions
def total_payment_per_week (x y : ℝ) : Prop :=
  x + y = 500

def x_payment_is_120_percent_of_y (x y : ℝ) : Prop :=
  x = 1.2 * y

-- Formulate the problem as a theorem to be proven
theorem payment_to_Y_is_227_27 (Y : ℝ) (X : ℝ) 
  (h1 : total_payment_per_week X Y) 
  (h2 : x_payment_is_120_percent_of_y X Y) : 
  Y = 227.27 :=
by
  sorry

end payment_to_Y_is_227_27_l205_205972


namespace cost_difference_is_35_88_usd_l205_205121

/-
  Mr. Llesis bought 50 kilograms of rice at different prices per kilogram from various suppliers.
  He bought:
  - 15 kilograms at €1.2 per kilogram from Supplier A
  - 10 kilograms at €1.4 per kilogram from Supplier B
  - 12 kilograms at €1.6 per kilogram from Supplier C
  - 8 kilograms at €1.9 per kilogram from Supplier D
  - 5 kilograms at €2.3 per kilogram from Supplier E

  He kept 7/10 of the total rice in storage and gave the rest to Mr. Everest.
  The current conversion rate is €1 = $1.15.
  
  Prove that the difference in cost in US dollars between the rice kept and the rice given away is $35.88.
-/

def euros_to_usd (euros : ℚ) : ℚ :=
  euros * (115 / 100)

def total_cost : ℚ := 
  (15 * 1.2) + (10 * 1.4) + (12 * 1.6) + (8 * 1.9) + (5 * 2.3)

def cost_kept : ℚ := (7/10) * total_cost
def cost_given : ℚ := (3/10) * total_cost

theorem cost_difference_is_35_88_usd :
  euros_to_usd cost_kept - euros_to_usd cost_given = 35.88 := 
sorry

end cost_difference_is_35_88_usd_l205_205121


namespace lateral_area_of_given_cone_l205_205076

noncomputable def lateral_area_cone (r h : ℝ) : ℝ :=
  let l := Real.sqrt (r^2 + h^2)
  (Real.pi * r * l)

theorem lateral_area_of_given_cone :
  lateral_area_cone 3 4 = 15 * Real.pi :=
by
  -- sorry to skip the proof
  sorry

end lateral_area_of_given_cone_l205_205076


namespace num_children_with_identical_cards_l205_205264

theorem num_children_with_identical_cards (children_mama children_nyanya children_manya total_children mixed_cards : ℕ) 
  (h_mama: children_mama = 20) 
  (h_nyanya: children_nyanya = 30) 
  (h_manya: children_manya = 40) 
  (h_total: total_children = children_mama + children_nyanya) 
  (h_mixed: mixed_cards = children_manya) 
  : total_children - children_manya = 10 :=
by
  -- Sorry to indicate the proof is skipped
  sorry

end num_children_with_identical_cards_l205_205264


namespace tangent_line_eq_l205_205133

noncomputable def f (x : ℝ) : ℝ := (x - 1) * Real.exp x

noncomputable def f' (x : ℝ) : ℝ := (x : ℝ) * Real.exp x

theorem tangent_line_eq (x : ℝ) (h : x = 0) : 
  ∃ (c : ℝ), (1 : ℝ) = 1 ∧ f x = c ∧ f' x = 0 ∧ (∀ y, y = c) :=
by
  sorry

end tangent_line_eq_l205_205133


namespace evaluate_expression_l205_205821

theorem evaluate_expression :
  (24^36) / (72^18) = 8^18 :=
by
  sorry

end evaluate_expression_l205_205821


namespace solve_system_l205_205716

section system_equations

variable (x y : ℤ)

def equation1 := 2 * x - y = 5
def equation2 := 5 * x + 2 * y = 8
def solution := x = 2 ∧ y = -1

theorem solve_system : (equation1 x y) ∧ (equation2 x y) ↔ solution x y := by
  sorry

end system_equations

end solve_system_l205_205716


namespace percentage_increase_l205_205223

theorem percentage_increase (C S : ℝ) (h1 : S = 4.2 * C) 
  (h2 : ∃ X : ℝ, (S - (C + (X / 100) * C) = (2 / 3) * S)) : 
  ∃ X : ℝ, (C + (X / 100) * C - C)/(C) = 40 / 100 := 
by
  sorry

end percentage_increase_l205_205223


namespace integer_solutions_count_l205_205001

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l205_205001


namespace polygon_sides_l205_205400

theorem polygon_sides (n : ℕ) (h1 : ∀ i < n, (n > 2) → (150 * n = (n - 2) * 180)) : n = 12 :=
by
  -- Proof omitted
  sorry

end polygon_sides_l205_205400


namespace fraction_in_orange_tin_l205_205484

variables {C : ℕ} -- assume total number of cookies as a natural number

theorem fraction_in_orange_tin (h1 : 11 / 12 = (1 / 6) + (5 / 12) + w)
  (h2 : 1 - (11 / 12) = 1 / 12) :
  w = 1 / 3 :=
by
  sorry

end fraction_in_orange_tin_l205_205484


namespace arc_length_of_pentagon_side_l205_205166

theorem arc_length_of_pentagon_side 
  (r : ℝ) (h : r = 4) :
  (2 * r * Real.pi * (72 / 360)) = (8 * Real.pi / 5) :=
by
  sorry

end arc_length_of_pentagon_side_l205_205166


namespace average_children_l205_205870

theorem average_children (total_families : ℕ) (avg_children_all : ℕ) 
  (childless_families : ℕ) (total_children : ℕ) (families_with_children : ℕ) : 
  total_families = 15 →
  avg_children_all = 3 →
  childless_families = 3 →
  total_children = total_families * avg_children_all →
  families_with_children = total_families - childless_families →
  (total_children / families_with_children : ℚ) = 3.8 :=
by
  intros
  sorry

end average_children_l205_205870


namespace kanul_cash_percentage_l205_205692

-- Define the conditions
def raw_materials_cost : ℝ := 3000
def machinery_cost : ℝ := 1000
def total_amount : ℝ := 5714.29
def total_spent := raw_materials_cost + machinery_cost
def cash := total_amount - total_spent

-- The goal is to prove the percentage of the total amount as cash is 30%
theorem kanul_cash_percentage :
  (cash / total_amount) * 100 = 30 := 
sorry

end kanul_cash_percentage_l205_205692


namespace P_eq_Q_l205_205694

open Set Real

def P : Set ℝ := {m | -1 < m ∧ m ≤ 0}
def Q : Set ℝ := {m | ∀ (x : ℝ), m * x^2 + 4 * m * x - 4 < 0}

theorem P_eq_Q : P = Q :=
by
  sorry

end P_eq_Q_l205_205694


namespace find_cos_sin_sum_l205_205656

-- Define the given condition: tan θ = 5/12 and 180° ≤ θ ≤ 270°.
variable (θ : ℝ)
variable (h₁ : Real.tan θ = 5 / 12)
variable (h₂ : π ≤ θ ∧ θ ≤ 3 * π / 2)

-- Define the main statement to prove.
theorem find_cos_sin_sum : Real.cos θ + Real.sin θ = -17 / 13 := by
  sorry

end find_cos_sin_sum_l205_205656


namespace integral_exp_integral_exp_example_l205_205729

theorem integral_exp (f : ℝ → ℝ) (a b : ℝ) (h_f : ∀ x, f x = exp x) :
  ∫ x in a..b, f x = exp 1 - 1 :=
by
  rw h_f
  exact integral_exp 0 1

# for the purpose of creating a proof that can be built successfully, we use 'sorry' to skip the proofs
theorem integral_exp_example : ∫ x in 0..1, exp x = exp 1 - 1 := 
by
  rw integral_exp
  sorry

end integral_exp_integral_exp_example_l205_205729


namespace initial_clothing_count_l205_205497

theorem initial_clothing_count 
  (donated_first : ℕ) 
  (donated_second : ℕ) 
  (thrown_away : ℕ) 
  (remaining : ℕ) 
  (h1 : donated_first = 5) 
  (h2 : donated_second = 3 * donated_first) 
  (h3 : thrown_away = 15) 
  (h4 : remaining = 65) :
  donated_first + donated_second + thrown_away + remaining = 100 :=
by
  sorry

end initial_clothing_count_l205_205497


namespace contradiction_of_distinct_roots_l205_205201

theorem contradiction_of_distinct_roots
  (a b c : ℝ)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hc : c ≠ 0)
  (distinct_abc : a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (H : ¬ (∃ x1 x2, x1 ≠ x2 ∧ (a * x1^2 + 2 * b * x1 + c = 0 ∨ b * x1^2 + 2 * c * x1 + a = 0 ∨ c * x1^2 + 2 * a * x1 + b = 0))) :
  False := 
sorry

end contradiction_of_distinct_roots_l205_205201


namespace isosceles_right_triangle_side_length_l205_205762

theorem isosceles_right_triangle_side_length
  (a b : ℝ)
  (h_triangle : a = b ∨ b = a)
  (h_hypotenuse : xy > yz)
  (h_area : (1 / 2) * a * b = 9) :
  xy = 6 :=
by
  -- proof will go here
  sorry

end isosceles_right_triangle_side_length_l205_205762


namespace ratio_sum_eq_l205_205574

variable {x y z : ℝ}

-- Conditions: 3x, 4y, 5z form a geometric sequence
def geom_sequence (x y z : ℝ) : Prop :=
  (∃ r : ℝ, 4 * y = 3 * x * r ∧ 5 * z = 4 * y * r)

-- Conditions: 1/x, 1/y, 1/z form an arithmetic sequence
def arith_sequence (x y z : ℝ) : Prop :=
  2 * x * z = y * z + x * y

-- Conclude: x/z + z/x = 34/15
theorem ratio_sum_eq (h1 : geom_sequence x y z) (h2 : arith_sequence x y z) : 
  (x / z + z / x) = (34 / 15) :=
sorry

end ratio_sum_eq_l205_205574


namespace union_M_N_eq_N_l205_205948

def M := {x : ℝ | x^2 - 2 * x ≤ 0}
def N := {x : ℝ | -1 ≤ x ∧ x ≤ 2}

theorem union_M_N_eq_N : M ∪ N = N := 
sorry

end union_M_N_eq_N_l205_205948


namespace problem1_problem2_l205_205155

theorem problem1 : (Real.sqrt 18 - Real.sqrt 8 - Real.sqrt 2) = 0 := 
by sorry

theorem problem2 : (6 * Real.sqrt 2 * Real.sqrt 3 + 3 * Real.sqrt 30 / Real.sqrt 5) = 9 * Real.sqrt 6 := 
by sorry

end problem1_problem2_l205_205155


namespace integer_solutions_count_l205_205004

theorem integer_solutions_count : 
  ∃ n, n = 3 ∧ ∀ x : ℤ, (4 < Real.sqrt (3 * x) ∧ Real.sqrt (3 * x) < 5) ↔ (x = 6 ∨ x = 7 ∨ x = 8) := by
  sorry

end integer_solutions_count_l205_205004


namespace route_B_is_quicker_l205_205702

theorem route_B_is_quicker : 
    let distance_A := 6 -- miles
    let speed_A := 30 -- mph
    let distance_B_total := 5 -- miles
    let distance_B_non_school := 4.5 -- miles
    let speed_B_non_school := 40 -- mph
    let distance_B_school := 0.5 -- miles
    let speed_B_school := 20 -- mph
    let time_A := (distance_A / speed_A) * 60 -- minutes
    let time_B_non_school := (distance_B_non_school / speed_B_non_school) * 60 -- minutes
    let time_B_school := (distance_B_school / speed_B_school) * 60 -- minutes
    let time_B := time_B_non_school + time_B_school -- minutes
    let time_difference := time_A - time_B -- minutes
    time_difference = 3.75 :=
sorry

end route_B_is_quicker_l205_205702


namespace all_children_receive_candy_l205_205053

-- Define f(x) function
def f (x n : ℕ) : ℕ := ((x * (x + 1)) / 2) % n

-- Define the problem statement: prove that all children receive at least one candy if n is a power of 2.
theorem all_children_receive_candy (n : ℕ) (h : ∃ m, n = 2^m) : 
    ∀ i : ℕ, i < n → ∃ x : ℕ, i = f x n := 
sorry

end all_children_receive_candy_l205_205053


namespace april_roses_l205_205306

theorem april_roses (R : ℕ) (h1 : 7 * (R - 4) = 35) : R = 9 :=
sorry

end april_roses_l205_205306


namespace second_eq_value_l205_205550

variable (x y z w : ℝ)

theorem second_eq_value (h1 : 4 * x * z + y * w = 3) (h2 : (2 * x + y) * (2 * z + w) = 15) : 
  x * w + y * z = 6 :=
by
  sorry

end second_eq_value_l205_205550


namespace avg_children_in_families_with_children_l205_205877

-- Define the conditions
def num_families : ℕ := 15
def avg_children_per_family : ℤ := 3
def num_childless_families : ℕ := 3

-- Total number of children among all families
def total_children : ℤ := num_families * avg_children_per_family

-- Number of families with children
def num_families_with_children : ℕ := num_families - num_childless_families

-- Average number of children in families with children, to be proven equal 3.8 when rounded to the nearest tenth.
theorem avg_children_in_families_with_children : (total_children : ℚ) / num_families_with_children = 3.8 := by
  -- Proof is omitted
  sorry

end avg_children_in_families_with_children_l205_205877


namespace max_articles_produced_l205_205676

variables (a b c d p q r s z : ℝ)
variables (h1 : d = (a^2 * b * c) / z)
variables (h2 : p * q * r ≤ s)

theorem max_articles_produced : 
  p * q * r * (a / z) = s * (a / z) :=
by
  sorry

end max_articles_produced_l205_205676


namespace num_packages_l205_205249

theorem num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) (h1 : total_shirts = 51) (h2 : shirts_per_package = 3) : total_shirts / shirts_per_package = 17 := by
  sorry

end num_packages_l205_205249


namespace find_3a_plus_4b_l205_205641

noncomputable def g (x : ℝ) := 3 * x - 6

noncomputable def f_inverse (x : ℝ) := (3 * x - 2) / 2

noncomputable def f (x : ℝ) (a b : ℝ) := a * x + b

theorem find_3a_plus_4b (a b : ℝ) (h1 : ∀ x, g x = 2 * f_inverse x - 4) (h2 : ∀ x, f_inverse (f x a b) = x) :
  3 * a + 4 * b = 14 / 3 :=
sorry

end find_3a_plus_4b_l205_205641


namespace fraction_value_l205_205069

theorem fraction_value
  (x y z : ℝ)
  (h1 : x / 2 = y / 3)
  (h2 : y / 3 = z / 5)
  (h3 : 2 * x + y ≠ 0) :
  (x + y - 3 * z) / (2 * x + y) = -10 / 7 := by
  -- Add sorry to skip the proof.
  sorry

end fraction_value_l205_205069


namespace initial_salty_cookies_l205_205707

theorem initial_salty_cookies
  (initial_sweet_cookies : ℕ) 
  (ate_sweet_cookies : ℕ) 
  (ate_salty_cookies : ℕ) 
  (ate_diff : ℕ) 
  (H1 : initial_sweet_cookies = 39)
  (H2 : ate_sweet_cookies = 32)
  (H3 : ate_salty_cookies = 23)
  (H4 : ate_diff = 9) :
  initial_sweet_cookies - ate_diff = 30 :=
by sorry

end initial_salty_cookies_l205_205707


namespace initial_games_l205_205922

-- Conditions
def games_given_away : ℕ := 7
def games_left : ℕ := 91

-- Theorem Statement
theorem initial_games (initial_games : ℕ) : 
  initial_games = games_left + games_given_away :=
by
  sorry

end initial_games_l205_205922


namespace roundTripAverageSpeed_l205_205163

noncomputable def averageSpeed (distAB distBC speedAB speedBC speedCB totalTime : ℝ) : ℝ :=
  let timeAB := distAB / speedAB
  let timeBC := distBC / speedBC
  let timeCB := distBC / speedCB
  let timeBA := totalTime - (timeAB + timeBC + timeCB)
  let totalDistance := 2 * (distAB + distBC)
  totalDistance / totalTime

theorem roundTripAverageSpeed :
  averageSpeed 150 230 80 88 100 9 = 84.44 :=
by
  -- The actual proof will go here, which is not required for this task.
  sorry

end roundTripAverageSpeed_l205_205163


namespace avg_children_in_families_with_children_l205_205832

theorem avg_children_in_families_with_children (total_families : ℕ) (average_children_per_family : ℕ) (childless_families : ℕ) :
  total_families = 15 →
  average_children_per_family = 3 →
  childless_families = 3 →
  (45 / (total_families - childless_families) : ℝ) = 3.8 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end avg_children_in_families_with_children_l205_205832


namespace translated_parabola_eq_l205_205276

-- Define the original parabola equation
def original_parabola (x : ℝ) : ℝ := 3 * x^2

-- Function to translate a parabola equation downward by a units
def translate_downward (f : ℝ → ℝ) (a : ℝ) (x : ℝ) : ℝ := f x - a

-- Function to translate a parabola equation rightward by b units
def translate_rightward (f : ℝ → ℝ) (b : ℝ) (x : ℝ) : ℝ := f (x - b)

-- The new parabola equation after translating the given parabola downward by 3 units and rightward by 2 units
def new_parabola (x : ℝ) : ℝ := 3 * x^2 - 12 * x + 9

-- The main theorem stating that translating the original parabola downward by 3 units and rightward by 2 units results in the new parabola equation
theorem translated_parabola_eq :
  ∀ x : ℝ, translate_rightward (translate_downward original_parabola 3) 2 x = new_parabola x :=
by
  sorry

end translated_parabola_eq_l205_205276


namespace songs_after_operations_l205_205477

-- Definitions based on conditions
def initialSongs : ℕ := 15
def deletedSongs : ℕ := 8
def addedSongs : ℕ := 50

-- Problem statement to be proved
theorem songs_after_operations : initialSongs - deletedSongs + addedSongs = 57 :=
by
  sorry

end songs_after_operations_l205_205477


namespace AB_eq_B_exp_V_l205_205892

theorem AB_eq_B_exp_V : 
  ∀ A B V : ℕ, 
    (A ≠ B) ∧ (B ≠ V) ∧ (A ≠ V) ∧ (B < 10 ∧ A < 10 ∧ V < 10) →
    (AB = 10 * A + B) →
    (AB = B^V) →
    (AB = 36 ∨ AB = 64 ∨ AB = 32) :=
by
  sorry

end AB_eq_B_exp_V_l205_205892


namespace rhombus_area_l205_205614

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 25) (h_d2 : d2 = 50) :
  (d1 * d2) / 2 = 625 := 
by
  sorry

end rhombus_area_l205_205614


namespace find_f_inv_486_l205_205257

open Function

noncomputable def f (x : ℕ) : ℕ := sorry -- placeholder for function definition

axiom f_condition1 : f 5 = 2
axiom f_condition2 : ∀ (x : ℕ), f (3 * x) = 3 * f x

theorem find_f_inv_486 : f⁻¹' {486} = {1215} := sorry

end find_f_inv_486_l205_205257


namespace negation_of_p_l205_205667

-- Given conditions
def p : Prop := ∃ x : ℝ, x^2 + 3 * x = 4

-- The proof problem to be solved 
theorem negation_of_p : ¬p ↔ ∀ x : ℝ, x^2 + 3 * x ≠ 4 := by
  sorry

end negation_of_p_l205_205667


namespace tangent_line_exponential_passing_through_origin_l205_205524

theorem tangent_line_exponential_passing_through_origin :
  ∃ (p : ℝ × ℝ) (m : ℝ), 
  (p = (1, Real.exp 1)) ∧ (m = Real.exp 1) ∧ 
  (∀ x : ℝ, x ≠ 1 → ¬ (∃ k : ℝ, k = (Real.exp x - 0) / (x - 0) ∧ k = Real.exp x)) :=
by 
  sorry

end tangent_line_exponential_passing_through_origin_l205_205524


namespace smallest_prime_divisor_of_sum_first_100_is_5_l205_205816

-- Conditions: The sum of the first 100 natural numbers
def sum_first_n_numbers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

-- Prime checking function to identify the smallest prime divisor
def smallest_prime_divisor (n : ℕ) : ℕ :=
  if n % 2 = 0 then 2 else
  if n % 3 = 0 then 3 else
  if n % 5 = 0 then 5 else
  n -- Such a simplification works because we know the answer must be within the first few primes.

-- Proof statement
theorem smallest_prime_divisor_of_sum_first_100_is_5 : smallest_prime_divisor (sum_first_n_numbers 100) = 5 :=
by
  -- Proof steps would follow here.
  sorry

end smallest_prime_divisor_of_sum_first_100_is_5_l205_205816


namespace fourth_friend_payment_l205_205533

theorem fourth_friend_payment (a b c d : ℕ) 
  (h1 : a = (1 / 3) * (b + c + d)) 
  (h2 : b = (1 / 4) * (a + c + d)) 
  (h3 : c = (1 / 5) * (a + b + d))
  (h4 : a + b + c + d = 84) : 
  d = 40 := by
sorry

end fourth_friend_payment_l205_205533


namespace find_angle_A_find_bc_range_l205_205092

noncomputable def triangle_problem (a b c : ℝ) (A B C : ℝ) : Prop :=
  (c * (a * Real.cos B - (1/2) * b) = a^2 - b^2) ∧ (A = Real.arccos (1/2))

theorem find_angle_A (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) :
  A = Real.pi / 3 := 
sorry

theorem find_bc_range (a b c : ℝ) (A B C : ℝ) (h : triangle_problem a b c A B C) (ha : a = Real.sqrt 3) :
  b + c ∈ Set.Icc (Real.sqrt 3) (2 * Real.sqrt 3) := 
sorry

end find_angle_A_find_bc_range_l205_205092


namespace total_points_sum_l205_205170

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 8
  else if n % 2 = 0 then 3
  else 0

def allie_rolls := [6, 2, 5, 3, 4]
def carlos_rolls := [3, 2, 2, 6, 1]

def score (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem total_points_sum :
  score allie_rolls + score carlos_rolls = 44 :=
by
  sorry

end total_points_sum_l205_205170


namespace number_of_indeterminate_conditions_l205_205498

noncomputable def angle_sum (A B C : ℝ) : Prop := A + B + C = 180
noncomputable def condition1 (A B C : ℝ) : Prop := A + B = C
noncomputable def condition2 (A B C : ℝ) : Prop := A = C / 6 ∧ B = 2 * (C / 6)
noncomputable def condition3 (A B : ℝ) : Prop := A = 90 - B
noncomputable def condition4 (A B C : ℝ) : Prop := A = B ∧ B = C
noncomputable def condition5 (A B C : ℝ) : Prop := 2 * A = C ∧ 2 * B = C
noncomputable def is_right_triangle (C : ℝ) : Prop := C = 90

theorem number_of_indeterminate_conditions (A B C : ℝ) :
  (angle_sum A B C) →
  (condition1 A B C → is_right_triangle C) →
  (condition2 A B C → is_right_triangle C) →
  (condition3 A B → is_right_triangle C) →
  (condition4 A B C → ¬ is_right_triangle C) →
  (condition5 A B C → is_right_triangle C) →
  ∃ n, n = 1 :=
sorry

end number_of_indeterminate_conditions_l205_205498


namespace exists_p_l205_205236

variable {M : Set ℤ} (hM : Set.Finite M) (zero_in_M : 0 ∈ M)
variable {f g : ℤ → ℤ} 
  (hf : ∀ x y, x ∈ M → y ∈ M → x ≤ y → f(x) ≥ f(y)) 
  (hg : ∀ x y, x ∈ M → y ∈ M → x ≤ y → g(x) ≥ g(y)) 
  (h_gf0 : g(f(0)) ≥ 0)

theorem exists_p (M : Set ℤ) [hM_finite : Set.Finite M] (zero_in_M : 0 ∈ M) 
  (f g : ℤ → ℤ) (hf : ∀ x y, x ∈ M → y ∈ M → x ≤ y → f(x) ≥ f(y)) 
  (hg : ∀ x y, x ∈ M → y ∈ M → x ≤ y → g(x) ≥ g(y))
  (h_gf0 : g(f(0)) ≥ 0) : 
  ∃ p ∈ M, g(f(p)) = p := 
by
  sorry

end exists_p_l205_205236


namespace range_of_a_l205_205241

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → ax^2 - 2 * x + 2 > 0) ↔ (a > 1/2) :=
sorry

end range_of_a_l205_205241


namespace rotation_matrix_150_l205_205363

def rotation_matrix (theta : ℝ) : Matrix ℝ := 
  ![
    ![Real.cos theta, -Real.sin theta], 
    ![Real.sin theta, Real.cos theta]
  ]

theorem rotation_matrix_150 :
  rotation_matrix (5 * Real.pi / 6) = 
  ![
    ![-Real.sqrt 3 / 2, -1 / 2], 
    ![1 / 2, -Real.sqrt 3 / 2]
  ] := by
  sorry

end rotation_matrix_150_l205_205363


namespace housewife_spent_on_oil_l205_205165

-- Define the conditions
variables (P A : ℝ)
variables (h_price_reduced : 0.7 * P = 70)
variables (h_more_oil : A / 70 = A / P + 3)

-- Define the theorem to be proven
theorem housewife_spent_on_oil : A = 700 :=
by
  sorry

end housewife_spent_on_oil_l205_205165


namespace mail_in_six_months_l205_205452

/-- The post office receives 60 letters and 20 packages per day. Each month has 30 days. -/
def daily_letters := 60
def daily_packages := 20
def days_per_month := 30
def months := 6

/-- Prove that the post office handles 14400 pieces of mail in six months. -/
theorem mail_in_six_months : (daily_letters + daily_packages) * days_per_month * months = 14400 :=
by
  sorry

end mail_in_six_months_l205_205452


namespace paint_intensity_change_l205_205594

theorem paint_intensity_change (intensity_original : ℝ) (intensity_new : ℝ) (fraction_replaced : ℝ) 
  (h1 : intensity_original = 0.40) (h2 : intensity_new = 0.20) (h3 : fraction_replaced = 1) :
  intensity_new = 0.20 :=
by
  sorry

end paint_intensity_change_l205_205594


namespace mrs_hilt_total_distance_l205_205427

-- Define the distances and number of trips
def distance_to_water_fountain := 30
def distance_to_staff_lounge := 45
def trips_to_water_fountain := 4
def trips_to_staff_lounge := 3

-- Calculate the total distance for Mrs. Hilt's trips
def total_distance := (distance_to_water_fountain * 2 * trips_to_water_fountain) + 
                      (distance_to_staff_lounge * 2 * trips_to_staff_lounge)
                      
theorem mrs_hilt_total_distance : total_distance = 510 := 
by
  sorry

end mrs_hilt_total_distance_l205_205427


namespace train_length_l205_205994

theorem train_length (L : ℕ) (V : ℕ) (platform_length : ℕ) (time_pole : ℕ) (time_platform : ℕ) 
    (h1 : V = L / time_pole) 
    (h2 : V = (L + platform_length) / time_platform) :
    L = 300 := 
by 
  -- The proof can be filled here
  sorry

end train_length_l205_205994


namespace cube_cut_problem_l205_205998

theorem cube_cut_problem (n s : ℕ) (h1 : n^3 - s^3 = 152) (h2 : ∀ i, i = 1 ∨ i = s)
  (h3 : s * s * s ≤ n * n * n) (h4 : n > 1) : n = 6 :=
by sorry

end cube_cut_problem_l205_205998


namespace binom_18_4_l205_205796

theorem binom_18_4 : Nat.choose 18 4 = 3060 :=
by
  sorry

end binom_18_4_l205_205796


namespace negation_of_p_is_universal_l205_205082

-- Define the proposition p
def p : Prop := ∃ x : ℝ, Real.exp x - x - 1 ≤ 0

-- The proof statement for the negation of p
theorem negation_of_p_is_universal : ¬p ↔ ∀ x : ℝ, Real.exp x - x - 1 > 0 :=
by sorry

end negation_of_p_is_universal_l205_205082


namespace minimum_value_l205_205540

theorem minimum_value (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  ∃ (y : ℝ), y = (c / (a + b)) + (b / c) ∧ y ≥ (Real.sqrt 2) - (1 / 2) :=
sorry

end minimum_value_l205_205540


namespace binom_18_4_l205_205809

theorem binom_18_4 : Nat.binomial 18 4 = 3060 :=
by
  -- We start the proof here.
  sorry

end binom_18_4_l205_205809


namespace bucket_capacity_l205_205156

theorem bucket_capacity (x : ℕ) (h₁ : 12 * x = 132 * 5) : x = 55 := by
  sorry

end bucket_capacity_l205_205156
