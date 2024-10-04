import Mathlib

namespace line_forms_equivalence_l339_339537

noncomputable def points (P Q : ℝ × ℝ) : Prop := 
  ∃ m c, ∃ b d, P = (b, m * b + c) ∧ Q = (d, m * d + c)

theorem line_forms_equivalence :
  points (-2, 3) (4, -1) →
  (∀ x y : ℝ, (y + 1) / (3 + 1) = (x - 4) / (-2 - 4)) ∧
  (∀ x y : ℝ, y + 1 = - (2 / 3) * (x - 4)) ∧
  (∀ x y : ℝ, y = - (2 / 3) * x + 5 / 3) ∧
  (∀ x y : ℝ, x / (5 / 2) + y / (5 / 3) = 1) :=
  sorry

end line_forms_equivalence_l339_339537


namespace range_of_a_if_proposition_l339_339971

theorem range_of_a_if_proposition :
  (∃ x : ℝ, |x - 1| + |x + a| < 3) → -4 < a ∧ a < 2 := by
  sorry

end range_of_a_if_proposition_l339_339971


namespace gcd_three_numbers_l339_339004

theorem gcd_three_numbers (a b c : ℕ) (h₁ : a = 13847) (h₂ : b = 21353) (h₃ : c = 34691) : Nat.gcd (Nat.gcd a b) c = 5 := by sorry

end gcd_three_numbers_l339_339004


namespace range_of_m_l339_339934

theorem range_of_m (m : ℝ) :
  (∃ x : ℝ, x > 0 ∧ (2 * x - m) / (x - 3) - 1 = x / (3 - x)) →
  m > 3 ∧ m ≠ 9 :=
by
  sorry

end range_of_m_l339_339934


namespace number_of_integers_between_sqrts_l339_339131

theorem number_of_integers_between_sqrts :
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  (upper_bound - lower_bound + 1) = 6 :=
by
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  have sqrt8_bound : (2 : ℝ) < sqrt8 ∧ sqrt8 < 3 := sorry
  have sqrt75_bound : (8 : ℝ) < sqrt75 ∧ sqrt75 < 9 := sorry
  have lower_bound_def : lower_bound = 3 := sorry
  have upper_bound_def : upper_bound = 8 := sorry
  show (upper_bound - lower_bound + 1) = 6
  calc
    upper_bound - lower_bound + 1 = 8 - 3 + 1 := by rw [upper_bound_def, lower_bound_def]
                             ... = 6 := by norm_num
  sorry

end number_of_integers_between_sqrts_l339_339131


namespace flower_pot_cost_difference_l339_339236

theorem flower_pot_cost_difference :
  ∃ d : ℝ, 
    (∑ i in (finset.range 6), (1.75 - i * d)) = 8.25 ∧ 
    d = 0.15 :=
by
  have h : ∑ i in (finset.range 6), i = (finset.range 6).sum id := by sorry
  have hsum : (∑ i in (finset.range 6), (1.75 - i * 0.15)) = 1.75 * 6 - 0.15 * (finset.range 6).sum id := by sorry
  use 0.15
  split
  · rw [hsum, h]
    -- We calculate (1.75 * 6) - 0.15 * (0 + 1 + 2 + 3 + 4 + 5)
    -- which simplifies the total cost
    sorry
  · exact rfl

end flower_pot_cost_difference_l339_339236


namespace smallest_k_divides_l339_339049

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l339_339049


namespace opposite_event_l339_339558

-- Given conditions
def at_least_one_defective (items : List Bool) : Prop :=
  ∃ item ∈ items, item = true

def at_most_zero_defective (items : List Bool) : Prop :=
  ∀ item ∈ items, item = false

-- Required proof statement
theorem opposite_event (items : List Bool) (h : length items = 2) :
  (¬ at_least_one_defective items) ↔ at_most_zero_defective items :=
by
  sorry

end opposite_event_l339_339558


namespace fixed_point_for_line_l339_339518

theorem fixed_point_for_line (m : ℝ) : (m * (1 - 1) + (1 - 1) = 0) :=
by
  sorry

end fixed_point_for_line_l339_339518


namespace area_of_isosceles_right_triangle_l339_339776

theorem area_of_isosceles_right_triangle (A B C : Type) [EuclideanGeometry A] 
  (h : triangle A B C) (h_isosceles_right : isIsoscelesRightTriangle A B C) 
  (angle_A_90 : angle A = 90) (AC_len : length (between C A) = 6) :
  area (triangle A B C) = 18 :=
sorry

end area_of_isosceles_right_triangle_l339_339776


namespace volume_of_tetrahedron_part_PABC_l339_339196

def point3D := (ℝ × ℝ × ℝ)

def isPoint (x y z : ℝ) : point3D := (x, y, z)

noncomputable def volume_of_tetrahedron_part (P A B C : point3D) : ℝ :=
  if (∀ x y z, (x, y, z) ∈ {P, A, B, C} -> x^2 + y^2 ≥ 1) then 4 * Real.sqrt 3 - 2 * Real.pi
  else 0

-- The main theorem to be proven
theorem volume_of_tetrahedron_part_PABC :
  let P : point3D := (0, 0, 2)
  let A : point3D := (0, 2, 0)
  let B : point3D := (Real.sqrt 3, -1, 0)
  let C : point3D := (-Real.sqrt 3, -1, 0)
  volume_of_tetrahedron_part P A B C = 4 * Real.sqrt 3 - 2 * Real.pi
:= by
  sorry

end volume_of_tetrahedron_part_PABC_l339_339196


namespace factorial_division_identity_l339_339506

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339506


namespace min_area_l339_339085

-- Definitions of the points, circle, and the tangency condition
variables {a b : ℝ} (ha : a > 1) (hb : b > 1) (A : ℝ × ℝ) (B : ℝ × ℝ)
def circle (x y : ℝ) := x^2 + y^2 - 2 * x - 2 * y + 1 = 0
def pointA : ℝ × ℝ := (2 * a, 0)
def pointB : ℝ × ℝ := (0, 2 * b)

-- The line AB is tangent to the circle
def tangent (C : ℝ × ℝ) (radius: ℝ) (line : ℝ → ℝ) := 
  let dist := (C.1 * 2*b + C.2 * 2*a - 2*a*2*b) / real.sqrt (4*a^2 + 4*b^2)
  in dist^2 = radius^2

-- The minimum area calculation
def area_triangle : ℝ := 2 * 2 * (2 + real.sqrt 2)

-- Prove the statement
theorem min_area (h1 : circle 1 1) (h2 : tangent (1, 1) 1 (λ x, -b/a * x + 2*b)) : 
  (∀ (x y : ℝ), triangle_area (0, 0) (2*a, 0) (0, 2*b) = 4 + 2 * real.sqrt 2) := 
  sorry

end min_area_l339_339085


namespace right_triangle_existence_l339_339863

noncomputable def construct_right_triangle (m n h : ℝ) : Prop :=
∃ (A B C H : ℝ × ℝ),
  (let AC := dist A C in
   let BC := dist B C in
   let CH := dist C H in
   (AC⁻¹ * BC⁻¹ = 1) ∧               -- triangle has a right angle at C
   (CH ∈ line A B) ∧                 -- CH is the altitude dropped to the hypotenuse
   (BC / AC = m / n) ∧               -- the ratio of the legs is m : n
   (CH = h))                         -- the altitude CH = h

theorem right_triangle_existence (m n h : ℝ) :
  construct_right_triangle m n h :=
sorry

end right_triangle_existence_l339_339863


namespace signage_painter_earns_l339_339823

-- Define the conditions
def east_side_addresses : list ℕ := list.map (λ n, 6 + n * 8) (list.range 25)
def west_side_addresses : list ℕ := list.map (λ n, 5 + n * 7) (list.range 25)

def digit_count (n : ℕ) : ℕ := (n + 9).toString.length

def total_cost (addresses : list ℕ) : ℕ :=
  2 * addresses.sum (digit_count)

-- Define the main theorem
theorem signage_painter_earns :
  total_cost east_side_addresses + total_cost west_side_addresses = 340 :=
by
  -- The proof is omitted
  sorry

end signage_painter_earns_l339_339823


namespace smallest_k_divides_l339_339043

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l339_339043


namespace find_ab_l339_339780

theorem find_ab (a b : ℝ) (h₁ : a + b = 8) (h₂ : a^4 + b^4 = 272) :
  ab (a b : ℝ) ≈ 17.26 :=
sorry

end find_ab_l339_339780


namespace book_cost_l339_339252

-- Define the problem parameters
variable (p : ℝ) -- cost of one book in dollars

-- Conditions given in the problem
def seven_copies_cost_less_than_15 (p : ℝ) : Prop := 7 * p < 15
def eleven_copies_cost_more_than_22 (p : ℝ) : Prop := 11 * p > 22

-- The theorem stating the cost is between the given bounds
theorem book_cost (p : ℝ) (h1 : seven_copies_cost_less_than_15 p) (h2 : eleven_copies_cost_more_than_22 p) : 
    2 < p ∧ p < (15 / 7 : ℝ) :=
sorry

end book_cost_l339_339252


namespace polynomial_divisibility_l339_339352

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end polynomial_divisibility_l339_339352


namespace stick_cut_into_pieces_l339_339715

-- Define the problem conditions
def divides (n m : ℕ) : Prop := ∃ k : ℕ, m = n * k

-- Define total scale lines for each type of division
def scale_lines (div1 div2 : ℕ) : ℕ := 
  div1 + div2 - (if divides 12 div2 then 1 else 0) - 1

-- Define the main theorem to be proved
theorem stick_cut_into_pieces : ∀ (div1 div2 : ℕ),
  div1 = 12 → div2 = 18 → scale_lines div1 div2 + 1 = 24 :=
by
  intros div1 div2 h_div1 h_div2
  rw [h_div1, h_div2]
  dsimp [scale_lines]
  rw [if_pos]
  norm_num
  exact ⟨3, rfl⟩  -- proving that 18 is divisible by 12 because 18 = 12 * 1.5
  sorry

-- Ensure the above theorem is standalone and concise

end stick_cut_into_pieces_l339_339715


namespace polynomial_division_quotient_l339_339543

noncomputable def P (x : ℝ) := 8 * x^3 + 5 * x^2 - 4 * x - 7
noncomputable def D (x : ℝ) := x + 3

theorem polynomial_division_quotient :
  ∀ x : ℝ, (P x) / (D x) = 8 * x^2 - 19 * x + 53 := sorry

end polynomial_division_quotient_l339_339543


namespace age_when_dog_born_l339_339714

theorem age_when_dog_born (current_age : ℕ) (dog_future_age : ℕ) (years_from_now : ℕ) (current_dog_age : ℕ) (age_difference : ℕ) :
  years_from_now = 2 →
  dog_future_age = 4 →
  current_age = 17 →
  current_dog_age = dog_future_age - years_from_now →
  age_difference = current_age - current_dog_age →
  age_difference = 15 :=
begin
  intros,
  sorry
end

end age_when_dog_born_l339_339714


namespace number_of_boys_in_school_l339_339761

theorem number_of_boys_in_school (x g : ℕ) (h1 : x + g = 400) (h2 : g = (x * 400) / 100) : x = 80 :=
by
  sorry

end number_of_boys_in_school_l339_339761


namespace problem_statement_l339_339738

variable (p q : ℤ)
def A : ℤ := 5^p
def B : ℤ := 7^q

theorem problem_statement : 35^(p * q) = A^q * B^p := sorry

end problem_statement_l339_339738


namespace total_bottles_calculation_l339_339331

theorem total_bottles_calculation
  (price_orange : ℝ := 0.70)   -- A bottle of orange juice costs $0.70.
  (price_apple : ℝ := 0.60)    -- A bottle of apple juice costs $0.60.
  (total_cost : ℝ := 46.20)    -- Total cost of all bottles is $46.20.
  (num_orange : ℕ := 42)       -- We bought 42 bottles of orange juice.
  : (num_orange + ((total_cost - (num_orange * price_orange)) / price_apple)) = 70 := 
begin
  sorry
end

end total_bottles_calculation_l339_339331


namespace calc_nabla_l339_339398

noncomputable def op_nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calc_nabla : (op_nabla (op_nabla 2 3) 4) = 11 / 9 :=
by
  unfold op_nabla
  sorry

end calc_nabla_l339_339398


namespace general_term_formula_minimized_l339_339979

noncomputable def general_term_of_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q > 0, (∀ n, a n = a 1 * q^(n-1)) ∧ a(2) - a(1) = 1 ∧
  a(3) = (λ q, q^2 / (q - 1): a(1)q^2) :=
  a 1 = 1 → a 3 = 4

theorem general_term_formula_minimized (a : ℕ → ℝ) :
  general_term_of_geometric_sequence a → ∀ n, a n = 2^(n-1) :=
sorry

end general_term_formula_minimized_l339_339979


namespace range_of_f_l339_339099

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 1

theorem range_of_f :
  set.range (f ∘ (λ x, x)) ⊆ set.Icc (-2 : ℝ) 7 :=
  sorry

end range_of_f_l339_339099


namespace factorial_div_l339_339495

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339495


namespace f_three_eq_four_l339_339059

def f : ℕ → ℕ
| x := if x ≥ 7 then x - 5 else f (x + 3)

theorem f_three_eq_four : f 3 = 4 := by
  sorry

end f_three_eq_four_l339_339059


namespace number_of_integers_between_sqrt8_and_sqrt75_l339_339126

theorem number_of_integers_between_sqrt8_and_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ 
  ∃ x : Fin n.succ, √8 < ↑x ∧ ↑x < √75 := 
by
  sorry

end number_of_integers_between_sqrt8_and_sqrt75_l339_339126


namespace sum_of_powers_of_two_fractions_l339_339000

theorem sum_of_powers_of_two_fractions (n : ℕ) (x : Fin n → ℕ) : 
  (∑ i : Fin n, 2^i / (x i)^2 = 1) ↔ (n = 1 ∨ n ≥ 3) :=
sorry

end sum_of_powers_of_two_fractions_l339_339000


namespace two_digit_number_exists_l339_339384

theorem two_digit_number_exists (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  (9 * x + 8) * (80 - 9 * x) = 1855 → (9 * x + 8 = 35 ∨ 9 * x + 8 = 53) := by
  sorry

end two_digit_number_exists_l339_339384


namespace max_sum_multiplication_table_l339_339754

theorem max_sum_multiplication_table :
  ∃ a b c d e f g h : ℕ,
  {a, b, c, d} ⊆ {2, 3, 5, 7, 11, 13, 17, 19} ∧
  {e, f, g, h} ⊆ {2, 3, 5, 7, 11, 13, 17, 19} ∧
  {a, b, c, d} ∩ {e, f, g, h} = ∅ ∧
  (a + b + c + d) + (e + f + g + h) = 77 ∧
  (a + b + c + d) * (e + f + g + h) = 1480 :=
begin
  sorry
end

end max_sum_multiplication_table_l339_339754


namespace probability_at_least_one_l339_339651

-- Conditions
constant pA : ℝ := 0.8
constant pB : ℝ := 0.7

-- The problem we want to prove
theorem probability_at_least_one :
  let qA := 1 - pA in
  let qB := 1 - pB in
  let p_neither := qA * qB in
  1 - p_neither = 0.94 :=
by simp [pA, pB, (1 - pA) * (1 - pB)]

end probability_at_least_one_l339_339651


namespace circumcircle_passing_fixed_point_l339_339344

noncomputable def acute_triangle (A B C : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] : Prop := 
-- Assuming conditions for an acute triangle
sorry

variable {A B C D P Q K X : Type}

-- Defining point on lines and intersection
def is_point_on_line (P : Type) (BC : Type) : Prop := sorry
def is_intersection (Q : Type) (line1 : Type) (line2 : Type) : Prop := sorry

-- Circumcircles definitions
def is_circumcircle (C P D : Type) : Type := sorry
def passes_through (circ : Type) (point : Type) : Prop := sorry

-- Defining the fixed point condition
def fixed_point (X : Type) (BC : Type) : Type := sorry

theorem circumcircle_passing_fixed_point (A B C D P Q K X : Type)
  [acute_triangle A B C]
  (hD : is_point_on_line D BC)
  (hQ : is_intersection Q (is_point_on_line A D) (is_point_on_line B C))
  (hP_distinct : P ≠ Q)
  (circ_cpd_cq : passes_through (is_circumcircle C P D) C)
  (circ_cpd_ck : passes_through (is_circumcircle C P D) K)
  (circ_akp_ax : passes_through (is_circumcircle A K P) X) :
  fixed_point X BC := sorry

end circumcircle_passing_fixed_point_l339_339344


namespace isaac_journey_time_l339_339705

def travel_time_total (speed : ℝ) (time1 : ℝ) (distance2 : ℝ) (rest_time : ℝ) (distance3 : ℝ) : ℝ :=
  let time2 := distance2 / speed
  let time3 := distance3 / speed
  time1 + time2 * 60 + rest_time + time3 * 60

theorem isaac_journey_time :
  travel_time_total 10 (30 : ℝ) 15 (30 : ℝ) 20 = 270 :=
by
  sorry

end isaac_journey_time_l339_339705


namespace find_amount_l339_339876

theorem find_amount (N : ℝ) (hN : N = 24) (A : ℝ) (hA : A = 0.6667 * N - 0.25 * N) : A = 10.0008 :=
by
  rw [hN] at hA
  sorry

end find_amount_l339_339876


namespace correct_statements_count_l339_339287

/--
Given the following conditions:
1. The opposite of π is -π
2. Numbers with opposite signs are opposite numbers to each other
3. The opposite of -3.8 is 3.8
4. A number and its opposite may be equal
5. Positive numbers and negative numbers are opposite to each other.

Prove that the number of correct statements among these conditions is 3.
-/
theorem correct_statements_count :
  (1 ∧ 3 ∧ 4 : ℕ) = 3 :=
sorry

end correct_statements_count_l339_339287


namespace expression_evaluation_l339_339873

theorem expression_evaluation : 
  2000 * 1995 * 0.1995 - 10 = 0.2 * 1995^2 - 10 := 
by 
  sorry

end expression_evaluation_l339_339873


namespace smallest_committees_exists_l339_339768

structure Senator :=
  (hates : Fin 51 → Fin 3) -- Each senator hates exactly 3 other senators

def num_committee (G : Fin 51 → Senator) : ℕ := 
  -- Function to determine the minimum number of committees
  sorry

theorem smallest_committees_exists : 
  ∃ n, (∀ G : Fin 51 → Senator, ∀ i j : Fin 51, i ≠ j → 
    ¬(i ∈ G j.hates) →
    num_committee G = n) ∧ n = 7 :=
sorry

end smallest_committees_exists_l339_339768


namespace value_of_expression_l339_339337

theorem value_of_expression (x : ℝ) (hx : x = -3) : 3 - x^(-3) = 82 / 27 := by
  rw [hx]
  norm_num
  sorry

end value_of_expression_l339_339337


namespace john_and_lisa_meet_at_midpoint_l339_339212

-- Define the conditions
def john_position : ℝ × ℝ := (2, 9)
def lisa_position : ℝ × ℝ := (-6, 1)

-- Assertion for their meeting point
theorem john_and_lisa_meet_at_midpoint :
  ∃ (x y : ℝ), (x, y) = ((john_position.1 + lisa_position.1) / 2,
                         (john_position.2 + lisa_position.2) / 2) :=
sorry

end john_and_lisa_meet_at_midpoint_l339_339212


namespace range_of_a_l339_339970

theorem range_of_a (a : ℝ)
  (h : ∀ (x y z : ℝ), 0 < x → 0 < y → 0 < z → x + y + z = 1 → |a - 1| ≥ sqrt (3 * x + 1) + sqrt (3 * y + 1) + sqrt (3 * z + 1)) :
  a ∈ Set.Iic (1 - 3 * Real.sqrt 2) ∪ Set.Ici (1 + 3 * Real.sqrt 2) := 
sorry

end range_of_a_l339_339970


namespace pure_imaginary_m_eq_2_l339_339588

theorem pure_imaginary_m_eq_2 (m : ℝ) :
  (∃ (z : ℂ), z = complex.mk (m^2 - 5 * m + 6) (m^2 - 3 * m) ∧ z.im ≠ 0 ∧ z.re = 0) → m = 2 :=
by
  sorry

end pure_imaginary_m_eq_2_l339_339588


namespace part_a_part_b_l339_339666

-- Definition for the function f(P) as sum of distances to given vertices in the lattice
def f (P : LatticePoint) (A : ℕ → LatticePoint) (n : ℕ) : ℕ :=
  (finset.range n).sum (λ i, distance P (A i))

-- Type definition for lattice points (custom definition based on problem context)
structure LatticePoint where
  x : ℤ
  y : ℤ

-- Example distance function (custom definition based on triangular lattice geometry)
def distance (P Q : LatticePoint) : ℕ :=
  sorry -- define the actual metric for the triangular lattice

/-
 Part (a) Statement: Prove that if no neighboring lattice point has a smaller f value,
 then the current point is the global minimum.
-/
theorem part_a (A : ℕ → LatticePoint) (n : ℕ) (P : LatticePoint)
  (h : ∀ Q, Q ≠ P → not (distance P Q = 1) ∨ f Q A n ≥ f P A n) :
  ∀ Q, f Q A n ≥ f P A n :=
sorry

/-
 Part (b) Statement: Provide a counterexample where the algorithm does not find the minimum.
-/
theorem part_b :
  ∃ (G : Graph) (A : ℕ → G.Vertices) (start : G.Vertices),
  ¬(∀ P : G.Vertices, ¬ (f P A 5 < f start A 5)) :=
sorry

end part_a_part_b_l339_339666


namespace number_of_integers_satisfying_inequality_l339_339148

theorem number_of_integers_satisfying_inequality : set.countable {x : ℤ | -6 ≤ 3 * x - 2 ∧ 3 * x - 2 ≤ 9} = 5 := 
sorry

end number_of_integers_satisfying_inequality_l339_339148


namespace steak_original_weight_l339_339213

theorem steak_original_weight (S : ℝ) (h_burned : S / 2 = B) (h_eaten : 0.8 * B = 12) : S = 30 :=
by
  have h1 : B = 15 := by
    rw [h_eaten]
    linarith
  have h2 : S / 2 = 15 := by
    rw [h1]
    exact h_burned
  linarith

end steak_original_weight_l339_339213


namespace bob_spending_over_limit_l339_339718

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end bob_spending_over_limit_l339_339718


namespace parking_cost_l339_339745

theorem parking_cost :
  ∃ (C : ℝ), (∀ (additional_hours_cost : ℝ) (average_cost : ℝ), 
      additional_hours_cost = 1.75 ∧ 
      average_cost = 3.0277777777777777 → 
      (C + 7 * additional_hours_cost) / 9 = average_cost) → 
  C = 15 :=
begin
  use 15,
  intros additional_hours_cost average_cost h,
  cases h with h1 h2,
  rw h1 at *,
  rw h2 at *,
  norm_num,
end

end parking_cost_l339_339745


namespace num_ordered_pairs_num_ordered_pairs_2014_l339_339758

theorem num_ordered_pairs (x y : ℕ) (h : x * y = 2014) : 
  2 * 19 * 53 = 2014 := 
by
  sorry

theorem num_ordered_pairs_2014 : 
  ∃! n, n = 8 ∧ ∀ x y : ℕ, x * y = 2014 → true :=
by
  use 8
  split
  { -- Proof that n = 8
    sorry },
  { -- Proof of uniqueness
    intros n' h'
    have prime_fact_2014 : 2 * 19 * 53 = 2014 := sorry
    have total_divisors : (1+1) * (1+1) * (1+1) = 8 := sorry
    rw h' at prime_fact_2014 total_divisors
    cases h'
    { apply rfl }
    { exact total_divisors } .
  }

end num_ordered_pairs_num_ordered_pairs_2014_l339_339758


namespace eq_x_squared_plus_1_l339_339625

theorem eq_x_squared_plus_1 {x : ℝ} : 3^(2 * x) + 9 = 10 * 3^x → (x^2 + 1 = 1 ∨ x^2 + 1 = 5) :=
by sorry

end eq_x_squared_plus_1_l339_339625


namespace exists_t_perpendicular_min_dot_product_coordinates_l339_339218

-- Definitions of points
def OA : ℝ × ℝ := (5, 1)
def OB : ℝ × ℝ := (1, 7)
def OC : ℝ × ℝ := (4, 2)

-- Definition of vector OM depending on t
def OM (t : ℝ) : ℝ × ℝ := (4 * t, 2 * t)

-- Definition of vector MA and MB
def MA (t : ℝ) : ℝ × ℝ := (5 - 4 * t, 1 - 2 * t)
def MB (t : ℝ) : ℝ × ℝ := (1 - 4 * t, 7 - 2 * t)

-- Dot product of vectors
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Proof that there exists a t such that MA ⊥ MB
theorem exists_t_perpendicular : ∃ t : ℝ, dot_product (MA t) (MB t) = 0 :=
by 
  sorry

-- Proof that coordinates of M minimizing MA ⋅ MB is (4, 2)
theorem min_dot_product_coordinates : ∃ t : ℝ, t = 1 ∧ (OM t) = (4, 2) :=
by
  sorry

end exists_t_perpendicular_min_dot_product_coordinates_l339_339218


namespace smallest_k_l339_339036

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l339_339036


namespace arithmetic_sequence_monotone_l339_339920

theorem arithmetic_sequence_monotone (a : ℕ → ℝ) (d : ℝ) (h_arithmetic : ∀ n, a (n + 1) - a n = d) :
  (a 2 > a 1) ↔ (∀ n, a (n + 1) > a n) :=
by 
  sorry

end arithmetic_sequence_monotone_l339_339920


namespace complement_of_M_l339_339938

def U : Set ℝ := {x | True}
def M : Set ℝ := {x | -3 ≤ x ∧ x < 5}
def complement_M : Set ℝ := {x | x < -3 ∨ x ≥ 5}

theorem complement_of_M : ∀ x : ℝ, (x ∈ U \ M) ↔ (x ∈ complement_M) :=
by
  intro x
  simp [U, M, complement_M, Set.diff, Set.mem_set_of_eq]
  sorry

end complement_of_M_l339_339938


namespace larger_semicircles_area_is_125_percent_larger_l339_339831

-- Define the basic properties of the problem dimensions
def rectangle_width : ℝ := 8
def rectangle_length : ℝ := 12

-- Define the radius of the semicircles attached to each side of the rectangle
def small_semicircle_radius : ℝ := rectangle_width / 2
def large_semicircle_radius : ℝ := rectangle_length / 2

-- Define the areas of the semicircles
def small_semicircle_area : ℝ := 0.5 * (Real.pi * (small_semicircle_radius ^ 2))
def large_semicircle_area : ℝ := 0.5 * (Real.pi * (large_semicircle_radius ^ 2))

-- Define the total area of the semicircles attached to the sides of the rectangle
def total_small_semicircles_area : ℝ := 2 * small_semicircle_area
def total_large_semicircles_area : ℝ := 2 * large_semicircle_area

-- Define the ratio of the total areas and calculate the percent increase
def area_ratio : ℝ := total_large_semicircles_area / total_small_semicircles_area
def percent_increase : ℝ := (area_ratio - 1) * 100

-- The statement to be proven
theorem larger_semicircles_area_is_125_percent_larger :
  percent_increase ≈ 125 :=
sorry

end larger_semicircles_area_is_125_percent_larger_l339_339831


namespace total_travel_time_in_minutes_l339_339712

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l339_339712


namespace range_of_a_l339_339925

variable {α : Type}

def A (x : ℝ) : Prop := 1 ≤ x ∧ x < 5
def B (x a : ℝ) : Prop := -a < x ∧ x ≤ a + 3

theorem range_of_a (a : ℝ) :
  (∀ x, B x a → A x) → a ≤ -1 := by
  sorry

end range_of_a_l339_339925


namespace card_sequence_probability_l339_339315

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l339_339315


namespace dice_sum_not_possible_l339_339781

theorem dice_sum_not_possible (a b c d : ℕ) (h₁ : 1 ≤ a ∧ a ≤ 6) (h₂ : 1 ≤ b ∧ b ≤ 6) 
(h₃ : 1 ≤ c ∧ c ≤ 6) (h₄ : 1 ≤ d ∧ d ≤ 6) (h_product : a * b * c * d = 216) : 
(a + b + c + d ≠ 15) ∧ (a + b + c + d ≠ 16) ∧ (a + b + c + d ≠ 18) :=
sorry

end dice_sum_not_possible_l339_339781


namespace parallelogram_ABCD_MNPQ_l339_339243

noncomputable section

variables {A B C D M N P Q : Type} [InnerProductSpace ℝ A]

def point_on_line (X Y Z : A) : Prop := ∃ k : ℝ, Z = (1 - k) • X + k • Y

variables (A B C D M N P Q : A)

-- Conditions as per step a)
axiom cond1 : point_on_line A B M
axiom cond2 : point_on_line B C N
axiom cond3 : point_on_line C D P
axiom cond4 : point_on_line D A Q
axiom cond5 : dist A M = dist C P
axiom cond6 : dist B N = dist D Q
axiom cond7 : dist B M = dist D P
axiom cond8 : dist N C = dist Q A

theorem parallelogram_ABCD_MNPQ :
  parallelogram A B C D ∧ parallelogram M N P Q := 
sorry

end parallelogram_ABCD_MNPQ_l339_339243


namespace det_AB_product_l339_339628

variable {A B : Matrix}

-- Condition 1: determinant of A is -3
axiom det_A : det A = -3

-- Condition 2: determinant of the inverse of B is 1/4
axiom det_B_inv : det B⁻¹ = 1/4

theorem det_AB_product : det (A ⋅ B) = -12 := by
  sorry

end det_AB_product_l339_339628


namespace no_valid_a_exists_l339_339774

theorem no_valid_a_exists (a : ℕ) (n : ℕ) (h1 : a > 1) (b := a * (10^n + 1)) :
  ¬ (∃ a : ℕ, b % (a^2) = 0) :=
by {
  sorry -- The actual proof is not required as per instructions.
}

end no_valid_a_exists_l339_339774


namespace part1_part2_l339_339574

noncomputable def A (a : ℝ) : set ℝ := {x | a - 1 < x ∧ x < 2 * a + 3 ∧ a > 0}
noncomputable def B : set ℝ := {x | -2 < x ∧ x < 4}

theorem part1 (a : ℝ) (h : a = 2) : A a ∪ B = {x | -2 < x ∧ x < 7} :=
by
  rw h
  sorry

theorem part2 (a : ℝ) : A a ∩ B = ∅ ↔ 5 ≤ a :=
by
  sorry

end part1_part2_l339_339574


namespace additional_grazed_area_correct_l339_339373

noncomputable def additional_grazed_area (r1 r2 : ℝ) : ℝ :=
  π * r2^2 - π * r1^2

theorem additional_grazed_area_correct :
  additional_grazed_area 10 23 = 429 * real.pi :=
by
  unfold additional_grazed_area
  norm_num
  sorry

end additional_grazed_area_correct_l339_339373


namespace farmer_red_milk_production_l339_339528

def bess := 2
def brownie := 3 * bess
def daisy := bess + 1
def ella := 1.5 * daisy
def flossie := (bess + brownie) / 2
def ginger := 2 * ella
def honey := ginger - 1

def daily_total := bess + brownie + daisy + ella + flossie + ginger + honey
def monthly_total := daily_total * 30

theorem farmer_red_milk_production : monthly_total = 1095 := by
  sorry

end farmer_red_milk_production_l339_339528


namespace FO_greater_than_DI_l339_339199

theorem FO_greater_than_DI (F I D O : Type) [ConvexQuadrilateral F I D O]
  (h1 : FI = DO) 
  (h2 : FI > DI) 
  (h3 : ∠FIO = ∠DIO) : 
  FO > DI :=
sorry

end FO_greater_than_DI_l339_339199


namespace range_of_sum_of_zeros_and_sines_l339_339092

noncomputable def function_has_two_distinct_zeros
  (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, 
    (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ 
    (0 ≤ x2 ∧ x2 ≤ π / 2) ∧ 
    (x1 ≠ x2) ∧ 
    (2 * sin (2 * x1 + π / 6) + a - 1 = 0) ∧ 
    (2 * sin (2 * x2 + π / 6) + a - 1 = 0)

theorem range_of_sum_of_zeros_and_sines
  (a : ℝ) (h : function_has_two_distinct_zeros a) :
  ∃ x1 x2 : ℝ,
    (0 ≤ x1 ∧ x1 ≤ π / 2) ∧ 
    (0 ≤ x2 ∧ x2 ≤ π / 2) ∧ 
    (x1 ≠ x2) ∧ 
    (2 * sin (2 * x1 + π / 6) + a - 1 = 0) ∧ 
    (2 * sin (2 * x2 + π / 6) + a - 1 = 0) ∧
    (1 + π / 3 ≤ x1 + x2 + sin (2 * x1 + π / 6) + sin (2 * x2 + π / 6) ∧ 
    x1 + x2 + sin (2 * x1 + π / 6) + sin (2 * x2 + π / 6) < 2 + π / 3) :=
begin
  sorry
end

end range_of_sum_of_zeros_and_sines_l339_339092


namespace calculate_area_E_l339_339725

-- Quadrilateral EFGH and extensions to E', F', G', H' with specified side lengths.
variable (EF F'F FG G'G GH H'H HE E'E : ℝ)
variable (area_EFGH : ℝ)

-- Given conditions
axiom EF_eq_F'F : EF = F'F := by rfl
axiom FG_eq_G'G : FG = G'G := by rfl
axiom GH_eq_H'H : GH = H'H := by rfl
axiom HE_eq_E'E : HE = E'E := by rfl
axiom x_eq_F'G : FG = 6 := by rfl
axiom areaEFGH : area_EFGH = 15 := by rfl

-- The goal
theorem calculate_area_E'F'G'H' :
  let E'F'G'H'_area := (2 * area_EFGH) + (5 * 6 * sqrt 3 / 2) := by sorry
E'F'G'H'_area = 57 :=
begin
  -- Sorry to skip the proof
  sorry
end

end calculate_area_E_l339_339725


namespace factorial_division_l339_339479

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339479


namespace annual_growth_rate_l339_339660

theorem annual_growth_rate (x : ℝ) (h : 2000 * (1 + x) ^ 2 = 2880) : x = 0.2 :=
by sorry

end annual_growth_rate_l339_339660


namespace factorial_div_sum_l339_339420

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339420


namespace area_of_one_trapezoid_l339_339665

-- Given conditions
variable (area_outer_triangle : ℝ)
variable (area_inner_triangle : ℝ)
variable (num_trapezoids : ℕ)

-- These are the conditions as given in the problem
def outer_triangle_area_condition : Prop := area_outer_triangle = 36
def inner_triangle_area_condition : Prop := area_inner_triangle = 4
def num_trapezoids_condition : Prop := num_trapezoids = 3

-- We need to show the area of one trapezoid is 32/3
theorem area_of_one_trapezoid (h1 : outer_triangle_area_condition)
                              (h2 : inner_triangle_area_condition)
                              (h3 : num_trapezoids_condition) :
                              (area_outer_triangle - area_inner_triangle) / num_trapezoids = 32 / 3 :=
by
  sorry

end area_of_one_trapezoid_l339_339665


namespace mn_greater_than_one_l339_339962

theorem mn_greater_than_one
  (M N : ℝ)
  (h₁ : log M (N^2) = log N (M^3))
  (h₂ : M ≠ N)
  (h₃ : M * N > 0)
  (h₄ : M ≠ 1)
  (h₅ : N ≠ 1) :
  M * N > 1 := 
sorry

end mn_greater_than_one_l339_339962


namespace smallest_k_divides_l339_339047

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l339_339047


namespace complex_evaluation_l339_339219

theorem complex_evaluation (a b : ℂ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : a^2 + a * b + b^2 = 0) : 
  (a^9 + b^9) / (a + b)^9 = -2 := 
by 
  sorry

end complex_evaluation_l339_339219


namespace factorial_div_sum_l339_339428

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339428


namespace smallest_k_for_divisibility_l339_339032

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l339_339032


namespace solve_inequality_l339_339866

theorem solve_inequality (x : ℝ) : x^3 - 9*x^2 - 16*x > 0 ↔ (x < -1 ∨ x > 16) := by
  sorry

end solve_inequality_l339_339866


namespace ellipse_axis_lengths_l339_339280

theorem ellipse_axis_lengths : 
  (∀ x y : ℝ, (x^2 / 16 + y^2 / 25 = 1 → 
  ∃ a b : ℝ, 2 * b = 10 ∧ 2 * a = 8)) :=
by
  intros x y h
  use 4
  use 5
  split
  sorry
  sorry

end ellipse_axis_lengths_l339_339280


namespace instantaneous_velocity_at_4_l339_339295

-- Define the motion equation of the object
def motion_eqn (t : ℝ) : ℝ := 1 - t + t^2

-- Define the instantaneous velocity of the object at a given time
def instantaneous_velocity (t : ℝ) : ℝ := (deriv motion_eqn) t

-- Define the specific time t = 4
def specific_time : ℝ := 4

-- Define the expected instantaneous velocity at t = 4
def expected_velocity : ℝ := 7

-- State the theorem to prove that the instantaneous velocity at t=4 seconds is 7 meters/second
theorem instantaneous_velocity_at_4 : instantaneous_velocity specific_time = expected_velocity :=
by
  sorry

end instantaneous_velocity_at_4_l339_339295


namespace amount_left_after_expenses_l339_339749

namespace GirlScouts

def totalEarnings : ℝ := 30
def poolEntryCosts : ℝ :=
  5 * 3.5 + 3 * 2.0 + 2 * 1.0
def transportationCosts : ℝ :=
  6 * 1.5 + 4 * 0.75
def snackCosts : ℝ :=
  3 * 3.0 + 4 * 2.5 + 3 * 2.0
def totalExpenses : ℝ :=
  poolEntryCosts + transportationCosts + snackCosts
def amountLeft : ℝ :=
  totalEarnings - totalExpenses

theorem amount_left_after_expenses :
  amountLeft = -32.5 :=
by
  sorry

end GirlScouts

end amount_left_after_expenses_l339_339749


namespace factorial_division_l339_339487

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339487


namespace factorial_division_l339_339414

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339414


namespace factorial_div_l339_339489

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339489


namespace part_I_part_II_l339_339605

def f (x a : ℝ) := x^3 + a*x^2 + 1

theorem part_I (a b : ℝ) (h_slope : deriv (λ x => f x a) 1 = -3) (h_B : f 1 a = b) :
  a = -3 ∧ b = -1 :=
by
  sorry

theorem part_II (A : ℝ) :
  (∀ x ∈ set.Icc (-1 : ℝ) 4, f x (-3) ≤ A - 1994) ↔ 2011 ≤ A :=
by
  sorry

end part_I_part_II_l339_339605


namespace permutations_of_3377_l339_339109

theorem permutations_of_3377 :
  multiset.card (finset.univ.image (λ (l : list ℕ), l.attach.val)
    .subtype ([3, 3, 7, 7].perm_of_fn _)) = 6 :=
sorry

end permutations_of_3377_l339_339109


namespace leading_coefficient_of_polynomial_l339_339608

theorem leading_coefficient_of_polynomial (f : ℕ → ℚ)
  (h : ∀ x : ℕ, f (x + 1) - f x = 8 * x^2 + 6 * x + 4) :
  polynomial.leadingCoeff (f : polynomial ℚ) = 8 / 3 :=    
sorry

end leading_coefficient_of_polynomial_l339_339608


namespace integer_pairs_solution_l339_339879

theorem integer_pairs_solution (a b : ℤ) : 
  (a - b - 1 ∣ a^2 + b^2 ∧ (a^2 + b^2) * 19 = (2 * a * b - 1) * 20) ↔
  (a, b) = (22, 16) ∨ (a, b) = (-16, -22) ∨ (a, b) = (8, 6) ∨ (a, b) = (-6, -8) :=
by 
  sorry

end integer_pairs_solution_l339_339879


namespace each_girl_gets_2_dollars_l339_339271

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end each_girl_gets_2_dollars_l339_339271


namespace bananas_oranges_equivalence_l339_339267

theorem bananas_oranges_equivalence :
  (3 / 4) * 12 * banana_value = 9 * orange_value →
  (2 / 3) * 6 * banana_value = 4 * orange_value :=
by
  intros h
  sorry

end bananas_oranges_equivalence_l339_339267


namespace most_reasonable_sampling_method_l339_339661

-- Define the conditions
axiom significant_differences_in_educational_stages : Prop
axiom insignificant_differences_between_genders : Prop

-- Define the options
inductive SamplingMethod
| SimpleRandomSampling
| StratifiedSamplingByGender
| StratifiedSamplingByEducationalStage
| SystematicSampling

-- State the problem as a theorem
theorem most_reasonable_sampling_method
  (H1 : significant_differences_in_educational_stages)
  (H2 : insignificant_differences_between_genders) :
  SamplingMethod.StratifiedSamplingByEducationalStage = SamplingMethod.StratifiedSamplingByEducationalStage :=
by
  -- Proof is skipped
  sorry

end most_reasonable_sampling_method_l339_339661


namespace salary_increase_more_than_76_percent_l339_339713

theorem salary_increase_more_than_76_percent (S : ℝ) :
  let increased_salary := S * (1.12 ^ 5) in
  let increase_percent := (increased_salary - S) / S * 100 in
  increase_percent > 76 :=
by
  let increased_salary := S * (1.12 ^ 5)
  let increase_percent := (increased_salary - S) / S * 100
  have h : increase_percent > 76
  sorry

end salary_increase_more_than_76_percent_l339_339713


namespace smallest_k_for_divisibility_l339_339030

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l339_339030


namespace sequence_ratio_l339_339778

theorem sequence_ratio (S T a b : ℕ → ℚ) (h_sum_ratio : ∀ (n : ℕ), S n / T n = (7*n + 2) / (n + 3)) :
  a 7 / b 7 = 93 / 16 :=
by
  sorry

end sequence_ratio_l339_339778


namespace hyperbola_eccentricity_gt_sqrt5_l339_339591

noncomputable def hyperbola_eccentricity (a b : ℝ) (h : a > 0 ∧ b > 0) (h_intersect :  ∀ m : ℝ, ∃ x y : ℝ, y = 2 * x + m ∧ x^2 / a^2 - y^2 / b^2 = 1) : ℝ :=
  let e := √(1 + (b / a)^2)
  e

theorem hyperbola_eccentricity_gt_sqrt5 (a b : ℝ) (h : a > 0 ∧ b > 0)
  (h_intersect : ∀ m : ℝ, ∃ x y : ℝ, y = 2 * x + m ∧ x^2 / a^2 - y^2 / b^2 = 1)
  : hyperbola_eccentricity a b h h_intersect > √5 := 
by
  sorry

end hyperbola_eccentricity_gt_sqrt5_l339_339591


namespace rectangle_area_proof_l339_339652

-- Definitions and given conditions
variable {A B C D M N : Type} [IsRectangle A B C D]
variable [Midpoint M B C]
variable [Midpoint N C D]
variable (perp_AM_MN : Perpendicular A M M N)
variable (AN_length : AN = 60)

-- Goal to prove
theorem rectangle_area_proof : 
  let m := 7200
  let n := 6
  100 * m + n = 720006 :=
by
  -- Definitions are used for constructing the proof 
  sorry

end rectangle_area_proof_l339_339652


namespace polygon_sides_l339_339974

theorem polygon_sides (sum_of_angles : ℕ) (h : sum_of_angles = 540) : ∃ n, (n - 2) * 180 = sum_of_angles ∧ n = 5 :=
by {
  use 5,
  split,
  { rw h,
    refl },
  { refl }
}

end polygon_sides_l339_339974


namespace weight_of_four_parts_l339_339869

theorem weight_of_four_parts (total_weight : ℝ) (num_parts : ℕ) : 
    (total_weight = 2) → (num_parts = 9) → 
    (4 * (total_weight / num_parts) = 8 / 9) :=
by
  intros h1 h2
  rw [h1, h2]
  repeat { rw one_ne_zero }
  norm_num
    } -- Placeholder for proof

end weight_of_four_parts_l339_339869


namespace sum_of_squares_leq_2006_l339_339268

theorem sum_of_squares_leq_2006 (a : Fin 59 → ℝ) 
    (h1 : ∀ i, a i ∈ Set.Icc (-2) 17) 
    (h2 : ∑ i, a i = 0) : 
    ∑ i, (a i)^2 ≤ 2006 := 
by 
    sorry

end sum_of_squares_leq_2006_l339_339268


namespace total_students_in_class_l339_339183

def total_students (chinese math both neither : ℕ) :=
  chinese + math - both + neither

theorem total_students_in_class (h_chinese : 15 = 15) (h_math : 18 = 18)
    (h_both : 8 = 8) (h_neither : 20 = 20) :
    total_students 15 18 8 20 = 45 :=
by
  rw [total_students, h_chinese, h_math, h_both, h_neither]
  norm_num
  sorry

end total_students_in_class_l339_339183


namespace area_of_JMQPON_l339_339664

theorem area_of_JMQPON (JKLM NOPM : Set (ℝ × ℝ))
  (H1 : ∃ l : ℝ, l * l = 25 ∧ JKLM = {z | ∃ (p : ℝ × ℝ), p ∈ JKLM ∧ dist p z ≤ l})
  (H2 : ∃ l : ℝ, l * l = 25 ∧ NOPM = {z | ∃ (p : ℝ × ℝ), p ∈ NOPM ∧ dist p z ≤ l})
  (Q_is_midKL : ∃ KL_mid : ℝ × ℝ, KL_mid = midpoint (K, L) ∧ Q = KL_mid)
  (Q_is_midNO : ∃ NO_mid : ℝ × ℝ, NO_mid = midpoint (N, O) ∧ Q = NO_mid)
  : area (polygon JMQPON) = 25 :=
sorry

end area_of_JMQPON_l339_339664


namespace factorial_division_l339_339488

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339488


namespace largest_number_obtained_l339_339838

theorem largest_number_obtained : 
  ∃ n : ℤ, 10 ≤ n ∧ n ≤ 99 ∧ (∀ m, 10 ≤ m ∧ m ≤ 99 → (250 - 3 * m)^2 ≤ (250 - 3 * n)^2) ∧ (250 - 3 * n)^2 = 4 :=
sorry

end largest_number_obtained_l339_339838


namespace dot_product_value_l339_339939

noncomputable def focal_distance : ℝ := real.sqrt 2

noncomputable def ellipse_equation (x y : ℝ) : Prop :=
  x ^ 2 / 4 + y ^ 2 / 2 = 1

def is_point_on_ellipse (x y : ℝ) : Prop :=
  ellipse_equation x y

def fixed_point_P : ℝ × ℝ := (-1, 0)

def fixed_point_N : ℝ × ℝ := (-7 / 4, 0)

def vector_NA (x y : ℝ) : ℝ × ℝ :=
  (x - (-7 / 4), y)

def vector_NB (x1 y1 x2 y2 : ℝ) : ℝ × ℝ :=
  (x2 - (-7 / 4), y2)

theorem dot_product_value (x1 y1 x2 y2 : ℝ) (h : ellipse_equation 1 (real.sqrt 6 / 2)) :
  ∃ k : ℝ, k ≠ 0 ∧ (vector_NA x1 y1) • (vector_NB x1 y1 x2 y2) = -15 / 16 :=
sorry

end dot_product_value_l339_339939


namespace magician_assistant_trick_l339_339364

theorem magician_assistant_trick :
  ∃ N : ℕ, (∀ (sequence : list ℕ), sequence.length = N →
  (∃ i : ℕ, i < N-1 ∧ 
  (∀ covered1 covered2, list.nth_le sequence i sorry = covered1 ∧ 
  list.nth_le sequence (i+1) sorry = covered2 → 
    (sequence.take i ++ sequence.drop (i+2)).length = N-2)) →
  N >= 101) :=
begin
  use 101,
  sorry
end

end magician_assistant_trick_l339_339364


namespace median_salary_l339_339977

def numEmployees : ℕ := 63

def salaries : List (ℕ × ℕ) :=
  [(1, 140000), (4, 95000), 
   (11, 78000), (8, 55000), 
   (39, 25000)]

/-- Calculate the median salary of the given employees' salaries -/
theorem median_salary : median_salary numEmployees salaries = 25000 := by
  sorry

/-- Helper function to compute the median of a salary distribution -/
noncomputable def median_salary (n : ℕ) (salaries : List (ℕ × ℕ)) : ℕ := by
  let sorted_employees := 
    salaries.foldl (λ acc (num, salary) => 
      acc.append (list.repeat salary num)) []
    let sorted_employees : List ℕ := 
      List.sort (≤) sorted_employees
    sorted_employees.get! (n / 2)

end median_salary_l339_339977


namespace factorial_division_sum_l339_339460

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339460


namespace beth_total_packs_l339_339397

def initial_packs := 4
def number_of_people := 10
def packs_per_person := initial_packs / number_of_people
def packs_found_later := 6

theorem beth_total_packs : packs_per_person + packs_found_later = 6.4 := by
  sorry

end beth_total_packs_l339_339397


namespace reduced_price_per_dozen_is_3_l339_339371

variable (P : ℝ) -- original price of an apple
variable (R : ℝ) -- reduced price of an apple
variable (A : ℝ) -- number of apples originally bought for Rs. 40
variable (cost_per_dozen_reduced : ℝ) -- reduced price per dozen apples

-- Define the conditions
axiom reduction_condition : R = 0.60 * P
axiom apples_bought_condition : 40 = A * P
axiom more_apples_condition : 40 = (A + 64) * R

-- Define the proof problem
theorem reduced_price_per_dozen_is_3 : cost_per_dozen_reduced = 3 :=
by
  sorry

end reduced_price_per_dozen_is_3_l339_339371


namespace power_mod_remainder_l339_339790

theorem power_mod_remainder :
  3 ^ 3021 % 13 = 1 :=
by
  sorry

end power_mod_remainder_l339_339790


namespace average_and_fourth_number_l339_339275

theorem average_and_fourth_number {x : ℝ} (h_avg : ((1 + 2 + 4 + 6 + 9 + 9 + 10 + 12 + x) / 9) = 7) :
  x = 10 ∧ 6 = 6 :=
by
  sorry

end average_and_fourth_number_l339_339275


namespace find_b_for_constant_remainder_l339_339054

theorem find_b_for_constant_remainder (b : ℚ) : 
    (∀ x : ℚ, 3*x^2 - 2*x + 4 ≠ 0) →
    let p : ℚ[x] := 8*x^3 + 5*x^2 + b*x - 8 in
    let q : ℚ[x] := 3*x^2 - 2*x + 4 in
    let r : ℚ[x] := p % q in
    r.degree < 1 → 
    b = -98/9 :=
by
  sorry

end find_b_for_constant_remainder_l339_339054


namespace total_cost_proof_l339_339302

def tuition_fee : ℕ := 1644
def room_and_board_cost : ℕ := tuition_fee - 704
def total_cost : ℕ := tuition_fee + room_and_board_cost

theorem total_cost_proof : total_cost = 2584 := 
by
  sorry

end total_cost_proof_l339_339302


namespace factorial_div_sum_l339_339429

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339429


namespace isaac_journey_time_l339_339706

def travel_time_total (speed : ℝ) (time1 : ℝ) (distance2 : ℝ) (rest_time : ℝ) (distance3 : ℝ) : ℝ :=
  let time2 := distance2 / speed
  let time3 := distance3 / speed
  time1 + time2 * 60 + rest_time + time3 * 60

theorem isaac_journey_time :
  travel_time_total 10 (30 : ℝ) 15 (30 : ℝ) 20 = 270 :=
by
  sorry

end isaac_journey_time_l339_339706


namespace solve_problem_l339_339748

-- Define the function f and the conditions on it
axiom f : ℤ → ℤ
axiom condition1 : ∀ m n : ℤ, f(m + f(f(n))) = -f(f(m + 1)) - n
axiom g : ℤ → ℤ
axiom condition2 : ∀ n : ℤ, g(n) = g(f(n)) 
axiom g_poly : ∀ p : ℤ[X], p.eval n = g(n) → p = (X^2 + X).eval n
  
-- Statement to prove: f(1991) = -1992 and g is a polynomial in (n^2 + n)
theorem solve_problem : (f 1991 = -1992) ∧ (∃ p : ℤ[X], ∀ n : ℤ, g(n) = p.eval (n^2 + n)) :=
by
  sorry

end solve_problem_l339_339748


namespace expression_equals_36_l339_339164

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l339_339164


namespace expression_equals_36_l339_339161

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l339_339161


namespace probability_age_less_than_20_l339_339190

theorem probability_age_less_than_20 (total : ℕ) (ages_gt_30 : ℕ) (ages_lt_20 : ℕ) 
    (h1 : total = 150) (h2 : ages_gt_30 = 90) (h3 : ages_lt_20 = total - ages_gt_30) :
    (ages_lt_20 : ℚ) / total = 2 / 5 :=
by
  simp [h1, h2, h3]
  sorry

end probability_age_less_than_20_l339_339190


namespace perpendicular_vector_parallel_vector_l339_339560

variables {a b : ℝ × ℝ}
def vec_a := (1, 2) : ℝ × ℝ
def vec_b := (-1, 3) : ℝ × ℝ

theorem perpendicular_vector (k : ℝ) : 
  (k * vec_a + vec_b).1 * (vec_a - 3 * vec_b).1 + (k * vec_a + vec_b).2 * (vec_a - 3 * vec_b).2 = 0 → 
  k = -2.5 := 
sorry

theorem parallel_vector (k : ℝ) : 
  (vec_a - 3 * vec_b).1 * (k * vec_a + vec_b).2 = (vec_a - 3 * vec_b).2 * (k * vec_a + vec_b).1 → 
  k = -1/3 :=
sorry

end perpendicular_vector_parallel_vector_l339_339560


namespace compare_a_b_c_l339_339221

theorem compare_a_b_c :
  let a := (3 / 5) ^ (2 / 5)
  let b := (2 / 5) ^ (3 / 5)
  let c := (2 / 5) ^ (2 / 5)
  a > c ∧ c > b :=
by {
  let a := (3 / 5) ^ (2 / 5),
  let b := (2 / 5) ^ (3 / 5),
  let c := (2 / 5) ^ (2 / 5),
  sorry
}

end compare_a_b_c_l339_339221


namespace correct_statement_count_l339_339285

theorem correct_statement_count :
  let s1 := (∀ (x : ℝ), is_pi x → -x = -π)
  let s2 := (∀ (x y : ℝ), has_opposite_sign x y → is_opposite x y)
  let s3 := (∀ (x : ℝ), is_negative x → -x = 3.8)
  let s4 := (∃ (x : ℝ), x = -x)
  let s5 := (∀ (x y : ℝ), is_positive_negative x y → is_opposite x y)
in s1 ∧ s3 ∧ s4 ∧ ¬s2 ∧ ¬s5 → option_correct = 3 :=
begin
  intro,
  -- The proof is omitted as only statement formulation is required.
  sorry
end

end correct_statement_count_l339_339285


namespace sum_bn_999_l339_339892

def is_multiple (a b : ℕ) : Prop := b % a = 0

def bn (n : ℕ) : ℕ :=
  if is_multiple 56 n then 8
  else if is_multiple 72 n then 9
  else if is_multiple 63 n then 7
  else 0

theorem sum_bn_999 : ∑ n in Finset.range 1000, bn n = 358 := by
  sorry

end sum_bn_999_l339_339892


namespace factorial_expression_l339_339441

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339441


namespace intersection_of_M_and_N_l339_339234

open Set

def M : Set ℤ := {-1, 0, 1}

def N : Set ℝ := { x | x^2 ≤ x }

theorem intersection_of_M_and_N :
  M ∩ N = {0, 1} :=
sorry

end intersection_of_M_and_N_l339_339234


namespace factorial_ending_zeros_l339_339296

theorem factorial_ending_zeros (n : ℕ) (h : n = 15) : 
  ∃ k : ℕ, 18 ^ k ∣ nat.factorial n ∧ ¬ 18 ^ (k + 1) ∣ nat.factorial n := 
by
  use 3
  -- sorry

end factorial_ending_zeros_l339_339296


namespace profit_percent_l339_339801

theorem profit_percent (P C : ℝ) (h : (2 / 3) * P = 0.88 * C) : P - C = 0.32 * C → (P - C) / C * 100 = 32 := by
  sorry

end profit_percent_l339_339801


namespace measure_of_angle_D_l339_339777

-- Defining the triangle with given conditions
def is_isosceles (D E F : Type) [has_angle D] [has_angle F] : Prop :=
  angle_is_congruent D F

def isosceles_triangle_conditions (D E F : Type) [has_angle D] [has_angle E] [has_angle F] : Prop :=
  (is_isosceles D E F) ∧ (angle_measure F = 3 * angle_measure E)

-- Stating the problem to prove the measure of angle D
theorem measure_of_angle_D (D E F : Type) [has_angle D] [has_angle E] [has_angle F] 
  (h : isosceles_triangle_conditions D E F) :
  angle_measure D = 540 / 7 := sorry

end measure_of_angle_D_l339_339777


namespace points_concyclic_l339_339756

-- Define a structure for a triangle and its key points
structure Triangle (α : Type*) :=
  (A B C : α)          -- Vertices of the triangle
  (H : α)              -- Orthocenter

-- Define a proof that certain points lie on a single circle
theorem points_concyclic {α : Type*} [euclidean_geometry α] 
  (T : Triangle α) 
  (mid_BC mid_CA mid_AB : α) -- Midpoints of the sides
  (A1 A2 B1 B2 C1 C2 : α)    -- Intersection points
  (H_orthocenter : T.H = orthocenter T.A T.B T.C)
  (A1A2_circle : is_circle (mid_BC) T.H A1 A2)
  (B1B2_circle : is_circle (mid_CA) T.H B1 B2)
  (C1C2_circle : is_circle (mid_AB) T.H C1 C2)
  : concyclic {A1, A2, B1, B2, C1, C2} :=
by
  sorry

end points_concyclic_l339_339756


namespace translation_result_l339_339174

-- Define the initial point A
def A : (ℤ × ℤ) := (-2, 3)

-- Define the translation function
def translate (p : (ℤ × ℤ)) (delta_x delta_y : ℤ) : (ℤ × ℤ) :=
  (p.1 + delta_x, p.2 - delta_y)

-- The theorem stating the resulting point after translation
theorem translation_result :
  translate A 3 1 = (1, 2) :=
by
  -- Skipping proof with sorry
  sorry

end translation_result_l339_339174


namespace train_speed_is_72_kmph_l339_339360

-- Definitions for the conditions
def train_length : ℝ := 350  -- in meters
def platform_length : ℝ := 250  -- in meters
def crossing_time : ℝ := 30  -- in seconds

-- Total distance covered
def total_distance : ℝ := train_length + platform_length

-- Speed in m/s
def speed_mps : ℝ := total_distance / crossing_time

-- Conversion factor from m/s to km/hr
def conversion_factor : ℝ := 3.6

-- Speed in km/hr
def speed_kmph : ℝ := speed_mps * conversion_factor

-- Proof that the speed of the train is 72 km/hr
theorem train_speed_is_72_kmph : speed_kmph = 72 := by
  sorry

end train_speed_is_72_kmph_l339_339360


namespace final_quantity_of_milk_l339_339798

-- Initially, a vessel is filled with 45 litres of pure milk
def initial_milk : Nat := 45

-- First operation: removing 9 litres of milk and replacing with water
def first_operation_milk(initial_milk : Nat) : Nat := initial_milk - 9
def first_operation_water : Nat := 9

-- Second operation: removing 9 litres of the mixture and replacing with water
def milk_fraction_mixture(milk : Nat) (total : Nat) : Rat := milk / total
def water_fraction_mixture(water : Nat) (total : Nat) : Rat := water / total

def second_operation_milk(milk : Nat) (total : Nat) (removed : Nat) : Rat := 
  milk - (milk_fraction_mixture milk total) * removed
def second_operation_water(water : Nat) (total : Nat) (removed : Nat) : Rat := 
  water - (water_fraction_mixture water total) * removed + removed

-- Prove the final quantity of milk
theorem final_quantity_of_milk : second_operation_milk 36 45 9 = 28.8 := by
  sorry

end final_quantity_of_milk_l339_339798


namespace ratio_of_max_coefficients_leq_succ_n_l339_339682

noncomputable theory

open Real -- for real numbers manipulation

theorem ratio_of_max_coefficients_leq_succ_n
  (n : ℕ)
  (a_n a_{n-1} ... a_0 c_{n+1} c_n ... c_0 : ℝ)
  (r : ℝ)
  (f g : ℝ → ℝ)
  (max_coeff_f : ℝ)
  (max_coeff_g : ℝ)
  (f_def : ∀ x, f x = a_n * (x ^ n) + a_{n-1} * (x ^ (n-1)) + ... + a_0)
  (g_def : ∀ x, g x = c_{n+1} * (x ^ (n+1)) + c_n * (x ^ n) + ... + c_0)
  (max_coeff_f_def : max_coeff_f = max (abs a_n) (max (abs a_{n-1}) ... (max (abs a_1) (abs a_0))))
  (max_coeff_g_def : max_coeff_g = max (abs c_{n+1}) (max (abs c_n) ... (max (abs c_1) (abs c_0))))
  (g_eq : ∀ x, g x = (x + r) * f x)
  (Hf : f ≠ 0)
  (Hg : g ≠ 0) :
  max_coeff_f / max_coeff_g ≤ n + 1 := 
sorry

end ratio_of_max_coefficients_leq_succ_n_l339_339682


namespace points_on_parabola_l339_339563

theorem points_on_parabola (a : ℝ) (y1 y2 y3 : ℝ) 
  (h_a : a < -1) 
  (h1 : y1 = (a - 1)^2) 
  (h2 : y2 = a^2) 
  (h3 : y3 = (a + 1)^2) : 
  y1 > y2 ∧ y2 > y3 :=
by
  sorry

end points_on_parabola_l339_339563


namespace sum_of_slopes_l339_339994

-- Define the curves and the conditions
def curve_C (ρ θ : ℝ) : Prop := ρ = 4 * real.cos θ

def curve_M (x y : ℝ) : Prop := x - 2 * y + 2 = 0 ∧ x > 0

-- Parametric equations of curve M
def parametric_curve_M (k : ℝ) (x y : ℝ) : Prop := 
  (x = 2 / (2 * k - 1) ∧ y = 2 * k / (2 * k - 1)) ∧ k > 1/2

-- Given conditions for intersection points A and B
def intersection (k : ℝ) : Prop := 
  curve_M (2 / (2 * k - 1)) (2 * k / (2 * k - 1)) ∧ 
  curve_C (real.sqrt ((2 / (2 * k - 1))^2 + (2 * k / (2 * k - 1))^2)) 
          (real.arctan ((2 * k / (2 * k - 1)) / (2 / (2 * k - 1))))

-- Main theorem statement
theorem sum_of_slopes : 
  (∃ A B : ℝ → ℝ, A 1 ∧ A 3 ∧ B 1 ∧ B 3) → 
  (Σ (slope : ℝ), slope = 4) := 
by
  sorry

end sum_of_slopes_l339_339994


namespace integer_count_between_sqrt8_and_sqrt75_l339_339146

theorem integer_count_between_sqrt8_and_sqrt75 :
  let least_integer_greater_than_sqrt8 := 3
      greatest_integer_less_than_sqrt75 := 8
  in (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6 :=
by
  let least_integer_greater_than_sqrt8 := 3
  let greatest_integer_less_than_sqrt75 := 8
  show (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339146


namespace initial_action_figures_l339_339678

theorem initial_action_figures (initial : ℕ) (added : ℕ) (total : ℕ) 
  (h1 : added = 6) (h2 : total = 10) (h3 : initial + added = total) :
  initial = 4 :=
by 
  rw [h1, h2, add_comm] at h3
  have h4 : initial + 6 = 10 := h3
  linarith

end initial_action_figures_l339_339678


namespace cannot_combine_with_sqrt3_l339_339392

theorem cannot_combine_with_sqrt3 :
  let A := -sqrt 3
  let B := sqrt (1 / 3)
  let C := sqrt 12
  let D := sqrt 18
  ¬exists (k : ℝ), D = k * sqrt 3 := by
  sorry

end cannot_combine_with_sqrt3_l339_339392


namespace part_a_l339_339349

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end part_a_l339_339349


namespace carA_travel_before_B_starts_l339_339325

variables (speedB : ℝ) (dist : ℝ := 330) (extraDistA : ℝ := 30)
def speedA := (5/6) * speedB

noncomputable def timeToMeet := dist / (speedA + speedB)

def distA := timeToMeet * speedA
def distB := timeToMeet * speedB

def headStartDist := distA - distB - extraDistA

theorem carA_travel_before_B_starts : headStartDist = 55 := 
by
  sorry

end carA_travel_before_B_starts_l339_339325


namespace part_a_part_b_part_c_l339_339648

-- Defining the conditions for part (a)
theorem part_a (polygons : set (set Prop')) (area : Prop' → ℝ) :
  (∀ p ∈ polygons, area p ≥ 1/2) →
  (∃ p q ∈ polygons, p ≠ q ∧ (area (p ∩ q)) ≥ 1/5) :=
by sorry

-- Defining the conditions for part (b)
theorem part_b (polygons : set (set Prop')) (area : Prop' → ℝ) :
  (∀ p ∈ polygons, area p ≥ 1/2) →
  (∃ p q r ∈ polygons, p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ (area (p ∩ q ∩ r)) ≥ 1/20) :=
by sorry

-- Defining the conditions for part (c)
theorem part_c (polygons : set (set Prop')) (area : Prop' → ℝ) :
  (∀ p q ∈ polygons, p ≠ q → (area (p ∩ q)) ≥ 1/4) →
  (∃ p q r ∈ polygons, p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ (area (p ∩ q ∩ r)) ≥ 3/40) :=
by sorry

end part_a_part_b_part_c_l339_339648


namespace sector_area_correct_l339_339916

noncomputable def sector_area (arc_length radius : ℝ) : ℝ :=
  0.5 * arc_length * radius

theorem sector_area_correct {radius arc_length : ℝ} (h_radius : radius = 4) (h_arc_length : arc_length = 12) :
  sector_area arc_length radius = 24 :=
by
  unfold sector_area
  rw [h_radius, h_arc_length]
  norm_num
  sorry

end sector_area_correct_l339_339916


namespace factorial_sum_division_l339_339452

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339452


namespace max_value_of_b_l339_339897

theorem max_value_of_b {m b : ℚ} (x : ℤ) 
  (line_eq : ∀ x : ℤ, 0 < x ∧ x ≤ 200 → 
    ¬ ∃ (y : ℤ), y = m * x + 3)
  (m_range : 1/3 < m ∧ m < b) :
  b = 69/208 :=
by
  sorry

end max_value_of_b_l339_339897


namespace negation_proposition_l339_339968

theorem negation_proposition (p : Prop) (h : ∀ x : ℝ, 2 * x^2 + 1 > 0) : ¬p ↔ ∃ x : ℝ, 2 * x^2 + 1 ≤ 0 :=
sorry

end negation_proposition_l339_339968


namespace cost_error_l339_339842

theorem cost_error (initial_amount remaining_amount juice_and_cupcake_cost: ℕ):
  initial_amount = 75 → 
  remaining_amount = 8 → 
  juice_and_cupcake_cost = 40 → 
  initial_amount - remaining_amount ≠ juice_and_cupcake_cost :=
by
  intros h_initial h_remaining h_total_cost
  rw [h_initial, h_remaining] at *
  change 75 - 8 ≠ 40
  sorry

end cost_error_l339_339842


namespace greatest_b_no_minus_six_in_range_l339_339334

open Real

theorem greatest_b_no_minus_six_in_range :
  ∃ (b : ℤ), (b = 8) → (¬ ∃ x : ℝ, x^2 + (b : ℝ) * x + 15 = -6) :=
by {
  -- We need to find the largest integer b such that -6 is not in the range of f(x) = x^2 + bx + 15
  sorry
}

end greatest_b_no_minus_six_in_range_l339_339334


namespace inequality_proof_l339_339727

theorem inequality_proof (a b c : ℝ)
  (h : ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → |c * x^2 + b * x + a| ≤ 2 :=
by
  sorry

end inequality_proof_l339_339727


namespace factorial_sum_division_l339_339455

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339455


namespace factorial_expression_l339_339448

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339448


namespace complex_number_real_imag_parts_equal_l339_339972

theorem complex_number_real_imag_parts_equal (a : ℝ) (i : ℂ) (h1 : i = complex.I) :
  let z := (a - complex.I) * (1 - complex.I) * complex.I in
  z.re = z.im → a = 0 :=
by
  sorry

end complex_number_real_imag_parts_equal_l339_339972


namespace factorial_div_sum_l339_339427

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339427


namespace solve_equation_2021_2020_l339_339260

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end solve_equation_2021_2020_l339_339260


namespace solve_4_times_3_l339_339964

noncomputable def custom_operation (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem solve_4_times_3 : custom_operation 4 3 = 13 := by
  -- Here the proof would be provided, for now we use sorry
  sorry

end solve_4_times_3_l339_339964


namespace isosceles_triangle_perimeter_l339_339069

theorem isosceles_triangle_perimeter (a b : ℝ) (h₀ : is_isosceles_triangle a b) (h₁ : (a - 3)^2 + |b - 4| = 0) :
  (perimeter a b = 10 ∨ perimeter a b = 11) :=
sorry

def is_isosceles_triangle (a b : ℝ) : Prop :=
  a = b ∨ a = b / 2 ∨ b = a / 2

def perimeter (a b : ℝ) : ℝ :=
  if a = b then 2 * a + b
  else if b = a * 2 then a + 2 * b
  else a + b

end isosceles_triangle_perimeter_l339_339069


namespace first_operation_result_l339_339650

def pattern (x y : ℕ) : ℕ :=
  if (x, y) = (3, 7) then 27
  else if (x, y) = (4, 5) then 32
  else if (x, y) = (5, 8) then 60
  else if (x, y) = (6, 7) then 72
  else if (x, y) = (7, 8) then 98
  else 26

theorem first_operation_result : pattern 2 3 = 26 := by
  sorry

end first_operation_result_l339_339650


namespace rectangle_area_l339_339175

theorem rectangle_area (a b : ℝ) (x : ℝ) 
  (h1 : x^2 + (x / 2)^2 = (a + b)^2) 
  (h2 : x > 0) : 
  x * (x / 2) = (2 * (a + b)^2) / 5 := 
by 
  sorry

end rectangle_area_l339_339175


namespace integer_count_between_sqrt8_and_sqrt75_l339_339138

noncomputable def m : ℤ := Int.ceil (Real.sqrt 8)
noncomputable def n : ℤ := Int.floor (Real.sqrt 75)

theorem integer_count_between_sqrt8_and_sqrt75 : (n - m + 1) = 6 := by
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339138


namespace smallest_k_divides_l339_339051

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l339_339051


namespace domain_tangent_l339_339746

def is_not_in_domain (x : ℝ) : Prop :=
  ∃ (k : ℤ), x = (k * (Real.pi / 2) + Real.pi / 12)

theorem domain_tangent {x : ℝ} : ¬ is_not_in_domain x ↔ 
  ∀ (k : ℤ), x ≠ (k * (Real.pi / 2) + Real.pi / 12) :=
begin
  sorry
end

end domain_tangent_l339_339746


namespace total_students_at_year_end_l339_339658

def initial_students : ℝ := 10.0
def added_students : ℝ := 4.0
def new_students : ℝ := 42.0

theorem total_students_at_year_end : initial_students + added_students + new_students = 56.0 :=
by
  sorry

end total_students_at_year_end_l339_339658


namespace smallest_k_l339_339038

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l339_339038


namespace find_a4_l339_339568

noncomputable def seq (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧ 
  (∀ n : ℕ, a (n + 1) = 3 * S n + 2) ∧
  (∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1))

theorem find_a4 : 
  ∀ (a S : ℕ → ℕ), seq a S → a 4 = 80 :=
by
  intro a S h
  have h1 : a 1 = 1 := h.1
  have h2 : ∀ n : ℕ, a (n + 1) = 3 * S n + 2 := h.2.1
  have h3 : ∀ n : ℕ, S n = ∑ i in Finset.range n, a (i + 1) := h.2.2
  sorry

end find_a4_l339_339568


namespace E_is_always_integer_l339_339552

theorem E_is_always_integer (k n : ℕ) (hk : 1 ≤ k) (hk_lt : k < n) :
  let C_n_k := nat.choose n k in
  let E := (n - 2 * k - 2) * C_n_k / (k + 2) in
  E % 1 = 0 :=
by
  sorry

end E_is_always_integer_l339_339552


namespace time_difference_halfway_point_l339_339799

theorem time_difference_halfway_point 
  (T_d : ℝ) 
  (T_s : ℝ := 2 * T_d) 
  (H_d : ℝ := T_d / 2) 
  (H_s : ℝ := T_s / 2) 
  (diff_time : ℝ := H_s - H_d) : 
  T_d = 35 →
  T_s = 2 * T_d →
  diff_time = 17.5 :=
by
  intros h1 h2
  sorry

end time_difference_halfway_point_l339_339799


namespace count_integers_between_sqrt8_and_sqrt75_l339_339123

theorem count_integers_between_sqrt8_and_sqrt75 : 
  let n1 := Real.sqrt 8,
      n2 := Real.sqrt 75 in 
  ∃ count : ℕ, count = (Int.floor n2 - Int.ceil n1) + 1 ∧ count = 6 :=
by
  let n1 := Real.sqrt 8
  let n2 := Real.sqrt 75
  use (Int.floor n2 - Int.ceil n1 + 1)
  simp only [Real.sqrt]
  sorry

end count_integers_between_sqrt8_and_sqrt75_l339_339123


namespace cody_spent_tickets_l339_339395

theorem cody_spent_tickets (initial_tickets lost_tickets remaining_tickets : ℝ) (h1 : initial_tickets = 49.0) (h2 : lost_tickets = 6.0) (h3 : remaining_tickets = 18.0) :
  initial_tickets - lost_tickets - remaining_tickets = 25.0 :=
by
  sorry

end cody_spent_tickets_l339_339395


namespace imaginary_part_of_z_l339_339696

open Complex

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + I) * x + (1 - I) * y = 2) : 
  im ((x + I) / (y - I)) = 1 :=
by
  sorry

end imaginary_part_of_z_l339_339696


namespace travel_allowance_increase_20_l339_339244

def employees_total : ℕ := 480
def employees_no_increase : ℕ := 336
def employees_salary_increase_percentage : ℕ := 10

def employees_salary_increase : ℕ :=
(employees_salary_increase_percentage * employees_total) / 100

def employees_travel_allowance_increase : ℕ :=
employees_total - (employees_salary_increase + employees_no_increase)

def travel_allowance_increase_percentage : ℕ :=
(employees_travel_allowance_increase * 100) / employees_total

theorem travel_allowance_increase_20 :
  travel_allowance_increase_percentage = 20 :=
by sorry

end travel_allowance_increase_20_l339_339244


namespace equidistant_point_x_coord_l339_339367

theorem equidistant_point_x_coord :
  ∃ x y : ℝ, y = x ∧ dist (x, y) (x, 0) = dist (x, y) (0, y) ∧ dist (x, y) (0, y) = dist (x, y) (x, 5 - x)
    → x = 5 / 2 :=
by sorry

end equidistant_point_x_coord_l339_339367


namespace distance_from_A_to_y_axis_l339_339278

-- Define the coordinates of point A
def point_A : ℝ × ℝ := (-3, 4)

-- Define the distance function from a point to the y-axis
def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  abs p.1

-- State the theorem
theorem distance_from_A_to_y_axis :
  distance_to_y_axis point_A = 3 :=
  by
    -- This part will contain the proof, but we omit it with 'sorry' for now.
    sorry

end distance_from_A_to_y_axis_l339_339278


namespace polynomial_identity_l339_339167

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l339_339167


namespace probability_six_distinct_numbers_l339_339789

theorem probability_six_distinct_numbers :
  let total_outcomes := 6^6
  let distinct_outcomes := Nat.factorial 6
  let probability := (distinct_outcomes:ℚ) / (total_outcomes:ℚ)
  probability = 5 / 324 :=
sorry

end probability_six_distinct_numbers_l339_339789


namespace min_line_segments_l339_339229

theorem min_line_segments (n : ℕ) (p : fin n → ℝ × ℝ)
  (h1 : ∀ i j k : fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬ collinear ({p i, p j, p k}))
  (h2 : ∀ a b c d : fin n, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d 
    → ∃ x y z : fin n, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ {x, y, z} ⊆ {a, b, c, d} ∧ 
      p x = line_segment (p x) (p y) ∧ p y = line_segment (p y) (p z) ∧ p z = line_segment (p z) (p x))
  : ∃ m, m = (n - 1) * (n - 2) / 2 :=
sorry

end min_line_segments_l339_339229


namespace two_numbers_match_in_two_positions_l339_339375

theorem two_numbers_match_in_two_positions 
  (N A B C D E : ℕ)
  (hN : N < 1000000)
  (hA : A < 1000000)
  (hB : B < 1000000)
  (hC : C < 1000000)
  (hD : D < 1000000)
  (hE : E < 1000000)
  (hNA : ∃a b c: ℕ, (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (N / 10^a % 10 = A / 10^a % 10) ∧ (N / 10^b % 10 = A / 10^b % 10) ∧ (N / 10^c % 10 = A / 10^c % 10)))
  (hNB : ∃a b c: ℕ, (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (N / 10^a % 10 = B / 10^a % 10) ∧ (N / 10^b % 10 = B / 10^b % 10) ∧ (N / 10^c % 10 = B / 10^c % 10)))
  (hNC : ∃a b c: ℕ, (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (N / 10^a % 10 = C / 10^a % 10) ∧ (N / 10^b % 10 = C / 10^b % 10) ∧ (N / 10^c % 10 = C / 10^c % 10)))
  (hND : ∃a b c: ℕ, (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (N / 10^a % 10 = D / 10^a % 10) ∧ (N / 10^b % 10 = D / 10^b % 10) ∧ (N / 10^c % 10 = D / 10^c % 10)))
  (hNE : ∃a b c: ℕ, (a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ (N / 10^a % 10 = E / 10^a % 10) ∧ (N / 10^b % 10 = E / 10^b % 10) ∧ (N / 10^c % 10 = E / 10^c % 10))) :
  ∃ (X Y : ℕ), (X ∈ {A, B, C, D, E}) ∧ (Y ∈ {A, B, C, D, E}) ∧ X ≠ Y ∧ ∃a b: ℕ, (a ≠ b ∧ (X / 10^a % 10 = Y / 10^a % 10) ∧ (X / 10^b % 10 = Y / 10^b % 10)) :=
by
  sorry

end two_numbers_match_in_two_positions_l339_339375


namespace contrapositive_proposition_l339_339277

theorem contrapositive_proposition (x : ℝ) : (x ≤ -1 ∨ x ≥ 1) → (x^2 ≥ 1) :=
by
  sorry

end contrapositive_proposition_l339_339277


namespace television_price_reduction_l339_339836

variable (P : ℝ) (F : ℝ)
variable (h : F = 0.56 * P - 50)

theorem television_price_reduction :
  F / P = 0.56 - 50 / P :=
by {
  sorry
}

end television_price_reduction_l339_339836


namespace total_pencils_l339_339765

def number_of_pencils_in_drawer (A B : ℕ) : ℕ := 
  A + B

theorem total_pencils (A B : ℕ) (hA : A = 115) (hB : B = 100) : number_of_pencils_in_drawer A B = 215 := by
  rw [number_of_pencils_in_drawer, hA, hB]
  -- Now we should show 115 + 100 = 215.
  have h_add : 115 + 100 = 215 := by 
    -- This can be shown using the omega tactic or by direct evaluation.
    norm_num
  exact h_add

end total_pencils_l339_339765


namespace distinct_triangles_in_octahedron_l339_339110

theorem distinct_triangles_in_octahedron : 
  let V := 8, E := 12 in
  ∃ (T : finset (finset (fin 8))), 
    (∀ (t ∈ T), ∃ (e ∈ edges_of_octahedron), e ⊆ t) ∧ 
    T.card = 12 :=
by
  sorry

end distinct_triangles_in_octahedron_l339_339110


namespace susan_bottles_l339_339269

def cups_needed : ℝ := 12
def mL_per_bottle : ℝ := 250
def cup_to_liter : ℝ := 0.24
def mL_per_liter : ℝ := 1000

theorem susan_bottles : ((cups_needed * cup_to_liter * mL_per_liter) / mL_per_bottle).ceil = 12 :=
by
  sorry

end susan_bottles_l339_339269


namespace problem_statement_l339_339063
open complex real

noncomputable theory

def z1 : ℂ := 1 - 𝒾
def z2 (x y : ℝ) : ℂ := x + y * 𝒾
def OZ1 : ℂ := 1 - 𝒾  -- vector representation of z1

theorem problem_statement (x y : ℝ) (h1: x = 0 ∨ (OZ1 = z1 ∧ (x = -y ∨ x = y))) :
  (x = 0 → z2 x y = y * 𝒾) ∧
  (OZ1 = z1 ∧ x = -y → x + y = 0) ∧
  (OZ1 = z1 ∧ x = y → abs (z1 + z2 x y) = abs (z1 - z2 x y)) :=
begin
  sorry
end

end problem_statement_l339_339063


namespace sum_of_first_10_common_elements_l339_339888

def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n
def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

theorem sum_of_first_10_common_elements :
  let common_elements := λ k : ℕ, if (k % 2 = 0) then Some (geometric_progression k) else None in
  (List.range 20).filter_map common_elements).take 10).sum = 3495250 :=
sorry

end sum_of_first_10_common_elements_l339_339888


namespace factorial_division_l339_339413

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339413


namespace problem_statement_l339_339553

def f_k (k n : ℕ) : ℕ := ∑ m in (Finset.filter (λ m, m ∣ n) (Finset.range (n + 1))), m ^ k

theorem problem_statement 
  (a b : ℕ) 
  (h_div : ∀ n : ℕ, f_k a n ∣ f_k b n) : a = b := 
sorry

end problem_statement_l339_339553


namespace primes_pos_int_solutions_l339_339687

theorem primes_pos_int_solutions 
  (p : ℕ) [hp : Fact (Nat.Prime p)] (a b : ℕ) (h1 : ∃ k : ℤ, (4 * a + p : ℤ) + k * (4 * b + p : ℤ) = b * k * a)
  (h2 : ∃ m : ℤ, (a^2 : ℤ) + m * (b^2 : ℤ) = b * m * a) : a = b ∨ a = b * p :=
  sorry

end primes_pos_int_solutions_l339_339687


namespace trigonometric_identity_l339_339343

noncomputable def ctg (x : ℝ) : ℝ := cos x / sin x
noncomputable def tg (x : ℝ) : ℝ := sin x / cos x

theorem trigonometric_identity (α : ℝ) :
  ctg α - tg α - 2 * tg (2 * α) - 4 * tg (4 * α) = 8 * ctg (8 * α) :=
sorry

end trigonometric_identity_l339_339343


namespace imaginary_part_of_z_l339_339599

theorem imaginary_part_of_z (z : ℂ) (h : (1 + 2 * complex.I) * z = 4 + 3 * complex.I) : z.im = -1 :=
sorry

end imaginary_part_of_z_l339_339599


namespace initial_eggs_proof_l339_339308

-- Definitions based on the conditions provided
def initial_eggs := 7
def added_eggs := 4
def total_eggs := 11

-- The statement to be proved
theorem initial_eggs_proof : initial_eggs + added_eggs = total_eggs :=
by
  -- Placeholder for proof
  sorry

end initial_eggs_proof_l339_339308


namespace solve_log_problem_l339_339100

noncomputable def log_exprs (x : ℝ) : ℝ × ℝ × ℝ :=
  (Real.log (6 * x - 14) / Real.log (Real.sqrt (x / 3 + 3)), 
   2 * Real.log (x - 1) / Real.log (6 * x - 14), 
   Real.log (x / 3 + 3) / Real.log (x - 1))

theorem solve_log_problem (x : ℝ) (hx1 : x ≠ 14 / 6) (hx2 : x ≠ 1) (hx3 : x ≠ -3) :
  let (a, b, c) := log_exprs x in 
  (a = b ∧ c = a - 1 ∨ b = c ∧ a = b - 1 ∨ c = a ∧ b = c - 1) → x = 3 :=
by 
  let (a, b, c) := log_exprs x in 
  assume h : (a = b ∧ c = a - 1 ∨ b = c ∧ a = b - 1 ∨ c = a ∧ b = c - 1),
  sorry

end solve_log_problem_l339_339100


namespace projection_of_AB_on_BC_l339_339924

noncomputable def point (x y : ℝ) := (x, y)

def A := point 1 1
def B := point 0 2
def C := point (-1) (-1)

def vector_sub (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
  (p2.1 - p1.1, p2.2 - p1.2)

def magnitude (v : ℝ × ℝ) : ℝ :=
  real.sqrt (v.1 * v.1 + v.2 * v.2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

def vector_scale (k : ℝ) (v : ℝ × ℝ) : ℝ × ℝ :=
  (k * v.1, k * v.2)

theorem projection_of_AB_on_BC :
  let AB := vector_sub A B in
  let BC := vector_sub B C in
  let mag_AB := magnitude AB in
  let mag_BC := magnitude BC in
  let cos_theta := dot_product AB BC / (mag_AB * mag_BC) in
  let projection_magnitude := mag_AB * abs cos_theta in
  let unit_BC := vector_scale (1 / mag_BC) BC in
  let projection := vector_scale projection_magnitude unit_BC in
  projection = (1/5, 3/5) :=
by
  intros
  sorry

end projection_of_AB_on_BC_l339_339924


namespace each_girl_gets_2_dollars_l339_339270

theorem each_girl_gets_2_dollars :
  let debt := 40
  let lulu_savings := 6
  let nora_savings := 5 * lulu_savings
  let tamara_savings := nora_savings / 3
  let total_savings := tamara_savings + nora_savings + lulu_savings
  total_savings - debt = 6 → (total_savings - debt) / 3 = 2 :=
by
  sorry

end each_girl_gets_2_dollars_l339_339270


namespace find_missing_fraction_l339_339298

theorem find_missing_fraction : 
  let x := -(11 : ℚ) / 60 in 
  (1/3 : ℚ) + (1/2) + (-5/6) + (1/5) + (1/4) + (-2/15) + x = 
  (2 / 15 + 5 / 30 + -10 / 12 + 3 / 15 + 4 / 16 + -4 / 30) + x := 
  0.13333333333333333 :=
by {
  sorry
}

end find_missing_fraction_l339_339298


namespace triangle_angle_A_l339_339673

theorem triangle_angle_A (a c : ℝ) (C A : ℝ) 
  (h1 : a = 4 * Real.sqrt 3)
  (h2 : c = 12)
  (h3 : C = Real.pi / 3)
  (h4 : a < c) :
  A = Real.pi / 6 :=
sorry

end triangle_angle_A_l339_339673


namespace exponentiation_identity_l339_339332

theorem exponentiation_identity :
  (5^4)^2 = 390625 :=
  by sorry

end exponentiation_identity_l339_339332


namespace expression_equals_36_l339_339163

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l339_339163


namespace find_a_increasing_intervals_l339_339071

noncomputable def f (a x : ℝ) : ℝ := 2 * (cos x)^2 + a * sin (2 * x) + 1

theorem find_a (h : f a (π / 3) = 0) : a = -sqrt 3 := sorry

theorem increasing_intervals (h : a = -sqrt 3) (k : ℤ) :
  (∀ x, k * π - 2 * π / 3 < x ∧ x < k * π - π / 6 → f a x > f a (x - ε) ∀ ε > 0) :=
sorry

end find_a_increasing_intervals_l339_339071


namespace domain_of_f_parity_of_f_l339_339941

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 ((x - 1) / (x + 1))

theorem domain_of_f :
  {x : ℝ | f x ∈ ℝ} = {x : ℝ | x < -1 ∨ x > 1} :=
sorry

theorem parity_of_f :
  ∀ x : ℝ, x < -1 ∨ x > 1 → f (-x) = -f x :=
sorry

end domain_of_f_parity_of_f_l339_339941


namespace water_level_eq_l339_339328

theorem water_level_eq (h : ℝ) (ρ_water ρ_oil : ℝ) (h_final : ℝ) :
  h = 40 → ρ_water = 1000 → ρ_oil = 700 →
  let h1 := h_final in
  let h2 := 40 - h_final in
  ρ_water * h1 = ρ_oil * h2 →
  h_final = 16.47 :=
by
  intros h_eq rho_water_eq rho_oil_eq h1 h2 hyd_eq
  have h_leq : h1 + h2 = 40 := by
    rw [h1, h2]
    linarith
  sorry

end water_level_eq_l339_339328


namespace num_divisors_multiple_of_4_9_fact_correct_l339_339111

open Nat

noncomputable def num_divisors_multiple_of_4_9_fact : ℕ :=
  let fact_9 := factorial 9
  let prime_factors := (2 ^ 7) * (3 ^ 4) * 5 * 7
  let choices_for_a := 6 -- 2 to 7 inclusive
  let choices_for_b := 5 -- 0 to 4 inclusive
  let choices_for_c := 2 -- 0 to 1 inclusive
  let choices_for_d := 2 -- 0 to 1 inclusive
  choices_for_a * choices_for_b * choices_for_c * choices_for_d

theorem num_divisors_multiple_of_4_9_fact_correct : num_divisors_multiple_of_4_9_fact = 120 := by
  sorry

end num_divisors_multiple_of_4_9_fact_correct_l339_339111


namespace part_a_part_b_l339_339921

def z1 : ℂ := 1 + complex.i
def z2 : ℂ := 1 - complex.i

theorem part_a : complex.conj z1 = z2 := by
  sorry

theorem part_b : complex.abs z1 = complex.abs z2 := by
  sorry

end part_a_part_b_l339_339921


namespace problem1_1_problem1_2_problem2_l339_339611

-- Definition of sets A and B
def setA (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 4}
def setB : Set ℝ := {x | x < -5 ∨ x > 3}

-- Problem (1)
theorem problem1_1 : setA 1 ∩ setB = {x | 3 < x ∧ x ≤ 5} :=
by
  sorry

theorem problem1_2 : setA 1 ∪ setB = {x | x < -5 ∨ x ≥ 1} :=
by
  sorry

-- Problem (2)
theorem problem2 (m : ℝ) (h : setA m ⊆ setB) : m ∈ Iio (-9) ∪ Ioi 3 :=
by
  sorry

end problem1_1_problem1_2_problem2_l339_339611


namespace triangle_ABC_properties_l339_339353

open Complex

/-- Points A, B, C in the complex plane -/
def A : ℂ := 1
def B : ℂ := 2 + I
def C : ℂ := -1 + 2 * I

/-- Prove that the triangle ABC is right-angled and has an area of 2 -/
theorem triangle_ABC_properties :
  let AB := B - A
  let BC := C - B
  let CA := C - A
  let is_right_angled := ∥AB∥^2 + ∥CA∥^2 = ∥BC∥^2
  let area := 1 / 2 * (∥C - A∥) * (∥B - A∥)
  is_right_angled ∧ area = 2 :=
by
  sorry

end triangle_ABC_properties_l339_339353


namespace range_of_a_l339_339642

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (e^x - a)^2 + x^2 - 2 * a * x + a^2 ≤ 1 / 2) ↔ a = 1 / 2 :=
by
  sorry

end range_of_a_l339_339642


namespace calculate_savings_l339_339898

variables (F P : ℝ)
noncomputable def fox_price : ℝ := 15
noncomputable def pony_price : ℝ := 18

noncomputable def total_fox_price : ℝ := 3 * fox_price
noncomputable def total_pony_price : ℝ := 2 * pony_price

noncomputable def total_savings : ℝ := (total_fox_price * F / 100) + (total_pony_price * P / 100)

theorem calculate_savings (h : F + P = 18) : total_savings F P = (45 * F / 100) + (36 * P / 100) :=
by {
  rw [total_fox_price, total_pony_price, fox_price, pony_price],
  sorry
}

end calculate_savings_l339_339898


namespace factorial_computation_l339_339470

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339470


namespace count_integers_between_sqrt8_sqrt75_l339_339116

theorem count_integers_between_sqrt8_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ ∀ i : ℤ, (3 ≤ i ∧ i ≤ 8) ↔ (sqrt 8 < i ∧ i < sqrt 75) :=
by
  use 6
  sorry

end count_integers_between_sqrt8_sqrt75_l339_339116


namespace shrimp_price_l339_339822

theorem shrimp_price (y : ℝ) (h : 0.6 * (y / 4) = 2.25) : y = 15 :=
sorry

end shrimp_price_l339_339822


namespace water_level_equilibrium_l339_339326

theorem water_level_equilibrium (h : ℝ) (ρ_water ρ_oil : ℝ) :
  h = 40 → ρ_water = 1000 → ρ_oil = 700 →
  let h_1 := 40 / (1 + ρ_water / ρ_oil)
  in h_1 ≈ 16.47 :=
by
  sorry

end water_level_equilibrium_l339_339326


namespace complex_number_properties_l339_339912

open Complex

theorem complex_number_properties (z : ℂ) (h : z = 2 / (1 + (⟨0, sqrt 3⟩ : ℂ))) :
  (z.re = 1 / 2) ∧ (conj z = 1 / z) :=
by
  sorry

end complex_number_properties_l339_339912


namespace angle_divided_by_three_in_third_quadrant_l339_339081

theorem angle_divided_by_three_in_third_quadrant
  (α : ℝ)
  (hα_Q3 : π < α ∧ α < 3 * π / 2)
  (hCosAbs : |cos (α / 3)| = -cos (α / 3)) :
  π + 2 * π < α / 3 ∧ α / 3 < π + 4 * π :=
sorry

end angle_divided_by_three_in_third_quadrant_l339_339081


namespace integer_count_between_sqrt8_and_sqrt75_l339_339142

theorem integer_count_between_sqrt8_and_sqrt75 :
  let least_integer_greater_than_sqrt8 := 3
      greatest_integer_less_than_sqrt75 := 8
  in (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6 :=
by
  let least_integer_greater_than_sqrt8 := 3
  let greatest_integer_less_than_sqrt75 := 8
  show (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339142


namespace polynomial_identity_l339_339166

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l339_339166


namespace percentage_of_juniors_is_22_l339_339982

-- Define the number of students as a constant
def total_students : ℕ := 800

-- Define the number of seniors
def seniors : ℕ := 160

-- The percentage of students that are sophomores
def percentage_sophomores : ℝ := 0.25

-- The number of sophomores based on the percentage
def sophomores : ℕ := percentage_sophomores * total_students

-- The number of freshmen is 64 more than the number of sophomores
def freshmen : ℕ := sophomores + 64

-- The number of juniors is the remainder of the total students
def juniors : ℕ := total_students - (freshmen + sophomores + seniors)

-- Calculate the percentage of juniors
def percentage_juniors : ℝ := (juniors : ℝ) / total_students * 100

-- The theorem states that the percentage of juniors is 22%
theorem percentage_of_juniors_is_22 : percentage_juniors = 22 := by
  sorry

end percentage_of_juniors_is_22_l339_339982


namespace base_d_digit_difference_l339_339740

theorem base_d_digit_difference (A C d : ℕ) (h1 : d > 8)
  (h2 : d * A + C + (d * C + C) = 2 * d^2 + 3 * d + 2) :
  (A - C = d + 1) :=
sorry

end base_d_digit_difference_l339_339740


namespace smallest_k_divides_l339_339024

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l339_339024


namespace factorial_division_identity_l339_339499

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339499


namespace relationship_among_a_b_c_l339_339222

noncomputable def a : ℝ := Real.logb 1.2 0.8
noncomputable def b : ℝ := Real.logb 0.7 0.8
noncomputable def c : ℝ := (1.2 : ℝ)^0.8

theorem relationship_among_a_b_c : a < b ∧ b < c := by  
  sorry

end relationship_among_a_b_c_l339_339222


namespace factorial_division_l339_339411

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339411


namespace binary_mul_1101_111_eq_1001111_l339_339012

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end binary_mul_1101_111_eq_1001111_l339_339012


namespace scout_troop_profit_l339_339374

noncomputable def candy_bar_cost : ℝ := 3 / 4
noncomputable def candy_bar_sell_price : ℝ := 2 / 3
def total_candy_bars : ℕ := 2000
def unsold_candy_bars : ℕ := 50
def total_cost : ℝ := total_candy_bars * candy_bar_cost
def sold_candy_bars : ℕ := total_candy_bars - unsold_candy_bars
def total_revenue : ℝ := sold_candy_bars * candy_bar_sell_price
def profit : ℝ := total_revenue - total_cost

theorem scout_troop_profit :
  profit = -200 := by
  sorry

end scout_troop_profit_l339_339374


namespace find_angle_y_l339_339005

theorem find_angle_y (angle_ABC angle_ABD angle_ADB y : ℝ)
  (h1 : angle_ABC = 115)
  (h2 : angle_ABD = 180 - angle_ABC)
  (h3 : angle_ADB = 30)
  (h4 : angle_ABD + angle_ADB + y = 180) :
  y = 85 := 
sorry

end find_angle_y_l339_339005


namespace factorial_expression_l339_339445

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339445


namespace factorial_sum_division_l339_339458

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339458


namespace calc_c_15_l339_339700

noncomputable def c : ℕ → ℝ
| 0 => 1 -- This case won't be used, setup for pattern match
| 1 => 3
| 2 => 5
| (n+3) => c (n+2) * c (n+1)

theorem calc_c_15 : c 15 = 3 ^ 235 :=
sorry

end calc_c_15_l339_339700


namespace charlie_wins_probability_l339_339388

theorem charlie_wins_probability :
  let p_not_six := 5 / 6,
      p_six := 1 / 6,
      infinite_sum := ∑' (n : ℕ), p_not_six^(3 * (n + 1)) * p_six in
  infinite_sum = 125 / 546 := by
  sorry

end charlie_wins_probability_l339_339388


namespace six_digit_numbers_count_l339_339621

theorem six_digit_numbers_count :
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3)) = 60 := 
by
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  calc
    (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3))
      = 720 / (6 * 2 * 1) : by rw [Nat.factorial_six, Nat.factorial_three, Nat.factorial_two, Nat.factorial_one]
  ... = 720 / 12 : by norm_num
  ... = 60 : by norm_num

end six_digit_numbers_count_l339_339621


namespace two_degrees_above_zero_l339_339155

-- Define the concept of temperature notation
def temperature_notation (temp: ℝ) : String :=
  if temp < 0 then "-" ++ temp.nat_abs.toString ++ "°C"
  else "+" ++ temp.toString ++ "°C"

-- Given condition: -2 degrees Celsius is denoted as -2°C
def given_condition := temperature_notation (-2) = "-2°C"

-- Proof statement: 2 degrees Celsius above zero is denoted as +2°C given the condition
theorem two_degrees_above_zero : given_condition → temperature_notation 2 = "+2°C" := by
  intro h
  sorry

end two_degrees_above_zero_l339_339155


namespace find_A_l339_339675

variables (a c : ℝ) (C A : ℝ)

-- Given conditions
def condition_1 : a = 4 * real.sqrt 3 := sorry
def condition_2 : c = 12 := sorry
def condition_3 : C = real.pi / 3 := sorry

theorem find_A : A = real.pi / 6 :=
by
  -- apply the given conditions
  have h1 : a = 4 * real.sqrt 3 := condition_1,
  have h2 : c = 12 := condition_2,
  have h3 : C = real.pi / 3 := condition_3,
  sorry

end find_A_l339_339675


namespace geometric_sequence_product_l339_339584

theorem geometric_sequence_product 
  (a : ℕ → ℝ) (r : ℝ)
  (h_geom : ∀ n, a (n+1) = a n * r)
  (h_pos : ∀ n, a n > 0)
  (h_log_sum : Real.log (a 3) + Real.log (a 8) + Real.log (a 13) = 6) :
  a 1 * a 15 = 10000 := 
sorry

end geometric_sequence_product_l339_339584


namespace B348_divisibility_problem_solution_l339_339827

theorem B348_divisibility (B : ℕ) (hB : B < 10) (h : (B * 1000 + 348) % 16 = 0) : B = 5 ∨ B = 7 :=
by sorry

theorem problem_solution : ∃ B1 B2 : ℕ, B1 = 5 ∧ B2 = 7 ∧ B1 + B2 = 12 :=
by {
  use 5,
  use 7,
  split,
  { refl },
  split,
  { refl },
  { norm_num }
}

end B348_divisibility_problem_solution_l339_339827


namespace sum_of_coeffs_l339_339903

-- Given conditions
def poly_expansion (x : ℝ) : ℝ := (x + 1)^8
def a₀ : ℝ := 1
def a₈ : ℝ := 1

-- Proof problem
theorem sum_of_coeffs (a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  poly_expansion 0 = 1 + a₀ + a₁ * 0 + a₂ * 0^2 + a₃ * 0^3 + a₄ * 0^4 + a₅ * 0^5 + 
                      a₆ * 0^6 + a₇ * 0^7 + a₈ * 0^8 ∧
  poly_expansion 1 = 1 + a₁ * 1 + a₂ * 1^2 + a₃ * 1^3 + a₄ * 1^4 + a₅ * 1^5 + 
                      a₆ * 1^6 + a₇ * 1^7 + 1 * 1^8 →
  a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 254 :=
begin
  sorry
end

end sum_of_coeffs_l339_339903


namespace volume_of_cuboid_is_250_cm3_l339_339779

-- Define the edge length of the cube
def edge_length (a : ℕ) : ℕ := 5

-- Define the volume of a single cube
def cube_volume := (edge_length 5) ^ 3

-- Define the total volume of the cuboid formed by placing two such cubes in a line
def cuboid_volume := 2 * cube_volume

-- Theorem stating the volume of the cuboid formed
theorem volume_of_cuboid_is_250_cm3 : cuboid_volume = 250 := by
  sorry

end volume_of_cuboid_is_250_cm3_l339_339779


namespace probability_prime_or_multiple_of_4_l339_339870

def balls : List ℕ := [2, 3, 4, 5, 6, 7, 8, 9]

def isPrimeOrMultipleOf4 (n : ℕ) : Prop :=
  Nat.Prime n ∨ n % 4 = 0

def satisfyingBalls : List ℕ :=
  balls.filter isPrimeOrMultipleOf4

def numberOfSatisfyingBalls : ℕ := satisfyingBalls.length
def totalBalls : ℕ := balls.length

theorem probability_prime_or_multiple_of_4 :
  numberOfSatisfyingBalls / totalBalls = 3 / 4 :=
by
  have h : finishingProportion = 6 := rfl
  have h : totalBalls = 8 := rfl
  sorry

end probability_prime_or_multiple_of_4_l339_339870


namespace diameter_of_circumscribed_circle_l339_339635

theorem diameter_of_circumscribed_circle (a : ℝ) (A : ℝ) (D : ℝ) 
  (h1 : a = 12) (h2 : A = 30) : D = 24 :=
by
  sorry

end diameter_of_circumscribed_circle_l339_339635


namespace sum_of_possible_values_final_sum_of_values_l339_339948

noncomputable def mean (xs : List ℝ) := xs.sum / xs.length

def mode (xs : List ℝ) : ℝ :=
  xs.mode  

def median (xs : List ℝ) : ℝ :=
  let sorted := xs.sorted
  if h : sorted.length % 2 = 1 then
    sorted.get (sorted.length / 2)
  else
    (sorted.get (sorted.length / 2 - 1) + sorted.get (sorted.length / 2)) / 2

theorem sum_of_possible_values (x : ℝ)
  (h : mean [6,1,3,1,1,7,x] = median [6,1,3,1,1,7,x] + mode [6,1,3,1,1,7,x]
           ∧ mean [6,1,3,1,1,7,x] ≠ median [6,1,3,1,1,7,x]
           ∧ median [6,1,3,1,1,7,x] ≠ mode [6,1,3,1,1,7,x]) : 
  x = 11 ∨ x = 2 :=
sorry

theorem final_sum_of_values' : 
(sum_of_possible_values_list = [11, 2] → 
 sum_of_possible_values_list.sum = 13) :=
sorry

end sum_of_possible_values_final_sum_of_values_l339_339948


namespace factorial_division_l339_339418

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339418


namespace mean_of_four_integers_l339_339294

theorem mean_of_four_integers (x : ℝ) (h : (78 + 83 + 82 + x) / 4 = 80) : x = 77 ∧ x = 80 - 3 :=
by
  have h1 : 78 + 83 + 82 + x = 4 * 80 := by sorry
  have h2 : 78 + 83 + 82 = 243 := by sorry
  have h3 : 243 + x = 320 := by sorry
  have h4 : x = 320 - 243 := by sorry
  have h5 : x = 77 := by sorry
  have h6 : x = 80 - 3 := by sorry
  exact ⟨h5, h6⟩

end mean_of_four_integers_l339_339294


namespace median_length_eq_l339_339653

variable (A B C M N P G : Type) [Scalene A B C]
variable (A_N_length : ℝ)
variable (B_P_length : ℝ)
variable (area_tri : ℝ)

theorem median_length_eq {A B C : Triangle}
  (scaleneABC : ¬IsEquilateral A B C)
  (AN_length : med A N = 3)
  (BP_length : med B P = 6)
  (triangle_area : area △A B C = 3 * sqrt 15):
  med C M = 3 * sqrt 6 :=
by
  sorry

end median_length_eq_l339_339653


namespace factorial_sum_division_l339_339450

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339450


namespace factorial_division_identity_l339_339504

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339504


namespace factorial_div_sum_l339_339436

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339436


namespace points_on_line_relation_l339_339585

theorem points_on_line_relation (b y1 y2 y3 : ℝ) 
  (h1 : y1 = -3 * (-2) + b) 
  (h2 : y2 = -3 * (-1) + b) 
  (h3 : y3 = -3 * 1 + b) : 
  y1 > y2 ∧ y2 > y3 :=
sorry

end points_on_line_relation_l339_339585


namespace water_level_eq_l339_339329

theorem water_level_eq (h : ℝ) (ρ_water ρ_oil : ℝ) (h_final : ℝ) :
  h = 40 → ρ_water = 1000 → ρ_oil = 700 →
  let h1 := h_final in
  let h2 := 40 - h_final in
  ρ_water * h1 = ρ_oil * h2 →
  h_final = 16.47 :=
by
  intros h_eq rho_water_eq rho_oil_eq h1 h2 hyd_eq
  have h_leq : h1 + h2 = 40 := by
    rw [h1, h2]
    linarith
  sorry

end water_level_eq_l339_339329


namespace six_digit_numbers_count_l339_339622

theorem six_digit_numbers_count :
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3)) = 60 := 
by
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  calc
    (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3))
      = 720 / (6 * 2 * 1) : by rw [Nat.factorial_six, Nat.factorial_three, Nat.factorial_two, Nat.factorial_one]
  ... = 720 / 12 : by norm_num
  ... = 60 : by norm_num

end six_digit_numbers_count_l339_339622


namespace car_parking_arrangements_l339_339976

theorem car_parking_arrangements : 
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  (red_car_positions * arrange_black_cars) = 14400 := 
by
  let grid_size := 6
  let group_size := 3
  let red_car_positions := Nat.choose grid_size group_size
  let arrange_black_cars := Nat.factorial grid_size
  sorry

end car_parking_arrangements_l339_339976


namespace round_robin_tournament_650_l339_339191

theorem round_robin_tournament_650 :
  ∀ (teams : Finset ℕ), (∀ i ∈ teams, ∃ wins : Finset ℕ, (wins.card = 12) ∧ (∀ j ∈ wins, j ≠ i ∧ j ∈ teams)) →
  (∀ i ∈ teams, ∃ losses : Finset ℕ, (losses.card = 12) ∧ (∀ j ∈ losses, j ≠ i ∧ j ∈ teams)) →
  teams.card = 25 →
  ∃ triplets : Finset (Finset ℕ), (triplets.card = 650) ∧ (∀ t ∈ triplets, ∃ A B C, A ∈ teams ∧ B ∈ teams ∧ C ∈ teams ∧ A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ A beats B ∧ B beats C ∧ C beats A) :=
by
  sorry

end round_robin_tournament_650_l339_339191


namespace length_CK_12_angle_ABC_60_l339_339408

variables {ABC : Type} [triangle ABC]
variables {K K1 K2 K3 : Point}
variables {O O1 O2 : Center} 
variables (BK1 CK2 BC r R : ℝ) (B C : Point)

noncomputable def length_CK : ℝ :=
  CK2

noncomputable def angle_ABC : ℝ :=
  60

theorem length_CK_12 (h1 : BK1 = 4) (h2 : CK2 = 8) (h3 : BC = 18)
    (h4 : ∀ r1 r2, r1 = r2) : length_CK CK2 = 12 := 
sorry

theorem angle_ABC_60 (h5 : ∀ r1, r1 = r) (h6 : ∠K1BC = 30) : angle_ABC 60 = 60 :=
sorry

end length_CK_12_angle_ABC_60_l339_339408


namespace minimum_seats_necessary_l339_339394

theorem minimum_seats_necessary (n : ℕ) (h1 : n = 26) 
  (h2 : ∀ (x y : ℕ), 1 ≤ x → x < y → y ≤ n → ∃ p, ∀ (i : ℕ), x ≤ i → i < y → p i = true) : 
  ∃ s, s = 25 :=
by sorry

end minimum_seats_necessary_l339_339394


namespace multiples_of_8_has_highest_avg_l339_339341

def average_of_multiples (m : ℕ) (a b : ℕ) : ℕ :=
(a + b) / 2

def multiples_of_7_avg := average_of_multiples 7 7 196 -- 101.5
def multiples_of_2_avg := average_of_multiples 2 2 200 -- 101
def multiples_of_8_avg := average_of_multiples 8 8 200 -- 104
def multiples_of_5_avg := average_of_multiples 5 5 200 -- 102.5
def multiples_of_9_avg := average_of_multiples 9 9 189 -- 99

theorem multiples_of_8_has_highest_avg :
  multiples_of_8_avg > multiples_of_7_avg ∧
  multiples_of_8_avg > multiples_of_2_avg ∧
  multiples_of_8_avg > multiples_of_5_avg ∧
  multiples_of_8_avg > multiples_of_9_avg :=
by
  sorry

end multiples_of_8_has_highest_avg_l339_339341


namespace samantha_snuck_out_jellybeans_l339_339770

theorem samantha_snuck_out_jellybeans :
  ∃ S : ℕ, 90 - (S + 12) + (S + 12) / 2 = 72 ∧ S = 24 :=
begin
  use 24,
  split,
  {
    -- Show that 90 - (24 + 12) + (24 + 12) / 2 = 72
    have h1 : 90 - (24 + 12) + (24 + 12) / 2 = 72,
    {
      simp,
    },
    exact h1,
  },
  refl,
end

end samantha_snuck_out_jellybeans_l339_339770


namespace find_breadth_of_cuboid_l339_339534

variable (l : ℝ) (h : ℝ) (surface_area : ℝ) (b : ℝ)

theorem find_breadth_of_cuboid (hL : l = 10) (hH : h = 6) (hSA : surface_area = 480) 
  (hFormula : surface_area = 2 * (l * b + b * h + h * l)) : b = 11.25 := by
  sorry

end find_breadth_of_cuboid_l339_339534


namespace total_football_games_l339_339731

theorem total_football_games (games_this_year : ℕ) (games_last_year : ℕ) (total_games : ℕ) : 
  games_this_year = 14 → games_last_year = 29 → total_games = games_this_year + games_last_year → total_games = 43 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end total_football_games_l339_339731


namespace solve_x2_minus_8floorx_plus_7_eq_0_l339_339532

theorem solve_x2_minus_8floorx_plus_7_eq_0 (x : ℝ) :
  x^2 - 8 * (Real.floor x) + 7 = 0 ↔
  x = 1 ∨ x = Real.sqrt 33 ∨ x = Real.sqrt 41 ∨ x = 7 :=
by
  sorry

end solve_x2_minus_8floorx_plus_7_eq_0_l339_339532


namespace sum_of_four_numbers_in_ratio_is_correct_l339_339557

variable (A B C D : ℝ)
variable (h_ratio : A / B = 2 / 3 ∧ B / C = 3 / 4 ∧ C / D = 4 / 5)
variable (h_biggest : D = 672)

theorem sum_of_four_numbers_in_ratio_is_correct :
  A + B + C + D = 1881.6 :=
by
  sorry

end sum_of_four_numbers_in_ratio_is_correct_l339_339557


namespace length_of_room_l339_339751

theorem length_of_room 
  (width : ℝ) (total_cost : ℝ) (rate_per_sq_meter : ℝ) 
  (h_width : width = 3.75) 
  (h_total_cost : total_cost = 16500) 
  (h_rate_per_sq_meter : rate_per_sq_meter = 800) : 
  ∃ length : ℝ, length = 5.5 :=
by
  sorry

end length_of_room_l339_339751


namespace cubic_function_extreme_values_l339_339228

theorem cubic_function_extreme_values (a b c d : ℝ) :
  let f := λ x : ℝ, a * x^3 + b * x^2 + c * x + d
  let f' := λ x : ℝ, 3 * a * x^2 + 2 * b * x + c
  (∀ x : ℝ, x * f'(x) = 0 → x = 0 ∨ x = 2 ∨ x = -2) →
  f (2) ≤ f x ∧ f x ≤ f (-2) :=
by
  sorry

end cubic_function_extreme_values_l339_339228


namespace step1_1_step1_2_step1_3_conjecture_part3_1_part3_2_l339_339008

theorem step1_1 (x : ℕ) : (x + 1) * (x - 1) = x^2 - 1 := by sorry

theorem step1_2 (x : ℕ) : (x - 1) * (x^2 + x + 1) = x^3 - 1 := by sorry

theorem step1_3 (x : ℕ) : (x - 1) * (x^3 + x^2 + x + 1) = x^4 - 1 := by sorry

theorem conjecture (x n : ℕ) : (x - 1) * (x^n + x^{n-1} + ... + x + 1) = x^{n+1} - 1 := by sorry

theorem part3_1 : 2^5 + 2^4 + 2^3 + 2^2 + 2 + 1 = 2^6 - 1 := by sorry

theorem part3_2 (n : ℕ) : 2^{n-1} + 2^{n-2} + ... + 2 + 1 = 2^n - 1 := by sorry

end step1_1_step1_2_step1_3_conjecture_part3_1_part3_2_l339_339008


namespace six_digit_numbers_count_l339_339623

theorem six_digit_numbers_count :
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3)) = 60 := 
by
  let n := 6
  let m1 := 3
  let m2 := 2
  let m3 := 1
  calc
    (Nat.factorial n) / ((Nat.factorial m1) * (Nat.factorial m2) * (Nat.factorial m3))
      = 720 / (6 * 2 * 1) : by rw [Nat.factorial_six, Nat.factorial_three, Nat.factorial_two, Nat.factorial_one]
  ... = 720 / 12 : by norm_num
  ... = 60 : by norm_num

end six_digit_numbers_count_l339_339623


namespace smallest_k_for_divisibility_l339_339033

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l339_339033


namespace smallest_k_divides_l339_339050

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l339_339050


namespace solve_equation_2021_2020_l339_339258

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end solve_equation_2021_2020_l339_339258


namespace visits_possible_in_196_flights_l339_339187

theorem visits_possible_in_196_flights (V : Type) [Fintype V] [DecidableEq V] (E : V → V → Prop) (h1 : ∀ v1 v2 : V, ∃ p : List V, p.head = v1 ∧ p.ilast = some v2 ∧ ∀ i ∈ p.tail, E (p.get i) (p.get (i+1)))
  (h2 : Fintype.card V = 100) : ∃ route : List V, route.length ≤ 196 ∧ (∀ v ∈ Finset.univ, v ∈ route) :=
by
  -- Proof will be provided here
  sorry

end visits_possible_in_196_flights_l339_339187


namespace four_distinct_numbers_are_prime_l339_339533

-- Lean 4 statement proving the conditions
theorem four_distinct_numbers_are_prime : 
  ∃ (a b c d : ℕ), 
    a = 1 ∧ b = 2 ∧ c = 3 ∧ d = 5 ∧ 
    (Prime (a * b + c * d)) ∧ 
    (Prime (a * c + b * d)) ∧ 
    (Prime (a * d + b * c)) := 
sorry

end four_distinct_numbers_are_prime_l339_339533


namespace find_a_k_and_max_profit_l339_339933

-- Definitions for the given conditions
def daily_sales_volume (x : ℝ) (a k : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then a * (x - 4)^2 + 6 / (x - 1)
  else if 3 < x ∧ x ≤ 5 then k * x + 7
  else 0

def profit (x : ℝ) (a k : ℝ) : ℝ :=
  if 1 < x ∧ x ≤ 3 then ((x - 4)^2 + 6 / (x - 1)) * (x - 1)
  else if 3 < x ∧ x ≤ 5 then (-x + 7) * (x - 1)
  else 0

variables {a k : ℝ}

-- Problem statement
theorem find_a_k_and_max_profit :
  (daily_sales_volume 3 a k = 4 ∧
   ∀ x, (3 < x ∧ x ≤ 5) → (k * x + 7) ≥ 2) →
  (a = 1 ∧ k = -1 ∧
   (∀ x, profit x a k ≤ profit 2 a k)) :=
by
  sorry

end find_a_k_and_max_profit_l339_339933


namespace sum_geometric_series_l339_339852

theorem sum_geometric_series (a r n : ℕ) (h_a : a = 1) (h_r : r = 3) (h_n : n = 11) :
    ∑ k in Finset.range n, r^k = 88573 := by
  sorry

end sum_geometric_series_l339_339852


namespace find_a_plus_b_l339_339157

theorem find_a_plus_b (a b : ℤ) (h : 2*x^3 - a*x^2 - 5*x + 5 = (2*x^2 + a*x - 1)*(x - b) + 3) : a + b = 4 :=
by {
  -- Proof omitted
  sorry
}

end find_a_plus_b_l339_339157


namespace sum_first_2017_terms_seq_l339_339216

theorem sum_first_2017_terms_seq 
  (S : ℕ → ℝ) 
  (a : ℕ → ℝ) 
  (h1 : ∀ n, S n = ∑ i in Finset.range n, a (i + 1)) 
  (h2 : a 1 = 1) 
  (h3 : S 2017 / 2017 - S 2015 / 2015 = 1) :
  ∑ i in Finset.range 2017, (1 / S (i + 1)) = 2017 / 1009 := 
sorry

end sum_first_2017_terms_seq_l339_339216


namespace product_eq_l339_339891

theorem product_eq : 
  (∏ n in Finset.range (12 - 1) + 2, (1 - (1 / (n^2)))) = (13 / 24) :=
by
  sorry

end product_eq_l339_339891


namespace polar_equation_to_cartesian_l339_339639

noncomputable def polar_to_cartesian (ρ θ : ℝ) : ℝ × ℝ := (ρ * cos θ, ρ * sin θ)

theorem polar_equation_to_cartesian {ρ θ : ℝ} (h : ρ * (cos θ)^2 = 2 * sin θ) :
  (polar_to_cartesian ρ θ).1 ^ 2 = 2 * (polar_to_cartesian ρ θ).2 :=
sorry

end polar_equation_to_cartesian_l339_339639


namespace largest_prime_factor_8250_l339_339335

-- Define a function to check if a number is prime (using an existing library function)
def is_prime (n: ℕ) : Prop := Nat.Prime n

-- Define the given problem statement as a Lean theorem
theorem largest_prime_factor_8250 :
  ∃ p, is_prime p ∧ p ∣ 8250 ∧ 
    ∀ q, is_prime q ∧ q ∣ 8250 → q ≤ p :=
sorry -- The proof will be filled in later

end largest_prime_factor_8250_l339_339335


namespace quadrilateral_sides_area_l339_339649

variables (A B C D M N : Type*)
variables [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [metric_space M] [metric_space N]

-- Definitions of points and geometric conditions
variables (AB BC CD AD AM MD BN NC : ℝ)
variables (perimeter : ℝ)
variables (∠BAD ∠ABC : ℝ)

-- Given conditions and goal in Lean 4
def quadrilateral_ABCD := 
  convex_quadrilateral A B C D ∧
  angle_Bisector B intersects D at M ∧
  perp_from_A_to_BC intersect_BC_at N ∧
  BN = NC ∧ 
  AM = 2 * MD ∧ 
  perimeter = 5 + sqrt 3 ∧ 
  ∠BAD = pi / 2 ∧ 
  ∠ABC = pi / 3 ∧ 
  AB = 2 ∧ 
  BC = 2 ∧ 
  AD = sqrt 3 ∧ 
  CD = 1 ∧ 
  area_ABCD = 3 * sqrt 3 / 2

theorem quadrilateral_sides_area :
  ∃ (AB BC CD AD AM MD BN NC : ℝ), 
    quadrilateral_ABCD A B C D M N AB BC CD AD AM MD BN NC perimeter (pi / 2) (pi / 3) :=
    sorry

end quadrilateral_sides_area_l339_339649


namespace kayak_rental_cost_l339_339783

variable (K : ℕ) -- the cost of a kayak rental per day
variable (x : ℕ) -- the number of kayaks rented

-- Conditions
def canoe_cost_per_day : ℕ := 11
def total_revenue : ℕ := 460
def canoes_more_than_kayaks : ℕ := 5

def ratio_condition : Prop := 4 * x = 3 * (x + 5)
def total_revenue_condition : Prop := canoe_cost_per_day * (x + 5) + K * x = total_revenue

-- Main statement
theorem kayak_rental_cost :
  ratio_condition x →
  total_revenue_condition K x →
  K = 16 := by sorry

end kayak_rental_cost_l339_339783


namespace smallest_k_l339_339020

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l339_339020


namespace function_intersects_at_most_one_point_l339_339944

theorem function_intersects_at_most_one_point
  (f : ℝ → ℝ) (a : ℝ) :
  ∃! y, f a = y :=
begin
  sorry
end

end function_intersects_at_most_one_point_l339_339944


namespace largest_odd_integer_condition_solution_largest_odd_integer_is_105_l339_339867

theorem largest_odd_integer_condition (n : ℕ) (h1 : n % 2 = 1) (h2 : ∀ k : ℕ, 1 < k → k < n → k % 2 = 1 → gcd k n = 1 → prime k) :
  n ≤ 105 := sorry

theorem solution_largest_odd_integer_is_105 :
  ∃ n : ℕ, n = 105 ∧ n % 2 = 1 ∧ (∀ k : ℕ, 1 < k → k < n → k % 2 = 1 → gcd k n = 1 → prime k) :=
begin
  use 105,
  split,
  { refl },
  split,
  { norm_num },
  {
    intros k hk1 hk2 hk3 hkg,
    sorry -- Proof step goes here
  }
end

end largest_odd_integer_condition_solution_largest_odd_integer_is_105_l339_339867


namespace number_of_six_digit_integers_formed_with_repetition_l339_339618

theorem number_of_six_digit_integers_formed_with_repetition :
  ∃ n : ℕ, n = 60 ∧ nat.factorial 6 / (nat.factorial 3 * nat.factorial 2 * nat.factorial 1) = n :=
begin
  use 60,
  split,
  { refl },
  { sorry }
end

end number_of_six_digit_integers_formed_with_repetition_l339_339618


namespace max_daily_sales_revenue_l339_339759

def P (t : ℕ) : ℕ :=
  if (0 < t ∧ t < 25) then t + 20
  else if (25 ≤ t ∧ t ≤ 30) then -t + 70
  else 0

def Q (t : ℕ) : ℕ := 
  if (0 < t ∧ t ≤ 30) then -t + 40
  else 0

def y (t : ℕ) : ℕ :=
  P t * Q t

theorem max_daily_sales_revenue :
  ∃ t : ℕ, (0 < t ∧ t ≤ 30) ∧ y t = 1125 ∧ (∀ t' : ℕ, (0 < t' ∧ t' ≤ 30) → y t' ≤ 1125) :=
sorry

end max_daily_sales_revenue_l339_339759


namespace smallest_k_for_divisibility_l339_339031

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l339_339031


namespace determine_x_l339_339825

theorem determine_x (x : ℝ) (h : x > 0) :
  dist (2, 2) (x, 6) = 10 → x = 2 + 2 * real.sqrt 21 :=
by
  sorry

end determine_x_l339_339825


namespace find_angle_PLQ_l339_339663

-- Define the angles and lines in the problem
def EF : Type := sorry
def GH : Type := sorry
def IJ : Type := sorry
def KL : Type := sorry
def MN : Type := sorry
def P : Prop := sorry
def Q : Prop := sorry

-- Define the given angles
def angle_JKL : ℝ := 70
def angle_KLP : ℝ := 55
def angle_PQL : ℝ := 95

-- Define parallel lines
def parallel (line1 line2 : Type) : Prop := sorry
def intersects (line1 line2 : Type) (point : Prop) : Prop := sorry

-- Given conditions
axiom EF_parallel_GH : parallel EF GH
axiom IJ_intersects_EF_at_K : intersects IJ EF (P = K)
axiom IJ_intersects_GH_at_L : intersects IJ GH (Q = L)
axiom MN_intersects_EF_at_P : intersects MN EF P
axiom MN_intersects_GH_at_Q : intersects MN GH Q
axiom angle_sum_JKL_KLP : angle_JKL + angle_KLP = 180

-- Main theorem to prove
theorem find_angle_PLQ : angle_KLP = 110 → 110 + angle_PQL = 180 → angle_JKL + 70 + angle_PQL = 360 → 95 - 275 = -180 → 70 - 205 = -135 → ∠ PLQ = 85 := sorry

end find_angle_PLQ_l339_339663


namespace factorial_expression_l339_339446

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339446


namespace seven_a_plus_three_b_equals_four_sevenths_l339_339516

noncomputable def g (x : ℝ) : ℝ := 7 * x - 4

axiom f_inv (x : ℝ) : ℝ
axiom f (x : ℝ) : ℝ

axiom h1 : ∀ x, g x = f_inv x - 5
axiom h2 : ∀ x, f x = a * x + b
axiom h3 : ∀ x, f_inv (f x) = x

theorem seven_a_plus_three_b_equals_four_sevenths :
  7 * a + 3 * b = 4 / 7 := 
by
  sorry

end seven_a_plus_three_b_equals_four_sevenths_l339_339516


namespace complement_intersection_l339_339107

open Set

-- Define the universal set I, and sets M and N
def I : Set Nat := {1, 2, 3, 4, 5}
def M : Set Nat := {1, 2, 3}
def N : Set Nat := {3, 4, 5}

-- Lean statement to prove the desired result
theorem complement_intersection : (I \ N) ∩ M = {1, 2} := by
  sorry

end complement_intersection_l339_339107


namespace fraction_habitable_surface_l339_339179

def fraction_exposed_land : ℚ := 3 / 8
def fraction_inhabitable_land : ℚ := 2 / 3

theorem fraction_habitable_surface :
  fraction_exposed_land * fraction_inhabitable_land = 1 / 4 := by
    -- proof steps omitted
    sorry

end fraction_habitable_surface_l339_339179


namespace part1_part2_l339_339098

open Real

noncomputable def f (x : ℝ) : ℝ := sin x * (cos x - sqrt 3 * sin x)

theorem part1 :
  ∃ T > 0, ∀ x : ℝ, f (x + T) = f x ∧ T = π := 
by
  sorry

theorem part2 :
  ∃ max min, 
  (∀ x ∈ Icc (-π/3) (5*π/12), f x ≤ max) ∧ 
  (∀ x ∈ Icc (-π/3) (5*π/12), min ≤ f x) ∧ 
  max = 1 - sqrt 3 / 2 ∧
  min = -sqrt 3 :=
by
  sorry

end part1_part2_l339_339098


namespace temperature_notation_l339_339153

-- Define what it means to denote temperatures in degrees Celsius
def denote_temperature (t : ℤ) : String :=
  if t < 0 then "-" ++ toString t ++ "°C"
  else if t > 0 then "+" ++ toString t ++ "°C"
  else toString t ++ "°C"

-- Theorem statement
theorem temperature_notation (t : ℤ) (ht : t = 2) : denote_temperature t = "+2°C" :=
by
  -- Proof goes here
  sorry

end temperature_notation_l339_339153


namespace quadratic_inequality_solution_l339_339760

theorem quadratic_inequality_solution (a b : ℝ)
  (h : ∀ x : ℝ, -1/2 < x ∧ x < 1/3 → ax^2 + bx + 2 > 0) :
  a + b = -14 :=
sorry

end quadratic_inequality_solution_l339_339760


namespace Niraek_donut_holes_count_l339_339847

/-- Represents the radius of each worker's donut holes --/
def Niraek_radius : ℝ := 5
def Theo_radius : ℝ := 9
def Akshaj_radius : ℝ := 11

/-- Calculate surface area of a sphere --/
def surface_area (r : ℝ) : ℝ := 4 * real.pi * r * r

/-- Calculate the LCM of three numbers --/
def lcm_three (a b c : ℕ) : ℕ := Nat.lcm a (Nat.lcm b c)

theorem Niraek_donut_holes_count :
  ∀ (r1 r2 r3 : ℝ),
    r1 = Niraek_radius →
    r2 = Theo_radius →
    r3 = Akshaj_radius →
    (lcm_three (surface_area r1).nat_abs (surface_area r2).nat_abs (surface_area r3).nat_abs) / (surface_area r1).nat_abs = 10659 :=
by
  intros r1 r2 r3 hr1 hr2 hr3
  rw [← hr1, ← hr2, ← hr3]
  sorry

end Niraek_donut_holes_count_l339_339847


namespace artificial_scarcity_strategy_interview_strategy_l339_339737

-- Conditions for Part (a)
variable (high_end_goods : Type)      -- Type representing high-end goods
variable (manufacturers : high_end_goods → Prop)  -- Property of being a manufacturer
variable (sufficient_resources : high_end_goods → Prop) -- Sufficient resources to produce more
variable (demand : high_end_goods → ℤ)   -- Demand for the product
variable (supply : high_end_goods → ℤ)   -- Supply of the product 
variable (price : ℤ)   -- Price of the product

-- Theorem for Part (a)
theorem artificial_scarcity_strategy (H1 : ∀ g : high_end_goods, manufacturers g → sufficient_resources g → demand g = 3000 ∧ supply g = 200 ∧ price = 15000) :
  ∀ g : high_end_goods, manufacturers g → maintain_exclusivity g :=
sorry

-- Conditions for Part (b)
variable (interview_required : high_end_goods → Prop)   -- Interview requirement for purchase
variable (purchase_history : Prop)  -- Previous purchase history

-- Advantages and Disadvantages from Part (b)
def selective_clientele : Prop := sorry   -- Definition of selective clientele
def enhanced_exclusivity : Prop := sorry   -- Definition of enhanced exclusivity
def increased_transaction_costs : Prop := sorry   -- Definition of increased transaction costs

-- Theorem for Part (b)
theorem interview_strategy (H2 : ∀ g : high_end_goods, manufacturers g → interview_required g → purchase_history) :
  (selective_clientele ∧ enhanced_exclusivity) ∧ increased_transaction_costs :=
sorry

end artificial_scarcity_strategy_interview_strategy_l339_339737


namespace sunscreen_cost_l339_339681

variable (x : ℝ)  -- Denote the cost of each bottle of sunscreen before the discount.

-- Conditions
def one_bottle_per_month : Prop := True  -- Juanita goes through 1 bottle of sunscreen a month (trivial)
def bottles_per_year : Prop := 12 * x = 12 * x  -- She buys 12 bottles.
def discount_rate : Prop := 0.30 = 0.30  -- 30% discount.
def cost_after_discount : Prop := 0.70 * 12 * x = 252  -- Cost after the discount is $252.

-- Theorem stating the cost of each bottle before the discount.
theorem sunscreen_cost (h1 : one_bottle_per_month) 
                       (h2 : bottles_per_year) 
                       (h3 : discount_rate) 
                       (h4 : cost_after_discount) :
                       x = 30 := 
by sorry

end sunscreen_cost_l339_339681


namespace factorial_div_sum_l339_339426

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339426


namespace count_integers_between_sqrt8_sqrt75_l339_339112

theorem count_integers_between_sqrt8_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ ∀ i : ℤ, (3 ≤ i ∧ i ≤ 8) ↔ (sqrt 8 < i ∧ i < sqrt 75) :=
by
  use 6
  sorry

end count_integers_between_sqrt8_sqrt75_l339_339112


namespace ratio_is_five_to_one_l339_339282

noncomputable def ratio_of_numbers (A B : ℕ) : ℚ :=
  A / B

theorem ratio_is_five_to_one (A B : ℕ) (hA : A = 20) (hLCM : Nat.lcm A B = 80) : ratio_of_numbers A B = 5 := by
  -- Proof omitted
  sorry

end ratio_is_five_to_one_l339_339282


namespace magnitude_of_c_l339_339587

variables (a b : EuclideanSpace ℝ 3)
variables (ha : ‖a‖ = 1) (hb : ‖b‖ = 2) 
variables (angle_ab : real.angle_between a b = 2 * real.pi / 3)
noncomputable def c := 2 • a - b

theorem magnitude_of_c : ‖c a b‖ = 2 * real.sqrt 3 :=
begin
  sorry
end

end magnitude_of_c_l339_339587


namespace intersection_sets_m_n_l339_339106

theorem intersection_sets_m_n :
  let M := { x : ℝ | (2 - x) / (x + 1) ≥ 0 }
  let N := { x : ℝ | x > 0 }
  M ∩ N = { x : ℝ | 0 < x ∧ x ≤ 2 } :=
by
  sorry

end intersection_sets_m_n_l339_339106


namespace sum_first_10_common_elements_l339_339885

theorem sum_first_10_common_elements :
  let AP := λ n : ℕ, 4 + 3 * n
  let GP := λ k : ℕ, 10 * 2^k
  let common_elements k := 10 * 4^k
  (Σ i in Finset.range 10, common_elements i) = 3495250 := by
  sorry

end sum_first_10_common_elements_l339_339885


namespace evaluate_f_at_7_l339_339074

theorem evaluate_f_at_7 (f : ℝ → ℝ)
  (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, f (x + 4) = f x)
  (h3 : ∀ x : ℝ, 0 < x ∧ x < 2 → f x = -x + 4) :
  f 7 = -3 :=
by
  sorry

end evaluate_f_at_7_l339_339074


namespace find_m_n_sum_l339_339871

universe u

-- Definitions based on the problem conditions
def U : Type := sorry -- Equilateral triangle
def P : Type := { radius : ℝ } -- Circle with given radius
def Q : P -- Circle Q with radius 4 internally tangent to P
def R : P -- Circle R with radius 3 internally tangent to P
def S : P -- Circle S with radius 3 internally tangent to P
def T : P -- Circle T externally tangent to Q, R, S

-- Specific properties
axiom P_radius : (P → { r : ℝ // r = 12 })
axiom Q_radius : (Q → { r : ℝ // r = 4 })
axiom R_radius : (R → { r : ℝ // r = 3 })
axiom S_radius : (S → { r : ℝ // r = 3 })
axiom T_radius : (T → { r : ℝ // r = (45 / 4) + 4 })

-- Theorem to prove
theorem find_m_n_sum : let T_r := (45 / 4) + 4 in
  ∃ (m' n' : ℕ), m' + n' = 65 ∧ m' ≠ 0 ∧ n' ≠ 0 ∧ gcd m' n' = 1 :=
by sorry

end find_m_n_sum_l339_339871


namespace ordering_of_a_b_c_l339_339906

theorem ordering_of_a_b_c (a b c : ℝ) (h₁ : a = 2 ^ 0.2) (h₂ : b = 0.4 ^ 0.2) (h₃ : c = 0.4 ^ 0.6) : a > b ∧ b > c := by
  sorry

end ordering_of_a_b_c_l339_339906


namespace factorial_div_l339_339491

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339491


namespace min_value_frac_sum_l339_339292

theorem min_value_frac_sum 
  (A : ℝ × ℝ)
  (H1 : ∀ λ : ℝ, (λ + 1) * (A.1) - (λ + 2) * (A.2) + λ = 0)
  (H2 : ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ m * (A.1) + n * (A.2) + 2 = 0) : 
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ ((2 / m) + (1 / n) = 9 / 2) :=
by
  sorry

end min_value_frac_sum_l339_339292


namespace larger_integer_is_neg4_l339_339966

-- Definitions of the integers used in the problem
variables (x y : ℤ)

-- Conditions given in the problem
def condition1 : x + y = -9 := sorry
def condition2 : x - y = 1 := sorry

-- The theorem to prove
theorem larger_integer_is_neg4 (h1 : x + y = -9) (h2 : x - y = 1) : x = -4 := 
sorry

end larger_integer_is_neg4_l339_339966


namespace proof_problem_l339_339954

def RealSets (A B : Set ℝ) : Set ℝ :=
let complementA := {x | -2 < x ∧ x < 3}
let unionAB := complementA ∪ B
unionAB

theorem proof_problem :
  let A := {x : ℝ | (x + 2) * (x - 3) ≥ 0}
  let B := {x : ℝ | x > 1}
  let complementA := {x : ℝ | -2 < x ∧ x < 3}
  let unionAB := complementA ∪ B
  unionAB = {x : ℝ | x > -2} :=
by
  sorry

end proof_problem_l339_339954


namespace count_integers_between_sqrt8_and_sqrt75_l339_339118

theorem count_integers_between_sqrt8_and_sqrt75 : 
  let n1 := Real.sqrt 8,
      n2 := Real.sqrt 75 in 
  ∃ count : ℕ, count = (Int.floor n2 - Int.ceil n1) + 1 ∧ count = 6 :=
by
  let n1 := Real.sqrt 8
  let n2 := Real.sqrt 75
  use (Int.floor n2 - Int.ceil n1 + 1)
  simp only [Real.sqrt]
  sorry

end count_integers_between_sqrt8_and_sqrt75_l339_339118


namespace min_dist_between_curves_l339_339541

theorem min_dist_between_curves:
  let P : ℝ × ℝ := (ln 2, 1) in
  let dist_to_line := |ln 2 - 1| / sqrt 2 in
  ∀ (Q : ℝ × ℝ), Q.2 = ln (2 * Q.1) →
  Q.2 = x → 
  dist_to_line = (sqrt 2 * (1 - ln 2)) :=
begin
  sorry
end

end min_dist_between_curves_l339_339541


namespace probability_of_triangle_sides_l339_339369

open Real

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

noncomputable def calculate_probability : ℝ :=
  let valid_x (x : ℝ) : Prop :=
    0 < x ∧ x < 10 ∧ is_valid_triangle (sqrt x) (sqrt (x + 7)) (sqrt (10 - x))
  let probability :=
    ((∫ x in (0:ℝ)..10, indicator valid_x 1) / 10)
  in probability

theorem probability_of_triangle_sides : calculate_probability = 22 / 25 :=
  sorry

end probability_of_triangle_sides_l339_339369


namespace incorrect_conclusion_of_quadratic_function_l339_339555

theorem incorrect_conclusion_of_quadratic_function :
  let f := λ x : ℝ, (x - 2)^2 + 6 in
  ¬(f 0 = 6) :=
by
  let f := λ x : ℝ, (x - 2)^2 + 6
  have h1 : f 0 = 10 := by simp [f]
  have h2 : ¬(10 = 6) := by norm_num
  exact h2 (eq.symm h1)

end incorrect_conclusion_of_quadratic_function_l339_339555


namespace probability_sequence_correct_l339_339317

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l339_339317


namespace inequality_solution_l339_339091

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem inequality_solution (a b : ℝ) 
  (h1 : ∀ (x : ℝ), f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ (x : ℝ), f a b (-2 * x) < 0 ↔ x < -3 / 2 ∨ x > 1 / 2 :=
sorry

end inequality_solution_l339_339091


namespace problem_a_problem_b_l339_339894

-- Conditions: Definition of m(n)
-- m(n) is the minimum number of elements of a set S such that:
-- (i) {1, n} ⊆ S ⊆ {1, 2, ..., n}
-- (ii) Any element of S, distinct from 1, is equal to the sum of two (not necessarily distinct) elements from S.

def m (n : ℕ) : ℕ := sorry

-- Problem (a):
theorem problem_a (n : ℕ) (h : n ≥ 2) : m n ≥ 1 + Nat.log2 n :=
sorry

-- Problem (b):
theorem problem_b : ∃ᶠ n in at_top, m n = m (n + 1) ∧ n ≥ 2 :=
sorry

end problem_a_problem_b_l339_339894


namespace smallest_k_for_divisibility_l339_339029

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l339_339029


namespace factorial_computation_l339_339472

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339472


namespace range_of_a_l339_339952

noncomputable def A : Set ℝ := {x | x^2 ≤ 1}
noncomputable def B (a : ℝ) : Set ℝ := {x | x ≤ a}

theorem range_of_a (a : ℝ) (h : A ∪ B a = B a) : a ≥ 1 := 
by
  sorry

end range_of_a_l339_339952


namespace unique_plants_in_beds_l339_339645

theorem unique_plants_in_beds :
  let X := 600
  let Y := 500
  let Z := 400
  let XY := 80
  let XZ := 120
  let YZ := 70
  let XYZ := 0
  X + Y + Z - XY - XZ - YZ + XYZ = 1230 := by
{
  -- Define the values from the conditions
  let X := 600
  let Y := 500
  let Z := 400
  let XY := 80
  let XZ := 120
  let YZ := 70
  let XYZ := 0
  
  -- Calculate the unique plants using inclusion-exclusion principle
  have h : X + Y + Z - XY - XZ - YZ + XYZ = 1230,
  { simp [X, Y, Z, XY, XZ, YZ, XYZ] },
  exact h,
}

end unique_plants_in_beds_l339_339645


namespace temperature_notation_l339_339154

-- Define what it means to denote temperatures in degrees Celsius
def denote_temperature (t : ℤ) : String :=
  if t < 0 then "-" ++ toString t ++ "°C"
  else if t > 0 then "+" ++ toString t ++ "°C"
  else toString t ++ "°C"

-- Theorem statement
theorem temperature_notation (t : ℤ) (ht : t = 2) : denote_temperature t = "+2°C" :=
by
  -- Proof goes here
  sorry

end temperature_notation_l339_339154


namespace number_of_integers_between_sqrts_l339_339132

theorem number_of_integers_between_sqrts :
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  (upper_bound - lower_bound + 1) = 6 :=
by
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  have sqrt8_bound : (2 : ℝ) < sqrt8 ∧ sqrt8 < 3 := sorry
  have sqrt75_bound : (8 : ℝ) < sqrt75 ∧ sqrt75 < 9 := sorry
  have lower_bound_def : lower_bound = 3 := sorry
  have upper_bound_def : upper_bound = 8 := sorry
  show (upper_bound - lower_bound + 1) = 6
  calc
    upper_bound - lower_bound + 1 = 8 - 3 + 1 := by rw [upper_bound_def, lower_bound_def]
                             ... = 6 := by norm_num
  sorry

end number_of_integers_between_sqrts_l339_339132


namespace susie_large_rooms_count_l339_339181

theorem susie_large_rooms_count:
  (∀ small_rooms medium_rooms large_rooms : ℕ,  
    (small_rooms = 4) → 
    (medium_rooms = 3) → 
    (large_rooms = x) → 
    (225 = small_rooms * 15 + medium_rooms * 25 + large_rooms * 35) → 
    x = 2) :=
by
  intros small_rooms medium_rooms large_rooms
  intros h1 h2 h3 h4
  sorry

end susie_large_rooms_count_l339_339181


namespace union_A_B_intersection_complements_l339_339688
open Set

noncomputable def A : Set ℤ := {x | x^2 - 5 * x + 6 = 0}
noncomputable def B : Set ℤ := {x | x^2 - x - 2 = 0}
def U : Set ℤ := {x | abs x ≤ 3}

theorem union_A_B :
  A ∪ B = { -1, 2, 3 } :=
by sorry

theorem intersection_complements :
  (U \ A) ∩ (U \ B) = { -3, -2, 0, 1 } :=
by sorry

end union_A_B_intersection_complements_l339_339688


namespace ordered_pair_sqrt_eq_l339_339007

theorem ordered_pair_sqrt_eq {a b : ℕ} (h_pos : 0 < a ∧ 0 < b) (h_lt : a < b) :
  sqrt (1 + sqrt (27 + 18 * sqrt 3)) = sqrt a + sqrt b ↔ (a = 1 ∧ b = 3) := 
sorry

end ordered_pair_sqrt_eq_l339_339007


namespace binomial_x2_coefficient_l339_339598

theorem binomial_x2_coefficient :
  let expr := (2 * x - x^(-1/3))^6
  in coefficient expr 2 = -160 :=
sorry

end binomial_x2_coefficient_l339_339598


namespace digit_deletion_properties_l339_339987

theorem digit_deletion_properties (N : ℕ) (m M d : ℕ) (h_N : N = 1234567891011 /*...*/ 9899100) 
  (h_digits_N : N.to_digits.length = 192) 
  (h_m : m = /* smallest 92-digit number from N */) 
  (h_M : M = /* largest 92-digit number from N */) 
  (h_M_digits : M.to_digits.length = 92)
  (h_m_digits : m.to_digits.length = 92) 
  (h_d : d = M - m) :
  (d % 24 = 0) ∧
  (exists m_M_mean, m_M_mean = (m + M) / 2 ∧ /* m_M_mean valid from N */) ∧
  (exists M_d6, M_d6 = M - d / 6 ∧ /* M_d6 valid from N */) ∧
  (exists m_d8, m_d8 = m + d / 8 ∧ /* m_d8 valid from N */) ∧
  (exists M_d8, M_d8 = M - d / 8 ∧ /* M_d8 valid from N */) :=
by
  sorry

end digit_deletion_properties_l339_339987


namespace rectangle_ratio_l339_339370

noncomputable def ratio_of_sides (a b : ℝ) : ℝ := a / b

theorem rectangle_ratio (a b d : ℝ) (h1 : d = Real.sqrt (a^2 + b^2)) (h2 : (a/b)^2 = b/d) : 
  ratio_of_sides a b = (Real.sqrt 5 - 1) / 3 :=
by sorry

end rectangle_ratio_l339_339370


namespace order_of_t_l339_339909

noncomputable def t1 (α : ℝ) := (Real.tan α) ^ (Real.tan α)
noncomputable def t2 (α : ℝ) := (Real.tan α) ^ (Real.cos α)
noncomputable def t3 (α : ℝ) := (Real.cot α) ^ (Real.sin α)
noncomputable def t4 (α : ℝ) := (Real.cot α) ^ (Real.cot α)

theorem order_of_t (α : ℝ) (hα1 : 0 < α) (hα2 : α < Real.pi / 4) :
  t2 α < t1 α ∧ t1 α < t3 α ∧ t3 α < t4 α :=
by
  sorry

end order_of_t_l339_339909


namespace reflection_invariance_l339_339988

-- Define points in a three-dimensional space
structure Point :=
(x y z : ℝ)

-- Define reflection around a point
def reflect (O A : Point) : Point :=
{ x := 2 * O.x - A.x, 
  y := 2 * O.y - A.y, 
  z := 2 * O.z - A.z }

-- Define the sequence of reflections
def reflect_sequence (O1 O2 O3 A : Point) : Point :=
let A1 := reflect O1 A in
let A2 := reflect O2 A1 in
let A3 := reflect O3 A2 in
let A4 := reflect O1 A3 in
let A5 := reflect O2 A4 in
reflect O3 A5

-- The final theorem to prove: after the sequence of reflections, the point returns to A
theorem reflection_invariance (O1 O2 O3 A : Point) :
  reflect_sequence O1 O2 O3 A = A :=
sorry

end reflection_invariance_l339_339988


namespace no_positive_integer_k_exists_l339_339522

theorem no_positive_integer_k_exists :
  ∀ (k : ℕ), 0 < k → 
  let p := 6 * k + 1 in 
  Nat.Prime p → (Nat.choose (3 * k) k) % p = 1 → false := 
by
  intro k hk p h_prime h_choose
  let p := 6 * k + 1
  assume h_prime : Nat.Prime p 
  assume h_choose : (Nat.choose (3 * k) k) % p = 1
  sorry

end no_positive_integer_k_exists_l339_339522


namespace binary_multiplication_l339_339010

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end binary_multiplication_l339_339010


namespace smallest_n_equal_sums_l339_339224

def sum_first_n_arithmetic (a d n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

theorem smallest_n_equal_sums : ∀ (n : ℕ), 
  sum_first_n_arithmetic 7 4 n = sum_first_n_arithmetic 15 3 n → n ≠ 0 → n = 7 := by
  intros n h1 h2
  sorry

end smallest_n_equal_sums_l339_339224


namespace min_sum_a_b_l339_339572

theorem min_sum_a_b (a b : ℕ) (h1 : a ≠ b) (h2 : 0 < a ∧ 0 < b) (h3 : (1/a + 1/b) = 1/12) : a + b = 54 :=
sorry

end min_sum_a_b_l339_339572


namespace remainder_division_l339_339881

def f (x : ℝ) : ℝ := x^3 - 4 * x + 7

theorem remainder_division (x : ℝ) : f 3 = 22 := by
  sorry

end remainder_division_l339_339881


namespace factorial_computation_l339_339476

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339476


namespace solution_to_problem_l339_339095

variable (a : ℝ)

def f (x : ℝ) : ℝ := ((x + 1)^2 + a * sin x) / (x^2 + 1) + 3

theorem solution_to_problem (h : f (Real.log (Real.log 5 / Real.log 2)) = 5) :
  f (Real.log (Real.log 2 / Real.log 5)) = 3 :=
sorry

end solution_to_problem_l339_339095


namespace sum_of_two_digit_divisors_of_125_l339_339223

theorem sum_of_two_digit_divisors_of_125 :
  let d : ℕ := 125,
  ∑ i in {n | n ∣ d ∧ 10 ≤ n ∧ n < 100}, i = 25 :=
by
  exact sum_of_two_digit_divisors_of_125

end sum_of_two_digit_divisors_of_125_l339_339223


namespace find_x_log_l339_339875

theorem find_x_log (x : ℝ) (h : log 10 (5 * x) = 3) : x = 200 :=
by sorry

end find_x_log_l339_339875


namespace a_4_is_54_l339_339201

open Function

/-- Define the sequence a_n using the given recursive formula. -/
def a : ℕ → ℕ
| 0     := 2
| (n+1) := 3 * a n

/-- Statement to prove that the fourth term in the sequence is 54. -/
theorem a_4_is_54 : a 4 = 54 :=
by
  sorry

end a_4_is_54_l339_339201


namespace sequence_1729th_term_l339_339283

noncomputable def digit_cubes_sum (n : ℕ) : ℕ :=
(n.digits 10).map (λ x => x^3).sum

def sequence (a : ℕ) : ℕ → ℕ
| 0 => a
| (n+1) => digit_cubes_sum (sequence n)

theorem sequence_1729th_term :
  sequence 1729 1728 = 370 :=
by
  sorry

end sequence_1729th_term_l339_339283


namespace domain_of_f_l339_339536

noncomputable def f (x : ℝ) : ℝ := Real.log(1 - 2 * Real.sin x) + Real.sqrt((2 * Real.pi - x) / x)

def domain_condition_1 (x : ℝ) : Prop := 1 - 2 * Real.sin x > 0

def domain_condition_2 (x : ℝ) : Prop := (2 * Real.pi - x) / x ≥ 0

def domain (x : ℝ) : Prop := (0 < x ∧ x < Real.pi / 6) ∨ (5 * Real.pi / 6 < x ∧ x ≤ 2 * Real.pi)

theorem domain_of_f : ∀ x : ℝ, domain_condition_1 x ∧ domain_condition_2 x ↔ domain x :=
by
  sorry

end domain_of_f_l339_339536


namespace part_a_part_b_l339_339861

-- Definition of set A
def A (k : ℕ) : ℝ := 1 + 1 / (k : ℝ)

-- Definition of f(x)
def f (x : ℕ) : ℕ := sorry -- f(x) is minimal number of elements products from A give x.

-- Part (a)
theorem part_a (x : ℕ) (h : x ≥ 2) : 
  ∃ (l : list ℝ), (∀ (a ∈ l), ∃ k, A k = a) ∧ x = l.prod := sorry

-- Part (b)
theorem part_b : ∃ (x y : ℕ), (x ≥ 2) ∧ (y ≥ 2) ∧ f (x * y) < f x + f y := sorry

end part_a_part_b_l339_339861


namespace determine_k_l339_339686

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x + 1

theorem determine_k : ∃ k : ℝ, (∀ x : ℝ, f(x + k) = x^2 + 2*x + 1) ∧ k = 2 :=
begin
  have f_xk := λ k x, f (x + k) = (x + k - 1) ^ 2,
  -- The proof steps will go here
  sorry
end

end determine_k_l339_339686


namespace correct_operation_l339_339793

variable (a b : ℝ)

theorem correct_operation :
  ¬ (a^2 + a^3 = a^5) ∧
  ¬ ((a^2)^3 = a^5) ∧
  ¬ (a^2 * a^3 = a^6) ∧
  ((-a * b)^5 / (-a * b)^3 = a^2 * b^2) :=
by
  sorry

end correct_operation_l339_339793


namespace water_displaced_volume_square_l339_339358

-- Given conditions:
def radius : ℝ := 5
def height : ℝ := 10
def cube_side : ℝ := 6

-- Theorem statement for the problem
theorem water_displaced_volume_square (r h s : ℝ) (w : ℝ) 
  (hr : r = 5) 
  (hh : h = 10) 
  (hs : s = 6) : 
  (w * w) = 13141.855 :=
by 
  sorry

end water_displaced_volume_square_l339_339358


namespace card_sequence_probability_l339_339314

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l339_339314


namespace range_a_l339_339359

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f(x) = f(-x)

def periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f(x + p) = f(x)

theorem range_a (f : ℝ → ℝ) (a : ℝ) :
  even_function f ∧
  (∀ x, f(x + 2) = f(x) + f(1)) ∧
  f(0) = -1 ∧
  (∀ x, x ∈ Ioo (-1) 0 → deriv f x < 0) ∧
  (∃ x1 x2 x3, [x1, x2, x3].pairwise (≠) ∧ [x1, x2, x3].all (λ x, x ∈ Icc a 3 ∧ f(x) = 0))
  → a ∈ Ioo (-3) (-1) ∨ a = -1 :=
sorry

end range_a_l339_339359


namespace problem_solution_l339_339963

theorem problem_solution (a d e : ℕ) (ha : 0 < a ∧ a < 10) (hd : 0 < d ∧ d < 10) (he : 0 < e ∧ e < 10) :
  ((10 * a + d) * (10 * a + e) = 100 * a ^ 2 + 110 * a + d * e) ↔ (d + e = 11) := by
  sorry

end problem_solution_l339_339963


namespace range_of_f_l339_339519

def f (x : ℝ) : ℝ := x^2 - 4 * x + 2

theorem range_of_f : set.range (λ x : ℝ, f x) = set.Icc (-2 : ℝ) (7 : ℝ) :=
by
  sorry

end range_of_f_l339_339519


namespace inscribed_circle_radii_l339_339743

theorem inscribed_circle_radii (R : ℝ) (hR : 0 < R) :
  let r_minor := R / 4,
      r_major := 3 * R / 4,
      θ := 120
  in ∀ (AB chord : ℝ) (hAB : 0 < AB) (hθ : θ = 120),
    True → (r_minor = R / 4 ∧ r_major = 3 * R / 4) := 
by
  intros
  simp only [true_and]
  exact sorry

end inscribed_circle_radii_l339_339743


namespace modulus_of_z_l339_339928

def imaginary_unit := Complex.I
def z := (1 - imaginary_unit) / (1 + imaginary_unit)

theorem modulus_of_z : Complex.abs z = 1 :=
  sorry

end modulus_of_z_l339_339928


namespace factorial_division_sum_l339_339465

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339465


namespace polynomial_count_l339_339868

theorem polynomial_count :
  let count_possibilities (n : ℕ) (coeffs : list ℤ) := 
    coeffs.length = n + 1 ∧ n + (coeffs.map Int.natAbs).sum = 5 in
  ∃ p : ℕ → list ℤ → ℕ, p = λ n coeffs, if count_possibilities n coeffs then 1 else 0 ∧ 
    (list.range 6).sum (λ n, list.sum (list.map (p n) (list.finSubsetsLen (n + 1)))) = 19 := 
by
  sorry

end polynomial_count_l339_339868


namespace log_base_change_l339_339158

theorem log_base_change (x : ℝ) (h : Real.logBase 49 (x - 6) = 1 / 2) : 
  1 / Real.logBase x 5 = Real.log 13 / Real.log 5 := 
by 
  sorry

end log_base_change_l339_339158


namespace sequence_recurrence_sum_first_100_terms_l339_339917

def sequence (n : ℕ) : ℤ :=
  if n = 0 then 2018 else         -- Define an artificial a_0
  if n = 1 then 2017 else
  (sequence (n - 1)) - (sequence (n - 2))

def S (n : ℕ) : ℤ := 
  (Finset.range n).sum (λ k, sequence k)

theorem sequence_recurrence :
  (∀ n, 2 ≤ n → sequence n = sequence (n - 1) - sequence (n - 2)) ∧ (sequence 0 = 2018) ∧ (sequence 1 = 2017) := by
  split
  { intros n hn,
    unfold sequence,
    simp only [if_pos (Nat.one_lt_succ_succ.mp hn)] }
  split
  { rfl }
  { rfl }

theorem sum_first_100_terms :
  S 100 = 2016 := by
  sorry

end sequence_recurrence_sum_first_100_terms_l339_339917


namespace problem_solution_l339_339172

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l339_339172


namespace sin_minus_pi_over_3_eq_neg_four_fifths_l339_339627

theorem sin_minus_pi_over_3_eq_neg_four_fifths
  (α : ℝ)
  (h : Real.cos (α + π / 6) = 4 / 5) :
  Real.sin (α - π / 3) = - (4 / 5) :=
by
  sorry

end sin_minus_pi_over_3_eq_neg_four_fifths_l339_339627


namespace triangle_inequality_l339_339668

variable {a b c r R : ℝ}

theorem triangle_inequality
  {Δ s : ℝ} (hΔ : Δ = sqrt (s * (s - a) * (s - b) * (s - c))) (hr : r = Δ / s) (hR : R = a * b * c / (4 * Δ))
  (h_a : a > 0) (h_b : b > 0) (h_c : c > 0) (h_r : r > 0) (h_R : R > 0) :
  1 / (2 * r * R) ≤ 1 / 3 * (1 / a + 1 / b + 1 / c) ^ 2 ∧
  1 / 3 * (1 / a + 1 / b + 1 / c) ^ 2 ≤ 1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 ∧
  1 / a ^ 2 + 1 / b ^ 2 + 1 / c ^ 2 ≤ 1 / (4 * r ^ 2) := sorry

end triangle_inequality_l339_339668


namespace fundraiser_successful_l339_339547

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end fundraiser_successful_l339_339547


namespace angle_in_third_quadrant_l339_339626

-- Definitions for quadrants
def in_fourth_quadrant (α : ℝ) : Prop := 270 < α ∧ α < 360
def in_third_quadrant (β : ℝ) : Prop := 180 < β ∧ β < 270

theorem angle_in_third_quadrant (α : ℝ) (h : in_fourth_quadrant α) : in_third_quadrant (180 - α) :=
by
  -- Proof goes here
  sorry

end angle_in_third_quadrant_l339_339626


namespace find_b_in_triangle_l339_339203

theorem find_b_in_triangle
  (A C : ℝ) (a c : ℝ) (h₁ : C = 2 * A) (h₂ : a = 34) (h₃ : c = 60) :
  ∃ b, b ≈ 9.67 := 
sorry

end find_b_in_triangle_l339_339203


namespace factorial_computation_l339_339478

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339478


namespace sin_BAD_over_sin_CAD_eq_sqrt_6_over_6_l339_339671

theorem sin_BAD_over_sin_CAD_eq_sqrt_6_over_6
  (A B C D : Type)
  [triangle : Triangle A B C]
  (hB : angle B = 60)
  (hC : angle C = 45)
  (hD : divides_segment D B C (2/3)) :

  (sin (angle A B D)) / (sin (angle A C D)) = (sqrt 6) / 6 := 
  sorry

end sin_BAD_over_sin_CAD_eq_sqrt_6_over_6_l339_339671


namespace factorial_div_l339_339492

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339492


namespace problem_solution_l339_339544

theorem problem_solution (n : ℝ) (h : (1 / (n + 1) + 2 / (n + 1) + n / (n + 1) + 1 / (n + 2) = 4)) :
  n = (-3 + Real.sqrt 6) / 3 ∨ n = (-3 - Real.sqrt 6) / 3 :=
sorry

end problem_solution_l339_339544


namespace sin_C_eq_cos_A_tan_B_l339_339207

theorem sin_C_eq_cos_A_tan_B
  (A B C res : ℝ)
  (intersect : ∃ O, 
      O ∈ (AA_1 : line) ∧ 
      O ∈ (BB_1 : line) ∧ 
      O ∈ (CC_1 : line)) :
  sin C = cos A * tan B :=
begin
  -- Proof proceeds as outlined, but it is not needed here.
  sorry
end

end sin_C_eq_cos_A_tan_B_l339_339207


namespace segments_not_equal_l339_339306

theorem segments_not_equal (n : ℕ) (hn : n = 2022) :
  (∃ (red_points blue_points : Finset ℕ), 
    red_points.card = n / 2 ∧ blue_points.card = n / 2 ∧
    ∀ (x y : ℕ), x ∈ red_points → y ∈ blue_points → x < y) →
  (let red_to_blue := ∑ x in red_points, ∑ y in blue_points, y - x,
       blue_to_red := ∑ y in blue_points, ∑ x in red_points, x - y in
   red_to_blue ≠ blue_to_red) :=
sorry

end segments_not_equal_l339_339306


namespace welders_correct_l339_339265

-- Define the initial number of welders
def initial_welders := 12

-- Define the conditions:
-- 1. Total work is 1 job that welders can finish in 3 days.
-- 2. 9 welders leave after the first day.
-- 3. The remaining work is completed by (initial_welders - 9) in 8 days.

theorem welders_correct (W : ℕ) (h1 : W * 1/3 = 1) (h2 : (W - 9) * 8 = 2 * W) : 
  W = initial_welders :=
by
  sorry

end welders_correct_l339_339265


namespace integer_count_between_sqrt8_and_sqrt75_l339_339140

noncomputable def m : ℤ := Int.ceil (Real.sqrt 8)
noncomputable def n : ℤ := Int.floor (Real.sqrt 75)

theorem integer_count_between_sqrt8_and_sqrt75 : (n - m + 1) = 6 := by
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339140


namespace measure_angle_QR_XZ_l339_339643

-- Define the conditions
variables (X Y Z B C Q R : Type*)
variables [NormedAddCommGroup X] [InnerProductSpace ℝ X] [NormedAddCommGroup Y] [InnerProductSpace ℝ Y]
variables [NormedAddCommGroup Z] [InnerProductSpace ℝ Z] [NormedAddCommGroup B] [InnerProductSpace ℝ B]
variables [NormedAddCommGroup C] [InnerProductSpace ℝ C] [NormedAddCommGroup Q] [InnerProductSpace ℝ Q]
variables [NormedAddCommGroup R] [InnerProductSpace ℝ R]

-- Conditions from problem in point a)
noncomputable def angle_X := 30
noncomputable def angle_Z := 74
noncomputable def length_XZ := 12
noncomputable def length_ZB := 2
noncomputable def length_BC := 2
noncomputable def midpoint_XQ := 6
noncomputable def midpoint_BR := 1

-- Proof statement of the problem in point c)
theorem measure_angle_QR_XZ :
  ∃ (Q R : X) (B C : X) (XZ_bar QR_bar : set (set X)),
  (angle_X = 30 ∧
  angle_Z = 74 ∧
  length_XZ = 12 ∧
  length_ZB = 2 ∧
  length_BC = 2 ∧
  midpoint_XQ = 6 ∧
  midpoint_BR = 1) →
  (angle (QR_bar) (XZ_bar) = 38) :=
sorry

end measure_angle_QR_XZ_l339_339643


namespace parallelogram_splits_line_slope_l339_339511

theorem parallelogram_splits_line_slope :
  let p := 16
  let q := 3
  let vertices := [(3, 25), (3, 64), (18, 87), (18, 48)]
  -- Condition: A line from the origin splits this figure into two congruent polygons
  (∃ m n : ℕ, nat.gcd m n = 1 ∧ m = p ∧ n = q) ∧ 
  (∃ slope : ℚ, slope = 16 / 3) ∧ (∃ pq_sum : ℕ, pq_sum = 19) :=
by {
  sorry
}

end parallelogram_splits_line_slope_l339_339511


namespace find_ellipse_constants_l339_339841

noncomputable def ellipse_constants (a b h k : ℝ) : Prop :=
  (a = 8 * Real.sqrt 2) ∧ (b = 8 * Real.sqrt 7) ∧ (h = 0) ∧ (k = 4)

theorem find_ellipse_constants :
  ∃ (a b h k : ℝ),
    ellipse_constants a b h k ∧
    let f1 := (0 : ℝ, 0 : ℝ) in
    let f2 := (0 : ℝ, 8 : ℝ) in
    let p := (7 : ℝ, 4 : ℝ) in
    dist p f1 + dist p f2 = 16 * Real.sqrt 2 ∧
    dist f1 f2 = 8 ∧
    (h = 0 ∧ k = 4) :=
begin
  use [8 * Real.sqrt 2, 8 * Real.sqrt 7, 0, 4],
  split,
  { unfold ellipse_constants, simp, },
  split,
  { simp, },
  split,
  { simp, },
  { simp, }
end

end find_ellipse_constants_l339_339841


namespace existence_of_subset_A_l339_339908

def M : Set ℚ := {x : ℚ | 0 < x ∧ x < 1}

theorem existence_of_subset_A :
  ∃ A ⊆ M, ∀ m ∈ M, ∃! (S : Finset ℚ), (∀ a ∈ S, a ∈ A) ∧ (S.sum id = m) :=
sorry

end existence_of_subset_A_l339_339908


namespace cantor_set_cardinality_same_as_interval_cantor_set_sum_difference_l339_339217

noncomputable def cantor_set : set ℝ := {x | ∀n:ℕ, (x ∈ ((⋃ i, (0, (1/3)^n] : set ℝ)) ∪ (⋃ i, [(2/3) * (i^n), 1])) }

theorem cantor_set_cardinality_same_as_interval : cardinality cantor_set = cardinality (Icc 0 1) := 
by sorry

theorem cantor_set_sum_difference : 
  (⨆ {x y : ℝ} (hx : x ∈ cantor_set) (hy : y ∈ cantor_set), (x + y ∈ Icc 0 2))
  ∧ (⨆ {x y : ℝ} (hx : x ∈ cantor_set) (hy : y ∈ cantor_set), (x - y ∈ Icc (-1) 1)) :=
by sorry

end cantor_set_cardinality_same_as_interval_cantor_set_sum_difference_l339_339217


namespace factorial_div_sum_l339_339434

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339434


namespace factorial_div_sum_l339_339424

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339424


namespace max_surface_area_l339_339208

theorem max_surface_area (x y z : ℝ) (h₁ : x * y = 1) (h₂ : x + y + z = 5) : 
  ∃ (x y z : ℝ), x = 2 ∧ y = 1 / 2 ∧ z = 5 / 2 :=
begin
  sorry
end

end max_surface_area_l339_339208


namespace factorial_sum_division_l339_339456

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339456


namespace time_after_increment_l339_339676

def Time := (nat × nat × nat)

def initialTime : Time := (17, 45, 0)  -- 5:45:00 p.m. in 24-hour format

def timeIncrement : nat := 9999  -- seconds

def expectedTime : Time := (20, 31, 39)  -- 8:31:39 p.m. in 24-hour format

theorem time_after_increment : 
  let (h, m, s) := initialTime in
  let (dh, dm, ds) := expectedTime in
  let total_seconds := h * 3600 + m * 60 + s + timeIncrement in
  (total_seconds / 3600) % 24 = dh ∧ 
  (total_seconds / 60) % 60 = dm ∧
  total_seconds % 60 = ds :=
by 
  sorry

end time_after_increment_l339_339676


namespace sum_of_nonneg_real_numbers_inequality_l339_339230

open BigOperators

variables {α : Type*} [LinearOrderedField α]

theorem sum_of_nonneg_real_numbers_inequality 
  (a : ℕ → α) (n : ℕ)
  (h_nonneg : ∀ i : ℕ, 0 ≤ a i) : 
  (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j) * (∑ j in Finset.Icc i (n - 1), a j ^ 2))) 
  ≤ (∑ i in Finset.range n, (a i * (∑ j in Finset.range (i + 1), a j)) ^ 2) :=
sorry

end sum_of_nonneg_real_numbers_inequality_l339_339230


namespace no_minimum_value_l339_339096

noncomputable def f (x a : ℝ) := log a (x^2 + a * x + 4)

theorem no_minimum_value (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) : 
  (0 < a ∧ a < 1) ∨ (a ≥ 4) := sorry

end no_minimum_value_l339_339096


namespace hexagon_area_ratio_l339_339320

theorem hexagon_area_ratio (ABC : Triangle)
  (A1 A2 B1 B2 C1 C2 : Point)
  (h_eq_tri : equilateral ABC)
  (h_trisect_A : trisects (line_through A A1) (line_through A A2) (BC))
  (h_trisect_B : trisects (line_through B B1) (line_through B B2) (CA))
  (h_trisect_C : trisects (line_through C C1) (line_through C C2) (AB))
  : (area (hexagon_formed_by_lines ABC ([A1, A2], [B1, B2], [C1, C2])) / area ABC) = 1 / 10 := 
sorry

end hexagon_area_ratio_l339_339320


namespace triangles_on_circle_l339_339305

theorem triangles_on_circle (n : ℕ) (h : n ≥ 6) : 
  (∑ i in {3,4,5,6}, if i = 3 then (∑ x in finset.range i, nat.choose n i) else 
                     if i = 4 then 4 * (∑ x in finset.range i, nat.choose n i)
                     else
                     if i = 5 then 5 * (∑ x in finset.range i, nat.choose n i)
                     else (∑ x in finset.range i, nat.choose n i)) = (nat.choose n 3 + 4 * nat.choose n 4 + 5 * nat.choose n 5 + nat.choose n 6) := 
begin
  sorry
end

end triangles_on_circle_l339_339305


namespace factorial_division_l339_339416

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339416


namespace find_B_l339_339613

def A (a : ℝ) : Set ℝ := {3, Real.log a / Real.log 2}
def B (a b : ℝ) : Set ℝ := {a, b}

theorem find_B (a b : ℝ) (hA : A a = {3, 2}) (hB : B a b = {a, b}) (h : (A a) ∩ (B a b) = {2}) :
  B a b = {2, 4} :=
sorry

end find_B_l339_339613


namespace binary_multiplication_l339_339009

theorem binary_multiplication : (0b1101 * 0b111 = 0b1001111) :=
by {
  -- placeholder for proof
  sorry
}

end binary_multiplication_l339_339009


namespace smallest_k_l339_339035

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l339_339035


namespace part_I_solution_set_part_II_inequality_l339_339807

noncomputable theory

-- Translating the problem for Part I
def inequality_solution_set := { x : ℝ | x ≥ 1/2 ∨ x ≤ Real.log2 (Real.sqrt 2 - 1) }

theorem part_I_solution_set : 
  ∀ x : ℝ, 2^x + 2^(Real.abs x) ≥ 2 * Real.sqrt 2 ↔ x ∈ inequality_solution_set :=
sorry

-- Translating the problem for Part II
theorem part_II_inequality (a b : ℝ) (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (a^2 / m) + (b^2 / n) ≥ ((a + b)^2 / (m + n)) :=
sorry

end part_I_solution_set_part_II_inequality_l339_339807


namespace factorial_expression_l339_339442

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339442


namespace factorial_div_sum_l339_339435

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339435


namespace factorial_div_sum_l339_339422

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339422


namespace min_area_triangle_l339_339084

theorem min_area_triangle 
  (a b : ℝ) (ha : 1 < a) (hb : 1 < b) 
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 - 2*x - 2*y = -1)
  (h_tangent : ∀ (x y : ℝ), b * x + a * y - 2 * a * b = 0 → 
    |b * 1 + a * 1 - 2 * a * b| / sqrt (a^2 + b^2) = 1) :
  2 * a * b = 3 + 2 * sqrt 2 :=
by
  sorry

end min_area_triangle_l339_339084


namespace count_integers_between_sqrt8_and_sqrt75_l339_339120

theorem count_integers_between_sqrt8_and_sqrt75 : 
  let n1 := Real.sqrt 8,
      n2 := Real.sqrt 75 in 
  ∃ count : ℕ, count = (Int.floor n2 - Int.ceil n1) + 1 ∧ count = 6 :=
by
  let n1 := Real.sqrt 8
  let n2 := Real.sqrt 75
  use (Int.floor n2 - Int.ceil n1 + 1)
  simp only [Real.sqrt]
  sorry

end count_integers_between_sqrt8_and_sqrt75_l339_339120


namespace min_value_f_increasing_f_neg_half_pi_zero_decreasing_f_zero_half_pi_l339_339251

def f (x : ℝ) : ℝ := cos x ^ 2 + 4 * cos x + 1

theorem min_value_f : ∀ x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 2), f x ≥ -2 :=
begin
  sorry
end

theorem increasing_f_neg_half_pi_zero : ∀ x₁ x₂, 
  (-Real.pi / 2) < x₁ ∧ x₁ < x₂ ∧ x₂ < 0 → f x₁ ≤ f x₂ :=
begin
  sorry
end

theorem decreasing_f_zero_half_pi : ∀ x₁ x₂, 
  0 < x₁ ∧ x₁ < x₂ ∧ x₂ < Real.pi / 2 → f x₁ ≥ f x₂ :=
begin
  sorry
end

end min_value_f_increasing_f_neg_half_pi_zero_decreasing_f_zero_half_pi_l339_339251


namespace smallest_k_divides_l339_339022

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l339_339022


namespace function_properties_l339_339078

theorem function_properties (a : ℝ) (f : ℝ → ℝ) (h1 : f x = x^a) (h2 : f 8 = 4) : 
    (∀ x : ℝ, f x = x^(2/3)) ∧ 
    f 0 = 0 ∧ 
    (∀ x : ℝ, f (-x) = f x) ∧ 
    (Range f = {y | y ≥ 0}) ∧ 
    (∀ x : ℝ, x < 0 → f x ≥ f (x + 1)) := 
by
  sorry

end function_properties_l339_339078


namespace transform_into_product_l339_339805

theorem transform_into_product : 447 * (Real.sin (75 * Real.pi / 180) + Real.sin (15 * Real.pi / 180)) = 447 * Real.sqrt 6 / 2 := by
  sorry

end transform_into_product_l339_339805


namespace multiplication_verification_l339_339796

theorem multiplication_verification (x : ℕ) (h : 23 - x = 4) : 23 * x = 437 := by
  sorry

end multiplication_verification_l339_339796


namespace smallest_k_divides_l339_339027

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l339_339027


namespace incircle_center_line_midpoints_l339_339253

variable {A B C D M N O : Type}
variable [Geometry A B C]
variable [incircle_center O A B C] -- O is the center of the incircle of triangle ABC
variable [touches_incircle_side O A B C D] -- D is the point where the incircle touches side BC
variable [midpoint M B C] -- M is the midpoint of BC 
variable [midpoint N A D] -- N is the midpoint of AD

theorem incircle_center_line_midpoints :
  lies_on_line O M N :=
sorry

end incircle_center_line_midpoints_l339_339253


namespace integer_count_between_sqrt8_and_sqrt75_l339_339139

noncomputable def m : ℤ := Int.ceil (Real.sqrt 8)
noncomputable def n : ℤ := Int.floor (Real.sqrt 75)

theorem integer_count_between_sqrt8_and_sqrt75 : (n - m + 1) = 6 := by
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339139


namespace smallest_k_divides_l339_339023

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l339_339023


namespace acute_triangle_inequality_l339_339995

theorem acute_triangle_inequality {A B C : ℝ} (h1 : A + B + C = π) (h2 : 0 < A ∧ A < π / 2) (h3 : 0 < B ∧ B < π / 2) (h4 : 0 < C ∧ C < π / 2) :
  (cos A / (cos B * cos C) + cos B / (cos A * cos C) + cos C / (cos A * cos B)) ≥ 3 * (1 / (1 + cos A) + 1 / (1 + cos B) + 1 / (1 + cos C)) :=
sorry

end acute_triangle_inequality_l339_339995


namespace find_abc_l339_339160

noncomputable def abc_value (a b c : ℝ) : ℝ := a * b * c

theorem find_abc 
  (a b c : ℝ) 
  (h_pos_a : 0 < a)
  (h_pos_b : 0 < b)
  (h_pos_c : 0 < c)
  (h1 : a * (b + c) = 156)
  (h2 : b * (c + a) = 168)
  (h3 : c * (a + b) = 180) : 
  abc_value a b c = 762 :=
sorry

end find_abc_l339_339160


namespace rectangle_square_problem_l339_339834

theorem rectangle_square_problem
  (m n x : ℕ)
  (h : 2 * (m + n) + 2 * x = m * n)
  (h2 : m * n - x^2 = 2 * (m + n)) :
  x = 2 ∧ ((m = 3 ∧ n = 10) ∨ (m = 6 ∧ n = 4)) :=
by {
  -- Proof goes here
  sorry
}

end rectangle_square_problem_l339_339834


namespace solution_of_equation_l339_339263

noncomputable def solve_equation (x : ℝ) : Prop :=
  2021 * x^(2020/202) - 1 = 2020 * x ∧ x ≥ 0

theorem solution_of_equation : solve_equation 1 :=
by {
  sorry,
}

end solution_of_equation_l339_339263


namespace smallest_k_divides_l339_339040

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l339_339040


namespace factorial_division_sum_l339_339463

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339463


namespace ratio_of_boys_to_girls_l339_339184

-- Given conditions
def B : ℕ := 40
def G : ℕ := B + 64

-- Proof statement
theorem ratio_of_boys_to_girls : B : G = 5 : 13 := 
by
  -- Here we should provide the proof, but we'll leave it as sorry for now.
  sorry

end ratio_of_boys_to_girls_l339_339184


namespace number_of_six_digit_integers_formed_with_repetition_l339_339620

theorem number_of_six_digit_integers_formed_with_repetition :
  ∃ n : ℕ, n = 60 ∧ nat.factorial 6 / (nat.factorial 3 * nat.factorial 2 * nat.factorial 1) = n :=
begin
  use 60,
  split,
  { refl },
  { sorry }
end

end number_of_six_digit_integers_formed_with_repetition_l339_339620


namespace compute_expression_l339_339857

theorem compute_expression : 1004^2 - 996^2 - 1002^2 + 998^2 = 8000 := by
  sorry

end compute_expression_l339_339857


namespace cos_pi_div_4_minus_alpha_l339_339053

theorem cos_pi_div_4_minus_alpha (α : ℝ) (h : Real.sin (α + π/4) = 5/13) : 
  Real.cos (π/4 - α) = 5/13 :=
by
  sorry

end cos_pi_div_4_minus_alpha_l339_339053


namespace min_r_minus_p_l339_339597

theorem min_r_minus_p : ∃ (p q r : ℕ), p * q * r = 362880 ∧ p < q ∧ q < r ∧ (∀ p' q' r' : ℕ, (p' * q' * r' = 362880 ∧ p' < q' ∧ q' < r') → r - p ≤ r' - p') ∧ r - p = 39 :=
by
  sorry

end min_r_minus_p_l339_339597


namespace factorial_div_sum_l339_339423

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339423


namespace fundraiser_total_money_l339_339546

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end fundraiser_total_money_l339_339546


namespace domino_tile_count_l339_339983

theorem domino_tile_count (low high : ℕ) (tiles_standard_set : ℕ) (range_standard_set : ℕ) (range_new_set : ℕ) :
  range_standard_set = 6 → tiles_standard_set = 28 →
  low = 0 → high = 12 →
  range_new_set = 13 → 
  (∀ n, 0 ≤ n ∧ n ≤ range_standard_set → ∀ m, n ≤ m ∧ m ≤ range_standard_set → n ≤ m → true) →
  (∀ n, 0 ≤ n ∧ n ≤ range_new_set → ∀ m, n ≤ m ∧ m <= range_new_set → n <= m → true) →
  tiles_new_set = 91 :=
by
  intros h_range_standard h_tiles_standard h_low h_high h_range_new h_standard_pairs h_new_pairs
  --skipping the proof
  sorry

end domino_tile_count_l339_339983


namespace radius_of_smaller_circles_l339_339188

theorem radius_of_smaller_circles (r R : ℝ) (hR : R = 9)
  (h : ∃ (centers : (ℝ×ℝ) → Prop), 
    (∀ (c1 c2 c3 : ℝ × ℝ), centers c1 ∧ centers c2 ∧ centers c3 →
      (dist c1 c2 = 2 * r ∧ dist c2 c3 = 2 * r ∧ dist c3 c1 = 2 * r)) ∧ 
    (∃ (C : ℝ × ℝ), 
      centers C ∧ dist C c1 = R ∧ dist C c2 = R ∧ dist C c3 = R)) :
  r = (9 * (Real.sqrt 3 - 1)) / 2 :=
begin
  sorry
end

end radius_of_smaller_circles_l339_339188


namespace minimized_target_value_l339_339949

noncomputable def parametric_line (t : ℝ) : ℝ × ℝ :=
  (1 + t / 2, 2 + (3 * t) / 2)

def cartesian_curve (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

noncomputable def scaling_transformation (x y : ℝ) : ℝ × ℝ :=
  (2 * x, y)

def transformed_curve (x' y' : ℝ) : Prop :=
  (x' / 2)^2 + y'^2 = 1

def target_function (x y : ℝ) : ℝ :=
  x + 2 * sqrt 3 * y

theorem minimized_target_value (x y : ℝ) (h : transformed_curve x y) :
  target_function x y = -4 :=
sorry

end minimized_target_value_l339_339949


namespace units_digit_of_subtraction_is_seven_l339_339290

theorem units_digit_of_subtraction_is_seven (a b c: ℕ) (h1: a = c + 3) (h2: b = 2 * c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  result % 10 = 7 :=
by
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  let result := original_number - reversed_number
  sorry

end units_digit_of_subtraction_is_seven_l339_339290


namespace shortest_distance_l339_339689

noncomputable def P (t : ℝ) : ℝ × ℝ × ℝ := (t + 1, -t + 1, 2 * t + 3)
noncomputable def Q (s : ℝ) : ℝ × ℝ × ℝ := (-s + 2, 2 * s + 3, s + 2)

def distance_squared (P Q : ℝ × ℝ × ℝ) : ℝ :=
  (P.1 - Q.1) ^ 2 + (P.2 - Q.2) ^ 2 + (P.3 - Q.3) ^ 2

theorem shortest_distance : ∃ t s, distance_squared (P t) (Q s) = 82 :=
by
  sorry

end shortest_distance_l339_339689


namespace soldiers_height_order_l339_339733

theorem soldiers_height_order {n : ℕ} (a b : Fin n → ℝ) 
  (ha : ∀ i j, i ≤ j → a i ≥ a j) 
  (hb : ∀ i j, i ≤ j → b i ≥ b j) 
  (h : ∀ i, a i ≤ b i) :
  ∀ i, a i ≤ b i :=
  by sorry

end soldiers_height_order_l339_339733


namespace smallest_k_l339_339037

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l339_339037


namespace find_divisor_l339_339177

theorem find_divisor (n k : ℤ) (h1 : n % 30 = 16) : (2 * n) % 30 = 2 :=
by
  sorry

end find_divisor_l339_339177


namespace division_quotient_l339_339242

theorem division_quotient :
  ∃ Q : ℕ, 122 = 20 * Q + 2 ∧ Q = 6 :=
by {
  use 6,
  split,
  { norm_num },
  { refl }
  sorry -- This skips the proof but indicates the correct structure
}

end division_quotient_l339_339242


namespace period_f_axis_of_symmetry_f_max_value_f_l339_339097

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x - Real.pi / 5)

theorem period_f :
  ∃ T > 0, ∀ x, f (x + T) = f x ∧ T = Real.pi := sorry

theorem axis_of_symmetry_f (k : ℤ) :
  ∀ x, 2 * x - Real.pi / 5 = Real.pi / 4 + k * Real.pi → x = 9 * Real.pi / 40 + k * Real.pi / 2 := sorry

theorem max_value_f :
  ∃ x ∈ Set.Icc (0 : ℝ) (Real.pi / 2), f x = 1 ∧ x = 7 * Real.pi / 20 := sorry

end period_f_axis_of_symmetry_f_max_value_f_l339_339097


namespace number_of_zeros_of_F_l339_339077

-- Define the function f(x) = ln(x)
def f (x : ℝ) : ℝ := Real.log x

-- Define the derivative of f(x)
def f' (x : ℝ) : ℝ := 1 / x

-- Define the function F(x) = f(x) - f'(x)
def F (x : ℝ) : ℝ := f x - f' x

-- State the theorem to find the number of zeros of F(x)
theorem number_of_zeros_of_F : ∃! x > 0, F x = 0 := sorry

end number_of_zeros_of_F_l339_339077


namespace expr_eq_neg11_l339_339806

noncomputable def expr : Real :=
  (- (1 / 27)) ^ (- (1 / 3)) + (Real.log 16 / Real.log 3) * (- Real.log 9 / Real.log 2)

theorem expr_eq_neg11 : expr = -11 :=
by
  sorry

end expr_eq_neg11_l339_339806


namespace problem_solution_l339_339171

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l339_339171


namespace smallest_k_l339_339021

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l339_339021


namespace sign_not_determined_by_sum_of_cube_and_square_l339_339299

theorem sign_not_determined_by_sum_of_cube_and_square :
  ∃ (x y : ℝ), (x^3 + x^2 = y^3 + y^2) ∧ (x * y < 0) ∧ (x ≠ y) ∧ (¬ (x ∈ ℤ)) ∧ (¬ (y ∈ ℤ)) :=
by
  sorry

end sign_not_determined_by_sum_of_cube_and_square_l339_339299


namespace count_integers_between_sqrt8_sqrt75_l339_339115

theorem count_integers_between_sqrt8_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ ∀ i : ℤ, (3 ≤ i ∧ i ≤ 8) ↔ (sqrt 8 < i ∧ i < sqrt 75) :=
by
  use 6
  sorry

end count_integers_between_sqrt8_sqrt75_l339_339115


namespace line_l_eq_length_of_BC_l339_339914

-- Definitions and assumptions
def line_through_points (A B : ℝ × ℝ) := 
  ∃ m b, ∀ (x y : ℝ), A = (x, y) → B = (x, y) → y = m * x + b

def parallel_lines (l1 l2 : ℝ × ℝ → Prop) := 
  ∃ m1 m2 b1 b2, l1 = (λ (p : ℝ × ℝ), p.2 = m1 * p.1 + b1) ∧
                    l2 = (λ (p : ℝ × ℝ), p.2 = m2 * p.1 + b2) ∧
                    m1 = m2

def perpendicular_lines (l1 l2 : ℝ × ℝ → Prop) := 
  ∃ m1 m2 b1 b2, l1 = (λ (p : ℝ × ℝ), p.2 = m1 * p.1 + b1) ∧
                    l2 = (λ (p : ℝ × ℝ), p.2 = m2 * p.1 + b2) ∧
                    m1 * m2 = -1

def equation_of_line (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) := 
  l = (λ (p : ℝ × ℝ), p.2 = 2 * p.1 - 3)

def point_of_intersection (l1 l2 : ℝ × ℝ → Prop) (C : ℝ × ℝ) :=
  l1 C ∧ l2 C

def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Main theorem statements
theorem line_l_eq (a : ℝ) (A B : ℝ × ℝ) (l : ℝ × ℝ → Prop) (m : ℝ × ℝ → Prop)
  (A_eq : A = (2, a)) (B_eq : B = (a, -1)) (m_eq : m = (λ (p : ℝ × ℝ), p.2 = 2 * p.1 + 2))
  (h_parallel : parallel_lines l m) : 
  equation_of_line A B l := 
by 
  sorry 

theorem length_of_BC (a : ℝ) (A B C : ℝ × ℝ) (l p m : ℝ × ℝ → Prop)
  (A_eq : A = (2, a)) (B_eq : B = (a, -1)) (C_eq : C = (0, 2)) (p_eq : p = (λ (p : ℝ × ℝ), p.2 = -1 / 2 * p.1 + 2))
  (m_eq : m = (λ (p : ℝ × ℝ), p.2 = 2 * p.1 + 2)) (h_parallel : parallel_lines l m) (h_perpendicular : perpendicular_lines l p) (h_intersection : point_of_intersection p m C) : 
  distance B C = real.sqrt 10 := 
by 
  sorry 

end line_l_eq_length_of_BC_l339_339914


namespace sum_of_angles_is_ninety_degrees_l339_339844

-- Define the circle and the points
def circle_divided_into_12_equal_arcs := true

-- Define the central angles and corresponding alphas and betas
def central_angle_A_and_I :=  120
def central_angle_G_and_E :=  60
def alpha := central_angle_G_and_E / 2
def beta := central_angle_A_and_I / 2

-- Define the statement to prove
theorem sum_of_angles_is_ninety_degrees  
    (h : circle_divided_into_12_equal_arcs) : alpha + beta = 90 :=
by
  have h₁ : alpha = 30 := rfl
  have h₂ : beta = 60 := rfl
  calc 
    alpha + beta = 30 + 60 := by rw [h₁, h₂]
    ... = 90 := by rfl

end sum_of_angles_is_ninety_degrees_l339_339844


namespace smallest_domain_of_g_is_2_l339_339864

noncomputable def g : ℤ → ℤ := 
  fun x => 
    if x = 12 then 45 
    else if x % 2 = 1 then 4 * x + 2 
    else if (x % 2 = 0 ∧ x / 3 * 3 = x) then x / 3 
    else 0 -- g undefined

theorem smallest_domain_of_g_is_2 : 
  ∃ S : set ℤ, 
  (∀ a, g a ≠ 0 → a ∈ S) ∧
  (∀ b, g b ≠ 0 → b ∈ S) ∧
  12 ∈ S ∧
  45 ∈ S ∧
  182 ∉ S ∧
  S.card = 2 :=
by { sorry }

end smallest_domain_of_g_is_2_l339_339864


namespace area_outside_circle_l339_339719

/-- Define the equilateral triangle and inscribed circle. -/
def equilateral_triangle (a : ℝ) := 
  ∃ (A B C : ℝ×ℝ), (dist A B = a) ∧ (dist B C = a) ∧ (dist A C = a)

/-- Define the circle centered at the centroid of the triangle with given radius. -/
def inscribed_circle (a : ℝ) := 
  ∃ (O : ℝ×ℝ) (r : ℝ), (r = a / 3) ∧ ∀ (P : ℝ×ℝ), dist O P = r ↔ (P on inscribed_circle a)

/-- Prove that the area of the part of the triangle lying outside the circle is a^2(3√3 - π)/18. -/
theorem area_outside_circle (a : ℝ) (h : a > 0) :
  ∀ (T : equilateral_triangle a) (C : inscribed_circle a),
  let A := triangle_area T in
  let C := circle_area C in
  A - C = a^2 * (3 * Real.sqrt 3 - Real.pi) / 18 
:= 
by
  -- skipping proof
  sorry

end area_outside_circle_l339_339719


namespace two_digit_number_solution_l339_339633

theorem two_digit_number_solution : ∃ (x y z : ℕ), (1 ≤ x ∧ x ≤ 9) ∧ (1 ≤ y ∧ y ≤ 9) ∧ 10 * x + y = 10 * 5 + 3 ∧ 10 * y + x = 10 * 3 + 5 ∧ 3 * z = 3 * 15 ∧ 2 * z = 2 * 15 := by
  sorry

end two_digit_number_solution_l339_339633


namespace factorial_expression_l339_339444

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339444


namespace solve_for_y_l339_339734

theorem solve_for_y {y : ℝ} : (y - 5)^4 = 16 → y = 7 :=
by
  sorry

end solve_for_y_l339_339734


namespace circle_center_radius_l339_339281

/-
Given:
- The endpoints of a diameter are (2, -3) and (-8, 7).

Prove:
- The center of the circle is (-3, 2).
- The radius of the circle is 5√2.
-/

noncomputable def center_and_radius (A B : ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let Cx := ((A.1 + B.1) / 2)
  let Cy := ((A.2 + B.2) / 2)
  let radius := Real.sqrt ((A.1 - Cx) * (A.1 - Cx) + (A.2 - Cy) * (A.2 - Cy))
  (Cx, Cy, radius)

theorem circle_center_radius :
  center_and_radius (2, -3) (-8, 7) = (-3, 2, 5 * Real.sqrt 2) :=
by
  sorry

end circle_center_radius_l339_339281


namespace megan_picture_shelves_l339_339239

def books_per_shelf : ℕ := 7
def mystery_shelves : ℕ := 8
def total_books : ℕ := 70
def total_mystery_books : ℕ := mystery_shelves * books_per_shelf
def total_picture_books : ℕ := total_books - total_mystery_books
def picture_shelves : ℕ := total_picture_books / books_per_shelf

theorem megan_picture_shelves : picture_shelves = 2 := 
by sorry

end megan_picture_shelves_l339_339239


namespace true_propositions_l339_339955

noncomputable def discriminant_leq_zero : Prop :=
  let a := 1
  let b := -1
  let c := 2
  b^2 - 4 * a * c ≤ 0

def proposition_1 : Prop := discriminant_leq_zero

def proposition_2 (x : ℝ) : Prop :=
  abs x ≥ 0 → x ≥ 0

def proposition_3 : Prop :=
  5 > 2 ∧ 3 < 7

theorem true_propositions : proposition_1 ∧ proposition_3 ∧ ¬∀ x : ℝ, proposition_2 x :=
by
  sorry

end true_propositions_l339_339955


namespace area_of_pentagon_ABCDE_l339_339510

-- Define the given conditions
def is_convex_pentagon (vertices : Fin 5 → (ℝ × ℝ)) := 
  True -- This is a simplification, detailed geometric definition omitted

def side_length (P : Fin 5 → (ℝ × ℝ)) (i j : Fin 5) (length : ℝ) :=
  (P i).dist (P j) = length

def interior_angle (vertices : Fin 5 → ℝ × ℝ) (index : Fin 5) (angle : ℕ) := 
  True -- Similarly, detailed computation of interior angles is omitted

-- Condition synthesis based on the problem statement
def ABCDE_satisfies_conditions (ABCDE : Fin 5 → ℝ × ℝ) := 
  side_length ABCDE 0 1 1.5 ∧
  side_length ABCDE 1 2 1.5 ∧
  side_length ABCDE 2 3 3 ∧
  side_length ABCDE 3 4 3 ∧
  side_length ABCDE 4 0 1.5 ∧
  interior_angle ABCDE 0 120 ∧
  interior_angle ABCDE 1 120 ∧
  is_convex_pentagon ABCDE

-- The problem statement: proving the area of ABCDE
theorem area_of_pentagon_ABCDE (ABCDE : Fin 5 → ℝ × ℝ) 
  (h : ABCDE_satisfies_conditions ABCDE) :
  area ABCDE = 7.03125 * Real.sqrt 3 :=
sorry

end area_of_pentagon_ABCDE_l339_339510


namespace smallest_k_l339_339018

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l339_339018


namespace four_pow_a_l339_339562

-- Definition: a = log_2 3
def a : ℝ := Real.logBase 2 3

-- Theorem: 4^a = 9
theorem four_pow_a : 4^a = 9 := by
  sorry

end four_pow_a_l339_339562


namespace complex_fraction_simplify_l339_339631

variable (i : ℂ)
variable (h : i^2 = -1)

theorem complex_fraction_simplify :
  (1 - i) / ((1 + i) ^ 2) = -1/2 - i/2 :=
by
  sorry

end complex_fraction_simplify_l339_339631


namespace twelve_point_sphere_in_tetrahedron_l339_339249

noncomputable def equilateral_tetrahedron := Type
variables (A B C D : equilateral_tetrahedron)
variables (H H1 H2 H3 : equilateral_tetrahedron)  -- Feet of the altitudes
variables (P P1 P2 P3 : equilateral_tetrahedron)  -- Midpoints of the altitudes
variable  O : equilateral_tetrahedron  -- Centroid

-- The proof statement
theorem twelve_point_sphere_in_tetrahedron :
  ∃ R : Float, 
    ∀ p ∈ {H, H1, H2, H3, P, P1, P2, P3, O},
    dist O p = R := sorry

end twelve_point_sphere_in_tetrahedron_l339_339249


namespace arithmetic_progression_sum_l339_339108

theorem arithmetic_progression_sum (a d S n : ℤ) (h_a : a = 32) (h_d : d = -4) (h_S : S = 132) :
  (n = 6 ∨ n = 11) :=
by
  -- Start the proof here
  sorry

end arithmetic_progression_sum_l339_339108


namespace other_root_is_neg_2_l339_339967

theorem other_root_is_neg_2 (k : ℝ) (h : Polynomial.eval 0 (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) : 
  ∃ t : ℝ, (Polynomial.eval t (Polynomial.C k + Polynomial.X * 2 + Polynomial.X ^ 2) = 0) ∧ t = -2 :=
by
  sorry

end other_root_is_neg_2_l339_339967


namespace factorial_expression_l339_339440

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339440


namespace cos_alpha_beta_half_eq_sqrt_3_div_15_l339_339559

theorem cos_alpha_beta_half_eq_sqrt_3_div_15
  (α β : ℝ)
  (h1 : α - β = π / 3)
  (h2 : cos α + cos β = 1 / 5) :
  cos ((α + β) / 2) = sqrt 3 / 15 :=
by
  sorry

end cos_alpha_beta_half_eq_sqrt_3_div_15_l339_339559


namespace randy_piggy_bank_l339_339726

theorem randy_piggy_bank : 
  ∀ (initial_amount trips_per_month cost_per_trip months_per_year total_spent_left : ℕ),
  initial_amount = 200 →
  cost_per_trip = 2 →
  trips_per_month = 4 →
  months_per_year = 12 →
  total_spent_left = initial_amount - (cost_per_trip * trips_per_month * months_per_year) →
  total_spent_left = 104 :=
by
  intros initial_amount trips_per_month cost_per_trip months_per_year total_spent_left
  sorry

end randy_piggy_bank_l339_339726


namespace sufficient_but_not_necessary_condition_l339_339581

theorem sufficient_but_not_necessary_condition (a b : ℝ) :
  (a > 1 ∧ b > 2) → (a + b > 3 ∧ a * b > 2) ∧ ¬((a + b > 3 ∧ a * b > 2) → (a > 1 ∧ b > 2)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l339_339581


namespace total_travel_time_in_minutes_l339_339710

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l339_339710


namespace ahhua_correct_answers_l339_339837

variables (q : ℕ) (x : ℕ)
constant correct_answers_ahhua : ℕ

-- Conditions
axiom same_number_of_questions : true
axiom points_per_correct_answer : ∀ n : ℕ, 10 * n
axiom points_per_wrong_answer : ∀ n : ℕ, -5 * n
axiom ahhua_score_difference : ∀ (n m : ℕ), (10 * m - 5 * (q - m)) - (10 * n - 5 * (q - n)) = 30
axiom correct_answers_ahhua_five : correct_answers_ahhua = 5

-- Question and correct answer
theorem ahhua_correct_answers (correct_answers_ahhua == 5) (correct_answers_ahhua - x == 30) : x = 3 :=
by
  sorry

end ahhua_correct_answers_l339_339837


namespace angle_C_is_120_degrees_l339_339702

theorem angle_C_is_120_degrees (l m : ℝ) (A B C : ℝ) (hal : l = m) 
  (hA : A = 100) (hB : B = 140) : C = 120 := 
by 
  sorry

end angle_C_is_120_degrees_l339_339702


namespace alyssa_has_cookies_l339_339840

theorem alyssa_has_cookies
    (A : ℕ)
    (Aiyanna_has_140_cookies : ℕ)
    (difference_is_11 : ℕ) :
    Aiyanna_has_140_cookies = 140 →
    difference_is_11 = 11 →
    (A - 140 = 11) →
    A = 151 :=
by 
  intros h1 h2 h3
  rw [←h2, ←h1] at h3
  linarith

end alyssa_has_cookies_l339_339840


namespace arithmetic_sequence_sum_S15_l339_339595

theorem arithmetic_sequence_sum_S15 (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hs5 : S 5 = 10) (hs10 : S 10 = 30) 
  (has : ∀ n, S n = n * (2 * a 1 + (n - 1) * a 2) / 2) : 
  S 15 = 60 := 
sorry

end arithmetic_sequence_sum_S15_l339_339595


namespace total_travel_time_l339_339707

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l339_339707


namespace expression_eq_neg12_l339_339404

theorem expression_eq_neg12 :
  ((-real.sqrt 6) ^ 2) - 3 * real.sqrt 2 * real.sqrt 18 = -12 :=
by
  sorry

end expression_eq_neg12_l339_339404


namespace sahil_selling_price_l339_339345

-- Define the conditions
def purchased_price := 9000
def repair_cost := 5000
def transportation_charges := 1000
def profit_percentage := 50 / 100

-- Calculate the total cost
def total_cost := purchased_price + repair_cost + transportation_charges

-- Calculate the selling price
def selling_price := total_cost + (profit_percentage * total_cost)

-- The theorem to prove the selling price
theorem sahil_selling_price : selling_price = 22500 :=
by
  -- This is where the proof would go, but we skip it with sorry.
  sorry

end sahil_selling_price_l339_339345


namespace frank_bought_3_decks_l339_339899

theorem frank_bought_3_decks (P : ℕ) (D : ℕ) 
    (deck_price : P = 7) 
    (friend_decks : D = 2) 
    (total_spent : 7 * F + 7 * D = 35) 
    (total_spent_proof : total_spent) : 
    ∃ F, 7 * F + 14 = 35 ∧ F = 3 := 
by 
  -- Given Deck Price
  have deck_price := 7
  -- Frank's friend's cost
  have friend_cost := 14
  -- Total Amount spent
  have total_cost := 35
  -- Proving the equation holds
  existsi 3
  -- Ensuring the conditions hold
  split 
  . exact total_spent_proof
  . refl
  
example : ∃ (F : ℕ), 7 * F + 14 = 35 ∧ F = 3 := sorry

end frank_bought_3_decks_l339_339899


namespace expression_evaluation_l339_339853

theorem expression_evaluation (a b c : ℤ) (h₁ : a = 8) (h₂ : b = 10) (h₃ : c = 3) :
  (2 * a - (b - 2 * c)) - ((2 * a - b) - 2 * c) + 3 * (a - c) = 27 :=
by
  have ha : a = 8 := h₁
  have hb : b = 10 := h₂
  have hc : c = 3 := h₃
  rw [ha, hb, hc]
  sorry

end expression_evaluation_l339_339853


namespace sum_of_products_l339_339762

theorem sum_of_products {a b c : ℝ}
  (h1 : a ^ 2 + b ^ 2 + c ^ 2 = 138)
  (h2 : a + b + c = 20) :
  a * b + b * c + c * a = 131 := 
by
  sorry

end sum_of_products_l339_339762


namespace factorial_div_sum_l339_339430

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339430


namespace prove_correctness_l339_339641

-- Define the first 100 positive even integers and their sum S1
def sum_first_n_even (n : ℕ) : ℕ :=
  n * (n + 1)

-- Define the sum of positive even integers from a to b
def sum_even_range (a b : ℕ) (h : a ≤ b) : ℕ :=
  let n := (b - a) / 2 + 1 in
  n * (a + b) / 2

-- Define the given problem statement
def problem_statement : Prop :=
  let S1 := sum_first_n_even 100 in
  let S2 := sum_even_range 102 400 (by norm_num) in
  S1 + S2 = 47750

-- The theorem stating the proof problem
theorem prove_correctness : problem_statement :=
by sorry

end prove_correctness_l339_339641


namespace minimum_value_expression_l339_339232

theorem minimum_value_expression 
  (x1 x2 x3 x4 x5 : ℝ) 
  (h_pos : 0 < x1 ∧ 0 < x2 ∧ 0 < x3 ∧ 0 < x4 ∧ 0 < x5) 
  (h_cond : x1^3 + x2^3 + x3^3 + x4^3 + x5^3 = 1) : 
  ∃ y, y = (3 * Real.sqrt 3) / 2 ∧ 
  (y = (x1 / (1 - x1^2) + x2 / (1 - x2^2) + x3 / (1 - x3^2) + x4 / (1 - x4^2) + x5 / (1 - x5^2))) :=
sorry

end minimum_value_expression_l339_339232


namespace polynomial_divisibility_l339_339351

theorem polynomial_divisibility (n : ℕ) : (∀ x : ℤ, (x^2 + x + 1 ∣ x^(2*n) + x^n + 1)) ↔ (3 ∣ n) := by
  sorry

end polynomial_divisibility_l339_339351


namespace num_even_factors_of_n_l339_339965

-- Define the given number n
def n : ℕ := 2^4 * 3^3 * 5^2 * 7

-- Define the condition that a number is even
def is_even (k : ℕ) : Prop := ∃ i, k = 2 * i

-- Define a factor of n
def is_factor (a k : ℕ) : Prop := a * k = n

-- Prove the number of even positive factors of n
theorem num_even_factors_of_n : 
  { k : ℕ // k > 0 ∧ is_factor k n ∧ is_even k }.card = 96 := 
by sorry

end num_even_factors_of_n_l339_339965


namespace count_integers_between_sqrt8_and_sqrt75_l339_339122

theorem count_integers_between_sqrt8_and_sqrt75 : 
  let n1 := Real.sqrt 8,
      n2 := Real.sqrt 75 in 
  ∃ count : ℕ, count = (Int.floor n2 - Int.ceil n1) + 1 ∧ count = 6 :=
by
  let n1 := Real.sqrt 8
  let n2 := Real.sqrt 75
  use (Int.floor n2 - Int.ceil n1 + 1)
  simp only [Real.sqrt]
  sorry

end count_integers_between_sqrt8_and_sqrt75_l339_339122


namespace no_infinite_harmonic_mean_sequence_l339_339250

theorem no_infinite_harmonic_mean_sequence :
  ¬ ∃ (a : ℕ → ℕ), (∀ n, a n = a 0 → False) ∧
                   (∀ i, 1 ≤ i → a i = (2 * a (i - 1) * a (i + 1)) / (a (i - 1) + a (i + 1))) :=
sorry

end no_infinite_harmonic_mean_sequence_l339_339250


namespace find_angle_A_find_side_a_l339_339927

-- Define the conditions
def triangle_sides (a b c : ℝ) (A B C : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0 ∧ A + B + C = π

def given_condition (a c A C : ℝ) : Prop := 
  (a > 0 ∧ c > 0 ∧ A > 0 ∧ C > 0) ∧ 
  (√3 * a / c = (Real.cos A + 2) / Real.sin C)

def area_condition (b c A : ℝ) : Prop :=
  ∃ (area : ℝ), area = √3 ∧ (1 / 2) * b * c * Real.sin A = √3 / 4 * b * c

-- Theorem proving A = 2π/3 given the conditions
theorem find_angle_A (a c A C : ℝ) 
  (h1 : given_condition a c A C) : 
  A = 2 * Real.pi / 3 := 
sorry

-- Theorem proving a = √21 given additional conditions
theorem find_side_a (a b c A : ℝ) 
  (h1 : triangle_sides a b c A B C)
  (h2 : given_condition a c A C) 
  (h3 : area_condition b c A) 
  (h4 : b + c = 5) : 
  a = Real.sqrt 21 := 
sorry

end find_angle_A_find_side_a_l339_339927


namespace find_m_l339_339594

theorem find_m (m : ℝ) (x : ℝ) (h : 2*x + m = 1) (hx : x = -1) : m = 3 := 
by
  rw [hx] at h
  linarith

end find_m_l339_339594


namespace days_x_worked_before_y_l339_339802

noncomputable def x_work_days : ℝ := 40
noncomputable def y_work_days : ℝ := 30
noncomputable def y_finish_days : ℝ := 24

theorem days_x_worked_before_y (W : ℝ) : 
  let x_rate := W / x_work_days;
      y_rate := W / y_work_days;
      work_done_by_y := y_finish_days * y_rate;
      work_done_by_x := W - work_done_by_y in
  work_done_by_x / x_rate = 8 :=
by
  sorry

end days_x_worked_before_y_l339_339802


namespace acme_vowel_soup_sequences_l339_339835

/-!
## Problem Statement
Acme Corporation has revised its alphabet soup, now including each of the vowels (A, E, I, O, U) of the English alphabet four times, while consonants are excluded. Prove that the number of different seven-letter sequences that can be formed from this revised bowl of Acme Vowel Soup is 78125.
-/

theorem acme_vowel_soup_sequences : 
  let vowels := ['A', 'E', 'I', 'O', 'U'] in
  let num_vowels := 4 in
  let sequence_length := 7 in
  (vowels.length ^ sequence_length = 78125) := by
  sorry

end acme_vowel_soup_sequences_l339_339835


namespace probability_sequence_correct_l339_339316

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l339_339316


namespace integer_count_between_sqrt8_and_sqrt75_l339_339144

theorem integer_count_between_sqrt8_and_sqrt75 :
  let least_integer_greater_than_sqrt8 := 3
      greatest_integer_less_than_sqrt75 := 8
  in (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6 :=
by
  let least_integer_greater_than_sqrt8 := 3
  let greatest_integer_less_than_sqrt75 := 8
  show (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339144


namespace solution_z_sq_eq_neg_4_l339_339052

theorem solution_z_sq_eq_neg_4 (x y : ℝ) (i : ℂ) (z : ℂ) (h : z = x + y * i) (hi : i^2 = -1) : 
  z^2 = -4 ↔ z = 2 * i ∨ z = -2 * i := 
by
  sorry

end solution_z_sq_eq_neg_4_l339_339052


namespace abs_e_pi_minus_six_l339_339400

noncomputable def e : ℝ := 2.718
noncomputable def pi : ℝ := 3.14159

theorem abs_e_pi_minus_six : |e + pi - 6| = 0.14041 := by
  sorry

end abs_e_pi_minus_six_l339_339400


namespace intersection_M_N_l339_339104

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℝ := {x | Real.log x / Real.log 4 ≥ 1}

theorem intersection_M_N :
  M ∩ N = {4, 5} :=
sorry

end intersection_M_N_l339_339104


namespace circle_radius_c_value_l339_339057

theorem circle_radius_c_value (x y c : ℝ) (h₁ : x^2 + 8 * x + y^2 + 10 * y + c = 0) (h₂ : (x+4)^2 + (y+5)^2 = 25) :
  c = -16 :=
by sorry

end circle_radius_c_value_l339_339057


namespace evaluate_64_pow_3_div_2_l339_339525

theorem evaluate_64_pow_3_div_2 : (64 : ℝ)^(3/2) = 512 := by
  -- given 64 = 2^6
  have h : (64 : ℝ) = 2^6 := by norm_num
  -- use this substitution and properties of exponents
  rw [h, ←pow_mul]
  norm_num
  sorry -- completing the proof, not needed based on the guidelines

end evaluate_64_pow_3_div_2_l339_339525


namespace nurses_quit_count_l339_339289

-- Initial Definitions
def initial_doctors : ℕ := 11
def initial_nurses : ℕ := 18
def doctors_quit : ℕ := 5
def total_remaining_staff : ℕ := 22

-- Remaining Doctors Calculation
def remaining_doctors : ℕ := initial_doctors - doctors_quit

-- Theorem to prove the number of nurses who quit
theorem nurses_quit_count : initial_nurses - (total_remaining_staff - remaining_doctors) = 2 := by
  sorry

end nurses_quit_count_l339_339289


namespace factorial_div_sum_l339_339431

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339431


namespace time_for_new_circle_l339_339812

theorem time_for_new_circle 
  (rounds : ℕ) (time : ℕ) (k : ℕ) (original_time_per_round new_time_per_round : ℝ) 
  (h1 : rounds = 8) 
  (h2 : time = 40) 
  (h3 : k = 10) 
  (h4 : original_time_per_round = time / rounds)
  (h5 : new_time_per_round = original_time_per_round * k) :
  new_time_per_round = 50 :=
by {
  sorry
}

end time_for_new_circle_l339_339812


namespace mean_less_than_median_l339_339549

def students : ℕ := 18

def days_frequencies : List (ℕ × ℕ) :=
  [(0, 2), (1, 3), (2, 4), (3, 5), (4, 2), (5, 1), (6, 1)]

-- Statement in Lean
theorem mean_less_than_median :
  let total_days_missed := (days_frequencies.foldr (λ (x : ℕ × ℕ) (acc : ℕ), acc + x.1 * x.2) 0)
      mean_days_missed := total_days_missed / (students : ℚ)
      median_days_missed := 3 in
  mean_days_missed - median_days_missed = - (1 / 3 : ℚ) :=
by
  sorry

end mean_less_than_median_l339_339549


namespace period_f_eq_pi_increasing_interval_f_max_min_values_f_l339_339090

noncomputable def f (x : ℝ) : ℝ := sqrt 2 * real.sin (2 * x + π / 4) + 2

theorem period_f_eq_pi : ∃ T > 0, (∀ x : ℝ, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x : ℝ, f (x + T') = f x) → T' ≥ T) := sorry

theorem increasing_interval_f : ∀ k : ℤ, 
  ∀ x ∈ (Icc (-3 * π / 8 + k * π) (π / 8 + k * π)), 
  ∃ μ _, ( μ = 2 * x + π / 4 ∧ - (π / 2) + 2 * k * π ≤ μ ∧ μ ≤ (π / 2) + 2 * k * π) := sorry

theorem max_min_values_f : max (f 0) (f (π / 2)) = 2 + sqrt 2 ∧ min (f 0) (f (π / 2)) = 1 := sorry

end period_f_eq_pi_increasing_interval_f_max_min_values_f_l339_339090


namespace closest_integer_to_shaded_area_l339_339816

-- Definitions based on conditions
def rectangle_area (length : ℝ) (width : ℝ) : ℝ := length * width
def circle_area (diameter : ℝ) : ℝ := let r := diameter / 2 in Real.pi * r * r

-- Specific conditions for the problem
def length : ℝ := 4
def width : ℝ := 6
def diameter : ℝ := 2

-- Calculate areas based on definitions
def area_rectangle : ℝ := rectangle_area length width
def area_circle : ℝ := circle_area diameter
def area_shaded : ℝ := area_rectangle - area_circle

-- The final proof statement
theorem closest_integer_to_shaded_area : Int := 21

end closest_integer_to_shaded_area_l339_339816


namespace exists_point_Y_l339_339804

noncomputable def angle (O A X : Point) : ℝ :=
sorry -- Angle in radians from OA to OX (0 <= angle < 2*pi)

def circle_radius (O X : Point) (α : ℝ) : ℝ :=
dist O X + α / dist O X

noncomputable def C (O X : Point) (α : ℝ) : Circle :=
Circle.mk O (circle_radius O X α)

def colored_plane (colors : Type) [fintype colors] :=
Point → colors

theorem exists_point_Y (O A : Point) (colors : Type) [fintype colors] 
	(coloring : colored_plane colors) :
	∃ Y : Point, angle O A Y > 0 ∧ ∃ Z : Point, distance O Z = circle_radius O Y (angle O A Y) ∧ coloring Y = coloring Z :=
by sorry

end exists_point_Y_l339_339804


namespace modulus_of_z_l339_339911

-- Define the complex number z and the imaginary unit i
def complex_number (i : ℂ) : ℂ :=
  let z := i * (2 + i)
  z

-- Prove that the modulus of z is sqrt(5)
theorem modulus_of_z (i : ℂ) (hi : i * i = -1) :
  abs (complex_number i) = Real.sqrt 5 :=
sorry

end modulus_of_z_l339_339911


namespace number_of_integers_between_sqrts_l339_339135

theorem number_of_integers_between_sqrts :
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  (upper_bound - lower_bound + 1) = 6 :=
by
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  have sqrt8_bound : (2 : ℝ) < sqrt8 ∧ sqrt8 < 3 := sorry
  have sqrt75_bound : (8 : ℝ) < sqrt75 ∧ sqrt75 < 9 := sorry
  have lower_bound_def : lower_bound = 3 := sorry
  have upper_bound_def : upper_bound = 8 := sorry
  show (upper_bound - lower_bound + 1) = 6
  calc
    upper_bound - lower_bound + 1 = 8 - 3 + 1 := by rw [upper_bound_def, lower_bound_def]
                             ... = 6 := by norm_num
  sorry

end number_of_integers_between_sqrts_l339_339135


namespace triangle_ineq_l339_339206

theorem triangle_ineq (A B C L K : Point) 
                       (hBL : is_angle_bisector A B C L)
                       (hLK : dist L K = dist A B)
                       (hAKBC : parallel (line A K) (line B C)) :
  dist A B > dist B C := 
sorry

end triangle_ineq_l339_339206


namespace complex_number_in_third_quadrant_l339_339960

theorem complex_number_in_third_quadrant (a : ℝ) (h : 3 < a ∧ a < 5) :
  let z := complex.mk (a^2 - 8 * a + 15) (a^2 - 5 * a - 14) in 
  z.re < 0 ∧ z.im < 0 :=
by
  sorry

end complex_number_in_third_quadrant_l339_339960


namespace syllogism_major_minor_premise_l339_339202

theorem syllogism_major_minor_premise
(people_of_Yaan_strong_unyielding : Prop)
(people_of_Yaan_Chinese : Prop)
(all_Chinese_strong_unyielding : Prop) :
  all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese → (all_Chinese_strong_unyielding = all_Chinese_strong_unyielding ∧ people_of_Yaan_Chinese = people_of_Yaan_Chinese) :=
by
  intros h
  exact ⟨rfl, rfl⟩

end syllogism_major_minor_premise_l339_339202


namespace factorial_division_l339_339483

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339483


namespace angle_between_vectors_l339_339923

noncomputable theory

open Real EuclideanSpace

variable {V : Type*} [inner_product_space ℝ V] [finite_dimensional ℝ V]

theorem angle_between_vectors (a b : V) (ha : a ≠ 0) (hb : b ≠ 0)
  (h : ∥a∥ = ∥b∥)
  (h2 : (2 • a + b) ⬝ b = 0) : angle a b = real.arccos (-1 / 2) := 
begin
  sorry
end

end angle_between_vectors_l339_339923


namespace proof_problem_l339_339596

open Nat

variables {a_n b_n : ℕ → ℕ}
variables {S_n T_n : ℕ → ℕ}
variables {n : ℕ}

-- Given conditions:
axiom sum_condition : ∀ (n : ℕ), T_n n ≠ 0 → S_n n * (3 * n - 5) = T_n n * (2 * n - 5)

-- Definition of terms in sequences and their sums.
def a_seq (k : ℕ) := a_n k
def b_seq (k : ℕ) := b_n k
def S_sum (k : ℕ) := S_n k
def T_sum (k : ℕ) := T_n k

-- The statement we need to prove:
theorem proof_problem (h1 : a_seq 7 = a_n 1) (h2 : a_seq 9 = a_n 9)
  (h3 : b_seq 2 = b_n 1) (h4 : b_seq 8 = b_n 9) 
  (h5 : b_seq 4 = b_n 1) (h6 : b_seq 6 = b_n 9) 
  (h7 : T_sum 9 ≠ 0) :
  (a_seq 7 / (b_seq 2 + b_seq 8) + a_seq 3 / (b_seq 4 + b_seq 6)) = 13 / 22 :=
by
  sorry

end proof_problem_l339_339596


namespace muffy_vs_scruffy_l339_339850

-- Definitions of weights
def weight_puffy : ℕ := weight_muffy + 5
def weight_scruffy : ℕ := 12
def total_weight : ℕ := weight_puffy + weight_muffy = 23

-- Definition of weight difference
def weight_diff := weight_scruffy - weight_muffy = 3

-- Prove that the weight difference between Scruffy and Muffy is 3 ounces
theorem muffy_vs_scruffy (weight_muffy : ℕ) (weight_puffy : ℕ) (weight_scruffy : ℕ) (total_weight : ℕ) :
  weight_puffy = weight_muffy + 5 → weight_scruffy = 12 → total_weight = 23 → weight_diff :=
by
  intros h1 h2 h3
  -- Have something like
  sorry

end muffy_vs_scruffy_l339_339850


namespace infinite_integers_repr_l339_339724

theorem infinite_integers_repr : ∀ (k : ℕ), k > 1 →
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  (a - 1) / b + (b - 1) / c + (c - 1) / a = k + 1 :=
by
  intros k hk
  let a := (2 * k + 1) * k
  let b := (2 * k + 1) * k - 1
  let c := 2 * k + 1
  sorry

end infinite_integers_repr_l339_339724


namespace count_true_statements_is_3_l339_339958

-- Given non-zero vectors a and b
variables (a b : V) [vector_space ℝ V] [module ℝ V]   -- Ensure they are in some vector space over ℝ
variable (a_nonzero : a ≠ 0)
variable (b_nonzero : b ≠ 0)

-- Definitions of the truth of each statement
def statement1 : Prop := 
  same_direction (2 • a) a ∧ (∥ 2 • a ∥ = 2 * ∥ a ∥)

def statement2 : Prop :=
  opposite_direction (-2 • a) (5 • a) ∧ (∥ -2 • a ∥ = (2 / 5) * ∥ 5 • a ∥)

def statement3 : Prop :=
  opposite_vectors (-2 • a) (2 • a)

def statement4 : Prop :=
  opposite_vectors (a - b) (-(b - a))

-- The main theorem to prove the number of true statements
theorem count_true_statements_is_3 : 
  (statement1 a ∧ statement2 a ∧ statement3 a ∧ ¬ statement4 a) ↔ true :=
sorry  -- Proof not included

end count_true_statements_is_3_l339_339958


namespace factorial_sum_division_l339_339454

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339454


namespace problem_1_problem_2_l339_339600

theorem problem_1 (a : ℕ → ℤ) (h : ∀ x : ℝ, (1 + x)^6 * (1 - 2 * x)^5 = ∑ i in range 12, a i * x^i) :
  ∑ i in finset.range 11, a (i + 1) = -65 := 
sorry

theorem problem_2 (a : ℕ → ℤ) (h : ∀ x : ℝ, (1 + x)^6 * (1 - 2 * x)^5 = ∑ i in range 12, a i * x^i) :
  ∑ i in finset.range 6, a (2 * i) = -32 := 
sorry

end problem_1_problem_2_l339_339600


namespace factorial_div_sum_l339_339425

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339425


namespace unit_prices_cost_relationship_l339_339194

-- Conditions in the problem
def price_relation (x : ℝ) := 1.2 * x
def units_A (x : ℝ) := 30000 / (1.2 * x)
def units_B (x : ℝ) := 15000 / x
def available_units (units_A units_B : ℝ) := units_A = units_B + 4

-- Proving unit prices
theorem unit_prices (x : ℝ) :
  (available_units (units_A x) (units_B x)) → x = 2500 :=
by sorry  -- Proof of this theorem is not required

def price_A := price_relation 2500
def price_B := 2500

-- Definition for total cost function and minimum purchase cost
def total_cost (a : ℝ) := 500 * a + 75000
def min_cost := 78000

-- Purchase constraints
def purchase_constraints (a : ℝ) := a ≥ 6

-- Proving the functional relationship and the minimum cost
theorem cost_relationship (a : ℝ) :
  (purchase_constraints a) → total_cost a = 78000 :=
by sorry  -- Proof of this theorem is not required

end unit_prices_cost_relationship_l339_339194


namespace find_value_of_a_l339_339930

theorem find_value_of_a :
  ∃ (a : ℕ), (1 ≤ a ∧ a ≤ 9) ∧ (0.1 * a = 1 / a) → a = 6 :=
sorry

end find_value_of_a_l339_339930


namespace temperature_decrease_l339_339178

-- Define the conditions
def temperature_rise (temp_increase: ℤ) : ℤ := temp_increase

-- Define the claim to be proved
theorem temperature_decrease (temp_decrease: ℤ) : temperature_rise 3 = 3 → temperature_rise (-6) = -6 :=
by
  sorry

end temperature_decrease_l339_339178


namespace sequence_inequality_l339_339950

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 
  if n = 0 then 1/2
  else a (n - 1) + (1 / (n:ℚ)^2) * (a (n - 1))^2

theorem sequence_inequality (n : ℕ) : 
  1 - 1 / 2 ^ (n + 1) ≤ a n ∧ a n < 7 / 5 := 
sorry

end sequence_inequality_l339_339950


namespace number_of_six_digit_integers_formed_with_repetition_l339_339619

theorem number_of_six_digit_integers_formed_with_repetition :
  ∃ n : ℕ, n = 60 ∧ nat.factorial 6 / (nat.factorial 3 * nat.factorial 2 * nat.factorial 1) = n :=
begin
  use 60,
  split,
  { refl },
  { sorry }
end

end number_of_six_digit_integers_formed_with_repetition_l339_339619


namespace factorial_division_identity_l339_339505

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339505


namespace table_tennis_scenarios_count_l339_339984

theorem table_tennis_scenarios_count :
  let scenarios := [(3, 0), (3, 1), (3, 2)].sum (λ x, (nat.choose 4 (2 * x.1 - 3))).sum (λ x, 2 * x) in
  scenarios = 20 :=
by sorry

end table_tennis_scenarios_count_l339_339984


namespace convex_quadrilateral_fixed_point_l339_339683

variables {A B : Point} 

noncomputable def fixed_point (A B : Point) : Point := sorry

theorem convex_quadrilateral_fixed_point
  (A B : Point) 
  (ABCD : Quadrilateral)
  (h₁ : distance A B = distance B C)
  (h₂ : distance A D = distance D C)
  (h₃ : ∠ADC = 90) :
  ∃ (P : Point), P = fixed_point A B ∧ line_passes_through DC P := 
by {
  sorry
}

end convex_quadrilateral_fixed_point_l339_339683


namespace bakery_new_cakes_count_l339_339848

def cakes_sold := 91
def more_cakes_bought := 63

theorem bakery_new_cakes_count : (91 + 63) = 154 :=
by
  sorry

end bakery_new_cakes_count_l339_339848


namespace maximum_cells_covered_at_least_five_times_l339_339773

theorem maximum_cells_covered_at_least_five_times :
  let areas := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  let total_covered := List.sum areas
  let exact_coverage := 1 * 1 + 2 * 2 + 3 * 3 + 4 * 4
  let remaining_coverage := total_covered - exact_coverage
  let max_cells_covered_at_least_five := remaining_coverage / 5
  max_cells_covered_at_least_five = 5 :=
by
  sorry

end maximum_cells_covered_at_least_five_times_l339_339773


namespace alberto_spent_more_l339_339387

noncomputable def alberto_total_before_discount : ℝ := 2457 + 374 + 520
noncomputable def alberto_discount : ℝ := 0.05 * alberto_total_before_discount
noncomputable def alberto_total_after_discount : ℝ := alberto_total_before_discount - alberto_discount

noncomputable def samara_total_before_tax : ℝ := 25 + 467 + 79 + 150
noncomputable def samara_tax : ℝ := 0.07 * samara_total_before_tax
noncomputable def samara_total_after_tax : ℝ := samara_total_before_tax + samara_tax

noncomputable def amount_difference : ℝ := alberto_total_after_discount - samara_total_after_tax

theorem alberto_spent_more : amount_difference = 2411.98 :=
by
  sorry

end alberto_spent_more_l339_339387


namespace cone_from_sector_l339_339338

theorem cone_from_sector (sector_angle : ℝ) (circle_radius : ℝ) 
  (base_circumference := (sector_angle / 360) * (2 * Real.pi * circle_radius)) 
  (base_radius := base_circumference / (2 * Real.pi))
  (slant_height := circle_radius) :
  sector_angle = 270 → circle_radius = 12 → base_radius = 9 ∧ slant_height = 12 :=
by
  intros h1 h2
  rw [h1, h2]
  have h3 : base_circumference = (270 / 360) * (2 * Real.pi * 12), by rw [h1, h2]
  have h4 : base_circumference = 18 * Real.pi, by norm_num at h3; exact h3
  have h5 : base_radius = 9, by { field_simp [base_circumference, h4], norm_num }
  have h6 : slant_height = 12, by exact h2
  exact ⟨h5, h6⟩

end cone_from_sector_l339_339338


namespace count_k_chains_l339_339901

-- Define the context and given conditions.
def is_k_chain (k : ℕ) (k_gt_zero : k > 0) (I : ℕ → Set ℤ)
  (subset_property : ∀ i j, 1 ≤ i → i ≤ j → j ≤ k → I j ⊆ I i) : Prop :=
    (∀ i, 1 ≤ i → i ≤ k → 168 ∈ I i) ∧ 
    (∀ i, 1 ≤ i → i ≤ k → ∀ x y ∈ I i, x - y ∈ I i)

-- The theorem statement we want to prove
theorem count_k_chains (k : ℕ) (k_gt_zero : k > 0) (I : ℕ → Set ℤ)
  (subset_property : ∀ i j, 1 ≤ i → i ≤ j → j ≤ k → I j ⊆ I i)
  (chain_property : is_k_chain k k_gt_zero I subset_property) :
  ∃ n, n = (k + 1) * (k + 1) * Nat.choose (k + 3) 3 :=
sorry

end count_k_chains_l339_339901


namespace factorial_div_l339_339497

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339497


namespace exists_arithmetic_progression_in_harmonic_sequence_l339_339551

theorem exists_arithmetic_progression_in_harmonic_sequence (n : ℕ) : 
  ∃ (a : ℕ → ℚ) (d : ℚ) (seq : fin n → ℚ), 
    (∀ i, seq i = a i * d) ∧ 
    (∀ i, seq (i+1) - seq i = seq 1 - seq 0) :=
by sorry

end exists_arithmetic_progression_in_harmonic_sequence_l339_339551


namespace sum_first_10_common_elements_l339_339887

theorem sum_first_10_common_elements :
  let AP := λ n : ℕ, 4 + 3 * n
  let GP := λ k : ℕ, 10 * 2^k
  let common_elements k := 10 * 4^k
  (Σ i in Finset.range 10, common_elements i) = 3495250 := by
  sorry

end sum_first_10_common_elements_l339_339887


namespace fundraiser_successful_l339_339548

-- Defining the conditions
def num_students_bringing_brownies := 30
def brownies_per_student := 12
def num_students_bringing_cookies := 20
def cookies_per_student := 24
def num_students_bringing_donuts := 15
def donuts_per_student := 12
def price_per_treat := 2

-- Calculating the total number of each type of treat
def total_brownies := num_students_bringing_brownies * brownies_per_student
def total_cookies := num_students_bringing_cookies * cookies_per_student
def total_donuts := num_students_bringing_donuts * donuts_per_student

-- Calculating the total number of treats
def total_treats := total_brownies + total_cookies + total_donuts

-- Calculating the total money raised
def total_money_raised := total_treats * price_per_treat

theorem fundraiser_successful : total_money_raised = 2040 := by
    -- We introduce a sorry here because we are not providing the proof steps.
    sorry

end fundraiser_successful_l339_339548


namespace parallelogram_area_correct_l339_339002

-- Define the base and height of the parallelogram
def base : ℕ := 22
def height : ℕ := 21

-- Define the area calculation for a parallelogram
def area_parallelogram (b h : ℕ) : ℕ := b * h

-- Verify that the area equals 462 square centimeters
theorem parallelogram_area_correct : area_parallelogram base height = 462 := 
by
  unfold area_parallelogram
  simp
  sorry

end parallelogram_area_correct_l339_339002


namespace sum_of_reciprocals_squared_inequality_l339_339240

theorem sum_of_reciprocals_squared_inequality :
  1 + ∑ k in finset.range (2013+1).filter (λ x, x > 1), (1 / (k : ℝ)^2) < (4025 / 2013 : ℝ) := 
sorry

end sum_of_reciprocals_squared_inequality_l339_339240


namespace expansion_coefficient_l339_339082

theorem expansion_coefficient (a : ℤ) (x : ℝ) 
  (h : (1 + a)^5 = -1) : 
  let T := (x^2 + a / x) in 
  let term := λ r, (-2)^r * (Nat.choose 5 r) * x^(10 - 3 * r) in
  a = -2 → term 3 = -80 := by
  sorry

end expansion_coefficient_l339_339082


namespace count_integers_between_sqrt8_sqrt75_l339_339114

theorem count_integers_between_sqrt8_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ ∀ i : ℤ, (3 ≤ i ∧ i ≤ 8) ↔ (sqrt 8 < i ∧ i < sqrt 75) :=
by
  use 6
  sorry

end count_integers_between_sqrt8_sqrt75_l339_339114


namespace factorial_division_sum_l339_339468

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339468


namespace vector_sum_correct_l339_339402

def v1 : ℝ × ℝ × ℝ := (5, -3, 8)
def v2 : ℝ × ℝ × ℝ := (-2, 4, 1)
def v3 : ℝ × ℝ × ℝ := (3, -6, -9)
def v_sum : ℝ × ℝ × ℝ := (6, -5, 0)

theorem vector_sum_correct : (v1.1 + v2.1 + v3.1, v1.2 + v2.2 + v3.2, v1.3 + v2.3 + v3.3) = v_sum :=
by
  sorry

end vector_sum_correct_l339_339402


namespace polynomial_identity_l339_339165

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l339_339165


namespace sum_sqrt_series_inequality_l339_339722

theorem sum_sqrt_series_inequality (n : ℕ) (h : n ≥ 2) :
    1 + (∑ i in Finset.range (n + 1).filter (λ k => k ≥ 2), (1 : ℝ) / (Real.sqrt k)) > Real.sqrt n := 
sorry

end sum_sqrt_series_inequality_l339_339722


namespace sum_of_series_l339_339517

theorem sum_of_series :
  (∑' n : ℕ, (3^n) / (3^(3^n) + 1)) = 1 / 2 :=
sorry

end sum_of_series_l339_339517


namespace six_digit_integers_count_l339_339615

theorem six_digit_integers_count : 
  let digits := [2, 2, 2, 5, 5, 9] in
  multiset.card (multiset.of_list (list.permutations digits).erase_dup) = 60 :=
by
  sorry

end six_digit_integers_count_l339_339615


namespace tetrahedron_distinct_paintings_l339_339512

theorem tetrahedron_distinct_paintings : 
  let vertices := 4
  let colors := ["blue", "blue", "red", "green"]
  let symmetries := 12 / -- the number of different symmetries, including all rotations
  ∑ fixed_colorings := 12 / -- fixed coloring considering each symmetry
  symmetries * fixed_colorings =
  3 :=
sorry

end tetrahedron_distinct_paintings_l339_339512


namespace max_product_of_perpendiculars_is_centroid_l339_339880

-- Define a triangle by its vertices
structure Triangle (α : Type) :=
(A B C : α)

-- Define a point in 2D plane
structure Point (α : Type) :=
(x y : α)

-- Given a triangle in a 2D plane
variables {α : Type} [Field α]

-- Function to calculate the centroid of a triangle
noncomputable def centroid (T : Triangle (Point α)) : Point α :=
{ x := (T.A.x + T.B.x + T.C.x) / 3,
  y := (T.A.y + T.B.y + T.C.y) / 3 }

-- Defining the perpendicular distance from a point to a line (side of the triangle)
-- This is a placeholder definition. Actual implementation should be more detailed.
noncomputable def perpendicular_distance (P : Point α) (T : Triangle (Point α)) : α :=
sorry -- Placeholder for the perpendicular distance calculation

-- Main theorem statement
theorem max_product_of_perpendiculars_is_centroid
  (T : Triangle (Point α)) :
  ∃ G : Point α, G = centroid T ∧ ∀ P : Point α, ∏ d in { dist | dist = perpendicular_distance P T}, d ≤ ∏ d in { dist | dist = perpendicular_distance (centroid T) T}, d :=
sorry -- Proof not required

end max_product_of_perpendiculars_is_centroid_l339_339880


namespace sum_first_2016_terms_l339_339859

variable {a : ℕ → ℕ} {S : ℕ → ℕ} {C : ℕ → ℝ}

-- Here are the conditions given in the problem:
-- 1. A sequence of positive terms {a_n} with partial sum S_n
-- 2. 2S_n = a_n^2 + a_n for all natural numbers n
-- 3. C_n = (-1)^n * (2a_n + 1) / (2S_n)

-- Define partial sum S_n such that 2S_n = a_n^2 + a_n
def partial_sum_condition (n : ℕ) : Prop :=
  2 * (S n) = (a n)^2 + a n

-- Define the sequence C_n
def sequence_C (n : ℕ) : ℝ :=
  (-1)^(n : ℤ) * (2 * a n + 1) / (2 * S n)

-- Define the summation T_m = sum of first m terms of sequence C_n
def summation_T (m : ℕ) : ℝ :=
  (Finset.range m).sum (λ n, sequence_C (n + 1))

-- Prove that the sum of the first 2016 terms of the sequence {C_n} is -2015/2016
theorem sum_first_2016_terms : summation_T 2016 = (- 2015 / 2016) :=
  sorry

end sum_first_2016_terms_l339_339859


namespace smallest_k_l339_339019

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l339_339019


namespace rhombus_parallel_MQ_NP_l339_339662

open EuclideanGeometry

variables {A B C D E F G H M N P Q O : Point}
variables [Rhombus ABCD]
-- incircle O touches the sides at E, F, G, H respectively
variables (he : TangentAt O E A B)
variables (hf : TangentAt O F B C)
variables (hg : TangentAt O G C D)
variables (hh : TangentAt O H D A)
-- tangents at arcs EF and GH intersecting at points M, N, P, and Q respectively
variables (hm : IntersectTangent EF M A B)
variables (hn : IntersectTangent EF N B C)
variables (hp : IntersectTangent GH P C D)
variables (hq : IntersectTangent GH Q D A)

theorem rhombus_parallel_MQ_NP 
  (rhombus ABCD)
  (incircle O)
  (tangentO_E : TangentAt O E A B)
  (tangentO_F : TangentAt O F B C)
  (tangentO_G : TangentAt O G C D)
  (tangentO_H : TangentAt O H D A)
  (tangentEF_M : IntersectTangent EF M A B)
  (tangentEF_N : IntersectTangent EF N B C)
  (tangentGH_P : IntersectTangent GH P C D)
  (tangentGH_Q : IntersectTangent GH Q D A)
  : Parallel MQ NP := 
sorry

end rhombus_parallel_MQ_NP_l339_339662


namespace cartesian_equation_of_circle_distance_sum_PA_PB_l339_339989

variable (t : ℝ)
def line_parametric (t : ℝ) : ℝ × ℝ := 
  (2 - (real.sqrt 2)/2 * t, 1 + (real.sqrt 2)/2 * t)

variable (θ : ℝ)
def circle_polar (θ : ℝ) : ℝ := 4 * real.cos θ

theorem cartesian_equation_of_circle :
  ∀ (x y : ℝ), (x^2 + y^2 = 4 * x) ↔ ((x-2)^2 + y^2 = 4) :=
  by
    intros x y
    sorry

theorem distance_sum_PA_PB :
  let P := (2,1)
  ∀ (A B : ℝ × ℝ), (A = line_parametric t) ∧ (B = line_parametric t) ∧ (A ≠ B) ∧ ((A.1 - 2)^2 + (A.2)^2 = 4) ∧ ((B.1 - 2)^2 + (B.2)^2 = 4) →
    (real.abs (P.1 - A.1) + real.abs (P.2 - A.2)) + (real.abs (P.1 - B.1) + real.abs (P.2 - B.2)) = real.sqrt 14 :=
  by
    sorry

end cartesian_equation_of_circle_distance_sum_PA_PB_l339_339989


namespace situps_combined_l339_339895

theorem situps_combined (peter_situps : ℝ) (greg_per_set : ℝ) (susan_per_set : ℝ) 
                        (peter_per_set : ℝ) (sets : ℝ) 
                        (peter_situps_performed : peter_situps = sets * peter_per_set) 
                        (greg_situps_performed : sets * greg_per_set = 4.5 * 6)
                        (susan_situps_performed : sets * susan_per_set = 3.75 * 6) :
    peter_situps = 37.5 ∧ greg_per_set = 4.5 ∧ susan_per_set = 3.75 ∧ peter_per_set = 6.25 → 
    4.5 * 6 + 3.75 * 6 = 49.5 :=
by
  sorry

end situps_combined_l339_339895


namespace compare_abc_l339_339905

noncomputable def a : ℝ := (1 / 6) ^ (1 / 2)
noncomputable def b : ℝ := Real.log 1 / 3 / Real.log 6
noncomputable def c : ℝ := Real.log 1 / 7 / Real.log (1 / 6)

theorem compare_abc : c > a ∧ a > b := by
  sorry

end compare_abc_l339_339905


namespace gcd_9125_4277_l339_339784

theorem gcd_9125_4277 : Nat.gcd 9125 4277 = 1 :=
by
  -- proof by Euclidean algorithm steps
  sorry

end gcd_9125_4277_l339_339784


namespace five_twos_make_24_l339_339247

theorem five_twos_make_24 :
  ∃ a b c d e : ℕ, a = 2 ∧ b = 2 ∧ c = 2 ∧ d = 2 ∧ e = 2 ∧
  ((a + b + c) * (d + e) = 24) :=
by
  sorry

end five_twos_make_24_l339_339247


namespace problem_statement_l339_339741

variable {α : Type} [LinearOrder α] {f : α → ℝ} {x_0 : α}

def is_local_max (f : α → ℝ) (x : α) := ∃ ε > 0, ∀ x', abs (x' - x) < ε → f x' ≤ f x
def is_local_min (f : α → ℝ) (x : α) := ∃ ε > 0, ∀ x', abs (x' - x) < ε → f x' ≥ f x

theorem problem_statement (h_dom : ∀ x, x ∈ ℝ)
    (h_local_max : x_0 ≠ 0 ∧ is_local_max f x_0) :
    is_local_min (λ x, -f (-x)) (-x_0) :=
sorry

end problem_statement_l339_339741


namespace jessica_coins_worth_l339_339211

theorem jessica_coins_worth :
  ∃ (n d : ℕ), n + d = 30 ∧ 5 * (30 - d) + 10 * d = 165 :=
by {
  sorry
}

end jessica_coins_worth_l339_339211


namespace rebecca_less_than_toby_l339_339771

-- Define the conditions
variable (x : ℕ) -- Thomas worked x hours
variable (tobyHours : ℕ := 2 * x - 10) -- Toby worked 10 hours less than twice what Thomas worked
variable (rebeccaHours : ℕ := 56) -- Rebecca worked 56 hours

-- Define the total hours worked in one week
axiom total_hours_worked : x + tobyHours + rebeccaHours = 157

-- The proof goal
theorem rebecca_less_than_toby : tobyHours - rebeccaHours = 8 := 
by
  -- (proof steps would go here)
  sorry

end rebecca_less_than_toby_l339_339771


namespace problem_in_circle_l339_339657

theorem problem_in_circle 
  {O : Type*} [metric_space O] [normed_add_comm_group O] [normed_space ℝ O]
  (A C B D P Q R : O) 
  (hO: (dist O A = dist O C) ∧ (dist O B = dist O D)) 
  (h_perp: ∠ A O C = 90 ∧ ∠ B O D = 90)
  (h_chord: ∃ R, segment_intersection (A, P) (B, D) = some R) 
  (h_tangent: is_tangent PQ O P): 
  dist A P * dist A R = dist A R * dist A B :=
by sorry

end problem_in_circle_l339_339657


namespace problem_part1_problem_part2_l339_339068

open Nat

noncomputable def a : ℕ → ℤ
| 1       := -3
| (n + 2) := 4 * (n + 1) - 2 - a (n + 1)

def T (n : ℕ) : ℤ := (Finset.range (2 * n)).sum (λ k => a (k + 1))
def b (n : ℕ) : ℤ := T n + 6 * n

theorem problem_part1 (n : ℕ) : ∃ r : ℤ, ∀ n : ℕ, a (n + 1) - 2 * n = r * (-1)^n := 
sorry

theorem problem_part2 (n : ℕ) : (Finset.range n).sum (λ k => (1 : ℚ) / (b (k + 1))) < 1 / 4 :=
sorry

end problem_part1_problem_part2_l339_339068


namespace factorial_div_l339_339498

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339498


namespace number_of_girls_l339_339309

theorem number_of_girls (total_kids : ℕ) (number_of_boys : ℕ) (h1 : total_kids = 9) (h2 : number_of_boys = 6) : total_kids - number_of_boys = 3 := 
by 
  rw [h1, h2]
  exact rfl
  sorry

end number_of_girls_l339_339309


namespace find_k_l339_339582

variables {a b : ℝ}
variables (k : ℝ)

-- Assume a and b are non-collinear unit vectors
axiom unit_vector_a : ∥a∥ = 1
axiom unit_vector_b : ∥b∥ = 1
axiom non_collinear : a ≠ b

-- Given that a + b is perpendicular to ka - b
axiom perpendicular : (a + b) • (k * a - b) = 0

-- Prove that k = 1
theorem find_k : k = 1 :=
by
  sorry

end find_k_l339_339582


namespace proof_problem_l339_339612

def a_sequence (n : ℕ) : ℕ 
def b_sequence (n : ℕ) : ℕ 
def c_sequence (n : ℕ) : ℝ 
def S_n (n : ℕ) : ℝ 

theorem proof_problem :
  -- Conditions
  (∀ n : ℕ, n > 0 → (λ a_ : ℕ, ∏ i in finset.range n, a i = (sqrt 2) ^ (b_sequence n))) →
  (a_sequence 1 = 2) →
  (b_sequence 3 = 6 + b_sequence 2) →
  (∀ n : ℕ, n > 0 → c_sequence n = (1 / a_sequence n) - (1 / (b_sequence n))) →
  (∀ n : ℕ, S_n n = (finset.sum finset.range n) c_sequence) →
  -- To prove
  a_sequence 3 = 8 ∧
  (∀ n : ℕ, b_sequence n = n * (n + 1)) ∧
  (∀ n : ℕ, S_n n = 1 / (n + 1) - 1 / (2 ^ n)) ∧
  (∀ n k : ℕ, S_k ≥ S_n → k = 4) :=
begin
  sorry
end

end proof_problem_l339_339612


namespace probability_between_bounds_l339_339593

noncomputable def normal_distribution (μ σ : ℝ) : OrElse := sorry

theorem probability_between_bounds 
  (σ : ℝ) (hσ : 0 < σ)
  (X : OrElse) (hX : X = normal_distribution 0 σ)
  (h : P(X > 2) = 0.023) :
  P(-2 ≤ X ∧ X ≤ 2) = 0.954 :=
sorry

end probability_between_bounds_l339_339593


namespace isaac_journey_time_l339_339704

def travel_time_total (speed : ℝ) (time1 : ℝ) (distance2 : ℝ) (rest_time : ℝ) (distance3 : ℝ) : ℝ :=
  let time2 := distance2 / speed
  let time3 := distance3 / speed
  time1 + time2 * 60 + rest_time + time3 * 60

theorem isaac_journey_time :
  travel_time_total 10 (30 : ℝ) 15 (30 : ℝ) 20 = 270 :=
by
  sorry

end isaac_journey_time_l339_339704


namespace math_problem_l339_339200

-- Define the parametric equations of curve C
def curve_C (θ : ℝ) : ℝ × ℝ := (2 * real.cos θ, sqrt 3 * real.sin θ)

-- Define the rectangular coordinates of points A and B
def A : ℝ × ℝ := (1, 1)
def B : ℝ × ℝ := (0, 3)

-- Define the general equation of curve C
def general_eq_C (x y : ℝ) : Prop := (x ^ 2 / 4) + (y ^ 2 / 3) = 1

-- Define the slope of a line passing through two points
def slope (p1 p2 : ℝ × ℝ) : ℝ := (p2.2 - p1.2) / (p2.1 - p1.1)

-- Define the roots product
def roots_product (a b c : ℝ) : ℝ := c / a

-- The theorem
theorem math_problem :
  (∀ θ, (curve_C θ).1 ^ 2 / 4 + (curve_C θ).2 ^ 2 / 3 = 1) ∧
  (slope A B = -2) ∧
  (roots_product (19 / 5) (48 / sqrt 5) 24 = 120 / 19) :=
by repeat { sorry }

end math_problem_l339_339200


namespace min_area_l339_339086

-- Definitions of the points, circle, and the tangency condition
variables {a b : ℝ} (ha : a > 1) (hb : b > 1) (A : ℝ × ℝ) (B : ℝ × ℝ)
def circle (x y : ℝ) := x^2 + y^2 - 2 * x - 2 * y + 1 = 0
def pointA : ℝ × ℝ := (2 * a, 0)
def pointB : ℝ × ℝ := (0, 2 * b)

-- The line AB is tangent to the circle
def tangent (C : ℝ × ℝ) (radius: ℝ) (line : ℝ → ℝ) := 
  let dist := (C.1 * 2*b + C.2 * 2*a - 2*a*2*b) / real.sqrt (4*a^2 + 4*b^2)
  in dist^2 = radius^2

-- The minimum area calculation
def area_triangle : ℝ := 2 * 2 * (2 + real.sqrt 2)

-- Prove the statement
theorem min_area (h1 : circle 1 1) (h2 : tangent (1, 1) 1 (λ x, -b/a * x + 2*b)) : 
  (∀ (x y : ℝ), triangle_area (0, 0) (2*a, 0) (0, 2*b) = 4 + 2 * real.sqrt 2) := 
  sorry

end min_area_l339_339086


namespace find_x0_l339_339093

-- Defining the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
if h : x > 0 then Real.log x else x⁻²

-- Stating the proof goal
theorem find_x0 (x0 : ℝ) (h : f x0 = 1) : x0 = 10 :=
by
  sorry

end find_x0_l339_339093


namespace probability_of_specific_sequence_l339_339312

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l339_339312


namespace sum_of_squares_of_roots_l339_339858

noncomputable def polynomial_2020 : Polynomial ℚ := Polynomial.X ^ 2020 + 45 * Polynomial.X ^ 2017 + 4 * Polynomial.X ^ 4 + 405

theorem sum_of_squares_of_roots : 
  let roots := polynomial_2020.roots in
  (roots.map (λ r, r^2)).sum = 0 := 
by
  sorry

end sum_of_squares_of_roots_l339_339858


namespace factorial_division_sum_l339_339462

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339462


namespace nick_paths_from_origin_to_16_16_odd_direction_changes_l339_339624

theorem nick_paths_from_origin_to_16_16_odd_direction_changes :
  let total_paths := 2 * Nat.choose 30 15 in
  ∃ f : (Nat × Nat) → (List (Nat × Nat)), 
    (f (0, 0)).length = 32 ∧ 
    (∀ i, f (0, 0).nth i ≠ none → 
        (f (0, 0).nth i = some (f (0, 0).nth (i+1)).get_or_else (0, 0) ∧ 
        ∃ n, odd n ∧ n = (List.attach (f (0, 0)).filter (λ p, 
           (p.1.snd = 1 ∧ p.2.snd = 0) ∨ (p.1.snd = 0 ∧ p.2.snd = 1)).length)) :=
sorry

end nick_paths_from_origin_to_16_16_odd_direction_changes_l339_339624


namespace function_identity_l339_339630

theorem function_identity (f : ℕ → ℝ) (h : ∀ x : ℕ, f(x+1) = 2 * f(x)) :
  ∀ x : ℕ, f(x) = 2^x := sorry

end function_identity_l339_339630


namespace number_of_integers_between_sqrt8_and_sqrt75_l339_339124

theorem number_of_integers_between_sqrt8_and_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ 
  ∃ x : Fin n.succ, √8 < ↑x ∧ ↑x < √75 := 
by
  sorry

end number_of_integers_between_sqrt8_and_sqrt75_l339_339124


namespace pq_over_aq_eq_sqrt3_over_3_l339_339961

open real

-- Definitions

-- Given that AB and CD are perpendicular diameters of circle Q
variables {Q : Type*} [metric_space Q] [normed_group Q] [normed_space ℝ Q]
variables (A B C D P Q_center : Q)
variables (r : ℝ) -- radius of the circle

-- Conditions from the problem
hypothesis h1 : dist A Q_center = r
hypothesis h2 : dist B Q_center = r
hypothesis h3 : dist C Q_center = r
hypothesis h4 : dist D Q_center = r
hypothesis h5 : angle A Q_center D = π / 2
hypothesis h6 : angle P Q_center A = 0
hypothesis h7 : angle Q_center P C = π / 3

-- The statement we need to prove
theorem pq_over_aq_eq_sqrt3_over_3 : (dist P Q_center) / r = sqrt 3 / 3 :=
sorry

end pq_over_aq_eq_sqrt3_over_3_l339_339961


namespace direction_vector_correct_l339_339752

noncomputable def P : Matrix (Fin 3) (Fin 3) ℚ :=
  ![
    ![3 / 17, -2 / 17, -2 / 5],
    ![-2 / 17, 1 / 34, 1 / 5],
    ![-2 / 5, 1 / 5, 4 / 5]
  ]

def direction_vector : Fin 3 → ℤ :=
  ⟨ 15, -10, -34 ⟩

theorem direction_vector_correct :
  matrix.mul_vec P (λ i, if i = 0 then 1 else 0) = (coe ∘ direction_vector : Fin 3 → ℚ) :=
by
  -- prove the theorem
  sorry

end direction_vector_correct_l339_339752


namespace science_homework_is_50_minutes_l339_339720

-- Define the times for each homework and project in minutes
def total_time : ℕ := 3 * 60  -- 3 hours converted to minutes
def math_homework : ℕ := 45
def english_homework : ℕ := 30
def history_homework : ℕ := 25
def special_project : ℕ := 30

-- Define a function to compute the time for science homework
def science_homework_time 
  (total_time : ℕ) 
  (math_time : ℕ) 
  (english_time : ℕ) 
  (history_time : ℕ) 
  (project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

-- The theorem to prove the time Porche's science homework takes
theorem science_homework_is_50_minutes : 
  science_homework_time total_time math_homework english_homework history_homework special_project = 50 := 
sorry

end science_homework_is_50_minutes_l339_339720


namespace problem_I_1_problem_I_2_problem_I_3_problem_II_l339_339204

noncomputable theory

-- Definitions corresponding to conditions from part (I)
def triangle_area_1 (b : ℝ) (c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * Real.sin A
def triangle_area_2 (b : ℝ) (c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * Real.sin A
def triangle_area_3 (b : ℝ) (c : ℝ) (A : ℝ) : ℝ := 1 / 2 * b * c * Real.sin A

-- Definitions corresponding to condition and maximum function from part (II)
def cos_B_C_sum (B : ℝ) (C : ℝ) : ℝ := Real.cos B + Real.cos C

-- Statement for problem (I)
theorem problem_I_1 : ∀ (b : ℝ) (c : ℝ), b = 2 → Real.sin c = 2 * Real.sin b → triangle_area_1 b c (Real.acos (b^2 + c^2 - 2*1) / (2*b*c)) = 2 * Real.sqrt 3 :=
begin
  sorry
end

theorem problem_I_2 : ∀ (a : ℝ) (b : ℝ), a = Real.sqrt 7 → b = 2 → ∃ c, b^2 + c^2 = a^2 + b*c ∧ triangle_area_2 b c (Real.acos (b^2 + c^2 - a^2) / (2*b*c)) = 3 * Real.sqrt 3 / 2 :=
begin
  sorry
end 

theorem problem_I_3 : ∀ (a : ℝ) (b : ℝ), a = Real.sqrt 7 → Real.sin (2 * c) = Real.sin b → ∃ b c, b^2 + c^2 - a^2 = b*c ∧ triangle_area_3 b c (Real.acos (b^2 + c^2 - a^2) / (2*b*c)) = 7 * Real.sqrt 3 / 6 :=
begin
  sorry
end 

-- Statement for problem (II)
theorem problem_II : ∀ (B : ℝ) (C : ℝ), (cos_B_C_sum (B) (C) = Real.sin (B + π / 6) ∧ ∃ (A : ℝ), A = π / 3 ∧ (B < 2 / 3 * π) ∧ Real.sin (B + π / 6) = 1) :=
begin
  sorry
end

end problem_I_1_problem_I_2_problem_I_3_problem_II_l339_339204


namespace intersection_M_N_l339_339953

def set_M : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def set_N : Set ℝ := {x | log 2 x > 1}

theorem intersection_M_N :
  set_M ∩ set_N = {x | 2 < x ∧ x ≤ 3} :=
by
  sorry

end intersection_M_N_l339_339953


namespace problem_solution_l339_339169

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l339_339169


namespace books_added_l339_339766

theorem books_added (initial_books final_books : ℕ) (h1 : initial_books = 38) (h2 : final_books = 48) : final_books - initial_books = 10 :=
by
  rw [h1, h2]
  sorry

end books_added_l339_339766


namespace factorial_division_sum_l339_339466

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339466


namespace max_integer_is_110003_l339_339340

def greatest_integer : Prop :=
  let a := 100004
  let b := 110003
  let c := 102002
  let d := 100301
  let e := 100041
  b > a ∧ b > c ∧ b > d ∧ b > e

theorem max_integer_is_110003 : greatest_integer :=
by
  sorry

end max_integer_is_110003_l339_339340


namespace factorial_division_l339_339409

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339409


namespace part1_daily_sales_profit_at_60_part2_selling_price_1350_l339_339819

-- Definitions from conditions
def cost_per_piece : ℕ := 40
def selling_price_50_sales_volume : ℕ := 100
def sales_decrease_per_dollar : ℕ := 2
def max_selling_price : ℕ := 65

-- Problem Part (1)
def profit_at_60_yuan := 
  let selling_price := 60
  let profit_per_piece := selling_price - cost_per_piece
  let sales_decrease := (selling_price - 50) * sales_decrease_per_dollar
  let sales_volume := selling_price_50_sales_volume - sales_decrease
  let daily_profit := profit_per_piece * sales_volume
  daily_profit

theorem part1_daily_sales_profit_at_60 : profit_at_60_yuan = 1600 := by
  sorry

-- Problem Part (2)
def selling_price_for_1350_profit :=
  let desired_profit := 1350
  let sales_volume (x : ℕ) := selling_price_50_sales_volume - sales_decrease_per_dollar * (x - 50)
  let profit_per_x_piece (x : ℕ) := x - cost_per_piece
  let daily_sales_profit (x : ℕ) := (profit_per_x_piece x) * (sales_volume x)
  daily_sales_profit

theorem part2_selling_price_1350 : 
  ∃ x, x ≤ max_selling_price ∧ selling_price_for_1350_profit x = 1350 ∧ x = 55 := by
  sorry

end part1_daily_sales_profit_at_60_part2_selling_price_1350_l339_339819


namespace solve_equation_2021_2020_l339_339259

theorem solve_equation_2021_2020 (x : ℝ) (hx : x ≥ 0) :
  2021 * (x^2020)^(1/202) - 1 = 2020 * x ↔ x = 1 :=
by {
  sorry
}

end solve_equation_2021_2020_l339_339259


namespace paul_has_five_dogs_l339_339245

theorem paul_has_five_dogs
  (w1 w2 w3 w4 w5 : ℕ)
  (food_per_10_pounds : ℕ)
  (total_food_required : ℕ)
  (h1 : w1 = 20)
  (h2 : w2 = 40)
  (h3 : w3 = 10)
  (h4 : w4 = 30)
  (h5 : w5 = 50)
  (h6 : food_per_10_pounds = 1)
  (h7 : total_food_required = 15) :
  (w1 / 10 * food_per_10_pounds) +
  (w2 / 10 * food_per_10_pounds) +
  (w3 / 10 * food_per_10_pounds) +
  (w4 / 10 * food_per_10_pounds) +
  (w5 / 10 * food_per_10_pounds) = total_food_required → 
  5 = 5 :=
by
  intros
  sorry

end paul_has_five_dogs_l339_339245


namespace probability_all_white_balls_drawn_l339_339810

theorem probability_all_white_balls_drawn (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h_total : total_balls = 15) (h_white : white_balls = 7) (h_black : black_balls = 8) (h_drawn : drawn_balls = 7) :
  (Nat.choose 7 7 : ℚ) / (Nat.choose 15 7 : ℚ) = 1 / 6435 := by
sorry

end probability_all_white_balls_drawn_l339_339810


namespace proof_problem_l339_339694

noncomputable def problem (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧
  (∀ x y, 0 < x ∧ x < y → f x < f y) ∧
  (f (-3) = 0) →
  {x : ℝ | x * f x > 0} = {x : ℝ | x < -3 ∨ x > 3}

theorem proof_problem (f : ℝ → ℝ) :
  problem f :=
begin
  sorry
end

end proof_problem_l339_339694


namespace common_difference_arithmetic_sequence_l339_339579

theorem common_difference_arithmetic_sequence 
    (a : ℕ → ℝ) 
    (S₅ : ℝ)
    (h_arith : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h₁ : a 4 + a 6 = 6)
    (h₂ : S₅ = (a 1 + a 2 + a 3 + a 4 + a 5))
    (h_S₅_val : S₅ = 10) :
  ∃ d : ℝ, d = (a 5 - a 1) / 4 ∧ d = 1/2 := 
by
  sorry

end common_difference_arithmetic_sequence_l339_339579


namespace households_using_both_brands_l339_339365

def households_surveyed : ℕ := 200
def neither_brand : ℕ := 80
def only_brand_W : ℕ := 60
def ratio_only_brand_B_to_both_brands : ℕ := 3

theorem households_using_both_brands :
  ∃ x : ℕ, (80 + 60 + 3 * x + x = 200) ∧ x = 15 :=
by
  exist 15
  simp
  sorry

end households_using_both_brands_l339_339365


namespace factorial_division_l339_339484

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339484


namespace angle_C_magnitude_l339_339922

variable {a b c : ℝ}

-- Conditions from the problem
def triangle_sides (a b c : ℝ) : Prop :=
a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0

def determinant_condition (a b c : ℝ) : Prop :=
(a * a - c * c) = (-b * b + a * b)

-- Proof goal
theorem angle_C_magnitude (a b c : ℝ) (ht : triangle_sides a b c) (hd : determinant_condition a b c) : 
  ∠C = π / 3 :=
sorry

end angle_C_magnitude_l339_339922


namespace smallest_positive_divisor_l339_339583

theorem smallest_positive_divisor
  (a b x₀ y₀ : ℤ)
  (h₀ : a ≠ 0 ∨ b ≠ 0)
  (h₁ : ∀ x y, a * x₀ + b * y₀ ≤ 0 ∨ a * x + b * y ≥ a * x₀ + b * y₀)
  (h₂ : 0 < a * x₀ + b * y₀):
  ∀ x y : ℤ, a * x₀ + b * y₀ ∣ a * x + b * y := 
sorry

end smallest_positive_divisor_l339_339583


namespace count_integers_between_sqrt8_and_sqrt75_l339_339119

theorem count_integers_between_sqrt8_and_sqrt75 : 
  let n1 := Real.sqrt 8,
      n2 := Real.sqrt 75 in 
  ∃ count : ℕ, count = (Int.floor n2 - Int.ceil n1) + 1 ∧ count = 6 :=
by
  let n1 := Real.sqrt 8
  let n2 := Real.sqrt 75
  use (Int.floor n2 - Int.ceil n1 + 1)
  simp only [Real.sqrt]
  sorry

end count_integers_between_sqrt8_and_sqrt75_l339_339119


namespace find_a_b_l339_339808

-- Definitions and conditions from the problem
def parabola (a b : ℝ) (x : ℝ) := a*x^2 + b*x - 7

def tangent_line (x y : ℝ) := 4 * x - y - 3 = 0

-- The Lean 4 statement for the proof problem
theorem find_a_b (a b : ℝ) :  
  parabola a b 1 = 1 ∧ tangent_line 1 (parabola a b 1) ∧ (derivative (parabola a b) eval_at 1 = 4) →
  a = -4 ∧ b = 12 :=
by sorry

end find_a_b_l339_339808


namespace jaxon_toys_l339_339679

-- Definitions as per the conditions
def toys_jaxon : ℕ := sorry
def toys_gabriel : ℕ := 2 * toys_jaxon
def toys_jerry : ℕ := 2 * toys_jaxon + 8
def total_toys : ℕ := toys_jaxon + toys_gabriel + toys_jerry

-- Theorem to prove
theorem jaxon_toys : total_toys = 83 → toys_jaxon = 15 := sorry

end jaxon_toys_l339_339679


namespace trig_identity_shift_l339_339321

theorem trig_identity_shift:
  (∀ x : ℝ, sin (2 * x + π / 6) = sin (2 * (x - π / 12) + π / 3)) :=
by
  sorry

end trig_identity_shift_l339_339321


namespace log_difference_l339_339872

theorem log_difference (log4_256 log4_1div64 : ℚ) (h1 : log4_256 = real.log 256 / real.log 4) 
                       (h2 : log4_1div64 = real.log (1/64) / real.log 4) :
  log4_256 - log4_1div64 = 7 := by
begin
  -- This is a placeholder, proof needs to be filled in
  sorry
end

end log_difference_l339_339872


namespace sum_p_q_r_eq_2_l339_339981

-- Define the sequence as given in the problem
def b (m : ℕ) : ℕ := 
  if h : m > 0 then 
    let k := (Nat.sqrt m)
    2 * k
  else 0

-- Define the sum p + q + r problem
theorem sum_p_q_r_eq_2 : ( ∃ p q r : ℤ, (∀ m : ℕ, 0 < m → b m = p * (Nat.sqrt (m + q)) + r) ∧ p + q + r = 2) :=
sorry

end sum_p_q_r_eq_2_l339_339981


namespace factorial_division_l339_339486

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339486


namespace expression_numerator_l339_339636

theorem expression_numerator (p q : ℕ) (E : ℕ) 
  (h1 : p * 5 = q * 4)
  (h2 : (18 / 7) + (E / (2 * q + p)) = 3) : E = 6 := 
by 
  sorry

end expression_numerator_l339_339636


namespace integer_count_between_sqrt8_and_sqrt75_l339_339137

noncomputable def m : ℤ := Int.ceil (Real.sqrt 8)
noncomputable def n : ℤ := Int.floor (Real.sqrt 75)

theorem integer_count_between_sqrt8_and_sqrt75 : (n - m + 1) = 6 := by
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339137


namespace expression_equals_36_l339_339162

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2 * (x + 2) * (4 - x) + (4 - x)^2 = 36 :=
by
  sorry

end expression_equals_36_l339_339162


namespace dave_deleted_apps_l339_339513

theorem dave_deleted_apps :
  ∃ d : ℕ, d = 150 - 65 :=
sorry

end dave_deleted_apps_l339_339513


namespace pin_code_permutations_count_l339_339839

theorem pin_code_permutations_count : (4.factorial = 24) := 
by {
    -- The proof would go here
    sorry
}

end pin_code_permutations_count_l339_339839


namespace Expected_and_Variance_l339_339067

variables (p : ℝ) (h : 0 ≤ p ∧ p ≤ 1)

def P (xi : ℕ) : ℝ := 
  if xi = 0 then p else if xi = 1 then 1 - p else 0

def E_xi : ℝ := 0 * P p 0 + 1 * P p 1

def D_xi : ℝ := (0 - E_xi p)^2 * P p 0 + (1 - E_xi p)^2 * P p 1

theorem Expected_and_Variance :
  (E_xi p = 1 - p) ∧ (D_xi p = p * (1 - p)) :=
sorry

end Expected_and_Variance_l339_339067


namespace north_east_paths_no_cross_red_l339_339149

theorem north_east_paths_no_cross_red : 
  let to_column (m n : ℕ) := choose (m + n) m  -- Number of paths to (m, n) in grid
                      /- Paths to critical points C and D with steps constraints -/
  let paths_through_C := to_column 7 1 * to_column 7 1
  let paths_through_D := to_column 7 3 * to_column 7 3
  let total_paths := paths_through_C + paths_through_D
  total_paths = 1274 := 
by 
  sorry -- This line is just a placeholder to indicate the proof is skipped

end north_east_paths_no_cross_red_l339_339149


namespace total_travel_time_l339_339709

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l339_339709


namespace factorial_sum_division_l339_339453

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339453


namespace ball_distribution_l339_339246

theorem ball_distribution:
  ∀ (white_balls black_balls boxes : ℕ),
  white_balls = 4 →
  black_balls = 5 →
  boxes = 3 →
  (∀ i, 1 ≤ i → i ≤ boxes → (∀ j, 1 ≤ j → j ≤ 2 → 
    (white_balls = 4 → black_balls = 5 → boxes = 3 → 
      (∃ (w b : ℕ) (w_dists b_dists : finset (fin boxes) → ℕ), 
        (w_dists = λ b, if b = 0 then 2 else 1) ∧ 
        (b_dists = λ b, if b = 0 then 3 else if b = 1 then 2 else 1) ∧ 
        white_balls = w → 
        black_balls = b → 
        boxes = 3 →
        λ n, w_dists = 3 ∧ (w_dists 0 + w_dists 1 + w_dists 2 = white_balls) ∧ 
        λ n, b_dists = 3 ∧ (b_dists 0 + b_dists 1 + b_dists 2 = black_balls) → 
        (∃ (total_arrangements : ℕ), total_arrangements = 18))))) :=
begin
  intros white_balls black_balls boxes hw hb hb1,
  intros,
  sorry
end

end ball_distribution_l339_339246


namespace ordering_of_radii_l339_339407

noncomputable def radius_A : ℝ := real.sqrt 16

noncomputable def radius_B : ℝ := real.sqrt (16π / π)

noncomputable def radius_C : ℝ := 10π / (2π)

theorem ordering_of_radii : radius_A = radius_B ∧ radius_A < radius_C :=
by
  have h1 : radius_A = 4 := by
    rw [radius_A]
    exact real.sqrt_sq (by norm_num)
  have h2 : radius_B = 4 := by 
    rw [radius_B, div_eq_mul_inv, mul_comm, real.sqrt_mul, mul_one]
    exact real.sqrt_sq (by norm_num)
  have h3 : radius_C = 5 := by
    rw [radius_C, div_eq_mul_one_div]
    norm_num
  exact ⟨h1.symm ▸ h2.symm ▸ rfl, h1.symm ▸ h3.symm ▸ by norm_num⟩


end ordering_of_radii_l339_339407


namespace parabola_hyperbola_focus_l339_339589

-- Definitions for the conditions
def focus_parabola (a : ℝ) : ℝ × ℝ :=
(0, a / 4)

def foci_hyperbola (n : ℕ) : Set (ℝ × ℝ) :=
{ (0, 2), (0, -2) }

-- The statement to prove
theorem parabola_hyperbola_focus (a : ℝ) (H : focus_parabola a ∈ foci_hyperbola 2) : a = 8 ∨ a = -8 :=
sorry

end parabola_hyperbola_focus_l339_339589


namespace find_S0_l339_339655

-- Define the arithmetic sequence and conditions
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n, a (n + 1) = a n + d

-- Define that the specified terms form a geometric sequence
def geometric_condition (a : ℕ → ℝ) := a 2 * a 8 = a 6 * a 6

-- Constants given in the problem
def a1 := 20
def common_diff : ℝ := -1/2 -- given from the solution

-- Define the sum of the first n terms of the arithmetic sequence
def sum_of_arithmetic (a : ℕ → ℝ) (n : ℕ) := ∑ i in finset.range n, a i

-- The final proof statement
theorem find_S0 (a : ℕ → ℝ) (d : ℝ) (h_seq : arithmetic_sequence a d) (h_a1 : a 0= 20) 
(h_geometric : geometric_condition a) :
  sum_of_arithmetic a 0 = 10 :=
by
  sorry

end find_S0_l339_339655


namespace value_of_b_l339_339288

theorem value_of_b (b : ℝ) : 
  (∃ P Q : ℝ × ℝ, P = (2, 4) ∧ Q = (8, 10) ∧ 
  (∀ M : ℝ × ℝ, M = ((2 + 8) / 2, (4 + 10) / 2) ∧ 
  ∃ l : ℝ → ℝ, (∀ x : ℝ, l x = b - x) ∧ l (fst M) = snd M)) → b = 12 := 
sorry

end value_of_b_l339_339288


namespace find_A_l339_339674

variables (a c : ℝ) (C A : ℝ)

-- Given conditions
def condition_1 : a = 4 * real.sqrt 3 := sorry
def condition_2 : c = 12 := sorry
def condition_3 : C = real.pi / 3 := sorry

theorem find_A : A = real.pi / 6 :=
by
  -- apply the given conditions
  have h1 : a = 4 * real.sqrt 3 := condition_1,
  have h2 : c = 12 := condition_2,
  have h3 : C = real.pi / 3 := condition_3,
  sorry

end find_A_l339_339674


namespace find_a_l339_339089

def f (a x : ℝ) := a^x + log a x

theorem find_a (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) 
  (h3 : ∃ x ∈ set.Icc 1 2, ∀ y ∈ set.Icc 1 2, f a y ≤ f a x) 
  (h4 : ∃ x ∈ set.Icc 1 2, ∀ y ∈ set.Icc 1 2, f a x ≤ f a y) 
  (h5 : ∃ x_max x_min ∈ set.Icc 1 2, f a x_max + f a x_min = log a 2 + 6) : a = 2 := 
sorry

end find_a_l339_339089


namespace complex_magnitude_l339_339907

theorem complex_magnitude (z : ℂ) (i_unit : ℂ) (h : i_unit = complex.I) : 
  (z = (1 - i_unit) / i_unit) → |z| = real.sqrt (2) :=
by
  intro hz
  rw hz
  simp
  sorry

end complex_magnitude_l339_339907


namespace possible_values_of_m_l339_339669

theorem possible_values_of_m
  (m : ℕ)
  (h1 : ∃ (m' : ℕ), m = m' ∧ 0 < m)            -- m is a positive integer
  (h2 : 2 * (m - 1) + 3 * (m + 2) > 4 * (m - 5))    -- AB + AC > BC
  (h3 : 2 * (m - 1) + 4 * (m + 5) > 3 * (m + 2))    -- AB + BC > AC
  (h4 : 3 * (m + 2) + 4 * (m + 5) > 2 * (m - 1))    -- AC + BC > AB
  (h5 : 3 * (m + 2) > 2 * (m - 1))                  -- AC > AB
  (h6 : 4 * (m + 5) > 3 * (m + 2))                  -- BC > AC
  : m ≥ 7 := 
sorry

end possible_values_of_m_l339_339669


namespace area_unpainted_region_l339_339323

-- Define the parameters and conditions given in the problem
def width_board1 : ℝ := 5 -- width of the first board
def width_board2 : ℝ := 7 -- width of the second board
def angle_crossing : ℝ := 45 -- angle at their crossing, in degrees

-- Convert angle from degrees to radians for trigonometric calculations
def angle_crossing_rad : ℝ := Real.pi * angle_crossing / 180

-- The area computation statement
theorem area_unpainted_region : 
  (width_board1 * width_board1 * Real.sqrt 2) = 25 * Real.sqrt 2 :=
by 
  -- Translate the problem conditions
  have board1_width : width_board1 = 5 := rfl,
  have board2_width : width_board2 = 7 := rfl,
  have crossing_angle : angle_crossing = 45 := rfl,
  have crossing_angle_rad : angle_crossing_rad = Real.pi * 45 / 180 := rfl,

  -- Proof outline skipped
  sorry

end area_unpainted_region_l339_339323


namespace paving_rate_l339_339291

variables (L W C : ℝ) (Area : ℝ)
hypothesis hL : L = 5.5
hypothesis hW : W = 4
hypothesis hC : C = 17600
hypothesis hArea : Area = L * W

theorem paving_rate :
  C / Area = 800 :=
by
  rw [hArea, hL, hW, hC]
  simp
  sorry

end paving_rate_l339_339291


namespace factorial_div_sum_l339_339419

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339419


namespace inverse_proportion_quad_l339_339638

theorem inverse_proportion_quad (k : ℝ) : (∀ x : ℝ, x > 0 → (k + 1) / x < 0) ∧ (∀ x : ℝ, x < 0 → (k + 1) / x > 0) ↔ k < -1 :=
by
  sorry

end inverse_proportion_quad_l339_339638


namespace problem_statement_l339_339062
open complex real

noncomputable theory

def z1 : ℂ := 1 - 𝒾
def z2 (x y : ℝ) : ℂ := x + y * 𝒾
def OZ1 : ℂ := 1 - 𝒾  -- vector representation of z1

theorem problem_statement (x y : ℝ) (h1: x = 0 ∨ (OZ1 = z1 ∧ (x = -y ∨ x = y))) :
  (x = 0 → z2 x y = y * 𝒾) ∧
  (OZ1 = z1 ∧ x = -y → x + y = 0) ∧
  (OZ1 = z1 ∧ x = y → abs (z1 + z2 x y) = abs (z1 - z2 x y)) :=
begin
  sorry
end

end problem_statement_l339_339062


namespace quadratic_polynomial_l339_339013

def q (x : ℝ) : ℝ := -x^2 - 6 * x + 27

theorem quadratic_polynomial :
  (q (-9) = 0) ∧ (q 3 = 0) ∧ (q 6 = -45) ↔ ∀ x, q(x) = -x^2 - 6*x + 27 :=
by
  sorry

end quadratic_polynomial_l339_339013


namespace disk_tangency_position_l339_339362

theorem disk_tangency_position :
  ∀ (r_clock : ℝ) (r_disk : ℝ) (start_position : ℕ)
    (clockwise : Bool) (initial_arrow_right : Bool),
    r_clock = 30 → r_disk = 15 → start_position = 3 →
    clockwise = true → initial_arrow_right = true →
    tangent_position(r_clock, r_disk, start_position, clockwise, initial_arrow_right) = 9 := 
by 
  sorry

end disk_tangency_position_l339_339362


namespace max_value_of_k_l339_339929

theorem max_value_of_k (x y : ℝ) (hx : 0 < x) (hy : 0 < y)
  (h : ∀ x y : ℝ, 0 < x → 0 < y → (x + 2 * y) / (x * y) ≥ k / (2 * x + y)) :
  k ≤ 9 :=
by
  sorry

end max_value_of_k_l339_339929


namespace true_propositions_eq_option_B_l339_339769

-- Defining the propositions as per given conditions
def proposition_1 : Prop := (∀ x y : ℝ, x + y = 0 → (y = -x)) → (∀ x y : ℝ, (y = -x) → x + y = 0)
def proposition_2 : Prop := ¬(∀ (T1 T2 : Triangle), congruent T1 T2 → area T1 = area T2)
def proposition_3 : Prop := (∀ q : ℝ, q ≤ 1 → ∃ x : ℝ, x * (x + 2) + q = 0) → (∀ q : ℝ, q ≤ 1 → (∃ x : ℝ, x * (x + 2) + q = 0))
def proposition_4 : Prop := (¬(∀ (T : Triangle), scalene T → (interior_angles T = 60 ∧ 60 ∧ 60))) → (¬(∀ (T : Triangle), interior_angles T = 60 ∧ 60 ∧ 60 → scalene T))

-- The theorem statement that proves the true propositions are {proposition_1, proposition_3}
theorem true_propositions_eq_option_B : 
    (let P1 := (∀ x y : ℝ, x + y = 0 → (y = -x)) → (∀ x y : ℝ, (y = -x) → x + y = 0), 
         P2 := ¬(∀ (T1 T2 : Triangle), congruent T1 T2 → area T1 = area T2), 
         P3 := (∀ q : ℝ, q ≤ 1 → ∃ x : ℝ, x * (x + 2) + q = 0) → (∀ q : ℝ, q ≤ 1 → (∃ x : ℝ, x * (x + 2) + q = 0)), 
         P4 := (¬(∀ (T : Triangle), scalene T → (interior_angles T = 60 ∧ 60 ∧ 60))) → (¬(∀ (T : Triangle), interior_angles T = 60 ∧ 60 ∧ 60 → scalene T))) 
    ∧ ({P1, P3} = {p: Prop | p = P1 ∨ p = P3})
sorry

end true_propositions_eq_option_B_l339_339769


namespace two_digit_number_exists_l339_339385

theorem two_digit_number_exists (x : ℕ) (h1 : 1 ≤ x) (h2 : x ≤ 9) :
  (9 * x + 8) * (80 - 9 * x) = 1855 → (9 * x + 8 = 35 ∨ 9 * x + 8 = 53) := by
  sorry

end two_digit_number_exists_l339_339385


namespace factorial_division_identity_l339_339502

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339502


namespace number_of_integers_between_sqrts_l339_339133

theorem number_of_integers_between_sqrts :
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  (upper_bound - lower_bound + 1) = 6 :=
by
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  have sqrt8_bound : (2 : ℝ) < sqrt8 ∧ sqrt8 < 3 := sorry
  have sqrt75_bound : (8 : ℝ) < sqrt75 ∧ sqrt75 < 9 := sorry
  have lower_bound_def : lower_bound = 3 := sorry
  have upper_bound_def : upper_bound = 8 := sorry
  show (upper_bound - lower_bound + 1) = 6
  calc
    upper_bound - lower_bound + 1 = 8 - 3 + 1 := by rw [upper_bound_def, lower_bound_def]
                             ... = 6 := by norm_num
  sorry

end number_of_integers_between_sqrts_l339_339133


namespace triangle_cos_C_correct_l339_339182

noncomputable def triangle_cos_C (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : ℝ :=
  Real.cos C -- This will be defined correctly in the proof phase.

theorem triangle_cos_C_correct (A B C : ℝ) (hABC : A + B + C = π)
  (hSinA : Real.sin A = 3 / 5) (hCosB : Real.cos B = 5 / 13) : 
  triangle_cos_C A B C hABC hSinA hCosB = 16 / 65 :=
sorry

end triangle_cos_C_correct_l339_339182


namespace smallest_y_for_perfect_fourth_power_l339_339826

-- Define the conditions
def x : ℕ := 7 * 24 * 48
def y : ℕ := 6174

-- The theorem we need to prove
theorem smallest_y_for_perfect_fourth_power (x y : ℕ) 
  (hx : x = 7 * 24 * 48) 
  (hy : y = 6174) : ∃ k : ℕ, (∃ z : ℕ, z * z * z * z = x * y) :=
sorry

end smallest_y_for_perfect_fourth_power_l339_339826


namespace smallest_k_divides_l339_339041

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l339_339041


namespace sum_of_first_10_common_elements_correct_l339_339882

noncomputable def sum_of_first_10_common_elements : ℕ :=
  let ap := {n | ∃ m : ℕ, n = 4 + 3 * m}
  let gp := {n | ∃ k : ℕ, n = 10 * 2^k}
  let common_elements := {n | n ∈ ap ∧ n ∈ gp}
  let common_elements_list := List.filter (λ n, n ∈ common_elements) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  List.sum (List.take 10 common_elements_list)

theorem sum_of_first_10_common_elements_correct : 
  sum_of_first_10_common_elements = 3495250 :=
sorry

end sum_of_first_10_common_elements_correct_l339_339882


namespace fruit_prob_l339_339197

variable (O A B S : ℕ) 

-- Define the conditions
variables (H1 : O + A + B + S = 32)
variables (H2 : O - 5 = 3)
variables (H3 : A - 3 = 7)
variables (H4 : S - 2 = 4)
variables (H5 : 3 + 7 + 4 + B = 20)

-- Define the proof problem
theorem fruit_prob :
  (O = 8) ∧ (A = 10) ∧ (B = 6) ∧ (S = 6) → (O + S) / (O + A + B + S) = 7 / 16 := 
by
  sorry

end fruit_prob_l339_339197


namespace cone_development_angle_l339_339637

theorem cone_development_angle
  (r : ℝ)
  (h_cone : True) -- Placeholder for all conditions if needed for completeness
  : central_angle r = 180 :=
sorry

end cone_development_angle_l339_339637


namespace find_m_plus_n_l339_339366

noncomputable def probability_to_axes (a b : ℕ) : ℚ :=
if a = 0 ∧ b = 0 then 1
else if a = 0 ∧ b > 0 then 0
else if a > 0 ∧ b = 0 then 0
else (1 / 4) * probability_to_axes (a - 1) b + (1 / 4) * probability_to_axes a (b - 1) + (1 / 4) * probability_to_axes (a - 1) (b - 1) + (1 / 4) * probability_to_axes (a - 2) (b - 1)

theorem find_m_plus_n : 
  let p := probability_to_axes 6 6 in 
  ∃ m n : ℕ, p = m / 4^n ∧ ¬(4 ∣ m) ∧ m + n = 138 :=
sorry

end find_m_plus_n_l339_339366


namespace average_speed_round_trip_l339_339235

noncomputable def average_speed (d : ℝ) (v_to v_from : ℝ) : ℝ :=
  let time_to := d / v_to
  let time_from := d / v_from
  let total_time := time_to + time_from
  let total_distance := 2 * d
  total_distance / total_time

theorem average_speed_round_trip (d : ℝ) :
  average_speed d 60 40 = 48 :=
by
  sorry

end average_speed_round_trip_l339_339235


namespace min_club_members_l339_339818

theorem min_club_members (n : ℕ) :
  (∀ k : ℕ, k = 8 ∨ k = 9 ∨ k = 11 → n % k = 0) ∧ (n ≥ 300) → n = 792 :=
sorry

end min_club_members_l339_339818


namespace range_of_m_l339_339088

noncomputable def f : ℝ → ℝ → ℝ :=
  λ m x, if x < 2 then 2^(x - m) else (m * x) / (4 * x^2 + 16)

theorem range_of_m {x1 : ℝ} (hx1 : x1 ≥ 2) :
  ∃ x2 : ℝ, x2 ≤ 2 ∧ f ≠ x1 x)x1 m = f ≠ x1 x)x2 m → 
  m ≤ 4 := 
begin
  sorry
end

end range_of_m_l339_339088


namespace determine_n_l339_339524

open Nat

def candy_game (n : ℕ) : Prop :=
  ∀ k : ℕ, ∃ k', (k' (k' + 1) / 2) % n = k

theorem determine_n :
  ∀ n : ℕ, candy_game n ↔ ∃ k : ℕ, n = 2^k :=
sorry

end determine_n_l339_339524


namespace percentage_of_students_enrolled_is_40_l339_339173

def total_students : ℕ := 880
def not_enrolled_in_biology : ℕ := 528
def enrolled_in_biology : ℕ := total_students - not_enrolled_in_biology
def percentage_enrolled : ℕ := (enrolled_in_biology * 100) / total_students

theorem percentage_of_students_enrolled_is_40 : percentage_enrolled = 40 := by
  -- Beginning of the proof
  sorry

end percentage_of_students_enrolled_is_40_l339_339173


namespace solution_of_equation_l339_339261

noncomputable def solve_equation (x : ℝ) : Prop :=
  2021 * x^(2020/202) - 1 = 2020 * x ∧ x ≥ 0

theorem solution_of_equation : solve_equation 1 :=
by {
  sorry,
}

end solution_of_equation_l339_339261


namespace fraction_of_price_l339_339849

def original_price : ℝ := 180
def savings : ℝ := 80
def paid_price : ℝ := original_price - savings
def fraction_price_paid : ℝ := paid_price / original_price
def expected_fraction : ℝ := 11 / 18

theorem fraction_of_price (h : paid_price = original_price * fraction_price_paid - 10)
    : fraction_price_paid = expected_fraction :=
by
  -- We only need to state the theorem, not prove it
  sorry

end fraction_of_price_l339_339849


namespace lcm_10_to_30_l339_339539

def list_of_ints := [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]

def lcm_of_list (l : List Nat) : Nat :=
  l.foldr Nat.lcm 1

theorem lcm_10_to_30 : lcm_of_list list_of_ints = 232792560 :=
  sorry

end lcm_10_to_30_l339_339539


namespace artificial_scarcity_strategy_interview_strategy_l339_339736

-- Conditions for Part (a)
variable (high_end_goods : Type)      -- Type representing high-end goods
variable (manufacturers : high_end_goods → Prop)  -- Property of being a manufacturer
variable (sufficient_resources : high_end_goods → Prop) -- Sufficient resources to produce more
variable (demand : high_end_goods → ℤ)   -- Demand for the product
variable (supply : high_end_goods → ℤ)   -- Supply of the product 
variable (price : ℤ)   -- Price of the product

-- Theorem for Part (a)
theorem artificial_scarcity_strategy (H1 : ∀ g : high_end_goods, manufacturers g → sufficient_resources g → demand g = 3000 ∧ supply g = 200 ∧ price = 15000) :
  ∀ g : high_end_goods, manufacturers g → maintain_exclusivity g :=
sorry

-- Conditions for Part (b)
variable (interview_required : high_end_goods → Prop)   -- Interview requirement for purchase
variable (purchase_history : Prop)  -- Previous purchase history

-- Advantages and Disadvantages from Part (b)
def selective_clientele : Prop := sorry   -- Definition of selective clientele
def enhanced_exclusivity : Prop := sorry   -- Definition of enhanced exclusivity
def increased_transaction_costs : Prop := sorry   -- Definition of increased transaction costs

-- Theorem for Part (b)
theorem interview_strategy (H2 : ∀ g : high_end_goods, manufacturers g → interview_required g → purchase_history) :
  (selective_clientele ∧ enhanced_exclusivity) ∧ increased_transaction_costs :=
sorry

end artificial_scarcity_strategy_interview_strategy_l339_339736


namespace smallest_k_l339_339039

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l339_339039


namespace evaluate_fg_l339_339698

def f (x : ℝ) : ℝ := 3 * Real.sqrt x + 18 / Real.sqrt x

def g (x : ℝ) : ℝ := 3 * x^2 - 3 * x - 4

theorem evaluate_fg : f (g 4) = 14.25 * Real.sqrt 2 := by
  sorry

end evaluate_fg_l339_339698


namespace bob_spending_over_limit_l339_339717

theorem bob_spending_over_limit : 
  ∀ (necklace_price book_price limit total_cost amount_over_limit : ℕ),
  necklace_price = 34 →
  book_price = necklace_price + 5 →
  limit = 70 →
  total_cost = necklace_price + book_price →
  amount_over_limit = total_cost - limit →
  amount_over_limit = 3 :=
by
  intros
  sorry

end bob_spending_over_limit_l339_339717


namespace tangent_line_formula_at_A_eq0_l339_339747

open Real 

/--
Let \( f \colon \mathbb{R} \to \mathbb{R} \) be defined as \( f(x) = \exp(x) + 2x \).
Let \( A \) be the point \((0, f(0))\), i.e., \( A = (0, 1) \).
We want to prove that the equation of the tangent line to the curve \( y = f(x) \) at point \( A \) is \( 3x - y + 1 = 0 \).
-/
theorem tangent_line_formula_at_A_eq0 : 
  let f (x : ℝ) := exp x + 2 * x,
      A := (0 : ℝ, 1 : ℝ)
  in  (3 : ℝ) * (0 : ℝ) - (1 : ℝ) + (1 : ℝ) = (0 : ℝ) → 
      (f 0 = 1) →
      (∀ x, deriv f x = (exp x + 2)) → 
      (3 * (0 : ℝ) - (f 0) + 1 = 0) :=
by
  sorry

end tangent_line_formula_at_A_eq0_l339_339747


namespace set_union_eq_l339_339951

open Set

noncomputable def A : Set ℤ := {x | x^2 - x = 0}
def B : Set ℤ := {-1, 0}
def C : Set ℤ := {-1, 0, 1}

theorem set_union_eq :
  A ∪ B = C :=
by {
  sorry
}

end set_union_eq_l339_339951


namespace parabola_and_hyperbola_focus_equal_l339_339176

noncomputable def parabola_focus (p : ℝ) : (ℝ × ℝ) :=
(2, 0)

noncomputable def hyperbola_focus : (ℝ × ℝ) :=
(2, 0)

theorem parabola_and_hyperbola_focus_equal
  (p : ℝ)
  (h_parabola : parabola_focus p = (2, 0))
  (h_hyperbola : hyperbola_focus = (2, 0)) :
  p = 4 := by
  sorry

end parabola_and_hyperbola_focus_equal_l339_339176


namespace solution_for_m_exactly_one_solution_l339_339521

theorem solution_for_m_exactly_one_solution (m : ℚ) : 
  (∀ x : ℚ, (x - 3) / (m * x + 4) = 2 * x → 
            (2 * m * x^2 + 7 * x + 3 = 0)) →
  (49 - 24 * m = 0) → 
  m = 49 / 24 :=
by
  intro h1 h2
  sorry

end solution_for_m_exactly_one_solution_l339_339521


namespace factorial_division_l339_339417

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339417


namespace find_quadratic_equation_l339_339530

noncomputable def quadratic_equation_with_root (r : ℝ) : Prop :=
  ∃ (a b c : ℚ), (a ≠ 0) ∧ a * r^2 + b * r + c = 0

theorem find_quadratic_equation : quadratic_equation_with_root (sqrt 5 - 3) ∧ 
                                 ∀ (x : ℝ), ((x^2 + 6 * x - 4) = 0 ↔ x = sqrt 5 - 3 ∨ x = -sqrt 5 - 3) :=
by
  sorry

end find_quadratic_equation_l339_339530


namespace find_final_sale_price_l339_339015

noncomputable def original_price := 600
noncomputable def discount_seq : List Float := [22.0, 35.0, 15.0, 7.0]

def apply_discount (price : Float) (discount : Float) : Float :=
  price - (price * (discount / 100.0))

def successive_discounts (price : Float) (discounts : List Float) : Float :=
  discounts.foldl apply_discount price

theorem find_final_sale_price :
  successive_discounts original_price discount_seq = 240.47 :=
by
  sorry

end find_final_sale_price_l339_339015


namespace roots_of_polynomial_l339_339014

noncomputable def polynomial := (6 : ℚ) * X^4 + 19 * X^3 - 51 * X^2 + 20 * X

theorem roots_of_polynomial :
  ∀ (x : ℚ), polynomial.eval x polynomial = 0 ↔ x ∈ {0, 1/2, 4/3, -5} :=
by
  sorry

end roots_of_polynomial_l339_339014


namespace hyperbola_center_l339_339535

theorem hyperbola_center :
  ∀ (x y : ℝ), 
  (4 * x + 8)^2 / 36 - (3 * y - 6)^2 / 25 = 1 → (x, y) = (-2, 2) :=
by
  intros x y h
  sorry

end hyperbola_center_l339_339535


namespace sum_three_digit_numbers_l339_339319

theorem sum_three_digit_numbers : 
  let digits := {0, 1, 2, 3, 4, 5, 8}
  let num_digits := 7
  let non_zero_digits := {1, 2, 3, 4, 5, 8}
  let num_non_zero_digits := 6
  (∑ (i : ℕ) in non_zero_digits, i * 49 * 100 + 
   ∑ (j : ℕ) in digits, j * 42 * 10 + 
   ∑ (k : ℕ) in digits, k * 42) = 123326 :=
by
  sorry

end sum_three_digit_numbers_l339_339319


namespace isosceles_triangle_max_s_l339_339986

variable (A B C P A' B' C' : Type) [MetricSpace A]
variables {a b : ℝ} (h1 : a ≤ b) (h2 : dist A B = b) (h3 : dist A C = b) (h4 : dist B C = a)
variables (h5 : dist A P + dist P A' = dist A A') (h6 : dist B P + dist P B' = dist B B') (h7 : dist C P + dist P C' = dist C C')

theorem isosceles_triangle_max_s 
    (AB AC BC : A ≠ B ∧ A ≠ C ∧ B ≠ C)
    (P_inside : Metric.inside_triangle A B C P)
    : dist A A' + dist B B' + dist C C' ≤ 2 * b + a :=
begin
  sorry
end

end isosceles_triangle_max_s_l339_339986


namespace problem1_solution_problem2_solution_l339_339735

theorem problem1_solution (x : ℝ) (h : 5 / (x - 1) = 1 / (2 * x + 1)) : x = -2 / 3 := sorry

theorem problem2_solution (x : ℝ) (h : 1 / (x - 2) + 2 = (1 - x) / (2 - x)) : false := sorry

end problem1_solution_problem2_solution_l339_339735


namespace geometric_sequence_min_value_l339_339936

theorem geometric_sequence_min_value 
  (a b c : ℝ)
  (h1 : b^2 = ac)
  (h2 : b = -Real.exp 1) :
  ac = Real.exp 2 := 
by
  sorry

end geometric_sequence_min_value_l339_339936


namespace smallest_n_satisfying_f_l339_339699

noncomputable def f : ℕ → ℤ
| 1 := 0
| n := if isPrime n then 1 else
       let factors := (factors n).map (λ p => (p, f p)) in
       factors.foldl (λ acc (p, fp) => acc + (n / p) * fp) 0

theorem smallest_n_satisfying_f : ∃ n : ℕ, n ≥ 2015 ∧ f n = n ∧ ∀ m : ℕ, m ≥ 2015 → f m = m → n ≤ m :=
  let n := 3125 in
  ⟨n, by decide, by decide, by decide⟩

end smallest_n_satisfying_f_l339_339699


namespace angle_bisector_in_triangle_l339_339684

open EuclideanGeometry

theorem angle_bisector_in_triangle
  {A B C D E F : Point}
  (hABC : ∠ A B C = 90)
  (hABBC : distance A B > distance B C)
  (hBD : distance B D = distance B C)
  (hE : foot_of_perpendicular D A C E)
  (hF : reflection_point B C D F) :
  is_angle_bisector E C (angle A B F) :=
sorry

end angle_bisector_in_triangle_l339_339684


namespace seashells_total_l339_339730

theorem seashells_total (S J : ℕ) (hS : S = 35) (hJ : J = 18) : S + J = 53 :=
by
  rw [hS, hJ]
  -- Here, we would compute the sum 35 + 18, but we use sorry to skip the actual proof steps.
  sorry

end seashells_total_l339_339730


namespace six_digit_integers_count_l339_339617

theorem six_digit_integers_count : 
  let digits := [2, 2, 2, 5, 5, 9] in
  multiset.card (multiset.of_list (list.permutations digits).erase_dup) = 60 :=
by
  sorry

end six_digit_integers_count_l339_339617


namespace fraction_of_defective_engines_l339_339767

theorem fraction_of_defective_engines
  (total_batches : ℕ)
  (engines_per_batch : ℕ)
  (non_defective_engines : ℕ)
  (H1 : total_batches = 5)
  (H2 : engines_per_batch = 80)
  (H3 : non_defective_engines = 300)
  : (total_batches * engines_per_batch - non_defective_engines) / (total_batches * engines_per_batch) = 1 / 4 :=
by
  -- Proof goes here.
  sorry

end fraction_of_defective_engines_l339_339767


namespace prob_AB_l339_339902

noncomputable def P (event : Type) : ℝ := sorry -- assume this is the probability function

variables (A B : Type) -- events

-- Given conditions
axiom P_B_given_A : P B | A = 1/3
axiom P_A : P A = 2/5

-- The goal
theorem prob_AB : P (A ∩ B) = 2 / 15 := sorry

end prob_AB_l339_339902


namespace distinguishable_paintings_l339_339820

noncomputable def num_distinguishable_paintings : ℕ :=
  let total_arrangements : ℕ := nat.factorial 5
  let num_rotations : ℕ := 8
  total_arrangements / num_rotations

theorem distinguishable_paintings : num_distinguishable_paintings = 15 :=
by {
  have total_arrangements := 120,
  have num_rotations := 8,
  show total_arrangements / num_rotations = 15,
  exact nat.div_eq_of_eq_mul_right zero_lt_eight (by norm_num : 8 * 15 = 120)
}

end distinguishable_paintings_l339_339820


namespace solution_of_equation_l339_339262

noncomputable def solve_equation (x : ℝ) : Prop :=
  2021 * x^(2020/202) - 1 = 2020 * x ∧ x ≥ 0

theorem solution_of_equation : solve_equation 1 :=
by {
  sorry,
}

end solution_of_equation_l339_339262


namespace republicans_in_house_l339_339763

theorem republicans_in_house (D R : ℕ) (h1 : D + R = 434) (h2 : R = D + 30) : R = 232 :=
by sorry

end republicans_in_house_l339_339763


namespace proof_fraction_problem_l339_339529

def fraction_problem :=
  (1 / 5 + 1 / 3) / (3 / 4 - 1 / 8) = 64 / 75

theorem proof_fraction_problem : fraction_problem :=
by
  sorry

end proof_fraction_problem_l339_339529


namespace fisherman_catch_total_l339_339821

theorem fisherman_catch_total (initial_lines broken_lines : ℕ) (fish_per_line : ℕ) : 
  initial_lines = 226 →
  broken_lines = 3 →
  fish_per_line = 3 →
  (initial_lines - broken_lines) * fish_per_line = 669 := 
by 
  intros h1 h2 h3 
  rw [h1, h2, h3]
  exact rfl

end fisherman_catch_total_l339_339821


namespace jerry_remaining_money_l339_339210

-- Define initial money
def initial_money := 18

-- Define amount spent on video games
def spent_video_games := 6

-- Define amount spent on a snack
def spent_snack := 3

-- Define total amount spent
def total_spent := spent_video_games + spent_snack

-- Define remaining money after spending
def remaining_money := initial_money - total_spent

theorem jerry_remaining_money : remaining_money = 9 :=
by
  sorry

end jerry_remaining_money_l339_339210


namespace complex_ratio_of_cubes_l339_339231

theorem complex_ratio_of_cubes (x y z : ℂ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 10) (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 8 :=
by
  sorry

end complex_ratio_of_cubes_l339_339231


namespace find_original_numbers_l339_339382

-- Definitions corresponding to the conditions in a
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ x y : ℕ, x + y = 8 ∧ n = 10 * x + y

-- Definitions to state the condition about the swapped number and their product
def swapped_number (n : ℕ) : ℕ :=
  let x := n / 10 in
  let y := n % 10 in
  10 * y + x

def product_of_numbers (n : ℕ) : Prop :=
  n * swapped_number(n) = 1855

-- Statement combining conditions and correct answer
theorem find_original_numbers (n : ℕ) :
  is_valid_number n ∧ product_of_numbers n → n = 35 ∨ n = 53 :=
sorry

end find_original_numbers_l339_339382


namespace calc1_calc2_l339_339809

theorem calc1 : 0.064^(1/3) - ((-7)/8)^0 + ((-2)^3)^(-4/3) + | -0.01|^(1/2) = -7/16 := by sorry

theorem calc2 : (3/4) * log 10 25 + 2^(log 2 3) + log 10 (2 * (sqrt 2)) = 9/2 := by sorry

end calc1_calc2_l339_339809


namespace middle_group_frequency_l339_339998

theorem middle_group_frequency (f : ℕ) (A : ℕ) (h_total : A + f = 100) (h_middle : f = A) : f = 50 :=
by
  sorry

end middle_group_frequency_l339_339998


namespace closest_integer_sum_l339_339401

theorem closest_integer_sum :
  let s := (2000 / 9) * (∑ k in range (4 + 1), 1 / k - ∑ j in range (4), 1 / (98 + j)) in
  s.floor = 458 :=
by
  sorry

end closest_integer_sum_l339_339401


namespace chuck_distance_outbound_l339_339241

noncomputable def calculate_distance (total_time hours: ℝ) (speed_outbound speed_return: ℝ) : ℝ :=
  let T1 := total_time / (1 + (speed_outbound / speed_return))
  in speed_outbound * T1

theorem chuck_distance_outbound :
  let total_time := 3 in
  let speed_outbound := 16 in
  let speed_return := 24 in
  calculate_distance total_time speed_outbound speed_return = 28.8 := 
by
  sorry

end chuck_distance_outbound_l339_339241


namespace cafeteria_extra_fruits_l339_339297

theorem cafeteria_extra_fruits (red_apples green_apples bananas oranges students : ℕ) (fruits_per_student : ℕ)
  (h1 : red_apples = 33)
  (h2 : green_apples = 23)
  (h3 : bananas = 17)
  (h4 : oranges = 12)
  (h5 : students = 21)
  (h6 : fruits_per_student = 2) :
  (red_apples + green_apples + bananas + oranges) - (students * fruits_per_student) = 43 :=
by
  sorry

end cafeteria_extra_fruits_l339_339297


namespace part_1_part_2_l339_339604

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a * Real.log x + (1 - x^2) / x^2

theorem part_1 (a : ℝ) :
  (∃ x : ℝ, f a x = 0) → a = 2 :=
  sorry

theorem part_2 (a : ℝ) :
  a = 2 → ∀ x ∈ Icc (1 : ℝ) (2 : ℝ), f a x ≤ (deriv (f a)) x :=
  sorry

end part_1_part_2_l339_339604


namespace ratio_PA_AB_l339_339670

-- Define the triangle and its ratio conditions
variable (A B C P : Type)
variables (hABC : Triangle A B C)
variables (hAC_CB : ratio A C C B = 3 / 4)
variables (hAP_BA : BisectorExAngle C P B A)

-- State the theorem that proves the required ratio
theorem ratio_PA_AB (PA AB : ℝ) (h : ratio PA AB = 3 / 1) : ratio PA AB = 3 / 1 :=
sorry

end ratio_PA_AB_l339_339670


namespace area_ratio_1_over_11_l339_339573

namespace Geometry

open_locale real

structure Point :=
(x : ℝ)
(y : ℝ)

noncomputable def midpoint (A B : Point) : Point :=
⟨(A.x + B.x) / 2, (A.y + B.y) / 2⟩

noncomputable def rect_area (A B C D : Point) : ℝ :=
abs ((B.x - A.x) * (D.y - A.y))

noncomputable def triangle_area (P Q R : Point) : ℝ :=
0.5 * abs ((Q.x - P.x) * (R.y - P.y) - (R.x - P.x) * (Q.y - P.y))

noncomputable def ratio_of_areas (P Q R : Point) (A B C D : Point) (E F : Point) : ℝ :=
(triangle_area P Q R) / (triangle_area A B E + triangle_area A E F)

def Point_A : Point := ⟨0, 0⟩
def Point_B : Point := ⟨2, 0⟩
def Point_C : Point := ⟨2, 1⟩
def Point_D : Point := ⟨0, 1⟩
noncomputable def Point_E : Point := midpoint Point_B Point_D
def Point_F : Point := ⟨0, 3/4⟩

theorem area_ratio_1_over_11 :
  ratio_of_areas Point_D Point_F Point_E Point_A Point_B Point_C Point_D Point_E Point_F = 1 / 11 :=
sorry

end Geometry

end area_ratio_1_over_11_l339_339573


namespace cube_root_floor_sum_equiv_l339_339690

def floor (x : ℝ) : ℝ := x.toInt

theorem cube_root_floor_sum_equiv : 
  (∑ k in Finset.range 2000, floor (Real.cbrt ((k + 1 : ℝ) * (k + 2) * (k + 3))) ) = 2001000 := 
by 
  sorry

end cube_root_floor_sum_equiv_l339_339690


namespace factorial_computation_l339_339477

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339477


namespace greatest_power_of_2_in_12_500_minus_6_500_l339_339785

theorem greatest_power_of_2_in_12_500_minus_6_500 :
  Nat.findGreatestPowerOfTwo (12^500 - 6^500) = 501 :=
by
  sorry

end greatest_power_of_2_in_12_500_minus_6_500_l339_339785


namespace sum_of_interior_angles_l339_339300

theorem sum_of_interior_angles (h_triangle : ∀ (a b c : ℝ), a + b + c = 180)
    (h_quadrilateral : ∀ (a b c d : ℝ), a + b + c + d = 360) :
  (∀ (n : ℕ), n ≥ 3 → ∀ (angles : Fin n → ℝ), (Finset.univ.sum angles) = (n-2) * 180) :=
by
  intro n h_n angles
  sorry

end sum_of_interior_angles_l339_339300


namespace part_1_property_part_2_property_part_3_geometric_l339_339691

-- Defining properties
def prop1 (a : ℕ → ℕ) (i j m: ℕ) : Prop := i > j ∧ (a i)^2 / (a j) = a m
def prop2 (a : ℕ → ℕ) (n k l: ℕ) : Prop := n ≥ 3 ∧ k > l ∧ (a n) = (a k)^2 / (a l)

-- Part I: Sequence {a_n = n} check for property 1
theorem part_1_property (a : ℕ → ℕ) (h : ∀ n, a n = n) : ¬∃ i j m, prop1 a i j m := by
  sorry

-- Part II: Sequence {a_n = 2^(n-1)} check for property 1 and 2
theorem part_2_property (a : ℕ → ℕ) (h : ∀ n, a n = 2^(n-1)) : 
  (∀ i j, ∃ m, prop1 a i j m) ∧ (∀ n k l, prop2 a n k l) := by
  sorry

-- Part III: Increasing sequence that satisfies both properties is a geometric sequence
theorem part_3_geometric (a : ℕ → ℕ) (h_inc : ∀ n m, n < m → a n < a m) 
  (h_prop1 : ∀ i j, i > j → ∃ m, prop1 a i j m)
  (h_prop2 : ∀ n, n ≥ 3 → ∃ k l, k > l ∧ (a n) = (a k)^2 / (a l)) : 
  ∃ r, ∀ n, a (n + 1) = r * a n := by
  sorry

end part_1_property_part_2_property_part_3_geometric_l339_339691


namespace compute_value_l339_339515

def Δ (p q : ℕ) : ℕ := p^3 - q

theorem compute_value : Δ (5^Δ 2 7) (4^Δ 4 8) = 125 - 4^56 := by
  sorry

end compute_value_l339_339515


namespace solution_l339_339257

noncomputable def problem (x : ℝ) : Prop :=
  2021 * (x ^ (2020/202)) - 1 = 2020 * x

theorem solution (x : ℝ) (hx : x ≥ 0) : problem x → x = 1 := 
begin
  sorry
end

end solution_l339_339257


namespace unique_sequence_l339_339531

noncomputable def seq : ℕ → ℝ
| 0     := 1
| 1     := (1 - (1 - √5) / 2)
| n + 2 := seq n - seq (n + 1)

theorem unique_sequence :
  (∀ n, seq n > 0) ∧ seq 0 = 1 ∧ (∀ n, seq n - seq (n + 1) = seq (n + 2)) ∧
  (∀ n, seq n = (1 - √5) / 2 ^ n) :=
sorry

end unique_sequence_l339_339531


namespace adult_tickets_sold_l339_339380

theorem adult_tickets_sold (A S : ℕ) (h1 : S = 3 * A) (h2 : A + S = 600) : A = 150 :=
by
  sorry

end adult_tickets_sold_l339_339380


namespace rhombus_properties_l339_339372

noncomputable def area_of_rhombus (d1 d2 : ℝ) : ℝ := (d1 * d2) / 2
noncomputable def side_length_of_rhombus (d1 d2 : ℝ) : ℝ := Real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)

theorem rhombus_properties (d1 d2 : ℝ) (h1 : d1 = 18) (h2 : d2 = 16) :
  area_of_rhombus d1 d2 = 144 ∧ side_length_of_rhombus d1 d2 = Real.sqrt 145 := by
  sorry

end rhombus_properties_l339_339372


namespace world_cup_surveys_l339_339274

theorem world_cup_surveys (total_fans : ℕ) (male_fans female_fans : ℕ) (a : ℕ) 
  (less_than_32M : ℕ) (at_least_32M : ℕ) (less_than_32F : ℕ) (at_least_32F : ℕ) :
  total_fans = 400 →
  male_fans = 200 →
  female_fans = 200 →
  less_than_32M = a + 20 →
  at_least_32M = a + 20 →
  less_than_32F = a + 40 →
  at_least_32F = a →
  less_than_32M + at_least_32M + less_than_32F + at_least_32F = total_fans →
  a = 80 ∧ let n := total_fans; let ad := 100 * 80; let bc := 100 * 120 in
  n * (ad - bc) ^ 2 / (200 * 200 * 220 * 180) > 3.841 :=
by
  intros total_fans_400 male_fans_200 female_fans_200 less32m at_least32m less32f at_least32f total_fans_eq;
  assume a_val;
  sorry

end world_cup_surveys_l339_339274


namespace factorial_div_sum_l339_339433

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339433


namespace smallest_integer_m_l339_339336

theorem smallest_integer_m (m : ℕ) : m > 1 ∧ m % 13 = 2 ∧ m % 5 = 2 ∧ m % 3 = 2 → m = 197 := 
by 
  sorry

end smallest_integer_m_l339_339336


namespace angle_bisector_problem_l339_339279

open EuclideanGeometry

theorem angle_bisector_problem (A B C D M K : Point)
(hABCD_convex : ConvexQuadrilateral A B C D)
(hM_intersect : IntersectingPoint (AC ⟶ BD) M)
(hK_bisector : AngleBisector (∠ ACD) (Ray BA) K)
(h_condition : MA * MC + MA * CD = MB * MD) :
  ∠ BKC = ∠ CDB := 
sorry

end angle_bisector_problem_l339_339279


namespace coords_P_origin_l339_339991

variable (x y : Int)
def point_P := (-5, 3)

theorem coords_P_origin : point_P = (-5, 3) := 
by 
  -- Proof to be written here
  sorry

end coords_P_origin_l339_339991


namespace simplify_expression_l339_339926

theorem simplify_expression (a b c d : ℝ) (h1 : a + b = 0) (h2 : c * d = 1) :
  -5 * a + 2017 * c * d - 5 * b = 2017 :=
by
  sorry

end simplify_expression_l339_339926


namespace seq_condition_general_formula_S_n_bounds_l339_339569

noncomputable def a (n : ℕ) : ℚ := by
  if n = 0 then exact 0
  else exact 2 / n

theorem seq_condition (n : ℕ) (hn : n > 0) :
  (∑ k in Finset.range n, (k+1)^2 * a (k+1)) = n^2 + n := sorry

theorem general_formula (n : ℕ) (hn : n > 0) : a n = 2 / n := sorry

def S (n : ℕ) : ℚ := ∑ k in Finset.range n, a k * a (k+1)

theorem S_n_bounds (n : ℕ) (hn : n > 0) : 2 ≤ S n ∧ S n < 4 := sorry

end seq_condition_general_formula_S_n_bounds_l339_339569


namespace grid_coloring_count_l339_339066

/-- Let n be a positive integer with n ≥ 2. Each of the 2n vertices in a 2 × n grid need to be 
colored red (R), yellow (Y), or blue (B). The three vertices at the endpoints are already colored 
as shown in the problem description. For the remaining 2n-3 vertices, each vertex must be colored 
exactly one color, and adjacent vertices must be colored differently. We aim to show that the 
number of distinct ways to color the vertices is 3^(n-1). -/
theorem grid_coloring_count (n : ℕ) (hn : n ≥ 2) : 
  ∃ a_n b_n c_n : ℕ, 
    (a_n + b_n + c_n = 3^(n-1)) ∧ 
    (a_n = b_n) ∧ 
    (a_n = 2 * b_n + c_n) := 
by 
  sorry

end grid_coloring_count_l339_339066


namespace race_length_l339_339646

theorem race_length (L : ℝ) 
  (h1 : L > 13)
  (h2 : ∃ B1 C1 : ℝ, B1 = L - 10 ∧ C1 = L - 13)
  (h3 : ∃ B2 C2 : ℝ, B2 = 180 ∧ C2 = 174)
  (h4 : (B2 / B1) = (C2 / C1)) 
  : L = 100 :=
by sorry

end race_length_l339_339646


namespace slope_angle_parametric_l339_339520

noncomputable def slope_angle (x y : ℝ → ℝ) : ℝ :=
  arctan ((y 1 - y 0) / (x 1 - x 0))

theorem slope_angle_parametric :
  let x t := 5 - 3 * t in
  let y t := 3 + (Real.sqrt 3) * t in
  slope_angle x y = 150 :=
by sorry

end slope_angle_parametric_l339_339520


namespace probability_all_white_balls_drawn_l339_339811

theorem probability_all_white_balls_drawn (total_balls white_balls black_balls drawn_balls : ℕ) 
  (h_total : total_balls = 15) (h_white : white_balls = 7) (h_black : black_balls = 8) (h_drawn : drawn_balls = 7) :
  (Nat.choose 7 7 : ℚ) / (Nat.choose 15 7 : ℚ) = 1 / 6435 := by
sorry

end probability_all_white_balls_drawn_l339_339811


namespace min_value_x1x2_squared_inequality_ab_l339_339102

def D : Set (ℝ × ℝ) := 
  { p | ∃ x1 x2, p = (x1, x2) ∧ x1 + x2 = 2 ∧ x1 > 0 ∧ x2 > 0 }

-- Part 1: Proving the minimum value of x1^2 + x2^2 in set D is 2
theorem min_value_x1x2_squared (x1 x2 : ℝ) (h : (x1, x2) ∈ D) : 
  x1^2 + x2^2 ≥ 2 := 
sorry

-- Part 2: Proving the inequality for any (a, b) in set D
theorem inequality_ab (a b : ℝ) (h : (a, b) ∈ D) : 
  (1 / (a + 2 * b) + 1 / (2 * a + b)) ≥ (2 / 3) := 
sorry

end min_value_x1x2_squared_inequality_ab_l339_339102


namespace no_primes_in_range_l339_339550

theorem no_primes_in_range (n : ℕ) (hn : n > 2) : 
  ∀ k, n! + 2 < k ∧ k < n! + n + 1 → ¬Prime k := 
sorry

end no_primes_in_range_l339_339550


namespace max_factors_crossed_out_without_changing_roots_l339_339787

theorem max_factors_crossed_out_without_changing_roots :
  (∃ ys : finset ℕ, finset.card ys = 1007 ∧
    ∀ (x : ℕ), (∃ i ∈ ys, sin ((i : ℝ) * π / x) = 0) ↔
    (∃ j ∈ (finset.range 2016 \ ys), sin ((j : ℝ) * π / x) = 0)) := 
sorry

end max_factors_crossed_out_without_changing_roots_l339_339787


namespace sum_of_first_10_common_elements_correct_l339_339884

noncomputable def sum_of_first_10_common_elements : ℕ :=
  let ap := {n | ∃ m : ℕ, n = 4 + 3 * m}
  let gp := {n | ∃ k : ℕ, n = 10 * 2^k}
  let common_elements := {n | n ∈ ap ∧ n ∈ gp}
  let common_elements_list := List.filter (λ n, n ∈ common_elements) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  List.sum (List.take 10 common_elements_list)

theorem sum_of_first_10_common_elements_correct : 
  sum_of_first_10_common_elements = 3495250 :=
sorry

end sum_of_first_10_common_elements_correct_l339_339884


namespace accurate_K_announcement_l339_339985

theorem accurate_K_announcement :
  ∀ (K : ℝ) (e : ℝ), K = 3.68547 → e = 0.00256 →
  (∀ v, v ∈ set.Icc (K - e) (K + e) → round (10 * v) / 10 = 3.7) :=
by
  intros K e hK he v hv
  sorry

end accurate_K_announcement_l339_339985


namespace min_distance_midpoint_to_origin_l339_339634

theorem min_distance_midpoint_to_origin (A B : ℝ × ℝ)
  (hA : A.1 + A.2 = 7)
  (hB : B.1 + B.2 = 5) :
  let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2) in
  (M.1 ^ 2 + M.2 ^ 2) = 18 :=
by 
  sorry

end min_distance_midpoint_to_origin_l339_339634


namespace part_1_a_part_1_b_part_2_l339_339959

open Set

variable (a : ℝ)

def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | x^2 - 4 > 0}
def B (a : ℝ) : Set ℝ := {x : ℝ | x ≤ a}
def compl_U_A : Set ℝ := compl A

theorem part_1_a :
  A ∩ B 1 = {x : ℝ | x < -2} :=
by
  sorry

theorem part_1_b :
  A ∪ B 1 = {x : ℝ | x > 2 ∨ x ≤ 1} :=
by
  sorry

theorem part_2 :
  compl_U_A ⊆ B a → a ≥ 2 :=
by
  sorry

end part_1_a_part_1_b_part_2_l339_339959


namespace count_integers_between_sqrt8_sqrt75_l339_339117

theorem count_integers_between_sqrt8_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ ∀ i : ℤ, (3 ≤ i ∧ i ≤ 8) ↔ (sqrt 8 < i ∧ i < sqrt 75) :=
by
  use 6
  sorry

end count_integers_between_sqrt8_sqrt75_l339_339117


namespace smallest_k_divides_l339_339042

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l339_339042


namespace problem_statement_l339_339685

theorem problem_statement :
  ∀ (ω : Fin (2020!) → ℂ)
  (hω : ∀ k, is_root (X^(2020!) - 1) (ω k))
  (n : ℕ)
  (hn : ∀ k, ∃ m, 2^m ∣ (2^(2019!) - 1) / (ω k)^(2020) + 2)
  (a b : ℕ)
  (h_ab : a > 0 ∧ b > 0 ∧ n = a! + b ∧ a is_largest_max),
  (a + b) % 1000 = 31 := sorry

end problem_statement_l339_339685


namespace min_area_triangle_l339_339083

theorem min_area_triangle 
  (a b : ℝ) (ha : 1 < a) (hb : 1 < b) 
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 - 2*x - 2*y = -1)
  (h_tangent : ∀ (x y : ℝ), b * x + a * y - 2 * a * b = 0 → 
    |b * 1 + a * 1 - 2 * a * b| / sqrt (a^2 + b^2) = 1) :
  2 * a * b = 3 + 2 * sqrt 2 :=
by
  sorry

end min_area_triangle_l339_339083


namespace evaluate_polynomial_l339_339893

theorem evaluate_polynomial (x : ℝ) : x * (x * (x * (x - 3) - 5) + 9) + 2 = x^4 - 3 * x^3 - 5 * x^2 + 9 * x + 2 := by
  sorry

end evaluate_polynomial_l339_339893


namespace find_k_value_for_unique_real_solution_l339_339542

noncomputable def cubic_has_exactly_one_real_solution (k : ℝ) : Prop :=
    ∃! x : ℝ, 4*x^3 + 9*x^2 + k*x + 4 = 0

theorem find_k_value_for_unique_real_solution :
  ∃ (k : ℝ), k > 0 ∧ cubic_has_exactly_one_real_solution k ∧ k = 6.75 :=
sorry

end find_k_value_for_unique_real_solution_l339_339542


namespace necessary_not_sufficient_condition_l339_339632

-- Define the necessary conditions for the equation to represent a hyperbola
def represents_hyperbola (k : ℝ) : Prop :=
  k > 5 ∨ k < -2

-- Define the condition for k
axiom k_in_real (k : ℝ) : Prop

-- The proof statement
theorem necessary_not_sufficient_condition (k : ℝ) (hk : k_in_real k) :
  (∃ (k_val : ℝ), k_val > 5 ∧ k = k_val) → represents_hyperbola k ∧ ¬ (represents_hyperbola k → k > 5) :=
by
  sorry

end necessary_not_sufficient_condition_l339_339632


namespace percentage_of_y_in_relation_to_25_percent_of_x_l339_339721

variable (y x : ℕ) (p : ℕ)

-- Conditions
def condition1 : Prop := (y = (p * 25 * x) / 10000)
def condition2 : Prop := (y * x = 100 * 100)
def condition3 : Prop := (y = 125)

-- The proof goal
theorem percentage_of_y_in_relation_to_25_percent_of_x :
  condition1 y x p ∧ condition2 y x ∧ condition3 y → ((y * 100) / (25 * x / 100) = 625)
:= by
-- Here we would insert the proof steps, but they are omitted as per the requirements.
sorry

end percentage_of_y_in_relation_to_25_percent_of_x_l339_339721


namespace cone_surface_area_l339_339590

theorem cone_surface_area (radius sector_radius arc_length : ℝ)
  (hr : radius = 2) (hsector_radius : sector_radius = 4) (harc_length : arc_length = 4 * real.pi) :
  let lateral_surface_area := (1/2) * sector_radius * arc_length,
      base_area := real.pi * radius ^ 2,
      total_surface_area := lateral_surface_area + base_area
  in total_surface_area = 12 * real.pi :=
by
  sorry

end cone_surface_area_l339_339590


namespace problem1_problem2_l339_339405

section Calculations

-- Problem 1
theorem problem1 : sqrt 8 / sqrt 2 + (sqrt 5 + 3) * (sqrt 5 - 3) = -2 := by
  sorry

-- Problem 2
theorem problem2 : sqrt 27 + abs (1 - sqrt 3) + (1 / 3 : ℝ)⁻¹ - (π - 3)^0 = 4 * sqrt 3 + 1 := by
  sorry

end Calculations

end problem1_problem2_l339_339405


namespace sum_of_first_10_common_elements_l339_339890

def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n
def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

theorem sum_of_first_10_common_elements :
  let common_elements := λ k : ℕ, if (k % 2 = 0) then Some (geometric_progression k) else None in
  (List.range 20).filter_map common_elements).take 10).sum = 3495250 :=
sorry

end sum_of_first_10_common_elements_l339_339890


namespace num_points_on_plane_num_points_second_quadrant_l339_339103

-- Definition of the set M
def M : Set ℤ := {-3, -2, -1, 0, 1, 2}

-- Predicate to check if a point is in the second quadrant
def is_second_quadrant (a b : ℤ) : Prop := a < 0 ∧ b > 0

-- Statement 1: P can represent 36 different points on the plane.
theorem num_points_on_plane : (∃ count : ℕ, count = 36) :=
by
  let points := (Finset.product M M).card
  have h : points = 36 := sorry
  exact ⟨36, h⟩ 

-- Statement 2: P can represent 6 points in the second quadrant.
theorem num_points_second_quadrant : (∃ count : ℕ, count = 6) :=
by
  let points := Finset.card (Finset.filter (λ (ab : ℤ × ℤ), is_second_quadrant ab.1 ab.2) (Finset.product M M))
  have h : points = 6 := sorry
  exact ⟨6, h⟩

end num_points_on_plane_num_points_second_quadrant_l339_339103


namespace hyperbola_eccentricity_proof_l339_339070

noncomputable def hyperbola_eccentricity (a b c : ℝ) (h : a > b ∧ b > 0) : ℝ :=
  let F := (c, 0)
  let area_OAB := (12 * a^2) / 7 
  sqrt (1 + (b / a)^2)

theorem hyperbola_eccentricity_proof (a b c : ℝ) (h : a > b ∧ b > 0) :
  hyperbola_eccentricity a b c h = 5 / 4 :=
sorry

end hyperbola_eccentricity_proof_l339_339070


namespace refuel_cost_is_950_l339_339368

def smaller_plane_tank_size : ℕ := 60
def larger_plane_tank_size : ℕ := smaller_plane_tank_size + smaller_plane_tank_size / 2
def special_plane_tank_size : ℕ := 200

def conventional_fuel_cost_per_liter : ℕ := 0.5
def special_fuel_cost_per_liter : ℕ := 1

def service_fee_regular : ℕ := 100
def service_fee_special : ℕ := 200

def total_fuel_cost : ℕ :=
  2 * smaller_plane_tank_size * conventional_fuel_cost_per_liter 
  + 2 * larger_plane_tank_size * conventional_fuel_cost_per_liter 
  + special_plane_t_tank_size * special_fuel_cost_per_liter

def total_service_fee : ℕ :=
  4 * service_fee_regular 
  + service_fee_special

def total_cost : ℕ := total_fuel_cost + total_service_fee

theorem refuel_cost_is_950 : total_cost = 950 := by
  sorry

end refuel_cost_is_950_l339_339368


namespace distance_ratios_equal_l339_339101

noncomputable def curveEq (x y : ℝ) : Prop :=
  x^2 = 4 * y

noncomputable def parametricLineEq (t α : ℝ) : ℝ × ℝ :=
  (2 + t * cos α, 2 + t * sin α)

noncomputable def symmetricAboutXeq2 (α : ℝ) : ℝ :=
  π - α

theorem distance_ratios_equal (P A B C D : ℝ × ℝ) (α : ℝ) :
  -- Given conditions
  let curveE := ∀ x y: ℝ, curveEq x y
  let lineL := parametricLineEq in
  let l1_intersects_E := ∀ t1, curveEq (2 + t1 * cos α) (2 + t1 * sin α) = true
  let l2_intersects_E := ∀ t2, curveEq (2 + t2 * cos (symmetricAboutXeq2 α)) (2 + t2 * sin (symmetricAboutXeq2 α)) = true
  -- To Prove:
  let PA := dist P A in
  let PB := dist P B in
  let PC := dist P C in
  let PD := dist P D in
  (PA * PB = PC * PD) → 
  |PA| / |PD| = |PC| / |PB| :=
begin
  sorry
end

end distance_ratios_equal_l339_339101


namespace relationship_between_a_and_b_l339_339935

-- Define the given linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k^2 + 1) * x + 1

-- Formalize the relationship between a and b given the points and the linear function
theorem relationship_between_a_and_b (a b k : ℝ) 
  (hP : a = linear_function k (-4))
  (hQ : b = linear_function k 2) :
  a < b := 
by
  sorry  -- Proof to be filled in by the theorem prover

end relationship_between_a_and_b_l339_339935


namespace car_average_speed_l339_339355

theorem car_average_speed (S : ℝ) 
  (h1 : ∀ (n : ℕ), n ≥ 1 → let T := (5 : ℕ) * n in average_speed (T) = S + 5 * n)
  (h2 : average_speed (5 : ℕ) = S)
  (h3 : distance_traveled (15 : ℕ) = 3) 
  : S = 26 :=
sorry

end car_average_speed_l339_339355


namespace temp_at_first_campsite_elevation_of_second_campsite_l339_339386

-- Definitions from conditions
def initial_temperature : ℝ := -2
def temp_decrease_per_km : ℝ := -6 -- 3°C per 0.5 km => 6°C per km

-- Statement for Proof Problem Part 1
theorem temp_at_first_campsite 
  (ascend_distance : ℝ) 
  (initial_temp : ℝ) 
  (temp_decrease_per_km : ℝ) :
  ascend_distance = 2.5 ∧ initial_temp = initial_temperature ∧ temp_decrease_per_km = -6 → 
  (initial_temp + (ascend_distance * temp_decrease_per_km / 0.5) / 2) = -17 :=
by
  intros,
  sorry

-- Statement for Proof Problem Part 2
theorem elevation_of_second_campsite
  (temp_at_second_campsite : ℝ)
  (initial_temp : ℝ)
  (temp_decrease_per_km : ℝ) :
  temp_at_second_campsite = -29 ∧ initial_temp = initial_temperature ∧ temp_decrease_per_km = -6 → 
  ((initial_temp - temp_at_second_campsite) / temp_decrease_per_km) * 0.5 = 4.5 :=
by
  intros,
  sorry

end temp_at_first_campsite_elevation_of_second_campsite_l339_339386


namespace log_equation_solution_l339_339072

theorem log_equation_solution (m : ℝ) (h : (log 4 / log 3) * (log 8 / log 4) * (log m / log 8) = log 16 / log 4) : m = 9 :=
by sorry

end log_equation_solution_l339_339072


namespace div_by_4_count_1_to_100_l339_339693

-- Function to compute the last two digits of n^2
def last_two_digits_square (n : ℕ) : ℕ :=
  (n * n) % 100

-- Predicate to check if an integer k satisfies the condition that b_k is divisible by 4
def divisible_by_4 (k : ℕ) : Prop :=
  k % 4 = 0 ∨ k % 4 = 2

-- Main statement to prove
theorem div_by_4_count_1_to_100 : 
  (Finset.filter (λ k, divisible_by_4 k) (Finset.range 101)).card = 50 :=
by
  sorry

end div_by_4_count_1_to_100_l339_339693


namespace roots_are_imaginary_l339_339055

theorem roots_are_imaginary (m : ℝ) :
  (∀ (x : ℝ), x^2 - 4 * m * x + (5 * m^2 + 2) = 0 → (∃ y ∈ ℂ, x = y.re)) →
  (5 * m^2 + 2 = 9) →
  (let Δ := (4 * m)^2 - 4 * (5 * m^2 + 2) in Δ < 0) :=
by
  intros h condition
  have m_sol : m = sqrt (7 / 5) ∨ m = -sqrt (7 / 5),
  { sorry },
  have Δ_val : (4 * m)^2 - 4 * (5 * m^2 + 2) = -68 / 5,
  { sorry },
  show -68 / 5 < 0, by linarith

end roots_are_imaginary_l339_339055


namespace factorial_div_sum_l339_339438

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339438


namespace bryce_received_raisins_l339_339614

theorem bryce_received_raisins
  (C B : ℕ)
  (h1 : B = C + 8)
  (h2 : C = B / 3) :
  B = 12 :=
by sorry

end bryce_received_raisins_l339_339614


namespace smallest_k_l339_339016

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l339_339016


namespace factorial_division_sum_l339_339464

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339464


namespace find_m_l339_339607

def line (m : ℝ) : ℝ × ℝ → Prop :=
  λ p : ℝ × ℝ, 2 * p.1 + m * p.2 - 8 = 0

def circle (m : ℝ) : ℝ × ℝ → Prop :=
  λ p : ℝ × ℝ, (p.1 - m) ^ 2 + p.2 ^ 2 = 4

noncomputable def is_right_triangle (A B C : ℝ × ℝ) : Prop :=
  let dAB := (A.1 - B.1)^2 + (A.2 - B.2)^2
  let dBC := (B.1 - C.1)^2 + (B.2 - C.2)^2
  let dAC := (A.1 - C.1)^2 + (A.2 - C.2)^2
  (dAB = dBC + dAC ∧ dBC = dAB + dAC) ∨ 
  (dBC = dAB + dAC ∧ dAC = dAB + dBC) ∨ 
  (dAC = dAB + dBC ∧ dAB = dAC + dBC)

theorem find_m (m : ℝ) (A B C : ℝ × ℝ)
  (h1 : ∀ p, line m p → circle m p) (h2 : is_right_triangle A B C) :
  m = 2 ∨ m = 14 :=
sorry

end find_m_l339_339607


namespace polynomial_in_x_minus_half_a_squared_l339_339757

-- Given a polynomial function p
variable (p : ℝ → ℝ)

-- Given a constant a
variable (a : ℝ)

-- Condition: For any x, p(x) = p(a - x)
axiom symm_cond : ∀ x : ℝ, p(x) = p(a - x)

-- Prove that p(x) can be expressed as a polynomial in (x - a/2)^2
theorem polynomial_in_x_minus_half_a_squared :
  ∃ h : ℝ → ℝ, ∀ x : ℝ, p(x) = h ((x - a / 2)^2) :=
sorry

end polynomial_in_x_minus_half_a_squared_l339_339757


namespace unique_solution_l339_339060

noncomputable def f (a b x : ℝ) := 2 * (a + b) * Real.exp (2 * x) + 2 * a * b
noncomputable def g (a b x : ℝ) := 4 * Real.exp (2 * x) + a + b

theorem unique_solution (a b : ℝ) (h1 : a > b) (h2 : b > 0) : 
  ∃! x, f a b x = ( (a^(1/3) + b^(1/3))/2 )^3 * g a b x :=
sorry

end unique_solution_l339_339060


namespace eccentricities_ratio_l339_339073

/-- 
Given the eccentricities of an ellipse (e₁) and a hyperbola (e₂) 
with common foci F₁ and F₂, and a common point P such that
the vectors PF₁ and PF₂ are orthogonal, the value of 
(e₁² + e₂²)/(e₁ * e₂)² is 2.
--/
theorem eccentricities_ratio (e₁ e₂ : ℝ) (F₁ F₂ P : ℝ × ℝ) :
  ∃ (c : ℝ) (a m : ℝ), 
    (∀ (PF₁ PF₂ : ℝ × ℝ),
      PF₁ ≠ 0 ∧ PF₂ ≠ 0 ∧ 
      PF₁.1 * PF₂.1 + PF₁.2 * PF₂.2 = 0 ∧
      (PF₁.1 - PF₂.1)^2 + (PF₁.2 - PF₂.2)^2 = 4c^2 ∧
      |PF₁ - F₁| + |PF₂ - F₂| = 2a ∧
      |PF₁ - F₁| - |PF₂ - F₂| = 2m
    → (1 / e₁^2 + 1 / e₂^2 = 2) → 
      (e₁^2 + e₂^2) / (e₁ * e₂)^2 = 2).

sorry

end eccentricities_ratio_l339_339073


namespace smallest_k_divides_l339_339045

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l339_339045


namespace regular_17_gon_L_plus_R_l339_339832

theorem regular_17_gon_L_plus_R : 
  (let L := 17 in let R := 360 / 17 in L + R = 649 / 17) := 
by
  sorry

end regular_17_gon_L_plus_R_l339_339832


namespace tan_double_angle_cos_sum_angle_l339_339577

open Real

theorem tan_double_angle (α : ℝ) (h1 : cos α = -4 / 5) (h2 : α ∈ Ioo (π / 2) π) :
  tan (2 * α) = -24 / 7 := sorry

theorem cos_sum_angle (α : ℝ) (h1 : cos α = -4 / 5) (h2 : α ∈ Ioo (π / 2) π) :
  cos (α + π / 3) = (-4 - 3 * sqrt 3) / 10 := sorry

end tan_double_angle_cos_sum_angle_l339_339577


namespace factorial_expression_l339_339439

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339439


namespace find_original_number_l339_339192

noncomputable def three_digit_number (d e f : ℕ) := 100 * d + 10 * e + f

/-- Given conditions and the sum S, determine the original three-digit number -/
theorem find_original_number (S : ℕ) (d e f : ℕ) (h1 : 0 ≤ d ∧ d ≤ 9)
  (h2 : 0 ≤ e ∧ e ≤ 9) (h3 : 0 ≤ f ∧ f ≤ 9) (h4 : S = 4321) :
  three_digit_number d e f = 577 :=
sorry


end find_original_number_l339_339192


namespace alpha_and_2beta_l339_339576

theorem alpha_and_2beta (α β : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 0 < β ∧ β < π / 2) 
  (h_tan_alpha : Real.tan α = 1 / 8) (h_sin_beta : Real.sin β = 1 / 3) :
  α + 2 * β = Real.arctan (15 / 56) := by
  sorry

end alpha_and_2beta_l339_339576


namespace minimum_value_abs_z_l339_339226

noncomputable def min_abs_z {z : ℂ} (h : ∥z - 5 * complex.I∥ + ∥z - 6∥ = real.sqrt 61) : ℝ :=
  real.sqrt 30 / real.sqrt 61

theorem minimum_value_abs_z (z : ℂ) (h : ∥z - 5 * complex.I∥ + ∥z - 6∥ = real.sqrt 61) :
    ∃ z : ℂ, ∀ w : ℂ, (∥w - 5 * complex.I∥ + ∥w - 6∥ = real.sqrt 61) → ∥w∥ ≥ min_abs_z h :=
sorry

end minimum_value_abs_z_l339_339226


namespace projection_of_apex_is_incenter_l339_339969

variable (A B C S O : Point)
variable (triangle_ABC : Triangle)
variable (pyramid_SABC : Pyramid S A B C)
variable (is_scalene : triangle_ABC.is_scalene)
variable (equal_dihedral_angles : 
  ∀ {face1 face2 : Plane},
  face1 ∈ lateral_faces pyramid_SABC →
  face2 = base_face pyramid_SABC →
  dihedral_angle face1 face2 = dihedral_angle (base_face pyramid_SABC) (lateral_faces pyramid_SABC).head)
variable (O_within_triangle : triangle_ABC.has_vertex O)
variable (projection_property : 
  ∀ p ∈ [A, B, C], 
  projection_under_vertex S p = ⟨x, y, z⟩ → congruent_triangle (SABC : Pyramid).to_triangle)

theorem projection_of_apex_is_incenter :
  within_triangle ABC O ->
  is_scalene ABC → 
  equal_dihedral_angles → 
  projection_property →
  triangle_ABC.is_incenter O := 
sorry

end projection_of_apex_is_incenter_l339_339969


namespace factorial_sum_division_l339_339451

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339451


namespace factorial_division_identity_l339_339508

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339508


namespace problem1_problem2_l339_339946

-- Problem 1: Proving the range of m values for the given inequality
theorem problem1 (m : ℝ) : (∀ x : ℝ, |x + 1| + |x - m| ≥ 3) ↔ (m ≤ -4 ∨ m ≥ 2) :=
sorry

-- Problem 2: Proving the range of m values given a non-empty solution set for the inequality
theorem problem2 (m : ℝ) : (∃ x : ℝ, |m + 1| - 2 * m ≥ x^2 - x) ↔ (m ≤ 5/4) :=
sorry

end problem1_problem2_l339_339946


namespace horse_drinking_water_l339_339677

-- Definitions and conditions

def initial_horses : ℕ := 3
def added_horses : ℕ := 5
def total_horses : ℕ := initial_horses + added_horses
def bathing_water_per_day : ℕ := 2
def total_water_28_days : ℕ := 1568
def days : ℕ := 28
def daily_water_total : ℕ := total_water_28_days / days

-- The statement looking to prove
theorem horse_drinking_water (D : ℕ) : 
  (total_horses * (D + bathing_water_per_day) = daily_water_total) → 
  D = 5 := 
by
  -- Add proof steps here
  sorry

end horse_drinking_water_l339_339677


namespace general_term_a_minimum_n_for_b_l339_339079

variable {a : ℕ → ℝ}
variable {b : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {T : ℕ → ℝ}

noncomputable def seq_a (n : ℕ) : ℝ :=
  if n = 1 then 1 else a n

axiom a_monotonic : ∀ n : ℕ, seq_a n ≤ seq_a (n + 1)
axiom a_1_eq : seq_a 1 = 1
axiom sum_a_n : ∀ n : ℕ, n ≥ 1 → S n = (1/2) * seq_a (n + 1) + n + 1

noncomputable def b_seq (n : ℕ) : ℝ :=
1 / (seq_a n * seq_a (n + 1))

noncomputable def T_sum (n : ℕ) : ℝ :=
(1/2) * (1 + 1/2 - 1/(n + 1) - 1/(n + 2))

theorem general_term_a (n : ℕ) : n ≥ 1 → seq_a n = -3^n + 1 :=
begin
  sorry
end

theorem minimum_n_for_b (n : ℕ) : T_sum n > (9/19) ↔ n ≥ 3 :=
begin
  sorry
end

end general_term_a_minimum_n_for_b_l339_339079


namespace find_correct_value_l339_339342

-- Definitions based on the problem's conditions
def incorrect_calculation (x : ℤ) : Prop := 7 * x = 126
def correct_value (x : ℤ) (y : ℤ) : Prop := x / 6 = y

theorem find_correct_value :
  ∃ (x y : ℤ), incorrect_calculation x ∧ correct_value x y ∧ y = 3 := by
  sorry

end find_correct_value_l339_339342


namespace find_y_intercept_of_parallel_line_l339_339080

theorem find_y_intercept_of_parallel_line (a b : ℝ) 
  (h1 : ∃ m : ℝ, m = 2 ∧ is_parallel_line l1 l2)
  (h2 : passes_through_point l2 (-1, 1)) :
  y_intercept_of_line l2 = (0, 3) :=
sorry

end find_y_intercept_of_parallel_line_l339_339080


namespace quadrilateral_trapezoid_parallelogram_l339_339723

theorem quadrilateral_trapezoid_parallelogram
  (A B C D M E F : Type*)
  [add_comm_group A] [add_comm_group B] [add_comm_group C] [add_comm_group D]
  [has_scalar ℝ A] [has_scalar ℝ B] [has_scalar ℝ C] [has_scalar ℝ D]
  (midpoint_M : M = midpoint A B)
  (midpoint_E : E = midpoint A D)
  (midpoint_F : F = midpoint B C)
  (midline_condition : dist E F = 0.5 * (dist A B + dist C D)) :
  parallel A C :=
by
  sorry

end quadrilateral_trapezoid_parallelogram_l339_339723


namespace no_non_integer_point_exists_l339_339214

variable (b0 b1 b2 b3 b4 b5 u v : ℝ)

def q (x y : ℝ) : ℝ := b0 + b1 * x + b2 * y + b3 * x^2 + b4 * x * y + b5 * y^2

theorem no_non_integer_point_exists
    (h₀ : q b0 b1 b2 b3 b4 b5 0 0 = 0)
    (h₁ : q b0 b1 b2 b3 b4 b5 1 0 = 0)
    (h₂ : q b0 b1 b2 b3 b4 b5 (-1) 0 = 0)
    (h₃ : q b0 b1 b2 b3 b4 b5 0 1 = 0)
    (h₄ : q b0 b1 b2 b3 b4 b5 0 (-1) = 0)
    (h₅ : q b0 b1 b2 b3 b4 b5 1 1 = 0) :
  ∀ u v : ℝ, (¬ ∃ (n m : ℤ), u = n ∧ v = m) → q b0 b1 b2 b3 b4 b5 u v ≠ 0 :=
by
  sorry

end no_non_integer_point_exists_l339_339214


namespace soft_contact_lens_cost_l339_339393

/-- The cost of a pair of soft contact lenses is 150 dollars given specific sales conditions. -/
theorem soft_contact_lens_cost
  (H : ℕ)
  (S : ℕ)
  (h_lens_cost : 85)
  (soft_lens_count : S)
  (hard_lens_count : H)
  (total_pairs : 11)
  (total_sales : 1455)
  (lens_cost : 85 * H + S * (H + 5) = total_sales)
  (lens_count : H + (H + 5) = total_pairs) :
  S = 150 :=
by
  sorry

end soft_contact_lens_cost_l339_339393


namespace number_of_statements_implying_target_l339_339862

variable (p q : Prop)
def statement_1 : Prop := p → q
def statement_2 : Prop := p ∨ ¬q
def statement_3 : Prop := ¬p ∧ q
def statement_4 : Prop := ¬p ∨ q
def target : Prop := p ∧ ¬q

theorem number_of_statements_implying_target : 
  ({statement_1 p q, statement_2 p q, statement_3 p q, statement_4 p q}.count (λ s, s → target p q) = 0) :=
by
  sorry

end number_of_statements_implying_target_l339_339862


namespace common_tangent_line_exists_l339_339900

/-- Given a cube and an arbitrary point P in space from which perpendiculars are dropped onto 
    the faces of the cube, resulting in six segments that are the diagonals of six distinct cubes.
    To prove: The six spheres, each tangent to all the edges of its respective cubes, have a 
    common tangent line through the point P and parallel to the main diagonal of the original cube. --/
theorem common_tangent_line_exists 
  (cube : Type*) 
  (P : cube) 
  (perpendiculars : list (cube × cube)) 
  (heq : perpendiculars.length = 6) 
  (spheres : list ℝ)
  (hspheres : spheres.length = 6) :
  ∃ (line : cube → cube), ∀ (s : ℝ), s ∈ spheres → is_tangent_line_to_sphere (line P) s :=
sorry

end common_tangent_line_exists_l339_339900


namespace value_of_5_star_3_l339_339755

def operation_star (a b : ℝ) : ℝ := a^2 + (2 * a) / b

theorem value_of_5_star_3 : operation_star 5 3 = 85 / 3 :=
by
  sorry

end value_of_5_star_3_l339_339755


namespace smallest_k_divides_l339_339026

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l339_339026


namespace steps_per_level_l339_339855

def number_of_steps_per_level (blocks_per_step total_blocks total_levels : ℕ) : ℕ :=
  (total_blocks / blocks_per_step) / total_levels

theorem steps_per_level (blocks_per_step : ℕ) (total_blocks : ℕ) (total_levels : ℕ) (h1 : blocks_per_step = 3) (h2 : total_blocks = 96) (h3 : total_levels = 4) :
  number_of_steps_per_level blocks_per_step total_blocks total_levels = 8 := 
by
  sorry

end steps_per_level_l339_339855


namespace polynomial_identity_l339_339168

theorem polynomial_identity (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := by
  sorry

end polynomial_identity_l339_339168


namespace equilateral_triangle_distances_relation_l339_339999

theorem equilateral_triangle_distances_relation
  (m n p h : ℝ)
  (H : ∀ (A B C : ℝ), equilateral_triangle A B C h)
  (D : ∀ (l : ℝ), distances_to_line l m n p) : 
  (m - n) ^ 2 + (n - p) ^ 2 + (p - m) ^ 2 = 2 * h ^ 2 := 
sorry

end equilateral_triangle_distances_relation_l339_339999


namespace factorial_division_sum_l339_339461

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339461


namespace smallest_period_is_pi_interval_of_monotonic_increase_min_and_max_values_in_interval_l339_339601

noncomputable def f (x : ℝ) := 2 * cos x * (sin x + cos x)

-- Part (I) Prove the smallest positive period of f(x) is π
theorem smallest_period_is_pi : ∃ T > 0, (∀ x, f (x + T) = f x) ∧ (∀ T' > 0, (∀ x, f (x + T') = f x) → T' ≥ T) ∧ T = π := sorry

-- Part (II) Prove the interval of monotonic increase is [kπ - 3π/8, kπ + π/8] for k ∈ Z
theorem interval_of_monotonic_increase (k : ℤ) : ∀ x, (k * π - 3 * π / 8 ≤ x) ∧ (x ≤ k * π + π / 8) ↔ (∀ y, (k * π - 3 * π / 8 ≤ y ∧ y ≤ k * π + π / 8) → f y ≤ f (y + 1)) := sorry

-- Part (III) Prove minimum and maximum values of f(x) in the interval [-π/4, π/4] are 0 and √2 + 1 respectively
theorem min_and_max_values_in_interval : ∃ x_min x_max, x_min ∈ Icc (-π / 4) (π / 4) ∧ x_max ∈ Icc (-π / 4) (π / 4) ∧ f x_min = 0 ∧ f x_max = sqrt 2 + 1 := sorry

end smallest_period_is_pi_interval_of_monotonic_increase_min_and_max_values_in_interval_l339_339601


namespace factorial_div_l339_339496

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339496


namespace hyperbola_eccentricity_range_l339_339975

theorem hyperbola_eccentricity_range 
    (A B C D : Point) 
    (M : Hyperbola) 
    (h_square : is_square A B C D) 
    (h_points_on_hyperbola : (A ∈ M) ∧ (B ∈ M) ∧ (C ∈ M) ∧ (D ∈ M)) :
    eccentricity_range M = (sqrt 2, +∞) := 
sorry

end hyperbola_eccentricity_range_l339_339975


namespace students_apply_to_universities_l339_339980

theorem students_apply_to_universities
  (n : ℕ) -- the number of students
  (U : Fin 5 → Set ℕ) -- universities, each represented as a set of students applying
  (hU : ∀ i, (U i).card ≥ n / 2) -- at least half of the students apply to each university
  : ∃ (i j : Fin 5), i ≠ j ∧ (U i ∩ U j).card ≥ n / 5 :=
sorry

end students_apply_to_universities_l339_339980


namespace integral_result_l339_339215

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (Real.cos (2 * x) - (a + 2) * Real.cos x + (a + 1)) / Real.sin x

def a_condition (a : ℝ) : Prop :=
  (tendsto (fun x => f a x / x) (𝓝 0) (𝓝 (1 / 2)))

theorem integral_result
  (a : ℝ) (h_a : a_condition a) :
  a = 2 →
  ∫ x in (Real.pi / 3) .. (Real.pi / 2), (1 / f a x) = 1 / 2 :=
sorry

end integral_result_l339_339215


namespace ellipses_same_eccentricity_l339_339940

variables {a b k : ℝ}
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : k > 0)

theorem ellipses_same_eccentricity (h1 : a > 0) (h2 : b > 0) (h3 : k > 0)
  (ellipse1 : ∀ x y, x^2 / a^2 + y^2 / b^2 = 1)
  (ellipse2 : ∀ x y, x^2 / a^2 + y^2 / b^2 = k) :
  (sqrt (a^2 - b^2) / a) = (sqrt (a^2 - b^2) / a) :=
by {
  sorry
}

end ellipses_same_eccentricity_l339_339940


namespace sqrt_product_identity_l339_339399

noncomputable def sqrt_product_simplified (q : ℝ) : ℝ :=
  real.sqrt(40 * q) * real.sqrt(20 * q) * real.sqrt(10 * q)

theorem sqrt_product_identity (q : ℝ) : sqrt_product_simplified q = 40 * q * real.sqrt(5 * q) :=
by
  sorry

end sqrt_product_identity_l339_339399


namespace factorial_division_sum_l339_339459

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339459


namespace numberOfLineups_is_correct_l339_339363

noncomputable def numberOfLineups (players : Finset ℕ) : ℕ :=
  let remainingPlayers := players \ {0, 1, 2, 3, 4, 5}
  let noTriplets := Finset.card (Finset.powersetLen 7 remainingPlayers)
  let oneFromEachTriplet := 3 * 3 * Finset.card (Finset.powersetLen 5 remainingPlayers)
  let twoFromFirstTriplet := Finset.card ({0, 1, 2}.powersetLen 2) * Finset.card (Finset.powersetLen 5 remainingPlayers)
  let twoFromSecondTriplet := Finset.card ({3, 4, 5}.powersetLen 2) * Finset.card (Finset.powersetLen 5 remainingPlayers)
  let oneFromFirstTwoFromSecond := 3 * 3 * Finset.card (Finset.powersetLen 4 remainingPlayers)
  let oneFromSecondTwoFromFirst := 3 * 3 * Finset.card (Finset.powersetLen 4 remainingPlayers)
  noTriplets + oneFromEachTriplet + twoFromFirstTriplet + twoFromSecondTriplet + oneFromFirstTwoFromSecond + oneFromSecondTwoFromFirst

theorem numberOfLineups_is_correct :
  numberOfLineups (Finset.range 18) = 21582 := by
  sorry

end numberOfLineups_is_correct_l339_339363


namespace min_num_occurrences_of_5_l339_339592

theorem min_num_occurrences_of_5 
  (l : List ℕ) 
  (hl_len : l.length = 18)
  (median_eq_5 : (l.nth 8 = some 5) ∧ (l.nth 9 = some 5))
  (percentile_75_eq_5 : l.nth 13 = some 5) 
  : count l 5 ≥ 6 :=
sorry

end min_num_occurrences_of_5_l339_339592


namespace factorial_computation_l339_339474

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339474


namespace factorial_sum_division_l339_339449

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339449


namespace evaluate_expression_l339_339379

theorem evaluate_expression : (5^2 - 4^2)^3 = 729 :=
by
  sorry

end evaluate_expression_l339_339379


namespace coordinates_of_P_l339_339993

/-- In the Cartesian coordinate system, given a point P with coordinates (-5, 3),
    prove that its coordinates with respect to the origin are (-5, 3). -/
theorem coordinates_of_P :
  ∀ (P : ℝ × ℝ), P = (-5, 3) → P = (-5, 3) :=
by
  intro P h,
  exact h

end coordinates_of_P_l339_339993


namespace number_of_integers_between_sqrt8_and_sqrt75_l339_339127

theorem number_of_integers_between_sqrt8_and_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ 
  ∃ x : Fin n.succ, √8 < ↑x ∧ ↑x < √75 := 
by
  sorry

end number_of_integers_between_sqrt8_and_sqrt75_l339_339127


namespace find_x_l339_339629

open Real

theorem find_x (x : ℝ) (h : (x / 6) / 3 = 6 / (x / 3)) : x = 18 ∨ x = -18 :=
by
  sorry

end find_x_l339_339629


namespace arccos_ineq_solution_l339_339001

noncomputable def arccos_ineq (x : ℝ) : Prop := 
  ∃ y : ℝ, y = arccos x ∧ 2 * (arcsin x) = arccos x

theorem arccos_ineq_solution (x : ℝ) (h : -1 ≤ x ∧ x ≤ 1) (h_decreasing : ∀ x y : ℝ, x < y → arccos x > arccos y) 
  (h_increasing : ∀ x y : ℝ, x < y → arcsin x < arcsin y) : 
  (arccos x < 2 * arcsin x ↔ x ∈ set.Icc 0 (1 : ℝ) ∧ x ∈ set.Ioc (1 / sqrt 2) 1) :=
sorry

end arccos_ineq_solution_l339_339001


namespace zero_points_a_eq_1_max_value_on_interval_max_value_T_a_l339_339606

-- Definition of given function
def f (x a : ℝ) : ℝ := -x * abs (x - 2 * a) + 1

-- Part (Ⅰ): Zeros of the function for a = 1
theorem zero_points_a_eq_1 : 
  ∀ x : ℝ, f x 1 = 0 ↔ x = 1 ∨ x = 1 + Real.sqrt 2 := 
by
  sorry

-- Part (Ⅱ): Maximum value of the function on [1,2] for a in (0, 3/2)
theorem max_value_on_interval :
  ∀ (a : ℝ), 0 < a ∧ a < 3 / 2 → 
  ∀ x ∈ set.Icc (1 : ℝ) 2,
  (f x a ≤ 
    if 0 < a ∧ a ≤ 1 / 2 then
      2 * a
    else if 1 / 2 < a ∧ a < 1 then
      1
    else
      5 - 4 * a) :=
by
  sorry

-- Part (Ⅲ): Expression for T(a)
theorem max_value_T_a :
  ∀ (a : ℝ), a > 0 → 
  (∃ T_a : ℝ, 0 ≤ T_a ∧ 
    (∀ x ∈ set.Icc (0 : ℝ) T_a, abs (f x a) ≤ 1) ↔
      (if a ≥ Real.sqrt 2 then
         T_a = a - Real.sqrt (a ^ 2 - 2)
       else
         T_a = a + Real.sqrt (a ^ 2 + 2))) :=
by
  sorry

end zero_points_a_eq_1_max_value_on_interval_max_value_T_a_l339_339606


namespace number_of_clients_l339_339833

theorem number_of_clients (C : ℕ)
  (h1 : ∀ n, ∃ k, n = 10 ∧ k = 3)
  (h2 : ∀ client k, k = 2)
  (h3 : ∀ n m, n * m = 30) :
  C = 15 :=
by
  -- The proof follows from the given conditions and solving the equation 2C = 30
  sorry

end number_of_clients_l339_339833


namespace collinear_m_n_q_l339_339058

theorem collinear_m_n_q
  (A B C P M N Q : Type)
  [Point A] [Point B] [Point C] [Point P]
  [Point M] [Point N] [Point Q]
  [Circumcircle A B C P]
  (h₁ : Parallel P BC M)
  (h₂ : Parallel P CA N)
  (h₃ : Parallel P AB Q) :
  Collinear M N Q :=
sorry

end collinear_m_n_q_l339_339058


namespace inequality_proof_l339_339075

theorem inequality_proof {n : ℕ} 
  (h1 : 0 < n)
  (x : Fin n → ℝ) 
  (y : Fin n → ℝ) 
  (z : Fin (2 * n) → ℝ)
  (h2 : ∀ i j : Fin n, (z ⟨i + j + 1, sorry⟩)^2 ≥ x i * y j) :
  let M := Finset.max' (Finset.image z (Finset.range n).succ)
  in ( (M + ∑ i : Fin (2 * n), if 1 ≤ i.val then z i else 0) / (2 * n))^2 ≥
     (∑ i : Fin n, x i) / n * (∑ i : Fin n, y i) / n :=
by admit

end inequality_proof_l339_339075


namespace no_fourth_quadrant_minimum_area_l339_339947

noncomputable def line_intercept (k : ℝ) : ℝ × ℝ :=
  let x_intercept := -(k + 2) / k
  let y_intercept := 2 + k
  (x_intercept, y_intercept)

-- Prove the range of k for no fourth quadrant intersection
theorem no_fourth_quadrant (k : ℝ) : 
  (∀ x y : ℝ, y ≠ 0 ∧ x > 0 → k * x - y + 2 + k ≠ 0) ↔ (0 ≤ k) :=
sorry

-- Prove the minimum area of triangle AOB
theorem minimum_area (k : ℝ) :
  (∀ x y : ℝ, k > 0 → 
    let (x_intercept, y_intercept) := line_intercept k
    let S := 1 / 2 * abs x_intercept * abs y_intercept
    k = 2 → S = 4 ∧ (∀ k', k' ≠ 2 → let (_, _) := line_intercept k' in S ≥ 4)) :=
sorry

end no_fourth_quadrant_minimum_area_l339_339947


namespace equal_divide_remaining_amount_all_girls_l339_339272

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end equal_divide_remaining_amount_all_girls_l339_339272


namespace card_sequence_probability_l339_339313

-- Conditions about the deck and card suits
def standard_deck : ℕ := 52
def diamond_count : ℕ := 13
def spade_count : ℕ := 13
def heart_count : ℕ := 13

-- Definition of the problem statement
def diamond_first_prob : ℚ := diamond_count / standard_deck
def spade_second_prob : ℚ := spade_count / (standard_deck - 1)
def heart_third_prob : ℚ := heart_count / (standard_deck - 2)

-- Theorem statement for the required probability
theorem card_sequence_probability : 
    diamond_first_prob * spade_second_prob * heart_third_prob = 13 / 780 :=
by
  sorry

end card_sequence_probability_l339_339313


namespace monotonic_decreasing_interval_f_l339_339753

-- Defining the quadratic function and the logarithmic composite function.
noncomputable def t (x : ℝ) : ℝ := -x^2 + 2*x + 3
noncomputable def f (x : ℝ) : ℝ := Real.logBase (1/2) (t x)

-- Defining the property of the monotonic decreasing interval of f.
theorem monotonic_decreasing_interval_f :
  ∀ x, -1 < x ∧ x ≤ 1 → f x > f (x + ε) ∀ ε > 0 :=
sorry

end monotonic_decreasing_interval_f_l339_339753


namespace solve_log_inequality_l339_339910

-- Definition for the problem conditions
def log_inequality (x : ℝ) : Prop :=
  log x (2 * x^2 + x - 1) > log x 2 - 1

-- Theorem stating the range of x based on the given condition
theorem solve_log_inequality (x : ℝ) (h : log_inequality x) : x > 1 / 2 ∧ x ≠ 1 :=
sorry

end solve_log_inequality_l339_339910


namespace oranges_initially_in_box_l339_339276

variable (O : ℕ) -- initial number of oranges
variable (total_fruit : ℕ := 14 + O) -- total fruit initially
variable (rem_fruit : ℕ := 14 + (O - 6)) -- total fruit after removing 6 oranges

theorem oranges_initially_in_box :
  (0.7 * rem_fruit = 14) → (total_fruit = 14 + O) → O = 12 :=
by
  intros h1 h2
  sorry

end oranges_initially_in_box_l339_339276


namespace james_off_road_vehicles_l339_339209

section 
variable (number_of_dirt_bikes : ℕ)
variable (price_per_dirt_bike : ℕ)
variable (number_of_off_road_vehicles : ℕ)
variable (price_per_off_road_vehicle : ℕ)
variable (registration_fee : ℕ)
variable (total_cost : ℕ)

theorem james_off_road_vehicles :
  number_of_dirt_bikes = 3 →
  price_per_dirt_bike = 150 →
  price_per_off_road_vehicle = 300 →
  registration_fee = 25 →
  total_cost = 1825 →
  number_of_off_road_vehicles = 4 :=
by
  intros h1 h2 h3 h4 h5
  have h_dirt_bikes_cost : 3 * 150 = 450 := by norm_num
  have h_total : 450 + number_of_off_road_vehicles * (300 + 25) = total_cost := sorry
  rw [h1, h2, h3, h4, h5] at h_total
  norm_num at h_total
  obtain rfl : number_of_off_road_vehicles = 4 := sorry
  assumption

end

end james_off_road_vehicles_l339_339209


namespace common_ratio_of_geometric_sequence_is_1_or_4_l339_339586

variable (a1 d : ℝ)
variable (b : ℕ → ℝ)

-- Given conditions
def a_seq (n : ℕ) : ℝ := a1 + (n - 1) * d
def a_3 : ℝ := a_seq a1 d 3
def a_11 : ℝ := a_seq a1 d 11

axiom h1 : a_3^2 = a1 * a_11
axiom h2 : a_3 = b 3
axiom h3 : a1 = b 4
axiom h4 : a_11 = b 5

-- Theorem to prove
theorem common_ratio_of_geometric_sequence_is_1_or_4 (q : ℝ) :
  (q = 1 ∨ q = 4) := sorry

end common_ratio_of_geometric_sequence_is_1_or_4_l339_339586


namespace shape_is_cone_l339_339195

-- Definitions based on conditions
structure SphericalCoordinates :=
  (rho theta phi : ℝ)

constant c : ℝ

-- Function describing the shape
def shape_described (P : SphericalCoordinates) : Prop :=
  P.phi = c

-- The theorem to prove the correct shape
theorem shape_is_cone (P : SphericalCoordinates) : shape_described P → P.shape = "Cone" :=
sorry

end shape_is_cone_l339_339195


namespace megan_folders_l339_339238

theorem megan_folders (initial_files deleted_files files_per_folder : ℕ) (h1 : initial_files = 237)
    (h2 : deleted_files = 53) (h3 : files_per_folder = 12) :
    let remaining_files := initial_files - deleted_files
    let total_folders := (remaining_files / files_per_folder) + 1
    total_folders = 16 := 
by
  sorry

end megan_folders_l339_339238


namespace vertices_locus_is_circle_l339_339956

open_locale classical

-- Define the points O and G in the plane
variables (O G : affine ℝ)

-- Assume there exists a circle with center O and radius R
noncomputable def locus_vertices 
  (O G : affine ℝ) 
  (R : ℝ) 
  (hR : 0 < R) : Prop :=
  ∀ (P : affine ℝ), 
    (∃ (triangle : affine ℝ × affine ℝ × affine ℝ),
    let O_circumcenter := (triangle.fst, O) ∧ 
    let G_centroid := (triangle.snd, G) ∧
    dist O P = R 😈) →
    P ∈ metric.sphere O R 

-- The theorem stating the result
theorem vertices_locus_is_circle
  (O G : affine ℝ) 
  (R : ℝ) 
  (hR : 0 < R) : locus_vertices O G R hR :=
sorry

end vertices_locus_is_circle_l339_339956


namespace max_a1_l339_339931

theorem max_a1 (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, n > 0 → a n > 0)
  (h_eq : ∀ n : ℕ, n > 0 → 2 + a n * (a (n + 1) - a (n - 1)) = 0 ∨ 2 - a n * (a (n + 1) - a (n - 1)) = 0)
  (h_a20 : a 20 = a 20) :
  ∃ max_a1 : ℝ, max_a1 = 512 := 
sorry

end max_a1_l339_339931


namespace equal_payment_l339_339237

-- Define the costs per pound and the quantities
def cost_chicken_per_pound := 4.50
def cost_beef_per_pound := 4.00
def cost_oil_per_liter := 1.00
def cost_fish_per_pound := 3.00
def cost_vegetables_per_pound := 2.00

def quantity_chicken := 2.0
def quantity_beef := 3.0
def quantity_oil := 1.0
def quantity_fish := 5.0
def quantity_vegetables := 1.5

-- Define the discounts
def discount_beef := 0.10
def discount_loyalty_card := 0.05

-- Define the total calculated payments
noncomputable def total_before_discounts := 
  (quantity_chicken * cost_chicken_per_pound) + 
  (quantity_beef * cost_beef_per_pound) + 
  (quantity_oil * cost_oil_per_liter) + 
  (quantity_fish * cost_fish_per_pound) + 
  (quantity_vegetables * cost_vegetables_per_pound)

noncomputable def total_after_beef_discount := 
  total_before_discounts - (quantity_beef * cost_beef_per_pound * discount_beef)

noncomputable def final_total_cost := 
  total_after_beef_discount - (total_after_beef_discount * discount_loyalty_card)

noncomputable def amount_per_person := final_total_cost / 3

-- The goal: Prove that each person should pay approximately $12.60
theorem equal_payment : amount_per_person ≈ 12.60 := sorry

end equal_payment_l339_339237


namespace number_of_integers_between_sqrt8_and_sqrt75_l339_339125

theorem number_of_integers_between_sqrt8_and_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ 
  ∃ x : Fin n.succ, √8 < ↑x ∧ ↑x < √75 := 
by
  sorry

end number_of_integers_between_sqrt8_and_sqrt75_l339_339125


namespace correct_statement_about_quadratic_function_l339_339556

theorem correct_statement_about_quadratic_function :
  let y := λ x : ℝ, -2 * (x + 3) ^ 2
  ∃! s : String, s = "The axis of symmetry is the line x = -3" ∧
                   ((s = "Opens upwards" → false) ∧
                    (s = "The axis of symmetry is the line x = -3" → true) ∧
                    (s = "When x > -4, y decreases as x increases" → false) ∧
                    (s = "The coordinates of the vertex are (-2, -3)" → false)) := 
sorry

end correct_statement_about_quadratic_function_l339_339556


namespace minimum_value_of_expression_l339_339220

theorem minimum_value_of_expression (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) :
  a^2 + b^2 + c^2 + (3 / (a + b + c)^2) ≥ 2 :=
sorry

end minimum_value_of_expression_l339_339220


namespace problem1_problem2_l339_339105

-- Define the sets A and B
def setA (a : ℝ) : Set ℝ := { x | a - 1 < x ∧ x < 2 * a + 1 }
def setB : Set ℝ := { x | 0 < x ∧ x < 1 }

-- Prove that for a = 1/2, A ∩ B = { x | 0 < x ∧ x < 1 }
theorem problem1 : setA (1/2) ∩ setB = { x : ℝ | 0 < x ∧ x < 1 } :=
by
  sorry

-- Prove that if A ∩ B = ∅, then a ≤ -1/2 or a ≥ 2
theorem problem2 (a : ℝ) (h : setA a ∩ setB = ∅) : a ≤ -1/2 ∨ a ≥ 2 :=
by
  sorry

end problem1_problem2_l339_339105


namespace factorial_division_identity_l339_339500

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339500


namespace pick_three_cards_in_order_l339_339376

theorem pick_three_cards_in_order (deck_size : ℕ) (first_card_ways : ℕ) (second_card_ways : ℕ) (third_card_ways : ℕ) 
  (total_combinations : ℕ) (h1 : deck_size = 52) (h2 : first_card_ways = 52) 
  (h3 : second_card_ways = 51) (h4 : third_card_ways = 50) (h5 : total_combinations = first_card_ways * second_card_ways * third_card_ways) : 
  total_combinations = 132600 := 
by 
  sorry

end pick_three_cards_in_order_l339_339376


namespace triangle_angle_A_l339_339672

theorem triangle_angle_A (a c : ℝ) (C A : ℝ) 
  (h1 : a = 4 * Real.sqrt 3)
  (h2 : c = 12)
  (h3 : C = Real.pi / 3)
  (h4 : a < c) :
  A = Real.pi / 6 :=
sorry

end triangle_angle_A_l339_339672


namespace find_x_y_l339_339997

theorem find_x_y (x y : ℝ) : 
    (3 * x + 2 * y + 5 * x + 7 * x = 360) →
    (x = y) →
    (x = 360 / 17) ∧ (y = 360 / 17) := by
  intros h₁ h₂
  sorry

end find_x_y_l339_339997


namespace f_at_2_l339_339564

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := a * x^5 + b * x^3 + c * x - 8

-- Given conditions
variable {a b c : ℝ}
variable h1 : f (-2) a b c = 10

-- The theorem to be proved
theorem f_at_2 (a b c : ℝ) (h1 : f (-2) a b c = 10) : f 2 a b c = -26 := by
  sorry

end f_at_2_l339_339564


namespace iesha_school_books_l339_339150

theorem iesha_school_books (total_books sports_books : ℕ)
  (h1 : total_books = 58) (h2 : sports_books = 39) :
  total_books - sports_books = 19 :=
by
  rw [h1, h2]
  exact Nat.sub_self 39

end iesha_school_books_l339_339150


namespace problem_solution_l339_339170

-- Define the main theorem
theorem problem_solution (x : ℝ) : (x + 2) ^ 2 + 2 * (x + 2) * (4 - x) + (4 - x) ^ 2 = 36 := 
by
  sorry

end problem_solution_l339_339170


namespace fraction_solution_l339_339877

theorem fraction_solution (N : ℝ) (h : N = 12.0) : (0.6667 * N + 1) = (3/4) * N := by 
  sorry

end fraction_solution_l339_339877


namespace factorial_div_l339_339494

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339494


namespace inverse_function_l339_339786

def f (x : ℝ) : ℝ := 7 - 3 * x

def g (x : ℝ) : ℝ := (7 - x) / 3

theorem inverse_function :
  ∀ x : ℝ, f (g x) = x :=
by
  sorry

end inverse_function_l339_339786


namespace complement_of_not_connected_is_connected_l339_339227

variable {V : Type*} [Fintype V] [DecidableEq V]

open Classical

def complement (G : SimpleGraph V) : SimpleGraph V :=
{ Adj := λ v w, v ≠ w ∧ ¬G.Adj v w,
  symm := λ v w ⟨hne, na⟩ => ⟨hne.symm, na⟩,
  loopless := λ v ⟨hne, _⟩ => (hne rfl).elim }

theorem complement_of_not_connected_is_connected (G : SimpleGraph V) (hG : ¬Connected G) :
  Connected (complement G) := 
sorry

end complement_of_not_connected_is_connected_l339_339227


namespace machines_bottle_production_l339_339800

theorem machines_bottle_production :
  (∀ (rate_per_machine : ℕ), 
    (6 * rate_per_machine = 330) → 
    (∃ (bottles_produced : ℕ), (10 * rate_per_machine * 4 = bottles_produced) ∧ bottles_produced = 2200)) :=
begin
  sorry
end

end machines_bottle_production_l339_339800


namespace find_x_l339_339578

-- Define the vectors and the dot product condition
def a : ℝ × ℝ × ℝ := (-3, 2, 5)
def b (x : ℝ) : ℝ × ℝ × ℝ := (1, x, -1)

def dot_product (v1 v2 : ℝ × ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3

-- Statement of the problem in Lean 4
theorem find_x (h : dot_product a (b x) = 2) : x = 5 := sorry

end find_x_l339_339578


namespace product_of_squared_roots_plus_one_l339_339151

noncomputable def poly : Polynomial ℂ := Polynomial.C 5 + Polynomial.X + 20 * Polynomial.X^2 + Polynomial.X^3

theorem product_of_squared_roots_plus_one :
  let roots := Complex.roots poly in
  (∀ a ∈ roots, ∃! p : ℂ, poly.eval p = 0 ∧ p = a) →
  (∀ b ∈ roots, ∃! q : ℂ, poly.eval q = 0 ∧ q = b) →
  (∀ c ∈ roots, ∃! r : ℂ, poly.eval r = 0 ∧ r = c) →
  (a^2 + 1) * (b^2 + 1) * (c^2 + 1) = 229
:= sorry

end product_of_squared_roots_plus_one_l339_339151


namespace ellipse_standard_eq_and_area_l339_339076

theorem ellipse_standard_eq_and_area (
  (center : ℝ × ℝ) (focus_left focus_right : ℝ × ℝ) 
  (focal_length major_minor_ratio : ℝ)
  (l : ℝ → ℝ)
  (intersects_ellipse : ℝ × ℝ → ℝ) 
  (area_of_triangle : ℝ × ℝ → ℝ)
  (P Q : ℝ × ℝ)
) :
  center = (0, 0) ∧ focus_left = (-√3, 0) ∧ focus_right = (√3, 0) ∧
  focal_length = 2 * √3 ∧ major_minor_ratio = 2 ∧ 
  (∀ y, l y = √3 / 3 * y + 1) ∧
  (∃ P Q, intersects_ellipse P = 0 ∧ intersects_ellipse Q = 0) →
  intersects_ellipse (0, 1) = 0 ∧ 
  intersects_ellipse (-√3, 0) = 0 ∧
  area_of_triangle P Q = 8 * √3 / 7 :=
by
  sorry

end ellipse_standard_eq_and_area_l339_339076


namespace problem_statement_l339_339932

noncomputable def f (x : ℝ) : ℝ := if x ∈ Set.Ico 0 1 then Real.log10 (x + 1) else 0

axiom f_periodic_odd : ∀ x : ℝ, f (x + 2) = f x ∧ f (-x) = -f x

theorem problem_statement :
  f (2016 / 5) + Real.log10 18 = 1 :=
by
  -- Definitions and conditions translated directly.
  let f' := f
  have periodic := f_periodic_odd
  sorry

end problem_statement_l339_339932


namespace pizza_problem_l339_339357

theorem pizza_problem (diameter : ℝ) (sectors : ℕ) (h1 : diameter = 18) (h2 : sectors = 4) : 
  let R := diameter / 2 
  let θ := (2 * Real.pi / sectors : ℝ)
  let m := 2 * R * Real.sin (θ / 2) 
  (m^2 = 162) := by
  sorry

end pizza_problem_l339_339357


namespace measure_angle_D_is_540_over_7_l339_339322

-- Let y be the measure of angle E in degrees
def measure_angle_E (y : ℝ) : Prop :=
  ∀ (Δ : triangle),
  Δ.is_isosceles ∧
  Δ.D ≡ Δ.F ∧
  Δ.F = 3 * Δ.E ∧
  (Δ.E + Δ.F + Δ.D = 180) →
  Δ.D = 3 * (180 / 7)

-- Theorem stating the measure of angle D in the isosceles triangle DEF
theorem measure_angle_D_is_540_over_7 {y : ℝ} :
  measure_angle_E y :=
by
  sorry

end measure_angle_D_is_540_over_7_l339_339322


namespace max_value_y_l339_339152

open Real

theorem max_value_y (x : ℝ) (h : -1 < x ∧ x < 1) : 
  ∃ y_max, y_max = 0 ∧ ∀ y, y = x / (x - 1) + x → y ≤ y_max :=
by
  have y : ℝ := x / (x - 1) + x
  use 0
  sorry

end max_value_y_l339_339152


namespace count_integers_between_sqrt8_and_sqrt75_l339_339121

theorem count_integers_between_sqrt8_and_sqrt75 : 
  let n1 := Real.sqrt 8,
      n2 := Real.sqrt 75 in 
  ∃ count : ℕ, count = (Int.floor n2 - Int.ceil n1) + 1 ∧ count = 6 :=
by
  let n1 := Real.sqrt 8
  let n2 := Real.sqrt 75
  use (Int.floor n2 - Int.ceil n1 + 1)
  simp only [Real.sqrt]
  sorry

end count_integers_between_sqrt8_and_sqrt75_l339_339121


namespace AugustHasFiveFridays_l339_339266

theorem AugustHasFiveFridays (N : Nat) :
  (has_five_tuesdays_july : hasFiveTuesdays (July N)) ∧
  (both_have_31_days : bothHave31Days (July N) (August N)) →
  ∃ (d : Day), d = friday ∧ mustOccurFiveTimes (August N) d :=
by
  sorry

end AugustHasFiveFridays_l339_339266


namespace factorial_div_sum_l339_339437

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339437


namespace generalFormulaAndSum_l339_339571

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Conditions
def isArithmeticSequence : Prop :=
  ∃ (d ≠ 0), ∀ n, a n = a 0 + d * n

def condition1 : Prop :=
  a 3 = 5

def geometricCondition : Prop :=
  let a1 := a 0; let a7 := a 0 + 6 * d; let a5 := a 0 + 4 * d;
  (a 7) ^ 2 = a 1 * a 5

-- Proving general formula and sum
theorem generalFormulaAndSum :
  isArithmeticSequence a d ∧ condition1 a d ∧ geometricCondition a d →
  (∀ n, a n = -2 * n + 11) ∧ (∀ n, let sumOddIndices := ∑ i in (Finset.range n).filter (λ k, k % 2 ≠ 0), a k, 
   sumOddIndices = -2 * n ^ 2 + 11 * n) :=
by sorry

end generalFormulaAndSum_l339_339571


namespace vasya_can_construct_polyhedron_l339_339330

-- Definition of a polyhedron using given set of shapes
-- where the original set of shapes can form a polyhedron
def original_set_can_form_polyhedron (squares triangles : ℕ) : Prop :=
  squares = 1 ∧ triangles = 4

-- Transformation condition: replacing 2 triangles with 2 squares
def replacement_condition (initial_squares initial_triangles replaced_squares replaced_triangles : ℕ) : Prop :=
  initial_squares + 2 = replaced_squares ∧ initial_triangles - 2 = replaced_triangles

-- Proving that new set of shapes can form a polyhedron
theorem vasya_can_construct_polyhedron :
  ∃ (new_squares new_triangles : ℕ),
    (original_set_can_form_polyhedron 1 4)
    ∧ (replacement_condition 1 4 new_squares new_triangles)
    ∧ (new_squares = 3 ∧ new_triangles = 2) :=
by
  sorry

end vasya_can_construct_polyhedron_l339_339330


namespace probability_sequence_correct_l339_339318

noncomputable def probability_of_sequence : ℚ :=
  (13 / 52) * (13 / 51) * (13 / 50)

theorem probability_sequence_correct :
  probability_of_sequence = 2197 / 132600 :=
by
  sorry

end probability_sequence_correct_l339_339318


namespace pairs_satisfying_divisibility_l339_339878

theorem pairs_satisfying_divisibility (a n : ℕ) (h_a_pos : 0 < a) (h_n_pos : 0 < n) :
  n ∣ (a + 1)^n - a^n ↔ n = 1 :=
begin
  sorry
end

end pairs_satisfying_divisibility_l339_339878


namespace sequence_an_is_one_l339_339918

theorem sequence_an_is_one (a : ℕ → ℕ) (h : ∀ n, a 1 + 3 * a 2 + ∑ i in finset.range (n - 1), ((2 * (i + 2) - 1) * a (i + 2)) = n ^ 2) :
  ∀ n, a n = 1 := sorry

end sequence_an_is_one_l339_339918


namespace smallest_positive_period_function_l339_339390

def f_A (x : ℝ) : ℝ := sin (2 * x - (π / 3))
def f_B (x : ℝ) : ℝ := sin (2 * x - (π / 6))
def f_C (x : ℝ) : ℝ := sin (2 * x + (π / 6))
def f_D (x : ℝ) : ℝ := sin ((x / 2) + (π / 6))

theorem smallest_positive_period_function :
  (∃ f : ℝ → ℝ,
  ∀ x, f ∈ {f_A, f_B, f_C, f_D} ∧
  (∀ x, f(x + π) = f(x)) ∧
  (∀ x, f(π / 3 - x) = f(π / 3 + x))) →
  f = f_B :=
sorry

end smallest_positive_period_function_l339_339390


namespace latest_time_temperature_80_l339_339186

theorem latest_time_temperature_80 (t : ℝ) : (-t^2 + 10 * t + 60 = 80) -> t ≤ 5 + real.sqrt 5 :=
by sorry

end latest_time_temperature_80_l339_339186


namespace factorial_division_l339_339482

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339482


namespace grade_C_percentage_is_correct_l339_339764
open List

-- Define the grading scale and student scores as constants
def grading_scale := [(95, 100), (90, 94), (85, 89), (80, 84), (77, 79), (73, 76), (70, 72), (0, 69)]
def student_scores : List ℕ := [98, 75, 86, 77, 60, 94, 72, 79, 69, 82, 70, 93, 74, 87, 78, 84, 95, 73]

-- Define the criteria for grades
def is_grade_C (score : ℕ) : Prop := 73 ≤ score ∧ score ≤ 76

-- Count the number of students with grade C
def count_grade_C_students : ℕ := (student_scores.filter is_grade_C).length

-- Define the total number of students
def total_students : ℕ := student_scores.length

-- Calculate the percentage of students with grade C
def percentage_grade_C_students : Float := (count_grade_C_students.toFloat / total_students.toFloat) * 100

-- Prove that the percentage is 16.67%
theorem grade_C_percentage_is_correct : percentage_grade_C_students = 16.67 := by
  sorry

end grade_C_percentage_is_correct_l339_339764


namespace f_has_exactly_one_zero_point_a_range_condition_l339_339087

noncomputable def f (x : ℝ) : ℝ := - (1 / 2) * Real.log x + 2 / (x + 1)

theorem f_has_exactly_one_zero_point :
  ∃! x : ℝ, 1 < x ∧ x < Real.exp 2 ∧ f x = 0 := sorry

theorem a_range_condition (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (1 / Real.exp 1) 1 → ∀ t : ℝ, t ∈ Set.Icc (1 / 2) 2 → f x ≥ t^3 - t^2 - 2 * a * t + 2) → a ≥ 5 / 4 := sorry

end f_has_exactly_one_zero_point_a_range_condition_l339_339087


namespace integer_count_between_sqrt8_and_sqrt75_l339_339143

theorem integer_count_between_sqrt8_and_sqrt75 :
  let least_integer_greater_than_sqrt8 := 3
      greatest_integer_less_than_sqrt75 := 8
  in (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6 :=
by
  let least_integer_greater_than_sqrt8 := 3
  let greatest_integer_less_than_sqrt75 := 8
  show (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339143


namespace equal_divide_remaining_amount_all_girls_l339_339273

theorem equal_divide_remaining_amount_all_girls 
    (debt : ℕ) (savings_lulu : ℕ) (savings_nora : ℕ) (savings_tamara : ℕ)
    (total_savings : ℕ) (remaining_amount : ℕ)
    (each_girl_gets : ℕ)
    (Lulu_saved : savings_lulu = 6)
    (Nora_saved_multiple_of_Lulu : savings_nora = 5 * savings_lulu)
    (Nora_saved_multiple_of_Tamara : savings_nora = 3 * savings_tamara)
    (total_saved_calculated : total_savings = savings_nora + savings_tamara + savings_lulu)
    (debt_value : debt = 40)
    (remaining_calculated : remaining_amount = total_savings - debt)
    (division_among_girls : each_girl_gets = remaining_amount / 3) :
  each_girl_gets = 2 := 
sorry

end equal_divide_remaining_amount_all_girls_l339_339273


namespace count_integers_between_sqrt8_sqrt75_l339_339113

theorem count_integers_between_sqrt8_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ ∀ i : ℤ, (3 ≤ i ∧ i ≤ 8) ↔ (sqrt 8 < i ∧ i < sqrt 75) :=
by
  use 6
  sorry

end count_integers_between_sqrt8_sqrt75_l339_339113


namespace games_played_by_player_3_l339_339772

theorem games_played_by_player_3 (games_1 games_2 : ℕ) (rotation_system : ℕ) :
  games_1 = 10 → games_2 = 21 →
  rotation_system = (games_2 - games_1) →
  rotation_system = 11 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end games_played_by_player_3_l339_339772


namespace nearest_known_points_le_6_nearest_known_points_le_3_l339_339565

namespace Problem2815

open finite

-- Given a finite set of points in the plane, each selected point should have 
-- no more than 6 points as its nearest known points.
theorem nearest_known_points_le_6 (S : set (ℝ × ℝ)) [fintype S] : 
  ∃ T ⊆ S, ∀ x ∈ T, (point_distances x S).countp (λ d, d ≤ inf_dist x S) ≤ 6 := by
sorry

-- Given a finite set of points in the plane, each selected point should have 
-- no more than 3 points as its nearest known points.
theorem nearest_known_points_le_3 (S : set (ℝ × ℝ)) [fintype S] : 
  ∃ T ⊆ S, ∀ x ∈ T, (point_distances x S).countp (λ d, d ≤ inf_dist x S) ≤ 3 := by
sorry

end Problem2815

end nearest_known_points_le_6_nearest_known_points_le_3_l339_339565


namespace sum_of_first_10_common_elements_l339_339889

def arithmetic_progression (n : ℕ) : ℕ := 4 + 3 * n
def geometric_progression (k : ℕ) : ℕ := 10 * 2^k

theorem sum_of_first_10_common_elements :
  let common_elements := λ k : ℕ, if (k % 2 = 0) then Some (geometric_progression k) else None in
  (List.range 20).filter_map common_elements).take 10).sum = 3495250 :=
sorry

end sum_of_first_10_common_elements_l339_339889


namespace factor_difference_of_squares_196_l339_339874

theorem factor_difference_of_squares_196 (x : ℝ) : x^2 - 196 = (x - 14) * (x + 14) := by
  sorry

end factor_difference_of_squares_196_l339_339874


namespace six_digit_integers_count_l339_339616

theorem six_digit_integers_count : 
  let digits := [2, 2, 2, 5, 5, 9] in
  multiset.card (multiset.of_list (list.permutations digits).erase_dup) = 60 :=
by
  sorry

end six_digit_integers_count_l339_339616


namespace factorial_div_l339_339493

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339493


namespace common_chord_length_l339_339205

theorem common_chord_length
  (A B C D M : Point)
  (hBC : dist B C = 4)
  (hMedian : dist A M = 3)
  (hMidpoint : M = midpoint B C)
  (h_tangent_B : circle A B D tangent BC B)
  (h_tangent_C : circle A C D tangent BC C) : 
  dist A D = 5 / 3 := 
sorry

end common_chord_length_l339_339205


namespace part_a_l339_339350

theorem part_a (n : ℕ) : ((x^2 + x + 1) ∣ (x^(2 * n) + x^n + 1)) ↔ (n % 3 = 0) := sorry

end part_a_l339_339350


namespace find_x_l339_339514

def f (x : ℝ) : ℝ := 2 * x - 5
def f_inv (x : ℝ) : ℝ := (x + 5) / 2

theorem find_x (x : ℝ) : f x = f_inv x → x = 5 := by
  sorry

end find_x_l339_339514


namespace smallest_k_divides_l339_339025

noncomputable def f (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

noncomputable def g (z : ℂ) (k : ℕ) : ℂ := z^k - 1

theorem smallest_k_divides (k : ℕ) : k = 84 :=
by
  sorry

end smallest_k_divides_l339_339025


namespace john_free_throws_l339_339680

theorem john_free_throws 
  (hit_rate : ℝ) 
  (shots_per_foul : ℕ) 
  (fouls_per_game : ℕ) 
  (total_games : ℕ) 
  (percentage_played : ℝ) 
  : hit_rate = 0.7 → 
    shots_per_foul = 2 → 
    fouls_per_game = 5 → 
    total_games = 20 → 
    percentage_played = 0.8 → 
    ∃ (total_free_throws : ℕ), total_free_throws = 112 := 
by
  intros
  sorry

end john_free_throws_l339_339680


namespace factorial_division_l339_339415

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339415


namespace evaluate_area_expression_l339_339697

-- Define the regular hexagon and its properties
structure RegularHexagon :=
  (A B C D E F : Point) -- Vertices
  (area : ℝ) -- Total area

def hexagon := RegularHexagon.mk A B C D E F 1

-- Define midpoint and intersection points
def M : Point := midpoint D E
def X : Point := intersection AC BM
def Y : Point := intersection BF AM
def Z : Point := intersection AC BF

-- Define areas of polygon
def area (P : Polygon) : ℝ :=
  -- Some definition of area of polygon P
  sorry

-- Statements derived from given problem
axiom regular_hexagon_area : area (hexagon A B C D E F) = 1
axiom midpoint_DE : M = midpoint D E
axiom intersection_AC_BM : X = intersection AC BM
axiom intersection_BF_AM : Y = intersection BF AM
axiom intersection_AC_BF : Z = intersection AC BF

-- Main theorem
theorem evaluate_area_expression :
  area (polygon B X C) + area (polygon A Y F) + area (polygon A B Z) - area (polygon M X Z Y) = 0 :=
by
  sorry

end evaluate_area_expression_l339_339697


namespace smallest_nine_ten_eleven_consecutive_sum_l339_339791

theorem smallest_nine_ten_eleven_consecutive_sum :
  ∃ n : ℕ, n > 0 ∧ (n % 9 = 0) ∧ (n % 10 = 5) ∧ (n % 11 = 0) ∧ n = 495 :=
by {
  sorry
}

end smallest_nine_ten_eleven_consecutive_sum_l339_339791


namespace distinct_plants_in_garden_l339_339189

theorem distinct_plants_in_garden :
  let X := 600
  let Y := 500
  let Z := 400
  let XY := 100
  let YZ := 80
  let XZ := 120
  let XYZ := 30
  X + Y + Z - XY - YZ - XZ + XYZ = 1230 :=
by
  let X := 600
  let Y := 500
  let Z := 400
  let XY := 100
  let YZ := 80
  let XZ := 120
  let XYZ := 30
  calc
    X + Y + Z - XY - YZ - XZ + XYZ
    = 600 + 500 + 400 - 100 - 80 - 120 + 30 : by rfl
    = 1500 - 300 + 30 : by rfl
    = 1230 : by rfl

end distinct_plants_in_garden_l339_339189


namespace total_value_of_item_l339_339792

theorem total_value_of_item (V : ℝ) (h1 : 0.07 * (V - 1000) = 87.50) :
  V = 2250 :=
by
  sorry

end total_value_of_item_l339_339792


namespace prime_factors_difference_242858_l339_339788

theorem prime_factors_difference_242858 :
  ∃ p1 p2: ℕ, nat.prime p1 ∧ nat.prime p2 ∧ p1 * p2 ≠ 0 ∧ p1 ≠ p2 ∧ p1 = 97 ∧ p2 = 17 ∧ p1 - p2 = 80 := 
by
  sorry

end prime_factors_difference_242858_l339_339788


namespace diameter_of_double_area_square_l339_339346

-- Define the given conditions and the problem to be solved
theorem diameter_of_double_area_square (d₁ : ℝ) (d₁_eq : d₁ = 4 * Real.sqrt 2) :
  ∃ d₂ : ℝ, d₂ = 8 :=
by
  -- Define the conditions
  let s₁ := d₁ / Real.sqrt 2
  have s₁_sq : s₁ ^ 2 = (d₁ ^ 2) / 2 := by sorry -- Pythagorean theorem

  let A₁ := s₁ ^ 2
  have A₁_eq : A₁ = 16 := by sorry -- Given diagonal, thus area

  let A₂ := 2 * A₁
  have A₂_eq : A₂ = 32 := by sorry -- Double the area

  let s₂ := Real.sqrt A₂
  have s₂_eq : s₂ = 4 * Real.sqrt 2 := by sorry -- Side length of second square

  let d₂ := s₂ * Real.sqrt 2
  have d₂_eq : d₂ = 8 := by sorry -- Diameter of the second square

  -- Prove the theorem
  existsi d₂
  exact d₂_eq

end diameter_of_double_area_square_l339_339346


namespace shyam_weight_increase_l339_339304

theorem shyam_weight_increase (x : ℝ) 
    (h1 : x > 0)
    (ratio : ∀ Ram Shyam : ℝ, (Ram / Shyam) = 7 / 5)
    (ram_increase : ∀ Ram : ℝ, Ram' = Ram + 0.1 * Ram)
    (total_weight_after : Ram' + Shyam' = 82.8)
    (total_weight_increase : 82.8 = 1.15 * total_weight) :
    (Shyam' - Shyam) / Shyam * 100 = 22 :=
by
  sorry

end shyam_weight_increase_l339_339304


namespace min_value_h_l339_339094

def f (x : ℝ) : ℝ := 1 / Real.cos x

def g (x : ℝ) : ℝ := 1 / Real.cos (x - Real.pi / 3)

def h (x : ℝ) : ℝ := f(x) + g(x)

theorem min_value_h : 
  ∀ x : ℝ, -Real.pi / 6 < x ∧ x < Real.pi / 2 →
  ∃ m : ℝ, m = (4 * Real.sqrt 3) / 3 ∧ ∀ y : ℝ, -Real.pi / 6 < y ∧ y < Real.pi / 2 → h(y) ≥ m :=
by
  sorry

end min_value_h_l339_339094


namespace january_1_is_monday_l339_339185

theorem january_1_is_monday
  (days_in_january : ℕ)
  (mondays_in_january : ℕ)
  (thursdays_in_january : ℕ) :
  days_in_january = 31 ∧ mondays_in_january = 5 ∧ thursdays_in_january = 5 → 
  ∃ d : ℕ, d = 1 ∧ (d % 7 = 1) :=
by
  sorry

end january_1_is_monday_l339_339185


namespace sum_base_2_digits_of_4_digit_base_10_numbers_l339_339814

theorem sum_base_2_digits_of_4_digit_base_10_numbers :
  let min4digit := 1000
  let max4digit := 9999
  let values := [10, 11, 12, 13, 14]
  ∑ d in values, d = 60 :=
by sorry

end sum_base_2_digits_of_4_digit_base_10_numbers_l339_339814


namespace range_of_expression_l339_339061

noncomputable def f (a b x : ℝ) := a * x - b

theorem range_of_expression (a b : ℝ)
  (h1 : ∀ x ∈ Set.Icc (-1 : ℝ) 1, 0 ≤ f a b x ∧ f a b x ≤ 1) :
  ∃ y ∈ Set.Icc (-4 / 5) (2 / 7), ∃ x, y = (3 * a + b + 1) / (a + 2 * b - 2) :=
sorry

end range_of_expression_l339_339061


namespace number_of_integers_between_sqrts_l339_339134

theorem number_of_integers_between_sqrts :
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  (upper_bound - lower_bound + 1) = 6 :=
by
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  have sqrt8_bound : (2 : ℝ) < sqrt8 ∧ sqrt8 < 3 := sorry
  have sqrt75_bound : (8 : ℝ) < sqrt75 ∧ sqrt75 < 9 := sorry
  have lower_bound_def : lower_bound = 3 := sorry
  have upper_bound_def : upper_bound = 8 := sorry
  show (upper_bound - lower_bound + 1) = 6
  calc
    upper_bound - lower_bound + 1 = 8 - 3 + 1 := by rw [upper_bound_def, lower_bound_def]
                             ... = 6 := by norm_num
  sorry

end number_of_integers_between_sqrts_l339_339134


namespace B_overall_gain_l339_339824

-- Define the compound interest formula
def compound_interest (P r : ℝ) (n t : ℕ) : ℝ :=
  P * (1 + r / n)^(n * t)

variables (P_A P_C P_D P_E r_A r_C r_D r_E : ℝ)
variables (n_A n_C n_D n_E t_A t_C t_D t_E : ℕ)

-- Principal amounts
def P_A := 10000.0
def P_C := 4000.0
def P_D := 3000.0
def P_E := 3000.0

-- Annual interest rates (in decimal form)
def r_A := 0.08
def r_C := 0.10
def r_D := 0.12
def r_E := 0.115

-- Number of times the interest is compounded per year
def n_A := 1
def n_C := 2
def n_D := 4
def n_E := 1

-- Time periods in years
def t_A := 4
def t_C := 3
def t_D := 2
def t_E := 4

-- Future values for each amount
def A_A := compound_interest P_A r_A n_A t_A
def A_C := compound_interest P_C r_C n_C t_C
def A_D := compound_interest P_D r_D n_D t_D
def A_E := compound_interest P_E r_E n_E t_E

-- Total amount B owes to A
def total_owe_to_A := A_A

-- Total amount C, D, and E owe to B
def total_owe_to_B := A_C + A_D + A_E

-- B's overall gain
def B_gain := total_owe_to_B - total_owe_to_A

-- Proving B's overall gain
theorem B_overall_gain : B_gain = 280.46 :=
by {
  have A_A_val : A_A = 13604.90 := by sorry,
  have A_C_val : A_C = 5360.38 := by sorry,
  have A_D_val : A_D = 3800.31 := by sorry,
  have A_E_val : A_E = 4724.67 := by sorry,
  have owe_to_B : total_owe_to_B = 13885.36 := by sorry,
  have owe_to_A : total_owe_to_A = 13604.90 := by sorry,
  show B_gain = 280.46,
  sorry
}

end B_overall_gain_l339_339824


namespace integer_count_between_sqrt8_and_sqrt75_l339_339141

noncomputable def m : ℤ := Int.ceil (Real.sqrt 8)
noncomputable def n : ℤ := Int.floor (Real.sqrt 75)

theorem integer_count_between_sqrt8_and_sqrt75 : (n - m + 1) = 6 := by
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339141


namespace ratio_of_areas_l339_339356

variables {R : ℝ} (A B C O O₁ : Point)
variables (h1 : right_triangle A B C)
variables (h2 : circumscribed_circle R A B C O)
variables (h3 : same_radius_circle_tangent_legs R A B C O₁)

theorem ratio_of_areas : 
  let S_triangle := (1/2) * (distance A C) * (distance B C) in
  let S_common := ((5 * Real.pi - 6) / 12) * R^2 in
  (S_triangle / S_common) = (3 * Real.sqrt 3) / (5 * Real.pi - 3) :=
sorry

end ratio_of_areas_l339_339356


namespace people_arrangement_count_l339_339307

-- Definitions for people and constraints
inductive Person
| A | B | C | D | E

open Person

-- Define adjacency constraint
def not_adjacent (p1 p2 : Person) (arrangement : List Person) : Prop :=
  ∀ i, arrangement.nth i = some p1 → (arrangement.nth (i + 1) ≠ some p2 ∧ arrangement.nth (i - 1) ≠ some p2)

-- Define the main problem condition
def valid_arrangement (arrangement : List Person) : Prop :=
  arrangement = [A, B, C, D, E].perm ∧
  not_adjacent A C arrangement ∧
  not_adjacent B C arrangement

-- Theorem statement
theorem people_arrangement_count : ∃ (n : ℕ), n = 36 ∧ ∃ arrs : Finset (List Person), arrs.card = n ∧ ∀ arr ∈ arrs, valid_arrangement arr :=
sorry

end people_arrangement_count_l339_339307


namespace factorial_division_identity_l339_339503

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339503


namespace factorial_computation_l339_339475

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339475


namespace problem1_problem2_l339_339264

noncomputable def problem1_solution1 : ℝ := (2 + Real.sqrt 6) / 2
noncomputable def problem1_solution2 : ℝ := (2 - Real.sqrt 6) / 2

theorem problem1 (x : ℝ) : 
  (2 * x ^ 2 - 4 * x - 1 = 0) ↔ (x = problem1_solution1 ∨ x = problem1_solution2) :=
by
  sorry

theorem problem2 : 
  (4 * (x + 2) ^ 2 - 9 * (x - 3) ^ 2 = 0) ↔ (x = 1 ∨ x = 13) :=
by
  sorry

end problem1_problem2_l339_339264


namespace probability_of_specific_sequence_l339_339311

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l339_339311


namespace integer_count_between_sqrt8_and_sqrt75_l339_339145

theorem integer_count_between_sqrt8_and_sqrt75 :
  let least_integer_greater_than_sqrt8 := 3
      greatest_integer_less_than_sqrt75 := 8
  in (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6 :=
by
  let least_integer_greater_than_sqrt8 := 3
  let greatest_integer_less_than_sqrt75 := 8
  show (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339145


namespace binary_mul_1101_111_eq_1001111_l339_339011

theorem binary_mul_1101_111_eq_1001111 :
  let n1 := 0b1101 -- binary representation of 13
  let n2 := 0b111  -- binary representation of 7
  let product := 0b1001111 -- binary representation of 79
  n1 * n2 = product :=
by
  sorry

end binary_mul_1101_111_eq_1001111_l339_339011


namespace find_a_l339_339580

theorem find_a (a : ℤ) : 0 ≤ a ∧ a ≤ 13 ∧ (51^2015 + a) % 13 = 0 → a = 1 :=
by { sorry }

end find_a_l339_339580


namespace factorial_division_sum_l339_339467

theorem factorial_division_sum :
  (8! + 9!) / 7! = 80 := by
  sorry

end factorial_division_sum_l339_339467


namespace g_sum_eq_l339_339056

def g (n : ℕ) : ℝ := Real.logBase 3003 (n ^ 3)

theorem g_sum_eq : g 7 + g 11 + g 13 = 9 / 4 :=
by
  sorry

end g_sum_eq_l339_339056


namespace extremum_f_monotonicity_interval_l339_339348

noncomputable def f (a x : ℝ) := Real.exp x + a * x
noncomputable def g (a x : ℝ) := a * x - Real.log x

theorem extremum_f (a : ℝ) (h₁ : a ≤ 0) : 
    ∃ x, f a x = -a + a * Real.log (-a) := by sorry

theorem monotonicity_interval (a : ℝ) (h₁ : a ≤ 0) :
    ∀ x, x ∈ Iio (-1:ℝ) → (f a x)' = (g a x)' := by sorry

end extremum_f_monotonicity_interval_l339_339348


namespace simplify_expr1_simplify_expr2_l339_339347

-- First problem
theorem simplify_expr1 : (0.027 : ℝ) ^ (2 / 3) + ( (27 / 125) : ℝ) ^ (-1 / 3) - ( (2 * (7 / 9)) : ℝ) ^ 0.5 = 9 / 100 :=
sorry

-- Second problem
theorem simplify_expr2 (a b : ℝ) : 
  ( (1 / 4) : ℝ) ^ (-1 / 2) * (( (4 * a * b⁻¹).sqrt) ^ 3 / ( (0.1 : ℝ) ^ -1 * (a ^ 3 * b ^ -3) ^ (1 / 2))) = 8 / 5 :=
sorry

end simplify_expr1_simplify_expr2_l339_339347


namespace factorial_div_l339_339490

def eight_factorial := Nat.factorial 8
def nine_factorial := Nat.factorial 9
def seven_factorial := Nat.factorial 7

theorem factorial_div : (eight_factorial + nine_factorial) / seven_factorial = 80 := by
  sorry

end factorial_div_l339_339490


namespace total_travel_time_l339_339708

-- We define the conditions as assumptions in the Lean 4 statement.

def riding_rate := 10  -- miles per hour
def time_first_part_minutes := 30  -- initial 30 minutes in minutes
def additional_distance_1 := 15  -- another 15 miles
def rest_time_minutes := 30  -- resting for 30 minutes
def remaining_distance := 20  -- remaining distance of 20 miles

theorem total_travel_time : 
    let time_first_part := time_first_part_minutes
    let time_second_part := (additional_distance_1 : ℚ) / riding_rate * 60  -- convert hours to minutes
    let time_third_part := rest_time_minutes
    let time_fourth_part := (remaining_distance : ℚ) / riding_rate * 60  -- convert hours to minutes
    time_first_part + time_second_part + time_third_part + time_fourth_part = 270 :=
by
  sorry

end total_travel_time_l339_339708


namespace find_circle_equation_find_m_l339_339915

-- Definitions for the conditions
def line_l (x y : ℝ) : Prop := x - √3 * y + 1 = 0
def circle_center_positive_x (x y : ℝ) : Prop := y = 0 ∧ x > 0

-- Given that a circle is tangent to line l and the y-axis
structure circle_tangent (x y r : ℝ) :=
  (radius_positive : r > 0)
  (tangent_to_line : abs (x + 1) / 2 = r)
  (tangent_to_y_axis : x = r)
  
-- Part 1: Equation of the circle
theorem find_circle_equation (x y r : ℝ) (h : circle_center_positive_x x y) (ht : circle_tangent x y r) :
  (x = 1) ∧ (r = 1) ∧ (x - 1)^2 + y^2 = 1 := 
sorry

-- Part 2: Finding the value of m
def line_m (x y m : ℝ) : Prop := mx + y + (1/2)*m = 0

theorem find_m (m : ℝ) 
  (h : ∀ (x y : ℝ), circle_center_positive_x x y → circle_tangent x y 1 → line_m x y m → 
    ∃ A B : ℝ × ℝ, (A ≠ B) ∧ ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 3)) :
  m = √11 / 11 ∨ m = -√11 / 11 :=
sorry

end find_circle_equation_find_m_l339_339915


namespace factorial_computation_l339_339471

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339471


namespace arrangements_count_l339_339324

variable (students : List String)
variable (pos : Fin 5)
variable (boyA : String)
variable (boys : Set String)
variable (girls : Set String)

def is_adjacent (indices : List Nat) : Bool :=
  match indices with
  | i :: j :: _ => i + 1 = j
  | _ => false

theorem arrangements_count (students = ["boy1", "boyA", "girl1", "girl2", "girl3"] : 
  boyA ∈ students ∧
  (boyA ≠ students.head ∧ boyA ≠ students.ilast) ∧ 
  (∃ g1 g2, g1 ≠ g2 ∧ g1 ∈ girls ∧ g2 ∈ girls ∧ g2 ∈ students.tail ∧ 
  ∃ g3 ∈ students, ¬(g3 ∈ Set.insert g1 {g2}) ∧ 
  is_adjacent [students.indexOf g1, students.indexOf g2]) ∧
  is_perm students ["boy1", "boyA", "girl1", "girl2", "girl3"]
  → count_arrangements students = 48) :=
sorry

end arrangements_count_l339_339324


namespace zoe_remaining_pictures_l339_339797

theorem zoe_remaining_pictures (pictures_per_book : ℕ) (books_count : ℕ)(colored_pictures : ℕ) :
  books_count = 2 → pictures_per_book = 44 → colored_pictures = 20 → 
  (books_count * pictures_per_book) - colored_pictures = 68 := 
by
  assume h1 h2 h3,
  rw [h1, h2, h3],
  norm_num,
  sorry

end zoe_remaining_pictures_l339_339797


namespace weighted_average_is_correct_l339_339978

structure Semester :=
  (courses : ℕ)
  (avg_grade : ℝ)

def year1_fall : Semester := ⟨4, 75⟩
def year1_spring : Semester := ⟨5, 90⟩
def year2_fall : Semester := ⟨6, 85⟩
def year2_spring : Semester := ⟨3, 95⟩
def year3_fall : Semester := ⟨5, 100⟩
def year3_spring : Semester := ⟨6, 88⟩
def year4_fall : Semester := ⟨4, 80⟩
def year4_spring : Semester := ⟨5, 92⟩

noncomputable def calculate_weighted_average (semesters : list Semester) : ℝ :=
  let total_points := semesters.map (λ s, s.courses * s.avg_grade).sum in
  let total_courses := semesters.map (λ s, s.courses).sum in
  total_points / total_courses

theorem weighted_average_is_correct :
  calculate_weighted_average [year1_fall, year1_spring, year2_fall, year2_spring, year3_fall, year3_spring, year4_fall, year4_spring] = 88.2 :=
by
  sorry

end weighted_average_is_correct_l339_339978


namespace unique_chair_arrangement_l339_339647

theorem unique_chair_arrangement (n : ℕ) (h : n = 49)
  (h1 : ∀ i j : ℕ, (n = i * j) → (i ≥ 2) ∧ (j ≥ 2)) :
  ∃! i j : ℕ, (n = i * j) ∧ (i ≥ 2) ∧ (j ≥ 2) :=
by
  sorry

end unique_chair_arrangement_l339_339647


namespace factorial_computation_l339_339473

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339473


namespace water_level_equilibrium_l339_339327

theorem water_level_equilibrium (h : ℝ) (ρ_water ρ_oil : ℝ) :
  h = 40 → ρ_water = 1000 → ρ_oil = 700 →
  let h_1 := 40 / (1 + ρ_water / ρ_oil)
  in h_1 ≈ 16.47 :=
by
  sorry

end water_level_equilibrium_l339_339327


namespace smallest_k_l339_339034

def polynomial_p (z : ℤ) : polynomial ℤ :=
  z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k (k : ℕ) (z : ℤ) : 
  (∀ z, polynomial_p z ∣ (z^k - 1)) ↔ k = 126 :=
sorry

end smallest_k_l339_339034


namespace number_of_integers_between_sqrts_l339_339130

theorem number_of_integers_between_sqrts :
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  (upper_bound - lower_bound + 1) = 6 :=
by
  let sqrt8 := Real.sqrt 8
  let sqrt75 := Real.sqrt 75
  let lower_bound := Int.ceil sqrt8
  let upper_bound := Int.floor sqrt75
  have sqrt8_bound : (2 : ℝ) < sqrt8 ∧ sqrt8 < 3 := sorry
  have sqrt75_bound : (8 : ℝ) < sqrt75 ∧ sqrt75 < 9 := sorry
  have lower_bound_def : lower_bound = 3 := sorry
  have upper_bound_def : upper_bound = 8 := sorry
  show (upper_bound - lower_bound + 1) = 6
  calc
    upper_bound - lower_bound + 1 = 8 - 3 + 1 := by rw [upper_bound_def, lower_bound_def]
                             ... = 6 := by norm_num
  sorry

end number_of_integers_between_sqrts_l339_339130


namespace rectangle_width_eq_six_l339_339830

theorem rectangle_width_eq_six (w : ℝ) :
  ∃ w, (3 * w = 25 - 7) ↔ w = 6 :=
by
  -- Given the conditions as stated:
  -- Length of the rectangle: 3 inches
  -- Width of the square: 5 inches
  -- Difference in area between the square and the rectangle: 7 square inches
  -- We can show that the width of the rectangle is 6 inches.
  sorry

end rectangle_width_eq_six_l339_339830


namespace f_2016_eq_2017_l339_339943

-- Definition of the function f(x)
def f : ℕ → ℝ
| x := if x ≤ 1 then Real.logb 2 (5 - x) else f (x - 1) + 1

-- The Lean theorem statement to prove f(2016) = 2017
theorem f_2016_eq_2017 : f 2016 = 2017 := 
sorry

end f_2016_eq_2017_l339_339943


namespace find_quotient_l339_339716

theorem find_quotient (dividend divisor remainder quotient : ℕ) 
  (h_dividend : dividend = 171) 
  (h_divisor : divisor = 21) 
  (h_remainder : remainder = 3) 
  (h_div_eq : dividend = divisor * quotient + remainder) :
  quotient = 8 :=
by sorry

end find_quotient_l339_339716


namespace olympic_auspicious_sum_l339_339742

noncomputable def log_base (b x : ℝ) : ℝ := real.log x / real.log b

def a_n (n : ℕ) : ℝ := log_base (n + 1) (n + 2)

def is_olympic_auspicious_number (k : ℕ) : Prop :=
  ∃ m : ℕ, log_base 2 (k + 2) = m

def olympic_auspicious_numbers_in_interval (n : ℕ) : list ℕ :=
  (list.range (n + 1)).filter (λ k, 1 ≤ k ∧ k ≤ 2012 ∧ is_olympic_auspicious_number k)

theorem olympic_auspicious_sum :
  (olympic_auspicious_numbers_in_interval 2012).sum = 2026 :=
sorry

end olympic_auspicious_sum_l339_339742


namespace factorial_computation_l339_339469

theorem factorial_computation : (8.factorial + 9.factorial) / 7.factorial = 80 :=
by sorry

end factorial_computation_l339_339469


namespace sum_of_parallel_segments_l339_339728

theorem sum_of_parallel_segments (AB CD : ℝ) (AB_div : (0 : set ℝ) = set.range (λ i, 8 * i / 200))
    (CB_div : (0 : set ℝ) = set.range (λ j, 6 * j / 200)) :
    (2 * (∑ k in finset.range 200, (10 * (200 - k) / 200)) - 10 = 2000) :=
by
  sorry

end sum_of_parallel_segments_l339_339728


namespace coords_P_origin_l339_339990

variable (x y : Int)
def point_P := (-5, 3)

theorem coords_P_origin : point_P = (-5, 3) := 
by 
  -- Proof to be written here
  sorry

end coords_P_origin_l339_339990


namespace total_travel_time_in_minutes_l339_339711

def riding_rate : ℝ := 10 -- 10 miles per hour
def initial_riding_time : ℝ := 30 -- 30 minutes
def another_riding_distance : ℝ := 15 -- 15 miles
def resting_time : ℝ := 30 -- 30 minutes
def remaining_distance : ℝ := 20 -- 20 miles

theorem total_travel_time_in_minutes :
  initial_riding_time +
  (another_riding_distance / riding_rate * 60) +
  resting_time +
  (remaining_distance / riding_rate * 60) = 270 :=
by
  sorry

end total_travel_time_in_minutes_l339_339711


namespace triangle_is_obtuse_l339_339180

theorem triangle_is_obtuse
  (A B C : ℝ)
  (h_angle_sum : A + B + C = π)
  (h_condition : cos A * cos B > sin A * sin B) :
  C > π / 2 :=
sorry

end triangle_is_obtuse_l339_339180


namespace ababab_divisible_by_13_l339_339794

theorem ababab_divisible_by_13 (a b : ℕ) (ha: a < 10) (hb: b < 10) : 
  13 ∣ (100000 * a + 10000 * b + 1000 * a + 100 * b + 10 * a + b) := 
by
  sorry

end ababab_divisible_by_13_l339_339794


namespace geom_seq_a_sum_seq_b_over_a_l339_339567

-- Definition of the sequence {a_n}
def seq_a (a : ℕ → ℕ) (S : ℕ → ℕ) : Prop :=
  a 1 = 0 ∧ ∀ n : ℕ, n > 0 → S n + n = a (n + 1)

-- Theorem 1: Prove that {a_n + 1} is a geometric sequence
theorem geom_seq_a (a S : ℕ → ℕ) (h : seq_a a S) : 
  ∃ r : ℕ, r > 1 ∧ ∀ n : ℕ, n > 0 → a (n + 1) + 1 = r * (a n + 1) := sorry

-- Definition of the sequence {b_n}
def seq_b (b : ℕ → ℕ) (T : ℕ → ℕ) : Prop :=
  b 1 = 1 ∧ ∀ n : ℕ, n > 0 → T (n + 1) / (n + 1) = T n / n + 1 / 2

-- Theorem 2: Find the sum of the first n terms of {b_n / (a_n + 1)}
theorem sum_seq_b_over_a (a b : ℕ → ℕ) (T : ℕ → ℕ) (h_a : seq_a a (λ n, (a n * (a n + 1)) / 2)) (h_b : seq_b b T) :
  ∀ n : ℕ, n > 0 → (∑ i in finset.range n, b i / (a i + 1)) = 4 - (n + 2) / 2 ^ (n - 1) := sorry

end geom_seq_a_sum_seq_b_over_a_l339_339567


namespace bianca_arrives_at_aria_start_at_9_45_l339_339843

-- Definitions and conditions from the problem
def starts_at_8 (person : String) : Prop := person = "Aria" ∨ person = "Bianca"
def walks_directly_towards (person1 person2 : String) : Prop := 
  (startswith person1 "A" ∧ startswith person2 "B") ∨ (startswith person1 "B" ∧ startswith person2 "A")
def pass_each_other_at (time : String) : Prop := time = "8:42 a.m."
def aria_arrives_at_B_at (time : String) : Prop := time = "9:10 a.m."

-- Definition of the function to determine Bianca's arrival time at Aria's starting point
def bianca_arrival_time (bianca_start_time : String) : String := "9:45 a.m."

-- Theorem statement for the proof problem
theorem bianca_arrives_at_aria_start_at_9_45 :
  (∀ person, starts_at_8 person) →
  (∀ person1 person2, walks_directly_towards person1 person2) →
  pass_each_other_at "8:42 a.m." →
  aria_arrives_at_B_at "9:10 a.m." →
  bianca_arrival_time "8:00 a.m." = "9:45 a.m." :=
  sorry

end bianca_arrives_at_aria_start_at_9_45_l339_339843


namespace two_degrees_above_zero_l339_339156

-- Define the concept of temperature notation
def temperature_notation (temp: ℝ) : String :=
  if temp < 0 then "-" ++ temp.nat_abs.toString ++ "°C"
  else "+" ++ temp.toString ++ "°C"

-- Given condition: -2 degrees Celsius is denoted as -2°C
def given_condition := temperature_notation (-2) = "-2°C"

-- Proof statement: 2 degrees Celsius above zero is denoted as +2°C given the condition
theorem two_degrees_above_zero : given_condition → temperature_notation 2 = "+2°C" := by
  intro h
  sorry

end two_degrees_above_zero_l339_339156


namespace prove_a_lt_two_l339_339865

-- Define the conditions
def is_partition_of_infinite_subsets {n : ℕ} (A : Fin n → Set ℕ) : Prop :=
  (∀ i, A i ≠ ∅ ∧ ∀ x y ∈ A i, x ≠ y → |x - y| ≥ a^i) ∧
  (∀ i, ∃ m, ∃_inf (λ j, j * m) (A i)) ∧
  (∀ x, ∃ i, x ∈ A i) ∧
  (∅ = 0)

-- Statement
theorem prove_a_lt_two (a : ℝ) :
  (∃ (n : ℕ) (A : Fin n → Set ℕ), is_partition_of_infinite_subsets A ∧ ∀ i, ∀ x y ∈ A i, x ≠ y → |x - y| ≥ a^i) → a < 2 := 
sorry

end prove_a_lt_two_l339_339865


namespace max_area_rectangle_from_triangle_l339_339829

theorem max_area_rectangle_from_triangle :
  ∀ (a b : ℤ), (a = 40) → (b = 60) → 
  let area_triangle := (a * b / 2) in
  let area_rectangle := (area_triangle / 2) in
  area_rectangle = 600 :=
by
  intros a b h1 h2
  have h3 : a * b = 40 * 60 := by rw [h1, h2]
  have h4 : a * b / 2 = 40 * 60 / 2 := by rw [h1, h2]
  have h5 : (40 * 60) / 2 / 2 = 600 := by norm_num
  rw [← h3, ← h4, h5]
  sorry

end max_area_rectangle_from_triangle_l339_339829


namespace solve_problem_l339_339509

def avg2 (a b : ℚ) : ℚ := (a + b) / 2
def avg4 (a b c d : ℚ) : ℚ := (a + b + c + d) / 4

theorem solve_problem :
  avg4 (avg3 2 2 (-1)) (avg2 1 3) 7 (avg2 4 (5 - 2)) = 27 / 8 := by
  sorry

end solve_problem_l339_339509


namespace carter_performance_nights_l339_339854

theorem carter_performance_nights
  (sets_per_show : ℕ)
  (sets_tossed : ℕ)
  (total_sets_used : ℕ)
  (h_sets_per_show : sets_per_show = 5)
  (h_sets_tossed : sets_tossed = 6)
  (h_total_sets_used : total_sets_used = 330) :
  ∃ n : ℕ, n = 30 :=
by {
  let sets_per_night := sets_per_show + sets_tossed,
  have h_sets_per_night : sets_per_night = 11 := by rw [h_sets_per_show, h_sets_tossed],
  let n := total_sets_used / sets_per_night,
  have h_n : n = 30 := by rw [h_sets_per_night, h_total_sets_used]; norm_num,
  use n,
  exact h_n,
  sorry
}

end carter_performance_nights_l339_339854


namespace fundraiser_total_money_l339_339545

def number_of_items (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student : ℕ) : ℕ :=
  (students1 * brownies_per_student) + (students2 * cookies_per_student) + (students3 * donuts_per_student)

def total_money_raised (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) : ℕ :=
  number_of_items students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student * price_per_item

theorem fundraiser_total_money (students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item : ℕ) :
  students1 = 30 → students2 = 20 → students3 = 15 → brownies_per_student = 12 → cookies_per_student = 24 → donuts_per_student = 12 → price_per_item = 2 → 
  total_money_raised students1 students2 students3 brownies_per_student cookies_per_student donuts_per_student price_per_item = 2040 :=
  by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end fundraiser_total_money_l339_339545


namespace factorial_division_identity_l339_339507

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339507


namespace find_original_numbers_l339_339383

-- Definitions corresponding to the conditions in a
def is_valid_number (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ ∃ x y : ℕ, x + y = 8 ∧ n = 10 * x + y

-- Definitions to state the condition about the swapped number and their product
def swapped_number (n : ℕ) : ℕ :=
  let x := n / 10 in
  let y := n % 10 in
  10 * y + x

def product_of_numbers (n : ℕ) : Prop :=
  n * swapped_number(n) = 1855

-- Statement combining conditions and correct answer
theorem find_original_numbers (n : ℕ) :
  is_valid_number n ∧ product_of_numbers n → n = 35 ∨ n = 53 :=
sorry

end find_original_numbers_l339_339383


namespace delta_y_over_delta_x_l339_339602

theorem delta_y_over_delta_x (Δx : ℝ) :
  let f := λ x : ℝ, 2 * x^2 - 4 in
  let y1 := f 1 in
  let Δy := f (1 + Δx) - y1 in
  Δy / Δx = 4 + 2 * Δx :=
by
  let f := λ x : ℝ, 2 * x^2 - 4
  let y1 := f 1
  let Δy := f (1 + Δx) - y1
  have h1 : Δy = 2 * Δx^2 + 4 * Δx := by sorry
  have h2 : Δy / Δx = (2 * Δx^2 + 4 * Δx) / Δx := by sorry
  have h3 : (2 * Δx^2 + 4 * Δx) / Δx = 2 * Δx + 4 := by sorry
  sorry

end delta_y_over_delta_x_l339_339602


namespace coeff_abs_sum_eq_729_l339_339233

-- Given polynomial (2x - 1)^6 expansion
theorem coeff_abs_sum_eq_729 (x a_0 a_1 a_2 a_3 a_4 a_5 a_6 : ℝ) :
  (2 * x - 1) ^ 6 = a_6 * x ^ 6 + a_5 * x ^ 5 + a_4 * x ^ 4 + a_3 * x ^ 3 + a_2 * x ^ 2 + a_1 * x + a_0 →
  |a_0| + |a_1| + |a_2| + |a_3| + |a_4| + |a_5| + |a_6| = 729 :=
by
  sorry

end coeff_abs_sum_eq_729_l339_339233


namespace num_pairs_3_pow_m_add_7_pow_n_div_10_eq_l339_339896

theorem num_pairs_3_pow_m_add_7_pow_n_div_10_eq :
  let pairs := {p : Nat × Nat | 1 ≤ p.1 ∧ p.1 ≤ 100 ∧ 101 ≤ p.2 ∧ p.2 ≤ 205 ∧ (3^p.1 + 7^p.2) % 10 = 0}
  in pairs.card = 2625 :=
by
  let pairs := {p : Nat × Nat | 1 ≤ p.1 ∧ p.1 ≤ 100 ∧ 101 ≤ p.2 ∧ p.2 ≤ 205 ∧ (3^p.1 + 7^p.2) % 10 = 0}
  have pair_count := pairs.card
  have eq_2625 := pair_count = 2625
  exact eq_2625

end num_pairs_3_pow_m_add_7_pow_n_div_10_eq_l339_339896


namespace number_of_bus_stops_l339_339293

theorem number_of_bus_stops : ∃ s : ℕ, (s = 7) ∧
  -- Define conditions
  (∀ (L : ℕ), (L = s * (s - 1) / 6) ∧
  (∀ (l1 l2 : set ℕ), l1 ≠ l2 → (∃! x ∈ l1 ∩ l2, x)
     → (l1.card = 3) ∧ (l2.card = 3)) ∧
  (∀ (p1 p2 : ℕ), p1 ≠ p2 → (∃! l, l.card = 3 ∧ p1 ∈ l ∧ p2 ∈ l))) :=
by sorry

end number_of_bus_stops_l339_339293


namespace sum_first_10_common_elements_l339_339886

theorem sum_first_10_common_elements :
  let AP := λ n : ℕ, 4 + 3 * n
  let GP := λ k : ℕ, 10 * 2^k
  let common_elements k := 10 * 4^k
  (Σ i in Finset.range 10, common_elements i) = 3495250 := by
  sorry

end sum_first_10_common_elements_l339_339886


namespace factorial_div_sum_l339_339421

theorem factorial_div_sum (Q: ℕ) (hQ: Q = (8! + 9!) / 7!) : Q = 80 := by
  sorry

end factorial_div_sum_l339_339421


namespace imo_1974_q1_l339_339248

open Nat

theorem imo_1974_q1 (n : ℕ) : ¬ (∑ k in range (n + 1), 2^(3 * k) * Nat.choose (2 * n + 1) (2 * k + 1)) % 5 = 0 :=
sorry

end imo_1974_q1_l339_339248


namespace simplify_expression_l339_339254

theorem simplify_expression (a b : ℝ) :
  3 * a - 4 * b + 2 * a^2 - (7 * a - 2 * a^2 + 3 * b - 5) = -4 * a - 7 * b + 4 * a^2 + 5 :=
by
  sorry

end simplify_expression_l339_339254


namespace select_two_groups_l339_339389

theorem select_two_groups (n : ℕ) (r1 r2 excl : ℕ) (ages : List ℕ) 
  (h1 : ages.length = 30) 
  (h2 : r1 = 12) 
  (h3 : r2 = 15) 
  (h4 : excl = 3)
  (h5 : ages.nodup) 
  (h6 : (∀ i j, i < j → ages.nth i < ages.nth j)) :
  nat.choose 30 3 = 4060 := 
  sorry

end select_two_groups_l339_339389


namespace last_locker_opened_l339_339377

theorem last_locker_opened :
  ∃ n : ℕ, n = 494 ∧
    ∀ k : ℕ, (1 ≤ k ∧ k ≤ 500) →
    (k ≠ n → ∃ l : ℕ, ((1 ≤ l ∧ l ≤ 500) → (l mod 3 = 0) ∧ ∀ m : ℕ, ((1 ≤ m ∧ m ≤ 500) ∧ (m ≠ l) → m < l))) :=
by { sorry }

end last_locker_opened_l339_339377


namespace log2_T_eq_1005_l339_339851

noncomputable def T : ℝ := 
    ((1 : ℂ) + complex.I)^(2011 : ℕ) + ((1 : ℂ) - complex.I)^(2011 : ℕ) 
    |> (λ z, ((z + complex.conj z) / 2).re)

theorem log2_T_eq_1005 : log (T : ℝ) / log 2 = 1005 := 
by
  sorry

end log2_T_eq_1005_l339_339851


namespace edge_length_is_eight_l339_339782

-- Define the conditions as assumptions
variable (a : ℕ) (cond1_valid cond2_valid : Prop)

-- Condition 1: ratio condition
def cond1_valid : Prop := 6 * a^2 / a^3 = 4 / 9

-- Condition 2: proportion condition
def cond2_valid : Prop := 6 * a^2 = 3 * (12 * a)

-- The main goal to prove edge length is 8 cm
theorem edge_length_is_eight (h1 : ¬ cond1_valid) (h2 : cond2_valid) : a + 2 = 8 := 
by
  sorry

end edge_length_is_eight_l339_339782


namespace smallest_k_divides_l339_339044

noncomputable def polynomial := λ (z : ℂ), z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

theorem smallest_k_divides (k : ℕ) :
  (∀ z : ℂ, polynomial z ∣ z^k - 1) → (k = 24) :=
by { sorry }

end smallest_k_divides_l339_339044


namespace isosceles_triangle_max_inscribed_circle_l339_339538

noncomputable def largest_inscribed_circle_base_length : ℝ :=
  let y := λ x : ℝ, (x * sqrt ((2 - x) / (2 + x)))^2 in
  if h : 0 < sqrt 5 - 1 ∧ sqrt 5 - 1 < 2 then sqrt 5 - 1 else 0

noncomputable def inscribed_circle_diameter : ℝ :=
  sqrt (10 * sqrt 5 - 22)

theorem isosceles_triangle_max_inscribed_circle :
  ∃ x : ℝ, x = sqrt 5 - 1 ∧ 
           ∀ x' : ℝ, (0 < x' ∧ x' < 2) → 
           (λ x, x * sqrt ((2 - x) / (2 + x)))^2 x' ≤ (λ x, x * sqrt ((2 - x) / (2 + x)))^2 (sqrt 5 - 1) :=
by 
  sorry

end isosceles_triangle_max_inscribed_circle_l339_339538


namespace integer_count_between_sqrt8_and_sqrt75_l339_339136

noncomputable def m : ℤ := Int.ceil (Real.sqrt 8)
noncomputable def n : ℤ := Int.floor (Real.sqrt 75)

theorem integer_count_between_sqrt8_and_sqrt75 : (n - m + 1) = 6 := by
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339136


namespace n_squared_plus_n_is_even_l339_339732

theorem n_squared_plus_n_is_even (n : ℤ) : Even (n^2 + n) :=
by
  sorry

end n_squared_plus_n_is_even_l339_339732


namespace probability_of_specific_sequence_l339_339310

def probFirstDiamond : ℚ := 13 / 52
def probSecondSpadeGivenFirstDiamond : ℚ := 13 / 51
def probThirdHeartGivenDiamondSpade : ℚ := 13 / 50

def combinedProbability : ℚ :=
  probFirstDiamond * probSecondSpadeGivenFirstDiamond * probThirdHeartGivenDiamondSpade

theorem probability_of_specific_sequence :
  combinedProbability = 2197 / 132600 := by
  sorry

end probability_of_specific_sequence_l339_339310


namespace hayley_stickers_l339_339354

theorem hayley_stickers (S F x : ℕ) (hS : S = 72) (hF : F = 9) (hx : x = S / F) : x = 8 :=
by
  sorry

end hayley_stickers_l339_339354


namespace factorial_div_sum_l339_339432

theorem factorial_div_sum :
  (fact 8 + fact 9) / fact 7 = 80 :=
by sorry

end factorial_div_sum_l339_339432


namespace sum_of_solutions_l339_339695

def g (x : ℝ) : ℝ := 3 * x + 2

def g_inv (x : ℝ) : ℝ := (x - 2) / 3

theorem sum_of_solutions : 
  ∑ x in {x : ℝ | g_inv(x) = g(x⁻¹)}, x = 8 := 
by
  sorry

end sum_of_solutions_l339_339695


namespace percent_parrots_among_non_pelicans_l339_339654

theorem percent_parrots_among_non_pelicans 
  (parrots_percent pelicans_percent owls_percent sparrows_percent : ℝ) 
  (H1 : parrots_percent = 40) 
  (H2 : pelicans_percent = 20) 
  (H3 : owls_percent = 15) 
  (H4 : sparrows_percent = 100 - parrots_percent - pelicans_percent - owls_percent)
  (H5 : pelicans_percent / 100 < 1) :
  parrots_percent / (100 - pelicans_percent) * 100 = 50 :=
by sorry

end percent_parrots_among_non_pelicans_l339_339654


namespace factorial_division_l339_339481

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339481


namespace part1_solution_part2_solution_l339_339403

noncomputable def part1_expression := 0.25 * (1 / 2) ^ -4 - 4 / (Real.sqrt 5 - 1) ^ 0 - (1 / 16) ^ (-1 / 2)
theorem part1_solution : part1_expression = -4 := sorry

noncomputable def part2_expression := Real.log 25 + Real.log 2 * Real.log 50 + (Real.log 2)^2
theorem part2_solution : part2_expression = 2 := sorry

end part1_solution_part2_solution_l339_339403


namespace coordinates_of_P_l339_339992

/-- In the Cartesian coordinate system, given a point P with coordinates (-5, 3),
    prove that its coordinates with respect to the origin are (-5, 3). -/
theorem coordinates_of_P :
  ∀ (P : ℝ × ℝ), P = (-5, 3) → P = (-5, 3) :=
by
  intro P h,
  exact h

end coordinates_of_P_l339_339992


namespace total_distance_walked_l339_339795

-- Define the conditions
def home_to_school : ℕ := 750
def half_distance : ℕ := home_to_school / 2
def return_home : ℕ := half_distance
def home_to_school_again : ℕ := home_to_school

-- Define the theorem statement
theorem total_distance_walked : 
  half_distance + return_home + home_to_school_again = 1500 := by
  sorry

end total_distance_walked_l339_339795


namespace sin_plus_cos_eq_tan_minus_cot_eq_l339_339561

open Real

variables (α : ℝ)

def sin_minus_cos (α : ℝ) : Prop := sin α - cos α = sqrt 10 / 5
def alpha_range (α : ℝ) : Prop := α ∈ Ioo π (2 * π)

theorem sin_plus_cos_eq (h1 : sin_minus_cos α) (h2 : alpha_range α) : 
  sin α + cos α = -2 * sqrt 10 / 5 := 
sorry

theorem tan_minus_cot_eq (h1 : sin_minus_cos α) (h2 : alpha_range α) : 
  tan α - cot α = -8 / 3 :=
sorry

end sin_plus_cos_eq_tan_minus_cot_eq_l339_339561


namespace h_at_3_l339_339860

-- Define the function h(x)
def h (x : ℝ) : ℝ :=
  ((x + 2) * (x^2 + 1) * (x^4 + 1) * ... * (x^(2^2008) + 1) - 2) / (x^(2^2009 - 1) - 1)

-- Prove h(3) = 3
theorem h_at_3 : h 3 = 3 :=
by
  -- The main proof goes here
  sorry

end h_at_3_l339_339860


namespace ratio_of_andy_age_in_5_years_to_rahim_age_l339_339644

def rahim_age_now : ℕ := 6
def andy_age_now : ℕ := rahim_age_now + 1
def andy_age_in_5_years : ℕ := andy_age_now + 5
def ratio (a b : ℕ) : ℕ := a / b

theorem ratio_of_andy_age_in_5_years_to_rahim_age : ratio andy_age_in_5_years rahim_age_now = 2 := by
  sorry

end ratio_of_andy_age_in_5_years_to_rahim_age_l339_339644


namespace correct_statements_count_l339_339286

/--
Given the following conditions:
1. The opposite of π is -π
2. Numbers with opposite signs are opposite numbers to each other
3. The opposite of -3.8 is 3.8
4. A number and its opposite may be equal
5. Positive numbers and negative numbers are opposite to each other.

Prove that the number of correct statements among these conditions is 3.
-/
theorem correct_statements_count :
  (1 ∧ 3 ∧ 4 : ℕ) = 3 :=
sorry

end correct_statements_count_l339_339286


namespace coefficient_x5_in_expansion_l339_339333

theorem coefficient_x5_in_expansion : 
  let f := fun x => (1 + x + x^2)^9 in
  coeff f 5 = 882 :=
by
  -- Skipping the proof with 'sorry'
  sorry

end coefficient_x5_in_expansion_l339_339333


namespace find_k_l339_339957

variables {V : Type*} [AddCommGroup V] [Module ℝ V]

def non_collinear (e1 e2 : V) : Prop :=
  ¬ ∃ (m : ℝ), e1 = m • e2

variables (e1 e2 : V) (h_non_collinear : non_collinear e1 e2)
variables (a : V) (b : V)
variables (k : ℝ)

-- Given conditions
def vector_a := 2 • e1 + 3 • e2

def vector_b := k • e1 - 4 • e2

def parallel (v1 v2 : V) : Prop :=
  ∃ (m : ℝ), v1 = m • v2

axiom parallel_ab : parallel vector_a vector_b

-- To be proved
theorem find_k (e1 e2 : V) (h_non_collinear : non_collinear e1 e2)
  (h_a : a = vector_a) (h_b : b = vector_b) (h_parallel : parallel a b) : k = -8 / 3 :=
by sorry

end find_k_l339_339957


namespace car_trip_cost_proof_l339_339406

def car_trip_cost 
  (d1 d2 d3 d4 : ℕ) 
  (efficiency : ℕ) 
  (cost_per_gallon : ℕ) 
  (total_distance : ℕ) 
  (gallons_used : ℕ) 
  (cost : ℕ) : Prop :=
  d1 = 8 ∧
  d2 = 6 ∧
  d3 = 12 ∧
  d4 = 2 * d3 ∧
  efficiency = 25 ∧
  cost_per_gallon = 250 ∧
  total_distance = d1 + d2 + d3 + d4 ∧
  gallons_used = total_distance / efficiency ∧
  cost = gallons_used * cost_per_gallon ∧
  cost = 500

theorem car_trip_cost_proof : car_trip_cost 8 6 12 (2 * 12) 25 250 (8 + 6 + 12 + (2 * 12)) ((8 + 6 + 12 + (2 * 12)) / 25) (((8 + 6 + 12 + (2 * 12)) / 25) * 250) :=
by 
  sorry

end car_trip_cost_proof_l339_339406


namespace maximum_utilization_rate_80_l339_339845

noncomputable def maximum_utilization_rate (side_length : ℝ) (AF : ℝ) (BF : ℝ) : ℝ :=
  let area_square := side_length * side_length
  let length_rectangle := side_length
  let width_rectangle := AF / 2
  let area_rectangle := length_rectangle * width_rectangle
  (area_rectangle / area_square) * 100

theorem maximum_utilization_rate_80:
  maximum_utilization_rate 4 2 1 = 80 := by
  sorry

end maximum_utilization_rate_80_l339_339845


namespace angle_measure_l339_339301

theorem angle_measure (x : ℝ) (h1 : (180 - x) = 3*x - 2) : x = 45.5 :=
by
  sorry

end angle_measure_l339_339301


namespace BI_is_correct_length_l339_339303

noncomputable def BI_length : ℝ := 6 - 3 * Real.sqrt 2

theorem BI_is_correct_length (A B C I : Type) 
  [IsTriangle A B C]
  (isosceles_right : IsIsoscelesRightTriangle A B C)
  (AB_eq_6 : dist A B = 6)
  (AC_eq_6 : dist A C = 6)
  (BAC_eq_90 : angle A B C = π / 2)
  (I_is_incenter : IsIncenter I A B C) :
  dist B I = BI_length := 
sorry

end BI_is_correct_length_l339_339303


namespace solution_l339_339256

noncomputable def problem (x : ℝ) : Prop :=
  2021 * (x ^ (2020/202)) - 1 = 2020 * x

theorem solution (x : ℝ) (hx : x ≥ 0) : problem x → x = 1 := 
begin
  sorry
end

end solution_l339_339256


namespace smallest_k_divides_l339_339046

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l339_339046


namespace inequality_holds_l339_339159

theorem inequality_holds (a b : ℝ) (h : a < b) (h₀ : b < 0) : - (1 / a) < - (1 / b) :=
sorry

end inequality_holds_l339_339159


namespace tickets_count_l339_339659

theorem tickets_count (x y: ℕ) (h : 3 * x + 5 * y = 78) : 
  ∃ n : ℕ , n = 6 :=
sorry

end tickets_count_l339_339659


namespace incorrect_expression_l339_339339

theorem incorrect_expression :
  ¬ ((+6) - (-6) = 0) := by
  sorry

end incorrect_expression_l339_339339


namespace Locus_of_M_l339_339381

theorem Locus_of_M 
(M A B C A' B' C' : ℝ)
(h_inside_triangle : ∀ (x : ℝ), x ∈ triangle ABC)
(h_projections : A' = foot_of_perpendicular M B C ∧ B' = foot_of_perpendicular M C A ∧ C' = foot_of_perpendicular M A B)
(h_condition : MA * MA' = MB * MB' ∧ MB * MB' = MC * MC') :
(M ∈ circumcircle ABC ∨ M = orthocenter ABC) :=
sorry

end Locus_of_M_l339_339381


namespace integer_count_between_sqrt8_and_sqrt75_l339_339147

theorem integer_count_between_sqrt8_and_sqrt75 :
  let least_integer_greater_than_sqrt8 := 3
      greatest_integer_less_than_sqrt75 := 8
  in (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6 :=
by
  let least_integer_greater_than_sqrt8 := 3
  let greatest_integer_less_than_sqrt75 := 8
  show (greatest_integer_less_than_sqrt75 - least_integer_greater_than_sqrt8 + 1) = 6
  sorry

end integer_count_between_sqrt8_and_sqrt75_l339_339147


namespace distance_skew_A1C1_B1E_l339_339913

def point : Type := (ℝ × ℝ × ℝ)
def vector : Type := point

def A1 : point := (2, 0, 2)
def C1 : point := (0, 2, 2)
def B1 : point := (2, 2, 2)
def E : point := (0, 1, 0)

def distance_between_skew_lines (A1 C1 B1 E : point) : ℝ :=
  -- distance calculation here

theorem distance_skew_A1C1_B1E : distance_between_skew_lines A1 C1 B1 E = (4 * Real.sqrt 17) / 17 := 
  sorry

end distance_skew_A1C1_B1E_l339_339913


namespace smallest_k_l339_339017

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end smallest_k_l339_339017


namespace sum_of_first_10_common_elements_correct_l339_339883

noncomputable def sum_of_first_10_common_elements : ℕ :=
  let ap := {n | ∃ m : ℕ, n = 4 + 3 * m}
  let gp := {n | ∃ k : ℕ, n = 10 * 2^k}
  let common_elements := {n | n ∈ ap ∧ n ∈ gp}
  let common_elements_list := List.filter (λ n, n ∈ common_elements) [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
  List.sum (List.take 10 common_elements_list)

theorem sum_of_first_10_common_elements_correct : 
  sum_of_first_10_common_elements = 3495250 :=
sorry

end sum_of_first_10_common_elements_correct_l339_339883


namespace value_of_a2016_l339_339570

-- Define the sequence according to the conditions given in the problem
noncomputable def a : ℕ → ℝ
| 1 := real.sqrt 3
| (n+1) := (a n).floor + 1 / (a n - (a n).floor)

-- The main theorem we need to prove, stating the value of a_{2016}
theorem value_of_a2016 : a 2016 = 3023 + (real.sqrt 3 - 1) / 2 :=
sorry

end value_of_a2016_l339_339570


namespace solution_l339_339255

noncomputable def problem (x : ℝ) : Prop :=
  2021 * (x ^ (2020/202)) - 1 = 2020 * x

theorem solution (x : ℝ) (hx : x ≥ 0) : problem x → x = 1 := 
begin
  sorry
end

end solution_l339_339255


namespace factorial_division_identity_l339_339501

theorem factorial_division_identity :
  (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_identity_l339_339501


namespace calculate_g_inv_g_inv_10_l339_339739

-- Define the function g and its inverse g_inv
def g (x : ℝ) : ℝ := x^2 + 2 * x + 1
def g_inv (x : ℝ) : ℝ := real.sqrt x - 1

-- The proof statement
theorem calculate_g_inv_g_inv_10 : g_inv (g_inv 10) = real.sqrt (real.sqrt 10 - 1) - 1 :=
by
  sorry

end calculate_g_inv_g_inv_10_l339_339739


namespace min_max_values_of_quadratic_l339_339609

noncomputable def min_value (a b: ℝ) : ℝ := - 7 / 2
noncomputable def max_value (a b: ℝ) : ℝ := 5 + 4 * Real.sqrt 5

theorem min_max_values_of_quadratic (a b: ℝ) (h1: (a + 1) ^ 2 + (b - 2) ^ 2 ≤ 4) (h2: a + b + 1 ≥ 0) :
  min_value a b = -7 / 2 ∧ max_value a b = 5 + 4 * Real.sqrt 5 := by
  sorry

end min_max_values_of_quadratic_l339_339609


namespace problem_statement_l339_339064

noncomputable def z1 : ℂ := (1 + complex.I) / complex.I
def z2 (x y : ℝ) : ℂ := x + y * complex.I

def conjugate_z1 : Prop := complex.conj z1 = 1 - complex.I
def purely_imaginary_z2 (x : ℝ) (y : ℝ) : Prop := x = 0 → (z2 0 y).im = y ∧ (z2 0 y).re = 0
def parallel_vectors (x y : ℝ) : Prop := (1, -1) = (x, y) → x + y = 0
def perpendicular_vectors (x y : ℝ) : Prop := (1 * x + (-1) * y = 0) → complex.abs (z1 + z2 x y) = complex.abs (z1 - z2 x y)

theorem problem_statement (x y : ℝ) :
  ¬ conjugate_z1 ∧ purely_imaginary_z2 x y ∧ parallel_vectors x y ∧ perpendicular_vectors x y :=
by
  have h1 : z1 = 1 - complex.I := by -- proof omitted
    sorry
  have h2 : conjugate_z1 := by
    sorry
  have h3 : purely_imaginary_z2 x y := by
    sorry
  have h4 : parallel_vectors x y := by
    sorry
  have h5 : perpendicular_vectors x y := by
    sorry
  exact ⟨h2, h3, h4, h5⟩

end problem_statement_l339_339064


namespace total_cost_of_replacing_floor_l339_339775

-- Dimensions of the first rectangular section
def length1 : ℕ := 8
def width1 : ℕ := 7

-- Dimensions of the second rectangular section
def length2 : ℕ := 6
def width2 : ℕ := 4

-- Cost to remove the old flooring
def cost_removal : ℕ := 50

-- Cost of new flooring per square foot
def cost_per_sqft : ℝ := 1.25

-- Total cost to replace the floor in both sections of the L-shaped room
theorem total_cost_of_replacing_floor 
  (A1 : ℕ := length1 * width1)
  (A2 : ℕ := length2 * width2)
  (total_area : ℕ := A1 + A2)
  (cost_flooring : ℝ := total_area * cost_per_sqft)
  : cost_removal + cost_flooring = 150 :=
sorry

end total_cost_of_replacing_floor_l339_339775


namespace trigonometric_identity_l339_339904

open Real

theorem trigonometric_identity (α : ℝ) (h : sin (α - (π / 12)) = 1 / 3) :
  cos (α + (17 * π / 12)) = 1 / 3 :=
sorry

end trigonometric_identity_l339_339904


namespace factorial_division_l339_339410

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339410


namespace certain_fraction_ratio_l339_339003

theorem certain_fraction_ratio :
  (∃ (x y : ℚ), (x / y) / (6 / 5) = (2 / 5) / 0.14285714285714288) →
  (∃ (x y : ℚ), x / y = 84 / 25) := 
  by
    intros h_ratio
    have h_rat := h_ratio
    sorry

end certain_fraction_ratio_l339_339003


namespace factorial_expression_l339_339443

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339443


namespace octagon_covered_polygon_covered_l339_339523

-- Define the properties of a rhombus and a regular polygon

structure Rhombus where
  side_length : ℝ
  angle1 : ℝ
  angle2 : ℝ
  angle_sum : angle1 + angle2 = 180

structure RegularPolygon where
  side_length : ℝ
  num_sides : ℕ
  interior_angle : ℝ
  angle_relation : interior_angle = (num_sides - 2) * 180 / num_sides

-- Define covering of a polygon with rhombuses
def covers (polygon : RegularPolygon) (rhombuses : list Rhombus) : Prop :=
  sorry

-- Octagon coverage problem
theorem octagon_covered :
  ∀ (c : ℝ),
  let octagon : RegularPolygon := {
    side_length := c,
    num_sides := 8,
    interior_angle := 135,
    angle_relation := by norm_num
  }
  let rhombus45 : Rhombus := {
    side_length := c,
    angle1 := 45,
    angle2 := 135,
    angle_sum := by norm_num
  }
  let rhombus90 : Rhombus := {
    side_length := c,
    angle1 := 90,
    angle2 := 90,
    angle_sum := by norm_num
  }
  covers octagon [rhombus45, rhombus45, rhombus45, rhombus45, rhombus90, rhombus90] := sorry

-- Generalizing for other polygons
theorem polygon_covered :
  ∀ (c : ℝ),
  ∀ (n : ℕ),
  ∃ (rhombi_set : list Rhombus),
  let polygon : RegularPolygon := {
    side_length := c,
    num_sides := n,
    interior_angle := (n - 2) * 180 / n,
    angle_relation := by norm_num
  }
  covers polygon rhombi_set := sorry

end octagon_covered_polygon_covered_l339_339523


namespace pq_bisects_ef_iff_parallel_bc_ad_l339_339193

variables (A B C D P Q E F : Point) (O1 O2 : Circle)

-- Conditions
def quadrilateral (A B C D : Point) : Prop :=
  convex (polygon_A_B_C_D)

def circles (O1 O2 : Circle) : Prop :=
  O1 ↔ passes_through A ∧ passes_through B ∧ tangent_to (line_segment C D P) ∧
  O2 ↔ passes_through C ∧ passes_through D ∧ tangent_to (line_segment A B Q)

-- Statement
theorem pq_bisects_ef_iff_parallel_bc_ad :
  (EF bisects PQ) ↔ (is_parallel (line_segment B C) (line_segment A D)) :=
sorry

end pq_bisects_ef_iff_parallel_bc_ad_l339_339193


namespace find_S15_l339_339667

noncomputable theory

-- Definitions and conditions
variable (S : ℕ → ℝ)
axiom geom_seq (n : ℕ) : S n = a * (1 - r^n) / (1 - r)

-- Given conditions
axiom S_5_eq_10 : S 5 = 10
axiom S_10_eq_50 : S 10 = 50

-- Proof goal
theorem find_S15 : S 15 = 210 := by sorry

end find_S15_l339_339667


namespace cannot_determine_substance_l339_339540

def mass_percentage_of_O (substance : Type) [HasMass substance] : ℝ := sorry

def mass_percentage_condition (substance : Type) [HasMass substance] : Prop :=
  mass_percentage_of_O substance = 21.62

theorem cannot_determine_substance (substance : Type) [HasMass substance] 
(h : mass_percentage_condition substance) : 
  ¬ (∃ unique_substance : Type, mass_percentage_of_O unique_substance = 21.62) :=
sorry

end cannot_determine_substance_l339_339540


namespace probability_valid_l339_339225

noncomputable def root_of_unity (k n : ℕ) : ℂ :=
  complex.exp (2 * real.pi * complex.I * k / n)

def valid_k (k : ℕ) : Prop :=
  let theta := 2 * real.pi * k / 2023 in
  complex.re (complex.exp (complex.I * theta)) ≥ (1 + real.sqrt 2) / 2

def count_valid_k : ℕ :=
  (finset.range 2023).filter valid_k.card

def probability_valid_k : ℝ :=
  count_valid_k / 2022

theorem probability_valid :
  probability_valid_k = n / 2022 :=
sorry

end probability_valid_l339_339225


namespace sum_a_b_eq_930_l339_339554

noncomputable def a : ℕ := 30
noncomputable def b : ℕ := 900

theorem sum_a_b_eq_930 : a + b = 930 := by
  have h1 : a = 30 := rfl
  have h2 : b = 900 := rfl
  have h3 : a + b = 30 + 900 := by rw [h1, h2]
  exact calc
    30 + 900 = 930 : by norm_num

end sum_a_b_eq_930_l339_339554


namespace sum_of_first_15_terms_of_arithmetic_sequence_l339_339973

theorem sum_of_first_15_terms_of_arithmetic_sequence 
  (a d : ℕ) 
  (h1 : (5 * (2 * a + 4 * d)) / 2 = 10) 
  (h2 : (10 * (2 * a + 9 * d)) / 2 = 50) :
  (15 * (2 * a + 14 * d)) / 2 = 120 :=
sorry

end sum_of_first_15_terms_of_arithmetic_sequence_l339_339973


namespace determine_a_l339_339942

def f (x : ℝ) : ℝ := 3 * x ^ 2 + 2 * x + 1

theorem determine_a (a : ℝ) (h1 : ∫ x in -1..1, f x = 2 * f a) (h2 : a > 0) : a = 1 / 3 :=
by
  sorry

end determine_a_l339_339942


namespace concurrent_lines_l339_339729

-- Condition 1: Definition of rectangles outside triangle ABC
variables (A B C A1 A2 B1 B2 C1 C2 : Type)
variable (Triangle_ABC : ∃ (A B C : Point), acute_triangle A B C)

-- Condition 2: Sum of angles equals 180 degrees
variable (angle_condition : ∠ B C1 C + ∠ C A1 A + ∠ A B1 B = 180)

-- Theorem statement
theorem concurrent_lines
  (Rectangles_outside: (B C C1 B2).rectangle ∧ (C  A A1 C2).rectangle ∧ (A B B1 A2).rectangle)
  (angle_sum_condition : angle_condition):
  concurrent B1 C2 C1 A2 A1 B2 := sorry

end concurrent_lines_l339_339729


namespace find_length_PF_l339_339996

-- Definitions of points and segments
variable {P Q R M L F : Type}

-- Coordinates and lengths
variable [metric_space P Q R]
variable [metric_space PR]
variable [metric_space PQ]
variable [metric_space RM]

-- Right angle at P condition
axiom right_angle_P : ∀(P Q R : triangle), PQR.is_right_angle_at P

-- Given Lengths
axiom PQ_length : PQ = 3
axiom PR_length : PR = 3 * real.sqrt 3

-- Altitude PL and Median RM intersect at F
axiom altitude_PL : is_altitude P L R
axiom median_RM : is_median R M Q P
axiom intersection_F : ∃F, intersection PL RM = F

-- Midpoint M of QR
axiom midpoint_M : midpoint Q R = M

-- Prove PF == 0.857 * sqrt 3
theorem find_length_PF : 
  let PL := (PQ * PR) / (PQ^2 + PR^2).sqrt in
  let PM := QR / 2 in
  let RL := PR^2 / QR in
  let LQ := QR - RL in
  let RX := QR - QX in
  let MX := 1/2 * PL in
  let FL := (MX * RL) / RX in
  PL - FL = 0.857 * real.sqrt 3 := 
sorry

end find_length_PF_l339_339996


namespace factorial_division_l339_339412

theorem factorial_division (n : ℕ) (hn : n = 7) : (8! + 9!) / n! = 80 :=
by
  sorry

end factorial_division_l339_339412


namespace geologists_prob_correct_l339_339198

def geologists_dist_probability : Real :=
  let num_paths := 8
  let speed := 4
  let time := 1
  let distance_threshold := 6
  let total_distance (angle_deg : Real) : Real := 
    Real.sqrt (2 * (speed * time)^2 - 2 * (speed * time)^2 * Real.cos (angle_deg * Real.pi / 180))
  let distances := List.map total_distance [45, 90, 135]
  let favorable_paths := distances.count (λ d, d > distance_threshold) * num_paths
  favorable_paths / (num_paths * num_paths)

theorem geologists_prob_correct :
  geologists_dist_probability = 0.375 :=
by
  sorry

end geologists_prob_correct_l339_339198


namespace freq_count_of_third_group_l339_339937

theorem freq_count_of_third_group
  (sample_size : ℕ) 
  (freq_third_group : ℝ) 
  (h1 : sample_size = 100) 
  (h2 : freq_third_group = 0.2) : 
  (sample_size * freq_third_group) = 20 :=
by 
  sorry

end freq_count_of_third_group_l339_339937


namespace determinant_is_zero_l339_339526

-- Define the matrix
def my_matrix (x y z : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := 
  ![![1, x + z, y - z],
    ![1, x + y + z, y - z],
    ![1, x + z, x + y]]

-- Define the property to prove
theorem determinant_is_zero (x y z : ℝ) :
  Matrix.det (my_matrix x y z) = 0 :=
by sorry

end determinant_is_zero_l339_339526


namespace smallest_number_l339_339391

-- Definitions for the numbers in different bases
def num1 : ℕ := nat.of_digits 2 [1, 0, 1, 0, 1, 1]  -- 101011 in base 2
def num2 : ℕ := nat.of_digits 3 [1, 2, 1, 0]  -- 1210 in base 3
def num3 : ℕ := nat.of_digits 8 [1, 1, 0]  -- 110 in base 8
def num4 : ℕ := nat.of_digits 12 [6, 8]  -- 68 in base 12

-- Statement to prove
theorem smallest_number : num2 < num1 ∧ num2 < num3 ∧ num2 < num4 := by
  sorry  -- Proof not required

end smallest_number_l339_339391


namespace ordered_triples_eq_l339_339006

theorem ordered_triples_eq :
  ∃! (x y z : ℤ), x + y = 4 ∧ xy - z^2 = 3 ∧ (x = 2 ∧ y = 2 ∧ z = 0) :=
by
  -- Proof goes here
  sorry

end ordered_triples_eq_l339_339006


namespace number_of_integers_between_sqrt8_and_sqrt75_l339_339129

theorem number_of_integers_between_sqrt8_and_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ 
  ∃ x : Fin n.succ, √8 < ↑x ∧ ↑x < √75 := 
by
  sorry

end number_of_integers_between_sqrt8_and_sqrt75_l339_339129


namespace halving_function_range_t_l339_339701

noncomputable def is_halving_function (f : ℝ → ℝ) (D : set ℝ) :=
  ∃ (a b : ℝ), a ≤ b ∧ Icc a b ⊆ D ∧ (set.range (λ x, f x) ∩ Icc a b) = Icc (a / 2) (b / 2)

theorem halving_function_range_t (D : set ℝ) : 
  (is_halving_function (λ x, real.log (2^x + t) / real.log 2) D) → 
  0 < t ∧ t < 1 / 4 :=
begin
  sorry
end

end halving_function_range_t_l339_339701


namespace find_ff_neg2_l339_339945

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^x else Real.log x / Real.log 2

theorem find_ff_neg2 : f (f (-2)) = -2 := by
  sorry

end find_ff_neg2_l339_339945


namespace sum_of_primitive_roots_mod_11_l339_339566

/-- Define a predicate for a primitive root modulo a prime -/
def is_primitive_root_mod (a p : ℕ) : Prop :=
  p.prime ∧ ∀ b : ℕ, b ∈ (list.range p.pred).map (λ i, (a^i % p)) ↔ b ∈ list.range p

/-- We want the sum of all integers in the set {1, 2, ..., 10} that are primitive roots modulo 11 to be 9 -/
theorem sum_of_primitive_roots_mod_11 :
  let p := 11,
      set_of_primitive_roots := {a : ℕ | is_primitive_root_mod a p ∧ a ∈ finset.range p} in 
  finset.sum finset.univ (λ a, if a ∈ set_of_primitive_roots then a else 0) = 9 := 
by
  sorry

end sum_of_primitive_roots_mod_11_l339_339566


namespace tau_mn_lt_tau_m_tau_n_sigma_mn_lt_sigma_m_sigma_n_l339_339575

theorem tau_mn_lt_tau_m_tau_n {p α β : ℕ} (hp : Nat.Prime p) :
    (Nat.tau (p ^ (α + β)) = α + β + 1) ∧
    (Nat.tau (p ^ α) * Nat.tau (p ^ β) = (α + 1) * (β + 1)) ∧
    (Nat.tau (p ^ (α + β)) < Nat.tau (p ^ α) * Nat.tau (p ^ β)) :=
    by 
      sorry

theorem sigma_mn_lt_sigma_m_sigma_n {p α β : ℕ} (hp : Nat.Prime p) :
    (Nat.sigma (p ^ α) = (p ^ (α + 1) - 1) / (p - 1)) ∧
    (Nat.sigma (p ^ β) = (p ^ (β + 1) - 1) / (p - 1)) ∧
    (Nat.sigma (p ^ (α + β)) = (p ^ (α + β + 1) - 1) / (p - 1)) ∧
    (Nat.sigma (p ^ α) * Nat.sigma (p ^ β) = ((p ^ (α + 1) - 1) * (p ^ (β + 1) - 1)) / (p - 1) ^ 2) ∧
    (Nat.sigma (p ^ (α + β)) < Nat.sigma (p ^ α) * Nat.sigma (p ^ β)) :=
    by 
      sorry

end tau_mn_lt_tau_m_tau_n_sigma_mn_lt_sigma_m_sigma_n_l339_339575


namespace factorial_expression_l339_339447

theorem factorial_expression : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_expression_l339_339447


namespace factorial_division_l339_339485

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339485


namespace expand_expression_l339_339527

theorem expand_expression (x y : ℝ) : 5 * (4 * x^3 - 3 * x * y + 7) = 20 * x^3 - 15 * x * y + 35 := 
sorry

end expand_expression_l339_339527


namespace crayons_left_l339_339703

def initial_green_crayons : ℝ := 5
def initial_blue_crayons : ℝ := 8
def initial_yellow_crayons : ℝ := 7
def given_green_crayons : ℝ := 3.5
def given_blue_crayons : ℝ := 1.25
def given_yellow_crayons : ℝ := 2.75
def broken_yellow_crayons : ℝ := 0.5

theorem crayons_left (initial_green_crayons initial_blue_crayons initial_yellow_crayons given_green_crayons given_blue_crayons given_yellow_crayons broken_yellow_crayons : ℝ) :
  initial_green_crayons - given_green_crayons + 
  initial_blue_crayons - given_blue_crayons + 
  initial_yellow_crayons - given_yellow_crayons - broken_yellow_crayons = 12 :=
by
  sorry

end crayons_left_l339_339703


namespace correct_statement_count_l339_339284

theorem correct_statement_count :
  let s1 := (∀ (x : ℝ), is_pi x → -x = -π)
  let s2 := (∀ (x y : ℝ), has_opposite_sign x y → is_opposite x y)
  let s3 := (∀ (x : ℝ), is_negative x → -x = 3.8)
  let s4 := (∃ (x : ℝ), x = -x)
  let s5 := (∀ (x y : ℝ), is_positive_negative x y → is_opposite x y)
in s1 ∧ s3 ∧ s4 ∧ ¬s2 ∧ ¬s5 → option_correct = 3 :=
begin
  intro,
  -- The proof is omitted as only statement formulation is required.
  sorry
end

end correct_statement_count_l339_339284


namespace factorial_sum_division_l339_339457

theorem factorial_sum_division : (8! + 9!) / 7! = 80 := by
  sorry

end factorial_sum_division_l339_339457


namespace new_nation_connected_l339_339813

   /--
   Given 1001 cities with each city having exactly 500 outgoing and 500 incoming roads. If a new nation is formed with 668 cities from the original 1001 cities,
   prove that any city in the new nation can reach any other city in this new nation without leaving its borders.
   -/
   theorem new_nation_connected (cities : Finset ℕ) (total : cities.card = 1001)
     (outgoing : ∀ c ∈ cities, (Finset.filter (λ d, road.connects c d) cities).card = 500)
     (incoming : ∀ c ∈ cities, (Finset.filter (λ d, road.connects d c) cities).card = 500)
     (new_nation : Finset ℕ) (new_nation_card : new_nation.card = 668)
     : ∀ c1 c2 ∈ new_nation, (∃ p : Path c1 c2, p ∈ new_nation) := sorry
   
end new_nation_connected_l339_339813


namespace num_valid_N_l339_339396

def sum_in_base_10 (a1 a2 a3 a4 a5 a6 b1 b2 b3 b4: ℕ) : ℕ :=
  243 * a1 + 81 * a2 + 27 * a3 + 9 * a4 + 3 * a5 + a6 + 343 * b1 + 49 * b2 + 7 * b3 + b4

def valid_N (N S : ℕ) : Prop :=
  2 * N % 1000 = S % 1000

theorem num_valid_N : ∃ n, n = 30 ∧ 
  ∀ (N : ℕ), 
    (100 ≤ N ∧ N < 1000) → 
    (∃ (a1 a2 a3 a4 a5 a6 b1 b2 b3 b4 : ℕ), 
      0 ≤ a1 ∧ a1 < 3 ∧ 0 ≤ a2 ∧ a2 < 3 ∧ 0 ≤ a3 ∧ a3 < 3 ∧ 0 ≤ a4 ∧ a4 < 3 ∧ 
      0 ≤ a5 ∧ a5 < 3 ∧ 0 ≤ a6 ∧ a6 < 3 ∧ 0 ≤ b1 ∧ b1 < 7 ∧ 0 ≤ b2 ∧ b2 < 7 ∧ 
      0 ≤ b3 ∧ b3 < 7 ∧ 0 ≤ b4 ∧ b4 < 7 ∧ 
      N = 3^5 * a1 + 3^4 * a2 + 3^3 * a3 + 3^2 * a4 + 3^1 * a5 + 3^0 * a6 ∧
      N = 7^3 * b1 + 7^2 * b2 + 7^1 * b3 + 7^0 * b4 ∧
      valid_N N (sum_in_base_10 a1 a2 a3 a4 a5 a6 b1 b2 b3 b4)) :=
begin
  sorry
end

end num_valid_N_l339_339396


namespace average_transformation_l339_339919

theorem average_transformation (a_1 a_2 a_3 a_4 a_5 : ℝ)
  (h_avg : (a_1 + a_2 + a_3 + a_4 + a_5) / 5 = 8) : 
  ((a_1 + 10) + (a_2 - 10) + (a_3 + 10) + (a_4 - 10) + (a_5 + 10)) / 5 = 10 := 
by
  sorry

end average_transformation_l339_339919


namespace rectangle_area_l339_339817

-- Define the conditions
def circle_tangent_to_adjacent_sides (r : ℝ) (rectangle : ℝ × ℝ) : Prop :=
  rectangle.1 = 2 * r ∧ rectangle.2 = r

def circle_passes_through_midpoint (r : ℝ) (rectangle : ℝ × ℝ) : Prop :=
  let d := Math.sqrt (rectangle.1 ^ 2 + rectangle.2 ^ 2) in
  let m := (d / 2) in
  m = Math.sqrt (r^2 + (rectangle.1 / 2) ^ 2)

-- Problem statement
theorem rectangle_area (r : ℝ) (rectangle : ℝ × ℝ) 
  (h_tangent : circle_tangent_to_adjacent_sides r rectangle)
  (h_midpoint : circle_passes_through_midpoint r rectangle) :
  rectangle.1 * rectangle.2 = 2 * r^2 := sorry

end rectangle_area_l339_339817


namespace heptagons_concurrent_l339_339610

-- Define the problem context
def regular_heptagon (A : Point) (B : List Point) : Prop :=
  ∃ (σ : Permutation B), 
    (∀ i, σ (σ (σ (σ (σ (σ (σ i)))))) = i) ∧
    (∀ i, Distance A (B i) = Distance A (B (i+1)))

def is_inscribed {A B C : Point} (k : Circle) (A B1 B2 B3 B4 B5 B6 : Point) : Prop :=
  inscribed k [A, B1, B2, B3, B4, B5, B6]

def circles_pass_through (A : Point) (k1 k2 : Circle) : Prop :=
  member A k1 ∧ member A k2

theorem heptagons_concurrent :
  ∀ (A : Point) (B1 B2 B3 B4 B5 B6 : Point) (C1 C2 C3 C4 C5 C6 : Point)
    (kB kC : Circle), 
  regular_heptagon A [B1, B2, B3, B4, B5, B6] →
  regular_heptagon A [C1, C2, C3, C4, C5, C6] →
  is_inscribed kB A B1 B2 B3 B4 B5 B6 →
  is_inscribed kC A C1 C2 C3 C4 C5 C6 →
  circles_pass_through A kB kC →
  ∃ M : Point, (M = A ∨ (forall i, collinear [M, point_list_i [B1, B2, ..., B6] i, point_list_i [C1, C2, ..., C6] i])) :=
sorry

end heptagons_concurrent_l339_339610


namespace vector_AD_length_l339_339603

noncomputable def f (x : ℝ) : ℝ := 4 * (Real.sin x) * (Real.cos (x - Real.pi / 6))

theorem vector_AD_length (A B C D : ℝ) (a b c : ℝ) (h1 : c = 2 * b) (h2 : c = 4) (h3 : D = (B + C) / 2)
  (h4 : ∀ x : ℝ, 0 ≤ f(x) ∧ f(x) ≤ f(A))
  (h5 : A = Real.pi / 3) :
  |A + D| = Real.sqrt 7 :=
by {
  -- The required proof needs to be done here.
  sorry
}

end vector_AD_length_l339_339603


namespace sum_first_2n_terms_l339_339656

-- Definitions from conditions
def a₁ : ℤ := -1
def b (n : ℕ) (a : ℕ → ℤ) : ℚ := (1 / 2) ^ a n
def b_product_condition (a : ℕ → ℤ) : Prop := (b 1 a) * (b 2 a) * (b 3 a) = 1 / 64
def a_general_term (n : ℕ) : ℤ := 3 * n - 4
def c (n : ℕ) : ℤ := (-1)^n * a_general_term n
def T (n : ℕ) : ℤ := ∑ i in finset.range (2 * n), c (i + 1)

-- Proof goal
theorem sum_first_2n_terms (n : ℕ) (a : ℤ) (h₁ : a = a₁)
  (h₂ : b_product_condition a_general_term) :
  T n = 3 * n :=
by
  -- Omitted proof
  sorry

end sum_first_2n_terms_l339_339656


namespace contrapositive_tan_l339_339744

variable (α : ℝ)

theorem contrapositive_tan (h : α = π / 4 → tan α = 1) : tan α ≠ 1 → α ≠ π / 4 :=
sorry

end contrapositive_tan_l339_339744


namespace smallest_k_for_divisibility_l339_339028

theorem smallest_k_for_divisibility (z : ℂ) (hz : z^7 = 1) : ∃ k : ℕ, (∀ m : ℕ, z ^ (m * k) = 1) ∧ k = 84 :=
sorry

end smallest_k_for_divisibility_l339_339028


namespace intersection_of_circumspheres_l339_339803

-- Defining the necessary structures:
variables {point : Type*} [inhabited point]

structure Tetrahedron (point : Type*) :=
(A B C D : point)

structure Plane (point : Type*) :=
(thru : point)
(perp_to : point)

def circumsphere (T : Tetrahedron point) : set point := sorry -- Assume the definition of circumsphere

-- Given conditions as Lean statements:
variables (A1 A2 A3 : point)
variables (B1 B2 B3 C1 C2 C3 D1 D2 D3 : point)
variables (E : point)
variables (l : set point)
variables (alpha1 alpha2 alpha3 beta1 beta2 beta3 gamma1 gamma2 gamma3 : Plane point)

-- Additional assumptions based on the conditions:
axiom planes_intersect_at_E : ∀ i, (alpha1.thru = B1 ∧ alpha2.thru = B2 ∧ alpha3.thru = B3) ∧ 
                                  (beta1.thru = C1 ∧ beta2.thru = C2 ∧ beta3.thru = C3) ∧
                                  (gamma1.thru = D1 ∧ gamma2.thru = D2 ∧ gamma3.thru = D3) ∧
                                  (alpha1.perp_to = A1 ∧ beta1.perp_to = A1 ∧ gamma1.perp_to = A1) ∧
                                  (alpha2.perp_to = A2 ∧ beta2.perp_to = A2 ∧ gamma2.perp_to = A2) ∧
                                  (alpha3.perp_to = A3 ∧ beta3.perp_to = A3 ∧ gamma3.perp_to = A3) ∧
                                  (∀ (p : point), p ∈ circumsphere ⟨A1, B1, C1, D1⟩ ∧
                                                  p ∈ circumsphere ⟨A2, B2, C2, D2⟩ ∧
                                                  p ∈ circumsphere ⟨A3, B3, C3, D3⟩ → p = E) 

axiom line_l_incident : ∀ i, A1 = A2 ∧ A2 = A3 ∨ (A1 ≠ A2 ∧ A2 ≠ A3 ∧ A3 ≠ A1 ∧ ∀ (p : point), p ∈ l → (E ∈ l ∨ E ∉ l))

-- Theorem statement:
theorem intersection_of_circumspheres :
  if A1 = A2 ∧ A2 = A3 then ∀ (p : point), p ∈ circumsphere ⟨A1, B1, C1, D1⟩ ∧
                                        p ∈ circumsphere ⟨A2, B2, C2, D2⟩ ∧
                                        p ∈ circumsphere ⟨A3, B3, C3, D3⟩ → p ∈ circumsphere ⟨A1, B1, C1, D1⟩
  else if E ∈ l then ∀ (p : point), p ∈ circumsphere ⟨A1, B1, C1, D1⟩ ∧
                                    p ∈ circumsphere ⟨A2, B2, C2, D2⟩ ∧
                                    p ∈ circumsphere ⟨A3, B3, C3, D3⟩ → p = E
  else ∃ center radius, ∀ (p : point), (dist p center) = radius → p ∈ circumsphere ⟨A1, B1, C1, D1⟩ ∧
                                        p ∈ circumsphere ⟨A2, B2, C2, D2⟩ ∧
                                        p ∈ circumsphere ⟨A3, B3, C3, D3⟩ :=
sorry  -- Proof to be constructed

end intersection_of_circumspheres_l339_339803


namespace max_d_value_l339_339361

theorem max_d_value (stones : list ℝ) :
  (∀ w ∈ stones, 0 < w ∧ w ≤ 2) →
  stones.sum = 100 →
  ∃ d > 0, (∀ (selected : list ℝ), selected ⊆ stones →
    |selected.sum - 10| ≥ d) ∧ d = 10 / 11 := 
by
  /- Proof goes here -/
  sorry

end max_d_value_l339_339361


namespace smallest_k_divides_l339_339048

-- Declare the polynomial p(z)
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

-- Proposition: The smallest positive integer k such that p(z) divides z^k - 1 is 120
theorem smallest_k_divides (k : ℕ) (h : ∀ z : ℂ, p z ∣ z^k - 1) : k = 120 :=
sorry

end smallest_k_divides_l339_339048


namespace problem_statement_l339_339065

noncomputable def z1 : ℂ := (1 + complex.I) / complex.I
def z2 (x y : ℝ) : ℂ := x + y * complex.I

def conjugate_z1 : Prop := complex.conj z1 = 1 - complex.I
def purely_imaginary_z2 (x : ℝ) (y : ℝ) : Prop := x = 0 → (z2 0 y).im = y ∧ (z2 0 y).re = 0
def parallel_vectors (x y : ℝ) : Prop := (1, -1) = (x, y) → x + y = 0
def perpendicular_vectors (x y : ℝ) : Prop := (1 * x + (-1) * y = 0) → complex.abs (z1 + z2 x y) = complex.abs (z1 - z2 x y)

theorem problem_statement (x y : ℝ) :
  ¬ conjugate_z1 ∧ purely_imaginary_z2 x y ∧ parallel_vectors x y ∧ perpendicular_vectors x y :=
by
  have h1 : z1 = 1 - complex.I := by -- proof omitted
    sorry
  have h2 : conjugate_z1 := by
    sorry
  have h3 : purely_imaginary_z2 x y := by
    sorry
  have h4 : parallel_vectors x y := by
    sorry
  have h5 : perpendicular_vectors x y := by
    sorry
  exact ⟨h2, h3, h4, h5⟩

end problem_statement_l339_339065


namespace symmetry_about_line_symmetry_about_point_interval_shift_l339_339750

noncomputable def f (x : Real) : Real :=
  Real.sin (2 * x) - Real.sqrt 3 * (Real.cos (x) ^ 2 - Real.sin (x) ^ 2)

theorem symmetry_about_line (x : Real) :
  f (Real.pi / 12 * 11 - x) = f (Real.pi / 12 * 11 + x) := sorry

theorem symmetry_about_point :
  f (2 * Real.pi / 3) = 0 := sorry

theorem interval_shift :
  ∀ (x : Real), x ∈ Ioo (-Real.pi / 12) (5 * Real.pi / 12) →
  f (x + Real.pi / 3) = f x := sorry

end symmetry_about_line_symmetry_about_point_interval_shift_l339_339750


namespace number_of_integers_between_sqrt8_and_sqrt75_l339_339128

theorem number_of_integers_between_sqrt8_and_sqrt75 : 
  ∃ n : ℕ, n = 6 ∧ 
  ∃ x : Fin n.succ, √8 < ↑x ∧ ↑x < √75 := 
by
  sorry

end number_of_integers_between_sqrt8_and_sqrt75_l339_339128


namespace student_A_more_stable_than_B_l339_339815

theorem student_A_more_stable_than_B 
    (avg_A : ℝ := 98) (avg_B : ℝ := 98) 
    (var_A : ℝ := 0.2) (var_B : ℝ := 0.8) : 
    var_A < var_B :=
by sorry

end student_A_more_stable_than_B_l339_339815


namespace max_value_of_k_l339_339846

theorem max_value_of_k (m : ℝ) (h₁ : 0 < m) (h₂ : m < 1/2) : 
  (1 / m + 2 / (1 - 2 * m)) ≥ 8 :=
sorry

end max_value_of_k_l339_339846


namespace factorial_division_l339_339480

theorem factorial_division : (8.factorial + 9.factorial) / 7.factorial = 80 := by
  sorry

end factorial_division_l339_339480


namespace johns_photo_world_sitting_fee_l339_339378

variable (J : ℝ)

theorem johns_photo_world_sitting_fee
  (h1 : ∀ n : ℝ, n = 12 → 2.75 * n + J = 1.50 * n + 140) : J = 125 :=
by
  -- We will skip the proof since it is not required by the problem statement.
  sorry

end johns_photo_world_sitting_fee_l339_339378


namespace stamps_max_l339_339640

theorem stamps_max (price_per_stamp : ℕ) (total_cents : ℕ) (h1 : price_per_stamp = 25) (h2 : total_cents = 5000) : 
  ∃ n : ℕ, (n * price_per_stamp ≤ total_cents) ∧ (∀ m : ℕ, (m > n) → (m * price_per_stamp > total_cents)) ∧ n = 200 := 
by
  sorry

end stamps_max_l339_339640


namespace cube_root_nested_l339_339856

theorem cube_root_nested (x : ℝ) (h : x = 8) : 
  Real.cbrt (2 * Real.cbrt (2 * Real.cbrt x)) = 2 ^ (5 / 9) :=
by
  sorry

end cube_root_nested_l339_339856


namespace johns_calorie_intake_l339_339828

theorem johns_calorie_intake
  (servings : ℕ)
  (calories_per_serving : ℕ)
  (total_calories : ℕ)
  (half_package_calories : ℕ)
  (h1 : servings = 3)
  (h2 : calories_per_serving = 120)
  (h3 : total_calories = servings * calories_per_serving)
  (h4 : half_package_calories = total_calories / 2)
  : half_package_calories = 180 :=
by sorry

end johns_calorie_intake_l339_339828


namespace coefficient_sum_eq_l339_339692

noncomputable def a_n (n : ℕ) := (3 - real.sqrt x)^n

theorem coefficient_sum_eq :
  (∑ n in finset.range 2016, a_n n / (3 ^ n)) / (A_2016^3) = 1 / 54 :=
sorry

end coefficient_sum_eq_l339_339692
