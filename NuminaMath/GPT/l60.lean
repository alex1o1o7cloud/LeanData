import Mathlib

namespace boat_travel_times_l60_60114

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l60_60114


namespace minimize_expression_l60_60621

theorem minimize_expression (a b c d : ℝ) (h1 : 1 ≤ a) (h2 : a ≤ b) (h3 : b ≤ c) (h4 : c ≤ d) (h5 : d ≤ 5) :
  (a - 1)^2 + (b / a - 1)^2 + (c / b - 1)^2 + (d / c - 1)^2 + (5 / d - 1)^2 ≥ 5 * (5^(1/5) - 1)^2 :=
by
  sorry

end minimize_expression_l60_60621


namespace part1_part2_part3_l60_60619

-- Part (1): Proving \( p \implies m > \frac{3}{2} \)
theorem part1 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0) → (m > 3 / 2) :=
by
  sorry

-- Part (2): Proving \( q \implies (m < -1 \text{ or } m > 2) \)
theorem part2 (m : ℝ) : (∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → (m < -1 ∨ m > 2) :=
by
  sorry

-- Part (3): Proving \( (p ∨ q) \implies ((-\infty, -1) ∪ (\frac{3}{2}, +\infty)) \)
theorem part3 (m : ℝ) : (∀ x : ℝ, x^2 + 2 * m - 3 > 0 ∨ ∃ x : ℝ, x^2 - 2 * m * x + m + 2 < 0) → ((m < -1) ∨ (3 / 2 < m)) :=
by
  sorry

end part1_part2_part3_l60_60619


namespace smallest_four_digit_number_l60_60396

theorem smallest_four_digit_number :
  ∃ m : ℕ, (1000 ≤ m) ∧ (m < 10000) ∧ (∃ n : ℕ, 21 * m = n^2) ∧ m = 1029 :=
by sorry

end smallest_four_digit_number_l60_60396


namespace fraction_spent_at_arcade_l60_60310

theorem fraction_spent_at_arcade :
  ∃ f : ℝ, 
    (2.25 - (2.25 * f) - ((2.25 - (2.25 * f)) / 3) = 0.60) → 
    f = 3 / 5 :=
by
  sorry

end fraction_spent_at_arcade_l60_60310


namespace time_to_odd_floor_l60_60678

-- Define the number of even-numbered floors
def evenFloors : Nat := 5

-- Define the number of odd-numbered floors
def oddFloors : Nat := 5

-- Define the time to climb one even-numbered floor
def timeEvenFloor : Nat := 15

-- Define the total time to reach the 10th floor
def totalTime : Nat := 120

-- Define the desired time per odd-numbered floor
def timeOddFloor : Nat := 9

-- Formalize the proof statement
theorem time_to_odd_floor : 
  (oddFloors * timeOddFloor = totalTime - (evenFloors * timeEvenFloor)) :=
by
  sorry

end time_to_odd_floor_l60_60678


namespace slant_height_l60_60135

-- Define the variables and conditions
variables (r A : ℝ)
-- Assume the given conditions
def radius := r = 5
def area := A = 60 * Real.pi

-- Statement of the theorem to prove the slant height
theorem slant_height (r A l : ℝ) (h_r : r = 5) (h_A : A = 60 * Real.pi) : l = 12 :=
sorry

end slant_height_l60_60135


namespace travel_time_l60_60117

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l60_60117


namespace max_a_b_c_d_l60_60405

theorem max_a_b_c_d (a c d b : ℤ) (hb : b > 0) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) 
: a + b + c + d = -5 :=
by
  sorry

end max_a_b_c_d_l60_60405


namespace find_integers_in_range_l60_60393

theorem find_integers_in_range :
  ∀ x : ℤ,
  (20 ≤ x ∧ x ≤ 50 ∧ (6 * x + 5) % 10 = 19) ↔
  x = 24 ∨ x = 29 ∨ x = 34 ∨ x = 39 ∨ x = 44 ∨ x = 49 :=
by sorry

end find_integers_in_range_l60_60393


namespace parallelogram_base_length_l60_60526

theorem parallelogram_base_length
  (height : ℝ) (area : ℝ) (base : ℝ) 
  (h1 : height = 18) 
  (h2 : area = 576) 
  (h3 : area = base * height) : 
  base = 32 :=
by
  rw [h1, h2] at h3
  sorry

end parallelogram_base_length_l60_60526


namespace point_Q_and_d_l60_60030

theorem point_Q_and_d :
  ∃ (a b c d : ℝ),
    (∀ x y z : ℝ, (x - 2)^2 + (y - 3)^2 + (z + 4)^2 = (x - a)^2 + (y - b)^2 + (z - c)^2) ∧
    (8 * a - 6 * b + 32 * c = d) ∧ a = 6 ∧ b = 0 ∧ c = 12 ∧ d = 151 :=
by
  existsi 6, 0, 12, 151
  sorry

end point_Q_and_d_l60_60030


namespace smallest_k_l60_60806

theorem smallest_k (k : ℕ) : 
  (∀ x, x ∈ [13, 7, 3, 5] → k % x = 1) ∧ k > 1 → k = 1366 :=
by
  sorry

end smallest_k_l60_60806


namespace probability_exactly_two_primes_correct_l60_60518

open BigOperators

-- Define the set of primes between 1 and 20
def primes : Finset ℕ := {2, 3, 5, 7, 11, 13, 17, 19}

-- Define the probability calculation
noncomputable def probability_of_prime : ℚ := 8 / 20
noncomputable def probability_of_non_prime : ℚ := 1 - probability_of_prime

-- Define the combined probability of exactly two primes and one non-prime
noncomputable def probability_two_primes (n : ℕ) (k : ℕ) : ℚ :=
(probability_of_prime ^ 2) * probability_of_non_prime * (k.choose 2) 

-- Specific case for n = 3, k = 3 (three dice)
noncomputable def probability_exactly_two_primes : ℚ := probability_two_primes 3 3

-- The target probability should match
theorem probability_exactly_two_primes_correct : probability_exactly_two_primes = 36 / 125 := by
  sorry

end probability_exactly_two_primes_correct_l60_60518


namespace tic_tac_toe_tie_fraction_l60_60801

theorem tic_tac_toe_tie_fraction
  (max_wins : ℚ := 4 / 9)
  (zoe_wins : ℚ := 5 / 12) :
  1 - (max_wins + zoe_wins) = 5 / 36 :=
by
  sorry

end tic_tac_toe_tie_fraction_l60_60801


namespace average_age_group_l60_60810

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = 15 * n) (h2 : T + 37 = 17 * (n + 1)) : n = 10 :=
by
  sorry

end average_age_group_l60_60810


namespace cos_identity_of_angle_l60_60273

open Real

theorem cos_identity_of_angle (α : ℝ) :
  sin (π / 6 + α) = sqrt 3 / 3 → cos (π / 3 - α) = sqrt 3 / 3 :=
by
  intro h
  sorry

end cos_identity_of_angle_l60_60273


namespace part_I_part_II_l60_60277

noncomputable def f (x a : ℝ) := 2 * |x - 1| - a
noncomputable def g (x m : ℝ) := - |x + m|

theorem part_I (a : ℝ) : 
  (∀ x : ℝ, g x 3 > -1 ↔ x = -3) :=
by
  sorry

theorem part_II (a : ℝ) (m : ℝ) :
  (∀ x : ℝ, f x a ≥ g x m) ↔ (a < 4) :=
by
  sorry

end part_I_part_II_l60_60277


namespace minimize_squared_distances_l60_60215

variable {P : ℝ}

/-- Points A, B, C, D, E are collinear with distances AB = 3, BC = 3, CD = 5, and DE = 7 -/
def collinear_points : Prop :=
  ∀ (A B C D E : ℝ), B = A + 3 ∧ C = B + 3 ∧ D = C + 5 ∧ E = D + 7

/-- Define the squared distance function -/
def squared_distances (P A B C D E : ℝ) : ℝ :=
  (P - A)^2 + (P - B)^2 + (P - C)^2 + (P - D)^2 + (P - E)^2

/-- Statement of the proof problem -/
theorem minimize_squared_distances :
  collinear_points →
  ∀ (A B C D E P : ℝ), 
    squared_distances P A B C D E ≥ 181.2 :=
by
  sorry

end minimize_squared_distances_l60_60215


namespace gary_money_left_l60_60710

variable (initialAmount : Nat)
variable (amountSpent : Nat)

theorem gary_money_left (h1 : initialAmount = 73) (h2 : amountSpent = 55) : initialAmount - amountSpent = 18 :=
by
  sorry

end gary_money_left_l60_60710


namespace unknown_rate_of_blankets_l60_60818

theorem unknown_rate_of_blankets (x : ℝ) :
  2 * 100 + 5 * 150 + 2 * x = 9 * 150 → x = 200 :=
by
  sorry

end unknown_rate_of_blankets_l60_60818


namespace cos_540_eq_neg_one_l60_60699

theorem cos_540_eq_neg_one : Real.cos (540 : ℝ) = -1 := by
  sorry

end cos_540_eq_neg_one_l60_60699


namespace twelve_women_reseated_l60_60473

def S (n : ℕ) : ℕ :=
  if n = 0 then 1
  else if n = 1 then 1
  else if n = 2 then 3
  else S (n - 1) + S (n - 2) + S (n - 3)

theorem twelve_women_reseated : S 12 = 1201 :=
by
  sorry

end twelve_women_reseated_l60_60473


namespace expand_product_l60_60848

theorem expand_product (y : ℝ) : 5 * (y - 6) * (y + 9) = 5 * y^2 + 15 * y - 270 := 
by
  sorry

end expand_product_l60_60848


namespace peter_age_l60_60004

variable (x y : ℕ)

theorem peter_age : 
  (x = (3 * y) / 2) ∧ ((4 * y - x) + 2 * y = 54) → x = 18 :=
by
  intro h
  cases h
  sorry

end peter_age_l60_60004


namespace find_a_if_f_even_l60_60542

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l60_60542


namespace point_reflection_y_l60_60432

def coordinates_with_respect_to_y_axis (x y : ℝ) : ℝ × ℝ :=
  (-x, y)

theorem point_reflection_y (x y : ℝ) (h : (x, y) = (-2, 3)) : coordinates_with_respect_to_y_axis x y = (2, 3) := by
  sorry

end point_reflection_y_l60_60432


namespace completing_the_square_l60_60066

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l60_60066


namespace product_formula_l60_60106

theorem product_formula :
  (3 + 5) * (3^2 + 5^2) * (3^4 + 5^4) * (3^8 + 5^8) *
  (3^16 + 5^16) * (3^32 + 5^32) * (3^64 + 5^64) *
  (3^128 + 5^128) = 3^256 - 5^256 := by
  sorry

end product_formula_l60_60106


namespace probability_of_finding_last_defective_product_on_fourth_inspection_l60_60374

theorem probability_of_finding_last_defective_product_on_fourth_inspection :
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  probability = 1 / 5 :=
by
  let total_products := 6
  let qualified_products := 4
  let defective_products := 2
  let probability := (4 / 6) * (3 / 5) * (2 / 4) * (1 / 3) + (4 / 6) * (2 / 5) * (3 / 4) * (1 / 3) + (2 / 6) * (4 / 5) * (3 / 4) * (1 / 3)
  have : probability = 1 / 5 := sorry
  exact this

end probability_of_finding_last_defective_product_on_fourth_inspection_l60_60374


namespace centroid_locus_l60_60414

open EuclideanGeometry 

variables (A B C A1 B1 C1 : Point ℝ^2) (l : Line ℝ^2)

def triangle_centroid (A B C : Point ℝ^2) : Point ℝ^2 :=
  let x := (A.x + B.x + C.x) / 3 
      y := (A.y + B.y + C.y) / 3
  in ⟨x, y⟩

def midpoint (P Q : Point ℝ^2) : Point ℝ^2 :=
  let x := (P.x + Q.x) / 2
      y := (P.y + Q.y) / 2
  in ⟨x, y⟩

def line_centroid (P Q R : Point ℝ^2) : Point ℝ^2 :=
  let x := (P.x + Q.x + R.x) / 3
      y := (P.y + Q.y + R.y) / 3
  in ⟨x, y⟩

def homothety (center : Point ℝ^2) (ratio : ℝ) (P : Point ℝ^2) : Point ℝ^2 :=
  ⟨center.x + ratio * (P.x - center.x), center.y + ratio * (P.y - center.y)⟩

theorem centroid_locus (hA1 : A1 ∈ l) (hB1 : B1 ∈ l) (hC1 : C1 ∈ l) :
  ∃ l' : Line ℝ^2, parallel l l' ∧ 
  ∀ A1 B1 C1, (A1 ∈ l) → (B1 ∈ l) → (C1 ∈ l) →
  let M := triangle_centroid A B C
      X := line_centroid A1 B1 C1 in
  (midpoint M X) ∈ l' ∧ 
  homothety M (1/2) (line_centroid A1 B1 C1) ∈ l' := 
sorry

end centroid_locus_l60_60414


namespace count_two_digit_primes_with_units_digit_three_l60_60919

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l60_60919


namespace minimum_protein_content_is_at_least_1_8_l60_60217

-- Define the net weight of the can and the minimum protein percentage
def netWeight : ℝ := 300
def minProteinPercentage : ℝ := 0.006

-- Prove that the minimum protein content is at least 1.8 grams
theorem minimum_protein_content_is_at_least_1_8 :
  netWeight * minProteinPercentage ≥ 1.8 := 
by
  sorry

end minimum_protein_content_is_at_least_1_8_l60_60217


namespace find_x_squared_plus_y_squared_l60_60738

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l60_60738


namespace solve_inequality_system_l60_60995

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60995


namespace sum_mod_9_l60_60643

theorem sum_mod_9 (x y z : ℕ) (h1 : x < 9) (h2 : y < 9) (h3 : z < 9) 
  (h4 : x > 0) (h5 : y > 0) (h6 : z > 0)
  (h7 : (x * y * z) % 9 = 1) (h8 : (7 * z) % 9 = 4) (h9 : (8 * y) % 9 = (5 + y) % 9) :
  (x + y + z) % 9 = 7 := 
by {
  sorry
}

end sum_mod_9_l60_60643


namespace length_FJ_is_35_l60_60517

noncomputable def length_of_FJ (h : ℝ) : ℝ :=
  let FG := 50
  let HI := 20
  let trapezium_area := (1 / 2) * (FG + HI) * h
  let half_trapezium_area := trapezium_area / 2
  let JI_area := (1 / 2) * 35 * h
  35

theorem length_FJ_is_35 (h : ℝ) : length_of_FJ h = 35 :=
  sorry

end length_FJ_is_35_l60_60517


namespace sum_of_roots_of_quadratic_l60_60593

theorem sum_of_roots_of_quadratic :
  ∀ (x1 x2 : ℝ), (Polynomial.eval x1 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) ∧ 
                 (Polynomial.eval x2 (Polynomial.C 1 * Polynomial.X^2 + Polynomial.C (-3) * Polynomial.X + Polynomial.C (-4)) = 0) -> 
                 x1 + x2 = 3 := 
by
  intro x1 x2
  intro H
  sorry

end sum_of_roots_of_quadratic_l60_60593


namespace small_disks_radius_l60_60488

theorem small_disks_radius (r : ℝ) (h : r > 0) :
  (2 * r ≥ 1 + r) → (r ≥ 1 / 2) := by
  intro hr
  linarith

end small_disks_radius_l60_60488


namespace value_of_nested_expression_l60_60858

def nested_expression : ℕ :=
  3 * (3 * (3 * (3 * (3 * (3 + 2) + 2) + 2) + 2) + 2) + 2

theorem value_of_nested_expression : nested_expression = 1457 := by
  sorry

end value_of_nested_expression_l60_60858


namespace no_intersection_of_sets_l60_60572

noncomputable def A (a b x y : ℝ) :=
  a * (Real.sin x + Real.sin y) + (b - 1) * (Real.cos x + Real.cos y) = 0

noncomputable def B (a b x y : ℝ) :=
  (b + 1) * Real.sin (x + y) - a * Real.cos (x + y) = a

noncomputable def C (a b : ℝ) :=
  ∀ z : ℝ, z^2 - 2 * (a - b) * z + (a + b)^2 - 2 > 0

theorem no_intersection_of_sets (a b x y : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2) (h3 : 0 < y) (h4 : y < Real.pi / 2) :
  (C a b) → ¬(∃ x y, A a b x y ∧ B a b x y) :=
by 
  sorry

end no_intersection_of_sets_l60_60572


namespace find_number_l60_60082

theorem find_number (x : ℕ) (h : (x / 5) - 154 = 6) : x = 800 := by
  sorry

end find_number_l60_60082


namespace find_k_for_two_identical_solutions_l60_60708

theorem find_k_for_two_identical_solutions (k : ℝ) :
  (∃ x : ℝ, x^2 = 4 * x + k) ∧ (∀ x : ℝ, x^2 = 4 * x + k → x = 2) ↔ k = -4 :=
by
  sorry

end find_k_for_two_identical_solutions_l60_60708


namespace distinct_paintings_of_square_l60_60520

theorem distinct_paintings_of_square : 
  let disks := ({0, 1, 2, 3} : Finset ℕ),
      colorings := { l // (l.count 0 = 2) ∧ (l.count 1 = 1) ∧ (l.count 2 = 1) } in
  let symmetries := { rot0 := disks, rot90 := disks, rot180 := disks,
                       rot270 := disks, refh := disks, refv := disks, refd1 := disks, refd2 := disks } in 
  colorings.card / symmetries.card = 3 :=
by sorry

end distinct_paintings_of_square_l60_60520


namespace ratio_sum_l60_60866

theorem ratio_sum {x y : ℚ} (h : x / y = 3 / 4) : (x + y) / y = 7 / 4 :=
sorry

end ratio_sum_l60_60866


namespace sine_theorem_l60_60357

theorem sine_theorem (a b c α β γ : ℝ) 
  (h1 : a / Real.sin α = b / Real.sin β) 
  (h2 : b / Real.sin β = c / Real.sin γ) 
  (h3 : α + β + γ = Real.pi) :
  a = b * Real.cos γ + c * Real.cos β ∧
  b = c * Real.cos α + a * Real.cos γ ∧
  c = a * Real.cos β + b * Real.cos α :=
by
  sorry

end sine_theorem_l60_60357


namespace completing_the_square_l60_60058

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l60_60058


namespace time_per_mask_after_first_hour_l60_60862

-- Define the conditions as given in the problem
def rate_in_first_hour := 1 / 4 -- Manolo makes one face-mask every four minutes
def total_face_masks := 45 -- Manolo makes 45 face-masks in four hours
def first_hour_duration := 60 -- The duration of the first hour in minutes
def total_duration := 4 * 60 -- The total duration in minutes (4 hours)

-- Define the number of face-masks made in the first hour
def face_masks_first_hour := first_hour_duration / 4 -- 60 minutes / 4 minutes per face-mask = 15 face-masks

-- Calculate the number of face-masks made in the remaining time
def face_masks_remaining_hours := total_face_masks - face_masks_first_hour -- 45 - 15 = 30 face-masks

-- Define the duration of the remaining hours
def remaining_duration := total_duration - first_hour_duration -- 180 minutes (3 hours)

-- The target is to prove that the rate after the first hour is 6 minutes per face-mask
theorem time_per_mask_after_first_hour : remaining_duration / face_masks_remaining_hours = 6 := by
  sorry

end time_per_mask_after_first_hour_l60_60862


namespace ascorbic_acid_weight_l60_60805

def molecular_weight (formula : String) : ℝ :=
  if formula = "C6H8O6" then 176.12 else 0

theorem ascorbic_acid_weight : molecular_weight "C6H8O6" = 176.12 :=
by {
  sorry
}

end ascorbic_acid_weight_l60_60805


namespace total_area_of_figure_l60_60000

theorem total_area_of_figure :
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  total_area = 89 := by
  -- Definitions
  let rect1_height := 7
  let rect1_width := 6
  let rect2_height := 2
  let rect2_width := 6
  let rect3_height := 5
  let rect3_width := 4
  let rect4_height := 3
  let rect4_width := 5
  let area1 := rect1_height * rect1_width
  let area2 := rect2_height * rect2_width
  let area3 := rect3_height * rect3_width
  let area4 := rect4_height * rect4_width
  let total_area := area1 + area2 + area3 + area4
  -- Proof
  sorry

end total_area_of_figure_l60_60000


namespace blind_box_problem_l60_60239

theorem blind_box_problem (x y : ℕ) :
  x + y = 135 ∧ 2 * x = 3 * y :=
sorry

end blind_box_problem_l60_60239


namespace count_two_digit_primes_with_units_digit_3_l60_60927

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l60_60927


namespace son_age_l60_60809

variable (F S : ℕ)
variable (h₁ : F = 3 * S)
variable (h₂ : F - 8 = 4 * (S - 8))

theorem son_age : S = 24 := 
by 
  sorry

end son_age_l60_60809


namespace geometric_series_sum_l60_60383

-- Definition of the geometric sum function in Lean
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * ((1 - r^n) / (1 - r))

-- Specific terms for the problem
def a : ℚ := 2
def r : ℚ := 2 / 5
def n : ℕ := 5

-- The target sum we aim to prove
def target_sum : ℚ := 10310 / 3125

-- The theorem stating that the calculated sum equals the target sum
theorem geometric_series_sum : geometric_sum a r n = target_sum :=
by sorry

end geometric_series_sum_l60_60383


namespace siblings_total_weekly_water_l60_60662

noncomputable def Theo_daily : ℕ := 8
noncomputable def Mason_daily : ℕ := 7
noncomputable def Roxy_daily : ℕ := 9

noncomputable def daily_to_weekly (daily : ℕ) : ℕ := daily * 7

theorem siblings_total_weekly_water :
  daily_to_weekly Theo_daily + daily_to_weekly Mason_daily + daily_to_weekly Roxy_daily = 168 := by
  sorry

end siblings_total_weekly_water_l60_60662


namespace choir_average_age_solution_l60_60650

noncomputable def choir_average_age (avg_f avg_m avg_c : ℕ) (n_f n_m n_c : ℕ) : ℕ :=
  (n_f * avg_f + n_m * avg_m + n_c * avg_c) / (n_f + n_m + n_c)

def choir_average_age_problem : Prop :=
  let avg_f := 32
  let avg_m := 38
  let avg_c := 10
  let n_f := 12
  let n_m := 18
  let n_c := 5
  choir_average_age avg_f avg_m avg_c n_f n_m n_c = 32

theorem choir_average_age_solution : choir_average_age_problem := by
  sorry

end choir_average_age_solution_l60_60650


namespace eq_a_2_l60_60552

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l60_60552


namespace house_height_proof_l60_60344

noncomputable def height_of_house (house_shadow tree_height tree_shadow : ℕ) : ℕ :=
  house_shadow * tree_height / tree_shadow

theorem house_height_proof
  (house_shadow_length : ℕ)
  (tree_height : ℕ)
  (tree_shadow_length : ℕ)
  (expected_house_height : ℕ)
  (Hhouse_shadow_length : house_shadow_length = 56)
  (Htree_height : tree_height = 21)
  (Htree_shadow_length : tree_shadow_length = 24)
  (Hexpected_house_height : expected_house_height = 49) :
  height_of_house house_shadow_length tree_height tree_shadow_length = expected_house_height :=
by
  rw [Hhouse_shadow_length, Htree_height, Htree_shadow_length, Hexpected_house_height]
  -- Here we should compute the value and show it is equal to 49
  sorry

end house_height_proof_l60_60344


namespace ac_length_l60_60792

theorem ac_length (AB : ℝ) (H1 : AB = 100)
    (BC AC : ℝ)
    (H2 : AC = (1 + Real.sqrt 5)/2 * BC)
    (H3 : AC + BC = AB) : AC = 75 - 25 * Real.sqrt 5 :=
by
  sorry

end ac_length_l60_60792


namespace part1_part2_l60_60411

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x * Real.log x + a

theorem part1 (tangent_at_e : ∀ x : ℝ, f x e = 2 * e) : a = e := sorry

theorem part2 (m : ℝ) (a : ℝ) (hm : 0 < m) :
  (if m ≤ 1 / (2 * Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (2 * m) a 
   else if 1 / (2 * Real.exp 1) < m ∧ m < 1 / (Real.exp 1) then 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f (1 / (Real.exp 1)) a 
   else 
     ∀ x ∈ Set.Icc m (2 * m), f x a ≥ f m a) :=
  sorry

end part1_part2_l60_60411


namespace number_of_sides_of_regular_polygon_l60_60502

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l60_60502


namespace smoothie_one_serving_ingredients_in_cups_containers_needed_l60_60254

theorem smoothie_one_serving_ingredients_in_cups :
  (0.2 + 0.1 + 0.2 + 1 * 0.125 + 2 * 0.0625 + 0.5).round = 1.25.round := sorry

theorem containers_needed :
  (5 * 1.25 / 1.5).ceil = 5 := sorry

end smoothie_one_serving_ingredients_in_cups_containers_needed_l60_60254


namespace quadratic_inequality_solution_set_l60_60723

theorem quadratic_inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | ax^2 - (2 + a) * x + 2 > 0} = {x | 2 / a < x ∧ x < 1} :=
sorry

end quadratic_inequality_solution_set_l60_60723


namespace express_An_l60_60178

noncomputable def A_n (A : ℝ) (n : ℤ) : ℝ :=
  (1 / 2^n) * ((A + (A^2 - 4).sqrt)^n + (A - (A^2 - 4).sqrt)^n)

theorem express_An (a : ℝ) (A : ℝ) (n : ℤ) (h : a + a⁻¹ = A) :
  (a^n + a^(-n)) = A_n A n := 
sorry

end express_An_l60_60178


namespace evaluate_expr_at_2_l60_60637

def expr (x : ℝ) : ℝ := (2 * x + 3) * (2 * x - 3) + (x - 2) ^ 2 - 3 * x * (x - 1)

theorem evaluate_expr_at_2 : expr 2 = 1 :=
by
  sorry

end evaluate_expr_at_2_l60_60637


namespace determine_ABC_l60_60308

-- Define values in the new base system
def base_representation (A B C : ℕ) : ℕ :=
  A * (A+1)^7 + A * (A+1)^6 + A * (A+1)^5 + C * (A+1)^4 + B * (A+1)^3 + B * (A+1)^2 + B * (A+1) + C

-- The conditions given by the problem
def condition (A B C : ℕ) : Prop :=
  ((A+1)^8 - 2*(A+1)^4 + 1) = base_representation A B C

-- The theorem to be proved
theorem determine_ABC : ∃ (A B C : ℕ), A = 2 ∧ B = 0 ∧ C = 1 ∧ condition A B C :=
by
  existsi 2
  existsi 0
  existsi 1
  unfold condition base_representation
  sorry

end determine_ABC_l60_60308


namespace even_function_implies_a_is_2_l60_60556

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l60_60556


namespace fraction_meaningful_l60_60428

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end fraction_meaningful_l60_60428


namespace total_fish_caught_total_l60_60381
-- Include the broad Mathlib library to ensure all necessary mathematical functions and definitions are available

-- Define the conditions based on the given problem
def brian_trips (chris_trips : ℕ) : ℕ := 2 * chris_trips
def chris_fish_per_trip (brian_fish_per_trip : ℕ) : ℕ := brian_fish_per_trip + (2/5 : ℚ) * brian_fish_per_trip
def total_fish_caught (chris_trips : ℕ) (brian_fish_per_trip chris_fish_per_trip : ℕ) : ℕ := 
  brian_trips chris_trips * brian_fish_per_trip + chris_trips * chris_fish_per_trip

-- State the main proof problem based on the question and conditions
theorem total_fish_caught_total :
  ∀ (chris_trips : ℕ) (brian_fish_per_trip : ℕ) (chris_fish_per_trip : ℕ),
  chris_trips = 10 →
  brian_fish_per_trip = 400 →
  chris_fish_per_trip = 560 →
  total_fish_caught chris_trips brian_fish_per_trip chris_fish_per_trip = 13600 :=
by
  intros chris_trips brian_fish_per_trip chris_fish_per_trip h_chris_trips h_brian_fish_per_trip h_chris_fish_per_trip
  rw [h_chris_trips, h_brian_fish_per_trip, h_chris_fish_per_trip]
  sorry -- Proof omitted

end total_fish_caught_total_l60_60381


namespace calculation_l60_60242

theorem calculation : 120 / 5 / 3 * 2 = 16 := by
  sorry

end calculation_l60_60242


namespace gcd_987654_876543_eq_3_l60_60803

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end gcd_987654_876543_eq_3_l60_60803


namespace find_varphi_l60_60654

theorem find_varphi (ϕ : ℝ) (h0 : 0 < ϕ ∧ ϕ < π / 2) :
  (∀ x₁ x₂, |(2 * Real.cos (2 * x₁)) - (2 * Real.cos (2 * x₂ - 2 * ϕ))| = 4 → 
    ∃ (x₁ x₂ : ℝ), |x₁ - x₂| = π / 6 
  ) → ϕ = π / 3 :=
by
  sorry

end find_varphi_l60_60654


namespace circle_equation_l60_60679

theorem circle_equation (x y : ℝ)
  (h_center : ∀ x y, (x - 3)^2 + (y - 1)^2 = r ^ 2)
  (h_origin : (0 - 3)^2 + (0 - 1)^2 = r ^ 2) :
  (x - 3) ^ 2 + (y - 1) ^ 2 = 10 := by
  sorry

end circle_equation_l60_60679


namespace spadesuit_evaluation_l60_60399

def spadesuit (x y : ℝ) : ℝ := (x + y) * (x - y)

theorem spadesuit_evaluation : spadesuit 3 (spadesuit 4 5) = -72 := by
  sorry

end spadesuit_evaluation_l60_60399


namespace probability_three_girls_l60_60754

theorem probability_three_girls :
  let p := 0.5 in
  (((nat.choose 6 3) * (p^3) * (p^3)) = (5 / 16)) :=
by sorry

end probability_three_girls_l60_60754


namespace hose_Z_fill_time_l60_60644

theorem hose_Z_fill_time (P X Y Z : ℝ) (h1 : X + Y = P / 3) (h2 : Y = P / 9) (h3 : X + Z = P / 4) (h4 : X + Y + Z = P / 2.5) : Z = P / 15 :=
sorry

end hose_Z_fill_time_l60_60644


namespace fraction_meaningful_l60_60425

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_l60_60425


namespace solve_inequality_system_l60_60993

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60993


namespace max_difference_primes_l60_60196

def is_prime (n : ℕ) := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def even_integer : ℕ := 138

theorem max_difference_primes (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ p + q = even_integer ∧ p ≠ q →
  (q - p) = 124 :=
by
  sorry

end max_difference_primes_l60_60196


namespace quadratic_roots_problem_l60_60175

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l60_60175


namespace count_prime_units_digit_3_eq_6_l60_60922

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l60_60922


namespace completing_the_square_l60_60047

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l60_60047


namespace trainers_hours_split_equally_l60_60334

noncomputable def dolphins := 12
noncomputable def hours_per_dolphin := 5
noncomputable def trainers := 4

theorem trainers_hours_split_equally :
  (dolphins * hours_per_dolphin) / trainers = 15 :=
by
  sorry

end trainers_hours_split_equally_l60_60334


namespace even_function_implies_a_eq_2_l60_60548

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l60_60548


namespace calculate_selling_price_l60_60822

noncomputable def purchase_price : ℝ := 225
noncomputable def overhead_expenses : ℝ := 20
noncomputable def profit_percent : ℝ := 22.448979591836732

noncomputable def total_cost : ℝ := purchase_price + overhead_expenses
noncomputable def profit : ℝ := (profit_percent / 100) * total_cost
noncomputable def selling_price : ℝ := total_cost + profit

theorem calculate_selling_price : selling_price = 300 := by
  sorry

end calculate_selling_price_l60_60822


namespace fraction_sum_l60_60836

theorem fraction_sum : (1 / 3 : ℚ) + (5 / 9 : ℚ) = (8 / 9 : ℚ) :=
by
  sorry

end fraction_sum_l60_60836


namespace find_m_l60_60950

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end find_m_l60_60950


namespace value_of_b_l60_60745

theorem value_of_b (b : ℝ) :
  (∀ x : ℝ, 3 * (5 + b * x) = 18 * x + 15) → b = 6 :=
by
  intro h
  -- Proving that b = 6
  sorry

end value_of_b_l60_60745


namespace angle_ratio_half_l60_60955

theorem angle_ratio_half (a b c : ℝ) (A B C : ℝ) (h1 : a^2 = b * (b + c))
  (h2 : A = 2 * B ∨ A + 2 * B = Real.pi) 
  (h3 : A + B + C = Real.pi) : 
  (B / A = 1 / 2) :=
sorry

end angle_ratio_half_l60_60955


namespace complete_the_square_l60_60052

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l60_60052


namespace problem_statement_l60_60714

open Real

theorem problem_statement (a b : ℝ) (n : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : (1/a) + (1/b) = 1) (hn_pos : 0 < n) :
  (a + b) ^ n - a ^ n - b ^ n ≥ 2 ^ (2 * n) - 2 ^ (n + 1) :=
sorry -- proof to be provided

end problem_statement_l60_60714


namespace problem_solution_l60_60108

noncomputable def complex_expression : ℝ :=
  (-(1/2) * (1/100))^5 * ((2/3) * (2/100))^4 * (-(3/4) * (3/100))^3 * ((4/5) * (4/100))^2 * (-(5/6) * (5/100)) * 10^30

theorem problem_solution : complex_expression = -48 :=
by
  sorry

end problem_solution_l60_60108


namespace length_AC_l60_60402

open Real

noncomputable def net_south_north (south north : ℝ) : ℝ := south - north
noncomputable def net_east_west (east west : ℝ) : ℝ := east - west
noncomputable def distance (a b : ℝ) : ℝ := sqrt (a^2 + b^2)

theorem length_AC :
  let A : ℝ := 0
  let south := 30
  let north := 20
  let east := 40
  let west := 35
  let net_south := net_south_north south north
  let net_east := net_east_west east west
  distance net_south net_east = 5 * sqrt 5 :=
by
  sorry

end length_AC_l60_60402


namespace factorial_product_lt_sum_factorial_l60_60716

open BigOperators

theorem factorial_product_lt_sum_factorial (a : ℕ → ℕ) (n : ℕ) (hpos : ∀ i, i < n → 0 < a i) :
  (∏ i in Finset.range n, (a i)!) < ((∑ i in Finset.range n, a i) + 1)! :=
by sorry

end factorial_product_lt_sum_factorial_l60_60716


namespace tan_2x_eq_sin_x_has_three_solutions_l60_60147

theorem tan_2x_eq_sin_x_has_three_solutions :
  ∃ (S : Finset ℝ), (∀ x ∈ S, 0 ≤ x ∧ x ≤ 2 * Real.pi ∧ Real.tan (2 * x) = Real.sin x) ∧ S.card = 3 :=
by
  sorry

end tan_2x_eq_sin_x_has_three_solutions_l60_60147


namespace max_value_at_x_eq_2_l60_60863

noncomputable def quadratic_function (x : ℝ) : ℝ := -(x - 2)^2 - 3

theorem max_value_at_x_eq_2 : ∀ x : ℝ, quadratic_function x ≤ quadratic_function 2 := by
  sorry

end max_value_at_x_eq_2_l60_60863


namespace num_two_digit_prime_with_units_digit_3_eq_6_l60_60893

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l60_60893


namespace complete_the_square_l60_60051

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l60_60051


namespace total_payment_is_correct_l60_60442

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end total_payment_is_correct_l60_60442


namespace length_of_AB_l60_60597
-- Import the necessary libraries

-- Define the quadratic function
def quad (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define a predicate to state that x is a root of the quadratic
def is_root (x : ℝ) : Prop := quad x = 0

-- Define the length between the intersection points
theorem length_of_AB :
  (is_root (-1)) ∧ (is_root 3) → |3 - (-1)| = 4 :=
by {
  sorry
}

end length_of_AB_l60_60597


namespace problem_x2_plus_y2_l60_60741

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l60_60741


namespace count_two_digit_primes_with_units_digit_3_l60_60909

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l60_60909


namespace translation_symmetric_graphs_l60_60163

/-- The graph of the function f(x)=sin(x/π + φ) is translated to the right by θ (θ>0) units to obtain the graph of the function g(x).
    On the graph of f(x), point A is translated to point B, let x_A and x_B be the abscissas of points A and B respectively.
    If the axes of symmetry of the graphs of f(x) and g(x) coincide, then the real values that can be taken as x_A - x_B are -2π² or -π². -/
theorem translation_symmetric_graphs (θ : ℝ) (hθ : θ > 0) (x_A x_B : ℝ) (φ : ℝ) :
  ((x_A - x_B = -2 * π^2) ∨ (x_A - x_B = -π^2)) :=
sorry

end translation_symmetric_graphs_l60_60163


namespace exists_five_distinct_nat_numbers_l60_60750

theorem exists_five_distinct_nat_numbers 
  (a b c d e : ℕ)
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h_no_div_3 : ¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e))
  (h_no_div_4 : ¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e))
  (h_no_div_5 : ¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) :
  (∃ (a b c d e : ℕ),
    (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e) ∧
    (¬(3 ∣ a) ∧ ¬(3 ∣ b) ∧ ¬(3 ∣ c) ∧ ¬(3 ∣ d) ∧ ¬(3 ∣ e)) ∧
    (¬(4 ∣ a) ∧ ¬(4 ∣ b) ∧ ¬(4 ∣ c) ∧ ¬(4 ∣ d) ∧ ¬(4 ∣ e)) ∧
    (¬(5 ∣ a) ∧ ¬(5 ∣ b) ∧ ¬(5 ∣ c) ∧ ¬(5 ∣ d) ∧ ¬(5 ∣ e)) ∧
    (∀ x y z : ℕ, x ≠ y ∧ y ≠ z ∧ x ≠ z → x + y + z = a + b + c + d + e → (x + y + z) % 3 = 0) ∧
    (∀ w x y z : ℕ, w ≠ x ∧ x ≠ y ∧ y ≠ z ∧ w ≠ y ∧ w ≠ z ∧ x ≠ z → w + x + y + z = a + b + c + d + e → (w + x + y + z) % 4 = 0) ∧
    (a + b + c + d + e) % 5 = 0) :=
  sorry

end exists_five_distinct_nat_numbers_l60_60750


namespace correct_transformation_l60_60351

-- Conditions given in the problem
def cond_A (a : ℤ) : Prop := a + 3 = 9 → a = 3 + 9
def cond_B (x : ℤ) : Prop := 4 * x = 7 * x - 2 → 4 * x - 7 * x = 2
def cond_C (a : ℤ) : Prop := 2 * a - 2 = -6 → 2 * a = 6 + 2
def cond_D (x : ℤ) : Prop := 2 * x - 5 = 3 * x + 3 → 2 * x - 3 * x = 3 + 5

-- Prove that the transformation in condition D is correct
theorem correct_transformation : (∀ a : ℤ, ¬cond_A a) ∧ (∀ x : ℤ, ¬cond_B x) ∧ (∀ a : ℤ, ¬cond_C a) ∧ (∀ x : ℤ, cond_D x) :=
by {
  -- Proof is provided in the solution and skipped here
  sorry
}

end correct_transformation_l60_60351


namespace solve_inequality_l60_60703

noncomputable def valid_x_values : set ℝ :=
  {x | x ∈ set.Icc 3.790 5 \ set.Icc 5 5 ∪ set.Icc 5 7.067}

theorem solve_inequality (x : ℝ) :
  (x ∈ valid_x_values) ↔ ((x * (x + 2) / (x - 5) ^ 2) ≥ 15) :=
sorry

end solve_inequality_l60_60703


namespace attendance_methods_probability_AB_probability_each_event_l60_60569

open Finset

-- Step 1
/-- Given 6 individuals labeled as A, B, C, etc.,
prove that the number of different attendance methods,
with at least one person required to attend, is 63 -/
theorem attendance_methods (n : ℕ) (hn : n = 6) :
  (2^n - 1) = 63 :=
by {
  rw hn,
  norm_num,
}

-- Step 2
/-- Given 6 individuals participating in 6 different events,
prove that the probability that individual A does not participate in the first event
and individual B does not participate in the third event is 7/10 -/
theorem probability_AB (n : ℕ) (hn : n = 6) :
  (504 / 720 : ℝ) = 7 / 10 :=
by {
  rw hn,
  norm_num,
}

-- Step 3
/-- Given 6 individuals participating in 4 different events,
prove that the probability that each event has at least one person participating is 195/512 -/
theorem probability_each_event (n : ℕ) (hn : n = 6) (m : ℕ) (hm : m = 4) :
  (1560 / m^n : ℝ) = 195 / 512 :=
by {
  rw [hn, hm],
  norm_num,
}

end attendance_methods_probability_AB_probability_each_event_l60_60569


namespace two_digit_primes_with_units_digit_three_l60_60939

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l60_60939


namespace exists_integers_a_b_c_d_and_n_l60_60447

theorem exists_integers_a_b_c_d_and_n (n a b c d : ℕ)
  (h1 : a = 10) 
  (h2 : b = 15) 
  (h3 : c = 8) 
  (h4 : d = 3) 
  (h5 : n = 16) :
  a^4 + b^4 + c^4 + 2 * d^4 = n^4 := by
  -- Proof goes here
  sorry

end exists_integers_a_b_c_d_and_n_l60_60447


namespace trig_identity_theorem_l60_60695

noncomputable def trig_identity_proof : Prop :=
  (1 + Real.cos (Real.pi / 9)) * 
  (1 + Real.cos (2 * Real.pi / 9)) * 
  (1 + Real.cos (4 * Real.pi / 9)) * 
  (1 + Real.cos (5 * Real.pi / 9)) = 
  (1 / 2) * (Real.sin (Real.pi / 9))^4

#check trig_identity_proof

theorem trig_identity_theorem : trig_identity_proof := by
  sorry

end trig_identity_theorem_l60_60695


namespace even_function_implies_a_eq_2_l60_60550

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l60_60550


namespace x_power_expression_l60_60765

theorem x_power_expression (x : ℝ) (h : x^3 - 3 * x = 5) : x^5 - 27 * x^2 = -22 * x^2 + 9 * x + 15 :=
by
  --proof goes here
  sorry

end x_power_expression_l60_60765


namespace cos_of_angle_complement_l60_60155

theorem cos_of_angle_complement (α : ℝ) (h : 90 - α = 30) : Real.cos α = 1 / 2 :=
by
  sorry

end cos_of_angle_complement_l60_60155


namespace tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l60_60887

open Real

axiom sin_add_half_pi_div_4_eq_zero (α : ℝ) : 
  sin (α + π / 4) + 2 * sin (α - π / 4) = 0

axiom tan_sub_half_pi_div_4_eq_inv_3 (β : ℝ) : 
  tan (π / 4 - β) = 1 / 3

theorem tan_alpha_eq_inv_3 (α : ℝ) (h : sin (α + π / 4) + 2 * sin (α - π / 4) = 0) : 
  tan α = 1 / 3 := sorry

theorem tan_alpha_add_beta_eq_1 (α β : ℝ) 
  (h1 : tan α = 1 / 3) (h2 : tan (π / 4 - β) = 1 / 3) : 
  tan (α + β) = 1 := sorry

end tan_alpha_eq_inv_3_tan_alpha_add_beta_eq_1_l60_60887


namespace find_fx_sum_roots_l60_60194

noncomputable def f : ℝ → ℝ
| x => if x = 2 then 1 else Real.log (abs (x - 2))

theorem find_fx_sum_roots
  (b c : ℝ)
  (x1 x2 x3 x4 x5 : ℝ)
  (h : ∀ x, (f x) ^ 2 + b * (f x) + c = 0)
  (h_distinct : x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x1 ≠ x5 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x2 ≠ x5 ∧ x3 ≠ x4 ∧ x3 ≠ x5 ∧ x4 ≠ x5 ) :
  f (x1 + x2 + x3 + x4 + x5) = Real.log 8 :=
sorry

end find_fx_sum_roots_l60_60194


namespace count_two_digit_primes_with_units_digit_3_l60_60916

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l60_60916


namespace bucky_savings_excess_l60_60834

def cost_of_game := 60
def saved_amount := 15
def fish_earnings_weekends (fish : String) : ℕ :=
  match fish with
  | "trout" => 5
  | "bluegill" => 4
  | "bass" => 7
  | "catfish" => 6
  | _ => 0

def fish_earnings_weekdays (fish : String) : ℕ :=
  match fish with
  | "trout" => 10
  | "bluegill" => 8
  | "bass" => 14
  | "catfish" => 12
  | _ => 0

def sunday_fish := 10
def weekday_fish := 3
def weekdays := 2

def sunday_fish_distribution := [
  ("trout", 3),
  ("bluegill", 2),
  ("bass", 4),
  ("catfish", 1)
]

noncomputable def sunday_earnings : ℕ :=
  sunday_fish_distribution.foldl (λ acc (fish, count) =>
    acc + count * fish_earnings_weekends fish) 0

noncomputable def weekday_earnings : ℕ :=
  weekdays * weekday_fish * (
    fish_earnings_weekdays "trout" +
    fish_earnings_weekdays "bluegill" +
    fish_earnings_weekdays "bass")

noncomputable def total_earnings : ℕ :=
  sunday_earnings + weekday_earnings

noncomputable def total_savings : ℕ :=
  total_earnings + saved_amount

theorem bucky_savings_excess :
  total_savings - cost_of_game = 76 :=
by sorry

end bucky_savings_excess_l60_60834


namespace harry_total_travel_time_l60_60291

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l60_60291


namespace g_at_8_l60_60333

def g (x : ℝ) : ℝ := sorry

axiom g_property : ∀ x y : ℝ, x * g y = y * g x

axiom g_at_24 : g 24 = 12

theorem g_at_8 : g 8 = 4 := by
  sorry

end g_at_8_l60_60333


namespace count_two_digit_prime_numbers_with_units_digit_3_l60_60898

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l60_60898


namespace value_of_v_over_u_l60_60486

variable (u v : ℝ) 

theorem value_of_v_over_u (h : u - v = (u + v) / 2) : v / u = 1 / 3 :=
by
  sorry

end value_of_v_over_u_l60_60486


namespace quadrant_of_angle_l60_60942

-- Define the quadrant property
def is_in_second_quadrant (α : ℝ) : Prop :=
  (Real.sin α > 0) ∧ (Real.cos α < 0)

-- Prove that α is in the second quadrant given the conditions
theorem quadrant_of_angle (α : ℝ) (h_sin : Real.sin α > 0) (h_cos : Real.cos α < 0) : is_in_second_quadrant α :=
by
  exact ⟨h_sin, h_cos⟩

end quadrant_of_angle_l60_60942


namespace new_average_after_doubling_l60_60786

theorem new_average_after_doubling
  (avg : ℝ) (num_students : ℕ) (h_avg : avg = 40) (h_num_students : num_students = 10) :
  let total_marks := avg * num_students
  let new_total_marks := total_marks * 2
  let new_avg := new_total_marks / num_students
  new_avg = 80 :=
by
  sorry

end new_average_after_doubling_l60_60786


namespace opposite_of_expression_l60_60460

theorem opposite_of_expression : 
  let expr := 1 - (3 : ℝ)^(1/3)
  (-1 + (3 : ℝ)^(1/3)) = (3 : ℝ)^(1/3) - 1 :=
by 
  let expr := 1 - (3 : ℝ)^(1/3)
  sorry

end opposite_of_expression_l60_60460


namespace find_n_from_binomial_variance_l60_60871

variable (ξ : Type)
variable (n : ℕ)
variable (p : ℝ := 0.3)
variable (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p))

-- Given conditions
axiom binomial_distribution : p = 0.3 ∧ Var n p = 2.1

-- Prove n = 10
theorem find_n_from_binomial_variance (ξ : Type) (n : ℕ) (p : ℝ := 0.3) (Var : ℕ → ℝ → ℝ := λ n p => n * p * (1 - p)) :
  p = 0.3 ∧ Var n p = 2.1 → n = 10 :=
by
  sorry

end find_n_from_binomial_variance_l60_60871


namespace number_of_terminating_decimals_l60_60707

theorem number_of_terminating_decimals : 
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 299 → (∃ k : ℕ, n = 9 * k) → 
  ∃ count : ℕ, count = 33 := 
sorry

end number_of_terminating_decimals_l60_60707


namespace common_ratio_of_geometric_sequence_l60_60953

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ)
  (h1 : ∀ n, a (n + 1) = a n * q)
  (h2 : a 1 + a 4 = 9)
  (h3 : a 2 * a 3 = 8)
  (h4 : ∀ n, a n ≤ a (n + 1)) :
  q = 2 :=
by
  sorry

end common_ratio_of_geometric_sequence_l60_60953


namespace meet_without_contact_probability_l60_60043

noncomputable def prob_meet_without_contact : ℝ :=
  let total_area := 1
  let outside_area := (1 / 8) * 2
  total_area - outside_area

theorem meet_without_contact_probability :
  prob_meet_without_contact = 3 / 4 :=
by
  sorry

end meet_without_contact_probability_l60_60043


namespace intersection_of_P_and_Q_l60_60941

theorem intersection_of_P_and_Q (P Q : Set ℕ) (h1 : P = {1, 3, 6, 9}) (h2 : Q = {1, 2, 4, 6, 8}) :
  P ∩ Q = {1, 6} :=
by
  sorry

end intersection_of_P_and_Q_l60_60941


namespace number_of_sides_of_regular_polygon_l60_60503

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l60_60503


namespace teal_sales_revenue_l60_60431

theorem teal_sales_revenue :
  let pumpkin_pie_slices := 8
  let pumpkin_pie_price := 5
  let pumpkin_pies_sold := 4
  let custard_pie_slices := 6
  let custard_pie_price := 6
  let custard_pies_sold := 5
  let apple_pie_slices := 10
  let apple_pie_price := 4
  let apple_pies_sold := 3
  let pecan_pie_slices := 12
  let pecan_pie_price := 7
  let pecan_pies_sold := 2
  (pumpkin_pie_slices * pumpkin_pie_price * pumpkin_pies_sold) +
  (custard_pie_slices * custard_pie_price * custard_pies_sold) +
  (apple_pie_slices * apple_pie_price * apple_pies_sold) +
  (pecan_pie_slices * pecan_pie_price * pecan_pies_sold) = 
  628 := by
  sorry

end teal_sales_revenue_l60_60431


namespace completing_the_square_l60_60067

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l60_60067


namespace completing_the_square_l60_60056

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l60_60056


namespace angle_215_third_quadrant_l60_60877

-- Define the context of the problem
def angle_vertex_origin : Prop := true 

def initial_side_non_negative_x_axis : Prop := true

noncomputable def in_third_quadrant (angle: ℝ) : Prop := 
  180 < angle ∧ angle < 270 

-- The theorem to prove the condition given
theorem angle_215_third_quadrant : 
  angle_vertex_origin → 
  initial_side_non_negative_x_axis → 
  in_third_quadrant 215 :=
by
  intro _ _
  unfold in_third_quadrant
  sorry -- This is where the proof would go

end angle_215_third_quadrant_l60_60877


namespace expected_product_two_uniform_numbers_l60_60339

open ProbabilityTheory

noncomputable def expected_value_uniform (n : ℕ) : ℝ :=
  (Real.ofNat (n * (n + 1)) / 2) / n

theorem expected_product_two_uniform_numbers :
  let X Y : ℕ := 10,
  let E := expected_value_uniform 10
  in E * E = 30.25 :=
by
  sorry

end expected_product_two_uniform_numbers_l60_60339


namespace fraction_of_sides_area_of_triangle_l60_60433

-- Part (1)
theorem fraction_of_sides (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) : (a + b) / c = 2 :=
sorry

-- Part (2)
theorem area_of_triangle (A B C : ℝ) (a b c : ℝ) (h_triangle : A + B + C = π)
  (h_sines : 2 * (Real.tan A + Real.tan B) = (Real.tan A / Real.cos B) + (Real.tan B / Real.cos A))
  (h_sine_law : c = 2) (h_C : C = π / 3) : (1 / 2) * a * b * Real.sin C = Real.sqrt 3 :=
sorry

end fraction_of_sides_area_of_triangle_l60_60433


namespace negation_equiv_l60_60794

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l60_60794


namespace even_function_implies_a_is_2_l60_60557

noncomputable def f (a x : ℝ) : ℝ := (x * Real.exp x) / (Real.exp (a * x) - 1)

theorem even_function_implies_a_is_2 (a : ℝ) 
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
by
  sorry

end even_function_implies_a_is_2_l60_60557


namespace total_area_needed_l60_60230

-- Definitions based on conditions
def oak_trees_first_half := 100
def pine_trees_first_half := 100
def oak_trees_second_half := 150
def pine_trees_second_half := 150
def oak_tree_planting_ratio := 4
def pine_tree_planting_ratio := 2
def oak_tree_space := 4
def pine_tree_space := 2

-- Total area needed for tree planting during the entire year
theorem total_area_needed : (oak_trees_first_half * oak_tree_planting_ratio * oak_tree_space) + ((pine_trees_first_half + pine_trees_second_half) * pine_tree_planting_ratio * pine_tree_space) = 2600 :=
by
  sorry

end total_area_needed_l60_60230


namespace max_product_price_l60_60244

/-- Conditions: 
1. Company C sells 50 products.
2. The average retail price of the products is $2,500.
3. No product sells for less than $800.
4. Exactly 20 products sell for less than $2,000.
Goal:
Prove that the greatest possible selling price of the most expensive product is $51,000.
-/
theorem max_product_price (n : ℕ) (avg_price : ℝ) (min_price : ℝ) (threshold_price : ℝ) (num_below_threshold : ℕ) :
  n = 50 → 
  avg_price = 2500 → 
  min_price = 800 → 
  threshold_price = 2000 → 
  num_below_threshold = 20 → 
  ∃ max_price : ℝ, max_price = 51000 :=
by 
  sorry

end max_product_price_l60_60244


namespace min_value_of_function_l60_60527

theorem min_value_of_function (x : ℝ) (hx : x > 0) :
  ∃ y, y = (3 + x + x^2) / (1 + x) ∧ y = -1 + 2 * Real.sqrt 3 :=
sorry

end min_value_of_function_l60_60527


namespace unattainable_y_value_l60_60390

theorem unattainable_y_value (y : ℝ) (x : ℝ) (h : x ≠ -4 / 3) : ¬ (y = -1 / 3) :=
by {
  -- The proof is omitted for now. 
  -- We're only constructing the outline with necessary imports and conditions.
  sorry
}

end unattainable_y_value_l60_60390


namespace calculate_total_interest_rate_l60_60094

noncomputable def total_investment : ℝ := 10000
noncomputable def amount_invested_11_percent : ℝ := 3750
noncomputable def amount_invested_9_percent : ℝ := total_investment - amount_invested_11_percent
noncomputable def interest_rate_9_percent : ℝ := 0.09
noncomputable def interest_rate_11_percent : ℝ := 0.11

noncomputable def interest_from_9_percent : ℝ := interest_rate_9_percent * amount_invested_9_percent
noncomputable def interest_from_11_percent : ℝ := interest_rate_11_percent * amount_invested_11_percent

noncomputable def total_interest : ℝ := interest_from_9_percent + interest_from_11_percent

noncomputable def total_interest_rate : ℝ := (total_interest / total_investment) * 100

theorem calculate_total_interest_rate :
  total_interest_rate = 9.75 :=
by 
  sorry

end calculate_total_interest_rate_l60_60094


namespace camilla_blueberry_jelly_beans_l60_60243

theorem camilla_blueberry_jelly_beans (b c : ℕ) 
  (h1 : b = 3 * c)
  (h2 : b - 20 = 2 * (c - 5)) : 
  b = 30 := 
sorry

end camilla_blueberry_jelly_beans_l60_60243


namespace find_y_when_x_is_8_l60_60642

theorem find_y_when_x_is_8 : 
  ∃ k, (70 * 5 = k ∧ 8 * 25 = k) := 
by
  -- The proof will be filled in here
  sorry

end find_y_when_x_is_8_l60_60642


namespace trajectory_moving_point_hyperbola_l60_60684

theorem trajectory_moving_point_hyperbola {n m : ℝ} (h_neg_n : n < 0) :
    (∃ y < 0, (y^2 = 16) ∧ (m^2 = (n^2 / 4 - 4))) ↔ ( ∃ (y : ℝ), (y^2 / 16) - (m^2 / 4) = 1 ∧ y < 0 ) := 
sorry

end trajectory_moving_point_hyperbola_l60_60684


namespace completing_the_square_l60_60057

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l60_60057


namespace john_books_nights_l60_60752

theorem john_books_nights (n : ℕ) (cost_per_night discount amount_paid : ℕ) 
  (h1 : cost_per_night = 250)
  (h2 : discount = 100)
  (h3 : amount_paid = 650)
  (h4 : amount_paid = cost_per_night * n - discount) : 
  n = 3 :=
by
  sorry

end john_books_nights_l60_60752


namespace determine_coefficients_l60_60812

noncomputable def polynomial_coefficients (A B C D : ℝ) : Prop :=
  let P := (λ x, x^6 + 4 * x^5 + A * x^4 + B * x^3 + C * x^2 + D * x + 1)
  let Q := (λ x, x^6 - 4 * x^5 + A * x^4 - B * x^3 + C * x^2 - D * x + 1)
  ∃ b : ℝ, P = (λ x, (x^3 + 2 * x^2 + b * x + 1)^2) ∧ Q = (λ x, (x^3 - 2 * x^2 + b * x - 1)^2)

theorem determine_coefficients (A B C D : ℝ) :
  polynomial_coefficients A B C D → 
  (A = 8 ∧ B = 10 ∧ C = 8 ∧ D = 4) := 
sorry

end determine_coefficients_l60_60812


namespace consecutive_sum_to_20_has_one_set_l60_60587

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end consecutive_sum_to_20_has_one_set_l60_60587


namespace total_people_large_seats_is_84_l60_60646

-- Definition of the number of large seats
def large_seats : Nat := 7

-- Definition of the number of people each large seat can hold
def people_per_large_seat : Nat := 12

-- Definition of the total number of people that can ride on large seats
def total_people_large_seats : Nat := large_seats * people_per_large_seat

-- Statement that we need to prove
theorem total_people_large_seats_is_84 : total_people_large_seats = 84 := by
  sorry

end total_people_large_seats_is_84_l60_60646


namespace range_of_m_l60_60420

def point_P := (1, 1)
def circle_C1 (x y m : ℝ) := x^2 + y^2 + 2*x - m = 0

theorem range_of_m (m : ℝ) :
  (1 + 1)^2 + 1^2 > m + 1 → -1 < m ∧ m < 4 :=
by
  sorry

end range_of_m_l60_60420


namespace factor_difference_of_squares_l60_60257

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l60_60257


namespace one_fourth_of_8_point8_simplified_l60_60853

noncomputable def one_fourth_of (x : ℚ) : ℚ := x / 4

def convert_to_fraction (x : ℚ) : ℚ := 
  let num := 22
  let denom := 10
  num / denom

def simplify_fraction (num denom : ℚ) (gcd : ℚ) : ℚ := 
  (num / gcd) / (denom / gcd)

theorem one_fourth_of_8_point8_simplified : one_fourth_of 8.8 = (11 / 5) := 
by
  have h : one_fourth_of 8.8 = 2.2 := by sorry
  have h_frac : 2.2 = (22 / 10) := by sorry
  have h_simplified : (22 / 10) = (11 / 5) := by sorry
  rw [h, h_frac, h_simplified]
  exact rfl

end one_fourth_of_8_point8_simplified_l60_60853


namespace minimum_arc_length_of_curve_and_line_l60_60422

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end minimum_arc_length_of_curve_and_line_l60_60422


namespace store_earnings_correct_l60_60367

theorem store_earnings_correct :
  let graphics_cards_sold : ℕ := 10
  let hard_drives_sold : ℕ := 14
  let cpus_sold : ℕ := 8
  let ram_pairs_sold : ℕ := 4
  let graphics_card_price : ℝ := 600
  let hard_drive_price : ℝ := 80
  let cpu_price : ℝ := 200
  let ram_pair_price : ℝ := 60
  graphics_cards_sold * graphics_card_price +
  hard_drives_sold * hard_drive_price +
  cpus_sold * cpu_price +
  ram_pairs_sold * ram_pair_price = 8960 := 
by
  sorry

end store_earnings_correct_l60_60367


namespace num_two_digit_primes_with_units_digit_three_l60_60902

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l60_60902


namespace identity_proof_l60_60016

theorem identity_proof (a b : ℝ) : a^4 + b^4 + (a + b)^4 = 2 * (a^2 + a * b + b^2)^2 := 
sorry

end identity_proof_l60_60016


namespace num_solutions_abs_eq_l60_60579

theorem num_solutions_abs_eq (B : ℤ) (hB : B = 3) : 
  { x : ℤ | |x - 2| + |x + 1| = B }.finite.to_finset.card = 4 :=
by
  sorry

end num_solutions_abs_eq_l60_60579


namespace roots_eq_solution_l60_60174

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l60_60174


namespace amount_exceeds_l60_60294

theorem amount_exceeds (N : ℕ) (A : ℕ) (h1 : N = 1925) (h2 : N / 7 - N / 11 = A) :
  A = 100 :=
sorry

end amount_exceeds_l60_60294


namespace borrowed_years_l60_60771

noncomputable def principal : ℝ := 5396.103896103896
noncomputable def interest_rate : ℝ := 0.06
noncomputable def total_returned : ℝ := 8310

theorem borrowed_years :
  ∃ t : ℝ, (total_returned - principal) = principal * interest_rate * t ∧ t = 9 :=
by
  sorry

end borrowed_years_l60_60771


namespace least_common_duration_l60_60471

theorem least_common_duration 
    (P Q R : ℝ) 
    (x : ℝ)
    (T : ℝ)
    (h1 : P / Q = 7 / 5)
    (h2 : Q / R = 5 / 3)
    (h3 : 8 * P / (6 * Q) = 7 / 10)
    (h4 : (6 * 10) * R / (30 * T) = 1)
    : T = 6 :=
by
  sorry

end least_common_duration_l60_60471


namespace blue_pens_count_l60_60979

variable (x y : ℕ) -- Define x as the number of red pens and y as the number of blue pens.
variable (h1 : 5 * x + 7 * y = 102) -- Condition 1: Total cost equation.
variable (h2 : x + y = 16) -- Condition 2: Total number of pens equation.

theorem blue_pens_count : y = 11 :=
by
  sorry

end blue_pens_count_l60_60979


namespace molly_age_condition_l60_60448

-- Definitions
def S : ℕ := 38 - 6
def M : ℕ := 24

-- The proof problem
theorem molly_age_condition :
  (S / M = 4 / 3) → (S = 32) → (M = 24) :=
by
  intro h_ratio h_S
  sorry

end molly_age_condition_l60_60448


namespace minimum_possible_area_l60_60104

theorem minimum_possible_area (l w l_min w_min : ℝ) (hl : l = 5) (hw : w = 7) 
  (hl_min : l_min = l - 0.5) (hw_min : w_min = w - 0.5) : 
  l_min * w_min = 29.25 :=
by
  sorry

end minimum_possible_area_l60_60104


namespace math_problem_l60_60107

theorem math_problem : ((-7)^3 / 7^2 - 2^5 + 4^3 - 8) = 81 :=
by
  sorry

end math_problem_l60_60107


namespace machine_work_rates_l60_60038

theorem machine_work_rates :
  (∃ x : ℝ, (1 / (x + 4) + 1 / (x + 3) + 1 / (x + 2)) = 1 / x ∧ x = 1 / 2) :=
by
  sorry

end machine_work_rates_l60_60038


namespace sum_le_30_l60_60775

variable (a b x y : ℝ)
variable (ha_pos : 0 < a) (hb_pos : 0 < b) (hx_pos : 0 < x) (hy_pos : 0 < y)
variable (h1 : a * x ≤ 5) (h2 : a * y ≤ 10) (h3 : b * x ≤ 10) (h4 : b * y ≤ 10)

theorem sum_le_30 : a * x + a * y + b * x + b * y ≤ 30 := sorry

end sum_le_30_l60_60775


namespace harry_travel_time_l60_60283

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l60_60283


namespace arithmetic_sequence_m_value_l60_60719

theorem arithmetic_sequence_m_value (S : ℕ → ℤ) (m : ℕ) 
  (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) 
  (h_seq : ∀ n : ℕ, S n = (n + 1) / 2 * (2 * a₁ + n * d)) :
  m = 5 :=
by
  sorry

end arithmetic_sequence_m_value_l60_60719


namespace calc_1_calc_2_l60_60385

variable (x y : ℝ)

theorem calc_1 : (-x^2)^4 = x^8 := 
sorry

theorem calc_2 : (-x^2 * y)^3 = -x^6 * y^3 := 
sorry

end calc_1_calc_2_l60_60385


namespace harry_total_travel_time_l60_60290

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l60_60290


namespace perpendicular_lines_foot_of_perpendicular_l60_60575

theorem perpendicular_lines_foot_of_perpendicular 
  (m n p : ℝ) 
  (h1 : 2 * 2 + 3 * p - 1 = 0)
  (h2 : 3 * 2 - 2 * p + n = 0)
  (h3 : - (2 / m) * (3 / 2) = -1) 
  : p - m - n = 4 := 
by
  sorry

end perpendicular_lines_foot_of_perpendicular_l60_60575


namespace exists_distinct_numbers_satisfy_conditions_l60_60963

theorem exists_distinct_numbers_satisfy_conditions :
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a + b + c = 6) ∧
  (2 * b = a + c) ∧
  ((b^2 = a * c) ∨ (a^2 = b * c) ∨ (c^2 = a * b)) :=
by
  sorry

end exists_distinct_numbers_satisfy_conditions_l60_60963


namespace value_of_f_3_div_2_l60_60618

noncomputable def f : ℝ → ℝ := sorry

axiom periodic_f : ∀ x : ℝ, f (x + 2) = f x
axiom even_f : ∀ x : ℝ, f (x) = f (-x)
axiom f_in_0_1 : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1 → f (x) = x + 1

theorem value_of_f_3_div_2 : f (3 / 2) = 3 / 2 := by
  sorry

end value_of_f_3_div_2_l60_60618


namespace shortest_chord_line_l60_60132

theorem shortest_chord_line (x y : ℝ) (P : (ℝ × ℝ)) (C : ℝ → ℝ → Prop) (h₁ : C x y) (hx : P = (1, 1)) (hC : ∀ x y, C x y ↔ x^2 + y^2 = 4) : 
  ∃ a b c : ℝ, a = 1 ∧ b = 1 ∧ c = -2 ∧ a * x + b * y + c = 0 :=
by
  sorry

end shortest_chord_line_l60_60132


namespace quadratic_roots_property_l60_60169

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l60_60169


namespace radius_of_circle_l60_60027

theorem radius_of_circle (r : ℝ) (h : 3 * (2 * real.pi * r) = real.pi * r^2) : r = 6 :=
by {
    sorry
}

end radius_of_circle_l60_60027


namespace quadratic_roots_property_l60_60171

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l60_60171


namespace work_done_together_in_one_day_l60_60208

-- Defining the conditions
def time_to_finish_a : ℕ := 12
def time_to_finish_b : ℕ := time_to_finish_a / 2

-- Defining the work done in one day
def work_done_by_a_in_one_day : ℚ := 1 / time_to_finish_a
def work_done_by_b_in_one_day : ℚ := 1 / time_to_finish_b

-- The proof statement
theorem work_done_together_in_one_day : 
  work_done_by_a_in_one_day + work_done_by_b_in_one_day = 1 / 4 := by
  sorry

end work_done_together_in_one_day_l60_60208


namespace angela_initial_action_figures_l60_60832

theorem angela_initial_action_figures (X : ℕ) (h1 : X - (1/4 : ℚ) * X - (1/3 : ℚ) * (3/4 : ℚ) * X = 12) : X = 24 :=
sorry

end angela_initial_action_figures_l60_60832


namespace ellipse_focal_distance_l60_60240

theorem ellipse_focal_distance : 
  ∀ (x y : ℝ), (x^2 / 36 + y^2 / 9 = 9) → (∃ c : ℝ, c = 2 * Real.sqrt 3) :=
by
  sorry

end ellipse_focal_distance_l60_60240


namespace completing_square_l60_60061

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l60_60061


namespace total_yen_l60_60826

/-- 
Abe's family has a checking account with 6359 yen
and a savings account with 3485 yen.
-/
def checking_account : ℕ := 6359
def savings_account : ℕ := 3485

/-- 
Prove that the total amount of yen Abe's family has
is equal to 9844 yen.
-/
theorem total_yen : checking_account + savings_account = 9844 :=
by
  sorry

end total_yen_l60_60826


namespace perfect_square_base9_last_digit_l60_60946

-- We define the problem conditions
variable {b d f : ℕ} -- all variables are natural numbers
-- Condition 1: Base 9 representation of a perfect square
variable (n : ℕ) -- n is the perfect square number
variable (sqrt_n : ℕ) -- sqrt_n is the square root of n (so, n = sqrt_n^2)
variable (h1 : n = b * 9^3 + d * 9^2 + 4 * 9 + f)
variable (h2 : b ≠ 0)
-- The question becomes that the possible values of f are 0, 1, or 4
theorem perfect_square_base9_last_digit (h3 : n = sqrt_n^2) (hb : b ≠ 0) : 
  (f = 0) ∨ (f = 1) ∨ (f = 4) :=
by
  sorry

end perfect_square_base9_last_digit_l60_60946


namespace find_number_l60_60397

theorem find_number (x : ℤ) (h : x * 9999 = 806006795) : x = 80601 :=
sorry

end find_number_l60_60397


namespace min_x_squared_plus_y_squared_l60_60157

theorem min_x_squared_plus_y_squared (x y : ℝ) (h : (x + 5) * (y - 5) = 0) : x^2 + y^2 ≥ 50 := by
  sorry

end min_x_squared_plus_y_squared_l60_60157


namespace teamA_teamB_repair_eq_l60_60213

-- conditions
def teamADailyRepair (x : ℕ) := x -- represent Team A repairing x km/day
def teamBDailyRepair (x : ℕ) := x + 3 -- represent Team B repairing x + 3 km/day
def timeTaken (distance rate: ℕ) := distance / rate -- time = distance / rate

-- Proof problem statement
theorem teamA_teamB_repair_eq (x : ℕ) (hx : x > 0) (hx_plus_3 : x + 3 > 0) :
  timeTaken 6 (teamADailyRepair x) = timeTaken 8 (teamBDailyRepair x) → (6 / x = 8 / (x + 3)) :=
by
  intros h
  sorry

end teamA_teamB_repair_eq_l60_60213


namespace johnny_closed_days_l60_60609

theorem johnny_closed_days :
  let dishes_per_day := 40
  let pounds_per_dish := 1.5
  let price_per_pound := 8
  let weekly_expenditure := 1920
  let daily_pounds := dishes_per_day * pounds_per_dish
  let daily_cost := daily_pounds * price_per_pound
  let days_open := weekly_expenditure / daily_cost
  let days_in_week := 7
  let days_closed := days_in_week - days_open
  days_closed = 3 :=
by
  sorry

end johnny_closed_days_l60_60609


namespace solve_for_x_l60_60268

theorem solve_for_x (x : ℚ) : ((1/3 - x) ^ 2 = 4) → (x = -5/3 ∨ x = 7/3) :=
by
  sorry

end solve_for_x_l60_60268


namespace find_a_l60_60133

open Set

theorem find_a (a : ℝ) (A : Set ℝ) (B : Set ℝ)
  (hA : A = {1, 2})
  (hB : B = {-a, a^2 + 3})
  (hUnion : A ∪ B = {1, 2, 4}) :
  a = -1 :=
sorry

end find_a_l60_60133


namespace even_function_implies_a_eq_2_l60_60549

def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2
  (h : ∀ x : ℝ, f a x = f a (-x)) : a = 2 := 
sorry

end even_function_implies_a_eq_2_l60_60549


namespace george_total_blocks_l60_60266

-- Definitions (conditions).
def large_boxes : ℕ := 5
def small_boxes_per_large_box : ℕ := 8
def blocks_per_small_box : ℕ := 9
def individual_blocks : ℕ := 6

-- Mathematical proof problem statement.
theorem george_total_blocks :
  (large_boxes * small_boxes_per_large_box * blocks_per_small_box + individual_blocks) = 366 :=
by
  -- Placeholder for proof.
  sorry

end george_total_blocks_l60_60266


namespace distance_between_cities_A_B_l60_60653

-- Define the problem parameters
def train_1_speed : ℝ := 60 -- km/hr
def train_2_speed : ℝ := 75 -- km/hr
def start_time_train_1 : ℝ := 8 -- 8 a.m.
def start_time_train_2 : ℝ := 9 -- 9 a.m.
def meeting_time : ℝ := 12 -- 12 p.m.

-- Define the times each train travels
def hours_train_1_travelled := meeting_time - start_time_train_1
def hours_train_2_travelled := meeting_time - start_time_train_2

-- Calculate the distances covered by each train
def distance_train_1_cover := train_1_speed * hours_train_1_travelled
def distance_train_2_cover := train_2_speed * hours_train_2_travelled

-- Define the total distance between cities A and B
def distance_AB := distance_train_1_cover + distance_train_2_cover

-- The theorem to be proved
theorem distance_between_cities_A_B : distance_AB = 465 := 
  by
    -- placeholder for the proof
    sorry

end distance_between_cities_A_B_l60_60653


namespace min_chord_length_l60_60424

variable (α : ℝ)

def curve_eq (x y α : ℝ) :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line_eq (x : ℝ) :=
  x = Real.pi / 4

theorem min_chord_length :
  ∃ d, (∀ α : ℝ, ∃ y1 y2 : ℝ, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ d = |y2 - y1|) ∧
  (∀ α : ℝ, ∃ y1 y2, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ |y2 - y1| ≥ d) :=
sorry

end min_chord_length_l60_60424


namespace two_digit_primes_units_digit_3_count_l60_60906

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l60_60906


namespace min_real_roots_l60_60314

open Polynomial

-- Define the polynomial conditions
variables (g : Polynomial ℝ)
hyp : g.natDegree = 2010
roots : Multiset ℕ := g.roots.map abs_val.to_nat

-- Translate problem to Lean 4 statement
theorem min_real_roots (g : Polynomial ℝ) (h_coeff : g.natDegree = 2010)
  (dist_vals : Multiset.card (roots g) = 1008) : 
  ∃ (n : ℕ), (roots g).count(ℕ) ≥ 6 :=
sorry

end min_real_roots_l60_60314


namespace not_always_greater_quotient_l60_60229

theorem not_always_greater_quotient (a : ℝ) (b : ℝ) (ha : a ≠ 0) (hb : 0 < b) : ¬ (∀ b < 1, a / b > a) ∧ ¬ (∀ b > 1, a / b > a) :=
by sorry

end not_always_greater_quotient_l60_60229


namespace favorable_probability_l60_60764

noncomputable def probability_favorable_events (L : ℝ) : ℝ :=
  1 - (0.5 * (5 / 12 * L)^2 / (0.5 * L^2))

theorem favorable_probability (L : ℝ) (x y : ℝ)
  (h1 : 0 ≤ x) (h2 : x ≤ L)
  (h3 : 0 ≤ y) (h4 : y ≤ L)
  (h5 : 0 ≤ x + y) (h6 : x + y ≤ L)
  (h7 : x ≤ 5 / 12 * L) (h8 : y ≤ 5 / 12 * L)
  (h9 : x + y ≥ 7 / 12 * L) :
  probability_favorable_events L = 15 / 16 :=
by sorry

end favorable_probability_l60_60764


namespace investment_ratio_proof_l60_60825

noncomputable def investment_ratio {A_invest B_invest C_invest : ℝ} (profit total_profit : ℝ) (A_times_B : ℝ) : ℝ :=
  C_invest / (A_times_B * B_invest + B_invest + C_invest)

theorem investment_ratio_proof (A_invest B_invest C_invest : ℝ)
  (profit total_profit : ℝ) (A_times_B : ℝ) 
  (h_profit : total_profit = 55000)
  (h_C_share : profit = 15000.000000000002)
  (h_A_times_B : A_times_B = 3)
  (h_ratio_eq : A_times_B * B_invest + B_invest + C_invest = 11 * B_invest / 3) :
  (A_invest / C_invest = 2) :=
by
  sorry

end investment_ratio_proof_l60_60825


namespace problem1_l60_60490

theorem problem1 (f : ℚ → ℚ) (a : Fin 7 → ℚ) (h₁ : ∀ x, f x = (1 - 3 * x) * (1 + x) ^ 5)
  (h₂ : ∀ x, f x = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + a 5 * x^5 + a 6 * x^6) :
  a 0 + (1/3) * a 1 + (1/3^2) * a 2 + (1/3^3) * a 3 + (1/3^4) * a 4 + (1/3^5) * a 5 + (1/3^6) * a 6 = 
  (1 - 3 * (1/3)) * (1 + (1/3))^5 :=
by sorry

end problem1_l60_60490


namespace teal_total_sales_l60_60306

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end teal_total_sales_l60_60306


namespace harry_travel_time_l60_60287

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l60_60287


namespace regular_polygon_sides_l60_60506

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l60_60506


namespace num_two_digit_prime_numbers_with_units_digit_3_l60_60910

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l60_60910


namespace linear_equation_solution_l60_60954

theorem linear_equation_solution (m n : ℤ) (x y : ℤ)
  (h1 : x + 2 * y = 5)
  (h2 : x + y = 7)
  (h3 : x = -m)
  (h4 : y = -n) :
  (3 * m + 2 * n) / (5 * m - n) = 11 / 14 :=
by
  sorry

end linear_equation_solution_l60_60954


namespace count_two_digit_primes_with_units_digit_three_l60_60920

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l60_60920


namespace capacity_of_each_bag_is_approximately_63_l60_60035

noncomputable def capacity_of_bag (total_sand : ℤ) (num_bags : ℤ) : ℤ :=
  Int.ceil (total_sand / num_bags)

theorem capacity_of_each_bag_is_approximately_63 :
  capacity_of_bag 757 12 = 63 :=
by
  sorry

end capacity_of_each_bag_is_approximately_63_l60_60035


namespace number_of_two_digit_primes_with_units_digit_three_l60_60933

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l60_60933


namespace part1_k_real_part2_find_k_l60_60717

-- Part 1: Discriminant condition
theorem part1_k_real (k : ℝ) (h : x^2 + (2*k - 1)*x + k^2 - 1 = 0) : k ≤ 5 / 4 :=
by
  sorry

-- Part 2: Given additional conditions, find k
theorem part2_find_k (x1 x2 k : ℝ) (h_eq : x^2 + (2 * k - 1) * x + k^2 - 1 = 0)
  (h1 : x1 + x2 = 1 - 2 * k) (h2 : x1 * x2 = k^2 - 1) (h3 : x1^2 + x2^2 = 16 + x1 * x2) : k = -2 :=
by
  sorry

end part1_k_real_part2_find_k_l60_60717


namespace lineD_is_parallel_to_line1_l60_60689

-- Define the lines
def line1 (x y : ℝ) := x - 2 * y + 1 = 0
def lineA (x y : ℝ) := 2 * x - y + 1 = 0
def lineB (x y : ℝ) := 2 * x - 4 * y + 2 = 0
def lineC (x y : ℝ) := 2 * x + 4 * y + 1 = 0
def lineD (x y : ℝ) := 2 * x - 4 * y + 1 = 0

-- Define a function to check parallelism between lines
def are_parallel (f g : ℝ → ℝ → Prop) :=
  ∀ x y : ℝ, (f x y → g x y) ∨ (g x y → f x y)

-- Prove that lineD is parallel to line1
theorem lineD_is_parallel_to_line1 : are_parallel line1 lineD :=
by
  sorry

end lineD_is_parallel_to_line1_l60_60689


namespace find_x0_l60_60622

-- Defining the function f
def f (a c x : ℝ) : ℝ := a * x^2 + c

-- Defining the integral condition
def integral_condition (a c x0 : ℝ) : Prop :=
  (∫ x in (0 : ℝ)..(1 : ℝ), f a c x) = f a c x0

-- Proving the main statement
theorem find_x0 (a c x0 : ℝ) (h : a ≠ 0) (h_range : 0 ≤ x0 ∧ x0 ≤ 1) (h_integral : integral_condition a c x0) :
  x0 = Real.sqrt (1 / 3) :=
by
  sorry

end find_x0_l60_60622


namespace find_n_positive_integers_l60_60122

theorem find_n_positive_integers :
  ∀ n : ℕ, 0 < n →
  (∃ k : ℕ, (n^2 + 11 * n - 4) * n! + 33 * 13^n + 4 = k^2) ↔ n = 1 ∨ n = 2 :=
by
  sorry

end find_n_positive_integers_l60_60122


namespace probability_both_selected_l60_60665

theorem probability_both_selected (p_ram : ℚ) (p_ravi : ℚ) (h_ram : p_ram = 5/7) (h_ravi : p_ravi = 1/5) : 
  (p_ram * p_ravi = 1/7) := 
by
  sorry

end probability_both_selected_l60_60665


namespace maximize_k_l60_60404

open Real

theorem maximize_k (x y : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : log x + log y = 0)
  (h₄ : ∀ x y : ℝ, 0 < x → 0 < y → k * (x + 2 * y) ≤ x^2 + 4 * y^2) : k ≤ sqrt 2 :=
sorry

end maximize_k_l60_60404


namespace find_number_of_small_branches_each_branch_grows_l60_60846

theorem find_number_of_small_branches_each_branch_grows :
  ∃ x : ℕ, 1 + x + x^2 = 43 ∧ x = 6 :=
by {
  sorry
}

end find_number_of_small_branches_each_branch_grows_l60_60846


namespace sum_of_circle_center_coordinates_l60_60024

open Real

theorem sum_of_circle_center_coordinates :
  let x1 := 5
  let y1 := 3
  let x2 := -7
  let y2 := 9
  let x_m := (x1 + x2) / 2
  let y_m := (y1 + y2) / 2
  x_m + y_m = 5 := by
  sorry

end sum_of_circle_center_coordinates_l60_60024


namespace correct_transformation_of_95_sq_l60_60350

theorem correct_transformation_of_95_sq : 95^2 = 100^2 - 2 * 100 * 5 + 5^2 := by
  sorry

end correct_transformation_of_95_sq_l60_60350


namespace values_of_d_l60_60969

theorem values_of_d (a b c d : ℕ) 
  (h : (ad - 1) / (a + 1) + (bd - 1) / (b + 1) + (cd - 1) / (c + 1) = d) : 
  d = 1 ∨ d = 2 ∨ d = 3 := 
sorry

end values_of_d_l60_60969


namespace future_value_option_B_correct_l60_60323

noncomputable def future_value_option_B (p q : ℝ) : ℝ :=
  150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12

theorem future_value_option_B_correct (p q A₂ : ℝ) :
  A₂ = 150 * (1 + p / 100 / 2) ^ 6 * (1 + q / 100 / 4) ^ 12 →
  ∃ A₂, A₂ = future_value_option_B p q :=
by
  intro h
  use A₂
  exact h

end future_value_option_B_correct_l60_60323


namespace travel_time_l60_60118

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l60_60118


namespace baseball_cards_per_friend_l60_60781

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end baseball_cards_per_friend_l60_60781


namespace possible_values_l60_60532

theorem possible_values (a : ℝ) (h : a > 1) : ∃ (v : ℝ), (v = 5 ∨ v = 6 ∨ v = 7) ∧ (a + 4 / (a - 1) = v) :=
sorry

end possible_values_l60_60532


namespace inequality_proof_l60_60437

theorem inequality_proof 
  (a b c : ℝ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_cond : a + b + c + 3 * a * b * c ≥ (a * b)^2 + (b * c)^2 + (c * a)^2 + 3) :
  (a^3 + b^3 + c^3) / 3 ≥ (a * b * c + 2021) / 2022 :=
by 
  sorry

end inequality_proof_l60_60437


namespace prism_volume_l60_60785

noncomputable def volume_of_prism (x y z : ℝ) : ℝ :=
  x * y * z

theorem prism_volume (x y z : ℝ) (h1 : x * y = 40) (h2 : x * z = 50) (h3 : y * z = 100) :
  volume_of_prism x y z = 100 * Real.sqrt 2 :=
by
  sorry

end prism_volume_l60_60785


namespace solve_quadratic_l60_60638

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end solve_quadratic_l60_60638


namespace induction_transition_factor_l60_60201

theorem induction_transition_factor (k : ℕ) (h : k > 0) :
  let lhs_k := (list.range k).map (λ i, k + 1 + i).prod
  let lhs_k1 := (list.range (k+1)).map (λ i, k + 2 + i).prod
  lhs_k1 = (lhs_k * (2*k + 1) * (2*k + 2)) / (k + 1) :=
by
  sorry

end induction_transition_factor_l60_60201


namespace min_points_on_dodecahedron_min_points_on_icosahedron_l60_60480

-- Definitions for the dodecahedron problem
def dodecahedron_has_12_faces : Prop := true
def each_vertex_in_dodecahedron_belongs_to_3_faces : Prop := true

-- Proof statement for dodecahedron
theorem min_points_on_dodecahedron : dodecahedron_has_12_faces ∧ each_vertex_in_dodecahedron_belongs_to_3_faces → ∃ n, n = 4 :=
by
  sorry

-- Definitions for the icosahedron problem
def icosahedron_has_20_faces : Prop := true
def icosahedron_has_12_vertices : Prop := true
def each_vertex_in_icosahedron_belongs_to_5_faces : Prop := true
def vertices_of_icosahedron_grouped_into_6_pairs : Prop := true

-- Proof statement for icosahedron
theorem min_points_on_icosahedron : 
  icosahedron_has_20_faces ∧ icosahedron_has_12_vertices ∧ each_vertex_in_icosahedron_belongs_to_5_faces ∧ vertices_of_icosahedron_grouped_into_6_pairs → ∃ n, n = 6 :=
by
  sorry

end min_points_on_dodecahedron_min_points_on_icosahedron_l60_60480


namespace find_fraction_l60_60835

theorem find_fraction : 
  ∀ (x : ℚ), (120 - x * 125 = 45) → x = 3 / 5 :=
by
  intro x
  intro h
  sorry

end find_fraction_l60_60835


namespace juniors_in_sports_count_l60_60197

-- Definitions for given conditions
def total_students : ℕ := 500
def percent_juniors : ℝ := 0.40
def percent_juniors_in_sports : ℝ := 0.70

-- Definition to calculate the number of juniors
def number_juniors : ℕ := (percent_juniors * total_students : ℝ).toNat

-- Definition to calculate the number of juniors involved in sports
def number_juniors_in_sports : ℕ := (percent_juniors_in_sports * number_juniors : ℝ).toNat

-- Statement to prove the calculated number of juniors involved in sports
theorem juniors_in_sports_count : number_juniors_in_sports = 140 :=
sorry

end juniors_in_sports_count_l60_60197


namespace count_two_digit_primes_with_units_digit_three_l60_60921

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100
def units_digit_is_three (n : ℕ) : Prop := n % 10 = 3
def num_primes_with_units_digit_three : ℕ :=
  {n : ℕ | is_prime n ∧ is_two_digit n ∧ units_digit_is_three n}.card

theorem count_two_digit_primes_with_units_digit_three : num_primes_with_units_digit_three = 6 :=
by
  sorry

end count_two_digit_primes_with_units_digit_three_l60_60921


namespace point_distance_units_l60_60183

theorem point_distance_units (d : ℝ) (h : |d| = 4) : d = 4 ∨ d = -4 := 
sorry

end point_distance_units_l60_60183


namespace range_of_a_l60_60886

open Real

-- Definitions of the propositions p and q
def p (a : ℝ) : Prop := (2 - a > 0) ∧ (a + 1 > 0)

def discriminant (a : ℝ) : ℝ := 16 + 4 * a

def q (a : ℝ) : Prop := discriminant a ≥ 0

/--
Given propositions p and q defined above,
prove that the range of real number values for a 
such that ¬p ∧ q is true is
- 4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2
--/
theorem range_of_a (a : ℝ) : (¬ p a ∧ q a) → (-4 ≤ a ∧ a ≤ -1 ∨ a ≥ 2) :=
by
  sorry

end range_of_a_l60_60886


namespace length_of_second_train_is_319_95_l60_60815

noncomputable def length_of_second_train (length_first_train : ℝ) (speed_first_train_kph : ℝ) (speed_second_train_kph : ℝ) (time_to_cross_seconds : ℝ) : ℝ :=
  let speed_first_train_mps := speed_first_train_kph * 1000 / 3600
  let speed_second_train_mps := speed_second_train_kph * 1000 / 3600
  let relative_speed := speed_first_train_mps + speed_second_train_mps
  let total_distance_covered := relative_speed * time_to_cross_seconds
  let length_second_train := total_distance_covered - length_first_train
  length_second_train

theorem length_of_second_train_is_319_95 :
  length_of_second_train 180 120 80 9 = 319.95 :=
sorry

end length_of_second_train_is_319_95_l60_60815


namespace electronics_sale_negation_l60_60156

variables (E : Type) (storeElectronics : E → Prop) (onSale : E → Prop)

theorem electronics_sale_negation
  (H : ¬ ∀ e, storeElectronics e → onSale e) :
  (∃ e, storeElectronics e ∧ ¬ onSale e) ∧ ¬ ∀ e, storeElectronics e → onSale e :=
by
  -- Proving that at least one electronic is not on sale follows directly from the negation of the universal statement
  sorry

end electronics_sale_negation_l60_60156


namespace find_a_over_b_l60_60391

variable (x y z a b : ℝ)
variable (h₁ : 4 * x - 2 * y + z = a)
variable (h₂ : 6 * y - 12 * x - 3 * z = b)
variable (h₃ : b ≠ 0)

theorem find_a_over_b : a / b = -1 / 3 :=
by
  sorry

end find_a_over_b_l60_60391


namespace probability_of_both_types_probability_distribution_and_expectation_of_X_l60_60112

-- Definitions
def total_zongzi : ℕ := 8
def red_bean_paste_zongzi : ℕ := 2
def date_zongzi : ℕ := 6
def selected_zongzi : ℕ := 3

-- Part 1: The probability of selecting both red bean paste and date zongzi
theorem probability_of_both_types :
  let total_combinations := Nat.choose total_zongzi selected_zongzi
  let one_red_two_date := Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2
  let two_red_one_date := Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1
  (one_red_two_date + two_red_one_date) / total_combinations = 9 / 14 :=
by sorry

-- Part 2: The probability distribution and expectation of X
theorem probability_distribution_and_expectation_of_X :
  let P_X_0 := (Nat.choose red_bean_paste_zongzi 0 * Nat.choose date_zongzi 3) / Nat.choose total_zongzi selected_zongzi
  let P_X_1 := (Nat.choose red_bean_paste_zongzi 1 * Nat.choose date_zongzi 2) / Nat.choose total_zongzi selected_zongzi
  let P_X_2 := (Nat.choose red_bean_paste_zongzi 2 * Nat.choose date_zongzi 1) / Nat.choose total_zongzi selected_zongzi
  P_X_0 = 5 / 14 ∧ P_X_1 = 15 / 28 ∧ P_X_2 = 3 / 28 ∧
  (0 * P_X_0 + 1 * P_X_1 + 2 * P_X_2 = 3 / 4) :=
by sorry

end probability_of_both_types_probability_distribution_and_expectation_of_X_l60_60112


namespace monotone_increasing_interval_for_shifted_function_l60_60726

variable (f : ℝ → ℝ)

-- Given definition: f(x+1) is an even function
def even_function : Prop :=
  ∀ x, f (x+1) = f (-(x+1))

-- Given condition: f(x+1) is monotonically decreasing on [0, +∞)
def monotone_decreasing_on_nonneg : Prop :=
  ∀ x y, 0 ≤ x → 0 ≤ y → x ≤ y → f (x+1) ≥ f (y+1)

-- Theorem to prove: the interval on which f(x-1) is monotonically increasing is (-∞, 2]
theorem monotone_increasing_interval_for_shifted_function
  (h_even : even_function f)
  (h_mono_dec : monotone_decreasing_on_nonneg f) :
  ∀ x y, x ≤ 2 → y ≤ 2 → x ≤ y → f (x-1) ≤ f (y-1) :=
by
  sorry

end monotone_increasing_interval_for_shifted_function_l60_60726


namespace hyperbola_eccentricity_l60_60577

noncomputable def point_on_hyperbola (x y a b : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) : Prop :=
  (x^2 / a^2) - (y^2 / b^2) = 1

noncomputable def focal_length (a b c : ℝ) : Prop :=
  2 * c = 4

noncomputable def eccentricity (e c a : ℝ) : Prop :=
  e = c / a

theorem hyperbola_eccentricity 
  (a b c e : ℝ)
  (h_pos_a : a > 0)
  (h_pos_b : b > 0)
  (h_point_on_hyperbola : point_on_hyperbola 2 3 a b h_pos_a h_pos_b)
  (h_focal_length : focal_length a b c)
  : eccentricity e c a :=
sorry -- proof omitted

end hyperbola_eccentricity_l60_60577


namespace regular_polygon_sides_l60_60496

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l60_60496


namespace subtraction_problem_digits_sum_l60_60604

theorem subtraction_problem_digits_sum :
  ∃ (K L M N : ℕ), K < 10 ∧ L < 10 ∧ M < 10 ∧ N < 10 ∧ 
  ((6000 + K * 100 + 0 + L) - (900 + N * 10 + 4) = 2011) ∧ 
  (K + L + M + N = 17) :=
by
  sorry

end subtraction_problem_digits_sum_l60_60604


namespace price_of_turban_correct_l60_60415

noncomputable def initial_yearly_salary : ℝ := 90
noncomputable def initial_monthly_salary : ℝ := initial_yearly_salary / 12
noncomputable def raise : ℝ := 0.05 * initial_monthly_salary

noncomputable def first_3_months_salary : ℝ := 3 * initial_monthly_salary
noncomputable def second_3_months_salary : ℝ := 3 * (initial_monthly_salary + raise)
noncomputable def third_3_months_salary : ℝ := 3 * (initial_monthly_salary + 2 * raise)

noncomputable def total_cash_salary : ℝ := first_3_months_salary + second_3_months_salary + third_3_months_salary
noncomputable def actual_cash_received : ℝ := 80
noncomputable def price_of_turban : ℝ := actual_cash_received - total_cash_salary

theorem price_of_turban_correct : price_of_turban = 9.125 :=
by
  sorry

end price_of_turban_correct_l60_60415


namespace right_triangle_condition_l60_60768

def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 4
  | 2 => 4
  | n + 3 => fib (n + 2) + fib (n + 1)

theorem right_triangle_condition (n : ℕ) : 
  ∃ a b c, a = fib n * fib (n + 4) ∧ 
           b = fib (n + 1) * fib (n + 3) ∧ 
           c = 2 * fib (n + 2) ∧
           a * a + b * b = c * c :=
by sorry

end right_triangle_condition_l60_60768


namespace flour_to_add_l60_60972

-- Define the conditions
def total_flour_required : ℕ := 9
def flour_already_added : ℕ := 2

-- Define the proof statement
theorem flour_to_add : total_flour_required - flour_already_added = 7 := 
by {
    sorry
}

end flour_to_add_l60_60972


namespace partiallyFilledBoxes_l60_60608

/-- Define the number of cards Joe collected -/
def numPokemonCards : Nat := 65
def numMagicCards : Nat := 55
def numYuGiOhCards : Nat := 40

/-- Define the number of cards each full box can hold -/
def pokemonBoxCapacity : Nat := 8
def magicBoxCapacity : Nat := 10
def yuGiOhBoxCapacity : Nat := 12

/-- Define the partially filled boxes for each type -/
def pokemonPartialBox : Nat := numPokemonCards % pokemonBoxCapacity
def magicPartialBox : Nat := numMagicCards % magicBoxCapacity
def yuGiOhPartialBox : Nat := numYuGiOhCards % yuGiOhBoxCapacity

/-- Theorem to prove number of cards in each partially filled box -/
theorem partiallyFilledBoxes :
  pokemonPartialBox = 1 ∧
  magicPartialBox = 5 ∧
  yuGiOhPartialBox = 4 :=
by
  -- proof goes here
  sorry

end partiallyFilledBoxes_l60_60608


namespace cat_weight_problem_l60_60238

variable (female_cat_weight male_cat_weight : ℕ)

theorem cat_weight_problem
  (h1 : male_cat_weight = 2 * female_cat_weight)
  (h2 : female_cat_weight + male_cat_weight = 6) :
  female_cat_weight = 2 :=
by
  sorry

end cat_weight_problem_l60_60238


namespace y_coord_equidistant_l60_60478

theorem y_coord_equidistant (y : ℝ) :
  (dist (0, y) (3, 0) = dist (0, y) (1, -6)) ↔ y = -7 / 3 :=
by
  sorry

end y_coord_equidistant_l60_60478


namespace estimate_y_value_l60_60729

theorem estimate_y_value : 
  ∀ (x : ℝ), x = 25 → 0.50 * x - 0.81 = 11.69 :=
by 
  intro x h
  rw [h]
  norm_num


end estimate_y_value_l60_60729


namespace sum_xyz_l60_60337

theorem sum_xyz (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + 2 * (y - 1) * (z - 1) = 85)
  (h2 : y^2 + 2 * (z - 1) * (x - 1) = 84)
  (h3 : z^2 + 2 * (x - 1) * (y - 1) = 89) :
  x + y + z = 18 := 
by
  sorry

end sum_xyz_l60_60337


namespace speed_of_second_train_40_kmph_l60_60475

noncomputable def length_train_1 : ℝ := 140
noncomputable def length_train_2 : ℝ := 160
noncomputable def crossing_time : ℝ := 10.799136069114471
noncomputable def speed_train_1 : ℝ := 60

theorem speed_of_second_train_40_kmph :
  let total_distance := length_train_1 + length_train_2
  let relative_speed_mps := total_distance / crossing_time
  let relative_speed_kmph := relative_speed_mps * 3.6
  let speed_train_2 := relative_speed_kmph - speed_train_1
  speed_train_2 = 40 :=
by
  sorry

end speed_of_second_train_40_kmph_l60_60475


namespace average_pastries_per_day_l60_60362

-- Conditions
def pastries_on_monday := 2

def pastries_on_day (n : ℕ) : ℕ :=
  pastries_on_monday + n

def total_pastries_in_week : ℕ :=
  List.sum (List.map pastries_on_day (List.range 7))

def number_of_days_in_week : ℕ := 7

-- Theorem to prove
theorem average_pastries_per_day : (total_pastries_in_week / number_of_days_in_week) = 5 :=
by
  sorry

end average_pastries_per_day_l60_60362


namespace avg_median_max_k_m_r_s_t_l60_60455

theorem avg_median_max_k_m_r_s_t (
  k m r s t : ℕ 
) (h1 : k < m) (h2 : m < r) (h3 : r < s) (h4 : s < t)
  (h5 : 5 * 16 = k + m + r + s + t)
  (h6 : r = 17) : 
  t = 42 :=
by
  sorry

end avg_median_max_k_m_r_s_t_l60_60455


namespace roots_eq_solution_l60_60172

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l60_60172


namespace salad_cost_is_correct_l60_60262

-- Definitions of costs according to the given conditions
def muffin_cost : ℝ := 2
def coffee_cost : ℝ := 4
def soup_cost : ℝ := 3
def lemonade_cost : ℝ := 0.75

def breakfast_cost : ℝ := muffin_cost + coffee_cost
def lunch_cost : ℝ := breakfast_cost + 3

def salad_cost : ℝ := lunch_cost - (soup_cost + lemonade_cost)

-- Statement to prove
theorem salad_cost_is_correct : salad_cost = 5.25 :=
by
  sorry

end salad_cost_is_correct_l60_60262


namespace box_dimension_min_sum_l60_60021

theorem box_dimension_min_sum :
  ∃ (a b c : ℕ), a * b * c = 2310 ∧ a + b + c = 42 := by
  sorry

end box_dimension_min_sum_l60_60021


namespace find_a_if_f_even_l60_60543

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem find_a_if_f_even (a : ℝ) (h : ∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) : a = 2 :=
by 
  sorry

end find_a_if_f_even_l60_60543


namespace voronovich_inequality_l60_60711

theorem voronovich_inequality (a b c : ℝ) (h₀ : 0 ≤ a) (h₁ : 0 ≤ b) (h₂ : 0 ≤ c) (h₃ : a + b + c = 1) :
  (a^2 + b^2 + c^2)^2 + 6 * a * b * c ≥ a * b + b * c + c * a :=
by
  sorry

end voronovich_inequality_l60_60711


namespace alice_marble_groups_l60_60515

-- Define the number of each colored marble Alice has
def pink_marble := 1
def blue_marble := 1
def white_marble := 1
def black_marbles := 4

-- The function to count the number of different groups of two marbles Alice can choose
noncomputable def count_groups : Nat :=
  let total_colors := 4  -- Pink, Blue, White, and one representative black
  1 + (total_colors.choose 2)

-- The theorem statement 
theorem alice_marble_groups : count_groups = 7 := by 
  sorry

end alice_marble_groups_l60_60515


namespace two_digit_primes_with_units_digit_three_count_l60_60929

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l60_60929


namespace correct_relationship_in_triangle_l60_60002

theorem correct_relationship_in_triangle (A B C : ℝ) (h : A + B + C = Real.pi) :
  Real.sin (A + B) = Real.sin C :=
sorry

end correct_relationship_in_triangle_l60_60002


namespace smallest_x_no_triangle_l60_60232

def triangle_inequality_violated (a b c : ℝ) : Prop :=
a + b ≤ c ∨ a + c ≤ b ∨ b + c ≤ a

theorem smallest_x_no_triangle (x : ℕ) (h : ∀ x, triangle_inequality_violated (7 - x : ℝ) (24 - x : ℝ) (26 - x : ℝ)) : x = 5 :=
sorry

end smallest_x_no_triangle_l60_60232


namespace part1_l60_60141

open Set

variable (U : Set ℝ) (A : Set ℝ) (B : Set ℝ)

theorem part1 (U_eq : U = univ) 
  (A_eq : A = {x | (x - 5) / (x - 2) ≤ 0}) 
  (B_eq : B = {x | 1 < x ∧ x < 3}) :
  compl A ∩ compl B = {x | x ≤ 1 ∨ x > 5} := 
  sorry

end part1_l60_60141


namespace ram_account_balance_increase_l60_60340

theorem ram_account_balance_increase 
  (initial_deposit : ℕ := 500)
  (first_year_balance : ℕ := 600)
  (second_year_percentage_increase : ℕ := 32)
  (second_year_balance : ℕ := initial_deposit + initial_deposit * second_year_percentage_increase / 100) 
  (second_year_increase : ℕ := second_year_balance - first_year_balance) 
  : (second_year_increase * 100 / first_year_balance) = 10 := 
sorry

end ram_account_balance_increase_l60_60340


namespace min_chord_length_l60_60225

-- Definitions of the problem conditions
def circle_center : ℝ × ℝ := (2, 3)
def circle_radius : ℝ := 3
def point_P : ℝ × ℝ := (1, 1)

-- The mathematical statement to prove
theorem min_chord_length : 
  ∀ (A B : ℝ × ℝ), 
  (A ≠ B) ∧ ((A.1 - 2)^2 + (A.2 - 3)^2 = 9) ∧ ((B.1 - 2)^2 + (B.2 - 3)^2 = 9) ∧ 
  ((A.1 - 1) / (B.1 - 1) = (A.2 - 1) / (B.2 - 1)) → 
  dist A B ≥ 4 := 
sorry

end min_chord_length_l60_60225


namespace find_B_current_age_l60_60598

variable {A B C : ℕ}

theorem find_B_current_age (h1 : A + 10 = 2 * (B - 10))
                          (h2 : A = B + 7)
                          (h3 : C = (A + B) / 2) :
                          B = 37 := by
  sorry

end find_B_current_age_l60_60598


namespace sequences_of_lemon_recipients_l60_60624

theorem sequences_of_lemon_recipients :
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  total_sequences = 759375 :=
by
  let students := 15
  let days := 5
  let total_sequences := students ^ days
  have h : total_sequences = 759375 := by sorry
  exact h

end sequences_of_lemon_recipients_l60_60624


namespace henry_collected_points_l60_60959

def points_from_wins (wins : ℕ) : ℕ := wins * 5
def points_from_losses (losses : ℕ) : ℕ := losses * 2
def points_from_draws (draws : ℕ) : ℕ := draws * 3

def total_points (wins losses draws : ℕ) : ℕ := 
  points_from_wins wins + points_from_losses losses + points_from_draws draws

theorem henry_collected_points :
  total_points 2 2 10 = 44 := by
  -- The proof will go here
  sorry

end henry_collected_points_l60_60959


namespace find_x_squared_plus_y_squared_l60_60735

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l60_60735


namespace rhombus_area_l60_60092

def d1 : ℝ := 10
def d2 : ℝ := 30

theorem rhombus_area (d1 d2 : ℝ) : (d1 * d2) / 2 = 150 := by
  sorry

end rhombus_area_l60_60092


namespace parameterization_solution_l60_60023

/-- Proof problem statement:
  Given the line equation y = 3x - 11 and its parameterization representation,
  the ordered pair (s, h) that satisfies both conditions is (3, 15).
-/
theorem parameterization_solution : ∃ s h : ℝ, 
  (∀ t : ℝ, (∃ x y : ℝ, (x, y) = (s, -2) + t • (5, h)) ∧ y = 3 * x - 11) → 
  (s = 3 ∧ h = 15) :=
by
  -- introduce s and h 
  use 3
  use 15
  -- skip the proof
  sorry

end parameterization_solution_l60_60023


namespace three_digit_squares_div_by_4_count_l60_60150

theorem three_digit_squares_div_by_4_count : 
  (finset.card ((finset.filter (λ x, 
    x % 4 = 0) 
    (finset.image (λ n : ℕ, n * n) 
      (finset.range 32)).filter 
        (λ x, 100 ≤ x ∧ x < 1000))) = 11) := 
by 
  sorry

end three_digit_squares_div_by_4_count_l60_60150


namespace solve_inequality_system_l60_60997

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l60_60997


namespace students_selected_milk_l60_60833

theorem students_selected_milk
    (total_students : ℕ)
    (students_soda students_milk students_juice : ℕ)
    (soda_percentage : ℚ)
    (milk_percentage : ℚ)
    (juice_percentage : ℚ)
    (h1 : soda_percentage = 0.7)
    (h2 : milk_percentage = 0.2)
    (h3 : juice_percentage = 0.1)
    (h4 : students_soda = 84)
    (h5 : total_students = students_soda / soda_percentage)
    : students_milk = total_students * milk_percentage :=
by
    sorry

end students_selected_milk_l60_60833


namespace cost_of_bananas_and_cantaloupe_l60_60187

-- Define prices for different items
variables (a b c d e : ℝ)

-- Define the conditions as hypotheses
theorem cost_of_bananas_and_cantaloupe (h1 : a + b + c + d + e = 30)
    (h2 : d = 3 * a) (h3 : c = a - b) (h4 : e = a + b) :
    b + c = 5 := 
by 
  -- Initial proof setup
  sorry

end cost_of_bananas_and_cantaloupe_l60_60187


namespace range_of_a_l60_60348

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x > 1 → x + 1 / (x - 1) ≥ a) → a ≤ 3 :=
by
  intro h
  sorry

end range_of_a_l60_60348


namespace solve_linear_eq_l60_60829

theorem solve_linear_eq (x y : ℤ) : 2 * x + 3 * y = 0 ↔ (x, y) = (3, -2) := sorry

end solve_linear_eq_l60_60829


namespace negation_proposition_l60_60459

theorem negation_proposition :
  (¬ ∃ x : ℝ, x^3 + 5*x - 2 = 0) ↔ ∀ x : ℝ, x^3 + 5*x - 2 ≠ 0 :=
by sorry

end negation_proposition_l60_60459


namespace count_prime_units_digit_3_eq_6_l60_60924

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l60_60924


namespace find_old_weight_l60_60192

variable (avg_increase : ℝ) (num_persons : ℕ) (W_new : ℝ) (total_increase : ℝ) (W_old : ℝ)

theorem find_old_weight (h1 : avg_increase = 3.5) 
                        (h2 : num_persons = 7) 
                        (h3 : W_new = 99.5) 
                        (h4 : total_increase = num_persons * avg_increase) 
                        (h5 : W_new = W_old + total_increase) 
                        : W_old = 75 :=
by
  sorry

end find_old_weight_l60_60192


namespace proof_statements_imply_negation_l60_60110

-- Define propositions p, q, and r
variables (p q r : Prop)

-- Statement (1): p, q, and r are all true.
def statement_1 : Prop := p ∧ q ∧ r

-- Statement (2): p is true, q is false, and r is true.
def statement_2 : Prop := p ∧ ¬ q ∧ r

-- Statement (3): p is false, q is true, and r is false.
def statement_3 : Prop := ¬ p ∧ q ∧ ¬ r

-- Statement (4): p and r are false, q is true.
def statement_4 : Prop := ¬ p ∧ q ∧ ¬ r

-- The negation of "p and q are true, and r is false" is "¬(p ∧ q) ∨ r"
def negation : Prop := ¬(p ∧ q) ∨ r

-- Proof statement that each of the 4 statements implies the negation
theorem proof_statements_imply_negation :
  (statement_1 p q r → negation p q r) ∧
  (statement_2 p q r → negation p q r) ∧
  (statement_3 p q r → negation p q r) ∧
  (statement_4 p q r → negation p q r) :=
by
  sorry

end proof_statements_imply_negation_l60_60110


namespace satisfactory_fraction_l60_60161

-- Define the number of each grade
def num_A : Nat := 7
def num_B : Nat := 6
def num_C : Nat := 5
def num_D : Nat := 4
def num_satisfactory : Nat := num_A + num_B + num_C + num_D

-- Define the number of unsatisfactory grades (combined E's and F's)
def num_unsatisfactory : Nat := 8

-- Define the total number of students
def total_students : Nat := num_satisfactory + num_unsatisfactory

-- Prove that the fraction of satisfactory grades is 11/15
theorem satisfactory_fraction : (num_satisfactory : ℚ) / (total_students : ℚ) = 11 / 15 := by sorry

end satisfactory_fraction_l60_60161


namespace trigonometric_identity_l60_60590

-- Define the main theorem
theorem trigonometric_identity (α : ℝ) (h : 3 * Real.sin α + Real.cos α = 0) :
  1 / (Real.cos α ^ 2 + Real.sin (2 * α)) = 10 / 3 :=
by
  sorry

end trigonometric_identity_l60_60590


namespace similar_triangle_perimeter_l60_60102

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

def is_isosceles (T : Triangle) : Prop :=
  T.a = T.b ∨ T.a = T.c ∨ T.b = T.c

def similar_triangles (T1 T2 : Triangle) : Prop :=
  T1.a / T2.a = T1.b / T2.b ∧ T1.b / T2.b = T1.c / T2.c ∧ T1.a / T2.a = T1.c / T2.c

noncomputable def perimeter (T : Triangle) : ℝ :=
  T.a + T.b + T.c

theorem similar_triangle_perimeter
  (T1 T2 : Triangle)
  (T1_isosceles : is_isosceles T1)
  (T1_sides : T1.a = 7 ∧ T1.b = 7 ∧ T1.c = 12)
  (T2_similar : similar_triangles T1 T2)
  (T2_longest_side : T2.c = 30) :
  perimeter T2 = 65 :=
by
  sorry

end similar_triangle_perimeter_l60_60102


namespace num_two_digit_primes_with_units_digit_three_l60_60901

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l60_60901


namespace Ronaldinho_age_2018_l60_60627

variable (X : ℕ)

theorem Ronaldinho_age_2018 (h : X^2 = 2025) : X - (2025 - 2018) = 38 := by
  sorry

end Ronaldinho_age_2018_l60_60627


namespace problem_f_of_f_neg1_eq_neg1_l60_60009

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.logb 2 x

-- State the proposition to be proved
theorem problem_f_of_f_neg1_eq_neg1 : f (f (-1)) = -1 := by
  sorry

end problem_f_of_f_neg1_eq_neg1_l60_60009


namespace largest_prime_factor_of_891_l60_60203

theorem largest_prime_factor_of_891 : 
  ∃ p : ℕ, Nat.Prime p ∧ p ∣ 891 ∧ ∀ q : ℕ, Nat.Prime q ∧ q ∣ 891 → q ≤ p :=
by
  sorry

end largest_prime_factor_of_891_l60_60203


namespace inequality_system_solution_l60_60987

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60987


namespace intersection_M_N_l60_60882

open Set

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

theorem intersection_M_N :
  M ∩ N = {-1, 3} := 
sorry

end intersection_M_N_l60_60882


namespace find_other_endpoint_l60_60692

noncomputable def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ :=
((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)

theorem find_other_endpoint (O A B : ℝ × ℝ) 
    (hO : O = (2, 3)) 
    (hA : A = (-1, -1))
    (hMidpoint : O = midpoint A B) : 
    B = (5, 7) :=
sorry

end find_other_endpoint_l60_60692


namespace full_price_ticket_revenue_correct_l60_60089

-- Define the constants and assumptions
variables (f t : ℕ) (p : ℝ)

-- Total number of tickets sold
def total_tickets := (f + t = 180)

-- Total revenue from ticket sales
def total_revenue := (f * p + t * (p / 3) = 2600)

-- Full price ticket revenue
def full_price_revenue := (f * p = 975)

-- The theorem combines the above conditions to prove the correct revenue from full-price tickets
theorem full_price_ticket_revenue_correct :
  total_tickets f t →
  total_revenue f t p →
  full_price_revenue f p :=
by
  sorry

end full_price_ticket_revenue_correct_l60_60089


namespace find_overall_mean_score_l60_60012

variable (M N E : ℝ)
variable (m n e : ℝ)

theorem find_overall_mean_score :
  M = 85 → N = 75 → E = 65 →
  m / n = 4 / 5 → n / e = 3 / 2 →
  ((85 * m) + (75 * n) + (65 * e)) / (m + n + e) = 82 :=
by
  sorry

end find_overall_mean_score_l60_60012


namespace lock_settings_are_5040_l60_60688

def num_unique_settings_for_lock : ℕ := 10 * 9 * 8 * 7

theorem lock_settings_are_5040 : num_unique_settings_for_lock = 5040 :=
by
  sorry

end lock_settings_are_5040_l60_60688


namespace dr_jones_remaining_salary_l60_60252

theorem dr_jones_remaining_salary:
  let salary := 6000
  let house_rental := 640
  let food_expense := 380
  let electric_water_bill := (1/4) * salary
  let insurances := (1/5) * salary
  let taxes := (10/100) * salary
  let transportation := (3/100) * salary
  let emergency_costs := (2/100) * salary
  let total_expenses := house_rental + food_expense + electric_water_bill + insurances + taxes + transportation + emergency_costs
  let remaining_salary := salary - total_expenses
  remaining_salary = 1380 :=
by
  sorry

end dr_jones_remaining_salary_l60_60252


namespace determine_a_for_even_function_l60_60535

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l60_60535


namespace g_of_900_eq_34_l60_60168

theorem g_of_900_eq_34 (g : ℕ+ → ℝ) 
  (h_mul : ∀ x y : ℕ+, g (x * y) = g x + g y)
  (h_30 : g 30 = 17)
  (h_60 : g 60 = 21) :
  g 900 = 34 :=
sorry

end g_of_900_eq_34_l60_60168


namespace area_range_of_triangle_l60_60793

-- Defining the points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (0, -2)

-- Circle equation
def on_circle (P : ℝ × ℝ) : Prop :=
  (P.1 - 2) ^ 2 + P.2 ^ 2 = 2

-- Function to compute the area of triangle ABP
noncomputable def area_of_triangle (P : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 - P.1) * (B.2 - P.2) - (B.1 - P.1) * (A.2 - P.2))

-- The proof goal statement
theorem area_range_of_triangle (P : ℝ × ℝ) (hp : on_circle P) :
  2 ≤ area_of_triangle P ∧ area_of_triangle P ≤ 6 :=
sorry

end area_range_of_triangle_l60_60793


namespace value_of_a_l60_60945

theorem value_of_a (a : ℝ) (k : ℝ) (hA : -5 = k * 3) (hB : a = k * (-6)) : a = 10 :=
by
  sorry

end value_of_a_l60_60945


namespace a4_eq_12_l60_60570

-- Definitions of the sequences and conditions
def S (n : ℕ) : ℕ := 
  -- sum of the first n terms, initially undefined
  sorry  

def a (n : ℕ) : ℕ := 
  -- terms of the sequence, initially undefined
  sorry  

-- Given conditions
axiom a2_eq_3 : a 2 = 3
axiom Sn_recurrence : ∀ n ≥ 2, S (n + 1) = 2 * S n

-- Statement to prove
theorem a4_eq_12 : a 4 = 12 :=
  sorry

end a4_eq_12_l60_60570


namespace find_a_l60_60870

noncomputable def l1 (a : ℝ) (x y : ℝ) : ℝ := a * x + (a + 1) * y + 1
noncomputable def l2 (a : ℝ) (x y : ℝ) : ℝ := x + a * y + 2

def perp_lines (a : ℝ) : Prop :=
  let m1 := -a
  let m2 := -1 / a
  m1 * m2 = -1

theorem find_a (a : ℝ) : (perp_lines a) ↔ (a = 0 ∨ a = -2) := 
sorry

end find_a_l60_60870


namespace valid_digit_for_multiple_of_5_l60_60264

theorem valid_digit_for_multiple_of_5 (d : ℕ) (h : d < 10) : (45670 + d) % 5 = 0 ↔ d = 0 ∨ d = 5 :=
by
  sorry

end valid_digit_for_multiple_of_5_l60_60264


namespace complete_the_square_l60_60050

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l60_60050


namespace m_range_iff_four_distinct_real_roots_l60_60952

noncomputable def four_distinct_real_roots (m : ℝ) : Prop :=
∃ x1 x2 x3 x4 : ℝ, x1 ≠ x2 ∧ x1 ≠ x3 ∧ x1 ≠ x4 ∧ x2 ≠ x3 ∧ x2 ≠ x4 ∧ x3 ≠ x4 ∧
(x1^2 - 4 * |x1| + 5 = m) ∧
(x2^2 - 4 * |x2| + 5 = m) ∧
(x3^2 - 4 * |x3| + 5 = m) ∧
(x4^2 - 4 * |x4| + 5 = m)

theorem m_range_iff_four_distinct_real_roots (m : ℝ) :
  four_distinct_real_roots m ↔ 1 < m ∧ m < 5 :=
sorry

end m_range_iff_four_distinct_real_roots_l60_60952


namespace pears_left_l60_60164

theorem pears_left (jason_pears : ℕ) (keith_pears : ℕ) (mike_ate : ℕ) (total_pears : ℕ) (pears_left : ℕ) 
  (h1 : jason_pears = 46) 
  (h2 : keith_pears = 47) 
  (h3 : mike_ate = 12) 
  (h4 : total_pears = jason_pears + keith_pears) 
  (h5 : pears_left = total_pears - mike_ate) 
  : pears_left = 81 :=
by
  sorry

end pears_left_l60_60164


namespace find_x_squared_plus_y_squared_l60_60737

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l60_60737


namespace min_value_of_squared_sum_l60_60006

open Real

theorem min_value_of_squared_sum (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) :
  ∃ m, m = (x^2 + y^2 + z^2) ∧ m = 16 / 3 :=
by
  sorry

end min_value_of_squared_sum_l60_60006


namespace partition_impossible_l60_60386

def sum_of_list (l : List Int) : Int := l.foldl (· + ·) 0

theorem partition_impossible
  (l : List Int)
  (h : l = [-7, -4, -2, 3, 5, 9, 10, 18, 21, 33])
  (total_sum : Int := sum_of_list l)
  (target_diff : Int := 9) :
  ¬∃ (l1 l2 : List Int), 
    (l1 ++ l2 = l ∧ 
     sum_of_list l1 - sum_of_list l2 = target_diff ∧
     total_sum  = 86) := 
sorry

end partition_impossible_l60_60386


namespace abs_neg_one_third_l60_60648

theorem abs_neg_one_third : abs (- (1 / 3 : ℚ)) = 1 / 3 := 
by sorry

end abs_neg_one_third_l60_60648


namespace total_wet_surface_area_l60_60353

def cistern_length (L : ℝ) := L = 5
def cistern_width (W : ℝ) := W = 4
def water_depth (D : ℝ) := D = 1.25

theorem total_wet_surface_area (L W D A : ℝ) 
  (hL : cistern_length L) 
  (hW : cistern_width W) 
  (hD : water_depth D) :
  A = 42.5 :=
by
  subst hL
  subst hW
  subst hD
  sorry

end total_wet_surface_area_l60_60353


namespace determine_mass_l60_60477

noncomputable def mass_of_water 
  (P : ℝ) (t1 t2 : ℝ) (deltaT : ℝ) (cw : ℝ) : ℝ :=
  P * t1 / ((cw * deltaT) + ((cw * deltaT) / t2) * t1)

theorem determine_mass (P : ℝ) (t1 : ℝ) (deltaT : ℝ) (t2 : ℝ) (cw : ℝ) :
  P = 1000 → t1 = 120 → deltaT = 2 → t2 = 60 → cw = 4200 →
  mass_of_water P t1 deltaT t2 cw = 4.76 :=
by
  intros hP ht1 hdeltaT ht2 hcw
  sorry

end determine_mass_l60_60477


namespace correct_operation_l60_60074

theorem correct_operation (a : ℝ) : 2 * (a^2) * a = 2 * (a^3) := by sorry

end correct_operation_l60_60074


namespace p_sufficient_not_necessary_for_q_l60_60568

-- Define the propositions p and q
def is_ellipse (m : ℝ) : Prop := (1 / 4 < m) ∧ (m < 1)
def is_hyperbola (m : ℝ) : Prop := (0 < m) ∧ (m < 1)

-- Define the theorem to prove the relationship between p and q
theorem p_sufficient_not_necessary_for_q (m : ℝ) :
  (is_ellipse m → is_hyperbola m) ∧ ¬(is_hyperbola m → is_ellipse m) :=
sorry

end p_sufficient_not_necessary_for_q_l60_60568


namespace find_a_for_even_function_l60_60537

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l60_60537


namespace diff_of_two_numbers_l60_60331

theorem diff_of_two_numbers :
  ∃ D S : ℕ, (1650 = 5 * S + 5) ∧ (D = 1650 - S) ∧ (D = 1321) :=
sorry

end diff_of_two_numbers_l60_60331


namespace horse_problem_l60_60602

-- Definitions based on conditions:
def total_horses : ℕ := 100
def tiles_pulled_by_big_horse (x : ℕ) : ℕ := 3 * x
def tiles_pulled_by_small_horses (x : ℕ) : ℕ := (100 - x) / 3

-- The statement to prove:
theorem horse_problem (x : ℕ) : 
    tiles_pulled_by_big_horse x + tiles_pulled_by_small_horses x = 100 :=
sorry

end horse_problem_l60_60602


namespace findNumberOfIntegers_l60_60528

def arithmeticSeq (a d n : ℕ) : ℕ :=
  a + d * n

def isInSeq (n : ℕ) : Prop :=
  ∃ k : ℕ, k ≤ 33 ∧ n = arithmeticSeq 1 3 k

def validInterval (n : ℕ) : Bool :=
  (n + 1) / 3 % 2 = 1

theorem findNumberOfIntegers :
  ∃ count : ℕ, count = 66 ∧ (∀ n : ℕ, 1 ≤ n ∧ n ≤ 100 ∧ ¬isInSeq n → validInterval n = true) :=
sorry

end findNumberOfIntegers_l60_60528


namespace real_values_satisfying_inequality_l60_60704

theorem real_values_satisfying_inequality :
  ∀ x : ℝ, x ≠ 5 → (x * (x + 2)) / ((x - 5)^2) ≥ 15 ↔ x ∈ set.Iic 0.76 ∪ set.Ioo 5 10.1 := by
  sorry

end real_values_satisfying_inequality_l60_60704


namespace find_a_if_even_function_l60_60546

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l60_60546


namespace students_play_football_l60_60975

theorem students_play_football (total_students : ℕ) (C : ℕ) (B : ℕ) (neither : ℕ) (F : ℕ)
  (h1 : total_students = 460)
  (h2 : C = 175)
  (h3 : B = 90)
  (h4 : neither = 50)
  (h5 : total_students = neither + F + C - B) : 
  F = 325 :=
by 
  sorry

end students_play_football_l60_60975


namespace points_divisibility_l60_60966

theorem points_divisibility {k n : ℕ} (hkn : k ≤ n) (hpositive : 0 < n) 
  (hcondition : ∀ x : Fin n, (∃ m : ℕ, (∀ y : Fin n, x.val ≤ y.val → y.val ≤ x.val + 1 → True) ∧ m % k = 0)) :
  k ∣ n :=
sorry

end points_divisibility_l60_60966


namespace determine_f_3_2016_l60_60398

noncomputable def f : ℕ → ℕ → ℕ
| 0, y       => y + 1
| (x + 1), 0 => f x 1
| (x + 1), (y + 1) => f x (f (x + 1) y)

theorem determine_f_3_2016 : f 3 2016 = 2 ^ 2019 - 3 := by
  sorry

end determine_f_3_2016_l60_60398


namespace multiplication_problem_solution_l60_60001

theorem multiplication_problem_solution (a b c : ℕ) 
  (h1 : a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1) 
  (h2 : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h3 : (a * 100 + b * 10 + b) * c = b * 1000 + c * 100 + b * 10 + 1) : 
  a = 5 ∧ b = 3 ∧ c = 7 := 
sorry

end multiplication_problem_solution_l60_60001


namespace num_two_digit_prime_numbers_with_units_digit_3_l60_60912

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l60_60912


namespace mass_percentage_O_in_N2O3_l60_60394

variable (m_N : ℝ := 14.01)  -- Molar mass of nitrogen (N) in g/mol
variable (m_O : ℝ := 16.00)  -- Molar mass of oxygen (O) in g/mol
variable (n_N : ℕ := 2)      -- Number of nitrogen (N) atoms in N2O3
variable (n_O : ℕ := 3)      -- Number of oxygen (O) atoms in N2O3

theorem mass_percentage_O_in_N2O3 :
  let molar_mass_N2O3 := (n_N * m_N) + (n_O * m_O)
  let mass_O_in_N2O3 := n_O * m_O
  let percentage_O := (mass_O_in_N2O3 / molar_mass_N2O3) * 100
  percentage_O = 63.15 :=
by
  -- Formal proof here
  sorry

end mass_percentage_O_in_N2O3_l60_60394


namespace determine_a_for_even_function_l60_60534

theorem determine_a_for_even_function :
  ∃ a : ℝ, (∀ x : ℝ, x ≠ 0 → (∃ f : ℝ → ℝ, 
  f x = x * exp x / (exp (a * x) - 1) ∧
  (∀ x : ℝ, f (-x) = f x))) → a = 2 :=
by
  sorry

end determine_a_for_even_function_l60_60534


namespace sum_of_digits_of_N_l60_60233

theorem sum_of_digits_of_N :
  ∃ N : ℕ, (N * (N + 1)) / 2 = 3003 ∧ N.digits 10 = [7, 7] :=
by
  sorry

end sum_of_digits_of_N_l60_60233


namespace teal_total_sales_l60_60305

variable (pum_pie_slices_per_pie : ℕ) (cus_pie_slices_per_pie : ℕ)
variable (pum_pie_price_per_slice : ℕ) (cus_pie_price_per_slice : ℕ)
variable (pum_pies_sold : ℕ) (cus_pies_sold : ℕ)

def total_slices_sold (slices_per_pie pies_sold : ℕ) : ℕ :=
  slices_per_pie * pies_sold

def total_sales (slices_sold price_per_slice : ℕ) : ℕ :=
  slices_sold * price_per_slice

theorem teal_total_sales
  (h1 : pum_pie_slices_per_pie = 8)
  (h2 : cus_pie_slices_per_pie = 6)
  (h3 : pum_pie_price_per_slice = 5)
  (h4 : cus_pie_price_per_slice = 6)
  (h5 : pum_pies_sold = 4)
  (h6 : cus_pies_sold = 5) :
  (total_sales (total_slices_sold pum_pie_slices_per_pie pum_pies_sold) pum_pie_price_per_slice) +
  (total_sales (total_slices_sold cus_pie_slices_per_pie cus_pies_sold) cus_pie_price_per_slice) = 340 :=
by
  sorry

end teal_total_sales_l60_60305


namespace combined_ages_l60_60020

theorem combined_ages (h_age : ℕ) (diff : ℕ) (years_later : ℕ) (hurley_age : h_age = 14) 
                       (age_difference : diff = 20) (years_passed : years_later = 40) : 
                       h_age + diff + years_later * 2 = 128 := by
  sorry

end combined_ages_l60_60020


namespace sum_of_z_values_l60_60617

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x + 2

theorem sum_of_z_values (z : ℝ) : 
  (f (4 * z) = 13) → (∃ z1 z2 : ℝ, z1 = 1/8 ∧ z2 = -1/4 ∧ z1 + z2 = -1/8) :=
sorry

end sum_of_z_values_l60_60617


namespace total_molecular_weight_is_1317_12_l60_60845

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_S : ℝ := 32.06
def atomic_weight_H : ℝ := 1.01
def atomic_weight_O : ℝ := 16.00
def atomic_weight_C : ℝ := 12.01

def molecular_weight_Al2S3 : ℝ := (2 * atomic_weight_Al) + (3 * atomic_weight_S)
def molecular_weight_H2O : ℝ := (2 * atomic_weight_H) + (1 * atomic_weight_O)
def molecular_weight_CO2 : ℝ := (1 * atomic_weight_C) + (2 * atomic_weight_O)

def total_weight_7_Al2S3 : ℝ := 7 * molecular_weight_Al2S3
def total_weight_5_H2O : ℝ := 5 * molecular_weight_H2O
def total_weight_4_CO2 : ℝ := 4 * molecular_weight_CO2

def total_molecular_weight : ℝ := total_weight_7_Al2S3 + total_weight_5_H2O + total_weight_4_CO2

theorem total_molecular_weight_is_1317_12 : total_molecular_weight = 1317.12 := by
  sorry

end total_molecular_weight_is_1317_12_l60_60845


namespace mn_value_l60_60418

variables {x m n : ℝ} -- Define variables x, m, n as real numbers

theorem mn_value (h : x^2 + m * x - 15 = (x + 3) * (x + n)) : m * n = 10 :=
by {
  -- Sorry for skipping the proof steps
  sorry
}

end mn_value_l60_60418


namespace find_k_l60_60142

variables (a b : ℝ × ℝ)
variables (k : ℝ)

def vector_a : ℝ × ℝ := (1, 2)
def vector_b : ℝ × ℝ := (2, -1)

def k_a_plus_b (k : ℝ) : ℝ × ℝ := (k * vector_a.1 + vector_b.1, k * vector_a.2 + vector_b.2)
def a_minus_2b : ℝ × ℝ := (vector_a.1 - 2 * vector_b.1, vector_a.2 - 2 * vector_b.2)

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_k (k : ℝ) : dot_product (k_a_plus_b k) a_minus_2b = 0 ↔ k = 2 :=
by
  sorry

end find_k_l60_60142


namespace travel_time_l60_60116

/-- 
  We consider three docks A, B, and C. 
  The boat travels 3 km between docks.
  The travel must account for current (with the current and against the current).
  The time to travel over 3 km with the current is less than the time to travel 3 km against the current.
  Specific times for travel are given:
  - 30 minutes for 3 km against the current.
  - 18 minutes for 3 km with the current.
  
  Prove that the travel time between the docks can either be 24 minutes or 72 minutes.
-/
theorem travel_time (A B C : Type) (d : ℕ) (t_with_current t_against_current : ℕ) 
  (h_current : t_with_current < t_against_current)
  (h_t_with : t_with_current = 18) (h_t_against : t_against_current = 30) :
  d * t_with_current = 24 ∨ d * t_against_current = 72 := 
  sorry

end travel_time_l60_60116


namespace parabola_distance_l60_60947

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end parabola_distance_l60_60947


namespace minimal_pieces_required_for_cubes_l60_60343

theorem minimal_pieces_required_for_cubes 
  (e₁ e₂ n₁ n₂ n₃ : ℕ)
  (h₁ : e₁ = 14)
  (h₂ : e₂ = 10)
  (h₃ : n₁ = 13)
  (h₄ : n₂ = 11)
  (h₅ : n₃ = 6)
  (disassembly_possible : ∀ {x y z : ℕ}, x^3 + y^3 = z^3 → n₁^3 + n₂^3 + n₃^3 = 14^3 + 10^3)
  (cutting_constraints : ∀ d : ℕ, (d > 0) → (d ≤ e₁ ∨ d ≤ e₂) → (d ≤ n₁ ∨ d ≤ n₂ ∨ d ≤ n₃) → (d ≤ 6))
  : ∃ minimal_pieces : ℕ, minimal_pieces = 11 := 
sorry

end minimal_pieces_required_for_cubes_l60_60343


namespace population_net_change_l60_60823

theorem population_net_change
  (initial_population : ℝ)
  (year1_increase : initial_population * (6/5) = year1_population)
  (year2_increase : year1_population * (6/5) = year2_population)
  (year3_decrease : year2_population * (4/5) = year3_population)
  (year4_decrease : year3_population * (4/5) = final_population) :
  ((final_population - initial_population) / initial_population) * 100 = -8 :=
  sorry

end population_net_change_l60_60823


namespace n_squared_divides_2n_plus_1_l60_60702

theorem n_squared_divides_2n_plus_1 (n : ℕ) (hn : n > 0) :
  (n ^ 2) ∣ (2 ^ n + 1) ↔ (n = 1 ∨ n = 3) :=
by sorry

end n_squared_divides_2n_plus_1_l60_60702


namespace two_digit_primes_units_digit_3_count_l60_60904

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l60_60904


namespace all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l60_60220

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l60_60220


namespace equivalent_eq_l60_60153

variable {x y : ℝ}

theorem equivalent_eq (hx1 : x ≠ 0) (hx2 : x ≠ 3) (hy1 : y ≠ 0) (hy2 : y ≠ 5) :
  (3 / x + 2 / y = 1 / 3) ↔ (y = 6 * x / (x - 9)) :=
by
  sorry

end equivalent_eq_l60_60153


namespace sum_of_x_coords_f_eq_3_l60_60696

section
-- Define the piecewise linear function, splits into five segments
def f1 (x : ℝ) : ℝ := 2 * x + 6
def f2 (x : ℝ) : ℝ := -2 * x + 6
def f3 (x : ℝ) : ℝ := 2 * x + 2
def f4 (x : ℝ) : ℝ := -x + 2
def f5 (x : ℝ) : ℝ := 2 * x - 4

-- The sum of x-coordinates where f(x) = 3
noncomputable def x_coords_3_sum : ℝ := -1.5 + 0.5 + 3.5

-- Goal statement
theorem sum_of_x_coords_f_eq_3 : -1.5 + 0.5 + 3.5 = 2.5 := by
  sorry
end

end sum_of_x_coords_f_eq_3_l60_60696


namespace juniors_involved_in_sports_l60_60198

theorem juniors_involved_in_sports 
    (total_students : ℕ) (percentage_juniors : ℝ) (percentage_sports : ℝ) 
    (H1 : total_students = 500) 
    (H2 : percentage_juniors = 0.40) 
    (H3 : percentage_sports = 0.70) : 
    total_students * percentage_juniors * percentage_sports = 140 := 
by
  sorry

end juniors_involved_in_sports_l60_60198


namespace f_is_even_iff_a_is_2_l60_60539

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l60_60539


namespace two_digit_primes_units_digit_3_count_l60_60905

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def has_units_digit_3 (n : ℕ) : Prop :=
  n % 10 = 3

def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n < 100

theorem two_digit_primes_units_digit_3_count :
  {n : ℕ | is_two_digit n ∧ has_units_digit_3 n ∧ is_prime n}.to_finset.card = 6 :=
by
  sorry

end two_digit_primes_units_digit_3_count_l60_60905


namespace events_independent_l60_60766

variable {Ω : Type*} 
variable {P : MeasureTheory.Measure Ω}

theorem events_independent (A B : Set Ω) 
  (hA : P A = 0 ∨ P A = 1) 
  : P (A ∩ B) = P A * P B := sorry

end events_independent_l60_60766


namespace even_function_implies_a_eq_2_l60_60540

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l60_60540


namespace min_distance_l60_60574

variables {P Q : ℝ × ℝ}

def line (P : ℝ × ℝ) : Prop := 3 * P.1 + 4 * P.2 + 5 = 0
def circle (Q : ℝ × ℝ) : Prop := (Q.1 - 2) ^ 2 + (Q.2 - 2) ^ 2 = 4

theorem min_distance (P : ℝ × ℝ) (Q : ℝ × ℝ) (hP : line P) (hQ : circle Q) :
  ∃ d : ℝ, d = dist P Q ∧ d = 9 / 5 := sorry

end min_distance_l60_60574


namespace problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l60_60316

-- Problem 1: 1 / 0.25 = 4
theorem problem1 : 1 / 0.25 = 4 :=
by sorry

-- Problem 2: 0.25 / 0.1 = 2.5
theorem problem2 : 0.25 / 0.1 = 2.5 :=
by sorry

-- Problem 3: 1.2 / 1.2 = 1
theorem problem3 : 1.2 / 1.2 = 1 :=
by sorry

-- Problem 4: 4.01 * 1 = 4.01
theorem problem4 : 4.01 * 1 = 4.01 :=
by sorry

-- Problem 5: 0.25 * 2 = 0.5
theorem problem5 : 0.25 * 2 = 0.5 :=
by sorry

-- Problem 6: 0 / 2.76 = 0
theorem problem6 : 0 / 2.76 = 0 :=
by sorry

-- Problem 7: 0.8 / 1.25 = 0.64
theorem problem7 : 0.8 / 1.25 = 0.64 :=
by sorry

-- Problem 8: 3.5 * 2.7 = 9.45
theorem problem8 : 3.5 * 2.7 = 9.45 :=
by sorry

end problem1_problem2_problem3_problem4_problem5_problem6_problem7_problem8_l60_60316


namespace solve_percentage_of_X_in_B_l60_60011

variable (P : ℝ)

def liquid_X_in_A_percentage : ℝ := 0.008
def mass_of_A : ℝ := 200
def mass_of_B : ℝ := 700
def mixed_solution_percentage_of_X : ℝ := 0.0142
def target_percentage_of_P_in_B : ℝ := 0.01597

theorem solve_percentage_of_X_in_B (P : ℝ) 
  (h1 : mass_of_A * liquid_X_in_A_percentage + mass_of_B * P = (mass_of_A + mass_of_B) * mixed_solution_percentage_of_X) :
  P = target_percentage_of_P_in_B :=
sorry

end solve_percentage_of_X_in_B_l60_60011


namespace compare_fractions_l60_60693

theorem compare_fractions {x : ℝ} (h : 3 < x ∧ x < 4) : 
  (2 / 3) > ((5 - x) / 3) :=
by sorry

end compare_fractions_l60_60693


namespace n_minus_two_is_square_of_natural_number_l60_60467

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end n_minus_two_is_square_of_natural_number_l60_60467


namespace egg_cost_l60_60378

theorem egg_cost (toast_cost : ℝ) (E : ℝ) (total_cost : ℝ)
  (dales_toast : ℝ) (dales_eggs : ℝ) (andrews_toast : ℝ) (andrews_eggs : ℝ) :
  toast_cost = 1 → 
  dales_toast = 2 → 
  dales_eggs = 2 → 
  andrews_toast = 1 → 
  andrews_eggs = 2 → 
  total_cost = 15 →
  total_cost = (dales_toast * toast_cost + dales_eggs * E) + 
               (andrews_toast * toast_cost + andrews_eggs * E) →
  E = 3 :=
by
  sorry

end egg_cost_l60_60378


namespace count_two_digit_primes_with_units_digit_3_l60_60908

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l60_60908


namespace length_of_AB_is_1_l60_60576

variables {A B C : ℝ} -- Points defining the triangle vertices
variables {a b c : ℝ} -- Lengths of triangle sides opposite to angles A, B, C respectively
variables {α β γ : ℝ} -- Angles at points A B C
variables {s₁ s₂ s₃ : ℝ} -- Sin values of the angles

noncomputable def length_of_AB (a b c : ℝ) : ℝ :=
  if a + b + c = 4 ∧ a + b = 3 * c then 1 else 0

theorem length_of_AB_is_1 : length_of_AB a b c = 1 :=
by
  have h_perimeter : a + b + c = 4 := sorry
  have h_sin_condition : a + b = 3 * c := sorry
  simp [length_of_AB, h_perimeter, h_sin_condition]
  sorry

end length_of_AB_is_1_l60_60576


namespace Eddy_travel_time_l60_60253

theorem Eddy_travel_time :
  ∀ (T_F D_F D_E : ℕ) (S_ratio : ℝ),
    T_F = 4 →
    D_F = 360 →
    D_E = 600 →
    S_ratio = 2.2222222222222223 →
    ((D_F / T_F : ℝ) * S_ratio ≠ 0) →
    D_E / ((D_F / T_F) * S_ratio) = 3 :=
by
  intros T_F D_F D_E S_ratio ht hf hd hs hratio
  sorry  -- Proof to be provided

end Eddy_travel_time_l60_60253


namespace eulerian_path_exists_l60_60962

-- Define the graph structure

def figure_graph : SimpleGraph ℕ :=
  SimpleGraph.mk' { edges := { ⟨1, 2⟩, ⟨2, 3⟩, ⟨3, 4⟩, ⟨4, 1⟩, -- square 1
                               ⟨3, 5⟩, ⟨5, 6⟩, ⟨6, 1⟩, ⟨1, 2⟩, ⟨2, 3⟩, ⟨3, 6⟩, ⟨6, 5⟩, ⟨5, 4⟩},
                    sym := fun ⟨x, y⟩ h => by cases h; simp [edges] }

-- Define vertices for reference
def vertices : list ℕ := [1, 2, 3, 4, 5, 6]

-- This verifies the conditions stated in the problem; these should be demonstrated in the proof.
-- We must show that this graph satisfies the Eulerian path conditions, which requires proving the existence of such a path.

theorem eulerian_path_exists (G : SimpleGraph ℕ) (v : G.V) : 
  (∃ p : list G.V, G.IsEulerianPath p) ↔ count_odd_degree_neighbors G = 2 := sorry

end eulerian_path_exists_l60_60962


namespace dot_product_property_l60_60270

noncomputable def point_on_ellipse (x y : ℝ) : Prop :=
  (x^2 / 4) + (y^2 / 3) = 1

variables (x_P y_P : ℝ) (F1 F2 : ℝ × ℝ)

def is_focus (F : ℝ × ℝ) : Prop :=
  F = (1, 0) ∨ F = (-1, 0)

def radius_of_inscribed_circle (r : ℝ) : Prop :=
  r = 1 / 2

theorem dot_product_property (h1 : point_on_ellipse x_P y_P)
  (h2 : is_focus F1) (h3 : is_focus F2) (h4: radius_of_inscribed_circle (1/2)):
  (x_P^2 - 1 + y_P^2) = 9 / 4 :=
sorry

end dot_product_property_l60_60270


namespace find_relationship_l60_60272

theorem find_relationship (n m : ℕ) (a : ℚ) (h_pos_a : 0 < a) (h_pos_n : 0 < n) (h_pos_m : 0 < m) :
  (n > m ↔ (1 / n < a)) → m = ⌊1 / a⌋ :=
sorry

end find_relationship_l60_60272


namespace number_of_two_digit_primes_with_units_digit_three_l60_60931

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l60_60931


namespace train_crossing_time_approx_l60_60212

noncomputable def train_length : ℝ := 90 -- in meters
noncomputable def speed_kmh : ℝ := 124 -- in km/hr
noncomputable def conversion_factor : ℝ := 1000 / 3600 -- km/hr to m/s conversion factor
noncomputable def speed_ms : ℝ := speed_kmh * conversion_factor -- speed in m/s
noncomputable def time_to_cross : ℝ := train_length / speed_ms -- time in seconds

theorem train_crossing_time_approx :
  abs (time_to_cross - 2.61) < 0.01 := 
by 
  sorry

end train_crossing_time_approx_l60_60212


namespace total_wicks_l60_60100

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end total_wicks_l60_60100


namespace find_common_ratio_l60_60566

variable (a : ℕ → ℝ) -- represents the geometric sequence
variable (q : ℝ) -- represents the common ratio

-- conditions given in the problem
def a_3_condition : a 3 = 4 := sorry
def a_6_condition : a 6 = 1 / 2 := sorry

-- the general form of the geometric sequence
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a n = a 0 * q ^ n

-- the theorem we want to prove
theorem find_common_ratio (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 3 = 4) (h2 : a 6 = 1 / 2) 
  (hg : geometric_sequence a q) : q = 1 / 2 :=
sorry

end find_common_ratio_l60_60566


namespace count_five_digit_numbers_l60_60888

theorem count_five_digit_numbers :
  let digits := [3, 3, 3, 8, 8] in
  fintype.card {l : list ℕ // l.perm digits ∧ (∀ x ∈ l, x ∈ digits)} = 10 :=
by
  sorry

end count_five_digit_numbers_l60_60888


namespace g_symmetric_l60_60831

noncomputable def g (x : ℝ) : ℝ := |⌊2 * x⌋| - |⌊2 - 2 * x⌋|

theorem g_symmetric : ∀ x : ℝ, g x = g (1 - x) := by
  sorry

end g_symmetric_l60_60831


namespace candy_game_win_l60_60468

def winning_player (A B : ℕ) : String :=
  if (A % B = 0 ∨ B % A = 0) then "Player with forcing checks" else "No inevitable winner"

theorem candy_game_win :
  winning_player 1000 2357 = "Player with forcing checks" :=
by
  sorry

end candy_game_win_l60_60468


namespace solve_inequality_system_l60_60994

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60994


namespace locus_of_centers_l60_60330

theorem locus_of_centers (a b : ℝ) :
  (∃ r : ℝ, a^2 + b^2 = (r + 2)^2 ∧ (a - 3)^2 + b^2 = (9 - r)^2) →
  12 * a^2 + 169 * b^2 - 36 * a - 1584 = 0 :=
by
  sorry

end locus_of_centers_l60_60330


namespace parabola_distance_l60_60948

theorem parabola_distance (m : ℝ) (h : (∀ (p : ℝ), p = 1 / 2 → m = 4 * p)) : m = 2 :=
by
  -- Goal: Prove m = 2 given the conditions.
  sorry

end parabola_distance_l60_60948


namespace quadratic_roots_problem_l60_60177

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l60_60177


namespace tangent_line_at_3_l60_60728

noncomputable def f : ℝ → ℝ := sorry

theorem tangent_line_at_3
  (h_tangent : ∀ x, f x = 2 ∧ (deriv f x) = -1) :
  f 3 + deriv f 3 = 1 := 
by 
  specialize h_tangent 3
  sorry

end tangent_line_at_3_l60_60728


namespace find_function_l60_60259

theorem find_function (f : ℝ → ℝ) (c : ℝ) :
  (∀ x y : ℝ, (x - y) * f (x + y) - (x + y) * f (x - y) = 4 * x * y * (x ^ 2 - y ^ 2)) →
  (∀ x : ℝ, f x = x ^ 3 + c * x) :=
by
  -- The proof details will be filled here.
  sorry

end find_function_l60_60259


namespace find_x_converges_to_l60_60525

noncomputable def series_sum (x : ℝ) : ℝ := ∑' n : ℕ, (4 * (n + 1) - 2) * x^n

theorem find_x_converges_to (x : ℝ) (h : |x| < 1) :
  series_sum x = 60 → x = 29 / 30 :=
by
  sorry

end find_x_converges_to_l60_60525


namespace numOxygenAtoms_l60_60218

-- Define the conditions as hypothesis
def numCarbonAtoms : ℕ := 4
def numHydrogenAtoms : ℕ := 8
def molecularWeight : ℕ := 88
def atomicWeightCarbon : ℕ := 12
def atomicWeightHydrogen : ℕ := 1
def atomicWeightOxygen : ℕ := 16

-- The statement to be proved
theorem numOxygenAtoms :
  let totalWeightC := numCarbonAtoms * atomicWeightCarbon
  let totalWeightH := numHydrogenAtoms * atomicWeightHydrogen
  let totalWeightCH := totalWeightC + totalWeightH
  let weightOxygenAtoms := molecularWeight - totalWeightCH
  let numOxygenAtoms := weightOxygenAtoms / atomicWeightOxygen
  numOxygenAtoms = 2 :=
by {
  sorry
}

end numOxygenAtoms_l60_60218


namespace pipe_ratio_l60_60014

theorem pipe_ratio (A B : ℝ) (hA : A = 1 / 12) (hAB : A + B = 1 / 3) : B / A = 3 := by
  sorry

end pipe_ratio_l60_60014


namespace complete_the_square_l60_60049

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l60_60049


namespace correct_statements_l60_60623

def f (x : ℝ) (b : ℝ) (c : ℝ) := x * (abs x) + b * x + c

theorem correct_statements (b c : ℝ) :
  (∀ x, c = 0 → f (-x) b 0 = - f x b 0) ∧
  (∀ x, b = 0 → c > 0 → (f x 0 c = 0 → x = 0) ∧ ∀ y, f y 0 c ≤ 0) ∧
  (∀ x, ∃ k : ℝ, f (k + x) b c = f (k - x) b c) ∧
  ¬(∀ x, x > 0 → f x b c = c - b^2 / 2) :=
by
  sorry

end correct_statements_l60_60623


namespace baseball_cards_per_friend_l60_60780

theorem baseball_cards_per_friend (total_cards : ℕ) (total_friends : ℕ) (h1 : total_cards = 24) (h2 : total_friends = 4) : (total_cards / total_friends) = 6 := 
by
  sorry

end baseball_cards_per_friend_l60_60780


namespace completing_the_square_l60_60046

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l60_60046


namespace exists_divisible_by_2011_l60_60324

def a (n : ℕ) : ℕ := (List.range n).foldl (λ acc i => acc + 10 ^ i) 0

theorem exists_divisible_by_2011 : ∃ n, 1 ≤ n ∧ n ≤ 2011 ∧ 2011 ∣ a n := by
  sorry

end exists_divisible_by_2011_l60_60324


namespace total_dolls_l60_60630

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end total_dolls_l60_60630


namespace prob_next_black_ball_l60_60469

theorem prob_next_black_ball
  (total_balls : ℕ := 100) 
  (black_balls : Fin 101) 
  (next_black_ball_probability : ℚ := 2 / 3) :
  black_balls.val ≤ total_balls →
  ∃ p q : ℕ, Nat.gcd p q = 1 ∧ (p : ℚ) / q = next_black_ball_probability ∧ p + q = 5 :=
by
  intros h
  use 2, 3
  repeat { sorry }

end prob_next_black_ball_l60_60469


namespace dugu_team_prob_l60_60298

def game_prob (prob_win_first : ℝ) (prob_increase : ℝ) (prob_decrease : ℝ) : ℝ :=
  let p1 := prob_win_first
  let p2 := prob_win_first + prob_increase
  let p3 := prob_win_first + 2 * prob_increase
  let p4 := prob_win_first + 3 * prob_increase
  let p5 := prob_win_first + 4 * prob_increase
  let win_in_3 := p1 * p2 * p3
  let lose_first := (1 - prob_win_first)
  let win_then := prob_win_first
  let win_in_4a := lose_first * (prob_win_first - prob_decrease) * 
    prob_win_first * p2 * p3
  let win_in_4b := win_then * (1 - (prob_win_first + prob_increase)) *
    p2 * p3
  let win_in_4c := win_then * p2 * (1 - prob_win_first + prob_increase - 
    prob_decrease) * p4

  win_in_3 + win_in_4a + win_in_4b + win_in_4c

theorem dugu_team_prob : 
  game_prob 0.4 0.1 0.1 = 0.236 :=
by
  sorry

end dugu_team_prob_l60_60298


namespace smallest_m_plus_n_l60_60131

theorem smallest_m_plus_n (m n : ℕ) (h1 : m > n) (h2 : n ≥ 1) 
(h3 : 1000 ∣ 1978^m - 1978^n) : m + n = 106 :=
sorry

end smallest_m_plus_n_l60_60131


namespace inequality_system_solution_l60_60982

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60982


namespace find_a_l60_60544

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l60_60544


namespace monthly_rent_calculation_l60_60096

-- Definitions based on the problem conditions
def investment_amount : ℝ := 20000
def desired_annual_return_rate : ℝ := 0.06
def annual_property_taxes : ℝ := 650
def maintenance_percentage : ℝ := 0.15

-- Theorem stating the mathematically equivalent problem
theorem monthly_rent_calculation : 
  let required_annual_return := desired_annual_return_rate * investment_amount
  let total_annual_earnings := required_annual_return + annual_property_taxes
  let monthly_earnings_target := total_annual_earnings / 12
  let monthly_rent := monthly_earnings_target / (1 - maintenance_percentage)
  monthly_rent = 181.38 :=
by
  sorry

end monthly_rent_calculation_l60_60096


namespace fred_gave_sandy_balloons_l60_60864

theorem fred_gave_sandy_balloons :
  ∀ (original_balloons given_balloons final_balloons : ℕ),
    original_balloons = 709 →
    final_balloons = 488 →
    given_balloons = original_balloons - final_balloons →
    given_balloons = 221 := by
  sorry

end fred_gave_sandy_balloons_l60_60864


namespace division_problem_l60_60682

theorem division_problem (n : ℕ) (h : n / 4 = 12) : n / 3 = 16 := by
  sorry

end division_problem_l60_60682


namespace total_payment_is_correct_l60_60443

def length : ℕ := 30
def width : ℕ := 40
def construction_cost_per_sqft : ℕ := 3
def sealant_cost_per_sqft : ℕ := 1
def total_area : ℕ := length * width
def total_cost_per_sqft : ℕ := construction_cost_per_sqft + sealant_cost_per_sqft
def total_cost : ℕ := total_area * total_cost_per_sqft

theorem total_payment_is_correct : total_cost = 4800 := by
  sorry

end total_payment_is_correct_l60_60443


namespace coloring_impossible_l60_60606

theorem coloring_impossible :
  ¬ ∃ (color : ℕ → Prop), (∀ n m : ℕ, (m = n + 5 → color n ≠ color m) ∧ (m = 2 * n → color n ≠ color m)) :=
sorry

end coloring_impossible_l60_60606


namespace polynomial_expansion_l60_60589

variable (a_0 a_1 a_2 a_3 a_4 : ℝ)

theorem polynomial_expansion :
  ((a_0 + a_2 + a_4)^2 - (a_1 + a_3)^2 = 1) :=
by
  sorry

end polynomial_expansion_l60_60589


namespace inequality_system_solution_l60_60988

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60988


namespace polygon_perimeter_greater_than_2_l60_60869

-- Definition of the conditions
variable (polygon : Set (ℝ × ℝ))
variable (A B : ℝ × ℝ)
variable (P : ℝ)

axiom point_in_polygon (p : ℝ × ℝ) : p ∈ polygon
axiom A_in_polygon : A ∈ polygon
axiom B_in_polygon : B ∈ polygon
axiom path_length_condition (γ : ℝ → ℝ × ℝ) (γ_in_polygon : ∀ t, γ t ∈ polygon) (hA : γ 0 = A) (hB : γ 1 = B) : ∀ t₁ t₂, 0 ≤ t₁ → t₁ ≤ t₂ → t₂ ≤ 1 → dist (γ t₁) (γ t₂) > 1

-- Statement to prove
theorem polygon_perimeter_greater_than_2 : P > 2 :=
sorry

end polygon_perimeter_greater_than_2_l60_60869


namespace inequality_system_solution_l60_60983

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60983


namespace diameter_of_inscribed_circle_l60_60980

theorem diameter_of_inscribed_circle (a b c r : ℝ) (h_right_triangle : a^2 + b^2 = c^2) (h_radius : r = (a + b - c) / 2) : 
  2 * r = a + b - c :=
by
  sorry

end diameter_of_inscribed_circle_l60_60980


namespace baseball_cards_per_friend_l60_60782

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end baseball_cards_per_friend_l60_60782


namespace interval_necessary_not_sufficient_l60_60084

theorem interval_necessary_not_sufficient :
  (∀ x, x^2 - x - 2 = 0 → (-1 ≤ x ∧ x ≤ 2)) ∧ (∃ x, x^2 - x - 2 = 0 ∧ ¬(-1 ≤ x ∧ x ≤ 2)) → False :=
by
  sorry

end interval_necessary_not_sufficient_l60_60084


namespace calc_15_op_and_op2_l60_60250

def op1 (x : ℤ) : ℤ := 10 - x
def op2 (x : ℤ) : ℤ := x - 10

theorem calc_15_op_and_op2 :
  op2 (op1 15) = -15 :=
by
  sorry

end calc_15_op_and_op2_l60_60250


namespace quadratic_inequality_solution_l60_60698

theorem quadratic_inequality_solution : 
  {x : ℝ | x^2 - 5 * x + 6 > 0 ∧ x ≠ 3} = {x : ℝ | x < 2} ∪ {x : ℝ | x > 3} :=
by
  sorry

end quadratic_inequality_solution_l60_60698


namespace attendance_difference_l60_60973

theorem attendance_difference :
  let a := 65899
  let b := 66018
  b - a = 119 :=
sorry

end attendance_difference_l60_60973


namespace movement_down_l60_60307

def point := (ℤ × ℤ)

theorem movement_down (C D : point) (hC : C = (1, 2)) (hD : D = (1, -1)) :
  D = (C.1, C.2 - 3) :=
by
  sorry

end movement_down_l60_60307


namespace even_function_implies_a_eq_2_l60_60541

theorem even_function_implies_a_eq_2 (a : ℝ) : 
  (∀ x, (f : ℝ → ℝ) x = (λ x : ℝ, x * exp x / (exp (a * x) - 1)) x -> (f (-x) = f x)) -> a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l60_60541


namespace solve_quadratic_l60_60639

theorem solve_quadratic (x : ℝ) (h1 : x > 0) (h2 : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
sorry

end solve_quadratic_l60_60639


namespace distinctPermutations_test_l60_60890

noncomputable def distinctPermutationsCount (s : Multiset ℕ) : ℕ :=
  (s.card.factorial) / (s.count 3.factorial * s.count 8.factorial)

theorem distinctPermutations_test : distinctPermutationsCount {3, 3, 3, 8, 8} = 10 := by
  sorry

end distinctPermutations_test_l60_60890


namespace hunter_3_proposal_l60_60493

theorem hunter_3_proposal {hunter1_coins hunter2_coins hunter3_coins : ℕ} :
  hunter3_coins = 99 ∧ hunter1_coins = 1 ∧ (hunter1_coins + hunter3_coins + hunter2_coins = 100) :=
  sorry

end hunter_3_proposal_l60_60493


namespace expansion_coeff_x_cubed_l60_60295

-- Define the primary condition
def sum_coeffs_condition (n : ℕ) := (3 : ℕ) ^ n = 32

-- Define the main expression
def binom_expansion_term (n r : ℕ) : ℤ := (-1) ^ r * (3 ^ (n - r)) * (Nat.binomial n r) * (x : ℕ ^ (n - 2 * r))

-- Define the target term we are interested in
def target_term_coefficient (n r : ℕ) (target_power : ℕ) :=
  n = 5 ∧ r = 1 ∧ target_power = 3

theorem expansion_coeff_x_cubed :
  Π (n r : ℕ), sum_coeffs_condition n → target_term_coefficient n r (5 - 2 * r) →
  binom_expansion_term n r = -405 :=
by
  intros n r h_n h_term
  -- Proof steps will go here.
  -- Adding sorry to skip the proof as instructed.
  sorry

end expansion_coeff_x_cubed_l60_60295


namespace num_solutions_l60_60857

-- Let x be a real number
variable (x : ℝ)

-- Define the given equation
def equation := (x^2 - 4) * (x^2 - 1) = (x^2 + 3*x + 2) * (x^2 - 8*x + 7)

-- Theorem: The number of values of x that satisfy the equation is 3
theorem num_solutions : ∃ (S : Finset ℝ), (∀ x, x ∈ S ↔ equation x) ∧ S.card = 3 := 
by
  sorry

end num_solutions_l60_60857


namespace calculate_expression_l60_60382

theorem calculate_expression :
  (4 * Nat.factorial 6 + 20 * Nat.factorial 5) / Nat.factorial 7 = 22 / 21 :=
by
  have fact_6_pos : Nat.factorial 6 > 0 := Nat.factorial_pos 6
  have fact_5_pos : Nat.factorial 5 > 0 := Nat.factorial_pos 5
  have fact_7_pos : Nat.factorial 7 > 0 := Nat.factorial_pos 7

  have mul_pos_1 : 4 * Nat.factorial 6 > 0 := mul_pos (by norm_num) fact_6_pos
  have mul_pos_2 : 20 * Nat.factorial 5 > 0 := mul_pos (by norm_num) fact_5_pos
  have add_pos : 4 * Nat.factorial 6 + 20 * Nat.factorial 5 > 0 := add_pos mul_pos_1 mul_pos_2

  have exp_nonneg : 0 < Nat.factorial 7 := fact_7_pos

  norm_num
  sorry

end calculate_expression_l60_60382


namespace complex_division_l60_60580

theorem complex_division (z : ℂ) (hz : (3 + 4 * I) * z = 25) : z = 3 - 4 * I :=
sorry

end complex_division_l60_60580


namespace problem_statement_l60_60867

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := 2⁻¹
noncomputable def c : ℝ := Real.log 6 / Real.log 5

theorem problem_statement : b < a ∧ a < c := by
  sorry

end problem_statement_l60_60867


namespace breadth_decrease_percentage_l60_60789

theorem breadth_decrease_percentage
  (L B : ℝ)
  (hLpos : L > 0)
  (hBpos : B > 0)
  (harea_change : (1.15 * L) * (B - p/100 * B) = 1.035 * (L * B)) :
  p = 10 := 
sorry

end breadth_decrease_percentage_l60_60789


namespace green_hat_cost_l60_60342

theorem green_hat_cost (G : ℝ) (total_hats : ℕ) (blue_hats : ℕ) (green_hats : ℕ) (blue_cost : ℝ) (total_cost : ℝ) 
    (h₁ : blue_hats = 85) (h₂ : blue_cost = 6) (h₃ : green_hats = 90) (h₄ : total_cost = 600) 
    (h₅ : total_hats = blue_hats + green_hats) 
    (h₆ : total_cost = blue_hats * blue_cost + green_hats * G) : 
    G = 1 := by
  sorry

end green_hat_cost_l60_60342


namespace boat_travel_times_l60_60115

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l60_60115


namespace correct_answers_count_l60_60601

theorem correct_answers_count
  (c w : ℕ)
  (h1 : 4 * c - 2 * w = 420)
  (h2 : c + w = 150) : 
  c = 120 :=
sorry

end correct_answers_count_l60_60601


namespace total_insects_l60_60036

theorem total_insects (leaves : ℕ) (ladybugs_per_leaf : ℕ) (ants_per_leaf : ℕ) (caterpillars_every_third_leaf : ℕ) :
  leaves = 84 →
  ladybugs_per_leaf = 139 →
  ants_per_leaf = 97 →
  caterpillars_every_third_leaf = 53 →
  (84 * 139) + (84 * 97) + (53 * (84 / 3)) = 21308 := 
by
  sorry

end total_insects_l60_60036


namespace regular_polygon_sides_l60_60510

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l60_60510


namespace simple_interest_calculation_l60_60777

-- Defining the given values
def principal : ℕ := 1500
def rate : ℕ := 7
def time : ℕ := rate -- time is the same as the rate of interest

-- Define the simple interest calculation
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Proof statement
theorem simple_interest_calculation : simple_interest principal rate time = 735 := by
  sorry

end simple_interest_calculation_l60_60777


namespace two_digit_primes_with_units_digit_three_l60_60938

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l60_60938


namespace find_a_of_even_function_l60_60562

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l60_60562


namespace completing_the_square_l60_60068

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l60_60068


namespace valid_license_plates_count_l60_60824

-- Defining the total number of choices for letters and digits
def num_letter_choices := 26
def num_digit_choices := 10

-- Function to calculate the total number of valid license plates
def total_license_plates := num_letter_choices ^ 3 * num_digit_choices ^ 4

-- The proof statement
theorem valid_license_plates_count : total_license_plates = 175760000 := 
by 
  -- The placeholder for the proof
  sorry

end valid_license_plates_count_l60_60824


namespace pages_in_each_book_l60_60400

variable (BooksRead DaysPerBook TotalDays : ℕ)

theorem pages_in_each_book (h1 : BooksRead = 41) (h2 : DaysPerBook = 12) (h3 : TotalDays = 492) : (TotalDays / DaysPerBook) * DaysPerBook = 492 :=
by
  sorry

end pages_in_each_book_l60_60400


namespace square_of_99_is_9801_l60_60206

theorem square_of_99_is_9801 : 99 ^ 2 = 9801 := 
by
  sorry

end square_of_99_is_9801_l60_60206


namespace company_pays_240_per_month_l60_60483

-- Conditions as definitions
def box_length : ℕ := 15
def box_width : ℕ := 12
def box_height : ℕ := 10
def total_volume : ℕ := 1080000      -- 1.08 million cubic inches
def price_per_box_per_month : ℚ := 0.4

-- The volume of one box
def box_volume : ℕ := box_length * box_width * box_height

-- Calculate the number of boxes
def number_of_boxes : ℕ := total_volume / box_volume

-- Total amount paid per month for record storage
def total_amount_paid_per_month : ℚ := number_of_boxes * price_per_box_per_month

-- Theorem statement to prove
theorem company_pays_240_per_month : total_amount_paid_per_month = 240 := 
by 
  sorry

end company_pays_240_per_month_l60_60483


namespace sum_of_ages_l60_60659

-- Definitions of John's age and father's age according to the given conditions
def John's_age := 15
def Father's_age := 2 * John's_age + 32

-- The proof problem statement
theorem sum_of_ages : John's_age + Father's_age = 77 :=
by
  -- Here we would substitute and simplify according to the given conditions
  sorry

end sum_of_ages_l60_60659


namespace hydrochloric_acid_required_l60_60395

-- Define the quantities for the balanced reaction equation
def molesOfAgNO3 : ℕ := 2
def molesOfHNO3 : ℕ := 2
def molesOfHCl : ℕ := 2

-- Define the condition for the reaction (balances the equation)
def balanced_reaction (x y z w : ℕ) : Prop :=
  x = y ∧ x = z ∧ y = w

-- The goal is to prove that the number of moles of HCl needed is 2
theorem hydrochloric_acid_required :
  balanced_reaction molesOfAgNO3 molesOfHCl molesOfHNO3 2 →
  molesOfHCl = 2 :=
by sorry

end hydrochloric_acid_required_l60_60395


namespace arithmetic_sequence_a1_d_l60_60274

theorem arithmetic_sequence_a1_d (a_1 a_2 a_3 a_5 d : ℤ)
  (h1 : a_5 = a_1 + 4 * d)
  (h2 : a_1 + a_2 + a_3 = 3)
  (h3 : a_2 = a_1 + d)
  (h4 : a_3 = a_1 + 2 * d) :
  a_1 = -2 ∧ d = 3 :=
by
  have h_a2 : a_2 = 1 := sorry
  have h_a5 : a_5 = 10 := sorry
  have h_d : d = 3 := sorry
  have h_a1 : a_1 = -2 := sorry
  exact ⟨h_a1, h_d⟩

end arithmetic_sequence_a1_d_l60_60274


namespace total_sum_of_ages_is_correct_l60_60660

-- Definition of conditions
def ageOfYoungestChild : Nat := 4
def intervals : Nat := 3

-- Total sum calculation
def sumOfAges (ageOfYoungestChild intervals : Nat) :=
  let Y := ageOfYoungestChild
  Y + (Y + intervals) + (Y + 2 * intervals) + (Y + 3 * intervals) + (Y + 4 * intervals)

theorem total_sum_of_ages_is_correct : sumOfAges 4 3 = 50 :=
by
  sorry

end total_sum_of_ages_is_correct_l60_60660


namespace existence_of_points_on_AC_l60_60774

theorem existence_of_points_on_AC (A B C M : ℝ) (hAB : abs (A - B) = 2) (hBC : abs (B - C) = 1) :
  ((abs (A - M) + abs (B - M) = abs (C - M)) ↔ (M = A - 1) ∨ (M = A + 1)) :=
by
  sorry

end existence_of_points_on_AC_l60_60774


namespace max_value_of_f_prime_div_f_l60_60407

def f (x : ℝ) : ℝ := sorry

theorem max_value_of_f_prime_div_f (f : ℝ → ℝ) (h1 : ∀ x, deriv f x - f x = 2 * x * Real.exp x) (h2 : f 0 = 1) :
  ∀ x > 0, (deriv f x / f x) ≤ 2 :=
sorry

end max_value_of_f_prime_div_f_l60_60407


namespace pairwise_products_same_digit_l60_60039

theorem pairwise_products_same_digit
  (a b c : ℕ)
  (h_ab : a % 10 ≠ b % 10)
  (h_ac : a % 10 ≠ c % 10)
  (h_bc : b % 10 ≠ c % 10)
  : (a * b % 10 = a * c % 10) ∧ (a * b % 10 = b * c % 10) :=
  sorry

end pairwise_products_same_digit_l60_60039


namespace tin_silver_ratio_l60_60359

/-- Assuming a metal bar made of an alloy of tin and silver weighs 40 kg, 
    and loses 4 kg in weight when submerged in water,
    where 10 kg of tin loses 1.375 kg in water and 5 kg of silver loses 0.375 kg, 
    prove that the ratio of tin to silver in the bar is 2 : 3. -/
theorem tin_silver_ratio :
  ∃ (T S : ℝ), 
    T + S = 40 ∧ 
    0.1375 * T + 0.075 * S = 4 ∧ 
    T / S = 2 / 3 := 
by
  sorry

end tin_silver_ratio_l60_60359


namespace find_a_if_g_even_l60_60138

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 2 then x - 1 else if -2 ≤ x ∧ x ≤ 0 then -1 else 0

noncomputable def g (x : ℝ) (a : ℝ) : ℝ := (f x) + a * x

theorem find_a_if_g_even (a : ℝ) : (∀ x : ℝ, f x + a * x = f (-x) + a * (-x)) → a = -1/2 :=
by
  intro h
  sorry

end find_a_if_g_even_l60_60138


namespace maria_ann_age_problem_l60_60971

theorem maria_ann_age_problem
  (M A : ℕ)
  (h1 : M = 7)
  (h2 : M = A - 3) :
  ∃ Y : ℕ, 7 - Y = 1 / 2 * (10 - Y) := by
  sorry

end maria_ann_age_problem_l60_60971


namespace eq_a_2_l60_60553

def f (a x : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

-- Definition of even function: ∀ x, f(-x) = f(x)
def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem eq_a_2 (a : ℝ) : (even_function (f a) → a = 2) ∧ (a = 2 → even_function (f a)) :=
by
  sorry

end eq_a_2_l60_60553


namespace find_value_of_a_squared_b_plus_ab_squared_l60_60584

theorem find_value_of_a_squared_b_plus_ab_squared 
  (a b : ℝ) 
  (h1 : a + b = -3) 
  (h2 : ab = 2) : 
  a^2 * b + a * b^2 = -6 :=
by 
  sorry

end find_value_of_a_squared_b_plus_ab_squared_l60_60584


namespace almonds_addition_l60_60755

theorem almonds_addition (walnuts almonds total_nuts : ℝ) 
  (h_walnuts : walnuts = 0.25) 
  (h_total_nuts : total_nuts = 0.5)
  (h_sum : total_nuts = walnuts + almonds) : 
  almonds = 0.25 := by
  sorry

end almonds_addition_l60_60755


namespace common_points_count_l60_60705

variable (x y : ℝ)

def curve1 : Prop := x^2 + 4 * y^2 = 4
def curve2 : Prop := 4 * x^2 + y^2 = 4
def curve3 : Prop := x^2 + y^2 = 1

theorem common_points_count : ∀ (x y : ℝ), curve1 x y ∧ curve2 x y ∧ curve3 x y → false := by
  intros
  sorry

end common_points_count_l60_60705


namespace add_base_12_l60_60373

def a_in_base_10 := 10
def b_in_base_10 := 11
def c_base := 12

theorem add_base_12 : 
  let a := 10
  let b := 11
  (3 * c_base ^ 2 + 12 * c_base + 5) + (2 * c_base ^ 2 + a * c_base + b) = 6 * c_base ^ 2 + 3 * c_base + 4 :=
by
  sorry

end add_base_12_l60_60373


namespace divisor_of_3825_is_15_l60_60205

theorem divisor_of_3825_is_15 : ∃ d, 3830 - 5 = 3825 ∧ 3825 % d = 0 ∧ d = 15 := by
  sorry

end divisor_of_3825_is_15_l60_60205


namespace two_digit_primes_with_units_digit_three_count_l60_60930

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l60_60930


namespace stream_speed_l60_60484

variables (v : ℝ) (swimming_speed : ℝ) (ratio : ℝ)

theorem stream_speed (hs : swimming_speed = 4.5) (hr : ratio = 2) (h : (swimming_speed - v) / (swimming_speed + v) = 1 / ratio) :
  v = 1.5 :=
sorry

end stream_speed_l60_60484


namespace fred_sheets_left_l60_60709

def sheets_fred_had_initially : ℕ := 212
def sheets_jane_given : ℕ := 307
def planned_percentage_more : ℕ := 50
def given_percentage : ℕ := 25

-- Prove that after all transactions, Fred has 389 sheets left
theorem fred_sheets_left :
  let planned_sheets := (sheets_jane_given * 100) / (planned_percentage_more + 100)
  let sheets_jane_actual := planned_sheets + (planned_sheets * planned_percentage_more) / 100
  let total_sheets := sheets_fred_had_initially + sheets_jane_actual
  let charles_given := (total_sheets * given_percentage) / 100
  let fred_sheets_final := total_sheets - charles_given
  fred_sheets_final = 389 := 
by
  sorry

end fred_sheets_left_l60_60709


namespace auntie_em_can_park_l60_60091

-- Definition of the problem conditions
def parking_lot_spaces := 18
def required_car_spaces := 12
def suv_spaces := 2
def remaining_spaces := parking_lot_spaces - required_car_spaces

-- Definition of the solution: number of ways to choose 6 spaces (remaining spaces)
def total_ways_to_choose_spaces := (Finset.image (λ x : Fin 18, 18.choose x) Finset.univ).card

-- Definition of unfavorable configurations where no two empty spaces are adjacent
def unfavorable_ways_to_choose_spaces := (Finset.image (λ x : Fin 13, 13.choose x) Finset.univ).card

-- Definition of the ratio
def probability_not_able_to_park := (unfavorable_ways_to_choose_spaces: ℚ) / (total_ways_to_choose_spaces: ℚ)
def probability_able_to_park := 1 - probability_not_able_to_park

-- Proof of the given probability
theorem auntie_em_can_park : probability_able_to_park = 1403 / 1546 := by
  sorry

end auntie_em_can_park_l60_60091


namespace find_B_find_b_l60_60746

-- Definitions
def a (b c : ℝ) (C B : ℝ) : ℝ := b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B
def area (a c B : ℝ) : ℝ := 1/2 * a * c * Real.sin B

-- Given conditions in Lean statements
variables (a c b : ℝ)
variables (A B C : ℝ)
variable (triangle_ABC : a = b * Real.cos C + (Real.sqrt 3 / 3) * c * Real.sin B)
variable (sum_ac : a + c = 6)
variable (area_eq : area a c B = 3 * Real.sqrt 3 / 2)

-- Prove the value of angle B
theorem find_B : B = Real.pi / 3 
by sorry

-- Prove the length of side b
theorem find_b : b = 3 * Real.sqrt 2 
by sorry

end find_B_find_b_l60_60746


namespace monthly_rent_requirement_l60_60370

noncomputable def initial_investment : Float := 200000
noncomputable def annual_return_rate : Float := 0.06
noncomputable def annual_insurance_cost : Float := 4500
noncomputable def maintenance_percentage : Float := 0.15
noncomputable def required_monthly_rent : Float := 1617.65

theorem monthly_rent_requirement :
  let annual_return := initial_investment * annual_return_rate
  let annual_cost_with_insurance := annual_return + annual_insurance_cost
  let monthly_required_net := annual_cost_with_insurance / 12
  let rental_percentage_kept := 1 - maintenance_percentage
  let monthly_rental_full := monthly_required_net / rental_percentage_kept
  monthly_rental_full = required_monthly_rent := 
by
  sorry

end monthly_rent_requirement_l60_60370


namespace length_of_one_side_of_regular_octagon_l60_60668

theorem length_of_one_side_of_regular_octagon
  (a b : ℕ)
  (h_pentagon : a = 16)   -- Side length of regular pentagon
  (h_total_yarn_pentagon : b = 80)  -- Total yarn for pentagon
  (hpentagon_yarn_length : 5 * a = b)  -- Total yarn condition
  (hoctagon_total_sides : 8 = 8)   -- Number of sides of octagon
  (hoctagon_side_length : 10 = b / 8)  -- Side length condition for octagon
  : 10 = 10 :=
by
  sorry

end length_of_one_side_of_regular_octagon_l60_60668


namespace jake_time_to_row_lake_l60_60435

noncomputable def time_to_row_lake (side_length miles_per_side : ℝ) (swim_time_per_mile minutes_per_mile : ℝ) : ℝ :=
  let swim_speed := 60 / swim_time_per_mile -- miles per hour
  let row_speed := 2 * swim_speed          -- miles per hour
  let total_distance := 4 * side_length    -- miles
  total_distance / row_speed               -- hours

theorem jake_time_to_row_lake :
  time_to_row_lake 15 20 = 10 := sorry

end jake_time_to_row_lake_l60_60435


namespace anne_total_bottle_caps_l60_60827

/-- 
Anne initially has 10 bottle caps 
and then finds another 5 bottle caps.
-/
def anne_initial_bottle_caps : ℕ := 10
def anne_found_bottle_caps : ℕ := 5

/-- 
Prove that the total number of bottle caps
Anne ends with is equal to 15.
-/
theorem anne_total_bottle_caps : 
  anne_initial_bottle_caps + anne_found_bottle_caps = 15 :=
by 
  sorry

end anne_total_bottle_caps_l60_60827


namespace Megan_popsicles_l60_60319

def minutes_in_hour : ℕ := 60

def total_minutes (hours : ℕ) (minutes : ℕ) : ℕ :=
  hours * minutes_in_hour + minutes

def popsicle_time : ℕ := 18

def popsicles_consumed (total_minutes : ℕ) (popsicle_time : ℕ) : ℕ :=
  total_minutes / popsicle_time

theorem Megan_popsicles (hours : ℕ) (minutes : ℕ) (popsicle_time : ℕ)
  (total_minutes : ℕ) (h_hours : hours = 5) (h_minutes : minutes = 36) (h_popsicle_time : popsicle_time = 18)
  (h_total_minutes : total_minutes = (5 * 60 + 36)) :
  popsicles_consumed 336 popsicle_time = 18 :=
by 
  sorry

end Megan_popsicles_l60_60319


namespace solve_inequality_system_l60_60998

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l60_60998


namespace find_a_even_function_l60_60559

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l60_60559


namespace complement_union_l60_60885

open Set

-- Define U to be the set of all real numbers
def U := @univ ℝ

-- Define the domain A for the function y = sqrt(x-2) + sqrt(x+1)
def A := {x : ℝ | x ≥ 2}

-- Define the domain B for the function y = sqrt(2x+4) / (x-3)
def B := {x : ℝ | x ≥ -2 ∧ x ≠ 3}

-- Theorem about the union of the complements
theorem complement_union : (U \ A ∪ U \ B) = {x : ℝ | x < 2 ∨ x = 3} := 
by
  sorry

end complement_union_l60_60885


namespace roots_quadratic_sum_product_l60_60724

theorem roots_quadratic_sum_product :
  (∀ x1 x2 : ℝ, (∀ x, x^2 - 4 * x + 3 = 0 → x = x1 ∨ x = x2) → (x1 + x2 - x1 * x2 = 1)) :=
by
  sorry

end roots_quadratic_sum_product_l60_60724


namespace probability_exactly_one_correct_l60_60976

def P_A := 0.7
def P_B := 0.8

def P_A_correct_B_incorrect := P_A * (1 - P_B)
def P_A_incorrect_B_correct := (1 - P_A) * P_B

theorem probability_exactly_one_correct :
  P_A_correct_B_incorrect + P_A_incorrect_B_correct = 0.38 :=
by
  sorry

end probability_exactly_one_correct_l60_60976


namespace rectangle_perimeter_l60_60228

variable (a b : ℕ)

theorem rectangle_perimeter (h1 : a ≠ b) (h2 : ab = 8 * (a + b)) : 
  2 * (a + b) = 66 := 
sorry

end rectangle_perimeter_l60_60228


namespace total_votes_l60_60600

theorem total_votes (V : ℝ) 
  (h1 : 0.5 / 100 * V = 0.005 * V) 
  (h2 : 50.5 / 100 * V = 0.505 * V) 
  (h3 : 0.505 * V - 0.005 * V = 3000) : 
  V = 6000 := 
by
  sorry

end total_votes_l60_60600


namespace P_cubed_plus_7_is_composite_l60_60417

theorem P_cubed_plus_7_is_composite (P : ℕ) (h_prime_P : Nat.Prime P) (h_prime_P3_plus_5 : Nat.Prime (P^3 + 5)) : ¬ Nat.Prime (P^3 + 7) ∧ (P^3 + 7).factors.length > 1 :=
by
  sorry

end P_cubed_plus_7_is_composite_l60_60417


namespace yellow_white_flowers_count_l60_60236

theorem yellow_white_flowers_count
    (RY RW : Nat)
    (hRY : RY = 17)
    (hRW : RW = 14)
    (hRedMoreThanWhite : (RY + RW) - (RW + YW) = 4) :
    ∃ YW, YW = 13 := 
by
  sorry

end yellow_white_flowers_count_l60_60236


namespace parallel_vectors_x_value_l60_60144

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 3)
def b (x : ℝ) : ℝ × ℝ := (6, x)

-- Define what it means for vectors to be parallel (they are proportional)
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 = v.2 * w.1

-- Theorem to prove
theorem parallel_vectors_x_value :
  ∀ x : ℝ, parallel a (b x) → x = 9 :=
by
  intros x h
  sorry

end parallel_vectors_x_value_l60_60144


namespace ratio_of_work_speeds_l60_60372

theorem ratio_of_work_speeds (B_speed : ℚ) (combined_speed : ℚ) (A_speed : ℚ) 
  (h1 : B_speed = 1/12) 
  (h2 : combined_speed = 1/4) 
  (h3 : A_speed + B_speed = combined_speed) : 
  A_speed / B_speed = 2 := 
sorry

end ratio_of_work_speeds_l60_60372


namespace polynomial_divisibility_l60_60123

theorem polynomial_divisibility (P : Polynomial ℝ) (h_nonconstant : ∃ n : ℕ, P.degree = n ∧ n ≥ 1)
  (h_div : ∀ x : ℝ, P.eval (x^3 + 8) = 0 → P.eval (x^2 - 2*x + 4) = 0) :
  ∃ a : ℝ, ∃ n : ℕ, a ≠ 0 ∧ P = Polynomial.C a * Polynomial.X ^ n :=
sorry

end polynomial_divisibility_l60_60123


namespace gcd_consecutive_terms_l60_60670

theorem gcd_consecutive_terms (n : ℕ) : 
  Nat.gcd (2 * Nat.factorial n + n) (2 * Nat.factorial (n + 1) + (n + 1)) = 1 :=
by
  sorry

end gcd_consecutive_terms_l60_60670


namespace first_term_arithmetic_sequence_median_1010_last_2015_l60_60677

theorem first_term_arithmetic_sequence_median_1010_last_2015 (a₁ : ℕ) :
  let median := 1010
  let last_term := 2015
  (a₁ + last_term = 2 * median) → a₁ = 5 :=
by
  intros
  sorry

end first_term_arithmetic_sequence_median_1010_last_2015_l60_60677


namespace max_abs_f_le_f0_f1_l60_60760

noncomputable def f (a b x : ℝ) : ℝ := 3 * a * x^2 - 2 * (a + b) * x + b

theorem max_abs_f_le_f0_f1 (a b : ℝ) (h : 0 < a) (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) :
  |f a b x| ≤ max (|f a b 0|) (|f a b 1|) :=
sorry

end max_abs_f_le_f0_f1_l60_60760


namespace problem_1_problem_2_l60_60583

noncomputable def distance_between_parallel_lines (A B C1 C2 : ℝ) : ℝ :=
  let numerator := |C1 - C2|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

noncomputable def distance_point_to_line (A B C x0 y0 : ℝ) : ℝ :=
  let numerator := |A * x0 + B * y0 + C|
  let denominator := Real.sqrt (A^2 + B^2)
  numerator / denominator

theorem problem_1 : distance_between_parallel_lines 2 1 (-1) 1 = 2 * Real.sqrt 5 / 5 :=
  by sorry

theorem problem_2 : distance_point_to_line 2 1 (-1) 0 2 = Real.sqrt 5 / 5 :=
  by sorry

end problem_1_problem_2_l60_60583


namespace completing_square_l60_60063

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l60_60063


namespace equivalent_solution_eq1_eqC_l60_60807

-- Define the given equation
def eq1 (x y : ℝ) : Prop := 4 * x - 8 * y - 5 = 0

-- Define the candidate equations
def eqA (x y : ℝ) : Prop := 8 * x - 8 * y - 10 = 0
def eqB (x y : ℝ) : Prop := 8 * x - 16 * y - 5 = 0
def eqC (x y : ℝ) : Prop := 8 * x - 16 * y - 10 = 0
def eqD (x y : ℝ) : Prop := 12 * x - 24 * y - 10 = 0

-- The theorem that we need to prove
theorem equivalent_solution_eq1_eqC : ∀ x y, eq1 x y ↔ eqC x y :=
by
  sorry

end equivalent_solution_eq1_eqC_l60_60807


namespace math_problem_l60_60756

noncomputable def proof_problem (k : ℝ) (a b k1 k2 : ℝ) : Prop :=
  (a*b) = 7/k ∧ (a + b) = (k-1)/k ∧ (k1^2 - 18*k1 + 1) = 0 ∧ (k2^2 - 18*k2 + 1) = 0 ∧ 
  (a/b + b/a = 3/7) → (k1/k2 + k2/k1 = 322)

theorem math_problem (k a b k1 k2 : ℝ) : proof_problem k a b k1 k2 :=
by
  sorry

end math_problem_l60_60756


namespace minimum_loadings_to_prove_first_ingot_weighs_1kg_l60_60166

theorem minimum_loadings_to_prove_first_ingot_weighs_1kg :
  ∀ (w : Fin 11 → ℕ), 
    (∀ i, w i = i + 1) →
    (∃ s₁ s₂ : Finset (Fin 11), 
       s₁.card ≤ 6 ∧ s₂.card ≤ 6 ∧ 
       s₁.sum w = 11 ∧ s₂.sum w = 11 ∧ 
       (∀ s : Finset (Fin 11), s.sum w = 11 → s ≠ s₁ ∧ s ≠ s₂) ∧
       (w 0 = 1)) := sorry -- Fill in the proof here

end minimum_loadings_to_prove_first_ingot_weighs_1kg_l60_60166


namespace simplify_expression_l60_60519

theorem simplify_expression: 3 * Real.sqrt 48 - 6 * Real.sqrt (1 / 3) + (Real.sqrt 3 - 1) ^ 2 = 8 * Real.sqrt 3 + 4 := by
  sorry

end simplify_expression_l60_60519


namespace total_amount_to_pay_l60_60753

theorem total_amount_to_pay (cost_earbuds cost_smartwatch : ℕ) (tax_rate_earbuds tax_rate_smartwatch : ℚ) 
  (h1 : cost_earbuds = 200) (h2 : cost_smartwatch = 300) 
  (h3 : tax_rate_earbuds = 0.15) (h4 : tax_rate_smartwatch = 0.12) : 
  (cost_earbuds + cost_earbuds * tax_rate_earbuds + cost_smartwatch + cost_smartwatch * tax_rate_smartwatch = 566) := 
by 
  sorry

end total_amount_to_pay_l60_60753


namespace number_of_pigs_l60_60814

variable (cows pigs : Nat)

theorem number_of_pigs (h1 : 2 * (7 + pigs) = 32) : pigs = 9 := by
  sorry

end number_of_pigs_l60_60814


namespace raj_earns_more_l60_60322

theorem raj_earns_more :
  let cost_per_sqft := 2
  let raj_length := 30
  let raj_width := 50
  let lena_length := 40
  let lena_width := 35
  let raj_area := raj_length * raj_width
  let lena_area := lena_length * lena_width
  let raj_earnings := raj_area * cost_per_sqft
  let lena_earnings := lena_area * cost_per_sqft
  raj_earnings - lena_earnings = 200 :=
by
  sorry

end raj_earns_more_l60_60322


namespace total_length_proof_l60_60482

def length_of_first_tape : ℝ := 25
def overlap : ℝ := 3
def number_of_tapes : ℝ := 64

def total_tape_length : ℝ :=
  let effective_length_per_subsequent_tape := length_of_first_tape - overlap
  let length_of_remaining_tapes := effective_length_per_subsequent_tape * (number_of_tapes - 1)
  length_of_first_tape + length_of_remaining_tapes

theorem total_length_proof : total_tape_length = 1411 := by
  sorry

end total_length_proof_l60_60482


namespace angle_A_sides_b_c_l60_60956

noncomputable def triangle_angles (a b c : ℝ) (A B C : ℝ) : Prop :=
  a * Real.sin C - Real.sqrt 3 * c * Real.cos A = 0

theorem angle_A (a b c A B C : ℝ) (h1 : triangle_angles a b c A B C) :
  A = Real.pi / 3 :=
by sorry

noncomputable def triangle_area (a b c S : ℝ) : Prop :=
  S = Real.sqrt 3 ∧ a = 2

theorem sides_b_c (a b c S : ℝ) (h : triangle_area a b c S) :
  b = 2 ∧ c = 2 :=
by sorry

end angle_A_sides_b_c_l60_60956


namespace eggs_per_omelet_l60_60454

theorem eggs_per_omelet:
  let small_children_tickets := 53
  let older_children_tickets := 35
  let adult_tickets := 75
  let senior_tickets := 37
  let smallChildrenOmelets := small_children_tickets * 0.5
  let olderChildrenOmelets := older_children_tickets
  let adultOmelets := adult_tickets * 2
  let seniorOmelets := senior_tickets * 1.5
  let extra_omelets := 25
  let total_omelets := smallChildrenOmelets + olderChildrenOmelets + adultOmelets + seniorOmelets + extra_omelets
  let total_eggs := 584
  total_eggs / total_omelets = 2 := 
by
  sorry

end eggs_per_omelet_l60_60454


namespace trivia_competition_points_l60_60376

theorem trivia_competition_points 
  (total_members : ℕ := 120) 
  (absent_members : ℕ := 37) 
  (points_per_member : ℕ := 24) : 
  (total_members - absent_members) * points_per_member = 1992 := 
by
  sorry

end trivia_competition_points_l60_60376


namespace range_of_m_l60_60720

variables {m x : ℝ}

def p (m : ℝ) : Prop := (16 * (m - 2)^2 - 16 > 0) ∧ (m - 2 < 0)
def q (m : ℝ) : Prop := (9 * m^2 - 4 < 0)
def pq (m : ℝ) : Prop := (p m ∨ q m) ∧ ¬(q m)

theorem range_of_m (h : pq m) : m ≤ -2/3 ∨ (2/3 ≤ m ∧ m < 1) :=
sorry

end range_of_m_l60_60720


namespace unique_solution_l60_60851

theorem unique_solution (a b : ℤ) (h : a > b ∧ b > 0) (hab : a * b - a - b = 1) : a = 3 ∧ b = 2 := by
  sorry

end unique_solution_l60_60851


namespace triangle_side_length_difference_l60_60389

theorem triangle_side_length_difference (x : ℤ) :
  (2 < x ∧ x < 16) → (∀ y : ℤ, (2 < y ∧ y < 16) → (3 ≤ y) ∧ (y ≤ 15)) →
  (∀ z : ℤ, (3 ≤ z ∨ z ≤ 15) → (15 - 3 = 12)) := by
  sorry

end triangle_side_length_difference_l60_60389


namespace find_x_squared_plus_y_squared_l60_60736

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : 
  x^2 + y^2 = 342 := 
sorry

end find_x_squared_plus_y_squared_l60_60736


namespace find_a_of_even_function_l60_60563

noncomputable def f (a x : ℝ) : ℝ := (x * exp x) / (exp (a * x) - 1)

theorem find_a_of_even_function (hf : ∀ x : ℝ, f a (-x) = f a x) : a = 2 :=
by {
    sorry
}

end find_a_of_even_function_l60_60563


namespace basic_computer_price_l60_60676

variable (C P : ℕ)

theorem basic_computer_price 
  (h1 : C + P = 2500)
  (h2 : P = (C + 500 + P) / 3) : 
  C = 1500 := 
sorry

end basic_computer_price_l60_60676


namespace exists_person_with_girls_as_neighbors_l60_60199

theorem exists_person_with_girls_as_neighbors (boys girls : Nat) (sitting : Nat) 
  (h_boys : boys = 25) (h_girls : girls = 25) (h_sitting : sitting = boys + girls) :
  ∃ p : Nat, p < sitting ∧ (p % 2 = 1 → p.succ % sitting % 2 = 0) := 
by
  sorry

end exists_person_with_girls_as_neighbors_l60_60199


namespace regular_polygon_sides_l60_60507

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l60_60507


namespace perpendicular_condition_l60_60349

-- Condition definition
def is_perpendicular (a : ℝ) : Prop :=
  let line1_slope := -1
  let line2_slope := - (a / 2)
  (line1_slope * line2_slope = -1)

-- Statement of the theorem
theorem perpendicular_condition (a : ℝ) :
  is_perpendicular a ↔ a = -2 :=
sorry

end perpendicular_condition_l60_60349


namespace age_of_eldest_child_l60_60335

theorem age_of_eldest_child (age_sum : ∀ (x : ℕ), x + (x + 2) + (x + 4) + (x + 6) + (x + 8) = 40) :
  ∃ x, x + 8 = 12 :=
by {
  sorry
}

end age_of_eldest_child_l60_60335


namespace problem_l60_60167

-- Conditions
def a_n (n : ℕ) : ℚ := (1/3)^(n-1)

def b_n (n : ℕ) : ℚ := n * (1/3)^n

-- Sums over the first n terms
def S_n (n : ℕ) : ℚ := (3/2) - (1/2) * (1/3)^n

def T_n (n : ℕ) : ℚ := (3/4) - (1/4) * (1/3)^n - (n/2) * (1/3)^n

-- Problem: Prove T_n < S_n / 2
theorem problem (n : ℕ) : T_n n < S_n n / 2 :=
by sorry

end problem_l60_60167


namespace count_three_digit_perfect_squares_divisible_by_4_l60_60149

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l60_60149


namespace xyz_sum_eq_40_l60_60813

theorem xyz_sum_eq_40
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h1 : x^2 + x * y + y^2 = 75)
  (h2 : y^2 + y * z + z^2 = 16)
  (h3 : z^2 + x * z + x^2 = 91) :
  x * y + y * z + z * x = 40 :=
sorry

end xyz_sum_eq_40_l60_60813


namespace hyperbola_condition_l60_60564

theorem hyperbola_condition (m n : ℝ) : 
  (mn < 0) ↔ (∀ x y : ℝ, ∃ k ∈ {a : ℝ | a ≠ 0}, (x^2 / m + y^2 / n = 1)) := sorry

end hyperbola_condition_l60_60564


namespace radius_of_circle_l60_60025

theorem radius_of_circle (r : ℝ) : 3 * 2 * Real.pi * r = Real.pi * r^2 → r = 6 :=
by {
  intro h,
  have h1 : 6 * Real.pi * r = Real.pi * r^2 := by rw [←mul_assoc, ←h],
  have h2 : 6 * r = r^2 := by rw [←mul_div_cancel_left 'Real.pi, h1],
  have h3 : r^2 - 6 * r = 0 := by ring,
  have h4 : r * (r - 6) = 0 := by rw h3,
  cases eq_zero_or_eq_zero_of_mul_eq_zero h4 with h5 h6,
  { exact h5, },
  { exact h6, }
} sorry

end radius_of_circle_l60_60025


namespace eval_expression_l60_60847

theorem eval_expression :
  2^0 + 9^5 / 9^3 = 82 :=
by
  have h1 : 2^0 = 1 := by sorry
  have h2 : 9^5 / 9^3 = 9^(5-3) := by sorry
  have h3 : 9^(5-3) = 9^2 := by sorry
  have h4 : 9^2 = 81 := by sorry
  sorry

end eval_expression_l60_60847


namespace missile_time_equation_l60_60784

variable (x : ℝ)

def machToMetersPerSecond := 340
def missileSpeedInMach := 26
def secondsPerMinute := 60
def distanceToTargetInKilometers := 12000
def kilometersToMeters := 1000

theorem missile_time_equation :
  (missileSpeedInMach * machToMetersPerSecond * secondsPerMinute * x) / kilometersToMeters = distanceToTargetInKilometers :=
sorry

end missile_time_equation_l60_60784


namespace print_shop_x_charge_l60_60263

theorem print_shop_x_charge :
  ∃ (x : ℝ), 60 * x + 90 = 60 * 2.75 ∧ x = 1.25 :=
by
  sorry

end print_shop_x_charge_l60_60263


namespace proof_x_squared_plus_y_squared_l60_60734

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l60_60734


namespace earbuds_cost_before_tax_l60_60309

-- Define the conditions
variable (C : ℝ) -- The cost before tax
variable (taxRate : ℝ := 0.15)
variable (totalPaid : ℝ := 230)

-- Define the main question in Lean
theorem earbuds_cost_before_tax : C + taxRate * C = totalPaid → C = 200 :=
by
  sorry

end earbuds_cost_before_tax_l60_60309


namespace largest_k_divides_3n_plus_1_l60_60529

theorem largest_k_divides_3n_plus_1 (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, k = 2 ∧ n % 2 = 1 ∧ 2^k ∣ 3^n + 1 ∨ k = 1 ∧ n % 2 = 0 ∧ 2^k ∣ 3^n + 1 :=
sorry

end largest_k_divides_3n_plus_1_l60_60529


namespace b_joined_after_x_months_l60_60513

-- Establish the given conditions as hypotheses
theorem b_joined_after_x_months
  (a_start_capital : ℝ)
  (b_start_capital : ℝ)
  (profit_ratio : ℝ)
  (months_in_year : ℝ)
  (a_capital_time : ℝ)
  (b_capital_time : ℝ)
  (a_profit_ratio : ℝ)
  (b_profit_ratio : ℝ)
  (x : ℝ)
  (h1 : a_start_capital = 3500)
  (h2 : b_start_capital = 9000)
  (h3 : profit_ratio = 2 / 3)
  (h4 : months_in_year = 12)
  (h5 : a_capital_time = 12)
  (h6 : b_capital_time = 12 - x)
  (h7 : a_profit_ratio = 2)
  (h8 : b_profit_ratio = 3)
  (h_ratio : (a_start_capital * a_capital_time) / (b_start_capital * b_capital_time) = profit_ratio) :
  x = 5 :=
by
  sorry

end b_joined_after_x_months_l60_60513


namespace closing_price_l60_60690

theorem closing_price
  (opening_price : ℝ)
  (increase_percentage : ℝ)
  (h_opening_price : opening_price = 15)
  (h_increase_percentage : increase_percentage = 6.666666666666665) :
  opening_price * (1 + increase_percentage / 100) = 16 :=
by
  sorry

end closing_price_l60_60690


namespace arithmetic_sequence_tenth_term_l60_60022

theorem arithmetic_sequence_tenth_term :
  ∀ (a : ℚ) (a_20 : ℚ) (a_10 : ℚ),
    a = 5 / 11 →
    a_20 = 9 / 11 →
    a_10 = a + (9 * ((a_20 - a) / 19)) →
    a_10 = 1233 / 2309 :=
by
  intros a a_20 a_10 h_a h_a_20 h_a_10
  sorry

end arithmetic_sequence_tenth_term_l60_60022


namespace wall_clock_time_at_car_5PM_l60_60450

-- Define the initial known conditions
def initial_time : ℕ := 7 -- 7:00 AM
def wall_time_at_10AM : ℕ := 10 -- 10:00 AM
def car_time_at_10AM : ℕ := 11 -- 11:00 AM
def car_time_at_5PM : ℕ := 17 -- 5:00 PM = 17:00 in 24-hour format

-- Define the calculations for the rate of the car clock
def rate_of_car_clock : ℚ := (car_time_at_10AM - initial_time : ℚ) / (wall_time_at_10AM - initial_time : ℚ) -- rate = 4/3

-- Prove the actual time according to the wall clock when the car clock shows 5:00 PM
theorem wall_clock_time_at_car_5PM :
  let elapsed_real_time := (car_time_at_5PM - car_time_at_10AM) * (3 : ℚ) / (4 : ℚ)
  let actual_time := wall_time_at_10AM + elapsed_real_time
  (actual_time : ℚ) = 15 + (15 / 60 : ℚ) := -- 3:15 PM as 15.25 in 24-hour time
by
  sorry

end wall_clock_time_at_car_5PM_l60_60450


namespace determine_c_absolute_value_l60_60451

theorem determine_c_absolute_value
  (a b c : ℤ)
  (h_gcd : Int.gcd (Int.gcd a b) c = 1)
  (h_root : a * (Complex.mk 3 1)^4 + b * (Complex.mk 3 1)^3 + c * (Complex.mk 3 1)^2 + b * (Complex.mk 3 1) + a = 0) :
  |c| = 109 := 
sorry

end determine_c_absolute_value_l60_60451


namespace characterization_of_points_l60_60245

def satisfies_eq (x : ℝ) (y : ℝ) : Prop :=
  max x (x^2) + min y (y^2) = 1

theorem characterization_of_points :
  ∀ x y : ℝ,
  satisfies_eq x y ↔
  ((x < 0 ∨ x > 1) ∧ (y < 0 ∨ y > 1) ∧ y ≤ 0 ∧ y = 1 - x^2) ∨
  ((x < 0 ∨ x > 1) ∧ (0 < y ∧ y < 1) ∧ x^2 + y^2 = 1 ∧ x ≤ -1 ∧ x > 0) ∨
  ((0 < x ∧ x < 1) ∧ (y < 0 ∨ y > 1) ∧ false) ∨
  ((0 < x ∧ x < 1) ∧ (0 < y ∧ y < 1) ∧ y^2 = 1 - x) :=
sorry

end characterization_of_points_l60_60245


namespace min_value_a_l60_60873

theorem min_value_a (a b : ℕ) (h1: a = b - 2005) 
  (h2: ∃ p q : ℕ, p > 0 ∧ q > 0 ∧ p + q = a ∧ p * q = b) : a ≥ 95 := sorry

end min_value_a_l60_60873


namespace completing_the_square_l60_60073

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l60_60073


namespace wax_current_amount_l60_60146

theorem wax_current_amount (wax_needed wax_total : ℕ) (h : wax_needed + 11 = wax_total) : 11 = wax_total - wax_needed :=
by
  sorry

end wax_current_amount_l60_60146


namespace find_a_if_even_function_l60_60547

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * exp x / (exp (a * x) - 1)

theorem find_a_if_even_function :
  (∀ x : ℝ, x ≠ 0 → f a (-x) = f a x) → a = 2 :=
by
  sorry

end find_a_if_even_function_l60_60547


namespace segment_length_reflection_l60_60042

theorem segment_length_reflection (Z : ℝ×ℝ) (Z' : ℝ×ℝ) (hx : Z = (5, 2)) (hx' : Z' = (5, -2)) :
  dist Z Z' = 4 := by
  sorry

end segment_length_reflection_l60_60042


namespace completing_the_square_l60_60069

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l60_60069


namespace cost_price_bicycle_A_l60_60685

variable {CP_A CP_B SP_C : ℝ}

theorem cost_price_bicycle_A (h1 : CP_B = 1.25 * CP_A) (h2 : SP_C = 1.25 * CP_B) (h3 : SP_C = 225) :
  CP_A = 144 :=
by
  sorry

end cost_price_bicycle_A_l60_60685


namespace decimal_to_base7_l60_60522

-- Define the decimal number
def decimal_number : ℕ := 2011

-- Define the base-7 conversion function
def to_base7 (n : ℕ) : List ℕ :=
  if n < 7 then [n]
  else to_base7 (n / 7) ++ [n % 7]

-- Calculate the base-7 representation of 2011
def base7_representation : List ℕ := to_base7 decimal_number

-- Prove that the base-7 representation of 2011 is [5, 6, 0, 2]
theorem decimal_to_base7 : base7_representation = [5, 6, 0, 2] :=
  by sorry

end decimal_to_base7_l60_60522


namespace exists_x0_l60_60279

noncomputable def f (x a : ℝ) : ℝ := x^2 + (Real.log (3 * x))^2 - 2 * a * (x + 3 * Real.log (3 * x)) + 10 * a^2

theorem exists_x0 (a : ℝ) (h : a = 1 / 30) : ∃ x0 : ℝ, f x0 a ≤ 1 / 10 := 
by
  sorry

end exists_x0_l60_60279


namespace inequality_solution_range_l60_60596

theorem inequality_solution_range (m : ℝ) :
  (∃ x : ℝ, 2 * x - 6 + m < 0 ∧ 4 * x - m > 0) → m < 4 :=
by
  intro h
  sorry

end inequality_solution_range_l60_60596


namespace calc_expression_correct_l60_60109

noncomputable def calc_expression : Real :=
  Real.sqrt 8 - (1 / 3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2

theorem calc_expression_correct :
  calc_expression = 3 - Real.sqrt 3 :=
sorry

end calc_expression_correct_l60_60109


namespace jigi_scored_55_percent_l60_60301

noncomputable def jigi_percentage (max_score : ℕ) (avg_score : ℕ) (gibi_pct mike_pct lizzy_pct : ℕ) : ℕ := sorry

theorem jigi_scored_55_percent :
  jigi_percentage 700 490 59 99 67 = 55 :=
sorry

end jigi_scored_55_percent_l60_60301


namespace cost_of_one_shirt_l60_60675

theorem cost_of_one_shirt
  (J S : ℝ)
  (h1 : 3 * J + 2 * S = 69)
  (h2 : 2 * J + 3 * S = 76) :
  S = 18 :=
by
  sorry

end cost_of_one_shirt_l60_60675


namespace probability_wife_selection_l60_60090

theorem probability_wife_selection (P_H P_only_one P_W : ℝ)
  (h1 : P_H = 1 / 7)
  (h2 : P_only_one = 0.28571428571428575)
  (h3 : P_only_one = (P_H * (1 - P_W)) + (P_W * (1 - P_H))) :
  P_W = 1 / 5 :=
by
  sorry

end probability_wife_selection_l60_60090


namespace solve_for_x_l60_60189

theorem solve_for_x (x : ℚ) : (1 / 3) + (1 / x) = (3 / 4) → x = 12 / 5 :=
by
  intro h
  -- Proof goes here
  sorry

end solve_for_x_l60_60189


namespace symmetric_parabola_equation_l60_60193

theorem symmetric_parabola_equation (x y : ℝ) (h : y^2 = 2 * x) : (y^2 = -2 * (x + 2)) :=
by
  sorry

end symmetric_parabola_equation_l60_60193


namespace geometric_sequence_problem_l60_60269

noncomputable def geometric_sequence (a : ℕ → ℝ) :=
  ∀ n, a n = a 0 * (1 / 2) ^ n

theorem geometric_sequence_problem 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : ∀ n, S n = (a 0 * (1 - (1 / 2 : ℝ) ^ n)) / (1 - (1 / 2)))
  (h3 : a 0 + a 2 = 5 / 2)
  (h4 : a 1 + a 3 = 5 / 4) :
  ∀ n, S n / a n = 2 ^ n - 1 :=
by
  sorry

end geometric_sequence_problem_l60_60269


namespace geometric_series_sum_l60_60759

theorem geometric_series_sum : 
  let a := 6
  let r := - (2 / 5)
  let s := a / (1 - r)
  s = 30 / 7 :=
by
  let a := 6
  let r := -(2 / 5)
  let s := a / (1 - r)
  show s = 30 / 7
  sorry

end geometric_series_sum_l60_60759


namespace solve_inequality_system_l60_60996

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l60_60996


namespace mary_initial_sugar_eq_4_l60_60770

/-- Mary is baking a cake. The recipe calls for 7 cups of sugar and she needs to add 3 more cups of sugar. -/
def total_sugar : ℕ := 7
def additional_sugar : ℕ := 3

theorem mary_initial_sugar_eq_4 :
  ∃ initial_sugar : ℕ, initial_sugar + additional_sugar = total_sugar ∧ initial_sugar = 4 :=
sorry

end mary_initial_sugar_eq_4_l60_60770


namespace milk_for_6_cookies_l60_60338

/-- Given conditions for baking cookies -/
def quarts_to_pints : ℕ := 2 -- 2 pints in a quart
def milk_for_24_cookies : ℕ := 5 -- 5 quarts of milk for 24 cookies

/-- Theorem to prove the number of pints needed to bake 6 cookies -/
theorem milk_for_6_cookies : 
  (milk_for_24_cookies * quarts_to_pints * 6 / 24 : ℝ) = 2.5 := 
by 
  sorry -- Proof is omitted

end milk_for_6_cookies_l60_60338


namespace number_of_items_in_U_l60_60664

theorem number_of_items_in_U (U A B : Finset ℕ)
  (hB : B.card = 41)
  (not_A_nor_B : U.card - A.card - B.card + (A ∩ B).card = 59)
  (hAB : (A ∩ B).card = 23)
  (hA : A.card = 116) :
  U.card = 193 :=
by sorry

end number_of_items_in_U_l60_60664


namespace find_xy_l60_60571

theorem find_xy (x y : ℝ) (h1 : x + y = 5) (h2 : x^3 + y^3 = 125) : x * y = 0 :=
by
  sorry

end find_xy_l60_60571


namespace son_l60_60354

theorem son's_age (S F : ℕ) (h1 : F = S + 26) (h2 : F + 2 = 2 * (S + 2)) : S = 24 :=
by
  sorry

end son_l60_60354


namespace total_stars_l60_60778

-- Define the daily stars earned by Shelby
def shelby_monday : Nat := 4
def shelby_tuesday : Nat := 6
def shelby_wednesday : Nat := 3
def shelby_thursday : Nat := 5
def shelby_friday : Nat := 2
def shelby_saturday : Nat := 3
def shelby_sunday : Nat := 7

-- Define the daily stars earned by Alex
def alex_monday : Nat := 5
def alex_tuesday : Nat := 3
def alex_wednesday : Nat := 6
def alex_thursday : Nat := 4
def alex_friday : Nat := 7
def alex_saturday : Nat := 2
def alex_sunday : Nat := 5

-- Define the total stars earned by Shelby in a week
def total_shelby_stars : Nat := shelby_monday + shelby_tuesday + shelby_wednesday + shelby_thursday + shelby_friday + shelby_saturday + shelby_sunday

-- Define the total stars earned by Alex in a week
def total_alex_stars : Nat := alex_monday + alex_tuesday + alex_wednesday + alex_thursday + alex_friday + alex_saturday + alex_sunday

-- The proof problem statement
theorem total_stars (total_shelby_stars total_alex_stars : Nat) : total_shelby_stars + total_alex_stars = 62 := by
  sorry

end total_stars_l60_60778


namespace not_on_line_l60_60592

-- Defining the point (0,20)
def pt : ℝ × ℝ := (0, 20)

-- Defining the line equation
def line (m b : ℝ) (p : ℝ × ℝ) : Prop := p.2 = m * p.1 + b

-- The proof problem stating that for all real numbers m and b, if m + b < 0, 
-- then the point (0, 20) cannot be on the line y = mx + b
theorem not_on_line (m b : ℝ) (h : m + b < 0) : ¬line m b pt := by
  sorry

end not_on_line_l60_60592


namespace sphere_radius_l60_60811

/-- Given the curved surface area (CSA) of a sphere and its formula, 
    prove that the radius of the sphere is 4 cm.
    Conditions:
    - CSA = 4πr²
    - Curved surface area is 64π cm²
-/
theorem sphere_radius (r : ℝ) (h : 4 * Real.pi * r^2 = 64 * Real.pi) : r = 4 := by
  sorry

end sphere_radius_l60_60811


namespace minimum_zeros_l60_60034

theorem minimum_zeros (n : ℕ) (a : Fin n → ℤ) (h : n = 2011)
  (H : ∀ i j k : Fin n, a i + a j + a k ∈ Set.range a) : 
  ∃ (num_zeros : ℕ), num_zeros ≥ 2009 ∧ (∃ f : Fin (num_zeros) → Fin n, ∀ i : Fin (num_zeros), a (f i) = 0) :=
sorry

end minimum_zeros_l60_60034


namespace karens_speed_l60_60610

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end karens_speed_l60_60610


namespace karens_speed_l60_60611

noncomputable def average_speed_karen (k : ℝ) : Prop :=
  let late_start_in_hours := 4 / 60
  let total_distance_karen := 24 + 4
  let time_karen := total_distance_karen / k
  let distance_tom_start := 45 * late_start_in_hours
  let distance_tom_total := distance_tom_start + 45 * time_karen
  distance_tom_total = 24

theorem karens_speed : average_speed_karen 60 :=
by
  sorry

end karens_speed_l60_60611


namespace minvalue_expression_l60_60620

theorem minvalue_expression (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 1) :
    9 * z / (3 * x + y) + 9 * x / (y + 3 * z) + 4 * y / (x + z) ≥ 3 := 
by
  sorry

end minvalue_expression_l60_60620


namespace exterior_angle_regular_polygon_l60_60499

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l60_60499


namespace seating_arrangements_l60_60472

theorem seating_arrangements :
  ∀ (chairs people : ℕ), 
  chairs = 8 → 
  people = 3 → 
  (∃ gaps : ℕ, gaps = 4) → 
  (∀ pos, pos = Nat.choose 3 4) → 
  pos = 24 :=
by
  intros chairs people h1 h2 h3 h4
  have gaps := 4
  have pos := Nat.choose 4 3
  sorry

end seating_arrangements_l60_60472


namespace teal_sales_l60_60304

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end teal_sales_l60_60304


namespace monotonicity_of_f_inequality_f_l60_60565

section
variables {f : ℝ → ℝ}
variables (h_dom : ∀ x, x > 0 → f x > 0)
variables (h_f2 : f 2 = 1)
variables (h_fxy : ∀ x y, f (x * y) = f x + f y)
variables (h_pos : ∀ x, 1 < x → f x > 0)

-- Monotonicity of f(x)
theorem monotonicity_of_f :
  ∀ x1 x2, 0 < x1 → x1 < x2 → f x1 < f x2 :=
sorry

-- Inequality f(x) + f(x-2) ≤ 3 
theorem inequality_f (x : ℝ) :
  2 < x ∧ x ≤ 4 → f x + f (x - 2) ≤ 3 :=
sorry

end

end monotonicity_of_f_inequality_f_l60_60565


namespace graph_not_in_second_quadrant_l60_60595

theorem graph_not_in_second_quadrant (b : ℝ) (h : ∀ x < 0, 2^x + b - 1 < 0) : b ≤ 0 :=
sorry

end graph_not_in_second_quadrant_l60_60595


namespace xiao_ming_second_half_time_l60_60078

theorem xiao_ming_second_half_time :
  ∀ (total_distance : ℕ) (speed1 : ℕ) (speed2 : ℕ), 
    total_distance = 360 →
    speed1 = 5 →
    speed2 = 4 →
    let t_total := total_distance / (speed1 + speed2) * 2
    let half_distance := total_distance / 2
    let t2 := half_distance / speed2
    half_distance / speed2 + (half_distance / speed1) = 44 :=
sorry

end xiao_ming_second_half_time_l60_60078


namespace number_of_rectangles_required_l60_60744

theorem number_of_rectangles_required
  (width : ℝ) (area : ℝ) (total_length : ℝ) (length : ℝ)
  (H1 : width = 42) (H2 : area = 1638) (H3 : total_length = 390) (H4 : length = area / width)
  : (total_length / length) = 10 := 
sorry

end number_of_rectangles_required_l60_60744


namespace bake_sale_money_raised_correct_l60_60105

def bake_sale_money_raised : Prop :=
  let chocolate_chip_cookies_baked := 4 * 12
  let oatmeal_raisin_cookies_baked := 6 * 12
  let regular_brownies_baked := 2 * 12
  let sugar_cookies_baked := 6 * 12
  let blondies_baked := 3 * 12
  let cream_cheese_swirled_brownies_baked := 5 * 12
  let chocolate_chip_cookies_price := 1.50
  let oatmeal_raisin_cookies_price := 1.00
  let regular_brownies_price := 2.50
  let sugar_cookies_price := 1.25
  let blondies_price := 2.75
  let cream_cheese_swirled_brownies_price := 3.00
  let chocolate_chip_cookies_sold := 0.75 * chocolate_chip_cookies_baked
  let oatmeal_raisin_cookies_sold := 0.85 * oatmeal_raisin_cookies_baked
  let regular_brownies_sold := 0.60 * regular_brownies_baked
  let sugar_cookies_sold := 0.90 * sugar_cookies_baked
  let blondies_sold := 0.80 * blondies_baked
  let cream_cheese_swirled_brownies_sold := 0.50 * cream_cheese_swirled_brownies_baked
  let total_money_raised := 
    chocolate_chip_cookies_sold * chocolate_chip_cookies_price + 
    oatmeal_raisin_cookies_sold * oatmeal_raisin_cookies_price + 
    regular_brownies_sold * regular_brownies_price + 
    sugar_cookies_sold * sugar_cookies_price + 
    blondies_sold * blondies_price + 
    cream_cheese_swirled_brownies_sold * cream_cheese_swirled_brownies_price
  total_money_raised = 397.00

theorem bake_sale_money_raised_correct : bake_sale_money_raised := by
  sorry

end bake_sale_money_raised_correct_l60_60105


namespace number_of_two_digit_integers_l60_60136

def digits : Finset ℕ := {2, 4, 6, 7, 8}

theorem number_of_two_digit_integers : 
  (digits.card * (digits.card - 1)) = 20 := 
by
  sorry

end number_of_two_digit_integers_l60_60136


namespace even_quadratic_iff_b_zero_l60_60808

-- Define a quadratic function
def quadratic (a b c x : ℝ) : ℝ := a * x ^ 2 + b * x + c

-- State the theorem
theorem even_quadratic_iff_b_zero (a b c : ℝ) : 
  (∀ x : ℝ, quadratic a b c x = quadratic a b c (-x)) ↔ b = 0 := 
by
  sorry

end even_quadratic_iff_b_zero_l60_60808


namespace regular_polygon_sides_l60_60495

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l60_60495


namespace x_intercept_of_line_l60_60260

-- Definition of line equation
def line_eq (x y : ℝ) : Prop := 4 * x + 7 * y = 28

-- Proposition that the x-intercept of the line 4x + 7y = 28 is (7, 0)
theorem x_intercept_of_line : line_eq 7 0 :=
by
  show 4 * 7 + 7 * 0 = 28
  sorry

end x_intercept_of_line_l60_60260


namespace travel_times_either_24_or_72_l60_60120

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l60_60120


namespace unique_zero_of_f_l60_60453

theorem unique_zero_of_f (f : ℝ → ℝ) (h1 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 16) 
  (h2 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 8) (h3 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 4) 
  (h4 : ∃! x, f x = 0 ∧ 0 < x ∧ x < 2) : ¬ ∃ x, f x = 0 ∧ 2 ≤ x ∧ x < 16 := 
by
  sorry

end unique_zero_of_f_l60_60453


namespace total_cost_eq_4800_l60_60440

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end total_cost_eq_4800_l60_60440


namespace eccentricity_of_hyperbola_l60_60409

theorem eccentricity_of_hyperbola (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_asymp : 3 * a + b = 0) :
    let c := Real.sqrt (a^2 + b^2)
    let e := c / a
    e = Real.sqrt 10 :=
by
  sorry

end eccentricity_of_hyperbola_l60_60409


namespace exterior_angle_regular_polygon_l60_60498

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l60_60498


namespace count_two_digit_primes_ending_in_3_l60_60936

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l60_60936


namespace total_time_to_complete_l60_60040

noncomputable def time_to_clean_keys (n : Nat) (t : Nat) : Nat := n * t

def assignment_time : Nat := 10
def time_per_key : Nat := 3
def remaining_keys : Nat := 14

theorem total_time_to_complete :
  time_to_clean_keys remaining_keys time_per_key + assignment_time = 52 := by
  sorry

end total_time_to_complete_l60_60040


namespace log_expression_eq_zero_l60_60085

noncomputable def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem log_expression_eq_zero : 2 * log_base 5 10 + log_base 5 0.25 = 0 :=
by
  sorry

end log_expression_eq_zero_l60_60085


namespace find_original_price_l60_60769

theorem find_original_price (reduced_price : ℝ) (percent : ℝ) (original_price : ℝ) 
  (h1 : reduced_price = 6) (h2 : percent = 0.25) (h3 : reduced_price = percent * original_price) : 
  original_price = 24 :=
sorry

end find_original_price_l60_60769


namespace f_is_even_iff_a_is_2_l60_60538

def f (a : ℝ) (x : ℝ) : ℝ := x * Real.exp x / (Real.exp (a * x) - 1)

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

theorem f_is_even_iff_a_is_2 : (∀ x, f 2 (-x) = f 2 x) ↔ ∀ a, a = 2 := 
sorry

end f_is_even_iff_a_is_2_l60_60538


namespace negation_of_exists_gt0_and_poly_gt0_l60_60797

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end negation_of_exists_gt0_and_poly_gt0_l60_60797


namespace intersect_sets_example_l60_60722

open Set

theorem intersect_sets_example : 
  let A := {x : ℝ | -1 < x ∧ x ≤ 3}
  let B := {x : ℝ | x = -2 ∨ x = -1 ∨ x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3 ∨ x = 4}
  A ∩ B = {x : ℝ | x = 0 ∨ x = 1 ∨ x = 2 ∨ x = 3} :=
by
  sorry

end intersect_sets_example_l60_60722


namespace computation_l60_60388

theorem computation :
  (13 + 12)^2 - (13 - 12)^2 = 624 :=
by
  sorry

end computation_l60_60388


namespace sausage_thickness_correct_l60_60158

noncomputable def earth_radius := 6000 -- in km
noncomputable def distance_to_sun := 150000000 -- in km
noncomputable def sausage_thickness := 44 -- in km

theorem sausage_thickness_correct :
  let R := earth_radius
  let L := distance_to_sun
  let r := Real.sqrt ((4 * R^3) / (3 * L))
  abs (r - sausage_thickness) < 10 * sausage_thickness :=
by
  sorry

end sausage_thickness_correct_l60_60158


namespace number_in_circle_Y_l60_60661

section
variables (a b c d X Y : ℕ)

theorem number_in_circle_Y :
  a + b + X = 30 ∧
  c + d + Y = 30 ∧
  a + b + c + d = 40 ∧
  X + Y + c + b = 40 ∧
  X = 9 → Y = 11 := by
  intros h
  sorry
end

end number_in_circle_Y_l60_60661


namespace boat_travel_times_l60_60113

theorem boat_travel_times (d_AB d_BC : ℕ) 
  (t_against_current t_with_current t_total_A t_total_C : ℕ) 
  (h_AB : d_AB = 3) (h_BC : d_BC = 3) 
  (h_against_current : t_against_current = 10) 
  (h_with_current : t_with_current = 8)
  (h_total_A : t_total_A = 24)
  (h_total_C : t_total_C = 72) :
  (t_total_A = 24 ∨ t_total_A = 72) ∧ (t_total_C = 24 ∨ t_total_C = 72) := 
by 
  sorry

end boat_travel_times_l60_60113


namespace ratio_comparison_l60_60327

theorem ratio_comparison (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) (h_m_lt_n : m < n) :
  (m + 3) / (n + 3) > m / n :=
sorry

end ratio_comparison_l60_60327


namespace percentage_difference_l60_60209

variable (x y : ℝ)
variable (hxy : x = 6 * y)

theorem percentage_difference : ((x - y) / x) * 100 = 83.33 := by
  sorry

end percentage_difference_l60_60209


namespace find_i_when_x_is_0_point3_l60_60292

noncomputable def find_i (x : ℝ) (i : ℝ) : Prop :=
  (10 * x + 2) / 4 - (3 * x - 6) / 18 = (2 * x + 4) / i

theorem find_i_when_x_is_0_point3 : find_i 0.3 2.9993 :=
by
  sorry

end find_i_when_x_is_0_point3_l60_60292


namespace maximum_k_l60_60010

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a * x - 2

noncomputable def f_prime (x : ℝ) (a : ℝ) : ℝ := Real.exp x - a

theorem maximum_k (x : ℝ) (h₀ : x > 0) (k : ℤ) (a := 1) (h₁ : (x - k) * f_prime x a + x + 1 > 0) : k = 2 :=
sorry

end maximum_k_l60_60010


namespace find_ab_and_m_l60_60533

noncomputable def f (a b x : ℝ) : ℝ := a * x^3 + b * x^2

theorem find_ab_and_m (a b m : ℝ) (P : ℝ × ℝ)
  (h1 : P = (-1, -2))
  (h2 : ∀ (x : ℝ), (3 * a * x^2 + 2 * b * x) = -1/3 ↔ x = -1)
  (h3 : ∀ (x : ℝ), f a b x = a * x ^ 3 + b * x ^ 2)
  : (a = -13/3 ∧ b = -19/3) ∧ (0 < m ∧ m < 38/39) :=
sorry

end find_ab_and_m_l60_60533


namespace hyperbola_eccentricity_l60_60134

theorem hyperbola_eccentricity (a b : ℝ) (h_asymptote : a = 3 * b) : 
    (a^2 + b^2) / a^2 = 10 / 9 := 
by
    sorry

end hyperbola_eccentricity_l60_60134


namespace smallest_positive_period_and_monotonic_interval_max_value_in_interval_l60_60139

def f (x : ℝ) : ℝ := (sin (π + x) - sqrt 3 * cos x * sin (2 * x)) / (2 * cos (π - x)) - 1 / 2

-- Proof Problem 1: Smallest positive period and monotonically decreasing intervals
theorem smallest_positive_period_and_monotonic_interval :
  (∀ x, f (x + π) = f x) ∧
  (∀ k : ℤ, ∀ x ∈ [π / 3 + k * π, π / 2 + k * π], f x ≤ f (π / 3 + k * π)) ∧
  (∀ k : ℤ, ∀ x ∈ [π / 2 + k * π, 5 * π / 6 + k * π], f x ≤ f (π / 2 + k * π)) :=
by sorry

-- Proof Problem 2: Maximum value and corresponding x in the interval
theorem max_value_in_interval :
  (∀ x ∈ (0, π / 2), f x ≤ 1) ∧ (f (π / 3) = 1) :=
by sorry

end smallest_positive_period_and_monotonic_interval_max_value_in_interval_l60_60139


namespace solve_for_three_times_x_plus_ten_l60_60940

theorem solve_for_three_times_x_plus_ten (x : ℝ) (h_eq : 5 * x - 7 = 15 * x + 21) : 3 * (x + 10) = 21.6 := by
  sorry

end solve_for_three_times_x_plus_ten_l60_60940


namespace december_sales_fraction_l60_60312

theorem december_sales_fraction (A : ℚ) : 
  let sales_jan_to_nov := 11 * A
  let sales_dec := 5 * A
  let total_sales := sales_jan_to_nov + sales_dec
  (sales_dec / total_sales) = 5 / 16 :=
by
  sorry

end december_sales_fraction_l60_60312


namespace new_paint_intensity_l60_60326

theorem new_paint_intensity (V : ℝ) (h1 : V > 0) :
    let initial_intensity := 0.5
    let replaced_fraction := 0.4
    let replaced_intensity := 0.25
    let new_intensity := (0.3 + 0.1 * replaced_fraction)  -- derived from (0.6 * 0.5 + 0.4 * 0.25)
    new_intensity = 0.4 :=
by
    sorry

end new_paint_intensity_l60_60326


namespace total_water_filled_jars_l60_60964

theorem total_water_filled_jars :
  ∃ x : ℕ, 
    16 * (1/4) + 12 * (1/2) + 8 * 1 + 4 * 2 + x * 3 = 56 ∧
    16 + 12 + 8 + 4 + x = 50 :=
by
  sorry

end total_water_filled_jars_l60_60964


namespace find_largest_number_l60_60124

theorem find_largest_number 
  (a b c : ℕ) 
  (h1 : a + b = 16) 
  (h2 : a + c = 20) 
  (h3 : b + c = 23) : 
  c = 19 := 
sorry

end find_largest_number_l60_60124


namespace solve_inequality_system_l60_60989

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60989


namespace invalid_perimeters_l60_60666

theorem invalid_perimeters (x : ℕ) (h1 : 18 < x) (h2 : x < 42) :
  (42 + x ≠ 58) ∧ (42 + x ≠ 85) :=
by
  sorry

end invalid_perimeters_l60_60666


namespace darnel_jogging_l60_60249

variable (j s : ℝ)

theorem darnel_jogging :
  s = 0.875 ∧ s = j + 0.125 → j = 0.750 :=
by
  intros h
  have h1 : s = 0.875 := h.1
  have h2 : s = j + 0.125 := h.2
  sorry

end darnel_jogging_l60_60249


namespace equilibrium_temperature_l60_60226

-- Initial conditions for heat capacities and masses
variables (c_B c_W m_B m_W : ℝ) (h : c_W * m_W = 3 * c_B * m_B)

-- Initial temperatures
def T_W_initial := 100
def T_B_initial := 20
def T_f_initial := 80

-- Final equilibrium temperature after second block is added
def final_temp := 68

theorem equilibrium_temperature (t : ℝ)
  (h_first_eq : c_W * m_W * (T_W_initial - T_f_initial) = c_B * m_B * (T_f_initial - T_B_initial))
  (h_second_eq : c_W * m_W * (T_f_initial - t) + c_B * m_B * (T_f_initial - t) = c_B * m_B * (t - T_B_initial)) :
  t = final_temp :=
by 
  sorry

end equilibrium_temperature_l60_60226


namespace students_playing_both_correct_l60_60747

def total_students : ℕ := 36
def football_players : ℕ := 26
def long_tennis_players : ℕ := 20
def neither_players : ℕ := 7
def students_playing_both : ℕ := 17

theorem students_playing_both_correct :
  total_students - neither_players = (football_players + long_tennis_players) - students_playing_both :=
by 
  sorry

end students_playing_both_correct_l60_60747


namespace exponential_inequality_example_l60_60029

theorem exponential_inequality_example (a b : ℝ) (h : 1.5 > 0 ∧ 1.5 ≠ 1) (h2 : 2.3 < 3.2) : 1.5 ^ 2.3 < 1.5 ^ 3.2 :=
by 
  sorry

end exponential_inequality_example_l60_60029


namespace right_triangle_of_condition_l60_60446

theorem right_triangle_of_condition
  (α β γ : ℝ)
  (h_sum : α + β + γ = 180)
  (h_trig : Real.sin γ - Real.cos α = Real.cos β) :
  (α = 90) ∨ (β = 90) :=
sorry

end right_triangle_of_condition_l60_60446


namespace common_root_implies_remaining_roots_l60_60179

variables {R : Type*} [LinearOrderedField R]

theorem common_root_implies_remaining_roots
  (a b c x1 x2 x3 : R) 
  (h_non_zero_a : a ≠ 0)
  (h_non_zero_b : b ≠ 0)
  (h_non_zero_c : c ≠ 0)
  (h_a_ne_b : a ≠ b)
  (h_common_root1 : x1^2 + a*x1 + b*c = 0)
  (h_common_root2 : x1^2 + b*x1 + c*a = 0)
  (h_root2_eq : x2^2 + a*x2 + b*c = 0)
  (h_root3_eq : x3^2 + b*x3 + c*a = 0)
  : x2^2 + c*x2 + a*b = 0 ∧ x3^2 + c*x3 + a*b = 0 :=
sorry

end common_root_implies_remaining_roots_l60_60179


namespace solve_quadratic_l60_60640

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end solve_quadratic_l60_60640


namespace number_of_outcomes_probability_div_by_four_probability_on_line_l60_60363

open Finset

-- Define the labels on the balls.
def labels := {1, 3, 5, 7, 9}

-- Define the set of all possible outcomes when drawing two balls simultaneously.
def possible_outcomes : Finset (ℕ × ℕ) :=
  finset.powersetLen 2 labels
  |>.image (λ s, (s.to_list.nth 0, s.to_list.nth 1))
  |>.filter (λ p, p.1 < p.2)

-- Calculate the probability that the sum of the labels is divisible by 4.
def div_by_four_subset : Finset (ℕ × ℕ) :=
  possible_outcomes.filter (λ p, (p.1 + p.2) % 4 = 0)

-- Calculate the probability that the point lies on the line y = x + 2.
def line_subset : Finset (ℕ × ℕ) :=
  possible_outcomes.filter (λ p, p.2 = p.1 + 2)

-- Prove the number of possible outcomes.
theorem number_of_outcomes : possible_outcomes.card = 10 := by
  sorry

-- Prove the probability that the sum of the labels is divisible by 4.
theorem probability_div_by_four : (div_by_four_subset.card : ℚ) / possible_outcomes.card = 3 / 5 := by
  sorry

-- Prove the probability that the point lies on the line y = x + 2.
theorem probability_on_line : (line_subset.card : ℚ) / possible_outcomes.card = 2 / 5 := by
  sorry

end number_of_outcomes_probability_div_by_four_probability_on_line_l60_60363


namespace new_student_bmi_l60_60356

theorem new_student_bmi 
(average_weight_29 : ℚ)
(average_height_29 : ℚ)
(average_weight_30 : ℚ)
(average_height_30 : ℚ)
(new_student_height : ℚ)
(bmi : ℚ)
(h1 : average_weight_29 = 28)
(h2 : average_height_29 = 1.5)
(h3 : average_weight_30 = 27.5)
(h4 : average_height_30 = 1.5)
(h5 : new_student_height = 1.4)
: bmi = 6.63 := 
sorry

end new_student_bmi_l60_60356


namespace completing_the_square_l60_60048

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l60_60048


namespace father_age_is_32_l60_60037

noncomputable def father_age (D F : ℕ) : Prop :=
  F = 4 * D ∧ (F + 5) + (D + 5) = 50

theorem father_age_is_32 (D F : ℕ) (h : father_age D F) : F = 32 :=
by
  sorry

end father_age_is_32_l60_60037


namespace initial_quantity_of_A_l60_60364

noncomputable def initial_quantity_of_A_in_can (initial_total_mixture : ℤ) (x : ℤ) := 7 * x

theorem initial_quantity_of_A
  (initial_ratio_A : ℤ) (initial_ratio_B : ℤ) (initial_ratio_C : ℤ)
  (initial_total_mixture : ℤ) (drawn_off_mixture : ℤ) (new_quantity_of_B : ℤ)
  (new_ratio_A : ℤ) (new_ratio_B : ℤ) (new_ratio_C : ℤ)
  (h1 : initial_ratio_A = 7) (h2 : initial_ratio_B = 5) (h3 : initial_ratio_C = 3)
  (h4 : initial_total_mixture = 15 * x)
  (h5 : new_ratio_A = 7) (h6 : new_ratio_B = 9) (h7 : new_ratio_C = 3)
  (h8 : drawn_off_mixture = 18)
  (h9 : new_quantity_of_B = 5 * x - (5 / 15) * 18 + 18)
  (h10 : (7 * x - (7 / 15) * 18) / new_quantity_of_B = 7 / 9) :
  initial_quantity_of_A_in_can initial_total_mixture x = 54 :=
by
  sorry

end initial_quantity_of_A_l60_60364


namespace problem1_problem2_l60_60280

-- Define the function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := abs (x - 2) - abs (x + a)

-- Problem 1: Find x such that f(x) < -2 when a = 1
theorem problem1 : 
  {x : ℝ | f x 1 < -2} = {x | x > 3 / 2} :=
sorry

-- Problem 2: Find the range of values for 'a' when -2 + f(y) ≤ f(x) ≤ 2 + f(y) for all x, y ∈ ℝ
theorem problem2 : 
  (∀ x y : ℝ, -2 + f y a ≤ f x a ∧ f x a ≤ 2 + f y a) ↔ (-3 ≤ a ∧ a ≤ -1) :=
sorry

end problem1_problem2_l60_60280


namespace increase_by_multiplication_l60_60683

theorem increase_by_multiplication (n : ℕ) (h : n = 14) : (15 * n) - n = 196 :=
by
  -- Skip the proof
  sorry

end increase_by_multiplication_l60_60683


namespace circle_condition_l60_60276

noncomputable def circle_eq (x y m : ℝ) : Prop := x^2 + y^2 - x + y + m = 0

theorem circle_condition (m : ℝ) : (∀ x y : ℝ, circle_eq x y m) → m < 1 / 4 :=
by
  sorry

end circle_condition_l60_60276


namespace two_digit_prime_numbers_with_units_digit_3_count_l60_60897

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l60_60897


namespace no_two_champion_teams_l60_60299

theorem no_two_champion_teams
  (T : Type) 
  (M : T -> T -> Prop)
  (superior : T -> T -> Prop)
  (champion : T -> Prop)
  (h1 : ∀ A B, M A B ∨ (∃ C, M A C ∧ M C B) → superior A B)
  (h2 : ∀ A, champion A ↔ ∀ B, superior A B)
  (h3 : ∀ A B, M A B ∨ M B A)
  : ¬ ∃ A B, champion A ∧ champion B ∧ A ≠ B := 
sorry

end no_two_champion_teams_l60_60299


namespace calc_eq_neg_ten_thirds_l60_60776

theorem calc_eq_neg_ten_thirds :
  (7 / 4 - 7 / 8 - 7 / 12) / (-7 / 8) + (-7 / 8) / (7 / 4 - 7 / 8 - 7 / 12) = -10 / 3 := by 
sorry

end calc_eq_neg_ten_thirds_l60_60776


namespace max_ab_if_circles_tangent_l60_60282

theorem max_ab_if_circles_tangent (a b : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h_eq : ∀ x y : ℝ, (x + a)^2 + (y - 2)^2 = 1 → (x - b)^2 + (y - 2)^2 = 4)
  (h_tangent : (a + b) = 3):
  ab ≤ 9 / 4 := by
  sorry

end max_ab_if_circles_tangent_l60_60282


namespace completing_square_l60_60062

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l60_60062


namespace triangle_perimeter_not_78_l60_60200

theorem triangle_perimeter_not_78 (x : ℝ) (h1 : 11 < x) (h2 : x < 37) : 13 + 24 + x ≠ 78 :=
by
  -- Using the given conditions to show the perimeter is not 78
  intro h
  have h3 : 48 < 13 + 24 + x := by linarith
  have h4 : 13 + 24 + x < 74 := by linarith
  linarith

end triangle_perimeter_not_78_l60_60200


namespace negation_of_exists_gt0_and_poly_gt0_l60_60796

theorem negation_of_exists_gt0_and_poly_gt0 :
  (¬ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5 * x₀ + 6 > 0)) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0) :=
by sorry

end negation_of_exists_gt0_and_poly_gt0_l60_60796


namespace nyusha_wins_probability_l60_60013

-- Define the number of coins Nyusha and Barash have
def N : ℕ := 2022
def B : ℕ := 2023

-- Define the probability of heads for each coin
def p : ℝ := 0.5

-- Define the probability that Nyusha wins given the conditions
theorem nyusha_wins_probability : 
  (probability (λ (X Y : ℕ), X > Y ∨ (X = Y)) = 0.5) :=
sorry

end nyusha_wins_probability_l60_60013


namespace yan_distance_ratio_l60_60103

theorem yan_distance_ratio 
  (w x y : ℝ)
  (h1 : y / w = x / w + (x + y) / (10 * w)) :
  x / y = 9 / 11 :=
by
  sorry

end yan_distance_ratio_l60_60103


namespace composite_expression_l60_60445

theorem composite_expression (n : ℕ) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^3 + 6 * n^2 + 12 * n + 7 = a * b :=
by
  sorry

end composite_expression_l60_60445


namespace find_y_l60_60687

-- Given conditions
def x : Int := 129
def student_operation (y : Int) : Int := x * y - 148
def result : Int := 110

-- The theorem statement
theorem find_y :
  ∃ y : Int, student_operation y = result ∧ y = 2 := 
sorry

end find_y_l60_60687


namespace problem_statement_l60_60188

open Probability

variables {Ω : Type*} {P : Measure Ω} {A B C : Set Ω}

-- Conditions given in the problem
axiom condition1 : condProb P A C > condProb P B C
axiom condition2 : condProb P A Cᶜ > condProb P B Cᶜ
axiom condition3 : P C ≠ 0
axiom condition4 : P Cᶜ ≠ 0

-- Statement to prove
theorem problem_statement : P A > P B :=
by
  sorry

end problem_statement_l60_60188


namespace worker_usual_time_l60_60669

theorem worker_usual_time (T : ℝ) (S : ℝ) (h₀ : S > 0) (h₁ : (4 / 5) * S * (T + 10) = S * T) : T = 40 :=
sorry

end worker_usual_time_l60_60669


namespace average_score_l60_60652

variable (K M : ℕ) (E : ℕ)

theorem average_score (h1 : (K + M) / 2 = 86) (h2 : E = 98) :
  (K + M + E) / 3 = 90 :=
by
  sorry

end average_score_l60_60652


namespace total_truck_loads_l60_60369

-- Using definitions from conditions in (a)
def sand : ℝ := 0.16666666666666666
def dirt : ℝ := 0.3333333333333333
def cement : ℝ := 0.16666666666666666

-- The proof statement based on the correct answer in (b)
theorem total_truck_loads : sand + dirt + cement = 0.6666666666666666 := 
by
  sorry

end total_truck_loads_l60_60369


namespace rain_stop_time_on_first_day_l60_60607

-- Define the problem conditions
def raining_time_day1 (x : ℕ) : Prop :=
  let start_time := 7 * 60 -- start time in minutes
  let stop_time := start_time + x * 60 -- stop time in minutes
  stop_time = 17 * 60 -- stop at 17:00 (5:00 PM)

def total_raining_time_46_hours (x : ℕ) : Prop :=
  x + (x + 2) + 2 * (x + 2) = 46

-- Main statement
theorem rain_stop_time_on_first_day (x : ℕ) (h1 : total_raining_time_46_hours x) : raining_time_day1 x :=
  sorry

end rain_stop_time_on_first_day_l60_60607


namespace largest_pos_int_divisible_l60_60345

theorem largest_pos_int_divisible (n : ℕ) (h1 : n > 0) (h2 : n + 11 ∣ n^3 + 101) : n = 1098 :=
sorry

end largest_pos_int_divisible_l60_60345


namespace other_endpoint_diameter_l60_60691

theorem other_endpoint_diameter (O : ℝ × ℝ) (A : ℝ × ℝ) (B : ℝ × ℝ) 
  (hO : O = (2, 3)) (hA : A = (-1, -1)) 
  (h_midpoint : O = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) : B = (5, 7) := by
  sorry

end other_endpoint_diameter_l60_60691


namespace vera_operations_impossible_l60_60868

theorem vera_operations_impossible (N : ℕ) : (N % 3 ≠ 0) → ¬(∃ k : ℕ, ((N + 3 * k) % 5 = 0) → ((N + 3 * k) / 5) = 1) :=
by
  sorry

end vera_operations_impossible_l60_60868


namespace parity_sum_matches_parity_of_M_l60_60489

theorem parity_sum_matches_parity_of_M (N M : ℕ) (even_numbers odd_numbers : ℕ → ℤ)
  (hn : ∀ i, i < N → even_numbers i % 2 = 0)
  (hm : ∀ i, i < M → odd_numbers i % 2 ≠ 0) : 
  (N + M) % 2 = M % 2 := 
sorry

end parity_sum_matches_parity_of_M_l60_60489


namespace total_number_of_items_l60_60865

theorem total_number_of_items (total_items : ℕ) (selected_items : ℕ) (h1 : total_items = 50) (h2 : selected_items = 10) : total_items = 50 :=
by
  exact h1

end total_number_of_items_l60_60865


namespace interval_decrease_log_l60_60457

theorem interval_decrease_log (a : ℝ) (h₁ : 0 < a) (h₂ : a ≠ 1) (h₃ : a > 1) : 
  Set.Ioo (Real.Inf $ Set.Ioo Real.Inf Real.Inf) (-1) := 
sorry

end interval_decrease_log_l60_60457


namespace smallest_c_ineq_l60_60313

noncomputable def smallest_c {d : ℕ → ℕ} (h_d : ∀ n > 0, d n ≤ d n + 1) := Real.sqrt 3

theorem smallest_c_ineq (d : ℕ → ℕ) (h_d : ∀ n > 0, (d n) ≤ d n + 1) :
  ∀ n : ℕ, n > 0 → d n ≤ smallest_c h_d * (Real.sqrt n) :=
sorry

end smallest_c_ineq_l60_60313


namespace real_roots_condition_l60_60967

-- Definitions based on conditions
def polynomial (x : ℝ) : ℝ := x^4 - 6 * x - 1
def is_root (a : ℝ) : Prop := polynomial a = 0

-- The statement we want to prove
theorem real_roots_condition (a b : ℝ) (ha: is_root a) (hb: is_root b) : 
  (a * b + 2 * a + 2 * b = 1.5 + Real.sqrt 3) := 
sorry

end real_roots_condition_l60_60967


namespace harry_travel_time_l60_60284

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l60_60284


namespace Peggy_dolls_l60_60628

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end Peggy_dolls_l60_60628


namespace completing_square_l60_60060

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l60_60060


namespace f_max_min_l60_60079

def f : ℝ → ℝ := sorry
axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom cauchy_f : ∀ x y : ℝ, f (x + y) = f x + f y
axiom less_than_zero : ∀ x : ℝ, x > 0 → f x < 0
axiom f_one : f 1 = -2

theorem f_max_min : (∀ x ∈ [-3, 3], f (-3) = 6 ∧ f 3 = -6) :=
by sorry

end f_max_min_l60_60079


namespace dreamy_vacation_note_probability_l60_60430

theorem dreamy_vacation_note_probability :
  ∃ n : ℕ, n ≤ 5 ∧ 
  (probability_binomial 5 0.4 n = 0.2304) :=
sorry

end dreamy_vacation_note_probability_l60_60430


namespace quadrilateral_inequality_l60_60003

open EuclideanGeometry

noncomputable theory
open_locale big_operators

variables {A B C D P K : Point}
variables (h1 : ConvexQuadrilateral A B C D)
variables (h2 : AB ≈ CD)
variables (h3 : ∠ P B A + ∠ P C D = 180)

theorem quadrilateral_inequality (h1 : ConvexQuadrilateral A B C D) (h2 : AB ≈ CD)
  (h3 : ∠ P B A + ∠ P C D = 180) : PB + PC < AD :=
sorry

end quadrilateral_inequality_l60_60003


namespace white_ball_probability_l60_60401

theorem white_ball_probability :
  ∀ (n : ℕ), (2/(n+2) = 2/5) → (n = 3) → (n/(n+2) = 3/5) :=
by
  sorry

end white_ball_probability_l60_60401


namespace area_ratio_BDF_FDCE_l60_60960

-- Define the vertices of the triangle
variables {A B C : Point}
-- Define the points on the sides and midpoints
variables {E D F : Point}
-- Define angles and relevant properties
variables (angle_CBA : Angle B C A = 72)
variables (midpoint_E : Midpoint E A C)
variables (ratio_D : RatioSegment B D D C = 2)
-- Define intersection point F
variables (intersect_F : IntersectLineSegments (LineSegment A D) (LineSegment B E) = F)

theorem area_ratio_BDF_FDCE (h_angle : angle_CBA = 72) 
  (h_midpoint_E : midpoint_E) (h_ratio_D : ratio_D) (h_intersect_F : intersect_F)
  : area_ratio (Triangle.area B D F) (Quadrilateral.area F D C E) = 1 / 5 :=
sorry

end area_ratio_BDF_FDCE_l60_60960


namespace cory_fruit_eating_orders_l60_60248

open Nat

theorem cory_fruit_eating_orders : 
    let apples := 4
    let oranges := 3
    let bananas := 2
    let grape := 1
    let total_fruits := apples + oranges + bananas + grape
    apples + oranges + bananas + grape = 10 →
    total_fruits = 10 →
    apples ≥ 1 →
    factorial 9 / (factorial 3 * factorial 3 * factorial 2 * factorial 1) = 5040 :=
by
  intros apples oranges bananas grape total_fruits h_total h_sum h_apples
  sorry

end cory_fruit_eating_orders_l60_60248


namespace range_of_a_maximum_of_z_l60_60970

-- Problem 1
theorem range_of_a (a b : ℝ) (h1 : a + 2 * b = 9) (h2 : |9 - 2 * b| + |a + 1| < 3) :
  -2 < a ∧ a < 1 :=
sorry

-- Problem 2
theorem maximum_of_z (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2 * b = 9) :
  ∃ z, z = a * b^2 ∧ z ≤ 27 :=
sorry


end range_of_a_maximum_of_z_l60_60970


namespace intersection_complement_l60_60730

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)
variable (hU : U = {1, 2, 3, 4, 5}) (hA : A = {1, 2, 3}) (hB : B = {1, 4})

theorem intersection_complement :
  A ∩ (U \ B) = {2, 3} := by
  sorry

end intersection_complement_l60_60730


namespace tan_of_right_triangle_l60_60524

theorem tan_of_right_triangle (A B C : ℝ) (h : A^2 + B^2 = C^2) (hA : A = 30) (hC : C = 37) : 
  (37^2 - 30^2).sqrt / 30 = (469).sqrt / 30 := by
  sorry

end tan_of_right_triangle_l60_60524


namespace count_two_digit_prime_numbers_ending_in_3_l60_60914

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l60_60914


namespace count_prime_units_digit_3_eq_6_l60_60923

theorem count_prime_units_digit_3_eq_6 :
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.card = 6 :=
by
  sorry

end count_prime_units_digit_3_eq_6_l60_60923


namespace stones_on_perimeter_of_square_l60_60799

theorem stones_on_perimeter_of_square (n : ℕ) (h : n = 5) : 
  4 * n - 4 = 16 :=
by
  sorry

end stones_on_perimeter_of_square_l60_60799


namespace sin_seventeen_pi_over_four_l60_60859

theorem sin_seventeen_pi_over_four : Real.sin (17 * Real.pi / 4) = Real.sqrt 2 / 2 := sorry

end sin_seventeen_pi_over_four_l60_60859


namespace number_of_dress_designs_l60_60224

theorem number_of_dress_designs :
  let colors := 5
  let patterns := 4
  let sleeve_designs := 3
  colors * patterns * sleeve_designs = 60 := by
  sorry

end number_of_dress_designs_l60_60224


namespace inequality_system_solution_l60_60985

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60985


namespace even_function_implies_a_eq_2_l60_60561

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l60_60561


namespace total_wicks_20_l60_60098

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_wicks_20_l60_60098


namespace solve_quadratic_l60_60641

theorem solve_quadratic (x : ℝ) (h₁ : x > 0) (h₂ : 3 * x^2 - 7 * x - 6 = 0) : x = 3 :=
by
  sorry

end solve_quadratic_l60_60641


namespace right_triangle_conditions_l60_60828

-- Definitions
def is_right_triangle (A B C : ℝ) : Prop :=
  A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)

-- Conditions
def cond1 (A B C : ℝ) : Prop := A + B = C
def cond2 (A B C : ℝ) : Prop := A / B = 1 / 2 ∧ B / C = 2 / 3
def cond3 (A B C : ℝ) : Prop := A = B ∧ B = 2 * C
def cond4 (A B C : ℝ) : Prop := A = 2 * B ∧ B = 3 * C

-- Problem statement
theorem right_triangle_conditions (A B C : ℝ) :
  (cond1 A B C → is_right_triangle A B C) ∧
  (cond2 A B C → is_right_triangle A B C) ∧
  ¬(cond3 A B C → is_right_triangle A B C) ∧
  ¬(cond4 A B C → is_right_triangle A B C) :=
by
  sorry

end right_triangle_conditions_l60_60828


namespace regular_polygon_sides_l60_60512

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l60_60512


namespace marbles_difference_l60_60436

theorem marbles_difference : 10 - 8 = 2 :=
by
  sorry

end marbles_difference_l60_60436


namespace total_files_deleted_l60_60841

theorem total_files_deleted 
  (initial_files : ℕ) (initial_apps : ℕ)
  (deleted_files1 : ℕ) (deleted_apps1 : ℕ)
  (added_files1 : ℕ) (added_apps1 : ℕ)
  (deleted_files2 : ℕ) (deleted_apps2 : ℕ)
  (added_files2 : ℕ) (added_apps2 : ℕ)
  (final_files : ℕ) (final_apps : ℕ)
  (h_initial_files : initial_files = 24)
  (h_initial_apps : initial_apps = 13)
  (h_deleted_files1 : deleted_files1 = 5)
  (h_deleted_apps1 : deleted_apps1 = 3)
  (h_added_files1 : added_files1 = 7)
  (h_added_apps1 : added_apps1 = 4)
  (h_deleted_files2 : deleted_files2 = 10)
  (h_deleted_apps2 : deleted_apps2 = 4)
  (h_added_files2 : added_files2 = 5)
  (h_added_apps2 : added_apps2 = 7)
  (h_final_files : final_files = 21)
  (h_final_apps : final_apps = 17) :
  (deleted_files1 + deleted_files2 = 15) := 
by
  sorry

end total_files_deleted_l60_60841


namespace no_such_point_exists_l60_60686

theorem no_such_point_exists 
  (side_length : ℝ)
  (original_area : ℝ)
  (total_area_after_first_rotation : ℝ)
  (total_area_after_second_rotation : ℝ)
  (no_overlapping_exists : Prop) :
  side_length = 12 → 
  original_area = 144 → 
  total_area_after_first_rotation = 211 → 
  total_area_after_second_rotation = 287 →
  no_overlapping_exists := sorry

end no_such_point_exists_l60_60686


namespace factor_t_squared_minus_144_l60_60256

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end factor_t_squared_minus_144_l60_60256


namespace determine_asymptotes_l60_60408

noncomputable def asymptotes_of_hyperbola (a b : ℝ) (ha : 0 < a) (hb : 0 < b) : Prop :=
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  (∀ x y : ℝ, (y = x * (Real.sqrt 2 / 2) ∨ y = -x * (Real.sqrt 2 / 2)))

theorem determine_asymptotes (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (2 * a = 2 * Real.sqrt 2) ∧ (2 * b = 2) → 
  asymptotes_of_hyperbola a b ha hb :=
by
  intros h
  sorry

end determine_asymptotes_l60_60408


namespace range_of_values_for_a_l60_60951

theorem range_of_values_for_a (a : ℝ) : 
  (∃ x1 x2 : ℝ, x1 > 0 ∧ x2 < 0 ∧ x1^2 + a * x1 + a^2 - 1 = 0 ∧ x2^2 + a * x2 + a^2 - 1 = 0) → (-1 < a ∧ a < 1) :=
by
  sorry

end range_of_values_for_a_l60_60951


namespace min_integer_solution_l60_60656

theorem min_integer_solution (x : ℤ) (h1 : 3 - x > 0) (h2 : (4 * x / 3 : ℚ) + 3 / 2 > -(x / 6)) : x = 0 := by
  sorry

end min_integer_solution_l60_60656


namespace circle_radius_given_circumference_l60_60487

theorem circle_radius_given_circumference (C : ℝ) (hC : C = 3.14) : ∃ r : ℝ, C = 2 * Real.pi * r ∧ r = 0.5 := 
by
  sorry

end circle_radius_given_circumference_l60_60487


namespace gcd_987654_876543_eq_3_l60_60802

theorem gcd_987654_876543_eq_3 :
  Nat.gcd 987654 876543 = 3 :=
sorry

end gcd_987654_876543_eq_3_l60_60802


namespace flight_cost_A_to_B_l60_60185

-- Definitions based on conditions in the problem
def distance_AB : ℝ := 2000
def flight_cost_per_km : ℝ := 0.10
def booking_fee : ℝ := 100

-- Statement: Given the distances and cost conditions, the flight cost from A to B is $300
theorem flight_cost_A_to_B : distance_AB * flight_cost_per_km + booking_fee = 300 := by
  sorry

end flight_cost_A_to_B_l60_60185


namespace log_base_30_of_8_l60_60701

theorem log_base_30_of_8 (a b : Real) (h1 : Real.log 5 = a) (h2 : Real.log 3 = b) : 
    Real.logb 30 8 = 3 * (1 - a) / (b + 1) := 
  sorry

end log_base_30_of_8_l60_60701


namespace Peggy_dolls_l60_60629

theorem Peggy_dolls (initial_dolls granny_dolls birthday_dolls : ℕ) (h1 : initial_dolls = 6) (h2 : granny_dolls = 30) (h3 : birthday_dolls = granny_dolls / 2) : 
  initial_dolls + granny_dolls + birthday_dolls = 51 := by
  sorry

end Peggy_dolls_l60_60629


namespace volume_of_polyhedron_l60_60033

theorem volume_of_polyhedron (V : ℝ) (hV : 0 ≤ V) :
  ∃ P : ℝ, P = V / 6 :=
by
  sorry

end volume_of_polyhedron_l60_60033


namespace count_two_digit_prime_numbers_with_units_digit_3_l60_60899

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l60_60899


namespace find_a_l60_60008

def lambda : Set ℝ := { x | ∃ (a b : ℤ), x = a + b * Real.sqrt 3 }

theorem find_a (a : ℤ) (x : ℝ)
  (h1 : x = 7 + a * Real.sqrt 3)
  (h2 : x ∈ lambda)
  (h3 : (1 / x) ∈ lambda) :
  a = 4 ∨ a = -4 :=
sorry

end find_a_l60_60008


namespace fraction_sum_equals_zero_l60_60384

theorem fraction_sum_equals_zero :
  (1 / 12) + (2 / 12) + (3 / 12) + (4 / 12) + (5 / 12) + (6 / 12) + (7 / 12) + (8 / 12) + (9 / 12) - (45 / 12) = 0 :=
by
  sorry

end fraction_sum_equals_zero_l60_60384


namespace find_sum_abc_l60_60613

-- Define the real numbers a, b, c
variables {a b c : ℝ}

-- Define the conditions that a, b, c are positive reals.
axiom ha_pos : 0 < a
axiom hb_pos : 0 < b
axiom hc_pos : 0 < c

-- Define the condition that a^2 + b^2 + c^2 = 989
axiom habc_sq : a^2 + b^2 + c^2 = 989

-- Define the condition that (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013
axiom habc_sq_sum : (a+b)^2 + (b+c)^2 + (c+a)^2 = 2013

-- The proposition to be proven
theorem find_sum_abc : a + b + c = 32 :=
by
  -- ...(proof goes here)
  sorry

end find_sum_abc_l60_60613


namespace smallest_lcm_example_l60_60943

noncomputable def smallest_lcm (a b : ℕ) : ℕ :=
  if h : a > 999 ∧ a < 10000 ∧ b > 999 ∧ b < 10000 ∧ gcd a b = 5 then
    Nat.lcm a b
  else 0

theorem smallest_lcm_example :
  smallest_lcm 1005 1010 = 203010 :=
by
  unfold smallest_lcm
  split_ifs
  · simp [h]
  · contradiction

end smallest_lcm_example_l60_60943


namespace smallest_positive_period_l60_60251

theorem smallest_positive_period :
  ∀ (x : ℝ), 5 * Real.sin ((π / 6) - (π / 3) * x) = 5 * Real.sin ((π / 6) - (π / 3) * (x + 6)) :=
by
  sorry

end smallest_positive_period_l60_60251


namespace num_two_digit_prime_with_units_digit_3_eq_6_l60_60894

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l60_60894


namespace remainder_calculation_l60_60346

theorem remainder_calculation : 
  ∀ (dividend divisor quotient remainder : ℕ), 
  dividend = 158 →
  divisor = 17 →
  quotient = 9 →
  dividend = divisor * quotient + remainder →
  remainder = 5 :=
by
  intros dividend divisor quotient remainder hdividend hdivisor hquotient heq
  sorry

end remainder_calculation_l60_60346


namespace regular_polygon_sides_l60_60508

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l60_60508


namespace common_divisors_count_l60_60891

-- Given conditions
def num1 : ℕ := 9240
def num2 : ℕ := 8000

-- Prime factorizations from conditions
def fact_num1 : List ℕ := [2^3, 3^1, 5^1, 7^2]
def fact_num2 : List ℕ := [2^6, 5^3]

-- Computing gcd based on factorizations
def gcd : ℕ := 2^3 * 5^1

-- The goal is to prove the number of divisors of gcd is 8
theorem common_divisors_count : 
  ∃ d, d = (3+1)*(1+1) ∧ d = 8 := 
by
  sorry

end common_divisors_count_l60_60891


namespace area_enclosed_is_one_third_l60_60191

theorem area_enclosed_is_one_third :
  ∫ x in (0:ℝ)..1, (x^(1/2) - x^2 : ℝ) = (1/3 : ℝ) :=
by
  sorry

end area_enclosed_is_one_third_l60_60191


namespace fraction_red_knights_magical_l60_60957

theorem fraction_red_knights_magical (total_knights red_knights blue_knights magical_knights : ℕ)
  (fraction_red fraction_magical : ℚ)
  (frac_red_mag : ℚ) :
  (red_knights = total_knights * fraction_red) →
  (fraction_red = 3 / 8) →
  (magical_knights = total_knights * fraction_magical) →
  (fraction_magical = 1 / 4) →
  (frac_red_mag * red_knights + (frac_red_mag / 3) * blue_knights = magical_knights) →
  (frac_red_mag = 3 / 7) :=
by
  -- Skipping proof
  sorry

end fraction_red_knights_magical_l60_60957


namespace calc_two_pow_a_mul_two_pow_b_l60_60293

theorem calc_two_pow_a_mul_two_pow_b {a b : ℕ} (h1 : 0 < a) (h2 : 0 < b) (h3 : (2^a)^b = 2^2) :
  2^a * 2^b = 8 :=
sorry

end calc_two_pow_a_mul_two_pow_b_l60_60293


namespace two_digit_prime_numbers_with_units_digit_3_count_l60_60895

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l60_60895


namespace regular_polygon_sides_l60_60509

theorem regular_polygon_sides (n : ℕ) (h : 1 < n) (exterior_angle : ℝ) (h_ext : exterior_angle = 18) :
  n * exterior_angle = 360 → n = 20 :=
by 
  sorry

end regular_polygon_sides_l60_60509


namespace three_digit_numbers_mod_1000_l60_60761

theorem three_digit_numbers_mod_1000 (n : ℕ) (h_lower : 100 ≤ n) (h_upper : n ≤ 999) : 
  (n^2 ≡ n [MOD 1000]) ↔ (n = 376 ∨ n = 625) :=
by sorry

end three_digit_numbers_mod_1000_l60_60761


namespace max_travel_within_budget_l60_60816

noncomputable def rental_cost_per_day : ℝ := 30
noncomputable def insurance_fee_per_day : ℝ := 10
noncomputable def mileage_cost_per_mile : ℝ := 0.18
noncomputable def budget : ℝ := 75
noncomputable def minimum_required_travel : ℝ := 100

theorem max_travel_within_budget : ∀ (rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel), 
  rental_cost_per_day = 30 → 
  insurance_fee_per_day = 10 → 
  mileage_cost_per_mile = 0.18 → 
  budget = 75 →
  minimum_required_travel = 100 →
  (minimum_required_travel + (budget - rental_cost_per_day - insurance_fee_per_day - mileage_cost_per_mile * minimum_required_travel) / mileage_cost_per_mile) = 194 := 
by
  intros rental_cost_per_day insurance_fee_per_day mileage_cost_per_mile budget minimum_required_travel h₁ h₂ h₃ h₄ h₅
  rw [h₁, h₂, h₃, h₄, h₅]
  sorry

end max_travel_within_budget_l60_60816


namespace power_ordering_l60_60697

theorem power_ordering (a b c : ℝ) : 
  (a = 2^30) → (b = 6^10) → (c = 3^20) → (a < b) ∧ (b < c) :=
by
  intros ha hb hc
  rw [ha, hb, hc]
  have h1 : 6^10 = (3 * 2)^10 := by sorry
  have h2 : 3^20 = (3^10)^2 := by sorry
  have h3 : 2^30 = (2^10)^3 := by sorry
  sorry

end power_ordering_l60_60697


namespace cooling_constant_l60_60671

theorem cooling_constant (θ0 θ1 θ t k : ℝ) (h1 : θ1 = 60) (h0 : θ0 = 15) (ht : t = 3) (hθ : θ = 42)
  (h_temp_formula : θ = θ0 + (θ1 - θ0) * Real.exp (-k * t)) :
  k = 0.17 :=
by sorry

end cooling_constant_l60_60671


namespace quarters_for_chips_l60_60318

def total_quarters : ℕ := 16
def quarters_for_soda : ℕ := 12

theorem quarters_for_chips : (total_quarters - quarters_for_soda) = 4 :=
  by 
    sorry

end quarters_for_chips_l60_60318


namespace regular_polygon_sides_l60_60497

theorem regular_polygon_sides (n : ℕ) (h : 360 = 18 * n) : n = 20 := 
by 
  sorry

end regular_polygon_sides_l60_60497


namespace almond_walnut_ratio_is_5_to_2_l60_60088

-- Definitions based on conditions
variables (A W : ℕ)
def almond_ratio_to_walnut_ratio := A / (2 * W)
def weight_of_almonds := 250
def total_weight := 350
def weight_of_walnuts := total_weight - weight_of_almonds

-- Theorem to prove
theorem almond_walnut_ratio_is_5_to_2
  (h_ratio : almond_ratio_to_walnut_ratio A W = 250 / 100)
  (h_weights : weight_of_walnuts = 100) :
  A = 5 ∧ 2 * W = 2 := by
  sorry

end almond_walnut_ratio_is_5_to_2_l60_60088


namespace correct_exponentiation_l60_60075

theorem correct_exponentiation (a : ℝ) :
  (a^2 * a^3 = a^5) ∧
  (a^2 + a^3 ≠ a^5) ∧
  (a^6 + a^2 ≠ a^4) ∧
  (3 * a^3 - a^2 ≠ 2 * a) :=
by
  sorry

end correct_exponentiation_l60_60075


namespace completing_the_square_l60_60065

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l60_60065


namespace roots_eq_solution_l60_60173

noncomputable def roots_eq (a b c x : ℝ) : Prop :=
  a * x^2 + b * x + c = 0

noncomputable def quadratic_roots (m n : ℝ) : Prop :=
  roots_eq 1 (-2) (-2025) m ∧ roots_eq 1 (-2) (-2025) n

theorem roots_eq_solution (m n : ℝ) (hm : roots_eq 1 (-2) (-2025) m) (hn : roots_eq 1 (-2) (-2025) n) : 
  m^2 - 3 * m - n = 2023 := 
sorry

end roots_eq_solution_l60_60173


namespace teal_sales_l60_60303

theorem teal_sales
  (pumpkin_pie_slices : ℕ := 8)
  (custard_pie_slices : ℕ := 6)
  (pumpkin_pie_price : ℕ := 5)
  (custard_pie_price : ℕ := 6)
  (pumpkin_pies_sold : ℕ := 4)
  (custard_pies_sold : ℕ := 5) :
  let total_pumpkin_slices := pumpkin_pie_slices * pumpkin_pies_sold
  let total_custard_slices := custard_pie_slices * custard_pies_sold
  let total_pumpkin_sales := total_pumpkin_slices * pumpkin_pie_price
  let total_custard_sales := total_custard_slices * custard_pie_price
  let total_sales := total_pumpkin_sales + total_custard_sales
  total_sales = 340 :=
by
  sorry

end teal_sales_l60_60303


namespace sum_of_absolute_values_l60_60275

theorem sum_of_absolute_values (S : ℕ → ℤ) (a : ℕ → ℤ) :
  (∀ n, S n = n^2 - 4 * n + 2) →
  a 1 = -1 →
  (∀ n, 1 < n → a n = 2 * n - 5) →
  ((abs (a 1) + abs (a 2) + abs (a 3) + abs (a 4) + abs (a 5) +
    abs (a 6) + abs (a 7) + abs (a 8) + abs (a 9) + abs (a 10)) = 66) :=
by
  intros hS a1_eq ha_eq
  sorry

end sum_of_absolute_values_l60_60275


namespace two_digit_primes_with_units_digit_three_l60_60937

theorem two_digit_primes_with_units_digit_three :
  (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n ≥ 10))).card = 6 := 
sorry

end two_digit_primes_with_units_digit_three_l60_60937


namespace find_m_from_hyperbola_and_parabola_l60_60410

theorem find_m_from_hyperbola_and_parabola (a m : ℝ) 
  (h_eccentricity : (Real.sqrt (a^2 + 4)) / a = 3 * Real.sqrt 5 / 5) 
  (h_focus_coincide : (m / 4) = -3) : m = -12 := 
  sorry

end find_m_from_hyperbola_and_parabola_l60_60410


namespace charity_donation_ratio_l60_60751

theorem charity_donation_ratio :
  let total_winnings := 114
  let hot_dog_cost := 2
  let remaining_amount := 55
  let donation_amount := 114 - (remaining_amount + hot_dog_cost)
  donation_amount = 55 :=
by
  sorry

end charity_donation_ratio_l60_60751


namespace image_of_neg2_3_preimages_2_neg3_l60_60758

variables {A B : Type}
def f (x y : ℤ) : ℤ × ℤ := (x + y, x * y)

-- Prove that the image of (-2, 3) under f is (1, -6)
theorem image_of_neg2_3 : f (-2) 3 = (1, -6) := sorry

-- Find the preimages of (2, -3) under f
def preimages_of_2_neg3 (p : ℤ × ℤ) : Prop := f p.1 p.2 = (2, -3)

theorem preimages_2_neg3 : preimages_of_2_neg3 (-1, 3) ∧ preimages_of_2_neg3 (3, -1) := sorry

end image_of_neg2_3_preimages_2_neg3_l60_60758


namespace ethanol_percentage_in_fuel_B_l60_60237

theorem ethanol_percentage_in_fuel_B 
  (tank_capacity : ℕ)
  (fuel_A_vol : ℕ)
  (ethanol_in_A_percentage : ℝ)
  (ethanol_total : ℝ)
  (ethanol_A_vol : ℝ)
  (fuel_B_vol : ℕ)
  (ethanol_B_vol : ℝ)
  (ethanol_B_percentage : ℝ) 
  (h1 : tank_capacity = 204)
  (h2 : fuel_A_vol = 66)
  (h3 : ethanol_in_A_percentage = 0.12)
  (h4 : ethanol_total = 30)
  (h5 : ethanol_A_vol = fuel_A_vol * ethanol_in_A_percentage)
  (h6 : ethanol_B_vol = ethanol_total - ethanol_A_vol)
  (h7 : fuel_B_vol = tank_capacity - fuel_A_vol)
  (h8 : ethanol_B_percentage = (ethanol_B_vol / fuel_B_vol) * 100) :
  ethanol_B_percentage = 16 :=
by sorry

end ethanol_percentage_in_fuel_B_l60_60237


namespace increase_interval_abs_diff_l60_60655

noncomputable def abs_diff (x : ℝ) : ℝ := abs (x - 1) + abs (x + 1)

theorem increase_interval_abs_diff : ∀ x : ℝ, 1 < x → ∃ ε > 0, ∀ y ∈ set.Ioc x (x + ε), abs_diff y > abs_diff x := by
  sorry

end increase_interval_abs_diff_l60_60655


namespace increasing_on_interval_l60_60429

theorem increasing_on_interval (a : ℝ) : (∀ x : ℝ, x > 1/2 → (2 * x + a + 1 / x^2) ≥ 0) → a ≥ -3 :=
by
  intros h
  -- Rest of the proof would go here
  sorry

end increasing_on_interval_l60_60429


namespace cookies_per_day_l60_60860

theorem cookies_per_day (cost_per_cookie : ℕ) (total_spent : ℕ) (days_in_march : ℕ) (h1 : cost_per_cookie = 16) (h2 : total_spent = 992) (h3 : days_in_march = 31) :
  (total_spent / cost_per_cookie) / days_in_march = 2 :=
by sorry

end cookies_per_day_l60_60860


namespace symmetric_points_y_axis_l60_60419

theorem symmetric_points_y_axis (a b : ℤ) 
  (h1 : a + 1 = 2) 
  (h2 : b + 2 = 3) : 
  a + b = 2 :=
by
  sorry

end symmetric_points_y_axis_l60_60419


namespace men_count_eq_eight_l60_60491

theorem men_count_eq_eight (M W B : ℕ) (total_earnings : ℝ) (men_wages : ℝ)
  (H1 : M = W) (H2 : W = B) (H3 : B = 8)
  (H4 : total_earnings = 105) (H5 : men_wages = 7) :
  M = 8 := 
by 
  -- We need to show M = 8 given conditions
  sorry

end men_count_eq_eight_l60_60491


namespace quadratic_roots_problem_l60_60176

theorem quadratic_roots_problem (m n : ℝ) (h1 : m^2 - 2 * m - 2025 = 0) (h2 : n^2 - 2 * n - 2025 = 0) (h3 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 :=
sorry

end quadratic_roots_problem_l60_60176


namespace even_function_implies_a_eq_2_l60_60555

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l60_60555


namespace january_roses_l60_60790

theorem january_roses (r_october r_november r_december r_february r_january : ℕ)
  (h_october_november : r_november = r_october + 12)
  (h_november_december : r_december = r_november + 12)
  (h_december_january : r_january = r_december + 12)
  (h_january_february : r_february = r_january + 12) :
  r_january = 144 :=
by {
  -- The proof would go here.
  sorry
}

end january_roses_l60_60790


namespace problem_x2_plus_y2_l60_60742

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l60_60742


namespace fraction_value_l60_60032

theorem fraction_value : (1998 - 998) / 1000 = 1 :=
by
  sorry

end fraction_value_l60_60032


namespace min_chord_length_l60_60423

variable (α : ℝ)

def curve_eq (x y α : ℝ) :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line_eq (x : ℝ) :=
  x = Real.pi / 4

theorem min_chord_length :
  ∃ d, (∀ α : ℝ, ∃ y1 y2 : ℝ, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ d = |y2 - y1|) ∧
  (∀ α : ℝ, ∃ y1 y2, curve_eq (Real.pi / 4) y1 α ∧ curve_eq (Real.pi / 4) y2 α ∧ |y2 - y1| ≥ d) :=
sorry

end min_chord_length_l60_60423


namespace lowest_possible_price_l60_60674

theorem lowest_possible_price
  (manufacturer_suggested_price : ℝ := 45)
  (regular_discount_percentage : ℝ := 0.30)
  (sale_discount_percentage : ℝ := 0.20)
  (regular_discounted_price : ℝ := manufacturer_suggested_price * (1 - regular_discount_percentage))
  (final_price : ℝ := regular_discounted_price * (1 - sale_discount_percentage)) :
  final_price = 25.20 :=
by sorry

end lowest_possible_price_l60_60674


namespace day50_yearM_minus1_is_Friday_l60_60961

-- Define weekdays
inductive Weekday
| Sunday | Monday | Tuesday | Wednesday | Thursday | Friday | Saturday

open Weekday

-- Define days of the week for specific days in given years
def day_of (d : Nat) (reference_day : Weekday) (reference_day_mod : Nat) : Weekday :=
  match (reference_day_mod + d - 1) % 7 with
  | 0 => Sunday
  | 1 => Monday
  | 2 => Tuesday
  | 3 => Wednesday
  | 4 => Thursday
  | 5 => Friday
  | 6 => Saturday
  | _ => Thursday -- This case should never occur due to mod 7

def day250_yearM : Weekday := Thursday
def day150_yearM1 : Weekday := Thursday

-- Theorem to prove
theorem day50_yearM_minus1_is_Friday :
    day_of 50 day250_yearM 6 = Friday :=
sorry

end day50_yearM_minus1_is_Friday_l60_60961


namespace arithmetic_sum_s6_l60_60876

theorem arithmetic_sum_s6 (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℕ) 
  (h1 : ∀ n, a (n+1) - a n = d)
  (h2 : a 1 = 2)
  (h3 : S 4 = 20)
  (hS : ∀ n, S n = n * a 1 + (n * (n - 1) / 2) * d) :
  S 6 = 42 :=
by sorry

end arithmetic_sum_s6_l60_60876


namespace problem_equivalence_l60_60416

theorem problem_equivalence :
  (∃ a a1 a2 a3 a4 a5 : ℝ, ((1 - x)^5 = a + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5)) → 
  ∀ (a a1 a2 a3 a4 a5 : ℝ), (1 - 1)^5 = a + a1 + a2 + a3 + a4 + a5 →
  (1 + 1)^5 = a - a1 + a2 - a3 + a4 - a5 →
  (a + a2 + a4) * (a1 + a3 + a5) = -256 := 
by
  intros h a a1 a2 a3 a4 a5 e1 e2
  sorry

end problem_equivalence_l60_60416


namespace probability_cond_satisfied_l60_60315

-- Define the floor and log conditions
def cond1 (x : ℝ) : Prop := ⌊Real.log x / Real.log 2 + 1⌋ = ⌊Real.log x / Real.log 2⌋
def cond2 (x : ℝ) : Prop := ⌊Real.log (2 * x) / Real.log 2 + 1⌋ = ⌊Real.log (2 * x) / Real.log 2⌋
def valid_interval (x : ℝ) : Prop := 0 < x ∧ x < 1

-- Main theorem stating the proof problem
theorem probability_cond_satisfied : 
  (∀ (x : ℝ), valid_interval x → cond1 x → cond2 x → x ∈ Set.Icc (0.25:ℝ) 0.5) → 
  (0.5 - 0.25) / 1 = 1 / 4 := 
by
  -- Proof omitted
  sorry

end probability_cond_satisfied_l60_60315


namespace find_a_even_function_l60_60558

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

theorem find_a_even_function :
  (is_even_function (given_function 2)) → true :=
by
  intro h
  sorry

end find_a_even_function_l60_60558


namespace greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l60_60804

theorem greatest_possible_sum_of_two_consecutive_integers_product_lt_1000 : 
  ∃ n : ℤ, (n * (n + 1) < 1000) ∧ (n + (n + 1) = 63) :=
sorry

end greatest_possible_sum_of_two_consecutive_integers_product_lt_1000_l60_60804


namespace blue_hat_cost_l60_60202

theorem blue_hat_cost :
  ∀ (total_hats green_hats total_price green_hat_price blue_hat_price) 
  (B : ℕ),
  total_hats = 85 →
  green_hats = 30 →
  total_price = 540 →
  green_hat_price = 7 →
  blue_hat_price = B →
  (30 * 7) + (55 * B) = 540 →
  B = 6 := sorry

end blue_hat_cost_l60_60202


namespace melanie_turnips_l60_60181

theorem melanie_turnips (benny_turnips total_turnips melanie_turnips : ℕ) 
  (h1 : benny_turnips = 113) 
  (h2 : total_turnips = 252) 
  (h3 : total_turnips = benny_turnips + melanie_turnips) : 
  melanie_turnips = 139 :=
by
  sorry

end melanie_turnips_l60_60181


namespace number_of_sides_of_regular_polygon_l60_60501

variable {α : Type*}

noncomputable def exterior_angle_of_regular_polygon (n : ℕ) : ℝ := 360 / n

theorem number_of_sides_of_regular_polygon (exterior_angle : ℝ) (n : ℕ) (h₁ : exterior_angle = 18) (h₂ : exterior_angle_of_regular_polygon n = exterior_angle) : n = 20 :=
by {
  -- Translation of given conditions
  have sum_exterior_angles : 360 = n * exterior_angle_of_regular_polygon n,
  -- Based on h₂ and h₁ provided
  rw [h₂, h₁] at sum_exterior_angles,
  -- Perform necessary simplifications
  have : 20 = (360 / 18 : ℕ),
  simp,
}

end number_of_sides_of_regular_polygon_l60_60501


namespace regular_price_of_tire_l60_60101

theorem regular_price_of_tire (p : ℝ) (h : 2 * p + p / 2 = 270) : p = 108 :=
sorry

end regular_price_of_tire_l60_60101


namespace chloe_at_least_85_nickels_l60_60387

-- Define the given values
def shoe_cost : ℝ := 45.50
def ten_dollars : ℝ := 10.0
def num_ten_dollar_bills : ℕ := 4
def quarter_value : ℝ := 0.25
def num_quarters : ℕ := 5
def nickel_value : ℝ := 0.05

-- Define the statement to be proved
theorem chloe_at_least_85_nickels (n : ℕ) 
  (H1 : shoe_cost = 45.50)
  (H2 : ten_dollars = 10.0)
  (H3 : num_ten_dollar_bills = 4)
  (H4 : quarter_value = 0.25)
  (H5 : num_quarters = 5)
  (H6 : nickel_value = 0.05) :
  4 * ten_dollars + 5 * quarter_value + n * nickel_value >= shoe_cost → n >= 85 :=
by {
  sorry
}

end chloe_at_least_85_nickels_l60_60387


namespace artifacts_per_wing_l60_60227

theorem artifacts_per_wing
  (total_wings : ℕ)
  (num_paintings : ℕ)
  (num_artifacts : ℕ)
  (painting_wings : ℕ)
  (large_paintings_wings : ℕ)
  (small_paintings_wings : ℕ)
  (small_paintings_per_wing : ℕ)
  (artifact_wings : ℕ)
  (wings_division : total_wings = painting_wings + artifact_wings)
  (paintings_division : painting_wings = large_paintings_wings + small_paintings_wings)
  (num_large_paintings : large_paintings_wings = 2)
  (num_small_paintings : small_paintings_wings * small_paintings_per_wing = num_paintings - large_paintings_wings)
  (num_artifact_calc : num_artifacts = 8 * num_paintings)
  (artifact_wings_div : artifact_wings = total_wings - painting_wings)
  (artifact_calc : num_artifacts / artifact_wings = 66) :
  num_artifacts / artifact_wings = 66 := 
by
  sorry

end artifacts_per_wing_l60_60227


namespace factor_expression_l60_60694

theorem factor_expression (x : ℝ) :
  (12 * x ^ 5 + 33 * x ^ 3 + 10) - (3 * x ^ 5 - 4 * x ^ 3 - 1) = x ^ 3 * (9 * x ^ 2 + 37) + 11 :=
by {
  -- Provide the skeleton for the proof using simplification
  sorry
}

end factor_expression_l60_60694


namespace louisa_average_speed_l60_60444

theorem louisa_average_speed :
  ∃ v : ℝ, 
  (100 / v = 175 / v - 3) ∧ 
  v = 25 :=
by
  sorry

end louisa_average_speed_l60_60444


namespace vasim_share_l60_60375

theorem vasim_share (x : ℝ)
  (h_ratio : ∀ (f v r : ℝ), f = 3 * x ∧ v = 5 * x ∧ r = 6 * x)
  (h_diff : 6 * x - 3 * x = 900) :
  5 * x = 1500 :=
by
  try sorry

end vasim_share_l60_60375


namespace inverse_proposition_of_divisibility_by_5_l60_60458

theorem inverse_proposition_of_divisibility_by_5 (n : ℕ) :
  (n % 10 = 5 → n % 5 = 0) → (n % 5 = 0 → n % 10 = 5) :=
sorry

end inverse_proposition_of_divisibility_by_5_l60_60458


namespace a_can_finish_remaining_work_in_5_days_l60_60080

theorem a_can_finish_remaining_work_in_5_days (a_work_rate b_work_rate : ℝ) (total_days_b_works : ℝ):
  a_work_rate = 1/15 → 
  b_work_rate = 1/15 → 
  total_days_b_works = 10 → 
  ∃ (remaining_days_for_a : ℝ), remaining_days_for_a = 5 :=
by
  intros h1 h2 h3
  -- We are skipping the proof itself
  sorry

end a_can_finish_remaining_work_in_5_days_l60_60080


namespace isosceles_right_triangle_example_l60_60076

theorem isosceles_right_triangle_example :
  (5 = 5) ∧ (5^2 + 5^2 = (5 * Real.sqrt 2)^2) :=
by {
  sorry
}

end isosceles_right_triangle_example_l60_60076


namespace tournament_participants_l60_60492

theorem tournament_participants (x : ℕ) (h1 : ∀ g b : ℕ, g = 2 * b)
  (h2 : ∀ p : ℕ, p = 3 * x) 
  (h3 : ∀ G B : ℕ, G + B = (3 * x * (3 * x - 1)) / 2)
  (h4 : ∀ G B : ℕ, G / B = 7 / 9) 
  (h5 : x = 11) :
  3 * x = 33 :=
by
  sorry

end tournament_participants_l60_60492


namespace smallest_positive_integer_n_l60_60261

noncomputable def matrix_330 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![
    ![Real.cos (330 * Real.pi / 180), -Real.sin (330 * Real.pi / 180)],
    ![Real.sin (330 * Real.pi / 180), Real.cos (330 * Real.pi / 180)]
  ]

theorem smallest_positive_integer_n (n : ℕ) (h : matrix_330 ^ n = 1) : n = 12 := sorry

end smallest_positive_integer_n_l60_60261


namespace isosceles_triangle_perimeter_l60_60516

variable (a b : ℕ) 

theorem isosceles_triangle_perimeter (h1 : a = 3) (h2 : b = 6) : 
  ∃ P, (a = 3 ∧ b = 6 ∧ P = 15 ∨ b = 3 ∧ a = 6 ∧ P = 15) := by
  use 15
  sorry

end isosceles_triangle_perimeter_l60_60516


namespace somu_age_relation_l60_60328

-- Somu’s present age (S) is 20 years
def somu_present_age : ℕ := 20

-- Somu’s age is one-third of his father’s age (F)
def father_present_age : ℕ := 3 * somu_present_age

-- Proof statement: Y years ago, Somu's age was one-fifth of his father's age
theorem somu_age_relation : ∃ (Y : ℕ), somu_present_age - Y = (1 : ℕ) / 5 * (father_present_age - Y) ∧ Y = 10 :=
by
  have h := "" -- Placeholder for the proof steps
  sorry

end somu_age_relation_l60_60328


namespace factor_difference_of_squares_l60_60258

theorem factor_difference_of_squares (t : ℝ) : t^2 - 144 = (t - 12) * (t + 12) :=
by
  sorry

end factor_difference_of_squares_l60_60258


namespace harry_travel_time_l60_60288

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l60_60288


namespace average_pastries_sold_per_day_l60_60361

noncomputable def pastries_sold_per_day (n : ℕ) : ℕ :=
  match n with
  | 0 => 2 -- Monday
  | _ + 1 => 2 + n + 1

theorem average_pastries_sold_per_day :
  (∑ i in Finset.range 7, pastries_sold_per_day i) / 7 = 5 :=
by
  sorry

end average_pastries_sold_per_day_l60_60361


namespace min_value_of_a_l60_60872

theorem min_value_of_a (a : ℝ) : 
  (∀ x > 1, x + a / (x - 1) ≥ 5) → a ≥ 4 :=
sorry

end min_value_of_a_l60_60872


namespace cellphone_surveys_l60_60371

theorem cellphone_surveys
  (regular_rate : ℕ)
  (total_surveys : ℕ)
  (higher_rate_multiplier : ℕ)
  (total_earnings : ℕ)
  (higher_rate_bonus : ℕ)
  (x : ℕ) :
  regular_rate = 10 → total_surveys = 100 →
  higher_rate_multiplier = 130 → total_earnings = 1180 →
  higher_rate_bonus = 3 → (10 * (100 - x) + 13 * x = 1180) →
  x = 60 :=
by
  sorry

end cellphone_surveys_l60_60371


namespace not_every_tv_owner_has_pass_l60_60377

variable (Person : Type) (T P G : Person → Prop)

-- Condition 1: There exists a television owner who is not a painter.
axiom exists_tv_owner_not_painter : ∃ x, T x ∧ ¬ P x 

-- Condition 2: If someone has a pass to the Gellért Baths and is not a painter, they are not a television owner.
axiom pass_and_not_painter_imp_not_tv_owner : ∀ x, (G x ∧ ¬ P x) → ¬ T x

-- Prove: Not every television owner has a pass to the Gellért Baths.
theorem not_every_tv_owner_has_pass :
  ¬ ∀ x, T x → G x :=
by
  sorry -- Proof omitted

end not_every_tv_owner_has_pass_l60_60377


namespace travel_times_either_24_or_72_l60_60119

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l60_60119


namespace count_two_digit_prime_numbers_ending_in_3_l60_60915

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l60_60915


namespace largest_n_exists_ints_l60_60844

theorem largest_n_exists_ints (n : ℤ) :
  (∃ x y z : ℤ, n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 4 * x + 4 * y + 4 * z - 12) →
  n ≤ 10 :=
sorry

end largest_n_exists_ints_l60_60844


namespace inequality_of_function_inequality_l60_60727

noncomputable def f (x : ℝ) : ℝ := (Real.log (x + Real.sqrt (x^2 + 1))) + 2 * x + Real.sin x

theorem inequality_of_function_inequality (x1 x2 : ℝ) (h : f x1 + f x2 > 0) : x1 + x2 > 0 :=
sorry

end inequality_of_function_inequality_l60_60727


namespace loaned_books_count_l60_60673

variable (x : ℕ) -- x is the number of books loaned out during the month

theorem loaned_books_count 
  (initial_books : ℕ) (returned_percentage : ℚ) (remaining_books : ℕ)
  (h1 : initial_books = 75)
  (h2 : returned_percentage = 0.80)
  (h3 : remaining_books = 66) :
  x = 45 :=
by
  -- Proof can be inserted here
  sorry

end loaned_books_count_l60_60673


namespace total_cost_eq_4800_l60_60441

def length := 30
def width := 40
def cost_per_square_foot := 3
def cost_of_sealant_per_square_foot := 1

theorem total_cost_eq_4800 : 
  (length * width * cost_per_square_foot) + (length * width * cost_of_sealant_per_square_foot) = 4800 :=
by
  sorry

end total_cost_eq_4800_l60_60441


namespace Shekar_weighted_average_l60_60636

def score_weighted_sum (scores_weights : List (ℕ × ℚ)) : ℚ :=
  scores_weights.foldl (fun acc sw => acc + (sw.1 * sw.2 : ℚ)) 0

def Shekar_scores_weights : List (ℕ × ℚ) :=
  [(76, 0.20), (65, 0.15), (82, 0.10), (67, 0.15), (55, 0.10), (89, 0.05), (74, 0.05),
   (63, 0.10), (78, 0.05), (71, 0.05)]

theorem Shekar_weighted_average : score_weighted_sum Shekar_scores_weights = 70.55 := by
  sorry

end Shekar_weighted_average_l60_60636


namespace expected_pourings_correct_l60_60773

section
  /-- Four glasses are arranged in a row: the first and third contain orange juice, 
      the second and fourth are empty. Valya can take a full glass and pour its 
      contents into one of the two empty glasses each time. -/
  def initial_state : List Bool := [true, false, true, false]
  def target_state : List Bool := [false, true, false, true]

  /-- Define a function to calculate the expected number of pourings required to 
      reach the target state from the initial state given the probabilities of 
      transitions. -/
  noncomputable def expected_number_of_pourings (init : List Bool) (target : List Bool) : ℕ :=
    if init = initial_state ∧ target = target_state then 6 else 0

  /-- Prove that the expected number of pourings required to transition from 
      the initial state [true, false, true, false] to the target state [false, true, false, true] is 6. -/
  theorem expected_pourings_correct :
    expected_number_of_pourings initial_state target_state = 6 :=
  by
    -- Proof omitted
    sorry
end

end expected_pourings_correct_l60_60773


namespace two_digit_primes_with_units_digit_three_count_l60_60928

theorem two_digit_primes_with_units_digit_three_count : 
  (finset.card (finset.filter (λ n, nat.prime n) (finset.filter (λ n, n % 10 = 3) (finset.range 100).filter (λ n, n > 9)))) = 6 :=
sorry

end two_digit_primes_with_units_digit_three_count_l60_60928


namespace find_x_plus_y_l60_60763

variables {x y : ℝ}

def f (t : ℝ) : ℝ := t^2003 + 2002 * t

theorem find_x_plus_y (hx : f (x - 1) = -1) (hy : f (y - 2) = 1) : x + y = 3 :=
by
  sorry

end find_x_plus_y_l60_60763


namespace find_function_expression_l60_60278

theorem find_function_expression (f : ℝ → ℝ) (h : ∀ x : ℝ, f (x^2 - 1) = x^4 + 1) :
  ∀ x : ℝ, x ≥ -1 → f x = x^2 + 2*x + 2 :=
sorry

end find_function_expression_l60_60278


namespace voting_proposal_l60_60749

theorem voting_proposal :
  ∀ (T Votes_against Votes_in_favor More_votes_in_favor : ℕ),
    T = 290 →
    Votes_against = (40 * T) / 100 →
    Votes_in_favor = T - Votes_against →
    More_votes_in_favor = Votes_in_favor - Votes_against →
    More_votes_in_favor = 58 :=
by sorry

end voting_proposal_l60_60749


namespace arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l60_60358

-- Problem (a)
theorem arithmetic_sequence_a (x1 x2 x3 x4 x5 : ℕ) (h : (x1 = 2 ∧ x2 = 5 ∧ x3 = 10 ∧ x4 = 13 ∧ x5 = 15)) : 
  ∃ a b c, (a = 5 ∧ b = 10 ∧ c = 15 ∧ b - a = c - b ∧ b - a > 0) := 
sorry

-- Problem (b)
theorem find_p_q (p q : ℕ) (h : ∃ d, (7 - p = d ∧ q - 7 = d ∧ 13 - q = d)) : 
  p = 4 ∧ q = 10 :=
sorry

-- Problem (c)
theorem find_c_minus_a (a b c : ℕ) (h : ∃ d, (b - a = d ∧ c - b = d ∧ (a + 21) - c = d)) :
  c - a = 14 :=
sorry

-- Problem (d)
theorem find_y_values (y : ℤ) (h : ∃ d, ((2*y + 3) - (y - 6) = d ∧ (y*y + 2) - (2*y + 3) = d) ) :
  y = 5 ∨ y = -2 :=
sorry

end arithmetic_sequence_a_find_p_q_find_c_minus_a_find_y_values_l60_60358


namespace inequality_system_solution_l60_60986

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60986


namespace at_least_one_female_team_l60_60128

open Classical

namespace Probability

-- Define the Problem
noncomputable def prob_at_least_one_female (females males : ℕ) (team_size : ℕ) :=
  let total_students := females + males
  let total_ways := Nat.choose total_students team_size
  let ways_all_males := Nat.choose males team_size
  1 - (ways_all_males / total_ways : ℝ)

-- Verify the given problem against the expected answer
theorem at_least_one_female_team :
  prob_at_least_one_female 1 3 2 = 1 / 2 := by
  sorry

end Probability

end at_least_one_female_team_l60_60128


namespace inequality_ab2_bc2_ca2_leq_27_div_8_l60_60614

theorem inequality_ab2_bc2_ca2_leq_27_div_8 (a b c : ℝ) (h : a ≥ b) (h1 : b ≥ c) (h2 : c ≥ 0) (h3 : a + b + c = 3) :
  ab^2 + bc^2 + ca^2 ≤ 27 / 8 :=
sorry

end inequality_ab2_bc2_ca2_leq_27_div_8_l60_60614


namespace solve_inequality_l60_60531

theorem solve_inequality (x : ℝ) : 2 * x^2 + 8 * x ≤ -6 ↔ -3 ≤ x ∧ x ≤ -1 :=
by
  sorry

end solve_inequality_l60_60531


namespace min_total_weight_l60_60081

theorem min_total_weight (crates: Nat) (weight_per_crate: Nat) (h1: crates = 6) (h2: weight_per_crate ≥ 120): 
  crates * weight_per_crate ≥ 720 :=
by
  sorry

end min_total_weight_l60_60081


namespace regular_polygon_sides_l60_60505

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l60_60505


namespace harry_total_travel_time_l60_60289

def bus_time_already_sitting : Nat := 15
def bus_time_remaining : Nat := 25
def walk_fraction := 1 / 2

def bus_time_total : Nat := bus_time_already_sitting + bus_time_remaining
def walk_time : Nat := bus_time_total * walk_fraction

theorem harry_total_travel_time : bus_time_total + walk_time = 60 := by
  sorry

end harry_total_travel_time_l60_60289


namespace find_x_y_l60_60651

theorem find_x_y (x y : ℝ) (h1 : (10 + 25 + x + y) / 4 = 20) (h2 : x * y = 156) :
  (x = 12 ∧ y = 33) ∨ (x = 33 ∧ y = 12) :=
by
  sorry

end find_x_y_l60_60651


namespace find_multiple_of_hats_l60_60379

/-
   Given:
   - Fire chief Simpson has 15 hats.
   - Policeman O'Brien now has 34 hats.
   - Before he lost one, Policeman O'Brien had 5 more hats than a certain multiple of Fire chief Simpson's hats.
   Prove:
   The multiple of Fire chief Simpson's hats that Policeman O'Brien had before he lost one is 2.
-/

theorem find_multiple_of_hats :
  ∃ x : ℕ, 34 + 1 = 5 + 15 * x ∧ x = 2 :=
by
  sorry

end find_multiple_of_hats_l60_60379


namespace num_two_digit_prime_numbers_with_units_digit_3_l60_60911

def two_digit_prime_numbers_with_units_digit_3 : Finset ℕ :=
  {n | n > 9 ∧ n < 100 ∧ n % 10 = 3 ∧ Nat.Prime n}.to_finset

theorem num_two_digit_prime_numbers_with_units_digit_3 :
  two_digit_prime_numbers_with_units_digit_3.card = 6 :=
by
  sorry

end num_two_digit_prime_numbers_with_units_digit_3_l60_60911


namespace original_wage_l60_60234

theorem original_wage (W : ℝ) (h : 1.5 * W = 42) : W = 28 :=
by
  sorry

end original_wage_l60_60234


namespace original_price_l60_60821

theorem original_price (P S : ℝ) (h1 : S = 1.25 * P) (h2 : S - P = 625) : P = 2500 := by
  sorry

end original_price_l60_60821


namespace sum_of_coords_D_eq_eight_l60_60633

def point := (ℝ × ℝ)

def N : point := (4, 6)
def C : point := (10, 2)

def is_midpoint (M A B : point) : Prop :=
  M.1 = (A.1 + B.1) / 2 ∧ M.2 = (A.2 + B.2) / 2

theorem sum_of_coords_D_eq_eight
  (D : point)
  (h_midpoint : is_midpoint N C D) :
  D.1 + D.2 = 8 :=
by 
  sorry

end sum_of_coords_D_eq_eight_l60_60633


namespace integer_solutions_count_l60_60578

theorem integer_solutions_count (B : ℤ) (C : ℤ) (h : B = 3) : C = 4 :=
by
  sorry

end integer_solutions_count_l60_60578


namespace completing_the_square_l60_60045

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l60_60045


namespace cyclic_trapezoid_radii_relation_l60_60788

variables (A B C D O : Type)
variables (AD BC : Type)
variables (r1 r2 r3 r4 : ℝ)

-- Conditions
def cyclic_trapezoid (A B C D: Type) (AD BC: Type): Prop := sorry
def intersection (A B C D O : Type): Prop := sorry
def radius_incircle (triangle : Type) (radius : ℝ): Prop := sorry

theorem cyclic_trapezoid_radii_relation
  (h1: cyclic_trapezoid A B C D AD BC)
  (h2: intersection A B C D O)
  (hr1: radius_incircle AOD r1)
  (hr2: radius_incircle AOB r2)
  (hr3: radius_incircle BOC r3)
  (hr4: radius_incircle COD r4):
  (1 / r1) + (1 / r3) = (1 / r2) + (1 / r4) :=
sorry

end cyclic_trapezoid_radii_relation_l60_60788


namespace fish_caught_l60_60380

noncomputable def total_fish_caught (chris_trips : ℕ) (chris_fish_per_trip : ℕ) (brian_trips : ℕ) (brian_fish_per_trip : ℕ) : ℕ :=
  chris_trips * chris_fish_per_trip + brian_trips * brian_fish_per_trip

theorem fish_caught (chris_trips : ℕ) (brian_factor : ℕ) (brian_fish_per_trip : ℕ) (ratio_numerator : ℕ) (ratio_denominator : ℕ) :
  chris_trips = 10 → brian_factor = 2 → brian_fish_per_trip = 400 → ratio_numerator = 3 → ratio_denominator = 5 →
  total_fish_caught chris_trips (brian_fish_per_trip * ratio_denominator / ratio_numerator) (chris_trips * brian_factor) brian_fish_per_trip = 14660 :=
by
  intros h_chris_trips h_brian_factor h_brian_fish_per_trip h_ratio_numer h_ratio_denom
  rw [h_chris_trips, h_brian_factor, h_brian_fish_per_trip, h_ratio_numer, h_ratio_denom]
  -- adding actual arithmetic would resolve the statement correctly
  sorry

end fish_caught_l60_60380


namespace find_n_l60_60271

theorem find_n (a b c : ℕ) (n : ℕ) (h₁ : a^2 + b^2 = c^2) (h₂ : n > 2) 
    (h₃ : (a^n + b^n + c^n)^2 = 2 * (a^(2*n) + b^(2*n) + c^(2*n))) : n = 4 := 
sorry

end find_n_l60_60271


namespace total_wicks_l60_60099

-- Amy bought a 15-foot spool of string.
def spool_length_feet : ℕ := 15

-- Since there are 12 inches in a foot, convert the spool length to inches.
def spool_length_inches : ℕ := spool_length_feet * 12

-- The string is cut into an equal number of 6-inch and 12-inch wicks.
def wick_pair_length : ℕ := 6 + 12

-- Prove that the total number of wicks she cuts is 20.
theorem total_wicks : (spool_length_inches / wick_pair_length) * 2 = 20 := by
  sorry

end total_wicks_l60_60099


namespace lewis_speed_is_90_l60_60842

noncomputable def david_speed : ℝ := 50 -- mph
noncomputable def distance_chennai_hyderabad : ℝ := 350 -- miles
noncomputable def distance_meeting_point : ℝ := 250 -- miles

theorem lewis_speed_is_90 :
  ∃ L : ℝ, 
    (∀ t : ℝ, david_speed * t = distance_meeting_point) →
    (∀ t : ℝ, L * t = (distance_chennai_hyderabad + (distance_meeting_point - distance_chennai_hyderabad))) →
    L = 90 :=
by
  sorry

end lewis_speed_is_90_l60_60842


namespace no_primes_between_factorial_plus_3_and_factorial_plus_2n_l60_60125

theorem no_primes_between_factorial_plus_3_and_factorial_plus_2n (n : ℕ) (h : n > 2) :
  ∀ k, n! + 3 ≤ k → k ≤ n! + 2n → ¬ prime k :=
by
  sorry

end no_primes_between_factorial_plus_3_and_factorial_plus_2n_l60_60125


namespace necessary_condition_ac_eq_bc_l60_60126

theorem necessary_condition_ac_eq_bc {a b c : ℝ} (hc : c ≠ 0) : (ac = bc ↔ a = b) := by
  sorry

end necessary_condition_ac_eq_bc_l60_60126


namespace proof_complement_U_A_l60_60884

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4, 5}

-- Define the set A
def A : Set ℕ := {2, 3, 4}

-- Define the complement of A with respect to U
def complement_U_A : Set ℕ := { x ∈ U | x ∉ A }

-- The theorem statement
theorem proof_complement_U_A :
  complement_U_A = {1, 5} :=
by
  -- Proof goes here
  sorry

end proof_complement_U_A_l60_60884


namespace product_of_even_and_odd_is_odd_l60_60767

def even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x
def odd_product (f g : ℝ → ℝ) : Prop := ∀ x, (f x) * (g x) = - (f x) * (g x)
 
theorem product_of_even_and_odd_is_odd 
  (f : ℝ → ℝ) (g : ℝ → ℝ) 
  (h1 : even_function f) 
  (h2 : odd_function g) : odd_product f g :=
by
  sorry

end product_of_even_and_odd_is_odd_l60_60767


namespace product_of_five_consecutive_not_square_l60_60981

theorem product_of_five_consecutive_not_square (n : ℤ) :
  ¬ ∃ k : ℤ, (n * (n + 1) * (n + 2) * (n + 3) * (n + 4) = k^2) :=
by
  sorry

end product_of_five_consecutive_not_square_l60_60981


namespace minimum_arc_length_of_curve_and_line_l60_60421

-- Definition of the curve C and the line x = π/4
def curve (x y α : ℝ) : Prop :=
  (x - Real.arcsin α) * (x - Real.arccos α) + (y - Real.arcsin α) * (y + Real.arccos α) = 0

def line (x : ℝ) : Prop :=
  x = Real.pi / 4

-- Statement of the proof problem: the minimum value of d as α varies
theorem minimum_arc_length_of_curve_and_line : 
  (∀ α : ℝ, ∃ d : ℝ, (∃ y : ℝ, curve (Real.pi / 4) y α) → 
    (d = Real.pi / 2)) :=
sorry

end minimum_arc_length_of_curve_and_line_l60_60421


namespace number_of_positive_integer_solutions_l60_60798

theorem number_of_positive_integer_solutions (n k : ℕ) (h : n ≥ k) :
    ∃ solutions,
    (∀ x : ℕ → ℕ, (∀ i, 1 ≤ x i) → (sum x {0..k-1} = n) ↔ ∃ f : ℕ → ℕ, sum f {0..k-1} = n - k)
    → solutions = (Nat.choose (n - 1) (k - 1)) :=
by
  sorry

end number_of_positive_integer_solutions_l60_60798


namespace one_fourth_of_8_point_8_l60_60855

-- Definition of taking one fourth of a number
def oneFourth (x : ℝ) : ℝ := x / 4

-- Problem statement: One fourth of 8.8 is 11/5 when expressed as a simplified fraction
theorem one_fourth_of_8_point_8 : oneFourth 8.8 = 11 / 5 := by
  sorry

end one_fourth_of_8_point_8_l60_60855


namespace correct_average_weight_is_58_6_l60_60329

noncomputable def initial_avg_weight : ℚ := 58.4
noncomputable def num_boys : ℕ := 20
noncomputable def incorrect_weight : ℚ := 56
noncomputable def correct_weight : ℚ := 60
noncomputable def correct_avg_weight := (initial_avg_weight * num_boys + (correct_weight - incorrect_weight)) / num_boys

theorem correct_average_weight_is_58_6 :
  correct_avg_weight = 58.6 :=
sorry

end correct_average_weight_is_58_6_l60_60329


namespace complete_the_square_l60_60053

theorem complete_the_square (x : ℝ) :
  (x^2 - 6 * x + 8 = 0) → ((x - 3)^2 = 1) := 
by
  sorry

end complete_the_square_l60_60053


namespace count_two_digit_prime_numbers_with_units_digit_3_l60_60900

-- Definition of two-digit numbers with a units digit of 3
def two_digit_numbers_with_units_digit_3 : List ℕ := [13, 23, 33, 43, 53, 63, 73, 83, 93]

-- Definition of checking for primality
def is_prime (n : ℕ) : Prop := ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

-- Definition of two-digit prime numbers with a units digit of 3
def two_digit_prime_numbers_with_units_digit_3 : List ℕ := 
  two_digit_numbers_with_units_digit_3.filter is_prime

-- Statement of the theorem
theorem count_two_digit_prime_numbers_with_units_digit_3 : two_digit_prime_numbers_with_units_digit_3.length = 6 :=
by
  sorry

end count_two_digit_prime_numbers_with_units_digit_3_l60_60900


namespace min_value_of_x_under_conditions_l60_60743

noncomputable def S (x y z : ℝ) : ℝ := (z + 1)^2 / (2 * x * y * z)

theorem min_value_of_x_under_conditions :
  ∀ (x y z : ℝ), x > 0 → y > 0 → z > 0 → x^2 + y^2 + z^2 = 1 →
  (∃ x_min : ℝ, S x y z = S x_min x_min (Real.sqrt 2 - 1) ∧ x_min = Real.sqrt (Real.sqrt 2 - 1)) :=
by
  intros x y z hx hy hz hxyz
  use Real.sqrt (Real.sqrt 2 - 1)
  sorry

end min_value_of_x_under_conditions_l60_60743


namespace sequence_a10_l60_60412

theorem sequence_a10 : 
  (∃ (a : ℕ → ℤ), 
    a 1 = -1 ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n) - a (2*n - 1) = 2^(2*n-1)) ∧ 
    (∀ n : ℕ, n ≥ 1 → a (2*n + 1) - a (2*n) = 2^(2*n))) → 
  (∃ a : ℕ → ℤ, a 10 = 1021) :=
by
  intro h
  obtain ⟨a, h1, h2, h3⟩ := h
  sorry

end sequence_a10_l60_60412


namespace num_two_digit_prime_with_units_digit_3_eq_6_l60_60892

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def units_digit_is_3 (n : ℕ) : Prop := n % 10 = 3

def two_digit_number (n : ℕ) : Prop := n ≥ 10 ∧ n < 100

theorem num_two_digit_prime_with_units_digit_3_eq_6 :
  ∃ S : Finset ℕ, (∀ n ∈ S, is_prime n ∧ units_digit_is_3 n ∧ two_digit_number n) ∧ S.card = 6 :=
by
  sorry

end num_two_digit_prime_with_units_digit_3_eq_6_l60_60892


namespace consecutive_sum_to_20_has_one_set_l60_60588

theorem consecutive_sum_to_20_has_one_set :
  ∃ n a : ℕ, (n ≥ 2) ∧ (a ≥ 1) ∧ (n * (2 * a + n - 1) = 40) ∧
  (n = 5 ∧ a = 2) ∧ 
  (∀ n' a', (n' ≥ 2) → (a' ≥ 1) → (n' * (2 * a' + n' - 1) = 40) → (n' = 5 ∧ a' = 2)) := sorry

end consecutive_sum_to_20_has_one_set_l60_60588


namespace sally_popped_3_balloons_l60_60965

-- Defining the conditions
def joans_initial_balloons : ℕ := 9
def jessicas_balloons : ℕ := 2
def total_balloons_now : ℕ := 6

-- Definition for the number of balloons Sally popped
def sally_balloons_popped : ℕ := joans_initial_balloons - (total_balloons_now - jessicas_balloons)

-- The theorem statement
theorem sally_popped_3_balloons : sally_balloons_popped = 3 := 
by
  -- Proof omitted; use sorry
  sorry

end sally_popped_3_balloons_l60_60965


namespace math_problem_l60_60265

noncomputable def A (k : ℝ) : ℝ := k - 5
noncomputable def B (k : ℝ) : ℝ := k + 2
noncomputable def C (k : ℝ) : ℝ := k / 2
noncomputable def D (k : ℝ) : ℝ := 2 * k

theorem math_problem (k : ℝ) (h : A k + B k + C k + D k = 100) : 
  (A k) * (B k) * (C k) * (D k) =  (161 * 224 * 103 * 412) / 6561 :=
by
  sorry

end math_problem_l60_60265


namespace problem1_problem2_l60_60214

-- Problem1
theorem problem1 (a : ℤ) (h : a = -2) :
    ( (a^2 + a) / (a^2 - 3 * a) / (a^2 - 1) / (a - 3) - 1 / (a + 1) = 2 / 3) :=
by 
  sorry

-- Problem2
theorem problem2 (x : ℤ) :
    ( (x^2 - 1) / (x - 4) / (x + 1) / (4 - x) = 1 - x) :=
by 
  sorry

end problem1_problem2_l60_60214


namespace general_term_formula_of_arithmetic_seq_l60_60461

noncomputable def arithmetic_seq (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem general_term_formula_of_arithmetic_seq 
  (a : ℕ → ℝ) (h_arith : arithmetic_seq a)
  (h1 : a 3 * a 7 = -16) 
  (h2 : a 4 + a 6 = 0) :
  (∀ n : ℕ, a n = 2 * n - 10) ∨ (∀ n : ℕ, a n = -2 * n + 10) :=
by
  sorry

end general_term_formula_of_arithmetic_seq_l60_60461


namespace factor_t_squared_minus_144_l60_60255

theorem factor_t_squared_minus_144 (t : ℝ) : 
  t ^ 2 - 144 = (t - 12) * (t + 12) := 
by 
  -- Here you would include the proof steps which are not needed for this task.
  sorry

end factor_t_squared_minus_144_l60_60255


namespace smallest_n_factorial_l60_60452

theorem smallest_n_factorial (a b c m n : ℕ) (h1 : a + b + c = 2020)
(h2 : c > a + 100)
(h3 : m * 10^n = a! * b! * c!)
(h4 : ¬ (10 ∣ m)) : 
  n = 499 :=
sorry

end smallest_n_factorial_l60_60452


namespace average_page_count_l60_60599

theorem average_page_count 
  (n1 n2 n3 n4 : ℕ)
  (p1 p2 p3 p4 total_students : ℕ)
  (h1 : n1 = 8)
  (h2 : p1 = 3)
  (h3 : n2 = 10)
  (h4 : p2 = 5)
  (h5 : n3 = 7)
  (h6 : p3 = 2)
  (h7 : n4 = 5)
  (h8 : p4 = 4)
  (h9 : total_students = 30) :
  (n1 * p1 + n2 * p2 + n3 * p3 + n4 * p4) / total_students = 36 / 10 := 
sorry

end average_page_count_l60_60599


namespace earnings_correct_l60_60017

def price_8inch : ℝ := 5
def price_12inch : ℝ := 2.5 * price_8inch
def price_16inch : ℝ := 3 * price_8inch
def price_20inch : ℝ := 4 * price_8inch
def price_24inch : ℝ := 5.5 * price_8inch

noncomputable def earnings_monday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 1 * price_16inch + 2 * price_20inch + 1 * price_24inch

noncomputable def earnings_tuesday : ℝ :=
  5 * price_8inch + 1 * price_12inch + 4 * price_16inch + 2 * price_24inch

noncomputable def earnings_wednesday : ℝ :=
  4 * price_8inch + 3 * price_12inch + 3 * price_16inch + 1 * price_20inch

noncomputable def earnings_thursday : ℝ :=
  2 * price_8inch + 2 * price_12inch + 2 * price_16inch + 1 * price_20inch + 3 * price_24inch

noncomputable def earnings_friday : ℝ :=
  6 * price_8inch + 4 * price_12inch + 2 * price_16inch + 2 * price_20inch

noncomputable def earnings_saturday : ℝ :=
  1 * price_8inch + 3 * price_12inch + 3 * price_16inch + 4 * price_20inch + 2 * price_24inch

noncomputable def earnings_sunday : ℝ :=
  3 * price_8inch + 2 * price_12inch + 4 * price_16inch + 3 * price_20inch + 1 * price_24inch

noncomputable def total_earnings : ℝ :=
  earnings_monday + earnings_tuesday + earnings_wednesday + earnings_thursday + earnings_friday + earnings_saturday + earnings_sunday

theorem earnings_correct : total_earnings = 1025 := by
  -- proof goes here
  sorry

end earnings_correct_l60_60017


namespace laptop_price_l60_60180

theorem laptop_price (upfront_percent : ℝ) (upfront_payment full_price : ℝ)
  (h1 : upfront_percent = 0.20)
  (h2 : upfront_payment = 240)
  (h3 : upfront_payment = upfront_percent * full_price) :
  full_price = 1200 := 
sorry

end laptop_price_l60_60180


namespace Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l60_60127

def Q (x : ℂ) (n : ℕ) : ℂ := (x + 1)^n + x^n + 1
def P (x : ℂ) : ℂ := x^2 + x + 1

-- Part a) Q(x) is divisible by P(x) if and only if n ≡ 2 (mod 6) or n ≡ 4 (mod 6)
theorem Q_divisible_by_P (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x) = 0 ↔ (n % 6 = 2 ∨ n % 6 = 4) := sorry

-- Part b) Q(x) is divisible by P(x)^2 if and only if n ≡ 4 (mod 6)
theorem Q_divisible_by_P_squared (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^2 = 0 ↔ n % 6 = 4 := sorry

-- Part c) Q(x) is never divisible by P(x)^3
theorem Q_not_divisible_by_P_cubed (x : ℂ) (n : ℕ) : 
  (Q x n) % (P x)^3 ≠ 0 := sorry

end Q_divisible_by_P_Q_divisible_by_P_squared_Q_not_divisible_by_P_cubed_l60_60127


namespace percentage_is_40_l60_60154

variables (num : ℕ) (perc : ℕ)

-- Conditions
def ten_percent_eq_40 : Prop := 10 * num = 400
def certain_percentage_eq_160 : Prop := perc * num = 160 * 100

-- Statement to prove
theorem percentage_is_40 (h1 : ten_percent_eq_40 num) (h2 : certain_percentage_eq_160 num perc) : perc = 40 :=
sorry

end percentage_is_40_l60_60154


namespace tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l60_60837

theorem tan_45_add_reciprocal_half_add_abs_neg_two_eq_five :
  (Real.tan (Real.pi / 4) + (1 / 2)⁻¹ + |(-2 : ℝ)|) = 5 :=
by
  -- Assuming the conditions provided in part a)
  have h1 : Real.tan (Real.pi / 4) = 1 := by sorry
  have h2 : (1 / 2 : ℝ)⁻¹ = 2 := by sorry
  have h3 : |(-2 : ℝ)| = 2 := by sorry

  -- Proof of the problem using the conditions
  rw [h1, h2, h3]
  norm_num

end tan_45_add_reciprocal_half_add_abs_neg_two_eq_five_l60_60837


namespace dot_product_two_a_plus_b_with_a_l60_60143

-- Define vector a
def a : ℝ × ℝ := (2, -1)

-- Define vector b
def b : ℝ × ℝ := (-1, 2)

-- Define the scalar multiplication of vector a by 2
def two_a : ℝ × ℝ := (2 * a.1, 2 * a.2)

-- Define the vector addition of 2a and b
def two_a_plus_b : ℝ × ℝ := (two_a.1 + b.1, two_a.2 + b.2)

-- Define dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

-- Prove that the dot product of (2 * a + b) and a equals 6
theorem dot_product_two_a_plus_b_with_a :
  dot_product two_a_plus_b a = 6 :=
by
  sorry

end dot_product_two_a_plus_b_with_a_l60_60143


namespace quadrilateral_ABCD_r_plus_s_l60_60162

noncomputable def AB_is (AB : Real) (r s : Nat) : Prop :=
  AB = r + Real.sqrt s

theorem quadrilateral_ABCD_r_plus_s :
  ∀ (BC CD AD : Real) (mA mB : ℕ) (r s : ℕ), 
  BC = 7 → 
  CD = 10 → 
  AD = 8 → 
  mA = 60 → 
  mB = 60 → 
  AB_is AB r s →
  r + s = 99 :=
by intros BC CD AD mA mB r s hBC hCD hAD hMA hMB hAB_is
   sorry

end quadrilateral_ABCD_r_plus_s_l60_60162


namespace least_number_of_square_tiles_l60_60210

-- Definitions based on conditions
def room_length_cm : ℕ := 672
def room_width_cm : ℕ := 432

-- Correct Answer is 126 tiles

-- Lean Statement for the proof problem
theorem least_number_of_square_tiles : 
  ∃ tile_size tiles_needed, 
    (tile_size = Int.gcd room_length_cm room_width_cm) ∧
    (tiles_needed = (room_length_cm / tile_size) * (room_width_cm / tile_size)) ∧
    tiles_needed = 126 := 
by
  sorry

end least_number_of_square_tiles_l60_60210


namespace number_of_soccer_campers_l60_60514

-- Conditions as definitions in Lean
def total_campers : ℕ := 88
def basketball_campers : ℕ := 24
def football_campers : ℕ := 32
def soccer_campers : ℕ := total_campers - (basketball_campers + football_campers)

-- Theorem statement to prove
theorem number_of_soccer_campers : soccer_campers = 32 := by
  sorry

end number_of_soccer_campers_l60_60514


namespace cylindrical_to_rectangular_multiplied_l60_60392

theorem cylindrical_to_rectangular_multiplied :
  let r := 7
  let θ := Real.pi / 4
  let z := -3
  let x := r * Real.cos θ
  let y := r * Real.sin θ
  (2 * x, 2 * y, 2 * z) = (7 * Real.sqrt 2, 7 * Real.sqrt 2, -6) := 
by
  sorry

end cylindrical_to_rectangular_multiplied_l60_60392


namespace regular_soda_count_l60_60681

theorem regular_soda_count 
  (diet_soda : ℕ) 
  (additional_soda : ℕ) 
  (h1 : diet_soda = 19) 
  (h2 : additional_soda = 41) 
  : diet_soda + additional_soda = 60 :=
by
  sorry

end regular_soda_count_l60_60681


namespace area_ADC_proof_l60_60958

-- Definitions for the given conditions and question
variables (BD DC : ℝ) (ABD_area ADC_area : ℝ)

-- Conditions
def ratio_condition := BD / DC = 3 / 2
def ABD_area_condition := ABD_area = 30

-- Question rewritten as proof problem
theorem area_ADC_proof (h1 : ratio_condition BD DC) (h2 : ABD_area_condition ABD_area) :
  ADC_area = 20 :=
sorry

end area_ADC_proof_l60_60958


namespace contrapositive_property_l60_60787

def is_divisible_by_6 (n : ℤ) : Prop := n % 6 = 0
def is_divisible_by_2 (n : ℤ) : Prop := n % 2 = 0

theorem contrapositive_property :
  (∀ n : ℤ, is_divisible_by_6 n → is_divisible_by_2 n) ↔ (∀ n : ℤ, ¬ is_divisible_by_2 n → ¬ is_divisible_by_6 n) :=
by
  sorry

end contrapositive_property_l60_60787


namespace range_of_a_l60_60281

open Set

theorem range_of_a (a : ℝ) :
  (M : Set ℝ) = { x | -1 ≤ x ∧ x ≤ 2 } →
  (N : Set ℝ) = { x | 1 - 3 * a < x ∧ x ≤ 2 * a } →
  M ∩ N = M →
  1 ≤ a :=
by
  intro hM hN h_inter
  sorry

end range_of_a_l60_60281


namespace least_positive_integer_l60_60204

theorem least_positive_integer :
  ∃ N : ℕ, 
    (N % 7 = 5) ∧ 
    (N % 8 = 6) ∧ 
    (N % 9 = 7) ∧ 
    (N % 10 = 8) ∧
    N = 2518 :=
begin
  use 2518,
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  split, { exact Nat.mod_eq_of_lt dec_trivial (by norm_num) },
  refl
end

end least_positive_integer_l60_60204


namespace cosine_of_eight_times_alpha_l60_60231

theorem cosine_of_eight_times_alpha (α : ℝ) (hypotenuse : ℝ) 
  (cos_α : ℝ) (cos_2α : ℝ) (cos_4α : ℝ) 
  (h₀ : hypotenuse = Real.sqrt (1^2 + (Real.sqrt 2)^2))
  (h₁ : cos_α = (Real.sqrt 2) / hypotenuse)
  (h₂ : cos_2α = 2 * cos_α^2 - 1)
  (h₃ : cos_4α = 2 * cos_2α^2 - 1)
  (h₄ : cos_8α = 2 * cos_4α^2 - 1) :
  cos_8α = 17 / 81 := 
  by
  sorry

end cosine_of_eight_times_alpha_l60_60231


namespace sum_numbers_eq_432_l60_60462

theorem sum_numbers_eq_432 (n : ℕ) (h : (n * (n + 1)) / 2 = 432) : n = 28 :=
sorry

end sum_numbers_eq_432_l60_60462


namespace angle_B_value_triangle_perimeter_l60_60406

open Real

variables {A B C a b c : ℝ}

-- Statement 1
theorem angle_B_value (h1 : a = b * sin A + sqrt 3 * a * cos B) : B = π / 2 := by
  sorry

-- Statement 2
theorem triangle_perimeter 
  (h1 : B = π / 2)
  (h2 : b = 4)
  (h3 : (1 / 2) * a * c = 4) : 
  a + b + c = 4 + 4 * sqrt 2 := by
  sorry


end angle_B_value_triangle_perimeter_l60_60406


namespace find_value_l60_60485

theorem find_value : (100 + (20 / 90)) * 90 = 120 := by
  sorry

end find_value_l60_60485


namespace max_value_f_value_of_f_at_alpha_l60_60137

noncomputable def f (x : ℝ) : ℝ :=
  2 * (Real.cos (x / 2)) ^ 2 + Real.sqrt 3 * Real.sin x

theorem max_value_f :
  (∀ x, f x ≤ 3)
  ∧ (∃ x, f x = 3)
  ∧ {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} = {x : ℝ | ∃ k : ℤ, x = (π / 3) + 2 * k * π} :=
sorry

theorem value_of_f_at_alpha {α : ℝ} (h : Real.tan (α / 2) = 1 / 2) :
  f α = (8 + 4 * Real.sqrt 3) / 5 :=
sorry

end max_value_f_value_of_f_at_alpha_l60_60137


namespace n_minus_two_is_square_of_natural_number_l60_60464

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end n_minus_two_is_square_of_natural_number_l60_60464


namespace arith_seq_ratio_l60_60615

variable {S T : ℕ → ℚ}

-- Conditions
def is_arith_seq_sum (S : ℕ → ℚ) (a : ℕ → ℚ) :=
  ∀ n, S n = n * (2 * a 1 + (n - 1) * a n) / 2

def ratio_condition (S T : ℕ → ℚ) :=
  ∀ n, S n / T n = (2 * n - 1) / (3 * n + 2)

-- Main theorem
theorem arith_seq_ratio
  (a b : ℕ → ℚ)
  (h1 : is_arith_seq_sum S a)
  (h2 : is_arith_seq_sum T b)
  (h3 : ratio_condition S T)
  : a 7 / b 7 = 25 / 41 :=
sorry

end arith_seq_ratio_l60_60615


namespace angle_measure_l60_60830

theorem angle_measure (x : ℝ) 
  (h : x = 2 * (90 - x) - 60) : 
  x = 40 := 
  sorry

end angle_measure_l60_60830


namespace tagged_fish_in_second_catch_l60_60159

-- Definitions and conditions
def total_fish_in_pond : ℕ := 1750
def tagged_fish_initial : ℕ := 70
def fish_caught_second_time : ℕ := 50
def ratio_tagged_fish : ℚ := tagged_fish_initial / total_fish_in_pond

-- Theorem statement
theorem tagged_fish_in_second_catch (T : ℕ) : (T : ℚ) / fish_caught_second_time = ratio_tagged_fish → T = 2 :=
by
  sorry

end tagged_fish_in_second_catch_l60_60159


namespace radius_of_circle_l60_60026

theorem radius_of_circle (r : ℝ) (h : 3 * 2 * Real.pi * r = Real.pi * r ^ 2) : r = 6 := 
by 
  sorry

end radius_of_circle_l60_60026


namespace travel_times_either_24_or_72_l60_60121

variable (A B C : String)
variable (travel_time : String → String → Float)
variable (current : Float)

-- Conditions:
-- 1. Travel times are 24 minutes or 72 minutes
-- 2. Traveling from dock B cannot be balanced with current constraints
-- 3. A 3 km travel with the current is 24 minutes
-- 4. A 3 km travel against the current is 72 minutes

theorem travel_times_either_24_or_72 :
  (∀ (P Q : String), P = A ∨ P = B ∨ P = C ∧ Q = A ∨ Q = B ∨ Q = C →
  (travel_time A C = 72 ∨ travel_time C A = 24)) :=
by
  intros
  sorry

end travel_times_either_24_or_72_l60_60121


namespace exterior_angle_regular_polygon_l60_60500

theorem exterior_angle_regular_polygon (exterior_angle : ℝ) (sides : ℕ) (h : exterior_angle = 18) : sides = 20 :=
by
  -- Use the condition that the sum of exterior angles of any polygon is 360 degrees.
  have sum_exterior_angles : ℕ := 360
  -- Set up the equation 18 * sides = 360
  have equation : 18 * sides = sum_exterior_angles := sorry
  -- Therefore, sides = 20
  sorry

end exterior_angle_regular_polygon_l60_60500


namespace man_walking_rate_l60_60819

theorem man_walking_rate (x : ℝ) 
  (woman_rate : ℝ := 15)
  (woman_time_after_passing : ℝ := 2 / 60)
  (man_time_to_catch_up : ℝ := 4 / 60)
  (distance_woman : ℝ := woman_rate * woman_time_after_passing)
  (distance_man : ℝ := x * man_time_to_catch_up)
  (h : distance_man = distance_woman) :
  x = 7.5 :=
sorry

end man_walking_rate_l60_60819


namespace unknown_number_eq_0_5_l60_60347

theorem unknown_number_eq_0_5 : 
  ∃ x : ℝ, x + ((2 / 3) * (3 / 8) + 4) - (8 / 16) = 4.25 ∧ x = 0.5 :=
by
  use 0.5
  sorry

end unknown_number_eq_0_5_l60_60347


namespace even_function_implies_a_eq_2_l60_60554

def f (x a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) :
  (∀ x : ℝ, f x a = f (-x) a) → a = 2 :=
by
  sorry

end even_function_implies_a_eq_2_l60_60554


namespace find_d_l60_60083

theorem find_d (a d : ℕ) (h1 : a = 105) (h2 : a ^ 3 = 21 * 25 * 45 * d) : d = 49 :=
sorry

end find_d_l60_60083


namespace solve_inequality_system_l60_60991

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60991


namespace livestock_allocation_l60_60366

theorem livestock_allocation :
  ∃ (x y z : ℕ), x + y + z = 100 ∧ 20 * x + 6 * y + z = 200 ∧ x = 5 ∧ y = 1 ∧ z = 94 :=
by
  sorry

end livestock_allocation_l60_60366


namespace perpendicular_tangent_lines_l60_60878

def f (x : ℝ) : ℝ := x^3 + 1

noncomputable def tangent_line_eqs (x₀ : ℝ) (y₀ : ℝ) : Prop :=
  (3 * x₀ - y₀ - 1 = 0) ∨ (3 * x₀ - y₀ + 3 = 0)

theorem perpendicular_tangent_lines (x₀ : ℝ) (hx₀ : x₀ = 1 ∨ x₀ = -1) :
  tangent_line_eqs x₀ (f x₀) := by
  sorry

end perpendicular_tangent_lines_l60_60878


namespace count_three_digit_perfect_squares_divisible_by_4_l60_60148

theorem count_three_digit_perfect_squares_divisible_by_4 :
  ∃ (n : ℕ), n = 11 ∧ ∀ (k : ℕ), 10 ≤ k ∧ k ≤ 31 → (∃ m : ℕ, m^2 = k^2 ∧ 100 ≤ m^2 ∧ m^2 ≤ 999 ∧ m^2 % 4 = 0) := 
sorry

end count_three_digit_perfect_squares_divisible_by_4_l60_60148


namespace solve_inequality_system_l60_60999

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x) ∧ (x ≤ 1) :=
by
  sorry

end solve_inequality_system_l60_60999


namespace n_minus_two_is_square_of_natural_number_l60_60466

theorem n_minus_two_is_square_of_natural_number (n : ℕ) (h_n : n ≥ 3) (h_odd_m : Odd (1 / 2 * n * (n - 1))) :
  ∃ k : ℕ, n - 2 = k^2 := 
  by
  sorry

end n_minus_two_is_square_of_natural_number_l60_60466


namespace extreme_value_a_one_range_of_a_l60_60582

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x + 3

theorem extreme_value_a_one :
  ∀ x > 0, f x 1 ≤ f 1 1 := 
sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f x a ≤ 0) → a ≥ Real.exp 2 :=
sorry

end extreme_value_a_one_range_of_a_l60_60582


namespace diminished_value_is_seven_l60_60494

theorem diminished_value_is_seven (x y : ℕ) (hx : x = 280)
  (h_eq : x / 5 + 7 = x / 4 - y) : y = 7 :=
by {
  sorry
}

end diminished_value_is_seven_l60_60494


namespace probability_of_sum_5_when_two_dice_rolled_l60_60474

theorem probability_of_sum_5_when_two_dice_rolled :
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  (favorable_outcomes / total_possible_outcomes : ℝ) = (1 / 9 : ℝ) :=
by
  let total_possible_outcomes := 36
  let favorable_outcomes := 4
  have h : (favorable_outcomes : ℝ) / (total_possible_outcomes : ℝ) = (1 / 9 : ℝ) := sorry
  exact h

end probability_of_sum_5_when_two_dice_rolled_l60_60474


namespace proof_x_squared_plus_y_squared_l60_60731

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l60_60731


namespace original_price_l60_60820

theorem original_price (P : ℝ) (h₁ : P - 0.30 * P = 0.70 * P) (h₂ : P - 0.20 * P = 0.80 * P) (h₃ : 0.70 * P + 0.80 * P = 50) :
  P = 100 / 3 :=
by
  -- Proof skipped
  sorry

end original_price_l60_60820


namespace three_pow_sub_two_pow_prime_power_prime_l60_60449

theorem three_pow_sub_two_pow_prime_power_prime (n : ℕ) (hn : n > 0) (hp : ∃ p k : ℕ, Nat.Prime p ∧ 3^n - 2^n = p^k) : Nat.Prime n := 
sorry

end three_pow_sub_two_pow_prime_power_prime_l60_60449


namespace probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l60_60360

-- Definitions for the conditions
def total_balls : ℕ := 18
def initial_red_balls : ℕ := 12
def initial_white_balls : ℕ := 6
def probability_red_ball : ℚ := initial_red_balls / total_balls
def probability_white_ball_after_removal (x : ℕ) : ℚ := initial_white_balls / (total_balls - x)

-- Statement of the proof problem
theorem probability_red_ball_is_two_thirds : probability_red_ball = 2 / 3 := 
by sorry

theorem red_balls_taken_out_is_three : ∃ x : ℕ, probability_white_ball_after_removal x = 2 / 5 ∧ x = 3 := 
by sorry

end probability_red_ball_is_two_thirds_red_balls_taken_out_is_three_l60_60360


namespace maximum_diagonal_intersections_l60_60087

theorem maximum_diagonal_intersections (n : ℕ) (h : n ≥ 4) : 
  ∃ k, k = (n * (n - 1) * (n - 2) * (n - 3)) / 24 :=
by sorry

end maximum_diagonal_intersections_l60_60087


namespace operation_result_l60_60530

def star (a b c : ℝ) : ℝ := (a + b + c) ^ 2

theorem operation_result (x : ℝ) : star (x - 1) (1 - x) 1 = 1 := 
by
  sorry

end operation_result_l60_60530


namespace completing_the_square_l60_60072

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l60_60072


namespace curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l60_60879

-- Define the equation of the curve C
def curve_C (a x y : ℝ) : Prop := a * x^2 + a * y^2 - 2 * a^2 * x - 4 * y = 0

-- Prove that curve C is a circle
theorem curve_C_is_circle (a : ℝ) (h : a ≠ 0) :
  ∃ (h_c : ℝ), ∃ (k : ℝ), ∃ (r : ℝ), r > 0 ∧ ∀ (x y : ℝ), curve_C a x y ↔ (x - h_c)^2 + (y - k)^2 = r^2
:= sorry

-- Prove that the area of triangle AOB is constant
theorem area_AOB_constant (a : ℝ) (h : a ≠ 0) :
  ∃ (A B : ℝ × ℝ), (A = (2 * a, 0) ∧ B = (0, 4 / a)) ∧ 1/2 * (2 * a) * (4 / a) = 4
:= sorry

-- Find valid a and equation of curve C given conditions of line l and points M, N
theorem find_valid_a_and_curve_eq (a : ℝ) (h : a ≠ 0) :
  ∀ (M N : ℝ × ℝ), (|M.1 - 0| = |N.1 - 0| ∧ |M.2 - 0| = |N.2 - 0|) → (M.1 = N.1 ∧ M.2 = N.2) →
  y = -2 * x + 4 →  a = 2 ∧ ∀ (x y : ℝ), curve_C 2 x y ↔ x^2 + y^2 - 4 * x - 2 * y = 0
:= sorry

end curve_C_is_circle_area_AOB_constant_find_valid_a_and_curve_eq_l60_60879


namespace average_water_per_day_l60_60974

-- Define the given conditions as variables/constants
def day1 := 318
def day2 := 312
def day3_morning := 180
def day3_afternoon := 162

-- Define the total water added on day 3
def day3 := day3_morning + day3_afternoon

-- Define the total water added over three days
def total_water := day1 + day2 + day3

-- Define the number of days
def days := 3

-- The proof statement: the average water added per day is 324 liters
theorem average_water_per_day : total_water / days = 324 :=
by
  -- Placeholder for the proof
  sorry

end average_water_per_day_l60_60974


namespace proof_x_squared_plus_y_squared_l60_60732

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l60_60732


namespace problem_equation_has_solution_l60_60603

noncomputable def x (real_number : ℚ) : ℚ := 210 / 23

theorem problem_equation_has_solution (x_value : ℚ) : 
  (3 / 7) + (7 / x_value) = (10 / x_value) + (1 / 10) → 
  x_value = 210 / 23 :=
by
  intro h
  sorry

end problem_equation_has_solution_l60_60603


namespace regular_polygon_sides_l60_60504

theorem regular_polygon_sides (theta : ℝ) (h : theta = 18) : 
  ∃ n : ℕ, 360 / theta = n ∧ n = 20 :=
by
  use 20
  constructor
  apply eq_of_div_eq_mul
  rw h
  norm_num
  sorry

end regular_polygon_sides_l60_60504


namespace sets_of_consecutive_integers_summing_to_20_l60_60585

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end sets_of_consecutive_integers_summing_to_20_l60_60585


namespace pastries_calculation_l60_60838

theorem pastries_calculation 
    (G : ℕ) (C : ℕ) (P : ℕ) (F : ℕ)
    (hG : G = 30) 
    (hC : C = G - 5)
    (hP : P = G - 5)
    (htotal : C + P + F + G = 97) :
    C - F = 8 ∧ P - F = 8 :=
by
  sorry

end pastries_calculation_l60_60838


namespace xyz_value_l60_60573

theorem xyz_value (x y z : ℝ)
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 49)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 21) :
  x * y * z = 28 / 3 :=
by
  sorry

end xyz_value_l60_60573


namespace inequality_system_solution_l60_60984

theorem inequality_system_solution (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := 
by
  sorry

end inequality_system_solution_l60_60984


namespace find_a9_l60_60616

variable (a : ℕ → ℤ)  -- Arithmetic sequence
variable (S : ℕ → ℤ)  -- Sum of the first n terms

-- Conditions provided in the problem
axiom Sum_condition : S 8 = 4 * a 3
axiom Term_condition : a 7 = -2
axiom Sum_def : ∀ n, S n = (n * (a 1 + a n)) / 2

-- Hypothesis for common difference
def common_diff (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

-- Proving that a_9 equals -6 given the conditions
theorem find_a9 (d : ℤ) : common_diff a d → a 9 = -6 :=
by
  intros h
  sorry

end find_a9_l60_60616


namespace exists_three_irrationals_l60_60352

theorem exists_three_irrationals
    (x1 x2 x3 : ℝ)
    (h1 : ¬ ∃ q : ℚ, x1 = q)
    (h2 : ¬ ∃ q : ℚ, x2 = q)
    (h3 : ¬ ∃ q : ℚ, x3 = q)
    (sum_integer : ∃ n : ℤ, x1 + x2 + x3 = n)
    (sum_reciprocals_integer : ∃ m : ℤ, (1/x1) + (1/x2) + (1/x3) = m) :
  true :=
sorry

end exists_three_irrationals_l60_60352


namespace arithmetic_sequence_ninth_term_l60_60296

variable {a : ℕ → ℕ}

def is_arithmetic_sequence (a : ℕ → ℕ) :=
  ∃ d : ℕ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) :=
  (n * (2 * a 1 + (n - 1) * (a 2 - a 1))) / 2

theorem arithmetic_sequence_ninth_term
  (a: ℕ → ℕ)
  (h_arith: is_arithmetic_sequence a)
  (h_sum_5: sum_of_first_n_terms a 5 = 75)
  (h_a4: a 4 = 2 * a 2) :
  a 9 = 45 :=
sorry

end arithmetic_sequence_ninth_term_l60_60296


namespace find_sum_l60_60874

noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry
noncomputable def c : ℝ := sorry

theorem find_sum (h₁ : a * b = 2 * (a + b))
                (h₂ : b * c = 3 * (b + c))
                (h₃ : c * a = 4 * (a + c))
                (ha : a ≠ 0)
                (hb : b ≠ 0)
                (hc : c ≠ 0) 
                : a + b + c = 1128 / 35 :=
by
  sorry

end find_sum_l60_60874


namespace total_wicks_20_l60_60097

theorem total_wicks_20 (string_length_ft : ℕ) (length_wick_1 length_wick_2 : ℕ) (wicks_1 wicks_2 : ℕ) :
  string_length_ft = 15 →
  length_wick_1 = 6 →
  length_wick_2 = 12 →
  wicks_1 = wicks_2 →
  (string_length_ft * 12) = (length_wick_1 * wicks_1 + length_wick_2 * wicks_2) →
  (wicks_1 + wicks_2) = 20 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end total_wicks_20_l60_60097


namespace unique_positive_integer_solution_l60_60762

theorem unique_positive_integer_solution (p : ℕ) (hp : Nat.Prime p) (hop : p % 2 = 1) :
  ∃! (x y : ℕ), x^2 + p * x = y^2 ∧ x > 0 ∧ y > 0 :=
sorry

end unique_positive_integer_solution_l60_60762


namespace middle_admitted_is_correct_l60_60647

-- Define the total number of admitted people.
def total_admitted := 100

-- Define the proportions of South, North, and Middle volumes.
def south_ratio := 11
def north_ratio := 7
def middle_ratio := 2

-- Calculating the total ratio.
def total_ratio := south_ratio + north_ratio + middle_ratio

-- Hypothesis that we are dealing with the correct ratio and total.
def middle_admitted (total_admitted : ℕ) (total_ratio : ℕ) (middle_ratio : ℕ) : ℕ :=
  total_admitted * middle_ratio / total_ratio

-- Proof statement
theorem middle_admitted_is_correct :
  middle_admitted total_admitted total_ratio middle_ratio = 10 :=
by
  -- This line would usually contain the detailed proof steps, which are omitted here.
  sorry

end middle_admitted_is_correct_l60_60647


namespace evaluate_g_sum_l60_60612

def g (a b : ℚ) : ℚ :=
if a + b ≤ 5 then (a^2 * b - a + 3) / (3 * a) 
else (a * b^2 - b - 3) / (-3 * b)

theorem evaluate_g_sum : g 3 2 + g 3 3 = -1 / 3 :=
by
  sorry

end evaluate_g_sum_l60_60612


namespace count_two_digit_primes_with_units_digit_3_l60_60925

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l60_60925


namespace x_squared_minus_y_squared_l60_60297

theorem x_squared_minus_y_squared (x y : ℝ) (h1 : x + y = 9) (h2 : x - y = 3) : x^2 - y^2 = 27 := by
  sorry

end x_squared_minus_y_squared_l60_60297


namespace probability_A_mean_X_l60_60019

-- Define the conditions and events
noncomputable def prob_pass_level_1 : ℝ := 3 / 4
noncomputable def prob_pass_level_2 : ℝ := 2 / 3
noncomputable def prob_pass_level_3 : ℝ := 1 / 2
noncomputable def prob_continue : ℝ := 1 / 2

-- Define the events A1 and A2 based on the problem description
noncomputable def event_A1 : ℝ := prob_pass_level_1 * prob_continue * (1 - prob_pass_level_2)
noncomputable def event_A2 : ℝ := prob_pass_level_1 * prob_continue * prob_pass_level_2 * prob_continue * (1 - prob_pass_level_3)

-- Prove the probability that the guest successfully passes the first level but receives zero charity fund
theorem probability_A : (event_A1 + event_A2) = 3 / 16 := sorry

-- Define the probability distribution of X
noncomputable def prob_X_0 : ℝ := 1 - prob_pass_level_1 + (event_A1 + event_A2)
noncomputable def prob_X_1000 : ℝ := prob_pass_level_1 * prob_continue
noncomputable def prob_X_3000 : ℝ := prob_pass_level_1 * prob_continue * prob_pass_level_2 * prob_continue
noncomputable def prob_X_6000 : ℝ := prob_pass_level_1 * prob_continue * prob_pass_level_2 * prob_continue * prob_pass_level_3

-- Prove the mean of X
theorem mean_X : (0 * prob_X_0 + 1000 * prob_X_1000 + 3000 * prob_X_3000 + 6000 * prob_X_6000) = 1125 := sorry

end probability_A_mean_X_l60_60019


namespace range_of_PF1_minus_PF2_l60_60883

noncomputable def ellipse_property (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : Prop :=
  ∃ f : ℝ, f = (2 * Real.sqrt 5 / 5) * x0 ∧ f > 0 ∧ f < 2

theorem range_of_PF1_minus_PF2 (x0 : ℝ) (h1 : 0 < x0) (h2 : x0 < Real.sqrt 5) : 
  ellipse_property x0 h1 h2 := by
  sorry

end range_of_PF1_minus_PF2_l60_60883


namespace zero_point_exists_in_interval_l60_60791

noncomputable def f (x : ℝ) : ℝ := x + 2^x

theorem zero_point_exists_in_interval :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ f x = 0 :=
by
  existsi -0.5 -- This is not a formal proof; the existi -0.5 is just for example purposes
  sorry

end zero_point_exists_in_interval_l60_60791


namespace solve_system_l60_60190

variable {a b c : ℝ}
variable {x y z : ℝ}
variable {e1 e2 e3 : ℤ} -- Sign variables should be integers to express ±1 more easily 

axiom ax1 : x * (x + y) + z * (x - y) = a
axiom ax2 : y * (y + z) + x * (y - z) = b
axiom ax3 : z * (z + x) + y * (z - x) = c

theorem solve_system :
  (e1 = 1 ∨ e1 = -1) ∧ (e2 = 1 ∨ e2 = -1) ∧ (e3 = 1 ∨ e3 = -1) →
  x = (1/2) * (e1 * Real.sqrt (a + b) - e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) ∧
  y = (1/2) * (e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) - e3 * Real.sqrt (c + a)) ∧
  z = (1/2) * (-e1 * Real.sqrt (a + b) + e2 * Real.sqrt (b + c) + e3 * Real.sqrt (c + a)) :=
sorry -- proof goes here

end solve_system_l60_60190


namespace point_N_coordinates_l60_60567

/--
Given:
- point M with coordinates (5, -6)
- vector a = (1, -2)
- the vector NM equals 3 times vector a
Prove:
- the coordinates of point N are (2, 0)
-/

theorem point_N_coordinates (x y : ℝ) :
  let M := (5, -6)
  let a := (1, -2)
  let NM := (5 - x, -6 - y)
  3 * a = NM → 
  (x = 2 ∧ y = 0) :=
by 
  intros
  sorry

end point_N_coordinates_l60_60567


namespace completing_the_square_l60_60055

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l60_60055


namespace minimum_score_for_fourth_term_l60_60041

variable (score1 score2 score3 score4 : ℕ)
variable (avg_required : ℕ)

theorem minimum_score_for_fourth_term :
  score1 = 80 →
  score2 = 78 →
  score3 = 76 →
  avg_required = 85 →
  4 * avg_required - (score1 + score2 + score3) ≤ score4 :=
by
  sorry

end minimum_score_for_fourth_term_l60_60041


namespace Wilson_sledding_l60_60077

variable (T S : ℕ)

theorem Wilson_sledding (h1 : S = T / 2) (h2 : (2 * T) + (3 * S) = 14) : T = 4 := by
  sorry

end Wilson_sledding_l60_60077


namespace shaded_area_correct_l60_60160

def first_rectangle_area (w l : ℕ) : ℕ := w * l
def second_rectangle_area (w l : ℕ) : ℕ := w * l
def overlap_triangle_area (b h : ℕ) : ℕ := (b * h) / 2
def total_shaded_area (area1 area2 overlap : ℕ) : ℕ := area1 + area2 - overlap

theorem shaded_area_correct :
  let w1 := 4
  let l1 := 12
  let w2 := 5
  let l2 := 10
  let b := 4
  let h := 5
  let area1 := first_rectangle_area w1 l1
  let area2 := second_rectangle_area w2 l2
  let overlap := overlap_triangle_area b h
  total_shaded_area area1 area2 overlap = 88 := 
by
  sorry

end shaded_area_correct_l60_60160


namespace positive_difference_of_solutions_l60_60657

theorem positive_difference_of_solutions :
  let a := 1
  let b := -6
  let c := -28
  let discriminant := b^2 - 4 * a * c
  let solution1 := 3 + (Real.sqrt discriminant) / 2
  let solution2 := 3 - (Real.sqrt discriminant) / 2
  have h_discriminant : discriminant = 148 := by sorry
  Real.sqrt 148 = 2 * Real.sqrt 37 :=
 sorry

end positive_difference_of_solutions_l60_60657


namespace counting_adjacent_numbers_l60_60977

open Finset

/- The original problem translated into a Lean theorem statement -/
theorem counting_adjacent_numbers (n : ℕ) (k : ℕ) (h1 : n = 49) (h2 : k = 6) :
  (choose n k) - (choose (n - k) k) = (choose 49 6) - (choose 44 6) :=
by
  rw [h1, h2, choose, choose] sorry

end counting_adjacent_numbers_l60_60977


namespace arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l60_60216

open Nat

axiom students : Fin 7 → Type -- Define students indexed by their position in the line.

noncomputable def fact : ℕ → ℕ
| 0       => 1
| (n + 1) => (n + 1) * fact n

theorem arrangements_A_and_B_together :
  (2 * fact 6) = 1440 := 
by 
  sorry

theorem arrangements_A_not_head_B_not_tail :
  (fact 7 - 2 * fact 6 + fact 5) = 3720 := 
by 
  sorry

theorem arrangements_A_and_B_not_next :
  (3600) = 3600 := 
by 
  sorry

theorem arrangements_one_person_between_A_and_B :
  (fact 5 * 2) = 1200 := 
by 
  sorry

end arrangements_A_and_B_together_arrangements_A_not_head_B_not_tail_arrangements_A_and_B_not_next_arrangements_one_person_between_A_and_B_l60_60216


namespace max_value_of_expression_l60_60195

theorem max_value_of_expression (m : ℝ) : 4 - |2 - m| ≤ 4 :=
by 
  sorry

end max_value_of_expression_l60_60195


namespace two_digit_prime_numbers_with_units_digit_3_count_l60_60896

-- Conditions

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def two_digit_numbers_with_units_digit_3 := {13, 23, 33, 43, 53, 63, 73, 83, 93}

-- Final statement
theorem two_digit_prime_numbers_with_units_digit_3_count : 
  (finset.filter is_prime two_digit_numbers_with_units_digit_3).card = 6 := 
  sorry

end two_digit_prime_numbers_with_units_digit_3_count_l60_60896


namespace ratio_a_b_c_l60_60130

-- Given condition 14(a^2 + b^2 + c^2) = (a + 2b + 3c)^2
theorem ratio_a_b_c (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : 
  a / b = 1 / 2 ∧ b / c = 2 / 3 :=
by 
  sorry

end ratio_a_b_c_l60_60130


namespace tan_theta_minus_pi_four_l60_60594

theorem tan_theta_minus_pi_four (θ : ℝ) (h1 : π < θ) (h2 : θ < 3 * π / 2) (h3 : Real.sin θ = -3/5) :
  Real.tan (θ - π / 4) = -1 / 7 :=
sorry

end tan_theta_minus_pi_four_l60_60594


namespace cost_of_dog_l60_60317

-- Given conditions
def dollars_misha_has : ℕ := 34
def dollars_misha_needs_earn : ℕ := 13

-- Formal statement of the mathematic proof
theorem cost_of_dog : dollars_misha_has + dollars_misha_needs_earn = 47 := by
  sorry

end cost_of_dog_l60_60317


namespace combined_moles_l60_60772

def balanced_reaction (NaHCO3 HC2H3O2 H2O : ℕ) : Prop :=
  NaHCO3 + HC2H3O2 = H2O

theorem combined_moles (NaHCO3 HC2H3O2 : ℕ) 
  (h : balanced_reaction NaHCO3 HC2H3O2 3) : 
  NaHCO3 + HC2H3O2 = 6 :=
sorry

end combined_moles_l60_60772


namespace tomatoes_difference_is_50_l60_60476

variable (yesterday_tomatoes today_tomatoes total_tomatoes : ℕ)

theorem tomatoes_difference_is_50 
  (h1 : yesterday_tomatoes = 120)
  (h2 : total_tomatoes = 290)
  (h3 : total_tomatoes = today_tomatoes + yesterday_tomatoes) :
  today_tomatoes - yesterday_tomatoes = 50 := sorry

end tomatoes_difference_is_50_l60_60476


namespace quadratic_roots_eccentricities_l60_60031

theorem quadratic_roots_eccentricities :
  (∃ x y : ℝ, 3 * x^2 - 4 * x + 1 = 0 ∧ 3 * y^2 - 4 * y + 1 = 0 ∧ 
              (0 ≤ x ∧ x < 1) ∧ y = 1) :=
by
  -- Proof would go here
  sorry

end quadratic_roots_eccentricities_l60_60031


namespace arithmetic_sequence_property_l60_60439

theorem arithmetic_sequence_property
  (a : ℕ → ℝ)
  (S : ℕ → ℝ)
  (h1 : ((a 6 - 1)^3 + 2013 * (a 6 - 1)^3 = 1))
  (h2 : ((a 2008 - 1)^3 = -2013 * (a 2008 - 1)^3))
  (sum_formula : ∀ n, S n = n * a n) : 
  S 2013 = 2013 ∧ a 2008 < a 6 := 
sorry

end arithmetic_sequence_property_l60_60439


namespace problem_l60_60223

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end problem_l60_60223


namespace count_two_digit_primes_ending_in_3_l60_60935

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l60_60935


namespace baseball_cards_per_friend_l60_60783

theorem baseball_cards_per_friend (total_cards friends : ℕ) (h_total : total_cards = 24) (h_friends : friends = 4) : total_cards / friends = 6 :=
by
  sorry

end baseball_cards_per_friend_l60_60783


namespace variance_stability_l60_60207

theorem variance_stability (S2_A S2_B : ℝ) (hA : S2_A = 1.1) (hB : S2_B = 2.5) : ¬(S2_B < S2_A) :=
by {
  sorry
}

end variance_stability_l60_60207


namespace regular_polygon_sides_l60_60511

theorem regular_polygon_sides (angle : ℝ) (h_angle : angle = 18) : ∃ n : ℕ, n = 20 :=
by
  have sum_exterior_angles : ℝ := 360
  have num_sides := sum_exterior_angles / angle
  have h_num_sides : num_sides = 20 := by
    calc
      num_sides = 360 / 18 : by rw [h_angle]
      ... = 20 : by norm_num
  use 20
  rw ← h_num_sides
  sorry

end regular_polygon_sides_l60_60511


namespace count_two_digit_primes_with_units_digit_3_l60_60926

theorem count_two_digit_primes_with_units_digit_3 :
  let numbers_with_units_digit_3 := [13, 23, 33, 43, 53, 63, 73, 83, 93] in
  let primes := numbers_with_units_digit_3.filter (λ n, Nat.Prime n) in
  primes.length = 6 := by
  sorry

end count_two_digit_primes_with_units_digit_3_l60_60926


namespace find_original_price_l60_60005

theorem find_original_price (sale_price : ℕ) (discount : ℕ) (original_price : ℕ) 
  (h1 : sale_price = 60) 
  (h2 : discount = 40) 
  (h3 : original_price = sale_price / ((100 - discount) / 100)) : original_price = 100 :=
by
  sorry

end find_original_price_l60_60005


namespace convert_cylindrical_to_rectangular_l60_60247

-- Definitions of the conversion from cylindrical to rectangular coordinates
def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

-- The given cylindrical coordinates point
def point_cylindrical : ℝ × ℝ × ℝ := (5, Real.pi / 3, 2)

-- The expected rectangular coordinates result
def expected_point_rectangular : ℝ × ℝ × ℝ := (2.5, 5 * Real.sqrt 3 / 2, 2)

-- The theorem statement to prove
theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 5 (Real.pi / 3) 2 = expected_point_rectangular :=
by
  sorry

end convert_cylindrical_to_rectangular_l60_60247


namespace count_two_digit_prime_numbers_ending_in_3_l60_60913

-- Defining the set of two-digit numbers ending in 3
def two_digit_numbers_ending_in_3 := set.range (λ n, 10 * n + 3) ∩ {n | 10 ≤ n ∧ n < 100}

-- Defining the predicate for prime numbers
def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

-- The set of two-digit prime numbers ending in 3
def two_digit_prime_numbers_ending_in_3 : set ℕ := 
  {n | n ∈ two_digit_numbers_ending_in_3 ∧ is_prime n}

-- The theorem stating the number of such prime numbers is 6
theorem count_two_digit_prime_numbers_ending_in_3 : 
  (two_digit_prime_numbers_ending_in_3.to_finset.card = 6) :=
by sorry

end count_two_digit_prime_numbers_ending_in_3_l60_60913


namespace stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l60_60341

theorem stratified_sampling_number_of_boys (total_students : Nat) (num_girls : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : num_girls = 50) (h3 : selected_students = 25) :
  (total_students - num_girls) * selected_students / total_students = 15 :=
  sorry

theorem stratified_sampling_probability_of_boy (total_students : Nat) (selected_students : Nat)
  (h1 : total_students = 125) (h2 : selected_students = 25) :
  selected_students / total_students = 1 / 5 :=
  sorry

end stratified_sampling_number_of_boys_stratified_sampling_probability_of_boy_l60_60341


namespace convert_cylindrical_to_rectangular_l60_60246

noncomputable theory

open Real

-- Define the cylindrical coordinates
def r : ℝ := 5
def theta : ℝ := π / 3
def z_cylindrical : ℝ := 2

-- Define the expected rectangular coordinates
def x_rect : ℝ := 2.5
def y_rect : ℝ := 5 * sqrt(3) / 2
def z_rect : ℝ := 2

-- Lean 4 statement to verify conversion
theorem convert_cylindrical_to_rectangular
  (r θ z_cylindrical x_rect y_rect z_rect : ℝ)
  (hr : r = 5) (htheta : θ = π / 3) (hz : z_cylindrical = 2)
  (hx : x_rect = 5 * cos (π / 3)) (hy : y_rect = 5 * sin (π / 3)) (hz_rect : z_rect = z_cylindrical) :
  (x_rect, y_rect, z_rect) = (2.5, 5 * sqrt(3) / 2, 2) :=
by { 
  rw [hr, htheta, hz] at *,
  rw cos_pi_div_three at hx, 
  rw sin_pi_div_three at hy,
  exact ⟨hx, hy, hz_rect⟩,
  sorry
}

end convert_cylindrical_to_rectangular_l60_60246


namespace completing_the_square_l60_60071

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l60_60071


namespace completing_the_square_l60_60064

theorem completing_the_square (x : ℝ) : (x^2 - 6 * x + 8 = 0) -> ((x - 3)^2 = 1) := by
  sorry

end completing_the_square_l60_60064


namespace intersection_of_A_and_B_l60_60881

def A : Set ℕ := {0, 1, 2}
def B : Set ℕ := {x | x > 1}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end intersection_of_A_and_B_l60_60881


namespace circle_radius_l60_60028

theorem circle_radius (r : ℝ) (h_circumference : 2 * Real.pi * r) 
                      (h_area : Real.pi * r^2) 
                      (h_equation : 3 * (2 * Real.pi * r) = Real.pi * r^2) : 
                      r = 6 :=
by
  sorry

end circle_radius_l60_60028


namespace fraction_meaningful_l60_60427

theorem fraction_meaningful (a : ℝ) : (∃ x, x = 2 / (a + 1)) ↔ a ≠ -1 :=
by
  sorry

end fraction_meaningful_l60_60427


namespace sin_product_l60_60839

theorem sin_product :
  (Real.sin (12 * Real.pi / 180)) * 
  (Real.sin (36 * Real.pi / 180)) *
  (Real.sin (72 * Real.pi / 180)) *
  (Real.sin (84 * Real.pi / 180)) = 1 / 16 := 
by
  sorry

end sin_product_l60_60839


namespace determine_angle_range_l60_60403

variable (α : ℝ)

theorem determine_angle_range 
  (h1 : 0 < α) 
  (h2 : α < 2 * π) 
  (h_sin : Real.sin α < 0) 
  (h_cos : Real.cos α > 0) : 
  (3 * π / 2 < α ∧ α < 2 * π) := 
sorry

end determine_angle_range_l60_60403


namespace no_strictly_greater_polynomials_l60_60311

noncomputable def transformation (P : Polynomial ℝ) (k : ℕ) (a : ℝ) : Polynomial ℝ := 
  P + Polynomial.monomial k (2 * a) - Polynomial.monomial (k + 1) a

theorem no_strictly_greater_polynomials (P Q : Polynomial ℝ) 
  (H1 : ∃ (n : ℕ) (a : ℝ), Q = transformation P n a)
  (H2 : ∃ (n : ℕ) (a : ℝ), P = transformation Q n a) : 
  ∃ x : ℝ, P.eval x = Q.eval x :=
sorry

end no_strictly_greater_polynomials_l60_60311


namespace calculate_abc_over_def_l60_60355

theorem calculate_abc_over_def
  (a b c d e f : ℚ)
  (h1 : a / b = 1 / 3)
  (h2 : b / c = 2)
  (h3 : c / d = 1 / 2)
  (h4 : d / e = 3)
  (h5 : e / f = 1 / 2) :
  (a * b * c) / (d * e * f) = 1 / 2 :=
by
  sorry

end calculate_abc_over_def_l60_60355


namespace candy_distribution_l60_60663

theorem candy_distribution (n k : ℕ) (h1 : 3 < n) (h2 : n < 15) (h3 : 195 - n * k = 8) : k = 17 :=
  by
    sorry

end candy_distribution_l60_60663


namespace completing_the_square_l60_60044

theorem completing_the_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  sorry

end completing_the_square_l60_60044


namespace fraction_meaningful_l60_60426

theorem fraction_meaningful (a : ℝ) : (∃ b, b = 2 / (a + 1)) → a ≠ -1 :=
by
  sorry

end fraction_meaningful_l60_60426


namespace solve_inequality_system_l60_60990

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60990


namespace negation_equiv_l60_60795

-- Original proposition
def original_proposition (x : ℝ) : Prop := x > 0 ∧ x^2 - 5 * x + 6 > 0

-- Negated proposition
def negated_proposition : Prop := ∀ x : ℝ, x > 0 → x^2 - 5 * x + 6 ≤ 0

-- Statement of the theorem to prove
theorem negation_equiv : ¬(∃ x : ℝ, original_proposition x) ↔ negated_proposition :=
by sorry

end negation_equiv_l60_60795


namespace count_two_digit_primes_with_units_digit_3_l60_60917

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l60_60917


namespace minimum_cost_peking_opera_l60_60463

theorem minimum_cost_peking_opera (T p₆ p₁₀ : ℕ) (xₛ yₛ : ℕ) :
  T = 140 ∧ p₆ = 6 ∧ p₁₀ = 10 ∧ xₛ + yₛ = T ∧ yₛ ≥ 2 * xₛ →
  6 * xₛ + 10 * yₛ = 1216 ∧ xₛ = 46 ∧ yₛ = 94 :=
by
   -- Proving this is skipped (left as a sorry)
  sorry

end minimum_cost_peking_opera_l60_60463


namespace students_like_basketball_l60_60748

variable (B C B_inter_C B_union_C : ℕ)

theorem students_like_basketball (hC : C = 8) (hB_inter_C : B_inter_C = 3) (hB_union_C : B_union_C = 17) 
    (h_incl_excl : B_union_C = B + C - B_inter_C) : B = 12 := by 
  -- Given: 
  --   C = 8
  --   B_inter_C = 3
  --   B_union_C = 17
  --   B_union_C = B + C - B_inter_C
  -- Prove: 
  --   B = 12
  sorry

end students_like_basketball_l60_60748


namespace inequality_solution_l60_60706

theorem inequality_solution (x : ℝ) :
    (∀ t : ℝ, abs (t - 3) + abs (2 * t + 1) ≥ abs (2 * x - 1) + abs (x + 2)) ↔ 
    (-1 / 2 ≤ x ∧ x ≤ 5 / 6) :=
by
  sorry

end inequality_solution_l60_60706


namespace river_flow_volume_l60_60093

/-- Given a river depth of 7 meters, width of 75 meters, 
and flow rate of 4 kilometers per hour,
the volume of water running into the sea per minute 
is 35,001.75 cubic meters. -/
theorem river_flow_volume
  (depth : ℝ) (width : ℝ) (rate_kmph : ℝ)
  (depth_val : depth = 7)
  (width_val : width = 75)
  (rate_val : rate_kmph = 4) :
  ( width * depth * (rate_kmph * 1000 / 60) ) = 35001.75 :=
by
  rw [depth_val, width_val, rate_val]
  sorry

end river_flow_volume_l60_60093


namespace quadratic_roots_property_l60_60170

theorem quadratic_roots_property (m n : ℝ) 
  (h1 : ∀ x, x^2 - 2 * x - 2025 = (x - m) * (x - n))
  (h2 : m + n = 2) : 
  m^2 - 3 * m - n = 2023 := 
by 
  sorry

end quadratic_roots_property_l60_60170


namespace solve_for_n_l60_60111

theorem solve_for_n :
  ∃ n : ℤ, n + (n + 1) + (n + 2) + 3 = 15 ∧ n = 3 :=
by
  sorry

end solve_for_n_l60_60111


namespace proof_x_squared_plus_y_squared_l60_60733

def problem_conditions (x y : ℝ) :=
  x - y = 18 ∧ x*y = 9

theorem proof_x_squared_plus_y_squared (x y : ℝ) 
  (h : problem_conditions x y) : 
  x^2 + y^2 = 342 :=
by
  sorry

end proof_x_squared_plus_y_squared_l60_60733


namespace complement_intersection_l60_60413

open Set

theorem complement_intersection (U A B : Set ℕ)
  (hU : U = {1, 2, 3, 4, 5})
  (hA : A = {1, 5})
  (hB : B = {2, 4}) :
  ((U \ A) ∩ B) = {2, 4} :=
by
  sorry

end complement_intersection_l60_60413


namespace problem1_problem2_problem3_problem4_l60_60018

-- Problem 1
theorem problem1 (x : ℝ) (h : x * (5 * x + 4) = 5 * x + 4) : x = -4 / 5 ∨ x = 1 := 
sorry

-- Problem 2
theorem problem2 (x : ℝ) (h : -3 * x^2 + 22 * x - 24 = 0) : x = 6 ∨ x = 4 / 3 := 
sorry

-- Problem 3
theorem problem3 (x : ℝ) (h : (x + 8) * (x + 1) = -12) : x = -4 ∨ x = -5 := 
sorry

-- Problem 4
theorem problem4 (x : ℝ) (h : (3 * x + 2) * (x + 3) = x + 14) : x = -4 ∨ x = 2 / 3 := 
sorry

end problem1_problem2_problem3_problem4_l60_60018


namespace inequality_solutions_l60_60880

theorem inequality_solutions (a : ℚ) :
  (∀ x : ℕ, 0 < x ∧ x ≤ 3 → 3 * (x - 1) < 2 * (x + a) - 5) →
  (∃ x : ℕ, 0 < x ∧ x = 4 → ¬ (3 * (x - 1) < 2 * (x + a) - 5)) →
  (5 / 2 < a ∧ a ≤ 3) :=
sorry

end inequality_solutions_l60_60880


namespace glorias_ratio_l60_60145

variable (Q : ℕ) -- total number of quarters
variable (dimes : ℕ) -- total number of dimes, given as 350
variable (quarters_left : ℕ) -- number of quarters left

-- Given conditions
def conditions (Q dimes quarters_left : ℕ) : Prop :=
  dimes = 350 ∧
  quarters_left = (3 * Q) / 5 ∧
  (dimes + quarters_left = 392)

-- The ratio of dimes to quarters left
def ratio_of_dimes_to_quarters_left (dimes quarters_left : ℕ) : ℕ × ℕ :=
  let gcd := Nat.gcd dimes quarters_left
  (dimes / gcd, quarters_left / gcd)

theorem glorias_ratio (Q : ℕ) (quarters_left : ℕ) : conditions Q 350 quarters_left → ratio_of_dimes_to_quarters_left 350 quarters_left = (25, 3) := by 
  sorry

end glorias_ratio_l60_60145


namespace percentage_increase_consumption_l60_60336

theorem percentage_increase_consumption
  (T C : ℝ) 
  (h_tax : ∀ t, t = 0.60 * T)
  (h_revenue : ∀ r, r = 0.75 * T * C) :
  1.25 * C = (0.75 * T * C) / (0.60 * T) := by
sorry

end percentage_increase_consumption_l60_60336


namespace archer_scores_distribution_l60_60365

structure ArcherScores where
  hits_40 : ℕ
  hits_39 : ℕ
  hits_24 : ℕ
  hits_23 : ℕ
  hits_17 : ℕ
  hits_16 : ℕ
  total_score : ℕ

theorem archer_scores_distribution
  (dora : ArcherScores)
  (reggie : ArcherScores)
  (finch : ArcherScores)
  (h1 : dora.total_score = 120)
  (h2 : reggie.total_score = 110)
  (h3 : finch.total_score = 100)
  (h4 : dora.hits_40 + dora.hits_39 + dora.hits_24 + dora.hits_23 + dora.hits_17 + dora.hits_16 = 6)
  (h5 : reggie.hits_40 + reggie.hits_39 + reggie.hits_24 + reggie.hits_23 + reggie.hits_17 + reggie.hits_16 = 6)
  (h6 : finch.hits_40 + finch.hits_39 + finch.hits_24 + finch.hits_23 + finch.hits_17 + finch.hits_16 = 6)
  (h7 : 40 * dora.hits_40 + 39 * dora.hits_39 + 24 * dora.hits_24 + 23 * dora.hits_23 + 17 * dora.hits_17 + 16 * dora.hits_16 = 120)
  (h8 : 40 * reggie.hits_40 + 39 * reggie.hits_39 + 24 * reggie.hits_24 + 23 * reggie.hits_23 + 17 * reggie.hits_17 + 16 * reggie.hits_16 = 110)
  (h9 : 40 * finch.hits_40 + 39 * finch.hits_39 + 24 * finch.hits_24 + 23 * finch.hits_23 + 17 * finch.hits_17 + 16 * finch.hits_16 = 100)
  (h10 : dora.hits_40 = 1)
  (h11 : dora.hits_39 = 0)
  (h12 : dora.hits_24 = 0) :
  dora.hits_40 = 1 ∧ dora.hits_16 = 5 ∧ 
  reggie.hits_23 = 2 ∧ reggie.hits_16 = 4 ∧ 
  finch.hits_17 = 4 ∧ finch.hits_16 = 2 :=
sorry

end archer_scores_distribution_l60_60365


namespace sum_arithmetic_sequence_satisfies_conditions_l60_60456

theorem sum_arithmetic_sequence_satisfies_conditions :
  ∀ (a : ℕ → ℤ) (d : ℤ),
  (a 1 = 1) ∧ (d ≠ 0) ∧ ((a 3)^2 = (a 2) * (a 6)) →
  (6 * a 1 + (6 * 5 / 2) * d = -24) :=
by
  sorry

end sum_arithmetic_sequence_satisfies_conditions_l60_60456


namespace total_students_calculation_l60_60626

variable (x : ℕ)
variable (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ)
variable (total_students : ℕ)
variable (remaining_jelly_beans : ℕ)

-- Defining the number of boys as per the problem's conditions
def boys (x : ℕ) : ℕ := 2 * x + 3

-- Defining the jelly beans given to girls
def jelly_beans_given_to_girls (x girls_jelly_beans : ℕ) : Prop :=
  girls_jelly_beans = 2 * x * x

-- Defining the jelly beans given to boys
def jelly_beans_given_to_boys (x boys_jelly_beans : ℕ) : Prop :=
  boys_jelly_beans = 3 * (2 * x + 3) * (2 * x + 3)

-- Defining the total jelly beans given out
def total_jelly_beans_given_out (girls_jelly_beans boys_jelly_beans total_jelly_beans : ℕ) : Prop :=
  total_jelly_beans = girls_jelly_beans + boys_jelly_beans

-- Defining the total number of students
def total_students_in_class (x total_students : ℕ) : Prop :=
  total_students = x + boys x

-- Proving that the total number of students is 18 under given conditions
theorem total_students_calculation (h1 : jelly_beans_given_to_girls x girls_jelly_beans)
                                   (h2 : jelly_beans_given_to_boys x boys_jelly_beans)
                                   (h3 : total_jelly_beans_given_out girls_jelly_beans boys_jelly_beans total_jelly_beans)
                                   (h4 : total_jelly_beans - remaining_jelly_beans = 642)
                                   (h5 : remaining_jelly_beans = 3) :
                                   total_students = 18 :=
by
  sorry

end total_students_calculation_l60_60626


namespace no_rational_roots_of_polynomial_l60_60850

theorem no_rational_roots_of_polynomial :
  ¬ ∃ (x : ℚ), (3 * x^4 - 7 * x^3 - 4 * x^2 + 8 * x + 3 = 0) :=
by
  sorry

end no_rational_roots_of_polynomial_l60_60850


namespace giselle_paint_l60_60129

theorem giselle_paint (x : ℚ) (h1 : 5/7 = x/21) : x = 15 :=
by
  sorry

end giselle_paint_l60_60129


namespace find_num_terms_in_AP_l60_60300

-- Define the necessary conditions and prove the final result
theorem find_num_terms_in_AP
  (a d : ℝ) (n : ℕ)
  (h_even : n % 2 = 0)
  (h_last_term_difference : (n - 1 : ℝ) * d = 7.5)
  (h_sum_odd_terms : n * (a + (n - 2 : ℝ) / 2 * d) = 60)
  (h_sum_even_terms : n * (a + ((n - 1 : ℝ) / 2) * d + d) = 90) :
  n = 12 := 
sorry

end find_num_terms_in_AP_l60_60300


namespace problem_x2_plus_y2_l60_60739

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l60_60739


namespace simplify_expression_l60_60481

theorem simplify_expression: ∀ (a m n : ℤ),
  (n % 2 = 1) → 
  (m > n) → 
  ((-a)^n = -a^n) → 
  (a^m / a^n = a^(m-n)) → 
  (-5)^5 / 5^3 + 3^4 - 6 = 50 :=
by
  intros a m n hn hm hneg hdiv
  sorry

end simplify_expression_l60_60481


namespace min_odd_integers_l60_60470

theorem min_odd_integers :
  ∀ (a b c d e f g h : ℤ),
  a + b + c = 30 →
  a + b + c + d + e + f = 58 →
  a + b + c + d + e + f + g + h = 73 →
  ∃ (odd_count : ℕ), odd_count = 1 :=
by
  sorry

end min_odd_integers_l60_60470


namespace additional_charge_per_2_5_mile_l60_60165

theorem additional_charge_per_2_5_mile (x : ℝ) : 
  (∀ (total_charge distance charge_per_segment initial_fee : ℝ),
    total_charge = 5.65 →
    initial_fee = 2.5 →
    distance = 3.6 →
    charge_per_segment = (3.6 / (2/5)) →
    total_charge = initial_fee + charge_per_segment * x → 
    x = 0.35) :=
by
  intros total_charge distance charge_per_segment initial_fee
  intros h_total_charge h_initial_fee h_distance h_charge_per_segment h_eq
  sorry

end additional_charge_per_2_5_mile_l60_60165


namespace solve_inequality_system_l60_60992

theorem solve_inequality_system (x : ℝ) :
  (x > -6 - 2 * x) ∧ (x ≤ (3 + x) / 4) ↔ (-2 < x ∧ x ≤ 1) := by
  sorry

end solve_inequality_system_l60_60992


namespace triangle_side_length_b_l60_60605

/-
In a triangle ABC with angles such that ∠C = 4∠A, and sides such that a = 35 and c = 64, prove that the length of side b is 140 * cos²(A).
-/
theorem triangle_side_length_b (A C : ℝ) (a c : ℝ) (hC : C = 4 * A) (ha : a = 35) (hc : c = 64) :
  ∃ (b : ℝ), b = 140 * (Real.cos A) ^ 2 :=
by
  sorry

end triangle_side_length_b_l60_60605


namespace min_value_expression_l60_60968

theorem min_value_expression (x y : ℝ) : (∃ z : ℝ, (forall x y : ℝ, z ≤ 5*x^2 + 4*y^2 - 8*x*y + 2*x + 4) ∧ z = 3) :=
sorry

end min_value_expression_l60_60968


namespace problem_l60_60222

noncomputable def investment : ℝ := 13500
noncomputable def total_yield : ℝ := 19000
noncomputable def orchard_price_per_kg : ℝ := 4
noncomputable def market_price_per_kg (x : ℝ) : ℝ := x
noncomputable def daily_sales_rate_market : ℝ := 1000
noncomputable def days_to_sell_all (yield : ℝ) (rate : ℝ) : ℝ := yield / rate

-- Condition that x > 4
axiom x_gt_4 : ∀ (x : ℝ), x > 4

theorem problem (
  x : ℝ
) (hx : x > 4) : 
  -- Part 1
  days_to_sell_all total_yield daily_sales_rate_market = 19 ∧
  -- Part 2
  (total_yield * market_price_per_kg x - total_yield * orchard_price_per_kg) = 19000 * x - 76000 ∧
  -- Part 3
  (6000 * orchard_price_per_kg + (total_yield - 6000) * x - investment) = 13000 * x + 10500 :=
by sorry

end problem_l60_60222


namespace all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l60_60221

-- Conditions
def investment := 13500  -- in yuan
def total_yield := 19000 -- in kg
def price_orchard := 4   -- in yuan/kg
def price_market (x : ℝ) := x -- in yuan/kg
def market_daily_sale := 1000 -- in kg/day

-- Part 1: Days to sell all fruits in the market
theorem all_fruits_sold_in_market (x : ℝ) (h : x > 4) : total_yield / market_daily_sale = 19 :=
by
  sorry

-- Part 2: Income difference between market and orchard sales
theorem market_vs_orchard_income_diff (x : ℝ) (h : x > 4) : total_yield * price_market x - total_yield * price_orchard = 19000 * x - 76000 :=
by
  sorry

-- Part 3: Total profit from selling partly in the orchard and partly in the market
theorem total_profit (x : ℝ) (h : x > 4) : 6000 * price_orchard + (total_yield - 6000) * price_market x - investment = 13000 * x + 10500 :=
by
  sorry

end all_fruits_sold_in_market_market_vs_orchard_income_diff_total_profit_l60_60221


namespace intersection_range_l60_60140

noncomputable def f (a : ℝ) (x : ℝ) := a * x
noncomputable def g (x : ℝ) := Real.log x
noncomputable def F (a : ℝ) (x : ℝ) := f a x - g x

theorem intersection_range (a : ℝ) :
  (∃ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 ∧ f a x1 = g x1 ∧ f a x2 = g x2) ↔
  0 < a ∧ a < 1 / Real.exp 1 := by
  sorry

end intersection_range_l60_60140


namespace one_fourth_of_8_point_8_is_fraction_l60_60854

theorem one_fourth_of_8_point_8_is_fraction:
  (1 / 4) * 8.8 = 11 / 5 :=
by sorry

end one_fourth_of_8_point_8_is_fraction_l60_60854


namespace parts_per_day_system_l60_60632

variable (x y : ℕ)

def personA_parts_per_day (x : ℕ) : ℕ := x
def personB_parts_per_day (y : ℕ) : ℕ := y

-- First condition
def condition1 (x y : ℕ) : Prop :=
  6 * x = 5 * y

-- Second condition
def condition2 (x y : ℕ) : Prop :=
  30 + 4 * x = 4 * y - 10

theorem parts_per_day_system (x y : ℕ) :
  condition1 x y ∧ condition2 x y :=
sorry

end parts_per_day_system_l60_60632


namespace expected_disease_count_l60_60184

/-- Define the probability of an American suffering from the disease. -/
def probability_of_disease := 1 / 3

/-- Define the sample size of Americans surveyed. -/
def sample_size := 450

/-- Calculate the expected number of individuals suffering from the disease in the sample. -/
noncomputable def expected_number := probability_of_disease * sample_size

/-- State the theorem: the expected number of individuals suffering from the disease is 150. -/
theorem expected_disease_count : expected_number = 150 :=
by
  -- Proof is required but skipped using sorry.
  sorry

end expected_disease_count_l60_60184


namespace randy_final_amount_l60_60978

-- Conditions as definitions
def initial_dollars : ℝ := 30
def initial_euros : ℝ := 20
def lunch_cost : ℝ := 10
def ice_cream_percentage : ℝ := 0.25
def snack_percentage : ℝ := 0.10
def conversion_rate : ℝ := 0.85

-- Main proof statement without the proof body
theorem randy_final_amount :
  let euros_in_dollars := initial_euros / conversion_rate
  let total_dollars := initial_dollars + euros_in_dollars
  let dollars_after_lunch := total_dollars - lunch_cost
  let ice_cream_cost := dollars_after_lunch * ice_cream_percentage
  let dollars_after_ice_cream := dollars_after_lunch - ice_cream_cost
  let snack_euros := initial_euros * snack_percentage
  let snack_dollars := snack_euros / conversion_rate
  let final_dollars := dollars_after_ice_cream - snack_dollars
  final_dollars = 30.30 :=
by
  sorry

end randy_final_amount_l60_60978


namespace range_q_l60_60332

def q (x : ℝ ) : ℝ := x^4 + 4 * x^2 + 4

theorem range_q :
  (∀ y, ∃ x, 0 ≤ x ∧ q x = y ↔ y ∈ Set.Ici 4) :=
sorry

end range_q_l60_60332


namespace problem_x2_plus_y2_l60_60740

theorem problem_x2_plus_y2 (x y : ℝ) (h1 : x - y = 18) (h2 : x * y = 9) : x^2 + y^2 = 342 :=
sorry

end problem_x2_plus_y2_l60_60740


namespace completing_the_square_l60_60070

noncomputable def solve_quadratic (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let sqrt_disc := real.sqrt discriminant
  let x1 := (-b + sqrt_disc) / (2 * a)
  let x2 := (-b - sqrt_disc) / (2 * a)
  (x1, x2)

theorem completing_the_square (x : ℝ) :
  let c := x^2 - 6*x + 8
  c = 0 → (x-3)^2 = 1 :=
by
  intro h
  have h1 : x^2 - 6*x = -8 := by sorry
  have h2 : x^2 - 6*x + 9 = 9 - 8 := by sorry
  have h3 : (x-3)^2 = 1 := by sorry
  exact h3

end completing_the_square_l60_60070


namespace find_m_l60_60949

-- Define the conditions
def parabola_eq (m : ℝ) (x y : ℝ) : Prop := x^2 = m * y
def vertex_to_directrix_dist (d : ℝ) : Prop := d = 1 / 2

-- State the theorem
theorem find_m (m : ℝ) (x y d : ℝ) 
  (h1 : parabola_eq m x y) 
  (h2 : vertex_to_directrix_dist d) :
  m = 2 :=
by
  sorry

end find_m_l60_60949


namespace completing_square_l60_60059

theorem completing_square (x : ℝ) : x^2 - 6 * x + 8 = 0 → (x - 3)^2 = 1 :=
by
  intro h
  sorry

end completing_square_l60_60059


namespace even_function_implies_a_eq_2_l60_60560

def f (x : ℝ) (a : ℝ) : ℝ := x * exp x / (exp (a * x) - 1)

theorem even_function_implies_a_eq_2 (a : ℝ) (h_even : ∀ x, f (-x) a = f x a) : a = 2 :=
by
  intro x
  sorry

end even_function_implies_a_eq_2_l60_60560


namespace find_a_l60_60545

noncomputable def f (x a : ℝ) : ℝ := (x * (Real.exp x)) / (Real.exp (a * x) - 1)

theorem find_a (a : ℝ) (h : ∀ x, f x a = f (-x) a) : a = 2 :=
begin
  sorry
end

end find_a_l60_60545


namespace number_of_two_digit_primes_with_units_digit_three_l60_60932

open Nat

def is_two_digit (n : Nat) : Prop := 10 ≤ n ∧ n < 100
def has_units_digit_three (n : Nat) : Prop := n % 10 = 3
def is_prime (n : Nat) : Prop := n > 1 ∧ (∀ m : Nat, m ∣ n → m = 1 ∨ m = n)

def count_primes_with_units_digit_three : Nat :=
  (Finset.range 100).filter (λ x, is_two_digit x ∧ has_units_digit_three x ∧ is_prime x).card

theorem number_of_two_digit_primes_with_units_digit_three : count_primes_with_units_digit_three = 6 := by
  sorry

end number_of_two_digit_primes_with_units_digit_three_l60_60932


namespace enrollment_difference_l60_60667

theorem enrollment_difference :
  let Varsity := 1680
  let Northwest := 1170
  let Central := 1840
  let Greenbriar := 1090
  let Eastside := 1450
  Central - Greenbriar = 750 := 
by
  intros Varsity Northwest Central Greenbriar Eastside
  -- calculate the difference
  have h1 : 750 = 750 := rfl
  sorry

end enrollment_difference_l60_60667


namespace abs_neg_fraction_is_positive_l60_60649

-- Define the given negative fraction
def neg_fraction := (-1 : ℝ) / 3

-- The absolute value of the given fraction
def abs_of_neg_fraction := abs neg_fraction

-- Define the expected absolute value (correct answer)
def expected_abs_value := (1 : ℝ) / 3

-- The theorem stating that the absolute value of -1/3 is 1/3
theorem abs_neg_fraction_is_positive : abs_of_neg_fraction = expected_abs_value := by
  sorry

end abs_neg_fraction_is_positive_l60_60649


namespace adult_ticket_price_l60_60680

/-- 
The community center sells 85 tickets and collects $275 in total.
35 of those tickets are adult tickets. Each child's ticket costs $2.
We want to find the price of an adult ticket.
-/
theorem adult_ticket_price 
  (total_tickets : ℕ) 
  (total_revenue : ℚ) 
  (adult_tickets_sold : ℕ) 
  (child_ticket_price : ℚ)
  (h1 : total_tickets = 85)
  (h2 : total_revenue = 275) 
  (h3 : adult_tickets_sold = 35) 
  (h4 : child_ticket_price = 2) 
  : ∃ A : ℚ, (35 * A + 50 * 2 = 275) ∧ (A = 5) :=
by
  sorry

end adult_ticket_price_l60_60680


namespace num_zeros_in_expansion_l60_60152

noncomputable def bigNum := (10^11 - 2) ^ 2

theorem num_zeros_in_expansion : ∀ n : ℕ, bigNum = n ↔ (n = 9999999999900000000004) := sorry

end num_zeros_in_expansion_l60_60152


namespace john_total_payment_l60_60434

theorem john_total_payment :
  let cost_per_appointment := 400
  let total_appointments := 3
  let pet_insurance_cost := 100
  let insurance_coverage := 0.80
  let first_appointment_cost := cost_per_appointment
  let subsequent_appointments := total_appointments - 1
  let subsequent_appointments_cost := subsequent_appointments * cost_per_appointment
  let covered_cost := subsequent_appointments_cost * insurance_coverage
  let uncovered_cost := subsequent_appointments_cost - covered_cost
  let total_cost := first_appointment_cost + pet_insurance_cost + uncovered_cost
  total_cost = 660 :=
by
  sorry

end john_total_payment_l60_60434


namespace range_of_a_l60_60861

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (x > 0) ∧ (π^x = (a + 1) / (2 - a))) → (1 / 2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l60_60861


namespace find_derivative_at_2_l60_60875

-- Define f(x) as a generic quadratic function symmetric about x = 1
def f (a b : ℝ) (x : ℝ) : ℝ := a * (x - 1) ^ 2 + b

-- The derivative of f(x)
def f' (a b : ℝ) (x : ℝ) : ℝ := 2 * a * (x - 1)

-- The main theorem asserting the result
theorem find_derivative_at_2 (a b : ℝ) (h_slope : f' a b 0 = -2) : f' a b 2 = 2 :=
by
  -- Proof goes here
  sorry

end find_derivative_at_2_l60_60875


namespace harry_travel_time_l60_60286

variables (bus_time1 bus_time2 : ℕ) (walk_ratio : ℕ)
-- Conditions based on the problem
-- Harry has already been sat on the bus for 15 minutes.
def part1_time : ℕ := 15

-- and he knows the rest of the journey will take another 25 minutes.
def part2_time : ℕ := 25

-- The total bus journey time
def total_bus_time : ℕ := part1_time + part2_time

-- The walk from the bus stop to his house will take half the amount of time the bus journey took.
def walk_time : ℕ := total_bus_time / 2

-- Total travel time
def total_travel_time : ℕ := total_bus_time + walk_time

-- Rewrite the proof problem statement
theorem harry_travel_time : total_travel_time = 60 := by
  sorry

end harry_travel_time_l60_60286


namespace quad_to_square_l60_60645

theorem quad_to_square (a b z : ℝ)
  (h_dim : a = 9) 
  (h_dim2 : b = 16) 
  (h_area : a * b = z * z) :
  z = 12 :=
by
  -- Proof outline would go here, but let's skip the actual proof for this definition.
  sorry

end quad_to_square_l60_60645


namespace simplify_abs_expression_l60_60325

/-- Simplify the expression: |-4^3 + 5^2 - 6| and prove the result is equal to 45 -/
theorem simplify_abs_expression :
  |(- 4 ^ 3 + 5 ^ 2 - 6)| = 45 :=
by
  sorry

end simplify_abs_expression_l60_60325


namespace find_a4_l60_60718

def arithmetic_sequence_sum (n : ℕ) (a₁ d : ℤ) : ℤ := n / 2 * (2 * a₁ + (n - 1) * d)

theorem find_a4 (a₁ d : ℤ) (S₅ S₉ : ℤ) 
  (h₁ : arithmetic_sequence_sum 5 a₁ d = 35)
  (h₂ : arithmetic_sequence_sum 9 a₁ d = 117) :
  (a₁ + 3 * d) = 20 := 
sorry

end find_a4_l60_60718


namespace n_minus_two_is_square_of_natural_number_l60_60465

theorem n_minus_two_is_square_of_natural_number 
  (n m : ℕ) 
  (hn: n ≥ 3) 
  (hm: m = n * (n - 1) / 2) 
  (hm_odd: m % 2 = 1)
  (unique_rem: ∀ i j : ℕ, i ≠ j → (i + j) % m ≠ (i + j) % m) :
  ∃ k : ℕ, n - 2 = k * k := 
sorry

end n_minus_two_is_square_of_natural_number_l60_60465


namespace count_two_digit_primes_with_units_digit_3_l60_60907

theorem count_two_digit_primes_with_units_digit_3 : 
  ∃ n, n = 6 ∧ 
    (∀ k, 10 ≤ k ∧ k < 100 → k % 10 = 3 → Prime k → 
      k = 13 ∨ k = 23 ∨ k = 43 ∨ k = 53 ∨ k = 73 ∨ k = 83) :=
by {
  sorry
}

end count_two_digit_primes_with_units_digit_3_l60_60907


namespace count_two_digit_primes_with_units_digit_3_l60_60918

open Nat

def is_two_digit_prime_with_units_digit_3 (n : ℕ) : Prop :=
  n / 10 > 0 ∧ n / 10 < 10 ∧ is_prime n ∧ n % 10 = 3

theorem count_two_digit_primes_with_units_digit_3 : 
  (finset.filter is_two_digit_prime_with_units_digit_3 (finset.range 100)).card = 6 := 
by 
  sorry

end count_two_digit_primes_with_units_digit_3_l60_60918


namespace inequality_implies_strict_inequality_l60_60321

theorem inequality_implies_strict_inequality (x y z : ℝ) (h : x^2 + x * y + x * z < 0) : y^2 > 4 * x * z :=
sorry

end inequality_implies_strict_inequality_l60_60321


namespace one_fourth_of_8_8_is_11_over_5_l60_60852

theorem one_fourth_of_8_8_is_11_over_5 :
  ∀ (a b : ℚ), a = 8.8 → b = 1/4 → b * a = 11/5 :=
by
  assume a b : ℚ,
  assume ha : a = 8.8,
  assume hb : b = 1/4,
  sorry

end one_fourth_of_8_8_is_11_over_5_l60_60852


namespace max_a_is_2_l60_60721

noncomputable def max_value_of_a (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) : ℝ :=
  2

theorem max_a_is_2 (a b c : ℝ) (h1 : a + b + c = 0) (h2 : a^2 + b^2 + c^2 = 6) :
  max_value_of_a a b c h1 h2 = 2 :=
sorry

end max_a_is_2_l60_60721


namespace completing_the_square_l60_60054

-- Define the initial quadratic equation
def quadratic_eq (x : ℝ) := x^2 - 6*x + 8 = 0

-- The expected result after completing the square
def completed_square_eq (x : ℝ) := (x - 3)^2 = 1

-- The theorem statement
theorem completing_the_square : ∀ x : ℝ, quadratic_eq x → completed_square_eq x :=
by
  intro x h
  -- The steps to complete the proof would go here, but we're skipping it with sorry.
  sorry

end completing_the_square_l60_60054


namespace num_unique_permutations_l60_60889

-- Given: digits 3 and 8 being repeated as described.
-- Show: number of different permutations of the digits 3, 3, 3, 8, 8 is 10.

theorem num_unique_permutations : 
  let digits := [3, 3, 3, 8, 8] in
  let total_permutations := (5!).nat_abs in             -- 5! permutations
  let repeats_correction := (3!).nat_abs * (2!).nat_abs in -- Adjusting for repeated 3's and 8's
  let unique_permutations := total_permutations / repeats_correction in
  unique_permutations = 10 :=
by
  sorry

end num_unique_permutations_l60_60889


namespace min_value_sum_inverse_sq_l60_60715

theorem min_value_sum_inverse_sq (x y z : ℝ) (h_pos_x : 0 < x) (h_pos_y : 0 < y) (h_pos_z : 0 < z) (h_sum : x + y + z = 1) : 
  (39 + 1/x + 4/y + 9/z) ≥ 25 :=
by
    sorry

end min_value_sum_inverse_sq_l60_60715


namespace range_of_target_function_l60_60658

noncomputable def target_function (x : ℝ) : ℝ :=
  1 - 1 / (x^2 - 1)

theorem range_of_target_function :
  ∀ y : ℝ, ∃ x : ℝ, x ≠ 1 ∧ x ≠ -1 ∧ target_function x = y ↔ y ∈ (Set.Iio 1 ∪ Set.Ici 2) :=
by
  sorry

end range_of_target_function_l60_60658


namespace solve_pair_N_n_l60_60849

def is_solution_pair (N n : ℕ) : Prop :=
  N ^ 2 = 1 + n * (N + n)

def fibonacci : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fibonacci n + fibonacci (n + 1)

theorem solve_pair_N_n (N n : ℕ) (i : ℕ) :
  is_solution_pair N n ↔ N = fibonacci (i + 1) ∧ n = fibonacci i := sorry

end solve_pair_N_n_l60_60849


namespace number_is_composite_l60_60186

theorem number_is_composite : ∃ k l : ℕ, k * l = 53 * 83 * 109 + 40 * 66 * 96 ∧ k > 1 ∧ l > 1 :=
by
  have h1 : 53 + 96 = 149 := by norm_num
  have h2 : 83 + 66 = 149 := by norm_num
  have h3 : 109 + 40 = 149 := by norm_num
  sorry

end number_is_composite_l60_60186


namespace Nadia_distance_is_18_l60_60182

-- Variables and conditions
variables (x : ℕ)

-- Definitions based on conditions
def Hannah_walked (x : ℕ) : ℕ := x
def Nadia_walked (x : ℕ) : ℕ := 2 * x
def total_distance (x : ℕ) : ℕ := Hannah_walked x + Nadia_walked x

-- The proof statement
theorem Nadia_distance_is_18 (h : total_distance x = 27) : Nadia_walked x = 18 :=
by
  sorry

end Nadia_distance_is_18_l60_60182


namespace ratio_proof_l60_60712

variable {x y : ℝ}

theorem ratio_proof (h : x / y = 2 / 3) : x / (x + y) = 2 / 5 :=
by
  sorry

end ratio_proof_l60_60712


namespace Mongolian_Mathematical_Olympiad_54th_l60_60007

theorem Mongolian_Mathematical_Olympiad_54th {a b c : ℝ} (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) :
  a^4 + b^4 + c^4 + (a^2 / (b + c)^2) + (b^2 / (c + a)^2) + (c^2 / (a + b)^2) ≥ a * b + b * c + c * a :=
sorry

end Mongolian_Mathematical_Olympiad_54th_l60_60007


namespace find_x_l60_60211

theorem find_x (x : ℝ) (h : 0.25 * x = 200 - 30) : x = 680 := 
by
  sorry

end find_x_l60_60211


namespace election_winning_candidate_votes_l60_60800

theorem election_winning_candidate_votes (V : ℕ) 
  (h1 : V = (4 / 7) * V + 2000 + 4000) : 
  (4 / 7) * V = 8000 :=
by
  sorry

end election_winning_candidate_votes_l60_60800


namespace peaches_total_l60_60235

theorem peaches_total (n P : ℕ) (h1 : P - 6 * n = 57) (h2 : P = 9 * (n - 6) + 3) : P = 273 :=
by
  sorry

end peaches_total_l60_60235


namespace min_value_of_y_l60_60713

theorem min_value_of_y (x : ℝ) (hx : x > 0) : (∃ y, y = x + 4 / x^2 ∧ ∀ z, z = x + 4 / x^2 → z ≥ 3) :=
sorry

end min_value_of_y_l60_60713


namespace smartphone_charging_time_l60_60368

theorem smartphone_charging_time :
  ∀ (T S : ℕ), T = 53 → T + (1 / 2 : ℚ) * S = 66 → S = 26 :=
by
  intros T S hT equation
  sorry

end smartphone_charging_time_l60_60368


namespace students_left_correct_l60_60302

-- Define the initial number of students
def initial_students : ℕ := 8

-- Define the number of new students
def new_students : ℕ := 8

-- Define the final number of students
def final_students : ℕ := 11

-- Define the number of students who left during the year
def students_who_left : ℕ :=
  (initial_students + new_students) - final_students

theorem students_left_correct : students_who_left = 5 :=
by
  -- Instantiating the definitions
  let initial := initial_students
  let new := new_students
  let final := final_students

  -- Calculation of students who left
  let L := (initial + new) - final

  -- Asserting the result
  show L = 5
  sorry

end students_left_correct_l60_60302


namespace present_age_of_son_l60_60672

variable (S M : ℕ)

-- Conditions
def condition1 : Prop := M = S + 32
def condition2 : Prop := M + 2 = 2 * (S + 2)

-- Theorem stating the required proof
theorem present_age_of_son : condition1 S M ∧ condition2 S M → S = 30 := by
  sorry

end present_age_of_son_l60_60672


namespace count_two_digit_primes_ending_in_3_l60_60934

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def ends_in_3 (n : ℕ) : Prop := n % 10 = 3

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

theorem count_two_digit_primes_ending_in_3 : 
  (finset.filter (λ n, is_two_digit n ∧ ends_in_3 n ∧ is_prime n) (finset.range 100)).card = 6 :=
sorry

end count_two_digit_primes_ending_in_3_l60_60934


namespace sets_of_consecutive_integers_summing_to_20_l60_60586

def sum_of_consecutive_integers (a n : ℕ) : ℕ := n * a + (n * (n - 1)) / 2

theorem sets_of_consecutive_integers_summing_to_20 : 
  (∃ (a n : ℕ), n ≥ 2 ∧ sum_of_consecutive_integers a n = 20) ∧ 
  (∀ (a1 n1 a2 n2 : ℕ), 
    (n1 ≥ 2 ∧ sum_of_consecutive_integers a1 n1 = 20 ∧ 
    n2 ≥ 2 ∧ sum_of_consecutive_integers a2 n2 = 20) → 
    (a1 = a2 ∧ n1 = n2)) :=
sorry

end sets_of_consecutive_integers_summing_to_20_l60_60586


namespace stickers_in_either_not_both_l60_60700

def stickers_shared := 12
def emily_total_stickers := 22
def mia_unique_stickers := 10

theorem stickers_in_either_not_both : 
  (emily_total_stickers - stickers_shared) + mia_unique_stickers = 20 :=
by
  sorry

end stickers_in_either_not_both_l60_60700


namespace num_two_digit_primes_with_units_digit_three_l60_60903

def is_prime (n : ℕ) : Prop := Nat.Prime n

def has_units_digit_three (n : ℕ) : Prop := n % 10 == 3

def is_two_digit (n : ℕ) : Prop := n >= 10 ∧ n < 100

theorem num_two_digit_primes_with_units_digit_three : 
  {n | is_two_digit n ∧ is_prime n ∧ has_units_digit_three n}.to_finset.card = 6 := 
by 
  -- Fill in proof here
  sorry

end num_two_digit_primes_with_units_digit_three_l60_60903


namespace not_necessarily_divisor_l60_60438

def consecutive_product (k : ℤ) : ℤ := k * (k + 1) * (k + 2) * (k + 3)

theorem not_necessarily_divisor (k : ℤ) (hk : 8 ∣ consecutive_product k) : ¬ (48 ∣ consecutive_product k) :=
sorry

end not_necessarily_divisor_l60_60438


namespace harry_travel_time_l60_60285

def t_bus1 : ℕ := 15
def t_bus2 : ℕ := 25
def t_bus_journey : ℕ := t_bus1 + t_bus2
def t_walk : ℕ := t_bus_journey / 2
def t_total : ℕ := t_bus_journey + t_walk

theorem harry_travel_time : t_total = 60 := by
  -- Will be proved afterwards
  sorry

end harry_travel_time_l60_60285


namespace even_function_implies_a_eq_2_l60_60551

theorem even_function_implies_a_eq_2 (a : ℝ) 
  (h : ∀ x : ℝ, f x = f (-x)) 
  (f : ℝ → ℝ := λ x, (x * exp x) / (exp (a * x) - 1)) : a = 2 :=
sorry

end even_function_implies_a_eq_2_l60_60551


namespace cost_jam_l60_60523

noncomputable def cost_of_jam (N B J : ℕ) : ℝ :=
  N * J * 5 / 100

theorem cost_jam (N B J : ℕ) (h₁ : N > 1) (h₂ : 4 * N + 20 = 414) :
  cost_of_jam N B J = 2.25 := by
  sorry

end cost_jam_l60_60523


namespace identical_lines_pairs_count_l60_60856

theorem identical_lines_pairs_count : 
  ∃ P : Finset (ℝ × ℝ), (∀ p ∈ P, 
    (∃ a b, p = (a, b) ∧ 
      (∀ x y, 2 * x + a * y + b = 0 ↔ b * x + 3 * y - 9 = 0))) ∧ P.card = 2 :=
sorry

end identical_lines_pairs_count_l60_60856


namespace three_digit_squares_div_by_4_count_l60_60151

theorem three_digit_squares_div_by_4_count : 
  (finset.card ((finset.filter (λ x, 
    x % 4 = 0) 
    (finset.image (λ n : ℕ, n * n) 
      (finset.range 32)).filter 
        (λ x, 100 ≤ x ∧ x < 1000))) = 11) := 
by 
  sorry

end three_digit_squares_div_by_4_count_l60_60151


namespace not_sum_of_squares_of_form_4m_plus_3_l60_60015

theorem not_sum_of_squares_of_form_4m_plus_3 (n m : ℤ) (h : n = 4 * m + 3) : 
  ¬ ∃ a b : ℤ, n = a^2 + b^2 :=
by
  sorry

end not_sum_of_squares_of_form_4m_plus_3_l60_60015


namespace find_range_of_function_l60_60086

variable (a : ℝ) (x : ℝ)

def func (a x : ℝ) : ℝ := x^2 - 2*a*x - 1

theorem find_range_of_function (a : ℝ) :
  if a < 0 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -1 ≤ y ∧ y ≤ 3 - 4*a
  else if 0 ≤ a ∧ a ≤ 1 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ 3 - 4*a
  else if 1 < a ∧ a ≤ 2 then
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ -(a^2 + 1) ≤ y ∧ y ≤ -1
  else
    ∀ y, (∃ x, 0 ≤ x ∧ x ≤ 2 ∧ y = func a x) ↔ 3 - 4*a ≤ y ∧ y ≤ -1
:= sorry

end find_range_of_function_l60_60086


namespace system1_solution_system2_solution_l60_60779

-- Define the first system of equations and its solution
theorem system1_solution (x y : ℝ) : 
    (3 * (x - 1) = y + 5 ∧ 5 * (y - 1) = 3 * (x + 5)) ↔ (x = 5 ∧ y = 7) :=
sorry

-- Define the second system of equations and its solution
theorem system2_solution (x y a : ℝ) :
    (2 * x + 4 * y = a ∧ 7 * x - 2 * y = 3 * a) ↔ 
    (x = (7 / 16) * a ∧ y = (1 / 32) * a) :=
sorry

end system1_solution_system2_solution_l60_60779


namespace expression_takes_many_values_l60_60521

theorem expression_takes_many_values (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ -2) :
  (∃ y : ℝ, y ≠ 0 ∧ y ≠ (y + 1) ∧ 
    (3 * x ^ 2 + 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 7) / ((x - 3) * (x + 2)) = y) :=
by
  sorry

end expression_takes_many_values_l60_60521


namespace find_a_for_even_function_l60_60536

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def given_function (a : ℝ) (x : ℝ) : ℝ :=
  x * real.exp x / (real.exp (a * x) - 1)

theorem find_a_for_even_function :
  ∀ (a : ℝ), is_even_function (given_function a) ↔ a = 2 :=
by 
  intro a
  sorry

end find_a_for_even_function_l60_60536


namespace area_of_sector_l60_60725

-- Given conditions
def central_angle : ℝ := 2
def perimeter : ℝ := 8

-- Define variables and expressions
variable (r l : ℝ)

-- Equations based on the conditions
def eq1 := l + 2 * r = perimeter
def eq2 := l = central_angle * r

-- Assertion of the correct answer
theorem area_of_sector : ∃ r l : ℝ, eq1 r l ∧ eq2 r l ∧ (1 / 2 * l * r = 4) := by
  sorry

end area_of_sector_l60_60725


namespace distinct_prime_factors_exists_l60_60320

theorem distinct_prime_factors_exists (m n : ℕ) (hm : m > 0) (hn : n > 0) :
  ∃ k : ℕ, k > 0 ∧ (nat.factorization (2^k - m)).support.card ≥ n :=
by
  sorry

end distinct_prime_factors_exists_l60_60320


namespace taishan_maiden_tea_prices_l60_60817

theorem taishan_maiden_tea_prices (x y : ℝ) 
  (h1 : 30 * x + 20 * y = 6000)
  (h2 : 24 * x + 18 * y = 5100) :
  x = 100 ∧ y = 150 :=
by
  sorry

end taishan_maiden_tea_prices_l60_60817


namespace prescribedDosageLessThanTypical_l60_60219

noncomputable def prescribedDosage : ℝ := 12
noncomputable def bodyWeight : ℝ := 120
noncomputable def typicalDosagePer15Pounds : ℝ := 2
noncomputable def typicalDosage : ℝ := (bodyWeight / 15) * typicalDosagePer15Pounds
noncomputable def percentageDecrease : ℝ := ((typicalDosage - prescribedDosage) / typicalDosage) * 100

theorem prescribedDosageLessThanTypical :
  percentageDecrease = 25 :=
by
  sorry

end prescribedDosageLessThanTypical_l60_60219


namespace range_of_m_l60_60581

theorem range_of_m (m : ℝ) (y_P : ℝ) (h1 : -3 ≤ y_P) (h2 : y_P ≤ 0) :
  m = (2 + y_P) / 2 → -1 / 2 ≤ m ∧ m ≤ 1 :=
by
  sorry

end range_of_m_l60_60581


namespace smallest_lcm_l60_60944

theorem smallest_lcm (a b : ℕ) (h₁ : 1000 ≤ a ∧ a < 10000) (h₂ : 1000 ≤ b ∧ b < 10000) (h₃ : Nat.gcd a b = 5) : 
  Nat.lcm a b = 201000 :=
sorry

end smallest_lcm_l60_60944


namespace complex_fraction_evaluation_l60_60757

open Complex

theorem complex_fraction_evaluation (c d : ℂ) (hz : c ≠ 0) (hz' : d ≠ 0) (h : c^2 + c * d + d^2 = 0) :
  (c^12 + d^12) / (c^3 + d^3)^4 = 1 / 8 := 
by sorry

end complex_fraction_evaluation_l60_60757


namespace muffin_selection_count_l60_60634

/-- Number of ways Sam can buy six muffins from four types: blueberry, chocolate chip, bran, almond -/
theorem muffin_selection_count : 
  ∀(b ch br a : ℕ), b + ch + br + a = 6 → 
    (finset.card (finset.filter (λ (x : finset ℕ), x.sum = 6) ((finset.range 4).powerset))) = 84 :=
begin
  intros b ch br a h,
  have h1 : nat.choose (6 + 4 - 1) 3 = 84,
  { calc nat.choose 9 3 = 84 : by norm_num },
  exact h1,
end

end muffin_selection_count_l60_60634


namespace linear_equation_variables_l60_60591

theorem linear_equation_variables (m n : ℤ) (h1 : 3 * m - 2 * n = 1) (h2 : n - m = 1) : m = 0 ∧ n = 1 :=
by {
  sorry
}

end linear_equation_variables_l60_60591


namespace plastic_skulls_number_l60_60840

-- Define the conditions
def num_broomsticks : ℕ := 4
def num_spiderwebs : ℕ := 12
def num_pumpkins := 2 * num_spiderwebs
def num_cauldron : ℕ := 1
def budget_left_to_buy : ℕ := 20
def num_left_to_put_up : ℕ := 10
def total_decorations : ℕ := 83

-- The number of plastic skulls calculation as a function
def num_other_decorations : ℕ :=
  num_broomsticks + num_spiderwebs + num_pumpkins + num_cauldron + budget_left_to_buy + num_left_to_put_up

def num_plastic_skulls := total_decorations - num_other_decorations

-- The theorem to be proved
theorem plastic_skulls_number : num_plastic_skulls = 12 := by
  sorry

end plastic_skulls_number_l60_60840


namespace train_speed_l60_60095

theorem train_speed (length : ℝ) (time : ℝ) (h_length : length = 630) (h_time : time = 36) :
  (length / 1000) / (time / 3600) = 63 :=
by
  rw [h_length, h_time]
  sorry

end train_speed_l60_60095


namespace sandy_correct_sums_l60_60635

theorem sandy_correct_sums :
  ∃ c i : ℤ,
  c + i = 40 ∧
  4 * c - 3 * i = 72 ∧
  c = 27 :=
by 
  sorry

end sandy_correct_sums_l60_60635


namespace marbles_lost_l60_60625

theorem marbles_lost (initial_marbs remaining_marbs marbles_lost : ℕ)
  (h1 : initial_marbs = 38)
  (h2 : remaining_marbs = 23)
  : marbles_lost = initial_marbs - remaining_marbs :=
by
  sorry

end marbles_lost_l60_60625


namespace integral_equals_pi_l60_60241

noncomputable def integral_of_quarter_circle_area : ℝ :=
  ∫ x in (0 : ℝ)..2, real.sqrt (4 - x^2)

theorem integral_equals_pi :
  integral_of_quarter_circle_area = real.pi :=
sorry

end integral_equals_pi_l60_60241


namespace sum_of_numbers_eq_8140_l60_60479

def numbers : List ℤ := [1200, 1300, 1400, 1510, 1530, 1200]

theorem sum_of_numbers_eq_8140 : (numbers.sum = 8140) :=
by
  sorry

end sum_of_numbers_eq_8140_l60_60479


namespace count_right_triangles_with_conditions_l60_60843

theorem count_right_triangles_with_conditions :
  ∃ n : ℕ, n = 10 ∧
    (∀ (a b : ℕ),
      (a ^ 2 + b ^ 2 = (b + 2) ^ 2) →
      (b < 100) →
      (∃ k : ℕ, a = 2 * k ∧ k ^ 2 = b + 1) →
      n = 10) :=
by
  -- The proof goes here
  sorry

end count_right_triangles_with_conditions_l60_60843


namespace yz_sub_zx_sub_xy_l60_60267

theorem yz_sub_zx_sub_xy (x y z : ℝ) (h1 : x - y - z = 19) (h2 : x^2 + y^2 + z^2 ≠ 19) :
  yz - zx - xy = 171 := by
  sorry

end yz_sub_zx_sub_xy_l60_60267


namespace total_dolls_l60_60631

def initial_dolls : ℕ := 6
def grandmother_dolls : ℕ := 30
def received_dolls : ℕ := grandmother_dolls / 2

theorem total_dolls : initial_dolls + grandmother_dolls + received_dolls = 51 :=
by
  -- Simplify the right hand side
  sorry

end total_dolls_l60_60631
