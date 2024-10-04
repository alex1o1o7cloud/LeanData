import Mathlib

namespace minimum_value_of_reciprocal_product_l282_282940

theorem minimum_value_of_reciprocal_product (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + a * b + 2 * b = 30) : 
  ∃ m : ℝ, m = 1 / (a * b) ∧ m = 1 / 18 :=
sorry

end minimum_value_of_reciprocal_product_l282_282940


namespace concentration_of_second_solution_l282_282470

theorem concentration_of_second_solution 
    (W : ℝ) -- Weight of the original solution
    (h_orig : W > 0)
    (h_orig_conc : 0.10 * W) -- Original solution is 10% sugar by weight
    (h_replacement : W / 4) -- One fourth of the solution was replaced
    (h_final_conc : 0.16 * W) -- Final solution is 16% sugar by weight
  : (∃ S : ℝ, 0 <= S ∧ S = 34) :=
sorry

end concentration_of_second_solution_l282_282470


namespace B_completes_remaining_work_in_23_days_l282_282068

noncomputable def A_work_rate : ℝ := 1 / 45
noncomputable def B_work_rate : ℝ := 1 / 40
noncomputable def combined_work_rate : ℝ := A_work_rate + B_work_rate
noncomputable def work_done_together_in_9_days : ℝ := combined_work_rate * 9
noncomputable def remaining_work : ℝ := 1 - work_done_together_in_9_days
noncomputable def days_B_completes_remaining_work : ℝ := remaining_work / B_work_rate

theorem B_completes_remaining_work_in_23_days :
  days_B_completes_remaining_work = 23 :=
by 
  -- Proof omitted - please fill in the proof steps
  sorry

end B_completes_remaining_work_in_23_days_l282_282068


namespace base10_to_base8_440_l282_282892

theorem base10_to_base8_440 :
  ∃ k1 k2 k3,
    k1 = 6 ∧
    k2 = 7 ∧
    k3 = 0 ∧
    (440 = k1 * 64 + k2 * 8 + k3) ∧
    (64 = 8^2) ∧
    (8^3 > 440) :=
sorry

end base10_to_base8_440_l282_282892


namespace water_consumption_per_week_l282_282010

-- Definitions for the given conditions
def bottles_per_day := 2
def quarts_per_bottle := 1.5
def additional_ounces_per_day := 20
def days_per_week := 7
def ounces_per_quart := 32

-- Theorem to state the problem
theorem water_consumption_per_week :
  bottles_per_day * quarts_per_bottle * ounces_per_quart + additional_ounces_per_day 
  * days_per_week = 812 := 
by 
  sorry

end water_consumption_per_week_l282_282010


namespace arrange_bulbs_l282_282772

-- Define the conditions
def blue_bulbs : ℕ := 7
def red_bulbs : ℕ := 6
def white_bulbs : ℕ := 10

-- Calculate the binomial coefficients
def binom1 : ℕ := Nat.choose (blue_bulbs + red_bulbs) blue_bulbs
def binom2 : ℕ := Nat.choose (blue_bulbs + red_bulbs + 1) white_bulbs

-- Main theorem to prove the number of arrangements equals 1717716
theorem arrange_bulbs : binom1 * binom2 = 1717716 := sorry

end arrange_bulbs_l282_282772


namespace trig_identity_proof_l282_282945

theorem trig_identity_proof (x : ℝ) (h : sin (x + π / 3) = 1 / 3) :
  sin (5 * π / 3 - x) - cos (2 * x - π / 3) = 4 / 9 :=
by {
  sorry
}

end trig_identity_proof_l282_282945


namespace gcd_of_4410_and_10800_l282_282529

theorem gcd_of_4410_and_10800 : Nat.gcd 4410 10800 = 90 := 
by 
  sorry

end gcd_of_4410_and_10800_l282_282529


namespace goods_train_length_is_420_l282_282081

/-- The man's train speed in km/h. -/
def mans_train_speed_kmph : ℝ := 64

/-- The goods train speed in km/h. -/
def goods_train_speed_kmph : ℝ := 20

/-- The time taken for the trains to pass each other in seconds. -/
def passing_time_s : ℝ := 18

/-- The relative speed of two trains traveling in opposite directions in m/s. -/
noncomputable def relative_speed_mps : ℝ := 
  (mans_train_speed_kmph + goods_train_speed_kmph) * 1000 / 3600

/-- The length of the goods train in meters. -/
noncomputable def goods_train_length_m : ℝ := relative_speed_mps * passing_time_s

/-- The theorem stating the length of the goods train is 420 meters. -/
theorem goods_train_length_is_420 :
  goods_train_length_m = 420 :=
sorry

end goods_train_length_is_420_l282_282081


namespace prime_P_satisfies_condition_l282_282810

theorem prime_P_satisfies_condition
    (P : ℕ) 
    (h_prime : nat.prime P) 
    (h_prime_pow : nat.prime (P^6 + 3)) : 
    P^10 + 3 = 1027 :=
by
    sorry

end prime_P_satisfies_condition_l282_282810


namespace area_triangle_conditions_l282_282096

namespace Geometry

variables {A B C P Q R : Type} [AffineSpace A B C] [Point P Q R] 
variables (area : ∀ (X Y Z : Type), AffineSpace X Y Z → Type)

-- Definitions for the conditions
def is_triangle (X Y Z : Type) [AffineSpace X Y Z] : Prop := sorry -- Need actual definition
def on_side (P : Type) (XY : AffineSpace) : Prop := sorry -- Need actual definition for points being on sides

-- Conditions as hypotheses
variables (h1 : is_triangle A B C)
variables (h2 : on_side P BC)
variables (h3 : on_side Q CA)
variables (h4 : on_side R AB)
variables (cbp_pc : BP ≤ PC)
variables (ccq_qa : CQ ≤ QA)
variables (car_rb : AR ≤ RB)

-- Main proof statement
theorem area_triangle_conditions :
  (∃ (Δ1 : AffineSpace), Δ1 ∈ {AQR, BRP, CPQ} ∧ area Δ1 ≤ area PQR) ∧
  (cbp_pc → ccq_qa → car_rb → area PQR ≥ 1 / 4 * area ABC) :=
sorry

end Geometry

end area_triangle_conditions_l282_282096


namespace average_increase_l282_282070

theorem average_increase (A A' : ℕ) (runs_in_17th : ℕ) (total_innings : ℕ) (new_avg : ℕ) 
(h1 : total_innings = 17)
(h2 : runs_in_17th = 87)
(h3 : new_avg = 39)
(h4 : A' = new_avg)
(h5 : 16 * A + runs_in_17th = total_innings * new_avg) 
: A' - A = 3 := by
  sorry

end average_increase_l282_282070


namespace minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l282_282323

-- Definition for the minimum number of uninteresting vertices
def minimum_uninteresting_vertices (n : ℕ) (h : n > 3) : ℕ := 2

-- Theorem for the minimum number of uninteresting vertices
theorem minimum_uninteresting_vertices_correct (n : ℕ) (h : n > 3) :
  minimum_uninteresting_vertices n h = 2 := 
sorry

-- Definition for the maximum number of unusual vertices
def maximum_unusual_vertices (n : ℕ) (h : n > 3) : ℕ := 3

-- Theorem for the maximum number of unusual vertices
theorem maximum_unusual_vertices_correct (n : ℕ) (h : n > 3) :
  maximum_unusual_vertices n h = 3 :=
sorry

end minimum_uninteresting_vertices_correct_maximum_unusual_vertices_correct_l282_282323


namespace part1_part2_l282_282212

-- Step (1): Finding g(2a)
theorem part1 (k : ℤ) (f g : ℝ → ℝ) (a := k * Real.pi / 2):
  (∀ x : ℝ, f x = Real.cos x ^ 2) → 
  (∀ x : ℝ, g x = 1 / 2 + (Real.sqrt 3) * (Real.sin x * Real.cos x)) → 
  g (2 * a) = 1 / 2 := by 
  sorry

-- Step (2): Finding the range of h(x)
theorem part2 (f g : ℝ → ℝ) (h : ℝ → ℝ := λ x, f x + g x) :
  (∀ x : ℝ, f x = Real.cos x ^ 2) → 
  (∀ x : ℝ, g x = 1 / 2 + (Real.sqrt 3) * (Real.sin x * Real.cos x)) → 
  ∀ x : ℝ, 0 ≤ x ∧ x ≤ Real.pi / 2 → 
  1 / 2 ≤ h x ∧ h x ≤ 2 := by 
  sorry

end part1_part2_l282_282212


namespace train_speed_l282_282049

theorem train_speed (length time_speed: ℝ) (h1 : length = 400) (h2 : time_speed = 16) : length / time_speed = 25 := 
by
    sorry

end train_speed_l282_282049


namespace apples_count_l282_282701

def total_apples (mike_apples nancy_apples keith_apples : Nat) : Nat :=
  mike_apples + nancy_apples + keith_apples

theorem apples_count :
  total_apples 7 3 6 = 16 :=
by
  rfl

end apples_count_l282_282701


namespace cookie_contest_l282_282018

theorem cookie_contest (A B : ℚ) (hA : A = 5/6) (hB : B = 2/3) :
  A - B = 1/6 :=
by 
  sorry

end cookie_contest_l282_282018


namespace value_of_a_l282_282216

/--
Given that x = 3 is a solution to the equation 3x - 2a = 5,
prove that a = 2.
-/
theorem value_of_a (x a : ℤ) (h : 3 * x - 2 * a = 5) (hx : x = 3) : a = 2 :=
by
  sorry

end value_of_a_l282_282216


namespace binary_div_4_remainder_l282_282463

/-- 
  Question: What is the remainder when the binary number 100101110010_2 is divided by 4?
  Conditions: 
  - The given number is a binary number.
  - Identify that only the last two digits influence the remainder when divided by 4.
  Correct Answer:
  - The remainder is 2 (base 10).
-/
theorem binary_div_4_remainder : 
  let b := 0b100101110010 in 
  b % 4 = 2 :=
by sorry

end binary_div_4_remainder_l282_282463


namespace part1_part2_l282_282968

section

variable (a x : ℝ)

def A : Set ℝ := { x | x ≤ -1 } ∪ { x | x ≥ 5 }
def B (a : ℝ) : Set ℝ := { x | 2 * a ≤ x ∧ x ≤ a + 2 }

-- Part 1
theorem part1 (h : a = -1) :
  B a = { x | -2 ≤ x ∧ x ≤ 1 } ∧
  (A ∩ B a) = { x | -2 ≤ x ∧ x ≤ -1 } ∧
  (A ∪ B a) = { x | x ≤ 1 ∨ x ≥ 5 } := 
sorry

-- Part 2
theorem part2 (h : A ∩ B a = B a) :
  a ≤ -3 ∨ a > 2 := 
sorry

end

end part1_part2_l282_282968


namespace space_diagonals_in_polyhedron_l282_282075

noncomputable def number_of_space_diagonals (vertices edges triangular_faces quadrilateral_faces : ℕ) : ℕ :=
  let total_segments := (vertices * (vertices - 1)) / 2
      total_face_diagonals := 2 * quadrilateral_faces
  in total_segments - edges - total_face_diagonals

theorem space_diagonals_in_polyhedron : 
  ∀ (vertices edges faces triangular_faces quadrilateral_faces : ℕ), 
  vertices = 30 ∧ edges = 72 ∧ faces = 44 ∧ triangular_faces = 30 ∧ quadrilateral_faces = 14 →
  number_of_space_diagonals vertices edges triangular_faces quadrilateral_faces = 335 :=
by 
  intros vertices edges faces triangular_faces quadrilateral_faces h
  cases h with verts h1
  cases h1 with edgs h2
  cases h2 with fs h3
  cases h3 with trngl_faces qdrl_faces
  simp only [verts, edgs, fs, trngl_faces, qdrl_faces, number_of_space_diagonals]
  rw [verts, edgs, trngl_faces, qdrl_faces]
  have h5 : (30 * (30 - 1)) / 2 = 435, by norm_num,
  have h6 : 2 * 14 = 28, by norm_num,
  rw [h5, h6],
  norm_num
  sorry

end space_diagonals_in_polyhedron_l282_282075


namespace total_amount_paid_l282_282807

-- Definitions from the conditions
def quantity_grapes : ℕ := 8
def rate_grapes : ℕ := 70
def quantity_mangoes : ℕ := 9
def rate_mangoes : ℕ := 60

-- Main statement to prove
theorem total_amount_paid :
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes) = 1100 :=
by
  sorry

end total_amount_paid_l282_282807


namespace initial_rope_length_l282_282842

theorem initial_rope_length : 
  ∀ (π : ℝ), 
  ∀ (additional_area : ℝ) (new_rope_length : ℝ), 
  additional_area = 933.4285714285714 →
  new_rope_length = 21 →
  ∃ (initial_rope_length : ℝ), 
  additional_area = π * (new_rope_length^2 - initial_rope_length^2) ∧
  initial_rope_length = 12 :=
by
  sorry

end initial_rope_length_l282_282842


namespace solve_inequality_l282_282719

variable (x : ℝ)

def valid_domain (x : ℝ) : Prop :=
  (cos (π * x) > 0) ∧ (cos (π * x) ≠ 1) ∧ (x ≠ 3)

def inequality (x : ℝ) : Prop :=
  (3^(x^2 - 1) - 9 * 3^(5 * x + 3)) * log (cos (π)) (x^2 - 6 * x + 9) ≥ 0

theorem solve_inequality :
  (∀ x, valid_domain x → inequality x → 
    x ∈ set.Ioo (-0.5) 0 ∪ set.Ioo 0 0.5 ∪ 
    set.Ioo 1.5 2 ∪ set.Ioo 4 4.5 ∪ 
    set.Ioo 5.5 6) :=
by
  sorry

end solve_inequality_l282_282719


namespace paul_money_left_l282_282705

-- Conditions
def cost_of_bread : ℕ := 2
def cost_of_butter : ℕ := 3
def cost_of_juice : ℕ := 2 * cost_of_bread
def total_money : ℕ := 15

-- Definition of total cost
def total_cost := cost_of_bread + cost_of_butter + cost_of_juice

-- Statement of the theorem
theorem paul_money_left : total_money - total_cost = 6 := by
  -- Sorry, implementation skipped
  sorry

end paul_money_left_l282_282705


namespace sqrt_inequality_l282_282302

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_l282_282302


namespace common_divisors_60_90_l282_282626

theorem common_divisors_60_90 :
  ∃ (count : ℕ), 
  (∀ d, d ∣ 60 ∧ d ∣ 90 ↔ d ∈ {1, 2, 3, 5, 6, 10, 15, 30}) ∧ 
  count = 8 :=
by
  sorry

end common_divisors_60_90_l282_282626


namespace isosceles_triangle_angle_sum_l282_282330

theorem isosceles_triangle_angle_sum (x : ℝ) (h1 : x = 50 ∨ x = 65 ∨ x = 80) : (50 + 65 + 80 = 195) :=
by sorry

end isosceles_triangle_angle_sum_l282_282330


namespace integral_evaluation_l282_282118

noncomputable def integral_result : ℝ :=
  ∫ x in (set.Icc 0 real.pi), ((9 * x^2 + 9 * x + 11) * real.cos (3 * x))

theorem integral_evaluation :
  integral_result = -2 * real.pi - 2 :=
sorry

end integral_evaluation_l282_282118


namespace find_g_function_l282_282748

variable (g : ℝ → ℝ)

noncomputable def g_valid : Prop :=
∀ x y : ℝ, g(x + y) = 4^y * g(x) + 3^x * g(y)

noncomputable def g_condition : Prop :=
g(1) = 2

theorem find_g_function : g_valid g ∧ g_condition g → ∀ x : ℝ, g(x) = 2 * (4^x - 3^x) :=
sorry

end find_g_function_l282_282748


namespace time_needed_to_gather_remaining_flowers_l282_282351

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l282_282351


namespace zero_in_interval_l282_282233

theorem zero_in_interval {b : ℝ} (f : ℝ → ℝ)
  (h₁ : ∀ x, f x = 2 * b * x - 3 * b + 1)
  (h₂ : b > 1/5)
  (h₃ : b < 1) :
  ∃ x, -1 < x ∧ x < 1 ∧ f x = 0 :=
by
  sorry

end zero_in_interval_l282_282233


namespace triangle_isosceles_or_right_l282_282338

theorem triangle_isosceles_or_right
  (A B C : ℝ)
  (a b c : ℝ)
  (hABC : A + B + C = π)
  (ha : a = b * cos B / cos A) :
  (A = B) ∨ (A + B = π / 2) :=
sorry

end triangle_isosceles_or_right_l282_282338


namespace option_A_cannot_be_true_l282_282207

variable (a : ℕ → ℝ) (S : ℕ → ℝ)
variable (r : ℝ) -- common ratio for the geometric sequence
variable (n : ℕ) -- number of terms

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) :=
  ∀ n, a (n + 1) = r * a n

def sum_of_geometric_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) :=
  S 0 = a 0 ∧ ∀ n, S (n + 1) = S n + a (n + 1)

theorem option_A_cannot_be_true
  (h_geom : is_geometric_sequence a r)
  (h_sum : sum_of_geometric_sequence a S) :
  a 2016 * (S 2016 - S 2015) ≠ 0 :=
sorry

end option_A_cannot_be_true_l282_282207


namespace construct_all_naturals_starting_from_4_l282_282438

-- Define the operations f, g, h
def f (n : ℕ) : ℕ := 10 * n
def g (n : ℕ) : ℕ := 10 * n + 4
def h (n : ℕ) : ℕ := if n % 2 = 0 then n / 2 else n  -- h is only meaningful if n is even

-- Main theorem: prove that starting from 4, every natural number can be constructed
theorem construct_all_naturals_starting_from_4 :
  ∀ (n : ℕ), ∃ (k : ℕ), (f^[k] 4 = n ∨ g^[k] 4 = n ∨ h^[k] 4 = n) :=
by sorry


end construct_all_naturals_starting_from_4_l282_282438


namespace symmetry_center_of_f_l282_282427

-- Define the function f(x)
def f (x : ℝ) : ℝ := cos (2 * x - π / 6) * sin (2 * x) - 1 / 4

-- State the theorem about the symmetry center
theorem symmetry_center_of_f : (exists x : ℝ, (x, 0) = (7 * π / 24, 0)) ∧ is_symmetric_about_y f :=
by
  sorry

end symmetry_center_of_f_l282_282427


namespace tom_days_to_finish_l282_282774

noncomputable def days_to_finish_show
  (episodes : Nat) 
  (minutes_per_episode : Nat) 
  (hours_per_day : Nat) : Nat :=
  let total_minutes := episodes * minutes_per_episode
  let total_hours := total_minutes / 60
  total_hours / hours_per_day

theorem tom_days_to_finish :
  days_to_finish_show 90 20 2 = 15 :=
by
  -- the proof steps go here
  sorry

end tom_days_to_finish_l282_282774


namespace candies_per_packet_l282_282869

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l282_282869


namespace triangle_axy_obtuse_l282_282752

theorem triangle_axy_obtuse
  (A B C D X Y : Point)
  (inscribed_sphere_touches : ∃ (S : Sphere), touches S (Plane B C D) X)
  (exscribed_sphere_touches : ∃ (T : Sphere), touches T (Plane B C D) Y)
  : obtuse_triangle A X Y := 
sorry

end triangle_axy_obtuse_l282_282752


namespace set_intersection_complement_eq_l282_282614

-- Defining the universal set
def U := {0, 1, 2, 3, 4}

-- Defining the sets M and N
def M := {0, 1, 2}
def N := {2, 3}

-- Defining the complement of N with respect to U
def complement_U_N := {x | x ∈ U ∧ x ∉ N}

-- Stating the proof problem
theorem set_intersection_complement_eq :
  M ∩ complement_U_N = {0, 1} :=
sorry

end set_intersection_complement_eq_l282_282614


namespace equilateral_triangle_in_square_l282_282707
-- Required to bring in all the necessary math libraries

-- Definition of a point and various geometric primitives
structure Point (α : Type) := (x : α) (y : α)
structure Triangle (α : Type) := (A B C : Point α)
structure Square (α : Type) := (A B C D : Point α)

-- Lean statement specifying the conditions and the proof goal
theorem equilateral_triangle_in_square
  (α : Type) [ordered_field α] -- α should allow geometric operations
  (A B C D : Point α) -- Vertices of the square
  (P : Point α) -- Any point on the perimeter of the square
  (h_sq : Square α) -- Ensuring A, B, C, D form a square
  : ∃ (Q R : Point α), (Triangle.mk P Q R).form = equilateral_triangle ∧ 
    P.is_on_perimeter_of (Square.mk A B C D) ∧ 
    Q.is_on_perimeter_of (Square.mk A B C D) ∧ 
    R.is_on_perimeter_of (Square.mk A B C D) := 
sorry

end equilateral_triangle_in_square_l282_282707


namespace domain_of_f_l282_282741

noncomputable def f (x : ℝ) : ℝ := 1 / (Real.sqrt (2 * x - 3))

theorem domain_of_f :
  ∀ x, f x = 1 / (Real.sqrt (2 * x - 3))
  → ∃ (a : ℝ), ∀ (x : ℝ), (2 * x - 3 > 0) ↔ (x > a) :=
by
  exists 3 / 2
  sorry

end domain_of_f_l282_282741


namespace range_a_if_no_solution_l282_282377

def f (x : ℝ) : ℝ := abs (x - abs (2 * x - 4))

theorem range_a_if_no_solution (a : ℝ) :
  (∀ x : ℝ, f x > 0 → false) → a < 1 :=
by
  sorry

end range_a_if_no_solution_l282_282377


namespace largest_trio_sum_l282_282066

def is_trio (a b c : ℕ) : Prop :=
  a ≠ b ∧ b ≠ c ∧ c ≠ a ∧
  (a ∣ b ∨ b ∣ a ∨ b ∣ c ∨ c ∣ b ∨ c ∣ a ∨ a ∣ c) ∧
  a ∈ finset.Icc 1 2002 ∧ b ∈ finset.Icc 1 2002 ∧ c ∈ finset.Icc 1 2002

theorem largest_trio_sum :
  ∀ a b c : ℕ, is_trio a b c → a + b + c ≤ 4002
  ∧ (∃ a ∈ {1, 2, 7, 11, 13, 14, 22, 26, 77, 91, 143, 154, 182, 286}, is_trio a (2002 - a) 2002 ∧ a + (2002 - a) + 2002 = 4004) :=
by sorry

end largest_trio_sum_l282_282066


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282284

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282284


namespace necessary_but_not_sufficient_l282_282144

theorem necessary_but_not_sufficient (a c : ℝ) : 
  (c ≠ 0) → (∀ (x y : ℝ), ax^2 + y^2 = c → (c = 0 → false) ∧ (c ≠ 0 → (∃ x y : ℝ, ax^2 + y^2 = c))) :=
by
  sorry

end necessary_but_not_sufficient_l282_282144


namespace completing_the_square_l282_282044

theorem completing_the_square :
  ∀ x : ℝ, x^2 - 4 * x - 2 = 0 ↔ (x - 2)^2 = 6 :=
by
  sorry

end completing_the_square_l282_282044


namespace sufficient_but_not_necessary_condition_l282_282613

-- Define set P
def P : Set ℝ := {1, 2, 3, 4}

-- Define set Q
def Q : Set ℝ := {x | 0 < x ∧ x < 5}

-- Theorem statement
theorem sufficient_but_not_necessary_condition (x : ℝ) :
  (x ∈ P → x ∈ Q) ∧ (¬(x ∈ Q → x ∈ P)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l282_282613


namespace smallest_n_l282_282967

-- Define the sequence with the given recurrence relation
def a : ℕ → ℝ
| 1       := 9
| (n + 1) := (4 - a n) / 3

-- Define the sum of the first n terms
def S (n : ℕ) : ℝ :=
  ∑ i in Finset.range n, a (i + 1)

-- Define the condition to be proven
theorem smallest_n (n : ℕ) (hn : n = 7) :
  |S n - n - 6| < 1 / 125 :=
sorry

end smallest_n_l282_282967


namespace coefficient_x3y5_in_expansion_l282_282040

theorem coefficient_x3y5_in_expansion : 
  binomial 8 5 = 56 :=
by 
  sorry

end coefficient_x3y5_in_expansion_l282_282040


namespace part_I_part_II_l282_282570

variable (a : ℝ)

def A (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ a + 3 }
def B : Set ℝ := { x | x > 1 ∨ x < -6 }

theorem part_I : (A a ∩ B = ∅) → -6 ≤ a ∧ a ≤ -2 :=
by
  sorry

theorem part_II : (A a ∪ B = B) → (a < -9 ∨ a > 1) :=
by
  sorry

end part_I_part_II_l282_282570


namespace function_tangent_and_max_k_l282_282600

noncomputable def f (x : ℝ) : ℝ := x * Real.log x + 2 * x - 1

theorem function_tangent_and_max_k 
  (x : ℝ) (h1 : 0 < x) 
  (h2 : 3 * x - y - 2 = 0) : 
  (∀ k : ℤ, (∀ x : ℝ, 1 < x → k < (f x) / (x - 1)) → k ≤ 4) := 
sorry

end function_tangent_and_max_k_l282_282600


namespace find_x_l282_282633

theorem find_x (x : ℝ) (h : (3 * x - 4) / 7 = 15) : x = 109 / 3 :=
by sorry

end find_x_l282_282633


namespace event_1_is_random_l282_282664

-- Conditions
def coin_toss_event : Prop := 
  "Tossing a coin twice in succession and getting heads both times"

def attraction_event : Prop := 
  "Opposite charges attract each other"

def freezing_event : Prop := 
  "Water freezes at 100°C under standard atmospheric pressure"

-- Random event definition
def is_random_event (event : Prop) : Prop :=
  ¬(event = true)

-- Theorem stating that the coin toss event is random
theorem event_1_is_random : is_random_event coin_toss_event :=
sorry

end event_1_is_random_l282_282664


namespace area_of_triangle_ABC_l282_282376

noncomputable theory

variables (O A B C : ℝ × ℝ × ℝ)
variables (OA OB OC : ℝ) (α : ℝ)

-- Definitions from conditions
def is_origin (O : ℝ × ℝ × ℝ) := O = (0, 0, 0)
def on_positive_x_axis (A : ℝ × ℝ × ℝ) := ∃ x : ℝ, x > 0 ∧ A = (x, 0, 0) ∧ OA = x
def on_positive_y_axis (B : ℝ × ℝ × ℝ) := ∃ y : ℝ, y > 0 ∧ B = (0, y, 0) ∧ OB = y
def on_positive_z_axis (C : ℝ × ℝ × ℝ) := ∃ z : ℝ, z > 0 ∧ C = (0, 0, z) ∧ OC = z
def angle_BAC_eq_45 (α : ℝ) := α = real.pi / 4

-- Define the area of the triangle
def triangle_area (A B C : ℝ × ℝ × ℝ) : ℝ :=
  let AB := real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2 + (A.3 - B.3)^2) in
  let AC := real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2 + (A.3 - C.3)^2) in
  let sin_α := real.sin α in
  0.5 * AB * AC * sin_α

-- The theorem
theorem area_of_triangle_ABC :
  is_origin O →
  on_positive_x_axis A →
  on_positive_y_axis B →
  on_positive_z_axis C →
  OA = 4 →
  OB = 5 →
  OC = 5 →
  angle_BAC_eq_45 α →
  triangle_area A B C = 41 * real.sqrt 2 / 4 :=
by
  intros hO hA hB hC hOA hOB hOC hα
  sorry

end area_of_triangle_ABC_l282_282376


namespace triangle_area_ratio_l282_282051

-- Define the conditions and theorem
theorem triangle_area_ratio (ABC : Type*) [triangle ABC]
  (A B C E F G : ABC)
  (k : ℝ)
  (h1 : 0 < k)
  (h2 : k < 1)
  (h3 : AE / EB = k)
  (h4 : BF / FC = k)
  (h5 : CG / GA = k)
  :
  area (AF ∩ BG ∩ CE) / area ABC = (1 - k)^2 / (k^2 + k + 1) :=
sorry

end triangle_area_ratio_l282_282051


namespace sqrt_x_minus_3_domain_l282_282312

theorem sqrt_x_minus_3_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_domain_l282_282312


namespace time_needed_to_gather_remaining_flowers_l282_282350

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l282_282350


namespace moles_of_CO2_l282_282913

theorem moles_of_CO2 (moles_HCl moles_CaCO3 : ℕ) 
  (balanced_eq : CaCO3 + 2 * HCl → CaCl2 + CO2 + H2O) 
  (h_HCl : moles_HCl = 6) 
  (h_CaCO3 : moles_CaCO3 = 3) : 
  (∃ moles_CO2 : ℕ, moles_CO2 = 3) :=
sorry

end moles_of_CO2_l282_282913


namespace six_people_painting_l282_282901

variables (People Hours Const : ℝ)

-- Definition of work done (Work constant)
def work_done (n : ℝ) (t : ℝ) : ℝ := n * t

-- Given conditions
def eight_people_painting : work_done 8 12 = 96 := by 
  unfold work_done 
  simp

-- Proof we need to show
theorem six_people_painting : work_done 6 16 = 96 :=
by
  unfold work_done
  simp
  sorry

end six_people_painting_l282_282901


namespace max_Xs_is_five_l282_282393

-- Define the problem in the Lean 4 statement
def max_Xs_on_3x3_grid : Prop :=
  ∀ (grid : Fin 3 × Fin 3 → Bool),
    (∀ r, ∃ c, (grid (r, 0) = false) ∨ (grid (r, 1) = false) ∨ (grid (r, 2) = false)) ∧
    (∀ c, ∃ r, (grid (0, c) = false) ∨ (grid (1, c) = false) ∨ (grid (2, c) = false)) →
    ∑ i : Fin 3, ∑ j : Fin 3, (if grid (i, j) then 1 else 0) ≤ 5

theorem max_Xs_is_five : max_Xs_on_3x3_grid :=
  sorry

end max_Xs_is_five_l282_282393


namespace number_times_half_squared_eq_eight_l282_282035

theorem number_times_half_squared_eq_eight : 
  ∃ n : ℝ, n * (1/2)^2 = 2^3 := 
sorry

end number_times_half_squared_eq_eight_l282_282035


namespace exists_excursion_with_minimal_participation_l282_282645

theorem exists_excursion_with_minimal_participation (students : Fin 20 → Type) (excursions : Type) [Fintype excursions] (attends : students → excursions → Prop) :
  ∃ e : excursions, ∀ s : Fin 20, attends s e → ∃ k : ℕ, k ≥ Fintype.card excursions / 20 :=
sorry

end exists_excursion_with_minimal_participation_l282_282645


namespace problem_statement_l282_282642

-- Definitions for conditions
def cond_A : Prop := ∃ B : ℝ, B = 45 ∨ B = 135
def cond_B : Prop := ∃ C : ℝ, C = 90
def cond_C : Prop := false
def cond_D : Prop := ∃ B : ℝ, 0 < B ∧ B < 60

-- Prove that only cond_A has two possibilities
theorem problem_statement : cond_A ∧ ¬cond_B ∧ ¬cond_C ∧ ¬cond_D :=
by 
  -- Lean proof goes here
  sorry

end problem_statement_l282_282642


namespace original_price_l282_282783

theorem original_price (P : ℕ) (h : (1 / 8) * P = 8) : P = 64 :=
sorry

end original_price_l282_282783


namespace acute_angles_sum_l282_282588

theorem acute_angles_sum
  (α β γ : ℝ)
  (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2)
  (hγ : 0 < γ ∧ γ < π / 2)
  (h_sin_sum : sin α ^ 2 + sin β ^ 2 + sin γ ^ 2 = 1) :
  π / 2 < α + β + γ ∧ α + β + γ < 3 * π / 4 := 
by
  sorry

end acute_angles_sum_l282_282588


namespace point_between_circles_l282_282961

theorem point_between_circles 
  (a b c x1 x2 : ℝ)
  (ellipse_eq : ∀ x y, (x^2 / a^2) + (y^2 / b^2) = 1)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c > 0)
  (quad_eq : a * x1^2 + b * x1 - c = 0)
  (quad_eq2 : a * x2^2 + b * x2 - c = 0)
  (sum_roots : x1 + x2 = -b / a)
  (prod_roots : x1 * x2 = -c / a) :
  1 < x1^2 + x2^2 ∧ x1^2 + x2^2 < 2 :=
sorry

end point_between_circles_l282_282961


namespace ella_model_height_l282_282146

noncomputable def actual_height_meters : ℝ := 60
noncomputable def base_radius_meters : ℝ := 10
noncomputable def actual_base_volume_liters : ℝ := 18850
noncomputable def model_base_volume_liters : ℝ := 0.01885

def volume_ratio : ℝ := actual_base_volume_liters / model_base_volume_liters
def scale_factor : ℝ := volume_ratio^(1/3 : ℝ)
def model_height_meters : ℝ := actual_height_meters / scale_factor
def model_height_cm : ℝ := model_height_meters * 100

theorem ella_model_height : model_height_cm = 60 := by
  sorry

end ella_model_height_l282_282146


namespace Larry_sessions_per_day_eq_2_l282_282358

variable (x : ℝ)
variable (sessions_per_day_time : ℝ)
variable (feeding_time_per_day : ℝ)
variable (total_time_per_day : ℝ)

theorem Larry_sessions_per_day_eq_2
  (h1: sessions_per_day_time = 30 * x)
  (h2: feeding_time_per_day = 12)
  (h3: total_time_per_day = 72) :
  x = 2 := by
  sorry

end Larry_sessions_per_day_eq_2_l282_282358


namespace problem1_problem2_problem3_l282_282636

-- Problem 1
def s_type_sequence (a : ℕ → ℕ) : Prop := 
∀ n ≥ 1, a (n+1) - a n > 3

theorem problem1 (a : ℕ → ℕ) (h₀ : a 1 = 4) (h₁ : a 2 = 8) 
  (h₂ : ∀ n ≥ 2, a n + a (n - 1) = 8 * n - 4) : s_type_sequence a := 
sorry

-- Problem 2
theorem problem2 (a : ℕ → ℕ) (h₀ : ∀ n m, a (n * m) = (a n) ^ m)
  (b : ℕ → ℕ) (h₁ : ∀ n, b n = (3 * a n) / 4)
  (h₂ : s_type_sequence a)
  (h₃ : ¬ s_type_sequence b) : 
  (∀ n, a n = 2^(n+1)) ∨ (∀ n, a n = 2 * 3^(n-1)) ∨ (∀ n, a n = 5^ (n-1)) :=
sorry

-- Problem 3
theorem problem3 (c : ℕ → ℕ) 
  (h₀ : c 2 = 9)
  (h₁ : ∀ n ≥ 2, (1 / n - 1 / (n + 1)) * (2 + 1 / c n) ≤ 1 / c (n - 1) + 1 / c n 
               ∧ 1 / c (n - 1) + 1 / c n ≤ (1 / n - 1 / (n + 1)) * (2 + 1 / c (n-1))) :
  ∃ f : ℕ → ℕ, (s_type_sequence c) ∧ (∀ n, c n = (n + 1)^2) := 
sorry

end problem1_problem2_problem3_l282_282636


namespace ladder_base_length_l282_282475

theorem ladder_base_length {a b c : ℕ} (h1 : c = 13) (h2 : b = 12) (h3 : a^2 + b^2 = c^2) :
  a = 5 := 
by 
  sorry

end ladder_base_length_l282_282475


namespace area_EFGH_l282_282510

structure Point where
  x : ℝ
  y : ℝ

def E := Point.mk 2 1
def F := Point.mk 2 4
def G := Point.mk 7 1
def H := Point.mk 7 (-3)

def trapezoid_area (p1 p2 p3 p4 : Point) : ℝ :=
  let base1 := (p2.y - p1.y).abs
  let base2 := (p4.y - p3.y).abs
  let height := (p3.x - p1.x).abs
  0.5 * (base1 + base2) * height

theorem area_EFGH : trapezoid_area E F G H = 17.5 := by
  sorry

end area_EFGH_l282_282510


namespace equal_distances_sum_of_distances_moving_distances_equal_l282_282974

-- Define the points A, B, origin O, and moving point P
def A : ℝ := -1
def B : ℝ := 3
def O : ℝ := 0

-- Define the moving point P
def P (x : ℝ) : ℝ := x

-- Define the velocities of each point
def vP : ℝ := -1
def vA : ℝ := -5
def vB : ℝ := -20

-- Proof statement ①: Distance from P to A and B are equal implies x = 1
theorem equal_distances (x : ℝ) (h : abs (x + 1) = abs (x - 3)) : x = 1 :=
sorry

-- Proof statement ②: Sum of distances from P to A and B is 5 implies x = -3/2 or 7/2
theorem sum_of_distances (x : ℝ) (h : abs (x + 1) + abs (x - 3) = 5) : x = -3/2 ∨ x = 7/2 :=
sorry

-- Proof statement ③: Moving distances equal at times t = 4/15 or 2/23
theorem moving_distances_equal (t : ℝ) (h : abs (4 * t + 1) = abs (19 * t - 3)) : t = 4/15 ∨ t = 2/23 :=
sorry

end equal_distances_sum_of_distances_moving_distances_equal_l282_282974


namespace justin_additional_time_l282_282355

theorem justin_additional_time (classmates : ℕ) (gathering_hours : ℕ) (minutes_per_flower : ℕ) 
  (flowers_lost : ℕ) : gathering_hours = 2 →
  minutes_per_flower = 10 →
  flowers_lost = 3 →
  classmates = 30 →
  let flowers_gathered := (gathering_hours * 60) / minutes_per_flower in
  let flowers_remaining := flowers_gathered - flowers_lost in
  let flowers_needed := classmates - flowers_remaining in
  let additional_time := flowers_needed * minutes_per_flower in
  additional_time = 210 :=
begin
  intros,
  unfold flowers_gathered flowers_remaining flowers_needed additional_time,
  rw [gathering_hours_eq, minutes_per_flower_eq, flowers_lost_eq, classmates_eq],
  norm_num,
end

end justin_additional_time_l282_282355


namespace imaginary_part_of_conjugate_l282_282485

variable (z : ℂ)
variable (h : (1 + complex.I) * z = 1 - complex.I)

theorem imaginary_part_of_conjugate :
  complex.im (complex.conj z) = 1 :=
  sorry

end imaginary_part_of_conjugate_l282_282485


namespace product_sequence_l282_282119

theorem product_sequence :
  (∏ n in Finset.range (2009 - 4 + 1), (n + 4) / (n + 2)) = 2010 / 4 :=
by
  sorry

example : (2010 / 4) = 502.5 :=
by norm_num

end product_sequence_l282_282119


namespace hyperbola_triangle_area_l282_282366

noncomputable def point_on_hyperbola (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

noncomputable def foci_distance (a : ℝ) : ℝ :=
  2 * real.sqrt (a^2 + 9)

def triangle_area (m : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : ℝ :=
  let |PF₁| := real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) in
  let |PF₂| := real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) in
  0.5 * |PF₁| * |PF₂| * real.sin (real.pi / 3)

theorem hyperbola_triangle_area
  (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ)
  (hP : point_on_hyperbola P.1 P.2)
  (hfoci : F₁ = (-5, 0) ∧ F₂ = (5, 0)) :
  triangle_area 4 P F₁ F₂ = 9 * real.sqrt 3 :=
by
  sorry

end hyperbola_triangle_area_l282_282366


namespace angle_between_plane_and_base_l282_282085

noncomputable def rhombus_base := sorry -- Define the base as a rhombus with one acute angle of 60 degrees
noncomputable def intersecting_plane := sorry -- Define the plane intersecting the prism forming a square cross section

theorem angle_between_plane_and_base (rhombus_base : Type) (intersecting_plane : Type) : 
    ∃ α : ℝ, α = 54.7 :=
by
  have rhombus_base_property : (acute_angle rhombus_base = 60) := sorry
  have intersecting_plane_property : (forms_square_with inter_plane rhombus_base) := sorry
  exists 54.7
  sorry

end angle_between_plane_and_base_l282_282085


namespace circle_radius_squared_84_l282_282821

-- Definitions and conditions
variables (r : ℝ)
variables (A B C D P : ℝ)
variables (AB CD : ℝ)
variables (BP : ℝ)
variables (angleAPD : ℝ)

-- Given conditions
def conditions := AB = 12 ∧ CD = 8 ∧ BP = 10 ∧ angleAPD = 60 ∧
-- Circle with center O radius r:
∃ O : ℝ, dist O A = r ∧ dist O B = r ∧ dist O C = r ∧ dist O D = r ∧
-- Intersecting point P (extended chords):
(P = some_point)

-- Proof statement
theorem circle_radius_squared_84 (h : conditions r A B C D P AB CD BP angleAPD): r^2 = 84 :=
sorry

end circle_radius_squared_84_l282_282821


namespace sum_of_coordinates_of_S_l282_282247

-- Definitions of the vertices as given in the conditions
def P : (ℝ × ℝ) := (-3, -2)
def R : (ℝ × ℝ) := (9, 1)
def Q : (ℝ × ℝ) := (2, -5)

-- Goal: Find the sum of the coordinates of vertex S
theorem sum_of_coordinates_of_S :
  ∃ S : (ℝ × ℝ), (let (x, y) := S in x + y = 8) ∧ 
  (let (px, py) := P in let (qx, qy) := Q in let (rx, ry) := R in
    2 * 3 = px + rx ∧ 2 * (-0.5) = py + ry ∧
    2 * 3 = qx + x ∧ 2 * (-0.5) = qy + y) :=
sorry

end sum_of_coordinates_of_S_l282_282247


namespace seq_integer_l282_282966

theorem seq_integer (a : ℕ → ℤ) (h1 : a 1 = 1) (h2 : a 2 = 1) (h3 : a 3 = 249)
(h_rec : ∀ n, a (n + 3) = (1991 + a (n + 2) * a (n + 1)) / a n) :
∀ n, ∃ b : ℤ, a n = b :=
by
  sorry

end seq_integer_l282_282966


namespace area_quadrilateral_ADEC_l282_282663

open Real

def isMidpoint (D A B : Point) := dist A D = dist D B

theorem area_quadrilateral_ADEC 
  (A B C D E : Point) 
  (hC : ∠C = 90) 
  (hMidpoint : isMidpoint D A B) 
  (hPerpendicular : ∠DE = 90) 
  (hAB : dist A B = 24) 
  (hAC : dist A C = 15): 
  area A D E C = 82.9023 := 
by
  sorry

end area_quadrilateral_ADEC_l282_282663


namespace unique_k_l282_282114

def is_prime (n : ℕ) : Prop := Nat.Prime n

def is_root (p q : ℕ) (k : ℕ) : Prop :=
  x^2 - 75x + k = (x - p) * (x - q)

theorem unique_k (p q k : ℕ) (hp : is_prime p) (hq : is_prime q)
    (h_eq : p + q = 75) (h_k : k = p * q) : k = 146 :=
by
  sorry

end unique_k_l282_282114


namespace simplify_expression1_simplify_expression2_l282_282716

variable {x y : ℝ} -- Declare x and y as real numbers

theorem simplify_expression1 :
  3 * x^2 - (7 * x - (4 * x - 3) - 2 * x^2) = 5 * x^2 - 3 * x - 3 :=
sorry

theorem simplify_expression2 :
  3 * x^2 * y - (2 * x * y - 2 * (x * y - (3/2) * x^2 * y) + x^2 * y^2) = - x^2 * y^2 :=
sorry

end simplify_expression1_simplify_expression2_l282_282716


namespace matrix_solution_l282_282486

theorem matrix_solution
    (p q : ℝ)
    (hR : let R := Matrix![[p, q], [-4/7, -3/7]]
          in R * R = (-1 : ℝ) • (1 : Matrix (Fin 2) (Fin 2) ℝ)) :
    (p, q) = (3/7, -7/4) :=
sorry

end matrix_solution_l282_282486


namespace jennifer_initial_oranges_l282_282344

theorem jennifer_initial_oranges (O : ℕ) : 
  ∀ (pears apples remaining_fruits : ℕ),
    pears = 10 →
    apples = 2 * pears →
    remaining_fruits = pears - 2 + apples - 2 + O - 2 →
    remaining_fruits = 44 →
    O = 20 :=
by
  intros pears apples remaining_fruits h1 h2 h3 h4
  sorry

end jennifer_initial_oranges_l282_282344


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282285

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282285


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282279

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282279


namespace section_Diligence_students_before_transfer_l282_282995

-- Define the variables
variables (D_after I_after D_before : ℕ)

-- Problem Statement
theorem section_Diligence_students_before_transfer :
  ∀ (D_after I_after: ℕ),
    2 + D_after = I_after
    ∧ D_after + I_after = 50 →
    ∃ D_before, D_before = D_after - 2 ∧ D_before = 23 :=
by
sorrry

end section_Diligence_students_before_transfer_l282_282995


namespace find_x_l282_282036

theorem find_x :
  ∃ x : ℝ, 8 * x - (5 * 0.85 / 2.5) = 5.5 ∧ x = 0.9 :=
begin
  use 0.9,
  split,
  { calc
    8 * 0.9 - (5 * 0.85 / 2.5) = 7.2 - (5 * 0.85 / 2.5) : by norm_num
    ... = 7.2 - 1.7 : by norm_num
    ... = 5.5 : by norm_num },
  { refl }
end

end find_x_l282_282036


namespace option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l282_282465

theorem option_A_incorrect : ¬(∃ (s: set ℕ), s = {1} ∧ {1} ∈ {1, 3}) := by 
  sorry

theorem option_B_incorrect : ¬(∃ (x: ℕ), ∀ (s: set ℕ), x = 1 → x ⊆ {1, 2}) := by 
  sorry

theorem option_C_incorrect : ¬(∃ (s: set ℕ), s = ∅ ∧ ∅ ∈ {0}) := by 
  sorry

theorem option_D_correct : ∀ (s: set ℕ), s = ∅ → s ⊆ ∅ :=
  by intro h; rw [h]; exact set.subset.refl ∅

end option_A_incorrect_option_B_incorrect_option_C_incorrect_option_D_correct_l282_282465


namespace ab_op_eq_l282_282527

theorem ab_op_eq :
  ∀ a b : ℚ, a * b = (1/a) + (1/b) →
             a - b = 9 →
             a / b = 20 →
             (1/a + 1/b) = 19 / 60 :=
by
  intros a b h1 h2 h3
  have : a = 20 * b,
    from eq_div_iff_mul_eq.mp h3,
  have : a - b = 9,
    from h2,
  sorry

end ab_op_eq_l282_282527


namespace solution_set_inequality_l282_282421

theorem solution_set_inequality (x : ℝ) : (1 - x) * (2 + x) < 0 ↔ x < -2 ∨ x > 1 :=
by
  -- Proof omitted
  sorry

end solution_set_inequality_l282_282421


namespace smallest_sum_60_l282_282546

noncomputable def primes : List ℕ := [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def four_consecutive_primes_sum_divisible_by_5 (p1 p2 p3 p4 : ℕ) : Prop :=
  List.nth primes n1 = p1 ∧ List.nth primes (n1+1) = p2 ∧ List.nth primes (n1+2) = p3 ∧ List.nth primes (n1+3) = p4 ∧
  (p1 + p2 + p3 + p4) % 5 = 0

theorem smallest_sum_60 : ∃ p1 p2 p3 p4,
  (p1 = 11) ∧ (p2 = 13) ∧ (p3 = 17) ∧ (p4 = 19) ∧
  four_consecutive_primes_sum_divisible_by_5 p1 p2 p3 p4 ∧
  (∀ (q1 q2 q3 q4 : ℕ), 
    four_consecutive_primes_sum_divisible_by_5 q1 q2 q3 q4 → 
    p1 + p2 + p3 + p4 ≤ q1 + q2 + q3 + q4) :=
by 
  sorry

end smallest_sum_60_l282_282546


namespace greatest_possible_gcd_3Hn_n_plus_1_l282_282200

theorem greatest_possible_gcd_3Hn_n_plus_1 (n : ℕ) (h : n > 0) :
  let H_n := 2 * n^2 - n in gcd (3 * H_n) (n + 1) ≤ 12 :=
by
  let H_n := 2 * n^2 - n
  have h₁ : H_n = 2 * n^2 - n := rfl
  have h₂ : 3 * H_n = 6 * n^2 - 3 * n := by simp [h₁]
  have h₃ : gcd (6 * n^2 - 3 * n) (n + 1) = gcd (12, n + 1) :=
    by sorry  -- arithmetic and gcd properties
  have h₄ : gcd (12, n + 1) ≤ 12 := by simp [gcd_le_right, gcd_le_left]
  exact h₄

end greatest_possible_gcd_3Hn_n_plus_1_l282_282200


namespace periods_of_multiples_l282_282714

variable {α : Type*} {β : Type*} [Add α] [HasSmul ℕ α] {f : α → β}

-- The condition that l is a period of f
def is_period (f : α → β) (l : α) : Prop :=
  ∀ x, f (x + l) = f x

-- The theorem stating that nl is a period for any natural number n
theorem periods_of_multiples (f : α → β) (l : α) (h : is_period f l) (n : ℕ) :
  is_period f (n • l) :=
by
  intro x
  induction n with k ih
  case zero => simp [is_period] -- Base case n = 0
  case succ => 
    rw [Nat.succ_eq_add_one, add_smul, add_assoc]
    exact eq.trans (h (x + (k • l))) (ih x)

end periods_of_multiples_l282_282714


namespace square_ratio_in_triangles_is_12_over_13_l282_282844

theorem square_ratio_in_triangles_is_12_over_13 :
  ∀ (x y : ℝ), 
    (∃ (triangle1 triangle2 : right_triangle),
     triangle1.hypotenuse = 13 ∧ triangle1.legs = (5, 12) ∧
     triangle2.hypotenuse = 13 ∧ triangle2.legs = (5, 12) ∧
     (∃ (sq1 sq2 : inscribed_square),
        sq1.side = x ∧ sq1.position = .vertex_on_right_angle_vertex ∧
        sq2.side = y ∧ sq2.position = .side_on_hypotenuse)) → 
    x / y = 12 / 13 :=
by
  sorry

end square_ratio_in_triangles_is_12_over_13_l282_282844


namespace correct_statements_about_regression_line_l282_282462

open Real

-- Define the conditions
def mean_x : ℝ := (2 + 4 + 6 + 8 + 10) / 5
def mean_y : ℝ := (17 + 16 + 14 + 13 + 11) / 5
def regression_line_coeff : ℝ := (mean_y - 18.7) / mean_x

-- The main proof problem
theorem correct_statements_about_regression_line :
  (mean_x = 6) ∧ (mean_y = 14.2) ∧ (regression_line_coeff = -0.75) →
  ∃ b : ℝ, (mean_y = b * mean_x + 18.7) ∧ (b < 0) :=
by
  sorry

end correct_statements_about_regression_line_l282_282462


namespace M_inter_N_is_1_to_2_l282_282612

-- Define the sets M and N
noncomputable def M : Set (ℝ × ℝ) := { p | ∃ x, p = (x, 2*x) ∧ x > 0 }
noncomputable def N : Set ℝ := { x | ∃ y, y = log(2*x - x^2) ∧ 0 < x ∧ x < 2 }

-- The statement we aim to prove
theorem M_inter_N_is_1_to_2 : (M ∩ N) = { x | 1 < x ∧ x < 2 } := 
sorry

end M_inter_N_is_1_to_2_l282_282612


namespace justin_additional_time_l282_282357

theorem justin_additional_time (classmates : ℕ) (gathering_hours : ℕ) (minutes_per_flower : ℕ) 
  (flowers_lost : ℕ) : gathering_hours = 2 →
  minutes_per_flower = 10 →
  flowers_lost = 3 →
  classmates = 30 →
  let flowers_gathered := (gathering_hours * 60) / minutes_per_flower in
  let flowers_remaining := flowers_gathered - flowers_lost in
  let flowers_needed := classmates - flowers_remaining in
  let additional_time := flowers_needed * minutes_per_flower in
  additional_time = 210 :=
begin
  intros,
  unfold flowers_gathered flowers_remaining flowers_needed additional_time,
  rw [gathering_hours_eq, minutes_per_flower_eq, flowers_lost_eq, classmates_eq],
  norm_num,
end

end justin_additional_time_l282_282357


namespace solve_floor_equation_l282_282541

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end solve_floor_equation_l282_282541


namespace digit_at_2009th_position_l282_282878

theorem digit_at_2009th_position : 
  let seq := (List.range (2009 * 3)).join.map (fun n => toString (n + 1)).mkString in
  seq[2009 - 1] = '0' := 
by
  sorry

end digit_at_2009th_position_l282_282878


namespace pairs_equality_check_l282_282103

theorem pairs_equality_check :
  (\frac{3^2}{4} ≠ (\frac{3}{4})^2) ∧ 
  (-1^2013 = (-1)^2025) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-\frac{2^2}{3} ≠ \frac{(-2)^2}{3}) :=
by {
  sorry, -- skipping proof steps
}

end pairs_equality_check_l282_282103


namespace smallest_possible_other_integer_l282_282750

theorem smallest_possible_other_integer (x m n : ℕ) (h1 : x > 0) (h2 : m = 70) 
  (h3 : gcd m n = x + 7) (h4 : lcm m n = x * (x + 7)) : n = 20 :=
sorry

end smallest_possible_other_integer_l282_282750


namespace minimum_construction_cost_l282_282891

noncomputable def length (v d : ℝ) : ℝ :=
  x

noncomputable def width (v d : ℝ) : ℝ :=
  v / (d * x)

noncomputable def cost_function (x : ℝ) : ℝ :=
  4 * 120 + 4 * x * 80 + (16 / x) * 80

theorem minimum_construction_cost :
  let v := 8
  let d := 2
  let x := 2
  (2 * v / d) = 4 →
  cost_function x = 1760 :=
by
  intros v d x h
  rw cost_function
  sorry

end minimum_construction_cost_l282_282891


namespace traditionalist_fraction_l282_282073

variable (P T : ℕ)
variable (h1 : ∀(i : ℕ), i < 15 → number_of_traditionalists_in_province i = T)
variable (h2 : T = P / 20)
variable (country_traditionalist_fraction : ℚ)

theorem traditionalist_fraction (h1 : ∀(i : ℕ), i < 15 → number_of_traditionalists_in_province i = T)
    (h2 : T = P / 20) : country_traditionalist_fraction = 3 / 7 := by
  sorry

end traditionalist_fraction_l282_282073


namespace min_max_values_f_l282_282942

def f (x : ℝ) : ℝ := 4^(x - 1/2) - 3 * 2^x + 5

theorem min_max_values_f : ∃ (xmin xmax : ℝ), ∀ x ∈ Icc (0 : ℝ) 2, f x ≥ xmin ∧ f x ≤ xmax ∧ xmin = 1/2 ∧ xmax = 5/2 :=
by
  sorry

end min_max_values_f_l282_282942


namespace find_weight_of_a_l282_282471

-- Define the weights
variables (a b c d e : ℝ)

-- Given conditions
def condition1 := (a + b + c) / 3 = 50
def condition2 := (a + b + c + d) / 4 = 53
def condition3 := (b + c + d + e) / 4 = 51
def condition4 := e = d + 3

-- Proof goal
theorem find_weight_of_a : condition1 a b c → condition2 a b c d → condition3 b c d e → condition4 d e → a = 73 :=
by
  intros h1 h2 h3 h4
  sorry

end find_weight_of_a_l282_282471


namespace log_x_y_eq_pi_l282_282482

theorem log_x_y_eq_pi {x y : ℝ} (pos_x : 0 < x) (pos_y : 0 < y) :
  let r := log 2 (x^3)
  let C := log 2 (y^6)
  let circle_circum := 2 * pi * r
  C = circle_circum
  → log x y = pi := 
by
  intros r C circle_circum H
  rw [log_pow, mul_comm 3, log_pow] at *
  sorry

end log_x_y_eq_pi_l282_282482


namespace determine_n_l282_282634

theorem determine_n (n : ℝ) (h1 : n > 0) (h2 : sqrt (n^2 + n^2 + n^2 + n^2) = 64) : n = 32 :=
by
  -- Proof goes here
  -- At this point, we are only providing the statement, not the complete proof
  sorry

end determine_n_l282_282634


namespace smallest_m_exists_l282_282556

theorem smallest_m_exists :
  ∃ m : ℕ, (∃ k₁ : ℕ, m = 2 ^ 35 * 3 ^ 35 * 5 ^ 84 * 7 ^ 90 ∧ 5 * m = k₁ ^ 5) ∧
           (∃ k₂ : ℕ, 6 * m = k₂ ^ 6) ∧
           (∃ k₃ : ℕ, 7 * m = k₃ ^ 7) :=
by {
  use 2 ^ 35 * 3 ^ 35 * 5 ^ 84 * 7 ^ 90,
  split,
  { use (5 * (2 ^ 35 * 3 ^ 35 * 5 ^ 83 * 7 ^ 90))^{1/5}
    sorry },
  split,
  { use (6 * (2 ^ 34 * 3 ^ 34 * 5 ^ 84 * 7 ^ 90))^{1/6}
    sorry },
  { use (7 * (2 ^ 35 * 3 ^ 35 * 5 ^ 84 * 7 ^ 89))^{1/7}
    sorry }
}

end smallest_m_exists_l282_282556


namespace union_eq_intersection_eq_l282_282971

open Set

variable (x : ℝ)

def U := Set.univ

def A := { x : ℝ | -1 < x ∧ x < 3 }

def B := { x : ℝ | 1 ≤ x ∧ x < 4 }

theorem union_eq : A ∪ B = { x : ℝ | -1 < x ∧ x < 4 } :=
by
  sorry

theorem intersection_eq : A ∩ B = { x : ℝ | 1 ≤ x ∧ x < 3 } :=
by
  sorry

end union_eq_intersection_eq_l282_282971


namespace pentagon_area_calculation_l282_282474

def area_of_pentagon
  (radius_F : ℝ) (radius_G : ℝ)
  (side_square : ℝ) (side_triangle : ℝ)
  (angleOMB : ℝ) (sin_angle_OMB : ℝ) : Prop :=
  radius_F = 10 ∧ radius_G = 4 ∧
  let side_square := 10 * Real.sqrt 2 in
  let area_square := side_square^2 in
  let side_triangle := 4 * Real.sqrt 3 in
  let area_triangle := (Real.sqrt 3 / 4) * side_triangle^2 in
  let area_OMB := 1 / 2 * side_square * side_triangle * sin_angle_OMB in
  angleOMB = 105 ∧ sin_angle_OMB = (Real.sqrt 6 + Real.sqrt 2) / 4 ∧
  let total_area := area_square + area_triangle + 2 * area_OMB in
  total_area = 260 + 32 * Real.sqrt 3

theorem pentagon_area_calculation : ∃ a b c, 
  c = 3 ∧ a = 260 ∧ b = 32 ∧
  ∀ radius_F radius_G side_square side_triangle angleOMB sin_angle_OMB,
  area_of_pentagon radius_F radius_G side_square side_triangle angleOMB sin_angle_OMB →
  a + b + c = 295 :=
begin
  sorry
end

end pentagon_area_calculation_l282_282474


namespace ceil_floor_sum_l282_282148

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end ceil_floor_sum_l282_282148


namespace probability_sum_less_than_16_l282_282436

-- The number of possible outcomes when three six-sided dice are rolled
def total_outcomes : ℕ := 6 * 6 * 6

-- The number of favorable outcomes where the sum of the dice is less than 16
def favorable_outcomes : ℕ := (6 * 6 * 6) - (3 + 3 + 3 + 1)

-- The probability that the sum of the dice is less than 16
def probability_less_than_16 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_less_than_16 : probability_less_than_16 = 103 / 108 := 
by sorry

end probability_sum_less_than_16_l282_282436


namespace solve_abc_l282_282939

theorem solve_abc (a b c : ℕ) (h1 : a > b ∧ b > c) 
  (h2 : 34 - 6 * (a + b + c) + (a * b + b * c + c * a) = 0) 
  (h3 : 79 - 9 * (a + b + c) + (a * b + b * c + c * a) = 0) : 
  a = 10 ∧ b = 3 ∧ c = 2 := 
by
  sorry

end solve_abc_l282_282939


namespace partition_square_into_smaller_squares_partition_cube_into_smaller_cubes_l282_282400

theorem partition_square_into_smaller_squares (n : ℕ) (h : n ≥ 6) :
  ∃ (squares : fin n → set (ℝ × ℝ)), (∀ i, ∃ a b, squares i = set.Icc a (a + b) ×ˢ set.Icc b (b + a)) :=
sorry

theorem partition_cube_into_smaller_cubes (d n : ℕ) (N : ℕ) (h : n ≥ N) :
  ∃ (cubes : fin n → set (fin d → ℝ)), (∀ i, ∃ a b, cubes i = set.Icc (λ _, a) (λ _, a + b)) :=
sorry

end partition_square_into_smaller_squares_partition_cube_into_smaller_cubes_l282_282400


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282286

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282286


namespace find_x_plus_y_l282_282982

theorem find_x_plus_y (x y : ℚ) (h1 : |x| + x + y = 10) (h2 : x + |y| - y = 12) : x + y = 18 / 5 := 
by
  sorry

end find_x_plus_y_l282_282982


namespace sqrt_inequality_l282_282303

theorem sqrt_inequality (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
  sorry

end sqrt_inequality_l282_282303


namespace store_A_cheaper_than_store_B_l282_282781

noncomputable def store_A_full_price : ℝ := 125
noncomputable def store_A_discount_pct : ℝ := 0.08
noncomputable def store_B_full_price : ℝ := 130
noncomputable def store_B_discount_pct : ℝ := 0.10

noncomputable def final_price_A : ℝ :=
  store_A_full_price * (1 - store_A_discount_pct)

noncomputable def final_price_B : ℝ :=
  store_B_full_price * (1 - store_B_discount_pct)

theorem store_A_cheaper_than_store_B :
  final_price_B - final_price_A = 2 :=
by
  sorry

end store_A_cheaper_than_store_B_l282_282781


namespace sum_of_squares_of_roots_l282_282557

theorem sum_of_squares_of_roots :
  (∃ r1 r2 : ℝ, (r1 + r2 = 10 ∧ r1 * r2 = 16) ∧ (r1^2 + r2^2 = 68)) :=
by
  sorry

end sum_of_squares_of_roots_l282_282557


namespace problem_solution_l282_282159

theorem problem_solution (x : ℝ) (h1 : x > 9) 
(h2 : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x ∈ Set.Ici 18 := sorry

end problem_solution_l282_282159


namespace sum_geometric_sequence_first_eight_terms_l282_282187

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l282_282187


namespace diameter_of_circumscribed_circle_l282_282299

noncomputable def circumscribed_circle_diameter (a : ℝ) (A : ℝ) : ℝ :=
  a / Real.sin A

theorem diameter_of_circumscribed_circle :
  circumscribed_circle_diameter 15 (Real.pi / 4) = 15 * Real.sqrt 2 :=
by
  sorry

end diameter_of_circumscribed_circle_l282_282299


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282272

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282272


namespace train_speed_l282_282497

noncomputable def train_length : ℝ := 2500
noncomputable def time_to_cross_pole : ℝ := 35

noncomputable def speed_in_kmph (distance : ℝ) (time : ℝ) : ℝ :=
  (distance / time) * 3.6

theorem train_speed :
  speed_in_kmph train_length time_to_cross_pole = 257.14 := by
  sorry

end train_speed_l282_282497


namespace smallest_number_divisible_by_9_with_conditions_l282_282458

theorem smallest_number_divisible_by_9_with_conditions :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 
           (∃ a b c d : ℕ, n = 1000*a + 100*b + 10*c + d ∧ 
                           a ≠ 0 ∧ 
                           a ≥ 2 ∧ 
                           (a % 2 = 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0 ∧ d % 2 ≠ 0 ∨ 
                            a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0 ∧ d % 2 ≠ 0 ∨ 
                            a % 2 = 0 ∧ b % 2 ≠ 0 ∧ c % 2 ≠ 0 ∧ d % 2 = 0 ∨ 
                            a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 = 0 ∧ d % 2 ≠ 0 ∨ 
                            a % 2 ≠ 0 ∧ b % 2 = 0 ∧ c % 2 ≠ 0 ∧ d % 2 = 0 ∨ 
                            a % 2 ≠ 0 ∧ b % 2 ≠ 0 ∧ c % 2 = 0 ∧ d % 2 = 0) ∧ 
                           (a + b + c + d) % 9 = 0) ∧ n = 2493 :=
begin
  use 2493,
  split,
  { norm_num, }, -- Proving 1000 ≤ 2493 < 10000
  split,
  { norm_num, }, -- Secondary split might be an overkill
  use [2, 4, 9, 3],
  split,
  { norm_num, }, -- 2493 = 1000*2 + 100*4 + 10*9 + 3
  split,
  { norm_num, }, -- 2 ≠ 0 by norm, no need for advanced rules
  split,
  { norm_num, }, -- and so on.
  split,
  { norm_num, }, -- All splits inspecting modulo equality.
  sorry -- Remaining hid. Hiking/horizontal enumeration omitted, obviously
end

end smallest_number_divisible_by_9_with_conditions_l282_282458


namespace complex_magnitude_l282_282536

theorem complex_magnitude (z : ℂ) (h : z = 3 - 4 * complex.I) : complex.abs z = 5 :=
by
  -- sorry to skip the proof
  sorry

end complex_magnitude_l282_282536


namespace find_base_a_l282_282589

noncomputable def log_base (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem find_base_a (a : ℝ) : 
  (∀ (P : ℝ × ℝ), P = (x, log_base a x) → ∃ x > 0, x - log_base a x + 1 = 2) → 
  (0 < a) → 
  (∃ x, (deriv (λ x : ℝ, x - log_base a x + 1) x = 0 ∧ 
         is_min (λ x : ℝ, x - log_base a x + 1) x)) → 
  a = ℯ :=
by 
  -- conditions and definitions translation
  intros _ _ _ 
  sorry

end find_base_a_l282_282589


namespace determinant_transformed_matrix_l282_282364

variables {R : Type*} [Field R]
variables (a b c : Fin 3 → R)

def determinant (a b c : Fin 3 → R) : R :=
a 0 * (b 1 * c 2 - c 1 * b 2) -
a 1 * (b 0 * c 2 - c 0 * b 2) +
a 2 * (b 0 * c 1 - c 0 * b 1)

theorem determinant_transformed_matrix :
  let D := determinant a b c in
  determinant (fun i => 2 * a i + 3 * b i) (fun i => 3 * b i + 4 * c i) (fun i => 4 * c i + 2 * a i) = 24 * D :=
by
  sorry

end determinant_transformed_matrix_l282_282364


namespace find_polar_coordinates_of_point_P_l282_282915

def cartesian_to_polar (x y : ℝ) : ℝ × ℝ :=
  let r := Real.sqrt (x^2 + y^2)
  let θ := Real.arctan2 y x
  (r, θ)

def polar_coordinates_of_point_P : Prop :=
  cartesian_to_polar 1 (-Real.sqrt 3) = (2, 5 * Real.pi / 3)

theorem find_polar_coordinates_of_point_P :
  polar_coordinates_of_point_P :=
  by
    sorry

end find_polar_coordinates_of_point_P_l282_282915


namespace sqrt_meaningful_range_l282_282306

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 3)) → x ≥ 3 := sorry

end sqrt_meaningful_range_l282_282306


namespace circle_intersection_area_l282_282517

/-- Circles A, B, and C each have radius 2. 
Circles A and B share one point of tangency. 
Circle C intersects circle A at one of the endpoints of the line segment between their centers.
Circle C is tangent to the midpoint of the line segment between the centers of circles A and B.
Prove that the area inside circle C but outside circles A and B is (10 * π) / 3 + √3. -/
theorem circle_intersection_area :
  let r := 2
  let area_C := π * r^2
  let overlap_area := (π / 3 - (√3 / 2)) * 2
  area_C - overlap_area = (10 * π) / 3 + √3 := by
  sorry

end circle_intersection_area_l282_282517


namespace calculate_area_of_triangle_tangent_line_l282_282116

noncomputable def areaOfTriangleTangent := 
  let f : ℝ → ℝ := λ x, Real.exp (-x)
  let M := (1 : ℝ, Real.exp (-1))
  let tangentLine := λ x, -Real.exp (-1) * x + 2 * Real.exp (-1)
  let A := (2 : ℝ, 0)
  let B := (0 : ℝ, 2 * Real.exp (-1))
  let C := (0 : ℝ, 0)
  let base := 2 - 0
  let height := 2 * Real.exp (-1)
  (1 / 2) * base * height

theorem calculate_area_of_triangle_tangent_line :
  areaOfTriangleTangent = 2 / Real.exp 1 :=
by
  sorry

end calculate_area_of_triangle_tangent_line_l282_282116


namespace height_of_triangle_l282_282956

theorem height_of_triangle (α β γ c : ℝ)
  (h_triangle : α + β + γ = π) :
  height m_c corresponding to side c in a triangle given angles α, β, γ is equal to
  m_c = (c * sin α * sin β) / (sin γ)
:= by
  -- Here we would provide the proof steps to show the equivalence.
  sorry

end height_of_triangle_l282_282956


namespace probability_standard_weight_l282_282820

noncomputable def total_students : ℕ := 500
noncomputable def standard_students : ℕ := 350

theorem probability_standard_weight : (standard_students : ℚ) / (total_students : ℚ) = 7 / 10 :=
by {
  sorry
}

end probability_standard_weight_l282_282820


namespace maximum_value_a_plus_b_plus_c_plus_d_maximum_value_ab_plus_cd_l282_282379

theorem maximum_value_a_plus_b_plus_c_plus_d (a_1 a_2 : ℝ) (a : Fin 40 → ℝ) 
  (h₁ : ∑ i, a i = 0)
  (h₂ : ∀ i : Fin 40, |a i - a ((i + 1) % 40)| ≤ 1) : 
  (a 9 + a 19 + a 29 + a 39) ≤ 10 :=
  sorry

theorem maximum_value_ab_plus_cd (a_1 a_2 : ℝ) (a : Fin 40 → ℝ) 
  (h₁ : ∑ i, a i = 0)
  (h₂ : ∀ i : Fin 40, |a i - a ((i + 1) % 40)| ≤ 1) : 
  (a 9 * a 19 + a 29 * a 39) ≤ (425 / 8) :=
  sorry

end maximum_value_a_plus_b_plus_c_plus_d_maximum_value_ab_plus_cd_l282_282379


namespace coeff_x4_in_expansion_l282_282137

theorem coeff_x4_in_expansion : 
  let a := (fun x => x^3 / 3)
  let b := (fun x => -3 / x^2)
  let expansion := (fun x => (a x + b x)^9)
  ∀ (x : ℚ), \\ compute the coefficient of x^4 in the expansion*(expansion x) = 0 := 
by 
  sorry

end coeff_x4_in_expansion_l282_282137


namespace pascal_triangle_row51_sum_l282_282033

theorem pascal_triangle_row51_sum : (Nat.choose 51 4) + (Nat.choose 51 6) = 18249360 :=
by
  sorry

end pascal_triangle_row51_sum_l282_282033


namespace cosine_of_angle_between_vectors_l282_282975

-- Define the vectors a and b
def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (5, 12)

-- Function to calculate the dot product of two vectors
def dot_product (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

-- Function to calculate the norm of a vector
def norm (u : ℝ × ℝ) : ℝ :=
  Real.sqrt (u.1^2 + u.2^2)

-- The desired statement to prove
theorem cosine_of_angle_between_vectors :
  (dot_product a b) / (norm a * norm b) = 63 / 65 :=
by
  sorry

end cosine_of_angle_between_vectors_l282_282975


namespace minimum_cost_to_guess_number_l282_282834

-- Define the modified Fibonacci sequence
def modified_fibonacci (n : ℕ) : ℕ :=
  if n = 1 then 2
  else if n = 2 then 3
  else modified_fibonacci (n - 1) + modified_fibonacci (n - 2)

noncomputable def min_cost_guarantee := 11

theorem minimum_cost_to_guess_number :
  (∃ S : set (fin 145), ∀ n ∈ (finset.range 145).val, 2 * card {s ∈ S | n ∈ s} + card {s ∈ S | n ∉ s} + card S = min_cost_guarantee) :=
sorry

end minimum_cost_to_guess_number_l282_282834


namespace sqrt_x_minus_3_domain_l282_282310

theorem sqrt_x_minus_3_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_domain_l282_282310


namespace jimmy_points_lost_for_bad_behavior_l282_282346

theorem jimmy_points_lost_for_bad_behavior (points_per_exam : ℕ) (num_exams : ℕ) (points_needed : ℕ)
  (extra_points_allowed : ℕ) (total_points_earned : ℕ) (current_points : ℕ)
  (h1 : points_per_exam = 20) (h2 : num_exams = 3) (h3 : points_needed = 50)
  (h4 : extra_points_allowed = 5) (h5 : total_points_earned = points_per_exam * num_exams)
  (h6 : current_points = points_needed + extra_points_allowed) :
  total_points_earned - current_points = 5 :=
by
  sorry

end jimmy_points_lost_for_bad_behavior_l282_282346


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282264

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282264


namespace find_largest_number_l282_282008

noncomputable def largest_of_three_numbers (x y z : ℝ) : ℝ :=
  if x ≥ y ∧ x ≥ z then x
  else if y ≥ x ∧ y ≥ z then y
  else z

theorem find_largest_number (x y z : ℝ) (h1 : x + y + z = 3) (h2 : xy + xz + yz = -11) (h3 : xyz = 15) :
  largest_of_three_numbers x y z = Real.sqrt 5 := by
  sorry

end find_largest_number_l282_282008


namespace harmonic_property_l282_282202

-- Define the harmonic number function
noncomputable def h : ℕ → ℚ
| 0     := 0  -- although not needed in the actual problem, we define h(0) for completeness
| (n+1) := h n + 1 / (n + 1)

-- The mathematical problem rewritten in Lean 4
theorem harmonic_property (n : ℕ) (hn : 2 ≤ n) : 
  (n : ℚ) + (∑ i in finset.range (n), h (i + 1)) = (n : ℚ) * h n :=
sorry

end harmonic_property_l282_282202


namespace convert_cylindrical_to_rectangular_l282_282895

noncomputable def cylindrical_to_rectangular (r θ z : ℝ) : ℝ × ℝ × ℝ :=
  (r * Real.cos θ, r * Real.sin θ, z)

theorem convert_cylindrical_to_rectangular :
  cylindrical_to_rectangular 7 (5 * Real.pi / 4) (-3) = (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, -3) :=
by
  sorry

end convert_cylindrical_to_rectangular_l282_282895


namespace circles_intersect_l282_282140

theorem circles_intersect (
  C1 : ∀ (x y : ℝ), x^2 + y^2 - 4 * x + 6 * y = 0,
  C2 : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 6 * y - 26 = 0
) : ∃ p : ℝ × ℝ, C1 p.1 p.2 ∧ C2 p.1 p.2 := sorry

end circles_intersect_l282_282140


namespace intersection_sets_l282_282585

theorem intersection_sets :
  let M := {2, 3, 4, 5}
  let N := {3, 4, 5}
  M ∩ N = {3, 4, 5} :=
by
  sorry

end intersection_sets_l282_282585


namespace sum_of_sheets_at_least_four_power_m_l282_282785

theorem sum_of_sheets_at_least_four_power_m 
  (m : ℕ) : 
  let n := 2^m in 
  let steps := m * 2^(m-1) in
  let init_sheets : List ℕ := List.replicate n 1 in
  let final_sheets : List ℕ := 
    (List.range steps).foldl (λ sheets _, 
      let (a, b) := sheets.get_two_distinct_indices in
      let ab := sheets.nth_le a + sheets.nth_le b in
      sheets.modify a ab . modify b ab
    ) init_sheets 
  in 
  final_sheets.sum ≥ 4^m :=
sorry

end sum_of_sheets_at_least_four_power_m_l282_282785


namespace baker_cakes_l282_282506

theorem baker_cakes (a b c : ℕ) (h_a : a = 121) (h_b : b = 105) (h_c : c = 170) : a - b + c = 186 := 
by
  rw [h_a, h_b, h_c]
  norm_num

end baker_cakes_l282_282506


namespace sqrt_expression_bound_l282_282361

theorem sqrt_expression_bound
  (a b c d : ℝ)
  (ha : 0 ≤ a ∧ a ≤ 1)
  (hb : 0 ≤ b ∧ b ≤ 1)
  (hc : 0 ≤ c ∧ c ≤ 1)
  (hd : 0 ≤ d ∧ d ≤ 2) :
  (\sqrt (a^3 + (2 - b)^3) + \sqrt (b^3 + (2 - c)^3) + \sqrt (c^3 + (2 - d)^3) + \sqrt (d^3 + (3 - a)^3)) ≤ 5 + 2 * \sqrt 2 :=
sorry

end sqrt_expression_bound_l282_282361


namespace smallest_n_for_cool_matrix_l282_282371

def isCoolMatrix (A : Matrix (Fin 2005) (Fin 2005) (Fin (n + 1))) : Prop :=
  ∀ i j : Fin 2005, ∀ k l : Fin 2005, i ≠ k ∨ j ≠ l → 
  (Set.ofFinset (Matrix.row A i) ≠ Set.ofFinset (Matrix.row A k) ∧ 
   Set.ofFinset (Matrix.col A j) ≠ Set.ofFinset (Matrix.col A l))

theorem smallest_n_for_cool_matrix : ∃ (n : ℕ), n ≥ 13 ∧ ∃ (A : Matrix (Fin 2005) (Fin 2005) (Fin (n + 1))), isCoolMatrix A :=
by {
  sorry
}

end smallest_n_for_cool_matrix_l282_282371


namespace total_oranges_in_stack_l282_282832

-- Definitions based on the given conditions
def base_layer_oranges : Nat := 5 * 8
def second_layer_oranges : Nat := 4 * 7
def third_layer_oranges : Nat := 3 * 6
def fourth_layer_oranges : Nat := 2 * 5
def fifth_layer_oranges : Nat := 1 * 4

-- Theorem statement equivalent to the math problem
theorem total_oranges_in_stack : base_layer_oranges + second_layer_oranges + third_layer_oranges + fourth_layer_oranges + fifth_layer_oranges = 100 :=
by
  sorry

end total_oranges_in_stack_l282_282832


namespace neznaika_expression_evals_correctly_l282_282704

theorem neznaika_expression_evals_correctly :
  ∃ (n1 n2 n3 n4 n5 n6 n7 n8 n9 : ℕ),
    n1 ≠ n2 ∧ n1 ≠ n3 ∧ n1 ≠ n4 ∧ n1 ≠ n5 ∧ n1 ≠ n6 ∧ n1 ≠ n7 ∧ n1 ≠ n8 ∧ n1 ≠ n9 ∧
    n2 ≠ n3 ∧ n2 ≠ n4 ∧ n2 ≠ n5 ∧ n2 ≠ n6 ∧ n2 ≠ n7 ∧ n2 ≠ n8 ∧ n2 ≠ n9 ∧
    n3 ≠ n4 ∧ n3 ≠ n5 ∧ n3 ≠ n6 ∧ n3 ≠ n7 ∧ n3 ≠ n8 ∧ n3 ≠ n9 ∧
    n4 ≠ n5 ∧ n4 ≠ n6 ∧ n4 ≠ n7 ∧ n4 ≠ n8 ∧ n4 ≠ n9 ∧
    n5 ≠ n6 ∧ n5 ≠ n7 ∧ n5 ≠ n8 ∧ n5 ≠ n9 ∧
    n6 ≠ n7 ∧ n6 ≠ n8 ∧ n6 ≠ n9 ∧
    n7 ≠ n8 ∧ n7 ≠ n9 ∧
    n8 ≠ n9 ∧
    1 ≤ n1 ∧ n1 ≤ 9 ∧
    1 ≤ n2 ∧ n2 ≤ 9 ∧
    1 ≤ n3 ∧ n3 ≤ 9 ∧
    1 ≤ n4 ∧ n4 ≤ 9 ∧
    1 ≤ n5 ∧ n5 ≤ 9 ∧
    1 ≤ n6 ∧ n6 ≤ 9 ∧
    1 ≤ n7 ∧ n7 ≤ 9 ∧
    1 ≤ n8 ∧ n8 ≤ 9 ∧
    1 ≤ n9 ∧ n9 ≤ 9 ∧
    ((((0 + n1) + n2) * n3 + n4) * n5 + n6) = 2015 :=
by {
  let n1 := 3,
  let n2 := 2,
  let n3 := 8,
  let n4 := 4,
  let n5 := 9,
  let n6 := 7,
  let n7 := 5,

  have n1_ne_n2 : n1 ≠ n2 := by sorry,
  have n1_ne_n3 : n1 ≠ n3 := by sorry,
  have n1_ne_n4 : n1 ≠ n4 := by sorry,
  have n1_ne_n5 : n1 ≠ n5 := by sorry,
  have n1_ne_n6 : n1 ≠ n6 := by sorry,
  have n1_ne_n7 : n1 ≠ n7 := by sorry,
  have n2_ne_n3 : n2 ≠ n3 := by sorry,
  have n2_ne_n4 : n2 ≠ n4 := by sorry,
  have n2_ne_n5 : n2 ≠ n5 := by sorry,
  have n2_ne_n6 : n2 ≠ n6 := by sorry,
  have n2_ne_n7 : n2 ≠ n7 := by sorry,
  have n3_ne_n4 : n3 ≠ n4 := by sorry,
  have n3_ne_n5 : n3 ≠ n5 := by sorry,
  have n3_ne_n6 : n3 ≠ n6 := by sorry,
  have n3_ne_n7 : n3 ≠ n7 := by sorry,
  have n4_ne_n5 : n4 ≠ n5 := by sorry,
  have n4_ne_n6 : n4 ≠ n6 := by sorry,
  have n4_ne_n7 : n4 ≠ n7 := by sorry,
  have n5_ne_n6 : n5 ≠ n6 := by sorry,
  have n5_ne_n7 : n5 ≠ n7 := by sorry,
  have n6_ne_n7 : n6 ≠ n7 := by sorry,

  have n1_in_bound : 1 ≤ n1 ∧ n1 ≤ 9 := by sorry,
  have n2_in_bound : 1 ≤ n2 ∧ n2 ≤ 9 := by sorry,
  have n3_in_bound : 1 ≤ n3 ∧ n3 ≤ 9 := by sorry,
  have n4_in_bound : 1 ≤ n4 ∧ n4 ≤ 9 := by sorry,
  have n5_in_bound : 1 ≤ n5 ∧ n5 ≤ 9 := by sorry,
  have n6_in_bound : 1 ≤ n6 ∧ n6 ≤ 9 := by sorry,
  have n7_in_bound : 1 ≤ n7 ∧ n7 ≤ 9 := by sorry,

  let expr := (((((0 + n1) + n2) * n3 + n4) * n5 + n6) * n7,
  have expr_eq_2015 : expr = 2015 := by sorry,

  exact ⟨n1, n2, n3, n4, n5, n6, n7, n8, n9, n1_ne_n2, n1_ne_n3, n1_ne_n4, n1_ne_n5, n1_ne_n6, n1_ne_n7, n2_ne_n3, n2_ne_n4, n2_ne_n5, n2_ne_n6, n2_ne_n7, n3_ne_n4, n3_ne_n5, n3_ne_n6, n3_ne_n7, n4_ne_n5, n4_ne_n6, n4_ne_n7, n5_ne_n6, n5_ne_n7, n6_ne_n7, n1_in_bound, n2_in_bound, n3_in_bound, n4_in_bound, n5_in_bound, n6_in_bound, n7_in_bound, expr_eq_2015⟩
}

end neznaika_expression_evals_correctly_l282_282704


namespace central_angle_of_sector_l282_282934

theorem central_angle_of_sector (l S : ℝ) (r : ℝ) (θ : ℝ) 
  (h1 : l = 5) 
  (h2 : S = 5) 
  (h3 : S = (1 / 2) * l * r) 
  (h4 : l = θ * r): θ = 2.5 := by
  sorry

end central_angle_of_sector_l282_282934


namespace justin_additional_time_l282_282356

theorem justin_additional_time (classmates : ℕ) (gathering_hours : ℕ) (minutes_per_flower : ℕ) 
  (flowers_lost : ℕ) : gathering_hours = 2 →
  minutes_per_flower = 10 →
  flowers_lost = 3 →
  classmates = 30 →
  let flowers_gathered := (gathering_hours * 60) / minutes_per_flower in
  let flowers_remaining := flowers_gathered - flowers_lost in
  let flowers_needed := classmates - flowers_remaining in
  let additional_time := flowers_needed * minutes_per_flower in
  additional_time = 210 :=
begin
  intros,
  unfold flowers_gathered flowers_remaining flowers_needed additional_time,
  rw [gathering_hours_eq, minutes_per_flower_eq, flowers_lost_eq, classmates_eq],
  norm_num,
end

end justin_additional_time_l282_282356


namespace coins_player_1_received_l282_282423

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end coins_player_1_received_l282_282423


namespace values_of_a_l282_282969

noncomputable def M : Set ℝ := {x | x^2 = 1}

noncomputable def N (a : ℝ) : Set ℝ := 
  if a = 0 then ∅ else {x | a * x = 1}

theorem values_of_a (a : ℝ) : (N a ⊆ M) ↔ (a = -1 ∨ a = 0 ∨ a = 1) := by
  sorry

end values_of_a_l282_282969


namespace genuine_coins_probability_l282_282817

-- Statement of the problem
theorem genuine_coins_probability:
  let total_coins := 12
  let genuine_coins := 9
  let counterfeit_coins := 3
  let first_pair_genuine := (genuine_coins / total_coins) * ((genuine_coins - 1) / (total_coins - 1))
  let second_pair_genuine := ((genuine_coins - 2) / (total_coins - 2)) * ((genuine_coins - 3) / (total_coins - 3))
  let prob_all_genuine := first_pair_genuine * second_pair_genuine
  (prob_all_genuine / (prob_all_genuine + _)) = 14 / 55 := sorry

end genuine_coins_probability_l282_282817


namespace coefficient_x3y5_in_expansion_l282_282037

theorem coefficient_x3y5_in_expansion (x y : ℕ) : 
  @binomial 8 3 * x^3 * y^5 = x^3 * y^5 * 56 := 
sorry

end coefficient_x3y5_in_expansion_l282_282037


namespace greatest_b_l282_282449

theorem greatest_b (b : ℤ) (h : ∀ x : ℝ, x^2 + b * x + 20 ≠ -6) : b = 10 := sorry

end greatest_b_l282_282449


namespace range_of_a_l282_282941

open Set

def A (a : ℝ) : Set ℝ := { x | a - 1 ≤ x ∧ x ≤ 2 * a + 1 }
def B : Set ℝ := { x | -2 ≤ x ∧ x ≤ 4 }

theorem range_of_a (a : ℝ) (h : A a ∪ B = B) : a ∈ Iio (-2) ∪ Icc (-1) (3 / 2) :=
by
  sorry

end range_of_a_l282_282941


namespace tax_deduction_cents_l282_282507

def bob_hourly_wage : ℝ := 25
def tax_rate : ℝ := 0.025

theorem tax_deduction_cents :
  (bob_hourly_wage * 100 * tax_rate) = 62.5 :=
by
  -- This is the statement that needs to be proven.
  sorry

end tax_deduction_cents_l282_282507


namespace residue_of_f_l282_282555

noncomputable def f (z : ℂ) : ℂ := 1 / (z^4 + 1)

def roots : List ℂ := [exp (complex.I * (Real.pi / 4)), exp (complex.I * (3 * Real.pi / 4)), 
                       exp (-complex.I * (3 * Real.pi / 4)), exp (-complex.I * (Real.pi / 4))]

def residues : List ℂ := [1 / 4 * exp (-complex.I * (3 * Real.pi / 4)), -1 / 4, 
                          -1 / 4, 1 / 4 * exp (complex.I * (3 * Real.pi / 4))]

theorem residue_of_f :
  List.map (fun z : ℂ => complex.residue f z) roots = residues :=
by
  sorry

end residue_of_f_l282_282555


namespace f_zero_unique_l282_282747

theorem f_zero_unique (f : ℝ → ℝ) (h : ∀ x y : ℝ, f (x + y) = f x + f (xy)) : f 0 = 0 :=
by {
  -- proof goes here
  sorry
}

end f_zero_unique_l282_282747


namespace sqrt_x_minus_1_meaningful_example_l282_282630

theorem sqrt_x_minus_1_meaningful_example :
  ∃ x : ℝ, x - 1 ≥ 0 ∧ x = 2 :=
by
  use 2
  split
  · linarith
  · refl

end sqrt_x_minus_1_meaningful_example_l282_282630


namespace count_palindromes_1000_2000_l282_282251

theorem count_palindromes_1000_2000 : 
  ∃ (n : ℕ), n = 10 ∧
  (∀ (x : ℕ), 1000 ≤ x ∧ x < 2000
  → (int.digits 10 x).reverse = (int.digits 10 x) 
  → int.digits 10 x = [1, b, b, 1] ∧ ∃ b: ℕ, b < 10) :=
sorry

end count_palindromes_1000_2000_l282_282251


namespace students_more_than_pets_l282_282109

-- Definitions for the conditions
def number_of_classrooms := 5
def students_per_classroom := 22
def rabbits_per_classroom := 3
def hamsters_per_classroom := 2

-- Total number of students in all classrooms
def total_students := number_of_classrooms * students_per_classroom

-- Total number of pets in all classrooms
def total_pets := number_of_classrooms * (rabbits_per_classroom + hamsters_per_classroom)

-- The theorem to prove
theorem students_more_than_pets : 
  total_students - total_pets = 85 :=
by
  sorry

end students_more_than_pets_l282_282109


namespace units_digit_47_4_plus_28_4_l282_282034

theorem units_digit_47_4_plus_28_4 (units_digit_47 : Nat := 7) (units_digit_28 : Nat := 8) :
  (47^4 + 28^4) % 10 = 7 :=
by
  sorry

end units_digit_47_4_plus_28_4_l282_282034


namespace period_of_sin_minus_cos_l282_282454

def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem period_of_sin_minus_cos : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by
  intros x
  sorry

end period_of_sin_minus_cos_l282_282454


namespace percent_decrease_correct_l282_282672
-- Definitions
def originalPrice : ℝ := 100
def salePrice : ℝ := 55
def percentDecrease (op sp : ℝ) : ℝ := ((op - sp) / op) * 100

-- Theorem statement
theorem percent_decrease_correct :
  percentDecrease originalPrice salePrice = 45 := 
by 
  sorry

end percent_decrease_correct_l282_282672


namespace D_score_l282_282919

noncomputable def score_A : ℕ := 94

variables (A B C D E : ℕ)

-- Conditions
def A_scored : A = score_A := sorry
def B_highest : B > A := sorry
def C_average_AD : (C * 2) = A + D := sorry
def D_average_five : (D * 5) = A + B + C + D + E := sorry
def E_score_C2 : E = C + 2 := sorry

-- Question
theorem D_score : D = 96 :=
by {
  sorry
}

end D_score_l282_282919


namespace k1_eq_k2_l282_282973

-- Definitions of the basic structure
variables {R1 R2 d : ℝ} -- Radii of the circles and distance between centers
variables {O1 O2 A1 A2 : ℝ} -- Centers and intersection points

-- Hypotheses
axiom circles_non_intersecting : R1 < d - R2
axiom radius_condition : R1 < R2
axiom farthest_point_definition (O1 O2 A1 A2 : ℝ) : True -- simplify point definition for this setup

-- Circles K1 and K2 construction conditions
def K1_radius : ℝ := 2 * R1 * R2 / d -- Function definition
def K2_radius : ℝ := 2 * R1 * R2 / d -- Function definition

-- The final theorem to prove
theorem k1_eq_k2 : K1_radius = K2_radius :=
by
  sorry

end k1_eq_k2_l282_282973


namespace forces_arithmetic_progression_ratio_l282_282755

theorem forces_arithmetic_progression_ratio 
  (a d : ℝ) 
  (h1 : ∀ (x y z : ℝ), IsArithmeticProgression x y z → x = a ∧ y = a + d ∧ z = a + 2d)
  (h2 : a^2 + (a + d)^2 = (a + 2d)^2)
  (h3 : a ≠ 0 ∧ d ≠ 0) :
  d / a = 1 / 3 :=
by
  sorry

end forces_arithmetic_progression_ratio_l282_282755


namespace kids_on_Monday_l282_282348

-- Defining the conditions
def kidsOnTuesday : ℕ := 10
def difference : ℕ := 8

-- Formulating the theorem to prove the number of kids Julia played with on Monday
theorem kids_on_Monday : kidsOnTuesday + difference = 18 := by
  sorry

end kids_on_Monday_l282_282348


namespace probability_sum_of_10_l282_282641

theorem probability_sum_of_10 (total_outcomes : ℕ) 
  (h1 : total_outcomes = 6^4) : 
  (46 / total_outcomes) = 23 / 648 := by
  sorry

end probability_sum_of_10_l282_282641


namespace only_zero_and_one_square_equal_themselves_l282_282802

theorem only_zero_and_one_square_equal_themselves (x: ℝ) : (x^2 = x) ↔ (x = 0 ∨ x = 1) :=
by sorry

end only_zero_and_one_square_equal_themselves_l282_282802


namespace sum_of_first_eight_terms_l282_282193

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l282_282193


namespace math_equivalence_l282_282981

theorem math_equivalence (m n : ℤ) (h : |m - 2023| + (n + 2024)^2 = 0) : (m + n) ^ 2023 = -1 := 
by
  sorry

end math_equivalence_l282_282981


namespace car_speed_second_hour_l282_282000

variable (x : ℝ)
variable (s1 : ℝ := 100)
variable (avg_speed : ℝ := 90)
variable (total_time : ℝ := 2)

-- The Lean statement equivalent to the problem
theorem car_speed_second_hour : (100 + x) / 2 = 90 → x = 80 := by 
  intro h
  have h₁ : 2 * 90 = 100 + x := by 
    linarith [h]
  linarith [h₁]

end car_speed_second_hour_l282_282000


namespace geometric_sequence_101_l282_282594

variable {α : Type*} [Field α]

def geometric_sequence (a : α) (q : α) (n : ℕ) : α := a * q^n

theorem geometric_sequence_101
  (a : α)
  (q : α)
  (h1 : geometric_sequence a q 2 = 3)
  (h2 : geometric_sequence a q 2015 + geometric_sequence a q 2016 = 0) :
  geometric_sequence a q 100 = 3 :=
begin
  sorry
end

end geometric_sequence_101_l282_282594


namespace find_number_of_appliances_l282_282524

-- Declare the constants related to the problem.
def commission_per_appliance : ℝ := 50
def commission_percent : ℝ := 0.1
def total_selling_price : ℝ := 3620
def total_commission : ℝ := 662

-- Define the theorem to solve for the number of appliances sold.
theorem find_number_of_appliances (n : ℝ) 
  (H : n * commission_per_appliance + commission_percent * total_selling_price = total_commission) : 
  n = 6 := 
sorry

end find_number_of_appliances_l282_282524


namespace isosceles_triangle_perimeter_correct_l282_282984

noncomputable def isosceles_triangle_perimeter (x y : ℝ) : ℝ :=
  if x = y then 2 * x + y else if (2 * x > y ∧ y > 2 * x - y) ∨ (2 * y > x ∧ x > 2 * y - x) then 2 * y + x else 0

theorem isosceles_triangle_perimeter_correct (x y : ℝ) (h : |x - 5| + (y - 8)^2 = 0) :
  isosceles_triangle_perimeter x y = 18 ∨ isosceles_triangle_perimeter x y = 21 := by
sorry

end isosceles_triangle_perimeter_correct_l282_282984


namespace sqrt_meaningful_for_x_l282_282319

theorem sqrt_meaningful_for_x (x : ℝ) : (∃ r : ℝ, r = real.sqrt (x - 3)) ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_for_x_l282_282319


namespace DEF_area_inequality_l282_282473

noncomputable def TriangleABC : Type := { area : ℝ // area = 1 }

variables (A B C D E F : ℝ) (triangleABC : TriangleABC)

-- Conditions
def on_BC (D : ℝ) : Prop := D ∈ [B, C]
def on_CA (E : ℝ) : Prop := E ∈ [C, A]
def on_AB (F : ℝ) : Prop := F ∈ [A, B]

def is_cyclic_quad (A F D E : ℝ) : Prop := (A F D E) is cyclic

def area_triangle_def (DEF : ℝ) (EF AD : ℝ) : Prop := 
  DEF ≤ EF^2 / (4 * AD^2)

-- The statement to prove
theorem DEF_area_inequality (triangleABC : TriangleABC) (D E F : ℝ) 
  (hD: on_BC D) (hE: on_CA E) (hF: on_AB F) (h_cyclic: is_cyclic_quad A F D E) :
  ∃ DEF EF AD, area_triangle_def DEF EF AD := 
sorry

end DEF_area_inequality_l282_282473


namespace maximum_area_of_rectangle_with_given_perimeter_l282_282414

noncomputable def perimeter : ℝ := 30
noncomputable def area (length width : ℝ) : ℝ := length * width
noncomputable def max_area : ℝ := 56.25

theorem maximum_area_of_rectangle_with_given_perimeter :
  ∃ length width : ℝ, 2 * length + 2 * width = perimeter ∧ area length width = max_area :=
sorry

end maximum_area_of_rectangle_with_given_perimeter_l282_282414


namespace find_m_range_l282_282205

variable (m : ℝ)

def p : Prop := ∃ x_0 : ℝ, x_0^2 + 2*(m - 3)*x_0 + 1 < 0
def q : Prop := ∀ x : ℝ, 4*x^2 + 4*(m - 2)*x + 1 > 0

theorem find_m_range (hm : p ∨ q) (hn : ¬(p ∧ q)) : m > 4 ∨ m ≤ 1 ∨ (2 ≤ m ∧ m < 3) := sorry

end find_m_range_l282_282205


namespace bisection_interval_l282_282444

def f(x : ℝ) := x^3 - 2 * x - 5

theorem bisection_interval :
  f 2 < 0 ∧ f 3 > 0 ∧ f 2.5 > 0 →
  ∃ a b : ℝ, a = 2 ∧ b = 2.5 ∧ f a * f b ≤ 0 :=
by
  sorry

end bisection_interval_l282_282444


namespace limit_of_sequence_l282_282880

noncomputable def u (n : ℕ) : ℝ := (2 * n^2 + 2) / (2 * n^2 + 1)

theorem limit_of_sequence :
  filter.tendsto (λ n : ℕ, u n ^ n^2) filter.at_top (𝓝 (Real.sqrt Real.exp)) :=
sorry

end limit_of_sequence_l282_282880


namespace value_of_a_l282_282813

def f (x : ℝ) (a : ℝ) : ℝ := if x < 1 then 3 * x + 2 else x ^ 2 + a * x

theorem value_of_a (a : ℝ) (h : f (f 0 a) a = 4 * a) : a = 2 := by
  sorry

end value_of_a_l282_282813


namespace length_of_bridge_bridge_length_is_correct_l282_282056

theorem length_of_bridge (L : ℝ) (v : ℝ) (t : ℝ) (l_train : ℝ) (length_bridge : ℝ) :
  l_train = 180 ∧ v = 60 ∧ t = 45 ∧ length_bridge = 570.15 → 
  length_bridge = v * 1000 / 3600 * t - l_train := by sorry

-- Provide the assumed conditions
theorem bridge_length_is_correct :
  length_of_bridge v t l_train length_bridge := by
    -- Use proper values for the conditions to match the given problem
    have h : l_train = 180 ∧ v = 60 ∧ t = 45 ∧ length_bridge = 570.15 :=
      by simp [l_train, v, t, length_bridge]
    sorry

end length_of_bridge_bridge_length_is_correct_l282_282056


namespace section_Diligence_students_before_transfer_l282_282996

-- Define the variables
variables (D_after I_after D_before : ℕ)

-- Problem Statement
theorem section_Diligence_students_before_transfer :
  ∀ (D_after I_after: ℕ),
    2 + D_after = I_after
    ∧ D_after + I_after = 50 →
    ∃ D_before, D_before = D_after - 2 ∧ D_before = 23 :=
by
sorrry

end section_Diligence_students_before_transfer_l282_282996


namespace complex_conjugate_product_l282_282375

theorem complex_conjugate_product (z : ℂ) (h : z = 1 + ⟨0, 1⟩) : z * (∗z) = 2 := by
  sorry

end complex_conjugate_product_l282_282375


namespace maria_possible_combinations_l282_282383

def digits : set ℕ := {1, 2, 3, 4, 5, 6}

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k
def is_odd (n : ℕ) : Prop := ∃ k, n = 2 * k + 1

def follows_rule (d1 d2 : ℕ) : Prop :=
  (is_even d1 → is_odd d2) ∧ (is_odd d1 → is_even d2)

def valid_code (code : list ℕ) : Prop :=
  code.length = 4 ∧
  (∀ (i : ℕ), i < 3 → follows_rule (code.nth_le i sorry) (code.nth_le (i + 1) sorry)) ∧
  (∀ (d : ℕ), d ∈ code → d ∈ digits)

theorem maria_possible_combinations :
  ∃ count : nat, count = 162 ∧ (∀ code : list ℕ, valid_code code → code.length = 4) :=
begin
  use 162,
  split,
  { refl },
  { sorry }
end

end maria_possible_combinations_l282_282383


namespace part1_part2_l282_282336

theorem part1 (A B C a b c : ℝ) (h1 : 3 * a * Real.cos A = Real.sqrt 6 * (c * Real.cos B + b * Real.cos C)) :
    Real.tan (2 * A) = 2 * Real.sqrt 2 := sorry

theorem part2 (A B C a b c S : ℝ) 
  (h_sin_B : Real.sin (Real.pi / 2 + B) = 2 * Real.sqrt 2 / 3)
  (hc : c = 2 * Real.sqrt 2) :
    S = 2 * Real.sqrt 2 / 3 := sorry

end part1_part2_l282_282336


namespace non_empty_solution_set_range_l282_282639

theorem non_empty_solution_set_range (a : ℝ) :
  (∃ x : ℝ, |x + 2| - |x - 1| < a) → a > -3 :=
sorry

end non_empty_solution_set_range_l282_282639


namespace max_height_reached_l282_282843

def h (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

theorem max_height_reached : ∃ t : ℝ, h(t) = 40 := 
sorry

end max_height_reached_l282_282843


namespace math_scores_distribution_l282_282824

-- Define the normal distribution with the given mean and percentage conditions
def normal_distribution (X : Type) [MeasureSpace X] : Prop :=
  ∃ (μ : ℝ) (σ : ℝ), μ = 90 ∧ P(X ≤ 60) = 0.10 ∧ ∀ s, normalDist(s) = 1/(σ * sqrt(2 * π)) * exp(-(s - μ)^2 / (2 * σ^2))

-- Theorem stating the result
theorem math_scores_distribution (X : Type) [MeasureSpace X] :
  normal_distribution X →
  ∃ (percentage : ℝ), percentage = 0.40 ∧ percentage_of_candidates_between (X, 90, 120) = percentage :=
by
  sorry

end math_scores_distribution_l282_282824


namespace find_length_MA_l282_282575

theorem find_length_MA {O A M B C : Type*} [NormedSpace ℝ (EuclideanSpace ℝ)] 
  (circle_center : O) (radius_one : dist O B = 1) (radius_one' : dist O C = 1) 
  (is_tangent : ∀ (P : Type*), ∃ (B' C' : Type*), P = A → is_tangent B' C') 
  (M_on_circle : dist O M = 1) 
  (areas_equal : area O B M C = area A B M C) :
  dist A M = 1 :=
sorry

end find_length_MA_l282_282575


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282282

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282282


namespace number_of_arrangements_without_adjacent_ABC_l282_282773

def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | (n + 1) => (n + 1) * factorial n

def permutations (n r : ℕ) : ℕ :=
  factorial n / factorial (n - r)

theorem number_of_arrangements_without_adjacent_ABC (total_people : ℕ) (A B C : Type) (others : Fin 4 → Type) :
  total_people = 7 → 
  permutations 4 4 = 24 →
  permutations 5 3 = 60 →
  24 * 60 = 1440 :=
by
  intros htotal hperm1 hperm2
  rw [htotal, hperm1, hperm2]
  exact rfl

end number_of_arrangements_without_adjacent_ABC_l282_282773


namespace num_common_divisors_60_90_l282_282623

theorem num_common_divisors_60_90 : 
  let n1 := 60
  let n2 := 90
  let gcd := 30 -- GCD calculated from prime factorizations
  let divisors_of_gcd := [1, 2, 3, 5, 6, 10, 15, 30]
  in divisors_of_gcd.length = 8 :=
by
  sorry

end num_common_divisors_60_90_l282_282623


namespace marble_probability_l282_282069

noncomputable def total_marbles : ℕ := 15 + 10 + 5

noncomputable def total_draws : ℕ := 4

noncomputable def total_outcomes : ℕ := (Nat.choose total_marbles total_draws)

noncomputable def red_selections : ℕ := Nat.choose 15 2

noncomputable def blue_selection : ℕ := Nat.choose 10 1

noncomputable def green_selection : ℕ := Nat.choose 5 1

noncomputable def favorable_outcomes : ℕ := red_selections * blue_selection * green_selection

noncomputable def probability : ℚ := favorable_outcomes / total_outcomes

theorem marble_probability :
  probability = 350 / 1827 := 
by 
  sorry

end marble_probability_l282_282069


namespace incorrect_option_D_l282_282605

def inverse_proportion (x : ℝ) := -6 / x

theorem incorrect_option_D :
  (inverse_proportion 3 = -2) ∧ -- Checking option A: graph passes through (3, -2)
  (∀ x > 0, inverse_proportion x < 0) ∧ (∀ x < 0, inverse_proportion x > 0) ∧ -- Checking option B: graph in second and fourth quadrants
  (∀ x > 0, ∀ x1 > x, inverse_proportion x1 > inverse_proportion x) ∧ -- Checking option C: y increases as x increases for x > 0
  (∀ x < 0, ∀ x1 < x, inverse_proportion x1 > inverse_proportion x) → false := -- Checking option D: y decreases as x increases for x < 0, should return false
by
  intro h
  sorry

end incorrect_option_D_l282_282605


namespace largest_m_for_2310_divides_prod_pow_l282_282203

def pow (n : ℕ) : ℕ :=
  if h : n ≥ 2 then
    let p := (Nat.min_fac n) in
    let k := Nat.find (λ k, ¬Nat.divisible (n / p^k) p) in
    p^k
  else 1

theorem largest_m_for_2310_divides_prod_pow :
  (∀ (n : ℕ), n ≥ 2 → pow n = (Nat.min_fac n) ^ (Nat.find (λ k, ¬Nat.divisible (n / (Nat.min_fac n)^k) (Nat.min_fac n)))) →
  ∀ (n : ℕ), n ≥ 2 →
  (2310^2) ∣ ∏ x in Finset.range (4500 + 1).filter(λ x, x ≥ 2), pow x :=
by
  sorry

end largest_m_for_2310_divides_prod_pow_l282_282203


namespace area_ratio_greater_l282_282015

theorem area_ratio_greater 
  {A B C A1 B1 C1 : Type*}
  (triangle_ABC : is_triangle A B C)
  (triangle_A1B1C1 : is_triangle A1 B1 C1)
  (acute_ABC : is_acute A B C)
  (acute_A1B1C1 : is_acute A1 B1 C1)
  (B1_on_BC : is_on_segment B C B1)
  (C1_on_BC : is_on_segment B C C1)
  (A1_inside_ABC : is_inside A1 A B C):
  (area A B C / (distance A B + distance A C) > area A1 B1 C1 / (distance A1 B1 + distance A1 C1)) :=
sorry

end area_ratio_greater_l282_282015


namespace andrena_more_than_christel_l282_282526

variables (debelin_start christel_start debelin_give christel_give : ℕ)
variables (andrena_more_than_debelin : ℕ)

def dollsDebelinAfter : ℕ := debelin_start - debelin_give
def dollsChristelAfter : ℕ := christel_start - christel_give
def dollsAndrena_start (A : ℕ) : ℕ := A
def dollsAndrena_received : ℕ := debelin_give + christel_give
def dollsAndrenaAfter (A : ℕ) : ℕ := dollsAndrena_start A + dollsAndrena_received

axiom debelin_dolls_start : debelin_start = 20
axiom christel_dolls_start : christel_start = 24
axiom debelin_doll_give : debelin_give = 2
axiom christel_doll_give : christel_give = 5
axiom andrena_dolls_after (A : ℕ) : A + dollsAndrena_received = dollsDebelinAfter 20 - 2 + andrena_more_than_debelin
axiom andrena_more_doll_than_debelin : debelin_start - debelin_give = 18
axiom andrena_more_debelin : andrena_more_debelin = 3

theorem andrena_more_than_christel (A : ℕ) : (dollsAndrenaAfter A) - (dollsChristelAfter christel_start christel_give) = 2 :=
by {
  sorry
}

end andrena_more_than_christel_l282_282526


namespace count_two_digit_perfect_squares_divisible_by_four_l282_282258

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l282_282258


namespace alley_width_l282_282650

noncomputable def calculate_width (l k h : ℝ) : ℝ :=
  l / 2

theorem alley_width (k h l w : ℝ) (h1 : k = (l * (Real.sin (Real.pi / 3)))) (h2 : h = (l * (Real.sin (Real.pi / 6)))) :
  w = calculate_width l k h :=
by
  sorry

end alley_width_l282_282650


namespace min_knows_all_attendees_l282_282228

def gathering (P : Type) (attends : P → Prop) : Prop :=
  ∃ n, n = 1982 ∧ ∀ (A B C D : P), attends A ∧ attends B ∧ attends C ∧ attends D → 
    (A = B ∨ A = C ∨ A = D ∨ B = C ∨ B = D ∨ C = D ∨ knows A B C D)

def knows (A B C D : P) : Prop :=
  (knows_all A [B, C, D] ∨ knows_all B [A, C, D] ∨ knows_all C [A, B, D] ∨ knows_all D [A, B, C])

def knows_all (X : P) (Y : list P) : Prop :=
  ∀ y ∈ Y, knows X y

theorem min_knows_all_attendees (P : Type) (attends : P → Prop) :
  gathering P attends → (∃ x, x = 1979 ∧ ∀ (p ∈ attendees_set), knows_all p attendees_set) :=
sorry

end min_knows_all_attendees_l282_282228


namespace correct_answer_l282_282504

-- Define the problem conditions and question
def equation (y : ℤ) : Prop := y + 2 = -3

-- Prove that the correct answer is y = -5
theorem correct_answer : ∀ y : ℤ, equation y → y = -5 :=
by
  intros y h
  unfold equation at h
  linarith

end correct_answer_l282_282504


namespace arrangements_count_l282_282439

-- Definitions of conditions
constant n_teachers : ℕ := 3
constant n_classes : ℕ := 6
constant classes_per_teacher : ℕ := 2

-- Theorem statement
theorem arrangements_count : 
  (finset.powersetLen classes_per_teacher (finset.range n_classes)).card = 90 := 
sorry

end arrangements_count_l282_282439


namespace mod_inverse_13_997_l282_282451

-- The theorem statement
theorem mod_inverse_13_997 : ∃ x : ℕ, 0 ≤ x ∧ x < 997 ∧ (13 * x) % 997 = 1 ∧ x = 767 := 
by
  sorry

end mod_inverse_13_997_l282_282451


namespace sum_geometric_sequence_l282_282180

theorem sum_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 1/3) (h_r : r = 1/3) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in S_n = 3280/6561 :=
by
  sorry

end sum_geometric_sequence_l282_282180


namespace kim_boxes_sold_on_tuesday_l282_282678

theorem kim_boxes_sold_on_tuesday :
  ∀ (T W Th F : ℕ),
  (T = 3 * W) →
  (W = 2 * Th) →
  (Th = 3 / 2 * F) →
  (F = 600) →
  T = 5400 :=
by
  intros T W Th F h1 h2 h3 h4
  sorry

end kim_boxes_sold_on_tuesday_l282_282678


namespace sqrt_2x_plus_y_eq_4_l282_282226

theorem sqrt_2x_plus_y_eq_4 (x y : ℝ) 
  (h1 : (3 * x + 1) = 4) 
  (h2 : (2 * y - 1) = 27) : 
  Real.sqrt (2 * x + y) = 4 := 
by 
  sorry

end sqrt_2x_plus_y_eq_4_l282_282226


namespace quotient_of_division_is_123_l282_282916

theorem quotient_of_division_is_123 :
  let d := 62976
  let v := 512
  d / v = 123 := by
  sorry

end quotient_of_division_is_123_l282_282916


namespace two_distinct_rectangles_l282_282086

theorem two_distinct_rectangles (a b : ℝ) (h : a < b) :
  ∃ (x y : ℝ), (x + y = (a + b) / 2) ∧ (x * y = (a * b) / 2) ∧ (x = a ∧ y = b ∨ x = b ∧ y = a) :=
by
  existsi a, b,
  split,
  calc a + b = (a + b) : by simp
  ... = (a + b) / 1 : by rw div_one
  ... = (a + b) / 2 + (a + b) / 2 : by rw ←[({a + b} / 2) + ({a + b} / 2)]
  ... = (a + b) / 2 : by simp,
  split,
  calc a * b = a * b / 1 : by rw div_one
  ... = (a * b) / 2 + (a * b) / 2 : by rw ←[(a * b) / 2 + (a * b) / 2]
  ... = (a * b) / 2 : by simp,
  left,
  constructor,
  refl,
  refl

end two_distinct_rectangles_l282_282086


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282263

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282263


namespace product_identity_l282_282887

theorem product_identity : 
  (7^3 - 1) / (7^3 + 1) * 
  (8^3 - 1) / (8^3 + 1) * 
  (9^3 - 1) / (9^3 + 1) * 
  (10^3 - 1) / (10^3 + 1) * 
  (11^3 - 1) / (11^3 + 1) = 
  133 / 946 := 
by
  sorry

end product_identity_l282_282887


namespace theodore_crafts_20_wooden_statues_every_month_l282_282430

noncomputable def stone_statue_cost := 20
noncomputable def wooden_statue_cost := 5
noncomputable def stone_statues_per_month := 10
noncomputable def tax_rate := 0.10
noncomputable def total_earnings_after_taxes := 270

noncomputable def earnings_from_stone_statues := stone_statues_per_month * stone_statue_cost
noncomputable def total_earnings_before_taxes := total_earnings_after_taxes / (1 - tax_rate)
noncomputable def earnings_from_wooden_statues := total_earnings_before_taxes - earnings_from_stone_statues
noncomputable def number_of_wooden_statues := earnings_from_wooden_statues / wooden_statue_cost

theorem theodore_crafts_20_wooden_statues_every_month :
  number_of_wooden_statues = 20 :=
by
  sorry

end theodore_crafts_20_wooden_statues_every_month_l282_282430


namespace inclination_angle_range_l282_282988

theorem inclination_angle_range
  (k : ℝ)
  (h_intersect_first_quadrant : (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ y = k * x - sqrt 3 ∧ 2 * x + 3 * y - 6 = 0)) :
  ∃ θ : ℝ, θ ∈ Set.Ioo (Real.pi / 6) (Real.pi / 2) ∧ tan θ = k :=
by
  sorry

end inclination_angle_range_l282_282988


namespace transistor_count_2010_l282_282097

theorem transistor_count_2010 :
  ∀ (doubles : ℕ → ℕ) (quadruples : ℕ → ℕ),
  (∀ n, doubles n = n * 2) → -- doubles every two years
  (∀ n, quadruples n = n * 4) → -- quadruples every two years
  let year_1992 := 2_000_000 in
  let year_2000 := doubles (doubles (doubles (doubles year_1992))) in -- 2^4 times
  let year_2010 := quadruples (quadruples (quadruples (quadruples (quadruples year_2000)))) in -- 4^5 times
  year_2010 = 32_768_000_000 :=
by
  intros doubles quadruples h_doubles h_quadruples year_1992 year_2000 year_2010
  sorry

end transistor_count_2010_l282_282097


namespace find_c_l282_282301

-- Given that the function f(x) = 2^x + c passes through the point (2,5),
-- Prove that c = 1.
theorem find_c (c : ℝ) : (∃ (f : ℝ → ℝ), (∀ x, f x = 2^x + c) ∧ (f 2 = 5)) → c = 1 := by
  sorry

end find_c_l282_282301


namespace spherical_coordinates_equivalence_l282_282658

theorem spherical_coordinates_equivalence
  (ρ θ φ : ℝ)
  (h_ρ : ρ > 0)
  (h_θ : 0 ≤ θ ∧ θ < 2 * Real.pi)
  (h_φ : φ = 2 * Real.pi - (7 * Real.pi / 4)) :
  (ρ, θ, φ) = (4, 3 * Real.pi / 4, Real.pi / 4) :=
by 
  sorry

end spherical_coordinates_equivalence_l282_282658


namespace store_A_cheaper_than_store_B_l282_282782

noncomputable def store_A_full_price : ℝ := 125
noncomputable def store_A_discount_pct : ℝ := 0.08
noncomputable def store_B_full_price : ℝ := 130
noncomputable def store_B_discount_pct : ℝ := 0.10

noncomputable def final_price_A : ℝ :=
  store_A_full_price * (1 - store_A_discount_pct)

noncomputable def final_price_B : ℝ :=
  store_B_full_price * (1 - store_B_discount_pct)

theorem store_A_cheaper_than_store_B :
  final_price_B - final_price_A = 2 :=
by
  sorry

end store_A_cheaper_than_store_B_l282_282782


namespace prime_cannot_be_sum_of_three_squares_l282_282208

theorem prime_cannot_be_sum_of_three_squares (p : ℕ) (hp : Nat.Prime p) (hmod : p % 8 = 7) :
  ¬∃ a b c : ℤ, p = a^2 + b^2 + c^2 :=
by
  sorry

end prime_cannot_be_sum_of_three_squares_l282_282208


namespace candies_per_packet_l282_282868

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l282_282868


namespace repeatingDecimal_in_lowest_terms_l282_282152

noncomputable def repeatingDecimalToFraction (n : ℕ) := 0.35 -- placeholder definition for 0.\overline{35}

theorem repeatingDecimal_in_lowest_terms : repeatingDecimalToFraction 35 = (5 : ℚ) / 14 := 
by sorry

end repeatingDecimal_in_lowest_terms_l282_282152


namespace polynomial_identity_l282_282898

noncomputable def p (x : ℝ) : ℝ := sorry

theorem polynomial_identity (p : ℝ → ℝ) (h_poly : ∀ x, p((x + 1)^3) = (p(x) + 1)^3)
  (h_initial : p 0 = 0) : ∀ x, p x = x :=
sorry

end polynomial_identity_l282_282898


namespace sequence_is_geometric_find_sum_T_l282_282381

-- Goal I: Prove that {a_n} defined by S_n = 3a_n - 2 is a geometric sequence
theorem sequence_is_geometric (S : ℕ → ℝ) (a : ℕ → ℝ) :
  (∀ n, S n = 3 * a n - 2) →
  (∃ r : ℝ, ∀ n, a (n + 1) = r * a n) :=
by
  sorry

-- Goal II: Given b_(n+1) = a_n + b_n and b_1 = -3, find T_n = 4(3/2)^n - 5n - 4
theorem find_sum_T (a b : ℕ → ℝ) (T : ℕ → ℝ) :
  (∀ n, S (n + 1) - S n = 3 * a (n + 1) - 3 * a n) →
  (b 1 = -3) →
  (∀ n, b (n + 1) = a n + b n) →
  (T n = 4 * (3 / 2) ^ n - 5 * n - 4) :=
by
  sorry

end sequence_is_geometric_find_sum_T_l282_282381


namespace plane_length_approximation_l282_282859

theorem plane_length_approximation:
  (total_length hangar planes : ℝ) (num_planes : ℕ) 
  (h1 : total_length = 300) (h2 : num_planes = 7) :
  total_length / num_planes ≈ 42.86 := 
sorry

end plane_length_approximation_l282_282859


namespace maximize_f_l282_282087

def divisors_count (n : ℕ) : ℕ := (finset.range n.succ).filter (λ d, n % d = 0).card

def D (a b : ℕ) : ℕ := divisors_count(a) + divisors_count(b) - 1

def perimeter (a b : ℕ) : ℕ := 2 * (a + b)

def f (a b : ℕ) : ℚ := (D a b : ℚ) / (perimeter a b : ℚ)

theorem maximize_f : ∀ a b : ℕ, (2 ≤ a ∧ 2 ≤ b) → f a b ≤ f 2 2 :=
by
  -- Proof omitted
  sorry

#eval maximize_f 2 2 (by simp)

end maximize_f_l282_282087


namespace marys_total_cards_l282_282385

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought_by_mary : ℕ := 40

theorem marys_total_cards :
  initial_cards - torn_cards + cards_from_fred + cards_bought_by_mary = 76 :=
by
  sorry

end marys_total_cards_l282_282385


namespace find_coordinates_of_P0_find_equation_of_l_l282_282960

noncomputable def curve (x : ℝ) : ℝ := x^3 + x - 2

def tangent_slope (x : ℝ) : ℝ := 3 * x^2 + 1

def is_in_third_quadrant (p : ℝ × ℝ) : Prop := p.1 < 0 ∧ p.2 < 0

def line_eq (a b c x y : ℝ) : Prop := a * x + b * y + c = 0

/-- Problem statement 1: Find the coordinates of P₀ --/
theorem find_coordinates_of_P0 (p0 : ℝ × ℝ)
    (h_tangent_parallel : tangent_slope p0.1 = 4)
    (h_third_quadrant : is_in_third_quadrant p0) :
    p0 = (-1, -4) :=
sorry

/-- Problem statement 2: Find the equation of line l --/
theorem find_equation_of_l (P0 : ℝ × ℝ)
    (h_P0_coordinates: P0 = (-1, -4))
    (h_perpendicular : ∀ (l1_slope : ℝ), l1_slope = 4 → ∃ l_slope : ℝ, l_slope = (-1) / 4)
    (x y : ℝ) : 
    line_eq 1 4 17 x y :=
sorry

end find_coordinates_of_P0_find_equation_of_l_l282_282960


namespace repair_time_l282_282505

-- Definitions for the conditions
def vp : ℝ -- pedestrian's speed, constant
def vc : ℝ -- cyclist's speed, constant except for the repair interval

def t1 := 9 -- starting time for pedestrian in hours
def t2 := 10 -- starting time for cyclist in hours
def t3 := 10.5 -- time when the cyclist first catches up to the pedestrian in hours
def t4 := 13 -- time when the cyclist catches up again after the repair in hours

-- The catch-up condition at 10:30, which gives us v_c = 3 * v_p
def catch_up_condition : Prop := 
  vc * (t3 - t2) = vp * (t3 - t1)

-- The duration definitions
def total_time_cyclist_in_motion := t4 - t2 -- total time cyclist was in motion
def total_time_cycle_diff := vp * (t4 - t1) / vc -- effective time cyclist would have taken if no repair

-- The main statement to be proved
theorem repair_time {vp vc : ℝ} (h : catch_up_condition) 
: total_time_cyclist_in_motion * 60 - total_time_cycle_diff * 60 = 100 := 
sorry

end repair_time_l282_282505


namespace coefficient_x3_in_expansion_l282_282735

theorem coefficient_x3_in_expansion :
  let x : ℝ := by sorry,
  let expansion := (1 + x)^2 * (1 - 2 * x)^5,
  let term := -8 * Nat.choose 5 3 + 8 * Nat.choose 5 2 - 2 * Nat.choose 5 1 in
  coefficient_of_expansion expansion 3 = term := by sorry

end coefficient_x3_in_expansion_l282_282735


namespace joshuas_share_l282_282676

theorem joshuas_share (total amount : ℝ) (joshua_share : ℝ) (justin_share: ℝ) 
  (h1: total amount = 40) 
  (h2: joshua_share = 3 * justin_share) 
  (h3: total amount = joshua_share + justin_share) 
: joshua_share = 30 := 
by  sorry

end joshuas_share_l282_282676


namespace common_difference_of_common_terms_l282_282004

def sequence_a (n : ℕ) : ℕ := 4 * n - 3
def sequence_b (k : ℕ) : ℕ := 3 * k - 1

theorem common_difference_of_common_terms :
  ∃ (d : ℕ), (∀ (m : ℕ), 12 * m + 5 ∈ { x | ∃ (n k : ℕ), sequence_a n = x ∧ sequence_b k = x }) ∧ d = 12 := 
sorry

end common_difference_of_common_terms_l282_282004


namespace count_two_digit_perfect_squares_divisible_by_four_l282_282256

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l282_282256


namespace binary_to_base5_conversion_l282_282522

theorem binary_to_base5_conversion : 
  let n := 45 in -- decimal conversion of 101101_2
    (n = 2^5 + 2^3 + 2^2 + 2^0) ∧ -- condition for binary to decimal
    -- condition for decimal to base-5
    (n % 5 = 0) ∧ ((n / 5) % 5 = 4) ∧ ((n / 5 / 5) % 5 = 1) 
    → n.to_nat_repr 5 = "140" :=
by
  sorry

end binary_to_base5_conversion_l282_282522


namespace water_consumption_per_week_l282_282009

-- Definitions for the given conditions
def bottles_per_day := 2
def quarts_per_bottle := 1.5
def additional_ounces_per_day := 20
def days_per_week := 7
def ounces_per_quart := 32

-- Theorem to state the problem
theorem water_consumption_per_week :
  bottles_per_day * quarts_per_bottle * ounces_per_quart + additional_ounces_per_day 
  * days_per_week = 812 := 
by 
  sorry

end water_consumption_per_week_l282_282009


namespace dice_sum_less_than_16_l282_282435

open Probability

def dice_sum_probability : ℚ :=
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let valid_outcomes := outcomes.filter (λ (t : ℕ × ℕ × ℕ), t.1 + t.2.1 + t.2.2 < 16)
  valid_outcomes.card / outcomes.card

theorem dice_sum_less_than_16 : dice_sum_probability = 103 / 108 :=
by
  sorry

end dice_sum_less_than_16_l282_282435


namespace fabric_cost_correct_l282_282030

def livre_to_sou(livres : ℕ) : ℕ := livres * 20
def sou_to_denier(sous : ℕ) : ℕ := sous * 12
def ell_cost_to_denier(livres : ℕ) (sous : ℕ) (deniers : ℕ) : ℕ :=
  (livre_to_sou(livres) * 12) + (sou_to_denier(sous)) + deniers

def total_cost_in_deniers(quantity : ℚ) (cost_per_ell : ℕ) : ℚ :=
  quantity * cost_per_ell

def convert_deniers(total_deniers : ℚ) : (ℕ × ℕ × ℚ) :=
  let total_sous := total_deniers / 12
  let rem_deniers := total_deniers % 12
  let total_livres := total_sous / 20
  let rem_sous := total_sous % 20
  (total_livres.toNat, rem_sous.toNat, rem_deniers)

theorem fabric_cost_correct :
  let livres := 42
  let sous := 17
  let deniers := 11
  let quantity := 15 + 13 / 16
  let cost_per_ell := ell_cost_to_denier(livres, sous, deniers)
  let total := total_cost_in_deniers(quantity, cost_per_ell)
  let (final_livres, final_sous, final_deniers) := convert_deniers(total)
  final_livres = 682 ∧ final_sous = 15 ∧ final_deniers = 9 + (3 / 4) :=
by sorry

end fabric_cost_correct_l282_282030


namespace area_ratio_inequality_l282_282014

variable (A B C A1 B1 C1 : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]

variable (triangle_ABC : Triangle A B C)
variable (triangle_A1B1C1 : Triangle A1 B1 C1)

variable (B1_on_BC : B1 ∈ Segment B C)
variable (C1_on_BC : C1 ∈ Segment B C)
variable (A1_inside_ABC : A1 ∈ Interior (Triangle A B C))

variable (S S1 : ℝ)
variable (S_eq_area_ABC : S = area (Triangle A B C))
variable (S1_eq_area_A1B1C1 : S1 = area (Triangle A1 B1 C1))

theorem area_ratio_inequality :
  let AB := dist A B
  let AC := dist A C
  let A1B1 := dist A1 B1
  let A1C1 := dist A1 C1
  S / (AB + AC) > S1 / (A1B1 + A1C1) :=
sorry

end area_ratio_inequality_l282_282014


namespace range_of_a_monotonically_decreasing_l282_282601

-- Given function f(x)
def f (x : ℝ) (a : ℝ) : ℝ := -x^3 + a*x^2 - x - 1

-- Derivative of f(x)
def f' (x : ℝ) (a : ℝ) : ℝ := -3*x^2 + a*x - 1

-- Definition of monotonically decreasing function
def is_monotonically_decreasing (f' : ℝ → ℝ) : Prop :=
  ∀ x, f' x ≤ 0
  
-- Main theorem statement
theorem range_of_a_monotonically_decreasing (a : ℝ) :
  (∀ x, f' x a ≤ 0) ↔ (-real.sqrt 3 ≤ a ∧ a ≤ real.sqrt 3) :=
by
  sorry

end range_of_a_monotonically_decreasing_l282_282601


namespace maria_wins_with_2029_coins_l282_282648

/-- The type representing the two players -/
inductive Player
| Lucas
| Maria

/-- The rules of the game:
1. Lucas can remove 3 or 5 coins, unless only 2 coins remain.
2. Maria can remove 2 or 4 coins.
3. The game begins with Lucas.
4. Whoever takes the last coin wins.
5. The game starts with a given number of coins. -/
structure Game where
  initial_coins : Nat
  player_turn  : Player
  remove_coins : (Player → Nat → List Nat)
  win_condition : (Nat → Player → Bool)
  deriving Repr

/-- Define the remove_coins function according to the rules given -/
def remove_coins : Player → Nat → List Nat
| Player.Lucas, n =>
  if n = 2 then []  -- Lucas loses if only 2 coins are left
  else if n >= 5 then [3, 5]
  else if n >= 3 then [3]
  else []
| Player.Maria, n =>
  if n >= 4 then [2, 4]
  else if n >= 2 then [2]
  else []

/-- Define the win condition function -/
def win_condition : Nat → Player → Bool
| 0, player => true
| 1, player => true
| _, _ => false

/-- The problem that needs to be proved -/
theorem maria_wins_with_2029_coins :
  ∃ best_strategy : (Nat → Player → Nat),
  (Game.mk 2029 Player.Lucas remove_coins win_condition).player_turn = Player.Maria :=
sorry

end maria_wins_with_2029_coins_l282_282648


namespace min_size_A_intersection_l282_282583

theorem min_size_A_intersection (m a b : ℕ) (hab_coprime : Nat.gcd a b = 1) (A : Set ℕ) 
  (hA : ∀ n : ℕ, n > 0 → a * n ∈ A ∧ b * n ∈ A) :
  ∃ (H : Set ℕ), H = {x | x ≤ m ∧ (x\b) + (x\(b^2)) + \ldots = 1} 
  ∧ (|A ∩ {n | n ≤ m}| = |H|) :=
sorry

end min_size_A_intersection_l282_282583


namespace relay_race_athlete_orders_l282_282815

def athlete_count : ℕ := 4
def cannot_run_first_leg (athlete : ℕ) : Prop := athlete = 1
def cannot_run_fourth_leg (athlete : ℕ) : Prop := athlete = 2

theorem relay_race_athlete_orders : 
  ∃ (number_of_orders : ℕ), number_of_orders = 14 := 
by 
  -- Proof is omitted because it’s not required as per instructions.
  sorry

end relay_race_athlete_orders_l282_282815


namespace collinear_K_E_H_l282_282418

open EuclideanGeometry

-- Definitions of points and circles
variables {A B C D E F G H K : Point}
variables (circ_ABCDE : Circle)
variables (circ_BFC : Circle)
variables (circ_CGD : Circle)

-- Conditions
variables 
(h_circumscribed : circ_ABCDE.CircumscribedQuadrilateral A B C D)
(h_center_K : circ_ABCDE.center = K)
(h_intersection_E : LinesIntersectAt A C B D E)
(h_intersection_F : LinesIntersectAt A B C D F)
(h_intersection_G : LinesIntersectAt B C D A G)
(h_second_intersection_H : SecondIntersection circ_BFC circ_CGD H C)

-- Goal: Prove that K, E, and H are collinear
theorem collinear_K_E_H :
  Collinear3 K E H :=
sorry

end collinear_K_E_H_l282_282418


namespace count_two_digit_perfect_squares_divisible_by_four_l282_282254

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l282_282254


namespace sqrt_meaningful_range_l282_282308

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 3)) → x ≥ 3 := sorry

end sqrt_meaningful_range_l282_282308


namespace envelopes_with_conditions_l282_282392

theorem envelopes_with_conditions :
  let cards := {1, 2, 3, 4, 5, 6}
  let envelopes := {E1, E2, E3}
  ∃ f : (cards → envelopes),
  (∀ (card1 card2 : nat), card1 ∈ cards → card2 ∈ cards → f card1 = f card2 → (card1 = 1 → card2 = 2) ∧ (card1 = 2 → card2 = 1)) ∧
  (∀ (e : envelopes), ∃! l : list nat, (∀ (c : nat), c ∈ l → c ∈ cards) ∧ (list.length l = 2) ∧ (∀ (c : nat), c ∈ l → f c = e)) →
  (∀ f : (cards → envelopes), (∃ (isValid : ∀ e, (∃! l : list nat, (∀ c, c ∈ l → c ∈ cards) ∧ (list.length l = 2) ∧ (∀ (c : nat), c ∈ l → f c = e)))) → 
  int.ofNat (finset.card ((equiv.perm.symm signed_circle_perm).toFun cards (equiv.perm.support_signed_circle_perm.symm env assignments)⁻¹ ‘‘ assignments.toFinset.ι_unstripped env assignments [E1, E2, E3])) = 18)) :=
begin
  sorry
end

end envelopes_with_conditions_l282_282392


namespace local_minimum_of_reflected_function_l282_282062

noncomputable def f : ℝ → ℝ := sorry

theorem local_minimum_of_reflected_function (f : ℝ → ℝ) (x_0 : ℝ) (h1 : x_0 ≠ 0) (h2 : ∃ ε > 0, ∀ x, abs (x - x_0) < ε → f x ≤ f x_0) :
  ∃ δ > 0, ∀ x, abs (x - (-x_0)) < δ → -f (-x) ≥ -f (-x_0) :=
sorry

end local_minimum_of_reflected_function_l282_282062


namespace prove_additional_minutes_needed_l282_282354

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l282_282354


namespace trigonometric_identity_l282_282946

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + Real.pi / 3) = 1 / 3) :
  Real.sin (5 * Real.pi / 3 - x) - Real.cos (2 * x - Real.pi / 3) = 4 / 9 :=
by
  sorry

end trigonometric_identity_l282_282946


namespace sum_computation_l282_282124

noncomputable def ceil_minus_floor (x : ℝ) : ℝ :=
  if x ≠ ⌊x⌋ then 1 else 0

def is_power_of_three (n : ℕ) : Prop :=
  ∃ (j : ℕ), 3^j = n

theorem sum_computation :
  (∑ k in Finset.range 501, k * (ceil_minus_floor (Real.log k / Real.log 3))) = 124886 :=
by
  sorry

end sum_computation_l282_282124


namespace count_two_digit_perfect_squares_divisible_by_four_l282_282253

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l282_282253


namespace phi_value_g_function_range_l282_282230

theorem phi_value (ϕ : ℝ) (h_condition : 0 < ϕ ∧ ϕ < π) (h_point : (f : ℝ → ℝ) := λ x, (1/2) * Real.cos (2 * x - ϕ), f (π / 6) = 1 / 2) : ϕ = π / 3 :=
sorry

theorem g_function_range (g : ℝ → ℝ) (h_transform : g = λ x, (1/2) * Real.cos (4 * x - π / 3)) : Set.range (λ x, g x) (Set.Icc 0 (π / 4)) = Set.Icc (-1 / 4) (1 / 2) :=
sorry

end phi_value_g_function_range_l282_282230


namespace tetrahedron_midpoint_segments_intersect_and_bisect_l282_282709

-- Assume a tetrahedron ABCD
variables {A B C D : Point}

-- Define the midpoints of edges AB, CD, AC, BD, AD, and BC
def M : Point := midpoint A B
def N : Point := midpoint C D
def P : Point := midpoint A C
def Q : Point := midpoint B D
def R : Point := midpoint A D
def S : Point := midpoint B C

-- The line segments connecting the midpoints
def MN : Line := line_through M N
def PQ : Line := line_through P Q
def RS : Line := line_through R S

-- Goal: Prove that MN, PQ, and RS intersect at one point and are bisected by that point
theorem tetrahedron_midpoint_segments_intersect_and_bisect :
  ∃ O : Point, (O ∈ MN) ∧ (O ∈ PQ) ∧ (O ∈ RS) ∧ (segmented_by O MN) ∧ (segmented_by O PQ) ∧ (segmented_by O RS) :=
sorry

end tetrahedron_midpoint_segments_intersect_and_bisect_l282_282709


namespace negation_proof_l282_282758

variable {f : ℕ → ℕ}

theorem negation_proof (h : ∀ n : ℕ, f(n) ∈ ℕ ∧ f(n) > n) : 
  (∃ n₀ : ℕ, f(n₀) ∉ ℕ ∨ f(n₀) ≤ n₀) :=
sorry

end negation_proof_l282_282758


namespace sqrt_x_minus_3_domain_l282_282311

theorem sqrt_x_minus_3_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_domain_l282_282311


namespace sqrt_meaningful_for_x_l282_282320

theorem sqrt_meaningful_for_x (x : ℝ) : (∃ r : ℝ, r = real.sqrt (x - 3)) ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_for_x_l282_282320


namespace number_of_trips_l282_282725

theorem number_of_trips
  (fill_time : ℕ) 
  (drive_time_one_way : ℕ) 
  (total_moving_time_hrs : ℕ)
  (fill_time_eq : fill_time = 15)
  (drive_time_one_way_eq : drive_time_one_way = 30)
  (total_moving_time_hrs_eq : total_moving_time_hrs = 7) :
  let total_moving_time := total_moving_time_hrs * 60 in
  let round_trip_time := fill_time + 2 * drive_time_one_way in
  total_moving_time / round_trip_time = 5 := 
by
  sorry

end number_of_trips_l282_282725


namespace solve_log_equation_l282_282422

theorem solve_log_equation :
  ∀ x : ℝ, (log x)^2 - 2 * (log x) - 3 = 0 ↔ x = 0.1 ∨ x = 1000 :=
by
  sorry

end solve_log_equation_l282_282422


namespace binomial_coeff_10_5_factorial_of_binomial_10_5_minus_5_l282_282126

open Nat

-- Define binomial coefficient
def binomial (n k : ℕ) : ℕ := n.choose k

-- Define factorial
def factorial (n : ℕ) : ℕ := n!

-- Statement 1: Prove binomial coefficient
theorem binomial_coeff_10_5 : binomial 10 5 = 252 := by
  sorry

-- Statement 2: Prove the factorial operation after subtracting 5 from binomial coefficient
theorem factorial_of_binomial_10_5_minus_5 : factorial (binomial 10 5 - 5) = 247! := by
  sorry

end binomial_coeff_10_5_factorial_of_binomial_10_5_minus_5_l282_282126


namespace area_sequence_limit_l282_282074
noncomputable def sequence_limit (m : ℝ) : ℝ :=
  let a : ℕ → ℝ := λ k, m * (√2 / 2) ^ k
  let r : ℕ → ℝ := λ k, 1/2 * a k
  let A : ℕ → ℝ := λ k, π * (r k)^2
  let S : ℕ → ℝ := λ n, ∑ k in finset.range n, A (k+1)
  lim (λ n, S n)

theorem area_sequence_limit (m : ℝ) : sequence_limit m = (π * m^2) / 2 := by
  sorry

end area_sequence_limit_l282_282074


namespace sum_reciprocals_five_l282_282765

theorem sum_reciprocals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  1/x + 1/y = 5 :=
begin
  sorry
end

end sum_reciprocals_five_l282_282765


namespace compute_division_l282_282885

theorem compute_division : 0.182 / 0.0021 = 86 + 14 / 21 :=
by
  sorry

end compute_division_l282_282885


namespace ratio_DF_FE_l282_282671

noncomputable def midpoint (A C : Point) : Point := (A + C) / 2

variables {A B C D E F N : Point}
variables (AB BC : ℝ)

-- Given conditions
def AB_length := AB = 10
def BC_length := BC = 18
def midpoint_N := N = midpoint A C
def D_on_BC (BD BE : ℝ) := BD = 3 * BE ∧ BE + BD = BC -- D subdivides BC
def E_on_AB (BE : ℝ) := BE + (10 - BE) = AB -- E subdivides AB

-- Intersection point computation (F)
def F_intersection (t s : ℝ) (x : ℝ) :=
  t * ((x • B) + ((10 - x) • A)) / 10 + 
  (1 - t) * ((3 * x • C) + ((18 - 3 * x) • B)) / 18 = 
  s * ((A + C) / 2) + (1 - s) * A

-- Proof goal stating the ratio DF to FE
theorem ratio_DF_FE (x F : ℝ) : 
  AB_length → 
  BC_length → 
  midpoint_N → 
  D_on_BC (3 * x) x →
  E_on_AB x →
  ∃ F, ∃ t s, F_intersection t s x → (1-t)/t = 1/3 :=
by
  intros hAB hBC hN hD hE hF
  sorry

end ratio_DF_FE_l282_282671


namespace problem1_problem2_l282_282924

variable (a b c d x : ℝ)

def det2x2 (a b c d : ℝ) := a * d - b * c

theorem problem1 : det2x2 5 6 7 8 = -2 := by
  have h1 : det2x2 5 6 7 8 = 5 * 8 - 6 * 7 := rfl
  have h2 : 5 * 8 - 6 * 7 = 40 - 42 := rfl
  have h3 : 40 - 42 = -2 := by norm_num
  exact (eq.trans h1 (eq.trans h2 h3))

theorem problem2 (h : x^2 - 3 * x + 1 = 0) : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = 6 * x + 1 := by
  have h1 : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = (x + 1) * (x - 1) - (3 * x) * (x - 2) := rfl
  have h2 : (x + 1) * (x - 1) = x^2 - 1 := by ring
  have h3 : (3 * x) * (x - 2) = 3 * x * x - 3 * x * 2 := by ring
  have h4 : 3 * x * x - 3 * x * 2 = 3 * x^2 - 6 * x := by ring
  have h5 : det2x2 (x + 1) (3 * x) (x - 2) (x - 1) = (x^2 - 1) - (3 * x^2 - 6 * x) := by
    rw [h1, h2, h4]
  have h6 : (x^2 - 1) - (3 * x^2 - 6 * x) = -2 * x^2 + 6 * x - 1 := by ring
  have h7 : -2 * x^2 + 6 * x - 1 = -2 * (x^2 - 3 * x) - 1 := by ring
  rw [h7]
  have h8 : x^2 - 3 * x = -1 := by linarith
  rw [h8]
  norm_num
  sorry

end problem1_problem2_l282_282924


namespace perfect_squares_two_digit_divisible_by_4_count_l282_282266

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l282_282266


namespace sqrt_meaningful_range_l282_282307

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 3)) → x ≥ 3 := sorry

end sqrt_meaningful_range_l282_282307


namespace measure_of_angle_a_in_decagon_is_18_degrees_l282_282743

theorem measure_of_angle_a_in_decagon_is_18_degrees 
  (O A B : Point)
  (h1 : regular_polygon O 10)
  (h2 : dist O A = dist O B)
  (triangle_isosceles : is_isosceles_triangle O A B) : 
  angle_measure A O B = 18 :=
sorry

end measure_of_angle_a_in_decagon_is_18_degrees_l282_282743


namespace number_of_good_colorings_l282_282906

theorem number_of_good_colorings (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) : 
  ∃ (good_colorings : ℕ), good_colorings = 6 * (2^n - 4 + 4 * 2^(m-2)) :=
sorry

end number_of_good_colorings_l282_282906


namespace units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l282_282196

/-- Find the units digit of the largest power of 2 that divides into (2^5)! -/
theorem units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial : ∃ d : ℕ, d = 8 := by
  sorry

end units_digit_of_largest_power_of_two_dividing_2_pow_5_factorial_l282_282196


namespace correct_conclusions_l282_282770

-- Definitions for the events and probabilities
variable {Ω : Type*} [ProbabilitySpace Ω]
variables (A1 A2 A3 B : Event Ω)

-- Given conditions
variable [h1 : Probabilityable A1 = 5/10]
variable [h2 : Probabilityable A2 = 2/10]
variable [h3 : Probabilityable A3 = 3/10]
variable [h4 : Probabilityable B = 9/22]
variable [h5 : Probability B A1 = 2/5]
variable [h6 : Independent B A1]
variable [h7 : MutuallyExclusive A1 A2 A3]

-- Goal to prove
theorem correct_conclusions : (A1 = 1/2) ∧ (A2 = 1/5) ∧ (A3 = 3/10) ∧ (B = 9/22) :=
by
-- Proof not required, so we skip it with sorry
sorry

end correct_conclusions_l282_282770


namespace difference_of_two_smallest_integers_l282_282175

/--
The difference between the two smallest integers greater than 1 which, when divided by any integer 
\( k \) in the range from \( 3 \leq k \leq 13 \), leave a remainder of \( 2 \), is \( 360360 \).
-/
theorem difference_of_two_smallest_integers (n m : ℕ) (h_n : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → n % k = 2) (h_m : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → m % k = 2) (h_smallest : m > n) :
  m - n = 360360 :=
sorry

end difference_of_two_smallest_integers_l282_282175


namespace definite_integral_value_l282_282511

noncomputable def integral_value (a b : ℝ) (f : ℝ → ℝ) : ℝ :=
  (∫ x in a..b, f x)

def integrand (x : ℝ) : ℝ :=
  (x + 2)^3 * (Real.log (x + 2))^2

theorem definite_integral_value :
  integral_value (-1) 0 integrand = 4 * (Real.log 2)^2 - 2 * Real.log 2 + 15 / 32 :=
by
  sorry

end definite_integral_value_l282_282511


namespace sum_of_first_16_terms_l282_282935

theorem sum_of_first_16_terms (a : ℕ → ℝ) (h_arith_seq : ∀ n m, a (n + 1) - a n = a (m + 1) - a m) 
  (h_condition : a 2 + a 15 = 1) : (∑ i in Finset.range 16, a (i + 1)) = 8 := 
sorry

end sum_of_first_16_terms_l282_282935


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282278

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282278


namespace rectangle_perimeter_l282_282710

theorem rectangle_perimeter
  (A B C D W X Y Z : Type)
  (WA WB : ℝ) (AB BC CD DA : ℝ)
  (midpoint1 : midpoint A B W)
  (midpoint2 : midpoint B C X)
  (midpoint3 : midpoint C D Y)
  (midpoint4 : midpoint D A Z)
  (rhombus_WXYZ : rhombus W X Y Z)
  (WA_eq : WA = 20)
  (XB_eq : XB = 30)
  (diagonal_WY : WY = 50)
  : perimeter_ABCD = 200 :=
by
  sorry

end rectangle_perimeter_l282_282710


namespace obtain_13121_not_obtain_12131_l282_282412

open Nat

def trans (m n : ℕ) : ℕ := m + n + m * n

def can_obtain_from_1_and_2 (x : ℕ) : Prop :=
  ∃ a b : ℕ, a ≥ 2 ∧ b ≥ 3 ∧ a * b - 1 = x

theorem obtain_13121 : can_obtain_from_1_and_2 13121 :=
by
  use 2, 6561
  simp [can_obtain_from_1_and_2]
  sorry

theorem not_obtain_12131 : ¬ can_obtain_from_1_and_2 12131 :=
by
  intro h
  simp [can_obtain_from_1_and_2] at h
  obtain ⟨a, b, ha, hb, hab_eq⟩ := h
  refine absurd (hab_eq.symm ▸ _) sorry
  sorry

end obtain_13121_not_obtain_12131_l282_282412


namespace gain_percent_l282_282985

variable (C S : ℝ)

theorem gain_percent (h : 50 * C = 28 * S) : ((S - C) / C) * 100 = 78.57 := by
  sorry

end gain_percent_l282_282985


namespace fraction_S_over_T_l282_282890

def S : ℚ := ∑ k in finset.range 50, (1 : ℚ) / ((2 * k + 1) * (2 * k + 2))
def T : ℚ := ∑ k in finset.range 50, (1 : ℚ) / ((50 + k + 1) * (100 - k + 1))

theorem fraction_S_over_T : S / T = 151 / 4 := by
  sorry

end fraction_S_over_T_l282_282890


namespace solve_system_l282_282580

theorem solve_system (n : ℕ) (x : Fin n → ℝ) (h : 2 ≤ n) :
  (∀ (i : Fin n), let i_succ := if i.val + 1 < n then i.val + 1 else 0 
   in ( max (i.val + 1 : ℝ) (x ⟨i_succ, sorry⟩) = (i.val + 1 : ℝ) * x ⟨(i.val + 2) % n, sorry⟩ ) → 
   ∀ (j : Fin n), x j = 1 :=
begin
  sorry
end

end solve_system_l282_282580


namespace sum_geometric_sequence_first_eight_terms_l282_282186

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l282_282186


namespace candies_per_packet_l282_282867

-- Define the given conditions
def monday_to_friday_candies_per_day := 2
def weekend_candies_per_day := 1
def weekdays := 5
def weekends := 2
def weeks := 3
def packets := 2

-- Calculate the number of candies Bobby eats in a week
def candies_per_week := (monday_to_friday_candies_per_day * weekdays) + (weekend_candies_per_day * weekends)

-- Calculate the total number of candies Bobby eats in the given 3 weeks
def total_candies_in_3_weeks := candies_per_week * weeks

-- Divide the total number of candies by the number of packets to find the candies per packet
theorem candies_per_packet : total_candies_in_3_weeks / packets = 18 := 
by
  -- Adding the proof placeholder
  sorry

end candies_per_packet_l282_282867


namespace train_crossing_time_l282_282620

/-- How long does a train that is 370 m long and running at the speed of 135 km/hr take to cross a bridge that is 625 m in length? -/
theorem train_crossing_time : 
  let length_of_train := 370 -- in meters
  let length_of_bridge := 625 -- in meters
  let speed_of_train_kmh := 135 -- in km/hr
  let speed_of_train_ms := speed_of_train_kmh * 1000 / 3600 -- convert to m/s
  let total_distance := length_of_train + length_of_bridge -- in meters
  let time_to_cross := total_distance / speed_of_train_ms -- in seconds
  abs (time_to_cross - 26.53) < 0.01 := -- approximately 26.53 seconds
by
  sorry

end train_crossing_time_l282_282620


namespace ibrahim_savings_l282_282629

theorem ibrahim_savings (mp3_cost cd_cost father_contribution lacks_savings total : ℕ) 
  (H1 : mp3_cost = 120)
  (H2 : cd_cost = 19)
  (H3 : father_contribution = 20)
  (H4 : lacks_savings = 64)
  (H5 : total = mp3_cost + cd_cost) :
  (father_contribution + lacks_savings) + 55 = total :=
by
  rw [H1, H2, H3, H4]
  -- These lines use the given conditions to obtain the total cost
  have H6 : total = 120 + 19 := by rw [H1, H2]
  -- Substitute the values into the equation
  show 20 + 64 + 55 = 139 := by 
    linarith
  sorry

end ibrahim_savings_l282_282629


namespace distance_le_25_over_8_l282_282643

open Real
noncomputable theory

def within_rectangle (p : ℝ × ℝ) (width height : ℝ) := 
  0 ≤ p.1 ∧ p.1 ≤ width ∧ 0 ≤ p.2 ∧ p.2 ≤ height

theorem distance_le_25_over_8 :
  ∀ (P1 P2 P3 P4 : ℝ × ℝ),
    within_rectangle P1 4 3 →
    within_rectangle P2 4 3 →
    within_rectangle P3 4 3 →
    within_rectangle P4 4 3 →
    ∃ (i j : ℕ), i ≠ j ∧ dist (i.th P1 P2 P3 P4) (j.th P1 P2 P3 P4) ≤ 25 / 8 := sorry

end distance_le_25_over_8_l282_282643


namespace value_of_x_l282_282295

variable (w x y : ℝ)

theorem value_of_x 
  (h_avg : (w + x) / 2 = 0.5)
  (h_eq : (7 / w) + (7 / x) = 7 / y)
  (h_prod : w * x = y) :
  x = 0.5 :=
sorry

end value_of_x_l282_282295


namespace common_divisors_60_90_l282_282625

theorem common_divisors_60_90 :
  ∃ (count : ℕ), 
  (∀ d, d ∣ 60 ∧ d ∣ 90 ↔ d ∈ {1, 2, 3, 5, 6, 10, 15, 30}) ∧ 
  count = 8 :=
by
  sorry

end common_divisors_60_90_l282_282625


namespace periodicity_a_n_rational_decimal_main_l282_282692

def units_digit (n : ℕ) : ℕ := n % 10

def sum_of_squares (n : ℕ) : ℕ := (Finset.range (n + 1)).sum (λ i => i^2)

def a_n (n : ℕ) : ℕ := units_digit (sum_of_squares n)

theorem periodicity_a_n : ∀ n : ℕ, a_n (n + 20) = a_n n :=
  sorry

theorem rational_decimal (a : ℕ → ℕ) (h : ∀ n : ℕ, a (n + 20) = a n) : 
  ∃ (r : ℚ), 0.a_1 a_2 a_3 ... = r := 
  sorry

theorem main : ∃ r : ℚ, 0.a_1 a_2 ... = r :=
  rational_decimal a_n periodicity_a_n

end periodicity_a_n_rational_decimal_main_l282_282692


namespace intersection_of_sets_l282_282584

open Set

theorem intersection_of_sets (A B : Set ℤ) (hA : A = {-1, 0, 1}) (hB : B = {0, 1, 2}) : A ∩ B = {0, 1} :=
by
  rw [hA, hB]
  simp
  sorry

end intersection_of_sets_l282_282584


namespace derivative_at_1_l282_282222

variables {f : ℝ → ℝ} {x : ℝ} (a : ℝ) (b : ℝ)

-- Conditions
def tangent_at_1 (f : ℝ → ℝ) := (∀ x, f x = a * x + b) → (a = e ∧ b = -e) 
def point_on_graph := f 1 = e - e 

-- The statement to be proved
theorem derivative_at_1 (hf : ∀ x, tangent_at_1 f) (h : point_on_graph) : deriv f 1 = e :=
sorry

end derivative_at_1_l282_282222


namespace number_of_adults_l282_282863

theorem number_of_adults (
  (cost_per_meal : ℕ) (children_count : ℕ) (total_bill : ℕ)
  (h_cost : cost_per_meal = 3)
  (h_children : children_count = 5)
  (h_total_bill : total_bill = 21)
) : ∃ A : ℕ, total_bill = 3 * A + (children_count * cost_per_meal) ∧ A = 2 :=
by {
  use 2,
  split,
  { rw [h_cost, h_children, h_total_bill],
    norm_num },
  { refl }
}

end number_of_adults_l282_282863


namespace number_of_distinct_prime_divisors_of_1170_l282_282978

theorem number_of_distinct_prime_divisors_of_1170 : 
    (∃ (p : List ℕ), p = [2, 3, 5, 13] ∧ (∀ x ∈ p, Nat.Prime x) ∧ (List.prod p^p.count(p)) = 1170) → 
    List.length p = 4 :=
by sorry

end number_of_distinct_prime_divisors_of_1170_l282_282978


namespace P_sufficient_but_not_necessary_for_Q_l282_282240

def P (x : ℝ) : Prop := (x - 3) * (x + 1) > 0
def Q (x : ℝ) : Prop := x^2 - 2 * x + 1 > 0

theorem P_sufficient_but_not_necessary_for_Q : 
  (∀ x : ℝ, P x → Q x) ∧ ¬ (∀ x : ℝ, Q x → P x) :=
by 
  sorry

end P_sufficient_but_not_necessary_for_Q_l282_282240


namespace trajectory_of_X_is_star_shape_l282_282577

noncomputable def isTrajectoryStarShaped 
  (n : ℕ) (hn : n ≥ 5) (O : Point) (A B : Point) 
  (P : Polygon) (hP : P.is_regular n ∧ P.is_center O ∧ P.has_vertices [A, B]) 
  (XYZ OAB : Triangle) (hCongruent : XYZ ≅ OAB) 
  (initially : XYZ = OAB) : 
  Prop :=
  ∃ trajectory, 
  (∀ t ∈ trajectory, t.is_star_shaped) ∧ 
  (∀ y z ∈ P.perimeter, y ≠ z → X ∈ P.interior ∧ 
   y.z_moves_everywhere → XYZ.moves_as_specified)

theorem trajectory_of_X_is_star_shape 
  (n : ℕ) 
  (hn : n ≥ 5)
  (O A B : Point)
  (P : Polygon) 
  (hP : P.is_regular n ∧ P.is_center O ∧ P.has_vertices [A, B])
  (XYZ OAB : Triangle)
  (hCongruent : XYZ ≅ OAB)
  (initially : XYZ = OAB) : 
  isTrajectoryStarShaped n hn O A B P hP XYZ OAB hCongruent initially := by 
  sorry

end trajectory_of_X_is_star_shape_l282_282577


namespace sum_WY_condition_l282_282533

-- Define the set of integers
def S : set ℕ := {1, 2, 3, 5}

-- Define the conditions and the question as a theorem
theorem sum_WY_condition (W X Y Z : ℕ) (hW : W ∈ S) (hX : X ∈ S) (hY : Y ∈ S) (hZ : Z ∈ S)
  (h_diff : W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧ X ≠ Y ∧ X ≠ Z ∧ Y ≠ Z)
  (h_frac : (W : ℚ) / X + (Y : ℚ) / Z = 3) : W + Y = 6 := 
sorry

end sum_WY_condition_l282_282533


namespace recommendation_plans_l282_282003

variable (A B C D : Type)
variable (SchoolA SchoolB SchoolC : Type)
variable (students : List A)
variable (schools : List B)
variable (studentA : A)

-- Number of different recommendation plans
theorem recommendation_plans (h1 : students = [A, B, C, D])
  (h2 : schools = [SchoolA, SchoolB, SchoolC])
  (h_not_A_A : ¬(studentA = A ∧ SchoolA = A))
  : (number_of_recommendation_plans students schools = 24) :=
sorry

end recommendation_plans_l282_282003


namespace staircase_steps_sum_digits_l282_282865

open Nat

def ceiling_div (a b : ℕ) : ℕ :=
  if a % b = 0 then a / b else a / b + 1

theorem staircase_steps_sum_digits : 
  ∃ (n_values : List ℕ), 
    (∀ n ∈ n_values, ceiling_div n 3 - ceiling_div n 6 = 11)
    → ∑ n in n_values = 342
    ∧ (Nat.digits 10 342).sum = 9 :=
  sorry

end staircase_steps_sum_digits_l282_282865


namespace probability_correct_l282_282113

noncomputable
def probability_all_letters_from_midnight : ℚ :=
let p_road : ℚ := 2 / 6 in
let p_lights : ℚ := 1 / 20 in
let p_time : ℚ := 1 / 4 in
p_road * p_lights * p_time

theorem probability_correct :
  probability_all_letters_from_midnight = 1 / 240 :=
by
  sorry

end probability_correct_l282_282113


namespace zadam_solution_l282_282562

noncomputable def ZadamProbability 
  (p : ℕ → ℕ → ℚ)
  (satisfies_sum : ∀ (m : Fin 7 → ℕ), (∑ i, (i.succ * m i).toNat) = 35 → m 4 = 1) :=
  ∑ m in {m : Fin 7 → ℕ | (∑ i, (i.succ * m i).toNat) = 35}, p 4 1 = 4/5

theorem zadam_solution :
  ZadamProbability
  (λ i k, (2 ^ i - 1)/2 ^ (i * k))
  (fun m => satisfies_sum m) := sorry

end zadam_solution_l282_282562


namespace cars_sold_first_day_l282_282072

theorem cars_sold_first_day (c_2 c_3 : ℕ) (total : ℕ) (h1 : c_2 = 16) (h2 : c_3 = 27) (h3 : total = 57) :
  ∃ c_1 : ℕ, c_1 + c_2 + c_3 = total ∧ c_1 = 14 :=
by
  sorry

end cars_sold_first_day_l282_282072


namespace min_distance_from_circle_to_line_l282_282217

noncomputable def min_distance_point_to_line : ℝ :=
  let circle_center := (0 : ℝ, 0 : ℝ)
  let radius := 1
  let line_coefficients := (1 : ℝ, 1 : ℝ, -2 * Real.sqrt 2)
  let distance := (Real.abs (line_coefficients.2))
                  / (Real.sqrt ((line_coefficients.1)^2 + (line_coefficients.2)^2))
  distance - radius

theorem min_distance_from_circle_to_line : min_distance_point_to_line = 1 :=
sorry

end min_distance_from_circle_to_line_l282_282217


namespace eccentricity_of_ellipse_l282_282762

-- We define the semi-focal distance, a, b, e and the intersection point where y = 2x
variables (a b c : ℝ) (h1 : a > b) (h2 : b > 0)
def ellipse_eq_1 (x y : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def line_eq (x y : ℝ) : Prop := y = 2 * x
def intersection_point (x : ℝ) := (x = c) ∧ (2 * x = 2 * c)

-- We start the theorem stating that eccentricity e is equal to sqrt(2) - 1
theorem eccentricity_of_ellipse : 
  ∀ (a b c e : ℝ) (h1: a > b) (h2: b > 0),
  (c^2 / a^2 + 4 * c^2 / (a^2 - c^2) = 1) → e = sqrt(2) - 1 :=
by
  intros a b c e h1 h2 h_eq
  sorry

end eccentricity_of_ellipse_l282_282762


namespace machine_work_time_l282_282105

theorem machine_work_time
  (shirts_yesterday : ℕ)
  (rate_yesterday : ℕ)
  (downtime_yesterday : ℕ)
  (shirts_yesterday = 9)
  (rate_yesterday = 1/2)
  (downtime_yesterday = 20)
  : (shirts_yesterday * 2 + downtime_yesterday = 38) :=
by
  sorry

end machine_work_time_l282_282105


namespace z_in_first_quadrant_l282_282958

namespace ComplexQuadrant

def z : ℂ := (Complex.I) / ((1 : ℂ) + Complex.I)

def isFirstQuadrant (z : ℂ) : Prop :=
  0 < z.re ∧ 0 < z.im

theorem z_in_first_quadrant : isFirstQuadrant z := by
  sorry

end ComplexQuadrant

end z_in_first_quadrant_l282_282958


namespace find_x_values_l282_282162

theorem find_x_values (x : ℝ) (h : x > 9) : 
  (sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) ↔ (x ≥ 18) :=
sorry

end find_x_values_l282_282162


namespace evaluate_sum_l282_282905

-- Define the sum expression
def sum_expression (n : ℕ) : ℚ :=
  ∑ k in (finset.range n).map nat.succ, (-1)^k * (k^3 + k^2 + 1) / ((k+1)! : ℚ)

-- The theorem statement
theorem evaluate_sum :
  sum_expression 50 = 126001 / 51! :=
sorry

end evaluate_sum_l282_282905


namespace primes_dividing_expression_l282_282156

theorem primes_dividing_expression (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) : 
  6 * p * q ∣ p^3 + q^2 + 38 ↔ (p = 3 ∧ (q = 5 ∨ q = 13)) := 
sorry

end primes_dividing_expression_l282_282156


namespace least_value_y_l282_282789

theorem least_value_y
  (h : ∀ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 → -3 ≤ y) : 
  ∃ y : ℝ, 5 * y ^ 2 + 7 * y + 3 = 6 ∧ y = -3 :=
by
  sorry

end least_value_y_l282_282789


namespace max_omega_translate_sine_l282_282775

theorem max_omega_translate_sine (ω : ℝ) (hω : ω > 0) :
  (∀ x : ℝ, x ∈ Icc (-π / 3) (π / 4) → 2 * sin (ω * (x - π / (3 * ω)) + π / 3) = 2 * sin (ω * x)) →
  (∀ x : ℝ, x ∈ Icc (-π / 3) (π / 4) → deriv (λ x, 2 * sin (ω * x)) x ≥ 0) →
  ω ≤ 2 :=
by
  sorry

end max_omega_translate_sine_l282_282775


namespace range_of_m_l282_282638

theorem range_of_m {f : ℝ → ℝ} (h : ∀ x, f x = x^2 - 6*x - 16)
  {a b : ℝ} (h_domain : ∀ x, 0 ≤ x ∧ x ≤ a → ∃ y, f y ≤ b) 
  (h_range : ∀ y, -25 ≤ y ∧ y ≤ -16 → ∃ x, f x = y) : 3 ≤ a ∧ a ≤ 6 := 
sorry

end range_of_m_l282_282638


namespace angle_PSU_in_triangle_PQR_l282_282669

theorem angle_PSU_in_triangle_PQR (P Q R S T U : Type)
  [Triangle P Q R] [Foot S P Q R] [Circumcenter T P Q R] [DiameterEnd U P T] 
  (h1 : ∠P R Q = 60) 
  (h2 : ∠R Q P = 80) 
  (h3 : ∠P S Q = 90 - ∠P R Q) 
  (h4 : ∠P Q T = 2 * ∠R Q P) 
  (h5 : ∠P T Q = (180 - ∠P Q T) / 2) :
  ∠P S U = 20 :=
begin
  sorry
end

end angle_PSU_in_triangle_PQR_l282_282669


namespace isosceles_triangle_base_length_l282_282092

theorem isosceles_triangle_base_length (x : ℝ) (h1 : 2 * x + 2 * x + x = 20) : x = 4 :=
sorry

end isosceles_triangle_base_length_l282_282092


namespace dice_sum_less_than_16_l282_282434

open Probability

def dice_sum_probability : ℚ :=
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6)
  let valid_outcomes := outcomes.filter (λ (t : ℕ × ℕ × ℕ), t.1 + t.2.1 + t.2.2 < 16)
  valid_outcomes.card / outcomes.card

theorem dice_sum_less_than_16 : dice_sum_probability = 103 / 108 :=
by
  sorry

end dice_sum_less_than_16_l282_282434


namespace hyperbola_equation_l282_282572

-- Definitions based on given conditions
def a := 2 * Real.sqrt 5
def pointA := (2 : ℝ, -5 : ℝ)

-- The Lean theorem statement for the hyperbola's equation
theorem hyperbola_equation (h₁ : a = 2 * Real.sqrt 5)
    (h₂ : ∃ k, ∀ y, k * y^2 = k y^2)
    (h₃ : ∀ (x y : ℝ), (x, y) = pointA → 
      ∃ b, (y^2 / 20) - (x^2 / b^2) = 1) :
    ∃ b, b^2 = 16 :=
sorry

end hyperbola_equation_l282_282572


namespace inradius_of_triangle_l282_282416

theorem inradius_of_triangle (p A : ℝ) (h₁ : p = 42) (h₂ : A = 105) : 
  ∃ r : ℝ, (2 * A = r * p) ∧ r = 5 :=
by
  -- First, we state the condition given the formula for area incorporating inradius
  have h_formula : 2 * A = r * p := by sorry
  -- Then, we can derive the inradius r given the values of p and A specified by the conditions
  existsi (5 : ℝ)
  split
  exact h_formula
  exact sorry

end inradius_of_triangle_l282_282416


namespace probability_both_in_picture_l282_282397

noncomputable def lap_time_rachel : ℝ := 75
noncomputable def lap_time_robert : ℝ := 95
noncomputable def min_time : ℕ := 900
noncomputable def max_time : ℕ := 960

def in_picture_rachel (t : ℝ) : Prop :=
  ∃ k : ℤ, real.abs (t - k * lap_time_rachel) ≤ lap_time_rachel / 5

def in_picture_robert (t : ℝ) : Prop :=
  ∃ k : ℤ, real.abs (t - k * lap_time_robert) ≤ lap_time_robert / 5

theorem probability_both_in_picture :
  (∃ t ∈ set.Icc (min_time:ℝ) (max_time:ℝ), in_picture_rachel t ∧ in_picture_robert t) ↔
  (1 / 4) := 
sorry

end probability_both_in_picture_l282_282397


namespace geometric_sequence_ratio_l282_282576

-- Definitions and conditions from part a)
def q : ℚ := 1 / 2

def sum_of_first_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * (1 - q ^ n) / (1 - q)

def a_n (a1 : ℚ) (n : ℕ) : ℚ :=
  a1 * q ^ (n - 1)

-- Theorem representing the proof problem from part c)
theorem geometric_sequence_ratio (a1 : ℚ) : 
  (sum_of_first_n a1 4) / (a_n a1 3) = 15 / 2 := 
sorry

end geometric_sequence_ratio_l282_282576


namespace derivative_F_at_1_zero_l282_282547

-- Define the function F and its derivative
noncomputable def F (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x^3 - 1) + f (1 - x^3)

-- The theorem statement
theorem derivative_F_at_1_zero (f : ℝ → ℝ) (f' : ℝ → ℝ) (hf : ∀ x, has_deriv_at f (f' x) x) :
  deriv (F f) 1 = 0 :=
sorry

end derivative_F_at_1_zero_l282_282547


namespace problem1_problem2_l282_282607

-- Definitions related to the given problem
def polar_curve (ρ θ : ℝ) : Prop :=
  ρ^2 = 9 / (Real.cos θ^2 + 9 * Real.sin θ^2)

def standard_curve (x y : ℝ) : Prop :=
  x^2 / 9 + y^2 = 1

-- Proving the standard equation of the curve
theorem problem1 (ρ θ : ℝ) (h : polar_curve ρ θ) : ∃ x y, standard_curve x y :=
  sorry

-- Proving the perpendicular condition and its consequence
theorem problem2 (ρ1 ρ2 α : ℝ)
  (hA : polar_curve ρ1 α)
  (hB : polar_curve ρ2 (α + π/2))
  (perpendicular : ∀ (A B : (ℝ × ℝ)), A ≠ B → A.1 * B.1 + A.2 * B.2 = 0) :
  (1 / ρ1^2) + (1 / ρ2^2) = 10 / 9 :=
  sorry

end problem1_problem2_l282_282607


namespace greatest_possible_large_chips_l282_282431

theorem greatest_possible_large_chips : 
  ∃ s l p: ℕ, s + l = 60 ∧ s = l + 2 * p ∧ Prime p ∧ l = 28 :=
by
  sorry

end greatest_possible_large_chips_l282_282431


namespace play_area_l282_282100

theorem play_area (posts : ℕ) (space : ℝ) (extra_posts : ℕ) (short_posts long_posts : ℕ) (short_spaces long_spaces : ℕ) 
  (short_length long_length area : ℝ)
  (h1 : posts = 24) 
  (h2 : space = 5)
  (h3 : extra_posts = 6)
  (h4 : long_posts = short_posts + extra_posts)
  (h5 : 2 * short_posts + 2 * long_posts - 4 = posts)
  (h6 : short_spaces = short_posts - 1)
  (h7 : long_spaces = long_posts - 1)
  (h8 : short_length = short_spaces * space)
  (h9 : long_length = long_spaces * space)
  (h10 : area = short_length * long_length) :
  area = 675 := 
sorry

end play_area_l282_282100


namespace polynomial_A_l282_282989

theorem polynomial_A (A a : ℝ) (h : A * (a + 1) = a^2 - 1) : A = a - 1 :=
sorry

end polynomial_A_l282_282989


namespace joshuas_share_l282_282677

theorem joshuas_share (total amount : ℝ) (joshua_share : ℝ) (justin_share: ℝ) 
  (h1: total amount = 40) 
  (h2: joshua_share = 3 * justin_share) 
  (h3: total amount = joshua_share + justin_share) 
: joshua_share = 30 := 
by  sorry

end joshuas_share_l282_282677


namespace vasya_petya_distance_l282_282446

theorem vasya_petya_distance :
  ∀ (D : ℝ), 
    (3 : ℝ) ≠ 0 → (6 : ℝ) ≠ 0 →
    ((D / 3) + (D / 6) = 2.5) →
    ((D / 6) + (D / 3) = 3.5) →
    D = 12 := 
by
  intros D h3 h6 h1 h2
  sorry

end vasya_petya_distance_l282_282446


namespace problem1_problem2_l282_282883

-- Definition for the first proof problem
theorem problem1 (a b : ℝ) (h : a ≠ b) :
  (a^2 / (a - b) - b^2 / (a - b)) = a + b :=
by
  sorry

-- Definition for the second proof problem
theorem problem2 (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 0) :
  ((x^2 - 1) / ((x^2 + 2 * x + 1)) / (x^2 - x) / (x + 1)) = 1 / x :=
by
  sorry

end problem1_problem2_l282_282883


namespace number_of_students_in_class_l282_282404

theorem number_of_students_in_class :
  ∃ N : ℕ, (∀ ages : Fin N → ℕ, 
    (∑ i, ages i = 15 * N) ∧
    (∑ i in Finset.range 5, ages i = 70) ∧ 
    (∑ i in Finset.range 9, ages (i + 5) = 144) ∧ 
    (ages 14 = 11)) → N = 15 :=
sorry

end number_of_students_in_class_l282_282404


namespace scientific_notation_of_393000_l282_282737

theorem scientific_notation_of_393000 :
  ∃ a n : ℝ, (1 ≤ a ∧ a < 10) ∧ n ∈ ℤ ∧ 393000 = a * 10^n ∧ a = 3.93 ∧ n = 5 :=
by
  use 3.93
  use 5
  split
  { split
    { norm_num }
    { norm_num } }
  split
  { norm_num }
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }
  sorry

end scientific_notation_of_393000_l282_282737


namespace prob_B_independent_l282_282021

-- Definitions based on the problem's conditions
def prob_A := 0.7
def prob_A_union_B := 0.94

-- With these definitions established, we need to state the theorem.
-- The theorem should express that the probability of B solving the problem independently (prob_B) is 0.8.
theorem prob_B_independent : 
    (∃ (prob_B: ℝ), prob_A = 0.7 ∧ prob_A_union_B = 0.94 ∧ prob_B = 0.8) :=
by
    sorry

end prob_B_independent_l282_282021


namespace surface_area_of_box_l282_282830

def cube_edge_length : ℕ := 1
def cubes_required : ℕ := 12

theorem surface_area_of_box (l w h : ℕ) (h1 : l * w * h = cubes_required / cube_edge_length ^ 3) :
  (2 * (l * w + w * h + h * l) = 32 ∨ 2 * (l * w + w * h + h * l) = 38 ∨ 2 * (l * w + w * h + h * l) = 40) :=
  sorry

end surface_area_of_box_l282_282830


namespace arithmetic_sequence_general_formula_geometric_sequence_sum_l282_282936

noncomputable def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ ∑ i in finset.range 3, a (i + 1) = 0

noncomputable def geometric_sequence_sum_formula (a b : ℕ → ℤ) (B : ℕ → ℤ) : Prop :=
  b 1 = 2 * (a 1) ∧ b 2 = a 6 ∧ ∀ n, B n = 2 * (1 + 2^n) / 3

theorem arithmetic_sequence_general_formula (a : ℕ → ℤ) :
  arithmetic_sequence a → ∀ n, a n = 2 - n :=
sorry

theorem geometric_sequence_sum (a b : ℕ → ℤ) (B : ℕ → ℤ) :
  arithmetic_sequence a → geometric_sequence_sum_formula a b B →
  ∀ n, B n = 2 * (1 + 2^n) / 3 :=
sorry

end arithmetic_sequence_general_formula_geometric_sequence_sum_l282_282936


namespace trig_identity_proof_l282_282944

theorem trig_identity_proof (x : ℝ) (h : sin (x + π / 3) = 1 / 3) :
  sin (5 * π / 3 - x) - cos (2 * x - π / 3) = 4 / 9 :=
by {
  sorry
}

end trig_identity_proof_l282_282944


namespace candies_markus_l282_282384

theorem candies_markus (m k s : ℕ) (h_initial_m : m = 9) (h_initial_k : k = 5) (h_total_s : s = 10) :
  (m + s) / 2 = 12 := by
  sorry

end candies_markus_l282_282384


namespace KA_eq_KT_l282_282368

namespace ProofProblem

open EuclideanGeometry

variables (A B C O M N T X Y K : Point)
variables (ω : Circle)
variables (circumcircle_ABC : IsCircumcircle ω A B C)
variables (M_mid_AB : Midpoint M A B)
variables (N_mid_AC : Midpoint N A C)
variables (T_mid_arc_BC_no_A : IsArcMidpoint T ω B C)
variables (circumcircle_AMT : IsCircumcircle (circumcircle A M T) A M T)
variables (circumcircle_ANT : IsCircumcircle (circumcircle A N T) A N T)
variables (X_on_perp_bisector_AC : OnPerpendicularBisector X A C)
variables (Y_on_perp_bisector_AB : OnPerpendicularBisector Y A B)
variables (X_inside_ABC : InsideTriangle X A B C)
variables (Y_inside_ABC : InsideTriangle Y A B C)
variables (K_intersection_MN_XY : Intersection K (Line M N) (Line X Y))

theorem KA_eq_KT :
  KA = KT :=
sorry

end KA_eq_KT_l282_282368


namespace measure_angle_DAB_eq_30_l282_282652

-- Define that ABCDEF is a regular hexagon
variables (A B C D E F : Type) [regular_hexagon A B C D E F]

-- Each interior angle of a regular hexagon measures 120 degrees
axiom interior_angle_regular_hexagon (a b c : Type) [regular_hexagon a b c] : angle a b c = 120

-- Diagonal AD is drawn
variables (D : Type) [is_diagonal A D]

-- Define that measure of angle DAB is 30 degrees
theorem measure_angle_DAB_eq_30 : angle D A B = 30 :=
by sorry

end measure_angle_DAB_eq_30_l282_282652


namespace identify_odd_function_with_period_l282_282104

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def period (f : ℝ → ℝ) (T : ℝ) : Prop := T > 0 ∧ ∀ x, f (x + T) = f x

theorem identify_odd_function_with_period :
  ∃ (f : ℝ → ℝ), f = (λ x, cos (2 * x + π / 2)) ∧ 
  is_odd_function f ∧ period f π ∧
  (∀ g, (g = (λ x, sin (2 * x + π / 2)) → ¬ is_odd_function g ∨ ¬ period g π) ∧
       (g = (λ x, sin 2 * x + cos 2 * x) → ¬ is_odd_function g ∨ ¬ period g π) ∧
       (g = (λ x, sin x + cos x) → ¬ is_odd_function g ∨ ¬ period g π)) :=
sorry

end identify_odd_function_with_period_l282_282104


namespace max_k_value_max_min_sum_l282_282811

open Finset

namespace SubsetSum

def distinct_subset_sums (A : Finset ℕ) : Prop :=
  ∀ (B1 B2 : Finset ℕ), B1 ⊆ A → B2 ⊆ A → B1 ≠ B2 → 
  (B1.sum id ≠ B2.sum id)

theorem max_k_value 
  (A : Finset ℕ) 
  (hA : A ⊆ (range 17).filter (λ x, x > 0)) 
  (hA_card : A.card = k) 
  (hA_distinct : distinct_subset_sums A) : 
  k ≤ 5 := 
by
  sorry

theorem max_min_sum 
  (A : Finset ℕ) 
  (hA : A ⊆ (range 17).filter (λ x, x > 0)) 
  (hA_card : A.card = k) 
  (k_le_5 : k ≤ 5) 
  (hA_distinct : distinct_subset_sums A) : 
  16 ≤ A.sum id ∧ A.sum id ≤ 66 := 
by
  sorry

end SubsetSum

end max_k_value_max_min_sum_l282_282811


namespace sqrt_product_simplify_l282_282879

theorem sqrt_product_simplify (q : ℝ) (hq_pos : 0 < q) :
  sqrt (15 * q) * sqrt (20 * (q ^ 3)) * sqrt (12 * (q ^ 5)) = 60 * (q ^ 4) * sqrt q := by
  sorry

end sqrt_product_simplify_l282_282879


namespace zoo_visitors_l282_282391

theorem zoo_visitors (visitors_friday : ℕ) 
  (h1 : 3 * visitors_friday = 3750) :
  visitors_friday = 1250 := 
sorry

end zoo_visitors_l282_282391


namespace value_of_real_number_m_l282_282221

theorem value_of_real_number_m (m : ℝ) : (1 + complex.i : ℂ) ∈ {x : ℂ | x * x + (m * x : ℂ) + 2 = 0} → m = -2 :=
by
  sorry

end value_of_real_number_m_l282_282221


namespace percent_safflower_in_brand_b_l282_282831

theorem percent_safflower_in_brand_b :
  ∀ (S : ℝ),
  let A_millet := 0.40,
      A_sunflower := 0.60,
      B_millet := 0.65,
      mix_millet := 0.50,
      mix_A := 0.60,
      mix_B := 0.40 in
  (mix_A * A_millet) + (mix_B * B_millet) = mix_millet →
  S = 100 - 65 → 
  S = 35 := 
by
  intros S A_millet A_sunflower B_millet mix_millet mix_A mix_B h1 h2
  exact h2

end percent_safflower_in_brand_b_l282_282831


namespace can_form_set_l282_282045

-- Define each group of objects based on given conditions
def famous_movie_stars : Type := sorry
def small_rivers_in_our_country : Type := sorry
def students_2012_senior_class_Panzhihua : Type := sorry
def difficult_high_school_math_problems : Type := sorry

-- Define the property of having well-defined elements
def has_definite_elements (T : Type) : Prop := sorry

-- The groups in terms of propositions
def group_A : Prop := ¬ has_definite_elements famous_movie_stars
def group_B : Prop := ¬ has_definite_elements small_rivers_in_our_country
def group_C : Prop := has_definite_elements students_2012_senior_class_Panzhihua
def group_D : Prop := ¬ has_definite_elements difficult_high_school_math_problems

-- We need to prove that group C can form a set
theorem can_form_set : group_C :=
by
  sorry

end can_form_set_l282_282045


namespace length_PQ_l282_282032

theorem length_PQ (X Y M N P Q : ℝ) 
  (XY MN PQ : ℝ)
  (h_parallel : XY ∥ MN ∧ MN ∥ PQ)
  (hXY : XY = 120)
  (hMN : MN = 80) :
  PQ = 48 := 
sorry

end length_PQ_l282_282032


namespace player1_coins_l282_282426

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end player1_coins_l282_282426


namespace big_eight_league_total_games_l282_282730

-- Definitions of given conditions
def num_divisions : ℕ := 3
def num_teams_per_division : ℕ := 4
def games_within_division (n : ℕ) : ℕ := n * (n-1) / 2 * 2

def games_within_league (num_divisions num_teams_per_division : ℕ) : ℕ :=
  let within_division_games := games_within_division num_teams_per_division in
  num_divisions * within_division_games

def games_between_divisions (num_divisions num_teams_per_division : ℕ) : ℕ :=
  let num_teams := num_divisions * num_teams_per_division in
  num_teams_per_division * (num_teams - num_teams_per_division * num_divisions)

def total_games (num_divisions num_teams_per_division : ℕ) : ℕ :=
  games_within_league num_divisions num_teams_per_division +
  (games_between_divisions num_divisions num_teams_per_division / 2)  -- divide by 2 to remove double count

-- Proof Statement
theorem big_eight_league_total_games : total_games num_divisions num_teams_per_division = 228 := by 
  sorry

end big_eight_league_total_games_l282_282730


namespace number_of_candies_in_a_packet_l282_282871

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l282_282871


namespace reimbursement_proof_l282_282679

-- Define the rates
def rate_industrial_weekday : ℝ := 0.36
def rate_commercial_weekday : ℝ := 0.42
def rate_weekend : ℝ := 0.45

-- Define the distances for each day
def distance_monday : ℝ := 18
def distance_tuesday : ℝ := 26
def distance_wednesday : ℝ := 20
def distance_thursday : ℝ := 20
def distance_friday : ℝ := 16
def distance_saturday : ℝ := 12

-- Calculate the reimbursement for each day
def reimbursement_monday : ℝ := distance_monday * rate_industrial_weekday
def reimbursement_tuesday : ℝ := distance_tuesday * rate_commercial_weekday
def reimbursement_wednesday : ℝ := distance_wednesday * rate_industrial_weekday
def reimbursement_thursday : ℝ := distance_thursday * rate_commercial_weekday
def reimbursement_friday : ℝ := distance_friday * rate_industrial_weekday
def reimbursement_saturday : ℝ := distance_saturday * rate_weekend

-- Calculate the total reimbursement
def total_reimbursement : ℝ :=
  reimbursement_monday + reimbursement_tuesday + reimbursement_wednesday +
  reimbursement_thursday + reimbursement_friday + reimbursement_saturday

-- State the theorem to be proven
theorem reimbursement_proof : total_reimbursement = 44.16 := by
  sorry

end reimbursement_proof_l282_282679


namespace proof_problem_l282_282134

noncomputable def midpoint (A B C : Point) : Prop :=
  midpoint D B C

noncomputable def lies_on (E line_AD : Line) : Prop :=
  lies_on E line_AD

noncomputable def right_angle (A B C : Point) : Prop :=
  ∠ CEA = 90

noncomputable def angle_equality (A B C E : Point) : Prop :=
  ∠ ACE = ∠ B

theorem proof_problem
  (A B C D E : Point)
  (AD : Line)
  (h1 : midpoint D B C)
  (h2 : lies_on E AD)
  (h3 : right_angle A C E)
  (h4 : angle_equality A B C E)
  : AB = AC ∨ ∠ A = 90 :=
sorry

end proof_problem_l282_282134


namespace find_XY2_l282_282688

-- Define the triangle and lengths
variables {A B C T X Y : Type}
variables [Bearing A B] [Bearing A C] [Bearing T X] [Bearing T Y]
variables (BT CT BC : ℝ)
variables (TX TY XY : ℝ)

-- Given conditions
axiom BT_eq_CT : BT = 20
axiom BC_eq : BC = 24
axiom projection_eq : TX * TX + TY * TY + XY * XY = 1588

-- Prove the required condition
theorem find_XY2 : XY * XY = 617.56 :=
sorry

end find_XY2_l282_282688


namespace simplify_expression_l282_282805

theorem simplify_expression (a : ℝ) (h₁ : a ≠ 0) (h₂ : a ≠ -3) (h₃ : a ≠ -2/3) :
  ((9 - 4 * a^(-2)) / (3 * a^(-1/2) + 2 * a^(-3/2)) - (1 + a^(-1) - 6 * a^(-2)) / (a^(-1/2) + 3 * a^(-3/2)))^4 = 16 * a^2 :=
by sorry

end simplify_expression_l282_282805


namespace S_formula_l282_282763

def f (n : ℕ) : ℕ := (2 * n) * (2 * n) - 1

def S (n : ℕ) : ℕ := (∑ k in Finset.range(n + 1), 1 / f k)

theorem S_formula (n : ℕ) : S n = n / (2 * n + 1) :=
by sorry

end S_formula_l282_282763


namespace regular_hexagon_ratio_l282_282665

-- Define a regular hexagon
structure RegularHexagon (A B C D E F : Type) :=
  (eq_lengths : ∀ {x y : A ∪ B ∪ C ∪ D ∪ E ∪ F}, length x = length y)
  (interior_angle : ∀ {x y z: A ∪ B ∪ C ∪ D ∪ E ∪ F}, angle (x, y, z) = 120)

-- Predicate for the diagonals intersecting at G
def intersects_at (A B C D E F G : Type) (diag1 : line A F ∩ line C F = {G})
                                         (diag2 : line B D ∩ line D F = {G}) : Prop :=
  ∃ (G : Type), line A F ∩ line C F = {G} ∧ line B D ∩ line D F = {G}

-- The theorem statement
theorem regular_hexagon_ratio (A B C D E F G : Type)
  (hex : RegularHexagon A B C D E F)
  (intersect : intersects_at A B C D E F G) :
  area (hex.quad FEDG) / area (hex.triangle BCG) = 5 :=
sorry

end regular_hexagon_ratio_l282_282665


namespace prime_pq_2001_l282_282615

theorem prime_pq_2001 (p q : ℤ) (m : ℤ) (hp : nat.prime p) (hq : nat.prime q) (hne : p ≠ q)
  (hpm : p^2 - 2001 * p + m = 0) (hqm : q^2 - 2001 * q + m = 0) : p^2 + q^2 = 3996005 :=
by
  sorry

end prime_pq_2001_l282_282615


namespace minute_hand_length_l282_282753

noncomputable def length_minute_hand (A : ℝ) (θ : ℝ) : ℝ :=
  real.sqrt (2 * A / θ)

theorem minute_hand_length :
  length_minute_hand 15.274285714285716 (π / 3) ≈ 5.4 :=
by {
  sorry
}

end minute_hand_length_l282_282753


namespace sum_geometric_sequence_l282_282182

theorem sum_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 1/3) (h_r : r = 1/3) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in S_n = 3280/6561 :=
by
  sorry

end sum_geometric_sequence_l282_282182


namespace coefficient_x3y5_in_expansion_l282_282039

theorem coefficient_x3y5_in_expansion : 
  binomial 8 5 = 56 :=
by 
  sorry

end coefficient_x3y5_in_expansion_l282_282039


namespace hyperbola_eccentricity_l282_282238

-- Definitions based on the given conditions
variable (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
def hyperbola_eq : Prop := ∀ x y, (x^2 / a^2) - (y^2 / b^2) = 1

-- Hypothesis based on the symmetric points about the two asymptotes
variable (slope_asymptote : b = a)

-- Definition of the eccentricity
def eccentricity (c a : ℝ) := c / a

-- The main statement: proving the eccentricity of the hyperbola is sqrt 2
theorem hyperbola_eccentricity (h₃ : hyperbola_eq a b) (h₄ : slope_asymptote) : eccentricity (√2 * a) a = √2 := 
by
  sorry

end hyperbola_eccentricity_l282_282238


namespace player1_coins_l282_282425

theorem player1_coins (coin_distribution : Fin 9 → ℕ) :
  let rotations := 11
  let player_4_coins := 90
  let player_8_coins := 35
  ∀ player : Fin 9, player = 0 → 
    let player_1_coins := coin_distribution player
    (coin_distribution 3 = player_4_coins) →
    (coin_distribution 7 = player_8_coins) →
    player_1_coins = 57 := 
sorry

end player1_coins_l282_282425


namespace factorize_polynomial_l282_282537

theorem factorize_polynomial (a b : ℝ) : 
  a^3 * b - 9 * a * b = a * b * (a + 3) * (a - 3) :=
by sorry

end factorize_polynomial_l282_282537


namespace volume_increase_by_eight_l282_282419

noncomputable def sphere_volume (r : ℝ) : ℝ :=
  (4 / 3) * π * (r^3)

theorem volume_increase_by_eight (r : ℝ) :
  sphere_volume (2 * r) = 8 * sphere_volume r :=
by
  sorry

end volume_increase_by_eight_l282_282419


namespace scientific_notation_of_393000_l282_282740

theorem scientific_notation_of_393000 : 
  ∃ (a : ℝ) (n : ℤ), a = 3.93 ∧ n = 5 ∧ (393000 = a * 10^n) := 
by
  use 3.93
  use 5
  sorry

end scientific_notation_of_393000_l282_282740


namespace carter_family_children_l282_282731

variable (f m x y : ℕ)

theorem carter_family_children 
  (avg_family : (3 * y + m + x * y) / (2 + x) = 25)
  (avg_mother_children : (m + x * y) / (1 + x) = 18)
  (father_age : f = 3 * y)
  (simplest_case : y = x) :
  x = 8 :=
by
  -- Proof to be provided
  sorry

end carter_family_children_l282_282731


namespace total_distance_swam_l282_282703

theorem total_distance_swam (molly_swam_saturday : ℕ) (molly_swam_sunday : ℕ) (h1 : molly_swam_saturday = 400) (h2 : molly_swam_sunday = 300) : molly_swam_saturday + molly_swam_sunday = 700 := by 
    sorry

end total_distance_swam_l282_282703


namespace minimum_distance_is_10_div_sqrt_149_m_plus_n_equals_159_l282_282568

structure point :=
  (x : ℝ)
  (y : ℝ)

def garfield_start : point := ⟨0, 0⟩
def odie_start : point := ⟨25, 0⟩
def target : point := ⟨9, 12⟩

def garfield_speed : ℝ := 7
def odie_speed : ℝ := 10

noncomputable
def direction_vector (p1 p2 : point) : point :=
  ⟨p2.x - p1.x, p2.y - p1.y⟩

noncomputable
def magnitude (p : point) : ℝ :=
  real.sqrt (p.x^2 + p.y^2)

noncomputable
def unit_vector (p : point) : point :=
  let m := magnitude p in
  ⟨p.x / m, p.y / m⟩

noncomputable
def position (start : point) (speed : ℝ) (unit_dir : point) (t : ℝ) : point :=
  ⟨start.x + speed * unit_dir.x * t, start.y + speed * unit_dir.y * t⟩

noncomputable
def distance (p1 p2 : point) (t : ℝ) : ℝ :=
  magnitude ⟨p2.x - p1.x, p2.y - p1.y⟩

theorem minimum_distance_is_10_div_sqrt_149 :
  let garfield_dir := unit_vector (direction_vector garfield_start target),
      odie_dir := unit_vector (direction_vector odie_start target),
      d := λ t : ℝ, distance (position garfield_start garfield_speed garfield_dir t)
                            (position odie_start odie_speed odie_dir t)
  in ∃ t : ℝ, d t = 10 / real.sqrt 149 :=
sorry

theorem m_plus_n_equals_159 : 10 + 149 = 159 := by norm_num

end minimum_distance_is_10_div_sqrt_149_m_plus_n_equals_159_l282_282568


namespace average_scores_equal_l282_282930

theorem average_scores_equal :
  let male_scores := [92, 89, 93, 90]
  let female_scores := [92, 88, 93, 91]
  (male_scores.sum / male_scores.length = female_scores.sum / female_scores.length) :=
by
  let male_scores := [92, 89, 93, 90]
  let female_scores := [92, 88, 93, 91]
  have male_avg : male_scores.sum / male_scores.length = 91 := rfl
  have female_avg : female_scores.sum / female_scores.length = 91 := rfl
  show male_scores.sum / male_scores.length = 91
  sorry

end average_scores_equal_l282_282930


namespace max_min_of_f_on_interval_l282_282411

-- Conditions
def f (x : ℝ) : ℝ := x^3 - 3 * x + 1
def interval : Set ℝ := Set.Icc (-3) 0

-- Problem statement
theorem max_min_of_f_on_interval : 
  ∃ (max min : ℝ), max = 1 ∧ min = -17 ∧ 
  (∀ x ∈ interval, f x ≤ max) ∧ 
  (∀ x ∈ interval, f x ≥ min) := 
sorry

end max_min_of_f_on_interval_l282_282411


namespace sequence_bound_l282_282359

-- Definitions and assumptions based on the conditions
def valid_sequence (a : ℕ → ℕ) (N : ℕ) (m : ℕ) :=
  (1 ≤ a 1) ∧ (a m ≤ N) ∧ (∀ i j, 1 ≤ i → i < j → j ≤ m → a i < a j) ∧ 
  (∀ i j, 1 ≤ i → i < j → j ≤ m → Nat.lcm (a i) (a j) ≤ N)

-- The main theorem to prove
theorem sequence_bound (a : ℕ → ℕ) (N : ℕ) (m : ℕ) 
  (h : valid_sequence a N m) : m ≤ 2 * Nat.floor (Real.sqrt N) :=
sorry

end sequence_bound_l282_282359


namespace power_function_value_l282_282225

theorem power_function_value (f : ℝ → ℝ) (h : ∀ x, f x = x ^ (-2)) (hx : f(2) = 1/4) : f(4) = 1/16 :=
by
  -- conditions and definition of the power function are implied in the theorem declaration
  sorry

end power_function_value_l282_282225


namespace max_norm_z_sub_three_plus_four_i_l282_282293
noncomputable theory
open Complex Real

theorem max_norm_z_sub_three_plus_four_i (z : ℂ) (α : ℝ) (h1 : z = Complex.cos α + Complex.sin α * Complex.I) : 
  ∃ M : ℝ, M = 6 ∧ ∀ z : ℂ, ∃ α : ℝ, z = Complex.cos α + Complex.sin α * Complex.I → Complex.abs (z - 3 - 4 * Complex.I) ≤ M :=
by sorry

end max_norm_z_sub_three_plus_four_i_l282_282293


namespace quadratic_roots_transformation_l282_282380

theorem quadratic_roots_transformation {a b c r s : ℝ}
  (h1 : r + s = -b / a)
  (h2 : r * s = c / a) :
  (∃ p q : ℝ, p = a * r + 2 * b ∧ q = a * s + 2 * b ∧ 
     (∀ x, x^2 - 3 * b * x + 2 * b^2 + a * c = (x - p) * (x - q))) :=
by
  sorry

end quadratic_roots_transformation_l282_282380


namespace quadratic_roots_eq_k_quadratic_inequality_k_range_l282_282926

theorem quadratic_roots_eq_k (k : ℝ) (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0)
  (h3: (2 + 3) = (2/k)) : k = 2/5 :=
by sorry

theorem quadratic_inequality_k_range (k : ℝ) 
  (h1 : k > 0)
  (h2 : ∀ x : ℝ, 2 < x → x < 3 → k*x^2 - 2*x + 6*k < 0) 
: 0 < k ∧ k <= 2/5 :=
by sorry

end quadratic_roots_eq_k_quadratic_inequality_k_range_l282_282926


namespace quadratic_has_one_solution_l282_282197

theorem quadratic_has_one_solution (k : ℝ) : (4 : ℝ) * (4 : ℝ) - k ^ 2 = 0 → k = 8 ∨ k = -8 := by
  sorry

end quadratic_has_one_solution_l282_282197


namespace find_sum_p_q_r_l282_282367

noncomputable def vector_space := ℝ → ℝ → ℝ → ℝ

variables {V : Type*} [inner_product_space ℝ V]
variables (a b c : V) (p q r : ℝ)

-- Condition 1: a, b, and c are orthogonal unit vectors
def orthogonal_unit_vectors : Prop :=
  inner a b = 0 ∧ inner b c = 0 ∧ inner c a = 0 ∧ 
  inner a a = 1 ∧ inner b b = 1 ∧ inner c c = 1

-- Condition 2: a = p (a × b) + q (b × c) + r (c × a)
def vector_equation : Prop := 
  a = p • (a × b) + q • (b × c) + r • (c × a)

-- Condition 3: a · (b × c) = 2
def dot_product_value : Prop := 
  inner a (b × c) = 2

-- Proof statement
theorem find_sum_p_q_r 
  (h1 : orthogonal_unit_vectors a b c)
  (h2 : vector_equation a b c p q r)
  (h3 : dot_product_value a b c) : 
  p + q + r = 1 / 2 := 
sorry

end find_sum_p_q_r_l282_282367


namespace simplify_expr_l282_282401

noncomputable def expr : ℝ := (18 * 10^10) / (6 * 10^4) * 2

theorem simplify_expr : expr = 6 * 10^6 := sorry

end simplify_expr_l282_282401


namespace point_A_moved_to_vertex_3_l282_282077

-- Definitions of the initial conditions and rotation.
def cube := ℕ -- Representing vertex numbers by natural numbers.

def initial_vertex (A : cube) :=
  A = 0  -- Let's assume vertex 0 is where the green, far white, and right lower white faces meet.

def rotate_vertex (v : cube) : cube :=
  match v with
  | 0 => 3
  | 1 => 0
  | 2 => 1
  | 3 => 2
  | 4 => 7
  | 5 => 4
  | 6 => 5
  | 7 => 6
  | _ => v -- If it's not within the range of 0-7

-- Lean 4 statement to prove the movement of A to new vertex after rotation
theorem point_A_moved_to_vertex_3 (A : cube) (h : initial_vertex A) : rotate_vertex A = 3 :=
  by
  rw [h]
  exact rfl

end point_A_moved_to_vertex_3_l282_282077


namespace smallest_X_for_T_div_10_l282_282687

/-- Assume T is a positive integer whose only digits are 0s and 1s and X = T / 10 is an integer. 
Prove that the smallest possible value of X is 1. -/
theorem smallest_X_for_T_div_10 (T : ℕ) (h1 : T > 0) (h2 : ∀ d ∈ (T.digits 10), d = 0 ∨ d = 1) (h3 : ∃ X : ℕ, T = 10 * X) : 
  ∃ X : ℕ, X = 1 ∧ T = 10 * X :=
begin
  sorry
end

end smallest_X_for_T_div_10_l282_282687


namespace length_of_larger_cuboid_l282_282826

theorem length_of_larger_cuboid
  (n : ℕ)
  (l_small : ℝ) (w_small : ℝ) (h_small : ℝ)
  (w_large : ℝ) (h_large : ℝ)
  (V_large : ℝ)
  (n_eq : n = 56)
  (dim_small : l_small = 5 ∧ w_small = 3 ∧ h_small = 2)
  (dim_large : w_large = 14 ∧ h_large = 10)
  (V_large_eq : V_large = n * (l_small * w_small * h_small)) :
  ∃ l_large : ℝ, l_large = V_large / (w_large * h_large) ∧ l_large = 12 := by
  sorry

end length_of_larger_cuboid_l282_282826


namespace exists_root_satisfying_inequality_l282_282155

theorem exists_root_satisfying_inequality (n : ℕ) (x : ℝ) :
  n^2 * x^2 - (2 * n^2 + n) * x + (n^2 + n - 6) ≤ 0 → x = 1 :=
sorry

end exists_root_satisfying_inequality_l282_282155


namespace needed_amount_ratio_to_profit_is_61_96_l282_282847

-- Define the conditions of the problem
def profit : ℤ := 960
def half_profit : ℤ := profit / 2
def donations : ℤ := 310
def total_with_donations : ℤ := half_profit + donations
def above_goal : ℤ := 180
def needed_amount : ℤ := total_with_donations - above_goal

-- Define the simplified ratio function
def gcd (a b : ℤ) : ℤ := Int.gcd a b
def simplify_ratio (a b : ℤ) : (ℤ × ℤ) :=
  let d := gcd a b
  (a / d, b / d)

-- The statement verifying the ratio
theorem needed_amount_ratio_to_profit_is_61_96 : 
  simplify_ratio needed_amount profit = (61, 96) :=
by
  have h1 : needed_amount = 610 := rfl
  have h2 : profit = 960 := rfl
  rw [h1, h2, gcd, simplify_ratio]
  simp
  sorry

end needed_amount_ratio_to_profit_is_61_96_l282_282847


namespace determinant_transformed_matrix_l282_282365

variables {R : Type*} [Field R]
variables (a b c : Fin 3 → R)

def determinant (a b c : Fin 3 → R) : R :=
a 0 * (b 1 * c 2 - c 1 * b 2) -
a 1 * (b 0 * c 2 - c 0 * b 2) +
a 2 * (b 0 * c 1 - c 0 * b 1)

theorem determinant_transformed_matrix :
  let D := determinant a b c in
  determinant (fun i => 2 * a i + 3 * b i) (fun i => 3 * b i + 4 * c i) (fun i => 4 * c i + 2 * a i) = 24 * D :=
by
  sorry

end determinant_transformed_matrix_l282_282365


namespace percentage_of_200_l282_282447

theorem percentage_of_200 : ((1/4) / 100) * 200 = 0.5 := 
by
  sorry

end percentage_of_200_l282_282447


namespace parabola_slope_proof_l282_282489

variable (p : ℝ → ℝ → Prop) -- predicate for the parabola y^2 = 4x
variable (F M A B : ℝ × ℝ) -- points (F for focus, M for entry, A for reflection, B for exit)
variable (slope : ℝ → ℝ → ℝ → ℝ → ℝ) -- slope function

-- Define the conditions
def parabola (y x : ℝ) : Prop := y^2 = 4 * x
def focus : ℝ × ℝ := (1, 0)
def pointM : ℝ × ℝ := (3, 1)
def pointA : ℝ × ℝ := (1 / 4, 1)

-- Define the slope function
def line_slope (Ax Ay Bx By : ℝ) : ℝ := (By - Ay) / (Bx - Ax)

-- The proof statement
theorem parabola_slope_proof :
  parabola 1 (1 / 4) →
  focus = (1, 0) →
  pointM = (3, 1) →
  pointA = (1 / 4, 1) →
  B ∈ {p | parabola (p.snd) (p.fst)} →
  line_slope (1 / 4) 1 1 0 = -4 / 3 :=
by
  intros h_parabola h_focus h_pointM h_pointA h_pointB
  sorry

end parabola_slope_proof_l282_282489


namespace find_initial_pebbles_l282_282561

/-- Define the problem parameters -/
variables (x1 x2 x3 x4 x5 S : ℝ)
variables (total_pebbles : ℝ)

-- Conditions
def initial_conditions (x1 x2 x3 x4 x5 : ℝ) (total_pebbles : ℝ) : Prop :=
  total_pebbles = x1 + x2 + x3 + x4 + x5 ∧
  total_pebbles = 1990

def redistribution_rule1 (x1 x2 : ℝ) (S : ℝ) : Prop :=
  x1 - 5 + 0 * 1 = S

def redistribution_rule2 (x2 x3 : ℝ) (S : ℝ) : Prop :=
  x2 - (2 * 1) + 1 * 2 = S

def redistribution_rule3 (x3 x4 : ℝ) (S : ℝ) : Prop :=
  x3 - (3 * 1) + 2 * 3 = S

def redistribution_rule4 (x4 x5 : ℝ) (S : ℝ) : Prop :=
  x4 - (4 * 1) + 3 * 4 = S

def redistribution_rule5 (x5 x1 : ℝ) (S : ℝ) : Prop :=
  x5 + 1 * 5 - 4 * x1 = S

theorem find_initial_pebbles :
  ∃ (x1 x2 x3 x4 x5 S : ℝ), 
    initial_conditions x1 x2 x3 x4 x5 1990 ∧
    redistribution_rule1 x1 x2 S ∧
    redistribution_rule2 x2 x3 S ∧
    redistribution_rule3 x3 x4 S ∧
    redistribution_rule4 x4 x5 S ∧
    redistribution_rule5 x5 x1 S :=
sorry

end find_initial_pebbles_l282_282561


namespace power_function_solution_l282_282608

theorem power_function_solution (m : ℤ)
  (h1 : ∃ (f : ℝ → ℝ), ∀ x : ℝ, f x = x^(-m^2 + 2 * m + 3) ∧ ∀ x, f x = f (-x))
  (h2 : ∀ x : ℝ, x > 0 → (x^(-m^2 + 2 * m + 3)) < x^(-m^2 + 2 * m + 3 + x)) :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^4 :=
by
  sorry

end power_function_solution_l282_282608


namespace coefficient_x2_in_expansion_l282_282333

noncomputable def binomial_coefficient (n k : ℕ) : ℕ :=
nat.choose n k

theorem coefficient_x2_in_expansion :
  let n := 8 in
  let term_coefficient r := binomial_coefficient n r * (-1)^r in
  ∃ r, (2 * r = n - 2) ∧ (term_coefficient r = -56) :=
begin
  let n := 8,
  let term_coefficient := λ r, binomial_coefficient n r * (-1)^r,
  existsi 3,
  split,
  {
    -- Proof that 2 * r = n - 2 for r = 3
    exact dec_trivial,
  },
  {
    -- Proof that the term coefficient at r = 3 is -56
    exact dec_trivial,
  }
end

end coefficient_x2_in_expansion_l282_282333


namespace pythagorean_theorem_l282_282023

theorem pythagorean_theorem (a b c : ℝ) (h : a^2 + b^2 = c^2) : c^2 = a^2 + b^2 :=
sorry

end pythagorean_theorem_l282_282023


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282262

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282262


namespace sqrt_condition_l282_282304

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_condition_l282_282304


namespace significant_digits_side_of_square_l282_282530

theorem significant_digits_side_of_square {A : ℝ} (hA: A = 2.3406) :
  significant_digits (sqrt A) = 5 := 
sorry

end significant_digits_side_of_square_l282_282530


namespace B_representation_l282_282569

def A : Set ℤ := {-1, 2, 3, 4}

def f (x : ℤ) : ℤ := x^2 - 2*x + 2

def B : Set ℤ := {y | ∃ x ∈ A, y = f x}

theorem B_representation : B = {2, 5, 10} :=
by {
  -- Proof to be provided
  sorry
}

end B_representation_l282_282569


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282277

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282277


namespace f_lg_lg2_l282_282231

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * Math.sin x + 4

theorem f_lg_lg2 (a b : ℝ)
  (h : f a b (Float.lg (Float.log2 10)) = 5) :
  f a b (Float.lg (Float.lg 2)) = 3 :=
sorry

end f_lg_lg2_l282_282231


namespace sqrt_equation_solution_l282_282172

theorem sqrt_equation_solution (x : ℝ) (h₀ : 9 < x) 
  (h₁ : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 21 := 
sorry

end sqrt_equation_solution_l282_282172


namespace board_stabilizes_l282_282127

-- Define the initial configuration of the 6x6 board
inductive color
| black
| white

def board : Type := fin 6 → fin 6 → color

-- Define the rule for changing the board's coloring
def change_color (b : board) (i j : fin 6) : color :=
  let neighbors := [(i-1,j), (i+1,j), (i,j-1), (i,j+1)] in
  if (neighbors.filter (fun (n : fin 6 × fin 6) => b n.1 n.2 = color.black)).length ≥ 2 then
    color.black
  else
    b i j

-- Define the function to evolve the board over time
def evolve (b : board) (n : ℕ) : board :=
  nat.rec_on n b (fun _ b' => 
    fun i j => change_color b' i j
  )

-- The initial configuration (as per the given puzzle)
def initial_board : board := sorry -- Initialize based on puzzle's initial state

-- The proof statement
noncomputable def board_after_12_seconds : Prop :=
evolve initial_board 12 = (fun _ _ => color.black)

noncomputable def board_after_13_seconds : Prop :=
evolve initial_board 13 = (fun _ _ => color.black)

-- Conjecture to be proven
theorem board_stabilizes :
  board_after_12_seconds ∧ board_after_13_seconds :=
by sorry

end board_stabilizes_l282_282127


namespace octopus_undewater_cave_age_l282_282106

def octal_to_decimal := 2 * 8^2 + 4 * 8^1 + 5 * 8^0

theorem octopus_undewater_cave_age : octal_to_decimal = 165 :=
by
  unfold octal_to_decimal
  norm_num
  -- proof steps can follow here
  sorry

end octopus_undewater_cave_age_l282_282106


namespace force_magnitudes_ratio_l282_282756

theorem force_magnitudes_ratio (a d : ℝ) (h1 : (a + 2 * d)^2 = a^2 + (a + d)^2) :
  ∃ k : ℝ, k > 0 ∧ (a + d) = a * (4 / 3) ∧ (a + 2 * d) = a * (5 / 3) :=
by
  sorry

end force_magnitudes_ratio_l282_282756


namespace abs_expression_eq_five_l282_282513

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end abs_expression_eq_five_l282_282513


namespace sum_of_cubes_eq_l282_282389

theorem sum_of_cubes_eq : (∑ k in Finset.range (8 + 1), k^3) = 1296 :=
by 
  sorry

end sum_of_cubes_eq_l282_282389


namespace geometric_sequence_sum_l282_282190

theorem geometric_sequence_sum :
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  (a * (1 - r^n) / (1 - r)) = 3280 / 6561 :=
by {
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  calc
  (a * (1 - r^n) / (1 - r)) = (1/3 * (1 - (1/3)^8) / (1 - 1/3)) : by rw a; rw r
  ... = 3280 / 6561 : sorry
}

end geometric_sequence_sum_l282_282190


namespace cannot_form_right_triangle_can_form_right_triangle_A_can_form_right_triangle_B_can_form_right_triangle_C_l282_282800

theorem cannot_form_right_triangle (a b c : ℝ) : 
  ¬ ((a = 2 ∧ b = 3 ∧ c = 4) ∧ a^2 + b^2 = c^2) :=
by {
  rintro ⟨⟨rfl, rfl, rfl⟩, h⟩,
  have : 4 + 9 = 16 := by norm_num,
  linarith,
  sorry
}

theorem can_form_right_triangle_A (a b c : ℝ) : 
  (a = 5 ∧ b = 12 ∧ c = 13) → (a^2 + b^2 = c^2) :=
by {
  rintro ⟨rfl, rfl, rfl⟩,
  norm_num,
}

theorem can_form_right_triangle_B (a b c : ℝ) : 
  (a = real.sqrt 2 ∧ b = real.sqrt 3 ∧ c = real.sqrt 5) → (a^2 + b^2 = c^2) :=
by {
  rintro ⟨rfl, rfl, rfl⟩,
  simp, 
  norm_num,
}

theorem can_form_right_triangle_C (a b c : ℝ) : 
  (a = real.sqrt 7 ∧ b = 3 ∧ c = 4) → (a^2 + b^2 = c^2) :=
by {
  rintro ⟨rfl, rfl, rfl⟩,
  norm_num,
}

end cannot_form_right_triangle_can_form_right_triangle_A_can_form_right_triangle_B_can_form_right_triangle_C_l282_282800


namespace alpha_in_second_quadrant_l282_282296

theorem alpha_in_second_quadrant (α : Real) 
  (h1 : Real.sin (2 * α) < 0) 
  (h2 : Real.cos α - Real.sin α < 0) : 
  π / 2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l282_282296


namespace sqrt_x_minus_3_domain_l282_282313

theorem sqrt_x_minus_3_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_x_minus_3_domain_l282_282313


namespace circumcircles_tangent_l282_282093

open EuclideanGeometry

variables (A B C I X Y Z : Point)
variables (ω Γ : Circle)

-- Conditions
def condition1 := inscribedCircle ω A B C I
def condition2 := circumscribedCircle Γ A I B
def condition3 := intersectAtTwoPoints ω Γ X Y
def condition4 := commonTangentsIntersectAt ω Γ Z

-- Theorem: The circumcircles of triangles ABC and XYZ are tangent to each other.
theorem circumcircles_tangent
  (h1 : condition1)
  (h2 : condition2)
  (h3 : condition3)
  (h4 : condition4) :
  tangent (circumcircle A B C) (circumcircle X Y Z) :=
sorry

end circumcircles_tangent_l282_282093


namespace time_needed_to_gather_remaining_flowers_l282_282349

-- conditions
def classmates : ℕ := 30
def time_per_flower : ℕ := 10
def gathering_time : ℕ := 2 * 60
def lost_flowers : ℕ := 3

-- question and proof goal
theorem time_needed_to_gather_remaining_flowers : 
  let flowers_needed := classmates - ((gathering_time / time_per_flower) - lost_flowers)
  flowers_needed * time_per_flower = 210 :=
by
  sorry

end time_needed_to_gather_remaining_flowers_l282_282349


namespace no_constant_c_l282_282378

-- Define function f satisfying the given condition
def f (x : ℚ) : ℚ := sorry

-- Hypothesis that f satisfies the given condition for all rationals x and y
axiom f_additive_property (x y : ℚ) : f(x + y) - f(x) - f(y) ∈ ℤ

-- Statement to prove: There does not exist a constant c ∈ ℚ such that for all x ∈ ℚ, f(x) - c * x ∈ ℤ
theorem no_constant_c (c : ℚ) (H : ∀ x : ℚ, f(x) - c * x ∈ ℤ) : false :=
sorry

end no_constant_c_l282_282378


namespace concyclic_points_l282_282326

noncomputable def Rectangle (A B C D : Point) : Prop :=
  ∃ (α β γ δ : ℝ), 
  A.x = 0 ∧ A.y = 0 ∧
  B.x = α ∧ B.y = 0 ∧
  C.x = α ∧ C.y = β ∧
  D.x = 0 ∧ D.y = β ∧
  α > 0 ∧ β > 0

noncomputable def Midpoint (E B C : Point) : Prop :=
  E.x = (B.x + C.x) / 2 ∧ E.y = (B.y + C.y) / 2

noncomputable def FootOfPerpendicular (F A P B : Point) : Prop :=
  (F.x - A.x) * (B.x - P.x) + (F.y - A.y) * (B.y - P.y) = 0

theorem concyclic_points (A B C D E F P G : Point) 
  (h1 : Rectangle A B C D)
  (h2 : (dist B C) = 2 * (dist A B))
  (h3 : Midpoint E B C)
  (h4 : P.x = 0 ∧ 0 < P.y ∧ P.y < (dist A D))
  (h5 : FootOfPerpendicular F A P B)
  (h6 : FootOfPerpendicular G D P C) : 
  ∃ (circle : Circle), circle.contains E ∧ circle.contains F ∧ circle.contains P ∧ circle.contains G :=
sorry

end concyclic_points_l282_282326


namespace sqrt_domain_l282_282315

theorem sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_domain_l282_282315


namespace angelina_speed_grocery_to_gym_l282_282860

-- Define the conditions
variables (v : ℝ) (v_pos : 0 < v)
def time_home_to_grocery := 250 / v
def time_grocery_to_gym := 180 / v

-- Given condition
def time_difference := time_home_to_grocery v - time_grocery_to_gym v = 70

-- Lean statement to prove Angelina's speed from the grocery to the gym
theorem angelina_speed_grocery_to_gym (v_pos : 0 < v) (h : time_difference v) : 2 * v = 2 := 
by sorry

end angelina_speed_grocery_to_gym_l282_282860


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282281

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282281


namespace probability_at_least_two_heads_l282_282699

theorem probability_at_least_two_heads (n : ℕ) (p : ℚ) (h₀ : n = 5) (h₁ : p = 1/2) :
  ∑ k in (finset.range (n + 1)), if k ≤ 1 then (nat.choose n k) * p^k * (1 - p)^(n - k) else 0 
  = 16/16 - 3/16 := 
sorry

end probability_at_least_two_heads_l282_282699


namespace circle_radius_touching_square_extension_l282_282822

theorem circle_radius_touching_square_extension
  (side : ℝ)
  (r : ℝ)
  (sin_36 : ℝ)
  (h1 : side = 2 - real.sqrt(5 - real.sqrt 5))
  (h2 : sin_36 = real.sqrt(5 - real.sqrt 5) / (2 * real.sqrt 2)) :
  r = real.sqrt(5 - real.sqrt 5) :=
by
  -- The proof is omitted here
  sorry

end circle_radius_touching_square_extension_l282_282822


namespace sum_of_solutions_l282_282795

  theorem sum_of_solutions :
    (∃ x : ℝ, x = abs (2 * x - abs (50 - 2 * x)) ∧ ∃ y : ℝ, y = abs (2 * y - abs (50 - 2 * y)) ∧ ∃ z : ℝ, z = abs (2 * z - abs (50 - 2 * z)) ∧ (x + y + z = 170 / 3)) :=
  sorry
  
end sum_of_solutions_l282_282795


namespace least_pos_int_div_by_3_5_7_l282_282788

/-
  Prove that the least positive integer divisible by the primes 3, 5, and 7 is 105.
-/

theorem least_pos_int_div_by_3_5_7 : ∃ (n : ℕ), n > 0 ∧ (n % 3 = 0) ∧ (n % 5 = 0) ∧ (n % 7 = 0) ∧ n = 105 :=
by 
  sorry

end least_pos_int_div_by_3_5_7_l282_282788


namespace triangle_congruence_l282_282727

open EuclideanGeometry

theorem triangle_congruence (A B C A' B' C' : Point)
  (h1 : ∠ A = ∠ A')
  (h2 : length (segment A B) = length (segment A' B'))
  (h3 : length (segment B C) = length (segment B' C'))
  (h4 : ∠ C = 90°) :
  triangle ABC ≅ triangle A'B'C' := by
  sorry

end triangle_congruence_l282_282727


namespace trig_order_l282_282980

theorem trig_order (θ : ℝ) (h1 : -Real.pi / 8 < θ) (h2 : θ < 0) : Real.tan θ < Real.sin θ ∧ Real.sin θ < Real.cos θ := 
sorry

end trig_order_l282_282980


namespace magnitude_of_vector_diff_l282_282976

variables {V : Type*} [inner_product_space ℝ V]

variables (a b : V)
variable (ha : ∥a∥ = 2)
variable (hb : ∥b∥ = 1)
variable (h_perp : ⟪a + b, a⟫ = 0)

theorem magnitude_of_vector_diff : ∥a - (2:ℝ) • b∥ = 2 * real.sqrt 6 :=
by sorry

end magnitude_of_vector_diff_l282_282976


namespace solve_for_x_l282_282167

theorem solve_for_x (x : ℝ) (hx : x > 9) 
  (h : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 18 := 
  sorry

end solve_for_x_l282_282167


namespace sqrt_equation_solution_l282_282173

theorem sqrt_equation_solution (x : ℝ) (h₀ : 9 < x) 
  (h₁ : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 21 := 
sorry

end sqrt_equation_solution_l282_282173


namespace bob_smallest_possible_number_l282_282854

theorem bob_smallest_possible_number :
  ∀ (n : ℕ), prime_factorization n = {2 := 2, 3 := 2, 5 := 1} →
    n = 180 :=
by sorry

end bob_smallest_possible_number_l282_282854


namespace sqrt_domain_l282_282314

theorem sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_domain_l282_282314


namespace traversable_for_k_3_and_k_ge_5_non_traversable_for_k_eq_4_l282_282429

def is_hamiltonian_knight (board : List (List ℕ)) : Prop :=
  ∃ path : List (ℕ × ℕ), 
  ∀ s ∈ path, 
  ∀ t ∈ path, 
  knight_moves s t ∧ no_duplicates path ∧ path.length = 4 * length board

def checkerboard (width length : ℕ) : (ℕ × ℕ) → Prop :=
  λ (x, y), (x + y) % 2 = 0

def knight_moves (s t : ℕ × ℕ) : Prop :=
  let dx := abs (s.1 - t.1) in
  let dy := abs (s.2 - t.2) in
  (dx = 2 ∧ dy = 1) ∨ (dx = 1 ∧ dy = 2)

def no_duplicates (p : List (ℕ × ℕ)) : Prop :=
  ∀ (i j : ℕ), i < p.length ∧ j < p.length → p[i] = p[j] → i = j

theorem traversable_for_k_3_and_k_ge_5 (k : ℤ) (h : k ≥ 3) : 
  (k = 3 ∨ k ≥ 5) → 
  ∃ board : List (List ℕ), 
    length board = k ∧ width board = 4 ∧ 
    (∀ x y, checkerboard 4 k (x, y) → (is_hamiltonian_knight board)) :=
sorry

theorem non_traversable_for_k_eq_4 :
  ∃ board : List (List ℕ), 
    length board = 4 ∧ width board = 4 ∧ 
    ¬ (is_hamiltonian_knight board) :=
sorry

end traversable_for_k_3_and_k_ge_5_non_traversable_for_k_eq_4_l282_282429


namespace students_in_diligence_before_transfer_l282_282994

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l282_282994


namespace find_C_l282_282145

def M := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

def A : Set ℕ := {a1, a2, a3, a4}
def B : Set ℕ := {b1, b2, b3, b4}
def C : Set ℕ := {c1, c2, c3, c4}

namespace PartitionProof

  theorem find_C (a1 a2 a3 a4 b1 b2 b3 b4 c1 c2 c3 c4 : ℕ) (h_disjoint_A_B_C : M = A ∪ B ∪ C)
    (h_A : A = {a1, a2, a3, a4})
    (h_B : B = {b1, b2, b3, b4})
    (h_C : C = {c1, c2, c3, c4})
    (h_ordered_C : c1 < c2 < c3 < c4)
    (h_sum : ∀ k ∈ {1, 2, 3, 4}, (A.toList.nth (k - 1)).get_or_else 0 + (B.toList.nth (k - 1)).get_or_else 0 = (C.toList.nth (k - 1)).get_or_else 0) :
    C = {8, 9, 10, 12} ∨ C = {7, 9, 11, 12} ∨ C = {6, 10, 11, 12} :=
  sorry

end PartitionProof

end find_C_l282_282145


namespace batsman_average_increase_l282_282478

theorem batsman_average_increase 
    (A : ℝ) 
    (h1 : 11 * A + 80 = 12 * 47) : 
    47 - A = 3 := 
by 
  -- Proof goes here
  sorry

end batsman_average_increase_l282_282478


namespace sum_of_squares_of_roots_l282_282560

theorem sum_of_squares_of_roots : 
  ∀ (r1 r2 : ℝ), (r1 + r2 = 10) ∧ (r1 * r2 = 16) → (r1^2 + r2^2 = 68) :=
by
  intros r1 r2 h
  cases h with h1 h2
  sorry

end sum_of_squares_of_roots_l282_282560


namespace percentage_salary_l282_282017

theorem percentage_salary (m n : ℕ) (h1 : m + n = 594) (h2 : n = 270) : (m * 100 / n) = 120 :=
by
  -- Given the problem, we need to prove that given the conditions:
  -- 1. m + n = 594
  -- 2. n = 270
  -- We should find that (m * 100 / n) = 120.
  have h3 : m = 324 :=
    calc
      m = 594 - n : by rw [h1, h2, nat.add_sub_cancel_left]
      _ = 324      : by rw [h2]
  -- Convert the fraction m/n into the percentage
  rw [h3]
  -- Simplifying m*100/n with the known values of m and n
  have : (324 * 100) / 270 = 120 :=
    calc
      (324 * 100) / 270 = (324 / 270) * 100 : by rw mul_div_assoc'
      _ = 1.2 * 100                         : by norm_num
      _ = 120                               : by norm_num
  exact this

end percentage_salary_l282_282017


namespace train_passes_bridge_in_20_seconds_l282_282499

def train_length : ℕ := 360
def bridge_length : ℕ := 140
def train_speed_kmh : ℕ := 90

noncomputable def train_speed_ms : ℝ := (train_speed_kmh * 1000) / 3600
noncomputable def total_distance : ℕ := train_length + bridge_length
noncomputable def travel_time : ℝ := total_distance / train_speed_ms

theorem train_passes_bridge_in_20_seconds :
  travel_time = 20 := by
  sorry

end train_passes_bridge_in_20_seconds_l282_282499


namespace sum_of_inscribed_sphere_volumes_l282_282209

theorem sum_of_inscribed_sphere_volumes :
  let height := 3
  let angle := Real.pi / 3
  let r₁ := height / 3 -- Radius of the first inscribed sphere
  let geometric_ratio := 1 / 3
  let volume (r : ℝ) := (4 / 3) * Real.pi * r^3
  let volumes : ℕ → ℝ := λ n => volume (r₁ * geometric_ratio^(n - 1))
  let total_volume := ∑' n, volumes n
  total_volume = (18 * Real.pi) / 13 :=
by
  sorry

end sum_of_inscribed_sphere_volumes_l282_282209


namespace find_c_l282_282132

noncomputable def f (x : ℝ) : ℝ := 3 - 8 * x + 2 * x^2 - 7 * x^3 + 6 * x^4
noncomputable def g (x : ℝ) : ℝ := 2 - 3 * x + x^3 - 2 * x^4

theorem find_c : ∃ c : ℝ, ∀ x : ℝ, (f x + c * g x).degree = 3 :=
by {
  let f := 3 - 8 * x + 2 * x^2 - 7 * x^3 + 6 * x^4,
  let g := 2 - 3 * x + x^3 - 2 * x^4,
  use 3,
  sorry 
}

end find_c_l282_282132


namespace joel_garden_size_l282_282347

-- Definitions based on the conditions
variable (G : ℕ) -- G is the size of Joel's garden.

-- Condition 1: Half of the garden is for fruits.
def half_garden_fruits (G : ℕ) := G / 2

-- Condition 2: Half of the garden is for vegetables.
def half_garden_vegetables (G : ℕ) := G / 2

-- Condition 3: A quarter of the fruit section is used for strawberries.
def quarter_fruit_section (G : ℕ) := (half_garden_fruits G) / 4

-- Condition 4: The quarter for strawberries takes up 8 square feet.
axiom strawberry_section : quarter_fruit_section G = 8

-- Hypothesis: The size of Joel's garden is 64 square feet.
theorem joel_garden_size : G = 64 :=
by
  -- Insert the logical progression of the proof here.
  sorry

end joel_garden_size_l282_282347


namespace part1_part2_part3_l282_282213

noncomputable theory
open Real

-- Part 1
theorem part1 (x : ℝ) : (let f (x : ℝ) := cos x + sin x
                             α : ℝ := π / 2
                         in (f x) * (f (x + α))) = cos (2 * x) :=
by
  sorry

-- Part 2
theorem part2 (x : ℝ) : (let g (x : ℝ) := 2 * cos x * (cos x + sqrt 3 * sin x)
                         α : ℝ := -π / 3
                         f (x : ℝ) := 2 * cos x
                         in g x) = (f x) * (f (x + α)) :=
by
  sorry

-- Part 3
theorem part3 (x1 x2 : ℝ) (f (x : ℝ) := abs (sin x) + cos x)
              (α : ℝ := π / 2)
              (g1 g2 : ℝ) : (∃ x1 x2 : ℝ, ∀ x : ℝ, (let g (x : ℝ) := (f x) * (f (x + α)) 
                                                     in g x1 ≤ g x ∧ g x ≤ g x2)) → 
                            (min (abs (x1 - x2))) = 3 * π / 4 :=
by
  sorry

end part1_part2_part3_l282_282213


namespace perfect_squares_two_digit_divisible_by_4_count_l282_282269

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l282_282269


namespace limit_sqrt_arctg_sin_cos_as_x_to_0_l282_282886

theorem limit_sqrt_arctg_sin_cos_as_x_to_0 :
  tendsto (fun x => sqrt (arctan x * (sin (1/x))^2 + 5 * cos x)) (nhds 0) (nhds (sqrt 5)) :=
begin
  sorry
end

end limit_sqrt_arctg_sin_cos_as_x_to_0_l282_282886


namespace train_length_l282_282498

theorem train_length {v_t v_m : ℝ} (t : ℝ) (h_vt : v_t = 63) (h_vm : v_m = 3) (h_t : t = 29.997600191984642) :
  (500 : ℝ) = 
  (let relative_speed := (v_t - v_m) * (1000 / 3600 : ℝ) in
  relative_speed * t) :=
by
  sorry

end train_length_l282_282498


namespace students_in_diligence_before_transfer_l282_282992

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l282_282992


namespace rectangle_area_eq_l282_282786

theorem rectangle_area_eq (x : ℝ) (h : (x - 2) * (2x + 5) = 8x - 6) : x = 4 :=
sorry

end rectangle_area_eq_l282_282786


namespace average_cardinality_of_subsets_l282_282028

open Finset

/-- theorem at about the average cardinality of subsets of a finite set -/
theorem average_cardinality_of_subsets (n : ℕ) : 
  let subsets := powerset (range (n + 1)) in
  (∑ s in subsets, s.card) / subsets.card = n / 2 := 
by {
  -- we still need the replacements for the actual proof steps.
  sorry
}

end average_cardinality_of_subsets_l282_282028


namespace locus_of_P_l282_282683

variables (C : Set Point) (d : Set Point) (M : Point) [IsCircle C] [IsLine d] [Tangent d C]
variables (E : Point) [IsTangentPoint E d C] 

def symmetric_point (E M : Point) : Point := sorry -- Later will be defined using geometric transformations

variables (F : Point) [SymmetricPoint F E M]

def diametrically_opposite_point (E : Point) [IsCircle C] : Point := sorry -- Defined as the point diametrically opposite to E on circle C

variables (E' : Point) [DiametricallyOppositePoint E' E C]

theorem locus_of_P (P : Point) :
  (∃ (Q R : Point) (hQd : Q ∈ d) (hRd : R ∈ d), midpoint M Q R ∧ incircle C (triangle P Q R))
  ↔ P ∈ line_segment E' (symmetric_point E M) :=
sorry

end locus_of_P_l282_282683


namespace water_volume_per_minute_l282_282806

theorem water_volume_per_minute (depth width : ℝ) (flow_rate_kmph : ℝ) 
  (H_depth : depth = 5) 
  (H_width : width = 35) 
  (H_flow_rate_kmph : flow_rate_kmph = 2) : 
  (depth * width * (flow_rate_kmph * 1000 / 60)) = 5832.75 :=
by
  sorry

end water_volume_per_minute_l282_282806


namespace wheat_field_area_in_hectares_l282_282490

def area_of_parallelogram (base height : ℝ) : ℝ := base * height

theorem wheat_field_area_in_hectares :
  ∀ (base height : ℝ),
    base = 140 ∧ height = 30 →
    (area_of_parallelogram base height / 10000) = 0.42 :=
by
  intros base height h
  cases h with h_base h_height
  rw [h_base, h_height]
  have : area_of_parallelogram 140 30 = 140 * 30 := rfl
  rw this
  norm_num
  sorry

end wheat_field_area_in_hectares_l282_282490


namespace probability_sum_less_than_16_l282_282437

-- The number of possible outcomes when three six-sided dice are rolled
def total_outcomes : ℕ := 6 * 6 * 6

-- The number of favorable outcomes where the sum of the dice is less than 16
def favorable_outcomes : ℕ := (6 * 6 * 6) - (3 + 3 + 3 + 1)

-- The probability that the sum of the dice is less than 16
def probability_less_than_16 : ℚ := favorable_outcomes / total_outcomes

theorem probability_sum_less_than_16 : probability_less_than_16 = 103 / 108 := 
by sorry

end probability_sum_less_than_16_l282_282437


namespace mildred_oranges_l282_282702

theorem mildred_oranges (original after given : ℕ) (h1 : original = 77) (h2 : after = 79) (h3 : given = after - original) : given = 2 :=
by
  sorry

end mildred_oranges_l282_282702


namespace determinant_transformed_columns_l282_282363

variables {A B C : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
variables (a b c : A)
variable {D : ℝ}

def determinant (M : Matrix (Fin 3) (Fin 3) ℝ) : ℝ :=
  Matrix.det M

theorem determinant_transformed_columns {a b c : A} (hD : determinant ![a, b, c] = D) :
  determinant ![2 • a + 3 • b, 3 • b + 4 • c, 4 • c + 2 • a] = 48 * D :=
sorry

end determinant_transformed_columns_l282_282363


namespace solve_for_x_l282_282168

theorem solve_for_x (x : ℝ) (hx : x > 9) 
  (h : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 18 := 
  sorry

end solve_for_x_l282_282168


namespace polynomial_satisfies_l282_282157

noncomputable def polynomial_solution (f : ℂ[X]) : Prop :=
  ∀ x : ℂ, f(3 * x) / f(x) = 729 * (x - 3) / (x - 243)

theorem polynomial_satisfies (f : ℂ[X]) :
  polynomial_solution f ↔
    ∃ a : ℂ, f = a * X * (X - 243) * (X - 81) * (X - 27) * (X - 9) * X^2 :=
sorry

end polynomial_satisfies_l282_282157


namespace problem_statement_l282_282611

def A : Set ℝ := { x | 3 < x ∧ x < 10 }
def B : Set ℝ := { x | x^2 - 9 * x + 14 < 0 }
def C (m : ℝ) : Set ℝ := { x | 5 - m < x ∧ x < 2 * m }

theorem problem_statement (m : ℝ) :
  (A ∩ B = { x : ℝ | 3 < x ∧ x < 7 }) ∧
  ((Aᶜ) ∪ B = { x : ℝ | x < 7 } ∪ { x | x ≥ 10 }) ∧
  m ∈ (Iio (2 : ℝ) ∪ Iic (2 : ℝ)) :=
sorry

#eval problem_statement

end problem_statement_l282_282611


namespace price_of_turbans_l282_282618

theorem price_of_turbans : 
  ∀ (salary_A salary_B salary_C : ℝ) (months_A months_B months_C : ℕ) (payment_A payment_B payment_C : ℝ)
    (prorated_salary_A prorated_salary_B prorated_salary_C : ℝ),
  salary_A = 120 → 
  salary_B = 150 → 
  salary_C = 180 → 
  months_A = 8 → 
  months_B = 7 → 
  months_C = 10 → 
  payment_A = 80 → 
  payment_B = 87.50 → 
  payment_C = 150 → 
  prorated_salary_A = (salary_A * (months_A / 12 : ℝ)) → 
  prorated_salary_B = (salary_B * (months_B / 12 : ℝ)) → 
  prorated_salary_C = (salary_C * (months_C / 12 : ℝ)) → 
  ∃ (price_A price_B price_C : ℝ),
  price_A = payment_A - prorated_salary_A ∧ 
  price_B = payment_B - prorated_salary_B ∧ 
  price_C = payment_C - prorated_salary_C ∧ 
  price_A = 0 ∧ price_B = 0 ∧ price_C = 0 := 
by
  sorry

end price_of_turbans_l282_282618


namespace correct_conclusions_l282_282339

noncomputable def triangle := Type

variables {A B C : triangle}
variables {a b c : ℝ} -- sides opposite to angles A, B, and C
variables (k : ℝ)
variables (h1 : b + c = 4 * k) (h2 : c + a = 5 * k) (h3 : a + b = 6 * k)

def sin_ratio : Prop := (sin a / sin A) = (sin b / sin B) = (sin c / sin C) = 7 / 5 / 3

def dot_product_condition : Prop := (b * c * cos A < 0)

def area_condition (c : ℝ) : Prop := if c = 6 then triangle_area ∆ ABC = 15 else false

def circumcircle_radius_condition : Prop := if b + c = 8 then circumcircle_radius ∆ ABC = 7 * sqrt 3 / 3 else false

theorem correct_conclusions :
  sin_ratio ∧ ¬dot_product_condition ∧ ¬area_condition c ∧ circumcircle_radius_condition :=
by sorry

end correct_conclusions_l282_282339


namespace mike_and_john_ratio_l282_282700

noncomputable def TacoGrandePlate : ℝ := sorry
-- Cost of the side salad, cheesy fries, and diet cola defined as constants
def sideSaladCost : ℝ := 2
def cheesyFriesCost : ℝ := 4
def dietColaCost : ℝ := 2

-- Mike's total bill
def M : ℝ := TacoGrandePlate + sideSaladCost + cheesyFriesCost + dietColaCost

-- John's total bill
def J : ℝ := TacoGrandePlate

-- Condition: Combined total cost of Mike and John's lunch is $24
axiom combined_cost : TacoGrandePlate + (TacoGrandePlate + sideSaladCost + cheesyFriesCost + dietColaCost) = 24

theorem mike_and_john_ratio : M / J = 2 :=
by
  sorry

end mike_and_john_ratio_l282_282700


namespace average_cardinality_of_subsets_l282_282027

theorem average_cardinality_of_subsets (n : ℕ) :
  (∑ A in Finset.powerset (Finset.range n), A.card) / 2^n = n / 2 := 
by
  sorry

end average_cardinality_of_subsets_l282_282027


namespace part_I_part_II_l282_282243

noncomputable def universal_set := Set.univ
noncomputable def A (a : ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem part_I (h : a = 0.5) :
  (Set.compl (B 0.5)) ∩ (A 0.5) = Ioc (9 / 4) (5 / 2) := sorry

theorem part_II (h : ∀ x, x ∈ A a → x ∈ B a) :
  a ∈ (Ioo (-1 / 2) (1 / 3) ∪ Ioo (1 / 3) ((3 - Real.sqrt 5) / 2)) := sorry

end part_I_part_II_l282_282243


namespace exists_single_side_13_no_single_side_exists_l282_282395

section Polygon

-- Define a polygon with n sides
structure Polygon (n : ℕ) :=
  (sides : Fin n → ℝ × ℝ)
  (no_collinear : ∀ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k → ¬Collinear (sides i) (sides j) (sides k))

-- Define a function that checks if a line contains exactly one side of a polygon
def contains_exactly_one_side (p : Polygon n) (l : Line) : Prop :=
  ∃ s : ℕ, l_contains_line l (p.sides s)

-- Theorem for n = 13
theorem exists_single_side_13 : ∀ (p : Polygon 13), ∃ (l : Line), contains_exactly_one_side p l :=
by sorry

-- Theorem for n > 13
theorem no_single_side_exists (n : ℕ) (h : n > 13) : ∃ (p : Polygon n), ∀ (l : Line), ¬ contains_exactly_one_side p l :=
by sorry

end Polygon

end exists_single_side_13_no_single_side_exists_l282_282395


namespace correct_crease_length_l282_282804

-- Define the problem constraints and question
def crease_length (θ : ℝ) : ℝ :=
  3 * (1 / Mathlib.cos θ ^ 2) * (1 / Mathlib.sin θ)

theorem correct_crease_length (θ : ℝ) : 
  crease_length θ = 3 * (1 / Mathlib.cos θ ^ 2) * (1 / Mathlib.sin θ) :=
by
  sorry

end correct_crease_length_l282_282804


namespace inequality_subtraction_l282_282292

theorem inequality_subtraction (a b : ℝ) (h : a < b) : a - 5 < b - 5 := 
by {
  sorry
}

end inequality_subtraction_l282_282292


namespace borel_sets_cardinality_lebesgue_measurable_sets_cardinality_combined_result_l282_282373

open Cardinal

noncomputable def c := Cardinal.mk ℝ

theorem borel_sets_cardinality : ∀ (c : Cardinal), (c = Cardinal.mk ℝ) → (Cardinal.mk (set.borel ℝ) = c) :=
begin
  intros c hc,
  rw hc, -- Using the fact that \(\mathfrak{c}\) is the cardinality of the real numbers
  sorry
end

theorem lebesgue_measurable_sets_cardinality : ∀ (c : Cardinal), (c = Cardinal.mk ℝ) → (Cardinal.mk (measure_theory.measurable_set ℝ) = 2^c) :=
begin
  intros c hc,
  rw hc, -- Using the fact that \(\mathfrak{c}\) is the cardinality of the real numbers
  sorry
end

-- Combine both results in one conclusive form if necessary
theorem combined_result : ∀ (c : Cardinal), (c = Cardinal.mk ℝ) → ((Cardinal.mk (set.borel ℝ) = c) ∧ (Cardinal.mk (measure_theory.measurable_set ℝ) = 2^c)) :=
begin
  intros c hc,
  split,
  { apply borel_sets_cardinality, exact hc },
  { apply lebesgue_measurable_sets_cardinality, exact hc }
end

end borel_sets_cardinality_lebesgue_measurable_sets_cardinality_combined_result_l282_282373


namespace problem_solution_l282_282161

theorem problem_solution (x : ℝ) (h1 : x > 9) 
(h2 : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x ∈ Set.Ici 18 := sorry

end problem_solution_l282_282161


namespace smallest_number_of_students_l282_282651

theorem smallest_number_of_students (n : ℕ)
  (h1 : 6 * 120) 
  (h2 : ∀ k ≤ (n - 6), k ≥ 70)
  (h3 : (6 * 120 + (n - 6) * 70) ≥ 85 * n) :
  n = 20 :=
sorry

end smallest_number_of_students_l282_282651


namespace transform_to_all_ones_l282_282771

noncomputable def can_make_all_one (numbers : List ℕ) : Prop :=
  ∀ (n : ℕ), 2 ≤ n → (numbers.length = n) →
  (∀ a ∈ numbers,  
  ∃ N, N = List.lcm numbers ∧ 
  let numbers' := List.map (λ x, if x = a then N / x else x) numbers in
  can_make_all_one numbers')

theorem transform_to_all_ones (numbers : List ℕ) (h : ∀ x ∈ numbers, 0 < x) (n : ℕ) (hn : 2 ≤ n) (hl : numbers.length = n) :
  can_make_all_one numbers :=
sorry

end transform_to_all_ones_l282_282771


namespace sqrt_equation_solution_l282_282170

theorem sqrt_equation_solution (x : ℝ) (h₀ : 9 < x) 
  (h₁ : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 21 := 
sorry

end sqrt_equation_solution_l282_282170


namespace angle_DFE_max_area_DIO_l282_282668

/-- Given triangle DEF where DE = EF = 1, and ∠DEF = 45°, determine ∠DFE
    when the area of triangle DIO is maximized, where H is the orthocenter,
    I is the incenter, and O is the circumcenter of triangle DEF. --/
theorem angle_DFE_max_area_DIO
  (D E F : Type*)
  (DE EF : ℝ)
  (angle_DEF : ℝ)
  (H I O : Type*)
  [orthocenter H D E F]
  [incenter I D E F]
  [circumcenter O D E F]
  (h1 : DE = 1)
  (h2 : EF = 1)
  (h3 : angle_DEF = 45) :
  ∃ (θ : ℝ), θ = 90 := sorry

end angle_DFE_max_area_DIO_l282_282668


namespace find_a1_of_geom_series_l282_282764

noncomputable def geom_series_sum (a₁ q : ℝ) (n : ℕ) : ℝ :=
  a₁ * (1 - q^n) / (1 - q)

theorem find_a1_of_geom_series (a₁ q : ℝ) (S : ℕ → ℝ)
  (h1 : S 6 = 9 * S 3)
  (h2 : S 5 = 62)
  (neq1 : q ≠ 1)
  (neqm1 : q ≠ -1) :
  a₁ = 2 :=
by
  have eq1 : S 6 = geom_series_sum a₁ q 6 := sorry
  have eq2 : S 3 = geom_series_sum a₁ q 3 := sorry
  have eq3 : S 5 = geom_series_sum a₁ q 5 := sorry
  sorry

end find_a1_of_geom_series_l282_282764


namespace sqrt_nested_expr_l282_282900

theorem sqrt_nested_expr (x : ℝ) (hx : 0 ≤ x) : 
  (x * (x * (x * x)^(1 / 2))^(1 / 2))^(1 / 2) = (x^7)^(1 / 4) :=
sorry

end sqrt_nested_expr_l282_282900


namespace sum_reciprocals_five_l282_282766

theorem sum_reciprocals_five (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 5 * x * y) : 
  1/x + 1/y = 5 :=
begin
  sorry
end

end sum_reciprocals_five_l282_282766


namespace simplify_fractions_sum_l282_282402

theorem simplify_fractions_sum :
  (48 / 72) + (30 / 45) = 4 / 3 := 
by
  sorry

end simplify_fractions_sum_l282_282402


namespace max_divisors_1_to_15_l282_282390

-- Define what it means for a number to be a divisor
def is_divisor (a b : ℕ) : Prop := b % a = 0

-- Define a function to count the number of divisors of a number
def num_divisors (n : ℕ) : ℕ := 
  (finset.range (n + 1)).filter (λ d, is_divisor d n).card

-- Define the set of numbers from 1 to 15
def numbers := (finset.range 16).filter (λ x, x > 0)

-- The theorem proving that 12 has the greatest number of divisors
theorem max_divisors_1_to_15 : 
  ∀ n ∈ numbers, num_divisors n ≤ num_divisors 12 := 
sorry

end max_divisors_1_to_15_l282_282390


namespace find_cylinder_radius_l282_282493

noncomputable def cylinder_radius (r : ℝ) : Prop :=
  let diameter_cone := 12
  let height_cone := 15
  let similarity_ratio := (height_cone - 2 * r) / r
  (diameter_cone / 2) / height_cone = similarity_ratio / r

theorem find_cylinder_radius : ∃ r : ℝ, cylinder_radius r ∧ r = 10 / 3 :=
begin
  sorry
end

end find_cylinder_radius_l282_282493


namespace thomas_friends_fraction_l282_282481

noncomputable def fraction_of_bars_taken (x : ℝ) (initial_bars : ℝ) (returned_bars : ℝ) 
  (piper_bars : ℝ) (remaining_bars : ℝ) : ℝ :=
  x / initial_bars

theorem thomas_friends_fraction 
  (initial_bars : ℝ)
  (total_taken_by_all : ℝ)
  (returned_bars : ℝ)
  (piper_bars : ℝ)
  (remaining_bars : ℝ)
  (h_initial : initial_bars = 200)
  (h_remaining : remaining_bars = 110)
  (h_taken : 200 - 110 = 90)
  (h_total_taken_by_all : total_taken_by_all = 90)
  (h_returned : returned_bars = 5)
  (h_x_calculation : 2 * (total_taken_by_all + returned_bars - initial_bars) + initial_bars = total_taken_by_all + returned_bars)
  : fraction_of_bars_taken ((total_taken_by_all + returned_bars - initial_bars) + 2 * initial_bars) initial_bars returned_bars piper_bars remaining_bars = 21 / 80 :=
  sorry

end thomas_friends_fraction_l282_282481


namespace new_person_weight_l282_282055

theorem new_person_weight
  (avg_increase : ℝ) (original_person_weight : ℝ) (num_people : ℝ) (new_weight : ℝ)
  (h1 : avg_increase = 2.5)
  (h2 : original_person_weight = 85)
  (h3 : num_people = 8)
  (h4 : num_people * avg_increase = new_weight - original_person_weight):
    new_weight = 105 :=
by
  sorry

end new_person_weight_l282_282055


namespace persistence_of_2_persistence_iff_2_l282_282514

def is_persistent (T : ℝ) : Prop :=
  ∀ (a b c d : ℝ), (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧
                    a ≠ 1 ∧ b ≠ 1 ∧ c ≠ 1 ∧ d ≠ 1) →
    (a + b + c + d = T) →
    (1 / a + 1 / b + 1 / c + 1 / d = T) →
    (1 / (1 - a) + 1 / (1 - b) + 1 / (1 - c) + 1 / (1 - d) = T)

theorem persistence_of_2 : is_persistent 2 :=
by
  -- The proof is omitted as per instructions
  sorry

theorem persistence_iff_2 (T : ℝ) : is_persistent T ↔ T = 2 :=
by
  -- The proof is omitted as per instructions
  sorry

end persistence_of_2_persistence_iff_2_l282_282514


namespace minimum_value_l282_282369

theorem minimum_value (a c : ℝ) (ha : a ≠ 0) (hc : c ≠ 0) :
  ∀ a c, a^2 + c^2 + 1/a^2 + c/a + 1/c^2 ≥ real.sqrt 15 :=
by
  sorry

end minimum_value_l282_282369


namespace prove_additional_minutes_needed_l282_282352

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l282_282352


namespace probability_three_people_pointing_each_other_l282_282433

theorem probability_three_people_pointing_each_other :
  ∃ (S : Finset (Fin 5)) (h : S.card = 3),
    let P := S,
    (∀ (x ∈ P) (y ∈ (P.erase x)),
      ∃ (z ∈ (P.erase x).erase y),
        true) -- Since every person must point at two others in the subset
→ ∑ (x ∈ P) ∑ (y ∈ (P.erase x)) ∑ (z ∈ (P.erase x).erase y),
    (1/(choose 4 2))^3 = 5/108
:= sorry

end probability_three_people_pointing_each_other_l282_282433


namespace adams_markup_l282_282500

def rate_of_markup (S : ℝ) (profit_percent expenses_percent : ℝ) : ℝ :=
  let C := S * (1 - (profit_percent + expenses_percent)) in
  ((S - C) / C) * 100

theorem adams_markup :
  rate_of_markup 10 0.12 0.18 = 42.857 := by
  sorry

end adams_markup_l282_282500


namespace highest_attendance_l282_282528

def available_on (person : String) (day : String) : Prop :=
  match person, day with
  | "Amy", "Mon" => True
  | "Amy", "Wed" => True
  | "Amy", "Thu" => True
  | "Bob", "Tue" => True
  | "Bob", "Fri" => True
  | "Charlie", "Mon" => True
  | "Charlie", "Tue" => True
  | "Charlie", "Wed" => True
  | "Charlie", "Sat" => True
  | "Diana", "Wed" => True
  | "Diana", "Thu" => True
  | "Diana", "Sat" => True
  | "Evan", "Mon" => True
  | "Evan", "Thu" => True
  | "Evan", "Fri" => True
  | _, _ => False

def attendance (day : String) : ℕ :=
  ["Amy", "Bob", "Charlie", "Diana", "Evan"].countP (λ person => available_on person day)

theorem highest_attendance :
  ∃ d : String, (attendance d = 3) ∧ (d = "Tue" ∨ d = "Fri" ∨ d = "Sat") :=
by  
  sorry

end highest_attendance_l282_282528


namespace all_numbers_even_l282_282581

theorem all_numbers_even
  (A B C D E : ℤ)
  (h1 : (A + B + C) % 2 = 0)
  (h2 : (A + B + D) % 2 = 0)
  (h3 : (A + B + E) % 2 = 0)
  (h4 : (A + C + D) % 2 = 0)
  (h5 : (A + C + E) % 2 = 0)
  (h6 : (A + D + E) % 2 = 0)
  (h7 : (B + C + D) % 2 = 0)
  (h8 : (B + C + E) % 2 = 0)
  (h9 : (B + D + E) % 2 = 0)
  (h10 : (C + D + E) % 2 = 0) :
  (A % 2 = 0) ∧ (B % 2 = 0) ∧ (C % 2 = 0) ∧ (D % 2 = 0) ∧ (E % 2 = 0) :=
sorry

end all_numbers_even_l282_282581


namespace rectangle_boundary_length_rounded_is_correct_l282_282839

noncomputable def rectangle_boundary_length : Real :=
  let w := Real.sqrt 36 in              -- Width w = 6 since w^2 = 36
  let l := 2 * w in                     -- Length l = 2w = 12
  let width_segments := 6 / 3 in        -- Each width segment = 2
  let length_segments := 12 / 3 in      -- Each length segment = 4
  let quarter_circle_arcs := 4 * Real.pi in -- 4 quarter-circle arcs, each length = π * 2
  let straight_segments := 8 + 16 in    -- Width segments (4 of 2 each) + length segments (4 of 4 each)
  let total_length := quarter_circle_arcs + straight_segments in
  total_length.toReal.roundToDigits 1    -- Rounded to nearest tenth

theorem rectangle_boundary_length_rounded_is_correct :
  rectangle_boundary_length = 36.6 := by
  sorry

end rectangle_boundary_length_rounded_is_correct_l282_282839


namespace polynomial_divides_l282_282690

theorem polynomial_divides (a b c d x y : ℤ) (h: x ≠ y) : 
  (x - y) ∣ (a * x^3 + b * x^2 + c * x + d - (a * y^3 + b * y^2 + c * y + d)) :=
by 
  -- Define the polynomial P(X)
  let P := λ X : ℤ, a * X^3 + b * X^2 + c * X + d
  -- State the desired assertion
  have h1 : (x - y) ∣ (P x - P y),
  -- Provide sorry for the proof
  sorry

-- The above imports necessary libraries, sets up the theorem statement, and asserts using the given conditions.

end polynomial_divides_l282_282690


namespace part1_property_P_part2_max_n_l282_282689

open Nat

-- Definitions based on given conditions
def f (a b : ℕ) (n : ℕ) : ℕ := (a + b) ^ n

-- Property P
def has_property_P (a b n : ℕ) : Prop :=
  ∃ k : ℕ, 1 ≤ k ∧ k ≤ n - 1 ∧ 
    binomial n (k - 1) + binomial n (k + 1) = 2 * binomial n k

-- Problem 1
theorem part1_property_P (a b : ℕ) : has_property_P a b 7 :=
sorry

-- Problem 2
theorem part2_max_n (a b : ℕ) : ∃ k, 1 ≤ k ∧ k ≤ 1934 - 1 ∧ 
  (binomial 1934 (k - 1) + binomial 1934 (k + 1) = 2 * binomial 1934 k) :=
sorry

end part1_property_P_part2_max_n_l282_282689


namespace P_of_7_l282_282374

noncomputable def P (x : ℝ) : ℝ := 12 * (x - 1) * (x - 2) * (x - 3) * (x - 4)^2 * (x - 5)^2 * (x - 6)

theorem P_of_7 : P 7 = 51840 :=
by
  sorry

end P_of_7_l282_282374


namespace score_at_least_118_l282_282862

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := 
  λ x, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * σ^2))

theorem score_at_least_118 
  (μ σ : ℝ) (μ_value : μ = 98) (σ_value : σ = 10)
  (n_students : ℕ) (rank : ℕ) 
  (prob_data : ∀ x σ, 
    x = μ - 2 * σ → 
    (μ + 2 * σ) = 118 → 
    prob_data = 0.9545 ∧ 0.5 * (1 - prob_data) = 0.02275)
  (top_rank_probability : ℝ) 
  (top_rank_probability_value : top_rank_probability = 9100 / 400000):
  ∃ X : ℝ, normal_distribution μ σ X ∧ X ≥ 118 :=
by
  sorry

end score_at_least_118_l282_282862


namespace triangle_cosine_quota_range_l282_282948

theorem triangle_cosine_quota_range 
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : 0 < A ∧ A < π/2)
  (h2 : 0 < B ∧ B < π/2)
  (h3 : 0 < C ∧ C < π/2)
  (angle_sum : A + B + C = π)
  (B_eq : B = π / 4)
  (acute_triangle : is_acute_triangle A B C) :
  -1 < (a * Real.cos C - c * Real.cos A) / b ∧ (a * Real.cos C - c * Real.cos A) / b < 1 :=
sorry

end triangle_cosine_quota_range_l282_282948


namespace number_of_students_in_Diligence_before_transfer_l282_282998

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l282_282998


namespace visitors_equal_cats_l282_282864

-- Definition for conditions
def visitors_pets_cats (V C : ℕ) : Prop :=
  (∃ P : ℕ, P = 3 * V ∧ P = 3 * C)

-- Statement of the proof problem
theorem visitors_equal_cats {V C : ℕ}
  (h : visitors_pets_cats V C) : V = C :=
by sorry

end visitors_equal_cats_l282_282864


namespace valid_passwords_count_l282_282857

-- Define the total number of unrestricted passwords (each digit can be 0-9)
def total_passwords := 10^5

-- Define the number of restricted passwords (those starting with the sequence 8,3,2)
def restricted_passwords := 10^2

-- State the main theorem to be proved
theorem valid_passwords_count : total_passwords - restricted_passwords = 99900 := by
  sorry

end valid_passwords_count_l282_282857


namespace sin_double_angle_cos_alpha_l282_282214

theorem sin_double_angle (α β : ℝ) (hα_range : α ∈ Ioo (π / 2) π)
  (hβ_range : β ∈ Ioo 0 (π / 2))
  (h_cos_diff : cos (α - β) = 1 / 7)
  (h_sum_angles : α + β = 2 * π / 3) :
  sin (2 * α - 2 * β) = 8 * sqrt 3 / 49 :=
sorry

theorem cos_alpha (α β : ℝ) (hα_range : α ∈ Ioo (π / 2) π)
  (hβ_range : β ∈ Ioo 0 (π / 2))
  (h_cos_diff : cos (α - β) = 1 / 7)
  (h_sum_angles : α + β = 2 * π / 3) :
  cos α = -sqrt 7 / 14 :=
sorry

end sin_double_angle_cos_alpha_l282_282214


namespace sum_of_symmetric_points_on_curve_l282_282237

noncomputable def symmetric_point_sum (p q : ℝ × ℝ) : Prop :=
  let f := λ x : ℝ, x^3 + 3 * x^2 + x
  (p.2 = f p.1) ∧ (q.2 = f q.1) ∧ (p ≠ q) ∧ 
  (p.1 + q.1 = -2 * (-1)) -- Center at x=-1

theorem sum_of_symmetric_points_on_curve : 
  ∃ y : ℝ,
  ∀ p q : ℝ × ℝ,
  symmetric_point_sum p q → p.2 + q.2 = y :=
sorry

end sum_of_symmetric_points_on_curve_l282_282237


namespace min_L_pieces_correct_l282_282450

noncomputable def min_L_pieces : ℕ :=
  have pieces : Nat := 11
  pieces

theorem min_L_pieces_correct :
  min_L_pieces = 11 := 
by
  sorry

end min_L_pieces_correct_l282_282450


namespace f_at_one_f_decreasing_f_min_on_interval_l282_282223

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
axiom f_defined : ∀ x, 0 < x → ∃ y, f y = y
axiom f_eq : ∀ x1 x2, 0 < x1 → 0 < x2 → f (x1 / x2) = f x1 - f x2
axiom f_neg : ∀ x, 1 < x → f x < 0

-- Proof statements
theorem f_at_one : f 1 = 0 := sorry

theorem f_decreasing : ∀ x1 x2, 0 < x1 → 0 < x2 → x1 < x2 → f x1 > f x2 := sorry

axiom f_at_three : f 3 = -1

theorem f_min_on_interval : ∀ x, 2 ≤ x ∧ x ≤ 9 → f x ≥ -2 := sorry

end f_at_one_f_decreasing_f_min_on_interval_l282_282223


namespace sum_of_specific_primes_l282_282141

theorem sum_of_specific_primes : 
  ∑ p in { p : ℕ | Prime p ∧ ¬ ∃ x : ℤ, 3 * (6 * x + 1) ≡ 4 [MOD p] }, p = 5 :=
by sorry

end sum_of_specific_primes_l282_282141


namespace solve_for_x_l282_282169

theorem solve_for_x (x : ℝ) (hx : x > 9) 
  (h : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 18 := 
  sorry

end solve_for_x_l282_282169


namespace find_shift_b_l282_282888

-- Define the periodic function f
variable (f : ℝ → ℝ)
-- Define the condition on f
axiom f_periodic : ∀ x, f (x - 30) = f x

-- The theorem we want to prove
theorem find_shift_b : ∃ b > 0, (∀ x, f ((x - b) / 3) = f (x / 3)) ∧ b = 90 := 
by
  sorry

end find_shift_b_l282_282888


namespace MF1_dot_MF2_range_proof_l282_282972

noncomputable def MF1_dot_MF2_range : Set ℝ :=
  Set.Icc (24 - 16 * Real.sqrt 3) (24 + 16 * Real.sqrt 3)

theorem MF1_dot_MF2_range_proof :
  ∀ (M : ℝ × ℝ), (Prod.snd M + 4) ^ 2 + (Prod.fst M) ^ 2 = 12 →
    (Prod.fst M) ^ 2 + (Prod.snd M) ^ 2 - 4 ∈ MF1_dot_MF2_range :=
by
  sorry

end MF1_dot_MF2_range_proof_l282_282972


namespace arithmetic_sequence_common_difference_l282_282744

theorem arithmetic_sequence_common_difference (d : ℚ) (a₁ : ℚ) (h : a₁ = -10)
  (h₁ : ∀ n ≥ 10, a₁ + (n - 1) * d > 0) :
  10 / 9 < d ∧ d ≤ 5 / 4 :=
by
  sorry

end arithmetic_sequence_common_difference_l282_282744


namespace count_prime_divisors_fact_plus_square_eq_pi_l282_282520

open Nat

/--
The number of distinct prime divisors of \( n! + n^2 \) for any \( n > 1 \) 
is equal to the number of primes less than or equal to \( n \).
-/
theorem count_prime_divisors_fact_plus_square_eq_pi (n : ℕ) (h : n > 1) :
  (count_factors (factorize (fact n + n^2))).keys.length = primeCount n := 
sorry

end count_prime_divisors_fact_plus_square_eq_pi_l282_282520


namespace perfect_squares_two_digit_divisible_by_4_count_l282_282268

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l282_282268


namespace sqrt_meaningful_for_x_l282_282321

theorem sqrt_meaningful_for_x (x : ℝ) : (∃ r : ℝ, r = real.sqrt (x - 3)) ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_for_x_l282_282321


namespace min_value_f_side_c_length_l282_282963

-- Define the function and necessary conditions
noncomputable def f (x : ℝ) : ℝ := sin (2 * x + π / 6) + 2 * sin x ^ 2

-- Condition: Graph passes through (π/6, 3/2)
axiom f_passing : f (π / 6) = 3 / 2

-- Condition: Absolute value of φ < π / 2
axiom phi_bound : |π / 6| < π / 2

-- Define function on the interval [0, π/2]
def f_interval (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) : ℝ := f x

-- Triangle conditions: acute angle C, area, side lengths
axiom triangle_ABC : ∃ (a b c : ℝ), 0 < C ∧ C < π / 2 ∧ a + b = 6 ∧ 1/2 * a * b * sin C = 2 * sqrt 3 ∧ x = C ∧ ab = a * b = 8 ∧ (c * c = a * a + b * b - 2 * a * b * cos C)

-- Prove the minimum value of f(x) on [0, π/2] is 1/2
theorem min_value_f : ∃ x, 0 ≤ x ∧ x ≤ π / 2 ∧ f x = 1 / 2 := 
sorry

-- Prove that the length of side c of triangle ABC is 2 * sqrt 3
theorem side_c_length : ∃ c, c = 2 * sqrt 3 :=
sorry


end min_value_f_side_c_length_l282_282963


namespace geometry_problem_l282_282778

/-
  Two congruent circles are centered at points X and Y respectively,
  each passing through the center of the other. A line parallel to the line XY
  intersects the two circles at points M, N on one circle and P, Q on the other, respectively.
  Prove that the degree measure of ∠MQP is 90°.
-/

noncomputable def problem : Prop :=
∀ (X Y M N P Q : Point),
  (congruent_circles X Y) ∧
  (passes_through_center X Y M N P Q) ∧
  (parallel (line XY) (line MNPQ)) →
  (angle_degrees M Q P = 90)

theorem geometry_problem : problem := 
sorry

end geometry_problem_l282_282778


namespace expression_value_l282_282950

theorem expression_value (m n a b x : ℤ) (h1 : m = -n) (h2 : a * b = 1) (h3 : |x| = 3) :
  x = 3 ∨ x = -3 → (x = 3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = 26) ∧
                  (x = -3 → x^3 - (1 + m + n - a * b) * x^2010 + (m + n) * x^2007 + (-a * b)^2009 = -28) := by
  sorry

end expression_value_l282_282950


namespace initial_amount_l282_282837

theorem initial_amount (X : ℚ) (F : ℚ) :
  (∀ (X F : ℚ), F = X * (3/4)^3 → F = 37 → X = 37 * 64 / 27) :=
by
  sorry

end initial_amount_l282_282837


namespace find_workers_l282_282432

def total_workers := 20
def male_work_days := 2
def female_work_days := 3

theorem find_workers (X Y : ℕ) 
  (h1 : X + Y = total_workers)
  (h2 : X / male_work_days + Y / female_work_days = 1) : 
  X = 12 ∧ Y = 8 :=
sorry

end find_workers_l282_282432


namespace speed_of_man_in_still_water_correct_l282_282047

def upstream_speed : ℝ := 25 -- Upstream speed in kmph
def downstream_speed : ℝ := 39 -- Downstream speed in kmph
def speed_in_still_water : ℝ := 32 -- The speed of the man in still water

theorem speed_of_man_in_still_water_correct :
  (upstream_speed + downstream_speed) / 2 = speed_in_still_water :=
by
  sorry

end speed_of_man_in_still_water_correct_l282_282047


namespace exist_bounded_sum_c_le_4_c_eq_3_c_gt_3_false_l282_282655

-- Define the infinite sheet of graph paper with a number written in each cell
noncomputable def grid (x y : ℤ) : ℝ := sorry

-- Condition: The sum of any square's numbers does not exceed 1 in absolute value
def square_sum_condition : Prop :=
  ∀ x y k, (∑ i in finset.range k, ∑ j in finset.range k, grid (x + i) (y + j)) ≤ 1

-- (a) Prove the existence of a constant c such that the sum of the numbers in any rectangle is at most c
theorem exist_bounded_sum (h : square_sum_condition) : ∃ c : ℝ, ∀ a b x y, 
  (∑ i in finset.range a, ∑ j in finset.range b, grid (x + i) (y + j)) ≤ c :=
sorry

-- (b) Prove that c can be taken as 4
theorem c_le_4 (h : square_sum_condition) : 
  ∃ c : ℝ, c = 4 ∧ (∀ a b x y, (∑ i in finset.range a, ∑ j in finset.range b, grid (x + i) (y + j)) ≤ c) :=
sorry

-- (c) Prove that c = 3
theorem c_eq_3 (h : square_sum_condition) : 
  ∃ c : ℝ, c = 3 ∧ (∀ a b x y, (∑ i in finset.range a, ∑ j in finset.range b, grid (x + i) (y + j)) ≤ c) :=
sorry

-- (d) Construct an example showing that for c > 3, the statement is false
theorem c_gt_3_false (h : square_sum_condition) : 
  ¬ (∀ c : ℝ, c > 3 → ∀ a b x y, (∑ i in finset.range a, ∑ j in finset.range b, grid (x + i) (y + j)) ≤ c) :=
sorry

end exist_bounded_sum_c_le_4_c_eq_3_c_gt_3_false_l282_282655


namespace log_square_value_l282_282796

noncomputable def log_10 (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_square_value :
  (log_10 (3 * log_10 1000)) ^ 2 = 0.910116 :=
by
  have h1 : log_10 1000 = 3 := by sorry
  have h2 : log_10 (3 * 3) = log_10 9 := by sorry
  have h3 : log_10 9 = 0.954 := by sorry
  show (log_10 (3 * log_10 1000)) ^ 2 = 0.910116 from by sorry

end log_square_value_l282_282796


namespace sum_of_first_eight_terms_l282_282192

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l282_282192


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282274

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282274


namespace num_elements_in_M_inter_N_l282_282609

def is_nat (n : ℤ) := 0 ≤ n

def M := {x : ℤ | -2 ≤ x ∧ x < 4 ∧ is_nat x}
def N := {x : ℤ | (x + 1) * (3 - x) ≥ 0}

theorem num_elements_in_M_inter_N : 
  (finset.filter (λ x, x ∈ M ∧ x ∈ N) (finset.range 4)).card = 3 := 
by
  sorry

end num_elements_in_M_inter_N_l282_282609


namespace value_of_x_l282_282990

noncomputable def x : ℕ :=
  let start := 50
  let end := 70
  (end - start + 1) * ((start + end) / 2)

noncomputable def y : ℕ :=
  let start := 50
  let end := 70
  (end - start) / 2 + 1

theorem value_of_x :
  x + y = 1271 → x = 1260 := by
  sorry

end value_of_x_l282_282990


namespace paper_folding_shapes_and_areas_l282_282461

variable (n : ℕ)

def initial_area := 240

theorem paper_folding_shapes_and_areas :
  let number_of_shapes := n + 1 in
  let area_sum := initial_area * (3 - (n + 3) / (2 ^ n : ℝ)) in
  (number_of_shapes = n + 1) ∧ 
  (∑ k in Finset.range n, initial_area * ((k + 1) / (2 ^ k : ℝ))) = area_sum :=
sorry

end paper_folding_shapes_and_areas_l282_282461


namespace similar_right_triangle_hypotenuse_length_l282_282090

theorem similar_right_triangle_hypotenuse_length :
  ∀ (a b c d : ℝ), a = 15 → c = 39 → d = 45 → 
  (b^2 = c^2 - a^2) → 
  ∃ e : ℝ, e = (c * (d / b)) ∧ e = 48.75 :=
by
  intros a b c d ha hc hd hb
  sorry

end similar_right_triangle_hypotenuse_length_l282_282090


namespace laptop_sticker_price_l282_282345

theorem laptop_sticker_price (x : ℝ) (h₁ : 0.70 * x = 0.80 * x - 50 - 30) : x = 800 := 
  sorry

end laptop_sticker_price_l282_282345


namespace knight_exits_l282_282819

-- Definitions based on the conditions provided in the problem
def Hall : Type := ℕ
def Door : Type := ℕ
def Outside := Unit

structure Castle where
  doors : ℕ
  leads_to : Door → (Hall ⊕ Outside)
  halls : Hall → ℕ -- number of doors in each hall

variables (c : Castle)

-- Hypothesis based on the problem conditions
axiom (h_door_count : c.doors ≥ 1)
axiom (h_halls_door_count : ∀ h : Hall, c.halls h ≥ 2)

-- Statement of the problem
theorem knight_exits (n : ℕ) (h : Castle) (k : ℕ) : (∃ strat : (Hall → Hall), (∀ t : ℕ, t ≤ 2 * n - 4) → strat (2 * k) = ⟨⟩)
        sorry

end knight_exits_l282_282819


namespace probability_p_satisfies_equation_l282_282628

theorem probability_p_satisfies_equation :
  let S := {p : ℕ | 1 ≤ p ∧ p ≤ 15 ∧ ∃ q : ℤ, p * q - 5 * p - 3 * q = -1},
      N := {p : ℕ | 1 ≤ p ∧ p ≤ 15} in
  ∃ fraction : ℚ,
  (fraction = 1 / 3) ∧
  (↑(Finset.card (Finset.filter (λ (p : ℕ), p ∈ S) (Finset.filter (λ (p : ℕ), p ∈ N) (Finset.range 16))) : ℚ) =
   fraction * ↑(Finset.card (Finset.filter (λ (p : ℕ), p ∈ N) (Finset.range 16)) : ℚ)) :=
sorry

end probability_p_satisfies_equation_l282_282628


namespace sum_of_squares_of_roots_l282_282558

theorem sum_of_squares_of_roots :
  (∃ r1 r2 : ℝ, (r1 + r2 = 10 ∧ r1 * r2 = 16) ∧ (r1^2 + r2^2 = 68)) :=
by
  sorry

end sum_of_squares_of_roots_l282_282558


namespace probability_neither_prime_nor_composite_l282_282053

theorem probability_neither_prime_nor_composite : 
  let set_of_numbers := finset.range (101) in
  let neither_prime_nor_composite := {1} in
  let total_numbers := set_of_numbers.card in
  let favorable_cases := neither_prime_nor_composite.card in
  (favorable_cases : ℝ) / total_numbers = 1 / 100 :=
sorry

end probability_neither_prime_nor_composite_l282_282053


namespace orchid_bushes_in_park_l282_282002

theorem orchid_bushes_in_park (current_bushes new_bushes : ℕ) (h₁ : current_bushes = 2) (h₂ : new_bushes = 4) : current_bushes + new_bushes = 6 := 
by
  rw [h₁, h₂]
  sorry

end orchid_bushes_in_park_l282_282002


namespace victoria_money_given_l282_282025

noncomputable def total_money_given : ℕ :=
  let pizza_cost := 2 * 12 in
  let juice_cost := 2 * 2 in
  let total_spent := pizza_cost + juice_cost in
  let amount_to_return := 22 in
  total_spent + amount_to_return 

theorem victoria_money_given : total_money_given = 50 := by
  sorry

end victoria_money_given_l282_282025


namespace find_f_sqrt50_l282_282693

noncomputable def f (x : ℝ) : ℝ :=
if x.is_integer then 7 * x + 3 else floor x + 7

theorem find_f_sqrt50 : f (Real.sqrt 50) = 14 := by
  sorry

end find_f_sqrt50_l282_282693


namespace number_of_solutions_l282_282179

noncomputable def f (x : ℝ) : ℝ := 
  ∑ n in Finset.range 100, if even n then 1 / (x - (2 * n + 1)) else - (2 * n + 1) / (x - (2 * n + 1))

theorem number_of_solutions : 
  let num_solutions := 
    (Finset.range 100).filter (λ n, f (2 * n + 1) * f (2 * (n + 1) + 1) < 0)).card + 1 in
  num_solutions = 101 :=
by sorry

end number_of_solutions_l282_282179


namespace largest_prime_factor_3113_l282_282031

theorem largest_prime_factor_3113 : ∃ p : ℕ, prime p ∧ p ∣ 3113 ∧ ∀ q : ℕ, prime q ∧ q ∣ 3113 → q ≤ p := 
sorry

end largest_prime_factor_3113_l282_282031


namespace my_age_now_l282_282065

theorem my_age_now (Y S : ℕ) (h1 : Y - 9 = 5 * (S - 9)) (h2 : Y = 3 * S) : Y = 54 := by
  sorry

end my_age_now_l282_282065


namespace greatest_cars_with_ac_not_racing_stripes_l282_282054

def total_cars : ℕ := 100
def without_ac : ℕ := 49
def at_least_racing_stripes : ℕ := 51

theorem greatest_cars_with_ac_not_racing_stripes :
  (total_cars - without_ac) - (at_least_racing_stripes - without_ac) = 49 :=
by
  unfold total_cars without_ac at_least_racing_stripes
  sorry

end greatest_cars_with_ac_not_racing_stripes_l282_282054


namespace permutation_count_l282_282331

-- Define the sequence as a list
def seq : List ℕ := [1, 2, 3, 4, 5, 6]

-- Function to check if a sequence has no four consecutive terms increasing or decreasing.
def no_four_consecutive_increasing_or_decreasing (l : List ℕ) : Prop :=
  ∀ (i : ℕ), i + 3 < l.length →
    ¬ ((l.get? i < l.get? (i + 1)) ∧ (l.get? (i + 1) < l.get? (i + 2)) ∧ (l.get? (i + 2) < l.get? (i + 3))) ∧ ¬ ((l.get? i > l.get? (i + 1)) ∧ (l.get? (i + 1) > l.get? (i + 2)) ∧ (l.get? (i + 2) > l.get? (i + 3)))

-- The main theorem to prove
theorem permutation_count : 
  ( (List.permutations seq).filter no_four_consecutive_increasing_or_decreasing ).length = 16 := 
  sorry

end permutation_count_l282_282331


namespace area_of_AOB_intersection_points_l282_282931

-- Definition of the parametric equations for curve C
def curve_C (φ : ℝ) : ℝ × ℝ :=
  (2 * Real.cos φ, Real.sin φ)

-- Definition of the parametric equations for line l
def line_l (t : ℝ) : ℝ × ℝ :=
  (2 + (Real.sqrt 2 / 2) * t, (Real.sqrt 2 / 2) * t)

-- Theorem 1: Area of triangle ΔAOB
theorem area_of_AOB :
  let OA := Real.sqrt (8 / 5),
      OB := Real.sqrt (8 / 5) in
  (1 / 2) * OA * OB = 4 / 5 :=
sorry

-- Theorem 2: Intersection points of curve C and line l
theorem intersection_points :
  (∃ φ t : ℝ, curve_C φ = line_l t) →
  (curve_C 0 = (2, 0)) ∧ (curve_C (-4 * (Real.sqrt 2 / 5)) = (6 / 5, -4 / 5)) :=
sorry

end area_of_AOB_intersection_points_l282_282931


namespace regular_octagon_side_length_l282_282387

theorem regular_octagon_side_length
  (side_length_pentagon : ℕ)
  (total_wire_length : ℕ)
  (side_length_octagon : ℕ) :
  side_length_pentagon = 16 →
  total_wire_length = 5 * side_length_pentagon →
  side_length_octagon = total_wire_length / 8 →
  side_length_octagon = 10 := 
sorry

end regular_octagon_side_length_l282_282387


namespace doubled_cylinder_volume_l282_282495

-- Given conditions
def original_volume (r h : ℝ) : ℝ := π * r^2 * h
def new_volume (r h : ℝ) : ℝ := π * (2 * r)^2 * (2 * h)

-- The proof problem statement
theorem doubled_cylinder_volume (r h : ℝ) (hV : original_volume r h = 6) : new_volume r h = 48 :=
by
  let h1 : original_volume r h = π * r^2 * h := rfl
  let h2 : new_volume r h = π * (2 * r)^2 * (2 * h) := rfl
  have h3 : h1 ▸ hV = 6 := hV
  suffices new_volume r h = 8 * (π * r^2 * h) by sorry
  show new_volume r h = 8 * 6 from sorry
  sorry

end doubled_cylinder_volume_l282_282495


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282287

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282287


namespace exists_pairs_of_real_numbers_l282_282341

theorem exists_pairs_of_real_numbers :
  ∃ (m k : ℝ), k^4 + (m - 1)^2 = 5 ∧ k < 0 ∧ (1 - Real.sqrt 5 ≤ m ∧ m ≤ 1 + Real.sqrt 5) ∧ k ∈ ℤ := 
sorry

end exists_pairs_of_real_numbers_l282_282341


namespace sum_geometric_sequence_l282_282183

theorem sum_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 1/3) (h_r : r = 1/3) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in S_n = 3280/6561 :=
by
  sorry

end sum_geometric_sequence_l282_282183


namespace A_B_days_together_l282_282476

variable (W : ℝ) -- total work
variable (x : ℝ) -- days A and B worked together
variable (A_B_rate : ℝ) -- combined work rate of A and B
variable (A_rate : ℝ) -- work rate of A
variable (B_days : ℝ) -- days A worked alone after B left

-- Conditions:
axiom condition1 : A_B_rate = W / 40
axiom condition2 : A_rate = W / 80
axiom condition3 : B_days = 6
axiom condition4 : (x * A_B_rate + B_days * A_rate = W)

-- We want to prove that x = 37:
theorem A_B_days_together : x = 37 :=
by
  sorry

end A_B_days_together_l282_282476


namespace ceil_floor_sum_l282_282149

theorem ceil_floor_sum :
  (Int.ceil (7 / 3 : ℚ)) + (Int.floor (-7 / 3 : ℚ)) = 0 := 
sorry

end ceil_floor_sum_l282_282149


namespace quadrilateral_area_correct_l282_282838

def point := (ℝ × ℝ)

def quadrilateral_area (A B C D : point) : ℝ :=
  let area_triangle (P Q R : point) : ℝ :=
    abs (P.1 * (Q.2 - R.2) + Q.1 * (R.2 - P.2) + R.1 * (P.2 - Q.2)) / 2
  area_triangle A B C + area_triangle A C D

def A : point := (1, 1)
def B : point := (1, 5)
def C : point := (3, 5)
def D : point := (2006, 2003)

theorem quadrilateral_area_correct :
  quadrilateral_area A B C D = 4014 :=
by
  sorry

end quadrilateral_area_correct_l282_282838


namespace prism_volume_is_correct_l282_282089

-- Define geometric terms and conditions
structure Point3D :=
(x : ℝ)
(y : ℝ)
(z : ℝ)

structure TrianglePrism :=
(A B C A1 B1 C1 : Point3D)
(K : Point3D)  -- Midpoint of AA1
(D : Point3D)  -- A point such that CD is diameter
(r : ℝ)  -- Sphere radius

noncomputable def volume_prism (prism : TrianglePrism) : ℝ :=
  let h := 2 * Real.sqrt 6 in  -- Height h is calculated as 2sqrt(6)
  let s := Real.sqrt 6 in       -- Side length s calculated as sqrt(6)
  let area_base := (Real.sqrt 3 / 4) * (s^2) in  -- Area of the equilateral triangle base
  area_base * h

-- Lean theorem statement for the volume of the triangular prism
theorem prism_volume_is_correct
  (A B C A1 B1 C1 K D : Point3D)
  (h : ℝ)
  (r : ℝ)
  (hyp1 : dist C K = 2 * Real.sqrt 3)
  (hyp2 : dist D K = 2 * Real.sqrt 2)
  (prism : TrianglePrism)
  (hyp_prism : prism = { A := A, B := B, C := C, A1 := A1, B1 := B1, C1 := C1, K := K, D := D, r := r}) :
  volume_prism prism = 9 * Real.sqrt 2 := by
  sorry

end prism_volume_is_correct_l282_282089


namespace percentage_refund_l282_282466

theorem percentage_refund
  (initial_amount : ℕ)
  (sweater_cost : ℕ)
  (tshirt_cost : ℕ)
  (shoes_cost : ℕ)
  (amount_left_after_refund : ℕ)
  (refund_percentage : ℕ) :
  initial_amount = 74 →
  sweater_cost = 9 →
  tshirt_cost = 11 →
  shoes_cost = 30 →
  amount_left_after_refund = 51 →
  refund_percentage = 90 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end percentage_refund_l282_282466


namespace find_b_l282_282957

noncomputable def given_complex (b : ℝ) := (2 - complex.I * b) / (1 + 2 * complex.I)

theorem find_b (b : ℝ) (h : (complex.re (given_complex b)) = - (complex.im (given_complex b))) : b = -2 :=
by
  sorry

end find_b_l282_282957


namespace problem1_problem2_l282_282061

open Real

/-- Problem 1: Simplify trigonometric expression. -/
theorem problem1 : 
  (sqrt (1 - 2 * sin (10 * pi / 180) * cos (10 * pi / 180)) /
  (sin (170 * pi / 180) - sqrt (1 - sin (170 * pi / 180)^2))) = -1 :=
sorry

/-- Problem 2: Given tan(θ) = 2, find the value.
  Required to prove: 2 + sin(θ) * cos(θ) - cos(θ)^2 equals 11/5 -/
theorem problem2 (θ : ℝ) (h : tan θ = 2) :
  2 + sin θ * cos θ - cos θ^2 = 11 / 5 :=
sorry

end problem1_problem2_l282_282061


namespace problem1_problem2_l282_282695

-- Define propositions P and Q under the given conditions
def P (a x : ℝ) : Prop := 2 * x^2 - 5 * a * x - 3 * a^2 < 0

def Q (x : ℝ) : Prop := (2 * Real.sin x > 1) ∧ (x^2 - x - 2 < 0)

-- Problem 1: Prove that if a = 2 and p ∧ q holds true, then the range of x is (π/6, 2)
theorem problem1 (x : ℝ) (hx1 : P 2 x ∧ Q x) : (Real.pi / 6 < x ∧ x < 2) :=
sorry

-- Problem 2: Prove that if ¬P is a sufficient but not necessary condition for ¬Q, then the range of a is [2/3, ∞)
theorem problem2 (a : ℝ) (h₁ : ∀ x, Q x → P a x) (h₂ : ∃ x, Q x → ¬P a x) : a ≥ 2 / 3 :=
sorry

end problem1_problem2_l282_282695


namespace find_original_number_l282_282798

theorem find_original_number : ∃ (N : ℤ), (∃ (k : ℤ), N - 30 = 87 * k) ∧ N = 117 :=
by
  sorry

end find_original_number_l282_282798


namespace multiplicative_functions_proof_l282_282006

theorem multiplicative_functions_proof 
  (a b c d : ℕ) 
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d)
  (h_gcd_abcd : a * b * c * d ≠ 1 ∧ 
                gcd a b = 1 ∧ gcd a c = 1 ∧ gcd a d = 1 ∧ 
                gcd b c = 1 ∧ gcd b d = 1 ∧ gcd c d = 1)
  (f g : ℕ → ℕ)
  (h_fg_range : ∀ n, f(n) ∈ {0, 1} ∧ g(n) ∈ {0, 1})
  (h_fg_mult : ∀ m n, f(m * n) = f(m) * f(n) ∧ g(m * n) = g(m) * g(n))
  (h_fg_eq : ∀ n, f(a * n + b) = g(c * n + d)) :
  (∀ n, f(a * n + b) = 0 ∧ g(c * n + d) = 0) ∨ 
  (∃ k > 0, ∀ n, gcd n k = 1 → f(n) = 1 ∧ g(n) = 1) :=
sorry

end multiplicative_functions_proof_l282_282006


namespace bonus_for_each_100_cards_l282_282877

theorem bonus_for_each_100_cards (earnings_per_card : ℕ → ℕ) (bonus_per_100_cards : ℕ → ℕ) :
  earnings_per_card 200 + bonus_per_100_cards 2 = 160 → earnings_per_card 200 = 140 → bonus_per_100_cards 2 = 20 → bonus_per_100_cards 1 = 10 :=
by
  intros h1 h2 h3
  have h4 : bonus_per_100_cards 1 = bonus_per_100_cards 2 / 2 := sorry
  rw h3 at h4
  exact sorry

end bonus_for_each_100_cards_l282_282877


namespace fraction_identity_l282_282150

variable (a b : ℝ)

theorem fraction_identity (h : a ≠ b) :
  (a⁻¹ - b⁻¹) ≠ 0 → (a⁻³ - b⁻³) / (a⁻¹ - b⁻¹) = a⁻² + a⁻¹ * b⁻¹ + b⁻² :=
by
  sorry

end fraction_identity_l282_282150


namespace scientific_notation_of_393000_l282_282739

theorem scientific_notation_of_393000 : 
  ∃ (a : ℝ) (n : ℤ), a = 3.93 ∧ n = 5 ∧ (393000 = a * 10^n) := 
by
  use 3.93
  use 5
  sorry

end scientific_notation_of_393000_l282_282739


namespace nine_digit_palindrome_count_l282_282024

theorem nine_digit_palindrome_count :
  let digits := {5, 6, 7, 8, 9} in
  ∃ n, (∀ m ∈ digits, m < 10) ∧
  set.to_finset {x ∈ digits | x ≠ 0}.card = 5 ∧
  let palindrome_count := (5:ℤ) ^ 5 in
  3125 = palindrome_count :=
by
  let digits := {5, 6, 7, 8, 9}
  use (5:ℤ) ^ 5
  split
  {
    intros m hm
    fin_cases hm <;> simp
  }
  split
  {
    rw [set.to_finset_card, nat.card_eq_fintype_card]
    dec_trivial
  }
  simp
  sorry

end nine_digit_palindrome_count_l282_282024


namespace log_equation_solution_l282_282717

theorem log_equation_solution (x : ℝ) (h : log 3 x + log 9 x = 5) : x = 3^(10 / 3) := 
by
  sorry

end log_equation_solution_l282_282717


namespace part_one_solution_part_two_solution_l282_282602

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - a| + |x - 2|

-- Part (1): "When a = 1, find the solution set of the inequality f(x) ≥ 3"
theorem part_one_solution (x : ℝ) : f x 1 ≥ 3 ↔ x ≤ 0 ∨ x ≥ 3 :=
by sorry

-- Part (2): "If f(x) ≥ 2a - 1, find the range of values for a"
theorem part_two_solution (a : ℝ) : (∀ x : ℝ, f x a ≥ 2 * a - 1) ↔ a ≤ 1 :=
by sorry

end part_one_solution_part_two_solution_l282_282602


namespace final_concentration_of_salt_solution_l282_282827

theorem final_concentration_of_salt_solution
  (initial_concentration : ℝ)
  (volume_ratios : ℕ × ℕ × ℕ)
  (small_ball_overflow : ℝ)
  (total_initial_volume : ℝ)
  (refill_with_pure_water : ℝ) :
  initial_concentration = 0.15 →
  volume_ratios = (10, 5, 3) →
  small_ball_overflow = 0.1 * total_initial_volume →
  refill_with_pure_water = total_initial_volume →
  (let initial_salt_volume := initial_concentration * total_initial_volume in
   let overflow_volumes := [0.1 * total_initial_volume, 
                            0.1 * total_initial_volume * 5 / 3, 
                            0.1 * total_initial_volume * 10 / 3] in
   let remaining_volume := total_initial_volume - overflow_volumes.sum in
   let final_salt_concentration := initial_salt_volume / total_initial_volume in
   final_salt_concentration = 15%
  ) sorry

end final_concentration_of_salt_solution_l282_282827


namespace people_in_last_group_l282_282884

open Nat

theorem people_in_last_group :
  let questions_first_group := 6 * 2,
      questions_second_group := 11 * 2,
      questions_third_group := (8 - 1) * 2 + 3 * 2,
      questions_total := 68,
      questions_first_three_groups := questions_first_group + questions_second_group + questions_third_group,
      questions_last_group := questions_total - questions_first_three_groups,
      people_last_group := questions_last_group / 2 
  in people_last_group = 7 :=
by
  let questions_first_group := 6 * 2;
  let questions_second_group := 11 * 2;
  let questions_third_group := (8 - 1) * 2 + 3 * 2;
  let questions_total := 68;
  let questions_first_three_groups := questions_first_group + questions_second_group + questions_third_group;
  let questions_last_group := questions_total - questions_first_three_groups;
  let people_last_group := questions_last_group / 2;
  have : people_last_group = 7, from rfl;
  exact this

end people_in_last_group_l282_282884


namespace traversable_2015_checkerboard_l282_282734

-- We need to make the function noncomputable due to the complexity of traversal and checking such a large board.
noncomputable def is_traversable (n: ℕ) (start: prod ℕ ℕ) (end: prod ℕ ℕ): Prop :=
  -- Define the checkerboard properties, move conditions, starting and ending positions. 
  -- The specifics of the movement and board traversal are abstracted for this statement.

  ∀ (board: fin n × fin n → bool) (h_checkerboard: ∀ i j, board (i, j) = (i + j) % 2 = 0),
  ∃ (path: list (prod (fin n) (fin n))),
    (path.head = start) ∧ (path.last = end) ∧
    (∀ (i ∈ finset.range (path.length - 1)),
      (let ⟨x1, y1⟩ := path.nth_le i _ in
      let ⟨x2, y2⟩ := path.nth_le (i + 1) _ in
      (x1 = x2 ∧ abs (y1 - y2) = 1 ∨ abs (x1 - x2) = 1 ∧ y1 = y2))) ∧
    (path.to_finset.card = n*n) -- Ensure every cell is covered exactly once.

-- Define the problem for a 2015×2015 board with the token starting and ending at black cells.
theorem traversable_2015_checkerboard:
  is_traversable 2015 (0, 0) (2014, 2014)
:= sorry -- Proof would be provided here, omitted as per instructions.

end traversable_2015_checkerboard_l282_282734


namespace scientific_notation_of_393000_l282_282738

theorem scientific_notation_of_393000 :
  ∃ a n : ℝ, (1 ≤ a ∧ a < 10) ∧ n ∈ ℤ ∧ 393000 = a * 10^n ∧ a = 3.93 ∧ n = 5 :=
by
  use 3.93
  use 5
  split
  { split
    { norm_num }
    { norm_num } }
  split
  { norm_num }
  split
  { exact rfl }
  split
  { exact rfl }
  { exact rfl }
  sorry

end scientific_notation_of_393000_l282_282738


namespace sign_of_cos_cos1_sin_sin1_l282_282803

theorem sign_of_cos_cos1_sin_sin1 :
  ∀ (x y : ℝ), (0 < x ∧ x < 1) →
               (0 < y ∧ y < 1) →
               (∀ u v : ℝ, (0 ≤ u ∧ u ≤ v ∧ v ≤ π / 2) → cos u ≥ cos v) →
               (∀ u v : ℝ, (0 ≤ u ∧ u ≤ v ∧ v ≤ π / 2) → sin u ≤ sin v) →
  (cos (cos 1) - cos 1) * (sin (sin 1) - sin 1) < 0 := 
by
  assume x y hx hy hdecreasing_cos hincreasing_sin
  sorry

end sign_of_cos_cos1_sin_sin1_l282_282803


namespace period_of_sin_sub_cos_l282_282452

open Real

theorem period_of_sin_sub_cos :
  ∃ T > 0, ∀ x, sin x - cos x = sin (x + T) - cos (x + T) ∧ T = 2 * π := sorry

end period_of_sin_sub_cos_l282_282452


namespace solve_initial_apple_count_l282_282818

noncomputable def initial_apple_count (A : ℕ) : Prop :=
  let apples_for_pie := A / 2
  let remaining_apples := A / 2 - 25
  let apples_given_away := 0.20 * (A / 2 - 25)
  (remaining_apples - apples_given_away = 6) → A = 65

theorem solve_initial_apple_count : initial_apple_count 65 :=
sorry

end solve_initial_apple_count_l282_282818


namespace concert_tickets_price_l282_282496

theorem concert_tickets_price (x : ℕ) (h1 : ∃ a b : ℕ, a * x = 36 ∧ b * x = 90) : 
  {x : ℕ | ∃ a b : ℕ, a * x = 36 ∧ b * x = 90}.finite.card = 6 :=
by
  sorry

end concert_tickets_price_l282_282496


namespace maximum_value_ratio_l282_282597

variables {a b : ℝ} (h₀ : a > b > 0)
def ellipse (x y : ℝ) := (x^2 / a^2) + (y^2 / b^2) = 1

variables {F1 F2 P D M N : ℝ × ℝ} 

-- Define the points F1 and F2 as the foci, with the distance from the origin
def is_focus1 (F1 : ℝ × ℝ) : Prop := F1.fst = -sqrt(a^2 - b^2) ∧ F1.snd = 0
def is_focus2 (F2 : ℝ × ℝ) : Prop := F2.fst = sqrt(a^2 - b^2) ∧ F2.snd = 0

-- P is any point on the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop := ellipse a b P.fst P.snd

-- Assume D is the angle bisector, and M, N are the perpendicular feet as defined
-- (Note: we assume positions/values for simplification here; actual positions should be derived as mentioned in the problem context)
def is_angle_bisector (P D F1 F2 : ℝ × ℝ) : Prop := true  -- Definition assumed for completeness

def perpendicular_feet (D M N : ℝ × ℝ) : Prop := true  -- Definition assumed for completeness

-- We are seeking to prove this particular ratio given the conditions
theorem maximum_value_ratio
  (h₁ : is_focus1 F1) (h₂ : is_focus2 F2) (h₃ : is_on_ellipse a b P)
  (h₄ : is_angle_bisector P D F1 F2)
  (h₅ : perpendicular_feet D M N) :
  ∃ (max_ratio : ℝ), max_ratio = (b^2 * (a^2 - b^2)) / a^4 :=
sorry

end maximum_value_ratio_l282_282597


namespace students_in_class_l282_282001

theorem students_in_class
  (total_points_needed : ℕ)
  (points_per_vegetable : ℕ)
  (weeks : ℕ)
  (days_per_week : ℕ)
  (average_days_per_week_eating : ℕ)
  (total_days : ℕ)
  (points_per_student : ℕ) :
  total_points_needed = 200 →
  points_per_vegetable = 2 →
  weeks = 2 →
  days_per_week = 5 →
  average_days_per_week_eating = 2 →
  total_days = weeks * average_days_per_week_eating →
  points_per_student = total_days * points_per_vegetable →
  (total_points_needed / points_per_student = 25) :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3, h4, h5] at h6
  rw [h2, h6] at h7
  rw [h7]
  sorry

end students_in_class_l282_282001


namespace probability_one_black_one_red_l282_282005

theorem probability_one_black_one_red (R B : Finset ℕ) (hR : R.card = 2) (hB : B.card = 3) :
  (2 : ℚ) / 5 = (6 + 6) / (5 * 4) := by
  sorry

end probability_one_black_one_red_l282_282005


namespace bill_original_profit_percentage_l282_282866

-- Definitions of the conditions used in the problem
def purchase_price := (P : ℝ)
def selling_price := (S : ℝ)
def correction_condition := (approx : ℝ → ℝ → Prop) := 
  λ (x y : ℝ), |x - y| < 1 -- Adjust as per the definition of approximately equal.

-- Custom function to define profit percentage
def profit_percentage (P S: ℝ) : ℝ := ((S - P) / P) * 100

-- Main theorem
theorem bill_original_profit_percentage (P S : ℝ) (h1 : correction_condition S 770) (h2 : 1.17 * P = S + 49) :
  profit_percentage P S = 10 := 
by
  sorry

end bill_original_profit_percentage_l282_282866


namespace part1_part2_l282_282809

-- Part 1: Prove the positive integer solutions to the given system of inequalities
theorem part1 (x : ℤ) : 2 * (x - 1) ≥ -4 ∧ (3 * x - 6) / 2 < x - 1 → x ∈ {1, 2, 3} :=
by
  sorry

-- Part 2: Prove the solution to the given equation
theorem part2 (x : ℝ) : (3 / (x - 2) = 5 / (2 - x) - 1) → x = -6 :=
by
  sorry

end part1_part2_l282_282809


namespace sum_of_first_eight_terms_l282_282195

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l282_282195


namespace units_digit_of_2137_pow_753_l282_282290

theorem units_digit_of_2137_pow_753 : 
  let units_digits := [7, 9, 3, 1]
  let base_unit := 7
  let exponent := 753
  let pattern_length := units_digits.length
  (base_unit ^ exponent) % 10 = units_digits[exponent % pattern_length] :=
by
  have base_digit : Nat := 7
  have exponent' : Nat := 753
  have pattern : List Nat := [7, 9, 3, 1]
  have pattern_length : Nat := pattern.length
  have remainder := exponent' % pattern_length
  show (base_digit ^ exponent') % 10 = pattern.get remainder
  sorry

end units_digit_of_2137_pow_753_l282_282290


namespace jason_seashells_after_giving_l282_282343

-- Define the number of seashells Jason originally found
def original_seashells : ℕ := 49

-- Define the number of seashells Jason gave to Tim
def seashells_given : ℕ := 13

-- Prove that the number of seashells Jason now has is 36
theorem jason_seashells_after_giving : original_seashells - seashells_given = 36 :=
by
  -- This is where the proof would go
  sorry

end jason_seashells_after_giving_l282_282343


namespace tips_fraction_l282_282848

theorem tips_fraction (S T : ℝ) (h : T / (S + T) = 0.6363636363636364) : T / S = 1.75 :=
sorry

end tips_fraction_l282_282848


namespace sqrt_meaningful_for_x_l282_282318

theorem sqrt_meaningful_for_x (x : ℝ) : (∃ r : ℝ, r = real.sqrt (x - 3)) ↔ x ≥ 3 := by
  sorry

end sqrt_meaningful_for_x_l282_282318


namespace difference_largest_smallest_solution_l282_282720

/-- Define the integer part of a real number. -/
def int_part (x : ℝ) : ℤ := intFloor x

/-- Define the cube root function. -/
noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

/-- Main proof statement. -/
theorem difference_largest_smallest_solution (x : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 ≤ 2 * (int_part (cube_root x₁ + 0.5) + int_part (cube_root x₁)) ∧
                 x₂^2 ≤ 2 * (int_part (cube_root x₂ + 0.5) + int_part (cube_root x₂)) ∧
                 x₁² ≠ x₂²) →
  (1 : ℝ) = (1 - 0 : ℝ) :=
sorry

end difference_largest_smallest_solution_l282_282720


namespace floor_diff_l282_282903

theorem floor_diff {x : ℝ} (h : x = 12.7) : 
  (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ) * (⌊x⌋ : ℤ) = 17 :=
by
  have h1 : x = 12.7 := h
  have hx2 : x^2 = 161.29 := by sorry
  have hfloor : ⌊x⌋ = 12 := by sorry
  have hfloor2 : ⌊x^2⌋ = 161 := by sorry
  sorry

end floor_diff_l282_282903


namespace valueOf_b50_l282_282897

def sequence (b : ℕ → ℕ) : Prop :=
  (b 1 = 5) ∧ (∀ n : ℕ, n ≥ 1 → b (n+1) = b n + 3 * n)

theorem valueOf_b50 (b : ℕ → ℕ) (H : sequence b) : b 50 = 3680 :=
by
  sorry

end valueOf_b50_l282_282897


namespace pine_tree_taller_than_birch_l282_282382

def height_birch : ℚ := 49 / 4
def height_pine : ℚ := 74 / 4

def height_difference : ℚ :=
  height_pine - height_birch

theorem pine_tree_taller_than_birch :
  height_difference = 25 / 4 :=
by
  sorry

end pine_tree_taller_than_birch_l282_282382


namespace pull_ups_of_fourth_student_l282_282845

theorem pull_ups_of_fourth_student 
  (avg_pullups : ℕ) 
  (num_students : ℕ) 
  (pullups_first : ℕ) 
  (pullups_second : ℕ) 
  (pullups_third : ℕ) 
  (pullups_fifth : ℕ) 
  (H_avg : avg_pullups = 10) 
  (H_students : num_students = 5) 
  (H_first : pullups_first = 9) 
  (H_second : pullups_second = 12) 
  (H_third : pullups_third = 9) 
  (H_fifth : pullups_fifth = 8) : 
  ∃ (pullups_fourth : ℕ), pullups_fourth = 12 := by
  sorry

end pull_ups_of_fourth_student_l282_282845


namespace compute_length_XY_l282_282128

   -- Let us define a right-angled triangle with the given properties
   structure RightTriangle where
     X Y Z : Type
     angle_X : Real
     angle_Z : Real
     side_XZ : Real
     hypotenuse_is_double_side : ∀ {XY : Real}, XY = side_XZ / 2 → angle_X = 90 ∧ angle_Z = 30

   -- Example instantiation of the specific triangle 
   def triangle : RightTriangle :=
   {
     X := ℝ,
     Y := ℝ,
     Z := ℝ,
     angle_X := 90,
     angle_Z := 30,
     side_XZ := 24,
     hypotenuse_is_double_side := sorry
   }

   theorem compute_length_XY (t : RightTriangle) : ∀ {XY : ℝ}, XY = t.side_XZ / 2 → XY = 12 :=
   by
     intro XY hXY
     rw [hXY]
     have hXZ_eq_24 : t.side_XZ = 24 := sorry
     rw [hXZ_eq_24]
     norm_num
   
   
end compute_length_XY_l282_282128


namespace rope_cut_ratio_l282_282067

theorem rope_cut_ratio (L : ℕ) (a b : ℕ) (hL : L = 40) (ha : a = 2) (hb : b = 3) :
  L / (a + b) * a = 16 :=
by
  sorry

end rope_cut_ratio_l282_282067


namespace KL_eq_LM_triangle_KLM_equilateral_l282_282932

variables {Point : Type}
variables (A B C D O K L M : Point)

-- Define conditions
variables (is_convex : ConvexQuadrilateral A B C D)
variables (inside : WithinQuadrilateral O A B C D)
variables (angle_AOB_eq_120 : angle A O B = 120)
variables (angle_COD_eq_120 : angle C O D = 120)
variables (eq_AO_OB : dist A O = dist B O)
variables (eq_CO_OD : dist C O = dist D O)
variables (midpoint_K : Midpoint K A B)
variables (midpoint_L : Midpoint L B C)
variables (midpoint_M : Midpoint M C D)

-- Prove that KL = LM
theorem KL_eq_LM : dist K L = dist L M :=
by sorry

-- Prove that triangle KLM is equilateral
theorem triangle_KLM_equilateral :
  EquilateralTriangle K L M :=
by sorry

end KL_eq_LM_triangle_KLM_equilateral_l282_282932


namespace max_value_on_interval_l282_282204

noncomputable def f (x : ℝ) := 2 * x ^ 3 - 6 * x ^ 2 + 10

theorem max_value_on_interval :
  (∀ x ∈ Set.Icc (1 : ℝ) 3, f 2 <= f x) → 
  ∃ y ∈ Set.Icc (1 : ℝ) 3, ∀ z ∈ Set.Icc (1 : ℝ) 3, f y >= f z :=
by
  sorry

end max_value_on_interval_l282_282204


namespace factorize_1_factorize_2_factorize_3_l282_282153

theorem factorize_1 (x : ℝ) : x^4 - 9*x^2 = x^2 * (x + 3) * (x - 3) :=
sorry

theorem factorize_2 (x y : ℝ) : 25*x^2*y + 20*x*y^2 + 4*y^3 = y * (5*x + 2*y)^2 :=
sorry

theorem factorize_3 (x y a : ℝ) : x^2 * (a - 1) + y^2 * (1 - a) = (a - 1) * (x + y) * (x - y) :=
sorry

end factorize_1_factorize_2_factorize_3_l282_282153


namespace unattainable_y_l282_282563

theorem unattainable_y (x : ℚ) (y : ℚ) (h : y = (1 - 2 * x) / (3 * x + 4)) (hx : x ≠ -4 / 3) : y ≠ -2 / 3 :=
by {
  sorry
}

end unattainable_y_l282_282563


namespace average_cardinality_of_subsets_l282_282029

open Finset

/-- theorem at about the average cardinality of subsets of a finite set -/
theorem average_cardinality_of_subsets (n : ℕ) : 
  let subsets := powerset (range (n + 1)) in
  (∑ s in subsets, s.card) / subsets.card = n / 2 := 
by {
  -- we still need the replacements for the actual proof steps.
  sorry
}

end average_cardinality_of_subsets_l282_282029


namespace final_water_amount_l282_282487

-- Define the given conditions
def percentAcid (x : ℝ) := 0.1 * x
def percentWater (x : ℝ) := 0.9 * x
def pureAcid := 5
def finalWaterPercentage := 0.4

-- Define the final mixture conditions
def finalMixture (x : ℝ) := pureAcid + x
def amountOfWaterInFinalMixture (x : ℝ) := percentWater x

-- The equation representing the final mixture percentage
def mixtureEquation (x : ℝ) := finalWaterPercentage * finalMixture x = percentWater x

-- The proof statement that the amount of water in the final mixture is 3.6 liters
theorem final_water_amount : ∃ x : ℝ, mixtureEquation x ∧ amountOfWaterInFinalMixture x = 3.6 :=
by
  sorry

end final_water_amount_l282_282487


namespace four_consecutive_even_impossible_l282_282464

def is_four_consecutive_even_sum (S : ℕ) : Prop :=
  ∃ n : ℤ, S = 4 * n + 12

theorem four_consecutive_even_impossible :
  ¬ is_four_consecutive_even_sum 34 :=
by
  sorry

end four_consecutive_even_impossible_l282_282464


namespace explicit_form_l282_282603

-- Define the functional equation
def f (x : ℝ) : ℝ := sorry

-- Define the condition that f(x) satisfies
axiom functional_equation (x : ℝ) (h : x ≠ 0) : f x = 2 * f (1 / x) + 3 * x

-- State the theorem that we need to prove
theorem explicit_form (x : ℝ) (h : x ≠ 0) : f x = -x - (2 / x) :=
by
  sorry

end explicit_form_l282_282603


namespace ellipse_properties_l282_282211

-- Definition for the given conditions
def ellipse (x y a b : ℝ) : Prop := (x^2 / a^2) + (y^2 / b^2) = 1

-- Define the conditions given
def eccentricity (a c : ℝ) : Prop := c / a = sqrt (3 / 2)
def point_on_ellipse (a b : ℝ) : Prop := 
  ∃ x y : ℝ, x = 1 ∧ y = sqrt (3 / 2) ∧ ellipse x y a b

-- The main proof task
theorem ellipse_properties (a b c : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : eccentricity a c) (h4 : point_on_ellipse a b) :
  (a = 2 ∧ b = 1 ∧ ellipse x y a b = (x^2 / 4) + y^2 = 1) ∧
  (∀ (P Q : ℝ × ℝ) (k t : ℝ), intersect_line_ellipse P Q a b k t → 
  slopes_geometric_sequence P Q k → k = -1 / 2) :=
by sorry

end ellipse_properties_l282_282211


namespace general_term_a_T_value_problem_conditions_l282_282595

noncomputable def S (n : ℕ) (a : ℕ → ℝ) : ℝ := ∑ i in finset.range n, a i

noncomputable def a : ℕ → ℝ
| 0     := 0
| 1     := 2 / 3
| (n+1) := 1 / 3 * a n

noncomputable def b (n : ℕ) (S : ℕ → ℝ) : ℕ → ℝ
| n     := real.log (1 - S (n + 1)) / real.log (1 / 3)

noncomputable def T (n : ℕ) (b : ℕ → ℝ) : ℝ :=
∑ i in finset.range n, 1 / (b i * b (i + 1))

theorem general_term_a (n : ℕ) : a n = 2 * (1 / 3) ^ n :=
begin
  sorry
end

theorem T_value (n : ℕ) (S : ℕ → ℝ) (b : ℕ → ℝ) : T n b = n / (2 * (n + 2)) :=
begin
  sorry
end

theorem problem_conditions (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ (n : ℕ), S n + (1 / 2) * a n = 1) →
  (∀ (n : ℕ), b n (S) = real.log (1 - S (n + 1)) / real.log (1 / 3)) →
  (∀ (n : ℕ), T n (b n S) = ∑ i in finset.range n, 1 / (b n S i * b n S (i + 1))) →
  a 1 = 2 / 3 ∧ (∀ (n : ℕ), a (n + 1) = 1 / 3 * a n) ∧ S 1 = 1 - 1 / 2 * a 1 ∧ 
  (∀ (n > 1), S n = 1 - 1 / 2 * a n) :=
begin
  sorry
end

end general_term_a_T_value_problem_conditions_l282_282595


namespace magnitude_a_sub_2b_l282_282617

variables (a b : EuclideanVector)
variables (a_norm : ‖a‖ = 1) (b_norm : ‖b‖ = 2)
variables (perp_ab : (a + b) ⬝ a = 0)

theorem magnitude_a_sub_2b :
  ‖a - 2 • b‖ = Real.sqrt 21 := by
  sorry

end magnitude_a_sub_2b_l282_282617


namespace inequality_proof_l282_282206

noncomputable theory

-- Define the variables and conditions
variables {a b c : ℝ}
variables (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0)

-- Define the main proposition which is to prove the inequality
theorem inequality_proof (h_nonneg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0) (h_not_all_zero : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  (a * (b + c) - b * c) / (b^2 + c^2) + (b * (a + c) - a * c) / (a^2 + c^2) + (c * (a + b) - a * b) / (a^2 + b^2) ≥ 3 / 2 :=
by 
  sorry

end inequality_proof_l282_282206


namespace square_inscribed_ab_neg1_l282_282091

theorem square_inscribed_ab_neg1 :
  (∃ (a b : ℝ), 
    (∃ (s1 s2 : ℝ), s1 = 9 ∧ s2 = 16 ∧ 
    (a + b = 4) ∧ 
    (a^2 + b^2 = 18) ∧ 
    ((∃ (t t' : ℝ), t = s1 ∧ t' = s2 ∧ (2 * ab + a^2 + b^2 = s1 + s2)) ∧ 
    (a * b = -1)))) 
  sorry

end square_inscribed_ab_neg1_l282_282091


namespace section_Diligence_students_before_transfer_l282_282997

-- Define the variables
variables (D_after I_after D_before : ℕ)

-- Problem Statement
theorem section_Diligence_students_before_transfer :
  ∀ (D_after I_after: ℕ),
    2 + D_after = I_after
    ∧ D_after + I_after = 50 →
    ∃ D_before, D_before = D_after - 2 ∧ D_before = 23 :=
by
sorrry

end section_Diligence_students_before_transfer_l282_282997


namespace position_of_VBGTP_l282_282445

noncomputable def factorial (n : Nat) : Nat :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem position_of_VBGTP :
  let letters := ['B', 'G', 'P', 'T', 'V']
  let words := List.permutations letters
  let sorted_words := List.sort words
  List.index_of ('V', 'B', 'G', 'T', 'P') sorted_words + 1 = 104 :=
by
  sorry

end position_of_VBGTP_l282_282445


namespace circle_equation_center_line_l282_282176

theorem circle_equation_center_line (x y : ℝ) :
  -- Conditions
  (∀ (x1 y1 : ℝ), x1 + y1 - 2 = 0 → (x = 1 ∧ y = 1)) ∧
  ((x - 1)^2 + (y - 1)^2 = 4) ∧
  -- Points A and B
  (∀ (xA yA : ℝ), xA = 1 ∧ yA = -1 ∨ xA = -1 ∧ yA = 1 →
    ((xA - x)^2 + (yA - y)^2 = 4)) :=
by
  sorry

end circle_equation_center_line_l282_282176


namespace train_passes_man_in_10_seconds_l282_282048

-- Definitions based on conditions
def train_length : ℕ := 150
def train_speed_km_h : ℕ := 62
def man_speed_km_h : ℕ := 8

-- The theorem statement we need to prove
theorem train_passes_man_in_10_seconds :
  let relative_speed_m_s := (train_speed_km_h - man_speed_km_h) * 1000 / 3600 in
  let time_in_seconds := train_length / relative_speed_m_s in
  time_in_seconds = 10 :=
by
  sorry

end train_passes_man_in_10_seconds_l282_282048


namespace distance_AF_minimum_distance_MN_l282_282239

-- Define the conditions
variables {p : ℝ} (h_pos : p > 0)
def parabola (x y : ℝ) := x^2 = 2 * p * y
def circle (x y : ℝ) := x^2 + y^2 = 1
def focus := (0, 1) -- Given that focus F is (0, 1)

-- Problem 1: Prove the distance |AF| is √5 - 1.
theorem distance_AF (A : ℝ × ℝ) (hA_on_parabola : parabola p A.1 A.2) (hA_on_circle : circle A.1 A.2) :
  |real.sqrt ((A.1 - 0)^2 + (A.2 - 1)^2)| = real.sqrt 5 - 1 := sorry

-- Problem 2: Prove the minimum value of |MN| and corresponding p.
theorem minimum_distance_MN (M N : ℝ × ℝ)
  (hM_on_parabola : parabola p M.1 M.2) (hN_on_circle : circle N.1 N.2)
  (hl_tangent_to_parabola : ∀ x y, parabola p x y → ∀ t, (x, y) ≠ (M.1, M.2) → x * t - p * y - p * M.2 = 0)
  (hl_tangent_to_circle : ∀ x y, circle x y → ∀ t, (x, y) ≠ (N.1, N.2) → x * t - y - 1 = 0) :
  ∃ (p : ℝ) (h_pos : p > 0), let d := real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) in
  d = 2 * real.sqrt 2 ∧ p = real.sqrt 3 := sorry

end distance_AF_minimum_distance_MN_l282_282239


namespace remainder_14_div_5_l282_282457

theorem remainder_14_div_5 : 14 % 5 = 4 := by
  sorry

end remainder_14_div_5_l282_282457


namespace sqrt_condition_l282_282305

theorem sqrt_condition (x : ℝ) (h : 2 * x - 1 ≥ 0) : x ≥ 1 / 2 :=
sorry

end sqrt_condition_l282_282305


namespace sum_geometric_sequence_first_eight_terms_l282_282184

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l282_282184


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282260

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282260


namespace triangle_area_l282_282405

theorem triangle_area (A B C : ℝ × ℝ) (hA : A = (-1, 1)) (hB : B = (1, 2)) (hCy : C.2 = 0)
  (hCentroid : let G := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3) in G = (1, 1)) :
  let area := (1 / 2 : ℝ) * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2)) in
  area = 3 := by
  sorry

end triangle_area_l282_282405


namespace modular_expression_l282_282540

theorem modular_expression :
  (let a := (3 * 103 + 9 * 37 + 4 * 71) % 120 in a = 86) :=
by {
  let a := (3 * 103 + 9 * 37 + 4 * 71) % 120,
  have : a = 86,
  sorry,
}

end modular_expression_l282_282540


namespace percentage_markup_on_cost_price_l282_282413

theorem percentage_markup_on_cost_price 
  (SP : ℝ) (CP : ℝ) (hSP : SP = 6400) (hCP : CP = 5565.217391304348) : 
  ((SP - CP) / CP) * 100 = 15 :=
by
  -- proof would go here
  sorry

end percentage_markup_on_cost_price_l282_282413


namespace max_number_of_smaller_boxes_l282_282468

-- Defining the dimensions and volumes
def large_box_length : Real := 8
def large_box_width : Real := 7
def large_box_height : Real := 6
def small_box_length : Real := 0.04
def small_box_width : Real := 0.07
def small_box_height : Real := 0.06

def volume_large_box : Real := large_box_length * large_box_width * large_box_height
def volume_small_box : Real := small_box_length * small_box_width * small_box_height
def max_smaller_boxes (V_large V_small : Real) : Real := V_large / V_small

-- Statement of the theorem
theorem max_number_of_smaller_boxes :
  max_smaller_boxes volume_large_box volume_small_box = 2000000 := by
  sorry

end max_number_of_smaller_boxes_l282_282468


namespace least_possible_value_of_f_l282_282911

theorem least_possible_value_of_f :
  ∃ x : ℝ, (1 + Real.cos (2 * x) ≠ 0) ∧ (1 - Real.cos (2 * x) ≠ 0) ∧ 
  (∀ y : ℝ, (1 + Real.cos (2 * y) ≠ 0) ∧ (1 - Real.cos (2 * y) ≠ 0) → 
    f y ≥ 32) ∧ 
  f x = 32
where
  f (x : ℝ) : ℝ := 
    (9 / (1 + Real.cos (2 * x))) + (25 / (1 - Real.cos (2 * x))) := 
sorry

end least_possible_value_of_f_l282_282911


namespace parabolas_distance_l282_282518

theorem parabolas_distance :
  let E : ℝ × ℝ → ℝ := λ p, Real.sqrt (p.1^2 + p.2^2) + |p.2 - 1|
  (∀ p : ℝ × ℝ, E p = 5 → (∃ v1 v2 : ℝ × ℝ, v1 = (0, 3) ∧ v2 = (0, -2) ∧ Real.dist v1 v2 = 5)) :=
by
  sorry

end parabolas_distance_l282_282518


namespace division_of_fractions_l282_282622

theorem division_of_fractions : (1 / 10) / (1 / 5) = 1 / 2 :=
by
  sorry

end division_of_fractions_l282_282622


namespace eggs_donated_to_charity_is_13_dozen_l282_282653

variable (SourceA_eggs_tuesday : ℕ)
variable (SourceA_eggs_thursday : ℕ)
variable (SourceB_eggs_monday : ℕ)
variable (SourceB_eggs_wednesday : ℕ)
variable (eggs_to_market : ℕ)
variable (eggs_to_mall : ℕ)
variable (eggs_to_bakery : ℕ)
variable (eggs_for_pie : ℕ)
variable (eggs_to_neighbor : ℕ)

-- Define the number of eggs collected from each source
def total_eggs_collected_from_sourceA := SourceA_eggs_tuesday + SourceA_eggs_thursday
def total_eggs_collected_from_sourceB := SourceB_eggs_monday + SourceB_eggs_wednesday

-- Define the total number of eggs collected in a week
def total_eggs_collected := total_eggs_collected_from_sourceA SourceA_eggs_tuesday SourceA_eggs_thursday
                            + total_eggs_collected_from_sourceB SourceB_eggs_monday SourceB_eggs_wednesday

-- Define the total number of eggs distributed in a week
def total_eggs_distributed := eggs_to_market + eggs_to_mall + eggs_to_bakery + eggs_for_pie + eggs_to_neighbor

-- Define the amount of eggs donated to charity
def eggs_donated := total_eggs_collected - total_eggs_distributed

-- Define the theorem
theorem eggs_donated_to_charity_is_13_dozen
  (hSourceA_eggs_tuesday : SourceA_eggs_tuesday = 8)
  (hSourceA_eggs_thursday : SourceA_eggs_thursday = 8)
  (hSourceB_eggs_monday : SourceB_eggs_monday = 6)
  (hSourceB_eggs_wednesday : SourceB_eggs_wednesday = 6)
  (heggs_to_market : eggs_to_market = 3)
  (heggs_to_mall : eggs_to_mall = 5)
  (heggs_to_bakery : eggs_to_bakery = 2)
  (heggs_for_pie : eggs_for_pie = 4)
  (heggs_to_neighbor : eggs_to_neighbor = 1)
  : eggs_donated SourceA_eggs_tuesday SourceA_eggs_thursday
                 SourceB_eggs_monday SourceB_eggs_wednesday
                 eggs_to_market eggs_to_mall eggs_to_bakery
                 eggs_for_pie eggs_to_neighbor = 13 := 
by
  sorry

end eggs_donated_to_charity_is_13_dozen_l282_282653


namespace range_of_s_l282_282793

def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_of_s :
  set.range s = set.univ \ {0} :=
sorry

end range_of_s_l282_282793


namespace exists_two_users_with_1992_and_1993_calls_l282_282480

-- Definitions corresponding to the conditions
def at_least_1993_calls (user A : Type) : Prop :=
  ∃ (users : Set Type), (users.size ≥ 1993) ∧ ∀ u ∈ users, 
  ∀ (v ∈ users), (u ≠ v → called u v) 

def distinct_calls (u v : Type) (called : Type → Type → Prop) : Prop :=
  ∀ x, (calls u = x ∧ calls v = x) → 
  (∀ w, (called u w ∨ called v w) → ¬called u w ∨ ¬called v w) 

-- Given type variables for users and calls
variables {A : Type} {users : Set Type} {calls : Type → Nat} {called : Type → Type → Prop}

-- Final Lean theorem statement
theorem exists_two_users_with_1992_and_1993_calls :
  at_least_1993_calls A → 
  (∀ u v, distinct_calls u v called) → 
  (∀ p q, ¬ called p q → (users ∩ {p, q}).size ≤ 1) →
  ∃ (B₁ B₂ : A), (calls B₁ = 1992) ∧ (calls B₂ = 1993) := 
sorry

end exists_two_users_with_1992_and_1993_calls_l282_282480


namespace number_of_green_balls_l282_282477

theorem number_of_green_balls (r g : ℕ): r = 8 → (r + g) ≠ 0 → (r : ℝ) / (r + g) = 1 / 3 → g = 16 :=
by
  intro hr hrg hprob
  rw [hr] at *
  have h : 8 / (8 + g) = 1 / 3, from hprob
  sorry

end number_of_green_balls_l282_282477


namespace candy_per_packet_l282_282873

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l282_282873


namespace max_value_frac_sum_l282_282372

theorem max_value_frac_sum :
  ∃ (a b : ℕ), a ∈ {2, 3, 4, 5, 6, 7, 8} ∧ b ∈ {2, 3, 4, 5, 6, 7, 8} ∧
  (∀ a b ∈ {2, 3, 4, 5, 6, 7, 8}, (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b) ≤ 89 / 287) ∧
  (a : ℚ) / (10 * b + a) + (b : ℚ) / (10 * a + b) = 89 / 287 :=
by
  sorry

end max_value_frac_sum_l282_282372


namespace initial_average_is_correct_l282_282927

def initial_average_daily_production (n : ℕ) (today_production : ℕ) (new_average : ℕ) (initial_average : ℕ) :=
  let total_initial_production := initial_average * n
  let total_new_production := total_initial_production + today_production
  let total_days := n + 1
  total_new_production = new_average * total_days

theorem initial_average_is_correct :
  ∀ (A n today_production new_average : ℕ),
    n = 19 →
    today_production = 90 →
    new_average = 52 →
    initial_average_daily_production n today_production new_average A →
    A = 50 := by
    intros A n today_production new_average hn htoday hnew havg
    sorry

end initial_average_is_correct_l282_282927


namespace find_p_q_l282_282681

noncomputable def roots_of_polynomial (a b c : ℝ) :=
  a^3 - 2018 * a + 2018 = 0 ∧ b^3 - 2018 * b + 2018 = 0 ∧ c^3 - 2018 * c + 2018 = 0

theorem find_p_q (a b c : ℝ) (p q : ℕ) 
  (h1 : roots_of_polynomial a b c)
  (h2 : 0 < p ∧ p ≤ q) 
  (h3 : (a^(p+q) + b^(p+q) + c^(p+q))/(p+q) = (a^p + b^p + c^p)/p * (a^q + b^q + c^q)/q) : 
  p^2 + q^2 = 20 := 
sorry

end find_p_q_l282_282681


namespace choose_socks_l282_282534

open Nat

theorem choose_socks :
  (Nat.choose 8 4) = 70 :=
by 
  sorry

end choose_socks_l282_282534


namespace base_8_to_decimal_77_eq_63_l282_282448

-- Define the problem in Lean 4
theorem base_8_to_decimal_77_eq_63 (k a1 a2 : ℕ) (h_k : k = 8) (h_a1 : a1 = 7) (h_a2 : a2 = 7) :
    a2 * k^1 + a1 * k^0 = 63 := 
by
  -- Placeholder for proof
  sorry

end base_8_to_decimal_77_eq_63_l282_282448


namespace eq_motion_x_M_proof_eq_motion_y_M_proof_trajectory_M_proof_speed_M_proof_l282_282519

variables {R : Type*} [Real R]

-- Define the problem constants
def OA : R := 90
def AB : R := 90
def AM : R := 45
def omega : R := 10

-- Define the motion parameters
variables (theta t : R)

-- Define the coordinates of A
def x_A := OA * cos theta
def y_A := OA * sin theta

-- Define the coordinates of B
def x_B := x_A
def y_B : R := 0

-- Define the coordinates of M
def x_M := (x_A + x_B) / 2
def y_M := (y_A + y_B) / 2

-- Define the parametric equations of motion for M
def eq_motion_x_M := x_M = OA * cos theta
def eq_motion_y_M := y_M = AM * sin theta

-- Define trajectory equation for M
def trajectory_M := (x_M / OA)^2 + (y_M / AM)^2 = 1

-- Define velocity components
def v_x := -omega * OA * sin (omega * t)
def v_y := omega * AM * cos (omega * t)

-- Define the magnitude of velocity
def velocity_M := sqrt (v_x^2 + v_y^2)

-- Define the equation for speed of M
def speed_M := velocity_M = (OA * omega) / sqrt 2

-- The proof statements
theorem eq_motion_x_M_proof : eq_motion_x_M := sorry
theorem eq_motion_y_M_proof : eq_motion_y_M := sorry
theorem trajectory_M_proof : trajectory_M := sorry
theorem speed_M_proof : speed_M := sorry

end eq_motion_x_M_proof_eq_motion_y_M_proof_trajectory_M_proof_speed_M_proof_l282_282519


namespace probability_sum_is_multiple_of_4_l282_282443

open ProbabilityTheory

noncomputable def SpinnerA := [1, 2, 3]
noncomputable def SpinnerB := [2, 3, 3, 4]

def sumIsMultipleOf4 (a b : ℕ) := (a + b) % 4 = 0

def eventProbability (A B : List ℕ) : ℚ :=
  let outcomes := for a in A, b in B, a + b 
  let favorable := List.countp (λ s, s % 4 = 0) outcomes
  favorable / (A.length * B.length)

theorem probability_sum_is_multiple_of_4 :
  eventProbability SpinnerA SpinnerB = 1 / 4 :=
by
  sorry

end probability_sum_is_multiple_of_4_l282_282443


namespace find_x_values_l282_282163

theorem find_x_values (x : ℝ) (h : x > 9) : 
  (sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) ↔ (x ≥ 18) :=
sorry

end find_x_values_l282_282163


namespace probability_three_even_dice_l282_282711

theorem probability_three_even_dice :
  let p_even := 1 / 2
  let combo := Nat.choose 5 3
  let probability := combo * (p_even ^ 3) * ((1 - p_even) ^ 2)
  probability = 5 / 16 := 
by
  sorry

end probability_three_even_dice_l282_282711


namespace Iggy_runs_8_miles_on_Thursday_l282_282322

theorem Iggy_runs_8_miles_on_Thursday :
  ∀ (x : ℕ),
    (running_time : ℕ → ℕ → ℕ) -- A function for total running time calculation
    (monday_miles = 3) 
    (tuesday_miles = 4) 
    (wednesday_miles = 6) 
    (friday_miles = 3) 
    (pace = 10) 
    (total_minutes = 240) 
    (running_time monday_miles pace +
    running_time tuesday_miles pace +
    running_time wednesday_miles pace +
    running_time friday_miles pace +
    running_time x pace = total_minutes) 
    : x = 8 :=
by
  sorry

-- Definition of running_time function
def running_time (miles pace: ℕ) : ℕ := miles * pace

end Iggy_runs_8_miles_on_Thursday_l282_282322


namespace evaluate_expression_l282_282904

theorem evaluate_expression (c d : ℝ) (h_c : c = 3) (h_d : d = 2) : 
  (c^2 + d + 1)^2 - (c^2 - d - 1)^2 = 80 := by 
  sorry

end evaluate_expression_l282_282904


namespace store_a_cheaper_by_two_l282_282779

def store_a_price : ℝ := 125
def store_b_price : ℝ := 130
def store_a_discount : ℝ := 0.08
def store_b_discount : ℝ := 0.10

def final_price_store_a : ℝ := store_a_price - (store_a_price * store_a_discount)
def final_price_store_b : ℝ := store_b_price - (store_b_price * store_b_discount)

theorem store_a_cheaper_by_two :
  final_price_store_b - final_price_store_a = 2 :=
by
  unfold final_price_store_b final_price_store_a store_a_price store_b_price store_a_discount store_b_discount
  have h₁ : store_b_price - store_b_price * store_b_discount = 117 := by norm_num
  have h₂ : store_a_price - store_a_price * store_a_discount = 115 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end store_a_cheaper_by_two_l282_282779


namespace remaining_diagonals_intersections_l282_282829

theorem remaining_diagonals_intersections (n : ℕ) (k m : ℕ) (h_nk : n = 20) (h_k : k = 14) (h_m : m = 8) : 
  let remaining_vertices_on_k_side := k - 2,
      remaining_vertices_on_m_side := m - 2
  in remaining_vertices_on_k_side * remaining_vertices_on_m_side = 72 :=
  sorry

end remaining_diagonals_intersections_l282_282829


namespace volume_of_H2_gas_produced_l282_282143

noncomputable def ideal_gas_law (P V n R T : ℝ) : Prop :=
  P * V = n * R * T

theorem volume_of_H2_gas_produced :
  ∀ (T P : ℝ) (moles_H2SO4 moles_Zn : ℝ), 
  T = 300 → P = 1 → moles_H2SO4 = 3 → moles_Zn = 3 → 
  ideal_gas_law P 0 3 0.0821 T → 
  (∀ V, ideal_gas_law 1 V 3 0.0821 300) → V = 73.89 :=
begin
  intros T P moles_H2SO4 moles_Zn hT hP hHSO4 hZn hIGL,
  sorry
end

end volume_of_H2_gas_produced_l282_282143


namespace num_distinct_five_digit_integers_with_product_of_digits_18_l282_282899

theorem num_distinct_five_digit_integers_with_product_of_digits_18 :
  ∃ (n : ℕ), n = 70 ∧ ∀ (a b c d e : ℕ),
    a * b * c * d * e = 18 ∧ 
    1 ≤ a ∧ a ≤ 9 ∧ 1 ≤ b ∧ b ≤ 9 ∧ 1 ≤ c ∧ c ≤ 9 ∧ 1 ≤ d ∧ d ≤ 9 ∧ 1 ≤ e ∧ e ≤ 9 → 
    (∃ (s : Finset (Fin 100000)), s.card = n) :=
  sorry

end num_distinct_five_digit_integers_with_product_of_digits_18_l282_282899


namespace minimum_large_flasks_correct_l282_282080

noncomputable def min_large_flasks (total_flasks : ℕ) (threshold: ℝ) : ℕ :=
  let P_A (N : ℕ) : ℝ :=
    (real.to_nnreal ((N - 50) * (N - 50) + 2450)) / 4950
  in
  (Nat.find (λ N, (N ≥ 2 ∧ N ≤ total_flasks - 2 ∧ P_A N < threshold)) 46)

theorem minimum_large_flasks_correct (N : ℕ) :
  min_large_flasks 100 (1 / 2) = 46 := sorry

end minimum_large_flasks_correct_l282_282080


namespace number_of_candies_in_a_packet_l282_282870

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l282_282870


namespace tangent_line_equation_l282_282549

noncomputable def tangent_eqn (f : ℝ → ℝ) (a b : ℝ) (tangent_line : ℝ → ℝ → Prop) : Prop :=
  (∀ x, x ≠ a → tangent_line (f a + (f a)' (x - a)) (x)) ∧ 
  f x = x ^ 3 - 2 * x + 4 ∧ 
  f 1 = 3 ∧
  tangent_line 1 (1 - 3 + 2 = 0)

theorem tangent_line_equation : tangent_eqn (x^3 - 2*x + 4) 1 3 (λ x y, x - y + 2 = 0) :=
sorry

end tangent_line_equation_l282_282549


namespace isosceles_BDE_l282_282335

-- Definitions based on conditions
def right_triangle (A B C : Type) (angle_A : ℝ) : Prop := angle_A = 90

def angle_bisector (A B C D : Type) (angle_BAC : ℝ) : Prop := 
  ∃ (D : Type), angle_BAC / 2 = D

def perpendicular (B C E : Type) : Prop := 
  ∃ (E : Type), ∠ (B E) = 90

-- Final statement to prove
theorem isosceles_BDE {A B C D E : Type} (h1 : right_triangle A B C 90)
  (h2 : angle_bisector A B C D) 
  (h3 : perpendicular B C E) : 
  BE = BD :=
by
  sorry

end isosceles_BDE_l282_282335


namespace moving_disk_area_l282_282210

noncomputable def grazed_area (r : ℝ) : ℝ :=
  1 - (20 - Real.pi) * r^2

theorem moving_disk_area (r : ℝ) (h : r = 1 / 6) : grazed_area r = 1 - (20 - Real.pi) * (r * r) := by
  simp [grazed_area]
  rw h
  sorry

end moving_disk_area_l282_282210


namespace problem_solution_l282_282297

noncomputable def find_a3_and_sum (a0 a1 a2 a3 a4 a5 : ℝ) : Prop :=
  (∀ x : ℝ, x^5 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4 + a5 * (x + 2)^5) →
  (a3 = 40 ∧ a0 + a1 + a2 + a4 + a5 = -41)

theorem problem_solution {a0 a1 a2 a3 a4 a5 : ℝ} :
  find_a3_and_sum a0 a1 a2 a3 a4 a5 :=
by
  intros h
  sorry

end problem_solution_l282_282297


namespace smallest_X_divisible_18_l282_282686

theorem smallest_X_divisible_18 (T : ℕ) (hT : T % 18 = 0) (h_digits : ∀ d ∈ (T.digits 10), d = 0 ∨ d = 1) : 
  ∃ X : ℕ, X = 6172833 ∧ T = X * 18 :=
begin
  use 6172833,
  split,
  { refl },
  { sorry }
end

end smallest_X_divisible_18_l282_282686


namespace impossible_return_l282_282107

def Point := (ℝ × ℝ)

-- Conditions
def is_valid_point (p: Point) : Prop :=
  let (a, b) := p
  ∃ a_int b_int : ℤ, (a = a_int + b_int * Real.sqrt 2 ∧ b = a_int + b_int * Real.sqrt 2)

def valid_movement (p q: Point) : Prop :=
  let (x1, y1) := p
  let (x2, y2) := q
  abs x2 > abs x1 ∧ abs y2 > abs y1 

-- Theorem statement
theorem impossible_return (start: Point) (h: start = (1, Real.sqrt 2)) 
  (valid_start: is_valid_point start) :
  ∀ (p: Point), (is_valid_point p ∧ valid_movement start p) → p ≠ start :=
sorry

end impossible_return_l282_282107


namespace roundness_of_24300000_l282_282882

def roundness (n : ℕ) : ℕ :=
  let factors := n.factors.to_multiset in
  factors.count 2 + factors.count 3 + factors.count 5

theorem roundness_of_24300000 : roundness 24300000 = 15 :=
by
  sorry

end roundness_of_24300000_l282_282882


namespace no_real_solution_to_abs_eq_l282_282174

theorem no_real_solution_to_abs_eq (x : ℝ): ¬ (|x^2 - 3| = 2x + 6) :=
sorry

end no_real_solution_to_abs_eq_l282_282174


namespace increasing_function_range_l282_282300

def f (x a : ℝ) : ℝ := x^2 + a * x + 1 / x

theorem increasing_function_range (a : ℝ) :
  (∀ x y : ℝ, 1 / 2 < x ∧ x < y → f x a < f y a) ↔ 3 ≤ a :=
sorry

end increasing_function_range_l282_282300


namespace solve_equation_l282_282403

-- Define the given equation as Lean definitions
def equation_lhs_numerator (x : ℝ) :=
  x * (x + 3) - 4 * (3 - (5 * x - (1 - 2 * x)) * (x - 1)^2 / 4)

def equation_lhs_denominator (x : ℝ) :=
  -4 * x / 3 - (-1 - ((5 / 2) * (x + 6 / 5) - x) + x / 6)

def equation_rhs (x : ℝ) :=
  x / 2 * (x + 6) - (-x * (-3 * (x - 4) + 2 * (3 * x - 5) - 10))

def original_equation (x : ℝ) :=
  equation_lhs_numerator x / equation_lhs_denominator x =
  equation_rhs x

-- Proof problem to show the solutions
theorem solve_equation :
  (original_equation (15 / 4)) ∨ (original_equation (1 / 3)) :=
by
  sorry

end solve_equation_l282_282403


namespace area_of_pentagon_l282_282836

-- Given conditions
variables (a b c d e r s : ℕ)

-- Side lengths of the pentagon
axiom sides : {a, b, c, d, e} = {12, 15, 18, 30, 34}

-- Pythagorean theorem condition for the triangular corner
axiom pythagorean_theorem : r^2 + s^2 = e^2

-- Relationships between sides of the pentagon and the triangular corner legs
axiom rel1 : r = b - d
axiom rel2 : s = c - a

-- The main goal: the area of the pentagon
theorem area_of_pentagon : (b * c - 1/2 * r * s) = 804 :=
by
  sorry

end area_of_pentagon_l282_282836


namespace product_of_solutions_l282_282554

theorem product_of_solutions (x : ℝ) (h : 2^(3*x + 1) - 17 * 2^(2*x) + 2^(x + 3) = 0) : 
  ∃ a b : ℝ, (2^(3*a + 1) - 17 * 2^(2*a) + 2^(a + 3) = 0) ∧ (2^(3*b + 1) - 17 * 2^(2*b) + 2^(b + 3) = 0) ∧ a * b = -3 := 
sorry

end product_of_solutions_l282_282554


namespace limit_of_sequence_l282_282881

noncomputable def u (n : ℕ) : ℝ := (2 * n^2 + 2) / (2 * n^2 + 1)

theorem limit_of_sequence :
  filter.tendsto (λ n : ℕ, u n ^ n^2) filter.at_top (𝓝 (Real.sqrt Real.exp)) :=
sorry

end limit_of_sequence_l282_282881


namespace sum_of_angles_from_center_l282_282088

-- Define a regular pentagon inscribed in a circle
def regular_pentagon_inscribed (O : Point) (r : ℝ) (A B C D E : Point) : Prop :=
  is_on_circle A O r ∧ 
  is_on_circle B O r ∧ 
  is_on_circle C O r ∧
  is_on_circle D O r ∧
  is_on_circle E O r ∧ 
  are_equally_spaced A B C D E O

-- The main theorem to prove
theorem sum_of_angles_from_center (O : Point) (r : ℝ) (A B C D E : Point) :
  regular_pentagon_inscribed O r A B C D E →
  sum_of_angles O [A, B, C, D, E] = 630 :=
  sorry
 
end sum_of_angles_from_center_l282_282088


namespace g_prime_positive_l282_282235

noncomputable def f (a x : ℝ) := a * x - a * x ^ 2 - Real.log x

noncomputable def g (a x : ℝ) := -2 * (a * x - a * x ^ 2 - Real.log x) - (2 * a + 1) * x ^ 2 + a * x

def g_zero (a x1 x2 : ℝ) := g a x1 = 0 ∧ g a x2 = 0

def x1_x2_condition (x1 x2 : ℝ) := x1 < x2 ∧ x2 < 4 * x1

theorem g_prime_positive (a x1 x2 : ℝ) (h1 : g_zero a x1 x2) (h2 : x1_x2_condition x1 x2) :
  (deriv (g a) ((2 * x1 + x2) / 3)) > 0 := by
  sorry

end g_prime_positive_l282_282235


namespace maria_workday_end_l282_282532

def time_in_minutes (h : ℕ) (m : ℕ) : ℕ := h * 60 + m

def start_time : ℕ := time_in_minutes 7 25
def lunch_break : ℕ := 45
def noon : ℕ := time_in_minutes 12 0
def work_hours : ℕ := 8 * 60
def end_time : ℕ := time_in_minutes 16 10

theorem maria_workday_end : start_time + (noon - start_time) + lunch_break + (work_hours - (noon - start_time)) = end_time := by
  sorry

end maria_workday_end_l282_282532


namespace parallel_or_perpendicular_lines_l282_282131

-- Define the lines
def line1 : ℝ → ℝ := λ x, 2 * x + 3
def line2 : ℝ → ℝ := λ x, 3 * x + 2 -- Simplified from 2y = 6x + 4
def line3 : ℝ → ℝ := λ x, 2 * x - (1 / 3) -- Simplified from 3y = 6x - 1
def line4 : ℝ → ℝ := λ x, (1 / 2) * x - 2 -- Simplified from 4y = 2x - 8
def line5 : ℝ → ℝ := λ x, (2 / 5) * x - 2 -- Simplified from 5y = 2x - 10

-- Prove that there is exactly 1 pair of lines that are either parallel or perpendicular
theorem parallel_or_perpendicular_lines : 
  (∃! (l1 l2 : ℝ → ℝ), (l1 = line1 ∨ l1 = line2 ∨ l1 = line3 ∨ l1 = line4 ∨ l1 = line5) ∧ 
                          (l2 = line1 ∨ l2 = line2 ∨ l2 = line3 ∨ l2 = line4 ∨ l2 = line5) ∧ 
                          l1 ≠ l2 ∧ 
                          ((∃ m1 m2, l1 = λ x, m1 * x + 3 ∧ l2 = λ x, m2 * x + 3 ∧ m1 = m2) ∨ 
                           (∃ m1 m2, l1 = λ x, m1 * x + 3 ∧ l2 = λ x, m2 * x + 3 ∧ m1 * m2 = -1))) := sorry

end parallel_or_perpendicular_lines_l282_282131


namespace probability_at_least_one_boy_one_girl_l282_282420

def boys := 12
def girls := 18
def total_members := 30
def committee_size := 6

def total_ways := Nat.choose total_members committee_size
def all_boys_ways := Nat.choose boys committee_size
def all_girls_ways := Nat.choose girls committee_size
def all_boys_or_girls_ways := all_boys_ways + all_girls_ways
def complementary_probability := all_boys_or_girls_ways / total_ways
def desired_probability := 1 - complementary_probability

theorem probability_at_least_one_boy_one_girl :
  desired_probability = (574287 : ℚ) / 593775 :=
  sorry

end probability_at_least_one_boy_one_girl_l282_282420


namespace range_of_a_l282_282242

-- Define the propositions p and q
def prop_p (a : ℝ) : Prop := ∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0

def prop_q (a : ℝ) : Prop := ∃ x : ℝ, x^2 + 2 * a * x + 2 - a = 0

-- Define the main theorem which combines both propositions and infers the range of a
theorem range_of_a (a : ℝ) : prop_p a ∧ prop_q a → a ≤ -2 := sorry

end range_of_a_l282_282242


namespace shoveling_time_l282_282706

def shoveling_rate (h : Nat) : Int :=
  25 - h

def total_snow_volume (width length depth : Float) : Float :=
  width * length * depth

theorem shoveling_time :
  let volume := total_snow_volume 5 12 2.5
  let rec remaining_snow (h : Nat) : Int :=
    if volume - (List.sum (List.map shoveling_rate (List.range h))).toFloat <= 0 then h
    else remaining_snow (h + 1)
  volume == 150 ∧ remaining_snow 0 = 7 :=
by
  sorry

end shoveling_time_l282_282706


namespace largest_and_sum_of_six_consecutive_numbers_l282_282108

theorem largest_and_sum_of_six_consecutive_numbers (n : ℕ) 
(h : 2 * ((n - 2) + (n - 1) + n) = (n + 1) + (n + 2) + (n + 3)) : 
  ((max (max (max (max (n - 2) (n - 1)) n) (n + 1)) (n + 2)) (n + 3) = 7)
  ∧ ((n - 2) + (n - 1) + n + (n + 1) + (n + 2) + (n + 3) = 27) :=
by
 sorry

end largest_and_sum_of_six_consecutive_numbers_l282_282108


namespace sum_of_squares_of_roots_l282_282559

theorem sum_of_squares_of_roots : 
  ∀ (r1 r2 : ℝ), (r1 + r2 = 10) ∧ (r1 * r2 = 16) → (r1^2 + r2^2 = 68) :=
by
  intros r1 r2 h
  cases h with h1 h2
  sorry

end sum_of_squares_of_roots_l282_282559


namespace triangle_incenter_DJ_length_l282_282776

theorem triangle_incenter_DJ_length :
  ∀ (DEF : Triangle) (DE DF EF : ℝ) (J : Point),
  DEF.side (0, 1) = DE ∧ DEF.side (1, 2) = DF ∧ DEF.side (2, 0) = EF ∧
  J = incenter DEF ∧ 
  DE = 28 ∧ DF = 17 ∧ EF = 39 →
  dist DEF.vertices.0 J = 3 :=
by
  intros DEF DE DF EF J H
  cases H with H1 H2
  cases H2 with H3 H4
  cases H4 with H5 H6
  cases H6 with H7 H8
  cases H8 with H9 H10
  cases H10 with H11 H12
  cases H12 with H13 H14
  -- The proof would go here
  sorry

end triangle_incenter_DJ_length_l282_282776


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282280

theorem count_two_digit_perfect_squares_divisible_by_4 : 
  {n : ℕ | 10 ≤ n ∧ n < 100 ∧ ∃ k : ℕ, n = k^2 ∧ k^2 % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282280


namespace bryden_quarter_sale_l282_282484

theorem bryden_quarter_sale 
  (face_value : ℝ)
  (num_quarters : ℕ)
  (percentage : ℝ)
  (face_value_eq : face_value = 0.25)
  (num_quarters_eq : num_quarters = 7)
  (percentage_eq : percentage = 1500) :
  let multiplier := percentage / 100
  let total_face_value := num_quarters * face_value
  let total_received := multiplier * total_face_value
  total_received = 26.25 :=
by
  have h1 : multiplier = 15 := by sorry
  have h2 : total_face_value = 1.75 := by sorry
  have h3 : total_received = 26.25 := by sorry
  exact h3

end bryden_quarter_sale_l282_282484


namespace distance_PQ_l282_282659

-- Conditions
def parametric_eq_line (α : ℝ) (t : ℝ) : ℝ × ℝ :=
(x = 1 + t * cos α, y = t * sin α)

def polar_eq_curve (ρ θ : ℝ) : Prop :=
ρ * cos θ * cos θ - 4 * sin θ = 0

def point_P : ℝ × ℝ := (1, 0)

def point_M_polar : ℝ × ℝ := (1, π / 2)

-- Proving the value of |PQ|
theorem distance_PQ (P Q : ℝ × ℝ) (α : ℝ) (hα : α ≠ π / 2) :
  -- Signature of line l through M
  ∃ (l : ℝ → ℝ × ℝ), 
    (∀ t, parametric_eq_line α t == (1 + t * cos α, t * sin α)) →
    -- Signature of curve C
    ∃ (C : ℝ × ℝ → Prop), 
      (∀ (ρ θ : ℝ), polar_eq_curve ρ θ → C (ρ * cos θ, ρ * sin θ)) →
      -- Midpoint Q and distance PQ
      Q = (3 * sqrt 2, sqrt 2)  → 
      dist P Q = 3 * sqrt 2 :=
sorry

end distance_PQ_l282_282659


namespace median_of_set_with_conditions_l282_282399

theorem median_of_set_with_conditions (x : Finset ℤ) (h_card : x.card = 10) (h_range : x.sup - x.inf = 10) (h_max : x.sup = 20) :
  median_of_set x = 14.5 :=
sorry

end median_of_set_with_conditions_l282_282399


namespace total_visible_surface_area_of_tower_l282_282918

-- Define the volumes of the cubes
def volumes : List ℕ := [1, 27, 64, 125, 216]

-- Calculate the side length of a cube given its volume
def side_length (v : ℕ) : ℕ := Int.to_nat (Int.root 3 v)

-- The main theorem to prove the visible surface area calculation
theorem total_visible_surface_area_of_tower (V : List ℕ)
  (hl : V = volumes) :
  let side_lengths := V.map side_length
  let areas : List ℕ := 
    [(6^2 - 4 * ((1/2) * (5^2/2 +
    5 * side_length 6 * 5) *
    side_length 5 / V.tail.head) - 44 * 3),(5^2 - 
    4 * (4^2 / 4)),(4^2 - 
    4 * (3^4 / 27)),(3^2 - 
    4 * (1^5)),(top_length area S top_area)
    
  ∑ z ≤ l,i → P &
E|(intersect S area .side0 area)]
where
∀  𝓡₁ L₂ one : _
 P:=(1090-34*150) = cell * vol * nside 864ellij Z))
- 86410^3) = Z Z varea
  | S  corresponding_vol S , sides E 𝓣 10
  
:= ∑₀ J + My > 𝟏 ⊂ sol
  
∑_(volumes.every  STEM as).subset + )
( S = 1020.

:= 800000 /engineering_constructed_valid)).  t_status

end total_visible_surface_area_of_tower_l282_282918


namespace mod_product_equiv_l282_282125

theorem mod_product_equiv : 
  (2031 * 2032 * 2033 * 2034) % 7 = 3 := 
by
  have h1 : 2031 % 7 = 1 := by norm_num
  have h2 : 2032 % 7 = 2 := by norm_num
  have h3 : 2033 % 7 = 3 := by norm_num
  have h4 : 2034 % 7 = 4 := by norm_num
  calc
    (2031 * 2032 * 2033 * 2034) % 7
        = (1 * 2 * 3 * 4) % 7        : by rw [h1, h2, h3, h4]
    ... = 24 % 7                      : by norm_num
    ... = 3                           : by norm_num

end mod_product_equiv_l282_282125


namespace set_b_is_pythagorean_triple_l282_282801

theorem set_b_is_pythagorean_triple :
  ∃ (a b c : ℕ), a^2 + b^2 = c^2 ∧ a = 9 ∧ b = 40 ∧ c = 41 :=
by
  use 9
  use 40
  use 41
  split
  sorry

end set_b_is_pythagorean_triple_l282_282801


namespace sphere_surface_area_given_conditions_l282_282579

-- Definitions and conditions given in the problem
def side_length : ℝ := 6
def angle_OA_plane_ABC : ℝ := 45

-- Statements about the relationships in the problem
def circumradius_equilateral_triangle (a : ℝ) : ℝ := a / (Real.sqrt 3)

noncomputable def radius_sphere (R_triangle : ℝ) : ℝ :=
  R_triangle / (Real.cos (angle_OA_plane_ABC * Real.pi / 180))

noncomputable def surface_area_sphere (R_sphere : ℝ) : ℝ :=
  4 * Real.pi * R_sphere^2

-- Main theorem stating the solution to the problem
theorem sphere_surface_area_given_conditions :
  surface_area_sphere (radius_sphere (circumradius_equilateral_triangle side_length)) = 96 * Real.pi :=
by
  sorry

end sphere_surface_area_given_conditions_l282_282579


namespace measure_of_alpha_l282_282219

def point_P : ℝ × ℝ := (Real.sin (40 * Real.pi / 180), 1 + Real.cos (40 * Real.pi / 180))

theorem measure_of_alpha :
  ∃ α : ℝ, (α < Real.pi / 2 ∧ α > 0) ∧ (point_P = (Real.sin α, 1 + Real.cos α) ∧ α = 70 * Real.pi / 180) :=
by
  sorry

end measure_of_alpha_l282_282219


namespace triangle_ps_value_l282_282337

theorem triangle_ps_value
  (PQ PR PS QR QS SR : ℝ)
  (h1 : PQ = 13)
  (h2 : PR = 20)
  (h3 : QS / SR = 3 / 4)
  (h4 : QR^2 = QS^2 + SR^2)
  (h5 : PQ^2 = PS^2 + QS^2)
  (h6 : PR^2 = PS^2 + SR^2) :
  PS = 8 * real.sqrt 2 :=
sorry

end triangle_ps_value_l282_282337


namespace cheyenne_profit_l282_282121

noncomputable def total_pots : ℕ := 80
noncomputable def pots_cracked : ℕ := (2 / 5 : ℚ) * total_pots
noncomputable def remaining_pots : ℕ := total_pots - pots_cracked
noncomputable def cost_per_pot : ℚ := 15
noncomputable def first_tier_price : ℚ := 40
noncomputable def second_tier_price : ℚ := 35
noncomputable def third_tier_price : ℚ := 30
noncomputable def bulk_discount : ℚ := 0.05
noncomputable def pots_for_bulk_discount : ℕ := 10

noncomputable def revenue_first_tier : ℚ := min 20 remaining_pots * first_tier_price
noncomputable def revenue_second_tier : ℚ := min (remaining_pots - min 20 remaining_pots) 20 * second_tier_price
noncomputable def revenue_third_tier : ℚ := max (remaining_pots - 40) 0 * third_tier_price
noncomputable def total_revenue_before_discount : ℚ := revenue_first_tier + revenue_second_tier + revenue_third_tier
noncomputable def discount_amount : ℚ := if remaining_pots ≥ pots_for_bulk_discount then total_revenue_before_discount * bulk_discount else 0
noncomputable def total_revenue_after_discount : ℚ := total_revenue_before_discount - discount_amount
noncomputable def total_production_cost : ℚ := remaining_pots * cost_per_pot
noncomputable def profit : ℚ := total_revenue_after_discount - total_production_cost

theorem cheyenne_profit : profit = 933 := by
  sorry

end cheyenne_profit_l282_282121


namespace arithmetic_sequence_sum_l282_282578

theorem arithmetic_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) (a_1 : ℚ) (d : ℚ) (m : ℕ) 
    (ha1 : a_1 = 2) 
    (ha2 : a 2 + a 8 = 24)
    (ham : 2 * a m = 24) 
    (h_sum : ∀ n, S n = (n * (2 * a_1 + (n - 1) * d)) / 2) 
    (h_an : ∀ n, a n = a_1 + (n - 1) * d) : 
    S (2 * m) = 265 / 2 :=
by
    sorry

end arithmetic_sequence_sum_l282_282578


namespace cos_b_is_one_half_l282_282670

theorem cos_b_is_one_half (a b c : ℝ) (A B C : ℝ) 
    (H1 : a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0) 
    (H2 : a * a + b * b = c * c + 2 * a * b * (cos B)) 
    (H3 : cos B / cos C = b / (2 * a - c)) 
    (H4 : 1/2 * b * c * sin A = 3 * Real.sqrt 3 / 4) 
    (H5 : b = Real.sqrt 3) 
    : cos B = 1/2 := 
    sorry

end cos_b_is_one_half_l282_282670


namespace intersect_condition_l282_282606

theorem intersect_condition (m : ℕ) (h : m ≠ 0) : 
  (∃ x y : ℝ, (3 * x - 2 * y = 0) ∧ ((x - m)^2 + y^2 = 1)) → m = 1 :=
by 
  sorry

end intersect_condition_l282_282606


namespace cost_of_patent_is_correct_l282_282896

-- Defining the conditions
def c_parts : ℕ := 3600
def p : ℕ := 180
def n : ℕ := 45

-- Calculation of total revenue
def total_revenue : ℕ := n * p

-- Calculation of cost of patent
def cost_of_patent (total_revenue c_parts : ℕ) : ℕ := total_revenue - c_parts

-- The theorem to be proved
theorem cost_of_patent_is_correct (R : ℕ) (H : R = total_revenue) : cost_of_patent R c_parts = 4500 :=
by
  -- this is where your proof will go
  sorry

end cost_of_patent_is_correct_l282_282896


namespace servant_leaves_after_9_months_l282_282248

theorem servant_leaves_after_9_months :
  ∀ (months_worked : ℕ) (salary_in_rupes : ℕ) (turban_price : ℕ) (final_salary : ℕ),
    let monthly_salary := (salary_in_rupes + turban_price) / 12 in
    months_worked * monthly_salary + turban_price = final_salary →
    months_worked = 9 :=
by
  intros months_worked salary_in_rupes turban_price final_salary monthly_salary H
  sorry

end servant_leaves_after_9_months_l282_282248


namespace joshua_share_is_30_l282_282674

-- Definitions based on the conditions
def total_amount_shared : ℝ := 40
def ratio_joshua_justin : ℝ := 3

-- Proposition to prove
theorem joshua_share_is_30 (J : ℝ) (Joshua_share : ℝ) :
  J + ratio_joshua_justin * J = total_amount_shared → 
  Joshua_share = ratio_joshua_justin * J → 
  Joshua_share = 30 :=
sorry

end joshua_share_is_30_l282_282674


namespace value_of_f_neg3_l282_282599

def f (x : ℝ) : ℝ :=
  if x >= 0 then x + 1 else f (x + 2)

theorem value_of_f_neg3 : f (-3) = 2 := by
  sorry

end value_of_f_neg3_l282_282599


namespace fraction_undefined_at_one_l282_282459

theorem fraction_undefined_at_one (x : ℤ) (h : x = 1) : (x / (x - 1) = 1) := by
  have h : 1 / (1 - 1) = 1 := sorry
  sorry

end fraction_undefined_at_one_l282_282459


namespace inequality_transfers_l282_282586

variables (a b c d : ℝ)

theorem inequality_transfers (h1 : a > b) (h2 : c > d) : a - d > b - c := by
  sorry

end inequality_transfers_l282_282586


namespace correct_statements_hyperbola_l282_282964

theorem correct_statements_hyperbola 
  (a b : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (F1 F2 M N : ℝ × ℝ) 
  (origin : ℝ × ℝ := (0, 0))
  (h_hyperbola : ∀ (x y : ℝ), (x, y) ∈ { p : ℝ × ℝ | (p.1^2 / a^2) - (p.2^2 / b^2) = 1})
  (h_line_through_F2 : ∃ (L : ℝ × ℝ → Prop), ∀ p, L p ↔ ∃ k, p = (k * F2.1, k * F2.2))
  (h_intersects : ∃ M N, (M, N) ∈ { p : ℝ × ℝ | (p.1, p.2) ∈ { q : ℝ × ℝ | (q.1^2 / a^2) - (q.2^2 / b^2) = 1 } ∧ ∀ k, (k * F2.1, k * F2.2) = p.1 ∧ p.2 })
  (h_angles : ∠ (N, F1, F2) = 2 * ∠ (N, F2, F1))
  (h_vectors : vectorAdd origin N + vectorAdd origin F2 = 2 * vectorAdd origin M)
  :
  (area_triangle M F1 F2 = Real.sqrt 3 * a^2) ∧ 
  (Real.tan (angle M F1 F2) = Real.sqrt 3 / 5) := sorry

end correct_statements_hyperbola_l282_282964


namespace tree_sidewalk_space_l282_282325

theorem tree_sidewalk_space
  (num_trees : ℕ)
  (distance_between_trees : ℝ)
  (total_road_length : ℝ)
  (total_gaps : ℝ)
  (space_each_tree : ℝ)
  (H1 : num_trees = 11)
  (H2 : distance_between_trees = 14)
  (H3 : total_road_length = 151)
  (H4 : total_gaps = (num_trees - 1) * distance_between_trees)
  (H5 : space_each_tree = (total_road_length - total_gaps) / num_trees)
  : space_each_tree = 1 := 
by
  sorry

end tree_sidewalk_space_l282_282325


namespace regression_value_at_e10_l282_282593

-- Define y_hat as the regression equation
def y_hat (b : ℝ) (x : ℝ) : ℝ := b * Real.log x + 0.24

-- Given data points
def data_points : List (ℝ × ℝ) :=
  [(Real.exp 1, 1), (Real.exp 3, 2), (Real.exp 4, 3), (Real.exp 6, 4), (Real.exp 7, 5)]

-- Calculate means of ln(x) and y
def mean_t : ℝ := (1 + 3 + 4 + 6 + 7) / 5
def mean_y : ℝ := (1 + 2 + 3 + 4 + 5) / 5

-- Calculate the slope b
def slope_b : ℝ := (mean_y - 0.24) / mean_t

-- The theorem to prove
theorem regression_value_at_e10 : y_hat slope_b (Real.exp 10) ≈ 6.81 :=
by
  sorry

end regression_value_at_e10_l282_282593


namespace length_of_first_wall_l282_282294

def work_done (persons : ℕ) (days : ℕ) : ℕ := persons * days

theorem length_of_first_wall :
  let W1 := work_done 18 42,
      W2 := work_done 30 18,
      L2 := 100 in
  (W1 / W2 : ℚ) = 7 / 5 → L1 = 140 :=
by
  intro h,
  rw [W1, W2],
  simp at h,
  sorry

end length_of_first_wall_l282_282294


namespace union_of_A_and_B_l282_282616

def A (x : ℝ) : Set ℝ := {x^2, 2 * x - 1, -4}
def B (x : ℝ) : Set ℝ := {x - 5, 1 - x, 9}
def H : {9 : ℝ} ⊆ A (-3) ∩ B (-3) := by 
sorry

theorem union_of_A_and_B :
  Set.union (A (-3)) (B (-3)) = {-8, -7, -4, 4, 9} := by 
sorry

end union_of_A_and_B_l282_282616


namespace complex_pure_imaginary_is_3i_l282_282637

/-- If the complex number z = (a + 3 * complex.i) / (1 - 2 * complex.i) 
    (where a is a real number and i is the imaginary unit) is a pure 
    imaginary number, then the complex number z is 3 * i. -/
theorem complex_pure_imaginary_is_3i (a : ℝ) (z : ℂ)
  (h : z = (a + 3 * complex.i) / (1 - 2 * complex.i))
  (hz : ∃ b : ℝ, z = b * complex.i) :
  a = 6 ∧ z = 3 * complex.i :=
sorry

end complex_pure_imaginary_is_3i_l282_282637


namespace mother_daughter_age_l282_282849

theorem mother_daughter_age (x : ℕ) :
  let mother_age := 42
  let daughter_age := 8
  (mother_age + x = 3 * (daughter_age + x)) → x = 9 :=
by
  let mother_age := 42
  let daughter_age := 8
  intro h
  sorry

end mother_daughter_age_l282_282849


namespace fixed_point_inevitability_l282_282923

-- Definition of the problem condition
def passes_through_fixed_point (k : ℝ) (f : ℝ → ℝ) : Prop :=
  f 5 = 225

-- The function defined by the given equation
def f (x k : ℝ) : ℝ := 9 * x^2 + k * x - 5 * k

-- The Lean 4 statement
theorem fixed_point_inevitability : ∀ k : ℝ, passes_through_fixed_point k (f k) := 
by
  intros k
  -- The function applied to the point (5, 225)
  unfold passes_through_fixed_point
  -- The proof that it passes through the point
  calc f 5 k = 9 * (5 : ℝ)^2 + k * (5 : ℝ) - 5 * k : rfl
           ... = 225 : by ring

end fixed_point_inevitability_l282_282923


namespace sum_of_largest_and_smallest_l282_282627

def digits := {3, 5, 7, 8}

def largest_two_digit (digits: Set Nat) : Nat :=
  let largest := digits.max'
  let second_largest := digits.erase largest |>.max'
  10 * largest + second_largest

def smallest_two_digit (digits: Set Nat) : Nat :=
  let smallest := digits.min'
  let second_smallest := digits.erase smallest |>.min'
  10 * smallest + second_smallest

theorem sum_of_largest_and_smallest :
  largest_two_digit digits + smallest_two_digit digits = 122 :=
by
  sorry

end sum_of_largest_and_smallest_l282_282627


namespace theta_eq_c_describes_plane_l282_282921

variables {ρ : ℝ} {θ : ℝ} {φ : ℝ}
variable c : ℝ

def is_azimuthal_angle (θ : ℝ) : Prop :=
  θ = c

theorem theta_eq_c_describes_plane (h : is_azimuthal_angle θ) : 
  ∀ (ρ : ℝ) (φ : ℝ), True := 
by
  sorry

end theta_eq_c_describes_plane_l282_282921


namespace minimum_munificence_of_monic_cubic_polynomial_l282_282564

noncomputable def munificence (q : ℝ → ℝ) : ℝ :=
  real.Sup (set.image (λ x, abs (q x)) (set.Icc (-1 : ℝ) 1))

theorem minimum_munificence_of_monic_cubic_polynomial :
  ∃ (b c d : ℝ), ∀ q : ℝ → ℝ, q = λ x, x^3 + b * x^2 + c * x + d → munificence q = 1 :=
sorry

end minimum_munificence_of_monic_cubic_polynomial_l282_282564


namespace candles_to_choose_l282_282122

theorem candles_to_choose (C : ℕ)
  (h1 : ∀ (k : ℕ), k = 2 → (∃ n, n = C.choose k))
  (h2 : ∀ (k : ℕ), k = 8 → (∃ n, n = 9.choose k))
  (h3 : ∃ n, n = 54):
  C = 4 :=
by
  sorry

end candles_to_choose_l282_282122


namespace trig_identity_l282_282814

theorem trig_identity (α : ℝ) : 
  sin^2 (π + α) - cos (π + α) * cos (-α) + 1 = 2 := 
by
  sorry

end trig_identity_l282_282814


namespace area_ratio_greater_l282_282016

theorem area_ratio_greater 
  {A B C A1 B1 C1 : Type*}
  (triangle_ABC : is_triangle A B C)
  (triangle_A1B1C1 : is_triangle A1 B1 C1)
  (acute_ABC : is_acute A B C)
  (acute_A1B1C1 : is_acute A1 B1 C1)
  (B1_on_BC : is_on_segment B C B1)
  (C1_on_BC : is_on_segment B C C1)
  (A1_inside_ABC : is_inside A1 A B C):
  (area A B C / (distance A B + distance A C) > area A1 B1 C1 / (distance A1 B1 + distance A1 C1)) :=
sorry

end area_ratio_greater_l282_282016


namespace ability_upgrade_ways_l282_282649

theorem ability_upgrade_ways :
  let n := 16 -- Total levels
  let k := 2 -- Number of upgrades required for ability C
  let options := 3 -- Number of choices per level for abilities A and B
  let levels_after_c := n - k -- Levels left after upgrading ability C twice
  ∑ i in (finset.range 10).map (λ i => 10 - i) * options^levels_after_c = 5 * options^n :=
by
  -- Definitions used in the enumerations based on constraints can be defined here
  let c_upgrade_ways := ∑ i in (finset.range 10), (10 - i)
  let other_upgrade_ways := options^levels_after_c
  calc
  c_upgrade_ways * other_upgrade_ways = 45 * 3^14 : sorry -- Splitting and combining ways
diexiststo  5 * 3^16  : by sorry -- Combining the result
some 

end ability_upgrade_ways_l282_282649


namespace max_gretel_points_l282_282249

-- Define the 7x8 board
def board := fin 7 × fin 8

-- Define the scoring system
def gretel_points (row_pieces : ℕ) (col_pieces : ℕ) : ℕ :=
  4 * row_pieces + 3 * col_pieces

-- Define total moves per player
def total_moves := 28

-- Hansel goes first
def hansel_first := true

-- Game ends when all cells are filled
def game_end : Prop := (total_moves * 2 = 7 * 8)

-- Maximum points Gretel can earn
def max_points_gretel := 700

theorem max_gretel_points : ∀ (initial_board : board → option bool),
  (∀ move : ℕ, move < 2 * total_moves → 
    ∃ cell : board, initial_board cell = none) →
  gretel_points 0 0 = 0 →
  game_end →
  max_points_gretel = 700 :=
by 
  intros initial_board moves_exist initial_points game_end_proved
  sorry

end max_gretel_points_l282_282249


namespace concurrency_of_cevians_l282_282784

theorem concurrency_of_cevians :
  ∀ (A B C D E F G H I : Type)
  [is_point A] [is_point B] [is_point C] [is_point D] [is_point E] [is_point F] [is_point G] [is_point H] [is_point I],
  (equilateral_triangle A B D) →
  (equilateral_triangle B C E) →
  (equilateral_triangle C A F) →
  (is_midpoint_of G B D) →
  (is_midpoint_of H B E) →
  (is_centroid_of I C A F) →
  concurrent (line_through A H) (line_through C G) (line_through B I) :=
by {
  sorry
}

end concurrency_of_cevians_l282_282784


namespace hyperbola_asymptote_l282_282224

theorem hyperbola_asymptote (a : ℝ) (h : a > 0) (P : 2 * (sqrt a) = 2 * (sqrt 3)) : a = 3 :=
by
  -- Proof goes here
  sorry

end hyperbola_asymptote_l282_282224


namespace range_of_a_l282_282962

noncomputable def f (a : ℝ) : Piecewise ℝ ℝ
  | x <= 1 := -x^2 - a * x - 5
  | 1 < x := a / x

theorem range_of_a (a : ℝ) : (∀ x y, x ≤ y → f a x ≤ f a y) → (a ∈ set.Icc (-3) (-2)) :=
by
  intro hf
  sorry

end range_of_a_l282_282962


namespace coins_player_1_received_l282_282424

def round_table := List Nat
def players := List Nat
def coins_received (table: round_table) (player_idx: Nat) : Nat :=
sorry -- the function to calculate coins received by player's index

-- Define the given conditions
def sectors : round_table := [1, 2, 3, 4, 5, 6, 7, 8, 9]
def num_players := 9
def num_rotations := 11
def player_4 := 4
def player_8 := 8
def player_1 := 1
def coins_player_4 := 90
def coins_player_8 := 35

theorem coins_player_1_received : coins_received sectors player_1 = 57 :=
by
  -- Setup the conditions
  have h1 : coins_received sectors player_4 = 90 := sorry
  have h2 : coins_received sectors player_8 = 35 := sorry
  -- Prove the target statement
  show coins_received sectors player_1 = 57
  sorry

end coins_player_1_received_l282_282424


namespace contrapositive_of_proposition_l282_282736

theorem contrapositive_of_proposition (f : ℝ → ℝ) (a b : ℝ) :
  (a + b >= 0) → (f(a) + f(b) >= f(-a) + f(-b)) →
  (f(a) + f(b) < f(-a) + f(-b)) → (a + b < 0) := 
by
  intro h1
  intro h2
  intro h3
  -- Proof goes here
  sorry

end contrapositive_of_proposition_l282_282736


namespace number_of_different_natural_numbers_is_15_l282_282007

/-- Given three cards labeled 1, 2, and 3, the total number of different natural numbers 
    that can be formed using these cards is 15.
 -/
theorem number_of_different_natural_numbers_is_15 : 
  let cards := {1, 2, 3} in
  ∃ (n : ℕ), n = 15 ∧ (∀ m ∈ cards, ∃ p : ℕ, p ∈ (permute cards) ∧ (1 ≤ p /\ p ≤ m)) :=
  sorry

end number_of_different_natural_numbers_is_15_l282_282007


namespace prove_additional_minutes_needed_l282_282353

-- Assume the given conditions as definitions in Lean 4
def number_of_classmates := 30
def initial_gathering_time := 120   -- in minutes (2 hours)
def time_per_flower := 10           -- in minutes
def flowers_lost := 3

-- Calculate the flowers gathered initially
def initial_flowers_gathered := initial_gathering_time / time_per_flower

-- Calculate flowers remaining after loss
def flowers_remaining := initial_flowers_gathered - flowers_lost

-- Calculate additional flowers needed
def additional_flowers_needed := number_of_classmates - flowers_remaining

-- Therefore, calculate the additional minutes required to gather the remaining flowers
def additional_minutes_needed := additional_flowers_needed * time_per_flower

theorem prove_additional_minutes_needed :
  additional_minutes_needed = 210 :=
by 
  unfold additional_minutes_needed additional_flowers_needed flowers_remaining initial_flowers_gathered
  sorry

end prove_additional_minutes_needed_l282_282353


namespace water_intake_proof_l282_282012

variable {quarts_per_bottle : ℕ} {bottles_per_day : ℕ} {extra_ounces_per_day : ℕ} 
variable {days_per_week : ℕ} {ounces_per_quart : ℕ} 

def total_weekly_water_intake 
    (quarts_per_bottle : ℕ) 
    (bottles_per_day : ℕ) 
    (extra_ounces_per_day : ℕ) 
    (ounces_per_quart : ℕ) 
    (days_per_week : ℕ) 
    (correct_answer : ℕ) : Prop :=
    (quarts_per_bottle * ounces_per_quart * bottles_per_day + extra_ounces_per_day) * days_per_week = correct_answer

theorem water_intake_proof : 
    total_weekly_water_intake 3 2 20 32 7 812 := 
by
    sorry

end water_intake_proof_l282_282012


namespace ellipse_proof_l282_282591

noncomputable def ellipse_equation (C : ℝ → ℝ → Prop) : Prop :=
∀ (x y : ℝ), C x y ↔ (x^2 / 4 + y^2 / 3 = 1)

noncomputable def point_on_ellipse (C : ℝ → ℝ → Prop) (M : ℝ × ℝ) : Prop :=
C M.1 M.2

noncomputable def eccentricity_condition (a b : ℝ) (e : ℝ) : Prop :=
e = 1/2

noncomputable def max_area_condition (C : ℝ → ℝ → Prop) (l : ℝ → ℝ → Prop) (O : ℝ × ℝ) (M : ℝ × ℝ) : Prop :=
let N := midpoint (intersection_points l C) in
N_on_line_OM : (N.2 = 1/2 * N.1) → max_triangle_area O (intersection_points l C) = sqrt 3

theorem ellipse_proof :
  (∃ (a b : ℝ), a > b ∧ b > 0 ∧ eccentricity_condition a b (1/2)) →
  (∃ (C : ℝ → ℝ → Prop), (∀ (x y : ℝ), C x y ↔ (x^2 / 4 + y^2 / 3 = 1)) ∧ point_on_ellipse C (sqrt 3, sqrt 3 / 2)) ∧ 
  (∃ (l : ℝ → ℝ → Prop), l_does_not_pass_origin l → max_area_condition C l (0,0) (sqrt 3, sqrt 3 / 2)) :=
sorry

end ellipse_proof_l282_282591


namespace distinct_ways_to_distribute_l282_282979

theorem distinct_ways_to_distribute :
  ∃ (n : ℕ), n = 7 ∧ ∀ (balls : ℕ) (boxes : ℕ)
    (indistinguishable_balls : Prop := true) 
    (indistinguishable_boxes : Prop := true), 
    balls = 6 → boxes = 3 → 
    indistinguishable_balls → 
    indistinguishable_boxes → 
    n = 7 :=
by
  sorry

end distinct_ways_to_distribute_l282_282979


namespace ellen_smoothie_ingredients_l282_282902

theorem ellen_smoothie_ingredients :
  let strawberries := 0.2
  let yogurt := 0.1
  let orange_juice := 0.2
  strawberries + yogurt + orange_juice = 0.5 :=
by
  sorry

end ellen_smoothie_ingredients_l282_282902


namespace min_f_l282_282587

noncomputable def f (x y z : ℝ) : ℝ :=
  (3 * x^2 - x) / (1 + x^2) + 
  (3 * y^2 - y) / (1 + y^2) + 
  (3 * z^2 - z) / (1 + z^2)

theorem min_f (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (hxyz : x + y + z = 1) :
  f x y z = 0 ↔ x = 1/3 ∧ y = 1/3 ∧ z = 1/3 :=
sorry

end min_f_l282_282587


namespace irr_number_among_given_list_l282_282856

def is_rational (x : ℚ) : Prop := sorry
def is_irrational (x : ℝ) : Prop := ¬(∃ q : ℚ, q.to_real = x)

noncomputable def num1 : ℚ := 31415 / 10000
noncomputable def num2 : ℝ := Real.sqrt 7
noncomputable def num3 : ℤ := 3
noncomputable def num4 : ℚ := 1 / 3

theorem irr_number_among_given_list :
  is_irrational num2 :=
sorry

end irr_number_among_given_list_l282_282856


namespace particle_maximum_height_l282_282491

theorem particle_maximum_height :
  (∀ (t : ℝ), (180 * t - 18 * t^2) ≤ 450) ∧ (∃ (t : ℝ), 180 * t - 18 * t^2 = 450) := 
by
  -- Conditions
  let h : ℝ → ℝ := λ t, 180 * t - 18 * t^2
  
  -- To show that for every t, the value of h(t) <= 450
  have h_le_450 : ∀ t : ℝ, h t ≤ 450 := sorry
  
  -- To show that there exists a t such that the value of h(t) = 450
  have h_450 : ∃ t : ℝ, h t = 450 := by
    use 5
    simp [h]
    linarith
  
  exact ⟨h_le_450, h_450⟩

end particle_maximum_height_l282_282491


namespace number_of_integer_length_chords_through_point_l282_282394

theorem number_of_integer_length_chords_through_point 
  (r : ℝ) (d : ℝ) (P_is_5_units_from_center : d = 5) (circle_has_radius_13 : r = 13) :
  ∃ n : ℕ, n = 3 := by
  sorry

end number_of_integer_length_chords_through_point_l282_282394


namespace problem_solution_l282_282160

theorem problem_solution (x : ℝ) (h1 : x > 9) 
(h2 : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x ∈ Set.Ici 18 := sorry

end problem_solution_l282_282160


namespace molds_cost_is_correct_l282_282525

variables (total_money : ℝ) (cost_popsicle_sticks : ℝ) (popsicle_sticks_in_pack : ℕ)
          (cost_per_bottle : ℝ) (popsicles_per_bottle : ℕ) (leftover_sticks : ℕ)

def molds_cost (total_money : ℝ) (cost_popsicle_sticks : ℝ) (popsicle_sticks_in_pack : ℕ)
               (cost_per_bottle : ℝ) (popsicles_per_bottle : ℕ) (leftover_sticks : ℕ) : ℝ :=
total_money - (cost_popsicle_sticks + (popsicle_sticks_in_pack - leftover_sticks) / popsicles_per_bottle * cost_per_bottle)

theorem molds_cost_is_correct :
  molds_cost 10 1 100 2 20 40 = 3 :=
by
  -- Proof goes here
  sorry

end molds_cost_is_correct_l282_282525


namespace number_of_students_in_Diligence_before_transfer_l282_282999

-- Define the total number of students and the transfer information
def total_students : ℕ := 50
def transferred_students : ℕ := 2

-- Define the number of students in Diligence before the transfer
def students_in_Diligence_before : ℕ := 23

-- Let's prove that the number of students in Diligence before the transfer is 23
theorem number_of_students_in_Diligence_before_transfer :
  (total_students / 2) - transferred_students = students_in_Diligence_before :=
by {
  -- The proof is omitted as instructed
  sorry
}

end number_of_students_in_Diligence_before_transfer_l282_282999


namespace sum_geometric_sequence_l282_282181

theorem sum_geometric_sequence (a r : ℝ) (n : ℕ) (h_a : a = 1/3) (h_r : r = 1/3) (h_n : n = 8) :
  let S_n := a * (1 - r^n) / (1 - r) in S_n = 3280/6561 :=
by
  sorry

end sum_geometric_sequence_l282_282181


namespace TC_eq_KD_eq_AS_l282_282751

-- Declaration of points and properties
variables {A B C M N K D T S : Point}
variables {I : Point}

-- Conditions of the problem
variables (incircle_tangent_M : tangential_to (incircle I) A B M)
variables (incircle_tangent_N : tangential_to (incircle I) B C N)
variables (incircle_tangent_K : tangential_to (incircle I) C A K)
variables (midpoint_D : midpoint D A C)
variables (parallel_l_MN : parallel (line_through D) (line_through M N))
variables (intersection_l_BC : intersect (line_through D) (line_through B C) T)
variables (intersection_l_BD : intersect (line_through D) (line_through B D) S)

-- Theorem statement
theorem TC_eq_KD_eq_AS
  (incircle_tangent_M : tangential_to (incircle I) A B M)
  (incircle_tangent_N : tangential_to (incircle I) B C N)
  (incircle_tangent_K : tangential_to (incircle I) C A K)
  (midpoint_D : midpoint D A C)
  (parallel_l_MN : parallel (line_through D) (line_through M N))
  (intersection_l_BC : intersect (line_through D) (line_through B C) T)
  (intersection_l_BD : intersect (line_through D) (line_through B D) S)
  : length (segment T C) = length (segment K D) ∧ length (segment K D) = length (segment A S) ∧ length (segment T C) = length (segment A S) := sorry

end TC_eq_KD_eq_AS_l282_282751


namespace toms_age_l282_282098

theorem toms_age
  (T : ℕ)
  (h_adam_age : 8)
  (h_combined_age_in_12_years : 8 + 12 + (T + 12) = 44) :
  T = 12 :=
by
  sorry

end toms_age_l282_282098


namespace altitudes_outside_0_2_3_or_4_l282_282621

-- Define a general tetrahedron with vertices A, B, C, D
variables (A B C D : Point)

-- Define conditions on dihedral angles (faces intersecting at line)
variable (dihedral_angle_ABC_ABD : Angle)
variable (dihedral_angle_ABCD_ABD : Angle)
-- More variables representing the other dihedral angles if required...

-- The problem needs us to prove the number of altitudes outside the tetrahedron

definition altitude_outside_the_tetrahedron (A B C D : Point) : ℕ :=
-- This would mathematically define the number of altitudes outside
sorry

theorem altitudes_outside_0_2_3_or_4 
    (h1 : dihedral_angle_ABC_ABD.is_obtuse ∨ dihedral_angle_ABC_ABD.is_acute)
    (h2 : dihedral_angle_ABCD_ABD.is_obtuse ∨ dihedral_angle_ABCD_ABD.is_acute)
    -- Additional conditions h3, h4, as needed for other dihedral angles
    :
    altitude_outside_the_tetrahedron A B C D = 0 ∨ 
    altitude_outside_the_tetrahedron A B C D = 2 ∨ 
    altitude_outside_the_tetrahedron A B C D = 3 ∨
    altitude_outside_the_tetrahedron A B C D = 4 := 
sorry

end altitudes_outside_0_2_3_or_4_l282_282621


namespace sum_of_solutions_f_eq_3_l282_282694

def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x - 5 else x ^ 2 - 1

theorem sum_of_solutions_f_eq_3 : ∃ x : ℝ, f x = 3 ∧ x = 2 := 
begin
  -- Here will go the proof
  sorry
end

end sum_of_solutions_f_eq_3_l282_282694


namespace nancy_pictures_at_zoo_l282_282046

theorem nancy_pictures_at_zoo (deleted museum remaining : ℕ) (H1 : deleted = 38) (H2 : museum = 8) (H3 : remaining = 19) : ∃ (zoo : ℕ), zoo = 49 :=
by
  use 49
  split
  sorry

end nancy_pictures_at_zoo_l282_282046


namespace sqrt_meaningful_range_l282_282309

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y = sqrt (x - 3)) → x ≥ 3 := sorry

end sqrt_meaningful_range_l282_282309


namespace max_consecutive_integers_sum_48_l282_282787

-- Define the sum of consecutive integers
def sum_consecutive_integers (a N : ℤ) : ℤ :=
  (N * (2 * a + N - 1)) / 2

-- Define the main theorem
theorem max_consecutive_integers_sum_48 : 
  ∃ N a : ℤ, sum_consecutive_integers a N = 48 ∧ (∀ N' : ℤ, ((N' * (2 * a + N' - 1)) / 2 = 48) → N' ≤ N) :=
sorry

end max_consecutive_integers_sum_48_l282_282787


namespace geom_seq_sum_of_squares_l282_282334

noncomputable theory
open BigOperators

-- Definitions for geometric sequence and sum
def S : ℕ → ℕ := λ n, 2^n - 1
def an : ℕ → ℕ
| 1     := 1
| (n+1) := S (n+1) - S n

theorem geom_seq_sum_of_squares (n : ℕ) :
  ∑ i in Finset.range n, (an (i + 1))^2 = (4^n - 1) / 3 := by
  sorry

end geom_seq_sum_of_squares_l282_282334


namespace train_crossing_time_l282_282977

-- Definitions
def train_length : ℝ := 150
def train_speed_kmh : ℝ := 72
def bridge_length : ℝ := 132

-- Conversion factor from km/hr to m/s
def kmh_to_ms (v : ℝ) : ℝ := v * (1000 / 3600)

-- Total distance to be traveled
def total_distance := train_length + bridge_length

-- Speed in m/s
def train_speed_ms := kmh_to_ms train_speed_kmh

-- Time calculation
def time_to_cross_bridge := total_distance / train_speed_ms

-- Theorem stating that the time is indeed 14.1 seconds
theorem train_crossing_time : time_to_cross_bridge = 14.1 := by
  sorry

end train_crossing_time_l282_282977


namespace largest_integer_k_l282_282136

noncomputable def T : ℕ → ℝ
| 1     := 3
| (n+1) := 3 ^ (T n)

def A := (T 2005) ^ (T 2005)
def B := (T 2005) ^ A

theorem largest_integer_k :
  ∃ k : ℕ, (k = 2005) ∧ (∀ n : ℕ, n > 2005 → (nat.pred 2005 = 2004) → (nat.pred (nat.pred ... n times ... (T 2005)))) = 3 :=
sorry

end largest_integer_k_l282_282136


namespace general_solution_linear_diophantine_l282_282910

theorem general_solution_linear_diophantine (a b c : ℤ) (h_coprime : Int.gcd a b = 1)
    (x1 y1 : ℤ) (h_particular_solution : a * x1 + b * y1 = c) :
    ∃ (t : ℤ), (∃ (x y : ℤ), x = x1 + b * t ∧ y = y1 - a * t ∧ a * x + b * y = c) ∧
               (∃ (x' y' : ℤ), x' = x1 - b * t ∧ y' = y1 + a * t ∧ a * x' + b * y' = c) :=
by
  sorry

end general_solution_linear_diophantine_l282_282910


namespace ratio_of_shaded_area_to_non_shaded_area_l282_282440

noncomputable def equilateral_triangle_shaded_area_ratio (s : ℝ) : ℝ :=
  let area_triangle : ℝ := (Math.sqrt 3) / 4 * s^2
  let area_inner_triangle : ℝ := (Math.sqrt 3) / 16 * s^2
  let trapezoid_area (b1 b2 h : ℝ) : ℝ := (1/2) * (b1 + b2) * h
  let area_trapezoids : ℝ := 3 * trapezoid_area (s/2) (s/4) ((Math.sqrt 3)/4 * s)
  let shaded_area : ℝ := area_inner_triangle + area_trapezoids
  let non_shaded_area : ℝ := area_triangle - shaded_area
  shaded_area / non_shaded_area

theorem ratio_of_shaded_area_to_non_shaded_area (s : ℝ) (s_pos : 0 < s) :
  equilateral_triangle_shaded_area_ratio s = 11 / 21 :=
by
  sorry

end ratio_of_shaded_area_to_non_shaded_area_l282_282440


namespace degrees_to_radians_l282_282894

theorem degrees_to_radians (π_radians : ℝ) : 150 * π_radians / 180 = 5 * π_radians / 6 :=
by sorry

end degrees_to_radians_l282_282894


namespace integers_on_blackboard_l282_282713

-- Definitions from conditions
def frac_part (x : ℚ) : ℚ := x - ⌊x⌋

-- The problem statement
theorem integers_on_blackboard (a : Fin n → ℚ) 
(h1 : Multiset.map frac_part (Multiset.ofFinFun a) = 
      Multiset.map frac_part (Multiset.ofFinFun (λ i, a i * a i))) : 
∀ i, a i ∈ ℤ :=
by
  sorry

end integers_on_blackboard_l282_282713


namespace proof_complement_union_l282_282063

open Set

variable (U A B: Set Nat)

def complement_equiv_union (U A B: Set Nat) : Prop :=
  (U \ A) ∪ B = {0, 2, 3, 6}

theorem proof_complement_union: 
  U = {0, 1, 3, 5, 6, 8} → 
  A = {1, 5, 8} → 
  B = {2} → 
  complement_equiv_union U A B :=
by
  intros h1 h2 h3
  subst h1
  subst h2
  subst h3
  -- Proof omitted
  sorry

end proof_complement_union_l282_282063


namespace num_elements_in_all_sets_l282_282682

open Set

def exactly_one (A B C : Set ℕ) : Set ℕ :=
  (A ∪ B ∪ C) \ ((A ∩ B) ∪ (A ∩ C) ∪ (B ∩ C))

def exactly_two (A B C : Set ℕ) : Set ℕ :=
  ((A ∩ B) ∪ (A ∩ C) ∪ (B ∩ C)) \ (A ∩ B ∩ C)

def exactly_three (A B C : Set ℕ) : Set ℕ :=
  A ∩ B ∩ C

theorem num_elements_in_all_sets
  (A B C : Set ℕ)
  (hA : A.card = 100)
  (hB : B.card = 50)
  (hC : C.card = 48)
  (h1 : (exactly_one A B C).card = 2 * (exactly_two A B C).card)
  (h2 : (exactly_one A B C).card = 3 * (exactly_three A B C).card):
  (exactly_three A B C).card = 22 :=
by
  sorry

end num_elements_in_all_sets_l282_282682


namespace proj_u_onto_v_l282_282920

open Real

noncomputable def vector := ℝ × ℝ

noncomputable def dot_product (u v : vector) : ℝ :=
u.1 * v.1 + u.2 * v.2

noncomputable def proj (u v : vector) : vector :=
let uv := dot_product u v
let vv := dot_product v v
(uv / vv) * v.1, (uv / vv) * v.2

theorem proj_u_onto_v :
  let u := (3, 4) : vector
  let v := (2, -1) : vector
  proj u v = (4/5, -2/5) :=
by 
sorry

end proj_u_onto_v_l282_282920


namespace geometric_sequence_sum_l282_282191

theorem geometric_sequence_sum :
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  (a * (1 - r^n) / (1 - r)) = 3280 / 6561 :=
by {
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  calc
  (a * (1 - r^n) / (1 - r)) = (1/3 * (1 - (1/3)^8) / (1 - 1/3)) : by rw a; rw r
  ... = 3280 / 6561 : sorry
}

end geometric_sequence_sum_l282_282191


namespace range_of_s_l282_282794

def s (x : ℝ) : ℝ := 1 / (2 - x) ^ 3

theorem range_of_s :
  set.range s = set.univ \ {0} :=
sorry

end range_of_s_l282_282794


namespace triangle_inequality_and_equality_condition_l282_282708

theorem triangle_inequality_and_equality_condition
    (A B C M : Point)
    (d_a d_b d_c r R : ℝ)
    (hM_inside : M ∈ triangle ABC)
    (h_d_a : d_a = dist_from_point_to_side M (side BC))
    (h_d_b : d_b = dist_from_point_to_side M (side CA))
    (h_d_c : d_c = dist_from_point_to_side M (side AB))
    (h_r : r = inradius ABC)
    (h_R : R = circumradius ABC) :
  (d_a^2 + d_b^2 + d_c^2 ≥ 12 * r^4 / R^2) ∧ 
  ((d_a^2 + d_b^2 + d_c^2 = 12 * r^4 / R^2) ↔ 
  (is_equilateral ABC ∧ M = centroid_of_triangle ABC)) :=
by sorry

end triangle_inequality_and_equality_condition_l282_282708


namespace div_powers_same_base_l282_282120

variable (x : ℝ)

theorem div_powers_same_base : x^8 / x^2 = x^6 :=
by
  sorry

end div_powers_same_base_l282_282120


namespace find_A_l282_282083

theorem find_A (A : ℕ) (B : ℕ) (h₁ : 0 ≤ B ∧ B ≤ 999) (h₂ : 1000 * A + B = A * (A + 1) / 2) : A = 1999 :=
  sorry

end find_A_l282_282083


namespace isosceles_trapezoid_base_ratio_correct_l282_282329

def isosceles_trapezoid_ratio (x y a b : ℝ) : Prop :=
  b = 2 * x ∧ a = 2 * y ∧ a + b = 10 ∧ (y * (Real.sqrt 2 + 1) = 5) →

  (a / b = (2 * (Real.sqrt 2) - 1) / 2)

theorem isosceles_trapezoid_base_ratio_correct: ∃ (x y a b : ℝ), 
  isosceles_trapezoid_ratio x y a b := sorry

end isosceles_trapezoid_base_ratio_correct_l282_282329


namespace product_of_slopes_l282_282604

theorem product_of_slopes (a b e x y : ℝ) (h_hyperbola : x^2 / a^2 - y^2 / b^2 = 1)
  (h_eccentricity : e = real.sqrt 3) (h_relation : b = real.sqrt 2 * a) :
  ((y / (x + a)) * (y / (-x + a)) = -2) := 
  sorry

end product_of_slopes_l282_282604


namespace find_B_find_area_l282_282991

variables (A B C a b c : ℝ)

-- Given conditions in the form of definitions
def condition1 := ∀ (A B C : ℝ), 0 < B ∧ B < π
def condition2 := ∀ (a b c : ℝ), 0 < a ∧ 0 < b ∧ 0 < c

def vector_m := (a / 2, c / 2)
def vector_n := (cos C, cos A)

def dot_product_condition := (a / 2) * cos C + (c / 2) * cos A = b * cos B
def cos_sin_condition := cos ((A - C) / 2) = sqrt 3 * sin A
def vector_magnitude := (a ^ 2 + c ^ 2) / 4 = 5

-- Questions/formalized statements
theorem find_B (h1 : condition1 A B C) (h2 : condition2 a b c) 
  (h3 : vector_magnitude a c) 
  (h4 : dot_product_condition a b B C A) 
  (h5 : cos_sin_condition A C) : 
  B = π / 3 := sorry

theorem find_area (h1 : condition1 A B C) (h2 : condition2 a b c) 
  (h3 : vector_magnitude a c) 
  (h4 : dot_product_condition a b B C A) 
  (h5 : cos_sin_condition A C) :
  let area := 1 / 2 * a * 2 * sqrt 3
  in area = 2 * sqrt 3 := sorry

end find_B_find_area_l282_282991


namespace tan_alpha_plus_pi_over_4_and_alpha_interval_l282_282571

theorem tan_alpha_plus_pi_over_4_and_alpha_interval 
  (α : ℝ)
  (h1 : Real.tan (α + Real.pi / 4) = 1 / 2)
  (h2 : α ∈ Ioo (-Real.pi / 2) 0) 
  : (2 * Real.sin α ^ 2 + Real.sin (2 * α)) / Real.cos (α - Real.pi / 4) = -2 * Real.sqrt 5 / 5 := 
by 
  sorry

end tan_alpha_plus_pi_over_4_and_alpha_interval_l282_282571


namespace unpronounceable_words_count_l282_282057

-- Define the alphabet
def alphabet : Set Char := {'A', 'B'}

-- Define what it means for a word to be unpronounceable
def unpronounceable (word : List Char) : Prop :=
  ∃ (n : ℕ) (letter : Char), n < word.length - 2 
  ∧ letter ∈ alphabet 
  ∧ (word[n] = letter ∧ word[n+1] = letter ∧ word[n+2] = letter)

-- Define the length of the words we are considering
def word_length := 7

-- Define the set of all words of a given length
def all_words : List (List Char) :=
  (List.replicate 128 ['A', 'B']) -- List of all possible 7-letter words using 'A' and 'B'

-- Define the set of unpronounceable words
def unpronounceable_words : List (List Char) :=
  all_words.filter unpronounceable

-- Define the number of unpronounceable words
def count_unpronounceable_words : Nat :=
  unpronounceable_words.length

-- The theorem statement
theorem unpronounceable_words_count : count_unpronounceable_words = 86 :=
by sorry

end unpronounceable_words_count_l282_282057


namespace prob_xiao_li_chooses_technology_prob_xiao_li_and_xiao_ning_same_area_l282_282022

open Classical

variable (students_courses : Finset Char) (students_choices : Finset (Char × Char))

-- Five course areas A, B, C, D, E
def course_areas : Finset Char := {'A', 'B', 'C', 'D', 'E'}

-- 1. The probability that Xiao Li chooses the technology course area
theorem prob_xiao_li_chooses_technology : 1 / (course_areas.card : ℝ) = 1 / 5 := by
  sorry

-- 2. Probability that Xiao Li and Xiao Ning choose the same course area
theorem prob_xiao_li_and_xiao_ning_same_area : 
  (course_areas.filter (λ a, a ∈ students_choices.image (Prod.fst) ∧ a ∈ students_choices.image (Prod.snd))).card / 
  (course_areas.card * course_areas.card : ℝ) = 1 / 5 := by
  sorry

end prob_xiao_li_chooses_technology_prob_xiao_li_and_xiao_ning_same_area_l282_282022


namespace ellipse_minor_axis_length_l282_282370

noncomputable def minorAxisLength (a b c : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : c = (a^2 - b^2)^0.5) : ℝ :=
2 * b

theorem ellipse_minor_axis_length (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : ∀ P : ℝ × ℝ, (P.1^2 / a^2 + P.2^2 / b^2 = 1) → 
  ∃ (F : ℝ × ℝ), (∀ P, |(P.1 - F.1, P.2 - F.2)| ≤ 5 ∧ |(P.1 - F.1, P.2 - F.2)| ≥ 1))
  (h4 : c = (a^2 - b^2)^0.5) : minorAxisLength a b c h1 h2 h4 = 2 * (5:ℝ)^0.5 :=
sorry

end ellipse_minor_axis_length_l282_282370


namespace unique_exponential_function_l282_282102

def is_monotonic_increasing (f : ℝ → ℝ) : Prop :=
  ∀ a b : ℝ, a < b → f a < f b

theorem unique_exponential_function : ∃! (f : ℝ → ℝ), 
  (∀ x y : ℝ, f (x + y) = f x * f y) ∧ is_monotonic_increasing f :=
by {
  existsi (λ x : ℝ, 3 ^ x),
  split,
  { 
    intros x y,
    exact pow_add 3 x y,
  },
  { 
    intros a b h,
    rw [←rpow_nat_inv_iff],
    exact rpow_lt_rpow_of_exponent_lt (by norm_num) (by norm_num) (by assumption),
  }
}

end unique_exponential_function_l282_282102


namespace vector_dot_product_l282_282245

-- Definitions
def vec_a : ℝ × ℝ := (1, 3)
def vec_b : ℝ × ℝ := (-2, -1)

-- Theorem to prove
theorem vector_dot_product : 
  ((vec_a.1 + vec_b.1, vec_a.2 + vec_b.2) : ℝ × ℝ) • (2 * vec_a.1 + vec_b.1, 2 * vec_a.2 + vec_b.2) = 10 :=
by
  sorry

end vector_dot_product_l282_282245


namespace tau_eq_div_three_l282_282565

def tau (n : ℕ) : ℕ := {
  if n = 0 then 0
  else (List.range n).count (λ k, n % (k + 1) = 0)
}

theorem tau_eq_div_three (n : ℕ) (h : τ(n) = n / 3) : n = 9 ∨ n = 18 ∨ n = 24 := by
  sorry

end tau_eq_div_three_l282_282565


namespace problem1_problem2_l282_282943

-- (I)
variables {a b : ℝ^3}
variables ha : ‖a‖ = 1
variables hb : ‖b‖ = 1
variables h : ‖a - 2 • b‖ = 2

theorem problem1 : ‖a - b‖ = sqrt(3) / sqrt(2) := sorry

-- (II)
variables angle_ab : real.angle (a) (b) = real.angle.ofDegrees 60
variables m : ℝ^3 := a + b
variables n : ℝ^3 := a - 3 • b

theorem problem2 : 
  let cos_theta := (inner m n) / (‖m‖ * ‖n‖) in
  cos_theta = -sqrt(21) / 7 := sorry

end problem1_problem2_l282_282943


namespace units_digit_p_l282_282218

theorem units_digit_p (p : ℕ) (h1 : p % 2 = 0) (h2 : ((p ^ 3 % 10) - (p ^ 2 % 10)) % 10 = 0) 
(h3 : (p + 4) % 10 = 0) : p % 10 = 6 :=
sorry

end units_digit_p_l282_282218


namespace max_area_of_rectangle_l282_282635

theorem max_area_of_rectangle (L : ℝ) (hL : L = 16) :
  ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 8 → A = x * (8 - x)) ∧ A = 16 :=
by
  sorry

end max_area_of_rectangle_l282_282635


namespace evaluate_g_at_neg2_l282_282151

def g (x : ℝ) : ℝ := x^2 - 2*x + 3

theorem evaluate_g_at_neg2 : g (-2) = 11 := by
  sorry

end evaluate_g_at_neg2_l282_282151


namespace area_of_trapezium_l282_282020

/-- Two parallel sides of a trapezium are 4 cm and 5 cm respectively. 
    The perpendicular distance between the parallel sides is 6 cm.
    Prove that the area of the trapezium is 27 cm². -/
theorem area_of_trapezium (a b h : ℝ) (ha : a = 4) (hb : b = 5) (hh : h = 6) : 
  (1/2) * (a + b) * h = 27 := 
by 
  sorry

end area_of_trapezium_l282_282020


namespace mouse_jump_distance_l282_282749

theorem mouse_jump_distance
  (g f m : ℕ)
  (hg : g = 25)
  (hf : f = g + 32)
  (hm : m = f - 26) :
  m = 31 := by
  sorry

end mouse_jump_distance_l282_282749


namespace sum_fractional_powers_is_integer_l282_282566

theorem sum_fractional_powers_is_integer (n : ℕ) (hn : n > 1) 
    (a : Fin n → ℕ) (h_unique : Function.Injective a) 
    (k : ℕ) :
    ∃ m : ℕ, 
      m = ∑ i : Fin n, a i ^ k / ∏ j : Fin n, if j = i then 1 else a i - a j :=
begin
  sorry
end

end sum_fractional_powers_is_integer_l282_282566


namespace water_intake_proof_l282_282011

variable {quarts_per_bottle : ℕ} {bottles_per_day : ℕ} {extra_ounces_per_day : ℕ} 
variable {days_per_week : ℕ} {ounces_per_quart : ℕ} 

def total_weekly_water_intake 
    (quarts_per_bottle : ℕ) 
    (bottles_per_day : ℕ) 
    (extra_ounces_per_day : ℕ) 
    (ounces_per_quart : ℕ) 
    (days_per_week : ℕ) 
    (correct_answer : ℕ) : Prop :=
    (quarts_per_bottle * ounces_per_quart * bottles_per_day + extra_ounces_per_day) * days_per_week = correct_answer

theorem water_intake_proof : 
    total_weekly_water_intake 3 2 20 32 7 812 := 
by
    sorry

end water_intake_proof_l282_282011


namespace comparison_of_functions_on_interval_l282_282130

variables (a b : ℝ) (f g : ℝ → ℝ)
hypotheses 
  (h_diff_f : ∀ x ∈ Icc a b, differentiable_at ℝ f x)
  (h_diff_g : ∀ x ∈ Icc a b, differentiable_at ℝ g x)
  (h_f_prime_less_g_prime : ∀ x ∈ Icc a b, deriv f x < deriv g x)

theorem comparison_of_functions_on_interval :
  ∀ x, a < x ∧ x < b → f x + g a < g x + f a :=
by 
  assume x h,
  sorry

end comparison_of_functions_on_interval_l282_282130


namespace number_of_ways_to_assign_roles_l282_282937

theorem number_of_ways_to_assign_roles :
  let members := { Alice, Bob, Carol, Dave, Eliza : Type }
  let roles := { president, secretary, treasurer : Type }
  -- Number of ways to choose 3 members from 5 and assign them 3 distinct roles
  (choose 5 3 * 3!) = 60 := 
by 
  -- sorry is used to skip the proof because only the theorem statement is needed.
  sorry

end number_of_ways_to_assign_roles_l282_282937


namespace system_of_equations_solution_l282_282724

theorem system_of_equations_solution (x y z : ℕ) :
  x + y + z = 6 ∧ xy + yz + zx = 11 ∧ xyz = 6 ↔
  (x, y, z) = (1, 2, 3) ∨ (x, y, z) = (1, 3, 2) ∨ 
  (x, y, z) = (2, 1, 3) ∨ (x, y, z) = (2, 3, 1) ∨ 
  (x, y, z) = (3, 1, 2) ∨ (x, y, z) = (3, 2, 1) := by
  sorry

end system_of_equations_solution_l282_282724


namespace number_of_subsets_of_intersection_l282_282610

open Set

/-- A and B are defined as sets of integers -/
def A : Set ℕ := {1, 2, 3}

def B : Set ℕ := {1, 3, 4}

/-- Statement: the number of subsets of the intersection of A and B is 4 -/
theorem number_of_subsets_of_intersection :
  (A ∩ B).powerset.toFinset.card = 4 := by
  sorry

end number_of_subsets_of_intersection_l282_282610


namespace no_real_solution_l282_282201

def floor (t : ℝ) : ℤ := Int.floor t

theorem no_real_solution (x : ℝ) :
  (floor x + floor (2 * x) + floor (4 * x) + floor (8 * x) + floor (16 * x) + floor (32 * x) ≠ 12345) := 
  sorry

end no_real_solution_l282_282201


namespace period_of_sin_sub_cos_l282_282453

open Real

theorem period_of_sin_sub_cos :
  ∃ T > 0, ∀ x, sin x - cos x = sin (x + T) - cos (x + T) ∧ T = 2 * π := sorry

end period_of_sin_sub_cos_l282_282453


namespace simplify_expression_l282_282715

theorem simplify_expression : (8 * (15 / 9) * (-45 / 40) = -(1 / 15)) :=
by
  sorry

end simplify_expression_l282_282715


namespace number_of_ways_to_assign_students_l282_282324

-- Definitions of combinations
noncomputable def combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Number of male students
def male_students := 30

-- Number of students to be selected
def students_to_select := 10

-- Number of ways to select 10 students from 30
def selection_ways := combination male_students students_to_select

-- Number of ways to divide 10 students into two indistinguishable groups of 5 each
def division_ways := combination students_to_select 5 / 2

-- Total number of ways to assign the students
def total_ways := selection_ways * division_ways

-- Lean statement for the problem
theorem number_of_ways_to_assign_students : total_ways = (combination 30 10 * combination 10 5) / 2 :=
by
  sorry

end number_of_ways_to_assign_students_l282_282324


namespace locus_of_vertex_C_l282_282094

theorem locus_of_vertex_C 
  (A B C D : ℝ) 
  (AB_length : A = 4)
  (AD_length : D = 3)
  (AD_angle : ∠ (create_vector A B) (create_vector A D) = 30) : 
  ∃ (center : ℝ × ℝ) (radius : ℝ), 
  center = ( (3 * Real.sqrt 3) / 2, 3 / 2 ) 
  ∧ radius = 2 
  ∧ (∃ C_point : ℝ × ℝ, 
        distance C_point center = 2) := 
sorry

end locus_of_vertex_C_l282_282094


namespace smallest_square_area_is_121_l282_282327

-- Given conditions
def rectangle1 := (2, 3)
def rectangle2 := (3, 4)
def rectangle3 := (1, 4)

-- Function to calculate the smallest possible area of containing square
noncomputable def smallest_possible_square_area (rect1 rect2 rect3 : (ℕ × ℕ)) : ℕ := 
  let largest_side (rect : (ℕ × ℕ)) : ℕ := rect.1.max rect.2
  let total_length := largest_side rect1 + largest_side rect2 + largest_side rect3
  let side_length := total_length.max (rect1.1 + rect2.1).max (rect1.2 + rect2.2) in
  side_length * side_length

-- Proof problem statement
theorem smallest_square_area_is_121 : 
  smallest_possible_square_area rectangle1 rectangle2 rectangle3 = 121 := 
by sorry

end smallest_square_area_is_121_l282_282327


namespace tangency_conditions_l282_282360

variables {A B C D E : Type} [InscribedQuadrilateral A B C D] (mid_E : Midpoint E B D)
variables (Γ1 Γ2 Γ3 Γ4 : Circumcircle)
variables [CircleTangentToLine Γ4 D E A C D]

theorem tangency_conditions
  (h1 : Circumcircle A E B Γ1)
  (h2 : Circumcircle B E C Γ2)
  (h3 : Circumcircle C E D Γ3)
  (h4 : Circumcircle D E A Γ4)
  (h_tangent : TangentToLine Γ4 C D) :
  TangentToLine Γ1 B C ∧ TangentToLine Γ2 A B ∧ TangentToLine Γ3 A D :=
by
  sorry

end tangency_conditions_l282_282360


namespace coordinates_of_point_P_x_axis_coordinates_of_point_P_distance_y_axis_l282_282938

def point_P (a : ℝ) : ℝ × ℝ := (2 * a - 1, a + 3)

theorem coordinates_of_point_P_x_axis (a : ℝ) (h : a + 3 = 0) : point_P a = (-7, 0) :=
  by
    have ha : a = -3 := by linarith
    rw [ha]
    sorry

theorem coordinates_of_point_P_distance_y_axis (a : ℝ) (h : |2 * a - 1| = 5) :
  point_P a = (-5, 1) ∨ point_P a = (5, 6) :=
  by
    cases abs_sub_eq_iff (2 * a - 1) 5 with ha hb
    all_goals
      have h1 := ha
      have h2 := hb
      -- a = 3 or a = -2
      sorry

end coordinates_of_point_P_x_axis_coordinates_of_point_P_distance_y_axis_l282_282938


namespace full_time_employees_l282_282110

theorem full_time_employees (total_employees worked_year_part non_full_time_no_year full_time_year: ℕ) 
    (h1: total_employees = 130) 
    (h2: worked_year_part = 100) 
    (h3: non_full_time_no_year = 20) 
    (h4: full_time_year = 30) : 
  let full_time_count := total_employees - (worked_year_part - full_time_year + non_full_time_no_year) in
  full_time_count = 40 :=
by 
  unfold full_time_count
  rw [h1, h2, h3, h4]
  sorry  -- Proof only needs to check the statement's validity without actual proof steps

end full_time_employees_l282_282110


namespace solve_equation_l282_282545

theorem solve_equation (x : ℝ) :
  (1 / ((x - 2) * (x - 3)) + 1 / ((x - 3) * (x - 4)) + 1 / ((x - 4) * (x - 5)) = 1 / 8) ↔ (x = 13 ∨ x = -2) :=
by
  sorry

end solve_equation_l282_282545


namespace max_point_same_l282_282553

theorem max_point_same (A B C P Q : Point) (triangle : Triangle A B C) :
  (∀ R : Point, R ∈ triangle → PA + PB + PC ≤ RA + RB + RC)
  → (∀ S : Point, S ∈ triangle → QA^2 + QB^2 + QC^2 ≤ SA^2 + SB^2 + SC^2)
  → P = Q :=
begin
  sorry
end

end max_point_same_l282_282553


namespace sum_of_ages_l282_282099

theorem sum_of_ages (A B C : ℕ)
  (h1 : A = C + 8)
  (h2 : A + 10 = 3 * (C - 6))
  (h3 : B = 2 * C) :
  A + B + C = 80 := 
by 
  sorry

end sum_of_ages_l282_282099


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282283

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282283


namespace find_diameter_of_field_l282_282909

noncomputable def cost_per_meter : ℝ := 1.50
noncomputable def total_cost : ℝ := 131.95
noncomputable def circumference : ℝ := total_cost / cost_per_meter
noncomputable def diameter : ℝ := circumference / Real.pi

theorem find_diameter_of_field : diameter ≈ 28 :=
by
  sorry

end find_diameter_of_field_l282_282909


namespace sum_prime_factors_of_2_pow_8_minus_1_l282_282769

theorem sum_prime_factors_of_2_pow_8_minus_1 : 
  ∃ (primes : List ℕ), primes.PrimeFactors 255 ∧ primes.Sum = 25 :=
by
  sorry

end sum_prime_factors_of_2_pow_8_minus_1_l282_282769


namespace min_ω_value_l282_282410

def min_ω (ω : Real) : Prop :=
  ω > 0 ∧ (∃ k : Int, ω = 2 * k + 2 / 3)

theorem min_ω_value : ∃ ω : Real, min_ω ω ∧ ω = 2 / 3 := by
  sorry

end min_ω_value_l282_282410


namespace variance_of_data_l282_282846

theorem variance_of_data : 
  let data := [10, 6, 8, 5, 6]
  let n := data.length
  let mean := (data.sum / n)
  let variance := (data.map (λ x => (x - mean)^2)).sum / n
  in variance = 3.2 :=
by
  -- Calculation parts will be proven here
  sorry

end variance_of_data_l282_282846


namespace range_of_s_l282_282792

def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem range_of_s :
  set.range (λ x, s x) = set.Ioo (−∞) 0 ∪ set.Ioo 0 ∞ :=
sorry

end range_of_s_l282_282792


namespace ticket_cost_at_30_years_l282_282041

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end ticket_cost_at_30_years_l282_282041


namespace find_x_values_l282_282164

theorem find_x_values (x : ℝ) (h : x > 9) : 
  (sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) ↔ (x ≥ 18) :=
sorry

end find_x_values_l282_282164


namespace sin_rotated_angle_l282_282227

theorem sin_rotated_angle (α β : ℝ) 
  (h_cos_alpha : cos α = -1 / 3)
  (h_alpha_quadrant : π / 2 < α ∧ α < π)
  (h_beta_def : β = α + π) :
  sin β = - (2 * Real.sqrt 2) / 3 :=
by
  -- Proof to be filled here
  sorry

end sin_rotated_angle_l282_282227


namespace solve_for_y_l282_282718

noncomputable def f (y : ℝ) := real.cbrt (30 * y + real.cbrt (30 * y + real.cbrt (30 * y + 14)))

theorem solve_for_y : ∃ y : ℝ, f(y) = 14 ∧ y = 91 := by
  sorry

end solve_for_y_l282_282718


namespace total_stairs_climbed_l282_282673

-- Definitions of the conditions
def jonny_stairs := 4872
def julia_stairs := Int.round (2 * Real.sqrt (jonny_stairs / 2) + 15)

-- Lean statement to prove the total stairs climbed
theorem total_stairs_climbed : jonny_stairs + julia_stairs = 4986 := by
  sorry

end total_stairs_climbed_l282_282673


namespace sum_of_first_eight_terms_l282_282194

-- Define the first term, common ratio, and the number of terms
def a : ℚ := 1 / 3
def r : ℚ := 1 / 3
def n : ℕ := 8

-- Sum of the first n terms of a geometric sequence
def geometric_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

-- Proof statement
theorem sum_of_first_eight_terms : geometric_sum a r n = 3280 / 6561 :=
by
  sorry

end sum_of_first_eight_terms_l282_282194


namespace area_of_triangle_ABC_l282_282823

theorem area_of_triangle_ABC :
  ∀ (A B C D E : Type) (r R : ℝ),
  r = 2 →
  R = 4 →
  (∃ (ABC : Triangle) (inscribed : Circle) (circumscribed : Circle),
    inscribed.radius = r ∧
    circumscribed.radius = R ∧
    inscribed.tangent_to_side ABC BC D ∧
    circumscribed.tangent_to_extensions ABC AB AC E ∧
    circumscribed.tangent_to_side ABC BC E ∧
    ABC.angle_C = 120) →
  ABC.area = 56 / sqrt 3 := 
by {
  intros,
  sorry
}

end area_of_triangle_ABC_l282_282823


namespace Alyssa_spent_in_total_l282_282855

def amount_paid_for_grapes : ℝ := 12.08
def refund_for_cherries : ℝ := 9.85
def total_spent : ℝ := amount_paid_for_grapes - refund_for_cherries

theorem Alyssa_spent_in_total : total_spent = 2.23 := by
  sorry

end Alyssa_spent_in_total_l282_282855


namespace matt_age_proof_l282_282064

def james_age_3_years_ago : ℕ := 27
def years_passed : ℕ := 3
def additional_years : ℕ := 5

def james_current_age : ℕ := james_age_3_years_ago + years_passed
def james_age_in_5_years : ℕ := james_current_age + additional_years
def matt_age_in_5_years : ℕ := 2 * james_age_in_5_years
def matt_current_age : ℕ := matt_age_in_5_years - additional_years

theorem matt_age_proof : matt_current_age = 65 := by
  unfold james_age_3_years_ago years_passed additional_years james_current_age james_age_in_5_years matt_age_in_5_years matt_current_age
  norm_num
  sorry

end matt_age_proof_l282_282064


namespace matrix_invertibility_solution_l282_282177

def matrix_invertibility_problem : Prop :=
  let M := ![![4, -2], ![8, -4]]
  ∃ (inv : Matrix (Fin 2) (Fin 2) ℝ), M.det = 0 → inv = (0 : Matrix (Fin 2) (Fin 2) ℝ)

theorem matrix_invertibility_solution : matrix_invertibility_problem :=
by
  let M := ![![4, -2], ![8, -4]]
  have h_det : M.det = 0 := by
    -- Calculating determinant
    simp
  let zero_matrix := (0 : Matrix (Fin 2) (Fin 2) ℝ)
  -- Proving the zero matrix is the inverse when the determinant is 0.
  use zero_matrix
  intro h
  rw h
  sorry

end matrix_invertibility_solution_l282_282177


namespace solve_for_x_l282_282166

theorem solve_for_x (x : ℝ) (hx : x > 9) 
  (h : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 18 := 
  sorry

end solve_for_x_l282_282166


namespace exists_in_B_or_C_l282_282244

-- Definitions based on the conditions
def sequence_A : Set ℕ := { m | ∃ k : ℕ, k ≥ 1 ∧ m = 10^k }

def binary_length (k : ℕ) : ℕ := (Real.log 10 / Real.log 2).to_nat * k + 1

def quinary_length (k : ℕ) : ℕ := (Real.log 10 / Real.log 5).to_nat * k + 1

def sequence_B : Set ℕ := { n | ∃ k : ℕ, k ≥ 1 ∧ n = binary_length k }

def sequence_C : Set ℕ := { n | ∃ k : ℕ, k ≥ 1 ∧ n = quinary_length k }

-- Main theorem to prove
theorem exists_in_B_or_C (n : ℕ) (h : n > 1) : n ∈ sequence_B ∨ n ∈ sequence_C :=
  sorry

end exists_in_B_or_C_l282_282244


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282259

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282259


namespace area_of_square_inscribed_in_circle_l282_282483

-- Definitions based on conditions
def circle_eq (x y : ℝ) : Prop := 2 * x^2 = -2 * y^2 + 24 * x + 8 * y + 36
def is_inscribed (circle center radius : ℝ → ℝ → Prop) (square : ℝ) : Prop := ∀ x y, circle x y → square = (2 * radius)^2

-- Theorem to prove
theorem area_of_square_inscribed_in_circle {x y : ℝ} (h : circle_eq x y) :
  ∃ r center, (circle_eq (x+6) (y-2)) → ((2 * r)^2 = 56) :=
sorry

end area_of_square_inscribed_in_circle_l282_282483


namespace wire_problem_l282_282095

theorem wire_problem (a b : ℝ) (h_perimeter : a = b) : a / b = 1 := by
  sorry

end wire_problem_l282_282095


namespace cricket_bat_selling_price_l282_282076

theorem cricket_bat_selling_price 
  (profit : ℝ)
  (profit_percentage : ℝ)
  (selling_price : ℝ) 
  (h_profit : profit = 205) 
  (h_percentage : profit_percentage = 31.782945736434108) 
  (h_selling_price : selling_price = 850) : 
  selling_price = (205 / (31.782945736434108 / 100)) + 205 :=
by {
  rw [h_profit, h_percentage],
  sorry,
}

end cricket_bat_selling_price_l282_282076


namespace plums_total_correct_l282_282386

-- Define the number of plums picked by Melanie, Dan, and Sally
def plums_melanie : ℕ := 4
def plums_dan : ℕ := 9
def plums_sally : ℕ := 3

-- Define the total number of plums picked
def total_plums : ℕ := plums_melanie + plums_dan + plums_sally

-- Theorem stating the total number of plums picked
theorem plums_total_correct : total_plums = 16 := by
  sorry

end plums_total_correct_l282_282386


namespace number_of_selections_l282_282712

-- Define the given conditions
def num_colors : ℕ := 6
def num_select : ℕ := 2

-- Formalize the problem in Lean
theorem number_of_selections : Nat.choose num_colors num_select = 15 := by
  -- Proof omitted
  sorry

end number_of_selections_l282_282712


namespace problem_area_calculation_l282_282949

noncomputable def side_opposite_angle (angle : ℝ) := sorry
noncomputable def triangle_area (a b c A : ℝ) := 1 / 2 * b * c * real.sin A

theorem problem :
  ∀ (a b c A B C : ℝ),
    a = side_opposite_angle A ∧
    b = side_opposite_angle B ∧
    c = side_opposite_angle C ∧
    b * real.sin A = 2 * a * real.sin B →
    A = real.pi / 3 :=
by
  sorry

theorem area_calculation :
  ∀ (a b c A : ℝ),
    a = real.sqrt 7 ∧
    2 * b - c = 4 ∧
    A = real.pi / 3 →
    triangle_area a b c A = 3 * real.sqrt 3 / 2 :=
by
  sorry

end problem_area_calculation_l282_282949


namespace max_neg_int_prod_neg_any_zero_prod_zero_l282_282129

noncomputable def seq : ℕ := 10

def is_negative_product (s : List ℤ) : Prop :=
  s.product < 0

def is_zero_product (s : List ℤ) : Prop :=
  s.product = 0

def max_negative_count (s : List ℤ) : ℕ :=
  s.filter (λ x => x < 0).length

theorem max_neg_int_prod_neg (s : List ℤ) (h1 : s.length = seq) (h2 : is_negative_product s) :
  max_negative_count s ≤ 9 := 
sorry

theorem any_zero_prod_zero (s : List ℤ) (h1 : s.length = seq) (h2 : ∃ x ∈ s, x = 0) :
  is_zero_product s :=
sorry

end max_neg_int_prod_neg_any_zero_prod_zero_l282_282129


namespace triangle_perimeter_l282_282552

def Point := ℝ × ℝ

noncomputable def distance (p1 p2: Point) : ℝ :=
  real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

noncomputable def perimeter (A B C: Point) : ℝ :=
  distance A B + distance B C + distance C A

theorem triangle_perimeter :
  let A := (1, 2) : Point
  let B := (1, 8) : Point
  let C := (5, 5) : Point
  perimeter A B C = 16 :=
by
  let A := (1, 2) : Point
  let B := (1, 8) : Point
  let C := (5, 5) : Point
  sorry

end triangle_perimeter_l282_282552


namespace problem_solution_l282_282142

noncomputable def solve_problem : Prop :=
  ∃ (d : ℝ), 
    (∃ int_part : ℤ, 
        (3 * int_part^2 - 12 * int_part + 9 = 0 ∧ ⌊d⌋ = int_part) ∧
        ∀ frac_part : ℝ,
            (4 * frac_part^3 - 8 * frac_part^2 + 3 * frac_part - 0.5 = 0 ∧ frac_part = d - ⌊d⌋) )
    ∧ (d = 1.375 ∨ d = 3.375)

theorem problem_solution : solve_problem :=
by sorry

end problem_solution_l282_282142


namespace quadratic_real_solutions_l282_282058

theorem quadratic_real_solutions (x y : ℝ) :
  (∃ z : ℝ, 16 * z^2 + 4 * x * y * z + (y^2 - 3) = 0) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

end quadratic_real_solutions_l282_282058


namespace smallest_lambda_complex_l282_282917

theorem smallest_lambda_complex :
  ∃ λ : ℝ, λ = 1 ∧ ∀ z1 z2 z3 : ℂ, |z1| < 1 → |z2| < 1 → |z3| < 1 → z1 + z2 + z3 = 0 →
    |z1 * z2 + z2 * z3 + z3 * z1|^2 + |z1 * z2 * z3|^2 < λ :=
by
  use 1
  split
  · rfl
  · intros z1 z2 z3 hz1 hz2 hz3 hsum
    sorry

end smallest_lambda_complex_l282_282917


namespace f_is_odd_compare_f_l282_282236

def f (x : ℝ) : ℝ := x + 1 / x

-- Part 1: Prove that f is an odd function
theorem f_is_odd (x : ℝ) (hx : x ≠ 0) : f (-x) = -f (x) := by
  sorry

-- Part 2: Prove that f(a) > f(b) for a > b > 1
theorem compare_f (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : a > b) : f(a) > f(b) := by
  sorry

end f_is_odd_compare_f_l282_282236


namespace a_is_one_l282_282596

def curve (a : ℝ) (x : ℝ) : ℝ := x^3 - 2 * a * x^2 + 2 * a * x

def slope (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 4 * a * x + 2 * a

theorem a_is_one (a : ℤ) 
  (h : ∀ x : ℝ, slope a x > 0) :
  a = 1 :=
sorry

end a_is_one_l282_282596


namespace angles_same_terminal_side_30_l282_282101

theorem angles_same_terminal_side_30 (k : ℤ) :
  (∃ k : ℤ, -30 = 30 + k * 360) ∧ (∃ k : ℤ, 390 = 30 + k * 360) :=
by
  split;
  { use k,
    sorry
  }

end angles_same_terminal_side_30_l282_282101


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282276

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282276


namespace number_of_correct_statements_is_zero_l282_282746

variable {a b : ℝ}
variable (f : ℝ → ℝ)
variable (x : ℝ)

/- Define propositions based on the conditions -/
def prop1 : Prop := ∀ x ∈ set.Icc a b, f x = 0 → (x, 0) = (x, f x)
def prop2 : Prop := ∀ x ∈ set.Icc a b, f x = 0 → continuous_on f (set.Icc a b) → (∃ x₀, x₀ ∈ set.Icc a b ∧ f x₀ = 0)
def prop3 : Prop := ∀ x ∈ set.Icc a b, (f x = 0 ↔ x ∈ { y | f y = 0 }) ∧ ((f x = 0 → x ∈ { y | f y = 0 }) ∧ (x ∈ { y | f y = 0 } → f x = 0))
def prop4 : Prop := ∀ x₁ x₂ ∈ set.Icc a b, (x₂ - x₁).abs < ε → (∃ x₀, x₀ ∈ set.Icc a b ∧ f x₀ = 0 ∧ (f x₀ = x₀ ∨ ∃ n, (f^[n] x₀) = x₀))

/-- The main proposition that states the correct number of true statements -/
theorem number_of_correct_statements_is_zero :
  ¬ prop1 ∧ ¬ prop2 ∧ ¬ prop3 ∧ ¬ prop4 := 
by
  sorry

end number_of_correct_statements_is_zero_l282_282746


namespace speed_of_first_half_l282_282084

-- Define variables for the different parts of the journey
variable (total_distance : ℝ) (total_time : ℝ) (second_half_speed : ℝ)
variable (first_half_distance second_half_distance : ℝ)

-- Define the conditions
def conditions : Prop :=
  total_distance = 225 ∧
  total_time = 10 ∧
  second_half_speed = 24 ∧
  first_half_distance = total_distance / 2 ∧
  second_half_distance = total_distance / 2

-- Define the first_half_speed which we need to prove
def first_half_speed (first_half_time : ℝ) : ℝ :=
  first_half_distance / first_half_time

-- Final statement to be proved
theorem speed_of_first_half:
  ∀ (total_distance total_time second_half_speed : ℝ)
    (first_half_distance second_half_distance : ℝ)
    (first_half_time second_half_time : ℝ),
  conditions total_distance total_time second_half_speed first_half_distance second_half_distance →
  second_half_time = second_half_distance / second_half_speed →
  first_half_time = total_time - second_half_time →
  first_half_speed first_half_distance first_half_time ≈ 21.18 := 
by
  intros total_distance total_time second_half_speed first_half_distance second_half_distance first_half_time second_half_time
  intro h_cond
  intro h_second_half_time
  intro h_first_half_time
  apply real.eq_of_abs_sub_lt_all_pos
  use 0.01 -- Let's assume we're okay with an approximation error of 0.01
  intro ε_pos
  -- proof skipped
  sorry


end speed_of_first_half_l282_282084


namespace right_angle_triangle_sides_l282_282503

theorem right_angle_triangle_sides (a b c : Nat) : 
  (a = 3 ∧ b = 4 ∧ c = 6 ∨ 
   a = 7 ∧ b = 24 ∧ c = 25 ∨ 
   a = 6 ∧ b = 8 ∧ c = 10 ∨ 
   a = 9 ∧ b = 12 ∧ c = 15) → 
  (¬(a = 3 ∧ b = 4 ∧ c = 6) ↔ a * a + b * b = c * c) → 
  (a, b, c) = (3, 4, 6) :=
begin
  intros,
  cases H;
  sorry
end

end right_angle_triangle_sides_l282_282503


namespace total_cookies_in_box_l282_282342

-- Definitions from the conditions
def oldest_son_cookies : ℕ := 4
def youngest_son_cookies : ℕ := 2
def days_box_lasts : ℕ := 9

-- Total cookies consumed per day
def daily_cookies_consumption : ℕ := oldest_son_cookies + youngest_son_cookies

-- Theorem statement: total number of cookies in the box
theorem total_cookies_in_box : (daily_cookies_consumption * days_box_lasts) = 54 := by
  sorry

end total_cookies_in_box_l282_282342


namespace mutually_perpendicular_chords_l282_282019

variable {Point : Type} [AddGroup Point] [Module ℝ Point]

noncomputable def vector_sum (O A B C D M : Point) : Prop :=
  (\overrightarrow{OA} + \overrightarrow{OB} + \overrightarrow{OC} + \overrightarrow{OD} = 2 * \overrightarrow{OM})

theorem mutually_perpendicular_chords (O A B C D M : Point)
  (intersect_at_M : Point)
  (mut_perpendicular: AB ⟂ CD)
  (cond_center : ∃ O, is_center O)
  (intersect_cond : A ≠ B ∧ C ≠ D ∧ AB ∩ CD = M) : vector_sum O A B C D M := 
sorry

end mutually_perpendicular_chords_l282_282019


namespace max_students_l282_282646

-- Define the set of days in September
def DaysInSeptember : Finset ℕ := Finset.range 30

-- Define a student visiting pool subset property
def student_visits (s : Finset ℕ) (H : ∀ x ∈ s, x < 30) := s ⊆ DaysInSeptember

-- The main theorem: maximum number of students is 28
theorem max_students (m : ℕ) (students_visits : Fin m (Finset ℕ))
  (H : ∀ i ≠ j, ∀ d ∈ students_visits i, d ∉ students_visits j ∧ ∀ d' ∈ students_visits j, d' ∉ students_visits i)
  (H_unique : ∀ i, ∃ n, (students_visits i).card = n ∧ ∀ j ≠ i, (students_visits j).card ≠ n) :
  m ≤ 28 :=
  sorry

end max_students_l282_282646


namespace prob_master_degree_prob_distribution_X_expected_value_X_education_relation_l282_282852

section CityZ
-- Define the proportions of education levels
def education_levels (gender : String) (level : String) : Float :=
  match gender, level with
  | "Male", "No Schooling"          => 0.00
  | "Male", "Primary School"        => 0.03
  | "Male", "Junior High School"    => 0.14
  | "Male", "High School"           => 0.11
  | "Male", "College (Associate)"   => 0.07
  | "Male", "College (Bachelor)"    => 0.11
  | "Male", "Master's Degree"       => 0.03
  | "Male", "Doctoral Degree"       => 0.01
  | "Female", "No Schooling"        => 0.01
  | "Female", "Primary School"      => 0.04
  | "Female", "Junior High School"  => 0.11
  | "Female", "High School"         => 0.11
  | "Female", "College (Associate)" => 0.08
  | "Female", "College (Bachelor)"  => 0.12
  | "Female", "Master's Degree"     => 0.03
  | "Female", "Doctoral Degree"     => 0.00
  | "Total", "No Schooling"         => 0.01
  | "Total", "Primary School"       => 0.07
  | "Total", "Junior High School"   => 0.25
  | "Total", "High School"          => 0.22
  | "Total", "College (Associate)"  => 0.15
  | "Total", "College (Bachelor)"   => 0.23
  | "Total", "Master's Degree"      => 0.06
  | "Total", "Doctoral Degree"      => 0.01
  | _, _                             => 0.0

-- Define the proportion of residents aged 15 and above
def proportion_aged_15_and_above : Float := 0.85

-- Proving the first part (Ⅰ)
theorem prob_master_degree :
  (proportion_aged_15_and_above * education_levels "Total" "Master's Degree" = 0.051) := by
  sorry

-- Proportions of Bachelor's degree or higher
def prop_bachelor_or_higher : Float := (education_levels "Total" "College (Bachelor)") + (education_levels "Total" "Master's Degree") + (education_levels "Total" "Doctoral Degree")

-- Individual probabilities
def prob_X_0 : Float := (1 - prop_bachelor_or_higher) ^ 2
def prob_X_1 : Float := 2 * prop_bachelor_or_higher * (1 - prop_bachelor_or_higher)
def prob_X_2 : Float := prop_bachelor_or_higher ^ 2

-- Proving the second part (Ⅱ)
theorem prob_distribution_X :
  (prob_X_0 = 0.49) ∧ (prob_X_1 = 0.42) ∧ (prob_X_2 = 0.09) := by
  sorry

-- Expected value of X
def expectation_X : Float := 0 * prob_X_0 + 1 * prob_X_1 + 2 * prob_X_2

-- Proving expected value of X
theorem expected_value_X :
  (expectation_X = 0.6) := by
  sorry

-- Average years of education levels
def years_of_education (level : String) : Nat :=
  match level with
  | "No Schooling"          => 0
  | "Primary School"        => 6
  | "Junior High School"    => 9
  | "High School"           => 12
  | _                       => 16

def avg_years (gender : String) : Float :=
  List.foldl (λ acc level, acc + education_levels gender level * (years_of_education level)) 0.0
    ["No Schooling", "Primary School", "Junior High School", "High School", "College (Associate)"]

-- Variables a and b
def a : Float := avg_years "Male"
def b : Float := avg_years "Female"

-- Proving the third part (Ⅲ)
theorem education_relation :
  (a > b) := by
  sorry

end CityZ

end prob_master_degree_prob_distribution_X_expected_value_X_education_relation_l282_282852


namespace max_price_per_sock_l282_282853

theorem max_price_per_sock
  (total_money : ℤ)
  (entrance_fee : ℤ)
  (sales_tax_rate : ℚ)
  (number_of_socks : ℕ)
  (sock_price : ℕ):
  total_money = 180 →
  entrance_fee = 3 →
  sales_tax_rate = 0.06 →
  number_of_socks = 20 →
  let money_after_fee := total_money - entrance_fee in
  let total_expenditure := ceil ((money_after_fee : ℚ) / (1 + sales_tax_rate)) in
  let price_per_sock := (total_expenditure / number_of_socks) in
  price_per_sock.floor = 8 :=
by intros
   sorry

end max_price_per_sock_l282_282853


namespace area_ratio_inequality_l282_282013

variable (A B C A1 B1 C1 : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace A1] [MetricSpace B1] [MetricSpace C1]

variable (triangle_ABC : Triangle A B C)
variable (triangle_A1B1C1 : Triangle A1 B1 C1)

variable (B1_on_BC : B1 ∈ Segment B C)
variable (C1_on_BC : C1 ∈ Segment B C)
variable (A1_inside_ABC : A1 ∈ Interior (Triangle A B C))

variable (S S1 : ℝ)
variable (S_eq_area_ABC : S = area (Triangle A B C))
variable (S1_eq_area_A1B1C1 : S1 = area (Triangle A1 B1 C1))

theorem area_ratio_inequality :
  let AB := dist A B
  let AC := dist A C
  let A1B1 := dist A1 B1
  let A1C1 := dist A1 C1
  S / (AB + AC) > S1 / (A1B1 + A1C1) :=
sorry

end area_ratio_inequality_l282_282013


namespace coefficient_of_1_over_x_squared_in_expansion_l282_282987

theorem coefficient_of_1_over_x_squared_in_expansion :
  let n := 8
  let x := polynomial.Ring x ℝ
  let binom (a : ℕ) (b : ℕ) : ℕ := nat.choose a b
  let coeff := polynomial.coeff
  (∀ k, binom n 2 = binom n 6) →
  coeff x (2 * (8 - 2) - 2) (polynomial.expand x (binom n 5)) = 56 :=
by
  sorry

end coefficient_of_1_over_x_squared_in_expansion_l282_282987


namespace sum_of_prime_factors_of_sum_2006_to_2036_l282_282767

noncomputable def sum_of_integers (a b : ℕ) : ℕ :=
  (b - a + 1) * (a + b) / 2

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

noncomputable def prime_factors (n : ℕ) : List ℕ :=
  (List.range (n+1)).filter (λ p, is_prime p ∧ p ∣ n)

noncomputable def sum_of_prime_factors (n : ℕ) : ℕ :=
  (prime_factors n).sum

theorem sum_of_prime_factors_of_sum_2006_to_2036 :
  sum_of_prime_factors (sum_of_integers 2006 2036) = 121 :=
sorry

end sum_of_prime_factors_of_sum_2006_to_2036_l282_282767


namespace solve_floor_equation_l282_282542

theorem solve_floor_equation (x : ℝ) :
  (⌊⌊2 * x⌋ - 1 / 2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
sorry

end solve_floor_equation_l282_282542


namespace quadrilateral_ABCD_perimeter_l282_282656

noncomputable def perimeter_ABCD (AB BC CD : ℝ) (BD : ℝ) (AD : ℝ) : ℝ := AB + BC + CD + AD

theorem quadrilateral_ABCD_perimeter :
  let A B C D : Point := Point.mk 0 0 -- arbitrary points for sake of defining
  let AB : ℝ := 15
  let BC : ℝ := 20
  let CD : ℝ := 9
  let BD : ℝ := Real.sqrt (BC^2 + CD^2)
  let AD : ℝ := Real.sqrt (AB^2 + BD^2)
  perimeter_ABCD AB BC CD AD = 44 + Real.sqrt 706 := by
  sorry

end quadrilateral_ABCD_perimeter_l282_282656


namespace sum_of_two_integers_l282_282417

theorem sum_of_two_integers (x y : ℕ) 
  (h1 : x * y + x + y = 137) 
  (h2 : Nat.coprime x y) 
  (h3x : x < 30) 
  (h3y : y < 30) : 
  x + y = 27 :=
sorry

end sum_of_two_integers_l282_282417


namespace max_consecutive_odd_prime_exponents_is_seven_l282_282059

def has_all_odd_exponents (n : ℕ) : Prop :=
  ∀ p : ℕ, (p.prime → ∃ k : ℕ, n = p ^ (2 * k + 1))

theorem max_consecutive_odd_prime_exponents_is_seven :
  ∃ (s : Finset ℕ), (s.card = 7) ∧ (∀ n ∈ s, has_all_odd_exponents n) ∧ 
  ∀ t : Finset ℕ, (∀ n ∈ t, has_all_odd_exponents n) → t.card ≤ 7 :=
sorry

end max_consecutive_odd_prime_exponents_is_seven_l282_282059


namespace min_value_cos_sin_l282_282550

noncomputable def min_value_expression : ℝ :=
  -1 / 2

theorem min_value_cos_sin (θ : ℝ) (hθ1 : 0 ≤ θ) (hθ2 : θ ≤ 3 * Real.pi / 2) :
  ∃ (y : ℝ), y = Real.cos (θ / 3) * (1 - Real.sin θ) ∧ y = min_value_expression :=
sorry

end min_value_cos_sin_l282_282550


namespace negation_of_universal_proposition_l282_282241

theorem negation_of_universal_proposition (p : ∀ x : ℝ, 2 ^ x > 0) : 
  ¬ (∀ x : ℝ, 2 ^ x > 0) = ∃ x : ℝ, 2 ^ x ≤ 0 :=
sorry

end negation_of_universal_proposition_l282_282241


namespace no_perfect_squares_in_seq_l282_282574

def seq (x : ℕ → ℤ) : Prop :=
  x 0 = 1 ∧ x 1 = 3 ∧ ∀ n : ℕ, 0 < n → x (n + 1) = 6 * x n - x (n - 1)

theorem no_perfect_squares_in_seq (x : ℕ → ℤ) (n : ℕ) (h_seq : seq x) :
  ¬ ∃ k : ℤ, k * k = x (n + 1) :=
by
  sorry

end no_perfect_squares_in_seq_l282_282574


namespace total_vegetables_l282_282696

theorem total_vegetables 
  (x y z g : ℕ)
  (h1 : x = 5 / 3 * y)
  (h2 : z = 2 * (0.5 * y))
  (h3 : g = 0.5 * (x / 4) - 3)
  (kristin_bell_peppers : 2 = 2)
  (kristin_green_beans : 20 = x / 4)
  (hx : x = 80)
  (hy : y = 48)
  (hz : z = 48)
  (hg : g = 7) :
  x + y + z + g = 183 :=
by sorry

end total_vegetables_l282_282696


namespace sqrt_domain_l282_282316

theorem sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_domain_l282_282316


namespace complex_argument_l282_282523

noncomputable def complex_number := -1 + Complex.i * Real.sqrt 3
noncomputable def modulus := Complex.abs complex_number
noncomputable def argument := Complex.arg complex_number

theorem complex_argument :
  argument = 2 * Real.pi / 3 := by
  sorry

end complex_argument_l282_282523


namespace moles_of_KHSO4_formed_l282_282252

-- Chemical reaction definition
def reaction (n_KOH n_H2SO4 : ℕ) : ℕ :=
  if n_KOH = n_H2SO4 then n_KOH else 0

-- Given conditions
def moles_KOH : ℕ := 2
def moles_H2SO4 : ℕ := 2

-- Proof statement to be proved
theorem moles_of_KHSO4_formed : reaction moles_KOH moles_H2SO4 = 2 :=
by sorry

end moles_of_KHSO4_formed_l282_282252


namespace keno_probability_no_digit_8_l282_282456

theorem keno_probability_no_digit_8 :
  let total_slots := 80
  let draw_slots := 20
  let numbers_with_digit_8 := {8, 18, 28, 38, 48, 58, 68, 78}
  let total_numbers_not_digit_8 := 71
  let total_combinations := Nat.choose total_slots draw_slots
  let favorable_combinations := Nat.choose total_numbers_not_digit_8 draw_slots
  let probability : ℝ := favorable_combinations / total_combinations
  probability ≈ 0.063748 := by
  sorry

end keno_probability_no_digit_8_l282_282456


namespace expected_difference_is_correct_l282_282501

def fair_eight_sided_die := {1, 2, 3, 4, 5, 6, 7, 8}

def perfect_square (n : ℕ) : Prop := n = 1 ∨ n = 4

def prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7
  
noncomputable def probability (p : ℕ → Prop) := 
  (set.filter p fair_eight_sided_die).size.toReal / fair_eight_sided_die.size.toReal

def coffee_days := 365 * probability perfect_square

def tea_days := 365 * probability prime

def expected_difference := coffee_days - tea_days

theorem expected_difference_is_correct : expected_difference = -91.25 := sorry

end expected_difference_is_correct_l282_282501


namespace number_of_pairs_of_ballet_slippers_l282_282908

def price_of_high_heels : ℤ := 60
def fraction_price_of_ballet_slippers : ℚ := 2 / 3
def total_payment : ℤ := 260

theorem number_of_pairs_of_ballet_slippers : ∃ x : ℤ, 
  let price_ballet_slipper := (fraction_price_of_ballet_slippers * price_of_high_heels) in
  (price_of_high_heels + x * price_ballet_slipper) = total_payment ∧ x = 5 :=
by
  sorry

end number_of_pairs_of_ballet_slippers_l282_282908


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282275

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282275


namespace number_of_candies_in_a_packet_l282_282872

theorem number_of_candies_in_a_packet 
  (two_packets : ℕ)
  (candies_per_day_weekday : ℕ)
  (candies_per_day_weekend : ℕ)
  (weeks : ℕ)
  (total_candies : ℕ)
  (packet_size : ℕ)
  (H1 : two_packets = 2)
  (H2 : candies_per_day_weekday = 2)
  (H3 : candies_per_day_weekend = 1)
  (H4 : weeks = 3)
  (H5 : packet_size > 0)
  (H6 : total_candies = packets * packet_size)
  (H7 : total_candies = 3 * (5 * candies_per_day_weekday + 2 * candies_per_day_weekend))
  : packet_size = 18 :=
by
  sorry

end number_of_candies_in_a_packet_l282_282872


namespace number_of_subsets_of_P_l282_282970

open Set

variable {S T P : Set ℕ}
variable {a : ℕ}

def S := {3, a}
def T := {x ∈ ℤ | x^2 - 3*x < 0 ∧ x ∈ (Set.Icc 0 3)}
def P := S ∪ T

theorem number_of_subsets_of_P (hS : S ∩ T = {1}) :
  ∃ n, n = |P| ∧ n = 8 := by
  sorry

end number_of_subsets_of_P_l282_282970


namespace multiplier_for_difference_l282_282835

theorem multiplier_for_difference (number sum diff rem : ℕ) 
  (h_sum : sum = 555 + 445)
  (h_diff : diff = 555 - 445)
  (h_number : number = 220040)
  (h_rem : rem = 40) :
  number = (sum * (number // sum) * diff) + rem → number // sum = 2
  :=
by {
  intros,
  have h_eqn : number = sum * (number // sum) * diff + rem,
  { exact h } ,
  have x : number // sum = 2,
  { sorry },
  exact x
}

end multiplier_for_difference_l282_282835


namespace perfect_squares_two_digit_divisible_by_4_count_l282_282270

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l282_282270


namespace factor_polynomial_l282_282407

theorem factor_polynomial (x y z : ℤ) :
  x * (y - z) ^ 3 + y * (z - x) ^ 3 + z * (x - y) ^ 3 = (x - y) * (y - z) * (z - x) * (x + y + z) := 
by
  sorry

end factor_polynomial_l282_282407


namespace factorization_of_polynomial_l282_282538

theorem factorization_of_polynomial : ∀ x : ℝ, x^2 - x - 42 = (x + 6) * (x - 7) :=
by
  sorry

end factorization_of_polynomial_l282_282538


namespace arithmetic_sequence_sum_l282_282660

variable {a_n : ℕ → ℤ}

def is_arithmetic_sequence (a_n : ℕ → ℤ) : Prop :=
  ∀ (m n k : ℕ), m < n → (n - m) = k → a_n n = a_n m + k * (a_n 1 - a_n 0)

theorem arithmetic_sequence_sum :
  is_arithmetic_sequence a_n →
  a_n 2 = 5 →
  a_n 6 = 33 →
  a_n 3 + a_n 5 = 38 :=
by
  intros h_seq h_a2 h_a6
  sorry

end arithmetic_sequence_sum_l282_282660


namespace period_of_sin_minus_cos_l282_282455

def f (x : ℝ) : ℝ := Real.sin x - Real.cos x

theorem period_of_sin_minus_cos : ∀ x : ℝ, f (x + 2 * Real.pi) = f x :=
by
  intros x
  sorry

end period_of_sin_minus_cos_l282_282455


namespace point_quadrant_l282_282986

-- Definitions of conditions
def has_two_distinct_real_roots (a : ℝ) : Prop :=
  let Δ := 1 + a in
  a ≠ 0 ∧ Δ > 0

def point_in_fourth_quadrant (x y : ℝ) : Prop :=
  x > 0 ∧ y < 0

-- The theorem statement
theorem point_quadrant (a : ℝ) (h : has_two_distinct_real_roots a) :
  point_in_fourth_quadrant (a + 1) (-3 - a) :=
sorry

end point_quadrant_l282_282986


namespace angle_B_range_l282_282582

theorem angle_B_range (A B C : ℝ) (h1 : A ≤ B) (h2 : B ≤ C) (h3 : A + B + C = 180) (h4 : 2 * B = 5 * A) :
  0 < B ∧ B ≤ 75 :=
by
  sorry

end angle_B_range_l282_282582


namespace tangent_line_eq_l282_282548

theorem tangent_line_eq (P : ℝ × ℝ) (hP : P = (2, 2))
  (circle_eq : ∀ x y : ℝ, (x - 1)^2 + y^2 = 5 ↔ x^2 + y^2 - 2x + 1 = 5):
  ∃ k : ℝ, (y = k * x - k * 2 + 2) ∧ k = -1 / 2 :=
by
  -- Definitions
  let x := P.1
  let y := P.2
  let h := circle_eq

  -- Verify that P satisfies the circle equation
  have hP_circle : (2 - 1)^2 + 2^2 = 5
  { calc
      (2 - 1)^2 + 2^2 = 1 + 4 : by ring
                     ... = 5 : by norm_num },
  sorry

end tangent_line_eq_l282_282548


namespace burgers_needed_for_other_half_l282_282515

-- Define the given conditions
def burger_cooking_time (b : ℕ) : ℕ := 4 * 2 * b
def grill_capacity : ℕ := 5
def guests_half (total_guests : ℕ) : ℕ := total_guests / 2
def guests_first_half_burgers (half_guests : ℕ) : ℕ := half_guests * 2
def total_cooking_time : ℕ := 72

-- State the problem as a theorem in Lean
theorem burgers_needed_for_other_half
  (total_guests : ℕ)
  (htg : burger_cooking_time grill_capacity = 8)
  (hav : guests_half total_guests = 15)
  (h15 : guests_first_half_burgers 15 = 30)
  (hct : total_cooking_time = 72) :
  let total_burgers := (total_cooking_time / 8) * grill_capacity,
      other_half_burgers := total_burgers - guests_first_half_burgers 15 in
  other_half_burgers = 15 :=
by
  sorry

end burgers_needed_for_other_half_l282_282515


namespace solution_set_of_floor_equation_l282_282543

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end solution_set_of_floor_equation_l282_282543


namespace expected_value_of_biased_die_l282_282071

-- Definitions
def prob_one := 1 / 12
def prob_two := 1 / 12
def prob_three := 1 / 12
def prob_four := 1 / 4
def prob_five := 1 / 4
def prob_six := 1 / 4

def win_amount := 4
def lose_amount := -3

-- Statement to be proved
theorem expected_value_of_biased_die :
  (prob_one + prob_two + prob_three) * win_amount + 
  (prob_four + prob_five + prob_six) * lose_amount = -1.25 :=
by sorry

end expected_value_of_biased_die_l282_282071


namespace b_value_l282_282640

theorem b_value (x : ℝ) (b : ℝ) (h : x > 3000) (h1 : abs((1.2 / (b * x - 406)) - 3) < ε) : b = 0.4 :=
by sorry

end b_value_l282_282640


namespace baseball_card_value_decrease_l282_282467

theorem baseball_card_value_decrease (V0 : ℝ) (V1 V2 : ℝ) :
  V1 = V0 * 0.5 → V2 = V1 * 0.9 → (V0 - V2) / V0 * 100 = 55 :=
by 
  intros hV1 hV2
  sorry

end baseball_card_value_decrease_l282_282467


namespace max_distance_travel_l282_282441

-- Each car can carry at most 24 barrels of gasoline
def max_gasoline_barrels : ℕ := 24

-- Each barrel allows a car to travel 60 kilometers
def distance_per_barrel : ℕ := 60

-- The maximum distance one car can travel one way on a full tank
def max_one_way_distance := max_gasoline_barrels * distance_per_barrel

-- Total trip distance for the furthest traveling car
def total_trip_distance := 2160

-- Distance the other car turns back
def turn_back_distance := 360

-- Formalize in Lean
theorem max_distance_travel :
  (∃ x : ℕ, x = turn_back_distance ∧ max_gasoline_barrels * distance_per_barrel = 360) ∧
  (∃ y : ℕ, y = max_one_way_distance * 3 - turn_back_distance * 6 ∧ y = total_trip_distance) :=
by
  sorry

end max_distance_travel_l282_282441


namespace find_x_complementary_l282_282661

-- Define the conditions.
def are_complementary (a b : ℝ) : Prop := a + b = 90

-- The main theorem statement with the condition and conclusion.
theorem find_x_complementary : ∀ x : ℝ, are_complementary (2*x) (3*x) → x = 18 := 
by
  intros x h
  -- sorry is a placeholder for the proof.
  sorry

end find_x_complementary_l282_282661


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282271

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282271


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282261

-- Define the range of integers we are interested in
def two_digit_perfect_squares_divisible_by_4 : List Nat :=
  [4, 5, 6, 7, 8, 9].filter (λ n => (n * n >= 10) ∧ (n * n < 100) ∧ ((n * n) % 4 = 0))

-- Statement of the math proof problem
theorem count_two_digit_perfect_squares_divisible_by_4 :
  two_digit_perfect_squares_divisible_by_4.length = 3 :=
sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282261


namespace unit_price_ratio_l282_282509

theorem unit_price_ratio (v p : ℝ) (h_vx : 1.25 * v) (h_px : 0.85 * p) :
  (0.85 * p / (1.25 * v)) / (p / v) = 17 / 25 := 
by
  sorry

end unit_price_ratio_l282_282509


namespace sqrt_domain_l282_282317

theorem sqrt_domain (x : ℝ) : (∃ y : ℝ, y = sqrt (x - 3)) ↔ x ≥ 3 :=
by
  sorry

end sqrt_domain_l282_282317


namespace intersecting_lines_fixed_point_l282_282965

variable (p a b : ℝ)
variable (h1 : a ≠ 0)
variable (h2 : b ≠ 0)
variable (h3 : b^2 ≠ 2 * p * a)

def parabola (M : ℝ × ℝ) : Prop := M.2^2 = 2 * p * M.1

def fixed_points (A B : ℝ × ℝ) : Prop :=
  A = (a, b) ∧ B = (-a, 0)

def intersect_parabola (M1 M2 M : ℝ × ℝ) : Prop :=
  parabola p M ∧ parabola p M1 ∧ parabola p M2 ∧ M ≠ M1 ∧ M ≠ M2

theorem intersecting_lines_fixed_point (M M1 M2 : ℝ × ℝ)
  (hP : parabola p M) 
  (hA : (a, b) ≠ M) 
  (hB : (-a, 0) ≠ M) 
  (h_intersect : intersect_parabola p M1 M2 M) :
  ∃ C : ℝ × ℝ, C = (a, 2 * p * a / b) :=
sorry

end intersecting_lines_fixed_point_l282_282965


namespace gcd_cubed_and_sum_l282_282199

theorem gcd_cubed_and_sum (n : ℕ) (h_pos : 0 < n) (h_gt_square : n > 9) : 
  Nat.gcd (n^3 + 27) (n + 3) = n + 3 := 
sorry

end gcd_cubed_and_sum_l282_282199


namespace trip_total_time_trip_average_speed_l282_282828

structure Segment where
  distance : ℝ -- in kilometers
  speed : ℝ -- average speed in km/hr
  break_time : ℝ -- in minutes

def seg1 := Segment.mk 12 13 15
def seg2 := Segment.mk 18 16 30
def seg3 := Segment.mk 25 20 45
def seg4 := Segment.mk 35 25 60
def seg5 := Segment.mk 50 22 0

noncomputable def total_time_minutes (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + (s.distance / s.speed) * 60 + s.break_time) 0

noncomputable def total_distance (segs : List Segment) : ℝ :=
  segs.foldl (λ acc s => acc + s.distance) 0

noncomputable def overall_average_speed (segs : List Segment) : ℝ :=
  total_distance segs / (total_time_minutes segs / 60)

def segments := [seg1, seg2, seg3, seg4, seg5]

theorem trip_total_time : total_time_minutes segments = 568.24 := by sorry
theorem trip_average_speed : overall_average_speed segments = 14.78 := by sorry

end trip_total_time_trip_average_speed_l282_282828


namespace fg_value_l282_282983

def g (x : ℤ) : ℤ := 4 * x - 3
def f (x : ℤ) : ℤ := 6 * x + 2

theorem fg_value : f (g 5) = 104 := by
  sorry

end fg_value_l282_282983


namespace determinant_transformed_columns_l282_282362

variables {A B C : Type} [AddCommGroup A] [Module ℝ A] [AddCommGroup B] [Module ℝ B]
variables (a b c : A)
variable {D : ℝ}

def determinant (M : Matrix (Fin 3) (Fin 3) ℝ) : ℝ :=
  Matrix.det M

theorem determinant_transformed_columns {a b c : A} (hD : determinant ![a, b, c] = D) :
  determinant ![2 • a + 3 • b, 3 • b + 4 • c, 4 • c + 2 • a] = 48 * D :=
sorry

end determinant_transformed_columns_l282_282362


namespace reduction_in_jury_running_time_l282_282516

def week1_miles : ℕ := 2
def week2_miles : ℕ := 2 * week1_miles + 3
def week3_miles : ℕ := (9 * week2_miles) / 7
def week4_miles : ℕ := 4

theorem reduction_in_jury_running_time : week3_miles - week4_miles = 5 :=
by
  -- sorry specifies the proof is skipped
  sorry

end reduction_in_jury_running_time_l282_282516


namespace quadrilateral_is_kite_or_parallelogram_l282_282733

variable (A B C D : Type)
variables [IsPoint A] [IsPoint B] [IsPoint C] [IsPoint D]
variable (O : Type)
variables [IsPoint O] [IsIntersection O A C] [IsIntersection O B D]

def equal_angles_at_A_and_C (α β : Angle) : Prop := α = β
def bisected_diagonal_by_other (O A C B D: Type) : Prop := 
  is_middle_point O A C ∧ is_middle_point O B D

theorem quadrilateral_is_kite_or_parallelogram
  (h1 : equal_angles_at_A_and_C α β)
  (h2 : bisected_diagonal_by_other O A C B D) : 
  is_kite ABCD ∨ is_parallelogram ABCD := 
sorry

end quadrilateral_is_kite_or_parallelogram_l282_282733


namespace problem_solution_l282_282158

theorem problem_solution (x : ℝ) (h1 : x > 9) 
(h2 : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x ∈ Set.Ici 18 := sorry

end problem_solution_l282_282158


namespace Jack_Income_Ratio_l282_282535

noncomputable def Ernie_current_income (x : ℕ) : ℕ :=
  (4 / 5) * x

noncomputable def Jack_current_income (combined_income Ernie_current_income : ℕ) : ℕ :=
  combined_income - Ernie_current_income

theorem Jack_Income_Ratio (Ernie_previous_income combined_income : ℕ) (h₁ : Ernie_previous_income = 6000) (h₂ : combined_income = 16800) :
  let Ernie_current := Ernie_current_income Ernie_previous_income
  let Jack_current := Jack_current_income combined_income Ernie_current
  (Jack_current / Ernie_previous_income) = 2 := by
  sorry

end Jack_Income_Ratio_l282_282535


namespace sqrt_subtraction_result_l282_282797

theorem sqrt_subtraction_result : 
  (Real.sqrt (49 + 36) - Real.sqrt (36 - 0)) = 4 :=
by
  sorry

end sqrt_subtraction_result_l282_282797


namespace mustache_area_is_96_l282_282488

noncomputable def mustache_area : ℝ :=
  ∫ x in -24..24, (6 + 6 * Real.cos (Real.pi * x / 24)) -
                  (4 + 4 * Real.cos (Real.pi * x / 24))

theorem mustache_area_is_96 : mustache_area = 96 := by
  sorry

end mustache_area_is_96_l282_282488


namespace students_in_diligence_before_transfer_l282_282993

theorem students_in_diligence_before_transfer (D I : ℕ) 
  (h1 : D + 2 = I - 2) 
  (h2 : D + I = 50) : 
  D = 23 := 
by
  sorry

end students_in_diligence_before_transfer_l282_282993


namespace perfect_squares_two_digit_divisible_by_4_count_l282_282265

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l282_282265


namespace shaded_area_z_shape_l282_282154

theorem shaded_area_z_shape (L W s1 s2 : ℕ) (hL : L = 6) (hW : W = 4) (hs1 : s1 = 2) (hs2 : s2 = 1) :
  (L * W - (s1 * s1 + s2 * s2)) = 19 := by
  sorry

end shaded_area_z_shape_l282_282154


namespace constant_term_of_expansion_l282_282117

noncomputable def const_term_of_expansion := (160 : ℕ)

theorem constant_term_of_expansion (x : ℚ) :
  let expr := (1 / x^2 + 4 * x^2 + 4) in
  (expr ^ 3).coeff 0 = const_term_of_expansion := sorry

end constant_term_of_expansion_l282_282117


namespace shirt_sales_analysis_l282_282761

theorem shirt_sales_analysis :
  let daily_sales := [2, 3, 5, 1, 1]
  let mode := 5
  ∃ shirt_size, shirt_size = 42 ∧ mode = 5 :=
by
  let daily_sales := [2, 3, 5, 1, 1]
  let mode := 5
  use 42
  split
  sorry

end shirt_sales_analysis_l282_282761


namespace inequality_solution_diff_l282_282723

def integer_part (y : ℝ) : ℤ := int.floor y

noncomputable def cube_root (x : ℝ) : ℝ := real.cbrt x

theorem inequality_solution_diff :
  ∀ x : ℝ, x^2 ≤ 2 * (integer_part (cube_root x + 0.5) + integer_part (cube_root x)) →
  let solutions := { x | x^2 ≤ 2 * (integer_part (cube_root x + 0.5) + integer_part (cube_root x))} in
  let max_solution := Sup solutions in
  let min_solution := Inf solutions in
  max_solution - min_solution = 1 := by
sorry

end inequality_solution_diff_l282_282723


namespace find_s_l282_282479

variable (x t s : ℝ)

-- Conditions
#check (0.75 * x) / 60  -- Time for the first part of the trip
#check 0.25 * x  -- Distance for the remaining part of the trip
#check t - (0.75 * x) / 60  -- Time for the remaining part of the trip
#check 40 * t  -- Solving for x from average speed relation

-- Prove the value of s
theorem find_s (h1 : x = 40 * t) (h2 : s = (0.25 * x) / (t - (0.75 * x) / 60)) : s = 20 := by sorry

end find_s_l282_282479


namespace min_g_is_correct_l282_282729

noncomputable def g (Y P Q R S : ℝ × ℝ × ℝ) : ℝ :=
  (dist P Y) + (dist Q Y) + (dist R Y) + (dist S Y)

noncomputable def min_g_value (P Q R S : ℝ × ℝ × ℝ)
  [dist P Q = 50] [dist R S = 50]
  [dist P R = 26] [dist Q S = 26]
  [dist P S = 42] [dist R Q = 42] : ℝ :=
  4 * Real.sqrt 482

theorem min_g_is_correct (P Q R S : ℝ × ℝ × ℝ)
  (h1 : dist P Q = 50) (h2 : dist R S = 50)
  (h3 : dist P R = 26) (h4 : dist Q S = 26)
  (h5 : dist P S = 42) (h6 : dist R Q = 42) : a + b = 486 := 
begin
  let a := 4,
  let b := 482,
  have h7 : a * Real.sqrt b = min_g_value P Q R S h1 h2 h3 h4 h5 h6,
  {
    sorry
  },
  have h8 : a + b = 486,
  {
    sorry
  },
  exact h8,
end

end min_g_is_correct_l282_282729


namespace solve_for_a_l282_282215

theorem solve_for_a
  (a : ℝ) (n : ℕ)
  (h₀ : a ≠ 0)
  (h₁ : 1 < n)
  (expansion : ∀ x : ℝ, (∑ k in finset.range (n + 1), (nat.choose n k) * (x / a)^k) = ∑ k in finset.range (n + 1), a_k * x^k)
  (h₂ : a₁ = 3)
  (h₃ : a₂ = 4) :
  a = 3 :=
sorry

end solve_for_a_l282_282215


namespace joshua_share_is_30_l282_282675

-- Definitions based on the conditions
def total_amount_shared : ℝ := 40
def ratio_joshua_justin : ℝ := 3

-- Proposition to prove
theorem joshua_share_is_30 (J : ℝ) (Joshua_share : ℝ) :
  J + ratio_joshua_justin * J = total_amount_shared → 
  Joshua_share = ratio_joshua_justin * J → 
  Joshua_share = 30 :=
sorry

end joshua_share_is_30_l282_282675


namespace trigonometric_identity_l282_282947

theorem trigonometric_identity 
  (x : ℝ) 
  (h : Real.sin (x + Real.pi / 3) = 1 / 3) :
  Real.sin (5 * Real.pi / 3 - x) - Real.cos (2 * x - Real.pi / 3) = 4 / 9 :=
by
  sorry

end trigonometric_identity_l282_282947


namespace find_q_of_polynomial_with_conditions_l282_282925

theorem find_q_of_polynomial_with_conditions (p q r s : ℝ) (z w : ℂ) (hx_poly : x^4 + p * x^3 + q * x^2 + r * x + s = 0)
    (h_real_coeffs : ∀ coefs, coefs ∈ [p, q, r, s] → coefs ∈ ℝ)
    (h_nonreal_roots : (∃ z w : ℂ, z ≠ conj z ∧ w ≠ conj w ∧ hx_poly.has_root z ∧ hx_poly.has_root w ∧ hx_poly.has_root (conj z) ∧ hx_poly.has_root (conj w)))
    (h_product : z * w = 7 + i)
    (h_sum : conj z + conj w = 2 + 3 * i) : 
    q = 91 := sorry

end find_q_of_polynomial_with_conditions_l282_282925


namespace candy_per_packet_l282_282874

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l282_282874


namespace correct_answer_l282_282644

variables (x y : ℝ)

def cost_equations (x y : ℝ) : Prop :=
  (2 * x + 3 * y = 120) ∧ (2 * x - y = 20)

theorem correct_answer : cost_equations x y :=
sorry

end correct_answer_l282_282644


namespace part_I_part_I_line_part_II_distance_l282_282954

-- Definition for the conversion of polar equation to rectangular equation
def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Math.cos θ, ρ * Math.sin θ)

-- Given conditions
def curve_C_polar_eq (ρ θ : ℝ) : Prop :=
  ρ = 4 * Math.cos θ

def line_l_param_eq (t : ℝ) : ℝ × ℝ :=
  (1 - t, t)

-- Problem (I): Prove that the rectangular equation of curve C and the general equation of line l
theorem part_I (ρ θ : ℝ) (h : curve_C_polar_eq ρ θ) : (∃ x y : ℝ, (x, y) = polar_to_rectangular ρ θ ∧ (x - 2)^2 + y^2 = 4) :=
by
  sorry

theorem part_I_line (t : ℝ) : ∃ x y : ℝ, (x, y) = line_l_param_eq t ∧ x + y - 1 = 0 :=
by
  sorry

-- Problem (II): Prove that the distance |PQ| is sqrt(14)
theorem part_II_distance (t1 t2 : ℝ) (h1 : ∃ x1 y1 : ℝ, (x1, y1) = line_l_param_eq t1 ∧ (x1 - 2)^2 + y1^2 = 4)
  (h2 : ∃ x2 y2 : ℝ, (x2, y2) = line_l_param_eq t2 ∧ (x2 - 2)^2 + y2^2 = 4) : 
  abs (t1 - t2) = Real.sqrt 14 :=
by
  sorry

end part_I_part_I_line_part_II_distance_l282_282954


namespace expected_difference_l282_282508

noncomputable def fair_eight_sided_die := [2, 3, 4, 5, 6, 7, 8]

def is_prime (n : ℕ) : Prop := 
  n = 2 ∨ n = 3 ∨ n = 5 ∨ n = 7

def is_composite (n : ℕ) : Prop := 
  n = 4 ∨ n = 6 ∨ n = 8

def unsweetened_cereal_days := (4 / 7) * 365
def sweetened_cereal_days := (3 / 7) * 365

theorem expected_difference :
  unsweetened_cereal_days - sweetened_cereal_days = 53 := by
  sorry

end expected_difference_l282_282508


namespace rins_helmet_craters_l282_282135

def craters_in_Rins_helmet (craters_dan : ℕ) (craters_difference : ℕ) (extra_craters : ℕ) : ℕ :=
  let craters_daniel := craters_dan - craters_difference
  let combined_craters := craters_dan + craters_daniel
  combined_craters + extra_craters

theorem rins_helmet_craters :
  craters_in_Rins_helmet 35 10 15 = 75 :=
by
  unfold craters_in_Rins_helmet
  simp
  sorry

end rins_helmet_craters_l282_282135


namespace count_palindromes_1000_2000_l282_282250

theorem count_palindromes_1000_2000 : 
  ∃ (n : ℕ), n = 10 ∧
  (∀ (x : ℕ), 1000 ≤ x ∧ x < 2000
  → (int.digits 10 x).reverse = (int.digits 10 x) 
  → int.digits 10 x = [1, b, b, 1] ∧ ∃ b: ℕ, b < 10) :=
sorry

end count_palindromes_1000_2000_l282_282250


namespace count_two_digit_perfect_squares_divisible_by_four_l282_282257

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l282_282257


namespace maximum_area_of_rectangle_with_given_perimeter_l282_282415

noncomputable def perimeter : ℝ := 30
noncomputable def area (length width : ℝ) : ℝ := length * width
noncomputable def max_area : ℝ := 56.25

theorem maximum_area_of_rectangle_with_given_perimeter :
  ∃ length width : ℝ, 2 * length + 2 * width = perimeter ∧ area length width = max_area :=
sorry

end maximum_area_of_rectangle_with_given_perimeter_l282_282415


namespace athlete_A_mode_and_percentile_athlete_A_variance_athlete_B_variance_more_stable_athlete_l282_282111

-- Constants for Athlete A's scores
def scores_A : List ℕ := [7, 8, 7, 9, 5, 4, 9, 10, 7, 4]

-- Constants for Athlete B's scores
def scores_B : List ℕ := [9, 5, 7, 8, 7, 6, 8, 6, 7, 7]

-- Proof of mode and 85th percentile
theorem athlete_A_mode_and_percentile :
  (mode scores_A = 7) ∧ (percentile 85 scores_A = 9) :=
by
  sorry

-- Proof of variance of Athlete A's scores
theorem athlete_A_variance :
  variance scores_A = 4 :=
by
  sorry

-- Proof of variance of Athlete B's scores
theorem athlete_B_variance :
  variance scores_B = 1.2 :=
by
  sorry

-- Proof of more stable performer being Athlete B
theorem more_stable_athlete :
  more_stable scores_B scores_A :=
by
  sorry

end athlete_A_mode_and_percentile_athlete_A_variance_athlete_B_variance_more_stable_athlete_l282_282111


namespace symmetric_point_correct_l282_282914

open Real

def point : Type := (ℝ × ℝ × ℝ)
def line : Type := (ℝ → point)

noncomputable def symmetric_point (M : point) (L : line) : point := 
  let M' := (1, 2, 3)
  M'

theorem symmetric_point_correct : 
  let M := (0, -3, -2)
  let L : line := λ t, (0.5, -1.5 - t, 1.5 + t)
  symmetric_point M L = (1, 2, 3) :=
by 
  sorry

end symmetric_point_correct_l282_282914


namespace sin_add_cos_value_sin2_over_diff_l282_282246

variable (α : ℝ)
def m := (Real.cos α - Real.sqrt 2 / 3, -1)
def n := (Real.sin α, 1)
def collinear_vectors := m.1 * n.2 - m.2 * n.1 = 0
def alpha_in_range : Prop := α ≥ -Real.pi / 2 ∧ α ≤ 0

theorem sin_add_cos_value (h1 : collinear_vectors α) (h2 : alpha_in_range α) :
  Real.sin α + Real.cos α = Real.sqrt 2 / 3 := sorry

theorem sin2_over_diff (h1 : collinear_vectors α) (h2 : alpha_in_range α) :
  Real.sin (2 * α) / (Real.sin α - Real.cos α) = 7 / 12 := sorry

end sin_add_cos_value_sin2_over_diff_l282_282246


namespace count_two_digit_perfect_squares_divisible_by_four_l282_282255

theorem count_two_digit_perfect_squares_divisible_by_four : ∃ n, n = 3 ∧
  (∀ k, (10 ≤ k ∧ k < 100) → (∃ m, k = m^2) → k % 4 = 0 → ∃ p, (p = 16 ∨ p = 36 ∨ p = 64) ∧ p = k) := 
by 
  use 3
  intro k h1 h2 h3
  cases h2 with m hm
  sorry

end count_two_digit_perfect_squares_divisible_by_four_l282_282255


namespace deductive_reasoning_problem_l282_282619

noncomputable theory

-- Definitions for the basic geometric entities
structure Line :=
parallel_to_plane : Plane → Prop

structure Plane := 
  -- Assume a plane can be defined appropriately
  in_plane : Line → Prop

-- Conditions from the problem
def major_premise (l : Line) (p : Plane) : Prop := 
  ∀ (l1 : Line), l.parallel_to_plane p → p.in_plane l1 → l.parallel_to_plane l1

def minor_premise (a b : Line) (p : Plane) : Prop := 
  b.parallel_to_plane p ∧ p.in_plane a

-- Define the conclusion
def conclusion (a b : Line) : Prop := b.parallel_to_plane a

-- Statement of the proof problem
theorem deductive_reasoning_problem (a b : Line) (p : Plane) : 
  ¬ (major_premise b p) ∧ minor_premise a b p ∧ ¬ (conclusion a b) :=
by
  sorry

end deductive_reasoning_problem_l282_282619


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282273

theorem count_two_digit_perfect_squares_divisible_by_4 :
  {n : ℕ | n ∈ (set.range (λ m, m ^ 2)) ∧ 10 ≤ n ∧ n < 100 ∧ n % 4 = 0}.to_finset.card = 3 :=
by
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282273


namespace probability_of_drawing_4_black_cards_l282_282567

-- Definitions matching given conditions
def num_black_cards := 26
def num_total_cards := 52

-- Function to compute binomial coefficient
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Definitions for specific binomial coefficients used in the problem
def choose_black := binom num_black_cards 4
def choose_total := binom num_total_cards 4

-- The probability computed as a fraction
noncomputable def probability := (choose_black : ℚ) / choose_total

-- The expected outcome of the probability
def expected_probability := (276 : ℚ) / 4998

-- Statement to be proved
theorem probability_of_drawing_4_black_cards : probability = expected_probability := 
by 
  sorry

end probability_of_drawing_4_black_cards_l282_282567


namespace find_angle_A_find_cos_sum_l282_282654

noncomputable def triangle_angles (A B C : ℝ) (a b c : ℝ) :=
  (acute_triangle : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2)
  ∧ (a2 : a > 0)
  ∧ (b2 : b > 0)
  ∧ (c2 : c > 0)
  ∧ (abc_relation : A + B + C = π)
  ∧ (law_of_cosines : sqrt 3 * c * cos A - a * cos C + b - 2 * c = 0)

theorem find_angle_A (A B C a b c : ℝ) (h : triangle_angles A B C a b c) : 
  A = π / 3 :=
sorry

theorem find_cos_sum (A B C : ℝ) (a b c : ℝ) 
  (h1 : triangle_angles A B C a b c)
  (h2 : A = π / 3) :
  ∃ range, range = Ioc (sqrt 3 / 2) 1 ∧ 
           ∀ B C, 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 → 
                   B + C = 2 * π / 3 → cos B + cos C ∈ range :=
sorry

end find_angle_A_find_cos_sum_l282_282654


namespace rectangle_area_l282_282933

theorem rectangle_area (p q : ℝ) (x : ℝ) (h1 : x^2 + (2 * x)^2 = (p + q)^2) : 
    2 * x^2 = (2 * (p + q)^2) / 5 := 
sorry

end rectangle_area_l282_282933


namespace degrees_of_remainder_l282_282043

-- Definitions based on the conditions
def divisor : Polynomial ℤ := -4*X^6 + 3*X^3 - 5*X + 7

-- Statement of the theorem to be proved
theorem degrees_of_remainder (p : Polynomial ℤ) (q r : Polynomial ℤ)
  (h : p = divisor * q + r)
  (hr : r.degree < divisor.degree) :
  r.degree ∈ {0, 1, 2, 3, 4, 5} := 
sorry

end degrees_of_remainder_l282_282043


namespace length_of_southwest_run_l282_282133

-- Define the coordinates of the points
def A : (ℝ × ℝ) := (0, 0)
def B : (ℝ × ℝ) := (0, 2)
def C : (ℝ × ℝ) := (3, 2)

-- Define the distance function
def dist (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Prove the distance from C to A is sqrt(13)
theorem length_of_southwest_run : dist C A = Real.sqrt 13 :=
by
  sorry

end length_of_southwest_run_l282_282133


namespace line_eq_l282_282742

theorem line_eq (m b : ℝ) 
  (h_slope : m = (4 + 2) / (3 - 1)) 
  (h_point : -2 = m * 1 + b) :
  m + b = -2 :=
by
  sorry

end line_eq_l282_282742


namespace sum_of_alternating_sums_l282_282929

def alternating_sum (s : Finset ℕ) :=
  s.sort (· > ·)  -- sort in decreasing order
    |> List.foldl (λ (acc : ℕ × Bool) x, if acc.2 then (acc.1 + x, !acc.2) else (acc.1 - x, !acc.2)) (0, true)
    |> Prod.fst

theorem sum_of_alternating_sums :
  let A := Finset.range 9 \ {0} in
  (Finset.powerset A).filter (λ s, s.nonempty)
    .sum (λ s, alternating_sum s) = 1024 :=
by 
  sorry

end sum_of_alternating_sums_l282_282929


namespace ticket_cost_at_30_years_l282_282042

noncomputable def initial_cost : ℝ := 1000000
noncomputable def halving_period_years : ℕ := 10
noncomputable def halving_factor : ℝ := 0.5

def cost_after_n_years (initial_cost : ℝ) (halving_factor : ℝ) (years : ℕ) (period : ℕ) : ℝ :=
  initial_cost * halving_factor ^ (years / period)

theorem ticket_cost_at_30_years (initial_cost halving_factor : ℝ) (years period: ℕ) 
  (h_initial_cost : initial_cost = 1000000)
  (h_halving_factor : halving_factor = 0.5)
  (h_years : years = 30)
  (h_period : period = halving_period_years) : 
  cost_after_n_years initial_cost halving_factor years period = 125000 :=
by 
  sorry

end ticket_cost_at_30_years_l282_282042


namespace hyperbola_eccentricity_l282_282220

theorem hyperbola_eccentricity (C : Type) (a b c e : ℝ)
  (h_asymptotes : ∀ x : ℝ, (∃ y : ℝ, y = x ∨ y = -x)) :
  a = b ∧ c = Real.sqrt (a^2 + b^2) ∧ e = c / a → e = Real.sqrt 2 := 
by
  sorry

end hyperbola_eccentricity_l282_282220


namespace solution_set_of_floor_equation_l282_282544

theorem solution_set_of_floor_equation (x : ℝ) : 
  (⌊⌊2 * x⌋ - 1/2⌋ = ⌊x + 3⌋) ↔ (3.5 ≤ x ∧ x < 4.5) :=
by sorry

end solution_set_of_floor_equation_l282_282544


namespace pa_squared_greater_pb_pc_l282_282667

-- Definitions of the side lengths of the triangle
def AB : ℝ := 2 * Real.sqrt 2
def AC : ℝ := Real.sqrt 2
def BC : ℝ := 2

-- Assume P is an arbitrary point on side BC, represented as a coefficient x
-- where BP = x and PC = BC - x
def BP (x : ℝ) : ℝ := x
def PC (x : ℝ) : ℝ := BC - x

-- Prove that PA^2 is greater than PB * PC for any point P on BC
theorem pa_squared_greater_pb_pc (x : ℝ) (h : 0 ≤ x ∧ x ≤ BC):
  let PA_squared := x^2 - 5 * x + 8
  let PB_PC := BP x * PC x
  PA_squared > PB_PC := by
  sorry

end pa_squared_greater_pb_pc_l282_282667


namespace blanch_breakfast_slices_l282_282112

-- Define the initial number of slices
def initial_slices : ℕ := 15

-- Define the slices eaten at different times
def lunch_slices : ℕ := 2
def snack_slices : ℕ := 2
def dinner_slices : ℕ := 5

-- Define the number of slices left
def slices_left : ℕ := 2

-- Calculate the total slices eaten during lunch, snack, and dinner
def total_eaten_ex_breakfast : ℕ := lunch_slices + snack_slices + dinner_slices

-- Define the slices eaten during breakfast
def breakfast_slices : ℕ := initial_slices - total_eaten_ex_breakfast - slices_left

-- The theorem to prove
theorem blanch_breakfast_slices : breakfast_slices = 4 := by
  sorry

end blanch_breakfast_slices_l282_282112


namespace count_two_digit_perfect_squares_divisible_by_4_l282_282288

-- Define what it means to be a two-digit number perfect square divisible by 4
def two_digit_perfect_squares_divisible_by_4 : List ℕ :=
  [16, 36, 64] -- Manually identified two-digit perfect squares which are divisible by 4

-- 6^2 = 36 and 8^2 = 64 both fit, hypothesis checks are already done manually in solution steps
def valid_two_digit_perfect_squares : List ℕ :=
  [16, 25, 36, 49, 64, 81] -- all two-digit perfect squares

-- Define the theorem statement
theorem count_two_digit_perfect_squares_divisible_by_4 :
  (two_digit_perfect_squares_divisible_by_4.count 16 + 
   two_digit_perfect_squares_divisible_by_4.count 36 +
   two_digit_perfect_squares_divisible_by_4.count 64) = 3 :=
by
  -- Proof would go here, omitted by "sorry"
  sorry

end count_two_digit_perfect_squares_divisible_by_4_l282_282288


namespace whatsapp_message_difference_l282_282816

theorem whatsapp_message_difference :
  ∃ W : ℕ, let total_messages := 300 + 200 + W + 2 * W in total_messages = 2000 → W - 200 = 300 :=
by
  sorry

end whatsapp_message_difference_l282_282816


namespace third_butcher_delivered_8_packages_l282_282697

variables (x y z t1 t2 t3 : ℕ)

-- Given Conditions
axiom h1 : x = 10
axiom h2 : y = 7
axiom h3 : 4 * x + 4 * y + 4 * z = 100
axiom t1_time : t1 = 8
axiom t2_time : t2 = 10
axiom t3_time : t3 = 18

-- Proof Problem
theorem third_butcher_delivered_8_packages :
  z = 8 :=
by
  -- proof to be filled
  sorry

end third_butcher_delivered_8_packages_l282_282697


namespace determinant_signs_no_all_positive_l282_282396

-- Define a 3x3 determinant with elements that can have any sign.
noncomputable def determinant : (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) × (ℝ × ℝ × ℝ) → ℝ := 
  λ ((a11, a12, a13), (a21, a22, a23), (a31, a32, a33)),
    a11 * a22 * a33 + a12 * a23 * a31 + a13 * a21 * a32 - 
    a13 * a22 * a31 - a11 * a23 * a32 - a12 * a21 * a33

-- The formal statement of the problem.
theorem determinant_signs_no_all_positive :
  ∀ (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℝ),
    ¬ (∀ ((a11 * a22 * a33) > 0 ∧
          (a12 * a23 * a31) > 0 ∧
          (a13 * a21 * a32) > 0 ∧
          (-(a13 * a22 * a31)) > 0 ∧
          (-(a11 * a23 * a32)) > 0 ∧
          (-(a12 * a21 * a33)) > 0)) :=
begin
  -- Proof (skipped)
  sorry
end

end determinant_signs_no_all_positive_l282_282396


namespace total_passengers_transportation_l282_282492

theorem total_passengers_transportation : 
  let passengers_one_way := 100
  let passengers_return := 60
  let first_trip_total := passengers_one_way + passengers_return
  let additional_trips := 3
  let additional_trips_total := additional_trips * first_trip_total
  let total_passengers := first_trip_total + additional_trips_total
  total_passengers = 640 := 
by
  sorry

end total_passengers_transportation_l282_282492


namespace factorize_expression_l282_282907

-- Variables used in the expression
variables (m n : ℤ)

-- The expression to be factored
def expr := 4 * m^3 * n - 16 * m * n^3

-- The desired factorized form of the expression
def factored := 4 * m * n * (m + 2 * n) * (m - 2 * n)

-- The proof problem statement
theorem factorize_expression : expr m n = factored m n :=
by sorry

end factorize_expression_l282_282907


namespace perfect_squares_two_digit_divisible_by_4_count_l282_282267

-- Define two-digit
def is_two_digit (n : ℤ) : Prop :=
  10 ≤ n ∧ n < 100

-- Define perfect square
def is_perfect_square (n : ℤ) : Prop :=
  ∃ k : ℤ, n = k^2

-- Define divisible by 4
def divisible_by_4 (n : ℤ) : Prop :=
  n % 4 = 0

-- Define the main statement: number of two-digit perfect squares that are divisible by 4 is 3
theorem perfect_squares_two_digit_divisible_by_4_count :
  { n : ℤ | is_two_digit n ∧ is_perfect_square n ∧ divisible_by_4 n }.size = 3 :=
by sorry

end perfect_squares_two_digit_divisible_by_4_count_l282_282267


namespace annual_income_is_correct_l282_282833

noncomputable def total_investment : ℝ := 4455
noncomputable def price_per_share : ℝ := 8.25
noncomputable def dividend_rate : ℝ := 12 / 100
noncomputable def face_value : ℝ := 10

noncomputable def number_of_shares : ℝ := total_investment / price_per_share
noncomputable def dividend_per_share : ℝ := dividend_rate * face_value
noncomputable def annual_income : ℝ := dividend_per_share * number_of_shares

theorem annual_income_is_correct : annual_income = 648 := by
  sorry

end annual_income_is_correct_l282_282833


namespace problem_solution_l282_282928

variables (p q : Prop)
def proposition_p := ∀ (x : ℝ), log x / log 2 > 0
def proposition_q := ∃ (x : ℝ), 2^x < 0

theorem problem_solution (h1 : ¬ proposition_p) (h2 : ¬ proposition_q) : (proposition_p ∨ ¬ proposition_q) :=
by
  sorry

end problem_solution_l282_282928


namespace interval_of_increase_l282_282138

noncomputable def function_of_interest (x : ℝ) : ℝ :=
  if h : 3 * x - x ^ 3 > 0 then log (3 * x - x ^ 3) else 0

theorem interval_of_increase :
  ∀ x, 0 < x ∧ x < 1 → monotone (function_of_interest x) :=
sorry

end interval_of_increase_l282_282138


namespace score_at_least_118_l282_282861

noncomputable def normal_distribution (μ σ : ℝ) : ℝ → ℝ := 
  λ x, (1 / (σ * Real.sqrt (2 * Real.pi))) * Real.exp (-(x - μ)^2 / (2 * σ^2))

theorem score_at_least_118 
  (μ σ : ℝ) (μ_value : μ = 98) (σ_value : σ = 10)
  (n_students : ℕ) (rank : ℕ) 
  (prob_data : ∀ x σ, 
    x = μ - 2 * σ → 
    (μ + 2 * σ) = 118 → 
    prob_data = 0.9545 ∧ 0.5 * (1 - prob_data) = 0.02275)
  (top_rank_probability : ℝ) 
  (top_rank_probability_value : top_rank_probability = 9100 / 400000):
  ∃ X : ℝ, normal_distribution μ σ X ∧ X ≥ 118 :=
by
  sorry

end score_at_least_118_l282_282861


namespace shaded_area_T_shape_l282_282539

theorem shaded_area_T_shape (a b c d e: ℕ) (square_side_length rect_length rect_width: ℕ)
  (h_side_lengths: ∀ x, x = 2 ∨ x = 4) (h_square: square_side_length = 6) 
  (h_rect_dim: rect_length = 4 ∧ rect_width = 2)
  (h_areas: [a, b, c, d, e] = [4, 4, 4, 8, 4]) :
  a + b + d + e = 20 :=
by
  sorry

end shaded_area_T_shape_l282_282539


namespace extreme_points_of_f_is_0_l282_282760

def f (x a : ℝ) : ℝ := x^3 + 3 * x^2 + 3 * x - a

def f_prime (x : ℝ) : ℝ := 3 * x^2 + 6 * x + 3

theorem extreme_points_of_f_is_0 (a : ℝ) : 
  (∀ x : ℝ, f_prime x ≥ 0) →
  (∀ x y : ℝ, x < y → f x < f y) →
  ∀ x y : ℝ, f x = f y → x = y :=
by
  assume h1 h2,
  intros x y hxy,
  sorry

end extreme_points_of_f_is_0_l282_282760


namespace tetrahedron_configurations_l282_282808

-- Define the transformation of the tetrahedron under the given constraints
structure Tetrahedron where
  vertex : ℕ → ℕ -- Function to map vertex number to another vertex number

-- Define the two types of rotation transformations
def rotation1 (t : Tetrahedron) : Tetrahedron :=
  sorry

def rotation2 (t : Tetrahedron) : Tetrahedron :=
  sorry

-- Define the configurations of the tetrahedron
def configurations : Finset Tetrahedron :=
  sorry

-- Main theorem to prove
theorem tetrahedron_configurations :
  ∃ configs : Finset Tetrahedron,
    (configs.card = 12) ∧
    (∀ x y ∈ configs, ∃ n ≤ 4, (rotation1^[n] x = y ∨ rotation2^[n] x = y)) ∧
    (¬ ∀ x y ∈ configs, ∃ seq : List (Tetrahedron → Tetrahedron), (∀ i < seq.length - 1, seq.nth_le i _ ≠ seq.nth_le (i + 1) _) ∧ (seq.head = x) ∧ (seq.last = y) ∧ (seq.length ≤ 6)) :=
begin
  sorry
end

end tetrahedron_configurations_l282_282808


namespace problem1_problem2_problem3_min_a_l282_282232

-- Problem 1
theorem problem1 (a : ℝ) :
  (∀ x : ℝ, f (2 - x) = f (2 + x)) ↔ a = -4 := sorry

-- Problem 2
theorem problem2 (a : ℝ) (x ∈ set.Icc (-2 : ℝ) (4 : ℝ)) :
  ∃ y ∈ set.Icc (-2 : ℝ) (4 : ℝ), f y = max (f (-2)) (f 4) := sorry

-- Problem 3
theorem problem3_min_a :
  (∀ x ∈ set.Icc (-2 : ℝ) (2 : ℝ), f x ≥ a) ↔ a = -7 := sorry

def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * x + 3

end problem1_problem2_problem3_min_a_l282_282232


namespace right_triangle_angle_division_l282_282858

theorem right_triangle_angle_division (A B C H M : Type)
  [triangle : triangle ABC]
  (angle_A : angle ABC A = (π / 6)) -- condition 1: ∠A = 30°
  (right_angle_C : angle ABC C = (π / 2)) -- condition 2: right angle at C
  (height_CH : height CH ABC C H) -- condition 3: CH is the height from C
  (median_CM : median CM ABC C M) -- condition 4: CM is the median from C
  : angle BCH = (π / 6) ∧ angle HCM = (π / 6) ∧ angle MCA = (π / 6) := 
sorry

end right_triangle_angle_division_l282_282858


namespace number_of_cds_on_shelf_l282_282531

-- Definitions and hypotheses
def cds_per_rack : ℕ := 8
def racks_per_shelf : ℕ := 4

-- Theorem statement
theorem number_of_cds_on_shelf :
  cds_per_rack * racks_per_shelf = 32 :=
by sorry

end number_of_cds_on_shelf_l282_282531


namespace range_of_s_l282_282791

def s (x : ℝ) : ℝ := 1 / (2 - x)^3

theorem range_of_s :
  set.range (λ x, s x) = set.Ioo (−∞) 0 ∪ set.Ioo 0 ∞ :=
sorry

end range_of_s_l282_282791


namespace store_a_cheaper_by_two_l282_282780

def store_a_price : ℝ := 125
def store_b_price : ℝ := 130
def store_a_discount : ℝ := 0.08
def store_b_discount : ℝ := 0.10

def final_price_store_a : ℝ := store_a_price - (store_a_price * store_a_discount)
def final_price_store_b : ℝ := store_b_price - (store_b_price * store_b_discount)

theorem store_a_cheaper_by_two :
  final_price_store_b - final_price_store_a = 2 :=
by
  unfold final_price_store_b final_price_store_a store_a_price store_b_price store_a_discount store_b_discount
  have h₁ : store_b_price - store_b_price * store_b_discount = 117 := by norm_num
  have h₂ : store_a_price - store_a_price * store_a_discount = 115 := by norm_num
  rw [h₁, h₂]
  norm_num
  sorry

end store_a_cheaper_by_two_l282_282780


namespace largest_sum_of_angles_l282_282657

theorem largest_sum_of_angles (x r : ℝ) (EFG_similar_HGF : similar_triangles EFG HGF)
  (EFGH_arith_seq : x + (x + r) + (x + 2 * r) + (x + 3 * r) = 360) :
  let θ1 := min x (min (x + r) (min (x + 2 * r) (x + 3 * r))),
      θ2 := max x (max (x + r) (max (x + 2 * r) (x + 3 * r)))
  in θ1 + θ2 = 180 :=
sorry

end largest_sum_of_angles_l282_282657


namespace negation_of_forall_l282_282759

theorem negation_of_forall {α : Type} (P : α → Prop) :
  ¬(∀ x, x > 1 → P x) ↔ ∃ x, x > 1 ∧ ¬ P x := by 
  sorry

end negation_of_forall_l282_282759


namespace proof_x_minus_y_eq_zero_l282_282573

theorem proof_x_minus_y_eq_zero (x y : ℝ) (h : (x + complex.I) * complex.I + y = 1 + 2 * complex.I) : x - y = 0 :=
sorry

end proof_x_minus_y_eq_zero_l282_282573


namespace lcm_of_fractions_l282_282469

theorem lcm_of_fractions :
  let nums := [4, 5, 9, 7],
      dens := [9, 7, 13, 15] in
  (nat.lcm_list nums) / (nat.gcd_list dens) = 1260 :=
by
  sorry

end lcm_of_fractions_l282_282469


namespace downstream_distance_l282_282850

variable (v d : ℝ)

-- Conditions
def woman_speed_still_water : ℝ := 5
def time_taken_each_way : ℝ := 6
def distance_upstream : ℝ := 6
def effective_speed_downstream : ℝ := woman_speed_still_water + v
def effective_speed_upstream : ℝ := woman_speed_still_water - v

-- Lean theorem statement
theorem downstream_distance :
  (time_taken_each_way = distance_upstream / effective_speed_upstream) →
  (time_taken_each_way = d / effective_speed_downstream) → 
  d = 54 := by
  sorry

end downstream_distance_l282_282850


namespace inequality_solution_diff_l282_282722

def integer_part (y : ℝ) : ℤ := int.floor y

noncomputable def cube_root (x : ℝ) : ℝ := real.cbrt x

theorem inequality_solution_diff :
  ∀ x : ℝ, x^2 ≤ 2 * (integer_part (cube_root x + 0.5) + integer_part (cube_root x)) →
  let solutions := { x | x^2 ≤ 2 * (integer_part (cube_root x + 0.5) + integer_part (cube_root x))} in
  let max_solution := Sup solutions in
  let min_solution := Inf solutions in
  max_solution - min_solution = 1 := by
sorry

end inequality_solution_diff_l282_282722


namespace sally_bought_48_eggs_l282_282398

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozens Sally bought
def dozens_sally_bought : ℕ := 4

-- Define the total number of eggs Sally bought
def total_eggs_sally_bought : ℕ := dozens_sally_bought * eggs_in_a_dozen

-- Theorem stating the number of eggs Sally bought
theorem sally_bought_48_eggs : total_eggs_sally_bought = 48 :=
sorry

end sally_bought_48_eggs_l282_282398


namespace bristol_to_birmingham_routes_l282_282082

theorem bristol_to_birmingham_routes {B S C : ℕ} (hS : S = 3) (hC : C = 2) (hBSC : B * S * C = 36) : B = 6 :=
by
  -- conditions
  rw [hS, hC] at hBSC
  calc
    B * 6 = 36 : by rwa [mul_assoc 2 3 B, mul_comm 6 B, mul_comm 36 6]
    B = 6 : by linarith

end bristol_to_birmingham_routes_l282_282082


namespace length_of_median_half_l282_282812

variable {A B C M : ℝ × ℝ}
variable {x1 y1 x2 y2 x3 y3 : ℝ}

-- Coordinates of vertices
def vertex_A := (x1, y1 : ℝ)
def vertex_B := (x2, y2 : ℝ)
def vertex_C := (x3, y3 : ℝ)

-- Midpoint of B and C
def midpoint_BC (B C : ℝ × ℝ) : ℝ × ℝ :=
  ((B.1 + C.1) / 2, (B.2 + C.2) / 2)

-- Distance formula between two points
def distance (P Q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((Q.1 - P.1) ^ 2 + (Q.2 - P.2) ^ 2)

theorem length_of_median_half (A B C : ℝ × ℝ) :
  let M := midpoint_BC B C in
  distance A M = (1 / 2) * distance B C :=
by
  sorry

end length_of_median_half_l282_282812


namespace f_at_three_f_half_equals_neg_five_halves_f_periodic_l282_282590

noncomputable def f : ℝ → ℝ := sorry

-- Conditions:
-- 1. Domain f(x) is ℝ
-- 2. f(x + 1) is an odd function (∀ x : ℝ, f(1 - x) = -f(1 + x))
-- 3. For all x ∈ ℝ, f(x + 4) = f(-x)
axiom f_domain (x : ℝ) : true
axiom f_odd_x_plus_1 (x : ℝ) : f(1 - x) = -f(1 + x)
axiom f_property (x : ℝ) : f(x + 4) = f(-x)

-- Proof problems:
-- 1. Prove f(3) = 0
theorem f_at_three : f 3 = 0 := sorry

-- 2. Prove f(1/2) = -f(5/2)
theorem f_half_equals_neg_five_halves : f (1/2) = -f (5/2) := sorry

-- 3. Prove f has a period of 8
theorem f_periodic : ∀ x : ℝ, f (x + 8) = f x := sorry

end f_at_three_f_half_equals_neg_five_halves_f_periodic_l282_282590


namespace function_value_range_l282_282428

theorem function_value_range :
  (∀ x : ℝ, -1 ≤ sin x ∧ sin x ≤ 1 → 
  ∃ y : ℝ, y = cos x ^ 2 + sin x - 1 ∧ y ∈ set.Icc (-2 : ℝ) (1 / 4)) := 
sorry

end function_value_range_l282_282428


namespace average_cardinality_of_subsets_l282_282026

theorem average_cardinality_of_subsets (n : ℕ) :
  (∑ A in Finset.powerset (Finset.range n), A.card) / 2^n = n / 2 := 
by
  sorry

end average_cardinality_of_subsets_l282_282026


namespace constant_term_in_expansion_l282_282662

theorem constant_term_in_expansion :
  let f := (x - (2 / x^2))
  let expansion := f^9
  ∃ c: ℤ, expansion = c ∧ c = -672 :=
sorry

end constant_term_in_expansion_l282_282662


namespace number_of_terms_in_arithmetic_sequence_l282_282289

theorem number_of_terms_in_arithmetic_sequence : 
  ∀ (a d l : ℕ), a = 20 → d = 5 → l = 150 → 
  ∃ n : ℕ, l = a + (n - 1) * d ∧ n = 27 :=
by
  intros a d l ha hd hl
  use 27
  rw [ha, hd, hl]
  sorry

end number_of_terms_in_arithmetic_sequence_l282_282289


namespace teachers_count_l282_282078

def total_students : Nat := 10 + 10
def total_chaperones : Nat := 5
def total_individuals (teachers : Nat) : Nat := total_students + total_chaperones + teachers
def individuals_who_left : Nat := 10 + 2
def remaining_individuals (teachers : Nat) : Nat := total_individuals teachers - individuals_who_left

theorem teachers_count : ∃ teachers : Nat, remaining_individuals teachers = 15 ∧ teachers = 2 :=
by
  use 2
  unfold remaining_individuals total_individuals total_students total_chaperones individuals_who_left
  simp
  sorry

end teachers_count_l282_282078


namespace geometric_sequence_sum_l282_282188

theorem geometric_sequence_sum :
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  (a * (1 - r^n) / (1 - r)) = 3280 / 6561 :=
by {
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  calc
  (a * (1 - r^n) / (1 - r)) = (1/3 * (1 - (1/3)^8) / (1 - 1/3)) : by rw a; rw r
  ... = 3280 / 6561 : sorry
}

end geometric_sequence_sum_l282_282188


namespace Bruce_bought_8_kg_of_grapes_l282_282115

-- Defining the conditions
def rate_grapes := 70
def rate_mangoes := 55
def weight_mangoes := 11
def total_paid := 1165

-- Result to be proven
def cost_mangoes := rate_mangoes * weight_mangoes
def total_cost_grapes (G : ℕ) := rate_grapes * G
def total_cost (G : ℕ) := (total_cost_grapes G) + cost_mangoes

theorem Bruce_bought_8_kg_of_grapes (G : ℕ) (h : total_cost G = total_paid) : G = 8 :=
by
  sorry  -- Proof omitted

end Bruce_bought_8_kg_of_grapes_l282_282115


namespace smallest_third_term_arith_geo_seq_l282_282494

theorem smallest_third_term_arith_geo_seq :
  ∃ (a b c : ℝ) (d : ℕ), 
  a = 9 ∧ b = 9 + d ∧ c = 9 + 2 * d ∧
  ((14 + d)^2 = 9 * (34 + 2 * d)) ∧
  (b₁ = 14 + d) ∧ (c₁ = 34 + 2 * d) ∧
  (9, b₁, c₁) forms_geometric_progression ∧
  smallest_third_term c₁ = -3 :=
begin
  -- Proof will go here
  sorry
end

end smallest_third_term_arith_geo_seq_l282_282494


namespace monotonic_interval_l282_282234

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.sin (2 * x + φ)

theorem monotonic_interval (φ : ℝ) (k : ℤ) :
  f (Real.pi / 12) φ - f (-5 * Real.pi / 12) φ = 2 →
  f.monotonic_increasing_on (Set.Icc (k * Real.pi - 5 * Real.pi / 12) (k * Real.pi + Real.pi / 12)) :=
sorry

end monotonic_interval_l282_282234


namespace EFZY_cyclic_l282_282851

-- Definitions related to the problem setup
variables {A B C D E F X Y Z : Point} (incircle_ABC : Circle)
variables [triangle : Triangle A B C]
variables [incircle_ABC.touch_points : IncircleTouchPoints incircle_ABC A B C D E F]
variables [point_X : InsideTriangle X A B C]
variables [incircle_XBC : IncircleTouchPoints (incircle X B C) X B C D Y Z]

-- The theorem statement
theorem EFZY_cyclic : CyclicQuadrilateral E F Z Y :=
sorry

end EFZY_cyclic_l282_282851


namespace circle_intersection_probability_l282_282777

theorem circle_intersection_probability :
  let R := 2;
  let lower_bound := 0;
  let upper_bound := 3;
  let sqrt7 := Real.sqrt 7;
  let intersection_condition (C_X D_X : ℝ) := abs (C_X - D_X) ≤ sqrt7;
  let uniform_dist (a b : ℝ) := if lower_bound ≤ a ∧ a ≤ upper_bound ∧ 
                                 lower_bound ≤ b ∧ b ≤ upper_bound then 1 / 9 else 0;
  ∫ x in set.Icc lower_bound upper_bound, 
    ∫ y in set.Icc lower_bound upper_bound, 
      uniform_dist x y * indicator (λ p : ℝ × ℝ, intersection_condition p.1 p.2) (x, y)
  = 2 * sqrt7 / 3 := sorry

end circle_intersection_probability_l282_282777


namespace force_magnitudes_ratio_l282_282757

theorem force_magnitudes_ratio (a d : ℝ) (h1 : (a + 2 * d)^2 = a^2 + (a + d)^2) :
  ∃ k : ℝ, k > 0 ∧ (a + d) = a * (4 / 3) ∧ (a + 2 * d) = a * (5 / 3) :=
by
  sorry

end force_magnitudes_ratio_l282_282757


namespace find_x_values_l282_282165

theorem find_x_values (x : ℝ) (h : x > 9) : 
  (sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) ↔ (x ≥ 18) :=
sorry

end find_x_values_l282_282165


namespace point_in_second_quadrant_l282_282959

theorem point_in_second_quadrant (i : ℂ) (hi : i = complex.I) :
  let z := i * (1 + i) in
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end point_in_second_quadrant_l282_282959


namespace sum_of_reciprocal_products_l282_282745

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def first_term (a : ℕ → ℝ) :=
  a 1 = 2

def condition (a : ℕ → ℝ) :=
  (a 3)^2 = (a 1) * (a 7)

theorem sum_of_reciprocal_products (d : ℝ) (a : ℕ → ℝ)
  (h1 : first_term a)
  (h2 : arithmetic_sequence a d)
  (h3 : d ≠ 0)
  (h4 : condition a) :
  (∑ n in range 2019, 1 / (a n * a (n + 1))) = 2019 / 4042 :=
sorry

end sum_of_reciprocal_products_l282_282745


namespace convex_functions_exist_l282_282472

noncomputable def exponential_function (x : ℝ) : ℝ :=
  4 - 5 * (1 / 2) ^ x

noncomputable def inverse_tangent_function (x : ℝ) : ℝ :=
  (10 / Real.pi) * Real.arctan x - 1

theorem convex_functions_exist :
  ∃ (f1 f2 : ℝ → ℝ),
    (∀ x, 0 < x → f1 x = exponential_function x) ∧
    (∀ x, 0 < x → f2 x = inverse_tangent_function x) ∧
    (∀ x, 0 < x → f1 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x, 0 < x → f2 x ∈ Set.Ioo (-1 : ℝ) 4) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f1 x1 + f1 x2 < 2 * f1 ((x1 + x2) / 2)) ∧
    (∀ x1 x2 : ℝ, 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 →
      f2 x1 + f2 x2 < 2 * f2 ((x1 + x2) / 2)) :=
sorry

end convex_functions_exist_l282_282472


namespace mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l282_282178

-- Definitions for molar masses used in calculations
def molar_mass_Ca := 40.08
def molar_mass_O := 16.00
def molar_mass_H := 1.01
def molar_mass_Na := 22.99
def molar_mass_C := 12.01
def molar_mass_K := 39.10
def molar_mass_S := 32.07

-- Molar masses of the compounds
def molar_mass_CaOH2 := molar_mass_Ca + 2 * molar_mass_O + 2 * molar_mass_H
def molar_mass_Na2CO3 := 2 * molar_mass_Na + molar_mass_C + 3 * molar_mass_O
def molar_mass_K2SO4 := 2 * molar_mass_K + molar_mass_S + 4 * molar_mass_O

-- Mass of O in each compound
def mass_O_CaOH2 := 2 * molar_mass_O
def mass_O_Na2CO3 := 3 * molar_mass_O
def mass_O_K2SO4 := 4 * molar_mass_O

-- Mass percentages of O in each compound
def mass_percent_O_CaOH2 := (mass_O_CaOH2 / molar_mass_CaOH2) * 100
def mass_percent_O_Na2CO3 := (mass_O_Na2CO3 / molar_mass_Na2CO3) * 100
def mass_percent_O_K2SO4 := (mass_O_K2SO4 / molar_mass_K2SO4) * 100

theorem mass_percent_O_CaOH2_is_correct :
  mass_percent_O_CaOH2 = 43.19 := by sorry

theorem mass_percent_O_Na2CO3_is_correct :
  mass_percent_O_Na2CO3 = 45.29 := by sorry

theorem mass_percent_O_K2SO4_is_correct :
  mass_percent_O_K2SO4 = 36.73 := by sorry

end mass_percent_O_CaOH2_is_correct_mass_percent_O_Na2CO3_is_correct_mass_percent_O_K2SO4_is_correct_l282_282178


namespace tile_replacement_impossible_l282_282840

theorem tile_replacement_impossible :
  ¬ (∃ T : (ℕ × ℕ) → bool, 
       (∀ x y, T (x+2, y) = T (x, y) ∧ T (x, y+2) = T (x, y)) ∧ -- Tiling pattern condition
       (∀ x y, T (x, y) = ff → (T (x+1, y) = ff ∧ T (x, y+1) = ff) ∨ 
                           (T (x, y+1) = ff ∨ T (x, y+2) = ff) ∨
                           (T (x+1, y) = ff ∨ T (x+2, y) = ff)) ∧ -- Condition for 2x2 and 1x4 tiles
       ∃ (x y : ℕ), T (x, y) ≠ T (x+1, y) ∧ T (x, y) ≠ T (x, y+1)) → false := sorry

end tile_replacement_impossible_l282_282840


namespace integral_f_eq_4_l282_282598

def f (x : ℝ) : ℝ := if x < 0 then 1 else Real.cos x

theorem integral_f_eq_4 : ∫ x in (-3 : ℝ)..(Real.pi / 2), f x = 4 := by
  sorry

end integral_f_eq_4_l282_282598


namespace number_of_extremum_points_of_f_l282_282951

noncomputable def f (x : ℝ) : ℝ :=
if x ≤ 0 then (x + 1)^3 * Real.exp (x + 1) else (-(x + 1))^3 * Real.exp (-(x + 1))

theorem number_of_extremum_points_of_f :
  (∃ x1 x2 x3 : ℝ, x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3 ∧ 
    ((f (x1 - epsilon) < f x1 ∧ f x1 > f (x1 + epsilon)) ∨ (f (x1 - epsilon) > f x1 ∧ f x1 < f (x1 + epsilon))) ∧
    ((f (x2 - epsilon) < f x2 ∧ f x2 > f (x2 + epsilon)) ∨ (f (x2 - epsilon) > f x2 ∧ f x2 < f (x2 + epsilon))) ∧
    ((f (x3 - epsilon) < f x3 ∧ f x3 > f (x3 + epsilon)) ∨ (f (x3 - epsilon) > f x3 ∧ f x3 < f (x3 + epsilon)))) :=
sorry

end number_of_extremum_points_of_f_l282_282951


namespace age_difference_l282_282732

-- Define the ages of Patrick, Michael, and Monica
variables (P M Mo : ℕ)

-- Define the ratios and sum of ages
def ratios_sum_of_ages (P M Mo : ℕ) : Prop :=
  (P:num) = 3 / 8 * (245 - Mo) ∧ 
  (Mo:num) = 5 / 8 * (245 - P) ∧ 
  P + M + Mo = 245

-- Prove the age difference
theorem age_difference (P M Mo : ℕ) (h : ratios_sum_of_ages P M Mo) : 
  (Mo:num) - (P:num) = 80 :=
sorry

end age_difference_l282_282732


namespace max_fully_connected_groups_mo_space_city_l282_282698

/-- 
MO Space City consists of 99 space stations, connected by tubular passages between any two stations. 
99 of these passages are two-way main roads, while all other passages are strictly one-way. 
A set of four stations is called a fully connected four-station group if it is possible to travel 
from any one of these stations to any other within the group through the passages.
Prove that the maximum number of fully connected four-station groups is 2052072.
-/
theorem max_fully_connected_groups_mo_space_city
  (n : ℕ) (hn : n = 99) (main_roads : ℕ) (hmain_roads : main_roads = 99)
  (one_way_passages : finset (fin n) → finset (fin n) → Prop)
  (two_way_roads : finset (fin n) → finset (fin n) → Prop)
  (connected_four_station_group : finset (fin n) → Prop) : 
  ∃ (max_groups : ℕ), max_groups = 2052072 :=
by
  sorry

end max_fully_connected_groups_mo_space_city_l282_282698


namespace log_101600_l282_282298

noncomputable def log10 : ℝ → ℝ := sorry

theorem log_101600 (h : log10 102 = 0.3010) : log10 101600 = 3.3010 :=
by
  have log1000 : log10 1000 = 3 := sorry
  calc
    log10 101600 = log10 (102 * 1000) : by rw [←mul_assoc]
    ... = log10 102 + log10 1000 : by sorry -- using log(a * b) = log(a) + log(b)
    ... = 0.3010 + 3 : by rw [h, log1000]
    ... = 3.3010 : by norm_num

end log_101600_l282_282298


namespace x1_sufficient_not_necessary_l282_282060

theorem x1_sufficient_not_necessary : (x : ℝ) → (x = 1 ↔ (x - 1) * (x + 2) = 0) ∧ ∀ x, (x = 1 ∨ x = -2) → (x - 1) * (x + 2) = 0 ∧ (∀ y, (y - 1) * (y + 2) = 0 → (y = 1 ∨ y = -2)) :=
by
  sorry

end x1_sufficient_not_necessary_l282_282060


namespace radius_of_sphere_l282_282079

theorem radius_of_sphere (r : ℝ) (c : ℝ) (θ : ℝ) :
  tan θ = 1 →
  tan θ = r / 12 →
  tan θ = c / 8 →
  r = 12 := by
  intros h1 h2 h3
  sorry

end radius_of_sphere_l282_282079


namespace surface_area_of_sphere_containing_P_ABCD_l282_282953

noncomputable def surface_area_sphere (R : ℝ) :=
  4 * Real.pi * R^2

theorem surface_area_of_sphere_containing_P_ABCD :
  ∃ (R : ℝ), 
    R^2 = 7 / 3 ∧
    surface_area_sphere R = 28 / 3 * Real.pi :=
by {
  -- problem definitions
  let PA := 2,
  let PD := 2,
  let AB := 2,
  let angle_APD := 60,
  let P := (1, 0, Real.sqrt 3),
  let A := (0, 0, 0),
  let D := (2, 0, 0),
  let B := (2, 2, 0),
  let C := (0, 2, 0),
  have h1 : PA = 2 := rfl,
  have h2 : PD = 2 := rfl,
  have h3 : angle_APD = 60 := rfl,
  have h4 : AB = 2 := rfl,
  havepoints_on_sphere : true := trivial, -- points baseline condition
  -- The actual proof of R and the sphere's surface area
  sorry
}

end surface_area_of_sphere_containing_P_ABCD_l282_282953


namespace sum_of_tangents_constant_l282_282666

theorem sum_of_tangents_constant (A B C A' X Y Z : Point) 
  (h1 : A ≠ B) (h2 : A ≠ C) (h3 : AB < AC)
  (h4 : tangent (excircle A A') BC at A')
  (h5 : X ∈ line_segment A A')
  (h6 : ¬ intersects_segment A' X ω)
  (h7 : tangents_from X to ω intersect BC at Y Z) :
  XY + XZ = c :=
sorry

end sum_of_tangents_constant_l282_282666


namespace find_x_l282_282198

noncomputable def x : ℝ := 24500 / 43.13380281690141

theorem find_x :
  (14 ^ 2) * (5 ^ 3) / x = 43.13380281690141 :=
by
  -- According to the condition provided
  have h1 : 14 ^ 2 = 196 := by norm_num
  have h2 : 5 ^ 3 = 125 := by norm_num
  have h3 : 196 * 125 = 24500 := by norm_num
  show 24500 / x = 43.13380281690141
  sorry

end find_x_l282_282198


namespace correct_proposition_l282_282502

-- Definitions of propositions given in the conditions
def propA : Prop := ¬ (∃ x : ℝ, x^2 - 1 < 0) ↔ ∀ x : ℝ, x^2 - 1 > 0
def propB : Prop := ¬ (if x = 3 then x^2 - 2*x - 3 = 0) ↔ (if x ≠ 3 then x^2 - 2*x - 3 ≠ 0)
def propC : Prop := ∃ q : Quadrilateral, q.isEquilateral ∧ ¬ q.isSquare
def propD : Prop := (∀ x y : ℝ, cos x = cos y → x = y) ↔ (∀ x y : ℝ, x ≠ y → cos x ≠ cos y)

-- The correct proposition according to the problem is B
theorem correct_proposition : propB := 
by 
  -- This is where you would normally include the proof steps 
  sorry

end correct_proposition_l282_282502


namespace part_a_part_b_part_c_l282_282680

-- Define A as a set of positive integers
def is_in_A : ℕ → Prop

-- Properties of set A
axiom A_property_i : ∀ n m : ℕ, is_in_A n → is_in_A m → is_in_A (n + m)
axiom A_property_ii : ∀ p : ℕ, prime p → (¬ ∀ n : ℕ, is_in_A n → p ∣ n)

-- Part (a): Show that such m1 and m2 exist
theorem part_a (n1 n2 : ℕ) (hn1 : is_in_A n1) (hn2 : is_in_A n2) (h_diff : n2 - n1 > 1):
  ∃ m1 m2 : ℕ, is_in_A m1 ∧ is_in_A m2 ∧ 0 < m2 - m1 ∧ m2 - m1 < n2 - n1 := sorry

-- Part (b): Show that there are two consecutive integers in A
theorem part_b : ∃ n : ℕ, is_in_A n ∧ is_in_A (n + 1) := sorry

-- Part (c): Show that if n0 and n0+1 are in A and n ≥ n0^2, then n ∈ A
theorem part_c (n0 n : ℕ) (h0 : is_in_A n0) (h1 : is_in_A (n0 + 1)) (h_geq : n ≥ n0^2) : 
  is_in_A n := sorry

end part_a_part_b_part_c_l282_282680


namespace sqrt_equation_solution_l282_282171

theorem sqrt_equation_solution (x : ℝ) (h₀ : 9 < x) 
  (h₁ : sqrt (x - 6 * sqrt (x - 9)) + 3 = sqrt (x + 6 * sqrt (x - 9)) - 3) : 
  x = 21 := 
sorry

end sqrt_equation_solution_l282_282171


namespace coefficient_x3y5_in_expansion_l282_282038

theorem coefficient_x3y5_in_expansion (x y : ℕ) : 
  @binomial 8 3 * x^3 * y^5 = x^3 * y^5 * 56 := 
sorry

end coefficient_x3y5_in_expansion_l282_282038


namespace forces_arithmetic_progression_ratio_l282_282754

theorem forces_arithmetic_progression_ratio 
  (a d : ℝ) 
  (h1 : ∀ (x y z : ℝ), IsArithmeticProgression x y z → x = a ∧ y = a + d ∧ z = a + 2d)
  (h2 : a^2 + (a + d)^2 = (a + 2d)^2)
  (h3 : a ≠ 0 ∧ d ≠ 0) :
  d / a = 1 / 3 :=
by
  sorry

end forces_arithmetic_progression_ratio_l282_282754


namespace ratio_of_radii_l282_282442

variables {R1 R2 : ℝ}
variables (l1 l2 : Type) [metric_space l1] [metric_space l2] (ω1 ω2 : Type)
variables {O1 : ℝ} {O2 : ℝ} (A B D E C : ℝ)
variables (S_BO1CO2 S_BO2E : ℝ)

-- Conditions
axiom tangent_l1_ω1 : tangent l1 ω1 A
axiom tangent_l2_ω1 : tangent l2 ω1 B
axiom tangent_l1_ω2 : tangent l1 ω2 D
axiom intersects_B_E_l2ω2 : intersects l2 ω2 B E
axiom intersects_C : intersects ω1 ω2 C
axiom O2_between_l1_l2 : between l1 O2 l2

axiom area_ratio : S_BO1CO2 / S_BO2E = 6 / 5

-- Goal: Prove that the ratio of the radii R2 / R1 is 7 / 6
theorem ratio_of_radii : R2 / R1 = 7 / 6 := by
  sorry

end ratio_of_radii_l282_282442


namespace find_x_add_inv_l282_282952

theorem find_x_add_inv (x : ℝ) (h : x^3 + 1 / x^3 = 110) : x + 1 / x = 5 :=
sorry

end find_x_add_inv_l282_282952


namespace sin_pi_div_two_add_A_eq_one_div_three_l282_282291

theorem sin_pi_div_two_add_A_eq_one_div_three (A : ℝ) (h : cos (π + A) = -1/3) : sin (π / 2 + A) = 1 / 3 :=
sorry

end sin_pi_div_two_add_A_eq_one_div_three_l282_282291


namespace correct_propositions_l282_282229

-- Define the propositions as conditions
def prop1 : Prop := ∀ (Π₁ Π₂ : Plane) (ℓ : Line), (Π₁ ∥ ℓ) → (Π₂ ∥ ℓ) → (Π₁ ∥ Π₂)
def prop2 : Prop := ∀ (Π₁ Π₂ Π₃ : Plane), (Π₁ ∥ Π₂) → (Π₂ ∥ Π₃) → (Π₁ ∥ Π₃)
def prop3 : Prop := ∀ (Π₁ Π₂ : Plane) (ℓ : Line), (Π₁ ⟂ ℓ) → (Π₂ ⟂ ℓ) → (Π₁ ∥ Π₂)
def prop4 : Prop := ∀ (ℓ₁ ℓ₂ ℓ : Line), (angle ℓ₁ ℓ = angle ℓ₂ ℓ) → (ℓ₁ ∥ ℓ₂)

-- We need to prove that only propositions 2 and 3 are correct
theorem correct_propositions : prop2 ∧ prop3 ∧ ¬prop1 ∧ ¬prop4 :=
by
  sorry

end correct_propositions_l282_282229


namespace common_points_one_l282_282139

def circle (x y : ℝ) : Prop := x^2 + y^2 = 16
def vertical_line (x : ℝ) : Prop := x = 4

theorem common_points_one :
  ∃! (p : ℝ × ℝ), circle p.1 p.2 ∧ vertical_line p.1 := by
  sorry

end common_points_one_l282_282139


namespace incorrect_a_for_quadratic_inequality_l282_282955

theorem incorrect_a_for_quadratic_inequality
  (a b c : ℝ)
  (h_sol_set : ∀ x : ℝ, x ∈ Ioo (-1 / 2) 2 → ax^2 + bx + c > 0):
  ¬(a > 0) :=
sorry

end incorrect_a_for_quadratic_inequality_l282_282955


namespace max_volume_pyramid_l282_282790

theorem max_volume_pyramid
  (AB AC: ℝ) (sin_BAC: ℝ)
  (h_AB: AB = 5)
  (h_AC: AC = 8)
  (h_sin_BAC: sin_BAC = 4/5) :
  ∃ (V: ℝ), 
  V = 10 * sqrt (137/3) :=
by
  sorry

end max_volume_pyramid_l282_282790


namespace geometric_sequence_sum_l282_282189

theorem geometric_sequence_sum :
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  (a * (1 - r^n) / (1 - r)) = 3280 / 6561 :=
by {
  let a := (1:ℚ)/3
  let r := (1:ℚ)/3
  let n := 8
  calc
  (a * (1 - r^n) / (1 - r)) = (1/3 * (1 - (1/3)^8) / (1 - 1/3)) : by rw a; rw r
  ... = 3280 / 6561 : sorry
}

end geometric_sequence_sum_l282_282189


namespace total_employees_with_advanced_degrees_l282_282647

theorem total_employees_with_advanced_degrees 
  (total_employees : ℕ) 
  (num_females : ℕ) 
  (num_males_college_only : ℕ) 
  (num_females_advanced_degrees : ℕ)
  (h1 : total_employees = 180)
  (h2 : num_females = 110)
  (h3 : num_males_college_only = 35)
  (h4 : num_females_advanced_degrees = 55) :
  ∃ num_employees_advanced_degrees : ℕ, num_employees_advanced_degrees = 90 :=
by
  have num_males := total_employees - num_females
  have num_males_advanced_degrees := num_males - num_males_college_only
  have num_employees_advanced_degrees := num_males_advanced_degrees + num_females_advanced_degrees
  use num_employees_advanced_degrees
  sorry

end total_employees_with_advanced_degrees_l282_282647


namespace abs_expression_eq_five_l282_282512

theorem abs_expression_eq_five : 
  abs (abs (-abs (-2 + 3) - 2) + 2) = 5 :=
by
  sorry -- proof omitted

end abs_expression_eq_five_l282_282512


namespace train_length_l282_282050

noncomputable def speed_in_mps (speed_in_kmph : ℝ) : ℝ :=
  speed_in_kmph * 1000 / 3600

def relative_speed (train_speed : ℝ) (bike_speed : ℝ) : ℝ :=
  speed_in_mps train_speed - speed_in_mps bike_speed

def length_of_train (relative_speed : ℝ) (time : ℕ) : ℝ :=
  relative_speed * time

theorem train_length :
  ∀ (train_speed bike_speed : ℝ) (time : ℕ),
    train_speed = 100 → bike_speed = 64 → time = 12 →
    length_of_train (relative_speed train_speed bike_speed) time = 120 :=
by
  intros train_speed bike_speed time h1 h2 h3
  simp [h1, h2, h3, relative_speed, length_of_train, speed_in_mps, mul_div_cancel]
  norm_num
  sorry

end train_length_l282_282050


namespace fourth_year_students_without_glasses_l282_282768

theorem fourth_year_students_without_glasses (total_students: ℕ) (x: ℕ) (y: ℕ) 
  (h1: total_students = 1152) 
  (h2: total_students = 8 * x - 32) 
  (h3: x = 148) 
  (h4: 2 * y + 10 = x) 
  : y = 69 :=
by {
sorry
}

end fourth_year_students_without_glasses_l282_282768


namespace sqrt_x_minus_1_meaningful_example_l282_282631

theorem sqrt_x_minus_1_meaningful_example :
  ∃ x : ℝ, x - 1 ≥ 0 ∧ x = 2 :=
by
  use 2
  split
  · linarith
  · refl

end sqrt_x_minus_1_meaningful_example_l282_282631


namespace min_Sn_is_630_T_n_correct_l282_282332

open Classical

variable {α : Type*} [LinearOrder α] 

def arithmetic_seq (a d : α) (n : ℕ) : α := a + n * d

noncomputable def S (a d : ℕ) (n : ℕ) : ℕ := (n * (2 * a + (n - 1) * d)) / 2

noncomputable def T (a d : ℕ) (n : ℕ) : ℕ := 
  if n ≤ 21 then 
    abs(S a (-d) n) 
  else 
    abs(S a (-d) n - 2 * S a (-d) 21)

theorem min_Sn_is_630 (a_15 a_16 a_17 a_9 : ℤ) (d : ℤ) (n : ℕ) (S_s : ℤ) :
  a_15 + a_16 + a_17 = -45 ∧
  a_9 = -36 ∧
  S_s = S (arithmetic_seq (-36) 3 n) 3 n →
  S_s = -630 := sorry

theorem T_n_correct (a_15 a_16 a_17 a_9 : ℤ) (d : ℤ) (n : ℕ) (T_s : ℤ) :
  a_15 + a_16 + a_17 = -45 ∧
  a_9 = -36 ∧ 
  T_s = T (arithmetic_seq (-36) 3 n) 3 n →
  (T_s = (if n ≤ 21 then -3/2 * n^2 + 123/2 * n else 3/2 * n^2 - 123/2 * n + 1260)) := sorry

end min_Sn_is_630_T_n_correct_l282_282332


namespace find_numbers_with_conditions_l282_282551

open Nat

-- Defining condition as a predicate
def has_four_divisors_with_three_not_exceeding_10 (n : ℕ) : Prop :=
  ∃ (d1 d2 d3 d4 : ℕ), 
    {d1, d2, d3, d4}.to_finset = (divisors n).to_finset ∧
    {d1, d2, d3, d4}.card = 4 ∧
    ∃ t1 t2 t3 : ℕ, {t1, t2, t3} ⊂ {d1, d2, d3, d4} ∧
    t1 ≤ 10 ∧ t2 ≤ 10 ∧ t3 ≤ 10

-- Problem statement
theorem find_numbers_with_conditions : 
  (finset.filter (λ n, has_four_divisors_with_three_not_exceeding_10 n) (finset.range 101)).card = 8 :=
sorry

end find_numbers_with_conditions_l282_282551


namespace locus_is_circle_l282_282406

open Complex

noncomputable def circle_center (a b : ℝ) : ℂ := Complex.ofReal (-a / (a^2 + b^2)) + Complex.I * (b / (a^2 + b^2))
noncomputable def circle_radius (a b : ℝ) : ℝ := 1 / Real.sqrt (a^2 + b^2)

theorem locus_is_circle (z0 z1 z : ℂ) (h1 : abs (z1 - z0) = abs z1) (h2 : z0 ≠ 0) (h3 : z1 * z = -1) :
  ∃ (a b : ℝ), z0 = Complex.ofReal a + Complex.I * b ∧
    (∃ c : ℂ, z = c ∧ 
      (c.re + a / (a^2 + b^2))^2 + (c.im - b / (a^2 + b^2))^2 = 1 / (a^2 + b^2)) := by
  sorry

end locus_is_circle_l282_282406


namespace an_plus_an_minus_1_eq_two_pow_n_l282_282922

def a_n (n : ℕ) : ℕ := sorry -- Placeholder for the actual function a_n

theorem an_plus_an_minus_1_eq_two_pow_n (n : ℕ) (h : n ≥ 4) : a_n (n - 1) + a_n n = 2^n := 
by
  sorry

end an_plus_an_minus_1_eq_two_pow_n_l282_282922


namespace alex_score_l282_282388

theorem alex_score (n : ℕ) (avg19 avg20 alex : ℚ)
  (h1 : n = 20)
  (h2 : avg19 = 72)
  (h3 : avg20 = 74)
  (h_totalscore19 : 19 * avg19 = 1368)
  (h_totalscore20 : 20 * avg20 = 1480)
  (h_alexscore : alex = 112) :
  alex = (1480 - 1368 : ℚ) := 
sorry

end alex_score_l282_282388


namespace derivative_of_function_l282_282409

open Real

theorem derivative_of_function : ∀ x : ℝ, deriv (λ x : ℝ, 2 * x + cos x) x = 2 - sin x :=
by
  intro x
  sorry

end derivative_of_function_l282_282409


namespace concurrency_of_bisectors_l282_282889

-- Define the problem settings and hypothesis
variables (A B C D P : Type) [convex_quad A B C D] [interior_point P A B C D]
variables (α : Type) (_ : ∠ PAD = α) (_ : ∠ PBA = 2 * α) (_ : ∠ DPA = 3 * α)
variables (_ : ∠ CBP = α) (_ : ∠ BAP = 2 * α) (_ : ∠ BPC = 3 * α)

-- Define the final Lean theorem with the specified concurrency conclusion
theorem concurrency_of_bisectors (A B C D P : Type)
  [convex_quad A B C D] [interior_point P A B C D]
  (α : Type) (h1 : ∠ PAD = α) (h2 : ∠ PBA = 2 * α) (h3 : ∠ DPA = 3 * α)
  (h4 : ∠ CBP = α) (h5 : ∠ BAP = 2 * α) (h6 : ∠ BPC = 3 * α) :
  concurrent (angle_bisector (∠ ADP)) 
             (angle_bisector (∠ PCB)) 
             (perpendicular_bisector (segment AB)) :=
by
  sorry

end concurrency_of_bisectors_l282_282889


namespace inverse_proportion_relationship_l282_282592

theorem inverse_proportion_relationship
  (y : ℝ → ℝ)
  (h₁ : y = λ x, 6 / x)
  (y1 y2 : ℝ)
  (h2 : y 2 = y1)
  (h3 : y 3 = y2) : y1 > y2 := by
  sorry

end inverse_proportion_relationship_l282_282592


namespace remainder_of_expression_l282_282460

theorem remainder_of_expression (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 :=
sorry

end remainder_of_expression_l282_282460


namespace find_a1_l282_282726

noncomputable theory

def cubic_polynomial (a_3 a_1 : ℝ) (x : ℝ) := a_3 * x^3 - x^2 + a_1 * x - 7

def roots_condition (α β γ : ℝ) : Prop :=
  ∃ t : ℝ, 
    t = 225 * α^2 / (α^2 + 7) ∧
    t = 144 * β^2 / (β^2 + 7) ∧
    t = 100 * γ^2 / (γ^2 + 7)

theorem find_a1
  (a_3 a_1 : ℝ) (α β γ : ℝ)
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0)
  (h_roots : has_root (cubic_polynomial a_3 a_1) α ∧ has_root (cubic_polynomial a_3 a_1) β ∧ has_root (cubic_polynomial a_3 a_1) γ)
  (h_cond : roots_condition α β γ) :
  a_1 = 130.6667 := 
sorry

end find_a1_l282_282726


namespace midpoint_velocity_of_rod_l282_282841

theorem midpoint_velocity_of_rod :
  ∀ (v1 v2 : ℝ), 
    (v1 = 5) →
    (v2 = 4) →
    let vertical_component_A := real.sqrt (v1^2 - v2^2) in
    let vertical_component_B := vertical_component_A / 2 in
    let vO := real.sqrt (v2^2 + vertical_component_B^2) in
    vO = real.sqrt 18.25 :=
by
  intros v1 v2 H1 H2
  let vertical_component_A := real.sqrt (v1^2 - v2^2)
  let vertical_component_B := vertical_component_A / 2
  let vO := real.sqrt (v2^2 + vertical_component_B^2)
  have H3 : vO = real.sqrt (16 + 2.25) := sorry
  -- Completing the proof here is not necessary as mentioned
  sorry

end midpoint_velocity_of_rod_l282_282841


namespace length_of_second_square_l282_282876

-- Define conditions as variables
def Area_flag := 135
def Area_square1 := 40
def Area_square3 := 25

-- Define the length variable for the second square
variable (L : ℕ)

-- Define the area of the second square in terms of L
def Area_square2 : ℕ := 7 * L

-- Lean statement to be proved
theorem length_of_second_square :
  Area_square1 + Area_square2 L + Area_square3 = Area_flag → L = 10 :=
by sorry

end length_of_second_square_l282_282876


namespace sum_geometric_sequence_first_eight_terms_l282_282185

theorem sum_geometric_sequence_first_eight_terms :
  let a_0 := (1 : ℚ) / 3
  let r := (1 : ℚ) / 3
  let n := 8
  let S_n := a_0 * (1 - r^n) / (1 - r)
  S_n = 6560 / 19683 := 
by
  sorry

end sum_geometric_sequence_first_eight_terms_l282_282185


namespace triangle_inequality_property_l282_282684

-- Define basic geometry entities
structure Triangle (α : Type _) [Add α] [Neg α] [TopologicalSpace α] :=
(A B C : α)

variable {α : Type _} [Add α] [Neg α] [TopologicalSpace α] [MetricSpace α]

-- Define the centroid of a triangle
def centroid (T : Triangle α) : α :=
  (T.A + T.B + T.C) / 3

-- Define the distance from centroid to each vertex
def s1 (T : Triangle α) : α :=
  dist (centroid T) T.A + dist (centroid T) T.B + dist (centroid T) T.C

-- Define the perimeter of the triangle
def s2 (T : Triangle α) : α :=
  dist T.A T.B + dist T.B T.C + dist T.C T.A

-- Define the main theorem to be proven
theorem triangle_inequality_property (T : Triangle α) : 
  s1 T > (2 : α) / 3 * s2 T ∧ s1 T < s2 T := 
sorry

end triangle_inequality_property_l282_282684


namespace no_solutions_eq_l282_282799

theorem no_solutions_eq (x y : ℝ) : (x + y)^2 ≠ x^2 + y^2 + 1 :=
by sorry

end no_solutions_eq_l282_282799


namespace equivalent_proof_problem_l282_282052

theorem equivalent_proof_problem (x : ℤ) (h : (x - 5) / 7 = 7) : (x - 14) / 10 = 4 :=
by
  sorry

end equivalent_proof_problem_l282_282052


namespace no_solutions_for_10_tuples_l282_282912

theorem no_solutions_for_10_tuples (x : ℕ → ℝ) :
  (∑ i in finset.range 10, (x i - x (i + 1))^2) = (1 - x 0)^2 + (∑ i in finset.range 1 9, (x i - x (i + 1))^2) + x 10^2 ∧ 
  ((x 0 - x 1)^2 + (x 1 - x 2)^2 + ... + (x 9 - x10)^2 + x10^2 = 3/11 ) → x = 0 := 
begin
  sorry
end

end no_solutions_for_10_tuples_l282_282912


namespace hex_1F4B_is_8011_l282_282893

def hex_to_dec (s: String) : Nat :=
  s.foldl (λ acc c, acc * 16 + (c.toNat - if '0' ≤ c ∧ c ≤ '9' then '0'.toNat else if 'A' ≤ c ∧ c ≤ 'F' then 'A'.toNat - 10 else 'a'.toNat - 10)) 0

theorem hex_1F4B_is_8011 : hex_to_dec "1F4B" = 8011 := by
  -- Apply the conversion function to "1F4B" and equate it to 8011
  sorry

end hex_1F4B_is_8011_l282_282893


namespace area_difference_l282_282340

theorem area_difference (BC : ℝ) (hBC : BC = 8 * Real.sqrt 2) 
  (right_triangle_ABC : ∃ A B C, ∠B = ∠C ∧ is_right_triangle A B C) 
  (fits_in_square : ∃ s, BC = s * Real.sqrt 2) : 
  ∃ Δ, Δ = 32 := sorry

end area_difference_l282_282340


namespace candy_per_packet_l282_282875

-- Define the conditions as hypotheses
def bobby_weekly_candies (mon_to_fri_candies : ℕ) (sat_sun_candies : ℕ) : ℕ :=
  mon_to_fri_candies + sat_sun_candies

def total_candies_in_n_weeks (weekly_candies : ℕ) (n : ℕ) : ℕ :=
  weekly_candies * n

theorem candy_per_packet
  (mon_to_fri_candies_per_day : ℕ)
  (sat_sun_candies_per_day : ℕ)
  (days_mon_to_fri : ℕ)
  (days_weekend : ℕ)
  (num_weeks : ℕ)
  (total_packets : ℕ)
  (candies_per_packet : ℕ)
  (h1 : mon_to_fri_candies_per_day = 2)
  (h2 : sat_sun_candies_per_day = 1)
  (h3 : days_mon_to_fri = 5)
  (h4 : days_weekend = 2)
  (h5 : num_weeks = 3)
  (h6 : total_packets = 2)
  (h7 : candies_per_packet = (total_candies_in_n_weeks (bobby_weekly_candies (mon_to_fri_candies_per_day * days_mon_to_fri) (sat_sun_candies_per_day * days_weekend)) num_weeks) / total_packets) :
  candies_per_packet = 18 :=
sorry

end candy_per_packet_l282_282875


namespace smallest_result_l282_282521

theorem smallest_result : 
  (∃ a b c ∈ {2, 4, 6, 8, 10, 12} , a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (min ((a + b) * c) ((a + c) * b) ((b + c) * a)) = 20) :=
sorry

end smallest_result_l282_282521


namespace problem_solution_l282_282728

noncomputable def g : ℕ → ℕ :=
  λ x, if x = 1 then 4
       else if x = 2 then 6
       else if x = 3 then 2
       else if x = 4 then 1
       else 0  -- Assuming other values are not used

theorem problem_solution : 
  let a := 1, b := g(g a),
      c := 3, d := g(g c),
      e := 4, f := g(g e) in
  (a * b) + (c * d) + (e * f) = 35 :=
by
  have gb1 : g(g 1) = 1 := by repeat {unfold g};simp
  have gb3 : g(g 3) = 6 := by repeat {unfold g};simp
  have gb4 : g(g 4) = 4 := by repeat {unfold g};simp
  let a := 1, b := 1,
      c := 3, d := 6,
      e := 4, f := 4
  show 1 * 1 + 3 * 6 + 4 * 4 = 35 from by simp

end problem_solution_l282_282728


namespace percentage_failed_both_l282_282328

theorem percentage_failed_both 
    (p_h p_e p_p p_pe : ℝ)
    (h_p_h : p_h = 32)
    (h_p_e : p_e = 56)
    (h_p_p : p_p = 24)
    : p_pe = 12 := by 
    sorry

end percentage_failed_both_l282_282328


namespace proof_P_otimes_Q_l282_282685

-- Define the sets P and Q
def P : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }
def Q : Set ℝ := { x | 1 < x }

-- Define the operation ⊗ between sets
def otimes (P Q : Set ℝ) : Set ℝ := { x | x ∈ P ∪ Q ∧ x ∉ P ∩ Q }

-- Prove that P ⊗ Q = [0,1] ∪ (2, +∞)
theorem proof_P_otimes_Q :
  otimes P Q = {x | (0 ≤ x ∧ x ≤ 1) ∨ (2 < x)} :=
by
 sorry

end proof_P_otimes_Q_l282_282685


namespace num_common_divisors_60_90_l282_282624

theorem num_common_divisors_60_90 : 
  let n1 := 60
  let n2 := 90
  let gcd := 30 -- GCD calculated from prime factorizations
  let divisors_of_gcd := [1, 2, 3, 5, 6, 10, 15, 30]
  in divisors_of_gcd.length = 8 :=
by
  sorry

end num_common_divisors_60_90_l282_282624


namespace pieces_of_toast_bread_made_l282_282147

-- Definitions based on provided conditions
def loaf_original_slices : ℕ := 27
def andy_ate_twice : ℕ := 3 * 2
def emma_left_slices_after_toasting : ℕ := 1

-- Lean equivalent proof statement
theorem pieces_of_toast_bread_made : (loaf_original_slices - andy_ate_twice) // 2 = 10 :=
by
  -- Declaration of variables and constants used in the problem
  let slices_remaining := loaf_original_slices - andy_ate_twice
  let pieces_of_toast := slices_remaining // 2
  
  -- Assertion to be proved
  have h : pieces_of_toast = 10 :=
    by
      sorry  -- Proof goes here

  exact h

end pieces_of_toast_bread_made_l282_282147


namespace locus_centroid_parabola_l282_282691

variables (m : ℝ) (n : ℕ)

noncomputable def P_n_equation (m : ℝ) (n : ℕ) : ℝ → ℝ → Prop := 
  λ x y, y^2 = (m / (3^n)) * (x - (m / 4) * (1 - (1 / (3^n))))

theorem locus_centroid_parabola (m : ℝ) (n : ℕ) :
  ∃ (P_n : ℝ → ℝ → Prop), P_n = P_n_equation m n :=
begin
  use P_n_equation m n,
  sorry,
end

end locus_centroid_parabola_l282_282691


namespace difference_largest_smallest_solution_l282_282721

/-- Define the integer part of a real number. -/
def int_part (x : ℝ) : ℤ := intFloor x

/-- Define the cube root function. -/
noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

/-- Main proof statement. -/
theorem difference_largest_smallest_solution (x : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁^2 ≤ 2 * (int_part (cube_root x₁ + 0.5) + int_part (cube_root x₁)) ∧
                 x₂^2 ≤ 2 * (int_part (cube_root x₂ + 0.5) + int_part (cube_root x₂)) ∧
                 x₁² ≠ x₂²) →
  (1 : ℝ) = (1 - 0 : ℝ) :=
sorry

end difference_largest_smallest_solution_l282_282721


namespace option_C_holds_l282_282632

theorem option_C_holds (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a - b / a > b - a / b := 
  sorry

end option_C_holds_l282_282632


namespace base_area_of_cuboid_l282_282825

theorem base_area_of_cuboid (V h : ℝ) (hv : V = 144) (hh : h = 8) : ∃ A : ℝ, A = 18 := by
  sorry

end base_area_of_cuboid_l282_282825


namespace sum_computation_l282_282123

noncomputable def ceil_minus_floor (x : ℝ) : ℝ :=
  if x ≠ ⌊x⌋ then 1 else 0

def is_power_of_three (n : ℕ) : Prop :=
  ∃ (j : ℕ), 3^j = n

theorem sum_computation :
  (∑ k in Finset.range 501, k * (ceil_minus_floor (Real.log k / Real.log 3))) = 124886 :=
by
  sorry

end sum_computation_l282_282123


namespace derivative_evaluation_l282_282408

def function_f : ℝ → ℝ := λ x, (1 / 3) * x^3 - ( by exact f'' (-1)) * x^2 - x

theorem derivative_evaluation :
  (∀ (f : ℝ → ℝ), f'' : ℝ → ℝ → ℝ) → -- Assuming existence of f'' that could be derived from f
  f''' 1 = 0 :=
by
  sorry

end derivative_evaluation_l282_282408
