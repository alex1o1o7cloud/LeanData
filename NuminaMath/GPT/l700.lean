import Mathlib

namespace maple_trees_required_l700_700535

/-- Prove that the number of maple trees Gloria needs to sell is 23 given the conditions. -/
theorem maple_trees_required : 
  ∀ (cabin_price cash amount_left_after_purchase num_cypress_trees num_pine_trees earnings_per_cypress earnings_per_maple earnings_per_pine : ℤ),
  cabin_price = 129000 ->
  cash = 150 ->
  amount_left_after_purchase = 350 ->
  num_cypress_trees = 20 ->
  num_pine_trees = 600 ->
  earnings_per_cypress = 100 ->
  earnings_per_maple = 300 ->
  earnings_per_pine = 200 ->
  let total_needed := cabin_price - cash - amount_left_after_purchase in
  let total_from_cypress_pine := num_cypress_trees * earnings_per_cypress + num_pine_trees * earnings_per_pine in
  let remaining_needed := total_needed - total_from_cypress_pine in
  let maple_trees := (remaining_needed + earnings_per_maple - 1) / earnings_per_maple in -- rounding up the division
  maple_trees = 23 :=
by
  intros cabin_price cash amount_left_after_purchase num_cypress_trees num_pine_trees earnings_per_cypress earnings_per_maple earnings_per_pine
  intros
  sorry

end maple_trees_required_l700_700535


namespace number_of_pencils_l700_700970

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l700_700970


namespace fraction_sum_reciprocal_l700_700329

theorem fraction_sum_reciprocal :
  10 * (3/12 + 4/12 + 6/12)⁻¹ = 40 / 3 :=
by
  sorry

end fraction_sum_reciprocal_l700_700329


namespace count_integers_500_to_700_with_digit_sum_18_l700_700547

theorem count_integers_500_to_700_with_digit_sum_18 :
  (∃ f : ℕ → ℕ, (∀ n, 500 ≤ n ∧ n < 700 → (n.digitSum = 18 → f n = 1) ∧ (n.digitSum ≠ 18 → f n = 0)) ∧ 
  (∑ n in finset.Icc 500 699, f n) = 25) :=
by
  sorry

end count_integers_500_to_700_with_digit_sum_18_l700_700547


namespace trains_meet_l700_700324

structure Train (departure_time : ℕ) (speed : ℕ) 

structure Station (distance_from_start : ℕ)

def meet_time (trainA trainB : Train) (stationA stationB : Station) (stationC : Station) : ℕ :=
  -- lean function to compute the meeting time of two trains
  sorry

theorem trains_meet (trainA : Train) (trainB : Train) 
  (stationA stationB : Station) (stationC : Station) 
  (hA_dep : trainA.departure_time = 945)
  (hA_speed : trainA.speed = 60)
  (hB_dep : trainB.departure_time = 1000)
  (hB_speed : trainB.speed = 80)
  (hDist : stationA.distance_from_start + stationB.distance_from_start = 300)
  (hC_dist : stationC.distance_from_start = 150) : 
  meet_time trainA trainB stationA stationB stationC = 1108 :=
sorry

end trains_meet_l700_700324


namespace symmetric_line_equation_l700_700303

theorem symmetric_line_equation (x y : ℝ) :
  (∃ x y : ℝ, 3 * x + 4 * y = 2) →
  (4 * x + 3 * y = 2) :=
by
  intros h
  sorry

end symmetric_line_equation_l700_700303


namespace min_n_prob_lt_1_1005_l700_700030

theorem min_n_prob_lt_1_1005 : 
  ∃ n : ℕ, (1 ≤ n ∧ n ≤ 1005) ∧ (∀ m : ℕ, (1 ≤ m ∧ m < n) → 1 / (2 * m + 1) > 1 / 2010) ∧ 1 / (2 * n + 1) < 1 / 2010 := 
by {
  have h₁ : 1 / (2 * 1005 + 1) < 1 / 2010 := by norm_num,
  use 1005,
  split,
  exact ⟨1, by norm_num⟩,
  split,
  intros m h2_low h2_high,
  have h2m : 2 * m + 1 ≤ 2010 := by linarith,
  rw ← one_div_lt_one_div (by linarith) (by norm_num) at h2m,
  exact h2m,
  exact h₁,
  sorry
}

end min_n_prob_lt_1_1005_l700_700030


namespace percentage_selected_in_state_B_l700_700996

theorem percentage_selected_in_state_B (appeared: ℕ) (selectedA: ℕ) (selected_diff: ℕ)
  (percentage_selectedA: ℝ)
  (h1: appeared = 8100)
  (h2: percentage_selectedA = 6.0)
  (h3: selectedA = appeared * (percentage_selectedA / 100))
  (h4: selected_diff = 81)
  (h5: selectedB = selectedA + selected_diff) :
  ((selectedB : ℝ) / appeared) * 100 = 7 := 
  sorry

end percentage_selected_in_state_B_l700_700996


namespace floor_and_ceil_sum_l700_700036

theorem floor_and_ceil_sum : ⌊1.999⌋ + ⌈3.001⌉ = 5 := 
by
  sorry

end floor_and_ceil_sum_l700_700036


namespace smallest_k_for_product_l700_700096

theorem smallest_k_for_product (k : ℕ) :
  (let num := int.of_nat (10^k - 1) * 7 in
   let prod := num * 9 in
   nat.digits 10 prod |>.sum = 900)
  → k = 100 :=
by
  -- Calculation pattern and logic follows as per solution above
  sorry

end smallest_k_for_product_l700_700096


namespace full_day_students_l700_700002

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l700_700002


namespace total_peaches_is_85_l700_700648

-- Definitions based on conditions
def initial_peaches : ℝ := 61.0
def additional_peaches : ℝ := 24.0

-- Statement to prove
theorem total_peaches_is_85 :
  initial_peaches + additional_peaches = 85.0 := 
by sorry

end total_peaches_is_85_l700_700648


namespace largest_angle_isosceles_triangle_l700_700575

theorem largest_angle_isosceles_triangle (A B C : ℕ) 
  (h_isosceles : A = B) 
  (h_base_angle : A = 50) : 
  max A (max B C) = 80 := 
by 
  -- proof is omitted  
  sorry

end largest_angle_isosceles_triangle_l700_700575


namespace find_f_2017_div_2_l700_700117

noncomputable def is_odd_function {X Y : Type*} [AddGroup X] [AddGroup Y] (f : X → Y) :=
  ∀ x : X, f (-x) = -f x

noncomputable def is_periodic_function {X Y : Type*} [AddGroup X] [AddGroup Y] (p : X) (f : X → Y) :=
  ∀ x : X, f (x + p) = f x

noncomputable def f : ℝ → ℝ 
| x => if -1 ≤ x ∧ x ≤ 0 then x * x + x else sorry

theorem find_f_2017_div_2 : f (2017 / 2) = 1 / 4 :=
by
  have h_odd : is_odd_function f := sorry
  have h_period : is_periodic_function 2 f := sorry
  unfold f
  sorry

end find_f_2017_div_2_l700_700117


namespace range_of_a_range_of_m_l700_700524

variable (a m : ℝ)
variable (f g : ℝ → ℝ)

-- Define the function f(x) = x^2 + ax + 2
def f (x : ℝ) : ℝ := x^2 + a * x + 2

-- Define the function g(x) = -x + m
def g (x : ℝ) : ℝ := -x + m

namespace Proof1

-- Condition: For any x ∈ [-1, 1], the inequality f(x) ≤ 2a(x-1) + 4 always holds
axiom condition_f (x : ℝ) (h : x ∈ Icc (-1 : ℝ) 1) : f x ≤ 2 * a * (x - 1) + 4

-- Theorem: The range of real number a is (-∞, 1/3]
theorem range_of_a (a : ℝ) : (∀ (x : ℝ), x ∈ Icc (-1 : ℝ) 1 → f x ≤ 2 * a * (x - 1) + 4) → a ≤ 1 / 3 :=
sorry

end Proof1

namespace Proof2

-- Define the function f(x) = x^2 - 3x + 2
def f (x : ℝ) : ℝ := x^2 - 3 * x + 2

-- Condition: For any x1 ∈ [1, 4], there exists x2 ∈ (1, 8) such that f(x1) = g(x2)
axiom condition_g (x1 : ℝ) (h1 : x1 ∈ Icc (1 : ℝ) 4) : ∃ (x2 : ℝ), x2 ∈ Ioo (1 : ℝ) 8 ∧ f x1 = g x2

-- Theorem: The range of real number m is (7, 31/4)
theorem range_of_m (m : ℝ) : 
  (∀ (x1 : ℝ), x1 ∈ Icc (1 : ℝ) 4 → ∃ (x2 : ℝ), x2 ∈ Ioo (1 : ℝ) 8 ∧ f x1 = g x2) →
  m > 7 ∧ m < 31 / 4 :=
sorry

end Proof2

end range_of_a_range_of_m_l700_700524


namespace unique_solution_l700_700808

theorem unique_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x) * f(y * f(x) - 1) = x^2 * f(y) - f(x))
  → (∀ x : ℝ, f(x) = x) :=
by
  assume h : ∀ x y : ℝ, f(x) * f(y * f(x) - 1) = x^2 * f(y) - f(x)
  sorry

end unique_solution_l700_700808


namespace parabola_translation_l700_700192

theorem parabola_translation : 
  ∀ x y : ℝ, y = (x + 2)^2 → ∃ x' y' : ℝ, y' = (x' + 2 - 2)^2 ∧ (x', y') = (0, 0) :=
by
  intros x y h
  use 0, 0
  rw h
  simp
  sorry

end parabola_translation_l700_700192


namespace number_of_divisors_l700_700862

theorem number_of_divisors (count_n : ℕ) : count_n = 5 :=
  let divisors := [1, 2, 4, 7, 14, 28]
  let valid_ns :=
    divisors.filter (λ d, let n := d - 1 in n > 0 ∧ 14 * n % (n * (n + 1) / 2) = 0)
  valid_ns.length = 5

end number_of_divisors_l700_700862


namespace least_three_digit_7_heavy_l700_700774

-- Define what it means to be a 7-heavy number
def is_7_heavy (n : ℕ) : Prop :=
  n % 7 > 4

-- Define the property of being three-digit
def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

-- The statement to prove
theorem least_three_digit_7_heavy : ∃ n, is_7_heavy n ∧ is_three_digit n ∧ ∀ m, is_7_heavy m ∧ is_three_digit m → n ≤ m :=
begin
  use [104],
  split,
  { -- Proof that 104 is 7-heavy
    show is_7_heavy 104,
    simp [is_7_heavy], -- Calculation: 104 % 7 = 6 which is > 4
    norm_num,
  },
  split,
  { -- Proof that 104 is a three-digit number
    show is_three_digit 104,
    simp [is_three_digit],
    norm_num,
  },
  { -- Proof that 104 is the smallest 7-heavy three-digit number
    intros m hm,
    cases hm with hm1 hm2,
    suffices : 104 ≤ m,
    exact this,
    calc 104 ≤ 100 + 7 - 1 : by norm_num
        ... ≤ m            : by linarith [hm2.left, hm2.right],
    sorry,
  }
sorry

end least_three_digit_7_heavy_l700_700774


namespace mrs_hilt_more_l700_700633

-- Define the values of the pennies, nickels, and dimes.
def value_penny : ℝ := 0.01
def value_nickel : ℝ := 0.05
def value_dime : ℝ := 0.10

-- Define the count of coins Mrs. Hilt has.
def mrs_hilt_pennies : ℕ := 2
def mrs_hilt_nickels : ℕ := 2
def mrs_hilt_dimes : ℕ := 2

-- Define the count of coins Jacob has.
def jacob_pennies : ℕ := 4
def jacob_nickels : ℕ := 1
def jacob_dimes : ℕ := 1

-- Calculate the total amount of money Mrs. Hilt has.
def mrs_hilt_total : ℝ :=
  mrs_hilt_pennies * value_penny
  + mrs_hilt_nickels * value_nickel
  + mrs_hilt_dimes * value_dime

-- Calculate the total amount of money Jacob has.
def jacob_total : ℝ :=
  jacob_pennies * value_penny
  + jacob_nickels * value_nickel
  + jacob_dimes * value_dime

-- Prove that Mrs. Hilt has $0.13 more than Jacob.
theorem mrs_hilt_more : mrs_hilt_total - jacob_total = 0.13 := by
  sorry

end mrs_hilt_more_l700_700633


namespace ordered_triple_l700_700221

theorem ordered_triple (a b c : ℝ) (h1 : 4 < a) (h2 : 4 < b) (h3 : 4 < c) 
  (h_eq : (a + 3)^2 / (b + c - 3) + (b + 5)^2 / (c + a - 5) + (c + 7)^2 / (a + b - 7) = 45) 
  : (a, b, c) = (12, 10, 8) :=
  sorry

end ordered_triple_l700_700221


namespace solve_y_l700_700653

theorem solve_y : 
  ∃ y : ℚ, (5 * y + 2) / (6 * y - 3) = 3 / 4 ∧ y = -17 / 2 := 
by
  use -17 / 2
  split
  · field_simp
    linarith
  · rfl

end solve_y_l700_700653


namespace distance_to_other_focus_l700_700017

theorem distance_to_other_focus 
  (x y : ℝ) 
  (h_ellipse : (x^2 / 16 + y^2 / 12 = 1))
  (M : ℝ × ℝ)
  (h_point_on_ellipse : (M.1^2 / 16 + M.2^2 / 12 = 1))
  (distance_to_focus_1 : ℝ)
  (h_distance_1 : distance_to_focus_1 = 3):
  let a := 4 in
  let total_distance := 2 * a in
  let distance_to_other_focus := total_distance - distance_to_focus_1 in
  distance_to_other_focus = 5 :=
by
  sorry

end distance_to_other_focus_l700_700017


namespace total_miles_for_15_dollars_l700_700316

-- Definitions from conditions
def initial_fare : ℝ := 3.0 -- Initial fare for first 0.75 miles
def additional_fare_rate : ℝ := 0.25 / 0.05 -- Additional fare rate per 0.05 mile
def tip : ℝ := 3.0 -- Tip amount
def total_amount : ℝ := 15.0 -- Total amount to be spent
def first_miles : ℝ := 0.75 -- The first mile distance covered by initial fare

-- Proof statement: Prove that the total number of miles for $15 including a $3 tip is 2.55
theorem total_miles_for_15_dollars : 
  let x := 2.55 in
  total_amount - tip = initial_fare + additional_fare_rate * (x - first_miles) :=
  by
    sorry

end total_miles_for_15_dollars_l700_700316


namespace factorize_x_squared_minus_four_l700_700450

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l700_700450


namespace car_speed_l700_700738

theorem car_speed (distance time : ℝ) (h_distance : distance = 275) (h_time : time = 5) : (distance / time = 55) :=
by
  sorry

end car_speed_l700_700738


namespace num_ways_to_arrange_PERCEPTION_l700_700833

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l700_700833


namespace centroid_plane_intersection_l700_700621

theorem centroid_plane_intersection (α β γ p q r : ℝ) :
  (p = α / 3) →
  (q = β / 3) →
  (r = γ / 3) →
  (1 / (Real.sqrt ((1 / α ^ 2) + (1 / β ^ 2) + (1 / γ ^ 2))) = 2) →
  (1 / p ^ 2) + (1 / q ^ 2) + (1 / r ^ 2) = 2.25 :=
begin
  intros hp hq hr hd,
  have h1 : 1 / α ^ 2 + 1 / β ^ 2 + 1 / γ ^ 2 = 1 / 4,
  { rw [← Real.inv_inj],
    simp_rw [Real.inv_sqrt, Real.pow_sqrt, ← div_eq_inv_mul],
    rw hd,
    norm_num },
  have hp_q_r : ((1 / p ^ 2) + (1 / q ^ 2) + (1 / r ^ 2)) = (9 / (α^2)) + (9 / (β^2)) + (9 / (γ^2)),
  { rw [hp, hq, hr],
    field_simp [pow_two] },
  rw hp_q_r,
  rw h1,
  norm_num,
end

end centroid_plane_intersection_l700_700621


namespace election_votes_l700_700715

theorem election_votes (V : ℝ) 
    (h1 : ∃ c1 c2 : ℝ, c1 + c2 = V ∧ c1 = 0.60 * V ∧ c2 = 0.40 * V)
    (h2 : ∃ m : ℝ, m = 280 ∧ 0.60 * V - 0.40 * V = m) : 
    V = 1400 :=
by
  sorry

end election_votes_l700_700715


namespace length_YB_of_triangle_XYZ_l700_700566

noncomputable def triangle_XYZ (X : ℝ × ℝ) (Y : ℝ × ℝ) (Z : ℝ × ℝ) : Prop :=
  ∠ Z X Y = 60 ∧ ∠ X Y Z = 30 ∧ dist X Z = 2

def midpoint (A B : ℝ × ℝ) : ℝ × ℝ :=
  ( ( A.1 + B.1 ) / 2, ( A.2 + B.2 ) / 2 )

def perp (P Q R S : ℝ × ℝ) : Prop :=
  (Q.2 - P.2) * (S.2 - R.2) + (Q.1 - P.1) * (S.1 - R.1) = 0

noncomputable def has_length (P Q : ℝ × ℝ) (l : ℝ) : Prop :=
  dist P Q = l

theorem length_YB_of_triangle_XYZ :
  ∀ (X Y Z N D B : ℝ × ℝ), triangle_XYZ X Y Z →
    N = midpoint X Z →
    D.1 = 1 →
    D.2 = 0 →
    perp X D Y N →
    B.1 = 2 - D.1 →
    B.2 = 0 →
    has_length Y B (2 * real.sqrt 3 / 3) :=
by
  intros X Y Z N D B hXYZ hN hD1 hD2 hPerp hB1 hB2
  sorry

end length_YB_of_triangle_XYZ_l700_700566


namespace part_i_part_ii_l700_700460

-- Definition of Part (i)
theorem part_i (n : ℕ) (Hn : n > 1) (pieces : Finset (Fin n × Fin n)) (Hpieces : pieces.card = 2 * n) :
  ∃ (p1 p2 p3 p4 : Fin n × Fin n), 
  (p1 ∈ pieces ∧ p2 ∈ pieces ∧ p3 ∈ pieces ∧ p4 ∈ pieces) ∧ 
  ((p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2) ∨ 
   (p1.1 = p3.1 ∧ p2.1 = p4.1 ∧ p1.2 = p2.2 ∧ p3.2 = p4.2)) :=
begin
  sorry
end

-- Definition of Part (ii)
theorem part_ii (n : ℕ) (Hn : n > 1) :
  ∃ (pieces : Finset (Fin n × Fin n)), 
    pieces.card = 2 * n - 1 ∧ 
    ∀ (p1 p2 p3 p4 : Fin n × Fin n),
      p1 ∈ pieces → p2 ∈ pieces → p3 ∈ pieces → p4 ∈ pieces →
      ¬ ((p1.1 = p2.1 ∧ p3.1 = p4.1 ∧ p1.2 = p3.2 ∧ p2.2 = p4.2) ∨ 
         (p1.1 = p3.1 ∧ p2.1 = p4.1 ∧ p1.2 = p2.2 ∧ p3.2 = p4.2)) :=
begin
  sorry
end

end part_i_part_ii_l700_700460


namespace triangle_constructibility_l700_700020

noncomputable def construct_triangle (r AM MS : ℝ) : Prop :=
  ∃ A B C M S O : Point, 
    Circumcircle A B C = r ∧ 
    Segment_length A M = AM ∧ 
    Segment_length M S = MS ∧ 
    Orthocenter A B C M ∧ 
    Centroid A B C S ∧ 
    Euler_line A B C M S O

theorem triangle_constructibility (r AM MS : ℝ) : construct_triangle r AM MS :=
sorry

end triangle_constructibility_l700_700020


namespace problem1_problem2_problem3_l700_700934

-- Problem 1
theorem problem1 (k : ℝ) (h : k ≥ 0) : 
  ∀ x, x ∈ set.Ici (Real.sqrt (2 * k + 1)) → monotone_on (λ x, x + (2 * k + 1) / x) (set.Ici (Real.sqrt (2 * k + 1))) :=
sorry

-- Problem 2
theorem problem2 (k m : ℝ) (h : k ∈ set.Icc 1 7) (h2 : ∀ x, x ∈ set.Icc 2 3 → x + (2 * k + 1) / x ≥ m) : 
  m ≤ 7 / 2 :=
sorry

-- Problem 3
theorem problem3 (k : ℝ) (h : k > 0) : 
  ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ (∀ x, ∃ t, f (|2 ^ x - 1|) - 3 * k - 2 = 0) :=
sorry

end problem1_problem2_problem3_l700_700934


namespace range_of_z_l700_700976

theorem range_of_z (x y : ℝ) (h : x^2 + y^2 = 1) : 
  ∃ z : Set ℝ, z = { k | k = (y - 1) / (x + 2) } ∧ z ⊆ Icc (-4/3 : ℝ) (0 : ℝ) := sorry

end range_of_z_l700_700976


namespace smallest_proportion_first_fruit_l700_700099

-- Definitions based on conditions
def ratio : list ℕ := [1, 2, 3]
def total_cups : ℕ := 6
def is_min (x : ℕ) (l : list ℕ) : Prop := ∀ (y : ℕ), y ∈ l → x ≤ y

-- Statement to prove the answer
theorem smallest_proportion_first_fruit :
  is_min (ratio.nthLe 0 (by decide)) ratio :=
sorry

end smallest_proportion_first_fruit_l700_700099


namespace rose_apples_l700_700646

theorem rose_apples (total_apples friends : ℕ) (h_total : total_apples = 9) (h_friends : friends = 3) :
  total_apples / friends = 3 :=
by
  rw [h_total, h_friends]
  sorry

end rose_apples_l700_700646


namespace man_walk_time_l700_700374

theorem man_walk_time (speed_kmh : ℕ) (distance_km : ℕ) (time_min : ℕ) 
  (h1 : speed_kmh = 10) (h2 : distance_km = 7) : time_min = 42 :=
by
  sorry

end man_walk_time_l700_700374


namespace cyclic_quadrilateral_iff_tangent_circles_l700_700693

theorem cyclic_quadrilateral_iff_tangent_circles
  {O1 O2 : Type} [MetricSpace O1] [MetricSpace O2]
  {r1 r2 : ℝ} 
  {A B C D T : MetricSpace.Point O1} 
  (h_circle1 : Metric.sphere O1 r1) 
  (h_circle2 : Metric.sphere O2 r2)
  (h_tangent1 : Metric.tangent_at A B O1 O2)
  (h_tangent2 : Metric.tangent_at C D O1 O2)
  (h_touch : Metric.touching O1 O2 T) : 
  (Metric.cyclic_quadrilateral A B C D ↔ Metric.tangent_circles O1 O2) :=
begin
  sorry
end

end cyclic_quadrilateral_iff_tangent_circles_l700_700693


namespace problem_solution_l700_700561

theorem problem_solution (x : ℝ) (h : (18 / 100) * 42 = (27 / 100) * x) : x = 28 :=
sorry

end problem_solution_l700_700561


namespace problem_1_problem_2a_problem_2b_problem_3_l700_700493

noncomputable def a : ℕ → ℝ
| 0     := 1
| 1     := 2
| 2     := r
| (n+3) := a n + 2

noncomputable def b : ℕ → ℝ
| 0     := 1
| 1     := 0
| 2     := -1
| 3     := 0
| (n+4) := b n

noncomputable def T (n : ℕ) : ℝ :=
(∑ k in finset.range n, b k * a k)

theorem problem_1 (h : ∑ k in finset.range 9, a k = 34) : r = 3 :=
sorry

theorem problem_2a : T 12 = -4 :=
sorry

theorem problem_2b (n : ℕ) : T (12 * n) = -4 * n :=
sorry

theorem problem_3 (r_pos : r > 0)
  (h : ∃ m : ℕ,∃ t1 t2 t3 t4 : ℕ, 
  t1 < 12 ∧ t2 < 12 ∧ t3 < 12 ∧ t4 < 12 ∧
  t1 ≠ t2 ∧ t1 ≠ t3 ∧ t1 ≠ t4 ∧ t2 ≠ t3 ∧ t2 ≠ t4 ∧ t3 ≠ t4 ∧
  T (12 * m + t1) = 100 ∧ T (12 * m + t2) = 100 ∧
  T (12 * m + t3) = 100 ∧ T (12 * m + t4) = 100) : 
  r = 1 ∧ 
  ∃ m t1 t2 t3 t4 : ℕ, T (12 * m + t1) = 100 ∧
  T (12 * m + t2) = 100 ∧ T (12 * m + t3) = 100 ∧
  T (12 * m + t4) = 100 :=
sorry

end problem_1_problem_2a_problem_2b_problem_3_l700_700493


namespace floor_equation_solution_l700_700047

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l700_700047


namespace translate_parabola_l700_700692

theorem translate_parabola (x : ℝ) :
  let y := -4 * x^2
  let y' := -4 * (x + 2)^2 - 3
  y' = -4 * (x + 2)^2 - 3 := 
sorry

end translate_parabola_l700_700692


namespace problem_Ⅰ_problem_Ⅱ_l700_700529

-- Define the parametric equation of curve C₁
def parametric_C₁ (α : ℝ) : ℝ × ℝ := (3 * Real.cos α, Real.sin α)

-- Define the polar equation of curve C₂ in Cartesian form
def cartesian_C₂ (x y : ℝ) : Prop := x - y - 2 = 0

-- Define max distance of |OP|
def max_OP_distance : ℝ := 3

-- Define the equation of curve C₁ in Cartesian form
def cartesian_C₁ (x y : ℝ) : Prop := x^2 / 9 + y^2 = 1

-- Define problem Ⅰ
theorem problem_Ⅰ :
  (∀ x y, cartesian_C₂ x y ↔ x - y - 2 = 0) ∧
  (∀ α, ∃ max_d, max_d = √(8 * (Real.cos α)^2 + 1) → max_d = max_OP_distance) :=
by sorry

-- Define intersection points A and B and point E on the x-axis
def intersection_points (C₁ C₂ : ℝ × ℝ → Prop) : set (ℝ × ℝ) :=
  {p | C₁ p ∧ C₂ p}

-- Define problem Ⅱ
theorem problem_Ⅱ :
  let E := (2, 0) in
  let C₁ : ℝ × ℝ → Prop := λ p, cartesian_C₁ p.1 p.2 in
  let C₂ : ℝ × ℝ → Prop := λ p, cartesian_C₂ p.1 p.2 in
  ∃ t1 t2 : ℝ,
  (t1 + t2 = - (2 * Real.sqrt 2) / 5 ∧ t1 * t2 = -1) →
  abs t1 + abs t2 = 6 * Real.sqrt 3 / 5 :=
by sorry

end problem_Ⅰ_problem_Ⅱ_l700_700529


namespace tangent_normal_lines_l700_700336

noncomputable def parametric_eqns : (ℝ → ℝ) × (ℝ → ℝ) :=
  (λ t, Real.arcsin (t / Real.sqrt (1 + t ^ 2)),
   λ t, Real.arccos (1 / Real.sqrt (1 + t ^ 2)))

def t0 : ℝ := -1

theorem tangent_normal_lines :
  let x0 := Real.arcsin (-1 / Real.sqrt 2)
  let y0 := Real.arccos (1 / Real.sqrt 2)
  let slope_tangent := 2
  let slope_normal := -1 / 2
  let tangent_line := (λ x : ℝ, slope_tangent * x + 3 * Real.pi / 4)
  let normal_line := (λ x : ℝ, slope_normal * x + Real.pi / 8)
  ∃ t0 : ℝ, x0 = -Real.pi/4 ∧ y0 = Real.pi/4 ∧
    tangent_line x0 = 2 * x0 + (3 * Real.pi / 4) ∧
    normal_line x0 = -x0 / 2 + (Real.pi / 8) :=
begin
  existsi t0,
  -- The rest of the proof, as suggested, is omitted.
  sorry
end

end tangent_normal_lines_l700_700336


namespace instantaneous_velocity_at_t4_l700_700733

def position (t : ℝ) : ℝ := t^2 - t + 2

theorem instantaneous_velocity_at_t4 : 
  (deriv position 4) = 7 := 
by
  sorry

end instantaneous_velocity_at_t4_l700_700733


namespace exists_prime_divisor_gt_10_2012_l700_700461

def S (n : ℕ) : ℕ := ∑ i in Finset.range (n + 1), Nat.factorial i

theorem exists_prime_divisor_gt_10_2012 :
  ∃ n, ∃ p : ℕ, Nat.Prime p ∧ 10 ^ 2012 < p ∧ p ∣ S n :=
sorry

end exists_prime_divisor_gt_10_2012_l700_700461


namespace usual_time_to_cover_distance_l700_700697

variable (S T : ℝ)

-- Conditions:
-- 1. The man walks at 40% of his usual speed.
-- 2. He takes 24 minutes more to cover the same distance at this reduced speed.
-- 3. Usual speed is S.
-- 4. Usual time to cover the distance is T.

def usual_speed := S
def usual_time := T
def reduced_speed := 0.4 * S
def extra_time := 24

-- Question: Prove the man's usual time to cover the distance is 16 minutes.
theorem usual_time_to_cover_distance : T = 16 := 
by
  have speed_relation : S / (0.4 * S) = (T + 24) / T :=
    sorry
  have simplified_speed_relation : 2.5 = (T + 24) / T :=
    sorry
  have cross_multiplication_step : 2.5 * T = T + 24 :=
    sorry
  have solve_for_T_step : 1.5 * T = 24 :=
    sorry
  have final_step : T = 16 :=
    sorry
  exact final_step

end usual_time_to_cover_distance_l700_700697


namespace solve_for_k_l700_700838

theorem solve_for_k : ∃ k : ℤ, 2^5 - 10 = 5^2 + k ∧ k = -3 := by
  sorry

end solve_for_k_l700_700838


namespace permutations_PERCEPTION_l700_700822

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l700_700822


namespace find_e_l700_700018

theorem find_e (d e f : ℝ) (Q : ℝ → ℝ) (zeros_mean_eq : (∑ r in {r, s, t}, r) / 3 = -f)
                (zeros_product_eq : (∏ r in {r, s, t}, r) = -f)
                (coeff_sum_eq : 1 + d + e + f = -f)
                (y_intercept: Q 0 = 4)
                (poly_def: ∀ x, Q x = x^3 + d * x^2 + e * x + f) : e = 11 :=
by
  sorry

end find_e_l700_700018


namespace find_hyperbola_equation_l700_700527

-- Define the hyperbola conditions
variables {x y : ℝ} (a b : ℝ)
def hyperbola : Prop := (a > 0) ∧ (b > 0) ∧ (y = x * (sqrt 3 / 3) ∨ y = -x * (sqrt 3 / 3)) ∧ 
                         dist (a, 0) (x, x * (sqrt 3 / 3)) = 1

theorem find_hyperbola_equation (h : hyperbola 2 (2 * sqrt 3 / 3)) : 
  (x^2 / 4) - (3 * y^2 / 4) = 1 :=
sorry

end find_hyperbola_equation_l700_700527


namespace part1_part2_l700_700931

def f (x : ℝ) (ω : ℝ) := (√3) * Real.sin (2 * ω * x) + 2 * (Real.cos (ω * x))^2

theorem part1 (ω : ℝ) :
  (∀ x, f x ω = 2 * Real.sin(2 * x + (π/6)) + 1) →
  (∀ x, disjoint (Icc (π/6 + x * π) (2*π/3 + x * π)) ∧ ∥ ω ∥ = 1) :=
sorry

theorem part2 (m : ℝ) :
  (∃ x, -π/4 < x ∧ x < π/4 ∧ f x 1 = m) →
  (1 - √3 < m ∧ m ≤ 2) :=
sorry

end part1_part2_l700_700931


namespace quadrilateral_area_correct_l700_700756

open Real
open Function
open Classical

noncomputable def quadrilateral_area : ℝ :=
  let A := (0, 0)
  let B := (2, 3)
  let C := (5, 0)
  let D := (3, -2)
  let vector_cross_product (u v : ℝ × ℝ) : ℝ := u.1 * v.2 - u.2 * v.1
  let area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ := 0.5 * abs (vector_cross_product (p2 - p1) (p3 - p1))
  area_triangle A B D + area_triangle B C D

theorem quadrilateral_area_correct : quadrilateral_area = 17 / 2 :=
  sorry

end quadrilateral_area_correct_l700_700756


namespace solution_set_l700_700067

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l700_700067


namespace modulus_of_complex_l700_700137

theorem modulus_of_complex (z : ℂ) (h : (1 - complex.I) * z = 2 * complex.I) : complex.abs z = real.sqrt 2 :=
sorry

end modulus_of_complex_l700_700137


namespace class_average_weight_l700_700319

theorem class_average_weight :
  (24 * 40 + 16 * 35 + 18 * 42 + 22 * 38) / (24 + 16 + 18 + 22) = 38.9 :=
by
  -- skipped proof
  sorry

end class_average_weight_l700_700319


namespace box_box_15_eq_60_l700_700102

def sum_of_factors (n : ℕ) : ℕ :=
  (Finset.filter (λ d, n % d = 0) (Finset.range (n + 1))).sum

theorem box_box_15_eq_60 : sum_of_factors (sum_of_factors 15) = 60 :=
by
  have h15 : sum_of_factors 15 = 24 := by sorry
  rw [h15]
  have h24 : sum_of_factors 24 = 60 := by sorry
  rw [h24]
  exact rfl

end box_box_15_eq_60_l700_700102


namespace part_I_monotonic_decreasing_intervals_part_II_minimum_value_l700_700920

noncomputable def f (x : ℝ) (m : ℝ) : ℝ :=
  -x^3 + 3 * x^2 + 9 * x + m

-- Part (I)
theorem part_I_monotonic_decreasing_intervals (m : ℝ) :
  (∀ x, -x^3 + 3 * x^2 + 9 * x + m < 0 → x < -1 ∨ x > 3) :=
sorry

-- Part (II)
theorem part_II_minimum_value (m : ℝ) :
  (∀ x, f 2 m = 20 → f (-1) m = -7) :=
sorry

end part_I_monotonic_decreasing_intervals_part_II_minimum_value_l700_700920


namespace floor_eq_l700_700064

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l700_700064


namespace minimum_determinant_value_l700_700088

def determinant (θ : ℝ) : ℝ :=
| 1       1        1         |
| 1  (1 + tan θ)  1  |
| (1 + cot θ)  1  1 |

theorem minimum_determinant_value :
  ∀ θ : ℝ, tan θ ≠ 0 ∧ cot θ ≠ 0 → determinant θ = -1 := 
begin
  sorry
end

end minimum_determinant_value_l700_700088


namespace necessary_but_not_sufficient_condition_l700_700375

theorem necessary_but_not_sufficient_condition
  (a : ℝ)
  (h : ∃ x : ℝ, a * x^2 - 2 * x + 1 < 0) :
  (a < 2 ∧ a < 3) :=
by
  sorry

end necessary_but_not_sufficient_condition_l700_700375


namespace sufficient_condition_for_quadratic_l700_700551

theorem sufficient_condition_for_quadratic (a : ℝ) : 
  (∃ (x : ℝ), (x > a) ∧ (x^2 - 5*x + 6 ≥ 0)) ∧ 
  (¬(∀ (x : ℝ), (x^2 - 5*x + 6 ≥ 0) → (x > a))) ↔ 
  a ≥ 3 :=
by
  sorry

end sufficient_condition_for_quadratic_l700_700551


namespace divisor_is_22_l700_700705

theorem divisor_is_22 (n d : ℤ) (h1 : n % d = 12) (h2 : (2 * n) % 11 = 2) : d = 22 :=
by
  sorry

end divisor_is_22_l700_700705


namespace car_travel_distance_l700_700737

variable (b t : Real)
variable (h1 : b > 0)
variable (h2 : t > 0)

theorem car_travel_distance (b t : Real) (h1 : b > 0) (h2 : t > 0) :
  let rate := b / 4
  let inches_in_yard := 36
  let time_in_seconds := 5 * 60
  let distance_in_inches := (rate / t) * time_in_seconds
  let distance_in_yards := distance_in_inches / inches_in_yard
  distance_in_yards = (25 * b) / (12 * t) := by
  sorry

end car_travel_distance_l700_700737


namespace original_rectangle_area_l700_700300

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l700_700300


namespace divide_7_acute_triangles_l700_700421

theorem divide_7_acute_triangles (A B C : Type*) (hABC : ∀ (α β γ : ℝ), ∠A + ∠B + ∠C = π ∧ ∠C > π / 2) : 
  ∃ T1 T2 T3 T4 T5 T6 T7 : Type*, 
  (∀ i, 1 ≤ i ∧ i ≤ 7 → ∃ a b c, T_i = triangle a b c ∧ ∠a < π / 2 ∧ ∠b < π / 2 ∧ ∠c < π / 2) ∧ 
  (triangle A B C = T1 ∪ T2 ∪ T3 ∪ T4 ∪ T5 ∪ T6 ∪ T7) :=
sorry

end divide_7_acute_triangles_l700_700421


namespace elgin_amount_is_15_l700_700393

def total_amount : ℕ := 80
def abs_diff_a_b : ℕ := 12
def abs_diff_b_c : ℕ := 7
def abs_diff_c_d : ℕ := 5
def abs_diff_d_e : ℕ := 4
def abs_diff_e_a : ℕ := 16

theorem elgin_amount_is_15 
  (A B C D E : ℕ)
  (h1 : |(A - B)| = abs_diff_a_b)
  (h2 : |(B - C)| = abs_diff_b_c)
  (h3 : |(C - D)| = abs_diff_c_d)
  (h4 : |(D - E)| = abs_diff_d_e)
  (h5 : |(E - A)| = abs_diff_e_a)
  (h6 : A + B + C + D + E = total_amount) :
  E = 15 :=
sorry

end elgin_amount_is_15_l700_700393


namespace arithmetic_geometric_progression_l700_700668

theorem arithmetic_geometric_progression (a b : ℝ) :
  (b = 2 - a) ∧ (b = 1 / a ∨ b = -1 / a) →
  (a = 1 ∧ b = 1) ∨
  (a = 1 + Real.sqrt 2 ∧ b = 1 - Real.sqrt 2) ∨
  (a = 1 - Real.sqrt 2 ∧ b = 1 + Real.sqrt 2) :=
by
  sorry

end arithmetic_geometric_progression_l700_700668


namespace value_of_quotient_l700_700558

variable (a b c d : ℕ)

theorem value_of_quotient 
  (h1 : a = 3 * b)
  (h2 : b = 2 * c)
  (h3 : c = 5 * d) :
  (a * c) / (b * d) = 15 :=
by
  sorry

end value_of_quotient_l700_700558


namespace count_permutations_perception_l700_700813

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l700_700813


namespace number_equality_l700_700957

theorem number_equality (n : ℚ) : 
  (4 / 5) * n * (5 / 6) * n = n ↔ n = 0 ∨ n = 3 / 2 :=
by
  intro h
  calc
    (4 / 5) * n * (5 / 6) * n
    = (4 / 5) * (5 / 6) * n^2 : by ring
    ... = (20 / 30) * n^2 : by norm_num
    ... = (2 / 3) * n^2 : by norm_num
    ... = n : sorry

end number_equality_l700_700957


namespace city_map_scale_l700_700588

theorem city_map_scale 
  (map_length : ℝ) (actual_length_km : ℝ) (actual_length_cm : ℝ) (conversion_factor : ℝ)
  (h1 : map_length = 240) 
  (h2 : actual_length_km = 18)
  (h3 : actual_length_cm = actual_length_km * conversion_factor)
  (h4 : conversion_factor = 100000) :
  map_length / actual_length_cm = 1 / 7500 :=
by
  sorry

end city_map_scale_l700_700588


namespace parabola_conditions_l700_700866

-- Definitions based on conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 4*x - 3 + a

def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def intersects_at_2_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Proof Problem Statement
theorem parabola_conditions (a : ℝ) :
  (passes_through (quadratic_function a) 0 1 → a = 4) ∧
  (intersects_at_2_points (quadratic_function a) → (a = 3 ∨ a = 7)) :=
by
  sorry

end parabola_conditions_l700_700866


namespace total_circuit_boards_l700_700569

-- Define the given conditions
variable (T P F : ℕ)
variable (h1 : 64 = F) -- 64 circuit boards fail the verification process
variable (h2 : 456 = 64 + (1/8) * P) -- approximately 456 faulty circuit boards in total
variable (h3 : 1/8 * P = 392) -- 1/8 of the circuit boards that pass the verification process are faulty

-- Prove the question statement
theorem total_circuit_boards (T P F : ℕ) (h1 : F = 64) (h2 : 456 = 64 + (1/8) * P) (h3 : 1/8 * P = 392) : 
  T = 3200 :=
by
  -- Let P be the number of boards that pass verification
  let P := 392 * 8
  -- Let T be the total number of circuit boards
  let T := P + 64
  -- Show T equals the number of circuit boards
  show T = 3200 from sorry

end total_circuit_boards_l700_700569


namespace rectangular_to_polar_l700_700022

def convert_to_polar (x y : ℝ) : ℝ × ℝ :=
let r := real.sqrt (x^2 + y^2) in
let θ := real.arctan (y / x) in
if y < 0 then (r, 2 * real.pi - θ) else (r, θ)

theorem rectangular_to_polar (x y : ℝ) (r θ : ℝ) :
  convert_to_polar 3 (-3 * real.sqrt 3) = (r, θ) ↔ (r = 6 ∧ θ = 5 * real.pi / 3) :=
by
  sorry

end rectangular_to_polar_l700_700022


namespace sara_pears_left_l700_700647

theorem sara_pears_left (picked : ℕ) (given : ℕ) (h_picked : picked = 35) (h_given : given = 28) : picked - given = 7 :=
by
  rw [h_picked, h_given]
  rfl

end sara_pears_left_l700_700647


namespace values_of_a_and_b_range_of_m_l700_700521

-- Define the context of the function f
def f (a : ℝ) (x : ℝ) : ℝ := 1 - (a * 5^x) / (5^x + 1)

-- Define the conditions and the statement of the problem
theorem values_of_a_and_b :
  (∀ a b, f a 0 = 0 ∧ b - 3 + 2 * b = 0 → a = 2 ∧ b = 1) :=
by
  -- Proof omitted for brevity
  sorry

theorem range_of_m :
  (∀ (a b : ℝ), (0 < a) → b = 1 → 
   (∀ x, x ∈ (b-3, 2*b) → f a x) < ∀ y, y < x → f a y <
   ∀ (m : ℝ), f a (m - 1) + f a (2 * m + 1) > 0 → 
   m ∈ (-1 : ℝ, 0 : ℝ)) :=
by
  -- Proof omitted for brevity
  sorry

end values_of_a_and_b_range_of_m_l700_700521


namespace total_games_in_season_l700_700384

theorem total_games_in_season 
  (teams : ℕ) (divisions : ℕ) (teams_per_division : Π (d : ℕ), d < divisions → ℕ) 
  (games_per_team_in_division : ℕ) (games_per_team_between_divisions : ℕ)
  (h_teams : teams = 16) (h_divisions : divisions = 2)
  (h_teams_per_division : ∀ (d : ℕ) (hd : d < divisions), teams_per_division d hd = 8)
  (h_games_per_team_in_division : games_per_team_in_division = 3)
  (h_games_per_team_between_divisions : games_per_team_between_divisions = 2) :
  ∑ d in finset.range divisions, ∑ i in finset.range (teams_per_division d (nat.lt_of_lt_of_le d h_divisions.le)), 
    ∑ j in finset.range (teams_per_division d (nat.lt_of_lt_of_le d h_divisions.le)), if i < j then games_per_team_in_division else 0 +
  ∑ i in finset.range (teams_per_division 0 (nat.lt_of_lt_of_le 0 h_divisions.le)), 
    ∑ j in finset.range (teams_per_division 1 (nat.lt_of_lt_of_le 1 h_divisions.le)), games_per_team_between_divisions = 296 :=
  sorry

end total_games_in_season_l700_700384


namespace range_of_a_for_monotonicity_l700_700152

noncomputable def f (x : ℝ) (a : ℝ) := (Real.sqrt (x^2 + 1)) - a * x

theorem range_of_a_for_monotonicity (a : ℝ) (h : a > 0) :
  (∀ x y : ℝ, x > 0 → y > 0 → x < y → f x a < f y a) ↔ a ≥ 1 := sorry

end range_of_a_for_monotonicity_l700_700152


namespace maximum_obtuse_dihedral_angles_l700_700761

-- condition: define what a tetrahedron is and its properties
structure Tetrahedron :=
  (edges : Fin 6 → ℝ)   -- represents the 6 edges
  (dihedral_angles : Fin 6 → ℝ) -- represents the 6 dihedral angles

-- Define obtuse angle in degrees
def is_obtuse (angle : ℝ) : Prop := angle > 90 ∧ angle < 180

-- Theorem statement
theorem maximum_obtuse_dihedral_angles (T : Tetrahedron) : 
  (∃ count : ℕ, count = 3 ∧ (∀ i, is_obtuse (T.dihedral_angles i) → count <= 3)) := sorry

end maximum_obtuse_dihedral_angles_l700_700761


namespace point_lies_on_graph_l700_700925

noncomputable def f (x : ℝ) : ℝ := |x - 2| + |x + 2|

theorem point_lies_on_graph (a : ℝ) : f (-a) = f (a) :=
by
  sorry

end point_lies_on_graph_l700_700925


namespace min_value_g_l700_700087

def g (x : ℝ) : ℝ := x + (2 * x) / (x^2 + 2) + (x * (x + 5)) / (x^2 + 3) + (3 * (x + 3)) / (x * (x^2 + 3))

theorem min_value_g : ∃ x, x > 0 ∧ ∀ y > 0, g y ≥ g x ∧ g x = 6 :=
by sorry

end min_value_g_l700_700087


namespace hyperbola_equation_and_range_of_m_l700_700119

-- Defining the hyperbola
def hyperbola (x y : ℝ) : Prop := (x^2 / 3) - y^2 = 1

-- Defining the conditions for focal length and eccentricity
def focal_length (c : ℝ) : Prop := 2 * c = 4
def eccentricity (c a : ℝ) : Prop := c / a = 2 * Real.sqrt 3 / 3

-- Given the definitions of the hyperbola and its conditions
variables (c a b k m : ℝ)

-- Prove the equation and the range of m
theorem hyperbola_equation_and_range_of_m :
  focal_length c ∧ eccentricity c a ∧ b^2 = c^2 - a^2 ∧ k ≠ 0 ∧ m ≠ 0
  → (∀ x y, hyperbola x y)
  ∧ ( ∀ x1 x2 y1 y2,
        let M := ( (x1 + x2) / 2, (y1 + y2) / 2) in
        let A := (0, -1) in
        ( y1 = k * x1 + m ∧ y2 = k * x2 + m ∧
          ( - 1 < M.2 ∧  M.2 ≤ 4 ) →
          ( - 1 < m ∧ m < 0 ∨  4 < m ) )) ∧
      sorry

end hyperbola_equation_and_range_of_m_l700_700119


namespace minimum_workers_in_team_A_l700_700136

variable (a b c : ℤ)

theorem minimum_workers_in_team_A (h1 : b + 90 = 2 * (a - 90))
                               (h2 : a + c = 6 * (b - c)) :
  ∃ a ≥ 148, a = 153 :=
by
  sorry

end minimum_workers_in_team_A_l700_700136


namespace infinite_series_solution_l700_700858

theorem infinite_series_solution :
  ∃ x : ℝ, x = x^3 - x^4 + x^5 - x^6 + x^7 - ∞ ∧ abs x < 1 → x = (-1 + sqrt 5) / 2 := by
  sorry

end infinite_series_solution_l700_700858


namespace simplify_expression_l700_700011

theorem simplify_expression (a : ℝ) (h : a ≠ 2) : 
  (a^2 / (a - 2)) - (4 / (a - 2)) = a + 2 :=
by 
  sorry

end simplify_expression_l700_700011


namespace parabola_through_point_intersects_axes_at_two_points_l700_700865

open Real

def quadratic_function_a := λ a x : ℝ, x^2 - 4 * x - 3 + a

theorem parabola_through_point (a : ℝ) :
  (∃ y, quadratic_function_a a 0 = y ∧ y = 1) → a = 4 := by
  sorry

theorem intersects_axes_at_two_points (a : ℝ) :
  (∀ x y : ℝ, quadratic_function_a a x = 0 → quadratic_function_a a y = 0) →
  (∃ b, b^2 - 4 * 1 * (-3 + a) = 0) → a = 7 := by
  sorry

end parabola_through_point_intersects_axes_at_two_points_l700_700865


namespace properties_of_dataset_l700_700742

def ordered_data : List ℕ := [30, 31, 31, 37, 40, 46, 47, 57, 62, 67]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if acc.count x > l.count acc then x else acc) 0

def median (l : List ℕ) : ℕ :=
  let n := l.length
  if n % 2 = 1 then
    l.get! (n / 2)
  else
    (l.get! (n / 2 - 1) + l.get! (n / 2)) / 2

def range (l : List ℕ) : ℕ :=
  l.maximum.getD 0 - l.minimum.getD 0

def quantile (l : List ℕ) (q : ℚ) : ℕ :=
  let idx := (q * rat.ofInt l.length).floor.toNat
  (l.get! idx + l.get! (idx + 1)) / 2

theorem properties_of_dataset :
  let dataset := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30].sort
  mode dataset = 31 ∧
  median dataset ≠ 40 ∧
  range dataset = 37 ∧
  quantile dataset (10 / 100) = 30.5 := by
  sorry

end properties_of_dataset_l700_700742


namespace plus_one_eq_next_plus_l700_700860

theorem plus_one_eq_next_plus (m : ℕ) (h : m > 1) : (m^2 + m) + 1 = ((m + 1)^2 + (m + 1)) := by
  sorry

end plus_one_eq_next_plus_l700_700860


namespace colorful_tiling_l700_700391

theorem colorful_tiling :
  let N := (21 * (3^3 - 3 * 2^3 + 3)) + 
           (35 * (3^4 - 3 * 2^4 + 3)) + 
           (35 * (3^5 - 3 * 2^5 + 3)) + 
           (21 * (3^6 - 3 * 2^6 + 3)) + 
           (7 * (3^7 - 3 * 2^7 + 3)) + 
           (1 * (3^8 - 3 * 2^8 + 3))
  in N % 1000 = 331 := by
  sorry

end colorful_tiling_l700_700391


namespace original_rectangle_area_l700_700299

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l700_700299


namespace problem_triangle_abc_l700_700992

theorem problem_triangle_abc 
  {a b c : ℝ}
  (h1 : (a^2 + c^2 - b^2) * Real.tan (angleB : ℝ) = Real.sqrt 3 * (b^2 + c^2 - a^2))
  (h_area : area_of_triangle a b c = 3 / 2) :
  angleA = Real.pi / 3 ∧ (bc : ℝ) (cos_angleA : ℝ) (cos_angleB : ℝ) (a^2 - b^2) :=
  (Real.cos angleA := (bc - 4 * Real.sqrt 3) * (Real.cos angleA) + a * b * Real.cos angleB) / (a^2 - b^2) = 1 :=
sorry

end problem_triangle_abc_l700_700992


namespace bill_fine_amount_l700_700396

-- Define the conditions
def ounces_sold : ℕ := 8
def earnings_per_ounce : ℕ := 9
def amount_left : ℕ := 22

-- Calculate the earnings
def earnings : ℕ := ounces_sold * earnings_per_ounce

-- Define the fine as the difference between earnings and amount left
def fine : ℕ := earnings - amount_left

-- The proof problem to solve
theorem bill_fine_amount : fine = 50 :=
by
  -- Statements and calculations would go here
  sorry

end bill_fine_amount_l700_700396


namespace direction_vector_of_line_l700_700373

theorem direction_vector_of_line (x1 y1 x2 y2 c : ℝ) :
  (x1, y1) = (-6, 1) → (x2, y2) = (-1, 5) → c = 4 :=
by
  intros h1 h2
  have dvec := (x2 - x1, y2 - y1)
  rw [h1, h2] at dvec
  have : dvec = (5, 4) := by simp [dvec] -- This line ensures we get (5, 4)
  rw [this]
  reflexivity -- we conclude c = 4 here

end direction_vector_of_line_l700_700373


namespace range_of_a_l700_700528

theorem range_of_a (a : ℝ) : 
  (∃! x : ℤ, 4 - 2 * x ≥ 0 ∧ (1 / 2 : ℝ) * x - a > 0) ↔ -1 ≤ a ∧ a < -0.5 :=
by
  sorry

end range_of_a_l700_700528


namespace transformed_point_l700_700672

-- Definitions of the transformations
def rotate_y_180 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, p.2, -p.3)
def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (-p.1, p.2, p.3)
def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ := (p.1, -p.2, p.3)

def point_transformations (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  reflect_xz (rotate_y_180 (reflect_xz (reflect_yz (rotate_y_180 p))))

-- The initial point
def initial_point : ℝ × ℝ × ℝ := (2, 3, 4)

-- The final coordinates after all transformations
theorem transformed_point : point_transformations initial_point = (-2, 3, 4) := 
by
  simp [initial_point, rotate_y_180, reflect_yz, reflect_xz]
  sorry

end transformed_point_l700_700672


namespace knowledge_round_table_l700_700788

variable {V : Type*} [fintype V] [decidable_eq V]
variable (G : simple_graph V)

theorem knowledge_round_table (h : ∀ v : V, 1 ≤ G.degree v ∧ G.degree v ≤ fintype.card V - 2) (hn : 4 ≤ fintype.card V) :
  ∃ (a b c d : V), (G.adj a b ∧ G.adj b c ∧ G.adj c d ∧ G.adj d a) ∧
                   (¬ G.adj a d) ∧ (¬ G.adj b a) ∧ (¬ G.adj c b) ∧ (¬ G.adj d c) := 
sorry

end knowledge_round_table_l700_700788


namespace minimum_k_is_2018_l700_700760

-- Define a polynomial of degree 2017 with integer coefficients and leading coefficient 1
structure PolynomialDegree2017 :=
  (coeffs : Fin 2018 → ℤ)
  (leading_coeff : coeffs 0 = 1)

-- Define the minimum k for which the polynomial can be uniquely identified
def min_k_uniquely_identify (P : PolynomialDegree2017) (int_points : List ℤ) (product_value : ℤ) := 
  ∃ (k : ℕ), k = 2018 ∧ (∀ (Q : PolynomialDegree2017), (∀ (i < k), (P.coeffs i = Q.coeffs i) → P = Q))

theorem minimum_k_is_2018 
  (P : PolynomialDegree2017) 
  (int_points : List ℤ) 
  (product_value : ℤ) :
  (min_k_uniquely_identify P int_points product_value) :=
  sorry

end minimum_k_is_2018_l700_700760


namespace quadratic_trinomial_sum_l700_700745

theorem quadratic_trinomial_sum (x1 x2 p q : ℤ)
  (h1 : p = x1 + x2)
  (h2 : q = x1 * x2)
  (h3 : ∃ d : ℤ, x1 = x2 + d)
  (h4 : q = x2)
  (h5 : x1 > x2 ∧ x2 > q) : 
  x2 = -3 ∨ x2 = -2 → x2 + x2 = -5 :=
by
  intros h_cases
  cases h_cases;
  simp [int.add, int.neg_succ_of_nat_eq];
  admit

end quadratic_trinomial_sum_l700_700745


namespace intersection_A_complement_B_l700_700156

noncomputable def A := { x : ℝ | 1 + 2 * x - 3 * x^2 > 0 }
noncomputable def B := { x : ℝ | 2 * x * (4 * x - 1) < 0 }
noncomputable def complement_R (B : set ℝ) := { x : ℝ | x ∉ B }

theorem intersection_A_complement_B :
  A ∩ (complement_R B) = set.Ioo (-1/3) 0 ∪ set.Ico (1/4) 1 := 
begin
  sorry,
end

end intersection_A_complement_B_l700_700156


namespace trapezoid_area_eq_c_l700_700763

theorem trapezoid_area_eq_c (b c : ℝ) (hb : b = Real.sqrt c) (hc : 0 < c) :
    let shorter_base := b - 3
    let altitude := b
    let longer_base := b + 3
    let K := (1/2) * (shorter_base + longer_base) * altitude
    K = c :=
by
    sorry

end trapezoid_area_eq_c_l700_700763


namespace elevator_safety_l700_700753

theorem elevator_safety :
  let weights_of_athletes := [82.7, 78.7, 78.8, 77.3, 83.6, 85.4, 73, 80.6, 83.2, 71.3, 74.4]
  let number_of_athletes := weights_of_athletes.length
  let average_weight_athletes := weights_of_athletes.sum / number_of_athletes
  let median_weight_athletes := weights_of_athletes.toArray.qsort ((<) : Float → Float → Prop)![5]
  let weights_of_ladies := List.replicate 4 52.3
  let total_number := weights_of_athletes.length + weights_of_ladies.length
  let total_weight := weights_of_athletes.sum + weights_of_ladies.sum
  number_of_athletes = 11 ∧
  average_weight_athletes = 79 ∧
  median_weight_athletes = 78.8 ∧
  total_number ≤ 18 ∧
  total_weight < 1100 :=
by
  sorry

end elevator_safety_l700_700753


namespace prop_1_prop_2_l700_700893

section problem_conditions
variable (a : ℕ+ → ℝ)

-- Condition: sequence definition
def a_def (n : ℕ+) : ℝ :=
  if n = 1 then 1 / 2 else (a (n - 1)) - (a (n - 1))^2

-- Proposition 1
theorem prop_1 (n : ℕ+) : 1 ≤ (a n) / (a (n + 1)) ∧ (a n) / (a (n + 1)) ≤ 2 :=
  sorry

-- Definition of S_n
def S (n : ℕ+) : ℝ :=
  ∑ i in Finset.range (Nat.succ n), (a (i + 1))^2

-- Proposition 2
theorem prop_2 (n : ℕ+) : 1 / (2 * (↑n + 2)) ≤ (S a n) / (n : ℝ) ∧ (S a n) / (n : ℝ) ≤ 1 / (2 * (↑n + 1)) :=
  sorry

end problem_conditions

end prop_1_prop_2_l700_700893


namespace shared_total_l700_700799

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l700_700799


namespace circle_intersection_properties_l700_700499

noncomputable theory
open Real

def circle1_equation (x y : ℝ) : Prop := x^2 + y^2 - 2 * x - 3 = 0
def circle2_equation (x y : ℝ) : Prop := x^2 + y^2 - 2 * y - 1 = 0
def line_AB_equation (x y : ℝ) : Prop := x - y + 1 = 0

theorem circle_intersection_properties :
  ∃ (A B : ℝ × ℝ),
    (circle1_equation A.1 A.2) ∧ (circle2_equation A.1 A.2) ∧
    (circle1_equation B.1 B.2) ∧ (circle2_equation B.1 B.2) ∧
    (∃ l1 l2 : ℕ, l1 ≠ l2) ∧
    (∀ A B, (circle1_equation A.1 A.2) ∧ (circle2_equation A.1 A.2) →
            line_AB_equation A.1 A.2) ∧
    (∀ x y, (circle1_equation x y) → ∃ max_dist, max_dist = 2 + sqrt 2) := 
sorry

end circle_intersection_properties_l700_700499


namespace problem_900_integers_l700_700958

theorem problem_900_integers :
  (∃ x : ℝ, 0 < x ∧ x ≤ 1 ∧ n = ⌊3 * x⌋ + ⌊6 * x⌋ + ⌊9 * x⌋ + ⌊12 * x⌋) ↔ (n ∈ (set.range (λ i, 1 + i) : set ℤ) ∧ ∃ m : ℤ, m ∈ (set.range (λ i, i * 30 + n) : set ℤ) ∧ (m ≤ 1500)) :=
  sorry

end problem_900_integers_l700_700958


namespace f_f_2_is_2_l700_700149

def f (x : ℝ) : ℝ :=
  if x < 4 then 2^x else real.sqrt x

theorem f_f_2_is_2 : f (f 2) = 2 :=
by
  -- The proof would go here
  sorry

end f_f_2_is_2_l700_700149


namespace find_z_for_orthogonality_l700_700452

theorem find_z_for_orthogonality : 
  ∃ z : ℝ, let v1 := ⟨3, -1, 5⟩ : ℝ × ℝ × ℝ, 
               v2 := ⟨4, z, -2⟩ : ℝ × ℝ × ℝ in 
  (v1.1 * v2.1 + v1.2 * v2.2 + v1.3 * v2.3 = 0) ∧ z = 2 :=
by
  sorry

end find_z_for_orthogonality_l700_700452


namespace sarah_change_sum_l700_700649

theorem sarah_change_sum : 
  let valid_amounts := {x | x < 100 ∧ x % 5 = 4 ∧ x % 10 = 7} in
  ∑ x in valid_amounts, x = 497 :=
by
  sorry

end sarah_change_sum_l700_700649


namespace compute_sum_of_sqrt_fractions_l700_700659

variable {a b c : ℝ}
variable (hac : a * c / (a + b) + b * a / (b + c) + c * b / (c + a) = -12)
variable (hbc : b * c / (a + b) + c * a / (b + c) + a * b / (c + a) = 15)

theorem compute_sum_of_sqrt_fractions :
  (sqrt b / (a + b) + sqrt c / (b + c) + sqrt a / (c + a) = 1) :=
sorry

end compute_sum_of_sqrt_fractions_l700_700659


namespace least_three_digit_7_heavy_l700_700765

-- Define what it means for a number to be "7-heavy"
def is_7_heavy(n : ℕ) : Prop := n % 7 > 4

-- Smallest three-digit number
def smallest_three_digit_number : ℕ := 100

-- Least three-digit 7-heavy whole number
theorem least_three_digit_7_heavy : ∃ n, smallest_three_digit_number ≤ n ∧ is_7_heavy(n) ∧ ∀ m, smallest_three_digit_number ≤ m ∧ is_7_heavy(m) → n ≤ m := 
  sorry

end least_three_digit_7_heavy_l700_700765


namespace ball_selection_count_l700_700092

theorem ball_selection_count :
  let even := ∑ (k : ℕ), (k % 2 = 0)
  let odd := ∑ (k : ℕ), (k % 2 = 1)
in
∑ (r g y : ℕ) in { (2005, even, _) ∪ (2005, _, odd) }, 1
= Nat.binomial 2007 2 - Nat.binomial 1004 2 :=
sorry

end ball_selection_count_l700_700092


namespace shared_total_l700_700800

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l700_700800


namespace maximize_garden_area_l700_700380

theorem maximize_garden_area :
  ∀ (x: ℝ), 
    (8 * x = 2240) ∧ (x ≤ 300) ∧ (x ∈ set.Icc 0 300) → 
      let y := 280 - 2 * x in 
      y * x = 19600 → 
      y = 140 :=
sorry

end maximize_garden_area_l700_700380


namespace average_is_greater_by_approx_three_l700_700954

def weights : List ℕ := [60, 65, 12, 12, 14, 20, 115]

def median (l : List ℕ) : ℚ :=
  let sorted := l.qsort (· < ·)
  (sorted.get! (sorted.length / 2) + sorted.get! (sorted.length / 2 - 1)) / 2
  
def average (l : List ℕ) : ℚ :=
  l.sum / l.length

theorem average_is_greater_by_approx_three :
  abs (average weights - median weights) ≈ 3 :=
sorry

end average_is_greater_by_approx_three_l700_700954


namespace intersection_point_exists_correct_line_l700_700850

noncomputable def line1 (x y : ℝ) : Prop := 2 * x - 3 * y + 2 = 0
noncomputable def line2 (x y : ℝ) : Prop := 3 * x - 4 * y - 2 = 0
noncomputable def parallel_line (x y : ℝ) : Prop := 4 * x - 2 * y + 7 = 0
noncomputable def target_line (x y : ℝ) : Prop := 2 * x - y - 18 = 0

theorem intersection_point_exists (x y : ℝ) : line1 x y ∧ line2 x y → (x = 14 ∧ y = 10) := 
by sorry

theorem correct_line (x y : ℝ) : 
  (∃ (x y : ℝ), line1 x y ∧ line2 x y) ∧ parallel_line x y 
  → target_line x y :=
by sorry

end intersection_point_exists_correct_line_l700_700850


namespace greatest_k_for_7_pow_k_divides_fact_50_l700_700564

theorem greatest_k_for_7_pow_k_divides_fact_50 :
  let r := (List.range 50).map (fun x => x + 1).prod in
  ∃ k : ℕ, (7^k ∣ r) ∧ (∀ m : ℕ, (7^m ∣ r) → m ≤ 8) ∧ k = 8 :=
by {
  let r := (List.range 50).map (fun x => x + 1).prod,
  existsi 8,
  split,
  { sorry },
  split,
  { intros m hm,
    sorry },
  refl
}

end greatest_k_for_7_pow_k_divides_fact_50_l700_700564


namespace circle_equation_l700_700849

noncomputable def isCircleEquation (D E F : ℝ) : Prop := 
  ∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0

noncomputable def circle_center (D E : ℝ) : ℝ × ℝ :=
  (D / 2, E / 2)

noncomputable def perpendicularSlope (k1 k2 : ℝ) : Prop :=
  k1 * k2 = -1

noncomputable def passesThroughPoint (a b : ℝ) (D E F : ℝ) : Prop :=
  (a^2 + b^2 + D * a + E * b + F = 0)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ := 
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem circle_equation :
  ∃ (D E F : ℝ), 
  isCircleEquation D E F ∧ 
  passesThroughPoint (-2) (-4) D E F ∧ 
  passesThroughPoint 8 6 D E F ∧
  let C := circle_center D E in
  let B := (8.0, 6.0) in
  let kCB := (C.2 - B.2) / (C.1 - B.1) in
  let kL := -1 / 3 in
  perpendicularSlope kCB kL ∧
  (D = -11 ∧ E = 3 ∧ F = -30 ∧ (x - 11/2)² + (y + 3/2)² = 125/2) := 
sorry

end circle_equation_l700_700849


namespace roots_of_quadratic_l700_700507

theorem roots_of_quadratic (p q x1 x2 : ℕ) (h1 : p + q = 28) (h2 : x1 * x2 = q) (h3 : x1 + x2 = -p) (h4 : x1 > 0) (h5 : x2 > 0) : 
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end roots_of_quadratic_l700_700507


namespace instantaneous_speed_at_t_three_l700_700912

theorem instantaneous_speed_at_t_three :
  let s := λ t : ℝ, 2 * t^2 + 4 * t in
  let v := λ t : ℝ, deriv s t in
  v 3 = 16 :=
by
  sorry

end instantaneous_speed_at_t_three_l700_700912


namespace factorize_difference_of_squares_l700_700443

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
begin
  sorry
end

end factorize_difference_of_squares_l700_700443


namespace carlos_finishes_first_l700_700392

theorem carlos_finishes_first
  (a : ℝ) -- Andy's lawn area
  (r : ℝ) -- Andy's mowing rate
  (hBeth_lawn : ∀ (b : ℝ), b = a / 3) -- Beth's lawn area
  (hCarlos_lawn : ∀ (c : ℝ), c = a / 4) -- Carlos' lawn area
  (hCarlos_Beth_rate : ∀ (rc rb : ℝ), rc = r / 2 ∧ rb = r / 2) -- Carlos' and Beth's mowing rate
  : (∃ (ta tb tc : ℝ), ta = a / r ∧ tb = (2 * a) / (3 * r) ∧ tc = a / (2 * r) ∧ tc < tb ∧ tc < ta) :=
-- Prove that the mowing times are such that Carlos finishes first
sorry

end carlos_finishes_first_l700_700392


namespace equal_lengths_of_isosceles_triangle_l700_700229

theorem equal_lengths_of_isosceles_triangle {A B C D E F P Q : Type*}
  [triangle_ABC : triangle A B C]  -- Condition: Triangle ABC
  (D_feat : foot_of_altitude D B C)
  (E_feat : foot_of_altitude E C A)
  (F_feat : foot_of_altitude F A B)
  (P_intersection : intersection_point P (line_through E F) (circumcircle (triangle A B C)))
  (Q_intersection : intersection_point Q (line_through B P) (line_through D F)) :
  distance A P = distance A Q := sorry

end equal_lengths_of_isosceles_triangle_l700_700229


namespace sequence_geometric_sequence_sum_l700_700492

theorem sequence_geometric :
  ∀ (λ : ℝ) (a : ℕ → ℝ),
  (a 1 = 1) →
  (∀ n, a (n + 1) = 2 * a n + λ) →
  (λ ≠ -1) →
  ∀ n, a n = (1 + λ) * 2^(n - 1) - λ :=
by {
  sorry
}

theorem sequence_sum :
  ∀ (a : ℕ → ℝ) (T : ℕ → ℝ),
  (∃ λ, λ = 1) → -- This can be simplified as assumed λ = 1
  (a 1 = 1) →
  (∀ n, a (n + 1) = 2 * a n + 1) →
  (T 1 = 2 * 2) →
  (∀ n, T (n + 1) = (2 + 2 * 2^2 + 3 * 2^3 + ... + n * 2^n -
  n * 2^(n + 1))) →
  ∀ n, T n = (n - 1) * 2^(n + 1) + 2 :=
by {
  sorry
}

end sequence_geometric_sequence_sum_l700_700492


namespace boys_without_notebooks_l700_700182

/-
Given that:
1. There are 16 boys in Ms. Green's history class.
2. 20 students overall brought their notebooks to class.
3. 11 of the students who brought notebooks are girls.

Prove that the number of boys who did not bring their notebooks is 7.
-/

theorem boys_without_notebooks (total_boys : ℕ) (total_notebooks : ℕ) (girls_with_notebooks : ℕ)
  (hb : total_boys = 16) (hn : total_notebooks = 20) (hg : girls_with_notebooks = 11) : 
  (total_boys - (total_notebooks - girls_with_notebooks) = 7) :=
by
  sorry

end boys_without_notebooks_l700_700182


namespace cost_per_minute_l700_700213

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end cost_per_minute_l700_700213


namespace solution_to_diff_eq_l700_700267

variable {C : ℝ} {x y : ℝ}

-- Defining the differential equation conditions
def differential_eq (x y : ℝ) : Prop :=
  2 * x * y * Real.log y * (deriv fun _ => x) + (x^2 + y^2 * Real.sqrt (y^2 + 1)) * (deriv fun _ => y) = 0

-- Defining the candidate solution
def candidate_solution (x y : ℝ) (C : ℝ) : Prop :=
  3 * x^2 * Real.log y + Real.sqrt ((y^2 + 1)^3) = C

-- Proof problem statement
theorem solution_to_diff_eq (x y : ℝ) (C : ℝ) :
  differential_eq x y → candidate_solution x y C :=
sorry

end solution_to_diff_eq_l700_700267


namespace prob_exceed_seven_in_three_draws_is_correct_l700_700358

theorem prob_exceed_seven_in_three_draws_is_correct :
  let chips := {1, 2, 3, 4, 5, 6}
  let draws := list chips
  let no_replacement := ∀ (s : list ℕ), (s.nodup)
  let draw_sum_exceeds_seven :=
    let first_two_draws := draws.take 2
    let third_draw := draws.drop 2
    (first_two_draws.sum ≤ 7) ∧ (draws.take 3).sum > 7
  ∃ draws, no_replacement draws ∧ draw_sum_exceeds_seven ∧ 
    (probability draw_sum_exceeds_seven draws = 17 / 30) :=
begin
  sorry
end

end prob_exceed_seven_in_three_draws_is_correct_l700_700358


namespace part_one_part_two_l700_700399

-- Part (I)
theorem part_one : 
  let term1 := (-27/8 : ℝ) ^ (-2/3)
  let term2 := (0.002 : ℝ) ^ (-1/2)
  let term3 := -10 * (Real.sqrt 5 - 2) ^ (-1)
  let term4 := (Real.sqrt 2 - Real.sqrt 3) ^ 0
  term1 + term2 + term3 + term4 = -167 / 9 := sorry

-- Part (II)
theorem part_two : 
  let term1 := (1 / 2 : ℝ) * Real.log (32 / 49)
  let term2 := (-4 / 3 : ℝ) * Real.log (Real.sqrt 8)
  let term3 := Real.log (Real.sqrt 245)
  term1 + term2 + term3 = 1 / 2 := sorry

end part_one_part_two_l700_700399


namespace find_a_for_weakly_increasing_g_l700_700178

-- Defining the conditions for the problem
def is_increasing_on (f : ℝ → ℝ) (M : set ℝ) : Prop :=
  ∀ x y ∈ M, x < y → f x ≤ f y

def is_decreasing_on (f : ℝ → ℝ) (M : set ℝ) : Prop :=
  ∀ x y ∈ M, x < y → f y ≤ f x

def weakly_increasing_on (g : ℝ → ℝ) (M : set ℝ) : Prop :=
  is_increasing_on g M ∧ is_decreasing_on (λ x, g x / x) M

def g (x : ℝ) (a : ℝ) : ℝ := x^2 + (4 - a) * x + a

-- The main theorem to prove the value of a
theorem find_a_for_weakly_increasing_g :
  ∃ a : ℝ, weakly_increasing_on (λ x, g x a) {x | 0 < x ∧ x ≤ 2} :=
sorry

end find_a_for_weakly_increasing_g_l700_700178


namespace cos_11pi_over_6_l700_700430

theorem cos_11pi_over_6 : Real.cos (11 * Real.pi / 6) = Real.sqrt 3 / 2 :=
by
  sorry

end cos_11pi_over_6_l700_700430


namespace cos_alpha_value_l700_700129

theorem cos_alpha_value
  (a : ℝ) (h1 : π < a ∧ a < 3 * π / 2)
  (h2 : Real.tan a = 2) :
  Real.cos a = - (Real.sqrt 5) / 5 :=
sorry

end cos_alpha_value_l700_700129


namespace solve_b_l700_700311

theorem solve_b : (a b : ℤ) (h₁ : a * b = 2 * (a + b) + 1) (h₂ : b - a = 4) : b = 7 :=
by
  sorry

end solve_b_l700_700311


namespace function_domain_l700_700194

-- Definitions of the conditions
def func_defined (x : ℝ) : Prop :=
  x ≠ 3 ∧ x ≥ -1

-- The main theorem
theorem function_domain :
  ∀ (x : ℝ), func_defined x ↔ (x ≥ -1 ∧ x ≠ 3) :=
by
  intros
  simp [func_defined]
  split
  intro h
  exact h
  intro h
  exact h

end function_domain_l700_700194


namespace power_function_decreasing_intervals_l700_700666

theorem power_function_decreasing_intervals (a : ℝ) (h : (2:ℝ) ^ a = (1 / 2 : ℝ)) :
  (∀ x ∈ Set.Ioo (0:ℝ) (2:ℝ), a = -1) ∧ (∀ x : ℝ, x ≠ 0 → x ^ a < 0) :=
by
  have h₁ : a = -1 := sorry -- Prove that a = -1 from the given point
  have h₂ : ∀ x : ℝ, x ≠ 0 → x ^ -1 < 0 := sorry -- Prove strict monotonicity
  exact ⟨h₁, h₂⟩

end power_function_decreasing_intervals_l700_700666


namespace second_offset_length_l700_700454

theorem second_offset_length (D A o1 o2 : ℝ) (hD : D = 50) (hA : A = 450) (ho1 : o1 = 10) (hA_eq : A = 1/2 * D * (o1 + o2)) :
  o2 = 8 :=
by
  -- Introduce the conditions
  rw [hD, ho1] at hA_eq
  rw hA_eq at hA
  -- Prove the correct simplified version
  sorry

end second_offset_length_l700_700454


namespace probability_of_same_color_l700_700711

def bag_condition (num_green : ℕ) (num_white : ℕ) : Prop :=
  num_green = 9 ∧ num_white = 8

noncomputable def probability_same_color (num_green num_white : ℕ) : ℚ :=
  let total := num_green + num_white in
  let prob_both_green := (num_green / total : ℚ) * ((num_green - 1) / (total - 1)) in
  let prob_both_white := (num_white / total : ℚ) * ((num_white - 1) / (total - 1)) in
  prob_both_green + prob_both_white

theorem probability_of_same_color :
  bag_condition 9 8 → probability_same_color 9 8 = 8 / 17 :=
by
  intros h
  -- skip the proof part
  have := h
  sorry

end probability_of_same_color_l700_700711


namespace log_exp_inequality_l700_700222

theorem log_exp_inequality :
    let a := log 10 (0.2)
    let b := log 3 2
    let c := real.sqrt 5
    a < b ∧ b < c := 
    sorry

end log_exp_inequality_l700_700222


namespace pages_revised_only_once_l700_700675

variable (x : ℕ)

def rate_first_time_typing := 6
def rate_revision := 4
def total_pages := 100
def pages_revised_twice := 15
def total_cost := 860

theorem pages_revised_only_once : 
  rate_first_time_typing * total_pages 
  + rate_revision * x 
  + rate_revision * pages_revised_twice * 2 
  = total_cost 
  → x = 35 :=
by
  sorry

end pages_revised_only_once_l700_700675


namespace range_of_positive_integers_in_list_k_l700_700717

theorem range_of_positive_integers_in_list_k : 
  ∀ (k : List ℤ), (k.length = 10) → (-4 = k.head) → 
    ∃ (r : ℕ), r = 4 ∧ 
    ∀ (subk : List ℤ), (subk = k.filter (λ x, x > 0)) → 
      (range (subk) = r) := 
by
  sorry

end range_of_positive_integers_in_list_k_l700_700717


namespace power_function_value_l700_700512

variable (f : ℝ → ℝ)

theorem power_function_value
  (h1 : ∃ α : ℝ, f = λ x, x ^ α)
  (h2 : f 2 = 1 / 4) :
  f (1 / 2) = 4 :=
by sorry

end power_function_value_l700_700512


namespace Karlson_drink_ratio_l700_700602

noncomputable def conical_glass_volume_ratio (r h : ℝ) : Prop :=
  let V_fuzh := (1 / 3) * Real.pi * r^2 * h
  let V_Mal := (1 / 8) * V_fuzh
  let V_Karlsson := V_fuzh - V_Mal
  (V_Karlsson / V_Mal) = 7

theorem Karlson_drink_ratio (r h : ℝ) : conical_glass_volume_ratio r h := sorry

end Karlson_drink_ratio_l700_700602


namespace range_of_a_to_decreasing_f_l700_700941

noncomputable def f (x a : ℝ) : ℝ := (2 : ℝ)^(x * (x - a))

theorem range_of_a_to_decreasing_f :
  (∀ a x : ℝ, (0 < x ∧ x < 1) → 
    monotone_decreasing (λ x, f x a)) ↔ a ∈ set.Ici 2 := sorry

end range_of_a_to_decreasing_f_l700_700941


namespace part1_part2_l700_700922

noncomputable def f (x : ℝ) : ℝ := (x - 1) / x

def a (n : ℕ) : ℝ := f (n + 1)

theorem part1 (n : ℕ) : a n < 1 := by
  sorry

theorem part2 (n : ℕ) : a (n + 1) > a n := by
  sorry

end part1_part2_l700_700922


namespace taxi_fare_relation_travel_duration_by_fare_l700_700673

def taxi_fare (x : ℝ) : ℝ :=
  if 0 < x ∧ x <= 4 then
    10
  else if 4 < x ∧ x <= 18 then
    1.2 * x + 5.2
  else if x > 18 then
    1.8 * x - 5.6
  else
    0
  -- Return 0 if x is not in one of the defined intervals

theorem taxi_fare_relation :
  ∀ x : ℝ, if 0 < x ∧ x <= 4 then taxi_fare x = 10 else
           if 4 < x ∧ x <= 18 then taxi_fare x = 1.2 * x + 5.2 else
           if x > 18 then taxi_fare x = 1.8 * x - 5.6 else
           True :=
by
  intro x
  simp [taxi_fare]
  sorry

theorem travel_duration_by_fare :
  ∀ (x : ℝ), taxi_fare x = 30.4 → x = 20 :=
by
  intro x
  simp [taxi_fare]
  -- Check each condition based on the value of x
  sorry

end taxi_fare_relation_travel_duration_by_fare_l700_700673


namespace parabola_conditions_l700_700867

-- Definitions based on conditions
def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 4*x - 3 + a

def passes_through (f : ℝ → ℝ) (x y : ℝ) : Prop :=
  f x = y

def intersects_at_2_points (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

-- Proof Problem Statement
theorem parabola_conditions (a : ℝ) :
  (passes_through (quadratic_function a) 0 1 → a = 4) ∧
  (intersects_at_2_points (quadratic_function a) → (a = 3 ∨ a = 7)) :=
by
  sorry

end parabola_conditions_l700_700867


namespace solution_set_of_x_l700_700074

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l700_700074


namespace arithmetic_seq_proof_geometric_seq_proof_l700_700188

-- Arithmetic sequence conditions and proof
def arithmetic_seq (a : ℕ → ℕ) (S : ℕ → ℕ) :=
  (a 6 = 10) ∧ (S 5 = 5)

theorem arithmetic_seq_proof (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h : arithmetic_seq a S) : a 8 = 16 :=
sorry

-- Geometric sequence conditions and proof
def geometric_seq (b : ℕ → ℕ) :=
  (b 1 + b 3 = 10) ∧ (b 4 + b 6 = 5 / 4)

theorem geometric_seq_proof (b : ℕ → ℕ)
  (h : geometric_seq b) : S 5 = 31 / 2 :=
sorry

end arithmetic_seq_proof_geometric_seq_proof_l700_700188


namespace geometric_series_product_l700_700793

theorem geometric_series_product (a r : ℝ) 
  (h₁ : a * r * (r^9) * (r^10 - 1) / (r - 1) = 18)
  (h₂ : r^10 - 1 = 6 * a * r * (r^10 - 1)) :
  ∏ (i : ℕ) in finset.range 10, a * r^(i+1) = (1/6)^55 :=
by
  -- Lean theorem to be proven here.
  sorry

end geometric_series_product_l700_700793


namespace markers_each_student_in_last_group_l700_700366

theorem markers_each_student_in_last_group 
  (total_students : ℕ) (total_groups : ℕ) (total_boxes : ℕ) (markers_per_box : ℕ)
  (students_group1 : ℕ) (markers_per_student_group1 : ℕ)
  (students_group2 : ℕ) (markers_per_student_group2 : ℕ)
  (students_group3 : ℕ) (markers_per_student_group3 : ℕ)
  (students_group4 : ℕ) (markers_per_student_group4 : ℕ) :
  total_students = 68 →
  total_groups = 5 →
  total_boxes = 48 →
  markers_per_box = 6 →
  students_group1 = 12 →
  markers_per_student_group1 = 2 →
  students_group2 = 20 →
  markers_per_student_group2 = 3 →
  students_group3 = 15 →
  markers_per_student_group3 = 5 →
  students_group4 = 8 →
  markers_per_student_group4 = 8 →
  let total_markers := total_boxes * markers_per_box in
  let used_markers_group1 := students_group1 * markers_per_student_group1 in
  let used_markers_group2 := students_group2 * markers_per_student_group2 in
  let used_markers_group3 := students_group3 * markers_per_student_group3 in
  let used_markers_group4 := students_group4 * markers_per_student_group4 in
  let total_used_markers := used_markers_group1 + used_markers_group2 + used_markers_group3 + used_markers_group4 in
  let remaining_markers := total_markers - total_used_markers in
  let students_group5 := total_students - (students_group1 + students_group2 + students_group3 + students_group4) in
  remaining_markers / students_group5 = 5 :=
by
  intros
  sorry

end markers_each_student_in_last_group_l700_700366


namespace incorrect_option_A_l700_700335

theorem incorrect_option_A (x y : ℝ) :
  ¬(5 * x + y / 2 = (5 * x + y) / 2) :=
by sorry

end incorrect_option_A_l700_700335


namespace num_squares_in_6x6_grid_l700_700031

/-- Define the number of kxk squares in an nxn grid -/
def num_squares (n k : ℕ) : ℕ := (n + 1 - k) * (n + 1 - k)

/-- Prove the total number of different squares in a 6x6 grid is 86 -/
theorem num_squares_in_6x6_grid : 
  (num_squares 6 1) + (num_squares 6 2) + (num_squares 6 3) + (num_squares 6 4) = 86 :=
by sorry

end num_squares_in_6x6_grid_l700_700031


namespace sum_of_first_150_remainder_l700_700702

theorem sum_of_first_150_remainder :
  let n := 150
  let sum := n * (n + 1) / 2
  sum % 5600 = 125 :=
by
  sorry

end sum_of_first_150_remainder_l700_700702


namespace max_product_of_roots_l700_700235

noncomputable def polynomial_2023 (a : Fin 2023 → ℝ) : Polynomial ℝ :=
  Polynomial.monomial 2023 1 + ∑ i in Fin.range 2023, Polynomial.monomial i (a i)

theorem max_product_of_roots {a : Fin 2023 → ℝ}
  (hP : ∀ x : ℝ, polynomial_2023 a x ≠ 0 → 0 ≤ x ∧ x ≤ 1)
  (h_sum : polynomial_2023 a 0 + polynomial_2023 a 1 = 0)
  (h_roots : ∃ r : Fin 2023 → ℝ, (∀ i, polynomial_2023 a r i = 0) ∧ (∀ i, 0 ≤ r i ∧ r i ≤ 1)) :
  ∃ r : Fin 2023 → ℝ, (∀ i, polynomial_2023 a r i = 0) ∧ (∀ i, 0 ≤ r i ∧ r i ≤ 1) ∧ r.prod id = 2 ^ (-2023) :=
sorry

end max_product_of_roots_l700_700235


namespace properties_of_dataset_l700_700741

def ordered_data : List ℕ := [30, 31, 31, 37, 40, 46, 47, 57, 62, 67]

def mode (l : List ℕ) : ℕ :=
  l.foldl (λ acc x => if acc.count x > l.count acc then x else acc) 0

def median (l : List ℕ) : ℕ :=
  let n := l.length
  if n % 2 = 1 then
    l.get! (n / 2)
  else
    (l.get! (n / 2 - 1) + l.get! (n / 2)) / 2

def range (l : List ℕ) : ℕ :=
  l.maximum.getD 0 - l.minimum.getD 0

def quantile (l : List ℕ) (q : ℚ) : ℕ :=
  let idx := (q * rat.ofInt l.length).floor.toNat
  (l.get! idx + l.get! (idx + 1)) / 2

theorem properties_of_dataset :
  let dataset := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30].sort
  mode dataset = 31 ∧
  median dataset ≠ 40 ∧
  range dataset = 37 ∧
  quantile dataset (10 / 100) = 30.5 := by
  sorry

end properties_of_dataset_l700_700741


namespace number_of_real_solutions_l700_700983

theorem number_of_real_solutions (a : ℝ) (h : a ≤ 0) :
  let Δ := 2^2 - 4 * a * 1 in Δ ≥ 0 ∧ (Δ = 0 → (ax^2 + 2x + 1 = 0) has 1 real root) ∧ (Δ > 0 → (ax^2 + 2x + 1 = 0) has 2 distinct real roots) :=
sorry

end number_of_real_solutions_l700_700983


namespace maximize_S_n_l700_700905

noncomputable def a_n := ℕ → ℚ -- Define the arithmetic sequence as a sequence of rational numbers

variables {a_n : ℕ → ℚ} {S_n : ℕ → ℚ} (n : ℕ) (d : ℚ)

-- Define the conditions
def condition1 (d : ℚ) := a_n 1 + a_n 3 + a_n 5 = 105
def condition2 (d : ℚ) := (a_n 1 + d) + (a_n 1 + 3 * d) + (a_n 1 + 5 * d) = 99

-- Define the sum of the first n terms of the arithmetic sequence
noncomputable def S_n (n : ℕ) := n * a_n 1 + (n * (n - 1) / 2) * d

-- Lean statement for the problem
theorem maximize_S_n :
  condition1 d → condition2 d → (∀ n, S_n n = -(n - 20)^2 + 400) → (∃ n, n = 20 ∧ ∀ m, S_n m ≤ S_n n) :=
by intros h1 h2 hS; sorry

end maximize_S_n_l700_700905


namespace trigonometric_identity_l700_700199

variable (A B C a b c : ℝ)
variable (h_triangle : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2)
variable (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
variable (h_sum_angles : A + B + C = π)
variable (h_condition : (c / b) + (b / c) = (5 * Real.cos A) / 2)

theorem trigonometric_identity 
  (h_triangle_acute : 0 < A ∧ A < π/2 ∧ 0 < B ∧ B < π/2 ∧ 0 < C ∧ C < π/2) 
  (h_sides_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_sum_angles_eq : A + B + C = π) 
  (h_given : (c / b) + (b / c) = (5 * Real.cos A) / 2) : 
  (Real.tan A / Real.tan B) + (Real.tan A / Real.tan C) = 1/2 :=
by
  sorry

end trigonometric_identity_l700_700199


namespace least_possible_value_l700_700197

theorem least_possible_value (y q p : ℝ) (h1: 5 < y) (h2: y < 7)
  (hq: q = 7) (hp: p = 5) : q - p = 2 :=
by
  sorry

end least_possible_value_l700_700197


namespace count_valid_numbers_eq_13_l700_700541

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def in_range (n : ℕ) : Prop :=
  500 ≤ n ∧ n < 700

def is_valid (n : ℕ) : Prop :=
  in_range n ∧ sum_of_digits n = 18

def count_valid_numbers : ℕ :=
  (Finset.filter is_valid (Finset.range 700).filter (λ n, n ≥ 500)).card

theorem count_valid_numbers_eq_13 :
  count_valid_numbers = 13 :=
sorry

end count_valid_numbers_eq_13_l700_700541


namespace liters_to_cubic_decimeters_eq_l700_700352

-- Define the condition for unit conversion
def liter_to_cubic_decimeter : ℝ :=
  1 -- since 1 liter = 1 cubic decimeter

-- Prove the equality for the given quantities
theorem liters_to_cubic_decimeters_eq :
  1.5 = 1.5 * liter_to_cubic_decimeter :=
by
  -- Proof to be filled in
  sorry

end liters_to_cubic_decimeters_eq_l700_700352


namespace sum_abcd_for_system_eq_l700_700429

open Real

noncomputable def refined_form (x : ℝ) :=
  ∃ (a b c d : ℕ), d ≠ 0 ∧ x = (a + b * sqrt c) / d ∨ x = (a - b * sqrt c) / d ∧
    (∀ m, (a + b * sqrt c) / d ≠ (a - m * sqrt c) / d) ∧
    (∀ n, (a - b * sqrt c) / d ≠ (a + n * sqrt c) / d) ∧
    ∀ p q r s : ℕ, p + q * sqrt r ≠ s + q * sqrt r ∧ a + b + c + d = p + q + r + s . 

theorem sum_abcd_for_system_eq : ∀ (x y : ℝ),
  x + 2 * y = 5 ∧ 4 * x * y = 9 →
  ∃ (a b c d : ℕ), (x = (a + b * sqrt c) / d ∨ x = (a - b * sqrt c) / d) ∧
  a + b + c + d = 15 :=
by sorry

end sum_abcd_for_system_eq_l700_700429


namespace range_of_a_l700_700937

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (0:ℝ) (1:ℝ), f x = 2^(x*(x-a)) ∧ monotone_decreasing_on f (Ioo (0:ℝ) (1:ℝ))) :
  a ∈ set.Ici (2 : ℝ) := sorry

end range_of_a_l700_700937


namespace count_valid_numbers_eq_13_l700_700539

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def in_range (n : ℕ) : Prop :=
  500 ≤ n ∧ n < 700

def is_valid (n : ℕ) : Prop :=
  in_range n ∧ sum_of_digits n = 18

def count_valid_numbers : ℕ :=
  (Finset.filter is_valid (Finset.range 700).filter (λ n, n ≥ 500)).card

theorem count_valid_numbers_eq_13 :
  count_valid_numbers = 13 :=
sorry

end count_valid_numbers_eq_13_l700_700539


namespace locus_of_centers_of_spheres_l700_700947

variables {A B : Point} {S : Plane}

-- The type definitions for points and planes
variable (Point : Type)
variable (Plane : Type)

-- Axiom stating the condition that points A and B are on one side of the plane S
axiom points_on_one_side_of_plane : (A B : Point) (S : Plane), 
  ¬(A ∈ S ∧ B ∈ S)

-- Define paraboloid, parabola, and ellipse as geometric loci
noncomputable def paraboloid_of_revolution : Type := sorry
noncomputable def parabola : Type := sorry
noncomputable def ellipse : Type := sorry

-- The theorem stating the locus of the centers of spheres in different cases
theorem locus_of_centers_of_spheres (A B : Point) (S : Plane) 
  (h : points_on_one_side_of_plane A B S) :
  locus_of_centers_of_spheres A B S = 
    if A = B then paraboloid_of_revolution
    else if (line_through A B).parallel_to S then parabola
    else ellipse := sorry

end locus_of_centers_of_spheres_l700_700947


namespace calc_expr_eq_l700_700009

theorem calc_expr_eq : 2 + 3 / (4 + 5 / 6) = 76 / 29 := 
by 
  sorry

end calc_expr_eq_l700_700009


namespace cows_after_two_years_l700_700955

def initial_cows : ℕ := 200
def growth_rate : ℝ := 0.5

def cows_after_years (initial : ℕ) (rate : ℝ) (years: ℕ) : ℕ :=
  nat.floor $ initial * (1 + rate)^years

theorem cows_after_two_years : cows_after_years initial_cows growth_rate 2 = 450 := by
  sorry

end cows_after_two_years_l700_700955


namespace factorize_difference_of_squares_l700_700438

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l700_700438


namespace problem_solution_l700_700902

variable (A B C D P M N : Point)
variable (AB AD AP PC PD : Vector)

-- Assumption: ABCD is a rectangle, and PA is perpendicular to the plane ABCD.
-- Let's encode this:
axiom H1 : Rectangle ABCD
axiom H2 : PA ⊥ plane_of AB AD

-- Given conditions on the vectors:
axiom H3 : PM = 1/2 * PC
axiom H4 : PN = 2/3 * PD

-- Target to prove:
def x_y_z_sum : ℝ :=
  let MN := (2/3 * (AD - AP)) - (1/2 * (AB + AD - AP))
  let x := -(1/2 : ℝ)
  let y := (1/6 : ℝ)
  let z := -(1/6 : ℝ)
  x + y + z

theorem problem_solution : x_y_z_sum = -(1/2 : ℝ) :=
by
  sorry

end problem_solution_l700_700902


namespace solution_set_l700_700070

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l700_700070


namespace triangle_area_upper_bound_l700_700644

variable {α : Type u}
variable [LinearOrderedField α]
variable {A B C : α} -- Points A, B, C as elements of some field.

-- Definitions for the lengths of the sides, interpreted as scalar distances.
variable (AB AC : α)

-- Assume that AB and AC are lengths of sides of the triangle
-- Assume the area of the triangle is non-negative and does not exceed the specified bound.
theorem triangle_area_upper_bound (S : α) (habc : S = (1 / 2) * AB * AC) :
  S ≤ (1 / 2) * AB * AC := 
sorry

end triangle_area_upper_bound_l700_700644


namespace triangle_side_b_eq_l700_700595

   variable (a b c : Real) (A B C : Real)
   variable (cos_A sin_A : Real)
   variable (area : Real)
   variable (π : Real := Real.pi)

   theorem triangle_side_b_eq :
     cos_A = 1 / 3 →
     B = π / 6 →
     a = 4 * Real.sqrt 2 →
     sin_A = 2 * Real.sqrt 2 / 3 →
     b = (a * sin_B / sin_A) →
     b = 3 := sorry
   
end triangle_side_b_eq_l700_700595


namespace sum_possible_values_x2_sum_of_x2_values_l700_700743

theorem sum_possible_values_x2 (p q x1 x2 : ℤ) (h1 : x = λ x : ℤ, x^2 - p*x + q) 
  (h2 : x1 + x2 = p) (h3 : x1 * x2 = q) (h4 : x1 = x2 + d) (h5 : q = x2) 
  (h6 : x1 > x2 ∧ x2 > q) : (x2 = -3 ∨ x2 = -2) → x2 ∈ {-3, -2} :=
by
  sorry

theorem sum_of_x2_values : -3 + (-2) = -5 :=
by
  sorry

end sum_possible_values_x2_sum_of_x2_values_l700_700743


namespace magnitude_of_complex_expr_is_sqrt13_l700_700908

noncomputable
def magnitude_of_complex_expr: ℂ :=
  complex.abs ((5 - complex.I) / (1 + complex.I))

theorem magnitude_of_complex_expr_is_sqrt13 :
  magnitude_of_complex_expr = Real.sqrt 13 :=
by
  sorry

end magnitude_of_complex_expr_is_sqrt13_l700_700908


namespace gas_cost_per_gallon_is_4_l700_700794

noncomputable def cost_per_gallon (miles_per_gallon : ℝ) (total_miles : ℝ) (total_cost : ℝ) : ℝ :=
  total_cost / (total_miles / miles_per_gallon)

theorem gas_cost_per_gallon_is_4 :
  cost_per_gallon 32 432 54 = 4 := by
  sorry

end gas_cost_per_gallon_is_4_l700_700794


namespace arccos_cos_3pi_div_2_l700_700411

theorem arccos_cos_3pi_div_2 : real.arccos (real.cos (3 * real.pi / 2)) = real.pi / 2 := 
by 
  sorry

end arccos_cos_3pi_div_2_l700_700411


namespace movie_length_l700_700209

theorem movie_length (visit_frequency: ℕ) (num_visits: ℕ) (minutes_in_hour: ℕ) (movie_length_in_hours: ℝ) :
  visit_frequency = 50 →
  num_visits = 3 →
  minutes_in_hour = 60 →
  movie_length_in_hours = (visit_frequency * num_visits : ℕ) / minutes_in_hour :=
begin
  intros h1 h2 h3,
  sorry
end

end movie_length_l700_700209


namespace distinct_x_intercepts_l700_700163

def polynomial := (x : ℝ) → (x - 2) * (x ^ 2 + 6 * x + 9)

theorem distinct_x_intercepts : {x : ℝ | polynomial x = 0}.to_finset.card = 2 := by
sorry

end distinct_x_intercepts_l700_700163


namespace complement_of_P_in_U_l700_700159

open Set

theorem complement_of_P_in_U :
  let U := ℝ
  let P := {x : ℝ | x^2 ≤ 1}
  ∁ U P = {x : ℝ | x < -1} ∪ {x : ℝ | x > 1} :=
by
  sorry

end complement_of_P_in_U_l700_700159


namespace exists_two_vertices_with_singularity_sum_le_four_l700_700250

def convex_polyhedron (V E F : Type) [Fintype V] [Fintype E] [Fintype F] :=
  -- Definitions for vertices, edges, and faces of a polyhedron
sorry

def color (E : Type) := E → bool -- Coloring of edges (true for red, false for yellow)

def singular_face_angle {V E : Type} [Fintype E] (c : color E) (a b : E) : bool :=
  (c a) ≠ (c b) -- Face angle is singular if edges have different colors

def singularity_degree {V E F : Type} [Fintype V] [Fintype E] [Fintype F]
  (P : convex_polyhedron V E F) (c : color E) (v : V) : ℕ :=
  -- Calculate the number of singular face angles at a vertex
sorry

theorem exists_two_vertices_with_singularity_sum_le_four
  {V E F : Type} [Fintype V] [Fintype E] [Fintype F]
  (P : convex_polyhedron V E F) (c : color E) :
  ∃ (B C : V), singularity_degree P c B + singularity_degree P c C ≤ 4 :=
sorry

end exists_two_vertices_with_singularity_sum_le_four_l700_700250


namespace interior_surface_area_is_correct_l700_700271

-- Define the original dimensions of the rectangular sheet
def original_length : ℕ := 40
def original_width : ℕ := 50

-- Define the side length of the square corners
def corner_side : ℕ := 10

-- Define the area of the original sheet
def area_original : ℕ := original_length * original_width

-- Define the area of one square corner
def area_corner : ℕ := corner_side * corner_side

-- Define the total area removed by all four corners
def area_removed : ℕ := 4 * area_corner

-- Define the remaining area after the corners are removed
def area_remaining : ℕ := area_original - area_removed

-- The theorem to be proved
theorem interior_surface_area_is_correct : area_remaining = 1600 := by
  sorry

end interior_surface_area_is_correct_l700_700271


namespace distance_between_foci_l700_700863

noncomputable def distance_between_foci_ellipse : ℝ :=
  let a_sq := 613 / 100 in
  let b_sq := 613 / 400 in
  let c := Real.sqrt (a_sq - b_sq) in
  2 * c

theorem distance_between_foci :
  ∀ a b c d e, a * x^2 + b * x + c * y^2 + d * y + e = 0 →
  a = 25 ∧ b = -125 ∧ c = 4 ∧ d = 8 ∧ e = 16 →
  abs (distance_between_foci_ellipse - 4.3) < 0.1 := 
by 
  sorry

end distance_between_foci_l700_700863


namespace rhombus_perimeter_l700_700280

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l700_700280


namespace hania_age_in_five_years_l700_700584

-- Defining the conditions
variables (H S : ℕ)

-- First condition: Samir's age will be 20 in five years
def condition1 : Prop := S + 5 = 20

-- Second condition: Samir is currently half the age Hania was 10 years ago
def condition2 : Prop := S = (H - 10) / 2

-- The statement to prove: Hania's age in five years will be 45
theorem hania_age_in_five_years (H S : ℕ) (h1 : condition1 S) (h2 : condition2 H S) : H + 5 = 45 :=
sorry

end hania_age_in_five_years_l700_700584


namespace value_of_f_5_l700_700338

theorem value_of_f_5 (f : ℕ → ℕ) (y : ℕ)
  (h1 : ∀ x, f x = 2 * x^2 + y)
  (h2 : f 2 = 20) : f 5 = 62 :=
sorry

end value_of_f_5_l700_700338


namespace gray_region_area_l700_700406

noncomputable def area_of_gray_region : ℝ :=
  let r := 3 in
  let centerC := (5 : ℝ, 5 : ℝ) in
  let centerD := (14 : ℝ, 5 : ℝ) in
  let area_rectangle := (14 - 5) * 5 in
  let area_sector := (r * r * Real.pi) / 4 in
  area_rectangle - (2 * area_sector)

theorem gray_region_area :
  area_of_gray_region = 45 - (9 * Real.pi / 2) := by
  sorry

end gray_region_area_l700_700406


namespace quadratic_solutions_l700_700980

theorem quadratic_solutions (a : ℝ) (h : a ≤ 0) : 
  ∃ n, (n = 1 ∨ n = 2) ∧ (ax^2 + 2*x + 1 = 0).roots.length = n :=
by
  sorry

end quadratic_solutions_l700_700980


namespace betty_min_sugar_flour_oats_l700_700395

theorem betty_min_sugar_flour_oats :
  ∃ (s f o : ℕ), f ≥ 4 + 2 * s ∧ f ≤ 3 * s ∧ o = f + s ∧ s = 4 :=
by
  sorry

end betty_min_sugar_flour_oats_l700_700395


namespace regular_pyramid_angle_l700_700196

noncomputable def angle_between_line_and_plane (S A B C D O P : Point) : ℝ :=
sorry

theorem regular_pyramid_angle 
  (S A B C D O P : Point)
  (h1 : regular_pyramid S A B C D)
  (h2 : midpoint O (line S D))
  (h3 : midpoint P (segment S D))
  (h4 : dist S O = dist O D) :
  angle_between_line_and_plane B C (plane P A C) = 45 :=
sorry

end regular_pyramid_angle_l700_700196


namespace distinct_values_3_pow_frac_l700_700109

theorem distinct_values_3_pow_frac :
  let S := {1, 2, 3, 4, 5, 6}
  let distinctPairs := { p : Finset (ℕ × ℕ) // p ∈ (S.product S).filter (λ ab, ab.1 ≠ ab.2) }
  let exponentValues := distinctPairs.map (λ p, (3 : ℝ)^(p.1 / p.2))
  ∃ n, (exponentValues.toFinset.card = 22) :=
by sorry

end distinct_values_3_pow_frac_l700_700109


namespace floor_equation_solution_l700_700046

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l700_700046


namespace solution_to_geometric_sequence_l700_700681

variable {a r : ℝ}

-- Condition that all terms are positive and the given equation
axiom pos_terms (a r : ℝ) (h₁ : a > 0) (h₂ : r > 0) :
  a * a * r^2 + a * r * a * r^5 + 2 * (a * r^2)^2 = 36

-- Define the terms in the geometric sequence
def a₂ := a * r
def a₄ := a * r^3

theorem solution_to_geometric_sequence :
  a₂ + a₄ = 6 :=
  sorry

end solution_to_geometric_sequence_l700_700681


namespace proof_problem_l700_700473

-- Define the minimum function for two real numbers
def min (x y : ℝ) := if x < y then x else y

-- Define the real numbers sqrt30, a, and b
def sqrt30 := Real.sqrt 30

variables (a b : ℕ)

-- Define the conditions
def conditions := (min sqrt30 a = a) ∧ (min sqrt30 b = sqrt30) ∧ (b = a + 1)

-- State the theorem to prove
theorem proof_problem (h : conditions a b) : 2 * a - b = 4 :=
sorry

end proof_problem_l700_700473


namespace sum_in_range_l700_700008

noncomputable def mixed_number_sum : ℚ :=
  3 + 1/8 + 4 + 3/7 + 6 + 2/21

theorem sum_in_range : 13.5 ≤ mixed_number_sum ∧ mixed_number_sum < 14 := by
  sorry

end sum_in_range_l700_700008


namespace perception_permutations_count_l700_700825

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l700_700825


namespace parallel_lines_m_l700_700501

theorem parallel_lines_m (m : ℝ) :
  let A := (-6 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 6 : ℝ)
  let C := (0 : ℝ, -18 : ℝ)
  let D := (18 : ℝ, m)
  (-18 / 6) = ((m - 6) / 18) ↔ m = -48 :=
by
  let A := (-6 : ℝ, 0 : ℝ)
  let B := (0 : ℝ, 6 : ℝ)
  let C := (0 : ℝ, -18 : ℝ)
  let D := (18 : ℝ, m)
  have slope_AC : -18 / 6 = -3 := rfl
  have slope_BD : ((m - 6) / 18) = -3 ↔ m = -48 := sorry
  exact ⟨slope_BD.mp, slope_BD.mpr⟩

end parallel_lines_m_l700_700501


namespace linear_equation_conditions_l700_700177

theorem linear_equation_conditions (m n : ℤ) :
  (∀ x y : ℝ, 4 * x^(m - n) - 5 * y^(m + n) = 6 → 
    m - n = 1 ∧ m + n = 1) →
  m = 1 ∧ n = 0 :=
by
  sorry

end linear_equation_conditions_l700_700177


namespace cos_B_value_l700_700125

theorem cos_B_value (a b c : ℝ) (A B : ℝ) 
(h1 : b = 3) (h2 : c = 1) (h3 : A = 2 * B) 
(h4 : ∀ {x y z : ℝ}, x^2 = y^2 + z^2 - 2*y*z*(Real.cos (x / y z))) : 
(Real.cos B = (Real.sqrt 3 / 3)) :=
by sorry

end cos_B_value_l700_700125


namespace total_shaded_area_in_grid_is_nine_l700_700787

/-- In a 6x6 grid, given a spinner-like figure centered at (3, 3) which extends 1 unit in each
direction (left, right, up, downward) and includes four additional 1x1 square blocks touching
each corner of the spinner diagonally, the total area of the shaded region is 9 square units. -/
theorem total_shaded_area_in_grid_is_nine 
  (grid_size : ℕ)
  (center : ℕ × ℕ)
  (spinner_extension : ℕ)
  (additional_blocks : ℕ)
  (total_shaded_area : ℕ) 
  (h1 : grid_size = 6)
  (h2 : center = (3, 3))
  (h3 : spinner_extension = 1)
  (h4 : additional_blocks = 4)
  (H : total_shaded_area = 9) :
  total_shaded_area = 9 :=
by
  rw [h1, h2, h3, h4, H]
  sorry

end total_shaded_area_in_grid_is_nine_l700_700787


namespace cost_price_of_table_l700_700671

theorem cost_price_of_table (C S : ℝ) (h1 : S = 1.25 * C) (h2 : S = 4800) : C = 3840 := 
by 
  sorry

end cost_price_of_table_l700_700671


namespace height_of_trapezoid_l700_700456

theorem height_of_trapezoid (a b : ℝ) (c d: ℝ) (h: ℝ) : 
  a = 4 → b = 14 → c = 6 → d = 8 → h = 4.8 := 
by 
  assume (ha : a = 4) (hb : b = 14) (hc : c = 6) (hd : d = 8)
  -- sorry gives the expected height and proves the theorem 
  sorry

end height_of_trapezoid_l700_700456


namespace rhombus_perimeter_l700_700278

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l700_700278


namespace find_projection_vector_l700_700951

noncomputable def vector_p : ℝ × ℝ × ℝ := (74 / 53, 22 / 53, 132 / 53)

theorem find_projection_vector
    (a' : ℝ × ℝ × ℝ)
    (b' : ℝ × ℝ × ℝ)
    (c  : ℝ × ℝ × ℝ)
    (collinear : ∃ v : ℝ × ℝ × ℝ, ∀ t : ℝ, a' = (2 - 2 * t, -2 + 8 * t, 4 - 5 * t) ∧ b' = (0 + 6 * t, 6 + 6 * t, -1 - 1 * t) ∧ v = a' ∧ v = b')
    (parallel_b' : b' = (0, 3, -0.5)) :

    (∃ p' : ℝ × ℝ × ℝ, p' = vector_p ∧ 
        ( ∀ t : ℝ, ∃ t' : ℝ, p' = a' + (2 * t - 4 * t', -2 * t + 4 * t', 4 - 2.5 * t') ∧ p' = b' + (0 - t, 9t, 4t - t')) ∧ 
        (p'.1 * c.1 + p'.2 * c.2 + p'.3 * c.3 = 0)) :=
sorry

end find_projection_vector_l700_700951


namespace triangle_area_problem_l700_700372

theorem triangle_area_problem (r s: ℝ) :
  let Q := (0, 6)
  let P := (9, 0)
  let area_POQ := (1/2) * 9 * 6
  let area_TOP := (1/2) * 9 * s
  (s = 1.5) → 
  (T : ℝ × ℝ) (hT : T = (r, s)) → 
  T ∈ segment ℝ P Q → 
  (area_POQ = 4 * area_TOP) →
  (r + s = 7.5) :=
begin
  assume Q_def P_def area_POQ_def area_TOP_def hs,
  assume T hT hT_seg area_ratio,
  sorry
end

end triangle_area_problem_l700_700372


namespace two_a_minus_b_equals_four_l700_700471

theorem two_a_minus_b_equals_four (a b : ℕ) 
    (consec_integers : b = a + 1)
    (min_a : min (Real.sqrt 30) a = a)
    (min_b : min (Real.sqrt 30) b = Real.sqrt 30) : 
    2 * a - b = 4 := 
sorry

end two_a_minus_b_equals_four_l700_700471


namespace cyclist_traveled_18_miles_l700_700749

noncomputable def cyclist_distance (v t d : ℕ) : Prop :=
  (d = v * t) ∧ 
  (d = (v + 1) * (3 * t / 4)) ∧ 
  (d = (v - 1) * (t + 3))

theorem cyclist_traveled_18_miles : ∃ (d : ℕ), cyclist_distance 3 6 d ∧ d = 18 :=
by
  sorry

end cyclist_traveled_18_miles_l700_700749


namespace solution_set_sum_of_squares_l700_700679

noncomputable def pi : ℝ := real.pi

theorem solution_set_sum_of_squares :
  (∑ x in {x : ℝ | x^2 + x - 1 = x * pi^(x^2 - 1) + (x^2 - 1) * pi^x}, x^2) = 2 := 
  sorry

end solution_set_sum_of_squares_l700_700679


namespace find_n_for_f_zero_l700_700923

noncomputable def f (a b x : ℝ) := log a x + x - b

theorem find_n_for_f_zero (a b x_0 : ℝ) (h_a1 : 0 < a) (h_a2 : a ≠ 1) (h_a3 : 2 < a) (h_a4 : a < 3) 
  (h_b1 : 3 < b) (h_b2 : b < 4) (h_fx0 : f a b x_0 = 0) :
  ∃ n : ℕ, (n ≠ 0) ∧ (x_0 ∈ Ioo (n : ℝ) (n + 1)) ∧ n = 2 :=
by
  sorry

end find_n_for_f_zero_l700_700923


namespace odd_sum_subsets_card_l700_700961

def set_nums := {41, 57, 82, 113, 190, 201}
def is_odd (n : ℕ) : Prop := n % 2 = 1

noncomputable def odd_sum_subsets (s : Finset ℕ) (n : ℕ) : Finset (Finset ℕ) :=
  s.powerset.filter (λ t => t.card = n ∧ is_odd (t.sum id))

theorem odd_sum_subsets_card :
  (odd_sum_subsets {41, 57, 82, 113, 190, 201} 3).card = 8 :=
  sorry

end odd_sum_subsets_card_l700_700961


namespace EM_eq_half_BD_l700_700991

variables (A B C D E M H : Type*)
variables [has_angle A B C] [has_midpoint B C M] [has_perpendicular BD H] [has_angle_bisector BD A]

-- Assuming the given conditions are defined
def given_conditions : Prop :=
  angle ABC = 5 * angle ACB ∧
  angle_bisector BD A ∧
  perpendicular BD H ∧
  perpendicular DE BC ∧
  midpoint B C M

theorem EM_eq_half_BD (h : given_conditions A B C D E M H) : EM = (1/2) * BD :=
sorry

end EM_eq_half_BD_l700_700991


namespace solution_set_of_x_l700_700076

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l700_700076


namespace yellow_marbles_l700_700210

-- Define the conditions from a)
variables (total_marbles red blue green yellow : ℕ)
variables (h1 : total_marbles = 110)
variables (h2 : red = 8)
variables (h3 : blue = 4 * red)
variables (h4 : green = 2 * blue)
variables (h5 : yellow = total_marbles - (red + blue + green))

-- Prove the question in c)
theorem yellow_marbles : yellow = 6 :=
by
  -- Proof will be inserted here
  sorry

end yellow_marbles_l700_700210


namespace subset_interval_l700_700623

theorem subset_interval (a : ℝ) : 
  (∀ x : ℝ, (-a-1 < x ∧ x < -a+1 → -3 < x ∧ x < 1)) ↔ (0 ≤ a ∧ a ≤ 2) := 
by
  sorry

end subset_interval_l700_700623


namespace factorization_6x2_minus_24x_plus_18_l700_700110

theorem factorization_6x2_minus_24x_plus_18 :
    ∀ x : ℝ, 6 * x^2 - 24 * x + 18 = 6 * (x - 1) * (x - 3) :=
by
  intro x
  sorry

end factorization_6x2_minus_24x_plus_18_l700_700110


namespace hydrochloric_acid_mixture_l700_700356

theorem hydrochloric_acid_mixture (x : ℝ) (hx : 0.3 * x + 0.1 * (600 - x) = 90) :
    x = 150 ∧ (600 - x) = 450 :=
begin
  sorry
end

end hydrochloric_acid_mixture_l700_700356


namespace total_money_shared_l700_700798

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l700_700798


namespace angle_C_proof_l700_700896

theorem angle_C_proof
  (A B C : Type)
  [triangle A B C]
  (h1 : perimeter A B C = sqrt 2 + 1)
  (h2 : area A B C = 1 / 6 * sin C)
  (h3 : sin A + sin B = sqrt 2 * sin C)
  : C = π / 3 := by
  sorry

end angle_C_proof_l700_700896


namespace max_subset_T_l700_700868

-- Definition of the set S
def S : Type := { x : Fin 5 → Bool // ∀ i, x i = false ∨ x i = true }

-- Definition of the distance function d
def d (A B : S) : ℕ :=
  Finset.univ.sum (λ i, if A.val i = B.val i then 0 else 1)

-- Definition of the condition
def condition (T : Finset S) : Prop :=
  ∀ A B ∈ T, A ≠ B → d A B > 2

-- The statement to be proved: 
theorem max_subset_T (T : Finset S) (h : condition T) : T.card ≤ 4 :=
by sorry

end max_subset_T_l700_700868


namespace ball_selection_count_l700_700091

theorem ball_selection_count :
  let even := ∑ (k : ℕ), (k % 2 = 0)
  let odd := ∑ (k : ℕ), (k % 2 = 1)
in
∑ (r g y : ℕ) in { (2005, even, _) ∪ (2005, _, odd) }, 1
= Nat.binomial 2007 2 - Nat.binomial 1004 2 :=
sorry

end ball_selection_count_l700_700091


namespace correct_statements_l700_700924

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := x * Real.log x - m * x^2

theorem correct_statements (m : ℝ) :
  (∀ (x > 0), 1 + Real.log x - 2 * m * x ≤ 0 → m ≥ 1 / 2) ∧
  (m = 1 / Real.exp 1 → ∃ e, f e m = 0 ∧ ∀ x, f' x m = 0 → f' e m = 0) ∧
  ((m ≤ 0 ∨ m = 1 / Real.exp 1) → ∃! x, f x m = 0) :=
  sorry

end correct_statements_l700_700924


namespace floor_eq_l700_700062

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l700_700062


namespace gain_percent_l700_700565

variables (C S : ℝ)

theorem gain_percent (h : 24 * C = 16 * S) : (S - C) / C * 100 = 50 := by
  have h1 : S = 3 * C / 2 := by
    linarith [h]
  have gain : S - C = C / 2 := by
    linarith [h1]
  suffices (S - C) / C * 100 = 50 from this
  linarith [gain]

end gain_percent_l700_700565


namespace hotel_rooms_l700_700371

theorem hotel_rooms (h₁ : ∀ R : ℕ, (∃ n : ℕ, n = R * 3) → (∃ m : ℕ, m = 2 * R * 3) → m = 60) : (∃ R : ℕ, R = 10) :=
by
  sorry

end hotel_rooms_l700_700371


namespace find_matrix_N_l700_700086

open Matrix

def std_basis (i : Fin 3) : Fin 3 → ℝ
| ⟨0, _⟩ := if i = 0 then 1 else 0
| ⟨1, _⟩ := if i = 1 then 1 else 0
| ⟨2, _⟩ := if i = 2 then 1 else 0
| _ := 0

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ :=
  λ i j, if j = 0 then ![4, -1, 0] i else if j = 1 then ![-2, 6, 3] i else ![9, 4, -5] i

theorem find_matrix_N :
  (N ⬝ (λ i, (std_basis i))) 0 = ![4, -1, 0] ∧
  (N ⬝ (λ i, (std_basis i))) 1 = ![-2, 6, 3] ∧
  (N ⬝ (λ i, (std_basis i))) 2 = ![9, 4, -5] :=
by 
  sorry

end find_matrix_N_l700_700086


namespace more_green_than_blue_l700_700433

-- Definition of the ratio and total
def ratio_blue : ℕ := 3
def ratio_yellow : ℕ := 7
def ratio_green : ℕ := 8
def ratio_red : ℕ := 9
def total_parts : ℕ := ratio_blue + ratio_yellow + ratio_green + ratio_red
def total_disks : ℕ := 180

-- Number of disks per part
def disks_per_part : ℕ := total_disks / total_parts

-- Number of disks of each color
def blue_disks : ℕ := ratio_blue * disks_per_part
def green_disks : ℕ := ratio_green * disks_per_part

-- Theorem stating the number of more green disks than blue disks
theorem more_green_than_blue : green_disks - blue_disks = 35 := by
  -- Declaring our conditions
  have total_disks_condition : total_disks = 180 := rfl
  have ratio_condition : total_parts = 27 := rfl
  have parts_condition : disks_per_part = 7 := rfl
  have blue_condition : blue_disks = ratio_blue * disks_per_part := rfl
  have green_condition : green_disks = ratio_green * disks_per_part := rfl
  
  -- Rewriting with these conditions
  rw [blue_condition, green_condition, parts_condition, ratio_blue, ratio_green]
  -- Showing actual result of 56 - 21 equals 35
  exact rfl

-- Place sorry to skip the proof part if necessary

end more_green_than_blue_l700_700433


namespace bad_carrots_count_l700_700784

/-- 
Carol and her mom were picking carrots from their garden. 
Carol picked 29 and her mother picked 16. 
If only 38 of the carrots were good, how many bad carrots did they have? 
-/
theorem bad_carrots_count : 
  let total_carrots := 29 + 16 in
  let good_carrots := 38 in
  let bad_carrots := total_carrots - good_carrots in
  bad_carrots = 7 :=
by
  sorry

end bad_carrots_count_l700_700784


namespace percentage_increase_after_decrease_l700_700310

theorem percentage_increase_after_decrease (P : ℝ) :
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  (P_decreased * (1 + x / 100) = P_final) → x = 65.71 := 
by 
  intros
  let P_decreased := 0.70 * P
  let P_final := 1.16 * P
  let x := ((P_final / P_decreased) - 1) * 100
  have h : (P_decreased * (1 + x / 100) = P_final) := by assumption
  sorry

end percentage_increase_after_decrease_l700_700310


namespace part1_part2_l700_700525

def f (x a : ℝ) : ℝ := |x - a| + 2 * |x - 1|

theorem part1 (x : ℝ) : f x 2 > 5 ↔ x < - 1 / 3 ∨ x > 3 :=
by sorry

theorem part2 (a : ℝ) : (∃ x : ℝ, f x a ≤ |a - 2|) → a ≤ 3 / 2 :=
by sorry

end part1_part2_l700_700525


namespace tangent_line_at_zero_monotonic_decreasing_on_ℝ_extreme_point_proof_l700_700927

open Real

noncomputable def f (a x : ℝ) := ((a + 1) * x^2 + 3 * (a + 1) * x + a + 6) / (exp x)
noncomputable def f' (a x : ℝ) := -( (a + 1) * x^2 + (a + 1) * x + 3 - 2 * a ) / exp x

-- Part (1)
theorem tangent_line_at_zero (a : ℝ) (ha : a = -1) : 
  let f_x0 := f a 0
  let f'_x0 := f' a 0
  f_x0 = 5 → f'_x0 = -5 →
  ∀ (y x : ℝ), 5 * x + y - 5 = 0 := 
    by sorry

-- Part (2)
theorem monotonic_decreasing_on_ℝ (a : ℝ) : 
  (∀ x, f' a x ≤ 0) ↔  a ∈ set.Icc (-1 : ℝ) (11 / 9 : ℝ) := 
    by sorry

-- Part (3)
theorem extreme_point_proof (a : ℝ) (ha : a > 3) (x1 x2 : ℝ) (hx1hx2 : 
  (a + 1) * x1 ^ 2 + (a + 1) * x1 + 3 - 2 * a = 0 ∧ 
  (a + 1) * x2 ^ 2 + (a + 1) * x2 + 3 - 2 * a = 0 ∧ 
  x1 > x2) : 2 * x1 + x2 > -1 / 2 := 
    by sorry

end tangent_line_at_zero_monotonic_decreasing_on_ℝ_extreme_point_proof_l700_700927


namespace num_integers_with_digit_sum_18_l700_700544

-- Defining the conditions:
def is_digit_sum_18 (n : ℕ) := 
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 + d2 + d3 = 18

def within_bounds (n : ℕ) := 500 ≤ n ∧ n ≤ 700

-- The main theorem statement:
theorem num_integers_with_digit_sum_18 : {n : ℕ | within_bounds n ∧ is_digit_sum_18 n}.to_finset.card = 25 :=
by
  sorry

end num_integers_with_digit_sum_18_l700_700544


namespace find_angle_Q_l700_700027

noncomputable def hexagon_angle_sum := 720
def angle_B := 140
def angle_C := 100
def angle_D := 125
def angle_E := 130
def angle_F := 110

theorem find_angle_Q : (∃ Q : ℕ, Q + angle_B + angle_C + angle_D + angle_E + angle_F = hexagon_angle_sum) → (∃ Q, Q = 115) :=
by {
    intros h,
    sorry
}

end find_angle_Q_l700_700027


namespace isosceles_triangle_largest_angle_l700_700581

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_iso : A = B) (h_A : C = 50) :
  max A (max B (180 - A - B)) = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l700_700581


namespace possible_values_Tm_l700_700494

noncomputable def possibleValuesOfTm (ai : ℕ → ℝ × ℝ) (m : ℕ) : set ℝ :=
{ Tm_norm |
  let Tm := ∑ i in (finset.range m), vector2.mk (ai i).1 (ai i).2 in
  Tm_norm = norm Tm }

theorem possible_values_Tm (ai : ℕ → ℝ × ℝ)
    (h1 : ∀ i, (ai i).1 * (ai i).1 + (ai i).2 * (ai i).2 = 4)
    (h2 : ∀ i, (ai i).1 * (ai (i + 1)).1 + (ai i).2 * (ai (i + 1)).2 = 0)
    (m : ℕ) (hm : 2 ≤ m) :
  possibleValuesOfTm ai m = {0, 2, 2 * real.sqrt 2} := 
sorry

end possible_values_Tm_l700_700494


namespace pencils_across_diameter_l700_700964

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l700_700964


namespace estimate_is_less_l700_700390

-- Definitions of rounding conditions
section rounding

variables (x y z : ℕ) (hx : x > 0) (hy : y > 0) (hz : z > 0)

def round_up (n : ℕ) : ℕ := n + 1
def round_down (n : ℕ) : ℕ := n - 1

-- Define the estimated and exact values of the expression
def exact_value : ℝ := x / y + z
def estimated_value : ℝ := (round_down x) / (round_down y) + (round_up z)

-- The theorem we need to prove
theorem estimate_is_less : estimated_value x y z < exact_value x y z :=
by {
  sorry -- Proof goes here
}

end rounding

end estimate_is_less_l700_700390


namespace three_digit_count_exactly_17_more_than_two_digit_l700_700166

theorem three_digit_count_exactly_17_more_than_two_digit :
  {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ ∃ m : ℕ, 10 ≤ m ∧ m ≤ 99 ∧ n = m + 17}.card = 17 := 
begin
  sorry 
end

end three_digit_count_exactly_17_more_than_two_digit_l700_700166


namespace parallelogram_area_l700_700347

noncomputable def angle_ABC : ℝ := 30
noncomputable def AX : ℝ := 20
noncomputable def CY : ℝ := 22

theorem parallelogram_area (angle_ABC_eq : angle_ABC = 30)
    (AX_eq : AX = 20)
    (CY_eq : CY = 22)
    : ∃ (BC : ℝ), (BC * AX = 880) := sorry

end parallelogram_area_l700_700347


namespace least_three_digit_7_heavy_l700_700767

-- Define what it means for a number to be "7-heavy"
def is_7_heavy(n : ℕ) : Prop := n % 7 > 4

-- Smallest three-digit number
def smallest_three_digit_number : ℕ := 100

-- Least three-digit 7-heavy whole number
theorem least_three_digit_7_heavy : ∃ n, smallest_three_digit_number ≤ n ∧ is_7_heavy(n) ∧ ∀ m, smallest_three_digit_number ≤ m ∧ is_7_heavy(m) → n ≤ m := 
  sorry

end least_three_digit_7_heavy_l700_700767


namespace smallest_f_for_perfect_square_l700_700974

theorem smallest_f_for_perfect_square (f : ℕ) (h₁: 3150 = 2 * 3 * 5^2 * 7) (h₂: ∃ m : ℕ, 3150 * f = m^2) :
  f = 14 :=
sorry

end smallest_f_for_perfect_square_l700_700974


namespace move_point_right_l700_700176

theorem move_point_right (A : ℝ × ℝ) (x y : ℝ) (hA : A = (x, y)) : A' = (3, 2) :=
by
  have hx : x = 1 := by sorry
  have hy : y = 2 := by sorry
  have hA' : A' = (x + 2, y) := by sorry
  show A' = (3, 2) from by simp [hx, hy, hA', hA, add_comm]
  sorry

end move_point_right_l700_700176


namespace string_length_correct_l700_700363

noncomputable def cylinder_circumference : ℝ := 6
noncomputable def cylinder_height : ℝ := 18
noncomputable def number_of_loops : ℕ := 6

noncomputable def height_per_loop : ℝ := cylinder_height / number_of_loops
noncomputable def hypotenuse_per_loop : ℝ := Real.sqrt (cylinder_circumference ^ 2 + height_per_loop ^ 2)
noncomputable def total_string_length : ℝ := number_of_loops * hypotenuse_per_loop

theorem string_length_correct :
  total_string_length = 18 * Real.sqrt 5 := by
  sorry

end string_length_correct_l700_700363


namespace number_of_cows_on_boat_l700_700732

-- Definitions based on conditions
def number_of_sheep := 20
def number_of_dogs := 14
def sheep_drowned := 3
def cows_drowned := 2 * sheep_drowned  -- Twice as many cows drowned as did sheep.
def dogs_made_it_shore := number_of_dogs  -- All dogs made it to shore.
def total_animals_shore := 35
def total_sheep_shore := number_of_sheep - sheep_drowned
def total_sheep_cows_shore := total_animals_shore - dogs_made_it_shore
def cows_made_it_shore := total_sheep_cows_shore - total_sheep_shore

-- Theorem stating the problem
theorem number_of_cows_on_boat : 
  (cows_made_it_shore + cows_drowned) = 10 := by
  sorry

end number_of_cows_on_boat_l700_700732


namespace average_books_per_student_l700_700570

theorem average_books_per_student
  (total_students : ℕ)
  (students_0_books : ℕ)
  (students_1_book : ℕ)
  (students_2_books : ℕ)
  (students_at_least_3_books : ℕ)
  (total_students_eq : total_students = 38)
  (students_0_books_eq : students_0_books = 2)
  (students_1_book_eq : students_1_book = 12)
  (students_2_books_eq : students_2_books = 10)
  (students_at_least_3_books_eq : students_at_least_3_books = 14)
  (students_count_consistent : total_students = students_0_books + students_1_book + students_2_books + students_at_least_3_books) :
  (students_0_books * 0 + students_1_book * 1 + students_2_books * 2 + students_at_least_3_books * 3 : ℝ) / total_students = 1.947 :=
by
  sorry

end average_books_per_student_l700_700570


namespace eq_has_one_integral_root_l700_700302

theorem eq_has_one_integral_root :
  ∀ x : ℝ, (x - (9 / (x - 5)) = 4 - (9 / (x-5))) → x = 4 := by
  intros x h
  sorry

end eq_has_one_integral_root_l700_700302


namespace find_inscription_l700_700640

-- Definitions for the conditions
def identical_inscriptions (box1 box2 : String) : Prop :=
  box1 = box2

def conclusion_same_master (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini"

def cannot_identify_master (box : String) : Prop :=
  ¬(∀ (made_by : String → Prop), made_by "Bellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Cellini")

def single_casket_indeterminate (box : String) : Prop :=
  (∀ (made_by : String → Prop), made_by "Bellini" ∨ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini" ∧ made_by "Cellini") ∧
  ¬(∀ (made_by : String → Prop), made_by "Bellini")

-- Inscription on the boxes
def inscription := "At least one of these boxes was made by Cellini's son."

-- The Lean statement for the proof
theorem find_inscription (box1 box2 : String)
  (h1 : identical_inscriptions box1 box2)
  (h2 : conclusion_same_master box1)
  (h3 : cannot_identify_master box1)
  (h4 : single_casket_indeterminate box1) :
  box1 = inscription :=
sorry

end find_inscription_l700_700640


namespace sum_of_integers_l700_700506

theorem sum_of_integers (n : ℤ) (h : 0 < 3 * n ∧ 3 * n < 27) : 
  ∑ i in Finset.filter (λ x, 0 < x ∧ x < 9) (Finset.range 10), i = 36 := by
  sorry

end sum_of_integers_l700_700506


namespace alpha_perp_beta_l700_700607

-- Definitions
variables {a b : Line} {α β : Plane}

-- Conditions
def perp (l : Line) (p : Plane) := sorry -- Definition for line perpendicular to plane
def perp_lines (l₁ l₂ : Line) := sorry -- Definition for line perpendicular to line
def parallel (l : Line) (p : Plane) := sorry -- Definition for line parallel to plane

-- Given conditions
axiom h1 : perp_lines a b
axiom h2 : perp a α
axiom h3 : perp_lines b β

-- Conclusion to prove
theorem alpha_perp_beta : perp α β :=
sorry

end alpha_perp_beta_l700_700607


namespace increasing_log_abs_function_l700_700131

theorem increasing_log_abs_function {a : ℝ} (h : ∀ x y : ℝ, 0 < x → 0 < y → x < y → |Math.log (x + a)| < |Math.log (y + a)|) : 1 ≤ a :=
sorry

end increasing_log_abs_function_l700_700131


namespace prod_term_simplified_l700_700435

-- Define the sequence of terms
def term (k : ℕ) : ℚ := 1 - 1 / (k + 1)

-- Define the product of the terms from k=2 to 150
def prod_terms : ℚ := (finset.range 149).prod (λ k, term (k + 2))

-- Define the question as a statement
theorem prod_term_simplified :
  2 * prod_terms = 149 / 75 :=
by sorry

end prod_term_simplified_l700_700435


namespace find_smallest_n_l700_700855

-- Definition of the problem condition
def ends_in_6 (n : ℕ) : Prop :=
  n % 10 = 6

def placing_6_at_front (n : ℕ) : ℕ :=
  6 * Nat.pow 10 ((Nat.length_digits 10 (n / 10)) - 1) + n / 10

def four_times (n m : ℕ) : Prop :=
  m = 4 * n

-- The formal statement of the proof problem
theorem find_smallest_n : ∃ n : ℕ, ends_in_6 n ∧ four_times n (placing_6_at_front n) ∧ ∀ m : ℕ, ends_in_6 m ∧ four_times m (placing_6_at_front m) → n ≤ m :=
  sorry

end find_smallest_n_l700_700855


namespace find_x_squared_plus_y_squared_l700_700553

open Real

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 := by
  sorry

end find_x_squared_plus_y_squared_l700_700553


namespace count_non_zero_tenths_digit_decimals_l700_700465

def is_powers_of_2_and_5 (n : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ n → (p = 2 ∨ p = 5)

def has_non_zero_tenths_digit (n : ℕ) : Prop :=
  (n ≤ 50) ∧ is_powers_of_2_and_5 n

theorem count_non_zero_tenths_digit_decimals :
  { n | has_non_zero_tenths_digit n }.to_finset.card = 12 :=
by sorry

end count_non_zero_tenths_digit_decimals_l700_700465


namespace sophie_marble_exchange_l700_700270

theorem sophie_marble_exchange (sophie_initial_marbles joe_initial_marbles : ℕ) 
  (final_ratio : ℕ) (sophie_gives_joe : ℕ) : 
  sophie_initial_marbles = 120 → joe_initial_marbles = 19 → final_ratio = 3 → 
  (120 - sophie_gives_joe = 3 * (19 + sophie_gives_joe)) → sophie_gives_joe = 16 := 
by
  intros h1 h2 h3 h4
  sorry

end sophie_marble_exchange_l700_700270


namespace max_value_of_expression_l700_700135

-- Defining the given condition as a predicate
def condition (x y : ℝ) : Prop := x^2 + y^2 = 20 * x + 24 * y + 26

-- Goal: finding the maximum value of 5x + 3y subject to the condition
theorem max_value_of_expression :
  ∃ x y : ℝ, condition x y ∧ (∀ a b : ℝ, condition a b → 5 * a + 3 * b ≤ 73) :=
begin
  sorry
end

end max_value_of_expression_l700_700135


namespace baseball_game_earnings_l700_700357

theorem baseball_game_earnings
  (S : ℝ) (W : ℝ)
  (h1 : S = 2662.50)
  (h2 : W + S = 5182.50) :
  S - W = 142.50 :=
by
  sorry

end baseball_game_earnings_l700_700357


namespace f_val_sum_zero_l700_700498

variables (a b : ℝ)
def f (x : ℝ) : ℝ := 2018 * x^3 - real.sin x + b + 2

-- Odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f(x)

-- Domain condition
def domain_condition (a : ℝ) : Prop :=
  (a-4) ≠ (2*a-2)

-- Known conditions
axiom h1 : is_odd_function (f b)
axiom h2 : domain_condition a

-- Required proof
theorem f_val_sum_zero : f a + f b = 0 :=
sorry

end f_val_sum_zero_l700_700498


namespace compare_negatives_l700_700408

noncomputable def isNegative (x : ℝ) : Prop := x < 0
noncomputable def absValue (x : ℝ) : ℝ := if x < 0 then -x else x
noncomputable def sqrt14 : ℝ := Real.sqrt 14

theorem compare_negatives : -4 < -Real.sqrt 14 := by
  have h1: Real.sqrt 16 = 4 := by
    sorry
  
  have h2: absValue (-4) = 4 := by
    sorry

  have h3: absValue (-(sqrt14)) = sqrt14 := by
    sorry

  have h4: Real.sqrt 16 > Real.sqrt 14 := by
    sorry

  show -4 < -Real.sqrt 14
  sorry

end compare_negatives_l700_700408


namespace log_min_value_range_l700_700978

theorem log_min_value_range (a : ℝ) (h1 : 1 < a) (h2 : a < 2) : 
  ∃ x : ℝ, is_minimum (λ x, real.log a (x^2 - a * x + 1)) x := 
sorry

end log_min_value_range_l700_700978


namespace trig_identity_proof_l700_700349

variable (α : ℝ)

theorem trig_identity_proof : 
  16 * (Real.sin α)^5 - 20 * (Real.sin α)^3 + 5 * Real.sin α = Real.sin (5 * α) :=
  sorry

end trig_identity_proof_l700_700349


namespace cotangent_sum_identity_l700_700519

theorem cotangent_sum_identity (a b : ℝ) (h : a ≠ 0) (h1 : a > 2 * |b|) :
  (Real.cot (11 * Real.pi / 4 + 1 / 2 * Real.arccos (2 * b / a)) +
   Real.cot (11 * Real.pi / 4 - 1 / 2 * Real.arccos (2 * b / a))) = - a / b :=
by
  sorry

end cotangent_sum_identity_l700_700519


namespace expected_number_of_heads_after_flips_l700_700599

theorem expected_number_of_heads_after_flips :
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  expected_heads = 6500 / 81 :=
by
  let p_heads_after_tosses : ℚ := (1/3) + (2/9) + (4/27) + (8/81)
  let expected_heads : ℚ := 100 * p_heads_after_tosses
  show expected_heads = (6500 / 81)
  sorry

end expected_number_of_heads_after_flips_l700_700599


namespace rhombus_perimeter_l700_700291

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side_length := real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side_length = 16 * real.sqrt 13 :=
by
  let d1_half := d1 / 2
  let d2_half := d2 / 2
  have h3 : d1_half = 12 := by sorry
  have h4 : d2_half = 8 := by sorry
  let side_length := real.sqrt (d1_half ^ 2 + d2_half ^ 2)
  have h5 : side_length = real.sqrt 208 := by sorry
  have h6 : real.sqrt 208 = 4 * real.sqrt 13 := by sorry
  show 4 * side_length = 16 * real.sqrt 13
  from by
    rw [h6]
    rfl

end rhombus_perimeter_l700_700291


namespace expression_equals_3_l700_700010

noncomputable def expression : ℝ :=
  (real.sqrt 3 - 2) * real.sqrt 3 + real.sqrt 12

theorem expression_equals_3 : expression = 3 :=
  sorry

end expression_equals_3_l700_700010


namespace sin_2016_is_negative_l700_700317

-- Conditions definition
def sin_periodic (x : ℝ) : Prop := sin (x + 360) = sin x
def sin_third_quadrant (x : ℝ) (h : 180 < x ∧ x < 270) : sin x < 0
def sin_second_quadrant (x : ℝ) (h : 90 < x ∧ x < 180) : sin x > 0

-- Main statement
theorem sin_2016_is_negative : sin 2016 < 0 :=
by
  -- Use conditions as needed in the proof
  sorry

end sin_2016_is_negative_l700_700317


namespace original_area_of_doubled_rectangle_l700_700293

theorem original_area_of_doubled_rectangle (A_new : ℝ) (h : A_new = 32) :
  ∃ A : ℝ, A * 4 = A_new ∧ A = 8 :=
by {
  use 8,
  split,
  { norm_num, exact h.symm },
  { rfl }
}

end original_area_of_doubled_rectangle_l700_700293


namespace CamilleFarthestDistance_l700_700404

noncomputable def CamilleDodecahedronProblem : ℝ :=
  let pentagon_perimeter := 5
  -- Additional intermediate variables and geometric relations would go here.
  -- ...
  let L := sqrt ((17 + 7 * sqrt 5) / 2)
  L^2

theorem CamilleFarthestDistance :
  CamilleDodecahedronProblem = (17 + 7 * sqrt 5) / 2 := 
by
  -- Detailed proof steps would go here.
  sorry

end CamilleFarthestDistance_l700_700404


namespace floor_equation_solution_l700_700043

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l700_700043


namespace seq_sum_2008_eq_1339_l700_700489

noncomputable def sequence_sum_2008_terms : ℕ :=
  let seq : ℕ → ℝ
  | 0     => 1
  | 1     => a
  | (n+2) => abs (seq (n+1) - seq n)
  (List.sum (List.map seq (List.range 2008)))

theorem seq_sum_2008_eq_1339 (a : ℝ) (h : a ≥ 0) : sequence_sum_2008_terms a = 1339 := by
  sorry

end seq_sum_2008_eq_1339_l700_700489


namespace votes_for_candidate_a_l700_700186

theorem votes_for_candidate_a :
  let total_votes : ℝ := 560000
  let percentage_invalid : ℝ := 0.15
  let percentage_candidate_a : ℝ := 0.85
  let valid_votes := (1 - percentage_invalid) * total_votes
  let votes_candidate_a := percentage_candidate_a * valid_votes
  votes_candidate_a = 404600 :=
by
  sorry

end votes_for_candidate_a_l700_700186


namespace problem_statement_l700_700138

noncomputable def f : ℝ → ℝ := sorry

theorem problem_statement (h1 : ∀ x : ℝ, f (x + 2016) = f (-x + 2016))
    (h2 : ∀ x1 x2 : ℝ, 2016 ≤ x1 ∧ 2016 ≤ x2 ∧ x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0) :
    f 2019 < f 2014 ∧ f 2014 < f 2017 :=
sorry

end problem_statement_l700_700138


namespace Xiaoming_matches_l700_700337

theorem Xiaoming_matches (P : ℕ → ℚ) (θ : ℚ) (N : ℕ) (W : ℕ) (X : ℕ) :
  N = 20 → P N = 0.95 → W = 95 → P (N + θ) = 0.96 → 
  (θ = 5) := by
  sorry

end Xiaoming_matches_l700_700337


namespace men_with_all_attributes_le_l700_700722

theorem men_with_all_attributes_le (total men_with_tv men_with_radio men_with_ac: ℕ) (married_men: ℕ) 
(h_total: total = 100) 
(h_married_men: married_men = 84) 
(h_men_with_tv: men_with_tv = 75) 
(h_men_with_radio: men_with_radio = 85) 
(h_men_with_ac: men_with_ac = 70) : 
  ∃ x, x ≤ men_with_ac ∧ x ≤ married_men ∧ x ≤ men_with_tv ∧ x ≤ men_with_radio ∧ (x ≤ total) := 
sorry

end men_with_all_attributes_le_l700_700722


namespace scientific_notation_of_distance_l700_700276

theorem scientific_notation_of_distance :
  ∃ (n : ℝ), n = 384000 ∧ 384000 = n * 10^5 :=
sorry

end scientific_notation_of_distance_l700_700276


namespace tangency_condition_common_points_condition_l700_700146

-- Definition of the circle C and the line l
def circle_C : (ℝ × ℝ) → Prop := λ p, (p.1)^2 + (p.2)^2 - 8 * p.1 + 15 = 0
def line_l (k : ℝ) : (ℝ × ℝ) → Prop := λ p, p.2 = k * p.1 - 2

-- 1. Tangency Condition
theorem tangency_condition (k : ℝ) : (∃ p : ℝ × ℝ, circle_C p ∧ line_l k p) ∧ 
  (∀ p1 p2 : ℝ × ℝ, circle_C p1 ∧ line_l k p1 → circle_C p2 → line_l k p2 → p1 = p2) → 
  (k = (8 + real.sqrt 19) / 15 ∨ k = (8 - real.sqrt 19) / 15) :=
sorry

-- 2. Common Points Condition
theorem common_points_condition (k : ℝ) : (∃ p : ℝ × ℝ, line_l k p ∧ 
  (∃ q : ℝ × ℝ, circle_C q ∧ ((q.1 - p.1)^2 + (q.2 - p.2)^2 = 1))) → 
  (0 ≤ k ∧ k ≤ 4 / 3) :=
sorry

end tangency_condition_common_points_condition_l700_700146


namespace min_sum_of_dimensions_l700_700301

/-- A theorem to find the minimum possible sum of the three dimensions of a rectangular box 
with given volume 1729 inch³ and positive integer dimensions. -/
theorem min_sum_of_dimensions (x y z : ℕ) (h1 : x * y * z = 1729) : x + y + z ≥ 39 :=
by
  sorry

end min_sum_of_dimensions_l700_700301


namespace perception_num_permutations_l700_700818

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l700_700818


namespace exists_alpha_l700_700103

noncomputable def f_alpha (α : ℝ) (x : ℝ) := ⌊ α * x ⌋

theorem exists_alpha (n : ℕ) (h : n > 0) :
  ∃ α : ℝ, (α = real.exp (-1 / n^2)) ∧ (∀ k : ℕ, 1 ≤ k → k ≤ n → (k.fold (f_alpha α) (n * n / k) = (n * n - k) / k)) :=
sorry

end exists_alpha_l700_700103


namespace factory_output_decrease_l700_700670

theorem factory_output_decrease (O : ℝ) (h1 : 0 < O) :
  let increased_output1 := 1.20 * O
  let increased_output2 := 1.50 * increased_output1
  let increased_output3 := 1.25 * increased_output2
  ∃ decrease_percent : ℝ, decreased_output = decreased_output ->
  decrease_percent ≈ 55.56 :=
by {
  sorry
}

end factory_output_decrease_l700_700670


namespace find_a_b_extreme_points_l700_700150

noncomputable def f (a b x : ℝ) : ℝ := x^3 - 3 * a * x + b

theorem find_a_b (a b : ℝ) (h₁ : a ≠ 0) (h₂ : deriv (f a b) 2 = 0) (h₃ : f a b 2 = 8) : 
  a = 4 ∧ b = 24 :=
by
  sorry

noncomputable def f_deriv (a x : ℝ) : ℝ := 3 * x^2 - 3 * a

theorem extreme_points (a : ℝ) (h₁ : a > 0) : 
  (∃ x: ℝ, f_deriv a x = 0 ∧ 
      ((x = -Real.sqrt a ∧ f a 24 x = 40) ∨ 
       (x = Real.sqrt a ∧ f a 24 x = 16))) := 
by
  sorry

end find_a_b_extreme_points_l700_700150


namespace trigonometric_expression_evaluation_l700_700836

theorem trigonometric_expression_evaluation :
  (Real.cos (-585 * Real.pi / 180)) / 
  (Real.tan (495 * Real.pi / 180) + Real.sin (-690 * Real.pi / 180)) = Real.sqrt 2 :=
  sorry

end trigonometric_expression_evaluation_l700_700836


namespace find_x_from_perpendicular_vectors_l700_700113

noncomputable def perpendicular_vectors_example : Prop :=
  ∃ x y : ℝ, let a := (3 , x)
                ⬝ let b := (y, 1)
                ⬝ a.1 * b.1 + a.2 * b.2 = 0 ∧ 
                  x = -7 / 4

theorem find_x_from_perpendicular_vectors : perpendicular_vectors_example :=
sorry

end find_x_from_perpendicular_vectors_l700_700113


namespace veranda_area_l700_700379

theorem veranda_area (room_length room_width veranda_length_width veranda_width_width : ℝ)
  (h1 : room_length = 19)
  (h2 : room_width = 12)
  (h3 : veranda_length_width = 2.5)
  (h4 : veranda_width_width = 3)
  : (room_length + 2 * veranda_length_width) * (room_width + 2 * veranda_width_width) - room_length * room_width = 204 :=
by
  simp [h1, h2, h3, h4]
  norm_num
  done

end veranda_area_l700_700379


namespace decimal_to_octal_521_l700_700021

theorem decimal_to_octal_521 : 
  let decimal : ℕ := 521 in
  let octal : ℕ := 1011 in 
  ∀ (quotient remainder : ℕ), 
    (quotient = 521 / 8 ∧ remainder = 521 % 8 ∧ remainder = 1) →
    (quotient = 65 ∧ 65 / 8 = 8 ∧ 65 % 8 = 1 ∧ 8 / 8 = 1 ∧ 8 % 8 = 0) →
    nat.octal_repr decimal = octal :=
begin
  assume decimal octal,
  assume h1 h2,
  rw [nat.octal_repr_eq],
  sorry
end

end decimal_to_octal_521_l700_700021


namespace largest_sphere_within_prism_l700_700495

variables (M A B C D : ℝ)
variables (a x r : ℝ)

-- Conditions
axiom square_base : ∃ b : ℝ, A = ⟨0, 0⟩ ∧ B = ⟨b, 0⟩ ∧ C = ⟨b, b⟩ ∧ D = ⟨0, b⟩
axiom MA_MD : MA = MD
axiom MA_perp_AB : MA ⊥ AB
axiom area_AMD : 1 = 1/2 * a * x

-- Given
noncomputable def largest_sphere_radius : ℝ :=
(r = sqrt 2 - 1)

theorem largest_sphere_within_prism :
  square_base ∧ MA_MD ∧ MA_perp_AB ∧ (1 = 1/2 * a * x) →
  largest_sphere_radius a x
:= by
  intros,
  sorry

end largest_sphere_within_prism_l700_700495


namespace proof_problem_l700_700950

-- Definition of the propositions
def proposition_p := ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ (Real.log10 (a + b) = Real.log10 a + Real.log10 b)
def proposition_q := ∀ (l1 l2 : List (ℝ × ℝ × ℝ)), (¬(∃ p : ℝ × ℝ × ℝ, p ∈ l1 ∧ p ∈ l2) ∧ ¬(∀ (p q : ℝ × ℝ × ℝ), p ∈ l1 ∧ q ∈ l1 → (p, q) ∈ l2)) ↔ ¬(¬∃ (p : ℝ × ℝ × ℝ) (plane : ℝ × ℝ × ℝ), p ∈ l1 ∧ p ∈ l2 ∧ plane ∈ l1 ∧ plane ∈ l2)

-- Statement we want to prove
theorem proof_problem :
  (¬ proposition_p ∧ proposition_q) :=
by {
  sorry
}

end proof_problem_l700_700950


namespace inequality_proof_l700_700170

theorem inequality_proof (a b c : ℝ) (h₁ : 1 < b) (h₂ : b < a) (h₃ : 0 < c) (h₄ : c < 1) : a * log b c < b * log a c :=
  sorry

end inequality_proof_l700_700170


namespace total_amount_shared_l700_700802

theorem total_amount_shared (total_amount : ℝ) 
  (h_debby : total_amount * 0.25 = (total_amount - 4500))
  (h_maggie : total_amount * 0.75 = 4500) : total_amount = 6000 :=
begin
  sorry
end

end total_amount_shared_l700_700802


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700053

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700053


namespace lisa_score_is_85_l700_700207

def score_formula (c w : ℕ) : ℕ := 30 + 4 * c - w

theorem lisa_score_is_85 (c w : ℕ) 
  (score_equality : 85 = score_formula c w)
  (non_neg_w : w ≥ 0)
  (total_questions : c + w ≤ 30) :
  (c = 14 ∧ w = 1) :=
by
  sorry

end lisa_score_is_85_l700_700207


namespace floor_eq_l700_700063

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l700_700063


namespace translate_parabola_up_one_unit_l700_700778

theorem translate_parabola_up_one_unit (x : ℝ) :
  let y := 3 * x^2
  (y + 1) = 3 * x^2 + 1 :=
by
  -- Proof omitted
  sorry

end translate_parabola_up_one_unit_l700_700778


namespace chord_seen_angle_l700_700724

-- Definitions and Conditions
variables {O A B K L M : Type}

-- Defining the conditions of the problem
def is_circle_center (O : Point) (circ : Circle) : Prop :=
circle.center circ = O

def lies_on_circle (P : Point) (circ : Circle) : Prop :=
circle.contains circ P

def is_equilateral_triangle (T : Triangle) : Prop :=
T.angle_at T.A = 60 ∧ T.angle_at T.B = 60 ∧ T.angle_at T.C = 60

def divides_into_equal_parts (A B K L M : Point) : Prop :=
A ≠ B ∧ K ≠ L ∧ 
(t_segment A K).length = (t_segment K L).length ∧ 
(t_segment K L).length = (t_segment L B).length ∧ 
(l_segment A B).contains_segment (t_segment A K) ∧ 
(l_segment A B).contains_segment (t_segment K L) ∧ 
(l_segment A B).contains_segment (t_segment L B)

def subtended_angle (O A B : Point) : Angle :=
angle A O B

-- Theorem Definition
theorem chord_seen_angle (circ : Circle) 
    (vertex_on : lies_on_circle M circ) 
    (divides : divides_into_equal_parts A B K L M) 
    (equilateral : is_equilateral_triangle (triangle K L M)) : 
    subtended_angle O A B = 120 :=
sorry

end chord_seen_angle_l700_700724


namespace circular_garden_radius_l700_700364

def radius_of_circular_garden (r : ℝ) : Prop :=
  let C := 2 * Real.pi * r in
  let A := Real.pi * r^2 in
  C = (1 / 5) * A → r = 10

theorem circular_garden_radius (r : ℝ) (h : radius_of_circular_garden r) : r = 10 :=
by 
  sorry

end circular_garden_radius_l700_700364


namespace consecutive_integer_min_values_l700_700477

theorem consecutive_integer_min_values (a b : ℝ) 
  (consec : b = a + 1) 
  (min_a : a ≤ real.sqrt 30) 
  (min_b : b ≥ real.sqrt 30) : 
  2 * a - b = 4 := 
sorry

end consecutive_integer_min_values_l700_700477


namespace nathalie_cake_fraction_l700_700735

theorem nathalie_cake_fraction
    (cake_weight : ℕ)
    (pierre_ate : ℕ)
    (double_what_nathalie_ate : pierre_ate = 2 * (pierre_ate / 2))
    (pierre_ate_correct : pierre_ate = 100) :
    (pierre_ate / 2) / cake_weight = 1 / 8 :=
by
  sorry

end nathalie_cake_fraction_l700_700735


namespace john_text_messages_l700_700206

/-- John decides to get a new phone number and it ends up being a recycled number. 
    He used to get some text messages a day. 
    Now he is getting 55 text messages a day, 
    and he is getting 245 text messages per week that are not intended for him. 
    How many text messages a day did he used to get?
-/
theorem john_text_messages (m : ℕ) (h1 : 55 = m + 35) (h2 : 245 = 7 * 35) : m = 20 := 
by 
  sorry

end john_text_messages_l700_700206


namespace count_permutations_perception_l700_700814

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l700_700814


namespace handshaking_arrangements_l700_700571
noncomputable theory

def num_people : ℕ := 8
def handshakes_per_person : ℕ := 3
def modulo_value : ℕ := 1000

theorem handshaking_arrangements : 
  ∃ M : ℕ, (number_of_handshaking_arrangements num_people handshakes_per_person = M) ∧ 
  (M % modulo_value = 70) := 
by
  sorry

end handshaking_arrangements_l700_700571


namespace quadratic_no_real_roots_l700_700678

theorem quadratic_no_real_roots : ∀ x : ℝ, ¬(x ^ 2 - 3 * x + 3 = 0) :=
by
  assume x
  have Δ := (-3 : ℝ) ^ 2 - 4 * 1 * 3
  have h : Δ < 0 := by norm_num
  intro h_eq
  have h_nonneg : Δ ≥ 0 := by apply polynomial.discr_nonneg_of_root_real h_eq
  linarith

end quadratic_no_real_roots_l700_700678


namespace chef_pies_total_l700_700013

def chefPieSales : ℕ :=
  let small_shepherd_pies := 52 / 4
  let large_shepherd_pies := 76 / 8
  let small_chicken_pies := 80 / 5
  let large_chicken_pies := 130 / 10
  let small_vegetable_pies := 42 / 6
  let large_vegetable_pies := 96 / 12
  let small_beef_pies := 35 / 7
  let large_beef_pies := 105 / 14

  small_shepherd_pies + large_shepherd_pies + small_chicken_pies + large_chicken_pies +
  small_vegetable_pies + large_vegetable_pies +
  small_beef_pies + large_beef_pies

theorem chef_pies_total : chefPieSales = 80 := by
  unfold chefPieSales
  have h1 : 52 / 4 = 13 := by norm_num
  have h2 : 76 / 8 = 9 ∨ 76 / 8 = 10 := by norm_num -- rounding consideration
  have h3 : 80 / 5 = 16 := by norm_num
  have h4 : 130 / 10 = 13 := by norm_num
  have h5 : 42 / 6 = 7 := by norm_num
  have h6 : 96 / 12 = 8 := by norm_num
  have h7 : 35 / 7 = 5 := by norm_num
  have h8 : 105 / 14 = 7 ∨ 105 / 14 = 8 := by norm_num -- rounding consideration
  sorry

end chef_pies_total_l700_700013


namespace quadratic_solutions_l700_700981

theorem quadratic_solutions (a : ℝ) (h : a ≤ 0) : 
  ∃ n, (n = 1 ∨ n = 2) ∧ (ax^2 + 2*x + 1 = 0).roots.length = n :=
by
  sorry

end quadratic_solutions_l700_700981


namespace find_n_l700_700168

theorem find_n (n : ℕ) (h : 12^(4 * n) = (1/12)^(n - 30)) : n = 6 := 
by {
  sorry 
}

end find_n_l700_700168


namespace shared_total_l700_700801

theorem shared_total (total_amount : ℝ) (maggie_share : ℝ) (debby_percentage : ℝ)
  (h1 : debby_percentage = 0.25)
  (h2 : maggie_share = 4500)
  (h3 : maggie_share = (1 - debby_percentage) * total_amount) :
  total_amount = 6000 :=
by
  sorry

end shared_total_l700_700801


namespace ranking_is_fiona_daniel_emily_l700_700422

namespace FriendsOrdering

variables (Daniel Emily Fiona : ℕ) -- assume ages are represented as natural numbers
variable (different_ages : Daniel ≠ Emily ∧ Emily ≠ Fiona ∧ Daniel ≠ Fiona)
variable (exactly_one_true : (Emily > Daniel ∧ Emily > Fiona ∧ (Fiona > Daniel ∨ Fiona < Daniel))
                           ∨ (Fiona < Daniel ∧ Fiona < Emily ∧ (Emily > Daniel ∨ Emily < Daniel))
                           ∨ (Daniel > Fiona ∧ Daniel > Emily ∧ (Fiona < Emily ∨ Fiona < Emily)))

noncomputable def rank_friends_oldest_to_youngest : Prop :=
  (Fiona > Daniel ∧ Daniel > Emily)

theorem ranking_is_fiona_daniel_emily :
  different_ages →
  exactly_one_true →
  rank_friends_oldest_to_youngest Daniel Emily Fiona :=
by
  intros diff_ages ex_one_true
  -- Proof omitted
  sorry

end FriendsOrdering

end ranking_is_fiona_daniel_emily_l700_700422


namespace obtuse_triangle_angle_acute_l700_700616

theorem obtuse_triangle_angle_acute (A B C P : Point)
  (hABC : Triangle A B C)
  (hPA_longest : PA^2 > PB^2 + PC^2) :
  Angle BAC < Pi / 2 :=
sorry

end obtuse_triangle_angle_acute_l700_700616


namespace quadratic_roots_abs_difference_l700_700664

theorem quadratic_roots_abs_difference (m r s : ℝ) (h_eq: r^2 - (m+1) * r + m = 0) (k_eq: s^2 - (m+1) * s + m = 0) :
  |r + s - 2 * r * s| = |1 - m| := by
  sorry

end quadratic_roots_abs_difference_l700_700664


namespace matrix_product_l700_700946

def A : Matrix (Fin 3) (Fin 1) ℝ := ![![1], ![-1], ![0]]
def B : Vector3 ℝ := (1, 2, 1)

theorem matrix_product (A_transpose : (Fin 1) → (Fin 3) → ℝ) :
  A_transpose = (Matrix.transpose A) →
  (Matrix.mulVec A_transpose B) = -1 := 
by
  intros h
  rw [h]
  sorry

end matrix_product_l700_700946


namespace delta_max_success_ratio_l700_700999

theorem delta_max_success_ratio :
  ∃ (x y z w : ℕ),
  (0 < x ∧ x < (7 * y) / 12) ∧
  (0 < z ∧ z < (5 * w) / 8) ∧
  (y + w = 600) ∧
  (35 * x + 28 * z < 4200) ∧
  (x + z = 150) ∧ 
  (x + z) / 600 = 1 / 4 :=
by sorry

end delta_max_success_ratio_l700_700999


namespace forgot_to_mow_l700_700777

-- Definitions
def earning_per_lawn : ℕ := 9
def lawns_to_mow : ℕ := 12
def actual_earning : ℕ := 36

-- Statement to prove
theorem forgot_to_mow : (lawns_to_mow - (actual_earning / earning_per_lawn)) = 8 := by
  sorry

end forgot_to_mow_l700_700777


namespace cos_sin_exp_l700_700469

theorem cos_sin_exp (n : ℕ) (t : ℝ) (h : n ≤ 1000) :
  (Complex.exp (t * Complex.I)) ^ n = Complex.exp (n * t * Complex.I) :=
by
  sorry

end cos_sin_exp_l700_700469


namespace solve_log_equation_l700_700857

-- Define the original equation condition
def condition (x : ℝ) : Prop := log 2 (3^x - 5) = 2

-- State the theorem to prove x = 2 given the condition
theorem solve_log_equation : ∀ x : ℝ, condition x → x = 2 := 
by
  intro x
  intro cond
  sorry

end solve_log_equation_l700_700857


namespace find_f_prime_2_l700_700133

noncomputable def f (x : ℝ) : ℝ := x^2 * f' 1 - 3 * x

theorem find_f_prime_2 (f' : ℝ → ℝ) (h : ∀ x : ℝ, f x = x^2 * f' (1 : ℝ) - 3 * x) : f' 2 = 9 := by
  sorry

end find_f_prime_2_l700_700133


namespace find_a1_plus_b1_l700_700758

noncomputable def series (n : ℕ) : ℝ × ℝ :=
nat.rec_on n (a₁, b₁)
(λ n rec, (2 * real.sqrt 3 * rec.1 + rec.2, real.sqrt 3 * rec.2 - 2 * rec.1))

theorem find_a1_plus_b1 (a₁ b₁ : ℝ) 
(h1 : series 150 = (-1, 3)) : 
  a₁ + b₁ = - (1 / 4^149) := 
by sorry

#eval find_a1_plus_b1 (-1 / 4^149) -- Example to evaluate the theorem

end find_a1_plus_b1_l700_700758


namespace bowling_tournament_prize_orders_l700_700187
-- Import necessary Lean library

-- Define the conditions
def match_outcome (num_games : ℕ) : ℕ := 2 ^ num_games

-- Theorem statement
theorem bowling_tournament_prize_orders : match_outcome 5 = 32 := by
  -- This is the statement, proof is not required
  sorry

end bowling_tournament_prize_orders_l700_700187


namespace pond_volume_l700_700716

theorem pond_volume {L W H : ℝ} (hL : L = 20) (hW : W = 12) (hH : H = 5) : L * W * H = 1200 := by
  sorry

end pond_volume_l700_700716


namespace max_sum_arcs_l700_700652

/-- 
  Given 60 points on a circle: 
    - 30 points are red (R),
    - 20 points are blue (B),
    - 10 points are green (G),
  where each arc between adjacent points is assigned a number based on the colors:
    - 1 for an arc between red and green,
    - 2 for an arc between red and blue,
    - 3 for an arc between blue and green,
    - 0 for an arc between two points of the same color,
  the greatest possible sum of all the numbers assigned to the 60 arcs is 180.
-/
theorem max_sum_arcs : ∀ (R B G : ℕ), R = 30 → B = 20 → G = 10 → ∃ x y z r b g, 
  x + y + 2r = 60 ∧ y + z + 2b = 40 ∧ z + x + 2g = 20 ∧ 
  x + y + z + r + b + g = 60 ∧ 
  x + 2*y + 3*z = 180 := 
by sorry

end max_sum_arcs_l700_700652


namespace overtaking_time_l700_700710

theorem overtaking_time :
  ∀ t t_k : ℕ,
  (30 * t = 40 * (t - 5)) ∧ 
  (30 * t = 60 * t_k) →
  t = 20 ∧ t_k = 10 ∧ (20 - 10 = 10) :=
by
  sorry

end overtaking_time_l700_700710


namespace triangle_area_rational_of_rational_sides_and_bisectors_l700_700306

theorem triangle_area_rational_of_rational_sides_and_bisectors
  (a b c fa fb fc : ℚ)
  (habc : a + b > c)
  (habil : fa = (2 * c * (rational_cos_alpha_half alpha))/2)
  (hbia : fb = (2 * a * (rational_cos_beta_half beta))/2)
  (hcia : fc = (2 * b * (rational_cos_gamma_half gamma))/2)
  : ∃ (A : ℚ), A * A = 1/16 * (a + (b + c)/cos(alpha_half)) *
  (((1/2 * b * cos_alpha_half) + (2 * c * cos_beta_half)) /
  ((1/2 * (a + (-3)*b)*cos_alpha_half) +( 2* c^2 * cos_beta)))
  :
  sorry

end triangle_area_rational_of_rational_sides_and_bisectors_l700_700306


namespace average_remaining_checks_l700_700386

theorem average_remaining_checks (x y : ℕ) : 
  x + y = 30 →
  50 * x + 100 * y = 1800 →
  18 * 50 + 6 * 100 = 1500 →
  (18 * 50 + 6 * 100) / (18 + 6) = 62.50 :=
begin
  intros h1 h2 h3,
  sorry,
end

end average_remaining_checks_l700_700386


namespace num_gnomes_multiple_of_4_l700_700536

/-- A structure to hold information about a gnome and their voting behavior --/
structure Gnome :=
  (neighbors_same_vote : Bool)  -- True if both neighbors vote the same way
  (this_vote : ℤ)  -- 1 for "for", -1 for "against", 0 for "abstain"
  (next_vote : ℤ)  -- 1 for "for", -1 for "against", 0 for "abstain"

/-- Given the voting rules and initial conditions, show that the number of gnomes must be 
    a multiple of 4 --/
theorem num_gnomes_multiple_of_4 {n : ℕ} (gnomes : Fin n → Gnome) 
  (gold_glitter_voting : ∀ i, gnomes i .this_vote = 1)          -- All gnomes voted "for" for gold
  (thorin_abstain : Gnomes 0 .this_vote = 0)                      -- Thorin abstained for the Dragon
  (voting_rule1 : ∀ i, if (gnomes i).neighbors_same_vote then (gnomes (i + 1) % n).next_vote = (gnomes i).this_vote else true)      
  (voting_rule2 : ∀ i, if ¬(gnomes i).neighbors_same_vote then (gnomes (i + 1) % n).next_vote ≠ (gnomes i).this_vote) 
  : n % 4 = 0 :=
sorry

end num_gnomes_multiple_of_4_l700_700536


namespace num_terms_in_expansion_l700_700028

theorem num_terms_in_expansion (a b : ℕ) : 
  ∃ n, n = 10 ∧ ∀ x, x ∈ terms (([(2 * a + 5 * b)^3 * (2 * a - 5 * b)^3]^3)ᵖ) ↔ ∃ k, k ∈ finset.range 10 := 
sorry

end num_terms_in_expansion_l700_700028


namespace Carlos_wins_if_both_play_optimally_l700_700012

theorem Carlos_wins_if_both_play_optimally :
  ∀ (bricks : ℕ), bricks = 10 →
    (∀ k, 1 ≤ k ∧ k ≤ 3 →
      (Carlos_has_strategy : ∃ move : ℕ, 1 ≤ move ∧ move ≤ 3 ∧ (bricks - move) % 4 ≠ 0) →
      Carlos_wins)
sorry

end Carlos_wins_if_both_play_optimally_l700_700012


namespace sum_possible_values_x2_sum_of_x2_values_l700_700744

theorem sum_possible_values_x2 (p q x1 x2 : ℤ) (h1 : x = λ x : ℤ, x^2 - p*x + q) 
  (h2 : x1 + x2 = p) (h3 : x1 * x2 = q) (h4 : x1 = x2 + d) (h5 : q = x2) 
  (h6 : x1 > x2 ∧ x2 > q) : (x2 = -3 ∨ x2 = -2) → x2 ∈ {-3, -2} :=
by
  sorry

theorem sum_of_x2_values : -3 + (-2) = -5 :=
by
  sorry

end sum_possible_values_x2_sum_of_x2_values_l700_700744


namespace find_a4_l700_700532

-- Define the sequence based on the given conditions
def a : ℕ → ℚ
| 0     := 1  -- using 0-based indexing for convenience, a_1 corresponds to a(0)
| (n+1) := a n / (2 * a n + 3)

-- Theorem statement to be proved
theorem find_a4 : a 3 = 1 / 53 := 
sorry  -- proof to be filled in

end find_a4_l700_700532


namespace max_distance_compressed_circle_to_rotated_line_l700_700993

theorem max_distance_compressed_circle_to_rotated_line:
  (∀ (x y : ℝ), x^2 + y^2 = 4 →
    let C := ∀ (x y : ℝ), (x^2 / 4) + y^2 = 1 in
    ∀ (x y : ℝ), 3 * x - 2 * y - 8 = 0 →
      let l := ∀ (x y : ℝ), 2 * x + 3 * y = 8 in 
      ∃ d : ℝ, d = sqrt 13) :=
begin
  sorry
end

end max_distance_compressed_circle_to_rotated_line_l700_700993


namespace find_n_l700_700416

theorem find_n :
  ∑ i in finset.range(514), (i + 2) * 3^(i + 2) = 3^(515 + 8) :=
by
  sorry

end find_n_l700_700416


namespace similar_quadrilateral_if_angles_equal_l700_700645

-- Definitions for quadrilaterals and similarity
structure Quadrilateral (P : Type) :=
  (A B C D : P)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (angle_C : ℝ)
  (angle_D : ℝ)
  (diagonal_angle : ℝ)

-- Properties ensuring the angles and corresponding angles between diagonals
axiom equalAngles {P : Type} 
  (q1 q2 : Quadrilateral P) :
  q1.angle_A = q2.angle_A ∧
  q1.angle_B = q2.angle_B ∧
  q1.angle_C = q2.angle_C ∧
  q1.angle_D = q2.angle_D ∧
  q1.diagonal_angle = q2.diagonal_angle → q1 = q2

-- Prove that two quadrilaterals are similar given the angle conditions
theorem similar_quadrilateral_if_angles_equal {P : Type} 
  (q1 q2 : Quadrilateral P) :
  (q1.angle_A = q2.angle_A) ∧
  (q1.angle_B = q2.angle_B) ∧
  (q1.angle_C = q2.angle_C) ∧
  (q1.angle_D = q2.angle_D) ∧
  (q1.diagonal_angle = q2.diagonal_angle) →
  q1 = q2 :=
begin
  intros,
  sorry
end

end similar_quadrilateral_if_angles_equal_l700_700645


namespace diagonals_equal_l700_700587

variable {A B C D E : Type} [Inhabited A] [Inhabited B] [Inhabited C] [Inhabited D] [Inhabited E]

-- Assume we are working in Euclidean space
variables {P Q R S : Point}
variables (quadrilateral : Quadrilateral P Q R S)
variable (E : Point)
variable (angle_PA_PB : angle P A = angle D A)
variable (perpendicular_bisectors_intersect : ∃ E : Point, is_perpendicular_bisector P Q E ∧ is_perpendicular_bisector R S E)

theorem diagonals_equal (h₁ : ∀ E : Point, ∃ hP: E ∈ bisector_of P Q, ∃ hQ: E ∈ bisector_of R S, E ⊂ P A) 
     (h₂ : angle P A = angle D A)
     (h₃ : is_perpendicular_bisector P Q ∧ is_perpendicular_bisector R S) : 
     distance (P A R) = distance (Q D B) := 
sorry

end diagonals_equal_l700_700587


namespace cube_lateral_surface_area_l700_700667

theorem cube_lateral_surface_area (V : ℝ) (h_V : V = 125) : 
  ∃ A : ℝ, A = 100 :=
by
  sorry

end cube_lateral_surface_area_l700_700667


namespace perception_num_permutations_l700_700817

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l700_700817


namespace dad_strawberry_weight_l700_700628

theorem dad_strawberry_weight :
  ∀ (T L M D : ℕ), T = 36 → L = 8 → M = 12 → (D = T - L - M) → D = 16 :=
by
  intros T L M D hT hL hM hD
  rw [hT, hL, hM] at hD
  exact hD

end dad_strawberry_weight_l700_700628


namespace log_geom_seq_sum_bounded_l700_700121

-- Define the sequences and conditions
variable (a : ℕ → ℝ)
axiom a_pos : ∀ n, a n > 0
axiom a1 : a 1 = 1
axiom a_rec : ∀ n, a n = (a (n+1))^2 + 2 * a (n+1)

-- Problem I: Prove that {log2(a_n + 1)} is a geometric sequence
theorem log_geom_seq : ∀ n, Real.log2 (a (n+1) + 1) = (1/2) * Real.log2 (a n + 1) :=
by sorry

-- Define the sequence b and sum S_n
noncomputable def b (n : ℕ) : ℝ := n * Real.log2 (a n + 1)
noncomputable def S (n : ℕ) : ℝ := ∑ k in Finset.range n, b k

-- Problem II: Prove 1 ≤ S_n < 4 where S_n is the sum of the first n terms of sequence {b_n}
theorem sum_bounded : ∀ n, 1 ≤ S n ∧ S n < 4 :=
by sorry

end log_geom_seq_sum_bounded_l700_700121


namespace cucumber_new_weight_l700_700713

-- Definitions for the problem conditions
def initial_weight : ℝ := 100
def initial_water_percentage : ℝ := 0.99
def final_water_percentage : ℝ := 0.96
noncomputable def new_weight : ℝ := initial_weight * (1 - initial_water_percentage) / (1 - final_water_percentage)

-- The theorem stating the problem to be solved
theorem cucumber_new_weight : new_weight = 25 :=
by
  -- Skipping the proof for now
  sorry

end cucumber_new_weight_l700_700713


namespace quadrilateral_no_circumscribed_circle_is_existential_l700_700312

theorem quadrilateral_no_circumscribed_circle_is_existential :
  ∃ (Q : Type) (q : Q) (h : ¬(∃ (C : Type) (c : C), q ∈ c)), True :=
sorry

end quadrilateral_no_circumscribed_circle_is_existential_l700_700312


namespace find_least_n_l700_700217

def a : ℕ → ℕ
| 10 := 10
| (n + 1) := if h : n ≥ 10 then 100 * a n + (n + 1) else 0

theorem find_least_n (n : ℕ) (h₁ : n > 10) (h₂ : a n % 99 = 0) : n = 45 :=
sorry

end find_least_n_l700_700217


namespace long_side_length_is_correct_l700_700634

-- Definitions for the conditions of the problem
def box_long_side_height : ℕ := 6
def box_short_side_width : ℕ := 5
def box_short_side_height : ℕ := 6
def top_or_bottom_area : ℕ := 40
def total_velvet_needed : ℕ := 236
def num_long_sides : ℕ := 2
def num_short_sides : ℕ := 2
def num_tops_and_bottoms : ℕ := 2

-- The length of the long sides
def long_side_length (total_velvet_needed : ℕ) (short_side_width : ℕ) (short_side_height : ℕ)
  (box_long_side_height : ℕ) (top_or_bottom_area : ℕ) : ℕ :=
  let short_sides_area := num_short_sides * (short_side_width * short_side_height) in
  let tops_and_bottoms_area := num_tops_and_bottoms * top_or_bottom_area in
  let remaining_area := total_velvet_needed - short_sides_area - tops_and_bottoms_area in
  let one_long_side_area := remaining_area / num_long_sides in
  one_long_side_area / box_long_side_height

theorem long_side_length_is_correct :
  long_side_length total_velvet_needed box_short_side_width box_short_side_height box_long_side_height top_or_bottom_area = 8 :=
by sorry

end long_side_length_is_correct_l700_700634


namespace original_area_of_doubled_rectangle_l700_700294

theorem original_area_of_doubled_rectangle (A_new : ℝ) (h : A_new = 32) :
  ∃ A : ℝ, A * 4 = A_new ∧ A = 8 :=
by {
  use 8,
  split,
  { norm_num, exact h.symm },
  { rfl }
}

end original_area_of_doubled_rectangle_l700_700294


namespace problem_statement_l700_700100

variable {α : Type*}

def sequence_converges_to_sqrt2 (x : ℕ → ℝ) : Prop :=
  ∃ (a : ℝ), a = real.sqrt 2 ∧ ∀ ε > 0, ∃ N, ∀ n ≥ N, |x n - a| < ε

theorem problem_statement (x : ℕ → ℝ) (k : ℕ) :
  (sequence_converges_to_sqrt2 x) → (∀ ε > 0, ∃ n > k, |x n - real.sqrt 2| ≥ ε) :=
sorry

end problem_statement_l700_700100


namespace perpendicular_lines_slope_l700_700180

theorem perpendicular_lines_slope {m : ℝ} 
    (h1 : ∃ m : ℝ, line_through (mk_point (-2) m) (mk_point m 4))
    (h2 : slope (mk_line (mk_point (-2) m) (mk_point m 4)) * (-2) = (-1)) :
    m = 2 := 
by 
    sorry

end perpendicular_lines_slope_l700_700180


namespace skittles_initial_count_l700_700260

def skittles (number_friends number_per_friend samuel_eat : ℕ) : Prop :=
  (number_friends = 4) → 
  (number_per_friend = 3) → 
  (samuel_eat = 3) → 
  (4 * 3 + 3 = 15)

theorem skittles_initial_count : skittles 4 3 3 :=
by
  intro number_friends number_per_friend samuel_eat,
  sorry

end skittles_initial_count_l700_700260


namespace incorrect_statement_B_l700_700949

variables {a b c : ℝ}

-- Define the quadratic function
def quadratic_function (x : ℝ) : ℝ := a * x^2 + b * x + c

-- The Lean statement to prove statement (B) ac < 0 is incorrect given the conditions
theorem incorrect_statement_B (h₁ : a ≠ 0) -- condition that a is not zero, so it's a quadratic function
    (h₂ : ∀ x : ℝ, quadratic_function x = a * x^2 + b * x + c) :
    ¬ (a * c < 0) :=
sorry

end incorrect_statement_B_l700_700949


namespace magnitude_diff_is_sqrt_10_l700_700160

variable (a : ℝ × ℝ)
variable (b : ℝ × ℝ)
variable (x : ℝ)

-- Given conditions
def a := (x, 1)
def b := (1, -2)

-- a is perpendicular to b implies their dot product is zero
def perp (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0

-- Rewriting the problem as Lean 4 statement
theorem magnitude_diff_is_sqrt_10 (h : perp a b) (hx : x = 2) :
  real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2) = real.sqrt 10 :=
sorry

end magnitude_diff_is_sqrt_10_l700_700160


namespace estimate_total_children_l700_700480

noncomputable theory

variables (k m n : ℕ) (hn : n ≠ 0)

theorem estimate_total_children (h1 : k > 0) (h2 : m > 0) (h3 : n ≤ m) :
  ∃ (T : ℕ), T = k * (m / n) :=
by
  sorry

end estimate_total_children_l700_700480


namespace pencils_across_diameter_l700_700965

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l700_700965


namespace quadratic_solution_l700_700560

theorem quadratic_solution (x : ℝ) (h_eq : x^2 - 3 * x - 6 = 0) (h_neq : x ≠ 0) :
    x = (3 + Real.sqrt 33) / 2 ∨ x = (3 - Real.sqrt 33) / 2 :=
by
  sorry

end quadratic_solution_l700_700560


namespace properties_of_data_set_l700_700739

def data_set : List ℕ := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def sorted_data_set : List ℕ := [30, 31, 31, 37, 40, 46, 47, 57, 62, 67]

def mode (l : List ℕ) : ℕ :=
(d : ℕ × ℕ) ← list.maximumBy (λ d, l.count d) l
(_, x) := d
x

def range (l : List ℕ) : ℕ :=
(list.maximum l - list.minimum l)

def quantile (l : List ℕ) (q : ℕ) : ℝ :=
let pos := (q * l.length) / 100
let sorted := l.sort λ a b => a < b
let lower := sorted[pos]
let upper := sorted[pos + 1]
(lower + upper) / 2

theorem properties_of_data_set :
  mode data_set = 31 ∧
  range data_set = 37 ∧
  quantile data_set 10 = 30.5 :=
by sorry

end properties_of_data_set_l700_700739


namespace rhombus_perimeter_l700_700292

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side_length := real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side_length = 16 * real.sqrt 13 :=
by
  let d1_half := d1 / 2
  let d2_half := d2 / 2
  have h3 : d1_half = 12 := by sorry
  have h4 : d2_half = 8 := by sorry
  let side_length := real.sqrt (d1_half ^ 2 + d2_half ^ 2)
  have h5 : side_length = real.sqrt 208 := by sorry
  have h6 : real.sqrt 208 = 4 * real.sqrt 13 := by sorry
  show 4 * side_length = 16 * real.sqrt 13
  from by
    rw [h6]
    rfl

end rhombus_perimeter_l700_700292


namespace largest_avg_5_l700_700706

def arithmetic_avg (a l : ℕ) : ℚ :=
  (a + l) / 2

def multiples_avg_2 (n : ℕ) : ℚ :=
  arithmetic_avg 2 (n - (n % 2))

def multiples_avg_3 (n : ℕ) : ℚ :=
  arithmetic_avg 3 (n - (n % 3))

def multiples_avg_4 (n : ℕ) : ℚ :=
  arithmetic_avg 4 (n - (n % 4))

def multiples_avg_5 (n : ℕ) : ℚ :=
  arithmetic_avg 5 (n - (n % 5))

def multiples_avg_6 (n : ℕ) : ℚ :=
  arithmetic_avg 6 (n - (n % 6))

theorem largest_avg_5 (n : ℕ) (h : n = 101) : 
  multiples_avg_5 n > multiples_avg_2 n ∧ 
  multiples_avg_5 n > multiples_avg_3 n ∧ 
  multiples_avg_5 n > multiples_avg_4 n ∧ 
  multiples_avg_5 n > multiples_avg_6 n :=
by
  sorry

end largest_avg_5_l700_700706


namespace Ratio_PF_MN_l700_700531

noncomputable def ellipseFocus : Type :=
  ∃ F : ℝ×ℝ, F = (4, 0) ∧
  ∃ P : ℝ×ℝ, 
  ∃ M N : ℝ×ℝ,
    let x := (P.1), y := (P.2), k := (k:ℝ),
    ∃ k : ℝ, k ≠ 0 ∧
    (M.2 = k * (M.1 - 4)) ∧
    (N.2 = k * (N.1 - 4)) ∧
    (∀ (x y : ℝ), (x^2 / 25) + (y^2 / 9) = 1 → (x,y) = M ∨ (x,y) = N) ∧
    (∀ (x : ℝ), 0 + (36*k) / (9 + 25*k^2) = -(1 / k) * (x - (100*k^2 / (9 + 25*k^2))) → x = (64*k^2) / (9 + 25*k^2) → x = P.1) ∧
    (|4 - (64*k^2) / (9 + 25*k^2)| = (36*(1+k^2)) / (9 + 25*k^2)) ∧
    (|M.1 - N.1|^2 + |M.2 - N.2>|^2 = (90 * (1 + k^2)) / (9 + 25*k^2)) ∧
    (|4 - (64*k^2)/(9+25*k^2)| / ((90*(1+k^2)) / (9+25*k^2)) = 2 / 5)

theorem Ratio_PF_MN (F P M N : ellipseFocus) : 
  ∀ F P M N, (F : (4, 0)) → 
  (∃ k: ℝ, k ≠ 0 ∧  
    let k := k,
    (M.2 = k * (M.1 - 4)) ∧
    (N.2 = k * (N.1 - 4)) ∧
    (∀ (x y : ℝ), (x^2 / 25) + (y^2 / 9) = 1 → (x,y) = M ∨ (x,y) = N) ∧
    (∀ (x: ℝ), 0 + (36*k) / (9 + 25*k^2) = -(1 / k) * (x - (100*k^2 / (9 + 25*k^2))) → x = (64*k^2) / (9 + 25*k^2) → x = P.1) ∧
    (|4 - (64*k^2) / (9 + 25*k^2)| = (36*(1+k^2)) / (9 + 25*k^2)) ∧
    (|M.1 - N.1|^2 + |M.2 - N.2|^2 = (90 * (1 + k^2)) / (9 + 25*k^2)) ∧
    (|4 - (64*k^2)/(9+25*k^2)| / ((90*(1+k^2)) / (9+25*k^2)) = 2 / 5)) :=
by sorry

end Ratio_PF_MN_l700_700531


namespace symmetric_point_locus_is_circle_l700_700488

-- Define the two fixed points A and B
variables (A B : EuclideanSpace ℝ (Fin 2))

-- Define the symmetry point of A w.r.t. a line rotating around B
noncomputable def symmetric_point (θ : ℝ) : EuclideanSpace ℝ (Fin 2) :=
  let dir := Function.ContinuousMap.rotate θ (B - A) in
  2 • dir + A

-- State the proof problem in Lean
theorem symmetric_point_locus_is_circle :
  ∀ θ : ℝ, dist B (symmetric_point A B θ) = dist A B :=
sorry

end symmetric_point_locus_is_circle_l700_700488


namespace necessary_and_sufficient_condition_l700_700257

noncomputable def odd_function (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = -f (x)

noncomputable def periodic_function (ϕ : ℝ → ℝ) (T : ℝ) : Prop :=
∀ x, ϕ (x + T) = ϕ (x)

noncomputable def linear_function (f : ℝ → ℝ) (k h : ℝ) : Prop :=
∀ x, f (x) = k*x + h

noncomputable def center_of_symmetry (f : ℝ → ℝ) (a b : ℝ) : Prop :=
∀ x1 x2, f (x1) + f (x2) = 2*b → a = (x1 + x2) / 2

theorem necessary_and_sufficient_condition
  (f : ℝ → ℝ)
  (a b : ℝ)
  (k h T : ℝ)
  (h1 : odd_function f)
  (h2 : periodic_function (λ x, f x - k*x - h) T)
  (h3 : linear_function (λ x, k*x + h) k h)
  (h4 : center_of_symmetry f a b)
  (h5 : a ≠ 0 ∧ b ≠ 0) :
  b = a*k :=
sorry

end necessary_and_sufficient_condition_l700_700257


namespace floor_ceil_sum_l700_700039

theorem floor_ceil_sum :
  (⌊1.999⌋ + ⌈3.001⌉) = 5 :=
by
  have h1 : ⌊1.999⌋ = 1 := by sorry
  have h2 : ⌈3.001⌉ = 4 := by sorry
  rw [h1, h2]
  exact rfl

end floor_ceil_sum_l700_700039


namespace minimal_fence_length_l700_700369

-- Define the conditions as assumptions
axiom side_length : ℝ
axiom num_paths : ℕ
axiom path_length : ℝ

-- Assume the conditions given in the problem
axiom side_length_value : side_length = 50
axiom num_paths_value : num_paths = 13
axiom path_length_value : path_length = 50

-- Define the theorem to be proved
theorem minimal_fence_length : (num_paths * path_length) = 650 := by
  -- The proof goes here
  sorry

end minimal_fence_length_l700_700369


namespace part1_part2_l700_700930

def f (x : ℝ) (ω : ℝ) := (√3) * Real.sin (2 * ω * x) + 2 * (Real.cos (ω * x))^2

theorem part1 (ω : ℝ) :
  (∀ x, f x ω = 2 * Real.sin(2 * x + (π/6)) + 1) →
  (∀ x, disjoint (Icc (π/6 + x * π) (2*π/3 + x * π)) ∧ ∥ ω ∥ = 1) :=
sorry

theorem part2 (m : ℝ) :
  (∃ x, -π/4 < x ∧ x < π/4 ∧ f x 1 = m) →
  (1 - √3 < m ∧ m ≤ 2) :=
sorry

end part1_part2_l700_700930


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700050

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700050


namespace proof_l700_700901

-- Define the vectors a and b
def a : ℝ × ℝ := (2, 2)
def b (m : ℝ) : ℝ × ℝ := (1, m)

-- Define the condition
def condition (m : ℝ) : Prop :=
  let u := (2 * a.1 + b m.1, 2 * a.2 + b m.2)
  let v := (2 * a.1 - b m.1, 2 * a.2 - b m.2)
  (u.1 * u.1 + u.2 * u.2) = (v.1 * v.1 + v.2 * v.2)

-- Prove m = -1 and |b| = sqrt(2)
theorem proof (m : ℝ) (h : condition m) : m = -1 ∧ ∥b m∥ = Real.sqrt 2 :=
  sorry

end proof_l700_700901


namespace smallest_spherical_triangle_angle_l700_700307

-- Define the conditions
def is_ratio (a b c : ℕ) : Prop := a = 4 ∧ b = 5 ∧ c = 6
def sum_of_angles (α β γ : ℕ) : Prop := α + β + γ = 270

-- Define the problem statement
theorem smallest_spherical_triangle_angle 
  (a b c α β γ : ℕ)
  (h1 : is_ratio a b c)
  (h2 : sum_of_angles (a * α) (b * β) (c * γ)) :
  a * α = 72 := 
sorry

end smallest_spherical_triangle_angle_l700_700307


namespace carls_garden_area_is_correct_l700_700405

-- Define the conditions
def isRectangle (length width : ℕ) : Prop :=
∃ l w, l * w = length * width

def validFencePosts (shortSidePosts longSidePosts totalPosts : ℕ) : Prop :=
∃ x, totalPosts = 2 * x + 2 * (2 * x) - 4 ∧ x = shortSidePosts

def validSpacing (shortSideSpaces longSideSpaces : ℕ) : Prop :=
shortSideSpaces = 4 * (shortSideSpaces - 1) ∧ longSideSpaces = 4 * (longSideSpaces - 1)

def correctArea (shortSide longSide expectedArea : ℕ) : Prop :=
shortSide * longSide = expectedArea

-- Prove the conditions lead to the expected area
theorem carls_garden_area_is_correct :
  ∃ shortSide longSide,
  isRectangle shortSide longSide ∧
  validFencePosts 5 10 24 ∧
  validSpacing 5 10 ∧
  correctArea (4 * (5-1)) (4 * (10-1)) 576 :=
by
  sorry

end carls_garden_area_is_correct_l700_700405


namespace rhombus_perimeter_l700_700289

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side_length := real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side_length = 16 * real.sqrt 13 :=
by
  let d1_half := d1 / 2
  let d2_half := d2 / 2
  have h3 : d1_half = 12 := by sorry
  have h4 : d2_half = 8 := by sorry
  let side_length := real.sqrt (d1_half ^ 2 + d2_half ^ 2)
  have h5 : side_length = real.sqrt 208 := by sorry
  have h6 : real.sqrt 208 = 4 * real.sqrt 13 := by sorry
  show 4 * side_length = 16 * real.sqrt 13
  from by
    rw [h6]
    rfl

end rhombus_perimeter_l700_700289


namespace philosophical_enlightenment_l700_700348

theorem philosophical_enlightenment
  (significance_of_philosophy : ∀ (p : Prop), p → Prop)
  (philosophy_guides_people : ∀ (p : Prop), p → Prop)
  (philosophy_seeks_light : ∀ (p : Prop), p → Prop)
  (guidance_of_understanding : ∀ (p : Prop), p → Prop)
  (guidance_of_transformation : ∀ (p : Prop), p → Prop) :
  (significance_of_philosophy "Philosophy" → philosophy_guides_people "Philosophy" → 
   philosophy_seeks_light "Philosophy" → guidance_of_understanding "Philosophy" ∧
   guidance_of_transformation "Philosophy") →
  "Utilize philosophy to guide people in understanding and transforming the world" = "D" :=
by
  sorry

end philosophical_enlightenment_l700_700348


namespace find_DA_length_l700_700998

-- Definitions based on given conditions.
def is_rectangle (ABCD : Type) [preorder ABCD] : Prop := true  -- Simplified, actual definition will include geometrical validations.
def semicircle (O : Type) [preorder O] (diameter : ℝ) : Prop := true  -- Simplified.
def line (ℓ : Type) [preorder ℓ] (intersects : set (ℓ × _)) : Prop := true  -- Simplified.
def conditions (ABCD ℓ: Type) [preorder ABCD] [preorder ℓ] : Prop :=
  let AV := 40
  let AP := 60
  let VB := 80
  is_rectangle ABCD ∧ semicircle ABCD 6 ∧ line ℓ  -- placeholder, as actual intersect conditions would be complex.
  
theorem find_DA_length (ABCD ℓ: Type) [preorder ABCD] [preorder ℓ] (h : conditions ABCD ℓ):
  let k := 30
  let j := 6
  k + j = 36 := sorry

end find_DA_length_l700_700998


namespace geometric_series_modulo_l700_700806

theorem geometric_series_modulo :
  let S := (Finset.range 1502).sum (λ n => 9^n)
  in S % 2000 = 10 := by
  sorry

end geometric_series_modulo_l700_700806


namespace largest_number_l700_700721

def lcm (a b : ℕ) : ℕ := (a * b) / Nat.gcd a b

def hcf (a b : ℕ) : ℕ := Nat.gcd a b

theorem largest_number {a b c : ℕ} 
  (hcf_abc : hcf (hcf a b) c = 37)
  (lcm_factors : lcm a (lcm b c) = 37 * 17 * 19 * 23 * 29) :
  max a (max b c) = 7_976_237 := 
sorry

end largest_number_l700_700721


namespace find_x_squared_plus_y_squared_l700_700554

open Real

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 := by
  sorry

end find_x_squared_plus_y_squared_l700_700554


namespace quadratic_trinomial_sum_l700_700746

theorem quadratic_trinomial_sum (x1 x2 p q : ℤ)
  (h1 : p = x1 + x2)
  (h2 : q = x1 * x2)
  (h3 : ∃ d : ℤ, x1 = x2 + d)
  (h4 : q = x2)
  (h5 : x1 > x2 ∧ x2 > q) : 
  x2 = -3 ∨ x2 = -2 → x2 + x2 = -5 :=
by
  intros h_cases
  cases h_cases;
  simp [int.add, int.neg_succ_of_nat_eq];
  admit

end quadratic_trinomial_sum_l700_700746


namespace find_a7_a8_a12_l700_700898

noncomputable def arithmetic_sequence_sum (a d : ℕ → ℝ) (n : ℕ) : ℝ :=
  n / 2 * (a(1) + a(n))

noncomputable def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
∀ n : ℕ, ∃ d : ℝ, a(n + 1) = a(n) + d

noncomputable def Sn (a : ℕ → ℝ) (n : ℕ) : ℝ :=
n / 2 * (a(1) + a(n))

theorem find_a7_a8_a12 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : is_arithmetic_sequence a) (h2 : S 17 = 170) :
  a(7) + a(8) + a(12) = 30 :=
sorry

end find_a7_a8_a12_l700_700898


namespace isosceles_triangle_largest_angle_l700_700580

theorem isosceles_triangle_largest_angle (a b c : ℝ) (h1 : a = b) (h2 : b_angle = 50) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) 
  (h6 : a + b + c = 180) : c ≥ a ∨ c ≥ b → c = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l700_700580


namespace find_f_double_prime_l700_700884

variable (f : ℝ → ℝ)

axiom h₁ : ∃ (c : ℝ), f = λ x, x^2 + 3 * x * c

theorem find_f_double_prime :
  (∃ (c : ℝ), f = λ x, x^2 + 3 * x * c) → (∀ x, deriv (deriv f) x = deriv (deriv (λ x, x^2 + 3 * x * c)) 1) := 
by
  sorry

end find_f_double_prime_l700_700884


namespace is_concyclic_C_P_Q_X_l700_700228

-- Assume basic geometry concepts such as points, lines, triangles, and circles are defined in the existing context.

theorem is_concyclic_C_P_Q_X
  {A B C P Q R X : Point}
  (h1 : OnLine P BC)
  (h2 : OnLine Q CA)
  (h3 : OnLine R AB)
  (h4 : IsCyclic A Q R X)
  (h5 : IsCyclic B R P X) :
  IsCyclic C P Q X :=
sorry

end is_concyclic_C_P_Q_X_l700_700228


namespace closest_ratio_adults_children_l700_700660

theorem closest_ratio_adults_children 
  (a c : ℕ) 
  (H1 : 30 * a + 15 * c = 2550) 
  (H2 : a > 0) 
  (H3 : c > 0) : 
  (a = 57 ∧ c = 56) ∨ (a = 56 ∧ c = 58) :=
by
  sorry

end closest_ratio_adults_children_l700_700660


namespace ratio_circumference_area_12_l700_700748

-- Define the radius of the circular garden
def radius : ℝ := 12

-- Define the circumference of the circular garden
def circumference (r : ℝ) : ℝ := 2 * Real.pi * r

-- Define the area of the circular garden
def area (r : ℝ) : ℝ := Real.pi * r^2

-- Define the ratio of the circumference to the area
def ratio (r : ℝ) : ℝ := circumference r / area r

-- Theorem stating that the ratio for a circle of radius 12 is equal to 1/6
theorem ratio_circumference_area_12 :
  ratio radius = 1/6 :=
by
  sorry

end ratio_circumference_area_12_l700_700748


namespace none_positive_when_y_lt_0_l700_700174

theorem none_positive_when_y_lt_0 (y : ℝ) (hy : y < 0) : 
  ¬ (0 < y / |y|) ∧ ¬ (0 < -y^3) ∧ ¬ (0 < -3^y) ∧ ¬ (0 < -y^(-2)) ∧ ¬ (0 < sin y) :=
by 
  sorry

end none_positive_when_y_lt_0_l700_700174


namespace expected_product_value_l700_700398

-- Define the conditions of the problem
def cube_faces : List ℕ := [0, 0, 0, 1, 1, 2]

noncomputable def expected_value_product : ℚ :=
  let outcomes := do
    face1 <- cube_faces
    face2 <- cube_faces
    [face1 * face2]
  let outcome_probs := outcomes.freqBy (=) |>.val.map (λ x => (↑x : ℚ) / (cube_faces.length * cube_faces.length : ℚ))
  let expected_values := List.zip outcomes outcome_probs
  expected_values.sumBy (λ (outcome, prob) => (outcome : ℚ) * prob)

-- Statement of the theorem
theorem expected_product_value :
  expected_value_product = 4 / 9 :=
sorry

end expected_product_value_l700_700398


namespace range_of_a_l700_700936

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (0:ℝ) (1:ℝ), f x = 2^(x*(x-a)) ∧ monotone_decreasing_on f (Ioo (0:ℝ) (1:ℝ))) :
  a ∈ set.Ici (2 : ℝ) := sorry

end range_of_a_l700_700936


namespace min_n_for_solutions_16653_l700_700919

theorem min_n_for_solutions_16653 (x y z n : ℕ) (k : ℕ) (h : x + 11 * y + 11 * z = n) 
  (hn_pos : n > 0) (h_solutions : ∃ S : finset (ℕ × ℕ × ℕ), S.card = 16653 ∧ ∀ (t : ℕ × ℕ × ℕ), 
  t ∈ S → (t.1.1 + 11 * t.1.2 + 11 * t.2 = n ∧ t.1.1 > 0 ∧ t.1.2 > 0 ∧ t.2 > 0))
  : n = 2014 :=
sorry

end min_n_for_solutions_16653_l700_700919


namespace number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l700_700870

def five_digit_number_count : Nat :=
  -- Number of ways to select and arrange odd digits in two groups
  let group_odd_digits := (Nat.choose 3 2) * (Nat.factorial 2)
  -- Number of ways to arrange the even digits
  let arrange_even_digits := Nat.factorial 2
  -- Number of ways to insert two groups of odd digits into the gaps among even digits
  let insert_odd_groups := (Nat.factorial 3)
  -- Total ways
  group_odd_digits * arrange_even_digits * arrange_even_digits * insert_odd_groups

theorem number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72 :
  five_digit_number_count = 72 :=
by
  -- Placeholder for proof
  sorry

end number_of_five_digit_numbers_without_repeating_digits_with_two_adjacent_odds_is_72_l700_700870


namespace solution_set_of_x_l700_700075

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l700_700075


namespace total_students_l700_700641

theorem total_students 
  (S : ℕ) 
  (h1 : S / 3 + 3 * 8 + 2 = S)
  (h2 : 2 * ((2 * 13) - 3) + 1 = ℕ):
  S = 39 := 
sorry

end total_students_l700_700641


namespace average_age_of_students_is_14_l700_700183

noncomputable def average_age_of_students (student_count : ℕ) (teacher_age : ℕ) (combined_avg_age : ℕ) : ℕ :=
  let total_people := student_count + 1
  let total_combined_age := total_people * combined_avg_age
  let total_student_age := total_combined_age - teacher_age
  total_student_age / student_count

theorem average_age_of_students_is_14 :
  average_age_of_students 50 65 15 = 14 :=
by
  sorry

end average_age_of_students_is_14_l700_700183


namespace complex_polynomial_inequality_l700_700606

noncomputable def complex_polynomial (n : ℕ) : Type :=
  { P : ℂ → ℂ // ∃ (c : Finₓ (n + 1) → ℝ), P = λ z => ∑ i in Finₓ.range (n + 1), c i * z ^ (n - i) }

theorem complex_polynomial_inequality (n : ℕ) (P : complex_polynomial n) (hP : ∃ k, abs (P.val I) < 1) :
  ∃ (a b : ℝ), P.val (a + b * I) = 0 ∧ (a^2 + b^2 + 1)^2 < 4 * b^2 + 1 :=
sorry

end complex_polynomial_inequality_l700_700606


namespace viewers_ratio_Leila_Voltaire_l700_700696

open Real

theorem viewers_ratio_Leila_Voltaire 
(h_v_avg : ℝ)
(h_earn_per_view : ℝ)
(h_L_earn_per_week : ℝ)
(h_V_daily_viewers : h_v_avg = 50)
(h_earn_per_view : h_earn_per_view = 0.5)
(h_L_earn_per_week : h_L_earn_per_week = 350)
:
(L_weekly_views : ℝ) = h_L_earn_per_week / h_earn_per_view
(V_weekly_views : ℝ) = h_v_avg * 7
(L_weekly_views / V_weekly_views) = 2 :=
by sorry

end viewers_ratio_Leila_Voltaire_l700_700696


namespace typing_time_l700_700003

def original_speed : ℕ := 212
def reduction : ℕ := 40
def new_speed : ℕ := original_speed - reduction
def document_length : ℕ := 3440
def required_time : ℕ := 20

theorem typing_time :
  document_length / new_speed = required_time :=
by
  sorry

end typing_time_l700_700003


namespace max_gold_coins_l700_700251

variables (planks : ℕ)
          (windmill_planks windmill_gold : ℕ)
          (steamboat_planks steamboat_gold : ℕ)
          (airplane_planks airplane_gold : ℕ)

theorem max_gold_coins (h_planks: planks = 130)
                       (h_windmill: windmill_planks = 5 ∧ windmill_gold = 6)
                       (h_steamboat: steamboat_planks = 7 ∧ steamboat_gold = 8)
                       (h_airplane: airplane_planks = 14 ∧ airplane_gold = 19) :
  ∃ (gold : ℕ), gold = 172 :=
by
  sorry

end max_gold_coins_l700_700251


namespace edge_lengths_of_tetrahedron_l700_700872

-- Define the vertices of the Tetrahedron
structure Tetrahedron (α : Type) [InnerProductSpace ℝ α] :=
  (A B C D : α)

-- Define the condition that four out of six midpoints of the edges form a regular tetrahedron
def midpoints_form_regular_tetrahedron {α : Type} [InnerProductSpace ℝ α] 
  (t : Tetrahedron α) (length : ℝ) : Prop :=
  let mAB := 0.5 • (t.A + t.B) in
  let mAC := 0.5 • (t.A + t.C) in
  let mAD := 0.5 • (t.A + t.D) in
  let mBC := 0.5 • (t.B + t.C) in
  let mBD := 0.5 • (t.B + t.D) in
  let mCD := 0.5 • (t.C + t.D) in
  (dist mBC mBD = length ∧ dist mBC mCD = length ∧ dist mBD mCD = length) ∧
  (dist mAC mAD = length ∧ dist mAC mCD = length ∧ dist mAD mCD = length)

-- Define a function that gets the edge lengths of a tetrahedron
def edge_lengths_tetrahedron {α : Type} [InnerProductSpace ℝ α] (t : Tetrahedron α) : list ℝ :=
  [dist t.A t.B, dist t.A t.C, dist t.A t.D, dist t.B t.C, dist t.B t.D, dist t.C t.D]

-- Define the theorem
theorem edge_lengths_of_tetrahedron {α : Type} [InnerProductSpace ℝ α] 
  (t : Tetrahedron α) (h : midpoints_form_regular_tetrahedron t 1) :
  (edge_lengths_tetrahedron t = [2, 2, 2, 2, 2, 2 * Real.sqrt 2]) ∨ 
  (edge_lengths_tetrahedron t = [2 * Real.sqrt 2, 2, 2, 2, 2, 2]) ∨ 
  (edge_lengths_tetrahedron t = [2, 2 * Real.sqrt 2, 2, 2, 2, 2]) ∨ 
  (edge_lengths_tetrahedron t = [2, 2, 2 * Real.sqrt 2, 2, 2, 2]) ∨ 
  (edge_lengths_tetrahedron t = [2, 2, 2, 2 * Real.sqrt 2, 2, 2]) ∨ 
  (edge_lengths_tetrahedron t = [2, 2, 2, 2, 2 * Real.sqrt 2, 2]) ∨ 
  (edge_lengths_tetrahedron t = [2, 2, 2, 2, 2, 2 * Real.sqrt 2]) :=
sorry

end edge_lengths_of_tetrahedron_l700_700872


namespace volume_of_region_correct_l700_700459

noncomputable def volume_of_region : ℝ :=
  ∫ x in 0..1, (∑ y in 0..(⌊50 * (frac x) - ⌊x⌋)⌋, ∑ z in 0..(⌊50 * (frac x) - ⌊x⌋ - y⌋)) dx

theorem volume_of_region_correct:
  volume_of_region = 20912.5 :=
by
  sorry

end volume_of_region_correct_l700_700459


namespace perception_permutations_count_l700_700826

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l700_700826


namespace intersection_M_N_l700_700158

noncomputable def M : set ℝ := {x | x^2 + x - 2 < 0}
noncomputable def N : set ℝ := {x | real.logb (1 / 2) x > -1}

theorem intersection_M_N :
  M ∩ N = {x : ℝ | 0 < x ∧ x < 1} :=
by
  sorry

end intersection_M_N_l700_700158


namespace cos_sum_alpha_pi_over_3_l700_700112

theorem cos_sum_alpha_pi_over_3 (α : ℝ) 
  (h1 : sin α = 4 * real.sqrt 3 / 7) 
  (h2 : 0 < α ∧ α < real.pi / 2) : 
  cos (α + real.pi / 3) = -11 / 14 :=
by
  sorry

end cos_sum_alpha_pi_over_3_l700_700112


namespace medians_l700_700464

-- Define the data for groups A and B
def group_A_data := [28, 31, 39, 45, 42, 55, 58, 57, 66]
def group_B_data := [29, 34, 35, 48, 42, 46, 55, 53, 55, 67]

-- Define a function that computes the median of a list of integers
noncomputable def median (l : List Int) : Float :=
  let sorted_l := l.qsort (· < ·)
  let len := sorted_l.length
  if len % 2 = 1 then
    sorted_l.get! (len / 2) 
  else
    (sorted_l.get! (len / 2 - 1) + sorted_l.get! (len / 2)) / 2

-- Define the theorem to prove medians for groups A and B
theorem medians :
  median group_A_data = 45 ∧ median group_B_data = 47 := by
  sorry

end medians_l700_700464


namespace frames_cost_is_200_l700_700203

noncomputable def lenses_cost : ℕ := 500
noncomputable def insurance_coverage : ℕ := 80
noncomputable def coupon_discount : ℕ := 50
noncomputable def total_cost : ℕ := 250

theorem frames_cost_is_200 : ∃ F : ℕ, 
  let insurance_amount := (insurance_coverage * lenses_cost) / 100
  let remaining_lenses_cost := lenses_cost - insurance_amount
  let frame_cost_after_coupon := F - coupon_discount
  in (frame_cost_after_coupon + remaining_lenses_cost = total_cost) ∧ F = 200 :=
by
  sorry

end frames_cost_is_200_l700_700203


namespace factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l700_700436

-- Problem 1: Prove the factorization of 4x^2 - 25y^2
theorem factorize_diff_of_squares (x y : ℝ) : 4 * x^2 - 25 * y^2 = (2 * x + 5 * y) * (2 * x - 5 * y) := 
sorry

-- Problem 2: Prove the factorization of -3xy^3 + 27x^3y
theorem factorize_common_factor_diff_of_squares (x y : ℝ) : 
  -3 * x * y^3 + 27 * x^3 * y = -3 * x * y * (y + 3 * x) * (y - 3 * x) := 
sorry

end factorize_diff_of_squares_factorize_common_factor_diff_of_squares_l700_700436


namespace max_cylinder_volume_in_cone_l700_700853

theorem max_cylinder_volume_in_cone :
  ∃ x, (0 < x ∧ x < 1) ∧ ∀ y, (0 < y ∧ y < 1 → y ≠ x → ((π * (-2 * y^3 + 2 * y^2)) ≤ (π * (-2 * x^3 + 2 * x^2)))) ∧ 
  (π * (-2 * x^3 + 2 * x^2) = 8 * π / 27) := sorry

end max_cylinder_volume_in_cone_l700_700853


namespace min_ab_min_a_b_min_a2_b2_min_1a_1b_l700_700882

-- Definitions for the conditions
variables {a b : ℝ}
def condition1 := a > 0
def condition2 := b > 0
def condition3 := a + b + 3 = ab

-- Prove the minimum value of ab is 9
theorem min_ab : condition1 → condition2 → condition3 → ∃ (min_ab: ℝ), min_ab = 9 :=
by sorry

-- Prove the minimum value of a + b is 6
theorem min_a_b : condition1 → condition2 → condition3 → ∃ (min_a_b: ℝ), min_a_b = 6 :=
by sorry

-- Prove the minimum value of a^2 + b^2 is 18
theorem min_a2_b2 : condition1 → condition2 → condition3 → ∃ (min_a2_b2: ℝ), min_a2_b2 = 18 :=
by sorry

-- Prove the minimum value of 1/a + 1/b is 2/3
theorem min_1a_1b : condition1 → condition2 → condition3 → ∃ (min_1a_1b: ℝ), min_1a_1b = 2/3 :=
by sorry

end min_ab_min_a_b_min_a2_b2_min_1a_1b_l700_700882


namespace factorize_difference_of_squares_l700_700445

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
begin
  sorry
end

end factorize_difference_of_squares_l700_700445


namespace floor_eq_solution_l700_700055

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l700_700055


namespace lamp_turn_off_count_l700_700320

theorem lamp_turn_off_count : 
  ∀ (n : ℕ), n = 10 →
  (∀ (k i : ℕ), 1 ≤ i ∧ i ≤ k ∧ k < 10 ∧ k - i ≥ 2 → ∀ (off : Finset ℕ),
  off ⊆ Finset.range 10 ∧ off.card = 3 ∧ ¬ 0 ∈ off ∧ ¬ (n - 1) ∈ off →
  (∀ (x y ∈ off), x ≠ y → |x - y| > 1) → 
  (off.card.choose 3 = 20)) :=
by {
  sorry
}

end lamp_turn_off_count_l700_700320


namespace part_one_part_two_l700_700350

-- Conditions and Definitions
def f₁ (a : ℝ) (x : ℝ) : ℝ := a * x
axiom a_ne_zero (a : ℝ) : a ≠ 0

def f₂ (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, f(x) * f(y) = f(x + y)
axiom f_of_one (f : ℝ → ℝ) : f(1) = 2

-- Proving the first part for f(x) = ax
theorem part_one (a : ℝ) (x y : ℝ) (h : a_ne_zero a) : f₁ a x + f₁ a y = f₁ a (x + y) :=
by sorry

-- Proving the second part for finding specific f(5) given properties
theorem part_two {f : ℝ → ℝ} (h₁ : f₂ f) (h₂ : f_of_one f) : f 5 = 32 :=
by sorry

end part_one_part_two_l700_700350


namespace roots_of_quadratic_l700_700508

theorem roots_of_quadratic (p q x1 x2 : ℕ) (h1 : p + q = 28) (h2 : x1 * x2 = q) (h3 : x1 + x2 = -p) (h4 : x1 > 0) (h5 : x2 > 0) : 
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end roots_of_quadratic_l700_700508


namespace full_day_students_l700_700001

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l700_700001


namespace monotonically_decreasing_interval_l700_700151

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonically_decreasing_interval :
  { x : ℝ | 0 < x ∧ x < 1 / Real.exp 1 } = { x | f' x < 0 } :=
by
  sorry

end monotonically_decreasing_interval_l700_700151


namespace g_g_g_g_of_2_eq_242_l700_700233

def g (x : ℕ) : ℕ :=
  if x % 3 = 0 then x / 3 else 3 * x + 2

theorem g_g_g_g_of_2_eq_242 : g (g (g (g 2))) = 242 :=
by
  sorry

end g_g_g_g_of_2_eq_242_l700_700233


namespace total_games_in_conference_l700_700383

theorem total_games_in_conference (teams_per_division: ℕ) (divisions: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) :
  teams_per_division = 8 → 
  divisions = 2 → 
  intra_division_games = 3 → 
  inter_division_games = 2 → 
  let intradivision_total_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * divisions,
      interdivision_games_per_team := teams_per_division * inter_division_games,
      interdivision_total_games := (interdivision_games_per_team * (teams_per_division * divisions)) / 2,
      total_games := intradivision_total_games + interdivision_total_games
  in total_games = 296 :=
by
  intros h1 h2 h3 h4
  let intradivision_total_games :=
    (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * divisions
  let interdivision_games_per_team :=
    teams_per_division * inter_division_games
  let interdivision_total_games :=
    (interdivision_games_per_team * (teams_per_division * divisions)) / 2
  let total_games := intradivision_total_games + interdivision_total_games
  show total_games = 296
  sorry

end total_games_in_conference_l700_700383


namespace floor_eq_l700_700061

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l700_700061


namespace factorize_difference_of_squares_l700_700437

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l700_700437


namespace max_value_of_expression_l700_700614

theorem max_value_of_expression (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  ∃ (max_value : ℝ), max_value = 2 ∧ (∀ x y : ℝ, 0 < x → 0 < y → (x + y)^2 / (x^2 + y^2) ≤ max_value) :=
begin
  use 2,
  split,
  { 
    refl,
  },
  { 
    assume x y hx hy,
    have h : (x + y)^2 ≤ 2 * (x^2 + y^2),
    { 
      calc (x + y)^2 = x^2 + 2 * x * y + y^2 : by ring
      ... ≤ x^2 + x^2 + y^2 + y^2 : by linarith
      ... = 2 * (x^2 + y^2) : by ring,
    },
    exact div_le_of_le_mul (by linarith) h,
  },
end

end max_value_of_expression_l700_700614


namespace acute_triangle_in_1997_gon_l700_700367

theorem acute_triangle_in_1997_gon :
  ∀ (P : Type) [fintype P] [comm_ring P] (polygon : P) (vertices : fin 1997 → P),
    (∀ (i j : fin 1997), ∃! (triangles : fin 1995 → triangle), true) →
    (∃! (t : triangle), t.is_acute) := by
sory

end acute_triangle_in_1997_gon_l700_700367


namespace q_value_at_2_l700_700613

def q (x d e : ℤ) : ℤ := x^2 + d*x + e

theorem q_value_at_2 (d e : ℤ) 
  (h1 : ∃ p : ℤ → ℤ, ∀ x, x^4 + 8*x^2 + 49 = (q x d e) * (p x))
  (h2 : ∃ r : ℤ → ℤ, ∀ x, 2*x^4 + 5*x^2 + 36*x + 7 = (q x d e) * (r x)) :
  q 2 d e = 5 := 
sorry

end q_value_at_2_l700_700613


namespace arithmetic_prog_sum_one_inv_nat_l700_700847

noncomputable def is_arithmetic_progression (seq : List ℚ) : Prop :=
  ∀ i j, i < j ∧ j < seq.length → seq.nth i + seq.nth (j - i) = 2 * seq.nth j

def is_increasing (seq : List ℚ) : Prop :=
  ∀ i, i < seq.length - 1 → seq.nth i < seq.nth (i + 1)

def is_form_of_inv_nat (seq : List ℚ) : Prop :=
  ∀ i, ∃ k : ℕ, seq.nth i = 1 / k

def finite_sum_one (seq : List ℚ) : Prop :=
  list.sum seq = 1

theorem arithmetic_prog_sum_one_inv_nat (seq : List ℚ) :
  is_arithmetic_progression seq →
  is_increasing seq →
  finite_sum_one seq →
  is_form_of_inv_nat seq →
  seq = [1/6, 1/3, 1/2] :=
by {
  intros, 
  sorry
}

end arithmetic_prog_sum_one_inv_nat_l700_700847


namespace proof_problem_l700_700986

variables (p q : Prop)

theorem proof_problem (hpq : p ∨ q) (hnp : ¬p) : q :=
by
  sorry

end proof_problem_l700_700986


namespace no_12_digit_int_with_sum_76_l700_700253

theorem no_12_digit_int_with_sum_76 (N : ℕ) : 
  (N.to_digits.length = 12) ∧ 
  (∀ d ∈ N.to_digits, d = 1 ∨ d = 5 ∨ d = 9) ∧ 
  (37 ∣ N) ∧ 
  (N.to_digits.sum = 76) → false :=
by
  sorry

end no_12_digit_int_with_sum_76_l700_700253


namespace floor_equation_solution_l700_700045

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l700_700045


namespace books_at_end_of_year_l700_700241

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end books_at_end_of_year_l700_700241


namespace three_digit_permuted_mean_l700_700081

theorem three_digit_permuted_mean (N : ℕ) :
  (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧
    (N = 111 ∨ N = 222 ∨ N = 333 ∨ N = 444 ∨ N = 555 ∨ N = 666 ∨ N = 777 ∨ N = 888 ∨ N = 999 ∨
     N = 407 ∨ N = 518 ∨ N = 629 ∨ N = 370 ∨ N = 481 ∨ N = 592)) ↔
    (∃ x y z : ℕ, N = 100 * x + 10 * y + z ∧ x < 10 ∧ y < 10 ∧ z < 10 ∧ 7 * x = 3 * y + 4 * z) := by
sorry

end three_digit_permuted_mean_l700_700081


namespace prod_ab_eq_three_l700_700714

theorem prod_ab_eq_three (a b : ℝ) (h₁ : a - b = 5) (h₂ : a^2 + b^2 = 31) : a * b = 3 := 
sorry

end prod_ab_eq_three_l700_700714


namespace polynomial_coefficient_B_l700_700026

theorem polynomial_coefficient_B :
  (∃ (roots : Fin 4 → ℕ), (∀ i, 0 < roots i) ∧ (Finset.univ.sum roots = 7) ∧ 
  (∃ A B : ℤ, (Polynomial.from_roots (λ i, (roots i : ℤ)) = Polynomial.of_xs [4, -7, A, B, 24])) ∧ B = -12) :=
sorry

end polynomial_coefficient_B_l700_700026


namespace floor_ceil_sum_l700_700040

theorem floor_ceil_sum :
  (⌊1.999⌋ + ⌈3.001⌉) = 5 :=
by
  have h1 : ⌊1.999⌋ = 1 := by sorry
  have h2 : ⌈3.001⌉ = 4 := by sorry
  rw [h1, h2]
  exact rfl

end floor_ceil_sum_l700_700040


namespace no_solution_to_system_l700_700792

theorem no_solution_to_system :
  ¬ ∃ (x y : ℝ), (3 * x - 4 * y = 8) ∧ (6 * x - 8 * y = 18) :=
by
  sorry

end no_solution_to_system_l700_700792


namespace largest_angle_isosceles_triangle_l700_700577

theorem largest_angle_isosceles_triangle (A B C : ℕ) 
  (h_isosceles : A = B) 
  (h_base_angle : A = 50) : 
  max A (max B C) = 80 := 
by 
  -- proof is omitted  
  sorry

end largest_angle_isosceles_triangle_l700_700577


namespace monotonic_decreasing_interval_l700_700308

noncomputable def f (x : ℝ) := x^2 - 2 * |x|

theorem monotonic_decreasing_interval :
  {x : ℝ | ∃ y : ℝ, f y = f x ∧  (y ∈ Icc 0 1 ∨ y ∈ Iic (-1))} = Icc 0 1 ∪ Iic (-1) :=
sorry

end monotonic_decreasing_interval_l700_700308


namespace average_weight_of_all_boys_l700_700718

theorem average_weight_of_all_boys (total_boys_16 : ℕ) (avg_weight_boys_16 : ℝ)
  (total_boys_8 : ℕ) (avg_weight_boys_8 : ℝ) 
  (h1 : total_boys_16 = 16) (h2 : avg_weight_boys_16 = 50.25)
  (h3 : total_boys_8 = 8) (h4 : avg_weight_boys_8 = 45.15) : 
  (total_boys_16 * avg_weight_boys_16 + total_boys_8 * avg_weight_boys_8) / (total_boys_16 + total_boys_8) = 48.55 :=
by
  sorry

end average_weight_of_all_boys_l700_700718


namespace perception_permutations_count_l700_700828

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l700_700828


namespace find_lambda_l700_700953

def vector_a : ℝ × ℝ := (2, 4)
def vector_b : ℝ × ℝ := (1, 1)

def is_perpendicular (u v : ℝ × ℝ) : Prop :=
  (u.1 * v.1 + u.2 * v.2) = 0

theorem find_lambda 
  (λ : ℝ) 
  (h : is_perpendicular vector_b (vector_a.1 + λ * vector_b.1, vector_a.2 + λ * vector_b.2)) : 
  λ = -3 := 
sorry

end find_lambda_l700_700953


namespace num_divisors_16n5_l700_700917

noncomputable def num_divisors : ℕ → ℕ
| n := if n = 0 then 0 else (list.range (n + 1)).filter (λ d, n % d = 0).length

theorem num_divisors_16n5
  (n : ℕ)
  (h1 : 0 < n)
  (h2 : num_divisors (120 * n^3) = 120) :
  num_divisors (16 * n^5) = 126 := 
sorry

end num_divisors_16n5_l700_700917


namespace product_of_binomials_l700_700095

-- Definition of the binomials
def binomial1 (x : ℝ) : ℝ := 4 * x - 3
def binomial2 (x : ℝ) : ℝ := x + 7

-- The theorem to be proved
theorem product_of_binomials (x : ℝ) : 
  binomial1 x * binomial2 x = 4 * x^2 + 25 * x - 21 :=
by
  sorry

end product_of_binomials_l700_700095


namespace complement_U_A_l700_700626

def universal_set (x : ℝ) : Prop := exp x > 1

def domain_A (x : ℝ) : Prop := x > 1

theorem complement_U_A :
  {x : ℝ | (0 < x ∧ x ≤ 1)} = {x : ℝ | universal_set x} \ {x : ℝ | domain_A x} :=
by
  sorry

end complement_U_A_l700_700626


namespace ones_digit_of_prime_in_arithmetic_sequence_is_one_l700_700873

theorem ones_digit_of_prime_in_arithmetic_sequence_is_one 
  (p q r s : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (hs : Prime s) 
  (h₁ : p > 10) 
  (h₂ : q = p + 10) 
  (h₃ : r = q + 10) 
  (h₄ : s = r + 10) 
  (h₅ : s > r) 
  (h₆ : r > q) 
  (h₇ : q > p) : 
  p % 10 = 1 :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_is_one_l700_700873


namespace problem_number_of_ways_to_choose_2005_balls_l700_700094

def number_of_ways_to_choose_balls (n : ℕ) : ℕ :=
  binomial (n + 2) 2 - binomial ((n + 1) / 2 + 1) 2

theorem problem_number_of_ways_to_choose_2005_balls :
  number_of_ways_to_choose_balls 2005 = binomial 2007 2 - binomial 1004 2 :=
by
  -- Proof will be provided here.
  sorry

end problem_number_of_ways_to_choose_2005_balls_l700_700094


namespace S15_correct_l700_700533

noncomputable def sequence_sum (n : ℕ) : ℤ :=
  (list.sum (list.of_fn (λ i, if i % 2 = 0 then 4 * (i + 1) - 3 else - (4 * (i + 1) - 3)) n) : ℤ)

theorem S15_correct : sequence_sum 15 = 29 := 
sorry

end S15_correct_l700_700533


namespace sara_total_quarters_l700_700261

def initial_quarters : ℝ := 783.0
def given_quarters : ℝ := 271.0

theorem sara_total_quarters : initial_quarters + given_quarters = 1054.0 := 
by
  sorry

end sara_total_quarters_l700_700261


namespace proof_problem_l700_700926

noncomputable def f (x : ℝ) : ℝ := 2 * x - Math.cos x

def a_n (n : ℕ) : ℝ := a_1 + (n - 1) * (π / 8)
-- a_n is defined generally for n-th term where a_1 is the first term
-- and d is the common difference of π / 8

theorem proof_problem (a_1 : ℝ) (a_5 : ℝ) (a_3 : ℝ)
  (h_arith_seq : ∀ n : ℕ, a_n n = a_1 + (n - 1) * (π / 8))
  (h_a3 : a_3 = (a_1 + a_5) / 2)
  (h_sum_f : f(a_1) + f(a_2) + f(a_3) + f(a_4) + f(a_5) = 5 * π) :
  (f(a_3))^2 - a_1 * a_5 = (13 / 16) * (π^2) :=
by sorry

end proof_problem_l700_700926


namespace arithmetic_mean_snakes_length_l700_700417

open Nat

def is_snake {n m : ℕ} (grid : ℕ × ℕ → Prop) (snake : List (ℕ × ℕ)) : Prop :=
  (∀ c ∈ snake, grid c ∧ 1 ≤ c.1 ∧ c.1 ≤ n ∧ 1 ≤ c.2 ∧ c.2 ≤ m) ∧
  (∀ i < snake.length - 1, (snake.nth i).2 = (snake.nth (i + 1)).2 ∨ 
   (snake.nth i).1 = (snake.nth (i + 1)).1 - 1) ∧
  (snake.head.1 = 1) ∧ (snake.last.1 n = n)

noncomputable def expected_snake_length (n m : ℕ) : ℕ :=
  n * (m + 1) / 2

theorem arithmetic_mean_snakes_length (n m : ℕ) (h_n : n ≥ 2) (h_m : m ≥ 2) :
  ∃ snakes : List (List (ℕ × ℕ)), 
  (∀ snake ∈ snakes, is_snake (λ p, 1 ≤ p.1 ∧ p.1 ≤ n ∧ 1 ≤ p.2 ∧ p.2 ≤ m) snake) →
  let mean_length := (snakes.map List.length).sum / snakes.length in
  mean_length = expected_snake_length n m := sorry

end arithmetic_mean_snakes_length_l700_700417


namespace find_other_root_l700_700559

theorem find_other_root (m : ℝ) (h : 2^2 - 2 + m = 0) : 
  ∃ α : ℝ, α = -1 ∧ (α^2 - α + m = 0) :=
by
  -- Assuming x = 2 is a root, prove that the other root is -1.
  sorry

end find_other_root_l700_700559


namespace escher_consecutive_probability_l700_700244

theorem escher_consecutive_probability :
  let total_pieces := 12
  let escher_prints := 4
  let total_circular_permutations := (total_pieces - 1)!
  let remaining_pieces := total_pieces - escher_prints
  let block_positions := remaining_pieces - 1 -- Since one position is fixed
  let remaining_permutations := block_positions!
  let escher_permutations := escher_prints!
  (block_positions * remaining_permutations * escher_permutations) / total_circular_permutations = 1 / 47 :=
by 
  sorry

end escher_consecutive_probability_l700_700244


namespace rhombus_perimeter_l700_700279

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l700_700279


namespace rhombus_perimeter_l700_700286

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 24) (h_d2 : d2 = 16) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side = 16 * Real.sqrt 13 :=
by
  sorry

end rhombus_perimeter_l700_700286


namespace Karlson_drink_ratio_l700_700603

noncomputable def conical_glass_volume_ratio (r h : ℝ) : Prop :=
  let V_fuzh := (1 / 3) * Real.pi * r^2 * h
  let V_Mal := (1 / 8) * V_fuzh
  let V_Karlsson := V_fuzh - V_Mal
  (V_Karlsson / V_Mal) = 7

theorem Karlson_drink_ratio (r h : ℝ) : conical_glass_volume_ratio r h := sorry

end Karlson_drink_ratio_l700_700603


namespace parabola_origin_l700_700377

theorem parabola_origin (x y c : ℝ) (h : y = x^2 - 2 * x + c - 4) (h0 : (0, 0) = (x, y)) : c = 4 :=
by
  sorry

end parabola_origin_l700_700377


namespace part_I_part_II_1_part_II_2_l700_700892

noncomputable def f (x : ℝ) : ℝ := x^2 + 2 * x - 3

def g (x m : ℝ) : ℝ := m * f(x) + 1

theorem part_I : f = (λ x, x^2 + 2 * x - 3) := 
sorry

theorem part_II_1 {m : ℝ} (hm : m < 0) : ∃ x ∈ set.Iic 1, g x m = 0 :=
sorry

theorem part_II_2 {m : ℝ} (hm : 0 < m) : 
  if h₁ : m ≤ 8/7 
  then ∃ x ∈ set.Icc (-3) (3/2), |g x m| = (9/4) * m + 1 
  else ∃ x ∈ set.Icc (-3) (3/2), |g x m| = 4*m - 1 :=
sorry

end part_I_part_II_1_part_II_2_l700_700892


namespace parallel_cd_ab_l700_700201

open EuclideanGeometry

/-- Given a triangle ABC inscribed in a circle with center O, an angle bisector CH intersecting the
circle at H, and a line segment EF parallel to AB intersecting CH at K. The circumcircle of triangle
EFH intersects the circle at G. If line GK intersects the circle again at D, then prove CD is
parallel to AB. -/
theorem parallel_cd_ab 
  {A B C H E F K G D : Point} {O : Circle}
  (h_triangle : Triangle O A B C)
  (h_ac_ne_bc : A ≠ B)
  (h_ch_bisector : BisectsAngle C A B H (LineSegment.mk O C (Point.mk O H)))
  (h_ef_parallel_ab : LineParallel (Line.mk E F) (Line.mk A B))
  (h_intersect_k : IntersectsAt (Line.mk E F) (Line.mk C H) K)
  (h_circumcircle_efh : IsCircumcircle (Circle.mk E F H G))
  (h_intersect_gd : IntersectsAt (Line.mk G K) O D)
: LineParallel (Line.mk C D) (Line.mk A B) :=
sorry

end parallel_cd_ab_l700_700201


namespace molecular_weight_correct_l700_700700

def atomic_weight (atom : String) : ℝ :=
  match atom with
  | "C" => 12.01
  | "H" => 1.008
  | "O" => 16.00
  | _ => 0

def molecular_weight (formula : List (String × ℕ)) : ℝ :=
  formula.foldl (λ acc (atom, count) => acc + count * atomic_weight atom) 0

-- Molecular formula of C6H8O7
def C6H8O7 : List (String × ℕ) := [("C", 6), ("H", 8), ("O", 7)]

-- Total molecular weight
def total_molecular_weight : ℝ := 960

-- Number of moles (given this is the implicit in the problem, we do not need to state it directly)
def number_of_moles := (total_molecular_weight / molecular_weight C6H8O7)

theorem molecular_weight_correct :
  molecular_weight C6H8O7 * number_of_moles = total_molecular_weight :=
sorry

end molecular_weight_correct_l700_700700


namespace max_number_of_circular_triples_l700_700354

theorem max_number_of_circular_triples (players : Finset ℕ) (game_results : ℕ → ℕ → Prop) (total_players : players.card = 14)
  (each_plays_13_others : ∀ (p : ℕ) (hp : p ∈ players), ∃ wins losses : Finset ℕ, wins.card = 6 ∧ losses.card = 7 ∧
    (∀ w ∈ wins, game_results p w) ∧ (∀ l ∈ losses, game_results l p)) :
  (∃ (circular_triples : Finset (Finset ℕ)), circular_triples.card = 112 ∧
    ∀ t ∈ circular_triples, t.card = 3 ∧
    (∀ x y z : ℕ, x ∈ t ∧ y ∈ t ∧ z ∈ t → game_results x y ∧ game_results y z ∧ game_results z x)) := 
sorry

end max_number_of_circular_triples_l700_700354


namespace least_possible_product_of_primes_l700_700326

-- Define a prime predicate for a number greater than 20
def is_prime_over_20 (p : Nat) : Prop := Nat.Prime p ∧ p > 20

-- Define the two primes
def prime1 := 23
def prime2 := 29

-- Given the conditions, prove the least possible product of two distinct primes greater than 20 is 667
theorem least_possible_product_of_primes :
  ∃ p1 p2 : Nat, is_prime_over_20 p1 ∧ is_prime_over_20 p2 ∧ p1 ≠ p2 ∧ (p1 * p2 = 667) :=
by
  -- Theorem statement without proof
  existsi (prime1)
  existsi (prime2)
  have h1 : is_prime_over_20 prime1 := by sorry
  have h2 : is_prime_over_20 prime2 := by sorry
  have h3 : prime1 ≠ prime2 := by sorry
  have h4 : prime1 * prime2 = 667 := by sorry
  exact ⟨h1, h2, h3, h4⟩

end least_possible_product_of_primes_l700_700326


namespace min_stamps_needed_l700_700755

-- Let us define the function that finds the minimum number of stamps needed for 7.5 yuan.
def minStamps (s1 s2 s3 : ℝ) (target : ℝ) :=
  let candidates := [(⌊target / s3⌋, (target % s3) / (s1 + s2)), (⌊target / s2⌋, (target % s2) / (s1 + s3)), (⌊target / s1⌋, (target % s1) / (s2 + s3))]
  candidates.map (λ (n, m), n + m)
  |> List.min

theorem min_stamps_needed :
  minStamps 0.6 0.8 1.1 7.5 = 8 := by
  sorry

end min_stamps_needed_l700_700755


namespace ratio_nickels_dimes_quarters_l700_700313

theorem ratio_nickels_dimes_quarters :
  ∃ (r : ℕ × ℕ × ℕ),
    let v_nickels := 4 * 5 in
    let v_dimes := 6 * 10 in
    let v_quarters := 2 * 25 in
    let gcd := Nat.gcd (Nat.gcd v_nickels v_dimes) v_quarters in
    r = (v_nickels / gcd, v_dimes / gcd, v_quarters / gcd) ∧ r = (2, 6, 5) :=
by
  let v_nickels := 4 * 5
  let v_dimes := 6 * 10
  let v_quarters := 2 * 25
  let gcd := Nat.gcd (Nat.gcd v_nickels v_dimes) v_quarters
  use (v_nickels / gcd, v_dimes / gcd, v_quarters / gcd)
  sorry

end ratio_nickels_dimes_quarters_l700_700313


namespace sequence_S5_l700_700677

noncomputable def a : ℕ → ℤ
| 1     := 1
| (n+1) := 2 * a n + 1

noncomputable def S : ℕ → ℤ
| 0     := 0
| (n+1) := S n + a (n+1)

theorem sequence_S5 : S 5 = 57 := by
  have h1 : a 1 = 1 := rfl
  have h2 : ∀ n : ℕ, a (n+1) = 2 * a n + 1 := λ n, rfl
  sorry

end sequence_S5_l700_700677


namespace alpha_necessary_but_not_sufficient_for_beta_l700_700500

theorem alpha_necessary_but_not_sufficient_for_beta 
  (a b : ℝ) (hα : b * (b - a) ≤ 0) (hβ : a / b ≥ 1) : 
  (b * (b - a) ≤ 0) ↔ (a / b ≥ 1) := 
sorry

end alpha_necessary_but_not_sufficient_for_beta_l700_700500


namespace value_of_x_l700_700683

def x : ℚ :=
  (320 / 2) / 3

theorem value_of_x : x = 160 / 3 := 
by
  unfold x
  sorry

end value_of_x_l700_700683


namespace odd_prime_divisibility_two_prime_divisibility_l700_700619

theorem odd_prime_divisibility (p a n : ℕ) (hp : p % 2 = 1) (hp_prime : Nat.Prime p)
  (ha : a > 0) (hn : n > 0) (div_cond : p^n ∣ a^p - 1) : p^(n-1) ∣ a - 1 :=
sorry

theorem two_prime_divisibility (a n : ℕ) (ha : a > 0) (hn : n > 0) (div_cond : 2^n ∣ a^2 - 1) : ¬ 2^(n-1) ∣ a - 1 :=
sorry

end odd_prime_divisibility_two_prime_divisibility_l700_700619


namespace train_car_count_l700_700842

theorem train_car_count
    (cars_first_15_sec : ℕ)
    (time_first_15_sec : ℕ)
    (total_time_minutes : ℕ)
    (total_additional_seconds : ℕ)
    (constant_speed : Prop)
    (h1 : cars_first_15_sec = 9)
    (h2 : time_first_15_sec = 15)
    (h3 : total_time_minutes = 3)
    (h4 : total_additional_seconds = 30)
    (h5 : constant_speed) :
    0.6 * (3 * 60 + 30) = 126 := by
  sorry

end train_car_count_l700_700842


namespace count_integers_pm_1_to_pm_2018_l700_700164

theorem count_integers_pm_1_to_pm_2018:
  (∃ (S : finset ℤ), (∀ x ∈ S, x = ∑ i in (finset.range 2018).image (λ n, n + 1), (ite (even i) i (-i))) ∧ S.card = 2037172) :=
sorry

end count_integers_pm_1_to_pm_2018_l700_700164


namespace solution_set_l700_700072

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l700_700072


namespace dilation_image_l700_700663

noncomputable def dilation (center : ℂ) (k : ℝ) (w : ℂ) : ℂ :=
  k * (w - center) + center

theorem dilation_image (center : ℂ) (k : ℝ) (w : ℂ) (z : ℂ) :
  center = -1 + 4 * complex.I ∧
  k = -2 ∧
  w = 0 + 2 * complex.I ∧
  z = -3 + 8 * complex.I →
  z = dilation center k w :=
begin
  intros h,
  cases h with hc hk_hwz,
  cases hk_hwz with hk hwz,
  cases hwz with hw hz,
  rw [hc, hk, hw] at *,
  simp [dilation]
end

end dilation_image_l700_700663


namespace range_of_m_l700_700482

theorem range_of_m (m : ℝ) (x : ℝ) :
  (|1 - (x - 1) / 2| ≤ 3) →
  (x^2 - 2 * x + 1 - m^2 ≤ 0) →
  (m > 0) →
  (∃ (q_is_necessary_but_not_sufficient_for_p : Prop), q_is_necessary_but_not_sufficient_for_p →
  (m ≥ 8)) :=
by
  sorry

end range_of_m_l700_700482


namespace conic_section_type_hyperbola_l700_700835

theorem conic_section_type_hyperbola :
  (∃ x y : ℝ, (x-3)^2 = (5y+4)^2 - 175) → "H" :=
by {
  sorry
}

end conic_section_type_hyperbola_l700_700835


namespace AF_perpendicular_BE_l700_700780

open EuclideanGeometry

-- Definitions as per conditions
variable {A B C D E F : Point}
variable (triangle_ABC_isosceles : AB = AC)
variable (D_is_midpoint_BC : midpoint D B C)
variable (E_foot_perpendicular_D_AC : foot E D AC)
variable (F_is_midpoint_DE : midpoint F D E)

-- The statement to be proved
theorem AF_perpendicular_BE
  (h1 : is_isosceles_triangle A B C)
  (h2 : is_midpoint D B C)
  (h3 : is_perpendicular_foot E D A C)
  (h4 : is_midpoint F D E) :
  is_perpendicular (line_through_pts A F) (line_through_pts B E) :=
sorry

end AF_perpendicular_BE_l700_700780


namespace find_all_solutions_to_h_eq_0_l700_700622

def h (x : ℝ) : ℝ := if x < 0 then 4*x + 4 else 3*x - 18

theorem find_all_solutions_to_h_eq_0 :
  {x : ℝ | h x = 0} = {-1, 6} :=
by
  sorry

end find_all_solutions_to_h_eq_0_l700_700622


namespace pucks_cannot_return_to_original_position_l700_700321

theorem pucks_cannot_return_to_original_position 
  (A B C : Type) 
  (hitting : A → A → A → A) 
  (initial_position : A × A × A) 
  (hits : ℕ) : 
  hits = 25 → 
  initial_position ≠ (hitting (initial_position.1) (initial_position.2) (initial_position.3)) := 
by 
  sorry

end pucks_cannot_return_to_original_position_l700_700321


namespace wall_building_l700_700598

-- Definitions based on conditions
def total_work (m d : ℕ) : ℕ := m * d

-- Prove that if 30 men including 10 twice as efficient men work for 3 days, they can build the wall
theorem wall_building (m₁ m₂ d₁ d₂ : ℕ) (h₁ : total_work m₁ d₁ = total_work m₂ d₂) (m₁_eq : m₁ = 20) (d₁_eq : d₁ = 6) 
(h₂ : m₂ = 40) : d₂ = 3 :=
  sorry

end wall_building_l700_700598


namespace ranges_of_m_l700_700948

-- Definitions of propositions
def p (m : ℝ) : Prop := (m + 2) * (m - 3) > 0
def q (m : ℝ) : Prop := ∀ x : ℝ, ¬ (mx^2 + (m + 3)x + 4 = 0 ∧ x > 0)

def final_range (m : ℝ) : Prop := (m < -2) ∨ (0 ≤ m ∧ m ≤ 3)

theorem ranges_of_m (m : ℝ) (hpq1 : p m ∨ q m) (hpq2 : ¬ (p m ∧ q m)) :
  final_range m := by
  sorry

end ranges_of_m_l700_700948


namespace perception_permutations_count_l700_700829

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l700_700829


namespace equation_1_l700_700268

theorem equation_1 (x : ℝ) : 2 * x - 1 = 5 * x + 2 → x = -1 :=
by {
  intro h,
  sorry
}

end equation_1_l700_700268


namespace negation_proposition_l700_700155

theorem negation_proposition :
  (∀ x : ℝ, 3^x > 0) ↔ ¬ (∃ x : ℝ, 3^x ≤ 0) :=
by sorry

end negation_proposition_l700_700155


namespace square_field_area_l700_700851

/-- 
  Statement: Prove that the area of the square field is 69696 square meters 
  given that the wire goes around the square field 15 times and the total 
  length of the wire is 15840 meters.
-/
theorem square_field_area (rounds : ℕ) (total_length : ℕ) (area : ℕ) 
  (h1 : rounds = 15) (h2 : total_length = 15840) : 
  area = 69696 := 
by 
  sorry

end square_field_area_l700_700851


namespace triangle_area_5_5_6_l700_700427

theorem triangle_area_5_5_6 : 
  ∀ (a b c : ℝ), a = 5 ∧ b = 5 ∧ c = 6 → let s := (a + b + c) / 2 in
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c)) in
  area = 12 :=
by
  intros a b c h
  let s := (a + b + c) / 2
  let area := Real.sqrt (s * (s - a) * (s - b) * (s - c))
  have ha : a = 5 := h.1
  have hb : b = 5 := h.2.1
  have hc : c = 6 := h.2.2
  sorry

end triangle_area_5_5_6_l700_700427


namespace george_final_score_l700_700841

-- Definitions for points in the first half
def first_half_odd_points (questions : Nat) := 5 * 2
def first_half_even_points (questions : Nat) := 5 * 4
def first_half_bonus_points (questions : Nat) := 3 * 5
def first_half_points := first_half_odd_points 5 + first_half_even_points 5 + first_half_bonus_points 3

-- Definitions for points in the second half
def second_half_odd_points (questions : Nat) := 6 * 3
def second_half_even_points (questions : Nat) := 6 * 5
def second_half_bonus_points (questions : Nat) := 4 * 5
def second_half_points := second_half_odd_points 6 + second_half_even_points 6 + second_half_bonus_points 4

-- Definition of the total points
def total_points := first_half_points + second_half_points

-- The theorem statement to prove the total points
theorem george_final_score : total_points = 113 := by
  unfold total_points
  unfold first_half_points
  unfold second_half_points
  unfold first_half_odd_points first_half_even_points first_half_bonus_points
  unfold second_half_odd_points second_half_even_points second_half_bonus_points
  sorry

end george_final_score_l700_700841


namespace floor_eq_solution_l700_700059

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l700_700059


namespace find_D_l700_700128

noncomputable def point := (ℝ × ℝ)

def vector_add (u v : point) : point := (u.1 + v.1, u.2 + v.2)
def vector_sub (u v : point) : point := (u.1 - v.1, u.2 - v.2)
def scalar_multiplication (k : ℝ) (u : point) : point := (k * u.1, k * u.2)

namespace GeometryProblem

def A : point := (2, 3)
def B : point := (-1, 5)

def D : point := 
  let AB := vector_sub B A
  vector_add A (scalar_multiplication 3 AB)

theorem find_D : D = (-7, 9) := by
  sorry

end GeometryProblem

end find_D_l700_700128


namespace characteristic_value_isosceles_l700_700805

-- Define the properties of the isosceles triangle
def isosceles_triangle (A B C : Type) :=
  ∀ (AB AC : ℝ) (BC : ℝ),
    AB = 18 ∧ AC = 18 ∧ AB + AC + BC = 100

-- Define the characteristic value k of an isosceles triangle
def characteristic_value (base height : ℝ) : ℝ :=
  base / height

-- Formalize the proof context
theorem characteristic_value_isosceles (A B C : Type) :
  ∃ k, isosceles_triangle A B C → (k = 9/20) :=
begin
  sorry,
end

end characteristic_value_isosceles_l700_700805


namespace floor_and_ceiling_sum_l700_700033

theorem floor_and_ceiling_sum :
  (Int.floor 1.999 = 1) ∧ (Int.ceil 3.001 = 4) → (Int.floor 1.999 + Int.ceil 3.001 = 5) :=
by
  intro h
  cases h with h_floor h_ceil
  rw [h_floor, h_ceil]
  rfl

end floor_and_ceiling_sum_l700_700033


namespace factorize_x_squared_minus_four_l700_700448

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l700_700448


namespace tim_total_expenditure_l700_700432

def apple_price : ℕ := 1
def milk_price : ℕ := 3
def pineapple_price : ℕ := 4
def flour_price : ℕ := 6
def chocolate_price : ℕ := 10

def apple_quantity : ℕ := 8
def milk_quantity : ℕ := 4
def pineapple_quantity : ℕ := 3
def flour_quantity : ℕ := 3
def chocolate_quantity : ℕ := 1

def discounted_pineapple_price : ℕ := pineapple_price / 2
def discounted_milk_price : ℕ := milk_price - 1
def coupon_discount : ℕ := 10
def discount_threshold : ℕ := 50

def total_cost_before_coupon : ℕ :=
  (apple_quantity * apple_price) +
  (milk_quantity * discounted_milk_price) +
  (pineapple_quantity * discounted_pineapple_price) +
  (flour_quantity * flour_price) +
  chocolate_price

def final_price : ℕ :=
  if total_cost_before_coupon >= discount_threshold
  then total_cost_before_coupon - coupon_discount
  else total_cost_before_coupon

theorem tim_total_expenditure : final_price = 40 := by
  sorry

end tim_total_expenditure_l700_700432


namespace total_employees_in_buses_l700_700690

/-- The capacities and filled percentages of the buses in the problem. -/
def bus1_capacity : ℕ := 120
def bus2_capacity : ℕ := 150
def bus3_capacity : ℕ := 180

def bus1_filled_percent : ℚ := 55 / 100
def bus2_filled_percent : ℚ := 65 / 100
def bus3_filled_percent : ℚ := 80 / 100

/-- Calculations from the problem to find the number of employees. -/
def employees_bus1 := (bus1_filled_percent * bus1_capacity : ℚ).toNat
def employees_bus2 := (bus2_filled_percent * bus2_capacity : ℚ).toNat
def employees_bus3 := (bus3_filled_percent * bus3_capacity : ℚ).toNat

/-- The total number of employees in the three buses combined is 307. -/
theorem total_employees_in_buses : employees_bus1 + employees_bus2 + employees_bus3 = 307 :=
by
  /- placeholder for the proof which we skip -/
  sorry

end total_employees_in_buses_l700_700690


namespace value_of_x_squared_plus_y_squared_l700_700556

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l700_700556


namespace problem_equivalent_proof_l700_700791

theorem problem_equivalent_proof :
    (¬∀ a : ℝ, ∃ b : ℂ, b = a * complex.I)
    ∧ (¬∀ z w : ℂ, (z * w).im = 0 → (z * w).re = (z * w))
    ∧ ((∀ (x y : ℂ), x = 1 ∧ y = 1 → (x + y * complex.I = 1 + complex.I)) ∧ (¬(∀ (x y : ℂ), x = 1 ∧ y = 1 → (x + y * complex.I = 1 + complex.I))))
    ∧ (¬(0 > (-complex.I)))
:= sorry

end problem_equivalent_proof_l700_700791


namespace x0_range_l700_700888

noncomputable def range_of_x0 (n : ℕ) (h_n : 0 < n) (a b : ℝ) (h_b : 0 < b) (x : Fin n → ℝ)
  (h_sum : (∑ i:Fin n, x i) = a)
  (h_sumsq : (∑ i:Fin n, (x i)^2) = b) : Set ℝ :=
  if a^2 < (n+1) * b then
    {x_0 | (a - sqrt (n * (n+1) * b - a^2)) / (n+1) ≤ x_0 ∧ x_0 ≤ (a + sqrt (n * (n+1) * b - a^2)) / (n+1)}
  else if a^2 = (n+1) * b then
    {x_0 | x_0 = a / (n+1)}
  else
    ∅

theorem x0_range (n : ℕ) (h_n : 0 < n) (a b : ℝ) (h_b : 0 < b) (x : Fin n → ℝ)
  (h_sum : (∑ i:Fin n, x i) = a)
  (h_sumsq : (∑ i:Fin n, (x i)^2) = b) :
  ∃ x_0 : ℝ, x_0 ∈ range_of_x0 n h_n a b x h_sum h_sumsq :=
sorry

end x0_range_l700_700888


namespace stools_chopped_up_l700_700631

variable (chairs tables stools : ℕ)
variable (sticks_per_chair sticks_per_table sticks_per_stool : ℕ)
variable (sticks_per_hour hours total_sticks_from_chairs tables_sticks required_sticks : ℕ)

theorem stools_chopped_up (h1 : sticks_per_chair = 6)
                         (h2 : sticks_per_table = 9)
                         (h3 : sticks_per_stool = 2)
                         (h4 : sticks_per_hour = 5)
                         (h5 : chairs = 18)
                         (h6 : tables = 6)
                         (h7 : hours = 34)
                         (h8 : total_sticks_from_chairs = chairs * sticks_per_chair)
                         (h9 : tables_sticks = tables * sticks_per_table)
                         (h10 : required_sticks = hours * sticks_per_hour)
                         (h11 : total_sticks_from_chairs + tables_sticks = 162) :
                         stools = 4 := by
  sorry

end stools_chopped_up_l700_700631


namespace max_value_f_x_l700_700167

theorem max_value_f_x (x : ℝ) (h1 : 0 < x) (h2 : x < 1) :
  ∃ y, y = x * (1 - x) ∧ (∀ z, 0 < z → z < 1 → z * (1 - z) ≤ y) :=
begin
  use 1 / 4,
  split,
  { have h3 : x = 1 / 2, sorry },
  { intros z hz1 hz2,
    have h4 : z = 1 / 2, sorry,
    sorry },
end

end max_value_f_x_l700_700167


namespace range_dot_product_l700_700975

section Hyperbola

variable {a : ℝ} (h_a : a > 0)

def hyperbola (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 = 1

def dot_product (m n : ℝ) : ℝ :=
  m*m + 2*m + n*n

theorem range_dot_product :
  ∃ m ≥ sqrt 3, ∀ n, hyperbola (m * sqrt 3) n → 
  ∀ v : ℝ, v = dot_product m n → 
    v ∈ set.Ici (3 + 2*sqrt 3) :=
by
  sorry

end Hyperbola

end range_dot_product_l700_700975


namespace plan1_maximizes_B_winning_probability_l700_700409

open BigOperators

-- Definitions for the conditions
def prob_A_wins : ℚ := 3/4
def prob_B_wins : ℚ := 1/4

-- Plan 1 probabilities
def prob_B_win_2_0 : ℚ := prob_B_wins^2
def prob_B_win_2_1 : ℚ := (Nat.choose 2 1) * prob_B_wins * prob_A_wins * prob_B_wins
def prob_B_win_plan1 : ℚ := prob_B_win_2_0 + prob_B_win_2_1

-- Plan 2 probabilities
def prob_B_win_3_0 : ℚ := prob_B_wins^3
def prob_B_win_3_1 : ℚ := (Nat.choose 3 1) * prob_B_wins^2 * prob_A_wins * prob_B_wins
def prob_B_win_3_2 : ℚ := (Nat.choose 4 2) * prob_B_wins^2 * prob_A_wins^2 * prob_B_wins
def prob_B_win_plan2 : ℚ := prob_B_win_3_0 + prob_B_win_3_1 + prob_B_win_3_2

-- Theorem statement
theorem plan1_maximizes_B_winning_probability :
  prob_B_win_plan1 > prob_B_win_plan2 :=
by
  sorry

end plan1_maximizes_B_winning_probability_l700_700409


namespace circles_tangent_externally_l700_700351

theorem circles_tangent_externally :
  let C1 := {p : ℝ × ℝ | p.1^2 + p.2^2 = 1},
      C2 := {p : ℝ × ℝ | (p.1 - 2)^2 + p.2^2 = 1} in
  (∀ (x y : ℝ), ((x, y) ∈ C1 ↔ x^2 + y^2 = 1) ∧ ((x, y) ∈ C2 ↔ (x - 2)^2 + y^2 = 1)) →
  (∃ p, (p ∈ C1 ∧ p ∈ C2) →
    let centerC1 := (0, 0),
        radiusC1 := 1,
        centerC2 := (2, 0),
        radiusC2 := 1 in
    dist centerC1 centerC2 = radiusC1 + radiusC2) :=
sorry

end circles_tangent_externally_l700_700351


namespace triangle_obtuse_l700_700516

theorem triangle_obtuse 
    (a : ℝ) 
    (h : a > 0) 
    (sides_geo_prog : ∀ (b c : ℝ), b = a * √2 ∧ c = 2 * a) 
    (triangle_inequality : ∀ (b c : ℝ), (a + b > c) ∧ (a + c > b) ∧ (b + c > a)) :
    ∃ (θ : ℝ), θ > π / 2 ∧ θ < π ∧ ∃ (b c : ℝ), cos θ = (a^2 + b^2 - c^2) / (2 * a * b)
    sorry

end triangle_obtuse_l700_700516


namespace incorrect_option_A_l700_700895

variables {n : ℕ} {x y : Fin n → ℝ} {b a : ℝ}

def regression_line (x y : Fin n → ℝ) (b a : ℝ) : Fin n → ℝ := 
  fun i => b * x i + a

def mean (f : Fin n → ℝ) : ℝ := 
  (Fin.sum_univ n f) / n

def center_of_sample (x y : Fin n → ℝ) : ℝ × ℝ := 
  (mean x, mean y)

theorem incorrect_option_A (x y : Fin n → ℝ) (b a : ℝ) :
  ¬(∃ i, y i = regression_line x y b a (Fin.mk i sorry)) := 
sorry

end incorrect_option_A_l700_700895


namespace part_a_commutator_properties_part_b_ring_commutative_l700_700355

section
open_locale classical

variables (R : Type*) [ring R] (f : R → R)
variables [is_surjective f] [is_endomorphism f]

def commutator (a b : R) := a * b - b * a

theorem part_a_commutator_properties (x y : R) 
(h1 : ∀ x, commutator x (f x) = 0) 
(h2 : ∀ x, f (x + x) = f x + f x) :

(commutator x (f y) = commutator (f x) y) ∧ (x * commutator x y = f x * commutator x y) :=
sorry

variables (R : Type*) [division_ring R] [is_ring_with_faithful_module R]
variables (f : R → R) [is_surjective f] [is_endomorphism f] (hf : ∀ x, commutator x (f x) = 0) (h : ¬ ∀ x, f x = x)

theorem part_b_ring_commutative (hf : ∀ x, commutator x (f x) = 0) (h : f ≠ id) : 
  ∀ a b : R, a * b = b * a :=
sorry
end

end part_a_commutator_properties_part_b_ring_commutative_l700_700355


namespace smallest_five_digit_divisible_by_72_and_11_l700_700752

noncomputable def reverse_digits (n : ℕ) : ℕ :=
  -- reverse digits function
  sorry

theorem smallest_five_digit_divisible_by_72_and_11 :
  ∃ p : ℕ, 10000 ≤ p ∧ p ≤ 99999 ∧
  p % 72 = 0 ∧ reverse_digits p % 72 = 0 ∧ p % 11 = 0 ∧
  ∀ (q : ℕ), 10000 ≤ q ∧ q ≤ 99999 ∧ q % 72 = 0 ∧ reverse_digits q % 72 = 0 ∧ q % 11 = 0 → p ≤ q :=
  begin
    use 80001,
    split,
    linarith,
    split,
    linarith,
    split,
    norm_num,
    split,
    sorry,
    split,
    norm_num,
    -- To show it is the smallest such p, remaining cases can be handled.
    sorry,
  end

end smallest_five_digit_divisible_by_72_and_11_l700_700752


namespace positive_even_zero_count_l700_700608

variables (a b c : ℝ) (f : ℝ → ℝ)

noncomputable def zero_count_between (a b c : ℝ) (f : ℝ → ℝ) : ℕ :=
sorry -- Placeholder for the actual counting function implementation

theorem positive_even_zero_count :
  a < b ∧ b < c ∧ (f a * f b < 0) ∧ (f b * f c < 0) ∧ continuous f →
  ∃ n : ℕ, n > 0 ∧ even n ∧ zero_count_between a c f = n :=
sorry

end positive_even_zero_count_l700_700608


namespace student_chose_number_l700_700759

theorem student_chose_number (x : ℕ) (h : 2 * x - 138 = 112) : x = 125 :=
by
  sorry

end student_chose_number_l700_700759


namespace ceil_plus_one_l700_700861

def ceil (x : ℝ) : ℤ := ⌈x⌉

theorem ceil_plus_one (x : ℝ) : ceil (x + 1) = ceil x + 1 :=
by
  sorry

end ceil_plus_one_l700_700861


namespace polygon_sides_given_ratio_l700_700918

theorem polygon_sides_given_ratio (n : ℕ) 
  (h : (n - 2) * 180 / 360 = 9 / 2) : n = 11 :=
sorry

end polygon_sides_given_ratio_l700_700918


namespace num_ways_to_arrange_PERCEPTION_l700_700831

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l700_700831


namespace solution_set_l700_700068

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l700_700068


namespace area_triangle_PFO_l700_700914

-- Define the conditions
variable {P : ℝ × ℝ}
variable hP : P.1 * P.1 + P.2 * P.2 = 8 * P.1
variable hPF : dist P (4, 0) = 4

-- Define the point O and the focus F
def O : ℝ × ℝ := (0, 0)
def F : ℝ × ℝ := (4, 0)

-- State the theorem
theorem area_triangle_PFO : 
  let S := 1/2 * (abs (O.1 - F.1)) * (abs P.2) in
  S = 4 :=
sorry

end area_triangle_PFO_l700_700914


namespace contribution_per_employee_l700_700246

theorem contribution_per_employee : 
  ∀ (total_cost boss_contribution : ℝ) (todd_multiplier employees_count remaining_to_pay each_share : ℝ), 
  total_cost = 100 → 
  boss_contribution = 15 → 
  todd_multiplier = 2 → 
  employees_count = 5 → 
  todd_contribution = boss_contribution * todd_multiplier →
  remaining_to_pay = total_cost - todd_contribution - boss_contribution →
  each_share = remaining_to_pay / employees_count →
  each_share = 11 :=
begin
  intros total_cost boss_contribution todd_multiplier employees_count remaining_to_pay each_share,
  intro h_total_cost,
  intro h_boss_contribution,
  intro h_todd_multiplier,
  intro h_employees_count,
  intro h_todd_contribution,
  intro h_remaining_to_pay,
  intro h_each_share,
  simp at *,
  sorry
end

end contribution_per_employee_l700_700246


namespace count_permutations_perception_l700_700811

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l700_700811


namespace problem1_problem2_l700_700726

-- Problem 1
theorem problem1 : 4 * real.sin (real.pi / 3) + (1/3)⁻¹ + |(-2)| - real.sqrt 12 = 5 :=
by
  sorry

-- Problem 2
theorem problem2 (x y : ℝ) : (x - 2*y)^2 - x*(x - 4*y) = 4*y^2 :=
by
  sorry

end problem1_problem2_l700_700726


namespace smallest_x_abs_eq_15_l700_700856

theorem smallest_x_abs_eq_15 :
  ∃ x : ℝ, (|x - 8| = 15) ∧ ∀ y : ℝ, (|y - 8| = 15) → y ≥ x :=
sorry

end smallest_x_abs_eq_15_l700_700856


namespace find_f_l700_700502

noncomputable def f : ℝ → ℝ :=
λ x, Real.sin (4 * x + 5 * Real.pi / 6)

theorem find_f (ω : ℝ) (φ : ℝ) (x : ℝ) (h1 : ω > 0) (h2 : 0 < φ) (h3 : φ < Real.pi) 
               (h4 : f (Real.pi / 24) = 0) (h5 : ∀ x, f (Real.pi / 6 - x) = f (Real.pi / 6 + x)) 
               (h6 : ∀ x ∈ Ioo (Real.pi / 6) (Real.pi / 3), f.deriv x > 0) :
  f = λ x, Real.sin (4 * x + 5 * Real.pi / 6) :=
by
  sorry

end find_f_l700_700502


namespace value_is_correct_l700_700562

def certain_number : ℕ := 52
def value : ℕ := 5 * certain_number - 28

theorem value_is_correct : value = 232 := 
by
  unfold value certain_number
  calc
  5 * 52 - 28 = 260 - 28 := by norm_num
            ... = 232        := by norm_num

end value_is_correct_l700_700562


namespace plane_equation_through_points_perpendicular_l700_700455

theorem plane_equation_through_points_perpendicular {M N : ℝ × ℝ × ℝ} (hM : M = (2, -1, 4)) (hN : N = (3, 2, -1)) :
  ∃ A B C d : ℝ, (∀ x y z : ℝ, A * x + B * y + C * z + d = 0 ↔ (x, y, z) = M ∨ (x, y, z) = N ∧ A + B + C = 0) ∧
  (4, -3, -1, -7) = (A, B, C, d) := 
sorry

end plane_equation_through_points_perpendicular_l700_700455


namespace least_three_digit_7_heavy_l700_700772

/-- A number is 7-heavy if the remainder when the number is divided by 7 is greater than 4. -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The statement to be proved: The least three-digit 7-heavy number is 104. -/
theorem least_three_digit_7_heavy : ∃ n, 100 ≤ n ∧ n < 1000 ∧ is_7_heavy(n) ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ is_7_heavy(m) → n ≤ m :=
begin
    use 104,
    split,
    { exact dec_trivial, },
    split,
    { exact dec_trivial, },
    split,
    { change 104 % 7 > 4,
      exact dec_trivial, },
    { intros m h1 h2,
      sorry
    }
end

end least_three_digit_7_heavy_l700_700772


namespace gcd_poly_l700_700130

theorem gcd_poly (b : ℤ) (h : ∃ k : ℤ, b = 17 * (2 * k + 1)) : 
  Int.gcd (4 * b ^ 2 + 63 * b + 144) (2 * b + 7) = 1 := 
by 
  sorry

end gcd_poly_l700_700130


namespace parallel_lines_slope_equality_l700_700990

theorem parallel_lines_slope_equality (m : ℝ) : (∀ x y : ℝ, 3 * x + y - 3 = 0) ∧ (∀ x y : ℝ, 6 * x + m * y + 1 = 0) → m = 2 :=
by 
  sorry

end parallel_lines_slope_equality_l700_700990


namespace number_of_pencils_l700_700971

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l700_700971


namespace prime_digit_one_l700_700876

theorem prime_digit_one (p q r s : ℕ) (h1 : p > 10) (h2 : nat.prime p) (h3 : nat.prime q) (h4 : nat.prime r) (h5 : nat.prime s) 
                        (h6 : p < q) (h7 : q < r) (h8 : r < s) (h9 : q = p + 10) (h10 : r = q + 10) (h11 : s = r + 10) :
  (p % 10) = 1 := 
sorry

end prime_digit_one_l700_700876


namespace total_age_difference_is_twelve_l700_700682

variable {A B C : ℕ}

theorem total_age_difference_is_twelve (h1 : A + B > B + C) (h2 : C = A - 12) :
  (A + B) - (B + C) = 12 :=
by
  sorry

end total_age_difference_is_twelve_l700_700682


namespace rhombus_perimeter_l700_700288

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 24) (h_d2 : d2 = 16) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side = 16 * Real.sqrt 13 :=
by
  sorry

end rhombus_perimeter_l700_700288


namespace num_ways_to_convert_20d_l700_700611

theorem num_ways_to_convert_20d (n d q : ℕ) (h : 5 * n + 10 * d + 25 * q = 2000) (hn : n ≥ 2) (hq : q ≥ 1) :
    ∃ k : ℕ, k = 130 := sorry

end num_ways_to_convert_20d_l700_700611


namespace trader_sells_cloth_l700_700762

theorem trader_sells_cloth
  (total_SP : ℝ := 4950)
  (profit_per_meter : ℝ := 15)
  (cost_price_per_meter : ℝ := 51)
  (SP_per_meter : ℝ := cost_price_per_meter + profit_per_meter)
  (x : ℝ := total_SP / SP_per_meter) :
  x = 75 :=
by
  sorry

end trader_sells_cloth_l700_700762


namespace largest_angle_isosceles_triangle_l700_700576

theorem largest_angle_isosceles_triangle (A B C : ℕ) 
  (h_isosceles : A = B) 
  (h_base_angle : A = 50) : 
  max A (max B C) = 80 := 
by 
  -- proof is omitted  
  sorry

end largest_angle_isosceles_triangle_l700_700576


namespace standard_deviation_of_data_set_l700_700381

-- Given the average condition for the data set
def average_condition (a : ℝ) : Prop :=
  (11 + 13 + 15 + a + 19) / 5 = 15

-- The statement that we need to prove
theorem standard_deviation_of_data_set : 
  ∃ a : ℝ, average_condition a ∧ real.sqrt (1 / 5 * ( (11 - 15)^2 + (13 - 15)^2 + (15 - 15)^2 + (a - 15)^2 + (19 - 15)^2 )) = 2 * real.sqrt 2 :=
by
  sorry

end standard_deviation_of_data_set_l700_700381


namespace largest_number_of_square_plots_l700_700751

/-- A rectangular field measures 30 meters by 60 meters with 2268 meters of internal fencing to partition into congruent, square plots. The entire field must be partitioned with sides of squares parallel to the edges. Prove the largest number of square plots is 722. -/
theorem largest_number_of_square_plots (s n : ℕ) (h_length : 60 = n * s) (h_width : 30 = s * 2 * n) (h_fence : 120 * n - 90 ≤ 2268) :
(s * 2 * n) = 722 :=
sorry

end largest_number_of_square_plots_l700_700751


namespace num_ways_to_arrange_PERCEPTION_l700_700832

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l700_700832


namespace mike_hours_per_day_l700_700243

theorem mike_hours_per_day (total_hours : ℕ) (total_days : ℕ) (h_total_hours : total_hours = 15) (h_total_days : total_days = 5) : (total_hours / total_days) = 3 := by
  sorry

end mike_hours_per_day_l700_700243


namespace hexagram_arrangement_count_l700_700600

theorem hexagram_arrangement_count : 
  (Nat.factorial 12 / 12) = 39916800 := by
  sorry

end hexagram_arrangement_count_l700_700600


namespace consecutive_integer_min_values_l700_700478

theorem consecutive_integer_min_values (a b : ℝ) 
  (consec : b = a + 1) 
  (min_a : a ≤ real.sqrt 30) 
  (min_b : b ≥ real.sqrt 30) : 
  2 * a - b = 4 := 
sorry

end consecutive_integer_min_values_l700_700478


namespace divide_BC_into_three_equal_parts_l700_700346

variables {A B C K L : Point}
variables (triangle_ABC_equilateral : EquilateralTriangle A B C)
variables (semicircle_on_BC : Semicircle B C (diameter := Segment B C))
variables (arc_KL_divides_semicircle_equally : divides_into_equal_arcs semicircle_on_BC [K, L] 3)
variables (P Q : Point)
variables (intersect_AK_BC : Line A K ∩ Line B C = P)
variables (intersect_AL_BC : Line A L ∩ Line B C = Q)

theorem divide_BC_into_three_equal_parts :
  (segment_length B P = segment_length B Q) ∧ (segment_length B Q = segment_length Q C) ∧
  (segment_length P Q = segment_length Q C) :=
sorry

end divide_BC_into_three_equal_parts_l700_700346


namespace sum_of_digits_of_square_l700_700703

theorem sum_of_digits_of_square (X : ℕ) (h : X = 1111111) : 
  (nat.digits 10 (X * X)).sum = 49 := by
  sorry

end sum_of_digits_of_square_l700_700703


namespace nobel_prize_laureates_l700_700729

variable (Scientists : Type) -- consider the type of scientists
variables (attended : Scientists → Prop)
variables (wolfLaureate : Scientists → Prop)
variables (nobelLaureate : Scientists → Prop)

def attendedScientists := {s : Scientists | attended s}
def wolfPrizeLaureates := {s : Scientists | wolfLaureate s}
def nobelPrizeLaureates := {s : Scientists | nobelLaureate s}

noncomputable def workshop : set Scientists := attendedScientists

axiom H1 : ∃ S : set Scientists, S ⊆ workshop ∧ S.card = 31
axiom H2 : ∃ S : set Scientists, S ⊆ wolfPrizeLaureates ∧ S ⊆ nobelPrizeLaureates ∧ S.card = 12
axiom H3 : ∃ W N M : ℕ, W = (50 - 31) ∧ N = M + 3 ∧ W = N + M ∧ W.card = 19
axiom H4 : workshop.card = 50

theorem nobel_prize_laureates : ∃ K, K = 23 :=
by
  sorry

end nobel_prize_laureates_l700_700729


namespace problem_conditions_l700_700929

noncomputable def f (ω x : ℝ) : ℝ :=
  √3 * Real.sin (2 * ω * x) + 2 * Real.cos (ω * x) ^ 2

theorem problem_conditions (ω : ℝ) :
  (∀ x₀ x₁, f ω x₀ = f ω x₁ → abs (x₀ - x₁) = (π / 2)) →
  ω = 1 ∧ 
  (∀ k : ℤ, ∀ x, (π / 6) + ↑k * π ≤ x ∧ x ≤ (2 * π / 3) + ↑k * π → 
     (Real.sin (2 * x + π / 6) is decreasing)) ∧ 
  (∀ m, (∃ x, -π / 4 < x ∧ x < π / 4 ∧ f 1 x = m) ↔ 1 - √3 < m ∧ m ≤ 2) :=
by
  sorry

end problem_conditions_l700_700929


namespace equal_triangle_areas_l700_700227

theorem equal_triangle_areas 
  (ABCD : Square)
  (ω : Circle)
  (P : ω.Point)
  (P_on_shorter_arc_AB : P ∈ shorter_arc AB ω)
  (R : Point)
  (S : Point)
  (CP_inter_BD : line_intersection (line_through C P) (line_through B D) = R)
  (DP_inter_AC : line_intersection (line_through D P) (line_through A C) = S) :
  triangle_area A R B = triangle_area D S R :=
by
  sorry

end equal_triangle_areas_l700_700227


namespace triangle_inequality_l700_700764

theorem triangle_inequality (a b c : ℝ) (h : a + b + c = 1) : a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
sorry

end triangle_inequality_l700_700764


namespace distance_CD_of_ellipse_l700_700016

-- We define the equation of the ellipse and the problem statement
theorem distance_CD_of_ellipse :
  let C := (0, 4)  -- endpoint of the major axis
  let D := (2, 0)  -- endpoint of the minor axis
  ∀ (x y : ℝ), 4 * (x + 2)^2 + 16 * y^2 = 64 →
  dist (0, 4) (2, 0) = 2 * real.sqrt 5 :=
begin
  intros x y hyp,
  sorry,
end

end distance_CD_of_ellipse_l700_700016


namespace no_partition_exists_l700_700431

theorem no_partition_exists : ¬ ∃ n : ℕ, n > 1 ∧ ∀ (partition : Fin n → Set ℕ), 
  (∀ i, partition i ≠ ∅) ∧ (∀ (S : Fin n) (t : Finset (Fin n)), 
  t.card = n - 1 → S ∉ t → let ssum := (t.attach.map (fun t' => Classical.choose (∃ z ∈ partition t', True))).sum in 
  ssum ∈ partition S) :=
sorry

end no_partition_exists_l700_700431


namespace common_ratio_of_geometric_series_l700_700515

theorem common_ratio_of_geometric_series 
  (a1 q : ℝ) 
  (h1 : a1 + a1 * q^2 = 5) 
  (h2 : a1 * q + a1 * q^3 = 10) : 
  q = 2 := 
by 
  sorry

end common_ratio_of_geometric_series_l700_700515


namespace scheduling_ways_l700_700323

-- Definitions of the courses
def courses : List String := ["rituals", "music", "archery", "charioteering", "calligraphy", "mathematics"]

-- Ensure "rituals," "archery," and "charioteering" are not scheduled on three consecutive days.
def not_consecutive_three (sched : List String) : Prop :=
  ∀ i, 0 ≤ i ∧ i ≤ 3 → ¬(sched[i] = "rituals" ∧ sched[i+1] = "archery" ∧ sched[i+2] = "charioteering")

theorem scheduling_ways :
  ∃ scheds : List (List String), 
    (∀ sched ∈ scheds, sched.perm courses ∧ not_consecutive_three sched) ∧ 
    scheds.length = 576 := sorry

end scheduling_ways_l700_700323


namespace hyperbola_equation_exists_line_equation_exists_l700_700154

-- Define the hyperbola and its conditions
def hyperbola_eq (x y : ℝ) (a b : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ x^2 / a^2 - y^2 / b^2 = 1

-- Define the first goal: equation of the hyperbola
theorem hyperbola_equation_exists (a b : ℝ) (x y : ℝ) (Px Py : ℝ) (PF1_x PF1_y PF2_x PF2_y : ℝ) (h : hyperbola_eq x y a b) (hF1 : PF1_x = -2 ∧ PF1_y = 0) (hF2 : PF2_x = 2 ∧ PF2_y = 0) (hP : (Px, Py) = (3, sqrt 7) ∧ hyperbola_eq Px Py a b):
  (a^2 = 2 ∧ b^2 = 2) → hyperbola_eq x y (sqrt 2) (sqrt 2):= 
by sorry

-- Define the second goal: equation of the line given area of triangle
def line_eq (y x k : ℝ) : Prop :=
  y = k * x + 2

theorem line_equation_exists (k : ℝ) (x1 y1 x2 y2 : ℝ) (area : ℝ) (hE : line_eq y1 x1 k) (hF : line_eq y2 x2 k) (h_area : area = 2*sqrt 2 ∧ (1 - k^2) * x^2 - 4 * k * x - 6 = 0):
  k = sqrt 2 ∨ k = -sqrt 2 ∧ (line_eq y x (sqrt 2) ∨ line_eq y x (-sqrt 2)) := 
by sorry

end hyperbola_equation_exists_line_equation_exists_l700_700154


namespace log7_48_eq_a_plus_2b_l700_700403

variable (a b : ℝ)

-- Given conditions
axiom log7_3_eq_a : log 7 3 = a
axiom log7_4_eq_b : log 7 4 = b

-- Mathematical statement to prove
theorem log7_48_eq_a_plus_2b : log 7 48 = a + 2 * b :=
by
  sorry

end log7_48_eq_a_plus_2b_l700_700403


namespace Norine_retire_age_l700_700247

theorem Norine_retire_age:
  ∀ (A W : ℕ),
    (A = 50) →
    (W = 19) →
    (A + W = 85) →
    (A = 50 + 8) :=
by
  intros A W hA hW hAW
  sorry

end Norine_retire_age_l700_700247


namespace mutually_exclusive_A_B_head_l700_700262

variables (A_head B_head B_end : Prop)

def mut_exclusive (P Q : Prop) : Prop := ¬(P ∧ Q)

theorem mutually_exclusive_A_B_head (A_head B_head : Prop) :
  mut_exclusive A_head B_head :=
sorry

end mutually_exclusive_A_B_head_l700_700262


namespace distance_between_centers_of_two_circles_l700_700140

theorem distance_between_centers_of_two_circles
  (r : ℝ) (hR : r = 2)
  (d : ℝ) (hD : d = 2)
  (OE : ℝ) (hOE : OE = sqrt (r ^ 2 - (d / 2) ^ 2)) :
  OE = sqrt 3 := by
sorry

end distance_between_centers_of_two_circles_l700_700140


namespace calculate_prob_X2_calculate_prob_X4_A_wins_l700_700572

variables (A B : Type) [ProbabilitySpace A] [ProbabilitySpace B]
variable p1 : ℙ(λ _, true) = 0.5 -- Probability of A scoring when serving
variable p2 : ℙ(λ _, true) = 0.4 -- Probability of A scoring when B is serving

def point_independence (n : ℕ) : Prop :=
  ∀ (a1 a2 : fin n → A) (b1 b2 : fin n → B), ℙ(λ _, true) = ℙ(λ _, true)

theorem calculate_prob_X2 (P : ProbabilitySpace ℕ) (A1 A2 : Event P) :
  (P (A1 ∩ A2) + P (¬A1 ∩ ¬A2)) = P A1 * P A2 + P (¬A1) * P (¬A2) :=
sorry

theorem calculate_prob_X4_A_wins (P : ProbabilitySpace ℕ) (A1 A2 A3 A4 : Event P) :
  (P (¬A1 ∩ A2 ∩ A3 ∩ A4) + P (A1 ∩ ¬ A2 ∩ A3 ∩ A4)) = 
  (P (¬A1)) * P (A2) * P (A3) * P (A4) + P (A1) * P (¬A2) * P (A3) * P (A4) :=
sorry

end calculate_prob_X2_calculate_prob_X4_A_wins_l700_700572


namespace domain_f_parity_f_range_f_lt_zero_l700_700523

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := log a (1 - x) - log a (1 + x)

-- Function Domain
theorem domain_f {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, f x a = f x a → -1 < x ∧ x < 1 :=
sorry

-- Function Parity
theorem parity_f {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, f (-x) a = -f x a :=
sorry

-- Range of x for which f(x) < 0 when a > 1 and 0 < a < 1
theorem range_f_lt_zero {a : ℝ} (h1 : a > 0) (h2 : a ≠ 1) :
  (∀ x : ℝ, a > 1 → (f x a < 0 ↔ 0 < x ∧ x < 1)) ∧
  (∀ x : ℝ, 0 < a ∧ a < 1 → (f x a < 0 ↔ -1 < x ∧ x < 0)) :=
sorry

end domain_f_parity_f_range_f_lt_zero_l700_700523


namespace february_first_day_is_friday_l700_700997

theorem february_first_day_is_friday 
    (leap_year : ∀ (d : Nat), 1 ≤ d ∧ d ≤ 29 → d ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
                                                     11, 12, 13, 14, 15, 16, 17, 18, 19,
                                                     20, 21, 22, 23, 24, 25, 26, 27, 28, 29}) 
    (four_mondays_and_fridays : ∀ (D : Nat → Nat), 
                                (∀d, 1 ≤ d ∧ d ≤ 29 → D d = 5 ∨ D d = 1 ∨ D d = 2 ∨ 
                                   D d = 3 ∨ D d = 4 ∨ D d = 6 ∨ D d = 7) ∧
                                (∃ m1 m2 m3 m4, {m1, m2, m3, m4} = {1, 8, 15, 22}) ∧
                                (∃ f1 f2 f3 f4, {f1, f2, f3, f4} = {7, 14, 21, 28})) : 
    D 1 = 5 := 
sorry

end february_first_day_is_friday_l700_700997


namespace solve_for_z_l700_700487

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2 + I) : z = (1 / 2) + (3 / 2) * I :=
  sorry

end solve_for_z_l700_700487


namespace pencils_across_diameter_l700_700966

def radius_feet : ℝ := 14
def pencil_length_inches : ℝ := 6

theorem pencils_across_diameter : 
  (2 * radius_feet * 12 / pencil_length_inches) = 56 := 
by
  sorry

end pencils_across_diameter_l700_700966


namespace pencils_across_diameter_l700_700969

theorem pencils_across_diameter (r : ℝ) (pencil_length_inch : ℕ) (pencils : ℕ) :
  r = 14 ∧ pencil_length_inch = 6 ∧ pencils = 56 → 
  let d := 2 * r in
  let pencil_length_feet := pencil_length_inch / 12 in
  pencils = d / pencil_length_feet :=
begin
  sorry -- Proof is skipped
end

end pencils_across_diameter_l700_700969


namespace uniformly_integrable_families_l700_700230

-- Given Definitions
variables {α : Type*} {β : Type*} [MeasureSpace α] [NormedSpace ℝ β] [CompleteSpace β]

-- Uniform Integrability Definition
def uniform_integrable (f : ℕ → α → β) : Prop :=
  ∃ c, ∀ n, ∫ (a : α), (∥ f n a ∥) ∂a < c

-- Stochastic Domination Definition
def stochastic_domination (X Y : α → β) : Prop :=
  ∀ x > 0, ∫ (a : α) (ha : ∥ X a ∥ > x), 1 ∂ha < ∫ (a : α) (ha : ∥ Y a ∥ > x), 1 ∂ha

-- Lean Statement
theorem uniformly_integrable_families
  (xi : ℕ → α → β)
  (hxi : uniform_integrable xi)
  (λ : ℕ → ℝ)
  (hλsum : ∑ n, |λ n| = 1)
  (hL1 : ∃ (f : α → β), has_sum (λ n, λ n • xi n) f) : 
  (uniform_integrable (λ ζ, ∃ n, stochastic_domination ζ (xi n)) ∧
   uniform_integrable (λ y, has_sum (λ n, λ n • xi n) y)) :=
sorry

end uniformly_integrable_families_l700_700230


namespace estimate_production_in_March_l700_700750

theorem estimate_production_in_March 
  (monthly_production : ℕ → ℝ)
  (x y : ℝ)
  (hx : x = 3)
  (hy : y = x + 1) : y = 4 :=
by
  sorry

end estimate_production_in_March_l700_700750


namespace usual_price_of_detergent_l700_700479

theorem usual_price_of_detergent
  (loads_per_bottle : ℕ)
  (cost_per_load : ℝ)
  (sale_price_per_bottle : ℝ)
  (total_loads : ℕ)
  (total_cost : ℝ)
  (total_bottles : ℕ)
  (usual_price : ℝ)
  (h1 : loads_per_bottle = 80)
  (h2 : cost_per_load = 0.25)
  (h3 : sale_price_per_bottle = 20.00)
  (h4 : total_bottles = 2)
  (h5 : total_loads = loads_per_bottle * total_bottles)
  (h6 : total_cost = cost_per_load * total_loads)
  (h7 : h6 = sale_price_per_bottle * total_bottles) :
  usual_price = 20.00 :=
sorry

end usual_price_of_detergent_l700_700479


namespace convex_2015_gon_l700_700485

def point := (ℝ × ℝ)

def is_convex_quadrilateral (a b c d : point) : Prop :=
-- Assume function that checks if four points form a convex quadrilateral
sorry

def no_three_collinear (s : Finset point) : Prop :=
∀ (a b c : point) (ha : a ∈ s) (hb : b ∈ s) (hc : c ∈ s), 
-- Condition for no three points in the set to be collinear
sorry

theorem convex_2015_gon (S : Finset point) (hS_card : S.card = 2015)
  (hconvex_quad : ∀ (a b c d : point), a ∈ S → b ∈ S → c ∈ S → d ∈ S → is_convex_quadrilateral a b c d) :
  -- Prove that these points form the vertices of a convex 2015-gon
  -- Using the convex hull function in Mathlib
  ∃ K : Finset point, K.card = 2015 ∧ ∀ (x y z : point), x ∈ K ∧ y ∈ K ∧ z ∈ K → 
  -- Condition mentioning that all 2015 points form the convex hull (convex 2015-gon)
sorry

end convex_2015_gon_l700_700485


namespace potion_kits_needed_l700_700162

-- Definitions
def num_spellbooks := 5
def cost_spellbook_gold := 5
def cost_potion_kit_silver := 20
def num_owls := 1
def cost_owl_gold := 28
def silver_per_gold := 9
def total_silver := 537

-- Prove that Harry needs to buy 3 potion kits.
def Harry_needs_to_buy : Prop :=
  let cost_spellbooks_silver := num_spellbooks * cost_spellbook_gold * silver_per_gold
  let cost_owl_silver := num_owls * cost_owl_gold * silver_per_gold
  let total_cost_silver := cost_spellbooks_silver + cost_owl_silver
  let remaining_silver := total_silver - total_cost_silver
  let num_potion_kits := remaining_silver / cost_potion_kit_silver
  num_potion_kits = 3

theorem potion_kits_needed : Harry_needs_to_buy :=
  sorry

end potion_kits_needed_l700_700162


namespace range_of_a_for_monotonic_decreasing_l700_700939

theorem range_of_a_for_monotonic_decreasing (a : ℝ) :
(∀ x : ℝ, 0 < x ∧ x < 1 → 2 ^ (x * (x - a)) > 2 ^ (x * (x - a)) → a ∈ set.Ici 2) :=
begin
  sorry
end

end range_of_a_for_monotonic_decreasing_l700_700939


namespace sum_series_l700_700024

def F : ℕ → ℕ
| 0       := 1
| 1       := 4
| (n + 2) := 3 * F (n + 1) - F n

theorem sum_series :
  ∃ S : ℝ, ∑' n, (1 : ℝ) / F (3^n) = S :=
sorry

end sum_series_l700_700024


namespace necessary_but_not_sufficient_l700_700132

variable {R : Type*} [Real R]

-- Definition of a differentiable function
variable (f : R → R)
variable {x : R}
variable (h_diff : Differentiable R f)

-- The Lean 4 statement
theorem necessary_but_not_sufficient (h : deriv f x = 0) : 
  (∀ y, has_deriv_at f (deriv f y) y) → is_extremum f x → deriv f x = 0 :=
by
  -- Proof is not provided, as instructed
  sorry

end necessary_but_not_sufficient_l700_700132


namespace smaller_circle_radius_is_5sqrt3_over_3_l700_700747

noncomputable def radius_of_smaller_circle
  (A1 A2 : ℝ)
  (A1_A2_ap_A1_A2 : A1 + A2 = π * 5 * 5)
  (ap : 2 * A2 = A1 + (A1 + A2)) : ℝ :=
  let r := (A1 / π).sqrt in
  r

theorem smaller_circle_radius_is_5sqrt3_over_3 : 
  ∀ (A1 A2 : ℝ),
  A1 + A2 = π * 5 * 5 →
  (2 * A2 = A1 + (A1 + A2)) →
  (A1 / π).sqrt = 5 / (3.sqrt) := 
by 
  intros A1 A2 A1_A2_ap_A1_A2 ap
  sorry

end smaller_circle_radius_is_5sqrt3_over_3_l700_700747


namespace darrel_has_85_dimes_l700_700795

theorem darrel_has_85_dimes 
  (num_quarters : ℕ := 76)
  (num_nickels : ℕ := 20)
  (num_pennies : ℕ := 150)
  (fee_rate : ℚ := 0.10)
  (received_after_fee : ℚ := 27) :
  ∃ (num_dimes : ℕ), num_dimes = 85 := 
by
  let total_before_fee := received_after_fee / (1 - fee_rate)
  let value_quarters := num_quarters * 0.25
  let value_nickels := num_nickels * 0.05
  let value_pennies := num_pennies * 0.01
  let total_other_coins := value_quarters + value_nickels + value_pennies
  let value_dimes := total_before_fee - total_other_coins
  let num_dimes := value_dimes / 0.10
  use num_dimes
  sorry  -- Proof omitted

end darrel_has_85_dimes_l700_700795


namespace ratio_of_volumes_l700_700394

theorem ratio_of_volumes (C D : ℚ) (h1: C = (3/4) * C) (h2: D = (5/8) * D) : C / D = 5 / 6 :=
sorry

end ratio_of_volumes_l700_700394


namespace total_amount_shared_l700_700803

theorem total_amount_shared (total_amount : ℝ) 
  (h_debby : total_amount * 0.25 = (total_amount - 4500))
  (h_maggie : total_amount * 0.75 = 4500) : total_amount = 6000 :=
begin
  sorry
end

end total_amount_shared_l700_700803


namespace q_mth_root_of_unity_excluding_1_l700_700989

constant q : ℂ
constant m : ℕ

axiom periodic {n : ℕ} : q^(m + n) = q^n
axiom period_m_geq_2 : 2 ≤ m
axiom period_prime_m : Nat.Prime m

theorem q_mth_root_of_unity_excluding_1 : q^m = 1 ∧ q ≠ 1 :=
by sorry

end q_mth_root_of_unity_excluding_1_l700_700989


namespace kiera_fruit_cups_l700_700877

def muffin_cost : ℕ := 2
def fruit_cup_cost : ℕ := 3
def francis_muffins : ℕ := 2
def francis_fruit_cups : ℕ := 2
def kiera_muffins : ℕ := 2
def total_cost : ℕ := 17

theorem kiera_fruit_cups : ∃ kiera_fruit_cups : ℕ, muffin_cost * kiera_muffins + fruit_cup_cost * kiera_fruit_cups = total_cost - (muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups) :=
by
  let francis_cost := muffin_cost * francis_muffins + fruit_cup_cost * francis_fruit_cups
  let remaining_cost := total_cost - francis_cost
  let kiera_fruit_cups := remaining_cost / fruit_cup_cost
  exact ⟨kiera_fruit_cups, by sorry⟩

end kiera_fruit_cups_l700_700877


namespace smallest_positive_period_monotonic_intervals_max_min_values_and_intervals_l700_700520

open Real

noncomputable def f (x : ℝ) : ℝ := sin (π / 3 + 4 * x) + cos (4 * x - π / 6)

theorem smallest_positive_period : ∀ x : ℝ, f (x + π / 2) = f x :=
by
  -- proof to be filled
  sorry

theorem monotonic_intervals :
  ∀ k : ℤ,
    (∀ x ∈ Icc (-5 * π / 24 + k * π / 2) (π / 24 + k * π / 2), f x < f (x + π / 2 / 4)) ∧
    (∀ x ∈ Icc (π / 24 + k * π / 2) (7 * π / 24 + k * π / 2), f x > f (x + π / 2 / 4)) :=
by
  -- proof to be filled
  sorry

theorem max_min_values_and_intervals :
  ∀ x ∈ Icc 0 (π / 4),
    f x ≤ 2 ∧ ∃ y ∈ Icc 0 (π / 4), f y = 2 ∧
    f x ≥ -sqrt 3 ∧ ∃ y ∈ Icc 0 (π / 4), f y = -sqrt 3 :=
by
  -- proof to be filled
  sorry

end smallest_positive_period_monotonic_intervals_max_min_values_and_intervals_l700_700520


namespace problem_l700_700419

theorem problem (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z)
  (coprime : Nat.gcd X (Nat.gcd Y Z) = 1)
  (h : X * Real.log 3 / Real.log 100 + Y * Real.log 4 / Real.log 100 = Z):
  X + Y + Z = 4 :=
sorry

end problem_l700_700419


namespace floor_and_ceil_sum_l700_700037

theorem floor_and_ceil_sum : ⌊1.999⌋ + ⌈3.001⌉ = 5 := 
by
  sorry

end floor_and_ceil_sum_l700_700037


namespace greatest_int_less_than_50_satisfying_conditions_l700_700699

def satisfies_conditions (n : ℕ) : Prop :=
  n < 50 ∧ Int.gcd n 18 = 6

theorem greatest_int_less_than_50_satisfying_conditions :
  ∃ n : ℕ, satisfies_conditions n ∧ ∀ m : ℕ, satisfies_conditions m → m ≤ n ∧ n = 42 :=
by
  sorry

end greatest_int_less_than_50_satisfying_conditions_l700_700699


namespace matrix_power_4_l700_700414

open Matrix

def A : Matrix (fin 2) (fin 2) ℤ :=
  ![![2, -2], ![2, -1]]

theorem matrix_power_4 :
  A ^ 4 = ![![ -8, 8 ], ![ 0, 3 ]] :=
by
  sorry

end matrix_power_4_l700_700414


namespace permutations_PERCEPTION_l700_700823

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l700_700823


namespace range_of_a_to_decreasing_f_l700_700942

noncomputable def f (x a : ℝ) : ℝ := (2 : ℝ)^(x * (x - a))

theorem range_of_a_to_decreasing_f :
  (∀ a x : ℝ, (0 < x ∧ x < 1) → 
    monotone_decreasing (λ x, f x a)) ↔ a ∈ set.Ici 2 := sorry

end range_of_a_to_decreasing_f_l700_700942


namespace count_vectors_l700_700617

theorem count_vectors (S : Finset ℕ) (h1 : S = Finset.range 31 - {0}) :
  (∃ (n : ℕ), n = 90335 ∧
  (@Finset.filter (ℕ × ℕ × ℕ × ℕ)
    (λ p, p.1 < p.4 ∧ p.2 < p.3 ∧ p.3 < p.4)
    (@Finset.product (ℕ × ℕ) (ℕ × ℕ) _
      (@Finset.product ℕ ℕ _ Finset.univ Finset.univ) 
      (@Finset.product ℕ ℕ _ Finset.univ Finset.univ))).card = n) := by
    sorry

end count_vectors_l700_700617


namespace fractional_difference_l700_700428

def recurring72 : ℚ := 72 / 99
def decimal72 : ℚ := 72 / 100

theorem fractional_difference : recurring72 - decimal72 = 2 / 275 := by
  sorry

end fractional_difference_l700_700428


namespace undominated_implies_favorite_toy_l700_700685

-- Define strict preference ordering
def strict_preference {α : Type} (prefs : α → α → Prop) : Prop :=
  ∀ (x y : α), (prefs x y ∨ prefs y x) ∧ ¬ (prefs x x) ∧ ¬ (prefs y y)

-- Define a distribution type
def distribution (α : Type) :=  (Finₓ n) → α

-- Define the dominance of one distribution over another
def dominates {α : Type} (prefs : (Finₓ n) → (α → α → Prop)) (A B : distribution α) : Prop :=
  ∀ (i : Finₓ n), prefs i (A i) (B i) ∨ (A i) = (B i)

theorem undominated_implies_favorite_toy {α : Type} {n : ℕ} 
  (prefs : (Finₓ n) → (α → α → Prop)) 
  (h_prefs : ∀ i, strict_preference (prefs i))
  (D : distribution α) 
  (h_undominated : ¬∃ B ≠ D, dominates prefs B D) :
  ∃ i, ∀ j, prefs i (D i) j := sorry

end undominated_implies_favorite_toy_l700_700685


namespace perception_num_permutations_l700_700816

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l700_700816


namespace floor_and_ceiling_sum_l700_700034

theorem floor_and_ceiling_sum :
  (Int.floor 1.999 = 1) ∧ (Int.ceil 3.001 = 4) → (Int.floor 1.999 + Int.ceil 3.001 = 5) :=
by
  intro h
  cases h with h_floor h_ceil
  rw [h_floor, h_ceil]
  rfl

end floor_and_ceiling_sum_l700_700034


namespace factorize_difference_of_squares_l700_700442

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
begin
  sorry
end

end factorize_difference_of_squares_l700_700442


namespace negation_of_P_l700_700530

-- Define the proposition P
def P : Prop := ∀ x : ℝ, x^2 + 2*x + 2 > 0

-- State the negation of P
theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 2 ≤ 0 :=
by
  sorry

end negation_of_P_l700_700530


namespace only_zero_solution_l700_700256

theorem only_zero_solution (a b c n : ℤ) (h_gcd : Int.gcd (Int.gcd (Int.gcd a b) c) n = 1)
  (h_eq : 6 * (6 * a^2 + 3 * b^2 + c^2) = 5 * n^2) : 
  a = 0 ∧ b = 0 ∧ c = 0 ∧ n = 0 :=
sorry

end only_zero_solution_l700_700256


namespace sum_of_possible_integers_l700_700504

theorem sum_of_possible_integers (n : ℤ) (h : 0 < 3 * n ∧ 3 * n < 27) : 
  (∑ k in {k : ℤ | 0 < k ∧ k < 9}, k) = 36 :=
sorry

end sum_of_possible_integers_l700_700504


namespace relationship_of_a_b_c_l700_700889

noncomputable def f (x : ℝ) : ℝ := 
  if - (Real.pi / 2 : ℝ) < x ∧ x < Real.pi / 2 then 
    x + Real.sin x 
  else 
    f (Real.pi - x)

def a : ℝ := f 1
def b : ℝ := f 2
def c : ℝ := f 3

theorem relationship_of_a_b_c : b > a ∧ a > c := by
  sorry

end relationship_of_a_b_c_l700_700889


namespace largest_horizontal_vertical_sum_l700_700843

-- Define the main problem conditions and state the theorem
theorem largest_horizontal_vertical_sum (a b c d e : ℕ)
  (h1 : a ∈ {2, 5, 8, 11, 14})
  (h2 : b ∈ {2, 5, 8, 11, 14}) 
  (h3 : c ∈ {2, 5, 8, 11, 14}) 
  (h4 : d ∈ {2, 5, 8, 11, 14}) 
  (h5 : e ∈ {2, 5, 8, 11, 14})
  (h6 : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h7 : a + b + e = a + c + e)
  (h8 : a + c + e = b + d + e) 
  (h9 : b = c) : a + b + e = 27 := 
sorry

end largest_horizontal_vertical_sum_l700_700843


namespace center_of_symmetry_intervals_of_increase_l700_700148

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin x * cos x - cos x ^ 2 - 1 / 2

theorem center_of_symmetry : ∃ k : ℤ, ∃ c : (ℝ × ℝ), c = ((k * π / 2) + (π / 12), -1) :=
by
  sorry

theorem intervals_of_increase : 
  Set.Icc 0 π = Set.intervalUnion (Set.Icc 0 (π / 3)) (Set.Icc (5 * π / 6) π) :=
by
  sorry

end center_of_symmetry_intervals_of_increase_l700_700148


namespace true_discount_different_time_l700_700172

theorem true_discount_different_time (FV TD_initial TD_different : ℝ) (r : ℝ) (initial_time different_time : ℝ) 
  (h1 : r = initial_time / different_time)
  (h2 : FV = 110)
  (h3 : TD_initial = 10)
  (h4 : initial_time / different_time = 1 / 2) :
  TD_different = 2 * TD_initial :=
by
  sorry

end true_discount_different_time_l700_700172


namespace math_problem_proof_l700_700910

noncomputable def math_problem (x y z : ℝ) : Prop :=
  x >= 0 ∧ y >= 0 ∧ z >= 0 ∧ x + y + z = 1/2 →
  (sqrt x / (4 * x + 1) + sqrt y / (4 * y + 1) + sqrt z / (4 * z + 1) ≤ 3 * sqrt 6 / 10)

-- The statement to be proven
theorem math_problem_proof (x y z : ℝ) : math_problem x y z :=
sorry

end math_problem_proof_l700_700910


namespace smallest_solution_l700_700097

-- Define the floor function
def floor (x : ℝ) : ℤ := Int.floor x

-- Define the fractional part of x
def frac_part (x : ℝ) : ℝ := x - floor x

-- Statement of the problem in Lean 4
theorem smallest_solution (x : ℝ) :
  floor x = 3 + 50 * frac_part x ∧ 0 ≤ frac_part x ∧ frac_part x < 1 → x = 2.94 :=
sorry

end smallest_solution_l700_700097


namespace find_m_l700_700899

variable {S : ℕ → ℤ}
variable {m : ℕ}

/-- Given the sequences conditions, the value of m is 5 --/
theorem find_m (h1 : S (m - 1) = -2) (h2 : S m = 0) (h3 : S (m + 1) = 3) (h4 : 2 ≤ m) : m = 5 :=
sorry

end find_m_l700_700899


namespace cosine_inclination_parametric_eq_l700_700985

theorem cosine_inclination_parametric_eq
  (t : ℝ) (x y : ℝ)
  (h1 : x = 1 + 3 * t)
  (h2 : y = 2 - 4 * t) :
  ∃ θ : ℝ, cos θ = -3 / 5 :=
by
  sorry

end cosine_inclination_parametric_eq_l700_700985


namespace two_integer_solutions_l700_700959

theorem two_integer_solutions :
  {p : ℤ × ℤ // p.1 ^ 4 + p.2 ^ 2 = 4 * p.2}.to_finset.card = 2 :=
by sorry

end two_integer_solutions_l700_700959


namespace total_money_shared_l700_700796

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l700_700796


namespace solve_for_z_l700_700486

theorem solve_for_z (z : ℂ) (h : z * (1 - I) = 2 + I) : z = (1 / 2) + (3 / 2) * I :=
  sorry

end solve_for_z_l700_700486


namespace value_of_x_squared_plus_y_squared_l700_700555

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l700_700555


namespace remainder_of_M_div_51_l700_700015

-- Define M as the concatenation of integers from 1 to 50.
def M : ℕ := -- (123456789101112...4950)

-- Lean doesn't support dynamic concatenation as 1 statement because it's not numerics, skip defining M itself.

theorem remainder_of_M_div_51 :
  M % 51 = 15 :=
by
  sorry -- proof to be done

end remainder_of_M_div_51_l700_700015


namespace rhombus_perimeter_l700_700290

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  let side_length := real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side_length = 16 * real.sqrt 13 :=
by
  let d1_half := d1 / 2
  let d2_half := d2 / 2
  have h3 : d1_half = 12 := by sorry
  have h4 : d2_half = 8 := by sorry
  let side_length := real.sqrt (d1_half ^ 2 + d2_half ^ 2)
  have h5 : side_length = real.sqrt 208 := by sorry
  have h6 : real.sqrt 208 = 4 * real.sqrt 13 := by sorry
  show 4 * side_length = 16 * real.sqrt 13
  from by
    rw [h6]
    rfl

end rhombus_perimeter_l700_700290


namespace distance_between_x_intercepts_l700_700694
open Real

theorem distance_between_x_intercepts : 
  ∃ (x1 : ℝ), ∃ (x2 : ℝ), 
  (let l1 := λ x : ℝ, 4 * x - 28) ∧ 
  (let l2 := λ x : ℝ, 6 * x - 52) ∧ 
  l1 12 = 20 ∧ l2 12 = 20 ∧ 
  (l1 x1 = 0) ∧ (l2 x2 = 0) → 
  |x1 - x2| = 5 / 3 := 
by 
  sorry

end distance_between_x_intercepts_l700_700694


namespace rhombus_perimeter_l700_700287

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 24) (h_d2 : d2 = 16) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side = 16 * Real.sqrt 13 :=
by
  sorry

end rhombus_perimeter_l700_700287


namespace Dirichlet_correct_l700_700111

def Dirichlet_function (x : ℝ) : ℝ := if x ∈ ℚ then 1 else 0

theorem Dirichlet_correct (x : ℝ) : Dirichlet_function (Dirichlet_function x) = 1 :=
by
  -- mathematical proof goes here
  sorry

end Dirichlet_correct_l700_700111


namespace range_of_a_l700_700979

theorem range_of_a (a : ℝ) : (∀ x : ℝ, (1 < x ∧ x < 4) → 2 * x^2 - 8 * x - 4 - a > 0) ↔ a < -4 :=
begin
  sorry
end

end range_of_a_l700_700979


namespace problem_conditions_l700_700928

noncomputable def f (ω x : ℝ) : ℝ :=
  √3 * Real.sin (2 * ω * x) + 2 * Real.cos (ω * x) ^ 2

theorem problem_conditions (ω : ℝ) :
  (∀ x₀ x₁, f ω x₀ = f ω x₁ → abs (x₀ - x₁) = (π / 2)) →
  ω = 1 ∧ 
  (∀ k : ℤ, ∀ x, (π / 6) + ↑k * π ≤ x ∧ x ≤ (2 * π / 3) + ↑k * π → 
     (Real.sin (2 * x + π / 6) is decreasing)) ∧ 
  (∀ m, (∃ x, -π / 4 < x ∧ x < π / 4 ∧ f 1 x = m) ↔ 1 - √3 < m ∧ m ≤ 2) :=
by
  sorry

end problem_conditions_l700_700928


namespace problem_statement_l700_700376

noncomputable def probability_x_gt_y : ℝ :=
  let x := uniform [0, 1]
  let y := uniform [-1, 1]
  Pr (fun (p : (ℝ × ℝ)) => p.1 > p.2)
  
theorem problem_statement : probability_x_gt_y = 3 / 4 :=
sorry

end problem_statement_l700_700376


namespace sum_series_l700_700410

theorem sum_series :
  ∑' n:ℕ, (4 * n ^ 2 - 2 * n + 3) / 3 ^ n = 21 / 4 :=
sorry

end sum_series_l700_700410


namespace length_MN_in_triangle_l700_700200

noncomputable def length_MN (X Y Z M N : EuclideanGeometry.Point) : ℝ :=
EuclideanGeometry.distance M N

theorem length_MN_in_triangle 
  (X Y Z L K M N P Q : EuclideanGeometry.Point)
  (XY XZ YZ : ℝ)
  (h1 : EuclideanGeometry.distance X Y = 130)
  (h2 : EuclideanGeometry.distance X Z = 112)
  (h3 : EuclideanGeometry.distance Y Z = 125)
  (h4 : EuclideanGeometry.IsAngleBisector (EuclideanGeometry.angle X Y Z) Y L)
  (h5 : EuclideanGeometry.IsAngleBisector (EuclideanGeometry.angle Y X Z) X K)
  (h6 : EuclideanGeometry.Perpendicular Z M Y K)
  (h7 : EuclideanGeometry.Perpendicular Z N X L)
  (h8 : EuclideanGeometry.collinear X Y P)
  (h9 : EuclideanGeometry.collinear X Y Q)
  (h10 : EuclideanGeometry.midpoint Z P M)
  (h11 : EuclideanGeometry.midpoint Z Q N)
  (h12 : EuclideanGeometry.distance Y P = 125)
  (h13 : EuclideanGeometry.distance X Q = 112)
  (h14 : P ∉ {X, Y}) -- Assuming P and Q are distinct and not X or Y
  (h15 : Q ∉ {X, Y}) :
  length_MN X Y Z M N = 53.5 := sorry

end length_MN_in_triangle_l700_700200


namespace exam_maximum_marks_l700_700252

-- Theorem declaration
theorem exam_maximum_marks (M : ℕ) : 
  (∀ (pass : ℕ), pass = 0.25 * M → (185 + 25 = pass) → M = 840) :=
sorry

end exam_maximum_marks_l700_700252


namespace correct_multiplication_result_l700_700962

theorem correct_multiplication_result (x : ℕ) (h : x - 6 = 51) : x * 6 = 342 :=
  by
  sorry

end correct_multiplication_result_l700_700962


namespace relationship_among_abc_l700_700483

noncomputable def a : ℝ := 4 ^ 0.6
noncomputable def b : ℝ := 8 ^ 0.34
noncomputable def c : ℝ := (1 / 2) ^ (-0.9)

theorem relationship_among_abc : a > b ∧ b > c :=
by
  sorry

end relationship_among_abc_l700_700483


namespace least_k_for_b_integer_l700_700314

-- The sequence {b_n} with the given conditions
noncomputable def sequence_b (n : ℕ) : ℝ :=
  if h : n = 0 then 1
  else classical.some (nat.strong_rec_on n (λ n ih, by
    cases n with
    | zero => exact ⟨1, by simp⟩
    | succ n => 
        let prev_b := ih n (nat.lt_succ_self n)
        have h_n : n ≥ 1 := nat.succ_le_iff.2 (nat.zero_lt_succ n)
        let lhs := 3 ^ (classical.some prev_b - prev_b.some)
        let rhs := 1 + 1 / (n + 0.5)
        exact classical.some_spec ⟨lhs, rhs⟩))

/-- Prove that the least integer k > 1 for which b_k is an integer is k = 3 -/
theorem least_k_for_b_integer : ∃ k > 1, b_k = 3 := 
begin
  use 3,
  split,
  { exact nat.one_lt_bit1 1 },
  { sorry }
end

end least_k_for_b_integer_l700_700314


namespace range_of_t_value_of_t_diameter_6_l700_700518

-- Representing the condition that the given equation is a circle
def equation_is_circle (t : ℝ) := 
  ∀ (x y : ℝ), (x^2 + y^2 + (real.sqrt 3 * t + 1) * x + t * y + t^2 - 2 = 0)

-- Problem (1): Finding the range of values for t
theorem range_of_t (t : ℝ) (h : equation_is_circle t) : 
  t > - (3 * real.sqrt 3) / 2 := 
sorry

-- Problem (2): Finding the value of t given the diameter of the circle is 6
theorem value_of_t_diameter_6 (t : ℝ) (h : equation_is_circle t) 
  (diameter_eq_6 : 6 = 6) : 
  t = (9 * real.sqrt 3) / 2 := 
sorry

end range_of_t_value_of_t_diameter_6_l700_700518


namespace part1_part2_l700_700625

noncomputable def f (x : ℝ) : ℝ := x^3 + 1 / (1 + x)

theorem part1 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : f x ≥ 1 - x + x^2 := sorry

theorem part2 (x : ℝ) (h : 0 ≤ x) (h1 : x ≤ 1) : 3 / 4 < f x ∧ f x ≤ 3 / 2 := sorry

end part1_part2_l700_700625


namespace total_miles_calculation_l700_700236

def total_miles (W : ℕ) := W * 67.5

theorem total_miles_calculation (W : ℕ) (hW : W ≥ 2) :
  total_miles W = W * 67.5 :=
by
  -- Conditions
  let L := 5
  let S := 6
  let X := 1
  let Y := 2
  let Z := 3
  let F := 8
  
  -- Calculation for each day
  let mwf_miles_per_day := L + (L + X)
  let tt_miles_per_day := S + (S + Y)
  let sat_miles := F + (F - Z)

  -- Weekly calculations
  let miles_MWF := 3 * mwf_miles_per_day
  let miles_TT := 2 * tt_miles_per_day
  let weekly_avg_sat_miles := sat_miles / 2

  -- Summing up the total weekly miles
  let total_weekly_miles := miles_MWF + miles_TT + weekly_avg_sat_miles

  -- Verifying the total weekly miles
  have h_total_weekly_miles : total_weekly_miles = 33 + 28 + 6.5 := by
    -- simplifying the calculations
    sorry

  -- Proving the theorem
  show total_miles W = W * total_weekly_miles from
    calc
      total_miles W = W * 67.5 : by sorry

end total_miles_calculation_l700_700236


namespace region_area_is_one_l700_700637

open Complex Real

def region_area : Set ℂ :=
  {z : ℂ | 0 ≤ arg (z - 1) ∧ arg (z - 1) ≤ π / 4 ∧ Re z ≤ 2}

theorem region_area_is_one : area region_area = 1 := by 
  sorry

end region_area_is_one_l700_700637


namespace simplify_and_evaluate_expression_l700_700651

-- Define a and b with given values
def a := 1 / 2
def b := 1 / 3

-- Define the expression
def expr := 5 * (3 * a ^ 2 * b - a * b ^ 2) - (a * b ^ 2 + 3 * a ^ 2 * b)

-- State the theorem
theorem simplify_and_evaluate_expression : expr = 2 / 3 := 
by
  -- Proof can be inserted here
  sorry

end simplify_and_evaluate_expression_l700_700651


namespace hike_on_saturday_l700_700272

-- Define the conditions
variables (x : Real) -- distance hiked on Saturday
variables (y : Real) -- distance hiked on Sunday
variables (z : Real) -- total distance hiked

-- Define given values
def hiked_on_sunday : Real := 1.6
def total_hiked : Real := 9.8

-- The hypothesis: y + x = z
axiom hike_total : y + x = z

theorem hike_on_saturday : x = 8.2 :=
by
  sorry

end hike_on_saturday_l700_700272


namespace days_needed_to_finish_l700_700345

theorem days_needed_to_finish (D_x D_y worked_days : ℕ) (hx : D_x = 20) (hy : D_y = 15) (ht : worked_days = 9) :
  let remaining_work := 1 - (worked_days / D_y : ℚ) in
  let days_x := remaining_work / (1 / D_x : ℚ) in
  days_x = 8 :=
by {
  have h1 : remaining_work = 1 - (9 / 15 : ℚ), by simp [hy, ht],
  have h2 : remaining_work = 2 / 5, by norm_num [h1],
  have h3 : days_x = (2 / 5) / (1 / 20 : ℚ), by simp [h2, hx],
  have h4 : days_x = 8, by norm_num [h3],
  exact h4,
}

end days_needed_to_finish_l700_700345


namespace triangle_ABC_area_l700_700019

def point := (ℝ × ℝ)

def A : point := (0, 0)
def B : point := (1, 2)
def C : point := (2, 0)

def triangle_area (A B C : point) : ℝ :=
  0.5 * |(A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))|

theorem triangle_ABC_area :
  triangle_area A B C = 2 :=
by
  sorry

end triangle_ABC_area_l700_700019


namespace count_non_zero_tenths_digit_decimals_l700_700466

def is_powers_of_2_and_5 (n : ℕ) : Prop :=
  ∀ p : ℕ, p.prime → p ∣ n → (p = 2 ∨ p = 5)

def has_non_zero_tenths_digit (n : ℕ) : Prop :=
  (n ≤ 50) ∧ is_powers_of_2_and_5 n

theorem count_non_zero_tenths_digit_decimals :
  { n | has_non_zero_tenths_digit n }.to_finset.card = 12 :=
by sorry

end count_non_zero_tenths_digit_decimals_l700_700466


namespace average_distance_l700_700632

theorem average_distance (block_length : ℕ) (johnny_times : ℕ) (mickey_ratio : ℕ) (alex_extra : ℕ)
  (H1 : block_length = 250)
  (H2 : johnny_times = 8)
  (H3 : mickey_ratio = 2)
  (H4 : alex_extra = 2) :
  let johnny_distance := johnny_times * block_length in
  let mickey_distance := (johnny_times / mickey_ratio) * block_length in
  let alex_distance := (johnny_times / mickey_ratio + alex_extra) * block_length in
  (johnny_distance + mickey_distance + alex_distance) / 3 = 1500 :=
by
  sorry

end average_distance_l700_700632


namespace triangle_B_is_right_triangle_l700_700945

theorem triangle_B_is_right_triangle :
  let a := 1
  let b := 2
  let c := Real.sqrt 3
  a^2 + c^2 = b^2 :=
by
  sorry

end triangle_B_is_right_triangle_l700_700945


namespace probability_same_color_pair_l700_700208

theorem probability_same_color_pair : 
  let total_shoes := 28
  let black_pairs := 8
  let brown_pairs := 4
  let gray_pairs := 2
  total_shoes = 2 * (black_pairs + brown_pairs + gray_pairs) → 
  ∃ (prob : ℚ), prob = 7 / 32 := by
  sorry

end probability_same_color_pair_l700_700208


namespace rhombus_perimeter_l700_700277

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  ∃ p : ℝ, p = 16 * Real.sqrt 13 := by
sorry

end rhombus_perimeter_l700_700277


namespace inequality_solution_l700_700269

noncomputable def solveInequality (x : ℝ) : Prop :=
  x > 0 ∧ log 4 x ≤ 2 → 
    log (1 / 2) x - sqrt (2 - log 4 x) + 1 ≤ 0

theorem inequality_solution :
  ∀ x : ℝ, solveInequality x ↔ (1 / sqrt 2 ≤ x ∧ x ≤ 16) :=
begin
  -- proof steps would go here
  sorry
end

end inequality_solution_l700_700269


namespace find_percentage_of_chromium_in_second_alloy_l700_700586

-- Define the conditions and constants
def first_alloy_chromium_percentage := 12 / 100
def first_alloy_weight := 10
def resulting_alloy_chromium_percentage := 9 / 100
def resulting_alloy_weight := 40

-- The unknown variable
variable (x : ℝ)

-- The condition to find the percentage of chromium in the second alloy
def second_alloy_chromium_percentage := x / 100
def second_alloy_weight := 30

-- The total amount of chromium in the resulting alloy
def total_chromium_in_resulting_alloy := 3.6

-- The equation representing the problem
theorem find_percentage_of_chromium_in_second_alloy :
  first_alloy_chromium_percentage * first_alloy_weight +
  second_alloy_chromium_percentage * second_alloy_weight = total_chromium_in_resulting_alloy →
  x = 8 := by
  sorry

end find_percentage_of_chromium_in_second_alloy_l700_700586


namespace arithmetic_sequence_l700_700496

theorem arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) (h1 : a 2 = 3) (h2 : a (n - 1) = 17) (h3 : n ≥ 2) (h4 : (n * (3 + 17)) / 2 = 100) : n = 10 :=
sorry

end arithmetic_sequence_l700_700496


namespace isosceles_triangle_largest_angle_l700_700583

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_iso : A = B) (h_A : C = 50) :
  max A (max B (180 - A - B)) = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l700_700583


namespace sin_cos_identity_l700_700837

theorem sin_cos_identity :
  sin (347 * real.pi / 180) * cos (148 * real.pi / 180) +
  sin (77 * real.pi / 180) * cos (58 * real.pi / 180) = real.sqrt 2 / 2 :=
by
  sorry

end sin_cos_identity_l700_700837


namespace total_percentage_of_failed_candidates_l700_700574

theorem total_percentage_of_failed_candidates :
  ∀ (total_candidates girls boys : ℕ) (passed_boys passed_girls : ℝ),
    total_candidates = 2000 →
    girls = 900 →
    boys = total_candidates - girls →
    passed_boys = 0.34 * boys →
    passed_girls = 0.32 * girls →
    (total_candidates - (passed_boys + passed_girls)) / total_candidates * 100 = 66.9 :=
by
  intros total_candidates girls boys passed_boys passed_girls
  intro h_total_candidates
  intro h_girls
  intro h_boys
  intro h_passed_boys
  intro h_passed_girls
  sorry

end total_percentage_of_failed_candidates_l700_700574


namespace sin_alpha_parallel_line_l700_700911

theorem sin_alpha_parallel_line : 
  ∀ (α : ℝ), 
  (∃ k : ℝ, k = 1 / 2 ∧ ∀ x y : ℝ, x - 2 * y + 2 = 0 → y = k * x) →
  sin α = (√5) / 5 :=
by
  sorry

end sin_alpha_parallel_line_l700_700911


namespace prime_digit_one_l700_700875

theorem prime_digit_one (p q r s : ℕ) (h1 : p > 10) (h2 : nat.prime p) (h3 : nat.prime q) (h4 : nat.prime r) (h5 : nat.prime s) 
                        (h6 : p < q) (h7 : q < r) (h8 : r < s) (h9 : q = p + 10) (h10 : r = q + 10) (h11 : s = r + 10) :
  (p % 10) = 1 := 
sorry

end prime_digit_one_l700_700875


namespace minimize_sum_first_n_terms_l700_700237

-- Define the scenario of the arithmetic sequence
def arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  a + (n - 1) * d

-- Sum of the first n terms for an arithmetic sequence
def sum_arithmetic_sequence (a d : ℤ) (n : ℕ) : ℤ :=
  n * a + d * n * (n - 1) / 2

theorem minimize_sum_first_n_terms :
  ∃ (n : ℕ), 
  (∀ m : ℕ, sum_arithmetic_sequence (-11) d m ≥ sum_arithmetic_sequence (-11) d 6) ∧
  (a_4 = arithmetic_sequence (-11) d 4) ∧
  (a_6 = arithmetic_sequence (-11) d 6) ∧
  (a_4 + a_6 = -6) ∧
  n = 6 :=
begin
  sorry
end

end minimize_sum_first_n_terms_l700_700237


namespace least_prime_value_l700_700127

open Nat

theorem least_prime_value (n : ℕ) (hn_prime : Prime n) (hn_cond : 101 * n^2 ≤ 3600) : |n| = 2 :=
sorry

end least_prime_value_l700_700127


namespace tangent_line_value_l700_700887

noncomputable def f : ℝ → ℝ := 
  sorry

theorem tangent_line_value :
  (f 1) + (derivative f 1) = 0 :=
by
  -- Given: the equation of the tangent line at point M(1, f(1)) is y = -x + 2
  -- Find: f(1) + f'(1) = 0 under the conditions that f is differentiable on R
  sorry

end tangent_line_value_l700_700887


namespace probability_is_one_fourth_l700_700327

-- Defining the set of positive integers less than or equal to 6
def S : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define the predicate for the desired condition
def condition (a b : ℕ) : Prop := (a * b - a - b > 2)

-- The function to calculate the probability
noncomputable def probability : ℚ :=
  (Finset.card (Finset.filter (λ ab, condition ab.1 ab.2) (Finset.product S S ))) / (Finset.card (Finset.product S S))

-- The theorem to prove
theorem probability_is_one_fourth : probability = 1 / 4 := by
  sorry

end probability_is_one_fourth_l700_700327


namespace first_train_cross_time_l700_700538

noncomputable def time_to_cross_bridge 
  (length_first_train : ℕ) 
  (speed_first_train_kmph : ℕ) 
  (length_bridge : ℕ) 
  (length_second_train : ℕ) 
  (speed_second_train_kmph : ℕ) : ℝ :=
  let speed_first_train := (speed_first_train_kmph * 1000) / 3600 in
  let speed_second_train := (speed_second_train_kmph * 1000) / 3600 in
  let relative_speed := speed_first_train + speed_second_train in
  let total_distance := length_first_train + length_bridge in
  total_distance / relative_speed

theorem first_train_cross_time 
  (length_first_train : ℕ) 
  (speed_first_train_kmph : ℕ) 
  (length_bridge : ℕ) 
  (length_second_train : ℕ) 
  (speed_second_train_kmph : ℕ) 
  (h1 : length_first_train = 250) 
  (h2 : speed_first_train_kmph = 90) 
  (h3 : length_bridge = 300) 
  (h4 : length_second_train = 200) 
  (h5 : speed_second_train_kmph = 75) : 
  |time_to_cross_bridge length_first_train speed_first_train_kmph length_bridge length_second_train speed_second_train_kmph - 12| < 0.01 :=
by {
  sorry
}

end first_train_cross_time_l700_700538


namespace smallest_c_minus_a_l700_700322

theorem smallest_c_minus_a (a b c : ℕ) (h1 : a * b * c = 720) (h2 : a < b) (h3 : b < c) : c - a ≥ 24 :=
sorry

end smallest_c_minus_a_l700_700322


namespace floor_and_ceiling_sum_l700_700035

theorem floor_and_ceiling_sum :
  (Int.floor 1.999 = 1) ∧ (Int.ceil 3.001 = 4) → (Int.floor 1.999 + Int.ceil 3.001 = 5) :=
by
  intro h
  cases h with h_floor h_ceil
  rw [h_floor, h_ceil]
  rfl

end floor_and_ceiling_sum_l700_700035


namespace touch_point_on_BD_l700_700365

variables (A B C D S P Q K L : Type) [pyramid A B C D S] 
variables (insphere : inscribed_sphere A B C D S)
variables (segment_PD : segments P D) (segment_PC : segments P C)
variables (touch_ABS : touches insphere A B S K) (touch_BCS : touches insphere B C S L)
variables (coplanar_PK_QL : coplanar P K Q L)

theorem touch_point_on_BD (X : Type) [touch_point insphere A B C D X] :
  lies_on X B D :=
sorry

end touch_point_on_BD_l700_700365


namespace factorize_difference_of_squares_l700_700439

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l700_700439


namespace minimum_block_interchanges_l700_700490

theorem minimum_block_interchanges (n : ℕ) : ∃ k : ℕ, k = ⌊ (n + 1) / 2 ⌋ ∧ 
  ∀ a : List ℕ, a = List.range n.reverse -> ∃ b : List ℕ, b = List.range n ∧ 
  -- Function to determine if one list can be obtained from another using block swaps
  (∃ m : ℕ, m ≤ k ∧ exist_block_swap_sequence a b m ) := 
sorry

end minimum_block_interchanges_l700_700490


namespace shaded_area_ratio_l700_700195

theorem shaded_area_ratio {x : ℝ} (h1 : AC = 2 * x) (h2 : CB = 6 * x) (h3 : AB = 8 * x)
  (r1 : radiusAB = 4 * x) (r2 : radiusAC = x) (r3 : radiusCB = 3 * x)
  (h4 : CD ⊥ AB) (h5 : CD = 3 * x) : 
  (7 * Real.pi * x^2) / (9 * Real.pi * x^2) = 7 / 9 := 
by 
  sorry

end shaded_area_ratio_l700_700195


namespace problem_inequalities_l700_700105

theorem problem_inequalities (a : ℕ) :
  (∀ x : ℕ, 0 < x → 3 * x > 4 * x - 6 → 2 * x - a > -9 ↔ x = 3) →
  (a = 13 ∨ a = 14) :=
by
  sorry

end problem_inequalities_l700_700105


namespace isosceles_triangle_largest_angle_l700_700582

theorem isosceles_triangle_largest_angle (A B C : ℝ) (h_iso : A = B) (h_A : C = 50) :
  max A (max B (180 - A - B)) = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l700_700582


namespace two_a_minus_b_equals_four_l700_700472

theorem two_a_minus_b_equals_four (a b : ℕ) 
    (consec_integers : b = a + 1)
    (min_a : min (Real.sqrt 30) a = a)
    (min_b : min (Real.sqrt 30) b = Real.sqrt 30) : 
    2 * a - b = 4 := 
sorry

end two_a_minus_b_equals_four_l700_700472


namespace line_hyperbola_intersection_l700_700592

theorem line_hyperbola_intersection (x : ℝ) : 
  ∃ x : ℝ, x ≠ 0 ∧ (sqrt 3 * x = 1 / x) := sorry

end line_hyperbola_intersection_l700_700592


namespace C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l700_700191

noncomputable def parametric_C1 (α : ℝ) : ℝ × ℝ := (2 + Real.cos α, 4 + Real.sin α)

noncomputable def polar_C2 (ρ θ m : ℝ) : ℝ := ρ * (Real.cos θ - m * Real.sin θ) + 1

theorem C1_Cartesian_equation :
  ∀ (x y : ℝ), (∃ α : ℝ, parametric_C1 α = (x, y)) ↔ (x - 2)^2 + (y - 4)^2 = 1 := sorry

theorem C2_Cartesian_equation :
  ∀ (x y m : ℝ), (∃ ρ θ : ℝ, polar_C2 ρ θ m = 0 ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ)
  ↔ x - m * y + 1 = 0 := sorry

def closest_point_on_C1_to_x_axis : ℝ × ℝ := (2, 3)

theorem m_value_when_C2_passes_through_P :
  ∃ (m : ℝ), x - m * y + 1 = 0 ∧ x = 2 ∧ y = 3 ∧ m = 1 := sorry

end C1_Cartesian_equation_C2_Cartesian_equation_m_value_when_C2_passes_through_P_l700_700191


namespace roger_remaining_debt_l700_700259

-- Given definitions
def house_price : ℝ := 100000
def down_payment_percentage : ℝ := 0.20
def parents_payment_percentage : ℝ := 0.30

-- Equivalent proof problem in Lean 4 statement
theorem roger_remaining_debt : 
  let down_payment := house_price * down_payment_percentage,
      remaining_after_down := house_price - down_payment,
      parents_payment := remaining_after_down * parents_payment_percentage,
      roger_debt := remaining_after_down - parents_payment
  in roger_debt = 56000 := sorry

end roger_remaining_debt_l700_700259


namespace harmonic_bounded_inequality_l700_700142

-- Definitions from conditions
def sequence_a (n : ℕ) : ℕ := 3 * n + 2

def is_on_line (x y : ℕ) : Prop := x - y + 3 = 0

-- Given condition (sum of first 9 terms is 153)
def sum_first_9_terms : ℕ := ∑ i in finset.range 9, sequence_a (i + 1)
lemma sum_first_9_terms_is_153 : sum_first_9_terms = 153 := sorry

-- General term formula of the sequence {a_n}
lemma general_term_formula (n : ℕ) : sequence_a n = 3 * n + 2 := sorry

-- Defining the new sequence {b_n} by removing terms (General formula provided)
def sequence_b (n : ℕ) : ℕ := 3 * n * 2^n + 2

-- Sum of the first 'n' terms of {b_n}
def sum_b (n : ℕ) : ℕ := ∑ i in finset.range n, sequence_b (i + 1)
lemma sum_first_n_terms_B (n : ℕ) : sum_b n = 3 * ((n - 1) * 2^(n + 1) + 2) + 2 * n := sorry

-- Prove the inequality for the harmonic series-like sum
theorem harmonic_bounded_inequality (n : ℕ) :
  ∑ i in finset.range n, (1 / sequence_b (i + 1) : ℝ) < 1/4 := sorry

end harmonic_bounded_inequality_l700_700142


namespace unit_digit_G_1000_l700_700418

/-- Define what we mean by G_n. -/
def G (n : ℕ) : ℕ := 3 * 2^(2^n) + 4

/-- The units digit of a number. -/
def unit_digit (n : ℕ) : ℕ := n % 10

/-- The repeating cycle of units digits of 2^n. -/
def cycle_2_pow_units : list ℕ := [2, 4, 8, 6]

/-- The nth element in the repeating cycle of units digits of 2^n. -/
def cycle_2_pow_units_n (n : ℕ) : ℕ := cycle_2_pow_units (n % 4)

/-- Proof problem: Calculate the units digit of G_{1000}. -/
theorem unit_digit_G_1000 : unit_digit (G 1000) = 6 :=
by
  sorry

end unit_digit_G_1000_l700_700418


namespace minimum_value_exists_exp_gt_quadratic_l700_700933

open Real

-- Define the function f
def f (x : ℝ) : ℝ := exp x - 2 * x + 2

-- The Lean statement for the equivalent proof problem
theorem minimum_value_exists : ∃ x ∈ ℝ, ∀ x, f(x) ≥ f(log 2) ∧ f(log 2) = 2 * (2 - log 2) := sorry

theorem exp_gt_quadratic (x : ℝ) (h : x > 0) : exp x > x^2 - 2 * x + 1 := sorry

end minimum_value_exists_exp_gt_quadratic_l700_700933


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700049

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700049


namespace count_even_divisor_numbers_correct_l700_700548

-- Define the range for three-digit number
def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n <= 999

-- Define whether a number has an odd number of divisors
def has_odd_divisors (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

-- Count numbers with even divisors
def count_even_divisor_numbers : ℕ :=
  (Σ' (n : ℕ), is_three_digit n ∧ ¬ has_odd_divisors n).to_finset.card

theorem count_even_divisor_numbers_correct :
  count_even_divisor_numbers = 878 :=
by
  sorry

end count_even_divisor_numbers_correct_l700_700548


namespace equalities_hold_l700_700353

theorem equalities_hold : (12 / 60 = 0.2) ∧ (4 / 20 = 0.2) ∧ (0.2 = 20 / 100) := 
by {
  sorry,
}

end equalities_hold_l700_700353


namespace probability_within_smaller_circle_l700_700361

noncomputable def radius_large : ℝ := 4
noncomputable def radius_small : ℝ := 2

def area (r : ℝ) : ℝ := Real.pi * r^2

theorem probability_within_smaller_circle :
  let area_large := area radius_large
  let area_small := area radius_small
  (area_small / area_large) = 1 / 4 :=
by
  let area_large := area radius_large
  let area_small := area radius_small
  have h_area_large : area_large = 16 * Real.pi := by sorry
  have h_area_small : area_small = 4 * Real.pi := by sorry
  calc
    (area_small / area_large)
        = (4 * Real.pi / 16 * Real.pi) : sorry
    ... = 1 / 4 : sorry

end probability_within_smaller_circle_l700_700361


namespace prob_nth_letter_A_l700_700334

def toss_coin : Type := bool

noncomputable def letter_array (n : ℕ) : list char :=
  if n = 0 then []
  else if toss_coin then 'A' :: 'A' :: letter_array (n - 1)
    else 'B' :: letter_array (n - 1)

noncomputable def num_A (n : ℕ) : ℕ :=
  (letter_array n).count 'A'

noncomputable def num_B (n : ℕ) : ℕ :=
  (letter_array n).count 'B'

theorem prob_nth_letter_A (n : ℕ) : 
  0 < n → 
  (num_A n / (num_A n + num_B n)) = 
  (nat.rec_on n 1 (λ n hn, hn - 2 + nat.rec_on n 1 (λ n hn, hn - 1 + hn - 1))) / 
  (nat.rec_on n 1 (λ n hn, hn - 1 + nat.rec_on n 1 (λ n hn, hn - 1 + nat.rec_on n 1 (λ n hn, hn - 1 + hn - 1)))) := sorry

end prob_nth_letter_A_l700_700334


namespace matrix_exponentiation_l700_700413

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2],
    ![2, -1]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-4, 6],
    ![-6, 5]]

theorem matrix_exponentiation :
  A^4 = B :=
by
  sorry

end matrix_exponentiation_l700_700413


namespace collinear_iff_R1_even_k_concurrent_iff_R1_odd_k_l700_700126

-- Given definitions and conditions
variable {A B C A1 B1 C1 : Type}
           [IsPointOfTriangle A B C]
           [IsPointOnLine C1 AB (k = 3)]
           [IsPointOnLine A1 BC (k = 3)]
           [IsPointOnLine B1 CA (k = 3)]

-- Define the ratio R
noncomputable def R : ℝ :=
  (distance B A1 / distance C A1) * (distance C B1 / distance A B1) * (distance A C1 / distance B C1)
 
-- Double implication for collinearity
theorem collinear_iff_R1_even_k {A1 B1 C1 : Point} (h : IsPointOnLine A1 BC ∧ IsPointOfTriangle A1 B C ∧ IsPointOnLine B1 CA ∧ IsPointOfTriangle B1 C A ∧ IsPointOnLine C1 AB ∧ IsPointOfTriangle C1 A B) :
  (IsCollinear A1 B1 C1 ↔ R = 1 ∧ (k % 2 = 0)) :=
sorry

-- Double implication for concurrent lines
theorem concurrent_iff_R1_odd_k {A1 B1 C1 : Point} (h : IsPointOnLine A1 BC ∧ IsPointOnLine B1 CA ∧ IsPointOnLine C1 AB):
  (AreConcurrent AA1 BB1 CC1 ↔ R = 1 ∧ (k % 2 = 1)) :=
sorry

end collinear_iff_R1_even_k_concurrent_iff_R1_odd_k_l700_700126


namespace solution_set_of_x_l700_700073

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l700_700073


namespace collinear_A_X_Y_l700_700897

-- Definitions and conditions
variables {A B C D E X Y : Type}
variables [IsTriangle A B C]
variables [IsAltitudeFeetFrom B A C D]
variables [IsAltitudeFeetFrom C A B E]
variables [IsTangentPointsIntersect B C A Y]
variables [IsCircleTangent D E A Gamma1]
variables [IsCircleTangent E D A Gamma2]
variables [DoesIntersect Gamma1 Gamma2 X]

-- Prove that A, X, and Y are collinear
theorem collinear_A_X_Y 
  (ACute : IsAcuteTriangle A B C)
  (Altitudes : IsAltitudeFeetFroms B C A D E)
  (TangentsIntersect : IsTangentPointsIntersect B C A Y)
  (Circ1 : IsCircleTangent D E A Gamma1)
  (Circ2 : IsCircleTangent E D A Gamma2)
  (Intersects : DoesIntersect Gamma1 Gamma2 X) : 
  Collinear [A, X, Y] := 
  sorry

end collinear_A_X_Y_l700_700897


namespace trigonometric_identity_l700_700014

theorem trigonometric_identity :
  4 * real.sin (real.pi / 12) + real.tan (5 * real.pi / 12) = 
  (4 - 3 * real.cos (real.pi / 12)^2 + real.cos (real.pi / 12)) / real.sin (real.pi / 12) :=
by
  sorry

end trigonometric_identity_l700_700014


namespace max_possible_single_player_salary_l700_700757

theorem max_possible_single_player_salary 
  (num_players : ℕ) (min_salary_per_player : ℕ) (total_salary_cap : ℕ) 
  (h1 : num_players = 21)
  (h2 : min_salary_per_player = 20000)
  (h3 : total_salary_cap = 900000) 
  : ∃ s, s <= total_salary_cap ∧ (∃ salaries : fin num_players → ℕ, 
      (∀ i, salaries i >= min_salary_per_player) ∧ 
      (finset.univ.sum salaries = total_salary_cap) ∧ 
      (∃ j, salaries j = s)) → 
  max_possible_single_player_salary = 500000 := 
sorry

end max_possible_single_player_salary_l700_700757


namespace min_coins_required_l700_700328

-- Definitions corresponding to the conditions
def Coins := ℕ  -- We'll use natural numbers to represent coins
def PennyValue := 1  -- Value of a penny in cents
def NickelValue := 5  -- Value of a nickel in cents
def QuarterValue := 25  -- Value of a quarter in cents
def HalfDollarValue := 50  -- Value of a half-dollar in cents

-- Main theorem statement
theorem min_coins_required : ∀ total : ℕ,
  total < 100 →
  ∃ (p n q h : Coins), 
    p * PennyValue + n * NickelValue + q * QuarterValue + h * HalfDollarValue = total ∧ 
    p + n + q + h = 10 :=
by
  sorry

end min_coins_required_l700_700328


namespace mark_final_buttons_l700_700629

def mark_initial_buttons : ℕ := 14
def shane_factor : ℚ := 3.5
def lent_to_anna : ℕ := 7
def lost_fraction : ℚ := 0.5
def sam_fraction : ℚ := 2 / 3

theorem mark_final_buttons : 
  let shane_buttons := mark_initial_buttons * shane_factor
  let before_anna := mark_initial_buttons + shane_buttons
  let after_lending_anna := before_anna - lent_to_anna
  let anna_returned := lent_to_anna * (1 - lost_fraction)
  let after_anna_return := after_lending_anna + anna_returned
  let after_sam := after_anna_return - (after_anna_return * sam_fraction)
  round after_sam = 20 := 
by
  sorry

end mark_final_buttons_l700_700629


namespace find_x_squared_plus_y_squared_l700_700552

open Real

theorem find_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 := by
  sorry

end find_x_squared_plus_y_squared_l700_700552


namespace average_of_remaining_two_numbers_l700_700341

theorem average_of_remaining_two_numbers (a b c d e f : ℝ)
(h_avg_6 : (a + b + c + d + e + f) / 6 = 3.95)
(h_avg_2_1 : (a + b) / 2 = 3.4)
(h_avg_2_2 : (c + d) / 2 = 3.85) :
  (e + f) / 2 = 4.6 := 
sorry

end average_of_remaining_two_numbers_l700_700341


namespace sum_infinite_geometric_series_l700_700401

theorem sum_infinite_geometric_series :
  let a := 1
  let r := (1 : ℝ) / 3
  ∑' (n : ℕ), a * r ^ n = (3 : ℝ) / 2 :=
by
  sorry

end sum_infinite_geometric_series_l700_700401


namespace probability_correct_l700_700707

-- Define the basic setup of the problem
def is_valid_point (x y : ℕ) : Prop :=
  y = -x^2 + 3 * x

def points_on_parabola : list (ℕ × ℕ) :=
  [(1, 2), (2, 2)]

-- Check if a point is on the parabola
def is_on_parabola (x y : ℕ) :=
  (x, y) ∈ points_on_parabola.filter (λ xy, is_valid_point xy.1 xy.2)

-- Calculate the probability
def probability_on_parabola : ℚ :=
  (points_on_parabola.filter (λ xy, is_valid_point xy.1 xy.2)).length / 36

theorem probability_correct :
  probability_on_parabola = 1 / 18 :=
by
  sorry

end probability_correct_l700_700707


namespace work_duration_l700_700344

theorem work_duration (work_rate_x work_rate_y : ℚ) (time_x : ℕ) (total_work : ℚ) :
  work_rate_x = (1 / 20) → 
  work_rate_y = (1 / 12) → 
  time_x = 4 → 
  total_work = 1 →
  ((time_x * work_rate_x) + ((total_work - (time_x * work_rate_x)) / (work_rate_x + work_rate_y))) = 10 := 
by
  intro h1 h2 h3 h4
  rw [h1, h2, h3, h4]
  sorry

end work_duration_l700_700344


namespace sequence_formula_l700_700894

variable {α : Type*} 

def S (a : ℕ → ℕ) (n : ℕ) : ℕ := ∑ i in finset.range (n + 1), a i

theorem sequence_formula :
  ∀ (a : ℕ → ℕ),
    (a 1 = 2) →
    (∀ n : ℕ, 0 < n → a (n + 1) - S a n = 2) →
    ∀ n : ℕ, a n = 2^n := 
by
  intros a h1 h2 n
  sorry

end sequence_formula_l700_700894


namespace complex_number_solution_l700_700517

theorem complex_number_solution (z : ℂ) (h : (1 + complex.I) * z = 2 * complex.I) : 
  z = 1 + complex.I :=
sorry

end complex_number_solution_l700_700517


namespace locus_of_orthocenter_l700_700318

def point (α : Type) := α × α

variable {α : Type}

-- Define points A, B, and variable point C
def A (a : α) : point α := (-a, 0)
def B (a : α) : point α := (a, 0)
def C (k b : α) : point α := (k, b)

theorem locus_of_orthocenter (a k b : ℝ) :
  ∃ x y : ℝ, C k b = (x, y) → b * y = a^2 - x^2 :=
by
  sorry

end locus_of_orthocenter_l700_700318


namespace balloons_left_l700_700709

theorem balloons_left (yellow blue pink violet friends : ℕ) (total_balloons remainder : ℕ) 
  (hy : yellow = 20) (hb : blue = 24) (hp : pink = 50) (hv : violet = 102) (hf : friends = 9)
  (ht : total_balloons = yellow + blue + pink + violet) (hr : total_balloons % friends = remainder) : 
  remainder = 7 :=
by
  sorry

end balloons_left_l700_700709


namespace roots_of_quadratic_l700_700509

theorem roots_of_quadratic (p q x1 x2 : ℕ) (hp : p + q = 28) (hroots : ∀ x, x^2 + p * x + q = 0 → (x = x1 ∨ x = x2)) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end roots_of_quadratic_l700_700509


namespace proof_problem_l700_700474

-- Define the minimum function for two real numbers
def min (x y : ℝ) := if x < y then x else y

-- Define the real numbers sqrt30, a, and b
def sqrt30 := Real.sqrt 30

variables (a b : ℕ)

-- Define the conditions
def conditions := (min sqrt30 a = a) ∧ (min sqrt30 b = sqrt30) ∧ (b = a + 1)

-- State the theorem to prove
theorem proof_problem (h : conditions a b) : 2 * a - b = 4 :=
sorry

end proof_problem_l700_700474


namespace exists_permutation_satisfies_eq_l700_700725

theorem exists_permutation_satisfies_eq : 
  ∃ (a b c d e f : ℕ), 
    {a, b, c, d, e, f} = {1, 2, 3, 4, 5, 6} ∧ 
    (a-1)*(b-2)*(c-3)*(d-4)*(e-5)*(f-6) = 75 := 
by
  sorry

end exists_permutation_satisfies_eq_l700_700725


namespace min_abs_sum_of_cubes_l700_700143

open Complex

variable (z₁ z₂ : ℂ)

theorem min_abs_sum_of_cubes (h₁ : ∥z₁ + z₂∥ = 20) (h₂ : ∥z₁^2 + z₂^2∥ = 16) : 
  ∥z₁^3 + z₂^3∥ = 3520 := 
by 
  sorry

end min_abs_sum_of_cubes_l700_700143


namespace problem_f_3_eq_e_squared_l700_700883

-- The following Lean code states the problem and proof statement
theorem problem_f_3_eq_e_squared :
  (∀ x > 0, f (Real.log x + 1) = x) → f 3 = Real.exp 2 :=
sorry

end problem_f_3_eq_e_squared_l700_700883


namespace odd_function_alpha_l700_700522
open Real

noncomputable def f (x : ℝ) : ℝ :=
  cos x * (sin x + sqrt 3 * cos x) - sqrt 3 / 2

noncomputable def g (x : ℝ) (α : ℝ) : ℝ :=
  f (x + α)

theorem odd_function_alpha (α : ℝ) (a : α > 0) :
  (∀ x : ℝ, g x α = - g (-x) α) ↔ 
  ∃ k : ℕ, α = (2 * k - 1) * π / 6 := sorry

end odd_function_alpha_l700_700522


namespace value_of_f_2017_l700_700115

def f (x : ℕ) : ℕ := x^2 - x * (0 : ℕ) - 1

theorem value_of_f_2017 : f 2017 = 2016 * 2018 := by
  sorry

end value_of_f_2017_l700_700115


namespace circle_range_l700_700886

constant a : Real
constant O1_eq : a > -2 → (x y : Real) → x^2 + y^2 + 2*x - 2*a*y - 8*a - 15 = 0
constant O2_eq : a > -2 → (x y : Real) → x^2 + y^2 + 2*a*x - 2*a*y + a^2 - 4*a - 4 = 0

theorem circle_range (ha : a > -2) :
  (∃ (x y : Real), O1_eq ha x y ∧ O2_eq ha x y) →
  (-5/3 ≤ a ∧ a ≤ -1) ∨ (3 ≤ a) :=
sorry

end circle_range_l700_700886


namespace complex_norm_example_l700_700845

theorem complex_norm_example : 
  abs (-3 - (9 / 4 : ℝ) * I) = 15 / 4 := 
by
  sorry

end complex_norm_example_l700_700845


namespace total_games_in_season_l700_700385

theorem total_games_in_season 
  (teams : ℕ) (divisions : ℕ) (teams_per_division : Π (d : ℕ), d < divisions → ℕ) 
  (games_per_team_in_division : ℕ) (games_per_team_between_divisions : ℕ)
  (h_teams : teams = 16) (h_divisions : divisions = 2)
  (h_teams_per_division : ∀ (d : ℕ) (hd : d < divisions), teams_per_division d hd = 8)
  (h_games_per_team_in_division : games_per_team_in_division = 3)
  (h_games_per_team_between_divisions : games_per_team_between_divisions = 2) :
  ∑ d in finset.range divisions, ∑ i in finset.range (teams_per_division d (nat.lt_of_lt_of_le d h_divisions.le)), 
    ∑ j in finset.range (teams_per_division d (nat.lt_of_lt_of_le d h_divisions.le)), if i < j then games_per_team_in_division else 0 +
  ∑ i in finset.range (teams_per_division 0 (nat.lt_of_lt_of_le 0 h_divisions.le)), 
    ∑ j in finset.range (teams_per_division 1 (nat.lt_of_lt_of_le 1 h_divisions.le)), games_per_team_between_divisions = 296 :=
  sorry

end total_games_in_season_l700_700385


namespace num_ways_to_arrange_PERCEPTION_l700_700830

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l700_700830


namespace num_integers_with_digit_sum_18_l700_700542

-- Defining the conditions:
def is_digit_sum_18 (n : ℕ) := 
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 + d2 + d3 = 18

def within_bounds (n : ℕ) := 500 ≤ n ∧ n ≤ 700

-- The main theorem statement:
theorem num_integers_with_digit_sum_18 : {n : ℕ | within_bounds n ∧ is_digit_sum_18 n}.to_finset.card = 25 :=
by
  sorry

end num_integers_with_digit_sum_18_l700_700542


namespace parallel_vectors_have_specific_m_l700_700161

variable (m : ℝ)

def a := (m, 1)
def b := (3, m + 2)

theorem parallel_vectors_have_specific_m (h : a m ∥ b m) : m = -3 ∨ m = 1 := by
  sorry

end parallel_vectors_have_specific_m_l700_700161


namespace sum_of_special_primes_l700_700687

theorem sum_of_special_primes : 
  ∑ p in {p : ℕ | prime p ∧ ∃ n : ℕ+, 5 * p = (n^2 / 5)} = 52 :=
sorry

end sum_of_special_primes_l700_700687


namespace rectangular_prism_lateral_edge_length_l700_700973

-- Definition of the problem conditions
def is_rectangular_prism (v : ℕ) : Prop := v = 8
def sum_lateral_edges (l : ℕ) : ℕ := 4 * l

-- Theorem stating the problem to prove
theorem rectangular_prism_lateral_edge_length :
  ∀ (v l : ℕ), is_rectangular_prism v → sum_lateral_edges l = 56 → l = 14 :=
by
  intros v l h1 h2
  sorry

end rectangular_prism_lateral_edge_length_l700_700973


namespace consecutive_integer_min_values_l700_700476

theorem consecutive_integer_min_values (a b : ℝ) 
  (consec : b = a + 1) 
  (min_a : a ≤ real.sqrt 30) 
  (min_b : b ≥ real.sqrt 30) : 
  2 * a - b = 4 := 
sorry

end consecutive_integer_min_values_l700_700476


namespace main_theorem_l700_700098

noncomputable section

variables {n : ℕ} (a : Fin n → ℝ) (x y : ℝ)

def pos_seq (a : Fin n → ℝ) : Prop :=
  (∀ i j : Fin n, i ≤ j → a i ≤ a j ∨ a i ≥ a j) ∧ a 0 ≠ a (Fin.last n)

def valid_xy (a : Fin n → ℝ) (x y : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ (x / y ≥ (a 0 - a 1) / (a 0 - a (Fin.last n)))

theorem main_theorem (h_n : 2 < n)
    (h_seq : pos_seq a)
    (h_xy : valid_xy a x y) :
    (∑ i : Fin n, a i / (a (i + 1) * x + a (i + 2) * y)) ≥ (n / (x + y)) :=
      sorry

end main_theorem_l700_700098


namespace functional_solution_l700_700610

def functional_property (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), f (x * f y + 2 * x) = x * y + 2 * f x

theorem functional_solution (f : ℝ → ℝ) (h : functional_property f) : f 1 = 0 :=
by sorry

end functional_solution_l700_700610


namespace increasing_sequences_remainder_l700_700007

theorem increasing_sequences_remainder :
  let p := 1008 in
  let remainder := p % 1000 in
  remainder = 8 :=
by
  sorry

end increasing_sequences_remainder_l700_700007


namespace basketball_team_selection_l700_700642

noncomputable def binom : ℕ → ℕ → ℕ
| n, k :=
  if k > n then 0
  else Nat.choose n k

theorem basketball_team_selection (n : ℕ) (k : ℕ)
  (h₁ : n = 16)
  (h₂ : k = 7)
  (h₃ : ∀ (x y : ℕ), x + y = n ∧ (x = 2 ∨ y = 2) → k = 7): 
  binom 14 5 + binom 14 7 = 5434 := by
  sorry

end basketball_team_selection_l700_700642


namespace heather_biked_per_day_l700_700537

def total_kilometers_biked : ℝ := 320
def days_biked : ℝ := 8
def kilometers_per_day : ℝ := 40

theorem heather_biked_per_day : total_kilometers_biked / days_biked = kilometers_per_day := 
by
  -- Proof will be inserted here
  sorry

end heather_biked_per_day_l700_700537


namespace cross_section_area_in_prism_l700_700573

noncomputable def area_cross_section (a : ℝ) : ℝ :=
  (3 * Real.sqrt 3 / 64) * a^2

theorem cross_section_area_in_prism 
  (a : ℝ)
  (P A B C : Type)
  [triangle_prism : is_regular_triangular_prism P A B C]
  (base_length : base_length BC = a)
  (cross_section : ∃ cross_section : P, is_parallel cross_section BC ∧ is_perpendicular cross_section PBC ∧ angle cross_section ABC = 30) :
  area_cross_section a = (3 * Real.sqrt 3 / 64) * a^2 :=
begin
  sorry
end

end cross_section_area_in_prism_l700_700573


namespace declaration_day_of_week_l700_700273

theorem declaration_day_of_week :
  ∀ d : Nat,
  (d = day_of_week 1776 7 4) →
  (day_of_week 2026 7 4 = 6) →
  (count_leap_years 1776 2026 = 59) →
  (let total_days := 250 - 59 + 59 * 2;
         shifts := total_days % 7 in
    d = (6 + 7 - shifts) % 7) →
  d = 5 :=
begin
  sorry
end

end declaration_day_of_week_l700_700273


namespace acid_solution_final_percentage_l700_700656

theorem acid_solution_final_percentage (initial_volume : ℝ) (initial_percentage : ℝ)
  (fraction_replaced : ℝ) (replacement_percentage : ℝ) : 
  initial_percentage = 0.5 → 
  fraction_replaced = 0.5 → 
  replacement_percentage = 0.2 → 
  let final_percentage := (fraction_replaced * initial_percentage 
                          + (1 - fraction_replaced) * replacement_percentage) * 100 
  in final_percentage = 35 := 
by
  sorry

end acid_solution_final_percentage_l700_700656


namespace second_group_men_count_l700_700728

theorem second_group_men_count (work_const : ∀ (n m : ℕ), n * m = 2400)
: ∃ M : ℕ, M * 60 = 2400 :=
begin
  use 40,
  norm_num,
end

end second_group_men_count_l700_700728


namespace distance_between_vertices_l700_700218

noncomputable def vertex_distance : ℝ :=
let C := (2, 3) 
let D := (-3, 11) in
real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)

theorem distance_between_vertices :
  vertex_distance = real.sqrt 89 :=
by
  sorry

end distance_between_vertices_l700_700218


namespace find_constants_l700_700848

theorem find_constants (a b : ℚ) (h1 : 3 * a + b = 7) (h2 : a + 4 * b = 5) :
  a = 61 / 33 ∧ b = 8 / 11 :=
by
  sorry

end find_constants_l700_700848


namespace range_of_a_for_monotonic_decreasing_l700_700940

theorem range_of_a_for_monotonic_decreasing (a : ℝ) :
(∀ x : ℝ, 0 < x ∧ x < 1 → 2 ^ (x * (x - a)) > 2 ^ (x * (x - a)) → a ∈ set.Ici 2) :=
begin
  sorry
end

end range_of_a_for_monotonic_decreasing_l700_700940


namespace angle_equality_l700_700684

variables {A B C D E F : Type}
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D] [AffineSpace ℝ E] [AffineSpace ℝ F]
variables [vector [0]] [vector [1]] [vector [2]] [vector [3]]

noncomputable def parallelogram_vertex_on_side (A B C D E F : AffineSpace ℝ A) :=
  let AE := 2 * vector [2, point D, point C] in -- given AE = 2CD
  ∃ (F : AffineSpace ℝ [B, C]), -- given F lies on BC
  let AC := A - C in -- given AC = AD
  let AD := A - D 
  AC = AD → 
theorem angle_equality (A B C D E F : AffineSpace ℝ A) : 
  ∃ (Condition : parallelogram_vertex_on_side A B C D E F),
  ∠ (C D E) = ∠ (B E F) :=
sorry

end angle_equality_l700_700684


namespace solution_set_of_x_l700_700077

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l700_700077


namespace floor_eq_solution_l700_700058

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l700_700058


namespace probability_log3_integer_l700_700378

noncomputable def three_digit_numbers := {N : ℕ | 100 ≤ N ∧ N ≤ 999}

noncomputable def is_power_of_three (N : ℕ) : Prop := ∃ k : ℕ, N = 3^k

theorem probability_log3_integer : 
  (∀ N ∈ three_digit_numbers, (is_power_of_three N ∨ ¬is_power_of_three N)) →
  (∀ N ∈ three_digit_numbers, is_power_of_three N → N = 243 ∨ N = 729) →
  (∑ N in three_digit_numbers, 1) = 900 →
  P {N : ℕ | 100 ≤ N ∧ N ≤ 999 ∧ is_power_of_three N} = 2/900 :=
  sorry

end probability_log3_integer_l700_700378


namespace least_three_digit_7_heavy_l700_700773

-- Define what it means to be a 7-heavy number
def is_7_heavy (n : ℕ) : Prop :=
  n % 7 > 4

-- Define the property of being three-digit
def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

-- The statement to prove
theorem least_three_digit_7_heavy : ∃ n, is_7_heavy n ∧ is_three_digit n ∧ ∀ m, is_7_heavy m ∧ is_three_digit m → n ≤ m :=
begin
  use [104],
  split,
  { -- Proof that 104 is 7-heavy
    show is_7_heavy 104,
    simp [is_7_heavy], -- Calculation: 104 % 7 = 6 which is > 4
    norm_num,
  },
  split,
  { -- Proof that 104 is a three-digit number
    show is_three_digit 104,
    simp [is_three_digit],
    norm_num,
  },
  { -- Proof that 104 is the smallest 7-heavy three-digit number
    intros m hm,
    cases hm with hm1 hm2,
    suffices : 104 ≤ m,
    exact this,
    calc 104 ≤ 100 + 7 - 1 : by norm_num
        ... ≤ m            : by linarith [hm2.left, hm2.right],
    sorry,
  }
sorry

end least_three_digit_7_heavy_l700_700773


namespace mary_final_books_l700_700239

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end mary_final_books_l700_700239


namespace seq_identity_l700_700424

-- Define the sequence (a_n)
def seq (a : ℕ → ℕ) : Prop :=
  a 0 = 0 ∧ a 1 = 0 ∧ a 2 = 1 ∧ ∀ n, a (n + 3) = a (n + 1) + 1998 * a n

theorem seq_identity (a : ℕ → ℕ) (h : seq a) (n : ℕ) (hn : 0 < n) :
  a (2 * n - 1) = 2 * a n * a (n + 1) + 1998 * (a (n - 1))^2 :=
sorry

end seq_identity_l700_700424


namespace problem_number_of_ways_to_choose_2005_balls_l700_700093

def number_of_ways_to_choose_balls (n : ℕ) : ℕ :=
  binomial (n + 2) 2 - binomial ((n + 1) / 2 + 1) 2

theorem problem_number_of_ways_to_choose_2005_balls :
  number_of_ways_to_choose_balls 2005 = binomial 2007 2 - binomial 1004 2 :=
by
  -- Proof will be provided here.
  sorry

end problem_number_of_ways_to_choose_2005_balls_l700_700093


namespace integral_sin_sin_sq_converges_l700_700263

theorem integral_sin_sin_sq_converges :
  ∃ L : ℝ, tendsto (λ k : ℝ, ∫ x in 0..k, Real.sin x * Real.sin (x^2)) atTop (nhds L) := by
  sorry

end integral_sin_sin_sq_converges_l700_700263


namespace find_f3_l700_700885

variable (f : ℕ → ℕ)

axiom h : ∀ x : ℕ, f (x + 1) = x ^ 2

theorem find_f3 : f 3 = 4 :=
by
  sorry

end find_f3_l700_700885


namespace angle_YZX_60_degrees_l700_700407

theorem angle_YZX_60_degrees (A B C X Y Z : Type)
  (incircle_ABC : circle)
  (circumcircle_XYZ : circle)
  (hx : X ∈ segment B C)
  (hy : Y ∈ segment A B)
  (hz : Z ∈ segment A C)
  (h_ABC_angles : ∠A = 50 ∧ ∠B = 70 ∧ ∠C = 60)
  (h_incircle_ABC : is_incircle incircle_ABC (triangle A B C))
  (h_circumcircle_XYZ : is_circumcircle circumcircle_XYZ (triangle X Y Z)) :
  ∠YZX = 60 :=
by
  sorry

end angle_YZX_60_degrees_l700_700407


namespace num_integers_with_digit_sum_18_l700_700543

-- Defining the conditions:
def is_digit_sum_18 (n : ℕ) := 
  let d1 := n / 100
  let d2 := (n % 100) / 10
  let d3 := n % 10
  d1 + d2 + d3 = 18

def within_bounds (n : ℕ) := 500 ≤ n ∧ n ≤ 700

-- The main theorem statement:
theorem num_integers_with_digit_sum_18 : {n : ℕ | within_bounds n ∧ is_digit_sum_18 n}.to_finset.card = 25 :=
by
  sorry

end num_integers_with_digit_sum_18_l700_700543


namespace integral_sqrt_9_minus_x_squared_l700_700400

/-- 
  Prove that the definite integral of the square root of (9 - x^2) from -3 to 3 is 
  equal to 9π/2.
-/
theorem integral_sqrt_9_minus_x_squared :
  ∫ x in -3..3, sqrt (9 - x^2) = 9 * Real.pi / 2 :=
by
  sorry

end integral_sqrt_9_minus_x_squared_l700_700400


namespace sum_of_roots_l700_700458

theorem sum_of_roots :
  let P : Polynomial ℤ := (Polynomial.X - 1) ^ 1004 
                        + 2 * (Polynomial.X - 2) ^ 1003 
                        + ∑ i in Finset.range 1001, (i + 3) * (Polynomial.X - (i + 3)) ^ (1004 - (i + 3)) 
                        + 1004 * (Polynomial.X - 1004) in
  P.leadingCoeff = 1 →
  P.coeff 1003 = -1002 →
  -P.coeff 1003 / P.leadingCoeff = 1002 :=
by
  sorry

end sum_of_roots_l700_700458


namespace no_solutions_for_cos_cos_cos_cos_eq_sin_sin_sin_sin_l700_700654

theorem no_solutions_for_cos_cos_cos_cos_eq_sin_sin_sin_sin (x : ℝ) :
    ¬ (cos (cos (cos (cos x))) = sin (sin (sin (sin x)))) :=
sorry

end no_solutions_for_cos_cos_cos_cos_eq_sin_sin_sin_sin_l700_700654


namespace cosine_increasing_interval_l700_700304

theorem cosine_increasing_interval (a : ℝ) : 
  (∀ x y : ℝ, -π ≤ x ∧ x < y ∧ y ≤ a → cos x ≤ cos y) ↔ -π < a ∧ a ≤ 0 := 
by
  sorry

end cosine_increasing_interval_l700_700304


namespace floor_eq_solution_l700_700057

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l700_700057


namespace trig_identity_l700_700880

-- Given conditions
variables (α : ℝ) (h_tan : Real.tan (Real.pi - α) = -2)

-- The goal is to prove the desired equality.
theorem trig_identity :
  1 / (Real.cos (2 * α) + Real.cos α * Real.cos α) = -5 / 2 :=
by
  sorry

end trig_identity_l700_700880


namespace johns_weekly_allowance_l700_700956

theorem johns_weekly_allowance (A : ℝ) 
    (arcade_spent : A * 3 / 5 = arcade_spent)
    (toy_store_spent : (A - arcade_spent) * 1 / 3 = toy_store_spent)
    (last_spent : (A - arcade_spent - toy_store_spent) = 0.88) 
    : A = 3.30 :=
by
  -- Definitions and conditions would be presented here
  sorry

end johns_weekly_allowance_l700_700956


namespace permutations_PERCEPTION_l700_700821

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l700_700821


namespace intersection_conditions_range_condition_l700_700627

-- Define the sets A and B
def setA : Set ℝ := {x | 1 ≤ x ∧ x < 4}
def setB (a : ℝ) : Set ℝ := {x | 2 * a ≤ x ∧ x < 3 - a}

-- Define the universal complement of set A
def complementA : Set ℝ := {x | x < 1 ∨ x ≥ 4}

-- Conditions and question 1
theorem intersection_conditions (a : ℝ) (h : a = -2) :
  let B := setB a in
  let expected_x := {x | 1 ≤ x ∧ x < 4} in 
  let expected_complement := {x | (-4 ≤ x ∧ x < 1) ∨ (4 ≤ x ∧ x < 5)} in
    setA ∩ B = expected_x ∧ 
    complementA ∩ B = expected_complement :=
by sorry

-- Conditions and question 2
theorem range_condition (a : ℝ) :
  (∀ x, setA x ∨ setB a x → setA x) ↔ a ≥ 1 / 2 :=
by sorry

end intersection_conditions_range_condition_l700_700627


namespace factorize_x_squared_minus_four_l700_700447

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l700_700447


namespace invitations_per_package_l700_700785

-- Definitions based on conditions in the problem.
def numPackages : Nat := 5
def totalInvitations : Nat := 45

-- Definition of the problem and proof statement.
theorem invitations_per_package :
  totalInvitations / numPackages = 9 :=
by
  sorry

end invitations_per_package_l700_700785


namespace g_sqrt_50_l700_700232

def g (x : ℝ) : ℝ :=
  if x ∈ ℤ then 7 * x + 3 else Real.floor x + 7

theorem g_sqrt_50 : g (Real.sqrt 50) = 14 := by
  sorry

end g_sqrt_50_l700_700232


namespace number_of_initial_owls_l700_700657

-- Define the number of owls initially sitting on the fence.
def initial_owls (n : ℕ) : Prop :=
  n + 2 = 5

-- The final proof problem statement.
theorem number_of_initial_owls : ∃ n : ℕ, initial_owls n ∧ n = 3 :=
by
  have h : initial_owls 3 := by
    unfold initial_owls
    exact rfl
  use 3
  exact ⟨h, rfl⟩

end number_of_initial_owls_l700_700657


namespace sum_of_arcs_eq_180_l700_700779

/-- If three circles with equal radii intersect pairwise at points A, B, C, D, E, and F, 
    then the sum of the arcs AB, CD, and EF is 180 degrees. -/
theorem sum_of_arcs_eq_180 (
  O1 O2 O3 : Type
) (A B C D E F : Type) 
  (equal_radii : ∀ x y z : Type, (x = y) ∧ (y = z)) 
  (intersect_pairwise : ∀ x y, ∃ u v w, x = u ∩ v ∧ y = u ∩ w)
  : (arc_length A B + arc_length C D + arc_length E F) = 180 
:= sorry

end sum_of_arcs_eq_180_l700_700779


namespace roots_of_quadratic_l700_700510

theorem roots_of_quadratic (p q x1 x2 : ℕ) (hp : p + q = 28) (hroots : ∀ x, x^2 + p * x + q = 0 → (x = x1 ∨ x = x2)) (hx1_pos : x1 > 0) (hx2_pos : x2 > 0) :
  (x1 = 30 ∧ x2 = 2) ∨ (x1 = 2 ∧ x2 = 30) :=
sorry

end roots_of_quadratic_l700_700510


namespace curve_is_line_l700_700083

theorem curve_is_line (r θ : ℝ) (h : r = 1 / (2 * sin θ + 3 * cos θ)) :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧ ∀ (x y : ℝ), x * a + y * b + c = 0 ↔ (r = sqrt (x^2 + y^2) ∧ θ = arctan2 y x) :=
by
  sorry

end curve_is_line_l700_700083


namespace women_with_fair_hair_percentage_l700_700360

-- Define the conditions
variables {E : ℝ} (hE : E > 0)

def percent_factor : ℝ := 100

def employees_have_fair_hair (E : ℝ) : ℝ := 0.80 * E
def fair_hair_women (E : ℝ) : ℝ := 0.40 * (employees_have_fair_hair E)

-- Define the target proof statement
theorem women_with_fair_hair_percentage
  (h1 : E > 0)
  (h2 : employees_have_fair_hair E = 0.80 * E)
  (h3 : fair_hair_women E = 0.40 * (employees_have_fair_hair E)):
  (fair_hair_women E / E) * percent_factor = 32 := 
sorry

end women_with_fair_hair_percentage_l700_700360


namespace range_of_k_l700_700101

theorem range_of_k (k : ℝ) :
  (∀ m : ℝ, m ∈ set.Ioo 0 (3 / 2) → (2 / m + 1 / (3 - 2 * m) ≥ k^2 + 2 * k)) ↔ (-3 ≤ k ∧ k ≤ 1) :=
by
  sorry

end range_of_k_l700_700101


namespace base7_to_base10_conversion_l700_700368

theorem base7_to_base10_conversion : 
  let n := 2 * 7^2 + 3 * 7^1 + 1 * 7^0
  in n = 120 := 
by
  sorry

end base7_to_base10_conversion_l700_700368


namespace number_of_satisfying_integers_l700_700025

theorem number_of_satisfying_integers : 
  let S := { x : ℤ | -4 * x ≥ x + 9 ∧ -3 * x ≤ 15 ∧ -5 * x ≥ 3 * x + 21 }
  in S.card = 3 := 
by
  sorry

end number_of_satisfying_integers_l700_700025


namespace call_cost_per_minute_l700_700212

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end call_cost_per_minute_l700_700212


namespace seq_le_n_squared_l700_700231

theorem seq_le_n_squared (a : ℕ → ℕ) (h_increasing : ∀ n, a n < a (n + 1))
  (h_positive : ∀ n, 0 < a n)
  (h_property : ∀ t, ∃ i j, t = a i ∨ t = a i + a j) :
  ∀ n, a n ≤ n^2 :=
by {
  sorry
}

end seq_le_n_squared_l700_700231


namespace problem1_problem2_l700_700124

-- Define sequences and conditions for the problem
variable {a : ℕ → ℝ} {b : ℕ → ℝ} {S : ℕ → ℝ}
variable (λ μ : ℝ)

-- Definitions for problem (1)
def given_conditions_1 := (a 1 = 2) ∧ ∀ n ≥ 2, S n = 4 * a (n - 1)
def b_def := ∀ n ≥ 1, b n = a (n + 1) - 2 * a n

-- Statement for problem (1)
theorem problem1 (λ μ : ℝ) (Hλ : λ = 0) (Hμ : μ = 4) : 
  given_conditions_1 λ μ → b_def λ μ → ∀ n ≥ 1, (finset.range n).sum b = 2^(n + 1) - 2 :=
by
  sorry

-- Definitions for problem (2)
def given_conditions_2 := (a 1 = 2) ∧ (a 2 = 3) ∧ ∀ n ≥ 2, S n = (n / 2) * a n + a (n - 1)

-- Statement for problem (2)
theorem problem2 (λ μ : ℝ) (H_sum : λ + μ = 3/2) : 
    given_conditions_2 λ μ → ∀ n ≥ 1, a n = n + 1 :=
by 
  sorry

end problem1_problem2_l700_700124


namespace perpendicular_lines_b_value_l700_700029

theorem perpendicular_lines_b_value :
  ∀ (b : ℝ), 
    let dir1 := ⟨b, -3, 2⟩ in 
    let dir2 := ⟨2, 3, 4⟩ in 
    (dir1.1 * dir2.1 + dir1.2 * dir2.2 + dir1.3 * dir2.3 = 0) → 
    b = 1 / 2 := 
by 
  intros b dir1 dir2 h
  -- Proof is omitted
  sorry

end perpendicular_lines_b_value_l700_700029


namespace cost_per_minute_l700_700214

-- Conditions as Lean definitions
def initial_credit : ℝ := 30
def remaining_credit : ℝ := 26.48
def call_duration : ℝ := 22

-- Question: How much does a long distance call cost per minute?

theorem cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
by
  sorry

end cost_per_minute_l700_700214


namespace Z_real_axis_Z_first_quadrant_Z_on_line_l700_700108

-- Definitions based on the problem conditions
def Z_real (m : ℝ) : ℝ := m^2 + 5*m + 6
def Z_imag (m : ℝ) : ℝ := m^2 - 2*m - 15

-- Lean statement for the equivalent proof problem

theorem Z_real_axis (m : ℝ) :
  Z_imag m = 0 ↔ (m = -3 ∨ m = 5) := sorry

theorem Z_first_quadrant (m : ℝ) :
  (Z_real m > 0 ∧ Z_imag m > 0) ↔ (m > 5) := sorry

theorem Z_on_line (m : ℝ) :
  (Z_real m + Z_imag m + 5 = 0) ↔ (m = (-5 + Real.sqrt 41) / 2) := sorry

end Z_real_axis_Z_first_quadrant_Z_on_line_l700_700108


namespace sum_coefficients_l700_700879

theorem sum_coefficients (a_0 a_1 a_2 a_3 a_4 a_5 : ℚ) :
  (1 - 2 * (1 : ℚ))^5 = a_0 + a_1 * (1 : ℚ) + a_2 * (1 : ℚ)^2 + a_3 * (1 : ℚ)^3 + a_4 * (1 : ℚ)^4 + a_5 * (1 : ℚ)^5 →
  (1 - 2 * (0 : ℚ))^5 = a_0 →
  a_1 + a_2 + a_3 + a_4 + a_5 = -2 :=
by
  sorry

end sum_coefficients_l700_700879


namespace range_of_a_l700_700139

open Set

theorem range_of_a (a x : ℝ) (h : x^2 - 2 * x + 1 - a^2 < 0) (h2 : 0 < x) (h3 : x < 4) :
  a < -3 ∨ a > 3 :=
sorry

end range_of_a_l700_700139


namespace number_of_real_solutions_l700_700982

theorem number_of_real_solutions (a : ℝ) (h : a ≤ 0) :
  let Δ := 2^2 - 4 * a * 1 in Δ ≥ 0 ∧ (Δ = 0 → (ax^2 + 2x + 1 = 0) has 1 real root) ∧ (Δ > 0 → (ax^2 + 2x + 1 = 0) has 2 distinct real roots) :=
sorry

end number_of_real_solutions_l700_700982


namespace prime_half_sum_l700_700216

theorem prime_half_sum
  (a b c : ℕ)
  (ha : 0 < a)
  (hb : 0 < b)
  (hc : 0 < c)
  (h1 : Nat.Prime (a.factorial + b + c))
  (h2 : Nat.Prime (b.factorial + c + a))
  (h3 : Nat.Prime (c.factorial + a + b)) :
  Nat.Prime ((a + b + c + 1) / 2) := 
sorry

end prime_half_sum_l700_700216


namespace full_day_students_l700_700000

def total_students : ℕ := 80
def percentage_half_day_students : ℕ := 25

theorem full_day_students : 
  (total_students - (total_students * percentage_half_day_students / 100)) = 60 := by
  sorry

end full_day_students_l700_700000


namespace milk_pouring_problem_l700_700723

-- Define the initial state
def milkInitial (Can1 Can2 Pitcher5 Pitcher4 : ℕ) : Prop :=
  Can1 = 80 ∧ Can2 = 80 ∧ Pitcher5 = 0 ∧ Pitcher4 = 0

-- Define what constitutes an operation
inductive Operation
| FillFromCan1 (amount : ℕ) : Operation
| FillFromPitcher5ToPitcher4 (amount : ℕ) : Operation
| PourPitcher4BackToCan1 (amount : ℕ) : Operation
| PourPitcher5ToPitcher4 (amount : ℕ) : Operation
| Others (description : String) : Operation -- Allow defining new operations if needed

-- Define the final desired state
def desiredState (Can1 Can2 Pitcher5 Pitcher4 : ℕ) : Prop :=
  Can1 ≥ 0 ∧ Can2 ≥ 0 ∧ Pitcher5 = 2 ∧ Pitcher4 = 2

-- The statement of the problem
theorem milk_pouring_problem :
  ∃ (steps : list Operation),
  ∀ s1 s2 s3 s4 : ℕ,
  milkInitial s1 s2 s3 s4 →
  list.foldl (λ ⟨C1, C2, P5, P4⟩ op, match op with
    | Operation.FillFromCan1 amount => (C1 - amount, C2, P5 + amount, P4)
    | Operation.FillFromPitcher5ToPitcher4 amount => (C1, C2, P5 - amount, P4 + amount)
    | Operation.PourPitcher4BackToCan1 amount => (C1 + amount, C2, P5, P4 - amount)
    | Operation.PourPitcher5ToPitcher4 amount => (C1, C2, P5 - amount, P4 + amount)
    | Operation.Others _ => (C1, C2, P5, P4) end)
  (s1, s2, s3, s4) steps →
  desiredState s1 s2 s3 s4 := sorry

end milk_pouring_problem_l700_700723


namespace simplify_and_evaluate_l700_700650

-- Defining the variables with given values
def a : ℚ := 1 / 2
def b : ℚ := -2

-- Expression to be simplified and evaluated
def expression : ℚ := (2 * a + b) ^ 2 - (2 * a - b) * (a + b) - 2 * (a - 2 * b) * (a + 2 * b)

-- The main theorem
theorem simplify_and_evaluate : expression = 37 := by
  sorry

end simplify_and_evaluate_l700_700650


namespace jen_age_difference_l700_700205

-- Definitions as conditions given in the problem
def son_present_age := 16
def jen_present_age := 41

-- The statement to be proved
theorem jen_age_difference :
  3 * son_present_age - jen_present_age = 7 :=
by
  sorry

end jen_age_difference_l700_700205


namespace perception_num_permutations_l700_700815

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l700_700815


namespace value_of_x_squared_plus_y_squared_l700_700557

theorem value_of_x_squared_plus_y_squared (x y : ℝ) (h1 : (x + y)^2 = 4) (h2 : x * y = -6) : x^2 + y^2 = 16 :=
by
  sorry

end value_of_x_squared_plus_y_squared_l700_700557


namespace wine_cost_relation_l700_700734

theorem wine_cost_relation (total_cost cork_cost : ℝ) (h1 : total_cost = 2.10) (h2 : cork_cost = 2.05) :
  ∃ wine_cost : ℝ, wine_cost > cork_cost ∧ wine_cost = total_cost - cork_cost :=
by
  -- Define the cost of the wine without the cork
  let wine_cost := total_cost - cork_cost
  use wine_cost
  -- Prove that the wine cost without the cork is more than the cork cost
  split
  -- Proving the first part of the conjunction: wine_cost > cork_cost
  show wine_cost > cork_cost
  sorry
  -- Proving the second part of the conjunction: wine_cost = total_cost - cork_cost
  show wine_cost = total_cost - cork_cost
  sorry

end wine_cost_relation_l700_700734


namespace sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l700_700315

def price_reduction_table : List (ℕ × ℕ) := 
  [(5, 780), (10, 810), (15, 840), (20, 870), (25, 900), (30, 930), (35, 960)]

theorem sales_volume_increase_30_units_every_5_yuan :
  ∀ reduction volume1 volume2, (reduction + 5, volume1) ∈ price_reduction_table →
  (reduction + 10, volume2) ∈ price_reduction_table → volume2 - volume1 = 30 := sorry

theorem initial_sales_volume_750_units :
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (0, 750) ∉ price_reduction_table → 780 - 30 = 750 := sorry

theorem daily_sales_volume_at_540_yuan :
  ∀ P₀ P₁ volume, P₀ = 600 → P₁ = 540 → 
  (5, 780) ∈ price_reduction_table → (10, 810) ∈ price_reduction_table →
  (15, 840) ∈ price_reduction_table → (20, 870) ∈ price_reduction_table →
  (25, 900) ∈ price_reduction_table → (30, 930) ∈ price_reduction_table →
  (35, 960) ∈ price_reduction_table →
  volume = 750 + (P₀ - P₁) / 5 * 30 → volume = 1110 := sorry

end sales_volume_increase_30_units_every_5_yuan_initial_sales_volume_750_units_daily_sales_volume_at_540_yuan_l700_700315


namespace sum_squares_reciprocals_of_roots_l700_700612

theorem sum_squares_reciprocals_of_roots (p q r s : ℝ)
  (hroots : ∀ (w : ℂ), (w ^ 4 + (p : ℂ) * w ^ 3 + (q : ℂ) * w ^ 2 + (r : ℂ) * w + (s : ℂ) = 0)
    → ‖w‖ = 2) :
  ∑ w in {w : ℂ | w ^ 4 + p * w ^ 3 + q * w ^ 2 + r * w + s = 0}, (1 / w)^2 = q / 4 :=
by
  sorry

end sum_squares_reciprocals_of_roots_l700_700612


namespace abs_eq_k_solution_l700_700869

theorem abs_eq_k_solution (k : ℝ) (h : k > 4014) :
  {x : ℝ | |x - 2007| + |x + 2007| = k} = (Set.Iio (-2007)) ∪ (Set.Ioi (2007)) :=
by
  sorry

end abs_eq_k_solution_l700_700869


namespace cost_of_article_l700_700171

noncomputable def find_cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : Prop :=
  C = 168.57

theorem cost_of_article (C G : ℝ) (h1 : C + G = 240) (h2 : C + 1.12 * G = 320) : 
  find_cost_of_article C G h1 h2 :=
by
  sorry

end cost_of_article_l700_700171


namespace question_inequality_l700_700104

theorem question_inequality (x y z : ℝ) :
  x^2 + y^2 + z^2 - x*y - y*z - z*x ≥ max (3/4 * (x - y)^2) (max (3/4 * (y - z)^2) (3/4 * (z - x)^2)) := 
sorry

end question_inequality_l700_700104


namespace floor_ceil_sum_l700_700041

theorem floor_ceil_sum :
  (⌊1.999⌋ + ⌈3.001⌉) = 5 :=
by
  have h1 : ⌊1.999⌋ = 1 := by sorry
  have h2 : ⌈3.001⌉ = 4 := by sorry
  rw [h1, h2]
  exact rfl

end floor_ceil_sum_l700_700041


namespace sufficient_not_necessary_for_increasing_l700_700305

-- Definitions based on conditions from the problem
def sequence (c : ℝ) (n : ℕ) : ℝ := n^2 - c * n

-- The proof problem statement in Lean 4
theorem sufficient_not_necessary_for_increasing (c : ℝ) :
  (∀ n : ℕ, 0 < n → sequence c (n + 1) > sequence c n) → c ≤ 2 :=
sorry

end sufficient_not_necessary_for_increasing_l700_700305


namespace clara_stickers_left_l700_700786

-- Define the initial conditions as parameters or variables
variable start_stickers : ℕ
variable given_to_boy : ℕ
variable additional_stickers : ℕ
variable percent_given_to_classmates : ℕ
variable factor_exchanged_with_friend : ℕ
variable fraction_to_best_friends : ℕ

-- Define the conditions concretely with the given values
def initial_stickers := 100
def stickers_given_to_boy := 10
def stickers_received_from_teacher := 50
def percentage_given_to_classmates := 20 -- 20%
def fraction_exchanged := 1 / 3 -- one-third
def multiplier_new_stickers := 2 -- 2 times
def fraction_given_to_best_friends := 1 / 4 -- one-fourth

-- Calculate the stickers Clara would have left
noncomputable def final_stickers : ℕ :=
  let after_boy := initial_stickers - stickers_given_to_boy
  let after_teacher := after_boy + stickers_received_from_teacher
  let after_classmates := after_teacher - (after_teacher * percentage_given_to_classmates / 100)
  let exchanged_stickers := (after_classmates / 3).toNat
  let after_exchange := (after_classmates - exchanged_stickers + (exchanged_stickers * multiplier_new_stickers).toNat)
  let given_best_friends := (after_exchange / 4).toNat
  after_exchange - (given_best_friends * 3)

-- The final proof statement to show that Clara has 114 stickers left
theorem clara_stickers_left : final_stickers = 114 :=
sorry

end clara_stickers_left_l700_700786


namespace correct_propositions_l700_700589

/-
  We will define conditions and convert them into propositions, then assert
  which propositions are correct. We are interested in proving that
  proposition (2) and (4) are indeed correct given the mathematical definitions.
-/

-- Definitions of propositions based on the given conditions
def proposition1 (P1 P2 : Plane) (L : Line) : Prop := 
  Plane.parallel_to_line P1 L ∧ Plane.parallel_to_line P2 L → Plane.parallel P1 P2

def proposition2 (P1 P2 P3 : Plane) : Prop :=
  Plane.parallel P1 P3 ∧ Plane.parallel P2 P3 → Plane.parallel P1 P2

def proposition3 (L1 L2 L3 : Line) : Prop :=
  Line.perpendicular L1 L3 ∧ Line.perpendicular L2 L3 → Line.parallel L1 L2

def proposition4 (L1 L2 : Line) (P : Plane) : Prop :=
  Line.perpendicular_to_plane L1 P ∧ Line.perpendicular_to_plane L2 P → Line.parallel L1 L2

-- The theorem is to prove that propositions (2) and (4) are correct
theorem correct_propositions (P1 P2 P3 : Plane) (L1 L2 L3 : Line) :
  proposition2 P1 P2 P3 ∧ proposition4 L1 L2 P3 := by
  -- The proof is skipped.
  sorry

end correct_propositions_l700_700589


namespace cricket_target_runs_l700_700184

theorem cricket_target_runs 
  (run_rate1 : ℝ) (run_rate2 : ℝ) (overs : ℕ)
  (h1 : run_rate1 = 5.4) (h2 : run_rate2 = 10.6) (h3 : overs = 25) :
  (run_rate1 * overs + run_rate2 * overs = 400) :=
by sorry

end cricket_target_runs_l700_700184


namespace maximal_product_sum_l700_700258

theorem maximal_product_sum : 
  ∃ (k m : ℕ), 
  k = 671 ∧ 
  m = 2 ∧ 
  2017 = 3 * k + 2 * m ∧ 
  ∀ a b : ℕ, a + b = 2017 ∧ (a < k ∨ b < m) → a * b ≤ 3 * k * 2 * m
:= 
sorry

end maximal_product_sum_l700_700258


namespace apps_addition_vs_deletion_l700_700423

-- Defining the initial conditions
def initial_apps : ℕ := 21
def added_apps : ℕ := 89
def remaining_apps : ℕ := 24

-- The proof problem statement
theorem apps_addition_vs_deletion :
  added_apps - (initial_apps + added_apps - remaining_apps) = 3 :=
by
  sorry

end apps_addition_vs_deletion_l700_700423


namespace liars_positions_l700_700639

/-- Problem setup -/
def tribe : Type := {knight, liar}

def positions := fin 6

def distance (a b : positions) : ℕ := abs (a - b).val

def statement (p : positions) : Prop :=
match p with
| 1 => False -- No statement for person at position 1
| 2 => (abs ((2 : positions) - (nearest_tribesman p)).val = 2)
| 3 => (abs ((3 : positions) - (nearest_tribesman p)).val = 1)
| 4 => False -- No statement for person at position 4
| 5 => False -- No statement for person at position 5
| 6 => (abs ((6 : positions) - (nearest_tribesman p)).val = 3)
| _ => False

/-- Main theorem -/
theorem liars_positions : ∀ (arrangement : positions → tribe), 
    (3 knight = 3 liar) → 
    (statement 2 = True ∧  statement 3 = False ∧ statement 6 = False) →  
    arrangement 3 = liar ∧ arrangement 6 = liar :=
by 
  sorry

end liars_positions_l700_700639


namespace minimum_items_of_A_needed_l700_700688

-- Condition definitions
variable (x : ℕ)

-- Constraints based on the problem statement
def total_items := 10
def cost_A := 20
def cost_B := 50
def max_cost := 350

-- Inequality representing the cost constraint
def cost_inequality (x : ℕ) : Prop := (cost_A * x + cost_B * (total_items - x) ≤ max_cost)

-- The theorem to prove that at least 5 items of product A are needed.
theorem minimum_items_of_A_needed : ∃ (x : ℕ), 5 ≤ x ∧ cost_inequality x :=
by
  intros
  use 5
  unfold cost_inequality
  simp [cost_A, cost_B, total_items, max_cost]
  linarith

end minimum_items_of_A_needed_l700_700688


namespace gcd_n_cubed_minus_27_and_n_plus_3_l700_700491

theorem gcd_n_cubed_minus_27_and_n_plus_3 (n : ℕ) (h : n > 9) : 
  gcd (n^3 - 27) (n + 3) = if (n + 3) % 9 = 0 then 9 else 1 :=
by
  sorry

end gcd_n_cubed_minus_27_and_n_plus_3_l700_700491


namespace area_ratio_cos_prod_l700_700224

theorem area_ratio_cos_prod 
  {A B C : ℝ}  -- Assuming A, B, C are angles of triangle ABC
  (hApos : 0 < A) (hA : A < π)
  (hBpos : 0 < B) (hB : B < π)
  (hCpos : 0 < C) (hC : C < π)
  (hSum : A + B + C = π)  -- Condition that angles of triangle ABC sum to π
  : 
  ∃ (S_ABC S_DEF : ℝ),  -- Existence of areas for triangles ABC and DEF
  S_DEF / S_ABC = 2 * | (cos A) * (cos B) * (cos C) | := by
  sorry

end area_ratio_cos_prod_l700_700224


namespace sum_ratio_l700_700497

noncomputable def S (n : ℕ) : ℝ := sorry -- placeholder definition

def arithmetic_geometric_sum : Prop :=
  S 3 = 2 ∧ S 6 = 18

theorem sum_ratio :
  arithmetic_geometric_sum → S 10 / S 5 = 33 :=
by
  intros h 
  sorry 

end sum_ratio_l700_700497


namespace factorize_difference_of_squares_l700_700444

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
begin
  sorry
end

end factorize_difference_of_squares_l700_700444


namespace ellipse_properties_l700_700141

theorem ellipse_properties :
  (∀ (a b : ℝ), a = 2 * Real.sqrt 3 → b^2 = 4 → 
  let ellipse_eq := ∀ (x y : ℝ), x^2 / 12 + y^2 / 4 = 1 in 
  (ellipse_eq) ∧ 
  (∀ (m : ℝ), abs m < 4 → 
  let chord_length_eq := ∀ (x1 y1 x2 y2 : ℝ),
    y1 = x1 + m ∧ y2 = x2 + m ∧
    (x1 + x2) = -3 * m / 2 ∧ 
    x1 * x2 = (3 * m^2 - 12) / 4 ∧ 
    Real.sqrt 2 * Real.sqrt (-3 / 4 * m^2 + 12) = 3 * Real.sqrt 2 in
  chord_length_eq ∧ 
  let midpoint := (-3 * m / 4, m / 4) in
  let perp_bisector := ∀ (x0 : ℝ), y = 2 → x0 = -2 - m / 2 in
  perp_bisector → (x0 = -3 ∨ x0 = -1))))

end ellipse_properties_l700_700141


namespace stephanie_running_time_l700_700658

theorem stephanie_running_time
  (Speed : ℝ) (Distance : ℝ) (Time : ℝ)
  (h1 : Speed = 5)
  (h2 : Distance = 15)
  (h3 : Time = Distance / Speed) :
  Time = 3 :=
sorry

end stephanie_running_time_l700_700658


namespace quadrilateral_area_l700_700878

noncomputable def area_of_quadrilateral (a : ℝ) : ℝ :=
  let sqrt3 := Real.sqrt 3
  let num := a^2 * (9 - 5 * sqrt3)
  let denom := 12
  num / denom

theorem quadrilateral_area (a : ℝ) : area_of_quadrilateral a = (a^2 * (9 - 5 * Real.sqrt 3)) / 12 := by
  sorry

end quadrilateral_area_l700_700878


namespace range_of_a_to_decreasing_f_l700_700943

noncomputable def f (x a : ℝ) : ℝ := (2 : ℝ)^(x * (x - a))

theorem range_of_a_to_decreasing_f :
  (∀ a x : ℝ, (0 < x ∧ x < 1) → 
    monotone_decreasing (λ x, f x a)) ↔ a ∈ set.Ici 2 := sorry

end range_of_a_to_decreasing_f_l700_700943


namespace standard_deviation_of_data_set_l700_700988

variables {n : ℕ} {x : Fin n → ℝ}
noncomputable def std_dev (x : Fin n → ℝ) : ℝ := Real.sqrt ((1 / n) * (Finset.univ.sum (λ i, (x i - Finset.univ.sum (λ j, x j) / n)^2)))

theorem standard_deviation_of_data_set :
  (∑ i, (x i)^2 = 56) →
  (Finset.univ.sum (λ i, x i) / 40 = Real.sqrt 2 / 2) →
  std_dev (x : Fin 40 → ℝ) = (3 * Real.sqrt 10) / 10 :=
begin
  sorry
end

end standard_deviation_of_data_set_l700_700988


namespace exchange_rate_correct_l700_700590

def exchange_rate_jpy_to_cny (jpy : ℝ) : ℝ := jpy * 7.2 / 100
def amount_jpy := 60000
def expected_cny := 4320

theorem exchange_rate_correct :
  exchange_rate_jpy_to_cny amount_jpy = expected_cny :=
by
  sorry

end exchange_rate_correct_l700_700590


namespace find_circle_center_l700_700362

-- Define the conditions as hypotheses
def line1 (x y : ℝ) : Prop := 5 * x - 2 * y = 40
def line2 (x y : ℝ) : Prop := 5 * x - 2 * y = 10
def line_center_constraint (x y : ℝ) : Prop := 3 * x - 4 * y = 0

-- Define the function for the equidistant line
def line_eq (x y : ℝ) : Prop := 5 * x - 2 * y = 25

-- Prove that the center of the circle satisfying the given conditions is (50/7, 75/14)
theorem find_circle_center (x y : ℝ) 
(h1 : line_eq x y)
(h2 : line_center_constraint x y) : 
(x = 50 / 7 ∧ y = 75 / 14) :=
sorry

end find_circle_center_l700_700362


namespace find_BAC_angle_l700_700662

noncomputable def angle_of_trapezoid (α : ℝ) (AB BC : ℝ) : ℝ :=
  if h : AB = 2 * BC then 
    real.arctan (real.sin α / (2 + real.cos α))
  else 
    0  -- This is just to satisfy Lean's need for a function to always return a value.

theorem find_BAC_angle (α : ℝ) (AB BC : ℝ) (h1 : AB = 2 * BC) :
  angle_of_trapezoid α AB BC = real.arctan (real.sin α / (2 + real.cos α)) :=
by {
  unfold angle_of_trapezoid,
  rw [dif_pos],
  sorry -- Proof
}

end find_BAC_angle_l700_700662


namespace smallest_non_expressible_is_odd_l700_700620

def can_be_expressed (x : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 2^c - 2^d ≠ 0 ∧ x = (2^a - 2^b) / (2^c - 2^d)

def smallest_non_expressible : ℕ :=
  Inf { n : ℕ | ¬ can_be_expressed n ∧ 0 < n }

theorem smallest_non_expressible_is_odd : ¬can_be_expressed smallest_non_expressible ∧ 0 < smallest_non_expressible → odd smallest_non_expressible := 
by
  sorry

end smallest_non_expressible_is_odd_l700_700620


namespace concatenated_number_not_power_of_two_l700_700783

theorem concatenated_number_not_power_of_two :
  ∀ (N : ℕ), (∀ i, 11111 ≤ i ∧ i ≤ 99999) →
  (N ≡ 0 [MOD 11111]) → ¬ ∃ k, N = 2^k :=
by
  sorry

end concatenated_number_not_power_of_two_l700_700783


namespace angle_u_w_90_l700_700952

variables {V : Type*} [inner_product_space ℝ V]
variables (u v w : V)

-- Given conditions
axiom norm_u : ∥u∥ = 2
axiom norm_v : ∥v∥ = 3
axiom norm_w : ∥w∥ = 3
axiom u_vw_eq_zero : u × (u × w) + 2 • v = 0

-- Prove the angle between u and w is 90°
theorem angle_u_w_90 : real.angle (u, w) = real.pi / 2 :=
sorry

end angle_u_w_90_l700_700952


namespace trigonometric_identity_l700_700120

theorem trigonometric_identity 
  (α : ℝ)
  (tan_α : ℝ)
  (h_tan_α : tan_α = 2)
  (h1 : tan α = tan_α) :
  (cos (-π / 2 - α) * tan (π + α) - sin (π / 2 - α)) /
  (cos (3 * π / 2 + α) + cos (π - α)) = -5 := 
sorry

end trigonometric_identity_l700_700120


namespace cos_of_acute_angle_l700_700134

theorem cos_of_acute_angle (θ : ℝ) (hθ1 : 0 < θ ∧ θ < π / 2) (hθ2 : Real.sin θ = 1 / 3) :
  Real.cos θ = 2 * Real.sqrt 2 / 3 :=
by
  -- The proof steps will be filled here
  sorry

end cos_of_acute_angle_l700_700134


namespace rhombus_perimeter_l700_700285

theorem rhombus_perimeter (d1 d2 : ℝ) (h_d1 : d1 = 24) (h_d2 : d2 = 16) : 
  let side := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  in 4 * side = 16 * Real.sqrt 13 :=
by
  sorry

end rhombus_perimeter_l700_700285


namespace nearest_prime_to_2304_divisible_by_23_l700_700089

def is_prime (n : ℕ) : Prop := n > 1 ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def nearest_prime_multiple (x y p : ℕ) : Prop :=
  is_prime p ∧ (p % y = 0) ∧ (∀ q : ℕ, is_prime q → (q % y = 0) → abs (q - x) < abs (p - x) → false)

theorem nearest_prime_to_2304_divisible_by_23 : nearest_prime_multiple 2304 23 2323 :=
by
  sorry

end nearest_prime_to_2304_divisible_by_23_l700_700089


namespace parallelogram_base_length_l700_700082

theorem parallelogram_base_length (Area Height : ℝ) (h1 : Area = 216) (h2 : Height = 18) : 
  Area / Height = 12 := 
by 
  sorry

end parallelogram_base_length_l700_700082


namespace least_three_digit_7_heavy_l700_700766

-- Define what it means for a number to be "7-heavy"
def is_7_heavy(n : ℕ) : Prop := n % 7 > 4

-- Smallest three-digit number
def smallest_three_digit_number : ℕ := 100

-- Least three-digit 7-heavy whole number
theorem least_three_digit_7_heavy : ∃ n, smallest_three_digit_number ≤ n ∧ is_7_heavy(n) ∧ ∀ m, smallest_three_digit_number ≤ m ∧ is_7_heavy(m) → n ≤ m := 
  sorry

end least_three_digit_7_heavy_l700_700766


namespace choose_cooks_l700_700708

theorem choose_cooks (people : Finset ℕ) (cooks : ℕ) (total_people : people.card = 10)
  (cooks_number : cooks = 2) (alice_bob : ∀ (A B : ℕ), A ∈ people → B ∈ people → A ≠ B → A = 0 ∨ B = 1 → False) :
  (people.choose cooks).card - 1 = 44 :=
by
    sorry

end choose_cooks_l700_700708


namespace geometric_sequence_a3a5_l700_700591

theorem geometric_sequence_a3a5 (a : ℕ → ℝ) (r : ℝ) (h1 : ∀ n, a (n + 1) = a n * r) (h2 : a 4 = 5) : a 3 * a 5 = 25 :=
by
  sorry

end geometric_sequence_a3a5_l700_700591


namespace tan_expression_value_l700_700859

-- Definitions consistent with the given problem
def tan_75 : ℝ := Real.tan (75 * Real.pi / 180) 
def tan_150 : ℝ := Real.tan (150 * Real.pi / 180) 

-- Main Goal Statement
theorem tan_expression_value :
  (1 - tan_75^2) / tan_75 = -2 * Real.sqrt 3 :=
by
  sorry

end tan_expression_value_l700_700859


namespace max_distance_dog_origin_l700_700249

/-- 
Given a point (5, 1) and a circle with radius 7 feet centered at that point.
Calculate the maximum distance the dog can be from the origin (0, 0). 
--/
theorem max_distance_dog_origin : 
  let center := (5, 1)
  let radius := 7
  let origin := (0, 0)
  dist origin center + radius = Real.sqrt 26 + 7 :=
by
  Sorry

end max_distance_dog_origin_l700_700249


namespace count_integers_500_to_700_with_digit_sum_18_l700_700545

theorem count_integers_500_to_700_with_digit_sum_18 :
  (∃ f : ℕ → ℕ, (∀ n, 500 ≤ n ∧ n < 700 → (n.digitSum = 18 → f n = 1) ∧ (n.digitSum ≠ 18 → f n = 0)) ∧ 
  (∑ n in finset.Icc 500 699, f n) = 25) :=
by
  sorry

end count_integers_500_to_700_with_digit_sum_18_l700_700545


namespace jade_amount_giraffe_is_120_l700_700245

noncomputable def jade_amount_giraffe : ℝ := sorry

theorem jade_amount_giraffe_is_120 :
  let G := jade_amount_giraffe in
  (∃ (G : ℝ), 
     let giraffe_price := 150,
         elephant_price := 350,
         jade_total := 1920,
         revenue_diff := 400,
         giraffe_jade := G,
         elephant_jade := 2 * G,
         num_giraffes := jade_total / giraffe_jade,
         num_elephants := jade_total / elephant_jade,
         revenue_giraffes := giraffe_price * num_giraffes,
         revenue_elephants := elephant_price * num_elephants
     in revenue_elephants = revenue_giraffes + revenue_diff) →
  jade_amount_giraffe = 120 :=
begin
  sorry
end

end jade_amount_giraffe_is_120_l700_700245


namespace solve_for_x_l700_700265

theorem solve_for_x : 
  ∃ x : ℚ, 2 * x + 3 = 500 - (4 * x + 5 * x) + 7 ∧ x = 504 / 11 := 
by
  refine ⟨504 / 11, _⟩;
  sorry

end solve_for_x_l700_700265


namespace arithmetic_and_sum_l700_700904

noncomputable def arithmetic_seq (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

noncomputable def sum_of_arithmetic_seq (S3 : ℕ) (a4 : ℕ) : Prop :=
  ∃ a1 d, S3 = 3 * a1 + 3 * d ∧ a4 = a1 + 3 * d ∧ ∀ n, arithmetic_seq a1 d n = 2 * n - 1

noncomputable def b_seq (an : ℕ → ℕ) (n : ℕ) : ℝ :=
  1 / (an n * an (n + 1))

noncomputable def sum_b_seq (sum_S3 : ℕ → ℝ) (n : ℕ) : ℝ :=
  sum_S3 n = (n : ℝ) / (2 * n + 1)

theorem arithmetic_and_sum (S3 a4 : ℕ) (h1 : S3 = 9) (h2 : a4 = 7) :
  (sum_of_arithmetic_seq S3 a4 ∧ ∀ n, sum_b_seq (λ n, ∑ i in finset.range n, b_seq (λ n, 2 * n - 1) i) n) :=
by
  sorry

end arithmetic_and_sum_l700_700904


namespace minimize_a_l700_700220

open Polynomial

noncomputable def smallest_possible_value_a (P : ℤ[X]) (a : ℤ) : Prop :=
  (∀ x : ℤ, x ∈ {1, 2, 3, 4} → P.eval x = a) ∧
  (∀ x : ℤ, x ∈ {-1, -2, -3, -4} → P.eval x = -a) ∧
  (∀ n : ℤ, 0 < n → a = n → ∃ Q : ℤ[X], P = (X - 1) * (X - 2) * (X - 3) * (X - 4) * Q + C a)

theorem minimize_a (P : ℤ[X]) (a : ℤ) (h1 : ∀ (x : ℤ), x ∈ {1, 2, 3, 4} → P.eval x = a)
  (h2 : ∀ (x : ℤ), x ∈ {-1, -2, -3, -4} → P.eval x = -a) :
  ∃ n, n = 1680 ∧ a = n :=
begin
  sorry
end

end minimize_a_l700_700220


namespace triangle_area_is_4_l700_700331

-- Define the lines
def line1 (x : ℝ) : ℝ := 4
def line2 (x : ℝ) : ℝ := 2 + x
def line3 (x : ℝ) : ℝ := 2 - x

-- Define intersection points
def intersection1 : ℝ × ℝ := (2, 4)
def intersection2 : ℝ × ℝ := (-2, 4)
def intersection3 : ℝ × ℝ := (0, 2)

-- Function to calculate the area of a triangle using its vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs ((A.1 * B.2 + B.1 * C.2 + C.1 * A.2) - (A.2 * B.1 + B.2 * C.1 + C.2 * A.1))

-- Statement of the proof problem
theorem triangle_area_is_4 :
  ∀ A B C : ℝ × ℝ, A = intersection1 → B = intersection2 → C = intersection3 →
  triangle_area A B C = 4 := by
  sorry

end triangle_area_is_4_l700_700331


namespace problem1_problem2_l700_700643

open Real

-- Proof problem 1: Given condition and the required result.
theorem problem1 (x y : ℝ) (h : (x^2 + y^2 - 4) * (x^2 + y^2 + 2) = 7) :
  x^2 + y^2 = 5 :=
sorry

-- Proof problem 2: Solve the polynomial equation.
theorem problem2 (x : ℝ) :
  (x = sqrt 2 ∨ x = -sqrt 2 ∨ x = 2 ∨ x = -2) ↔ (x^4 - 6 * x^2 + 8 = 0) :=
sorry

end problem1_problem2_l700_700643


namespace min_value_of_f_l700_700854

open Real

def f (x : ℝ) : ℝ :=
  sqrt (x ^ 2 + (1 - x) ^ 2) + sqrt (2 * (1 - x) ^ 2)

theorem min_value_of_f :
  ∃ x : ℝ, (∀ y : ℝ, f y ≥ f x) ∧ f x = 1 := 
by
  sorry

end min_value_of_f_l700_700854


namespace hypotenuse_length_l700_700325

noncomputable def length_of_hypotenuse (PQ PR : ℝ) (ratio : ℝ) (QN MR : ℝ) : ℝ :=
  if PQ/4 = QN ∧ PR * (3/4) = MR then 
    real.sqrt (PQ^2 + PR^2)
  else 
    0

theorem hypotenuse_length 
  (PQ PR : ℝ)
  (ratio : ℝ)
  (QN : ℝ := 20)
  (MR : ℝ := 36)
  (h1 : ratio = 1/3)
  (h2 : PR * (3/4) = MR)
  (h3 : PQ / 4 = QN)
  : PQ^2 + PR^2 = 1596 :=
sorry

def hypotenuse_QR : ℝ := 2 * real.sqrt 399

end hypotenuse_length_l700_700325


namespace division_ways_12_hour_period_l700_700550

def total_seconds_in_12_hours := 43200

def divisors_count (n : ℕ) :=
  (finset.divisors n).card

theorem division_ways_12_hour_period :
  divisors_count total_seconds_in_12_hours = 84 := 
sorry

end division_ways_12_hour_period_l700_700550


namespace sum_series_equals_zero_l700_700618

def f (x : ℝ) : ℝ := x^2 * (1 - x)^2

theorem sum_series_equals_zero : 
  ∑ k in ((finset.range 2021).filter (λ n, 2 ≤ n ∧ n ≤ 2020)), 
    ((-1)^n) * f ((n : ℝ) / 2021) = 0 := 
sorry

end sum_series_equals_zero_l700_700618


namespace problem_part_I_problem_part_II_l700_700484

-- Define the function f(x) given by the problem
def f (x a : ℝ) : ℝ := x^2 - 2 * a * x + 5

-- Define the conditions for part (Ⅰ)
def conditions_part_I (a x : ℝ) : Prop :=
  (1 ≤ x ∧ x ≤ a) ∧ (1 ≤ f x a ∧ f x a ≤ a)

-- Lean statement for part (Ⅰ)
theorem problem_part_I (a : ℝ) (h : a > 1) :
  (∀ x, conditions_part_I a x) → a = 2 := by sorry

-- Define the conditions for part (Ⅱ)
def decreasing_on_interval (a : ℝ) : Prop :=
  ∀ x y, x ≤ y ∧ y ≤ 2 → f x a ≥ f y a

def abs_difference_condition (a : ℝ) : Prop :=
  ∀ x1 x2, 1 ≤ x1 ∧ x1 ≤ a + 1 ∧ 1 ≤ x2 ∧ x2 ≤ a + 1 → |f x1 a - f x2 a| ≤ 4

-- Lean statement for part (Ⅱ)
theorem problem_part_II (a : ℝ) (h : a > 1) :
  (decreasing_on_interval a) ∧ (abs_difference_condition a) → (2 ≤ a ∧ a ≤ 3) := by sorry

end problem_part_I_problem_part_II_l700_700484


namespace least_three_digit_7_heavy_l700_700776

-- Define what it means to be a 7-heavy number
def is_7_heavy (n : ℕ) : Prop :=
  n % 7 > 4

-- Define the property of being three-digit
def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

-- The statement to prove
theorem least_three_digit_7_heavy : ∃ n, is_7_heavy n ∧ is_three_digit n ∧ ∀ m, is_7_heavy m ∧ is_three_digit m → n ≤ m :=
begin
  use [104],
  split,
  { -- Proof that 104 is 7-heavy
    show is_7_heavy 104,
    simp [is_7_heavy], -- Calculation: 104 % 7 = 6 which is > 4
    norm_num,
  },
  split,
  { -- Proof that 104 is a three-digit number
    show is_three_digit 104,
    simp [is_three_digit],
    norm_num,
  },
  { -- Proof that 104 is the smallest 7-heavy three-digit number
    intros m hm,
    cases hm with hm1 hm2,
    suffices : 104 ≤ m,
    exact this,
    calc 104 ≤ 100 + 7 - 1 : by norm_num
        ... ≤ m            : by linarith [hm2.left, hm2.right],
    sorry,
  }
sorry

end least_three_digit_7_heavy_l700_700776


namespace polynomial_form_l700_700839

noncomputable def polynomial_with_quotient_derivative_as_polynomial (p : Polynomial ℝ) (n : ℕ) : Prop :=
  ∀ x : ℝ, Polynomial.quotient p (Polynomial.derivative p x) = Polynomial ℝ

theorem polynomial_form
  {p : Polynomial ℝ} {n : ℕ} (hp : polynomial_with_quotient_derivative_as_polynomial p n)
  (hdeg : p.degree = n) : 
  ∃ (a_0 : ℝ) (d : ℝ), 
  p = Polynomial.C a_0 * (Polynomial.X + Polynomial.C d) ^ n :=
sorry

end polynomial_form_l700_700839


namespace factorize_difference_of_squares_l700_700440

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l700_700440


namespace discount_percentage_is_30_l700_700425

theorem discount_percentage_is_30 
  (price_per_pant : ℝ) (num_of_pants : ℕ)
  (price_per_sock : ℝ) (num_of_socks : ℕ)
  (total_spend_after_discount : ℝ)
  (original_pants_price := num_of_pants * price_per_pant)
  (original_socks_price := num_of_socks * price_per_sock)
  (original_total_price := original_pants_price + original_socks_price)
  (discount_amount := original_total_price - total_spend_after_discount)
  (discount_percentage := (discount_amount / original_total_price) * 100) :
  (price_per_pant = 110) ∧ 
  (num_of_pants = 4) ∧ 
  (price_per_sock = 60) ∧ 
  (num_of_socks = 2) ∧ 
  (total_spend_after_discount = 392) →
  discount_percentage = 30 := by
  sorry

end discount_percentage_is_30_l700_700425


namespace gender_in_question_is_girl_l700_700185

theorem gender_in_question_is_girl
  (p_boy : ℝ)
  (p_gender : ℝ)
  (h1 : p_boy = 0.5)
  (h2 : p_gender = 0.5)
  (proportion : ℕ → ℕ → Prop)
  (h3 : ∀ boys girls, proportion boys girls → boys = girls): 
  (∃ gender : Type, proportion 1 1 → gender = "girl") :=
by
  sorry

end gender_in_question_is_girl_l700_700185


namespace rectangular_eqn_of_curve_C_l700_700513

-- Define the given polar equation
def polar_eqn (ρ θ : ℝ) : Prop :=
  ρ = 2 * sqrt 2 * cos (θ - π / 4)

-- Define the polar-rectangular coordinate transformation formulas
def polar_to_rectangular (ρ θ x y : ℝ) : Prop :=
  x = ρ * cos θ ∧ y = ρ * sin θ

-- The theorem we want to prove
theorem rectangular_eqn_of_curve_C (x y : ℝ) :
  (∃ ρ θ, polar_eqn ρ θ ∧ polar_to_rectangular ρ θ x y) →
  x^2 + y^2 - 2x - 2y = 0 :=
by
  -- Insert proof here
  sorry

end rectangular_eqn_of_curve_C_l700_700513


namespace value_of_M_l700_700169

theorem value_of_M (M : ℝ) (H : 0.25 * M = 0.55 * 1500) : M = 3300 := 
by
  sorry

end value_of_M_l700_700169


namespace largest_power_of_5_in_e_q_l700_700781

-- Define the sum q as specified in the conditions
noncomputable def q : ℝ := ∑ k in (finset.range 8).map (λ i, i + 1), (k : ℝ) * real.log k

-- Define e^q
noncomputable def e_q : ℝ := real.exp q

-- Statement of the problem
theorem largest_power_of_5_in_e_q :
  ∀ n : ℕ, (5 ^ 5 ∣ e_q) ∧ (∀ m > 5, ¬ 5 ^ m ∣ e_q) :=
by
  sorry

end largest_power_of_5_in_e_q_l700_700781


namespace elvins_fixed_monthly_charge_l700_700844

-- Definition of the conditions
def january_bill (F C_J : ℝ) : Prop := F + C_J = 48
def february_bill (F C_J : ℝ) : Prop := F + 2 * C_J = 90

theorem elvins_fixed_monthly_charge (F C_J : ℝ) (h_jan : january_bill F C_J) (h_feb : february_bill F C_J) : F = 6 :=
by
  sorry

end elvins_fixed_monthly_charge_l700_700844


namespace range_of_f_l700_700907

open Set

noncomputable def f (x : ℝ) (k : ℝ) (c : ℝ) := x^k + c

theorem range_of_f (k : ℝ) (c : ℝ) (h : 0 < k) : 
  range (λ x : ℝ, f x k c) ∩ Ioi 0 = Icc c (⊤ : ℝ) :=
by sorry

end range_of_f_l700_700907


namespace smallest_k_multiple_of_180_l700_700202

theorem smallest_k_multiple_of_180 (k : ℕ) (h : ∀ k : ℕ, 1^2 + 2^2 + 3^2 + ... + k^2 = k * (k + 1) * (2 * k + 1) / 6) : 
  ∃ k : ℕ, k = 112 ∧ k * (k + 1) * (2 * k + 1) % 1080 = 0 :=
by {
  let k := 112
  have h_k_multiple : (k * (k + 1) * (2 * k + 1)) % 1080 = 0 := sorry,
  use k,
  split,
  exact rfl,
  exact h_k_multiple,
  sorry 
}

end smallest_k_multiple_of_180_l700_700202


namespace matrix_power_4_l700_700415

open Matrix

def A : Matrix (fin 2) (fin 2) ℤ :=
  ![![2, -2], ![2, -1]]

theorem matrix_power_4 :
  A ^ 4 = ![![ -8, 8 ], ![ 0, 3 ]] :=
by
  sorry

end matrix_power_4_l700_700415


namespace quadratic_polynomial_with_root_and_leading_coeff_l700_700420

theorem quadratic_polynomial_with_root_and_leading_coeff 
  (z : ℂ) (a b c : ℝ) (h_root : z = 3 + 2 * complex.I) (leading_coeff : a = 2)
  (h_real_coeff : a, b, c ∈ ℝ) :
  a * X^2 + b * X + c = 2 * X^2 - 12 * X + 26 :=
by sorry

end quadratic_polynomial_with_root_and_leading_coeff_l700_700420


namespace factorize_difference_of_squares_l700_700441

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
sorry

end factorize_difference_of_squares_l700_700441


namespace sum_of_possible_integers_l700_700503

theorem sum_of_possible_integers (n : ℤ) (h : 0 < 3 * n ∧ 3 * n < 27) : 
  (∑ k in {k : ℤ | 0 < k ∧ k < 9}, k) = 36 :=
sorry

end sum_of_possible_integers_l700_700503


namespace mass_percentage_Br_l700_700006

-- Conditions
def mass_BaBr2 : ℝ := 8
def mass_SrBr2 : ℝ := 4
def molar_mass_BaBr2 : ℝ := 137.327 + 2 * 79.904
def molar_mass_SrBr2 : ℝ := 87.62 + 2 * 79.904

-- Definition of the mass of Br in each compound
def mass_Br_BaBr2 : ℝ := (mass_BaBr2 / molar_mass_BaBr2) * (2 * 79.904)
def mass_Br_SrBr2 : ℝ := (mass_SrBr2 / molar_mass_SrBr2) * (2 * 79.904)
def total_mass_Br : ℝ := mass_Br_BaBr2 + mass_Br_SrBr2
def total_mass_mixture : ℝ := mass_BaBr2 + mass_SrBr2

-- Proof that the mass percentage of Br is 57.42%
theorem mass_percentage_Br : 
  (total_mass_Br / total_mass_mixture) * 100 = 57.42 := 
by 
  sorry

end mass_percentage_Br_l700_700006


namespace sum_of_number_and_preceding_l700_700680

theorem sum_of_number_and_preceding (n : ℤ) (h : 6 * n - 2 = 100) : n + (n - 1) = 33 :=
by {
  sorry
}

end sum_of_number_and_preceding_l700_700680


namespace books_at_end_of_year_l700_700240

def init_books : ℕ := 72
def monthly_books : ℕ := 12 -- 1 book each month for 12 months
def books_bought1 : ℕ := 5
def books_bought2 : ℕ := 2
def books_gift1 : ℕ := 1
def books_gift2 : ℕ := 4
def books_donated : ℕ := 12
def books_sold : ℕ := 3

theorem books_at_end_of_year :
  init_books + monthly_books + books_bought1 + books_bought2 + books_gift1 + books_gift2 - books_donated - books_sold = 81 :=
by
  sorry

end books_at_end_of_year_l700_700240


namespace find_x_minus_4y_l700_700963

theorem find_x_minus_4y (x y : ℝ) (h1 : x + y = 5) (h2 : 2 * x - 3 * y = 10) : x - 4 * y = 5 :=
by 
  sorry

end find_x_minus_4y_l700_700963


namespace permutations_PERCEPTION_l700_700824

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l700_700824


namespace solve_for_x_l700_700266

theorem solve_for_x : 
  (∃ x : ℚ, (1/8 : ℚ)^(3 * x + 6) = 32^(x + 3) ∧ x = -33/14) := 
sorry

end solve_for_x_l700_700266


namespace least_three_digit_7_heavy_l700_700770

/-- A number is 7-heavy if the remainder when the number is divided by 7 is greater than 4. -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The statement to be proved: The least three-digit 7-heavy number is 104. -/
theorem least_three_digit_7_heavy : ∃ n, 100 ≤ n ∧ n < 1000 ∧ is_7_heavy(n) ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ is_7_heavy(m) → n ≤ m :=
begin
    use 104,
    split,
    { exact dec_trivial, },
    split,
    { exact dec_trivial, },
    split,
    { change 104 % 7 > 4,
      exact dec_trivial, },
    { intros m h1 h2,
      sorry
    }
end

end least_three_digit_7_heavy_l700_700770


namespace cosine_inequality_l700_700462

theorem cosine_inequality (a b c : ℝ) : ∃ x : ℝ, 
    a * Real.cos x + b * Real.cos (3 * x) + c * Real.cos (9 * x) ≥ (|a| + |b| + |c|) / 2 :=
sorry

end cosine_inequality_l700_700462


namespace num_distinct_combinations_l700_700754

-- Define the conditions
def num_dials : Nat := 4
def digits : List Nat := List.range 10  -- Digits from 0 to 9

-- Define what it means for a combination to have distinct digits
def distinct_digits (comb : List Nat) : Prop :=
  comb.length = num_dials ∧ comb.Nodup

-- The main statement for the theorem
theorem num_distinct_combinations : 
  ∃ (n : Nat), n = 5040 ∧ ∀ comb : List Nat, distinct_digits comb → comb.length = num_dials →
  (List.permutations digits).length = n :=
by
  sorry

end num_distinct_combinations_l700_700754


namespace total_hiking_distance_l700_700248

def saturday_distance : ℝ := 8.2
def sunday_distance : ℝ := 1.6
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ := saturday_distance + sunday_distance

theorem total_hiking_distance :
  total_distance saturday_distance sunday_distance = 9.8 :=
by
  -- The proof is omitted
  sorry

end total_hiking_distance_l700_700248


namespace ellipse_dimensions_constant_ratio_l700_700144

noncomputable def ellipse_params (a b : ℝ) (h1 : a > b) (h2 : b > 0) : ℝ × ℝ :=
  (a, b)

theorem ellipse_dimensions (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : (sqrt 3) / 2 = sqrt (1 - (b / a) ^ 2) )
  (h4 : (x y : ℝ) → (x^2 + (y - 3/2)^2 = 1) 
    → (y^2 / a^2 + x^2 / b^2 ≤ 1) 
    → (|arc_length x y (0, 3/2)| = 2π/3)) :
  a = 2 ∧ b = 1 :=
sorry

theorem constant_ratio (a b : ℝ) (h1 : a = 2) (h2 : b = 1) :
  (∃ (A B C D : ℝ × ℝ), (|AB_length A B (0, sqrt 3) | = a * b / a) ∧ (|CD_length C D (0, sqrt 3) | = 2 * a) ∧ ( 1 / |AB_length A B (0, sqrt 3) | + 1 / |CD_length C D (0, sqrt 3) | = 5 / 4)) :=
sorry

end ellipse_dimensions_constant_ratio_l700_700144


namespace floor_eq_solution_l700_700056

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l700_700056


namespace symmetry_of_function_neg_x_cubed_l700_700665

theorem symmetry_of_function_neg_x_cubed : 
  ∀ (x : ℝ),  f(-x) = -f(x) := 
by
  sorry

end symmetry_of_function_neg_x_cubed_l700_700665


namespace total_space_after_compaction_correct_l700_700635

noncomputable def problem : Prop :=
  let num_small_cans := 50
  let num_large_cans := 50
  let small_can_size := 20
  let large_can_size := 40
  let small_can_compaction := 0.30
  let large_can_compaction := 0.40
  let small_cans_compacted := num_small_cans * small_can_size * small_can_compaction
  let large_cans_compacted := num_large_cans * large_can_size * large_can_compaction
  let total_space_after_compaction := small_cans_compacted + large_cans_compacted
  total_space_after_compaction = 1100

theorem total_space_after_compaction_correct :
  problem :=
  by
    unfold problem
    sorry

end total_space_after_compaction_correct_l700_700635


namespace students_not_enrolled_in_either_l700_700995

variable (total_students french_students german_students both_students : ℕ)

theorem students_not_enrolled_in_either (h1 : total_students = 60)
                                        (h2 : french_students = 41)
                                        (h3 : german_students = 22)
                                        (h4 : both_students = 9) :
    total_students - (french_students + german_students - both_students) = 6 := by
  sorry

end students_not_enrolled_in_either_l700_700995


namespace factorize_x_squared_minus_four_l700_700449

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l700_700449


namespace initial_pepper_amount_l700_700004
-- Import the necessary libraries.

-- Declare the problem as a theorem.
theorem initial_pepper_amount (used left : ℝ) (h₁ : used = 0.16) (h₂ : left = 0.09) :
  used + left = 0.25 :=
by
  -- The proof is not required here.
  sorry

end initial_pepper_amount_l700_700004


namespace sum_four_least_tau_equals_eight_l700_700219

def tau (n : ℕ) : ℕ := n.divisors.card

theorem sum_four_least_tau_equals_eight :
  ∃ n1 n2 n3 n4 : ℕ, 
    tau n1 + tau (n1 + 1) = 8 ∧ 
    tau n2 + tau (n2 + 1) = 8 ∧
    tau n3 + tau (n3 + 1) = 8 ∧
    tau n4 + tau (n4 + 1) = 8 ∧
    n1 + n2 + n3 + n4 = 80 := 
sorry

end sum_four_least_tau_equals_eight_l700_700219


namespace original_rectangle_area_l700_700297

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l700_700297


namespace limit_point_of_pencil_of_circles_l700_700254

-- Define the power of a point O relative to a circle
def power (O A : Point) (R : ℝ) : ℝ := (dist O A) ^ 2 - R ^ 2

-- Orthogonality condition for two circles
def orthogonal_circles (A1 A2 : Point) (R1 R2 : ℝ) (d : ℝ) : Prop := 
  2 * dist A1 A2 = R1 ^ 2 + R2 ^ 2 - d ^ 2

-- Prove the main theorem as a statement in Lean 4
theorem limit_point_of_pencil_of_circles (O : Point) (circles : set Circle) :
  (∀ (C ∈ circles), power O C.center C.radius = 0) ↔ 
  (∀ (C1 C2 ∈ circles), orthogonal_circles C1.center C2.center C1.radius C2.radius (dist C1.center C2.center) → 
  O ∈ Circle.circumference C1 ∧ O ∈ Circle.circumference C2) := by 
  sorry

end limit_point_of_pencil_of_circles_l700_700254


namespace find_pairs_l700_700453

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

theorem find_pairs (a n : ℕ) (h1 : a ≥ n) (h2 : is_power_of_two ((a + 1)^n + a - 1)) :
  (a = 4 ∧ n = 3) ∨ (∃ k : ℕ, a = 2^k ∧ n = 1) :=
by
  sorry

end find_pairs_l700_700453


namespace proof_find_a_b_proof_exponential_function_cases_l700_700526

noncomputable def find_a_b (a b : ℝ) : Prop :=
  (a > 0) ∧ (a ≠ 1) ∧ 
  (6 = b * a) ∧ 
  (24 = b * a^3) ∧ 
  (a = 2) ∧ (b = 3)

noncomputable def exponential_function_cases (a : ℝ) : Prop :=
  (a - 1 / a = 1 → a = (1 + Real.sqrt 5) / 2) ∧ 
  (1 / a - a = 1 → a = (-1 + Real.sqrt 5) / 2)

theorem proof_find_a_b (a b : ℝ) : find_a_b a b :=
begin
  sorry
end

theorem proof_exponential_function_cases (a : ℝ) : exponential_function_cases a :=
begin
  sorry
end

end proof_find_a_b_proof_exponential_function_cases_l700_700526


namespace frustum_slant_height_l700_700514

theorem frustum_slant_height (r1 r2 V : ℝ) (h l : ℝ) 
    (H1 : r1 = 2) (H2 : r2 = 6) (H3 : V = 104 * π)
    (H4 : V = (1/3) * π * h * (r1^2 + r2^2 + r1 * r2)) 
    (H5 : h = 6)
    (H6 : l = Real.sqrt (h^2 + (r2 - r1)^2)) :
    l = 2 * Real.sqrt 13 :=
by sorry

end frustum_slant_height_l700_700514


namespace original_rectangle_area_l700_700298

theorem original_rectangle_area
  (A : ℝ)
  (h1 : ∀ (a : ℝ), a = 2 * A)
  (h2 : 4 * A = 32) : 
  A = 8 := 
by
  sorry

end original_rectangle_area_l700_700298


namespace pencils_across_diameter_l700_700968

theorem pencils_across_diameter (r : ℝ) (pencil_length_inch : ℕ) (pencils : ℕ) :
  r = 14 ∧ pencil_length_inch = 6 ∧ pencils = 56 → 
  let d := 2 * r in
  let pencil_length_feet := pencil_length_inch / 12 in
  pencils = d / pencil_length_feet :=
begin
  sorry -- Proof is skipped
end

end pencils_across_diameter_l700_700968


namespace bounded_area_is_25_over_6_l700_700397

noncomputable def bounded_area : ℝ :=
  ∫ x in 1..6, ((2*x - 2) - (x^2 - 5*x + 4))

theorem bounded_area_is_25_over_6 : bounded_area = 25 / 6 := by
  sorry

end bounded_area_is_25_over_6_l700_700397


namespace solution_set_l700_700071

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l700_700071


namespace max_square_division_l700_700457

theorem max_square_division (m : ℕ) :
  (∃ (r : ℕ → ℕ × ℕ), (∀ i, 1 ≤ i ∧ i ≤ 7 → let ⟨a, b⟩ := r i in 1 ≤ a ∧ a ≤ 14 ∧ 1 ≤ b ∧ b ≤ 14) ∧
  ∑ i in Finset.range 7, let ⟨a, b⟩ := r i in a * b = m * m) → m ≤ 22 := sorry

end max_square_division_l700_700457


namespace games_played_in_tournament_l700_700568

def number_of_games (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem games_played_in_tournament : number_of_games 18 = 153 :=
  by
    sorry

end games_played_in_tournament_l700_700568


namespace smallest_four_digit_divisible_by_3_and_8_l700_700332

theorem smallest_four_digit_divisible_by_3_and_8 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 3 = 0 ∧ n % 8 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 3 = 0 ∧ m % 8 = 0 → n ≤ m := by
  sorry

end smallest_four_digit_divisible_by_3_and_8_l700_700332


namespace factorize_difference_of_squares_l700_700446

theorem factorize_difference_of_squares (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) := 
begin
  sorry
end

end factorize_difference_of_squares_l700_700446


namespace limit_of_S_l700_700463

noncomputable def S (n : ℕ) : ℝ :=
  ∑ k in Finset.range 2010, (Real.cos (k.factorial * Real.pi / 2010)) ^ n

theorem limit_of_S :
  Tendsto (λ n, S n) atTop (𝓝 1944) := sorry

end limit_of_S_l700_700463


namespace cinematic_academy_member_count_l700_700691

theorem cinematic_academy_member_count (M : ℝ) 
  (h : (1 / 4) * M = 192.5) : M = 770 := 
by 
  -- proof omitted
  sorry

end cinematic_academy_member_count_l700_700691


namespace sum_a_l700_700226

def a (n : ℕ) : ℝ := 1 / ((n + 1) * real.sqrt n + n * real.sqrt (n + 1))

theorem sum_a (h : ∀ n, 1 ≤ n ∧ n ≤ 99) : 
  ∑ n in (finset.range 100).filter (λ n, 1 ≤ n ∧ n ≤ 99), a n = 9 / 10 :=
by
  sorry

end sum_a_l700_700226


namespace cos_sin_alpha_values_l700_700891

theorem cos_sin_alpha_values (m : ℝ) (cos α sin α : ℝ) 
  (h1 : α ∈ set.range (λ θ : ℝ, real.angle (θ : ℝ))) 
  (h2 : sin α = (m * sqrt 2) / 4) : 
  (cos α = -1 ∨ cos α = - (sqrt 6) / 4) ∧ 
  (sin α = 0 ∨ sin α = sqrt 10 / 4 ∨ sin α = - sqrt 10 / 4) :=
sorry

end cos_sin_alpha_values_l700_700891


namespace find_slope_example_l700_700871

noncomputable def find_slope (c1 c2 c3 c4 : ℝ × ℝ) (r : ℝ) : ℝ :=
  sorry

theorem find_slope_example :
  ∃ m : ℝ, ∀ (c1 c2 c3 c4 : ℝ × ℝ) (r : ℝ),
    c1 = (14, 92) →
    c2 = (17, 76) →
    c3 = (19, 84) →
    c4 = (25, 90) →
    r = 5 →
    let c1' := (c1.1 - 14, c1.2 - 76),
        c2' := (c2.1 - 14, c2.2 - 76),
        c3' := (c3.1 - 14, c3.2 - 76),
        c4' := (c4.1 - 14, c4.2 - 76) in
    let line := λ x, m * x - 3 * m in
    let dist (p : ℝ × ℝ) := abs (p.2 - line p.1) / sqrt (m^2 + 1) in
    (dist c1' + dist c2' + dist c3' + dist c4') / 4 = (dist c1' + dist c2' + dist c3' + dist c4') / 4 →
    |m| = find_slope c1 c2 c3 c4 5 :=
sorry

end find_slope_example_l700_700871


namespace range_of_a_l700_700935

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x ∈ Ioo (0:ℝ) (1:ℝ), f x = 2^(x*(x-a)) ∧ monotone_decreasing_on f (Ioo (0:ℝ) (1:ℝ))) :
  a ∈ set.Ici (2 : ℝ) := sorry

end range_of_a_l700_700935


namespace floor_equation_solution_l700_700044

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l700_700044


namespace price_increase_eq_20_percent_l700_700359

theorem price_increase_eq_20_percent (a x : ℝ) (h : a * (1 + x) * (1 + x) = a * 1.44) : x = 0.2 :=
by {
  -- This part will contain the proof steps.
  sorry -- Placeholder
}

end price_increase_eq_20_percent_l700_700359


namespace permutations_PERCEPTION_l700_700820

-- Define the word "PERCEPTION" and its letter frequencies
def word : String := "PERCEPTION"

def freq_P : Nat := 2
def freq_E : Nat := 2
def freq_R : Nat := 1
def freq_C : Nat := 1
def freq_T : Nat := 1
def freq_I : Nat := 1
def freq_O : Nat := 1
def freq_N : Nat := 1

-- Define the total number of letters in the word
def total_letters : Nat := 10

-- Calculate the number of permutations for the multiset
def permutations : Nat :=
  total_letters.factorial / (freq_P.factorial * freq_E.factorial)

-- Proof problem
theorem permutations_PERCEPTION :
  permutations = 907200 :=
by
  sorry

end permutations_PERCEPTION_l700_700820


namespace intersection_complement_l700_700157

open Set

-- Define sets A and B as provided in the conditions
def A : Set ℝ := {x | x ≤ 3}
def B : Set ℝ := {x | x < 2}

-- Define the theorem to prove the question is equal to the answer given the conditions
theorem intersection_complement : (A ∩ compl B) = {x | 2 ≤ x ∧ x ≤ 3} := by
  sorry

end intersection_complement_l700_700157


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700051

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700051


namespace perception_permutations_count_l700_700827

theorem perception_permutations_count :
  let n := 10
  let freq_P := 2
  let freq_E := 2
  let factorial := λ x : ℕ, (Nat.factorial x)
  factorial n / (factorial freq_P * factorial freq_E) = 907200 :=
by sorry

end perception_permutations_count_l700_700827


namespace slope_angle_correct_l700_700309

def parametric_line (α : ℝ) : Prop :=
  α = 50 * (Real.pi / 180)

theorem slope_angle_correct : ∀ (t : ℝ),
  parametric_line 50 →
  ∀ α : ℝ, α = 140 * (Real.pi / 180) :=
by
  intro t
  intro h
  intro α
  sorry

end slope_angle_correct_l700_700309


namespace find_solutions_l700_700234

def h (x : ℝ) : ℝ :=
if x < 2 then 4 * x + 10 else 3 * x - 12

theorem find_solutions (x : ℝ) : h x = 6 ↔ x = -1 ∨ x = 6 :=
by
  sorry

end find_solutions_l700_700234


namespace problem_l700_700921

noncomputable def f (x : ℝ) : ℝ := 1 - 2 * (Real.sin x) ^ 2

-- Questions to be proved:
-- 1. f is monotonically decreasing in the interval (0, π/2)
-- 2. f is not monotonic in the interval (π/4, 3*π/4)

theorem problem (x : ℝ) :
    (0 < x ∧ x < π / 2 → MonotoneDecreasingOn f {y | 0 < y ∧ y < π / 2}) ∧
    (π / 4 < x ∧ x < 3 * π / 4 → ¬MonotoneOn f {y | π / 4 < y ∧ y < 3 * π / 4}) :=
sorry

end problem_l700_700921


namespace jonathan_and_matthew_strawberries_l700_700601

variable (total_strawberries : ℕ) (matthew_and_zac : ℕ) (zac_alone : ℕ)

def strawberries_by_jonathan_and_matthew (total_strawberries = 550) (matthew_and_zac = 250) (zac_alone = 200) : ℕ :=
  total_strawberries - zac_alone

theorem jonathan_and_matthew_strawberries : strawberries_by_jonathan_and_matthew 550 250 200 = 350 := by
  simp [strawberries_by_jonathan_and_matthew]
  sorry

end jonathan_and_matthew_strawberries_l700_700601


namespace call_cost_per_minute_l700_700211

-- Definitions (conditions)
def initial_credit : ℝ := 30
def call_duration : ℕ := 22
def remaining_credit : ℝ := 26.48

-- The goal is to prove that the cost per minute of the call is 0.16
theorem call_cost_per_minute :
  (initial_credit - remaining_credit) / call_duration = 0.16 := 
sorry

end call_cost_per_minute_l700_700211


namespace car_actual_speed_l700_700736

theorem car_actual_speed
  (distance : ℝ)
  (time_hours : ℝ)
  (time_minutes : ℝ)
  (time_seconds : ℝ)
  (fraction_speed : ℝ)
  (speed : ℝ) 
  (actual_speed : ℝ)
  (h_distance : distance = 120)
  (h_time_hours : time_hours = 2)
  (h_time_minutes : time_minutes = 19)
  (h_time_seconds : time_seconds = 36)
  (h_fraction_speed : fraction_speed = 9 / 13)
  (h_speed : speed = 120 / ((2 * 3600 + 19 * 60 + 36) / 3600))
  (h_actual_speed : actual_speed = speed * (13 / 9)) : 
  actual_speed ≈ 74.5 := by
  sorry

end car_actual_speed_l700_700736


namespace sqrt_defined_iff_l700_700987

theorem sqrt_defined_iff (x : ℝ) : (∃ y : ℝ, y = real.sqrt (x - 4)) ↔ (x ≥ 4) :=
by sorry

end sqrt_defined_iff_l700_700987


namespace order_of_fractions_l700_700906

theorem order_of_fractions (a b c d : ℝ) (hpos_a : a > 0) (hpos_b : b > 0) (hpos_c : c > 0) (hpos_d : d > 0)
(hab : a > b) : (b / a) < (b + c) / (a + c) ∧ (b + c) / (a + c) < (a + d) / (b + d) ∧ (a + d) / (b + d) < (a / b) :=
by
  sorry

end order_of_fractions_l700_700906


namespace cattle_horses_problem_l700_700661

-- Define variables for the price of one horse and one cow
variables (x y : ℝ)

-- Condition 1: The total value of two horses and one cow exceeds ten thousand by half the value of half a horse
def condition1 : Prop := 2 * x + y - 10000 = (1/2) * x

-- Condition 2: The total value of one horse and two cows is less than ten thousand by half the value of half a cow
def condition2 : Prop := 10000 - (x + 2 * y) = (1/2) * y

-- The goal is to verify if the given equations satisfy the conditions
theorem cattle_horses_problem (x y : ℝ) : condition1 x y ∧ condition2 x y ↔ 
  (2 * x + y - 10000 = (1/2) * x ∧ 10000 - (x + 2 * y) = (1/2) * y) := 
begin
  sorry
end

end cattle_horses_problem_l700_700661


namespace sqrt_floor_squared_eq_16_l700_700042

theorem sqrt_floor_squared_eq_16 :
  (∃ (x y : ℝ), x = 4 ∧ y = 5 ∧ x < sqrt 24 ∧ sqrt 24 < y) →
  (floor (sqrt 24) ^ 2 = 16) :=
by
  intro h
  sorry

end sqrt_floor_squared_eq_16_l700_700042


namespace ten_tuples_sum_to_one_odd_l700_700597

theorem ten_tuples_sum_to_one_odd :
  ∃ n : ℕ, 
    (n = (∑ m in 
      { t : tuple (ℕ) 10 | (∀ i, t[i] > 0) ∧
        (∑ i in (fin_range 10), 1/((tuple_components t).lookup i) = 1), 
      1 }) ∧ 
    (odd n)) := 
sorry

end ten_tuples_sum_to_one_odd_l700_700597


namespace weight_of_e_l700_700719

variables (d e f : ℝ)

theorem weight_of_e
  (h_de_f : (d + e + f) / 3 = 42)
  (h_de : (d + e) / 2 = 35)
  (h_ef : (e + f) / 2 = 41) :
  e = 26 :=
by
  sorry

end weight_of_e_l700_700719


namespace sum_abs_factors_2023_poly_eq_267036_l700_700807

open Int

theorem sum_abs_factors_2023_poly_eq_267036 :
  let S := (finset.sum (finset.filter (λ d : ℤ, ∃ u v : ℤ, u + v = -d ∧ u * v = 2023 * d) (finset.range 20230)).to_finset ℤ) in
  |S| = 267036 := 
by
  sorry

end sum_abs_factors_2023_poly_eq_267036_l700_700807


namespace log_inequality_l700_700881

variable (a b : ℝ)

theorem log_inequality (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b :=
sorry

end log_inequality_l700_700881


namespace larger_number_of_two_l700_700720

-- Definitions
def hcf (a b : ℕ) : ℕ := 50
def lcm (a b : ℕ) := 463450
def factors_of_lcm := [13, 23, 31]

-- Theorem to prove
theorem larger_number_of_two (a b : ℕ) (h_a_b : hcf a b = 50) (lcm_factors : lcm a b = 50 * 13 * 23 * 31) : (a = 463450 ∨ b = 463450) := sorry

end larger_number_of_two_l700_700720


namespace count_integers_500_to_700_with_digit_sum_18_l700_700546

theorem count_integers_500_to_700_with_digit_sum_18 :
  (∃ f : ℕ → ℕ, (∀ n, 500 ≤ n ∧ n < 700 → (n.digitSum = 18 → f n = 1) ∧ (n.digitSum ≠ 18 → f n = 0)) ∧ 
  (∑ n in finset.Icc 500 699, f n) = 25) :=
by
  sorry

end count_integers_500_to_700_with_digit_sum_18_l700_700546


namespace moles_of_Cl2_required_l700_700165

theorem moles_of_Cl2_required :
  ∀ (C2H6 Cl2 C2Cl6 HCl : Type)
    [ring C2H6] [ring Cl2] [ring C2Cl6] [ring HCl],
  (∀ (x : ℝ), 2 * (6 * x / 1) = 12) := by
  intros _ _ _ _ _ _ _ _ x
  exact sorry

end moles_of_Cl2_required_l700_700165


namespace original_area_of_doubled_rectangle_l700_700296

theorem original_area_of_doubled_rectangle (A_new : ℝ) (h : A_new = 32) :
  ∃ A : ℝ, A * 4 = A_new ∧ A = 8 :=
by {
  use 8,
  split,
  { norm_num, exact h.symm },
  { rfl }
}

end original_area_of_doubled_rectangle_l700_700296


namespace angle_bisector_divides_perimeter_l700_700255

noncomputable def is_isosceles_triangle (A B C : Point) : Prop :=
  dist A C = dist B C

theorem angle_bisector_divides_perimeter (A B C L : Point) 
  (hCL : is_angle_bisector A B C L)
  (hPerimeter : dist A B + dist B C + dist C A = 2 * dist A L + 2 * dist B L) :
  is_isosceles_triangle A B C := by
  sorry

end angle_bisector_divides_perimeter_l700_700255


namespace proof_ineq_l700_700481

noncomputable def P (f g : ℤ → ℤ) (m n k : ℕ) :=
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = g y → m = m + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ f x = f y → n = n + 1) ∧
  (∀ x y : ℤ, -1000 ≤ x ∧ x ≤ 1000 ∧ -1000 ≤ y ∧ y ≤ 1000 ∧ g x = g y → k = k + 1)

theorem proof_ineq (f g : ℤ → ℤ) (m n k : ℕ) (h : P f g m n k) : 
  2 * m ≤ n + k :=
  sorry

end proof_ineq_l700_700481


namespace floor_and_ceil_sum_l700_700038

theorem floor_and_ceil_sum : ⌊1.999⌋ + ⌈3.001⌉ = 5 := 
by
  sorry

end floor_and_ceil_sum_l700_700038


namespace fraction_historical_fiction_new_releases_l700_700712

theorem fraction_historical_fiction_new_releases
  (total_books : ℕ)
  (historical_fiction_pct new_historical_pct new_non_historical_pct : ℚ)
  (H1 : historical_fiction_pct = 0.3)
  (H2 : new_historical_pct = 0.4)
  (H3 : new_non_historical_pct = 0.5)
  (total_books_pos : 0 < total_books) :
  let historical_fiction_books := historical_fiction_pct * total_books
      historical_fiction_new_releases := new_historical_pct * historical_fiction_books
      non_historical_fiction_books := (1 - historical_fiction_pct) * total_books
      non_historical_fiction_new_releases := new_non_historical_pct * non_historical_fiction_books
      total_new_releases := historical_fiction_new_releases + non_historical_fiction_new_releases
  in historical_fiction_new_releases / total_new_releases = 12 / 47 :=
sorry

end fraction_historical_fiction_new_releases_l700_700712


namespace count_terminating_nonzero_tenths_l700_700467

theorem count_terminating_nonzero_tenths :
  (∃ n <= 50, ∀ a b, n = 2^a * 5^b ∧ decimal_has_nonzero_tenths (1/n)) ↔ (∃ m, m = 6) :=
sorry

def decimal_has_nonzero_tenths (x : ℚ) : Prop :=
  let dec := (x.num : ℤ) / (10 * x.denom) in
  dec % 10 != 0

end count_terminating_nonzero_tenths_l700_700467


namespace mean_of_sequence_l700_700342

def mean (s : List ℕ) : ℚ := (s.sum : ℚ) / s.length

theorem mean_of_sequence :
  mean [1^2, 2^2, 3^2, 4^2, 5^2, 6^2, 7^2, 2] = 17.75 := by
sorry

end mean_of_sequence_l700_700342


namespace num_x_intercepts_cos_inv_x_l700_700809

theorem num_x_intercepts_cos_inv_x (a b : ℝ) (ha : a = 0.00005) (hb : b = 0.0005) :
  (set.filter (λ x : ℝ, x ∈ set.Ioo a b ∧ cos (1 / x) = 0) set.univ).card = 1432 :=
sorry

end num_x_intercepts_cos_inv_x_l700_700809


namespace find_circle_with_pq_diameter_l700_700727

noncomputable def circle_with_diameter_from_intersection 
  (C : ℝ → ℝ → Prop)
  (L : ℝ → ℝ → Prop) : Prop :=
  C = (λ x y, x^2 + y^2 + x - 6y + 3 = 0) ∧
  L = (λ x y, x + 2y - 3 = 0) ∧
  (∃ P Q : ℝ × ℝ, C P.1 P.2 ∧ L P.1 P.2 ∧ C Q.1 Q.2 ∧ L Q.1 Q.2 ∧
  (∀ x y, (x, y) ∈ set.segment ℝ P Q → (λ x y, x^2 + y^2 + 2x - 4y = 0) x y))

theorem find_circle_with_pq_diameter :
  circle_with_diameter_from_intersection
     (λ x y, x^2 + y^2 + x - 6y + 3 = 0)
     (λ x y, x + 2y - 3 = 0) :=
sorry

end find_circle_with_pq_diameter_l700_700727


namespace shaded_area_60_l700_700189

structure Quadrilateral where
  A B C D O E : Point
  AC BD: Line
  H_DE : Triangle D B C -> Nat
  d_c : Nat := 17
  d_e : Nat := 15
  area_ABC : Nat
  area_DBC : Nat
  height_equal : area_ABO = area_DCO

theorem shaded_area_60 (q : Quadrilateral) 
  (S_ABO_eq_S_DCO : q.area_ABC = q.area_DBC) : 
  area (Triangle.mk q.A q.C q.E) = 60 := sorry

end shaded_area_60_l700_700189


namespace line_circle_intersect_twice_l700_700549

theorem line_circle_intersect_twice 
  (line : ∀ x y : ℝ, 4 * x + 9 * y = 7) 
  (circle : ∀ x y : ℝ, x^2 + y^2 = 1):
  ∃ (x₁ x₂ y₁ y₂ : ℝ), (x₁ ≠ x₂) ∧ line x₁ y₁ ∧ line x₂ y₂ ∧ circle x₁ y₁ ∧ circle x₂ y₂ :=
by
  sorry

end line_circle_intersect_twice_l700_700549


namespace calculate_area_of_triangle_l700_700789

theorem calculate_area_of_triangle :
  let p1 := (5, -2)
  let p2 := (5, 8)
  let p3 := (12, 8)
  let area := (1 / 2) * ((p2.2 - p1.2) * (p3.1 - p2.1))
  area = 35 := 
by
  sorry

end calculate_area_of_triangle_l700_700789


namespace solution_set_l700_700069

-- Define the intervals for the solution set
def interval1 : Set ℝ := Set.Ico (5/3) 2
def interval2 : Set ℝ := Set.Ico 2 3

-- Define the function that we need to prove
def equation_holds (x : ℝ) : Prop := Int.floor (Int.floor (3 * x) - 1 / 3) = Int.floor (x + 3)

theorem solution_set :
  { x : ℝ | equation_holds x } = interval1 ∪ interval2 :=
by
  -- Placeholder for the proof
  sorry

end solution_set_l700_700069


namespace floor_eq_l700_700065

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l700_700065


namespace count_permutations_perception_l700_700812

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l700_700812


namespace distribution_plans_count_l700_700264

theorem distribution_plans_count :
  let boys := 3
  let girls := 3
  let students := boys + girls
  let Yucai_only_accepts_boy := true
  let each_other_must_accept_at_least_one := true
  number_of_possible_distribution_plans students Yucai_only_accepts_boy each_other_must_accept_at_least_one = 54 :=
by
  -- Proof omitted
  sorry

end distribution_plans_count_l700_700264


namespace smallest_six_digit_number_formed_l700_700389

theorem smallest_six_digit_number_formed (n : ℕ) : 
  ∃ x y z : ℕ, (0 ≤ x) ∧ (x ≤ 9) ∧ (0 ≤ y) ∧ (y ≤ 9) ∧ (0 ≤ z) ∧ (z ≤ 9) ∧ 
  let num := 325000 + 100 * x + 10 * y + z in
  num % 3 = 0 ∧ num % 4 = 0 ∧ num % 5 = 0 ∧ 
  num = 325020 :=
sorry

end smallest_six_digit_number_formed_l700_700389


namespace pencils_across_diameter_l700_700967

theorem pencils_across_diameter (r : ℝ) (pencil_length_inch : ℕ) (pencils : ℕ) :
  r = 14 ∧ pencil_length_inch = 6 ∧ pencils = 56 → 
  let d := 2 * r in
  let pencil_length_feet := pencil_length_inch / 12 in
  pencils = d / pencil_length_feet :=
begin
  sorry -- Proof is skipped
end

end pencils_across_diameter_l700_700967


namespace total_money_shared_l700_700797

theorem total_money_shared (T : ℝ) (h : 0.75 * T = 4500) : T = 6000 :=
by
  sorry

end total_money_shared_l700_700797


namespace find_f_prove_monotonicity_l700_700915

-- Define the function f and set it as an odd function with given conditions
def f (x : ℝ) : ℝ := (x^2 + 1) / (a * x + b)

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f (x)

def f_is_odd : Prop :=
  is_odd_function f

def f_value_at_minus1 : Prop :=
  f (-1) = -2

-- Define the target properties to prove
def f_analytical_expression (a b : ℝ) : Prop :=
  f = λ x, x + 1 / x

def f_monotonic_on_pos (f : ℝ → ℝ) : Prop :=
  ∀ x1 x2 : ℝ, 1 < x1 → x1 < x2 → x2 < ∞ → f x1 < f x2

-- We now state the theorems to match our proof problems
theorem find_f (a b : ℝ) (h1 : f_is_odd) (h2 : f_value_at_minus1) :
  f_analytical_expression a b :=
sorry

theorem prove_monotonicity (h : f_analytical_expression 1 0) :
  f_monotonic_on_pos f :=
sorry

end find_f_prove_monotonicity_l700_700915


namespace find_set_A_B_subset_A_A_union_B_is_R_A_intersection_B_nonempty_l700_700903

variable (x a : ℝ)

def A : Set ℝ := {x | -x^2 + 2 * x + 3 < 0}
def B : Set ℝ := {x | a - 2 ≤ x ∧ x ≤ 2 * a + 3}

theorem find_set_A :
  A = {x | x < -1 ∨ x > 3} :=
sorry

theorem B_subset_A (a : ℝ) :
  (∀ x, x ∈ B → x ∈ A) ↔ (a < -2 ∨ a > 5) :=
sorry

theorem A_union_B_is_R (a : ℝ) :
  (∀ x, x ∈ Set.univ → x ∈ A ∪ B) ↔ (0 ≤ a ∧ a ≤ 1) :=
sorry

theorem A_intersection_B_nonempty (a : ℝ) :
  (∃ x, x ∈ A ∧ x ∈ B) ↔ (a ≥ -5) :=
sorry

end find_set_A_B_subset_A_A_union_B_is_R_A_intersection_B_nonempty_l700_700903


namespace meet_in_time_l700_700343

-- Definitions based on conditions
def highway_length : ℝ := 175
def speed_car1 : ℝ := 25
def speed_car2 : ℝ := 45

-- Lean statement to prove
theorem meet_in_time : ∃ t : ℝ, t = 2.5 ∧ speed_car1 * t + speed_car2 * t = highway_length := by
  use 2.5
  simp [speed_car1, speed_car2, highway_length]
  split
  rfl
  norm_num
  sorry

end meet_in_time_l700_700343


namespace find_red_peaches_l700_700686

def num_red_peaches (red yellow green : ℕ) : Prop :=
  (green = red + 1) ∧ yellow = 71 ∧ green = 8

theorem find_red_peaches (red : ℕ) :
  num_red_peaches red 71 8 → red = 7 :=
by
  sorry

end find_red_peaches_l700_700686


namespace math_problem_l700_700106

noncomputable def x : ℝ := 0.25 * (y ^ 2)
noncomputable def y : ℝ := (1 / 3) * 600
noncomputable def z : ℝ := Real.sqrt 1521
noncomputable def w : ℝ := 0.48 * 10 ^ 3

noncomputable def result := 
  ((0.15 * (x - y) + 0.45 * (y - z) + 0.06 * (z - w)) - 0.33 * (w - x)) / 
  (0.52 * Real.sqrt x)

theorem math_problem :
  result = 89.069 :=
sorry

end math_problem_l700_700106


namespace ones_digit_of_prime_in_arithmetic_sequence_is_one_l700_700874

theorem ones_digit_of_prime_in_arithmetic_sequence_is_one 
  (p q r s : ℕ) 
  (hp : Prime p) 
  (hq : Prime q) 
  (hr : Prime r) 
  (hs : Prime s) 
  (h₁ : p > 10) 
  (h₂ : q = p + 10) 
  (h₃ : r = q + 10) 
  (h₄ : s = r + 10) 
  (h₅ : s > r) 
  (h₆ : r > q) 
  (h₇ : q > p) : 
  p % 10 = 1 :=
sorry

end ones_digit_of_prime_in_arithmetic_sequence_is_one_l700_700874


namespace rhombus_perimeter_l700_700281

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  let s := real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  in 4 * s = 16 * real.sqrt 13 := 
by {
  let a := d1 / 2,
  let b := d2 / 2,
  have h_a : a = 12 := by { rw h1, norm_num },
  have h_b : b = 8 := by { rw h2, norm_num },
  have h_s : s = real.sqrt (a^2 + b^2) := by refl,
  have h_s_val : s = 4 * real.sqrt 13 := by {
    rw [h_a, h_b], 
    norm_num,
    simp [real.sqrt_mul (show 16 > 0, by norm_num), show 13 > 0, by norm_num]
  },
  rw h_s_val,
  norm_num
}

end rhombus_perimeter_l700_700281


namespace unique_solution_exists_l700_700090

theorem unique_solution_exists :
  ∃! (x y : ℝ), 4^(x^2 + 2 * y) + 4^(2 * x + y^2) = Real.cos (Real.pi * x) ∧ (x, y) = (2, -2) :=
by
  sorry

end unique_solution_exists_l700_700090


namespace smallest_possible_edge_length_l700_700790

noncomputable def smallest_edge_length (K₁ K₂ : Type) [MetricSpace K₁] [MetricSpace K₂] (on_surface : K₂ → K₁) : ℝ :=
  Inf {l | ∃ (cube₁ : K₁) (cube₂ : K₂), edge_length cube₁ = 1 ∧ on_surface cube₂ = cube₁ ∧ edge_length cube₂ = l}

theorem smallest_possible_edge_length : ∃ (K₁ K₂ : Type) [MetricSpace K₁] [MetricSpace K₂] (on_surface : K₂ → K₁), 
  smallest_edge_length K₁ K₂ on_surface = 1 / Real.sqrt 2 := 
sorry

end smallest_possible_edge_length_l700_700790


namespace properties_of_data_set_l700_700740

def data_set : List ℕ := [67, 57, 37, 40, 46, 62, 31, 47, 31, 30]

def sorted_data_set : List ℕ := [30, 31, 31, 37, 40, 46, 47, 57, 62, 67]

def mode (l : List ℕ) : ℕ :=
(d : ℕ × ℕ) ← list.maximumBy (λ d, l.count d) l
(_, x) := d
x

def range (l : List ℕ) : ℕ :=
(list.maximum l - list.minimum l)

def quantile (l : List ℕ) (q : ℕ) : ℝ :=
let pos := (q * l.length) / 100
let sorted := l.sort λ a b => a < b
let lower := sorted[pos]
let upper := sorted[pos + 1]
(lower + upper) / 2

theorem properties_of_data_set :
  mode data_set = 31 ∧
  range data_set = 37 ∧
  quantile data_set 10 = 30.5 :=
by sorry

end properties_of_data_set_l700_700740


namespace average_annual_growth_rate_l700_700567

variables {x : ℝ}

theorem average_annual_growth_rate:
  (200 : ℝ) * (1 + x)^2 = 338 → x = 0.3 :=
by
  sorry

end average_annual_growth_rate_l700_700567


namespace find_a_l700_700223

theorem find_a (a b : ℝ) (f : ℝ → ℝ) (h₁ : f = λ x, a * x^3 + b) (h₂ : (λ x, 3 * a * x^2) (-1) = 3) : a = 1 :=
by {
  sorry
}

end find_a_l700_700223


namespace center_square_side_length_l700_700275

theorem center_square_side_length : 
  let side_length := 200 in
  let total_area := side_length * side_length in
  let L_region_area_fraction := 5 / 16 in
  let L_region_total_area := 4 * L_region_area_fraction * total_area in
  let center_square_area := total_area - L_region_total_area in
  sqrt center_square_area ≈ 173 := 
by 
  sorry

end center_square_side_length_l700_700275


namespace find_b_age_l700_700339

theorem find_b_age (a b : ℕ) (h1 : a + 10 = 2 * (b - 10)) (h2 : a = b + 9) : b = 39 :=
sorry

end find_b_age_l700_700339


namespace find_matrix_N_l700_700852

open Matrix

variable (u : Vector3 ℝ)

noncomputable def N : Matrix (Fin 3) (Fin 3) ℝ :=
  !\[0, -7, -3; 7, 0, 4; -3, 4, 0\]

theorem find_matrix_N (u : Vector3 ℝ) :
  (N.mulVec u) = Vector.crossProduct !\[4, -3, 7\] u :=
by 
  sorry

end find_matrix_N_l700_700852


namespace probability_below_8_l700_700674

theorem probability_below_8 
  (P10 P9 P8 : ℝ)
  (P10_eq : P10 = 0.24)
  (P9_eq : P9 = 0.28)
  (P8_eq : P8 = 0.19) :
  1 - (P10 + P9 + P8) = 0.29 := 
by
  sorry

end probability_below_8_l700_700674


namespace circle_eq_and_chord_length_l700_700116

noncomputable def midpoint (P Q : ℝ × ℝ) : ℝ × ℝ :=
  ((P.1 + Q.1) / 2, (P.2 + Q.2) / 2)

noncomputable def distance (P Q : ℝ × ℝ) : ℝ :=
  real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem circle_eq_and_chord_length :
  (∃ C : ℝ × ℝ, 
    C = midpoint (1, -2) (3, 4) ∧
    (∃ r : ℝ,
      r = (1 / 2) * distance (1, -2) (3, 4) ∧
      ∀ (x y : ℝ), (x - 2)^2 + (y - 1)^2 = 10 ∧
      (let d := (|2 * 2 - 1 + b|) / real.sqrt 5 in d^2 + real.sqrt 5^2 = 10 → b = 2 ∨ b = -8))) :=
sorry

end circle_eq_and_chord_length_l700_700116


namespace num_ways_to_arrange_PERCEPTION_l700_700834

open Finset

def word := "PERCEPTION"

def num_letters : ℕ := 10

def occurrences : List (Char × ℕ) :=
  [('P', 2), ('E', 2), ('R', 1), ('C', 1), ('E', 2), ('P', 2), ('T', 1), ('I', 2), ('O', 1), ('N', 1)]

def factorial (n : ℕ) : ℕ := List.range n.succ.foldl (· * ·) 1

noncomputable def num_distinct_arrangements (n : ℕ) (ks : List ℕ) : ℕ :=
  factorial n / ks.foldl (λ acc k => acc * factorial k) 1

theorem num_ways_to_arrange_PERCEPTION :
  num_distinct_arrangements num_letters [2, 2, 2, 1, 1, 1, 1, 1] = 453600 := 
by sorry

end num_ways_to_arrange_PERCEPTION_l700_700834


namespace matrix_exponentiation_l700_700412

open Matrix

def A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![2, -2],
    ![2, -1]]

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![-4, 6],
    ![-6, 5]]

theorem matrix_exponentiation :
  A^4 = B :=
by
  sorry

end matrix_exponentiation_l700_700412


namespace domain_of_sqrt_tan_x_minus_sqrt_3_l700_700084

noncomputable def domain_of_function : Set Real :=
  {x | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2}

theorem domain_of_sqrt_tan_x_minus_sqrt_3 :
  { x : Real | ∃ k : ℤ, k * Real.pi + Real.pi / 3 ≤ x ∧ x < k * Real.pi + Real.pi / 2 } = domain_of_function :=
by
  sorry

end domain_of_sqrt_tan_x_minus_sqrt_3_l700_700084


namespace floor_eq_l700_700066

theorem floor_eq (x : ℝ) :
  (⟨⟨3 * x⟩ - (1 / 3)⟩ = ⟨x + 3⟩) ↔ (x ∈ Set.Ico (4 / 3) (5 / 3)) := 
sorry

end floor_eq_l700_700066


namespace trajectory_of_point_is_line_l700_700890

theorem trajectory_of_point_is_line:
  (∀ x y : ℝ, (5 * real.sqrt ((x - 1) ^ 2 + (y - 2) ^ 2) = real.abs (3 * x + 4 * y - 11)) →
    ∃ m b : ℝ, ∀ x y : ℝ, (y = m * x + b)) :=
by
  sorry

end trajectory_of_point_is_line_l700_700890


namespace min_f_value_l700_700916

noncomputable def f (x : ℝ) : ℝ :=
  classical.some (exists_f (f' x, x))

-- Assuming f' (the derivative of f) is differentiable, meaning for all x in the domain, there exists f'' satisfying
-- f' (x) = f'' (x) * x + f' (x) - 1
def f' (x : ℝ) : ℝ :=
  -- Placeholder for the actual derivative based on conditions
  sorry

def f'' (x : ℝ) : ℝ :=
  -- Placeholder for the actual second derivative based on conditions
  sorry

example : f(1) = 0 := sorry

example : ∃ x : ℝ, x = 1 / Real.exp 1 ∧ f x = -1 / Real.exp 1 :=
begin
  sorry
end

theorem min_f_value : ∃ x : ℝ, f x = -1 / Real.exp 1 :=
begin
  -- We need to show that there exists x in (0, +∞) such that f(x) reaches its minimum value at -1/e.
  use (1 / Real.exp 1),
  sorry
end

end min_f_value_l700_700916


namespace euler_line_distance_sum_l700_700215

variable {Triangle : Type} [EuclideanGeometry Triangle]
variable {Point : Type} [AffineSpace Point Triangle]

def acute_triangle (A B C : Point) :=
  ∀ (α β γ : Real), 0 < α ∧ α < π / 2 ∧ 0 < β ∧ β < π / 2 ∧ 0 < γ ∧ γ < π / 2

variable (A B C H O A₁ B₁ C₁ E : Point)
variable (R p : Real)
variable [Circumcenter A B C O]
variable [Orthocenter A B C H]
variable [AcuteTriangle : acute_triangle A B C]

variable [AngleBisector A B H A₁]
variable [AngleBisector A C H A₁]
variable [Midpoint H O E]
variable [AngleBisector B C H B₁]
variable [AngleBisector B A H B₁]
variable [AngleBisector C A H C₁]
variable [AngleBisector C B H C₁]

theorem euler_line_distance_sum : EA_1 + EB_1 + EC_1 = p - (3/2) * R :=
sorry

end euler_line_distance_sum_l700_700215


namespace ratio_EH_HG_2_3_l700_700596

theorem ratio_EH_HG_2_3 {A B C D E F G H : Type}
  [AddCommGroup A] [Module ℝ A]
  (D_trisection : ∃ t : ℝ, t > 0 ∧ t < 1 ∧ t = 1/3 ∧ collinear D B C)
  (E_trisection : ∃ t : ℝ, t > 0 ∧ t < 1 ∧ t = 2/3 ∧ collinear E B C)
  (F_midpoint : midpoint F A C)
  (G_midpoint : midpoint G A B)
  (H_intersection : intersection H (line E G) (line D F)) :
  ratio ((dist E H) / (dist H G)) = 2/3 :=
sorry

end ratio_EH_HG_2_3_l700_700596


namespace constant_term_binomial_expansion_l700_700193

noncomputable def binom_coeff (n k : ℕ) : ℕ := Nat.choose n k

theorem constant_term_binomial_expansion :
  let n := 8 in
  ∃ (x : ℝ), (sqrt (x) - 2/x)^n = 112 :=
by
  sorry

end constant_term_binomial_expansion_l700_700193


namespace minimum_positive_omega_l700_700944

def f (ω x : ℝ) : ℝ := sin (ω * x) + cos (ω * x)

theorem minimum_positive_omega (x₁ ω : ℝ) 
    (h_condition : ∀ x, f ω x₁ ≤ f ω x ∧ f ω x ≤ f ω (x₁ + 2018)) :
    ω = π / 2018 :=
sorry

end minimum_positive_omega_l700_700944


namespace proof_problem_l700_700475

-- Define the minimum function for two real numbers
def min (x y : ℝ) := if x < y then x else y

-- Define the real numbers sqrt30, a, and b
def sqrt30 := Real.sqrt 30

variables (a b : ℕ)

-- Define the conditions
def conditions := (min sqrt30 a = a) ∧ (min sqrt30 b = sqrt30) ∧ (b = a + 1)

-- State the theorem to prove
theorem proof_problem (h : conditions a b) : 2 * a - b = 4 :=
sorry

end proof_problem_l700_700475


namespace exists_circle_with_exactly_2019_lattice_points_l700_700638

-- Defining the center of the circles
def M : ℝ × ℝ := (real.sqrt 2, real.sqrt 3)

-- Defining the family of circles with radius r
def circle (r : ℝ) : set (ℝ × ℝ) := {p | ((p.1 - (real.sqrt 2))^2 + (p.2 - (real.sqrt 3))^2 = r^2)}

-- Defining the set of all integer-coordinate points
def lattice_points : set (ℤ × ℤ) := {p | true}

-- Function to convert integer coordinate to real coordinate
def to_real_point (p : ℤ × ℤ) : ℝ × ℝ := (p.1.toReal, p.2.toReal)

-- Main theorem statement
theorem exists_circle_with_exactly_2019_lattice_points :
  ∃ (r : ℝ), (circle r).count (lattice_points.image to_real_point) = 2019 :=
sorry

end exists_circle_with_exactly_2019_lattice_points_l700_700638


namespace analytic_expression_of_f_range_of_a_l700_700118

noncomputable def f (x : ℝ) := 2 * sin (2 * x - π / 6)

theorem analytic_expression_of_f :
  (∃ A ω φ : ℝ, A > 0 ∧ ω > 0 ∧ abs φ < π / 2 ∧ (f = λ x, A * sin (ω * x + φ))
    ∧ (∀ x, f (x + π / ω) = f x)
    ∧ f (π / 12) = 0
    ∧ f (π / 2) = 1) →
  f = λ x, 2 * sin (2 * x - π / 6) :=
by
  sorry

theorem range_of_a (x : ℝ) (a : ℝ) :
  (x ∈ Icc (0 : ℝ) (π / 2) ∧ 2 * f x - a + 1 = 0) →
  -1 ≤ a ∧ a ≤ 5 :=
by
  sorry

end analytic_expression_of_f_range_of_a_l700_700118


namespace range_of_a_for_monotonic_decreasing_l700_700938

theorem range_of_a_for_monotonic_decreasing (a : ℝ) :
(∀ x : ℝ, 0 < x ∧ x < 1 → 2 ^ (x * (x - a)) > 2 ^ (x * (x - a)) → a ∈ set.Ici 2) :=
begin
  sorry
end

end range_of_a_for_monotonic_decreasing_l700_700938


namespace length_of_ST_l700_700593

-- Given: P, Q, and R are points forming triangle PQR with PQ = 40 and ∠R = 45°
-- S is the midpoint of PQ
-- T is the intersection of the perpendicular bisector of PQ with PR
-- Prove: The length of ST = 10√2

-- Define the geometry setting
variables (P Q R S T : Type) [LinearOrderedField T]
variables (PQ PR : T)
variables (angleR : T)
variable (midpoint : T)

-- State the conditions
axiom PQ_eq_40 : PQ = 40
axiom angleR_eq_45 : angleR = 45
axiom S_midpoint_of_PQ : midpoint = PQ / 2
axiom T_intersection_perpendicular : (∃ t : T, t = S)

-- Theorem statement
theorem length_of_ST : ∃ ST : T, ST = 10 * Real.sqrt 2 :=
sorry

end length_of_ST_l700_700593


namespace least_three_digit_7_heavy_l700_700769

/-- A number is 7-heavy if the remainder when the number is divided by 7 is greater than 4. -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The statement to be proved: The least three-digit 7-heavy number is 104. -/
theorem least_three_digit_7_heavy : ∃ n, 100 ≤ n ∧ n < 1000 ∧ is_7_heavy(n) ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ is_7_heavy(m) → n ≤ m :=
begin
    use 104,
    split,
    { exact dec_trivial, },
    split,
    { exact dec_trivial, },
    split,
    { change 104 % 7 > 4,
      exact dec_trivial, },
    { intros m h1 h2,
      sorry
    }
end

end least_three_digit_7_heavy_l700_700769


namespace cyclic_quadrilateral_angle_l700_700122

variables (A B C D E : Type) [circle A B C D]
variables (∠BAD ∠ADC ∠EBC : ℝ)
variables (line_AB_extended : (∠BAD = 80) ∧ (∠ADC = 40) ∧ ∠EBC = 40)

theorem cyclic_quadrilateral_angle :
  ∠EBC = 40 :=
begin
  sorry
end

end cyclic_quadrilateral_angle_l700_700122


namespace max_profit_at_60_l700_700388

noncomputable def profit (x : ℕ) : ℕ :=
  if x ≤ 30 then 900 * x - 15000
  else -10 * x * x + 1200 * x - 15000

theorem max_profit_at_60 :
  ∃ (x : ℕ), x = 60 ∧ profit x = 21000 :=
by
  use 60
  split
  { rfl }
  { sorry }

end max_profit_at_60_l700_700388


namespace least_three_digit_7_heavy_l700_700775

-- Define what it means to be a 7-heavy number
def is_7_heavy (n : ℕ) : Prop :=
  n % 7 > 4

-- Define the property of being three-digit
def is_three_digit (n : ℕ) : Prop :=
  n >= 100 ∧ n < 1000

-- The statement to prove
theorem least_three_digit_7_heavy : ∃ n, is_7_heavy n ∧ is_three_digit n ∧ ∀ m, is_7_heavy m ∧ is_three_digit m → n ≤ m :=
begin
  use [104],
  split,
  { -- Proof that 104 is 7-heavy
    show is_7_heavy 104,
    simp [is_7_heavy], -- Calculation: 104 % 7 = 6 which is > 4
    norm_num,
  },
  split,
  { -- Proof that 104 is a three-digit number
    show is_three_digit 104,
    simp [is_three_digit],
    norm_num,
  },
  { -- Proof that 104 is the smallest 7-heavy three-digit number
    intros m hm,
    cases hm with hm1 hm2,
    suffices : 104 ≤ m,
    exact this,
    calc 104 ≤ 100 + 7 - 1 : by norm_num
        ... ≤ m            : by linarith [hm2.left, hm2.right],
    sorry,
  }
sorry

end least_three_digit_7_heavy_l700_700775


namespace flagpole_height_l700_700370

theorem flagpole_height (h : ℝ) (shadow_flagpole : ℝ) (shadow_building : ℝ) (height_building : ℝ)
  (H_flagpole_shadow : shadow_flagpole = 45)
  (H_building_shadow : shadow_building = 55)
  (H_building_height : height_building = 22)
  (H_similar_triangles : h / shadow_flagpole = height_building / shadow_building) :
  h = 18 :=
by simp [H_flagpole_shadow, H_building_shadow, H_building_height, H_similar_triangles, h] ; sorry

end flagpole_height_l700_700370


namespace numBoysOnPlayground_l700_700689

-- Define the given data
variable (numGirls : ℝ) (extraBoys : ℝ)

-- Define the condition that helps us find the solution
def numBoys (numGirls extraBoys : ℝ) : ℝ := numGirls + extraBoys

-- Hypotheses based on the problem statement
axiom numGirlsIs28 : numGirls = 28
axiom extraBoysIs7 : extraBoys = 7

-- Statement to prove
theorem numBoysOnPlayground : numBoys numGirls extraBoys = 35 :=
by
  -- Using another axiom for the playground boy counting logic
  have h : numBoys 28 7 = 35 := by sorry
  exact h

#check numBoysOnPlayground

end numBoysOnPlayground_l700_700689


namespace number_of_pencils_l700_700972

-- Define the given conditions
def circle_radius : ℝ := 14 -- 14 feet radius
def pencil_length_inches : ℝ := 6 -- 6-inch pencil

noncomputable def pencil_length_feet : ℝ := pencil_length_inches / 12 -- convert 6 inches to feet

-- Statement of the problem in Lean
theorem number_of_pencils (r : ℝ) (p_len_inch : ℝ) (d : ℝ) (p_len_feet : ℝ) :
  r = circle_radius →
  p_len_inch = pencil_length_inches →
  d = 2 * r →
  p_len_feet = pencil_length_feet →
  d / p_len_feet = 56 :=
by
  intros hr hp hd hpl
  sorry

end number_of_pencils_l700_700972


namespace estimation_approx_equal_l700_700032

theorem estimation_approx_equal :
  let estimate := (fun n => if n % 100 < 50 then n - n % 100 else n + (100 - n % 100)) in
  estimate 208 + estimate 298 = 500 :=
by
  sorry

end estimation_approx_equal_l700_700032


namespace circle_line_tangent_l700_700179

theorem circle_line_tangent (m : ℝ) :
  (∃ x y : ℝ, x^2 + y^2 = 4 * m ∧ x + y = 2 * m) ↔ m = 2 :=
sorry

end circle_line_tangent_l700_700179


namespace foci_on_x_axis_l700_700145

theorem foci_on_x_axis (k : ℝ) : (∃ a b : ℝ, ∀ x y : ℝ, (x^2)/(3 - k) + (y^2)/(1 + k) = 1) ↔ -1 < k ∧ k < 1 :=
by
  sorry

end foci_on_x_axis_l700_700145


namespace mary_final_books_l700_700238

-- Initial number of books
def initial_books : ℕ := 72

-- Books received each month from book club for 12 months
def books_from_club : ℕ := 12 * 1

-- Books bought from different sources
def books_from_bookstore : ℕ := 5
def books_from_yard_sales : ℕ := 2

-- Books received as gifts
def books_from_daughter : ℕ := 1
def books_from_mother : ℕ := 4

-- Books gotten rid of
def books_donated : ℕ := 12
def books_sold : ℕ := 3

-- Final calculation
theorem mary_final_books : 
  initial_books + books_from_club + books_from_bookstore + books_from_yard_sales + books_from_daughter + books_from_mother - (books_donated + books_sold) = 81 :=
  by sorry

end mary_final_books_l700_700238


namespace hyperbola_eccentricity_l700_700153

-- Definitions based on conditions
def hyperbola_eq (a b : ℝ) (x y : ℝ) : Prop :=
  (a > 0) ∧ (b > 0) ∧ ((x^2 / a^2) - (y^2 / b^2) = 1)

def asymptote_eq (a b : ℝ) : Prop :=
  2 * (a / b) + 1 = 0

def eccentricity_eq (a b c e : ℝ) : Prop :=
  c = sqrt (a^2 + b^2) ∧ e = c / a

-- Theorem to prove the eccentricity
theorem hyperbola_eccentricity (a b : ℝ) (h0 : a > 0) (h1 : b > 0) (h2 : asymptote_eq a b) :
  ∃ t > 0, eccentricity_eq a (2*t) (sqrt 5 * t) (sqrt 5) :=
by {
  sorry
}

end hyperbola_eccentricity_l700_700153


namespace solution_functions_l700_700079

noncomputable def f : ℝ → ℝ := sorry

lemma functional_equation (x y : ℝ) : f(x * f(y)) + x = x * y + f(x) :=
  sorry

theorem solution_functions (f : ℝ → ℝ) :
  (∀ x y : ℝ, f(x * f(y)) + x = x * y + f(x)) →
  (f = (λ x, x) ∨ f = (λ x, -x)) :=
  sorry

end solution_functions_l700_700079


namespace connect_point_to_intersection_l700_700636

noncomputable def sheet := Type*
noncomputable def point := Type*
noncomputable def circle (P : point) := Type*

variables {sheet : Type*} {P P' : point} {k1 k2 k'1 k'2 : circle P}
variables {M : point}

-- Axioms and conditions
axiom exists_point_on_sheet : ∃ (P : point), P ∈ sheet
axiom exists_two_arcs : ∃ (k1 k2 : circle P), k1 ∈ sheet ∧ k2 ∈ sheet
axiom intersection_point_not_on_sheet : ∀ (M : point), M ∉ sheet

-- The theorem to be proved:
theorem connect_point_to_intersection (P : point) (k1 k2: circle P) :
  ∃ (M : point), ∃ (line : set point), M ∈ (k1 ∩ k2) ∧ P ∈ line ∧ M ∈ line :=
sorry

end connect_point_to_intersection_l700_700636


namespace fibonacci_ratio_l700_700274

noncomputable def fib : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+2) := fib (n+1) + fib n

theorem fibonacci_ratio :
  (∑ i in finset.range 2023, (fib i) ^ 2) / (2 * (fib 2022) * (fib 2023)) = 1 / 2 :=
by
  sorry

end fibonacci_ratio_l700_700274


namespace abs_diff_squares_eq_300_l700_700330

theorem abs_diff_squares_eq_300 : 
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  |a^2 - b^2| = 300 := 
by
  let a := (103 : ℚ) / 2 
  let b := (97 : ℚ) / 2
  sorry

end abs_diff_squares_eq_300_l700_700330


namespace count_valid_numbers_eq_13_l700_700540

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def in_range (n : ℕ) : Prop :=
  500 ≤ n ∧ n < 700

def is_valid (n : ℕ) : Prop :=
  in_range n ∧ sum_of_digits n = 18

def count_valid_numbers : ℕ :=
  (Finset.filter is_valid (Finset.range 700).filter (λ n, n ≥ 500)).card

theorem count_valid_numbers_eq_13 :
  count_valid_numbers = 13 :=
sorry

end count_valid_numbers_eq_13_l700_700540


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700052

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700052


namespace decreasing_sequence_l700_700909

noncomputable def a (n : ℕ) : ℝ :=
∑ k in Finset.range n, 1 / ((k + 1) * (n - k))

theorem decreasing_sequence (n : ℕ) (h : 2 ≤ n) : a (n + 1) < a n :=
sorry

end decreasing_sequence_l700_700909


namespace rhombus_perimeter_l700_700283

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  let s := real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  in 4 * s = 16 * real.sqrt 13 := 
by {
  let a := d1 / 2,
  let b := d2 / 2,
  have h_a : a = 12 := by { rw h1, norm_num },
  have h_b : b = 8 := by { rw h2, norm_num },
  have h_s : s = real.sqrt (a^2 + b^2) := by refl,
  have h_s_val : s = 4 * real.sqrt 13 := by {
    rw [h_a, h_b], 
    norm_num,
    simp [real.sqrt_mul (show 16 > 0, by norm_num), show 13 > 0, by norm_num]
  },
  rw h_s_val,
  norm_num
}

end rhombus_perimeter_l700_700283


namespace marble_selection_probability_l700_700731

theorem marble_selection_probability :
  let red_marbles := 3
  let blue_marbles := 3
  let green_marbles := 2
  let total_marbles := red_marbles + blue_marbles + green_marbles
  let choose (n k : ℕ) := n.choose k
    
  let total_ways := choose total_marbles 3
  let successful_ways := red_marbles * blue_marbles * green_marbles
  in (successful_ways : ℚ) / (total_ways : ℚ) = 9 / 28 :=
by
  -- include exact proof here
  sorry

end marble_selection_probability_l700_700731


namespace simplify_sqrt_mul_l700_700005

theorem simplify_sqrt_mul : (Real.sqrt 5 * Real.sqrt (4 / 5) = 2) :=
by
  sorry

end simplify_sqrt_mul_l700_700005


namespace ratio_triangle_to_rectangle_l700_700190

theorem ratio_triangle_to_rectangle 
  (l w : ℝ) 
  (hl : 0 < l) 
  (hw : 0 < w) 
  (M : Point) 
  (N : Point)
  (hM : M.x = l / 3 ∧ M.y = 0)
  (hN : N.x = l ∧ N.y = w / 4) :
  (area_triangle M N (Point.mk 0 0)) / (l * w) = 1 / 24 :=
by 
  sorry

end ratio_triangle_to_rectangle_l700_700190


namespace equation_one_solution_equation_two_solution_l700_700655

-- Define the conditions and prove the correctness of solutions to the equations
theorem equation_one_solution (x : ℝ) (h : 3 / (x - 2) = 9 / x) : x = 3 :=
by
  sorry

theorem equation_two_solution (x : ℝ) (h : x / (x + 1) = 2 * x / (3 * x + 3) - 1) : x = -3 / 4 :=
by
  sorry

end equation_one_solution_equation_two_solution_l700_700655


namespace circle_area_increase_l700_700181

-- Define the variables and condition
variable (r : ℝ) 

-- Define the original area
def original_area : ℝ := π * r^2

-- Define the new radius
def new_radius : ℝ := 2.5 * r

-- Define the new area
def new_area : ℝ := π * (new_radius r)^2

-- Define the increase in area
def increase_in_area : ℝ := new_area r - original_area r

-- The theorem to prove the percentage increase is 525%
theorem circle_area_increase (r : ℝ) (h : r > 0) : 
  (increase_in_area r) / (original_area r) * 100 = 525 := 
  sorry

end circle_area_increase_l700_700181


namespace perception_num_permutations_l700_700819

def factorial : ℕ → ℕ 
| 0       := 1
| (n + 1) := (n + 1) * factorial n

def perception_arrangements : ℕ :=
  let total_letters := 10
  let repetitions_P := 2
  let repetitions_E := 2
  factorial total_letters / (factorial repetitions_P * factorial repetitions_E)

theorem perception_num_permutations :
  perception_arrangements = 907200 :=
by sorry

end perception_num_permutations_l700_700819


namespace four_digit_numbers_divisible_by_90_l700_700960

theorem four_digit_numbers_divisible_by_90 : 
  let is_valid a b := (a + b) % 9 = 0 ∧ b % 2 = 0
  let nums := { (a, b) // is_valid a b ∧ 1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 }
  (Finset.card nums) = 5 :=
by
  sorry

end four_digit_numbers_divisible_by_90_l700_700960


namespace limit_tan_ln_ratio_l700_700782

theorem limit_tan_ln_ratio : 
  (Real.tendsto (λ x : ℝ, (Real.tan (π * (1 + x / 2)) / Real.log (x + 1))) 0 (𝓝 (π / 2))) :=
by
  sorry

end limit_tan_ln_ratio_l700_700782


namespace fewer_trombone_than_trumpet_l700_700676

theorem fewer_trombone_than_trumpet 
  (flute_players : ℕ)
  (trumpet_players : ℕ)
  (trombone_players : ℕ)
  (drummers : ℕ)
  (clarinet_players : ℕ)
  (french_horn_players : ℕ)
  (total_members : ℕ) :
  flute_players = 5 →
  trumpet_players = 3 * flute_players →
  clarinet_players = 2 * flute_players →
  drummers = trombone_players + 11 →
  french_horn_players = trombone_players + 3 →
  total_members = flute_players + clarinet_players + trumpet_players + trombone_players + drummers + french_horn_players →
  total_members = 65 →
  trombone_players = 7 ∧ trumpet_players - trombone_players = 8 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  rw [h1, h2, h3] at h6
  sorry

end fewer_trombone_than_trumpet_l700_700676


namespace solution_set_of_x_l700_700078

noncomputable def satisfies_condition (x : ℝ) : Prop :=
  ⌊⌊3 * x⌋ - 1/3⌋ = ⌊x + 3⌋

theorem solution_set_of_x :
  { x : ℝ | satisfies_condition x } = { x : ℝ | 5/3 ≤ x ∧ x < 7/3 } :=
by
  sorry

end solution_set_of_x_l700_700078


namespace eliminate_denominator_correctness_l700_700704

-- Define the initial equality with fractions
def initial_equation (x : ℝ) := (2 * x - 3) / 5 = (2 * x) / 3 - 3

-- Define the resulting expression after eliminating the denominators
def eliminated_denominators (x : ℝ) := 3 * (2 * x - 3) = 5 * 2 * x - 3 * 15

-- The theorem states that given the initial equation, the eliminated denomination expression holds true
theorem eliminate_denominator_correctness (x : ℝ) :
  initial_equation x → eliminated_denominators x := by
  sorry

end eliminate_denominator_correctness_l700_700704


namespace sum_of_integers_l700_700505

theorem sum_of_integers (n : ℤ) (h : 0 < 3 * n ∧ 3 * n < 27) : 
  ∑ i in Finset.filter (λ x, 0 < x ∧ x < 9) (Finset.range 10), i = 36 := by
  sorry

end sum_of_integers_l700_700505


namespace socks_probability_l700_700434

theorem socks_probability :
  let total_socks := 18
  let total_pairs := (total_socks.choose 2)
  let gray_socks := 12
  let white_socks := 6
  let gray_pairs := (gray_socks.choose 2)
  let white_pairs := (white_socks.choose 2)
  let same_color_pairs := gray_pairs + white_pairs
  same_color_pairs / total_pairs = (81 / 153) :=
by
  sorry

end socks_probability_l700_700434


namespace least_three_digit_7_heavy_l700_700771

/-- A number is 7-heavy if the remainder when the number is divided by 7 is greater than 4. -/
def is_7_heavy (n : ℕ) : Prop := n % 7 > 4

/-- The statement to be proved: The least three-digit 7-heavy number is 104. -/
theorem least_three_digit_7_heavy : ∃ n, 100 ≤ n ∧ n < 1000 ∧ is_7_heavy(n) ∧ ∀ m, 100 ≤ m ∧ m < 1000 ∧ is_7_heavy(m) → n ≤ m :=
begin
    use 104,
    split,
    { exact dec_trivial, },
    split,
    { exact dec_trivial, },
    split,
    { change 104 % 7 > 4,
      exact dec_trivial, },
    { intros m h1 h2,
      sorry
    }
end

end least_three_digit_7_heavy_l700_700771


namespace factorize_x_squared_minus_four_l700_700451

theorem factorize_x_squared_minus_four (x : ℝ) : x^2 - 4 = (x + 2) * (x - 2) :=
by
  sorry

end factorize_x_squared_minus_four_l700_700451


namespace factorial_division_result_l700_700402

theorem factorial_division_result : (11.factorial / (7.factorial * 4.factorial)) * 2 = 660 := 
by
  sorry

end factorial_division_result_l700_700402


namespace combined_resistance_parallel_l700_700340

theorem combined_resistance_parallel (x y : ℝ) (r : ℝ) (hx : x = 3) (hy : y = 5) 
  (h : 1 / r = 1 / x + 1 / y) : r = 15 / 8 :=
by
  sorry

end combined_resistance_parallel_l700_700340


namespace find_f_pi_l700_700932

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ :=
  Real.tan (ω * x + Real.pi / 3)

theorem find_f_pi (ω : ℝ) (h_positive : ω > 0) (h_period : Real.pi / ω = 3 * Real.pi) :
  f (ω := ω) Real.pi = -Real.sqrt 3 :=
by
  -- ω is given to be 1/3 by the condition h_period, substituting that 
  -- directly might be clearer for stating the problem accurately
  have h_omega : ω = 1 / 3 := by
    sorry
  rw [h_omega]
  sorry


end find_f_pi_l700_700932


namespace seating_arrangement_l700_700730

theorem seating_arrangement (D R : ℕ) (democratic_leader : ℕ) (total_politicians : ℕ) :
  D = 6 → R = 5 → total_politicians = D + R →
  democratic_leader = 1 →
  (total_politicians - 1)! = 3628800 :=
by
  intros hD hR htotal hleader
  sorry

end seating_arrangement_l700_700730


namespace m_above_x_axis_m_on_line_l700_700107

namespace ComplexNumberProblem

def above_x_axis (m : ℝ) : Prop :=
  m^2 - 2 * m - 15 > 0

def on_line (m : ℝ) : Prop :=
  2 * m^2 + 3 * m - 4 = 0

theorem m_above_x_axis (m : ℝ) : above_x_axis m → (m < -3 ∨ m > 5) :=
  sorry

theorem m_on_line (m : ℝ) : on_line m → 
  (m = (-3 + Real.sqrt 41) / 4) ∨ (m = (-3 - Real.sqrt 41) / 4) :=
  sorry

end ComplexNumberProblem

end m_above_x_axis_m_on_line_l700_700107


namespace maximum_OB_length_l700_700695

noncomputable def max_OB_length (O A B : Point) (angle_O_p : angle O A O B = 45) (AB_length : dist A B = 2) : ℝ :=
  2 * Real.sqrt 2

theorem maximum_OB_length (O A B : Point) (h₁ : angle O A O B = 45) (h₂ : dist A B = 2) :
  ∃ OB_max : ℝ, OB_max = 2 * Real.sqrt 2 :=
by
  use 2 * Real.sqrt 2
  sorry

end maximum_OB_length_l700_700695


namespace rhombus_perimeter_l700_700284

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  let s := real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  in 4 * s = 16 * real.sqrt 13 := 
by {
  let a := d1 / 2,
  let b := d2 / 2,
  have h_a : a = 12 := by { rw h1, norm_num },
  have h_b : b = 8 := by { rw h2, norm_num },
  have h_s : s = real.sqrt (a^2 + b^2) := by refl,
  have h_s_val : s = 4 * real.sqrt 13 := by {
    rw [h_a, h_b], 
    norm_num,
    simp [real.sqrt_mul (show 16 > 0, by norm_num), show 13 > 0, by norm_num]
  },
  rw h_s_val,
  norm_num
}

end rhombus_perimeter_l700_700284


namespace geometric_sequence_seventh_term_l700_700913

-- Define the positive sequence and conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n m k : ℕ, a n * a m = a k ^ 2

axioms 
  (a : ℕ → ℝ)
  (h1 : is_geometric_sequence a)
  (h2 : ∀ n, a n > 0)
  (h3 : a 4 * a 10 = 16)

-- The theorem stating that a_7 == 4
theorem geometric_sequence_seventh_term : a 7 = 4 :=
by
  sorry

end geometric_sequence_seventh_term_l700_700913


namespace valid_seating_arrangements_l700_700585

theorem valid_seating_arrangements :
  let total_arrangements := Nat.factorial 10
  let restricted_arrangements := Nat.factorial 7 * Nat.factorial 4
  total_arrangements - restricted_arrangements = 3507840 :=
by
  sorry

end valid_seating_arrangements_l700_700585


namespace typing_time_is_approximately_l700_700204

-- Definitions for conditions
constant words_per_minute : ℝ := 32
constant total_words : ℝ := 7125

-- Definition of the derived total time in hours
def time_in_hours : ℝ := (total_words / words_per_minute) / 60

-- Proving that the time in hours is approximately 3.71
theorem typing_time_is_approximately :
  |time_in_hours - 3.71| < 0.01 := by
  -- The proof will go here.
  sorry

end typing_time_is_approximately_l700_700204


namespace least_three_digit_7_heavy_l700_700768

-- Define what it means for a number to be "7-heavy"
def is_7_heavy(n : ℕ) : Prop := n % 7 > 4

-- Smallest three-digit number
def smallest_three_digit_number : ℕ := 100

-- Least three-digit 7-heavy whole number
theorem least_three_digit_7_heavy : ∃ n, smallest_three_digit_number ≤ n ∧ is_7_heavy(n) ∧ ∀ m, smallest_three_digit_number ≤ m ∧ is_7_heavy(m) → n ≤ m := 
  sorry

end least_three_digit_7_heavy_l700_700768


namespace minimum_value_expression_l700_700615

open Real

theorem minimum_value_expression (x y z : ℝ) (hxyz : x * y * z = 1 / 2) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x + y) * (2 * y + 3 * z) * (x * z + 2) ≥ 4 * sqrt 6 :=
sorry

end minimum_value_expression_l700_700615


namespace find_OP_l700_700198

variables (ABC : Type) [triangle ABC]
variables (A B C P Q O : ABC)
variables (OQ CQ OP : ℝ)

-- Conditions from the problem
def centroid (O : ABC) (Q : ABC) (P : ABC) (OQ : ℝ) (CQ : ℝ) : Prop :=
  OQ = 4 ∧ CQ = 3 * OQ

theorem find_OP (h1 : centroid O Q P OQ CQ) : OP = 8 :=
by
  sorry

end find_OP_l700_700198


namespace integer_roots_of_polynomial_l700_700426

theorem integer_roots_of_polynomial :
  let p : ℤ[X] := X^3 - 2*X^2 + 3*X - 17 in
  ∀ x : ℤ, p.eval x = 0 → x ∈ {-17, -1, 1, 17} :=
by
  sorry

end integer_roots_of_polynomial_l700_700426


namespace smallest_a_l700_700225

variable {a b : ℝ}

theorem smallest_a (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : ∀ x : ℤ, sin (a * x + b) = sin (17 * x + π)) :
  a = 17 :=
sorry

end smallest_a_l700_700225


namespace simplify_fractional_expression_l700_700609

variable {a b c : ℝ}

theorem simplify_fractional_expression 
  (h_nonzero_a : a ≠ 0)
  (h_nonzero_b : b ≠ 0)
  (h_nonzero_c : c ≠ 0)
  (h_sum : a + b + c = 1) :
  (1 / (b^2 + c^2 - a^2)) + (1 / (a^2 + c^2 - b^2)) + (1 / (a^2 + b^2 - c^2)) = 
  3 / (2 * (-b - c + b * c)) :=
sorry

end simplify_fractional_expression_l700_700609


namespace sum_of_all_possible_values_l700_700147

theorem sum_of_all_possible_values (x y : ℤ) :
  x^2 - x * y + x = 2018 ∧ y^2 - y * x - y = 52 → (x - y = 45 ∨ x - y = -46) → 45 + (-46) = -1 :=
begin
  sorry
end

end sum_of_all_possible_values_l700_700147


namespace inequality_proof_l700_700605

variable {f : ℝ → ℝ}
variable (L : ℝ)
variable (h_diff : Differentiable ℝ f)
variable (h_pos : ∀ x, f x > 0)
variable (h_lip : ∀ x y, |f' x - f' y| ≤ L * |x - y|)

theorem inequality_proof : ∀ x, (f' x)^2 < 2 * L * f x := by
  sorry

end inequality_proof_l700_700605


namespace count_permutations_perception_l700_700810

def factorial : ℕ → ℕ
| 0     := 1
| (n+1) := (n+1) * factorial n

def num_permutations (word : String) : ℕ :=
  let total_letters := word.length
  let freq_map := word.to_list.groupBy id
  let fact_chars := freq_map.toList.map (λ (c, l) => factorial l.length)
  factorial total_letters / fact_chars.foldl (*) 1

theorem count_permutations_perception :
  num_permutations "PERCEPTION" = 907200 := by
  sorry

end count_permutations_perception_l700_700810


namespace isosceles_triangle_largest_angle_l700_700578

theorem isosceles_triangle_largest_angle (a b c : ℝ) (h1 : a = b) (h2 : b_angle = 50) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) 
  (h6 : a + b + c = 180) : c ≥ a ∨ c ≥ b → c = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l700_700578


namespace number_of_solutions_for_ffx_eq_10_l700_700175

def f (x : ℝ) : ℝ :=
  if x ≤ 2 then x^2 + 2 else x - 1

theorem number_of_solutions_for_ffx_eq_10 :
  set.count { x : ℝ | f(f(x)) = 10 } = 2 := 
by sorry

end number_of_solutions_for_ffx_eq_10_l700_700175


namespace david_coin_flips_l700_700023

open BigOperators

def probability_three_heads (p_heads : ℚ) (p_tails : ℚ) (n_flips : ℕ) (k_heads : ℕ) : ℚ :=
  let binomial_coefficient (n k : ℕ) : ℕ := nat.choose n k
  binomial_coefficient n_flips k_heads * (p_heads ^ k_heads) * (p_tails ^ (n_flips - k_heads))

theorem david_coin_flips :
  probability_three_heads (1/3) (2/3) 8 3 = 1792 / 6561 :=
by
  unfold probability_three_heads
  unfold binomial_coefficient
  norm_num -- perform the necessary numeric computations and simplifications
  sorry -- skipping the detailed proof steps

end david_coin_flips_l700_700023


namespace count_terminating_nonzero_tenths_l700_700468

theorem count_terminating_nonzero_tenths :
  (∃ n <= 50, ∀ a b, n = 2^a * 5^b ∧ decimal_has_nonzero_tenths (1/n)) ↔ (∃ m, m = 6) :=
sorry

def decimal_has_nonzero_tenths (x : ℚ) : Prop :=
  let dec := (x.num : ℤ) / (10 * x.denom) in
  dec % 10 != 0

end count_terminating_nonzero_tenths_l700_700468


namespace isosceles_triangle_largest_angle_l700_700579

theorem isosceles_triangle_largest_angle (a b c : ℝ) (h1 : a = b) (h2 : b_angle = 50) (h3 : 0 < a) (h4 : 0 < b) (h5 : 0 < c) 
  (h6 : a + b + c = 180) : c ≥ a ∨ c ≥ b → c = 80 :=
by
  sorry

end isosceles_triangle_largest_angle_l700_700579


namespace value_of_ak_l700_700123

noncomputable def Sn (n : ℕ) : ℤ := n^2 - 9 * n
noncomputable def a (n : ℕ) : ℤ := Sn n - Sn (n - 1)

theorem value_of_ak (k : ℕ) (hk : 5 < a k ∧ a k < 8) : a k = 6 := by
  sorry

end value_of_ak_l700_700123


namespace two_a_minus_b_equals_four_l700_700470

theorem two_a_minus_b_equals_four (a b : ℕ) 
    (consec_integers : b = a + 1)
    (min_a : min (Real.sqrt 30) a = a)
    (min_b : min (Real.sqrt 30) b = Real.sqrt 30) : 
    2 * a - b = 4 := 
sorry

end two_a_minus_b_equals_four_l700_700470


namespace mary_friends_count_l700_700630

-- Definitions based on conditions
def total_stickers := 50
def stickers_left := 8
def total_students := 17
def classmates := total_students - 1 -- excluding Mary

-- Defining the proof problem
theorem mary_friends_count (F : ℕ) (h1 : 4 * F + 2 * (classmates - F) = total_stickers - stickers_left) :
  F = 5 :=
by sorry

end mary_friends_count_l700_700630


namespace michael_truck_meeting_times_l700_700242

def michael_speed : ℕ := 6 -- Michael's speed in feet per second
def bench_spacing : ℕ := 300 -- Distance between benches in feet
def truck_speed : ℕ := 8 -- Truck's speed in feet per second
def truck_stop_time : ℕ := 45 -- Truck's stopping time at each bench in seconds

theorem michael_truck_meeting_times :
  ∀ (initial_michael_position initial_truck_position : ℕ),
  initial_michael_position = 0 →
  initial_truck_position = bench_spacing →
  ∃ (meetings : ℕ), meetings = 2 :=
begin
  sorry
end

end michael_truck_meeting_times_l700_700242


namespace remainder_2021_2025_mod_17_l700_700701

theorem remainder_2021_2025_mod_17 : 
  (2021 * 2022 * 2023 * 2024 * 2025) % 17 = 0 :=
by 
  -- Proof omitted for brevity
  sorry

end remainder_2021_2025_mod_17_l700_700701


namespace operation_value_l700_700669

def operation (a b : ℝ) : ℝ := (a / b) + (b / a)

theorem operation_value : operation 4 8 = 5 / 2 := 
by 
sorюч-/- светлый слово

end operation_value_l700_700669


namespace total_games_in_conference_l700_700382

theorem total_games_in_conference (teams_per_division: ℕ) (divisions: ℕ) (intra_division_games: ℕ) (inter_division_games: ℕ) :
  teams_per_division = 8 → 
  divisions = 2 → 
  intra_division_games = 3 → 
  inter_division_games = 2 → 
  let intradivision_total_games := (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * divisions,
      interdivision_games_per_team := teams_per_division * inter_division_games,
      interdivision_total_games := (interdivision_games_per_team * (teams_per_division * divisions)) / 2,
      total_games := intradivision_total_games + interdivision_total_games
  in total_games = 296 :=
by
  intros h1 h2 h3 h4
  let intradivision_total_games :=
    (teams_per_division * (teams_per_division - 1) / 2) * intra_division_games * divisions
  let interdivision_games_per_team :=
    teams_per_division * inter_division_games
  let interdivision_total_games :=
    (interdivision_games_per_team * (teams_per_division * divisions)) / 2
  let total_games := intradivision_total_games + interdivision_total_games
  show total_games = 296
  sorry

end total_games_in_conference_l700_700382


namespace factorial_units_digit_l700_700173

theorem factorial_units_digit (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (hba : a < b) : 
  ¬ (∃ k : ℕ, (b! - a!) % 10 = 7) := 
sorry

end factorial_units_digit_l700_700173


namespace negation_of_universal_statement_l700_700563

def logarithmic_function (f : ℝ → ℝ) := sorry -- Define what it means to be a logarithmic function
def monotonic_function (f : ℝ → ℝ) := sorry -- Define what it means to be a monotonic function

def P : Prop := ∀ f, logarithmic_function f → monotonic_function f

theorem negation_of_universal_statement : ¬P ↔ ∃ f, logarithmic_function f ∧ ¬monotonic_function f := by
  sorry

end negation_of_universal_statement_l700_700563


namespace total_amount_shared_l700_700804

theorem total_amount_shared (total_amount : ℝ) 
  (h_debby : total_amount * 0.25 = (total_amount - 4500))
  (h_maggie : total_amount * 0.75 = 4500) : total_amount = 6000 :=
begin
  sorry
end

end total_amount_shared_l700_700804


namespace solve_fractional_equation_l700_700080

theorem solve_fractional_equation :
  {x : ℝ | 1 / (x^2 + 8 * x - 6) + 1 / (x^2 + 5 * x - 6) + 1 / (x^2 - 14 * x - 6) = 0}
  = {3, -2, -6, 1} :=
by
  sorry

end solve_fractional_equation_l700_700080


namespace functional_inequality_solution_l700_700846

theorem functional_inequality_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, 2 + f x * f y ≤ x * y + 2 * f (x + y + 1)) ↔ (∀ x : ℝ, f x = x + 2) :=
by
  sorry

end functional_inequality_solution_l700_700846


namespace find_integer_l700_700085

theorem find_integer (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 9) (h3 : n ≡ -437 [MOD 10]) : n = 3 :=
by sorry

end find_integer_l700_700085


namespace parabola_through_point_intersects_axes_at_two_points_l700_700864

open Real

def quadratic_function_a := λ a x : ℝ, x^2 - 4 * x - 3 + a

theorem parabola_through_point (a : ℝ) :
  (∃ y, quadratic_function_a a 0 = y ∧ y = 1) → a = 4 := by
  sorry

theorem intersects_axes_at_two_points (a : ℝ) :
  (∀ x y : ℝ, quadratic_function_a a x = 0 → quadratic_function_a a y = 0) →
  (∃ b, b^2 - 4 * 1 * (-3 + a) = 0) → a = 7 := by
  sorry

end parabola_through_point_intersects_axes_at_two_points_l700_700864


namespace minimum_area_triangle_l700_700900

theorem minimum_area_triangle :
  ∃ (a b c : ℝ) (e : ℝ), 
    (∀ (x y : ℝ), (x, y) = (0, 1) → 
    ((y^2 = b^2 → x^2/a^2 + y^2 = 1) ∧ 
      e = sqrt 2 / 2 ∧ 
      b = 1 ∧ a = sqrt 2 ∧ c = sqrt a^2 - b^2) →
    ∀ (l l1 l2 : ℝ → Prop),
    (l1 = λ p : ℝ × ℝ, p.1 - p.2 = 0) ∧ 
    (l2 = λ p : ℝ × ℝ, p.1 + p.2 = 0) ∧ 
    (∃ (P Q O : ℝ × ℝ),
      l (P.1) = k * P.1 + m ∧ 
      l (Q.1) = k * Q.1 + m ∧ 
      O = (0, 0) ∧ triangle_area O P Q → 
      ∀ k : ℝ, k ≠ 1 ∧ k ≠ -1 →
      1) := sorry

end minimum_area_triangle_l700_700900


namespace interval_of_monotonic_increase_range_of_k_l700_700624

noncomputable def f (x : ℝ) : ℝ := -2 * real.cos x - x
noncomputable def g (x k : ℝ) : ℝ := real.log x + k / x
noncomputable def h (x : ℝ) : ℝ := -2 * x - x * real.log x

theorem interval_of_monotonic_increase (k : ℤ) : 
  ∀ x, (2 * k * real.pi + (real.pi / 6) < x ∧ x < 2 * k * real.pi + (5 * real.pi / 6)) → 
  2 * real.sin x - 1 > 0 := 
sorry

theorem range_of_k : 
  ∀ k x : ℝ, 
  (0 ≤ x ∧ x ≤ 1) → 
  f 0 = -2 → 
  (f 0 < g x k) → 
  k > -1 + (1/2) * real.log 2 :=
sorry

end interval_of_monotonic_increase_range_of_k_l700_700624


namespace polygon_interior_angles_sum_l700_700984

theorem polygon_interior_angles_sum {n : ℕ} 
  (h1 : ∀ (k : ℕ), k > 2 → (360 = k * 40)) :
  180 * (9 - 2) = 1260 :=
by
  sorry

end polygon_interior_angles_sum_l700_700984


namespace mean_of_remaining_students_l700_700994

theorem mean_of_remaining_students
  (n : ℕ) (h : n > 20)
  (mean_score_first_15 : ℝ)
  (mean_score_next_5 : ℝ)
  (overall_mean_score : ℝ) :
  mean_score_first_15 = 10 →
  mean_score_next_5 = 16 →
  overall_mean_score = 11 →
  ∀ a, a = (11 * n - 230) / (n - 20) := by
sorry

end mean_of_remaining_students_l700_700994


namespace time_after_increment_l700_700977

-- Define the current time in minutes
def current_time_minutes : ℕ := 15 * 60  -- 3:00 p.m. in minutes

-- Define the time increment in minutes
def time_increment : ℕ := 1567

-- Calculate the total time in minutes after the increment
def total_time_minutes : ℕ := current_time_minutes + time_increment

-- Convert total time back to hours and minutes
def calculated_hours : ℕ := total_time_minutes / 60
def calculated_minutes : ℕ := total_time_minutes % 60

-- The expected hours and minutes after the increment
def expected_hours : ℕ := 17 -- 17:00 hours which is 5:00 p.m.
def expected_minutes : ℕ := 7 -- 7 minutes

theorem time_after_increment :
  (calculated_hours - 24 * (calculated_hours / 24) = expected_hours) ∧ (calculated_minutes = expected_minutes) :=
by
  sorry

end time_after_increment_l700_700977


namespace floor_eq_solution_l700_700060

theorem floor_eq_solution (x : ℝ) :
  (⟦⟦3 * x⟧ - 1 / 3⟧ = ⟦x + 3⟧) ↔ (5 / 3 ≤ x ∧ x < 7 / 3) :=
sorry

end floor_eq_solution_l700_700060


namespace vasya_can_win_l700_700840

-- Define the capacities of the pots
def pots := (3, 5, 7)

-- Define the initial state of the pots (all empty)
def initial_state := (0, 0, 0)

-- Define the state transitions (actions by Vasya and Dima)
inductive action 
| fill_3 | fill_5 | fill_7
| transfer_3_to_5 | transfer_3_to_7
| transfer_5_to_3 | transfer_5_to_7
| transfer_7_to_3 | transfer_7_to_5
| empty_3 | empty_5 | empty_7

-- Define what it means to be in a game state
def game_state := (ℕ, ℕ, ℕ)

-- Define the optimal play condition
def optimal_play (s: game_state) : Prop := sorry  -- specify optimal moves here

-- Define the condition for winning the game
def can_measure_1_liter := ∃ s: game_state, optimal_play s ∧ (s = (1, _, _) ∨ s = (_, 1, _) ∨ s = (_, _, 1))

theorem vasya_can_win : can_measure_1_liter :=
begin
  sorry -- proof omitted
end

end vasya_can_win_l700_700840


namespace rhombus_perimeter_l700_700282

theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) : 
  let s := real.sqrt ((d1 / 2)^2 + (d2 / 2)^2)
  in 4 * s = 16 * real.sqrt 13 := 
by {
  let a := d1 / 2,
  let b := d2 / 2,
  have h_a : a = 12 := by { rw h1, norm_num },
  have h_b : b = 8 := by { rw h2, norm_num },
  have h_s : s = real.sqrt (a^2 + b^2) := by refl,
  have h_s_val : s = 4 * real.sqrt 13 := by {
    rw [h_a, h_b], 
    norm_num,
    simp [real.sqrt_mul (show 16 > 0, by norm_num), show 13 > 0, by norm_num]
  },
  rw h_s_val,
  norm_num
}

end rhombus_perimeter_l700_700282


namespace floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700054

open Real

theorem floor_floor_3x_sub_third_eq_floor_x_add_3 (x : ℝ) :
  (⌊⌊3 * x⌋ - 1 / 3⌋ = ⌊x + 3⌋) ↔ (5 / 3 ≤ x ∧ x < 3) := 
sorry

end floor_floor_3x_sub_third_eq_floor_x_add_3_l700_700054


namespace polynomial_remainder_l700_700333

theorem polynomial_remainder :
  ∀ z : ℂ, ∃ r : ℂ,
    (2 * z^4 - 3 * z^3 + 5 * z^2 - 7 * z + 6) =
    (2 * z - 3) * (z^3 + z^2 - 4 * z + 5) + r ∧
    r = 21 :=
begin
  intro z,
  use 21,
  split,
  { sorry },
  { refl }
end

end polynomial_remainder_l700_700333


namespace find_x_of_perpendicular_l700_700534

open Real

variables (x : ℝ)
def e : EuclideanSpace ℝ (Fin 3) := ![1, 2, 1]
def n : EuclideanSpace ℝ (Fin 3) := ![1/2, x, 1/2]

theorem find_x_of_perpendicular (h : inner e n = 0) : x = 1 := by 
  sorry

end find_x_of_perpendicular_l700_700534


namespace train_crossing_time_l700_700387

theorem train_crossing_time :
  ∀ (length_train length_platform1 length_platform2 time1 speed total_distance time2 : ℝ),
    length_train = 230 ∧
    length_platform1 = 130 ∧
    length_platform2 = 250 ∧
    time1 = 15 ∧
    speed = (length_train + length_platform1) / time1 ∧
    total_distance = length_train + length_platform2 →
    time2 = total_distance / speed →
    time2 = 20 :=
by
  intros length_train length_platform1 length_platform2 time1 speed total_distance time2
  assume h
  cases h with ht1 ht2
  cases ht2 with hp1 ht3
  cases ht3 with hp2 ht4
  cases ht4 with ht1_eq_time15 hs_eq_speed
  cases hs_eq_speed with ht_total_dist_htot time2_eq
  rw [ht1, hp1, hp2, ht1_eq_time15] at hs_eq_speed
  rw [ht_total_dist_htot] at time2_eq
  rw [time2_eq]
  sorry

end train_crossing_time_l700_700387


namespace original_area_of_doubled_rectangle_l700_700295

theorem original_area_of_doubled_rectangle (A_new : ℝ) (h : A_new = 32) :
  ∃ A : ℝ, A * 4 = A_new ∧ A = 8 :=
by {
  use 8,
  split,
  { norm_num, exact h.symm },
  { rfl }
}

end original_area_of_doubled_rectangle_l700_700295


namespace find_slope_k_l700_700511

-- Definitions based on the conditions given in the problem
def slope_line1 : ℝ := sqrt 3 / 3
def inclination_line1 : ℝ := Real.arctan slope_line1
def inclination_line2 : ℝ := 2 * inclination_line1
def slope_line2 : ℝ := Real.tan inclination_line2

-- The proof statement: Prove that k = sqrt 3 given the conditions
theorem find_slope_k : 
  let k := slope_line2 
  k = sqrt 3 := by 
    sorry

end find_slope_k_l700_700511


namespace dot_product_AC_AD_eq_sqrt3_l700_700594

variables {V : Type*} [inner_product_space ℝ V]

-- Given conditions
variables (A B C D : V) (A B C D : V)
variable (orthogonal_AD_AB : inner_product_space.is_ortho ℝ (A - D) (A - B))
variable (length_AD : ∥D - A∥ = 1)
variable (BC_eq_sqrt3_BD : C - B = (sqrt 3) • (D - B))


--Prove the dot product
theorem dot_product_AC_AD_eq_sqrt3 :
  (C - A) ⬝ (D - A) = sqrt 3 :=
sorry

end dot_product_AC_AD_eq_sqrt3_l700_700594


namespace gcd_128_144_512_l700_700698

-- Definitions based on the conditions given
def fact128 := 2^7
def fact144 := 2^4 * 3^2
def fact512 := 2^9

-- The statement of the problem translated into Lean 4
theorem gcd_128_144_512 : gcd 128 (gcd 144 512) = 16 :=
by
  rw [gcd_comm 144 512, gcd_assoc]
  rw [gcd_eq_right (dvd_refl 128)]
  rw [gcd_comm 128, gcd_assoc]
  rw [gcd_eq_right (dvd_refl 128)]
  sorry

end gcd_128_144_512_l700_700698


namespace floor_equation_solution_l700_700048

theorem floor_equation_solution {x : ℝ} (h1 : ⌊⌊ 3 * x ⌋₊ - (1 / 3)⌋₊ = ⌊ x + 3 ⌋₊) (h2 : ⌊ 3 * x ⌋₊ ∈ ℤ) : 
  2 ≤ x ∧ x < 7 / 3 :=
sorry

end floor_equation_solution_l700_700048


namespace sequence_converges_to_one_l700_700604

theorem sequence_converges_to_one (a : ℕ → ℝ) (x : ℕ → ℝ) 
  (h1 : ∀ n, 1 / 2 < a n ∧ a n < 1) 
  (h2 : x 0 = 0) 
  (h3 : ∀ n, x (n + 1) = (a (n + 1) + x n) / (1 + a (n + 1) * x n)) : 
  ∃ l, (∀ ε > 0, ∃ N, ∀ n ≥ N, | x n - l | < ε) ∧ l = 1 := 
by
  sorry

end sequence_converges_to_one_l700_700604


namespace option_D_may_not_hold_l700_700114

theorem option_D_may_not_hold (a b c : ℝ) (h : a = b) : c = 0 → ¬ (a / c = b / c) :=
by
  intro hc
  rw [hc, div_zero, div_zero, eq_self_iff_true, not_true]
  exact false.elim sorry

end option_D_may_not_hold_l700_700114
