import Mathlib

namespace least_positive_integer_divisible_by_first_ten_integers_l469_469385

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469385


namespace geometric_sequence_formula_l469_469832

theorem geometric_sequence_formula 
  (a : ℕ → ℕ) (b : ℕ → ℕ) (c : ℕ → ℕ) (S : ℕ → ℕ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = q * a n)
  (h_q_gt : q > 1)
  (h_a3a5 : a 3 + a 5 = 40)
  (h_a4 : a 4 = 16)
  (h_bn : ∀ n, b n = Int.log 2 (a n)) :
  (∀ n, a n = 2^n) ∧ (∀ n, S n = (n - 1) * 2^(n + 1) + 2) := 
by
  sorry

end geometric_sequence_formula_l469_469832


namespace find_difference_l469_469273

variable (a b c d e f : ℝ)

-- Conditions
def cond1 : Prop := a - b = c + d + 9
def cond2 : Prop := a + b = c - d - 3
def cond3 : Prop := e = a^2 + b^2
def cond4 : Prop := f = c^2 + d^2
def cond5 : Prop := f - e = 5 * a + 2 * b + 3 * c + 4 * d

-- Problem Statement
theorem find_difference (h1 : cond1 a b c d) (h2 : cond2 a b c d) (h3 : cond3 a b e) (h4 : cond4 c d f) (h5 : cond5 a b c d e f) : a - c = 3 :=
sorry

end find_difference_l469_469273


namespace sequence_general_formula_sum_l469_469133

-- Definitions related to the sequence and its conditions
def sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, (3 - a (n + 1)) * (6 + a n) = 18 ∧ a 1 = 1

-- Definition of the general formula
def general_formula (a : ℕ → ℝ) : Prop :=
  ∀ n, a n = 3 / (2 ^ (n + 1) - 1)

-- Definition of the sum condition
def sum_condition (a : ℕ → ℝ) : Prop :=
  ∀ n, (∑ i in Finset.range n, a (i + 1)) < 3

-- Theorem encapsulating both parts of the problem
theorem sequence_general_formula_sum :
  ∃ (a : ℕ → ℝ), sequence a ∧ general_formula a ∧ sum_condition a :=
sorry

end sequence_general_formula_sum_l469_469133


namespace least_divisible_1_to_10_l469_469536

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469536


namespace lcm_first_ten_integers_l469_469629

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469629


namespace log_base_range_l469_469871

theorem log_base_range (a : ℝ) :
  (∃ b : ℝ, b = log (a - 2) (5 - a) ∧ a > 2 ∧ a ≠ 3 ∧ a < 5) ↔ (a ∈ set.Ioo 2 3 ∨ a ∈ set.Ioo 3 5) :=
by sorry

end log_base_range_l469_469871


namespace arrange_numbers_l469_469750

theorem arrange_numbers
  (a b c d : ℕ)
  (h1 : Nat.gcd a b = 1)
  (h2 : Nat.gcd a c = 1)
  (h3 : Nat.gcd a d = 1)
  (h4 : Nat.gcd b c = 1)
  (h5 : Nat.gcd b d = 1)
  (h6 : Nat.gcd c d = 1) :
  ∃ (ab cd ad bc abcd : ℕ),
  ab = a * b ∧
  cd = c * d ∧
  ad = a * d ∧
  bc = b * c ∧
  abcd = a * b * c * d ∧
  Nat.gcd ab abcd > 1 ∧
  Nat.gcd cd abcd > 1 ∧
  Nat.gcd ad abcd > 1 ∧
  Nat.gcd bc abcd > 1 ∧
  Nat.gcd ab cd = 1 ∧
  Nat.gcd ab ad = 1 ∧
  Nat.gcd ab bc = 1 ∧
  Nat.gcd cd ad = 1 ∧
  Nat.gcd cd bc = 1 :=
begin
  -- The actual proof steps will go here
  sorry
end

end arrange_numbers_l469_469750


namespace fewer_seats_right_l469_469895

-- Definitions based on conditions
def left_seats : Nat := 15
def seats_per_person : Nat := 3
def back_seat_capacity : Nat := 11
def bus_capacity : Nat := 92

-- Main statement
theorem fewer_seats_right :
  let right_seats := (bus_capacity - (left_seats * seats_per_person + back_seat_capacity)) / seats_per_person in
  left_seats - right_seats = 3 :=
by
 sorry

end fewer_seats_right_l469_469895


namespace lcm_first_ten_numbers_l469_469585

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469585


namespace triangle_is_not_right_triangle_l469_469818

theorem triangle_is_not_right_triangle (A B C : ℕ) (a b c : ℝ)
  (h1 : A : B : C = 3 : 4 : 5)
  (h2 : A + B + C = 180)
  (h3 : a = 5)
  (h4 : b = 12)
  (h5 : c = 13)
  (h6 : a^2 + b^2 = c^2)
  (h7 : A - B = C) : 
  (A ≠ 90 ∧ B ≠ 90 ∧ C ≠ 90) := 
by {
  sorry,
}

end triangle_is_not_right_triangle_l469_469818


namespace phi_value_l469_469331

theorem phi_value (f g : ℝ → ℝ) 
  (x1 x2 : ℝ) (φ : ℝ) 
  (cond1: ∀ x, g x = f (x - φ)) 
  (cond2 : ∀ x, f x = sin (2 * x))
  (cond3 : |f x1 - g x2| = 2) 
  (cond4 : |x1 - x2| = π / 3)
  (range_phi : 0 < φ ∧ φ < π / 2) : 
  φ = π / 6 :=
by 
  sorry

end phi_value_l469_469331


namespace planet_colonization_combinations_l469_469751

theorem planet_colonization_combinations :
  ∃ (a b : ℕ),
  (a ≤ 9) ∧ (b ≤ 9) ∧
  (3 * a + 2 * b = 27) ∧
   (nat.choose 9 a * nat.choose 9 b = 3024) :=
by
  sorry

end planet_colonization_combinations_l469_469751


namespace enlarged_banner_height_l469_469303

-- Definitions and theorem statement
theorem enlarged_banner_height 
  (original_width : ℝ) 
  (original_height : ℝ) 
  (new_width : ℝ) 
  (scaling_factor : ℝ := new_width / original_width ) 
  (new_height : ℝ := original_height * scaling_factor) 
  (h1 : original_width = 3) 
  (h2 : original_height = 2) 
  (h3 : new_width = 15): 
  new_height = 10 := 
by 
  -- The proof would go here
  sorry

end enlarged_banner_height_l469_469303


namespace maximum_area_of_triangle_abc_l469_469837

noncomputable def max_triangle_area (a b c : ℝ) (A B C : ℝ) [0 < a ∧ 0 < b ∧ 0 < c ∧ a = 2 ∧ 
  sin A ≠ 0 ∧ sin B ≠ 0 ∧ sin C ≠ 0 ∧ sin A - sin B / sin C = (c - b) / (2 + b)] : ℝ :=
if h : ∃ (b c : ℝ), 2 = a ∧ (b * c * sin A) / 2 = sqrt 3 then sqrt 3 else 0

theorem maximum_area_of_triangle_abc (a b c A B C : ℝ) 
  (h1 : a = 2) 
  (h2 : sin A ≠ 0)
  (h3 : sin B ≠ 0)
  (h4 : sin C ≠ 0)
  (h5 : (sin A - sin B) / sin C = (c - b) / (2 + b)) :
  max_triangle_area a b c A B C = sqrt 3 :=
sorry

end maximum_area_of_triangle_abc_l469_469837


namespace points_on_surface_l469_469766

variable (x y z : ℝ)

def lies_on_surface (x y z : ℝ) : Prop :=
  y^3 + 2 * x * y - 3 * z = 0

theorem points_on_surface :
  lies_on_surface 0 1 (1/3) ∧
  lies_on_surface 1 2 4 ∧
  ¬ lies_on_surface 1 1 2 :=
by
  -- Point A (0, 1, 1/3) lies on the surface
  unfold lies_on_surface
  sorry

  -- Point B (1, 2, 4) lies on the surface
  unfold lies_on_surface
  sorry

  -- Point C (1, 1, 2) does not lie on the surface
  unfold lies_on_surface
  sorry

end points_on_surface_l469_469766


namespace find_fourth_student_l469_469359

theorem find_fourth_student 
  (total_students : ℕ) (sample_size : ℕ) (sample_interval : ℕ)
  (sample_members : list ℕ) (h1 : total_students = 48) (h2 : sample_size = 4)
  (h3 : sample_interval = total_students / sample_size) (h4 : sample_members = [5, 29, 41]) :
  exists fourth_student : ℕ, fourth_student = 17 :=
by
  have sample_interval_correct : sample_interval = 12 := by
    rw [←h1, ←h2]
    exact nat.div_eq_of_eq_mul (48).symm
    
  have fourth_student_in_sample : 17 ∈ sample_members ∨ 17 = 17 := by
    sorry  -- the proof step that 17 should logically follow
  
  exact ⟨17, rfl⟩

end find_fourth_student_l469_469359


namespace isabella_weeks_worked_l469_469243

theorem isabella_weeks_worked :
  (let hourly_wage := 5
       hours_per_day := 5
       days_per_week := 6
       total_earnings := 1050 in
   let daily_earnings := hourly_wage * hours_per_day
       weekly_earnings := daily_earnings * days_per_week in
   total_earnings / weekly_earnings = 7) := 
sorry

end isabella_weeks_worked_l469_469243


namespace triangles_congruent_by_sss_l469_469740

theorem triangles_congruent_by_sss (h1 : ∀ (tri1 tri2 : Triangle), 
  (tri1.is_isosceles ∧ tri2.is_isosceles) ∧ 
  (tri1.side1 = 3 ∧ tri1.side2 = 6) ∧ 
  (tri2.side1 = 3 ∧ tri2.side2 = 6) → 
  tri1 ≅ tri2) : 
  ∀ (tri1 tri2 : Triangle), 
  (tri1.is_isosceles ∧ tri2.is_isosceles) ∧ 
  (tri1.side1 = 3 ∧ tri1.side2 = 6) ∧ 
  (tri2.side1 = 3 ∧ tri2.side2 = 6) → 
  tri1 ≅ tri2 := 
by 
  sorry

end triangles_congruent_by_sss_l469_469740


namespace probability_of_alternating_colors_l469_469705

-- The setup for our problem
variables (B W : Type) 

/-- A box contains 5 white balls and 5 black balls.
    Prove that the probability that all of my draws alternate colors
    is 1/126. -/
theorem probability_of_alternating_colors (W B : ℕ) (hw : W = 5) (hb : B = 5) :
  let total_ways := Nat.choose 10 5 in
  let successful_ways := 2 in
  successful_ways / total_ways = (1 : ℚ) / 126 :=
sorry

end probability_of_alternating_colors_l469_469705


namespace least_common_multiple_first_ten_integers_l469_469570

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469570


namespace system_of_equations_solution_l469_469987

theorem system_of_equations_solution (x y : ℝ) (h1 : |y - x| - (|x| / x) + 1 = 0) (h2 : |2 * x - y| + |x + y - 1| + |x - y| + y - 1 = 0) (hx : x ≠ 0) :
  (0 < x ∧ x ≤ 0.5 ∧ y = x) :=
by
  sorry

end system_of_equations_solution_l469_469987


namespace number_of_students_l469_469912

-- Define parameters and conditions
variables (B G : ℕ) -- number of boys and girls

-- Condition: each boy is friends with exactly two girls
axiom boys_to_girls : ∀ (B G : ℕ), 2 * B = 3 * G

-- Condition: total number of children in the class
axiom total_children : ∀ (B G : ℕ), B + G = 31

-- Define the theorem that proves the correct number of students
theorem number_of_students : (B G : ℕ) → 2 * B = 3 * G → B + G = 31 → B + G = 35 :=
by
  sorry

end number_of_students_l469_469912


namespace james_paid_l469_469928

-- Define the given conditions
def steak_egg_price : ℝ := 16
def dessert_price_james : ℝ := 5
def drink_price_james : ℝ := 3

def chicken_fried_steak_price : ℝ := 14
def dessert_price_friend : ℝ := 4
def drink_price_friend : ℝ := 2

def james_tip_percentage : ℝ := 0.20
def friend_tip_percentage : ℝ := 0.15

def james_meal_cost : ℝ := steak_egg_price + dessert_price_james + drink_price_james
def friend_meal_cost : ℝ := chicken_fried_steak_price + dessert_price_friend + drink_price_friend
def total_bill : ℝ := james_meal_cost + friend_meal_cost

def james_share_before_tip : ℝ := total_bill / 2
def james_tip : ℝ := james_share_before_tip * james_tip_percentage
def james_total_payment : ℝ := james_share_before_tip + james_tip

-- The theorem stating that James paid $26.40 in total
theorem james_paid (h : james_total_payment = 26.40) : james_total_payment = 26.40 := by
  exact h

end james_paid_l469_469928


namespace triangle_is_isosceles_l469_469906

open Real

variables (α β γ : ℝ) (a b : ℝ)

theorem triangle_is_isosceles
(h1 : a + b = tan (γ / 2) * (a * tan α + b * tan β)) :
α = β :=
by
  sorry

end triangle_is_isosceles_l469_469906


namespace floor_sqrt_27_squared_eq_25_l469_469080

theorem floor_sqrt_27_squared_eq_25 :
  (⌊Real.sqrt 27⌋)^2 = 25 :=
by
  have H1 : 5 < Real.sqrt 27 := sorry
  have H2 : Real.sqrt 27 < 6 := sorry
  have floor_sqrt_27_eq_5 : ⌊Real.sqrt 27⌋ = 5 := sorry
  rw floor_sqrt_27_eq_5
  norm_num

end floor_sqrt_27_squared_eq_25_l469_469080


namespace power_in_center_l469_469918

variables {G : Type*} [group G] [fintype G]
variables {n : ℕ} (h : n ≥ 1) (h_isomorphism : function.bijective (λ x : G, x^n))

theorem power_in_center (a : G) : a^(n-1) ∈ subgroup.center G :=
sorry

end power_in_center_l469_469918


namespace odd_function_ab_l469_469843

noncomputable def f (x : ℝ) : ℝ := (1/3) * x^3 + x^2 + x + 4/3

theorem odd_function_ab (a b : ℝ) (h : ∀ x : ℝ, g (x) = f (x + a) + b) (odd_g : ∀ x : ℝ, g (-x) = -g (x)) : a + b = -2 :=
by
  have g := λ x : ℝ, f (x + a) + b
  sorry

end odd_function_ab_l469_469843


namespace lcm_first_ten_integers_l469_469618

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469618


namespace probability_of_alternating_colors_l469_469704

-- The setup for our problem
variables (B W : Type) 

/-- A box contains 5 white balls and 5 black balls.
    Prove that the probability that all of my draws alternate colors
    is 1/126. -/
theorem probability_of_alternating_colors (W B : ℕ) (hw : W = 5) (hb : B = 5) :
  let total_ways := Nat.choose 10 5 in
  let successful_ways := 2 in
  successful_ways / total_ways = (1 : ℚ) / 126 :=
sorry

end probability_of_alternating_colors_l469_469704


namespace ratio_of_sums_l469_469833

noncomputable def infinite_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ :=
a1 * q ^ (n - 1)

noncomputable def geometric_sequence_sum (a1 q : ℝ) (n : ℕ) : ℝ :=
if q == 1 then n * a1 else a1 * (1 - q ^ n) / (1 - q)

theorem ratio_of_sums (a1 q : ℝ)
  (h1 : infinite_geometric_sequence a1 q 2,
   h2 : infinite_geometric_sequence a1 q 5,
   h3 : infinite_geometric_sequence a1 q 8,
   h_arithmetic : 6 * infinite_geometric_sequence a1 q 5 = infinite_geometric_sequence a1 q 2 + 9 * infinite_geometric_sequence a1 q 8) :
  geometric_sequence_sum a1 q 6 / geometric_sequence_sum a1 q 3 = 4 / 3 :=
sorry

end ratio_of_sums_l469_469833


namespace total_votes_l469_469982

theorem total_votes (initial_score : ℤ) (current_score : ℤ) (likes_percentage : ℝ) (x : ℕ) :
  initial_score = 0 →
  current_score = 50 →
  likes_percentage = 0.75 →
  current_score = (int.of_nat (likes_percentage * x)) - (int.of_nat ((1 - likes_percentage) * x)) →
  x = 100 :=
by
  intros h_initial_score h_current_score h_likes_percentage h_score_eq
  sorry

end total_votes_l469_469982


namespace least_positive_integer_divisible_by_first_ten_l469_469489

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469489


namespace roots_in_interval_l469_469738

theorem roots_in_interval (P : Polynomial ℝ) (h : ∀ i, P.coeff i = 1 ∨ P.coeff i = 0 ∨ P.coeff i = -1) : 
  ∀ x : ℝ, P.eval x = 0 → -2 ≤ x ∧ x ≤ 2 :=
by {
  -- Proof omitted
  sorry
}

end roots_in_interval_l469_469738


namespace least_common_multiple_of_first_ten_integers_l469_469365

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469365


namespace range_of_m_l469_469840

noncomputable def f (x : ℝ) : ℝ := -x^3 + 6 * x^2 - 9 * x

def tangents_condition (m : ℝ) : Prop := ∃ x : ℝ, (-3 * x^2 + 12 * x - 9) * (x + 1) + m = -x^3 + 6 * x^2 - 9 * x

theorem range_of_m (m : ℝ) : tangents_condition m → -11 < m ∧ m < 16 :=
sorry

end range_of_m_l469_469840


namespace lcm_first_ten_numbers_l469_469586

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469586


namespace production_rate_increase_is_60_l469_469743

-- Definitions of the conditions
def initialProductionRate : ℝ := 36  -- cogs per hour
def initialOrder : ℝ := 60  -- cogs
def additionalOrder : ℝ := 60  -- cogs
def overallAverageOutput : ℝ := 45  -- cogs per hour

-- The production rate of the assembly line after the speed was increased
def productionRateAfterIncrease (x : ℝ) : Prop :=
  (initialOrder / initialProductionRate) + (additionalOrder / x) = (initialOrder + additionalOrder) / overallAverageOutput

-- The statement we need to prove: the production rate after the increase is 60 cogs per hour
theorem production_rate_increase_is_60 : ∃ x : ℝ, x = 60 ∧ productionRateAfterIncrease x := by
  exists 60
  split
  rfl
  sorry

end production_rate_increase_is_60_l469_469743


namespace range_of_a_l469_469869

theorem range_of_a (a : ℝ) : (2 < a ∧ a < 5 ∧ a ≠ 3) ↔ (a ∈ set.Ioo 2 3 ∪ set.Ioo 3 5) :=
by
  sorry

end range_of_a_l469_469869


namespace carnations_count_l469_469026

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end carnations_count_l469_469026


namespace least_common_multiple_of_first_10_integers_l469_469522

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469522


namespace sum_of_fractions_l469_469846

def f (x : ℝ) : ℝ := sin ((5 * Real.pi / 3) * x + Real.pi / 6) + (3 * x) / (2 * x - 1)

theorem sum_of_fractions : 
  (∑ k in (Finset.range (2016)).filter (λ k, k % 2 = 1), f (k / 2016)) = 1512 := 
  sorry

end sum_of_fractions_l469_469846


namespace lcm_first_ten_integers_l469_469627

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469627


namespace original_price_of_sweater_l469_469733

theorem original_price_of_sweater (sold_price : ℝ) (discount : ℝ) (original_price : ℝ) 
    (h1 : sold_price = 120) (h2 : discount = 0.40) (h3: (1 - discount) * original_price = sold_price) : 
    original_price = 200 := by 
  sorry

end original_price_of_sweater_l469_469733


namespace locus_of_orthocenters_inside_circle_l469_469794

-- Define the circle and associated radius and center
variables {O : Point} {R : ℝ} (C : Circle O R)
variables (A B C : Point) (Δ : Triangle A B C)

-- Axiom: The definition of orthocenters and circles.
axiom orthocenter_of_triangle : Triangle → Point
axiom circle_contains_point : Circle → Point → Prop

-- Prove the locus of orthocenters lies inside the larger circle
theorem locus_of_orthocenters_inside_circle 
  (H : Point) 
  (h₁ : Triangle A B C)
  (h₂ : A, B, C ∈ C.points)
  (h₃ : H = orthocenter_of_triangle h₁) :
  circle_contains_point (Circle O (3 * R)) H :=
  sorry

end locus_of_orthocenters_inside_circle_l469_469794


namespace qin_jiushao_value_at_2_l469_469761

-- Define the polynomial function and its evaluation using Qin Jiushao's algorithm
def polynomial_function (x : ℝ) : ℝ :=
  2 * x^4 + 3 * x^3 + 5 * x - 4

def qin_jiushao_algorithm (x : ℝ) : ℝ :=
  let v₀ := x in
  let v₁ := 2 * x + 3 in
  let v₂ := v₁ * x + 0 in
  v₂

-- Prove that v₂ equals 14 when x equals 2
theorem qin_jiushao_value_at_2 : qin_jiushao_algorithm 2 = 14 := by
  sorry

end qin_jiushao_value_at_2_l469_469761


namespace solution_set_of_inequality_l469_469167

def f (x : ℝ) : ℝ := Real.exp x - Real.exp (-x)

theorem solution_set_of_inequality (x : ℝ) : f (2 * x - 1) + f (-x - 1) > 0 ↔ x > 2 :=
by
  sorry

end solution_set_of_inequality_l469_469167


namespace solution1_solution2_l469_469053

noncomputable def problem1 : ℝ := log 27 / log 3 + log 25 / log 10 + log 4 / log 10 + 7 ^ (log 2 / log 7) + (-9.8)^0

theorem solution1 : problem1 = 8 := by
  sorry

noncomputable def problem2 : ℝ := (8 / 27) ^ (-2 / 3) - (3 * π * π ^ (2 / 3)) + real.sqrt ((2 - π)^2)

theorem solution2 : problem2 = 1 / 4 := by
  sorry

end solution1_solution2_l469_469053


namespace least_common_multiple_of_first_ten_positive_integers_l469_469467

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469467


namespace least_common_multiple_of_first_10_integers_l469_469517

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469517


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469384

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469384


namespace area_quadrilateral_l469_469223

theorem area_quadrilateral (EF GH: ℝ) (EHG: ℝ) 
  (h1 : EF = 9) (h2 : GH = 12) (h3 : GH = EH) (h4 : EHG = 75) 
  (a b c : ℕ)
  : 
  (∀ (a b c : ℕ), 
  a = 26 ∧ b = 18 ∧ c = 6 → 
  a + b + c = 50) := 
sorry

end area_quadrilateral_l469_469223


namespace cubic_trinomial_degree_l469_469217

theorem cubic_trinomial_degree (n : ℕ) (P : ℕ → ℕ →  ℕ → Prop) : 
  (P n 5 4) → n = 3 := 
  sorry

end cubic_trinomial_degree_l469_469217


namespace max_Cauchy_distribution_convergence_l469_469270

open ProbabilityTheory

-- define the conditions described in step a)
def cauchy_pdf (x : ℝ) : ℝ := 1 / (π * (1 + x^2))

-- iid random variables following Cauchy distribution
axiom X : ℕ → ℝ
axiom X_iid : ∀ i j, i ≠ j → Indep X i X j
axiom X_Cauchy : ∀ n, CDF_X (λ X, cauchy_pdf Xᵢ) = 1

-- M_n is the maximum of the first n variables
def M_n (n : ℕ) : ℝ := finset.max' (finset.range n) X

-- T is a random variable following exponential distribution with parameter 1/π
axiom T_exp : ∀ t : ℝ, distribution.exp_dist 1/π t → ∀ x, (∑ n, X^(-t)) = E[t]

-- the main statement to be proved
theorem max_Cauchy_distribution_convergence :
  tendsto (λ (n : ℕ), (M_n n) / (n : ℝ)) at_top (distribution.exp_dist 1/π) := sorry

end max_Cauchy_distribution_convergence_l469_469270


namespace solve_for_x_l469_469098

theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → (x = 27) := by
  intro h
  sorry

end solve_for_x_l469_469098


namespace initial_money_amounts_l469_469354

theorem initial_money_amounts (a b c : ℝ) :
  (∀ a b c : ℝ,
      let a' := a - b - c,
          b' := 2 * b,
          c' := 2 * c in
      (∀ a' b' c' : ℝ,
          let a'' := 2 * a',
              b'' := b' - a' - c' + b',
              c'' := 4 * c in
          (∀ a'' b'' c'' : ℝ,
              let a''' := a'' + a'',
                  b''' := b'' + b'',
                  c''' := c'' - (a' + b') - c' in
              a''' = 8 ∧ b''' = 8 ∧ c''' = 8))) →
  (a = 13 ∧ b = 7 ∧ c = 4) :=
by
  intros
  sorry

end initial_money_amounts_l469_469354


namespace average_speed_of_person_l469_469721

theorem average_speed_of_person : 
  ∀ (d : ℕ) (v_up v_down : ℕ), 
  d = 400 → 
  v_up = 50 → 
  v_down = 80 → 
  (2 * d) / ((d / v_up) + (d / v_down)) = 800 / 13 := 
by
  intros d v_up v_down h1 h2 h3
  rw [h1, h2, h3]
  sorry

end average_speed_of_person_l469_469721


namespace batsman_average_46_innings_l469_469689

theorem batsman_average_46_innings {hs ls t_44 : ℕ} (h_diff: hs - ls = 180) (h_avg_44: t_44 = 58 * 44) (h_hiscore: hs = 194) : 
  (t_44 + hs + ls) / 46 = 60 := 
sorry

end batsman_average_46_innings_l469_469689


namespace lcm_first_ten_integers_l469_469622

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469622


namespace least_divisible_1_to_10_l469_469530

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469530


namespace least_common_multiple_first_ten_l469_469560

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469560


namespace least_common_multiple_of_first_ten_l469_469615

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469615


namespace find_x_l469_469110

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 12) : x = 27 :=
begin
  sorry
end

end find_x_l469_469110


namespace race_outcome_permutations_l469_469120

theorem race_outcome_permutations : 
  let participants := 6 in
  let outcomes := Nat.factorial participants / Nat.factorial (participants - 4) in
  outcomes = 360 :=
by {
  let participants := 6,
  have h : outcomes = Nat.factorial participants / Nat.factorial (participants - 4),
  exact h,
  simp only [Nat.factorial],
  norm_num,
}

end race_outcome_permutations_l469_469120


namespace least_divisible_1_to_10_l469_469532

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469532


namespace complex_number_problem_l469_469758

theorem complex_number_problem :
  ( (1 + Complex.i) / (1 - Complex.i) ) ^ 2017 = Complex.i := 
sorry

end complex_number_problem_l469_469758


namespace range_and_distance_l469_469841

-- Define the function f(x) with given parameters
noncomputable def f (x : ℝ) : ℝ := 4 * cos x * sin (x - (π / 3)) + sqrt 3

-- Define the interval for x
def interval (a b x : ℝ) : Prop := a ≤ x ∧ x ≤ b

-- The main theorem combining both proof problems
theorem range_and_distance :
  -- Part 1: Range of f(x) when x is in [0, π/2]
  (∀ x : ℝ, interval 0 (π / 2) x → -sqrt 3 ≤ f x ∧ f x ≤ 2) ∧
  -- Part 2: Shortest distance between intersection points with y = 1
  (∃ a b : ℝ, a < b ∧ f a = 1 ∧ f b = 1 ∧ ∀ c : ℝ, a < c ∧ c < b → f c ≠ 1 ∧ (b - a = π / 3)) :=
sorry

end range_and_distance_l469_469841


namespace movie_ticket_percentage_decrease_l469_469251

theorem movie_ticket_percentage_decrease (old_price new_price : ℝ) 
  (h1 : old_price = 100) 
  (h2 : new_price = 80) :
  ((old_price - new_price) / old_price) * 100 = 20 := 
by
  sorry

end movie_ticket_percentage_decrease_l469_469251


namespace least_positive_integer_divisible_by_first_ten_l469_469477

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469477


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469391

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469391


namespace least_positive_integer_divisible_by_first_ten_l469_469473

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469473


namespace scientific_notation_of_1_59_million_l469_469659

-- Define the condition that "million" means \(10^6\).
def million := 10^6

-- Define the given number in the problem.
def given_number := 1.59 * million

-- Prove that the given number is equivalent to \(1.59 \times 10^6\).
theorem scientific_notation_of_1_59_million : given_number = 1.59 * 10^6 := by
  -- This is where the proof would go
  sorry

end scientific_notation_of_1_59_million_l469_469659


namespace find_a7_l469_469148

variable {a : ℕ → ℝ}

-- Conditions
def is_increasing_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, 1 < q ∧ ∀ n : ℕ, a (n + 1) = a n * q

axiom a3_eq_4 : a 3 = 4
axiom harmonic_condition : (1 / a 1 + 1 / a 5 = 5 / 8)
axiom increasing_geometric : is_increasing_geometric_sequence a

-- The problem is to prove that a 7 = 16 given the above conditions.
theorem find_a7 : a 7 = 16 :=
by
  -- Proof goes here
  sorry

end find_a7_l469_469148


namespace base8_subtraction_l469_469084

theorem base8_subtraction : 
  let a := 6 * 8^2 + 4 * 8^1 + 1 * 8^0
  let b := 3 * 8^2 + 2 * 8^1 + 4 * 8^0
  a - b = 3 * 8^2 + 1 * 8^1 + 7 * 8^0 := 
by 
  sorry

end base8_subtraction_l469_469084


namespace mahmoud_gets_at_least_two_heads_l469_469961

def probability_of_at_least_two_heads := 1 - ((1/2)^5 + 5 * (1/2)^5)

theorem mahmoud_gets_at_least_two_heads (n : ℕ) (hn : n = 5) :
  probability_of_at_least_two_heads = 13 / 16 :=
by
  simp only [probability_of_at_least_two_heads, hn]
  sorry

end mahmoud_gets_at_least_two_heads_l469_469961


namespace least_common_multiple_of_first_ten_integers_l469_469380

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469380


namespace find_r_l469_469885

variable (p r s : ℝ)

theorem find_r (h : ∀ x : ℝ, (y : ℝ) = x^2 + p * x + r + s → (y = 10 ↔ x = -p / 2)) : r = 10 - s + p^2 / 4 := by
  sorry

end find_r_l469_469885


namespace integral_f_l469_469820

noncomputable def f (x : ℝ) : ℝ :=
if h : x < 1 then x^2 else 2 - x

theorem integral_f :
  ∫ x in 0 .. 2, f x = 5 / 6 :=
by
  sorry

end integral_f_l469_469820


namespace line_intersects_circle_probability_l469_469725

theorem line_intersects_circle_probability :
  let k := [(-1: ℝ), 1]
  let circle := (∃ x y : ℝ, (x - 5)^2 + y^2 = 9)
  ∃ k ∈ k, (∃ x y : ℝ, (x - 5)^2 + y^2 = 9 ∧ y = k * x) = 3 / 4 :=
sorry

end line_intersects_circle_probability_l469_469725


namespace sqrt_floor_square_l469_469076

theorem sqrt_floor_square {x : ℝ} (hx : 27 = x) (h1 : sqrt 25 < sqrt x) (h2 : sqrt x < sqrt 36) :
  (⌊sqrt x⌋.to_real)^2 = 25 :=
by {
  have hsqrt : 5 < sqrt x ∧ sqrt x < 6, by {
    split; linarith,
  },
  have h_floor_sqrt : ⌊sqrt x⌋ = 5, by {
    exact int.floor_eq_iff.mpr ⟨int.lt_floor_add_one.mpr hsqrt.2, hsqrt.1⟩,
  },
  rw h_floor_sqrt,
  norm_num,
  sorry  -- proof elided
}

end sqrt_floor_square_l469_469076


namespace least_common_multiple_1_to_10_l469_469493

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469493


namespace least_positive_integer_divisible_by_first_ten_l469_469480

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469480


namespace least_common_multiple_1_to_10_l469_469500

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469500


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469392

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469392


namespace Kesten_Spitzer_Whitman_theorem_l469_469677

noncomputable def i.i.d_sequence (X : ℕ → ℝ) : Prop :=
∀ (n m : ℕ), n ≠ m → Statistics.IID X n X m

noncomputable def partial_sum (X : ℕ → ℝ) (n : ℕ) : ℝ :=
∑ i in finset.range n, X i

noncomputable def count_distinct (X : ℕ → ℝ) (n : ℕ) : ℕ :=
(finset.range n).filter (λ i, X i != 0).card

noncomputable def first_return_to_zero (X : ℕ → ℝ) : ℕ :=
inf {n : ℕ | n > 0 ∧ X n = 0}

theorem Kesten_Spitzer_Whitman_theorem 
  (X : ℕ → ℝ) (h_iid : i.i.d_sequence X) 
  (h_distinct : ∀ n, (count_distinct (partial_sum X) n) = (count_distinct X n) ) :
  filter_at_top (λ n, (count_distinct (partial_sum X) n) / n) 
    (tendsto_const_nhds (π (λ n, first_return_to_zero (partial_sum X n)) = ∞))) sorry

end Kesten_Spitzer_Whitman_theorem_l469_469677


namespace angle_BOC_120_degrees_l469_469221

variables {A B C H I O : Type}
variables [Triangle ABC : Geometry]
variables (orthocenter H ABC : Geometry)
variables (incenter I ABC : Geometry)
variables (circumcenter O ABC : Geometry)
constants (is_cyclic_quad : CyclicQuadrilateral B H I C)

theorem angle_BOC_120_degrees (hBHC : angle H B C = angle B + angle C)
  (hBIC : angle I B C = 180 - (1 / 2) * angle A) :
  angle B O C = 120 :=
sorry

end angle_BOC_120_degrees_l469_469221


namespace alternating_sequence_probability_l469_469695

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l469_469695


namespace jason_attended_games_l469_469930

-- Define the conditions as given in the problem
def games_planned_this_month : ℕ := 11
def games_planned_last_month : ℕ := 17
def games_missed : ℕ := 16

-- Define the total number of games planned
def games_planned_total : ℕ := games_planned_this_month + games_planned_last_month

-- Define the number of games attended
def games_attended : ℕ := games_planned_total - games_missed

-- Prove that Jason attended 12 games
theorem jason_attended_games : games_attended = 12 := by
  -- The proof is omitted, but the theorem statement is required
  sorry

end jason_attended_games_l469_469930


namespace log_expression_value_l469_469769

def log_base (b x : ℝ) : ℝ := Real.log x / Real.log b
noncomputable def lg (x : ℝ) : ℝ := log_base 10 x

theorem log_expression_value :
  log_base 3 (Real.sqrt 27) + lg 25 + lg 4 + 7 ^ (log_base 7 2) + (-9.8) ^ 0 = 13 / 2 :=
by
  sorry

end log_expression_value_l469_469769


namespace least_common_multiple_of_first_ten_integers_l469_469381

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469381


namespace length_of_chord_l469_469208

noncomputable def parabola_params := { p := 2 }

structure ParabolaChord where
  x1 x2 : ℝ

noncomputable def midpoint_x_coord (chord : ParabolaChord) : ℝ :=
  (chord.x1 + chord.x2) / 2

noncomputable def chord_length (chord : ParabolaChord) : ℝ :=
  (chord.x1 + chord.x2 + parabola_params.p)

theorem length_of_chord (chord : ParabolaChord) (h : midpoint_x_coord chord = 2) :
  chord_length chord = 6 :=
by
  sorry

end length_of_chord_l469_469208


namespace least_common_multiple_1_to_10_l469_469444

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469444


namespace determine_a_l469_469190

-- Define the conditions of the problem.
def linear_in_x_and_y (a : ℝ) : Prop :=
  (a - 2) * (x : ℝ)^(abs a - 1) + 3 * (y : ℝ) = 1

-- Prove that a = -2 under the condition defined.
theorem determine_a (a : ℝ) (h : ∀ x y : ℝ, linear_in_x_and_y a) : a = -2 :=
sorry

end determine_a_l469_469190


namespace total_canoes_built_by_april_l469_469050

theorem total_canoes_built_by_april
  (initial : ℕ)
  (production_increase : ℕ → ℕ) 
  (total_canoes : ℕ) :
  initial = 5 →
  (∀ n, production_increase n = 3 * n) →
  total_canoes = initial + production_increase initial + production_increase (production_increase initial) + production_increase (production_increase (production_increase initial)) →
  total_canoes = 200 :=
by
  intros h_initial h_production h_total
  sorry

end total_canoes_built_by_april_l469_469050


namespace f_sum_of_squares_l469_469276

noncomputable def f (a x : ℝ) [h : fact (a > 0)] [h' : fact (a ≠ 1)] := real.log x / real.log a

theorem f_sum_of_squares {a : ℝ} {x : fin 2017 → ℝ}
  (ha : 0 < a) (ha' : a ≠ 1)
  (hf : f a (∏ i, x i) = 8) :
  (∑ i, f a ((x i) ^ 2)) = 16 :=
by
  sorry

end f_sum_of_squares_l469_469276


namespace tangent_line_and_intervals_monotonicity_l469_469957

def f (m : ℝ) (x : ℝ) : ℝ := (1 / 3) * x^3 + m * x^2 + 1
def f' (m : ℝ) (x : ℝ) : ℝ := x^2 + 2 * m * x

theorem tangent_line_and_intervals_monotonicity : 
  (∀ (m x : ℝ), f' m (-1) = 3 → m = 1) ∧ 
  (∀ x : ℝ, f 1 x = (1 / 3) * x^3 + x^2 + 1) ∧
  (∀ x : ℝ, f 1 1 = 7 / 3) ∧ 
  (3 * 1 - 3 * (7 / 3) + 4 = 0) ∧
  (∀ x : ℝ, x > 0 ∨ x < -2 → f' 1 x > 0) ∧ 
  (∀ x : ℝ, -2 < x ∧ x < 0 → f' 1 x < 0) :=
by
  sorry

end tangent_line_and_intervals_monotonicity_l469_469957


namespace least_common_multiple_1_to_10_l469_469443

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469443


namespace carnations_count_l469_469028

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end carnations_count_l469_469028


namespace more_permutations_with_T_than_without_l469_469016

-- Definition of property T
def has_property_T (n : ℕ) (p : List ℕ) : Prop :=
  ∃ i : ℕ, i < 2 * n - 1 ∧ (n : ℤ) = (p.get! i - p.get! (i + 1)).natAbs

-- The main theorem to prove
theorem more_permutations_with_T_than_without (n : ℕ) (hn : 0 < n) :
  ∃ (s : Finset (List ℕ)) (ht : Finset (List ℕ)) (hf : Finset (List ℕ)),
    (∀ p ∈ s, Permutation p (List.range (2 * n)) ∧ has_property_T n p) ∧
    (∀ p ∈ ht, Permutation p (List.range (2 * n)) ∧ has_property_T n p) ∧
    (∀ p ∈ hf, Permutation p (List.range (2 * n)) ∧ ¬ has_property_T n p) ∧
    s.card < ht.card + hf.card :=
by
  sorry

end more_permutations_with_T_than_without_l469_469016


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469388

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469388


namespace least_common_multiple_of_first_ten_l469_469599

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469599


namespace area_of_triangle_DMC_l469_469908

theorem area_of_triangle_DMC (ABCD : Type) [square ABCD] (area_ABCD : area ABCD = 16) :
  ∃ D M C : ABCD, M = midpoint D C ∧ right_triangle D M C → area_triangle D M C = 2 :=
by
  sorry

end area_of_triangle_DMC_l469_469908


namespace amber_max_ounces_l469_469042

-- Define the problem parameters:
def cost_candy : ℝ := 1
def ounces_candy : ℝ := 12
def cost_chips : ℝ := 1.4
def ounces_chips : ℝ := 17
def total_money : ℝ := 7

-- Define the number of bags of each item Amber can buy:
noncomputable def bags_candy := (total_money / cost_candy).to_int
noncomputable def bags_chips  := (total_money / cost_chips).to_int

-- Define the total ounces of each item:
noncomputable def total_ounces_candy := bags_candy * ounces_candy
noncomputable def total_ounces_chips := bags_chips * ounces_chips

-- Problem statement asking to prove Amber gets the most ounces by buying chips:
theorem amber_max_ounces : max total_ounces_candy total_ounces_chips = total_ounces_chips :=
by sorry

end amber_max_ounces_l469_469042


namespace lcm_first_ten_l469_469411

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469411


namespace bugs_eaten_total_l469_469915

theorem bugs_eaten_total :
  let gecko := 18
  let lizard := gecko / 2
  let frog := lizard * 3
  let tortoise := gecko - (0.25 * gecko)
  let toad := frog + (0.50 * frog)
  let crocodile := gecko + toad
  let turtle := crocodile / 3
  gecko + lizard + frog + tortoise + toad + crocodile + turtle = 186 :=
begin
  sorry
end

end bugs_eaten_total_l469_469915


namespace least_common_multiple_of_first_ten_positive_integers_l469_469458

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469458


namespace solve_for_y_l469_469803

theorem solve_for_y : (12^3 * 6^2) / 432 = 144 := 
by 
  sorry

end solve_for_y_l469_469803


namespace inequality_proof_l469_469951

theorem inequality_proof (a b c : ℝ) :
  a * b + b * c + c * a + max (|a - b|) (max (|b - c|) (|c - a|)) ≤ 1 + (1 / 3) * (a + b + c) ^ 2 :=
sorry

end inequality_proof_l469_469951


namespace local_minimum_at_2_l469_469164

noncomputable def f (x m : ℝ) : ℝ := x * (x - m)^2

theorem local_minimum_at_2 (m : ℝ) (h : 2 * (2 - m)^2 + 2 * 4 * (2 - m) = 0) : m = 6 :=
by
  sorry

end local_minimum_at_2_l469_469164


namespace alternating_colors_probability_l469_469692

def box_contains_five_white_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 1) = 5
def box_contains_five_black_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 0) = 5
def balls_drawn_one_at_a_time : Prop := true -- This condition is trivially satisfied without more specific constraints

theorem alternating_colors_probability (h1 : box_contains_five_white_balls) (h2 : box_contains_five_black_balls) (h3 : balls_drawn_one_at_a_time) :
  ∃ p : ℚ, p = 1 / 126 :=
sorry

end alternating_colors_probability_l469_469692


namespace solve_for_x_l469_469114

  theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → x = 27 :=
  by
    sorry
  
end solve_for_x_l469_469114


namespace ratio_circle_areas_is_four_l469_469241

-- Definitions and conditions
def AB (x : ℝ) := 5 * x
def CB (x : ℝ) := x
def AC (x : ℝ) := 2 * (CB x)

def radius_AC (x : ℝ) := (AC x) / 2
def radius_CB (x : ℝ) := (CB x) / 2

-- Areas of the circles
def area_circle_AC (x : ℝ) := Real.pi * (radius_AC x)^2
def area_circle_CB (x : ℝ) := Real.pi * (radius_CB x)^2

-- Theorem to prove the ratio is 4
theorem ratio_circle_areas_is_four (x : ℝ) (h₀ : x > 0) : 
  (area_circle_AC x) / (area_circle_CB x) = 4 := by
  sorry

end ratio_circle_areas_is_four_l469_469241


namespace lcm_first_ten_numbers_l469_469593

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469593


namespace least_common_multiple_of_first_ten_l469_469613

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469613


namespace simplify_div_expression_l469_469304

theorem simplify_div_expression (x : ℝ) (h : x = Real.sqrt 3 - 1) :
  (x - 1) / (x^2 + 2 * x + 1) / (1 - 2 / (x + 1)) = Real.sqrt 3 / 3 :=
sorry

end simplify_div_expression_l469_469304


namespace lcm_first_ten_l469_469400

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469400


namespace hydroflow_pumps_water_l469_469314

-- Definitions
def gallons_per_hour : ℕ := 360
def time_in_hours : ℚ := 1 / 2

-- Theorem statement
theorem hydroflow_pumps_water (gallons_per_hour : ℕ) (time_in_hours : ℚ) : gallons_per_hour * time_in_hours = 180 :=
by {
  rw [gallons_per_hour, time_in_hours],
  exact 360 * (1 / 2) = 180
}

end hydroflow_pumps_water_l469_469314


namespace no_integer_solution_l469_469865

theorem no_integer_solution (x y : ℤ) : ¬(x^4 + y^2 = 4 * y + 4) :=
by
  sorry

end no_integer_solution_l469_469865


namespace product_of_all_positive_integral_n_l469_469799

theorem product_of_all_positive_integral_n :
  ∃ n : ℕ, ∃ p : ℕ, prime p ∧ n^2 - 18 * n + 159 = p ∧ ∀ m : ℕ, (m^2 - 18 * m + 159 = p → m = n) 
  → (n^2 - 18 * n + 159 = p → (p = 2 ∧ (n^2 - 18 * n + 157 = 0)) ∧ (n * (18 - n)) = 157) :=
by
  sorry

end product_of_all_positive_integral_n_l469_469799


namespace find_m_value_l469_469817

def vectors_parallel (a1 a2 b1 b2 : ℝ) : Prop := a1 * b2 = a2 * b1

theorem find_m_value (m : ℝ) :
  let a := (6, 3)
  let b := (m, 2)
  vectors_parallel a.1 a.2 b.1 b.2 ↔ m = 4 :=
by
  intro H
  obtain ⟨_, _⟩ := H
  sorry

end find_m_value_l469_469817


namespace calculate_expression_l469_469868

theorem calculate_expression (a b : ℝ) (h₁ : 40^a = 5) (h₂ : 40^b = 8) :
    10 ^ ((1 - a - b) / (2 * (1 - b))) = 1 := 
sorry

end calculate_expression_l469_469868


namespace tangent_line_at_1_range_of_a_mean_value_average_function_l469_469149

open Real

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * log x + x^2 - 4 * x

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := (a - 2) * x

-- Problem 1
theorem tangent_line_at_1 (a : ℝ) (h : a = 1) : ∀ x y, y + 3 = -(x - 1) ↔ x + y + 2 = 0 :=
sorry

-- Problem 2
theorem range_of_a (a : ℝ) 
  (h : ∀ x ∈ set.Icc (1/exp 1) exp 1, f a x ≥ g a x): a ≤ -1 :=
sorry

-- Problem 3
theorem mean_value_average_function (a : ℝ) :
  (∀ x1 x2, 0 < x1 ∧ x1 < x2 →
    ∃ x0, x0 = (x1 + x2) / 2 ∧ 
          (deriv (f a) x0 = (f a x2 - f a x1) / (x2 - x1))
  ) ↔ a = 0 :=
sorry

end tangent_line_at_1_range_of_a_mean_value_average_function_l469_469149


namespace quadratic_real_roots_range_find_k_l469_469163

theorem quadratic_real_roots_range (k : ℝ) (h : ∃ x1 x2 : ℝ, x^2 - 2 * (k - 1) * x + k^2 = 0):
  k ≤ 1/2 :=
  sorry

theorem find_k (k : ℝ) (x1 x2 : ℝ) (h₁ : x^2 - 2 * (k - 1) * x + k^2 = 0)
  (h₂ : x₁ * x₂ + x₁ + x₂ - 1 = 0) (h_range : k ≤ 1/2) :
    k = -3 :=
  sorry

end quadratic_real_roots_range_find_k_l469_469163


namespace quadratic_root_real_coeff_l469_469145

theorem quadratic_root_real_coeff (a : ℝ) : (is_root (λ x : ℂ, x^2 + a * x + 2) (1 - I)) → a = -2 := 
by
  intro h
  sorry

end quadratic_root_real_coeff_l469_469145


namespace conjugate_sq_and_cube_l469_469976

variable {a b : ℝ}

theorem conjugate_sq_and_cube (a b : ℝ) : 
  let z1 := a + b * Complex.i,
      z2 := a - b * Complex.i in
  (z1^2 = (a^2 - b^2) + 2 * a * b * Complex.i) ∧ 
  (z2^2 = (a^2 - b^2) - 2 * a * b * Complex.i) ∧
  (z1^3 = (a^3 - 3 * a * b^2) + 3 * (a^2) * b * Complex.i) ∧ 
  (z2^3 = (a^3 - 3 * a * b^2) - 3 * (a^2) * b * Complex.i) := by
  sorry

end conjugate_sq_and_cube_l469_469976


namespace profitable_allocation_2015_l469_469675

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end profitable_allocation_2015_l469_469675


namespace linear_equation_solution_l469_469198

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l469_469198


namespace rep_votes_for_candidate_A_l469_469896

def V : ℝ := 1 -- Assume total number of registered voters is normalized to 1
def Dem_votes_percent : ℝ := 0.7
def Rep_votes_percent : ℝ := 1 - Dem_votes_percent
def Dem_support_percent : ℝ := 0.8
def Total_support_percent : ℝ := 0.65

theorem rep_votes_for_candidate_A : 
  let Dem_votes := V * Dem_votes_percent;
  let Rep_votes := V * Rep_votes_percent;
  let Dem_support := Dem_votes * Dem_support_percent;
  let required_Rep_support := Total_support_percent * V - Dem_support;
  let R := required_Rep_support / Rep_votes
  in R = 0.3 :=
by 
  sorry

end rep_votes_for_candidate_A_l469_469896


namespace smallest_x_value_l469_469653

theorem smallest_x_value : ∃ x : ℤ, 7 - 6 * |x| < -11 ∧ ∀ y : ℤ, 7 - 6 * |y| < -11 → x ≤ y :=
begin
  use -4,
  split,
  { 
    -- show that 7 - 6 * |-4| < -11
    sorry 
  },
  {
    intro y,
    -- show that for all integers y where 7 - 6 * |y| < -11, -4 <= y
    sorry
  }
end

end smallest_x_value_l469_469653


namespace find_b_value_l469_469203

theorem find_b_value (b : ℕ) 
  (h1 : 5 ^ 5 * b = 3 * 15 ^ 5) 
  (h2 : b = 9 ^ 3) : b = 729 :=
by
  sorry

end find_b_value_l469_469203


namespace least_common_multiple_of_first_ten_positive_integers_l469_469471

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469471


namespace cyclic_quadrilateral_with_diameter_l469_469920

-- Given definitions and conditions
variables (A B C D E G : Point) -- Points of interest
variables (hABC : Angle A B C = 30°) (hBAC : Angle B A C = 15°)
variables (hPerp_CD_AC : ∃ D, Perpendicular C D A C)
variables (hPerpBisector_AB : ∃ E, PerpendicularBisector A B)
variables (hExtend_AB_BC : G = Extend AB (Distance B C))

-- To prove that these points are concyclic and the circle's diameter is √2 * AB
theorem cyclic_quadrilateral_with_diameter :
  Concyclic B G E C ∧ Diameter (Circumcircle B G E C) = (√2) * (Distance A B) :=
sorry

end cyclic_quadrilateral_with_diameter_l469_469920


namespace part1_part2_part3_l469_469977

def f (x : ℝ) : ℝ := 1 / (x + 1)

theorem part1 (h3 : 3 > 0) (h4 : 4 > 0) :
  f 3 + f (1 / 3) = 1 ∧ f 4 + f (1 / 4) = 1 :=
by {
  have h₁ : f 3 + f (1 / 3) = 1, 
  { unfold f, 
    calc (1 / (3 + 1)) + (1 / ( (1 : ℝ) / 3 + 1)) = (1 / 4) + (3 / 4) : by ring
                                                         ... = 1 : by norm_num },
  have h₂ : f 4 + f (1 / 4) = 1,
  { unfold f, 
    calc (1 / (4 + 1)) + (1 / ((1 : ℝ) / 4 + 1)) = (1 / 5) + (4 / 5) : by ring
                                                         ... = 1 : by norm_num },
  exact ⟨h₁, h₂⟩
}

theorem part2 (x : ℝ) (hx : x > 0) : f x + f (1 / x) = 1 :=
by {
  unfold f, 
  calc (1 / (x + 1)) + (1 / ((1 : ℝ) / x + 1)) = (1 / (x + 1)) + (x / (x + 1)) : by ring
                                              ... = (x + 1) / (x + 1) : by ring
                                              ... = 1 : by norm_num
}

theorem part3 :
  f 2023 + f 2022 + f 2021 + ... + f 2 + f 1 + f (1 / 2) + ... + f (1 / 2023) = 2022.5 :=
by {
  sorry
}

end part1_part2_part3_l469_469977


namespace black_and_white_films_l469_469709

theorem black_and_white_films (y x B : ℕ) 
  (h1 : ∀ B, B = 40 * x)
  (h2 : (4 * y : ℚ) / (((y / x : ℚ) * B / 100) + 4 * y) = 10 / 11) :
  B = 40 * x :=
by sorry

end black_and_white_films_l469_469709


namespace find_range_of_a_l469_469168

theorem find_range_of_a :
  (∀ x ∈ [0, 2], ∀ y ∈ [0, 2], abs ((x ^ 2 - 2 * a * x + 4) - (y ^ 2 - 2 * a * y + 4)) < 4) →
  (0 < a ∧ a < 2) := 
sorry

end find_range_of_a_l469_469168


namespace least_common_multiple_first_ten_l469_469546

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469546


namespace sum_of_perimeters_correct_l469_469344

noncomputable def sum_of_perimeters (s w : ℝ) : ℝ :=
  let l := 2 * w
  let square_area := s^2
  let rectangle_area := l * w
  let sq_perimeter := 4 * s
  let rect_perimeter := 2 * l + 2 * w
  sq_perimeter + rect_perimeter

theorem sum_of_perimeters_correct (s w : ℝ) (h1 : s^2 + 2 * w^2 = 130) (h2 : s^2 - 2 * w^2 = 50) :
  sum_of_perimeters s w = 12 * Real.sqrt 10 + 12 * Real.sqrt 5 :=
by sorry

end sum_of_perimeters_correct_l469_469344


namespace least_divisible_1_to_10_l469_469535

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469535


namespace kevin_total_cost_l469_469022

noncomputable def total_cost_including_tax : ℕ :=
  let tshirt_cost := 6 * 8 in
  let sweater_cost := 4 * 18 in
  let jacket_cost_before_discount := 5 * 80 in
  let discount := (10 * jacket_cost_before_discount) / 100 in
  let jacket_cost_after_discount := jacket_cost_before_discount - discount in
  let total_cost_after_discount := tshirt_cost + sweater_cost + jacket_cost_after_discount in
  let sales_tax := (5 * total_cost_after_discount) / 100 in
  total_cost_after_discount + sales_tax

theorem kevin_total_cost : total_cost_including_tax = 504 := sorry

end kevin_total_cost_l469_469022


namespace ab_zero_if_conditions_l469_469678

theorem ab_zero_if_conditions 
  (a b : ℤ)
  (h : |a - b| + |a * b| = 2) : a * b = 0 :=
  sorry

end ab_zero_if_conditions_l469_469678


namespace least_divisible_1_to_10_l469_469534

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469534


namespace soccer_team_lineups_l469_469023

theorem soccer_team_lineups 
  (team : Finset ℕ) (quadruplets : Finset ℕ) 
  (h_team_size : team.card = 18) 
  (h_quadruplets_size : quadruplets.card = 4) 
  (h_quadruplets_subset : quadruplets ⊆ team) : 
  let valid_lineups := (team.card.choose 8) - (team.erase_fin 4).card.choose 4 in
  valid_lineups = 42757 := 
by 
  sorry

end soccer_team_lineups_l469_469023


namespace least_common_multiple_of_first_ten_l469_469605

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469605


namespace least_common_multiple_of_first_10_integers_l469_469518

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469518


namespace ordered_pairs_count_l469_469942

theorem ordered_pairs_count :
  let univ_set := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}
  let pairs_count := (cardinal.mk {p : (finset ℕ) × (finset ℕ) // 
    p.1 ∪ p.2 = univ_set ∧ 
    p.1 ∩ p.2 = ∅ ∧ 
    p.1.card ∉ p.1 ∧ 
    p.2.card ∉ p.2}).to_nat
  pairs_count = 3172 := by sorry

end ordered_pairs_count_l469_469942


namespace winning_probabilities_l469_469358

theorem winning_probabilities (p : ℚ) (h : p = 1/2) :
  let P_4_2 := (nat.choose 4 2) * (p^2 * (1-p)^2),
      P_6_3 := (nat.choose 6 3) * (p^3 * (1-p)^3) in
  P_4_2 > P_6_3 :=
by
  sorry

end winning_probabilities_l469_469358


namespace find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469104

theorem find_x_such_that_sqrt_5x_plus_9_eq_12 : ∀ x : ℝ, sqrt (5 * x + 9) = 12 → x = 27 := 
by
  intro x
  sorry

end find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469104


namespace lcm_first_ten_positive_integers_l469_469431

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469431


namespace lcm_first_ten_integers_l469_469633

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469633


namespace students_in_school_l469_469916

variable (S : ℝ)
variable (W : ℝ)
variable (L : ℝ)

theorem students_in_school {S W L : ℝ} 
  (h1 : W = 0.55 * 0.25 * S)
  (h2 : L = 0.45 * 0.25 * S)
  (h3 : W = L + 50) : 
  S = 2000 := 
sorry

end students_in_school_l469_469916


namespace least_common_multiple_1_to_10_l469_469448

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469448


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469394

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469394


namespace asymptotes_eq_3_l469_469173

-- Define the hyperbola and its properties
section
variable {a b : ℝ} (h_a : a > 0) (h_b : b > 0) 

-- Define the hyperbola equation and eccentricity
def hyperbola_eq := ∀ x y : ℝ, (x^2 / a^2 - y^2 / b^2 = 1)
def eccentricity := 2

theorem asymptotes_eq_3 {x y : ℝ} (h : hyperbola_eq a b x y) :
  eccentricity = 2 → (y = sqrt(3) * x ∨ y = - sqrt(3) * x) :=
by
  sorry

end asymptotes_eq_3_l469_469173


namespace amber_max_ounces_l469_469039

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l469_469039


namespace maximum_value_xy_xz_yz_l469_469272

noncomputable def max_value_of_expression : ℝ :=
  let max_val := 164.57 in
  max_val

theorem maximum_value_xy_xz_yz (x y z : ℝ) (h : 2 * x + 3 * y + z = 6) :
  xy + xz + yz ≤ max_value_of_expression :=
sorry

end maximum_value_xy_xz_yz_l469_469272


namespace arc_length_of_sector_l469_469209

theorem arc_length_of_sector (θ : ℝ) (r : ℝ) (L : ℝ) (hθ : θ = 40) (hr : r = 18) :
  L = (θ / 360) * 2 * real.pi * r → L = 4 * real.pi :=
by
  intros h
  rw [hθ, hr] at h
  norm_num at h
  exact h

end arc_length_of_sector_l469_469209


namespace least_divisible_1_to_10_l469_469540

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469540


namespace least_common_multiple_of_first_10_integers_l469_469513

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469513


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469396

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469396


namespace songs_listened_l469_469663

theorem songs_listened (x y : ℕ) 
  (h1 : y = 9) 
  (h2 : y = 2 * (Nat.sqrt x) - 5) 
  : y + x = 58 := 
  sorry

end songs_listened_l469_469663


namespace perfect_square_division_l469_469271

theorem perfect_square_division (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_div : (ab + 1) ∣ (a^2 + b^2)) :
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
begin
  sorry
end

end perfect_square_division_l469_469271


namespace color_elements_l469_469264

open Set

variables {A : Type} [Fintype A]

theorem color_elements (A : Finset A) (hA : A.card = 2017)
  (k : ℕ) (A_i : Fin k → Finset A)
  (h_union : (Finset.univ.bUnion A_i) = A)
  (h_inter : ∀ i j, i ≠ j → (A_i i ∩ A_i j).card ≤ 1) :
  ∃ (red blue : Finset A), (A.filter (λ x, x ∈ blue)).card ≥ 64 ∧ ∀ i, (A_i i ∩ red).nonempty :=
sorry

end color_elements_l469_469264


namespace unit_circle_rotation_l469_469235

def sin_pi_over_3 : ℝ := Real.sin (π / 3)
def cos_pi_over_3 : ℝ := Real.cos (π / 3)

theorem unit_circle_rotation
  (P : ℝ × ℝ)
  (hP : P = (1 / 2, sqrt 3 / 2))
  (α : ℝ)
  (hcos : Real.cos α = 1 / 2)
  (hsin : Real.sin α = sqrt 3 / 2) :
  ∃ Q : ℝ × ℝ, Q = (-1 / 2, sqrt 3 / 2) :=
by
  use (-1 / 2, sqrt 3 / 2)
  sorry

end unit_circle_rotation_l469_469235


namespace least_common_multiple_1_to_10_l469_469499

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469499


namespace isosceles_triangle_of_vector_condition_l469_469189

variables {V : Type} [inner_product_space ℝ V]

def shape_of_triangle (O A B C : V) (h : (B - C) ∙ (B + C - 2 * O) = 0) : Prop :=
  ∥A - B∥ = ∥A - C∥

theorem isosceles_triangle_of_vector_condition (O A B C : V)
  (h : (B - C) ∙ (B + C - 2 * O) = 0) : shape_of_triangle O A B C h :=
sorry

end isosceles_triangle_of_vector_condition_l469_469189


namespace polar_coordinates_intersection_l469_469681

noncomputable def intersection_point (theta : ℝ) (rho : ℝ) : Prop :=
  (ρ = 2 * Real.sin θ) ∧ (ρ = 2 * Real.cos θ) ∧ (ρ > 0) ∧ (0 ≤ θ) ∧ (θ < Real.pi / 2)

theorem polar_coordinates_intersection :
  intersection_point (Real.pi / 4) (Real.sqrt 2) :=
by
  sorry

end polar_coordinates_intersection_l469_469681


namespace lcm_first_ten_positive_integers_l469_469422

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469422


namespace least_common_multiple_first_ten_l469_469638

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469638


namespace polynomial_factorization_l469_469213

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end polynomial_factorization_l469_469213


namespace smallest_n_for_gn_gt_20_l469_469202

def g (n : ℕ) : ℕ := sorry -- definition of the sum of the digits to the right of the decimal of 1 / 3^n

theorem smallest_n_for_gn_gt_20 : ∃ n : ℕ, n > 0 ∧ g n > 20 ∧ ∀ m, 0 < m ∧ m < n -> g m ≤ 20 :=
by
  -- here should be the proof
  sorry

end smallest_n_for_gn_gt_20_l469_469202


namespace linear_equation_a_neg2_l469_469195

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l469_469195


namespace simplify_fraction_l469_469309

variable (x : ℝ)

theorem simplify_fraction : (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l469_469309


namespace lcm_first_ten_numbers_l469_469591

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469591


namespace parallelogram_area_l469_469787

noncomputable def area_of_parallelogram (BD h_a h_b : ℝ) : ℝ :=
  let sin_beta := h_b / BD in
  let sin_gamma := h_a / BD in
  let cos_beta := Real.sqrt (1 - sin_beta^2) in
  let cos_gamma := Real.sqrt (1 - sin_gamma^2) in
  let sin_alpha := sin_beta * cos_gamma + sin_gamma * cos_beta in
  (h_a * h_b) / sin_alpha

theorem parallelogram_area (BD : ℝ) (h_a h_b : ℝ) (h_BD : BD = 5) (h_ha : h_a = 2) (h_hb : h_b = 3) :
  area_of_parallelogram BD h_a h_b = 150 / (3 * Real.sqrt 21 + 8) :=
by
  rw [h_BD, h_ha, h_hb]
  have sin_beta := 3 / 5
  have sin_gamma := 2 / 5
  have cos_beta := Real.sqrt (1 - sin_beta^2)
  have cos_gamma := Real.sqrt (1 - sin_gamma^2)
  have sin_alpha := sin_beta * cos_gamma + sin_gamma * cos_beta
  have actual_area := (2 * 3) / sin_alpha
  exact sorry

end parallelogram_area_l469_469787


namespace K_set_I_K_set_III_K_set_IV_K_set_V_l469_469054

-- Definitions for the problem conditions
def K (x y z : ℤ) : ℤ :=
  (x + 2 * y + 3 * z) * (2 * x - y - z) * (y + 2 * z + 3 * x) +
  (y + 2 * z + 3 * x) * (2 * y - z - x) * (z + 2 * x + 3 * y) +
  (z + 2 * x + 3 * y) * (2 * z - x - y) * (x + 2 * y + 3 * z)

-- The equivalent form as a product of terms
def K_equiv (x y z : ℤ) : ℤ :=
  (y + z - 2 * x) * (z + x - 2 * y) * (x + y - 2 * z)

-- Proof statements for each set of numbers
theorem K_set_I : K 1 4 9 = K_equiv 1 4 9 := by
  sorry

theorem K_set_III : K 4 9 1 = K_equiv 4 9 1 := by
  sorry

theorem K_set_IV : K 1 8 11 = K_equiv 1 8 11 := by
  sorry

theorem K_set_V : K 5 8 (-2) = K_equiv 5 8 (-2) := by
  sorry

end K_set_I_K_set_III_K_set_IV_K_set_V_l469_469054


namespace arith_sequence_S4_over_S12_l469_469139

-- Given an arithmetic sequence with sums S_n, and S_8 = -3 * S_4 ≠ 0
def S (n : ℕ) : ℤ := sorry

axiom (arith_sequence_sum : ∃ (S : ℕ → ℤ), S 8 = -3 * S 4 ∧ S 4 ≠ 0)

theorem arith_sequence_S4_over_S12 : ∀ (S : ℕ → ℤ), (S 8 = -3 * S 4 ∧ S 4 ≠ 0) → (S 4 : ℚ) / (S 12 : ℚ) = -1 / 12 := by
  intro S h
  sorry

end arith_sequence_S4_over_S12_l469_469139


namespace angles_equilateral_triangles_l469_469236

-- Definitions
def is_equilateral_triangle (T : Triangle) := 
  T.angle_A = 60 ∧ T.angle_B = 60 ∧ T.angle_C = 60

variables {A B C D E F : Point}
variables {T₁ T₂ : Triangle}

-- Given conditions
variables (h₁ : is_equilateral_triangle T₁)
variables (h₂ : is_equilateral_triangle T₂)
variables (a b c : ℝ)

-- Points A, B, C form triangle T₁ and points D, E, F form triangle T₂ 
-- within T₁ which is specified in the conditions
variables (triangle_ABC : Triangle A B C = T₁)
variables (triangle_DEF : Triangle D E F = T₂)
variables (angle_BFD := ∠B F D = a)
variables (angle_ADE := ∠A D E = b)
variables (angle_FEC := ∠F E C = c)

-- Proof
theorem angles_equilateral_triangles (h₁ : is_equilateral_triangle T₁) 
                                      (h₂ : is_equilateral_triangle T₂)
                                      (angle_BFD : ∠ B F D = a)
                                      (angle_ADE : ∠ A D E = b)
                                      (angle_FEC : ∠ F E C = c) : 
                                      a = 60 ∧ b = 60 ∧ c = 60 := 
by
  sorry

end angles_equilateral_triangles_l469_469236


namespace simplify_expression_l469_469879

theorem simplify_expression (a b c : ℝ) 
  (h1 : |a| + a = 0) 
  (h2 : |ab| = ab) 
  (h3 : |c| - c = 0) : 
  |b| - |a + b| - |c - b| + |a - c| = b :=
by
  sorry

end simplify_expression_l469_469879


namespace lcm_first_ten_integers_l469_469630

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469630


namespace jerry_paid_more_l469_469726

theorem jerry_paid_more (pizza_cost : ℝ) (cost_per_slice : ℝ) (jerry_slices : ℕ) (tom_slices : ℕ) :
  pizza_cost = 18 →
  cost_per_slice = 1.5 →
  jerry_slices = 10 →
  tom_slices = 2 →
  jerry_slices * cost_per_slice - tom_slices * cost_per_slice = 12 :=
by
  intro h1 h2 h3 h4,
  rw [h1, h2, h3, h4],
  norm_num,
  sorry

end jerry_paid_more_l469_469726


namespace amber_max_ounces_l469_469041

-- Define the problem parameters:
def cost_candy : ℝ := 1
def ounces_candy : ℝ := 12
def cost_chips : ℝ := 1.4
def ounces_chips : ℝ := 17
def total_money : ℝ := 7

-- Define the number of bags of each item Amber can buy:
noncomputable def bags_candy := (total_money / cost_candy).to_int
noncomputable def bags_chips  := (total_money / cost_chips).to_int

-- Define the total ounces of each item:
noncomputable def total_ounces_candy := bags_candy * ounces_candy
noncomputable def total_ounces_chips := bags_chips * ounces_chips

-- Problem statement asking to prove Amber gets the most ounces by buying chips:
theorem amber_max_ounces : max total_ounces_candy total_ounces_chips = total_ounces_chips :=
by sorry

end amber_max_ounces_l469_469041


namespace sum_of_sequence_l469_469327

theorem sum_of_sequence (n : ℕ) :
  (\sum k in Finset.range n, k^2 + 2^k) = (n * (n + 1) * (2 * n + 1)) / 6 + 2^(n + 1) - 2 := by
  sorry

end sum_of_sequence_l469_469327


namespace mrs_lopez_ticket_cost_l469_469284

def ticket_cost : Nat → Nat 
  | 0 => 11  -- Adult (Mrs. Lopez)
  | 1 => 11  -- Adult (husband)
  | 2 => 9   -- Senior (parent 1)
  | 3 => 9   -- Senior (parent 2)
  | 4 => 8   -- Child (age 7)
  | 5 => 8   -- Child (age 10)
  | 6 => 11  -- Child (age 14)
  | _ => 0   -- This default case won't be reached

def total_cost := (List.range 7).sum ticket_cost

theorem mrs_lopez_ticket_cost :
  total_cost = 67 :=
by
  -- Steps to prove the total cost equal to 67 would go here
  sorry

end mrs_lopez_ticket_cost_l469_469284


namespace perpendicular_AD_BC_l469_469254

open Real

variables {A B C D E F P Q R M N : Point}
-- Assume basic geometric objects and their relationships
variables (triangle : Triangle A B C)
variables (on_bc : D ∈ Segment B C)
variables (on_ca : E ∈ Segment C A)
variables (on_ab : F ∈ Segment A B)
variables (concurrence : Concurrent (Line AD) (Line BE) (Line CF) P)

-- Line passing through A and its intersections with DE and DF
variables (line_through_A : ∃ l, Line l ∧ A ∈ l)
variables (intersect_DE : Q ∈ Ray DE ∧ Q ∈ Line_line_through_A)
variables (intersect_DF : R ∈ Ray DF ∧ R ∈ Line_line_through_A)

-- Points on rays DB and DC
variables (M_on_DB : M ∈ Ray DB)
variables (N_on_DC : N ∈ Ray DC)

-- Given geometric condition
variables (geo_condition : (QN^2 / DN) + (RM^2 / DM) = ((DQ + DR)^2 - 2 * RQ^2 + 2 * DM * DN) / MN)

-- Desired conclusion
theorem perpendicular_AD_BC : Perpendicular (Line AD) (Line BC) :=
sorry

end perpendicular_AD_BC_l469_469254


namespace change_in_responses_max_min_diff_l469_469754

open Classical

theorem change_in_responses_max_min_diff :
  let initial_yes := 40
  let initial_no := 40
  let initial_undecided := 20
  let end_yes := 60
  let end_no := 30
  let end_undecided := 10
  let min_change := 20
  let max_change := 80
  max_change - min_change = 60 := by
  intros; sorry

end change_in_responses_max_min_diff_l469_469754


namespace lcm_first_ten_positive_integers_l469_469425

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469425


namespace least_common_multiple_first_ten_l469_469639

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469639


namespace least_common_multiple_first_ten_integers_l469_469563

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469563


namespace complex_triple_sum_eq_sqrt3_l469_469260

noncomputable section

open Complex

theorem complex_triple_sum_eq_sqrt3 {a b c : ℂ} (h1 : abs a = 1) (h2 : abs b = 1) (h3 : abs c = 1)
  (h4 : a + b + c ≠ 0) (h5 : a^2 / (b * c) + b^2 / (a * c) + c^2 / (a * b) = 3) : abs (a + b + c) = Real.sqrt 3 :=
by
  sorry

end complex_triple_sum_eq_sqrt3_l469_469260


namespace lcm_first_ten_positive_integers_l469_469421

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469421


namespace soldiers_in_groups_l469_469672

theorem soldiers_in_groups (x : ℕ) (h1 : x % 2 = 1) (h2 : x % 3 = 2) (h3 : x % 5 = 3) : x % 30 = 23 :=
by
  sorry

end soldiers_in_groups_l469_469672


namespace seashells_given_to_brothers_l469_469737

theorem seashells_given_to_brothers :
  ∃ B : ℕ, 180 - 40 - B = 2 * 55 ∧ B = 30 := by
  sorry

end seashells_given_to_brothers_l469_469737


namespace perpendicular_line_equation_l469_469326

theorem perpendicular_line_equation (x y : ℝ) (h : 2 * x + y + 3 = 0) (hx : ∃ c : ℝ, x - 2 * y + c = 0) :
  (c = 7 ↔ ∀ p : ℝ × ℝ, p = (-1, 3) → (p.1 - 2 * p.2 + 7 = 0)) :=
sorry

end perpendicular_line_equation_l469_469326


namespace find_alpha_beta_l469_469176

noncomputable def a : ℕ → ℕ
| n := let k := (√(8 * n + 1) - 1) / 2 in
       if n < (k * (k + 1)) / 2 then k else k + 1

noncomputable def S : ℕ → ℝ
| n := (∑ i in range (n + 1), a i : ℝ)

theorem find_alpha_beta :
  ∃ (α β : ℝ), (0 < α ∧ 0 < β) ∧ α = 3 / 2 ∧ β = (sqrt 2) / 3 ∧
  Tendsto (λ n, S n / (n : ℝ) ^ α) atTop (𝓝 β) := 
sorry

end find_alpha_beta_l469_469176


namespace problem_l469_469950

open Set

def E : Set ℕ := {n | n ≥ 1 ∧ n ≤ 200}
def G : Set ℕ := {a | a ∈ E ∧ ∃ i, 1 ≤ i ∧ i ≤ 100}

theorem problem (G : Set ℕ) (hG1: ∀ i j, 1 ≤ i ∧ i ≤ j ∧ j ≤ 100 → (G i + G j ≠ 201)) 
  (hG2 : ∑ x in G, x = 10080) :
  (∃ k, k % 4 = 0 ∧ ∀ a ∈ G, (a.is_odd ∧ G.count a = k ∧ ∑ x in G, x^2 = 1353400)) :=
sorry

end problem_l469_469950


namespace floor_sqrt_27_square_l469_469074

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end floor_sqrt_27_square_l469_469074


namespace math_pattern_l469_469285

theorem math_pattern (n : ℕ) : (2 * n - 1) * (2 * n + 1) = (2 * n) ^ 2 - 1 :=
by
  sorry

end math_pattern_l469_469285


namespace least_common_multiple_first_ten_l469_469550

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469550


namespace train_speed_approx_l469_469024

/-- A train 200 meters long completely crosses a 300 meters long bridge in 45 seconds. -/
def train_length : ℕ := 200 -- Length of train in meters
def bridge_length : ℕ := 300 -- Length of bridge in meters
def crossing_time : ℕ := 45 -- Time in seconds

/-- Total distance the train travels while crossing the bridge -/
def total_distance : ℕ := train_length + bridge_length -- Total Distance

/-- Speed of the train in meters per second -/
noncomputable def train_speed : ℚ := total_distance / crossing_time

/-- Prove that the speed of the train is approximately 11.11 meters per second -/
theorem train_speed_approx : train_speed ≈ 11.11 := by
  sorry

end train_speed_approx_l469_469024


namespace PaulineDressCost_l469_469293

-- Lets define the variables for each dress cost
variable (P Jean Ida Patty : ℝ)

-- Condition statements
def condition1 : Prop := Patty = Ida + 10
def condition2 : Prop := Ida = Jean + 30
def condition3 : Prop := Jean = P - 10
def condition4 : Prop := P + Jean + Ida + Patty = 160

-- The proof problem statement
theorem PaulineDressCost : 
  condition1 Patty Ida →
  condition2 Ida Jean →
  condition3 Jean P →
  condition4 P Jean Ida Patty →
  P = 30 := by
  sorry

end PaulineDressCost_l469_469293


namespace sin_cos_identity_tan_identity_l469_469125

open Real

namespace Trigonometry

variable (α : ℝ)

-- Given conditions
def given_conditions := (sin α + cos α = (1/5)) ∧ (0 < α) ∧ (α < π)

-- Prove that sin(α) * cos(α) = -12/25
theorem sin_cos_identity (h : given_conditions α) : sin α * cos α = -12/25 := 
sorry

-- Prove that tan(α) = -4/3
theorem tan_identity (h : given_conditions α) : tan α = -4/3 :=
sorry

end Trigonometry

end sin_cos_identity_tan_identity_l469_469125


namespace least_common_multiple_of_first_ten_integers_l469_469376

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469376


namespace polly_cooking_time_l469_469972

theorem polly_cooking_time  :
  (∀ d : ℕ, d < 7 → 20 * d + 5 * d + if d < 4 then 10 * d else 90 = 305) →
  ∀ k, k = 3 → k = 90 :=
begin
  sorry
end

end polly_cooking_time_l469_469972


namespace find_ratio_AS_AB_find_volume_cone_l469_469291

-- Define the points and conditions from the problem
variables (S A B C D L : Point)

-- Definition of the pyramid and the point L on SC
def is_regular_quadrilateral_pyramid (S A B C D : Point) : Prop := sorry
def point_on_segment (L S C : Point) (ratio : ℝ) : Prop := sorry

-- Ratio condition for point L on edge SC
axiom ratio_CL_LS : point_on_segment L S C (1/5)

-- Define additional conditions
axiom pyr_SABCD : is_regular_quadrilateral_pyramid S A B C D
axiom height_SABCD_is_6 : height S A B C D = 6

-- Define theorems to find the ratio AS: AB and the volume of the cone
theorem find_ratio_AS_AB : AS / AB = sqrt(5) / 2 := sorry

theorem find_volume_cone : volume (cone_with_apex_and_base L (circle_containing_vertices B D S)) = 125 * pi * sqrt(2) / (3 * sqrt(3)) := sorry

end find_ratio_AS_AB_find_volume_cone_l469_469291


namespace billy_books_page_count_l469_469049

def numberOfPagesInBook (hours_per_day : ℕ) (days : ℕ) (video_game_percentage : ℚ) (reading_speed : ℕ) (num_books : ℕ) : ℕ :=
  let total_free_time := hours_per_day * days
  let reading_percentage := 1 - video_game_percentage
  let reading_time := (total_free_time * reading_percentage).to_nat
  let total_pages := reading_time * reading_speed
  total_pages / num_books

theorem billy_books_page_count :
  numberOfPagesInBook 8 2 0.75 60 3 = 80 :=
by
  sorry

end billy_books_page_count_l469_469049


namespace least_common_multiple_1_to_10_l469_469445

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469445


namespace ellipse_circle_parallelogram_condition_l469_469822

theorem ellipse_circle_parallelogram_condition
  (a b : ℝ)
  (C₀ : ∀ x y : ℝ, x^2 + y^2 = 1)
  (C₁ : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h : a > 0 ∧ b > 0 ∧ a > b) :
  1 / a^2 + 1 / b^2 = 1 := by
  sorry

end ellipse_circle_parallelogram_condition_l469_469822


namespace combined_sum_correct_l469_469760

-- Define the sum of integers in a range
def sum_of_integers (a b : Int) : Int := (b - a + 1) * (a + b) / 2

-- Define the sum of squares of integers in a range
def sum_of_squares (a b : Int) : Int :=
  let sum_sq (n : Int) : Int := n * (n + 1) * (2 * n + 1) / 6
  sum_sq b - sum_sq (a - 1)

-- Define the combined sum function
def combined_sum (a b c d : Int) : Int :=
  sum_of_integers a b + sum_of_squares c d

-- Theorem statement: Prove the combined sum of integers from -50 to 40 and squares of integers from 10 to 40 is 21220
theorem combined_sum_correct :
  combined_sum (-50) 40 10 40 = 21220 :=
by
  -- leaving the proof as a sorry
  sorry

end combined_sum_correct_l469_469760


namespace draw_parallel_line_through_point_l469_469180

open EuclideanGeometry

-- Defining the problem setting in Lean:

theorem draw_parallel_line_through_point (l₁ l₂ : Line) (P : Point)
  (hl : Parallel l₁ l₂) (hP : ¬ (OnLine P l₁) ∧ ¬ (OnLine P l₂)) :
  ∃ Q, Parallel (LineThrough P Q) l₁ ∧ Parallel (LineThrough P Q) l₂ :=
by
  sorry

end draw_parallel_line_through_point_l469_469180


namespace least_common_multiple_first_ten_integers_l469_469577

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469577


namespace jones_trip_time_comparison_l469_469933

theorem jones_trip_time_comparison (v : ℝ) (h₀ : v > 0) :
    let t1 := 50 / v
    let t2 := 300 / (3 * v)
    t2 = 2 * t1 := by
begin
  let t1 := 50 / v,
  let t2 := 300 / (3 * v),
  have h_t2 : t2 = 100 / v := by
  calc
    t2 = 300 / (3 * v) : by sorry
    ... = 100 / v : by sorry,
  have h : t2 = 2 * t1 := by
  calc
    t2 = 100 / v : by sorry
    ... = 2 * (50 / v) : by sorry,
  exact h,
  sorry,
end

end jones_trip_time_comparison_l469_469933


namespace log_div_log_inv_simplifies_to_neg_one_l469_469656

theorem log_div_log_inv_simplifies_to_neg_one : 
  log 16 / log (1/16) = -1 :=
by 
  sorry

end log_div_log_inv_simplifies_to_neg_one_l469_469656


namespace abs_diff_condition_l469_469315

theorem abs_diff_condition (x y : ℝ) : (8 - 3 : ℝ).abs - (x - y).abs = 3 → (x - y).abs = 2 :=
by
  intro h
  sorry

end abs_diff_condition_l469_469315


namespace amber_max_ounces_l469_469038

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l469_469038


namespace num_dates_with_digit_sum_n_l469_469268

def digit_sum (d : ℕ) : ℕ :=
  (d / 1000) + ((d % 1000) / 100) + ((d % 100) / 10) + (d % 10)

def valid_date (d : ℕ) : Prop :=
  let mm := d / 100
      dd := d % 100
  mm >= 1 ∧ mm <= 12 ∧ dd >= 1 ∧ dd <= if mm = 2 then 28 else 30

theorem num_dates_with_digit_sum_n (n : ℕ) : n = 15 :=
  let dates := (List.range' 101 9325).filter (λ d => valid_date d ∧ digit_sum d = n)
  dates.length = 1131 := sorry

end num_dates_with_digit_sum_n_l469_469268


namespace hyperbola_is_solution_line_m_is_solution_l469_469853

-- Step 1: Define the hyperbola and the conditions
noncomputable def hyperbola_equation (a b : ℝ) : Prop := 
  a > 0 ∧ b > 0 ∧ (∃ e : ℝ, 
    e = 2 * Real.sqrt 3 / 3 ∧ 
    (∀ x y : ℝ, (x^2 / a^2) - (y^2 / b^2) = 1))

-- Hyperbola solution proof
theorem hyperbola_is_solution (a b : ℝ) : 
  hyperbola_equation a b → a = Real.sqrt 3 ∧ b = 1 := 
by
  sorry

-- Step 2: Define the line m and the conditions for intersection and dot product
noncomputable def line_intersection (a b : ℝ) (k : ℝ) : Prop :=
  let line_eqn := y = k * x - 1
  in ∃ x₁ y₁ x₂ y₂ : ℝ, 
    (y₁ = k * x₁ - 1) ∧ (y₂ = k * x₂ - 1) ∧ 
    (x₁^2 / a^2 - y₁^2 / b^2 = 1) ∧ (x₂^2 / a^2 - y₂^2 / b^2 = 1) ∧ 
    (x₁ * x₂ + y₁ * y₂ = -23)

-- Line m solution proof
theorem line_m_is_solution (k : ℝ) : 
  line_intersection (Real.sqrt 3) 1 k → 
  k = 1 / 2 ∨ k = -1 / 2 := 
by
  sorry

end hyperbola_is_solution_line_m_is_solution_l469_469853


namespace maximize_profit_l469_469227

noncomputable def production (m k : ℝ) : ℝ := 3 - k / (m + 1)

def profit (m : ℝ) : ℝ := 28 - m - 16 / (m + 1)

lemma k_value (m : ℝ) (h₁ : m = 0) (h₂ : production m k = 1) : k = 2 :=
by {
  simp [production, h₁] at h₂,
  calc
  1 = 3 - k : by simpa using h₂
  ... = k = 2 : by linarith 
}

lemma profit_maximizer : profit 3 = 21 :=
by {
  calc
  profit 3 = 28 - 3 - 16 / (3 + 1) : by simp [profit]
  ... = 21 : by norm_num
}

theorem maximize_profit : ∃ m, m = 3 ∧ profit m = 21 := 
⟨3, rfl, profit_maximizer⟩

#check k_value
#check profit
#check maximize_profit

end maximize_profit_l469_469227


namespace least_common_multiple_of_first_ten_l469_469610

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469610


namespace find_a_plus_b_l469_469849

noncomputable def polynomial_extremum : Prop :=
  ∃ (a b : ℝ), let f := λ x : ℝ, a * x^3 + 3 * x^2 - 6 * a * x + b in
  (∀ x, (f 2 = 9) ∧ (deriv f 2 = 0)) ∧ (a + b = -13)

theorem find_a_plus_b : polynomial_extremum :=
by
  use [-2, -11]
  simp [polynomial_extremum]
  sorry

end find_a_plus_b_l469_469849


namespace least_common_multiple_of_first_10_integers_l469_469516

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469516


namespace stove_cost_l469_469247

-- Given conditions
def T : ℝ := 1400
def W (S : ℝ) : ℝ := (1 / 6) * S
def total_cost (S : ℝ) : ℝ := S + W S

-- Problem statement
theorem stove_cost (S : ℝ) (h : total_cost S = T) : S = 1200 :=
by 
  sorry

end stove_cost_l469_469247


namespace nested_expression_sum_l469_469768

theorem nested_expression_sum : 
  4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4))))))))))) = 5592404 :=
by 
  sorry

end nested_expression_sum_l469_469768


namespace daryl_age_l469_469807

theorem daryl_age (d j : ℕ) 
  (h1 : d - 4 = 3 * (j - 4)) 
  (h2 : d + 5 = 2 * (j + 5)) :
  d = 31 :=
by sorry

end daryl_age_l469_469807


namespace least_common_multiple_of_first_ten_integers_l469_469368

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469368


namespace no_valid_n_for_conditions_l469_469805

theorem no_valid_n_for_conditions :
  ∀ (n : ℕ), (100 ≤ n / 5 ∧ n / 5 ≤ 999) ∧ (100 ≤ 5 * n ∧ 5 * n ≤ 999) → false :=
by
  sorry

end no_valid_n_for_conditions_l469_469805


namespace tangent_slope_at_1_l469_469340

-- Define the function
def f (x : ℝ) : ℝ := x^3 + (1 / 2) * x^2 - 1

-- Statement of the theorem to prove
theorem tangent_slope_at_1 : (derivative f 1) = 4 :=
by
  sorry

end tangent_slope_at_1_l469_469340


namespace necessary_but_not_sufficient_condition_l469_469126

-- Define the condition p: x^2 - x < 0
def p (x : ℝ) : Prop := x^2 - x < 0

-- Define the necessary but not sufficient condition
def necessary_but_not_sufficient (x : ℝ) : Prop := -1 < x ∧ x < 1

-- State the theorem
theorem necessary_but_not_sufficient_condition :
  ∀ x : ℝ, p x → necessary_but_not_sufficient x :=
sorry

end necessary_but_not_sufficient_condition_l469_469126


namespace lcm_first_ten_numbers_l469_469588

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469588


namespace lcm_first_ten_integers_l469_469628

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469628


namespace least_common_multiple_of_first_ten_positive_integers_l469_469463

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469463


namespace keiko_ephraim_same_heads_l469_469935

theorem keiko_ephraim_same_heads :
  let outcomes := [("HH", "HH"), ("HH", "HT"), ("HH", "TH"), ("HH", "TT"),
                   ("HT", "HH"), ("HT", "HT"), ("HT", "TH"), ("HT", "TT"),
                   ("TH", "HH"), ("TH", "HT"), ("TH", "TH"), ("TH", "TT"),
                   ("TT", "HH"), ("TT", "HT"), ("TT", "TH"), ("TT", "TT")],
      same_heads := λ (x : String × String),
        (x.fst.count (λ c, c = 'H') = x.snd.count (λ c, c = 'H')),
      matching_outcomes := outcomes.filter same_heads
  in (matching_outcomes.length : ℚ) / (outcomes.length : ℚ) = 3 / 8 := by
  sorry

end keiko_ephraim_same_heads_l469_469935


namespace f_neg2_minus_f_neg3_l469_469159

-- Given conditions
variable (f : ℝ → ℝ)
variable (odd_f : ∀ x, f (-x) = - f x)
variable (h : f 3 - f 2 = 1)

-- Goal to prove
theorem f_neg2_minus_f_neg3 : f (-2) - f (-3) = 1 := by
  sorry

end f_neg2_minus_f_neg3_l469_469159


namespace find_five_numbers_with_largest_sum_l469_469858

open Nat

theorem find_five_numbers_with_largest_sum :
  ∃ (a b c d e : ℕ), a + b + c + d + e = a * b * c * d * e ∧ 
  (∀ (a' b' c' d' e' : ℕ), a' + b' + c' + d' + e' = a' * b' * c' * d' * e' → 
  a' + b' + c' + d' + e' ≤ a + b + c + d + e) ∧
  a + b + c + d + e = 10 :=
begin
  sorry
end

end find_five_numbers_with_largest_sum_l469_469858


namespace volume_calculation_l469_469763

open Real

def volume_bounded_by_surfaces : ℝ :=
  ∫ (x : ℝ) in 0..6, ∫ (y : ℝ) in (sqrt x)..(2 * sqrt x), ∫ (z : ℝ) in 0..(6 - x), 1

theorem volume_calculation :
  volume_bounded_by_surfaces = (48 / 5) * sqrt 6 :=
sorry

end volume_calculation_l469_469763


namespace convex_quadrilateral_segments_l469_469860

-- Definitions based on the given conditions
def polygon (n : ℕ) : Type := {lst : list (ℝ × ℝ) // lst.length = n ∧ n % 2 = 1}

structure conditions (P1 P2 : polygon) : Prop :=
  (distinct_lines : ∀ (s1 ∈ P1.1) (s2 ∈ P2.1), ¬collinear s1 s2)
  (no_three_intersect : ∀ (s1 ∈ P1.1) (s2 ∈ P1.1) (s3 ∈ P2.1), s1 ∉ segment s2 s3)

-- Equivalent proof problem as Lean theorem statement
theorem convex_quadrilateral_segments {P1 P2 : polygon} 
  (conds : conditions P1 P2) : 
  ∃ s₁ ∈ P1.1, ∃ s₂ ∈ P2.1, opposite_sides_of_convex_quadrilateral s₁ s₂ := 
sorry

end convex_quadrilateral_segments_l469_469860


namespace sum_points_on_fff_graph_l469_469172

noncomputable def f : ℕ → ℕ
| 1 := 3
| 2 := 1
| 3 := 5
| _ := 0  -- default case to handle other inputs for completeness, not actually needed

theorem sum_points_on_fff_graph : 
  let a : ℕ := 1,
      b : ℕ := f(f a),
      c : ℕ := 2,
      d : ℕ := f(f c) in
  a * b + c * d = 11 :=
by
  let a := 1
  let b := f(f a)
  let c := 2
  let d := f(f c)
  have h_b : b = 5 := by rfl -- since f(f(1)) = f(3) = 5
  have h_d : d = 3 := by rfl -- since f(f(2)) = f(1) = 3
  calc
    a * b + c * d = 1 * 5 + 2 * 3 : by rw [h_b, h_d]
              ... = 5 + 6       : by rfl
              ... = 11          : by rfl

end sum_points_on_fff_graph_l469_469172


namespace indonesian_mo_2002_p2_l469_469922

theorem indonesian_mo_2002_p2 (p : ℕ) : 
  (∃ k : ℤ, 3 * p + 25 = k * (2 * p - 5)) → 
  (∃ m : ℕ, m ≠ 0 → m = p ∧ m ∈ {3, 5, 9, 35}) :=
by
  sorry

end indonesian_mo_2002_p2_l469_469922


namespace bus_passengers_after_third_stop_l469_469350

theorem bus_passengers_after_third_stop
    (initial : ℕ)
    (off1 : ℕ)
    (off2 : ℕ)
    (on2 : ℕ)
    (off3 : ℕ)
    (on3 : ℕ) :
    initial = 50 →
    off1 = 15 →
    off2 = 8 →
    on2 = 2 →
    off3 = 4 →
    on3 = 3 →
    ((initial - off1 - off2 + on2 - off3 + on3) = 28) :=
by
  intros h_initial h_off1 h_off2 h_on2 h_off3 h_on3
  rw [h_initial, h_off1, h_off2, h_on2, h_off3, h_on3]
  norm_num

end bus_passengers_after_third_stop_l469_469350


namespace least_common_multiple_first_ten_l469_469646

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469646


namespace solve_for_x_l469_469099

theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → (x = 27) := by
  intro h
  sorry

end solve_for_x_l469_469099


namespace bag_with_cracks_number_l469_469984

def marbles : List ℕ := [18, 19, 21, 23, 25, 34]

def total_marbles : ℕ := marbles.sum

def modulo_3 (n : ℕ) : ℕ := n % 3

theorem bag_with_cracks_number :
  ∃ (c : ℕ), c ∈ marbles ∧ 
    (total_marbles - c) % 3 = 0 ∧
    c = 23 :=
by 
  sorry

end bag_with_cracks_number_l469_469984


namespace floor_sqrt_27_square_l469_469073

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end floor_sqrt_27_square_l469_469073


namespace find_n_from_average_l469_469020

theorem find_n_from_average :
  ∀ (n : ℕ), (∑ i in finset.range (n+1), i * i) / (∑ i in finset.range (n+1), i) = 2037 → n = 3055 :=
by
  sorry

end find_n_from_average_l469_469020


namespace G_at_8_eq_144_l469_469941

noncomputable def G : ℝ → ℝ := sorry

theorem G_at_8_eq_144
  (h_poly : ∀ x, polynomial G)
  (h_G4 : G 4 = 24)
  (h_ratio : ∀ x : ℝ, G(x + 2) ≠ 0 → G(2 * x) / G(x + 2) = 4 - (16 * x + 24) / (x ^ 2 + 4 * x + 8)) :
  G 8 = 144 :=
sorry

end G_at_8_eq_144_l469_469941


namespace initial_milk_amount_l469_469755

-- Definitions
def milk_per_milkshake : ℕ := 4
def ice_cream_per_milkshake : ℕ := 12
def total_ice_cream : ℕ := 192
def leftover_milk : ℕ := 8

-- Proof statement
theorem initial_milk_amount
  (milk_per_milkshake : ℕ)
  (ice_cream_per_milkshake : ℕ)
  (total_ice_cream : ℕ)
  (leftover_milk : ℕ) :
  milk_per_milkshake = 4 →
  ice_cream_per_milkshake = 12 →
  total_ice_cream = 192 →
  leftover_milk = 8 →
  initial_milk_amount = 72 :=
by
  sorry

end initial_milk_amount_l469_469755


namespace sequence_k_value_l469_469134

theorem sequence_k_value {k : ℕ} (h : 9 < (2 * k - 8) ∧ (2 * k - 8) < 12) 
  (Sn : ℕ → ℤ) (hSn : ∀ n, Sn n = n^2 - 7*n) 
  : k = 9 :=
by
  sorry

end sequence_k_value_l469_469134


namespace least_common_multiple_first_ten_integers_l469_469569

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469569


namespace prime_square_mod_12_l469_469298

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end prime_square_mod_12_l469_469298


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469382

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469382


namespace part1_part2_l469_469847

def f (x a : ℝ) := x^2 + 4 * a * x + 2 * a + 6

theorem part1 (a : ℝ) : (∃ x : ℝ, f x a = 0) ↔ (a = -1 ∨ a = 3 / 2) := 
by 
  sorry

def g (a : ℝ) := 2 - a * |a + 3|

theorem part2 (a : ℝ) :
  (-1 ≤ a ∧ a ≤ 3 / 2) →
  -19 / 4 ≤ g a ∧ g a ≤ 4 :=
by 
  sorry

end part1_part2_l469_469847


namespace least_common_multiple_of_first_ten_integers_l469_469371

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469371


namespace vol_A_correct_vol_B_correct_l469_469720

-- Conditions
def sheet_length : ℝ := 48
def sheet_width : ℝ := 36

def corner_square_side : ℝ := 8
def middle_square_side : ℝ := 12

-- Intermediate calculations
def new_length : ℝ := sheet_length - 2 * corner_square_side
def new_width : ℝ := sheet_width - 2 * corner_square_side
def height : ℝ := corner_square_side

-- Volumes
def volume_A : ℝ := new_length * new_width * height

def larger_rectangle_area : ℝ := new_length * new_width
def smaller_square_area : ℝ := middle_square_side * middle_square_side
def base_area_B : ℝ := larger_rectangle_area - smaller_square_area

def volume_B : ℝ := base_area_B * height

-- Theorems
theorem vol_A_correct : volume_A = 5120 := by
  sorry

theorem vol_B_correct : volume_B = 3968 := by
  sorry

end vol_A_correct_vol_B_correct_l469_469720


namespace total_amount_paid_l469_469745

def price_grapes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_mangoes (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_pineapple (kg: ℕ) (rate: ℕ) : ℕ := kg * rate
def price_kiwi (kg: ℕ) (rate: ℕ) : ℕ := kg * rate

theorem total_amount_paid :
  price_grapes 14 54 + price_mangoes 10 62 + price_pineapple 8 40 + price_kiwi 5 30 = 1846 :=
by
  sorry

end total_amount_paid_l469_469745


namespace max_sum_distances_on_semicircle_l469_469277

theorem max_sum_distances_on_semicircle (r : ℝ) :
  ∃ M : ℝ × ℝ, ∀ A B : ℝ × ℝ, 
  (A = (-r, 0)) ∧ (B = (r, 0)) → 
  (∃ θ : ℝ, M = (2 * r * cos θ, 2 * r * sin θ)) →
  (2 * r * (sin θ + cos θ) ≤ 2 * sqrt 2 * r) :=
sorry

end max_sum_distances_on_semicircle_l469_469277


namespace trig_identity_l469_469160

theorem trig_identity (α : ℝ) (x y : ℝ) (h_ray : 3 * x - 4 * y = 0) (h_x_neg : x < 0) (h_initial : α = real.atan2 y x) :
  real.sin α - real.cos α = 1 / 5 :=
sorry

end trig_identity_l469_469160


namespace george_team_final_round_average_required_less_than_record_l469_469811

theorem george_team_final_round_average_required_less_than_record :
  ∀ (old_record average_score : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ),
    old_record = 287 →
    players = 4 →
    rounds = 10 →
    current_score = 10440 →
    (old_record - ((rounds * (old_record * players) - current_score) / players)) = 27 :=
by
  -- Given the values and conditions, prove the equality here
  sorry

end george_team_final_round_average_required_less_than_record_l469_469811


namespace odd_n_implies_distinct_remainders_l469_469953

theorem odd_n_implies_distinct_remainders (n : ℕ) (a : Fin n → ℕ) (h_permutation : ∀ i, 1 ≤ a i ∧ a i ≤ n ∧ ∀ j, i ≠ j → a i ≠ a j):
  (∀ i j, i < j → (List.sum (List.map a (FinRange i)) % (n + 1)) ≠ (List.sum (List.map a (FinRange j)) % (n + 1))) →
  Odd n :=
sorry

end odd_n_implies_distinct_remainders_l469_469953


namespace least_common_multiple_first_ten_l469_469647

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469647


namespace least_common_multiple_first_ten_l469_469643

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469643


namespace find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469105

theorem find_x_such_that_sqrt_5x_plus_9_eq_12 : ∀ x : ℝ, sqrt (5 * x + 9) = 12 → x = 27 := 
by
  intro x
  sorry

end find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469105


namespace inverse_sine_function_l469_469776

theorem inverse_sine_function :
  (∀ x : ℝ, -real.pi / 2 ≤ x ∧ x ≤ real.pi / 2 → ∃ y : ℝ, -1 ≤ y ∧ y ≤ 1 ∧ y = real.arcsin x) :=
begin
  sorry
end

end inverse_sine_function_l469_469776


namespace evaluate_expression_zero_l469_469082

-- Main proof statement
theorem evaluate_expression_zero :
  ∀ (a d c b : ℤ),
    d = c + 5 →
    c = b - 8 →
    b = a + 3 →
    a = 3 →
    a - 1 ≠ 0 →
    d - 6 ≠ 0 →
    c + 4 ≠ 0 →
    (a + 3) * (d - 3) * (c + 9) = 0 :=
by
  intros a d c b hd hc hb ha h1 h2 h3
  sorry -- The proof goes here

end evaluate_expression_zero_l469_469082


namespace least_common_multiple_of_first_ten_l469_469612

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469612


namespace price_36kg_apples_l469_469749

-- Definitions based on given conditions
def cost_per_kg_first_30 (l : ℕ) (n₁ : ℕ) (total₁ : ℕ) : Prop :=
  n₁ = 10 ∧ l = total₁ / n₁

def total_cost_33kg (l q : ℕ) (total₂ : ℕ) : Prop :=
  30 * l + 3 * q = total₂

-- Question to prove
def total_cost_36kg (l q : ℕ) (cost_36 : ℕ) : Prop :=
  30 * l + 6 * q = cost_36

theorem price_36kg_apples (l q cost_36 : ℕ) :
  (cost_per_kg_first_30 l 10 200) →
  (total_cost_33kg l q 663) →
  cost_36 = 726 :=
by
  intros h₁ h₂
  sorry

end price_36kg_apples_l469_469749


namespace chocolate_pieces_per_box_l469_469355

theorem chocolate_pieces_per_box (initial_boxes : ℕ) (given_away_boxes : ℕ) (remaining_pieces : ℕ) (remaining_boxes : ℕ) (pieces_per_box : ℕ) :
  initial_boxes = 12 → 
  given_away_boxes = 7 → 
  remaining_pieces = 30 → 
  remaining_boxes = initial_boxes - given_away_boxes →
  pieces_per_box = remaining_pieces / remaining_boxes → 
  pieces_per_box = 6 :=
by
  intros h1 h2 h3 h4 h5
  rw [h1, h2, h3] at h4 h5
  have h6 : remaining_boxes = 5 := by norm_num [h4]
  have h7 : pieces_per_box = 6 := by norm_num [h5, h6]
  exact h7

end chocolate_pieces_per_box_l469_469355


namespace fraction_books_sold_l469_469005

theorem fraction_books_sold :
  (∃ B F : ℝ, 3.50 * (B - 40) = 280.00000000000006 ∧ B ≠ 0 ∧ F = ((B - 40) / B) ∧ B = 120) → (F = 2 / 3) :=
by
  intro h
  obtain ⟨B, F, h1, h2, e⟩ := h
  sorry

end fraction_books_sold_l469_469005


namespace min_avg_less_than_old_record_l469_469813

variable old_record_avg : ℕ := 287
variable num_players : ℕ := 4
variable num_rounds : ℕ := 10
variable points_scored_9_rounds : ℕ := 10440

theorem min_avg_less_than_old_record:
  let total_points_needed := old_record_avg * num_players * num_rounds in
  let points_needed_final_round := total_points_needed - points_scored_9_rounds in
  let min_avg_final_round := points_needed_final_round / num_players in
  min_avg_final_round = old_record_avg - 27 :=
by
  sorry

end min_avg_less_than_old_record_l469_469813


namespace least_common_multiple_first_ten_l469_469544

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469544


namespace exists_line_slope_arithmetic_sequence_l469_469275

theorem exists_line_slope_arithmetic_sequence :
  ∃ l : ℝ → ℝ, (∀ M : ℝ × ℝ, M.1 = l M.2 → 
    ∃ A B : ℝ × ℝ, 
    A ∈ {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1} ∧ B ∈ {p : ℝ × ℝ | p.2 ^ 2 = 4 * p.1} ∧ 
    2 * ((M.2 - 0) / (M.1 - 1)) = ((M.2 - A.2) / (M.1 - A.1)) + ((M.2 - B.2) / (M.1 - B.1)))) ∧
    (∀ x, l x = -1) :=
begin
  sorry
end

end exists_line_slope_arithmetic_sequence_l469_469275


namespace bus_passengers_after_third_stop_l469_469351

theorem bus_passengers_after_third_stop
    (initial : ℕ)
    (off1 : ℕ)
    (off2 : ℕ)
    (on2 : ℕ)
    (off3 : ℕ)
    (on3 : ℕ) :
    initial = 50 →
    off1 = 15 →
    off2 = 8 →
    on2 = 2 →
    off3 = 4 →
    on3 = 3 →
    ((initial - off1 - off2 + on2 - off3 + on3) = 28) :=
by
  intros h_initial h_off1 h_off2 h_on2 h_off3 h_on3
  rw [h_initial, h_off1, h_off2, h_on2, h_off3, h_on3]
  norm_num

end bus_passengers_after_third_stop_l469_469351


namespace least_common_multiple_first_ten_l469_469645

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469645


namespace least_distinct_values_l469_469012

/-- 
Given a list of 2018 positive integers with a unique mode occurring exactly 10 times,
prove that the least number of distinct values in the list is 225.
-/
theorem least_distinct_values {α : Type*} (l : list α) (hl_len : l.length = 2018) (hm : ∃ m, ∀ x ∈ l, count l x ≤ count l m ∧ count l m = 10 ∧ ∀ y ≠ x, count l y < 10) :
  ∃ n, n = 225 ∧ (∀ x ∈ (l.erase_dup), count l x ≤ 10) :=
sorry

end least_distinct_values_l469_469012


namespace find_element_in_A_l469_469156

def A : Type := ℝ × ℝ
def B : Type := ℝ × ℝ

def f (p : A) : B := (p.1 + 2 * p.2, 2 * p.1 - p.2)

theorem find_element_in_A : ∃ p : A, f p = (3, 1) ∧ p = (1, 1) := by
  sorry

end find_element_in_A_l469_469156


namespace lcm_first_ten_positive_integers_l469_469428

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469428


namespace painting_rate_l469_469322

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end painting_rate_l469_469322


namespace intersection_of_A_and_B_l469_469940

def N_star := { x : ℕ // x > 0 }

def A : Set N_star := { x | 2^x.val < 4 }

def B : Set ℝ := { x | -1 < x ∧ x < 2 }

theorem intersection_of_A_and_B : A ∩ { x : N_star | (x : ℝ) ∈ B } = { ⟨1, by decide⟩ } :=
by 
  sorry

end intersection_of_A_and_B_l469_469940


namespace bouquet_cost_l469_469752

theorem bouquet_cost (c : ℕ) : (c / 25 = 30 / 15) → c = 50 := by
  sorry

end bouquet_cost_l469_469752


namespace total_dinners_l469_469288

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end total_dinners_l469_469288


namespace seq_eventually_periodic_l469_469728

-- Definitions from the problem
def floor (x : ℝ) : ℤ := int.floor x
def frac (x : ℝ) : ℝ := x - (floor x : ℝ)
def seq (a₀ : ℝ) : ℕ → ℝ
| 0       := a₀
| (n + 1) := (floor (seq n) : ℝ) * frac (seq n)

-- Theorem statement
theorem seq_eventually_periodic (a₀ : ℝ) :
  ∃ i₀ : ℕ, ∀ i ≥ i₀, seq a₀ i = seq a₀ (i + 2) :=
sorry

end seq_eventually_periodic_l469_469728


namespace jessica_original_watermelons_l469_469249

theorem jessica_original_watermelons (watermelons_left : ℕ) (watermelons_eaten : ℕ) 
    (h1 : watermelons_left = 8) 
    (h2 : watermelons_eaten = 27) : 
    watermelons_left + watermelons_eaten = 35 :=
by 
    rw [h1, h2]
    exact rfl

end jessica_original_watermelons_l469_469249


namespace neg_p_equiv_l469_469836

variable (I : Set ℝ)

def p : Prop := ∀ x ∈ I, x / (x - 1) > 0

theorem neg_p_equiv :
  ¬p I ↔ ∃ x ∈ I, x / (x - 1) ≤ 0 ∨ x - 1 = 0 :=
by
  sorry

end neg_p_equiv_l469_469836


namespace sum_binom_ineq_l469_469301

theorem sum_binom_ineq (j n : ℕ) : 
  ∑ k in Finset.range (n + 1), k ^ j * (Nat.choose n k) ≥ 2 ^ (n - j) * n ^ j :=
sorry

end sum_binom_ineq_l469_469301


namespace least_common_multiple_first_ten_l469_469640

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469640


namespace maximize_x3y4_l469_469263

noncomputable def max_product (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + y = 50) : ℝ :=
  x^3 * y^4

theorem maximize_x3y4 (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = 50) :
  max_product x y hx hy h ≤ max_product (150/7) (200/7) (by norm_num) (by norm_num) (by norm_num) :=
  sorry

end maximize_x3y4_l469_469263


namespace least_common_multiple_1_to_10_l469_469436

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469436


namespace chromium_percentage_l469_469910

theorem chromium_percentage (x : ℝ) : 
  (15 * x / 100 + 35 * 8 / 100 = 50 * 8.6 / 100) → 
  x = 10 := 
sorry

end chromium_percentage_l469_469910


namespace lcm_first_ten_numbers_l469_469581

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469581


namespace least_positive_integer_divisible_by_first_ten_l469_469486

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469486


namespace lcm_first_ten_numbers_l469_469594

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469594


namespace range_of_a_l469_469821

-- Defining the function f(x) = x^2 + 2ax - 1
def f (x a : ℝ) : ℝ := x^2 + 2 * a * x - 1

-- Conditions: x1, x2 ∈ [1, +∞) and x1 < x2
variables (x1 x2 a : ℝ)
variables (h1 : 1 ≤ x1) (h2 : 1 ≤ x2) (h3 : x1 < x2)

-- Statement of the proof problem:
theorem range_of_a (hf_ineq : x2 * f x1 a - x1 * f x2 a < a * (x1 - x2)) : a ≤ 2 :=
sorry

end range_of_a_l469_469821


namespace books_loaned_out_during_month_l469_469015

-- Define the initial conditions
def initial_books : ℕ := 75
def remaining_books : ℕ := 65
def loaned_out_percentage : ℝ := 0.80
def returned_books_ratio : ℝ := loaned_out_percentage
def not_returned_ratio : ℝ := 1 - returned_books_ratio
def difference : ℕ := initial_books - remaining_books

-- Define the main theorem
theorem books_loaned_out_during_month : ∃ (x : ℕ), not_returned_ratio * (x : ℝ) = (difference : ℝ) ∧ x = 50 :=
by
  existsi 50
  simp [not_returned_ratio, difference]
  sorry

end books_loaned_out_during_month_l469_469015


namespace total_dinners_sold_203_l469_469286

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end total_dinners_sold_203_l469_469286


namespace parabola_pqr_sum_l469_469784

-- Define the conditions given in the problem
def is_parabola (p q r : ℚ) : Prop :=
  ∃ (y : ℚ → ℚ), y = λ x, (p * x^2 + q * x + r)

def has_vertex (p q r : ℚ) (vertex : ℚ × ℚ) : Prop :=
  vertex = (3, -1)

def contains_point (p q r : ℚ) (point : ℚ × ℚ) : Prop :=
  let (x, y) := point in y = p * x^2 + q * x + r

-- Statement to prove p + q + r == 11/9 given the conditions
theorem parabola_pqr_sum (p q r : ℚ) 
  (h1 : is_parabola p q r)
  (h2 : has_vertex p q r (3, -1))
  (h3 : contains_point p q r (0, 4)) :
  p + q + r = 11 / 9 := by
  sorry

end parabola_pqr_sum_l469_469784


namespace total_dinners_l469_469289

def monday_dinners := 40
def tuesday_dinners := monday_dinners + 40
def wednesday_dinners := tuesday_dinners / 2
def thursday_dinners := wednesday_dinners + 3

theorem total_dinners : monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners = 203 := by
  sorry

end total_dinners_l469_469289


namespace least_common_multiple_of_first_10_integers_l469_469520

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469520


namespace subs_div_mod_100_l469_469687

def a : ℕ → ℕ
| 0       := 1
| (n + 1) := 5 * (11 - n) * a n

theorem subs_div_mod_100 : (a 0 + a 1 + a 2 + a 3 + a 4) % 100 = 56 :=
by
  have a_0 : a 0 = 1 := rfl
  have a_1 : a 1 = 5 * (11 - 0) * a 0 := rfl
  have a_2 : a 2 = 5 * (11 - 1) * a 1 := rfl
  have a_3 : a 3 = 5 * (11 - 2) * a 2 := rfl
  have a_4 : a 4 = 5 * (11 - 3) * a 3 := rfl
  sorry

end subs_div_mod_100_l469_469687


namespace problem_statement_l469_469261

open Real Polynomial

theorem problem_statement (a1 a2 a3 d1 d2 d3 : ℝ) 
  (h : ∀ x : ℝ, x^8 - x^7 + x^6 - x^5 + x^4 - x^3 + x^2 - x + 1 =
                 (x^2 + a1 * x + d1) * (x^2 + a2 * x + d2) * (x^2 + a3 * x + d3) * (x^2 - 1)) :
  a1 * d1 + a2 * d2 + a3 * d3 = -1 := 
sorry

end problem_statement_l469_469261


namespace lcm_first_ten_integers_l469_469621

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469621


namespace largest_real_part_of_z_add_w_l469_469988

open Complex

noncomputable def problem_conditions (z w u : ℂ) (x y : ℝ) : Prop :=
  abs z = 1 ∧ abs w = 1 ∧ (z * conj w + conj z * w = 2) ∧ abs (z - u) = abs (w - u)

noncomputable def problem_conditions_u (x y : ℝ) : ℂ := x + complex.I * y

theorem largest_real_part_of_z_add_w
  (z w : ℂ)
  (x y : ℝ)
  (u := problem_conditions_u x y)
  (h : problem_conditions z w u x y) :
  real.part (z + w) ≤ 2 :=
by
  sorry

end largest_real_part_of_z_add_w_l469_469988


namespace least_common_multiple_of_first_ten_integers_l469_469370

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469370


namespace least_common_multiple_of_first_ten_l469_469601

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469601


namespace twentieth_term_arithmetic_sequence_eq_neg49_l469_469991

-- Definitions based on the conditions
def a1 : ℤ := 8
def d : ℤ := 5 - 8
def a (n : ℕ) : ℤ := a1 + (n - 1) * d

-- The proof statement
theorem twentieth_term_arithmetic_sequence_eq_neg49 : a 20 = -49 :=
by 
  -- Proof will be inserted here
  sorry

end twentieth_term_arithmetic_sequence_eq_neg49_l469_469991


namespace pond_diameter_l469_469123

theorem pond_diameter 
  (h k r : ℝ)
  (H1 : (4 - h) ^ 2 + (11 - k) ^ 2 = r ^ 2)
  (H2 : (12 - h) ^ 2 + (9 - k) ^ 2 = r ^ 2)
  (H3 : (2 - h) ^ 2 + (7 - k) ^ 2 = (r - 1) ^ 2) :
  2 * r = 9.2 :=
sorry

end pond_diameter_l469_469123


namespace Romeo_bars_of_chocolate_l469_469981

theorem Romeo_bars_of_chocolate 
  (cost_per_bar : ℕ) (packaging_cost : ℕ) (total_sale : ℕ) (profit : ℕ) (x : ℕ) :
  cost_per_bar = 5 →
  packaging_cost = 2 →
  total_sale = 90 →
  profit = 55 →
  (total_sale - (cost_per_bar + packaging_cost) * x = profit) →
  x = 5 :=
by
  sorry

end Romeo_bars_of_chocolate_l469_469981


namespace euler_prime_counting_l469_469973

open Filter
open Topology

noncomputable def prime_counting_function (n : ℕ) : ℕ :=
  (Finset.filter (λ p, p.Prime) (Finset.range n.succ)).card

theorem euler_prime_counting (h : ∀ n : ℕ, prime_counting_function n = π(n)) :
  tendsto (λ n, (prime_counting_function n : ℝ) / n) at_top (𝓝 0) :=
by
  sorry

end euler_prime_counting_l469_469973


namespace mandy_yoga_time_l469_469924

-- Define the conditions
def ratio_swimming := 1
def ratio_running := 2
def ratio_gym := 3
def ratio_biking := 5
def ratio_yoga := 4

def time_biking := 30

-- Define the Lean 4 statement to prove
theorem mandy_yoga_time : (time_biking / ratio_biking) * ratio_yoga = 24 :=
by
  sorry

end mandy_yoga_time_l469_469924


namespace find_p_l469_469958

theorem find_p (A B C p q r s : ℝ) 
  (hroot1 : A * r^2 + B * r + C = 0)
  (hroot2 : A * s^2 + B * s + C = 0)
  (hroot3 : r^2 + s^2 + p = 0)
  (hroot4 : r^2 * s^2 = q) :
  p = (2 * A * C - B^2) / A^2 :=
sorry

end find_p_l469_469958


namespace number_of_solutions_l469_469778

theorem number_of_solutions :
  (∀ (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 → x ≠ 0 ∧ x ≠ 5) →
  ∃! (x : ℝ), (3 * x ^ 3 - 15 * x ^ 2) / (x ^ 2 - 5 * x) = 2 * x - 6 :=
by
  sorry

end number_of_solutions_l469_469778


namespace area_CPQ_l469_469238

variables (A B C P Q : Type)
variables [T : has_area (triangle A B C)] (hABC_area : area (triangle A B C) = 36)
variables (hP_on_AC : P ∈ line_segment A C) (hQ_on_AB : Q ∈ line_segment A B)
variables (hAP_fraction : dist A P = (1/3) * dist A C) (hAQ_fraction : dist A Q = (1/3) * dist A B)

theorem area_CPQ 
  (hP_on_AC : P ∈ line_segment A C) 
  (hQ_on_AB : Q ∈ line_segment A B) 
  (hAP_fraction : dist A P = (1/3) * dist A C) 
  (hAQ_fraction : dist A Q = (1/3) * dist A B)
  (hABC_area : area (triangle A B C) = 36) : 
  area (triangle C P Q) = 8 := 
sorry

end area_CPQ_l469_469238


namespace freq_in_interval_50_infty_is_correct_l469_469019

-- Definitions for the frequencies and total samples
def frequencies : List (ℕ × ℕ) :=
  [(10, 20, 2), (20, 30, 3), (30, 40, 4), (40, 50, 5),
   (50, 60, 4), (60, 70, 2)]

def total_samples : ℕ := 20

-- Define the property to prove
def freq_interval_50_infty : ℝ := 
  (4 + 2 : ℕ) / total_samples

theorem freq_in_interval_50_infty_is_correct :
  freq_interval_50_infty = 0.3 := by
  sorry

end freq_in_interval_50_infty_is_correct_l469_469019


namespace cylinder_volume_l469_469333

theorem cylinder_volume (h_side : ℝ) (radius_side : ℝ) :
  (h_side = 2 * real.cbrt real.pi) ∧ (radius_side = 2 * real.cbrt real.pi) →
  ∃ V : ℝ, V = 2 :=
by
  sorry

end cylinder_volume_l469_469333


namespace least_common_multiple_first_ten_l469_469561

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469561


namespace statement_A_statement_B_statement_D_correct_statements_l469_469158

noncomputable def f : ℝ → ℝ := sorry -- Define the function f according to problem conditions

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom even_shift_f : ∀ x : ℝ, f (x + 1) = f(- (x + 1))

-- Additional data derived from conditions given in problem
axiom specific_interval_f : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f (x) = x^2

theorem statement_A : ∀ x : ℝ, f (x + 4) = f x := sorry
theorem statement_B : set.range f = {y : ℝ | -1 ≤ y ∧ y ≤ 1} := sorry
axiom not_monotonic_in_interval : ¬∀ x : ℝ, -4 ≤ x ∧ x ≤ -2 → f (x+1) < f x
theorem statement_D : ∀ x : ℝ, f (4 - x) = f (4 + x) := sorry

-- Encapsulate the correct answers
theorem correct_statements : 
  (∀ x, f (x + 4) = f x) ∧ 
  (set.range f = {y | -1 ≤ y ∧ y ≤ 1}) ∧ 
  ¬(∀ x, -4 ≤ x ∧ x ≤ -2 → f (x+1) < f x) ∧ 
  (∀ x, f (4 - x) = f (4 + x)) :=
  ⟨statement_A, statement_B, not_monotonic_in_interval, statement_D⟩

end statement_A_statement_B_statement_D_correct_statements_l469_469158


namespace find_length_of_AB_l469_469234

-- Define the setup of the problem
variable (A B C D : Type) [EuclideanTriangle A B C] [RightAngle (Angle B A C)]
variable (a b : ℝ)
variable (AD DC AC BC AB : ℝ)

-- Create the conditions
def triangle_conditions : Prop :=
  AD = 2 * DC ∧ AD = a ∧ BC = b ∧ AC = AD + DC

-- The target equation to prove
def target : ℝ := sqrt((9 * a^2 / 4) - b^2)

-- The theorem stating AB length
theorem find_length_of_AB (h : triangle_conditions AD DC AC BC AB a b) : AB = target a b :=
by sorry

end find_length_of_AB_l469_469234


namespace least_common_multiple_first_ten_l469_469557

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469557


namespace least_tax_l469_469747

def annualIncome : ℝ := 4500000

def rent : ℝ := 60000
def cosmetics : ℝ := 40000
def salaries : ℝ := 120000
def insurance : ℝ := 36000
def advertising : ℝ := 15000
def training : ℝ := 12000
def miscellaneous : ℝ := 20000

def monthlyExpenses : ℝ := rent + cosmetics + salaries + insurance + advertising + training + miscellaneous
def annualExpenses : ℝ := monthlyExpenses * 12
def expensesPaid (fraction : ℝ) : ℝ := fraction * annualExpenses

def taxableIncomeOSNO : ℝ := annualIncome - annualExpenses
def taxOSNO (rate : ℝ) : ℝ := taxableIncomeOSNO * rate

def taxUSN_Income (rate : ℝ) : ℝ := annualIncome * rate
def taxReduction (insurancePaid : ℝ) (maxReduction : ℝ) : ℝ := min (maxReduction * taxUSN_Income 0.06) insurancePaid
def finalTaxUSN_Income (insurPaid : ℝ) : ℝ := taxUSN_Income 0.06 - taxReduction insurPaid 0.50

def taxableIncomeUSN_Expenses (expenseFraction : ℝ) : ℝ := annualIncome - expensesPaid expenseFraction
def taxUSN_Expenses (rate : ℝ) : ℝ := taxableIncomeUSN_Expenses 0.45 * rate
def minTax (rate : ℝ) : ℝ := annualIncome * rate
def finalTaxUSN_Expenses (minRate : ℝ) (rate : ℝ) : ℝ := max (taxUSN_Expenses rate) (minTax minRate)

theorem least_tax {annualIncome annualExpenses : ℝ} :
  let annualExpenses := 12 * (rent + cosmetics + salaries + insurance + advertising + training + miscellaneous) in
  let taxOSNO := (annualIncome - annualExpenses) * 0.20 in
  let taxUSN_Income := annualIncome * 0.06 - min (0.50 * (annualIncome * 0.06)) (0.45 * insurance * 12) in
  let taxUSN_Expenses := max ((annualIncome - 0.45 * annualExpenses) * 0.15) (annualIncome * 0.01) in
  min taxOSNO (min taxUSN_Income taxUSN_Expenses) = 135000 :=
by
  have := annualIncome = 4500000
  have := rent = 720000
  have := cosmetics = 480000
  have := salaries = 1440000
  have := insurance = 432000
  have := advertising = 180000
  have := training = 144000
  have := miscellaneous = 240000
  sorry

end least_tax_l469_469747


namespace least_common_multiple_of_first_10_integers_l469_469508

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469508


namespace collinear_vectors_parallel_right_angle_triangle_abc_l469_469861

def vec_ab (k : ℝ) : ℝ × ℝ := (2 - k, -1)
def vec_ac (k : ℝ) : ℝ × ℝ := (1, k)

-- Prove that if vectors AB and AC are collinear, then k = 1 ± √2
theorem collinear_vectors_parallel (k : ℝ) :
  (2 - k) * k - 1 = 0 ↔ k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2 :=
by
  sorry

def vec_bc (k : ℝ) : ℝ × ℝ := (k - 1, k + 1)

-- Prove that if triangle ABC is right-angled, then k = 1 or k = -1 ± √2
theorem right_angle_triangle_abc (k : ℝ) :
  ( (2 - k) * 1 + (-1) * k = 0 ∨ (k - 1) * 1 + (k + 1) * k = 0 ) ↔ 
  k = 1 ∨ k = -1 + Real.sqrt 2 ∨ k = -1 - Real.sqrt 2 :=
by
  sorry

end collinear_vectors_parallel_right_angle_triangle_abc_l469_469861


namespace parabola_line_intersection_l469_469716

theorem parabola_line_intersection (y x : ℝ) (x1 x2 : ℝ) :
    let F := (2, 0)
    let L := (λ y, y = x - 2)
    let parabola := (λ y, y^2 = 8 * x)
    let A := (x1, y1)
    let B := (x2, y2)
    let inter_section_points := solve_system (λ y : ℝ, y = x - 2) (λ y : ℝ, y^2 = 8 * x) 
    let distances := ∀ x F, A B
    let parabola_eq := ∀ x, y^2 = 8 * x
    let focus := Fo := (2, 0)
    |FA| * |FB| = (x1 + 2) * (x2 + 2) = 32
  sorry

end parabola_line_intersection_l469_469716


namespace lcm_first_ten_l469_469416

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469416


namespace circle_equation_through_points_l469_469325

-- Definitions of the points A, B, and C
def A : ℝ × ℝ := (-1, -1)
def B : ℝ × ℝ := (2, 2)
def C : ℝ × ℝ := (-1, 1)

-- Prove that the equation of the circle passing through A, B, and C is (x - 1)^2 + y^2 = 5
theorem circle_equation_through_points :
  ∃ (D E F : ℝ), (∀ x y : ℝ, 
  x^2 + y^2 + D * x + E * y + F = 0 ↔
  x = -1 ∧ y = -1 ∨ 
  x = 2 ∧ y = 2 ∨ 
  x = -1 ∧ y = 1) ∧ 
  ∀ (x y : ℝ), x^2 + y^2 + D * x + E * y + F = 0 ↔ (x - 1)^2 + y^2 = 5 :=
by
  sorry

end circle_equation_through_points_l469_469325


namespace vladimir_can_invest_more_profitably_l469_469673

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end vladimir_can_invest_more_profitably_l469_469673


namespace least_common_multiple_first_ten_l469_469549

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469549


namespace least_common_multiple_of_first_ten_integers_l469_469369

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469369


namespace overall_gain_percentage_correct_l469_469722

def overall_gain_percentage (costs : list ℝ) (shipping_fees : list ℝ) (tax_rate : ℝ) (selling_prices : list ℝ) : ℝ :=
  let subtotals := list.map₂ (λ c s, c + s) costs shipping_fees in
  let total_costs := list.map (λ s, s * (1 + tax_rate)) subtotals in
  let total_cost_price := list.sum total_costs in
  let total_selling_price := list.sum selling_prices in
  ((total_selling_price - total_cost_price) / total_cost_price) * 100

theorem overall_gain_percentage_correct :
  overall_gain_percentage [900, 1200, 1500] [50, 75, 100] 0.05 [1080, 1320, 1650] = 0.84 := 
sorry

end overall_gain_percentage_correct_l469_469722


namespace least_common_multiple_of_first_ten_l469_469611

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469611


namespace find_t_find_angle_l469_469128

noncomputable theory

open RealEuclideanSpace

variables {E : Type*} [InnerProductSpace ℝ E] (a b : E)
variables (t : ℝ)

def is_unit (v : E) := ‖v‖ = 1

def is_magnitude_2 (v : E) := ‖v‖ = 2

def angle_eq_pi_third (a b : E) : Prop := ∠ a b = π / 3

def m_def (a b : E) := 3 • a - b

def n_def (a b : E) (t : ℝ) := t • a + 2 • b

def orthogonal (u v : E) : Prop := ⟪u, v⟫ = 0

theorem find_t (h1 : is_unit a) (h2 : is_magnitude_2 b) (h3 : angle_eq_pi_third a b) 
              (h4 : orthogonal (m_def a b) (n_def a b t)) : t = 1 :=
sorry

theorem find_angle (h1 : is_unit a) (h2 : is_magnitude_2 b) (h3 : angle_eq_pi_third a b) 
                   (h4 : t = 2) : ∠ (m_def a b) (n_def a b 2) = arccos (1 / 7) :=
sorry

end find_t_find_angle_l469_469128


namespace problem_II_S16_l469_469138

noncomputable def a_seq (n : ℕ) : ℤ := n + 1

def b_seq (a_seq : ℕ → ℤ) (n : ℕ) : ℤ :=
  if n % 2 = 1 then 2 ^ a_seq n else (2 / 3) * a_seq n

noncomputable def S (b_seq : (ℕ → ℤ) → ℕ → ℤ) (a_seq : ℕ → ℤ) (n : ℕ) : ℤ :=
  (Finset.range n).sum (b_seq a_seq)

theorem problem_II_S16 :
  S b_seq a_seq 16 = 1 / 3 * 4 ^ 9 + 52 :=
sorry

end problem_II_S16_l469_469138


namespace least_common_multiple_first_ten_l469_469642

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469642


namespace general_term_series1_l469_469661

theorem general_term_series1 (n : ℕ) : 
  let a := (λ n : ℕ, 1 / (n^2 : ℚ)) in
  a n = 1 / (n^2 : ℚ) :=
sorry

end general_term_series1_l469_469661


namespace lcm_first_ten_integers_l469_469632

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469632


namespace bulbs_sampling_l469_469688

theorem bulbs_sampling (total_bulbs : ℕ) (twenty_w_bulbs forty_w_bulbs sixty_w_bulbs sample_bulbs : ℕ)
  (ratio_twenty forty sixty : ℚ)
  (h_total : total_bulbs = 400)
  (h_ratio : ratio_twenty = 4 ∧ ratio_forty = 3 ∧ ratio_sixty = 1)
  (h_sum_ratio : ratio_twenty + ratio_forty + ratio_sixty = 8) 
  (h_type_count : twenty_w_bulbs = (ratio_twenty / 8 : ℚ) * total_bulbs ∧ 
                   forty_w_bulbs = (ratio_forty / 8 : ℚ) * total_bulbs ∧ 
                   sixty_w_bulbs = (ratio_sixty / 8 : ℚ) * total_bulbs)
  (h_sample : sample_bulbs = 40) :
  let sample_twenty := (twenty_w_bulbs / total_bulbs : ℚ) * sample_bulbs,
      sample_forty := (forty_w_bulbs / total_bulbs : ℚ) * sample_bulbs,
      sample_sixty := (sixty_w_bulbs / total_bulbs : ℚ) * sample_bulbs in
  sample_twenty = 20 ∧ sample_forty = 15 ∧ sample_sixty = 5 :=
by sorry

end bulbs_sampling_l469_469688


namespace team_leads_per_supervisor_l469_469710

def num_workers : ℕ := 390
def num_supervisors : ℕ := 13
def leads_per_worker_ratio : ℕ := 10

theorem team_leads_per_supervisor : (num_workers / leads_per_worker_ratio) / num_supervisors = 3 :=
by
  sorry

end team_leads_per_supervisor_l469_469710


namespace smallest_possible_b_l469_469313

theorem smallest_possible_b (a b : ℕ) (h1 : a - b = 6) 
  (h2 : Nat.gcd ((a ^ 3 + b ^ 3) / (a + b)) (a * b) = 9) : b = 3 :=
by
  sorry

end smallest_possible_b_l469_469313


namespace relay_race_conditions_l469_469808

theorem relay_race_conditions :
  ∃ arrangements : ℕ,
    (let A := 1 in let B := 2 in
     -- Condition 1: A and B in the middle two legs
     (arrangements = 60 ∧ 
      (run_middle_legs A B ∧ selection(others, excluding A, B)) ∨ 
      -- Condition 2: Only one among A and B selected and not run middle legs
      (arrangements = 480 ∧
       (run_excluding_middle(A ∨ B) ∧ selection(others, excluding A, B))) ∨
      -- Condition 3: Both A and B selected must run adjacent legs
      (arrangements = 180 ∧
       (run_adjacent_legs(A, B) ∧ selection(others, excluding A, B)) ))) :=
by
  sorry

-- Definitions used in the theorem
def run_middle_legs (A B : athlete) : Prop := -- define that A and B are in middle two legs
sorry

def run_excluding_middle (A_or_B : athlete) : Prop := -- define that either A or B is excluded from middle legs
sorry

def run_adjacent_legs (A B : athlete) : Prop := -- define that A and B are adjacent
sorry

def selection (others : list athlete, exclusions : list athlete) : Prop := -- define selection of remaining athletes
sorry

end relay_race_conditions_l469_469808


namespace isbn_check_digit_l469_469783

theorem isbn_check_digit (
  A B C E F G H I J : ℕ, 
  y : ℕ,
  hA : A = 9, hB : B = 6, hC : C = 2, hE : E = 7, hF : F = 0, 
  hG : G = 7, hH : H = 0, hI : I = 1, hJ : J = 5,
  S : ℕ := 10 * A + 9 * B + 8 * C + 7 * y + 6 * E + 5 * F + 4 * G + 3 * H + 2 * I,
  r : ℕ := S % 11,
  hJ_cond : (r ≠ 0 ∧ r ≠ 1 → J = 11 - r) ∧ (r = 0 → J = 0) ∧ (r = 1 → J = 0)
  ) : y = 7 := by
  sorry

end isbn_check_digit_l469_469783


namespace tan_2alpha_val_l469_469814

theorem tan_2alpha_val (α : ℝ) (h : 2 * sin (2 * α) = 1 + cos (2 * α)) : 
    tan (2 * α) = 4 / 3 ∨ tan (2 * α) = 0 :=
by
  sorry

end tan_2alpha_val_l469_469814


namespace integral_evaluation_l469_469759

-- Define the integrand function
def integrand (x : ℝ) : ℝ := (1 + Real.sin x - Real.cos x) ^ (-2)

-- Define the definite integral bounds
def a : ℝ := 2 * Real.arctan (1 / 2)
def b : ℝ := Real.pi / 2

-- State the theorem
theorem integral_evaluation : ∫ x in set.Icc a b, integrand x = 1 / 2 :=
by
  -- Initial setup and conditions
  let f := λ x : ℝ, (1 + Real.sin x - Real.cos x) ^ (-2)
  let lower_bound := 2 * Real.arctan (1 / 2)
  let upper_bound := Real.pi / 2

  -- Setting up the integral definition
  have h1 : ∫ x in set.Icc lower_bound upper_bound, f x = ∫ x in set.Icc a b, integrand x :=
    sorry -- prove equivalence of definitions

  -- Main integral evaluation
  have h2 : ∫ x in set.Icc lower_bound upper_bound, f x = 1 / 2 :=
    sorry -- prove the integral evaluates to 1/2

  -- Conclude equivalence
  exact eq.trans h1 h2


end integral_evaluation_l469_469759


namespace least_common_multiple_first_ten_l469_469650

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469650


namespace least_common_multiple_1_to_10_l469_469492

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469492


namespace sum_of_coordinates_l469_469222

theorem sum_of_coordinates {X Y Z G H J : (ℝ × ℝ)}
  (X_X : X = (0, 8))
  (Y_Y : Y = (0, 0))
  (Z_Z : Z = (10, 0))
  (G_G : G = ((X.1 + Y.1) / 2, (X.2 + Y.2) / 2))
  (H_H : H = ((Y.1 + Z.1) / 2, (Y.2 + Z.2) / 2))
  (line_XH : ∀ x, J.2 = -8 / 5 * J.1 + 8 ∧ J = (x, -8 / 5 * x + 8))
  (line_YG : J.1 = 0):
  J = (0, 8) → J.1 + J.2 = 8 :=
by intros; rw J; sorry

end sum_of_coordinates_l469_469222


namespace polynomial_factorization_l469_469214

theorem polynomial_factorization (m : ℝ) : 
  (∀ x : ℝ, (x - 7) * (x + 5) = x^2 + mx - 35) → m = -2 :=
by
  sorry

end polynomial_factorization_l469_469214


namespace alternating_colors_probability_l469_469690

def box_contains_five_white_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 1) = 5
def box_contains_five_black_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 0) = 5
def balls_drawn_one_at_a_time : Prop := true -- This condition is trivially satisfied without more specific constraints

theorem alternating_colors_probability (h1 : box_contains_five_white_balls) (h2 : box_contains_five_black_balls) (h3 : balls_drawn_one_at_a_time) :
  ∃ p : ℚ, p = 1 / 126 :=
sorry

end alternating_colors_probability_l469_469690


namespace f_of_2_eq_minus_1_l469_469876

def f (x : ℝ) : ℝ := sorry

theorem f_of_2_eq_minus_1 (f : ℝ → ℝ) (h : ∀ x, f(x) + 2 * f(1 / x) = 3 * x) : f 2 = -1 := 
by
  have h1 : f(2) + 2 * f(1 / 2) = 6 := h 2
  have h2 : f(1 / 2) + 2 * f(2) = 3 / 2 := h (1 / 2)
  sorry

end f_of_2_eq_minus_1_l469_469876


namespace time_to_produce_syrup_l469_469931

def cherries_per_quart : ℕ := 500
def hours_per_300_cherries : ℕ := 2
def total_hours_for_9_quarts : ℕ := 33
def quarts : ℕ := 9
def cherries_per_300 : ℕ := 300

theorem time_to_produce_syrup :
  let total_cherries_required := cherries_per_quart * quarts in
  let sets_of_cherries := total_cherries_required / cherries_per_300 in
  let time_picking_cherries := sets_of_cherries * hours_per_300_cherries in
  let time_making_syrup := total_hours_for_9_quarts - time_picking_cherries in
  time_making_syrup = 3 := sorry

end time_to_produce_syrup_l469_469931


namespace at_least_one_is_one_l469_469146

theorem at_least_one_is_one (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0):
  (1/x + 1/y + 1/z = 1) → (1/(x + y + z) = 1) → (x = 1 ∨ y = 1 ∨ z = 1) :=
by
  sorry

end at_least_one_is_one_l469_469146


namespace lcm_first_ten_positive_integers_l469_469435

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469435


namespace least_common_multiple_1_to_10_l469_469505

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469505


namespace multiplicity_of_X_minus_one_l469_469274

/-- 
  Let p > 3 be a prime number and k > 0 be a positive integer.
  Let f(X) = X^(p^k-1) + X^(p^k-2) + ... + X + 1.
  The multiplicity r such that (X - 1)^r divides f(X) \mod p, 
  but (X - 1)^(r+1) does not divide f(X) \mod p, is r = p^k.
-/
theorem multiplicity_of_X_minus_one (p : ℕ) (hp : Prime p) (hpg3 : 3 < p) 
  (k : ℕ) (hk : 0 < k) :
  let f := λ X : ℕ, (list.range (p^k)).sum (λ i, X^i)
  ∃ r : ℕ, (∀ X : ℕ, (X - 1)^r ∣ f X) ∧ (∀ X : ℕ, ¬(X - 1)^(r+1) ∣ f X) ∧ r = p^k :=
sorry

end multiplicity_of_X_minus_one_l469_469274


namespace jacqueline_gave_jane_88_fruits_l469_469927

theorem jacqueline_gave_jane_88_fruits (plums guavas apples oranges bananas fruits_left : ℕ) 
  (h1 : plums = 25) 
  (h2 : guavas = 30) 
  (h3 : apples = 36) 
  (h4 : oranges = 20) 
  (h5 : bananas = 15) 
  (h6 : fruits_left = 38) : 
  let initial_fruits := plums + guavas + apples + oranges + bananas in
  initial_fruits - fruits_left = 88 := 
by
  have h_initial : initial_fruits = 126 := sorry
  sorry

end jacqueline_gave_jane_88_fruits_l469_469927


namespace janice_total_flights_l469_469246

theorem janice_total_flights (u d: ℕ) (u_times d_times: ℕ) (flights_up flights_down: ℕ) 
    (hu: flights_up = u * u_times) (hd: flights_down = d * d_times) 
    (hu_value: u = 3) (u_times_value: u_times = 5) (d_value: d = 3) (d_times_value: d_times = 3) :
    flights_up + flights_down = 24 := 
by
  have h1 : flights_up = 3 * 5 := by rw [hu_value, u_times_value, hu]
  have h2 : flights_down = 3 * 3 := by rw [d_value, d_times_value, hd]
  rw [h1, h2]
  sorry

end janice_total_flights_l469_469246


namespace ocho_friends_total_l469_469965

theorem ocho_friends_total (boys_play_theater : ℕ) (half_friends_are_boys : Ocho.total_friends / 2 = boys_play_theater)
                          (boys_play_theater_eq_4 : boys_play_theater = 4) :
  Ocho.total_friends = 8 :=
by {
  have half_friends_are_girls : Ocho.total_friends / 2 = 4,
  sorry,
  exact half_friends_are_girls,
}

end ocho_friends_total_l469_469965


namespace lcm_first_ten_numbers_l469_469592

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469592


namespace simplify_and_evaluate_expr_l469_469305

theorem simplify_and_evaluate_expr (x y : ℚ) (h1 : x = -3/8) (h2 : y = 4) :
  (x - 2 * y) ^ 2 + (x - 2 * y) * (x + 2 * y) - 2 * x * (x - y) = 3 :=
by
  sorry

end simplify_and_evaluate_expr_l469_469305


namespace least_common_multiple_first_ten_l469_469651

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469651


namespace volume_parallelepiped_l469_469065

theorem volume_parallelepiped :
  ∃ (m n p : ℕ), let volume := (228 + 85 * Real.pi) / 3 in
  (m + n * Real.pi) / p = volume ∧ Nat.coprime n p ∧ (m + n + p) = 316 :=
by
  sorry

end volume_parallelepiped_l469_469065


namespace greatest_number_remainder_l469_469088

theorem greatest_number_remainder (G R : ℕ) (h1 : 150 % G = 50) (h2 : 230 % G = 5) (h3 : 175 % G = R) (h4 : ∀ g, g ∣ 100 → g ∣ 225 → g ∣ (175 - R) → g ≤ G) : R = 0 :=
by {
  -- This is the statement only; the proof is omitted as per the instructions.
  sorry
}

end greatest_number_remainder_l469_469088


namespace find_y_for_orthogonality_l469_469085

theorem find_y_for_orthogonality (y : ℝ) : (3 * y + 7 * (-4) = 0) → y = 28 / 3 := by
  sorry

end find_y_for_orthogonality_l469_469085


namespace physics_politics_related_probability_one_physics_one_politics_l469_469897

-- Definitions for conditions
def number_of_students := 200
def percent_physics : ℝ := 0.6
def percent_politics : ℝ := 0.75
def both_physics_politics := 80

-- Calculation based on provided conditions
def students_physics := number_of_students * percent_physics
def students_politics := number_of_students * percent_politics

-- Contingency table values
def a := both_physics_politics
def b := students_physics - both_physics_politics
def c := students_politics - both_physics_politics
def d := number_of_students - (students_physics + students_politics - both_physics_politics)

-- For K^2 calculation and comparison
def n := number_of_students
def K2 := (n * (a * d - b * c)^2) / ((a + b) * (c + d) * (a + c) * (b + d))
def critical_value : ℝ := 10.828

-- Probability calculation for part (2)
def choices_A := Set.of_list ["Physics", "Chemistry", "Biology", "History", "Geography", "Politics"]
def choices_B := Set.of_list ["Physics", "Chemistry", "Biology", "History", "Geography", "Politics"]
def choices_C := Set.of_list ["Physics", "Chemistry", "Biology"]

def total_prob := (3/5 : ℝ)

-- Statement for part (1)
theorem physics_politics_related : 
  K2 > critical_value := by
  sorry

-- Statement for part (2)
theorem probability_one_physics_one_politics : 
  (3/5 : ℝ) := by
  sorry

end physics_politics_related_probability_one_physics_one_politics_l469_469897


namespace johnny_walking_dogs_l469_469932

theorem johnny_walking_dogs (total_legs : ℕ) (human_legs : ℕ) (dog_legs : ℕ) (n_dogs : ℕ) : 
  total_legs = 12 ∧ human_legs = 4 ∧ dog_legs = 4 → n_dogs = 2 :=
begin
  sorry,
end

end johnny_walking_dogs_l469_469932


namespace min_value_f_l469_469777

noncomputable def f (x : ℝ) : ℝ := 2 * x^2 + 24 * x + 128 / x^3

theorem min_value_f : ∃ x > 0, f x = 168 :=
by
  sorry

end min_value_f_l469_469777


namespace ribbon_tape_remaining_l469_469934

theorem ribbon_tape_remaining 
  (initial_length used_for_ribbon used_for_gift : ℝ)
  (h_initial: initial_length = 1.6)
  (h_ribbon: used_for_ribbon = 0.8)
  (h_gift: used_for_gift = 0.3) : 
  initial_length - used_for_ribbon - used_for_gift = 0.5 :=
by 
  sorry

end ribbon_tape_remaining_l469_469934


namespace least_common_multiple_of_first_ten_positive_integers_l469_469464

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469464


namespace ratio_roses_to_lilacs_l469_469124

theorem ratio_roses_to_lilacs
  (L: ℕ) -- number of lilacs sold
  (G: ℕ) -- number of gardenias sold
  (R: ℕ) -- number of roses sold
  (hL: L = 10) -- defining lilacs sold as 10
  (hG: G = L / 2) -- defining gardenias sold as half the lilacs
  (hTotal: R + L + G = 45) -- defining total flowers sold as 45
  : R / L = 3 :=
by {
  -- The actual proof would go here, but we skip it as per instructions
  sorry
}

end ratio_roses_to_lilacs_l469_469124


namespace tan_sum_l469_469873

theorem tan_sum (x y : ℝ) (hx1 : sin x + sin y = 5 / 13) (hx2 : cos x + cos y = 12 / 13) :
  tan x + tan y = 240 / 119 :=
by
  sorry

end tan_sum_l469_469873


namespace vector_orthogonal_vector_parallel_l469_469862

noncomputable def vector_a := (1, -1, 0)
noncomputable def vector_b := (-1, 0, 1)
noncomputable def vector_c := (2, -3, 1)

theorem vector_orthogonal : (vector_a + 5 * vector_b) • vector_c = 0 :=
sorry

theorem vector_parallel : ∃ k : ℝ, vector_a = k • (vector_b - vector_c) :=
sorry

end vector_orthogonal_vector_parallel_l469_469862


namespace T_n_less_than_one_l469_469856

def a : ℕ+ → ℕ
| 1       := 1
| (n + 1) := 2 * a n + 1

def c (n : ℕ+) := (2 ^ (n : ℕ)) / ((a n) * a (n + 1))

def T (n : ℕ+) := ∑ k in Finset.range n, c (k + 1)

theorem T_n_less_than_one (n : ℕ+) : T n < 1 := sorry

end T_n_less_than_one_l469_469856


namespace rosy_days_proof_l469_469668

-- Define Mary's work rate
def mary_work_rate : ℚ := 1 / 11

-- Define Rosy's efficiency relative to Mary
def rosy_efficiency : ℚ := 1.10

-- Given Rosy's work rate
def rosy_work_rate : ℚ := rosy_efficiency * mary_work_rate

-- The number of days Rosy takes to complete the work
def rosy_days : ℕ := 1 / rosy_work_rate

-- Assert that Rosy takes 10 days to complete the work
theorem rosy_days_proof : rosy_days = 10 := sorry

end rosy_days_proof_l469_469668


namespace find_x_l469_469113

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 12) : x = 27 :=
begin
  sorry
end

end find_x_l469_469113


namespace students_participate_in_both_l469_469903

theorem students_participate_in_both (total_students band_students sports_students either_band_or_sports_students : ℕ)
    (h_total : total_students = 320)
    (h_band : band_students = 85)
    (h_sports : sports_students = 200)
    (h_either_band_or_sports : either_band_or_sports_students = 225)
    : exists number_of_students_both_activities : ℕ, number_of_students_both_activities = 60 :=
by
  use 60
  sorry

end students_participate_in_both_l469_469903


namespace mod_equiv_m_in_range_l469_469201

theorem mod_equiv_m_in_range :
  ∃ m ∈ (set.range (λ n, 150 + n)).inter (set.Icc 150 201),
  (25 - 98) ≡ m [MOD 53] :=
by {
  let c := 25,
  let d := 98,
  let m := 192,
  have h_c : c ≡ 25 [MOD 53] := by refl,
  have h_d : d ≡ 98 [MOD 53] := by refl,
  have h_cd_mod : (c - d) % 53 = (-20) % 53 := by norm_num,
  have h_cd_equiv : (c - d) ≡ -20 [MOD 53] := int.modeq_of_dvd,
  have h_cd_33 : (-20) % 53 = 33 := by norm_num,
  have h_range : m ∈ set.range (λ n, 150 + n).inter (set.Icc 150 201),
  { use 42,
    split,
    { use 42,
      refl },
    { split,
      linarith,
      linarith } },
  use m,
  exact ⟨h_range, int.modeq.trans h_cd_equiv (int.modeq.of_eq h_cd_33)⟩
} 

end mod_equiv_m_in_range_l469_469201


namespace least_common_multiple_of_first_ten_positive_integers_l469_469459

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469459


namespace rabbit_catches_up_at_time_l469_469356

-- Definitions of the conditions
def rabbit_acceleration := 2 * 60 -- in miles per hour squared (120 mph²)
def cat_speed := 20 -- in miles per hour
def head_start_time := 15 / 60 -- 0.25 hours (15 minutes converted to hours)
def cat_start_distance := cat_speed * head_start_time -- 5 miles

-- Theorem statement
theorem rabbit_catches_up_at_time : ∃ t : ℝ, t = (15 / 60) + Real.sqrt (5 / (0.5 * rabbit_acceleration)) :=
by
  sorry

end rabbit_catches_up_at_time_l469_469356


namespace min_a1_l469_469947

noncomputable def a : ℕ → ℝ
| 0       => arbitrary ℝ -- this represents a_0 for convenience.
| (n + 1) => 8 * a n - (n + 1)^2

theorem min_a1 : ∀ n > 1, ∃ a1 : ℝ, a 1 = 8 * a 0 - 1^2 ∧ a 0 > 0 ∧ a 1 = (2 : ℝ) / 7 :=
by
  sorry

end min_a1_l469_469947


namespace no_solution_iff_n_eq_minus_half_l469_469881

theorem no_solution_iff_n_eq_minus_half (n x y z : ℝ) :
  (¬∃ x y z : ℝ, 2 * n * x + y = 2 ∧ n * y + z = 2 ∧ x + 2 * n * z = 2) ↔ n = -1 / 2 :=
by
  sorry

end no_solution_iff_n_eq_minus_half_l469_469881


namespace towel_bleach_percentage_decrease_l469_469735

-- Define the problem
theorem towel_bleach_percentage_decrease (L B : ℝ) (x : ℝ) (h_length : 0 < L) (h_breadth : 0 < B) 
  (h1 : 0.64 * L * B = 0.8 * L * (1 - x / 100) * B) :
  x = 20 :=
by
  -- The actual proof is not needed, providing "sorry" as a placeholder for the proof.
  sorry

end towel_bleach_percentage_decrease_l469_469735


namespace part1_l469_469955

def p (m x : ℝ) := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) := (x + 2)^2 < 1

theorem part1 (x : ℝ) (m : ℝ) (hm : m = -2) : p m x ∧ q x ↔ -3 < x ∧ x ≤ -2 :=
by
  unfold p q
  sorry

end part1_l469_469955


namespace kriptonita_price_difference_le_100_l469_469968

/- Definitions for conditions -/
variables {City : Type} (diametric_opposite : City → City)
variables (Road : City → City → Prop)
variables (Price : City → ℝ)

/- Conditions -/
axiom diametric_opposite_involution : ∀ P : City, diametric_opposite (diametric_opposite P) = P
axiom road_symmetry : ∀ P Q : City, Road P Q ↔ Road (diametric_opposite P) (diametric_opposite Q)
axiom roads_do_not_cross : ∀ P Q R S : City, Road P Q → Road R S → P = R ∨ P = S ∨ Q = R ∨ Q = S
axiom connected_graph : ∀ P Q : City, ∃ path : list City, path.head = P ∧ path.last = Q ∧ ∀ (i : ℕ), i < path.length - 1 → Road (path.nth_le i sorry) (path.nth_le (i + 1) sorry)
axiom price_difference_constraint : ∀ P Q : City, Road P Q → abs (Price P - Price Q) ≤ 100

/- Main theorem -/
theorem kriptonita_price_difference_le_100 :
  ∃ P : City, abs (Price P - Price (diametric_opposite P)) ≤ 100 :=
sorry -- Proof goes here

end kriptonita_price_difference_le_100_l469_469968


namespace all_positive_integers_l469_469939

def regular_n_gon (n : ℕ) : Type := sorry -- As the exact construction of a regular n-gon is being abstracted

-- Define the concept of a triangle within an n-gon
def triangle (n : ℕ) (gon : regular_n_gon n) (i j k : ℕ) : Type := sorry

-- Define triangle types
inductive triangle_type
| acute
| right
| obtuse

-- Define a function that determines the type of a triangle
def triangle_type_of_triangle {n : ℕ} (t : triangle n (regular_n_gon n) 0 0 0) : triangle_type := sorry

theorem all_positive_integers (n : ℕ) (h_pos : n > 0) :
  ∀ (σ : perm (fin n)), ∃ (i j k : fin n), 
    triangle_type_of_triangle (triangle n (regular_n_gon n) i j k) =
    triangle_type_of_triangle (triangle n (regular_n_gon n) (σ i) (σ j) (σ k)) :=
sorry

end all_positive_integers_l469_469939


namespace least_divisible_1_to_10_l469_469526

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469526


namespace coffee_consumption_l469_469245

-- Defining the necessary variables and conditions
variable (Ivory_cons Brayan_cons : ℕ)
variable (hr : ℕ := 1)
variable (hrs : ℕ := 5)

-- Condition: Brayan drinks twice as much coffee as Ivory
def condition1 := Brayan_cons = 2 * Ivory_cons

-- Condition: Brayan drinks 4 cups of coffee in an hour
def condition2 := Brayan_cons = 4

-- The proof problem
theorem coffee_consumption : ∀ (Ivory_cons Brayan_cons : ℕ), (Brayan_cons = 2 * Ivory_cons) → 
  (Brayan_cons = 4) → 
  ((Brayan_cons * hrs) + (Ivory_cons * hrs) = 30) :=
by
  intro hBrayan hIvory hr
  sorry

end coffee_consumption_l469_469245


namespace solve_for_X_l469_469780

variable (X Y : ℝ)
variable (hY : Y = 16 * X)

theorem solve_for_X (hY : Y = 16 * X) : X = Y / 16 := by
  rw [hY]
  ring
  sorry

end solve_for_X_l469_469780


namespace quadratic_does_not_pass_third_quadrant_l469_469171

-- Definitions of the functions
def linear_function (a b x : ℝ) : ℝ := -a * x + b
def quadratic_function (a b x : ℝ) : ℝ := -a * x^2 + b * x

-- Conditions
variables (a b : ℝ)
axiom a_nonzero : a ≠ 0
axiom passes_first_third_fourth : ∀ x, (linear_function a b x > 0 ∧ x > 0) ∨ (linear_function a b x < 0 ∧ x < 0) ∨ (linear_function a b x < 0 ∧ x > 0)

-- Theorem stating the problem
theorem quadratic_does_not_pass_third_quadrant :
  ¬ (∃ x, quadratic_function a b x < 0 ∧ x < 0) := 
sorry

end quadratic_does_not_pass_third_quadrant_l469_469171


namespace Benny_spent_95_dollars_l469_469048

theorem Benny_spent_95_dollars
    (amount_initial : ℕ)
    (amount_left : ℕ)
    (amount_spent : ℕ) :
    amount_initial = 120 →
    amount_left = 25 →
    amount_spent = amount_initial - amount_left →
    amount_spent = 95 :=
by
  intros h_initial h_left h_spent
  rw [h_initial, h_left] at h_spent
  exact h_spent

end Benny_spent_95_dollars_l469_469048


namespace magic_square_det_div_sum_is_int_l469_469320

def is_magic_square (M : Matrix (Fin 3) (Fin 3) ℤ) : Prop := 
  ∃ N : ℤ,
    -- Each row sums to N
    (M 0 0 + M 0 1 + M 0 2 = N) ∧
    (M 1 0 + M 1 1 + M 1 2 = N) ∧
    (M 2 0 + M 2 1 + M 2 2 = N) ∧
    -- Each column sums to N
    (M 0 0 + M 1 0 + M 2 0 = N) ∧
    (M 0 1 + M 1 1 + M 2 1 = N) ∧
    (M 0 2 + M 1 2 + M 2 2 = N) ∧
    -- Each diagonal sums to N
    (M 0 0 + M 1 1 + M 2 2 = N) ∧
    (M 0 2 + M 1 1 + M 2 0 = N)

theorem magic_square_det_div_sum_is_int (M : Matrix (Fin 3) (Fin 3) ℤ) 
  (h : is_magic_square M) :
  ∃ k : ℤ, ((Matrix.det M) : ℚ) / (∑ i j, (M i j) : ℚ) = k := 
sorry

end magic_square_det_div_sum_is_int_l469_469320


namespace function_not_differentiable_at_l469_469045

noncomputable def is_differentiable_at (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ f', ContinuousAt (λ x, (f x - f' * x) / x) 0

theorem function_not_differentiable_at :
  ¬ is_differentiable_at (λ x => (1/x) + 2*x) 0 :=
sorry

end function_not_differentiable_at_l469_469045


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469399

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469399


namespace range_f_value_f_B_l469_469166

def f (x : ℝ) : ℝ := 2 * sqrt 3 * sin x * cos x - 3 * (sin x) ^ 2 - (cos x) ^ 2 + 3

theorem range_f (x : ℝ) (hx : 0 ≤ x ∧ x ≤ π / 2) :
  0 ≤ f x ∧ f x ≤ 3 := 
sorry


noncomputable def f_B (A B C a b c : ℝ) (h1 : b / a = sqrt 3) 
  (h2 : sin (2 * A + C) / sin A = 2 + 2 * cos (A + C)) : ℝ := f B

theorem value_f_B (A B C a b c : ℝ) (h1 : b / a = sqrt 3) 
  (h2 : sin (2 * A + C) / sin A = 2 + 2 * cos (A + C)) :
  f_B A B C a b c h1 h2 = 2 := 
sorry

end range_f_value_f_B_l469_469166


namespace num_zeros_in_decimal_expansion_of_one_div_40_pow_40_l469_469068

theorem num_zeros_in_decimal_expansion_of_one_div_40_pow_40 :
  let x := (1 / (40 ^ 40)) in
  let y := x.div 1 in
  76 = number_of_trailing_zeros_in_decimal y :=
sorry

end num_zeros_in_decimal_expansion_of_one_div_40_pow_40_l469_469068


namespace domain_g_l469_469839

variable (f : ℝ → ℝ)

-- Defining the condition that the domain of f(x) is [1, 3]
def domain_f (x : ℝ) : Prop := 1 <= x ∧ x <= 3

-- The function g(x)
def g (x : ℝ) : ℝ := f (3 * x - 2) / (2 * x - 3)

-- Proving the domain of g(x) == [1, 3/2) ∪ (3/2, 5/3]
theorem domain_g : {x : ℝ | 1 ≤ x ∧ x < 3/2 ∨ 3/2 < x ∧ x ≤ 5/3} = 
  {x : ℝ | domain_f (3 * x - 2) ∧ 2 * x - 3 ≠ 0} :=
by
  sorry

end domain_g_l469_469839


namespace least_common_multiple_of_first_ten_positive_integers_l469_469455

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469455


namespace least_common_multiple_first_ten_l469_469634

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469634


namespace highest_power_of_2_divides_n_highest_power_of_3_divides_n_l469_469775

noncomputable def n : ℕ := 15^4 - 11^4

theorem highest_power_of_2_divides_n : ∃ k : ℕ, 2^4 = 16 ∧ 2^(k) ∣ n :=
by
  sorry

theorem highest_power_of_3_divides_n : ∃ m : ℕ, 3^0 = 1 ∧ 3^(m) ∣ n :=
by
  sorry

end highest_power_of_2_divides_n_highest_power_of_3_divides_n_l469_469775


namespace minimize_G_l469_469867

def F (p q : ℝ) := -4*p*q + 5*p*(1 - q) + 2*(1 - p)*q - 3*(1 - p)*(1 - q)

def G (p : ℝ) := max (F p 0) (F p 1)

theorem minimize_G : ∃ p, 0 ≤ p ∧ p ≤ 1 ∧ p = 5 / 14 ∧ ∀ p', 0 ≤ p' → p' ≤ 1 → G p ≤ G p' :=
by
  sorry

end minimize_G_l469_469867


namespace least_positive_integer_divisible_by_first_ten_l469_469487

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469487


namespace lcm_first_ten_positive_integers_l469_469432

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469432


namespace residue_mod_2000_l469_469265

theorem residue_mod_2000 :
  let T := (∑ k in finset.range 2000, if k % 2 = 0 then - (k + 1) else k + 1)
  in T % 2000 = 0 :=
by
  let T := (∑ k in finset.range 2000, if k % 2 = 0 then - (k + 1) else k + 1)
  have h : T = 0 := sorry
  exact (congr_arg (λ x, x % 2000) h)

end residue_mod_2000_l469_469265


namespace diet_soda_bottles_l469_469007

def total_bottles : ℕ := 17
def regular_soda_bottles : ℕ := 9

theorem diet_soda_bottles : total_bottles - regular_soda_bottles = 8 := by
  sorry

end diet_soda_bottles_l469_469007


namespace savings_percentage_is_22_16_l469_469980

-- Definitions
def basic_salary : ℝ := 1250
def commission_rate : ℝ := 0.10
def total_sales : ℝ := 23600
def monthly_expenses : ℝ := 2888

-- Calculate the commission.
def commission : ℝ := commission_rate * total_sales

-- Calculate total earnings.
def total_earnings : ℝ := basic_salary + commission

-- Calculate savings.
def savings : ℝ := total_earnings - monthly_expenses

-- Calculate percentage of savings.
def savings_percentage : ℝ := (savings / total_earnings) * 100

-- Theorem statement
theorem savings_percentage_is_22_16 : savings_percentage = 22.16 :=
by
  -- Placeholder for the proof using sorry to pass Lean compilation
  sorry

end savings_percentage_is_22_16_l469_469980


namespace lcm_first_ten_integers_l469_469619

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469619


namespace least_common_multiple_1_to_10_l469_469437

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469437


namespace fraction_of_janes_age_is_five_eighths_l469_469929

/-- Jane's current age -/
def jane_current_age : ℕ := 34

/-- Number of years ago Jane stopped babysitting -/
def years_since_stopped_babysitting : ℕ := 10

/-- Current age of the oldest child Jane could have babysat -/
def oldest_child_current_age : ℕ := 25

/-- Calculate Jane's age when she stopped babysitting -/
def jane_age_when_stopped_babysitting : ℕ := jane_current_age - years_since_stopped_babysitting

/-- Calculate the child's age when Jane stopped babysitting -/
def oldest_child_age_when_jane_stopped : ℕ := oldest_child_current_age - years_since_stopped_babysitting 

/-- Calculate the fraction of Jane's age that the child could be at most -/
def babysitting_age_fraction : ℚ := (oldest_child_age_when_jane_stopped : ℚ) / (jane_age_when_stopped_babysitting : ℚ)

theorem fraction_of_janes_age_is_five_eighths :
  babysitting_age_fraction = 5 / 8 :=
by 
  -- Declare the proof steps (this part is the placeholder as proof is not required)
  sorry

end fraction_of_janes_age_is_five_eighths_l469_469929


namespace three_digit_number_is_112_l469_469734

theorem three_digit_number_is_112 (a b c : ℕ) (h1 : 1 ≤ a ∧ a ≤ 9) (h2 : 0 ≤ b ∧ b ≤ 9) (h3 : 1 ≤ c ∧ c ≤ 9) (h4 : 100 * a + 10 * b + c = 56 * c) :
  100 * a + 10 * b + c = 112 :=
by sorry

end three_digit_number_is_112_l469_469734


namespace max_largest_distinct_natural_numbers_l469_469319

theorem max_largest_distinct_natural_numbers (a b c d e : ℕ) 
  (h_distinct : list.nodup [a, b, c, d, e]) 
  (h_avg : (a + b + c + d + e) / 5 = 12) 
  (h_median : median [a, b, c, d, e] = 17) : 
  (a = 24 ∨ b = 24 ∨ c = 24 ∨ d = 24 ∨ e = 24) :=
sorry

end max_largest_distinct_natural_numbers_l469_469319


namespace linear_equation_a_neg2_l469_469193

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l469_469193


namespace least_common_multiple_1_to_10_l469_469440

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469440


namespace lcm_first_ten_integers_l469_469616

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469616


namespace product_of_faces_and_vertices_of_cube_l469_469091

def number_of_faces := 6
def number_of_vertices := 8

theorem product_of_faces_and_vertices_of_cube : number_of_faces * number_of_vertices = 48 := 
by 
  sorry

end product_of_faces_and_vertices_of_cube_l469_469091


namespace least_positive_integer_divisible_by_first_ten_l469_469488

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469488


namespace length_of_room_l469_469335

theorem length_of_room (width : ℝ) (cost_per_sq_meter : ℝ) (total_cost : ℝ) (L : ℝ) 
  (h_width : width = 2.75)
  (h_cost_per_sq_meter : cost_per_sq_meter = 600)
  (h_total_cost : total_cost = 10725)
  (h_area_cost_eq : total_cost = L * width * cost_per_sq_meter) : 
  L = 6.5 :=
by 
  simp [h_width, h_cost_per_sq_meter, h_total_cost, h_area_cost_eq] at *
  sorry

end length_of_room_l469_469335


namespace least_common_multiple_of_first_ten_l469_469603

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469603


namespace least_divisible_1_to_10_l469_469531

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469531


namespace simplify_fraction_l469_469308

variable (x : ℝ)

theorem simplify_fraction : (x + 1) / (x^2 + 2*x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l469_469308


namespace ceiling_summation_2019_l469_469767

noncomputable def ceiling_of_summation : ℤ :=
  ⌈ ∑' k in (set.Ici 2018 : set ℕ), (2019.factorial - 2018.factorial) / k.factorial ⌉

theorem ceiling_summation_2019 :
  ceiling_of_summation = 2019 :=
sorry

end ceiling_summation_2019_l469_469767


namespace num_valid_subsets_l469_469863

-- Defining the set from which subsets are selected
def S : Finset ℕ := Finset.range 15 -- Finset.range n gives {0, 1, 2, ..., n-1}

-- Defining a particular subset of size 7 whose sum is divisible by 14
def valid_subsets := { A : Finset ℕ // A ⊆ S ∧ A.card = 7 ∧ (A.sum id) % 14 = 0 }

-- Problem Statement: Verifying the number of such subsets is exactly 245
theorem num_valid_subsets : Fintype.card valid_subsets = 245 :=
  sorry

end num_valid_subsets_l469_469863


namespace pd_pe_pf_triangle_iff_p_inside_xyz_l469_469253

-- Definitions of the points and triangles
variables {A B C P I D E F X Y Z : Type}

-- Definitions: I is the incenter, DEF is the pedal triangle of P, XYZ is the Cevian triangle of I
def is_incenter (I : Type) (A B C : Type) : Prop := sorry
def is_pedal_triangle (D E F P A B C : Type) : Prop := sorry
def is_cevian_triangle (X Y Z I A B C : Type) : Prop := sorry

-- Definitions of the points D, E, F and X, Y, Z as intersections
def is_foot_of_altitude (D P B C : Type) : Prop := sorry
def is_foot_of_altitude (E P C A : Type) : Prop := sorry
def is_foot_of_altitude (F P A B : Type) : Prop := sorry

def is_intersection (X A I B C : Type) : Prop := sorry
def is_intersection (Y B I A C : Type) : Prop := sorry
def is_intersection (Z C I A B : Type) : Prop := sorry

-- Prove equivalence of conditions
theorem pd_pe_pf_triangle_iff_p_inside_xyz
  (h_incenter : is_incenter I A B C)
  (h_pedal : is_pedal_triangle D E F P A B C)
  (h_cevian : is_cevian_triangle X Y Z I A B C)
  (h_footD : is_foot_of_altitude D P B C)
  (h_footE : is_foot_of_altitude E P C A)
  (h_footF : is_foot_of_altitude F P A B)
  (h_intX : is_intersection X A I B C)
  (h_intY : is_intersection Y B I A C)
  (h_intZ : is_intersection Z C I A B) :
  triangle_formed_by_sides (PD : ℝ) (PE : ℝ) (PF : ℝ) ↔ point_inside_triangle P X Y Z :=
sorry

end pd_pe_pf_triangle_iff_p_inside_xyz_l469_469253


namespace triangle_ABC_proof_l469_469240

noncomputable def triangle_proof (A B C : ℝ) (a b c : ℝ) : Prop :=
  A = 45 ∧ C = 30 ∧ c = 10 → 
  B = 105 ∧ a = 10 * real.sqrt 2 ∧ b = 5 * (real.sqrt 6 + real.sqrt 2)

theorem triangle_ABC_proof : 
  triangle_proof 45 105 30 (10 * real.sqrt 2) (5 * (real.sqrt 6 + real.sqrt 2)) 10 :=
begin
  sorry
end

end triangle_ABC_proof_l469_469240


namespace smallest_integer_solution_m_l469_469886

theorem smallest_integer_solution_m :
  (∃ x y m : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) →
  ∃ m : ℤ, (∀ x y : ℝ, 2 * x + y = 4 ∧ x + 2 * y = -3 * m + 2 ∧ x - y > -3/2) ↔ m = -1 :=
by
  sorry

end smallest_integer_solution_m_l469_469886


namespace chord_length_of_line_and_circle_l469_469992

theorem chord_length_of_line_and_circle :
  let c := (-2, 2) in
  let r := Real.sqrt 2 in
  let d := (|-2 - 2 + 3|) / (Real.sqrt 2) in
  let chord_length := 2 * (Real.sqrt (2 - (d ^ 2))) in
  (x - y + 3 = 0) ∧ ((x + 2) ^ 2 + (y - 2) ^ 2 = 2) → chord_length = Real.sqrt 6 :=
by
  sorry

end chord_length_of_line_and_circle_l469_469992


namespace probability_abs_diff_two_l469_469809

open Finset

def S : Finset ℤ := {1, 2, 3, 4, 5}

def num_pairs_with_abs_diff_two : ℤ :=
  (S.product S).filter (λ (x : ℤ × ℤ), x.1 ≠ x.2 ∧ |x.1 - x.2| = 2).card

def total_pairs : ℤ :=
  (S.product S).filter (λ (x : ℤ × ℤ), x.1 ≠ x.2).card

theorem probability_abs_diff_two : 
  (num_pairs_with_abs_diff_two / total_pairs : ℝ) = 3 / 10 :=
by 
  -- Insert steps for the proof here or transformation from ℤ to ℝ w.r.t cardinality.
  sorry

end probability_abs_diff_two_l469_469809


namespace least_common_multiple_first_ten_integers_l469_469566

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469566


namespace extremum_points_range_l469_469165

noncomputable def f (a x : ℝ) : ℝ := (1 / 3) * x ^ 3 + a * x ^ 2 + x + 1

theorem extremum_points_range (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f a x1 = f a x2) ↔ a ∈ Set.Ioo (-∞ : ℝ) (-1) ∪ Set.Ioo (1) (∞ : ℝ) :=
by
  sorry

end extremum_points_range_l469_469165


namespace sin_double_angle_log_simplification_l469_469679

-- Problem 1: Prove sin(2 * α) = 7 / 25 given sin(α - π / 4) = 3 / 5
theorem sin_double_angle (α : ℝ) (h : Real.sin (α - Real.pi / 4) = 3 / 5) : Real.sin (2 * α) = 7 / 25 :=
by
  sorry

-- Problem 2: Prove 2 * log₅ 10 + log₅ 0.25 = 2
theorem log_simplification : 2 * Real.log 10 / Real.log 5 + Real.log (0.25) / Real.log 5 = 2 :=
by
  sorry

end sin_double_angle_log_simplification_l469_469679


namespace least_divisible_1_to_10_l469_469543

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469543


namespace tangent_line_at_zero_range_of_a_minimum_value_eq_existence_of_b_l469_469141

-- f(x) = e^x and g(x) = ax - ln(x)
def f (x : ℝ) : ℝ := Real.exp x
def g (a x : ℝ) : ℝ := a * x - Real.log x
def h (a x : ℝ) : ℝ := Real.exp x - a * x

-- Part 1: Prove the equation of the tangent line to y = e^x at x = 0
theorem tangent_line_at_zero : 
  ∀ x : ℝ, 
    f x = x + 1 ↔ x = 0 := sorry

-- Part 2: Prove the range of a such that ∀ x ∈ (1, +∞), g(x) < x ln(x) + a
theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 1 < x → g a x < x * Real.log x + a) ↔ a ≤ 2 := sorry

-- Part 3(i): Prove a = 1 given the minimum value requirement
theorem minimum_value_eq (a : ℝ) :
  (∀ x : ℝ, h a x = g a x) → a = 1 := sorry

-- Part 3(ii): Prove existence of b > 1 with three distinct roots in arithmetic progression
theorem existence_of_b :
  ∃ b : ℝ, b > 1 ∧ 
  (∃ x₁ x₂ x₃ : ℝ, (h 1 x₁ = b ∧ g 1 x₁ = b) ∧ 
                   (h 1 x₂ = b ∧ g 1 x₂ = b) ∧ 
                   (h 1 x₃ = b ∧ g 1 x₃ = b) ∧ 
                   x₁ < x₂ ∧ x₂ < x₃ ∧ 
                   2 * x₂ = x₁ + x₃) := sorry

end tangent_line_at_zero_range_of_a_minimum_value_eq_existence_of_b_l469_469141


namespace calculate_expression_l469_469757

theorem calculate_expression :
  ((-1 -2 -3 -4 -5 -6 -7 -8 -9 -10) * (1 -2 +3 -4 +5 -6 +7 -8 +9 -10) = 275) :=
by
  sorry

end calculate_expression_l469_469757


namespace smallest_N_of_seven_integers_is_twelve_l469_469983

open BigOperators

theorem smallest_N_of_seven_integers_is_twelve {a : Fin 7 → ℕ} 
  (h_distinct : Function.Injective a)
  (h_product : (∏ i, a i) = n^3) : 
  ∃ i, a i = 12 ∧ ∀ j, a j ≤ a i := sorry

end smallest_N_of_seven_integers_is_twelve_l469_469983


namespace base_amount_calculation_l469_469003

theorem base_amount_calculation (tax_amount : ℝ) (tax_rate : ℝ) (base_amount : ℝ) 
  (h1 : tax_amount = 82) (h2 : tax_rate = 82) : base_amount = 100 :=
by
  -- Proof will be provided here.
  sorry

end base_amount_calculation_l469_469003


namespace unique_factorial_representation_l469_469295

theorem unique_factorial_representation (n : ℕ) : 
  ∃! (a : ℕ → ℤ), 
    (∀ i, 0 ≤ a i ∧ a i ≤ i ∧ ∃ k, k ≤ n ∧ (∑ i in finset.range (k + 1), a i * (nat.factorial i)) = n ∧ a k ≠ 0) :=
sorry

end unique_factorial_representation_l469_469295


namespace lcm_first_ten_numbers_l469_469590

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469590


namespace exists_triangle_with_given_angle_bisectors_l469_469181

open scoped Classical

variables {P Q : Point} {f : Line}

/-- Given two points \( P \) and \( Q \) in the plane and a line \( f \), this theorem states that there exists a triangle \( ABC \) with vertex \( A \) on \( f \), 
such that the angle bisector from \( A \) lies along \( f \), and the intersection points of the angle bisectors of \( \angle B \) and \( \angle C \) with sides \( AB \) 
and \( AC \) are \( P \) and \( Q \), respectively. -/
theorem exists_triangle_with_given_angle_bisectors (h1 : is_point P) (h2 : is_point Q) (h3 : is_line f) :
  ∃ (A B C : Point), (is_point A) ∧ (is_point B) ∧ (is_point C) ∧ 
  (A ∈ f) ∧ 
  (angle_bisector_line A (triangle_side AB AC) = f) ∧ 
  (angle_bisector_line B (triangle_side AP B) = P) ∧ 
  (angle_bisector_line C (triangle_side AQ C) = Q) :=
sorry

end exists_triangle_with_given_angle_bisectors_l469_469181


namespace inverse_ln_of_ln_eq_exp_add_one_l469_469998

noncomputable def inverse_ln (x : ℝ) : ℝ := 
  if x > 1 then ln (x - 1) else 0

theorem inverse_ln_of_ln_eq_exp_add_one (x : ℝ) (hx : x ∈ ℝ) : 
  inverse_ln x = e^x + 1 := by sorry

end inverse_ln_of_ln_eq_exp_add_one_l469_469998


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469386

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469386


namespace least_common_multiple_1_to_10_l469_469439

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469439


namespace least_common_multiple_of_first_ten_positive_integers_l469_469454

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469454


namespace no_integers_satisfy_l469_469046

theorem no_integers_satisfy (a b c d : ℤ) : ¬ (a^4 + b^4 + c^4 + 2016 = 10 * d) :=
sorry

end no_integers_satisfy_l469_469046


namespace points_for_victory_l469_469899

theorem points_for_victory (V : ℕ) :
  (∃ (played total_games : ℕ) (points_after_games : ℕ) (remaining_games : ℕ) (needed_points : ℕ) 
     (draw_points defeat_points : ℕ) (minimum_wins : ℕ), 
     played = 5 ∧
     total_games = 20 ∧ 
     points_after_games = 12 ∧
     remaining_games = total_games - played ∧
     needed_points = 40 - points_after_games ∧
     draw_points = 1 ∧
     defeat_points = 0 ∧
     minimum_wins = 7 ∧
     7 * V ≥ needed_points ∧
     remaining_games = total_games - played ∧
     needed_points = 28) → V = 4 :=
sorry

end points_for_victory_l469_469899


namespace least_common_multiple_of_first_ten_l469_469608

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469608


namespace least_divisible_1_to_10_l469_469529

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469529


namespace sum_midpoints_x_y_l469_469343

theorem sum_midpoints_x_y (a b c d e f : ℝ) :
  a + b + c = 15 →
  d + e + f = 9 →
  (∑ i in [a, b, c].combinations 2, i.sum / 2) = 15 ∧
  (∑ i in [d, e, f].combinations 2, i.sum / 2) = 9 :=
by
  sorry

end sum_midpoints_x_y_l469_469343


namespace find_radius_l469_469708

noncomputable def r : ℝ := 2
noncomputable def s : ℝ := 2 + 2 * Real.sqrt 2

axiom radius_condition : ∃ s : ℝ, 
  (2 * s)^2 + (2 * s)^2 = (4 + 2 * s)^2 ∧ s > 0

theorem find_radius : s = 2 + 2 * Real.sqrt 2 := by
  apply exists.elim radius_condition
  intros s h
  have h1 : (2 * s)^2 + (2 * s)^2 = (4 + 2 * s)^2 := h.1
  have h2 : s > 0 := h.2
  sorry

end find_radius_l469_469708


namespace least_common_multiple_first_ten_integers_l469_469576

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469576


namespace linear_equation_solution_l469_469197

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l469_469197


namespace least_common_multiple_first_ten_integers_l469_469568

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469568


namespace maddie_watched_138_on_monday_l469_469960

-- Define the constants and variables from the problem statement
def total_episodes : ℕ := 8
def minutes_per_episode : ℕ := 44
def watched_thursday : ℕ := 21
def watched_friday_episodes : ℕ := 2
def watched_weekend : ℕ := 105

-- Calculate the total minutes watched from all episodes
def total_minutes : ℕ := total_episodes * minutes_per_episode

-- Calculate the minutes watched on Friday
def watched_friday : ℕ := watched_friday_episodes * minutes_per_episode

-- Calculate the total minutes watched on weekdays excluding Monday
def watched_other_days : ℕ := watched_thursday + watched_friday + watched_weekend

-- Statement to prove that Maddie watched 138 minutes on Monday
def minutes_watched_on_monday : ℕ := total_minutes - watched_other_days

-- The final statement for proof in Lean 4
theorem maddie_watched_138_on_monday : minutes_watched_on_monday = 138 := by
  -- This theorem should be proved using the above definitions and calculations, proof skipped with sorry
  sorry

end maddie_watched_138_on_monday_l469_469960


namespace area_ratio_correct_l469_469211

variables (r_s : ℝ) (r_r : ℝ) 

-- Defining the relationship given in the conditions
def diameter_condition : Prop :=
  2 * r_r = (3 + Real.sqrt 2) * r_s

-- Defining the areas of circles r and s
def area_r (r_r : ℝ) : ℝ := Real.pi * r_r^2
def area_s (r_s : ℝ) : ℝ := Real.pi * r_s^2

-- Statement of the theorem
theorem area_ratio_correct (r_s r_r : ℝ) (h : diameter_condition r_s r_r) : 
  (area_r r_r) / (area_s r_s) * 100 = 487.1 :=
sorry

end area_ratio_correct_l469_469211


namespace least_common_multiple_first_ten_l469_469635

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469635


namespace lcm_first_ten_positive_integers_l469_469418

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469418


namespace find_n_l469_469852

noncomputable def tangent_line_problem (x0 : ℝ) (n : ℕ) : Prop :=
(x0 ∈ Set.Ioo (Real.sqrt n) (Real.sqrt (n + 1))) ∧
(∃ m : ℝ, 0 < m ∧ m < 1 ∧ (2 * x0 = 1 / m) ∧ (x0^2 = (Real.log m - 1)))

theorem find_n (x0 : ℝ) (n : ℕ) :
  tangent_line_problem x0 n → n = 2 :=
sorry

end find_n_l469_469852


namespace smallest_k_l469_469830

theorem smallest_k (n : ℕ) (h_pos : n > 0) :
  ∃ k, (∀ (A ⊆ finset.range (2 * n + 1)), A.card = k → 
    ∃ a b c d ∈ A, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b + c + d = 4 * n + 1) 
    ∧ k = n + 3 :=
sorry

end smallest_k_l469_469830


namespace find_set_A_l469_469177

-- Define the set A based on the condition that its elements satisfy a quadratic equation.
def A (a : ℝ) : Set ℝ := {x | x^2 + 2 * x + a = 0}

-- Assume 1 is an element of set A
axiom one_in_A (a : ℝ) (h : 1 ∈ A a) : a = -3

-- The final theorem to prove: Given 1 ∈ A a, A a should be {-3, 1}
theorem find_set_A (a : ℝ) (h : 1 ∈ A a) : A a = {-3, 1} :=
by sorry

end find_set_A_l469_469177


namespace systematic_sampling_counts_l469_469685

-- 600 students numbered from 001 to 600
def num_students : ℕ := 600

-- Number of students to sample
def sample_size : ℕ := 50

-- First randomly drawn number
def first_drawn_num : ℕ := 3

-- Interval for systematic sampling
def interval : ℕ := 12

-- Camps divisions
def camp1_range : set ℕ := {x | 1 ≤ x ∧ x ≤ 300}
def camp2_range : set ℕ := {x | 301 ≤ x ∧ x ≤ 495}
def camp3_range : set ℕ := {x | 496 ≤ x ∧ x ≤ 600}

-- Function to determine the camp of a student number
def camp (n : ℕ) : ℕ :=
  if 1 ≤ n ∧ n ≤ 300 then 1
  else if 301 ≤ n ∧ n ≤ 495 then 2
  else if 496 ≤ n ∧ n ≤ 600 then 3
  else 0
  
-- Function to generate the systematic sample
def systematic_sample (first : ℕ) (k : ℕ) (size : ℕ) : list ℕ :=
  (list.range size).map (λ i, first + i * k)

-- The sample based on the given first drawn number and interval
def sample : list ℕ := systematic_sample first_drawn_num interval sample_size

-- Function to count numbers in a given camp
def count_in_camp (camp_num : ℕ) (samp : list ℕ) : ℕ :=
  (samp.filter (λ n, camp n = camp_num)).length

-- Theorem to be proven
theorem systematic_sampling_counts :
  (count_in_camp 1 sample = 25) ∧ (count_in_camp 2 sample = 17) ∧ (count_in_camp 3 sample = 8) :=
by sorry

end systematic_sampling_counts_l469_469685


namespace probability_of_alternating_draws_l469_469700

theorem probability_of_alternating_draws :
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls in
  let successful_orders : ℕ := 2,
      total_arrangements : ℕ := Nat.choose total_balls white_balls,
      probability : ℚ := successful_orders / total_arrangements in
  probability = 1 / 126 :=
by
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls
  let successful_orders : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let probability : ℚ := successful_orders / total_arrangements
  sorry

end probability_of_alternating_draws_l469_469700


namespace sequence_is_constant_l469_469945

-- Define the sequence a_n and the partial sum S_k
variable (a : ℕ → ℝ)

-- The partial sum S_k is the sum of the first k terms of the sequence a_n
def S (k : ℕ) : ℝ := ∑ i in finset.range k, a i

-- The given condition that S_{k+1} + S_k = a_{k+1} for all natural numbers k
axiom sum_condition (k : ℕ) : S (k + 1) + S k = a (k + 1)

-- The theorem to prove that the sequence is constant, specifically that a_n = 0 for all n
theorem sequence_is_constant : ∀ n : ℕ, a n = 0 := 
by 
  sorry

end sequence_is_constant_l469_469945


namespace least_tax_l469_469746

def annualIncome : ℝ := 4500000

def rent : ℝ := 60000
def cosmetics : ℝ := 40000
def salaries : ℝ := 120000
def insurance : ℝ := 36000
def advertising : ℝ := 15000
def training : ℝ := 12000
def miscellaneous : ℝ := 20000

def monthlyExpenses : ℝ := rent + cosmetics + salaries + insurance + advertising + training + miscellaneous
def annualExpenses : ℝ := monthlyExpenses * 12
def expensesPaid (fraction : ℝ) : ℝ := fraction * annualExpenses

def taxableIncomeOSNO : ℝ := annualIncome - annualExpenses
def taxOSNO (rate : ℝ) : ℝ := taxableIncomeOSNO * rate

def taxUSN_Income (rate : ℝ) : ℝ := annualIncome * rate
def taxReduction (insurancePaid : ℝ) (maxReduction : ℝ) : ℝ := min (maxReduction * taxUSN_Income 0.06) insurancePaid
def finalTaxUSN_Income (insurPaid : ℝ) : ℝ := taxUSN_Income 0.06 - taxReduction insurPaid 0.50

def taxableIncomeUSN_Expenses (expenseFraction : ℝ) : ℝ := annualIncome - expensesPaid expenseFraction
def taxUSN_Expenses (rate : ℝ) : ℝ := taxableIncomeUSN_Expenses 0.45 * rate
def minTax (rate : ℝ) : ℝ := annualIncome * rate
def finalTaxUSN_Expenses (minRate : ℝ) (rate : ℝ) : ℝ := max (taxUSN_Expenses rate) (minTax minRate)

theorem least_tax {annualIncome annualExpenses : ℝ} :
  let annualExpenses := 12 * (rent + cosmetics + salaries + insurance + advertising + training + miscellaneous) in
  let taxOSNO := (annualIncome - annualExpenses) * 0.20 in
  let taxUSN_Income := annualIncome * 0.06 - min (0.50 * (annualIncome * 0.06)) (0.45 * insurance * 12) in
  let taxUSN_Expenses := max ((annualIncome - 0.45 * annualExpenses) * 0.15) (annualIncome * 0.01) in
  min taxOSNO (min taxUSN_Income taxUSN_Expenses) = 135000 :=
by
  have := annualIncome = 4500000
  have := rent = 720000
  have := cosmetics = 480000
  have := salaries = 1440000
  have := insurance = 432000
  have := advertising = 180000
  have := training = 144000
  have := miscellaneous = 240000
  sorry

end least_tax_l469_469746


namespace max_rank_awarded_l469_469683

theorem max_rank_awarded (num_participants rank_threshold total_possible_points : ℕ)
  (H1 : num_participants = 30)
  (H2 : rank_threshold = (30 * 29 / 2 : ℚ) * 0.60)
  (H3 : total_possible_points = (30 * 29 / 2)) :
  ∃ max_awarded : ℕ, max_awarded ≤ 23 :=
by {
  -- Proof omitted
  sorry
}

end max_rank_awarded_l469_469683


namespace dog_food_weighs_more_l469_469283

def weight_in_ounces (weight_in_pounds: ℕ) := weight_in_pounds * 16
def total_food_weight (cat_food_bags dog_food_bags: ℕ) (cat_food_pounds dog_food_pounds: ℕ) :=
  (cat_food_bags * weight_in_ounces cat_food_pounds) + (dog_food_bags * weight_in_ounces dog_food_pounds)

theorem dog_food_weighs_more
  (cat_food_bags: ℕ) (cat_food_pounds: ℕ) (dog_food_bags: ℕ) (total_weight_ounces: ℕ) (ounces_in_pound: ℕ)
  (H1: cat_food_bags * weight_in_ounces cat_food_pounds = 96)
  (H2: total_food_weight cat_food_bags dog_food_bags cat_food_pounds dog_food_pounds = total_weight_ounces)
  (H3: ounces_in_pound = 16) :
  dog_food_pounds - cat_food_pounds = 2 := 
by sorry

end dog_food_weighs_more_l469_469283


namespace min_distinct_values_l469_469009

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) 
  (h_mode : mode_count = 10) (h_total : total_count = 2018) 
  (h_distinct : ∀ k, k ≠ mode_count → k < 10) : 
  n ≥ 225 :=
by
  sorry

end min_distinct_values_l469_469009


namespace lcm_first_ten_integers_l469_469620

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469620


namespace Q_lies_in_third_quadrant_l469_469883

theorem Q_lies_in_third_quadrant (b : ℝ) (P_in_fourth_quadrant : 2 > 0 ∧ b < 0) :
    b < 0 ∧ -2 < 0 ↔
    (b < 0 ∧ -2 < 0) :=
by
  sorry

end Q_lies_in_third_quadrant_l469_469883


namespace least_common_multiple_of_first_10_integers_l469_469514

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469514


namespace alternating_sequence_probability_l469_469697

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l469_469697


namespace carry_count_l469_469742

-- Define the range of numbers we are considering
def num_range := {n : ℕ | 1 ≤ n ∧ n ≤ 9999}

-- Define the constant 1993
def constant := 1993

-- Define a function to check if there's a carry operation when adding two numbers
def has_carry (a b : ℕ) : Prop :=
  ∃ i, (a / 10^i) % 10 + (b / 10^i) % 10 ≥ 10

-- State the theorem
theorem carry_count :
  (num_range.filter (λ n, has_carry n constant)).card = 9937 :=
sorry

end carry_count_l469_469742


namespace marys_downhill_speed_l469_469925

-- Context: Define the distance between Mary's home and school
def distance : ℝ := 1.5

-- Context: Define the time Mary takes for the downhill walk in hours
def time_downhill : ℝ := 15 / 60 -- Convert 15 minutes to hours

-- Theorem statement: Calculate her speed for the downhill walk
theorem marys_downhill_speed : (distance / time_downhill = 6) :=
by
  -- This is just a placeholder for the actual proof
  -- The proof can be added here
  sorry

end marys_downhill_speed_l469_469925


namespace cesaro_sum_100_l469_469804

def Cesaro_sum {n : ℕ} (a : Fin n → ℝ) : ℝ :=
  (Finset.univ.sum (λ k, (Finset.range (k + 1)).sum a)) / n

theorem cesaro_sum_100 (a : Fin 99 → ℝ) (h : Cesaro_sum a = 1000) :
  Cesaro_sum (fun i => if i = 0 then 1 else a (i.pred)) = 991 :=
by
  sorry

end cesaro_sum_100_l469_469804


namespace find_x_l469_469111

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 12) : x = 27 :=
begin
  sorry
end

end find_x_l469_469111


namespace total_junk_mail_l469_469013

-- Definitions for conditions
def houses_per_block : Nat := 17
def pieces_per_house : Nat := 4
def blocks : Nat := 16

-- Theorem stating that the mailman gives out 1088 pieces of junk mail in total
theorem total_junk_mail : houses_per_block * pieces_per_house * blocks = 1088 := by
  sorry

end total_junk_mail_l469_469013


namespace passes_through_point_l469_469996

def f (a x : ℝ) : ℝ := a^(x - 1) + 2

theorem passes_through_point {a : ℝ} (h₁ : 0 < a) (h₂ : a ≠ 1) : f a 1 = 3 :=
by {
  unfold f,
  rw [sub_self, pow_zero],
  norm_num,
}

end passes_through_point_l469_469996


namespace question1_question2_question3_question4_l469_469748

theorem question1 : (2 * 3) ^ 2 = 2 ^ 2 * 3 ^ 2 := by admit

theorem question2 : (-1 / 2 * 2) ^ 3 = (-1 / 2) ^ 3 * 2 ^ 3 := by admit

theorem question3 : (3 / 2) ^ 2019 * (-2 / 3) ^ 2019 = -1 := by admit

theorem question4 (a b : ℝ) (n : ℕ) (h : 0 < n): (a * b) ^ n = a ^ n * b ^ n := by admit

end question1_question2_question3_question4_l469_469748


namespace triangle_type_l469_469898

theorem triangle_type (a b c : ℝ) (A B C : ℝ) (h1 : A = 30) (h2 : a = 2 * b ∨ b = 2 * c ∨ c = 2 * a) :
  (C > 90 ∨ B > 90) ∨ C = 90 :=
sorry

end triangle_type_l469_469898


namespace least_common_multiple_of_first_10_integers_l469_469525

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469525


namespace triangulation_methods_l469_469131

noncomputable def catalan (n : ℕ) : ℕ :=
  if n = 0 then 1 
  else (2 * (2 * n - 1) * catalan (n - 1)) / (n + 1)

theorem triangulation_methods (n : ℕ) (h : n ≥ 2) :
  let A := convex_polygon (n + 1)
  let h_n := number_of_triangulations A
  h_n = catalan n :=
sorry

end triangulation_methods_l469_469131


namespace exists_triangle_in_M_no_triangle_in_M_l469_469911

open Classical

variable {A : Type} [LinearOrder A]
variable {n : ℕ} (n_gt_1 : 1 < n)
variable (points : Fin 2n → A)
variable (no_three_collinear : ∀ (i j k : Fin 2n), i ≠ j → j ≠ k → k ≠ i → ¬Collinear (insert 0 (finset.image points {i, j, k})))
variable (M : Finset (Fin 2n × Fin 2n)) (size_M : M.card = n^2 + 1)

theorem exists_triangle_in_M (α : A) : 
  ∃ r s t : Fin 2n, r ≠ s ∧ s ≠ t ∧ t ≠ r ∧ (r, s) ∈ M ∧ (s, t) ∈ M ∧ (t, r) ∈ M := sorry

variable (M' : Finset (Fin 2n × Fin 2n)) (size_M'_le_n2 : M'.card ≤ n^2)

theorem no_triangle_in_M'_possible (α : A) :
  ¬ ∃ r s t : Fin 2n, r ≠ s ∧ s ≠ t ∧ t ≠ r ∧ (r, s) ∈ M' ∧ (s, t) ∈ M' ∧ (t, r) ∈ M' := sorry

end exists_triangle_in_M_no_triangle_in_M_l469_469911


namespace least_common_multiple_1_to_10_l469_469497

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469497


namespace A_eq_A_swap_l469_469806

def sequence_property (n m : ℕ) (seq : ℕ → ℕ) : Prop :=
  (∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (finset.univ.card (filter (λ x, seq x = k) (range (n * m)))) = m) ∧
  (∀ i j k : ℕ, 1 ≤ i ∧ i ≤ n * m ∧ 1 ≤ j ∧ j ≤ k ∧ k ≤ n → 
    (finset.univ.card (filter (λ x, seq x = j) (range i))) ≥ 
    (finset.univ.card (filter (λ x, seq x = k) (range i))))

def A (n m : ℕ) : ℕ :=
  finset.card {seq // sequence_property n m seq}

theorem A_eq_A_swap (n m : ℕ) (h1 : n ≥ 1) (h2 : m ≥ 1) : A n m = A m n :=
  sorry

end A_eq_A_swap_l469_469806


namespace lcm_first_ten_l469_469406

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469406


namespace least_common_multiple_first_ten_l469_469641

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469641


namespace circle_through_points_and_intercepts_l469_469087

noncomputable def circle_eq (x y D E F : ℝ) : ℝ := x^2 + y^2 + D * x + E * y + F

theorem circle_through_points_and_intercepts :
  ∃ (D E F : ℝ), 
    circle_eq 4 2 D E F = 0 ∧
    circle_eq (-1) 3 D E F = 0 ∧ 
    D + E = -2 ∧
    circle_eq x y (-2) 0 (-12) = 0 :=
by
  unfold circle_eq
  sorry

end circle_through_points_and_intercepts_l469_469087


namespace least_divisible_1_to_10_l469_469541

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469541


namespace circles_externally_tangent_l469_469859

theorem circles_externally_tangent:
  let C1_center := (3 : ℝ, 0 : ℝ) in
  let C2_center := (0 : ℝ, -4 : ℝ) in
  let C1_radius := 1 in
  let C2_radius := 4 in
  dist C1_center C2_center = C1_radius + C2_radius :=
by
  sorry

end circles_externally_tangent_l469_469859


namespace solve_log_equation_l469_469985

theorem solve_log_equation (x : ℝ) (h1 : x > 0) :
  (3 / log x / log 2 = 4 * x - 5) ↔ (x = 2 ∨ x = 1/2) :=
by
  sorry

end solve_log_equation_l469_469985


namespace find_a3_l469_469831

-- Given conditions
def sequence_sum (S : ℕ → ℕ) : Prop :=
  ∀ n, S n = n^2 + n

-- Define the sequence term calculation from the sum function.
def seq_term (S : ℕ → ℕ) (n : ℕ) : ℕ :=
  S n - S (n - 1)

theorem find_a3 (S : ℕ → ℕ) (h : sequence_sum S) :
  seq_term S 3 = 6 :=
by
  sorry

end find_a3_l469_469831


namespace least_common_multiple_first_ten_integers_l469_469564

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469564


namespace gcd_1248_1001_l469_469792

theorem gcd_1248_1001 : Nat.gcd 1248 1001 = 13 := by
  sorry

end gcd_1248_1001_l469_469792


namespace equal_distances_from_chord_to_center_l469_469233

theorem equal_distances_from_chord_to_center (circle1 circle2 : Circle) (h_congruent : circle1 ≡ circle2)
  (θ₁ θ₂ : CentralAngle)
  (h_equal_angles : θ₁ ≡ θ₂)
  (arc₁ arc₂ : Arc)
  (h_equal_arcs : arc₁ ≡ arc₂)
  (chord₁ chord₂ : Chord)
  (h_equal_chords : chord₁ ≡ chord₂) :
  distance_from_chord_to_center chord₁ = distance_from_chord_to_center chord₂ :=
sorry

end equal_distances_from_chord_to_center_l469_469233


namespace least_positive_integer_divisible_by_first_ten_l469_469472

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469472


namespace least_common_multiple_first_ten_l469_469648

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469648


namespace separate_colors_by_line_l469_469292

theorem separate_colors_by_line
  (brown_points green_points : Finset (ℝ × ℝ))
  (h_brown_card : brown_points.card = 101)
  (h_green_card : green_points.card = 101)
  (h_no_collinear : ∀ p1 p2 p3, p1 ∈ brown_points ∨ p1 ∈ green_points → p2 ∈ brown_points ∨ p2 ∈ green_points → p3 ∈ brown_points ∨ p3 ∈ green_points → 
    (p1.1 ≠ p2.1 ∧ p1.1 ≠ p3.1 ∧ p2.1 ≠ p3.1 → ¬ collinear p1 p2 p3))
  (h_sum_brown : (Finset.card brown_points).choose 2 = 5050 ∧ (Finset.sum (Finset.pairwise_disjoint brown_points (λ x y, dist x y))) = 1)
  (h_sum_green : (Finset.card green_points).choose 2 = 5050 ∧ (Finset.sum (Finset.pairwise_disjoint green_points (λ x y, dist x y))) = 1)
  (h_sum_mixed : (Finset.card (brown_points × green_points)) = 10201 ∧ (Finset.sum (brown_points ×ˢ green_points) (λ ⟨b, g⟩, dist b g)) = 400) :
  ∃ l : ℝ × ℝ → Prop, ∀ p ∈ brown_points, ∀ q ∈ green_points, l p ∧ ¬ l q :=
sorry

end separate_colors_by_line_l469_469292


namespace least_perimeter_tr_l469_469892

noncomputable theory

variables {D E F : ℝ}
variables {d e f : ℕ}

def is_triangle (a b c : ℝ) : Prop := a + b > c ∧ b + c > a ∧ c + a > b

def cos_D : ℝ := 9 / 16
def cos_E : ℝ := 12 / 13
def cos_F : ℝ := -1 / 3

-- Calculate sine values from cosine values using sin²(x) + cos²(x) = 1
def sin_D : ℝ := real.sqrt (1 - cos_D^2)
def sin_E : ℝ := real.sqrt (1 - cos_E^2)
def sin_F : ℝ := real.sqrt (1 - cos_F^2)

-- Define the ratio of sides based on the Law of Sines
def ratio_d : ℝ := sin_D
def ratio_e : ℝ := sin_E
def ratio_f : ℝ := sin_F

-- Define a function that calculates the perimeter given the side lengths
def triangle_perimeter (d e f : ℕ) : ℕ := d + e + f

-- Actual Lean statement to prove
theorem least_perimeter_tr DEF_is_triangle : 
  (d, e, f : ℕ) → D = d ∧ E = e ∧ F = f ∧ 
  is_triangle D E F ∧ 
  real.cos D = cos_D ∧
  real.cos E = cos_E ∧
  real.cos F = cos_F →
  triangle_perimeter d e f = 30 := by
  sorry

end least_perimeter_tr_l469_469892


namespace max_value_f_for_x_neg_l469_469877

noncomputable def f : ℝ → ℝ := λ x, (12 / x) + 3 * x

theorem max_value_f_for_x_neg (x : ℝ) (h : x < 0) : 
  ∃ k, (∀ y, (y < 0 → f y ≤ k)) ∧ k = -12 := 
sorry

end max_value_f_for_x_neg_l469_469877


namespace polynomial_addition_l469_469032

variable (x : ℝ)

def p := 3 * x^4 + 2 * x^3 - 5 * x^2 + 9 * x - 2
def q := -3 * x^4 - 5 * x^3 + 7 * x^2 - 9 * x + 4

theorem polynomial_addition : p x + q x = -3 * x^3 + 2 * x^2 + 2 := by
  sorry

end polynomial_addition_l469_469032


namespace least_common_multiple_of_first_10_integers_l469_469519

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469519


namespace problem_inequality_l469_469143

theorem problem_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (x - y + z) * (y - z + x) * (z - x + y) ≤ x * y * z := sorry

end problem_inequality_l469_469143


namespace operation_8_to_cube_root_16_l469_469655

theorem operation_8_to_cube_root_16 : ∃ (x : ℕ), x = 8 ∧ (x * x = (Nat.sqrt 16)^3) :=
by
  sorry

end operation_8_to_cube_root_16_l469_469655


namespace rate_per_sqm_is_correct_l469_469324

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end rate_per_sqm_is_correct_l469_469324


namespace least_common_multiple_first_ten_l469_469551

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469551


namespace gcd_20020_11011_l469_469793

theorem gcd_20020_11011 : Nat.gcd 20020 11011 = 1001 := 
by
  sorry

end gcd_20020_11011_l469_469793


namespace least_common_multiple_1_to_10_l469_469438

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469438


namespace least_divisible_1_to_10_l469_469533

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469533


namespace mark_more_hours_l469_469959

-- Definitions based on the conditions
variables (Pat Kate Mark Alex : ℝ)
variables (total_hours : ℝ)
variables (h1 : Pat + Kate + Mark + Alex = 350)
variables (h2 : Pat = 2 * Kate)
variables (h3 : Pat = (1 / 3) * Mark)
variables (h4 : Alex = 1.5 * Kate)

-- Theorem statement with the desired proof target
theorem mark_more_hours (Pat Kate Mark Alex : ℝ) (h1 : Pat + Kate + Mark + Alex = 350) 
(h2 : Pat = 2 * Kate) (h3 : Pat = (1 / 3) * Mark) (h4 : Alex = 1.5 * Kate) : 
Mark - (Kate + Alex) = 116.66666666666667 := sorry

end mark_more_hours_l469_469959


namespace least_common_multiple_1_to_10_l469_469442

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469442


namespace folded_triangle_is_isosceles_l469_469339

noncomputable def is_isosceles (a b c : ℝ) := a = b ∨ b = c ∨ c = a

def triangle_side_lengths (AB BC CA : ℝ) : Prop :=
  AB = 10 ∧ BC = 8 ∧ CA = 12

def folded_isosceles (A B C F G : ℝ × ℝ) : Prop :=
  let CG := 4 in
  let GF := 4 in
  let CF := 10 - 4 in
  is_isosceles CG GF CF

theorem folded_triangle_is_isosceles (A B C F G : ℝ × ℝ) (h1 : triangle_side_lengths 10 8 12) (h2 : folded_isosceles A B C F G) :
  is_isosceles (4 : ℝ) (4 : ℝ) (6 : ℝ) :=
  sorry

end folded_triangle_is_isosceles_l469_469339


namespace log_base_range_l469_469872

theorem log_base_range (a : ℝ) :
  (∃ b : ℝ, b = log (a - 2) (5 - a) ∧ a > 2 ∧ a ≠ 3 ∧ a < 5) ↔ (a ∈ set.Ioo 2 3 ∨ a ∈ set.Ioo 3 5) :=
by sorry

end log_base_range_l469_469872


namespace complex_add_inv_real_iff_norm_one_l469_469127

-- Definitions for the given complex number z = a + bi and condition b ≠ 0
variables {a b : ℝ} (h : b ≠ 0)
def z : ℂ := complex.mk a b
def one_over_z : ℂ := complex.inv z

-- The main theorem to be proven
theorem complex_add_inv_real_iff_norm_one : (z + one_over_z).im = 0 ↔ complex.norm z = 1 :=
by sorry

end complex_add_inv_real_iff_norm_one_l469_469127


namespace mean_age_of_seven_friends_l469_469900

theorem mean_age_of_seven_friends 
  (mean_age_group1: ℕ)
  (mean_age_group2: ℕ)
  (n1: ℕ)
  (n2: ℕ)
  (total_friends: ℕ) :
  mean_age_group1 = 147 → 
  mean_age_group2 = 161 →
  n1 = 3 → 
  n2 = 4 →
  total_friends = 7 →
  (mean_age_group1 * n1 + mean_age_group2 * n2) / total_friends = 155 := by
  sorry

end mean_age_of_seven_friends_l469_469900


namespace loss_percentage_correct_l469_469711

noncomputable def loss_percentage_problem 
  (orig_price : ℕ) 
  (discount_rate : ℚ)  
  (sales_tax_rate : ℚ) 
  (shipping_fee : ℕ) 
  (selling_price_before_deductions : ℕ) 
  (seller_commission_rate : ℚ) 
  (charity_donation_rate : ℚ) 
  : ℚ :=
let
  -- Calculating price after discount
  discount_amount := orig_price * discount_rate,
  price_after_discount := orig_price - discount_amount,
  -- Calculating price after adding sales tax
  sales_tax_amount := price_after_discount * sales_tax_rate,
  price_after_sales_tax := price_after_discount + sales_tax_amount,
  -- Total cost price including shipping fee
  total_cost_price := price_after_sales_tax + shipping_fee,
  -- Calculating total deductions
  seller_commission_amount := selling_price_before_deductions * seller_commission_rate,
  charity_donation_amount := selling_price_before_deductions * charity_donation_rate,
  total_deductions := seller_commission_amount + charity_donation_amount,
  -- Actual selling price after deductions
  actual_selling_price := selling_price_before_deductions - total_deductions,
  -- Calculating loss and loss percentage
  loss := total_cost_price - actual_selling_price,
  loss_percentage := (loss / total_cost_price) * 100
in
loss_percentage

theorem loss_percentage_correct :
  loss_percentage_problem 1500 0.05 0.10 300 1620 0.05 0.02 = 19.33 := 
sorry

end loss_percentage_correct_l469_469711


namespace root_in_interval_imp_range_m_l469_469157

theorem root_in_interval_imp_range_m (m : ℝ) (f : ℝ → ℝ) (h : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0) : 2 < m ∧ m < 4 :=
by
  have exists_x : ∃ x, (1 < x ∧ x < 2) ∧ f x = 0 := h
  sorry

end root_in_interval_imp_range_m_l469_469157


namespace lcm_first_ten_l469_469402

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469402


namespace binomial_coeff_divisibility_l469_469975

theorem binomial_coeff_divisibility (n k : ℕ) (hn : 0 < n) (hk : 0 < k) : n ∣ (Nat.choose n k) * Nat.gcd n k :=
sorry

end binomial_coeff_divisibility_l469_469975


namespace max_ounces_amber_can_get_l469_469035

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l469_469035


namespace least_common_multiple_1_to_10_l469_469504

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469504


namespace lcm_first_ten_positive_integers_l469_469420

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469420


namespace angle_equal_l469_469346

-- Definitions to set up the problem conditions
structure Triangle (α : Type) [MetricSpace α] :=
(A B C : α)

structure Midpoint (α : Type) [MetricSpace α] :=
(midpoint : α)
(proj_point : α)

structure TangentMeet (α : Type) [MetricSpace α] :=
(TangentPt : α)
(pt : α)

-- Assuming the given properties and definitions of points, midpoints, and tangents
variables {α : Type} [MetricSpace α]
variables (A B C T U P Q R S : α)
variables (ABC : Triangle α)
variables (MidAPQ : Midpoint α := ⟨Q, P⟩)
variables (MidBR : Midpoint α := ⟨S, R⟩)
variables (TanTangentsA : TangentMeet α := ⟨T, C⟩)
variables (TanTangentsB : TangentMeet α := ⟨U, C⟩)

-- The theorem statement
theorem angle_equal (ABC : Triangle α) [noncomputable_Lean4 A B C]
  (Q_is_midpoint : Midpoint α) (S_is_midpoint : Midpoint α)
  (AT_meets_BC_P : ∃ P : α, line_through ABC.A ABC.T ∧ line_through ABC.B ABC.C)
  (BU_meets_CA_R : ∃ R : α, line_through ABC.B ABC.U ∧ line_through ABC.C ABC.A) :
  angle ABC.A ABC.B Q_is_midpoint.midpoint = angle ABC.A S_is_midpoint.midpoint :=
sorry

end angle_equal_l469_469346


namespace cos_of_4_arcsin_one_fourth_l469_469762

theorem cos_of_4_arcsin_one_fourth : 
  cos (4 * arcsin (1 / 4)) = 17 / 32 :=
by
  sorry

end cos_of_4_arcsin_one_fourth_l469_469762


namespace probability_of_alternating_colors_l469_469702

-- The setup for our problem
variables (B W : Type) 

/-- A box contains 5 white balls and 5 black balls.
    Prove that the probability that all of my draws alternate colors
    is 1/126. -/
theorem probability_of_alternating_colors (W B : ℕ) (hw : W = 5) (hb : B = 5) :
  let total_ways := Nat.choose 10 5 in
  let successful_ways := 2 in
  successful_ways / total_ways = (1 : ℚ) / 126 :=
sorry

end probability_of_alternating_colors_l469_469702


namespace least_common_multiple_of_first_10_integers_l469_469524

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469524


namespace least_divisible_1_to_10_l469_469539

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469539


namespace least_positive_integer_divisible_by_first_ten_l469_469484

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469484


namespace least_common_multiple_first_ten_l469_469558

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469558


namespace digits_solution_l469_469874

theorem digits_solution (a b c d : ℕ) (h1 : 3 * (10 * c) * (10 * d + 4) = 100 * a + 10 * b + 8)
  (h2 : c ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) (h3 : d ∈ {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}) 
  : c + d = 5 := 
sorry

end digits_solution_l469_469874


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469393

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469393


namespace alternating_sequence_probability_l469_469696

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l469_469696


namespace general_formula_for_arithmetic_sequence_sum_Tn_inequality_l469_469161

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n + d

def right_triangle (a b c : ℤ) : Prop :=
  a * a + b * b = c * c

variable {a : ℕ → ℤ}

theorem general_formula_for_arithmetic_sequence 
  (h : arithmetic_sequence a 2)
  (h_right_triangle : right_triangle (a 2) (a 3) (a 4)) :
  a = λ n, 2 * n + 2 := sorry

def sum_Tn (a : ℕ → ℤ) (n : ℕ) : ℚ :=
  ∑ k in Finset.range n, 1 / (a k * a (k + 1))

theorem sum_Tn_inequality 
  (h : arithmetic_sequence a 2)
  (h_a_n : a = λ n, 2 * n + 2)
  (n : ℕ) :
  sum_Tn a n < 1 / 8 := sorry

end general_formula_for_arithmetic_sequence_sum_Tn_inequality_l469_469161


namespace least_common_multiple_first_ten_l469_469649

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469649


namespace least_common_multiple_first_ten_l469_469636

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469636


namespace least_common_multiple_1_to_10_l469_469447

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469447


namespace sin_cos_identity_l469_469188

theorem sin_cos_identity (a b θ : ℝ) (h : (sin θ) ^ 6 / a + (cos θ) ^ 6 / b = 1 / (a + b)) :
  (sin θ) ^ 12 / a ^ 5 + (cos θ) ^ 12 / b ^ 5 = 1 / (a + b) ^ 5 :=
sorry

end sin_cos_identity_l469_469188


namespace find_angle_C_find_perimeter_l469_469921

-- Definitions related to the triangle problem
variables {A B C : ℝ}
variables {a b c : ℝ} -- sides opposite to A, B, C

-- Condition: (2a - b) * cos C = c * cos B
def condition_1 (a b c C B : ℝ) : Prop := (2 * a - b) * Real.cos C = c * Real.cos B

-- Given C in radians (part 1: find angle C)
theorem find_angle_C 
  (A B C : ℝ) (a b c : ℝ) 
  (h1 : condition_1 a b c C B) 
  (H1 : 0 < C) (H2 : C < Real.pi) :
  C = Real.pi / 3 := 
sorry

-- More conditions for part 2
variables (area : ℝ) -- given area of triangle
def condition_2 (a b C area : ℝ) : Prop := 0.5 * a * b * Real.sin C = area

-- Given c = 2 and area = sqrt(3) (part 2: find perimeter)
theorem find_perimeter 
  (A B C : ℝ) (a b : ℝ) (c : ℝ) (area : ℝ) 
  (h2 : condition_2 a b C area) 
  (Hc : c = 2) (Harea : area = Real.sqrt 3) :
  a + b + c = 6 := 
sorry

end find_angle_C_find_perimeter_l469_469921


namespace solve_for_x_l469_469118

  theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → x = 27 :=
  by
    sorry
  
end solve_for_x_l469_469118


namespace lcm_first_ten_numbers_l469_469583

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469583


namespace fraction_div_subtract_l469_469363

theorem fraction_div_subtract : 
  (5 / 6 : ℚ) / (9 / 10) - (1 / 15) = 116 / 135 := 
by 
  sorry

end fraction_div_subtract_l469_469363


namespace not_all_congruent_l469_469707

theorem not_all_congruent (T : Type) [triangle : T] (smaller_triangles : fin 5 → T) :
  ¬ (∀ i j : fin 5, i ≠ j → smaller_triangles i ≅ smaller_triangles j) := 
sorry

end not_all_congruent_l469_469707


namespace least_common_multiple_of_first_ten_integers_l469_469364

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469364


namespace lcm_first_ten_integers_l469_469625

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469625


namespace solve_for_x_l469_469116

  theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → x = 27 :=
  by
    sorry
  
end solve_for_x_l469_469116


namespace lcm_first_ten_positive_integers_l469_469419

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469419


namespace tiling_impossible_l469_469967

theorem tiling_impossible (board : Fin 6 × Fin 6 → Fin 2)
    (condition : ∀ i, ∃ j, board i j = 1 ∧ ∀ j₁ j₂, board i j₁ = 1 → board i j₂ = 1 → j₁ = j₂):
    ¬(∃ tiling : Fin 6 × Fin 6 → Fin 2, 
        (∀ tile : Fin 6 × Fin 6, tiling tile = 2) ∧ 
        (∃ cover : Fin 6 × Fin 6 → (Fin 6 × Fin 6) × (Fin 6 × Fin 6), 
            (∀ place : (Fin 6 × Fin 6) × (Fin 6 × Fin 6), 
                cover place = 1))) :=
by
    sorry

end tiling_impossible_l469_469967


namespace arithmetic_mean_a_X_l469_469278

def M : Finset ℕ := Finset.range 1000

def a_X (X : Finset ℕ) : ℕ :=
  X.min' (Finset.nonempty_of_ne_empty (X.ne_empty_of_mem (Finset.min'_mem X))) +
  X.max' (Finset.nonempty_of_ne_empty (X.ne_empty_of_mem (Finset.max'_mem X)))

theorem arithmetic_mean_a_X :
  let n := 1000 in
  let non_empty_subsets := Finset.powerset M \ {∅} in
  let sum_a_X := ∑ X in non_empty_subsets, a_X X in
  let count := non_empty_subsets.card in
  (sum_a_X / count : ℤ) = 1001 :=
sorry

end arithmetic_mean_a_X_l469_469278


namespace equal_segments_and_parallel_lines_l469_469971

theorem equal_segments_and_parallel_lines
  (circle : Type*)
  [metric_space circle]
  [nonempty circle]
  (O : circle) (A B C D T1 T2 : circle)
  (hAB : A ≠ B) (hC_ext_AB : collinear (finset.of_list [A, B, C]))
  (hD_ext_AB : collinear (finset.of_list [A, B, D]))
  (hT1C : tangent (finset.of_list [C, T1]))
  (hT2D : tangent (finset.of_list [D, T2]))
  (hCT1_eq_DT2 : dist C T1 = dist D T2) :
  dist A C = dist B D ∧ line_parallel (line_through T1 T2) (line_through A B) :=
  sorry

end equal_segments_and_parallel_lines_l469_469971


namespace set_intersection_example_l469_469178

theorem set_intersection_example : 
  let A := {-1, 1, 2, 4}
  let B := {-1, 0, 2}
  A ∩ B = {-1, 2} :=
by
  sorry

end set_intersection_example_l469_469178


namespace problem_I_solution_set_l469_469262

def f1 (x : ℝ) : ℝ := |2 * x| + |x - 1| -- since a = -1

theorem problem_I_solution_set :
  {x : ℝ | f1 x ≤ 4} = Set.Icc (-1 : ℝ) ((5 : ℝ) / 3) :=
sorry

end problem_I_solution_set_l469_469262


namespace line_representation_l469_469824

variable {R : Type*} [Field R]
variable (f : R → R → R)
variable (x0 y0 : R)

def not_on_line (P : R × R) (f : R → R → R) : Prop :=
  f P.1 P.2 ≠ 0

theorem line_representation (P : R × R) (hP : not_on_line P f) :
  ∃ l : R → R → Prop, (∀ x y, l x y ↔ f x y - f P.1 P.2 = 0) ∧ (l P.1 P.2) ∧ 
  ∀ x y, f x y = 0 → ∃ n : R, ∀ x1 y1, (l x1 y1 → f x1 y1 = n * (f x y)) :=
sorry

end line_representation_l469_469824


namespace combination_12_choose_5_l469_469225

theorem combination_12_choose_5 :
  (12.choose 5) = 792 :=
by
  sorry

end combination_12_choose_5_l469_469225


namespace least_common_multiple_1_to_10_l469_469453

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469453


namespace simplify_expression_l469_469764

variable (a b : ℝ)

theorem simplify_expression : -3 * a * (2 * a - 4 * b + 2) + 6 * a = -6 * a ^ 2 + 12 * a * b := by
  sorry

end simplify_expression_l469_469764


namespace find_x_l469_469109

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 12) : x = 27 :=
begin
  sorry
end

end find_x_l469_469109


namespace least_common_multiple_1_to_10_l469_469506

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469506


namespace area_of_triangle_ABC_l469_469279

theorem area_of_triangle_ABC :
  ∃ (a b : ℝ), 
    (a ≠ 0 ∧ b ≠ 0) ∧ 
    let Cx := -(a + b),
        Cy := -(a + 3 * b),
        hypotenuse_condition := (a - b) ^ 2 + (a - 3 * b) ^ 2 = 2500,
        perpendicular_condition := 3 * (a + b) ^ 2 = -(a + 3 * b) ^ 2
    in hypotenuse_condition ∧ perpendicular_condition ∧ 
       let area := abs (a * b) / 2
    in area = 3750 / 59 := 
sorry

end area_of_triangle_ABC_l469_469279


namespace least_positive_integer_divisible_by_first_ten_l469_469482

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469482


namespace least_common_multiple_1_to_10_l469_469495

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469495


namespace lcm_first_ten_l469_469412

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469412


namespace finite_distinct_values_l469_469267

noncomputable def g (x : ℝ) : ℝ := 6 * x - x ^ 2

def sequence (x0 : ℝ) : ℕ → ℝ 
| 0 := x0
| (n+1) := g (sequence n)

theorem finite_distinct_values (x0 : ℝ) (h : 0 ≤ x0 ∧ x0 ≤ 6) : 
  ∃ N : ℕ, ∀ n m : ℕ, n ≥ N ∧ m ≥ N → sequence x0 n = sequence x0 m := by
  sorry

end finite_distinct_values_l469_469267


namespace min_value_a_condition_l469_469882

theorem min_value_a_condition (a : ℝ) : 
  (∀ θ ∈ set.Icc (0 : ℝ) (Real.pi / 2), 4 + 2 * Real.sin θ * Real.cos θ - a * Real.sin θ - a * Real.cos θ ≤ 0) → a ≥ 4 := sorry

end min_value_a_condition_l469_469882


namespace equation_of_tangent_circle_l469_469994

-- Define the center of the circle
def center : ℝ × ℝ := (2, 1)

-- Define the line that the circle is tangent to
def tangent_line (x : ℝ) : ℝ := x = 0

-- Define the distance from the center to the tangent line
def distance_from_center_to_line (c : ℝ × ℝ) (l : ℝ → Prop) : ℝ :=
  abs c.1

-- State the theorem to be proved
theorem equation_of_tangent_circle : 
  distance_from_center_to_line center tangent_line = 2 →
  ∃ R, (x y : ℝ), (x - 2)^2 + (y - 1)^2 = R ∧ R = 4 :=
by
  intros h
  use 4
  sorry

end equation_of_tangent_circle_l469_469994


namespace find_expression_l469_469259

def B : ℂ := 3 + 2 * Complex.I
def Q : ℂ := -5 * Complex.I
def R : ℂ := 1 + Complex.I
def T : ℂ := 3 - 4 * Complex.I

theorem find_expression : B * R + Q + T = 4 + Complex.I := by
  sorry

end find_expression_l469_469259


namespace fixed_cost_is_50000_l469_469724

-- Definition of conditions
def fixed_cost : ℕ := 50000
def books_sold : ℕ := 10000
def revenue_per_book : ℕ := 9 - 4

-- Theorem statement: Proving that the fixed cost of making books is $50,000
theorem fixed_cost_is_50000 (F : ℕ) (h : revenue_per_book * books_sold = F) : 
  F = fixed_cost :=
by sorry

end fixed_cost_is_50000_l469_469724


namespace non_intersecting_segments_exists_l469_469219

theorem non_intersecting_segments_exists :
  ∀ (boys girls : ℕ), 
  (boys = 10) ∧ (girls = 10) →
  ∃ (segments : list (ℕ × ℕ)), 
    (segments.length = 10) ∧ 
    (∀ (i j : ℕ), i ≠ j → 
     ∀ (s1 s2 : (ℕ × ℕ)), 
     s1 ∈ segments → 
     s2 ∈ segments → 
     non_intersecting s1 s2) :=
begin
  sorry
end

-- Definitions/Assumptions for non_intersecting (this would ideally be part of the broader context)
def non_intersecting : (ℕ × ℕ) → (ℕ × ℕ) → Prop :=
-- A dummy definition, replace with actual logic
λ s1 s2, true -- Replace with actual non-intersecting logic

end non_intersecting_segments_exists_l469_469219


namespace claire_photos_l469_469280

theorem claire_photos (C : ℕ) (h1 : 3 * C = C + 20) : C = 10 :=
sorry

end claire_photos_l469_469280


namespace equilateral_centers_form_equilateral_and_centroid_coincides_l469_469781

theorem equilateral_centers_form_equilateral_and_centroid_coincides
  (A B C : Type)
  [metric_space A] [metric_space B] [metric_space C]
  {triangle_ABC : triangle A B C}
  {equilateral_ABK : equilateral_triangle A B}
  {equilateral_ACL : equilateral_triangle A C}
  {equilateral_BCM : equilateral_triangle B C} :
  let O_A := center equilateral_ABK,
  let O_B := center equilateral_ACL,
  let O_C := center equilateral_BCM,
  centroid (triangle O_A O_B O_C) = centroid (triangle_ABC) ∧
  is_equilateral (triangle O_A O_B O_C) :=
sorry

end equilateral_centers_form_equilateral_and_centroid_coincides_l469_469781


namespace least_common_multiple_first_ten_l469_469553

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469553


namespace part1_part2_l469_469130

noncomputable def f (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  ln (a * x + b) + x^2

noncomputable def f' (x : ℝ) (a : ℝ) (b : ℝ) : ℝ :=
  (a / (a * x + b)) + 2 * x

theorem part1 (a b : ℝ) (ha : a ≠ 0) (h_tangent : (∃ y, y = f 1 a b ∧ ∀ (x : ℝ), y = x → f'(1) a b = 1))
  : a = -1 ∧ b = 2 := 
  sorry

theorem part2 (a b : ℝ) (ha : a > 0) (h_le : (∀ x : ℝ, f x a b ≤ x^2 + x)) 
  : a * b ≤ exp 1 / 2 := 
  sorry

end part1_part2_l469_469130


namespace least_divisible_1_to_10_l469_469542

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469542


namespace smallest_period_and_interval_area_of_triangle_l469_469844

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin x * Real.cos x + 2 * Real.sqrt 3 * Real.cos x ^ 2 - Real.sqrt 3

-- Define the function and the question about the period and interval
theorem smallest_period_and_interval (k : ℤ) :
  (∃ (T : ℝ), ∀ x, f (x + T) = f x ∧ T = π) ∧
  (∀ k : ℤ, ∃ t : ℝ, t ∈ set.Icc (k * π + π / 12) (k * π + 7 * π / 12) → ∀ x ∈ set.Icc (k * π + π / 12) (k * π + 7 * π / 12), f x ≤ f t) :=
sorry

-- Define sides and angles for the triangle, and the required area calculation
theorem area_of_triangle (a b c : ℝ) (A B C : ℝ) (R : ℝ) (sin_B_C : ℝ)
  (h1 : a = 7)
  (h2 : f ((A / 2) - (π / 6)) = Real.sqrt 3)
  (h3 : Real.sin B + Real.sin C = 13 * Real.sqrt 3 / 14)
  (h4 : A = π / 3)
  (h5 : 2 * R = a / Real.sin A)
  (h6 : b + c = 13)
  (h7 : b * c = 40) :
  1 / 2 * b * c * Real.sin A = 10 * Real.sqrt 3 :=
sorry

end smallest_period_and_interval_area_of_triangle_l469_469844


namespace area_RZX_eq_24_l469_469730

-- Define the square WXYZ and its properties
variables {W X Y Z P Q R : Point}
variable (area_WXYZ : ℝ)
variable (WXYZ_square : is_square W XYZ)
variable (area_YPRQ : ℝ)

-- Conditions
def conditions : Prop :=
  WXYZ_square ∧
  area_WXYZ = 144 ∧
  (position P on_segment Y Z such_that distance Y P = 4 ∧ distance P Z = 8) ∧
  (midpoint Q on_segment W P) ∧
  (midpoint R on_segment X P) ∧
  area_YPRQ = 30

-- Question: Prove the area of triangle RZX is 24 units
theorem area_RZX_eq_24 :
  conditions →
  triangle_area R Z X = 24 :=
sorry

end area_RZX_eq_24_l469_469730


namespace find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469102

theorem find_x_such_that_sqrt_5x_plus_9_eq_12 : ∀ x : ℝ, sqrt (5 * x + 9) = 12 → x = 27 := 
by
  intro x
  sorry

end find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469102


namespace least_common_multiple_first_ten_l469_469637

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469637


namespace angle_B_less_than_pi_div_two_l469_469891

variable {a b c A B C : ℝ}
variable {ABC : Triangle ℝ}

-- Definition: reciprocals of the three sides form a geometric sequence
def reciprocals_form_geometric_sequence (a b c : ℝ) : Prop :=
  (1 / b)^(2 : ℝ) = (1 / (a * c))

-- The statement to be proved
theorem angle_B_less_than_pi_div_two 
  (h1 : Triangle ABC)
  (h2 : reciprocals_form_geometric_sequence a b c) :
  B < π / 2 :=
sorry

end angle_B_less_than_pi_div_two_l469_469891


namespace total_dinners_sold_203_l469_469287

def monday_dinners : ℕ := 40
def tuesday_dinners : ℕ := monday_dinners + 40
def wednesday_dinners : ℕ := tuesday_dinners / 2
def thursday_dinners : ℕ := wednesday_dinners + 3

def total_dinners_sold : ℕ := monday_dinners + tuesday_dinners + wednesday_dinners + thursday_dinners

theorem total_dinners_sold_203 : total_dinners_sold = 203 := by
  sorry

end total_dinners_sold_203_l469_469287


namespace polynomial_at_3_l469_469944

-- Define the polynomial P(x) with constraints
def polynomial (x : ℤ) := ∀ (b : ℕ → ℤ) (n : ℕ), 
  (∀ i, i ≤ n → 0 ≤ b i ∧ b i < 5) ∧
  (∑ i in range (n+1), b i * x^i) 

noncomputable def P (x : ℤ) := 3 + x^2

theorem polynomial_at_3 : polynomial P → P 3 = 12 := by {
  sorry
}

end polynomial_at_3_l469_469944


namespace inequality_solution_set_l469_469823

noncomputable def f (x : ℝ) : ℝ := sorry

theorem inequality_solution_set (f : ℝ → ℝ)
  (h_diff : ∀ x, differentiable_at ℝ f x)
  (h_inequality : ∀ x, f x > f' x + 2)
  (h_odd : ∀ x, f x - 2019 = -(f (-x) - 2019)) :
  {x : ℝ | f x - 2017 * real.exp x < 2} = set.Ioi 0 :=
begin
  sorry
end

end inequality_solution_set_l469_469823


namespace least_common_multiple_1_to_10_l469_469452

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469452


namespace floor_sqrt_27_squared_eq_25_l469_469079

theorem floor_sqrt_27_squared_eq_25 :
  (⌊Real.sqrt 27⌋)^2 = 25 :=
by
  have H1 : 5 < Real.sqrt 27 := sorry
  have H2 : Real.sqrt 27 < 6 := sorry
  have floor_sqrt_27_eq_5 : ⌊Real.sqrt 27⌋ = 5 := sorry
  rw floor_sqrt_27_eq_5
  norm_num

end floor_sqrt_27_squared_eq_25_l469_469079


namespace M_subset_N_l469_469851

variable (f g : ℝ → ℝ) (a : ℝ)

def M : Set ℝ := {x | abs (f x) + abs (g x) < a}
def N : Set ℝ := {x | abs (f x + g x) < a}

theorem M_subset_N (h : a > 0) : M f g a ⊆ N f g a := by
  sorry

end M_subset_N_l469_469851


namespace percentage_change_approximately_47_l469_469962

noncomputable def total_price (prices : List ℝ) : ℝ :=
  List.sum prices

noncomputable def total_expense (prices : List ℝ) (tip : ℝ) : ℝ :=
  total_price prices + tip

noncomputable def change (payment : ℝ) (expense : ℝ) : ℝ :=
  payment - expense

noncomputable def percentage_of_change (change : ℝ) (payment : ℝ) : ℝ :=
  (change / payment) * 100

theorem percentage_change_approximately_47 :
  let prices := [8.99, 5.99, 3.59, 2.99, 1.49, 0.99]
  let tip := 2
  let payment := 50
  let totalExpense := total_expense prices tip
  let changeReceived := change payment totalExpense
  let changePercentage := percentage_of_change changeReceived payment
  changePercentage ≈ 47 :=
by
  sorry

end percentage_change_approximately_47_l469_469962


namespace initial_percentage_of_alcohol_l469_469715

variable (P : ℝ)
variables (x y : ℝ) (initial_percent replacement_percent replaced_quantity final_percent : ℝ)

def whisky_problem :=
  initial_percent = P ∧
  replacement_percent = 0.19 ∧
  replaced_quantity = 2/3 ∧
  final_percent = 0.26 ∧
  (P * (1 - replaced_quantity) + replacement_percent * replaced_quantity = final_percent)

theorem initial_percentage_of_alcohol :
  whisky_problem P 0.40 0.19 (2/3) 0.26 := sorry

end initial_percentage_of_alcohol_l469_469715


namespace irreducible_fractions_in_interval_l469_469062

theorem irreducible_fractions_in_interval (n : ℕ) (h_pos : 0 < n) (I : set ℝ)
  (hI_len : ∃ a b : ℝ, a < b ∧ b - a = 1 / n)
  (F_n := {x : ℚ | ∃ (p q : ℤ), p ≠ 0 ∧ q ≠ 0 ∧ gcd (p.nat_abs) (q.nat_abs) = 1 ∧ 1 ≤ q ∧ (q : ℕ) ≤ n ∧ x = p / q}) :
  F_n.countable ∧ F_n ∩ I).card ≤ (n + 1) / 2 := 
sorry

end irreducible_fractions_in_interval_l469_469062


namespace lcm_first_ten_positive_integers_l469_469434

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469434


namespace flag_designs_l469_469066

noncomputable def num_possible_flags : ℕ := 108

theorem flag_designs (colors : fin 3) (horizontal_stripes : fin 3) (optional_vertical_stripe : option (fin 3)) : num_possible_flags = 108 :=
by
  sorry

end flag_designs_l469_469066


namespace ajay_distance_l469_469033

/- Definitions -/
def speed : ℝ := 50 -- Ajay's speed in km/hour
def time : ℝ := 30 -- Time taken in hours

/- Theorem statement -/
theorem ajay_distance : (speed * time = 1500) :=
by
  sorry

end ajay_distance_l469_469033


namespace final_state_l469_469361

theorem final_state (initial_white : ℕ) (initial_black : ℕ) (end_white : ℕ) (end_black : ℕ) (end_total : ℕ) : 
  initial_white = 2015 → 
  initial_black = 2015 → 
  end_total = 3 → 
  odd initial_black → 
  (∀ n m : ℕ, (m % 2) = 1 → 3 - (2 * n - m) = 3) →
  end_white = 2 ∧ end_black = 1 := 
by
  intros h_initial_white h_initial_black h_end_total h_odd_initial_black h_invariant
  simp [h_initial_white, h_initial_black, h_end_total, h_odd_initial_black, h_invariant]
  sorry

end final_state_l469_469361


namespace valid_three_digit_numbers_count_l469_469186

noncomputable def count_valid_three_digit_numbers : Nat :=
  let valid_second_digits := (1 to 9).toList
  let valid_third_digit (b : Nat) : List Nat := (0 to b-1).toList
  let sum_choices_for_c := valid_second_digits.map (fun b => valid_third_digit b).map List.length
  let total_choices := sum_choices_for_c.foldl (+) 0
  let total_valid_numbers := total_choices * 9  -- multiply by the 9 choices for the hundreds digit
  total_valid_numbers

theorem valid_three_digit_numbers_count : count_valid_three_digit_numbers = 405 := 
  by sorry

end valid_three_digit_numbers_count_l469_469186


namespace painting_rate_l469_469321

/-- Define various dimensions and costs for the room -/
def room_length : ℝ := 10
def room_width  : ℝ := 7
def room_height : ℝ := 5

def door_width  : ℝ := 1
def door_height : ℝ := 3
def num_doors   : ℕ := 2

def large_window_width  : ℝ := 2
def large_window_height : ℝ := 1.5
def num_large_windows   : ℕ := 1

def small_window_width  : ℝ := 1
def small_window_height : ℝ := 1.5
def num_small_windows   : ℕ := 2

def painting_cost : ℝ := 474

/-- The rate for painting the walls is Rs. 3 per sq m -/
theorem painting_rate : (painting_cost / 
  ((2 * (room_length * room_height) + 2 * (room_width * room_height)) -
   (num_doors * (door_width * door_height) +
    num_large_windows * (large_window_width * large_window_height) +
    num_small_windows * (small_window_width * small_window_height)))) = 3 := 
by 
  -- Proof is omitted
  sorry

end painting_rate_l469_469321


namespace find_values_l469_469147

theorem find_values (a b c : ℕ) 
  (h1 : ∃ x y : ℝ, x = real.sqrt (a - 4) ∧ y = (2 - 2 * b)^2 ∧ x * y < 0) 
  (h2 : c = int.floor (real.sqrt 10)) : 
  a = 4 ∧ b = 1 ∧ c = 3 :=
by 
  sorry

end find_values_l469_469147


namespace least_common_multiple_first_ten_l469_469548

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469548


namespace least_common_multiple_1_to_10_l469_469498

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469498


namespace speed_of_ship_in_km_per_hr_l469_469021

/-- Conditions -/
def length_of_ship : ℝ := 450
def length_of_bridge : ℝ := 900
def time_to_pass_bridge : ℝ := 202.48

/-- Prove the speed of the ship in km/hr -/
theorem speed_of_ship_in_km_per_hr (h1 : length_of_ship = 450) (h2 : length_of_bridge = 900) (h3 : time_to_pass_bridge = 202.48) : 
  let total_distance := length_of_ship + length_of_bridge in
  let speed_m_per_s := total_distance / time_to_pass_bridge in
  let speed_km_per_hr := speed_m_per_s * 3.6 in
  speed_km_per_hr ≈ 24 :=
by sorry

end speed_of_ship_in_km_per_hr_l469_469021


namespace least_common_multiple_of_first_10_integers_l469_469511

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469511


namespace floor_sqrt_27_square_l469_469075

theorem floor_sqrt_27_square : (Int.floor (Real.sqrt 27))^2 = 25 :=
by
  sorry

end floor_sqrt_27_square_l469_469075


namespace vladimir_can_invest_more_profitably_l469_469674

-- Conditions and parameters
def p_buckwheat_initial : ℝ := 70 -- initial price of buckwheat in RUB/kg
def p_buckwheat_2017 : ℝ := 85 -- price of buckwheat in early 2017 in RUB/kg
def rate_2015 : ℝ := 0.16 -- interest rate for annual deposit in 2015
def rate_2016 : ℝ := 0.10 -- interest rate for annual deposit in 2016
def rate_2yr : ℝ := 0.15 -- interest rate for two-year deposit per year

-- Amounts after investments
def amount_annual : ℝ := p_buckwheat_initial * (1 + rate_2015) * (1 + rate_2016)
def amount_2yr : ℝ := p_buckwheat_initial * (1 + rate_2yr)^2

-- Prove that the best investment amount is greater than the 2017 buckwheat price
theorem vladimir_can_invest_more_profitably : max amount_annual amount_2yr > p_buckwheat_2017 := by
  sorry

end vladimir_can_invest_more_profitably_l469_469674


namespace least_common_multiple_of_first_ten_l469_469598

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469598


namespace tens_digit_of_square_ending_in_six_odd_l469_469887

theorem tens_digit_of_square_ending_in_six_odd 
   (N : ℤ) 
   (a : ℤ) 
   (b : ℕ) 
   (hle : 0 ≤ b) 
   (hge : b < 10) 
   (hexp : N = 10 * a + b) 
   (hsqr : (N^2) % 10 = 6) : 
   ∃ k : ℕ, (N^2 / 10) % 10 = 2 * k + 1 :=
sorry -- Proof goes here

end tens_digit_of_square_ending_in_six_odd_l469_469887


namespace floor_sqrt_27_squared_eq_25_l469_469081

theorem floor_sqrt_27_squared_eq_25 :
  (⌊Real.sqrt 27⌋)^2 = 25 :=
by
  have H1 : 5 < Real.sqrt 27 := sorry
  have H2 : Real.sqrt 27 < 6 := sorry
  have floor_sqrt_27_eq_5 : ⌊Real.sqrt 27⌋ = 5 := sorry
  rw floor_sqrt_27_eq_5
  norm_num

end floor_sqrt_27_squared_eq_25_l469_469081


namespace least_common_multiple_of_first_10_integers_l469_469509

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469509


namespace lcm_first_ten_positive_integers_l469_469424

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469424


namespace sum_of_eighth_powers_of_roots_l469_469311

noncomputable def quadratic_roots (a b c : ℝ) : ℝ × ℝ :=
  let discriminant := b^2 - 4 * a * c
  let root_disc := Real.sqrt discriminant
  ((-b + root_disc) / (2 * a), (-b - root_disc) / (2 * a))

theorem sum_of_eighth_powers_of_roots :
  let (p, q) := quadratic_roots 1 (-Real.sqrt 7) 1
  p^2 + q^2 = 5 ∧ p^4 + q^4 = 23 ∧ p^8 + q^8 = 527 :=
by
  sorry

end sum_of_eighth_powers_of_roots_l469_469311


namespace white_ball_in_last_six_boxes_l469_469684

theorem white_ball_in_last_six_boxes :
  ∀ (b w : Nat) (boxes : Fin 20 → Fin 6 → Bool),
    b = 60 → w = 60 →
    (∀ i : Fin 14, card (boxes i) (λ x, boxes i x = true) > card (boxes i) (λ x, boxes i x = false)) →
    ∃ j : Fin 6, card (boxes (14 + j)) (λ x, boxes (14 + j) x = false) = 6 :=
by
  intros b w boxes hb hw h
  sorry

end white_ball_in_last_six_boxes_l469_469684


namespace probability_of_alternating_colors_l469_469703

-- The setup for our problem
variables (B W : Type) 

/-- A box contains 5 white balls and 5 black balls.
    Prove that the probability that all of my draws alternate colors
    is 1/126. -/
theorem probability_of_alternating_colors (W B : ℕ) (hw : W = 5) (hb : B = 5) :
  let total_ways := Nat.choose 10 5 in
  let successful_ways := 2 in
  successful_ways / total_ways = (1 : ℚ) / 126 :=
sorry

end probability_of_alternating_colors_l469_469703


namespace no_quadratic_trinomial_with_integer_coeff_pow_of_two_l469_469923

open Mathlib.Real

theorem no_quadratic_trinomial_with_integer_coeff_pow_of_two : ¬ ∃ (a b c : ℤ), ∀ (x : ℕ), ∃ (k : ℕ), (a * (x:ℤ)^2 + b * (x:ℤ) + c = 2^k) :=
sorry

end no_quadratic_trinomial_with_integer_coeff_pow_of_two_l469_469923


namespace lcm_first_ten_l469_469413

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469413


namespace jerry_wants_to_raise_average_l469_469248

theorem jerry_wants_to_raise_average {current_avg desired_avg : ℝ} {score4 : ℕ} 
  (h1 : current_avg = 78) (h2 : score4 = 86) : desired_avg = current_avg + 2 :=
by
  let total_3 := 3 * current_avg
  have h_total_3 : total_3 = 234 := by sorry -- Given Jerry’s average on the first 3 tests
  let total_4 := total_3 + score4
  have h_desired_avg : total_4 = 4 * desired_avg := by sorry -- The total score for 4 tests should be 4 * desired_avg
  have h_total_4 : total_4 = 320 := by
    rewrite [←h_total_3]
    rewrite [h2]
    sorry -- Simplify the total score
  have h_solve_avg : desired_avg = 80 := by
    rewrite [←h_total_4]
    sorry -- Solve the equation 320 = 4 * desired_avg
  show desired_avg = current_avg + 2
    rewrite [h1]
    rewrite [h_solve_avg]
    sorry -- Compare with current_avg

end jerry_wants_to_raise_average_l469_469248


namespace lcm_first_ten_l469_469407

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469407


namespace least_common_multiple_of_first_10_integers_l469_469510

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469510


namespace probability_of_alternating_draws_l469_469701

theorem probability_of_alternating_draws :
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls in
  let successful_orders : ℕ := 2,
      total_arrangements : ℕ := Nat.choose total_balls white_balls,
      probability : ℚ := successful_orders / total_arrangements in
  probability = 1 / 126 :=
by
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls
  let successful_orders : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let probability : ℚ := successful_orders / total_arrangements
  sorry

end probability_of_alternating_draws_l469_469701


namespace least_positive_integer_divisible_by_first_ten_l469_469478

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469478


namespace isosceles_triangle_fraction_outside_circle_l469_469226

theorem isosceles_triangle_fraction_outside_circle :
  ∀ (ABC : Triangle) (r : ℝ), 
  is_isosceles ABC ∧ tangents_to_circle ABC r ∧ angle_BAC ABC = 120 → 
  fraction_outside_area ABC r = 1 - (π / (4 * sqrt 3)) :=
sorry

end isosceles_triangle_fraction_outside_circle_l469_469226


namespace min_max_pieces_three_planes_l469_469072

theorem min_max_pieces_three_planes : 
  ∃ (min max : ℕ), (min = 4) ∧ (max = 8) := by
  sorry

end min_max_pieces_three_planes_l469_469072


namespace shortest_side_length_l469_469239

noncomputable def proof_problem : Prop :=
  ∃ (A B C : ℝ) (a b c : ℝ),
    tan A = 1 / 4 ∧
    tan B = 3 / 5 ∧
    c = sqrt 17 ∧
    C = π - A - B ∧
    sin A / a = sin B / b ∧
    sin C / c = sin A / a ∧
    (a < b ∧ a < c) → a = sqrt 2

theorem shortest_side_length : proof_problem := sorry

end shortest_side_length_l469_469239


namespace area_under_curve_l469_469256

noncomputable def area_bounded (u : ℝ) (hU : 0 ≤ u) : ℝ :=
  real.exp(u)

theorem area_under_curve (u : ℝ) (hU : 0 ≤ u) (x y : ℝ) (hx : x = (real.exp u + real.exp (-u)) / 2) (hy : y = (real.exp u - real.exp (-u)) / 2) 
  (hCurve : x^2 - y^2 = 1) : 
  area_bounded u hU = u / 2 :=
sorry

end area_under_curve_l469_469256


namespace imaginary_part_of_conjugate_l469_469835

-- Assumptions for the given conditions
variables {z : ℂ}  -- z is a complex number
variables (hz1 : z.re > 0) (hz2 : z.im > 0)  -- z is in the first quadrant

-- The complex condition given in the problem
theorem imaginary_part_of_conjugate (hz3 : z^2 + 2 * conj z = 2) : z.im + (-1) = 0 :=
begin
  -- Proof goes here
  sorry
end

end imaginary_part_of_conjugate_l469_469835


namespace circle_intersection_l469_469231

theorem circle_intersection : 
  ∀ (O : ℝ × ℝ), ∃ (m n : ℤ), (dist (O.1, O.2) (m, n) ≤ 100 + 1/14) := 
sorry

end circle_intersection_l469_469231


namespace sequence_sum_2015_l469_469175

theorem sequence_sum_2015 :
  let a : ℕ → ℤ := λ n, 
    if n % 6 = 0 then 2014
    else if n % 6 = 1 then 2015
    else if n % 6 = 2 then 1
    else if n % 6 = 3 then -2014
    else if n % 6 = 4 then -2015
    else -1 
  in (Finset.range 2015).sum a = 1 :=
by
  sorry

end sequence_sum_2015_l469_469175


namespace circle_in_fourth_quadrant_l469_469212

theorem circle_in_fourth_quadrant (a : ℝ) :
  (∃ (x y: ℝ), x^2 + y^2 - 2 * a * x + 4 * a * y + 6 * a^2 - a = 0 ∧ (a > 0) ∧ (-2 * y < 0)) → (0 < a ∧ a < 1) :=
by
  sorry

end circle_in_fourth_quadrant_l469_469212


namespace find_radius_l469_469317

-- Defining the conditions as given in the math problem
def sectorArea (r : ℝ) (L : ℝ) : ℝ := 0.5 * r * L

theorem find_radius (h1 : sectorArea r 5.5 = 13.75) : r = 5 :=
by sorry

end find_radius_l469_469317


namespace find_value_of_2p_plus_q_l469_469857

theorem find_value_of_2p_plus_q :
  ∃ p q : ℝ, 
  (A = {x : ℝ | x^2 + p * x - 3 = 0}) ∧ 
  (B = {x : ℝ | x^2 - q * x - p = 0}) ∧ 
  (A ∩ B = {-1}) ∧ 
  2 * p + q = -7 :=
by
  let A := {x : ℝ | x^2 + p * x -3 = 0}
  let B := {x : ℝ | x^2 - q * x - p = 0}
  exists p q
  split
  repeat sorry

end find_value_of_2p_plus_q_l469_469857


namespace lcm_first_ten_integers_l469_469626

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469626


namespace length_greater_than_3w_by_2_l469_469334

-- Define the data types and conditions
variable (w l x : ℝ)
variable (P : ℝ) := 100
variable (l_val : ℝ) := 38

-- Assume the given conditions
axiom length_is_3w_plus_x : l = 3 * w + x
axiom perimeter_is_100 : P = 2 * l + 2 * w
axiom length_is_38 : l = l_val

-- Now we prove that x = 2
theorem length_greater_than_3w_by_2 : x = 2 :=
by
  -- Here we will put the proof but for now we use sorry
  sorry

end length_greater_than_3w_by_2_l469_469334


namespace least_common_multiple_of_first_ten_positive_integers_l469_469469

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469469


namespace cos_identity_proof_l469_469347

open Real

theorem cos_identity_proof : 
  2 * cos (16 * (π / 180)) * cos (29 * (π / 180)) - cos (13 * (π / 180)) = sqrt 2 / 2 :=
by
  sorry

end cos_identity_proof_l469_469347


namespace percentage_corresponding_to_120_l469_469205

variable (x p : ℝ)

def forty_percent_eq_160 := (0.4 * x = 160)
def p_times_x_eq_120 := (p * x = 120)

theorem percentage_corresponding_to_120 (h₁ : forty_percent_eq_160 x) (h₂ : p_times_x_eq_120 x p) :
  p = 0.30 :=
sorry

end percentage_corresponding_to_120_l469_469205


namespace quadratic_function_max_point_l469_469997

theorem quadratic_function_max_point (a b c m : ℝ) (h1 : ∀ x, x = 2 → y = a * x^2 + b * x + c = 6)
    (h2 : ∀ x, x = 0 → y = a * x^2 + b * x + c = -10)
    (h3 : ∀ x, x = 5 → y = a * x^2 + b * x + c = m) : m = -30 := by
  sorry

end quadratic_function_max_point_l469_469997


namespace least_common_multiple_1_to_10_l469_469450

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469450


namespace units_digit_proof_l469_469070

def units_digit_expr (a b : ℂ) : ℤ :=
  (a^25 - b^25 + 2 * a^91).re.to_int % 10

theorem units_digit_proof : 
  let a := 17 + Real.sqrt 251
  let b := 17 - Real.sqrt 251
  units_digit_expr a b = 0 := by
  intro a b
  have h1 : a = 17 + Real.sqrt 251 := rfl
  have h2 : b = 17 - Real.sqrt 251 := rfl
  sorry

end units_digit_proof_l469_469070


namespace proof_system_solution_l469_469986

noncomputable def solve_system (a b c : ℝ) :=
  log 3 a + log 3 b + log 3 c = 0 ∧
  3^(3^a) + 3^(3^b) + 3^(3^c) = 81

theorem proof_system_solution : 
  ∀ (a b c : ℝ), solve_system a b c → 
  (a = 1) ∧ (b = 1) ∧ (c = 1) :=
by
  intro a b c
  intro h
  sorry

end proof_system_solution_l469_469986


namespace length_of_platform_l469_469736

theorem length_of_platform (train_length : ℝ) (train_speed_kmph : ℝ) (crossing_time_sec : ℝ) : 
  train_length = 450 → train_speed_kmph = 110 → crossing_time_sec = 40 → 
  (let train_speed_mps := train_speed_kmph * (1000 / 3600) in
   let distance_covered := train_speed_mps * crossing_time_sec in 
   let platform_length := distance_covered - train_length in 
   platform_length ≈ 772.222) :=
begin
  intros h1 h2 h3,
  let train_speed_mps := train_speed_kmph * (1000 / 3600),
  let distance_covered := train_speed_mps * crossing_time_sec,
  let platform_length := distance_covered - train_length,
  have h_platform_length : platform_length ≈ 772.222,
  {
    sorry
  },
  exact h_platform_length,
end

end length_of_platform_l469_469736


namespace dishwasher_manager_wage_ratio_l469_469753

theorem dishwasher_manager_wage_ratio
  (chef_wage dishwasher_wage manager_wage : ℝ)
  (h1 : chef_wage = 1.22 * dishwasher_wage)
  (h2 : dishwasher_wage = r * manager_wage)
  (h3 : manager_wage = 8.50)
  (h4 : chef_wage = manager_wage - 3.315) :
  r = 0.5 :=
sorry

end dishwasher_manager_wage_ratio_l469_469753


namespace lcm_first_ten_l469_469414

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469414


namespace sub_neg_eq_add_problem_l469_469056

theorem sub_neg_eq_add (a b : Int) : a - (-b) = a + b := sorry

theorem problem (a b : Int) : -14 - (-26) = 12 :=
by 
  have h1 : -14 - (-26) = -14 + 26 := by rw sub_neg_eq_add
  have h2 : -14 + 26 = 26 - 14 := by rw add_comm ; rw sub_eq_add_neg
  have h3 : 26 - 14 = 12 := by norm_num
  rw [h1, h2, h3]
  sorry

end sub_neg_eq_add_problem_l469_469056


namespace average_marks_l469_469662

noncomputable def TatuyaScore (IvannaScore : ℝ) : ℝ :=
2 * IvannaScore

noncomputable def IvannaScore (DorothyScore : ℝ) : ℝ :=
(3/5) * DorothyScore

noncomputable def DorothyScore : ℝ := 90

noncomputable def XanderScore (TatuyaScore IvannaScore DorothyScore : ℝ) : ℝ :=
((TatuyaScore + IvannaScore + DorothyScore) / 3) + 10

noncomputable def SamScore (IvannaScore : ℝ) : ℝ :=
(3.8 * IvannaScore) + 5.5

noncomputable def OliviaScore (SamScore : ℝ) : ℝ :=
(3/2) * SamScore

theorem average_marks :
  let I := IvannaScore DorothyScore
  let T := TatuyaScore I
  let S := SamScore I
  let O := OliviaScore S
  let X := XanderScore T I DorothyScore
  let total_marks := T + I + DorothyScore + X + O + S
  (total_marks / 6) = 145.458333 := by sorry

end average_marks_l469_469662


namespace least_positive_integer_divisible_by_first_ten_l469_469483

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469483


namespace max_digit_sum_of_fraction_l469_469875

theorem max_digit_sum_of_fraction (a b c : ℕ) (h1 : a ≤ 9) (h2 : b ≤ 9) (h3 : c ≤ 9)
  (h4 : ∃ y : ℕ, y ∈ {2, 4, 5, 8, 10} ∧ 0.abc * y = 1) : a + b + c ≤ 8 :=
by {
  have h_valid_range : (0 < y) ∧ (y ≤ 12), { sorry },  -- proof omited for brevity
  cases y,
  case 2 {
    sorry  -- detailed proof logic here
  },
  case 4 {
    sorry  -- detailed proof logic here
  },
  case 5 {
    sorry  -- detailed proof logic here
  },
  case 8 {
    sorry  -- detailed proof logic here
  },
  case 10 {
    sorry  -- detailed proof logic here
  }
}

end max_digit_sum_of_fraction_l469_469875


namespace least_common_multiple_first_ten_l469_469554

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469554


namespace lcm_first_ten_positive_integers_l469_469423

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469423


namespace students_with_exactly_two_skills_l469_469902

-- Definitions based on the conditions:
def total_students : ℕ := 150
def students_can_write : ℕ := total_students - 60 -- 150 - 60 = 90
def students_can_direct : ℕ := total_students - 90 -- 150 - 90 = 60
def students_can_produce : ℕ := total_students - 40 -- 150 - 40 = 110

-- The theorem statement
theorem students_with_exactly_two_skills :
  students_can_write + students_can_direct + students_can_produce - total_students = 110 := 
sorry

end students_with_exactly_two_skills_l469_469902


namespace lcm_first_ten_l469_469417

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469417


namespace solve_for_x_l469_469117

  theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → x = 27 :=
  by
    sorry
  
end solve_for_x_l469_469117


namespace least_common_multiple_of_first_ten_positive_integers_l469_469461

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469461


namespace prime_gt_three_square_mod_twelve_l469_469299

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end prime_gt_three_square_mod_twelve_l469_469299


namespace inequality_proof_l469_469312

theorem inequality_proof (a b c d : ℝ) (h : a > 0) (h : b > 0) (h : c > 0) (h : d > 0)
  (h₁ : (a * b) / (c * d) = (a + b) / (c + d)) : (a + b) * (c + d) ≥ (a + c) * (b + d) :=
sorry

end inequality_proof_l469_469312


namespace kylie_first_hour_apples_l469_469936

variable (A : ℕ) -- The number of apples picked in the first hour

-- Definitions based on the given conditions
def applesInFirstHour := A
def applesInSecondHour := 2 * A
def applesInThirdHour := A / 3

-- Total number of apples picked in all three hours
def totalApplesPicked := applesInFirstHour + applesInSecondHour + applesInThirdHour

-- The given condition that the total number of apples picked is 220
axiom total_is_220 : totalApplesPicked = 220

-- Proving that the number of apples picked in the first hour is 66
theorem kylie_first_hour_apples : A = 66 := by
  sorry

end kylie_first_hour_apples_l469_469936


namespace Grace_tower_height_l469_469058

/-- Define the height of Clyde's tower and the height of Grace's tower --/
variables (C G : ℕ)

/-- Conditions provided --/
axiom condition_1 : G = 8 * C
axiom condition_2 : G = C + 35

/-- Proof statement of the problem --/
theorem Grace_tower_height : G = 40 :=
by sorry

end Grace_tower_height_l469_469058


namespace least_common_multiple_first_ten_integers_l469_469573

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469573


namespace min_pos_solution_eqn_l469_469093

theorem min_pos_solution_eqn (x : ℝ) (h : (⌊x^2⌋ : ℤ) - (⌊x⌋ : ℤ)^2 = 25) : x = 7 * Real.sqrt 3 :=
sorry

end min_pos_solution_eqn_l469_469093


namespace proof_method_characterization_l469_469657

-- Definitions of each method
def synthetic_method := "proceeds from cause to effect, in a forward manner"
def analytic_method := "seeks the cause from the effect, working backwards"
def proof_by_contradiction := "assumes the negation of the proposition to be true, and derives a contradiction"
def mathematical_induction := "base case and inductive step: which shows that P holds for all natural numbers"

-- Main theorem to prove
theorem proof_method_characterization :
  (analytic_method == "seeks the cause from the effect, working backwards") :=
by
  sorry

end proof_method_characterization_l469_469657


namespace boys_in_class_l469_469893

theorem boys_in_class
  (g b : ℕ)
  (h_ratio : g = (3 * b) / 5)
  (h_total : g + b = 32) :
  b = 20 :=
sorry

end boys_in_class_l469_469893


namespace percentage_of_alcohol_in_mixture_A_l469_469964

theorem percentage_of_alcohol_in_mixture_A (x : ℝ) :
  (10 * x / 100 + 5 * 50 / 100 = 15 * 30 / 100) → x = 20 :=
by
  intro h
  sorry

end percentage_of_alcohol_in_mixture_A_l469_469964


namespace least_common_multiple_of_first_ten_integers_l469_469378

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469378


namespace solve_for_x_l469_469097

theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → (x = 27) := by
  intro h
  sorry

end solve_for_x_l469_469097


namespace find_a_l469_469154

theorem find_a (a : ℝ) : 
  (∃ (a : ℝ), a * 15 + 6 = -9) → a = -1 :=
by
  intro h
  sorry

end find_a_l469_469154


namespace integral_x_minus_inv_x_l469_469782

theorem integral_x_minus_inv_x :
  ∫ x in 1 .. 2, (x - (1 / x)) = 1 - real.log 2 :=
by
  sorry

end integral_x_minus_inv_x_l469_469782


namespace pawn_placement_ways_l469_469187

theorem pawn_placement_ways :
  let board_size := 8
  ∃ (ways : ℕ), 
  ways = 3^16 ∧ 
  ways = (∏ i in (Finset.range board_size), 9) :=
by
  sorry

end pawn_placement_ways_l469_469187


namespace january_25_is_thursday_l469_469880

theorem january_25_is_thursday (h : weekday.december25 = weekday.Monday) :
  weekday.january25 = weekday.Thursday :=
sorry

end january_25_is_thursday_l469_469880


namespace range_of_x_l469_469170

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + x + 1

-- State the theorem to prove the condition
theorem range_of_x (x : ℝ) : f (1 - x) + f (2 * x) > 2 ↔ x > -1 :=
by {
  sorry -- Proof placeholder
}

end range_of_x_l469_469170


namespace sum_reciprocal_product_l469_469266

variables {R : Type*} [OrderedField R]

def f (x : R) (xs : List R) : R := xs.foldr (λ x_i acc, acc * (x - x_i)) 1

def f' (x : R) (xs : List R) : R := xs.foldr (λ x_i acc, acc * (x - x_i), x ≠ x_i) 0 -- derivative placeholder

theorem sum_reciprocal_product (xs : List R) (k : ℕ) (h_distinct : ∀ i j : ℕ, i ≠ j → xs.nth i ≠ xs.nth j) (h_nz : ∀ i : ℕ, xs.nth i ≠ 0) :
  (if h : (0 : ℕ) ≤ k ∧ k ≤ xs.length - 2 then 
    ∑ i in (Finset.range xs.length), (xs.nth i) ^ k / (f'.nth i)
  else 
    ∑ i in (Finset.range xs.length), (xs.nth i) ^ (xs.length - 1) / (f'.nth i)
  ) = if h : (0 : ℕ) ≤ k ∧ k ≤ xs.length - 2 then 0 else 1 := by
  sorry

end sum_reciprocal_product_l469_469266


namespace max_ounces_amber_can_get_l469_469036

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l469_469036


namespace collinear_M_B_T_l469_469901

open EuclideanGeometry

theorem collinear_M_B_T
  (ABC : Triangle)
  (circumcircle : Circle)
  (scalene : ABC.scalene)
  (AB_AC_ineq : ABC.AB < ABC.AC)
  (PB PC : Point)
  (tangents : Tangent circumsircle PB ∧ Tangent circumsircle PC)
  (R : Point)
  (R_on_arc_AC : R ∈ circumcircle ∧ R.on_arc AC ∧ R ∉ arc B)
  (PR : Line)
  (PR_intersect_Q : PR ∩ circumcircle = {Q})
  (I : Point)
  (incenter : Incenter ABC I)
  (ID_perp_BC : Perpendicular ID ABC.BC)
  (QD_intersect_G : QD ∩ circumcircle = {G})
  (line_through_I_perp_AI : Line)
  (intersection_M_N : line_through_I_perp_AI ∩ AG = {M} ∧ line_through_I_perp_AI ∩ AC = {N})
  (S : Point)
  (S_mid_of_AR : S = midpoint_arc AR)
  (SN_intersect_T : Intersection SN circumcircle = T)
  (parallel_AR_BC : Parallel AR BC) :
  Collinear {M, B, T}
:= sorry

end collinear_M_B_T_l469_469901


namespace length_of_BC_l469_469034

theorem length_of_BC (b : ℝ) (area : ℝ) :
  (∀ (x : ℝ), ∃ (y : ℝ), y = 2 * x^2) →
  (∀ (A B C : (ℝ × ℝ)), A = (0, 0) ∧ 
      (∃ (b : ℝ), B = (-b, 2 * b^2) ∧ C = (b, 2 * b^2) ∧ 
      area = 128 ∧ area = 1/2 * (C.1 - B.1) * (2 * b^2))) →
  2 * b = 8 :=
by
  intros hyp1 hyp2
  have fact1 : 2 * b^3 = 128 := sorry
  have fact2 : b = real.cbrt 64 := sorry
  have fact3 : b = 4 := sorry
  exact eq.symm ((mul_eq_mul_right_iff.mpr (or.inl fact3)))

end length_of_BC_l469_469034


namespace limit_a_n_l469_469337

variable {a : ℕ → ℝ}

theorem limit_a_n (h : tendsto (λ n : ℕ, a (n + 2) - a n) atTop (𝓝 0)) :
  tendsto (λ n : ℕ, (a (n + 1) - a n) / n) atTop (𝓝 0) :=
sorry

end limit_a_n_l469_469337


namespace lcm_first_ten_integers_l469_469631

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469631


namespace least_common_multiple_of_first_ten_l469_469604

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469604


namespace max_partial_sum_l469_469230

variable (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ)
variable (S : ℕ → ℤ)

-- Define the arithmetic sequence and the conditions given
def arithmetic_sequence (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a_n n = a_1 + n * d

def condition1 (a_1 : ℤ) : Prop := a_1 > 0

def condition2 (a_n : ℕ → ℤ) (d : ℤ) : Prop := 3 * (a_n 8) = 5 * (a_n 13)

-- Define the partial sum of the arithmetic sequence
def partial_sum (S : ℕ → ℤ) (a_n : ℕ → ℤ) : Prop :=
∀ n : ℕ, S n = n * (a_n 1 + a_n n) / 2

-- Define the main problem: Prove that S_20 is the greatest
theorem max_partial_sum (a_n : ℕ → ℤ) (a_1 : ℤ) (d : ℤ) (S : ℕ → ℤ) :
  arithmetic_sequence a_n a_1 d →
  condition1 a_1 →
  condition2 a_n d →
  partial_sum S a_n →
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → S 20 ≥ S n := by
  sorry

end max_partial_sum_l469_469230


namespace work_done_together_l469_469660

-- Conditions
def P_rate : ℝ := 1 / 3
def Q_rate : ℝ := 1 / 9
def combined_rate : ℝ := P_rate + Q_rate

-- Time they work together
def t : ℝ := 2

-- Work done by P alone in the additional 20 minutes
def P_additional_work : ℝ := 1 / 9

-- The equation that needs to hold true
theorem work_done_together : combined_rate * t + P_additional_work = 1 := sorry

end work_done_together_l469_469660


namespace geo_progression_perfect_square_sum_l469_469825

noncomputable theory

open_locale big_operators -- To use ∑ notation for sums

def is_GeoProgression (a : ℕ) (r : ℕ) (n : ℕ) (terms : list ℕ) : Prop :=
  terms = (list.finRange n).map (λ i, a * r^i)

def sum_terms (terms : list ℕ) : ℕ := ∑ i in terms, id i

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m, m * m = n

theorem geo_progression_perfect_square_sum :
  ∃ (a r : ℕ) (terms : list ℕ),
  length terms = 6 ∧
  is_GeoProgression a r 6 terms ∧
  sum_terms terms = 344 ∧
  sum_terms (terms.filter is_perfect_square) = 27 :=
begin
  sorry
end

end geo_progression_perfect_square_sum_l469_469825


namespace hansol_weight_l469_469889

variables (Hb Hs : ℝ)

theorem hansol_weight :
  Hb + Hs = 88 ∧ Hb = Hs + 4 → Hs = 42 :=
by
  intros h,
  sorry

end hansol_weight_l469_469889


namespace least_common_multiple_first_ten_l469_469644

theorem least_common_multiple_first_ten :
  ∃ (n : ℕ), (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ n = 2520 := 
sorry

end least_common_multiple_first_ten_l469_469644


namespace angle_ABD_30_l469_469137

-- Given a triangle ABC
variables {A B C D : Type} [InnerProductSpace ℝ (EuclideanSpace ℝ (Fin 3))]

-- Median condition: BD is a median
def isMedian (A B C D : EuclideanSpace ℝ (Fin 3)) : Prop :=
  dist B D = dist B C / 2 ∧ dist D C = dist A C / 2

-- Conditions: 
-- 1. BD is perpendicular to BC
axiom angle_DBC_90 (A B C D : EuclideanSpace ℝ (Fin 3)) 
  (hMedian : isMedian A B C D) : ∠BDC = π / 2

-- 2. BD = √3/4 * AB
axiom magnitude_BD (A B C D : EuclideanSpace ℝ (Fin 3)) 
  (hMedian : isMedian A B C D) :
  dist B D = sqrt(3) / 4 * dist A B

-- Conclude that ABD = 30 degrees
theorem angle_ABD_30 (A B C D : EuclideanSpace ℝ (Fin 3))
  (hMedian : isMedian A B C D) 
  (hAngle : angle_DBC_90 A B C D hMedian)
  (hMag : magnitude_BD A B C D hMedian) : 
  ∠ABD = π / 6 := 
sorry -- Proof here

end angle_ABD_30_l469_469137


namespace polygon_divided_l469_469723

theorem polygon_divided (p q r : ℕ) : p - q + r = 1 :=
sorry

end polygon_divided_l469_469723


namespace perfect_square_trinomial_l469_469878

theorem perfect_square_trinomial (m : ℝ) :
  ∃ (a : ℝ), (∀ (x : ℝ), x^2 - 2*(m-3)*x + 16 = (x - a)^2) ↔ (m = 7 ∨ m = -1) := by
  sorry

end perfect_square_trinomial_l469_469878


namespace sqrt_floor_square_l469_469078

theorem sqrt_floor_square {x : ℝ} (hx : 27 = x) (h1 : sqrt 25 < sqrt x) (h2 : sqrt x < sqrt 36) :
  (⌊sqrt x⌋.to_real)^2 = 25 :=
by {
  have hsqrt : 5 < sqrt x ∧ sqrt x < 6, by {
    split; linarith,
  },
  have h_floor_sqrt : ⌊sqrt x⌋ = 5, by {
    exact int.floor_eq_iff.mpr ⟨int.lt_floor_add_one.mpr hsqrt.2, hsqrt.1⟩,
  },
  rw h_floor_sqrt,
  norm_num,
  sorry  -- proof elided
}

end sqrt_floor_square_l469_469078


namespace least_common_multiple_of_first_ten_positive_integers_l469_469457

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469457


namespace solve_for_b_l469_469200

theorem solve_for_b (b : ℚ) (h : b + b / 4 - 1 = 3 / 2) : b = 2 :=
sorry

end solve_for_b_l469_469200


namespace ten_percent_of_n_l469_469706

variable (n f : ℝ)

theorem ten_percent_of_n (h : n - (1 / 4 * 2) - (1 / 3 * 3) - f * n = 27) : 
  0.10 * n = 0.10 * (28.5 / (1 - f)) :=
by
  simp only [*, mul_one_div_cancel, mul_sub, sub_eq_add_neg, add_div, div_self, one_div, mul_add]
  sorry

end ten_percent_of_n_l469_469706


namespace least_common_multiple_of_first_10_integers_l469_469521

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469521


namespace parallelogram_area_l469_469788

noncomputable def sin_degree (d : ℝ) : ℝ := Real.sin (d * Real.pi / 180)

theorem parallelogram_area (a b : ℝ) (θ : ℝ) (h : 0 < b) : 
  a = 26 → b = 14 → θ = 37 →
  let height := b * sin_degree(θ) in
  let area := a * height in
  area = 219.0552 :=
by
  sorry

end parallelogram_area_l469_469788


namespace lcm_first_ten_l469_469401

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469401


namespace least_positive_integer_divisible_by_first_ten_l469_469474

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469474


namespace oranges_thrown_away_l469_469732

theorem oranges_thrown_away (initial_oranges old_oranges_thrown new_oranges final_oranges : ℕ) 
    (h1 : initial_oranges = 34)
    (h2 : new_oranges = 13)
    (h3 : final_oranges = 27)
    (h4 : initial_oranges - old_oranges_thrown + new_oranges = final_oranges) :
    old_oranges_thrown = 20 :=
by
  sorry

end oranges_thrown_away_l469_469732


namespace ophelia_age_is_20_5_l469_469224

-- Define the constants and variables in the given problem
variable (O L M : ℤ) -- Current ages of Ophelia, Lennon, and Mike
constant h1 : O + 15 = 35 * (L + 15) / 10
constant h2 : M + 15 = 2 * (O - L)
constant h3 : M + 15 = 75 * (O + L + 30) / 100
constant h4 : L = 7
constant h5 : M = L + 5

-- Lean proposition that proves Ophelia's current age
theorem ophelia_age_is_20_5 : O = 41 / 2 :=
by
  have : ∀ (x : ℤ), 35 * x / 10 = 7 * x / 2 := sorry
  have : ∀ (x : ℤ), 75 * x / 100 = 3 * x / 4 := sorry
  rw [this, this, h4, h5] at h1 h3
  have : 12 + 15 = 2 * (O - 7) := by rw h2
  sorry

end ophelia_age_is_20_5_l469_469224


namespace annual_population_addition_l469_469913

noncomputable def net_population_increase_per_day : ℕ :=
  (24 / 8) - 1

noncomputable def net_population_increase_per_year : ℕ :=
  net_population_increase_per_day * 365

theorem annual_population_addition :
  round (net_population_increase_per_year : ℝ) 100 = 700 :=
by
  have births_per_day := 24 / 8
  have deaths_per_day := 1
  have net_increase_per_day := births_per_day - deaths_per_day
  have annual_increase := net_increase_per_day * 365
  calc
    round (annual_increase : ℝ) 100 = round (730 : ℝ) 100 := by sorry
    ... = 700 := by sorry

end annual_population_addition_l469_469913


namespace solve_for_y_l469_469071

theorem solve_for_y (y : ℝ) (h : (2 * y * sqrt (y^3))^(1/5) = 5) : 
  y = 25 * 2^(-2/5) := sorry

end solve_for_y_l469_469071


namespace females_in_group_l469_469121

theorem females_in_group (n F M : ℕ) (Index_F Index_M : ℝ) 
  (h1 : n = 25) 
  (h2 : Index_F = (n - F) / n)
  (h3 : Index_M = (n - M) / n) 
  (h4 : Index_F - Index_M = 0.36) :
  F = 8 := 
by
  sorry

end females_in_group_l469_469121


namespace correct_propositions_l469_469328

theorem correct_propositions :
  ∀ (a b x y : ℝ)
    (h1 : |a - b| < 1)
    (h2 : |x| < 2)
    (h3 : |y| > 3),
      (|a| < |b| + 1) ∧
      (|a + b| - 2*|a| ≤ |a - b|) ∧
      (|x / y| < 2 / 3) :=
by
  intros a b x y h1 h2 h3
  split
  { -- Proposition ①
    calc |a| = |a - b + b| : by sorry
           ... ≤ |a - b| + |b| : by sorry
           ... < 1 + |b| : by sorry },
  split
  { -- Proposition ②
    calc |a + b| - 2*|a| ≤ |a| + |b| - 2*|a| : by sorry
           ... = |b| - |a| : by sorry
           ... ≤ |a - b| : by sorry },
  { -- Proposition ③
    calc |x / y| = |x| / |y| : by sorry
           ... < 2 / |y| : by sorry
           ... < 2 / 3 : by sorry }

end correct_propositions_l469_469328


namespace alternating_sequence_probability_l469_469694

theorem alternating_sequence_probability : 
  let total_balls := 10 -- Total number of balls
  let white_balls := 5 -- Number of white balls
  let black_balls := 5 -- Number of black balls
  let successful_sequences := 2 -- Number of successful alternating sequences (BWBWBWBWBW and WBWBWBWBWB)
  let total_arrangements := Nat.choose total_balls white_balls -- Binomial coefficient for total arrangements
  (successful_sequences : ℚ) / total_arrangements = 1 / 126 :=
by
  sorry

end alternating_sequence_probability_l469_469694


namespace count_valid_quadratic_polynomials_l469_469943

theorem count_valid_quadratic_polynomials :
  let P (x : ℝ) := (x - 1) * (x - 3) * (x - 5)
  in ∃ (R : Polynomial ℝ), R.degree = 3 ∧
  (∃ (Q : Polynomial ℝ), Q.degree = 2 ∧ P(Q) = Polynomial.comp P Q * R) :=
sorry

end count_valid_quadratic_polynomials_l469_469943


namespace least_common_multiple_of_first_10_integers_l469_469515

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469515


namespace cylinder_surface_area_l469_469153

noncomputable def surface_area_of_cylinder (r l : ℝ) : ℝ :=
  2 * Real.pi * r * (r + l)

theorem cylinder_surface_area (r : ℝ) (h_radius : r = 1) (l : ℝ) (h_length : l = 2 * r) :
  surface_area_of_cylinder r l = 6 * Real.pi := by
  -- Using the given conditions and definition, we need to prove the surface area is 6π
  sorry

end cylinder_surface_area_l469_469153


namespace find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469106

theorem find_x_such_that_sqrt_5x_plus_9_eq_12 : ∀ x : ℝ, sqrt (5 * x + 9) = 12 → x = 27 := 
by
  intro x
  sorry

end find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469106


namespace affine_function_solution_l469_469785

theorem affine_function_solution {f : ℝ → ℝ} :
  (∀ x y : ℝ, f(x^3) - f(y^3) = (x^2 + xy + y^2) * (f(x) - f(y))) →
  ∃ a b : ℝ, ∀ x : ℝ, f(x) = a * x + b :=
by
  sorry

end affine_function_solution_l469_469785


namespace least_divisible_1_to_10_l469_469528

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469528


namespace infinite_n_exists_r_s_t_l469_469132

noncomputable def a (n : ℕ) : ℝ := n^(1/3 : ℝ)
noncomputable def b (n : ℕ) : ℝ := 1 / (a n - ⌊a n⌋)
noncomputable def c (n : ℕ) : ℝ := 1 / (b n - ⌊b n⌋)

theorem infinite_n_exists_r_s_t :
  ∃ (n : ℕ) (r s t : ℤ), (0 < n ∧ ¬∃ k : ℕ, n = k^3) ∧ (¬(r = 0 ∧ s = 0 ∧ t = 0)) ∧ (r * a n + s * b n + t * c n = 0) :=
sorry

end infinite_n_exists_r_s_t_l469_469132


namespace distance_between_first_and_last_trees_l469_469990

theorem distance_between_first_and_last_trees
  (num_trees : ℕ)
  (dist_first_to_fifth : ℝ)
  (num_trees_eq : num_trees = 10)
  (dist_first_to_fifth_eq : dist_first_to_fifth = 100) :
  let num_spaces := num_trees - 1
  in let dist_between_trees := dist_first_to_fifth / 4
  in dist_between_trees * (num_trees - 1) = 225 := sorry

end distance_between_first_and_last_trees_l469_469990


namespace lcm_first_ten_l469_469415

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469415


namespace least_common_multiple_first_ten_l469_469555

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469555


namespace even_cos_function_l469_469850

theorem even_cos_function {ϕ : ℝ} (h: ∀ x : ℝ, cos(3*x + ϕ) = cos(3*(-x) + ϕ)) : ∃ k : ℤ, ϕ = k * real.pi := 
sorry

end even_cos_function_l469_469850


namespace hcf_of_two_numbers_l469_469332

theorem hcf_of_two_numbers (H : ℕ) (hcf : Nat.gcd A B = H) (A = 350) (lcm : Nat.lcm A B = 13 * 14 * H) : H = 70 := 
by 
  sorry

end hcf_of_two_numbers_l469_469332


namespace least_common_multiple_of_first_10_integers_l469_469512

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469512


namespace lcm_first_ten_positive_integers_l469_469433

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469433


namespace percentage_loss_original_price_l469_469000

-- Definitions from conditions
def SP1 := 810
def SP2 := 990
def gain := 10 / 100

-- Lean statement to prove percentage of loss
theorem percentage_loss_original_price (CP : ℝ) (H_percent_gain : 1.1 * CP = 990) :
  let loss := CP - SP1 in
  let percentage_loss := (loss / CP) * 100 in
  percentage_loss = 10 := 
sorry

end percentage_loss_original_price_l469_469000


namespace find_decreasing_function_l469_469044

theorem find_decreasing_function :
  ∃ f : ℝ → ℝ, (∀ x : ℝ, f x = -2 * x) ∧
  (∀ g : ℝ → ℝ, g = (λ x, 2 / x) ∨ g = (λ x, -2 / x) ∨ g = (λ x, 2 * x) ∨ g = (λ x, -2 * x) → 
               (∀ x : ℝ, (f x > f (x + 1)) → 
                 ¬(g = (λ x, 2 / x) ∧ g = (λ x, -2 / x) ∧ g = (λ x, 2 * x) ∧ g ≠ (λ x, -2 * x)))) :=
by
  -- Detailed proof to be filled in
  sorry

end find_decreasing_function_l469_469044


namespace prob_two_girls_l469_469031

variable (Pboy Pgirl : ℝ)

-- Conditions
def prob_boy : Prop := Pboy = 1 / 2
def prob_girl : Prop := Pgirl = 1 / 2

-- The theorem to be proven
theorem prob_two_girls (h₁ : prob_boy Pboy) (h₂ : prob_girl Pgirl) : (Pgirl * Pgirl) = 1 / 4 :=
by
  sorry

end prob_two_girls_l469_469031


namespace lcm_first_ten_l469_469403

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469403


namespace triangle_identity_theorem_l469_469242

noncomputable def triangle_identity (a b c A B C : ℝ) : Prop := 
  let cos_half_angle_difference (x y : ℝ) := (cos (x - y) / 2)
  (cos_half_angle_difference B C) = ((b + c) / (2 * a)) →
  (cos_half_angle_difference C A) = ((c + a) / (2 * b)) →
  (cos_half_angle_difference A B) = ((a + b) / (2 * c)) →
  (a^2 * (cos_half_angle_difference B C)^2 / (cos B + cos C) +
   b^2 * (cos_half_angle_difference C A)^2 / (cos C + cos A) +
   c^2 * (cos_half_angle_difference A B)^2 / (cos A + cos B) = a*b + b*c + c*a)

theorem triangle_identity_theorem (a b c A B C : ℝ)
  (h1 : (cos (B - C) / 2) = ((b + c) / (2 * a)))
  (h2 : (cos (C - A) / 2) = ((c + a) / (2 * b)))
  (h3 : (cos (A - B) / 2) = ((a + b) / (2 * c))) :
  a^2 * (cos (B - C) / 2)^2 / (cos B + cos C) +
  b^2 * (cos (C - A) / 2)^2 / (cos C + cos A) +
  c^2 * (cos (A - B) / 2)^2 / (cos A + cos B) = a*b + b*c + c*a := by
  sorry

end triangle_identity_theorem_l469_469242


namespace lcm_first_ten_numbers_l469_469596

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469596


namespace least_common_multiple_of_first_ten_l469_469602

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469602


namespace lcm_first_ten_integers_l469_469623

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469623


namespace car_speed_first_hour_l469_469341

theorem car_speed_first_hour (x : ℝ) (h1 : 40) (h2 : (x + h1) / 2 = 90) : x = 140 := 
by
  sorry

end car_speed_first_hour_l469_469341


namespace least_common_multiple_of_first_ten_integers_l469_469375

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469375


namespace range_h_l469_469069

noncomputable def h (t : ℝ) : ℝ := (t^2 + t + 1) / (t^2 + 2)

theorem range_h : {y : ℝ | ∃ t : ℝ, h(t) = y} = {y : ℝ | 1 / 2 ≤ y ∧ y ≤ 3 / 4} := by
  sorry

end range_h_l469_469069


namespace correct_statements_count_l469_469142

variables {α β : Type} [plane α] [plane β]
variables {l m : Type} [line l] [line m]

-- Defining the conditions
def l_perp_alpha (l : Type) [line l] (α : Type) [plane α] : Prop :=
  -- Definition for l being perpendicular to plane α
  sorry

def m_in_beta (m : Type) [line m] (β : Type) [plane β] : Prop :=
  -- Definition for m being contained in plane β
  sorry

def alpha_parallel_beta (α β : Type) [plane α] [plane β] : Prop :=
  -- Definition for planes α and β being parallel
  sorry

def l_parallel_m (l m : Type) [line l] [line m] : Prop :=
  -- Definition for lines l and m being parallel
  sorry

def alpha_perp_beta (α β : Type) [plane α] [plane β] : Prop :=
  -- Definition for planes α and β being perpendicular
  sorry

def l_perp_m (l m : Type) [line l] [line m] : Prop :=
  -- Definition for lines l and m being perpendicular
  sorry

-- Statement of the proof problem
theorem correct_statements_count :
  l_perp_alpha l α ∧ m_in_beta m β →
  (alpha_parallel_beta α β → l_perp_m l m) ∧
  ¬(l_parallel_m l m → l_parallel_beta l β) ∧
  ¬(alpha_perp_beta α β → l_parallel_m l m) ∧
  ¬(l_perp_m l m → l_perp_beta l β) →
  true :=
  sorry

end correct_statements_count_l469_469142


namespace number_of_days_b_worked_l469_469666

variables (d_a : ℕ) (d_c : ℕ) (total_earnings : ℝ)
variables (wage_ratio : ℝ) (wage_c : ℝ) (d_b : ℕ) (wages : ℝ)
variables (total_wage_a : ℝ) (total_wage_c : ℝ) (total_wage_b : ℝ)

-- Given conditions
def given_conditions :=
  d_a = 6 ∧
  d_c = 4 ∧
  wage_c = 95 ∧
  wage_ratio = wage_c / 5 ∧
  wages = 3 * wage_ratio ∧
  total_earnings = 1406 ∧
  total_wage_a = d_a * wages ∧
  total_wage_c = d_c * wage_c ∧
  total_wage_b = d_b * (4 * wage_ratio) ∧
  total_wage_a + total_wage_b + total_wage_c = total_earnings

-- Theorem to prove
theorem number_of_days_b_worked :
  given_conditions d_a d_c total_earnings wage_ratio wage_c d_b wages total_wage_a total_wage_c total_wage_b →
  d_b = 9 :=
by
  intro h
  sorry

end number_of_days_b_worked_l469_469666


namespace desired_salt_percentage_l469_469727

def salt_percent (salt_mass total_mass : ℝ) : ℝ := (salt_mass / total_mass) * 100

def solution_A_salt_percentage : ℝ := 0.40
def solution_B_salt_percentage : ℝ := 0.90

def solution_A_mass : ℝ := 28
def solution_B_mass : ℝ := 112

def total_salt_mass : ℝ := solution_A_salt_percentage * solution_A_mass + solution_B_salt_percentage * solution_B_mass

def total_mixture_mass : ℝ := solution_A_mass + solution_B_mass

theorem desired_salt_percentage :
  salt_percent total_salt_mass total_mixture_mass = 80 :=
by
  sorry

end desired_salt_percentage_l469_469727


namespace product_of_roots_l469_469092

theorem product_of_roots (a b c : ℤ) (h_eqn : a = 12 ∧ b = 60 ∧ c = -720) :
  (c : ℚ) / a = -60 :=
by sorry

end product_of_roots_l469_469092


namespace lcm_first_ten_l469_469410

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469410


namespace every_nat_is_fibonacci_or_sum_of_distinct_fibonacci_l469_469974

def fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| (n+2) := fibonacci n + fibonacci (n+1)

def is_fibonacci (n : ℕ) : Prop :=
  ∃ k : ℕ, fibonacci k = n

def can_be_sum_of_distinct_fibonacci (n : ℕ) : Prop :=
  ∀ (given_fib : ℕ → ℕ), (∀ m : ℕ, given_fib m = fibonacci m) → 
  ∃ (S : finset ℕ), (∀ m ∈ S, given_fib m ∈ finset.univ) ∧ 
  n = S.sum (λ m, given_fib m)

theorem every_nat_is_fibonacci_or_sum_of_distinct_fibonacci (n : ℕ) :
  is_fibonacci n ∨ can_be_sum_of_distinct_fibonacci n :=
by
  -- Proof goes here
  sorry

end every_nat_is_fibonacci_or_sum_of_distinct_fibonacci_l469_469974


namespace at_least_one_not_less_than_two_l469_469954

theorem at_least_one_not_less_than_two (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c):
  (a + 1 / b) ≥ 2 ∨ (b + 1 / c) ≥ 2 ∨ (c + 1 / a) ≥ 2 :=
sorry

end at_least_one_not_less_than_two_l469_469954


namespace ratio_of_adults_to_children_closest_to_one_l469_469316

theorem ratio_of_adults_to_children_closest_to_one (a c : ℕ) 
  (h₁ : 25 * a + 12 * c = 1950) 
  (h₂ : a ≥ 1) 
  (h₃ : c ≥ 1) : (a : ℚ) / (c : ℚ) = 27 / 25 := 
by 
  sorry

end ratio_of_adults_to_children_closest_to_one_l469_469316


namespace find_k_l469_469218

theorem find_k (k : ℝ) :
  (∀ x : ℝ, 1 ≤ x ∧ x ≤ 3 ↔ |k * x - 4| ≤ 2) → k = 2 :=
by
  sorry

end find_k_l469_469218


namespace rate_per_sqm_is_correct_l469_469323

-- Definitions of the problem conditions
def room_length : ℝ := 10
def room_width : ℝ := 7
def room_height : ℝ := 5

def door_width : ℝ := 1
def door_height : ℝ := 3

def window1_width : ℝ := 2
def window1_height : ℝ := 1.5
def window2_width : ℝ := 1
def window2_height : ℝ := 1.5

def number_of_doors : ℕ := 2
def number_of_window2 : ℕ := 2

def total_cost : ℝ := 474

-- Our goal is to prove this rate
def expected_rate_per_sqm : ℝ := 3

-- Wall area calculations
def wall_area : ℝ :=
  2 * (room_length * room_height) + 2 * (room_width * room_height)

def doors_area : ℝ :=
  number_of_doors * (door_width * door_height)

def window1_area : ℝ :=
  window1_width * window1_height

def window2_area : ℝ :=
  number_of_window2 * (window2_width * window2_height)

def total_unpainted_area : ℝ :=
  doors_area + window1_area + window2_area

def paintable_area : ℝ :=
  wall_area - total_unpainted_area

-- Proof goal
theorem rate_per_sqm_is_correct : total_cost / paintable_area = expected_rate_per_sqm :=
by
  sorry

end rate_per_sqm_is_correct_l469_469323


namespace find_m_value_l469_469215

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end find_m_value_l469_469215


namespace coefficient_x3_l469_469989

theorem coefficient_x3 (n : ℕ) (h : (5 - 3 : ℤ) ^ n - 2 ^ n = 240) :
  binomial n 3 * 5^3 * (-3)^(n-3) = binomial n 3 * 5^3 * (-3)^(n-3) :=
by sorry

end coefficient_x3_l469_469989


namespace polynomial_inequality_l469_469937

theorem polynomial_inequality (P : ℝ[X])
  (h_nonneg_coeffs : ∀ n, 0 ≤ P.coeff n)
  (h_initial : P.eval 1 * P.eval 1 ≥ 1) :
  ∀ x > 0, P.eval x * P.eval (1 / x) ≥ 1 :=
begin
  sorry -- the proof is omitted as per the instructions
end

end polynomial_inequality_l469_469937


namespace sqrt_floor_square_l469_469077

theorem sqrt_floor_square {x : ℝ} (hx : 27 = x) (h1 : sqrt 25 < sqrt x) (h2 : sqrt x < sqrt 36) :
  (⌊sqrt x⌋.to_real)^2 = 25 :=
by {
  have hsqrt : 5 < sqrt x ∧ sqrt x < 6, by {
    split; linarith,
  },
  have h_floor_sqrt : ⌊sqrt x⌋ = 5, by {
    exact int.floor_eq_iff.mpr ⟨int.lt_floor_add_one.mpr hsqrt.2, hsqrt.1⟩,
  },
  rw h_floor_sqrt,
  norm_num,
  sorry  -- proof elided
}

end sqrt_floor_square_l469_469077


namespace investment_share_l469_469969

variable (P_investment Q_investment : ℝ)

theorem investment_share (h1 : Q_investment = 60000) (h2 : P_investment / Q_investment = 2 / 3) : P_investment = 40000 := by
  sorry

end investment_share_l469_469969


namespace yen_checking_account_l469_469030

theorem yen_checking_account (savings : ℕ) (total : ℕ) (checking : ℕ) (h1 : savings = 3485) (h2 : total = 9844) (h3 : checking = total - savings) :
  checking = 6359 :=
by
  rw [h1, h2] at h3
  exact h3

end yen_checking_account_l469_469030


namespace alternating_colors_probability_l469_469691

def box_contains_five_white_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 1) = 5
def box_contains_five_black_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 0) = 5
def balls_drawn_one_at_a_time : Prop := true -- This condition is trivially satisfied without more specific constraints

theorem alternating_colors_probability (h1 : box_contains_five_white_balls) (h2 : box_contains_five_black_balls) (h3 : balls_drawn_one_at_a_time) :
  ∃ p : ℚ, p = 1 / 126 :=
sorry

end alternating_colors_probability_l469_469691


namespace least_common_multiple_1_to_10_l469_469490

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469490


namespace product_nonreal_roots_l469_469800

noncomputable def polynomial : Polynomial ℂ := Polynomial.X^5 - 5 * Polynomial.X^4 + 10 * Polynomial.X^3 - 10 * Polynomial.X^2 + 5 * Polynomial.X - 243

theorem product_nonreal_roots : 
  (∏ (x : ℂ) in (Polynomial.roots polynomial).toFinset.filter (λ x, x ≠ conj x), x) = (1 + root 5 242) ^ 4 := 
sorry

end product_nonreal_roots_l469_469800


namespace Vermont_clicked_ads_l469_469362

-- Definitions based on conditions
def ads_first : ℕ := 18
def ads_second : ℕ := 2 * ads_first
def ads_third : ℕ := (ads_second ^ 2) / 6
def ads_fourth : ℕ := (5 * (ads_first + ads_second + ads_third)) / 8
def ads_fifth : ℕ := 15 + 2 * Int.floor (Real.sqrt ads_third)
def ads_sixth : ℕ := (ads_first + ads_second + ads_third) - 42

def total_ads : ℕ := ads_first + ads_second + ads_third + ads_fourth + ads_fifth + ads_sixth
def clicked_ads : ℕ := (3 * total_ads) / 5

-- Theorem statement
theorem Vermont_clicked_ads : clicked_ads = 426 := by
  -- This is where the proof would go
  sorry

end Vermont_clicked_ads_l469_469362


namespace correct_proposition_l469_469658

-- Definitions of the propositions
def propositionA (T1 T2 : Triangle) (h1 : isRightTriangle T1) (h2 : isRightTriangle T2)
  (h3 : ∠T1.acute1 = ∠T2.acute1 ∧ ∠T1.acute2 = ∠T2.acute2) : Prop :=
  T1 ≅ T2

def propositionB (T1 T2 : Triangle) (h1 : isRightTriangle T1) (h2 : isRightTriangle T2)
  (h3 : length T1.leg1 = length T2.leg1 ∧ length T1.leg2 = length T2.leg2) : Prop :=
  T1 ≅ T2

def propositionC (P : Parallelogram) : Prop :=
  ∠P.angle1 = ∠P.angle2

def propositionD (Q : Quadrilateral) (h1 : Q.oppositeSidesParallel) (h2 : Q.oppositeSidesEqual) : Prop :=
  isParallelogram Q

-- Theorem statement confirming only Proposition B is true
theorem correct_proposition : 
  ∃ (T1 T2 : Triangle) (h1 : isRightTriangle T1) (h2 : isRightTriangle T2) (h3 : length T1.leg1 = length T2.leg1 ∧ length T1.leg2 = length T2.leg2), 
    (propositionB T1 T2 h1 h2 h3) := 
    sorry

end correct_proposition_l469_469658


namespace least_common_multiple_of_first_ten_integers_l469_469377

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469377


namespace max_ounces_amber_can_get_l469_469037

theorem max_ounces_amber_can_get :
  let money := 7
  let candy_cost := 1
  let candy_ounces := 12
  let chips_cost := 1.40
  let chips_ounces := 17
  let max_ounces := max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces)
  max_ounces = 85 := 
by
  sorry

end max_ounces_amber_can_get_l469_469037


namespace perimeter_of_equilateral_figure_l469_469061

theorem perimeter_of_equilateral_figure :
  ∀ (A B C D E F G H I J : Type) 
  [equilateral_triangle ABC] [equilateral_triangle ADE] [equilateral_triangle EFG]
  [midpoint D A C] [midpoint G A E]
  [square EHIJ]
  (h1: distance A B = 6)
  (h2: ∀ _ _ _ [equilateral_triangle _], equilateral _ _ _ = true)
  (h3: ∀ _ _ _ [midpoint _ _ _], midpoint_length _ _ _ = true),
  perimeter A B C D E F G H I J = 37.5 :=
by sorry

end perimeter_of_equilateral_figure_l469_469061


namespace lcm_first_ten_numbers_l469_469582

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469582


namespace farmer_spending_l469_469006

theorem farmer_spending (X : ℝ) (hc : 0.80 * X + 0.60 * X = 49) : X = 35 := 
by
  sorry

end farmer_spending_l469_469006


namespace price_of_each_armchair_l469_469252

theorem price_of_each_armchair
  (sofa_price : ℕ)
  (coffee_table_price : ℕ)
  (total_invoice : ℕ)
  (num_armchairs : ℕ)
  (h_sofa : sofa_price = 1250)
  (h_coffee_table : coffee_table_price = 330)
  (h_invoice : total_invoice = 2430)
  (h_num_armchairs : num_armchairs = 2) :
  (total_invoice - (sofa_price + coffee_table_price)) / num_armchairs = 425 := 
by 
  sorry

end price_of_each_armchair_l469_469252


namespace least_common_multiple_of_first_ten_integers_l469_469373

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469373


namespace ways_A_and_B_same_community_l469_469122

-- Define the number of medical staff members and communities
def num_medical_staff : ℕ := 4
def num_communities : ℕ := 3

-- Define the condition that each community must have at least one person
def each_community_has_at_least_one_person : Prop :=
  ∀ (assignments : Fin num_medical_staff → Fin num_communities), 
    (∀ (c : Fin num_communities), ∃ (p : Fin num_medical_staff), assignments p = c)

-- Define the main theorem statement
theorem ways_A_and_B_same_community :
  (∃ (assignments : Fin num_medical_staff → Fin num_communities), 
    each_community_has_at_least_one_person assignments ∧ 
    (assignments 0 = assignments 1) ∧ 
    ∃ num_ways : ℕ, num_ways = 6) :=
  sorry

end ways_A_and_B_same_community_l469_469122


namespace count_positive_numbers_with_cube_roots_less_than_20_l469_469866

theorem count_positive_numbers_with_cube_roots_less_than_20 :
  {n : ℕ | n > 0 ∧ ∃ x, x = n ∧ x^(1/3) < 20}.card = 8000 := 
sorry

end count_positive_numbers_with_cube_roots_less_than_20_l469_469866


namespace number_of_good_integers_l469_469296

open Nat

theorem number_of_good_integers
  (p : ℕ) (hp_prime : Prime p) (hp_ge_3 : p ≥ 3) :
  ∃ c : ℝ, ∀ n : ℕ, n > 0 ∧ p ∣ n! + 1 → n ≤ c * p ^ (2 / 3) :=
sorry

end number_of_good_integers_l469_469296


namespace mass_percentage_Cl_in_BaCl2_is_34_04_l469_469795

def molar_mass_Ba : ℝ := 137.327
def molar_mass_Cl : ℝ := 35.453

def molar_mass_BaCl2 : ℝ := molar_mass_Ba + 2 * molar_mass_Cl
def mass_Cl_in_BaCl2 : ℝ := 2 * molar_mass_Cl
def mass_percentage_Cl_in_BaCl2 : ℝ := (mass_Cl_in_BaCl2 / molar_mass_BaCl2) * 100

theorem mass_percentage_Cl_in_BaCl2_is_34_04 : mass_percentage_Cl_in_BaCl2 = 34.04 := by
  sorry

end mass_percentage_Cl_in_BaCl2_is_34_04_l469_469795


namespace least_divisible_1_to_10_l469_469527

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469527


namespace simplify_fraction_l469_469306

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) : (x + 1) / (x^2 + 2 * x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l469_469306


namespace jason_correct_answers_l469_469220

-- Definitions for the problem
def total_problems : ℕ := 12
def points_correct : ℕ := 4
def points_incorrect : ℕ := -1
def jason_score : ℤ := 33

-- Prove the number of correct answers Jason had
theorem jason_correct_answers : ∃ c w : ℕ, 
  c + w = total_problems ∧ 
  points_correct * (c : ℤ) + points_incorrect * (w : ℤ) = jason_score ∧ 
  c = 9 := 
by
  sorry

end jason_correct_answers_l469_469220


namespace number_of_even_runs_sequences_of_length_16_l469_469063

theorem number_of_even_runs_sequences_of_length_16 : 
    let seq_count : ℕ → ℕ := λ n, if n < 2 then if n = 0 then 1 else 0 else 2 * seq_count (n - 2)
    seq_count 16 = 256
:= by
  sorry

end number_of_even_runs_sequences_of_length_16_l469_469063


namespace lcm_first_ten_numbers_l469_469597

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469597


namespace sqrt_abs_is_limited_growth_function_sin_sq_is_limited_growth_function_l469_469890

open Real

def limited_growth_function (f : ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ ∀ x, f(x + a) ≤ f(x) + b

theorem sqrt_abs_is_limited_growth_function : limited_growth_function (λ x, sqrt (abs x)) :=
sorry

theorem sin_sq_is_limited_growth_function : limited_growth_function (λ x, sin (x^2)) :=
sorry

end sqrt_abs_is_limited_growth_function_sin_sq_is_limited_growth_function_l469_469890


namespace least_common_multiple_first_ten_integers_l469_469574

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469574


namespace nut_weights_l469_469002

noncomputable def part_weights (total_weight : ℝ) (total_parts : ℝ) : ℝ :=
  total_weight / total_parts

theorem nut_weights
  (total_weight : ℝ)
  (parts_almonds parts_walnuts parts_cashews ratio_pistachios_to_almonds : ℝ)
  (total_parts_without_pistachios total_parts_with_pistachios weight_per_part : ℝ)
  (weights_almonds weights_walnuts weights_cashews weights_pistachios : ℝ) :
  parts_almonds = 5 →
  parts_walnuts = 3 →
  parts_cashews = 2 →
  ratio_pistachios_to_almonds = 1 / 4 →
  total_parts_without_pistachios = parts_almonds + parts_walnuts + parts_cashews →
  total_parts_with_pistachios = total_parts_without_pistachios + (parts_almonds * ratio_pistachios_to_almonds) →
  weight_per_part = total_weight / total_parts_with_pistachios →
  weights_almonds = parts_almonds * weight_per_part →
  weights_walnuts = parts_walnuts * weight_per_part →
  weights_cashews = parts_cashews * weight_per_part →
  weights_pistachios = (parts_almonds * ratio_pistachios_to_almonds) * weight_per_part →
  total_weight = 300 →
  weights_almonds = 133.35 ∧
  weights_walnuts = 80.01 ∧
  weights_cashews = 53.34 ∧
  weights_pistachios = 33.34 :=
by
  intros
  sorry

end nut_weights_l469_469002


namespace prime_square_mod_12_l469_469297

theorem prime_square_mod_12 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) : p^2 % 12 = 1 := 
by
  sorry

end prime_square_mod_12_l469_469297


namespace min_avg_less_than_old_record_l469_469812

variable old_record_avg : ℕ := 287
variable num_players : ℕ := 4
variable num_rounds : ℕ := 10
variable points_scored_9_rounds : ℕ := 10440

theorem min_avg_less_than_old_record:
  let total_points_needed := old_record_avg * num_players * num_rounds in
  let points_needed_final_round := total_points_needed - points_scored_9_rounds in
  let min_avg_final_round := points_needed_final_round / num_players in
  min_avg_final_round = old_record_avg - 27 :=
by
  sorry

end min_avg_less_than_old_record_l469_469812


namespace lcm_first_ten_l469_469404

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469404


namespace line_and_circle_intersect_l469_469855

-- Define the line and the circle
def line (x y : ℝ) := y = x + 1
def circle (x y : ℝ) := x^2 + y^2 = 1

-- Define a predicate for the line and circle intersecting
def intersecting (line : ℝ → ℝ → Prop) (circle : ℝ → ℝ → Prop) : Prop :=
  ∃ (x y : ℝ), line x y ∧ circle x y

-- The main statement we want to prove
theorem line_and_circle_intersect (l : ℝ → ℝ → Prop) (C : ℝ → ℝ → Prop) 
  (hl : ∀ x y, l x y ↔ y = x + 1) 
  (hC : ∀ x y, C x y ↔ x^2 + y^2 = 1) : 
  intersecting l C :=
by
  sorry

end line_and_circle_intersect_l469_469855


namespace least_common_multiple_of_first_ten_l469_469607

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469607


namespace circle_equation_l469_469790

theorem circle_equation : 
  ∃ (x y : ℝ), (∃ k : ℝ, k = √2) ∧ (center : ℝ × ℝ) ∧ (pt : ℝ × ℝ)
    (center = (2, -1)) ∧ (pt = (3, 0)) ∧
    (x - center.1)^2 + (y - center.2)^2 = (k)^2 :=
sorry

end circle_equation_l469_469790


namespace smallest_positive_period_of_f_value_of_a_for_minimum_value_l469_469845

noncomputable theory

def f (x a : ℝ) : ℝ := 2 * sin (2 * x - π / 6) + a

theorem smallest_positive_period_of_f :
  ∀ (a : ℝ), ∃ T > 0, ∀ x : ℝ, f (x + T) a = f x a :=
by sorry

theorem value_of_a_for_minimum_value :
  ∃ (a : ℝ), (∀ x ∈ Icc (0 : ℝ) (π / 2), f x a ≥ -2) :=
by sorry

end smallest_positive_period_of_f_value_of_a_for_minimum_value_l469_469845


namespace probability_units_digit_l469_469713

noncomputable def probability_units_digit_condition : ℚ :=
  let total_outcomes := 10
  let favorable_outcomes := 5
  favorable_outcomes / total_outcomes

theorem probability_units_digit {n : ℕ} (hl : 10000 ≤ n) (hr : n ≤ 99999) :
  probability_units_digit_condition = 1 / 2 :=
by
  sorry

end probability_units_digit_l469_469713


namespace find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469103

theorem find_x_such_that_sqrt_5x_plus_9_eq_12 : ∀ x : ℝ, sqrt (5 * x + 9) = 12 → x = 27 := 
by
  intro x
  sorry

end find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469103


namespace least_common_multiple_first_ten_integers_l469_469567

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469567


namespace units_digit_1_to_99_is_5_l469_469654

noncomputable def units_digit_of_product_of_odds : ℕ :=
  let seq := List.range' 1 99;
  (seq.filter (λ n => n % 2 = 1)).prod % 10

theorem units_digit_1_to_99_is_5 : units_digit_of_product_of_odds = 5 :=
by sorry

end units_digit_1_to_99_is_5_l469_469654


namespace second_expression_l469_469318

theorem second_expression (a x : ℕ) (h₁ : (2 * a + 16 + x) / 2 = 79) (h₂ : a = 30) : x = 82 := by
  sorry

end second_expression_l469_469318


namespace plane_split_into_regions_l469_469771

theorem plane_split_into_regions : 
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  ∃ regions : ℕ, regions = 7 :=
by
  let line1 (x : ℝ) := 3 * x
  let line2 (x : ℝ) := (1 / 3) * x
  let line3 (x : ℝ) := 4 * x
  existsi 7
  sorry

end plane_split_into_regions_l469_469771


namespace max_red_bulbs_l469_469686

theorem max_red_bulbs (n r b : ℕ) (h₀: n = 50) 
  (h₁ : r + b = n) 
  (h₂ : ∀ i ∈ finset.range n, (i % 3 ≠ 0) → (r > 0 → (finset.mem (set.range r) i → finset.mem (set.range b) (i + 1) ∨ finset.mem (set.range b) (i - 1)))) : 
  r = 33 :=
by
  sorry

end max_red_bulbs_l469_469686


namespace r_plus_s_l469_469999

-- Definitions of points P and Q based on the conditions given
def P : ℝ × ℝ := (9, 0)
def Q : ℝ × ℝ := (0, 6)

-- Definition of the line equation
def line_eq (x : ℝ) : ℝ := -2/3 * x + 6

-- T is a point on the line segment PQ
variable {T : ℝ × ℝ}
def on_line_segment (T : ℝ × ℝ) : Prop := T.2 = line_eq T.1

-- The area of triangle POQ is four times the area of triangle TOP
#check sorry -- proof of the statement (skipping proof steps)

theorem r_plus_s (r s : ℝ) (hT : on_line_segment (r, s)) (h_area : ½ * 9 * 6 = 4 * (½ * 9 * s)) : r + s = 8.25 :=
by sorry

end r_plus_s_l469_469999


namespace lateral_surface_area_of_prism_l469_469089

theorem lateral_surface_area_of_prism (h : ℝ) (angle : ℝ) (h_pos : 0 < h) (angle_eq : angle = 60) :
  ∃ S : ℝ, S = 6 * h^2 :=
by
  sorry

end lateral_surface_area_of_prism_l469_469089


namespace lcm_first_ten_integers_l469_469617

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469617


namespace lcm_first_ten_l469_469405

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469405


namespace least_common_multiple_1_to_10_l469_469502

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469502


namespace article_cost_l469_469669

theorem article_cost {c : ℝ} (h₁ : c = 75) (h₂ : ∀ x : ℝ, x > 0 → (x * 0.20)) :
  let increased_cost := c + (c * 0.20)
  let decreased_cost := increased_cost - (increased_cost * 0.20)
  decreased_cost = 72 := 
by
  sorry

end article_cost_l469_469669


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469397

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469397


namespace solve_problem_l469_469067

noncomputable def problem_statement : Prop :=
  ∃ (a b c : ℕ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    2 * real.sqrt (real.cbrt 7 - real.cbrt 6) = real.cbrt a - real.cbrt b + real.cbrt c ∧ 
    a + b + c = 54

theorem solve_problem : problem_statement := sorry

end solve_problem_l469_469067


namespace distance_to_airport_l469_469773

theorem distance_to_airport
  (t : ℝ)
  (d : ℝ)
  (h1 : 45 * (t + 1) + 20 = d)
  (h2 : d - 65 = 65 * (t - 1))
  : d = 390 := by
  sorry

end distance_to_airport_l469_469773


namespace hyperbola_eccentricity_l469_469854

theorem hyperbola_eccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0)
    (h₃ : ∀ x y : ℝ, y = b/a * x → y ≠ -b/a * x) :
    let c := sqrt (a^2 + b^2) in
    let e := c / a in 
    e = 2 :=
by
    sorry

end hyperbola_eccentricity_l469_469854


namespace probability_of_alternating_draws_l469_469699

theorem probability_of_alternating_draws :
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls in
  let successful_orders : ℕ := 2,
      total_arrangements : ℕ := Nat.choose total_balls white_balls,
      probability : ℚ := successful_orders / total_arrangements in
  probability = 1 / 126 :=
by
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls
  let successful_orders : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let probability : ℚ := successful_orders / total_arrangements
  sorry

end probability_of_alternating_draws_l469_469699


namespace lcm_first_ten_numbers_l469_469589

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469589


namespace seashells_total_correct_l469_469183

-- Define the initial counts for Henry, John, and Adam.
def initial_seashells_Henry : ℕ := 11
def initial_seashells_John : ℕ := 24
def initial_seashells_Adam : ℕ := 17

-- Define the total initial seashells collected by all.
def total_initial_seashells : ℕ := 83

-- Calculate Leo's initial seashells.
def initial_seashells_Leo : ℕ := total_initial_seashells - (initial_seashells_Henry + initial_seashells_John + initial_seashells_Adam)

-- Define the changes occurred when they returned home.
def extra_seashells_Henry : ℕ := 3
def given_away_seashells_John : ℕ := 5
def percentage_given_away_Leo : ℕ := 40
def extra_seashells_Leo : ℕ := 5

-- Define the final number of seashells each person has.
def final_seashells_Henry : ℕ := initial_seashells_Henry + extra_seashells_Henry
def final_seashells_John : ℕ := initial_seashells_John - given_away_seashells_John
def given_away_seashells_Leo : ℕ := (initial_seashells_Leo * percentage_given_away_Leo) / 100
def final_seashells_Leo : ℕ := initial_seashells_Leo - given_away_seashells_Leo + extra_seashells_Leo
def final_seashells_Adam : ℕ := initial_seashells_Adam

-- Define the total number of seashells they have now.
def total_final_seashells : ℕ := final_seashells_Henry + final_seashells_John + final_seashells_Leo + final_seashells_Adam

-- Proposition that asserts the total number of seashells is 74.
theorem seashells_total_correct :
  total_final_seashells = 74 :=
sorry

end seashells_total_correct_l469_469183


namespace range_of_phi_l469_469330

theorem range_of_phi (phi : ℝ) : 
  (0 < phi ∧ phi < π / 2) ∧
  (∀ x, 0 ≤ x ∧ x ≤ π / 3 → derivative (λ x, real.sin (2 * x - 2 * phi)) x > 0) ∧
  (∃ k : ℤ, - π / 3 < (phi - π / 2) + k * π ∧ (phi - π / 2) + k * π < - π / 12)
  ↔ (π / 6 < phi ∧ phi ≤ π / 4) := 
sorry

end range_of_phi_l469_469330


namespace man_swims_speed_l469_469014

theorem man_swims_speed (v_m v_s : ℝ) (h_downstream : 28 = (v_m + v_s) * 2) (h_upstream : 12 = (v_m - v_s) * 2) : v_m = 10 := 
by sorry

end man_swims_speed_l469_469014


namespace least_common_multiple_1_to_10_l469_469451

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469451


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469398

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469398


namespace solve_for_x_l469_469096

theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → (x = 27) := by
  intro h
  sorry

end solve_for_x_l469_469096


namespace possible_slopes_of_line_l469_469717

theorem possible_slopes_of_line
  (m : ℝ)
  (intersects : ∃ x y : ℝ, y = m*x - 3 ∧ 4*x^2 + 25*y^2 = 100) :
  m ∈ set.Iic (-real.sqrt (4 / 55)) ∪ set.Ici (real.sqrt (4 / 55)) :=
sorry

end possible_slopes_of_line_l469_469717


namespace prob1_l469_469680

variable {α : Type*} [AddCommGroup α] [Module ℤ α]

def arithmetic_seq (d a₁ : α) (n : ℕ) : α := a₁ + n * d
def sum_arithmetic_seq (a₁ d : α) (n : ℕ) : α :=
  n * a₁ + n * (n - 1) / 2 * d

theorem prob1 (a₄ a₅ : ℤ) (d : ℤ) :
  (a₄ = 4) → (a₄ + a₅ + (a₅ + 2 * d) = 15) →
  (arithmetic_seq d ((4 - 3 * d) : ℤ) 9 =
  10 + 9 * 1) → sum_arithmetic_seq ((4 - 3*d) : ℤ) d 10 = 55 := 
by 
  intros _ _
  sorry

end prob1_l469_469680


namespace least_positive_integer_divisible_by_first_ten_l469_469485

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469485


namespace greatest_integer_2e_minus_5_l469_469948

noncomputable def e : ℝ := 2.718

theorem greatest_integer_2e_minus_5 : ⌊2 * e - 5⌋ = 0 :=
by
  -- This is a placeholder for the actual proof. 
  sorry

end greatest_integer_2e_minus_5_l469_469948


namespace mean_proportional_81_100_l469_469090

/- Define the mean proportional (geometric mean) -/
def mean_proportional (A B : ℕ) : ℕ := Int.sqrt (A * B)

/- Prove that the mean proportional between 81 and 100 is 90 -/
theorem mean_proportional_81_100 : mean_proportional 81 100 = 90 := by
  sorry

end mean_proportional_81_100_l469_469090


namespace cross_section_area_l469_469017

noncomputable def prism_cross_section_area 
    (a b c : ℝ) (d : ℝ) 
    (side_length : ℝ) : Prop :=
  ∀ (vertices : list (ℝ × ℝ × ℝ)),
    -- Conditions: vertices of the base, the plane equation,
    -- and the orientation are implicitly given in the context of 
    -- the vertices list and plane constants.
    -- Given plane equation: a * x + b * y + c * z = d
    -- Given side length of base: side_length
    side_length = 12 ∧
    a = 3 ∧ b = -5 ∧ c = 2 ∧ d = 30 ∧
    -- The maximum area of cross-section should be 135
    ∃ E F G H : ℝ × ℝ × ℝ,
      -- Definitions of E, F, G, H as per the solution's information:
      (E = (6 * real.sqrt 2 * real.cos θ, 6 * real.sqrt 2 * real.sin θ, (45 * real.sqrt 2 * real.sin θ - 30 * real.sqrt 2 * real.cos θ + 30) / 2) ∧
       F = (-6 * real.sqrt 2 * real.sin θ, 6 * real.sqrt 2 * real.cos θ, (45 * real.sqrt 2 * real.cos θ + 30 * real.sqrt 2 * real.sin θ + 30) / 2) ∧
       G = (-6 * real.sqrt 2 * real.cos θ, -6 * real.sqrt 2 * real.sin θ, (-45 * real.sqrt 2 * real.sin θ + 30 * real.sqrt 2 * real.cos θ + 30) / 2) ∧
       H = (6 * real.sqrt 2 * real.sin θ, -6 * real.sqrt 2 * real.cos θ, (-45 * real.sqrt 2 * real.cos θ - 30 * real.sqrt 2 * real.sin θ + 30) / 2)) ∧
    -- The area calculation check
    (let ME := (72, -135 / 2, 72) in
     let MF := (72 * real.cos θ, -135 / 2 * real.sin θ, 72) in
     (2 * (1 / 2 * (real.sqrt (ME.1 ^ 2 + ME.2 ^ 2 + ME.3 ^ 2)))) = 135)

theorem cross_section_area
    (side_length : ℝ) : prism_cross_section_area 3 (-5) 2 30 side_length :=
by sorry

end cross_section_area_l469_469017


namespace papers_left_after_giving_away_l469_469281

variable (x : ℕ)

-- Given conditions:
def sheets_in_desk : ℕ := 50
def sheets_in_backpack : ℕ := 41
def total_initial_sheets := sheets_in_desk + sheets_in_backpack

-- Prove that Maria has 91 - x sheets left after giving away x sheets
theorem papers_left_after_giving_away (h : total_initial_sheets = 91) : 
  ∀ d b : ℕ, d = sheets_in_desk → b = sheets_in_backpack → 91 - x = total_initial_sheets - x :=
by
  sorry

end papers_left_after_giving_away_l469_469281


namespace least_common_multiple_first_ten_l469_469559

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469559


namespace probability_of_alternating_draws_l469_469698

theorem probability_of_alternating_draws :
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls in
  let successful_orders : ℕ := 2,
      total_arrangements : ℕ := Nat.choose total_balls white_balls,
      probability : ℚ := successful_orders / total_arrangements in
  probability = 1 / 126 :=
by
  let white_balls : ℕ := 5,
      black_balls : ℕ := 5,
      total_balls : ℕ := white_balls + black_balls
  let successful_orders : ℕ := 2
  let total_arrangements : ℕ := Nat.choose total_balls white_balls
  let probability : ℚ := successful_orders / total_arrangements
  sorry

end probability_of_alternating_draws_l469_469698


namespace sine_of_2012_eq_neg_sine_of_32_l469_469051

theorem sine_of_2012_eq_neg_sine_of_32 :
  sin (2012 * real.pi / 180) = -sin (32 * real.pi / 180) :=
by
  sorry

end sine_of_2012_eq_neg_sine_of_32_l469_469051


namespace sum_first_10_terms_l469_469244

def geometric_sequence (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r^n) / (1 - r)

theorem sum_first_10_terms (a r : ℝ) (S20 S30 : ℝ) 
  (h1 : geometric_sequence a r 20 = 30) 
  (h2 : geometric_sequence a r 30 = 70) : 
  geometric_sequence a r 10 = 10 :=
by
  sorry

end sum_first_10_terms_l469_469244


namespace least_common_multiple_of_first_ten_positive_integers_l469_469456

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469456


namespace problem1_problem2_problem3_l469_469828

-- Definition of S(n) and residue array
def S (n : ℕ) : Type := Array (Array ℝ)
def is_residue_array (A A' : S n) : Prop :=
  ∀ j, (∀ i, A[i][j] = 0 → A'[i][j] = 0) ∧ 
       (∃ i, A[i][j] ≠ 0 → A'[i][j] = A[i][j] ∧ A'[(1-i)%2][j] = 0)

-- Row and Column sums
def R (i : ℕ) (A : S n) : ℝ := (A[i].toList).sum
def C (j : ℕ) (A : S n) : ℝ := (Array.map (fun row => row[j]) A).toList.sum

-- Problem 1
theorem problem1 : 
  let A := #[#[0.1, 0.1, 1], #[0, 0, 0.1]] in
  ∃ A', is_residue_array A A' ∧ R 0 A' = R 1 A' := 
sorry

-- Problem 2
theorem problem2 : 
  ∀ (A : S 2), 
  (∀ j, C j A = 1) → ∃ A', is_residue_array A A' ∧ 
                                R 0 A' ≤ 2/3 ∧ 
                                R 1 A' ≤ 2/3 := 
sorry

-- Problem 3
theorem problem3 : 
  ∀ (A : S 23), 
  (∀ j, C j A = 1) → ∃ A', is_residue_array A A' ∧ 
                                R 0 A' ≤ 6 ∧ 
                                R 1 A' ≤ 6 := 
sorry

end problem1_problem2_problem3_l469_469828


namespace length_of_shorter_leg_l469_469894

variable (h x : ℝ)

theorem length_of_shorter_leg 
  (h_med : h / 2 = 5 * Real.sqrt 3) 
  (hypotenuse_relation : h = 2 * x) 
  (median_relation : h / 2 = h / 2) :
  x = 5 := by sorry

end length_of_shorter_leg_l469_469894


namespace correct_response_l469_469083

def conversation_response (options: list string) :=
  "better luck next time" ∈ options

theorem correct_response :
  let context := ["The first speaker is expressing disappointment at not winning the game.",
                  "The second speaker's response should be sympathetic or encouraging.",
                  "Options": ["you are right", "best wishes", "better luck next time", "congratulations"]]
  show conversation_response context[2] :=
sorry

end correct_response_l469_469083


namespace determine_a_l469_469191

-- Define the conditions of the problem.
def linear_in_x_and_y (a : ℝ) : Prop :=
  (a - 2) * (x : ℝ)^(abs a - 1) + 3 * (y : ℝ) = 1

-- Prove that a = -2 under the condition defined.
theorem determine_a (a : ℝ) (h : ∀ x y : ℝ, linear_in_x_and_y a) : a = -2 :=
sorry

end determine_a_l469_469191


namespace find_x_l469_469108

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 12) : x = 27 :=
begin
  sorry
end

end find_x_l469_469108


namespace probability_of_odd_product_greater_than_14_l469_469310

open Classical

noncomputable theory

-- Definitions according to the conditions
def balls := {1, 2, 3, 4, 5, 6}
def is_odd (n : ℕ) : Prop := n % 2 = 1
def random_choice (s : Set ℕ) : Set (ℕ × ℕ) := {p | p.1 ∈ s ∧ p.2 ∈ s}

-- Conditions expressed in Lean
def product_odd_and_greater_than_14 (p : ℕ × ℕ) : Prop :=
  is_odd (p.1) ∧ is_odd (p.2) ∧ p.1 * p.2 > 14

-- Probability calculation
def favorable_outcomes : Set (ℕ × ℕ) := (random_choice balls).filter product_odd_and_greater_than_14
def total_outcomes : Set (ℕ × ℕ) := random_choice balls
def probability := (favorable_outcomes.toFinset.card : ℚ) / total_outcomes.toFinset.card

-- The main theorem statement
theorem probability_of_odd_product_greater_than_14 : probability = 1 / 12 := by
  sorry

end probability_of_odd_product_greater_than_14_l469_469310


namespace moles_of_Iron_sulfate_correct_l469_469797

-- Defining the initial moles of Iron and Sulfuric acid
def moles_of_Iron : ℕ := 2
def moles_of_Sulfuric_acid : ℕ := 2

-- Defining the balanced chemical equation relationship in the form of a 1:1 ratio
def balanced_equation_ratio : ℕ → ℕ := λ x, x

-- The main goal to prove
def moles_of_Iron_sulfate_formed : ℕ :=
  balanced_equation_ratio moles_of_Iron

-- The theorem stating the equivalence
theorem moles_of_Iron_sulfate_correct :
  moles_of_Iron_sulfate_formed = 2 := 
  by
    rw [moles_of_Iron_sulfate_formed]
    rw [balanced_equation_ratio]
    rw [moles_of_Iron]
    -- Assuming the balanced_equation_ratio correctly models the 1:1 relationship:
    -- The equation becomes 2 = 2
    sorry

end moles_of_Iron_sulfate_correct_l469_469797


namespace isosceles_triangle_angle_measure_l469_469744

theorem isosceles_triangle_angle_measure:
  ∀ (α β : ℝ), (α = 112.5) → (2 * β + α = 180) → β = 33.75 :=
by
  intros α β hα h_sum
  sorry

end isosceles_triangle_angle_measure_l469_469744


namespace sum_of_cubes_of_first_n_odd_numbers_l469_469094

theorem sum_of_cubes_of_first_n_odd_numbers (n : ℕ) : 
  ∑ k in Finset.range n, (2 * k + 1)^3 = n^2 * (2 * n^2 - 1) := sorry

end sum_of_cubes_of_first_n_odd_numbers_l469_469094


namespace phase_shift_of_cosine_l469_469798

theorem phase_shift_of_cosine :
  ∀ x : ℝ,
  let y := 2 * real.cos (2 * x + π / 4) in
  phase_shift y = -π / 8 :=
by
  intros
  sorry  -- Proof not required, just the statement.

end phase_shift_of_cosine_l469_469798


namespace chocolates_in_box_l469_469664

noncomputable def total_chocolates : ℕ := 63

-- Define the conditions
def one_third_has_nuts (C : ℕ) : Prop := C / 3 > 0
def one_third_filled_with_caramel (C : ℕ) : Prop := C / 3 > 0
def one_third_neither (C : ℕ) : Prop := C / 3 > 0

def chocolates_eaten_by_alice (caramel_chocolates : ℕ) : Prop :=
    caramel_chocolates ≡ 6

def chocolates_eaten_by_bob (no_nuts_no_caramel_chocolates : ℕ) : Prop :=
    no_nuts_no_caramel_chocolates ≡ 4

def chocolates_eaten_by_claire (nuts_chocolates : ℕ) (no_nuts_no_caramel_chocolates : ℕ) : Prop :=
    nuts_chocolates ≡ 0.7 * (nuts_chocolates + no_nuts_no_caramel_chocolates) ∧
    no_nuts_no_caramel_chocolates ≡ 3

def remaining_chocolates (C remaining : ℕ) : Prop :=
    C - 6 - 4 - 0.7 * (C / 3) - 3 = remaining

-- The proof goal
theorem chocolates_in_box (C : ℕ) (remaining : ℕ) :
    one_third_has_nuts C →
    one_third_filled_with_caramel C →
    one_third_neither C →
    chocolates_eaten_by_alice (C / 3) →
    chocolates_eaten_by_bob (C / 3) →
    chocolates_eaten_by_claire (C / 3) (C / 3) →
    remaining_chocolates C remaining →
    remaining = 36 →
    C = total_chocolates :=
sorry

end chocolates_in_box_l469_469664


namespace determine_a_l469_469192

-- Define the conditions of the problem.
def linear_in_x_and_y (a : ℝ) : Prop :=
  (a - 2) * (x : ℝ)^(abs a - 1) + 3 * (y : ℝ) = 1

-- Prove that a = -2 under the condition defined.
theorem determine_a (a : ℝ) (h : ∀ x y : ℝ, linear_in_x_and_y a) : a = -2 :=
sorry

end determine_a_l469_469192


namespace difference_of_A_and_B_l469_469995

def A : ℕ := (∑ k in Finset.range 19, (2*k + 1) * (2*k + 2)) + 39
def B : ℕ := 1 + (∑ k in Finset.range 18, (2*k + 2) * (2*k + 3)) + 38 * 39

theorem difference_of_A_and_B : |A - B| = 722 := by
  sorry

end difference_of_A_and_B_l469_469995


namespace min_value_at_x_eq_a_possible_a_values_l469_469174

def quadratic_function_y (x a: ℝ) : ℝ :=
  2 * x^2 - 4 * a * x + a^2 + 2 * a + 2

theorem min_value_at_x_eq_a (a: ℝ) : 
  min_value (quadratic_function_y x a) = 3 - (a - 1)^2 := 
by
  sorry

theorem possible_a_values (a: ℝ) :
  ∀ x,
    -1 ≤ x ∧ x ≤ 2 → 
    min_value (quadratic_function_y x a) = 2 → 
    a = 0 ∨ a = 2 ∨ a = -3 - sqrt 7 ∨ a = 4 :=
by
  sorry

end min_value_at_x_eq_a_possible_a_values_l469_469174


namespace alcohol_percentage_new_mixture_l469_469682

namespace AlcoholMixtureProblem

def original_volume : ℝ := 3
def alcohol_percentage : ℝ := 0.33
def additional_water_volume : ℝ := 1
def new_volume : ℝ := original_volume + additional_water_volume
def alcohol_amount : ℝ := original_volume * alcohol_percentage

theorem alcohol_percentage_new_mixture : (alcohol_amount / new_volume) * 100 = 24.75 := by
  sorry

end AlcoholMixtureProblem

end alcohol_percentage_new_mixture_l469_469682


namespace least_common_multiple_of_first_ten_integers_l469_469374

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469374


namespace find_a2_b2_c2_l469_469952

theorem find_a2_b2_c2 (a b c : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0) 
  (h4 : a + b + c = 1) (h5 : a^3 + b^3 + c^3 = a^5 + b^5 + c^5 + 1) : 
  a^2 + b^2 + c^2 = 7 / 5 := 
sorry

end find_a2_b2_c2_l469_469952


namespace amber_max_ounces_l469_469043

-- Define the problem parameters:
def cost_candy : ℝ := 1
def ounces_candy : ℝ := 12
def cost_chips : ℝ := 1.4
def ounces_chips : ℝ := 17
def total_money : ℝ := 7

-- Define the number of bags of each item Amber can buy:
noncomputable def bags_candy := (total_money / cost_candy).to_int
noncomputable def bags_chips  := (total_money / cost_chips).to_int

-- Define the total ounces of each item:
noncomputable def total_ounces_candy := bags_candy * ounces_candy
noncomputable def total_ounces_chips := bags_chips * ounces_chips

-- Problem statement asking to prove Amber gets the most ounces by buying chips:
theorem amber_max_ounces : max total_ounces_candy total_ounces_chips = total_ounces_chips :=
by sorry

end amber_max_ounces_l469_469043


namespace area_of_KLMN_l469_469970

theorem area_of_KLMN (A B C D K L M N : ℝ) 
  (hAKB : equilateral_triangle A K B)
  (hBLC : equilateral_triangle B L C)
  (hCMD : equilateral_triangle C M D)
  (hDNA : equilateral_triangle D N A)
  (h_area_ABCD : (square ABCD).area = 16) :
  (quadrilateral KLMN).area = 32 + 16 * real.sqrt 3 :=
sorry

end area_of_KLMN_l469_469970


namespace least_common_multiple_first_ten_integers_l469_469575

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469575


namespace willams_land_percentage_l469_469667

variable (farm_tax_collected total_tax mr_willam_tax : ℝ)
variable (total_taxable_land : ℝ)

-- Conditions
def farm_tax_collected := 3840
def total_tax := 3840
def mr_willam_tax := 480

-- Problem: Prove that Mr. Willam's land percentage is 12.5%
theorem willams_land_percentage :
  mr_willam_tax / farm_tax_collected = 1 / 8 -> mr_willam_tax / total_tax = 1 / 8 :=
  by sorry

end willams_land_percentage_l469_469667


namespace determine_constants_l469_469786

theorem determine_constants (P Q R : ℚ) :
  (∀ x : ℚ, x ≠ 1 → x ≠ 4 → x ≠ 6 → 
    (x^2 - 5*x + 6) / ((x - 1) * (x - 4) * (x - 6)) =
    P / (x - 1) + Q / (x - 4) + R / (x - 6)) →
  P = 2 / 15 ∧ Q = 1 / 3 ∧ R = 0 :=
by {
  sorry
}

end determine_constants_l469_469786


namespace least_common_multiple_first_ten_integers_l469_469562

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469562


namespace emily_cell_phone_cost_l469_469001

noncomputable def base_cost : ℝ := 25
noncomputable def included_hours : ℝ := 25
noncomputable def cost_per_text : ℝ := 0.1
noncomputable def cost_per_extra_minute : ℝ := 0.15
noncomputable def cost_per_gigabyte : ℝ := 2

noncomputable def emily_texts : ℝ := 150
noncomputable def emily_hours : ℝ := 26
noncomputable def emily_data : ℝ := 3

theorem emily_cell_phone_cost : 
  let texts_cost := emily_texts * cost_per_text
  let extra_minutes_cost := (emily_hours - included_hours) * 60 * cost_per_extra_minute
  let data_cost := emily_data * cost_per_gigabyte
  base_cost + texts_cost + extra_minutes_cost + data_cost = 55 := by
  sorry

end emily_cell_phone_cost_l469_469001


namespace sin_cos_range_l469_469907

theorem sin_cos_range (A B C : ℝ) (h_triangle : A + B + C = Real.pi)
  (h_acute : 0 < A ∧ A < Real.pi / 2 ∧ 0 < B ∧ B < Real.pi / 2 ∧ 0 < C ∧ C < Real.pi / 2)
  (h_arithmetic : 2 * B = A + C) :
  ∀ (y : ℝ), y = sin A - cos (A - C + 2 * B) → y ∈ Ioo (-1) 2 :=
by
  sorry

end sin_cos_range_l469_469907


namespace prime_gt_three_square_mod_twelve_l469_469300

theorem prime_gt_three_square_mod_twelve (p : ℕ) (h_prime: Prime p) (h_gt_three: p > 3) : (p^2) % 12 = 1 :=
by
  sorry

end prime_gt_three_square_mod_twelve_l469_469300


namespace rationalize_denominator_l469_469302

theorem rationalize_denominator (A B C : ℤ) (hA : A = 11) (hB : B = 5) (hC : C = 5) :
  (2 + Real.sqrt 5) / (3 - Real.sqrt 5) = (11 + 5 * Real.sqrt 5) / 4 ∧ A * B * C = 275 := 
by {
  have h1 : (2 + Real.sqrt 5) * (3 + Real.sqrt 5) / ((3 - Real.sqrt 5) * (3 + Real.sqrt 5)) = (11 + 5 * Real.sqrt 5) / 4,
  {
    calc
      (2 + Real.sqrt 5) * (3 + Real.sqrt 5) / ((3 - Real.sqrt 5) * (3 + Real.sqrt 5))
        = (11 + 5 * Real.sqrt 5) / 4 : sorry
  },
  split,
  { exact h1 },
  { rw [hA, hB, hC], norm_num }
}

end rationalize_denominator_l469_469302


namespace least_common_multiple_of_first_ten_positive_integers_l469_469462

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469462


namespace intersection_complement_l469_469888

universe u

def U := Real

def M : Set Real := { x | -2 ≤ x ∧ x ≤ 2 }

def N : Set Real := { x | x * (x - 3) ≤ 0 }

def complement_U (S : Set Real) : Set Real := { x | x ∉ S }

theorem intersection_complement :
  M ∩ (complement_U N) = { x | -2 ≤ x ∧ x < 0 } := by
  sorry

end intersection_complement_l469_469888


namespace find_k_eq_3_l469_469802

theorem find_k_eq_3 (k : ℝ) (h : k ≠ 0) :
  ∀ x : ℝ, (x^2 - k) * (x + k) = x^3 - k * (x^2 + x + 3) → k = 3 :=
by sorry

end find_k_eq_3_l469_469802


namespace tan_diff_l469_469199

theorem tan_diff (α β : ℝ) (hα : Real.tan α = 3) (hβ : Real.tan β = 2) : Real.tan (α - β) = 1 / 7 := by
  sorry

end tan_diff_l469_469199


namespace additional_visitors_needed_l469_469718

def current_visitors : ℕ := 1879564

def distinct_digits (n : ℕ) : Prop :=
  let digits := n.digits 10 in
  digits.nodup

theorem additional_visitors_needed :
  distinct_digits current_visitors →
  ∃ k : ℕ, k = 38 ∧ distinct_digits (current_visitors + k) :=
begin
  sorry
end

end additional_visitors_needed_l469_469718


namespace max_positive_root_eq_l469_469770

theorem max_positive_root_eq (b c : ℝ) (h_b : |b| ≤ 3) (h_c : |c| ≤ 3) : 
  ∃ x, x = (3 + Real.sqrt 21) / 2 ∧ x^2 + b * x + c = 0 ∧ x ≥ 0 :=
by
  sorry

end max_positive_root_eq_l469_469770


namespace equation_of_desired_line_l469_469791

section
  -- Define the conditions
  variable (x y : ℝ)
  def line1 := 2 * x - y + 4 = 0
  def line2 := x - y + 5 = 0
  def perp_line := x - 2 * y = 0

  -- Define the intersection point of line1 and line2
  def intersect_point := ∃ x y, line1 x y ∧ line2 x y
  def point := (1, 6)

  -- Define the slope of the given line and the slope of the desired perpendicular line
  def slope_of_given_line := 1 / 2
  def slope_of_desired_line := -2

  -- Define the desired line
  def desired_line := 2 * x + y - 8 = 0

  -- The theorem to be proved
  theorem equation_of_desired_line : 
      (2 * point.1 - point.2 + 4 = 0) ∧ (point.1 - point.2 + 5 = 0) 
      → (point.2 - 6 = -2 * (point.1 - 1)) 
      → (2 * point.1 + point.2 - 8 = 0) := by
    intros
    sorry
end

end equation_of_desired_line_l469_469791


namespace Cindy_correct_result_l469_469057

/-- Define the variable x and the condition that results from Cindy's calculations. -/
variable (x : ℝ)
variable (Cindy_incorrect_result : (x - 7) / 5 = 23)

theorem Cindy_correct_result : (x = 122) → (27 / 5 = 5.2) → ((x + 7) / 5 = 26) :=
by
  intro hx hfive
  rw [hx]
  sorry

end Cindy_correct_result_l469_469057


namespace initial_salary_increase_l469_469338

theorem initial_salary_increase :
  ∃ x : ℝ, 5000 * (1 + x/100) * 0.95 = 5225 := by
  sorry

end initial_salary_increase_l469_469338


namespace lcm_first_ten_positive_integers_l469_469427

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469427


namespace cost_per_set_l469_469978

variable {C : ℝ} -- Define the variable cost per set.

theorem cost_per_set
  (initial_outlay : ℝ := 10000) -- Initial outlay for manufacturing.
  (revenue_per_set : ℝ := 50) -- Revenue per set sold.
  (sets_sold : ℝ := 500) -- Sets produced and sold.
  (profit : ℝ := 5000) -- Profit from selling 500 sets.

  (h_profit_eq : profit = (revenue_per_set * sets_sold) - (initial_outlay + C * sets_sold)) :
  C = 20 :=
by
  -- Proof to be filled in later.
  sorry

end cost_per_set_l469_469978


namespace disc_covers_24_squares_l469_469004

/-- 
  A problem setup with an 8 x 8 checkerboard and a disc whose diameter equals the side length of each square. 
  The disc is positioned such that its center is at the corner of one of the central squares. 
  Prove that the number of squares completely covered by the disc is 24.
-/
theorem disc_covers_24_squares
  (side_length : ℝ) (r : ℝ := side_length / 2) : 
  let center := (7 * side_length / 2, 7 * side_length / 2) in
  let total_squares := 64 in
  let covered_squares := 
    finset.filter (λ (i : ℕ × ℕ), 
      let square_center := 
        (i.1 * side_length + side_length / 2, 
         i.2 * side_length + side_length / 2) in
      let distance := 
        real.sqrt ((square_center.1 - center.1) ^ 2 
                 + (square_center.2 - center.2) ^ 2) in
      distance + side_length * (real.sqrt 2 / 2) ≤ r)
    (finset.Icc (0, 0) (7, 7)) 
  in
  finset.card covered_squares = 24 :=
by
  sorry

end disc_covers_24_squares_l469_469004


namespace linear_equation_a_neg2_l469_469194

theorem linear_equation_a_neg2 (a : ℝ) :
  (∃ x y : ℝ, (a - 2) * x ^ (|a| - 1) + 3 * y = 1) ∧
  (∀ x : ℝ, x ≠ 0 → x ^ (|a| - 1) ≠ 1) →
  a = -2 :=
by
  sorry

end linear_equation_a_neg2_l469_469194


namespace borgnine_tarantulas_needed_l469_469756

def total_legs_goal : ℕ := 1100
def chimp_legs : ℕ := 12 * 4
def lion_legs : ℕ := 8 * 4
def lizard_legs : ℕ := 5 * 4
def tarantula_legs : ℕ := 8

theorem borgnine_tarantulas_needed : 
  let total_legs_seen := chimp_legs + lion_legs + lizard_legs
  let legs_needed := total_legs_goal - total_legs_seen
  let num_tarantulas := legs_needed / tarantula_legs
  num_tarantulas = 125 := 
by
  sorry

end borgnine_tarantulas_needed_l469_469756


namespace part_a_part_b_part_c_part_d_l469_469237

noncomputable theory

-- Part (a)
theorem part_a : (73 + 27 + 72 + 37 = 209) := 
by sorry

-- Part (b)
theorem part_b (b c x y : ℕ) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) (hx : 1 ≤ x ∧ x ≤ 9) (hy : 1 ≤ y ∧ y ≤ 9 ) : 
  (57 + 10*b + 7 + 5*c + 70 = x*10 + (b+1) + (c-3)*10 + y) :=
by sorry

-- Part (c)
theorem part_c (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) (hd : 1 ≤ d ∧ d ≤ 9) : 
  ((10*a + b + 10*c + d + 10*c + d + a*10 + 10*d + b) - (10*(a+1) + (b-2) + 10*(c-1) + (d+1) + 10*(c-1) + (d+1) + (a+1)*10 + 10*(d+1) + (b-2)) = 0) :=
by sorry

-- Part (d)
theorem part_d (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 1 ≤ b ∧ b ≤ 9) (hc : 1 ≤ c ∧ c ≤ 9) (hd : 1 ≤ d ∧ d ≤ 9) : 
  ((10*a + b + 10*c + d + 10*c + d + a*10 + 10*d + b) = 104) :=
by sorry

end part_a_part_b_part_c_part_d_l469_469237


namespace sum_of_coeffs_not_containing_x_l469_469801

theorem sum_of_coeffs_not_containing_x :
  (∑ k in Finset.range (5 + 1), ↑(Nat.choose 5 k) * (-5)^k) = -1024 :=
by
  sorry

end sum_of_coeffs_not_containing_x_l469_469801


namespace find_m_l469_469182

  noncomputable def vector_a : ℝ × ℝ := (real.sqrt 3, 1)
  noncomputable def angle_between_vectors : ℝ := 2 * real.pi / 3

  def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)

  theorem find_m (m : ℝ) (h : angle_between_vectors = 2 * real.pi / 3)
  (dot_product_eq : vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 =
  real.sqrt ((vector_a.1)^2 + (vector_a.2)^2) * real.sqrt ((m)^2 + 1) * real.cos angle_between_vectors) :
  m = -real.sqrt 3 :=
  sorry
  
end find_m_l469_469182


namespace find_value_of_a_l469_469827

theorem find_value_of_a 
  (P : ℝ × ℝ)
  (a : ℝ)
  (α : ℝ)
  (point_on_terminal_side : P = (-4, a))
  (sin_cos_condition : Real.sin α * Real.cos α = Real.sqrt 3 / 4) : 
  a = -4 * Real.sqrt 3 ∨ a = - (4 * Real.sqrt 3 / 3) :=
sorry

end find_value_of_a_l469_469827


namespace least_divisible_1_to_10_l469_469538

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469538


namespace question1_question2_l469_469144

section

variables {x m : ℝ}
def p : Prop := (x + 1) * (2 - x) ≥ 0
def q : Prop := ∀ x, x^2 + 2 * m * x - m + 6 > 0

theorem question1 (hq : q) : -3 < m ∧ m < 2 := sorry

theorem question2 (hs : p ⊆ q) : -3 < m ∧ m ≤ 2 := sorry

end

end question1_question2_l469_469144


namespace coins_problem_l469_469282

theorem coins_problem (x y : ℕ) (h1 : x + y = 20) (h2 : x + 5 * y = 80) : x = 5 :=
by
  sorry

end coins_problem_l469_469282


namespace least_common_multiple_first_ten_integers_l469_469579

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469579


namespace least_common_multiple_of_first_ten_integers_l469_469366

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469366


namespace least_common_multiple_of_first_ten_positive_integers_l469_469466

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469466


namespace least_positive_integer_divisible_by_first_ten_l469_469479

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469479


namespace lcm_first_ten_l469_469408

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469408


namespace george_team_final_round_average_required_less_than_record_l469_469810

theorem george_team_final_round_average_required_less_than_record :
  ∀ (old_record average_score : ℕ) (players : ℕ) (rounds : ℕ) (current_score : ℕ),
    old_record = 287 →
    players = 4 →
    rounds = 10 →
    current_score = 10440 →
    (old_record - ((rounds * (old_record * players) - current_score) / players)) = 27 :=
by
  -- Given the values and conditions, prove the equality here
  sorry

end george_team_final_round_average_required_less_than_record_l469_469810


namespace root_expr_calculation_l469_469055

theorem root_expr_calculation : (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := 
by 
  sorry

end root_expr_calculation_l469_469055


namespace compare_f_values_l469_469848

def f (x : ℝ) : ℝ := x^2 + Real.cos x

theorem compare_f_values :
  let a := f 1
  let b := f (Real.log 4 / Real.log (1 / 2))
  let c := f (Real.log (Real.sqrt 2 / 2) / Real.log 2)
  c < a ∧ a < b := by
  sorry

end compare_f_values_l469_469848


namespace matrix_not_invertible_iff_x_eq_16_div_19_l469_469095

theorem matrix_not_invertible_iff_x_eq_16_div_19 (x : ℝ) :
  ¬(matrix.det ![(2 + x), 9; (4 - x), 10] ≠ 0) ↔ x = 16 / 19 := 
by
  sorry

end matrix_not_invertible_iff_x_eq_16_div_19_l469_469095


namespace inclusion_exclusion_l469_469360

variables {Ω : Type*} [ProbabilitySpace Ω]
variables {A : Finset (Set Ω)}

theorem inclusion_exclusion (n : ℕ) (A : Fin n → Set Ω) :
  ProbabilitySpace.Probability (⋃ i, A i)
  = ∑ (s : Finset (Fin n)), (-1)^(s.card + 1) * ProbabilitySpace.Probability (⋂ i ∈ s, A i) :=
sorry

end inclusion_exclusion_l469_469360


namespace find_a1_general_term_sum_of_terms_l469_469136

-- Given conditions
variable (a : ℕ → ℝ) (S : ℕ → ℝ)
axiom h_condition : ∀ n, S n = (3 / 2) * a n - (1 / 2)

-- Specific condition for finding a1
axiom h_S1_eq_1 : S 1 = 1

-- Prove statements
theorem find_a1 : a 1 = 1 :=
by
  sorry

theorem general_term (n : ℕ) : n ≥ 1 → a n = 3 ^ (n - 1) :=
by
  sorry

theorem sum_of_terms (n : ℕ) : n ≥ 1 → S n = (3 ^ n - 1) / 2 :=
by
  sorry

end find_a1_general_term_sum_of_terms_l469_469136


namespace least_common_multiple_of_first_ten_positive_integers_l469_469460

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469460


namespace least_distinct_values_l469_469011

/-- 
Given a list of 2018 positive integers with a unique mode occurring exactly 10 times,
prove that the least number of distinct values in the list is 225.
-/
theorem least_distinct_values {α : Type*} (l : list α) (hl_len : l.length = 2018) (hm : ∃ m, ∀ x ∈ l, count l x ≤ count l m ∧ count l m = 10 ∧ ∀ y ≠ x, count l y < 10) :
  ∃ n, n = 225 ∧ (∀ x ∈ (l.erase_dup), count l x ≤ 10) :=
sorry

end least_distinct_values_l469_469011


namespace least_common_multiple_1_to_10_l469_469503

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469503


namespace sand_needed_for_sandbox_l469_469712

def length1 : ℕ := 50
def width1 : ℕ := 30
def length2 : ℕ := 20
def width2 : ℕ := 15
def area_per_bag : ℕ := 80
def weight_per_bag : ℕ := 30

theorem sand_needed_for_sandbox :
  (length1 * width1 + length2 * width2 + area_per_bag - 1) / area_per_bag * weight_per_bag = 690 :=
by sorry

end sand_needed_for_sandbox_l469_469712


namespace least_common_multiple_first_ten_integers_l469_469571

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469571


namespace range_of_m_l469_469956

noncomputable def f (x : ℝ) : ℝ := x^3 - (1/2)*x^2 - 2*x + 5

theorem range_of_m {m : ℝ} (h : ∀ x ∈ set.Icc (-1 : ℝ) 2, f x < m) : m > 7 := 
sorry

end range_of_m_l469_469956


namespace distance_lightning_to_nearest_half_mile_l469_469290

theorem distance_lightning_to_nearest_half_mile
  (time_sec : ℕ)
  (speed_sound_fps : ℕ)
  (feet_per_mile : ℕ)
  (h_time : time_sec = 12)
  (h_speed : speed_sound_fps = 1100)
  (h_feet_per_mile : feet_per_mile = 5280) :
  (speed_sound_fps * time_sec) / feet_per_mile = 2.5 := 
by
  sorry

end distance_lightning_to_nearest_half_mile_l469_469290


namespace min_value_of_expression_l469_469269

theorem min_value_of_expression (p q r s t u v w : ℝ) (h1 : p * q * r * s = 16) (h2 : t * u * v * w = 25) : 
  (pt)^2 + (qu)^2 + (rv)^2 + (sw)^2 ≥ 400 :=
begin
  sorry,
end

end min_value_of_expression_l469_469269


namespace cubic_root_of_determinant_l469_469257

open Complex 
open Matrix

noncomputable def matrix_d (a b c n : ℂ) : Matrix (Fin 3) (Fin 3) ℂ :=
  ![
    ![b + n^3 * c, n * (c - b), n^2 * (b - c)],
    ![n^2 * (c - a), c + n^3 * a, n * (a - c)],
    ![n * (b - a), n^2 * (a - b), a + n^3 * b]
  ]

theorem cubic_root_of_determinant (a b c n : ℂ) (h : a * b * c = 1) :
  (det (matrix_d a b c n))^(1/3 : ℂ) = n^3 + 1 :=
  sorry

end cubic_root_of_determinant_l469_469257


namespace probability_all_same_tribe_l469_469905

-- Define the conditions as hypotheses
def num_contestants : ℕ := 24
def num_tribes : ℕ := 3
def tribe_size : ℕ := num_contestants / num_tribes
def num_quitters : ℕ := 3

-- Main theorem stating the probability
theorem probability_all_same_tribe :
  (div (3 * nat.choose tribe_size num_quitters) (nat.choose num_contestants num_quitters)) = 1 / 12 := 
by 
  sorry

end probability_all_same_tribe_l469_469905


namespace binary_to_decimal_1101_is_13_l469_469772

-- Define the binary number and its decimal conversion
def binary_number : List ℕ := [1, 1, 0, 1]

-- Define the positions and corresponding powers of 2 for binary number
def powers_of_two : List ℕ := [2^3, 2^2, 2^1, 2^0]

-- Compute the decimal value of the binary number
def binary_to_decimal (b : List ℕ) (p : List ℕ) : ℕ :=
  List.sum (List.zipWith (λ (digit power), digit * power) b p)

-- The theorem to prove
theorem binary_to_decimal_1101_is_13 : binary_to_decimal binary_number powers_of_two = 13 := by
  -- The proof is omitted
  sorry

end binary_to_decimal_1101_is_13_l469_469772


namespace workouts_difference_l469_469184

theorem workouts_difference
  (workouts_monday : ℕ := 8)
  (workouts_tuesday : ℕ := 5)
  (workouts_wednesday : ℕ := 12)
  (workouts_thursday : ℕ := 17)
  (workouts_friday : ℕ := 10) :
  workouts_thursday - workouts_tuesday = 12 := 
by
  sorry

end workouts_difference_l469_469184


namespace johnson_and_martinez_home_runs_l469_469228

def monthly_home_runs_johnson : (ℕ → ℕ) :=
  λ month,
    match month with
    | 3 => 2
    | 4 => 9
    | 5 => 5
    | 6 => 11
    | 7 => 9
    | 8 => 7
    | 9 => 12
    | _ => 0

def monthly_home_runs_martinez : (ℕ → ℕ) :=
  λ month,
    match month with
    | 3 => 0
    | 4 => 4
    | 5 => 8
    | 6 => 17
    | 7 => 3
    | 8 => 9
    | 9 => 8
    | _ => 0

def cumulative_home_runs (f : ℕ → ℕ) (month : ℕ) : ℕ :=
  (List.range month).sum (λ m => f (m + 3))

theorem johnson_and_martinez_home_runs :
  (∀ month ≤ 9, cumulative_home_runs monthly_home_runs_johnson month ≠ cumulative_home_runs monthly_home_runs_martinez month) ∧
  (cumulative_home_runs monthly_home_runs_johnson 7 = 55) ∧
  (cumulative_home_runs monthly_home_runs_martinez 7 = 49) :=
by
  -- proof skipped
  sorry

end johnson_and_martinez_home_runs_l469_469228


namespace solution1_solution2_l469_469765

noncomputable def problem1 : ℝ :=
  |1 - Real.sqrt 2| + Real.cbrt (-64) - Real.sqrt (1 / 2)

noncomputable def problem2 : ℝ :=
  (3 - 2 * Real.sqrt 5) * (3 + 2 * Real.sqrt 5) + (1 - Real.sqrt 5) ^ 2

theorem solution1 : problem1 = (Real.sqrt 2) / 2 - 5 :=
by sorry

theorem solution2 : problem2 = -5 - 2 * Real.sqrt 5 :=
by sorry

end solution1_solution2_l469_469765


namespace linear_equation_solution_l469_469196

theorem linear_equation_solution (a : ℝ) (x y : ℝ) 
    (h : (a - 2) * x^(|a| - 1) + 3 * y = 1) 
    (h1 : ∀ (x y : ℝ), (a - 2) ≠ 0)
    (h2 : |a| - 1 = 1) : a = -2 :=
by
  sorry

end linear_equation_solution_l469_469196


namespace problem_statement_l469_469294

variable (a b c d : ℝ)

-- Definitions for the conditions
def condition1 := a + b + c + d = 100
def condition2 := (a / (b + c + d)) + (b / (a + c + d)) + (c / (a + b + d)) + (d / (a + b + c)) = 95

-- The theorem which needs to be proved
theorem problem_statement (h1 : condition1 a b c d) (h2 : condition2 a b c d) :
  (1 / (b + c + d)) + (1 / (a + c + d)) + (1 / (a + b + d)) + (1 / (a + b + c)) = 99 / 100 := by
  sorry

end problem_statement_l469_469294


namespace least_common_multiple_1_to_10_l469_469449

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469449


namespace polynomial_divisible_x_minus_2_l469_469779

theorem polynomial_divisible_x_minus_2 (m : ℝ) : 
  (3 * 2^2 - 9 * 2 + m = 0) → m = 6 :=
by
  sorry

end polynomial_divisible_x_minus_2_l469_469779


namespace average_students_count_l469_469348

-- Definitions based on conditions
variables (a b c d : ℕ)     -- Number of students in each category
variables (total_students excellent_answers average_answers poor_answers : ℕ)

-- Given conditions
def student_conditions := (total_students = 30) ∧
                          (excellent_answers = 19) ∧
                          (average_answers = 12) ∧
                          (poor_answers = 9)

def student_relationships := (a + b + c + d = total_students) ∧
                             (a + b + c = excellent_answers) ∧
                             (b + c = average_answers) ∧
                             (c = poor_answers)

-- Prove the number of average students is 20
theorem average_students_count : student_conditions ∧ student_relationships → (c + d = 20) :=
begin
  intros h,
  sorry
end

end average_students_count_l469_469348


namespace least_common_multiple_of_first_ten_integers_l469_469372

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469372


namespace sequence_term_expression_l469_469162

theorem sequence_term_expression (S : ℕ → ℕ) (a : ℕ → ℕ) (h₁ : ∀ n, S n = 3^n + 1) :
  (a 1 = 4) ∧ (∀ n, n ≥ 2 → a n = 2 * 3^(n-1)) :=
by
  sorry

end sequence_term_expression_l469_469162


namespace least_divisible_1_to_10_l469_469537

open Nat

noncomputable def lcm_of_first_ten_positive_integers : ℕ :=
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10

theorem least_divisible_1_to_10 : lcm_of_first_ten_positive_integers = 2520 :=
  sorry

end least_divisible_1_to_10_l469_469537


namespace find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469107

theorem find_x_such_that_sqrt_5x_plus_9_eq_12 : ∀ x : ℝ, sqrt (5 * x + 9) = 12 → x = 27 := 
by
  intro x
  sorry

end find_x_such_that_sqrt_5x_plus_9_eq_12_l469_469107


namespace least_common_multiple_of_first_ten_integers_l469_469367

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469367


namespace sufficient_but_not_necessary_for_ax_square_pos_l469_469819

variables (a x : ℝ)

theorem sufficient_but_not_necessary_for_ax_square_pos (h : a > 0) : 
  (a > 0 → ax^2 > 0) ∧ ((ax^2 > 0) → a > 0) :=
sorry

end sufficient_but_not_necessary_for_ax_square_pos_l469_469819


namespace least_positive_integer_divisible_by_first_ten_l469_469476

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469476


namespace problem_statement_l469_469140

-- Define the ellipse equation and conditions
def ellipse_equation (a : ℝ) (x y: ℝ) : Prop := 
  (a > 1) → (x^2 / a^2 + y^2 = 1)

-- Define the foci and the circle intersecting condition
def focus (a : ℝ) : ℝ := sqrt(a^2 - 1)
def circle_intersects (a : ℝ) : Prop := 
  (2 * (focus a))^2 = a^2 + 4

-- Point P and its x-coordinate condition
def x_coord_P (k : ℝ) : ℝ := -1 / 2 + 1 / (4 * k^2 + 2)
def x_coord_condition (k : ℝ) : Prop := 
  -1/4 ≤ x_coord_P k ∧ x_coord_P k < 0

-- Absolute distance |AB| and its minimum value condition
def absolute_distance_eq (k : ℝ) : ℝ := 
  2 * sqrt(2) * (1/2 + 1/(2 * (2 * k^2 + 1)))
def minimum_AB_distance (k : ℝ) : ℝ := 3 * sqrt(2) / 2

theorem problem_statement (a : ℝ) (k : ℝ) :
  (ellipse_equation a x y) →
  (circle_intersects a) →
  (x_coord_condition k) →
  ∃ (min_AB : ℝ), min_AB = minimum_AB_distance k :=
sorry

end problem_statement_l469_469140


namespace least_positive_integer_divisible_by_first_ten_l469_469481

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469481


namespace least_common_multiple_1_to_10_l469_469441

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469441


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469390

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469390


namespace lcm_first_ten_positive_integers_l469_469430

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469430


namespace least_number_of_marbles_l469_469729

theorem least_number_of_marbles :
  ∃ n, (∀ d ∈ ({3, 4, 5, 7, 8} : Set ℕ), d ∣ n) ∧ n = 840 :=
by
  sorry

end least_number_of_marbles_l469_469729


namespace solve_for_x_l469_469115

  theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → x = 27 :=
  by
    sorry
  
end solve_for_x_l469_469115


namespace find_mn_l469_469919

variables {A B C D P Q : Point}
variables {x y : ℝ}

-- Conditions translated to Lean
axiom ab_cd : dist A B < dist C D
axiom ab_perp_bc : perp A B B C
axiom ab_parallel_cd : parallel A B C D
axiom ac_bd_perp_at_p : intersect A C B D P ∧ perp A C B D
axiom q_on_ca_past_a : on_ray Q A C ∧ dist A Q > dist A C
axiom qd_perp_dc : perp Q D D C

-- Provided equation
axiom given_eqn : (QP / AP) + (AP / QP) = ((51 / 14)^4 - 2)

-- Proof goal
theorem find_mn : (BP / AP - AP / BP) = (47 / 14) ∧ 61 = 47 + 14 :=
by
  sorry


end find_mn_l469_469919


namespace crossing_time_l469_469671

-- Definitions
def length_train : ℝ := 120
def time_train1 : ℝ := 10
def time_train2 : ℝ := 15

-- Speeds
def speed_train1 : ℝ := length_train / time_train1
def speed_train2 : ℝ := length_train / time_train2

-- Relative speed and total distance when traveling in opposite directions
def speed_relative : ℝ := speed_train1 + speed_train2
def total_distance : ℝ := 2 * length_train

-- Theorem to prove the crossing time
theorem crossing_time : total_distance / speed_relative = 12 := 
by
  sorry

end crossing_time_l469_469671


namespace least_common_multiple_1_to_10_l469_469501

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469501


namespace check_statements_l469_469979

noncomputable def f (x : ℝ) : ℝ := 4 * Real.sin (2 * x + Real.pi / 3)

theorem check_statements :
  (∀ x : ℝ, f(x + Real.pi) = f x) ∧
  (¬∀ x : ℝ, f x = 4 * Real.sin (2 * x + Real.pi / 3)) ∧
  (f 0 = 4) :=
by
  split
  · sorry -- Prove that the minimum positive period of the function is π
  split
  · sorry -- Prove that the initial phase is not 2x + π/3 but π/3
  · sorry -- Prove that the amplitude of the function is 4

end check_statements_l469_469979


namespace infinite_solutions_exists_l469_469938

theorem infinite_solutions_exists
  (n : ℕ)
  (A : Fin (n + 1) → ℕ)
  (hA_pos : ∀ i, 0 < A i)
  (hA_gcd : ∀ i : Fin n, gcd (A i) (A n) = 1) : 
  ∃ S : ℕ → (Fin (n + 1) → ℕ), 
    (∀ k, (∑ i : Fin n, (S k i) ^ (A i)) = (S k n) ^ (A n)) ∧ 
    (∀ k, ∀ i, 0 < S k i) :=
sorry

end infinite_solutions_exists_l469_469938


namespace cos_triple_angle_l469_469207

-- Define the variables
variable (θ : ℝ)

-- State the condition
axiom h : cos θ = 1/3

-- State the theorem
theorem cos_triple_angle : cos (3 * θ) = -23/27 := by
  sorry

end cos_triple_angle_l469_469207


namespace target_runs_l469_469914

theorem target_runs (run_rate_20_overs : ℝ) (run_rate_30_overs : ℝ) (overs_first_phase : ℝ) (overs_second_phase : ℝ) : 
  run_rate_20_overs = 4.8 ∧ run_rate_30_overs = 6.866666666666666 ∧ overs_first_phase = 20 ∧ overs_second_phase = 30 →
  let runs_first_phase := overs_first_phase * run_rate_20_overs in
  let runs_second_phase := overs_second_phase * run_rate_30_overs in
  let total_runs := runs_first_phase + runs_second_phase in
  total_runs = 302 :=
begin
  sorry
end

end target_runs_l469_469914


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469395

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469395


namespace least_common_multiple_1_to_10_l469_469494

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469494


namespace carnations_count_l469_469027

theorem carnations_count (c : ℕ) : 
  (9 * 6 = 54) ∧ (47 ≤ c + 47) ∧ (c + 47 = 54) → c = 7 := 
by
  sorry

end carnations_count_l469_469027


namespace range_of_a_l469_469870

theorem range_of_a (a : ℝ) : (2 < a ∧ a < 5 ∧ a ≠ 3) ↔ (a ∈ set.Ioo 2 3 ∪ set.Ioo 3 5) :=
by
  sorry

end range_of_a_l469_469870


namespace sum_of_elements_in_A_l469_469342

def A : Set ℤ := { n | n^3 < 2022 ∧ 2022 < 3^n }

theorem sum_of_elements_in_A : ∑ n in finset.filter (λ n => n ∈ A) (finset.Icc 7 12), n = 57 :=
by
  sorry

end sum_of_elements_in_A_l469_469342


namespace least_common_multiple_of_first_ten_l469_469606

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469606


namespace vector_operations_l469_469816

theorem vector_operations :
  let a := (2, 1) : ℝ × ℝ
  let b := (-3, 4) : ℝ × ℝ
  (3 * a.1 + 4 * b.1, 3 * a.2 + 4 * b.2) = (-6, 19) ∧
  (4 * a.1 - 2 * b.1, 4 * a.2 - 2 * b.2) = (14, -4) :=
by
  let a := (2, 1) : ℝ × ℝ
  let b := (-3, 4) : ℝ × ℝ
  split
  · exact sorry
  · exact sorry

end vector_operations_l469_469816


namespace lcm_first_ten_l469_469409

-- Define the set of first ten positive integers
def first_ten_integers : List ℕ := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the LCM of a list of integers
noncomputable def lcm_list (l : List ℕ) : ℕ :=
List.foldr Nat.lcm 1 l

-- The theorem stating that the LCM of the first ten integers is 2520
theorem lcm_first_ten : lcm_list first_ten_integers = 2520 := by
  sorry

end lcm_first_ten_l469_469409


namespace maximize_binomial_difference_l469_469255

theorem maximize_binomial_difference :
  ∃ a : ℕ, 0 ≤ a ∧ a ≤ (10^6 - 1) ∧ 
    (∀ b : ℕ, (0 ≤ b ∧ b ≤ 10^6 - 1) → 
      (binom 10^6 (a + 1) - binom 10^6 a) ≥ (binom 10^6 (b + 1) - binom 10^6 b)) ∧ 
    a = 499499 := by
  sorry

end maximize_binomial_difference_l469_469255


namespace odd_function_condition_l469_469946

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (a + 2 / (x - 1))

theorem odd_function_condition (a : ℝ) :
  (f a = (λ x, Real.log (a + 2 / (x - 1))) ∧ (∀ x, f a (-x) = -f a x)) ↔ a = 1 :=
sorry

end odd_function_condition_l469_469946


namespace at_least_one_sum_diverges_l469_469258

noncomputable def f : ℕ → ℕ := sorry -- Assume f is a bijection ℕ → ℕ

axiom f_bijective : Function.Bijective f

theorem at_least_one_sum_diverges : 
  (Filter.Tendsto (λ N, ∑ n in Finset.range N, 1 / (n + f n)) Filter.atTop Filter.atBot) ∨
  (Filter.Tendsto (λ N, ∑ n in Finset.range N, (1 / n - 1 / f n)) Filter.atTop Filter.atBot) :=
sorry

end at_least_one_sum_diverges_l469_469258


namespace max_pawns_l469_469652

def chessboard : Type := ℕ × ℕ -- Define a chessboard as a grid of positions (1,1) to (8,8)
def e4 : chessboard := (5, 4) -- Define the position e4
def symmetric_wrt_e4 (p1 p2 : chessboard) : Prop :=
  p1.1 + p2.1 = 10 ∧ p1.2 + p2.2 = 8 -- Symmetry condition relative to e4

def placed_on (pos : chessboard) : Prop := sorry -- placeholder for placement condition

theorem max_pawns (no_e4 : ¬ placed_on e4)
  (no_symmetric_pairs : ∀ p1 p2, symmetric_wrt_e4 p1 p2 → ¬ (placed_on p1 ∧ placed_on p2)) :
  ∃ max_pawns : ℕ, max_pawns = 39 :=
sorry

end max_pawns_l469_469652


namespace range_of_a_l469_469842

def f (a x : ℝ) : ℝ := (a * x^2 + 2*x - 1) / x

def domain (x : ℝ) : Prop := x ≥ 3/7

theorem range_of_a (a : ℝ) :
  (∀ x₁ x₂ : ℝ, domain x₁ → domain x₂ → x₁ < x₂ → f a x₁ > f a x₂) →
  a ≤ -49/9 :=
by
  sorry

end range_of_a_l469_469842


namespace length_of_PX_l469_469232

variables {C D W X P : Type*} 
variables [linear_ordered_field P]
variables (CD_WX_parallel : ∀ (a b : P), a = b)
variables (CW DP PX : P)

theorem length_of_PX 
  (h_parallel : CD_WX_parallel) 
  (h_CW : CW = 56) 
  (h_DP : DP = 16) 
  (h_PX : PX = 32) : PX = 112 / 3 := 
by 
  sorry

end length_of_PX_l469_469232


namespace rook_main_diagonal_moves_l469_469966

theorem rook_main_diagonal_moves
  (T : Type)
  (chessboard : T)
  (rook_start : chessboard)
  (rook_moves_exactly_once : Π (rook_path : list chessboard), rook_start ∈ rook_path ∧ ∀ p ∈ rook_path, visit_once p rook_path)
  (main_diagonals : list (list chessboard)) :
  ∀ diag ∈ main_diagonals, ∃ (idx : Nat), idx < List.length diag - 1 ∧
    (list.nth diag idx = list.nth diag (idx + 1) ∧ ¬(list.nth diag idx = rook_path idx) →
    (list.nth diag idx = rook_path (idx + 2))) :=
by
  sorry

end rook_main_diagonal_moves_l469_469966


namespace least_common_multiple_first_ten_l469_469545

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469545


namespace total_amount_paid_l469_469357

noncomputable def employee_payment (b_payment : ℝ) : ℝ :=
  let a_payment := 1.20 * b_payment
  a_payment + b_payment

theorem total_amount_paid (b_payment : ℝ) (h : b_payment = 249.99999999999997) :
  employee_payment b_payment = 550 :=
by
  let a_payment := 1.20 * b_payment
  have ha : a_payment = 299.99999999999996 := by
    sorry
  have h_total : a_payment + b_payment = Rs. 550 := by
    sorry
  exact h_total

end total_amount_paid_l469_469357


namespace ratio_solves_for_x_l469_469204

theorem ratio_solves_for_x (x : ℝ) (h : 0.60 / x = 6 / 4) : x = 0.4 :=
by
  -- The formal proof would go here.
  sorry

end ratio_solves_for_x_l469_469204


namespace cost_price_article_l469_469670

theorem cost_price_article (x : ℝ) (h : 56 - x = x - 42) : x = 49 :=
by sorry

end cost_price_article_l469_469670


namespace min_value_xy_l469_469838

theorem min_value_xy (x y : ℝ) (h1 : x + y = -1) (h2 : x < 0) (h3 : y < 0) :
  ∃ (xy_min : ℝ), (∀ (xy : ℝ), xy = x * y → xy + 1 / xy ≥ xy_min) ∧ xy_min = 17 / 4 :=
by
  sorry

end min_value_xy_l469_469838


namespace students_taking_mathematics_but_not_science_l469_469904

theorem students_taking_mathematics_but_not_science :
  ∀ (total_students both_subjects math_students science_students : ℕ),
    total_students = 36 →
    both_subjects = 2 →
    math_students = sc (science_students + both_subjects) * 4 / 3 → -- one-third more students in Mathematics than in Science
    (math_students - both_subjects) = 20 :=
by sorry

end students_taking_mathematics_but_not_science_l469_469904


namespace unique_2_digit_cyclic_permutation_divisible_l469_469834

def is_cyclic_permutation (n : ℕ) (M : ℕ → ℕ) : Prop :=
  ∀ i j : ℕ, i < n → j < n → M i = M j

def M (a : Fin 2 → ℕ) : ℕ := a 0 * 10 + a 1

theorem unique_2_digit_cyclic_permutation_divisible (a : Fin 2 → ℕ) (h0 : ∀ i, a i ≠ 0) :
  (M a) % (a 1 * 10 + a 0) = 0 → 
  (M a = 11) :=
by
  sorry

end unique_2_digit_cyclic_permutation_divisible_l469_469834


namespace max_volume_of_cube_in_tetrahedron_l469_469018

-- Define the edge length of the tetrahedron
def edge_length : ℝ := 2

-- Define the problem in terms of a mathematical proposition
theorem max_volume_of_cube_in_tetrahedron (V_cube : ℝ) :
  (∃ (s : ℝ), s = (edge_length * sqrt 3) / 9 ∧ V_cube = s^3) → V_cube = 8 * sqrt 3 / 243 :=
by
  sorry -- Proof goes here

end max_volume_of_cube_in_tetrahedron_l469_469018


namespace transformation_converges_l469_469829

def t (A : List ℕ) (i : ℕ) : ℕ :=
  A.take i |>.countp (λ x => x ≠ A.get! i)

theorem transformation_converges {n : ℕ} (h : 0 < n) (A : Fin (n + 1) → ℕ)
  (hA : ∀ i : Fin (n + 1), A i ≤ i) :
  ∃ k < n, let B : Fin (n + 1) → ℕ := λ i => t (List.ofFn A) i
  in ∀ i : Fin (n + 1), B i = A i :=
by
  sorry

end transformation_converges_l469_469829


namespace find_m_value_l469_469216

theorem find_m_value (m : ℤ) : (x^2 + m * x - 35 = (x - 7) * (x + 5)) → m = -2 :=
by
  sorry

end find_m_value_l469_469216


namespace sum_of_first_3030_terms_l469_469345

-- Define geometric sequence sum for n terms
noncomputable def geom_sum (a r : ℝ) (n : ℕ) : ℝ :=
  a * (1 - r ^ n) / (1 - r)

-- Given conditions
axiom geom_sum_1010 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 1010 = 100
axiom geom_sum_2020 (a r : ℝ) (hr : r ≠ 1) : geom_sum a r 2020 = 190

-- Prove that the sum of the first 3030 terms is 271
theorem sum_of_first_3030_terms (a r : ℝ) (hr : r ≠ 1) :
  geom_sum a r 3030 = 271 :=
by
  sorry

end sum_of_first_3030_terms_l469_469345


namespace least_common_multiple_1_to_10_l469_469507

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469507


namespace surface_area_cube_l469_469206

theorem surface_area_cube (a : ℕ) (b : ℕ) (h : a = 2) : b = 54 :=
  by
  sorry

end surface_area_cube_l469_469206


namespace b_equals_2a_theorem_l469_469349

noncomputable def b_equals_2a (n : ℕ) (a b : ℕ) : Prop :=
  ∃ (boys girls : list ℕ), 
    boys.length = girls.length ∧ 
    boys.length * 2 = n ∧ 
    a = boys.permutations.length ∧ 
    b = 2 * a

theorem b_equals_2a_theorem (n a b : ℕ) 
  (h1 : b_equals_2a n a b) : 
  b = 2 * a :=
by 
  obtain ⟨boys, girls, h_len, h_total, h_a, h_b⟩ := h1 
  exact h_b

end b_equals_2a_theorem_l469_469349


namespace determine_a_l469_469329

noncomputable def f (x a : ℝ) := -9 * x^2 - 6 * a * x + 2 * a - a^2

theorem determine_a (a : ℝ) 
  (h₁ : ∀ x ∈ Set.Icc (-1/3 : ℝ) (1/3 : ℝ), f x a ≤ f 0 a)
  (h₂ : f 0 a = -3) :
  a = 2 + Real.sqrt 6 := 
sorry

end determine_a_l469_469329


namespace cannot_form_set_l469_469741

-- Definitions based on the conditions
def definiteness (S : Set ℝ) : Prop := ∀ x y ∈ S, (x = y ∨ x ≠ y)
def distinctness (S : Set ℝ) : Prop := ∀ x y ∈ S, x = y → y = x
def unorderedness (S : Set ℝ) : Prop := True  -- Considering unorderedness is always true for sets in mathematical sense

-- The sets given in the problem
def all_positive_numbers : Set ℝ := {x | 0 < x}
def numbers_equal_to_2 : Set ℝ := {x | x = 2}
def numbers_close_to_0 (ε : ℝ) : Set ℝ := {x | -ε < x ∧ x < ε}
def even_numbers_not_equal_to_0 : Set ℝ := {x | ∃ n : ℤ, x = 2 * n ∧ x ≠ 0}

-- Proof statement: Numbers close to 0 cannot form a set
theorem cannot_form_set (ε : ℝ) (h_ε : ε > 0) : ¬ (definiteness (numbers_close_to_0 ε) ∧ distinctness (numbers_close_to_0 ε) ∧ unorderedness (numbers_close_to_0 ε)) :=
by
  sorry

end cannot_form_set_l469_469741


namespace lcm_first_ten_positive_integers_l469_469426

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469426


namespace least_common_multiple_1_to_10_l469_469491

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469491


namespace frog_minimum_jumps_l469_469714

theorem frog_minimum_jumps :
  ∃ (n : ℕ), n = 4 ∧ 
  (∃ (jumps : fin n → ℤ × ℤ),
  (∀ i, (let (a, b) := jumps i in a^2 + b^2 = 36) ∧
  (let (end_x, end_y) := (0, 0) + 
    jumps 0 + jumps 1 + jumps (i - 0) + jumps (i - 1) in end_x = 2 ∧ end_y = 1))) :=
sorry

end frog_minimum_jumps_l469_469714


namespace equilateral_triangle_and_center_l469_469025

-- Definitions based on given conditions
variables {A B C D : Type} -- Points in the plane
variables [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Distances between points
variables (a1 a2 a3 r : ℝ)
variable (sqrt3 : Real)

-- Given conditions
def triangle_satisfies_ratios (a1 a2 a3 : ℝ) (r : ℝ) :=
  r = Real.sqrt 3 ∧
  ∃ AD BD CD : ℝ, a1 = r * AD ∧ a2 = r * BD ∧ a3 = r * CD

-- Main theorem
theorem equilateral_triangle_and_center
  (h_ratios : triangle_satisfies_ratios a1 a2 a3 (Real.sqrt 3))
  (ABC_is_triangle : triangle A B C) :
  equilateral_triangle A B C ∧ centroid D A B C :=
begin
  sorry,
end

end equilateral_triangle_and_center_l469_469025


namespace lcm_first_ten_numbers_l469_469584

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469584


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469383

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469383


namespace josh_initial_money_l469_469250

/--
Josh spent $1.75 on a drink, and then spent another $1.25, and has $6.00 left. 
Prove that initially Josh had $9.00.
-/
theorem josh_initial_money : 
  ∃ (initial : ℝ), (initial - 1.75 - 1.25 = 6) ∧ initial = 9 := 
sorry

end josh_initial_money_l469_469250


namespace alternating_colors_probability_l469_469693

def box_contains_five_white_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 1) = 5
def box_contains_five_black_balls : Prop := ∃ (box : set ℕ), box.count (λ b, b = 0) = 5
def balls_drawn_one_at_a_time : Prop := true -- This condition is trivially satisfied without more specific constraints

theorem alternating_colors_probability (h1 : box_contains_five_white_balls) (h2 : box_contains_five_black_balls) (h3 : balls_drawn_one_at_a_time) :
  ∃ p : ℚ, p = 1 / 126 :=
sorry

end alternating_colors_probability_l469_469693


namespace lcm_first_ten_integers_l469_469624

theorem lcm_first_ten_integers : Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := by
  sorry

end lcm_first_ten_integers_l469_469624


namespace min_distinct_values_l469_469010

theorem min_distinct_values (n : ℕ) (mode_count : ℕ) (total_count : ℕ) 
  (h_mode : mode_count = 10) (h_total : total_count = 2018) 
  (h_distinct : ∀ k, k ≠ mode_count → k < 10) : 
  n ≥ 225 :=
by
  sorry

end min_distinct_values_l469_469010


namespace arithmetic_sequence_sum_l469_469909

variables {α : Type*} [linear_ordered_field α]

def is_arithmetic_sequence (a : ℕ → α) :=
  ∃ d : α, ∀ n : ℕ, a (n + 1) = a n + d

def sum_first_n_terms (a : ℕ → α) (n : ℕ) :=
  (n.succ * (a 0 + a n) / 2)

theorem arithmetic_sequence_sum (a : ℕ → α) (h : is_arithmetic_sequence a)
  (h1 : a 2 + a 4 = 14) :
  sum_first_n_terms a 6 = 49 :=
sorry

end arithmetic_sequence_sum_l469_469909


namespace find_percentage_l469_469884

theorem find_percentage (P : ℝ) (h : (P / 100) * 600 = (50 / 100) * 720) : P = 60 :=
by
  sorry

end find_percentage_l469_469884


namespace quadratic_max_product_roots_l469_469064

theorem quadratic_max_product_roots (a b : ℝ) (m : ℝ) :
  (a = 5) → (b = -10) →
  (∀ x, a * x^2 + b * x + m = 0 → ∃ x : ℝ, 5 * x^2 - 10 * x + m = 0) →
  (m ≤ 5) →
  (m = 5 → ∃ (r1 r2 : ℝ), r1 * r2 = 1) :=
begin
  intros h1 h2 h3 h4 h5,
  sorry
end

end quadratic_max_product_roots_l469_469064


namespace least_common_multiple_of_first_10_integers_l469_469523

theorem least_common_multiple_of_first_10_integers :
  Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 :=
sorry

end least_common_multiple_of_first_10_integers_l469_469523


namespace go_stones_arrangement_l469_469864

theorem go_stones_arrangement : 
  let total_stones := 3
  let distinct_stones := 2
  let n_white := 2
  let n_black := 1
  in nat.factorial total_stones / (nat.factorial n_white * nat.factorial n_black) = 3 :=
by
  let total_stones := 3
  let n_white := 2
  let n_black := 1
  have h1 : nat.factorial total_stones = 6 := by sorry
  have h2 : nat.factorial n_white = 2 := by sorry
  have h3 : nat.factorial n_black = 1 := by sorry
  show 6 / (2 * 1) = 3
  calc 6 / (2 * 1) = 3 : by sorry

end go_stones_arrangement_l469_469864


namespace least_positive_integer_divisible_by_first_ten_l469_469475

-- Define the first ten positive integers as a list
def firstTenPositiveIntegers : List ℕ :=
  [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

-- Define the problem of finding the least common multiple
theorem least_positive_integer_divisible_by_first_ten :
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 := 
sorry

end least_positive_integer_divisible_by_first_ten_l469_469475


namespace book_selection_l469_469008

theorem book_selection (math_books : ℕ) (literature_books : ℕ) (english_books : ℕ) :
  math_books = 3 → literature_books = 5 → english_books = 8 → math_books + literature_books + english_books = 16 :=
by
  intro h1 h2 h3
  rw [h1, h2, h3]
  rfl

end book_selection_l469_469008


namespace brownie_pieces_l469_469963

theorem brownie_pieces (tray_length tray_width piece_length piece_width : ℕ) (h1 : tray_length = 24) (h2 : tray_width = 30) (h3 : piece_length = 3) (h4 : piece_width = 4) : (tray_length * tray_width) / (piece_length * piece_width) = 60 :=
by
  rw [h1, h2, h3, h4]
  norm_num
  sorry

end brownie_pieces_l469_469963


namespace carnations_count_l469_469029

-- Define the conditions:
def vase_capacity : ℕ := 6
def number_of_roses : ℕ := 47
def number_of_vases : ℕ := 9

-- The goal is to prove that the number of carnations is 7:
theorem carnations_count : (number_of_vases * vase_capacity) - number_of_roses = 7 :=
by
  sorry

end carnations_count_l469_469029


namespace total_cost_is_576_l469_469665

-- Define the dimensions of the floor and carpet squares, and the cost of each carpet square
def floor_length : ℕ := 24
def floor_width : ℕ := 64
def carpet_side : ℕ := 8
def carpet_cost : ℕ := 24

-- Define the areas of the floor and a carpet square
def floor_area : ℕ := floor_length * floor_width
def carpet_area : ℕ := carpet_side * carpet_side

-- Calculate the number of carpet squares needed
def num_carpet_squares : ℕ := floor_area / carpet_area

-- Calculate the total cost
def total_cost : ℕ := num_carpet_squares * carpet_cost

-- Prove that the total cost is $576
theorem total_cost_is_576 : total_cost = 576 := by
  -- Use stored values
  have h1 : floor_area = 1536 := rfl
  have h2 : carpet_area = 64 := rfl
  have h3 : num_carpet_squares = 24 := by
    rw [floor_area, carpet_area]
    exact Nat.div_eq_of_eq_mul_right rfl.dec_trivial
  have h4 : total_cost = 24 * 24 := by
    rw [num_carpet_squares, carpet_cost]
  show total_cost = 576
  rw [h4]
  exact rfl

end total_cost_is_576_l469_469665


namespace find_all_polynomials_l469_469086

noncomputable theory

open Polynomial

theorem find_all_polynomials (f : ℕ[X])
  (h : ∀ (p : ℕ) (n : ℕ), (nat.prime p) → (0 < n) → ∃ (q m : ℕ), (nat.prime q) ∧ (0 < m) ∧ (eval (p ^ n) f = q ^ m)) :
  ∃ (n : ℕ), (0 < n) ∧ f = X^n ∨ ∃ (q m : ℕ), (nat.prime q) ∧ (0 < m) ∧ f = C q ^ m :=
sorry

end find_all_polynomials_l469_469086


namespace parabola_focus_distance_l469_469155

theorem parabola_focus_distance (p : ℝ) (h : p > 0) (dist_eq_five : real.sqrt ((p + 2)^2 + 3^2) = 5) : p = 4 :=
sorry

end parabola_focus_distance_l469_469155


namespace least_common_multiple_first_ten_l469_469547

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469547


namespace evaluate_expression_l469_469059

def binom (n k : ℕ) : ℕ := if h : k ≤ n then Nat.choose n k else 0

theorem evaluate_expression : 
  (binom 2 5 * 3 ^ 5) / binom 10 5 = 0 := by
  -- Given conditions:
  have h1 : binom 2 5 = 0 := by sorry
  have h2 : binom 10 5 = 252 := by sorry
  -- Proof goal:
  sorry

end evaluate_expression_l469_469059


namespace dividend_rate_calculation_l469_469719

-- Definitions and conditions
def face_value := 20 -- in Rs.
def market_value := 15 -- in Rs.
def desired_interest_rate := 0.12
def desired_interest_per_share := market_value * desired_interest_rate -- Rs. 1.80

-- Dividend rate calculation
def dividend_rate := (desired_interest_per_share / face_value) * 100 -- 9%

-- Theorem stating the problem
theorem dividend_rate_calculation :
  dividend_rate = 9 := by
  sorry

end dividend_rate_calculation_l469_469719


namespace point_not_in_fourth_quadrant_l469_469229

theorem point_not_in_fourth_quadrant (m : ℝ) : ¬(m-2 > 0 ∧ m+1 < 0) := 
by
  -- Since (m+1) - (m-2) = 3, which is positive,
  -- m+1 > m-2, thus the statement ¬(m-2 > 0 ∧ m+1 < 0) holds.
  sorry

end point_not_in_fourth_quadrant_l469_469229


namespace least_common_multiple_of_first_ten_integers_l469_469379

theorem least_common_multiple_of_first_ten_integers : 
  (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → 2520 % n = 0) ∧ 
  (∀ m : ℕ, (∀ n : ℕ, n ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10} → m % n = 0) → 2520 ≤ m) :=
by
  sorry

end least_common_multiple_of_first_ten_integers_l469_469379


namespace binomial_expansion_properties_l469_469151

theorem binomial_expansion_properties (x : ℝ) (n : ℕ) (h1 : n = 8) :
  -- The term with the largest binomial coefficient in the expansion is the 5th term
  (∀ k : ℕ, (k ≠ 4 -> (nat.choose n k) <= (nat.choose n 4))) ∧
  -- The coefficient of the term containing x in the expansion is 112
  (binom n 2 * 4 = 112) :=
by
  sorry

end binomial_expansion_properties_l469_469151


namespace bus_people_count_l469_469353

noncomputable def people_on_bus_after_third_stop (initial : ℕ) (first_stop_off : ℕ) 
  (second_stop_off second_stop_on : ℕ) 
  (third_stop_off third_stop_on : ℕ) : ℕ :=
  let after_first_stop := initial - first_stop_off in
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on in
  after_second_stop - third_stop_off + third_stop_on

theorem bus_people_count : 
  people_on_bus_after_third_stop 50 15 8 2 4 3 = 28 :=
by
  unfold people_on_bus_after_third_stop
  sorry

end bus_people_count_l469_469353


namespace part1_part2_part3_l469_469135

noncomputable def a : ℕ → ℕ
| 1 => 2
| n => 2 ^ n

noncomputable def b : ℕ → ℕ
| 1 => 1
| n => 2 * n - 1

noncomputable def c (n : ℕ) : ℕ :=
a n * b n

noncomputable def T (n : ℕ) : ℕ :=
∑ i in finset.range n, c (i + 1)

theorem part1 : a 1 = 2 ∧ a 2 = 4 := by
  -- proof omitted
  sorry

theorem part2 : (∀ n, a n = 2 ^ n) ∧ (∀ n, b n = 2 * n - 1) := by
  -- proof omitted
  sorry

theorem part3 : ∀ n, T n = (2 * n - 3) * 2 ^ (n + 1) + 6 := by
  -- proof omitted
  sorry

end part1_part2_part3_l469_469135


namespace least_common_multiple_1_to_10_l469_469446

theorem least_common_multiple_1_to_10 : Nat.lcm (1 :: (List.range 10.tail)) = 2520 := 
by 
  sorry

end least_common_multiple_1_to_10_l469_469446


namespace lcm_first_ten_positive_integers_l469_469429

open Nat

theorem lcm_first_ten_positive_integers : lcm 1 (lcm 2 (lcm 3 (lcm 4 (lcm 5 (lcm 6 (lcm 7 (lcm 8 (lcm 9 10))))))))) = 2520 := by
  sorry

end lcm_first_ten_positive_integers_l469_469429


namespace area_of_bounded_region_l469_469789

theorem area_of_bounded_region :
  let r1 := {p : ℝ × ℝ | p.1 = 2}
  let r2 := {p : ℝ × ℝ | p.2 = 3}
  let x_axis := {p : ℝ × ℝ | p.2 = 0}
  let y_axis := {p : ℝ × ℝ | p.1 = 0}
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  area (region) = 6 :=
by
  have r1_def : r1 = {p : ℝ × ℝ | p.1 = 2} := rfl
  have r2_def : r2 = {p : ℝ × ℝ | p.2 = 3} := rfl
  have x_axis_def : x_axis = {p : ℝ × ℝ | p.2 = 0} := rfl
  have y_axis_def : y_axis = {p : ℝ × ℝ | p.1 = 0} := rfl
  let region := {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3}
  have region_def : region = {p : ℝ × ℝ | 0 ≤ p.1 ∧ p.1 ≤ 2 ∧ 0 ≤ p.2 ∧ p.2 ≤ 3} := rfl
  sorry

end area_of_bounded_region_l469_469789


namespace total_tiles_l469_469731

theorem total_tiles (n : ℕ) (h : 2 * n - 1 = 133) : n^2 = 4489 :=
by
  sorry

end total_tiles_l469_469731


namespace cannot_achieve_all_plus_l469_469060

def initial_grid : List (List Bool) :=
  [[true, true, true, false],  -- true represents '+'
   [true, true, true, true],   -- false represents '-'
   [true, true, true, true],
   [true, true, true, true]]

-- Define a function that performs the flipping operation on a row
def flip_row (grid : List (List Bool)) (row : Nat) : List (List Bool) :=
  grid.updated row (grid[row]!!.map (!))

-- Define a function that performs the flipping operation on a column
def flip_column (grid : List (List Bool)) (col : Nat) : List (List Bool) :=
  grid.map (λ row => row.updated col (!(row[col]!!)))

-- Define a function that performs the flipping operation on a diagonal
def flip_diagonal (grid : List (List Bool)) (main_diagonal : Bool) : List (List Bool) :=
  if main_diagonal then
    grid.mapi (λ i row => row.updated i (!(row[i]!!)))  -- Main diagonal
  else
    grid.mapi (λ i row => row.updated (3 - i) (!(row[3 - i]!!)))  -- Anti-diagonal

-- Define the main theorem: It is impossible to achieve a grid with all '+' signs
theorem cannot_achieve_all_plus : 
  ¬∃ k (ops : Fin k → List (List Bool) → List (List Bool)), 
   ∀ i, ops i initial_grid = initial_grid.map (λ row => row.map (λ _ => true)) :=
begin
  sorry
end

end cannot_achieve_all_plus_l469_469060


namespace intersection_correct_l469_469179

noncomputable def M : Set ℝ := {x | Real.log x > 0}
def N : Set ℝ := {x | -3 ≤ x ∧ x ≤ 3}
def intersection_M_N : Set ℝ := {x | 1 < x ∧ x ≤ 3}

theorem intersection_correct : M ∩ N = intersection_M_N :=
by
  sorry

end intersection_correct_l469_469179


namespace least_common_multiple_of_first_ten_positive_integers_l469_469470

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469470


namespace solve_for_x_l469_469119

  theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → x = 27 :=
  by
    sorry
  
end solve_for_x_l469_469119


namespace feet_per_inch_of_model_l469_469336

def height_of_statue := 75 -- in feet
def height_of_model := 5 -- in inches

theorem feet_per_inch_of_model : (height_of_statue / height_of_model) = 15 :=
by
  sorry

end feet_per_inch_of_model_l469_469336


namespace least_common_multiple_of_first_ten_positive_integers_l469_469465

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469465


namespace constant_term_binomial_expansion_n_6_middle_term_coefficient_l469_469815

open Nat

-- Define the binomial expansion term
def binomial_term (n : ℕ) (r : ℕ) (x : ℝ) : ℝ :=
  (Nat.choose n r) * (2 ^ r) * x^(2 * (n-r) - r)

-- (I) Prove the constant term of the binomial expansion when n = 6
theorem constant_term_binomial_expansion_n_6 :
  binomial_term 6 4 (1 : ℝ) = 240 := 
sorry

-- (II) Prove the coefficient of the middle term under given conditions
theorem middle_term_coefficient (n : ℕ) :
  (Nat.choose 8 2 = Nat.choose 8 6) →
  binomial_term 8 4 (1 : ℝ) = 1120 := 
sorry

end constant_term_binomial_expansion_n_6_middle_term_coefficient_l469_469815


namespace terms_before_one_l469_469185

-- Define the sequence parameters
def a : ℤ := 100
def d : ℤ := -7
def nth_term (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the target term we are interested in
def target_term : ℤ := 1

-- Define the main theorem
theorem terms_before_one : ∃ n : ℕ, nth_term n = target_term ∧ (n - 1) = 14 := by
  sorry

end terms_before_one_l469_469185


namespace ellipse_standard_equation_l469_469152

theorem ellipse_standard_equation
  (F : ℝ × ℝ)
  (e : ℝ)
  (eq1 : F = (0, 1))
  (eq2 : e = 1 / 2) :
  ∃ (a b : ℝ), a = 2 ∧ b ^ 2 = 3 ∧ (∀ x y : ℝ, (y ^ 2 / 4) + (x ^ 2 / 3) = 1) :=
by
  sorry

end ellipse_standard_equation_l469_469152


namespace bus_people_count_l469_469352

noncomputable def people_on_bus_after_third_stop (initial : ℕ) (first_stop_off : ℕ) 
  (second_stop_off second_stop_on : ℕ) 
  (third_stop_off third_stop_on : ℕ) : ℕ :=
  let after_first_stop := initial - first_stop_off in
  let after_second_stop := after_first_stop - second_stop_off + second_stop_on in
  after_second_stop - third_stop_off + third_stop_on

theorem bus_people_count : 
  people_on_bus_after_third_stop 50 15 8 2 4 3 = 28 :=
by
  unfold people_on_bus_after_third_stop
  sorry

end bus_people_count_l469_469352


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469389

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469389


namespace lcm_first_ten_numbers_l469_469595

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469595


namespace complex_product_identity_l469_469210

noncomputable def z1 : ℂ := sorry
noncomputable def z2 : ℂ := sorry

theorem complex_product_identity (h1 : |z1| = 2) (h2 : |z2| = 3)
  (h3 : 3 * z1 - 2 * z2 = 2 - (1 : ℂ) * I) : 
  z1 * z2 = - (30 / 13 : ℂ) + (72 / 13 : ℂ) * I :=
sorry

end complex_product_identity_l469_469210


namespace solve_for_x_l469_469101

theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → (x = 27) := by
  intro h
  sorry

end solve_for_x_l469_469101


namespace solve_for_x_l469_469100

theorem solve_for_x (x : ℝ) : (sqrt (5 * x + 9) = 12) → (x = 27) := by
  intro h
  sorry

end solve_for_x_l469_469100


namespace sqrt_geq_cbrt_l469_469129

theorem sqrt_geq_cbrt (a b c d : ℝ) (h : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) :
  sqrt ((a^2 + b^2 + c^2 + d^2) / 4) ≥ (√[(a * b * c + a * b * d + a * c * d + b * c * d) / 4] : ℝ) :=
by 
  sorry

end sqrt_geq_cbrt_l469_469129


namespace probability_of_X_eq_Y_l469_469739

noncomputable def probability_X_eq_Y : ℝ :=
  let lower_bound := (-5 * Real.pi / 2)
  let upper_bound := (5 * Real.pi / 2)
  let area_square := (upper_bound - lower_bound) ^ 2
  let valid_intersections := 5
  valid_intersections / area_square

theorem probability_of_X_eq_Y :
  ∀ X Y : ℝ,
    lower_bound ≤ X ∧ X ≤ upper_bound ∧
    lower_bound ≤ Y ∧ Y ≤ upper_bound ∧
    cos (sin X) = cos (sin Y) →
    probability_X_eq_Y = 1 / 5 := by
  sorry

end probability_of_X_eq_Y_l469_469739


namespace trajectory_of_circle_center_is_ellipse_l469_469826

noncomputable def circleTrajectory (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) : Set (ℝ × ℝ) :=
  { (x, y) | ∃ θ : ℝ, x^2 + y^2 - 2 * a * x * cos θ - 2 * b * y * sin θ = 0 }

theorem trajectory_of_circle_center_is_ellipse (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a ≠ b) :
  ∃ e : Set (ℝ × ℝ), ellipse e ∧ circleTrajectory a b h1 h2 h3 = e := sorry

end trajectory_of_circle_center_is_ellipse_l469_469826


namespace simplify_fraction_l469_469307

theorem simplify_fraction (x : ℝ) (h : x ≠ -1) : (x + 1) / (x^2 + 2 * x + 1) = 1 / (x + 1) :=
by
  sorry

end simplify_fraction_l469_469307


namespace least_common_multiple_of_first_ten_l469_469614

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469614


namespace least_common_multiple_first_ten_integers_l469_469565

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469565


namespace least_common_multiple_first_ten_integers_l469_469578

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469578


namespace amber_max_ounces_l469_469040

theorem amber_max_ounces :
  ∀ (money : ℝ) (candy_cost : ℝ) (candy_ounces : ℝ) (chips_cost : ℝ) (chips_ounces : ℝ),
    money = 7 →
    candy_cost = 1 →
    candy_ounces = 12 →
    chips_cost = 1.4 →
    chips_ounces = 17 →
    max (money / candy_cost * candy_ounces) (money / chips_cost * chips_ounces) = 85 :=
by
  intros money candy_cost candy_ounces chips_cost chips_ounces
  intros h_money h_candy_cost h_candy_ounces h_chips_cost h_chips_ounces
  sorry

end amber_max_ounces_l469_469040


namespace least_common_multiple_of_first_ten_positive_integers_l469_469468

theorem least_common_multiple_of_first_ten_positive_integers :
  Nat.lcm (List.range 10).map Nat.succ = 2520 :=
by
  sorry

end least_common_multiple_of_first_ten_positive_integers_l469_469468


namespace lcm_first_ten_numbers_l469_469580

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469580


namespace find_y_is_90_l469_469796

-- Definitions for given conditions
def angle_ABC : ℝ := 120
def angle_ABD : ℝ := 180 - angle_ABC
def angle_BDA : ℝ := 30

-- The theorem to prove y = 90 degrees
theorem find_y_is_90 :
  ∃ y : ℝ, angle_ABD = 60 ∧ angle_BDA = 30 ∧ (30 + 60 + y = 180) → y = 90 :=
by
  sorry

end find_y_is_90_l469_469796


namespace least_common_multiple_first_ten_l469_469556

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469556


namespace least_common_multiple_of_first_ten_l469_469600

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469600


namespace triangle_area_is_9_l469_469052

-- Define the vertices of the triangle
def x1 : ℝ := 1
def y1 : ℝ := 2
def x2 : ℝ := 4
def y2 : ℝ := 5
def x3 : ℝ := 6
def y3 : ℝ := 1

-- Define the area calculation formula for the triangle
def triangle_area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  0.5 * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

-- The proof statement
theorem triangle_area_is_9 :
  triangle_area x1 y1 x2 y2 x3 y3 = 9 :=
by
  sorry

end triangle_area_is_9_l469_469052


namespace lcm_first_ten_numbers_l469_469587

theorem lcm_first_ten_numbers : Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) = 2520 := 
by
  sorry

end lcm_first_ten_numbers_l469_469587


namespace domain_ln_x_sq_minus_2_l469_469993

theorem domain_ln_x_sq_minus_2 :
  {x : ℝ | x^2 - 2 > 0} = {x : ℝ | x < -real.sqrt 2} ∪ {x : ℝ | x > real.sqrt 2} := by
  sorry

end domain_ln_x_sq_minus_2_l469_469993


namespace least_common_multiple_1_to_10_l469_469496

theorem least_common_multiple_1_to_10 : 
  ∃ (x : ℕ), (∀ n, 1 ≤ n ∧ n ≤ 10 → n ∣ x) ∧ x = 2520 :=
by
  exists 2520
  intros n hn
  sorry

end least_common_multiple_1_to_10_l469_469496


namespace radius_of_fourth_circle_proof_l469_469917

noncomputable def radius_of_fourth_circle (a r : ℝ) : ℝ := a * r^3

theorem radius_of_fourth_circle_proof :
  let a := 10
  let r := real.sqrt (real.sqrt (real.sqrt 2))
  radius_of_fourth_circle a r = 10 * real.sqrt 2 :=
by
  let a := 10
  let r := real.sqrt (real.sqrt (real.sqrt 2))
  calc
    radius_of_fourth_circle a r
        = a * r^3 : rfl
    ... = 10 * (real.sqrt (real.sqrt (real.sqrt 2)))^3 : rfl
    ... = 10 * real.sqrt 2 : sorry

end radius_of_fourth_circle_proof_l469_469917


namespace log_sec_eq_neg_half_log_l469_469949

theorem log_sec_eq_neg_half_log (k x d : ℝ) (h1 : 1 < k) (h2 : 0 < sin x) (h3 : 0 < cos x) 
(h4 : log k (1 / sin x) = d) : log k (1 / cos x) = -(1 / 2) * log k (1 - k^(-2 * d)) := 
by
  sorry

end log_sec_eq_neg_half_log_l469_469949


namespace same_heads_probability_l469_469926

theorem same_heads_probability
  (fair_coin : Real := 1/2)
  (biased_coin : Real := 5/8)
  (prob_Jackie_eq_Phil : Real := 77/225) :
  let m := 77
  let n := 225
  (m : ℕ) + (n : ℕ) = 302 := 
by {
  -- The proof would involve constructing the generating functions,
  -- calculating the sum of corresponding coefficients and showing that the
  -- resulting probability reduces to 77/225
  sorry
}

end same_heads_probability_l469_469926


namespace least_common_multiple_of_first_ten_l469_469609

theorem least_common_multiple_of_first_ten :
  Nat.lcm (1 :: 2 :: 3 :: 4 :: 5 :: 6 :: 7 :: 8 :: 9 :: 10 :: List.nil) = 2520 := by
  sorry

end least_common_multiple_of_first_ten_l469_469609


namespace profitable_allocation_2015_l469_469676

theorem profitable_allocation_2015 :
  ∀ (initial_price : ℝ) (final_price : ℝ)
    (annual_interest_2015 : ℝ) (two_year_interest : ℝ) (annual_interest_2016 : ℝ),
  initial_price = 70 ∧ final_price = 85 ∧ annual_interest_2015 = 0.16 ∧
  two_year_interest = 0.15 ∧ annual_interest_2016 = 0.10 →
  (initial_price * (1 + annual_interest_2015) * (1 + annual_interest_2016) > final_price) ∨
  (initial_price * (1 + two_year_interest)^2 > final_price) :=
by
  intros initial_price final_price annual_interest_2015 two_year_interest annual_interest_2016
  intro h
  sorry

end profitable_allocation_2015_l469_469676


namespace inequality_solution_minimum_mn_l469_469169

-- Proof Problem 1
theorem inequality_solution (x : ℝ) : (| x - 1 | ≥ 4 - | x + 1 |) ↔ (x ≤ -2 ∨ x ≥ 2) :=
by
  sorry

-- Proof Problem 2
theorem minimum_mn (m n : ℝ) (hm : m > 0) (hn : n > 0) :
  (∀ x, (| x - 1 | ≤ 1) ↔ (0 ≤ x ∧ x ≤ 2)) ∧ (1 / m + 1 / (2 * n) = 1) →
  (mn ≥ 2) ∧ (∀ k l, (k > 0 → l > 0 → (1 / k + 1 / (2 * l) = 1) → k * l ≥ 2) → (m * n = 2)) :=
by
  sorry

end inequality_solution_minimum_mn_l469_469169


namespace coefficient_of_x2_is_248_l469_469774

noncomputable def coefficient_x2 : ℕ :=
  let poly := (λ (x : ℝ), x^2 - 3*x + 2)
  in (expand_expr 4 poly).coeff 2

theorem coefficient_of_x2_is_248 : coefficient_x2 = 248 :=
by
  sorry

end coefficient_of_x2_is_248_l469_469774


namespace least_common_multiple_first_ten_integers_l469_469572

theorem least_common_multiple_first_ten_integers : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 1 2) 3) 4) 5) 6) 7) 8) 9) 10 = 2520 :=
sorry

end least_common_multiple_first_ten_integers_l469_469572


namespace least_positive_integer_divisible_by_first_ten_integers_l469_469387

theorem least_positive_integer_divisible_by_first_ten_integers : ∃ n : ℕ, 
  (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ n) ∧ 
  (∀ m : ℕ, (∀ k ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, k ∣ m) → 2520 ≤ m) := 
sorry

end least_positive_integer_divisible_by_first_ten_integers_l469_469387


namespace least_common_multiple_first_ten_l469_469552

theorem least_common_multiple_first_ten : ∃ n, n = Nat.lcm 1 (Nat.lcm 2 (Nat.lcm 3 (Nat.lcm 4 (Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 7 (Nat.lcm 8 (Nat.lcm 9 10)))))))) ∧ n = 2520 := 
  sorry

end least_common_multiple_first_ten_l469_469552


namespace abc_value_l469_469150

variables (a b c d e f : ℝ)
variables (h1 : b * c * d = 65)
variables (h2 : c * d * e = 750)
variables (h3 : d * e * f = 250)
variables (h4 : (a * f) / (c * d) = 0.6666666666666666)

theorem abc_value : a * b * c = 130 :=
by { sorry }

end abc_value_l469_469150


namespace assignment_plans_l469_469047

theorem assignment_plans {students towns : ℕ} (h_students : students = 5) (h_towns : towns = 3) :
  ∃ plans : ℕ, plans = 150 :=
by
  -- Given conditions
  have h1 : students = 5 := h_students
  have h2 : towns = 3 := h_towns
  
  -- The required number of assignment plans
  existsi 150
  -- Proof is not supplied
  sorry

end assignment_plans_l469_469047


namespace find_x_l469_469112

theorem find_x (x : ℝ) (h : sqrt (5 * x + 9) = 12) : x = 27 :=
begin
  sorry
end

end find_x_l469_469112
