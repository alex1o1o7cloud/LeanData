import Mathlib

namespace solve_equation_l352_352956

theorem solve_equation (x : ℝ) (h : x^2 - x + 1 ≠ 0) :
  (x^2 + x + 1 = 1 / (x^2 - x + 1)) ↔ x = 1 ∨ x = -1 :=
by sorry

end solve_equation_l352_352956


namespace negation_of_P_l352_352826

variable (x : ℝ)

def P : Prop := ∀ x : ℝ, x^2 + 2*x + 3 ≥ 0

theorem negation_of_P : ¬P ↔ ∃ x : ℝ, x^2 + 2*x + 3 < 0 :=
by sorry

end negation_of_P_l352_352826


namespace number_of_ordered_pairs_l352_352417

theorem number_of_ordered_pairs (m n : ℕ) (h_pos_m : 0 < m) (h_pos_n : 0 < n)
  (h_div : m * n ∣ 2008 * 2009 * 2010) :
  ∃ k, k = 480 := 
begin
  sorry
end

end number_of_ordered_pairs_l352_352417


namespace median_service_time_l352_352265

/-- Given a group of 10 volunteers from a certain high school youth volunteer association 
    and their service times for a week, we prove that the median service time is 4 hours. -/
theorem median_service_time (hrs : Fin 10 → ℕ)
  (h : hrs = ![2, 3, 3, 3, 4, 4, 5, 5, 5, 6]) : 
  median (Fin 10) hrs = 4 :=
by
  sorry

end median_service_time_l352_352265


namespace symmetric_points_on_parabola_l352_352472

theorem symmetric_points_on_parabola (x1 x2 y1 y2 m : ℝ)
  (h1: y1 = 2 * x1 ^ 2)
  (h2: y2 = 2 * x2 ^ 2)
  (h3: x1 * x2 = -1 / 2)
  (h4: y2 - y1 = 2 * (x2 ^ 2 - x1 ^ 2))
  (h5: (x1 + x2) / 2 = -1 / 4)
  (h6: (y1 + y2) / 2 = (x1 + x2) / 2 + m) :
  m = 3 / 2 :=
by
  sorry

end symmetric_points_on_parabola_l352_352472


namespace negation_is_correct_l352_352193

-- Define the condition: we have two integers a and b
variables (a b : ℤ)

-- Original proposition: If the sum of two integers is even, then both integers are even.
def original_proposition := (a + b) % 2 = 0 → (a % 2 = 0) ∧ (b % 2 = 0)

-- Negation of the proposition: There exist two integers such that their sum is even and not both are even.
def negation_of_proposition := (a + b) % 2 = 0 ∧ ¬((a % 2 = 0) ∧ (b % 2 = 0))

theorem negation_is_correct :
  ¬ original_proposition a b = negation_of_proposition a b :=
by
  sorry

end negation_is_correct_l352_352193


namespace field_total_area_l352_352653

noncomputable def total_area_of_field (side_length : ℝ) : ℝ :=
  let area_square := side_length ^ 2
  let radius_semicircle := side_length / 2
  let area_semicircle := (1 / 2) * Real.pi * radius_semicircle ^ 2
  area_square + area_semicircle

theorem field_total_area (h_side_length: 17) : total_area_of_field 17 ≈ 402.5 := 
by
  sorry

end field_total_area_l352_352653


namespace sum_of_squares_of_divisors_1800_l352_352406

def sum_of_squares_of_divisors (n : ℕ) : ℕ :=
  (List.range (n + 1)).filter (λ d, d > 0 ∧ n % d = 0).map (λ d, d * d).sum

theorem sum_of_squares_of_divisors_1800 :
  sum_of_squares_of_divisors 1800 = 5035485 :=
by
  sorry

end sum_of_squares_of_divisors_1800_l352_352406


namespace truncated_pyramid_diagonals_intersect_at_one_point_l352_352204

/--
Given a truncated pyramid with parallelogram bases ABCD and A'B'C'D' such that 
each pair of corresponding points (A, A'), (B, B'), (C, C'), (D, D') are in parallel planes, 
prove that the space diagonals intersect at one common point.
-/
theorem truncated_pyramid_diagonals_intersect_at_one_point
  (A B C D A' B' C' D' : Point)
  (h_parallelogram : Parallelogram ABCD)
  (h_parallelogram' : Parallelogram A'B'C'D')
  (h_parallel_planes : ParallelPlanes (Plane ABCD) (Plane A'B'C'D')) :
  ∃ O : Point, 
  (Line AC).intersection (Line A'C') = O ∧
  (Line BD).intersection (Line B'D') = O ∧
  (Line AD).intersection (Line A'D') = O ∧
  (Line BC).intersection (Line B'C') = O := 
sorry

end truncated_pyramid_diagonals_intersect_at_one_point_l352_352204


namespace total_books_for_girls_l352_352632

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l352_352632


namespace arithmetic_sequence_problem_l352_352995

variable {a b : ℕ → ℕ}
variable (S T : ℕ → ℕ)

-- Conditions
def condition (n : ℕ) : Prop :=
  S n / T n = (2 * n + 1) / (3 * n + 2)

-- Conjecture to prove
theorem arithmetic_sequence_problem (h : ∀ n, condition S T n) :
  (a 3 + a 11 + a 19) / (b 7 + b 15) = 129 / 130 := 
by
  sorry

end arithmetic_sequence_problem_l352_352995


namespace red_button_probability_l352_352905

/-
Mathematical definitions derived from the problem:
Initial setup:
- Jar A has 6 red buttons and 10 blue buttons.
- Same number of red and blue buttons are removed. Jar A retains 3/4 of original buttons.
- Calculate the final number of red buttons in Jar A and B, and determine the probability both selected buttons are red.
-/
theorem red_button_probability :
  let initial_red := 6
  let initial_blue := 10
  let total_buttons := initial_red + initial_blue
  let removal_fraction := 3 / 4
  let final_buttons := (3 / 4 : ℚ) * total_buttons
  let removed_buttons := total_buttons - final_buttons
  let removed_each_color := removed_buttons / 2
  let final_red_A := initial_red - removed_each_color
  let final_red_B := removed_each_color
  let prob_red_A := final_red_A / final_buttons
  let prob_red_B := final_red_B / removed_buttons
  prob_red_A * prob_red_B = 1 / 6 :=
by
  sorry

end red_button_probability_l352_352905


namespace binom_12_10_l352_352349

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l352_352349


namespace length_of_BC_l352_352292

theorem length_of_BC 
  (parabola : ∀ (x : ℝ), (x, x^2) ∈ set.univ : set (ℝ × ℝ))
  (A : (1, 1) ∈ set.univ)
  (BC_parallel_x : ∀ (a : ℝ), ∃ (B C : ℝ × ℝ), B = (-a, a^2) ∧ C = (a, a^2) ∧ B ≠ C)
  (area_ABC : 32 = (1/2) * (2 * a) * (a^2 - 1)) :
  ∃ (a : ℝ), 2 * a = 8 :=
by
  sorry

end length_of_BC_l352_352292


namespace find_m_l352_352802

def line (m : ℝ) (x y : ℝ) : Prop := m * x + y - 1 = 0

def circle (x y : ℝ) : Prop := x^2 + y^2 - 4 * x + 2 * y + 1 = 0

theorem find_m (m : ℝ) (c_x c_y : ℝ) (h₁ : circle c_x c_y) (h₂ : line m c_x c_y) : m = 1 :=
sorry

end find_m_l352_352802


namespace cousins_room_distributions_l352_352128

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l352_352128


namespace negate_prop_l352_352183

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l352_352183


namespace binom_12_10_l352_352351

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l352_352351


namespace minimum_abs_a_plus_b_l352_352480

theorem minimum_abs_a_plus_b {a b : ℤ} (h1 : |a| < |b|) (h2 : |b| ≤ 4) : ∃ (a b : ℤ), |a| + b = -4 :=
by
  sorry

end minimum_abs_a_plus_b_l352_352480


namespace number_of_boys_in_fifth_grade_l352_352897

-- Defining the conditions
def total_students := 450
def students_playing_soccer := 250
def percent_boys_playing_soccer := 0.86
def girls_not_playing_soccer := 95

-- Mathematically equivalent proof problem statement
theorem number_of_boys_in_fifth_grade :
  let boys_playing_soccer := percent_boys_playing_soccer * students_playing_soccer in
  let students_not_playing_soccer := total_students - students_playing_soccer in
  let boys_not_playing_soccer := students_not_playing_soccer - girls_not_playing_soccer in
  let total_boys := boys_playing_soccer + boys_not_playing_soccer in
  total_boys = 320 :=
by
  sorry

end number_of_boys_in_fifth_grade_l352_352897


namespace jessica_total_cost_l352_352906

def price_of_cat_toy : ℝ := 10.22
def price_of_cage : ℝ := 11.73
def price_of_cat_food : ℝ := 5.65
def price_of_catnip : ℝ := 2.30
def discount_rate : ℝ := 0.10
def tax_rate : ℝ := 0.07

def discounted_price_of_cat_toy : ℝ := price_of_cat_toy * (1 - discount_rate)
def total_cost_before_tax : ℝ := discounted_price_of_cat_toy + price_of_cage + price_of_cat_food + price_of_catnip
def sales_tax : ℝ := total_cost_before_tax * tax_rate
def total_cost_after_discount_and_tax : ℝ := total_cost_before_tax + sales_tax

theorem jessica_total_cost : total_cost_after_discount_and_tax = 30.90 := by
  sorry

end jessica_total_cost_l352_352906


namespace find_BD_l352_352921

theorem find_BD
  (A B C D : Point)
  (h_triangle : right_triangle A B C)
  (h_right_angle_B : angle A B C = 90)
  (h_circle : circle (midpoint B C) (dist B C / 2))
  (h_D_intersection : lies_on D (segment A C))
  (h_area : triangle_area A B C = 150)
  (h_AC : dist A C = 25)
  : dist B D = 12 := by
  sorry

end find_BD_l352_352921


namespace james_prom_cost_l352_352089

def total_cost (ticket_cost dinner_cost tip_percent limo_cost_per_hour limo_hours tuxedo_cost persons : ℕ) : ℕ :=
  (ticket_cost * persons) +
  ((dinner_cost * persons) + (tip_percent * dinner_cost * persons) / 100) +
  (limo_cost_per_hour * limo_hours) + tuxedo_cost

theorem james_prom_cost :
  total_cost 100 120 30 80 8 150 4 = 1814 :=
by
  sorry

end james_prom_cost_l352_352089


namespace base4_more_digits_than_base9_l352_352855

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l352_352855


namespace fraction_arithmetic_proof_l352_352222

theorem fraction_arithmetic_proof :
  (7 / 6) + (5 / 4) - (3 / 2) = 11 / 12 :=
by sorry

end fraction_arithmetic_proof_l352_352222


namespace harry_weekly_earnings_l352_352051

def dogs_walked_per_day : Nat → Nat
| 1 => 7  -- Monday
| 2 => 12 -- Tuesday
| 3 => 7  -- Wednesday
| 4 => 9  -- Thursday
| 5 => 7  -- Friday
| _ => 0  -- Other days (not relevant for this problem)

def payment_per_dog : Nat := 5

def daily_earnings (day : Nat) : Nat :=
  dogs_walked_per_day day * payment_per_dog

def total_weekly_earnings : Nat :=
  (daily_earnings 1) + (daily_earnings 2) + (daily_earnings 3) +
  (daily_earnings 4) + (daily_earnings 5)

theorem harry_weekly_earnings : total_weekly_earnings = 210 :=
by
  sorry

end harry_weekly_earnings_l352_352051


namespace matrix_A_to_power_4_l352_352328

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l352_352328


namespace matrix_pow_four_l352_352325

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l352_352325


namespace all_cubes_have_common_point_construct_cubes_no_common_point_l352_352426

-- Definitions for the first problem
def cube (x_min x_max y_min y_max z_min z_max : ℝ) := 
  { p : ℝ × ℝ × ℝ | x_min < p.1 ∧ p.1 < x_max ∧ y_min < p.2 ∧ p.2 < y_max ∧ z_min < p.3 ∧ p.3 < z_max }

def cubes_have_common_point (cubes : list (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)) :=
  ∃ (p : ℝ × ℝ × ℝ), ∀ c ∈ cubes, (∃ (x_min x_max y_min y_max z_min z_max : ℝ), 
  c = (x_min, x_max, y_min, y_max, z_min, z_max) ∧ cube x_min x_max y_min y_max z_min z_max p)

-- Statement for the first proof problem
theorem all_cubes_have_common_point :
  ∀ (cubes : list (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)),
  (∀ c1 c2 ∈ cubes, c1 ≠ c2 → ∃ (p : ℝ × ℝ × ℝ), 
    (∃ (x_min1 x_max1 y_min1 y_max1 z_min1 z_max1 : ℝ), 
      c1 = (x_min1, x_max1, y_min1, y_max1, z_min1, z_max1) ∧ cube x_min1 x_max1 y_min1 y_max1 z_min1 z_max1 p) ∧
    (∃ (x_min2 x_max2 y_min2 y_max2 z_min2 z_max2 : ℝ), 
      c2 = (x_min2, x_max2, y_min2, y_max2, z_min2, z_max2) ∧ cube x_min2 x_max2 y_min2 y_max2 z_min2 z_max2 p)) →
  length cubes = 100 →
  cubes_have_common_point cubes := 
by
  sorry

-- Definitions for the second problem
def cubes_three_intersect (cubes : list (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)) :=
  ∀ c1 c2 c3 ∈ cubes, c1 ≠ c2 → c2 ≠ c3 → c1 ≠ c3 → ∃ (p : ℝ × ℝ × ℝ), 
    (∃ (x_min1 x_max1 y_min1 y_max1 z_min1 z_max1 : ℝ), 
      c1 = (x_min1, x_max1, y_min1, y_max1, z_min1, z_max1) ∧ cube x_min1 x_max1 y_min1 y_max1 z_min1 z_max1 p) ∧
    (∃ (x_min2 x_max2 y_min2 y_max2 z_min2 z_max2 : ℝ), 
      c2 = (x_min2, x_max2, y_min2, y_max2, z_min2, z_max2) ∧ cube x_min2 x_max2 y_min2 y_max2 z_min2 z_max2 p) ∧
    (∃ (x_min3 x_max3 y_min3 y_max3 z_min3 z_max3 : ℝ), 
      c3 = (x_min3, x_max3, y_min3, y_max3, z_min3, z_max3) ∧ cube x_min3 x_max3 y_min3 y_max3 z_min3 z_max3 p)

-- Statement for the second proof problem
theorem construct_cubes_no_common_point :
  ∃ (cubes : list (ℝ × ℝ × ℝ × ℝ × ℝ × ℝ)),
  cubes_three_intersect cubes ∧
  ¬cubes_have_common_point cubes ∧
  length cubes = 100 :=
by
  sorry

end all_cubes_have_common_point_construct_cubes_no_common_point_l352_352426


namespace winning_team_played_draw_l352_352518

theorem winning_team_played_draw 
  (teams : Fin 16 → ℕ) 
  (points_distribution : ∀ i : Fin 16, teams i ≤ 27 ∧ ∀ j : Fin 16, i ≠ j → teams i ≠ teams j) 
  (seventh_team_points : teams ⟨6, Nat.lt_succ_self _⟩ = 21)
  (points_sum : ∑ i, teams i = 240) :
  ∃ j : Fin 16, teams j = 27 ∧ has_draw j :=
by
  -- Proof goes here
  sorry

end winning_team_played_draw_l352_352518


namespace cost_price_percentage_to_marked_price_l352_352170

def discount : ℝ := 0.20
def gain_percent : ℝ := 1.2222222222222223
def marked_price : ℝ := MP
def cost_price : ℝ := CP
def selling_price (MP : ℝ) : ℝ := MP * (1 - discount)

theorem cost_price_percentage_to_marked_price (MP CP : ℝ) 
  (h_gain : gain_percent = (selling_price MP - CP) / CP) :
  (CP / MP) * 100 = 36 :=
by
  sorry

end cost_price_percentage_to_marked_price_l352_352170


namespace calculate_perimeter_l352_352689

noncomputable def length_square := 8
noncomputable def breadth_square := 8 -- since it's a square, length and breadth are the same
noncomputable def length_rectangle := 8
noncomputable def breadth_rectangle := 4

noncomputable def combined_length := length_square + length_rectangle
noncomputable def combined_breadth := breadth_square 

noncomputable def perimeter := 2 * (combined_length + combined_breadth)

theorem calculate_perimeter : 
  length_square = 8 ∧ 
  breadth_square = 8 ∧ 
  length_rectangle = 8 ∧ 
  breadth_rectangle = 4 ∧ 
  perimeter = 48 := 
by 
  sorry

end calculate_perimeter_l352_352689


namespace binom_12_10_l352_352346

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l352_352346


namespace math_problem_l352_352865

theorem math_problem (x y a b c : ℝ) (h1 : x = -y) (h2 : a * b = 1) (h3 : |c| = 2) :
  (\left( \frac{x + y}{2} \right) ^ 2023 - (-a * b) ^ 2023 + c ^ 3 = 9) ∨
  (\left( \frac{x + y}{2} \right) ^ 2023 - (-a * b) ^ 2023 + c ^ 3 = -7) :=
by
  sorry

end math_problem_l352_352865


namespace area_triangle_PQS_l352_352537

-- Definitions: Trapezoid, Area of Trapezoid, Relationship between PQ and RS
variables {P Q R S : Type}

-- PQRS is a trapezoid
axiom trapezoidPQRS : Trapezoid PQRS

-- The area of PQRS is 18
axiom areaPQRS_is_18 : ∃ area : ℝ, area = 18

-- RS is three times the length of PQ
axiom RS_three_times_PQ : ∃ lPQ lRS : ℝ, lRS = 3 * lPQ

-- The statement to be proved
theorem area_triangle_PQS : ∃ area_triangle_PQS : ℝ, area_triangle_PQS = 4.5 :=
sorry

end area_triangle_PQS_l352_352537


namespace remainder_of_division_l352_352221

theorem remainder_of_division :
  ∀ (dividend divisor quotient remainder : ℕ), 
  dividend = (divisor * quotient + remainder) →
  dividend = 166 →
  divisor = 20 →
  quotient = 8 →
  remainder = 6 :=
by
  intros dividend divisor quotient remainder h_eq h_dividend h_divisor h_quotient
  have h1 : 166 = 20 * 8 + remainder, from h_eq.trans (by rw [h_dividend, h_divisor, h_quotient])
  have h2 : 166 = 160 + remainder, by rw [mul_comm, h1]
  have h3 : remainder = 6, by linarith
  exact h3

end remainder_of_division_l352_352221


namespace multiple_of_2018_with_2017_prefix_l352_352390

theorem multiple_of_2018_with_2017_prefix : ∃ n : ℕ, (2018 * n).digits.take 4 = [2, 0, 1, 7] :=
by
  let n := 9996
  existsi n
  sorry

end multiple_of_2018_with_2017_prefix_l352_352390


namespace find_remainder_l352_352929

-- Definitions based on the conditions
variables {R : Type*} [CommRing R] (p : R[X])

-- Conditions
def cond1 : p.eval 1 = 4 :=
by sorry -- Skip the proof for now

def cond2 : p.eval 2 = -3 :=
by sorry -- Skip the proof for now

def cond3 : p.eval (-3) = 1 :=
by sorry -- Skip the proof for now

-- The statement to prove
theorem find_remainder :
  (∀ q : R[X], ∃ r : R[X], r.degree < 3 ∧ p = q * (X - 1) * (X - 2) * (X + 3) + r) →
  (p % ((X - 1) * (X - 2) * (X + 3)) = X^2 - 10 * X + 13) :=
by {
  intros h q,
  have h1 := cond1,
  have h2 := cond2,
  have h3 := cond3,
  sorry -- Skipping the proof
}

end find_remainder_l352_352929


namespace tan_C_l352_352875

theorem tan_C (A B C : ℝ) (hABC : A + B + C = π) (tan_A : Real.tan A = 1 / 2) 
  (cos_B : Real.cos B = 3 * Real.sqrt 10 / 10) : Real.tan C = -1 :=
by
  sorry

end tan_C_l352_352875


namespace probability_triangle_area_l352_352682

noncomputable def probability_greater_area (P : Point) (ABC : Triangle) : ℝ :=
if is_inside P ABC 
then if area_of_triangle (triangle_ABP P ABC) > area_of_triangle (triangle_ACP P ABC) ∧ area_of_triangle (triangle_ABP P ABC) > area_of_triangle (triangle_BCP P ABC)
     then 1 / 3 else 0
else 0

theorem probability_triangle_area :
  ∀ (P : Point) (ABC : EquilateralTriangle), 
  (is_inside P ABC) →
  (∃ (prob : ℝ), probability_greater_area P ABC = 1 / 3) :=
by
  sorry

end probability_triangle_area_l352_352682


namespace cars_in_group_l352_352887

open Nat

theorem cars_in_group (C : ℕ) : 
  (47 ≤ C) →                  -- At least 47 cars in the group
  (53 ≤ C) →                  -- At least 53 cars in the group
  C ≥ 100 :=                  -- Conclusion: total cars is at least 100
by
  -- Begin the proof
  sorry                       -- Skip proof for now

end cars_in_group_l352_352887


namespace remainder_of_polynomial_division_l352_352652

noncomputable def p : ℚ[X] := 5*X^9 - 3*X^7 + 4*X^6 - 8*X^4 + 3*X^3 - 6*X + 5
noncomputable def d : ℚ[X] := 3*X - 6

theorem remainder_of_polynomial_division : p.eval 2 = 2321 := by
  sorry

end remainder_of_polynomial_division_l352_352652


namespace binom_12_10_l352_352345

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l352_352345


namespace sequence_property_l352_352829

def sequence (n : ℕ) : ℝ :=
match n with
| 0     => 0
| (n+1) => if n = 0 then 1 else sqrt (n+1) - sqrt n

theorem sequence_property:
  a1: sequence 1 = 1 ∧
  a2: sequence 2 = sqrt 2 - 1 ∧
  a3: sequence 3 = sqrt 3 - sqrt 2 ∧
  a4: sequence 4 = 2 - sqrt 3 ∧
  (∀ n : ℕ, n > 0 → sequence (n+1) + sequence n = sqrt (n+1) - sqrt (n-1)) ∧
  ∀ n : ℕ, n > 0 → sequence n = sqrt n - sqrt (n-1) :=
by
  sorry

end sequence_property_l352_352829


namespace count_multiples_l352_352053

theorem count_multiples (n : ℕ) (h_n : n = 300) :
  let multiples_of_2_and_5 := {m ∈ finset.range (n + 1) | m % 10 = 0}
  let multiples_of_2_and_5_not_3 := multiples_of_2_and_5.filter (λ m, m % 3 ≠ 0)
  let multiples_of_2_and_5_not_3_or_11 := multiples_of_2_and_5_not_3.filter (λ m, m % 11 ≠ 0)
  multiples_of_2_and_5_not_3_or_11.card = 18 :=
by {
  sorry
}

end count_multiples_l352_352053


namespace probability_no_adjacent_same_rolls_l352_352764

theorem probability_no_adjacent_same_rolls : 
  let A := [0, 1, 2, 3, 4, 5] -- Representing six faces of a die
  let rollings : List (A → ℕ) -- Each person rolls and the result is represented as a map from faces to counts (a distribution in effect)
  ∃ rollings : List (A → ℕ), 
    (∀ (i : Fin 5), rollings[i] ≠ rollings[(i + 1) % 5]) →
      probability rollings
    = 375 / 2592 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352764


namespace least_sum_of_exponents_of_powers_of_2_l352_352488

theorem least_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ s : Finset ℕ, (∑ x in s, 2^x = n) ∧ (∀ t : Finset ℕ, (∑ x in t, 2^x = n) → s.sum id ≤ t.sum id) :=
sorry

end least_sum_of_exponents_of_powers_of_2_l352_352488


namespace area_of_union_of_original_and_reflection_l352_352433

def area (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

noncomputable def reflected_area_union : ℝ :=
  let A := (2, 4)
  let B := (5, 3)
  let C := (7, 8)
  let A' := (4, 2)
  let B' := (3, 5)
  let C' := (8, 7)
  area 2 4 5 3 7 8 + area 4 2 3 5 8 7

theorem area_of_union_of_original_and_reflection : reflected_area_union = 17 := by
  sorry

end area_of_union_of_original_and_reflection_l352_352433


namespace simplify_sqrt_expression_l352_352586

theorem simplify_sqrt_expression :
  sqrt (12 + 8 * sqrt 3) - sqrt (12 - 8 * sqrt 3) = 2 * sqrt 3 + 2 * sqrt 6 :=
by
  -- proof goes here
  sorry

end simplify_sqrt_expression_l352_352586


namespace replaced_person_weight_l352_352968

theorem replaced_person_weight (new_person_weight : ℝ) (average_increase : ℝ) (num_persons : ℕ)
  (h1 : new_person_weight = 85)
  (h2 : average_increase = 2.5)
  (h3 : num_persons = 8) :
  let weight_increase := average_increase * num_persons in
  let replaced_person_weight := new_person_weight - weight_increase in
  replaced_person_weight = 65 := by
  sorry

end replaced_person_weight_l352_352968


namespace time_to_pass_pole_is_correct_l352_352660

def speed_kmph_to_mps (speed_kmph : Float) : Float :=
  speed_kmph * (1000 / 3600)

def time_to_pass_pole (train_length_m : Float) (train_speed_kmph : Float) : Float :=
  train_length_m / speed_kmph_to_mps(train_speed_kmph)

theorem time_to_pass_pole_is_correct :
  time_to_pass_pole 140 98 ≈ 5.14 := by
  sorry

end time_to_pass_pole_is_correct_l352_352660


namespace total_people_museum_l352_352199

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end total_people_museum_l352_352199


namespace split_subsets_into_equal_classes_l352_352556

-- defining the context of the conditions
variables {n k : ℕ} (h_nk : n > k ∧ k ≥ 1) {p : ℕ} (hp : Nat.Prime p)
  (h_div : p ∣ Nat.choose n k)

-- the theorem statement
theorem split_subsets_into_equal_classes :
  ∃ (C : Fin p → Set (Finset (Fin n))), 
    (∀ i, ∃ s : Finset (Fin n), s.card = k ∧ 
             ∀ t ∈ C i, t.card = k ∧ Finset.sum t id = Finset.sum s id) ∧
    (∀ i j, i ≠ j → C i ∩ C j = ∅) ∧
    (∀ i j, i ≠ j → ((C i).card = (C j).card)) :=
sorry

end split_subsets_into_equal_classes_l352_352556


namespace volume_of_cone_half_sector_l352_352266

-- Define the basic parameters of the problem
def radius : ℝ := 6
def sector_arc_length : ℝ := Real.pi * radius -- The arc length would be half the circumference
def cone_base_radius : ℝ := sector_arc_length / (2 * Real.pi)
def cone_height : ℝ := Real.sqrt (radius^2 - cone_base_radius^2)
def cone_volume : ℝ := (1 / 3) * Real.pi * (cone_base_radius^2) * cone_height

theorem volume_of_cone_half_sector (r : ℝ) (arc_length : ℝ) (base_radius : ℝ) (height : ℝ) (volume : ℝ) :
  r = 6 → arc_length = 6 * Real.pi → base_radius = 3 → height = 3 * Real.sqrt(3) → volume = 9 * Real.pi * Real.sqrt(3) :=
by
  intros hr harc hbase hheight hvol
  rw [hr, harc, hbase, hheight, hvol]
  sorry

end volume_of_cone_half_sector_l352_352266


namespace triangle_EF_plus_FD_eq_AB_l352_352545

theorem triangle_EF_plus_FD_eq_AB
  (A B C D E F : Point)
  (h_TRI_ABC : Triangle ABC)
  (h1 : ∠ BAC > (1/2) * ∠ ACB)
  (h2 : ∠ ABC > (1/2) * ∠ ACB)
  (hD_on_BC : D ∈ line BC)
  (hE_on_AC : E ∈ line AC)
  (h3 : ∠ BAD = (1/2) * ∠ ACB = ∠ EBA)
  (hF_bisects_ACB : F = (bisector ∠ ACB) ∩ line AB) :
  dist E F + dist F D = dist A B :=
by
  sorry

end triangle_EF_plus_FD_eq_AB_l352_352545


namespace probability_no_adjacent_same_rolls_l352_352770

theorem probability_no_adjacent_same_rolls :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6)
  let no_adjacent_same := outcomes.filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ a)
  (no_adjacent_same.card : ℚ) / outcomes.card = 25 / 108 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352770


namespace variance_is_0_02_l352_352673

def data_points : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

noncomputable def mean (l : List ℝ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℝ) : ℝ :=
  let m := mean l
  (l.map (λ x => (x - m) ^ 2)).sum / l.length

theorem variance_is_0_02 : variance data_points = 0.02 :=
by
  sorry

end variance_is_0_02_l352_352673


namespace cousins_rooms_distribution_l352_352111

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l352_352111


namespace base4_more_digits_than_base9_l352_352860

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l352_352860


namespace find_k_l352_352442

noncomputable def collinear (v1 v2 : Vector ℝ) : Prop :=
  ∃ λ : ℝ, v1 = λ • v2

theorem find_k
  (e1 e2 : Vector ℝ)
  (h_non_collinear : ¬collinear e1 e2)
  (h_collinear : collinear (e1 - 4 • e2) (k • e1 + e2)) :
  k = -1/4 :=
sorry

end find_k_l352_352442


namespace least_distance_PQ_l352_352517

-- Define the vertices of the tetrahedron in a 3D space.
structure Point := (x y z : ℝ)
def A : Point := ⟨0, 0, 0⟩
def B : Point := ⟨1, 0, 0⟩
def C : Point := ⟨1/2, (√3)/2, 0⟩
def D : Point := ⟨1/2, (√3)/6, (√6)/3⟩

-- Define the points P and Q as described in the problem.
def P : Point := ⟨2/3, 0, 0⟩
def Q : Point := ⟨2/3, √3/6, 2√6/9⟩

-- Function to calculate the distance between two points in 3D space.
def distance (P Q : Point) : ℝ := 
  sqrt ((P.x - Q.x)^2 + (P.y - Q.y)^2 + (P.z - Q.z)^2)

-- The theorem statement to prove the question.
theorem least_distance_PQ : distance P Q = (√2)/3 :=
by
  sorry -- Proof is omitted.

end least_distance_PQ_l352_352517


namespace march_1_falls_on_friday_l352_352883

-- Definitions of conditions
def march_days : ℕ := 31
def mondays_in_march : ℕ := 4
def thursdays_in_march : ℕ := 4

-- Lean 4 statement to prove March 1 falls on a Friday
theorem march_1_falls_on_friday 
  (h1 : march_days = 31)
  (h2 : mondays_in_march = 4)
  (h3 : thursdays_in_march = 4)
  : ∃ d : ℕ, d = 5 :=
by sorry

end march_1_falls_on_friday_l352_352883


namespace coprime_gcd_l352_352943

theorem coprime_gcd (a b : ℤ) (h : Int.gcd a b = 1) : Int.gcd (2 * a + b) (a * (a + b)) = 1 := 
sorry

end coprime_gcd_l352_352943


namespace average_age_of_class_l352_352602

theorem average_age_of_class 
  (avg_age_8 : ℕ → ℕ)
  (avg_age_6 : ℕ → ℕ)
  (age_15th : ℕ)
  (A : ℕ)
  (h1 : avg_age_8 8 = 112)
  (h2 : avg_age_6 6 = 96)
  (h3 : age_15th = 17)
  (h4 : 15 * A = (avg_age_8 8) + (avg_age_6 6) + age_15th)
  : A = 15 :=
by
  sorry

end average_age_of_class_l352_352602


namespace incircle_center_X_projection_l352_352140

noncomputable def triangle := {A B C D X : Type} 
    [metric_space X] 
    (A B C D X : X) 
    (hA B C : A ≠ B ∧ B ≠ C ∧ A ≠ C)
    (D_on_AB : is_collinear ({A, B, D} : set X))
    (I1 I2 : X)
    (incircle_ACD : circle I1 (triangle Incircle A C D))
    (incircle_BCD : circle I2 (triangle Incircle B C D))
    (I1_Touches : tangent incircle_ACD (circumcircle (triangle A C D)) (X))
    (I2_Touches : tangent incircle_BCD (circumcircle (triangle B C D)) (X))
    (X_common : collinear {D, X}) : Prop :=
∃ I : X,
  triangle Incircle A B C I ∧
  perpendicular_projection X (line_span {A, B}) I

axiom perpendicular_projection :
  ∀ {X : Type} [metric_space X] (P Q R : X),
  collinear {Q, R} → perpendicular P Q → perpendicular P R → P

theorem incircle_center_X_projection :
  ∀ {X : Type} [metric_space X] (A B C D X I1 I2 : X)
  (hA B C : A ≠ B ∧ B ≠ C ∧ A ≠ C)
  (D_on_AB : collinear {A, B, D})
  (incircle_ACD : circle I1 (triangle Incircle A C D))
  (incircle_BCD : circle I2 (triangle Incircle B C D))
  (I1_common : tangent incircle_ACD (circumcircle (triangle A C D)) (X))
  (I2_common : tangent incircle_BCD (circumcircle (triangle B C D)) (X)),
  exists (I : X), 
  triangle Incircle A B C I ∧ 
  perpendicular_projection X (line_span {A, B}) I :=
begin
  sorry
end

end incircle_center_X_projection_l352_352140


namespace cousins_rooms_distribution_l352_352113

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l352_352113


namespace function_properties_l352_352785

variable (X Y : Type) (f : X → Y)

theorem function_properties :
    ¬(∃ (A : Y → Prop), (∃ (y : Y), ∀ (x : X), f x ≠ y) ∨
                        (∀ (x1 x2 : X), f x1 = f x2 → x1 = x2) ∨
                        (Y = ∅)) :=
by
  sorry

end function_properties_l352_352785


namespace quadratic_form_solution_l352_352467

noncomputable def quadratic_function (a b c x : ℝ) : ℝ :=
  a * x^2 + b * x + c

def t (a b c : ℝ) : ℝ :=
  (a + 2 * b + 12 * c) / a

def meets_condition (a b c : ℝ) : Prop :=
  ∀ x : ℝ, quadratic_function a b c x ≥ 0

def axis_of_symmetry_condition (a b c : ℝ) : Prop :=
  -b / (2 * a) > c - 1 / 12

def minimum_t_value : ℝ := 2 / 3

def satisfies_CE_condition (a b c : ℝ) : Prop :=
  let CE := (sqrt ((1 / 6) - 0)^2 + (0 - c)^2) in
  CE > 5 / 24

theorem quadratic_form_solution
  (a b c : ℝ)
  (h1 : a > 0)
  (h2 : meets_condition a b c)
  (h3 : axis_of_symmetry_condition a b c)
  (h4 : t a b c = minimum_t_value)
  (h5 : satisfies_CE_condition 6 (-2) (1 / 6)) :
  a = 6 ∧ b = -2 ∧ c = 1 / 6 :=
by sorry

end quadratic_form_solution_l352_352467


namespace alex_bakes_cherry_pies_l352_352699

noncomputable def total_pies : ℕ := 24
noncomputable def apple_ratio : ℕ := 1
noncomputable def blueberry_ratio : ℕ := 4
noncomputable def cherry_ratio : ℕ := 3
noncomputable def ratio_sum : ℕ := apple_ratio + blueberry_ratio + cherry_ratio

theorem alex_bakes_cherry_pies :
  let pies_per_part := total_pies / ratio_sum in
  let cherry_pies := cherry_ratio * pies_per_part in
  cherry_pies = 9 :=
by
  sorry

end alex_bakes_cherry_pies_l352_352699


namespace geom_seq_sum_l352_352541

theorem geom_seq_sum (a : ℕ → ℝ) (n : ℕ) (q : ℝ) (h1 : a 1 = 2) (h2 : a 1 * a 5 = 64) :
  (a 1 * (1 - q^n)) / (1 - q) = 2^(n+1) - 2 := 
sorry

end geom_seq_sum_l352_352541


namespace vector_operation_correct_l352_352373

noncomputable def vector (x y : ℝ) : ℝ × ℝ := (x, y)

def norm (v : ℝ × ℝ) : ℝ := 
  let (x, y) := v
  real.sqrt (x^2 + y^2)

def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  let (x₁, y₁) := v₁
  let (x₂, y₂) := v₂
  x₁ * x₂ + y₁ * y₂

def angle_sine (v₁ v₂ : ℝ × ℝ) : ℝ :=
  let n₁ := norm v₁
  let n₂ := norm v₂
  let dp := dot_product v₁ v₂
  real.sqrt (1 - (dp / (n₁ * n₂))^2)

def vector_operation (a b : ℝ × ℝ) : ℝ :=
  (norm a) * (norm b) * (angle_sine a b)

theorem vector_operation_correct (x₁ y₁ x₂ y₂ : ℝ) : 
  let a := vector x₁ y₁
  let b := vector x₂ y₂
  vector_operation a b = |x₁ * y₂ - x₂ * y₁| :=
by
  sorry

end vector_operation_correct_l352_352373


namespace select_athlete_l352_352688

-- Conditions from the problem
variables (x_A x_B x_C : ℝ) (S2_A S2_B S2_C : ℝ)
variables (H1 : x_A = 176) (H2 : x_B = 173) (H3 : x_C = 176)
variables (H4 : S2_A = 10.5) (H5 : S2_B = 10.5) (H6 : S2_C = 42.1)

-- Statement to prove
theorem select_athlete : x_A = 176 → x_B = 173 → x_C = 176 → S2_A = 10.5 → S2_B = 10.5 → S2_C = 42.1 → "A" := 
by
  sorry

end select_athlete_l352_352688


namespace problem_solution_l352_352872

noncomputable def is_real_im_eq_im (a : ℝ) : Prop :=
let z := (1 - a * complex.i) / (2 + complex.i) in
  z.re = z.im

theorem problem_solution : ∃ a : ℝ, is_real_im_eq_im a ∧ a = -3 :=
by
  sorry

end problem_solution_l352_352872


namespace find_f_neg_2_l352_352411

-- Let f be a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the function g based on f
def g (x : ℝ) : ℝ := f x + x

-- The premises
variables (h1 : g (- (2 : ℝ)) = g (2 : ℝ)) (h2 : f 2 = 1)

-- The main statement to prove
theorem find_f_neg_2 : f (-2 : ℝ) = 5 := 
by
  sorry

end find_f_neg_2_l352_352411


namespace exists_exponential_function_satisfying_property_l352_352864

noncomputable def satisfies_property (f : ℝ → ℝ) := ∀ x y : ℝ, f(x + y) = f(x) * f(y)

theorem exists_exponential_function_satisfying_property :
  ∃ f : ℝ → ℝ, (∃ a : ℝ, 0 < a ∧ f = λ x, a^x) ∧ satisfies_property f :=
begin
  sorry
end

end exists_exponential_function_satisfying_property_l352_352864


namespace simplify_expression_l352_352590

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end simplify_expression_l352_352590


namespace original_numbers_unique_l352_352000

theorem original_numbers_unique (x1 x2 x3 x4 x5 : ℕ)
  (h1 : x1 < x2) (h2 : x2 < x3) (h3 : x3 < x4) (h4 : x4 < x5)
  (h_pairwise_sums : set (x1 + x2, x1 + x3, x1 + x4, x1 + x5, x2 + x3, x2 + x4, x2 + x5, x3 + x4, x3 + x5, x4 + x5) = {4, 8, 10, 12, 14, 18, 19, 21, 25, 29}) :
  {x1, x2, x3, x4, x5} = {1, 3, 7, 11, 18} :=
by
  sorry

end original_numbers_unique_l352_352000


namespace expected_score_shooting_competition_l352_352520

theorem expected_score_shooting_competition (hit_rate : ℝ)
  (miss_both_score : ℝ) (hit_one_score : ℝ) (hit_both_score : ℝ)
  (prob_0 : ℝ) (prob_10 : ℝ) (prob_15 : ℝ) :
  hit_rate = 4 / 5 →
  miss_both_score = 0 →
  hit_one_score = 10 →
  hit_both_score = 15 →
  prob_0 = (1 - 4 / 5) * (1 - 4 / 5) →
  prob_10 = 2 * (4 / 5) * (1 - 4 / 5) →
  prob_15 = (4 / 5) * (4 / 5) →
  (0 * prob_0 + 10 * prob_10 + 15 * prob_15) = 12.8 :=
by
  intros h_hit_rate h_miss_both_score h_hit_one_score h_hit_both_score
         h_prob_0 h_prob_10 h_prob_15
  sorry

end expected_score_shooting_competition_l352_352520


namespace sampling_is_stratified_l352_352281

structure School :=
  (total_male : ℕ)
  (total_female : ℕ)
  (sample_male : ℕ)
  (sample_female : ℕ)

def sampling_method (s : School) : String :=
  if s.total_male ≠ 0 ∧ s.total_female ≠ 0 ∧ s.sample_male ≠ 0 ∧ s.sample_female ≠ 0 ∧
      s.total_male * s.sample_female = s.total_female * s.sample_male
  then "stratified sampling"
  else "unknown"

theorem sampling_is_stratified :
  ∀ s : School, s = {total_male := 400, total_female := 600, sample_male := 40, sample_female := 60} →
    sampling_method s = "stratified sampling" :=
by
  intro s h
  rw [h]
  dsimp [sampling_method]
  split_ifs
  · sorry
  · sorry

end sampling_is_stratified_l352_352281


namespace angle_NHC_eq_60_l352_352787
-- Import the necessary libraries from Mathlib

-- Define an equilateral triangle and required midpoint calculations
def midpoint (p1 p2 : ℝ × ℝ) : ℝ × ℝ := ((p1.1 + p2.1) / 2, (p1.2 + p2.2) / 2)
def rotate_point (p origin : ℝ × ℝ) (angle : ℝ) : ℝ × ℝ :=
  let theta := angle * Real.pi / 180
  let cos_theta := Real.cos theta
  let sin_theta := Real.sin theta
  (origin.1 + (p.1 - origin.1) * cos_theta - (p.2 - origin.2) * sin_theta,
   origin.2 + (p.1 - origin.1) * sin_theta + (p.2 - origin.2) * cos_theta)

-- Define square ABCD with coordinates A(0,0), B(1,0), C(1,1), and D(0,1)
def A : ℝ × ℝ := (0, 0)
def B : ℝ × ℝ := (1, 0)
def C : ℝ × ℝ := (1, 1)
def D : ℝ × ℝ := (0, 1)

-- Define point S by rotating point C around point B by 60 degrees
def S : ℝ × ℝ := rotate_point C B 60

-- Define midpoints N of AS and H of CD
def N : ℝ × ℝ := midpoint A S
def H : ℝ × ℝ := midpoint C D

-- Define the theorem to prove that angle NHC in the defined configuration equals 60 degrees
theorem angle_NHC_eq_60 : 
  let angle (p1 p2 p3 : ℝ × ℝ) : ℝ := 
    let v1 := (p2.1 - p1.1, p2.2 - p1.2)
    let v2 := (p3.1 - p2.1, p3.2 - p2.2)
    let dot_product := v1.1 * v2.1 + v1.2 * v2.2
    let mag_v1 := Real.sqrt (v1.1 ^ 2 + v1.2 ^ 2)
    let mag_v2 := Real.sqrt (v2.1 ^ 2 + v2.2 ^ 2)
    Real.acos (dot_product / (mag_v1 * mag_v2)) * 180 / Real.pi
  angle N H C = 60 := sorry

end angle_NHC_eq_60_l352_352787


namespace negation_of_proposition_l352_352188

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l352_352188


namespace range_of_x_l352_352450

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f' (x : ℝ) : ℝ := sorry

axiom f_deriv : ∀ (x : ℝ), -4 ≤ x ∧ x ≤ 4 → f' x > f x
axiom main_condition : ∀ (x : ℝ), -4 ≤ x ∧ x ≤ 4 → e^(x-1) * f (1 + x) - f (2 * x) < 0

theorem range_of_x : ∀ (x : ℝ), -4 ≤ x ∧ x ≤ 4 → (e^(x-1) * f (1 + x) - f (2 * x) < 0) → 1 < x ∧ x ≤ 2 :=
  sorry

end range_of_x_l352_352450


namespace sequence_bijective_l352_352992

def sequence (n : ℕ) : ℕ :=
  if n = 1 then 1
  else if n % 2 = 0 then
    if (n / 2) % 2 = 0 then 2 * sequence (n / 2)
    else 2 * sequence (n / 2) + 1
  else
    if (n / 2) % 2 = 0 then 2 * sequence (n / 2) + 1
    else 2 * sequence (n / 2)

theorem sequence_bijective : ∀ n : ℕ, n > 0 → ∃! m : ℕ, m > 0 ∧ sequence m = n :=
by {
  sorry
}

end sequence_bijective_l352_352992


namespace policeman_speed_64_kmph_l352_352287

/-- 
Given:
1. The thief is spotted by a policeman from a distance of 160 meters.
2. The thief starts running as the policeman starts the chase.
3. The speed of the thief is 8 km/hr.
4. The thief runs 640 meters before he is overtaken.
Prove that the speed of the policeman is 64 km/hr.
-/
theorem policeman_speed_64_kmph (start_distance : ℝ) (thief_speed : ℝ) (thief_distance : ℝ)
  (h_start_distance : start_distance = 0.16) (h_thief_speed : thief_speed = 8)
  (h_thief_distance : thief_distance = 0.64) : 
  let dt := thief_distance + start_distance in
  let time := dt / thief_speed in
  (dt / time) = 64 := 
by {
  -- Definitions and assumptions
  let dt := thief_distance + start_distance,
  have : dt = 0.8, by sorry,
  let time := dt / thief_speed,
  -- Calculation step
  have : time = 0.1, by sorry,
  -- Result step
  have h_vp : dt / time = 64, by sorry,
  exact h_vp,
}

end policeman_speed_64_kmph_l352_352287


namespace binom_12_10_eq_66_l352_352355

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l352_352355


namespace y_value_l352_352024

theorem y_value (x y : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : x + 1 / y = 8) (h4 : y + 1 / x = 7 / 12) (h5 : x + y = 7) : y = 49 / 103 :=
by
  sorry

end y_value_l352_352024


namespace sum_of_three_exists_l352_352555

theorem sum_of_three_exists (n : ℤ) (X : Finset ℤ) 
  (hX_card : X.card = n + 2) 
  (hX_abs : ∀ x ∈ X, abs x ≤ n) : 
  ∃ a b c : ℤ, a ≠ b ∧ a ≠ c ∧ b ≠ c ∧ a ∈ X ∧ b ∈ X ∧ c ∈ X ∧ c = a + b := 
by 
  sorry

end sum_of_three_exists_l352_352555


namespace binomial_12_10_eq_66_l352_352360

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l352_352360


namespace projection_matrix_ratio_l352_352181

theorem projection_matrix_ratio
  (x y : ℚ)
  (h1 : (4/29) * x - (10/29) * y = x)
  (h2 : -(10/29) * x + (25/29) * y = y) :
  y / x = -5/2 :=
by
  sorry

end projection_matrix_ratio_l352_352181


namespace dice_sum_prime_probability_l352_352498

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def roll_dice_prob_prime : ℚ :=
  let total_outcomes := 6^7
  let prime_sums := [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
  let P := 80425 -- Assume pre-computed sum counts based on primes
  (P : ℚ) / total_outcomes

theorem dice_sum_prime_probability :
  roll_dice_prob_prime = 26875 / 93312 :=
by
  sorry

end dice_sum_prime_probability_l352_352498


namespace find_xy_l352_352643

variable {x y : ℝ}

theorem find_xy (h₁ : x + y = 10) (h₂ : x^3 + y^3 = 370) : x * y = 21 :=
by
  sorry

end find_xy_l352_352643


namespace base_digit_difference_l352_352846

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l352_352846


namespace cousins_in_rooms_l352_352133

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l352_352133


namespace minimum_students_both_l352_352718

variable (U : Type) -- U is the type representing the set of all students
variable (S_P S_C : Set U) -- S_P is the set of students who like physics, S_C is the set of students who like chemistry
variable (total_students : Nat) -- total_students is the total number of students
variable [Fintype U] -- U should be a finite set

-- conditions
variable (h_physics : (Fintype.card S_P).toFloat / total_students * 100 = 68)
variable (h_chemistry : (Fintype.card S_C).toFloat / total_students * 100 = 72)

-- statement of the theorem to be proved
theorem minimum_students_both : (Fintype.card (S_P ∩ S_C)).toFloat / total_students * 100 ≥ 40 := by
  sorry

end minimum_students_both_l352_352718


namespace white_square_area_l352_352570

theorem white_square_area (edge_length : ℝ) (green_paint_area : ℝ) (total_faces : ℕ) 
  (condition1 : edge_length = 12) (condition2 : green_paint_area = 432) (condition3 : total_faces = 6) : 
  let total_surface_area := total_faces * (edge_length * edge_length) in
  let green_area_per_face := green_paint_area / total_faces in
  let face_area := edge_length * edge_length in
  let white_area_per_face := face_area - green_area_per_face in
  white_area_per_face = 72 :=
by
  sorry

end white_square_area_l352_352570


namespace g_prime_at_1_l352_352454

def f (x : ℝ) : ℝ := sorry  -- Let f be some unspecified real function

-- Condition from the problem: the tangent line at (1, f(1)) is x - 2y + 1 = 0
axiom tangent_line_eq : ∀ x y : ℝ, (x, y) = (1, f 1) → x - 2*y + 1 = 0

-- Define g(x) as given in the problem
def g (x : ℝ) : ℝ := x / f x

-- State the goal that g'(1) = 1/2 using derivative
theorem g_prime_at_1 : deriv g 1 = 1 / 2 := by
  sorry

end g_prime_at_1_l352_352454


namespace exists_vector_sum_ge_L_div_pi_l352_352996

-- Given conditions.
variables (n : ℕ)
variables (v : Fin n → ℝ × ℝ) -- representing n vectors in the plane
variables (L : ℝ)
variables (h: ∑ i, (v i).fst^2 + (v i).snd^2 = L^2)

-- Statement that needs to be proven
theorem exists_vector_sum_ge_L_div_pi 
    (h_sum: (∑ i, (v i).fst^2 + (v i).snd^2).sqrt = L) : ∃ I : Finset (Fin n), (Finset.sum I (λ i, (v i).fst^2 + (v i).snd^2).sqrt ≥ L / π) :=
by {
    sorry
}

end exists_vector_sum_ge_L_div_pi_l352_352996


namespace probability_cello_viola_same_tree_l352_352260

noncomputable section

def cellos : ℕ := 800
def violas : ℕ := 600
def cello_viola_pairs_same_tree : ℕ := 100

theorem probability_cello_viola_same_tree : 
  (cello_viola_pairs_same_tree: ℝ) / ((cellos * violas : ℕ) : ℝ) = 1 / 4800 := 
by
  sorry

end probability_cello_viola_same_tree_l352_352260


namespace equilateral_triangles_circle_l352_352745

-- Definitions and conditions
structure Triangle :=
  (A B C : ℝ)
  (side_length : ℝ)
  (equilateral : side_length = 12)

structure Circle :=
  (S : ℝ)

def PointOnArc (P1 P2 P : ℝ) : Prop :=
  -- Definition to describe P lies on the arc P1P2
  sorry

-- Theorem stating the proof problem
theorem equilateral_triangles_circle
  (S : Circle)
  (T1 T2 : Triangle)
  (H1 : T1.side_length = 12)
  (H2 : T2.side_length = 12)
  (HAonArc : PointOnArc T2.B T2.C T1.A)
  (HBonArc : PointOnArc T2.A T2.B T1.B) :
  (T1.A - T2.A) ^ 2 + (T1.B - T2.B) ^ 2 + (T1.C - T2.C) ^ 2 = 288 :=
sorry

end equilateral_triangles_circle_l352_352745


namespace total_people_museum_l352_352200

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end total_people_museum_l352_352200


namespace kevin_run_distance_l352_352908

variable (Speed_flat : ℝ) (Time_flat : ℝ) (Speed_uphill : ℝ) (Time_uphill : ℝ) (Speed_downhill : ℝ) (Time_downhill : ℝ)

def Distance_flat := Speed_flat * Time_flat
def Distance_uphill := Speed_uphill * Time_uphill
def Distance_downhill := Speed_downhill * Time_downhill
def Total_distance := Distance_flat + Distance_uphill + Distance_downhill

theorem kevin_run_distance 
  (h1 : Speed_flat = 10) (h2 : Time_flat = 0.5)
  (h3 : Speed_uphill = 20) (h4 : Time_uphill = 0.5)
  (h5 : Speed_downhill = 8) (h6 : Time_downhill = 0.25) 
  : Total_distance Speed_flat Time_flat Speed_uphill Time_uphill Speed_downhill Time_downhill = 17 := 
by
  simp [Total_distance, Distance_flat, Distance_uphill, Distance_downhill, h1, h2, h3, h4, h5, h6]
  sorry

end kevin_run_distance_l352_352908


namespace total_action_figures_l352_352548

theorem total_action_figures (initial_figures cost_per_figure total_cost needed_figures : ℕ)
  (h1 : initial_figures = 7)
  (h2 : cost_per_figure = 8)
  (h3 : total_cost = 72)
  (h4 : needed_figures = total_cost / cost_per_figure)
  : initial_figures + needed_figures = 16 :=
by
  sorry

end total_action_figures_l352_352548


namespace hyperbola_condition_l352_352182

theorem hyperbola_condition (k : ℝ) : 
  (∃ (f : ℝ → ℝ → Prop), ∀ x y : ℝ, f x y ↔ x^2 / (k + 1) + y^2 / (k - 5) = 1 ∧
                              (k + 1)*(k - 5) < 0) ↔ k ∈ set.Ioo (-1 : ℝ) 5 := 
sorry

end hyperbola_condition_l352_352182


namespace expected_heads_after_four_tosses_l352_352095

/--
If Johann has 100 fair coins, and he flips all the coins. 
Any coin that lands on tails is tossed again, up to three more times. 
The expected number of coins that are now heads is 94.
-/
theorem expected_heads_after_four_tosses : 
  let p : ℚ := 1 / 2 + 1 / 4 + 1 / 8 + 1 / 16 in
  (100 * p).round = 94 :=
by
  let prob_heads_first : ℚ := 1 / 2
  let prob_heads_second : ℚ := (1 / 2) * (1 / 2)
  let prob_heads_third : ℚ := (1 / 4) * (1 / 2)
  let prob_heads_fourth : ℚ := (1 / 8) * (1 / 2)
  let p : ℚ := prob_heads_first + prob_heads_second + prob_heads_third + prob_heads_fourth
  have p_eq : p = 15 / 16 := sorry
  have E_heads : ℚ := 100 * p
  have round_E_heads : Int := E_heads.round
  show round_E_heads = 94, from sorry

end expected_heads_after_four_tosses_l352_352095


namespace negation_of_forall_statement_l352_352189

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l352_352189


namespace fourth_root_of_polynomial_l352_352640

theorem fourth_root_of_polynomial 
  (a b : ℝ)
  (h1 : a * 1^4 + (a + 2 * b) * 1^3 + (b - 3 * a) * 1^2 + (2 * a - 6) * 1 + (7 - a) = 0)
  (h2 : a * (-1)^4 + (a + 2 * b) * (-1)^3 + (b - 3 * a) * (-1)^2 + (2 * a - 6) * (-1) + (7 - a) = 0)
  (h3 : a * 2^4 + (a + 2 * b) * 2^3 + (b - 3 * a) * 2^2 + (2 * a - 6) * 2 + (7 - a) = 0) :
  ∃ r, r = -2 ∧ a * r^4 + (a + 2 * b) * r^3 + (b - 3 * a) * r^2 + (2 * a - 6) * r + (7 - a) = 0 :=
begin
  sorry
end

end fourth_root_of_polynomial_l352_352640


namespace binomial_12_10_eq_66_l352_352361

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l352_352361


namespace inverse_function_point_l352_352871

noncomputable def f (a : ℝ) (x : ℝ) := a^(x + 1)

theorem inverse_function_point (a : ℝ) (h_pos : 0 < a) (h_annoylem : f a (-1) = 1) :
  ∃ g : ℝ → ℝ, (∀ y, f a (g y) = y ∧ g (f a y) = y) ∧ g 1 = -1 :=
by
  sorry

end inverse_function_point_l352_352871


namespace find_a_if_lines_perpendicular_l352_352804

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end find_a_if_lines_perpendicular_l352_352804


namespace projection_of_straight_line_on_plane_l352_352197

def line_projection_on_plane (L : Type) (P : Type) 
  (is_perpendicular : L → P → Prop)
  (intersects_at_angle : L → P → Prop)
  (projection : L → P → Type) : Prop :=
∀ (l : L) (p : P), (is_perpendicular l p → projection l p = point) ∨ 
                     (intersects_at_angle l p → 
                       (projection l p = line_segment ∨ projection l p = straight_line))

theorem projection_of_straight_line_on_plane (L P : Type) 
  (is_perpendicular : L → P → Prop)
  (intersects_at_angle : L → P → Prop)
  (projection : L → P → Type) :
  ∀ (l : L) (p : P), (is_perpendicular l p → projection l p = point) ∨ 
                     (intersects_at_angle l p → 
                       (projection l p = line_segment ∨ projection l p = straight_line)) :=
sorry

end projection_of_straight_line_on_plane_l352_352197


namespace ratio_AM_AB_l352_352902

theorem ratio_AM_AB {A B C M : Type} [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace M]
    (h_midpoint : midpoint AC M)
    (h_BC : BC = (2 / 3) * AC)
    (h_angle : ∠ BMC = 2 * ∠ ABM) :
    AM / AB = 3 / (2 * sqrt 5) :=
sorry

end ratio_AM_AB_l352_352902


namespace hayden_ironing_weeks_l352_352310

variable (total_daily_minutes : Nat := 5 + 3)
variable (days_per_week : Nat := 5)
variable (total_minutes : Nat := 160)

def calculate_weeks (total_daily_minutes : Nat) (days_per_week : Nat) (total_minutes : Nat) : Nat :=
  total_minutes / (total_daily_minutes * days_per_week)

theorem hayden_ironing_weeks :
  calculate_weeks (5 + 3) 5 160 = 4 := 
by
  sorry

end hayden_ironing_weeks_l352_352310


namespace white_rabbit_hop_distance_per_minute_l352_352295

-- Definitions for given conditions
def brown_hop_per_minute : ℕ := 12
def total_distance_in_5_minutes : ℕ := 135
def brown_distance_in_5_minutes : ℕ := 5 * brown_hop_per_minute

-- The statement we need to prove
theorem white_rabbit_hop_distance_per_minute (W : ℕ) (h1 : brown_hop_per_minute = 12) (h2 : total_distance_in_5_minutes = 135) :
  W = 15 :=
by
  sorry

end white_rabbit_hop_distance_per_minute_l352_352295


namespace jenny_spent_180_minutes_on_bus_l352_352090

noncomputable def jennyBusTime : ℕ :=
  let timeAwayFromHome := 9 * 60  -- in minutes
  let classTime := 5 * 45  -- 5 classes each lasting 45 minutes
  let lunchTime := 45  -- in minutes
  let extracurricularTime := 90  -- 1 hour and 30 minutes
  timeAwayFromHome - (classTime + lunchTime + extracurricularTime)

theorem jenny_spent_180_minutes_on_bus : jennyBusTime = 180 :=
  by
  -- We need to prove that the total time Jenny was away from home minus time spent in school activities is 180 minutes.
  sorry  -- Proof to be completed.

end jenny_spent_180_minutes_on_bus_l352_352090


namespace find_200th_application_of_f_l352_352819

-- Define the repeating sequence as a list of digits
def repeating_sequence : List ℕ := [9, 1, 8, 2, 7, 3, 6, 4, 5]

-- Define the function f(n) which returns the nth element in the repeating sequence
def f (n : ℕ) : ℕ := repeating_sequence[(n - 1) % repeating_sequence.length]

-- Statement to prove
theorem find_200th_application_of_f :
  (Nat.iterate f 200 1) = 8 := sorry

end find_200th_application_of_f_l352_352819


namespace binom_12_10_l352_352344

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l352_352344


namespace complex_product_in_polar_form_eq_l352_352313

theorem complex_product_in_polar_form_eq :
  let cis (θ : ℝ) := Complex.mk (Real.cos θ) (Real.sin θ) in
  let z1 := 5 * cis (42 * Real.pi / 180) in
  let z2 := 4 * cis (85 * Real.pi / 180) in
  let z3 := 20 * cis (127 * Real.pi / 180) in
  (z1 * z2 = z3) ∧ (20 > 0) ∧ (0 ≤ 127) ∧ (127 < 360) :=
  by
    sorry

end complex_product_in_polar_form_eq_l352_352313


namespace lines_perpendicular_to_plane_are_parallel_l352_352563

variable {α : Type*} [MetricSpace α]

structure Line (P : Type*) :=
  (dir : P → P)
  (pt : P)
  (is_line : ∀ x y : P, x ≠ y → 
    ∃ (t : P), x + (t * dir pt) = y)

structure Plane (P : Type*) :=
  (norm : P → P)
  (pt : P)
  (is_plane : ∀ x y z : P, (x ≠ y ∧ y ≠ z ∧ z ≠ x) → 
    ∃ (λ1 λ2 : P), z = x + (λ1 * norm pt) + (λ2 * norm pt))

def perpendicular_to_plane {P : Type*} [MetricSpace P] 
  (m : Line P) (α : Plane P) : Prop :=
  ∀ pt : P, ∃ (k : P), m.dir pt = k * α.norm pt

def parallel_to_plane {P : Type*} [MetricSpace P] 
  (m : Line P) (α : Plane P) : Prop :=
  ∃ pt : P, ¬ ∃ (k : P), m.dir pt = k * α.norm pt

theorem lines_perpendicular_to_plane_are_parallel
  {P : Type*} [MetricSpace P]
  (m n : Line P) (α : Plane P)
  (h₁ : perpendicular_to_plane m α)
  (h₂ : perpendicular_to_plane n α)
  (h_diff : m ≠ n) :
  parallel m.pt n.pt :=
sorry

end lines_perpendicular_to_plane_are_parallel_l352_352563


namespace train_speed_l352_352290

open Real

theorem train_speed (train_length bridge_length : ℝ) (cross_time : ℝ) :
  train_length = 125 ∧ bridge_length = 250.03 ∧ cross_time = 30 →
  (train_length + bridge_length) / cross_time * 3.6 = 45.0036 :=
by
  intros h
  cases h with ht rest
  cases rest with hb hc
  rw [ht, hb, hc]
  sorry

end train_speed_l352_352290


namespace base_digit_difference_l352_352843

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l352_352843


namespace perpendicular_lines_m_value_l352_352503

-- Define the first line
def line1 (x y : ℝ) : Prop := 3 * x - y + 1 = 0

-- Define the second line
def line2 (x y : ℝ) (m : ℝ) : Prop := 6 * x - m * y - 3 = 0

-- Define the perpendicular condition for slopes of two lines
def perpendicular_slopes (m1 m2 : ℝ) : Prop := m1 * m2 = -1

-- Prove the value of m for perpendicular lines
theorem perpendicular_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, line1 x y → ∃ y', line2 x y' m) →
  (∀ x y : ℝ, ∃ x', line1 x y ∧ line2 x' y m) →
  perpendicular_slopes 3 (6 / m) →
  m = -18 :=
by
  sorry

end perpendicular_lines_m_value_l352_352503


namespace problem1_problem2_l352_352961

def f (x : ℝ) : ℝ := abs (2 * x + 1) - abs (x - 2)

theorem problem1 (x : ℝ) : f(x) > 2 ↔ (x > 1 ∨ x < -5) :=
by
  -- proof skipped
  sorry

theorem problem2 (t : ℝ) : (∀ x : ℝ, f(x) ≥ t^2 - 11/2 * t) → (1/2 ≤ t ∧ t ≤ 5) :=
by
  -- proof skipped
  sorry

end problem1_problem2_l352_352961


namespace best_model_fits_l352_352527

noncomputable def model_fits (R2_1 R2_2 R2_3 R2_4 : ℝ) : Prop :=
  R2_1 = 0.99 ∧ R2_2 = 0.89 ∧ R2_3 = 0.52 ∧ R2_4 = 0.16

theorem best_model_fits (R2_1 R2_2 R2_3 R2_4 : ℝ) 
  (h : model_fits R2_1 R2_2 R2_3 R2_4) : R2_1 > R2_2 ∧ R2_1 > R2_3 ∧ R2_1 > R2_4 :=
by 
  -- use h to obtain the conditions
  cases h with h1 h_rest,
  cases h_rest with h2 h_rest,
  cases h_rest with h3 h4,
  -- split the goal into individual comparisons
  split;
  -- show each comparison individually
  linarith;
  linarith;
  linarith;
  sorry

end best_model_fits_l352_352527


namespace cylinder_rectangle_diagonal_l352_352622

def diagonal_of_rectangle_formed_by_cylinder : ℝ :=
  sqrt ((12 : ℝ) ^ 2 + (16 : ℝ) ^ 2)

theorem cylinder_rectangle_diagonal :
  let h := 16
  let P := 12
  diagonal_of_rectangle_formed_by_cylinder = 20 :=
by
  sorry

end cylinder_rectangle_diagonal_l352_352622


namespace simplify_expression_when_x_is_3_l352_352362

theorem simplify_expression_when_x_is_3 : 
  (let x := 3 in
   (x^8 + 16 * x^4 + 64 + 2 * x^2) / (x^4 + 8 + x^2) = 98) :=
by
  sorry

end simplify_expression_when_x_is_3_l352_352362


namespace base4_more_digits_than_base9_l352_352856

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l352_352856


namespace max_area_quadrilateral_OAPF_l352_352083

noncomputable def ellipse := { p : ℝ × ℝ // (p.1)^2 / 9 + (p.2)^2 / 10 = 1 }

def A : ℝ × ℝ := (3, 0)

def F : ℝ × ℝ := (0, 1)

def P_in_first_quadrant (P : ellipse) : Prop := 
  0 < P.val.1 ∧ 0 < P.val.2

theorem max_area_quadrilateral_OAPF :
  ∀ P : ellipse, P_in_first_quadrant P →
  ∃ M : ℝ, M = (3 * real.sqrt 11) / 2 :=
sorry

end max_area_quadrilateral_OAPF_l352_352083


namespace sum_of_distinct_complex_numbers_eq_neg_one_l352_352048

theorem sum_of_distinct_complex_numbers_eq_neg_one
    (a b : ℂ) : 
    a ≠ b ∧ a * b ≠ 0 ∧ ({a, b} = {a^2, b^2}) → a + b = -1 :=
by
    intros h
    sorry

end sum_of_distinct_complex_numbers_eq_neg_one_l352_352048


namespace order_of_abc_l352_352863

noncomputable def a : ℤ := (-99 : ℤ) ^ 0
noncomputable def b : ℚ := (1 / 2 : ℚ) ^ (-1 : ℤ)
noncomputable def c : ℤ := (-2 : ℤ) ^ 2

theorem order_of_abc : c > b ∧ b > a := by
  -- evaluate a: (-99)^0 = 1
  have ha : a = 1 := pow_zero (-99)
  -- evaluate b: (1/2)^(-1) = 2
  have hb : b = 2 := one_div_pow (-1) 1 (by norm_num : (1/2 : ℚ) ≠ 0)
  -- evaluate c: (-2)^2 = 4
  have hc : c = 4 := pow_two (-2)

  -- now we prove c > b > a
  rw [ha, hb, hc]
  norm_num

end order_of_abc_l352_352863


namespace cousins_rooms_distribution_l352_352114

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l352_352114


namespace nat_solution_eq_l352_352749

theorem nat_solution_eq (a b: ℕ) :
  (⌊a ^ 2 / b⌋ + ⌊b ^ 2 / a⌋ = ⌊(a ^ 2 + b ^ 2) / (a * b)⌋ + a * b) ↔
  (b = a ^ 2 + 1 ∨ a = b ^ 2 + 1) := by
  sorry

end nat_solution_eq_l352_352749


namespace mod_calculation_l352_352722

theorem mod_calculation :
  (3 * 43 + 6 * 37) % 60 = 51 :=
by
  sorry

end mod_calculation_l352_352722


namespace expected_distance_closest_pair_five_points_l352_352410

theorem expected_distance_closest_pair_five_points : 
  let points := [0, 0.2, 0.4, 0.6, 0.8, 1] in 
  ∃ D : ℚ, 
  D = 1 / 24 :=
sorry

end expected_distance_closest_pair_five_points_l352_352410


namespace remainder_of_trailing_zeros_mod_100_l352_352919

noncomputable def factorial (n : ℕ) : ℕ :=
if n = 0 then 1 else n * factorial (n - 1)

def product_factorials_up_to (n : ℕ) : ℕ :=
(1 to n).product factorial

def number_of_trailing_zeros (n : ℕ) : ℕ :=
let fives := (1 to n).count (λ k => k % 5 = 0)
fives + (1 to n).count (λ k => k > 0 && k % 25 = 0)
-- Additional factors if we consider higher powers of 5
+ (1 to n).count (λ k => k > 0 && k % 125 = 0) -- for completeness

theorem remainder_of_trailing_zeros_mod_100 : (number_of_trailing_zeros 20) % 100 = 8 :=
by
  sorry

end remainder_of_trailing_zeros_mod_100_l352_352919


namespace Jiaqi_score_l352_352549

def reciprocal (x : ℝ) : ℝ := 1 / x

theorem Jiaqi_score :
  (reciprocal (Real.sqrt 2) ≠ -Real.sqrt 2) ∧
  (abs (-Real.sqrt 3) = Real.sqrt 3) ∧
  (Real.sqrt 4 = 2) ∧
  (∀ x, (x = 0 ∧ Real.sqrt x = x ∧ Real.cbrt x = x) ∨ (x = 1 ∧ Real.sqrt x = x ∧ Real.cbrt x = x)) ∧
  (Real.cbrt ((-2)^3) = -2) →
  ∑ i in {false, true, false, false, true}.to_finset, if i then 20 else 0 = 40 :=
by 
  sorry

end Jiaqi_score_l352_352549


namespace range_of_x_l352_352006

theorem range_of_x (x : ℝ) (h : ∃ θ : ℝ, θ > π / 2 ∧ θ < π ∧
  let AB := (x, 2 * x) in
  let AC := (3 * x, 2) in
  (AB.1 * AC.1 + AB.2 * AC.2 < 0) ) : -4/3 < x ∧ x < 0 :=
sorry

end range_of_x_l352_352006


namespace b_range_l352_352461

noncomputable def f (x a b : ℝ) : ℝ := x + (a / x) + b

theorem b_range (b : ℝ) :
  (∀ a ∈ set.Icc (1/2 : ℝ) 2, ∀ x ∈ set.Icc (1/4 : ℝ) 1, f x a b ≤ 10) → b ≤ 7 / 4 := by
  sorry

end b_range_l352_352461


namespace min_chameleons_to_turn_blue_l352_352634

-- Definitions related to colors and chameleons
inductive Color
| red
| blue
| green
| yellow
| other

-- Hypothetical color-changing rules based on interactions
def bite (biter : Color) (bitten : Color) : Color :=
match biter, bitten with
| Color.red, Color.green  => Color.blue
| Color.green, Color.red  => Color.red
| Color.red, Color.red    => Color.yellow
| _, _                    => bitten
end

-- Assumptions as given in the problem
noncomputable def sequence_of_bites_transforms_all_to_blue: ∀ (n : ℕ), (2023 ≤ n) → ∃ (steps : list (Color × Color)), (∀ i, i < n → (steps.nth i).isSome) ∧ (steps.nth_le (n-1) (by simp) = (Color.red, Color.blue)) :=
sorry

-- Minimum number of red chameleons to all turn blue
theorem min_chameleons_to_turn_blue : ∃ k, (∀ (n : ℕ), n ≤ k → (∀ remaining_bites : list (Color × Color), ∃ steps : list (Color × Color), steps.length = n ∧ last steps = (Color.red, Color.blue))) ∧ k = 5 :=
begin
  use 5,
  split,
  { intros n hn remaining_bites,
    sorry },
  { refl },
end

end min_chameleons_to_turn_blue_l352_352634


namespace triplet_not_equal_to_one_l352_352655

def A := (1/2, 1/3, 1/6)
def B := (2, -2, 1)
def C := (0.1, 0.3, 0.6)
def D := (1.1, -2.1, 1.0)
def E := (-3/2, -5/2, 5)

theorem triplet_not_equal_to_one (ha : A = (1/2, 1/3, 1/6))
                                (hb : B = (2, -2, 1))
                                (hc : C = (0.1, 0.3, 0.6))
                                (hd : D = (1.1, -2.1, 1.0))
                                (he : E = (-3/2, -5/2, 5)) :
  (1/2 + 1/3 + 1/6 = 1) ∧
  (2 + -2 + 1 = 1) ∧
  (0.1 + 0.3 + 0.6 = 1) ∧
  (1.1 + -2.1 + 1.0 ≠ 1) ∧
  (-3/2 + -5/2 + 5 = 1) :=
by {
  sorry
}

end triplet_not_equal_to_one_l352_352655


namespace missing_number_l352_352212

open Nat

theorem missing_number (a b c d e : ℕ) (sum : ℕ) :
  (a = 2) → (b = 4) → (c = 9) → (d = 17) → (e = 19) →
  ∃ f : ℕ, 
    (a + e = sum) ∧ (b + d = sum) ∧ (c + f = sum) → f = 12 := 
by
  intros ha hb hc hd he
  use 12
  rw [ha, hb, hc, hd, he]
  split; try { split };
  sorry

end missing_number_l352_352212


namespace sequence_solution_l352_352828

theorem sequence_solution (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2) (h_rec : ∀ n > 0, a (n + 1) = a n ^ 2) : 
  a n = 2 ^ 2 ^ (n - 1) :=
by
  sorry

end sequence_solution_l352_352828


namespace cannot_determine_right_triangle_l352_352702

theorem cannot_determine_right_triangle (A B C : ℝ) :
  (A - B = 90 ∨ A = B = 2 * C) → ¬ (A + B + C = 180 ∧ (A = 90 ∨ B = 90 ∨ C = 90)) :=
by
  intros h1 h2
  cases h1
  { sorry }
  { sorry }

end cannot_determine_right_triangle_l352_352702


namespace base4_more_digits_than_base9_l352_352861

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l352_352861


namespace log_properties_l352_352723

theorem log_properties:
  log(10) = 1 → (log(5) ^ 2 + log(2) * log(50) = 1) :=
begin
  sorry
end

end log_properties_l352_352723


namespace pyramid_cube_volume_l352_352275

-- Lean 4 statement for the proof problem
theorem pyramid_cube_volume : 
  let side_length_base := 2,
      height_base := sqrt 3,
      height_pyramid := (2 * sqrt 3) / 3,
      side_length_cube := (2 * sqrt 3) / 9 in
  (side_length_cube ^ 3) = 8 * sqrt 3 / 243 :=
by
  let side_length_base := 2
  let height_base := sqrt 3
  let height_pyramid := (2 * sqrt 3) / 3
  let side_length_cube := (2 * sqrt 3) / 9
  have cube_volume := side_length_cube ^ 3
  have volume_correct := (8 * sqrt 3) / 243
  exact sorry

end pyramid_cube_volume_l352_352275


namespace nonagon_side_equal_difference_diag_l352_352944

-- Define the regular nonagon and geometrical properties
variables {P : ℝ} -- Type for points considered in ℝ
variables (A B C D E F G H I : P) -- Vertices of the regular nonagon
variables (s : ℝ) -- Side length of the regular nonagon
variables (BD AE : ℝ) -- Length of the shortest (BD) and longest (AE) diagonals

-- Angle and triangle properties in the regular nonagon
variable (angle_BAE : ∠ B A E = 60) -- Inscribed angle at BAE is 60 degrees
variable (triangle_AKB_equilateral : ∀ K, →  ∠ A K B = 60 ∧ ∠ A B K = 60 ∧ ∠ K A B = 60) -- Triangle AKB is equilateral

-- Theorem statement
theorem nonagon_side_equal_difference_diag :
  s = AE - BD := sorry

end nonagon_side_equal_difference_diag_l352_352944


namespace study_days_needed_l352_352710

theorem study_days_needed
    (chapters : ℕ) (worksheets : ℕ)
    (hours_per_chapter : ℕ) (hours_per_worksheet : ℕ)
    (max_hours_per_day : ℕ)
    (break_minutes_per_hour : ℕ) (snack_breaks_per_day : ℕ)
    (snack_break_minutes : ℕ) (lunch_break_minutes : ℕ) :
    chapters = 2 →
    worksheets = 4 →
    hours_per_chapter = 3 →
    hours_per_worksheet = 1.5 →
    max_hours_per_day = 4 →
    break_minutes_per_hour = 10 →
    snack_breaks_per_day = 3 →
    snack_break_minutes = 10 →
    lunch_break_minutes = 30 →
    (15 / 4).ceil = 4 := by 
  sorry

end study_days_needed_l352_352710


namespace number_of_ways_to_put_cousins_in_rooms_l352_352115

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l352_352115


namespace circumcenter_on_A_l352_352247

section
variables {A B C A' B' : Type*}
variables [metric_space A] [metric_space B] [metric_space C] [metric_space A'] [metric_space B']
variable [linear_ordered_field ℝ]
variables (ABC : triangle A B C) (A'B'C : triangle A' B' C)
variables (right_triangle_ABC : ABC.is_right_triangle B)
variables (right_triangle_A'B'C : A'B'C.is_right_triangle B')
variables (similar_triangles : ∃ f: AffineEquiv ℝ ℝ, f ABC = A'B'C)
variables (A'_on_BC : ∃ t: ℝ, t > 1 ∧ A' = C + t • (C - B))

theorem circumcenter_on_A'B' : 
  let circumcenter := circumcenter (triangle A' A C) in
  on_line circumcenter A' B' := sorry
end

end circumcenter_on_A_l352_352247


namespace two_statements_true_l352_352832

theorem two_statements_true (a : ℝ) :
  (¬∃ x : ℝ, x + 1 / x = a) →
  (sqrt (a ^ 2 - 4 * a + 4) = 2 - a) →
  (∃! (x y : ℝ), x + y^2 = a ∧ x - sin(y) ^ 2 = -3) →
  (a = -3 ∨ (a > -2 ∧ a < 2)) :=
by
  intros h1 h2 h3
  sorry

end two_statements_true_l352_352832


namespace table_tennis_max_players_l352_352585

theorem table_tennis_max_players 
  (n : ℕ) 
  (h : ∀ (s : finset ℕ), s.card = 4 → ∃ (a b : ℕ), a ≠ b ∧ ∃ (x y : ℕ), (x ∈ s) ∧ (y ∈ s) ∧ x ≠ y ∧ a = score x ∧ b = score y) 
  (h_n : n ≥ 4) : 
  n ≤ 7 := 
sorry

end table_tennis_max_players_l352_352585


namespace ratio_of_areas_l352_352642

-- Define the triangle ABC and conditions

noncomputable def triangle (A B C : Type) : Type := sorry

axiom lines_parallel (DE FG HI JK BC : Type) : Type
axiom height_divided_equal (AD DJ JK KC : Type) : Type 
axiom line_equal_parts : AD = DJ ∧ DJ = JK ∧ JK = KC

-- Define the ratio of areas
def area_trapezoid_JKBC (A B C D E F G H I J K : Type) : ℝ := sorry
def area_triangle_ABC (A B C : Type) : ℝ := sorry

-- State the proposition
theorem ratio_of_areas
  (A B C D E F G H I J K : Type)
  [triangle A B C] [lines_parallel DE FG HI JK BC]
  [height_divided_equal AD DJ JK KC] [line_equal_parts]:
  (area_trapezoid_JKBC A B C D E F G H I J K) / (area_triangle_ABC A B C) = (1 : ℝ) / (5 : ℝ) := sorry

end ratio_of_areas_l352_352642


namespace cos_alpha_value_l352_352809

-- Define the point P
def P : ℝ × ℝ := (-5, 12)

-- Calculate the distance from the origin to P
def distance_from_origin (P : ℝ × ℝ) : ℝ :=
  Real.sqrt (P.1 ^ 2 + P.2 ^ 2)

-- Define the angle α and its cosine
noncomputable def cos_alpha (P : ℝ × ℝ) : ℝ :=
  P.1 / distance_from_origin P

-- Math proof statement
theorem cos_alpha_value : cos_alpha P = - 5 / 13 := 
by
  -- proof would go here
  sorry

end cos_alpha_value_l352_352809


namespace no_alpha_exists_l352_352380

theorem no_alpha_exists (α : ℝ) (hα1 : 0 < α) (hα2 : α < 1) :
  ¬(∃ (a : ℕ → ℝ), (∀ n : ℕ, 0 < a n) ∧ (∀ n : ℕ, 1 + a (n+1) ≤ a n + (α / n.succ) * a n)) :=
by
  sorry

end no_alpha_exists_l352_352380


namespace initial_students_count_l352_352167

-- Definitions based on conditions
def initial_average_age (T : ℕ) (n : ℕ) : Prop := T = 14 * n
def new_average_age_after_adding (T : ℕ) (n : ℕ) : Prop := (T + 5 * 17) / (n + 5) = 15

-- Main proposition stating the problem
theorem initial_students_count (n : ℕ) (T : ℕ) 
  (h1 : initial_average_age T n)
  (h2 : new_average_age_after_adding T n) :
  n = 10 :=
by
  sorry

end initial_students_count_l352_352167


namespace no_adjacent_same_roll_probability_l352_352766

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l352_352766


namespace find_multiplier_l352_352226

theorem find_multiplier (x : ℝ) : (4.5 / 6) * x = 9 → x = 12 := by
  intro h
  have h0 : 4.5 / 6 = 0.75 := by norm_num
  rw [h0] at h
  linarith

end find_multiplier_l352_352226


namespace binom_12_10_eq_66_l352_352340

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l352_352340


namespace solve_equations_l352_352593

-- Prove that the solutions to the given equations are correct.
theorem solve_equations :
  (∀ x : ℝ, (x * (x - 4) = 2 * x - 8) ↔ (x = 4 ∨ x = 2)) ∧
  (∀ x : ℝ, ((2 * x) / (2 * x - 3) - (4 / (2 * x + 3)) = 1) ↔ (x = 10.5)) :=
by
  sorry

end solve_equations_l352_352593


namespace cost_of_projector_and_whiteboard_l352_352308

variable (x : ℝ)

def cost_of_projector : ℝ := x
def cost_of_whiteboard : ℝ := x + 4000
def total_cost_eq_44000 : Prop := 4 * (x + 4000) + 3 * x = 44000

theorem cost_of_projector_and_whiteboard 
  (h : total_cost_eq_44000 x) : 
  cost_of_projector x = 4000 ∧ cost_of_whiteboard x = 8000 :=
by
  sorry

end cost_of_projector_and_whiteboard_l352_352308


namespace playful_not_brown_l352_352951

variables (Dog : Type) (Playful Brown CanSwim KnowsTricks : Dog → Prop)

theorem playful_not_brown
  (h1 : ∀ d : Dog, Playful d → KnowsTricks d)
  (h2 : ∀ d : Dog, Brown d → ¬ CanSwim d)
  (h3 : ∀ d : Dog, ¬ CanSwim d → ¬ KnowsTricks d) :
  ∀ d : Dog, Playful d → ¬ Brown d :=
by
  intro d hd
  have h_notSwim : ¬ CanSwim d := by sorry   -- Follows from the given conditions
  have h_notKnows : ¬ KnowsTricks d := by sorry  -- Follows from the given conditions
  have h_notBrown : ¬ Brown d := by sorry  -- Follows from the combined information
  exact h_notBrown d hd

end playful_not_brown_l352_352951


namespace base_digit_difference_l352_352845

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l352_352845


namespace inequality_proof_l352_352013

-- Defining the conditions
variable (a b : ℕ) (a_pos : 0 < a) (b_pos : 0 < b) (cond : 1 / a + 1 / b = 1)

-- Defining the theorem to be proved
theorem inequality_proof (n : ℕ) : 
  (a + b)^n - a^n - b^n ≥ 2^(2*n) - 2^(n+1) :=
by
  sorry

end inequality_proof_l352_352013


namespace intersection_of_M_and_N_l352_352470

def M : set ℕ := {0, 1, 2}
def N : set ℕ := { x | ∃ (a : ℕ), a ∈ M ∧ x = a^2 }

theorem intersection_of_M_and_N : M ∩ N = {0, 1} :=
by {
  -- Sorry is a placeholder to skip the actual proof.
  sorry
}

end intersection_of_M_and_N_l352_352470


namespace find_fx_plus_1_l352_352021

theorem find_fx_plus_1 (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x - 1) = x^2 + 4 * x - 5) : 
  ∀ x : ℤ, f (x + 1) = x^2 + 8 * x + 7 :=
sorry

end find_fx_plus_1_l352_352021


namespace strictly_convex_l352_352101

noncomputable def f (A : Point) (M : Point) (Mp : Point) : ℝ :=
  |A - M| / |M - Mp|

theorem strictly_convex
  (C : Circle)
  (A : Point)
  (M M' : Point)
  (D : interior C)
  (hA : A ∈ C)
  (h1 : ∀ (M : D), M' = (line A M ∩ C).some) :
  ∀ (M1 M2 : D), M1 ≠ M2 → 
  let P := midpoint M1 M2 in
  f A P (h1 P) < (f A M1 (h1 M1) + f A M2 (h1 M2)) / 2 :=
sorry

end strictly_convex_l352_352101


namespace population_exceeds_capacity_in_2082_l352_352529

-- Define the initial population
def initial_population := 500

-- Define the growth period
def growth_period := 20

-- Define the quadrupling factor
def growth_factor := 4

-- Define the starting year
def start_year := 2022

-- Define the maximum capacity of the region
def max_capacity := 31000 / 1.25

-- Define the population function over time
def population (t : ℕ) : ℕ :=
  initial_population * (growth_factor ^ (t / growth_period))

-- Define the target year
def target_year := 2082

-- Define the proof statement
theorem population_exceeds_capacity_in_2082 : population (target_year - start_year) > max_capacity :=
  by
    sorry

end population_exceeds_capacity_in_2082_l352_352529


namespace henry_distance_from_starting_point_l352_352052

noncomputable def north_distance_meters : ℝ := 10
noncomputable def east_distance_feet : ℝ := 30
noncomputable def conversion_factor : ℝ := 3.28084
noncomputable def initial_south_distance_feet : ℝ := 10 * conversion_factor
noncomputable def additional_south_distance_feet : ℝ := 40
noncomputable def south_distance_total_feet : ℝ := initial_south_distance_feet + additional_south_distance_feet

theorem henry_distance_from_starting_point : 
  let net_south_distance_feet := south_distance_total_feet - initial_south_distance_feet in
  let net_east_distance_feet := east_distance_feet in
  let distance_squared := net_south_distance_feet ^ 2 + net_east_distance_feet ^ 2 in
  sqrt distance_squared = 50 :=
by 
  sorry

end henry_distance_from_starting_point_l352_352052


namespace x_coordinate_of_first_point_l352_352542

theorem x_coordinate_of_first_point (m n : ℝ) :
  (m = 2 * n + 3) ↔ (∃ (p1 p2 : ℝ × ℝ), p1 = (m, n) ∧ p2 = (m + 2, n + 1) ∧ 
    (p1.1 = 2 * p1.2 + 3) ∧ (p2.1 = 2 * p2.2 + 3)) :=
by
  sorry

end x_coordinate_of_first_point_l352_352542


namespace least_sum_exponents_of_520_l352_352491

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l352_352491


namespace triangle_rotation_sum_l352_352216

theorem triangle_rotation_sum :
  ∃ (u v n : ℝ), (0 < n ∧ n < 180) ∧
    (D' = ⟨30, 20⟩ ∧ (0 - u = -v + 20) ∧ (0 - v = u - 30)) ∧
    (u - v = 30) ∧ (u + v = 20) ∧
    (n + u + v = 110) :=
  sorry

end triangle_rotation_sum_l352_352216


namespace min_links_to_remove_l352_352574

-- Define the grid dimensions and the grid lines
def grid := (10, 10)
def horizontal_lines := 11
def vertical_lines := 11

-- The number of links should be minimum such that each node has at most 3 remaining links
theorem min_links_to_remove (grid : ℕ × ℕ) (horizontal_lines vertical_lines : ℕ) : grid = (10, 10) ∧ horizontal_lines = 11 ∧ vertical_lines = 11 →
  ∃ min_links : ℕ, min_links = 41 ∧
  (∀ (node : ℕ × ℕ), node_fulfils_condition node grid horizontal_lines vertical_lines) :=
sorry

-- Definition of the node condition
def node_fulfils_condition (node : ℕ × ℕ) (grid : ℕ × ℕ) (horizontal_lines vertical_lines : ℕ) : Prop :=
  (node.1 > 0 ∧ node.1 < grid.1 + 1 ∧ node.2 > 0 ∧ node.2 < grid.2 + 1 →
  horizontal_links node + vertical_links node ≤ 3) ∧
  (node.1 = 0 ∨ node.1 = grid.1 + 1 ∨ node.2 = 0 ∨ node.2 = grid.2 + 1 →
  horizontal_links node + vertical_links node ≤ 2)

-- Definition of links at a node
def horizontal_links (node : ℕ × ℕ) : ℕ := if node.1 = 0 ∨ node.1 > 10 then 0 else 1
def vertical_links (node : ℕ × ℕ) : ℕ := if node.2 = 0 ∨ node.2 > 10 then 0 else 1

end min_links_to_remove_l352_352574


namespace exists_six_numbers_multiple_2002_l352_352011

theorem exists_six_numbers_multiple_2002 (a : Fin 41 → ℕ) (h : Function.Injective a) :
  ∃ (i j k l m n : Fin 41),
    i ≠ j ∧ k ≠ l ∧ m ≠ n ∧
    (a i - a j) * (a k - a l) * (a m - a n) % 2002 = 0 := sorry

end exists_six_numbers_multiple_2002_l352_352011


namespace natural_eq_rational_exists_diff_l352_352235

-- Part (a)
theorem natural_eq (x y : ℕ) (h : x^3 + y = y^3 + x) : x = y := 
by sorry

-- Part (b)
theorem rational_exists_diff (x y : ℚ) (h : x > 0 ∧ y > 0 ∧ x ≠ y ∧ x^3 + y = y^3 + x) : ∃ (x y : ℚ), x ≠ y ∧ x^3 + y = y^3 + x := 
by sorry

end natural_eq_rational_exists_diff_l352_352235


namespace divisor_of_poly_l352_352418

theorem divisor_of_poly (c : ℤ) : c = 2 →
    (∀ q : ℤ[X], (X^2 + X + c) * q = X^13 - X + 106 → ∃ q : ℤ[X], (X^2 + X + c) * q = X^13 - X + 106) := 
by
  sorry

end divisor_of_poly_l352_352418


namespace kite_diagonal_ratio_l352_352102

theorem kite_diagonal_ratio (a b x y : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a ≥ b)
  (hx1 : 0 ≤ x) (hx2 : x < a) (hy1 : 0 ≤ y) (hy2 : y < b)
  (orthogonal_diagonals : a^2 + y^2 = b^2 + x^2) :
  (a / b)^2 = 4 / 3 := 
sorry

end kite_diagonal_ratio_l352_352102


namespace limit_fraction_l352_352030

open Real

theorem limit_fraction (h₁ : tendsto (λ n : ℕ, (n : ℝ) * sin (1 / (n : ℝ))) at_top (𝓝 1)) :
  tendsto (λ n : ℕ, (5 - (n : ℝ)^2 * sin (1 / (n : ℝ))) / (2 * (n : ℝ) - 1)) at_top (𝓝 (1 / 2)) :=
sorry

end limit_fraction_l352_352030


namespace base4_more_digits_than_base9_l352_352853

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l352_352853


namespace boundary_shadow_function_l352_352624

theorem boundary_shadow_function 
    (r : ℝ) (O P : ℝ × ℝ × ℝ) (f : ℝ → ℝ)
    (h_radius : r = 1)
    (h_center : O = (1, 0, 1))
    (h_light_source : P = (1, -1, 2)) :
  (∀ x, f x = (x - 1) ^ 2 / 4 - 1) := 
by 
  sorry

end boundary_shadow_function_l352_352624


namespace circles_are_tangent_and_find_tangent_line_l352_352792

noncomputable def circle_center (a b c : ℝ) : (ℝ × ℝ) := (a, b)
noncomputable def circle_radius (a b : ℝ) : ℝ := c

theorem circles_are_tangent_and_find_tangent_line :
  let C := (circle_center 2 0, circle_radius 3)
  let D := (circle_center 5 4, circle_radius 2)
  let l₁ := (x : ℝ) -> x = 5
  let l₂ := (x y : ℝ) -> 7 * x - 24 * y + 61 = 0 in
  ((∃ P : ℝ × ℝ, ∃ Q : ℝ × ℝ, dist P Q = 5) ∧  (dist (2, 0) (5, 4) = 3 + 2)) ∧
  ( ∃ (P Q : ℝ) (l : ℝ -> ℝ), (l (5,4)) ∧  dist (circle_center 2 0) l = 3) :=
by
  sorry

end circles_are_tangent_and_find_tangent_line_l352_352792


namespace negate_prop_l352_352184

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l352_352184


namespace number_of_ways_to_put_cousins_in_rooms_l352_352119

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l352_352119


namespace updated_mean_of_decrement_l352_352240

theorem updated_mean_of_decrement 
  (mean_initial : ℝ)
  (num_observations : ℕ)
  (decrement_per_observation : ℝ)
  (h1 : mean_initial = 200)
  (h2 : num_observations = 50)
  (h3 : decrement_per_observation = 6) : 
  (mean_initial * num_observations - decrement_per_observation * num_observations) / num_observations = 194 :=
by
  sorry

end updated_mean_of_decrement_l352_352240


namespace ingot_weather_l352_352143

theorem ingot_weather (g b : ℕ) (hg : g + b = 7) :
  (g = 4) ↔ 
  let gold_factor := (1.3 ^ g) * (0.7 ^ b),
      silver_factor := (1.2 ^ g) * (0.8 ^ b)
  in (gold_factor < 1 ∧ silver_factor > 1) ∨ (gold_factor > 1 ∧ silver_factor < 1) :=
by
  sorry

end ingot_weather_l352_352143


namespace sum_525_as_consecutive_odds_l352_352892

noncomputable def count_ways_to_sum_525 : ℕ := 
  let sequences : List ℕ := 
    [ n | n ∈ (List.range 26).tail, -- n ranges from 2 to 25
      ∃ k, 2 * k + 2 * n - 2 = 1050 / n]
  sequences.length

theorem sum_525_as_consecutive_odds :
  count_ways_to_sum_525 = 6 := sorry

end sum_525_as_consecutive_odds_l352_352892


namespace safe_under_conditions_l352_352412

def p_safe (p : ℕ) (n : ℕ) : Prop :=
  ∀ k : ℤ, |n - p * k| > 3

def num_safe_less_than (p q r : ℕ) (N : ℕ) : ℕ :=
  let count := (1 to N).count (λ n, p_safe p n ∧ p_safe q n ∧ p_safe r n)
  count

theorem safe_under_conditions (N : ℕ) : num_safe_less_than 5 7 11 15000 = 0 :=
by
  sorry

end safe_under_conditions_l352_352412


namespace peaches_problem_l352_352597

theorem peaches_problem
  (steven_peaches : ℕ := 19)
  (jake_peaches : ℕ := steven_peaches - 12)
  (jill_peaches : ℕ := jake_peaches / 3)
  (hanna_peaches : ℕ := jake_peaches + 3)
  (lucy_peaches : ℕ := hanna_peaches + 5) :
  lucy_peaches + jill_peaches = 17 :=
by
  have steven_peaches_eq : steven_peaches = 19 := rfl
  have jake_peaches_eq : jake_peaches = steven_peaches - 12 := rfl
  have jake_value : jake_peaches = 7 := by rw [jake_peaches_eq, steven_peaches_eq, Nat.sub_self 12]
  have jill_peaches_eq : jill_peaches = jake_peaches / 3 := rfl
  have jill_value : jill_peaches = 2 := by rw [jill_peaches_eq, jake_value, Nat.div_self 3]
  have hanna_peaches_eq : hanna_peaches = jake_peaches + 3 := rfl
  have hanna_value : hanna_peaches = 10 := by rw [hanna_peaches_eq, jake_value, Nat.add_self 3]
  have lucy_peaches_eq : lucy_peaches = hanna_peaches + 5 := rfl
  have lucy_value : lucy_peaches = 15 := by rw [lucy_peaches_eq, hanna_value, Nat.add_self 5]
  have total_value : lucy_peaches + jill_peaches = 17 := by rw [lucy_value, jill_value, Nat.add_self 2]
  exact total_value

end peaches_problem_l352_352597


namespace modulus_of_z_l352_352776

def i : ℂ := complex.I
def z : ℂ := (1 - i) * (1 + i)^2 + 1

theorem modulus_of_z : complex.abs z = real.sqrt 13 :=
by
  sorry

end modulus_of_z_l352_352776


namespace largest_integer_solution_l352_352595

theorem largest_integer_solution (x : ℤ) : 
  (x - 3 * (x - 2) ≥ 4) → (2 * x + 1 < x - 1) → (x = -3) :=
by
  sorry

end largest_integer_solution_l352_352595


namespace split_marked_points_l352_352299

theorem split_marked_points (T : Type) (side_length : ℕ) (one_center : ℕ) (center_unmarked : Prop)
  (linear_set : set (set T)) :
  side_length = 111 ∧ one_center = 1 ∧ center_unmarked ∧ 
  (∀ s ∈ linear_set, ∃ l, l ⊆ T ∧ ∃ k, l = {p ∈ T | parallel_to_side T p k}) →
  ∃ (n : ℕ), n = 2^4107 :=
by sorry

end split_marked_points_l352_352299


namespace unique_polynomial_degree_ge_one_l352_352862

theorem unique_polynomial_degree_ge_one (f : ℝ → ℝ) (hf : ∃ n ≥ 1, ∃ (a : ℕ → ℝ), f = λ x, ∑ k in Finset.range (n+1), a k * x^k)
    (h1 : ∀ x, f (x^2) = f x ^ 2) (h2 : ∀ x, f (x^2) = f (f x)) : 
    f = λ x, x^2 :=
by 
  sorry

end unique_polynomial_degree_ge_one_l352_352862


namespace least_sum_of_exponents_520_l352_352484

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l352_352484


namespace ratio_of_sides_l352_352543

variable {A B C a b c : ℝ}

theorem ratio_of_sides
  (h1 : 2 * b * Real.sin (2 * A) = 3 * a * Real.sin B)
  (h2 : c = 2 * b) :
  a / b = Real.sqrt 2 := by
  sorry

end ratio_of_sides_l352_352543


namespace root_in_interval_l352_352163

noncomputable def f (x : ℝ) : ℝ := Real.log x + 3 * x - 10

theorem root_in_interval :
  ∃ c ∈ Ioo 2 3, f c = 0 :=
by
  have h0 : f 2 < 0 := by calc
    f 2 = Real.log 2 + 3 * 2 - 10 : rfl
    ... < 0 : by norm_num [Real.log, Real.log_two]

  have h1 : 0 < f 3 := by calc
    f 3 = Real.log 3 + 3 * 3 - 10 : rfl
    ... > 0 : by norm_num [Real.log, Real.log_three]

  exact IntermediateValueTheorem h0 h1 sorry

end root_in_interval_l352_352163


namespace no_adjacent_same_roll_probability_l352_352768

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l352_352768


namespace ordered_pairs_count_l352_352054

noncomputable def number_of_ordered_pairs : ℤ :=
  let S := {p : ℝ × ℕ | 
              (0 < p.fst) ∧ 
              (5 ≤ p.snd) ∧ (p.snd ≤ 25) ∧ 
              ((Real.log p.fst / Real.log p.snd) ^ 4 = Real.log (p.fst ^ 4) / Real.log p.snd) ∧ 
              (p.fst = p.snd ^ (Real.log p.fst / Real.log p.snd))} in
  Finset.card (Finset.filter (λ x, true) (Finset.image (prod.mk) (Finset.range (5, 26) × Finset.range (1, 1000000)))) -- placeholder for actual count

theorem ordered_pairs_count : number_of_ordered_pairs = 42 :=
sorry

end ordered_pairs_count_l352_352054


namespace selection_methods_l352_352419

theorem selection_methods (m f : ℕ) (hm : m = 4) (hf : f = 5) : 
  (∑ i in finset.range (min m 3 + 1), (nat.choose m i) * (nat.choose f (3 - i))) = 70 :=
by
  -- Convert the conditions to the actual fixed values
  rw [hm, hf]
  -- Calculate the sum of the valid combinations
  have hsum : (∑ i in finset.range 4, (nat.choose 4 i) * (nat.choose 5 (3 - i))) = 70, from sorry,
  exact hsum

end selection_methods_l352_352419


namespace cousin_distribution_count_l352_352121

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l352_352121


namespace find_a_even_function_lambda_in_E_find_m_n_l352_352452

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x+1)*(x+a)/x^2

theorem find_a_even_function (h : ∀ x : ℝ, f x a = f (-x) a) : a = -1 := 
by
  sorry

noncomputable def E (a : ℝ) : Set ℝ := {y | ∃ x : ℝ, x ∈ {-1, 1, 2} ∧ y = f x a}

noncomputable def λ := (Real.log 2)^2 + Real.log 2 * Real.log 5 + Real.log 5 - 1/4

theorem lambda_in_E (h : a = -1) : λ ∈ E a := 
by
  sorry

noncomputable def f' (x : ℝ) : ℝ := 1 - 1/x^2

theorem find_m_n (a_eq_neg1 : a = -1) (h : ∀ x : ℝ, f' x ∈ [2 - 3m, 2 - 3n]) : 
  {m n : ℝ // m = (3 + Real.sqrt 5)/2 ∧ n = (3 - Real.sqrt 5)/2 ∧ 0 < n ∧ n < m} := 
by
  sorry

end find_a_even_function_lambda_in_E_find_m_n_l352_352452


namespace simplify_expression_l352_352962

theorem simplify_expression (x : ℝ) : 
  ((3 * x - 6) - 5 * x) / 3 = - (2 / 3) * x - 2 :=
by sorry

end simplify_expression_l352_352962


namespace Sn_expression_l352_352105

-- Define the sequence and the sum of the terms
noncomputable def a : ℕ → ℤ
| 1       := 1
| (n + 2) := -2 * a (n + 1)

noncomputable def S : ℕ → ℤ
| 0       := 0
| (n + 1) := 2 * (1 + a (n + 1))

-- Define the properties
axiom a1_eq_one : a 1 = 1
axiom S_prop (n : ℕ) (hn : n ≥ 2) : S n = 2 * (1 + a n)

-- The final theorem
theorem Sn_expression (n : ℕ) (hn : n ≥ 1) : S n = 2 - 2^n :=
by
  sorry

end Sn_expression_l352_352105


namespace surface_area_of_sphere_of_rectangular_solid_l352_352684

theorem surface_area_of_sphere_of_rectangular_solid (length width height : ℝ) 
  (h1 : length = 2) (h2 : width = 2) (h3 : height = 1)
  (h4 : ∀ x y z w : ℝ, x ∈ {0, length} → y ∈ {0, width} → z ∈ {0, height} → w^2 = x^2 + y^2 + z^2) :
  ∀ (S : ℝ), S = 4 * Real.pi * (3/2)^2 → S = 9 * Real.pi :=
by sorry

end surface_area_of_sphere_of_rectangular_solid_l352_352684


namespace amount_after_two_years_l352_352385

theorem amount_after_two_years (P : ℝ) (r : ℝ) (n : ℕ) (hP : P = 83200) (hr : r = 0.125) (hn : n = 2) : P * (1 + r)^n = 105300 :=
by
  rw [hP, hr, hn]
  norm_num -- check the numeric calculation
  sorry -- rest of proof

end amount_after_two_years_l352_352385


namespace line_equation_with_max_distance_l352_352978

theorem line_equation_with_max_distance
  (P : ℝ × ℝ)
  (h₀ : P = (1, 2))
  (h₁ : ∃ l : ℝ → ℝ → Prop, ∀ (Q : ℝ × ℝ), l Q.1 Q.2 ↔ l 1 2 ∧ ∃ R, R ≠ 0 ∧ l R.1 R.2 = 0)
  (h₂ : ∀ (Q : ℝ × ℝ), Q = (a, b) → (a = 1) → (b = 2))
  : ∃ l : ℝ → ℝ → Prop, (∀ (Q : ℝ × ℝ), l Q.1 Q.2 ↔ x + 2 * y - 5 = 0) :=
 sorry

end line_equation_with_max_distance_l352_352978


namespace part1_extremum_part2_inequality_part3_absolute_value_l352_352460

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * x + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x

theorem part1_extremum (x : ℝ) (hx : 0 < x) :    
  let a := -1 in ∃ c : ℝ, (∀ y, 0 < y ∧ y < x → f a y < f a c) ∧ (∀ z, x < z → f a c > f a z) :=
sorry

theorem part2_inequality (m : ℝ) : (∃ x : ℝ, g x < x + m) ↔ 1 < m :=
sorry

theorem part3_absolute_value (x : ℝ) (hx : 0 < x) : 
  let a := 0 in |f a x - g x| > 2 :=
sorry

end part1_extremum_part2_inequality_part3_absolute_value_l352_352460


namespace partition_odd_numbers_l352_352558

def odd_numbers (n : ℕ) : Finset ℕ := Finset.range n |>.image (λ k, 2 * k + 1)
def has_partition_with_equal_sum (s : Finset ℕ) : Prop :=
  ∃ (A B : Finset ℕ), A ∩ B = ∅ ∧ A ∪ B = s ∧ A.sum id = B.sum id

theorem partition_odd_numbers (n : ℕ) :
  has_partition_with_equal_sum (odd_numbers n) ↔ (n % 2 = 0 ∧ n ≥ 4) := by
  sorry

end partition_odd_numbers_l352_352558


namespace find_a_if_lines_perpendicular_l352_352803

theorem find_a_if_lines_perpendicular (a : ℝ) :
  (∀ x, (y1 : ℝ) = a * x - 2 → (y2 : ℝ) = (a + 2) * x + 1 → y1 * y2 = -1) → a = -1 :=
by {
  sorry
}

end find_a_if_lines_perpendicular_l352_352803


namespace inequality_not_always_hold_l352_352928

theorem inequality_not_always_hold (a b : ℕ) 
  (ha : a > 0) (hb : b > 0) : ¬(∀ a b, a^3 + b^3 ≥ 2 * a * b^2) :=
sorry

end inequality_not_always_hold_l352_352928


namespace sum_of_asymptotes_l352_352176

theorem sum_of_asymptotes :
  let c := -3/2
  let d := -1
  c + d = -5/2 :=
by
  -- Definitions corresponding to the problem conditions
  let c := -3/2
  let d := -1
  -- Statement of the theorem
  show c + d = -5/2
  sorry

end sum_of_asymptotes_l352_352176


namespace petals_in_garden_l352_352740

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end petals_in_garden_l352_352740


namespace study_days_needed_l352_352709

theorem study_days_needed
    (chapters : ℕ) (worksheets : ℕ)
    (hours_per_chapter : ℕ) (hours_per_worksheet : ℕ)
    (max_hours_per_day : ℕ)
    (break_minutes_per_hour : ℕ) (snack_breaks_per_day : ℕ)
    (snack_break_minutes : ℕ) (lunch_break_minutes : ℕ) :
    chapters = 2 →
    worksheets = 4 →
    hours_per_chapter = 3 →
    hours_per_worksheet = 1.5 →
    max_hours_per_day = 4 →
    break_minutes_per_hour = 10 →
    snack_breaks_per_day = 3 →
    snack_break_minutes = 10 →
    lunch_break_minutes = 30 →
    (15 / 4).ceil = 4 := by 
  sorry

end study_days_needed_l352_352709


namespace largest_rectangle_area_l352_352194

theorem largest_rectangle_area (x y : ℝ) (h1 : 2*x + 2*y = 60) (h2 : x ≥ 2*y) : ∃ A, A = x*y ∧ A ≤ 200 := by
  sorry

end largest_rectangle_area_l352_352194


namespace proportion_problem_l352_352481

theorem proportion_problem 
  (x : ℝ) 
  (third_number : ℝ) 
  (h1 : 0.75 / x = third_number / 8) 
  (h2 : x = 0.6) 
  : third_number = 10 := 
by 
  sorry

end proportion_problem_l352_352481


namespace prime_sum_equals_106_l352_352566

open Nat

def is_prime_pair_with_condition (P : ℕ) : Prop :=
  Prime P ∧ Prime (P + 2) ∧ P * (P + 2) ≤ 2007

def sum_of_prime_pairs : ℕ :=
  (Finset.filter is_prime_pair_with_condition (Finset.range 2007)).sum id

theorem prime_sum_equals_106 : sum_of_prime_pairs = 106 := 
  sorry

end prime_sum_equals_106_l352_352566


namespace num_correct_statements_l352_352023

variables (m n : Type) [Line m] [Line n]
variables (α β : Type) [Plane α] [Plane β]

axiom diff_lines : m ≠ n
axiom non_coincident_planes : α ≠ β

axiom perp1 : ∀ {m α β : Type} [Line m] [Plane α], m ⊥ α → β ⊂ α → m ⊥ β
axiom parallel_lines : ∀ {m n α : Type} [Line m] [Line n] [Plane α], m ⊥ α → n ⊥ α → m || n
axiom not_parallel_line_plane : ∀ {m n α : Type} [Line m] [Line n] [Plane α], m ⊥ α → n ⊥ m → ¬(n || α)
axiom not_perpendicular_line_plane : ∀ {m α β : Type} [Line m] [Plane α] [Plane β], α ⊥ β → m || α → ¬(m ⊥ β)
axiom perpendicular_planes : ∀ {m α β : Type} [Line m] [Plane α] [Plane β], m ⊥ α → m || β → α ⊥ β

theorem num_correct_statements :
  ∃ n : ℕ, (n = 2) :=
by
have h1 : (parallel_lines m n α) := perp1 m α β -- Statement 1 is correct
have h2 : ¬(not_parallel_line_plane m n α) := sorry -- Statement 2 is incorrect
have h3 : ¬(not_perpendicular_line_plane m α β) := sorry -- Statement 3 is incorrect
have h4 : (perpendicular_planes m α β) := sorry -- Statement 4 is correct
exists 2
sorry

end num_correct_statements_l352_352023


namespace sequence_bounds_l352_352646

theorem sequence_bounds :
    ∀ (a : ℕ → ℝ), a 0 = 5 → (∀ n : ℕ, a (n + 1) = a n + 1 / a n) → 45 < a 1000 ∧ a 1000 < 45.1 :=
by
  intros a h0 h_rec
  sorry

end sequence_bounds_l352_352646


namespace monotonic_intervals_k_range_l352_352438

noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.exp x
noncomputable def F (x : ℝ) : ℝ := f x - g (x - 1)

-- Proof Problem Part (1)
theorem monotonic_intervals :
  (∀ x ∈ Set.Ioo 0 1, 0 < Deriv F x) ∧ (∀ x ∈ Set.Ioi 1, Deriv F x < 0) :=
  sorry

-- Proof Problem Part (2)
theorem k_range (k : ℝ) : (∀ x ∈ Set.Ici 1, x * f x - k * (x + 1) * f (g (x - 1)) ≤ 0) ↔ (k ∈ Set.Ici (1 / 2)) :=
  sorry

end monotonic_intervals_k_range_l352_352438


namespace base4_more_digits_than_base9_l352_352859

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l352_352859


namespace probability_sum_of_three_dice_eq_18_l352_352505

theorem probability_sum_of_three_dice_eq_18 : 
  (∃ (X Y Z : ℕ), 
    (1 ≤ X ∧ X ≤ 6) ∧ 
    (1 ≤ Y ∧ Y ≤ 6) ∧ 
    (1 ≤ Z ∧ Z ≤ 6) ∧ 
    (X + Y + Z = 18)) ↔ (1/216) :=
begin
  sorry
end

end probability_sum_of_three_dice_eq_18_l352_352505


namespace total_balloons_is_72_l352_352775

-- Definitions for the conditions from the problem
def fred_balloons : Nat := 10
def sam_balloons : Nat := 46
def dan_balloons : Nat := 16

-- The total number of red balloons is the sum of Fred's, Sam's, and Dan's balloons
def total_balloons (f s d : Nat) : Nat := f + s + d

-- The theorem stating the problem to be proved
theorem total_balloons_is_72 : total_balloons fred_balloons sam_balloons dan_balloons = 72 := by
  sorry

end total_balloons_is_72_l352_352775


namespace find_number_l352_352736

theorem find_number :
  ∃ x : ℝ, ((x / 9) - 13) / 7 - 8 = 13 ∧ x = 1440 :=
begin
  existsi 1440,
  split,
  { -- Let's prove the condition ((1440 / 9) - 13) / 7 - 8 = 13.
    calc
      ((1440 / 9) - 13) / 7 - 8
          = ((160 - 13) / 7) - 8  : by simp
      ... = (147 / 7) - 8        : by simp
      ... = 21 - 8               : by simp
      ... = 13                   : by simp },
  { -- And x = 1440
    refl }
end

end find_number_l352_352736


namespace probability_no_adjacent_same_rolls_l352_352772

theorem probability_no_adjacent_same_rolls :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6)
  let no_adjacent_same := outcomes.filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ a)
  (no_adjacent_same.card : ℚ) / outcomes.card = 25 / 108 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352772


namespace solution_proof_l352_352001

noncomputable def f (n : ℕ) : ℝ := Real.logb 143 (n^2)

theorem solution_proof : f 7 + f 11 + f 13 = 2 + 2 * Real.logb 143 7 := by
  sorry

end solution_proof_l352_352001


namespace shaded_area_of_coplanar_squares_l352_352639

variables (side1 side2 side3 : ℝ) (arrangement : side1 = 8 ∧ side2 = 5 ∧ side3 = 3)

theorem shaded_area_of_coplanar_squares (h : arrangement) : 
    ∃ area : ℝ, area = 13.75 :=
begin
  use 13.75,
  sorry
end

end shaded_area_of_coplanar_squares_l352_352639


namespace min_value_and_exp_value_l352_352033

noncomputable def x := 1 / 6
noncomputable def y := 1 / 3

-- Definitions for random variable ξ and probabilities
def ξ : Type := ℕ
def P : ξ → ℝ
| 1 => x
| 2 => 1 / 2
| 3 => y
| _ => 0

-- Sum of all probabilities must equal 1
def total_prob := P 1 + P 2 + P 3 = 1

-- Expected value of ξ
def E_ξ : ℝ := 1 * P 1 + 2 * P 2 + 3 * P 3

-- Proof statement
theorem min_value_and_exp_value :
  ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ (x + y = 1 / 2) ∧
  (∀ x' y', x' > 0 ∧ y' > 0 ∧ x' + y' = 1 / 2 → (1 / x + 4 / y ≤ 1 / x' + 4 / y')) ∧ (E_ξ = 13 / 6) :=
by
  -- Unproven definitions for x and y
  use [x, y]
  split
  { exact sorry }  -- Proof that x > 0
  split
  { exact sorry }  -- Proof that y > 0
  split
  { exact sorry }  -- Proof that x + y = 1 / 2
  split
  { exact sorry }  -- Proof that ∀ x' y', minimization holds
  { exact sorry }  -- Proof that E_ξ = 13 / 6


end min_value_and_exp_value_l352_352033


namespace find_x_l352_352239

theorem find_x
  (x : ℝ)
  (area1 : x^2 + 8 * x + 16)
  (area2 : 4 * x^2 - 12 * x + 9)
  (sum_perimeters : 32) :
  4 * (√(x^2 + 8 * x + 16)) + 4 * (√(4 * x^2 - 12 * x + 9)) = 32 →
  x = 7 / 3 := 
by
  sorry

end find_x_l352_352239


namespace max_value_1_unique_pair_a0_b0_l352_352827

-- First part
theorem max_value_1 (b : ℝ) : 
  let f (x : ℝ) := -x^2 + x + b in
  let M (b : ℝ) := max (|b - 2|) (|b|) in
  (M b) = if b ≤ 1 then |b - 2| else |b| :=
by
  sorry

-- Second part
theorem unique_pair_a0_b0 : 
  ∀ (a b : ℝ), 
  let f (x : ℝ) := -x^2 + a * x + b in
  let M (a b : ℝ) := max (|f 1|) (|f 2|) in
  ∀ a0 b0 : ℝ, 
    (M a b) ≥ (M a0 b0) -> (a0, b0) = (3, -17/8) :=
by
  sorry

end max_value_1_unique_pair_a0_b0_l352_352827


namespace price_reduction_l352_352615

theorem price_reduction (x : ℝ) 
  (initial_price : ℝ := 60) 
  (final_price : ℝ := 48.6) :
  initial_price * (1 - x) * (1 - x) = final_price :=
by
  sorry

end price_reduction_l352_352615


namespace log_product_to_sum_l352_352568

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := log a x

theorem log_product_to_sum
  (a : ℝ) (x : fin 2008 → ℝ)
  (h1 : a > 0)
  (h2 : a ≠ 1)
  (h3 : f (∏ i, x i) a = 8) :
  (∑ i, f ((x i)^2) a) = 16 :=
by apply sorry

end log_product_to_sum_l352_352568


namespace smallest_quotient_is_neg5_l352_352700

theorem smallest_quotient_is_neg5 :
  let nums := [-1, 2, -3, 0, 5]
  ∃ x y ∈ nums, x ≠ 0 ∧ y ≠ 0 ∧ x / y = -5 :=
by
  let nums := [-1, 2, -3, 0, 5]
  use 5
  use -1
  split
  -- proof that 5 and -1 are in nums
  sorry
  split
  -- proof that 5 ≠ 0
  sorry
  split
  -- proof that -1 ≠ 0
  sorry
  -- proof that 5 / -1 = -5
  sorry

end smallest_quotient_is_neg5_l352_352700


namespace base4_more_digits_than_base9_l352_352852

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l352_352852


namespace polygon_proof_l352_352297
noncomputable theory

-- Definitions and conditions
variable (p : ℝ) (s : ℝ) (P : ℝ) (S : ℝ) (m_quad : Prop) (m_obtuse : Prop)
def shifted_polygon := P - p > 6 ∧ S - s > 15
def shifted_quadrilateral := m_quad → P - p ≥ 8 ∧ S - s ≥ 16
def shifted_obtuse_polygon := m_obtuse → P - p < 8 ∧ S - s < 16

-- Lean statement combining all proof parts
theorem polygon_proof :
  p = 12 →
  (∀ (s P S : ℝ), shifted_polygon p s P S) →
  (∀ (m_quad : Prop), m_quad → shifted_quadrilateral p s P S m_quad) →
  (∀ (m_obtuse : Prop), m_obtuse → shifted_obtuse_polygon p s P S m_obtuse) :=
by
  intros hp hp_proof hq_proof ho_proof
  sorry

end polygon_proof_l352_352297


namespace least_number_divisible_by_conditions_l352_352178

theorem least_number_divisible_by_conditions : 
  ∃ n : ℕ, 
  n = 856 ∧ 
  ∀ (k : ℕ), 
  (k ∈ {32, 36, 54} → (n + 8) % k = 0) ∧ 
  32 = 32 := 
begin
  use 856,
  split,
  { refl },
  { intros k hk,
    fin_cases hk,
    { exact nat.mod_eq_zero_of_dvd (by norm_num : 864 % 32 = 0) },
    { exact nat.mod_eq_zero_of_dvd (by norm_num : 864 % 36 = 0) },
    { exact nat.mod_eq_zero_of_dvd (by norm_num : 864 % 54 = 0) },
  },
  refl,
end

end least_number_divisible_by_conditions_l352_352178


namespace angle_sum_bounds_l352_352233

theorem angle_sum_bounds {P A B C D : Point}
  (θ : ℝ) (h_distinct : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ P ≠ D ∧ A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
  (h_acute : 0 < θ ∧ θ < π/2)
  (h_angles : ∀ (X Y Z : Point), (X = A ∧ Y = P ∧ Z = B) ∨ (X = B ∧ Y = P ∧ Z = C) ∨ (X = C ∧ Y = P ∧ Z = D) ∨ (X = D ∧ Y = P ∧ Z = A) → angle X Y Z = θ) :
  let α := angle A P C,
      β := angle B P D
  in (α + β ≤ 2 * arccos (2 * cos θ - 1)) ∧ (α + β ≥ 0) :=
sorry

end angle_sum_bounds_l352_352233


namespace carousel_seat_coloring_l352_352205

theorem carousel_seat_coloring (total_seats : ℕ) (yc bc rc : ℕ)
  (h1 : total_seats = 100)
  (h2 : yc + bc + rc = total_seats)
  (h3 : ∃ n₃ n₂, n₃ < n₂ ∧ n₂ < total_seats ∧ n₃ + 3 = 23 ∧ n₂ + 3 = 23 ∧ pp + 49 = 7)
-- Each color block is contiguous.
(h4 : ∀ i j : ℕ, i < j →  j < total_seats
  → ((∃ k, k < total_seats ∧ i = k + yc) →
  ( ∃ t, t < total_seats ∧ j = t + bc) →
  pp - 3 = -49)
--Blue seat No. 7 is opposite red seat No. 3
(h5 : λ bc' rc')
(h6 : bc.* 4 = ∑ x in bc, 1, (bc' = nn))

--yellow seat No. 7 opposite red seat No. 23
  (h7 : λ yc' rc12')
-- sum of blue seats
(h8 : λ bc' rc', bc' = 13 + 6)

: yc = 34 ∧ bc = 20 ∧ rc = 46 := 
sorry

end carousel_seat_coloring_l352_352205


namespace prob_at_least_one_male_prob_distribution_and_expectation_l352_352259

-- Define constants
def num_students : ℕ := 8
def num_males : ℕ := 4
def num_females : ℕ := 4
def select_count : ℕ := 4

-- Part (1): Prove the probability that the selected 4 students include at least one male student
theorem prob_at_least_one_male : 
  let total_ways := Nat.choose num_students select_count,
      ways_no_males := Nat.choose num_females select_count in
  (1 - (ways_no_males / total_ways) : ℚ) = 69 / 70 :=
by 
  sorry

-- Part (2): Prove the probability distribution and expectation of the random variable X
theorem prob_distribution_and_expectation :
  let X_prob : Fin (select_count + 1) → ℚ := 
      λ k, (Nat.choose num_females k * Nat.choose num_males (select_count - k)) / Nat.choose num_students select_count,
      E_X := ∑ k in Finset.range (select_count + 1), k * X_prob k in
  (X_prob 0, X_prob 1, X_prob 2, X_prob 3, X_prob 4, E_X) = 
  (1 / 70, 8 / 35, 18 / 35, 8 / 35, 1 / 70, 2) :=
by 
  sorry

end prob_at_least_one_male_prob_distribution_and_expectation_l352_352259


namespace terminating_decimals_count_l352_352416

theorem terminating_decimals_count : finset.card (finset.filter (λ n : ℕ, 1 ≤ n ∧ n ≤ 990) (finset.range 991)) = 990 :=
by
  sorry

end terminating_decimals_count_l352_352416


namespace diamond_value_l352_352372

variable {a b : ℤ}

-- Define the operation diamond following the given condition.
def diamond (a b : ℤ) : ℚ := (1 : ℚ) / a + (1 : ℚ) / b

-- Define the conditions given in the problem.
axiom h1 : a + b = 10
axiom h2 : a * b = 24

-- State the target theorem.
theorem diamond_value : diamond a b = 5 / 12 :=
by
  sorry

end diamond_value_l352_352372


namespace cyclic_pentagon_iff_distance_condition_l352_352580

variables (A B C D E : Type)
variables (d : E → (A × B) → ℝ)

def cyclic_pentagon (ABCDE : Prop) : Prop :=
  ∀ (A B C D E : E), 
  (d E (A, B) * d E (C, D) = d E (A, C) * d E (B, D)) ∧
  (d E (A, C) * d E (B, D) = d E (A, D) * d E (B, C)) ↔
  (exists (R : ℝ), ∀ (A B C D E : E), distance_from_circle(E, (A, B, C, D, E), R))

def distance_from_circle (E : E) (points : (A × B × C × D × E)) (R : ℝ) : Prop := sorry

theorem cyclic_pentagon_iff_distance_condition :
  cyclic_pentagon A B C D E d :=
by sorry

end cyclic_pentagon_iff_distance_condition_l352_352580


namespace find_function_formula_l352_352816

-- Given conditions
axiom function_def (x : ℝ) : ℝ
axiom is_tangent_parallel (a b : ℝ) : f x = -3x → f'(-1) = -3

-- Proving the formula of the function f(x)
theorem find_function_formula (a b : ℝ) :
  (function_def x = ax^2 + bx - 1) ∧
  is_tangent_parallel a b →
  f x = -x^2 - 5x - 1 :=
by
  sorry

end find_function_formula_l352_352816


namespace max_sum_of_entries_in_table_l352_352987

def consecutive (m n : ℕ) : Prop := (m = n + 1) ∨ (n = m + 1)

theorem max_sum_of_entries_in_table :
  ∀ (a b c d e f g h : ℕ),
  a ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  b ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  c ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  d ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  e ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  f ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  g ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  h ∈ {2, 3, 5, 7, 11, 13, 17, 19} →
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h) →
  (b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h) →
  (c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h) →
  (d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h) →
  (e ≠ f ∧ e ≠ g ∧ e ≠ h) →
  (f ≠ g ∧ f ≠ h) →
  (g ≠ h) →
  (consecutive a e ∨ consecutive b f ∨ consecutive c g ∨ consecutive d h) →
  (a + b + c + d) * (e + f + g + h) ≤ 1440 :=
begin
  sorry
end

end max_sum_of_entries_in_table_l352_352987


namespace train_crossing_time_l352_352475

/-- 
  Prove that the train takes 75 seconds to cross the bridge given:
  - The train is 250 meters long.
  - The train's speed is 72 kmph.
  - The bridge is 1,250 meters long.
--/
theorem train_crossing_time :
  ∀ (train_length : ℝ) (train_speed_kmph : ℝ) (bridge_length : ℝ),
  train_length = 250 → 
  train_speed_kmph = 72 → 
  bridge_length = 1250 →
  let train_speed_mps := train_speed_kmph * (1000 / 3600)
      total_distance := train_length + bridge_length
      crossing_time := total_distance / train_speed_mps
  in crossing_time = 75 :=
by
  intros
  sorry

end train_crossing_time_l352_352475


namespace cousin_distribution_count_l352_352122

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l352_352122


namespace original_curve_eqn_l352_352641

-- Definitions based on conditions
def scaling_transformation_formula (x y : ℝ) : ℝ × ℝ :=
  (2 * x, 3 * y)

def transformed_curve (x'' y'' : ℝ) : Prop :=
  x''^2 + y''^2 = 1

-- The proof problem to be shown in Lean
theorem original_curve_eqn {x y : ℝ} (h : transformed_curve (2 * x) (3 * y)) :
  4 * x^2 + 9 * y^2 = 1 :=
sorry

end original_curve_eqn_l352_352641


namespace problem1_problem2_l352_352817

-- Problem Statements
-- (1) Prove that the function f(x) = x - 1/x is increasing on (0, +∞)
theorem problem1 {x : ℝ} (hx : 0 < x) : 
  ∀ x1 x2 : ℝ, 0 < x1 → 0 < x2 → x1 < x2 → f(x1) < f(x2) :=
  by
  let f (x : ℝ) := x - 1 / x
  sorry

-- (2) Solve for the range of real number values of m in the equation 
-- 2^t * f(4^t) - m * f(2^t) = 0 when t in [1, 2]
theorem problem2 {m t : ℝ} (ht : 1 ≤ t ∧ t ≤ 2) : 
  (5 ≤ m ∧ m ≤ 17) :=
  by
  let f (x : ℝ) := x - 1 / x
  let eqn := 2^t * (f (4^t)) - m * (f (2^t))
  have h_eq : eqn = 0 := sorry
  sorry

end problem1_problem2_l352_352817


namespace units_in_building_l352_352135

variable (U : ℕ)  -- total number of units in the building
variable (monthly_rent_per_resident annual_income total_annual_rent : ℝ)

-- Given conditions
def occupied_fraction := (3 / 4) * U
def monthly_rent_per_resident := 400
def annual_income_per_unit := 12 * monthly_rent_per_resident
def total_annual_rent := 360000
def income_equation := occupied_fraction * annual_income_per_unit = total_annual_rent

-- Concluding statement
theorem units_in_building (h : income_equation) : U = 100 :=
by sorry

end units_in_building_l352_352135


namespace arithmetic_prog_coprime_l352_352938

theorem arithmetic_prog_coprime :
  ∃ (a : ℕ → ℕ), ∀ (i j : ℕ), i ≠ j → 1 ≤ i → i ≤ 100 → 1 ≤ j → j ≤ 100 → Nat.coprime (a i) (a j) :=
by
  let a := λ (i : ℕ), 1 + (i - 1) * Nat.factorial 99
  use a
  intros i j h_ij h_i1 h_i100 h_j1 h_j100
  have h_dif : a i - a j = (i - j) * Nat.factorial 99 := by
    simp [a, sub_eq_iff_eq_add, mul_sub]
  apply Nat.coprime_of_dvd
  intro p hp
  cases hp
  · simp [Nat.gcd, hp] at h_ij
  · simp [Nat.gcd, hp] at h_ij
  · sorry

end arithmetic_prog_coprime_l352_352938


namespace matrix_power_four_correct_l352_352319

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l352_352319


namespace base_digit_difference_l352_352849

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l352_352849


namespace sport_formulation_water_l352_352238

theorem sport_formulation_water
  ( flavoring_ratio_standard : ℚ )
  ( corn_syrup_ratio_standard : ℚ )
  ( water_ratio_standard : ℚ )
  ( corn_syrup_sport : ℚ )
  ( sport_flavoring_to_corn_syrup : ℚ )
  ( sport_flavoring_to_water : ℚ ) :
  { flavoring_ratio_standard = 1 / 43;
    corn_syrup_ratio_standard = 12 / 43;
    water_ratio_standard = 30 / 43;
    sport_flavoring_to_corn_syrup = 3 * (1 / 12);
    sport_flavoring_to_water = (1 / 2) * (1 / 30);
    corn_syrup_sport = 5 } →
  water_sport = 18.75 :=
by
  sorry

end sport_formulation_water_l352_352238


namespace x_intercept_of_line_l352_352629

theorem x_intercept_of_line : 
  let p1 := (3 : ℝ, 9 : ℝ), 
      p2 := (-1 : ℝ, 1 : ℝ) in
  ∃ x : ℝ, y_intercept_of_line p1 p2 0 = x := -3 / 2 :=
sorry

end x_intercept_of_line_l352_352629


namespace calculate_expression_l352_352314

theorem calculate_expression : 
  ((-1 / 2) ^ 2) + (2 ^ (-2)) - ((2 - real.pi) ^ 0) = -1 / 2 :=
sorry

end calculate_expression_l352_352314


namespace amount_spent_on_drink_l352_352550

-- Definitions based on conditions provided
def initialAmount : ℝ := 9
def remainingAmount : ℝ := 6
def additionalSpending : ℝ := 1.25

-- Theorem to prove the amount spent on the drink
theorem amount_spent_on_drink : 
  initialAmount - remainingAmount - additionalSpending = 1.75 := 
by 
  sorry

end amount_spent_on_drink_l352_352550


namespace common_ratio_geometric_sequence_l352_352499

/--
Given a ∈ ℝ and the sequence:
  a + log 3 2017,
  a + log 9 2017,
  a + log 27 2017,
if they form a geometric sequence, prove that the common ratio is 1/3.
-/
theorem common_ratio_geometric_sequence (a : ℝ) (h : (a + log 3 2017), (a + log 9 2017), (a + log 27 2017) forms_geom_seq) : 
  ∃ q, q = 1/3 := 
sorry

end common_ratio_geometric_sequence_l352_352499


namespace petya_min_n_minimum_n_for_petya_win_l352_352245

def set_A : Finset ℕ := Finset.range 1003
def Petya_wins (n : ℕ) : Prop :=
  ∀ s ∈ set_A.powerset.filter (λ t, t.card = n), ∃ x y ∈ s, Nat.coprime x y

theorem petya_min_n : ∀ n ≥ 502, Petya_wins n :=
by sorry

lemma petya_min_n_value : ∀ n < 502, ¬ Petya_wins n :=
by sorry

-- The minimum n that Petya must name to guarantee a win is 502
theorem minimum_n_for_petya_win : ∃ n, n = 502 ∧ Petya_wins n :=
by
  existsi 502
  split
  · refl
  · apply petya_min_n
    exact dec_trivial

end petya_min_n_minimum_n_for_petya_win_l352_352245


namespace binom_12_10_l352_352347

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l352_352347


namespace diamond_eval_l352_352370

def diamond (i j : ℕ) : ℕ :=
  match i, j with
  | 1, 1 => 2 | 1, 2 => 1 | 1, 3 => 3 | 1, 4 => 5 | 1, 5 => 4
  | 2, 1 => 1 | 2, 2 => 5 | 2, 3 => 4 | 2, 4 => 3 | 2, 5 => 2
  | 3, 1 => 3 | 3, 2 => 4 | 3, 3 => 2 | 3, 4 => 1 | 3, 5 => 5
  | 4, 1 => 5 | 4, 2 => 2 | 4, 3 => 1 | 4, 4 => 4 | 4, 5 => 3
  | 5, 1 => 4 | 5, 2 => 3 | 5, 3 => 5 | 5, 4 => 2 | 5, 5 => 1
  | _, _ => 0

theorem diamond_eval :
  diamond (diamond 4 5) (diamond 1 3) = 2 :=
by {
  have h1 : diamond 4 5 = 3 := rfl,
  have h2 : diamond 1 3 = 3 := rfl,
  show diamond 3 3 = 2,
  exact rfl
}

end diamond_eval_l352_352370


namespace remainder_div_l352_352724

theorem remainder_div (n : ℕ) : (1 - 90 * Nat.choose 10 1 + 90^2 * Nat.choose 10 2 - 90^3 * Nat.choose 10 3 + 
  90^4 * Nat.choose 10 4 - 90^5 * Nat.choose 10 5 + 90^6 * Nat.choose 10 6 - 90^7 * Nat.choose 10 7 + 
  90^8 * Nat.choose 10 8 - 90^9 * Nat.choose 10 9 + 90^10 * Nat.choose 10 10) % 88 = 1 := by
  sorry

end remainder_div_l352_352724


namespace clock_angle_at_10am_l352_352304

theorem clock_angle_at_10am :
  let min_angle := 0 in
  let hour_hand_per_hour := 30 in
  let hour_position := 10 in
  let total_degrees_in_circle := 360 in
  let hour_hand_angle := hour_hand_per_hour * hour_position in
  let smaller_angle := total_degrees_in_circle - hour_hand_angle in
  smaller_angle = 60 :=
by
  trivial

end clock_angle_at_10am_l352_352304


namespace largest_negative_is_l352_352298

def largest_of_negatives (a b c d : ℚ) (largest : ℚ) : Prop := largest = max (max a b) (max c d)

theorem largest_negative_is (largest : ℚ) : largest_of_negatives (-2/3) (-2) (-1) (-5) largest → largest = -2/3 :=
by
  intro h
  -- We assume the definition and the theorem are sufficient to say largest = -2/3
  sorry

end largest_negative_is_l352_352298


namespace tanya_total_sticks_l352_352161

theorem tanya_total_sticks (n : ℕ) (h : n = 11) : 3 * (n * (n + 1) / 2) = 198 :=
by
  have H : n = 11 := h
  sorry

end tanya_total_sticks_l352_352161


namespace smallest_k_exists_l352_352734

theorem smallest_k_exists :
  ∃ k : ℕ, (∀ a n, (0 ≤ a ∧ a ≤ 1) → a^k * (1 - a)^n < (1 / (n + 1)^3)) ∧ k = 4 :=
by
  let k := 4
  use k
  intros a n h₀ h₁
  sorry

end smallest_k_exists_l352_352734


namespace least_sum_exponents_of_520_l352_352490

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l352_352490


namespace smallest_positive_integer_b_l352_352756
-- Import the necessary library

-- Define the conditions and problem statement
def smallest_b_factors (r s : ℤ) := r + s

theorem smallest_positive_integer_b :
  ∃ r s : ℤ, r * s = 1800 ∧ ∀ r' s' : ℤ, r' * s' = 1800 → smallest_b_factors r s ≤ smallest_b_factors r' s' :=
by
  -- Declare that the smallest positive integer b satisfying the conditions is 85
  use 45, 40
  -- Check the core condition
  have rs_eq_1800 := (45 * 40 = 1800)
  sorry

end smallest_positive_integer_b_l352_352756


namespace find_min_value_remainder_1000_l352_352003

def sum_of_digits (n : ℕ) (b : ℕ) : ℕ :=
  let rec digits_sum n acc :=
    if n = 0 then acc else digits_sum (n / b) (acc + n % b)
  digits_sum n 0

def f (n : ℕ) : ℕ :=
  sum_of_digits n 3

def g (n : ℕ) : ℕ :=
  sum_of_digits (f n) 7

def base_nine_digit_sum_equals_10 (n : ℕ) : Prop :=
  sum_of_digits (g n) 9 = 10

theorem find_min_value_remainder_1000 :
  let M := Nat.find_x (λ n, base_nine_digit_sum_equals_10 n)
  M % 1000 = 186 :=
sorry

end find_min_value_remainder_1000_l352_352003


namespace equilateral_triangle_cd_value_l352_352195

theorem equilateral_triangle_cd_value (c d : ℝ) (h_eq_triangle : ∃ (c d : ℝ), (0, 0), (c, 17), and (d, 53) form an equilateral triangle):
  c * d = 16011 / 9 :=
sorry

end equilateral_triangle_cd_value_l352_352195


namespace triangle_side_ratio_l352_352789

theorem triangle_side_ratio (A B C D E F P : Type)
  [incircle : touches_incircle A B B C C A B D E F]
  [right_angle : ∠ B = 90]
  [AD_intersects_P : intersects A D in P]
  [PF_perpendicular_PC : PF ⊥ PC] :
  ratio (side_lengths (triangle A B C)) = 3:4:5 := 
sorry

end triangle_side_ratio_l352_352789


namespace planes_intersecting_dodecahedron_in_hexagon_l352_352894

-- Define the structure of a regular dodecahedron
structure RegularDodecahedron :=
  (faces : Fin 12 → RegularPentagon)
  (vertices : Fin 20)
  (edges : Fin 30)
  (largeDiagonals : Fin 10)

-- The main theorem to be proved
theorem planes_intersecting_dodecahedron_in_hexagon (d : RegularDodecahedron) :
  ∃ n : ℕ, n = 30 ∧ 
  (∀ plane : Plane, 
   plane.intersects(dodecahedron := d) -> 
   plane.cross_section.shape = RegularHexagon) :=
sorry

end planes_intersecting_dodecahedron_in_hexagon_l352_352894


namespace sixteen_a_four_plus_one_div_a_four_l352_352990

theorem sixteen_a_four_plus_one_div_a_four (a : ℝ) (h : 2 * a - 1 / a = 3) :
  16 * a^4 + (1 / a^4) = 161 :=
sorry

end sixteen_a_four_plus_one_div_a_four_l352_352990


namespace parabola_properties_l352_352895

theorem parabola_properties :
  let C := {p : ℝ × ℝ // p.1 ^ 2 = 4 * p.2}
  let p : ℝ := 2
  let vertex := (0, 0)
  let focus := (1, 0)
  -- Standard equation of the parabola
  (∀ (x y : ℝ), y ^ 2 = 4 * x → x = p / 2 → y ^ 2 = 4 * x) ∧
  -- The moving line AB always passes through the fixed point (1, 0)
  (∀ (y1 y2 : ℝ), y1 * y2 = -4 → 
    let M := (-1, y1)
    let N := (-1, y2)
    let A := (4 / y1 ^ 2, -4 / y1)
    let B := (4 / y2 ^ 2, -4 / y2)
    let AB_slope := ((-4 / y1) - (-4 / y2)) / ((4 / y1 ^ 2) - (4 / y2 ^ 2))
    let AB := λ (x y : ℝ), y = AB_slope * (x - (4 / y1 ^ 2)) + (-4 / y1)
    AB 1 0 = 0) :=
sorry

end parabola_properties_l352_352895


namespace sam_grew_3_carrots_l352_352950

-- Let Sandy's carrots and the total number of carrots be defined
def sandy_carrots : ℕ := 6
def total_carrots : ℕ := 9

-- Define the number of carrots grown by Sam
def sam_carrots : ℕ := total_carrots - sandy_carrots

-- The theorem to prove
theorem sam_grew_3_carrots : sam_carrots = 3 := by
  sorry

end sam_grew_3_carrots_l352_352950


namespace problem_part1_problem_part2_l352_352899

def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * cos θ, ρ * sin θ)

def curve_C (θ : ℝ) : ℝ × ℝ :=
  polar_to_rectangular (2 * sin θ + 4 * cos θ) θ

def line_l (t : ℝ) : ℝ × ℝ :=
  ((sqrt 3 / 2) * t, 1 + (1 / 2) * t)

def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem problem_part1 (ρ θ : ℝ) (h0 : 0 ≤ θ ∧ θ < 2 * π) :
  let P := (4, 2)
  P = argmax (λ Q, distance (0, 0) Q) (curve_C θ) := sorry

theorem problem_part2 (M A B : ℝ × ℝ) :
  let t1 t2 := solutions_of_quadratic -2*sqrt 3 -1 t
  let A := line_l t1, B := line_l t2
  let MA := distance M A, MB := distance M B
  M = (1, π / 2)
  (1 / MA) + (1 / MB) = 4 := sorry

end problem_part1_problem_part2_l352_352899


namespace cousins_in_rooms_l352_352130

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l352_352130


namespace remainder_17_plus_x_mod_31_l352_352100

theorem remainder_17_plus_x_mod_31 {x : ℕ} (h : 13 * x ≡ 3 [MOD 31]) : (17 + x) % 31 = 22 := 
sorry

end remainder_17_plus_x_mod_31_l352_352100


namespace monthly_fee_for_second_plan_l352_352274

theorem monthly_fee_for_second_plan 
  (monthly_fee_first_plan : ℝ) 
  (rate_first_plan : ℝ) 
  (rate_second_plan : ℝ) 
  (minutes : ℕ) 
  (monthly_fee_second_plan : ℝ) :
  monthly_fee_first_plan = 22 -> 
  rate_first_plan = 0.13 -> 
  rate_second_plan = 0.18 -> 
  minutes = 280 -> 
  (22 + 0.13 * 280 = monthly_fee_second_plan + 0.18 * 280) -> 
  monthly_fee_second_plan = 8 := 
by
  intros h_fee_first_plan h_rate_first_plan h_rate_second_plan h_minutes h_equal_costs
  sorry

end monthly_fee_for_second_plan_l352_352274


namespace binom_12_10_l352_352342

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l352_352342


namespace intersection_of_AB_and_CD_l352_352079

-- Define the points A, B, C, D
def A : ℝ × ℝ × ℝ := (6, -7, 7)
def B : ℝ × ℝ × ℝ := (16, -17, 12)
def C : ℝ × ℝ × ℝ := (0, 3, -6)
def D : ℝ × ℝ × ℝ := (2, -5, 10)

-- Define the intersection point
def intersection_point : ℝ × ℝ × ℝ := (4 / 3, -7 / 3, 14 / 3)

-- Prove that the intersection point of AB and CD is the given point
theorem intersection_of_AB_and_CD :
  ∃ t s : ℝ, 
    let P := (6 + 10 * t, -7 - 10 * t, 7 + 5 * t) in
    let Q := (2 * s, 3 - 8 * s, -6 + 16 * s) in
    P = intersection_point ∧ Q = intersection_point :=
sorry

end intersection_of_AB_and_CD_l352_352079


namespace determine_percent_copper_l352_352213

variables (x y : ℝ)
variables (m1 m2 : ℝ)

-- Conditions
def condition1 : Prop := x + 40 = y
def condition2 : Prop := (x / 100) * m1 = 6
def condition3 : Prop := (y / 100) * m2 = 12
def condition4 : Prop := 0.36 * (m1 + m2) = 18
def condition5 : Prop := m1 + m2 = 18

-- Correct Answer
def answer : Prop := x = 20 ∧ y = 60

-- Proof problem
theorem determine_percent_copper (h1 : condition1) 
                               (h2 : condition2) 
                               (h3 : condition3) 
                               (h4 : condition4)
                               (h5 : condition5) : answer :=
by
  sorry

end determine_percent_copper_l352_352213


namespace centroid_property_l352_352874

variables {ABC : Type} [triangle ABC]
variables {A B C G P A' B' C' : ABC}
-- G is the centroid
variable (is_centroid : centroid G A B C)
-- P is an interior point
variable (interior_point : triangle.interior P A B C)
-- PG intersects sides at points A', B', C'
variables (PA' PB' PC' : line_intersection (line P G) ⟨ B, C ⟩ A' ∧ 
                            line_intersection (line P G) ⟨ C, A ⟩ B' ∧ 
                            line_intersection (line P G) ⟨ A, B ⟩ C')

theorem centroid_property :
  (A'P / A'G) + (B'P / B'G) + (C'P / C'G) = 3 :=
sorry

end centroid_property_l352_352874


namespace find_a_l352_352974

-- Define the binomial expansion condition
def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

-- Given conditions for the binomial expansion
def condition (a : ℝ) : Prop :=
  ∃ (r : ℕ), r = 2 ∧ ∑ i in Finset.range 3, binomial_coeff 10 i * (real.sqrt 1)^i * (a / (1^2))^(10 - i) = 180

-- Theorem statement
theorem find_a (a : ℝ) (h : condition a) : a = 2 ∨ a = -2 :=
by
  sorry

end find_a_l352_352974


namespace number_of_ways_to_make_divisible_by_45_l352_352539

-- Define the main problem as a hypotheses and a theorem to prove.
theorem number_of_ways_to_make_divisible_by_45 : 
  let known_digits_sum := 2 + 0 + 1 + 6 + 0 + 2 in
  let choices_per_pos := 9 in
  let num_positions := 5 in
  (2 * choices_per_pos ^ num_positions) = 13122 :=
by
  let known_digits_sum := 2 + 0 + 1 + 6 + 0 + 2 in
  let choices_per_pos := 9 in
  let num_positions := 5 in
  have choices_last_digit := 2,
  have total_choices := choices_last_digit * choices_per_pos ^ num_positions,
  exact total_choices = 13122

end number_of_ways_to_make_divisible_by_45_l352_352539


namespace not_always_all_ones_l352_352719

def transformation_step (board : Array (Array Int)) : Array (Array Int) :=
  let n := board.size
  let m := if n = 0 then 0 else board[0].size
  let get_or_default (i j : Int) := if 0 <= i ∧ i < n ∧ 0 <= j ∧ j < m then board[i.toNat][j.toNat] else 1
  Array.init n (λ i => Array.init m (λ j => 
    get_or_default (i-1) j * get_or_default (i+1) j *
    get_or_default i (j-1) * get_or_default i (j+1)))

def evolves_to_all_ones (initial_board : Array (Array Int)) : Prop :=
  let n := initial_board.size
  let m := if n = 0 then 0 else initial_board[0].size
  let rec aux (k : Nat) (board : Array (Array Int)) : Prop :=
    if k = 0 then False
    else if board.all (λ row => row.all (λ cell => cell = 1)) then True
    else aux (k - 1) (transformation_step board)
  aux (n * m) initial_board

theorem not_always_all_ones : 
  ∃ (initial_board : Array (Array Int)), initial_board.size = 9 ∧ initial_board.all (λ row => row.size = 9) ∧
  ∃ t, ¬ evolves_to_all_ones initial_board := 
sorry

end not_always_all_ones_l352_352719


namespace equal_parallelograms_iff_diagonal_l352_352941

variables {A B C D K L M N O : Point}
variables (hABCD : Parallelogram A B C D)
variables (hK : OnSegment K A B)
variables (hL : OnSegment L B C)
variables (hM : OnSegment M C D)
variables (hN : OnSegment N D A)
variables (hKM_parallel : Parallel (Line K M) (Line A B))
variables (hLN_parallel : Parallel (Line L N) (Line B C))
variables (hIntersect_O : Intersect (Line K M) (Line L N) O)

theorem equal_parallelograms_iff_diagonal :
  Area (Parallelogram K B L O) = Area (Parallelogram M D N O) ↔ OnLine O (Diagonal A C) :=
begin
  sorry
end

end equal_parallelograms_iff_diagonal_l352_352941


namespace rectangle_area_l352_352997

theorem rectangle_area (w l : ℕ) (h_sum : w + l = 14) (h_w : w = 6) : w * l = 48 := by
  sorry

end rectangle_area_l352_352997


namespace solve_inequality_for_all_real_l352_352594

noncomputable def problem_statement (x y : ℝ) : Prop :=
  2^(-cos x ^ 2) + 2^(-sin x ^ 2) ≥ sin y + cos y

theorem solve_inequality_for_all_real (x y : ℝ) : problem_statement x y :=
by sorry

end solve_inequality_for_all_real_l352_352594


namespace b_50_is_3678_l352_352374

def sequence_b (n : ℕ) : ℕ :=
  Nat.recOn n 3 (λ n b_n, b_n + 3 * n)

theorem b_50_is_3678 : sequence_b 50 = 3678 := by
  sorry

end b_50_is_3678_l352_352374


namespace ascending_order_l352_352007

noncomputable def a := Real.log 4 / Real.log 3 -- \( \log_{3}{4} \)
def b := 2⁻² -- \( 2^{-2} \)
noncomputable def c := Real.log 5 / Real.log (1 / 5) -- \( \log_{0.2}{5} \)

theorem ascending_order : c < b ∧ b < a := 
by
  -- Proof not required as per the instruction 
  sorry

end ascending_order_l352_352007


namespace minimum_dot_product_of_hyperbola_intersection_l352_352269

theorem minimum_dot_product_of_hyperbola_intersection (A B : ℝ × ℝ)
  (hA : A.1 ^ 2 - A.2 ^ 2 = 1)
  (hB : B.1 ^ 2 - B.2 ^ 2 = 1)
  (h_pos_A : 0 < A.1)
  (h_pos_B : 0 < B.1) :
  ∃ f : ℝ, f = (A.1 * B.1 + A.2 * B.2) ∧ f ≥ 1 :=
begin
  sorry,
end

end minimum_dot_product_of_hyperbola_intersection_l352_352269


namespace distinct_lines_count_l352_352465

theorem distinct_lines_count :
  let S := {-3, -2, -1, 0, 1, 2, 3}
  in ∀ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c 
     ∧ (b ≠ 0 ∧ -a / b > 0 ∨ a ≠ 0 ∧ -b / a > 0) 
     ↔ 
     43 
:=
sorry

end distinct_lines_count_l352_352465


namespace other_acute_angle_of_right_triangle_l352_352065

theorem other_acute_angle_of_right_triangle (a : ℝ) (h₀ : 0 < a ∧ a < 90) (h₁ : a = 20) :
  ∃ b, b = 90 - a ∧ b = 70 := by
    sorry

end other_acute_angle_of_right_triangle_l352_352065


namespace distance_between_skew_medians_regular_tetrahedron_l352_352395

/-- Define a regular tetrahedron with edge length 1. -/
structure regular_tetrahedron (T : Type) :=
  (A B C D : T)
  (edge_length : ∀ {X Y : T}, (X = A ∨ X = B ∨ X = C ∨ X = D) ∧ (Y = A ∨ Y = B ∨ Y = C ∨ Y = D) → (X ≠ Y) → dist X Y = 1)

/-- Define a median as the midpoint of a line segment. -/
def midpoint {T : Type} [metric_space T] (x y : T) : T :=
  (1 / 2) * x + (1 / 2) * y

/-- Define the projection of a point onto a plane perpendicular to a face and passing through an edge. -/
noncomputable def projection (p : T) (n : T) (v : T) : T :=
  p - (inner_product p n / inner_product n n) * n

/-- Define the distance between skew medians of a triangle in a regular tetrahedron of edge length 1. -/
theorem distance_between_skew_medians_regular_tetrahedron (T : Type) [metric_space T] :
  ∀ (t : regular_tetrahedron T),
  let A := t.A,
      B := t.B,
      C := t.C,
      D := t.D,
      K := midpoint A B,
      M := midpoint A C,
      N := midpoint C D,
      D1 := projection D (B - A) (C - A),
      M1 := projection M (B - A) (C - A),
      N1 := projection N (B - A) (C - A) in
  let d1 := dist K (D1 + (M1 - D1) * inner_product K (M1 - D1) / inner_product (M1 - D1) (M1 - D1)),
      d2 := dist K (N1 + (K - N1) * inner_product K (K - N1) / inner_product (K - N1) (K - N1)) in
  min d1 d2 = sqrt (1 / 10) :=
sorry

end distance_between_skew_medians_regular_tetrahedron_l352_352395


namespace base4_more_digits_than_base9_l352_352858

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l352_352858


namespace a_seq_monotone_sum_ineq_l352_352446

variable {n : ℕ}

-- Condition: Given a_n is the real root of the equation x^3 + x / 2^n = 1
def is_real_root (a : ℝ) (n : ℕ) : Prop :=
  a^3 + a / (2 ^ n) = 1

-- Define the sequence {a_n}
noncomputable def a_seq (n : ℕ) : ℝ :=
classical.some (real.exists_root_of_cont_mono (λ x, x^3 + x / (2 ^ n)) (by sorry) (by sorry))

-- Proposition 1: The sequence {a_n} is monotonically increasing
theorem a_seq_monotone (k : ℕ) (hk : k ∈ ℕ) : 
  let a_k := a_seq k,
      a_k1 := a_seq (k + 1)
  in a_k < a_k1 :=
sorry

-- Proposition 2: Summation inequality
theorem sum_ineq (n : ℕ) :
  ∑ k in range (n + 1), 1 / (k * (2 ^ k + 1) * a_seq k) < a_seq n :=
sorry

end a_seq_monotone_sum_ineq_l352_352446


namespace roots_are_complex_conjugates_l352_352559

theorem roots_are_complex_conjugates {a b : ℝ}
  (h1 : ∀ z : ℂ, z^2 + ((12 : ℝ) + a * complex.I) * z + (35 + b * complex.I) = 0 → ∃ (x y : ℝ), x ≠ 0 ∧ z = x + y * complex.I ∧ z = complex.conj (x + y * complex.I)) :
  a = 0 ∧ b = 0 :=
sorry

end roots_are_complex_conjugates_l352_352559


namespace age_problem_l352_352272

theorem age_problem 
  (A : ℕ) 
  (x : ℕ) 
  (h1 : 3 * (A + x) - 3 * (A - 3) = A) 
  (h2 : A = 18) : 
  x = 3 := 
by 
  sorry

end age_problem_l352_352272


namespace polynomial_roots_l352_352564

theorem polynomial_roots (P : ℝ → ℝ) (hP : ∃ (P0 P1 P2 : ℝ), ∀ x : ℝ, P(x) = P0 + P1 * x + P2 * x^2) :
  ∃ (x1 x2 : ℝ), x1 = (1 + Real.sqrt 5) / 2 ∧ x2 = (1 - Real.sqrt 5) / 2 ∧ P(x1) = 0 ∧ P(x2) = 0 :=
by
  sorry

end polynomial_roots_l352_352564


namespace number_of_black_squares_in_56th_row_l352_352264

def total_squares (n : Nat) : Nat := 3 + 2 * (n - 1)

def black_squares (n : Nat) : Nat :=
  if total_squares n % 2 == 1 then
    (total_squares n - 1) / 2
  else
    total_squares n / 2

theorem number_of_black_squares_in_56th_row :
  black_squares 56 = 56 :=
by
  sorry

end number_of_black_squares_in_56th_row_l352_352264


namespace area_equilateral_triangle_AMK_l352_352942

/-- Points K and M are located on side BC and the altitude BP of the acute-angled triangle ABC respectively. 
    Given AP = 3, PC = 11/2, and BK:KC = 10:1, prove that the area of the equilateral triangle AMK is 49 / sqrt 3. -/
theorem area_equilateral_triangle_AMK
  (A B C K M : Point)
  (hK_on_BC : K ∈ line_segment B C)
  (hM_on_BP : M ∈ line_segment B (projection B (line A C)))
  (AP : length A P = 3)
  (PC : length P C = 11 / 2)
  (BK_KC_ratio : length B K / length K C = 10 / 1) :
  area (equilateral_triangle A M K) = 49 / real.sqrt 3 :=
  sorry

end area_equilateral_triangle_AMK_l352_352942


namespace find_angle_x_l352_352896

theorem find_angle_x 
  (A B C D E : Type) 
  [geometry E] -- Assuming E is a point in the geometric space
  (h_intersect : intersect AB CD E)
  (h_equilateral : equilateral_triangle B C E)
  (h_right_triangle : right_triangle A D E)
  (h_angle_ADE_90 : angle A D E = 90):

  angle D E A = 60 → 
  angle x = 30 :=
by 
  sorry

end find_angle_x_l352_352896


namespace order_fractions_l352_352939

theorem order_fractions : (16/13 : ℚ) < 21/17 ∧ 21/17 < 20/15 :=
by {
  -- use cross-multiplication:
  -- 16*17 < 21*13 -> 272 < 273 -> true
  -- 16*15 < 20*13 -> 240 < 260 -> true
  -- 21*15 < 20*17 -> 315 < 340 -> true
  sorry
}

end order_fractions_l352_352939


namespace smallest_quotient_is_neg5_l352_352701

theorem smallest_quotient_is_neg5 :
  let nums := [-1, 2, -3, 0, 5]
  ∃ x y ∈ nums, x ≠ 0 ∧ y ≠ 0 ∧ x / y = -5 :=
by
  let nums := [-1, 2, -3, 0, 5]
  use 5
  use -1
  split
  -- proof that 5 and -1 are in nums
  sorry
  split
  -- proof that 5 ≠ 0
  sorry
  split
  -- proof that -1 ≠ 0
  sorry
  -- proof that 5 / -1 = -5
  sorry

end smallest_quotient_is_neg5_l352_352701


namespace book_price_increase_l352_352617

open Real

theorem book_price_increase (P : ℝ) :
    let P1 := P * 1.15,
        P2 := P1 * 1.20,
        P3 := P2 * 1.25 in
    P3 = P * 1.725 :=
by
  -- The proof steps will go here, for now we use 'sorry' to indicate incomplete proof.
  sorry

end book_price_increase_l352_352617


namespace problem_l352_352554

theorem problem (n : ℕ) (p : ℕ) (a b c : ℤ)
  (hn : 0 < n)
  (hp : Nat.Prime p)
  (h_eq : a^n + p * b = b^n + p * c)
  (h_eq2 : b^n + p * c = c^n + p * a) :
  a = b ∧ b = c := 
sorry

end problem_l352_352554


namespace actual_value_wrongly_copied_l352_352982

theorem actual_value_wrongly_copied (mean_initial : ℝ) (n : ℕ) (wrong_value : ℝ) (mean_correct : ℝ) :
  mean_initial = 140 → n = 30 → wrong_value = 135 → mean_correct = 140.33333333333334 →
  ∃ actual_value : ℝ, actual_value = 145 :=
by
  intros
  sorry

end actual_value_wrongly_copied_l352_352982


namespace find_m_l352_352837

open Real

noncomputable def a : ℝ × ℝ := (1, sqrt 3)
noncomputable def b (m : ℝ) : ℝ × ℝ := (3, m)
noncomputable def dot_prod (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem find_m (m : ℝ) (h : dot_prod a (b m) / magnitude a = 3) : m = sqrt 3 :=
by
  sorry

end find_m_l352_352837


namespace solve_system_l352_352731

noncomputable def validate_solution (x y z : ℝ) : Prop :=
  x = 1 ∧ y = 1 / 2 ∧ z = 1 / 3

theorem solve_system : validate_solution 1 (1 / 2) (1 / 3) :=
by
  have eq1 : (1 : ℝ) * (1 / 2) + (1 / 2) * (1 / 3) + (1 / 3) * (1) = 1,
  { norm_num },
  have eq2 : 5 * (1 : ℝ) + 8 * (1 / 2) + 9 * (1 / 3) = 12,
  { norm_num },
  exact ⟨rfl, rfl, rfl⟩

end solve_system_l352_352731


namespace max_students_is_p_squared_l352_352888

variable (m p n : ℕ)
variable (hp : Nat.Prime p)
variable (cond1 : ∀ x : Fin m, ∃ k : Fin n, True)
variable (cond2 : ∀ x : Fin m, ∑ y : Fin n, 1 ≤ p + 1)
variable (cond3 : ∀ y : Fin n, ∑ x : Fin m, 1 ≤ p)
variable (cond4 : ∀ x y : Fin m, x ≠ y → ∃ k : Fin n, k ∈ cond1 x ∧ k ∈ cond1 y)

theorem max_students_is_p_squared (p : ℕ) [h_prime : Fact (Nat.Prime p)]
    (h_cond1 : ∀ s : Fin m, ∃ subj : Fin n, subject subj taken_by s)
    (h_cond2 : ∀ s : Fin m, (∑ subj : Fin n, subject subj taken_by s) ≤ p + 1)
    (h_cond3 : ∀ subj : Fin n, (∑ stud : Fin m, subject subj taken_by stud) ≤ p)
    (h_cond4 : ∀ s1 s2 : Fin m, s1 ≠ s2 → ∃ subj : Fin n, subject subj taken_by s1 ∧ subject subj taken_by s2) :
    m ≤ p * p :=
sorry

end max_students_is_p_squared_l352_352888


namespace probability_pair_form_desired_line_l352_352427

theorem probability_pair_form_desired_line (points : Finset (Fin 5)) (h₀ : points.card = 5) (h₁ : ∀ p1 p2 p3 ∈ points, ¬ collinear ({p1, p2, p3} : Set (Fin 5))) :
  probability (exists_pair_points_form_line points) = 1 / 10 := 
sorry

end probability_pair_form_desired_line_l352_352427


namespace monotone_on_minus_pi_div_two_pi_div_two_no_extreme_points_a_ge_one_div_3_pi_sq_one_extreme_point_zero_lt_a_lt_one_div_3_pi_sq_l352_352040

section Part1

-- Define the function f when a = 0
def f (x : ℝ) : ℝ := 2 * sin x - x * cos x

-- Prove that f(x) is monotonically increasing on (-π/2, π/2)
theorem monotone_on_minus_pi_div_two_pi_div_two : 
  ∀ x : ℝ, x > -Real.pi / 2 ∧ x < Real.pi / 2 → (deriv f x) > 0 := sorry

end Part1

section Part2

-- Define the function f for the general case
def f (a x : ℝ) : ℝ := a * x^3 + 2 * sin x - x * cos x

-- Derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + cos x + x * sin x

-- Second derivative, g'
def g' (a x : ℝ) : ℝ := 6 * a * x + x * cos x

-- Prove number of extreme points on (0, π) for a >= 1/(3π^2)
theorem no_extreme_points_a_ge_one_div_3_pi_sq (a : ℝ) (h : a ≥ 1 / (3 * Real.pi^2)) : 
  ∀ x : ℝ, x > 0 ∧ x < Real.pi → (f' a x) ≠ 0 := sorry

-- Prove number of extreme points on (0, π) for 0 < a < 1/(3π^2)
theorem one_extreme_point_zero_lt_a_lt_one_div_3_pi_sq (a : ℝ) (h : 0 < a ∧ a < 1 / (3 * Real.pi^2)) : 
  ∃ x : ℝ, x > 0 ∧ x < Real.pi ∧ (f' a x) = 0 := sorry

end Part2

end monotone_on_minus_pi_div_two_pi_div_two_no_extreme_points_a_ge_one_div_3_pi_sq_one_extreme_point_zero_lt_a_lt_one_div_3_pi_sq_l352_352040


namespace check_incorrect_l352_352229

-- Definitions for vectors and magnitudes
variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

def is_unit_vector (e : V) : Prop :=
  ∥e∥ = 1

def are_parallel (u v : V) : Prop :=
  ∃ (k : ℝ), u = k • v

-- Definitions for conditions in the problem
def condition_A (e : V) : Prop :=
  is_unit_vector e → ∥e∥ = 1

def condition_B (a b c : V) : Prop :=
  a = 2 • c ∧ b = -4 • c → are_parallel a b

def condition_C (A B C D : V) : Prop :=
  (A - B) = k • (C - D) ∧ ∥A - D∥ = ∥B - C∥ → (
    -- This implies additional conditions for parallelogram, which we negate,
    -- we will need proper negation in proof but sorry place holder is provided.
    -- We do not need the proof steps, hence,
    sorry
  )

def condition_D (u v1 v2 : V) : Prop :=
  ¬are_parallel v1 v2 → ∃ (a b : ℝ), u = a • v1 + b • v2

-- Statement for checking the conditions
theorem check_incorrect (A B C D : V) : condition_C A B C D = False := sorry

end check_incorrect_l352_352229


namespace least_significant_digit_base_4_189_l352_352369

theorem least_significant_digit_base_4_189 :
  ∃ d rems q, (189 = 4 * q + d) ∧ (d = 1) ∧ (list.reversed (Nat.digitList 4 189)).head = d :=
by
  sorry

end least_significant_digit_base_4_189_l352_352369


namespace number_of_female_athletes_drawn_l352_352289

def total_athletes (male female : ℕ) : ℕ := male + female
def proportion_of_females (total female : ℕ) : ℚ := (female : ℚ) / (total : ℚ)
def expected_females_drawn (proportion : ℚ) (sample_size : ℕ) : ℚ := proportion * (sample_size : ℚ)

theorem number_of_female_athletes_drawn :
  let total := total_athletes 48 36 in
  let proportion := proportion_of_females total 36 in
  let expected_drawn := expected_females_drawn proportion 21 in
  expected_drawn = 9 := 
by
  let total := total_athletes 48 36
  let proportion := proportion_of_females total 36
  let expected_drawn := expected_females_drawn proportion 21
  have h_clean: (expected_drawn : ℚ) = 9 := sorry -- proof skipped
  exact h_clean

end number_of_female_athletes_drawn_l352_352289


namespace rectangle_ABCD_AB_eq_20_l352_352532

theorem rectangle_ABCD_AB_eq_20 {A B C D P Q : Type} (rect : is_rectangle A B C D)
  (P_on_BC : P ∈ line_segment B C) (BP_eq_20 : dist B P = 20) 
  (CP_eq_10 : dist C P = 10) (tan_APD_eq_2 : tan (angle A P D) = 2) :
  dist A B = 20 :=
sorry

end rectangle_ABCD_AB_eq_20_l352_352532


namespace find_common_difference_l352_352449

theorem find_common_difference
  (a_1 : ℕ := 1)
  (S : ℕ → ℕ)
  (h1 : S 5 = 20)
  (h2 : ∀ n, S n = n / 2 * (2 * a_1 + (n - 1) * d))
  : d = 3 / 2 := 
by 
  sorry

end find_common_difference_l352_352449


namespace parabola_passes_through_n_points_of_grid_l352_352285

theorem parabola_passes_through_n_points_of_grid (n : ℕ) :
  (∃ (a b c : ℝ), ∃ (p : fin n → ℝ × ℝ), 
    (∀ i, (p i).2 = a * (p i).1^2 + b * (p i).1 + c) ∧ 
    (∀ i j, i ≠ j → p i ≠ p j)) ↔ 
  n = 8 :=
by
  sorry

end parabola_passes_through_n_points_of_grid_l352_352285


namespace binomial_12_10_eq_66_l352_352359

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l352_352359


namespace decrease_in_average_age_l352_352603

variable (X : ℕ)

theorem decrease_in_average_age 
  (h1 : ∀ (X : ℕ), X + 120 = 160) 
  (h2 : ∀ (X : ℕ), 48 * X + 120 * 32 = 5760) 
  (h3 : 160 = 160) 
  : 48 - (5760 / 160) = 12 :=
by
  -- ℕ stands for natural numbers and is required for the proof
  have X := 40
  have total_original := 48 * X
  have total_new := 120 * 32
  have total_age := total_original + total_new
  have avg_age := total_age / 160
  have decrease := 48 - avg_age
  exact decrease

end decrease_in_average_age_l352_352603


namespace simplify_sqrt4_l352_352224

def simplified_expression : ℕ := 
  let x := 2^8 * 3^2 * 5^3
  let y := x^(1/4 : ℝ)
  y

theorem simplify_sqrt4 (a b : ℕ) (p : a = 4) (q : b = 1125) :
  a + b = 1129 :=
by
  sorry

end simplify_sqrt4_l352_352224


namespace holly_distance_l352_352839

def steps_per_mile : ℕ := 1500
def max_steps : ℕ := 10000
def reset_count : ℕ := 50
def final_steps : ℕ := 8000

theorem holly_distance : 
  let total_steps := (max_steps * reset_count) + final_steps in
  let miles_walked := total_steps / steps_per_mile in
  abs (miles_walked - 350) <= abs (miles_walked - 400) ∧
  abs (miles_walked - 350) <= abs (miles_walked - 450) ∧
  abs (miles_walked - 350) <= abs (miles_walked - 500) ∧
  abs (miles_walked - 350) <= abs (miles_walked - 550) :=
by
  sorry

end holly_distance_l352_352839


namespace cousin_distribution_count_l352_352120

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l352_352120


namespace display_total_cans_l352_352074

def row_num_cans (row : ℕ) : ℕ :=
  if row < 7 then 19 - 3 * (7 - row)
  else 19 + 3 * (row - 7)

def total_cans : ℕ :=
  List.sum (List.map row_num_cans (List.range 10))

theorem display_total_cans : total_cans = 145 := 
  sorry

end display_total_cans_l352_352074


namespace range_of_a_l352_352016

-- Define the odd function f(x)
def f (x : ℝ) (a : ℝ) : ℝ :=
  if x > 0 then x^2 + 3*x + a
  else if x < 0 then -(x^2 + 3*(-x) + a)
  else 0

-- Define the function g(x) in terms of f(x)
def g (x : ℝ) (a : ℝ) : ℝ :=
  f x a - x

-- The theorem statement translating the equivalent proof problem
theorem range_of_a (h : ∀ x ≠ 0, f x a = - f (-x) a) (h_zeros : (∃ y > 0, g y a = 0 ∧ ∀ x > y, g x a > 0)) :
  a < 0 :=
by
  sorry

end range_of_a_l352_352016


namespace f_f_of_2_equals_neg_2_l352_352820

noncomputable def f : ℝ → ℝ :=
  λ x, if 0 < x ∧ x < 1 then Real.log x / Real.log 2 else 1 / x^2

theorem f_f_of_2_equals_neg_2 : f (f 2) = -2 :=
by
  sorry

end f_f_of_2_equals_neg_2_l352_352820


namespace tangent_line_minimum_length_l352_352691

theorem tangent_line_minimum_length :
  ∀ (x y : ℝ), (x^2 + y^2 - 2*x + 4*y - 1 = 0) → (x + y = 5) → 2 * real.sqrt 3 = sorry :=
begin
  sorry
end

end tangent_line_minimum_length_l352_352691


namespace matrix_power_difference_l352_352920

def B : Matrix (Fin 2) (Fin 2) ℤ :=
  ![[3, 2], [0, -1]]

theorem matrix_power_difference :
  (B^15 - 4 • B^14) = ![[ -3, -4], [ 0, 5 ]] :=
by
  sorry

end matrix_power_difference_l352_352920


namespace find_pair_r_x_l352_352251

noncomputable def x_repr (r : ℕ) (n : ℕ) (ab : ℕ) : ℕ :=
  ab * (r * (r^(2*(n-1)) - 1) / (r^2 - 1))

noncomputable def x_squared_repr (r : ℕ) (n : ℕ) : ℕ :=
  (r^(4*n) - 1) / (r - 1)

theorem find_pair_r_x (r x : ℕ) (n : ℕ) (ab : ℕ) (r_leq_70 : r ≤ 70)
  (x_consistent: x = x_repr r n ab)
  (x_squared_consistent: x^2 = x_squared_repr r n)
  : (r = 7 ∧ x = 20) :=
begin
  sorry
end

end find_pair_r_x_l352_352251


namespace determine_number_of_true_statements_l352_352377

def equilateral_triangle_internal_angles := ∀ (t : Triangle), (t.equilateral) → (∀ a ∈ t.angles, a = 60)

def converse_equilateral_triangle (t : Triangle) := (∀ a ∈ t.angles, a = 60) → t.equilateral

def congruent_triangles_equal_area := ∀ (t1 t2 : Triangle), (t1 ≅ t2) → (area t1 = area t2)

def negation_congruent_triangles := ¬ (∀ (t1 t2 : Triangle), (t1 ≅ t2) → (area t1 = area t2))

def real_roots_condition (k : ℝ) := (k > 0) → (∃ x : ℝ, x^2 + 2*x - k = 0)

def inverse_negation_real_roots_condition (k : ℝ) := ¬ (¬ (real_roots_condition k))

noncomputable def correct_answer: ℕ := 2 -- two true statements

theorem determine_number_of_true_statements : 
  (converse_equilateral_triangle = true ∧ 
   negation_congruent_triangles = false ∧ 
   (∀ k > 0, inverse_negation_real_roots_condition k = true)) → 
  (correct_answer = 2) := 
sorry

end determine_number_of_true_statements_l352_352377


namespace parabola_c_value_correct_l352_352258

noncomputable def parabola_c_value (a b c : ℝ) : Prop :=
  (∀ y : ℝ, x : ℝ, (x = a * y^2 + b * y + c) →
                   ((y = -3 → x = 5) ∧ (y = 1 → x = 7)))


theorem parabola_c_value_correct :
  ∃ (a b : ℝ), parabola_c_value a b (49 / 8) :=
by
  exists (1 / 8)
  exists (3 / 4)
  sorry

end parabola_c_value_correct_l352_352258


namespace intersection_lies_on_circumcircle_l352_352280

-- Define the geometric entities and conditions
variables (O A₁ A₂ P : Type) [Point O] [Point A₁] [Point A₂] [Point P]
variables (l₁ l₂ : Line O)

-- Hypothesis: O is the center of rotation mapping line l₁ to line l₂,
-- point A₁ on line l₁ to point A₂ on line l₂, and P is their intersection.
hypothesis (hrot_line: ∀ (x : Point), x ∈ l₁ → ∃ y ∈ l₂, rotation O x y)
hypothesis (hrot_point: ∀ (x : Point), x = A₁ → ∃ y ∈ l₂, y = A₂ ∧ rotation O x y)
hypothesis (hintersect: ∀ (x : Point), x ∈ l₁ ∧ x ∈ l₂ → x = P)

-- Goal: Prove that P lies on the circumcircle of triangle A₁ O A₂
theorem intersection_lies_on_circumcircle :
  lies_on_circumcircle P (triangle A₁ O A₂) :=
by sorry

end intersection_lies_on_circumcircle_l352_352280


namespace perpendicular_AD_IP_l352_352925

-- Let's define the necessary geometric entities and conditions

variables {A B C I D L M P : Point}
variables {triangle_ABC : Triangle A B C}
variables [incircle_center_I : Incircle I triangle_ABC]
variables [tangent_D : Tangent D (line B C)]
variables [tangent_L : Tangent L (line A C)]
variables [tangent_M : Tangent M (line A B)]
variables [intersection_P : Intersection P (line M L) (line B C)]

theorem perpendicular_AD_IP 
  (h1: Center I (incircle triangle_ABC))
  (h2: Tangent D (line B C) (incircle triangle_ABC))
  (h3: Tangent L (line A C) (incircle triangle_ABC))
  (h4: Tangent M (line A B) (incircle triangle_ABC))
  (h5: Intersection P (line M L) (line B C)) :
  Perpendicular (line A D) (line I P) := sorry

end perpendicular_AD_IP_l352_352925


namespace area_of_12_gon_with_given_sides_l352_352513

noncomputable def is_convex_polygon_with_equal_angles 
  (n : ℕ) (lengths : Fin n → ℝ) : Prop :=
  (∀ i j, 0 ≤ lengths i ∧ lengths i = lengths j) ∧
  (∀ i, lengths i > 0) ∧
  ∑ i, lengths i = 2 * π

theorem area_of_12_gon_with_given_sides : 
  is_convex_polygon_with_equal_angles 12 
   (fun i => if i = 10 then 2 else if i < 10 then 1 else 0) → 
   (∃ area : ℝ, area = 8 + 4 * Real.sqrt 3) :=
by
  intro h
  use 8 + 4 * Real.sqrt 3
  sorry

end area_of_12_gon_with_given_sides_l352_352513


namespace set_satisfies_conditions_l352_352729

theorem set_satisfies_conditions (a : ℝ) :
  let a1 := 1
  let a2 := 1
  let a3 := 1
  let a4 := a
  let a5 := a
  in
  (a1 = a2 * a3) ∧
  (a2 = a1 * a3) ∧
  (a3 = a1 * a2) ∧
  (a4 = a1 * a5) ∧
  (a5 = a1 * a4) :=
by {
  let a1 := 1
  let a2 := 1
  let a3 := 1
  let a4 := a
  let a5 := a
  split;
  sorry
}

end set_satisfies_conditions_l352_352729


namespace probability_between_two_values_l352_352806

noncomputable def normal_dist (mean variance : ℝ) : measure_theory.measure ℝ :=
sorry -- assume we have a measure representing a normal distribution

theorem probability_between_two_values 
  (μ : ℝ) (σ² : ℝ) (hμ : μ = 3) (hx₄ : ∫ x in (-∞, 4], normal_dist μ σ² = 0.84) :
  ∫ x in (2, 4), normal_dist μ σ² = 0.68 :=
sorry

end probability_between_two_values_l352_352806


namespace money_left_after_shopping_l352_352091

-- Conditions
def cost_mustard_oil : ℤ := 2 * 13
def cost_pasta : ℤ := 3 * 4
def cost_sauce : ℤ := 1 * 5
def total_cost : ℤ := cost_mustard_oil + cost_pasta + cost_sauce
def total_money : ℤ := 50

-- Theorem to prove
theorem money_left_after_shopping : total_money - total_cost = 7 := by
  sorry

end money_left_after_shopping_l352_352091


namespace cousins_room_distributions_l352_352125

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l352_352125


namespace find_50th_term_index_l352_352730

def a_n (n : ℕ) : ℝ :=
  ∑ k in Finset.range n, Real.cos k

theorem find_50th_term_index :
  ∃ n, (∃ i, i < 50 ∧ a_n n > 0) ∧ n = 157 := 
by
  sorry

end find_50th_term_index_l352_352730


namespace range_of_k_l352_352866

theorem range_of_k (k : ℝ) : (∃ x y : ℝ, x^2 + k * y^2 = 2) ∧ (∀ x y : ℝ, y ≠ 0 → x^2 + k * y^2 = 2 → (x = 0 ∧ (∃ a : ℝ, a > 1 ∧ y = a))) → 0 < k ∧ k < 1 :=
sorry

end range_of_k_l352_352866


namespace r_class_variations_including_first_s_elements_l352_352841

variable (n r s : ℕ)

theorem r_class_variations_including_first_s_elements (h : s < r) : 
  (finset.univ.card ^ (n - s) * (r.factorial) / (r - s).factorial / (n - r).factorial) =
  (n - s)! * r! / (r - s)! / (n - r)!:=
  sorry

end r_class_variations_including_first_s_elements_l352_352841


namespace tan_alpha_l352_352445

theorem tan_alpha {α : ℝ} (h1 : sin α - cos α = 1 / 5) (h2 : 0 < α ∧ α < π) : tan α = 4 / 3 :=
sorry

end tan_alpha_l352_352445


namespace shrimp_cost_per_pound_l352_352218

theorem shrimp_cost_per_pound 
    (shrimp_per_guest : ℕ) 
    (num_guests : ℕ) 
    (shrimp_per_pound : ℕ) 
    (total_cost : ℝ)
    (H1 : shrimp_per_guest = 5)
    (H2 : num_guests = 40)
    (H3 : shrimp_per_pound = 20)
    (H4 : total_cost = 170) : 
    (total_cost / ((num_guests * shrimp_per_guest) / shrimp_per_pound) = 17) :=
by
    sorry

end shrimp_cost_per_pound_l352_352218


namespace min_value_ab_l352_352501

theorem min_value_ab (a b : ℝ) (h : 1/a + 2/b = real.sqrt (a * b)) : a * b ≥ real.sqrt 2 * 2 :=
by
sorry

end min_value_ab_l352_352501


namespace price_decrease_required_to_initial_l352_352196

theorem price_decrease_required_to_initial :
  let P0 := 100.0
  let P1 := P0 * 1.15
  let P2 := P1 * 0.90
  let P3 := P2 * 1.20
  let P4 := P3 * 0.70
  let P5 := P4 * 1.10
  let P6 := P5 * (1.0 - d / 100.0)
  P6 = P0 -> d = 5.0 :=
by
  sorry

end price_decrease_required_to_initial_l352_352196


namespace find_annual_interest_rate_l352_352306

variable (r : ℝ) -- The annual interest rate we want to prove

-- Define the conditions based on the problem statement
variable (I : ℝ := 300) -- interest earned
variable (P : ℝ := 10000) -- principal amount
variable (t : ℝ := 9 / 12) -- time in years

-- Define the simple interest formula condition
def simple_interest_formula : Prop :=
  I = P * r * t

-- The statement to prove
theorem find_annual_interest_rate : simple_interest_formula r ↔ r = 0.04 :=
  by
    unfold simple_interest_formula
    simp
    sorry

end find_annual_interest_rate_l352_352306


namespace determine_lambda_l352_352469

open Nat

def satisfies_recursive_relation (a : ℕ → ℤ) : Prop :=
  ∀ n, a (n + 1) = 2 * a n + 2 ^ n - 1

def is_arithmetic_sequence (b : ℕ → ℤ) : Prop :=
  ∃ d, ∀ n, b (n + 1) - b n = d

theorem determine_lambda (a : ℕ → ℤ) (λ : ℤ) :
  satisfies_recursive_relation a →
  is_arithmetic_sequence (λ n, (a n + λ) / (2 ^ n)) →
  λ = -1 :=
by
  intro h1 h2
  sorry

end determine_lambda_l352_352469


namespace pixel_final_color_after_2019_applications_l352_352690

def f (n : ℕ) : ℕ :=
  if n ≤ 19 then n + 4 else abs (129 - 2 * n)

theorem pixel_final_color_after_2019_applications : (f^[2019] 5) = 75 :=
  sorry

end pixel_final_color_after_2019_applications_l352_352690


namespace sum_of_minimum_values_of_P_Q_l352_352934

/- Monic quadratic polynomials P(x) and Q(x) -/
def isMonicQuadratic (f : ℝ → ℝ) := ∃ a b : ℝ, a = 1 ∧ f = λ x, a * x^2 + b * x + 0

/- Condition that P(Q(x)) has zeros at x = -7, -5, -3, -1 -/
def hasZerosPofQ (P Q : ℝ → ℝ) := 
  P (Q (-7)) = 0 ∧ P (Q (-5)) = 0 ∧ P (Q (-3)) = 0 ∧ P (Q (-1)) = 0

/- Condition that Q(P(x)) has zeros at x = -14, -12, -10, -8 -/
def hasZerosQofP (P Q : ℝ → ℝ) := 
  Q (P (-14)) = 0 ∧ Q (P (-12)) = 0 ∧ Q (P (-10)) = 0 ∧ Q (P (-8)) = 0

/- Defining the minimum value of a quadratic polynomial -/
noncomputable def minValueQuadratic (P : ℝ → ℝ) : ℝ :=
  let b := (λ x, (P x) - x^2) 0 in (P (-b/2))

/- The main theorem we want to prove -/
theorem sum_of_minimum_values_of_P_Q (P Q : ℝ → ℝ)
  (hP : isMonicQuadratic P) (hQ : isMonicQuadratic Q) (hPofQ : hasZerosPofQ P Q) (hQofP : hasZerosQofP P Q) :
  minValueQuadratic P + minValueQuadratic Q = -180 :=
sorry

end sum_of_minimum_values_of_P_Q_l352_352934


namespace sum_of_special_integers_l352_352177

theorem sum_of_special_integers : 
  (∑ n in finset.range 79, 2120 + n * 101) = 478661 := by
  sorry

end sum_of_special_integers_l352_352177


namespace election_total_valid_votes_l352_352891

theorem election_total_valid_votes (V B : ℝ) 
    (hA : 0.45 * V = B * V + 250) 
    (hB : 2.5 * B = 62.5) :
    V = 1250 :=
by
  sorry

end election_total_valid_votes_l352_352891


namespace product_of_solutions_of_abs_equation_l352_352403

theorem product_of_solutions_of_abs_equation :
  (∃ x₁ x₂ : ℚ, |5 * x₁ - 2| + 7 = 52 ∧ |5 * x₂ - 2| + 7 = 52 ∧ x₁ ≠ x₂ ∧ (x₁ * x₂ = -2021 / 25)) :=
sorry

end product_of_solutions_of_abs_equation_l352_352403


namespace base4_more_digits_than_base9_l352_352854

def base4_digits_1234 : ℕ := 6
def base9_digits_1234 : ℕ := 4

theorem base4_more_digits_than_base9 :
  base4_digits_1234 - base9_digits_1234 = 2 :=
by
  sorry

end base4_more_digits_than_base9_l352_352854


namespace solve_equation_l352_352813

-- Define x and y, and the given equation
noncomputable def equation (x : ℝ) (a : ℝ) : Prop :=
  x^4 + 4 * x^2 + 1 - a * x * (x^2 - 1) = 0

-- Define the quadratic equation after substitution
noncomputable def quadratic (y : ℝ) (a : ℝ) : Prop :=
  y^2 - a * y + 6 = 0

-- Define the conditions under which the roots of the quadratic equation are equal
noncomputable def equal_roots (a : ℝ) : Prop :=
  a = 2 * Real.sqrt 6 ∨ a = -2 * Real.sqrt 6

-- Define the expected values of x when the roots of the quadratic equation are equal
noncomputable def expected_x_values : set ℝ :=
  { x | x = (Real.sqrt 6 + Real.sqrt 10) / 2 ∨
          x = (Real.sqrt 6 - Real.sqrt 10) / 2 ∨
          x = (-Real.sqrt 6 + Real.sqrt 10) / 2 ∨
          x = (-Real.sqrt 6 - Real.sqrt 10) / 2 }

-- Define the main theorem
theorem solve_equation (a : ℝ) (x : ℝ) :
  (∀ x, equation x a → ∃ y, quadratic y a ∧ (a = 2 * Real.sqrt 6 → y = Real.sqrt 6 ∨ a = -2 * Real.sqrt 6 → y = -Real.sqrt 6)) →
  (equal_roots a → x ∈ expected_x_values) :=
sorry

end solve_equation_l352_352813


namespace incircle_tangency_triangle_congruency_and_cyclic_l352_352234

variables {X Y Z : Type}
variable  [triangle XYZ : Type]
variables {T W : XYZ → Prop}

theorem incircle_tangency (XYZ : Type) [triangle XYZ] (XY XZ YZ : ℝ) (XT XW : ℝ)
  (incircle_touches_XY_at_T : T X)
  (incircle_touches_XZ_at_W : W X) :
  XT = ((XY + XZ - YZ) / 2) :=
sorry

variables {A B C D : Type}
variable  [triangle ABC : Type]
variables {I J : Type}
variables {M N P K : Type}

theorem triangle_congruency_and_cyclic (ABC : Type) [triangle ABC] (D : ABC → Prop)
 (I J : Type) (IMK KNJ : Prop) (IDJK : Prop)
 (foot_perpendicular_D : D A)
 (incenter_ABD : I)
 (incenter_ACD : J)
 (incenter_touches_AD_at_M : M = I)
 (incenter_touches_AD_at_N : N = J)
 (incircle_ABC_touches_AB_at_P : P = A)
 (circle_with_center_A_radius_AP_intersects_AD_at_K : K = A):
  (IMK ∧ KNJ ∧ IDJK) :=
sorry

end incircle_tangency_triangle_congruency_and_cyclic_l352_352234


namespace negation_of_proposition_l352_352187

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l352_352187


namespace smaller_circle_radius_eq_l352_352267

variables (A₁ A₂ : ℝ) (r' : ℝ)

-- The conditions given in the problem
def larger_circle_area := π * 5^2 = A₁ + 2 * A₂
def arithmetic_progression := 2 * A₂ = 2 * A₁ + A₂

-- The target statement translated to Lean
theorem smaller_circle_radius_eq :
  larger_circle_area A₁ A₂ ∧ arithmetic_progression A₁ A₂ →
  r' = 5 * Real.sqrt 3 / 3 :=
sorry

end smaller_circle_radius_eq_l352_352267


namespace range_of_a_l352_352815

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a + (2 * x - 1) * Real.exp x

def F (a : ℝ) (x : ℝ) : ℝ :=
  f a x - a * x

theorem range_of_a (a : ℝ) : a < 1 ∧ ∃ x : ℤ, F a (x : ℝ) < 0 ∧ ∀ y : ℤ, y ≠ x → F a (y : ℝ) ≥ 0 ↔
  a ∈ Set.Ico (3 / (2 * Real.exp 1)) 1 := sorry

end range_of_a_l352_352815


namespace distinct_points_of_intersection_l352_352375

theorem distinct_points_of_intersection : 
  ∀ x y : ℝ, 
  (x^2 + 9*y^2 = 9) ∧ (9*x^2 + y^2 = 9) → 
  (card {p : ℝ × ℝ | (p.1^2 + 9*p.2^2 = 9) ∧ (9*p.1^2 + p.2^2 = 9)} = 4) :=
by
  sorry

end distinct_points_of_intersection_l352_352375


namespace max_distance_on_ellipse_l352_352448

-- Define the set of points on the ellipse
def is_on_ellipse (P : ℝ × ℝ) : Prop :=
  (P.1 ^ 2) / 4 + P.2 ^ 2 = 1

-- Distance function 
def distance (P Q : ℝ × ℝ) : ℝ :=
  ((P.1 - Q.1)^2 + (P.2 - Q.2)^2).sqrt

theorem max_distance_on_ellipse :
  ∀ P : ℝ × ℝ, is_on_ellipse P → distance P (0, 3) ≤ 4 :=
begin
  sorry
end

end max_distance_on_ellipse_l352_352448


namespace range_of_ω_l352_352174

noncomputable def f (ω x : ℝ) : ℝ := (2 * cos (ω * x) - 1) * sin (ω * x - π / 4)

def has_six_zeros_in_interval (ω : ℝ) : Prop :=
  ∃ (x : ℝ) (k : ℤ), 0 < x ∧ x < 3 * π ∧
  ((ω * x = π / 3 + 2 * k * π ∨ ω * x = -π / 3 + 2 * k * π) ∨ (ω * x = π / 4 + k * π))

theorem range_of_ω (ω : ℝ) (hω : ω > 0) (h6z : has_six_zeros_in_interval ω) : 
  (7 / 9) < ω ∧ ω ≤ 13 / 12 :=
sorry

end range_of_ω_l352_352174


namespace sqrt_fraction_value_l352_352795

-- Define the conditions
variables (a b : ℝ)
variables (h1 : a^2 - 6*a + 4 = 0)
variables (h2 : b^2 - 6*b + 4 = 0)
variables (h3 : a > b)
variables (h4 : b > 0)

-- Define the theorem
theorem sqrt_fraction_value : 
  (a^2 - 6*a + 4 = 0) → (b^2 - 6*b + 4 = 0) → (a > b) → (b > 0) → 
  (complex.abs ((complex.sqrt a - complex.sqrt b) / 
   (complex.sqrt a + complex.sqrt b)) = complex.abs (complex.sqrt 5 / 5)) := 
by sorry

end sqrt_fraction_value_l352_352795


namespace exists_P_M_and_fixed_points_l352_352430

structure ProjectiveTransformation (l : Type) :=
(map : l → l)
(h_map : ∀ {A B C A' B' C' : l}, unique (projective_mapping A B C A' B' C' := map))

variables {S : Type} {l : Type} [ProjectiveTransformation l] (A A' B B' C C' M : l)

theorem exists_P_M_and_fixed_points : 
  ∃ (P_m : l) (fixed_points : set l), 
    (unique P A B C = unique P_m A' B' C') ∧ 
    (∀ x, x ∈ fixed_points ↔ (P x = x)) :=
sorry

end exists_P_M_and_fixed_points_l352_352430


namespace compare_y1_y2_y3_l352_352025

noncomputable def y1 : ℝ := Real.logBase 0.7 0.8
noncomputable def y2 : ℝ := Real.logBase 1.1 0.9
noncomputable def y3 : ℝ := 1.1 ^ 0.9

theorem compare_y1_y2_y3 : y3 > y1 ∧ y1 > y2 := by
  -- proof steps will go here
  sorry

end compare_y1_y2_y3_l352_352025


namespace contingency_table_confidence_l352_352068

theorem contingency_table_confidence (k_squared : ℝ) (h1 : k_squared = 4.013) : 
  confidence_99 :=
  sorry

end contingency_table_confidence_l352_352068


namespace circles_intersect_l352_352017

def circle1 : (ℝ × ℝ) → ℝ := λ p, (p.1 + 1) * (p.1 + 1) + (p.2 + 1.5) * (p.2 + 1.5) - 9 / 4
def circle2 : (ℝ × ℝ) → ℝ := λ p, (p.1 + 2) * (p.1 + 2) + (p.2 + 1.5) * (p.2 + 1.5) - 17 / 4

def dist (p1 p2 : ℝ × ℝ) : ℝ := Real.sqrt ((p1.1 - p2.1) * (p1.1 - p2.1) + (p1.2 - p2.2) * (p1.2 - p2.2))
def radius1 : ℝ := 3 / 2
def radius2 : ℝ := Real.sqrt 17 / 2

theorem circles_intersect :
  let c1 := (-1, -1.5)
  let c2 := (-2, -1.5)
  let d := dist c1 c2
  d < radius1 + radius2 ∧ d > |radius1 - radius2|
:=
by {
  sorry
}

end circles_intersect_l352_352017


namespace probability_no_adjacent_same_rolls_l352_352769

theorem probability_no_adjacent_same_rolls :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6)
  let no_adjacent_same := outcomes.filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ a)
  (no_adjacent_same.card : ℚ) / outcomes.card = 25 / 108 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352769


namespace max_gcd_13n_plus_4_8n_plus_3_l352_352713

theorem max_gcd_13n_plus_4_8n_plus_3 (n : ℕ) (hn : n > 0) : 
  ∃ k : ℕ, k = 9 ∧ gcd (13 * n + 4) (8 * n + 3) = k := 
sorry

end max_gcd_13n_plus_4_8n_plus_3_l352_352713


namespace nell_gave_28_cards_l352_352136

theorem nell_gave_28_cards (original_cards : ℕ) (remaining_cards : ℕ) : 
  original_cards = 304 → remaining_cards = 276 → original_cards - remaining_cards = 28 :=
by
  intros h1 h2
  rw [h1, h2]
  sorry

end nell_gave_28_cards_l352_352136


namespace greatest_possible_value_of_y_l352_352158

theorem greatest_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 :=
sorry

end greatest_possible_value_of_y_l352_352158


namespace total_amount_contribution_l352_352946

theorem total_amount_contribution : 
  let r := 285
  let s := 35
  let a := 30
  let d := a / 2
  let c := 35
  r + s + a + d + c = 400 :=
by
  sorry

end total_amount_contribution_l352_352946


namespace distance_van_covers_l352_352294

-- Conditions stated in the problem
def time_initial : ℝ := 5 -- initial time in hours
def speed_new : ℝ := 80 -- new speed in kph
def factor : ℝ := 3 / 2 -- time factor

-- Prove that the distance D the van covers is 600 km
theorem distance_van_covers : 
  let time_new := factor * time_initial in
  let distance := speed_new * time_new in
  distance = 600 :=
sorry

end distance_van_covers_l352_352294


namespace probability_no_adjacent_same_rolls_l352_352771

theorem probability_no_adjacent_same_rolls :
  let outcomes := (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6).product (finset.range 6)
  let no_adjacent_same := outcomes.filter (λ ⟨⟨⟨⟨a, b⟩, c⟩, d⟩, e⟩, a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ a)
  (no_adjacent_same.card : ℚ) / outcomes.card = 25 / 108 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352771


namespace tan_of_given_trig_identity_l352_352059

theorem tan_of_given_trig_identity :
  (∀ α : ℝ, 2 * (cos((π / 2) - α)) - sin((3 / 2) * π + α) = -√5 -> tan(α) = 2) :=
begin
  sorry
end

end tan_of_given_trig_identity_l352_352059


namespace joey_route_length_l352_352094

-- Definitions
def time_one_way : ℝ := 1
def avg_speed : ℝ := 8
def return_speed : ℝ := 12

-- Theorem to prove
theorem joey_route_length : (∃ D : ℝ, D = 6 ∧ (D / avg_speed = time_one_way + D / return_speed)) :=
sorry

end joey_route_length_l352_352094


namespace circumcenter_incenter_orthocenter_eq_l352_352303

theorem circumcenter_incenter_orthocenter_eq
  {A B C O I H : Type*}
  [circumcircle_of_triangle_triangle ∆ABC₀O]
  [angle {A B C} = 60]
  [incenter_of_triangle ∆ABC I]
  [orthocenter_of_triangle ∆ABC H] :
  distance O I = distance H I :=
sorry

end circumcenter_incenter_orthocenter_eq_l352_352303


namespace hyperbola_eccentricity_correct_l352_352918

open Real

noncomputable def hyperbolaEccentricity (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) : ℝ := 
  if ((a² * b² / (b² - 3 * a²) = (a² + b²) / 4) ∧ (b² = (a² + b²) - a²)) then 
    sqrt (3) + 1 
  else 
    0 -- Fallback case, unreachable given the conditions

-- Statement of the problem
theorem hyperbola_eccentricity_correct (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) :
  hyperbolaEccentricity a b h₁ h₂ = sqrt (3) + 1 := sorry

end hyperbola_eccentricity_correct_l352_352918


namespace binom_12_10_eq_66_l352_352352

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l352_352352


namespace range_of_a_l352_352531

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (1 < x ∧ x < 4) ∧ (2 * x^2 - 9 * x + a < 0)) ↔ (a < 4) :=
by
  sorry

end range_of_a_l352_352531


namespace exists_x_in_interval_l352_352618

theorem exists_x_in_interval (a : ℝ) :
  (∃ x ∈ set.Icc 1 3, x^2 - 2 * x - a ≥ 0) ↔ a ≤ 3 :=
by
  sorry

end exists_x_in_interval_l352_352618


namespace probability_no_adjacent_same_rolls_l352_352761

theorem probability_no_adjacent_same_rolls : 
  let A := [0, 1, 2, 3, 4, 5] -- Representing six faces of a die
  let rollings : List (A → ℕ) -- Each person rolls and the result is represented as a map from faces to counts (a distribution in effect)
  ∃ rollings : List (A → ℕ), 
    (∀ (i : Fin 5), rollings[i] ≠ rollings[(i + 1) % 5]) →
      probability rollings
    = 375 / 2592 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352761


namespace hyperbola_eccentricity_range_l352_352464

theorem hyperbola_eccentricity_range (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  (∃ P₁ P₂ : { p : ℝ × ℝ // p ≠ (0, b) ∧ p ≠ (c, 0) ∧ ((0, b) - p).1 * ((c, 0) - p).1 + ((0, b) - p).2 * ((c, 0) - p).2 = 0},
   true) -- This encodes the existence of the required points P₁ and P₂ on line segment BF excluding endpoints
  → 1 < (Real.sqrt ((a^2 + b^2) / a^2)) ∧ (Real.sqrt ((a^2 + b^2) / a^2)) < (Real.sqrt 5 + 1)/2 :=
sorry

end hyperbola_eccentricity_range_l352_352464


namespace problem_l352_352800

noncomputable def f : ℝ → ℝ := sorry

theorem problem (x : ℝ) (h : ∀ x : ℝ, f (4 * x) = 4) : f (2 * x) = 4 :=
by
  sorry

end problem_l352_352800


namespace find_chocolate_chips_l352_352316

-- Definitions based on the problem conditions
def total_cookies := 48
def avg_chocolate_pieces_per_cookie := 3
def total_chocolate_pieces := total_cookies * avg_chocolate_ppieces_per_cookie

-- Given: M&Ms are one-third the number of chocolate chips
def num_mms (C : ℕ) := C / 3

-- Goal: Prove that the number of chocolate chips is 108
theorem find_chocolate_chips (C : ℕ) (h : total_chocolate_pieces = C + num_mms C) : C = 108 :=
 by
  -- The proof is omitted
  sorry

end find_chocolate_chips_l352_352316


namespace proof_problem_l352_352441

noncomputable def problem_conditions (a b : ℝ) (a_gt_b : a > b) (b_gt_0 : b > 0) (E : set (ℝ × ℝ)) : Prop :=
  E = {p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / a^2 + y^2 / b^2 = 1)}

noncomputable def is_arithmetic_seq (AF2 AB BF2 : ℝ) : Prop :=
  2 * AB = AF2 + BF2

theorem proof_problem :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧
  (E : set (ℝ × ℝ)) ∧
  (problem_conditions a b (E := E)) ∧
  ∀ (l : line) (F1 F2 : point) (A B : point),
  l.slope = 1 → 
  l.passes_through F1 → 
  E.contains A ∧ E.contains B →
  F1 = (-sqrt(a^2 - b^2), 0) ∧ F2 = (sqrt(a^2 - b^2), 0) →
  let AF2 := dist A F2 in let AB := dist A B in let BF2 := dist B F2 in
  is_arithmetic_seq AF2 AB BF2 →
  let P := (0, -1) in
  dist P A = dist P B →
  eccentricity = sqrt(2) / 2 ∧
  E = {p | ∃ x y : ℝ, p = (x, y) ∧ (x^2 / 18 + y^2 / 9 = 1)} :=
sorry

end proof_problem_l352_352441


namespace no_common_point_l352_352428

-- Define the conditions
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the condition that no circle's center lies within any of the other circles
def centers_disjoint (circles : Vector Circle 6) : Prop :=
  ∀ (i j : Fin 6), i ≠ j → ((circles[i].center - circles[j].center).abs > circles[i].radius) × 
                                    ((circles[i].center - circles[j].center).abs > circles[j].radius)

-- Define the problem statement
theorem no_common_point (circles : Vector Circle 6) (h : centers_disjoint circles) : 
  ¬ ∃ P : ℝ × ℝ, (∀ i : Fin 6, (circles[i].center - P).abs ≤ circles[i].radius) :=
sorry

end no_common_point_l352_352428


namespace sum_first_100_terms_l352_352084

def sequence (a : ℕ → ℕ) : Prop :=
  a 1 = 1 ∧
  a 2 = 2 ∧
  (∀ n : ℕ, 0 < n → a (n + 2) - a n = 1 + (-1)^(n : ℤ))

def S (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (Finset.range n).sum a

theorem sum_first_100_terms (a : ℕ → ℕ) (S : (ℕ → ℕ) → ℕ → ℕ ) :
  sequence a → S a 100 = 2600 :=
by
  intros
  sorry

end sum_first_100_terms_l352_352084


namespace regular_pyramid_volume_l352_352408

theorem regular_pyramid_volume (a : ℝ) : 
  let V := (a^3 * (5 + Real.sqrt 5)) / 24
  in V = (a^3 * (5 + Real.sqrt 5)) / 24 := 
by 
  sorry

end regular_pyramid_volume_l352_352408


namespace square_diagonal_cut_l352_352286

/--
Given a square with side length 10,
prove that cutting along the diagonal results in two 
right-angled isosceles triangles with dimensions 10, 10, 10*sqrt(2).
-/
theorem square_diagonal_cut (side_length : ℕ) (triangle_side1 triangle_side2 hypotenuse : ℝ) 
  (h_side : side_length = 10)
  (h_triangle_side1 : triangle_side1 = 10) 
  (h_triangle_side2 : triangle_side2 = 10)
  (h_hypotenuse : hypotenuse = 10 * Real.sqrt 2) : 
  triangle_side1 = side_length ∧ triangle_side2 = side_length ∧ hypotenuse = side_length * Real.sqrt 2 :=
by
  sorry

end square_diagonal_cut_l352_352286


namespace series_sum_l352_352311

theorem series_sum :
  (∑ k in Finset.range 2014, 1 / (k + 1) / (k + 2)) = 2014 / 2015 := sorry

end series_sum_l352_352311


namespace ratio_of_inscribed_circle_areas_l352_352651

theorem ratio_of_inscribed_circle_areas (s : ℝ) (hs : s > 0) : 
  let r1 := s / (2 * √3) in
  let r2 := s / 2 in
  let A1 := π * r1^2 in
  let A2 := π * r2^2 in
  (A1 / A2) = (1 / 3) :=
by
  sorry

end ratio_of_inscribed_circle_areas_l352_352651


namespace scientific_notation_l352_352088

def a : ℝ := 0.00000135
def b : ℝ := 1.35 * 10^(-6)

theorem scientific_notation : a = b := 
by 
  -- This demonstrates that they are mathematically equivalent.
  sorry

end scientific_notation_l352_352088


namespace hyperbola_eccentricity_l352_352980

noncomputable def eccentricity_of_hyperbola : ℝ :=
  2 * real.sqrt 3 / 3

theorem hyperbola_eccentricity (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : ∀ x y, y = x ^ 2 / 8 → (x, y) = (0, 2)) 
    (h4 : (2 * b * real.sqrt (4 / a ^ 2 - 1)) = 2 * real.sqrt 3 / 3) 
    (h5 : a ^ 2 + b ^ 2 = 4) : 
    eccentricity_of_hyperbola = 2 * real.sqrt 3 / 3 :=
by
  sorry

end hyperbola_eccentricity_l352_352980


namespace constant_S15_l352_352994

section ArithmeticSequence

variable {α : Type*} [CommRing α]

-- Definitions
def a (n : ℕ) : α := a₀ + (n - 1) * d -- General term of the arithmetic sequence

def S (n : ℕ) : α := n * (2 * a₀ + (n-1) * d) / 2 -- Sum of first n terms of the arithmetic sequence

-- Given condition
axiom h : a 2 + a 6 + a 16 = constant

-- Proof goal
theorem constant_S15 :
  (S 15) = 15 * a 8 :=
by sorry

end ArithmeticSequence

end constant_S15_l352_352994


namespace diet_sodas_sold_l352_352278

theorem diet_sodas_sold (R D : ℕ) (h1 : R + D = 64) (h2 : R / D = 9 / 7) : D = 28 := 
by
  sorry

end diet_sodas_sold_l352_352278


namespace lorie_total_bills_l352_352108

-- Definitions for the conditions
def initial_hundred_bills := 2
def hundred_to_fifty (bills : Nat) : Nat := bills * 2 / 100
def hundred_to_ten (bills : Nat) : Nat := (bills / 2) / 10
def hundred_to_five (bills : Nat) : Nat := (bills / 2) / 5

-- Statement of the problem
theorem lorie_total_bills : 
  let fifty_bills := hundred_to_fifty 100
  let ten_bills := hundred_to_ten 100
  let five_bills := hundred_to_five 100
  fifty_bills + ten_bills + five_bills = 2 + 5 + 10 :=
sorry

end lorie_total_bills_l352_352108


namespace calculate_tip_l352_352152

def pizzaPrice : Nat := 10
def toppingPrice : Nat := 1

def pizzasOrdered : Nat := 3
def totalToppings : Nat := 1 + 1 + 2

def totalCostWithoutTip : Nat := (pizzasOrdered * pizzaPrice) + (totalToppings * toppingPrice)
def totalOrderCost : Nat := 39

theorem calculate_tip : totalOrderCost - totalCostWithoutTip = 5 := 
by
  -- calculation
  have pizzaCost := pizzasOrdered * pizzaPrice
  have toppingCost := totalToppings * toppingPrice
  have totalCost := pizzaCost + toppingCost
  
  rw [Nat.add_comm, Nat.add_sub_assoc, totalCostWithoutTip, totalOrderCost]
  
  sorry

end calculate_tip_l352_352152


namespace ellipse_major_axis_eccentricity_exists_line_l_for_conditions_l352_352812

-- Part (I)
theorem ellipse_major_axis_eccentricity :
  (∃ a b c : ℝ, a = 2 * real.sqrt 2 ∧ b = 2 ∧ c = 2 ∧ 2 * a = 4 * real.sqrt 2 ∧ c / a = real.sqrt 2 / 2) :=
begin
  let E : ℝ → ℝ → Prop := λ x y, x^2 + 2 * y^2 = 8,
  existsi [2 * real.sqrt 2, 2, 2],
  split, refl, split, refl, split, refl, split,
  { sorry }, -- proof that 2 * (2 * real.sqrt 2) = 4 * real.sqrt 2
  { sorry }  -- proof that 2 / (2 * real.sqrt 2) = real.sqrt 2 / 2
end

-- Part (II)
theorem exists_line_l_for_conditions :
  ∃ l : ℝ → ℝ, (∃ k m : ℝ, (k = real.sqrt 2 / 2 ∨ k = -real.sqrt 2 / 2) ∧ (m = 2 * real.sqrt 5 / 5 ∨ m = -2 * real.sqrt 5 / 5) ∧ l x = k * x + m) :=
begin
  let E : ℝ → ℝ → Prop := λ x y, x^2 + 2 * y^2 = 8,
  existsi (λ x : ℝ, (real.sqrt 2 / 2) * x + 2 * real.sqrt 5 / 5),
  split,
  { existsi [real.sqrt 2 / 2, 2 * real.sqrt 5 / 5],
    split, left, refl,
    split, left, refl,
    sorry -- complete the proof if necessary
  }
end

end ellipse_major_axis_eccentricity_exists_line_l_for_conditions_l352_352812


namespace inequality_my_problem_l352_352778

theorem inequality_my_problem (a b c : ℝ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_sum : a * b + b * c + c * a = 1) :
  (Real.sqrt ((1 / a) + 6 * b)) + (Real.sqrt ((1 / b) + 6 * c)) + (Real.sqrt ((1 / c) + 6 * a)) ≤ (1 / (a * b * c)) :=
  sorry

end inequality_my_problem_l352_352778


namespace gcd_187_119_base5_l352_352312

theorem gcd_187_119_base5 :
  ∃ b : Nat, Nat.gcd 187 119 = 17 ∧ 17 = 3 * 5 + 2 ∧ 3 = 0 * 5 + 3 ∧ b = 3 * 10 + 2 := by
  sorry

end gcd_187_119_base5_l352_352312


namespace minimum_lines_intersecting_at_200_points_l352_352232

theorem minimum_lines_intersecting_at_200_points :
  ∃ m : ℕ, (m * (m - 1)) / 2 = 200 ∧ m = 21 := 
by
  existsi 21
  split
  {
    norm_num
  }

  {
    refl
  }

  sorry

end minimum_lines_intersecting_at_200_points_l352_352232


namespace count_pure_repeating_decimals_l352_352367

-- Definition of the range and condition
def is_finite_decimal (d : ℕ) : Prop := ∃ a b : ℕ, d = 2 ^ a * 5 ^ b

def num_pure_repeating_decimals : ℕ :=
  let total := 2004 in
  let num_finite := (Nat.floor (2005 / 2) + Nat.floor (2005 / 5) - Nat.floor (2005 / 10)) in
  total - num_finite

-- The theorem statement
theorem count_pure_repeating_decimals : num_pure_repeating_decimals = 801 := by
  sorry

end count_pure_repeating_decimals_l352_352367


namespace no_adjacent_same_roll_probability_l352_352765

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l352_352765


namespace number_of_even_three_digit_numbers_l352_352952

theorem number_of_even_three_digit_numbers : 
  ∃ (count : ℕ), 
  count = 12 ∧ 
  (∀ (d1 d2 : ℕ), (0 ≤ d1 ∧ d1 ≤ 4) ∧ (Even d1) ∧ (0 ≤ d2 ∧ d2 ≤ 4) ∧ (Even d2) ∧ d1 ≠ d2 →
   ∃ (d3 : ℕ), (d3 = 1 ∨ d3 = 3) ∧ 
   ∃ (units tens hundreds : ℕ), 
     (units ∈ [0, 2, 4]) ∧ 
     (tens ∈ [0, 2, 4]) ∧ 
     (hundreds ∈ [1, 3]) ∧ 
     (units ≠ tens) ∧ 
     (units ≠ hundreds) ∧ 
     (tens ≠ hundreds) ∧ 
     ((units + tens * 10 + hundreds * 100) % 2 = 0) ∧ 
     count = 12) :=
sorry

end number_of_even_three_digit_numbers_l352_352952


namespace count_pairs_l352_352429

-- Define the conditions:
variables (m n k : ℕ)
variables (a : list ℕ) (b : list ℕ)

-- Ensure the lengths of a and b match m and n respectively
axiom ha : a.length = m
axiom hb : b.length = n

-- Define r and s
def r : ℕ :=
if k < m then
  (m - k) * (m - k + 1) / 2
else
  0

def s : ℕ :=
if k < n - m then
  m * (2 * n - 2 * k - m + 1) / 2
else if k < n then
  (n - k) * (n - k + 1) / 2
else
  0

-- Main theorem statement
theorem count_pairs (h : m <= n) : 
  ∑ i in finset.range m, ∑ j in finset.range n, if |i - j| >= k then 1 else 0 = r m k + s m n k :=
sorry

end count_pairs_l352_352429


namespace sum_S_l352_352063

noncomputable def i : ℂ := Complex.I  -- The imaginary unit as defined in Lean

-- Define the sequence sum S
def S (n : ℕ) : ℂ :=
  ∑ k in Finset.range (n + 1), (k + 1) * i ^ k

theorem sum_S (n : ℕ) (h : 4 ∣ n) : S n = (n + 2 - n * i) / 2 := by
  sorry

end sum_S_l352_352063


namespace gcd_binomial_coeffs_eq_one_l352_352103

theorem gcd_binomial_coeffs_eq_one (n k : ℕ) (hn : n > k) (hk : k > 0) :
  Nat.gcd_list (List.of_fn (λ i : ℕ => Nat.choose (n + i) k) (k + 1)) = 1 :=
sorry

end gcd_binomial_coeffs_eq_one_l352_352103


namespace complete_circle_t_l352_352610

theorem complete_circle_t (t : ℝ) : 
    (∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ t → ∃ (x y : ℝ), x = r * cos θ ∧ y = r * sin θ ∧ r = sin θ)
    → t = 2 * Real.pi := 
by
  sorry

end complete_circle_t_l352_352610


namespace complement_intersection_l352_352999

open Set

variable {R : Type} [OrderedField R]

def M : Set R := { x | -2 ≤ x ∧ x ≤ 2 }
def N : Set R := { x | x < 1 }

theorem complement_intersection (x : R) : (x ∉ M) ∧ (x < 1) ↔ x < -2 := by
  sorry

end complement_intersection_l352_352999


namespace product_of_numerator_and_denominator_of_periodic_decimal_l352_352649

theorem product_of_numerator_and_denominator_of_periodic_decimal : 
  ∀ (x : ℚ), x = 0.\overline{012} → (∃ (a b : ℤ), a/b = x ∧ (∀ (g : ℤ), (g ∣ a ∧ g ∣ b) → g = 1) → a * b = 1332) :=
by
  sorry

end product_of_numerator_and_denominator_of_periodic_decimal_l352_352649


namespace find_ab_consecutive_integers_l352_352750

theorem find_ab_consecutive_integers:
  ∃ (a b : ℤ), ∀ n : ℤ, (∃ k : ℤ, a = k ∧ b = 1) ∨ (∃ k : ℤ, (a = 11 * k + 1 ∨ a = 11 * k - 1) ∧ b = 11) ↔
  ∀ n : ℤ, P(n) = (n^5 + a) / b ∈ ℤ ∧ P(n+1) = (n^5 + 1 + a) / b ∈ ℤ ∧ P(n+2) = (n^5 + 2 + a) / b ∈ ℤ :=
sorry

end find_ab_consecutive_integers_l352_352750


namespace least_sum_of_exponents_of_powers_of_2_l352_352487

theorem least_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ s : Finset ℕ, (∑ x in s, 2^x = n) ∧ (∀ t : Finset ℕ, (∑ x in t, 2^x = n) → s.sum id ≤ t.sum id) :=
sorry

end least_sum_of_exponents_of_powers_of_2_l352_352487


namespace sum_of_x_l352_352415

open Int

/-- Define the fractional part of x as x - floor x. -/
def frac (x : ℝ) : ℝ :=
  x - floor x

/-- Define the condition that the fractional part of x equals (1/5) * x. -/
def condition (x : ℝ) : Prop :=
  frac x = (1 / 5 : ℝ) * x

/-- The sum of all real numbers x for which the fractional part of x equals (1/5) * x
is 15/2. -/
theorem sum_of_x : (∑ x in {x : ℝ | condition x}, x) = 15 / 2 :=
by
  sorry

end sum_of_x_l352_352415


namespace find_equation_and_area_l352_352447

-- Given conditions as definitions in Lean
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y - 2 = 0
def line2 (x y : ℝ) : Prop := 2 * x + y + 2 = 0
def line3 (x y : ℝ) : Prop := x - 2 * y - 1 = 0
def line4 (x y : ℝ) : Prop := 2 * x + y + 2 = 0

-- Intersection point of line1 and line2
def P := (-2 : ℝ, 2 : ℝ)

-- Perpendicular condition: line4 is perpendicular to line3
def is_perpendicular (m1 m2 : ℝ) : Prop := m1 * m2 = -1
def slope_line3 : ℝ := 1 / 2
def slope_line4 : ℝ := -2

-- Triangle area calculation
def triangle_area (base height : ℝ) : ℝ := (1 / 2) * base * height

-- Prove the equation of line l and the area of the triangle
theorem find_equation_and_area :
  (∀ x y : ℝ, line4 x y ↔ 2 * x + y + 2 = 0) ∧
  (triangle_area (5 / 3) 2 = 5 / 3) :=
by
  sorry

end find_equation_and_area_l352_352447


namespace planting_equation_correct_l352_352215

-- Definitions based on given conditions
variable (x : ℝ) (x > 0)

-- Our given conditions translated into Lean statements
def total_trees := 480
def increased_rate := 𝔸 : x
def original_rate := (3/4) * x
def time_saved := 4

-- The statement we need to prove
theorem planting_equation_correct :
  (total_trees / original_rate) - (total_trees / increased_rate) = time_saved := by
  sorry

end planting_equation_correct_l352_352215


namespace hexagon_coloring_l352_352784

-- Definitions
def Hexagon (A B C D E : Type) := 
  (B ≠ D) ∧  -- B and D must have different colors
  (C ≠ B) ∧ (C ≠ D) ∧   -- C must have a different color than B and D
  (E ≠ D)

-- Given conditions
variables {A B C D E : Type}
axiom A_eq_red : ¬ (A = B) ∧ ¬ (A = D)
axiom B_adj : ¬ (B = A) ∧ ¬ (B = C) ∧ ¬ (B = D)
axiom C_adj : ¬ (C = B) ∧ ¬ (C = D)
axiom D_adj : ¬ (D = A) ∧ ¬ (D = B) ∧ ¬ (D = C) ∧ ¬ (D = E)
axiom E_adj : ¬ (E = D)

-- Proof problem statement
theorem hexagon_coloring (A B C D E : Type) 
  (h1: ¬ (A = B))
  (h2: ¬ (A = D))
  (h3: ¬ (B = D))
  (h4: ¬ (B = A))
  (h5: ¬ (B = C))
  (h6: ¬ (B = D))
  (h7: ¬ (C = B))
  (h8: ¬ (C = D))
  (h9: ¬ (D = A))
  (h10: ¬ (D = B))
  (h11: ¬ (D = C))
  (h12: ¬ (D = E))
  (h13: ¬ (E = D)) :
  ∃ (B' C' D' E' : Type), Hexagon B' C' D' E' ∧ 
    (B' ≠ D') ∧ (C' ≠ B') ∧ (C' ≠ D') ∧ (E' ≠ D') ∧ 
    (¬ (A = B') ∧ ¬ (A = D')) ∧ 
    (¬ (B' = A) ∧ ¬ (B' = C') ∧ ¬ (B' = D')) ∧ 
    (¬ (C' = B') ∧ ¬ (C' = D')) ∧ 
    (¬ (D' = A) ∧ ¬ (D' = B') ∧ ¬ (D' = C') ∧ ¬ (D' = E')) ∧ 
    (¬ (E' = D')) ∧ (finite {yellow, green}) :=
sorry

end hexagon_coloring_l352_352784


namespace matrix_A_to_power_4_l352_352331

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l352_352331


namespace groups_count_l352_352004

theorem groups_count (officers jawans: ℕ) (h_officers: officers = 10) (h_jawans: jawans = 15) :
    (choose 10 1) * (choose 9 2) * (choose 15 5) = 1081080 := by
  rw [h_officers, h_jawans]
  norm_num
  sorry

end groups_count_l352_352004


namespace fraction_equality_l352_352058

-- Defining the hypotheses and the goal
theorem fraction_equality (x y z a b c : ℝ)
  (h1 : x / a + y / b + z / c = 4)
  (h2 : a / x + b / y + c / z = 3) :
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 10 :=
by
  sorry

end fraction_equality_l352_352058


namespace find_red_balls_l352_352296

noncomputable def num_red_balls (total : ℕ) (num_blue : ℕ) (num_orange : ℕ) (pink_multiple: ℕ) d : ℕ :=
  total - (num_blue + num_orange + pink_multiple * num_orange)

theorem find_red_balls :
  ∀ (total num_blue num_orange pink_multiple : ℕ), 
  total = 50 → 
  num_blue = 10 → 
  num_orange = 5 → 
  pink_multiple = 3 → 
  num_red_balls total num_blue num_orange pink_multiple = 20 :=
by
  intros total num_blue num_orange pink_multiple h_total h_blue h_orange h_pink
  simp [num_red_balls, h_total, h_blue, h_orange, h_pink]
  rfl

end find_red_balls_l352_352296


namespace probability_multiple_4_or_15_l352_352242

theorem probability_multiple_4_or_15 (n : ℕ) (h1 : 1 ≤ n ∧ n ≤ 20) :
  (∃ k: ℕ, n = 4 * k ∨ n = 15 * k) → (∑ i in finset.range 21, if (i % 4 = 0 ∨ i % 15 = 0) then 1 else 0) = 6 →
  @probability (finset.range 21) (λ n, n % 4 = 0 ∨ n % 15 = 0) = 3 / 10 :=
begin
  have favorable_outcomes := finset.filter (λ n, n % 4 = 0 ∨ n % 15 = 0) (finset.range 21),
  have total_outcomes := finset.range 21,
  
  have num_favorable := finset.card favorable_outcomes,
  have num_total := finset.card total_outcomes,
  
  have num_favorable_correct : num_favorable = 6, sorry,
  have num_total_correct : num_total = 20, sorry,
  
  have probability := (num_favorable : ℚ) / num_total,
  
  show probability = 3 / 10, 
  { rw [num_favorable_correct, num_total_correct], norm_num }
end

end probability_multiple_4_or_15_l352_352242


namespace problem_217_values_of_n_l352_352002

def floor_sqrt (x : ℕ) : ℕ := Nat.sqrt x

def A_region (n : ℕ) : ℚ :=
  ∑ k in Finset.range (n + 1), if k^2 ≤ k ∧ k < (k + 1)^2 then 
    (k^3 + k * (k + 1)^2) / 2 * (2 * k + 1) 
  else 
    ((k * ((k^2 : ℚ) + n) * (n - k^2)) / 2)

theorem problem_217_values_of_n :
  ∃ S : Finset ℕ, (∀ n ∈ S, 2 ≤ n ∧ n ≤ 500 ∧ Even n ∧ (A_region n) % 4 = 0) ∧ S.card = 217 :=
by sorry

end problem_217_values_of_n_l352_352002


namespace union_of_A_B_l352_352830

open Set

variable {α : Type} [DecidableEq α]

def A : Set α := {1, 2, 3, 4}
def B (m : α) : Set α := {m, 4, 7}

theorem union_of_A_B (m : α) (h_inter : A ∩ B m = {1, 4}) : A ∪ B m = {1, 2, 3, 4, 7} := 
by
  have h_m : m = 1 := 
    sorry -- this represents the proof part which we are not required to provide
  rw [h_m]
  sorry -- this represents the proof part which we are not required to provide
  -- completion should show by the end of the proof

end union_of_A_B_l352_352830


namespace count_ball_distribution_l352_352253

theorem count_ball_distribution (A B C D : ℕ) (balls : ℕ) :
  (A + B > C + D ∧ A + B + C + D = balls) → 
  (balls = 30) →
  (∃ n, n = 2600) :=
by
  intro h_ball_dist h_balls
  sorry

end count_ball_distribution_l352_352253


namespace range_of_a_l352_352173

noncomputable def f (a x : ℝ) : ℝ :=
  if x ≤ 0 then (x - a)^2 else x + 1/x + a

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, f a 0 ≤ f a x) → 0 ≤ a ∧ a ≤ 2 :=
by
  sorry

end range_of_a_l352_352173


namespace least_sum_of_exponents_520_l352_352485

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l352_352485


namespace coin_stack_arrangements_l352_352151

theorem coin_stack_arrangements : 
  let gold_coins := 5
  let silver_coins := 3
  let total_coins := gold_coins + silver_coins
  in  (total_coins == 8) →
      (nomatch condition : no_two_adjacent_coins_same_face) →
      count_distinguishable_arrangements gold_coins silver_coins no_two_adjacent_coins_same_face = 504 :=
by
  sorry

end coin_stack_arrangements_l352_352151


namespace three_numbers_diff_le_half_l352_352636

theorem three_numbers_diff_le_half (x1 x2 x3 : ℝ) (h1 : 0 < x1) (h2 : x1 < 1)
  (h3 : 0 < x2) (h4 : x2 < 1) (h5 : 0 < x3) (h6 : x3 < 1) :
  ∃ (i j : ℕ), i ≠ j ∧ |[x1, x2, x3].nth i.get_or_else 0 - [x1, x2, x3].nth j.get_or_else 0| ≤ 0.5 := 
sorry

end three_numbers_diff_le_half_l352_352636


namespace no_real_roots_of_x_plus_sqrt_l352_352955

theorem no_real_roots_of_x_plus_sqrt (x : ℝ) : x + Real.sqrt(2 * x - 5) ≠ 5 :=
by
  sorry

end no_real_roots_of_x_plus_sqrt_l352_352955


namespace hyperbola_h_k_a_b_sum_eq_l352_352878

theorem hyperbola_h_k_a_b_sum_eq :
  ∃ (h k a b : ℝ), 
  h = 0 ∧ 
  k = 0 ∧ 
  a = 4 ∧ 
  (c : ℝ) = 8 ∧ 
  c^2 = a^2 + b^2 ∧ 
  h + k + a + b = 4 + 4 * Real.sqrt 3 := by
{ sorry }

end hyperbola_h_k_a_b_sum_eq_l352_352878


namespace two_cos_sixty_eq_one_l352_352626

theorem two_cos_sixty_eq_one (h : real.cos (real.pi / 3) = 1 / 2) : 2 * real.cos (real.pi / 3) = 1 := 
by 
  sorry

end two_cos_sixty_eq_one_l352_352626


namespace probability_both_selected_l352_352243

-- Given conditions
def jamie_probability : ℚ := 2 / 3
def tom_probability : ℚ := 5 / 7

-- Statement to prove
theorem probability_both_selected :
  jamie_probability * tom_probability = 10 / 21 :=
by
  sorry

end probability_both_selected_l352_352243


namespace total_flower_petals_l352_352743

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end total_flower_petals_l352_352743


namespace exists_constant_c_l352_352912

open Nat

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def p (n : ℕ) : ℕ := sum_of_exponents_of_prime_factorization n
def q (n : ℕ) (Q : Set ℕ) : ℕ := sum_of_exponents_in_prime_factorization n of_primes_in Q

def is_special (n : ℕ) (Q : Set ℕ) : Prop :=
  (p n + p (n + 1)) % 2 = 0 ∧ (q n Q + q (n + 1) Q) % 2 = 0

theorem exists_constant_c (Q : Set ℕ) (hQ : ∀ q ∈ Q, is_prime q) :
  ∃ c > 0, ∀ (N : ℕ), N > 100 → Nat.card {n | 1 ≤ n ∧ n ≤ N ∧ is_special n Q} ≥ c * N := 
sorry

end exists_constant_c_l352_352912


namespace probability_no_two_adjacent_same_roll_l352_352759

theorem probability_no_two_adjacent_same_roll :
  let total_rolls := 6^5 in
  let valid_rolls := 875 in
  (valid_rolls : ℚ) / total_rolls = 875 / 1296 :=
by
  sorry

end probability_no_two_adjacent_same_roll_l352_352759


namespace bird_nest_scientific_notation_l352_352601

def bird_nest_area : ℝ := 258000
def scientific_notation (x : ℝ) : ℝ × ℤ := 
  let a := x / (10 ^ (floor (log10 x)))
  let n := floor (log10 x)
  (a, n)

theorem bird_nest_scientific_notation : scientific_notation bird_nest_area = (2.6, 5) :=
by
  -- Steps of proof
  -- Apply definition of scientific notation
  -- Calculate the values of 'a' and 'n' for 258000
  sorry

end bird_nest_scientific_notation_l352_352601


namespace matrix_A_to_power_4_l352_352330

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l352_352330


namespace davonte_ran_further_than_mercedes_l352_352096

-- Conditions
variable (jonathan_distance : ℝ) (mercedes_distance : ℝ) (davonte_distance : ℝ)

-- Given conditions
def jonathan_ran := jonathan_distance = 7.5
def mercedes_ran_twice_jonathan := mercedes_distance = 2 * jonathan_distance
def mercedes_and_davonte_total := mercedes_distance + davonte_distance = 32

-- Prove the distance Davonte ran farther than Mercedes is 2 kilometers
theorem davonte_ran_further_than_mercedes :
  jonathan_ran jonathan_distance ∧
  mercedes_ran_twice_jonathan jonathan_distance mercedes_distance ∧
  mercedes_and_davonte_total mercedes_distance davonte_distance →
  davonte_distance - mercedes_distance = 2 :=
by
  sorry

end davonte_ran_further_than_mercedes_l352_352096


namespace minimal_ab_l352_352733

theorem minimal_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b)
(h : 1 / (a : ℝ) + 1 / (3 * b : ℝ) = 1 / 9) : a * b = 60 :=
sorry

end minimal_ab_l352_352733


namespace frustum_volume_l352_352604

theorem frustum_volume 
  (r₁ r₂ l h : ℝ) 
  (h1 : 2 * ℵ * r₂ = 4 * (2 * ℵ * r₁))
  (h2 : l = 5)
  (h3 : ℵ * (r₁ + r₂) * 5 = 25 * ℵ)
  (r1_eq : r₁ = 1)
  (r2_eq : r₂ = 4)
  (h_eq : h = 4) : 
  (1 / 3 * ℵ * h * (r₁^2 + r₂^2 + r₁ * r₂) = 28 * ℵ) :=
begin 
  sorry
end

end frustum_volume_l352_352604


namespace tortoise_wins_l352_352515

-- Definitions and conditions
def total_distance : Nat := 3000
def hare_speed : Nat := 300
def tortoise_speed : Nat := 50

-- Hare's running pattern
def hare_run_time : Nat -> Nat 
| 0 => 0
| 1 => 1
| 2 => 3
| 3 => 6
| n => hare_run_time (n - 1) + (n + 1)

-- Proving that the tortoise reaches the finish line first
theorem tortoise_wins : 
  (total_distance / tortoise_speed <= 60) ∧ (hare_run_time 10 < total_distance / hare_speed) → 
  tortoise_speed * (total_distance / tortoise_speed) = total_distance :=
by
  -- The proof is to be completed.
  sorry

end tortoise_wins_l352_352515


namespace negation_of_forall_statement_l352_352190

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l352_352190


namespace rice_in_each_container_l352_352584

theorem rice_in_each_container 
  (total_weight : ℚ) 
  (num_containers : ℕ)
  (conversion_factor : ℚ) 
  (equal_division : total_weight = 29 / 4 ∧ num_containers = 4 ∧ conversion_factor = 16) : 
  (total_weight / num_containers) * conversion_factor = 29 := 
by 
  sorry

end rice_in_each_container_l352_352584


namespace scientific_notation_1108200_l352_352697

theorem scientific_notation_1108200 :
  (∃ a : ℝ, ∃ n : ℤ, (1 ≤ a) ∧ (a < 10) ∧ (1108200 = a * 10^n) ∧ (a = 1.1082) ∧ (n = 6)) := 
begin
  sorry
end

end scientific_notation_1108200_l352_352697


namespace corner_contains_same_color_cells_l352_352261

theorem corner_contains_same_color_cells (colors : Finset (Fin 120)) :
  ∀ (coloring : Fin 2017 × Fin 2017 → Fin 120),
  ∃ (corner : Fin 2017 × Fin 2017 → Prop), 
    (∃ cell1 cell2, corner cell1 ∧ corner cell2 ∧ coloring cell1 = coloring cell2) := 
by 
  sorry

end corner_contains_same_color_cells_l352_352261


namespace irrational_sqrt_three_abs_lt_three_l352_352144

theorem irrational_sqrt_three_abs_lt_three :
  (irrational (real.sqrt 3)) ∧ (|real.sqrt 3| < 3) :=
by
  sorry

end irrational_sqrt_three_abs_lt_three_l352_352144


namespace range_of_tangent_slope_angle_l352_352578

def f (x : ℝ) : ℝ := (1/3) * x^3 - 2 * x^2 + 3 * x
def derivative_f (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem range_of_tangent_slope_angle :
  ∀ x, x ∈ set.Ico 0 π → (∃ α ∈ set.Ico 0 π, tan α = derivative_f x → α ∈ (set.Ico 0 (π/2) ∪ set.Ico (3*π/4) π)) :=
sorry

end range_of_tangent_slope_angle_l352_352578


namespace speed_of_man_l352_352692

-- Define the relevant quantities
def train_length : ℝ := 200 -- meters
def train_speed_kmh : ℝ := 60 -- km/h
def train_speed_mps : ℝ := train_speed_kmh * (1000 / 3600) -- conversion from km/h to m/s
def time_to_pass : ℝ := 10.909090909090908 -- seconds
def relative_speed : ℝ := train_length / time_to_pass -- relative speed when passing

-- Define the hypothesis and the conclusion
theorem speed_of_man (v_mps : ℝ) (v_kmh : ℝ) :
  (v_mps = relative_speed - train_speed_mps) ∧
  (v_kmh = v_mps * (3600 / 1000)) →
  v_kmh = 6 :=
by
  sorry

end speed_of_man_l352_352692


namespace polynomial_equiv_l352_352388

theorem polynomial_equiv : 
  (x : ℂ) → (x^8 - 16 = (x^2 - 2) * (x^2 + 2) * (x^2 - 2x + 2) * (x^2 + 2x + 2)) := 
by {
  intro x,
  -- Placeholder for proof
  sorry
}

end polynomial_equiv_l352_352388


namespace increasing_interval_of_log_composed_with_parabola_l352_352455

noncomputable def f : ℝ → ℝ := λ x, Real.log x / Real.log 3

def six_x_minus_x_sq (x : ℝ) : ℝ := 6 * x - x * x

def is_increasing (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
∀ x y, x ∈ I → y ∈ I → x < y → f x < f y

theorem increasing_interval_of_log_composed_with_parabola :
  is_increasing (λ x, f (six_x_minus_x_sq x)) {x : ℝ | 0 < x ∧ x < 3} :=
sorry

end increasing_interval_of_log_composed_with_parabola_l352_352455


namespace angle_DTO_in_triangle_DOG_l352_352901

theorem angle_DTO_in_triangle_DOG (DOG : Triangle)
  (h₁ : DOG.∠DGO = DOG.∠DOG)
  (h₂ : DOG.∠GOD = 30)
  (h₃ : Bisects DOG.OT DOG.∠DOG) :
  DOG.∠DTO = 67.5 :=
sorry

end angle_DTO_in_triangle_DOG_l352_352901


namespace student_3_courses_l352_352509

def a : ℕ → ℕ → ℕ := sorry
variable (m n : ℕ)
variable (m_pos : m > 0)
variable (n_pos : n > 0)

theorem student_3_courses (H : ∑ j in finset.range n, a 3 j = 2) : 
  (finset.filter (λ j, a 3 j = 1) (finset.range n)).card = 2 :=
sorry

end student_3_courses_l352_352509


namespace cone_lateral_surface_area_l352_352993

theorem cone_lateral_surface_area
  (l h : ℝ) (hl : l = 15) (hh : h = 9) :
  let r := Real.sqrt (l^2 - h^2) in
  A = Real.pi * r * l → A = 180 * Real.pi := 
by
  sorry

end cone_lateral_surface_area_l352_352993


namespace sequence_exists_l352_352546

theorem sequence_exists : ∃ (seq : ℕ → ℤ), 
  (∀ n : ℕ, n ≤ 1997 → seq n + seq (n+1) + seq (n+2) < 0) ∧ 
  (∑ i in finset.range 2000, seq i > 0) :=
begin
  sorry
end

end sequence_exists_l352_352546


namespace remaining_numbers_l352_352966

-- Define the problem statement in Lean 4
theorem remaining_numbers (S S5 S3 : ℝ) (A3 : ℝ) 
  (h1 : S / 8 = 20) 
  (h2 : S5 / 5 = 12) 
  (h3 : S3 = S - S5) 
  (h4 : A3 = 100 / 3) : 
  S3 / A3 = 3 :=
sorry

end remaining_numbers_l352_352966


namespace problem4_l352_352146

theorem problem4 (a : ℝ) : (a-1)^2 = a^3 - 2*a + 1 ↔ a = 0 ∨ a = 1 := 
by sorry

end problem4_l352_352146


namespace find_radius_range_l352_352423

theorem find_radius_range (m : ℝ) (r : ℝ) (P : ℝ × ℝ)
  (hl1 : ∀ (x y : ℝ), l1 x y = mx + y + 2m = 0)
  (hl2 : ∀ (x y : ℝ), l2 x y = x - my + 2m = 0)
  (intersection : ∃ (x y : ℝ), l1 x y = 0 ∧ l2 x y = 0 ∧ P = (x, y))
  (circle_C : ∀ (x y : ℝ), C x y = (x - 2)^2 + (y - 4)^2 = r^2)
  (on_circle_C : C (P.fst) (P.snd))
  (r_pos : r > 0) :
    2 * real.sqrt 2 ≤ r ∧ r ≤ 4 * real.sqrt 2 :=
sorry

end find_radius_range_l352_352423


namespace g_1993_at_4_l352_352924

def g (x : ℚ) : ℚ := (2 + x) / (2 - 4 * x)

def g_n : ℕ → ℚ → ℚ
  | 0, x     => x
  | (n+1), x => g (g_n n x)

theorem g_1993_at_4 : g_n 1993 4 = 11 / 20 :=
by
  sorry

end g_1993_at_4_l352_352924


namespace monotonic_increasing_interval_l352_352985

def f (a x : ℝ) : ℝ := (a * x) / (x^2 + 1)

theorem monotonic_increasing_interval {a x : ℝ} (h : a > 0) : 
  strict_mono_incr_on (λ x => f a x) (-1 : ℝ) (1 : ℝ) :=
sorry

end monotonic_increasing_interval_l352_352985


namespace similar_triangle_shortest_side_l352_352179

theorem similar_triangle_shortest_side (a b c : ℕ) (p : ℕ) (h : a = 8 ∧ b = 10 ∧ c = 12 ∧ p = 150) :
  ∃ x : ℕ, (x = p / (a + b + c) ∧ 8 * x = 40) :=
by
  sorry

end similar_triangle_shortest_side_l352_352179


namespace truncated_cone_volume_correct_l352_352293

-- Definitions for the problem
def larger_base_radius : ℝ := 10
def smaller_base_radius : ℝ := 3
def height_truncated_cone : ℝ := 9

-- Definition for the volume calculation of the truncated cone
noncomputable def volume_truncated_cone (R r h : ℝ) : ℝ :=
  let x := (R * h) / (R - r)
  let total_height := x + h
  let volume_large_cone := (1/3) * π * (R^2) * total_height
  let volume_small_cone := (1/3) * π * (r^2) * x
  volume_large_cone - volume_small_cone

-- The theorem to prove
theorem truncated_cone_volume_correct : 
  volume_truncated_cone larger_base_radius smaller_base_radius height_truncated_cone = (8757/21) * π := 
  by sorry

end truncated_cone_volume_correct_l352_352293


namespace probability_no_adjacent_same_rolls_l352_352762

theorem probability_no_adjacent_same_rolls : 
  let A := [0, 1, 2, 3, 4, 5] -- Representing six faces of a die
  let rollings : List (A → ℕ) -- Each person rolls and the result is represented as a map from faces to counts (a distribution in effect)
  ∃ rollings : List (A → ℕ), 
    (∀ (i : Fin 5), rollings[i] ≠ rollings[(i + 1) % 5]) →
      probability rollings
    = 375 / 2592 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352762


namespace problem_1_problem_2_l352_352041

noncomputable theory
open Real

def f (a x : ℝ) : ℝ := a * x^3 + 2 * sin x - x * cos x

theorem problem_1 :
  ∀ (x : ℝ), -π / 2 < x ∧ x < π / 2 →
  0 = 0 →
  (f 0 x) is_strictly_increasing_on Ioo (-π / 2) (π / 2)
:=
begin
  intro x,
  intro hx,
  intro h_zero,
  sorry
end

theorem problem_2 (a : ℝ) (h : a > 0) :
  ∀ (x : ℝ), 0 < x ∧ x < π →
  (if a ≥ 1 / (3 * π^2) then (count_extreme_points (f a) Ioo 0 π) = 0 
   else if 0 < a ∧ a < 1 / (3 * π^2) then (count_extreme_points (f a) Ioo 0 π) = 1
   else false)
:=
begin
  intro x,
  intro hx,
  cases classical.em (a >= 1 / (3 * π^2)) with h_ge h_lt,
  { 
    have : a ≥ 1 / (3 * π^2), from h_ge,
    sorry
  },
  { 
    cases classical.em (0 < a ∧ a < 1 / (3 * π^2)) with h_cases h_other,
    { 
      have : 0 < a ∧ a < 1 / (3 * π^2), from h_cases,
      sorry 
    },
    { 
      exfalso,
      apply h_other,
      split,
      { exact h },
      { rw not_lt at h_ge,
        exact h_ge }
    }
  }
end

end problem_1_problem_2_l352_352041


namespace smallest_six_digit_number_l352_352521

theorem smallest_six_digit_number : ∃ n : ℕ, n = 112642 ∧ 
(∀ (i : ℕ), 0 ≤ i ∧ i ≤ 3 → 
  (let num := n / (10 ^ (5 - i)) % 1000 in num % 6 = 0 ∨ num % 7 = 0)) :=
by {
  use 112642,
  split,
  { refl },
  { intros i hi,
    cases i with _ | _ | _ | _,
    all_goals {
      repeat { simp [nat.div, nat.mod] },
      dec_trivial,
    },
    sorry
  }
}

end smallest_six_digit_number_l352_352521


namespace log_shifted_through_2_neg1_l352_352463

theorem log_shifted_through_2_neg1 {a : ℝ} (h₀ : 0 < a) (h₁ : a ≠ 1) :
  ∃ x y, y = log a (x - 1) - 1 ∧ (x, y) = (2, -1) :=
begin
  use 2,
  use -1,
  split,
  { rw [← eq_sub_of_add_eq, sub_sub, sub_add_cancel], sorry },
  { refl, }
end

end log_shifted_through_2_neg1_l352_352463


namespace exponential_decreasing_condition_l352_352443

theorem exponential_decreasing_condition (a : ℝ) :
  (∀ x : ℝ, (a^x * real.log a) < 0) ↔ (0 < a ∧ a < 1) :=
by sorry

end exponential_decreasing_condition_l352_352443


namespace a_5_eq_12_l352_352032

noncomputable def S (n : ℕ) : ℕ := n^2 + 3 * n

noncomputable def a (n : ℕ) : ℕ :=
  if n = 1 then S 1
  else S n - S (n - 1)

theorem a_5_eq_12 : a 5 = 12 := by
  rw [a, if_neg]
  let h := S 5 - S 4
  rw [S, S] at h
  norm_num at h
  exact h
  sorry

end a_5_eq_12_l352_352032


namespace least_sum_of_exponents_520_l352_352482

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l352_352482


namespace number_of_ways_to_put_cousins_in_rooms_l352_352116

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l352_352116


namespace binom_12_10_eq_66_l352_352341

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l352_352341


namespace top_three_teams_max_points_l352_352077

theorem top_three_teams_max_points 
  (teams : ℕ) (games_per_team : ℕ) (points_win : ℕ) (points_draw : ℕ) 
  (points_loss : ℕ) (total_games : ℕ) (total_points : ℕ) 
  (points_per_top_team : ℕ) :  
  teams = 8 → 
  games_per_team = 7 * 2 → 
  points_win = 3 → 
  points_draw = 1 → 
  points_loss = 0 → 
  total_games = 56 → 
  total_points = 168 → 
  (∀ team, team ∈ {A, B, C} → points_per_top_team = 36) := by
  intros h_teams h_games h_points_win h_points_draw h_points_loss h_total_games h_total_points
  have hAB_points := <the calculation details or initialization>
  have hAC_points := <the calculation details or initialization>
  have hBC_points := <the calculation details or initialization>
  have h_not_top_teams_points := <the calculation details or initialization>
  -- Prove that under the given conditions, the maximal points each of the top three teams can achieve is 36
  sorry

end top_three_teams_max_points_l352_352077


namespace matrix_power_four_l352_352335

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l352_352335


namespace wang_hao_not_last_l352_352533

theorem wang_hao_not_last (total_players : ℕ) (players_to_choose : ℕ) 
  (wang_hao : ℕ) (ways_to_choose_if_not_last : ℕ) : 
  total_players = 6 ∧ players_to_choose = 3 → 
  ways_to_choose_if_not_last = 100 := 
by
  sorry

end wang_hao_not_last_l352_352533


namespace sum_of_every_third_term_l352_352169

theorem sum_of_every_third_term (a : ℕ → ℤ) (n : ℕ) (d : ℤ) 
  (h1 : d = 2)
  (h2 : ∑ k in Finset.range 30, a(k) = 100) :
  ∑ k in Finset.range 10, a(3 * (k + 1)) = 160 / 3 :=
  sorry

end sum_of_every_third_term_l352_352169


namespace blue_to_red_ratio_l352_352904

-- Define the conditions as given in the problem
def initial_red_balls : ℕ := 16
def lost_red_balls : ℕ := 6
def bought_yellow_balls : ℕ := 32
def total_balls_after_events : ℕ := 74

-- Based on the conditions, we define the remaining red balls and the total balls equation
def remaining_red_balls := initial_red_balls - lost_red_balls

-- Suppose B is the number of blue balls
def blue_balls (B : ℕ) : Prop :=
  remaining_red_balls + B + bought_yellow_balls = total_balls_after_events

-- Now, state the theorem to prove the ratio of blue balls to red balls is 16:5
theorem blue_to_red_ratio (B : ℕ) (h : blue_balls B) : B = 32 → B / remaining_red_balls = 16 / 5 :=
by
  intro B_eq
  subst B_eq
  have h1 : remaining_red_balls = 10 := rfl
  have h2 : 32 / 10  = 16 / 5 := by rfl
  exact h2

-- Note: The proof itself is skipped, so the statement is left with sorry.

end blue_to_red_ratio_l352_352904


namespace matrix_power_four_l352_352332

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l352_352332


namespace inverse_of_g_is_g_l352_352551

def g (k x : ℝ) : ℝ := (3 * x + 4) / (k * x - 3)

theorem inverse_of_g_is_g (k : ℝ) :
  (∃ f : ℝ → ℝ, ∀ x, g k (f x) = x ∧ g k x = f x) ↔ k ∈ set.Ioo ⊤ (-9/4) ∪ set.Ioo (-9/4) ⊥ := 
sorry

end inverse_of_g_is_g_l352_352551


namespace expansion_sum_eq_half_l352_352608

theorem expansion_sum_eq_half (n : ℕ) : 
  2⁻¹^n * (∑ k in finset.range(n), (finset.range(k).prod (λ i, (n+i))) * (1/2)^k / k.factorial) = 1/2 :=
by 
  sorry

end expansion_sum_eq_half_l352_352608


namespace matrix_pow_four_l352_352326

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l352_352326


namespace area_of_grey_region_l352_352644

open Nat

theorem area_of_grey_region
  (a1 a2 b : ℕ)
  (h1 : a1 = 8 * 10)
  (h2 : a2 = 9 * 12)
  (hb : b = 37)
  : (a2 - (a1 - b) = 65) := by
  sorry

end area_of_grey_region_l352_352644


namespace estimate_greater_than_exact_l352_352378

namespace NasreenRounding

variables (a b c d a' b' c' d' : ℕ)

-- Conditions: a, b, c, and d are large positive integers.
-- Definitions for rounding up and down
def round_up (n : ℕ) : ℕ := n + 1  -- Simplified model for rounding up
def round_down (n : ℕ) : ℕ := n - 1  -- Simplified model for rounding down

-- Conditions: a', b', c', and d' are the rounded values of a, b, c, and d respectively.
variable (h_round_a_up : a' = round_up a)
variable (h_round_b_down : b' = round_down b)
variable (h_round_c_down : c' = round_down c)
variable (h_round_d_down : d' = round_down d)

-- Question: Show that the estimate is greater than the original
theorem estimate_greater_than_exact :
  (a' / b' - c' * d') > (a / b - c * d) :=
sorry

end NasreenRounding

end estimate_greater_than_exact_l352_352378


namespace ellipse_eccentricity_ellipse_equation_l352_352168

theorem ellipse_eccentricity (b : ℝ) (a : ℝ) (c : ℝ) 
  (h1 : a = (3/2) * b)
  (h2 : c = 2)
  (h3 : sqrt ((3/2)^2 * b^2 - b^2) = c) : 
  let e := c / a in
  e = sqrt(5) / 3 := sorry

theorem ellipse_equation (b : ℝ) (a : ℝ) 
  (h1 : a = (3/2) * b)
  (h2 : b^2 = 16 / 5) : 
  (∀ x y : ℝ, (y^2 / (36 / 5) + x^2 / (16 / 5) = 1)) := sorry

end ellipse_eccentricity_ellipse_equation_l352_352168


namespace minimum_ab_value_l352_352073

variables {A B C a b c : ℝ}
noncomputable def area_of_triangle (a b c : ℝ) : ℝ := abs (1 / 2 * c)

theorem minimum_ab_value (h1 : 2 * c * cos B = 2 * a + b)
                        (h2 : area_of_triangle a b c = 1 / 2 * c) :
  ∃ (ab_min : ℝ), ab_min = 4 ∧ 
                  ∀ (a b : ℝ), (some conditions to link to the area and sides apply here) → ab ≥ ab_min :=
sorry

end minimum_ab_value_l352_352073


namespace n_in_S_implies_n2_in_S_l352_352565

def S (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0 ∧
  a ≥ b ∧ c ≥ d ∧ e ≥ f ∧
  n - 1 = a^2 + b^2 ∧ n = c^2 + d^2 ∧ n + 1 = e^2 + f^2

theorem n_in_S_implies_n2_in_S (n : ℕ) (h : S n) : S (n^2) :=
  sorry

end n_in_S_implies_n2_in_S_l352_352565


namespace distinct_nonzero_reals_xy_six_l352_352927

theorem distinct_nonzero_reals_xy_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + 6/x = y + 6/y) (h_distinct : x ≠ y) : x * y = 6 := 
sorry

end distinct_nonzero_reals_xy_six_l352_352927


namespace total_marbles_in_bag_l352_352257

theorem total_marbles_in_bag 
  (r b p : ℕ) 
  (h1 : 32 = r)
  (h2 : b = (7 * r) / 4) 
  (h3 : p = (3 * b) / 2) 
  : r + b + p = 172 := 
sorry

end total_marbles_in_bag_l352_352257


namespace y_intercept_probability_l352_352825

theorem y_intercept_probability (b : ℝ) (hb : b ∈ Set.Icc (-2 : ℝ) 3 ) :
  (∃ P : ℚ, P = (2 / 5)) := 
by 
  sorry

end y_intercept_probability_l352_352825


namespace number_of_children_is_4_l352_352625

-- Define the conditions from the problem
def youngest_child_age : ℝ := 1.5
def sum_of_ages : ℝ := 12
def common_difference : ℝ := 1

-- Define the number of children
def n : ℕ := 4

-- Prove that the number of children is 4 given the conditions
theorem number_of_children_is_4 :
  (∃ n : ℕ, (n / 2) * (2 * youngest_child_age + (n - 1) * common_difference) = sum_of_ages) ↔ n = 4 :=
by sorry

end number_of_children_is_4_l352_352625


namespace not_all_ten_on_boundary_of_same_square_l352_352540

open Function

variable (points : Fin 10 → ℝ × ℝ)

def four_points_on_square (A B C D : ℝ × ℝ) : Prop :=
  -- Define your own predicate to check if 4 points A, B, C, D are on the boundary of some square
  sorry 

theorem not_all_ten_on_boundary_of_same_square :
  (∀ A B C D : Fin 10, four_points_on_square (points A) (points B) (points C) (points D)) →
  ¬ (∃ square : ℝ × ℝ → Prop, ∀ i : Fin 10, square (points i)) :=
by
  intro h
  sorry

end not_all_ten_on_boundary_of_same_square_l352_352540


namespace determine_m_l352_352462

open Real

-- Define the function f(x)
def f (x m : ℝ) : ℝ := x - m * sqrt(x) + 5

-- Define the condition that f(x) > 1 for all x in [1, 9]
def f_gt_one_over_interval (m : ℝ) := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 9 → f x m > 1

-- State the theorem
theorem determine_m (m : ℝ) (h : f_gt_one_over_interval m) : m < 4 := 
sorry

end determine_m_l352_352462


namespace area_of_triangle_l352_352219

variables (A B C K : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space K]
variables (AC BC BK AK : ℝ)

def is_altitude (A B C K : Type) (AK : ℝ) : Prop :=
  ∃ h : K ∈ line_segment C B, dist A K = AK ∧ ∠ AKC = 90

axiom conditions :
  ∀ (A B C K : Type), is_altitude A B C K AK ∧ dist AC 10 ∧ dist BC 13 ∧ dist BK 7

theorem area_of_triangle (A B C K : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space K] (h : AC = 10) (h1 : BC = 13) (h2 : BK = 7) (h3: is_altitude A B C K 8) : 
  (1/2) * BC * AK = 52 :=
begin
  sorry
end

end area_of_triangle_l352_352219


namespace length_of_equal_sides_l352_352706

-- Definitions based on conditions
def isosceles_triangle (a b c : ℝ) : Prop :=
(a = b ∨ b = c ∨ a = c)

def is_triangle (a b c : ℝ) : Prop :=
(a + b > c) ∧ (b + c > a) ∧ (c + a > b)

def has_perimeter (a b c : ℝ) (P : ℝ) : Prop :=
a + b + c = P

def one_side_length (a : ℝ) : Prop :=
a = 3

-- The proof statement
theorem length_of_equal_sides (a b c : ℝ) :
isosceles_triangle a b c →
is_triangle a b c →
has_perimeter a b c 7 →
one_side_length a ∨ one_side_length b ∨ one_side_length c →
(b = 3 ∧ c = 3) ∨ (b = 2 ∧ c = 2) :=
by
  intros iso tri per side_length
  sorry

end length_of_equal_sides_l352_352706


namespace scientific_notation_15510000_l352_352256

theorem scientific_notation_15510000 : 
  ∃ (a : ℝ) (n : ℤ), (1 ≤ |a| ∧ |a| < 10) ∧ 15510000 = a * 10^n ∧ a = 1.551 ∧ n = 7 :=
by
  sorry

end scientific_notation_15510000_l352_352256


namespace probability_sum_is_18_l352_352508

noncomputable def probability_of_sum_18 : ℚ :=
  let single_die_prob : ℚ := 1 / 6 in
  (single_die_prob ^ 3)

theorem probability_sum_is_18 (d1 d2 d3 : ℕ) (h₁ : d1 ∈ {1, 2, 3, 4, 5, 6})
                                      (h₂ : d2 ∈ {1, 2, 3, 4, 5, 6})
                                      (h₃ : d3 ∈ {1, 2, 3, 4, 5, 6})
                                      (h_sum : d1 + d2 + d3 = 18) :
  probability_of_sum_18 = 1 / 216 :=
by sorry

end probability_sum_is_18_l352_352508


namespace binom_12_10_l352_352350

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l352_352350


namespace statement_a_statement_b_statement_c_statement_d_correct_statements_l352_352228

theorem statement_a (k b : ℝ) (h1 : k < 0) (h2 : b > 0) : (k, b) ∈ { p : ℝ × ℝ | p.1 < 0 ∧ p.2 > 0 } := 
by finish

theorem statement_b (k : ℝ) : ∃ p : ℝ × ℝ, p = (2, 3) ∧ k * (2 : ℝ) - (3 : ℝ) - 2 * k + 3 = 0 :=
by triv

theorem statement_c : ∀ x y, (x - 2) * (-√3) = (y + 1) → y + 1 = -√3 * (x - 2) := 
by intros; assumption

theorem statement_d : ∀ x y, y = -2 * x + 3 := 
by intros; assumption

theorem correct_statements : statement_a ∧ ¬statement_b ∧ statement_c ∧ ¬statement_d := 
by {
  repeat { try apply_and_right; try assumption; try exact statement_a; 
           try exact statement_b; try exact statement_c; try exact statement_d;
           try sorry }
}

end statement_a_statement_b_statement_c_statement_d_correct_statements_l352_352228


namespace probability_no_two_adjacent_same_roll_l352_352758

theorem probability_no_two_adjacent_same_roll :
  let total_rolls := 6^5 in
  let valid_rolls := 875 in
  (valid_rolls : ℚ) / total_rolls = 875 / 1296 :=
by
  sorry

end probability_no_two_adjacent_same_roll_l352_352758


namespace product_of_distances_l352_352811

noncomputable def ellipse : set (ℝ × ℝ) := {p | (p.1^2) / 6 + (p.2^2) / 2 = 1}
noncomputable def tangent_line (k m : ℝ) : set (ℝ × ℝ) := {p | p.2 = k * p.1 + m}
noncomputable def f1 : ℝ × ℝ := (-2, 0)
noncomputable def f2 : ℝ × ℝ := (2, 0)

noncomputable def distance_to_line (point : ℝ × ℝ) (k m : ℝ) : ℝ :=
  abs (k * point.1 - point.2 + m) / real.sqrt(1 + k^2)

theorem product_of_distances (k m : ℝ) :
  (y = kx + m) is tangent to ellipse → 
  (distance_to_line f1 k m) * (distance_to_line f2 k m) = 2 :=
sorry

end product_of_distances_l352_352811


namespace no_adjacent_same_roll_probability_l352_352767

-- We define probabilistic event on rolling a six-sided die and sitting around a circular table
noncomputable def probability_no_adjacent_same_roll : ℚ :=
  1 * (5/6) * (5/6) * (5/6) * (5/6) * (4/6)

theorem no_adjacent_same_roll_probability :
  probability_no_adjacent_same_roll = 625/1944 :=
by
  sorry

end no_adjacent_same_roll_probability_l352_352767


namespace cost_of_one_dozen_pens_l352_352605

variables (x : ℝ) (pen_cost pencil_cost dozen_pen_cost : ℝ)

-- Conditions
def cost_relationship : Prop :=
  pen_cost = 5 * pencil_cost ∧
  3 * pen_cost + 5 * pencil_cost = 150 ∧
  dozen_pen_cost = 12 * pen_cost

-- Proof goal: The cost of one dozen pens is Rs. 450.
theorem cost_of_one_dozen_pens (h : cost_relationship x pen_cost (pencil_cost x) dozen_pen_cost) :
  dozen_pen_cost = 450 :=
sorry

end cost_of_one_dozen_pens_l352_352605


namespace shadow_area_of_cube_l352_352138

theorem shadow_area_of_cube (a b : ℝ) (h1 : b > a) (h2 : a > 0) : 
  let A := (a^2 * b^2) / (b - a)^2 in
    A = (a^2 * b^2) / (b - a)^2 :=
sorry

end shadow_area_of_cube_l352_352138


namespace minoxidil_percentage_l352_352273

-- Define the conditions
variable (x : ℝ) -- percentage of Minoxidil in the solution to add
def pharmacist_scenario (x : ℝ) : Prop :=
  let amt_2_percent_solution := 70 -- 70 ml of 2% solution
  let percent_in_2_percent := 0.02
  let amt_of_2_percent := percent_in_2_percent * amt_2_percent_solution
  let amt_added_solution := 35 -- 35 ml of solution to add
  let total_volume := amt_2_percent_solution + amt_added_solution -- 105 ml in total
  let desired_percent := 0.03
  let desired_amt := desired_percent * total_volume
  amt_of_2_percent + (x / 100) * amt_added_solution = desired_amt

-- Define the proof problem statement
theorem minoxidil_percentage : pharmacist_scenario 5 := by
  -- Proof goes here
  sorry

end minoxidil_percentage_l352_352273


namespace Evelyn_bottle_caps_problem_l352_352747

theorem Evelyn_bottle_caps_problem (E : ℝ) (H1 : E - 18.0 = 45) : E = 63.0 := 
by
  sorry


end Evelyn_bottle_caps_problem_l352_352747


namespace ab_zero_l352_352613

theorem ab_zero
  (a b : ℤ)
  (h : ∀ (m n : ℕ), ∃ (k : ℤ), a * (m : ℤ) ^ 2 + b * (n : ℤ) ^ 2 = k ^ 2) :
  a * b = 0 :=
sorry

end ab_zero_l352_352613


namespace tetrahedron_angle_sum_gt_3pi_l352_352148

theorem tetrahedron_angle_sum_gt_3pi
(O A B C D : Point)
(α β γ a b c : ℝ)
(h₁ : inside_tetrahedron O A B C D)
(h₂ : seen_angle O A D = α)
(h₃ : seen_angle O B D = β)
(h₄ : seen_angle O C D = γ)
(h₅ : seen_angle O B C = a)
(h₆ : seen_angle O C A = b)
(h₇ : seen_angle O A B = c)
: α + β + γ + a + b + c > 3 * real.pi := 
sorry

end tetrahedron_angle_sum_gt_3pi_l352_352148


namespace gpa_ratio_l352_352175

theorem gpa_ratio (x y : ℕ) 
  (h1 : (\frac{(30 * x) + (33 * (y - x))}{y}) = 32) : 
  (\frac{x}{y}) = (\frac{1}{3}) :=
sorry

end gpa_ratio_l352_352175


namespace pedro_ball_reflection_intersection_l352_352721

theorem pedro_ball_reflection_intersection :
  ∀ (θ : ℝ), θ = 20 →
  let reflections := 5 in
  ∃ n : ℕ, n = reflections ∧ ball_trajectory_intersects θ n :=
by
  sorry

end pedro_ball_reflection_intersection_l352_352721


namespace original_volume_of_ice_l352_352612

theorem original_volume_of_ice (V : ℝ) : 
  (V * (1/4)^2 * (1/3) * (1/2) = 0.5) → V = 48 :=
by
  intro h
  have h1 : V * (1/4)^2 * (1/3) * (1/2) = V * (1/48)
  { field_simp, norm_num }
  rw h1 at h
  simp at h
  linarith

end original_volume_of_ice_l352_352612


namespace real_solutions_count_l352_352986

theorem real_solutions_count :
  (∀ x : ℝ, (|x^2 - 3 * x + 2| + |x^2 + 2 * x - 3| = 11) → x = -6/5 ∨ x = (1 + Real.sqrt 97) / 4) →
  (∃ x1 x2 : ℝ, (x1 = -6/5 ∧ x2 = (1 + Real.sqrt 97) / 4) ∧ ((x1 = -6/5 ∨ x1 = (1 + Real.sqrt 97) / 4) ∧ (x2 = -6/5 ∨ x2 = (1 + Real.sqrt 97) / 4) ∧ x1 ≠ x2)) :=
begin
  sorry
end

end real_solutions_count_l352_352986


namespace minimize_PR_RQ_l352_352439

-- Define the given points P and Q
def P : ℝ × ℝ := (-2, -3)
def Q : ℝ × ℝ := (3, 3)

-- Define the given value r
def r : ℝ := 2

-- Define the condition for point R such that PR + RQ is minimized
def line_PQ (x : ℝ) : ℝ :=
  (6 / 5) * x - (3 / 5)

-- Define point R with fixed x = r
def R : ℝ × ℝ := (r, line_PQ r)

-- State the theorem to be proved
theorem minimize_PR_RQ : (R.2 = 9 / 5) :=
by
  -- Placeholder for the actual proof
  sorry

end minimize_PR_RQ_l352_352439


namespace mikes_ride_is_46_miles_l352_352572

-- Define the conditions and the question in Lean 4
variable (M : ℕ)

-- Mike's cost formula
def mikes_cost (M : ℕ) : ℚ := 2.50 + 0.25 * M

-- Annie's total cost
def annies_miles : ℕ := 26
def annies_cost : ℚ := 2.50 + 5.00 + 0.25 * annies_miles

-- The proof statement
theorem mikes_ride_is_46_miles (h : mikes_cost M = annies_cost) : M = 46 :=
by sorry

end mikes_ride_is_46_miles_l352_352572


namespace angle_of_inclination_l352_352798

/-- Given that a line passes through points A(-2,0) and B(-5,3), the angle of inclination of the line is 135 degrees. -/
theorem angle_of_inclination (A B : ℝ × ℝ)
  (hA : A = (-2, 0)) (hB : B = (-5, 3)) : 
  angle_of_inclination A B = 135 := 
sorry

end angle_of_inclination_l352_352798


namespace determinant_of_matrix_A_l352_352725

def matrix_A : Matrix (Fin 2) (Fin 2) ℤ :=
  ![![4, 3], ![2, 1]]

theorem determinant_of_matrix_A : matrix.det matrix_A = -2 := by
  sorry

end determinant_of_matrix_A_l352_352725


namespace eq1_solution_eq2_solution_l352_352957

-- Theorem 1: Solution to (x+1)^3 = 64
theorem eq1_solution (x : ℝ) : (x + 1)^3 = 64 ↔ x = 3 := 
by sorry

-- Theorem 2: Solution to (2x+1)^2 = 81
theorem eq2_solution (x : ℝ) : (2x + 1)^2 = 81 ↔ (x = 4 ∨ x = -5) :=
by sorry

end eq1_solution_eq2_solution_l352_352957


namespace incenter_locus_l352_352519

-- Given central angle OAB is a right angle
variable (O A B P Q I : Point)
variable (arc_AB : Arc O A B)
variable (tangent_PQ : Tangent)
variable (PQ_intersects_OA : intersects PQ OA Q)
variable (P_on_arc_AB : OnArc P arc_AB)

-- Define a proof problem to find the locus of the incenter I of triangle OPQ
theorem incenter_locus:
  ∀ I, is_incenter I (triangle O P Q) → on_segment I (segment A B) ∧ I ≠ A ∧ I ≠ B :=
by
  sorry

end incenter_locus_l352_352519


namespace sphere_to_plane_distance_l352_352687

noncomputable def distance_between_sphere_and_plane (r : ℝ) (a b c : ℝ) (tangent_distance : ℝ) : ℝ := sorry

theorem sphere_to_plane_distance :
    distance_between_sphere_and_plane 8 17 17 16 = 6.4 := 
sorry

end sphere_to_plane_distance_l352_352687


namespace binomial_12_10_eq_66_l352_352358

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l352_352358


namespace quadratic_coefficients_l352_352082

theorem quadratic_coefficients :
  ∃ (b c : ℤ), (∀ x : ℤ, 3 * x^2 - 6 * x - 1 = 3 * x^2 + b * x + c) ∧
               b = -6 ∧
               c = -1 :=
by
  use -6, -1
  intro x
  simp
  split; refl

end quadratic_coefficients_l352_352082


namespace belfried_payroll_l352_352658

noncomputable def tax_paid (payroll : ℝ) : ℝ :=
  if payroll < 200000 then 0 else 0.002 * (payroll - 200000)

theorem belfried_payroll (payroll : ℝ) (h : tax_paid payroll = 400) : payroll = 400000 :=
by
  sorry

end belfried_payroll_l352_352658


namespace number_of_people_in_group_l352_352972

theorem number_of_people_in_group :
  ∀ (n : ℕ), (1.5 * n = 9) → n = 6 :=
begin
  intros n h,
  rw [mul_comm] at h,
  exact eq_of_mul_eq_mul_left (by norm_num : (0 : ℝ) < 1.5) h,
end

end number_of_people_in_group_l352_352972


namespace multiples_of_3_or_5_but_not_6_l352_352476

theorem multiples_of_3_or_5_but_not_6 (n : ℕ) (h1 : n ≤ 150) :
  (∃ m : ℕ, m ≤ 150 ∧ ((m % 3 = 0 ∨ m % 5 = 0) ∧ m % 6 ≠ 0)) ↔ n = 45 :=
by {
  sorry
}

end multiples_of_3_or_5_but_not_6_l352_352476


namespace probability_eventM_l352_352291

open Finset

-- Define the Asian and European countries as sets
def AsianCountries : Finset String := {"A1", "A2", "A3"}
def EuropeanCountries : Finset String := {"B1", "B2", "B3"}
def AllCountries : Finset String := AsianCountries ∪ EuropeanCountries

-- Define the event of selecting 2 countries
def event (s : Finset String) : Set (Finset String) := {t | t ⊆ s ∧ t.card = 2}

-- Define the event M where both selected countries are Asian
def eventM : Set (Finset String) := {t | t ⊆ AsianCountries ∧ t.card = 2}

-- Define the probability of an event given the sample space
def probability (sample_space : Set (Finset String)) (event : Set (Finset String)) : ℚ :=
  (event.to_finset.card : ℚ) / (sample_space.to_finset.card : ℚ)

-- The final theorem to state the required probability
theorem probability_eventM : probability (event AllCountries) eventM = 1 / 5 := by
  sorry

end probability_eventM_l352_352291


namespace line_circle_intersection_max_AB_plus_EF_l352_352045

noncomputable def l1 (k : ℝ) := (1 + 4 * k) * x - (2 - 3 * k) * y + (2 - 14 * k) = 0
def C : x^2 + y^2 - 6 * x - 8 * y + 9 = 0

theorem line_circle_intersection (k : ℝ) :
  ∀ x y, (l1 k) -> C -> True :=
sorry

theorem max_AB_plus_EF :
  ∀ k, (∃ x y, l1 k) -> (∃ x y, C) ->
  (∃ A B E F, AB + EF ≤ 6 * sqrt 6) :=
sorry

end line_circle_intersection_max_AB_plus_EF_l352_352045


namespace original_prices_l352_352154

theorem original_prices 
  (S P J : ℝ)
  (hS : 0.80 * S = 780)
  (hP : 0.70 * P = 2100)
  (hJ : 0.90 * J = 2700) :
  S = 975 ∧ P = 3000 ∧ J = 3000 :=
by
  sorry

end original_prices_l352_352154


namespace vector_sum_magnitude_l352_352836

variable (a b : EuclideanSpace ℝ (Fin 3)) -- assuming 3-dimensional Euclidean space for vectors

-- Define the conditions
def mag_a : ℝ := 5
def mag_b : ℝ := 6
def dot_prod_ab : ℝ := -6

-- Prove the required magnitude condition
theorem vector_sum_magnitude (ha : ‖a‖ = mag_a) (hb : ‖b‖ = mag_b) (hab : inner a b = dot_prod_ab) :
  ‖a + b‖ = 7 :=
by
  sorry

end vector_sum_magnitude_l352_352836


namespace fraction_comparisons_l352_352389

theorem fraction_comparisons :
  (1 / 8 : ℝ) * (3 / 7) < (1 / 8) ∧ 
  (9 / 8 : ℝ) * (1 / 5) > (9 / 8) * (1 / 8) ∧ 
  (2 / 3 : ℝ) < (2 / 3) / (6 / 11) := by
    sorry

end fraction_comparisons_l352_352389


namespace AD_correct_l352_352577

section
variables {V : Type*} [AddCommGroup V] [VectorSpace ℝ V]
variables (a b c : V) -- Given vectors

-- Define the midpoint of BC as D
def midpoint (x y : V) := (x + y) / 2

-- Define vectors OA, OB, OC
def OA := a
def OB := b
def OC := c

-- Define the vector AD for the proof
def AD := let D := midpoint b c in D - a

-- Prove that AD is equal to the given expression
theorem AD_correct :
  AD a b c = (1 / 2) • (c + b) - a :=
by
  -- the proof would normally follow here, but instead we write sorry to skip it
  sorry

end

end AD_correct_l352_352577


namespace number_of_three_digit_numbers_l352_352477

theorem number_of_three_digit_numbers : 
  ∃ n : ℕ, n = 999 - 100 + 1 ∧ n = 900 :=
by
  use 900
  split
  . rfl
  rfl

end number_of_three_digit_numbers_l352_352477


namespace range_of_t_l352_352796

theorem range_of_t (a b c : ℝ) (t : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_inequality : ∀ a b c : ℝ, 0 < a → 0 < b → 0 < c → (1 / a^2) + (4 / b^2) + (t / c^2) ≥ 0) :
  t ≥ -9 :=
sorry

end range_of_t_l352_352796


namespace sequence_pos_integers_l352_352779

noncomputable def a : ℕ → ℤ 
| 0       := 1
| 1       := 1
| (n + 2) := (a (n + 1) ^ 2 + (-1) ^ n) / a n

theorem sequence_pos_integers (n : ℕ) : a n > 0 :=
sorry

end sequence_pos_integers_l352_352779


namespace binary_1001_is_9_l352_352368

def binary_to_decimal (b : List ℕ) : ℕ :=
  b.foldr (λ (x : ℕ) (acc : ℕ) → x + 2 * acc) 0

theorem binary_1001_is_9 : binary_to_decimal [1, 0, 0, 1] = 9 :=
by
  sorry

end binary_1001_is_9_l352_352368


namespace binom_12_10_eq_66_l352_352338

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l352_352338


namespace find_k_l352_352728

def f (x : ℝ) : ℝ := 7 * x^3 - 3 / x + 5
def g (x k : ℝ) : ℝ := x^3 - k + 2 * x

theorem find_k : f 3 - g 3 k = 5 → k = -155 := 
by
  sorry

end find_k_l352_352728


namespace sequence_2007th_number_l352_352203

-- Defining the sequence according to the given rule
def a (n : ℕ) : ℕ := 2 ^ n

theorem sequence_2007th_number : a 2007 = 2 ^ 2007 :=
by
  -- Proof is omitted
  sorry

end sequence_2007th_number_l352_352203


namespace haj_grocery_store_cost_l352_352935

variables (T : ℝ)
def salary_cost : ℝ := (2 / 5) * T
def remaining_after_salary : ℝ := T - salary_cost T
def delivery_cost : ℝ := (1 / 4) * remaining_after_salary T
def cost_of_orders_done : ℝ := 1800

theorem haj_grocery_store_cost : T - salary_cost T - delivery_cost T = cost_of_orders_done → T = 8000 :=
by sorry

end haj_grocery_store_cost_l352_352935


namespace cousins_room_distributions_l352_352129

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l352_352129


namespace range_of_a_l352_352098

variable {a x : ℝ}

def p (x : ℝ) := sqrt (2 * x - 1) ≤ 1
def q (x : ℝ) := (x - a) * (x - (a + 1)) ≤ 0

theorem range_of_a :
  (∀ x, p x → q x) ∧ ∃ x, ¬ (p x → q x) →
  (0 ≤ a ∧ a ≤ 1 / 2) :=
by
  sorry

end range_of_a_l352_352098


namespace tournament_total_players_l352_352890

theorem tournament_total_players :
  ∃ (n : ℕ), ∀ (players weakest_points : ℕ),
  (players = n + 5) ∧
  (∀ (player points : ℕ), points * 2 = weakest_points * 5 * 2) ∧
  (∀ (points : ℕ), points = (players * (players - 1)) / 2) ∧
  (n = 20) :=
begin
  sorry
end

end tournament_total_players_l352_352890


namespace least_sum_of_exponents_520_l352_352483

def is_power_of_two (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 2^k

def sum_of_distinct_powers_of_two (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a ≠ b ∧ 2^a + 2^b = n

theorem least_sum_of_exponents_520 :
  ∀ (a b : ℕ), sum_of_distinct_powers_of_two 520 → a ≠ b → 2^a + 2^b = 520 → a + b = 12 :=
by
  sorry

end least_sum_of_exponents_520_l352_352483


namespace least_sum_of_exponents_of_powers_of_2_l352_352486

theorem least_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ s : Finset ℕ, (∑ x in s, 2^x = n) ∧ (∀ t : Finset ℕ, (∑ x in t, 2^x = n) → s.sum id ≤ t.sum id) :=
sorry

end least_sum_of_exponents_of_powers_of_2_l352_352486


namespace jimmy_garden_servings_l352_352093

variables 
  (carrot_servings : ℕ)
  (corn_servings : ℕ)
  (green_bean_servings : ℕ)
  (tomato_servings : ℕ)
  (zucchini_servings : ℕ)
  (bell_pepper_servings : ℕ)
  (total_servings : ℕ)

noncomputable def garden_servings_proof : Prop :=
  let carrot_servings := 4 in
  let corn_servings := 5 * carrot_servings in
  let green_bean_servings := corn_servings / 2 in
  let tomato_servings := carrot_servings + 3 in
  let zucchini_servings := 4 * green_bean_servings in
  let bell_pepper_servings := corn_servings - 2 in

  let total_servings := 
    8 * carrot_servings +
    12 * corn_servings +
    10 * green_bean_servings +
    15 * tomato_servings +
    9 * zucchini_servings +
    7 * bell_pepper_servings in

  total_servings = 963

theorem jimmy_garden_servings : garden_servings_proof :=
by {
  sorry
}

end jimmy_garden_servings_l352_352093


namespace matrix_A_to_power_4_l352_352329

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l352_352329


namespace cousin_distribution_count_l352_352123

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l352_352123


namespace length_of_train_is_correct_l352_352694

-- Define the conditions with the provided data and given formulas.
def train_speed_kmh : ℝ := 63
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
def time_to_pass_tree : ℝ := 16
def train_length : ℝ := train_speed_ms * time_to_pass_tree

-- State the problem as a theorem in Lean 4.
theorem length_of_train_is_correct : train_length = 280 := by
  -- conditions are defined, need to calculate the length
  unfold train_length train_speed_ms
  -- specify the conversion calculation manually
  simp
  norm_num
  sorry

end length_of_train_is_correct_l352_352694


namespace six_digit_number_property_l352_352392

theorem six_digit_number_property {a b c d e f : ℕ} 
  (h1 : 1 ≤ a ∧ a < 10) (h2 : 0 ≤ b ∧ b < 10)
  (h3 : 0 ≤ c ∧ c < 10) (h4 : 0 ≤ d ∧ d < 10)
  (h5 : 0 ≤ e ∧ e < 10) (h6 : 0 ≤ f ∧ f < 10) 
  (h7 : 100000 ≤ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f ∧
        a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f < 1000000) :
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 3 * (f * 10^5 + a * 10^4 + b * 10^3 + c * 10^2 + d * 10 + e)) ↔ 
  (a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 428571 ∨ a * 10^5 + b * 10^4 + c * 10^3 + d * 10^2 + e * 10 + f = 857142) :=
sorry

end six_digit_number_property_l352_352392


namespace rainfall_ratio_l352_352738

theorem rainfall_ratio (R_1 R_2 : ℕ) (h1 : R_1 + R_2 = 25) (h2 : R_2 = 15) : R_2 / R_1 = 3 / 2 :=
by
  sorry

end rainfall_ratio_l352_352738


namespace Geoff_vote_percentage_l352_352516

theorem Geoff_vote_percentage (total_votes : ℕ) (threshold_percentage : ℝ) (extra_votes_needed : ℕ) (votes_to_win : ℕ) :
  total_votes = 6000 →
  threshold_percentage = 0.51 →
  extra_votes_needed = 3000 →
  votes_to_win = nat.ceil (threshold_percentage * total_votes) →
  let geoff_votes := votes_to_win - extra_votes_needed in
  (geoff_votes : ℝ) / (total_votes : ℝ) * 100 ≈ 1.02 :=
by
  intros h1 h2 h3 h4 h5
  let a := 6000
  let b := 0.51
  let c := 3000
  have hyp1 : total_votes = a := h1
  have hyp2 : threshold_percentage = b := h2
  have hyp3 : extra_votes_needed = c := h3
  let d := nat.ceil (b * a)
  have hyp4 : votes_to_win = d := h4
  let geoff_votes := d - c
  show (geoff_votes : ℝ) / (a : ℝ) * 100 ≈ 1.02
  sorry  -- Proof omitted

end Geoff_vote_percentage_l352_352516


namespace binomial_12_10_eq_66_l352_352357

theorem binomial_12_10_eq_66 :
  Nat.choose 12 10 = 66 := by
  sorry

end binomial_12_10_eq_66_l352_352357


namespace find_sides_of_triangle_l352_352396

open Real
open Triangle

noncomputable def triangle_ABC : Type := { a : ℝ // a > 0 } × { b : ℝ // b > 0 } × { c : ℝ // c > 0 }

theorem find_sides_of_triangle
  (ABC : triangle_ABC)
  (BC : ℝ)
  (alt_to_AC : ℝ)
  (alt_to_BC : ℝ)
  (hBC : BC = 8)
  (h_alt_to_AC : alt_to_AC = 6.4)
  (h_alt_to_BC : alt_to_BC = 4) :
  ∃ AB AC : ℝ, AC = 5 ∧ AB = sqrt 41 := 
begin
  sorry
end

end find_sides_of_triangle_l352_352396


namespace interval_of_monotonic_increase_maximum_area_of_triangle_l352_352036

noncomputable def f (x : ℝ) : ℝ := 
  cos x * sin (x + π / 3) - sqrt 3 * cos x^2 + sqrt 3 / 4

theorem interval_of_monotonic_increase (k : ℤ) : 
  ∃ (interval : Set ℝ), interval = Set.Icc (k * π - π / 12) (k * π + 5 * π / 12) ∧ ∀ x ∈ interval, 
  ∀ y ∈ interval, x ≤ y → f x ≤ f y :=
sorry

theorem maximum_area_of_triangle (A : ℝ) (b c : ℝ) (a := sqrt 3) (area := 3 * sqrt 3 / 4) : 
  0 < A ∧ A < π / 2 → ∃ B C, 
  ∃ (triangle_area : ℝ), 
  B + C = π - A ∧ 
  triangle_area = (1 / 2) * b * c * sin A ∧ 
  f A = sqrt 3 / 4 ∧ 
  b^2 + c^2 - 2 * b * c * cos A = a ^ 2 ∧
  triangle_area <= area :=
sorry

end interval_of_monotonic_increase_maximum_area_of_triangle_l352_352036


namespace total_sale_price_correct_l352_352263

-- Define original prices
def saree_price : ℝ := 400
def kurti_price : ℝ := 350
def earrings_price : ℝ := 200
def shoes_price : ℝ := 500

-- Define discounts as percentages
def saree_discounts : list ℝ := [20, 10, 15]
def kurti_discounts : list ℝ := [10, 20, 5]
def earrings_discounts : list ℝ := [5, 15, 10]
def shoes_discounts : list ℝ := [10, 20, 5]

-- Function to apply successive discounts to an initial price
def apply_discounts (price : ℝ) (discounts : list ℝ) : ℝ :=
  discounts.foldl (λ p d, p * ((100 - d) / 100)) price

-- Calculate final prices after discounts
noncomputable def saree_final_price : ℝ := apply_discounts saree_price saree_discounts
noncomputable def kurti_final_price : ℝ := apply_discounts kurti_price kurti_discounts
noncomputable def earrings_final_price : ℝ := apply_discounts earrings_price earrings_discounts
noncomputable def shoes_final_price : ℝ := apply_discounts shoes_price shoes_discounts

-- Total sale price
noncomputable def total_sale_price : ℝ :=
  saree_final_price + kurti_final_price + earrings_final_price + shoes_final_price

-- Proof statement
theorem total_sale_price_correct :
  total_sale_price = 971.55 :=
by
  -- Proof will be filled in later
  sorry

end total_sale_price_correct_l352_352263


namespace ninth_term_geom_seq_l352_352365

theorem ninth_term_geom_seq : 
  let a_1 : ℚ := 5
  let r : ℚ := 3 / 2
  let a_9 : ℚ := a_1 * r^8
  in a_9 = 32805 / 256 :=
by
  let a_1 : ℚ := 5
  let r : ℚ := 3 / 2
  let a_9 : ℚ := a_1 * r^8
  have h : a_9 = 32805 / 256
  sorry

end ninth_term_geom_seq_l352_352365


namespace number_of_isosceles_triangle_numbers_l352_352598

-- Definitions based on the conditions:
-- a, b, and c are digits forming the three-digit number n = 100 * a + 10 * b + c
def is_digit (d : ℕ) := d ≥ 0 ∧ d ≤ 9

-- Conditions for forming a triangle with sides a, b, c
def is_triangle (a b c : ℕ) := a + b > c ∧ a + c > b ∧ b + c > a

-- Definition of an isosceles triangle
def is_isosceles_triangle (a b c : ℕ) := 
  is_triangle a b c ∧ (a = b ∨ a = c ∨ b = c)

-- Definition of the three-digit number n and condition for n consisting valid digits
def valid_three_digit_number (n : ℕ) :=
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ is_digit a ∧ is_digit b ∧ is_digit c

-- Combining conditions: n is a three-digit number and a, b, c form an isosceles triangle
def valid_isosceles_triangle_number (n : ℕ) :=
  valid_three_digit_number n ∧ 
  ∃ a b c : ℕ, n = 100 * a + 10 * b + c ∧ is_isosceles_triangle a b c

-- Main theorem: There are exactly 165 valid three-digit numbers n where a, b, and c form an isosceles triangle
theorem number_of_isosceles_triangle_numbers : 
  {n : ℕ | valid_isosceles_triangle_number n}.to_finset.card = 165 := 
sorry

end number_of_isosceles_triangle_numbers_l352_352598


namespace tamika_greater_than_carlos_l352_352963

/-- Define the set of numbers for Tamika --/
def tamika_set : set ℕ := {8, 9, 10}

/-- Define the set of numbers for Carlos --/
def carlos_set : set ℕ := {3, 5, 6}

/-- Define a function to calculate all unique sums for Tamika --/
def tamika_sums (s : set ℕ) : set ℕ :=
  {a + b | a ∈ s, b ∈ s, a ≠ b}

/-- Define a function to calculate all unique products for Carlos --/
def carlos_products (s : set ℕ) : set ℕ :=
  {a * b | a ∈ s, b ∈ s, a ≠ b}

/-- Calculate the probability that Tamika's result is greater than Carlos's result --/
def tamika_greater_probability : ℚ :=
  let tamika_results := tamika_sums tamika_set
  let carlos_results := carlos_products carlos_set
  let favorable_outcomes := {(t, c) | t ∈ tamika_results, c ∈ carlos_results, t > c}
  let all_possible_outcomes := {(t, c) | t ∈ tamika_results, c ∈ carlos_results}
  (favorable_outcomes.to_finset.card : ℚ) / all_possible_outcomes.to_finset.card

theorem tamika_greater_than_carlos : tamika_greater_probability = 4 / 9 :=
by sorry

end tamika_greater_than_carlos_l352_352963


namespace num_periodic_functions_l352_352043

def f1 (x : ℝ) : ℝ := √2
def f2 (x : ℝ) : ℝ := Real.sin x + Real.cos (√2 * x)
def f3 (x : ℝ) : ℝ := Real.sin (x / √2) + Real.cos (√2 * x)
def f4 (x : ℝ) : ℝ := Real.sin (x^2)

theorem num_periodic_functions : 
  (∃ T1 > 0, ∀ x, f1 (x + T1) = f1 x) ∨
  (∃ T2 > 0, ∀ x, f2 (x + T2) = f2 x) ∨
  (∃ T3 > 0, ∀ x, f3 (x + T3) = f3 x) ∨
  (∃ T4 > 0, ∀ x, f4 (x + T4) = f4 x) → 
  false :=
sorry

end num_periodic_functions_l352_352043


namespace hilda_loan_compounding_difference_l352_352840

noncomputable def difference_due_to_compounding (P : ℝ) (r : ℝ) (t : ℝ) : ℝ :=
  let A_monthly := P * (1 + r / 12)^(12 * t)
  let A_annually := P * (1 + r)^t
  A_monthly - A_annually

theorem hilda_loan_compounding_difference :
  difference_due_to_compounding 8000 0.10 5 = 376.04 :=
sorry

end hilda_loan_compounding_difference_l352_352840


namespace variance_of_numbers_l352_352034

theorem variance_of_numbers (a : ℝ) :
  (1 + 2 + 3 + 4 + a) / 5 = 3 -> (1 / 5) * ((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (4 - 3)^2 + (a - 3)^2) = 2 :=
by {
  intro h,
  have : a = 5, sorry,  
  rw this,
  norm_num,
  sorry
}

end variance_of_numbers_l352_352034


namespace compound_interest_accrued_l352_352237

def compoundInterest (P r : ℝ) (n t : ℤ) : ℝ :=
  P * (1 + r / n)^(n * t)

theorem compound_interest_accrued :
  let P : ℝ := 14800
  let r : ℝ := 0.135
  let n : ℤ := 1
  let t : ℤ := 2
  let A : ℝ := compoundInterest P r n t
  round (A - P) = 4266 :=
by
  sorry

end compound_interest_accrued_l352_352237


namespace neg_ln_gt_zero_l352_352192

theorem neg_ln_gt_zero {x : ℝ} : (¬ ∀ x : ℝ, Real.log (x^2 + 1) > 0) ↔ ∃ x : ℝ, Real.log (x^2 + 1) ≤ 0 := by
  sorry

end neg_ln_gt_zero_l352_352192


namespace triangle_ABC_is_equilateral_l352_352153

variable (A B C A' B' C' : Type)
variable [metric_space A] [metric_space B] [metric_space C]

-- angle bisectors for triangle ABC
variable (AA' : line A A') (BB' : line B B') (CC' : line C C')

-- angle bisectors for triangle A'B'C'
variable (A'AA' : line A' A) (B'BB' : line B' B) (C'CC' : line C' C)

-- Assume needed conditions
axiom bisector_AA' : is_angle_bisector A B C A'
axiom bisector_BB' : is_angle_bisector B A C B'
axiom bisector_CC' : is_angle_bisector C A B C'

axiom bisector_A'AA' : is_angle_bisector A' B C A
axiom bisector_B'BB' : is_angle_bisector B' A C B
axiom bisector_C'CC' : is_angle_bisector C' A B C

-- Now we state the theorem
theorem triangle_ABC_is_equilateral :
  ∀ (triangle_ABC : is_triangle A B C),
  ∀ (triangle_A'B'C' : is_triangle A' B' C'),
  triangle_is_equilateral A B C :=
by
  sorry

end triangle_ABC_is_equilateral_l352_352153


namespace yellow_fraction_tripled_l352_352510

theorem yellow_fraction_tripled (y : ℕ) (h : y > 0) :
  let green := (3 / 4 : ℚ)
  let yellow := 1 - green
  let new_yellow := 3 * yellow
  let new_total := green + new_yellow
  new_yellow / new_total = (1 / 2 : ℚ) :=
by
  let green := 3 / 4
  let yellow := 1 - green
  let new_yellow := 3 * yellow
  let new_total := green + new_yellow
  have h1 : yellow = 1 / 4 := by
    norm_num
  have h2 : new_yellow = 3 / 4 := by
    norm_num1
  have h3 : new_total = 3 / 2 := by
    norm_num1
  have h4 : new_yellow / new_total = 1 / 2 := by
    norm_num
  exact h4

end yellow_fraction_tripled_l352_352510


namespace solve_equation_l352_352156

noncomputable def cot_square (x : Real) : Real :=
  (1 / (Real.sin x) ^ 2 - 1)

noncomputable def nested_sqrt (x : Real) : Real :=
  Real.sqrt (4 + 3 * Real.sqrt (4 + 3 * Real.sqrt (1 + 3 / (Real.sin x)^2)))

theorem solve_equation (x : Real) (n k : ℤ) : 
  nested_sqrt x = cot_square x →
  (x = Real.arccot 2 + Real.pi * n) ∨ (x = Real.pi - Real.arccot 2 + Real.pi * k) :=
sorry

end solve_equation_l352_352156


namespace x_coordinate_of_point_on_parabola_at_distance_6_from_focus_l352_352628

-- Definitions and conditions based on the problem statement.
def parabola_eq (P : ℝ × ℝ) : Prop := P.snd ^ 2 = 12 * P.fst

def focus_dist (P : ℝ × ℝ) (d : ℝ) : Prop :=
  let F : ℝ × ℝ := (3, 0) in
  real.dist P F = d

-- The final theorem statement proving the desired x-coordinate.
theorem x_coordinate_of_point_on_parabola_at_distance_6_from_focus :
  ∃ P : ℝ × ℝ, parabola_eq P ∧ focus_dist P 6 ∧ P.fst = 3 := sorry

end x_coordinate_of_point_on_parabola_at_distance_6_from_focus_l352_352628


namespace quadrant_of_z_l352_352009

noncomputable def z : ℂ := 3 / (1 + 2 * I)

theorem quadrant_of_z : ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y < 0 := by
  sorry

end quadrant_of_z_l352_352009


namespace pet_store_animals_l352_352619

noncomputable def ratio_cats_dogs := 3 / 4
noncomputable def ratio_dogs_parrots := 2 / 5
def number_of_cats := 18

-- Definition to infer number of dogs from number of cats
def calc_dogs (cats : ℕ) := 
   (cats / 3) * 4

-- Definition to infer number of parrots from number of dogs
def calc_parrots (dogs : ℕ) := 
   (dogs / 2) * 5

-- We'll now state the problem as a theorem
theorem pet_store_animals :
  let dogs := calc_dogs number_of_cats in
  let parrots := calc_parrots dogs in
  dogs = 24 ∧ parrots = 60 :=
by
  let dogs := calc_dogs number_of_cats
  let parrots := calc_parrots dogs
  sorry

end pet_store_animals_l352_352619


namespace necessary_condition_not_sufficient_condition_l352_352794

variable (M : Type) [metric_space M]

def is_hyperbola_trajectory (M : Type) [metric_space M] (F1 F2 : M) : Prop :=
  ∃ c : ℝ, ∀ (m : M), abs (dist m F1 - dist m F2) = c

theorem necessary_condition (F1 F2 : M) :
  (∀ (m : M), is_hyperbola_trajectory M F1 F2) →
  (∃ (c : ℝ), ∀ (m : M), abs (dist m F1 - dist m F2) = c) :=
by sorry

theorem not_sufficient_condition (F1 F2 : M) : 
  (∃ (c : ℝ), ∀ (m : M), abs (dist m F1 - dist m F2) = c) →
  ¬ (∀ (m : M), is_hyperbola_trajectory M F1 F2) :=
by sorry

end necessary_condition_not_sufficient_condition_l352_352794


namespace num_multiples_of_five_with_units_digit_five_l352_352055

theorem num_multiples_of_five_with_units_digit_five :
  let multiples_of_five := {n : ℕ | n % 5 = 0 ∧ n < 100 ∧ 0 < n} in
  let ones_digit_five := {n : ℕ | n % 10 = 5} in
  ∃ (count : ℕ), count = 10 ∧ count = set.card (multiples_of_five ∩ ones_digit_five) :=
by
  sorry

end num_multiples_of_five_with_units_digit_five_l352_352055


namespace cousin_distribution_count_l352_352124

-- Definition of cousins and rooms
def num_cousins : ℕ := 5
def num_rooms : ℕ := 4

-- Definition to count the number of distributions
noncomputable def count_cousin_distributions : ℕ :=
  let case1 := 1 in -- (5,0,0,0)
  let case2 := choose 5 1 in -- (4,1,0,0)
  let case3 := choose 5 3 in -- (3,2,0,0)
  let case4 := choose 5 3 in -- (3,1,1,0)
  let case5 := choose 5 2 * choose 3 2 in -- (2,2,1,0)
  let case6 := choose 5 2 in -- (2,1,1,1)
  case1 + case2 + case3 + case4 + case5 + case6

-- Theorem to prove
theorem cousin_distribution_count : count_cousin_distributions = 66 := by
  sorry

end cousin_distribution_count_l352_352124


namespace neg_p_l352_352793

open Real

variable {f : ℝ → ℝ}

theorem neg_p :
  (∀ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) ≥ 0) →
  ∃ x1 x2 : ℝ, (f x2 - f x1) * (x2 - x1) < 0 :=
sorry

end neg_p_l352_352793


namespace min_sum_of_exponents_of_powers_of_2_l352_352496

theorem min_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ (s : set ℕ), (∀ (k ∈ s), ∃ (m : ℕ), k = 2 ^ m) ∧ (s.sum id = 520) ∧ (s.sum id = s.card * s.card) → (s.sum id = 12) := sorry

end min_sum_of_exponents_of_powers_of_2_l352_352496


namespace not_necessarily_midpoints_l352_352575

open_locale classical

noncomputable theory

variables {A B C A1 B1 C1 : Type*}
variables [linear_ordered_field A] [linear_ordered_field B] [linear_ordered_field C]
variables [linear_ordered_field A1] [linear_ordered_field B1] [linear_ordered_field C1]

-- Defining the geometric context and conditions
def is_midpoint (M X Y : Type*) [linear_ordered_field M] [linear_ordered_field X] [linear_ordered_field Y] (MB MX : M) :=
  MB = MX / 2

def angles_equal (θ1 θ2 : real) := θ1 = θ2

-- Main theorem statement
theorem not_necessarily_midpoints 
  (h_midC1 : is_midpoint C1 A B)
  (h_angle1 : angles_equal (angle B1 C1 A1) (angle C))
  (h_angle2 : angles_equal (angle C1 A1 B1) (angle A))
  (h_angle3 : angles_equal (angle A1 B1 C1) (angle B)) :
  ¬ (is_midpoint A1 B C ∧ is_midpoint B1 A C) :=
sorry

end not_necessarily_midpoints_l352_352575


namespace cos_angle_E_l352_352526

variables {a b : ℝ} {β : ℝ}

theorem cos_angle_E (h1 : 150 = 150) (h2 : a ≠ b) (h3 : a + b = 220): 
  cos β = 11 / 15 := 
  sorry

end cos_angle_E_l352_352526


namespace solve_equation_l352_352157

noncomputable def cot_square (x : Real) : Real :=
  (1 / (Real.sin x) ^ 2 - 1)

noncomputable def nested_sqrt (x : Real) : Real :=
  Real.sqrt (4 + 3 * Real.sqrt (4 + 3 * Real.sqrt (1 + 3 / (Real.sin x)^2)))

theorem solve_equation (x : Real) (n k : ℤ) : 
  nested_sqrt x = cot_square x →
  (x = Real.arccot 2 + Real.pi * n) ∨ (x = Real.pi - Real.arccot 2 + Real.pi * k) :=
sorry

end solve_equation_l352_352157


namespace GC_perpendicular_AC_l352_352893

variables {A B C D E F G : Type*}
variables [parallelogram ABCD]
variables [line CE]
variables [line CF]
variables [perpendicular CE AB E]
variables [perpendicular CF AD F]
variables [line EF]
variables [intersection EF BD G]

theorem GC_perpendicular_AC : GC ⊥ AC := sorry

end GC_perpendicular_AC_l352_352893


namespace distance_point_to_line_l352_352975

def point := (ℝ × ℝ)
def line (A B C : ℝ) := {p : point // A * p.1 + B * p.2 + C = 0}
def distance (p : point) (l : line 3 4 (-26)) : ℝ := Real.abs ((3 * p.1 + 4 * p.2 + (-26)) / (Real.sqrt (3 ^ 2 + 4 ^ 2)))

theorem distance_point_to_line :
  distance (3, -2) ⟨(3, 4), rfl⟩ = 5 :=
by
  -- proof steps go here
  sorry

end distance_point_to_line_l352_352975


namespace counterexample_to_twin_prime_conjecture_l352_352364

-- Let p be a prime number.
noncomputable theory

/-- 
  The Twin Prime Conjecture proposes that there are infinitely many pairs of prime 
  numbers (p, p+2) where both numbers in the pair are primes.
  A counterexample to this conjecture is a proof that there are finitely many twin primes.
-/
theorem counterexample_to_twin_prime_conjecture (h : ∀ n : ℕ, ∃ p : ℕ, prime p ∧ prime (p + 2)) :
  ¬ (∃ f : ℕ → ℕ, (λ n => prime (f n) ∧ prime (f n + 2)) ∧ (∀ n m : ℕ, n ≠ m → f n ≠ f m)) :=
sorry

end counterexample_to_twin_prime_conjecture_l352_352364


namespace matrix_pow_four_l352_352322

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l352_352322


namespace binom_12_10_l352_352343

theorem binom_12_10 : nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_l352_352343


namespace triangle_ABC_is_right_triangle_l352_352139

open Function

variable {A B C P Q : Type} [AddCommGroup P]
variable {AP PB CP PQ : ℝ}
variable (midQ : midpoint ℝ A C = Q)

theorem triangle_ABC_is_right_triangle (h₁ : AP = 2 * PB) (h₂ : CP = 2 * PQ) (h_midQ : midpoint A C = Q) :
  ∃ (ABC : Triangle P), is_right_angle (angle ABC B C) := by 
sorry

end triangle_ABC_is_right_triangle_l352_352139


namespace a_plus_b_l352_352677

-- Definition of the line intersecting the graph
def intersects (k : ℝ) := ∃ (x : ℝ), (log 3 x = log 3 k ∨ log 3 (x + 6) = log 3 k)

-- Distance between points of intersection is given to be 1/3
def distance_condition (k : ℝ) := abs (log 3 k - log 3 (k + 6)) = 1 / 3

-- Definition of k given by integers a and b
def k_definition (k : ℝ) (a b : ℤ) := k = a + real.sqrt b

-- The final proof problem
theorem a_plus_b (k : ℝ) (a b : ℤ) (hk : k_definition k a b) (hd : distance_condition k) : a + b = 6 :=
by
  sorry

end a_plus_b_l352_352677


namespace intersection_M_N_l352_352047

open Set

noncomputable def M : Set ℝ := {-1, 0, 1}
noncomputable def N : Set ℝ := {x | x^2 + x ≤ 0}

theorem intersection_M_N : M ∩ N = {-1, 0} := sorry

end intersection_M_N_l352_352047


namespace cousins_room_distributions_l352_352126

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l352_352126


namespace ratio_increase_productivity_l352_352988

theorem ratio_increase_productivity (initial current: ℕ) 
  (h_initial: initial = 10) 
  (h_current: current = 25) : 
  (current - initial) / initial = 3 / 2 := 
by
  sorry

end ratio_increase_productivity_l352_352988


namespace PR_length_l352_352900

-- Definitions of the conditions
def is_right_angle (P Q R : Type) := true  -- Assuming there is a proof that ∠PQR = 90°
def sin (x : ℝ) := real.sin x  -- Using real.sin for the sin function

-- Given conditions
axiom triangle_PQR : Type
axiom is_right_angled_at_Q : is_right_angle P Q R
axiom sin_R : sin R = 3 / 5
axiom PQ_length : PQ = 15

-- Proving the length of PR
theorem PR_length : PR = 25 :=
by
  sorry

end PR_length_l352_352900


namespace bags_initially_made_l352_352744

-- Definitions based on conditions
def cost_per_bag : ℝ := 3.0
def selling_price_regular : ℝ := 6.0
def bags_sold_regular : ℕ := 15
def selling_price_discounted : ℝ := 4.0
def bags_sold_discounted : ℕ := 5
def net_profit : ℝ := 50.0

-- Statement to prove
theorem bags_initially_made : 
  let total_revenue := (bags_sold_regular * selling_price_regular) + 
                        (bags_sold_discounted * selling_price_discounted),
      total_cost_of_ingredients := total_revenue - net_profit,
      num_bags_made := total_cost_of_ingredients / cost_per_bag
  in num_bags_made = 20 :=
by
  -- Proof is not required in the task
  sorry

end bags_initially_made_l352_352744


namespace min_sum_of_exponents_of_powers_of_2_l352_352494

theorem min_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ (s : set ℕ), (∀ (k ∈ s), ∃ (m : ℕ), k = 2 ^ m) ∧ (s.sum id = 520) ∧ (s.sum id = s.card * s.card) → (s.sum id = 12) := sorry

end min_sum_of_exponents_of_powers_of_2_l352_352494


namespace cyclic_sum_inequality_l352_352579

open Real

theorem cyclic_sum_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
    (∑ cyc in [{x, y, z}],
    (sqrt (x / (y + z)) * sqrt ((x * y + x * z + y ^ 2 + z ^ 2) / (y ^ 2 + z ^ 2)))) 
    ≥ 2 * sqrt 2 :=
  sorry

end cyclic_sum_inequality_l352_352579


namespace base_digit_difference_l352_352848

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l352_352848


namespace negation_of_forall_statement_l352_352191

variable (x : ℝ)

theorem negation_of_forall_statement :
  (¬ ∀ x > 1, x - 1 > Real.log x) ↔ (∃ x > 1, x - 1 ≤ Real.log x) := by
  sorry

end negation_of_forall_statement_l352_352191


namespace angles_in_range_l352_352656

def angle_set (k : ℤ) : ℝ :=
  (π / 3) + 2 * k * π

def S (β : ℝ) : Prop :=
  ∃ k : ℤ, β = angle_set k

def β_in_range (β : ℝ) : Prop :=
  -2 * π ≤ β ∧ β < 4 * π

theorem angles_in_range :
  { β : ℝ | S β ∧ β_in_range β } = {- (5 * π / 3), π / 3, 7 * π / 3} :=
by {
  sorry
}

end angles_in_range_l352_352656


namespace number_of_ways_to_put_cousins_in_rooms_l352_352117

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l352_352117


namespace exists_reducible_polynomial_for_all_l352_352774

theorem exists_reducible_polynomial_for_all :
  ∃ g : Polynomial ℤ, ∀ (f : Fin 2018 → Polynomial ℤ),
  (∀ i, ¬(f i).isC) → ∀ i, ¬ irreducible (f i).comp g := by
sorಮ

end exists_reducible_polynomial_for_all_l352_352774


namespace magnitude_v_l352_352159

theorem magnitude_v (u v : ℂ) (h1 : u * v = 24 - 10 * complex.i) (h2 : complex.abs u = 5) : complex.abs v = 26 / 5 := 
  sorry

end magnitude_v_l352_352159


namespace rem_5_div_12_3_div_4_l352_352363

def rem (x y : ℝ) : ℝ :=
  x - y * ⌊x / y⌋

theorem rem_5_div_12_3_div_4 :
  rem (5 / 12) (3 / 4) = 5 / 12 :=
by sorry

end rem_5_div_12_3_div_4_l352_352363


namespace min_value_correct_l352_352398

noncomputable def min_value_expression : ℝ :=
  let expr : ℝ → ℝ := λ x, (15 - x) * (13 - x) * (15 + x) * (13 + x)
  in min (expr (sqrt 197)) (expr (-sqrt 197))

theorem min_value_correct :
  min_value_expression = -961 :=
by
  sorry

end min_value_correct_l352_352398


namespace number_of_solutions_l352_352401

theorem number_of_solutions :
  ∃ (x y z : ℝ), 
    (x = 4036 - 4037 * Real.sign (y - z)) ∧ 
    (y = 4036 - 4037 * Real.sign (z - x)) ∧ 
    (z = 4036 - 4037 * Real.sign (x - y)) :=
sorry

end number_of_solutions_l352_352401


namespace complete_square_solution_l352_352592

theorem complete_square_solution :
  ∀ (x : ℝ), (x^2 + 8*x + 9 = 0) → ((x + 4)^2 = 7) :=
by
  intro x h_eq
  sorry

end complete_square_solution_l352_352592


namespace remaining_area_is_344_l352_352164

def garden_length : ℕ := 20
def garden_width : ℕ := 18
def shed_side : ℕ := 4

def area_rectangle : ℕ := garden_length * garden_width
def area_shed : ℕ := shed_side * shed_side

def remaining_garden_area : ℕ := area_rectangle - area_shed

theorem remaining_area_is_344 : remaining_garden_area = 344 := by
  sorry

end remaining_area_is_344_l352_352164


namespace arithmetic_sequence_mean_median_sample_l352_352786

theorem arithmetic_sequence_mean_median_sample :
  ∃ d : ℕ, ∀ (a : ℕ → ℕ), 
    (∀ n : ℕ, 1 ≤ n ∧ n ≤ 20 → a n = 4 + (n - 1) * d) 
    ∧ a 1 = 4 
    ∧ a 20 = 42 
    → (let S_20 := ∑ n in finset.range 20, a (n + 1)
            mean := S_20 / 20
            a_10 := a 10
            a_11 := a 11
            median := (a_10 + a_11) / 2
       in mean = 23 ∧ median = 23) :=
sorry

end arithmetic_sequence_mean_median_sample_l352_352786


namespace sphere_surface_area_l352_352805

theorem sphere_surface_area (r : ℝ) (h : r = 3) : 4 * real.pi * r ^ 2 = 36 * real.pi := by
  rw [h]
  simp
  sorry

end sphere_surface_area_l352_352805


namespace base_digit_difference_l352_352847

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l352_352847


namespace exist_consecutive_natural_numbers_with_digit_sum_divisible_by_7_minimum_sum_of_digits_for_number_divisible_by_99_l352_352379

-- Part 1: Existential statement for consecutive natural numbers with digit sum divisible by 7
theorem exist_consecutive_natural_numbers_with_digit_sum_divisible_by_7 :
  ∃ n : ℕ, (nat.digits 10 n).sum % 7 = 0 ∧ (nat.digits 10 (n + 1)).sum % 7 = 0 :=
sorry

-- Part 2: Minimum sum of digits for number divisible by 99
theorem minimum_sum_of_digits_for_number_divisible_by_99 :
  (∃ n : ℕ, n % 99 = 0 ∧ (nat.digits 10 n).sum = 18) ∧ 
  (∀ m : ℕ, m % 99 = 0 → (nat.digits 10 m).sum >= 18) :=
sorry

end exist_consecutive_natural_numbers_with_digit_sum_divisible_by_7_minimum_sum_of_digits_for_number_divisible_by_99_l352_352379


namespace domain_of_f_when_a_is_5_range_of_a_when_domain_is_real_l352_352561

noncomputable theory

def f (x : ℝ) (a : ℝ) : ℝ := real.sqrt (abs (2 * x + 1) + abs (2 * x - 2) - a)

theorem domain_of_f_when_a_is_5 :
  { x : ℝ | ∃ y : ℝ, y = f x 5 } = { x : ℝ | x ≤ -1 } ∪ { x : ℝ | x ≥ 3 / 2 } :=
by
  sorry

theorem range_of_a_when_domain_is_real (a : ℝ) :
  (∀ x : ℝ, ∃ y : ℝ, y = f x a) → a ≤ 3 :=
by
  sorry

end domain_of_f_when_a_is_5_range_of_a_when_domain_is_real_l352_352561


namespace tara_marbles_modulo_l352_352162

noncomputable def binom (n k : ℕ) : ℕ := Nat.choose n k

theorem tara_marbles_modulo : 
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  N % 1000 = 564 :=
by
  let B := 6  -- number of blue marbles
  let Y := 12  -- additional yellow marbles for balance
  let total_marbles := B + Y  -- total marbles
  let arrangements := binom (total_marbles + B) B
  let N := arrangements
  have : N % 1000 = 564 := sorry
  exact this

end tara_marbles_modulo_l352_352162


namespace rectangle_area_l352_352538

theorem rectangle_area (x : ℝ) (h1 : 4 * rect_side_length = 160) :
  4 * (rect_side_length ^ 2) = 6400 / 9 :=
begin
  sorry
end

end rectangle_area_l352_352538


namespace fullness_related_to_sowing_date_probability_at_least_one_full_grain_expectation_and_variance_X_l352_352596

def weights_A := [3, 6, 11] -- counts for intervals from field A
def weights_B := [6, 10, 4] -- counts for intervals from field B

def significance_level := 0.025
def chi_square_threshold := 5.024
def full_grain_weight := 200

def calc_chi_square (a b c d : ℕ) : ℚ := 
  let n := a + b + c + d
  let numerator := n * (a * d - b * c)^2
  let denominator := (a + b) * (c + d) * (a + c) * (b + d)
  numerator / denominator

theorem fullness_related_to_sowing_date :
  calc_chi_square 11 9 4 16 > chi_square_threshold :=
by sorry

def prob_full_grain_A := 11 / 20
def prob_full_grain_B := 4 / 20

theorem probability_at_least_one_full_grain :
  1 - ((1 - prob_full_grain_A) * (1 - prob_full_grain_B)) = 89 / 100 :=
by sorry

def X := binomial 100 (11 / 20)

theorem expectation_and_variance_X :
  E(X) = 55 ∧ Var(X) = 99 / 4 :=
by sorry

end fullness_related_to_sowing_date_probability_at_least_one_full_grain_expectation_and_variance_X_l352_352596


namespace compare_f_values_l352_352669

noncomputable def f (x : ℝ) : ℝ :=
  x^2 - Real.cos x

theorem compare_f_values :
  f 0 < f 0.5 ∧ f 0.5 < f 0.6 :=
by {
  -- proof would go here
  sorry
}

end compare_f_values_l352_352669


namespace particular_solution_ODE_l352_352391

theorem particular_solution_ODE (y : ℝ → ℝ) (h : ∀ x, deriv y x + y x * Real.tan x = 0) (h₀ : y 0 = 2) :
  ∀ x, y x = 2 * Real.cos x :=
sorry

end particular_solution_ODE_l352_352391


namespace smallest_y_l352_352404

theorem smallest_y (y : ℕ) (h : 56 * y + 8 ≡ 6 [MOD 26]) : y = 6 := by
  sorry

end smallest_y_l352_352404


namespace sum_a_1999_l352_352046

def a : ℕ → ℕ
| 1 := 1
| 2 := 2
| n@(n' + 3) := (a n' + a (n' + 1)) / (a n' * a (n' + 1) - 1)

-- Conditions
axiom recurrence_relation (n : ℕ) : (a n) * (a (n + 1)) * (a (n + 2)) = (a n) + (a (n + 1)) + (a (n + 2))
axiom non_trivial (n : ℕ) : (a (n + 1)) * (a (n + 2)) ≠ 1

-- Main Theorem
theorem sum_a_1999 : (finset.sum (finset.range 1999) a) = 3997 := 
by sorry

end sum_a_1999_l352_352046


namespace four_digit_integer_l352_352171

theorem four_digit_integer (a b c d : ℕ) 
(h1: a + b + c + d = 14) (h2: b + c = 9) (h3: a - d = 1)
(h4: (a - b + c - d) % 11 = 0) : 1000 * a + 100 * b + 10 * c + d = 3542 :=
by
  sorry

end four_digit_integer_l352_352171


namespace range_of_a_l352_352424

theorem range_of_a (a : ℝ) :
  (∀ x: ℝ, |x - a| < 4 → -x^2 + 5 * x - 6 > 0) → (-1 ≤ a ∧ a ≤ 6) :=
by
  intro h
  sorry

end range_of_a_l352_352424


namespace problem1_problem2_l352_352018

-- Definition of sets A and B
def A : Set ℝ := { x | x^2 - 2*x - 3 < 0 }
def B (p : ℝ) : Set ℝ := { x | abs (x - p) > 1 }

-- Statement for the first problem
theorem problem1 : B 0 ∩ A = { x | 1 < x ∧ x < 3 } := 
by
  sorry

-- Statement for the second problem
theorem problem2 (p : ℝ) (h : A ∪ B p = B p) : p ≤ -2 ∨ p ≥ 4 := 
by
  sorry

end problem1_problem2_l352_352018


namespace fibonacci_polynomial_property_l352_352600

noncomputable def Fibonacci : ℕ → ℕ
| 0     := 0
| 1     := 1
| n + 2 := Fibonacci (n + 1) + Fibonacci n

noncomputable def P : ℕ → ℤ := sorry -- the unique polynomial matching the Fibonacci sequence for n ∈ {1, 2, ..., 10}

theorem fibonacci_polynomial_property :
  P(100) - ∑ k in Finset.range (98 + 1) \ Finset.range 11, P k = (1529 / 10 : ℚ) * (Nat.choose 98 9) + 144 :=
sorry

end fibonacci_polynomial_property_l352_352600


namespace smallest_value_l352_352500

theorem smallest_value (x : ℝ) (h1 : 1 < x) (h2 : x < 2) : 
  \min (x^3) (x^2) (2*x) (sqrt x) (1/x^2) = (1/x^2) :=
by sorry

end smallest_value_l352_352500


namespace sequence_sum_l352_352061

noncomputable def reciprocal_diff (a : ℚ) : ℚ := 1 / (1 - a)

lemma sequence_step1 : reciprocal_diff (-1/3) = 3/4 := by
  sorry

lemma sequence_step2 : reciprocal_diff (3/4) = 4 := by
  sorry

lemma sequence_step3 : reciprocal_diff 4 = -1/3 := by
  sorry

theorem sequence_sum (a : ℕ → ℚ) (h : ∀ n, a (n + 3) = a n) :
  (∑ i in Finset.range 3600, a i) = 5300 :=
by
  sorry

end sequence_sum_l352_352061


namespace matrix_power_four_l352_352334

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l352_352334


namespace domain_of_log_function_l352_352172

theorem domain_of_log_function :
  (∃ f : ℝ → ℝ, (∀ x, f x = log 2 (3 - x)) ∧ ∀ x, 3 - x > 0 → x < 3) :=
sorry

end domain_of_log_function_l352_352172


namespace range_m_graph_in_quadrants_l352_352502

theorem range_m_graph_in_quadrants (m : ℝ) :
  (∀ x : ℝ, x ≠ 0 → ((x > 0 → (m + 2) / x > 0) ∧ (x < 0 → (m + 2) / x < 0))) ↔ m > -2 :=
by 
  sorry

end range_m_graph_in_quadrants_l352_352502


namespace angie_pretzels_dave_pretzels_l352_352711

theorem angie_pretzels (B S A : ℕ) (hB : B = 12) (hS : S = B / 2) (hA : A = 3 * S) : A = 18 := by
  -- We state the problem using variables B, S, and A for Barry, Shelly, and Angie respectively
  sorry

theorem dave_pretzels (A S D : ℕ) (hA : A = 18) (hS : S = 12 / 2) (hD : D = 25 * (A + S) / 100) : D = 6 := by
  -- We use variables A and S from the first theorem, and introduce D for Dave
  sorry

end angie_pretzels_dave_pretzels_l352_352711


namespace num_correct_statements_l352_352208

theorem num_correct_statements : 
  (¬ ∃ (T : Type) [triangle T], ∃ (A B C : T), 
    is_angle_bisector_perpendicular A B C ∧ 
      (interior_angle A + interior_angle B + interior_angle C = 180)) ∧ 
  (¬ ∃ (T : Type) [triangle T], ∃ (A B C : T), 
    ratio_of_altitudes A B C = (1, 2, 3)) ∧ 
  (¬ ∃ (T : Type) [triangle T], ∃ (A B C : T), 
    is_median_not_less_than_half_sum_of_other_sides A B C) → 
  num_correct_statements = 0 := 
by
  sorry

-- Definitions required for the theorem above
def is_angle_bisector_perpendicular (A B C : T) : Prop :=
  -- Define the property that two angle bisectors are perpendicular
  sorry

def interior_angle (A : T) : ℝ :=
  -- Define the interior angle of the triangle
  sorry

def ratio_of_altitudes (A B C : T) : (ℝ × ℝ × ℝ) :=
  -- Define the ratio of the altitudes of the triangle
  sorry

def is_median_not_less_than_half_sum_of_other_sides (A B C : T) : Prop :=
  -- Define the property that one of the medians is not less than half the sum of the other two sides
  sorry

def num_correct_statements := 0 -- We state that the number of correct statements is 0

end num_correct_statements_l352_352208


namespace find_a_plus_d_l352_352661

theorem find_a_plus_d (a b c d : ℝ) (h1 : a + b = 5) (h2 : b + c = 6) (h3 : c + d = 3) : a + d = -1 := 
by 
  -- omit proof
  sorry

end find_a_plus_d_l352_352661


namespace percentage_of_employees_in_manufacturing_l352_352241

theorem percentage_of_employees_in_manufacturing (d total_degrees : ℝ) (h1 : d = 144) (h2 : total_degrees = 360) :
    (d / total_degrees) * 100 = 40 :=
by
  sorry

end percentage_of_employees_in_manufacturing_l352_352241


namespace least_sum_exponents_of_520_l352_352493

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l352_352493


namespace regular_seminar_fee_l352_352685

-- Define the main problem statement
theorem regular_seminar_fee 
  (F : ℝ) 
  (discount_per_teacher : ℝ) 
  (number_of_teachers : ℕ)
  (food_allowance_per_teacher : ℝ)
  (total_spent : ℝ) :
  discount_per_teacher = 0.95 * F →
  number_of_teachers = 10 →
  food_allowance_per_teacher = 10 →
  total_spent = 1525 →
  (number_of_teachers * discount_per_teacher + number_of_teachers * food_allowance_per_teacher = total_spent) →
  F = 150 := 
  by sorry

end regular_seminar_fee_l352_352685


namespace polyhedron_face_edge_relation_l352_352087

-- Definitions
variable (Polyhedron : Type)
variable [HasFaces Polyhedron] [HasEdges Polyhedron] [HasVertices Polyhedron]
variable (f : Polyhedron → ℕ)  -- number of faces
variable (a : Polyhedron → ℕ)  -- number of edges
variable (f_k : ℕ → Polyhedron → ℕ)  -- number of k-sided faces

-- Hypotheses / Conditions
axiom convex (P : Polyhedron) : IsConvex P

-- Theorem
theorem polyhedron_face_edge_relation (P : Polyhedron) (hc : IsConvex P) :
  2 * a P = ∑ k in (Ico 3 (k+1)), k * (f_k k P) := 
sorry

end polyhedron_face_edge_relation_l352_352087


namespace distance_to_origin_eq_three_l352_352607

theorem distance_to_origin_eq_three :
  let P := (1, 2, 2)
  let origin := (0, 0, 0)
  dist P origin = 3 := by
  sorry

end distance_to_origin_eq_three_l352_352607


namespace table_tennis_total_scores_l352_352092

theorem table_tennis_total_scores :
  ∃ n : ℕ, n = 8 ∧ (
    ∃ (a₁ b₁ a₂ b₂ a₃ b₃ : ℕ),
      -- Conditions for scoring in a set
      ((a₁ = 11 ∧ b₁ < 10) ∨ (b₁ = 11 ∧ a₁ < 10) ∨ (a₁ ≥ 10 ∧ b₁ ≥ 10 ∧ abs(a₁ - b₁) = 2)) ∧
      ((a₂ = 11 ∧ b₂ < 10) ∨ (b₂ = 11 ∧ a₂ < 10) ∨ (a₂ ≥ 10 ∧ b₂ ≥ 10 ∧ abs(a₂ - b₂) = 2)) ∧
      ((a₃ = 11 ∧ b₃ < 10) ∨ (b₃ = 11 ∧ a₃ < 10) ∨ (a₃ ≥ 10 ∧ b₃ ≥ 10 ∧ abs(a₃ - b₃) = 2)) ∧
      -- Condition for the total score
      (a₁ + b₁ + a₂ + b₂ + a₃ + b₃ = 30)
  ) :=
sorry

end table_tennis_total_scores_l352_352092


namespace matrix_power_four_correct_l352_352320

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l352_352320


namespace parking_arrangements_l352_352512

theorem parking_arrangements : 
  let num_parking_spaces := 7
  let num_models_of_cars := 3
  let num_empty_spaces := 4
  (fact num_models_of_cars) * (num_parking_spaces - num_empty_spaces) = 24 :=
by
  -- Definitions from conditions
  let num_parking_spaces := 7
  let num_models_of_cars := 3
  let num_empty_spaces := 4

  -- The proof is skipped with "sorry"
  sorry

end parking_arrangements_l352_352512


namespace find_r_x_l352_352249

open Nat

theorem find_r_x (r n : ℕ) (x : ℕ) (h_r_le_70 : r ≤ 70) (repr_x : x = (10 * r + 6) * (r ^ (2 * n) - 1) / (r ^ 2 - 1))
  (repr_x2 : x^2 = (r ^ (4 * n) - 1) / (r - 1)) :
  (r = 7 ∧ x = 26) :=
by
  sorry

end find_r_x_l352_352249


namespace solve_problem_l352_352223

def num : ℕ := 1 * 3 * 5 * 7
def den : ℕ := 1 + 2 + 3 + 4 + 5 + 6 + 7

theorem solve_problem : (num : ℚ) / den = 3.75 := 
by
  sorry

end solve_problem_l352_352223


namespace initial_rope_length_is_approximately_16_l352_352279

noncomputable def initial_rope_length : ℝ :=
  let pi := Real.pi
  let extra_area := 858
  let final_length := 23
  let final_area := pi * final_length^2
  let initial_area := final_area - extra_area
  Real.sqrt (initial_area / pi)

theorem initial_rope_length_is_approximately_16 :
  initial_rope_length ≈ 16 := by
  sorry

end initial_rope_length_is_approximately_16_l352_352279


namespace probability_no_adjacent_same_rolls_l352_352763

theorem probability_no_adjacent_same_rolls : 
  let A := [0, 1, 2, 3, 4, 5] -- Representing six faces of a die
  let rollings : List (A → ℕ) -- Each person rolls and the result is represented as a map from faces to counts (a distribution in effect)
  ∃ rollings : List (A → ℕ), 
    (∀ (i : Fin 5), rollings[i] ≠ rollings[(i + 1) % 5]) →
      probability rollings
    = 375 / 2592 :=
by
  sorry

end probability_no_adjacent_same_rolls_l352_352763


namespace sum_integer_solutions_correct_l352_352823

noncomputable def sum_of_integer_solutions (m : ℝ) : ℝ :=
  if (3 ≤ m ∧ m < 6) ∨ (-6 ≤ m ∧ m < -3) then -9 else 0

theorem sum_integer_solutions_correct (m : ℝ) :
  (∀ x : ℝ, (3 * x + m < 0 ∧ x > -5) → (∃ s : ℝ, s = sum_of_integer_solutions m ∧ s = -9)) :=
by
  sorry

end sum_integer_solutions_correct_l352_352823


namespace arithmetic_sequence_third_eighth_term_sum_l352_352015

variable {α : Type*} [AddCommGroup α] [Module ℚ α]

def arith_sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem arithmetic_sequence_third_eighth_term_sum {a : ℕ → ℚ} {S : ℕ → ℚ} 
  (h_seq: ∀ n, a n = a 1 + (n - 1) * d)
  (h_sum: arith_sequence_sum a S) 
  (h_S10 : S 10 = 4) : 
  a 3 + a 8 = 4 / 5 :=
by
  sorry

end arithmetic_sequence_third_eighth_term_sum_l352_352015


namespace bobs_corn_harvest_l352_352720

theorem bobs_corn_harvest : 
  let row1 := 82 // 8,
      row2 := 94 // 9,
      row3 := 78 // 7,
      row4 := 96 // 12,
      row5 := 85 // 10,
      row6 := 91 // 13,
      row7 := 88 // 11
  in 
  row1 + row2 + row3 + row4 + row5 + row6 + row7 = 62 :=
by 
  have h1 : ((82 : ℕ) // 8) = 10 := by sorry,
  have h2 : ((94 : ℕ) // 9) = 10 := by sorry,
  have h3 : ((78 : ℕ) // 7) = 11 := by sorry,
  have h4 : ((96 : ℕ) // 12) = 8 := by sorry,
  have h5 : ((85 : ℕ) // 10) = 8 := by sorry,
  have h6 : ((91 : ℕ) // 13) = 7 := by sorry,
  have h7 : ((88 : ℕ) // 11) = 8 := by sorry,
  calc
    row1 + row2 + row3 + row4 + row5 + row6 + row7
    _ = 10 + 10 + 11 + 8 + 8 + 7 + 8 := by rw [h1, h2, h3, h4, h5, h6, h7]
    _ = 62 : by norm_num

end bobs_corn_harvest_l352_352720


namespace definite_integral_ln_l352_352746

open Real

theorem definite_integral_ln :
  ∫ x in 1..2, (1 / x) = log 2 :=
by
  sorry

end definite_integral_ln_l352_352746


namespace ellipse_equation_constant_abscissa_l352_352435

-- Conditions
def isEllipse (a b : ℝ) (x y : ℝ) : Prop :=
  a > b ∧ b > 0 ∧ (x^2 / a^2 + y^2 / b^2 = 1)

def eccentricity (a b c : ℝ) : Prop :=
  c = a / 2

def maxAreaTriangle (a b c : ℝ) (area : ℝ) : Prop :=
  area = (1 / 2) * b * c

def lineThroughPointT (T : (ℝ × ℝ)) (M N : (ℝ × ℝ)) : Prop :=
  T = (-1, 0) ∧ M ≠ T ∧ N ≠ T

def pointsNotVerticies (A B M N : (ℝ × ℝ)) : Prop :=
  M ≠ A ∧ M ≠ B ∧ N ≠ A ∧ N ≠ B

-- 1. Prove the equation of the ellipse
theorem ellipse_equation (a b c : ℝ) (x y : ℝ) (e : ℝ) (area : ℝ) (T M N A B : (ℝ × ℝ)) :
  isEllipse a b x y →
  eccentricity a b c →
  maxAreaTriangle a b c area →
  lineThroughPointT T M N →
  pointsNotVerticies A B M N →
  (a = 2 ∧ b = sqrt 3) :=
  sorry

-- 2. Prove the abscissa of intersection point is a constant value
theorem constant_abscissa (a b c x_intersection : ℝ) (T M N A B : (ℝ × ℝ)) :
  isEllipse a b x_intersection (0 : ℝ) →
  eccentricity a b c →
  lineThroughPointT T M N →
  pointsNotVerticies A B M N →
  (x_intersection = -4) :=
  sorry

end ellipse_equation_constant_abscissa_l352_352435


namespace relationship_among_M_N_P_l352_352008

theorem relationship_among_M_N_P (x y : ℝ) (hx : x < y) (hy : y < 0) :
  let M := |x| in
  let N := |y| in
  let P := (|x + y|) / 2 in
  M > P ∧ P > N :=
by
  sorry

end relationship_among_M_N_P_l352_352008


namespace find_angle_ABC_l352_352898

theorem find_angle_ABC (A B C D : Point) (h1 : length AB = length AD) (h2 : angle DBC = 21) (h3 : angle ACB = 39) : angle ABC = 81 := 
sorry

end find_angle_ABC_l352_352898


namespace probability_no_two_adjacent_same_roll_l352_352760

theorem probability_no_two_adjacent_same_roll :
  let total_rolls := 6^5 in
  let valid_rolls := 875 in
  (valid_rolls : ℚ) / total_rolls = 875 / 1296 :=
by
  sorry

end probability_no_two_adjacent_same_roll_l352_352760


namespace binom_12_10_l352_352348

theorem binom_12_10 : nat.choose 12 10 = 66 := by
  sorry

end binom_12_10_l352_352348


namespace shape_of_cylindrical_coords_l352_352414

-- Define cylindrical coordinates
structure CylindricalCoordinates :=
  (r θ z : ℝ)

-- Define the conditions
variables (c d : ℝ)

-- State the theorem
theorem shape_of_cylindrical_coords (P : CylindricalCoordinates) 
  (hθ : P.θ = c) (hz : P.z = d) : ∃ L : set CylindricalCoordinates, 
  (∀ Q ∈ L, Q.θ = c ∧ Q.z = d) ∧ (∀ Q1 Q2 ∈ L, Q1 ≠ Q2 → Q1.r ≠ Q2.r) :=
sorry

end shape_of_cylindrical_coords_l352_352414


namespace max_area_triangle_ABC_l352_352044

noncomputable def hyperbola := {p : ℝ × ℝ // ∃ b : ℝ, 0 < b ∧ b < 2 ∧ p.1^2 / (4 - b^2) - p.2^2 / b^2 = 1}

theorem max_area_triangle_ABC : ∀ (b : ℝ), (0 < b) → (b < 2) → 
  (∃ area : ℝ, area = b * sqrt (4 - b^2) ∧ ∀ b_val : ℝ, (0 < b_val) → (b_val < 2) → (b_val * sqrt (4 - b_val^2) ≤ area) ∧ area = 2) :=
begin
  intros b hb1 hb2,
  have area_eq : b * sqrt (4 - b^2) ≤ 2  := sorry,
  use b * sqrt (4 - b^2),
  split,
  { exact rfl, },
  { intros b_val hb_val1 hb_val2,
    exact area_eq, }
end

end max_area_triangle_ABC_l352_352044


namespace angle_between_vectors_eq_pi_div_3_l352_352425

open Real EuclideanSpace

noncomputable theory

variables {V : Type*} [inner_product_space ℝ V]

theorem angle_between_vectors_eq_pi_div_3 
  {a b : V} (h₁ : ∥a∥ = 2) (h₂ : ∥b∥ = 2) 
  (h₃ : ⟪a + 2 • b, a - b⟫ = -2) :
  real.angle a b = real.pi / 3 :=
sorry

end angle_between_vectors_eq_pi_div_3_l352_352425


namespace max_value_diff_func_l352_352981

theorem max_value_diff_func (f : ℝ → ℝ) (a b : ℝ) (h : a < b)
    (hf_diff : ∀ x ∈ set.Icc a b, differentiable_at ℝ f x) :
    ∃ c ∈ set.Icc a b, (∀ x ∈ set.Icc a b, f x ≤ f c) ∧
        (c = a ∨ c = b ∨ (∃ d ∈ set.Icc a b, deriv f d = 0)) :=
begin
  sorry  -- Proof not required, just the statement.
end

end max_value_diff_func_l352_352981


namespace probability_sum_is_18_l352_352507

noncomputable def probability_of_sum_18 : ℚ :=
  let single_die_prob : ℚ := 1 / 6 in
  (single_die_prob ^ 3)

theorem probability_sum_is_18 (d1 d2 d3 : ℕ) (h₁ : d1 ∈ {1, 2, 3, 4, 5, 6})
                                      (h₂ : d2 ∈ {1, 2, 3, 4, 5, 6})
                                      (h₃ : d3 ∈ {1, 2, 3, 4, 5, 6})
                                      (h_sum : d1 + d2 + d3 = 18) :
  probability_of_sum_18 = 1 / 216 :=
by sorry

end probability_sum_is_18_l352_352507


namespace real_solutions_eq31_l352_352376

theorem real_solutions_eq31 :
  ∃ n, (∀ x ∈ Icc (-50 : ℝ) 50, x / 50 = real.cos x ↔ n = 31) := sorry

end real_solutions_eq31_l352_352376


namespace probability_sum_of_three_dice_eq_18_l352_352506

theorem probability_sum_of_three_dice_eq_18 : 
  (∃ (X Y Z : ℕ), 
    (1 ≤ X ∧ X ≤ 6) ∧ 
    (1 ≤ Y ∧ Y ≤ 6) ∧ 
    (1 ≤ Z ∧ Z ≤ 6) ∧ 
    (X + Y + Z = 18)) ↔ (1/216) :=
begin
  sorry
end

end probability_sum_of_three_dice_eq_18_l352_352506


namespace value_range_of_f_l352_352627

noncomputable def f (x : ℝ) : ℝ := sin x - cos (x + (Real.pi / 6))

theorem value_range_of_f :
  set.range (λ x, f x) = Icc (-Real.sqrt 3 / 2) (Real.sqrt 3) :=
by
  sorry

end value_range_of_f_l352_352627


namespace base_digit_difference_l352_352850

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l352_352850


namespace circle_equation_and_range_of_a_l352_352014

theorem circle_equation_and_range_of_a :
  (∃ m : ℤ, (x - m)^2 + y^2 = 25 ∧ (abs (4 * m - 29)) = 25) ∧
  (∀ a : ℝ, (a > 0 → (4 * (5 * a - 1)^2 - 4 * (a^2 + 1) > 0 → a > 5 / 12 ∨ a < 0))) :=
by
  sorry

end circle_equation_and_range_of_a_l352_352014


namespace bird_count_l352_352667

def initial_birds : ℕ := 24
def new_birds_1 : ℕ := 37
def new_birds_2 : ℕ := 15
def total_birds : ℕ := initial_birds + new_birds_1 + new_birds_2

theorem bird_count : total_birds = 76 :=
by
  have h1 : 24 + 37 = 61 := by decide
  have h2 : 61 + 15 = 76 := by decide
  rw [← add_assoc] at total_birds
  rw [h1, h2]
  exact rfl

end bird_count_l352_352667


namespace simplify_fraction_l352_352657

theorem simplify_fraction (d : ℝ) : (6 - 5 * d) / 9 - 3 = (-21 - 5 * d) / 9 :=
by
  sorry

end simplify_fraction_l352_352657


namespace marks_per_correct_answer_l352_352524

theorem marks_per_correct_answer (total_questions correct_answers wrong_answer_penalty total_marks : ℕ) 
    (h1 : total_questions = 60)
    (h2 : total_marks = 120)
    (h3 : correct_answers = 36)
    (h4 : wrong_answer_penalty = 1) : 
    (∃ x : ℕ, (correct_answers * x - ((total_questions - correct_answers) * wrong_answer_penalty) = total_marks)) → (x = 4) := 
by
  intro hx
  cases hx with x hx
  have h_total_invalid : total_questions - correct_answers = 60 - 36 := by rw [h1, h3]
  rw [h_total_invalid] at hx
  have h_final_eq : 36 * x - 24 = 120 := hx
  have h_solve : 36 * x = 144 := by linarith
  exact Eq.symm (nat.div_eq_of_eq_mul_left (by norm_num: 0 < 36) h_solve)
  
/-- marks_per_correct_answer theorem shows that the number of marks the student scores 
  for each correct answer is 4 under given conditions -/

end marks_per_correct_answer_l352_352524


namespace chessboard_distances_equality_l352_352737

open EuclideanGeometry

variable (A B : Fin 32 → Point) (P : Point)

theorem chessboard_distances_equality (hW : ∀ i : Fin 32, A i ∈ WhiteSquares) (hB : ∀ i : Fin 32, B i ∈ BlackSquares) :
    (∑ i : Fin 32, dist (A i) P) ^ 2 = (∑ i : Fin 32, dist (B i) P) ^ 2 :=
  sorry

end chessboard_distances_equality_l352_352737


namespace sally_has_more_cards_l352_352949

def SallyInitial : ℕ := 27
def DanTotal : ℕ := 41
def SallyBought : ℕ := 20
def SallyTotal := SallyInitial + SallyBought

theorem sally_has_more_cards : SallyTotal - DanTotal = 6 := by
  sorry

end sally_has_more_cards_l352_352949


namespace sally_score_is_12_5_l352_352514

-- Conditions
def correctAnswers : ℕ := 15
def incorrectAnswers : ℕ := 10
def unansweredQuestions : ℕ := 5
def pointsPerCorrect : ℝ := 1.0
def pointsPerIncorrect : ℝ := -0.25
def pointsPerUnanswered : ℝ := 0.0

-- Score computation
noncomputable def sallyScore : ℝ :=
  (correctAnswers * pointsPerCorrect) + 
  (incorrectAnswers * pointsPerIncorrect) + 
  (unansweredQuestions * pointsPerUnanswered)

-- Theorem to prove Sally's score is 12.5
theorem sally_score_is_12_5 : sallyScore = 12.5 := by
  sorry

end sally_score_is_12_5_l352_352514


namespace least_sum_of_exponents_of_powers_of_2_l352_352489

theorem least_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ s : Finset ℕ, (∑ x in s, 2^x = n) ∧ (∀ t : Finset ℕ, (∑ x in t, 2^x = n) → s.sum id ≤ t.sum id) :=
sorry

end least_sum_of_exponents_of_powers_of_2_l352_352489


namespace cucumber_to_tomato_ratio_l352_352307

variable (total_rows : ℕ) (space_per_row_tomato : ℕ) (tomatoes_per_plant : ℕ) (total_tomatoes : ℕ)

/-- Aubrey's Garden -/
theorem cucumber_to_tomato_ratio (total_rows_eq : total_rows = 15)
  (space_per_row_tomato_eq : space_per_row_tomato = 8)
  (tomatoes_per_plant_eq : tomatoes_per_plant = 3)
  (total_tomatoes_eq : total_tomatoes = 120) :
  let total_tomato_plants := total_tomatoes / tomatoes_per_plant
  let rows_tomato := total_tomato_plants / space_per_row_tomato
  let rows_cucumber := total_rows - rows_tomato
  (2 * rows_tomato = rows_cucumber)
:=
by
  sorry

end cucumber_to_tomato_ratio_l352_352307


namespace no_A_i_subsets_bound_l352_352910

variables (S : Finset ℕ) (n k : ℕ) (a : Fin k → ℕ)
variables (A : Fin k → Finset ℕ)

-- Conditions
def valid_conditions (S : Finset ℕ) (n k : ℕ) (a : Fin k → ℕ) (A : Fin k → Finset ℕ) : Prop :=
  (S.card = n) ∧ 
  (2 ≤ k) ∧ 
  (∀ i, 1 ≤ a i) ∧ 
  (∀ i, (A i).card = a i) ∧ 
  (∀ i1 i2, i1 ≠ i2 → A i1 ≠ A i2)

-- The statement to be proven
theorem no_A_i_subsets_bound {S : Finset ℕ} {n k : ℕ} {a : Fin k → ℕ} {A : Fin k → Finset ℕ}
  (h_valid : valid_conditions S n k a A) : 
  (∃ B : ℕ, B ≥ 2^n * (∏ i, (1 - 1 / 2^(a i)))) :=
by
  sorry

end no_A_i_subsets_bound_l352_352910


namespace union_A_B_eq_A_union_B_l352_352066

def A : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def B : Set ℝ := { x | x > 3 / 2 }

theorem union_A_B_eq_A_union_B :
  (A ∪ B) = { x | -1 ≤ x } :=
by
  sorry

end union_A_B_eq_A_union_B_l352_352066


namespace cucumber_weight_l352_352382

theorem cucumber_weight (initial_weight : ℝ) (initial_water_pct : ℝ) (final_water_pct : ℝ) : 
  initial_weight = 100 ∧ initial_water_pct = 0.99 ∧ final_water_pct = 0.98 → 
  let solid_weight := initial_weight * (1 - initial_water_pct) in
  let final_weight := solid_weight / (1 - final_water_pct) in
  final_weight = 50 :=
begin
  intros h,
  cases h with h1 h2,
  cases h2 with h2 h3,
  let solid_weight := h1 * (1 - h2),
  let final_weight := solid_weight / (1 - h3),
  sorry -- Proof here
end

end cucumber_weight_l352_352382


namespace boat_travel_distance_downstream_l352_352670

theorem boat_travel_distance_downstream :
  ∀ (speed_boat_still_water : ℝ)
    (speed_stream : ℝ)
    (time_downstream : ℝ),
    speed_boat_still_water = 40 →
    speed_stream = 5 →
    time_downstream = 1 →
    let speed_boat_downstream := speed_boat_still_water + speed_stream in
    let distance_downstream := speed_boat_downstream * time_downstream in
    distance_downstream = 45 :=
by
  intros speed_boat_still_water speed_stream time_downstream
  intros h1 h2 h3
  simp [h1, h2, h3]
  exact sorry

end boat_travel_distance_downstream_l352_352670


namespace number_of_ways_to_put_cousins_in_rooms_l352_352118

/-- Given 5 cousins and 4 identical rooms, the number of distinct ways to assign the cousins to the rooms is 52. -/
theorem number_of_ways_to_put_cousins_in_rooms : 
  let num_cousins := 5
  let num_rooms := 4
  number_of_ways_to_put_cousins_in_rooms num_cousins num_rooms := 52 :=
sorry

end number_of_ways_to_put_cousins_in_rooms_l352_352118


namespace find_k_l352_352050

def vector := (ℝ × ℝ)   -- Define type for 2D vectors

-- Define the specific vectors given in the problem
def a : vector := (1, 2)
def b : vector := (-3, 2)

-- Define the dot product of two vectors
def dot_product (v1 v2 : vector) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Define the operations with vectors
def scale (k : ℝ) (v : vector) : vector :=
  (k * v.1, k * v.2)

def add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

-- The key condition in the given problem
def k_condition (k : ℝ) : Prop :=
  dot_product (add (scale k a) b) (add a (scale (-3) b)) = 0

-- The theorem we need to prove
theorem find_k : ∃ k, k_condition k ∧ k = 19 := by
  use 19
  dsimp [k_condition, add, scale, dot_product, a, b]
  norm_num
  sorry

end find_k_l352_352050


namespace exam_standard_deviation_l352_352773

-- Define the mean score
def mean_score : ℝ := 74

-- Define the standard deviation and conditions
def standard_deviation (σ : ℝ) : Prop :=
  mean_score - 2 * σ = 58

-- Define the condition to prove
def standard_deviation_above_mean (σ : ℝ) : Prop :=
  (98 - mean_score) / σ = 3

theorem exam_standard_deviation {σ : ℝ} (h1 : standard_deviation σ) : standard_deviation_above_mean σ :=
by
  -- proof is omitted
  sorry

end exam_standard_deviation_l352_352773


namespace simplify_expression_l352_352589

theorem simplify_expression :
  (1 / (1 / (1 / 3 : ℝ)^1 + 1 / (1 / 3)^2 + 1 / (1 / 3)^3)) = (1 / 39 : ℝ) :=
by
  sorry

end simplify_expression_l352_352589


namespace problem_PAB_sliding_l352_352544

variables {P A B M N : Type} [MetricSpace P] [MetricSpace A] [MetricSpace B] [MetricSpace M] [MetricSpace N]

def midpoint (X Y : P) : P := sorry
def length (X Y : P) : ℝ := sorry
def area_triangle (X Y Z : P) : ℝ := sorry
def area_trapezoid (X Y Z W : P) : ℝ := sorry

noncomputable def triangle_PAB_condition (P A B : P) (k : ℝ) :=
  ∃ M N : P, midpoint P A = M ∧ midpoint P B = N ∧ length P A + length P B = k

theorem problem_PAB_sliding (P A B M N : P) (k : ℝ) :
  triangle_PAB_condition P A B k →
  (∀ (MN_does_not_change : length M N = length M N),
  ∀ (perimeter_does_not_change : length P A + length A B + length P B = k + length A B),
  ∃ (area_triangle_changes : ∃ h₁ h₂ : ℝ, area_triangle P A B h₁ ≠ area_triangle P A B h₂),
  ∃ (area_trapezoid_changes : ∃ h₁ h₂ : ℝ, area_trapezoid A B M N h₁ ≠ area_trapezoid A B M N h₂) ) :=
by sorry

end problem_PAB_sliding_l352_352544


namespace complex_in_third_quadrant_l352_352536

theorem complex_in_third_quadrant (a : ℝ) :
  let z := (1 + complex.I) * (a - complex.I) in
  z.re < 0 ∧ z.im < 0 ↔ a < -1 :=
by
  sorry

end complex_in_third_quadrant_l352_352536


namespace min_value_correct_l352_352397

noncomputable def min_value_expression : ℝ :=
  let expr : ℝ → ℝ := λ x, (15 - x) * (13 - x) * (15 + x) * (13 + x)
  in min (expr (sqrt 197)) (expr (-sqrt 197))

theorem min_value_correct :
  min_value_expression = -961 :=
by
  sorry

end min_value_correct_l352_352397


namespace calc_triple_hash_30_l352_352371

def hash_fn (N : ℝ) : ℝ := 0.6 * N + 2

theorem calc_triple_hash_30 :
  hash_fn (hash_fn (hash_fn 30)) = 10.4 :=
by 
  -- Proof goes here
  sorry

end calc_triple_hash_30_l352_352371


namespace probability_white_ball_l352_352078

def total_balls := 3 + 4 + 5
def white_balls := 3

theorem probability_white_ball : 
  (white_balls / total_balls.to_rat) = (1 / 4 : ℚ) :=
by
  -- proof steps will go here
  sorry

end probability_white_ball_l352_352078


namespace quadratic_inequality_cond_l352_352581

theorem quadratic_inequality_cond (a : ℝ) :
  (∀ x : ℝ, ax^2 - ax + 1 > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end quadratic_inequality_cond_l352_352581


namespace median_and_mode_of_scores_l352_352880

theorem median_and_mode_of_scores :
  ∀ (x y : ℕ),
  (2 + x + y + 1 = 10) →
  (23 * 10 = 230) →
  (30 * 2 + 25 * x + 20 * y + 15 * 1 = 230) →
  let scores := [15] ++ list.replicate y 20 ++ list.replicate x 25 ++ list.replicate 2 30 in
  let sorted_scores := list.sort (≤) scores in
  (list.length scores = 10) →
  (∀ n, sorted_scores = [15, 20, 20, 20, 20, 25, 25, 25, 30, 30]) →
  (sorted_scores.nth 4 + sorted_scores.nth 5) / 2 = 22.5 ∧
  (∃ freq, freq = 20 ∧ list.count sorted_scores 20 = freq) := 
by sorry

end median_and_mode_of_scores_l352_352880


namespace nature_reserve_percentage_hawks_l352_352076

theorem nature_reserve_percentage_hawks :
  ∃ (percentage_hawks : ℝ),
    (∀ (percentage_birds : ℝ), percentage_birds = 100 ->
    (∀ non_hawks (percentage_non_hawks : ℝ), percentage_non_hawks = percentage_birds - percentage_hawks ->
    (∀ paddyfield_warblers (percentage_paddyfield_warblers : ℝ), percentage_paddyfield_warblers = 0.4 * percentage_non_hawks ->
    (∀ kingfishers (percentage_kingfishers : ℝ), percentage_kingfishers = 0.1 * percentage_non_hawks ->
    (percentage_hawks + percentage_paddyfield_warblers + percentage_kingfishers = 65 -> percentage_hawks = 30))))) :=
by
  sorry

end nature_reserve_percentage_hawks_l352_352076


namespace simplest_quadratic_radical_l352_352227

theorem simplest_quadratic_radical :
  (∀ x y z w : ℝ, x = real.sqrt 12 → y = real.sqrt 15 → z = real.sqrt 8 → w = real.sqrt (1 / 2) →
  real.sqrt 15 ≠ 2 * real.sqrt 3 ∧ real.sqrt 15 ≠ 2 * real.sqrt 2 ∧ real.sqrt 15 ≠ (real.sqrt 2) / 2) :=
begin
  intros x y z w hx hy hz hw,
  rw hx at *,
  rw hy at *,
  rw hz at *,
  rw hw at *,
  sorry
end

end simplest_quadratic_radical_l352_352227


namespace inequality_property_l352_352780

-- Define the conditions
variables {a b : ℝ}
hypotheses (h1 : a + b < 0) (h2 : b > 0)

-- The statement to be proven
theorem inequality_property (h1 : a + b < 0) (h2 : b > 0) : a^2 > -a * b ∧ -a * b > b^2 :=
sorry

end inequality_property_l352_352780


namespace problem_l352_352444

theorem problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_sum : a + b + c = 1) :
  (1 / a - 1) * (1 / b - 1) * (1 / c - 1) ≥ 8 :=
by
  sorry

end problem_l352_352444


namespace find_a_max_b_l352_352451

variable (f : ℝ → ℝ := λ x, Real.exp x + Real.exp (-x) - 2 * Real.log x)

theorem find_a {a : ℕ} (h1 : a ≥ 2) (h2 : ∃ x₀ ∈ Ioo (1/2 : ℝ) 1, deriv (λ x => Real.exp x + Real.exp (-x) - a * Real.log x) x₀ = 0) :
  a = 2 := by
sorry

theorem max_b (b : ℤ) (h : ∀ x > 0, Real.exp x + Real.exp (-x) - 2 * Real.log x ≥ b) :
  b ≤ 3 := by
sorry

end find_a_max_b_l352_352451


namespace solve_series_sum_eq_l352_352748

theorem solve_series_sum_eq :
  ∃ (n : ℕ), (n > 0) ∧ ((1 + 3 + 5 + ⋯ + (2 * n - 1)) / (2 + 4 + 6 + ⋯ + 2 * n) = 101 / 102) := begin
  sorry
end

end solve_series_sum_eq_l352_352748


namespace proof_problem_find_P_l352_352958

theorem proof_problem (a b : ℝ) (h1 : a + 1 ≠ 0) (h2 : b - 1 ≠ 0) (h3 : a - b + 2 ≠ 0) 
    (h4 : a + 1 ≠ 0) (h5 : b + 1 ≠ 0) (cond : a + 1 ≠ 0 ∧ b - 1 ≠ 0 ∧ a - b + 2 ≠ 0) :
    a + 1 ≠ 0 ∧ b - 1 ≠ 0 ∧ a - b + 2 ≠ 0 ↔
    a - 1 ≠ 0 ∧ b - 1 ≠ 0 ∧ a + b = 2.

theorem find_P (a b : ℝ) (h1 : a ≠ -1) (h2 : b ≠ 1) (h3 : a - b + 2 ≠ 0) 
    (h4 : a + 1 ≠ 0) (h5 : b - 1 ≠ 0) 
    (h : a + 1 ≠ 0 → b - 1 ≠ 0 → a - b + 2 ≠ 0) 
    (h_cond : a + 1 = b + 1 + ((ab- a + a ) → ((a+1) = (a+2) ↔ (ab-a+b =2))) : 
    ab-a+b = 2 := by
  
  sorry

end proof_problem_find_P_l352_352958


namespace matrix_power_four_l352_352336

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l352_352336


namespace cricket_overs_played_initially_l352_352882

variables (x y : ℝ)

theorem cricket_overs_played_initially 
  (h1 : y = 3.2 * x)
  (h2 : 262 - y = 5.75 * 40) : 
  x = 10 := 
sorry

end cricket_overs_played_initially_l352_352882


namespace sqrt_of_16_is_4_l352_352664

def arithmetic_square_root (x : ℕ) : ℕ :=
  if x = 0 then 0 else Nat.sqrt x

theorem sqrt_of_16_is_4 : arithmetic_square_root 16 = 4 :=
by
  sorry

end sqrt_of_16_is_4_l352_352664


namespace certain_event_l352_352654

-- Define each event conditionally
def EventA : Prop := ∃ (seat_number : ℕ), seat_number % 2 = 0
def EventB : Prop := ∃ (time : ℕ), TV_channel(time) = "The Reader"
def EventC : Prop := ∀ (oil : Substance) (water : Substance), (oil.density < water.density) → oil.floats_on(water)
def EventD : Prop := ∃ (time : ℕ), Sun_position(time) = "west"

-- Define normal substances and their properties
structure Substance :=
  (density : ℝ)
  (floats_on : Substance → Prop)

-- Define instance of oil and water
noncomputable def oil : Substance :=
  { density := 0.8, floats_on := λ w, true }

noncomputable def water : Substance :=
  { density := 1.0, floats_on := λ o, o.density < water.density }

-- The theorem that states event C is a certain event
theorem certain_event : EventC :=
by 
  intro oil water h_density
  exact water.floats_on oil

-- Include sorry in place of proof.
sorry


end certain_event_l352_352654


namespace circles_position_relationship_l352_352069

theorem circles_position_relationship 
  (a b : ℝ) 
  (h : dist (0, 0) (ax + by + 1 = 0) = 1 / 2) : 
  let C₁ := {p : ℝ × ℝ | (p.1 - a)^2 + p.2^2 = 1},
      C₂ := {p : ℝ × ℝ | p.1^2 + (p.2 - b)^2 = 1} in
  (distance between centers of C₁ and C₂ = the sum of their radii) :=
begin
  sorry,
end

end circles_position_relationship_l352_352069


namespace ab_ac_bc_range_l352_352560

theorem ab_ac_bc_range (a b c : ℝ) (h : a + b + c = 1) : ab + ac + bc ∈ Icc (-∞) (1/2) := by
  sorry

end ab_ac_bc_range_l352_352560


namespace proof_inequality_l352_352931

variable {a b c d : ℝ}

theorem proof_inequality (h1 : a + b + c + d = 6) (h2 : a^2 + b^2 + c^2 + d^2 = 12) :
  36 ≤ 4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ∧
  4 * (a^3 + b^3 + c^3 + d^3) - (a^4 + b^4 + c^4 + d^4) ≤ 48 :=
sorry

end proof_inequality_l352_352931


namespace general_term_formula_sum_inequality_l352_352434

section
variable (a_n : ℕ → ℕ)
variable (b_n : ℕ → ℝ)
variable (S : ℕ → ℕ)

-- Conditions
def arithmetic_sequence (a_n : ℕ → ℕ) : Prop :=
  ∃ d a_1, ∀ n, a_n = a_1 + (n - 1) * d

def sum_of_first_n_terms (S : ℕ → ℕ) (a_n : ℕ → ℕ) : Prop :=
  ∀ n, S n = n * (a_n 1 + a_n n) / 2

axiom a2_eq_2 : a_n 2 = 2
axiom S11_eq_66 : S 11 = 66
axiom a_n_arithmetic : arithmetic_sequence a_n
axiom S_sum : sum_of_first_n_terms S a_n

-- General term
theorem general_term_formula : ∀ n, a_n n = n := sorry

-- Prove inequality
theorem sum_inequality : ∀ n, (∑ k in range n, b_n k) < 1 :=
by
  have h : ∀ n, b_n n = 1 / (a_n n * a_n (n + 1)) := sorry
  rw [h]
  sorry
end

end general_term_formula_sum_inequality_l352_352434


namespace cubes_not_arithmetic_progression_l352_352214

theorem cubes_not_arithmetic_progression (x y z : ℤ) (h1 : y = (x + z) / 2) (h2 : x ≠ y) (h3 : y ≠ z) : x^3 + z^3 ≠ 2 * y^3 :=
by
  sorry

end cubes_not_arithmetic_progression_l352_352214


namespace expression_evaluation_l352_352062

theorem expression_evaluation (m : ℝ) (h : m = Real.sqrt 2023 + 2) : m^2 - 4 * m + 5 = 2024 :=
by sorry

end expression_evaluation_l352_352062


namespace ratio_boys_girls_l352_352075

variable (S G : ℕ)

theorem ratio_boys_girls (h : (2 / 3 : ℚ) * G = (1 / 5 : ℚ) * S) :
  (S - G) * 3 = 7 * G := by
  -- Proof goes here
  sorry

end ratio_boys_girls_l352_352075


namespace some_students_not_club_members_l352_352511

-- Definitions for the sets
variable {students : Type}
variable (dishonest_students club_members honest_students : Set students)

-- Conditions
axiom some_students_dishonest : ∃ s, s ∈ dishonest_students
axiom all_club_members_honest : ∀ c, c ∈ club_members → c ∈ honest_students

-- The statement we need to prove
theorem some_students_not_club_members 
  (h1 : some_students_dishonest)
  (h2 : all_club_members_honest) : ∃ s, s ∈ (students \ club_members) := 
sorry

end some_students_not_club_members_l352_352511


namespace exists_perfect_square_in_sequence_l352_352791

noncomputable def p (c : ℕ) : ℕ := 
  if h : c > 1 
  then Nat.PrimeFactors.c (Finset.max' (Nat.factors c) sorry)
  else 1

def sequence (a : ℕ → ℕ) : Prop :=
  ∃ a0 : ℕ, a0 > 1 ∧ a = λ n, Nat.recOn n a0 (λ n an, an + p an)

theorem exists_perfect_square_in_sequence (a : ℕ → ℕ) (h : sequence a) : 
  ∃ n : ℕ, ∃ m : ℕ, a n = m * m := 
sorry

end exists_perfect_square_in_sequence_l352_352791


namespace ratio_of_a_b_l352_352535

theorem ratio_of_a_b (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b)
  (hC1 : ∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1)
  (hC2 : ∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1)
  (h_center : (0, Real.sqrt 2 * a)) : a / b = 2 + Real.sqrt 3 := 
sorry

end ratio_of_a_b_l352_352535


namespace total_people_museum_l352_352201

-- Conditions
def first_bus_people : ℕ := 12
def second_bus_people := 2 * first_bus_people
def third_bus_people := second_bus_people - 6
def fourth_bus_people := first_bus_people + 9

-- Question to prove
theorem total_people_museum : first_bus_people + second_bus_people + third_bus_people + fourth_bus_people = 75 :=
by
  -- The proof is skipped but required to complete the theorem
  sorry

end total_people_museum_l352_352201


namespace count_three_digit_odd_integers_l352_352056

theorem count_three_digit_odd_integers : 
  (set.count {n : ℕ | 100 ≤ n ∧ n ≤ 999 ∧ n % 2 = 1}) = 450 :=
sorry

end count_three_digit_odd_integers_l352_352056


namespace limit_arctan_not_exist_l352_352402

open Real

noncomputable def limit_right_arctan (f : ℝ → ℝ) : Prop :=
  limit_at (λ x : ℝ, arctan (1 / (x - 1))) 1 (1 : ℝ) = π / 2

noncomputable def limit_left_arctan (f : ℝ → ℝ) : Prop :=
  limit_at (λ x : ℝ, arctan (1 / (x - 1))) 1_ (1 : ℝ) = -π / 2

theorem limit_arctan_not_exist (f : ℝ → ℝ) :
  limit_right_arctan f ∧ limit_left_arctan f → ¬ limit_at (λ x : ℝ, arctan (1 / (x - 1))) 1 f :=
by
  sorry

end limit_arctan_not_exist_l352_352402


namespace cousins_in_rooms_l352_352134

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l352_352134


namespace extra_yellow_balls_dispatched_l352_352284

theorem extra_yellow_balls_dispatched :
  ∀ (initial_total : ℕ) (initial_ratio : ℚ) (dispatched_ratio : ℚ) (total_dispatched : ℕ),
    initial_total = 175 →
    initial_ratio = 1 / 1 →
    dispatched_ratio = 7 / 11 →
    total_dispatched = 175 →
    let initial_yellow := initial_total / 2,
        dispatched_yellow := (dispatched_ratio.denom * total_dispatched) / (dispatched_ratio.num + dispatched_ratio.denom)
    in dispatched_yellow - initial_yellow = 12 :=
by
  intros initial_total initial_ratio dispatched_ratio total_dispatched
  intro h_initial_total h_initial_ratio h_dispatched_ratio h_total_dispatched
  let initial_yellow := initial_total / 2
  have h1 : initial_yellow = 87, by
    rw [h_initial_total]
    exact Nat.div_le_self 175 2
  let dispatched_yellow := (dispatched_ratio.denom * total_dispatched) / (dispatched_ratio.num + dispatched_ratio.denom)
  have h2 : dispatched_yellow = 99, by
    rw [h_dispatched_ratio, h_total_dispatched]
    norm_num  -- simplifies the computations to yield 99
  suffices dispatched_yellow - initial_yellow = 12, from sorry,
  rw [h1, h2],
  norm_num
  sorry

end extra_yellow_balls_dispatched_l352_352284


namespace solution_l352_352383

open Real

noncomputable def problem : Prop :=
  (∀ n : ℕ, ∀ x : ℝ, 0 ≤ x ∧ x < π / 2 → 
    (floor (n * sin x) ≤ n * sin x) ∧
    (integral (0..π/2) (λ x, (floor (n * sin x)) / n) = integral (0..π/2) sin) → 
    (lim_{n→∞} ∫ (x : ℝ) in 0..(π / 2), (floor (n * sin x)) / n = 1))

theorem solution : problem :=
sorry

end solution_l352_352383


namespace rationalize_denominator_l352_352150

-- Defining the cube root function to work within Lean
noncomputable def cube_root (x : ℝ) : ℝ := x ^ (1 / 3 : ℝ)

-- Stating the problem as a theorem
theorem rationalize_denominator :
  let x := 216 in
  let num := 4 in
  let denom := cube_root x in
  (num / denom) = (2 / 3) :=
by
  -- Placeholder for the proof
  sorry

end rationalize_denominator_l352_352150


namespace vector_angle_solution_l352_352834

noncomputable def vector_angle_problem : Prop :=
  let a := (Real.sqrt 3, 1)
  let b := (Real.sqrt 3, -3) in  -- since we find x = sqrt(3) from conditions
  let sub_ab := (0, 4)  -- a - b
  let dot_product := sub_ab.1 * b.1 + sub_ab.2 * b.2 in
  let norm_ab := Real.sqrt (sub_ab.1^2 + sub_ab.2^2)
  let norm_b := Real.sqrt (b.1^2 + b.2^2) in
  let cos_theta := dot_product / (norm_ab * norm_b) in
  cos_theta = -Real.sqrt 3 / 2 ∧ Real.acos cos_theta = 5 * Real.pi / 6

theorem vector_angle_solution : vector_angle_problem := by 
  sorry

end vector_angle_solution_l352_352834


namespace matrix_pow_four_l352_352324

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l352_352324


namespace minimum_area_of_cyclic_quadrilateral_l352_352217

theorem minimum_area_of_cyclic_quadrilateral :
  ∀ (r1 r2 : ℝ), (r1 = 1) ∧ (r2 = 2) →
    ∃ (A : ℝ), A = 3 * Real.sqrt 3 ∧ 
    (∀ (q : ℝ) (circumscribed : q ≤ A),
      ∀ (p : Prop), (p = (∃ x y z w, 
        ∀ (cx : ℝ) (cy : ℝ) (cr : ℝ), 
          cr = r2 ∧ 
          (Real.sqrt ((x - cx)^2 + (y - cy)^2) = r2) ∧ 
          (Real.sqrt ((z - cx)^2 + (w - cy)^2) = r2) ∧ 
          (Real.sqrt ((x - cx)^2 + (w - cy)^2) = r1) ∧ 
          (Real.sqrt ((z - cx)^2 + (y - cy)^2) = r1)
      )) → q = A) :=
sorry

end minimum_area_of_cyclic_quadrilateral_l352_352217


namespace min_percentage_both_physics_chemistry_l352_352715

/--
Given:
- A certain school conducted a survey.
- 68% of the students like physics.
- 72% of the students like chemistry.

Prove that the minimum percentage of students who like both physics and chemistry is 40%.
-/
theorem min_percentage_both_physics_chemistry (P C : ℝ)
(hP : P = 0.68) (hC : C = 0.72) :
  ∃ B, B = P + C - 1 ∧ B = 0.40 :=
by
  sorry

end min_percentage_both_physics_chemistry_l352_352715


namespace equidistant_point_is_intersection_perp_bisectors_l352_352523

theorem equidistant_point_is_intersection_perp_bisectors {A B C : Type*} [MetricSpace A] [MetricSpace B] [MetricSpace C] (triangle : Triangle A B C) :
  ∃ (P : Type*), (isEquidistant P A ∧ isEquidistant P B ∧ isEquidistant P C) ↔ isIntersectionPerpendicularBisectors P triangle :=
sorry

end equidistant_point_is_intersection_perp_bisectors_l352_352523


namespace sin_15_mul_sin_75_l352_352254

theorem sin_15_mul_sin_75 : Real.sin (15 * Real.pi / 180) * Real.sin (75 * Real.pi / 180) = 1 / 4 :=
by
  sorry

end sin_15_mul_sin_75_l352_352254


namespace exists_m_n_l352_352797

theorem exists_m_n (a b : ℕ) (h_coprime : Nat.Coprime a b) :
  ∃ m n : ℕ, Nat.Totient b = m ∧ Nat.Totient a = n ∧ ab ∣ a^m + b^n - 1 := by
  existsi Nat.Totient b, Nat.Totient a
  sorry

end exists_m_n_l352_352797


namespace problem_statement_l352_352583

theorem problem_statement (x y : ℝ) (h₁ : x + y = 5) (h₂ : x * y = 3) : 
  x + (x^2 / y) + (y^2 / x) + y = 95 / 3 := 
sorry

end problem_statement_l352_352583


namespace count_goats_l352_352884

theorem count_goats (total_animals cows combined_sheep_and_goats : ℕ) :
  total_animals = 200 → 
  cows = 40 → 
  combined_sheep_and_goats = 56 → 
  200 - 40 - combined_sheep_and_goats = 104 :=
by
  intros h_total h_cows h_combined
  rw [h_total, h_cows]
  exact rfl

end count_goats_l352_352884


namespace min_sum_of_exponents_of_powers_of_2_l352_352495

theorem min_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ (s : set ℕ), (∀ (k ∈ s), ∃ (m : ℕ), k = 2 ^ m) ∧ (s.sum id = 520) ∧ (s.sum id = s.card * s.card) → (s.sum id = 12) := sorry

end min_sum_of_exponents_of_powers_of_2_l352_352495


namespace triangle_not_equilateral_l352_352788

theorem triangle_not_equilateral {A B C : Type} [triangle A B C]
  (dist_sides : ∀ (a b c : ℝ), a ≠ b ∧ b ≠ c ∧ c ≠ a)
  (equilateral_ABC1 : is_equilateral A B C_1)
  (equilateral_BCA1 : is_equilateral B C A_1)
  (equilateral_CAB1 : is_equilateral C A B_1) :
  ¬ is_equilateral A_1 B_1 C_1 := 
sorry

end triangle_not_equilateral_l352_352788


namespace infinite_series_sum_l352_352384

theorem infinite_series_sum :
  ∑' n : ℕ, (n > 0) → (3^n / (1 + 3^n + 3^(n + 1) + 3^(2 * n + 1))) = 1 / 4 :=
by sorry

end infinite_series_sum_l352_352384


namespace find_x_value_l352_352835

variable (x : ℝ)
def vec_a : ℝ × ℝ := (1, -1)
def vec_b : ℝ × ℝ := (1, 2)
def vec_c : ℝ × ℝ := (x, 1)
def scalar_mul (k : ℝ) (v : ℝ × ℝ) := (k * v.1, k * v.2)
def dot_product (v1 v2 : ℝ × ℝ) := v1.1 * v2.1 + v1.2 * v2.2

theorem find_x_value
  (h : dot_product (scalar_mul 2 vec_a) (vec_b.1 + vec_c.1, vec_b.2 + vec_c.2) = 0) :
  x = 2 :=
  sorry

end find_x_value_l352_352835


namespace rectangle_ratio_l352_352886

-- Definitions for the conditions
def side_length : ℝ := 3
def midpoint (a b : ℝ) : ℝ := (a + b) / 2
def P : ℝ := midpoint 0 side_length
def Q : ℝ := midpoint (-side_length) 0
def perpendicular (a b : ℝ) : Prop := (a - b) * (a - b) = side_length^2

-- Statement of the problem
theorem rectangle_ratio
  (side_len : ℝ)
  (mid_PQ : ℝ → ℝ → ℝ)
  (P_position Q_position : ℝ)
  (is_perpendicular : ℝ → ℝ → Prop)
  (rearrange_ratio : ℝ)
  (h_side_len : side_len = 3)
  (h_mid_PQ : mid_PQ 0 side_len = P)
  (h_Q_position : Q_position = midpoint (-side_len) 0)
  (h_is_perpendicular : is_perpendicular P Q)
  (h_rearrange : rearrange_ratio = 1) :
  XY / YZ = 1 :=
sorry

end rectangle_ratio_l352_352886


namespace matrix_power_four_l352_352333

def A : Matrix (Fin 2) (Fin 2) ℝ := ![![2, -1], ![1, 1]]

theorem matrix_power_four :
  (A^4) = ![![0, -9], ![9, -9]] :=
by
  sorry

end matrix_power_four_l352_352333


namespace smallest_Q_value_l352_352366

theorem smallest_Q_value (p q r : ℝ) :
  let Q := λ x : ℝ, x^3 + p * x^2 + q * x + r
  let A := -1 + p - q + r
  let B := 1 + p + q + r
  let C := -r
  let D := 1 + p + q + r
  C ≤ A ∧ C ≤ B ∧ C ≤ D :=
by sorry

end smallest_Q_value_l352_352366


namespace school_C_variance_is_12_l352_352620

theorem school_C_variance_is_12 :
  let n_total := 48
  let ratio_A_B_C := (3, 2, 1)
  let avg_score_total := 117
  let var_total := 21.5
  let avg_score_A := 118
  let avg_score_B := 114
  let var_A := 15
  let var_B := 21 in
  let (n_A, n_B, n_C) := (24, 16, 8)
  let avg_score_C := 120 in
  let var_C := 1 / n_total * 
    (n_A * (var_A + (avg_score_A - avg_score_total) ^ 2) +
    n_B * (var_B + (avg_score_B - avg_score_total) ^ 2) +
    n_C * (12 + (avg_score_C - avg_score_total) ^ 2)) in
  var_C = 12 :=
by
  simp only []
  sorry

end school_C_variance_is_12_l352_352620


namespace avg_pencil_cost_l352_352703

theorem avg_pencil_cost 
  (num_pencils : ℕ := 300) 
  (pencil_cost : ℝ := 28.50) 
  (shipping_cost : ℝ := 8.25) 
  (discount_rate : ℝ := 0.10) 
  (total_cost_cents : ℝ := 100 * ((1 - discount_rate) * pencil_cost + shipping_cost))
  (average_cost_cents : ℝ := total_cost_cents / num_pencils)
  : average_cost_cents ≈ 11 := 
  by
    sorry

end avg_pencil_cost_l352_352703


namespace hash_difference_l352_352064

def hash (x y : ℕ) : ℤ := x * y - 3 * x + y

theorem hash_difference :
  (hash 8 5) - (hash 5 8) = -12 :=
by
  sorry

end hash_difference_l352_352064


namespace no_valid_rook_placement_l352_352137

theorem no_valid_rook_placement :
  ∀ (r b g : ℕ), r + b + g = 50 →
  (2 * r ≤ b) →
  (2 * b ≤ g) →
  (2 * g ≤ r) →
  False :=
by
  -- Proof goes here
  sorry

end no_valid_rook_placement_l352_352137


namespace snowball_melt_percentage_l352_352282

-- Define initial conditions
variable (v : ℝ)                     -- Initial velocity of the snowball
variable (k : ℝ) := 0.02 / 100        -- Initial melt percentage (0.02%)
variable (lambda : ℝ) := 330 * 1000   -- Specific heat of fusion of snow in J/kg

-- State the theorem
theorem snowball_melt_percentage :
  let v2 := v / 2 in
  let melt_percent := 1 / 100 in      -- Expected melt percentage when velocity is halved
  (1 / 8) * (v^2) = lambda * (melt_percent / 100) :=
begin
  sorry
end

#check snowball_melt_percentage

end snowball_melt_percentage_l352_352282


namespace distance_focus_directrix_l352_352752

-- Definition of the parabola
def equation_parabola (x y : ℝ) : Prop := y^2 = 10 * x

-- Definition of the focus for the given parabola
def focus (focus_point : ℝ × ℝ) : Prop := focus_point = (5 / 2, 0)

-- Definition of the directrix for the given parabola
def directrix (x : ℝ) : Prop := x = - (5 / 2)

-- Definition of the distance formula between a point and a line
def distance_point_line (A B C x1 y1 : ℝ) : ℝ :=
  (abs (A * x1 + B * y1 + C)) / (real.sqrt (A^2 + B^2))

-- Theorem: The distance from the focus to the directrix of the parabola y^2 = 10x is 5
theorem distance_focus_directrix : ∀ (focus_point : ℝ × ℝ) (x_directrix : ℝ),
  focus focus_point →
  directrix x_directrix →
  distance_point_line 1 0 (- (5 / 2)) (focus_point.fst) (focus_point.snd) = 5 :=
by
  intros focus_point x_directrix h_focus h_directrix
  -- fill in proof here
  sorry

end distance_focus_directrix_l352_352752


namespace area_enclosed_by_curve_and_lines_l352_352165

theorem area_enclosed_by_curve_and_lines : 
  let y1 (x : ℝ) := 1 / x
  let y2 (x : ℝ) := x
  let y3 (x : ℝ) := 2
  let A := (1 : ℝ)
  let B := (2 : ℝ)
  let C := (1 / 2 : ℝ)
  ∫ x in C..A, (2 - y1 x) + ∫ x in A..B, (2 - x) = (3 / 2) - real.log 2 :=
by
  sorry

end area_enclosed_by_curve_and_lines_l352_352165


namespace cousins_in_rooms_l352_352131

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l352_352131


namespace coin_placement_ways_l352_352142

theorem coin_placement_ways :
  let board_rows := 2
  let board_columns := 100
  let coins := 99
  (∀ i j : ℕ, i < board_rows → j < board_columns → no_adjacent i j board_rows board_columns coins) →
  number_of_ways board_rows board_columns coins = 396 := 
by
  sorry

def no_adjacent (i j board_rows board_columns coins : ℕ) : Prop := 
  -- Define the adjacency condition here
  sorry

def number_of_ways (board_rows board_columns coins : ℕ) : ℕ := 
  -- Define the counting function here
  sorry

end coin_placement_ways_l352_352142


namespace find_r_x_l352_352250

open Nat

theorem find_r_x (r n : ℕ) (x : ℕ) (h_r_le_70 : r ≤ 70) (repr_x : x = (10 * r + 6) * (r ^ (2 * n) - 1) / (r ^ 2 - 1))
  (repr_x2 : x^2 = (r ^ (4 * n) - 1) / (r - 1)) :
  (r = 7 ∧ x = 26) :=
by
  sorry

end find_r_x_l352_352250


namespace max_group_size_l352_352614

theorem max_group_size 
  (students_class1 : ℕ) (students_class2 : ℕ) 
  (leftover_class1 : ℕ) (leftover_class2 : ℕ) 
  (h_class1 : students_class1 = 69) 
  (h_class2 : students_class2 = 86) 
  (h_leftover1 : leftover_class1 = 5) 
  (h_leftover2 : leftover_class2 = 6) : 
  Nat.gcd (students_class1 - leftover_class1) (students_class2 - leftover_class2) = 16 :=
by
  sorry

end max_group_size_l352_352614


namespace part1_l352_352562

noncomputable def f (x : ℝ) (b : ℝ) := log x + 2 * x ^ 2 - b * x

theorem part1 (b : ℝ) : (∀ x : ℝ, 0 < x → (1 / x + 4 * x - b) ≥ 0) → b ≤ 4 := 
sorry

end part1_l352_352562


namespace sum_y_coords_l352_352940

-- Define the conditions
def pointQ (y : ℝ) : Prop :=
  let Q : ℝ × ℝ := (4, y) in
  let P : ℝ × ℝ := (-1, -3) in
  (real.sqrt ((-1 - 4)^2 + (-3 - y)^2) = 15)

-- Define the proof problem
theorem sum_y_coords : (∀ y : ℝ, pointQ y → ∃! y1 y2 : ℝ, y1 = y2 ∨ y1 + y2 = -6) :=
by
  sorry

end sum_y_coords_l352_352940


namespace wolves_total_games_l352_352309

theorem wolves_total_games
  (x y : ℕ) -- Before district play, the Wolves had won x games out of y games.
  (hx : x = 40 * y / 100) -- The Wolves had won 40% of their basketball games before district play.
  (hx' : 5 * x = 2 * y)
  (hy : 60 * (y + 10) / 100 = x + 9) -- They finished the season having won 60% of their total games.
  : y + 10 = 25 := by
  sorry

end wolves_total_games_l352_352309


namespace base_digit_difference_l352_352851

theorem base_digit_difference : 
  let n := 1234 in
  let digits_base_4 := Nat.log n 4 + 1 in
  let digits_base_9 := Nat.log n 9 + 1 in
  digits_base_4 - digits_base_9 = 2 :=
by 
  let n := 1234
  let digits_base_4 := Nat.log n 4 + 1
  let digits_base_9 := Nat.log n 9 + 1
  sorry

end base_digit_difference_l352_352851


namespace allowable_combinations_l352_352230

theorem allowable_combinations (shirts ties restricted_pairs : ℕ) 
  (h_shirts : shirts = 8) 
  (h_ties : ties = 7) 
  (h_restricted_pairs : restricted_pairs = 3) : 
  shirts * ties - restricted_pairs = 53 := 
by 
  rw [h_shirts, h_ties, h_restricted_pairs]
  exact Nat.sub_eq_of_eq_add' (by rfl)

end allowable_combinations_l352_352230


namespace eccentricity_of_ellipse_l352_352457

-- Definitions of the ellipses and hyperbolas
def ellipse (a b : ℝ) (x y : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1
def hyperbola (m n : ℝ) (x y : ℝ) : Prop := x^2 / m^2 - y^2 / n^2 = 1

-- The condition of a regular hexagon formed by the asymptotes and intersection points with foci
def regular_hexagon_condition (a b m n : ℝ) : Prop :=
  ∃ c : ℝ, 
    (c^2 = a^2 - b^2) ∧
    (1 / 4 * (c / 2)^2 / a^2 + 3 / 4 * (sqrt 3 * c / 2)^2 / b^2 = 1)

-- The main theorem we want to prove
theorem eccentricity_of_ellipse (a b m n : ℝ) (h1 : a > b) (h2 : b > 0) 
  (h3 : regular_hexagon_condition a b m n) : 
  ∃ e : ℝ, e = (sqrt 3 - 1) := sorry

end eccentricity_of_ellipse_l352_352457


namespace driver_days_off_l352_352206

theorem driver_days_off 
  (drivers : ℕ) 
  (cars : ℕ) 
  (maintenance_rate : ℚ) 
  (days_in_month : ℕ)
  (needed_driver_days : ℕ)
  (x : ℚ) :
  drivers = 54 →
  cars = 60 →
  maintenance_rate = 0.25 →
  days_in_month = 30 →
  needed_driver_days = 45 * days_in_month →
  54 * (30 - x) = needed_driver_days →
  x = 5 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end driver_days_off_l352_352206


namespace replaced_person_weight_l352_352967

theorem replaced_person_weight (new_person_weight : ℝ) (average_increase : ℝ) (num_persons : ℕ)
  (h1 : new_person_weight = 85)
  (h2 : average_increase = 2.5)
  (h3 : num_persons = 8) :
  let weight_increase := average_increase * num_persons in
  let replaced_person_weight := new_person_weight - weight_increase in
  replaced_person_weight = 65 := by
  sorry

end replaced_person_weight_l352_352967


namespace gf_3_eq_495_l352_352097

def f (x : ℝ) : ℝ := x^2 + 4
def g (x : ℝ) : ℝ := 3 * x^2 - x + 1

theorem gf_3_eq_495 : g (f 3) = 495 := by
  sorry

end gf_3_eq_495_l352_352097


namespace cost_of_100_signs_l352_352211

-- Define the conditions given
def height := 7.8 -- in decimeters
def base := 9 -- in decimeters
def price_per_sqm := 90 -- in yuan

-- Use these conditions to state the theorem
theorem cost_of_100_signs :
  let area_decimeters := (base * height) / 2,
      area_meters := area_decimeters * 0.01,
      cost_per_sign := area_meters * price_per_sqm,
      total_cost := cost_per_sign * 100
  in total_cost = 3159 := 
by
  -- Placeholder proof, to be filled in
  sorry

end cost_of_100_signs_l352_352211


namespace conjugate_of_complex_number_l352_352781

open Complex

theorem conjugate_of_complex_number (z : ℂ) (h : z - I = I * z + 3) : conj z = 3 + 2 * I := 
sorry

end conjugate_of_complex_number_l352_352781


namespace perimeter_of_midpoint_quadrilateral_l352_352276

theorem perimeter_of_midpoint_quadrilateral (R : ℝ) (hR : 0 < R) :
    let rectangle_inscribed_in_circle : Type := { rect : ℝ × ℝ // rect.1 = 2 * R ∧ rect.2 = 2 * R }
    let midpoints_form_rhombus : Type := { quad : ℝ × ℝ // quad.1 = R ∧ quad.2 = R ∧ quad.1 = quad.2 }
    let perimeter_rhombus (quad : midpoints_form_rhombus) : ℝ := 4 * quad.1
    ∃ (rect : rectangle_inscribed_in_circle), ∃ (quad : midpoints_form_rhombus), perimeter_rhombus quad = 4 * R :=
by
  sorry

end perimeter_of_midpoint_quadrilateral_l352_352276


namespace first_player_wins_optimal_play_l352_352210

theorem first_player_wins_optimal_play (N : ℕ) (hN : N = 10000000) : 
  ∃ first_player_wins: Prop, first_player_wins :=
by
  -- conditions
  let valid_move := λ n : ℕ, ∃ (p : ℕ) (hprime: Nat.Prime p), ∃ k : ℕ, n = p^k,
  -- assertion
  have first_player_wins: Prop := 
    ∀ (remaining: ℕ), 
      valid_move remaining → 
      (N - remaining) % 6 ≠ 0 → 
      ∃ r (0 ≤ r < 6), (N - r) % 6 = 0
  ⟨first_player_wins, sorry⟩

end first_player_wins_optimal_play_l352_352210


namespace train_length_l352_352696

theorem train_length :
  let speed_kmph := 63
  let time_seconds := 16
  let speed_mps := (speed_kmph * 1000) / 3600
  let length_meters := speed_mps * time_seconds
  length_meters = 280 := 
by
  sorry

end train_length_l352_352696


namespace area_of_triangle_le_one_fourth_l352_352552

open Real

noncomputable def area_triangle (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1 / 2) * abs (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))

theorem area_of_triangle_le_one_fourth (t : ℝ) (x y : ℝ) (h_t : 0 < t ∧ t < 1) (h_x : 0 ≤ x ∧ x ≤ 1)
  (h_y : y = t * (2 * x - t)) :
  area_triangle t (t^2) 1 0 x y ≤ 1 / 4 :=
by
  sorry

end area_of_triangle_le_one_fourth_l352_352552


namespace number_of_possible_values_of_a_l352_352959

theorem number_of_possible_values_of_a :
  ∃ (a_values : Finset ℕ), 
    (∀ a ∈ a_values, 3 ∣ a ∧ a ∣ 18 ∧ a ∣ 27 ∧ 0 < a) ∧
    a_values.card = 2 :=
by
  sorry

end number_of_possible_values_of_a_l352_352959


namespace coins_percentage_l352_352645

-- Definitions of the values of each coin in cents
def penny_cents : ℕ := 1
def nickel_cents : ℕ := 5
def dime_cents : ℕ := 10
def quarter_cents : ℕ := 25
def half_dollar_cents : ℕ := 50

-- Definition that calculates the total value of coins in cents
def total_cents : ℕ := 
  penny_cents + nickel_cents + dime_cents + quarter_cents + half_dollar_cents

-- Proof statement: total cents as percentage of one dollar 
theorem coins_percentage (total_value : ℕ) (dollar_value : ℕ) (percentage : ℝ) 
  (h1 : total_value = total_cents)
  (h2 : dollar_value = 100)
  (h3 : percentage = (total_value.to_rat / dollar_value.to_rat) * 100) :
  percentage = 91 := 
by
  sorry

end coins_percentage_l352_352645


namespace trig_identity_l352_352735

open Real

theorem trig_identity : sin (20 * (π / 180)) * cos (10 * (π / 180)) - cos (200 * (π / 180)) * sin (10 * (π / 180)) = 1 / 2 := 
by
  sorry

end trig_identity_l352_352735


namespace milesAndDavisPopcorn_l352_352666

-- Definitions for the given conditions
def tablespoonsToCups (tbsp : ℕ) : ℕ := (tbsp / 2) * 4

def totalPopcornNeeded (joanie mitchell cliff : ℕ) : ℕ := joanie + mitchell + cliff

def totalPopcornProduced (tbsp : ℕ) : ℕ := tablespoonsToCups tbsp

-- The main problem statement in Lean
theorem milesAndDavisPopcorn
  (joanie_cups mitchell_cups cliff_cups total_tbsp : ℕ)
  (H1 : 2 = 4) -- 2 tablespoons of popcorn kernels will make 4 cups of popcorn (used in definition)
  (H2 : joanie_cups = 3)
  (H3 : mitchell_cups = 4)
  (H4 : cliff_cups = 3)
  (H5 : total_tbsp = 8) :
  let total_needed := totalPopcornNeeded 3 4 3,
      total_made := totalPopcornProduced 8 in
  total_made - total_needed = 6 := sorry

end milesAndDavisPopcorn_l352_352666


namespace hyperbola_eqn_line_eqn_perp_nonexistence_l352_352799

section hyperbola

open Real

variables {C : Type*} [metric_space C]

-- Condition (1): Asymptote lines of the hyperbola and semi-axis
def asymptote_lines (y : C → ℝ) (x : C → ℝ) : Prop :=
∀ p : C, y p = x p ∨ y p = -x p

def semi_axis (x : C → ℝ) (x_val : ℝ) : Prop :=
∀ p : C, ∃ k : ℝ, x p = k ∧ abs k = x_val / sqrt 2

-- Proof for equation of hyperbola
theorem hyperbola_eqn (y : C → ℝ) (x : C → ℝ)
  (ha : asymptote_lines y x)
  (hsa : semi_axis x (sqrt 2 / 2)) :
  ∀ p : C, x p ^ 2 - y p ^ 2 = 1 :=
sorry

-- Proof for equation of line l that intersects hyperbola C
theorem line_eqn (y : C → ℝ) {x : C → ℝ}
  (M : C) (hM : x M = -2 ∧ y M = 0)
  (Harea : abs (area_of_triangle O A B) = 2 * sqrt 3) :
  ∃ m : ℝ, (x = λ p, m * y p - 2) ∧ (m = sqrt 21 / 3 ∨ m = -sqrt 21 / 3) :=
sorry

-- Proof for non-existence of line l such that OA ⊥ OB
theorem perp_nonexistence (y : C → ℝ) {x : C → ℝ}
  (Harea : abs (area_of_triangle O A B) = 2 * sqrt 3) :
  ¬ ∃ l : ℝ, OA ⊥ OB :=
sorry

end hyperbola

end hyperbola_eqn_line_eqn_perp_nonexistence_l352_352799


namespace min_value_of_sum_l352_352611

theorem min_value_of_sum (a : ℝ) (h_a1 : a ≠ 1) 
  (m n : ℝ) (h_m : m > 0) (h_n : n > 0) (h_line : m - n * -1 - 1 = 0) 
  (h_cond : (1 : ℝ, -1) ∈ {p : ℝ × ℝ | ∃ a, p = (1, a - 2)} ) :
  ∃ (m n : ℝ), m + n = 1 ∧ (∀ m: ℝ, ∀ n: ℝ m > 0 ∧ n > 0 ∧ m + n = 1 → (1 / m + 2 / n) = 3 + 2 * Real.sqrt 2) :=
by {
  sorry
}

end min_value_of_sum_l352_352611


namespace cube_diagonal_distance_l352_352394

noncomputable def distance_between_diagonals := 
  let A1 := (0, 0, 1 : ℝ × ℝ × ℝ)
  let B := (1, 0, 0 : ℝ × ℝ × ℝ)
  let B1 := (1, 0, 1 : ℝ × ℝ × ℝ)
  let D1 := (1, 1, 1 : ℝ × ℝ × ℝ)
  (real_distance (line_segment A1 B) (line_segment B1 D1))

theorem cube_diagonal_distance :
  distance_between_diagonals = (√3)/3 :=
sorry

end cube_diagonal_distance_l352_352394


namespace rectangle_area_unchanged_l352_352683

-- Define lengths and widths as real numbers
variables {l w : ℝ}

-- Conditions from the problem translated into equations
def condition1 := (l + 7/2) * (w - 3/2) = l * w
def condition2 := (l - 7/2) * (w + 2) = l * w

-- The goal statement that needs to be proved
theorem rectangle_area_unchanged (h1 : condition1) (h2 : condition2) : l * w = 630 :=
by
  sorry

end rectangle_area_unchanged_l352_352683


namespace ball_hits_ground_at_time_l352_352976

theorem ball_hits_ground_at_time :
  ∃ t : ℚ, -9.8 * t^2 + 5.6 * t + 10 = 0 ∧ t = 131 / 98 :=
by
  sorry

end ball_hits_ground_at_time_l352_352976


namespace volume_of_inscribed_cube_l352_352283

theorem volume_of_inscribed_cube : 
  ∀ (r b c : ℝ), 
  (r = 6) ∧ (b = 4*real.sqrt 3) ∧
  (c = (4*real.sqrt 3)^3) →
  c = 192*real.sqrt 3 :=
by
  sorry

end volume_of_inscribed_cube_l352_352283


namespace ratio_of_down_payment_l352_352947

theorem ratio_of_down_payment (C D : ℕ) (daily_min : ℕ) (days : ℕ) (balance : ℕ) (total_cost : ℕ) 
  (h1 : total_cost = 120)
  (h2 : daily_min = 6)
  (h3 : days = 10)
  (h4 : balance = daily_min * days) 
  (h5 : D + balance = total_cost) : 
  D / total_cost = 1 / 2 := 
  by
  sorry

end ratio_of_down_payment_l352_352947


namespace cos_pi_sub_alpha_l352_352026

theorem cos_pi_sub_alpha (α : ℝ) 
  (h1 : α ∈ set.Ioo (π / 2) (3 * π / 2))
  (h2 : tan α = -12 / 5) :
  cos (π - α) = 5 / 13 :=
sorry

end cos_pi_sub_alpha_l352_352026


namespace area_of_triangle_OXO_l352_352917

noncomputable theory
open_locale classical

-- Definitions
variables {O O' P P' X : Type}
variables {dist : O → O' → ℝ}
variables {radius_C : ℝ := 1}
variables {radius_C' : ℝ := 2}

-- Lean 4 statement
theorem area_of_triangle_OXO' (h_tangent1 : dist O O' = radius_C + radius_C')
  (h_tangent2 : dist O O' P' = radius_C * radius_C')
  (h_tangent3 : dist O' O P = radius_C' * radius_C)
  (h_intersect : ∀ O O' P P' X, PointOfIntersection O O' P P' X) :
  area_triangle O X O' = (4 * real.sqrt 2 - real.sqrt 5) / 3 :=
sorry

end area_of_triangle_OXO_l352_352917


namespace train_carriages_l352_352633

theorem train_carriages (num_trains : ℕ) (total_wheels : ℕ) (rows_per_carriage : ℕ) 
  (wheels_per_row : ℕ) (carriages_per_train : ℕ) :
  num_trains = 4 →
  total_wheels = 240 →
  rows_per_carriage = 3 →
  wheels_per_row = 5 →
  carriages_per_train = 
    (total_wheels / (rows_per_carriage * wheels_per_row)) / num_trains →
  carriages_per_train = 4 :=
by
  sorry

end train_carriages_l352_352633


namespace constant_P_dist_l352_352302

theorem constant_P_dist (a b r : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) :
  ∀ (O A B P : ℝ × ℝ), (∃ k : ℝ, A = (-k*a, 0) ∧ B = (k*b, 0)) ∧ (P = (x, y) ∧ x^2 + y^2 = r^2) →
  ∃ C : ℝ, b * ((P.1 + k * a)^2 + P.2^2) + a * ((P.1 - k * b)^2 + P.2^2) = C :=
begin
  sorry
end

end constant_P_dist_l352_352302


namespace base_digit_difference_l352_352842

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l352_352842


namespace find_k_l352_352705

theorem find_k (k : ℝ) : 
  (∃ c1 c2 : ℝ, (2 * c1^2 + 5 * c1 = k) ∧ 
                (2 * c2^2 + 5 * c2 = k) ∧ 
                (c1 > c2) ∧ 
                (c1 - c2 = 5.5)) → 
  k = 12 := 
by
  intros h
  obtain ⟨c1, c2, h1, h2, h3, h4⟩ := h
  sorry

end find_k_l352_352705


namespace lim_seq_sum_eq_half_l352_352104

theorem lim_seq_sum_eq_half (a : ℝ) (h_pos : a > 0)
    (h_term : ∃ (r : ℕ), r < 9 ∧ (18 - 3 * r) = 12 ∧ a^(r) * Nat.choose 9 r = 4) :
    tendsto (λ n : ℕ, ∑ i in Finset.range (n + 1), a ^ (i + 1)) at_top (𝓝 (1 / 2)) :=
sorry

end lim_seq_sum_eq_half_l352_352104


namespace cousins_rooms_distribution_l352_352110

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l352_352110


namespace ice_cream_total_cost_l352_352576

-- Define the costs for each flavor
def cost_chocolate := 2.50
def cost_vanilla := 2.00
def cost_strawberry := 2.25
def cost_mint := 2.20

-- Define the costs for additional options
def cost_waffle_cone := 1.50
def cost_chocolate_chips := 1.00
def cost_fudge := 1.25
def cost_whipped_cream := 0.75

-- Define the quantities chosen by Pierre and his mother
def pierre_chocolate_scoops := 2
def pierre_mint_scoop := 1
def pierre_waffle_cone := 1
def pierre_chocolate_chips := 1

def mother_vanilla_scoops := 2
def mother_strawberry_scoop := 1
def mother_mint_scoop := 1
def mother_waffle_cone := 1
def mother_fudge := 1
def mother_whipped_cream := 1

-- Define the total cost for Pierre's and his mother's ice cream
def pierre_total : Float := (pierre_chocolate_scoops * cost_chocolate) + 
                            (pierre_mint_scoop * cost_mint) + 
                            (pierre_waffle_cone * cost_waffle_cone) + 
                            (pierre_chocolate_chips * cost_chocolate_chips)

def mother_total : Float := (mother_vanilla_scoops * cost_vanilla) + 
                            (mother_strawberry_scoop * cost_strawberry) + 
                            (mother_mint_scoop * cost_mint) + 
                            (mother_waffle_cone * cost_waffle_cone) + 
                            (mother_fudge * cost_fudge) + 
                            (mother_whipped_cream * cost_whipped_cream)

def total_cost : Float := pierre_total + mother_total

theorem ice_cream_total_cost : total_cost = 21.65 :=
by
  sorry

end ice_cream_total_cost_l352_352576


namespace first_number_in_proportion_l352_352067

variable (x y : ℝ)

theorem first_number_in_proportion
  (h1 : x = 0.9)
  (h2 : y / x = 5 / 6) : 
  y = 0.75 := 
  by 
    sorry

end first_number_in_proportion_l352_352067


namespace overall_labor_costs_l352_352547

noncomputable def construction_worker_daily_wage : ℝ := 100
noncomputable def electrician_daily_wage : ℝ := 2 * construction_worker_daily_wage
noncomputable def plumber_daily_wage : ℝ := 2.5 * construction_worker_daily_wage

noncomputable def total_construction_work : ℝ := 2 * construction_worker_daily_wage
noncomputable def total_electrician_work : ℝ := electrician_daily_wage
noncomputable def total_plumber_work : ℝ := plumber_daily_wage

theorem overall_labor_costs :
  total_construction_work + total_electrician_work + total_plumber_work = 650 :=
by
  sorry

end overall_labor_costs_l352_352547


namespace study_days_l352_352707

theorem study_days (chapters worksheets : ℕ) (chapter_hours worksheet_hours daily_study_hours hourly_break
                     snack_breaks_count snack_break time_lunch effective_hours : ℝ)
  (h1 : chapters = 2) 
  (h2 : worksheets = 4) 
  (h3 : chapter_hours = 3) 
  (h4 : worksheet_hours = 1.5) 
  (h5 : daily_study_hours = 4) 
  (h6 : hourly_break = 10 / 60) 
  (h7 : snack_breaks_count = 3) 
  (h8 : snack_break = 10 / 60) 
  (h9 : time_lunch = 30 / 60)
  (h10 : effective_hours = daily_study_hours - (hourly_break * (daily_study_hours - 1)) - (snack_breaks_count * snack_break) - time_lunch)
  : (chapters * chapter_hours + worksheets * worksheet_hours) / effective_hours = 4.8 :=
by
  sorry

end study_days_l352_352707


namespace min_sum_of_exponents_of_powers_of_2_l352_352497

theorem min_sum_of_exponents_of_powers_of_2 (n : ℕ) (h : n = 520) :
  ∃ (s : set ℕ), (∀ (k ∈ s), ∃ (m : ℕ), k = 2 ^ m) ∧ (s.sum id = 520) ∧ (s.sum id = s.card * s.card) → (s.sum id = 12) := sorry

end min_sum_of_exponents_of_powers_of_2_l352_352497


namespace arithmetic_geometric_problem_l352_352020

-- Definitions of arithmetic and geometric sequences
def isArithmetic (a b c d : ℝ) : Prop :=
  b + c = a + d

def isGeometric (a b c d e : ℝ) : Prop :=
  b * d = a * e ∧ b^2 = a * c ∧ c^2 = b * e

-- Problem statement in Lean
theorem arithmetic_geometric_problem (a1 a2 b1 b2 b3 : ℝ) 
  (h1 : isArithmetic 1 a1 a2 4)
  (h2 : isGeometric 1 b1 b2 b3 4) :
  ∃ a1 a2 b2 : ℝ, a1 + a2 = 5 ∧ b2 = 2 ∧ (a1 + a2) / b2 = 5 / 2 := 
by
  -- Specify values for the variables matching the solution
  use [a1 := 2, a2 := 3, b2 := 2]
  sorry

end arithmetic_geometric_problem_l352_352020


namespace fuel_remaining_l352_352225

noncomputable def initialFuel : ℝ := 60
noncomputable def consumptionRate : ℝ := 0.12
noncomputable def distanceTraveled (x : ℝ) : ℝ := x

theorem fuel_remaining (x : ℝ) : 
  let y := initialFuel - consumptionRate * distanceTraveled x in
  y = 60 - 0.12 * x := 
by
  sorry

end fuel_remaining_l352_352225


namespace new_acute_angle_l352_352983

/- Definitions -/
def initial_angle_A (ACB : ℝ) (angle_CAB : ℝ) := angle_CAB = 40
def rotation_degrees (rotation : ℝ) := rotation = 480

/- Theorem Statement -/
theorem new_acute_angle (ACB : ℝ) (angle_CAB : ℝ) (rotation : ℝ) :
  initial_angle_A angle_CAB ACB ∧ rotation_degrees rotation → angle_CAB = 80 := 
by
  intros h
  -- This is where you'd provide the proof steps, but we use 'sorry' to indicate the proof is skipped.
  sorry

end new_acute_angle_l352_352983


namespace f_2013_eq_2_l352_352801

noncomputable def f : ℝ → ℝ := sorry

variables (x : ℝ)

-- Condition 1: f(x+4) - f(x) = 2f(2) for any x in ℝ.
axiom h1 : ∀ x : ℝ, f(x + 4) - f(x) = 2 * f 2

-- Condition 2: The graph of y = f(x-1) is symmetric about x = 1.
-- y=f(x-1) symmetric about x=1 means f(x-1) = f(2-(x-1)), so f(x) = f(2-x).
axiom h2 : ∀ x : ℝ, f(x - 1) = f(2 - (x - 1))

-- Condition 3: f(1) = 2.
axiom h3 : f 1 = 2

-- The goal: f(2013) = 2
theorem f_2013_eq_2 : f 2013 = 2 :=
by sorry

end f_2013_eq_2_l352_352801


namespace weight_of_replaced_person_l352_352969

theorem weight_of_replaced_person (avg_increase : ℝ) (num_persons : ℕ) (new_person_weight : ℝ) :
  (num_persons = 8) →
  (avg_increase = 2.5) →
  (new_person_weight = 85) →
  ∃ W : ℝ, new_person_weight - (num_persons * avg_increase) = W ∧ W = 65 := 
by
  intros h1 h2 h3
  use new_person_weight - (num_persons * avg_increase)
  split
  { rw [h1, h2, h3], norm_num }
  { norm_num, }

end weight_of_replaced_person_l352_352969


namespace part1_part2_l352_352421

-- Part (1)
theorem part1 (a b : ℝ × ℝ) (h_a : a = (1, -2)) (h_perp : a.1 * b.1 + a.2 * b.2 = 0) (h_norm : b.1 * b.1 + b.2 * b.2 = 20) :
    b = (4, 2) ∨ b = (-4, -2) :=
sorry

-- Part (2)
theorem part2 (a b : ℝ × ℝ) (k : ℝ) (h_a : a = (1, -2)) (h_b : b = (-3, 2)) 
  (h_parallel : ∃ λ : ℝ, (k * a.1 - b.1, k * a.2 - b.2) = (λ * (a.1 + 2 * b.1), λ * (a.2 + 2 * b.2))) :
    k = -1/2 :=
sorry

end part1_part2_l352_352421


namespace johnny_distance_walked_l352_352937

variables (d r_m r_j t_m t_j D_j : ℕ)

theorem johnny_distance_walked :
  d = 45 →
  r_m = 3 →
  r_j = 4 →
  t_m = t_j + 1 →
  D_j = r_j * t_j →
  (r_m * t_m) + D_j = d →
  D_j = 24 :=
by
  intro h1 h2 h3 h4 h5 h6
  rw [h1, h2, h3] at h6
  sorry

end johnny_distance_walked_l352_352937


namespace range_of_m_l352_352873

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, mx^2 - mx - 1 < 0) ↔ m ∈ Icc (-4 : ℝ) 0 :=
by
  sorry

end range_of_m_l352_352873


namespace real_root_of_cubic_l352_352141

theorem real_root_of_cubic (c d : ℝ) (h₁ : (c * Complex.I^3 + 4 * Complex.I^2 + d * Complex.I - 100) = 0) 
                            (h₂ : (c * (-3 - 4 * Complex.I)^3 + 4 * (-3 - 4 * Complex.I)^2 + d * (-3 - 4 * Complex.I) - 100) = 0) : 
                            ∃ x : ℝ, (c * x^3 + 4 * x^2 + d * x - 100) = 0 ∧ x = -4 := 
begin
  sorry
end

end real_root_of_cubic_l352_352141


namespace range_of_a_l352_352923

def f (x : ℝ) : ℝ := real.exp x - real.exp (-x) - 2 * x

theorem range_of_a (a : ℝ) : 
  (f(a - 3) + f(2 * a^2) ≤ 0) ↔ (-3 / 2 ≤ a ∧ a ≤ 1) := by
sorry

end range_of_a_l352_352923


namespace sales_target_proof_l352_352569

-- Definitions based on conditions
def last_month_sales_wireless := 48
def increase_percent_wireless := 12.5

def last_month_sales_optical := 24
def increase_percent_optical := 37.5

def last_month_sales_trackball := 8
def increase_percent_trackball := 20.0

-- Sales calculations
def sales_target_wireless := 
  last_month_sales_wireless + (last_month_sales_wireless * increase_percent_wireless / 100)

def sales_target_optical := 
  last_month_sales_optical + (last_month_sales_optical * increase_percent_optical / 100)

def sales_target_trackball := 
  float.round (last_month_sales_trackball + (last_month_sales_trackball * increase_percent_trackball / 100))

-- Proof statement to achieve sales target
theorem sales_target_proof :
  sales_target_wireless = 54 ∧ 
  sales_target_optical = 33 ∧ 
  sales_target_trackball = 10 :=
  by
    sorry

end sales_target_proof_l352_352569


namespace find_divisor_l352_352647

theorem find_divisor (dividend quotient remainder divisor : ℕ) 
    (h_dividend : dividend = 166)
    (h_quotient : quotient = 9)
    (h_remainder : remainder = 4)
    (h_divisor_is_18 : divisor = 18) :
    dividend = (divisor * quotient) + remainder :=
by
  -- We prove the statement by substituting the given conditions and performing the necessary arithmetic
  rw [h_dividend, h_quotient, h_remainder, h_divisor_is_18]
  norm_num

end find_divisor_l352_352647


namespace smallest_number_of_points_in_T_l352_352686

def symmetric_points (T : Set (ℝ × ℝ)) : Prop :=
  (∀ (p : ℝ × ℝ), p ∈ T → (-p.1, -p.2) ∈ T) ∧
  (∀ (p : ℝ × ℝ), p ∈ T → (p.1, -p.2) ∈ T) ∧
  (∀ (p : ℝ × ℝ), p ∈ T → (-p.1, p.2) ∈ T) ∧
  (∀ (p : ℝ × ℝ), p ∈ T → (p.2, p.1) ∈ T) ∧
  (∀ (p : ℝ × ℝ), p ∈ T → (-p.2, -p.1) ∈ T)

theorem smallest_number_of_points_in_T :
  ∀ (T : Set (ℝ × ℝ)), (3, 4) ∈ T → symmetric_points T → ∃ (n : ℕ), n = 8 ∧ ∀ (U : Set (ℝ × ℝ)), (3, 4) ∈ U → symmetric_points U → (#U ≤ n) :=
by
  sorry

end smallest_number_of_points_in_T_l352_352686


namespace solve_for_x_l352_352155

theorem solve_for_x (x : ℤ) : 27 - 5 = 4 + x → x = 18 :=
by
  intro h
  sorry

end solve_for_x_l352_352155


namespace exactly_one_satisfies_conditions_l352_352035

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

def condition_two (f : ℝ → ℝ) : Prop :=
  ∀ (x1 x2 : ℝ), 0 < x1 ∧ 0 < x2 ∧ x1 ≠ x2 → (f x1 - f x2) / (x1 - x2) < 0

def functions : List (ℝ → ℝ) :=
  [λ x, x^2 + 1, λ x, -|x|, λ x, (1/2)^x, λ x, Real.log x / Real.log 2]

theorem exactly_one_satisfies_conditions :
  (∃ f ∈ functions, is_even_function f ∧ condition_two f) ∧
  (∀ f ∈ functions, is_even_function f ∧ condition_two f → f = λ x, -|x|) :=
by
  sorry

end exactly_one_satisfies_conditions_l352_352035


namespace probability_no_two_adjacent_same_roll_l352_352757

theorem probability_no_two_adjacent_same_roll :
  let total_rolls := 6^5 in
  let valid_rolls := 875 in
  (valid_rolls : ℚ) / total_rolls = 875 / 1296 :=
by
  sorry

end probability_no_two_adjacent_same_roll_l352_352757


namespace min_value_expression_l352_352400

theorem min_value_expression : ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
begin
  -- We claim that the minimum value of the given expression is -784.
  use sqrt 197,
  -- Expanding shows that this value occurs and is the minimum.
  sorry
end

end min_value_expression_l352_352400


namespace max_number_of_different_ages_l352_352965

def average_age : ℕ := 20
def stddev : ℕ := 8
def min_age : ℕ := average_age - stddev
def max_age : ℕ := average_age + stddev

theorem max_number_of_different_ages : (max_age - min_age + 1) = 17 := by
  unfold average_age stddev min_age max_age
  -- here we will directly compute the range, but in mathematical proof it will be proven
  have h: max_age - min_age + 1 = ((20 + 8) - (20 - 8) + 1) := by sorry
  rw h; exact 17

end max_number_of_different_ages_l352_352965


namespace convert_polar_to_cartesian_line_convert_parametric_to_cartesian_curve_minimum_value_absolute_difference_l352_352466

variables {α : Type*} [linear_ordered_field α]

-- Definitions of the conditions
def polar_line (ρ θ : α) := sqrt(2) * ρ * cos(θ + π / 4) = 4

def parametric_curve (α : α) : α × α :=
  (2 * cos α, sin α)

def cartesian_equation_curve (x y : α) :=
  (x / 2) ^ 2 + y ^ 2 = 1

-- Theorem statements
theorem convert_polar_to_cartesian_line {ρ θ : α} :
  polar_line ρ θ → x - y - 4 = 0 :=
sorry

theorem convert_parametric_to_cartesian_curve {x y : α} {α : α} :
  parametric_curve α = (x, y) → cartesian_equation_curve x y :=
sorry

theorem minimum_value_absolute_difference {x y : α} :
  cartesian_equation_curve x y → ∃ α, parametric_curve α = (x, y) ∧
  abs (x - y - 4) = 4 - sqrt(5) :=
sorry

end convert_polar_to_cartesian_line_convert_parametric_to_cartesian_curve_minimum_value_absolute_difference_l352_352466


namespace leading_digit_common_l352_352413

theorem leading_digit_common {n : ℕ} (h1 : same_leading_digit (2^n) (5^n)) : leading_digit = 3 :=
sorry

end leading_digit_common_l352_352413


namespace length_O1O2_l352_352991

-- Define the circles and their centers and radii
structure Circle where
  center : Point
  radius : ℝ

-- Define a point in Euclidean space
structure Point where
  x : ℝ
  y : ℝ

-- Define the condition: AT:BT = 1:2
def ratio (A T B : Point) : Prop :=
  let AT := (⬝ distance A T)
  let BT := (⬝ distance B T)
  AT / BT = 1 / 2

-- Define the circles and conditions
def circle1 := Circle.mk (Point.mk 0 0) 4 -- Center O1 at origin with radius 4
def circle2 := Circle.mk (Point.mk d 0) 6 -- Center O2 at (d, 0) with radius 6

-- The point T lies on the segment O1O2
def on_segment (T O1 O2 : Point) : Prop :=
  distance T O1 + distance T O2 = distance O1 O2

-- The length of O1O2 is at most 6 cm
def length_at_least_six (O1 O2 : Point) : Prop :=
  distance O1 O2 ≥ 6

-- Mathematically equivalent proof problem
theorem length_O1O2 
  (O1 O2 A B T : Point)
  (h1 : circle1.center = O1)
  (h2 : circle2.center = O2)
  (h3 : circle1.radius = 4)
  (h4 : circle2.radius = 6)
  (h5 : length_at_least_six O1 O2)
  (h6 : on_segment T O1 O2)
  (h7 : ratio A T B) :
  distance O1 O2 = 6 :=
sorry

end length_O1O2_l352_352991


namespace base_digit_difference_l352_352844

theorem base_digit_difference (n : ℕ) (h1 : n = 1234) : 
  (nat.log 4 n) + 1 - (nat.log 9 n) + 1 = 2 :=
by 
  -- Proof omitted with sorry
  sorry

end base_digit_difference_l352_352844


namespace misread_weight_is_correct_l352_352971

namespace Proof

variable {n : ℕ} (initial_avg correct_avg misread_correct_weight : ℝ)

noncomputable def misread_weight : ℝ :=
  (n * initial_avg - n * correct_avg + misread_correct_weight)

theorem misread_weight_is_correct (h1 : n = 20) 
  (h2 : initial_avg = 58.4) 
  (h3 : correct_avg = 58.9) 
  (h4 : misread_correct_weight = 66) :
  misread_weight 20 58.4 58.9 66 = 56 := by
  unfold misread_weight
  sorry

end Proof

end misread_weight_is_correct_l352_352971


namespace infinite_sequence_of_squares_fits_into_square_l352_352436

noncomputable def smallest_side_length_of_square_containing_sequence := 1.5

theorem infinite_sequence_of_squares_fits_into_square :
  ∀ (sequence : ℕ → ℝ), (∀ n, sequence n = 1 / (n + 1)) →
  ∃ (S : ℝ), S = smallest_side_length_of_square_containing_sequence ∧
  (∀ (n : ℕ), ∃ (x y : ℝ), 0 ≤ x ∧ x + sequence n ≤ S ∧ 0 ≤ y ∧ y + sequence n ≤ S) :=
  sorry

end infinite_sequence_of_squares_fits_into_square_l352_352436


namespace line_intersects_axes_l352_352678

theorem line_intersects_axes (a b : ℝ) (x1 y1 x2 y2 : ℝ) (h_points : (x1, y1) = (8, 2) ∧ (x2, y2) = (4, 6)) :
  (∃ x_intercept : ℝ, (x_intercept, 0) = (10, 0)) ∧ (∃ y_intercept : ℝ, (0, y_intercept) = (0, 10)) :=
by
  sorry

end line_intersects_axes_l352_352678


namespace odd_function_has_specific_a_l352_352070

def is_odd (f : ℝ → ℝ) : Prop :=
∀ x, f (-x) = - f x

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
x / ((2 * x + 1) * (x - a))

theorem odd_function_has_specific_a :
  ∀ a, is_odd (f a) → a = 1 / 2 :=
by sorry

end odd_function_has_specific_a_l352_352070


namespace total_books_for_girls_l352_352631

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l352_352631


namespace Toropyzhka_ate_food_l352_352979

-- Define the suspects and their statements
structure Person :=
  (name : String)
  (statement : Prop)

-- Define our suspects
def Siropchik : Person := ⟨"Siropchik", false⟩ -- Siropchik said he did not eat the dog food.
def Toropyzhka : Person := ⟨"Toropyzhka", Siropchik.statement ∨ Ponchik.statement⟩
def Ponchik : Person := ⟨"Ponchik", ¬Siropchik.statement⟩

def ate_food (p : Person) : Bool :=
  match p.name with
  | "Siropchik"  => ¬p.statement
  | "Toropyzhka" => ¬(Siropchik.statement ∨ Ponchik.statement)
  | "Ponchik"    => ¬¬Siropchik.statement
  | _            => false

-- The main theorem asserting the culprit
theorem Toropyzhka_ate_food : ate_food Toropyzhka = true :=
  sorry -- Proof goes here

end Toropyzhka_ate_food_l352_352979


namespace geometric_sequence_product_l352_352081

theorem geometric_sequence_product (a : ℕ → ℝ) (h : ∀ n, a (n + 1) = a 1 * (a 2 / a 1) ^ n)
  (h1 : a 1 * a 4 = -3) : a 2 * a 3 = -3 :=
by
  -- sorry is placed here to indicate the proof is not provided.
  sorry

end geometric_sequence_product_l352_352081


namespace infinite_fixed_points_l352_352913

noncomputable def f : ℕ+ → ℕ+ := sorry

theorem infinite_fixed_points :
  (∀ n : ℕ+, f n - n < 2021) →
  (∀ n : ℕ+, (nat.iterate f (f n) n) = n) →
  ∃ᶠ n in Filter.at_top, f n = n :=
by 
  sorry

end infinite_fixed_points_l352_352913


namespace constant_polynomial_Q_l352_352911

theorem constant_polynomial_Q
  (P Q : ℤ[X])
  (h_nonconst : ∀ R : ℚ[X], (R ∣ P) → (R ∣ Q) → is_constant R)
  (h_pos_P : ∀ n : ℕ, n > 0 → P.eval n > 0)
  (h_pos_Q : ∀ n : ℕ, n > 0 → Q.eval n > 0)
  (h_div : ∀ n : ℕ, n > 0 → (2 ^ (Q.eval n) - 1) ∣ (3 ^ (P.eval n) - 1)) :
  ∃ c : ℤ, ∀ x, Q.eval x = c :=
sorry

end constant_polynomial_Q_l352_352911


namespace probability_at_least_one_woman_l352_352504

theorem probability_at_least_one_woman (total_people : ℕ) (men : ℕ) (women : ℕ) (selected : ℕ)
  (h1 : total_people = 10) (h2 : men = 5) (h3 : women = 5) (h4 : selected = 3) :
  (1 - (men / total_people) * ((men - 1) / (total_people - 1)) * ((men - 2) / (total_people - 2))) = 5 / 6 :=
by
  sorry

end probability_at_least_one_woman_l352_352504


namespace sufficient_but_not_necessary_condition_l352_352973

theorem sufficient_but_not_necessary_condition (x : ℝ) : (0 < x ∧ x < 5) → |x - 2| < 3 :=
by
  sorry

end sufficient_but_not_necessary_condition_l352_352973


namespace divisibility_by_x_plus_1_l352_352824

theorem divisibility_by_x_plus_1 (x y : ℤ) (hx : x ≠ -1) (hy : y ≠ -1) 
  (h : (x^4 - 1) / (y + 1) + (y^4 - 1) / (x + 1) ∈ ℤ) : (x + 1) ∣ (x^4 * y^44 - 1) :=
by
  sorry

end divisibility_by_x_plus_1_l352_352824


namespace parallel_lines_slope_l352_352180

theorem parallel_lines_slope (a : ℝ) :
  (∀ (x y : ℝ), x + a * y + 6 = 0 ∧ (a - 2) * x + 3 * y + 2 * a = 0 → (1 / (a - 2) = a / 3)) →
  a = -1 :=
by {
  sorry
}

end parallel_lines_slope_l352_352180


namespace G_greater_F_l352_352005

theorem G_greater_F (x : ℝ) : 
  let F := 2*x^2 - 3*x - 2
  let G := 3*x^2 - 7*x + 5
  G > F := 
sorry

end G_greater_F_l352_352005


namespace segment_length_is_13_l352_352057

def point := (ℝ × ℝ)

def p1 : point := (2, 3)
def p2 : point := (7, 15)

noncomputable def distance (p1 p2 : point) : ℝ :=
  Real.sqrt ((p2.1 - p1.1) ^ 2 + (p2.2 - p1.2) ^ 2)

theorem segment_length_is_13 : distance p1 p2 = 13 := by
  sorry

end segment_length_is_13_l352_352057


namespace area_of_triangle_AOB_is_two_l352_352027

theorem area_of_triangle_AOB_is_two
    (A B : ℝ × ℝ)
    (O : ℝ × ℝ := (0, 0))
    (H1 : ∃ l : (ℝ × ℝ) → Prop, l A ∧ l B)
    (H2 : A ≠ O ∧ B ≠ O ∧ A ≠ B)
    (H3 : let M := (A.1 + B.1) / 2, (A.2 + B.2) / 2 in M ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 2})
    (H4 : ∀ p : ℝ × ℝ, p ∈ {p : ℝ × ℝ | p.1^2 - p.2^2 = 2} → let k := p.1 in k^2 > 2 ∨ k^2 < 2)
    : (1 / 2) * abs ((A.1 * (B.2 - O.2)) + (B.1 * (O.2 - A.2)) + (O.1 * (A.2 - B.2))) = 2 :=
by
  sorry

end area_of_triangle_AOB_is_two_l352_352027


namespace isosceles_triangle_vertex_angle_l352_352437

theorem isosceles_triangle_vertex_angle (B : ℝ) (V : ℝ) (h1 : B = 70) (h2 : B = B) (h3 : V + 2 * B = 180) : V = 40 ∨ V = 70 :=
by {
  sorry
}

end isosceles_triangle_vertex_angle_l352_352437


namespace problem_solving_ratio_l352_352571

theorem problem_solving_ratio 
  (total_mcqs : ℕ) (total_psqs : ℕ)
  (written_mcqs_fraction : ℚ) (total_remaining_questions : ℕ)
  (h1 : total_mcqs = 35)
  (h2 : total_psqs = 15)
  (h3 : written_mcqs_fraction = 2/5)
  (h4 : total_remaining_questions = 31) :
  (5 : ℚ) / 15 = (1 : ℚ) / 3 := 
by {
  -- given that 5 is the number of problem-solving questions already written,
  -- and 15 is the total number of problem-solving questions
  sorry
}

end problem_solving_ratio_l352_352571


namespace convex_polygon_angles_l352_352147

theorem convex_polygon_angles (n : ℕ) (h : n ≥ 3) (angles : Fin n → ℝ) 
    (h_convex : ∀ i, 0 < angles i ∧ angles i < 180) 
    (sum_angles : ∑ i, angles i = 180 * (n - 2)) :
    ∃ (count : ℕ), count ≤ 35 ∧ ∀ (i : Fin n), angles i < 170 → ∃ (j : Fin n), i = j:= 
sorry

end convex_polygon_angles_l352_352147


namespace measure_of_angle_E_l352_352528

-- Defining the conditions in Lean
variables (angle_N angle_M angle_B angle_R angle_U angle_S angle_E : ℝ)
-- angles in degrees

-- conditions
def angle_condition1 := angle_N = angle_M
def angle_condition2 := angle_B = angle_R
def angle_condition3 := angle_U + angle_S = 180
def hexagon_angles_sum := angle_N + angle_U + angle_M + angle_B + angle_E + angle_R + angle_S = 720

-- Theorem statement
theorem measure_of_angle_E :
  angle_condition1 →
  angle_condition2 →
  angle_condition3 →
  hexagon_angles_sum →
  angle_E = 180 :=
by
  intros h1 h2 h3 h4
  sorry

end measure_of_angle_E_l352_352528


namespace masks_in_package_l352_352300

theorem masks_in_package (parents siblings : ℕ) (days_per_mask total_days : ℕ) 
  (family_members : ℕ := 1 + parents + siblings)
  (days_per_mask_pos : 0 < days_per_mask)
  (family_members_pos : 0 < family_members)
  (total_days_pos : 0 < total_days) :
  total_days / days_per_mask * family_members = 100 := by
  sorry

# Define the conditions
def andrew_parents : ℕ := 2
def andrew_siblings : ℕ := 2
def change_frequency : ℕ := 4
def pack_duration : ℕ := 80

# Conclusion about the number of masks
example : masks_in_package andrew_parents andrew_siblings change_frequency pack_duration := by
  sorry

end masks_in_package_l352_352300


namespace matrix_pow_four_l352_352323

open Matrix

-- Define the matrix A
def A : Matrix (Fin 2) (Fin 2) ℤ := !!
  [ 2, -1,
    1,  1]

-- State the theorem with the final result
theorem matrix_pow_four :
  A ^ 4 = !!
  [ 0, -9,
    9, -9] :=
  sorry

end matrix_pow_four_l352_352323


namespace binom_12_10_eq_66_l352_352337

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l352_352337


namespace path_length_of_B_l352_352712

theorem path_length_of_B (B A D B' : Point) (BD : ℝ)
  (h1 : AD ∘ center = B)
  (h2 : ∃ PQ : Line, roll (ABD) PQ B B')
  (h3 : BD = 4 / π) :
  distance B B' = 8 :=
sorry

end path_length_of_B_l352_352712


namespace matrix_power_four_correct_l352_352321

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l352_352321


namespace sports_club_total_members_l352_352522

variables {Total B T B_and_T N : ℕ}

-- Total = B + T - (B ∩ T) + N
-- B = 17
-- T = 17
-- B ∩ T = 6
-- N = 2
theorem sports_club_total_members :
  B = 17 ∧ T = 17 ∧ B_and_T = 6 ∧ N = 2 →
  Total = B + T - B_and_T + N →
  Total = 30 :=
by
  intros h1 h2
  cases h1 with hB h1,
  cases h1 with hT h1,
  cases h1 with hB_and_T hN,
  rw [hB, hT, hB_and_T, hN] at h2,
  exact h2.symm

end sports_club_total_members_l352_352522


namespace product_of_real_solutions_eq_neg_ten_l352_352755

theorem product_of_real_solutions_eq_neg_ten :
  (∃ x s : ℝ, (1 / (4 * x) = (s - x) / 10) ∧ (∃ k : ℝ, 16 * s^2 - 160 = k ∧ k = 0) →
  s = sqrt 10 ∨ s = -sqrt 10 ∧ ((sqrt 10) * (-sqrt 10) = -10)) :=
by 
  sorry

end product_of_real_solutions_eq_neg_ten_l352_352755


namespace chocolate_chip_cookies_count_l352_352948

-- Definitions and conditions
def totalCookies(baggies: Nat, cookiesPerBag: Nat) : Nat := baggies * cookiesPerBag
def oatmealCookies : Nat := 25
def chocolateChipCookies(totalCookies: Nat, oatmealCookies: Nat) : Nat := totalCookies - oatmealCookies

-- The main statement to prove
theorem chocolate_chip_cookies_count : 
  ∀ (baggies cookiesPerBag : Nat), 
  baggies = 8 → cookiesPerBag = 6 → chocolateChipCookies (totalCookies baggies cookiesPerBag) oatmealCookies = 23 :=
by 
  intros baggies cookiesPerBag h_baggies h_cookiesPerBag
  rw [h_baggies, h_cookiesPerBag]
  rw [totalCookies, chocolateChipCookies]
  simp
  sorry

end chocolate_chip_cookies_count_l352_352948


namespace Kara_jogging_speed_l352_352381

theorem Kara_jogging_speed :
  ∃ s : ℝ, s = π / 3 ∧
  ∀ c d : ℝ, 
    let L_inner := 2 * d + 2 * π * c in
    let L_outer := 2 * d + 2 * π * (c + 8) in
    (L_outer - L_inner) / s = 48 :=
by
  sorry

end Kara_jogging_speed_l352_352381


namespace bolton_class_students_l352_352876

theorem bolton_class_students 
  (S : ℕ) 
  (H1 : 2/5 < 1)
  (H2 : 1/3 < 1)
  (C1 : (2 / 5) * (S:ℝ) + (2 / 5) * (S:ℝ) = 20) : 
  S = 25 := 
by
  sorry

end bolton_class_students_l352_352876


namespace cosine_theta_l352_352474

noncomputable def cosine_between_vectors (a b : ℝ) (h1 : ∥a∥ = 1) (h2 : ∥b∥ = 1) (h3 : ∥a + b∥ = sqrt 3) : ℝ :=
  (a • (a + 2 • b)) / (∥a∥ * ∥a + 2 • b∥) 

theorem cosine_theta (a b : EuclideanSpace ℝ (Fin 3)) 
  (h1 : ∥a∥ = 1) 
  (h2 : ∥b∥ = 1) 
  (h3 : ∥a + b∥ = sqrt 3) : 
  cosine_between_vectors a b h1 h2 h3 = (2 * sqrt 7) / 7 := 
by 
  sorry

end cosine_theta_l352_352474


namespace problem_1_problem_2_l352_352042

noncomputable theory
open Real

def f (a x : ℝ) : ℝ := a * x^3 + 2 * sin x - x * cos x

theorem problem_1 :
  ∀ (x : ℝ), -π / 2 < x ∧ x < π / 2 →
  0 = 0 →
  (f 0 x) is_strictly_increasing_on Ioo (-π / 2) (π / 2)
:=
begin
  intro x,
  intro hx,
  intro h_zero,
  sorry
end

theorem problem_2 (a : ℝ) (h : a > 0) :
  ∀ (x : ℝ), 0 < x ∧ x < π →
  (if a ≥ 1 / (3 * π^2) then (count_extreme_points (f a) Ioo 0 π) = 0 
   else if 0 < a ∧ a < 1 / (3 * π^2) then (count_extreme_points (f a) Ioo 0 π) = 1
   else false)
:=
begin
  intro x,
  intro hx,
  cases classical.em (a >= 1 / (3 * π^2)) with h_ge h_lt,
  { 
    have : a ≥ 1 / (3 * π^2), from h_ge,
    sorry
  },
  { 
    cases classical.em (0 < a ∧ a < 1 / (3 * π^2)) with h_cases h_other,
    { 
      have : 0 < a ∧ a < 1 / (3 * π^2), from h_cases,
      sorry 
    },
    { 
      exfalso,
      apply h_other,
      split,
      { exact h },
      { rw not_lt at h_ge,
        exact h_ge }
    }
  }
end

end problem_1_problem_2_l352_352042


namespace ending_number_set_A_l352_352953

-- Definition for sets A and B
def set_A (n : ℕ) : set ℕ := { x | 4 ≤ x ∧ x ≤ n }
def set_B : set ℕ := { x | 6 ≤ x ∧ x ≤ 20 }

-- Define the condition that there are 10 distinct integers in both sets
def common_elements_condition (n : ℕ) : Prop :=
  (set_A n ∩ set_B).card = 10

-- The statement to prove
theorem ending_number_set_A : ∃ n, common_elements_condition n ∧ n = 15 :=
by {
  use 15,
  split,
  { unfold common_elements_condition,
    sorry
  },
  { reflexivity }
}

end ending_number_set_A_l352_352953


namespace negation_of_proposition_l352_352186

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x)) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
sorry

end negation_of_proposition_l352_352186


namespace tangent_lines_through_P_eqns_l352_352753

-- Define the function and its derivative
def f (x : ℝ) : ℝ := x^3 - 2 * x
def f' (x : ℝ) : ℝ := 3 * x^2 - 2

-- The given point
def P : ℝ × ℝ := (2, 4)

-- Prove the equation of the tangent line passing through P
theorem tangent_lines_through_P_eqns :
  ∃ (m n : ℝ) (k : ℝ), n = f m ∧ k = f' m ∧ (P.snd - n = k * (P.fst - m)) ∧ 
  ((k = 10 ∧ (λ x, k * x - (k * m - n)) = λ x, 10 * x - 16) ∨ 
   (k = 1 ∧ (λ x, k * x - (k * m - n)) = λ x, x + 2)) :=
sorry

end tangent_lines_through_P_eqns_l352_352753


namespace find_k_l352_352606

def point_A := (0, 2)
def point_B := (-1, 0)
def direction_vector := (1, 2)

theorem find_k :
  ∃ k : ℝ, (1, k) = direction_vector :=
by
  use 2
  sorry

end find_k_l352_352606


namespace verify_logarithmic_roots_verify_algebraic_roots_l352_352662

noncomputable def logarithmic_roots : Prop :=
  ∀ x : ℝ, log (x - 1) x - log (x - 1) 6 = 2 ↔ x = 3/2 ∨ x = 2/3

noncomputable def algebraic_roots : Prop := 
  ∀ x : ℝ, x^4 - 39*x^3 + 462*x^2 - 1576*x + 1152 = 0 ↔ x = 1 ∨ x = 4 ∨ x = 16 ∨ x = 18

-- Proof that these roots satisfy the given conditions
theorem verify_logarithmic_roots : logarithmic_roots :=
  by sorry

theorem verify_algebraic_roots : algebraic_roots :=
  by sorry

end verify_logarithmic_roots_verify_algebraic_roots_l352_352662


namespace max_brownies_l352_352838

-- Define the conditions given in the problem
def is_rectangular_pieces (m n : ℕ) : Prop := 
  let total_brownies := m * n in
  let perimeter_brownies := 2 * m + 2 * n - 4 in
  total_brownies = 2 * perimeter_brownies

theorem max_brownies : ∃ m n: ℕ, is_rectangular_pieces m n ∧ m * n = 100 := 
by
  use 5 
  use 20
  unfold is_rectangular_pieces
  sorry

end max_brownies_l352_352838


namespace relationships_l352_352777

theorem relationships (α β : ℝ) (hα : 0 < α ∧ α < π) (hβ : 0 < β ∧ β < π)
  (h1 : sin α - cos α = 17 / 13) (h2 : sin β + cos β = 17 / 13) :
  (α > 3 * β) ∧ (α + β < π ∨ α + β > π) :=
by
  sorry

end relationships_l352_352777


namespace money_spent_on_video_games_l352_352478

theorem money_spent_on_video_games :
  let total_money := 50
  let fraction_books := 1 / 4
  let fraction_snacks := 2 / 5
  let fraction_apps := 1 / 5
  let spent_books := fraction_books * total_money
  let spent_snacks := fraction_snacks * total_money
  let spent_apps := fraction_apps * total_money
  let spent_other := spent_books + spent_snacks + spent_apps
  let spent_video_games := total_money - spent_other
  spent_video_games = 7.5 :=
by
  sorry

end money_spent_on_video_games_l352_352478


namespace part_1_part_2_part_3_l352_352037

def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x^2 else -x^2

theorem part_1 : f (f 2) = 16 := 
  sorry

theorem part_2 : ∀ x : ℝ, f (-x) = -f x := 
  sorry

theorem part_3 {k : ℝ} : (∀ t ∈ Icc 1 2, f (t^2 - 2*t) + f (k - 2*t^2) < 0) → k > 8 := 
  sorry

end part_1_part_2_part_3_l352_352037


namespace isosceles_triangle_EFA_at_A_l352_352916

-- Definitions of points and geometric entities in Lean 4
variable {Point : Type}
variable [euclidean_geometry : EuclideanGeometry Point]

open EuclideanGeometry

-- Declaring the relevant points and properties
variables (A B C H_B H_C E F : Point)
variable (Γ : Circle Point)

-- Hypotheses as per the conditions in (a)
hypothesis (h1 : Circle.PointsOn Γ A B C)
hypothesis (h2 : IsFootOfAltitude H_B B (Line.through A C))
hypothesis (h3 : IsFootOfAltitude H_C C (Line.through A B))
hypothesis (h4 : Line.Intersection (Line.through H_B H_C) Γ E)
hypothesis (h5 : Line.Intersection (Line.through H_B H_C) Γ F)

-- The goal based on the problem statement
theorem isosceles_triangle_EFA_at_A : dist A E = dist A F :=
sorry

end isosceles_triangle_EFA_at_A_l352_352916


namespace a_n_less_than_inverse_n_minus_1_l352_352553

theorem a_n_less_than_inverse_n_minus_1 
  (n : ℕ) (h1 : 2 ≤ n) 
  (a : ℕ → ℝ) 
  (h2 : ∀ k : ℕ, 1 ≤ k → k ≤ n-1 → (a (k-1) + a k) * (a k + a (k+1)) = a (k-1) - a (k+1)) 
  (h3 : ∀ m : ℕ, m ≤ n → 0 < a m) : 
  a n < 1 / (n - 1) :=
sorry

end a_n_less_than_inverse_n_minus_1_l352_352553


namespace property_5_l352_352145

theorem property_5 (x y : ℝ) : 
  (⌊x⌋ℤ + ⌊y⌋ℤ ≤ ⌊x + y⌋ℤ) ∧ (⌊x + y⌋ℤ ≤ ⌊x⌋ℤ + ⌊y⌋ℤ + 1) :=
sorry

end property_5_l352_352145


namespace lambda_range_l352_352822

theorem lambda_range (m : ℝ) (n : ℝ) (h₀ : 0 < n) (h₁ : (m - n) ^ 2 + (m - log n + λ) ^ 2 ≥ 2) : λ ≥ 1 ∨ λ ≤ -3 :=
sorry

end lambda_range_l352_352822


namespace max_rational_products_l352_352246

theorem max_rational_products (n : ℕ) (r : ℕ) (irr : ℕ) (table_size : ℕ) 
  (r_count : ℕ) (irr_count : ℕ) (distinct_r : Prop) (distinct_irr : Prop) 
  (rational : ℕ → Prop) (irrational : ℕ → Prop) 
  (constraint_rational : ∀ i, i < r_count → rational i) 
  (constraint_irrational : ∀ i, i < irr_count → irrational i) 
  (distinct_indices_r : ∀ i j, i < r_count → j < r_count → i ≠ j → rational i ≠ rational j) 
  (distinct_indices_irr : ∀ i j, i < irr_count → j < irr_count → i ≠ j → irrational i ≠ irrational j)
  : nat := 
  if h : r < irr then sorry else 625 -- Condition to check if the rational number count is less than irrational and returning the result accordingly

#eval max_rational_products 50 50 50 2500 50 50 sorry sorry sorry sorry sorry sorry

end max_rational_products_l352_352246


namespace solve_for_diamond_l352_352060

def digit_condition (d : ℕ) := 
  (d * 8 + 9) = (d * 9 + 6) ∧ d < 10

theorem solve_for_diamond : ∃ d : ℕ, digit_condition d ∧ d = 3 := 
by 
  use 3
  split
  · sorry
  · rfl

end solve_for_diamond_l352_352060


namespace petals_in_garden_l352_352741

def lilies_count : ℕ := 8
def tulips_count : ℕ := 5
def petals_per_lily : ℕ := 6
def petals_per_tulip : ℕ := 3

def total_petals : ℕ := lilies_count * petals_per_lily + tulips_count * petals_per_tulip

theorem petals_in_garden : total_petals = 63 := by
  sorry

end petals_in_garden_l352_352741


namespace inequality_proof_l352_352567

theorem inequality_proof
  (n : ℕ) 
  (a b c : ℕ → ℝ)
  (ha : ∀ i, 0 ≤ a i)
  (hb : ∀ i, 0 ≤ b i)
  (hc : ∀ i, 0 ≤ c i) 
  (M : ℝ)
  (hM : M = max (∑ i in Finset.range n, a i) (max (∑ i in Finset.range n, b i) (∑ i in Finset.range n, c i))) 
  :
  (∑ k in Finset.range n, ∑ i in Finset.range (k + 1), (a k * c i + b i * c k - a k * b i)) ≥ M * (∑ k in Finset.range n, c k) :=
by 
  sorry

end inequality_proof_l352_352567


namespace surface_area_ratio_l352_352989

theorem surface_area_ratio (a : ℝ) (r : ℝ) (h : r = (real.sqrt 3 * a) / 2) :
  let S_sphere := 4 * real.pi * r^2 in
  let S_cube := 6 * a^2 in
  (S_sphere / S_cube) = (real.pi / 2) :=
sorry

end surface_area_ratio_l352_352989


namespace distinct_cubes_assembly_l352_352231

-- Definitions based on the provided conditions
structure CubeConfig where
  blue : ℕ    -- number of blue unit cubes
  white : ℕ   -- number of white unit cubes
  total : ℕ   -- total number of unit cubes

-- Cube configuration for 2x2x2 cube with 6 blue and 2 white unit cubes
def myCubeConfig : CubeConfig := { blue := 6, white := 2, total := 8 }

theorem distinct_cubes_assembly (c : CubeConfig) (h : c = myCubeConfig) : 
  cube_assembly_distinct (c, c.total) = 4 := sorry

end distinct_cubes_assembly_l352_352231


namespace study_days_l352_352708

theorem study_days (chapters worksheets : ℕ) (chapter_hours worksheet_hours daily_study_hours hourly_break
                     snack_breaks_count snack_break time_lunch effective_hours : ℝ)
  (h1 : chapters = 2) 
  (h2 : worksheets = 4) 
  (h3 : chapter_hours = 3) 
  (h4 : worksheet_hours = 1.5) 
  (h5 : daily_study_hours = 4) 
  (h6 : hourly_break = 10 / 60) 
  (h7 : snack_breaks_count = 3) 
  (h8 : snack_break = 10 / 60) 
  (h9 : time_lunch = 30 / 60)
  (h10 : effective_hours = daily_study_hours - (hourly_break * (daily_study_hours - 1)) - (snack_breaks_count * snack_break) - time_lunch)
  : (chapters * chapter_hours + worksheets * worksheet_hours) / effective_hours = 4.8 :=
by
  sorry

end study_days_l352_352708


namespace hyperbola_equation_l352_352977

theorem hyperbola_equation 
  (a b : ℝ) 
  (H_asymptotes : a / b = sqrt 2 / 2)
  (H_pass : (2 : ℝ) * (2 : ℝ) / a^2 - (2 : ℝ) * (2 : ℝ) / b^2 = 1) :
  (a = sqrt 2) ∧ (b = 2) → 
  ∀ x y : ℝ, (y^2 / 2 - x^2 / 4 = 1) := 
sorry

end hyperbola_equation_l352_352977


namespace count_even_digits_in_512_base_7_l352_352754

def base7_representation (n : ℕ) : ℕ := 
  sorry  -- Assuming this function correctly computes the base-7 representation of a natural number

def even_digits_count (n : ℕ) : ℕ :=
  sorry  -- Assuming this function correctly counts the even digits in the base-7 representation

theorem count_even_digits_in_512_base_7 : 
  even_digits_count (base7_representation 512) = 0 :=
by
  sorry

end count_even_digits_in_512_base_7_l352_352754


namespace probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l352_352674

-- Define the total number of ways to choose 3 leaders from 6 students
def total_ways : ℕ := Nat.choose 6 3

-- Calculate the number of ways in which boy A or girl B is chosen
def boy_A_chosen_ways : ℕ := Nat.choose 4 2 + 4 * 2
def girl_B_chosen_ways : ℕ := Nat.choose 4 1 + Nat.choose 4 2
def either_boy_A_or_girl_B_chosen_ways : ℕ := boy_A_chosen_ways + girl_B_chosen_ways

-- Calculate the probability that either boy A or girl B is chosen
def probability_either_boy_A_or_girl_B : ℚ := either_boy_A_or_girl_B_chosen_ways / total_ways

-- Calculate the probability that girl B is chosen
def girl_B_total_ways : ℕ := Nat.choose 5 2
def probability_B : ℚ := girl_B_total_ways / total_ways

-- Calculate the probability that both boy A and girl B are chosen
def both_A_and_B_chosen_ways : ℕ := Nat.choose 4 1
def probability_AB : ℚ := both_A_and_B_chosen_ways / total_ways

-- Calculate the conditional probability P(A|B) given P(B)
def conditional_probability_A_given_B : ℚ := probability_AB / probability_B

-- Theorem statements
theorem probability_either_boy_A_or_girl_B_correct : probability_either_boy_A_or_girl_B = (4 / 5) := sorry
theorem probability_B_correct : probability_B = (1 / 2) := sorry
theorem conditional_probability_A_given_B_correct : conditional_probability_A_given_B = (2 / 5) := sorry

end probability_either_boy_A_or_girl_B_correct_probability_B_correct_conditional_probability_A_given_B_correct_l352_352674


namespace magnitude_2a_minus_b_l352_352833

noncomputable def unit_vector (v : ℝ × ℝ × ℝ) : Prop :=
  (v.1^2 + v.2^2 + v.3^2) = 1

def angle_120_degrees (a b : ℝ × ℝ × ℝ) : Prop :=
  real_inner a b = 1 / 2 * (1 * 1) * (-1 / 2) -- since cos(120°) = -1/2

theorem magnitude_2a_minus_b (a b : ℝ × ℝ × ℝ) 
  (h₁ : unit_vector a) 
  (h₂ : unit_vector b) 
  (h₃ : angle_120_degrees a b) : 
  real.sqrt ((2 * a.1 - b.1)^2 + (2 * a.2 - b.2)^2 + (2 * a.3 - b.3)^2) = real.sqrt 7 := 
by 
  sorry

end magnitude_2a_minus_b_l352_352833


namespace mr_curtis_roosters_l352_352573

theorem mr_curtis_roosters (total_chickens : ℕ) (egg_laying_hens : ℕ) (non_egg_laying_hens : ℕ) :
  total_chickens = 325 → egg_laying_hens = 277 → non_egg_laying_hens = 20 → 
  (total_chickens - (egg_laying_hens + non_egg_laying_hens) = 28) :=
by
  intros h_total h_egg_laying h_non_egg_laying
  rw [h_total, h_egg_laying, h_non_egg_laying]
  norm_num
  sorry

end mr_curtis_roosters_l352_352573


namespace length_of_train_is_correct_l352_352693

-- Define the conditions with the provided data and given formulas.
def train_speed_kmh : ℝ := 63
def train_speed_ms : ℝ := train_speed_kmh * (1000 / 3600)
def time_to_pass_tree : ℝ := 16
def train_length : ℝ := train_speed_ms * time_to_pass_tree

-- State the problem as a theorem in Lean 4.
theorem length_of_train_is_correct : train_length = 280 := by
  -- conditions are defined, need to calculate the length
  unfold train_length train_speed_ms
  -- specify the conversion calculation manually
  simp
  norm_num
  sorry

end length_of_train_is_correct_l352_352693


namespace geometric_sequence_S4_l352_352960

noncomputable section

def geometric_series_sum (a1 q : ℚ) (n : ℕ) : ℚ := 
  a1 * (1 - q^n) / (1 - q)

theorem geometric_sequence_S4 (a1 : ℚ) (q : ℚ)
  (h1 : a1 * q^3 = 2 * a1)
  (h2 : 5 / 2 = a1 * (q^3 + 2 * q^6)) :
  geometric_series_sum a1 q 4 = 30 := by
  sorry

end geometric_sequence_S4_l352_352960


namespace binom_12_10_eq_66_l352_352356

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l352_352356


namespace find_n_l352_352867

theorem find_n (n : ℕ) : (1/5)^35 * (1/4)^18 = 1/(n*(10)^35) → n = 2 :=
by
  sorry

end find_n_l352_352867


namespace angle_ZXY_l352_352086

theorem angle_ZXY (O X Y Z : Type) [triangle XYZ]
    (is_incircle : is_incenter O X Y Z)
    (angle_XYZ : ∡ Y X Z = 80)
    (angle_ZOY : ∡ Z O Y = 20) :
    ∡ Z X Y = 60 :=
sorry

end angle_ZXY_l352_352086


namespace meeting_attendance_l352_352305

theorem meeting_attendance (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + 2 * B = 11) : A + B = 6 :=
sorry

end meeting_attendance_l352_352305


namespace equation_C1_rectangular_min_distance_PQ_l352_352456

-- Given conditions
def polar_equation_C1 (ρ θ : ℝ) : Prop := ρ * Real.sin (θ + π / 3) + 2 * Real.sqrt 3 = 0
def parametric_C2 (θ : ℝ) : ℝ × ℝ := (2 * Real.cos θ, 2 * Real.sin θ)

-- Prove the converted equation of C1
theorem equation_C1_rectangular (ρ θ : ℝ) (x y : ℝ) 
  (hρ : polar_equation_C1 ρ θ) 
  (hx : x = ρ * Real.cos θ) 
  (hy : y = ρ * Real.sin θ) : 
  (Real.sqrt 3 * y + x + 4 * Real.sqrt 3 = 0) :=
begin
  -- Use the polar equation, parametric definitions and solve for the rectangular form
  sorry
end

-- Prove the minimum value of |PQ|
theorem min_distance_PQ (θ : ℝ) 
  (x₁ y₁ : ℝ)
  (x₂ y₂ : ℝ) 
  (h_C1 : Real.sqrt 3 * y₁ + x₁ + 4 * Real.sqrt 3 = 0) 
  (h_C2 : (x₂, y₂) = parametric_C2 θ) :
  let d := Real.abs (2 * Real.cos θ + 2 * Real.sqrt 3 * Real.sin θ + 4 * Real.sqrt 3) / 2 in
  d = 2 * Real.sqrt 3 - 2 :=
begin
  -- Use geometric definitions and distance calculations to prove
  sorry
end

end equation_C1_rectangular_min_distance_PQ_l352_352456


namespace a_2016_gt_6_l352_352202

noncomputable def a_seq (n : ℕ) : ℝ
| 1     := 1
| 2     := 3/2
| (n+1) := if h : 2 ≤ n 
           then -(a_seq n)^2 + a_seq n * a_seq (n+1) + a_seq n * a_seq (n-1) - a_seq (n+1) * a_seq (n-1) - 2 * a_seq n + a_seq (n+1) + a_seq (n-1) 
           else 0

theorem a_2016_gt_6 : a_seq 2016 > 6 := 
by 
sory

end a_2016_gt_6_l352_352202


namespace mean_median_difference_l352_352889

noncomputable def calculateDifference : ℕ → ℕ × ℕ × ℕ × ℕ × ℕ × ℚ
| n := let p72 := n * 15 / 100 in
       let p84 := n * 30 / 100 in
       let p86 := n * 25 / 100 in
       let p92 := n * 10 / 100 in
       let p98 := n * 20 / 100 in
       let median := 86 in
       let mean := (72 * p72 + 84 * p84 + 86 * p86 + 92 * p92 + 98 * p98) / n in
       (p72, p84, p86, p92, p98, mean - median)

theorem mean_median_difference : (calculateDifference 100).snd.snd.snd.snd.snd.snd = 0.3 :=
by sorry

end mean_median_difference_l352_352889


namespace distance_centers_internally_tangent_circles_l352_352870

theorem distance_centers_internally_tangent_circles :
    ∀ (O₁ O₂ : Type) (r₁ r₂ : ℝ), r₁ = 3 ∧ r₂ = 4 →
    -- d is the distance between the centers of O₁ and O₂
    ∃ d : ℝ, d = |r₂ - r₁| → d = 1 :=
by
  intro O₁ O₂ r₁ r₂ h
  let d := |r₂ - r₁|
  have : d = 1 := 
    by
      rw [h.1, h.2]
      simp [abs_sub]
  exact ⟨d, this⟩
  sorry

end distance_centers_internally_tangent_circles_l352_352870


namespace sqrt_sum_eq_seven_l352_352160

variable (y : ℝ)

theorem sqrt_sum_eq_seven (h : sqrt (64 - y^2) - sqrt (36 - y^2) = 4) : 
  sqrt (64 - y^2) + sqrt (36 - y^2) = 7 := 
by
  sorry

end sqrt_sum_eq_seven_l352_352160


namespace problem_sequence_l352_352085

def sequence (a : ℕ → ℤ) : Prop :=
  a 1 = 1 ∧ a 2 = 5 ∧ ∀ n : ℕ, a (n + 2) = a (n + 1) - a n

theorem problem_sequence
  (a : ℕ → ℤ) 
  (h : sequence a) : 
  a 1000 = -1 :=
sorry

end problem_sequence_l352_352085


namespace numeral_is_1_11_l352_352879

-- Define the numeral question and condition
def place_value_difference (a b : ℝ) : Prop :=
  10 * b - b = 99.99

-- Now we define the problem statement in Lean
theorem numeral_is_1_11 (a b : ℝ) (h : place_value_difference a b) : 
  a = 100 ∧ b = 11.11 ∧ (a - b = 99.99) :=
  sorry

end numeral_is_1_11_l352_352879


namespace cousins_in_rooms_l352_352132

theorem cousins_in_rooms : 
  (number_of_ways : ℕ) (cousins : ℕ) (rooms : ℕ)
  (ways : ℕ) (is_valid_distribution : (ℕ → ℕ))
  (h_cousins : cousins = 5)
  (h_rooms : rooms = 4)
  (h_number_of_ways : ways = 67)
  :
  ∃ (distribute : ℕ → ℕ → ℕ), distribute cousins rooms = ways :=
sorry

end cousins_in_rooms_l352_352132


namespace cube_volume_given_surface_area_l352_352998

/-- Surface area of a cube given the side length. -/
def surface_area (side_length : ℝ) := 6 * side_length^2

/-- Volume of a cube given the side length. -/
def volume (side_length : ℝ) := side_length^3

theorem cube_volume_given_surface_area :
  ∃ side_length : ℝ, surface_area side_length = 24 ∧ volume side_length = 8 :=
by
  sorry

end cube_volume_given_surface_area_l352_352998


namespace sum_is_correct_l352_352922

theorem sum_is_correct (a b c d : ℤ) 
  (h : a + 1 = b + 2 ∧ b + 2 = c + 3 ∧ c + 3 = d + 4 ∧ d + 4 = a + b + c + d + 7) : 
  a + b + c + d = -6 := 
by 
  sorry

end sum_is_correct_l352_352922


namespace ellipse_eccentricity_proof_l352_352609

noncomputable def ellipse_eccentricity (a b c e : ℝ) : Prop :=
  (c = Real.sqrt (a^2 - b^2)) ∧ (e = c / a) ∧ (2 * b^2 = a) ∧ (a = 2) ∧ (2 * b^2 / a = 1)

theorem ellipse_eccentricity_proof :
  ∃ e : ℝ, ∀ a b c : ℝ, ellipse_eccentricity a b c e → e = Real.sqrt 3 / 2 :=
begin
  sorry
end

end ellipse_eccentricity_proof_l352_352609


namespace symmetry_range_l352_352818

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.exp x - 1/2
noncomputable def g (x a : ℝ) : ℝ := x^2 + Real.log (x + a)

theorem symmetry_range (a : ℝ) : (∃ x < 0, f x = g (-x) a) ↔ a ∈ Set.Iio (Real.sqrt Real.exp 1) :=
by
  sorry

end symmetry_range_l352_352818


namespace weight_of_replaced_person_l352_352970

theorem weight_of_replaced_person (avg_increase : ℝ) (num_persons : ℕ) (new_person_weight : ℝ) :
  (num_persons = 8) →
  (avg_increase = 2.5) →
  (new_person_weight = 85) →
  ∃ W : ℝ, new_person_weight - (num_persons * avg_increase) = W ∧ W = 65 := 
by
  intros h1 h2 h3
  use new_person_weight - (num_persons * avg_increase)
  split
  { rw [h1, h2, h3], norm_num }
  { norm_num, }

end weight_of_replaced_person_l352_352970


namespace min_value_expression_l352_352399

theorem min_value_expression : ∃ x : ℝ, (15 - x) * (13 - x) * (15 + x) * (13 + x) ≥ -784 :=
begin
  -- We claim that the minimum value of the given expression is -784.
  use sqrt 197,
  -- Expanding shows that this value occurs and is the minimum.
  sorry
end

end min_value_expression_l352_352399


namespace rectangle_area_inscribed_in_semicircle_l352_352726

theorem rectangle_area_inscribed_in_semicircle (O P Q W X Y Z : Point)
  (circle_O : Circle)
  (diameter_PQ : Line)
  (rect_WXYZ : Rectangle)
  (diameter_cond : Circle.diameter circle_O diameter_PQ)
  (inscribe_cond : Rectangle.inscribe_in_semicircle rect_WXYZ diameter_PQ)
  (WZ_len : length (Segment W Z) = 20)
  (PW_len : length (Segment P W) = 12)
  (QZ_len : length (Segment Q Z) = 12) :
  area rect_WXYZ = 80 * sqrt 15 := sorry

end rectangle_area_inscribed_in_semicircle_l352_352726


namespace simplify_fraction_eq_one_over_thirty_nine_l352_352587

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end simplify_fraction_eq_one_over_thirty_nine_l352_352587


namespace sum_first_9_terms_arith_seq_l352_352808

variables {a : ℕ → ℝ} {b : ℕ → ℝ}

-- Definitions based on conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
∀ n m, a n * a m = a ((n + m) / 2) ^ 2

def is_arithmetic_sequence (b : ℕ → ℝ) : Prop :=
∀ n m, b m = b n + (m - n) * (b 1 - b 0)

def S_n (b : ℕ → ℝ) (n : ℕ) : ℝ :=
n / 2 * (2 * b 0 + (n - 1) * (b 1 - b 0))

-- Conditions
axiom a_geom : is_geometric_sequence a
axiom a4a6_eq_2a5 : a 4 * a 6 = 2 * a 5
axiom b_arith : is_arithmetic_sequence b
axiom b5_eq_2a5 : b 5 = 2 * a 5

-- Theorem to be proved
theorem sum_first_9_terms_arith_seq : S_n b 9 = 36 :=
by sorry

end sum_first_9_terms_arith_seq_l352_352808


namespace contradiction_proof_l352_352582

theorem contradiction_proof (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : ¬ (0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c ∧ 0 ≤ d) :=
by 
  sorry

end contradiction_proof_l352_352582


namespace correct_identification_as_per_views_l352_352714

inductive GeometryBody
| triangular_prism
| quadrangular_pyramid
| cone
| frustum

open GeometryBody

def identified_geometric_bodies : list GeometryBody :=
  [triangular_prism, quadrangular_pyramid, cone, frustum]

theorem correct_identification_as_per_views :
  identified_geometric_bodies = [triangular_prism, quadrangular_pyramid, cone, frustum] :=
by 
  -- We state the proof, but the details are omitted.
  sorry

end correct_identification_as_per_views_l352_352714


namespace least_possible_perimeter_l352_352071

theorem least_possible_perimeter (A B C : ℝ) (a b c : ℝ) (hA : cos A = 8/17) (hB : cos B = 5/13) (hC : cos C = -3/5) : a + b + c = 27 :=
sorry

end least_possible_perimeter_l352_352071


namespace equal_diagonals_of_convex_quadrilateral_l352_352782

theorem equal_diagonals_of_convex_quadrilateral
  (A B C D : Type)
  (convex_quadrilateral : ConvexQuadrilateral A B C D)
  (r_circle_ABC : ∃ r, inscribed_circle_radius A B C = r)
  (r_circle_BCD : ∃ r, inscribed_circle_radius B C D = r)
  (r_circle_CDA : ∃ r, inscribed_circle_radius C D A = r)
  (r_circle_DAB : ∃ r, inscribed_circle_radius D A B = r)
  (h_common_radius: ∀ r1 r2 r3 r4, r1 = r2 ∧ r2 = r3 ∧ r3 = r4) :
  distance A C = distance B D := 
sorry

end equal_diagonals_of_convex_quadrilateral_l352_352782


namespace actual_time_when_watch_reads_11_pm_is_correct_l352_352315

-- Define the conditions
def noon := 0 -- Time when Cassandra sets her watch to the correct time
def actual_time_2_pm := 120 -- 2:00 PM in minutes
def watch_time_2_pm := 113.2 -- 1:53 PM and 12 seconds in minutes (113 minutes + 0.2 minutes)

-- Define the goal
def actual_time_watch_reads_11_pm := 731.25 -- 12:22 PM and 15 seconds in minutes from noon

-- Provide the theorem statement without proof
theorem actual_time_when_watch_reads_11_pm_is_correct :
  actual_time_watch_reads_11_pm = 731.25 :=
sorry

end actual_time_when_watch_reads_11_pm_is_correct_l352_352315


namespace negate_prop_l352_352185

theorem negate_prop :
  ¬ (∀ x : ℝ, x > 1 → x - 1 > Real.log x) ↔ ∃ x : ℝ, x > 1 ∧ x - 1 ≤ Real.log x :=
by
  sorry

end negate_prop_l352_352185


namespace teams_working_together_l352_352599

theorem teams_working_together
    (m n : ℕ) 
    (hA : ∀ t : ℕ, t = m → (t ≥ 0)) 
    (hB : ∀ t : ℕ, t = n → (t ≥ 0)) : 
  ∃ t : ℕ, t = (m * n) / (m + n) :=
by
  sorry

end teams_working_together_l352_352599


namespace total_hamburgers_for_lunch_l352_352277

theorem total_hamburgers_for_lunch 
  (initial_hamburgers: ℕ) 
  (additional_hamburgers: ℕ)
  (h1: initial_hamburgers = 9)
  (h2: additional_hamburgers = 3)
  : initial_hamburgers + additional_hamburgers = 12 := 
by
  sorry

end total_hamburgers_for_lunch_l352_352277


namespace min_period_f_max_min_values_f_l352_352458

noncomputable def f (x : ℝ) : ℝ := 2 * real.sqrt 3 * real.sin x * real.cos x + 2 * (real.cos x)^2 - 1

theorem min_period_f : IsPeriodic f π :=
sorry

theorem max_min_values_f : Sup (f '' Icc (-(π / 6)) (π / 4)) = sqrt 3 ∧ Inf (f '' Icc (-(π / 6)) (π / 4)) = -2 :=
sorry

end min_period_f_max_min_values_f_l352_352458


namespace simplify_factorial_fraction_l352_352727

theorem simplify_factorial_fraction (N : ℕ) : 
  (N + 1)! / (N + 2)! = 1 / (N + 2) :=
by
  sorry

end simplify_factorial_fraction_l352_352727


namespace fraction_over_65_l352_352386

def num_people_under_21 := 33
def fraction_under_21 := 3 / 7
def total_people (N : ℕ) := N > 50 ∧ N < 100
def num_people (N : ℕ) := num_people_under_21 = fraction_under_21 * N

theorem fraction_over_65 (N : ℕ) : 
  total_people N → num_people N → N = 77 ∧ ∃ x, (x / 77) = x / 77 :=
by
  intro hN hnum
  sorry

end fraction_over_65_l352_352386


namespace plants_bought_each_year_l352_352591

-- Define the given conditions as Lean functions and values
def cost_per_plant : ℕ := 20
def total_spent : ℕ := 640
def start_year : ℕ := 1989
def end_year : ℕ := 2021

-- Calculate the number of years Lily has been buying plants
def number_of_years : ℕ := end_year - start_year

-- Prove the number of plants bought each year
theorem plants_bought_each_year : total_spent / number_of_years = cost_per_plant / cost_per_plant := 
by
  have h_years : number_of_years = 32 := rfl
  have h_amount_per_year : total_spent / number_of_years = 20 := by sorry
  have h_plants_per_year : cost_per_plant / cost_per_plant = 1 := by sorry
  rw [h_years, h_amount_per_year, h_plants_per_year]
  sorry

end plants_bought_each_year_l352_352591


namespace part1_part2_l352_352814

noncomputable def f (x : ℝ) : ℝ := Real.logBase 2 (2^x + 1) - x / 2
noncomputable def g (x a : ℝ) : ℝ := Real.logBase 4 (a - 2^x)

theorem part1 (b : ℝ) : ∀ x₁ x₂ : ℝ, (f x₁ = x₁ / 2 + b) → (f x₂ = x₂ / 2 + b) → x₁ = x₂ :=
sorry

theorem part2 (a : ℝ) (h : ∃ x : ℝ, f x = g x a) : 2 + 2*Real.sqrt 2 ≤ a ∧ a < ∞ :=
sorry

end part1_part2_l352_352814


namespace train_cross_time_l352_352659

-- Definitions of the conditions
def length_train : ℝ := 270
def speed_train_kmph : ℝ := 25
def speed_man_kmph : ℝ := 2
def relative_speed_kmph : ℝ := speed_train_kmph + speed_man_kmph

-- Convert the speed from km/h to m/s
def kmph_to_mps (speed_kmph : ℝ) : ℝ := speed_kmph * 5 / 18
def relative_speed_mps : ℝ := kmph_to_mps relative_speed_kmph

-- Definition of the correct answer
def time_to_cross : ℝ := length_train / relative_speed_mps

-- Lean 4 proof statement
theorem train_cross_time : time_to_cross = 36 :=
by
  -- Completion of the proof is skipped
  sorry

end train_cross_time_l352_352659


namespace length_of_rectangular_shape_l352_352621

theorem length_of_rectangular_shape (A W L : ℝ) (hA : A = 1.77) (hW : W = 3) : L = 0.59 :=
by
  -- Define the formula for the area of a rectangle
  let area_formula := L * W
  -- Condition that the area is given and width is known
  have hArea : A = area_formula, by apply eq.refl 
  -- Substitute the given values
  have hSub : L * W = 1.77, by rw [hA, hW]; norm_num
  -- Thus the length can be calculated
  sorry

end length_of_rectangular_shape_l352_352621


namespace range_of_y_l352_352422

noncomputable def f (x : ℝ) : ℝ := 2 + Real.log2 x

def y (x : ℝ) : ℝ := ⌊f x⌋^2 + f (x^2)

theorem range_of_y :
  (∀ x, x ∈ Set.Icc (1/4:ℝ) 4 → y x ∈ Set.Icc 1 13) ∧
  (∀ t, t ∈ Set.Icc (-1:ℝ) 1 → t^2 + 6*t + 6 ∈ Set.Icc 1 13) :=
by
  sorry

end range_of_y_l352_352422


namespace angle_P_measure_l352_352301

theorem angle_P_measure (P Q : ℝ) (h1 : P + Q = 180) (h2 : P = 5 * Q) : P = 150 := by
  sorry

end angle_P_measure_l352_352301


namespace third_root_of_polynomial_l352_352616

theorem third_root_of_polynomial (p q r : ℝ) (hq_prime : nat.prime (int.nat_abs q))
  (h_root1 : ∃ (x : ℝ), x^3 - 7 * x^2 + p * x + q = 0 ∧ x = 3 + real.sqrt 5)
  (h_root2 : ∃ (x : ℝ), x^3 - 7 * x^2 + p * x + q = 0 ∧ x = 3 - real.sqrt 5) :
  r = 1 :=
sorry

end third_root_of_polynomial_l352_352616


namespace proposition1_proposition2_proposition3_proposition4_correct_propositions_l352_352810

noncomputable def nearest_integer (x : ℝ) : ℤ :=
  if h : ∃ (m : ℤ), m - 1/2 < x ∧ x ≤ m + 1/2 then 
    Classical.choose h 
  else 
    0

noncomputable def f (x : ℝ) : ℝ := x - nearest_integer x

-- Proposition 1: The domain of f is ℝ, and the range is (-1/2, 1/2].
theorem proposition1 : ∀ x : ℝ, -1/2 < f x ∧ f x ≤ 1/2 := sorry

-- Proposition 2: The smallest positive period of f is 1.
theorem proposition2 : ∀ x : ℝ, f (x + 1) = f x := sorry

-- Proposition 3: The point (k, 0) is not the center of symmetry of the graph of f, where k ∈ ℤ.
theorem proposition3 : ∀ k : ℤ, ¬ ∀ x : ℝ, f (2 * k - x) = - f x := sorry

-- Proposition 4: f is not increasing on the interval (-1/2, 3/2].
theorem proposition4 : ¬ ∀ x y : ℝ, -1/2 < x ∧ x < y ∧ y ≤ 3/2 → f x < f y := sorry

-- The correct propositions are 1 and 2.
theorem correct_propositions : (proposition1, proposition2) := sorry

end proposition1_proposition2_proposition3_proposition4_correct_propositions_l352_352810


namespace sum_of_zeros_of_f_is_minus_4_l352_352930

noncomputable def f (x : ℝ) (f1 : ℝ) (f2 : ℝ) : ℝ :=
  (f1 * x^2 + f2 * x - 1) / x

theorem sum_of_zeros_of_f_is_minus_4 :
  ∀ f1 f2 : ℝ,
    f1 = 1 / 4 → f2 = 1 →
    (let x1 := -2 + 2 * Real.sqrt 2 in 
     let x2 := -2 - 2 * Real.sqrt 2 in
     x1 + x2 = -4) :=
by
  intros f1 f2 f1_eq f2_eq
  let x1 := -2 + 2 * Real.sqrt 2
  let x2 := -2 - 2 * Real.sqrt 2
  have h : x1 + x2 = -4 := sorry
  exact h

end sum_of_zeros_of_f_is_minus_4_l352_352930


namespace expand_and_simplify_expression_l352_352387

variable {x y : ℝ} {i : ℂ}

-- Declare i as the imaginary unit satisfying i^2 = -1
axiom imaginary_unit : i^2 = -1

theorem expand_and_simplify_expression :
  (x + 3 + i * y) * (x + 3 - i * y) + (x - 2 + 2 * i * y) * (x - 2 - 2 * i * y)
  = 2 * x^2 + 2 * x + 13 - 5 * y^2 :=
by
  sorry

end expand_and_simplify_expression_l352_352387


namespace repeating_decimal_product_of_lowest_fraction_l352_352650

theorem repeating_decimal_product_of_lowest_fraction :
  let x := (013 : ℕ) / (999 : ℕ)
  ∃ (p q : ℕ), (x = p / q) ∧ (Nat.gcd p q = 1) ∧ (p * q = 12987) := 
by {
  let x := 13 / 999,
  use [13, 999],
  split,
  { refl },
  split,
  { exact Nat.gcd_eq_one_of_coprime (Nat.coprime_of_divisors 13 999) },
  { exact rfl }
}

end repeating_decimal_product_of_lowest_fraction_l352_352650


namespace exponent_problem_l352_352479

theorem exponent_problem (x : ℝ) (h : 9^(5 * x) = 59049) : 9^(5 * x - 4) = 9 := 
sorry

end exponent_problem_l352_352479


namespace arithmetic_rounding_l352_352668

theorem arithmetic_rounding : 
  Real.round (3550 - (1002 / 20.04)) 2 = 3499.95 := 
by 
  sorry

end arithmetic_rounding_l352_352668


namespace max_distance_S_to_origin_l352_352431

noncomputable def max_distance_of_S : ℝ := 3

theorem max_distance_S_to_origin (θ : ℝ) (hθ : 0 ≤ θ ∧ θ ≤ π) :
  ∃ (z : ℂ), z = complex.of_real (Real.cos θ) + complex.I * (Real.sin θ) ∧
            |2 * conj z + complex.I * z| = max_distance_of_S := sorry

end max_distance_S_to_origin_l352_352431


namespace min_value_fraction_l352_352019

theorem min_value_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ 1) (hy : -1 ≤ y ∧ y ≤ 3) : 
  ∃ v, v = (x + y) / x ∧ v = -2 := 
by 
  sorry

end min_value_fraction_l352_352019


namespace find_cost_price_l352_352270

theorem find_cost_price (C : ℝ) (h1 : C * 1.05 = C + 0.05 * C)
  (h2 : 0.95 * C = C - 0.05 * C)
  (h3 : 1.05 * C - 4 = 1.045 * C) :
  C = 800 := sorry

end find_cost_price_l352_352270


namespace seq_general_formula_condition_1_seq_general_formula_condition_2_seq_general_formula_condition_3_sum_bn_l352_352106

theorem seq_general_formula_condition_1 (a₁ : ℕ) :
  (∀ n, a_n = a₁ * 2^(n-1)) ∧
  ((a₁ * 2^(1)) + (a₁ * 2^(3)) - 4 = 2 * (a₁ * 2^(2)))
  -> ∀ n, a_n = 2^n :=
sorry

theorem seq_general_formula_condition_2 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n, S n = 2 * a n - 2) ∧ (∀ n, a (n+1) = S (n+1) - S n)
  -> ∀ n, a n = 2^n :=
sorry

theorem seq_general_formula_condition_3 (S : ℕ → ℕ) :
  (∀ n, S n = 2^(n+1) - 2) ∧ (∀ n, a n = S n - S (n-1))
  -> ∀ n, a n = 2^n :=
sorry

theorem sum_bn (a : ℕ → ℕ) (n : ℕ) :
  (∀ n, a n = 2^n) -> 
  let b n := (1 + Real.log (2 a n)) / a n in
  let T n := ∑ i in finset.range (n + 1), b i in
  T n = 3 - (n + 3) / (2^n) :=
sorry

end seq_general_formula_condition_1_seq_general_formula_condition_2_seq_general_formula_condition_3_sum_bn_l352_352106


namespace sum_of_real_roots_l352_352405

theorem sum_of_real_roots : 
  let f := λ x : ℝ, x^4 - 8*x + 4 in
  ∑ root in (Multiset.filter (λ r, f r = 0) (Multiset.replicate 4 0)), root = -2 * Real.sqrt 2 :=
  sorry

end sum_of_real_roots_l352_352405


namespace jill_spent_more_l352_352907

def cost_per_ball_red : ℝ := 1.50
def cost_per_ball_yellow : ℝ := 1.25
def cost_per_ball_blue : ℝ := 1.00

def packs_red : ℕ := 5
def packs_yellow : ℕ := 4
def packs_blue : ℕ := 3

def balls_per_pack_red : ℕ := 18
def balls_per_pack_yellow : ℕ := 15
def balls_per_pack_blue : ℕ := 12

def balls_red : ℕ := packs_red * balls_per_pack_red
def balls_yellow : ℕ := packs_yellow * balls_per_pack_yellow
def balls_blue : ℕ := packs_blue * balls_per_pack_blue

def cost_red : ℝ := balls_red * cost_per_ball_red
def cost_yellow : ℝ := balls_yellow * cost_per_ball_yellow
def cost_blue : ℝ := balls_blue * cost_per_ball_blue

def combined_cost_yellow_blue : ℝ := cost_yellow + cost_blue

theorem jill_spent_more : cost_red = combined_cost_yellow_blue + 24 := by
  sorry

end jill_spent_more_l352_352907


namespace min_log2_in_interval_l352_352984

noncomputable def f (x : ℝ) : ℝ := Real.log x / Real.log 2

theorem min_log2_in_interval : ∃ x ∈ Icc (1:ℝ) (2:ℝ), ∀ y ∈ Icc (1:ℝ) (2:ℝ), f y ≥ f x ∧ f x = 0 :=
by
  sorry

end min_log2_in_interval_l352_352984


namespace ethanol_percentage_of_fuelB_l352_352704

theorem ethanol_percentage_of_fuelB 
  (fuel_tank_capacity : ℕ)
  (fuel_A_ethanol_percentage : ℚ)
  (fuel_A_volume : ℚ)
  (total_ethanol : ℚ)
  (fuel_A_ethanol_volume : fuel_A_ethanol_percentage / 100 * fuel_A_volume)
  (fuel_B_volume : fuel_tank_capacity - fuel_A_volume)
  (fuel_B_ethanol_volume : total_ethanol - fuel_A_ethanol_volume)
  : (fuel_B_ethanol_volume / fuel_B_volume * 100) = 16 :=
begin
  -- This is where the proof would go, but we omit it in the statement.
  sorry
end

end ethanol_percentage_of_fuelB_l352_352704


namespace problem_l352_352868

theorem problem (x y : ℚ) (h1 : x + y = 10 / 21) (h2 : x - y = 1 / 63) : 
  x^2 - y^2 = 10 / 1323 := 
by 
  sorry

end problem_l352_352868


namespace max_angle_line_plane_l352_352869

theorem max_angle_line_plane (θ : ℝ) (h_angle : θ = 72) :
  ∃ φ : ℝ, φ = 90 ∧ (72 ≤ φ ∧ φ ≤ 90) :=
by sorry

end max_angle_line_plane_l352_352869


namespace max_OP_OQ_product_l352_352244

theorem max_OP_OQ_product
    (x0 y0 r : ℝ)
    (hM_on_ellipse : (x0^2 / 4 + y0^2 = 1))
    (h_circle_eq : ∀ x y : ℝ, (x - x0)^2 + (y - y0)^2 = r^2)
    (k1 k2 : ℝ)
    (h_product_const : k1 * k2 = c) :
    ∃ max_value : ℝ, 
    max_value = (2 * sqrt(1 + k1^2) / sqrt(1 + 4 * k1^2)) * (2 * sqrt(1 + k2^2) / sqrt(1 + 4 * k2^2)) := sorry

end max_OP_OQ_product_l352_352244


namespace sum_f_values_l352_352038

def f (x : ℝ) : ℝ := 2 / (2^x + 1) + Real.sin x

theorem sum_f_values :
  f (-3) + f (-2) + f (-1) + f 0 + f 1 + f 2 + f 3 = 7 :=
by
  sorry

end sum_f_values_l352_352038


namespace university_admission_l352_352672

def students_ratio (x y z : ℕ) : Prop :=
  x * 5 = y * 2 ∧ y * 3 = z * 5

def third_tier_students : ℕ := 1500

theorem university_admission :
  ∀ x y z : ℕ, students_ratio x y z → z = third_tier_students → y - x = 1500 :=
by
  intros x y z hratio hthird
  sorry

end university_admission_l352_352672


namespace total_books_for_girls_l352_352630

theorem total_books_for_girls (num_girls : ℕ) (num_boys : ℕ) (total_books : ℕ)
  (h_girls : num_girls = 15)
  (h_boys : num_boys = 10)
  (h_books : total_books = 375) :
  num_girls * (total_books / (num_girls + num_boys)) = 225 :=
by
  sorry

end total_books_for_girls_l352_352630


namespace jia_jia_clover_count_l352_352936

theorem jia_jia_clover_count : ∃ x : ℕ, 3 * x + 4 = 100 ∧ x = 32 := by
  sorry

end jia_jia_clover_count_l352_352936


namespace ngonal_diagonal_property_l352_352909

theorem ngonal_diagonal_property (n : ℕ) (h : n ≥ 4)
    (h_cond : ∀ s : finset (fin n.succ), s.card = n-3 → 
        (∀ t1 t2 t3 : ℕ, 
          t1 + t2 + t3 = (finset.sum s id) 
          → True)): (even n ∨ ∃ k : ℕ, n = 4 * k + 1) :=
begin
  sorry
end

end ngonal_diagonal_property_l352_352909


namespace average_age_combined_group_l352_352166

def average_age_sixth_graders (n m : ℕ) (avg_n : ℤ) (avg_m : ℤ) : ℤ :=
  ((n * avg_n) + (m * avg_m)) / (n + m)

theorem average_age_combined_group :
  average_age_sixth_graders 40 30 12 45 = 26.14 :=
sorry

end average_age_combined_group_l352_352166


namespace distance_from_diagonal_intersection_to_base_l352_352525

theorem distance_from_diagonal_intersection_to_base (AD BC AB R : ℝ) (O : ℝ → Prop) (M N Q : ℝ) :
  (AD + BC + 2 * AB = 8) ∧
  (AD + BC) = 4 ∧
  (R = 1 / 2) ∧
  (2 = R * (AD + BC) / 2) ∧
  (BC = AD + 2 * AB) ∧
  (∀ x, x * (2 - x) = (1 / 2) ^ 2)  →
  (Q = (2 - Real.sqrt 3) / 4) :=
by
  intros
  sorry

end distance_from_diagonal_intersection_to_base_l352_352525


namespace measure_of_W_l352_352149

def parallelogram (V : Type) [add_comm_group V] [module ℝ V] (W X Y Z : V) : Prop :=
  -- Definition of a parallelogram by sides
  W - X = Z - Y ∧ W - Z = X - Y

variables {V : Type} [inner_product_space ℝ V]
variables (W X Y Z O : V)
variables (h1 : parallelogram V W X Y Z) (h2 : 0 < inner_product_space.angle W O X ∧ inner_product_space.angle W O X = π / 4)

theorem measure_of_W (h_parallelogram : parallelogram V W X Y Z)
  (intersect : O = (W + X) / 2 ∧ O = (Y + Z) / 2)
  (given_angle : inner_product_space.angle W O X = π / 4) :
  inner_product_space.angle W X Z = 7 * π / 9 :=
sorry

end measure_of_W_l352_352149


namespace train_length_l352_352695

theorem train_length :
  let speed_kmph := 63
  let time_seconds := 16
  let speed_mps := (speed_kmph * 1000) / 3600
  let length_meters := speed_mps * time_seconds
  length_meters = 280 := 
by
  sorry

end train_length_l352_352695


namespace intersection_of_complements_l352_352107

theorem intersection_of_complements {U S T : Set ℕ}
  (hU : U = {1, 2, 3, 4, 5, 6, 7, 8})
  (hS : S = {1, 3, 5})
  (hT : T = {3, 6}) :
  (U \ S) ∩ (U \ T) = {2, 4, 7, 8} :=
by
  sorry

end intersection_of_complements_l352_352107


namespace solution_l352_352010

theorem solution (x : ℕ) (h : nat.choose 10 x = nat.choose 10 (3 * x - 2)) : x = 1 ∨ x = 3 :=
by
  sorry

end solution_l352_352010


namespace base4_more_digits_than_base9_l352_352857

def base_digits (n : ℕ) (b : ℕ) : ℕ :=
(n.log b).to_nat + 1

theorem base4_more_digits_than_base9 (n : ℕ) (h : n = 1234) : base_digits 1234 4 = base_digits 1234 9 + 2 :=
by
  have h4 : base_digits 1234 4 = 6 := by sorry -- Proof steps to show base-4 has 6 digits 
  have h9 : base_digits 1234 9 = 4 := by sorry -- Proof steps to show base-9 has 4 digits
  rw [h4, h9]
  norm_num

end base4_more_digits_than_base9_l352_352857


namespace order_of_A_B_C_l352_352557

-- Define constants A, B, and C based on given conditions
def A : ℝ := Real.cos (1 / 2)
def B : ℝ := Real.cos (3 / 2)
def C : ℝ := Real.sin (3 / 2) - Real.sin (1 / 2)

-- State the theorem to prove the order of A, B, and C
theorem order_of_A_B_C : A > C ∧ C > B := by
  -- The proof steps are omitted; only the statement is required.
  sorry

end order_of_A_B_C_l352_352557


namespace estimate_proportion_of_households_owning_3_or_more_sets_l352_352877

noncomputable def households := 100000
noncomputable def ordinary_households := 99000
noncomputable def high_income_households := 1000

noncomputable def sample_ordinary := 990
noncomputable def sample_high_income := 100

noncomputable def households_with_3_or_more_sets := 120
noncomputable def ordinary_with_3_or_more_sets := 50
noncomputable def high_income_with_3_or_more_sets := 70

theorem estimate_proportion_of_households_owning_3_or_more_sets :
    (households_with_3_or_more_sets.succ / (sample_ordinary.succ + sample_high_income.succ) * 100) ≈ 5.7 :=
sorry

end estimate_proportion_of_households_owning_3_or_more_sets_l352_352877


namespace thre_points_monochromatic_triangle_l352_352012

open Classical BigInt

-- Definitions of no_three_collinear, colored_segments, monochromatic_triangle
def no_three_collinear (pts : list (ℝ × ℝ)) : Prop := sorry
def colored_segments (pts : list (ℝ × ℝ)) (k : ℕ) : Prop := sorry
def monochromatic_triangle (triangle : finset (list (fin 3))) : Prop := sorry

theorem thre_points_monochromatic_triangle 
  (N k : ℕ) 
  (pts : list (ℝ × ℝ))
  (hN : N = pts.length)
  (h_collinear : no_three_collinear pts)
  (h_colored_segments : colored_segments pts k) : 
  N > ⌊(k.factorial * Real.exp 1)⌋₊ →
  ∃ (triangle : finset (list (fin 3))), monochromatic_triangle triangle :=
begin
  sorry -- Proof goes here
end

end thre_points_monochromatic_triangle_l352_352012


namespace number_of_true_propositions_is_2_l352_352665

def line : Type := ℕ -- abstract definition for lines
def plane : Type := ℕ -- abstract definition for planes

-- relational definitions
constant parallel (l: line) (p: plane) : Prop
constant perpendicular (l: line) (p: plane) : Prop
constant coplanar_parallel (p1 p2: plane) : Prop
constant coplanar_perpendicular (p1 p2: plane) : Prop
constant lines_parallel (l1 l2: line) : Prop
constant lines_perpendicular (l1 l2: line) : Prop

-- Given conditions as assumptions
variables (m n: line) (alpha beta: plane)

axiom cond1 : parallel m alpha ∧ parallel n beta ∧ coplanar_parallel alpha beta → ¬ lines_parallel m n
axiom cond2 : parallel m alpha ∧ perpendicular n beta ∧ coplanar_perpendicular alpha beta → ¬ lines_parallel m n
axiom cond3 : perpendicular m alpha ∧ parallel n beta ∧ coplanar_parallel alpha beta → lines_perpendicular m n
axiom cond4 : perpendicular m alpha ∧ perpendicular n beta ∧ coplanar_perpendicular alpha beta → lines_perpendicular m n

theorem number_of_true_propositions_is_2 :
  (cond1) ∧ (¬ cond2) ∧ (cond3) ∧ (cond4) → (count_true_propositions [cond1, cond2, cond3, cond4] = 2) :=
sorry

end number_of_true_propositions_is_2_l352_352665


namespace sin_exp_intersections_count_l352_352732

open Real

theorem sin_exp_intersections_count : 
  (finset.card
    (finset.filter 
      (λ x, sin x = (1 / 3) ^ x) 
      (finset.Ico 0 (floor (150 * π) + 1)))) = 150 := 
sorry

end sin_exp_intersections_count_l352_352732


namespace binom_12_10_eq_66_l352_352339

theorem binom_12_10_eq_66 : Nat.choose 12 10 = 66 :=
by
  sorry

end binom_12_10_eq_66_l352_352339


namespace total_flower_petals_l352_352742

def num_lilies := 8
def petals_per_lily := 6
def num_tulips := 5
def petals_per_tulip := 3

theorem total_flower_petals :
  (num_lilies * petals_per_lily) + (num_tulips * petals_per_tulip) = 63 :=
by
  sorry

end total_flower_petals_l352_352742


namespace increasing_iff_range_a_three_distinct_real_roots_l352_352821

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 2 * a then x^2 + (2 - 2 * a) * x else - x^2 + (2 + 2 * a) * x

theorem increasing_iff_range_a (a : ℝ) :
  (∀ x₁ x₂, x₁ < x₂ → f a x₁ < f a x₂) ↔ -1 ≤ a ∧ a ≤ 1 :=
sorry

theorem three_distinct_real_roots (a t : ℝ) (h_a : -2 ≤ a ∧ a ≤ 2)
  (h_roots : ∃ x₁ x₂ x₃, x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₃ ≠ x₁ ∧
                           f a x₁ = t * f a (2 * a) ∧
                           f a x₂ = t * f a (2 * a) ∧
                           f a x₃ = t * f a (2 * a)) :
  1 < t ∧ t < 9 / 8 :=
sorry

end increasing_iff_range_a_three_distinct_real_roots_l352_352821


namespace clock_difference_1hr_l352_352207

theorem clock_difference_1hr (t : ℕ) (h1 : t = 15) :
  let fast_clock_time := t + t / 30
  let slow_clock_time := t - t / 30 in
  fast_clock_time - slow_clock_time = 1 :=
by
  let fast_clock_time := t + t / 30
  let slow_clock_time := t - t / 30
  sorry

end clock_difference_1hr_l352_352207


namespace solve_cubic_root_l352_352393

theorem solve_cubic_root (x : ℝ) : 
  (∛(5 - x) = -5/3) → (x = 260/27) :=
by {
  sorry
}

end solve_cubic_root_l352_352393


namespace transformed_graph_passes_through_4_neg1_l352_352453

def f : ℝ → ℝ := sorry -- We assume the existence of a function f

-- Given condition: f(1) = -1
def given_condition : Prop := f(1) = -1

-- The theorem statement we need to prove
theorem transformed_graph_passes_through_4_neg1 (h : given_condition) : f(4 - 3) = -1 := 
by
  rw [←h]
  sorry

end transformed_graph_passes_through_4_neg1_l352_352453


namespace find_z_l352_352407

theorem find_z :
  (sqrt 1.5) / (sqrt 0.81) + (sqrt z) / (sqrt 0.49) = 3.0751133491652576 → z = 1.44 :=
by
  sorry

end find_z_l352_352407


namespace square_perimeter_l352_352739

theorem square_perimeter (s : ℕ) (h : s = 8) : 4 * s = 32 := by
  rw [h]
  norm_num
  sorry

end square_perimeter_l352_352739


namespace ParticlePaths128_l352_352680

theorem ParticlePaths128 :
  let moves (p q : ℕ) := (p = 1 ∧ q = 0) ∨ (p = 0 ∧ q = 1) ∨ (p = 1 ∧ q = 1)
  ∧ ∀ x : ℕ × ℕ, x.1 ≤ 6 ∧ x.2 ≤ 6
  → (∃ f : ℕ → ℕ × ℕ, (f 0 = (0, 0)) ∧ (f 12 = (6, 6)) ∧ (∀ n, moves (f (n + 1)).1 (f n).1 ∧ moves (f (n + 1)).2 (f n).2 ∧ f n ≠ f (n - 1))
  → 128 :=
sorry

end ParticlePaths128_l352_352680


namespace verify_expressions_l352_352468

variables (x y : ℚ)

axiom ratio_x_y : x / y = 5 / 6

theorem verify_expressions :
  (x / y = 5 / 6) →
  ((x + 2 * y) / y = 17 / 6) ∧
  ((2 * x) / (3 * y) = 5 / 9) ∧
  ((y - x) / (2 * y) = 1 / 12) ∧
  ((x + y) / (2 * y) = 11 / 12) ∧ 
  (x / (y + x) = 5 / 11) :=
by
  intro ratio_x_y,
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  split,
  { sorry },
  { sorry }

end verify_expressions_l352_352468


namespace value_of_x_l352_352420

theorem value_of_x (x y : ℕ) (h₁ : x / y = 12 / 5) (h₂ : y = 25) : x = 60 :=
sorry

end value_of_x_l352_352420


namespace cousins_rooms_distribution_l352_352112

theorem cousins_rooms_distribution : 
  (∑ n in ({ (5,0,0,0), (4,1,0,0), (3,2,0,0), (3,1,1,0), (2,2,1,0), (2,1,1,1) } : finset (ℕ × ℕ × ℕ × ℕ)), 
    match n with 
    | (5,0,0,0) => 1
    | (4,1,0,0) => 5
    | (3,2,0,0) => 10 
    | (3,1,1,0) => 20 
    | (2,2,1,0) => 30 
    | (2,1,1,1) => 10 
    | _ => 0 
    end) = 76 := 
by 
  sorry

end cousins_rooms_distribution_l352_352112


namespace matrix_power_four_correct_l352_352318

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l352_352318


namespace maximum_revenue_l352_352675

noncomputable def optimal_advertising_time_allocation : ℕ × ℕ :=
(100, 200)

definition conditions (t_A t_B : ℕ) : Prop :=
t_A + t_B ≤ 300 ∧ 500 * t_A + 200 * t_B ≤ 90000

definition revenue (t_A t_B : ℕ) : ℚ :=
0.3 * t_A + 0.2 * t_B

theorem maximum_revenue :
  conditions 100 200 ∧ revenue 100 200 = 70 :=
by
  sorry

end maximum_revenue_l352_352675


namespace M_plus_N_eq_2_l352_352248

noncomputable def M : ℝ := 1^5 + 2^4 * 3^3 - (4^2 / 5^1)
noncomputable def N : ℝ := 1^5 - 2^4 * 3^3 + (4^2 / 5^1)

theorem M_plus_N_eq_2 : M + N = 2 := by
  sorry

end M_plus_N_eq_2_l352_352248


namespace cousins_room_distributions_l352_352127

theorem cousins_room_distributions : 
  let cousins := 5
  let rooms := 4
  let possible_distributions := (1 + 5 + 10 + 10 + 15 + 10 : ℕ)
  possible_distributions = 51 :=
by
  sorry

end cousins_room_distributions_l352_352127


namespace part_a_part_b_l352_352663

-- State definition for a given number of boxes and balls configuration.
def State (num_boxes : ℕ) : Type := list ℕ

-- Move definition: taking all balls from a given box and distributing them clockwise.
def move (s : State n) (i : ℕ) : State n :=
sorry -- implementation of the move

-- The main statement for part (a) proving the eventual repetition of states
theorem part_a (num_boxes : ℕ) (init : State num_boxes) :
  ∃ k, cyclic_state (iterate move k init) = init :=
sorry

-- The main statement for part (b) proving the reachability of any state from any initial state
theorem part_b (num_boxes : ℕ) (init target : State num_boxes) :
  reachable (move) init target :=
sorry

end part_a_part_b_l352_352663


namespace valid_arrangements_count_l352_352080

theorem valid_arrangements_count : 
  (card {l : List Char | l.permutes ['P', 'U', 'M', 'α', 'C'] ∧ 
         l.indexOf 'M' < l.indexOf 'α' ∧ 
         l.indexOf 'α' < l.indexOf 'C']) = 20 := 
  sorry

end valid_arrangements_count_l352_352080


namespace binom_12_10_eq_66_l352_352353

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l352_352353


namespace monotone_on_minus_pi_div_two_pi_div_two_no_extreme_points_a_ge_one_div_3_pi_sq_one_extreme_point_zero_lt_a_lt_one_div_3_pi_sq_l352_352039

section Part1

-- Define the function f when a = 0
def f (x : ℝ) : ℝ := 2 * sin x - x * cos x

-- Prove that f(x) is monotonically increasing on (-π/2, π/2)
theorem monotone_on_minus_pi_div_two_pi_div_two : 
  ∀ x : ℝ, x > -Real.pi / 2 ∧ x < Real.pi / 2 → (deriv f x) > 0 := sorry

end Part1

section Part2

-- Define the function f for the general case
def f (a x : ℝ) : ℝ := a * x^3 + 2 * sin x - x * cos x

-- Derivative of f
def f' (a x : ℝ) : ℝ := 3 * a * x^2 + cos x + x * sin x

-- Second derivative, g'
def g' (a x : ℝ) : ℝ := 6 * a * x + x * cos x

-- Prove number of extreme points on (0, π) for a >= 1/(3π^2)
theorem no_extreme_points_a_ge_one_div_3_pi_sq (a : ℝ) (h : a ≥ 1 / (3 * Real.pi^2)) : 
  ∀ x : ℝ, x > 0 ∧ x < Real.pi → (f' a x) ≠ 0 := sorry

-- Prove number of extreme points on (0, π) for 0 < a < 1/(3π^2)
theorem one_extreme_point_zero_lt_a_lt_one_div_3_pi_sq (a : ℝ) (h : 0 < a ∧ a < 1 / (3 * Real.pi^2)) : 
  ∃ x : ℝ, x > 0 ∧ x < Real.pi ∧ (f' a x) = 0 := sorry

end Part2

end monotone_on_minus_pi_div_two_pi_div_two_no_extreme_points_a_ge_one_div_3_pi_sq_one_extreme_point_zero_lt_a_lt_one_div_3_pi_sq_l352_352039


namespace measure_angle_BDC_l352_352255

theorem measure_angle_BDC 
  (A B C D : Type) 
  [tri : Triangle ABC]  -- Assume we have some structure to represent triangles
  (right_angle_C : ∠C = 90)
  (angle_A : ∠A = 30)
  (D_on_AC : PointOnLine D (LineAC A C))
  (bisector_BD : AngleBisector BD ∠ABC) :
  ∠BDC = 60 :=
by
  sorry

end measure_angle_BDC_l352_352255


namespace least_sum_exponents_of_520_l352_352492

theorem least_sum_exponents_of_520 : 
  ∀ (a b : ℕ), (520 = 2^a + 2^b) → a ≠ b → (a + b ≥ 12) :=
by
  -- Proof goes here
  sorry

end least_sum_exponents_of_520_l352_352492


namespace matrix_power_four_correct_l352_352317

theorem matrix_power_four_correct :
  let A := Matrix.of (fun i j => ![![2, -1], ![1, 1]].get i j) in
  A ^ 4 = Matrix.of (fun i j => ![![0, -9], ![9, -9]].get i j) :=
by
  sorry

end matrix_power_four_correct_l352_352317


namespace angle_equality_of_opf_and_oep_l352_352881
-- Import the necessary libraries

-- Define the conditions and prove the required statement in Lean 4
theorem angle_equality_of_opf_and_oep 
  (A B C D M E F O P : Type)
  (AD : Line)
  (EF_parallel_AD : parallel (line_through E F) AD)
  (circle_OP : Circle O P (line_length O M))
  (conv_quad : ConvexQuadrilateral A B C D)
  (intersection_AC_BD : intersect (line_through A C) (line_through B D) M)
  (intersection_line_M_EF_AB_CD_O : intersect (line_through M) (parallel_line_through M AD) = (line_through E F ∩ line_through B C) = O)
  (E_on_AB : on_line E (line_through A B))
  (F_on_CD : on_line F (line_through C D))
  (O_on_ext_BC : on_line O (extend_line (line_through B C)))
  : ∠ O P F = ∠ O E P := sorry

end angle_equality_of_opf_and_oep_l352_352881


namespace max_M_A_l352_352915

def subsets (start end : ℕ) : Set (Set ℕ) :=
  { s | ∀ x ∈ s, start ≤ x ∧ x ≤ end }

def disjoint (A B : Set ℕ) : Prop :=
  ∀ x, x ∈ A → x ∉ B 

def condition (A B : Set ℕ) : Prop :=
  ∀ n, n ∈ A → (2 * n + 2) ∈ B

noncomputable def M (A : Set ℕ) : ℕ :=
  A.sum id

theorem max_M_A : ∃ A B : Set ℕ,
  A ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} ∧
  B ⊆ {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20} ∧
  disjoint A B ∧
  condition A B ∧
  M A = 39 :=
by {
  sorry
}

end max_M_A_l352_352915


namespace solve_for_x_l352_352831

theorem solve_for_x (x y z : ℝ) (h1 : x * y = 8 - 3 * x - 2 * y) (h2 : y * z = 10 - 5 * y - 3 * z) (h3 : x * z = 40 - 5 * x - 4 * z) :
  x = 3 :=
begin
  sorry
end

end solve_for_x_l352_352831


namespace distance_AB_eq_2_sqrt_3_l352_352914

open Real

-- Definitions of points A and B in polar coordinates and condition
def A (θ₁ : ℝ) : ℝ × ℝ := (4 * cos θ₁, 4 * sin θ₁)
def B (θ₁ : ℝ) : ℝ × ℝ := (6 * cos (θ₁ - (π / 3)), 6 * sin (θ₁ - (π / 3)))

-- The distance function between two Cartesian points
def distance (P Q : ℝ × ℝ) : ℝ :=
  sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- The theorem to be proven
theorem distance_AB_eq_2_sqrt_3 (θ₁ : ℝ) (h : θ₁ - θ₁ + (π / 3) = π / 3) :
  distance (A θ₁) (B θ₁) = 2 * sqrt 3 :=
sorry

end distance_AB_eq_2_sqrt_3_l352_352914


namespace probability_of_three_integer_points_l352_352926

-- Definitions based on the conditions from a)
def is_center_inside_bounds (v : ℝ × ℝ) : Prop :=
  0 ≤ v.1 ∧ v.1 ≤ 1000 ∧ 0 ≤ v.2 ∧ v.2 ≤ 1000

def diagonal_endpoints : Prop :=
  (1 / 5, 8 / 5) = (1 / 5 : ℝ, 8 / 5 : ℝ) ∧ 
  (-1 / 5, -8 / 5) = (-1 / 5 : ℝ, -8 / 5 : ℝ)

-- Translating the mathematical proof problem to Lean statement
theorem probability_of_three_integer_points (S : set (ℝ × ℝ)) (v : ℝ × ℝ) :
  diagonal_endpoints →
  is_center_inside_bounds v →
  ∃ x : ℝ, ∃ y : ℝ, ∃ p q r : (ℝ × ℝ),
  p ∈ (T(v)) ∧ q ∈ (T(v)) ∧ r ∈ (T(v)) ∧ 
  p ≠ q ∧ q ≠ r ∧ r ≠ p ∧ 
  (p.1.to_int, p.2.to_int) ∈ int ∧ 
  (q.1.to_int, q.2.to_int) ∈ int ∧ 
  (r.1.to_int, r.2.to_int) ∈ int →
  (S.translate(v)).prob_contains_exactly_three_integer_points = 7 / 100 :=
by
  sorry

end probability_of_three_integer_points_l352_352926


namespace perpendicular_vectors_k_zero_l352_352473

theorem perpendicular_vectors_k_zero
  (k : ℝ)
  (a : ℝ × ℝ := (3, 1))
  (b : ℝ × ℝ := (1, 3))
  (c : ℝ × ℝ := (k, 2)) 
  (h : (a.1 - c.1, a.2 - c.2).1 * b.1 + (a.1 - c.1, a.2 - c.2).2 * b.2 = 0) :
  k = 0 :=
by
  sorry

end perpendicular_vectors_k_zero_l352_352473


namespace find_speed_goods_train_l352_352679

def speed_goods_train (V_m : ℕ) (t : ℝ) (d : ℝ) : ℝ :=
  let V_r := (d * 3600) / t
  V_r - V_m

theorem find_speed_goods_train :
  speed_goods_train 100 (9 / 3600) 0.28 = 12 :=
by
  sorry

end find_speed_goods_train_l352_352679


namespace curved_surface_area_of_cone_l352_352623

noncomputable def slant_height : ℝ := 22
noncomputable def radius : ℝ := 7
noncomputable def pi : ℝ := Real.pi

theorem curved_surface_area_of_cone :
  abs (pi * radius * slant_height - 483.22) < 0.01 := 
by
  sorry

end curved_surface_area_of_cone_l352_352623


namespace constant_term_in_expansion_is_60_l352_352031

noncomputable def binomial_expansion_constant_term (n : ℕ) : ℤ :=
  let coeff := (Nat.choose 6 4) in
  let term1 := 2 ^ (6 - 4) in
  let term2 := ((-1) ^ 4) in
  coeff * term1 * term2

theorem constant_term_in_expansion_is_60 (n : ℕ) (h : 2 ^ n = 64) :
  binomial_expansion_constant_term 6 = 60 := by
  sorry

end constant_term_in_expansion_is_60_l352_352031


namespace second_set_number_l352_352209

theorem second_set_number (x : ℕ) (sum1 : ℕ) (avg2 : ℕ) (total_avg : ℕ)
  (h1 : sum1 = 98) (h2 : avg2 = 11) (h3 : total_avg = 8)
  (h4 : 16 + x ≠ 0) :
  (98 + avg2 * x = total_avg * (x + 16)) → x = 10 :=
by
  sorry

end second_set_number_l352_352209


namespace simplify_f_value_of_f_l352_352783

def f (α : Real) : Real := (Real.cos (Real.pi / 2 + α) * Real.cos (Real.pi - α)) / Real.sin (Real.pi + α)

theorem simplify_f (α : Real) : f α = -Real.cos α := 
sorry

theorem value_of_f (α : Real) (hα : α > Real.pi ∧ α < 3 * Real.pi / 2) (hcos : Real.cos (α - 3 * Real.pi / 2) = 1 / 5) :
  f α = 2 * Real.sqrt 6 / 5 := 
sorry

end simplify_f_value_of_f_l352_352783


namespace find_S_m_plus_n_l352_352790

variables (a_n : ℕ → ℕ) (S : ℕ → ℕ)
variables (p q m n : ℕ)

-- Given conditions
def arithmetic_sequence : Prop := ∀ k, S k = p * k^2 + q * k
def Sn_eq_m : Prop := S n = m
def Sm_eq_n : Prop := S m = n
def m_ne_n : Prop := m ≠ n

-- The goal statement to prove
theorem find_S_m_plus_n (h_arith_seq: arithmetic_sequence S a_n)
                        (h_Sn_eq_m : Sn_eq_m S n m)
                        (h_Sm_eq_n : Sm_eq_n S m n)
                        (h_m_ne_n : m_ne_n m n) :
  S (m + n) = -(m + n) :=
by sorry

end find_S_m_plus_n_l352_352790


namespace volume_of_regular_triangular_pyramid_l352_352409

noncomputable def pyramid_volume (a b γ : ℝ) : ℝ :=
  (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2)

theorem volume_of_regular_triangular_pyramid (a b γ : ℝ) :
  pyramid_volume a b γ = (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2) :=
by
  sorry

end volume_of_regular_triangular_pyramid_l352_352409


namespace original_savings_l352_352933

theorem original_savings (tv_cost : ℝ) (furniture_fraction : ℝ) (total_fraction : ℝ) (original_savings : ℝ) :
  tv_cost = 300 → furniture_fraction = 3 / 4 → total_fraction = 1 → 
  (total_fraction - furniture_fraction) * original_savings = tv_cost →
  original_savings = 1200 :=
by 
  intros htv hfurniture htotal hsavings_eq
  sorry

end original_savings_l352_352933


namespace excess_percentage_l352_352530

theorem excess_percentage (A B : ℝ) (x : ℝ) 
  (hA' : A' = A * (1 + x / 100))
  (hB' : B' = B * (1 - 5 / 100))
  (h_area_err : A' * B' = 1.007 * (A * B)) : x = 6 :=
by
  sorry

end excess_percentage_l352_352530


namespace angle_comparison_sin_l352_352028

theorem angle_comparison_sin (a b c : ℝ) (A B C : ℝ) 
  (h_triangle: a = b * sin B / sin A)
  (h1 : sin B > sin C) : 
  B > C :=
sorry

end angle_comparison_sin_l352_352028


namespace minimum_students_both_l352_352717

variable (U : Type) -- U is the type representing the set of all students
variable (S_P S_C : Set U) -- S_P is the set of students who like physics, S_C is the set of students who like chemistry
variable (total_students : Nat) -- total_students is the total number of students
variable [Fintype U] -- U should be a finite set

-- conditions
variable (h_physics : (Fintype.card S_P).toFloat / total_students * 100 = 68)
variable (h_chemistry : (Fintype.card S_C).toFloat / total_students * 100 = 72)

-- statement of the theorem to be proved
theorem minimum_students_both : (Fintype.card (S_P ∩ S_C)).toFloat / total_students * 100 ≥ 40 := by
  sorry

end minimum_students_both_l352_352717


namespace placing_2_flowers_in_2_vases_l352_352637

noncomputable def num_ways_to_place_flowers (n k : ℕ) (h_n : n = 5) (h_k : k = 2) : ℕ :=
  Nat.choose n k * 2

theorem placing_2_flowers_in_2_vases :
  num_ways_to_place_flowers 5 2 rfl rfl = 20 := 
by
  sorry

end placing_2_flowers_in_2_vases_l352_352637


namespace intersection_of_sets_union_of_complement_and_set_l352_352932

def set1 := { x : ℝ | -1 < x ∧ x < 2 }
def set2 := { x : ℝ | x > 0 }
def complement_set2 := { x : ℝ | x ≤ 0 }
def intersection_set := { x : ℝ | 0 < x ∧ x < 2 }
def union_set := { x : ℝ | x < 2 }

theorem intersection_of_sets : 
  { x : ℝ | x ∈ set1 ∧ x ∈ set2 } = intersection_set := 
by 
  sorry

theorem union_of_complement_and_set : 
  { x : ℝ | x ∈ complement_set2 ∨ x ∈ set1 } = union_set := 
by 
  sorry

end intersection_of_sets_union_of_complement_and_set_l352_352932


namespace real_roots_in_intervals_l352_352236

theorem real_roots_in_intervals (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  ∃ x1 x2 : ℝ, (x1 = a / 3 ∨ x1 = -2 * b / 3) ∧ (x2 = a / 3 ∨ x2 = -2 * b / 3) ∧ x1 ≠ x2 ∧
  (a / 3 ≤ x1 ∧ x1 ≤ 2 * a / 3) ∧ (-2 * b / 3 ≤ x2 ∧ x2 ≤ -b / 3) ∧
  (x1 > 0 ∧ x2 < 0) ∧ (1 / x1 + 1 / (x1 - a) + 1 / (x1 + b) = 0) ∧
  (1 / x2 + 1 / (x2 - a) + 1 / (x2 + b) = 0) :=
sorry

end real_roots_in_intervals_l352_352236


namespace cyclic_quadrilateral_l352_352029

/-- Definitions of the points and properties regarding the incircle and internal point of a triangle --/
variables {A B C X D E F Y Z : Type*}
  [incircle_triangle_ABC : (D,E,F)]
  [internal_point_triangle_ABC : (X)]
  [incircle_triangle_XBC : (D,Y,Z)]

/-- Proof problem -/
theorem cyclic_quadrilateral (h1 : incircle_triangle_ABC = (D, E, F))
                            (h2 : internal_point_triangle_ABC = X)
                            (h3 : incircle_triangle_XBC = (D, Y, Z)) :
  cyclic_quadrilateral(E, F, Z, Y) :=
sorry

end cyclic_quadrilateral_l352_352029


namespace new_light_wattage_l352_352268

theorem new_light_wattage (w_old : ℕ) (p : ℕ) (w_new : ℕ) (h1 : w_old = 110) (h2 : p = 30) (h3 : w_new = w_old + (p * w_old / 100)) : w_new = 143 :=
by
  -- Using the conditions provided
  rw [h1, h2] at h3
  norm_num at h3
  exact h3

end new_light_wattage_l352_352268


namespace inequality_holds_up_to_4_inequality_fails_at_5_l352_352751

theorem inequality_holds_up_to_4 (n : ℕ) (h : n ∈ {1, 2, 3, 4}) :
  ∀ (a : Fin n → ℝ), (∀ i, 0 < a i ∧ a i ≤ 1) →
    (∑ i, (Real.sqrt (1 - a i)) / (a i)) ≤ 1 / (∏ i, a i) := by
  sorry

theorem inequality_fails_at_5 :
  ∃ (a : Fin 5 → ℝ), (∀ i, 0 < a i ∧ a i ≤ 1) ∧
    ((∑ i, (Real.sqrt (1 - a i)) / (a i)) > 1 / (∏ i, a i)) := by
  sorry

end inequality_holds_up_to_4_inequality_fails_at_5_l352_352751


namespace matrix_A_to_power_4_l352_352327

def matrix_A : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![2, -1], ![1, 1]]

def matrix_pow4 : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, -9], ![9, -9]]

theorem matrix_A_to_power_4 :
  matrix_A ^ 4 = matrix_pow4 :=
by
  sorry

end matrix_A_to_power_4_l352_352327


namespace simplify_division_l352_352954

theorem simplify_division :
  (27 * 10^9) / (9 * 10^5) = 30000 :=
  sorry

end simplify_division_l352_352954


namespace competition_end_time_is_5_35_am_l352_352262

def start_time : Nat := 15 * 60  -- 3:00 p.m. in minutes
def duration : Nat := 875  -- competition duration in minutes
def end_time : Nat := (start_time + duration) % (24 * 60)  -- competition end time in minutes

theorem competition_end_time_is_5_35_am :
  end_time = 5 * 60 + 35 :=  -- 5:35 a.m. in minutes
sorry

end competition_end_time_is_5_35_am_l352_352262


namespace number_of_ways_is_60_l352_352676

def sections : ℕ := 4

inductive Crop
| carrots
| lettuce
| tomatoes
| cucumbers

open Crop

def adjacency_condition_1 (s : Fin 4 → Crop) : Prop :=
  ∀ i : Fin 3, ¬(s i = carrots ∧ s (i + 1) = lettuce) ∧ ¬(s i = lettuce ∧ s (i + 1) = carrots)

def adjacency_condition_2 (s : Fin 4 → Crop) : Prop :=
  ∀ i : Fin 3, ¬(s i = tomatoes ∧ s (i + 1) = cucumbers) ∧ ¬(s i = cucumbers ∧ s (i + 1) = tomatoes)

def valid_arrangement (s : Fin 4 → Crop) : Prop :=
  adjacency_condition_1 s ∧ adjacency_condition_2 s

def num_ways_to_plant_crops : ℕ :=
  (multiset.filter valid_arrangement (finset.pi_finset (finset.univ : finset Crop)).val).card

theorem number_of_ways_is_60 : num_ways_to_plant_crops = 60 :=
by sorry

end number_of_ways_is_60_l352_352676


namespace radius_of_Q3_l352_352432

-- Define the given side lengths
def AB : ℝ := 78
def BC : ℝ := 78
def AC : ℝ := 60

-- Define the triangle \(ABC\) as isosceles with the given side lengths
def is_isosceles_triangle (A B C : Type) (AB BC AC : ℝ) : Prop :=
  AB = BC ∧ AB = 78 ∧ BC = 78 ∧ AC = 60

-- Define the radius \(r_1\) of circle \(Q_1\) as the inscribed circle in triangle \(ABC\)
def r1 (A B C : Type) (AB BC AC : ℝ) (area : ℝ) (s : ℝ) : ℝ :=
  let s := (AB + BC + AC) / 2
  let area := 2160 -- This is computed as per the solution
  area / s

-- Define the similarity ratio \(k\) between the subsequent triangles
def similarity_ratio : ℝ := 32 / 72

-- Define the radius \(r_2\) of circle \(Q_2\)
def r2 (A B C : Type) (r1 : ℝ) (k : ℝ) : ℝ :=
  k * r1

-- Define the radius \(r_3\) of circle \(Q_3\)
def r3 (r2 : ℝ) (k : ℝ) : ℝ :=
  k * r2

-- Top-level statement to prove
theorem radius_of_Q3 (A B C : Type) : is_isosceles_triangle A B C AB BC AC ->
  r3 (r2 A B C (r1 A B C AB BC AC 2160 108) similarity_ratio) similarity_ratio = 320 / 81 :=
by
  sorry

end radius_of_Q3_l352_352432


namespace minimum_value_l352_352099

theorem minimum_value (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + (1 / y)) * (x + (1 / y) - 1024) +
  (y + (1 / x)) * (y + (1 / x) - 1024) ≥ -524288 :=
by sorry

end minimum_value_l352_352099


namespace sum_difference_l352_352220

def even_sum (n : ℕ) : ℕ :=
  n * (n + 1)

def odd_sum (n : ℕ) : ℕ :=
  n^2

theorem sum_difference : even_sum 100 - odd_sum 100 = 100 := by
  sorry

end sum_difference_l352_352220


namespace playground_width_l352_352648

theorem playground_width 
  (W : ℕ)   -- Width of the playground
  (L : ℕ)   -- Length of the garden
  (GardenWidth : ℕ := 4)  -- Width of the garden
  (PlaygroundLength : ℕ := 16)  -- Length of the playground
  (GardenPerimeter : ℕ := 104)  -- Perimeter of the garden
  (garden_perim_eq : 2 * L + 2 * GardenWidth = GardenPerimeter)
  (area_eq : PlaygroundLength * W = GardenWidth * L) :
  W = 12 :=
begin
  sorry
end

end playground_width_l352_352648


namespace steven_more_peaches_than_apples_l352_352903

def steven_peaches : Nat := 17
def steven_apples : Nat := 16

theorem steven_more_peaches_than_apples : steven_peaches - steven_apples = 1 := by
  sorry

end steven_more_peaches_than_apples_l352_352903


namespace minimum_value_lambda_l352_352807

noncomputable def minimum_lambda : ℝ :=
  Inf {λ | λ > 0 ∧ (∀ x: ℝ, x > Real.exp 2 → λ * Real.exp(λ * x) - Real.log x ≥ 0)}
  
theorem minimum_value_lambda : minimum_lambda = 2 / Real.exp 2 := sorry

end minimum_value_lambda_l352_352807


namespace total_people_museum_l352_352198

def bus1 := 12
def bus2 := 2 * bus1
def bus3 := bus2 - 6
def bus4 := bus1 + 9
def total := bus1 + bus2 + bus3 + bus4

theorem total_people_museum : total = 75 := by
  sorry

end total_people_museum_l352_352198


namespace sum_of_sequence_l352_352459

noncomputable def f (x : ℝ) (α : ℝ) : ℝ := x ^ α

def a_n (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  1 / (f (n + 1) + f n)

def S_n (f : ℝ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum (λ i, a_n f (i + 1))

theorem sum_of_sequence :
  (∀ x, f x (1/2) = x ^ (1/2)) ∧ f 4 (1/2) = 2 →
  S_n (λ x, x ^ (1/2)) 2017 = sqrt 2018 - 1 := by
sorry

end sum_of_sequence_l352_352459


namespace probability_no_bribed_correct_l352_352885

open Nat

noncomputable def binom (n k : ℕ) : ℕ := choose n k

def probability_no_bribed_judges (N B s : ℕ) :=
  let V := N - B
  (binom V s) * (binom B 0) / (binom N s)

theorem probability_no_bribed_correct:
  (probability_no_bribed_judges 14 2 7) = (3 / 13 : ℚ) :=
by
  let N := 14
  let B := 2
  let V := N - B
  let s := 7
  rw [probability_no_bribed_judges, V, s, N, B]
  sorry

end probability_no_bribed_correct_l352_352885


namespace magnitude_b_eq_one_l352_352049

open Real

variables {V : Type*} [inner_product_space ℝ V]

theorem magnitude_b_eq_one
  (a b : V)
  (ha : ‖a‖ = 1)
  (hab : ‖a + b‖ = 1)
  (θ : real.angle)
  (hθ : θ = real.angle.of_deg 120)
  (h_dot : ⟪a, b⟫ = ‖b‖ * (cos θ)) :
  ‖b‖ = 1 :=
by
  -- This is just to keep the statement compilable.
  sorry

end magnitude_b_eq_one_l352_352049


namespace perimeter_of_square_l352_352964

theorem perimeter_of_square (s : ℝ) (hs : s^2 = 500) : 4 * s = 40 * Real.sqrt 5 := 
by
  have h_sqrt : s = Real.sqrt 500 := 
    by
      -- Proof of s being sqrt(500) is skipped for brevity (assuming it's well-known)
      sorry
  have h_perimeter : 4 * Real.sqrt 500 = 40 * Real.sqrt 5 := 
    by
      -- Simplification steps (assuming they are well-known or trivial)
      sorry
  exact eq.trans (congr_arg (λ x, 4 * x) h_sqrt) h_perimeter

end perimeter_of_square_l352_352964


namespace complement_intersection_l352_352471

open Set

-- Define the universal set U
def U : Set ℕ := {1, 2, 3, 4}

-- Define the set A
def A : Set ℕ := {1, 2}

-- Define the set B
def B : Set ℕ := {2, 3, 4}

-- Statement to be proven
theorem complement_intersection :
  (U \ A) ∩ B = {3, 4} :=
sorry

end complement_intersection_l352_352471


namespace pet_store_initial_house_cats_l352_352681

theorem pet_store_initial_house_cats
    (H : ℕ)
    (h1 : 13 + H - 10 = 8) :
    H = 5 :=
by
  sorry

end pet_store_initial_house_cats_l352_352681


namespace best_graph_for_percentage_of_components_in_air_l352_352698

-- Definitions of the graph types
def is_line_graph (g : Type) : Prop := ∃ t : Type, g = trend_over_time t
def is_bar_graph (g : Type) : Prop := ∃ c : Type, g = compare_categories c
def is_histogram (g : Type) : Prop := ∃ h : Type, g = frequency_distribution h
def is_pie_chart (g : Type) : Prop := ∃ p : Type, g = parts_of_whole p

-- Main statement
theorem best_graph_for_percentage_of_components_in_air (g : Type)
    (h_line_graph : is_line_graph g → False)
    (h_bar_graph : is_bar_graph g → False)
    (h_histogram : is_histogram g → False)
    (h_pie_chart : is_pie_chart g → True) :
    is_pie_chart g :=
by
  intro
  sorry

end best_graph_for_percentage_of_components_in_air_l352_352698


namespace find_pair_r_x_l352_352252

noncomputable def x_repr (r : ℕ) (n : ℕ) (ab : ℕ) : ℕ :=
  ab * (r * (r^(2*(n-1)) - 1) / (r^2 - 1))

noncomputable def x_squared_repr (r : ℕ) (n : ℕ) : ℕ :=
  (r^(4*n) - 1) / (r - 1)

theorem find_pair_r_x (r x : ℕ) (n : ℕ) (ab : ℕ) (r_leq_70 : r ≤ 70)
  (x_consistent: x = x_repr r n ab)
  (x_squared_consistent: x^2 = x_squared_repr r n)
  : (r = 7 ∧ x = 20) :=
begin
  sorry
end

end find_pair_r_x_l352_352252


namespace min_percentage_both_physics_chemistry_l352_352716

/--
Given:
- A certain school conducted a survey.
- 68% of the students like physics.
- 72% of the students like chemistry.

Prove that the minimum percentage of students who like both physics and chemistry is 40%.
-/
theorem min_percentage_both_physics_chemistry (P C : ℝ)
(hP : P = 0.68) (hC : C = 0.72) :
  ∃ B, B = P + C - 1 ∧ B = 0.40 :=
by
  sorry

end min_percentage_both_physics_chemistry_l352_352716


namespace perpendicular_lines_parallel_to_same_plane_are_parallel_l352_352022

variables {Point : Type} [linear_space Point]
variables (l m : set Point) (α : set Point)

def perpendicular (line : set Point) (plane : set Point) : Prop :=
  ∀ p₁ p₂ ∈ line, ∀ q₁ q₂ ∈ plane, ((q₁ - q₂) • (p₁ - p₂) = 0)

def parallel (line1 line2 : set Point) : Prop :=
  ∀ p₁ p₂ ∈ line1, ∀ q₁ q₂ ∈ line2, ((p₂ - p₁) = k * (q₂ - q₁)) ∨ ((p₁ - p₂) = k * (q₂ - q₁)) for some k : ℝ

theorem perpendicular_lines_parallel_to_same_plane_are_parallel :
  perpendicular l α → perpendicular m α → parallel l m :=
by sorry

end perpendicular_lines_parallel_to_same_plane_are_parallel_l352_352022


namespace sales_maximized_sales_amount_greater_l352_352671

-- Part 1
theorem sales_maximized (a : ℝ) (ha1 : 1/3 ≤ a) (ha2 : a < 1) :
  ∃ x : ℝ, x = 5 * (1 - a) / a := sorry

-- Part 2
theorem sales_amount_greater (x : ℝ) :
  (2 * x / 3 ≤ 3 * 10) ∧ (20 / 3 - 2 * x / 3 > 0) ↔ 0 < x ∧ x < 5 := sorry

end sales_maximized_sales_amount_greater_l352_352671


namespace area_OPA_l352_352534

variable (x : ℝ)

def y (x : ℝ) : ℝ := -x + 6

def A : ℝ × ℝ := (4, 0)
def O : ℝ × ℝ := (0, 0)
def P (x : ℝ) : ℝ × ℝ := (x, y x)

def area_triangle (O A P : ℝ × ℝ) : ℝ := 
  0.5 * abs (A.fst * P.snd + P.fst * O.snd + O.fst * A.snd - A.snd * P.fst - P.snd * O.fst - O.snd * A.fst)

theorem area_OPA : 0 < x ∧ x < 6 → area_triangle O A (P x) = 12 - 2 * x := by
  -- proof to be provided here
  sorry


end area_OPA_l352_352534


namespace find_BD_l352_352072

noncomputable def distance {α : Type*} [metric_space α] (x y : α) : ℝ := dist x y

variable {A B C D : Type*}
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D]

def condition1 (A B C : Type*) [metric_space A] [metric_space B] [metric_space C] : Prop :=
distance A C = 10 ∧ distance B C = 10 ∧ distance A B = 4

def condition2 (A B D : Type*) [metric_space A] [metric_space B] [metric_space D] : Prop :=
distance B D = distance A B + distance B D - 2 * distance A B

theorem find_BD {A B C D : Type*} [metric_space A] [metric_space B] [metric_space C] [metric_space D]
  (h1 : condition1 A B C) (h2 : condition2 A B D) (h3 : distance C D = 12) : 
  distance B D = 4 * real.sqrt 3 - 2 :=
sorry

end find_BD_l352_352072


namespace binom_12_10_eq_66_l352_352354

theorem binom_12_10_eq_66 : (nat.choose 12 10) = 66 := by
  sorry

end binom_12_10_eq_66_l352_352354


namespace triangle_perimeter_l352_352440

theorem triangle_perimeter (m : ℝ) (a b : ℝ) (h1 : 3 ^ 2 - 3 * (m + 1) + 2 * m = 0)
  (h2 : a ^ 2 - (m + 1) * a + 2 * m = 0)
  (h3 : b ^ 2 - (m + 1) * b + 2 * m = 0)
  (h4 : a = 3 ∨ b = 3)
  (h5 : a ≠ b ∨ a = b)
  (hAB : a ≠ b ∨ a = b) :
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ ≠ s₂ → s₁ + s₁ + s₂ = 10 ∨ s₁ + s₁ + s₂ = 11) ∨
  (∀ s₁ s₂ : ℝ, s₁ = a ∨ s₁ = b ∧ s₂ = a ∨ s₂ = b ∧ s₁ = s₂ → b + b + a = 10 ∨ b + b + a = 11) := by
  sorry

end triangle_perimeter_l352_352440


namespace three_digit_number_digits_difference_l352_352288

theorem three_digit_number_digits_difference (a b c : ℕ) (h1 : b = a + 1) (h2 : c = a + 2) (h3 : a < b) (h4 : b < c) :
  let original_number := 100 * a + 10 * b + c
  let reversed_number := 100 * c + 10 * b + a
  reversed_number - original_number = 198 := by
  sorry

end three_digit_number_digits_difference_l352_352288


namespace simplify_fraction_eq_one_over_thirty_nine_l352_352588

theorem simplify_fraction_eq_one_over_thirty_nine :
  let a1 := (1 / 3)^1
  let a2 := (1 / 3)^2
  let a3 := (1 / 3)^3
  (1 / (1 / a1 + 1 / a2 + 1 / a3)) = 1 / 39 :=
by
  sorry

end simplify_fraction_eq_one_over_thirty_nine_l352_352588


namespace average_price_of_returned_products_l352_352109

theorem average_price_of_returned_products:
  let initial_products := 15
  let initial_avg_price := 30 -- in cents
  let total_initial_cost := initial_avg_price * initial_products
  let returned_products := 4
  let remaining_products := initial_products - returned_products
  let remaining_avg_price := 25 -- in cents
  let total_remaining_cost := remaining_avg_price * remaining_products
  let total_returned_cost := total_initial_cost - total_remaining_cost
  let avg_returned_price := total_returned_cost / returned_products
  in avg_returned_price = 43.75 :=
by {
  -- Variables introduction based on conditions
  let initial_products := 15
  let initial_avg_price := 30 -- in cents
  let total_initial_cost := initial_avg_price * initial_products
  let returned_products := 4
  let remaining_products := initial_products - returned_products
  let remaining_avg_price := 25 -- in cents
  let total_remaining_cost := remaining_avg_price * remaining_products
  let total_returned_cost := total_initial_cost - total_remaining_cost
  let avg_returned_price := total_returned_cost / returned_products

  -- Proof with calculation
  have h_initial_total_cost := initial_avg_price * initial_products
  have h_total_remaining_cost := remaining_avg_price * remaining_products
  have h_total_returned_cost := h_initial_total_cost - h_total_remaining_cost
  have h_avg_returned_price := h_total_returned_cost / returned_products
  have h_final := h_avg_returned_price = 43.75

  exact h_final
}

end average_price_of_returned_products_l352_352109


namespace probability_rachel_robert_in_picture_l352_352945

theorem probability_rachel_robert_in_picture :
  let lap_rachel := 120 -- Rachel's lap time in seconds
  let lap_robert := 100 -- Robert's lap time in seconds
  let duration := 900 -- 15 minutes in seconds
  let picture_duration := 60 -- Picture duration in seconds
  let one_third_rachel := lap_rachel / 3 -- One third of Rachel's lap time
  let one_third_robert := lap_robert / 3 -- One third of Robert's lap time
  let rachel_in_window_start := 20 -- Rachel in the window from 20 to 100s
  let rachel_in_window_end := 100
  let robert_in_window_start := 0 -- Robert in the window from 0 to 66.66s
  let robert_in_window_end := 66.66
  let overlap_start := max rachel_in_window_start robert_in_window_start -- The start of overlap
  let overlap_end := min rachel_in_window_end robert_in_window_end -- The end of overlap
  let overlap_duration := overlap_end - overlap_start -- Duration of the overlap
  let probability := overlap_duration / picture_duration -- Probability of both in the picture
  probability = 46.66 / 60 := sorry

end probability_rachel_robert_in_picture_l352_352945


namespace number_of_toys_sold_l352_352271

theorem number_of_toys_sold (total_selling_price gain_per_toy cost_price_per_toy : ℕ)
  (h1 : total_selling_price = 25200)
  (h2 : gain_per_toy = 3 * cost_price_per_toy)
  (h3 : cost_price_per_toy = 1200) : 
  (total_selling_price - gain_per_toy) / cost_price_per_toy = 18 :=
by 
  sorry

end number_of_toys_sold_l352_352271


namespace num_teams_is_seventeen_l352_352635

-- Each team faces all other teams 10 times and there are 1360 games in total.
def total_teams (n : ℕ) : Prop := 1360 = (n * (n - 1) * 10) / 2

theorem num_teams_is_seventeen : ∃ n : ℕ, total_teams n ∧ n = 17 := 
by 
  sorry

end num_teams_is_seventeen_l352_352635


namespace probability_mixed_doubles_l352_352638

def num_athletes : ℕ := 6
def num_males : ℕ := 3
def num_females : ℕ := 3
def num_coaches : ℕ := 3

def choose (n k : ℕ) : ℕ := Nat.choose n k

-- Total number of ways to select athletes
def total_ways : ℕ :=
  (choose num_athletes 2) * (choose (num_athletes - 2) 2) * (choose (num_athletes - 4) 2)

-- Number of favorable ways to select mixed doubles teams
def favorable_ways : ℕ :=
  (choose num_males 1) * (choose num_females 1) *
  (choose (num_males - 1) 1) * (choose (num_females - 1) 1) *
  (choose 1 1) * (choose 1 1)

-- Probability calculation
def probability : ℚ := favorable_ways / total_ways

theorem probability_mixed_doubles :
  probability = 2/5 :=
by
  sorry

end probability_mixed_doubles_l352_352638
