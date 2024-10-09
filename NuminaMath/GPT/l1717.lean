import Mathlib

namespace smallest_k_l1717_171764

def u (n : ℕ) : ℕ := n^4 + 3 * n^2 + 2

def delta (k : ℕ) (u : ℕ → ℕ) : ℕ → ℕ :=
  match k with
  | 0 => u
  | k+1 => fun n => delta k u (n+1) - delta k u n

theorem smallest_k (n : ℕ) : ∃ k, (forall m, delta k u m = 0) ∧ 
                            (forall j, (∀ m, delta j u m = 0) → j ≥ k) := sorry

end smallest_k_l1717_171764


namespace fraction_of_phone_numbers_l1717_171754

-- Define the total number of valid 7-digit phone numbers
def totalValidPhoneNumbers : Nat := 7 * 10^6

-- Define the number of valid phone numbers that begin with 3 and end with 5
def validPhoneNumbersBeginWith3EndWith5 : Nat := 10^5

-- Prove the fraction of phone numbers that begin with 3 and end with 5 is 1/70
theorem fraction_of_phone_numbers (h : validPhoneNumbersBeginWith3EndWith5 = 10^5) 
(h2 : totalValidPhoneNumbers = 7 * 10^6) : 
validPhoneNumbersBeginWith3EndWith5 / totalValidPhoneNumbers = 1 / 70 := 
sorry

end fraction_of_phone_numbers_l1717_171754


namespace number_of_birds_is_122_l1717_171733

-- Defining the variables
variables (b m i : ℕ)

-- Define the conditions as part of an axiom
axiom heads_count : b + m + i = 300
axiom legs_count : 2 * b + 4 * m + 6 * i = 1112

-- We aim to prove the number of birds is 122
theorem number_of_birds_is_122 (h1 : b + m + i = 300) (h2 : 2 * b + 4 * m + 6 * i = 1112) : b = 122 := by
  sorry

end number_of_birds_is_122_l1717_171733


namespace prism_surface_area_l1717_171705

-- Define the base of the prism as an isosceles trapezoid ABCD
structure Trapezoid :=
(AB CD : ℝ)
(BC : ℝ)
(AD : ℝ)

-- Define the properties of the prism
structure Prism :=
(base : Trapezoid)
(diagonal_cross_section_area : ℝ)

-- Define the specific isosceles trapezoid from the problem
def myTrapezoid : Trapezoid :=
{ AB := 13, CD := 13, BC := 11, AD := 21 }

-- Define the specific prism from the problem with the given conditions
noncomputable def myPrism : Prism :=
{ base := myTrapezoid, diagonal_cross_section_area := 180 }

-- Define the total surface area as a function
noncomputable def total_surface_area (p : Prism) : ℝ :=
2 * (1 / 2 * (p.base.AD + p.base.BC) * (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2))) +
(p.base.AB + p.base.BC + p.base.CD + p.base.AD) * (p.diagonal_cross_section_area / (Real.sqrt ((1 / 2 * (p.base.AD + p.base.BC)) ^ 2 + (Real.sqrt ((p.base.CD) ^ 2 - ((p.base.AD - p.base.BC) / 2) ^ 2)) ^ 2)))

-- The proof problem in Lean
theorem prism_surface_area :
  total_surface_area myPrism = 906 :=
sorry

end prism_surface_area_l1717_171705


namespace division_addition_l1717_171770

theorem division_addition (n : ℕ) (h : 32 - 16 = n * 4) : n / 4 + 16 = 17 :=
by 
  sorry

end division_addition_l1717_171770


namespace trajectory_moving_point_hyperbola_l1717_171784

theorem trajectory_moving_point_hyperbola {n m : ℝ} (h_neg_n : n < 0) :
    (∃ y < 0, (y^2 = 16) ∧ (m^2 = (n^2 / 4 - 4))) ↔ ( ∃ (y : ℝ), (y^2 / 16) - (m^2 / 4) = 1 ∧ y < 0 ) := 
sorry

end trajectory_moving_point_hyperbola_l1717_171784


namespace pythagorean_triples_l1717_171747

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem pythagorean_triples :
  is_pythagorean_triple 3 4 5 ∧ is_pythagorean_triple 6 8 10 :=
by
  sorry

end pythagorean_triples_l1717_171747


namespace stock_percent_change_l1717_171720

-- define initial value of stock
def initial_stock_value (x : ℝ) := x

-- define value after first day's decrease
def value_after_day_one (x : ℝ) := 0.85 * x

-- define value after second day's increase
def value_after_day_two (x : ℝ) := 1.25 * value_after_day_one x

-- Theorem stating the overall percent change is 6.25%
theorem stock_percent_change (x : ℝ) (h : x > 0) :
  ((value_after_day_two x - initial_stock_value x) / initial_stock_value x) * 100 = 6.25 := by sorry

end stock_percent_change_l1717_171720


namespace isosceles_triangle_circumscribed_radius_and_height_l1717_171798

/-
Conditions:
- The isosceles triangle has two equal sides of 20 inches.
- The base of the triangle is 24 inches.

Prove:
1. The radius of the circumscribed circle is 5 inches.
2. The height of the triangle is 16 inches.
-/

theorem isosceles_triangle_circumscribed_radius_and_height 
  (h_eq_sides : ∀ A B C : Type, ∀ (AB AC : ℝ), ∀ (BC : ℝ), AB = 20 → AC = 20 → BC = 24) 
  (R : ℝ) (h : ℝ) : 
  R = 5 ∧ h = 16 := 
sorry

end isosceles_triangle_circumscribed_radius_and_height_l1717_171798


namespace not_all_prime_distinct_l1717_171758

theorem not_all_prime_distinct (a1 a2 a3 : ℕ) (h1 : a1 ≠ a2) (h2 : a2 ≠ a3) (h3 : a1 ≠ a3)
  (h4 : 0 < a1) (h5 : 0 < a2) (h6 : 0 < a3)
  (h7 : a1 ∣ (a2 + a3 + a2 * a3)) (h8 : a2 ∣ (a3 + a1 + a3 * a1)) (h9 : a3 ∣ (a1 + a2 + a1 * a2)) :
  ¬ (Nat.Prime a1 ∧ Nat.Prime a2 ∧ Nat.Prime a3) :=
by
  sorry

end not_all_prime_distinct_l1717_171758


namespace magnitude_of_b_l1717_171701

variables (a b : EuclideanSpace ℝ (Fin 3)) (θ : ℝ)

-- Defining the conditions
def vector_a_magnitude : Prop := ‖a‖ = 1
def vector_angle_condition : Prop := θ = Real.pi / 3
def linear_combination_magnitude : Prop := ‖2 • a - b‖ = 2 * Real.sqrt 3
def b_magnitude : Prop := ‖b‖ = 4

-- The statement we want to prove
theorem magnitude_of_b (h1 : vector_a_magnitude a) (h2 : vector_angle_condition θ) (h3 : linear_combination_magnitude a b) : b_magnitude b :=
sorry

end magnitude_of_b_l1717_171701


namespace girls_in_class4_1_l1717_171710

theorem girls_in_class4_1 (total_students grade: ℕ)
    (total_girls: ℕ)
    (students_class4_1: ℕ)
    (boys_class4_2: ℕ)
    (h1: total_students = 72)
    (h2: total_girls = 35)
    (h3: students_class4_1 = 36)
    (h4: boys_class4_2 = 19) :
    (total_girls - (total_students - students_class4_1 - boys_class4_2) = 18) :=
by
    sorry

end girls_in_class4_1_l1717_171710


namespace merchant_cost_price_l1717_171778

theorem merchant_cost_price (x : ℝ) (h₁ : x + (x^2 / 100) = 39) : x = 30 :=
sorry

end merchant_cost_price_l1717_171778


namespace problem1_problem2_l1717_171765

-- Problem 1: Prove that 3 * sqrt(20) - sqrt(45) + sqrt(1 / 5) = (16 * sqrt(5)) / 5
theorem problem1 : 3 * Real.sqrt 20 - Real.sqrt 45 + Real.sqrt (1 / 5) = (16 * Real.sqrt 5) / 5 := 
sorry

-- Problem 2: Prove that (sqrt(6) - 2 * sqrt(3))^2 - (2 * sqrt(5) + sqrt(2)) * (2 * sqrt(5) - sqrt(2)) = -12 * sqrt(2)
theorem problem2 : (Real.sqrt 6 - 2 * Real.sqrt 3) ^ 2 - (2 * Real.sqrt 5 + Real.sqrt 2) * (2 * Real.sqrt 5 - Real.sqrt 2) = -12 * Real.sqrt 2 := 
sorry

end problem1_problem2_l1717_171765


namespace wool_production_equivalence_l1717_171762

variable (x y z w v : ℕ)

def wool_per_sheep_of_breed_A_per_day : ℚ :=
  (y:ℚ) / ((x:ℚ) * (z:ℚ))

def wool_per_sheep_of_breed_B_per_day : ℚ :=
  2 * wool_per_sheep_of_breed_A_per_day x y z

def total_wool_produced_by_breed_B (x y z w v: ℕ) : ℚ :=
  (w:ℚ) * wool_per_sheep_of_breed_B_per_day x y z * (v:ℚ)

theorem wool_production_equivalence :
  total_wool_produced_by_breed_B x y z w v = 2 * (y:ℚ) * (w:ℚ) * (v:ℚ) / ((x:ℚ) * (z:ℚ)) := by
  sorry

end wool_production_equivalence_l1717_171762


namespace reciprocal_roots_k_value_l1717_171700

theorem reciprocal_roots_k_value :
  ∀ k : ℝ, (∀ r : ℝ, 5.2 * r^2 + 14.3 * r + k = 0 ∧ 5.2 * (1 / r)^2 + 14.3 * (1 / r) + k = 0) →
          k = 5.2 :=
by
  sorry

end reciprocal_roots_k_value_l1717_171700


namespace circle_radii_l1717_171797

noncomputable def smaller_circle_radius (r : ℝ) :=
  r = 4

noncomputable def larger_circle_radius (r : ℝ) :=
  r = 9

theorem circle_radii (r : ℝ) (h1 : ∀ (r: ℝ), (r + 5) - r = 5) (h2 : ∀ (r: ℝ), 2.4 * r = 2.4 * r):
  smaller_circle_radius r → larger_circle_radius (r + 5) :=
by
  sorry

end circle_radii_l1717_171797


namespace necessary_but_not_sufficient_condition_l1717_171711

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 2 → x > 1) ∧ ¬(x > 1 → x > 2) :=
by
  sorry

end necessary_but_not_sufficient_condition_l1717_171711


namespace sum_of_smallest_ns_l1717_171777

theorem sum_of_smallest_ns : ∀ n1 n2 : ℕ, (n1 ≡ 1 [MOD 4] ∧ n1 ≡ 2 [MOD 7]) ∧ (n2 ≡ 1 [MOD 4] ∧ n2 ≡ 2 [MOD 7]) ∧ n1 < n2 →
  n1 = 9 ∧ n2 = 37 → (n1 + n2 = 46) :=
by
  sorry

end sum_of_smallest_ns_l1717_171777


namespace largest_possible_value_l1717_171728

-- Definitions for the conditions
def lower_x_bound := -4
def upper_x_bound := -2
def lower_y_bound := 2
def upper_y_bound := 4

-- The proposition to prove
theorem largest_possible_value (x y : ℝ) 
    (h1 : lower_x_bound ≤ x) (h2 : x ≤ upper_x_bound)
    (h3 : lower_y_bound ≤ y) (h4 : y ≤ upper_y_bound) :
    ∃ v, v = (x + y) / x ∧ ∀ (w : ℝ), w = (x + y) / x → w ≤ 1/2 :=
by
  sorry

end largest_possible_value_l1717_171728


namespace percentage_heavier_l1717_171781

variables (J M : ℝ)

theorem percentage_heavier (hM : M ≠ 0) : 
  100 * ((J + 3) - M) / M = 100 * ((J + 3) - M) / M := 
sorry

end percentage_heavier_l1717_171781


namespace total_trip_hours_l1717_171718

-- Define the given conditions
def speed1 := 50 -- Speed in mph for the first 4 hours
def time1 := 4 -- First 4 hours
def distance1 := speed1 * time1 -- Distance covered in the first 4 hours

def speed2 := 80 -- Speed in mph for additional hours
def average_speed := 65 -- Average speed for the entire trip

-- Define the proof problem
theorem total_trip_hours (T : ℕ) (A : ℕ) :
  distance1 + (speed2 * A) = average_speed * T ∧ T = time1 + A → T = 8 :=
by
  sorry

end total_trip_hours_l1717_171718


namespace seashells_given_l1717_171719

theorem seashells_given (initial left given : ℕ) (h1 : initial = 8) (h2 : left = 2) (h3 : given = initial - left) : given = 6 := by
  sorry

end seashells_given_l1717_171719


namespace john_bought_two_shirts_l1717_171746

/-- The number of shirts John bought, given the conditions:
1. The first shirt costs $6 more than the second shirt.
2. The first shirt costs $15.
3. The total cost of the shirts is $24,
is equal to 2. -/
theorem john_bought_two_shirts
  (S : ℝ) 
  (first_shirt_cost : ℝ := 15)
  (second_shirt_cost : ℝ := S)
  (cost_difference : first_shirt_cost = second_shirt_cost + 6)
  (total_cost : first_shirt_cost + second_shirt_cost = 24)
  : 2 = 2 :=
by
  sorry

end john_bought_two_shirts_l1717_171746


namespace max_n_factoring_polynomial_l1717_171750

theorem max_n_factoring_polynomial :
  ∃ n A B : ℤ, (3 * n + A = 217) ∧ (A * B = 72) ∧ (3 * B + A = n) :=
sorry

end max_n_factoring_polynomial_l1717_171750


namespace general_term_correct_S_maximum_value_l1717_171775

noncomputable def general_term (n : ℕ) : ℤ :=
  if n = 1 then -1 + 24 else (-n^2 + 24 * n) - (-(n - 1)^2 + 24 * (n - 1))

noncomputable def S (n : ℕ) : ℤ :=
  -n^2 + 24 * n

theorem general_term_correct (n : ℕ) (h : 1 ≤ n) : general_term n = -2 * n + 25 := by
  sorry

theorem S_maximum_value : ∃ n : ℕ, S n = 144 ∧ ∀ m : ℕ, S m ≤ 144 := by
  existsi 12
  sorry

end general_term_correct_S_maximum_value_l1717_171775


namespace arithmetic_seq_sum_l1717_171736

theorem arithmetic_seq_sum (a : ℕ → ℝ) (h1 : ∀ n, a (n+1) - a n = a (n+2) - a (n+1))
  (h2 : a 3 + a 7 = 37) :
  a 2 + a 4 + a 6 + a 8 = 74 := 
sorry

end arithmetic_seq_sum_l1717_171736


namespace find_reciprocal_sum_l1717_171708

theorem find_reciprocal_sum
  (m n : ℕ)
  (h_sum : m + n = 72)
  (h_hcf : Nat.gcd m n = 6)
  (h_lcm : Nat.lcm m n = 210) :
  (1 / (m : ℚ)) + (1 / (n : ℚ)) = 6 / 105 :=
by
  sorry

end find_reciprocal_sum_l1717_171708


namespace octagon_perimeter_l1717_171753

theorem octagon_perimeter (n : ℕ) (side_length : ℝ) (h1 : n = 8) (h2 : side_length = 2) : 
  n * side_length = 16 :=
by
  sorry

end octagon_perimeter_l1717_171753


namespace max_xy_l1717_171703

theorem max_xy (x y : ℕ) (h1: 7 * x + 4 * y = 140) : ∃ x y, 7 * x + 4 * y = 140 ∧ x * y = 168 :=
by {
  sorry
}

end max_xy_l1717_171703


namespace maria_made_144_cookies_l1717_171707

def cookies (C : ℕ) : Prop :=
  (2 * 1 / 4 * C = 72)

theorem maria_made_144_cookies: ∃ (C : ℕ), cookies C ∧ C = 144 :=
by
  existsi 144
  unfold cookies
  sorry

end maria_made_144_cookies_l1717_171707


namespace sequence_a_1998_value_l1717_171767

theorem sequence_a_1998_value :
  (∃ (a : ℕ → ℕ),
    (∀ n : ℕ, 0 <= a n) ∧
    (∀ n m : ℕ, n < m → a n < a m) ∧
    (∀ k : ℕ, ∃ i j t : ℕ, k = a i + 2 * a j + 4 * a t) ∧
    a 1998 = 1227096648) := sorry

end sequence_a_1998_value_l1717_171767


namespace satisfies_differential_eqn_l1717_171761

noncomputable def y (x : ℝ) : ℝ := 5 * Real.exp (-2 * x) + (1 / 3) * Real.exp x

theorem satisfies_differential_eqn : ∀ x : ℝ, (deriv y x) + 2 * y x = Real.exp x :=
by
  -- The proof is to be provided
  sorry

end satisfies_differential_eqn_l1717_171761


namespace cone_shorter_height_ratio_l1717_171793

theorem cone_shorter_height_ratio 
  (circumference : ℝ) (original_height : ℝ) (volume_shorter_cone : ℝ) 
  (shorter_height : ℝ) (radius : ℝ) :
  circumference = 24 * Real.pi ∧ 
  original_height = 40 ∧ 
  volume_shorter_cone = 432 * Real.pi ∧ 
  2 * Real.pi * radius = circumference ∧ 
  volume_shorter_cone = (1 / 3) * Real.pi * radius^2 * shorter_height
  → shorter_height / original_height = 9 / 40 :=
by
  sorry

end cone_shorter_height_ratio_l1717_171793


namespace range_of_a_l1717_171714

theorem range_of_a (a : ℝ) : (∃ x : ℝ, (|x - 1| - |x - 3|) > a) → a < 2 :=
by
  sorry

end range_of_a_l1717_171714


namespace simplify_expr_l1717_171790

noncomputable def expr := 1 / (1 + 1 / (Real.sqrt 5 + 2))
noncomputable def simplified := (Real.sqrt 5 + 1) / 4

theorem simplify_expr : expr = simplified :=
by
  sorry

end simplify_expr_l1717_171790


namespace ellipse_iff_k_range_l1717_171757

theorem ellipse_iff_k_range (k : ℝ) :
  (∃ x y, (x ^ 2 / (1 - k)) + (y ^ 2 / (1 + k)) = 1) ↔ (-1 < k ∧ k < 1 ∧ k ≠ 0) :=
by
  sorry

end ellipse_iff_k_range_l1717_171757


namespace find_sum_zero_l1717_171741

open Complex

noncomputable def complex_numbers_satisfy (a1 a2 a3 : ℂ) : Prop :=
  a1^2 + a2^2 + a3^2 = 0 ∧
  a1^3 + a2^3 + a3^3 = 0 ∧
  a1^4 + a2^4 + a3^4 = 0

theorem find_sum_zero (a1 a2 a3 : ℂ) (h : complex_numbers_satisfy a1 a2 a3) :
  a1 + a2 + a3 = 0 :=
by {
  sorry
}

end find_sum_zero_l1717_171741


namespace largest_digit_divisible_by_6_l1717_171725

theorem largest_digit_divisible_by_6 : ∃ N : ℕ, N = 8 ∧ (45670 + N) % 6 = 0 :=
sorry

end largest_digit_divisible_by_6_l1717_171725


namespace percentage_of_second_solution_correct_l1717_171727

noncomputable def percentage_of_alcohol_in_second_solution : ℝ :=
  let total_liters := 80
  let percentage_final_solution := 0.49
  let volume_first_solution := 24
  let percentage_first_solution := 0.4
  let volume_second_solution := 56
  let total_alcohol_in_final_solution := total_liters * percentage_final_solution
  let total_alcohol_first_solution := volume_first_solution * percentage_first_solution
  let x := (total_alcohol_in_final_solution - total_alcohol_first_solution) / volume_second_solution
  x

theorem percentage_of_second_solution_correct : 
  percentage_of_alcohol_in_second_solution = 0.5285714286 := by sorry

end percentage_of_second_solution_correct_l1717_171727


namespace range_of_m_l1717_171749

noncomputable def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = -f x

noncomputable def is_monotonically_decreasing_in_domain (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y : ℝ, a ≤ x ∧ x < y ∧ y ≤ b → f x ≥ f y

theorem range_of_m (f : ℝ → ℝ)
  (h_odd : is_odd_function f)
  (h_mono : is_monotonically_decreasing_in_domain f (-2) 2) :
  ∀ m : ℝ, (f (1 - m) + f (1 - m^2) < 0) → -2 < m ∧ m < 1 :=
sorry

end range_of_m_l1717_171749


namespace speed_of_other_train_l1717_171789

theorem speed_of_other_train :
  ∀ (d : ℕ) (v1 v2 : ℕ), d = 120 → v1 = 30 → 
    ∀ (d_remaining : ℕ), d_remaining = 70 → 
    v1 + v2 = d_remaining → 
    v2 = 40 :=
by
  intros d v1 v2 h_d h_v1 d_remaining h_d_remaining h_rel_speed
  sorry

end speed_of_other_train_l1717_171789


namespace tens_digit_of_even_not_divisible_by_10_l1717_171751

theorem tens_digit_of_even_not_divisible_by_10 (N : ℕ) (h1 : N % 2 = 0) (h2 : N % 10 ≠ 0) :
  (N ^ 20) % 100 / 10 % 10 = 7 :=
sorry

end tens_digit_of_even_not_divisible_by_10_l1717_171751


namespace tangent_line_equation_l1717_171776

theorem tangent_line_equation (P : ℝ × ℝ) (hP : P.2 = P.1^2)
  (h_perpendicular : ∃ k : ℝ, k * -1/2 = -1) : 
  ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧ a = 2 ∧ b = -1 ∧ c = -1 :=
by
  sorry

end tangent_line_equation_l1717_171776


namespace square_side_length_l1717_171792

-- Problem conditions as Lean definitions
def length_rect : ℕ := 400
def width_rect : ℕ := 300
def perimeter_rect := 2 * length_rect + 2 * width_rect
def perimeter_square := 2 * perimeter_rect
def length_square := perimeter_square / 4

-- Proof statement
theorem square_side_length : length_square = 700 := 
by 
  -- (Any necessary tactics to complete the proof would go here)
  sorry

end square_side_length_l1717_171792


namespace part1_part2_l1717_171732

noncomputable def f (x a : ℝ) : ℝ := |x - a^2| + |x - (2 * a + 1)|

theorem part1 (x : ℝ) : f x 2 ≥ 4 ↔ x ≤ 3 / 2 ∨ x ≥ 11 / 2 :=
  sorry

theorem part2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ a ≤ -1 ∨ a ≥ 3 :=
  sorry

end part1_part2_l1717_171732


namespace james_weekly_hours_l1717_171760

def james_meditation_total : ℕ :=
  let weekly_minutes := (30 * 2 * 6) + (30 * 2 * 2) -- 1 hour/day for 6 days + 2 hours on Sunday
  weekly_minutes / 60

def james_yoga_total : ℕ :=
  let weekly_minutes := (45 * 2) -- 45 minutes on Monday and Friday
  weekly_minutes / 60

def james_bikeride_total : ℕ :=
  let weekly_minutes := 90
  weekly_minutes / 60

def james_dance_total : ℕ :=
  2 -- 2 hours on Saturday

def james_total_activity_hours : ℕ :=
  james_meditation_total + james_yoga_total + james_bikeride_total + james_dance_total

theorem james_weekly_hours : james_total_activity_hours = 13 := by
  sorry

end james_weekly_hours_l1717_171760


namespace miquels_theorem_l1717_171769

-- Define a triangle ABC with points D, E, F on sides BC, CA, and AB respectively
variables {A B C D E F : Type}

-- Assume we have a function that checks for collinearity of points
def is_on_side (X Y Z: Type) : Bool := sorry

-- Assume a function that returns the circumcircle of a triangle formed by given points
def circumcircle (X Y Z: Type) : Type := sorry 

-- Define the function that checks the intersection of circumcircles
def have_common_point (circ1 circ2 circ3: Type) : Bool := sorry

-- The theorem statement
theorem miquels_theorem (A B C D E F : Type) 
  (hD: is_on_side D B C) 
  (hE: is_on_side E C A) 
  (hF: is_on_side F A B) : 
  have_common_point (circumcircle A E F) (circumcircle B D F) (circumcircle C D E) :=
sorry

end miquels_theorem_l1717_171769


namespace exists_pretty_hexagon_max_area_pretty_hexagon_l1717_171715

-- Define the condition of a "pretty" hexagon
structure PrettyHexagon (L ℓ h : ℝ) : Prop :=
  (diag1 : (L + ℓ)^2 + h^2 = 1)
  (diag2 : (L + ℓ)^2 + h^2 = 1)
  (diag3 : (L + ℓ)^2 + h^2 = 1)
  (diag4 : (L + ℓ)^2 + h^2 = 1)
  (L_pos : L > 0) (L_lt_1 : L < 1)
  (ℓ_pos : ℓ > 0) (ℓ_lt_1 : ℓ < 1)
  (h_pos : h > 0) (h_lt_1 : h < 1)

-- Area of the hexagon given L, ℓ, and h
def hexagon_area (L ℓ h : ℝ) := 2 * (L + ℓ) * h

-- Question (a): Existence of a pretty hexagon with a given area
theorem exists_pretty_hexagon (k : ℝ) (hk : 0 < k ∧ k < 1) : 
  ∃ L ℓ h : ℝ, PrettyHexagon L ℓ h ∧ hexagon_area L ℓ h = k :=
sorry

-- Question (b): Maximum area of any pretty hexagon is at most 1
theorem max_area_pretty_hexagon : 
  ∀ L ℓ h : ℝ, PrettyHexagon L ℓ h → hexagon_area L ℓ h ≤ 1 :=
sorry

end exists_pretty_hexagon_max_area_pretty_hexagon_l1717_171715


namespace water_flow_total_l1717_171780

theorem water_flow_total
  (R1 R2 R3 : ℕ)
  (h1 : R2 = 36)
  (h2 : R2 = (3 / 2) * R1)
  (h3 : R3 = (5 / 4) * R2)
  : R1 + R2 + R3 = 105 :=
sorry

end water_flow_total_l1717_171780


namespace infinitely_many_solutions_eq_l1717_171788

theorem infinitely_many_solutions_eq {a b : ℝ} 
  (H : ∀ x : ℝ, a * (a - x) - b * (b - x) = 0) : a = b :=
sorry

end infinitely_many_solutions_eq_l1717_171788


namespace red_balls_removal_condition_l1717_171779

theorem red_balls_removal_condition (total_balls : ℕ) (initial_red_balls : ℕ) (r : ℕ) : 
  total_balls = 600 → 
  initial_red_balls = 420 → 
  60 * (total_balls - r) = 100 * (initial_red_balls - r) → 
  r = 150 :=
by
  sorry

end red_balls_removal_condition_l1717_171779


namespace find_a3_l1717_171702

theorem find_a3 (a0 a1 a2 a3 a4 : ℝ) :
  (∀ x : ℝ, x^4 = a0 + a1 * (x + 2) + a2 * (x + 2)^2 + a3 * (x + 2)^3 + a4 * (x + 2)^4) →
  a3 = -8 :=
by
  sorry

end find_a3_l1717_171702


namespace proof_P_otimes_Q_l1717_171706

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

end proof_P_otimes_Q_l1717_171706


namespace greatest_divisor_same_remainder_l1717_171752

theorem greatest_divisor_same_remainder (a b c : ℕ) (d1 d2 d3 : ℕ) (h1 : a = 41) (h2 : b = 71) (h3 : c = 113)
(hd1 : d1 = b - a) (hd2 : d2 = c - b) (hd3 : d3 = c - a) :
  Nat.gcd (Nat.gcd d1 d2) d3 = 6 :=
by
  -- some computation here which we are skipping
  sorry

end greatest_divisor_same_remainder_l1717_171752


namespace solve_fractional_eq_l1717_171742

theorem solve_fractional_eq (x : ℝ) (h₀ : x ≠ 2) (h₁ : x ≠ -2) :
  (3 / (x - 2) + 5 / (x + 2) = 8 / (x^2 - 4)) → (x = 3 / 2) :=
by sorry

end solve_fractional_eq_l1717_171742


namespace faye_earned_total_l1717_171712

-- Definitions of the necklace sales
def bead_necklaces := 3
def bead_price := 7
def gemstone_necklaces := 7
def gemstone_price := 10
def pearl_necklaces := 2
def pearl_price := 12
def crystal_necklaces := 5
def crystal_price := 15

-- Total amount calculation
def total_amount := 
  bead_necklaces * bead_price + 
  gemstone_necklaces * gemstone_price + 
  pearl_necklaces * pearl_price + 
  crystal_necklaces * crystal_price

-- Proving the total amount equals $190
theorem faye_earned_total : total_amount = 190 := by
  sorry

end faye_earned_total_l1717_171712


namespace find_k_values_l1717_171734

open Set

def A : Set ℝ := {x | x^2 + 2 * x - 3 = 0}
def B (k : ℝ) : Set ℝ := {x | x^2 - (k + 1) * x + k = 0}

theorem find_k_values (k : ℝ) : (A ∩ B k = B k) ↔ k ∈ ({1, -3} : Set ℝ) := by
  sorry

end find_k_values_l1717_171734


namespace guinea_pig_food_ratio_l1717_171709

-- Definitions of amounts of food consumed by each guinea pig
def first_guinea_pig_food : ℕ := 2
variable (x : ℕ)
def second_guinea_pig_food : ℕ := x
def third_guinea_pig_food : ℕ := x + 3

-- Total food requirement condition
def total_food_required := first_guinea_pig_food + second_guinea_pig_food x + third_guinea_pig_food x = 13

-- Define the ratio function
def ratio (a b : ℕ) : ℚ := a / b

-- The goal is to prove this ratio given the conditions
theorem guinea_pig_food_ratio (h : total_food_required x) : ratio (second_guinea_pig_food x) first_guinea_pig_food = 2 := by
  sorry

end guinea_pig_food_ratio_l1717_171709


namespace find_x_pow_3a_minus_b_l1717_171726

variable (x : ℝ) (a b : ℝ)
theorem find_x_pow_3a_minus_b (h1 : x^a = 2) (h2 : x^b = 9) : x^(3 * a - b) = 8 / 9 :=
  sorry

end find_x_pow_3a_minus_b_l1717_171726


namespace find_s_l1717_171721

theorem find_s 
  (a b c x s z : ℕ)
  (h1 : a + b = x)
  (h2 : x + c = s)
  (h3 : s + a = z)
  (h4 : b + c + z = 16) : 
  s = 8 := 
sorry

end find_s_l1717_171721


namespace sally_turnip_count_l1717_171774

theorem sally_turnip_count (total_turnips : ℕ) (mary_turnips : ℕ) (sally_turnips : ℕ) 
  (h1: total_turnips = 242) 
  (h2: mary_turnips = 129) 
  (h3: total_turnips = mary_turnips + sally_turnips) : 
  sally_turnips = 113 := 
by 
  sorry

end sally_turnip_count_l1717_171774


namespace smallest_number_l1717_171796

def binary_101010 : ℕ := 1 * 2^5 + 0 * 2^4 + 1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 0 * 2^0
def base5_111 : ℕ := 1 * 5^2 + 1 * 5^1 + 1 * 5^0
def octal_32 : ℕ := 3 * 8^1 + 2 * 8^0
def base6_54 : ℕ := 5 * 6^1 + 4 * 6^0

theorem smallest_number : octal_32 < binary_101010 ∧ octal_32 < base5_111 ∧ octal_32 < base6_54 :=
by
  sorry

end smallest_number_l1717_171796


namespace de_morgan_union_de_morgan_inter_l1717_171744

open Set

variable {α : Type*} (A B : Set α)

theorem de_morgan_union : ∀ (A B : Set α), 
  compl (A ∪ B) = compl A ∩ compl B := 
by 
  intro A B
  sorry

theorem de_morgan_inter : ∀ (A B : Set α), 
  compl (A ∩ B) = compl A ∪ compl B := 
by 
  intro A B
  sorry

end de_morgan_union_de_morgan_inter_l1717_171744


namespace gcd_7429_12345_l1717_171740

theorem gcd_7429_12345 : Int.gcd 7429 12345 = 1 := 
by 
  sorry

end gcd_7429_12345_l1717_171740


namespace num_positive_solutions_eq_32_l1717_171772

theorem num_positive_solutions_eq_32 : 
  ∃ n : ℕ, (∀ x y : ℕ, 4 * x + 7 * y = 888 → x > 0 ∧ y > 0) ∧ n = 32 :=
sorry

end num_positive_solutions_eq_32_l1717_171772


namespace plane_through_point_and_line_l1717_171723

noncomputable def point_on_plane (A B C D : ℤ) (x y z : ℤ) : Prop :=
  A * x + B * y + C * z + D = 0

def line_eq_1 (x y : ℤ) : Prop :=
  3 * x + 4 * y - 20 = 0

def line_eq_2 (y z : ℤ) : Prop :=
  -3 * y + 2 * z + 18 = 0

theorem plane_through_point_and_line 
  (A B C D : ℤ)
  (h_point : point_on_plane A B C D 1 9 (-8))
  (h_line1 : ∀ x y, line_eq_1 x y → point_on_plane A B C D x y 0)
  (h_line2 : ∀ y z, line_eq_2 y z → point_on_plane A B C D 0 y z)
  (h_gcd : Int.gcd (Int.gcd (Int.gcd (A.natAbs) (B.natAbs)) (C.natAbs)) (D.natAbs) = 1) 
  (h_pos : A > 0) :
  A = 75 ∧ B = -29 ∧ C = 86 ∧ D = 274 :=
sorry

end plane_through_point_and_line_l1717_171723


namespace number_of_women_l1717_171759

variable (W : ℕ) (x : ℝ)

-- Conditions
def daily_wage_men_and_women (W : ℕ) (x : ℝ) : Prop :=
  24 * 350 + W * x = 11600

def half_men_and_37_women (W : ℕ) (x : ℝ) : Prop :=
  12 * 350 + 37 * x = 24 * 350 + W * x

def daily_wage_man := (350 : ℝ)

-- Proposition to prove
theorem number_of_women (W : ℕ) (x : ℝ) (h1 : daily_wage_men_and_women W x)
  (h2 : half_men_and_37_women W x) : W = 16 := 
  by
  sorry

end number_of_women_l1717_171759


namespace a_5_eq_16_S_8_eq_255_l1717_171787

open Nat

-- Definitions from the conditions
def a : ℕ → ℕ
| 0     => 1
| (n+1) => 2 * a n

def S (n : ℕ) : ℕ :=
  (Finset.range n).sum a

-- Proof problem statements
theorem a_5_eq_16 : a 4 = 16 := sorry

theorem S_8_eq_255 : S 8 = 255 := sorry

end a_5_eq_16_S_8_eq_255_l1717_171787


namespace oyster_crab_ratio_l1717_171724

theorem oyster_crab_ratio
  (O1 C1 : ℕ)
  (h1 : O1 = 50)
  (h2 : C1 = 72)
  (h3 : ∃ C2 : ℕ, C2 = (2 * C1) / 3)
  (h4 : ∃ O2 : ℕ, O1 + C1 + O2 + C2 = 195) :
  ∃ ratio : ℚ, ratio = O2 / O1 ∧ ratio = (1 : ℚ) / 2 := 
by 
  sorry

end oyster_crab_ratio_l1717_171724


namespace exponentiation_equation_l1717_171766

theorem exponentiation_equation : 4^2011 * (-0.25)^2010 - 1 = 3 := 
by { sorry }

end exponentiation_equation_l1717_171766


namespace joe_left_pocket_initial_l1717_171768

-- Definitions from conditions
def total_money : ℕ := 200
def initial_left_pocket (L : ℕ) : ℕ := L
def initial_right_pocket (R : ℕ) : ℕ := R
def transfer_one_fourth (L : ℕ) : ℕ := L - L / 4
def add_to_right (R : ℕ) (L : ℕ) : ℕ := R + L / 4
def transfer_20 (L : ℕ) : ℕ := transfer_one_fourth L - 20
def add_20_to_right (R : ℕ) (L : ℕ) : ℕ := add_to_right R L + 20

-- Statement to prove
theorem joe_left_pocket_initial (L R : ℕ) (h₁ : L + R = total_money) 
  (h₂ : transfer_20 L = add_20_to_right R L) : 
  initial_left_pocket L = 160 :=
by
  sorry

end joe_left_pocket_initial_l1717_171768


namespace find_k_l1717_171704

theorem find_k 
  (x y k : ℚ) 
  (h1 : y = 4 * x - 1) 
  (h2 : y = -1 / 3 * x + 11) 
  (h3 : y = 2 * x + k) : 
  k = 59 / 13 :=
sorry

end find_k_l1717_171704


namespace max_f_l1717_171731

open Real

noncomputable def f (x y z : ℝ) := (1 - y * z + z) * (1 - z * x + x) * (1 - x * y + y)

theorem max_f (x y z : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : 0 < z) (h₄ : x * y * z = 1) :
  f x y z ≤ 1 ∧ (x = 1 ∧ y = 1 ∧ z = 1 → f x y z = 1) := sorry

end max_f_l1717_171731


namespace sum_reciprocal_eq_eleven_eighteen_l1717_171739

noncomputable def sum_reciprocal (n : ℕ) : ℝ := ∑' (n : ℕ), 1 / (n * (n + 3))

theorem sum_reciprocal_eq_eleven_eighteen :
  sum_reciprocal = 11 / 18 :=
by
  sorry

end sum_reciprocal_eq_eleven_eighteen_l1717_171739


namespace mr_green_expects_expected_potatoes_yield_l1717_171783

theorem mr_green_expects_expected_potatoes_yield :
  ∀ (length_steps width_steps: ℕ) (step_length yield_per_sqft: ℝ),
  length_steps = 18 →
  width_steps = 25 →
  step_length = 2.5 →
  yield_per_sqft = 0.75 →
  (length_steps * step_length) * (width_steps * step_length) * yield_per_sqft = 2109.375 :=
by
  intros length_steps width_steps step_length yield_per_sqft
  intros h_length_steps h_width_steps h_step_length h_yield_per_sqft
  rw [h_length_steps, h_width_steps, h_step_length, h_yield_per_sqft]
  sorry

end mr_green_expects_expected_potatoes_yield_l1717_171783


namespace three_digit_number_108_l1717_171773

theorem three_digit_number_108 (a b c : ℕ) (ha : a ≠ 0) (h₀ : a < 10) (h₁ : b < 10) (h₂ : c < 10) (h₃: 100*a + 10*b + c = 12*(a + b + c)) : 
  100*a + 10*b + c = 108 := 
by 
  sorry

end three_digit_number_108_l1717_171773


namespace number_of_roses_two_days_ago_l1717_171786

-- Define the conditions
variables (R : ℕ) 
-- Condition 1: Variable R is the number of roses planted two days ago.
-- Condition 2: The number of roses planted yesterday is R + 20.
-- Condition 3: The number of roses planted today is 2R.
-- Condition 4: The total number of roses planted over three days is 220.
axiom condition_1 : 0 ≤ R
axiom condition_2 : (R + (R + 20) + (2 * R)) = 220

-- Proof goal: Prove that R = 50 
theorem number_of_roses_two_days_ago : R = 50 :=
by sorry

end number_of_roses_two_days_ago_l1717_171786


namespace cube_difference_l1717_171729

theorem cube_difference (n : ℕ) (h: 0 < n) : (n + 1)^3 - n^3 = 3 * n^2 + 3 * n + 1 := 
sorry

end cube_difference_l1717_171729


namespace problem_solution_l1717_171743

variables {p q r : ℝ}

theorem problem_solution (h1 : (p + q) * (q + r) * (r + p) / (p * q * r) = 24)
  (h2 : (p - 2 * q) * (q - 2 * r) * (r - 2 * p) / (p * q * r) = 10) :
  ∃ m n : ℕ, (m.gcd n = 1 ∧ (p/q + q/r + r/p = m/n) ∧ m + n = 39) :=
sorry

end problem_solution_l1717_171743


namespace number_of_sodas_bought_l1717_171785

-- Definitions based on conditions
def cost_sandwich : ℝ := 1.49
def cost_two_sandwiches : ℝ := 2 * cost_sandwich
def cost_soda : ℝ := 0.87
def total_cost : ℝ := 6.46

-- We need to prove that the number of sodas bought is 4 given these conditions
theorem number_of_sodas_bought : (total_cost - cost_two_sandwiches) / cost_soda = 4 := by
  sorry

end number_of_sodas_bought_l1717_171785


namespace new_room_correct_size_l1717_171738

-- Definitions of conditions
def current_bedroom := 309 -- sq ft
def current_bathroom := 150 -- sq ft
def current_space := current_bedroom + current_bathroom
def new_room_size := 2 * current_space

-- Proving the new room size
theorem new_room_correct_size : new_room_size = 918 := by
  sorry

end new_room_correct_size_l1717_171738


namespace sum_of_fourth_powers_l1717_171717

theorem sum_of_fourth_powers (x y : ℝ) (h1 : x + y = 2) (h2 : x * y = 1) : x^4 + y^4 = 2 :=
by sorry

end sum_of_fourth_powers_l1717_171717


namespace arrange_letters_l1717_171795

-- Definitions based on conditions
def total_letters := 6
def identical_bs := 2 -- Number of B's that are identical
def distinct_as := 3  -- Number of A's that are distinct
def distinct_ns := 1  -- Number of N's that are distinct

-- Now formulate the proof statement
theorem arrange_letters :
    (Nat.factorial total_letters) / (Nat.factorial identical_bs) = 360 :=
by
  sorry

end arrange_letters_l1717_171795


namespace age_difference_l1717_171716

variable (P M Mo N : ℚ)

-- Given conditions as per problem statement
axiom ratio_P_M : (P / M) = 3 / 5
axiom ratio_M_Mo : (M / Mo) = 3 / 4
axiom ratio_Mo_N : (Mo / N) = 5 / 7
axiom sum_ages : P + M + Mo + N = 228

-- Statement to prove
theorem age_difference (ratio_P_M : (P / M) = 3 / 5)
                        (ratio_M_Mo : (M / Mo) = 3 / 4)
                        (ratio_Mo_N : (Mo / N) = 5 / 7)
                        (sum_ages : P + M + Mo + N = 228) :
  N - P = 69.5 := 
sorry

end age_difference_l1717_171716


namespace train_car_passengers_l1717_171794

theorem train_car_passengers (x : ℕ) (h : 60 * x = 732 + 228) : x = 16 :=
by
  sorry

end train_car_passengers_l1717_171794


namespace power_of_power_example_l1717_171755

theorem power_of_power_example : (3^2)^4 = 6561 := by
  sorry

end power_of_power_example_l1717_171755


namespace arrangements_ABC_together_l1717_171748

noncomputable def permutation_count_ABC_together (n : Nat) (unit_size : Nat) (remaining : Nat) : Nat :=
  (Nat.factorial unit_size) * (Nat.factorial (remaining + 1))

theorem arrangements_ABC_together : permutation_count_ABC_together 6 3 3 = 144 :=
by
  sorry

end arrangements_ABC_together_l1717_171748


namespace sin_double_angle_l1717_171756

open Real 

theorem sin_double_angle (α β : ℝ) 
  (h1 : π / 2 < β) 
  (h2 : β < α) 
  (h3 : α < 3 * π / 4) 
  (h4 : cos (α - β) = 12 / 13) 
  (h5 : sin (α + β) = -3 / 5) : 
  sin (2 * α) = -56 / 65 := 
by 
  sorry

end sin_double_angle_l1717_171756


namespace admission_counts_l1717_171771

-- Define the total number of ways to admit students under given conditions.
def ways_of_admission : Nat := 1518

-- Statement of the problem: given conditions, prove the result
theorem admission_counts (n_colleges : Nat) (n_students : Nat) (admitted_two_colleges : Bool) : 
  n_colleges = 23 → 
  n_students = 3 → 
  admitted_two_colleges = true →
  ways_of_admission = 1518 :=
by
  intros
  sorry

end admission_counts_l1717_171771


namespace smallest_m_n_l1717_171730

noncomputable def g (m n : ℕ) (x : ℝ) : ℝ := Real.arccos (Real.log (↑n * x) / Real.log (↑m))

theorem smallest_m_n (m n : ℕ) (h1 : 1 < m) (h2 : ∀ x : ℝ, -1 ≤ Real.log (↑n * x) / Real.log (↑m) ∧
                      Real.log (↑n * x) / Real.log (↑m) ≤ 1 ∧
                      (forall a b : ℝ,  a ≤ x ∧ x ≤ b -> b - a = 1 / 1007)) :
  m + n = 1026 :=
sorry

end smallest_m_n_l1717_171730


namespace fractions_equiv_x_zero_l1717_171722

theorem fractions_equiv_x_zero (x b : ℝ) (h : x + 3 * b ≠ 0) : 
  (x + 2 * b) / (x + 3 * b) = 2 / 3 ↔ x = 0 :=
by sorry

end fractions_equiv_x_zero_l1717_171722


namespace perp_vec_m_l1717_171763

theorem perp_vec_m (m : ℝ) : (1 : ℝ) * (-1 : ℝ) + 2 * m = 0 → m = 1 / 2 :=
by 
  intro h
  -- Translate the given condition directly
  sorry

end perp_vec_m_l1717_171763


namespace triangle_perimeter_ABF_l1717_171745

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := (x^2 / 25) + (y^2 / 21) = 1

-- Define the line
def line (x : ℝ) : Prop := x = -2

-- Define the foci of the ellipse
def right_focus : ℝ := 2
def left_focus : ℝ := -2

-- Points A and B are on the ellipse and line
def point_A (x y : ℝ) : Prop := ellipse x y ∧ line x
def point_B (x y : ℝ) : Prop := ellipse x y ∧ line x

-- Point F is the right focus of the ellipse
def point_F (x y : ℝ) : Prop := x = right_focus ∧ y = 0

-- Perimeter of the triangle ABF
def perimeter (A B F : ℝ × ℝ) : ℝ :=
  sorry -- Calculation of the perimeter of triangle ABF

-- Theorem statement that perimeter is 20
theorem triangle_perimeter_ABF 
  (A B F : ℝ × ℝ) 
  (hA : point_A (A.fst) (A.snd)) 
  (hB : point_B (B.fst) (B.snd))
  (hF : point_F (F.fst) (F.snd)) :
  perimeter A B F = 20 :=
sorry

end triangle_perimeter_ABF_l1717_171745


namespace relationship_a_b_c_d_l1717_171713

theorem relationship_a_b_c_d 
  (a b c d : ℤ)
  (h : (a + b + 1) * (d + a + 2) = (c + d + 1) * (b + c + 2)) : 
  a + b + c + d = -2 := 
sorry

end relationship_a_b_c_d_l1717_171713


namespace solve_for_y_l1717_171791

theorem solve_for_y (y : ℝ) (h : y^2 + 6 * y + 8 = -(y + 4) * (y + 6)) : y = -4 :=
by {
  sorry
}

end solve_for_y_l1717_171791


namespace complement_M_l1717_171782

def U : Set ℝ := Set.univ
def M : Set ℝ := {x | x^2 - 4 ≤ 0}
def C (s : Set ℝ) : Set ℝ := sᶜ -- complement of a set

theorem complement_M :
  C M = {x : ℝ | x < -2 ∨ x > 2} :=
by
  sorry

end complement_M_l1717_171782


namespace angle_opposite_c_exceeds_l1717_171737

theorem angle_opposite_c_exceeds (a b : ℝ) (c : ℝ) (C : ℝ) (h_a : a = 2) (h_b : b = 2) (h_c : c >= 4) : 
  C >= 120 := 
sorry

end angle_opposite_c_exceeds_l1717_171737


namespace compute_expression_l1717_171799

theorem compute_expression (a b c : ℝ) (h : a^3 - 6 * a^2 + 11 * a - 6 = 0 ∧ b^3 - 6 * b^2 + 11 * b - 6 = 0 ∧ c^3 - 6 * c^2 + 11 * c - 6 = 0) :
  (ab / c + bc / a + ca / b) = 49 / 6 := 
  by
  sorry -- Placeholder for the proof

end compute_expression_l1717_171799


namespace discount_problem_l1717_171735

theorem discount_problem (x : ℝ) (h : 560 * (1 - x / 100) * 0.70 = 313.6) : x = 20 := 
by
  sorry

end discount_problem_l1717_171735
