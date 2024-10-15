import Mathlib

namespace NUMINAMATH_GPT_range_of_a_l970_97030

def f (a x : ℝ) : ℝ := a * x^2 - 2 * x - |x^2 - a * x + 1|

def has_exactly_two_zeros (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f a x₁ = 0 ∧ f a x₂ = 0 ∧ ∀ x, x ≠ x₁ ∧ x ≠ x₂ → f a x ≠ 0

theorem range_of_a :
  { a : ℝ | has_exactly_two_zeros a } =
  { a : ℝ | (a < 0) ∨ (0 < a ∧ a < 1) ∨ (1 < a) } :=
sorry

end NUMINAMATH_GPT_range_of_a_l970_97030


namespace NUMINAMATH_GPT_theater_ticket_problem_l970_97033

noncomputable def total_cost_proof (x : ℝ) : Prop :=
  let cost_adult_tickets := 10 * x
  let cost_child_tickets := 8 * (x / 2)
  let cost_senior_tickets := 4 * (0.75 * x)
  cost_adult_tickets + cost_child_tickets + cost_senior_tickets = 58.65

theorem theater_ticket_problem (x : ℝ) (h : 6 * x + 5 * (x / 2) + 3 * (0.75 * x) = 42) : 
  total_cost_proof x :=
by
  sorry

end NUMINAMATH_GPT_theater_ticket_problem_l970_97033


namespace NUMINAMATH_GPT_inequality_holds_for_all_real_l970_97044

theorem inequality_holds_for_all_real (x : ℝ) : x^2 + 1 ≥ 2 * |x| := sorry

end NUMINAMATH_GPT_inequality_holds_for_all_real_l970_97044


namespace NUMINAMATH_GPT_breadth_of_room_is_6_l970_97088

theorem breadth_of_room_is_6 
(the_room_length : ℝ) 
(the_carpet_width : ℝ) 
(cost_per_meter : ℝ) 
(total_cost : ℝ) 
(h1 : the_room_length = 15) 
(h2 : the_carpet_width = 0.75) 
(h3 : cost_per_meter = 0.30) 
(h4 : total_cost = 36) : 
  ∃ (breadth_of_room : ℝ), breadth_of_room = 6 :=
sorry

end NUMINAMATH_GPT_breadth_of_room_is_6_l970_97088


namespace NUMINAMATH_GPT_find_triplets_l970_97068

noncomputable def phi (t : ℝ) : ℝ := 2 * t^3 + t - 2

theorem find_triplets (x y z : ℝ) (h1 : x^5 = phi y) (h2 : y^5 = phi z) (h3 : z^5 = phi x) :
  ∃ r : ℝ, (x = r ∧ y = r ∧ z = r) ∧ (r^5 = phi r) :=
by
  sorry

end NUMINAMATH_GPT_find_triplets_l970_97068


namespace NUMINAMATH_GPT_find_other_discount_l970_97041

theorem find_other_discount (P F d1 : ℝ) (H₁ : P = 70) (H₂ : F = 61.11) (H₃ : d1 = 10) : ∃ (d2 : ℝ), d2 = 3 :=
by 
  -- The proof will be provided here.
  sorry

end NUMINAMATH_GPT_find_other_discount_l970_97041


namespace NUMINAMATH_GPT_real_part_z_pow_2017_l970_97058

open Complex

noncomputable def z : ℂ := 1 + I

theorem real_part_z_pow_2017 : re (z ^ 2017) = 2 ^ 1008 := sorry

end NUMINAMATH_GPT_real_part_z_pow_2017_l970_97058


namespace NUMINAMATH_GPT_bug_at_vertex_A_after_8_meters_l970_97005

theorem bug_at_vertex_A_after_8_meters (P : ℕ → ℚ) (h₀ : P 0 = 1)
(h : ∀ n, P (n + 1) = 1/3 * (1 - P n)) : 
P 8 = 1823 / 6561 := 
sorry

end NUMINAMATH_GPT_bug_at_vertex_A_after_8_meters_l970_97005


namespace NUMINAMATH_GPT_equation_of_circle_given_diameter_l970_97089

def is_on_circle (center : ℝ × ℝ) (radius : ℝ) (p : ℝ × ℝ) : Prop :=
  (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2

theorem equation_of_circle_given_diameter :
  ∀ (A B : ℝ × ℝ), A = (-3,0) → B = (1,0) → 
  (∃ (x y : ℝ), is_on_circle (-1, 0) 2 (x, y)) ↔ (x + 1)^2 + y^2 = 4 :=
by
  sorry

end NUMINAMATH_GPT_equation_of_circle_given_diameter_l970_97089


namespace NUMINAMATH_GPT_boys_girls_rel_l970_97098

theorem boys_girls_rel (b g : ℕ) (h : g = 7 + 2 * (b - 1)) : b = (g - 5) / 2 := 
by sorry

end NUMINAMATH_GPT_boys_girls_rel_l970_97098


namespace NUMINAMATH_GPT_candy_problem_l970_97026

theorem candy_problem 
  (weightA costA : ℕ) (weightB costB : ℕ) (avgPrice per100 : ℕ)
  (hA : weightA = 300) (hCostA : costA = 5)
  (hCostB : costB = 7) (hAvgPrice : avgPrice = 150) (hPer100 : per100 = 100)
  (totalCost : ℕ) (hTotalCost : totalCost = costA + costB)
  (totalWeight : ℕ) (hTotalWeight : totalWeight = (totalCost * per100) / avgPrice) :
  (totalWeight = weightA + weightB) -> 
  weightB = 500 :=
by {
  sorry
}

end NUMINAMATH_GPT_candy_problem_l970_97026


namespace NUMINAMATH_GPT_average_speed_l970_97078

theorem average_speed 
  (total_distance : ℝ) (total_time : ℝ) 
  (h_distance : total_distance = 26) (h_time : total_time = 4) :
  (total_distance / total_time) = 6.5 :=
by
  rw [h_distance, h_time]
  norm_num

end NUMINAMATH_GPT_average_speed_l970_97078


namespace NUMINAMATH_GPT_total_attendance_l970_97042

-- Defining the given conditions
def adult_ticket_cost : ℕ := 8
def child_ticket_cost : ℕ := 1
def total_amount_collected : ℕ := 50
def number_of_child_tickets : ℕ := 18

-- Formulating the proof problem
theorem total_attendance (A : ℕ) (C : ℕ) (H1 : C = number_of_child_tickets)
  (H2 : adult_ticket_cost * A + child_ticket_cost * C = total_amount_collected) :
  A + C = 22 := by
  sorry

end NUMINAMATH_GPT_total_attendance_l970_97042


namespace NUMINAMATH_GPT_part1_part2_l970_97076

-- Part 1
theorem part1 (a : ℝ) : 
  (∀ x > -1, (x^2 + 3*x + 6) / (x + 1) ≥ a) ↔ (a ≤ 5) := 
  sorry

-- Part 2
theorem part2 (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 2*b = 1) : 
  2*a + (1/a) + 4*b + (8/b) ≥ 27 :=
  sorry

end NUMINAMATH_GPT_part1_part2_l970_97076


namespace NUMINAMATH_GPT_arithmetic_sequence_a1_d_l970_97085

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

end NUMINAMATH_GPT_arithmetic_sequence_a1_d_l970_97085


namespace NUMINAMATH_GPT_more_red_balls_l970_97031

theorem more_red_balls (red_packs yellow_packs pack_size : ℕ) (h1 : red_packs = 5) (h2 : yellow_packs = 4) (h3 : pack_size = 18) :
  (red_packs * pack_size) - (yellow_packs * pack_size) = 18 :=
by
  sorry

end NUMINAMATH_GPT_more_red_balls_l970_97031


namespace NUMINAMATH_GPT_factor_polynomial_l970_97071

theorem factor_polynomial (a b m n : ℝ) (h : |m - 4| + (n^2 - 8 * n + 16) = 0) :
  a^2 + 4 * b^2 - m * a * b - n = (a - 2 * b + 2) * (a - 2 * b - 2) :=
by
  sorry

end NUMINAMATH_GPT_factor_polynomial_l970_97071


namespace NUMINAMATH_GPT_prove_A_plus_B_plus_1_l970_97040

theorem prove_A_plus_B_plus_1 (A B : ℤ) 
  (h1 : B = A + 2)
  (h2 : 2 * A^2 + A + 6 + 5 * B + 2 = 7 * (A + B + 1) + 5) :
  A + B + 1 = 15 :=
by 
  sorry

end NUMINAMATH_GPT_prove_A_plus_B_plus_1_l970_97040


namespace NUMINAMATH_GPT_sum_x_coordinates_Q4_is_3000_l970_97028

-- Let Q1 be a 150-gon with vertices having x-coordinates summing to 3000
def Q1_x_sum := 3000
def Q2_x_sum := Q1_x_sum
def Q3_x_sum := Q2_x_sum
def Q4_x_sum := Q3_x_sum

-- Theorem to prove the sum of the x-coordinates of the vertices of Q4 is 3000
theorem sum_x_coordinates_Q4_is_3000 : Q4_x_sum = 3000 := by
  sorry

end NUMINAMATH_GPT_sum_x_coordinates_Q4_is_3000_l970_97028


namespace NUMINAMATH_GPT_parkway_elementary_students_l970_97077

/-- The total number of students in the fifth grade at Parkway Elementary School is 420,
given the following conditions:
1. There are 312 boys.
2. 250 students are playing soccer.
3. 78% of the students that play soccer are boys.
4. There are 53 girl students not playing soccer. -/
theorem parkway_elementary_students (boys : ℕ) (playing_soccer : ℕ) (percent_boys_playing : ℝ) (girls_not_playing_soccer : ℕ)
  (h1 : boys = 312)
  (h2 : playing_soccer = 250)
  (h3 : percent_boys_playing = 0.78)
  (h4 : girls_not_playing_soccer = 53) :
  ∃ total_students : ℕ, total_students = 420 :=
by
  sorry

end NUMINAMATH_GPT_parkway_elementary_students_l970_97077


namespace NUMINAMATH_GPT_fran_avg_speed_l970_97002

theorem fran_avg_speed (Joann_speed : ℕ) (Joann_time : ℚ) (Fran_time : ℕ) (distance : ℕ) (s : ℚ) : 
  Joann_speed = 16 → 
  Joann_time = 3.5 → 
  Fran_time = 4 → 
  distance = Joann_speed * Joann_time → 
  distance = Fran_time * s → 
  s = 14 :=
by
  intros hJs hJt hFt hD hF
  sorry

end NUMINAMATH_GPT_fran_avg_speed_l970_97002


namespace NUMINAMATH_GPT_inequality_holds_l970_97086

theorem inequality_holds (a : ℝ) (h : a ≠ 0) : |a + (1/a)| ≥ 2 :=
by
  sorry

end NUMINAMATH_GPT_inequality_holds_l970_97086


namespace NUMINAMATH_GPT_software_price_l970_97054

theorem software_price (copies total_revenue : ℝ) (P : ℝ) 
  (h1 : copies = 1200)
  (h2 : 0.5 * copies * P + 0.6 * (2 / 3) * (copies - 0.5 * copies) * P + 0.25 * (copies - 0.5 * copies - (2 / 3) * (copies - 0.5 * copies)) * P = total_revenue)
  (h3 : total_revenue = 72000) :
  P = 80.90 :=
by
  sorry

end NUMINAMATH_GPT_software_price_l970_97054


namespace NUMINAMATH_GPT_height_of_triangle_is_5_l970_97065

def base : ℝ := 4
def area : ℝ := 10

theorem height_of_triangle_is_5 :
  ∃ (height : ℝ), (base * height) / 2 = area ∧ height = 5 :=
by
  sorry

end NUMINAMATH_GPT_height_of_triangle_is_5_l970_97065


namespace NUMINAMATH_GPT_pints_in_vat_l970_97061

-- Conditions
def num_glasses : Nat := 5
def pints_per_glass : Nat := 30

-- Problem statement: prove that the total number of pints in the vat is 150
theorem pints_in_vat : num_glasses * pints_per_glass = 150 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_pints_in_vat_l970_97061


namespace NUMINAMATH_GPT_equal_numbers_possible_l970_97080

noncomputable def circle_operations (n : ℕ) (α : ℝ) : Prop :=
  (n ≥ 3) ∧ (∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n))

-- Statement of the theorem
theorem equal_numbers_possible (n : ℕ) (α : ℝ) (h1 : n ≥ 3) (h2 : α > 0) :
  circle_operations n α ↔ ∃ k : ℤ, α = 2 * Real.cos (k * Real.pi / n) :=
sorry

end NUMINAMATH_GPT_equal_numbers_possible_l970_97080


namespace NUMINAMATH_GPT_angle_supplement_complement_l970_97067

theorem angle_supplement_complement (a : ℝ) (h : 180 - a = 3 * (90 - a)) : a = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_supplement_complement_l970_97067


namespace NUMINAMATH_GPT_fallen_sheets_l970_97029

/-- The number of sheets that fell out of a book given the first page is 163
    and the last page contains the same digits but arranged in a different 
    order and ends with an even digit.
-/
theorem fallen_sheets (h1 : ∃ n, n = 163 ∧ 
                        ∃ m, m ≠ n ∧ (m = 316) ∧ 
                        m % 2 = 0 ∧ 
                        (∃ p1 p2 p3 q1 q2 q3, 
                         (p1, p2, p3) ≠ (q1, q2, q3) ∧ 
                         p1 ≠ q1 ∧ p2 ≠ q2 ∧ p3 ≠ q3 ∧ 
                         n = p1 * 100 + p2 * 10 + p3 ∧ 
                         m = q1 * 100 + q2 * 10 + q3)) :
  ∃ k, k = 77 :=
by
  sorry

end NUMINAMATH_GPT_fallen_sheets_l970_97029


namespace NUMINAMATH_GPT_cubic_equation_root_sum_l970_97057

theorem cubic_equation_root_sum (p q r : ℝ) (h1 : p + q + r = 6) (h2 : p * q + p * r + q * r = 11) (h3 : p * q * r = 6) :
  (p * q / r + p * r / q + q * r / p) = 49 / 6 := sorry

end NUMINAMATH_GPT_cubic_equation_root_sum_l970_97057


namespace NUMINAMATH_GPT_bread_left_l970_97047

def initial_bread : ℕ := 1000
def bomi_ate : ℕ := 350
def yejun_ate : ℕ := 500

theorem bread_left : initial_bread - (bomi_ate + yejun_ate) = 150 :=
by
  sorry

end NUMINAMATH_GPT_bread_left_l970_97047


namespace NUMINAMATH_GPT_problem_l970_97007

noncomputable def number_of_regions_four_planes (h1 : True) (h2 : True) : ℕ := 14

theorem problem (h1 : True) (h2 : True) : number_of_regions_four_planes h1 h2 = 14 :=
by sorry

end NUMINAMATH_GPT_problem_l970_97007


namespace NUMINAMATH_GPT_length_AE_l970_97024

theorem length_AE (AB CD AC AE ratio : ℝ) 
  (h_AB : AB = 10) 
  (h_CD : CD = 15) 
  (h_AC : AC = 18) 
  (h_ratio : ratio = 2 / 3) 
  (h_areas : ∀ (areas : ℝ), areas = 2 / 3)
  : AE = 7.2 := 
sorry

end NUMINAMATH_GPT_length_AE_l970_97024


namespace NUMINAMATH_GPT_probability_of_divisibility_l970_97021

noncomputable def is_prime_digit (d : ℕ) : Prop := d = 2 ∨ d = 3 ∨ d = 5 ∨ d = 7

noncomputable def is_prime_digit_number (n : ℕ) : Prop :=
  let digits := n.digits 10
  ∀ d ∈ digits, is_prime_digit d

noncomputable def is_divisible_by_3_and_4 (n : ℕ) : Prop :=
  n % 3 = 0 ∧ n % 4 = 0

theorem probability_of_divisibility (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999 ∨ 10 ≤ n ∧ n ≤ 99) →
  is_prime_digit_number n →
  ¬ is_divisible_by_3_and_4 n :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_probability_of_divisibility_l970_97021


namespace NUMINAMATH_GPT_find_y_coordinate_of_P_l970_97037

noncomputable def A : ℝ × ℝ := (-4, 0)
noncomputable def B : ℝ × ℝ := (-3, 2)
noncomputable def C : ℝ × ℝ := (3, 2)
noncomputable def D : ℝ × ℝ := (4, 0)
noncomputable def ell1 (P : ℝ × ℝ) : Prop := (P.1 + 4) ^ 2 / 25 + (P.2) ^ 2 / 9 = 1
noncomputable def ell2 (P : ℝ × ℝ) : Prop := (P.1 + 3) ^ 2 / 25 + ((P.2 - 2) ^ 2) / 16 = 1

theorem find_y_coordinate_of_P :
  ∃ y : ℝ,
    ell1 (0, y) ∧ ell2 (0, y) ∧
    y = 6 / 7 ∧
    6 + 7 = 13 :=
by
  sorry

end NUMINAMATH_GPT_find_y_coordinate_of_P_l970_97037


namespace NUMINAMATH_GPT_tangent_line_eqn_of_sine_at_point_l970_97043

theorem tangent_line_eqn_of_sine_at_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = Real.sin (x + Real.pi / 3)) →
  ∀ (p : ℝ × ℝ), p = (0, Real.sqrt 3 / 2) →
  ∃ (a b c : ℝ), a ≠ 0 ∧ b ≠ 0 ∧ (∀ x, f x = Real.sin (x + Real.pi / 3)) ∧
  (∀ x y, y = f x → a * x + b * y + c = 0 → x - 2 * y + Real.sqrt 3 = 0) :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_eqn_of_sine_at_point_l970_97043


namespace NUMINAMATH_GPT_number_of_members_l970_97000

-- Define the conditions
def knee_pad_cost : ℕ := 6
def jersey_cost : ℕ := knee_pad_cost + 7
def wristband_cost : ℕ := jersey_cost + 3
def cost_per_member : ℕ := 2 * (knee_pad_cost + jersey_cost + wristband_cost)
def total_expenditure : ℕ := 4080

-- Prove the number of members in the club
theorem number_of_members (h1 : knee_pad_cost = 6)
                          (h2 : jersey_cost = 13)
                          (h3 : wristband_cost = 16)
                          (h4 : cost_per_member = 70)
                          (h5 : total_expenditure = 4080) :
                          total_expenditure / cost_per_member = 58 := 
by 
  sorry

end NUMINAMATH_GPT_number_of_members_l970_97000


namespace NUMINAMATH_GPT_solve_inequality_l970_97097

def f (x : ℝ) : ℝ := |x + 1| - |x - 3|

theorem solve_inequality : ∀ x : ℝ, |f x| ≤ 4 :=
by
  intro x
  sorry

end NUMINAMATH_GPT_solve_inequality_l970_97097


namespace NUMINAMATH_GPT_find_a_degree_l970_97010

-- Definitions from conditions
def monomial_degree (x_exp y_exp : ℕ) : ℕ := x_exp + y_exp

-- Statement of the proof problem
theorem find_a_degree (a : ℕ) (h : monomial_degree 2 a = 6) : a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_degree_l970_97010


namespace NUMINAMATH_GPT_rationalize_denominator_XYZ_sum_l970_97050

noncomputable def a := (5 : ℝ)^(1/3)
noncomputable def b := (4 : ℝ)^(1/3)

theorem rationalize_denominator_XYZ_sum : 
  let X := 25
  let Y := 20
  let Z := 16
  let W := 1
  X + Y + Z + W = 62 :=
by 
  sorry

end NUMINAMATH_GPT_rationalize_denominator_XYZ_sum_l970_97050


namespace NUMINAMATH_GPT_transform_correct_l970_97039

variable {α : Type} [Mul α] [DecidableEq α]

theorem transform_correct (a b c : α) (h : a = b) : a * c = b * c :=
by sorry

end NUMINAMATH_GPT_transform_correct_l970_97039


namespace NUMINAMATH_GPT_distance_ratio_l970_97090

theorem distance_ratio (x : ℝ) (hx : abs x = 8) : abs (-4) / abs x = 1 / 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_distance_ratio_l970_97090


namespace NUMINAMATH_GPT_calculate_expression_l970_97020

theorem calculate_expression : 4 + (-8) / (-4) - (-1) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_calculate_expression_l970_97020


namespace NUMINAMATH_GPT_minimum_perimeter_rectangle_l970_97055

theorem minimum_perimeter_rectangle (S : ℝ) (hS : S > 0) :
  ∃ x y : ℝ, (x * y = S) ∧ (∀ u v : ℝ, (u * v = S) → (2 * (u + v) ≥ 4 * Real.sqrt S)) ∧ (x = Real.sqrt S ∧ y = Real.sqrt S) :=
by
  sorry

end NUMINAMATH_GPT_minimum_perimeter_rectangle_l970_97055


namespace NUMINAMATH_GPT_B_subset_A_iff_a_range_l970_97016

variable (a : ℝ)
def A : Set ℝ := {x | -2 ≤ x ∧ x < 4}
def B (a : ℝ) : Set ℝ := {x | x^2 - a*x - 4 ≤ 0}

theorem B_subset_A_iff_a_range :
  B a ⊆ A ↔ 0 ≤ a ∧ a < 3 :=
by
  sorry

end NUMINAMATH_GPT_B_subset_A_iff_a_range_l970_97016


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l970_97027

theorem arithmetic_sequence_sum 
  (a : ℕ → ℕ) 
  (h_arith_seq : ∀ n : ℕ, a n = 2 + (n - 5)) 
  (ha5 : a 5 = 2) : 
  (a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 2 * 9) := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l970_97027


namespace NUMINAMATH_GPT_hyperbola_eccentricity_l970_97046

theorem hyperbola_eccentricity (m : ℝ) (h : m > 0) 
(hyperbola_eq : ∀ (x y : ℝ), x^2 / 9 - y^2 / m = 1) 
(eccentricity : ∀ (e : ℝ), e = 2) 
: m = 27 :=
sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_l970_97046


namespace NUMINAMATH_GPT_find_largest_number_l970_97038

theorem find_largest_number :
  let a := -(abs (-3) ^ 3)
  let b := -((-3) ^ 3)
  let c := (-3) ^ 3
  let d := -(3 ^ 3)
  b = 27 ∧ b > a ∧ b > c ∧ b > d := by
  sorry

end NUMINAMATH_GPT_find_largest_number_l970_97038


namespace NUMINAMATH_GPT_system_has_negative_solution_iff_sum_zero_l970_97082

variables {a b c x y : ℝ}

-- Statement of the problem
theorem system_has_negative_solution_iff_sum_zero :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ (a * x + b * y = c) ∧ (b * x + c * y = a) ∧ (c * x + a * y = b)) ↔ (a + b + c = 0) := by
  sorry

end NUMINAMATH_GPT_system_has_negative_solution_iff_sum_zero_l970_97082


namespace NUMINAMATH_GPT_circles_intersect_l970_97017

theorem circles_intersect (m : ℝ) 
  (h₁ : ∃ x y, x^2 + y^2 = m) 
  (h₂ : ∃ x y, x^2 + y^2 + 6*x - 8*y + 21 = 0) : 
  9 < m ∧ m < 49 :=
by sorry

end NUMINAMATH_GPT_circles_intersect_l970_97017


namespace NUMINAMATH_GPT_second_quadrant_necessary_not_sufficient_l970_97070

variable (α : ℝ) -- Assuming α is a real number for generality.

-- Define what it means for an angle to be in the second quadrant (90° < α < 180°).
def in_second_quadrant (α : ℝ) : Prop :=
  90 < α ∧ α < 180

-- Define what it means for an angle to be obtuse (90° < α ≤ 180°).
def is_obtuse (α : ℝ) : Prop :=
  90 < α ∧ α ≤ 180

-- State the theorem to prove: 
-- "The angle α is in the second quadrant" is a necessary but not sufficient condition for "α is an obtuse angle".
theorem second_quadrant_necessary_not_sufficient : 
  (∀ α, is_obtuse α → in_second_quadrant α) ∧ 
  (∃ α, in_second_quadrant α ∧ ¬is_obtuse α) :=
sorry

end NUMINAMATH_GPT_second_quadrant_necessary_not_sufficient_l970_97070


namespace NUMINAMATH_GPT_number_is_3034_l970_97045

theorem number_is_3034 (number : ℝ) (h : number - 1002 / 20.04 = 2984) : number = 3034 :=
sorry

end NUMINAMATH_GPT_number_is_3034_l970_97045


namespace NUMINAMATH_GPT_jen_ate_eleven_suckers_l970_97074

/-- Representation of the sucker distribution problem and proving that Jen ate 11 suckers. -/
theorem jen_ate_eleven_suckers 
  (sienna_bailey : ℕ) -- Sienna's number of suckers is twice of what Bailey got.
  (jen_molly : ℕ)     -- Jen's number of suckers is twice of what Molly got plus 11.
  (molly_harmony : ℕ) -- Molly's number of suckers is 2 more than what she gave to Harmony.
  (harmony_taylor : ℕ)-- Harmony's number of suckers is 3 more than what she gave to Taylor.
  (taylor_end : ℕ)    -- Taylor ended with 6 suckers after eating 1 before giving 5 to Callie.
  (jen_start : ℕ)     -- Jen's initial number of suckers before eating half.
  (h1 : taylor_end = 6) 
  (h2 : harmony_taylor = taylor_end + 3) 
  (h3 : molly_harmony = harmony_taylor + 2) 
  (h4 : jen_molly = molly_harmony + 11) 
  (h5 : jen_start = jen_molly * 2) :
  jen_start / 2 = 11 := 
by
  -- given all the conditions, it would simplify to show
  -- that jen_start / 2 = 11
  sorry

end NUMINAMATH_GPT_jen_ate_eleven_suckers_l970_97074


namespace NUMINAMATH_GPT_circumscribed_circle_center_location_l970_97011

structure Trapezoid where
  is_isosceles : Bool
  angle_base : ℝ
  angle_between_diagonals : ℝ

theorem circumscribed_circle_center_location (T : Trapezoid)
  (h1 : T.is_isosceles = true)
  (h2 : T.angle_base = 50)
  (h3 : T.angle_between_diagonals = 40) :
  ∃ loc : String, loc = "Outside" := by
  sorry

end NUMINAMATH_GPT_circumscribed_circle_center_location_l970_97011


namespace NUMINAMATH_GPT_algebraic_expression_l970_97081

def ast (n : ℕ) : ℕ := sorry

axiom condition_1 : ast 1 = 1
axiom condition_2 : ∀ (n : ℕ), ast (n + 1) = 3 * ast n

theorem algebraic_expression (n : ℕ) :
  n > 0 → ast n = 3^(n - 1) :=
by
  -- Proof to be completed
  sorry

end NUMINAMATH_GPT_algebraic_expression_l970_97081


namespace NUMINAMATH_GPT_c_geq_one_l970_97091

theorem c_geq_one (a b : ℕ) (c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_eq : (a + 1 : ℝ) / (b + c) = (b : ℝ) / a) : c ≥ 1 :=
by sorry

end NUMINAMATH_GPT_c_geq_one_l970_97091


namespace NUMINAMATH_GPT_space_left_over_l970_97099

theorem space_left_over (D B : ℕ) (wall_length desk_length bookcase_length : ℝ) (h_wall : wall_length = 15)
  (h_desk : desk_length = 2) (h_bookcase : bookcase_length = 1.5) (h_eq : D = B)
  (h_max : 2 * D + 1.5 * B ≤ wall_length) :
  ∃ w : ℝ, w = wall_length - (D * desk_length + B * bookcase_length) ∧ w = 1 :=
by
  sorry

end NUMINAMATH_GPT_space_left_over_l970_97099


namespace NUMINAMATH_GPT_rectangles_with_one_gray_cell_l970_97072

-- Define the number of gray cells
def gray_cells : ℕ := 40

-- Define the total rectangles containing exactly one gray cell
def total_rectangles : ℕ := 176

-- The theorem we want to prove
theorem rectangles_with_one_gray_cell (h : gray_cells = 40) : total_rectangles = 176 := 
by 
  sorry

end NUMINAMATH_GPT_rectangles_with_one_gray_cell_l970_97072


namespace NUMINAMATH_GPT_polar_to_cartesian_l970_97059

theorem polar_to_cartesian :
  ∀ (ρ θ : ℝ), ρ = 3 ∧ θ = π / 6 → 
  (ρ * Real.cos θ, ρ * Real.sin θ) = (3 * Real.sqrt 3 / 2, 3 / 2) :=
by
  intro ρ θ
  rintro ⟨hρ, hθ⟩
  rw [hρ, hθ]
  sorry

end NUMINAMATH_GPT_polar_to_cartesian_l970_97059


namespace NUMINAMATH_GPT_union_A_B_compl_inter_A_B_l970_97015

-- Definitions based on the conditions
def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 1 ≤ x - 1 ∧ x - 1 < 3}

def B : Set ℝ := {x | 2 * x - 9 ≥ 6 - 3 * x}

-- The first proof statement
theorem union_A_B : A ∪ B = {x : ℝ | x ≥ 2} := by
  sorry

-- The second proof statement
theorem compl_inter_A_B : U \ (A ∩ B) = {x : ℝ | x < 3 ∨ x ≥ 4} := by
  sorry

end NUMINAMATH_GPT_union_A_B_compl_inter_A_B_l970_97015


namespace NUMINAMATH_GPT_neg_prop_p_equiv_l970_97025

open Classical

variable (x : ℝ)
def prop_p : Prop := ∀ x : ℝ, x^2 + 1 ≥ 0

theorem neg_prop_p_equiv : ¬ prop_p ↔ ∃ x : ℝ, x^2 + 1 < 0 := by
  sorry

end NUMINAMATH_GPT_neg_prop_p_equiv_l970_97025


namespace NUMINAMATH_GPT_new_home_fraction_l970_97056

variable {M H G : ℚ} -- Use ℚ (rational numbers)

def library_fraction (H : ℚ) (G : ℚ) (M : ℚ) : ℚ :=
  (1 / 3 * H + 2 / 5 * G + 1 / 2 * M) / M

theorem new_home_fraction (H_eq : H = 1 / 2 * M) (G_eq : G = 3 * H) :
  library_fraction H G M = 29 / 30 :=
by
  sorry

end NUMINAMATH_GPT_new_home_fraction_l970_97056


namespace NUMINAMATH_GPT_shirts_sold_l970_97018

theorem shirts_sold (initial_shirts remaining_shirts shirts_sold : ℕ) (h1 : initial_shirts = 49) (h2 : remaining_shirts = 28) : 
  shirts_sold = initial_shirts - remaining_shirts → 
  shirts_sold = 21 := 
by 
  sorry

end NUMINAMATH_GPT_shirts_sold_l970_97018


namespace NUMINAMATH_GPT_concentrate_to_water_ratio_l970_97035

theorem concentrate_to_water_ratio :
  ∀ (c w : ℕ), (∀ c, w = 3 * c) → (35 * 3 = 105) → (1 / 3 = (1 : ℝ) / (3 : ℝ)) :=
by
  intros c w h1 h2
  sorry

end NUMINAMATH_GPT_concentrate_to_water_ratio_l970_97035


namespace NUMINAMATH_GPT_find_length_second_platform_l970_97019

noncomputable def length_second_platform : Prop :=
  let train_length := 500  -- in meters
  let time_cross_platform := 35  -- in seconds
  let time_cross_pole := 8  -- in seconds
  let second_train_length := 250  -- in meters
  let time_cross_second_train := 45  -- in seconds
  let platform1_scale := 0.75
  let time_cross_platform1 := 27  -- in seconds
  let train_speed := train_length / time_cross_pole
  let platform1_length := train_speed * time_cross_platform1 - train_length
  let platform2_length := platform1_length / platform1_scale
  platform2_length = 1583.33

/- The proof is omitted -/
theorem find_length_second_platform : length_second_platform := sorry

end NUMINAMATH_GPT_find_length_second_platform_l970_97019


namespace NUMINAMATH_GPT_least_prime_in_sum_even_set_of_7_distinct_primes_l970_97004

noncomputable def is_prime (n : ℕ) : Prop := sorry -- Assume an implementation of prime numbers

theorem least_prime_in_sum_even_set_of_7_distinct_primes {q : Finset ℕ} 
  (hq_distinct : q.card = 7) 
  (hq_primes : ∀ n ∈ q, is_prime n) 
  (hq_sum_even : q.sum id % 2 = 0) :
  ∃ m ∈ q, m = 2 :=
by
  sorry

end NUMINAMATH_GPT_least_prime_in_sum_even_set_of_7_distinct_primes_l970_97004


namespace NUMINAMATH_GPT_ellens_initial_legos_l970_97092

-- Define the initial number of Legos as a proof goal
theorem ellens_initial_legos : ∀ (x y : ℕ), (y = x - 17) → (x = 2080) :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_ellens_initial_legos_l970_97092


namespace NUMINAMATH_GPT_sum_a2_a9_l970_97049

variable {a : ℕ → ℝ} -- Define the sequence a_n
variable {S : ℕ → ℝ} -- Define the sum sequence S_n

-- The conditions
def arithmetic_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ) : Prop :=
  S n = (n * (a 1 + a n)) / 2

axiom S_10 : arithmetic_sum S a 10
axiom S_10_value : S 10 = 100

-- The goal
theorem sum_a2_a9 (a : ℕ → ℝ) (S : ℕ → ℝ) (h1 : S 10 = 100) (h2 : arithmetic_sum S a 10) :
  a 2 + a 9 = 20 := 
sorry

end NUMINAMATH_GPT_sum_a2_a9_l970_97049


namespace NUMINAMATH_GPT_bread_rise_times_l970_97075

-- Defining the conditions
def rise_time : ℕ := 120
def kneading_time : ℕ := 10
def baking_time : ℕ := 30
def total_time : ℕ := 280

-- The proof statement
theorem bread_rise_times (n : ℕ) 
  (h1 : rise_time * n + kneading_time + baking_time = total_time) 
  : n = 2 :=
sorry

end NUMINAMATH_GPT_bread_rise_times_l970_97075


namespace NUMINAMATH_GPT_inverse_prop_l970_97013

theorem inverse_prop (a c : ℝ) : (∀ (a : ℝ), a > 0 → a * c^2 ≥ 0) → (∀ (x : ℝ), x * c^2 ≥ 0 → x > 0) :=
by
  sorry

end NUMINAMATH_GPT_inverse_prop_l970_97013


namespace NUMINAMATH_GPT_four_distinct_real_roots_l970_97093

noncomputable def f (x d : ℝ) : ℝ := x^2 + 10*x + d

theorem four_distinct_real_roots (d : ℝ) :
  (∀ r, f r d = 0 → (∃! x, f x d = r)) → d < 25 :=
by
  sorry

end NUMINAMATH_GPT_four_distinct_real_roots_l970_97093


namespace NUMINAMATH_GPT_most_likely_maximum_people_in_room_l970_97008

theorem most_likely_maximum_people_in_room :
  ∃ k, 1 ≤ k ∧ k ≤ 3000 ∧
    (∃ p : ℕ → ℕ → ℕ → ℕ, (p 1000 1000 1000) = 1019) ∧
    (∀ a b c : ℕ, a + b + c = 3000 → a ≤ 1019 ∧ b ≤ 1019 ∧ c ≤ 1019 → max a (max b c) = 1019) :=
sorry

end NUMINAMATH_GPT_most_likely_maximum_people_in_room_l970_97008


namespace NUMINAMATH_GPT_inequality_solution_l970_97048

theorem inequality_solution (x : ℝ) : (x^2 - x - 2 < 0) ↔ (-1 < x ∧ x < 2) :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l970_97048


namespace NUMINAMATH_GPT_girls_in_school_l970_97022

noncomputable def num_of_girls (total_students : ℕ) (sampled_students : ℕ) (sampled_diff : ℤ) : ℕ :=
  sorry

theorem girls_in_school :
  let total_students := 1600
  let sampled_students := 200
  let sampled_diff := 10
  num_of_girls total_students sampled_students sampled_diff = 760 :=
  sorry

end NUMINAMATH_GPT_girls_in_school_l970_97022


namespace NUMINAMATH_GPT_yuna_has_most_apples_l970_97062

def apples_count_jungkook : ℕ :=
  6 / 3

def apples_count_yoongi : ℕ :=
  4

def apples_count_yuna : ℕ :=
  5

theorem yuna_has_most_apples : apples_count_yuna > apples_count_yoongi ∧ apples_count_yuna > apples_count_jungkook :=
by
  sorry

end NUMINAMATH_GPT_yuna_has_most_apples_l970_97062


namespace NUMINAMATH_GPT_sum_of_integers_is_96_l970_97084

theorem sum_of_integers_is_96 (x y : ℤ) (h1 : x = 32) (h2 : y = 2 * x) : x + y = 96 := 
by
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_96_l970_97084


namespace NUMINAMATH_GPT_problem_l970_97066

theorem problem (a b : ℤ) (h : (2 * a + b) ^ 2 + |b - 2| = 0) : (-a - b) ^ 2014 = 1 := 
by
  sorry

end NUMINAMATH_GPT_problem_l970_97066


namespace NUMINAMATH_GPT_system1_solution_system2_solution_l970_97003

theorem system1_solution (x y : ℤ) (h1 : x - y = 2) (h2 : x + 1 = 2 * (y - 1)) :
  x = 7 ∧ y = 5 :=
sorry

theorem system2_solution (x y : ℤ) (h1 : 2 * x + 3 * y = 1) (h2 : (y - 1) * 3 = (x - 2) * 4) :
  x = 1 ∧ y = -1 / 3 :=
sorry

end NUMINAMATH_GPT_system1_solution_system2_solution_l970_97003


namespace NUMINAMATH_GPT_frequency_of_middle_group_l970_97063

theorem frequency_of_middle_group (sample_size : ℕ) (x : ℝ) (h : sample_size = 160) (h_rel_freq : x = 0.2) 
  (h_relation : x = (1 / 4) * (10 * x)) : 
  sample_size * x = 32 :=
by
  sorry

end NUMINAMATH_GPT_frequency_of_middle_group_l970_97063


namespace NUMINAMATH_GPT_min_weighings_to_order_four_stones_l970_97006

theorem min_weighings_to_order_four_stones : ∀ (A B C D : ℝ), 
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D → ∃ n, n = 5 :=
by sorry

end NUMINAMATH_GPT_min_weighings_to_order_four_stones_l970_97006


namespace NUMINAMATH_GPT_base_k_to_decimal_l970_97009

theorem base_k_to_decimal (k : ℕ) (h : 1 * k^2 + 3 * k + 2 = 30) : k = 4 :=
  sorry

end NUMINAMATH_GPT_base_k_to_decimal_l970_97009


namespace NUMINAMATH_GPT_total_problems_l970_97023

-- Definitions based on conditions
def math_pages : ℕ := 4
def reading_pages : ℕ := 6
def problems_per_page : ℕ := 4

-- Statement of the problem
theorem total_problems : math_pages + reading_pages * problems_per_page = 40 :=
by
  unfold math_pages reading_pages problems_per_page
  sorry

end NUMINAMATH_GPT_total_problems_l970_97023


namespace NUMINAMATH_GPT_fraction_simplify_l970_97052

variable (a b c : ℝ)

theorem fraction_simplify
  (h₀ : a ≠ 0)
  (h₁ : b ≠ 0)
  (h₂ : c ≠ 0)
  (h₃ : a + 2 * b + 3 * c ≠ 0) :
  (a^2 + 4 * b^2 - 9 * c^2 + 4 * a * b) / (a^2 + 9 * c^2 - 4 * b^2 + 6 * a * c) =
  (a + 2 * b - 3 * c) / (a - 2 * b + 3 * c) := by
  sorry

end NUMINAMATH_GPT_fraction_simplify_l970_97052


namespace NUMINAMATH_GPT_members_on_fathers_side_are_10_l970_97034

noncomputable def members_father_side (total : ℝ) (ratio : ℝ) (members_mother_side_more: ℝ) : Prop :=
  let F := total / (1 + ratio)
  F = 10

theorem members_on_fathers_side_are_10 :
  ∀ (total : ℝ) (ratio : ℝ), 
  total = 23 → 
  ratio = 0.30 →
  members_father_side total ratio (ratio * total) :=
by
  intros total ratio htotal hratio
  have h1 : total = 23 := htotal
  have h2 : ratio = 0.30 := hratio
  rw [h1, h2]
  sorry

end NUMINAMATH_GPT_members_on_fathers_side_are_10_l970_97034


namespace NUMINAMATH_GPT_distance_between_trees_l970_97051

theorem distance_between_trees (L : ℕ) (n : ℕ) (hL : L = 150) (hn : n = 11) (h_end_trees : n > 1) : 
  (L / (n - 1)) = 15 :=
by
  -- Replace with the appropriate proof
  sorry

end NUMINAMATH_GPT_distance_between_trees_l970_97051


namespace NUMINAMATH_GPT_possible_numbers_tom_l970_97087

theorem possible_numbers_tom (n : ℕ) (h1 : 180 ∣ n) (h2 : 75 ∣ n) (h3 : 500 < n ∧ n < 2500) : n = 900 ∨ n = 1800 :=
sorry

end NUMINAMATH_GPT_possible_numbers_tom_l970_97087


namespace NUMINAMATH_GPT_product_of_fractions_l970_97053

theorem product_of_fractions :
  (3/4) * (4/5) * (5/6) * (6/7) = 3/7 :=
by
  sorry

end NUMINAMATH_GPT_product_of_fractions_l970_97053


namespace NUMINAMATH_GPT_sum_of_x_for_ggg_eq_neg2_l970_97079

noncomputable def g (x : ℝ) := (x^2) / 3 + x - 2

theorem sum_of_x_for_ggg_eq_neg2 : (∃ x1 x2 : ℝ, (g (g (g x1)) = -2 ∧ g (g (g x2)) = -2 ∧ x1 ≠ x2)) ∧ (x1 + x2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_x_for_ggg_eq_neg2_l970_97079


namespace NUMINAMATH_GPT_count_valid_ks_l970_97060

theorem count_valid_ks : 
  ∃ (ks : Finset ℕ), (∀ k ∈ ks, k > 0 ∧ k ≤ 50 ∧ 
    ∀ n : ℕ, n > 0 → 7 ∣ (2 * 3^(6 * n) + k * 2^(3 * n + 1) - 1)) ∧ ks.card = 7 :=
sorry

end NUMINAMATH_GPT_count_valid_ks_l970_97060


namespace NUMINAMATH_GPT_angela_insects_l970_97036

theorem angela_insects:
  ∀ (A J D : ℕ), 
    A = J / 2 → 
    J = 5 * D → 
    D = 30 → 
    A = 75 :=
by
  intro A J D
  intro hA hJ hD
  sorry

end NUMINAMATH_GPT_angela_insects_l970_97036


namespace NUMINAMATH_GPT_pear_distribution_problem_l970_97083

-- Defining the given conditions as hypotheses
variables (G P : ℕ)

-- The first condition: P = G + 1
def condition1 : Prop := P = G + 1

-- The second condition: P = 2G - 2
def condition2 : Prop := P = 2 * G - 2

-- The main theorem to prove
theorem pear_distribution_problem (h1 : condition1 G P) (h2 : condition2 G P) :
  G = 3 ∧ P = 4 :=
by
  sorry

end NUMINAMATH_GPT_pear_distribution_problem_l970_97083


namespace NUMINAMATH_GPT_license_plate_configurations_l970_97032

theorem license_plate_configurations :
  (3 * 10^4 = 30000) :=
by
  sorry

end NUMINAMATH_GPT_license_plate_configurations_l970_97032


namespace NUMINAMATH_GPT_problem_statement_l970_97094

noncomputable def golden_ratio : ℝ := (1 + Real.sqrt 5) / 2

theorem problem_statement (S : ℝ) (h1 : S = golden_ratio) :
  S^(S^(S^2 - S⁻¹) - S⁻¹) - S⁻¹ = 0 :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l970_97094


namespace NUMINAMATH_GPT_boy_to_total_ratio_l970_97095

-- Problem Definitions
variables (b g : ℕ) -- number of boys and number of girls

-- Hypothesis: The probability of choosing a boy is (4/5) the probability of choosing a girl
def probability_boy := b / (b + g : ℕ)
def probability_girl := g / (b + g : ℕ)

theorem boy_to_total_ratio (h : probability_boy b g = (4 / 5) * probability_girl b g) : 
  b / (b + g : ℕ) = 4 / 9 :=
sorry

end NUMINAMATH_GPT_boy_to_total_ratio_l970_97095


namespace NUMINAMATH_GPT_sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l970_97096

open Real

namespace TriangleProofs

variables 
  (A B C : ℝ) 
  (a b c : ℝ) 
  (BA BC : ℝ) 
  (h1 : sin B = sqrt 7 / 4) 
  (h2 : (cos A / sin A + cos C / sin C = 4 * sqrt 7 / 7)) 
  (h3 : BA * BC = 3 / 2)
  (h4 : a = b ∧ c = b)

-- 1. Prove that sin A * sin C = sin^2 B
theorem sin_a_mul_sin_c_eq_sin_sq_b : sin A * sin C = sin B ^ 2 := 
by sorry

-- 2. Prove that 0 < B ≤ π / 3
theorem zero_lt_B_le_pi_div_3 : 0 < B ∧ B ≤ π / 3 := 
by sorry

-- 3. Find the magnitude of the vector sum.
theorem magnitude_BC_add_BA : abs (BC + BA) = 2 * sqrt 2 := 
by sorry

end TriangleProofs

end NUMINAMATH_GPT_sin_a_mul_sin_c_eq_sin_sq_b_zero_lt_B_le_pi_div_3_magnitude_BC_add_BA_l970_97096


namespace NUMINAMATH_GPT_find_special_numbers_l970_97069

theorem find_special_numbers :
  {N : ℕ | ∃ k m a, N = m + 10^k * a ∧ 0 ≤ a ∧ a < 10 ∧ 0 ≤ k ∧ m < 10^k 
                ∧ ¬(N % 10 = 0) 
                ∧ (N = 6 * (m + 10^(k+1) * (0 : ℕ))) } = {12, 24, 36, 48} := 
by sorry

end NUMINAMATH_GPT_find_special_numbers_l970_97069


namespace NUMINAMATH_GPT_xyz_problem_l970_97012

theorem xyz_problem (x y : ℝ) (h1 : x + y - x * y = 155) (h2 : x^2 + y^2 = 325) : |x^3 - y^3| = 4375 := by
  sorry

end NUMINAMATH_GPT_xyz_problem_l970_97012


namespace NUMINAMATH_GPT_minimum_value_l970_97073

theorem minimum_value (a b : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 2 * a + 3 * b = 1) : 
  26 ≤ (2 / a + 3 / b) :=
sorry

end NUMINAMATH_GPT_minimum_value_l970_97073


namespace NUMINAMATH_GPT_maximize_village_value_l970_97064

theorem maximize_village_value :
  ∃ (x y z : ℕ), 
  x + y + z = 20 ∧ 
  2 * x + 3 * y + 4 * z = 50 ∧ 
  (∀ x' y' z' : ℕ, 
      x' + y' + z' = 20 → 2 * x' + 3 * y' + 4 * z' = 50 → 
      (1.2 * x + 1.5 * y + 1.2 * z : ℝ) ≥ (1.2 * x' + 1.5 * y' + 1.2 * z' : ℝ)) ∧ 
  x = 10 ∧ y = 10 ∧ z = 0 := by 
  sorry

end NUMINAMATH_GPT_maximize_village_value_l970_97064


namespace NUMINAMATH_GPT_final_price_of_jacket_l970_97014

noncomputable def originalPrice : ℝ := 250
noncomputable def firstDiscount : ℝ := 0.60
noncomputable def secondDiscount : ℝ := 0.25

theorem final_price_of_jacket :
  let P := originalPrice
  let D1 := firstDiscount
  let D2 := secondDiscount
  let priceAfterFirstDiscount := P * (1 - D1)
  let finalPrice := priceAfterFirstDiscount * (1 - D2)
  finalPrice = 75 :=
by
  sorry

end NUMINAMATH_GPT_final_price_of_jacket_l970_97014


namespace NUMINAMATH_GPT_cube_identity_l970_97001

theorem cube_identity (x y : ℝ) (h1 : x + y = 5) (h2 : x^2 + y^2 = 17) : x^3 + y^3 = 65 := by
  sorry

end NUMINAMATH_GPT_cube_identity_l970_97001
