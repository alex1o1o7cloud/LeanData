import Mathlib

namespace smaller_number_l290_290180

theorem smaller_number (L S : ℕ) (h₁ : L - S = 2395) (h₂ : L = 6 * S + 15) : S = 476 :=
by
sorry

end smaller_number_l290_290180


namespace ram_weight_increase_percentage_l290_290330

theorem ram_weight_increase_percentage :
  ∃ r s r_new: ℝ,
  r / s = 4 / 5 ∧ 
  r + s = 72 ∧ 
  s * 1.19 = 47.6 ∧
  r_new = 82.8 - 47.6 ∧ 
  (r_new - r) / r * 100 = 10 :=
by
  sorry

end ram_weight_increase_percentage_l290_290330


namespace usual_time_of_train_l290_290338

theorem usual_time_of_train (S T : ℝ) (h_speed : S ≠ 0) 
(h_speed_ratio : ∀ (T' : ℝ), T' = T + 3/4 → S * T = (4/5) * S * T' → T = 3) : Prop :=
  T = 3

end usual_time_of_train_l290_290338


namespace find_other_root_of_quadratic_l290_290553

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end find_other_root_of_quadratic_l290_290553


namespace original_curve_equation_l290_290120

theorem original_curve_equation (x y : ℝ) (θ : ℝ) (hθ : θ = π / 4)
  (h : (∃ P : ℝ × ℝ, P = (x, y) ∧ (∃ P' : ℝ × ℝ, P' = (x * Real.cos θ - y * Real.sin θ, x * Real.sin θ + y * Real.cos θ) ∧ ((P'.fst)^2 - (P'.snd)^2 = 2)))) :
  x * y = -1 :=
sorry

end original_curve_equation_l290_290120


namespace negation_of_abs_x_minus_2_lt_3_l290_290376

theorem negation_of_abs_x_minus_2_lt_3 :
  ¬ (∀ x : ℝ, |x - 2| < 3) ↔ ∃ x : ℝ, |x - 2| ≥ 3 :=
by
  sorry

end negation_of_abs_x_minus_2_lt_3_l290_290376


namespace right_triangle_l290_290069

-- Definitions for each set of segments
def setA := (2 : ℕ, 3 : ℕ, 4 : ℕ)
def setB := (Real.sqrt 7, 3 : ℕ, 5 : ℕ)
def setC := (6 : ℕ, 8 : ℕ, 10 : ℕ)
def setD := (5 : ℕ, 12 : ℕ, 12 : ℕ)

-- Main statement that needs to be proven
theorem right_triangle : 
  ¬((setA.1^2 + setA.2^2 = setA.3^2)) ∧
  ¬((setB.1^2 + setB.2^2 = setB.3^2)) ∧
   (setC.1^2 + setC.2^2 = setC.3^2) ∧
  ¬((setD.1^2 + setD.2^2 = setD.3^2)) :=
by 
  sorry

end right_triangle_l290_290069


namespace shark_fin_falcata_area_is_correct_l290_290480

noncomputable def radius_large : ℝ := 3
noncomputable def center_large : ℝ × ℝ := (0, 0)

noncomputable def radius_small : ℝ := 3 / 2
noncomputable def center_small : ℝ × ℝ := (0, 3 / 2)

noncomputable def area_large_quarter_circle : ℝ := (1 / 4) * Real.pi * (radius_large ^ 2)
noncomputable def area_small_semicircle : ℝ := (1 / 2) * Real.pi * (radius_small ^ 2)

noncomputable def shark_fin_falcata_area (area_large_quarter_circle area_small_semicircle : ℝ) : ℝ := 
  area_large_quarter_circle - area_small_semicircle

theorem shark_fin_falcata_area_is_correct : 
  shark_fin_falcata_area area_large_quarter_circle area_small_semicircle = (9 * Real.pi) / 8 := 
by
  sorry

end shark_fin_falcata_area_is_correct_l290_290480


namespace mn_sum_l290_290006

theorem mn_sum (M N : ℚ) (h1 : (4 : ℚ) / 7 = M / 63) (h2 : (4 : ℚ) / 7 = 84 / N) : M + N = 183 := sorry

end mn_sum_l290_290006


namespace sqrt_fraction_simplification_l290_290111

theorem sqrt_fraction_simplification :
  (Real.sqrt ((25 / 49) - (16 / 81)) = (Real.sqrt 1241) / 63) := by
  sorry

end sqrt_fraction_simplification_l290_290111


namespace each_mouse_not_visit_with_every_other_once_l290_290995

theorem each_mouse_not_visit_with_every_other_once : 
    (∃ mice: Finset ℕ, mice.card = 24 ∧ (∀ f : ℕ → Finset ℕ, 
    (∀ n, (f n).card = 4) ∧ 
    (∀ i j, i ≠ j → (f i ∩ f j ≠ ∅) → (f i ∩ f j).card ≠ (mice.card - 1)))
    ) → false := 
by
  sorry

end each_mouse_not_visit_with_every_other_once_l290_290995


namespace video_call_cost_l290_290947

-- Definitions based on the conditions
def charge_rate : ℕ := 30    -- Charge rate in won per ten seconds
def call_duration : ℕ := 2 * 60 + 40  -- Call duration in seconds

-- The proof statement, anticipating the solution to be a total cost calculation
theorem video_call_cost : (call_duration / 10) * charge_rate = 480 :=
by
  -- Placeholder for the proof
  sorry

end video_call_cost_l290_290947


namespace ratio_50kg_to_05tons_not_100_to_1_l290_290599

theorem ratio_50kg_to_05tons_not_100_to_1 (weight1 : ℕ) (weight2 : ℕ) (r : ℕ) 
  (h1 : weight1 = 50) (h2 : weight2 = 500) (h3 : r = 100) : ¬ (weight1 * r = weight2) := 
by
  sorry

end ratio_50kg_to_05tons_not_100_to_1_l290_290599


namespace original_mixture_percentage_l290_290065

def mixture_percentage_acid (a w : ℕ) : ℚ :=
  a / (a + w)

theorem original_mixture_percentage (a w : ℕ) :
  (a / (a + w+2) = 1 / 4) ∧ ((a + 2) / (a + w + 4) = 2 / 5) → 
  mixture_percentage_acid a w = 1 / 3 :=
by
  sorry

end original_mixture_percentage_l290_290065


namespace minimize_expression_pos_int_l290_290688

theorem minimize_expression_pos_int (n : ℕ) (hn : 0 < n) : 
  (∀ m : ℕ, 0 < m → (m / 3 + 27 / m : ℝ) ≥ (9 / 3 + 27 / 9)) :=
sorry

end minimize_expression_pos_int_l290_290688


namespace two_pow_2023_mod_17_l290_290669

theorem two_pow_2023_mod_17 : (2 ^ 2023) % 17 = 4 := 
by
  sorry

end two_pow_2023_mod_17_l290_290669


namespace number_of_regular_soda_bottles_l290_290513

-- Define the total number of bottles and the number of diet soda bottles
def total_bottles : ℕ := 30
def diet_soda_bottles : ℕ := 2

-- Define the number of regular soda bottles
def regular_soda_bottles : ℕ := total_bottles - diet_soda_bottles

-- Statement of the main proof problem
theorem number_of_regular_soda_bottles : regular_soda_bottles = 28 := by
  -- Proof goes here
  sorry

end number_of_regular_soda_bottles_l290_290513


namespace no_real_roots_of_polynomial_l290_290478

noncomputable def p (x : ℝ) : ℝ := sorry

theorem no_real_roots_of_polynomial (p : ℝ → ℝ) (h_deg : ∃ n : ℕ, n ≥ 1 ∧ ∀ x: ℝ, p x = x^n) :
  (∀ x, p x * p (2 * x^2) = p (3 * x^3 + x)) →
  ¬ ∃ α : ℝ, p α = 0 := sorry

end no_real_roots_of_polynomial_l290_290478


namespace mean_days_jogged_l290_290751

open Real

theorem mean_days_jogged 
  (p1 : ℕ := 5) (d1 : ℕ := 1)
  (p2 : ℕ := 4) (d2 : ℕ := 3)
  (p3 : ℕ := 10) (d3 : ℕ := 5)
  (p4 : ℕ := 7) (d4 : ℕ := 10)
  (p5 : ℕ := 3) (d5 : ℕ := 15)
  (p6 : ℕ := 1) (d6 : ℕ := 20) : 
  ( (p1 * d1 + p2 * d2 + p3 * d3 + p4 * d4 + p5 * d5 + p6 * d6) / (p1 + p2 + p3 + p4 + p5 + p6) : ℝ) = 6.73 :=
by
  sorry

end mean_days_jogged_l290_290751


namespace value_of_t_plus_one_over_t_l290_290907

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l290_290907


namespace sum_repeating_decimals_as_fraction_l290_290681

-- Definitions for repeating decimals
def rep2 : ℝ := 0.2222
def rep02 : ℝ := 0.0202
def rep0002 : ℝ := 0.00020002

-- Prove the sum of the repeating decimals is equal to the given fraction
theorem sum_repeating_decimals_as_fraction :
  rep2 + rep02 + rep0002 = (2224 / 9999 : ℝ) :=
sorry

end sum_repeating_decimals_as_fraction_l290_290681


namespace vasya_numbers_l290_290817

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290817


namespace problem_1_problem_2_l290_290746

-- Define the propositions p and q
def proposition_p (x a : ℝ) := x^2 - (a + 1/a) * x + 1 < 0
def proposition_q (x : ℝ) := x^2 - 4 * x + 3 ≤ 0

-- Problem 1: Given a = 2 and both p and q are true, find the range of x
theorem problem_1 (a : ℝ) (x : ℝ) (ha : a = 2) (hp : proposition_p x a) (hq : proposition_q x) :
  1 ≤ x ∧ x < 2 :=
sorry

-- Problem 2: Prove that if p is a necessary but not sufficient condition for q, then 3 < a
theorem problem_2 (a : ℝ)
  (h_ns : ∀ x, proposition_q x → proposition_p x a)
  (h_not_s : ∃ x, ¬ (proposition_q x → proposition_p x a)) :
  3 < a :=
sorry

end problem_1_problem_2_l290_290746


namespace no_consecutive_numbers_adjacent_implies_probability_l290_290773

noncomputable def cube_faces := Fin 6

def consecutive_numbers (a b : ℕ) : Prop :=
  (a = 1 ∧ b = 6) ∨ (a = 6 ∧ b = 1) ∨ (b = a + 1) ∨ (a = b + 1)

def valid_cube_configuration (f : cube_faces → ℕ) : Prop :=
  ∀ i j, (f i = 1 ∧ f j = 2) ∨
         consecutive_numbers (f i) (f j) →
         ¬ adjacent i j

theorem no_consecutive_numbers_adjacent_implies_probability
  (f : cube_faces → ℕ) :
  (∀ i j : cube_faces, consecutive_numbers (f i) (f j) → ¬ adjacent i j) →
  24.to_rat / 120.to_rat = 1.to_rat / 5.to_rat :=
sorry

end no_consecutive_numbers_adjacent_implies_probability_l290_290773


namespace inscribed_circle_radius_in_quadrilateral_pyramid_l290_290593

theorem inscribed_circle_radius_in_quadrilateral_pyramid
  (a : ℝ) (α : ℝ)
  (h_pos : 0 < a) (h_α : 0 < α ∧ α < π / 2) :
  ∃ r : ℝ, r = a * Real.sqrt 2 / (1 + 2 * Real.cos α + Real.sqrt (4 * Real.cos α ^ 2 + 1)) :=
by
  sorry

end inscribed_circle_radius_in_quadrilateral_pyramid_l290_290593


namespace speed_in_still_water_l290_290214

-- Define the conditions: upstream and downstream speeds.
def upstream_speed : ℝ := 10
def downstream_speed : ℝ := 20

-- Define the still water speed theorem.
theorem speed_in_still_water : (upstream_speed + downstream_speed) / 2 = 15 := by
  sorry

end speed_in_still_water_l290_290214


namespace one_thirds_in_nine_thirds_l290_290281

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l290_290281


namespace abc_inequality_l290_290449

theorem abc_inequality 
  (a b c : ℝ) 
  (h1 : 0 < a) 
  (h2 : 0 < b) 
  (h3 : 0 < c) 
  (h4 : a * b * c = 1) 
  : 
  (ab / (a^5 + b^5 + ab) + bc / (b^5 + c^5 + bc) + ca / (c^5 + a^5 + ca) ≤ 1) := 
by 
  sorry

end abc_inequality_l290_290449


namespace vasya_numbers_l290_290801

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290801


namespace angle_BDC_is_30_l290_290592

theorem angle_BDC_is_30 
    (A E C B D : ℝ) 
    (hA : A = 50) 
    (hE : E = 60) 
    (hC : C = 40) : 
    BDC = 30 :=
by
  sorry

end angle_BDC_is_30_l290_290592


namespace sum_in_base5_correct_l290_290254

-- Define numbers in base 5
def n1 : ℕ := 231
def n2 : ℕ := 414
def n3 : ℕ := 123

-- Function to convert a number from base 5 to base 10
def base5_to_base10(n : ℕ) : ℕ :=
  let d0 := n % 10
  let d1 := (n / 10) % 10
  let d2 := (n / 100)
  d0 * 1 + d1 * 5 + d2 * 25

-- Convert the given numbers from base 5 to base 10
def n1_base10 : ℕ := base5_to_base10 n1
def n2_base10 : ℕ := base5_to_base10 n2
def n3_base10 : ℕ := base5_to_base10 n3

-- Base 10 sum
def sum_base10 : ℕ := n1_base10 + n2_base10 + n3_base10

-- Function to convert a number from base 10 to base 5
def base10_to_base5(n : ℕ) : ℕ :=
  let d0 := n % 5
  let d1 := (n / 5) % 5
  let d2 := (n / 25) % 5
  let d3 := (n / 125)
  d0 * 1 + d1 * 10 + d2 * 100 + d3 * 1000

-- Convert the sum from base 10 to base 5
def sum_base5 : ℕ := base10_to_base5 sum_base10

-- The theorem to prove the sum in base 5 is 1323_5
theorem sum_in_base5_correct : sum_base5 = 1323 := by
  -- Proof steps would go here, but we insert sorry to skip it
  sorry

end sum_in_base5_correct_l290_290254


namespace mean_is_12_point_8_l290_290768

variable (m : ℝ)
variable median_condition : m + 7 = 12

theorem mean_is_12_point_8 (m : ℝ) (median_condition : m + 7 = 12) : 
(mean := (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5) = 64 / 5 :=
by {
  sorry
}

end mean_is_12_point_8_l290_290768


namespace geometric_sequence_sum_l290_290302

theorem geometric_sequence_sum (k : ℕ) (h1 : a_1 = 1) (h2 : a_k = 243) (h3 : q = 3) : S_k = 364 := 
by 
  -- Sorry is used here to skip the proof
  sorry

end geometric_sequence_sum_l290_290302


namespace equation_D_has_two_distinct_real_roots_l290_290068

def quadratic_has_two_distinct_real_roots (a b c : ℝ) : Prop :=
  b^2 - 4 * a * c > 0

theorem equation_D_has_two_distinct_real_roots : quadratic_has_two_distinct_real_roots 1 2 (-1) :=
by {
  sorry
}

end equation_D_has_two_distinct_real_roots_l290_290068


namespace side_length_square_field_l290_290505

-- Definitions based on the conditions.
def time_taken := 56 -- in seconds
def speed := 9 * 1000 / 3600 -- in meters per second, converting 9 km/hr to m/s
def distance_covered := speed * time_taken -- calculating the distance covered in meters
def perimeter := 4 * 35 -- defining the perimeter given the side length is 35

-- Problem statement for proof: We need to prove that the calculated distance covered matches the perimeter.
theorem side_length_square_field : distance_covered = perimeter :=
by
  sorry

end side_length_square_field_l290_290505


namespace find_analytical_expression_of_f_l290_290721

variable (f : ℝ → ℝ)

theorem find_analytical_expression_of_f
  (h : ∀ x : ℝ, f (2 * x + 1) = 4 * x^2 + 4 * x) :
  ∀ x : ℝ, f x = x^2 - 1 :=
sorry

end find_analytical_expression_of_f_l290_290721


namespace greatest_integer_less_than_neg_21_over_5_l290_290838

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ n : ℤ, n < -21 / 5 ∧ ∀ m : ℤ, m < -21 / 5 → m ≤ n :=
begin
  use -5,
  split,
  { linarith },
  { intros m h,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l290_290838


namespace division_remainder_190_21_l290_290952

theorem division_remainder_190_21 :
  190 = 21 * 9 + 1 :=
sorry

end division_remainder_190_21_l290_290952


namespace range_of_a_l290_290298

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - a * x + a > 0) ↔ (0 < a ∧ a < 4) :=
sorry

end range_of_a_l290_290298


namespace total_trees_cut_down_l290_290735

-- Definitions based on conditions in the problem
def trees_per_day_james : ℕ := 20
def days_with_just_james : ℕ := 2
def total_trees_by_james := trees_per_day_james * days_with_just_james

def brothers : ℕ := 2
def days_with_brothers : ℕ := 3
def trees_per_day_brothers := (20 * (100 - 20)) / 100 -- 20% fewer than James
def trees_per_day_total := brothers * trees_per_day_brothers + trees_per_day_james

def total_trees_with_brothers := trees_per_day_total * days_with_brothers

-- The statement to be proved
theorem total_trees_cut_down : total_trees_by_james + total_trees_with_brothers = 136 := by
  sorry

end total_trees_cut_down_l290_290735


namespace symmetric_points_difference_l290_290417

-- We start by defining the points and their coordinates.
def point_A_x : ℤ := -4
def point_A_y : ℤ := 2

def point_B_x : ℤ := -point_A_x
def point_B_y : ℤ := -point_A_y

-- We state the theorem using the given condition:
theorem symmetric_points_difference : point_B_x - point_B_y = 6 :=
by
  -- We specify the values directly based on the properties of symmetry.
  have h1 : point_B_x = 4 := by {
    sorry
  }
  have h2 : point_B_y = -2 := by {
    sorry
  }
  -- Now we prove the required equality.
  calc 
    4 - (-2) = 4 + 2 := by sorry
    ... = 6 := by sorry

end symmetric_points_difference_l290_290417


namespace digits_sum_unique_l290_290192

variable (A B C D E F G H : ℕ)

theorem digits_sum_unique :
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧ A ≠ G ∧ A ≠ H ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧ B ≠ G ∧ B ≠ H ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧ C ≠ G ∧ C ≠ H ∧
  D ≠ E ∧ D ≠ F ∧ D ≠ G ∧ D ≠ H ∧
  E ≠ F ∧ E ≠ G ∧ E ≠ H ∧
  F ≠ G ∧ F ≠ H ∧
  G ≠ H ∧
  0 ≤ A ∧ A ≤ 9 ∧ 0 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9 ∧ 0 ≤ D ∧ D ≤ 9 ∧
  0 ≤ E ∧ E ≤ 9 ∧ 0 ≤ F ∧ F ≤ 9 ∧ 0 ≤ G ∧ G ≤ 9 ∧ 0 ≤ H ∧ H ≤ 9 ∧
  (A * 1000 + B * 100 + C * 10 + D) + (E * 1000 + F * 100 + G * 10 + H) = 10652 ∧
  A = 9 ∧ B = 5 ∧ C = 6 ∧ D = 7 ∧
  E = 1 ∧ F = 0 ∧ G = 8 ∧ H = 5 :=
sorry

end digits_sum_unique_l290_290192


namespace angie_pretzels_l290_290664

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end angie_pretzels_l290_290664


namespace solution_one_solution_two_l290_290259

section

variables {a x : ℝ}

def f (x : ℝ) (a : ℝ) := |2 * x - a| - |x + 1|

-- (1) Prove the solution set for f(x) > 2 when a = 1 is (-∞, -2/3) ∪ (4, ∞)
theorem solution_one (x : ℝ) : f x 1 > 2 ↔ x < -2/3 ∨ x > 4 :=
by sorry

-- (2) Prove the range of a for which f(x) + |x + 1| + x > a² - 1/2 always holds for x ∈ ℝ is (-1/2, 1)
theorem solution_two (a : ℝ) : 
  (∀ x, f x a + |x + 1| + x > a^2 - 1/2) ↔ -1/2 < a ∧ a < 1 :=
by sorry

end

end solution_one_solution_two_l290_290259


namespace find_a1_l290_290023

variable (a : ℕ → ℤ) (S : ℕ → ℤ)

def is_arithmetic_seq (a : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, a (n + 1) = a n + d

def sum_n_terms (S : ℕ → ℤ) (a : ℕ → ℤ) : Prop :=
  ∀ n, S n = (n * (a 1 + a n)) / 2

theorem find_a1 (h1 : is_arithmetic_seq a (-2)) 
               (h2 : sum_n_terms S a) 
               (h3 : S 10 = S 11) : 
  a 1 = 20 :=
sorry

end find_a1_l290_290023


namespace evaluate_g_l290_290108

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_l290_290108


namespace find_S6_l290_290270

-- sum of the first n terms of an arithmetic sequence
variable (S : ℕ → ℕ)

-- Given conditions
axiom S_2_eq_3 : S 2 = 3
axiom S_4_eq_15 : S 4 = 15

-- Theorem statement
theorem find_S6 : S 6 = 36 := sorry

end find_S6_l290_290270


namespace divisor_is_3_l290_290168

theorem divisor_is_3 (divisor quotient remainder : ℕ) (h_dividend : 22 = (divisor * quotient) + remainder) 
  (h_quotient : quotient = 7) (h_remainder : remainder = 1) : divisor = 3 :=
by
  sorry

end divisor_is_3_l290_290168


namespace ball_colors_l290_290970

theorem ball_colors (R G B : ℕ) (h1 : R + G + B = 15) (h2 : B = R + 1) (h3 : R = G) (h4 : B = G + 5) : false :=
by
  sorry

end ball_colors_l290_290970


namespace inequality_transformation_l290_290578

theorem inequality_transformation (a b : ℝ) (h : a > b) : -3 * a < -3 * b := by
  sorry

end inequality_transformation_l290_290578


namespace even_number_of_divisors_less_than_100_l290_290396

theorem even_number_of_divisors_less_than_100 :
  ∃ (count : ℕ), count = 90 ∧
  (∀ n : ℕ, 1 ≤ n ∧ n < 100 →
    (∃ (d : ℕ), d * d = n ∨ (number_of_divisors n % 2 = 0)) :=
begin
  -- the proof goes here
  sorry
end

end even_number_of_divisors_less_than_100_l290_290396


namespace intercepts_equal_lines_parallel_l290_290139

-- Definition of the conditions: line equations
def line_l (a : ℝ) : Prop := ∀ x y : ℝ, a * x + 3 * y + 1 = 0

-- Problem (1) : The intercepts of the line on the two coordinate axes are equal
theorem intercepts_equal (a : ℝ) (h : line_l a) : a = 3 := by
  sorry

-- Problem (2): The line is parallel to x + (a-2)y + a = 0
theorem lines_parallel (a : ℝ) (h : line_l a) : (∀ x y : ℝ, x + (a-2) * y + a = 0) → a = 3 := by
  sorry

end intercepts_equal_lines_parallel_l290_290139


namespace final_ranking_l290_290932

-- Define data types for participants and their initial positions
inductive Participant
| X
| Y
| Z

open Participant

-- Define the initial conditions and number of position changes
def initial_positions : List Participant := [X, Y, Z]

def position_changes : Participant → Nat
| X => 5
| Y => 0  -- Not given explicitly but derived from the conditions.
| Z => 6

-- Final condition stating Y finishes before X
def Y_before_X : Prop := True

-- The theorem stating the final ranking
theorem final_ranking :
  Y_before_X →
  (initial_positions = [X, Y, Z]) →
  (position_changes X = 5) →
  (position_changes Z = 6) →
  (position_changes Y = 0) →
  [Y, X, Z] = [Y, X, Z] :=
by
  intros
  exact rfl

end final_ranking_l290_290932


namespace sum_distances_from_point_to_faces_constant_l290_290319

theorem sum_distances_from_point_to_faces_constant 
  (T : Tetrahedron) 
  (P : Point) 
  (V : ℝ)
  (S : ℝ)
  (h1 h2 h3 h4 : ℝ)
  (h_distances : h1 + h2 + h3 + h4 = (3 * V) / S) :
  h1 + h2 + h3 + h4 = (3 * V) / S := 
sorry

end sum_distances_from_point_to_faces_constant_l290_290319


namespace t_plus_inv_t_eq_three_l290_290909

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l290_290909


namespace probability_red_buttons_l290_290018

/-- 
Initial condition: Jar A contains 6 red buttons and 10 blue buttons.
Carla removes the same number of red buttons as blue buttons from Jar A and places them in Jar B.
Jar A's state after action: Jar A retains 3/4 of its original number of buttons.
Question: What is the probability that both selected buttons are red? Express your answer as a common fraction.
-/
theorem probability_red_buttons :
  let initial_red_a := 6
  let initial_blue_a := 10
  let total_buttons_a := initial_red_a + initial_blue_a
  
  -- Jar A after removing buttons
  let retained_fraction := 3 / 4
  let remaining_buttons_a := retained_fraction * total_buttons_a
  let removed_buttons := total_buttons_a - remaining_buttons_a
  let removed_red_buttons := removed_buttons / 2
  let removed_blue_buttons := removed_buttons / 2
  
  -- Remaining red and blue buttons in Jar A
  let remaining_red_a := initial_red_a - removed_red_buttons
  let remaining_blue_a := initial_blue_a - removed_blue_buttons

  -- Total remaining buttons in Jar A
  let total_remaining_a := remaining_red_a + remaining_blue_a

  -- Jar B contains the removed buttons
  let total_buttons_b := removed_buttons
  
  -- Probability calculations
  let probability_red_a := remaining_red_a / total_remaining_a
  let probability_red_b := removed_red_buttons / total_buttons_b

  -- Combined probability of selecting red button from both jars
  probability_red_a * probability_red_b = 1 / 6 :=
by
  sorry

end probability_red_buttons_l290_290018


namespace future_tech_high_absentee_percentage_l290_290668

theorem future_tech_high_absentee_percentage :
  let total_students := 180
  let boys := 100
  let girls := 80
  let absent_boys_fraction := 1 / 5
  let absent_girls_fraction := 1 / 4
  let absent_boys := absent_boys_fraction * boys
  let absent_girls := absent_girls_fraction * girls
  let total_absent_students := absent_boys + absent_girls
  let absent_percentage := (total_absent_students / total_students) * 100
  (absent_percentage = 22.22) := 
by
  sorry

end future_tech_high_absentee_percentage_l290_290668


namespace binom_30_3_eq_4060_l290_290877

theorem binom_30_3_eq_4060 : nat.choose 30 3 = 4060 := 
by sorry

end binom_30_3_eq_4060_l290_290877


namespace MrSmithEnglishProof_l290_290948

def MrSmithLearningEnglish : Prop :=
  (∃ (decade: String) (age: String), 
    (decade = "1950's" ∧ age = "in his sixties") ∨ 
    (decade = "1950" ∧ age = "in the sixties") ∨ 
    (decade = "1950's" ∧ age = "over sixty"))
  
def correctAnswer : Prop :=
  MrSmithLearningEnglish →
  (∃ answer, answer = "D")

theorem MrSmithEnglishProof : correctAnswer :=
  sorry

end MrSmithEnglishProof_l290_290948


namespace dice_sum_ways_l290_290200

theorem dice_sum_ways : 
  let num_dice : ℕ := 8
  let target_sum : ℕ := 20
  let dice_max_value : ℕ := 6
  let adjusted_sum : ℕ := target_sum - num_dice
  ∑ (ways : ℕ) in (Finset.filter (fun (a : Fin num_dice → ℕ) => 
    (∑ i, a i = adjusted_sum) ∧ (∀ i, 0 ≤ a i ∧ a i ≤ dice_max_value - 1)) 
    (Finset.mk (Finsupp.support := num_dice) 1 adjusted_sum)), 
  ways = 50388 := 
by 
  sorry

end dice_sum_ways_l290_290200


namespace hours_buses_leave_each_day_l290_290526

theorem hours_buses_leave_each_day
  (num_buses : ℕ)
  (num_days : ℕ)
  (buses_per_half_hour : ℕ)
  (h1 : num_buses = 120)
  (h2 : num_days = 5)
  (h3 : buses_per_half_hour = 2) :
  (num_buses / num_days) / buses_per_half_hour = 12 :=
by
  sorry

end hours_buses_leave_each_day_l290_290526


namespace ratio_a3_a2_l290_290407

open BigOperators

def binomial_expansion (x : ℝ) : ℝ :=
  (1 - 3 * x) ^ 6

def a (k : ℕ) : ℝ :=
  ∑ r in Finset.range 7, if r = k then ↑(Nat.choose 6 r) * (-3) ^ r else 0

theorem ratio_a3_a2 : a 3 / a 2 = -4 :=
by
  sorry

end ratio_a3_a2_l290_290407


namespace find_percentage_l290_290211

noncomputable def percentage (X : ℝ) : ℝ := (377.8020134228188 * 100 * 5.96) / 1265

theorem find_percentage : percentage 178 = 178 := by
  -- Conditions
  let P : ℝ := 178
  let A : ℝ := 1265
  let divisor : ℝ := 5.96
  let result : ℝ := 377.8020134228188

  -- Define the percentage calculation
  let X := (result * 100 * divisor) / A

  -- Verify the calculation matches
  have h : X = P := by sorry

  trivial

end find_percentage_l290_290211


namespace Vasya_numbers_l290_290835

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290835


namespace seq_geq_4_l290_290451

def seq (a : ℕ → ℝ) : Prop :=
  a 1 = 5 ∧ ∀ n : ℕ, n ≥ 1 → a (n + 1) = (a n ^ 2 + 8 * a n + 16) / (4 * a n)

theorem seq_geq_4 (a : ℕ → ℝ) (h : seq a) : ∀ n : ℕ, n ≥ 1 → a n ≥ 4 :=
sorry

end seq_geq_4_l290_290451


namespace cost_to_fill_pool_l290_290782

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l290_290782


namespace tangent_line_of_circle_l290_290271

theorem tangent_line_of_circle (x y : ℝ)
    (C_def : (x - 2)^2 + (y - 3)^2 = 25)
    (P : (ℝ × ℝ)) (P_def : P = (-1, 7)) :
    (3 * x - 4 * y + 31 = 0) :=
sorry

end tangent_line_of_circle_l290_290271


namespace carpet_breadth_l290_290097

theorem carpet_breadth
  (b : ℝ)
  (h1 : ∀ b, ∃ l, l = 1.44 * b)
  (h2 : 4082.4 = 45 * ((1.40 * l) * (1.25 * b)))
  : b = 6.08 :=
by
  sorry

end carpet_breadth_l290_290097


namespace evaluate_g_expression_l290_290105

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g_expression :
  3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_expression_l290_290105


namespace solution_set_of_inequality_l290_290379

noncomputable def f (x : ℝ) : ℝ := 4^x - 2^(x + 1) - 3

theorem solution_set_of_inequality :
  { x : ℝ | f x < 0 } = { x : ℝ | x < Real.log 3 / Real.log 2 } :=
by
  sorry

end solution_set_of_inequality_l290_290379


namespace function_inequality_l290_290651

noncomputable def f (x : ℝ) : ℝ := sorry

theorem function_inequality (f : ℝ → ℝ) (h1 : ∀ x : ℝ, x ≥ 1 → f x ≤ x)
  (h2 : ∀ x : ℝ, x ≥ 1 → f (2 * x) / Real.sqrt 2 ≤ f x) :
  ∀ x ≥ 1, f x < Real.sqrt (2 * x) :=
sorry

end function_inequality_l290_290651


namespace length_of_second_dimension_l290_290517

def volume_of_box (w : ℝ) : ℝ :=
  (w - 16) * (46 - 16) * 8

theorem length_of_second_dimension (w : ℝ) (h_volume : volume_of_box w = 4800) : w = 36 :=
by
  sorry

end length_of_second_dimension_l290_290517


namespace increasing_function_on_interval_l290_290135

section
  variable (a b : ℝ)
  def f (x : ℝ) : ℝ := |x^2 - 2*a*x + b|

  theorem increasing_function_on_interval (h : a^2 - b ≤ 0) :
    ∀ x y : ℝ, a ≤ x → x ≤ y → f x ≤ f y := 
  sorry
end

end increasing_function_on_interval_l290_290135


namespace images_per_memory_card_l290_290602

-- Define the constants based on the conditions given in the problem
def daily_pictures : ℕ := 10
def years : ℕ := 3
def days_per_year : ℕ := 365
def cost_per_card : ℕ := 60
def total_spent : ℕ := 13140

-- Define the properties to be proved
theorem images_per_memory_card :
  (years * days_per_year * daily_pictures) / (total_spent / cost_per_card) = 50 :=
by
  sorry

end images_per_memory_card_l290_290602


namespace counts_of_arson_l290_290736

-- Define variables A (arson), B (burglary), L (petty larceny)
variables (A B L : ℕ)

-- Conditions given in the problem
def burglary_charges : Prop := B = 2
def petty_larceny_charges_relation : Prop := L = 6 * B
def total_sentence_calculation : Prop := 36 * A + 18 * B + 6 * L = 216

-- Prove that given these conditions, the counts of arson (A) is 3
theorem counts_of_arson (h1 : burglary_charges B)
                        (h2 : petty_larceny_charges_relation B L)
                        (h3 : total_sentence_calculation A B L) :
                        A = 3 :=
sorry

end counts_of_arson_l290_290736


namespace cost_to_fill_pool_l290_290784

-- Define the given conditions as constants
def filling_time : ℝ := 50
def flow_rate : ℝ := 100
def cost_per_10_gallons : ℝ := 0.01

-- Calculate total volume in gallons
def total_volume : ℝ := filling_time * flow_rate

-- Calculate the cost per gallon in dollars
def cost_per_gallon : ℝ := cost_per_10_gallons / 10

-- Define the total cost to fill the pool in dollars
def total_cost : ℝ := total_volume * cost_per_gallon

-- Prove that the total cost equals $5
theorem cost_to_fill_pool : total_cost = 5 := by
  unfold total_cost
  unfold total_volume
  unfold cost_per_gallon
  unfold filling_time
  unfold flow_rate
  unfold cost_per_10_gallons
  sorry

end cost_to_fill_pool_l290_290784


namespace f_one_zero_x_range_l290_290671

noncomputable def f : ℝ → ℝ := sorry

-- Conditions
-- f is defined for x > 0
variable (f : ℝ → ℝ)
variables (h_domain : ∀ x, x > 0 → ∃ y, f x = y)
variables (h1 : f 2 = 1)
variables (h2 : ∀ x y, x > 0 → y > 0 → f (x * y) = f x + f y)
variables (h3 : ∀ x y, x > y → f x > f y)

-- Question 1
theorem f_one_zero (hf1 : f 1 = 0) : True := 
  by trivial
  
-- Question 2
theorem x_range (x: ℝ) (hx: f 3 + f (4 - 8 * x) > 2) : x ≤ 1/3 := sorry

end f_one_zero_x_range_l290_290671


namespace number_of_distinct_lines_l290_290764

theorem number_of_distinct_lines (S : Finset ℕ) (h : S = {1, 2, 3, 4, 5}) :
  (S.card.choose 2) - 2 = 18 :=
by
  -- Conditions
  have hS : S = {1, 2, 3, 4, 5} := h
  -- Conclusion
  sorry

end number_of_distinct_lines_l290_290764


namespace phillip_remaining_money_l290_290457

def initial_money : ℝ := 95
def cost_oranges : ℝ := 14
def cost_apples : ℝ := 25
def cost_candy : ℝ := 6
def cost_eggs : ℝ := 12
def cost_milk : ℝ := 8
def discount_apples_rate : ℝ := 0.15
def discount_milk_rate : ℝ := 0.10

def discounted_cost_apples : ℝ := cost_apples * (1 - discount_apples_rate)
def discounted_cost_milk : ℝ := cost_milk * (1 - discount_milk_rate)

def total_spent : ℝ := cost_oranges + discounted_cost_apples + cost_candy + cost_eggs + discounted_cost_milk

def remaining_money : ℝ := initial_money - total_spent

theorem phillip_remaining_money : remaining_money = 34.55 := by
  -- Proof here
  sorry

end phillip_remaining_money_l290_290457


namespace remainder_101_pow_50_mod_100_l290_290197

theorem remainder_101_pow_50_mod_100 : (101 ^ 50) % 100 = 1 := by
  sorry

end remainder_101_pow_50_mod_100_l290_290197


namespace egg_price_l290_290173

theorem egg_price (num_eggs capital_remaining : ℕ) (total_cost price_per_egg : ℝ)
  (h1 : num_eggs = 30)
  (h2 : capital_remaining = 5)
  (h3 : total_cost = 5)
  (h4 : num_eggs - capital_remaining = 25)
  (h5 : 25 * price_per_egg = total_cost) :
  price_per_egg = 0.20 := sorry

end egg_price_l290_290173


namespace kerosene_sale_difference_l290_290432

noncomputable def rice_price : ℝ := 0.33
noncomputable def price_of_dozen_eggs := rice_price
noncomputable def price_of_one_egg := rice_price / 12
noncomputable def price_of_half_liter_kerosene := 4 * price_of_one_egg
noncomputable def price_of_one_liter_kerosene := 2 * price_of_half_liter_kerosene
noncomputable def kerosene_discounted := price_of_one_liter_kerosene * 0.95
noncomputable def kerosene_diff_cents := (price_of_one_liter_kerosene - kerosene_discounted) * 100

theorem kerosene_sale_difference :
  kerosene_diff_cents = 1.1 := by sorry

end kerosene_sale_difference_l290_290432


namespace green_chips_correct_l290_290740

-- Definitions
def total_chips : ℕ := 120
def blue_chips : ℕ := total_chips / 4
def red_chips : ℕ := total_chips * 20 / 100
def yellow_chips : ℕ := total_chips / 10
def non_green_chips : ℕ := blue_chips + red_chips + yellow_chips
def green_chips : ℕ := total_chips - non_green_chips

-- Statement to prove
theorem green_chips_correct : green_chips = 54 := by
  -- Proof would go here
  sorry

end green_chips_correct_l290_290740


namespace problem_prove_ω_and_delta_l290_290391

noncomputable def f (ω φ x : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem problem_prove_ω_and_delta (ω φ : ℝ) (h_ω : ω > 0) (h_φ : abs φ < π / 2) 
    (h_sym_axis : ∀ x, f ω φ x = f ω φ (-(x + π))) 
    (h_center_sym : ∃ c : ℝ, (c = π / 2) ∧ (f ω φ c = 0)) 
    (h_monotone_increasing : ∀ x, -π ≤ x ∧ x ≤ -π / 2 → f ω φ x < f ω φ (x + 1)) :
    (ω = 1 / 3) ∧ (∀ δ : ℝ, (∀ x : ℝ, f ω φ (x + δ) = f ω φ (-x + δ)) → ∃ k : ℤ, δ = 2 * π + 3 * k * π) :=
by
  sorry

end problem_prove_ω_and_delta_l290_290391


namespace probability_non_adjacent_zeros_l290_290573

theorem probability_non_adjacent_zeros (total_ones total_zeros : ℕ) (h₁ : total_ones = 3) (h₂ : total_zeros = 2) : 
  (total_zeros != 0 ∧ total_ones != 0 ∧ total_zeros + total_ones = 5) → 
  (prob_non_adjacent (total_ones + total_zeros) total_zeros = 0.6) :=
by
  sorry

def prob_non_adjacent (total num_zeros: ℕ) : ℚ :=
  let total_arrangements := (Nat.factorial total) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros)))
  let adjacent_arrangements := (Nat.factorial (total - num_zeros + 1)) / ((Nat.factorial num_zeros) * (Nat.factorial (total - num_zeros - 1)))
  let non_adjacent_arrangements := total_arrangements - adjacent_arrangements
  non_adjacent_arrangements / total_arrangements

end probability_non_adjacent_zeros_l290_290573


namespace arithmetic_sequence_ninth_term_l290_290044

theorem arithmetic_sequence_ninth_term (a d : ℤ)
  (h1 : a + 2 * d = 5)
  (h2 : a + 5 * d = 11) :
  a + 8 * d = 17 := by
  sorry

end arithmetic_sequence_ninth_term_l290_290044


namespace isosceles_triangle_perimeter_l290_290703

theorem isosceles_triangle_perimeter (a b : ℝ) (h₁ : 2 * a - 3 * b + 5 = 0) (h₂ : 2 * a + 3 * b - 13 = 0) :
  ∃ p : ℝ, p = 7 ∨ p = 8 :=
sorry

end isosceles_triangle_perimeter_l290_290703


namespace intersection_always_exists_minimum_chord_length_and_equation_l290_290911

open Real

noncomputable def circle_eq (x y : ℝ) : Prop :=
  4 * x^2 + 4 * y^2 - 4 * x - 8 * y - 11 = 0

noncomputable def line_eq (m x y : ℝ) : Prop :=
  (m - 1) * x + m * y = m + 1

theorem intersection_always_exists :
  ∀ (m : ℝ), ∃ (x y : ℝ), circle_eq x y ∧ line_eq m x y :=
by
  sorry

theorem minimum_chord_length_and_equation :
  ∃ (k : ℝ) (x y : ℝ), k = sqrt 3 ∧ (3 * x - 2 * y + 7 = 0) ∧
    ∀ m, ∃ (xp yp : ℝ), line_eq m xp yp ∧ ∃ (l1 l2 : ℝ), line_eq m l1 l2 ∧ 
    (circle_eq xp yp ∧ circle_eq l1 l2)  :=
by
  sorry

end intersection_always_exists_minimum_chord_length_and_equation_l290_290911


namespace alternating_students_count_l290_290644

theorem alternating_students_count :
  let num_male := 4
  let num_female := 5
  let arrangements := Nat.factorial num_female * Nat.factorial num_male
  arrangements = 2880 :=
by
  sorry

end alternating_students_count_l290_290644


namespace vasya_numbers_l290_290827

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290827


namespace subtract_500_from_sum_of_calculations_l290_290571

theorem subtract_500_from_sum_of_calculations (x : ℕ) (h : 423 - x = 421) : 
  (421 + 423 * x) - 500 = 767 := 
by
  sorry

end subtract_500_from_sum_of_calculations_l290_290571


namespace board_numbers_l290_290598

theorem board_numbers (a b c : ℕ) (h1 : a = 3) (h2 : b = 9) (h3 : c = 15)
    (op : ∀ x y z : ℕ, (x = y + z - t) → true)  -- simplifying the operation representation
    (min_number : ∃ x, x = 2013) : ∃ n m, n = 2019 ∧ m = 2025 := 
sorry

end board_numbers_l290_290598


namespace cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l290_290000

theorem cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle
  (surface_area : ℝ) (lateral_surface_unfolds_to_semicircle : Prop) :
  surface_area = 12 * Real.pi → lateral_surface_unfolds_to_semicircle → ∃ r : ℝ, r = 2 := by
  sorry

end cone_radius_of_surface_area_and_lateral_surface_unfolds_to_semicircle_l290_290000


namespace new_average_rent_l290_290623

theorem new_average_rent 
  (n : ℕ) (h_n : n = 4) 
  (avg_old : ℝ) (h_avg_old : avg_old = 800) 
  (inc_rate : ℝ) (h_inc_rate : inc_rate = 0.16) 
  (old_rent : ℝ) (h_old_rent : old_rent = 1250) 
  (new_rent : ℝ) (h_new_rent : new_rent = old_rent * (1 + inc_rate)) 
  (total_rent_old : ℝ) (h_total_rent_old : total_rent_old = n * avg_old)
  (total_rent_new : ℝ) (h_total_rent_new : total_rent_new = total_rent_old - old_rent + new_rent)
  (avg_new : ℝ) (h_avg_new : avg_new = total_rent_new / n) : 
  avg_new = 850 := 
sorry

end new_average_rent_l290_290623


namespace simple_interest_rate_l290_290492

theorem simple_interest_rate
  (SI : ℝ) (P : ℝ) (T : ℝ) (R : ℝ)
  (h1 : SI = 400)
  (h2 : P = 800)
  (h3 : T = 2) :
  R = 25 :=
by
  sorry

end simple_interest_rate_l290_290492


namespace five_dice_not_all_same_probability_l290_290057

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l290_290057


namespace largest_lcm_among_pairs_is_45_l290_290491

theorem largest_lcm_among_pairs_is_45 :
  max (max (max (max (max (Nat.lcm 15 3) (Nat.lcm 15 5)) (Nat.lcm 15 6)) (Nat.lcm 15 9)) (Nat.lcm 15 10)) (Nat.lcm 15 15) = 45 :=
by
  sorry

end largest_lcm_among_pairs_is_45_l290_290491


namespace square_side_length_properties_l290_290700

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end square_side_length_properties_l290_290700


namespace cat_toy_cost_correct_l290_290737

-- Define the initial amount of money Jessica had.
def initial_amount : ℝ := 11.73

-- Define the amount left after spending.
def amount_left : ℝ := 1.51

-- Define the cost of the cat toy.
def toy_cost : ℝ := initial_amount - amount_left

-- Theorem and statement to prove the cost of the cat toy.
theorem cat_toy_cost_correct : toy_cost = 10.22 := sorry

end cat_toy_cost_correct_l290_290737


namespace non_zero_real_x_solution_l290_290109

noncomputable section

variables {x : ℝ} (hx : x ≠ 0)

theorem non_zero_real_x_solution 
  (h : (3 * x)^5 = (9 * x)^4) : 
  x = 27 := by
  sorry

end non_zero_real_x_solution_l290_290109


namespace smallest_slice_area_l290_290954

theorem smallest_slice_area
  (a₁ : ℕ) (d : ℕ) (total_angle : ℕ) (r : ℕ) 
  (h₁ : a₁ = 30) (h₂ : d = 2) (h₃ : total_angle = 360) (h₄ : r = 10) :
  ∃ (n : ℕ) (smallest_angle : ℕ),
  n = 9 ∧ smallest_angle = 18 ∧ 
  ∃ (area : ℝ), area = 5 * Real.pi :=
by
  sorry


end smallest_slice_area_l290_290954


namespace average_age_combined_l290_290622

theorem average_age_combined (n1 n2 : ℕ) (avg1 avg2 : ℕ) 
  (h1 : n1 = 45) (h2 : n2 = 60) (h3 : avg1 = 12) (h4 : avg2 = 40) :
  (n1 * avg1 + n2 * avg2) / (n1 + n2) = 28 :=
by
  sorry

end average_age_combined_l290_290622


namespace compare_neg_fractions_l290_290670

theorem compare_neg_fractions : - (2 / 3 : ℝ) > - (5 / 7 : ℝ) := 
by 
  sorry

end compare_neg_fractions_l290_290670


namespace number_of_valid_m_values_l290_290633

noncomputable def polynomial (m : ℤ) (x : ℤ) : ℤ := 
  2 * (m - 1) * x ^ 2 - (m ^ 2 - m + 12) * x + 6 * m

noncomputable def discriminant (m : ℤ) : ℤ :=
  (m ^ 2 - m + 12) ^ 2 - 4 * 2 * (m - 1) * 6 * m

def is_perfect_square (n : ℤ) : Prop :=
  ∃ (k : ℤ), k * k = n

def has_integral_roots (m : ℤ) : Prop :=
  ∃ (r1 r2 : ℤ), polynomial m r1 = 0 ∧ polynomial m r2 = 0

def valid_m_values (m : ℤ) : Prop :=
  (discriminant m) > 0 ∧ is_perfect_square (discriminant m) ∧ has_integral_roots m

theorem number_of_valid_m_values : 
  (∃ M : List ℤ, (∀ m ∈ M, valid_m_values m) ∧ M.length = 4) :=
  sorry

end number_of_valid_m_values_l290_290633


namespace not_54_after_one_hour_l290_290963

theorem not_54_after_one_hour (n : ℕ) (initial_number : ℕ) (initial_factors : ℕ × ℕ)
  (h₀ : initial_number = 12)
  (h₁ : initial_factors = (2, 1)) :
  (∀ k : ℕ, k < 60 →
    ∀ current_factors : ℕ × ℕ,
    current_factors = (initial_factors.1 + k, initial_factors.2 + k) ∨
    current_factors = (initial_factors.1 - k, initial_factors.2 - k) →
    initial_number * (2 ^ (initial_factors.1 + k)) * (3 ^ (initial_factors.2 + k)) ≠ 54) :=
by
  sorry

end not_54_after_one_hour_l290_290963


namespace polynomial_value_at_2008_l290_290448

def f (a₀ a₁ a₂ a₃ a₄ : ℝ) (x : ℝ) : ℝ := a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4

theorem polynomial_value_at_2008 (a₀ a₁ a₂ a₃ a₄ : ℝ) (h₁ : a₄ ≠ 0)
  (h₀₃ : f a₀ a₁ a₂ a₃ a₄ 2003 = 24)
  (h₀₄ : f a₀ a₁ a₂ a₃ a₄ 2004 = -6)
  (h₀₅ : f a₀ a₁ a₂ a₃ a₄ 2005 = 4)
  (h₀₆ : f a₀ a₁ a₂ a₃ a₄ 2006 = -6)
  (h₀₇ : f a₀ a₁ a₂ a₃ a₄ 2007 = 24) :
  f a₀ a₁ a₂ a₃ a₄ 2008 = 274 :=
by sorry

end polynomial_value_at_2008_l290_290448


namespace probability_interval_l290_290132

noncomputable def normalDist := NormalDist.mk 1 1

theorem probability_interval :
  let ξ : ℝ → ℝ := fun x => normalDist.pdf x in
  ∫ x in Icc (-1) 3, ξ x = 0.954 :=
by
  -- Given conditions
  have hξ₃ : ∫ x in Iic 3, ξ x = 0.977 := sorry,
  -- Proof of the statement
  sorry

end probability_interval_l290_290132


namespace arithmetic_sequence_positive_l290_290314

theorem arithmetic_sequence_positive (d a_1 : ℤ) (n : ℤ) :
  (a_11 - a_8 = 3) -> 
  (S_11 - S_8 = 33) ->
  (n > 0) ->
  a_1 + (n-1) * d > 0 ->
  n = 10 :=
by
  sorry

end arithmetic_sequence_positive_l290_290314


namespace hall_length_width_difference_l290_290045

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L) 
  (h2 : L * W = 128) : 
  L - W = 8 :=
by
  sorry

end hall_length_width_difference_l290_290045


namespace train_travel_distance_l290_290648

theorem train_travel_distance
  (rate_miles_per_pound : Real := 5 / 2)
  (remaining_coal : Real := 160)
  (distance_per_pound := λ r, r / 2)
  (total_distance := λ rc dpp, rc * dpp) :
  total_distance remaining_coal rate_miles_per_pound = 400 := sorry

end train_travel_distance_l290_290648


namespace circular_pond_area_l290_290347

theorem circular_pond_area (AB CD : ℝ) (D_is_midpoint : Prop) (hAB : AB = 20) (hCD : CD = 12)
  (hD_midpoint : D_is_midpoint ∧ D_is_midpoint = (AB / 2 = 10)) :
  ∃ (A : ℝ), A = 244 * Real.pi :=
by
  sorry

end circular_pond_area_l290_290347


namespace carl_teaches_periods_l290_290364

theorem carl_teaches_periods (cards_per_student : ℕ) (students_per_class : ℕ) (pack_cost : ℕ) (amount_spent : ℕ) (cards_per_pack : ℕ) :
  cards_per_student = 10 →
  students_per_class = 30 →
  pack_cost = 3 →
  amount_spent = 108 →
  cards_per_pack = 50 →
  (amount_spent / pack_cost) * cards_per_pack / (cards_per_student * students_per_class) = 6 :=
by
  intros hc hs hp ha hpkg
  /- proof steps would go here -/
  sorry

end carl_teaches_periods_l290_290364


namespace fruit_juice_conversion_needed_l290_290085

theorem fruit_juice_conversion_needed
  (A_milk_parts B_milk_parts A_fruit_juice_parts B_fruit_juice_parts : ℕ)
  (y : ℕ)
  (x : ℕ)
  (convert_liters : ℕ)
  (A_juice_ratio_milk A_juice_ratio_fruit : ℚ)
  (B_juice_ratio_milk B_juice_ratio_fruit : ℚ) :
  (A_milk_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_milk →
  (A_fruit_juice_parts : ℚ) / (A_milk_parts + A_fruit_juice_parts) = A_juice_ratio_fruit →
  (B_milk_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_milk →
  (B_fruit_juice_parts : ℚ) / (B_milk_parts + B_fruit_juice_parts) = B_juice_ratio_fruit →
  (A_juice_ratio_milk * x = A_juice_ratio_fruit * x + y) →
  y = 14 →
  x = 98 :=
by sorry

end fruit_juice_conversion_needed_l290_290085


namespace vasya_numbers_l290_290818

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290818


namespace rhombus_area_l290_290696

theorem rhombus_area
  (d1 d2 : ℝ)
  (hd1 : d1 = 14)
  (hd2 : d2 = 20) :
  (d1 * d2) / 2 = 140 := by
  -- Problem: Given diagonals of length 14 cm and 20 cm,
  -- prove that the area of the rhombus is 140 square centimeters.
  sorry

end rhombus_area_l290_290696


namespace area_of_smaller_circle_l290_290789

noncomputable def radius_large_circle (x : ℝ) : ℝ := 2 * x
noncomputable def radius_small_circle (y : ℝ) : ℝ := y

theorem area_of_smaller_circle 
(pa ab : ℝ)
(r : ℝ)
(area : ℝ) 
(h1 : pa = 5) 
(h2 : ab = 5) 
(h3 : radius_large_circle r = 2 * radius_small_circle r)
(h4 : 2 * radius_small_circle r + radius_large_circle r = 10)
(h5 : area = Real.pi * (radius_small_circle r)^2) 
: area = 6.25 * Real.pi :=
by
  sorry

end area_of_smaller_circle_l290_290789


namespace find_m_if_parallel_l290_290393

-- Definitions of the lines and the condition for parallel lines
def line1 (m : ℝ) (x y : ℝ) : ℝ := (m - 1) * x + y + 2
def line2 (m : ℝ) (x y : ℝ) : ℝ := 8 * x + (m + 1) * y + (m - 1)

-- The condition for the lines to be parallel
def parallel (m : ℝ) : Prop :=
  (m - 1) / 8 = 1 / (m + 1) ∧ (m - 1) / 8 ≠ 2 / (m - 1)

-- The main theorem to prove
theorem find_m_if_parallel (m : ℝ) (h : parallel m) : m = 3 :=
sorry

end find_m_if_parallel_l290_290393


namespace johnny_earnings_l290_290939

theorem johnny_earnings :
  let job1 := 3 * 7
  let job2 := 2 * 10
  let job3 := 4 * 12
  let daily_earnings := job1 + job2 + job3
  let total_earnings := 5 * daily_earnings
  total_earnings = 445 :=
by
  sorry

end johnny_earnings_l290_290939


namespace Vasya_numbers_l290_290836

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290836


namespace sunzi_wood_problem_l290_290437

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l290_290437


namespace ant_travel_finite_path_exists_l290_290663

theorem ant_travel_finite_path_exists :
  ∃ (x y z t : ℝ), |x| < |y - z + t| ∧ |y| < |x - z + t| ∧ 
                   |z| < |x - y + t| ∧ |t| < |x - y + z| :=
by
  sorry

end ant_travel_finite_path_exists_l290_290663


namespace find_a_l290_290503

noncomputable def f (x : ℝ) : ℝ := 5^(abs x)

noncomputable def g (a x : ℝ) : ℝ := a*x^2 - x

theorem find_a (a : ℝ) (h : f (g a 1) = 1) : a = 1 := 
by
  sorry

end find_a_l290_290503


namespace octopus_dressing_orders_l290_290857

/-- A robotic octopus has four legs, and each leg needs to wear a glove before it can wear a boot.
    Additionally, it has two tentacles that require one bracelet each before putting anything on the legs.
    The total number of valid dressing orders is 1,286,400. -/
theorem octopus_dressing_orders : 
  ∃ (n : ℕ), n = 1286400 :=
by
  sorry

end octopus_dressing_orders_l290_290857


namespace correct_answer_is_B_l290_290982

def is_permutation_problem (desc : String) : Prop :=
  desc = "Permutation"

def check_problem_A : Prop :=
  ¬ is_permutation_problem "Selecting 2 out of 8 students to participate in a knowledge competition"

def check_problem_B : Prop :=
  is_permutation_problem "If 10 people write letters to each other once, how many letters are written in total"

def check_problem_C : Prop :=
  ¬ is_permutation_problem "There are 5 points on a plane, with no three points collinear, what is the maximum number of lines that can be determined by these 5 points"

def check_problem_D : Prop :=
  ¬ is_permutation_problem "From the numbers 1, 2, 3, 4, choose any two numbers to multiply, how many different results are there"

theorem correct_answer_is_B : check_problem_A ∧ check_problem_B ∧ check_problem_C ∧ check_problem_D → 
  ("B" = "B") := by
  sorry

end correct_answer_is_B_l290_290982


namespace dig_second_hole_l290_290501

theorem dig_second_hole (w1 h1 d1 w2 d2 : ℕ) (extra_workers : ℕ) (h2 : ℕ) :
  w1 = 45 ∧ h1 = 8 ∧ d1 = 30 ∧ extra_workers = 65 ∧
  w2 = w1 + extra_workers ∧ d2 = 55 →
  360 * d2 / d1 = w2 * h2 →
  h2 = 6 :=
by
  intros h cond
  sorry

end dig_second_hole_l290_290501


namespace find_k_l290_290883

theorem find_k (k : ℝ) : 
  let a := 6
  let b := 25
  let root := (-25 - Real.sqrt 369) / 12
  6 * root^2 + 25 * root + k = 0 → k = 32 / 3 :=
sorry

end find_k_l290_290883


namespace find_principal_amount_l290_290073

theorem find_principal_amount 
  (P₁ : ℝ) (r₁ t₁ : ℝ) (S₁ : ℝ)
  (P₂ : ℝ) (r₂ t₂ : ℝ) (C₂ : ℝ) :
  S₁ = (P₁ * r₁ * t₁) / 100 →
  C₂ = P₂ * ( (1 + r₂) ^ t₂ - 1) →
  S₁ = C₂ / 2 →
  P₁ = 2800 :=
by
  sorry

end find_principal_amount_l290_290073


namespace cousins_arrangement_l290_290454

def number_of_arrangements (cousins rooms : ℕ) (min_empty_rooms : ℕ) : ℕ := sorry

theorem cousins_arrangement : number_of_arrangements 5 4 1 = 56 := 
by sorry

end cousins_arrangement_l290_290454


namespace travel_distance_l290_290999

-- Define the conditions
def distance_10_gallons := 300 -- 300 miles on 10 gallons of fuel
def gallons_10 := 10 -- 10 gallons

-- Given the distance per gallon, calculate the distance for 15 gallons
def distance_per_gallon := distance_10_gallons / gallons_10

def gallons_15 := 15 -- 15 gallons

def distance_15_gallons := distance_per_gallon * gallons_15

-- Proof statement
theorem travel_distance (d_10 : distance_10_gallons = 300)
                        (g_10 : gallons_10 = 10)
                        (g_15 : gallons_15 = 15) :
  distance_15_gallons = 450 :=
  by
  -- The actual proof goes here
  sorry

end travel_distance_l290_290999


namespace division_of_fractions_l290_290278

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l290_290278


namespace maximum_value_of_3m_4n_l290_290967

noncomputable def max_value (m n : ℕ) : ℕ :=
  3 * m + 4 * n

theorem maximum_value_of_3m_4n 
  (m n : ℕ) 
  (h_even : ∀ i, i < m → (2 * (i + 1)) > 0) 
  (h_odd : ∀ j, j < n → (2 * j + 1) > 0)
  (h_sum : m * (m + 1) + n^2 ≤ 1987) 
  (h_odd_n : n % 2 = 1) :
  max_value m n ≤ 221 := 
sorry

end maximum_value_of_3m_4n_l290_290967


namespace kyungsoo_came_second_l290_290020

theorem kyungsoo_came_second
  (kyungsoo_jump : ℝ) (younghee_jump : ℝ) (jinju_jump : ℝ) (chanho_jump : ℝ)
  (h_kyungsoo : kyungsoo_jump = 2.3)
  (h_younghee : younghee_jump = 0.9)
  (h_jinju : jinju_jump = 1.8)
  (h_chanho : chanho_jump = 2.5) :
  kyungsoo_jump = 2.3 := 
by
  sorry

end kyungsoo_came_second_l290_290020


namespace part1_part2_l290_290275

noncomputable def U : Set ℝ := Set.univ

noncomputable def A (a: ℝ) : Set ℝ := {x | (x - 2) / (x - (3 * a + 1)) < 0}
noncomputable def B (a: ℝ) : Set ℝ := {x | (x - a^2 - 2) / (x - a) < 0}

theorem part1 (a : ℝ) (ha : a = 1/2) :
  (U \ (B a)) ∩ (A a) = {x | 9/4 ≤ x ∧ x < 5/2} :=
sorry

theorem part2 (p q : ℝ → Prop)
  (hp : ∀ x, p x → x ∈ A a) (hq : ∀ x, q x → x ∈ B a)
  (hq_necessary : ∀ x, p x → q x) :
  -1/2 ≤ a ∧ a ≤ (3 - Real.sqrt 5) / 2 :=
sorry

end part1_part2_l290_290275


namespace total_yellow_balloons_l290_290787

theorem total_yellow_balloons (n_tom : ℕ) (n_sara : ℕ) (h_tom : n_tom = 9) (h_sara : n_sara = 8) : n_tom + n_sara = 17 :=
by
  sorry

end total_yellow_balloons_l290_290787


namespace slowest_bailing_rate_proof_l290_290983

def distance : ℝ := 1.5 -- in miles
def rowing_speed : ℝ := 3 -- in miles per hour
def water_intake_rate : ℝ := 8 -- in gallons per minute
def sink_threshold : ℝ := 50 -- in gallons

noncomputable def solve_bailing_rate_proof : ℝ :=
  let time_to_shore_hours : ℝ := distance / rowing_speed
  let time_to_shore_minutes : ℝ := time_to_shore_hours * 60
  let total_water_intake : ℝ := water_intake_rate * time_to_shore_minutes
  let excess_water : ℝ := total_water_intake - sink_threshold
  let bailing_rate_needed : ℝ := excess_water / time_to_shore_minutes
  bailing_rate_needed

theorem slowest_bailing_rate_proof : solve_bailing_rate_proof ≤ 7 :=
  by
    sorry

end slowest_bailing_rate_proof_l290_290983


namespace square_root_of_25_squared_l290_290199

theorem square_root_of_25_squared :
  Real.sqrt (25 ^ 2) = 25 :=
sorry

end square_root_of_25_squared_l290_290199


namespace bianca_points_per_bag_l290_290101

theorem bianca_points_per_bag (total_bags : ℕ) (not_recycled : ℕ) (total_points : ℕ) 
  (h1 : total_bags = 17) 
  (h2 : not_recycled = 8) 
  (h3 : total_points = 45) : 
  total_points / (total_bags - not_recycled) = 5 :=
by
  sorry 

end bianca_points_per_bag_l290_290101


namespace prob_blue_lower_than_yellow_l290_290976

noncomputable def prob_bin_k (k : ℕ) : ℝ :=
  3^(-k : ℤ)

noncomputable def prob_same_bin : ℝ :=
  ∑' k, 3^(-2*k : ℤ)

theorem prob_blue_lower_than_yellow :
  (1 - prob_same_bin) / 2 = 7 / 16 :=
by
  -- proof goes here
  sorry

end prob_blue_lower_than_yellow_l290_290976


namespace people_in_each_column_l290_290299

theorem people_in_each_column
  (P : ℕ)
  (x : ℕ)
  (h1 : P = 16 * x)
  (h2 : P = 12 * 40) :
  x = 30 :=
sorry

end people_in_each_column_l290_290299


namespace vasya_numbers_l290_290799

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290799


namespace general_term_formula_minimum_sum_value_l290_290383

variable {a : ℕ → ℚ} -- The arithmetic sequence
variable {S : ℕ → ℚ} -- Sum of the first n terms of the sequence

-- Conditions
axiom a_seq_cond1 : a 2 + a 6 = 6
axiom S_sum_cond5 : S 5 = 35 / 3

-- Definitions
def a_n (n : ℕ) : ℚ := (2 / 3) * n + 1 / 3
def S_n (n : ℕ) : ℚ := (1 / 3) * (n^2 + 2 * n)

-- Hypotheses
axiom seq_def : ∀ n, a n = a_n n
axiom sum_def : ∀ n, S n = S_n n

-- Theorems to be proved
theorem general_term_formula : ∀ n, a n = (2 / 3 * n) + 1 / 3 := by sorry
theorem minimum_sum_value : ∀ n, S 1 ≤ S n := by sorry

end general_term_formula_minimum_sum_value_l290_290383


namespace unique_rectangle_exists_l290_290119

theorem unique_rectangle_exists (a b : ℝ) (h_ab : a < b) :
  ∃! (x y : ℝ), x < b ∧ y < b ∧ x + y = (a + b) / 2 ∧ x * y = (a * b) / 4 :=
by
  sorry

end unique_rectangle_exists_l290_290119


namespace amount_r_has_l290_290340

variable (p q r : ℕ)
variable (total_amount : ℕ)
variable (two_thirdsOf_pq : ℕ)

def total_money : Prop := (p + q + r = 4000)
def two_thirds_of_pq : Prop := (r = 2 * (p + q) / 3)

theorem amount_r_has : total_money p q r → two_thirds_of_pq p q r → r = 1600 := by
  intro h1 h2
  sorry

end amount_r_has_l290_290340


namespace pow_add_div_eq_l290_290538

   theorem pow_add_div_eq (a b c d e : ℕ) (h1 : b = 2) (h2 : c = 345) (h3 : d = 9) (h4 : e = 8 - 5) : 
     a = b^c + d^e -> a = 2^345 + 729 := 
   by 
     intros 
     sorry
   
end pow_add_div_eq_l290_290538


namespace probability_not_all_same_l290_290053

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l290_290053


namespace largest_multiple_of_7_less_than_neg_30_l290_290195

theorem largest_multiple_of_7_less_than_neg_30 (m : ℤ) (h1 : m % 7 = 0) (h2 : m < -30) : m = -35 :=
sorry

end largest_multiple_of_7_less_than_neg_30_l290_290195


namespace city_grid_sinks_l290_290625

-- Define the main conditions of the grid city
def cell_side_meter : Int := 500
def max_travel_km : Int := 1

-- Total number of intersections in a 100x100 grid
def total_intersections : Int := (100 + 1) * (100 + 1)

-- Number of sinks that need to be proven
def required_sinks : Int := 1300

-- Lean theorem statement to prove that given the conditions,
-- there are at least 1300 sinks (intersections that act as sinks)
theorem city_grid_sinks :
  ∀ (city_grid : Matrix (Fin 101) (Fin 101) IntersectionType),
  (∀ i j, i < 100 → j < 100 → cell_side_meter ≤ max_travel_km * 1000) →
  ∃ (sinks : Finset (Fin 101 × Fin 101)), 
  (sinks.card ≥ required_sinks) := sorry

end city_grid_sinks_l290_290625


namespace range_of_k_l290_290422

theorem range_of_k (k : ℝ) :
  (∃ x : ℝ, (k - 1) * x = 4 ∧ x < 2) → (k < 1 ∨ k > 3) := 
by 
  sorry

end range_of_k_l290_290422


namespace find_y_in_exponent_equation_l290_290885

theorem find_y_in_exponent_equation :
  ∃ y : ℕ, 8^3 + 8^3 + 8^3 + 8^3 = 2^y ∧ y = 11 :=
begin
  use 11,
  split,
  { have h1 : 8 = 2^3 := by norm_num,
    have h2 : 8^3 = (2^3)^3 := by congr,
    have h3 : (2^3)^3 = 2^(3 * 3) := by rw [←pow_mul],
    rw [h2, h3, pow_mul],
    norm_num,
  },
  { refl },
end

end find_y_in_exponent_equation_l290_290885


namespace sin_alpha_value_l290_290257

-- Given conditions
variables (α : ℝ) (h1 : Real.tan α = -5 / 12) (h2 : π / 2 < α ∧ α < π)

-- Assertion to prove
theorem sin_alpha_value : Real.sin α = 5 / 13 :=
by
  -- Proof goes here
  sorry

end sin_alpha_value_l290_290257


namespace coterminal_angle_l290_290677

theorem coterminal_angle (α : ℤ) : 
  ∃ k : ℤ, α = k * 360 + 283 ↔ ∃ k : ℤ, α = k * 360 - 437 :=
sorry

end coterminal_angle_l290_290677


namespace quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l290_290002

theorem quadratic_equation_root_conditions
  (k : ℝ)
  (h_discriminant : 4 * k - 3 > 0)
  (h_sum_product : ∀ (x1 x2 : ℝ),
    x1 + x2 = -(2 * k + 1) ∧ 
    x1 * x2 = k^2 + 1 →
    x1 + x2 + 2 * (x1 * x2) = 1) :
  k = 1 :=
by
  sorry

theorem quadratic_equation_distinct_real_roots
  (k : ℝ) :
  (∃ (x1 x2 : ℝ),
    x1 ≠ x2 ∧
    x1^2 + (2 * k + 1) * x1 + (k^2 + 1) = 0 ∧
    x2^2 + (2 * k + 1) * x2 + (k^2 + 1) = 0) ↔
  k > 3 / 4 :=
by
  sorry

end quadratic_equation_root_conditions_quadratic_equation_distinct_real_roots_l290_290002


namespace professor_oscar_review_questions_l290_290866

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l290_290866


namespace a7_value_l290_290718

theorem a7_value
  (a : ℕ → ℝ)
  (hx2 : ∀ n, n > 0 → a n ≠ 0)
  (slope_condition : ∀ n, n ≥ 2 → 2 * a n = 2 * a (n - 1) + 1)
  (point_condition : a 1 * 4 = 8) :
  a 7 = 5 :=
by
  sorry

end a7_value_l290_290718


namespace mike_typing_time_l290_290167

-- Definitions based on the given conditions
def original_speed : ℕ := 65
def speed_reduction : ℕ := 20
def document_words : ℕ := 810
def reduced_speed : ℕ := original_speed - speed_reduction

-- The statement to prove
theorem mike_typing_time : (document_words / reduced_speed) = 18 :=
  by
    sorry

end mike_typing_time_l290_290167


namespace Vasya_numbers_l290_290834

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290834


namespace graph_passes_through_fixed_point_l290_290470

noncomputable def fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : Prop :=
  (∀ x y : ℝ, y = a * x + 2 → (x, y) = (-1, 2))

theorem graph_passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) : fixed_point a h1 h2 :=
sorry

end graph_passes_through_fixed_point_l290_290470


namespace solve_triangle_l290_290617

noncomputable def triangle_side_lengths (a b c : ℝ) : Prop :=
  a = 10 ∧ b = 9 ∧ c = 17

theorem solve_triangle (a b c : ℝ) :
  (a ^ 2 - b ^ 2 = 19) ∧ 
  (126 + 52 / 60 + 12 / 3600 = 126.87) ∧ -- Converting the angle into degrees for simplicity
  (21.25 = 21.25)  -- Diameter given directly
  → triangle_side_lengths a b c :=
sorry

end solve_triangle_l290_290617


namespace rectangle_side_divisible_by_4_l290_290979

theorem rectangle_side_divisible_by_4 (a b : ℕ)
  (h1 : ∀ i, 1 ≤ i ∧ i ≤ a → i % 4 = 0)
  (h2 : ∀ j, 1 ≤ j ∧ j ≤ b → j % 4 = 0): 
  (a % 4 = 0) ∨ (b % 4 = 0) :=
sorry

end rectangle_side_divisible_by_4_l290_290979


namespace vasya_numbers_l290_290809

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290809


namespace triangle_perimeter_l290_290354

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l290_290354


namespace no_real_a_l290_290558

noncomputable def A (a : ℝ) : Set ℝ := {x | x^2 - a * x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5 * x + 6 = 0}

theorem no_real_a (a : ℝ) : ¬ ((A a ≠ B) ∧ (A a ∪ B = B) ∧ (∅ ⊂ (A a ∩ B))) :=
by
  intro h
  sorry

end no_real_a_l290_290558


namespace total_cost_l290_290159

-- Define the conditions
def dozen := 12
def cost_of_dozen_cupcakes := 10
def cost_of_dozen_cookies := 8
def cost_of_dozen_brownies := 12

def num_dozen_cupcakes := 4
def num_dozen_cookies := 3
def num_dozen_brownies := 2

-- Define the total cost for each type of treat
def total_cost_cupcakes := num_dozen_cupcakes * cost_of_dozen_cupcakes
def total_cost_cookies := num_dozen_cookies * cost_of_dozen_cookies
def total_cost_brownies := num_dozen_brownies * cost_of_dozen_brownies

-- The theorem to prove the total cost
theorem total_cost : total_cost_cupcakes + total_cost_cookies + total_cost_brownies = 88 := by
  -- Here would go the proof, but it's omitted as per the instructions
  sorry

end total_cost_l290_290159


namespace apples_in_each_crate_l290_290779

theorem apples_in_each_crate
  (num_crates : ℕ) 
  (num_rotten : ℕ) 
  (num_boxes : ℕ) 
  (apples_per_box : ℕ) 
  (total_good_apples : ℕ) 
  (total_apples : ℕ)
  (h1 : num_crates = 12) 
  (h2 : num_rotten = 160) 
  (h3 : num_boxes = 100) 
  (h4 : apples_per_box = 20) 
  (h5 : total_good_apples = num_boxes * apples_per_box) 
  (h6 : total_apples = total_good_apples + num_rotten) : 
  total_apples / num_crates = 180 := 
by 
  sorry

end apples_in_each_crate_l290_290779


namespace parabola_min_distance_a_l290_290915

noncomputable def directrix_distance (P : Real × Real) (a : Real) : Real :=
abs (P.2 + 1 / (4 * a))

noncomputable def distance (P Q : Real × Real) : Real :=
Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

theorem parabola_min_distance_a (a : Real) :
  (∀ (P : Real × Real), P.2 = a * P.1^2 → 
    distance P (2, 0) + directrix_distance P a = Real.sqrt 5) ↔ 
    a = 1 / 4 ∨ a = -1 / 4 :=
by
  sorry

end parabola_min_distance_a_l290_290915


namespace triangle_perimeter_l290_290355

theorem triangle_perimeter (a b c : ℕ) (h1 : a = 10) (h2 : b = 6) (h3 : c = 7) :
  a + b + c = 23 := by
  sorry

end triangle_perimeter_l290_290355


namespace collinearity_APQ_l290_290099

-- Assume we have points A, B, C, X, Y, P, Q
-- Assume we have a triangle ABC such that AB > AC
variables {A B C X Y P Q : Type*}
variables [affine_space ℝ A] [affine_space ℝ B] [affine_space ℝ C] [affine_space ℝ X]
variables [affine_space ℝ Y] [affine_space ℝ P] [affine_space ℝ Q]
variables [geom: A → B → C] -- assuming geometry space

-- Definition of bisector and angles
def is_bisector (b : A → X → B → ℝ) : Prop := 
∃ (A B C X : geom), (angles_in_triangles A B X = angles_in_triangles A C X)

-- Definitions for angles as given conditions
def angles_eq (α : B → A → X → Y → ℝ) : Prop := ∠(A, B, X) = ∠(A, C, Y)

-- Conditions although one may not detail all geometry defns
axiom H1 : AB > AC
axiom H2 : is_bisector A
axiom H3 : angles_eq A (B ∠ AC)(X ∠)Y
axiom P_def : intersects (extn_line A B X) (segment C Y) P
axiom circ_def : (int_circ_of_tris geom_def: B P Y) and (int_circ_of_tris geom_def: C P X Q)

-- Main theorem to be proved
theorem collinearity_APQ : collinear {A P Q} :=
by
 sorry

end collinearity_APQ_l290_290099


namespace relative_errors_are_equal_l290_290096

theorem relative_errors_are_equal :
  let e1 := 0.04
  let l1 := 20.0
  let e2 := 0.3
  let l2 := 150.0
  (e1 / l1) = (e2 / l2) :=
by
  sorry

end relative_errors_are_equal_l290_290096


namespace sum_all_3digit_numbers_with_remainder_2_when_divided_by_6_l290_290848

theorem sum_all_3digit_numbers_with_remainder_2_when_divided_by_6 :
  let seq := (List.range' 17 150).map (λ k, 6 * k + 2)
  seq.sum = 82500 :=
by
  sorry

end sum_all_3digit_numbers_with_remainder_2_when_divided_by_6_l290_290848


namespace find_quotient_l290_290150

theorem find_quotient :
  ∀ (remainder dividend divisor quotient : ℕ),
    remainder = 1 →
    dividend = 217 →
    divisor = 4 →
    quotient = (dividend - remainder) / divisor →
    quotient = 54 :=
by
  intros remainder dividend divisor quotient hr hd hdiv hq
  rw [hr, hd, hdiv] at hq
  norm_num at hq
  exact hq

end find_quotient_l290_290150


namespace derivative_at_1_l290_290291

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end derivative_at_1_l290_290291


namespace Mel_weight_is_70_l290_290232

-- Definitions and conditions
def MelWeight (M : ℕ) :=
  3 * M + 10

theorem Mel_weight_is_70 (M : ℕ) (h1 : 3 * M + 10 = 220) :
  M = 70 :=
by
  sorry

end Mel_weight_is_70_l290_290232


namespace calculate_expression_l290_290363

variable (x : ℝ)

theorem calculate_expression : (1/2 * x^3)^2 = 1/4 * x^6 := 
by 
  sorry

end calculate_expression_l290_290363


namespace triangle_ABC_area_l290_290968

-- definition of points A, B, and C
def A : (ℝ × ℝ) := (0, 2)
def B : (ℝ × ℝ) := (6, 0)
def C : (ℝ × ℝ) := (3, 7)

-- helper function to calculate area of triangle given vertices
def triangle_area (A B C : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1 * (B.2 - C.2) + B.1 * (C.2 - A.2) + C.1 * (A.2 - B.2))

theorem triangle_ABC_area :
  triangle_area A B C = 18 := by
  sorry

end triangle_ABC_area_l290_290968


namespace total_income_l290_290219

theorem total_income (I : ℝ) (h1 : 0.10 * I * 2 + 0.20 * I + 0.06 * (I - 0.40 * I) = 0.46 * I) (h2 : 0.54 * I = 500) : I = 500 / 0.54 :=
by
  sorry

end total_income_l290_290219


namespace probability_sum_formula_l290_290603

open ProbabilityTheory

-- Definitions for the problem
variable (n k : ℕ)
variable (p q r : ℝ)
variable (ξ : ℤ → ℝ)

def distribution_ξ : ℝ → ℝ :=
  λ x, if x = 2 then p else if x = 1 then q else if x = 0 then r else 0

noncomputable def P_n (n k : ℕ) (p q r : ℝ) : ℝ :=
  ∑ j in finset.range (n + 1), 
    if |k - n| ≤ j ∧ j ≤ (n + |k - n|) / 2 then 
      (nat.choose n j) * (nat.choose (n - j) (j - |k - n|)) * p ^ j * r ^ (j - |k - n|) * q ^ (n + |k - n| - 2 * j)
    else 
      0

-- Theorem to be proven
theorem probability_sum_formula (n k : ℕ) (p q r : ℝ) (h₀ : 0 ≤ p) (h₁ : 0 ≤ q) (h₂ : 0 ≤ r) (h₃ : p + q + r = 1) :
  P_n n k p q r = ∑ j in finset.range (n + 1), 
    if |k - n| ≤ j ∧ j ≤ (n + |k - n|) / 2 then 
      (nat.choose n j) * (nat.choose (n - j) (j - |k - n|)) * p ^ j * r ^ (j - |k - n|) * q ^ (n + |k - n| - 2 * j)
    else 
      0 :=
by sorry

end probability_sum_formula_l290_290603


namespace find_a_b_l290_290899

-- Define that the roots of the corresponding equality yield the specific conditions.
theorem find_a_b (a b : ℝ) :
    (∀ x : ℝ, x^2 + (a + 1) * x + ab > 0 ↔ (x < -1 ∨ x > 4)) →
    a = -4 ∧ b = 1 := 
by
    sorry

end find_a_b_l290_290899


namespace derivative_at_one_l290_290293

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end derivative_at_one_l290_290293


namespace solve_equation_l290_290043

theorem solve_equation (x : ℝ) (h : 3 * x ≠ 0) (h2 : x + 2 ≠ 0) : (2 / (3 * x) = 1 / (x + 2)) ↔ x = 4 := by
  sorry

end solve_equation_l290_290043


namespace locus_of_point_P_l290_290229

noncomputable def ellipse_locus
  (r : ℝ) (u v : ℝ) : Prop :=
  ∃ x1 y1 : ℝ,
    (x1^2 + y1^2 = r^2) ∧ (u - x1)^2 + v^2 = y1^2

theorem locus_of_point_P {r u v : ℝ} :
  (ellipse_locus r u v) ↔ ((u^2 / (2 * r^2)) + (v^2 / r^2) ≤ 1) :=
by sorry

end locus_of_point_P_l290_290229


namespace no_n_makes_g_multiple_of_5_and_7_l290_290255

def g (n : ℕ) : ℕ := 4 + 2 * n + 3 * n^2 + n^3 + 4 * n^4 + 3 * n^5

theorem no_n_makes_g_multiple_of_5_and_7 :
  ¬ ∃ n, (2 ≤ n ∧ n ≤ 100) ∧ (g n % 5 = 0 ∧ g n % 7 = 0) :=
by
  -- Proof goes here
  sorry

end no_n_makes_g_multiple_of_5_and_7_l290_290255


namespace rain_ratio_l290_290110

def monday_rain := 2 + 1 -- inches of rain on Monday
def wednesday_rain := 0 -- inches of rain on Wednesday
def thursday_rain := 1 -- inches of rain on Thursday
def average_rain_per_day := 4 -- daily average rain total
def days_in_week := 5 -- days in a week
def weekly_total_rain := average_rain_per_day * days_in_week

-- Theorem statement
theorem rain_ratio (tuesday_rain : ℝ) (friday_rain : ℝ) 
  (h1 : friday_rain = monday_rain + tuesday_rain + wednesday_rain + thursday_rain)
  (h2 : monday_rain + tuesday_rain + wednesday_rain + thursday_rain + friday_rain = weekly_total_rain) :
  tuesday_rain / monday_rain = 2 := 
sorry

end rain_ratio_l290_290110


namespace train_cross_bridge_time_l290_290659

def train_length : ℕ := 170
def train_speed_kmph : ℕ := 45
def bridge_length : ℕ := 205

def total_distance : ℕ := train_length + bridge_length
def train_speed_mps : ℕ := (train_speed_kmph * 1000) / 3600

theorem train_cross_bridge_time : (total_distance / train_speed_mps) = 30 := 
sorry

end train_cross_bridge_time_l290_290659


namespace locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l290_290316

noncomputable def locus_of_C (a x0 y0 ξ η : ℝ) : Prop :=
  (x0 - ξ) * η^2 - 2 * ξ * y0 * η + ξ^3 - 3 * x0 * ξ^2 - a^2 * ξ + 3 * a^2 * x0 = 0

noncomputable def special_case (a ξ η : ℝ) : Prop :=
  ξ = 0 ∨ ξ^2 + η^2 = a^2

theorem locus_of_C_general_case_eq_cubic (a x0 y0 ξ η : ℝ) (hs: locus_of_C a x0 y0 ξ η) : 
  locus_of_C a x0 y0 ξ η := 
  sorry

theorem locus_of_C_special_case_eq_y_axis_or_circle (a ξ η : ℝ) : 
  special_case a ξ η := 
  sorry

end locus_of_C_general_case_eq_cubic_locus_of_C_special_case_eq_y_axis_or_circle_l290_290316


namespace max_value_y_interval_l290_290273

noncomputable def y (x : ℝ) : ℝ := x^2 - 2 * x - 1

theorem max_value_y_interval : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 3 → y x ≤ 2) ∧ (∃ x : ℝ, 0 ≤ x ∧ x ≤ 3 ∧ y x = 2) 
:=
by
  sorry

end max_value_y_interval_l290_290273


namespace problem_solution_l290_290912

def positive (n : ℕ) : Prop := n > 0
def pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1
def divides (m n : ℕ) : Prop := ∃ k, n = k * m

theorem problem_solution (a b c : ℕ) :
  positive a → positive b → positive c →
  pairwise_coprime a b c →
  divides (a^2) (b^3 + c^3) →
  divides (b^2) (a^3 + c^3) →
  divides (c^2) (a^3 + b^3) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end problem_solution_l290_290912


namespace ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l290_290171

theorem ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one
  (m n : ℕ) : (10 ^ m + 1) % (10 ^ n - 1) ≠ 0 := 
  sorry

end ten_pow_m_plus_one_not_divisible_by_ten_pow_n_minus_one_l290_290171


namespace range_of_b_l290_290286

theorem range_of_b (b : ℝ) (h : Real.sqrt ((b-2)^2) = 2 - b) : b ≤ 2 :=
by {
  sorry
}

end range_of_b_l290_290286


namespace area_of_triangle_KBC_l290_290016

noncomputable def length_FE := 7
noncomputable def length_BC := 7
noncomputable def length_JB := 5
noncomputable def length_BK := 5

theorem area_of_triangle_KBC : (1 / 2 : ℝ) * length_BC * length_BK = 17.5 := by
  -- conditions: 
  -- 1. Hexagon ABCDEF is equilateral with each side of length s.
  -- 2. Squares ABJI and FEHG are formed outside the hexagon with areas 25 and 49 respectively.
  -- 3. Triangle JBK is equilateral.
  -- 4. FE = BC.
  sorry

end area_of_triangle_KBC_l290_290016


namespace find_x_range_l290_290709

noncomputable def f (x : ℝ) : ℝ := if h : x ≥ 0 then 3^(-x) else 3^(x)

theorem find_x_range (x : ℝ) (h1 : f 2 = -f (2*x - 1) ∧ f 2 < 0) : -1/2 < x ∧ x < 3/2 := by
  -- Proof goes here
  sorry

end find_x_range_l290_290709


namespace alyssa_limes_picked_l290_290662

-- Definitions for the conditions
def total_limes : ℕ := 57
def mike_limes : ℕ := 32

-- The statement to be proved
theorem alyssa_limes_picked :
  ∃ (alyssa_limes : ℕ), total_limes - mike_limes = alyssa_limes ∧ alyssa_limes = 25 :=
by
  have alyssa_limes : ℕ := total_limes - mike_limes
  use alyssa_limes
  sorry

end alyssa_limes_picked_l290_290662


namespace compute_expression_l290_290367

theorem compute_expression : 85 * 1305 - 25 * 1305 + 100 = 78400 := by
  sorry

end compute_expression_l290_290367


namespace dad_steps_eq_90_l290_290673

-- Define the conditions given in the problem
variables (masha_steps yasha_steps dad_steps : ℕ)

-- Conditions:
-- 1. Dad takes 3 steps while Masha takes 5 steps
-- 2. Masha takes 3 steps while Yasha takes 5 steps
-- 3. Together, Masha and Yasha made 400 steps
def conditions := dad_steps * 5 = 3 * masha_steps ∧ masha_steps * yasha_steps = 3 * yasha_steps ∧ 3 * yasha_steps = 400

-- Theorem stating the proof problem
theorem dad_steps_eq_90 : conditions masha_steps yasha_steps dad_steps → dad_steps = 90 :=
by
  sorry

end dad_steps_eq_90_l290_290673


namespace circle_second_x_intercept_l290_290213

theorem circle_second_x_intercept :
  ∀ (circle : ℝ × ℝ → Prop), (∀ (x y : ℝ), circle (x, y) ↔ (x - 5) ^ 2 + y ^ 2 = 25) →
    ∃ x : ℝ, (x ≠ 0 ∧ circle (x, 0) ∧ x = 10) :=
by {
  sorry
}

end circle_second_x_intercept_l290_290213


namespace binomial_30_3_l290_290878

theorem binomial_30_3 : Nat.choose 30 3 = 4060 := by
  sorry

end binomial_30_3_l290_290878


namespace circle_ratio_increase_l290_290424

theorem circle_ratio_increase (r : ℝ) (h : r + 2 ≠ 0) : 
  (2 * Real.pi * (r + 2)) / (2 * (r + 2)) = Real.pi :=
by
  sorry

end circle_ratio_increase_l290_290424


namespace HN_passes_through_fixed_point_l290_290127

noncomputable def ellipse_equation : Prop :=
  ∀ (x y : ℝ), (x, y) = (0, -2) ∨ (x, y) = (3 / 2, -1) → x^2 / 3 + y^2 / 4 = 1

theorem HN_passes_through_fixed_point (P M N H : ℝ × ℝ) :
  ∀ x y : ℝ, ellipse_equation →
  (∃ t : ℝ, x = 1 ∧ y = t) →
  (P = (1, -2)) →
  (∃ k : ℝ, x = 1 + k * (M.1 - 1) ∧ y = -2 + k * (M.2 + 2)) →
  ((∃ a b : ℝ, a = H.1 + 3y - x ∧ b = y) → ((x = 0 ∧ y = -2))) :=
by
  -- The proof steps are omitted as per the instructions
  sorry

end HN_passes_through_fixed_point_l290_290127


namespace not_a_factorization_l290_290066

open Nat

theorem not_a_factorization : ¬ (∃ (f g : ℝ → ℝ), (∀ (x : ℝ), x^2 + 6*x - 9 = f x * g x)) :=
by
  sorry

end not_a_factorization_l290_290066


namespace probability_bus_there_when_mark_arrives_l290_290647

noncomputable def isProbabilityBusThereWhenMarkArrives : Prop :=
  let busArrival : ℝ := 60 -- The bus can arrive from time 0 to 60 minutes (2:00 PM to 3:00 PM)
  let busWait : ℝ := 30 -- The bus waits for 30 minutes
  let markArrival : ℝ := 90 -- Mark can arrive from time 30 to 90 minutes (2:30 PM to 3:30 PM)
  let overlapArea : ℝ := 1350 -- Total shaded area where bus arrival overlaps with Mark's arrival
  let totalArea : ℝ := busArrival * (markArrival - 30)
  let probability := overlapArea / totalArea
  probability = 1 / 4

theorem probability_bus_there_when_mark_arrives : isProbabilityBusThereWhenMarkArrives :=
by
  sorry

end probability_bus_there_when_mark_arrives_l290_290647


namespace determine_coefficients_l290_290412

variable {α : Type} [Field α]
variables (a a1 a2 a3 : α)

theorem determine_coefficients (h : ∀ x : α, a + a1 * (x - 1) + a2 * (x - 1)^2 + a3 * (x - 1)^3 = x^3) :
  a = 1 ∧ a2 = 3 :=
by
  -- To be proven
  sorry

end determine_coefficients_l290_290412


namespace find_width_of_first_tract_l290_290143

-- Definitions based on given conditions
noncomputable def area_first_tract (W : ℝ) : ℝ := 300 * W
def area_second_tract : ℝ := 250 * 630
def combined_area : ℝ := 307500

-- The theorem we need to prove: width of the first tract is 500 meters
theorem find_width_of_first_tract (W : ℝ) (h : area_first_tract W + area_second_tract = combined_area) : W = 500 :=
by
  sorry

end find_width_of_first_tract_l290_290143


namespace incorrect_conclusion_l290_290564

noncomputable def f (x : ℝ) : ℝ := Real.sin x * Real.cos (2 * x)

theorem incorrect_conclusion :
  ¬ (∀ x : ℝ, f ( (3 * Real.pi) / 4 - x ) + f x = 0) :=
by
  sorry

end incorrect_conclusion_l290_290564


namespace guest_bedroom_ratio_l290_290515

theorem guest_bedroom_ratio 
  (lr_dr_kitchen : ℝ) (total_house : ℝ) (master_bedroom : ℝ) (guest_bedroom : ℝ) 
  (h1 : lr_dr_kitchen = 1000) 
  (h2 : total_house = 2300)
  (h3 : master_bedroom = 1040)
  (h4 : guest_bedroom = total_house - (lr_dr_kitchen + master_bedroom)) :
  guest_bedroom / master_bedroom = 1 / 4 := 
by
  sorry

end guest_bedroom_ratio_l290_290515


namespace correct_eqns_l290_290439

theorem correct_eqns (x y : ℝ) (h1 : x - y = 4.5) (h2 : 1/2 * x + 1 = y) :
  x - y = 4.5 ∧ 1/2 * x + 1 = y :=
by {
  exact ⟨h1, h2⟩,
}

end correct_eqns_l290_290439


namespace exponential_equation_solution_l290_290146

theorem exponential_equation_solution (b x : ℝ) (hb : b > 1) (hx : x > 0)
  (h : (3 * x)^(Real.log 3 / Real.log b) - (5 * x)^(Real.log 5 / Real.log b) = 0) :
  x = (3 / 5)^(Real.log b / Real.log (5 / 3)) :=
by
  sorry

end exponential_equation_solution_l290_290146


namespace determine_pairs_l290_290246

open Int

-- Definitions corresponding to the conditions of the problem:
def is_prime (p : ℕ) : Prop := Nat.Prime p
def condition1 (p n : ℕ) : Prop := is_prime p
def condition2 (p n : ℕ) : Prop := n ≤ 2 * p
def condition3 (p n : ℕ) : Prop := (n^(p-1)) ∣ ((p-1)^n + 1)

-- Main theorem statement:
theorem determine_pairs (n p : ℕ) (h1 : condition1 p n) (h2 : condition2 p n) (h3 : condition3 p n) :
  (n = 1 ∧ is_prime p) ∨ (n = 2 ∧ p = 2) ∨ (n = 3 ∧ p = 3) :=
sorry

end determine_pairs_l290_290246


namespace observations_decrement_l290_290047

theorem observations_decrement (n : ℤ) (h_n_pos : n > 0) : 200 - 15 = 185 :=
by
  sorry

end observations_decrement_l290_290047


namespace area_of_black_region_l290_290223

theorem area_of_black_region (side_small side_large : ℕ) 
  (h1 : side_small = 5) 
  (h2 : side_large = 9) : 
  (side_large * side_large) - (side_small * side_small) = 56 := 
by
  sorry

end area_of_black_region_l290_290223


namespace birth_date_of_id_number_l290_290333

def extract_birth_date (id_number : String) := 
  let birth_str := id_number.drop 6 |>.take 8
  let year := birth_str.take 4
  let month := birth_str.drop 4 |>.take 2
  let day := birth_str.drop 6
  (year, month, day)

theorem birth_date_of_id_number :
  extract_birth_date "320106194607299871" = ("1946", "07", "29") := by
  sorry

end birth_date_of_id_number_l290_290333


namespace complex_square_example_l290_290410

noncomputable def z : ℂ := 5 - 3 * Complex.I
noncomputable def i_squared : ℂ := Complex.I ^ 2

theorem complex_square_example : z ^ 2 = 34 - 30 * Complex.I := by
  have i_squared_eq : i_squared = -1 := by
    unfold i_squared
    rw [Complex.I_sq]
    rfl
  unfold z
  calc
    (5 - 3 * Complex.I) ^ 2
        = (5 ^ 2 - (3 * Complex.I) ^ 2 - 2 * 5 * 3 * Complex.I) : by
          ring
    ... = 25 - 9 * i_squared - 30 * Complex.I : by
          rw [Complex.mul_sq, Complex.I_sq]
    ... = 25 - 9 * (-1) - 30 * Complex.I : by
          rw [i_squared_eq]
    ... = 25 + 9 - 30 * Complex.I : by
          ring
    ... = 34 - 30 * Complex.I : by
          ring

end complex_square_example_l290_290410


namespace alloy_problem_l290_290013

theorem alloy_problem (x : ℝ) (h1 : 0.12 * x + 0.08 * 30 = 0.09333333333333334 * (x + 30)) : x = 15 :=
by
  sorry

end alloy_problem_l290_290013


namespace solve_quadratic_eq_l290_290174

theorem solve_quadratic_eq (x : ℝ) (h : x^2 - 2*x + 1 = 0) : x = 1 :=
by
  sorry

end solve_quadratic_eq_l290_290174


namespace maximum_gel_pens_l290_290209

theorem maximum_gel_pens 
  (x y z : ℕ) 
  (h1 : x + y + z = 20)
  (h2 : 10 * x + 50 * y + 80 * z = 1000)
  (hx : x > 0)
  (hy : y > 0)
  (hz : z > 0) 
  : y ≤ 13 :=
sorry

end maximum_gel_pens_l290_290209


namespace avg_xy_l290_290965

theorem avg_xy (x y : ℝ) (h : (4 + 6.5 + 8 + x + y) / 5 = 18) : (x + y) / 2 = 35.75 :=
by
  sorry

end avg_xy_l290_290965


namespace shifted_quadratic_eq_l290_290421

-- Define the original quadratic function
def orig_fn (x : ℝ) : ℝ := -x^2

-- Define the function after shifting 1 unit to the left
def shifted_left_fn (x : ℝ) : ℝ := - (x + 1)^2

-- Define the final function after also shifting 3 units up
def final_fn (x : ℝ) : ℝ := - (x + 1)^2 + 3

-- Prove the final function is the correctly transformed function from the original one
theorem shifted_quadratic_eq : ∀ (x : ℝ), final_fn x = - (x + 1)^2 + 3 :=
by 
  intro x
  sorry

end shifted_quadratic_eq_l290_290421


namespace scientific_notation_of_11090000_l290_290760

theorem scientific_notation_of_11090000 :
  ∃ (x : ℝ) (n : ℤ), 11090000 = x * 10^n ∧ x = 1.109 ∧ n = 7 :=
by
  -- skip the proof
  sorry

end scientific_notation_of_11090000_l290_290760


namespace anton_has_more_cards_than_ann_l290_290098

-- Define Heike's number of cards
def heike_cards : ℕ := 60

-- Define Anton's number of cards in terms of Heike's cards
def anton_cards (H : ℕ) : ℕ := 3 * H

-- Define Ann's number of cards as equal to Heike's cards
def ann_cards (H : ℕ) : ℕ := H

-- Theorem statement
theorem anton_has_more_cards_than_ann 
  (H : ℕ) (H_equals : H = heike_cards) : 
  anton_cards H - ann_cards H = 120 :=
by
  -- At this point, the actual proof would be inserted.
  sorry

end anton_has_more_cards_than_ann_l290_290098


namespace vasya_numbers_l290_290798

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290798


namespace find_coordinates_l290_290413

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end find_coordinates_l290_290413


namespace shaded_area_of_squares_l290_290859

theorem shaded_area_of_squares :
  let s_s := 4
  let s_L := 9
  let area_L := s_L * s_L
  let area_s := s_s * s_s
  area_L - area_s = 65 := sorry

end shaded_area_of_squares_l290_290859


namespace polygon_RS_ST_sum_l290_290322

theorem polygon_RS_ST_sum
  (PQ RS ST: ℝ)
  (PQ_eq : PQ = 10)
  (QR_eq : QR = 7)
  (TU_eq : TU = 6)
  (polygon_area : PQ * QR = 70)
  (PQRSTU_area : 70 = 70) :
  RS + ST = 80 :=
by
  sorry

end polygon_RS_ST_sum_l290_290322


namespace problem1_solution_set_problem2_proof_l290_290137

-- Define the function f(x) with a given value of a.
def f (x : ℝ) (a : ℝ) : ℝ := |x + a|

-- Problem 1: Solve the inequality f(x) ≥ 5 - |x - 2| when a = 1.
theorem problem1_solution_set (x : ℝ) :
  f x 1 ≥ 5 - |x - 2| ↔ x ∈ (Set.Iic (-2) ∪ Set.Ici 3) :=
sorry

-- Problem 2: Given the solution set of f(x) ≤ 5 is [-9, 1] and the equation 1/m + 1/(2n) = a, prove m + 2n ≥ 1
theorem problem2_proof (a m n : ℝ) (hma : a = 4) (hmpos : m > 0) (hnpos : n > 0) :
  (1 / m + 1 / (2 * n) = a) → m + 2 * n ≥ 1 :=
sorry

end problem1_solution_set_problem2_proof_l290_290137


namespace part1_part2_part3_l290_290086

section ShoppingMall

variable (x y a b : ℝ)
variable (cpaA spaA cpaB spaB : ℝ)
variable (n total_y yuan : ℝ)

-- Conditions given in the problem
def cost_price_A := 160
def selling_price_A := 220
def cost_price_B := 120
def selling_price_B := 160
def total_clothing := 100
def min_A_clothing := 60
def max_budget := 15000
def discount_diff := 4
def max_profit_with_discount := 4950

-- Definitions applied from conditions
def profit_per_piece_A := selling_price_A - cost_price_A
def profit_per_piece_B := selling_price_B - cost_price_B

-- Question 1: Functional relationship between y and x
theorem part1 : 
  (∀ (x : ℝ), x ≥ 0 → x ≤ total_clothing → 
  y = profit_per_piece_A * x + profit_per_piece_B * (total_clothing - x)) →
  y = 20 * x + 4000 := 
sorry

-- Question 2: Maximum profit under given cost constraints
theorem part2 : 
  (min_A_clothing ≤ x ∧ x ≤ 75 ∧ 
  (cost_price_A * x + cost_price_B * (total_clothing - x) ≤ max_budget)) →
  y = 20 * 75 + 4000 → 
  y = 5500 :=
sorry

-- Question 3: Determine a under max profit condition
theorem part3 : 
  (a - b = discount_diff ∧ 0 < a ∧ a < 20 ∧ 
  (20 - a) * 75 + 4000 + 100 * a - 400 = max_profit_with_discount) →
  a = 9 :=
sorry

end ShoppingMall

end part1_part2_part3_l290_290086


namespace expression_value_l290_290198

theorem expression_value : 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end expression_value_l290_290198


namespace zeros_not_adjacent_probability_l290_290574

def total_arrangements : ℕ := Nat.factorial 5

def adjacent_arrangements : ℕ := 2 * Nat.factorial 4

def probability_not_adjacent : ℚ := 
  1 - (adjacent_arrangements / total_arrangements)

theorem zeros_not_adjacent_probability :
  probability_not_adjacent = 0.6 := 
by 
  sorry

end zeros_not_adjacent_probability_l290_290574


namespace problem_statement_l290_290027

noncomputable def f (x : ℝ) : ℝ := sorry

axiom even_function : ∀ x : ℝ, f (-x) = f (x)
axiom functional_eq : ∀ x : ℝ, f (x + 6) = f (x) + f 3
axiom initial_condition : f 2 = 3

theorem problem_statement : f 2006 + f 2007 = 3 :=
by
  sorry

end problem_statement_l290_290027


namespace Vasya_numbers_l290_290833

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290833


namespace lily_catches_up_mary_in_60_minutes_l290_290165

theorem lily_catches_up_mary_in_60_minutes
  (mary_speed : ℝ) (lily_speed : ℝ) (initial_distance : ℝ)
  (h_mary_speed : mary_speed = 4)
  (h_lily_speed : lily_speed = 6)
  (h_initial_distance : initial_distance = 2) :
  ∃ t : ℝ, t = 60 := by
  sorry

end lily_catches_up_mary_in_60_minutes_l290_290165


namespace tan_theta_parallel_vectors_l290_290556

theorem tan_theta_parallel_vectors (θ : ℝ) (h : (sin θ, 1) ∥ (cos θ, -2)) : Real.tan θ = -2 :=
by
  -- Proof goes here
  sorry

end tan_theta_parallel_vectors_l290_290556


namespace total_ideal_matching_sets_l290_290161

-- Definitions based on the provided problem statement
def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def is_ideal_matching_set (A B : Set ℕ) : Prop := A ∩ B = {1, 3, 5}

-- Theorem statement for the total number of ideal matching sets
theorem total_ideal_matching_sets : ∃ n, n = 27 ∧ ∀ (A B : Set ℕ), A ⊆ U ∧ B ⊆ U ∧ is_ideal_matching_set A B → n = 27 := 
sorry

end total_ideal_matching_sets_l290_290161


namespace probability_star_top_card_is_one_fifth_l290_290657

-- Define the total number of cards in the deck
def total_cards : ℕ := 65

-- Define the number of star cards in the deck
def star_cards : ℕ := 13

-- Define the probability calculation
def probability_star_top_card : ℚ := star_cards / total_cards

-- State the theorem regarding the probability
theorem probability_star_top_card_is_one_fifth :
  probability_star_top_card = 1 / 5 :=
by
  sorry

end probability_star_top_card_is_one_fifth_l290_290657


namespace maurice_late_467th_trip_l290_290945

-- Define the recurrence relation
def p (n : ℕ) : ℚ := 
  if n = 0 then 0
  else 1 / 4 * (p (n - 1) + 1)

-- Define the steady-state probability
def steady_state_p : ℚ := 1 / 3

-- Define L_n as the probability Maurice is late on the nth day
def L (n : ℕ) : ℚ := 1 - p n

-- The main goal (probability Maurice is late on his 467th trip)
theorem maurice_late_467th_trip :
  L 467 = 2 / 3 :=
sorry

end maurice_late_467th_trip_l290_290945


namespace triangles_in_decagon_l290_290369

theorem triangles_in_decagon :
  let n := 10 in
  let k := 3 in
  Nat.choose n k = 120 := by
  sorry

end triangles_in_decagon_l290_290369


namespace roots_of_quadratic_l290_290145

theorem roots_of_quadratic (a b : ℝ) (h₁ : a + b = 2) (h₂ : a * b = -3) : a^2 + b^2 = 10 := 
by
  -- proof steps go here, but not required as per the instruction
  sorry

end roots_of_quadratic_l290_290145


namespace inverse_shifted_point_l290_290627

def odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def inverse_function (f g : ℝ → ℝ) : Prop := ∀ y, f (g y) = y ∧ ∀ x, g (f x) = x

theorem inverse_shifted_point
  (f : ℝ → ℝ)
  (hf_odd : odd_function f)
  (hf_point : f (-1) = 3)
  (g : ℝ → ℝ)
  (hg_inverse : inverse_function f g) :
  g (2 - 5) = 1 :=
by
  sorry

end inverse_shifted_point_l290_290627


namespace rectangle_other_side_l290_290222

theorem rectangle_other_side (A x y : ℝ) (hA : A = 1 / 8) (hx : x = 1 / 2) (hArea : A = x * y) :
    y = 1 / 4 := 
  sorry

end rectangle_other_side_l290_290222


namespace pizzas_served_during_lunch_l290_290997

def total_pizzas : ℕ := 15
def dinner_pizzas : ℕ := 6

theorem pizzas_served_during_lunch :
  ∃ lunch_pizzas : ℕ, lunch_pizzas = total_pizzas - dinner_pizzas :=
by
  use 9
  exact rfl

end pizzas_served_during_lunch_l290_290997


namespace min_phi_symmetry_l290_290614

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi / 6)

theorem min_phi_symmetry (phi : ℝ) (h_phi : phi > 0)
  (h_sym : ∀ x, Real.sin (x + Real.pi / 6 + phi) = -Real.sin (x + Real.pi / 6 + 2 * Real.pi - phi)) :
  phi = Real.pi / 2 :=
by 
  sorry

end min_phi_symmetry_l290_290614


namespace applicant_overall_score_l290_290854

-- Definitions for the conditions
def writtenTestScore : ℝ := 80
def interviewScore : ℝ := 60
def weightWrittenTest : ℝ := 0.6
def weightInterview : ℝ := 0.4

-- Theorem statement
theorem applicant_overall_score : 
  (writtenTestScore * weightWrittenTest) + (interviewScore * weightInterview) = 72 := 
by
  sorry

end applicant_overall_score_l290_290854


namespace max_expression_value_l290_290525

open Real

theorem max_expression_value (a b d x₁ x₂ x₃ x₄ : ℝ) 
  (h1 : (x₁^4 - a * x₁^3 + b * x₁^2 - a * x₁ + d = 0))
  (h2 : (x₂^4 - a * x₂^3 + b * x₂^2 - a * x₂ + d = 0))
  (h3 : (x₃^4 - a * x₃^3 + b * x₃^2 - a * x₃ + d = 0))
  (h4 : (x₄^4 - a * x₄^3 + b * x₄^2 - a * x₄ + d = 0))
  (h5 : (1 / 2 ≤ x₁ ∧ x₁ ≤ 2))
  (h6 : (1 / 2 ≤ x₂ ∧ x₂ ≤ 2))
  (h7 : (1 / 2 ≤ x₃ ∧ x₃ ≤ 2))
  (h8 : (1 / 2 ≤ x₄ ∧ x₄ ≤ 2)) :
  ∃ (M : ℝ), M = 5 / 4 ∧
  (∀ (y₁ y₂ y₃ y₄ : ℝ),
    (y₁^4 - a * y₁^3 + b * y₁^2 - a * y₁ + d = 0) →
    (y₂^4 - a * y₂^3 + b * y₂^2 - a * y₂ + d = 0) →
    (y₃^4 - a * y₃^3 + b * y₃^2 - a * y₃ + d = 0) →
    (y₄^4 - a * y₄^3 + b * y₄^2 - a * y₄ + d = 0) →
    (1 / 2 ≤ y₁ ∧ y₁ ≤ 2) →
    (1 / 2 ≤ y₂ ∧ y₂ ≤ 2) →
    (1 / 2 ≤ y₃ ∧ y₃ ≤ 2) →
    (1 / 2 ≤ y₄ ∧ y₄ ≤ 2) →
    (y = (y₁ + y₂) * (y₁ + y₃) * y₄ / ((y₄ + y₂) * (y₄ + y₃) * y₁)) →
    y ≤ M) := 
sorry

end max_expression_value_l290_290525


namespace cara_meets_don_distance_l290_290988

theorem cara_meets_don_distance (distance total_distance : ℝ) (cara_speed don_speed : ℝ) (delay : ℝ) 
  (h_total_distance : total_distance = 45)
  (h_cara_speed : cara_speed = 6)
  (h_don_speed : don_speed = 5)
  (h_delay : delay = 2) :
  distance = 30 :=
by
  have h := 1 / total_distance
  have : cara_speed * (distance / cara_speed) + don_speed * (distance / cara_speed - delay) = 45 := sorry
  exact sorry

end cara_meets_don_distance_l290_290988


namespace factorial_fraction_l290_290841

theorem factorial_fraction :
  (16.factorial / (6.factorial * 10.factorial) : ℚ) = 728 :=
by
  sorry

end factorial_fraction_l290_290841


namespace tens_digit_of_expression_l290_290493

theorem tens_digit_of_expression :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 1 :=
by sorry

end tens_digit_of_expression_l290_290493


namespace minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l290_290717

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := 
  Real.log (x + 1) + 2 / (x + 1) + a * x - 2

theorem minimum_value_when_a_is_1 : ∀ x : ℝ, ∃ m : ℝ, 
  (∀ y : ℝ, f y 1 ≥ f x 1) ∧ (f x 1 = m) :=
sorry

theorem range_of_a_given_fx_geq_0 : ∀ a : ℝ, (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → 0 ≤ f x a) ↔ 1 ≤ a :=
sorry

end minimum_value_when_a_is_1_range_of_a_given_fx_geq_0_l290_290717


namespace compute_expression_l290_290239

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end compute_expression_l290_290239


namespace pathway_width_l290_290089

theorem pathway_width {r1 r2 : ℝ} 
  (h1 : 2 * Real.pi * r1 - 2 * Real.pi * r2 = 20 * Real.pi)
  (h2 : r1 - r2 = 10) :
  r1 - r2 + 4 = 14 := 
by 
  sorry

end pathway_width_l290_290089


namespace simplify_sqrt_expression_l290_290459

theorem simplify_sqrt_expression :
  (Real.sqrt 726 / Real.sqrt 242) + (Real.sqrt 484 / Real.sqrt 121) = Real.sqrt 3 + 2 :=
by
  -- Proof goes here
  sorry

end simplify_sqrt_expression_l290_290459


namespace figure_count_mistake_l290_290953

theorem figure_count_mistake
    (b g : ℕ)
    (total_figures : ℕ)
    (boy_circles boy_squares girl_circles girl_squares : ℕ)
    (total_figures_counted : ℕ) :
  boy_circles = 3 → boy_squares = 8 → girl_circles = 9 → girl_squares = 2 →
  total_figures_counted = 4046 →
  (∃ (b g : ℕ), 11 * b + 11 * g ≠ 4046) :=
by
  intros
  sorry

end figure_count_mistake_l290_290953


namespace min_value_of_a_plus_b_l290_290723

theorem min_value_of_a_plus_b (a b c : ℝ) (C : ℝ) 
  (hC : C = 60) 
  (h : (a + b)^2 - c^2 = 4) : 
  a + b ≥ (4 * Real.sqrt 3) / 3 :=
by
  sorry

end min_value_of_a_plus_b_l290_290723


namespace neg_of_exists_lt_is_forall_ge_l290_290473

theorem neg_of_exists_lt_is_forall_ge :
  (¬ (∃ x : ℝ, x^2 - 2 * x + 1 < 0)) ↔ (∀ x : ℝ, x^2 - 2 * x + 1 ≥ 0) :=
by
  sorry

end neg_of_exists_lt_is_forall_ge_l290_290473


namespace johns_sixth_quiz_score_l290_290019

theorem johns_sixth_quiz_score
  (score1 score2 score3 score4 score5 : ℕ)
  (h1 : score1 = 85)
  (h2 : score2 = 90)
  (h3 : score3 = 88)
  (h4 : score4 = 92)
  (h5 : score5 = 95)
  : (∃ score6 : ℕ, (score1 + score2 + score3 + score4 + score5 + score6) / 6 = 90) :=
by
  use 90
  sorry

end johns_sixth_quiz_score_l290_290019


namespace t_plus_inv_t_eq_three_l290_290910

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l290_290910


namespace problem1_problem2_l290_290104

-- Problem 1
theorem problem1 : 5*Real.sqrt 2 - (Real.sqrt 18 + Real.sqrt (1/2)) = (3/2) * Real.sqrt 2 := by
  sorry

-- Problem 2
theorem problem2 : (2*Real.sqrt 3 - 1)^2 + (Real.sqrt 24) / (Real.sqrt 2) = 13 - 2*Real.sqrt 3 := by
  sorry

end problem1_problem2_l290_290104


namespace remainder_9_minus_n_plus_n_plus_5_mod_8_l290_290579

theorem remainder_9_minus_n_plus_n_plus_5_mod_8 (n : ℤ) : 
  ((9 - n) + (n + 5)) % 8 = 6 := by
  sorry

end remainder_9_minus_n_plus_n_plus_5_mod_8_l290_290579


namespace int_n_satisfying_conditions_l290_290545

theorem int_n_satisfying_conditions : 
  (∃! (n : ℤ), ∃ (k : ℤ), (n + 3 = k^2 * (23 - n)) ∧ n ≠ 23) :=
by
  use 2
  -- Provide a proof for this statement here
  sorry

end int_n_satisfying_conditions_l290_290545


namespace sunflower_is_taller_l290_290315

def sister_height_ft : Nat := 4
def sister_height_in : Nat := 3
def sunflower_height_ft : Nat := 6

def feet_to_inches (ft : Nat) : Nat := ft * 12

def sister_height := feet_to_inches sister_height_ft + sister_height_in
def sunflower_height := feet_to_inches sunflower_height_ft

def height_difference : Nat := sunflower_height - sister_height

theorem sunflower_is_taller : height_difference = 21 :=
by
  -- proof has to be provided:
  sorry

end sunflower_is_taller_l290_290315


namespace cheat_buying_percentage_l290_290858

-- Definitions for the problem
def profit_margin := 0.5
def cheat_selling := 0.2

-- Prove that the cheating percentage while buying is 20%
theorem cheat_buying_percentage : ∃ x : ℝ, (0 ≤ x ∧ x ≤ 1) ∧ x = 0.2 := by
  sorry

end cheat_buying_percentage_l290_290858


namespace diagonals_diff_heptagon_octagon_l290_290582

-- Define the function to calculate the number of diagonals in a polygon with n sides
def num_diagonals (n : ℕ) : ℕ := (n * (n - 3)) / 2

theorem diagonals_diff_heptagon_octagon : 
  let A := num_diagonals 7
  let B := num_diagonals 8
  B - A = 6 :=
by
  sorry

end diagonals_diff_heptagon_octagon_l290_290582


namespace lowest_point_graph_of_y_l290_290178

theorem lowest_point_graph_of_y (x : ℝ) (h : x > -1) :
  (x, (x^2 + 2 * x + 2) / (x + 1)) = (0, 2) ∧ ∀ y > -1, ( (y^2 + 2 * y + 2) / (y + 1) >= 2) := 
sorry

end lowest_point_graph_of_y_l290_290178


namespace find_value_of_a_l290_290267

theorem find_value_of_a (a : ℤ) (h1 : 0 < a) (h2 : a < 13) (h3 : (53 ^ 2017 + a) % 13 = 0) : a = 12 := 
by 
  sorry

end find_value_of_a_l290_290267


namespace correct_calculation_l290_290844

variable (a b : ℕ)

theorem correct_calculation : 3 * a * b - 2 * a * b = a * b := 
by sorry

end correct_calculation_l290_290844


namespace find_y_l290_290408

theorem find_y (x y : ℚ) (h1 : x = 153) (h2 : x^3 * y - 4 * x^2 * y + 4 * x * y = 350064) : 
  y = 40 / 3967 :=
by
  -- Proof to be filled in
  sorry

end find_y_l290_290408


namespace double_root_values_l290_290349

theorem double_root_values (b₃ b₂ b₁ s : ℤ) (h : ∀ x : ℤ, (x * (x - s)) ∣ (x^4 + b₃ * x^3 + b₂ * x^2 + b₁ * x + 36)) 
  : s = -6 ∨ s = -3 ∨ s = -2 ∨ s = -1 ∨ s = 1 ∨ s = 2 ∨ s = 3 ∨ s = 6 :=
sorry

end double_root_values_l290_290349


namespace pears_thrown_away_on_first_day_l290_290863

theorem pears_thrown_away_on_first_day (x : ℝ) (P : ℝ) 
  (h1 : P > 0)
  (h2 : 0.8 * P = P * 0.8)
  (total_thrown_percentage : (x / 100) * 0.2 * P + 0.2 * (1 - x / 100) * 0.2 * P = 0.12 * P ) : 
  x = 50 :=
by
  sorry

end pears_thrown_away_on_first_day_l290_290863


namespace travel_time_equation_l290_290774

theorem travel_time_equation (x : ℝ) (h1 : ∀ d : ℝ, d > 0) :
  (x / 160) - (x / 200) = 2.5 :=
sorry

end travel_time_equation_l290_290774


namespace minimize_expression_l290_290312

theorem minimize_expression (a b c : ℝ) (h₁ : 0 < a) (h₂ : 0 < b) (h₃ : 0 < c) :
  4 * a^3 + 8 * b^3 + 27 * c^3 + 1 / (3 * a * b * c) ≥ 6 := 
by 
  sorry

end minimize_expression_l290_290312


namespace train_travel_distance_l290_290650

theorem train_travel_distance
  (coal_per_mile_lb : ℝ)
  (remaining_coal_lb : ℝ)
  (travel_distance_per_unit_mile : ℝ)
  (units_per_unit_lb : ℝ)
  (remaining_units : ℝ)
  (total_distance : ℝ) :
  coal_per_mile_lb = 2 →
  remaining_coal_lb = 160 →
  travel_distance_per_unit_mile = 5 →
  units_per_unit_lb = remaining_coal_lb / coal_per_mile_lb →
  remaining_units = units_per_unit_lb →
  total_distance = remaining_units * travel_distance_per_unit_mile →
  total_distance = 400 :=
by
  sorry

end train_travel_distance_l290_290650


namespace find_t_l290_290890

variable (t : ℚ)

def point_on_line (p1 p2 p3 : ℚ × ℚ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_t (t : ℚ) : point_on_line (3, 0) (0, 7) (t, 8) → t = -3 / 7 := by
  sorry

end find_t_l290_290890


namespace evens_divisors_lt_100_l290_290406

theorem evens_divisors_lt_100 : 
  ∃ n : ℕ, n = 90 ∧ ∀ k : ℕ, (1 ≤ k < 100) → (even k ↔ (∃ m : ℕ, m * m = k)) ↔ (n = 90) := 
sorry

end evens_divisors_lt_100_l290_290406


namespace largest_option_is_B_l290_290494

/-- Define the provided options -/
def optionA : ℝ := Real.sqrt (Real.cbrt 56)
def optionB : ℝ := Real.sqrt (Real.cbrt 3584)
def optionC : ℝ := Real.sqrt (Real.cbrt 2744)
def optionD : ℝ := Real.cbrt (Real.sqrt 392)
def optionE : ℝ := Real.cbrt (Real.sqrt 448)

/-- The main theorem -/
theorem largest_option_is_B : optionB > optionA ∧ optionB > optionC ∧ optionB > optionD ∧ optionB > optionE :=
by
  sorry

end largest_option_is_B_l290_290494


namespace jessica_initial_money_l290_290738

def amount_spent : ℝ := 10.22
def amount_left : ℝ := 1.51
def initial_amount : ℝ := 11.73

theorem jessica_initial_money :
  amount_spent + amount_left = initial_amount := 
  by
    sorry

end jessica_initial_money_l290_290738


namespace remaining_area_correct_l290_290155

noncomputable def remaining_area_ABHFGD : ℝ :=
  let area_square_ABCD := 25
  let area_square_EFGD := 16
  let side_length_ABCD := Real.sqrt area_square_ABCD
  let side_length_EFGD := Real.sqrt area_square_EFGD
  let overlap_area := 8
  area_square_ABCD + area_square_EFGD - overlap_area

theorem remaining_area_correct :
  let area := remaining_area_ABHFGD
  area = 33 :=
by
  sorry

end remaining_area_correct_l290_290155


namespace arithmetic_sequence_eighth_term_l290_290326

-- Define the arithmetic sequence
def arithmetic_sequence (a d : ℚ) (n : ℕ) : ℚ :=
  a + (n - 1) * d

-- Specify the given conditions
def a1 : ℚ := 10 / 11
def a15 : ℚ := 8 / 9

-- Prove that the eighth term is equal to 89 / 99
theorem arithmetic_sequence_eighth_term :
  ∃ d : ℚ, arithmetic_sequence a1 d 15 = a15 →
             arithmetic_sequence a1 d 8 = 89 / 99 :=
by
  sorry

end arithmetic_sequence_eighth_term_l290_290326


namespace probability_two_white_marbles_l290_290646

theorem probability_two_white_marbles :
  let total_marbles := 12
  let white_marbles := 7
  let red_marbles := 5
  let first_draw_white := (white_marbles : ℚ) / total_marbles
  let second_draw_white := (white_marbles - 1 : ℚ) / (total_marbles - 1)
  (first_draw_white * second_draw_white) = (7 / 22 : ℚ) :=
begin
  let total_marbles := 12,
  let white_marbles := 7,
  let red_marbles := 5,
  let first_draw_white := (white_marbles : ℚ) / total_marbles,
  let second_draw_white := (white_marbles - 1 : ℚ) / (total_marbles - 1),
  calc
    first_draw_white * second_draw_white
    = (7 / 12) * (6 / 11) : by norm_num
    ... = 7 / 22 : by norm_num,
end

end probability_two_white_marbles_l290_290646


namespace number_of_integers_with_even_divisors_count_even_divisors_verification_l290_290403

theorem number_of_integers_with_even_divisors (n : ℕ) (h : n = 100) : 
  (card {x | (x < n) ∧ ∃ k, k * k = x} = 9) → 
  (card {x | (x < n) ∧ ¬(∃ k, k * k = x)} = n - 1 - 9) :=
by
  intro h_squares
  rw h
  trivial

open_locale classical
noncomputable def count_even_divisors_less_than_100 : ℕ :=
  90

theorem count_even_divisors_verification :
  count_even_divisors_less_than_100 = 90 :=
by
  sorry

end number_of_integers_with_even_divisors_count_even_divisors_verification_l290_290403


namespace part1_part2_l290_290116

variable (a b : ℝ)

-- Conditions
axiom abs_a_eq_4 : |a| = 4
axiom abs_b_eq_6 : |b| = 6

-- Part 1: If ab > 0, find the value of a - b
theorem part1 (h : a * b > 0) : a - b = 2 ∨ a - b = -2 := 
by
  -- Proof will go here
  sorry

-- Part 2: If |a + b| = -(a + b), find the value of a + b
theorem part2 (h : |a + b| = -(a + b)) : a + b = -10 ∨ a + b = -2 := 
by
  -- Proof will go here
  sorry

end part1_part2_l290_290116


namespace find_list_price_l290_290630

theorem find_list_price (P : ℝ) (h1 : 0.873 * P = 61.11) : P = 61.11 / 0.873 :=
by
  sorry

end find_list_price_l290_290630


namespace Vasya_numbers_l290_290832

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290832


namespace least_n_froods_l290_290015

def froods_score (n : ℕ) : ℕ := n * (n + 1) / 2
def eating_score (n : ℕ) : ℕ := n ^ 2

theorem least_n_froods :
    ∃ n : ℕ, 0 < n ∧ (froods_score n > eating_score n) ∧ (∀ m : ℕ, 0 < m ∧ m < n → froods_score m ≤ eating_score m) :=
  sorry

end least_n_froods_l290_290015


namespace f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l290_290714

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := (4^x + k * 2^x + 1) / (4^x + 2^x + 1)

-- Part (1)
theorem f_positive_for_all_x (k : ℝ) : (∀ x : ℝ, f x k > 0) ↔ k > -2 := sorry

-- Part (2)
theorem f_min_value_negative_two (k : ℝ) : (∀ x : ℝ, f x k ≥ -2) → k = -8 := sorry

-- Part (3)
theorem f_triangle_sides (k : ℝ) :
  (∀ x1 x2 x3 : ℝ, (f x1 k + f x2 k > f x3 k) ∧ (f x2 k + f x3 k > f x1 k) ∧ (f x3 k + f x1 k > f x2 k)) ↔ (-1/2 ≤ k ∧ k ≤ 4) := sorry

end f_positive_for_all_x_f_min_value_negative_two_f_triangle_sides_l290_290714


namespace find_sin_angle_BAD_l290_290705

def isosceles_right_triangle (A B C : ℝ → ℝ → Prop) (AB BC AC : ℝ) : Prop :=
  AB = 2 ∧ BC = 2 ∧ AC = 2 * Real.sqrt 2

def right_triangle_on_hypotenuse (A C D : ℝ → ℝ → Prop) (AC CD DA : ℝ) (DAC : ℝ) : Prop :=
  AC = 2 * Real.sqrt 2 ∧ CD = DA / 2 ∧ DAC = Real.pi / 6

def equal_perimeters (AC CD DA : ℝ) : Prop := 
  AC + CD + DA = 4 + 2 * Real.sqrt 2

theorem find_sin_angle_BAD :
  ∀ (A B C D : ℝ → ℝ → Prop) (AB BC AC CD DA : ℝ),
  isosceles_right_triangle A B C AB BC AC →
  right_triangle_on_hypotenuse A C D AC CD DA (Real.pi / 6) →
  equal_perimeters AC CD DA →
  Real.sin (2 * (Real.pi / 4 + Real.pi / 6)) = 1 / 2 :=
by
  intros
  sorry

end find_sin_angle_BAD_l290_290705


namespace minnie_mounts_time_period_l290_290243

theorem minnie_mounts_time_period (M D : ℕ) 
  (mickey_daily_mounts_eq : 2 * M - 6 = 14)
  (minnie_mounts_per_day_eq : M = D + 3) : 
  D = 7 := 
by
  sorry

end minnie_mounts_time_period_l290_290243


namespace toothpicks_15_l290_290487

def toothpicks (n : ℕ) : ℕ :=
  match n with
  | 0 => 0  -- Not used, placeholder for 1-based indexing.
  | 1 => 3
  | k+1 => let p := toothpicks k
           2 + if k % 2 = 0 then 1 else 0 + p

theorem toothpicks_15 : toothpicks 15 = 38 :=
by
  sorry

end toothpicks_15_l290_290487


namespace f_no_zeros_in_interval_f_zeros_in_interval_l290_290713

noncomputable def f (x : ℝ) : ℝ := (1 / 3) * x - Real.log x

theorem f_no_zeros_in_interval (x : ℝ) (hx1 : x > 1 / Real.exp 1) (hx2 : x < 1) :
  f x ≠ 0 := sorry

theorem f_zeros_in_interval (h1 : 1 < e) (x_exists : ∃ x, 1 < x ∧ x < Real.exp 1 ∧ f x = 0) :
  true := sorry

end f_no_zeros_in_interval_f_zeros_in_interval_l290_290713


namespace rebecca_eggs_l290_290758

theorem rebecca_eggs (groups eggs_per_group : ℕ) (h1 : groups = 3) (h2 : eggs_per_group = 6) : 
  (groups * eggs_per_group = 18) :=
by
  sorry

end rebecca_eggs_l290_290758


namespace total_numbers_l290_290960

theorem total_numbers (N : ℕ) (sum_total : ℝ) (avg_total : ℝ) (avg1 : ℝ) (avg2 : ℝ) (avg3 : ℝ) :
  avg_total = 6.40 → avg1 = 6.2 → avg2 = 6.1 → avg3 = 6.9 →
  sum_total = 2 * avg1 + 2 * avg2 + 2 * avg3 →
  N = sum_total / avg_total →
  N = 6 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end total_numbers_l290_290960


namespace y_value_l290_290884

theorem y_value (y : ℕ) : 8^3 + 8^3 + 8^3 + 8^3 = 2^y → y = 11 := 
by 
  sorry

end y_value_l290_290884


namespace history_students_count_l290_290514

theorem history_students_count
  (total_students : ℕ)
  (sample_students : ℕ)
  (physics_students_sampled : ℕ)
  (history_students_sampled : ℕ)
  (x : ℕ)
  (H1 : total_students = 1500)
  (H2 : sample_students = 120)
  (H3 : physics_students_sampled = 80)
  (H4 : history_students_sampled = sample_students - physics_students_sampled)
  (H5 : x = 1500 * history_students_sampled / sample_students) :
  x = 500 :=
by
  sorry

end history_students_count_l290_290514


namespace pair_factorial_power_of_5_l290_290252

theorem pair_factorial_power_of_5 (a b : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b)
  (h1 : ∃ k : ℕ, a! + b = 5^k) (h2 : ∃ m : ℕ, b! + a = 5^m) : a = 5 ∧ b = 5 :=
begin
  sorry
end

end pair_factorial_power_of_5_l290_290252


namespace circle_equation_solution_l290_290290

theorem circle_equation_solution
  (a : ℝ)
  (h1 : a ^ 2 = a + 2)
  (h2 : (2 * a / (a + 2)) ^ 2 - 4 * a / (a + 2) > 0) : 
  a = -1 := 
sorry

end circle_equation_solution_l290_290290


namespace average_rounds_rounded_is_3_l290_290474

-- Definitions based on conditions
def golfers : List ℕ := [3, 4, 3, 6, 2, 4]
def rounds : List ℕ := [0, 1, 2, 3, 4, 5]

noncomputable def total_rounds : ℕ :=
  List.sum (List.zipWith (λ g r => g * r) golfers rounds)

def total_golfers : ℕ := List.sum golfers

noncomputable def average_rounds : ℕ :=
  Int.natAbs (Int.ofNat total_rounds / total_golfers).toNat

theorem average_rounds_rounded_is_3 : average_rounds = 3 := by
  sorry

end average_rounds_rounded_is_3_l290_290474


namespace no_k_for_linear_function_not_in_second_quadrant_l290_290378

theorem no_k_for_linear_function_not_in_second_quadrant :
  ¬∃ k : ℝ, ∀ x < 0, (k-1)*x + k ≤ 0 :=
by
  sorry

end no_k_for_linear_function_not_in_second_quadrant_l290_290378


namespace solution_x_x_sub_1_eq_x_l290_290775

theorem solution_x_x_sub_1_eq_x (x : ℝ) : x * (x - 1) = x ↔ (x = 0 ∨ x = 2) :=
by {
  sorry
}

end solution_x_x_sub_1_eq_x_l290_290775


namespace train_crossing_time_l290_290661

theorem train_crossing_time (length_of_train : ℕ) (speed_kmh : ℕ) (speed_ms : ℕ) 
  (conversion_factor : speed_kmh * 1000 / 3600 = speed_ms) 
  (H1 : length_of_train = 180) 
  (H2 : speed_kmh = 72) 
  (H3 : speed_ms = 20) 
  : length_of_train / speed_ms = 9 := by
  sorry

end train_crossing_time_l290_290661


namespace xiaoqiang_expected_score_l290_290654

noncomputable def expected_score_xiaoqiang : ℚ :=
by
  let n := 25
  let p := 0.8
  let correct_points := 4
  let X := binomial n p -- Binomial distribution
  let score := correct_points * X
  have expected_score : E(score) = correct_points * E(X) := sorry
  have E_X : E(X) = n * p := sorry
  have result : expected_score_xiaoqiang = correct_points * n * p := sorry
  exact sorry

theorem xiaoqiang_expected_score : expected_score_xiaoqiang = 80 := sorry

end xiaoqiang_expected_score_l290_290654


namespace problem1_problem2_problem3_problem4_l290_290532

theorem problem1 : 25 - 9 + (-12) - (-7) = 4 := by
  sorry

theorem problem2 : (1 / 9) * (-2)^3 / ((2 / 3)^2) = -2 := by
  sorry

theorem problem3 : ((5 / 12) + (2 / 3) - (3 / 4)) * (-12) = -4 := by
  sorry

theorem problem4 : -(1^4) + (-2) / (-1/3) - |(-9)| = -4 := by
  sorry

end problem1_problem2_problem3_problem4_l290_290532


namespace probability_not_all_dice_same_l290_290056

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l290_290056


namespace distance_before_rest_l290_290336

theorem distance_before_rest (total_distance after_rest_distance : ℝ) (h1 : total_distance = 1) (h2 : after_rest_distance = 0.25) :
  total_distance - after_rest_distance = 0.75 :=
by sorry

end distance_before_rest_l290_290336


namespace length_RS_l290_290426

open Real

-- Given definitions and conditions
def PQ : ℝ := 10
def PR : ℝ := 10
def QR : ℝ := 5
def PS : ℝ := 13

-- Prove the length of RS
theorem length_RS : ∃ (RS : ℝ), RS = 6.17362 := by
  sorry

end length_RS_l290_290426


namespace solution_set_of_inequality_l290_290328

theorem solution_set_of_inequality :
  { x : ℝ | -x^2 + 3*x + 4 > 0 } = { x : ℝ | -1 < x ∧ x < 4 } := 
sorry

end solution_set_of_inequality_l290_290328


namespace years_of_interest_l290_290998

noncomputable def principal : ℝ := 2600
noncomputable def interest_difference : ℝ := 78

theorem years_of_interest (R : ℝ) (N : ℝ) (h : (principal * (R + 1) * N / 100) - (principal * R * N / 100) = interest_difference) : N = 3 :=
sorry

end years_of_interest_l290_290998


namespace brianne_january_savings_l290_290362

theorem brianne_january_savings (S : ℝ) (h : 16 * S = 160) : S = 10 :=
sorry

end brianne_january_savings_l290_290362


namespace floor_sum_value_l290_290741

theorem floor_sum_value (a b c d : ℝ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) (h_pos_d : d > 0)
(h1 : a^2 + b^2 = 2016) (h2 : c^2 + d^2 = 2016) (h3 : a * c = 1024) (h4 : b * d = 1024) :
  ⌊a + b + c + d⌋ = 127 := sorry

end floor_sum_value_l290_290741


namespace largest_distance_between_spheres_l290_290050

theorem largest_distance_between_spheres :
  let O1 := (3, -14, 8)
  let O2 := (-9, 5, -12)
  let d := Real.sqrt ((3 + 9)^2 + (-14 - 5)^2 + (8 + 12)^2)
  let r1 := 24
  let r2 := 50
  r1 + d + r2 = Real.sqrt 905 + 74 :=
by
  intro O1 O2 d r1 r2
  sorry

end largest_distance_between_spheres_l290_290050


namespace range_of_a_l290_290726

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, 3 < x ∧ x < 4 ∧ ax^2 - 4*a*x - 2 > 0) ↔ a < -2/3 :=
sorry

end range_of_a_l290_290726


namespace distance_between_points_l290_290231

theorem distance_between_points (points : Fin 7 → ℝ × ℝ) (diameter : ℝ)
  (h_diameter : diameter = 1)
  (h_points_in_circle : ∀ i : Fin 7, (points i).fst^2 + (points i).snd^2 ≤ (diameter / 2)^2) :
  ∃ (i j : Fin 7), i ≠ j ∧ (dist (points i) (points j) ≤ 1 / 2) := 
by
  sorry

end distance_between_points_l290_290231


namespace maximal_difference_of_areas_l290_290728

-- Given:
-- A circle of radius R
-- A chord of length 2x is drawn perpendicular to the diameter of the circle
-- The endpoints of this chord are connected to the endpoints of the diameter
-- We need to prove that under these conditions, the length of the chord 2x that maximizes the difference in areas of the triangles is R √ 2

theorem maximal_difference_of_areas (R x : ℝ) (h : 2 * x = R * Real.sqrt 2) :
  2 * x = R * Real.sqrt 2 :=
by
  sorry

end maximal_difference_of_areas_l290_290728


namespace relationship_a_b_c_l290_290311

open Real

theorem relationship_a_b_c (x : ℝ) (hx1 : e < x) (hx2 : x < e^2)
  (a : ℝ) (ha : a = log x)
  (b : ℝ) (hb : b = (1 / 2) ^ log x)
  (c : ℝ) (hc : c = exp (log x)) :
  c > a ∧ a > b :=
by {
  -- we state the theorem without providing the proof for now
  sorry
}

end relationship_a_b_c_l290_290311


namespace derivative_of_f_tangent_line_at_point_l290_290916

noncomputable def f (k : ℝ) (x : ℝ) := k * (x - 1) * Real.exp(x) + x^2

theorem derivative_of_f (k : ℝ) : 
  (Real.deriv (λ x => f k x)) = (λ x => k * x * Real.exp(x) + 2 * x) :=
by
  sorry

theorem tangent_line_at_point (k : ℝ) (h : k = -1 / Real.exp(1)) (y : ℝ) :
  ∃ m b, (∀ x, y = m * x + b) ∧ m = 1 ∧ b = 0 :=
by
  sorry

end derivative_of_f_tangent_line_at_point_l290_290916


namespace total_balloons_l290_290788

-- Defining constants for the number of balloons each person has
def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

-- Theorem stating the total number of balloons
theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  simp [tom_balloons, sara_balloons]
  sorry

end total_balloons_l290_290788


namespace vasya_numbers_l290_290821

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290821


namespace remainder_is_zero_l290_290196

def f (x : ℝ) : ℝ := x^3 - 5 * x^2 + 2 * x + 8

theorem remainder_is_zero : f 2 = 0 := by
  sorry

end remainder_is_zero_l290_290196


namespace five_dice_not_all_same_number_l290_290052
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l290_290052


namespace angie_bought_18_pretzels_l290_290667

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end angie_bought_18_pretzels_l290_290667


namespace grace_pennies_l290_290512

theorem grace_pennies (dime_value nickel_value : ℕ) (dimes nickels : ℕ) 
  (h₁ : dime_value = 10) (h₂ : nickel_value = 5) (h₃ : dimes = 10) (h₄ : nickels = 10) : 
  dimes * dime_value + nickels * nickel_value = 150 := 
by 
  sorry

end grace_pennies_l290_290512


namespace tangent_line_eq_l290_290039

/-- The equation of the tangent line to the curve y = 2x * tan x at the point x = π/4 is 
    (2 + π/2) * x - y - π^2/4 = 0. -/
theorem tangent_line_eq : ∀ x y : ℝ, 
  (y = 2 * x * Real.tan x) →
  (x = Real.pi / 4) →
  ((2 + Real.pi / 2) * x - y - Real.pi^2 / 4 = 0) :=
by
  intros x y h_curve h_point
  sorry

end tangent_line_eq_l290_290039


namespace dice_product_144_probability_l290_290201

theorem dice_product_144_probability :
  let S := {n : ℕ | 1 ≤ n ∧ n ≤ 6},
      events := {abc : ℕ × ℕ × ℕ | abc.1 ∈ S ∧ abc.2 ∈ S ∧ abc.3 ∈ S ∧ abc.1 * abc.2 * abc.3 = 144} in
  (|events| : ℝ) / 216 = 1 / 72 :=
by
  -- Assumption of finite set cardinality and mention of corresponding probabilities can be filled here
  sorry

end dice_product_144_probability_l290_290201


namespace value_of_stamp_collection_l290_290250

theorem value_of_stamp_collection 
  (n m : ℕ) (v_m : ℝ)
  (hn : n = 18) 
  (hm : m = 6)
  (hv_m : v_m = 15)
  (uniform_value : ∀ (k : ℕ), k ≤ m → v_m / m = v_m / k):
  ∃ v_total : ℝ, v_total = 45 :=
by 
  sorry

end value_of_stamp_collection_l290_290250


namespace geometric_sequence_sum_l290_290158

open Nat

noncomputable def geometric_sum (a q n : ℕ) : ℕ :=
  a * (1 - q^n) / (1 - q)

theorem geometric_sequence_sum (S : ℕ → ℕ) (q a₁ : ℕ)
  (h_q: q = 2)
  (h_S5: S 5 = 1)
  (h_S: ∀ n, S n = a₁ * (1 - q^n) / (1 - q)) :
  S 10 = 33 :=
by
  sorry

end geometric_sequence_sum_l290_290158


namespace Vasya_numbers_l290_290830

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290830


namespace intersection_is_correct_l290_290264

-- Define the sets A and B based on given conditions
def setA : Set ℝ := {x | ∃ y, y = Real.log (x - 2)}
def setB : Set ℝ := {y | ∃ x, y = Real.sqrt x + 4}

-- Define the intersection of sets A and B
def intersection : Set ℝ := {z | z ≥ 4}

-- The theorem stating that the intersection of A and B is exactly the set [4, +∞)
theorem intersection_is_correct : {x | ∃ y, y = Real.log (x - 2)} ∩ {y | ∃ x, y = Real.sqrt x + 4} = {z | z ≥ 4} :=
by
  sorry

end intersection_is_correct_l290_290264


namespace tangent_circle_distance_proof_l290_290004

noncomputable def tangent_circle_distance (R r : ℝ) (tangent_type : String) : ℝ :=
  if tangent_type = "external" then R + r else R - r

theorem tangent_circle_distance_proof (R r : ℝ) (tangent_type : String) (hR : R = 4) (hr : r = 3) :
  tangent_circle_distance R r tangent_type = 7 ∨ tangent_circle_distance R r tangent_type = 1 := by
  sorry

end tangent_circle_distance_proof_l290_290004


namespace greatest_t_value_exists_l290_290685

theorem greatest_t_value_exists (t : ℝ) : (∃ t, (t^2 - t - 56) / (t - 8) = 3 / (t + 5)) → ∃ t, (t = -4) := 
by
  intro h
  -- Insert proof here
  sorry

end greatest_t_value_exists_l290_290685


namespace calculate_area_ADC_l290_290301

def area_AD (BD DC : ℕ) (area_ABD : ℕ) := 
  area_ABD * DC / BD

theorem calculate_area_ADC
  (BD DC : ℕ) 
  (h_ratio : BD = 5 * DC / 2)
  (area_ABD : ℕ)
  (h_area_ABD : area_ABD = 35) :
  area_AD BD DC area_ABD = 14 := 
by 
  sorry

end calculate_area_ADC_l290_290301


namespace parabola_intersects_x_axis_two_points_l290_290140

theorem parabola_intersects_x_axis_two_points (m : ℝ) : 
    ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ mx^2 + (m-3)*x - 1 = 0 :=
by
  sorry

end parabola_intersects_x_axis_two_points_l290_290140


namespace rotation_image_of_D_l290_290917

def rotate_90_clockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.snd, -p.fst)

theorem rotation_image_of_D :
  rotate_90_clockwise (-3, 2) = (2, 3) :=
by
  sorry

end rotation_image_of_D_l290_290917


namespace Bobby_has_27_pairs_l290_290358

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end Bobby_has_27_pairs_l290_290358


namespace cubic_sum_l290_290926

theorem cubic_sum (a b c : ℝ) (h1 : a + b + c = 12) (h2 : ab + ac + bc = 30) :
  a^3 + b^3 + c^3 - 3 * a * b * c = 648 :=
  sorry

end cubic_sum_l290_290926


namespace g_of_1001_l290_290618

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x * y) + x = x * g y + g x
axiom g_of_1 : g 1 = -3

theorem g_of_1001 : g 1001 = -2001 := 
by sorry

end g_of_1001_l290_290618


namespace factorial_ratio_l290_290840

theorem factorial_ratio : Nat.factorial 16 / (Nat.factorial 6 * Nat.factorial 10) = 5120 := by
  sorry

end factorial_ratio_l290_290840


namespace Nell_initial_cards_l290_290030

theorem Nell_initial_cards (given_away : ℕ) (now_has : ℕ) : 
  given_away = 276 → now_has = 252 → (now_has + given_away) = 528 :=
by
  intros h_given_away h_now_has
  sorry

end Nell_initial_cards_l290_290030


namespace correct_option_l290_290248

-- Definition of the conditions
def conditionA : Prop := (Real.sqrt ((-1 : ℝ)^2) = 1)
def conditionB : Prop := (Real.sqrt ((-1 : ℝ)^2) = -1)
def conditionC : Prop := (Real.sqrt (-(1^2) : ℝ) = 1)
def conditionD : Prop := (Real.sqrt (-(1^2) : ℝ) = -1)

-- Proving the correct condition
theorem correct_option : conditionA := by
  sorry

end correct_option_l290_290248


namespace product_of_triangle_areas_not_end_in_1988_l290_290509

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end product_of_triangle_areas_not_end_in_1988_l290_290509


namespace sequence_terms_proof_l290_290012

theorem sequence_terms_proof (P Q R T U V W : ℤ) (S : ℤ) 
  (h1 : S = 10) 
  (h2 : P + Q + R + S = 40) 
  (h3 : Q + R + S + T = 40) 
  (h4 : R + S + T + U = 40) 
  (h5 : S + T + U + V = 40) 
  (h6 : T + U + V + W = 40) : 
  P + W = 40 := 
by 
  have h7 : P + Q + R + 10 = 40 := by rwa [h1] at h2
  have h8 : Q + R + 10 + T = 40 := by rwa [h1] at h3
  have h9 : R + 10 + T + U = 40 := by rwa [h1] at h4
  have h10 : 10 + T + U + V = 40 := by rwa [h1] at h5
  have h11 : T + U + V + W = 40 := h6
  sorry

end sequence_terms_proof_l290_290012


namespace quadrilateral_areas_product_l290_290508

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end quadrilateral_areas_product_l290_290508


namespace percentage_of_red_non_honda_cars_l290_290184

-- Define the conditions
def total_cars : ℕ := 900
def honda_cars : ℕ := 500
def red_per_100_honda_cars : ℕ := 90
def red_percent_total := 60

-- Define the question we want to answer
theorem percentage_of_red_non_honda_cars : 
  let red_honda_cars := (red_per_100_honda_cars / 100 : ℚ) * honda_cars
  let total_red_cars := (red_percent_total / 100 : ℚ) * total_cars
  let red_non_honda_cars := total_red_cars - red_honda_cars
  let non_honda_cars := total_cars - honda_cars
  (red_non_honda_cars / non_honda_cars) * 100 = (22.5 : ℚ) :=
by
  sorry

end percentage_of_red_non_honda_cars_l290_290184


namespace vasya_numbers_l290_290824

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290824


namespace arithmetic_progression_first_three_terms_l290_290894

theorem arithmetic_progression_first_three_terms 
  (S_n : ℤ) (d a_1 a_2 a_3 a_5 : ℤ)
  (h1 : S_n = 112) 
  (h2 : (a_1 + d) * d = 30)
  (h3 : (a_1 + 2 * d) + (a_1 + 4 * d) = 32) 
  (h4 : ∀ (n : ℕ), S_n = (n * (2 * a_1 + (n - 1) * d)) / 2) : 
  ((a_1 = 7 ∧ a_2 = 10 ∧ a_3 = 13) ∨ (a_1 = 1 ∧ a_2 = 6 ∧ a_3 = 11)) :=
sorry

end arithmetic_progression_first_three_terms_l290_290894


namespace initial_number_of_persons_l290_290961

theorem initial_number_of_persons (n : ℕ) (h1 : ∀ n, (2.5 : ℝ) * n = 20) : n = 8 := sorry

end initial_number_of_persons_l290_290961


namespace quadratic_roots_sum_square_l290_290749

theorem quadratic_roots_sum_square (u v : ℝ) 
  (h1 : u^2 - 5*u + 3 = 0) (h2 : v^2 - 5*v + 3 = 0) 
  (h3 : u ≠ v) : u^2 + v^2 + u*v = 22 := 
by
  sorry

end quadratic_roots_sum_square_l290_290749


namespace minimum_ab_l290_290707

theorem minimum_ab (a b : ℝ) (ha : a > 1) (hb : b > 1) (hab : ab + 2 = 2 * (a + b)) : ab ≥ 6 + 4 * Real.sqrt 2 :=
by
  sorry

end minimum_ab_l290_290707


namespace find_other_root_of_quadratic_l290_290554

theorem find_other_root_of_quadratic (m x_1 x_2 : ℝ) 
  (h_root1 : x_1 = 1) (h_eqn : ∀ x, x^2 - 4 * x + m = 0) : x_2 = 3 :=
by
  sorry

end find_other_root_of_quadratic_l290_290554


namespace vasya_numbers_l290_290819

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290819


namespace impossible_permuted_sum_l290_290303

def isPermutation (X Y : ℕ) : Prop :=
  -- Define what it means for two numbers to be permutations of each other.
  sorry

theorem impossible_permuted_sum (X Y : ℕ) (h1 : isPermutation X Y) (h2 : X + Y = (10^1111 - 1)) : false :=
  sorry

end impossible_permuted_sum_l290_290303


namespace quadratic_one_pos_one_neg_l290_290078

theorem quadratic_one_pos_one_neg (a : ℝ) : 
  (a < -1) → (∃ x1 x2 : ℝ, x1 * x2 < 0 ∧ x1 + x2 > 0 ∧ (x1^2 + x1 + a = 0 ∧ x2^2 + x2 + a = 0)) :=
sorry

end quadratic_one_pos_one_neg_l290_290078


namespace helga_extra_hours_last_friday_l290_290395

theorem helga_extra_hours_last_friday
  (weekly_articles : ℕ)
  (extra_hours_thursday : ℕ)
  (extra_articles_thursday : ℕ)
  (extra_articles_friday : ℕ)
  (articles_per_half_hour : ℕ)
  (half_hours_per_hour : ℕ)
  (usual_articles_per_day : ℕ)
  (days_per_week : ℕ)
  (articles_last_thursday_plus_friday : ℕ)
  (total_articles : ℕ) :
  (weekly_articles = (usual_articles_per_day * days_per_week)) →
  (extra_hours_thursday = 2) →
  (articles_per_half_hour = 5) →
  (half_hours_per_hour = 2) →
  (usual_articles_per_day = (articles_per_half_hour * 8)) →
  (extra_articles_thursday = (articles_per_half_hour * (extra_hours_thursday * half_hours_per_hour))) →
  (articles_last_thursday_plus_friday = weekly_articles + extra_articles_thursday) →
  (total_articles = 250) →
  (extra_articles_friday = total_articles - articles_last_thursday_plus_friday) →
  (extra_articles_friday = 30) →
  ((extra_articles_friday / articles_per_half_hour) = 6) →
  (3 = (6 / half_hours_per_hour)) :=
by
  intro hw1 hw2 hw3 hw4 hw5 hw6 hw7 hw8 hw9 hw10
  sorry

end helga_extra_hours_last_friday_l290_290395


namespace actual_distance_travelled_l290_290585

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end actual_distance_travelled_l290_290585


namespace lawnmower_blade_cost_l290_290652

theorem lawnmower_blade_cost (x : ℕ) : 4 * x + 7 = 39 → x = 8 :=
by
  sorry

end lawnmower_blade_cost_l290_290652


namespace outdoor_tables_count_l290_290624

variable (numIndoorTables : ℕ) (chairsPerIndoorTable : ℕ) (totalChairs : ℕ)
variable (chairsPerOutdoorTable : ℕ)

theorem outdoor_tables_count 
  (h1 : numIndoorTables = 8) 
  (h2 : chairsPerIndoorTable = 3) 
  (h3 : totalChairs = 60) 
  (h4 : chairsPerOutdoorTable = 3) :
  ∃ (numOutdoorTables : ℕ), numOutdoorTables = 12 := by
  admit

end outdoor_tables_count_l290_290624


namespace expand_product_l290_290680

theorem expand_product (x : ℝ) :
  (3 * x + 4) * (2 * x - 5) = 6 * x^2 - 7 * x - 20 :=
sorry

end expand_product_l290_290680


namespace number_of_sheets_l290_290307

theorem number_of_sheets (S E : ℕ) (h1 : S - E = 60) (h2 : 5 * E = S) : S = 150 := by
  sorry

end number_of_sheets_l290_290307


namespace actual_distance_travelled_l290_290586

theorem actual_distance_travelled :
  ∃ (D : ℝ), (D / 10 = (D + 20) / 14) ∧ D = 50 :=
by
  sorry

end actual_distance_travelled_l290_290586


namespace swimming_pool_cost_l290_290785

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l290_290785


namespace percent_problem_l290_290288

theorem percent_problem (x : ℝ) (h : 0.20 * x = 60) : 0.80 * x = 240 :=
sorry

end percent_problem_l290_290288


namespace min_value_xy_l290_290535

-- Defining the operation ⊗
def otimes (a b : ℝ) : ℝ := a * b - a - b

theorem min_value_xy (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : otimes x y = 3) : 9 ≤ x * y := by
  sorry

end min_value_xy_l290_290535


namespace pool_fill_time_l290_290113

-- Definitions according to conditions
def pool_volume : ℝ := 15000  -- pool volume in gallons

def hose_rate1 : ℝ := 2       -- rate of first type of hoses in gallons per minute
def hose_rate2 : ℝ := 3       -- rate of second type of hoses in gallons per minute

def hoses_count1 : ℕ := 2     -- number of first type of hoses
def hoses_count2 : ℕ := 2     -- number of second type of hoses

-- The main theorem to be proved
theorem pool_fill_time (volume : ℝ) (rate1 rate2 : ℝ) (count1 count2 : ℕ) :
  let total_rate := (rate1 * count1) + (rate2 * count2) in
  let time_minutes := volume / total_rate in
  let time_hours := time_minutes / 60 in
  volume = pool_volume →
  rate1 = hose_rate1 →
  rate2 = hose_rate2 →
  count1 = hoses_count1 →
  count2 = hoses_count2 →
  time_hours = 25 := 
sorry

end pool_fill_time_l290_290113


namespace t_plus_inv_t_eq_three_l290_290908

theorem t_plus_inv_t_eq_three {t : ℝ} (h : t^2 - 3 * t + 1 = 0) (hnz : t ≠ 0) : t + 1 / t = 3 :=
sorry

end t_plus_inv_t_eq_three_l290_290908


namespace wood_rope_length_equivalence_l290_290440

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l290_290440


namespace sum_factors_30_less_15_l290_290639

theorem sum_factors_30_less_15 : (1 + 2 + 3 + 5 + 6 + 10) = 27 := by
  sorry

end sum_factors_30_less_15_l290_290639


namespace line_perpendicular_l290_290296

theorem line_perpendicular (m : ℝ) : 
  -- Conditions
  (∀ x y : ℝ, x - 2 * y + 5 = 0 → y = 1/2 * x + 5/2) →  -- Slope of the first line
  (∀ x y : ℝ, 2 * x + m * y - 6 = 0 → y = -2/m * x + 6/m) →  -- Slope of the second line
  -- Perpendicular condition
  ((1/2) * (-2/m) = -1) →
  -- Conclusion
  m = 1 := 
sorry

end line_perpendicular_l290_290296


namespace projections_possibilities_l290_290913

-- Define the conditions: a and b are non-perpendicular skew lines, and α is a plane
variables {a b : Line} (α : Plane)

-- Non-perpendicular skew lines definition (external knowledge required for proper setup if not inbuilt)
def non_perpendicular_skew_lines (a b : Line) : Prop := sorry

-- Projections definition (external knowledge required for proper setup if not inbuilt)
def projections (a : Line) (α : Plane) : Line := sorry

-- The projections result in new conditions
def projected_parallel (a b : Line) (α : Plane) : Prop := sorry
def projected_perpendicular (a b : Line) (α : Plane) : Prop := sorry
def projected_same_line (a b : Line) (α : Plane) : Prop := sorry
def projected_line_and_point (a b : Line) (α : Plane) : Prop := sorry

-- Given the given conditions
variables (ha : non_perpendicular_skew_lines a b)

-- Prove the resultant conditions where the projections satisfy any 3 of the listed possibilities: parallel, perpendicular, line and point.
theorem projections_possibilities :
    (projected_parallel a b α ∨ projected_perpendicular a b α ∨ projected_line_and_point a b α) ∧
    ¬ projected_same_line a b α := sorry

end projections_possibilities_l290_290913


namespace probability_of_sum_less_than_product_l290_290790

-- Define the problem conditions
def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the event condition
def is_valid_pair (a b : ℕ) : Prop := (a ∈ set_of_numbers) ∧ (b ∈ set_of_numbers) ∧ (a * b > a + b)

-- Count the number of valid pairs
noncomputable def count_valid_pairs : ℕ :=
  set_of_numbers.sum (λ a, set_of_numbers.filter (is_valid_pair a).card)

-- Count the total possible pairs
noncomputable def total_pairs : ℕ :=
  set_of_numbers.card * set_of_numbers.card

-- Calculate the probability
noncomputable def probability : ℚ :=
  (count_valid_pairs : ℚ) / total_pairs

-- State the theorem
theorem probability_of_sum_less_than_product :
  probability = 25 / 49 :=
by sorry

end probability_of_sum_less_than_product_l290_290790


namespace seashells_left_l290_290606

theorem seashells_left (initial_seashells : ℕ) (given_seashells : ℕ) (remaining_seashells : ℕ) :
  initial_seashells = 75 → given_seashells = 18 → remaining_seashells = initial_seashells - given_seashells → remaining_seashells = 57 :=
by
  intros h1 h2 h3
  rw [h1, h2] at h3
  exact h3

end seashells_left_l290_290606


namespace compute_expression_l290_290238

theorem compute_expression : 12 * (1 / 17) * 34 = 24 := 
by {
  sorry
}

end compute_expression_l290_290238


namespace vasya_numbers_l290_290808

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290808


namespace min_value_of_expression_l290_290745

noncomputable def minValueExpr (a b c : ℝ) : ℝ :=
  a^2 + 9 * a * b + 9 * b^2 + 3 * c^2

theorem min_value_of_expression (a b c : ℝ) (h : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b * c = 27) :
  minValueExpr a b c >= 60 :=
by
  sorry

end min_value_of_expression_l290_290745


namespace find_preimage_l290_290381

def mapping (x y : ℝ) : ℝ × ℝ :=
  (x + y, x - y)

theorem find_preimage :
  mapping 2 1 = (3, 1) :=
by
  sorry

end find_preimage_l290_290381


namespace expression_equals_neg_one_l290_290880

theorem expression_equals_neg_one (b y : ℝ) (hb : b ≠ 0) (h₁ : y ≠ b) (h₂ : y ≠ -b) :
  ( (b / (b + y) + y / (b - y)) / (y / (b + y) - b / (b - y)) ) = -1 :=
sorry

end expression_equals_neg_one_l290_290880


namespace one_of_a_b_c_is_one_l290_290561

theorem one_of_a_b_c_is_one (a b c : ℝ) (h1 : a * b * c = 1) (h2 : a + b + c = (1 / a) + (1 / b) + (1 / c)) :
  a = 1 ∨ b = 1 ∨ c = 1 :=
by
  sorry -- proof to be filled in

end one_of_a_b_c_is_one_l290_290561


namespace probability_of_sum_less_than_product_l290_290791

-- Define the problem conditions
def set_of_numbers : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

-- Define the event condition
def is_valid_pair (a b : ℕ) : Prop := (a ∈ set_of_numbers) ∧ (b ∈ set_of_numbers) ∧ (a * b > a + b)

-- Count the number of valid pairs
noncomputable def count_valid_pairs : ℕ :=
  set_of_numbers.sum (λ a, set_of_numbers.filter (is_valid_pair a).card)

-- Count the total possible pairs
noncomputable def total_pairs : ℕ :=
  set_of_numbers.card * set_of_numbers.card

-- Calculate the probability
noncomputable def probability : ℚ :=
  (count_valid_pairs : ℚ) / total_pairs

-- State the theorem
theorem probability_of_sum_less_than_product :
  probability = 25 / 49 :=
by sorry

end probability_of_sum_less_than_product_l290_290791


namespace no_unsatisfactory_grades_l290_290170

theorem no_unsatisfactory_grades (total_students : ℕ)
  (top_marks : ℕ) (average_marks : ℕ) (good_marks : ℕ)
  (h1 : top_marks = total_students / 6)
  (h2 : average_marks = total_students / 3)
  (h3 : good_marks = total_students / 2) :
  total_students = top_marks + average_marks + good_marks := by
  sorry

end no_unsatisfactory_grades_l290_290170


namespace ratio_of_board_pieces_l290_290645

theorem ratio_of_board_pieces (S L : ℕ) (hS : S = 23) (hTotal : S + L = 69) : L / S = 2 :=
by
  sorry

end ratio_of_board_pieces_l290_290645


namespace remainder_of_c_plus_d_l290_290748

theorem remainder_of_c_plus_d (c d : ℕ) (k l : ℕ) 
  (hc : c = 120 * k + 114) 
  (hd : d = 180 * l + 174) : 
  (c + d) % 60 = 48 := 
by sorry

end remainder_of_c_plus_d_l290_290748


namespace store_paid_price_l290_290632

theorem store_paid_price (selling_price : ℕ) (less_amount : ℕ) 
(h1 : selling_price = 34) (h2 : less_amount = 8) : ∃ p : ℕ, p = selling_price - less_amount ∧ p = 26 := 
by
  sorry

end store_paid_price_l290_290632


namespace discriminant_eq_M_l290_290148

theorem discriminant_eq_M (a b c x0 : ℝ) (h1: a ≠ 0) (h2: a * x0^2 + b * x0 + c = 0) :
  (b^2 - 4 * a * c) = (2 * a * x0 + b)^2 :=
by
  sorry

end discriminant_eq_M_l290_290148


namespace polynomial_expansion_l290_290080

theorem polynomial_expansion (x : ℝ) : 
  (1 + x^2) * (1 - x^3) = 1 + x^2 - x^3 - x^5 :=
by sorry

end polynomial_expansion_l290_290080


namespace average_student_headcount_is_correct_l290_290531

noncomputable def average_student_headcount : ℕ :=
  let a := 11000
  let b := 10200
  let c := 10800
  let d := 11300
  (a + b + c + d) / 4

theorem average_student_headcount_is_correct :
  average_student_headcount = 10825 :=
by
  -- Proof will go here
  sorry

end average_student_headcount_is_correct_l290_290531


namespace total_black_balls_l290_290929

-- Conditions
def number_of_white_balls (B : ℕ) : ℕ := 6 * B

def total_balls (B : ℕ) : ℕ := B + number_of_white_balls B

-- Theorem to prove
theorem total_black_balls (h : total_balls B = 56) : B = 8 :=
by
  sorry

end total_black_balls_l290_290929


namespace tank_capacity_l290_290351

theorem tank_capacity (C : ℕ) (h₁ : C = 785) :
  360 - C / 4 - C / 8 = C / 12 :=
by 
  -- Assuming h₁: C = 785
  have h₁: C = 785 := by exact h₁
  -- Provide proof steps here (not required for the task)
  sorry

end tank_capacity_l290_290351


namespace geometric_sum_of_first_five_terms_l290_290935

theorem geometric_sum_of_first_five_terms (a_1 l : ℝ)
  (h₁ : ∀ r : ℝ, (2 * l = a_1 * (r - 1) ^ 2)) 
  (h₂ : ∀ (r : ℝ), a_1 * r ^ 3 = 8 * a_1):
  (a_1 + a_1 * (2 : ℝ) + a_1 * (2 : ℝ)^2 + a_1 * (2 : ℝ)^3 + a_1 * (2 : ℝ)^4) = 62 :=
by
  sorry

end geometric_sum_of_first_five_terms_l290_290935


namespace probability_two_students_same_school_l290_290636

/-- Definition of the problem conditions -/
def total_students : ℕ := 3
def total_schools : ℕ := 4
def total_basic_events : ℕ := total_schools ^ total_students
def favorable_events : ℕ := 36

/-- Theorem stating the probability of exactly two students choosing the same school -/
theorem probability_two_students_same_school : 
  favorable_events / (total_schools ^ total_students) = 9 / 16 := 
  sorry

end probability_two_students_same_school_l290_290636


namespace play_number_of_children_l290_290095

theorem play_number_of_children (A C : ℕ) (ticket_price_adult : ℕ) (ticket_price_child : ℕ)
    (total_people : ℕ) (total_money : ℕ)
    (h1 : ticket_price_adult = 8)
    (h2 : ticket_price_child = 1)
    (h3 : total_people = 22)
    (h4 : total_money = 50)
    (h5 : A + C = total_people)
    (h6 : ticket_price_adult * A + ticket_price_child * C = total_money) :
    C = 18 := sorry

end play_number_of_children_l290_290095


namespace probability_that_both_girls_select_same_colored_marble_l290_290204

noncomputable def probability_girls_same_marble : ℚ :=
  let total_marbles := 2 + 2 in
  let white_marbles := 2 in
  let black_marbles := 2 in
  let first_girl_white := white_marbles / total_marbles in
  let second_girl_white := (white_marbles - 1) / (total_marbles - 1) in
  let first_girl_black := black_marbles / total_marbles in
  let second_girl_black := (black_marbles - 1) / (total_marbles - 1) in
  (first_girl_white * second_girl_white) + 
  (first_girl_black * second_girl_black)

theorem probability_that_both_girls_select_same_colored_marble :
  probability_girls_same_marble = 1 / 3 := 
  sorry

end probability_that_both_girls_select_same_colored_marble_l290_290204


namespace max_a1_value_l290_290263

theorem max_a1_value (a : ℕ → ℕ) (h1 : ∀ n : ℕ, a (n+2) = a n + a (n+1))
    (h2 : ∀ n : ℕ, a n > 0) (h3 : a 5 = 60) : a 1 ≤ 11 :=
by 
  sorry

end max_a1_value_l290_290263


namespace Brad_has_9_green_balloons_l290_290077

theorem Brad_has_9_green_balloons
  (total_balloons : ℕ)
  (red_balloons : ℕ)
  (green_balloons : ℕ)
  (h1 : total_balloons = 17)
  (h2 : red_balloons = 8)
  (h3 : total_balloons = red_balloons + green_balloons) :
  green_balloons = 9 := 
sorry

end Brad_has_9_green_balloons_l290_290077


namespace five_dice_not_all_same_number_l290_290051
open Classical

noncomputable def probability_not_all_same (n : ℕ) : ℚ :=
  1 - 1 / (6^n)

theorem five_dice_not_all_same_number :
  probability_not_all_same 5 = 1295 / 1296 :=
by
  sorry

end five_dice_not_all_same_number_l290_290051


namespace number_of_integers_with_even_divisors_l290_290400

-- Define conditions
def N := 99

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

def P : finset ℕ := { n in finset.range (N + 1) | is_perfect_square n }

noncomputable def countP : ℕ := P.card

-- The statement to prove
theorem number_of_integers_with_even_divisors : 
  (N - countP) = 90 :=
by {
  sorry
}

end number_of_integers_with_even_divisors_l290_290400


namespace average_rainfall_correct_l290_290727

-- Define the leap year condition and days in February
def leap_year_february_days : ℕ := 29

-- Define total hours in a day
def hours_in_day : ℕ := 24

-- Define total rainfall in February 2012 in inches
def total_rainfall : ℕ := 420

-- Define total hours in February 2012
def total_hours_february : ℕ := leap_year_february_days * hours_in_day

-- Define the average rainfall calculation
def average_rainfall_per_hour : ℚ :=
  total_rainfall / total_hours_february

-- Theorem to prove the average rainfall is 35/58 inches per hour
theorem average_rainfall_correct :
  average_rainfall_per_hour = 35 / 58 :=
by 
  -- Placeholder for proof
  sorry

end average_rainfall_correct_l290_290727


namespace set_intersection_complement_l290_290142

theorem set_intersection_complement (U : Set ℝ) (A B : Set ℝ) 
  (hU : U = Set.univ) 
  (hA : ∀ x : ℝ, A x ↔ x^2 - x - 6 ≤ 0) 
  (hB : ∀ x : ℝ, B x ↔ Real.log x / Real.log (1/2) ≥ -1) :
  A ∩ (U \ B) = (Set.Icc (-2 : ℝ) 0 ∪ Set.Ioc 2 3) :=
by
  ext x
  -- Proof here would follow
  sorry

end set_intersection_complement_l290_290142


namespace probability_sum_less_than_product_l290_290792

def set_of_numbers := {1, 2, 3, 4, 5, 6, 7}

def count_valid_pairs : ℕ :=
  set_of_numbers.to_list.product set_of_numbers.to_list
    |>.count (λ (ab : ℕ × ℕ), (ab.1 - 1) * (ab.2 - 1) > 1)

def total_combinations := (set_of_numbers.to_list).length ^ 2

theorem probability_sum_less_than_product :
  (count_valid_pairs : ℚ) / total_combinations = 36 / 49 :=
by
  -- Placeholder for proof, since proof is not requested
  sorry

end probability_sum_less_than_product_l290_290792


namespace arithmetic_sequence_range_of_m_l290_290697

-- Conditions
variable {a : ℕ+ → ℝ} -- Sequence of positive terms
variable {S : ℕ+ → ℝ} -- Sum of the first n terms
variable (h : ∀ n, 2 * Real.sqrt (S n) = a n + 1) -- Relationship condition

-- Part 1: Prove that {a_n} is an arithmetic sequence
theorem arithmetic_sequence (n : ℕ+)
    (h1 : ∀ n, 2 * Real.sqrt (S n) = a n + 1)
    (h2 : S 1 = 1 / 4 * (a 1 + 1)^2) :
    ∃ d : ℝ, ∀ n, a (n + 1) = a n + d :=
sorry

-- Part 2: Find range of m
theorem range_of_m (T : ℕ+ → ℝ)
    (hT : ∀ n, T n = 1 / 4 * n + 1 / 8 * (1 - 1 / (2 * n + 1))) :
    ∃ m : ℝ, (6 / 7 : ℝ) < m ∧ m ≤ 10 / 9 ∧
    (∃ n₁ n₂ n₃ : ℕ+, (n₁ < n₂ ∧ n₂ < n₃) ∧ (∀ n, T n < m ↔ n₁ ≤ n ∧ n ≤ n₃)) :=
sorry

end arithmetic_sequence_range_of_m_l290_290697


namespace player_A_winning_probability_l290_290955

theorem player_A_winning_probability :
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  P_total - P_draw - P_B_wins = 1 / 6 :=
by
  let P_draw := 1 / 2
  let P_B_wins := 1 / 3
  let P_total := 1
  sorry

end player_A_winning_probability_l290_290955


namespace domain_inequality_l290_290928

theorem domain_inequality (m : ℝ) : 
  (∀ x : ℝ, m * x^2 - (1 - m) * x + m ≥ 0) ↔ (m ≥ 1/3) :=
by
  sorry

end domain_inequality_l290_290928


namespace radius_of_fourth_circle_is_12_l290_290597

theorem radius_of_fourth_circle_is_12 (r : ℝ) (radii : Fin 7 → ℝ) 
  (h_geometric : ∀ i, radii (Fin.succ i) = r * radii i) 
  (h_smallest : radii 0 = 6)
  (h_largest : radii 6 = 24) :
  radii 3 = 12 :=
by
  sorry

end radius_of_fourth_circle_is_12_l290_290597


namespace tan_alpha_value_l290_290901

variable (α : Real)
variable (h1 : Real.sin α = 4/5)
variable (h2 : π/2 < α ∧ α < π)

theorem tan_alpha_value : Real.tan α = -4/3 := by
  sorry

end tan_alpha_value_l290_290901


namespace B_pow_101_eq_B_pow_5_l290_290160

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, -1, 0],
    ![1, 0, 0],
    ![0, 0, 0]]

theorem B_pow_101_eq_B_pow_5 : B^101 = B := 
by sorry

end B_pow_101_eq_B_pow_5_l290_290160


namespace necessary_and_sufficient_n_geq_4_l290_290445

variables {A B C D : Type*} [Convexity A] [Convexity B] [Convexity C] [Convexity D]

def is_acute_angle (angle : Type*) := sorry -- Placeholder definition; properly define acute angle

def is_obtuse_angle (angle : Type*) := sorry -- Placeholder definition; properly define obtuse angle

def quadrilateral (A B C D : Type*) := sorry -- Placeholder for checking if A B C D forms a quadrilateral.

def angle_D_acute (D : Type*) (quad : quadrilateral A B C D) := is_acute_angle D

def n_obtuse_triangles (quad : quadrilateral A B C D) (n : ℕ) :=
  ∃ (triangles : list (Type* × Type* × Type*)),
    (∀ (t : Type* × Type* × Type*), t ∈ triangles → is_obtuse_angle t.fst ∧ is_obtuse_angle t.snd ∧ is_obtuse_angle t.snd) ∧
    length triangles = n

theorem necessary_and_sufficient_n_geq_4
  (quad : quadrilateral A B C D)
  (h : angle_D_acute D quad)
  (n : ℕ) :
  n_obtuse_triangles quad n ↔ n ≥ 4 :=
sorry -- Proof omitted

end necessary_and_sufficient_n_geq_4_l290_290445


namespace problem_statement_l290_290989

theorem problem_statement (h: 2994 * 14.5 = 175) : 29.94 * 1.45 = 1.75 := 
by {
  sorry
}

end problem_statement_l290_290989


namespace board_number_never_54_after_one_hour_l290_290964

/-- Suppose we start with the number 12. Each minute, the number on the board is either
    multiplied or divided by 2 or 3. After 60 minutes, prove that the number on the board cannot be 54. -/
theorem board_number_never_54_after_one_hour (initial : ℕ) (operations : ℕ → ℕ → ℕ)
  (h_initial : initial = 12)
  (h_operations : ∀ (t : ℕ) (n : ℕ), t < 60 → (operations t n = n * 2 ∨ operations t n = n / 2 
    ∨ operations t n = n * 3 ∨ operations t n = n / 3)) :
  ¬ (∃ final, initial = 12 ∧ (∀ t, t < 60 → final = operations t final) ∧ final = 54) :=
begin
  sorry
end

end board_number_never_54_after_one_hour_l290_290964


namespace evaluate_expression_l290_290679

theorem evaluate_expression (α : ℝ) (h : Real.tan α = 3) :
  (2 * Real.sin (2 * α) - 3 * Real.cos (2 * α)) / (4 * Real.sin (2 * α) + 5 * Real.cos (2 * α)) = -9 / 4 :=
sorry

end evaluate_expression_l290_290679


namespace production_cost_per_performance_l290_290462

theorem production_cost_per_performance
  (overhead : ℕ)
  (revenue_per_performance : ℕ)
  (num_performances : ℕ)
  (production_cost : ℕ)
  (break_even : num_performances * revenue_per_performance = overhead + num_performances * production_cost) :
  production_cost = 7000 :=
by
  have : num_performances = 9 := by sorry
  have : revenue_per_performance = 16000 := by sorry
  have : overhead = 81000 := by sorry
  exact sorry

end production_cost_per_performance_l290_290462


namespace gcd_m_pow_5_plus_125_m_plus_3_l290_290375

theorem gcd_m_pow_5_plus_125_m_plus_3 (m : ℕ) (h: m > 16) : 
  Nat.gcd (m^5 + 125) (m + 3) = Nat.gcd 27 (m + 3) :=
by
  -- Proof will be provided here
  sorry

end gcd_m_pow_5_plus_125_m_plus_3_l290_290375


namespace average_student_headcount_is_correct_l290_290530

noncomputable def average_student_headcount : ℕ :=
  let a := 11000
  let b := 10200
  let c := 10800
  let d := 11300
  (a + b + c + d) / 4

theorem average_student_headcount_is_correct :
  average_student_headcount = 10825 :=
by
  -- Proof will go here
  sorry

end average_student_headcount_is_correct_l290_290530


namespace lcm_of_36_48_75_l290_290542

-- Definitions of the numbers and their factorizations
def num1 := 36
def num2 := 48
def num3 := 75

def factor_36 := (2^2, 3^2)
def factor_48 := (2^4, 3^1)
def factor_75 := (3^1, 5^2)

def highest_power_2 := 2^4
def highest_power_3 := 3^2
def highest_power_5 := 5^2

def lcm_36_48_75 := highest_power_2 * highest_power_3 * highest_power_5

-- The theorem statement
theorem lcm_of_36_48_75 : lcm_36_48_75 = 3600 := by
  sorry

end lcm_of_36_48_75_l290_290542


namespace vertex_C_path_length_equals_l290_290656

noncomputable def path_length_traversed_by_C (AB BC CA : ℝ) (PQ QR : ℝ) : ℝ :=
  let BC := 3  -- length of side BC is 3 inches
  let AB := 2  -- length of side AB is 2 inches
  let CA := 4  -- length of side CA is 4 inches
  let PQ := 8  -- length of side PQ of the rectangle is 8 inches
  let QR := 6  -- length of side QR of the rectangle is 6 inches
  4 * BC * Real.pi

theorem vertex_C_path_length_equals (AB BC CA PQ QR : ℝ) :
  AB = 2 ∧ BC = 3 ∧ CA = 4 ∧ PQ = 8 ∧ QR = 6 →
  path_length_traversed_by_C AB BC CA PQ QR = 12 * Real.pi :=
by
  intros h
  have hAB : AB = 2 := h.1
  have hBC : BC = 3 := h.2.1
  have hCA : CA = 4 := h.2.2.1
  have hPQ : PQ = 8 := h.2.2.2.1
  have hQR : QR = 6 := h.2.2.2.2
  simp [path_length_traversed_by_C, hAB, hBC, hCA, hPQ, hQR]
  sorry

end vertex_C_path_length_equals_l290_290656


namespace membership_percentage_change_l290_290864

theorem membership_percentage_change :
  let initial_membership := 100.0
  let first_fall_membership := initial_membership * 1.04
  let first_spring_membership := first_fall_membership * 0.95
  let second_fall_membership := first_spring_membership * 1.07
  let second_spring_membership := second_fall_membership * 0.97
  let third_fall_membership := second_spring_membership * 1.05
  let third_spring_membership := third_fall_membership * 0.81
  let final_membership := third_spring_membership
  let total_percentage_change := ((final_membership - initial_membership) / initial_membership) * 100.0
  total_percentage_change = -12.79 :=
by
  sorry

end membership_percentage_change_l290_290864


namespace largest_three_digit_number_with_7_in_hundreds_l290_290490

def is_three_digit_number_with_7_in_hundreds (n : ℕ) : Prop := 
  100 ≤ n ∧ n < 1000 ∧ (n / 100) = 7

theorem largest_three_digit_number_with_7_in_hundreds : 
  ∀ (n : ℕ), is_three_digit_number_with_7_in_hundreds n → n ≤ 799 :=
by sorry

end largest_three_digit_number_with_7_in_hundreds_l290_290490


namespace one_div_a_plus_one_div_b_l290_290149

theorem one_div_a_plus_one_div_b (a b : ℝ) (h₀ : a ≠ b) (ha : a^2 - 3 * a + 2 = 0) (hb : b^2 - 3 * b + 2 = 0) :
  1 / a + 1 / b = 3 / 2 :=
by
  -- Proof goes here
  sorry

end one_div_a_plus_one_div_b_l290_290149


namespace compute_expression_l290_290241

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end compute_expression_l290_290241


namespace square_side_length_properties_l290_290701

theorem square_side_length_properties (a: ℝ) (h: a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by
  sorry

end square_side_length_properties_l290_290701


namespace ellipse_and_fixed_point_l290_290123

section EllipseFixedPoint

variables (E : set (ℝ × ℝ)) (A B P M N T H : ℝ × ℝ) (HN : ℝ) 

-- Assume the ellipse E has its center at the origin 
-- and passes through points A and B as defined.
def is_ellipse_center_origin (E : set (ℝ × ℝ)) : Prop :=
  (0, -2) ∈ E ∧ (3/2, -1) ∈ E

-- Given the lines and points definitions:
def is_line_through_point (P : ℝ × ℝ) (line : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, line x = (x, m * x + snd P)

def is_parallel (line1 line2 : ℝ → ℝ × ℝ) : Prop :=
  ∃ m : ℝ, ∀ x : ℝ, fst (line1 x) = fst (line2 x + m)

def vector_equality (MT TH HN : ℝ) : Prop :=
  MT = TH ∧ HN = TH

-- Define fixed point passing property:
def fixed_point_pass (HN_line : ℝ × ℝ) (fixed_point : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, (fst HN_line = k * fst fixed_point) ∧ (snd HN_line = k * snd fixed_point)

-- The proof problem given the conditions and what needs to be proved
theorem ellipse_and_fixed_point 
  (hE : is_ellipse_center_origin E)
  (hAB : is_line_through_point A (λ x, (x, (2/3) * x - 2)))
  (hM : M ∈ E)
  (hN : N ∈ E)
  (hT : is_parallel (λ y, (M.1, y)) (λ x, (T.1, snd P))) 
  (hTH : vector_equality (dist M T) (dist T H) (dist H N))
  : fixed_point_pass H (0, -2) := 
sorry

end EllipseFixedPoint

end ellipse_and_fixed_point_l290_290123


namespace ellipse_equation_and_fixed_point_proof_l290_290128

theorem ellipse_equation_and_fixed_point_proof :
  (∀ (m n : ℝ), (m > 0) → (n > 0) → (m ≠ n) →
                (m * 0^2 + n * (-2)^2 = 1) ∧ (m * (3/2)^2 + n * (-1)^2 = 1) → 
                (m = 1/3 ∧ n = 1/4)) ∧
                (∀ (M N : ℝ × ℝ), ∃ H : ℝ × ℝ,
                (∃ (P : ℝ × ℝ), P = (1, -2)) ∧
                (∃ (A : ℝ × ℝ), A = (0, -2)) ∧
                (∃ (B : ℝ × ℝ), B = (3/2, -1)) ∧
                (∃ (T : ℝ × ℝ), ∀ x, M.1 = x) ∧
                (∃ K : ℝ × ℝ, K = (0, -2)) →
                M.1 * N.2 - M.2 * N.1 = 0) :=
by
  sorry

end ellipse_equation_and_fixed_point_proof_l290_290128


namespace triple_root_possible_values_l290_290221

-- Definitions and conditions
def polynomial (x : ℤ) (b3 b2 b1 : ℤ) := x^4 + b3 * x^3 + b2 * x^2 + b1 * x + 24

-- The proof problem
theorem triple_root_possible_values 
  (r b3 b2 b1 : ℤ)
  (h_triple_root : polynomial r b3 b2 b1 = (x * (x - 1) * (x - 2)) * (x - r) ) :
  r = -2 ∨ r = -1 ∨ r = 1 ∨ r = 2 :=
by
  sorry

end triple_root_possible_values_l290_290221


namespace vasya_numbers_l290_290823

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290823


namespace third_stick_length_l290_290010

theorem third_stick_length (x : ℝ) (h1 : 2 > 0) (h2 : 5 > 0) (h3 : 3 < x) (h4 : x < 7) : x = 4 :=
by
  sorry

end third_stick_length_l290_290010


namespace remaining_painting_time_l290_290218

-- Define the conditions
def total_rooms : ℕ := 10
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 8

-- Define what we want to prove
theorem remaining_painting_time : (total_rooms - rooms_painted) * hours_per_room = 16 :=
by
  -- Here is where you would provide the proof
  sorry

end remaining_painting_time_l290_290218


namespace domain_function_1_domain_function_2_domain_function_3_l290_290893

-- Define the conditions and the required domain equivalence in Lean 4
-- Problem (1)
theorem domain_function_1 (x : ℝ): x + 2 ≠ 0 ∧ x + 5 ≥ 0 ↔ x ≥ -5 ∧ x ≠ -2 := 
sorry

-- Problem (2)
theorem domain_function_2 (x : ℝ): x^2 - 4 ≥ 0 ∧ 4 - x^2 ≥ 0 ∧ x^2 - 9 ≠ 0 ↔ (x = 2 ∨ x = -2) :=
sorry

-- Problem (3)
theorem domain_function_3 (x : ℝ): x - 5 ≥ 0 ∧ |x| ≠ 7 ↔ x ≥ 5 ∧ x ≠ 7 :=
sorry

end domain_function_1_domain_function_2_domain_function_3_l290_290893


namespace quadratic_other_root_l290_290551

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end quadratic_other_root_l290_290551


namespace sellable_fruit_l290_290483

theorem sellable_fruit :
  let total_oranges := 30 * 300
  let total_damaged_oranges := total_oranges * 10 / 100
  let sellable_oranges := total_oranges - total_damaged_oranges

  let total_nectarines := 45 * 80
  let nectarines_taken := 5 * 20
  let sellable_nectarines := total_nectarines - nectarines_taken

  let total_apples := 20 * 120
  let bad_apples := 50
  let sellable_apples := total_apples - bad_apples

  sellable_oranges + sellable_nectarines + sellable_apples = 13950 :=
by
  sorry

end sellable_fruit_l290_290483


namespace part1_minimum_value_of_f_part2_range_of_a_l290_290026

open Real

-- Define the functions f(x) and g(x)
def f (x : ℝ) := (x + 1) * log (x + 1)
def g (a x : ℝ) := a * x^2 + x

-- Problem (1): Prove the minimum value of f(x)
theorem part1_minimum_value_of_f :
  ∃ x : ℝ, f x = -(1 / exp 1) :=
sorry

-- Problem (2): Find the range of real number a
theorem part2_range_of_a (a : ℝ) :
  (∀ x : ℝ, x ≥ 0 → f x ≤ g a x) → a ≥ 1/2 :=
sorry

end part1_minimum_value_of_f_part2_range_of_a_l290_290026


namespace square_root_properties_l290_290699

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end square_root_properties_l290_290699


namespace total_books_l290_290102

theorem total_books (books_per_shelf : ℕ) (num_shelves : ℕ) (total : ℕ) :
  books_per_shelf = 56 → 
  num_shelves = 9 → 
  total = 504 →
  books_per_shelf * num_shelves = total :=
by
  intros h1 h2 h3
  rw [h1, h2]
  exact h3

end total_books_l290_290102


namespace find_a_l290_290747

noncomputable def f (x a : ℝ) : ℝ := x * (Real.exp x + a * Real.exp (-x))

theorem find_a (a : ℝ) : (∀ x : ℝ, f x a = -f (-x) a) → a = 1 :=
by
  sorry

end find_a_l290_290747


namespace expectation_conditioned_eq_variance_conditioned_eq_unconditional_expectation_eq_unconditional_variance_eq_l290_290446

noncomputable def expectation_conditioned (ξ : ℕ → ℝ) (τ : ℕ) :=
  τ * (ξ 1)

noncomputable def variance_conditioned (ξ : ℕ → ℝ) (τ : ℕ) :=
  τ * (variance ξ 1)

theorem expectation_conditioned_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  (E (λ ω, ∑ i in range(τ), ξ i ω) | τ) = expectation_conditioned ξ τ := by
  sorry

theorem variance_conditioned_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  (D (λ ω, ∑ i in range(τ), ξ i ω) | τ) = variance_conditioned ξ τ := by
  sorry

theorem unconditional_expectation_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  E (λ ω, ∑ i in range(τ), ξ i ω) = τ * (E (ξ 1)) := by
  sorry

theorem unconditional_variance_eq (ξ : ℕ → ℝ) (τ : ℕ) (h_iid : ∀ i j, i ≠ j → statistically_independent (ξ i) (ξ j))
  (h_ident_dist : ∀ i, identically_distributed (ξ i) (ξ 1)) :
  D (λ ω, ∑ i in range(τ), ξ i ω) = τ * (D (ξ 1)) + (D (τ) * (E (ξ 1))^2) := by
  sorry

end expectation_conditioned_eq_variance_conditioned_eq_unconditional_expectation_eq_unconditional_variance_eq_l290_290446


namespace soldiers_line_l290_290516

theorem soldiers_line (n x y z : ℕ) (h₁ : y = 6 * x) (h₂ : y = 7 * z)
                      (h₃ : n = x + y) (h₄ : n = 7 * x) (h₅ : n = 8 * z) : n = 98 :=
by 
  sorry

end soldiers_line_l290_290516


namespace sum_lent_is_10000_l290_290083

theorem sum_lent_is_10000
  (P : ℝ)
  (r : ℝ := 0.075)
  (t : ℝ := 7)
  (I : ℝ := P - 4750) 
  (H1 : I = P * r * t) :
  P = 10000 :=
sorry

end sum_lent_is_10000_l290_290083


namespace vasya_numbers_l290_290796

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290796


namespace find_value_of_expression_l290_290122

-- Given conditions
variable (a : ℝ)
variable (h_root : a^2 + 2 * a - 2 = 0)

-- Mathematically equivalent proof problem
theorem find_value_of_expression : 3 * a^2 + 6 * a + 2023 = 2029 :=
by
  sorry

end find_value_of_expression_l290_290122


namespace intersection_point_of_lines_l290_290881

theorem intersection_point_of_lines :
  ∃ (x y : ℚ), 
    (3 * y = -2 * x + 6) ∧ 
    (-2 * y = 7 * x + 4) ∧ 
    x = -24 / 17 ∧ 
    y = 50 / 17 :=
by
  sorry

end intersection_point_of_lines_l290_290881


namespace solution_l290_290518

-- Define the conditions based on the given problem
variables {A B C D : Type}
variables {AB BC CD DA : ℝ} (h1 : AB = 65) (h2 : BC = 105) (h3 : CD = 125) (h4 : DA = 95)
variables (cy_in_circle : CyclicQuadrilateral A B C D)
variables (circ_inscribed : TangentialQuadrilateral A B C D)

-- Function that computes the absolute difference between segments x and y on side of length CD
noncomputable def find_absolute_difference (x y : ℝ) (h5 : x + y = 125) : ℝ := |x - y|

-- The proof statement
theorem solution :
  ∃ (x y : ℝ), x + y = 125 ∧
  (find_absolute_difference x y (by sorry) = 14) := sorry

end solution_l290_290518


namespace prob_both_questions_correct_l290_290923

variable (P : String → ℝ)

-- Definitions for the given problem conditions
def P_A : ℝ := P "first question"
def P_B : ℝ := P "second question"
def P_A'B' : ℝ := P "neither question"

-- Given values for the problem
axiom P_A_given : P_A = 0.63
axiom P_B_given : P_B = 0.49
axiom P_A'B'_given : P_A'B' = 0.20

-- The theorem that needs to be proved
theorem prob_both_questions_correct : (P "first and second question") = 0.32 :=
by
  have P_A_union_B := 1 - P_A'B'
  have P_A_and_B := P_A + P_B - P_A_union_B
  exact sorry

#eval prob_both_questions_correct P

end prob_both_questions_correct_l290_290923


namespace min_value_expr_l290_290117

theorem min_value_expr (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (b^2 + 2) / (a + b) + a^2 / (a * b + 1) ≥ 2 :=
sorry

end min_value_expr_l290_290117


namespace beta_interval_solution_l290_290693

/-- 
Prove that the values of β in the set {β | β = π/6 + 2*k*π, k ∈ ℤ} 
that satisfy the interval (-2*π, 2*π) are β = π/6 or β = -11*π/6.
-/
theorem beta_interval_solution :
  ∀ β : ℝ, (∃ k : ℤ, β = (π / 6) + 2 * k * π) → (-2 * π < β ∧ β < 2 * π) →
  (β = π / 6 ∨ β = -11 * π / 6) :=
by
  intros β h_exists h_interval
  sorry

end beta_interval_solution_l290_290693


namespace factor_z4_minus_81_l290_290889

theorem factor_z4_minus_81 :
  (z^4 - 81) = (z - 3) * (z + 3) * (z^2 + 9) :=
by
  sorry

end factor_z4_minus_81_l290_290889


namespace choose_team_captains_l290_290862

open Nat

def binom (n k : ℕ) : ℕ :=
  if k > n then 0 else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_team_captains :
  let total_members := 15
  let shortlisted := 5
  let regular := total_members - shortlisted
  binom total_members 4 - binom regular 4 = 1155 :=
by
  sorry

end choose_team_captains_l290_290862


namespace rectangular_coordinates_of_polar_2_pi_over_3_l290_290489

noncomputable def polar_to_rectangular (ρ θ : ℝ) : ℝ × ℝ :=
  (ρ * Real.cos θ, ρ * Real.sin θ)

theorem rectangular_coordinates_of_polar_2_pi_over_3 :
  polar_to_rectangular 2 (Real.pi / 3) = (1, Real.sqrt 3) :=
by
  sorry

end rectangular_coordinates_of_polar_2_pi_over_3_l290_290489


namespace total_dress_designs_l290_290511

def num_colors := 5
def num_patterns := 6
def num_sizes := 3

theorem total_dress_designs : num_colors * num_patterns * num_sizes = 90 :=
by
  sorry

end total_dress_designs_l290_290511


namespace isosceles_triangle_perimeter_correct_l290_290041

-- Definitions based on conditions
def equilateral_triangle_side_length (perimeter : ℕ) : ℕ :=
  perimeter / 3

def isosceles_triangle_perimeter (side1 side2 base : ℕ) : ℕ :=
  side1 + side2 + base

-- Given conditions
def equilateral_triangle_perimeter : ℕ := 45
def equilateral_triangle_side : ℕ := equilateral_triangle_side_length equilateral_triangle_perimeter

-- The side of the equilateral triangle is also a leg of the isosceles triangle
def isosceles_triangle_leg : ℕ := equilateral_triangle_side
def isosceles_triangle_base : ℕ := 10

-- The problem to prove
theorem isosceles_triangle_perimeter_correct : 
  isosceles_triangle_perimeter isosceles_triangle_leg isosceles_triangle_leg isosceles_triangle_base = 40 :=
by
  sorry

end isosceles_triangle_perimeter_correct_l290_290041


namespace duty_person_C_l290_290936

/-- Given amounts of money held by three persons and a total custom duty,
    prove that the duty person C should pay is 17 when payments are proportional. -/
theorem duty_person_C (money_A money_B money_C total_duty : ℕ) (total_money : ℕ)
  (hA : money_A = 560) (hB : money_B = 350) (hC : money_C = 180) (hD : total_duty = 100)
  (hT : total_money = money_A + money_B + money_C) :
  total_duty * money_C / total_money = 17 :=
by
  -- proof goes here
  sorry

end duty_person_C_l290_290936


namespace intersection_property_l290_290450

theorem intersection_property (x_0 : ℝ) (h1 : x_0 > 0) (h2 : -x_0 = Real.tan x_0) :
  (x_0^2 + 1) * (Real.cos (2 * x_0) + 1) = 2 :=
sorry

end intersection_property_l290_290450


namespace problem_statement_l290_290898

theorem problem_statement : 
  (∀ (base : ℤ) (exp : ℕ), (-3) = base ∧ 2 = exp → (base ^ exp ≠ -9)) :=
by
  sorry

end problem_statement_l290_290898


namespace find_center_of_circle_l290_290686

noncomputable def center_of_circle (θ ρ : ℝ) : Prop :=
  ρ = (1 : ℝ) ∧ θ = (-Real.pi / (3 : ℝ))

theorem find_center_of_circle (θ ρ : ℝ) (h : ρ = Real.cos θ - Real.sqrt 3 * Real.sin θ) :
  center_of_circle θ ρ := by
  sorry

end find_center_of_circle_l290_290686


namespace cost_B_solution_l290_290488

variable (cost_B : ℝ)

/-- The number of items of type A that can be purchased with 1000 yuan 
is equal to the number of items of type B that can be purchased with 800 yuan. -/
def items_purchased_equality (cost_B : ℝ) : Prop :=
  1000 / (cost_B + 10) = 800 / cost_B

/-- The cost of each item of type A is 10 yuan more than the cost of each item of type B. -/
def cost_difference (cost_B : ℝ) : Prop :=
  cost_B + 10 - cost_B = 10

/-- The cost of each item of type B is 40 yuan. -/
theorem cost_B_solution (h1: items_purchased_equality cost_B) (h2: cost_difference cost_B) :
  cost_B = 40 := by
sorry

end cost_B_solution_l290_290488


namespace acres_used_for_corn_l290_290986

-- Define the conditions given in the problem
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio_parts : ℕ := ratio_beans + ratio_wheat + ratio_corn
def part_size : ℕ := total_land / total_ratio_parts

-- State the theorem to prove that the land used for corn is 376 acres
theorem acres_used_for_corn : (part_size * ratio_corn = 376) :=
  sorry

end acres_used_for_corn_l290_290986


namespace problem1_problem2_l290_290341

-- Problem 1
theorem problem1 (a b : ℝ) (ha : a > 0) (hb : b > 0) (hneq : a ≠ b) : 
  (a / Real.sqrt b) + (b / Real.sqrt a) > Real.sqrt a + Real.sqrt b :=
sorry

-- Problem 2
theorem problem2 (x : ℝ) (hx : x > -1) (m : ℕ) (hm : 0 < m) : 
  (1 + x)^m ≥ 1 + m * x :=
sorry

end problem1_problem2_l290_290341


namespace average_speed_is_80_l290_290467

def distance : ℕ := 100

def time : ℚ := 5 / 4  -- 1.25 hours expressed as a rational number

noncomputable def average_speed : ℚ := distance / time

theorem average_speed_is_80 : average_speed = 80 := by
  sorry

end average_speed_is_80_l290_290467


namespace probability_correct_l290_290202

open Finset

def standard_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

noncomputable def probability_abc_144 : ℚ :=
  let outcomes := (standard_die × standard_die × standard_die).filter (λ (t : ℕ × ℕ × ℕ), t.1 * t.2 * t.3 = 144)
  1 / 6 * 1 / 6 * 1 / 6 * outcomes.card

theorem probability_correct : probability_abc_144 = 1 / 72 := by
  unfold probability_abc_144
  sorry

end probability_correct_l290_290202


namespace identity_holds_for_all_real_numbers_l290_290690

theorem identity_holds_for_all_real_numbers (a b : ℝ) : 
  a^2 + b^2 + 2 * a * b = (a + b)^2 := 
by sorry

end identity_holds_for_all_real_numbers_l290_290690


namespace sale_percent_saved_l290_290206

noncomputable def percent_saved (P : ℝ) : ℝ := (3 * P) / (6 * P) * 100

theorem sale_percent_saved :
  ∀ (P : ℝ), P > 0 → percent_saved P = 50 :=
by
  intros P hP
  unfold percent_saved
  have hP_nonzero : 6 * P ≠ 0 := by linarith
  field_simp [hP_nonzero]
  norm_num
  sorry

end sale_percent_saved_l290_290206


namespace volume_of_cone_l290_290345

theorem volume_of_cone (d : ℝ) (h : ℝ) (r : ℝ) : 
  d = 10 ∧ h = 0.6 * d ∧ r = d / 2 → (1 / 3) * π * r^2 * h = 50 * π :=
by
  intro h1
  rcases h1 with ⟨h_d, h_h, h_r⟩
  sorry

end volume_of_cone_l290_290345


namespace range_of_m_l290_290719

theorem range_of_m (h : ¬ (∀ x : ℝ, ∃ m : ℝ, 4 ^ x - 2 ^ (x + 1) + m = 0) → false) : 
  ∀ m : ℝ, m ≤ 1 :=
by
  sorry

end range_of_m_l290_290719


namespace number_of_teams_l290_290210

-- Total number of players
def total_players : Nat := 12

-- Number of ways to choose one captain
def ways_to_choose_captain : Nat := total_players

-- Number of remaining players after choosing the captain
def remaining_players : Nat := total_players - 1

-- Number of players needed to form a team (excluding the captain)
def team_size : Nat := 5

-- Number of ways to choose 5 players from the remaining 11
def ways_to_choose_team (n k : Nat) : Nat := Nat.choose n k

-- Total number of different teams
def total_teams : Nat := ways_to_choose_captain * ways_to_choose_team remaining_players team_size

theorem number_of_teams : total_teams = 5544 := by
  sorry

end number_of_teams_l290_290210


namespace sam_morning_run_distance_l290_290612

variable (n : ℕ) (x : ℝ)

theorem sam_morning_run_distance (h : x + 2 * n * x + 12 = 18) : x = 6 / (1 + 2 * n) :=
by
  sorry

end sam_morning_run_distance_l290_290612


namespace find_y_l290_290922

theorem find_y (x y : ℝ) (h1 : 0.5 * x = 0.25 * y - 30) (h2 : x = 690) : y = 1500 :=
by
  sorry

end find_y_l290_290922


namespace mean_of_set_median_is_128_l290_290771

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l290_290771


namespace count_valid_rods_l290_290941

def isValidRodLength (d : ℕ) : Prop :=
  5 ≤ d ∧ d < 27

def countValidRodLengths (lower upper : ℕ) : ℕ :=
  upper - lower + 1

theorem count_valid_rods :
  let valid_rods_count := countValidRodLengths 5 26
  valid_rods_count = 22 :=
by
  sorry

end count_valid_rods_l290_290941


namespace sufficient_but_not_necessary_condition_l290_290163

theorem sufficient_but_not_necessary_condition (a : ℝ) : 
  (a > 0) → (|2 * a + 1| > 1) ∧ ¬((|2 * a + 1| > 1) → (a > 0)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l290_290163


namespace sum_of_edges_corners_faces_of_rectangular_prism_l290_290306

-- Definitions based on conditions
def rectangular_prism_edges := 12
def rectangular_prism_corners := 8
def rectangular_prism_faces := 6
def resulting_sum := rectangular_prism_edges + rectangular_prism_corners + rectangular_prism_faces

-- Statement we want to prove
theorem sum_of_edges_corners_faces_of_rectangular_prism :
  resulting_sum = 26 := 
by 
  sorry -- Placeholder for the proof

end sum_of_edges_corners_faces_of_rectangular_prism_l290_290306


namespace intersecting_lines_find_m_l290_290629

theorem intersecting_lines_find_m : ∃ m : ℚ, 
  (∃ x y : ℚ, y = 4*x + 2 ∧ y = -3*x - 18 ∧ y = 2*x + m) ↔ m = -26/7 :=
by
  sorry

end intersecting_lines_find_m_l290_290629


namespace age_problem_solution_l290_290973

theorem age_problem_solution 
  (x : ℕ) 
  (xiaoxiang_age : ℕ := 5) 
  (father_age : ℕ := 48) 
  (mother_age : ℕ := 42) 
  (h : (father_age + x) + (mother_age + x) = 6 * (xiaoxiang_age + x)) : 
  x = 15 :=
by {
  -- To be proved
  sorry
}

end age_problem_solution_l290_290973


namespace find_s_l290_290539

theorem find_s (s : ℝ) : (s, 7) ∈ line_through (0, 4) (-6, 1) → s = 6 :=
sorry

end find_s_l290_290539


namespace elena_bouquet_petals_l290_290888

def num_petals (count : ℕ) (petals_per_flower : ℕ) : ℕ :=
  count * petals_per_flower

theorem elena_bouquet_petals :
  let num_lilies := 4
  let lilies_petal_count := num_petals num_lilies 6
  
  let num_tulips := 2
  let tulips_petal_count := num_petals num_tulips 3

  let num_roses := 2
  let roses_petal_count := num_petals num_roses 5
  
  let num_daisies := 1
  let daisies_petal_count := num_petals num_daisies 12
  
  lilies_petal_count + tulips_petal_count + roses_petal_count + daisies_petal_count = 52 := by
  sorry

end elena_bouquet_petals_l290_290888


namespace book_contains_300_pages_l290_290082

-- The given conditions
def total_digits : ℕ := 792
def digits_per_page_1_to_9 : ℕ := 9 * 1
def digits_per_page_10_to_99 : ℕ := 90 * 2
def remaining_digits : ℕ := total_digits - digits_per_page_1_to_9 - digits_per_page_10_to_99
def pages_with_3_digits : ℕ := remaining_digits / 3

-- The total number of pages
def total_pages : ℕ := 99 + pages_with_3_digits

theorem book_contains_300_pages : total_pages = 300 := by
  sorry

end book_contains_300_pages_l290_290082


namespace total_notes_count_l290_290304

theorem total_notes_count :
  ∀ (rows : ℕ) (notes_per_row : ℕ) (blue_notes_per_red : ℕ) (additional_blue_notes : ℕ),
  rows = 5 →
  notes_per_row = 6 →
  blue_notes_per_red = 2 →
  additional_blue_notes = 10 →
  (rows * notes_per_row + (rows * notes_per_row * blue_notes_per_red + additional_blue_notes)) = 100 := by
  intros rows notes_per_row blue_notes_per_red additional_blue_notes
  sorry

end total_notes_count_l290_290304


namespace doubled_sum_of_squares_l290_290318

theorem doubled_sum_of_squares (a b : ℝ) : 
  2 * (a^2 + b^2) - (a - b)^2 = (a + b)^2 := 
by
  sorry

end doubled_sum_of_squares_l290_290318


namespace greatest_integer_less_than_neg_21_over_5_l290_290839

theorem greatest_integer_less_than_neg_21_over_5 :
  ∃ (z : ℤ), z < -21 / 5 ∧ ∀ (w : ℤ), w < -21 / 5 → w ≤ z :=
begin
  use -5,
  split,
  { norm_num },
  { intros w hw,
    linarith }
end

end greatest_integer_less_than_neg_21_over_5_l290_290839


namespace female_democrats_l290_290074

theorem female_democrats (F M : ℕ) (h1 : F + M = 840) (h2 : F / 2 + M / 4 = 280) : F / 2 = 140 :=
by 
  sorry

end female_democrats_l290_290074


namespace count_even_divisors_lt_100_l290_290405

-- Define the set of natural numbers less than 100
def nat_lt_100 := {n : ℕ | n < 100}

-- Define the set of perfect squares less than 100
def perfect_squares_lt_100 := {n : ℕ | ∃ k : ℕ, k^2 = n ∧ n < 100}

-- Define the set of natural numbers less than 100 with an even number of positive divisors
def even_divisors_lt_100 := nat_lt_100 \ perfect_squares_lt_100

-- Theorem stating the number of elements with even number of divisors is 90
theorem count_even_divisors_lt_100 : (even_divisors_lt_100).card = 90 := 
sorry

end count_even_divisors_lt_100_l290_290405


namespace initial_percentage_increase_l290_290519

theorem initial_percentage_increase (W R : ℝ) (P : ℝ) 
  (h1 : R = W * (1 + P / 100)) 
  (h2 : R * 0.75 = W * 1.3500000000000001) : P = 80 := 
by
  sorry

end initial_percentage_increase_l290_290519


namespace alcohol_percentage_l290_290761

theorem alcohol_percentage (P : ℝ) : 
  (0.10 * 300) + (P / 100 * 450) = 0.22 * 750 → P = 30 :=
by
  intros h
  sorry

end alcohol_percentage_l290_290761


namespace quadratic_not_proposition_l290_290372

def is_proposition (P : Prop) : Prop := ∃ (b : Bool), (b = true ∨ b = false)

theorem quadratic_not_proposition : ¬ is_proposition (∃ x : ℝ, x^2 + 2*x - 3 < 0) :=
by 
  sorry

end quadratic_not_proposition_l290_290372


namespace caloprian_lifespan_proof_l290_290872

open Real

noncomputable def timeDilation (delta_t : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  delta_t * sqrt (1 - (v ^ 2) / (c ^ 2))

noncomputable def caloprianMinLifeSpan (d : ℝ) (v : ℝ) (c : ℝ) : ℝ :=
  let earth_time := (d / v) * 2
  timeDilation earth_time v c

theorem caloprian_lifespan_proof :
  caloprianMinLifeSpan 30 0.3 1 = 20 * sqrt 91 :=
sorry

end caloprian_lifespan_proof_l290_290872


namespace distinct_lines_in_4x4_grid_l290_290920

open Nat

theorem distinct_lines_in_4x4_grid : 
  let n := 16
  let total_pairs := choose n 2
  let overcount_correction := 8
  let additional_lines := 4
  total_pairs - 2 * overcount_correction + additional_lines = 108 := 
by
  sorry

end distinct_lines_in_4x4_grid_l290_290920


namespace find_carl_age_l290_290038

variables (Alice Bob Carl : ℝ)

-- Conditions
def average_age : Prop := (Alice + Bob + Carl) / 3 = 15
def carl_twice_alice : Prop := Carl - 5 = 2 * Alice
def bob_fraction_alice : Prop := Bob + 4 = (3 / 4) * (Alice + 4)

-- Conjecture
theorem find_carl_age : average_age Alice Bob Carl ∧ carl_twice_alice Alice Carl ∧ bob_fraction_alice Alice Bob → Carl = 34.818 :=
by
  sorry

end find_carl_age_l290_290038


namespace daily_savings_in_dollars_l290_290029

-- Define the total savings and the number of days
def total_savings_in_dimes : ℕ := 3
def number_of_days : ℕ := 30

-- Define the conversion factor from dimes to dollars
def dime_to_dollar : ℝ := 0.10

-- Prove that the daily savings in dollars is $0.01
theorem daily_savings_in_dollars : total_savings_in_dimes / number_of_days * dime_to_dollar = 0.01 :=
by sorry

end daily_savings_in_dollars_l290_290029


namespace complex_square_l290_290409

theorem complex_square (z : ℂ) (i : ℂ) (h₁ : z = 5 - 3 * i) (h₂ : i * i = -1) : z^2 = 16 - 30 * i :=
by
  rw [h₁]
  sorry

end complex_square_l290_290409


namespace solve_arcsin_sin_l290_290035

theorem solve_arcsin_sin (x : ℝ) (h : -Real.pi / 2 ≤ x ∧ x ≤ Real.pi / 2) :
  Real.arcsin (Real.sin (2 * x)) = x ↔ x = 0 ∨ x = Real.pi / 3 ∨ x = -Real.pi / 3 :=
by
  sorry

end solve_arcsin_sin_l290_290035


namespace delta_max_success_ratio_l290_290433

theorem delta_max_success_ratio :
  ∃ a b c d : ℕ, 
    0 < a ∧ a < b ∧ (40 * a) < (21 * b) ∧
    0 < c ∧ c < d ∧ (4 * c) < (3 * d) ∧
    b + d = 600 ∧
    (a + c) / 600 = 349 / 600 :=
by
  sorry

end delta_max_success_ratio_l290_290433


namespace find_k_value_l290_290386

theorem find_k_value (k : ℝ) (h : (7 * (-1)^3 - 3 * (-1)^2 + k * -1 + 5 = 0)) :
  k^3 + 2 * k^2 - 11 * k - 85 = -105 :=
by {
  sorry
}

end find_k_value_l290_290386


namespace matvey_healthy_diet_l290_290944

theorem matvey_healthy_diet (n b_1 p_1 : ℕ) (h1 : n * b_1 - (n * (n - 1)) / 2 = 264) (h2 : n * p_1 + (n * (n - 1)) / 2 = 187) :
  n = 11 :=
by
  let buns_diff_pears := b_1 - p_1 - (n - 1)
  have buns_def : 264 = n * buns_diff_pears + n * (n - 1) / 2 := sorry
  have pears_def : 187 = n * buns_diff_pears - n * (n - 1) / 2 := sorry
  have diff : 77 = n * buns_diff_pears := sorry
  sorry

end matvey_healthy_diet_l290_290944


namespace probability_not_all_same_l290_290059

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l290_290059


namespace plane_speeds_l290_290190

theorem plane_speeds (v : ℕ) 
    (h1 : ∀ (t : ℕ), t = 5 → 20 * v = 4800): 
  v = 240 ∧ 3 * v = 720 := by
  sorry

end plane_speeds_l290_290190


namespace problem_1_problem_2_l290_290742

noncomputable def f (x : ℝ) := (x - Real.exp 1) / Real.exp x

theorem problem_1 :
  (∀ x, f' x > 0 → x ∈ set.Iic (Real.exp 1 + 1)) ∧
  (∀ x, f' x < 0 → x ∈ set.Ioi (Real.exp 1 + 1)) ∧
  (f (Real.exp 1 + 1) = Real.exp (-Real.exp 1 - 1)) :=
sorry

theorem problem_2 (c : ℝ) :
  (∀ x, x ∈ set.Ioi 0 →  2 * |Real.log x - Real.log 2| ≥ f x + c - Real.exp (-2)) → 
  c ≤ (Real.exp 1 - 1) / Real.exp 2 :=
sorry

end problem_1_problem_2_l290_290742


namespace find_parabola_eq_l290_290541

noncomputable def parabola_equation (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, y = -3 * x ^ 2 + 18 * x - 22 ↔ (x - 3) ^ 2 + 5 = y

theorem find_parabola_eq :
  ∃ a b c : ℝ, (vertex = (3, 5) ∧ axis_of_symmetry ∧ point_on_parabola = (2, 2)) →
  parabola_equation a b c :=
sorry

end find_parabola_eq_l290_290541


namespace value_of_D_l290_290595

theorem value_of_D (E F D : ℕ) (cond1 : E + F + D = 15) (cond2 : F + E = 11) : D = 4 := 
by
  sorry

end value_of_D_l290_290595


namespace parallel_lines_m_l290_290628

theorem parallel_lines_m (m : ℝ) :
  (∀ x y : ℝ, x + 2 * m * y - 1 = 0 → (3 * m - 1) * x - m * y - 1 = 0)
  → m = 0 ∨ m = 1 / 6 := 
sorry

end parallel_lines_m_l290_290628


namespace smallest_n_divisible_l290_290886

theorem smallest_n_divisible (n : ℕ) : (15 * n - 3) % 11 = 0 ↔ n = 9 := by
  sorry

end smallest_n_divisible_l290_290886


namespace total_rooms_in_hotel_l290_290605

def first_wing_floors : ℕ := 9
def first_wing_halls_per_floor : ℕ := 6
def first_wing_rooms_per_hall : ℕ := 32

def second_wing_floors : ℕ := 7
def second_wing_halls_per_floor : ℕ := 9
def second_wing_rooms_per_hall : ℕ := 40

def third_wing_floors : ℕ := 12
def third_wing_halls_per_floor : ℕ := 4
def third_wing_rooms_per_hall : ℕ := 50

def first_wing_total_rooms : ℕ := 
  first_wing_floors * first_wing_halls_per_floor * first_wing_rooms_per_hall

def second_wing_total_rooms : ℕ := 
  second_wing_floors * second_wing_halls_per_floor * second_wing_rooms_per_hall

def third_wing_total_rooms : ℕ := 
  third_wing_floors * third_wing_halls_per_floor * third_wing_rooms_per_hall

theorem total_rooms_in_hotel : 
  first_wing_total_rooms + second_wing_total_rooms + third_wing_total_rooms = 6648 := 
by 
  sorry

end total_rooms_in_hotel_l290_290605


namespace greatest_c_value_l290_290194

theorem greatest_c_value (c : ℤ) : 
  (∀ (x : ℝ), x^2 + (c : ℝ) * x + 20 ≠ -7) → c = 10 :=
by
  sorry

end greatest_c_value_l290_290194


namespace exists_invertible_int_matrix_l290_290022

theorem exists_invertible_int_matrix (m : ℕ) (k : Fin m → ℤ) : 
  ∃ A : Matrix (Fin m) (Fin m) ℤ,
    (∀ j, IsUnit (A + k j • (1 : Matrix (Fin m) (Fin m) ℤ))) :=
sorry

end exists_invertible_int_matrix_l290_290022


namespace all_positive_integers_are_nice_l290_290846

def isNice (n : ℕ) : Prop :=
  ∃ (k : ℕ) (a : Fin k → ℕ), (∀ i, ∃ m : ℕ, a i = 2 ^ m) ∧ n = (Finset.univ.sum a) / k

theorem all_positive_integers_are_nice : ∀ n : ℕ, 0 < n → isNice n := sorry

end all_positive_integers_are_nice_l290_290846


namespace triangle_base_second_l290_290466

theorem triangle_base_second (base1 height1 height2 : ℝ) 
  (h_base1 : base1 = 15) (h_height1 : height1 = 12) (h_height2 : height2 = 18) :
  let area1 := (base1 * height1) / 2
  let area2 := 2 * area1
  let base2 := (2 * area2) / height2
  base2 = 20 :=
by
  sorry

end triangle_base_second_l290_290466


namespace t_plus_reciprocal_l290_290904

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l290_290904


namespace odd_ints_divisibility_l290_290464

theorem odd_ints_divisibility (a b : ℤ) (ha_odd : a % 2 = 1) (hb_odd : b % 2 = 1) (hdiv : 2 * a * b + 1 ∣ a^2 + b^2 + 1) : a = b :=
sorry

end odd_ints_divisibility_l290_290464


namespace return_speed_is_33_33_l290_290084

noncomputable def return_speed (d: ℝ) (speed_to_b: ℝ) (avg_speed: ℝ): ℝ :=
  d / (3 + (d / avg_speed))

-- Conditions
def distance := 150
def speed_to_b := 50
def avg_speed := 40

-- Prove that the return speed is 33.33 miles per hour
theorem return_speed_is_33_33:
  return_speed distance speed_to_b avg_speed = 33.33 :=
by
  unfold return_speed
  sorry

end return_speed_is_33_33_l290_290084


namespace total_score_is_correct_l290_290609

def dad_points : ℕ := 7
def olaf_points : ℕ := 3 * dad_points
def total_points : ℕ := dad_points + olaf_points

theorem total_score_is_correct : total_points = 28 := by
  sorry

end total_score_is_correct_l290_290609


namespace find_x_from_equation_l290_290144

/-- If (1 / 8) * 2^36 = 4^x, then x = 16.5 -/
theorem find_x_from_equation (x : ℝ) (h : (1/8) * (2:ℝ)^36 = (4:ℝ)^x) : x = 16.5 :=
by sorry

end find_x_from_equation_l290_290144


namespace condition1_a_geq_1_l290_290694

theorem condition1_a_geq_1 (a : ℝ) :
  (∀ x ∈ ({1, 2, 3} : Set ℝ), a * x - 1 ≥ 0) → a ≥ 1 :=
by
sorry

end condition1_a_geq_1_l290_290694


namespace sum_of_interior_angles_l290_290763

theorem sum_of_interior_angles (n : ℕ) (h₁ : 180 * (n - 2) = 2340) : 
  180 * ((n - 3) - 2) = 1800 := by
  -- Here, we'll solve the theorem using Lean's capabilities.
  sorry

end sum_of_interior_angles_l290_290763


namespace probability_x_lt_2y_in_rectangle_l290_290655

theorem probability_x_lt_2y_in_rectangle :
  let ℝ := Real,
      rectangle := set.Icc (0, 0) (4, 3),
      region := {p : ℝ × ℝ | p.1 < 2 * p.2},
      area_of_triangle := (1 / 2) * 3 * 3,
      area_of_rectangle := 4 * 3,
      prob := area_of_triangle / area_of_rectangle in
  prob = 3 / 8 :=
by sorry

end probability_x_lt_2y_in_rectangle_l290_290655


namespace find_f_values_find_f_expression_l290_290164

variable (f : ℕ+ → ℤ)

-- Conditions in Lean
def is_increasing (f : ℕ+ → ℤ) : Prop :=
  ∀ {m n : ℕ+}, m < n → f m < f n

axiom h1 : is_increasing f
axiom h2 : f 4 = 5
axiom h3 : ∀ n : ℕ+, ∃ k : ℕ, f n = k
axiom h4 : ∀ m n : ℕ+, f m * f n = f (m * n) + f (m + n - 1)

-- Proof in Lean 4
theorem find_f_values : f 1 = 2 ∧ f 2 = 3 ∧ f 3 = 4 :=
by
  sorry

theorem find_f_expression : ∀ n : ℕ+, f n = n + 1 :=
by
  sorry

end find_f_values_find_f_expression_l290_290164


namespace vasya_numbers_l290_290812

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290812


namespace current_number_of_women_is_24_l290_290732

-- Define initial person counts based on the given ratio and an arbitrary factor x.
variables (x : ℕ)
def M_initial := 4 * x
def W_initial := 5 * x
def C_initial := 3 * x
def E_initial := 2 * x

-- Define the changes that happened to the room.
def men_after_entry := M_initial x + 2
def women_after_leaving := W_initial x - 3
def women_after_doubling := 2 * women_after_leaving x
def children_after_leaving := C_initial x - 5
def elderly_after_leaving := E_initial x - 3

-- Define the current counts after all changes.
def men_current := 14
def children_current := 7
def elderly_current := 6

-- Prove that the current number of women is 24.
theorem current_number_of_women_is_24 :
  men_after_entry x = men_current ∧
  children_after_leaving x = children_current ∧
  elderly_after_leaving x = elderly_current →
  women_after_doubling x = 24 :=
by
  sorry

end current_number_of_women_is_24_l290_290732


namespace sum_of_coefficients_l290_290103

def P (x : ℝ) : ℝ := 3 * (x^8 - 2 * x^5 + x^3 - 7) - 5 * (x^6 + 3 * x^2 - 6) + 2 * (x^4 - 5)

theorem sum_of_coefficients : P 1 = -19 := by
  sorry

end sum_of_coefficients_l290_290103


namespace stockholm_to_malmo_distance_l290_290468
-- Import the necessary library

-- Define the parameters for the problem.
def map_distance : ℕ := 120 -- distance in cm
def scale_factor : ℕ := 12 -- km per cm

-- The hypothesis for the map distance and the scale factor
axiom map_distance_hyp : map_distance = 120
axiom scale_factor_hyp : scale_factor = 12

-- Define the real distance function
def real_distance (d : ℕ) (s : ℕ) : ℕ := d * s

-- The problem statement: Prove that the real distance between the two city centers is 1440 km
theorem stockholm_to_malmo_distance : real_distance map_distance scale_factor = 1440 :=
by
  rw [map_distance_hyp, scale_factor_hyp]
  sorry

end stockholm_to_malmo_distance_l290_290468


namespace diagonal_AC_length_l290_290594

theorem diagonal_AC_length (AB BC CD DA : ℝ) (angle_ADC : ℝ) (h_AB : AB = 12) (h_BC : BC = 12) 
(h_CD : CD = 13) (h_DA : DA = 13) (h_angle_ADC : angle_ADC = 60) : 
  AC = 13 := 
sorry

end diagonal_AC_length_l290_290594


namespace lim_sum_D_R_l290_290895

noncomputable def D_R (R : ℝ) : set (ℤ × ℤ) :=
  { p | 0 < p.1^2 + p.2^2 ∧ p.1^2 + p.2^2 < R }

theorem lim_sum_D_R :
  ∀ R : ℝ, R > 1 → (tendsto (λ R, ∑ p in D_R R, (↑(-1) ^ (p.1 + p.2) / (↑(p.1^2 + p.2^2) : ℝ)))
    at_top (𝓝 (-π * log 2))) :=
begin
  sorry
end

end lim_sum_D_R_l290_290895


namespace triangle_area_is_correct_l290_290892

noncomputable def area_of_triangle (A B C : (ℝ × ℝ)) : ℝ :=
  (1 / 2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

theorem triangle_area_is_correct : 
  area_of_triangle (1, 3) (5, -2) (8, 6) = 23.5 := 
by
  sorry

end triangle_area_is_correct_l290_290892


namespace probability_not_all_same_l290_290060

theorem probability_not_all_same :
    let total_outcomes := 6 ^ 5 in
    let same_number_outcomes := 6 in
    let p_all_same := same_number_outcomes / total_outcomes in
    let p_not_all_same := 1 - p_all_same in
    p_not_all_same = 1295 / 1296 :=
by
  sorry

end probability_not_all_same_l290_290060


namespace symmetric_point_correct_l290_290179

-- Define the point and line
def point : ℝ × ℝ := (-1, 2)
def line (x : ℝ) : ℝ := x - 1

-- Define a function that provides the symmetric point with respect to the line
def symmetric_point (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ × ℝ :=
  -- Since this function is a critical part of the problem, we won't define it explicitly. Using a placeholder.
  sorry

-- The proof problem
theorem symmetric_point_correct : symmetric_point point line = (3, -2) :=
  sorry

end symmetric_point_correct_l290_290179


namespace equal_segments_l290_290934

open Geometry

noncomputable def cyclic_quadrilateral (A B C D O : Point) : Prop := 
  inscribed_in A B C D O ∧
  ∃ M : Point, 
    midpoint_arc_ADC M A D C O ∧
    perpendicular (line_through A C) (line_through B D) ∧
    ∃ E F : Point, 
      ∃ circle_MO_Pass (circle_through M O D), 
      intersect_Pt E F (line_through D A) (line_through D C) (circle_through M O D)

theorem equal_segments
  (A B C D O : Point)
  (h1 : inscribed_in A B C D O)
  (h2 : ∃ M, midpoint_arc_ADC M A D C O)
  (h3 : perpendicular (line_through A C) (line_through B D))
  (h4 : ∃ E F, intersect_Pt E F (line_through D A) (line_through D C) (circle_through (exists_snd h2) O D)) :
  length (segment_through B E) = length (segment_through B F) := 
sorry

end equal_segments_l290_290934


namespace common_terms_count_l290_290266

theorem common_terms_count (β : ℕ) (h1 : β = 55) (h2 : β + 1 = 56) : 
  ∃ γ : ℕ, γ = 6 :=
by
  sorry

end common_terms_count_l290_290266


namespace find_n_l290_290287

open Nat

def is_solution_of_comb_perm (n : ℕ) : Prop :=
    3 * (factorial (n-1) / (factorial (n-5) * factorial 4)) = 5 * (n-2) * (n-3)

theorem find_n (n : ℕ) (h : is_solution_of_comb_perm n) (hn : n ≠ 0) : n = 9 :=
by
  -- will fill proof steps if required
  sorry

end find_n_l290_290287


namespace james_initial_friends_l290_290937

theorem james_initial_friends (x : ℕ) (h1 : 19 = x - 2 + 1) : x = 20 :=
  by sorry

end james_initial_friends_l290_290937


namespace solve_equation_l290_290993

theorem solve_equation (x : ℝ) :
  (x + 1)^2 = (2 * x - 1)^2 ↔ (x = 0 ∨ x = 2) :=
by
  sorry

end solve_equation_l290_290993


namespace james_proof_l290_290938

def james_pages_per_hour 
  (writes_some_pages_an_hour : ℕ)
  (writes_5_pages_to_2_people_each_day : ℕ)
  (hours_spent_writing_per_week : ℕ) 
  (writes_total_pages_per_day : ℕ)
  (writes_total_pages_per_week : ℕ) 
  (pages_per_hour : ℕ) 
: Prop :=
  writes_some_pages_an_hour = writes_5_pages_to_2_people_each_day / hours_spent_writing_per_week

theorem james_proof
  (writes_some_pages_an_hour : ℕ := 10)
  (writes_5_pages_to_2_people_each_day : ℕ := 5 * 2)
  (hours_spent_writing_per_week : ℕ := 7)
  (writes_total_pages_per_day : ℕ := writes_5_pages_to_2_people_each_day)
  (writes_total_pages_per_week : ℕ := writes_total_pages_per_day * 7)
  (pages_per_hour : ℕ := writes_total_pages_per_week / hours_spent_writing_per_week)
: writes_some_pages_an_hour = pages_per_hour :=
by {
  sorry 
}

end james_proof_l290_290938


namespace area_larger_sphere_l290_290091

noncomputable def sphere_area_relation (A1: ℝ) (R1 R2: ℝ) := R2^2 / R1^2 * A1

-- Given Conditions
def radius_smaller_sphere : ℝ := 4.0  -- R1
def radius_larger_sphere : ℝ := 6.0    -- R2
def area_smaller_sphere : ℝ := 17.0    -- A1

-- Target Area Calculation based on Proportional Relationship
theorem area_larger_sphere :
  sphere_area_relation area_smaller_sphere radius_smaller_sphere radius_larger_sphere = 38.25 :=
by
  sorry

end area_larger_sphere_l290_290091


namespace ab_sum_eq_2_l290_290024

theorem ab_sum_eq_2 (a b : ℝ) (M : Set ℝ) (N : Set ℝ) (f : ℝ → ℝ) 
  (hM : M = {b / a, 1})
  (hN : N = {a, 0})
  (hf : ∀ x ∈ M, f x ∈ N)
  (f_def : ∀ x, f x = 2 * x) :
  a + b = 2 :=
by
  -- proof goes here.
  sorry

end ab_sum_eq_2_l290_290024


namespace find_coordinates_l290_290414

def A : Prod ℤ ℤ := (-3, 2)
def move_right (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst + 1, p.snd)
def move_down (p : Prod ℤ ℤ) : Prod ℤ ℤ := (p.fst, p.snd - 2)

theorem find_coordinates :
  move_down (move_right A) = (-2, 0) :=
by
  sorry

end find_coordinates_l290_290414


namespace probability_not_all_same_l290_290062

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l290_290062


namespace vasya_numbers_l290_290806

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290806


namespace max_number_of_girls_l290_290951

/-!
# Ballet Problem
Prove the maximum number of girls that can be positioned such that each girl is exactly 5 meters away from two distinct boys given that 5 boys are participating.
-/

theorem max_number_of_girls (boys : ℕ) (h_boys : boys = 5) : 
  ∃ girls : ℕ, girls = 20 ∧ ∀ g ∈ range girls, ∃ b1 b2 ∈ range boys, dist g b1 = 5 ∧ dist g b2 = 5 := 
sorry

end max_number_of_girls_l290_290951


namespace t_plus_reciprocal_l290_290902

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l290_290902


namespace max_x_add_2y_l290_290743

theorem max_x_add_2y (x y : ℝ) (h1 : 4 * x + 3 * y ≤ 10) (h2 : 3 * x + 6 * y ≤ 12) :
  x + 2 * y ≤ 4 :=
sorry

end max_x_add_2y_l290_290743


namespace determine_A_l290_290225

noncomputable def is_single_digit (n : ℕ) : Prop := n < 10

theorem determine_A (A B C : ℕ) (hABC : 3 * (100 * A + 10 * B + C) = 888)
  (hA_single_digit : is_single_digit A) (hB_single_digit : is_single_digit B) (hC_single_digit : is_single_digit C)
  (h_different : A ≠ B ∧ B ≠ C ∧ A ≠ C) : A = 2 := 
  sorry

end determine_A_l290_290225


namespace function_identity_l290_290891

theorem function_identity (f : ℕ → ℕ) 
  (h1 : f 2 = 2)
  (h2 : ∀ m n : ℕ, f (m * n) = f m * f n)
  (h3 : ∀ n : ℕ, f (n + 1) > f n) : 
  ∀ n : ℕ, f n = n :=
sorry

end function_identity_l290_290891


namespace verify_A_l290_290676

def matrix_A : Matrix (Fin 2) (Fin 2) ℚ :=
  ![![62 / 7, -9 / 7], ![2 / 7, 17 / 7]]

theorem verify_A :
  matrix_A.mulVec ![1, 3] = ![5, 7] ∧
  matrix_A.mulVec ![-2, 1] = ![-19, 3] :=
by
  sorry

end verify_A_l290_290676


namespace nails_needed_for_house_wall_l290_290256

theorem nails_needed_for_house_wall :
  let large_planks : Nat := 13
  let nails_per_large_plank : Nat := 17
  let additional_nails : Nat := 8
  large_planks * nails_per_large_plank + additional_nails = 229 := by
  sorry

end nails_needed_for_house_wall_l290_290256


namespace no_2014_ambiguous_integer_exists_l290_290090

theorem no_2014_ambiguous_integer_exists :
  ∀ k : ℕ, (∃ m : ℤ, k^2 - 8056 = m^2) → (∃ n : ℤ, k^2 + 8056 = n^2) → false :=
by
  -- Proof is omitted as per the instructions
  sorry

end no_2014_ambiguous_integer_exists_l290_290090


namespace vasya_numbers_l290_290811

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290811


namespace min_people_same_score_l290_290011

theorem min_people_same_score (participants : ℕ) (nA nB : ℕ) (pointsA pointsB : ℕ) (scores : Finset ℕ) :
  participants = 400 →
  nA = 8 →
  nB = 6 →
  pointsA = 4 →
  pointsB = 7 →
  scores.card = (nA + 1) * (nB + 1) - 6 →
  participants / scores.card < 8 :=
by
  intros h_participants h_nA h_nB h_pointsA h_pointsB h_scores_card
  sorry

end min_people_same_score_l290_290011


namespace correct_option_l290_290843

-- Definitions based on conditions in the problem
def optionA (a b : ℕ) : Prop := 3 * a * b - 2 * a * b = a * b
def optionB (y : ℕ) : Prop := 6 * y^2 - 2 * y^2 = 4
def optionC (a : ℕ) : Prop := 5 * a + a = 5 * a^2
def optionD (m n : ℕ) : Prop := m^2 * n - 3 * m * n^2 = -2 * m * n^2

-- The goal is to prove the correctness of Option A and the incorrectness of others
theorem correct_option (a b y m n : ℕ) : optionA a b ∧ ¬optionB y ∧ ¬optionC a ∧ ¬optionD m n :=
by {
  sorry -- proof goes here
}

end correct_option_l290_290843


namespace age_of_b_is_6_l290_290182

theorem age_of_b_is_6 (x : ℕ) (h1 : 5 * x / 3 * x = 5 / 3)
                         (h2 : (5 * x + 2) / (3 * x + 2) = 3 / 2) : 3 * x = 6 := 
by
  sorry

end age_of_b_is_6_l290_290182


namespace angle_in_first_quadrant_l290_290342

-- Define the condition and equivalence proof problem in Lean 4
theorem angle_in_first_quadrant (deg : ℤ) (h1 : deg = 721) : (deg % 360) > 0 := 
by 
  have : deg % 360 = 1 := sorry
  exact sorry

end angle_in_first_quadrant_l290_290342


namespace num_terms_simplified_expression_l290_290626

theorem num_terms_simplified_expression (x y z : ℕ) :
  let expr := ((x + y + z) ^ 2006 + (x - y - z) ^ 2006)
  term_count expr = 1008016 :=
begin
  -- The proof would go here
  sorry
end

end num_terms_simplified_expression_l290_290626


namespace mel_weight_l290_290233

variable (m : ℕ)

/-- Brenda's weight is 10 pounds more than three times Mel's weight. 
    Given Brenda's weight is 220 pounds, we prove Mel's weight is 70 pounds. -/
theorem mel_weight : (3 * m + 10 = 220) → (m = 70) :=
by
  intros h,
  sorry

end mel_weight_l290_290233


namespace number_of_subsets_of_set_A_l290_290475

theorem number_of_subsets_of_set_A : 
  (setOfSubsets : Finset (Finset ℕ)) = Finset.powerset {2, 4, 5} → 
  setOfSubsets.card = 8 :=
by
  sorry

end number_of_subsets_of_set_A_l290_290475


namespace find_a_l290_290392

noncomputable section

def f (x a : ℝ) : ℝ := Real.sqrt (1 + a * 4^x)

theorem find_a (a : ℝ) : 
  (∀ (x : ℝ), x ≤ -1 → 1 + a * 4^x ≥ 0) → a = -4 :=
sorry

end find_a_l290_290392


namespace professor_oscar_review_questions_l290_290865

-- Define the problem conditions.
def students_per_class : ℕ := 35
def questions_per_exam : ℕ := 10
def number_of_classes : ℕ := 5

-- Define the number of questions that must be reviewed.
def total_questions_to_review : ℕ := 1750

-- The theorem to be proved.
theorem professor_oscar_review_questions :
  students_per_class * questions_per_exam * number_of_classes = total_questions_to_review :=
by
  -- Here we write 'sorry' since we are not providing the full proof.
  sorry

end professor_oscar_review_questions_l290_290865


namespace q_can_be_true_or_false_l290_290297

theorem q_can_be_true_or_false (p q : Prop) (h1 : ¬(p ∧ q)) (h2 : ¬p) : q ∨ ¬q :=
by
  sorry

end q_can_be_true_or_false_l290_290297


namespace total_distance_traveled_l290_290987

theorem total_distance_traveled 
  (Vm : ℝ) (Vr : ℝ) (T_total : ℝ) (D : ℝ) 
  (H_Vm : Vm = 6) 
  (H_Vr : Vr = 1.2) 
  (H_T_total : T_total = 1) 
  (H_time_eq : D / (Vm - Vr) + D / (Vm + Vr) = T_total) 
  : 2 * D = 5.76 := 
by sorry

end total_distance_traveled_l290_290987


namespace second_more_than_third_l290_290081

def firstChapterPages : ℕ := 35
def secondChapterPages : ℕ := 18
def thirdChapterPages : ℕ := 3

theorem second_more_than_third : secondChapterPages - thirdChapterPages = 15 := by
  sorry

end second_more_than_third_l290_290081


namespace even_number_of_divisors_less_than_100_l290_290397

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k, k * k = n

theorem even_number_of_divisors_less_than_100 :
  (card { n : ℕ | 1 ≤ n ∧ n < 100 ∧ ¬ is_perfect_square n }) = 90 :=
by
  sorry

end even_number_of_divisors_less_than_100_l290_290397


namespace expected_rank_of_winner_in_tournament_l290_290850

noncomputable def expected_winner_rank (n : ℕ) (p : ℝ) : ℝ :=
  2^n - 2^n * p + p

theorem expected_rank_of_winner_in_tournament :
  expected_winner_rank 8 (3/5) = 103 := by
    sorry

end expected_rank_of_winner_in_tournament_l290_290850


namespace complementary_angle_beta_l290_290559

theorem complementary_angle_beta (α β : ℝ) (h_compl : α + β = 90) (h_alpha : α = 40) : β = 50 :=
by
  -- Skipping the proof, which initial assumption should be defined.
  sorry

end complementary_angle_beta_l290_290559


namespace maximumNumberOfGirls_l290_290950

theorem maximumNumberOfGirls {B : Finset ℕ} (hB : B.card = 5) :
  ∃ G : Finset ℕ, ∀ g ∈ G, ∃ b1 b2 : ℕ, b1 ≠ b2 ∧ b1 ∈ B ∧ b2 ∈ B ∧ dist g b1 = 5 ∧ dist g b2 = 5 ∧ G.card = 20 :=
sorry

end maximumNumberOfGirls_l290_290950


namespace tunnel_length_l290_290094

/-- A train travels at 80 kmph, enters a tunnel at 5:12 am, and leaves at 5:18 am.
    The length of the train is 1 km. Prove the length of the tunnel is 7 km. -/
theorem tunnel_length 
(speed : ℕ) (enter_time leave_time : ℕ) (train_length : ℕ) 
(h_enter : enter_time = 5 * 60 + 12) 
(h_leave : leave_time = 5 * 60 + 18) 
(h_speed : speed = 80) 
(h_train_length : train_length = 1) 
: ∃ tunnel_length : ℕ, tunnel_length = 7 :=
sorry

end tunnel_length_l290_290094


namespace min_value_of_E_l290_290638

noncomputable def E : ℝ := sorry

theorem min_value_of_E :
  (∀ x : ℝ, |E| + |x + 7| + |x - 5| ≥ 12) →
  (∃ x : ℝ, |x + 7| + |x - 5| = 12 → |E| = 0) :=
sorry

end min_value_of_E_l290_290638


namespace students_just_passed_l290_290847

theorem students_just_passed (total_students : ℕ) (first_division : ℕ) (second_division : ℕ) (just_passed : ℕ)
  (h1 : total_students = 300)
  (h2 : first_division = 26 * total_students / 100)
  (h3 : second_division = 54 * total_students / 100)
  (h4 : just_passed = total_students - (first_division + second_division)) :
  just_passed = 60 :=
sorry

end students_just_passed_l290_290847


namespace georgia_carnations_proof_l290_290455

-- Define the conditions
def carnation_cost : ℝ := 0.50
def dozen_cost : ℝ := 4.00
def friends_carnations : ℕ := 14
def total_spent : ℝ := 25.00

-- Define the answer
def teachers_dozen : ℕ := 4

-- Prove the main statement
theorem georgia_carnations_proof : 
  (total_spent - (friends_carnations * carnation_cost)) / dozen_cost = teachers_dozen :=
by
  sorry

end georgia_carnations_proof_l290_290455


namespace contradiction_even_odd_l290_290980

theorem contradiction_even_odd (a b c : ℕ) :
  (∃ x y z, (x = a ∧ y = b ∧ z = c) ∧ (¬((x % 2 = 0 ∧ y % 2 ≠ 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 = 0 ∧ z % 2 ≠ 0) ∨ 
                                          (x % 2 ≠ 0 ∧ y % 2 ≠ 0 ∧ z % 2 = 0)))) → false :=
by
  sorry

end contradiction_even_odd_l290_290980


namespace T_53_eq_38_l290_290675

def T (x y : ℕ) : ℕ := 4 * x + 6 * y

theorem T_53_eq_38 : T 5 3 = 38 := by
  sorry

end T_53_eq_38_l290_290675


namespace length_of_first_train_l290_290660

theorem length_of_first_train
  (speed_first : ℕ)
  (speed_second : ℕ)
  (length_second : ℕ)
  (distance_between : ℕ)
  (time_to_cross : ℕ)
  (h1 : speed_first = 10)
  (h2 : speed_second = 15)
  (h3 : length_second = 150)
  (h4 : distance_between = 50)
  (h5 : time_to_cross = 60) :
  ∃ L : ℕ, L = 100 :=
by
  sorry

end length_of_first_train_l290_290660


namespace equation_has_unique_integer_solution_l290_290546

theorem equation_has_unique_integer_solution:
  ∀ m n : ℤ, (5 + 3 * Real.sqrt 2) ^ m = (3 + 5 * Real.sqrt 2) ^ n → m = 0 ∧ n = 0 := by
  intro m n
  -- The proof is omitted
  sorry

end equation_has_unique_integer_solution_l290_290546


namespace sales_first_month_l290_290996

theorem sales_first_month (S1 S2 S3 S4 S5 S6 : ℝ) 
  (h2 : S2 = 7000) (h3 : S3 = 6800) (h4 : S4 = 7200) (h5 : S5 = 6500) (h6 : S6 = 5100)
  (avg : (S1 + S2 + S3 + S4 + S5 + S6) / 6 = 6500) : S1 = 6400 := by
  sorry

end sales_first_month_l290_290996


namespace constant_term_l290_290588

theorem constant_term (n r : ℕ) (h₁ : n = 9) (h₂ : r = 3) : 
  nat.choose n r = 84 :=
by
  rw [h₁, h₂]
  exact nat.choose_eq_factorial_div_factorial (by norm_num) (by norm_num)
  sorry

end constant_term_l290_290588


namespace polygon_area_is_correct_l290_290672

def points : List (ℕ × ℕ) := [
  (0, 0), (10, 0), (20, 0), (30, 10),
  (0, 20), (10, 20), (20, 30), (10, 30),
  (0, 30), (20, 10), (30, 20), (10, 10)
]

def polygon_area (ps : List (ℕ × ℕ)) : ℕ := sorry

theorem polygon_area_is_correct :
  polygon_area points = 9 := sorry

end polygon_area_is_correct_l290_290672


namespace division_of_fractions_l290_290280

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l290_290280


namespace total_parents_in_auditorium_l290_290185

def num_girls : ℕ := 6
def num_boys : ℕ := 8
def parents_per_child : ℕ := 2

theorem total_parents_in_auditorium (num_girls num_boys parents_per_child : ℕ) : num_girls + num_boys = 14 → 2 * (num_girls + num_boys) = 28 := by
  assume h: num_girls + num_boys = 14
  show 2 * (num_girls + num_boys) = 28, from
    calc
      2 * (num_girls + num_boys) = 2 * 14 : by rw h
      ... = 28 : by norm_num

end total_parents_in_auditorium_l290_290185


namespace find_a_l290_290604

variable a : ℝ

def A := {a^2, a+1, -3}
def B := {a-3, 2a-1, a^2+1}

theorem find_a (h : A ∩ B = {-3}) : a = -1 :=
by
  sorry

end find_a_l290_290604


namespace multiple_of_persons_l290_290762

variable (Persons Work : ℕ) (Rate : ℚ)

def work_rate (P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D
def multiple_work_rate (m P : ℕ) (W : ℕ) (D : ℕ) : ℚ := W / D

theorem multiple_of_persons
  (P : ℕ) (W : ℕ)
  (h1 : work_rate P W 12 = W / 12)
  (h2 : multiple_work_rate 1 P (W / 2) 3 = (W / 6)) :
  m = 2 :=
by sorry

end multiple_of_persons_l290_290762


namespace calculate_savings_l290_290320

noncomputable def monthly_salary : ℕ := 10000
noncomputable def spent_on_food (S : ℕ) : ℕ := (40 * S) / 100
noncomputable def spent_on_rent (S : ℕ) : ℕ := (20 * S) / 100
noncomputable def spent_on_entertainment (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def spent_on_conveyance (S : ℕ) : ℕ := (10 * S) / 100
noncomputable def total_spent (S : ℕ) : ℕ := spent_on_food S + spent_on_rent S + spent_on_entertainment S + spent_on_conveyance S
noncomputable def amount_saved (S : ℕ) : ℕ := S - total_spent S

theorem calculate_savings : amount_saved monthly_salary = 2000 :=
by
  sorry

end calculate_savings_l290_290320


namespace derivative_of_y_l290_290253

noncomputable def y (x : ℝ) : ℝ :=
  -1/4 * Real.arcsin ((5 + 3 * Real.cosh x) / (3 + 5 * Real.cosh x))

theorem derivative_of_y (x : ℝ) :
  deriv y x = 1 / (3 + 5 * Real.cosh x) :=
sorry

end derivative_of_y_l290_290253


namespace xy_ratio_l290_290557

variables (x y z t : ℝ)
variables (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t))

theorem xy_ratio (x y : ℝ) (hx : x > y) (hz : z = (x + y) / 2) (ht : t = Real.sqrt (x * y)) (h : x - y = 3 * (z - t)) :
  x / y = 25 :=
sorry

end xy_ratio_l290_290557


namespace find_total_bricks_l290_290234

variable (y : ℕ)
variable (B_rate : ℕ)
variable (N_rate : ℕ)
variable (eff_rate : ℕ)
variable (time : ℕ)
variable (reduction : ℕ)

-- The wall is completed in 6 hours
def completed_in_time (y B_rate N_rate eff_rate time reduction : ℕ) : Prop := 
  time = 6 ∧
  reduction = 8 ∧
  B_rate = y / 8 ∧
  N_rate = y / 12 ∧
  eff_rate = (B_rate + N_rate) - reduction ∧
  y = eff_rate * time

-- Prove that the number of bricks in the wall is 192
theorem find_total_bricks : 
  ∀ (y B_rate N_rate eff_rate time reduction : ℕ), 
  completed_in_time y B_rate N_rate eff_rate time reduction → 
  y = 192 := 
by 
  sorry

end find_total_bricks_l290_290234


namespace only_solutions_l290_290370

theorem only_solutions (m n : ℕ) (p : ℕ) (prime_p : Nat.Prime p) (condition : (Nat.choose m 2) - 1 = p^n) :
  (m = 5 ∧ n = 2 ∧ p = 3) ∨ (m = 8 ∧ n = 3 ∧ p = 3) :=
by
  sorry

end only_solutions_l290_290370


namespace cash_realized_without_brokerage_l290_290323

theorem cash_realized_without_brokerage
  (C : ℝ)
  (h1 : (1 / 4) * (1 / 100) = 1 / 400)
  (h2 : C + (C / 400) = 108) :
  C = 43200 / 401 :=
by
  sorry

end cash_realized_without_brokerage_l290_290323


namespace number_of_distinct_triangle_areas_l290_290317

noncomputable def distinct_triangle_area_counts : ℕ :=
sorry  -- Placeholder for the proof to derive the correct answer

theorem number_of_distinct_triangle_areas
  (G H I J K L : ℝ × ℝ)
  (h₁ : G.2 = H.2)
  (h₂ : G.2 = I.2)
  (h₃ : G.2 = J.2)
  (h₄ : H.2 = I.2)
  (h₅ : H.2 = J.2)
  (h₆ : I.2 = J.2)
  (h₇ : dist G H = 2)
  (h₈ : dist H I = 2)
  (h₉ : dist I J = 2)
  (h₁₀ : K.2 = L.2 - 2)  -- Assuming constant perpendicular distance between parallel lines
  (h₁₁ : dist K L = 2) : 
  distinct_triangle_area_counts = 3 :=
sorry  -- Placeholder for the proof

end number_of_distinct_triangle_areas_l290_290317


namespace goats_difference_l290_290228

-- Definitions of Adam's, Andrew's, and Ahmed's goats
def adam_goats : ℕ := 7
def andrew_goats : ℕ := 2 * adam_goats + 5
def ahmed_goats : ℕ := 13

-- The theorem to prove the difference in goats
theorem goats_difference : andrew_goats - ahmed_goats = 6 :=
by
  sorry

end goats_difference_l290_290228


namespace only_one_real_solution_l290_290389

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem only_one_real_solution (a : ℝ) (h : ∀ x : ℝ, abs (f x) = g a x → x = 1) : a < 0 := 
by
  sorry

end only_one_real_solution_l290_290389


namespace min_value_of_x_l290_290580

open Real

-- Defining the conditions
def condition1 (x : ℝ) : Prop := x > 0
def condition2 (x : ℝ) : Prop := log x ≥ 2 * log 3 + (1/3) * log x

-- Statement of the theorem
theorem min_value_of_x (x : ℝ) (h1 : condition1 x) (h2 : condition2 x) : x ≥ 27 :=
sorry

end min_value_of_x_l290_290580


namespace probability_multiple_of_6_or_8_l290_290034

theorem probability_multiple_of_6_or_8
  (n : ℕ)
  (h_n : n = 60)
  (multiples_6_count : ℕ)
  (h_multiples_6 : multiples_6_count = 10)
  (multiples_8_count : ℕ)
  (h_multiples_8 : multiples_8_count = 7)
  (multiples_6_and_8_count : ℕ)
  (h_multiples_6_and_8 : multiples_6_and_8_count = 2) :
  (15 / 60 : ℚ) = 1 / 4 := sorry

end probability_multiple_of_6_or_8_l290_290034


namespace percentage_male_red_ants_proof_l290_290147

noncomputable def percentage_red_ants : ℝ := 0.85
noncomputable def percentage_female_red_ants : ℝ := 0.45
noncomputable def percentage_male_red_ants : ℝ := percentage_red_ants * (1 - percentage_female_red_ants)

theorem percentage_male_red_ants_proof : percentage_male_red_ants = 0.4675 :=
by
  -- Proof will go here
  sorry

end percentage_male_red_ants_proof_l290_290147


namespace least_k_for_sum_divisible_l290_290313

theorem least_k_for_sum_divisible (n : ℕ) (hn : n > 0) : 
  (∃ k : ℕ, (∀ (xs : List ℕ), (xs.length = k) → (∃ ys : List ℕ, (ys.length % 2 = 0) ∧ (ys.sum % n = 0))) ∧ 
    (k = if n % 2 = 1 then 2 * n else n + 1)) :=
sorry

end least_k_for_sum_divisible_l290_290313


namespace salary_increase_after_three_years_l290_290949

-- Define the initial salary S and the raise percentage 12%
def initial_salary (S : ℝ) : ℝ := S
def raise_percentage : ℝ := 0.12

-- Define the salary after n raises
def salary_after_raises (S : ℝ) (n : ℕ) : ℝ :=
  S * (1 + raise_percentage)^n

-- Prove that the percentage increase after 3 years is 40.49%
theorem salary_increase_after_three_years (S : ℝ) :
  ((salary_after_raises S 3 - S) / S) * 100 = 40.49 :=
by sorry

end salary_increase_after_three_years_l290_290949


namespace square_root_properties_l290_290698

theorem square_root_properties (a : ℝ) (h : a^2 = 10) :
  a = Real.sqrt 10 ∧ (a^2 - 10 = 0) ∧ (3 < a ∧ a < 4) :=
by sorry

end square_root_properties_l290_290698


namespace total_gallons_l290_290682

def gallons_used (A F : ℕ) := F = 4 * A - 5

theorem total_gallons
  (A F : ℕ)
  (h1 : gallons_used A F)
  (h2 : F = 23) :
  A + F = 30 :=
by
  sorry

end total_gallons_l290_290682


namespace not_p_is_sufficient_but_not_necessary_for_not_q_l290_290706

variable (x : ℝ)

def proposition_p : Prop := |x| < 2
def proposition_q : Prop := x^2 - x - 2 < 0

theorem not_p_is_sufficient_but_not_necessary_for_not_q :
  (¬ proposition_p x) → (¬ proposition_q x) ∧ (¬ proposition_q x) → (¬ proposition_p x) → False := by
  sorry

end not_p_is_sufficient_but_not_necessary_for_not_q_l290_290706


namespace largest_circle_area_in_region_S_l290_290169

-- Define the region S
def region_S (x y : ℝ) : Prop :=
  |x + (1 / 2) * y| ≤ 10 ∧ |x| ≤ 10 ∧ |y| ≤ 10

-- The question is to determine the value of k such that the area of the largest circle 
-- centered at (0, 0) fitting inside region S is k * π.
theorem largest_circle_area_in_region_S :
  ∃ k : ℝ, k = 80 :=
sorry

end largest_circle_area_in_region_S_l290_290169


namespace evaluate_g_expression_l290_290106

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g_expression :
  3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_expression_l290_290106


namespace cone_sphere_ratio_l290_290520

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4 / 3) * Real.pi * r^3

noncomputable def volume_of_cone (r : ℝ) (h : ℝ) : ℝ :=
  (1 / 3) * Real.pi * (2 * r)^2 * h

theorem cone_sphere_ratio (r h : ℝ) (V_cone V_sphere : ℝ) (h_sphere : V_sphere = volume_of_sphere r)
  (h_cone : V_cone = volume_of_cone r h) (h_relation : V_cone = (1/3) * V_sphere) :
  (h / (2 * r) = 1 / 6) :=
by
  sorry

end cone_sphere_ratio_l290_290520


namespace minimum_red_beads_l290_290216

theorem minimum_red_beads (n : ℕ) (r : ℕ) (necklace : ℕ → Prop) :
  (necklace = λ k, n * k + r) 
  → (∀ i, (segment_contains_blue i 8 → segment_contains_red i 4))
  → (cyclic_beads necklace)
  → r ≥ 29 :=
by
  sorry

-- Definitions to support the theorem
def segment_contains_blue (i : ℕ) (b : ℕ) : Prop := 
sorry -- Placeholder for the predicate that checks if a segment contains exactly 'b' blue beads.

def segment_contains_red (i : ℕ) (r : ℕ) : Prop := 
sorry -- Placeholder for the predicate that checks if a segment contains at least 'r' red beads.

def cyclic_beads (necklace : ℕ → Prop) : Prop := 
sorry -- Placeholder for the property that defines the necklace as cyclic.

end minimum_red_beads_l290_290216


namespace ratio_of_green_to_yellow_l290_290308

def envelopes_problem (B Y G X : ℕ) : Prop :=
  B = 14 ∧
  Y = B - 6 ∧
  G = X * Y ∧
  B + Y + G = 46 ∧
  G / Y = 3

theorem ratio_of_green_to_yellow :
  ∃ B Y G X : ℕ, envelopes_problem B Y G X :=
by
  sorry

end ratio_of_green_to_yellow_l290_290308


namespace eccentricity_range_l290_290388

def hyperbola (a b x y : ℝ) : Prop := (a > 0) ∧ (b > 0) ∧ (x^2 / a^2 - y^2 / b^2 = 1)
def right_branch_hyperbola_P (a b c x y : ℝ) : Prop := hyperbola a b x y ∧ (c = a) ∧ (2 * c = a)

theorem eccentricity_range {a b c : ℝ} (h: hyperbola a b c c) (h1 : 2 * a = 2 * c) (h2 : c = a) :
  1 < (c / a) ∧ (c / a) ≤ (Real.sqrt 10 / 2 : ℝ) := by
  sorry

end eccentricity_range_l290_290388


namespace eliminate_xy_l290_290678

variable {R : Type*} [Ring R]

theorem eliminate_xy
  (x y a b c : R)
  (h1 : a = x + y)
  (h2 : b = x^3 + y^3)
  (h3 : c = x^5 + y^5) :
  5 * b * (a^3 + b) = a * (a^5 + 9 * c) :=
sorry

end eliminate_xy_l290_290678


namespace even_number_of_divisors_less_than_100_l290_290399

theorem even_number_of_divisors_less_than_100 : 
  ∃ n, n = 90 ∧ ∀ x < 100, (∃ k, k * k = x → false) = (x ∣ 99 - 9) :=
sorry

end even_number_of_divisors_less_than_100_l290_290399


namespace allison_uploads_480_hours_in_june_l290_290522

noncomputable def allison_upload_total_hours : Nat :=
  let before_june_16 := 10 * 15
  let from_june_16_to_23 := 15 * 8
  let from_june_24_to_end := 30 * 7
  before_june_16 + from_june_16_to_23 + from_june_24_to_end

theorem allison_uploads_480_hours_in_june :
  allison_upload_total_hours = 480 := by
  sorry

end allison_uploads_480_hours_in_june_l290_290522


namespace problem_1_problem_2_l290_290394

def M : Set ℕ := {0, 1}

def A := { p : ℕ × ℕ | p.fst ∈ M ∧ p.snd ∈ M }

def B := { p : ℕ × ℕ | p.snd = 1 - p.fst }

theorem problem_1 : A = {(0,0), (0,1), (1,0), (1,1)} :=
by
  sorry

theorem problem_2 : 
  let AB := { p ∈ A | p ∈ B }
  AB = {(1,0), (0,1)} ∧
  {S : Set (ℕ × ℕ) | S ⊆ AB} = {∅, {(1,0)}, {(0,1)}, {(1,0), (0,1)}} :=
by
  sorry

end problem_1_problem_2_l290_290394


namespace total_notes_l290_290305

theorem total_notes :
  let red_notes := 5 * 6 in
  let blue_notes_under_red := 2 * red_notes in
  let total_blue_notes := blue_notes_under_red + 10 in
  red_notes + total_blue_notes = 100 := by
  sorry

end total_notes_l290_290305


namespace containers_needed_l290_290611

-- Define the conditions: 
def weight_in_pounds : ℚ := 25 / 2
def ounces_per_pound : ℚ := 16
def ounces_per_container : ℚ := 50

-- Define the total weight in ounces
def total_weight_in_ounces := weight_in_pounds * ounces_per_pound

-- Theorem statement: Number of containers.
theorem containers_needed : total_weight_in_ounces / ounces_per_container = 4 := 
by
  -- Write the proof here
  sorry

end containers_needed_l290_290611


namespace average_weight_of_class_is_61_67_l290_290482

noncomputable def totalWeightA (avgWeightA : ℝ) (numStudentsA : ℕ) : ℝ := avgWeightA * numStudentsA
noncomputable def totalWeightB (avgWeightB : ℝ) (numStudentsB : ℕ) : ℝ := avgWeightB * numStudentsB
noncomputable def totalWeightClass (totalWeightA : ℝ) (totalWeightB : ℝ) : ℝ := totalWeightA + totalWeightB
noncomputable def totalStudentsClass (numStudentsA : ℕ) (numStudentsB : ℕ) : ℕ := numStudentsA + numStudentsB
noncomputable def averageWeightClass (totalWeightClass : ℝ) (totalStudentsClass : ℕ) : ℝ := totalWeightClass / totalStudentsClass

theorem average_weight_of_class_is_61_67 :
  averageWeightClass (totalWeightClass (totalWeightA 50 50) (totalWeightB 70 70))
    (totalStudentsClass 50 70) = 61.67 := by
  sorry

end average_weight_of_class_is_61_67_l290_290482


namespace geometric_sequence_a5_eq_neg1_l290_290560

-- Definitions for the conditions
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q, ∀ n, a (n + 1) = a n * q

def roots_of_quadratic (a3 a7 : ℝ) : Prop :=
  a3 + a7 = -4 ∧ a3 * a7 = 1

-- The statement to prove
theorem geometric_sequence_a5_eq_neg1 {a : ℕ → ℝ}
  (h_geo : is_geometric_sequence a)
  (h_roots : roots_of_quadratic (a 3) (a 7)) :
  a 5 = -1 :=
sorry

end geometric_sequence_a5_eq_neg1_l290_290560


namespace melanie_initial_plums_l290_290946

-- define the conditions as constants
def plums_given_to_sam : ℕ := 3
def plums_left_with_melanie : ℕ := 4

-- define the statement to be proven
theorem melanie_initial_plums : (plums_given_to_sam + plums_left_with_melanie = 7) :=
by
  sorry

end melanie_initial_plums_l290_290946


namespace log_xy_eq_5_over_11_l290_290005

-- Definitions of the conditions
axiom log_xy4_eq_one {x y : ℝ} : Real.log (x * y^4) = 1
axiom log_x3y_eq_one {x y : ℝ} : Real.log (x^3 * y) = 1

-- The statement to be proven
theorem log_xy_eq_5_over_11 {x y : ℝ} (h1 : Real.log (x * y^4) = 1) (h2 : Real.log (x^3 * y) = 1) :
  Real.log (x * y) = 5 / 11 :=
by
  sorry

end log_xy_eq_5_over_11_l290_290005


namespace jake_eats_papayas_in_one_week_l290_290435

variable (J : ℕ)
variable (brother_eats : ℕ := 5)
variable (father_eats : ℕ := 4)
variable (total_papayas_in_4_weeks : ℕ := 48)

theorem jake_eats_papayas_in_one_week (h : 4 * (J + brother_eats + father_eats) = total_papayas_in_4_weeks) : J = 3 :=
by
  sorry

end jake_eats_papayas_in_one_week_l290_290435


namespace intersection_point_l290_290536

theorem intersection_point (a b d x y : ℝ) (h1 : a = b + d) (h2 : a * x + b * y = b + 2 * d) :
    (x, y) = (-1, 1) :=
by
  sorry

end intersection_point_l290_290536


namespace vasya_numbers_l290_290797

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290797


namespace p_sufficient_but_not_necessary_for_q_l290_290704

theorem p_sufficient_but_not_necessary_for_q (x : ℝ) :
  (|x - 1| < 2 → x ^ 2 - 5 * x - 6 < 0) ∧ ¬ (x ^ 2 - 5 * x - 6 < 0 → |x - 1| < 2) :=
by
  sorry

end p_sufficient_but_not_necessary_for_q_l290_290704


namespace division_of_fractions_l290_290279

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l290_290279


namespace n_is_prime_l290_290978

variable {n : ℕ}

theorem n_is_prime (hn : n > 1) (hd : ∀ d : ℕ, d > 0 ∧ d ∣ n → d + 1 ∣ n + 1) :
  Prime n := 
sorry

end n_is_prime_l290_290978


namespace calculate_f_at_pi_div_6_l290_290549

noncomputable def f (x : ℝ) (ω φ : ℝ) : ℝ := 2 * Real.sin (ω * x + φ)

theorem calculate_f_at_pi_div_6 (ω φ : ℝ) 
  (h : ∀ x : ℝ, f (π / 3 + x) ω φ = f (-x) ω φ) :
  f (π / 6) ω φ = 2 ∨ f (π / 6) ω φ = -2 :=
sorry

end calculate_f_at_pi_div_6_l290_290549


namespace mark_weekly_reading_time_l290_290028

-- Define the conditions
def hours_per_day : ℕ := 2
def days_per_week : ℕ := 7
def additional_hours : ℕ := 4

-- State the main theorem to prove
theorem mark_weekly_reading_time : (hours_per_day * days_per_week) + additional_hours = 18 := 
by
  -- The proof steps are omitted as per instructions
  sorry

end mark_weekly_reading_time_l290_290028


namespace find_four_numbers_l290_290977

theorem find_four_numbers (a b c d : ℚ) :
  ((a + b = 1) ∧ (a + c = 5) ∧ 
   ((a + d = 8 ∧ b + c = 9) ∨ (a + d = 9 ∧ b + c = 8)) ) →
  ((a = -3/2 ∧ b = 5/2 ∧ c = 13/2 ∧ d = 19/2) ∨ 
   (a = -1 ∧ b = 2 ∧ c = 6 ∧ d = 10)) :=
  by
    sorry

end find_four_numbers_l290_290977


namespace probability_sum_less_than_product_l290_290793

def set_of_numbers := {1, 2, 3, 4, 5, 6, 7}

def count_valid_pairs : ℕ :=
  set_of_numbers.to_list.product set_of_numbers.to_list
    |>.count (λ (ab : ℕ × ℕ), (ab.1 - 1) * (ab.2 - 1) > 1)

def total_combinations := (set_of_numbers.to_list).length ^ 2

theorem probability_sum_less_than_product :
  (count_valid_pairs : ℚ) / total_combinations = 36 / 49 :=
by
  -- Placeholder for proof, since proof is not requested
  sorry

end probability_sum_less_than_product_l290_290793


namespace probability_of_non_adjacent_zeros_l290_290576

-- Define the total number of arrangements of 3 ones and 2 zeros
def totalArrangements : ℕ := Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

-- Define the number of arrangements where the 2 zeros are together
def adjacentZerosArrangements : ℕ := 2 * Nat.factorial 4 / (Nat.factorial 3 * Nat.factorial 1)

-- Calculate the desired probability
def nonAdjacentZerosProbability : ℚ := 
  1 - (adjacentZerosArrangements.toRat / totalArrangements.toRat)

theorem probability_of_non_adjacent_zeros :
  nonAdjacentZerosProbability = 3/5 :=
sorry

end probability_of_non_adjacent_zeros_l290_290576


namespace harry_blue_weights_l290_290873

theorem harry_blue_weights (B : ℕ) 
  (h1 : 2 * B + 17 = 25) : B = 4 :=
by {
  -- proof code here
  sorry
}

end harry_blue_weights_l290_290873


namespace cost_of_agricultural_equipment_max_units_of_type_A_l290_290079

-- Define cost equations
variables (x y : ℝ)

-- Define conditions as hypotheses
def condition1 : Prop := 2 * x + y = 4.2
def condition2 : Prop := x + 3 * y = 5.1

-- Prove the costs are respectively 1.5 and 1.2
theorem cost_of_agricultural_equipment (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 1.5 ∧ y = 1.2 := sorry

-- Define the maximum units constraint
def total_cost (m : ℕ) : ℝ := 1.5 * m + 1.2 * (2 * m - 3)

-- Prove the maximum units of type A is 3
theorem max_units_of_type_A (m : ℕ) (h : total_cost m ≤ 10) : m ≤ 3 := sorry

end cost_of_agricultural_equipment_max_units_of_type_A_l290_290079


namespace third_candidate_votes_l290_290933

-- Definition of the problem's conditions
variables (total_votes winning_votes candidate2_votes : ℕ)
variables (winning_percentage : ℚ)

-- Conditions given in the problem
def conditions : Prop :=
  winning_votes = 11628 ∧
  winning_percentage = 0.4969230769230769 ∧
  (total_votes : ℚ) = winning_votes / winning_percentage ∧
  candidate2_votes = 7636

-- The theorem we need to prove
theorem third_candidate_votes (total_votes winning_votes candidate2_votes : ℕ)
    (winning_percentage : ℚ)
    (h : conditions total_votes winning_votes candidate2_votes winning_percentage) :
    total_votes - (winning_votes + candidate2_votes) = 4136 := 
  sorry

end third_candidate_votes_l290_290933


namespace find_matrix_N_l290_290683

def matrix2x2 := ℚ × ℚ × ℚ × ℚ

def apply_matrix (M : matrix2x2) (v : ℚ × ℚ) : ℚ × ℚ :=
  let (a, b, c, d) := M;
  let (x, y) := v;
  (a * x + b * y, c * x + d * y)

theorem find_matrix_N : ∃ (N : matrix2x2), 
  apply_matrix N (3, 1) = (5, -1) ∧ 
  apply_matrix N (1, -2) = (0, 6) ∧ 
  N = (10/7, 5/7, 4/7, -19/7) :=
by {
  sorry
}

end find_matrix_N_l290_290683


namespace mean_of_set_is_12_point_8_l290_290767

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l290_290767


namespace bobby_shoes_l290_290360

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end bobby_shoes_l290_290360


namespace most_numerous_fruit_l290_290778

-- Define the number of boxes
def num_boxes_tangerines := 5
def num_boxes_apples := 3
def num_boxes_pears := 4

-- Define the number of fruits per box
def tangerines_per_box := 30
def apples_per_box := 20
def pears_per_box := 15

-- Calculate the total number of each fruit
def total_tangerines := num_boxes_tangerines * tangerines_per_box
def total_apples := num_boxes_apples * apples_per_box
def total_pears := num_boxes_pears * pears_per_box

-- State the theorem and prove it
theorem most_numerous_fruit :
  total_tangerines = 150 ∧ total_tangerines > total_apples ∧ total_tangerines > total_pears :=
by
  -- Add here the necessary calculations to verify the conditions
  sorry

end most_numerous_fruit_l290_290778


namespace number_composition_l290_290217

theorem number_composition :
  5 * 100000 + 6 * 100 + 3 * 10 + 6 * 0.01 = 500630.06 := 
by 
  sorry

end number_composition_l290_290217


namespace intersection_A_B_range_of_a_l290_290712

variable {A B C : Set ℝ}
variable {a : ℝ}

-- Given conditions
def f (x : ℝ) : ℝ := Real.sqrt (6 - 2 * x) + Real.log (x + 2)
def SetA : Set ℝ := {x | -2 < x ∧ x ≤ 3}
def SetB : Set ℝ := {x | x > 3 ∨ x < 2}
def SetC (a : ℝ) : Set ℝ := {x | x < 2 * a + 1}

-- Proof objectives
theorem intersection_A_B :
  (SetA ∩ SetB) = {x | -2 < x ∧ x < 2} := 
sorry

theorem range_of_a :
  ∀ {a : ℝ}, (SetC a ⊆ SetB) → a ≤ 1 / 2 := 
sorry

end intersection_A_B_range_of_a_l290_290712


namespace original_difference_in_books_l290_290852

theorem original_difference_in_books 
  (x y : ℕ) 
  (h1 : x + y = 5000) 
  (h2 : (1 / 2 : ℚ) * (x - 400) - (y + 400) = 400) : 
  x - y = 3000 := 
by 
  -- Placeholder for the proof
  sorry

end original_difference_in_books_l290_290852


namespace sequence_property_l290_290479

theorem sequence_property (k : ℝ) (h_k : 0 < k) (x : ℕ → ℝ)
  (h₀ : x 0 = 1)
  (h₁ : x 1 = 1 + k)
  (rec1 : ∀ n, x (2*n + 1) - x (2*n) = x (2*n) - x (2*n - 1))
  (rec2 : ∀ n, x (2*n) / x (2*n - 1) = x (2*n - 1) / x (2*n - 2)) :
  ∃ N, ∀ n ≥ N, x n > 1994 :=
by
  sorry

end sequence_property_l290_290479


namespace number_of_ways_to_assign_roles_l290_290856

theorem number_of_ways_to_assign_roles : 
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let men := 4
  let women := 5
  let total_roles := male_roles + female_roles + either_gender_roles
  let ways_to_assign_males := men * (men-1) * (men-2)
  let ways_to_assign_females := women * (women-1)
  let remaining_actors := men + women - male_roles - female_roles
  let ways_to_assign_either_gender := remaining_actors
  let total_ways := ways_to_assign_males * ways_to_assign_females * ways_to_assign_either_gender

  total_ways = 1920 :=
by
  sorry

end number_of_ways_to_assign_roles_l290_290856


namespace card_drawing_probability_l290_290691

theorem card_drawing_probability :
  let prob_of_ace_first := (4:ℚ) / 52,
      prob_of_2_second := (4:ℚ) / 51,
      prob_of_3_third := (4:ℚ) / 50,
      prob_of_4_fourth := (4:ℚ) / 49 in
  prob_of_ace_first * prob_of_2_second * prob_of_3_third * prob_of_4_fourth = 16 / 405525 :=
by
  sorry

end card_drawing_probability_l290_290691


namespace no_integer_solutions_19x2_minus_76y2_eq_1976_l290_290610

theorem no_integer_solutions_19x2_minus_76y2_eq_1976 :
  ∀ x y : ℤ, 19 * x^2 - 76 * y^2 ≠ 1976 :=
by sorry

end no_integer_solutions_19x2_minus_76y2_eq_1976_l290_290610


namespace factorable_quadratic_l290_290251

theorem factorable_quadratic (b : Int) : 
  (∃ m n p q : Int, 35 * m * p = 35 ∧ m * q + n * p = b ∧ n * q = 35) ↔ (∃ k : Int, b = 2 * k) :=
sorry

end factorable_quadratic_l290_290251


namespace power_comparison_l290_290365

noncomputable
def compare_powers : Prop := 
  1.5^(1 / 3.1) < 2^(1 / 3.1) ∧ 2^(1 / 3.1) < 2^(3.1)

theorem power_comparison : compare_powers :=
by
  sorry

end power_comparison_l290_290365


namespace one_thirds_in_nine_thirds_l290_290284

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l290_290284


namespace men_wages_l290_290502

theorem men_wages (W : ℕ) (wage : ℕ) :
  (5 + W + 8) * wage = 75 ∧ 5 * wage = W * wage ∧ W * wage = 8 * wage → 
  wage = 5 := 
by
  sorry

end men_wages_l290_290502


namespace correct_option_l290_290067

theorem correct_option (a b : ℝ) : (ab) ^ 2 = a ^ 2 * b ^ 2 :=
by sorry

end correct_option_l290_290067


namespace cost_to_fill_pool_l290_290781

-- Definitions based on conditions

def hours_to_fill_pool : ℕ := 50
def hose_rate : ℕ := 100  -- hose runs at 100 gallons per hour
def water_cost_per_10_gallons : ℕ := 1 -- cost is 1 cent for 10 gallons
def cents_to_dollars (cents : ℕ) : ℕ := cents / 100 -- Conversion from cents to dollars

-- Prove the cost to fill the pool is 5 dollars
theorem cost_to_fill_pool : 
  (hours_to_fill_pool * hose_rate / 10 * water_cost_per_10_gallons) / 100 = 5 :=
by sorry

end cost_to_fill_pool_l290_290781


namespace amount_c_l290_290205

theorem amount_c (a b c d : ℝ) :
  a + c = 350 →
  b + d = 450 →
  a + d = 400 →
  c + d = 500 →
  a + b + c + d = 750 →
  c = 225 :=
by 
  intros h1 h2 h3 h4 h5
  -- Proof omitted.
  sorry

end amount_c_l290_290205


namespace total_flowers_l290_290971

theorem total_flowers (pots: ℕ) (flowers_per_pot: ℕ) (h_pots: pots = 2150) (h_flowers_per_pot: flowers_per_pot = 128) :
    pots * flowers_per_pot = 275200 :=
by 
    sorry

end total_flowers_l290_290971


namespace davi_minimum_spending_l290_290674

-- Define the cost of a single bottle
def singleBottleCost : ℝ := 2.80

-- Define the cost of a box of six bottles
def boxCost : ℝ := 15.00

-- Define the number of bottles Davi needs to buy
def totalBottles : ℕ := 22

-- Calculate the minimum amount Davi will spend
def minimumCost : ℝ := 45.00 + 11.20 

-- The theorem to prove
theorem davi_minimum_spending :
  ∃ minCost : ℝ, minCost = 56.20 ∧ minCost = 3 * boxCost + 4 * singleBottleCost := 
by
  use 56.20
  sorry

end davi_minimum_spending_l290_290674


namespace find_y_coordinate_of_P_l290_290324

-- Define the conditions as Lean definitions
def distance_x_axis_to_P (P : ℝ × ℝ) :=
  abs P.2

def distance_y_axis_to_P (P : ℝ × ℝ) :=
  abs P.1

-- Lean statement of the problem
theorem find_y_coordinate_of_P (P : ℝ × ℝ)
  (h1 : distance_x_axis_to_P P = (1/2) * distance_y_axis_to_P P)
  (h2 : distance_y_axis_to_P P = 10) :
  P.2 = 5 ∨ P.2 = -5 :=
sorry

end find_y_coordinate_of_P_l290_290324


namespace cost_per_meter_of_fencing_l290_290350

/-- A rectangular farm has area 1200 m², a short side of 30 m, and total job cost 1560 Rs.
    Prove that the cost of fencing per meter is 13 Rs. -/
theorem cost_per_meter_of_fencing
  (A : ℝ := 1200)
  (W : ℝ := 30)
  (job_cost : ℝ := 1560)
  (L : ℝ := A / W)
  (D : ℝ := Real.sqrt (L^2 + W^2))
  (total_length : ℝ := L + W + D) :
  job_cost / total_length = 13 := 
sorry

end cost_per_meter_of_fencing_l290_290350


namespace quadratic_inequality_l290_290261

noncomputable def quadratic_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality (a b c : ℝ) 
  (h₁ : quadratic_function a b c 1 = quadratic_function a b c 3) 
  (h₂ : quadratic_function a b c 1 > quadratic_function a b c 4) : 
  a < 0 ∧ 4 * a + b = 0 :=
by
  sorry

end quadratic_inequality_l290_290261


namespace min_abs_diff_x1_x2_l290_290295

theorem min_abs_diff_x1_x2 (x1 x2 : ℝ) (f : ℝ → ℝ) (Hf : ∀ x, f x = Real.sin (π * x))
  (Hbounds : ∀ x, f x1 ≤ f x ∧ f x ≤ f x2) : |x1 - x2| = 1 := 
by
  sorry

end min_abs_diff_x1_x2_l290_290295


namespace point_on_or_outside_circle_l290_290544

theorem point_on_or_outside_circle (a : ℝ) : 
  let P := (a, 2 - a)
  let r := 2
  let center := (0, 0)
  let distance_square := (P.1 - center.1)^2 + (P.2 - center.2)^2
  distance_square >= r := 
by
  sorry

end point_on_or_outside_circle_l290_290544


namespace total_marbles_l290_290461

/--
Some marbles in a bag are red and the rest are blue.
If one red marble is removed, then one-seventh of the remaining marbles are red.
If two blue marbles are removed instead of one red, then one-fifth of the remaining marbles are red.
Prove that the total number of marbles in the bag originally is 22.
-/
theorem total_marbles (r b : ℕ) (h1 : (r - 1) / (r + b - 1) = 1 / 7) (h2 : r / (r + b - 2) = 1 / 5) :
  r + b = 22 := by
  sorry

end total_marbles_l290_290461


namespace probability_zeros_not_adjacent_is_0_6_l290_290575

-- Define the total number of arrangements of 5 elements where we have 3 ones and 2 zeros
def total_arrangements : Nat := 5.choose 2

-- Define the number of arrangements where 2 zeros are adjacent
def adjacent_zeros_arrangements : Nat := 4.choose 1 * 2

-- Define the probability that the 2 zeros are not adjacent
def probability_not_adjacent : Rat := (total_arrangements - adjacent_zeros_arrangements) / total_arrangements

-- Prove the desired probability is 0.6
theorem probability_zeros_not_adjacent_is_0_6 : probability_not_adjacent = 3 / 5 := by
  sorry

end probability_zeros_not_adjacent_is_0_6_l290_290575


namespace popsicle_sticks_ratio_l290_290033

/-- Sam, Sid, and Steve brought popsicle sticks for their group activity in their Art class. Sid has twice as many popsicle sticks as Steve. If Steve has 12 popsicle sticks and they can use 108 popsicle sticks for their Art class activity, prove that the ratio of the number of popsicle sticks Sam has to the number Sid has is 3:1. -/
theorem popsicle_sticks_ratio (Sid Sam Steve : ℕ) 
    (h1 : Sid = 2 * Steve) 
    (h2 : Steve = 12) 
    (h3 : Sam + Sid + Steve = 108) : 
    Sam / Sid = 3 :=
by 
    -- Proof steps go here
    sorry

end popsicle_sticks_ratio_l290_290033


namespace sunzi_wood_problem_l290_290436

theorem sunzi_wood_problem (x y : ℝ) (h1 : x - y = 4.5) (h2 : (1/2) * x + 1 = y) :
  (x - y = 4.5) ∧ ((1/2) * x + 1 = y) :=
by {
  exact ⟨h1, h2⟩
}

end sunzi_wood_problem_l290_290436


namespace probability_not_all_same_l290_290054

/-- What is the probability that when we roll five fair 6-sided dice, they won't all show the same number? -/
theorem probability_not_all_same :
  let total_outcomes := 6^5 in
  let same_number_outcomes := 6 in
  let probability_all_same := same_number_outcomes / total_outcomes.to_real in
  1 - probability_all_same = (1295 : ℝ) / 1296 :=
by
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  have probability_all_same := (same_number_outcomes : ℝ) / total_outcomes.to_real
  show 1 - probability_all_same = (1295 : ℝ) / 1296       
  sorry

end probability_not_all_same_l290_290054


namespace mean_is_12_point_8_l290_290769

variable (m : ℝ)
variable median_condition : m + 7 = 12

theorem mean_is_12_point_8 (m : ℝ) (median_condition : m + 7 = 12) : 
(mean := (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5) = 64 / 5 :=
by {
  sorry
}

end mean_is_12_point_8_l290_290769


namespace polygon_vertices_l290_290411

-- Define the number of diagonals from one vertex
def diagonals_from_one_vertex (n : ℕ) := n - 3

-- The main theorem stating the number of vertices is 9 given 6 diagonals from one vertex
theorem polygon_vertices (D : ℕ) (n : ℕ) (h : D = 6) (h_diagonals : diagonals_from_one_vertex n = D) :
  n = 9 := by
  sorry

end polygon_vertices_l290_290411


namespace solution_to_inequality_system_l290_290615

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end solution_to_inequality_system_l290_290615


namespace number_of_pounds_colombian_beans_l290_290876

def cost_per_pound_colombian : ℝ := 5.50
def cost_per_pound_peruvian : ℝ := 4.25
def total_weight : ℝ := 40
def desired_cost_per_pound : ℝ := 4.60
noncomputable def amount_colombian_beans (C : ℝ) : Prop := 
  let P := total_weight - C
  cost_per_pound_colombian * C + cost_per_pound_peruvian * P = desired_cost_per_pound * total_weight

theorem number_of_pounds_colombian_beans : ∃ C, amount_colombian_beans C ∧ C = 11.2 :=
sorry

end number_of_pounds_colombian_beans_l290_290876


namespace number_of_valid_six_digit_house_numbers_l290_290882

-- Define the set of two-digit primes less than 60
def two_digit_primes : List ℕ := [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59]

-- Define a predicate checking if a number is a two-digit prime less than 60
def is_valid_prime (n : ℕ) : Prop :=
  n ∈ two_digit_primes

-- Define the function to count distinct valid primes forming ABCDEF
def count_valid_house_numbers : ℕ :=
  let primes_count := two_digit_primes.length
  primes_count * (primes_count - 1) * (primes_count - 2)

-- State the main theorem
theorem number_of_valid_six_digit_house_numbers : count_valid_house_numbers = 1716 := by
  -- Showing the count of valid house numbers forms 1716
  sorry

end number_of_valid_six_digit_house_numbers_l290_290882


namespace chef_cherries_l290_290994

theorem chef_cherries :
  ∀ (total_cherries used_cherries remaining_cherries : ℕ),
    total_cherries = 77 →
    used_cherries = 60 →
    remaining_cherries = total_cherries - used_cherries →
    remaining_cherries = 17 :=
by
  sorry

end chef_cherries_l290_290994


namespace remainder_mod_of_a_squared_subtract_3b_l290_290695

theorem remainder_mod_of_a_squared_subtract_3b (a b : ℕ) (h₁ : a % 7 = 2) (h₂ : b % 7 = 5) (h₃ : a^2 > 3 * b) : 
  (a^2 - 3 * b) % 7 = 3 := 
sorry

end remainder_mod_of_a_squared_subtract_3b_l290_290695


namespace quadratic_other_root_l290_290552

theorem quadratic_other_root (m x2 : ℝ) (h₁ : 1^2 - 4*1 + m = 0) (h₂ : x2^2 - 4*x2 + m = 0) : x2 = 3 :=
sorry

end quadratic_other_root_l290_290552


namespace math_problem_l290_290419

noncomputable def log_base (base x : ℝ) : ℝ := Real.log x / Real.log base

theorem math_problem (a b c : ℝ) (h1 : ∃ k : ℤ, log_base c b = k)
  (h2 : log_base a (1 / b) > log_base a (Real.sqrt b) ∧ log_base a (Real.sqrt b) > log_base b (a^2)) :
  (∃ n : ℕ, n = 1 ∧ 
    ((1 / b > Real.sqrt b ∧ Real.sqrt b > a^2) ∨ 
    (Real.log b + log_base a a = 0) ∨ 
    (0 < a ∧ a < b ∧ b < 1) ∨ 
    (a * b = 1))) :=
by sorry

end math_problem_l290_290419


namespace find_time_same_height_l290_290343

noncomputable def height_ball (t : ℝ) : ℝ := 60 - 9 * t - 8 * t^2
noncomputable def height_bird (t : ℝ) : ℝ := 3 * t^2 + 4 * t

theorem find_time_same_height : ∃ t : ℝ, t = 20 / 11 ∧ height_ball t = height_bird t := 
by
  use 20 / 11
  sorry

end find_time_same_height_l290_290343


namespace find_k_l290_290849

theorem find_k (k t : ℝ) (h1 : t = 5) (h2 : (1/2) * (t^2) / ((k-1) * (k+1)) = 10) : 
  k = 3/2 := 
  sorry

end find_k_l290_290849


namespace integer_solutions_l290_290162

theorem integer_solutions (x : ℝ) (n : ℤ)
  (h1 : ⌊x⌋ = n) :
  3 * x - 2 * n + 4 = 0 ↔
  x = -4 ∨ x = (-14:ℚ)/3 ∨ x = (-16:ℚ)/3 :=
by sorry

end integer_solutions_l290_290162


namespace isosceles_triangle_perimeter_l290_290555

theorem isosceles_triangle_perimeter (a b : ℝ) (h_iso : a = 4 ∨ b = 4) (h_iso2 : a = 8 ∨ b = 8) : 
  (a = 4 ∧ b = 8 ∧ 4 + a + b = 16 ∨ 
  a = 4 ∧ b = 8 ∧ b + a + a = 20 ∨ 
  a = 8 ∧ b = 4 ∧ a + a + b = 20) :=
by sorry

end isosceles_triangle_perimeter_l290_290555


namespace vasya_numbers_l290_290795

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290795


namespace xiaoying_final_score_l290_290070

def speech_competition_score (score_content score_expression score_demeanor : ℕ) 
                             (weight_content weight_expression weight_demeanor : ℝ) : ℝ :=
  score_content * weight_content + score_expression * weight_expression + score_demeanor * weight_demeanor

theorem xiaoying_final_score :
  speech_competition_score 86 90 80 0.5 0.4 0.1 = 87 :=
by 
  sorry

end xiaoying_final_score_l290_290070


namespace vasya_numbers_l290_290815

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290815


namespace second_number_value_l290_290329

theorem second_number_value 
  (a b c : ℚ)
  (h1 : a + b + c = 120)
  (h2 : a / b = 3 / 4)
  (h3 : b / c = 2 / 5) :
  b = 480 / 17 :=
by
  sorry

end second_number_value_l290_290329


namespace calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l290_290344

noncomputable def cost_plan1_fixed (num_suits num_ties : ℕ) : ℕ :=
  if num_ties > num_suits then 200 * num_suits + 40 * (num_ties - num_suits)
  else 200 * num_suits

noncomputable def cost_plan2_fixed (num_suits num_ties : ℕ) : ℕ :=
  (200 * num_suits + 40 * num_ties) * 9 / 10

noncomputable def cost_plan1_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  200 * num_suits + 40 * (x - num_suits)

noncomputable def cost_plan2_variable (num_suits : ℕ) (x : ℕ) : ℕ :=
  (200 * num_suits + 40 * x) * 9 / 10

theorem calculate_fixed_payment :
  cost_plan1_fixed 20 22 = 4080 ∧ cost_plan2_fixed 20 22 = 4392 :=
by sorry

theorem calculate_variable_payment (x : ℕ) (hx : x > 20) :
  cost_plan1_variable 20 x = 40 * x + 3200 ∧ cost_plan2_variable 20 x = 36 * x + 3600 :=
by sorry

theorem compare_plans_for_x_eq_30 :
  cost_plan1_variable 20 30 < cost_plan2_variable 20 30 :=
by sorry


end calculate_fixed_payment_calculate_variable_payment_compare_plans_for_x_eq_30_l290_290344


namespace eight_and_five_l290_290476

def my_and (a b : ℕ) : ℕ := (a + b) ^ 2 * (a - b)

theorem eight_and_five : my_and 8 5 = 507 := 
  by sorry

end eight_and_five_l290_290476


namespace minimize_tangent_triangle_area_l290_290753

open Real

theorem minimize_tangent_triangle_area {a b x y : ℝ} 
  (h_ellipse : x^2 / a^2 + y^2 / b^2 = 1) :
  (∃ x y : ℝ, (x = a / sqrt 2 ∨ x = -a / sqrt 2) ∧ (y = b / sqrt 2 ∨ y = -b / sqrt 2)) :=
by
  -- Proof is omitted
  sorry

end minimize_tangent_triangle_area_l290_290753


namespace total_working_days_l290_290348

variables (x a b c : ℕ)

-- Given conditions
axiom bus_morning : b + c = 6
axiom bus_afternoon : a + c = 18
axiom train_commute : a + b = 14

-- Proposition to prove
theorem total_working_days : x = a + b + c → x = 19 :=
by
  -- Placeholder for Lean's automatic proof generation
  sorry

end total_working_days_l290_290348


namespace units_digit_of_7_power_19_l290_290063

theorem units_digit_of_7_power_19 : (7^19) % 10 = 3 := by
  sorry

end units_digit_of_7_power_19_l290_290063


namespace positive_diff_of_squares_l290_290635

theorem positive_diff_of_squares (a b : ℕ) (h1 : a + b = 40) (h2 : a - b = 10) : a^2 - b^2 = 400 := by
  sorry

end positive_diff_of_squares_l290_290635


namespace number_of_pages_in_chunk_l290_290212

-- Conditions
def first_page : Nat := 213
def last_page : Nat := 312

-- Define the property we need to prove
theorem number_of_pages_in_chunk : last_page - first_page + 1 = 100 := by
  -- skipping the proof
  sorry

end number_of_pages_in_chunk_l290_290212


namespace triangle_min_sum_l290_290724

-- Let a, b, and c be the sides of the triangle opposite to angles A, B, and C respectively
variables {a b c : ℝ}

-- Given conditions:
-- 1. (a + b)^2 - c^2 = 4
-- 2. C = 60 degrees, and by cosine rule, we have cos C = (a^2 + b^2 - c^2) / (2ab)
-- Since C = 60 degrees, cos C = 1/2
-- Therefore, (a^2 + b^2 - c^2) / (2ab) = 1/2

theorem triangle_min_sum (h1 : (a + b) ^ 2 - c ^ 2 = 4)
    (h2 : cos (real.pi / 3) = (a ^ 2 + b ^ 2 - c ^ 2) / (2 * a * b)) :
  a + b ≥ 2 * real.sqrt (4 / 3) :=
  by
    sorry

end triangle_min_sum_l290_290724


namespace originally_planned_days_l290_290506

def man_days (men : ℕ) (days : ℕ) : ℕ := men * days

theorem originally_planned_days (D : ℕ) (h : man_days 5 10 = man_days 10 D) : D = 5 :=
by 
  sorry

end originally_planned_days_l290_290506


namespace lean_l290_290692

open ProbabilityTheory

variable {Ω : Type*} [Fintype Ω] [DecidableEq Ω] (s : Finset Ω)

def red_balls := 5
def blue_balls := 4
def total_balls := red_balls + blue_balls

-- Events
def A (i : Fin 2) : Event Ω := {ω | is_red ω i}
def B (j : Fin 2) : Event Ω := {ω | is_blue ω j}

noncomputable def P (ev : Event Ω) : ℚ := (ev.card : ℚ) / (Fintype.card Ω)

theorem lean statement:
  (P(A 2) = 5 / 9) ∧ (P(A 2) + P(B 2) = 1) ∧ (P(A 2 | A 1) + P(B 2 | A 1) = 1)  := by
  sorry

end lean_l290_290692


namespace trivia_team_students_l290_290485

theorem trivia_team_students (not_picked : ℕ) (groups : ℕ) (students_per_group : ℕ) (h_not_picked : not_picked = 9) 
(h_groups : groups = 3) (h_students_per_group : students_per_group = 9) :
    not_picked + (groups * students_per_group) = 36 := by
  sorry

end trivia_team_students_l290_290485


namespace find_divisor_l290_290075

theorem find_divisor (D : ℕ) : 
  (242 % D = 15) ∧ 
  (698 % D = 27) ∧ 
  ((242 + 698) % D = 5) → 
  D = 42 := 
by 
  sorry

end find_divisor_l290_290075


namespace minimum_area_of_triangle_l290_290914

def parabola_focus : Prop :=
  ∃ F : ℝ × ℝ, F = (1, 0)

def on_parabola (A B : ℝ × ℝ) : Prop :=
  (A.2 ^ 2 = 4 * A.1 ∧ B.2 ^ 2 = 4 * B.1) ∧ (A.2 * B.2 < 0)

def dot_product_condition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

noncomputable def area (A B : ℝ × ℝ) : ℝ :=
  1 / 2 * abs (A.1 * B.2 - B.1 * A.2)

theorem minimum_area_of_triangle
  (A B : ℝ × ℝ)
  (h_focus : parabola_focus)
  (h_on_parabola : on_parabola A B)
  (h_dot : dot_product_condition A B) :
  ∃ C : ℝ, C = 4 * Real.sqrt 2 ∧ area A B = C :=
by
  sorry

end minimum_area_of_triangle_l290_290914


namespace even_divisors_count_lt_100_l290_290402

theorem even_divisors_count_lt_100 : 
  {n : ℕ | n < 100 ∧ n ≠ 0 ∧ ∃ k : ℕ, k * k = n } = { n : ℕ | n < 100 ∧ n ≠ 0 } \ 
  { n : ℕ | ∃ k : ℕ, k * k = n ∧ k < 100 } → 
  (card {n : ℕ | n < 100 ∧ n ≠ 0 ∧ even (finset.card (divisors n))} = 90) :=
begin
  sorry
end

end even_divisors_count_lt_100_l290_290402


namespace vasya_numbers_l290_290807

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290807


namespace distance_against_current_l290_290851

theorem distance_against_current (V_b V_c : ℝ) (h1 : V_b + V_c = 2) (h2 : V_b = 1.5) : 
  (V_b - V_c) * 3 = 3 := by
  sorry

end distance_against_current_l290_290851


namespace marked_price_percentage_l290_290861

variable (L P M S : ℝ)

-- Conditions
def original_list_price := 100               -- L = 100
def purchase_price := 70                     -- P = 70
def required_profit_price := 91              -- S = 91
def final_selling_price (M : ℝ) := 0.85 * M  -- S = 0.85M

-- Question: What percentage of the original list price should the marked price be?
theorem marked_price_percentage :
  L = original_list_price →
  P = purchase_price →
  S = required_profit_price →
  final_selling_price M = S →
  M = 107.06 := sorry

end marked_price_percentage_l290_290861


namespace ellipse_and_fixed_point_l290_290131

theorem ellipse_and_fixed_point (
  A : ℝ × ℝ := (0, -2),
  B : ℝ × ℝ := (3 / 2, -1),
  P : ℝ × ℝ := (1, -2)
) : ∃ (m n : ℝ), 
  m > 0 ∧ n > 0 ∧ 
  m ≠ n ∧ 
  ∀ (x y : ℝ), (m * x^2 + n * y^2 = 1) → 
  (0, -2) ∈ 
  {Q : ℝ × ℝ | 
   ∃ (k : ℝ), 
   ∃ (M N T H : ℝ × ℝ),
   ↔ ∃ (x₁ y₁ x₂ y₂ : ℝ), 
      (M = (x₁, y₁) ∧ N = (x₂, y₂)) 
      ∧ (k * 1 - 2 - k = P.1)
      ∧ (m * x₁^2 + n * y₁^2 = 1)
      ∧ (m * x₂^2 + n * y₂^2 = 1)
      ∧ (T = (ω₁, -2) for some ω₁)
      ∧ (H = 2 • T - M)
  }
:= sorry

end ellipse_and_fixed_point_l290_290131


namespace estimate_pi_l290_290049

theorem estimate_pi (m : ℝ) (n : ℝ) (a : ℝ) (b : ℝ) (h1 : m = 56) (h2 : n = 200) (h3 : a = 1/2) (h4 : b = 1/4) :
  (m / n) = (π / 4 - 1 / 2) ↔ π = 78 / 25 :=
by
  sorry

end estimate_pi_l290_290049


namespace time_after_seconds_l290_290734

def initialTime := Time.mk 10 15 30
def elapsedSeconds := 9999
def expectedTime := Time.mk 13 2 9

theorem time_after_seconds :
  initialTime.addSeconds elapsedSeconds = expectedTime :=
by
  sorry

end time_after_seconds_l290_290734


namespace value_of_t_plus_one_over_t_l290_290905

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l290_290905


namespace prob_zeros_not_adjacent_l290_290572

theorem prob_zeros_not_adjacent :
  let total_arrangements := (5.factorial : ℝ)
  let zeros_together_arrangements := (4.factorial : ℝ)
  let prob_zeros_together := (zeros_together_arrangements / total_arrangements)
  let prob_zeros_not_adjacent := 1 - prob_zeros_together
  prob_zeros_not_adjacent = 0.6 :=
by
  sorry

end prob_zeros_not_adjacent_l290_290572


namespace find_p_l290_290268

theorem find_p (p : ℝ) : 
  (Nat.choose 5 3) * p^3 = 80 → p = 2 :=
by
  intro h
  sorry

end find_p_l290_290268


namespace angle_ABC_is_45_l290_290958

theorem angle_ABC_is_45
  (x : ℝ)
  (h1 : ∀ (ABC : ℝ), x = 180 - ABC → x = 45) :
  2 * (x / 2) = (180 - x) / 6 → x = 45 :=
by
  sorry

end angle_ABC_is_45_l290_290958


namespace smallest_sum_p_q_l290_290924

theorem smallest_sum_p_q (p q : ℕ) (h1: p > 0) (h2: q > 0) (h3 : (∃ k1 k2 : ℕ, 7 ^ (p + 4) * 5 ^ q * 2 ^ 3 = (k1 * 7 *  k2 * 5 * (2 * 3))) ^ 3) :
  p + q = 5 :=
by
  -- Proof goes here
  sorry

end smallest_sum_p_q_l290_290924


namespace different_people_count_l290_290187

def initial_people := 9
def people_left := 6
def people_joined := 3
def total_different_people (initial_people people_left people_joined : ℕ) : ℕ :=
  initial_people + people_joined

theorem different_people_count :
  total_different_people initial_people people_left people_joined = 12 :=
by
  sorry

end different_people_count_l290_290187


namespace abc_inequality_l290_290744

theorem abc_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a * b + b * c + c * a = 1) :
  (a^2 + a * b + b^2) * (b^2 + b * c + c^2) * (c^2 + c * a + a^2) ≥ (a * b + b * c + c * a)^2 :=
sorry

end abc_inequality_l290_290744


namespace vasya_numbers_l290_290802

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290802


namespace complex_number_problem_l290_290134

open Complex -- Open the complex numbers namespace

theorem complex_number_problem 
  (z1 z2 : ℂ) 
  (h_z1 : z1 = 2 - I) 
  (h_z2 : z2 = -I) : 
  z1 / z2 + Complex.abs z2 = 2 + 2 * I := by
-- Definitions and conditions directly from (a)
  rw [h_z1, h_z2] -- Replace z1 and z2 with their given values
  sorry -- Proof to be filled in place of the solution steps

end complex_number_problem_l290_290134


namespace find_balls_l290_290230

theorem find_balls (x y : ℕ) (h1 : (x + y : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) - 1 / 15)
                   (h2 : (y + 18 : ℚ) / (x + y + 18) = (x + 18) / (x + y + 18) * 11 / 10) :
  x = 12 ∧ y = 15 :=
sorry

end find_balls_l290_290230


namespace trapezoid_area_l290_290193

theorem trapezoid_area (x : ℝ) (y : ℝ) :
  (∀ x, y = x + 1) →
  (∀ y, y = 12) →
  (∀ y, y = 7) →
  (∀ x, x = 0) →
  ∃ area,
  area = (1/2) * (6 + 11) * 5 ∧ area = 42.5 :=
by {
  sorry
}

end trapezoid_area_l290_290193


namespace no_attention_prob_l290_290224

noncomputable def prob_no_attention (p1 p2 p3 : ℝ) : ℝ :=
  (1 - p1) * (1 - p2) * (1 - p3)

theorem no_attention_prob :
  let p1 := 0.9
  let p2 := 0.8
  let p3 := 0.6
  prob_no_attention p1 p2 p3 = 0.008 :=
by
  unfold prob_no_attention
  sorry

end no_attention_prob_l290_290224


namespace applicant_overall_score_is_72_l290_290853

-- Define conditions as variables
variables (written_score : ℕ) (interview_score : ℕ) (written_weight : ℝ) (interview_weight : ℝ)

-- Define correct answer for the overall score calculation
def overall_score (written_score interview_score : ℕ) (written_weight interview_weight : ℝ) : ℝ :=
  written_score * written_weight + interview_score * interview_weight

-- We state the main theorem to prove the overall score is 72 given the conditions
theorem applicant_overall_score_is_72 :
  written_score = 80 → interview_score = 60 → written_weight = 0.6 → interview_weight = 0.4 →
  overall_score written_score interview_score written_weight interview_weight = 72 :=
by
  intros h_written_score h_interview_score h_written_weight h_interview_weight
  rw [h_written_score, h_interview_score, h_written_weight, h_interview_weight]
  norm_num
  rw [nat.cast_mul]
  sorry

end applicant_overall_score_is_72_l290_290853


namespace solution_is_permutations_l290_290460

noncomputable def solve_system (x y z : ℤ) : Prop :=
  x^2 = y * z + 1 ∧ y^2 = z * x + 1 ∧ z^2 = x * y + 1

theorem solution_is_permutations (x y z : ℤ) :
  solve_system x y z ↔ (x, y, z) = (1, 0, -1) ∨ (x, y, z) = (1, -1, 0) ∨ (x, y, z) = (0, 1, -1) ∨ (x, y, z) = (0, -1, 1) ∨ (x, y, z) = (-1, 1, 0) ∨ (x, y, z) = (-1, 0, 1) :=
by sorry

end solution_is_permutations_l290_290460


namespace min_value_l290_290942

-- Defining the conditions
variables {x y z : ℝ}

-- Problem statement translating the conditions
theorem min_value (h₁ : x > 0) (h₂ : y > 0) (h₃ : z > 0) (h₄ : x + y + z = 5) : 
  ∃ (minval : ℝ), minval = 36/5 ∧ ∀ w, w = (1/x + 4/y + 9/z) → w ≥ minval :=
by
  sorry

end min_value_l290_290942


namespace sum_of_reciprocals_l290_290776

variable (x y : ℝ)

theorem sum_of_reciprocals (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  (1 / x) + (1 / y) = 3 := 
sorry

end sum_of_reciprocals_l290_290776


namespace hexagon_largest_angle_measure_l290_290040

theorem hexagon_largest_angle_measure (x : ℝ) (a b c d e f : ℝ)
  (h_ratio: a = 2 * x) (h_ratio2: b = 3 * x)
  (h_ratio3: c = 3 * x) (h_ratio4: d = 4 * x)
  (h_ratio5: e = 4 * x) (h_ratio6: f = 6 * x)
  (h_sum: a + b + c + d + e + f = 720) :
  f = 2160 / 11 :=
by
  -- Proof is not required
  sorry

end hexagon_largest_angle_measure_l290_290040


namespace students_neither_cool_l290_290484

variable (total_students : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool : ℕ)

def only_cool_dads := cool_dads - both_cool
def only_cool_moms := cool_moms - both_cool
def only_cool := only_cool_dads + only_cool_moms + both_cool
def neither_cool := total_students - only_cool

theorem students_neither_cool (h1 : total_students = 40) (h2 : cool_dads = 18) (h3 : cool_moms = 22) (h4 : both_cool = 10) 
: neither_cool total_students cool_dads cool_moms both_cool = 10 :=
by 
  sorry

end students_neither_cool_l290_290484


namespace base_b_number_not_divisible_by_5_l290_290900

-- We state the mathematical problem in Lean 4 as a theorem.
theorem base_b_number_not_divisible_by_5 (b : ℕ) (hb : b = 12) : 
  ¬ ((3 * b^2 * (b - 1) + 1) % 5 = 0) := 
by sorry

end base_b_number_not_divisible_by_5_l290_290900


namespace percentage_received_certificates_l290_290729

theorem percentage_received_certificates (boys girls : ℕ) (pct_boys pct_girls : ℝ) :
    boys = 30 ∧ girls = 20 ∧ pct_boys = 0.1 ∧ pct_girls = 0.2 →
    ((pct_boys * boys + pct_girls * girls) / (boys + girls) * 100) = 14 := by
  sorry

end percentage_received_certificates_l290_290729


namespace cost_to_fill_pool_l290_290783

-- Define the given conditions as constants
def filling_time : ℝ := 50
def flow_rate : ℝ := 100
def cost_per_10_gallons : ℝ := 0.01

-- Calculate total volume in gallons
def total_volume : ℝ := filling_time * flow_rate

-- Calculate the cost per gallon in dollars
def cost_per_gallon : ℝ := cost_per_10_gallons / 10

-- Define the total cost to fill the pool in dollars
def total_cost : ℝ := total_volume * cost_per_gallon

-- Prove that the total cost equals $5
theorem cost_to_fill_pool : total_cost = 5 := by
  unfold total_cost
  unfold total_volume
  unfold cost_per_gallon
  unfold filling_time
  unfold flow_rate
  unfold cost_per_10_gallons
  sorry

end cost_to_fill_pool_l290_290783


namespace min_value_f_range_of_a_l290_290025
open Real

noncomputable def f (x : ℝ) : ℝ := (x + 1) * log (x + 1)
noncomputable def g (a x : ℝ) : ℝ := a * x^2 + x

theorem min_value_f : infimum (range f) = -1 / exp 1 := 
sorry

theorem range_of_a (a : ℝ) : (∀ x ≥ 0, f x ≤ g a x) ↔ a ≥ 1 / 2 := 
sorry

end min_value_f_range_of_a_l290_290025


namespace simplify_fraction_to_9_l290_290759

-- Define the necessary terms and expressions
def problem_expr := (3^12)^2 - (3^10)^2
def problem_denom := (3^11)^2 - (3^9)^2
def simplified_expr := problem_expr / problem_denom

-- State the theorem we want to prove
theorem simplify_fraction_to_9 : simplified_expr = 9 := 
by sorry

end simplify_fraction_to_9_l290_290759


namespace quadratic_intersects_x_axis_l290_290720

theorem quadratic_intersects_x_axis (a : ℝ) (h : a ≠ 0) :
  ∃ x : ℝ, a * x^2 - (3 * a + 1) * x + 3 = 0 := 
by {
  -- The proof will go here
  sorry
}

end quadratic_intersects_x_axis_l290_290720


namespace even_number_of_divisors_l290_290398

-- Proof statement: There are 90 positive integers less than 100 with an even number of divisors.
theorem even_number_of_divisors : 
  {n : ℕ | n < 100 ∧ ∃ k : ℕ, k ^ 2 = n}.toFinset.card = 90 := 
sorry

end even_number_of_divisors_l290_290398


namespace min_xy_l290_290001

theorem min_xy (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + 8 * y - x * y = 0) : xy ≥ 64 :=
by sorry

end min_xy_l290_290001


namespace number_of_ways_to_place_letters_l290_290756

open Finset

-- Define a set of positions
def positions : Finset (ℕ × ℕ) := Finset.univ.product Finset.univ

-- Define the condition that no letter appears more than once per row and column
def valid_placement (p1 p2 p3 p4 : (ℕ × ℕ)) : Prop :=
  p1.1 ≠ p2.1 ∧ p1.2 ≠ p2.2 ∧ 
  p1.1 ≠ p3.1 ∧ p1.2 ≠ p3.2 ∧ 
  p1.1 ≠ p4.1 ∧ p1.2 ≠ p4.2 ∧ 
  p2.1 ≠ p3.1 ∧ p2.2 ≠ p3.2 ∧ 
  p2.1 ≠ p4.1 ∧ p2.2 ≠ p4.2 ∧ 
  p3.1 ≠ p4.1 ∧ p3.2 ≠ p4.2

theorem number_of_ways_to_place_letters : 
  ∃ (ls : list (ℕ × ℕ)),
  (ls.length = 4) ∧ 
  (∀ p1 p2 p3 p4 ∈ ls, valid_placement p1 p2 p3 p4) ∧
  ls.nodup ∧
  3960 = 16.choose 2 * 4 * 14.choose 2 * 4 := 
sorry

end number_of_ways_to_place_letters_l290_290756


namespace mean_of_set_is_12_point_8_l290_290766

theorem mean_of_set_is_12_point_8 (m : ℝ) 
    (h1 : (m + 7) = 12) : (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := 
by
  sorry

end mean_of_set_is_12_point_8_l290_290766


namespace acute_triangle_probability_correct_l290_290887

noncomputable def acute_triangle_probability : ℝ :=
  let l_cube_vol := 1
  let quarter_cone_vol := (1/4) * (1/3) * Real.pi * (1^2) * 1
  let total_unfavorable_vol := 3 * quarter_cone_vol
  let favorable_vol := l_cube_vol - total_unfavorable_vol
  favorable_vol / l_cube_vol

theorem acute_triangle_probability_correct : abs (acute_triangle_probability - 0.2146) < 0.0001 :=
  sorry

end acute_triangle_probability_correct_l290_290887


namespace johns_final_push_time_l290_290990

theorem johns_final_push_time :
  ∃ t : ℝ, t = 17 / 4.2 := 
by
  sorry

end johns_final_push_time_l290_290990


namespace possible_area_l290_290431

theorem possible_area (A : ℝ) (B : ℝ) (L : ℝ × ℝ) (H₁ : L.1 = 13) (H₂ : L.2 = 14) (area_needed : ℝ) (H₃ : area_needed = 200) : 
∃ x y : ℝ, x = 13 ∧ y = 16 ∧ x * y ≥ area_needed :=
by
  sorry

end possible_area_l290_290431


namespace tina_total_leftover_l290_290780

def monthly_income : ℝ := 1000

def june_savings : ℝ := 0.25 * monthly_income
def june_expenses : ℝ := 200 + 0.05 * monthly_income
def june_leftover : ℝ := monthly_income - june_savings - june_expenses

def july_savings : ℝ := 0.20 * monthly_income
def july_expenses : ℝ := 250 + 0.15 * monthly_income
def july_leftover : ℝ := monthly_income - july_savings - july_expenses

def august_savings : ℝ := 0.30 * monthly_income
def august_expenses : ℝ := 250 + 50 + 0.10 * monthly_income
def august_gift : ℝ := 50
def august_leftover : ℝ := (monthly_income - august_savings - august_expenses) + august_gift

def total_leftover : ℝ :=
  june_leftover + july_leftover + august_leftover

theorem tina_total_leftover (I : ℝ) (hI : I = 1000) :
  total_leftover = 1250 := by
  rw [←hI] at *
  show total_leftover = 1250
  sorry

end tina_total_leftover_l290_290780


namespace range_of_m_l290_290380

theorem range_of_m (x y : ℝ) (m : ℝ) 
  (hx : x > 0) 
  (hy : y > 0) 
  (hineq : ∀ x > 0, ∀ y > 0, 2 * y / x + 8 * x / y ≥ m^2 + 2 * m) : 
  -4 ≤ m ∧ m ≤ 2 :=
sorry

end range_of_m_l290_290380


namespace tan_double_angle_l290_290258

variable {α β : ℝ}

theorem tan_double_angle (h1 : Real.tan (α + β) = 3) (h2 : Real.tan (α - β) = 2) : Real.tan (2 * α) = -1 := by
  sorry

end tan_double_angle_l290_290258


namespace ellipse_foci_on_x_axis_l290_290418

variable {a b : ℝ}

theorem ellipse_foci_on_x_axis (h : ∀ x y : ℝ, a * x^2 + b * y^2 = 1) (hc : ∀ x y : ℝ, (a * x^2 + b * y^2 = 1) → (1 / a > 1 / b ∧ 1 / b > 0))
  : 0 < a ∧ a < b :=
sorry

end ellipse_foci_on_x_axis_l290_290418


namespace vasya_numbers_l290_290816

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290816


namespace uncle_money_given_l290_290452

-- Definitions
def lizzy_mother_money : Int := 80
def lizzy_father_money : Int := 40
def candy_expense : Int := 50
def total_money_now : Int := 140

-- Theorem to prove
theorem uncle_money_given : (total_money_now - ((lizzy_mother_money + lizzy_father_money) - candy_expense)) = 70 := 
  by
    sorry

end uncle_money_given_l290_290452


namespace two_legged_birds_count_l290_290434

-- Definitions and conditions
variables {x y z : ℕ}
variables (heads_eq : x + y + z = 200) (legs_eq : 2 * x + 3 * y + 4 * z = 558)

-- The statement to prove
theorem two_legged_birds_count : x = 94 :=
sorry

end two_legged_birds_count_l290_290434


namespace num_pens_multiple_of_16_l290_290765

theorem num_pens_multiple_of_16 (Pencils Students : ℕ) (h1 : Pencils = 928) (h2 : Students = 16)
  (h3 : ∃ (Pn : ℕ), Pencils = Pn * Students) :
  ∃ (k : ℕ), ∃ (Pens : ℕ), Pens = 16 * k :=
by
  sorry

end num_pens_multiple_of_16_l290_290765


namespace seventieth_even_integer_l290_290837

theorem seventieth_even_integer : 2 * 70 = 140 :=
by
  sorry

end seventieth_even_integer_l290_290837


namespace binary_ternary_conversion_l290_290927

theorem binary_ternary_conversion (a b : ℕ) (h_b : b = 0 ∨ b = 1) (h_a : a = 0 ∨ a = 1 ∨ a = 2)
  (h_eq : 8 + 2 * b + 1 = 9 * a + 2) : 2 * a + b = 3 :=
by
  sorry

end binary_ternary_conversion_l290_290927


namespace wood_rope_length_equivalence_l290_290441

variable (x y : ℝ)

theorem wood_rope_length_equivalence :
  (x - y = 4.5) ∧ (y = (1 / 2) * x + 1) :=
  sorry

end wood_rope_length_equivalence_l290_290441


namespace no_such_integers_x_y_l290_290458

theorem no_such_integers_x_y (x y : ℤ) : x^2 + 1974 ≠ y^2 := by
  sorry

end no_such_integers_x_y_l290_290458


namespace smaller_cube_surface_area_l290_290860

theorem smaller_cube_surface_area (edge_length : ℝ) (h : edge_length = 12) :
  let sphere_diameter := edge_length
  let smaller_cube_side := sphere_diameter / Real.sqrt 3
  let surface_area := 6 * smaller_cube_side ^ 2
  surface_area = 288 := by
  sorry

end smaller_cube_surface_area_l290_290860


namespace angie_pretzels_l290_290665

theorem angie_pretzels (Barry_Shelly: ℕ) (Shelly_Angie: ℕ) :
  (Barry_Shelly = 12 / 2) → (Shelly_Angie = 3 * Barry_Shelly) → (Barry_Shelly = 6) → (Shelly_Angie = 18) :=
by
  intro h1 h2 h3
  sorry

end angie_pretzels_l290_290665


namespace third_side_length_is_six_l290_290181

theorem third_side_length_is_six
  (a b : ℝ) (c : ℤ)
  (h1 : a = 6.31) 
  (h2 : b = 0.82) 
  (h3 : (a + b > c) ∧ ((b : ℝ) + (c : ℝ) > a) ∧ (c + a > b)) 
  (h4 : 5.49 < (c : ℝ)) 
  (h5 : (c : ℝ) < 7.13) : 
  c = 6 :=
by
  -- Proof goes here
  sorry

end third_side_length_is_six_l290_290181


namespace sum_of_coefficients_eq_two_l290_290115

theorem sum_of_coefficients_eq_two {a b c : ℤ} (h : ∀ x : ℤ, x * (x + 1) = a + b * x + c * x^2) : a + b + c = 2 := 
by
  sorry

end sum_of_coefficients_eq_two_l290_290115


namespace union_A_B_equiv_l290_290384

def A : Set ℝ := {x : ℝ | x > 2}
def B : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 3}

theorem union_A_B_equiv : A ∪ B = {x : ℝ | x ≥ 1} :=
by
  sorry

end union_A_B_equiv_l290_290384


namespace vasya_numbers_l290_290826

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290826


namespace surface_is_plane_l290_290896

-- Define cylindrical coordinates
structure CylindricalCoordinate where
  r : ℝ
  θ : ℝ
  z : ℝ

-- Define the property for a constant θ
def isConstantTheta (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  coord.θ = c

-- Define the plane in cylindrical coordinates
def isPlane (c : ℝ) (coord : CylindricalCoordinate) : Prop :=
  isConstantTheta c coord

-- Theorem: The surface described by θ = c in cylindrical coordinates is a plane.
theorem surface_is_plane (c : ℝ) (coord : CylindricalCoordinate) :
    isPlane c coord ↔ isConstantTheta c coord := sorry

end surface_is_plane_l290_290896


namespace parabola_vertex_l290_290325

theorem parabola_vertex (x y : ℝ) : ∀ x y, (y^2 + 8 * y + 2 * x + 11 = 0) → (x = 5 / 2 ∧ y = -4) :=
by
  intro x y h
  sorry

end parabola_vertex_l290_290325


namespace min_sum_first_n_terms_l290_290138

theorem min_sum_first_n_terms {a : ℕ → ℤ} (h₁ : ∀ n, a n = 2 * n - 48) : 
  (∃ n, (n = 23 ∨ n = 24) ∧ 
         ∀ m, sum_first_n a n ≤ sum_first_n a m) :=
by
  sorry

end min_sum_first_n_terms_l290_290138


namespace correct_eqns_l290_290438

theorem correct_eqns (x y : ℝ) (h1 : x - y = 4.5) (h2 : 1/2 * x + 1 = y) :
  x - y = 4.5 ∧ 1/2 * x + 1 = y :=
by {
  exact ⟨h1, h2⟩,
}

end correct_eqns_l290_290438


namespace binom_20_4_plus_10_l290_290533

open Nat

noncomputable def binom (n k : ℕ) : ℕ := factorial n / (factorial k * factorial (n - k))

theorem binom_20_4_plus_10 :
  binom 20 4 + 10 = 4855 := by
  sorry

end binom_20_4_plus_10_l290_290533


namespace problem_l290_290130

noncomputable def ellipse_equation : Prop :=
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧ 
  (m * (0^2) + n * (-2)^2 = 1) ∧ 
  (m * (3/2)^2 + n * (-1)^2 = 1) ∧ 
  (∀ x y : ℝ, (m * x^2 + n * y^2 = 1) ↔ (x^2/3 + y^2/4 = 1))

noncomputable def line_HN_through_fixed_point (P M N H : ℝ × ℝ) : Prop :=
  ∀ k : ℝ,
    (P = (1, -2)) →
    (M.x * N.y + N.x * M.y = k) →
    (H = (3 * M.y + 6 - M.x, M.y)) →
    M ≠ N →
    collinear {(H.y + 2), (N.y + 2), -2}

theorem problem (P M N : ℝ × ℝ) :
  ellipse_equation ∧ line_HN_through_fixed_point P M N := 
sorry

end problem_l290_290130


namespace how_many_more_yellow_peaches_l290_290151

-- Definitions
def red_peaches : ℕ := 7
def yellow_peaches_initial : ℕ := 15
def green_peaches : ℕ := 8
def combined_red_green_peaches := red_peaches + green_peaches
def required_yellow_peaches := 2 * combined_red_green_peaches
def additional_yellow_peaches_needed := required_yellow_peaches - yellow_peaches_initial

-- Theorem statement
theorem how_many_more_yellow_peaches :
  additional_yellow_peaches_needed = 15 :=
by
  sorry

end how_many_more_yellow_peaches_l290_290151


namespace total_parents_in_auditorium_l290_290186

-- Define the conditions.
def girls : Nat := 6
def boys : Nat := 8
def total_kids : Nat := girls + boys
def parents_per_kid : Nat := 2
def total_parents : Nat := total_kids * parents_per_kid

-- The statement to prove.
theorem total_parents_in_auditorium : total_parents = 28 := by
  sorry

end total_parents_in_auditorium_l290_290186


namespace vasya_numbers_l290_290828

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290828


namespace probability_yellow_face_l290_290619

-- Define the total number of faces and the number of yellow faces on the die
def total_faces := 12
def yellow_faces := 4

-- Define the probability calculation
def probability_of_yellow := yellow_faces / total_faces

-- State the theorem
theorem probability_yellow_face : probability_of_yellow = 1 / 3 := by
  sorry

end probability_yellow_face_l290_290619


namespace total_ants_correct_l290_290226

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end total_ants_correct_l290_290226


namespace part1_monotonically_increasing_part2_positive_definite_l290_290716

-- Definition of the function
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + 2 * k * x + 4

-- Part 1: Proving the range of k for monotonically increasing function on [1, 4]
theorem part1_monotonically_increasing (k : ℝ) :
  (∀ x ∈ Set.Icc 1 4, ∀ y ∈ Set.Icc 1 4, x ≤ y → f k x ≤ f k y) ↔ k ≥ -1 :=
sorry

-- Part 2: Proving the range of k for f(x) > 0 for all x
theorem part2_positive_definite (k : ℝ) :
  (∀ x : ℝ, f k x > 0) ↔ k ∈ Set.Ioo (-2) 2 :=
sorry

end part1_monotonically_increasing_part2_positive_definite_l290_290716


namespace vasya_numbers_l290_290825

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290825


namespace vasya_numbers_l290_290813

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290813


namespace total_distance_traveled_is_correct_l290_290855

-- Definitions of given conditions
def Vm : ℕ := 8
def Vr : ℕ := 2
def round_trip_time : ℝ := 1

-- Definitions needed for intermediate calculations (speed computations)
def upstream_speed (Vm Vr : ℕ) : ℕ := Vm - Vr
def downstream_speed (Vm Vr : ℕ) : ℕ := Vm + Vr

-- The equation representing the total time for the round trip
def time_equation (D : ℝ) (Vm Vr : ℕ) : Prop :=
  D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time

-- Prove that the total distance traveled by the man is 7.5 km
theorem total_distance_traveled_is_correct : ∃ D : ℝ, D / upstream_speed Vm Vr + D / downstream_speed Vm Vr = round_trip_time ∧ 2 * D = 7.5 :=
by
  sorry

end total_distance_traveled_is_correct_l290_290855


namespace ellipse_solution_geometry_solution_l290_290129

section Ellipse

variables {m n : ℝ}

/-- Given ellipse properties and points A and B --/
def ellipse_equation (E : ℝ → ℝ → Prop) : Prop :=
  E 0 (-2) ∧ E (3/2) (-1)

theorem ellipse_solution : 
  (∀ x y, (m * x^2 + n * y^2 = 1 ↔ E x y)) →
  (m = 1/3 ∧ n = 1/4) →
  E = λ x y, (x^2 / 3 + y^2 / 4 = 1) := 
sorry
end Ellipse

section Geometry
variables {P M N T H : Point}
variables {k : ℝ}
variables {E : ℝ → ℝ → Prop} -- Equation of the ellipse
variables {f : Point → Point → Prop} -- Line function

/-- Given points P, M, N, T, H and their conditions
Prove that the line HN always passes through a fixed point (0, -2) --/
def fixed_point_proof (F : Point → Point → Prop) : Prop :=
  F (1, -2) M ∧ 
  F P M ∧ 
  F M N ∧ 
  F M T ∧ 
  F N H ∧ 
  F T H ∧ 
  (H = (0, -2))

theorem geometry_solution :
  (fixed_point_proof f) →
  (f H N ↔ f (0, -2) N) :=
sorry
end Geometry

end ellipse_solution_geometry_solution_l290_290129


namespace profit_percentage_is_correct_l290_290504

noncomputable def cost_price : ℝ := 47.50
noncomputable def selling_price : ℝ := 65.97
noncomputable def list_price := selling_price / 0.90
noncomputable def profit := selling_price - cost_price
noncomputable def profit_percentage := (profit / cost_price) * 100

theorem profit_percentage_is_correct : profit_percentage = 38.88 := by
  sorry

end profit_percentage_is_correct_l290_290504


namespace bobby_shoes_l290_290361

variable (Bonny_pairs Becky_pairs Bobby_pairs : ℕ)
variable (h1 : Bonny_pairs = 13)
variable (h2 : 2 * Becky_pairs - 5 = Bonny_pairs)
variable (h3 : Bobby_pairs = 3 * Becky_pairs)

theorem bobby_shoes : Bobby_pairs = 27 :=
by
  -- Use the conditions to prove the required theorem
  sorry

end bobby_shoes_l290_290361


namespace max_has_two_nickels_l290_290177

theorem max_has_two_nickels (n : ℕ) (nickels : ℕ) (coins_value_total : ℕ) :
  (coins_value_total = 15 * n) -> (coins_value_total + 10 = 16 * (n + 1)) -> 
  coins_value_total - nickels * 5 + nickels + 25 = 90 -> 
  n = 6 -> 
  2 = nickels := 
by 
  sorry

end max_has_two_nickels_l290_290177


namespace find_p_q_l290_290235

theorem find_p_q (D : ℝ) (p q : ℝ) (h_roots : ∀ x, x^2 + p * x + q = 0 → (x = D ∨ x = 1 - D))
  (h_discriminant : D = p^2 - 4 * q) :
  (p = -1 ∧ q = 0) ∨ (p = -1 ∧ q = 3 / 16) :=
by
  sorry

end find_p_q_l290_290235


namespace diophantine_no_nonneg_solutions_l290_290708

theorem diophantine_no_nonneg_solutions {a b : ℕ} (ha : 0 < a) (hb : 0 < b) (h_gcd : Nat.gcd a b = 1) :
  ∃ (c : ℕ), (a * b - a - b + 1) / 2 = (a - 1) * (b - 1) / 2 := 
sorry

end diophantine_no_nonneg_solutions_l290_290708


namespace regular_decagon_triangle_count_l290_290368

def regular_decagon (V : Type) := ∃ vertices : V, Fintype.card vertices = 10

theorem regular_decagon_triangle_count (V : Type) [Fintype V] (h : regular_decagon V)
: Fintype.card { triangle : Finset V // triangle.card = 3 } = 120 := by
  sorry

end regular_decagon_triangle_count_l290_290368


namespace quadratic_monotonic_range_l290_290566

theorem quadratic_monotonic_range {t : ℝ} (h : ∀ x1 x2 : ℝ, (1 < x1 ∧ x1 < 3) → (1 < x2 ∧ x2 < 3) → x1 < x2 → (x1^2 - 2 * t * x1 + 1 ≤ x2^2 - 2 * t * x2 + 1)) : 
  t ≤ 1 ∨ t ≥ 3 :=
by
  sorry

end quadratic_monotonic_range_l290_290566


namespace aluminum_iodide_mass_produced_l290_290371

theorem aluminum_iodide_mass_produced
  (mass_Al : ℝ) -- the mass of Aluminum used
  (molar_mass_Al : ℝ) -- molar mass of Aluminum
  (molar_mass_AlI3 : ℝ) -- molar mass of Aluminum Iodide
  (reaction_eq : ∀ (moles_Al : ℝ) (moles_AlI3 : ℝ), 2 * moles_Al = 2 * moles_AlI3) -- reaction equation which indicates a 1:1 molar ratio
  (mass_Al_value : mass_Al = 25.0) 
  (molar_mass_Al_value : molar_mass_Al = 26.98) 
  (molar_mass_AlI3_value : molar_mass_AlI3 = 407.68) :
  ∃ mass_AlI3 : ℝ, mass_AlI3 = 377.52 := by
  sorry

end aluminum_iodide_mass_produced_l290_290371


namespace mass_percentage_C_in_butanoic_acid_is_54_50_l290_290374

noncomputable def atomic_mass_C : ℝ := 12.01
noncomputable def atomic_mass_H : ℝ := 1.01
noncomputable def atomic_mass_O : ℝ := 16.00

noncomputable def molar_mass_butanoic_acid : ℝ :=
  (4 * atomic_mass_C) + (8 * atomic_mass_H) + (2 * atomic_mass_O)

noncomputable def mass_of_C_in_butanoic_acid : ℝ :=
  4 * atomic_mass_C

noncomputable def mass_percentage_C : ℝ :=
  (mass_of_C_in_butanoic_acid / molar_mass_butanoic_acid) * 100

theorem mass_percentage_C_in_butanoic_acid_is_54_50 :
  mass_percentage_C = 54.50 := by
  sorry

end mass_percentage_C_in_butanoic_acid_is_54_50_l290_290374


namespace time_for_machine_A_l290_290984

theorem time_for_machine_A (x : ℝ) (T : ℝ) (A B : ℝ) :
  (B = 2 * x / 5) → 
  (A + B = x / 2) → 
  (A = x / T) → 
  T = 10 := 
by 
  intros hB hAB hA
  sorry

end time_for_machine_A_l290_290984


namespace probability_greater_than_half_l290_290032

noncomputable def x : ℝ → ℝ
noncomputable def y : ℝ → ℝ

def probability_cond (x: ℝ) (y: ℝ) : set ℝ := 
  {ω | |x ω - y ω| > 1/2}

theorem probability_greater_than_half :
  ∀ x y : ℝ, P(probability_cond x y) = 3 / 8 := 
sorry

end probability_greater_than_half_l290_290032


namespace derivative_at_one_l290_290294

variable (x : ℝ)

def f (x : ℝ) := x^2 - 2*x + 3

theorem derivative_at_one : deriv f 1 = 0 := 
by 
  sorry

end derivative_at_one_l290_290294


namespace student_correct_answers_l290_290072

theorem student_correct_answers 
  (c w : ℕ) 
  (h1 : c + w = 60) 
  (h2 : 4 * c - w = 130) : 
  c = 38 :=
by
  sorry

end student_correct_answers_l290_290072


namespace part1_part2_part3_l290_290191

def determinant (a b c d : ℝ) : ℝ := a * d - b * c

theorem part1 : determinant (-3) (-2) 4 5 = -7 := by
  sorry

theorem part2 (x: ℝ) (h: determinant 2 (-2 * x) 3 (-5 * x) = 2) : x = -1/2 := by
  sorry

theorem part3 (m n x: ℝ) 
  (h1: determinant (8 * m * x - 1) (-8/3 + 2 * x) (3/2) (-3) = 
        determinant 6 (-1) (-n) x) : 
    m = -3/8 ∧ n = -7 := by
  sorry

end part1_part2_part3_l290_290191


namespace max_profit_max_profit_price_l290_290092

-- Definitions based on the conditions
def purchase_price : ℝ := 80
def initial_selling_price : ℝ := 120
def initial_sales : ℕ := 20
def extra_sales_per_unit_decrease : ℕ := 2
def cost_price_constraint (x : ℝ) : Prop := 0 < x ∧ x ≤ 40

-- Expression for the profit function
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

-- Prove the maximum profit given the conditions
theorem max_profit : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 :=
by
  sorry

-- Proving that the selling price for max profit is 105 yuan
theorem max_profit_price : ∃ x : ℝ, cost_price_constraint x ∧ profit_function x = 1250 ∧ (initial_selling_price - x) = 105 :=
by
  sorry

end max_profit_max_profit_price_l290_290092


namespace factorization_simplification_system_of_inequalities_l290_290992

-- Problem 1: Factorization
theorem factorization (x y : ℝ) : 
  x^2 * (x - 3) + y^2 * (3 - x) = (x - 3) * (x + y) * (x - y) := 
sorry

-- Problem 2: Simplification
theorem simplification (x : ℝ) (hx : x ≠ 0) (hx1 : 5 * x ≠ 3) (hx2 : 5 * x ≠ -3) : 
  (2 * x / (5 * x - 3)) / (3 / (25 * x^2 - 9)) * (x / (5 * x + 3)) = (2 / 3) * x^2 := 
sorry

-- Problem 3: System of inequalities
theorem system_of_inequalities (x : ℝ) : 
  ((x - 3) / 2 + 3 ≥ x + 1 ∧ 1 - 3 * (x - 1) < 8 - x) ↔ -2 < x ∧ x ≤ 1 := 
sorry

end factorization_simplification_system_of_inequalities_l290_290992


namespace system_solution_y_greater_than_five_l290_290537

theorem system_solution_y_greater_than_five (m x y : ℝ) :
  (y = (m + 1) * x + 2) → 
  (y = (3 * m - 2) * x + 5) → 
  y > 5 ↔ 
  m ≠ 3 / 2 := 
sorry

end system_solution_y_greater_than_five_l290_290537


namespace solve_for_x_l290_290925

theorem solve_for_x (x : ℝ) (h : 40 / x - 1 = 19) : x = 2 :=
by {
  sorry
}

end solve_for_x_l290_290925


namespace difference_between_numbers_l290_290048

theorem difference_between_numbers :
  ∃ X Y : ℕ, 
    100 ≤ X ∧ X < 1000 ∧
    100 ≤ Y ∧ Y < 1000 ∧
    X + Y = 999 ∧
    1000 * X + Y = 6 * (1000 * Y + X) ∧
    (X - Y = 715 ∨ Y - X = 715) :=
by
  sorry

end difference_between_numbers_l290_290048


namespace double_rooms_percentage_l290_290527

theorem double_rooms_percentage (S : ℝ) (h1 : 0 < S)
  (h2 : ∃ Sd : ℝ, Sd = 0.75 * S)
  (h3 : ∃ Ss : ℝ, Ss = 0.25 * S):
  (0.375 * S) / (0.625 * S) * 100 = 60 := 
by 
  sorry

end double_rooms_percentage_l290_290527


namespace min_fraction_value_l290_290112

theorem min_fraction_value (x : ℝ) (hx : x > 9) : ∃ y, y = 36 ∧ (∀ z, z = (x^2 / (x - 9)) → y ≤ z) :=
by
  sorry

end min_fraction_value_l290_290112


namespace no_same_last_four_digits_pow_l290_290524

theorem no_same_last_four_digits_pow (n m : ℕ) (hn : n > 0) (hm : m > 0) : 
  (5^n % 10000) ≠ (6^m % 10000) :=
by sorry

end no_same_last_four_digits_pow_l290_290524


namespace sum_of_six_angles_l290_290570

theorem sum_of_six_angles (a1 a2 a3 a4 a5 a6 : ℕ) (h1 : a1 + a3 + a5 = 180)
                           (h2 : a2 + a4 + a6 = 180) : 
                           a1 + a2 + a3 + a4 + a5 + a6 = 360 := 
by
  -- omitted proof
  sorry

end sum_of_six_angles_l290_290570


namespace even_function_value_l290_290385

theorem even_function_value (f : ℝ → ℝ) (h_even : ∀ x, f x = f (-x))
  (h_neg : ∀ x, x < 0 → f x = 2 ^ x) :
  f (Real.log 9 / Real.log 4) = 1 / 3 :=
by
  sorry

end even_function_value_l290_290385


namespace triangle_angles_arithmetic_progression_l290_290176

theorem triangle_angles_arithmetic_progression (α β γ : ℝ) (a c : ℝ) :
  (α < β) ∧ (β < γ) ∧ (α + β + γ = 180) ∧
  (∃ x : ℝ, β = α + x ∧ γ = β + x) ∧
  (a = c / 2) → 
  (α = 30) ∧ (β = 60) ∧ (γ = 90) :=
by
  intros h
  sorry

end triangle_angles_arithmetic_progression_l290_290176


namespace minnie_takes_longer_l290_290607

def minnie_speed_flat := 25 -- kph
def minnie_speed_downhill := 35 -- kph
def minnie_speed_uphill := 10 -- kph

def penny_speed_flat := 35 -- kph
def penny_speed_downhill := 45 -- kph
def penny_speed_uphill := 15 -- kph

def distance_flat := 25 -- km
def distance_downhill := 20 -- km
def distance_uphill := 15 -- km

noncomputable def minnie_time := 
  (distance_uphill / minnie_speed_uphill) + 
  (distance_downhill / minnie_speed_downhill) + 
  (distance_flat / minnie_speed_flat) -- hours

noncomputable def penny_time := 
  (distance_uphill / penny_speed_uphill) + 
  (distance_downhill / penny_speed_downhill) + 
  (distance_flat / penny_speed_flat) -- hours

noncomputable def minnie_time_minutes := minnie_time * 60 -- minutes
noncomputable def penny_time_minutes := penny_time * 60 -- minutes

noncomputable def time_difference := minnie_time_minutes - penny_time_minutes -- minutes

theorem minnie_takes_longer : time_difference = 130 :=
  sorry

end minnie_takes_longer_l290_290607


namespace find_k_parallel_lines_l290_290276

theorem find_k_parallel_lines (k : ℝ) : 
  (∀ x y, (k - 1) * x + y + 2 = 0 → 
            (8 * x + (k + 1) * y + k - 1 = 0 → False)) → 
  k = 3 :=
sorry

end find_k_parallel_lines_l290_290276


namespace total_questions_reviewed_l290_290868

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l290_290868


namespace ellipse_equation_fixed_point_l290_290126

theorem ellipse_equation_fixed_point (A B P : ℝ × ℝ) :
    A = (0, -2) →
    B = (3 / 2, -1) →
    P = (1, -2) →
    (∀ x y, (x = 0 → 4 * y^2 = 1) ∧ (x = 3 / 2 → 9 / 4 * x^2 + y^2 = 1) →
      (x^2 / 3 + y^2 / 4 = 1)) ∧
    (∀ M N T H : ℝ × ℝ,
      (M = P → 
      ∃ k, k ≠ 0 ∧ 
      M = (1, -2)) →
      N = (1, 2 * sqrt 6 / 3) →
      (∃ T, (T = ((-sqrt 6 + 3), -2 * sqrt 6 /3)) →
      (∃ H, H = ((-2 * sqrt 6 + 5), -2 * sqrt 6 /3)) →
      (HN = (H - N)))) →
        ∃ F, F = (0, -2) :=
sorry

end ellipse_equation_fixed_point_l290_290126


namespace parabola_equation_l290_290540

theorem parabola_equation :
  ∃ a b c : ℝ, (∀ x y : ℝ, y = a * x^2 + b * x + c ↔ y = -3 * x^2 + 18 * x - 22) ∧
    (∃ a : ℝ, ∀ x : ℝ, y = a * (x - 3) ^ 2 + 5 ↔ ∀ x y : ℝ, y = a * (x - 3)^2 + 5) ∧
    ∃ x y : ℝ, (x = 2 ∧ y = 2) → y = a * (x - 3)^2 + 5 :=
begin
  sorry
end

end parabola_equation_l290_290540


namespace expected_worth_of_coin_flip_l290_290523

-- Define the probabilities and gains/losses
def prob_heads := 2 / 3
def prob_tails := 1 / 3
def gain_heads := 5 
def loss_tails := -9

-- Define the expected value calculation for a coin flip
def expected_value := prob_heads * gain_heads + prob_tails * loss_tails

-- The theorem to be proven
theorem expected_worth_of_coin_flip : expected_value = 1 / 3 :=
by sorry

end expected_worth_of_coin_flip_l290_290523


namespace average_student_headcount_l290_290528

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l290_290528


namespace system_of_equations_has_two_solutions_l290_290777

theorem system_of_equations_has_two_solutions :
  ∃! (x y z : ℕ), x > 0 ∧ y > 0 ∧ z > 0 ∧ 
  xy + yz = 63 ∧ 
  xz + yz = 23 :=
sorry

end system_of_equations_has_two_solutions_l290_290777


namespace matching_polygons_pairs_l290_290794

noncomputable def are_matching_pairs (n m : ℕ) : Prop :=
  2 * ((n - 2) * 180 / n) = 3 * (360 / m)

theorem matching_polygons_pairs (n m : ℕ) :
  are_matching_pairs n m → (n, m) = (3, 9) ∨ (n, m) = (4, 6) ∨ (n, m) = (5, 5) ∨ (n, m) = (8, 4) :=
sorry

end matching_polygons_pairs_l290_290794


namespace triangle_perimeter_l290_290353

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end triangle_perimeter_l290_290353


namespace slips_with_3_l290_290036

-- Definitions of the conditions
def num_slips : ℕ := 15
def expected_value : ℚ := 5.4

-- Theorem statement
theorem slips_with_3 (y : ℕ) (t : ℕ := num_slips) (E : ℚ := expected_value) :
  E = (3 * y + 8 * (t - y)) / t → y = 8 :=
by
  sorry

end slips_with_3_l290_290036


namespace students_chocolate_milk_l290_290897

-- Definitions based on the problem conditions
def students_strawberry_milk : ℕ := 15
def students_regular_milk : ℕ := 3
def total_milks_taken : ℕ := 20

-- The proof goal
theorem students_chocolate_milk : total_milks_taken - (students_strawberry_milk + students_regular_milk) = 2 := by
  -- The proof steps will go here (not required as per instructions)
  sorry

end students_chocolate_milk_l290_290897


namespace vasya_numbers_l290_290810

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290810


namespace arc_length_l290_290962

theorem arc_length 
  (a : ℝ) 
  (α β : ℝ) 
  (hα : 0 < α) 
  (hβ : 0 < β) 
  (h1 : α + β < π) 
  :  ∃ l : ℝ, l = (a * (π - α - β) * (Real.sin α) * (Real.sin β)) / (Real.sin (α + β)) :=
sorry

end arc_length_l290_290962


namespace star_k_l290_290534

def star (x y : ℤ) : ℤ := x^2 - 2 * y + 1

theorem star_k (k : ℤ) : star k (star k k) = -k^2 + 4 * k - 1 :=
by 
  sorry

end star_k_l290_290534


namespace quadratic_shift_l290_290420

theorem quadratic_shift :
  ∀ (x : ℝ), (∃ (y : ℝ), y = -x^2) →
  (∃ (y : ℝ), y = -(x + 1)^2 + 3) :=
by
  intro x
  intro h
  use -(x + 1)^2 + 3
  sorry

end quadratic_shift_l290_290420


namespace inequality_sum_geq_three_l290_290310

theorem inequality_sum_geq_three
  (a b c : ℝ)
  (h : a * b * c = 1) :
  (1 + a + a * b) / (1 + b + a * b) + 
  (1 + b + b * c) / (1 + c + b * c) +
  (1 + c + a * c) / (1 + a + a * c) ≥ 3 := 
sorry

end inequality_sum_geq_three_l290_290310


namespace vasya_numbers_l290_290804

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290804


namespace solution_to_inequality_l290_290373

theorem solution_to_inequality :
  { x : ℝ | ((x^2 - 1) / (x - 4)^2) ≥ 0 } = { x : ℝ | x ≤ -1 ∨ (1 ≤ x ∧ x < 4) ∨ x > 4 } := 
sorry

end solution_to_inequality_l290_290373


namespace fill_pool_time_l290_290114

-- Define the conditions
def pool_volume : ℕ := 15000
def hoses1_rate : ℕ := 2
def hoses1_count : ℕ := 2
def hoses2_rate : ℕ := 3
def hoses2_count : ℕ := 2

-- Calculate the total delivery rate
def total_delivery_rate : ℕ :=
  (hoses1_rate * hoses1_count) + (hoses2_rate * hoses2_count)

-- Calculate the time to fill the pool in minutes
def time_to_fill_in_minutes : ℕ :=
  pool_volume / total_delivery_rate

-- Calculate the time to fill the pool in hours
def time_to_fill_in_hours : ℕ :=
  time_to_fill_in_minutes / 60

-- The theorem to prove
theorem fill_pool_time : time_to_fill_in_hours = 25 := by
  sorry

end fill_pool_time_l290_290114


namespace abs_x_ge_abs_4ax_l290_290076

theorem abs_x_ge_abs_4ax (a : ℝ) (h : ∀ x : ℝ, abs x ≥ 4 * a * x) : abs a ≤ 1 / 4 :=
sorry

end abs_x_ge_abs_4ax_l290_290076


namespace true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l290_290335

theorem true_if_a_gt_1_and_b_gt_1_then_ab_gt_1 (a b : ℝ) (ha : a > 1) (hb : b > 1) : ab > 1 :=
sorry

end true_if_a_gt_1_and_b_gt_1_then_ab_gt_1_l290_290335


namespace probability_not_all_same_l290_290061

-- Definitions of conditions from the problem
def six_sided_die_faces : ℕ := 6
def number_of_dice : ℕ := 5

-- Lean statement to prove the probability calculation
theorem probability_not_all_same : 
  let total_outcomes := six_sided_die_faces ^ number_of_dice in
  let all_same_outcomes := six_sided_die_faces in
  (1 - ((all_same_outcomes : ℚ) / total_outcomes)) = (1295 / 1296) := 
by
  sorry

end probability_not_all_same_l290_290061


namespace difference_between_blue_and_red_balls_l290_290046

-- Definitions and conditions
def number_of_blue_balls := ℕ
def number_of_red_balls := ℕ
def difference_between_balls (m n : ℕ) := m - n

-- Problem statement: Prove that the difference between number_of_blue_balls and number_of_red_balls
-- can be any natural number greater than 1.
theorem difference_between_blue_and_red_balls (m n : ℕ) (h1 : m > n) (h2 : 
  let P_same := (n * (n - 1) + m * (m - 1)) / ((n + m) * (n + m - 1))
  let P_diff := 2 * (n * m) / ((n + m) * (n + m - 1))
  P_same = P_diff
  ) : ∃ a : ℕ, a > 1 ∧ a = m - n :=
by
  sorry

end difference_between_blue_and_red_balls_l290_290046


namespace calculate_speed_l290_290009

theorem calculate_speed :
  ∀ (distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph : ℚ),
  distance_ft = 200 →
  time_sec = 2 →
  miles_per_ft = 1 / 5280 →
  hours_per_sec = 1 / 3600 →
  approx_speed_mph = 68.1818181818 →
  (distance_ft * miles_per_ft) / (time_sec * hours_per_sec) = approx_speed_mph :=
by
  intros distance_ft time_sec miles_per_ft hours_per_sec approx_speed_mph
  intro h_distance_eq h_time_eq h_miles_eq h_hours_eq h_speed_eq
  sorry

end calculate_speed_l290_290009


namespace total_questions_reviewed_l290_290869

theorem total_questions_reviewed (questions_per_student : ℕ) (students_per_class : ℕ) (number_of_classes : ℕ) :
  questions_per_student = 10 → students_per_class = 35 → number_of_classes = 5 →
  questions_per_student * students_per_class * number_of_classes = 1750 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end total_questions_reviewed_l290_290869


namespace madhav_rank_from_last_is_15_l290_290339

-- Defining the conditions
def class_size : ℕ := 31
def madhav_rank_from_start : ℕ := 17

-- Statement to be proved
theorem madhav_rank_from_last_is_15 :
  (class_size - madhav_rank_from_start + 1) = 15 := by
  sorry

end madhav_rank_from_last_is_15_l290_290339


namespace distance_travelled_l290_290584

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end distance_travelled_l290_290584


namespace range_of_quadratic_within_domain_l290_290327

noncomputable def quadratic_func (x : ℝ) : ℝ := x^2 - 2*x 

theorem range_of_quadratic_within_domain : 
  ∀ y, y ∈ set.range (λ x, quadratic_func x) ↔ ∃ x, -1 ≤ x ∧ x ≤ 3 ∧ y = quadratic_func x := by
  sorry

end range_of_quadratic_within_domain_l290_290327


namespace remainder_of_division_l290_290152

theorem remainder_of_division (dividend divisor quotient remainder : ℕ)
  (h1 : dividend = 55053)
  (h2 : divisor = 456)
  (h3 : quotient = 120)
  (h4 : remainder = dividend - divisor * quotient) : 
  remainder = 333 := by
  sorry

end remainder_of_division_l290_290152


namespace average_mb_per_hour_l290_290087

theorem average_mb_per_hour
  (days : ℕ)
  (original_space  : ℕ)
  (compression_rate : ℝ)
  (total_hours : ℕ := days * 24)
  (effective_space : ℝ := original_space * (1 - compression_rate))
  (space_per_hour : ℝ := effective_space / total_hours) :
  days = 20 ∧ original_space = 25000 ∧ compression_rate = 0.10 → 
  (Int.floor (space_per_hour + 0.5)) = 47 := by
  intros
  sorry

end average_mb_per_hour_l290_290087


namespace josh_payment_correct_l290_290940

/-- Josh's purchase calculation -/
def josh_total_payment : ℝ :=
  let string_cheese_cost := 0.10
  let number_of_cheeses_per_pack := 20
  let packs_bought := 3
  let sales_tax_rate := 0.12
  let cost_before_tax := packs_bought * number_of_cheeses_per_pack * string_cheese_cost
  let sales_tax := sales_tax_rate * cost_before_tax
  cost_before_tax + sales_tax

theorem josh_payment_correct :
  josh_total_payment = 6.72 := by
  sorry

end josh_payment_correct_l290_290940


namespace solution_a_eq_2_solution_a_in_real_l290_290272

-- Define the polynomial inequality for the given conditions
def inequality (x : ℝ) (a : ℝ) : Prop := 12 * x ^ 2 - a * x > a ^ 2

-- Proof statement for when a = 2
theorem solution_a_eq_2 :
  ∀ x : ℝ, inequality x 2 ↔ (x < - (1 : ℝ) / 2) ∨ (x > (2 : ℝ) / 3) :=
sorry

-- Proof statement for when a is in ℝ
theorem solution_a_in_real (a : ℝ) :
  ∀ x : ℝ, inequality x a ↔
    if h : 0 < a then (x < - a / 4) ∨ (x > a / 3)
    else if h : a = 0 then (x ≠ 0)
    else (x < a / 3) ∨ (x > - a / 4) :=
sorry

end solution_a_eq_2_solution_a_in_real_l290_290272


namespace intervals_of_monotonicity_max_min_on_interval_l290_290563

noncomputable def f (x : ℝ) : ℝ := x^3 - 3 * x

theorem intervals_of_monotonicity :
  (∀ x y : ℝ, x ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → y ∈ (Set.Iio (-1) ∪ Set.Ioi (1)) → x < y → f x < f y) ∧
  (∀ x y : ℝ, x ∈ (Set.Ioo (-1) 1) → y ∈ (Set.Ioo (-1) 1) → x < y → f x > f y) :=
by
  sorry

theorem max_min_on_interval :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → f x ≤ 2) ∧
  (∀ x : ℝ, x ∈ Set.Icc (-3) 2 → -18 ≤ f x) ∧
  ((∃ x₁ : ℝ, x₁ ∈ Set.Icc (-3) 2 ∧ f x₁ = 2) ∧ (∃ x₂ : ℝ, x₂ ∈ Set.Icc (-3) 2 ∧ f x₂ = -18)) :=
by
  sorry

end intervals_of_monotonicity_max_min_on_interval_l290_290563


namespace vasya_numbers_l290_290829

-- Define the conditions
def sum_eq_product (x y : ℝ) : Prop := x + y = x * y
def product_eq_quotient (x y : ℝ) : Prop := x * y = x / y

-- State the proof problem
theorem vasya_numbers : 
  ∃ x y : ℝ, sum_eq_product x y ∧ product_eq_quotient x y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end vasya_numbers_l290_290829


namespace vinnie_tips_l290_290456

variable (Paul Vinnie : ℕ)

def tips_paul := 14
def more_vinnie_than_paul := 16

theorem vinnie_tips :
  Vinnie = tips_paul + more_vinnie_than_paul :=
by
  unfold tips_paul more_vinnie_than_paul -- unfolding defined values
  exact sorry

end vinnie_tips_l290_290456


namespace sarah_min_days_l290_290956

theorem sarah_min_days (r P B : ℝ) (x : ℕ) (h_r : r = 0.1) (h_P : P = 20) (h_B : B = 60) :
  (P + r * P * x ≥ B) → (x ≥ 20) :=
by
  sorry

end sarah_min_days_l290_290956


namespace AplusBplusC_4_l290_290021

theorem AplusBplusC_4 :
  ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 1 ∧ Nat.gcd a c = 1 ∧ (a^2 + a * b + b^2 = c^2) ∧ (a + b + c = 4) :=
by
  sorry

end AplusBplusC_4_l290_290021


namespace allocation_schemes_count_l290_290568

theorem allocation_schemes_count (schools total_people : ℕ) (at_least_one_per_school : ℕ) :
  schools = 7 → total_people = 10 → at_least_one_per_school = 1 → 
  (∑ i in Finset.range schools, at_least_one_per_school) + 3 = total_people → 
  Nat.choose (total_people - 1) (schools - 1) = 84 :=
by
  intros h_schools h_people h_min selection_eq
  replace selection_eq : total_people - schools = 3 := by linarith [h_people, selection_eq]
  have : Nat.choose 9 6 = 84 := by
    sorry
  exact this

end allocation_schemes_count_l290_290568


namespace operation_is_commutative_and_associative_l290_290172

variables {S : Type} (op : S → S → S)

-- defining the properties given in the conditions
def idempotent (op : S → S → S) : Prop :=
  ∀ (a : S), op a a = a

def medial (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op (op b c) a

-- defining commutative and associative properties
def commutative (op : S → S → S) : Prop :=
  ∀ (a b : S), op a b = op b a

def associative (op : S → S → S) : Prop :=
  ∀ (a b c : S), op (op a b) c = op a (op b c)

-- statement of the theorem to prove
theorem operation_is_commutative_and_associative 
  (idemp : idempotent op) 
  (med : medial op) : commutative op ∧ associative op :=
sorry

end operation_is_commutative_and_associative_l290_290172


namespace problem_solution_l290_290141

theorem problem_solution (a : ℝ) : 
  ( ∀ x : ℝ, (ax - 1) * (x + 1) < 0 ↔ (x ∈ Set.Iio (-1) ∨ x ∈ Set.Ioi (-1 / 2)) ) →
  a = -2 :=
by
  sorry

end problem_solution_l290_290141


namespace proof_problem_l290_290269

variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x : ℝ, f (-x) = f x

-- Define what it means for a function to be increasing on (-∞, 0)
def is_increasing_on_neg (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, x < y → y < 0 → f x < f y

-- Define what it means for a function to be decreasing on (0, +∞)
def is_decreasing_on_pos (f : ℝ → ℝ) : Prop := ∀ x y : ℝ, 0 < x → x < y → f y < f x

theorem proof_problem 
  (h_even : is_even_function f) 
  (h_inc_neg : is_increasing_on_neg f) : 
  (∀ x : ℝ, f (-x) - f x = 0) ∧ (is_decreasing_on_pos f) :=
by
  sorry

end proof_problem_l290_290269


namespace charlie_older_than_bobby_by_three_l290_290601

variable (J C B x : ℕ)

def jenny_older_charlie_by_five (J C : ℕ) := J = C + 5
def charlie_age_when_jenny_twice_bobby_age (C x : ℕ) := C + x = 11
def jenny_twice_bobby (J B x : ℕ) := J + x = 2 * (B + x)

theorem charlie_older_than_bobby_by_three
  (h1 : jenny_older_charlie_by_five J C)
  (h2 : charlie_age_when_jenny_twice_bobby_age C x)
  (h3 : jenny_twice_bobby J B x) :
  (C = B + 3) :=
by
  sorry

end charlie_older_than_bobby_by_three_l290_290601


namespace Vasya_numbers_l290_290831

theorem Vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 :=
by {
  sorry
}

end Vasya_numbers_l290_290831


namespace number_of_sequences_mod_1000_l290_290237

theorem number_of_sequences_mod_1000 : 
  (nat.choose 1016 12) % 1000 = 16 :=
  sorry

end number_of_sequences_mod_1000_l290_290237


namespace typing_speed_ratio_l290_290991

variable (T M : ℝ)

-- Conditions
def condition1 : Prop := T + M = 12
def condition2 : Prop := T + 1.25 * M = 14

-- Proof statement
theorem typing_speed_ratio (h1 : condition1 T M) (h2 : condition2 T M) : M / T = 2 := by
  sorry

end typing_speed_ratio_l290_290991


namespace one_thirds_in_nine_thirds_l290_290283

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l290_290283


namespace five_dice_not_all_same_probability_l290_290058

noncomputable def probability_not_all_same : ℚ :=
  let total_outcomes := (6:ℚ) ^ 5
  let favorable_outcomes := (6:ℚ)
  1 - (favorable_outcomes / total_outcomes)

theorem five_dice_not_all_same_probability :
  probability_not_all_same = 1295 / 1296 :=
by
  unfold probability_not_all_same
  norm_cast
  simp
  sorry

end five_dice_not_all_same_probability_l290_290058


namespace average_student_headcount_l290_290529

variable (headcount_02_03 headcount_03_04 headcount_04_05 headcount_05_06 : ℕ)
variable {h_02_03 : headcount_02_03 = 10900}
variable {h_03_04 : headcount_03_04 = 10500}
variable {h_04_05 : headcount_04_05 = 10700}
variable {h_05_06 : headcount_05_06 = 11300}

theorem average_student_headcount : 
  (headcount_02_03 + headcount_03_04 + headcount_04_05 + headcount_05_06) / 4 = 10850 := 
by 
  sorry

end average_student_headcount_l290_290529


namespace opposite_of_reciprocal_negative_one_third_l290_290477

theorem opposite_of_reciprocal_negative_one_third : -(1 / (-1 / 3)) = 3 := by
  sorry

end opposite_of_reciprocal_negative_one_third_l290_290477


namespace b_alone_days_l290_290203

theorem b_alone_days {a b : ℝ} (h1 : a + b = 1/6) (h2 : a = 1/11) : b = 1/(66/5) :=
by sorry

end b_alone_days_l290_290203


namespace probability_not_all_dice_same_l290_290055

theorem probability_not_all_dice_same :
  let total_outcomes := 6^5
  let same_number_outcomes := 6
  let probability_same_number := same_number_outcomes / total_outcomes
  let probability_not_same_number := 1 - probability_same_number
  probability_not_same_number = (1295 : ℚ) / 1296 :=
by
  sorry

end probability_not_all_dice_same_l290_290055


namespace vasya_numbers_l290_290800

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l290_290800


namespace smallest_value_of_expression_l290_290499

theorem smallest_value_of_expression :
  ∃ (k l : ℕ), 36^k - 5^l = 11 := 
sorry

end smallest_value_of_expression_l290_290499


namespace value_of_t_plus_one_over_t_l290_290906

theorem value_of_t_plus_one_over_t
  (t : ℝ)
  (h1 : t^2 - 3 * t + 1 = 0)
  (h2 : t ≠ 0) :
  t + 1 / t = 3 :=
by
  sorry

end value_of_t_plus_one_over_t_l290_290906


namespace percentage_increase_in_expenses_l290_290653

theorem percentage_increase_in_expenses:
  ∀ (S : ℝ) (original_save_percentage new_savings : ℝ), 
  S = 5750 → 
  original_save_percentage = 0.20 →
  new_savings = 230 →
  (original_save_percentage * S - new_savings) / (S - original_save_percentage * S) * 100 = 20 :=
by
  intros S original_save_percentage new_savings HS Horiginal_save_percentage Hnew_savings
  rw [HS, Horiginal_save_percentage, Hnew_savings]
  sorry

end percentage_increase_in_expenses_l290_290653


namespace terminal_zeros_75_480_l290_290285

theorem terminal_zeros_75_480 :
  let x := 75
  let y := 480
  let fact_x := 5^2 * 3
  let fact_y := 2^5 * 3 * 5
  let product := fact_x * fact_y
  let num_zeros := min (3) (5)
  num_zeros = 3 :=
by
  sorry

end terminal_zeros_75_480_l290_290285


namespace rectangle_area_l290_290642

theorem rectangle_area
  (b : ℝ)
  (l : ℝ)
  (P : ℝ)
  (h1 : l = 3 * b)
  (h2 : P = 2 * (l + b))
  (h3 : P = 112) :
  l * b = 588 := by
  sorry

end rectangle_area_l290_290642


namespace relationship_of_new_stationary_points_l290_290245

noncomputable def g (x : ℝ) : ℝ := Real.sin x
noncomputable def h (x : ℝ) : ℝ := Real.log x
noncomputable def phi (x : ℝ) : ℝ := x^3

noncomputable def g' (x : ℝ) : ℝ := Real.cos x
noncomputable def h' (x : ℝ) : ℝ := 1 / x
noncomputable def phi' (x : ℝ) : ℝ := 3 * x^2

-- Definitions of the new stationary points
noncomputable def new_stationary_point_g (x : ℝ) : Prop := g x = g' x
noncomputable def new_stationary_point_h (x : ℝ) : Prop := h x = h' x
noncomputable def new_stationary_point_phi (x : ℝ) : Prop := phi x = phi' x

theorem relationship_of_new_stationary_points :
  ∃ (a b c : ℝ), (0 < a ∧ a < π) ∧ (1 < b ∧ b < Real.exp 1) ∧ (c ≠ 0) ∧
  new_stationary_point_g a ∧ new_stationary_point_h b ∧ new_stationary_point_phi c ∧
  c > b ∧ b > a :=
by
  sorry

end relationship_of_new_stationary_points_l290_290245


namespace vasya_numbers_l290_290820

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290820


namespace find_inradius_l290_290631

-- Define variables and constants
variables (P A : ℝ)
variables (s r : ℝ)

-- Given conditions as definitions
def perimeter_triangle : Prop := P = 36
def area_triangle : Prop := A = 45

-- Semi-perimeter definition
def semi_perimeter : Prop := s = P / 2

-- Inradius and area relationship
def inradius_area_relation : Prop := A = r * s

-- Theorem statement
theorem find_inradius (hP : perimeter_triangle P) (hA : area_triangle A) (hs : semi_perimeter P s) (har : inradius_area_relation A r s) :
  r = 2.5 :=
by
  sorry

end find_inradius_l290_290631


namespace cistern_emptied_fraction_l290_290972

variables (minutes : ℕ) (fractionA fractionB fractionC : ℚ)

def pipeA_rate := 1 / 2 / 12
def pipeB_rate := 1 / 3 / 15
def pipeC_rate := 1 / 4 / 20

def time_active := 5

def emptiedA := pipeA_rate * time_active
def emptiedB := pipeB_rate * time_active
def emptiedC := pipeC_rate * time_active

def total_emptied := emptiedA + emptiedB + emptiedC

theorem cistern_emptied_fraction :
  total_emptied = 55 / 144 := by
  sorry

end cistern_emptied_fraction_l290_290972


namespace diagonals_in_25_sided_polygon_l290_290247

-- Define a function to calculate the number of specific diagonals in a convex polygon with n sides
def number_of_diagonals (n : ℕ) : ℕ :=
  n * (n - 5) / 2

-- Theorem stating the number of diagonals for a convex polygon with 25 sides with the given condition
theorem diagonals_in_25_sided_polygon : number_of_diagonals 25 = 250 := 
sorry

end diagonals_in_25_sided_polygon_l290_290247


namespace total_questions_reviewed_l290_290870

theorem total_questions_reviewed (questions_per_student : ℕ) (students_per_class : ℕ) (number_of_classes : ℕ) :
  questions_per_student = 10 → students_per_class = 35 → number_of_classes = 5 →
  questions_per_student * students_per_class * number_of_classes = 1750 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  norm_num
  sorry

end total_questions_reviewed_l290_290870


namespace kyler_wins_one_game_l290_290755

theorem kyler_wins_one_game :
  ∃ (Kyler_wins : ℕ),
    (Kyler_wins + 3 + 2 + 2 = 6 ∧
    Kyler_wins + 3 = 6 ∧
    Kyler_wins = 1) := by
  sorry

end kyler_wins_one_game_l290_290755


namespace Alchemerion_is_3_times_older_than_his_son_l290_290871

-- Definitions of Alchemerion's age, his father's age and the sum condition
def Alchemerion_age : ℕ := 360
def Father_age (A : ℕ) := 2 * A + 40
def age_sum (A S F : ℕ) := A + S + F

-- Main theorem statement
theorem Alchemerion_is_3_times_older_than_his_son (S : ℕ) (h1 : Alchemerion_age = 360)
    (h2 : Father_age Alchemerion_age = 2 * Alchemerion_age + 40)
    (h3 : age_sum Alchemerion_age S (Father_age Alchemerion_age) = 1240) :
    Alchemerion_age / S = 3 :=
sorry

end Alchemerion_is_3_times_older_than_his_son_l290_290871


namespace sale_in_third_month_l290_290346

theorem sale_in_third_month (
  f1 f2 f4 f5 f6 average : ℕ
) (h1 : f1 = 7435) 
  (h2 : f2 = 7927) 
  (h4 : f4 = 8230) 
  (h5 : f5 = 7562) 
  (h6 : f6 = 5991) 
  (havg : average = 7500) :
  ∃ f3, f3 = 7855 ∧ f1 + f2 + f3 + f4 + f5 + f6 = average * 6 :=
by {
  sorry
}

end sale_in_third_month_l290_290346


namespace initial_volume_of_mixture_l290_290088

/-- A mixture contains 10% water. 
5 liters of water should be added to this so that the water becomes 20% in the new mixture.
Prove that the initial volume of the mixture is 40 liters. -/
theorem initial_volume_of_mixture 
  (V : ℚ) -- Define the initial volume of the mixture
  (h1 : 0.10 * V + 5 = 0.20 * (V + 5)) -- Condition on the mixture
  : V = 40 := -- The statement to prove
by
  sorry -- Proof not required

end initial_volume_of_mixture_l290_290088


namespace alpha_beta_square_inequality_l290_290921

theorem alpha_beta_square_inequality
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h2 : β ∈ Set.Icc (-Real.pi / 2) (Real.pi / 2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 :=
by
  sorry

end alpha_beta_square_inequality_l290_290921


namespace conic_section_eccentricity_l290_290133

noncomputable def eccentricity (m : ℝ) : ℝ :=
if m = 2 then 1 / Real.sqrt 2 else
if m = -2 then Real.sqrt 3 else
0

theorem conic_section_eccentricity (m : ℝ) (h : 4 * 1 = m * m) :
  eccentricity m = 1 / Real.sqrt 2 ∨ eccentricity m = Real.sqrt 3 :=
by
  sorry

end conic_section_eccentricity_l290_290133


namespace distance_travelled_l290_290583

variables (D : ℝ) (h_eq : D / 10 = (D + 20) / 14)

theorem distance_travelled : D = 50 :=
by sorry

end distance_travelled_l290_290583


namespace total_questions_reviewed_l290_290867

-- Defining the conditions
def questions_per_student : Nat := 10
def students_per_class : Nat := 35
def num_classes : Nat := 5

-- Define the total number of questions that Professor Oscar must review.
def total_questions : Nat := questions_per_student * students_per_class * num_classes

-- Theorem statement to be proved
theorem total_questions_reviewed : total_questions = 1750 := by
  sorry

end total_questions_reviewed_l290_290867


namespace price_of_sundae_l290_290071

theorem price_of_sundae (total_ice_cream_bars : ℕ) (total_sundae_price : ℝ)
                        (total_price : ℝ) (price_per_ice_cream_bar : ℝ) (num_ice_cream_bars : ℕ) (num_sundaes : ℕ)
                        (h1 : total_ice_cream_bars = num_ice_cream_bars)
                        (h2 : total_price = 200)
                        (h3 : price_per_ice_cream_bar = 0.40)
                        (h4 : num_ice_cream_bars = 200)
                        (h5 : num_sundaes = 200)
                        (h6 : total_ice_cream_bars * price_per_ice_cream_bar + total_sundae_price = total_price) :
  total_sundae_price / num_sundaes = 0.60 :=
sorry

end price_of_sundae_l290_290071


namespace scientific_notation_of_50000_l290_290175

theorem scientific_notation_of_50000 :
  50000 = 5 * 10^4 :=
sorry

end scientific_notation_of_50000_l290_290175


namespace find_second_number_in_second_set_l290_290621

theorem find_second_number_in_second_set :
    (14 + 32 + 53) / 3 = 3 + (21 + x + 22) / 3 → x = 47 :=
by intro h
   sorry

end find_second_number_in_second_set_l290_290621


namespace pete_and_raymond_spent_together_l290_290031

    def value_nickel : ℕ := 5
    def value_dime : ℕ := 10
    def value_quarter : ℕ := 25

    def pete_nickels_spent : ℕ := 4
    def pete_dimes_spent : ℕ := 3
    def pete_quarters_spent : ℕ := 2

    def raymond_initial : ℕ := 250
    def raymond_nickels_left : ℕ := 5
    def raymond_dimes_left : ℕ := 7
    def raymond_quarters_left : ℕ := 4
    
    def total_spent : ℕ := 155

    theorem pete_and_raymond_spent_together :
      (pete_nickels_spent * value_nickel + pete_dimes_spent * value_dime + pete_quarters_spent * value_quarter)
      + (raymond_initial - (raymond_nickels_left * value_nickel + raymond_dimes_left * value_dime + raymond_quarters_left * value_quarter))
      = total_spent :=
      by
        sorry
    
end pete_and_raymond_spent_together_l290_290031


namespace sum_smallest_largest_eq_2y_l290_290465

theorem sum_smallest_largest_eq_2y (n : ℕ) (y a : ℕ) 
  (h1 : 2 * a + 2 * (n - 1) / n = y) : 
  2 * y = (2 * a + 2 * (n - 1)) := 
sorry

end sum_smallest_largest_eq_2y_l290_290465


namespace find_other_number_l290_290207

theorem find_other_number (B : ℕ) (hcf_cond : Nat.gcd 36 B = 14) (lcm_cond : Nat.lcm 36 B = 396) : B = 66 :=
sorry

end find_other_number_l290_290207


namespace derivative_at_1_l290_290292

def f (x : ℝ) : ℝ := x^2 - 2 * x + 3

theorem derivative_at_1 : deriv f 1 = 0 :=
by
  -- Proof to be provided
  sorry

end derivative_at_1_l290_290292


namespace probability_not_pulling_prize_twice_l290_290590

theorem probability_not_pulling_prize_twice
  (favorable : ℕ)
  (unfavorable : ℕ)
  (total : ℕ := favorable + unfavorable)
  (P_prize : ℚ := favorable / total)
  (P_not_prize : ℚ := 1 - P_prize)
  (P_not_prize_twice : ℚ := P_not_prize * P_not_prize) :
  P_not_prize_twice = 36 / 121 :=
by
  have favorable : ℕ := 5
  have unfavorable : ℕ := 6
  have total : ℕ := favorable + unfavorable
  have P_prize : ℚ := favorable / total
  have P_not_prize : ℚ := 1 - P_prize
  have P_not_prize_twice : ℚ := P_not_prize * P_not_prize
  sorry

end probability_not_pulling_prize_twice_l290_290590


namespace marbles_left_l290_290166

def initial_marbles : ℝ := 9.0
def given_marbles : ℝ := 3.0

theorem marbles_left : initial_marbles - given_marbles = 6.0 := 
by
  sorry

end marbles_left_l290_290166


namespace cos_probability_ge_one_half_in_range_l290_290596

theorem cos_probability_ge_one_half_in_range :
  let interval_length := (Real.pi / 2) - (- (Real.pi / 2))
  let favorable_length := (Real.pi / 3) - (- (Real.pi / 3))
  (favorable_length / interval_length) = (2 / 3) := by
  sorry

end cos_probability_ge_one_half_in_range_l290_290596


namespace abs_diff_of_two_numbers_l290_290481

theorem abs_diff_of_two_numbers (x y : ℝ) (h1 : x + y = 34) (h2 : x * y = 240) : abs (x - y) = 14 :=
by
  sorry

end abs_diff_of_two_numbers_l290_290481


namespace derivative_at_2_l290_290390

theorem derivative_at_2 (f : ℝ → ℝ) (h : ∀ x, f x = x^2 * deriv f 2 + 5 * x) :
    deriv f 2 = -5/3 :=
by
  sorry

end derivative_at_2_l290_290390


namespace probability_margo_pairing_l290_290591

-- Definition of the problem
def num_students : ℕ := 32
def num_pairings (n : ℕ) : ℕ := n - 1
def favorable_pairings : ℕ := 2

-- Theorem statement
theorem probability_margo_pairing :
  num_students = 32 →
  ∃ (p : ℚ), p = favorable_pairings / num_pairings num_students ∧ p = 2/31 :=
by
  intros h
  -- The proofs are omitted for brevity.
  sorry

end probability_margo_pairing_l290_290591


namespace mean_of_set_median_is_128_l290_290770

theorem mean_of_set_median_is_128 (m : ℝ) (h : m + 7 = 12) : 
  (m + (m + 4) + (m + 7) + (m + 10) + (m + 18)) / 5 = 12.8 := by
  sorry

end mean_of_set_median_is_128_l290_290770


namespace quadrilateral_areas_product_l290_290507

noncomputable def areas_product_property (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) : Prop :=
  (S_ADP * S_BCP * S_ABP * S_CDP) % 10000 ≠ 1988
  
theorem quadrilateral_areas_product (S_ADP S_ABP S_CDP S_BCP : ℕ) (h1 : S_ADP * S_BCP = S_ABP * S_CDP) :
  areas_product_property S_ADP S_ABP S_CDP S_BCP h1 :=
by
  sorry

end quadrilateral_areas_product_l290_290507


namespace leopards_arrangement_correct_l290_290453

noncomputable def leopards_arrangement : Nat :=
  let shortestEndsWays : Nat := 2
  let tallestAdjWays : Nat := 6
  let arrangeTallestWays : Nat := Nat.factorial 2
  let arrangeRemainingWays : Nat := Nat.factorial 5
  shortestEndsWays * tallestAdjWays * arrangeTallestWays * arrangeRemainingWays

theorem leopards_arrangement_correct : leopards_arrangement = 2880 := by
  sorry

end leopards_arrangement_correct_l290_290453


namespace correct_translation_of_tradition_l290_290637

def is_adjective (s : String) : Prop :=
  s = "传统的"

def is_correct_translation (s : String) (translation : String) : Prop :=
  s = "传统的" → translation = "traditional"

theorem correct_translation_of_tradition : 
  is_adjective "传统的" ∧ is_correct_translation "传统的" "traditional" :=
by
  sorry

end correct_translation_of_tradition_l290_290637


namespace arrange_athletes_l290_290443

theorem arrange_athletes :
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  (Nat.choose athletes country_athletes) *
  (Nat.choose (athletes - country_athletes) country_athletes) *
  (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
  (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520 :=
by
  let athletes := 8
  let countries := 4
  let country_athletes := 2
  show (Nat.choose athletes country_athletes) *
       (Nat.choose (athletes - country_athletes) country_athletes) *
       (Nat.choose (athletes - 2 * country_athletes) country_athletes) *
       (Nat.choose (athletes - 3 * country_athletes) country_athletes) = 2520
  sorry

end arrange_athletes_l290_290443


namespace pairs_with_green_shirts_l290_290931

theorem pairs_with_green_shirts (red_shirts green_shirts total_pairs red_pairs : ℕ) 
    (h1 : red_shirts = 70) 
    (h2 : green_shirts = 58) 
    (h3 : total_pairs = 64) 
    (h4 : red_pairs = 34) 
    : (∃ green_pairs : ℕ, green_pairs = 28) := 
by 
    sorry

end pairs_with_green_shirts_l290_290931


namespace product_of_triangle_areas_not_end_in_1988_l290_290510

theorem product_of_triangle_areas_not_end_in_1988
  (a b c d : ℕ)
  (h1 : a * c = b * d)
  (hp : (a * b * c * d) = (a * c)^2)
  : ¬(∃ k : ℕ, (a * b * c * d) = 10000 * k + 1988) :=
sorry

end product_of_triangle_areas_not_end_in_1988_l290_290510


namespace problem_6509_l290_290975

theorem problem_6509 :
  ∃ (m n : ℕ), gcd m n = 1 ∧ (100 * m + n = 6509) ∧
  ∃ (A B C D E : EuclideanSpace ℝ (Fin 3)),
  dist A B = 13 ∧ dist B C = 14 ∧ dist C A = 15 ∧
  PointsOnLine A C D ∧ PointsOnLine A B E ∧
  CyclicQuadrilateral B C D E ∧
  PointOnBC (fold A D E) B C ∧
  sameDE ((dist D E) = (m:ℤ)/ (n:ℤ)) :=
begin
  sorry
end

end problem_6509_l290_290975


namespace inequality_proof_l290_290387

theorem inequality_proof (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  a / (b + c) + b / (c + d) + c / (d + a) + d / (a + b) ≥ 2 := 
sorry

end inequality_proof_l290_290387


namespace solve_trig_equation_l290_290985

open Real

theorem solve_trig_equation (k : ℕ) :
    (∀ x, 8.459 * cos x^2 * cos (x^2) * (tan (x^2) + 2 * tan x) + tan x^3 * (1 - sin (x^2)^2) * (2 - tan x * tan (x^2)) = 0) ↔
    (∃ k : ℕ, x = -1 + sqrt (π * k + 1) ∨ x = -1 - sqrt (π * k + 1)) :=
sorry

end solve_trig_equation_l290_290985


namespace bc_ad_divisible_by_u_l290_290463

theorem bc_ad_divisible_by_u 
  (a b c d u : ℤ) 
  (h1 : u ∣ a * c)
  (h2 : u ∣ b * c + a * d)
  (h3 : u ∣ b * d) : 
  u ∣ b * c ∧ u ∣ a * d :=
by
  sorry

end bc_ad_divisible_by_u_l290_290463


namespace box_dimensions_sum_l290_290658

theorem box_dimensions_sum (X Y Z : ℝ) (hXY : X * Y = 18) (hXZ : X * Z = 54) (hYZ : Y * Z = 36) (hX_pos : X > 0) (hY_pos : Y > 0) (hZ_pos : Z > 0) :
  X + Y + Z = 11 := 
sorry

end box_dimensions_sum_l290_290658


namespace Z_equals_i_l290_290262

noncomputable def Z : ℂ := (Real.sqrt 2 - (Complex.I ^ 3)) / (1 - Real.sqrt 2 * Complex.I)

theorem Z_equals_i : Z = Complex.I := 
by 
  sorry

end Z_equals_i_l290_290262


namespace largest_number_in_box_l290_290486

theorem largest_number_in_box
  (a : ℕ)
  (sum_eq_480 : a + (a + 1) + (a + 2) + (a + 10) + (a + 11) + (a + 12) = 480) :
  a + 12 = 86 :=
by
  sorry

end largest_number_in_box_l290_290486


namespace volume_of_sphere_l290_290562

noncomputable def sphere_volume (r : ℝ) : ℝ := (4 / 3) * Real.pi * r^3

def tetrahedron_on_sphere (A B C D : EuclideanSpace ℝ 3) (O : EuclideanSpace ℝ 3) : Prop :=
  dist O A = dist O B ∧
  dist O B = dist O C ∧
  dist O C = dist O D

theorem volume_of_sphere
  (A B C D : EuclideanSpace ℝ 3)
  (O : EuclideanSpace ℝ 3)
  (h1 : tetrahedron_on_sphere A B C D O)
  (h2 : (dist D C) ≠ 0)
  (h3 : (dist D (dist C Plane(ABC)) = 0))
  (h4 : dist A C = 2 * Real.sqrt 3)
  (h5 : ∀ (X Y Z : EuclideanSpace ℝ 3), (X = A ∧ Y = B ∧ Z = C) → equilateral_triangle X Y Z)
  (h6 : ∀ (X Y Z : EuclideanSpace ℝ 3), (X = A ∧ Y = C ∧ Z = D) → isosceles_triangle X Y Z)
  : sphere_volume (Real.sqrt 7) = (28 * Real.sqrt 7 / 3) * Real.pi :=
by
sory

end volume_of_sphere_l290_290562


namespace amount_each_person_needs_to_raise_l290_290640

theorem amount_each_person_needs_to_raise (Total_goal Already_collected Number_of_people : ℝ) 
(h1 : Total_goal = 2400) (h2 : Already_collected = 300) (h3 : Number_of_people = 8) : 
    (Total_goal - Already_collected) / Number_of_people = 262.5 := 
by
  sorry

end amount_each_person_needs_to_raise_l290_290640


namespace parallel_lines_when_m_is_neg7_l290_290918

-- Given two lines l1 and l2 defined as:
def l1 (m : ℤ) (x y : ℤ) := (3 + m) * x + 4 * y = 5 - 3 * m
def l2 (m : ℤ) (x y : ℤ) := 2 * x + (5 + m) * y = 8

-- The proof problem to show that l1 is parallel to l2 when m = -7
theorem parallel_lines_when_m_is_neg7 :
  ∃ m : ℤ, (∀ x y : ℤ, l1 m x y → l2 m x y) → m = -7 := 
sorry

end parallel_lines_when_m_is_neg7_l290_290918


namespace expected_number_of_heads_l290_290739

def probability_heads_after_up_to_four_flips : ℝ :=
  1 / 2 + 1 / 4 + 1 / 8 + 1 / 16

theorem expected_number_of_heads (n : ℕ) (h : n = 80)
  (p_heads : ℝ) (h_p_heads : p_heads = probability_heads_after_up_to_four_flips):
  (n : ℝ) * p_heads = 75 :=
by
  intros
  rw [h, h_p_heads]
  sorry

end expected_number_of_heads_l290_290739


namespace swimming_pool_cost_l290_290786

/-!
# Swimming Pool Cost Problem

Given:
* The pool takes 50 hours to fill.
* The hose runs at 100 gallons per hour.
* Water costs 1 cent for 10 gallons.

Prove that the total cost to fill the pool is 5 dollars.
-/

theorem swimming_pool_cost :
  let hours_to_fill := 50
  let hose_rate := 100  -- gallons per hour
  let cost_per_gallon := 0.01 / 10  -- dollars per gallon
  let total_volume := hours_to_fill * hose_rate  -- total volume in gallons
  let total_cost := total_volume * cost_per_gallon
  total_cost = 5 :=
by
  sorry

end swimming_pool_cost_l290_290786


namespace sum_of_cubes_of_roots_l290_290366

theorem sum_of_cubes_of_roots:
  (∀ r s t : ℝ, (r + s + t = 8) ∧ (r * s + s * t + t * r = 9) ∧ (r * s * t = 2) → r^3 + s^3 + t^3 = 344) :=
by
  intros r s t h
  have h1 := h.1
  have h2 := h.2.1
  have h3 := h.2.2
  sorry

end sum_of_cubes_of_roots_l290_290366


namespace triangle_base_length_l290_290620

theorem triangle_base_length (h : 3 = (b * 3) / 2) : b = 2 :=
by
  sorry

end triangle_base_length_l290_290620


namespace quiz_prob_distrib_l290_290428

theorem quiz_prob_distrib
  (num_eco_choice : ℕ)
  (total_eco : ℕ)
  (num_smart_choice : ℕ)
  (total_smart : ℕ)
  (ξ : ℕ → ℕ)
  (n : ℕ) :
  num_eco_choice = 3 →
  total_eco = 4 →
  num_smart_choice = 2 →
  total_smart = 2 →
  ξ 0 + ξ 1 + ξ 2 = n →
  ξ 1 / n = 3 / 5 ∧
  (ξ 0 / n = 1 / 5 ∧ ξ 1 / n = 3 / 5 ∧ ξ 2 / n = 1 / 5) ∧
  (ξ 0 = 0 → ξ 1 = ξ (1 : ℕ) → ξ 2 * 1 / n + ξ 1 * 1 / n + ξ 0 * 1 / n = 1) :=
begin
  sorry,
end

end quiz_prob_distrib_l290_290428


namespace vector_subtraction_l290_290569

-- Define the vectors a and b
def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-3, -4)

-- Statement we want to prove: 2a - b = (-1, 6)
theorem vector_subtraction : 2 • a - b = (-1, 6) := by
  sorry

end vector_subtraction_l290_290569


namespace monotonic_increasing_iff_l290_290589

open Real

noncomputable def f (x a : ℝ) : ℝ := abs ((exp x) / 2 - a / (exp x))

theorem monotonic_increasing_iff (a : ℝ) : 
  (∀ x ∈ set.Icc 1 2, ∀ y ∈ set.Icc 1 2, x ≤ y → f x a ≤ f y a) ↔ (- (exp 2)^2 / 2 ≤ a ∧ a ≤ (exp 2) / 2) := 
by 
  sorry

end monotonic_increasing_iff_l290_290589


namespace total_ants_correct_l290_290227

def abe_ants : ℕ := 4
def beth_ants : ℕ := abe_ants + (abe_ants / 2)
def cece_ants : ℕ := 2 * abe_ants
def duke_ants : ℕ := abe_ants / 2
def total_ants : ℕ := abe_ants + beth_ants + cece_ants + duke_ants

theorem total_ants_correct : total_ants = 20 :=
by
  sorry

end total_ants_correct_l290_290227


namespace complete_the_square_l290_290321

theorem complete_the_square (d e f : ℤ) (h1 : 0 < d)
    (h2 : ∀ x : ℝ, 100 * x^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f) :
  d + e + f = 112 := by
  sorry

end complete_the_square_l290_290321


namespace find_f2_l290_290118

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_f2 (f : ℝ → ℝ)
  (H1 : ∀ x y : ℝ, f (x + y) = f x + f y + 1)
  (H2 : f 8 = 15) :
  f 2 = 3 := 
sorry

end find_f2_l290_290118


namespace p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l290_290260

-- Define conditions
def p (x : ℝ) : Prop := -x^2 + 2 * x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2 * x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8 * x + 20 ≥ 0

variable {x m : ℝ}

-- Question 1
theorem p_sufficient_not_necessary_for_q (hp : ∀ x, p x → q x m) : m ≥ 3 :=
sorry

-- Defining negation of s and q
def neg_s (x : ℝ) : Prop := ¬s x
def neg_q (x m : ℝ) : Prop := ¬q x m

-- Question 2
theorem neg_s_sufficient_not_necessary_for_neg_q (hp : ∀ x, neg_s x → neg_q x m) : false :=
sorry

end p_sufficient_not_necessary_for_q_neg_s_sufficient_not_necessary_for_neg_q_l290_290260


namespace symmetric_poly_roots_identity_l290_290274

variable (a b c : ℝ)

theorem symmetric_poly_roots_identity (h1 : a + b + c = 6) (h2 : ab + bc + ca = 5) (h3 : abc = 1) :
  (1 / a^3) + (1 / b^3) + (1 / c^3) = 38 :=
by
  sorry

end symmetric_poly_roots_identity_l290_290274


namespace exists_n_for_pow_lt_e_l290_290332

theorem exists_n_for_pow_lt_e {p e : ℝ} (hp : 0 < p ∧ p < 1) (he : 0 < e) :
  ∃ n : ℕ, (1 - p) ^ n < e :=
sorry

end exists_n_for_pow_lt_e_l290_290332


namespace value_of_A_l290_290471

theorem value_of_A (M A T E H : ℤ) (hH : H = 8) (h1 : M + A + T + H = 31) (h2 : T + E + A + M = 40) (h3 : M + E + E + T = 44) (h4 : M + A + T + E = 39) : A = 12 :=
by
  sorry

end value_of_A_l290_290471


namespace min_value_of_reciprocal_sum_l290_290919

variable (m n : ℝ)
variable (a : ℝ × ℝ := (m, 1))
variable (b : ℝ × ℝ := (4 - n, 2))

theorem min_value_of_reciprocal_sum
  (h1 : m > 0) (h2 : n > 0)
  (h3 : a.1 * b.2 = a.2 * b.1) :
  (1/m + 8/n) = 9/2 :=
sorry

end min_value_of_reciprocal_sum_l290_290919


namespace pipe_B_fill_time_l290_290188

theorem pipe_B_fill_time (t : ℝ) :
  (1/10) + (2/t) - (2/15) = 1 ↔ t = 60/31 :=
by
  sorry

end pipe_B_fill_time_l290_290188


namespace solve_for_y_l290_290500

variable (x y z : ℝ)

theorem solve_for_y (h : 3 * x + 3 * y + 3 * z + 11 = 143) : y = 44 - x - z :=
by 
  sorry

end solve_for_y_l290_290500


namespace solve_prob_problem_l290_290064

open ProbabilityTheory

noncomputable def prob_problem (Ω : Type*) [MeasureSpace Ω] : Prop :=
  let rolls : Ω → ℕ × ℕ := sorry
  let is_event (p : (ℕ × ℕ) → Prop) : Event Ω := sorry
  let A : Event Ω := is_event (λ (x : ℕ × ℕ), x.1 + x.2 = 4)
  let B : Event Ω := is_event (λ (x : ℕ × ℕ), x.2 % 2 = 0)
  let C : Event Ω := is_event (λ (x : ℕ × ℕ), x.1 = x.2)
  let D : Event Ω := is_event (λ (x : ℕ × ℕ), (x.1 % 2 ≠ 0) ∨ (x.2 % 2 ≠ 0))
  (probability[D] = 3/4) ∧ (probability[B ⊓ D] = 1/4) ∧ (independent B C)

theorem solve_prob_problem : prob_problem := sorry

end solve_prob_problem_l290_290064


namespace carlos_payment_l290_290309

theorem carlos_payment (A B C : ℝ) (hB_lt_A : B < A) (hB_lt_C : B < C) :
    B + (0.35 * (A + B + C) - B) = 0.35 * A - 0.65 * B + 0.35 * C :=
by sorry

end carlos_payment_l290_290309


namespace union_sets_eq_real_l290_290567

def A : Set ℝ := {x | x ≥ 0}
def B : Set ℝ := {x | x < 1}

theorem union_sets_eq_real : A ∪ B = Set.univ :=
by
  sorry

end union_sets_eq_real_l290_290567


namespace even_divisors_less_than_100_l290_290401

theorem even_divisors_less_than_100 :
  let count_even_divisors := 
    let n := 100 in
    let perfect_squares := { m | ∃ k, k * k = m ∧ m < n } in
    let total_numbers := finset.range n in
    (total_numbers.card - perfect_squares.card) =
    90 
  in count_even_divisors = 90 :=
by
  let n := 100
  let perfect_squares : finset ℕ := finset.filter (λ m, ∃ k, k * k = m) (finset.range n)
  let total_numbers : finset ℕ := finset.range n
  have h : total_numbers.card = 99 := by sorry
  have p : perfect_squares.card = 9 := by sorry
  show total_numbers.card - perfect_squares.card = 90
  calc
    total_numbers.card - perfect_squares.card
      = 99 - 9 := by rw [h, p]
      = 90 := by norm_num

end even_divisors_less_than_100_l290_290401


namespace balls_sold_eq_13_l290_290752

-- Let SP be the selling price, CP be the cost price per ball, and loss be the loss incurred.
def SP : ℕ := 720
def CP : ℕ := 90
def loss : ℕ := 5 * CP
def total_CP (n : ℕ) : ℕ := n * CP

-- Given the conditions:
axiom loss_eq : loss = 5 * CP
axiom ball_CP_value : CP = 90
axiom selling_price_value : SP = 720

-- Loss is defined as total cost price minus selling price
def calculated_loss (n : ℕ) : ℕ := total_CP n - SP

-- The proof statement:
theorem balls_sold_eq_13 (n : ℕ) (h1 : calculated_loss n = loss) : n = 13 :=
by sorry

end balls_sold_eq_13_l290_290752


namespace numbers_product_l290_290634

theorem numbers_product (x y : ℝ) (h1 : x + y = 24) (h2 : x - y = 8) : x * y = 128 := by
  sorry

end numbers_product_l290_290634


namespace find_y_value_l290_290543

theorem find_y_value (y : ℝ) (h : 12^2 * y^3 / 432 = 72) : y = 6 :=
by
  sorry

end find_y_value_l290_290543


namespace largest_is_B_l290_290495

noncomputable def A := Real.sqrt (Real.sqrt (56 ^ (1 / 3)))
noncomputable def B := Real.sqrt (Real.sqrt (3584 ^ (1 / 3)))
noncomputable def C := Real.sqrt (Real.sqrt (2744 ^ (1 / 3)))
noncomputable def D := Real.sqrt (Real.sqrt (392 ^ (1 / 3)))
noncomputable def E := Real.sqrt (Real.sqrt (448 ^ (1 / 3)))

theorem largest_is_B : B > A ∧ B > C ∧ B > D ∧ B > E := by
  sorry

end largest_is_B_l290_290495


namespace part1_part2_l290_290003

def A (t : ℝ) : Prop :=
  ∀ x : ℝ, (t+2)*x^2 + 2*x + 1 > 0

def B (a x : ℝ) : Prop :=
  (a*x - 1)*(x + a) > 0

theorem part1 (t : ℝ) : A t ↔ t < -1 :=
sorry

theorem part2 (a : ℝ) : (∀ t : ℝ, t < -1 → ∀ x : ℝ, B a x) → (0 ≤ a ∧ a ≤ 1) :=
sorry

end part1_part2_l290_290003


namespace vasya_numbers_l290_290822

theorem vasya_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) : (x = 1 / 2 ∧ y = -1) ∨ (x = -1 ∧ y = 1 / 2) :=
by sorry

end vasya_numbers_l290_290822


namespace greatest_integer_less_than_PS_l290_290154

noncomputable def PS := (150 * Real.sqrt 2)

theorem greatest_integer_less_than_PS
  (PQ RS : ℝ)
  (PS : ℝ := PQ * Real.sqrt 2)
  (h₁ : PQ = 150)
  (h_midpoint : PS / 2 = PQ) :
  ∀ n : ℤ, n < PS → n = 212 :=
by
  -- Proof to be completed later
  sorry

end greatest_integer_less_than_PS_l290_290154


namespace angle_bisector_correct_length_l290_290730

-- Define the isosceles triangle with the given conditions
structure IsoscelesTriangle :=
  (base : ℝ)
  (lateral : ℝ)
  (is_isosceles : lateral = 20 ∧ base = 5)

-- Define the problem of finding the angle bisector
noncomputable def angle_bisector_length (tri : IsoscelesTriangle) : ℝ :=
  6

-- The main theorem to state the problem
theorem angle_bisector_correct_length (tri : IsoscelesTriangle) : 
  angle_bisector_length tri = 6 :=
by
  -- We state the theorem, skipping the proof (sorry)
  sorry

end angle_bisector_correct_length_l290_290730


namespace equal_ivan_petrovich_and_peter_ivanovich_l290_290600

theorem equal_ivan_petrovich_and_peter_ivanovich :
  (∀ n : ℕ, n % 10 = 0 → (n % 20 = 0) = (n % 200 = 0)) :=
by
  sorry

end equal_ivan_petrovich_and_peter_ivanovich_l290_290600


namespace exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l290_290733

-- Part (a): Proving the existence of such an arithmetic sequence with 2003 terms.
theorem exists_arithmetic_seq_2003_terms_perfect_powers :
  ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, n ≤ 2002 → ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

-- Part (b): Proving the non-existence of such an infinite arithmetic sequence.
theorem no_infinite_arithmetic_seq_perfect_powers :
  ¬ ∃ (a : ℕ) (d : ℕ), ∀ n : ℕ, ∃ (k m : ℕ), m > 1 ∧ a + n * d = k ^ m :=
by
  sorry

end exists_arithmetic_seq_2003_terms_perfect_powers_no_infinite_arithmetic_seq_perfect_powers_l290_290733


namespace min_red_beads_l290_290215

-- Define the structure of the necklace and the conditions
structure Necklace where
  total_beads : ℕ
  blue_beads : ℕ
  red_beads : ℕ
  cyclic : Bool
  condition : ∀ (segment : List ℕ), segment.length = 8 → segment.count blue_beads ≥ 12 → segment.count red_beads ≥ 4

-- The given problem condition
def given_necklace : Necklace :=
  { total_beads := 50,
    blue_beads := 50,
    red_beads := 0,
    cyclic := true,
    condition := sorry }

-- The proof problem: Minimum number of red beads required
theorem min_red_beads (n : Necklace) : n.red_beads ≥ 29 :=
by { sorry }

end min_red_beads_l290_290215


namespace find_c_l290_290577

theorem find_c (a b c : ℝ) (h : 1/a + 1/b = 1/c) : c = (a * b) / (a + b) := 
by
  sorry

end find_c_l290_290577


namespace hn_passes_fixed_point_l290_290124

-- Define the conditions
def ellipse_center_at_origin (x y : ℝ) : Prop := 
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ((x^2 / a) + (y^2 / b) = 1)

def passes_through_A_and_B (x y : ℝ) : Prop := 
  ∃ a b : ℝ, (a > 0 ∧ b > 0) ∧ 
  ((a * 0^2 + b * (-2)^2 = 1) ∧ 
   (a * (3/2)^2 + b * (-1)^2 = 1))

def equation_of_ellipse (x y : ℝ) : Prop := 
  (x^2 / 3) + (y^2 / 4) = 1

-- Define the fixed point condition
def fixed_point_HN (x y : ℝ) : Prop := 
  (0, -2)

-- The statement to be proven
theorem hn_passes_fixed_point : 
  ∀ (x y : ℝ), ellipse_center_at_origin x y ∧ passes_through_A_and_B x y ∧ equation_of_ellipse x y → fixed_point_HN (0, -2) :=
by
  intro x y h,
  sorry

end hn_passes_fixed_point_l290_290124


namespace tic_tac_toe_tie_fraction_l290_290189

theorem tic_tac_toe_tie_fraction :
  let amys_win : ℚ := 5 / 12
  let lilys_win : ℚ := 1 / 4
  1 - (amys_win + lilys_win) = 1 / 3 :=
by
  sorry

end tic_tac_toe_tie_fraction_l290_290189


namespace record_cost_calculation_l290_290974

theorem record_cost_calculation :
  ∀ (books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost : ℕ),
  books_owned = 200 →
  book_price = 3 / 2 →
  records_bought = 75 →
  money_left = 75 →
  total_selling_price = books_owned * book_price →
  money_spent_per_record = total_selling_price - money_left →
  record_cost = money_spent_per_record / records_bought →
  record_cost = 3 :=
by
  intros books_owned book_price records_bought money_left total_selling_price money_spent_per_record record_cost
  sorry

end record_cost_calculation_l290_290974


namespace sales_decrease_percentage_l290_290042

theorem sales_decrease_percentage 
  (P S : ℝ) 
  (P_new : ℝ := 1.30 * P) 
  (R : ℝ := P * S) 
  (R_new : ℝ := 1.04 * R) 
  (x : ℝ) 
  (S_new : ℝ := S * (1 - x/100)) 
  (h1 : 1.30 * P * S * (1 - x/100) = 1.04 * P * S) : 
  x = 20 :=
by
  sorry

end sales_decrease_percentage_l290_290042


namespace no_n_in_range_l290_290377

def g (n : ℕ) : ℕ := 7 + 4 * n + 6 * n ^ 2 + 3 * n ^ 3 + 4 * n ^ 4 + 3 * n ^ 5

theorem no_n_in_range
  : ¬ ∃ n : ℕ, 2 ≤ n ∧ n ≤ 100 ∧ g n % 11 = 0 := sorry

end no_n_in_range_l290_290377


namespace candidates_appeared_l290_290430

-- Define the conditions:
variables (A_selected B_selected : ℕ) (x : ℝ)

-- 12% candidates got selected in State A
def State_A_selected := 0.12 * x

-- 18% candidates got selected in State B
def State_B_selected := 0.18 * x

-- 250 more candidates got selected in State B than in State A
def selection_difference := State_B_selected = State_A_selected + 250

-- The statement to prove:
theorem candidates_appeared (h : selection_difference) : x = 4167 :=
by
  sorry

end candidates_appeared_l290_290430


namespace market_value_decrease_l290_290472

noncomputable def percentage_decrease_each_year : ℝ :=
  let original_value := 8000
  let value_after_two_years := 3200
  let p := 1 - (value_after_two_years / original_value)^(1 / 2)
  p * 100

theorem market_value_decrease :
  let p := percentage_decrease_each_year
  abs (p - 36.75) < 0.01 :=
by
  sorry

end market_value_decrease_l290_290472


namespace min_period_and_sym_center_l290_290136

open Real

noncomputable def func (x α β : ℝ) : ℝ :=
  sin (x - α) * cos (x - β)

theorem min_period_and_sym_center (α β : ℝ) :
  (∀ x, func (x + π) α β = func x α β) ∧ (func α 0 β = 0) :=
by
  sorry

end min_period_and_sym_center_l290_290136


namespace distance_triangle_four_points_l290_290121

variable {X : Type*} [MetricSpace X]

theorem distance_triangle_four_points (A B C D : X) :
  dist A D ≤ dist A B + dist B C + dist C D :=
by
  sorry

end distance_triangle_four_points_l290_290121


namespace area_and_cost_of_path_l290_290337

-- Define the dimensions of the rectangular grass field
def length_field : ℝ := 75
def width_field : ℝ := 55

-- Define the width of the path around the field
def path_width : ℝ := 2.8

-- Define the cost per square meter for constructing the path
def cost_per_sq_m : ℝ := 2

-- Define the total length and width including the path
def total_length : ℝ := length_field + 2 * path_width
def total_width : ℝ := width_field + 2 * path_width

-- Define the area of the entire field including the path
def area_total : ℝ := total_length * total_width

-- Define the area of the grass field alone
def area_field : ℝ := length_field * width_field

-- Define the area of the path alone
def area_path : ℝ := area_total - area_field

-- Define the cost of constructing the path
def cost_path : ℝ := area_path * cost_per_sq_m

-- The statement to be proved
theorem area_and_cost_of_path :
  area_path = 759.36 ∧ cost_path = 1518.72 := by
  sorry

end area_and_cost_of_path_l290_290337


namespace ratio_of_abc_l290_290548

theorem ratio_of_abc (a b c : ℝ) (h1 : a ≠ 0) (h2 : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : a / b = 1 / 2 ∧ a / c = 1 / 3 := 
sorry

end ratio_of_abc_l290_290548


namespace equal_constants_l290_290265

theorem equal_constants (a b : ℝ) :
  (∃ᶠ n in at_top, ⌊a * n + b⌋ ≥ ⌊a + b * n⌋) →
  (∃ᶠ m in at_top, ⌊a + b * m⌋ ≥ ⌊a * m + b⌋) →
  a = b :=
by
  sorry

end equal_constants_l290_290265


namespace ellipse_equation_proof_HN_fixed_point_l290_290125

-- Definitions of conditions
def ellipse_condition (x y m n : ℝ) : Prop :=
  m * x^2 + n * y^2 = 1

def point_A_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition 0 (-2) m n

def point_B_on_ellipse (m n : ℝ) : Prop :=
  ellipse_condition (3 / 2) (-1) m n

def equation_of_ellipse (x y : ℝ) : Prop :=
  (x^2 / 3) + (y^2 / 4) = 1

-- The proof statements
theorem ellipse_equation_proof (m n : ℝ) 
  (h₁ : point_A_on_ellipse m n)
  (h₂ : point_B_on_ellipse m n) :
  equation_of_ellipse :=
  sorry

theorem HN_fixed_point (P M N T H K : ℝ × ℝ)
  (hp : P = (1, -2))
  (hmn : ∃ M N, is_intersection P M N)
  (ht : ∃ T, is_intersection_M_par_AB_T M T)
  (hh : is_mid_point M T H)
  (hk : fixed_point H N K) :
  K = (0, -2) :=
  sorry

-- Helper definitions for intersection, mid point, and fixed point
def is_intersection (P M N : ℝ × ℝ) : Prop :=
  ∃ m : ℝ, m = ...

def is_intersection_M_par_AB_T (M T : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t = ...

def is_mid_point (M T H : ℝ × ℝ) : Prop :=
  ∃ h : ℝ, h = ...

def fixed_point (H N K : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, k = ...

end ellipse_equation_proof_HN_fixed_point_l290_290125


namespace range_of_m_l290_290879

noncomputable def set_M (m : ℝ) : Set ℝ := {x | x < m}
noncomputable def set_N : Set ℝ := {y | ∃ (x : ℝ), y = Real.log x / Real.log 2 - 1 ∧ 4 ≤ x}

theorem range_of_m (m : ℝ) : set_M m ∩ set_N = ∅ → m < 1 
:= by
  sorry

end range_of_m_l290_290879


namespace daily_pre_promotion_hours_l290_290236

-- Defining conditions
def weekly_additional_hours := 6
def hours_driven_in_two_weeks_after_promotion := 40
def days_in_two_weeks := 14
def hours_added_in_two_weeks := 2 * weekly_additional_hours

-- Math proof problem statement
theorem daily_pre_promotion_hours :
  (hours_driven_in_two_weeks_after_promotion - hours_added_in_two_weeks) / days_in_two_weeks = 2 :=
by
  sorry

end daily_pre_promotion_hours_l290_290236


namespace sequence_problem_l290_290017

theorem sequence_problem 
  (a : ℕ → ℕ) 
  (h1 : a 1 = 5) 
  (h2 : ∀ n : ℕ, a (n + 1) - a n = 3 + 4 * (n - 1)) : 
  a 50 = 4856 :=
sorry

end sequence_problem_l290_290017


namespace remainder_19_pow_19_plus_19_mod_20_l290_290334

theorem remainder_19_pow_19_plus_19_mod_20 : (19 ^ 19 + 19) % 20 = 18 := 
by {
  sorry
}

end remainder_19_pow_19_plus_19_mod_20_l290_290334


namespace coordinates_of_B_l290_290415

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end coordinates_of_B_l290_290415


namespace t_plus_reciprocal_l290_290903

theorem t_plus_reciprocal (t : ℝ) (h : t^2 - 3 * t + 1 = 0) (ht : t ≠ 0) : t + 1/t = 3 :=
by sorry

end t_plus_reciprocal_l290_290903


namespace average_words_per_hour_l290_290521

/-- Prove that given a total of 50,000 words written in 100 hours with the 
writing output increasing by 10% each subsequent hour, the average number 
of words written per hour is 500. -/
theorem average_words_per_hour 
(words_total : ℕ) 
(hours_total : ℕ) 
(increase : ℝ) :
  words_total = 50000 ∧ hours_total = 100 ∧ increase = 0.1 →
  (words_total / hours_total : ℝ) = 500 :=
by 
  intros h
  sorry

end average_words_per_hour_l290_290521


namespace vasya_numbers_l290_290805

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290805


namespace triangular_array_sum_digits_l290_290496

theorem triangular_array_sum_digits (N : ℕ) (h : N * (N + 1) / 2 = 2145) : (N / 10 + N % 10) = 11 := 
sorry

end triangular_array_sum_digits_l290_290496


namespace expected_value_unfair_die_correct_l290_290750

noncomputable def expected_value_unfair_die : ℚ :=
  (2 / 15) * (1 + 2 + 3 + 4 + 5 + 6 + 7) + (1 / 3) * 8

theorem expected_value_unfair_die_correct :
  expected_value_unfair_die = 6.4 :=
by
  rw [expected_value_unfair_die]
  have h1 : (2 / 15 : ℚ) * 28 = 56 / 15 := by norm_num
  have h2 : (1 / 3 : ℚ) * 8 = 8 / 3 := by norm_num
  have h3 : 56 / 15 + 8 / 3 = 96 / 15 := by norm_num
  have h4 : 96 / 15 = 32 / 5 := by norm_num
  have h5 : 32 / 5 = 6.4 := by norm_num
  exact Eq.trans (Eq.trans (Eq.trans (Eq.trans h1 h2) h3) h4) h5

end expected_value_unfair_die_correct_l290_290750


namespace find_phi_l290_290715

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x + Real.pi / 4)

theorem find_phi (phi : ℝ) (h_shift : ∀ x : ℝ, f (x + phi) = f (-x - phi)) : 
  phi = Real.pi / 8 :=
  sorry

end find_phi_l290_290715


namespace a_and_b_together_30_days_l290_290007

variable (R_a R_b : ℝ)

-- Conditions
axiom condition1 : R_a = 3 * R_b
axiom condition2 : R_a * 40 = (R_a + R_b) * 30

-- Question: prove that a and b together can complete the work in 30 days.
theorem a_and_b_together_30_days (R_a R_b : ℝ) (condition1 : R_a = 3 * R_b) (condition2 : R_a * 40 = (R_a + R_b) * 30) : true :=
by
  sorry

end a_and_b_together_30_days_l290_290007


namespace perimeter_triangle_ABI_l290_290731

universe u

variables {AC BC AB CD AD BD r x : ℝ}
variables {A B C D I : Type u}

theorem perimeter_triangle_ABI (h1 : AC = 5) 
                              (h2 : BC = 12)
                              (h3 : AB = 13) 
                              (h4 : CD = real.sqrt (5 * 12)) : 
                              x ≥ real.sqrt 15 → 
                              x = real.sqrt 15 → 
                              let P := AB + 2 * x in 
                              P = 13 + 2 * x := 
by
  sorry

end perimeter_triangle_ABI_l290_290731


namespace find_c_share_l290_290498

theorem find_c_share (a b c : ℕ) 
  (h1 : a + b + c = 1760)
  (h2 : ∃ x : ℕ, a = 3 * x ∧ b = 4 * x ∧ c = 5 * x)
  (h3 : 6 * a = 8 * b ∧ 8 * b = 20 * c) : 
  c = 250 :=
by
  sorry

end find_c_share_l290_290498


namespace sequence_periodic_from_some_term_l290_290547

def is_bounded (s : ℕ → ℤ) (M : ℤ) : Prop :=
  ∀ n, |s n| ≤ M

def is_periodic_from (s : ℕ → ℤ) (N : ℕ) (p : ℕ) : Prop :=
  ∀ n, s (N + n) = s (N + n + p)

theorem sequence_periodic_from_some_term (s : ℕ → ℤ) (M : ℤ) (h_bounded : is_bounded s M)
    (h_recurrence : ∀ n, s (n + 5) = (5 * s (n + 4) ^ 3 + s (n + 3) - 3 * s (n + 2) + s n) / (2 * s (n + 2) + s (n + 1) ^ 2 + s (n + 1) * s n)) :
    ∃ N p, is_periodic_from s N p := by
  sorry

end sequence_periodic_from_some_term_l290_290547


namespace boxes_per_day_l290_290943

theorem boxes_per_day (apples_per_box fewer_apples_per_day total_apples_two_weeks : ℕ)
  (h1 : apples_per_box = 40)
  (h2 : fewer_apples_per_day = 500)
  (h3 : total_apples_two_weeks = 24500) :
  (∃ x : ℕ, (7 * apples_per_box * x + 7 * (apples_per_box * x - fewer_apples_per_day) = total_apples_two_weeks) ∧ x = 50) := 
sorry

end boxes_per_day_l290_290943


namespace man_present_age_l290_290845

variable {P : ℝ}

theorem man_present_age (h1 : P = 1.25 * (P - 10)) (h2 : P = (5 / 6) * (P + 10)) : P = 50 :=
  sorry

end man_present_age_l290_290845


namespace negation_of_existence_implies_universal_l290_290772

theorem negation_of_existence_implies_universal :
  ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_of_existence_implies_universal_l290_290772


namespace imaginary_part_of_z_l290_290581

theorem imaginary_part_of_z (z : ℂ) (h : z * (1 + I) = 2 - 2 * I) : z.im = -2 :=
sorry

end imaginary_part_of_z_l290_290581


namespace quadratic_roots_pair_l290_290957

theorem quadratic_roots_pair (c d : ℝ) (h₀ : c ≠ 0) (h₁ : d ≠ 0) 
    (h₂ : ∀ x : ℝ, x^2 + c * x + d = 0 ↔ x = 2 * c ∨ x = 3 * d) : 
    (c, d) = (1 / 6, -1 / 6) := 
  sorry

end quadratic_roots_pair_l290_290957


namespace evaluate_g_l290_290107

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem evaluate_g : 3 * g 2 + 2 * g (-2) = 98 :=
by
  sorry

end evaluate_g_l290_290107


namespace rectangle_percentage_increase_l290_290966

theorem rectangle_percentage_increase (L W : ℝ) (P : ℝ) (h : (1 + P / 100) ^ 2 = 1.44) : P = 20 :=
by {
  -- skipped proof
  sorry
}

end rectangle_percentage_increase_l290_290966


namespace largest_base_conversion_l290_290981

theorem largest_base_conversion :
  let a := (3: ℕ)
  let b := (1 * 2^1 + 1 * 2^0: ℕ)
  let c := (3 * 8^0: ℕ)
  let d := (1 * 3^1 + 1 * 3^0: ℕ)
  max a (max b (max c d)) = d :=
by
  sorry

end largest_base_conversion_l290_290981


namespace range_of_k_l290_290725

theorem range_of_k (k : ℝ) :
  (∀ x : ℝ, (k^2 - 1) * x^2 - (k + 1) * x + 1 > 0) ↔ (1 ≤ k ∧ k ≤ 5 / 3) := 
sorry

end range_of_k_l290_290725


namespace ellipse_ratio_sum_l290_290242

theorem ellipse_ratio_sum :
  (∃ x y : ℝ, 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0) →
  (∃ a b : ℝ, (∀ (x y : ℝ), 3 * x^2 + 2 * x * y + 4 * y^2 - 20 * x - 30 * y + 60 = 0 → 
    (y = a * x ∨ y = b * x)) ∧ (a + b = 9)) :=
  sorry

end ellipse_ratio_sum_l290_290242


namespace vasya_numbers_l290_290803

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l290_290803


namespace interest_rate_l290_290093

theorem interest_rate (SI P : ℝ) (T : ℕ) (h₁: SI = 70) (h₂ : P = 700) (h₃ : T = 4) : 
  (SI / (P * T)) * 100 = 2.5 :=
by
  sorry

end interest_rate_l290_290093


namespace area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l290_290382

noncomputable def area_of_triangle (b h : ℝ) : ℝ :=
  (1 / 2) * b * h

noncomputable def area_trapezoid (b1 b2 h : ℝ) : ℝ :=
  (1 / 2) * (b1 + b2) * h

theorem area_triangle_ACD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 →
  area_of_triangle C 20 = 100 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

theorem area_trapezoid_ABCD_proof :
  ∀ (A B C D X Y : ℝ), 
  A = 24 → 
  C = 10 → 
  X = 6 → 
  Y = 8 → 
  B = 23 → 
  D = 27 → 
  area_trapezoid 24 10 24 = 260 :=
by
  intros A B C D X Y hAB hCD hAX hXY hXX1 hYY1
  sorry

end area_triangle_ACD_proof_area_trapezoid_ABCD_proof_l290_290382


namespace find_initial_cards_l290_290874

theorem find_initial_cards (B : ℕ) :
  let Tim_initial := 20
  let Sarah_initial := 15
  let Tim_after_give_to_Sarah := Tim_initial - 5
  let Sarah_after_give_to_Sarah := Sarah_initial + 5
  let Tim_after_receive_from_Sarah := Tim_after_give_to_Sarah + 2
  let Sarah_after_receive_from_Sarah := Sarah_after_give_to_Sarah - 2
  let Tim_after_exchange_with_Ben := Tim_after_receive_from_Sarah - 3
  let Ben_after_exchange := B + 13
  let Ben_after_all_transactions := 3 * Tim_after_exchange_with_Ben
  Ben_after_exchange = Ben_after_all_transactions -> B = 29 := by
  sorry

end find_initial_cards_l290_290874


namespace taxi_fare_l290_290183

theorem taxi_fare (x : ℝ) : 
  (2.40 + 2 * (x - 0.5) = 8) → x = 3.3 := by
  sorry

end taxi_fare_l290_290183


namespace shaded_area_of_three_circles_l290_290442

theorem shaded_area_of_three_circles :
  (∀ (r1 r2 : ℝ), (π * r1^2 = 100 * π) → (r2 = r1 / 2) → (shaded_area = (π * r1^2) / 2 + 2 * ((π * r2^2) / 2)) → (shaded_area = 75 * π)) :=
by
  sorry

end shaded_area_of_three_circles_l290_290442


namespace amount_paid_is_correct_l290_290444

-- Conditions given in the problem
def jimmy_shorts_count : ℕ := 3
def jimmy_short_price : ℝ := 15.0
def irene_shirts_count : ℕ := 5
def irene_shirt_price : ℝ := 17.0
def discount_rate : ℝ := 0.10

-- Define the total cost for jimmy
def jimmy_total_cost : ℝ := jimmy_shorts_count * jimmy_short_price

-- Define the total cost for irene
def irene_total_cost : ℝ := irene_shirts_count * irene_shirt_price

-- Define the total cost before discount
def total_cost_before_discount : ℝ := jimmy_total_cost + irene_total_cost

-- Define the discount amount
def discount_amount : ℝ := total_cost_before_discount * discount_rate

-- Define the total amount to pay
def total_amount_to_pay : ℝ := total_cost_before_discount - discount_amount

-- The proposition we need to prove
theorem amount_paid_is_correct : total_amount_to_pay = 117 := by
  sorry

end amount_paid_is_correct_l290_290444


namespace decreasing_intervals_tangent_line_eq_l290_290565

-- Define the function f and its derivative.
def f (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + 1
def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

-- Part 1: Prove intervals of monotonic decreasing.
theorem decreasing_intervals :
  (∀ x, f' x < 0 → x < -1 ∨ x > 3) := 
sorry

-- Part 2: Prove the tangent line equation.
theorem tangent_line_eq :
  15 * (-2) + (-13) + 27 = 0 :=
sorry

end decreasing_intervals_tangent_line_eq_l290_290565


namespace Paco_cookies_left_l290_290754

/-
Problem: Paco had 36 cookies. He gave 14 cookies to his friend and ate 10 cookies. How many cookies did Paco have left?
Solution: Paco has 12 cookies left.

To formally state this in Lean:
-/

def initial_cookies := 36
def cookies_given_away := 14
def cookies_eaten := 10

theorem Paco_cookies_left : initial_cookies - (cookies_given_away + cookies_eaten) = 12 :=
by
  sorry

/-
This theorem states that Paco has 12 cookies left given initial conditions.
-/

end Paco_cookies_left_l290_290754


namespace third_derivative_y_l290_290687

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.sin (5 * x - 3)

theorem third_derivative_y (x : ℝ) : 
  (deriv^[3] y x) = -150 * x * Real.sin (5 * x - 3) + (30 - 125 * x^2) * Real.cos (5 * x - 3) :=
by
  sorry

end third_derivative_y_l290_290687


namespace coordinates_of_A_in_second_quadrant_l290_290300

noncomputable def coordinates_A (m : ℤ) : ℤ × ℤ :=
  (7 - 2 * m, 5 - m)

theorem coordinates_of_A_in_second_quadrant (m : ℤ) (h1 : 7 - 2 * m < 0) (h2 : 5 - m > 0) :
  coordinates_A m = (-1, 1) := 
sorry

end coordinates_of_A_in_second_quadrant_l290_290300


namespace five_star_three_eq_ten_l290_290244

def operation (a b : ℝ) : ℝ := b^2 + 1

theorem five_star_three_eq_ten : operation 5 3 = 10 := by
  sorry

end five_star_three_eq_ten_l290_290244


namespace width_of_domain_of_g_l290_290289

variable (h : ℝ → ℝ) (dom_h : ∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x)

noncomputable def g (x : ℝ) : ℝ := h (x / 3)

theorem width_of_domain_of_g :
  (∀ x, -10 ≤ x ∧ x ≤ 10 → h x = h x) →
  (∀ y : ℝ, -30 ≤ y ∧ y ≤ 30 → h (y / 3) = h (y / 3)) →
  (∃ a b : ℝ, a = -30 ∧ b = 30 ∧  (∃ w : ℝ, w = b - a ∧ w = 60)) :=
by
  sorry

end width_of_domain_of_g_l290_290289


namespace remainder_when_divided_by_x_plus_2_l290_290710

-- Define the polynomial q(x) = D*x^4 + E*x^2 + F*x + 8
variable (D E F : ℝ)
def q (x : ℝ) : ℝ := D * x^4 + E * x^2 + F * x + 8

-- Given condition: q(2) = 12
axiom h1 : q D E F 2 = 12

-- Prove that q(-2) = 4
theorem remainder_when_divided_by_x_plus_2 : q D E F (-2) = 4 := by
  sorry

end remainder_when_divided_by_x_plus_2_l290_290710


namespace number_of_pies_is_correct_l290_290613

def weight_of_apples : ℕ := 120
def weight_for_applesauce (w : ℕ) : ℕ := w / 2
def weight_for_pies (w wholly_app : ℕ) : ℕ := w - wholly_app
def pies (weight_per_pie total_weight : ℕ) : ℕ := total_weight / weight_per_pie

theorem number_of_pies_is_correct :
  pies 4 (weight_for_pies weight_of_apples (weight_for_applesauce weight_of_apples)) = 15 :=
by
  sorry

end number_of_pies_is_correct_l290_290613


namespace remainder_of_sum_l290_290722

theorem remainder_of_sum (n : ℤ) : ((5 - n) + (n + 4)) % 5 = 4 := 
by 
  -- proof goes here
  sorry

end remainder_of_sum_l290_290722


namespace students_answered_both_correctly_l290_290497

theorem students_answered_both_correctly 
  (total_students : ℕ) (took_test : ℕ) 
  (q1_correct : ℕ) (q2_correct : ℕ)
  (did_not_take_test : ℕ)
  (h1 : total_students = 25)
  (h2 : q1_correct = 22)
  (h3 : q2_correct = 20)
  (h4 : did_not_take_test = 3)
  (h5 : took_test = total_students - did_not_take_test) :
  (q1_correct + q2_correct) - took_test = 20 := 
by 
  -- Proof skipped.
  sorry

end students_answered_both_correctly_l290_290497


namespace sum_of_final_two_numbers_l290_290643

theorem sum_of_final_two_numbers (x y T : ℕ) (h : x + y = T) :
  3 * (x + 5) + 3 * (y + 5) = 3 * T + 30 :=
by
  sorry

end sum_of_final_two_numbers_l290_290643


namespace part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l290_290037

-- Part 1: Prove existence of rectangle B with sides 2 + sqrt(2)/2 and 2 - sqrt(2)/2
theorem part1_exists_rectangle_B : 
  ∃ (x y : ℝ), (x + y = 4) ∧ (x * y = 7 / 2) :=
by
  sorry

-- Part 2: Prove non-existence of rectangle B for given sides of the known rectangle
theorem part2_no_rectangle_B : 
  ¬ ∃ (x y : ℝ), (x + y = 5 / 2) ∧ (x * y = 2) :=
by
  sorry

-- Part 3: General proof for any given sides of the known rectangle
theorem general_exists_rectangle_B (m n : ℝ) : 
  ∃ (x y : ℝ), (x + y = 3 * (m + n)) ∧ (x * y = 3 * m * n) :=
by
  sorry

end part1_exists_rectangle_B_part2_no_rectangle_B_general_exists_rectangle_B_l290_290037


namespace right_isosceles_hypotenuse_angle_l290_290153

theorem right_isosceles_hypotenuse_angle (α β : ℝ) (γ : ℝ)
  (h1 : α = 45) (h2 : β = 45) (h3 : γ = 90)
  (triangle_isosceles : α = β)
  (triangle_right : γ = 90) :
  γ = 90 :=
by
  sorry

end right_isosceles_hypotenuse_angle_l290_290153


namespace one_thirds_in_nine_thirds_l290_290282

theorem one_thirds_in_nine_thirds : ( (9 / 3) / (1 / 3) ) = 9 := 
by {
  sorry
}

end one_thirds_in_nine_thirds_l290_290282


namespace shaded_rectangle_area_l290_290156

theorem shaded_rectangle_area (side_length : ℝ) (x y : ℝ) 
  (h1 : side_length = 42) 
  (h2 : 4 * x + 2 * y = 168 - 4 * x) 
  (h3 : 2 * (side_length - y) + 2 * x = 168 - 4 * x)
  (h4 : 2 * (2 * x + y) = 168 - 4 * x) 
  (h5 : x = 18) :
  (2 * x) * (4 * x - (side_length - y)) = 540 := 
by
  sorry

end shaded_rectangle_area_l290_290156


namespace exists_infinite_triples_a_no_triples_b_l290_290757

-- Question (a)
theorem exists_infinite_triples_a : ∀ k : ℕ, ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2 - 1) :=
by {
  sorry
}

-- Question (b)
theorem no_triples_b : ¬ ∃ m n p : ℕ, 0 < m ∧ 0 < n ∧ 0 < p ∧ (4 * m * n - m - n = p^2) :=
by {
  sorry
}

end exists_infinite_triples_a_no_triples_b_l290_290757


namespace angie_bought_18_pretzels_l290_290666

theorem angie_bought_18_pretzels
  (B : ℕ := 12) -- Barry bought 12 pretzels
  (S : ℕ := B / 2) -- Shelly bought half as many pretzels as Barry
  (A : ℕ := 3 * S) -- Angie bought three times as many pretzels as Shelly
  : A = 18 := sorry

end angie_bought_18_pretzels_l290_290666


namespace elise_initial_dog_food_l290_290249

variable (initial_dog_food : ℤ)
variable (bought_first_bag : ℤ := 15)
variable (bought_second_bag : ℤ := 10)
variable (final_dog_food : ℤ := 40)

theorem elise_initial_dog_food :
  initial_dog_food + bought_first_bag + bought_second_bag = final_dog_food →
  initial_dog_food = 15 :=
by
  sorry

end elise_initial_dog_food_l290_290249


namespace ratio_of_democrats_l290_290331

theorem ratio_of_democrats (F M : ℕ) 
  (h1 : F + M = 990) 
  (h2 : (1 / 2 : ℚ) * F = 165) 
  (h3 : (1 / 4 : ℚ) * M = 165) : 
  (165 + 165) / 990 = 1 / 3 := 
by
  sorry

end ratio_of_democrats_l290_290331


namespace Bobby_has_27_pairs_l290_290359

-- Define the number of shoes Becky has
variable (B : ℕ)

-- Define the number of shoes Bonny has as 13, with the relationship to Becky's shoes
def Bonny_shoes : Prop := 2 * B - 5 = 13

-- Define the number of shoes Bobby has given Becky's count
def Bobby_shoes := 3 * B

-- Prove that Bobby has 27 pairs of shoes given the conditions
theorem Bobby_has_27_pairs (hB : Bonny_shoes B) : Bobby_shoes B = 27 := 
by 
  sorry

end Bobby_has_27_pairs_l290_290359


namespace AB_plus_C_eq_neg8_l290_290469

theorem AB_plus_C_eq_neg8 (A B C : ℤ) (g : ℝ → ℝ)
(hf : ∀ x > 3, g x > 0.5)
(heq : ∀ x, g x = x^2 / (A * x^2 + B * x + C))
(hasymp_vert : ∀ x, (A * (x + 3) * (x - 2) = 0 → x = -3 ∨ x = 2))
(hasymp_horiz : (1 : ℝ) / (A : ℝ) < 1) :
A + B + C = -8 :=
sorry

end AB_plus_C_eq_neg8_l290_290469


namespace probability_r20_to_r30_after_one_operation_l290_290550

noncomputable theory
open Classical

def sequence := list ℝ

def operation (seq : sequence) : sequence :=
(list.scanl (fun (acc : list ℝ) (x : ℝ) => if x < acc.head' then x :: acc.tail else x :: acc) seq).head'.reverse

def second_largest (l : list ℝ) : ℝ :=
(list.sort (≤) l).nth_le (l.length - 2) sorry

theorem probability_r20_to_r30_after_one_operation (seq : sequence) (h : seq.length = 40) (h_distinct : seq.nodup)
    : let r20 := seq.nth_le 19 sorry,
          r31 := seq.nth_le 30 sorry,
          sorted_seq := operation seq,
          p := sorted_seq.nth_le 29 sorry,
          prob := (1 : ℚ) / 31 * (1 : ℚ) / 30
      in p + q = 931 :=
sorry

end probability_r20_to_r30_after_one_operation_l290_290550


namespace oldest_child_age_l290_290959

theorem oldest_child_age (x : ℕ) (h_avg : (5 + 7 + 10 + x) / 4 = 8) : x = 10 :=
by
  sorry

end oldest_child_age_l290_290959


namespace total_volume_cylinder_cone_sphere_l290_290969

theorem total_volume_cylinder_cone_sphere (r h : ℝ) (π : ℝ)
  (hc : π * r^2 * h = 150 * π)
  (hv : ∀ (r h : ℝ) (π : ℝ), V_cone = 1/3 * π * r^2 * h)
  (hs : ∀ (r : ℝ) (π : ℝ), V_sphere = 4/3 * π * r^3) :
  V_total = 50 * π + (4/3 * π * (150^(2/3))) :=
by
  sorry

end total_volume_cylinder_cone_sphere_l290_290969


namespace even_number_of_divisors_lt_100_l290_290404

theorem even_number_of_divisors_lt_100 : 
  let n := 99 in
  let perfect_squares := {1, 4, 9, 16, 25, 36, 49, 64, 81} in
  n - perfect_squares.card = 90 := 
by
  sorry

end even_number_of_divisors_lt_100_l290_290404


namespace find_positive_real_solution_l290_290684

theorem find_positive_real_solution (x : ℝ) : 
  0 < x ∧ (1 / 2 * (4 * x^2 - 1) = (x^2 - 60 * x - 20) * (x^2 + 30 * x + 10)) ↔ 
  (x = 30 + Real.sqrt 919 ∨ x = -15 + Real.sqrt 216 ∧ 0 < -15 + Real.sqrt 216) :=
by sorry

end find_positive_real_solution_l290_290684


namespace gcd_power_diff_l290_290875

theorem gcd_power_diff (n m : ℕ) (h₁ : n = 2025) (h₂ : m = 2007) :
  (Nat.gcd (2^n - 1) (2^m - 1)) = 2^18 - 1 :=
by
  sorry

end gcd_power_diff_l290_290875


namespace length_of_segment_cutoff_l290_290008

-- Define the parabola equation
def parabola (x y : ℝ) := y^2 = 4 * (x + 1)

-- Define the line passing through the focus and perpendicular to the x-axis
def line_through_focus_perp_x_axis (x y : ℝ) := x = 0

-- The actual length calculation lemma
lemma segment_length : 
  ∀ (x y : ℝ), parabola x y → line_through_focus_perp_x_axis x y → y = 2 ∨ y = -2 :=
by sorry

-- The final theorem which gives the length of the segment
theorem length_of_segment_cutoff (y1 y2 : ℝ) :
  ∀ (x : ℝ), parabola x y1 → parabola x y2 → line_through_focus_perp_x_axis x y1 → line_through_focus_perp_x_axis x y2 → (y1 = 2 ∨ y1 = -2) ∧ (y2 = 2 ∨ y2 = -2) → abs (y2 - y1) = 4 :=
by sorry

end length_of_segment_cutoff_l290_290008


namespace range_of_a_l290_290423

theorem range_of_a (a : ℝ) (h : ∀ x, x > a → 2 * x + 2 / (x - a) ≥ 5) : a ≥ 1 / 2 :=
sorry

end range_of_a_l290_290423


namespace modulus_of_product_l290_290587

namespace ComplexModule

open Complex

-- Definition of the complex numbers z1 and z2
def z1 : ℂ := 1 + I
def z2 : ℂ := 2 - I

-- Definition of their product z1z2
def z1z2 : ℂ := z1 * z2

-- Statement we need to prove (the modulus of z1z2 is √10)
theorem modulus_of_product : abs z1z2 = Real.sqrt 10 := by
  sorry

end ComplexModule

end modulus_of_product_l290_290587


namespace zero_in_interval_l290_290447

noncomputable def f (x : ℝ) : ℝ := Real.log x + x - 4

theorem zero_in_interval : 
  ∃ x₀, f x₀ = 0 ∧ (2 : ℝ) < x₀ ∧ x₀ < (3 : ℝ) :=
by
  sorry

end zero_in_interval_l290_290447


namespace complement_U_A_l290_290930

open Set

def U : Set ℤ := {-2, -1, 0, 1, 2}
def A : Set ℤ := {x | x^2 < 3}

theorem complement_U_A :
  (U \ A) = {-2, 2} :=
sorry

end complement_U_A_l290_290930


namespace olaf_and_dad_total_score_l290_290608

variable (dad_score : ℕ)
variable (olaf_score : ℕ)
variable (total_score : ℕ)

-- Define conditions based on the problem
def condition1 : Prop := dad_score = 7
def condition2 : Prop := olaf_score = 3 * dad_score
def condition3 : Prop := total_score = olaf_score + dad_score

-- Define the theorem to be proven
theorem olaf_and_dad_total_score : condition1 ∧ condition2 ∧ condition3 → total_score = 28 := by
  intro h
  cases h with
    | intro h1 h'
    | intro h2 h3 =>
        -- Using the conditions given
        unfold condition1 at h1
        unfold condition2 at h2
        unfold condition3 at h3
        sorry

end olaf_and_dad_total_score_l290_290608


namespace no_real_solution_l290_290689

theorem no_real_solution (x y : ℝ) (h: y = 3 * x - 1) : ¬ (4 * y ^ 2 + y + 3 = 3 * (8 * x ^ 2 + 3 * y + 1)) :=
by
  sorry

end no_real_solution_l290_290689


namespace solution_to_inequality_system_l290_290616

theorem solution_to_inequality_system :
  (∀ x : ℝ, 2 * (x - 1) < x + 2 → (x + 1) / 2 < x → 1 < x ∧ x < 4) :=
by
  intros x h1 h2
  sorry

end solution_to_inequality_system_l290_290616


namespace rectangular_prism_volume_l290_290842

variables (a b c : ℝ)

theorem rectangular_prism_volume
  (h1 : a * b = 24)
  (h2 : b * c = 8)
  (h3 : c * a = 3) :
  a * b * c = 24 :=
by
  sorry

end rectangular_prism_volume_l290_290842


namespace vasya_numbers_l290_290814

theorem vasya_numbers :
  ∃ x y : ℝ, (x + y = x * y ∧ x * y = x / y) ∧ (x = 1/2 ∧ y = -1) :=
by
  sorry

end vasya_numbers_l290_290814


namespace smallest_d_l290_290220

theorem smallest_d (d : ℝ) : 
  (∃ d, 2 * d = Real.sqrt ((4 * Real.sqrt 3) ^ 2 + (d + 4) ^ 2)) →
  d = (2 * (2 - Real.sqrt 52)) / 3 :=
by
  sorry

end smallest_d_l290_290220


namespace triangle_perimeter_l290_290352

def triangle_side_lengths : ℕ × ℕ × ℕ := (10, 6, 7)

def perimeter (a b c : ℕ) : ℕ := a + b + c

theorem triangle_perimeter (a b c : ℕ) (h : (a, b, c) = triangle_side_lengths) : 
  perimeter a b c = 23 := by
  -- We formulate the statement and leave the proof for later
  sorry

end triangle_perimeter_l290_290352


namespace fraction_of_workers_read_Saramago_l290_290427

theorem fraction_of_workers_read_Saramago (S : ℚ) (total_workers : ℕ) (read_Kureishi_fraction : ℚ) 
  (read_both : ℕ) (read_neither_delta_from_Saramago_only : ℕ)
  (total_workers_eq : total_workers = 40) 
  (read_Kureishi_eq : read_Kureishi_fraction = 5 / 8)
  (read_both_eq : read_both = 2)
  (read_neither_delta_eq : read_neither_delta_from_Saramago_only = 1)
  (workers_eq : 2 + (total_workers * S - 2) + (total_workers * read_Kureishi_fraction - 2) + 
                ((total_workers * S - 2) - read_neither_delta_from_Saramago_only) = total_workers) :
  S = 9 / 40 := 
by
  sorry

end fraction_of_workers_read_Saramago_l290_290427


namespace no_politics_reporters_l290_290429

theorem no_politics_reporters (X Y Both XDontY YDontX International PercentageTotal : ℝ) 
  (hX : X = 0.35)
  (hY : Y = 0.25)
  (hBoth : Both = 0.20)
  (hXDontY : XDontY = 0.30)
  (hInternational : International = 0.15)
  (hPercentageTotal : PercentageTotal = 1.0) :
  PercentageTotal - ((X + Y - Both) - XDontY + International) = 0.75 :=
by sorry

end no_politics_reporters_l290_290429


namespace Ben_sales_value_l290_290100

noncomputable def value_of_sale (old_salary new_salary commission_ratio sales_required : ℝ) (diff_salary: ℝ) :=
  ∃ x : ℝ, 0.15 * x * sales_required = diff_salary ∧ x = 750

theorem Ben_sales_value (old_salary new_salary commission_ratio sales_required diff_salary: ℝ)
  (h1: old_salary = 75000)
  (h2: new_salary = 45000)
  (h3: commission_ratio = 0.15)
  (h4: sales_required = 266.67)
  (h5: diff_salary = old_salary - new_salary) :
  value_of_sale old_salary new_salary commission_ratio sales_required diff_salary :=
by
  sorry

end Ben_sales_value_l290_290100


namespace angle_between_east_and_south_is_90_degrees_l290_290356

-- Define the main theorem statement
theorem angle_between_east_and_south_is_90_degrees :
  ∀ (circle : Type) (num_rays : ℕ) (direction : ℕ → ℕ) (north east south : ℕ),
  num_rays = 12 →
  (∀ i, i < num_rays → direction i = (i * 360 / num_rays) % 360) →
  direction north = 0 →
  direction east = 90 →
  direction south = 180 →
  (min ((direction south - direction east) % 360) (360 - (direction south - direction east) % 360)) = 90 :=
by
  intros
  -- Skipped the proof
  sorry

end angle_between_east_and_south_is_90_degrees_l290_290356


namespace multiplication_schemes_correct_l290_290157

theorem multiplication_schemes_correct :
  ∃ A B C D E F G H I K L M N P : ℕ,
    A = 7 ∧ B = 7 ∧ C = 4 ∧ D = 4 ∧ E = 3 ∧ F = 0 ∧ G = 8 ∧ H = 3 ∧ I = 3 ∧ K = 8 ∧ L = 8 ∧ M = 0 ∧ N = 7 ∧ P = 7 ∧
    (A * 10 + B) * (C * 10 + D) * (A * 10 + B) = E * 100 + F * 10 + G ∧
    (C * 10 + G) * (K * 10 + L) = A * 100 + M * 10 + C ∧
    E * 100 + F * 10 + G / (H * 1000 + I * 100 + G * 10 + G) = (E * 100 + F * 10 + G) / (H * 1000 + I * 100 + G * 10 + G) ∧
    (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) = (A * 100 + M * 10 + C) / (N * 1000 + P * 100 + C * 10 + C) :=
sorry

end multiplication_schemes_correct_l290_290157


namespace angle_ACD_measure_l290_290014

theorem angle_ACD_measure {ABD BAE ABC ACD : ℕ} 
  (h1 : ABD = 125) 
  (h2 : BAE = 95) 
  (h3 : ABC = 180 - ABD) 
  (h4 : ABD + ABC = 180 ) : 
  ACD = 180 - (BAE + ABC) :=
by 
  sorry

end angle_ACD_measure_l290_290014


namespace coordinates_of_B_l290_290416

def pointA : Prod Int Int := (-3, 2)
def moveRight (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1 + units, p.2)
def moveDown (p : Prod Int Int) (units : Int) : Prod Int Int := (p.1, p.2 - units)
def pointB : Prod Int Int := moveDown (moveRight pointA 1) 2

theorem coordinates_of_B :
  pointB = (-2, 0) :=
sorry

end coordinates_of_B_l290_290416


namespace parallelogram_base_length_l290_290641

theorem parallelogram_base_length (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_eq_2b : h = 2 * b) (area_eq_98 : area = 98) 
  (area_def : area = b * h) : b = 7 :=
by
  sorry

end parallelogram_base_length_l290_290641


namespace train_travel_distance_l290_290649

def coal_efficiency := (5 : ℝ) / (2 : ℝ)  -- Efficiency in miles per pound
def coal_remaining := 160  -- Coal remaining in pounds
def distance_travelled := coal_remaining * coal_efficiency  -- Total distance the train can travel

theorem train_travel_distance : distance_travelled = 400 := 
by
  sorry

end train_travel_distance_l290_290649


namespace max_value_S_n_l290_290702

open Nat

noncomputable def a_n (n : ℕ) : ℤ := 20 + (n - 1) * (-2)

noncomputable def S_n (n : ℕ) : ℤ := n * 20 + (n * (n - 1)) * (-2) / 2

theorem max_value_S_n : ∃ n : ℕ, S_n n = 110 :=
by
  sorry

end max_value_S_n_l290_290702


namespace proposition_2_proposition_3_l290_290711

theorem proposition_2 (a b : ℝ) (h: a > |b|) : a^2 > b^2 := 
sorry

theorem proposition_3 (a b : ℝ) (h: a > b) : a^3 > b^3 := 
sorry

end proposition_2_proposition_3_l290_290711


namespace find_k_l290_290425

theorem find_k (x y k : ℝ) (h1 : x + y = 5 * k) (h2 : x - y = 9 * k) (h3 : x - 2 * y = 22) : k = 2 :=
by
  sorry

end find_k_l290_290425


namespace asha_savings_l290_290357

theorem asha_savings (brother father mother granny spending remaining total borrowed_gifted savings : ℤ) 
  (h1 : brother = 20)
  (h2 : father = 40)
  (h3 : mother = 30)
  (h4 : granny = 70)
  (h5 : spending = 3 * total / 4)
  (h6 : remaining = 65)
  (h7 : remaining = total - spending)
  (h8 : total = brother + father + mother + granny + savings)
  (h9 : borrowed_gifted = brother + father + mother + granny) :
  savings = 100 := by
    sorry

end asha_savings_l290_290357


namespace power_computation_l290_290208

theorem power_computation : (12 ^ (12 / 2)) = 2985984 := by
  sorry

end power_computation_l290_290208


namespace division_of_fractions_l290_290277

def nine_thirds : ℚ := 9 / 3
def one_third : ℚ := 1 / 3
def division := nine_thirds / one_third

theorem division_of_fractions : division = 9 := by
  sorry

end division_of_fractions_l290_290277


namespace compute_expression_l290_290240

theorem compute_expression : 12 * (1 / 17) * 34 = 24 :=
by sorry

end compute_expression_l290_290240
