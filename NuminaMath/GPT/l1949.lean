import Mathlib

namespace integral_sin3_cos_l1949_194954

open Real

theorem integral_sin3_cos :
  ∫ z in (π / 4)..(π / 2), sin z ^ 3 * cos z = 3 / 16 := by
  sorry

end integral_sin3_cos_l1949_194954


namespace pipes_fill_cistern_together_in_15_minutes_l1949_194968

-- Define the problem's conditions in Lean
def PipeA_rate := (1 / 2) / 15
def PipeB_rate := (1 / 3) / 10

-- Define the combined rate
def combined_rate := PipeA_rate + PipeB_rate

-- Define the time to fill the cistern by both pipes working together
def time_to_fill_cistern := 1 / combined_rate

-- State the theorem to prove
theorem pipes_fill_cistern_together_in_15_minutes :
  time_to_fill_cistern = 15 := by
  sorry

end pipes_fill_cistern_together_in_15_minutes_l1949_194968


namespace island_challenge_probability_l1949_194960
open Nat

theorem island_challenge_probability :
  let total_ways := choose 20 3
  let ways_one_tribe := choose 10 3
  let combined_ways := 2 * ways_one_tribe
  let probability := combined_ways / total_ways
  probability = (20 : ℚ) / 95 :=
by
  sorry

end island_challenge_probability_l1949_194960


namespace melanie_initial_plums_l1949_194918

-- define the conditions as constants
def plums_given_to_sam : ℕ := 3
def plums_left_with_melanie : ℕ := 4

-- define the statement to be proven
theorem melanie_initial_plums : (plums_given_to_sam + plums_left_with_melanie = 7) :=
by
  sorry

end melanie_initial_plums_l1949_194918


namespace shaded_area_l1949_194901

theorem shaded_area (r : ℝ) (π : ℝ) (shaded_area : ℝ) (h_r : r = 4) (h_π : π = 3) : shaded_area = 32.5 :=
by
  sorry

end shaded_area_l1949_194901


namespace theta_terminal_side_l1949_194928

theorem theta_terminal_side (alpha : ℝ) (theta : ℝ) (h1 : alpha = 1560) (h2 : -360 < theta ∧ theta < 360) :
    (theta = 120 ∨ theta = -240) := by
  -- The proof steps would go here
  sorry

end theta_terminal_side_l1949_194928


namespace flag_distance_false_l1949_194989

theorem flag_distance_false (track_length : ℕ) (num_flags : ℕ) (flag1_flagN : 2 ≤ num_flags)
  (h1 : track_length = 90) (h2 : num_flags = 10) :
  ¬ (track_length / (num_flags - 1) = 9) :=
by
  sorry

end flag_distance_false_l1949_194989


namespace lisa_socks_total_l1949_194920

def total_socks (initial : ℕ) (sandra : ℕ) (cousin_ratio : ℕ → ℕ) (mom_extra : ℕ → ℕ) : ℕ :=
  initial + sandra + cousin_ratio sandra + mom_extra initial

def cousin_ratio (sandra : ℕ) : ℕ := sandra / 5
def mom_extra (initial : ℕ) : ℕ := 3 * initial + 8

theorem lisa_socks_total :
  total_socks 12 20 cousin_ratio mom_extra = 80 := by
  sorry

end lisa_socks_total_l1949_194920


namespace find_line_equation_l1949_194930

-- Define the conditions for the x-intercept and inclination angle
def x_intercept (x : ℝ) (line : ℝ → ℝ) : Prop :=
  line x = 0

def inclination_angle (θ : ℝ) (k : ℝ) : Prop :=
  k = Real.tan θ

-- Define the properties of the line we're working with
def line (x : ℝ) : ℝ := -x + 5

theorem find_line_equation :
  x_intercept 5 line ∧ inclination_angle (3 * Real.pi / 4) (-1) → (∀ x, line x = -x + 5) :=
by
  intro h
  sorry

end find_line_equation_l1949_194930


namespace S_calculation_T_calculation_l1949_194909

def S (a b : ℕ) : ℕ := 4 * a + 6 * b
def T (a b : ℕ) : ℕ := 5 * a + 3 * b

theorem S_calculation : S 6 3 = 42 :=
by sorry

theorem T_calculation : T 6 3 = 39 :=
by sorry

end S_calculation_T_calculation_l1949_194909


namespace geometric_sequence_b_value_l1949_194998

theorem geometric_sequence_b_value (b : ℝ) (r : ℝ) (h1 : 210 * r = b) (h2 : b * r = 35 / 36) (hb : b > 0) : 
  b = Real.sqrt (7350 / 36) :=
by
  sorry

end geometric_sequence_b_value_l1949_194998


namespace max_distance_from_point_on_circle_to_line_l1949_194965

noncomputable def center_of_circle : ℝ × ℝ := (5, 3)
noncomputable def radius_of_circle : ℝ := 3
noncomputable def line_eqn (x y : ℝ) : ℝ := 3 * x + 4 * y - 2
noncomputable def distance_point_to_line (px py a b c : ℝ) : ℝ := (|a * px + b * py + c|) / (Real.sqrt (a * a + b * b))

theorem max_distance_from_point_on_circle_to_line :
  let Cx := (center_of_circle.1)
  let Cy := (center_of_circle.2)
  let d := distance_point_to_line Cx Cy 3 4 (-2)
  d + radius_of_circle = 8 := by
  sorry

end max_distance_from_point_on_circle_to_line_l1949_194965


namespace van_capacity_l1949_194951

theorem van_capacity (s a v : ℕ) (h1 : s = 2) (h2 : a = 6) (h3 : v = 2) : (s + a) / v = 4 := by
  sorry

end van_capacity_l1949_194951


namespace ratio_of_triangle_and_hexagon_l1949_194983

variable {n m : ℝ}

-- Conditions:
def is_regular_hexagon (ABCDEF : Type) : Prop := sorry
def area_of_hexagon (ABCDEF : Type) (n : ℝ) : Prop := sorry
def area_of_triangle_ACE (ABCDEF : Type) (m : ℝ) : Prop := sorry
  
theorem ratio_of_triangle_and_hexagon
  (ABCDEF : Type)
  (H1 : is_regular_hexagon ABCDEF)
  (H2 : area_of_hexagon ABCDEF n)
  (H3 : area_of_triangle_ACE ABCDEF m) :
  m / n = 2 / 3 := 
  sorry

end ratio_of_triangle_and_hexagon_l1949_194983


namespace distance_between_foci_of_hyperbola_l1949_194950

theorem distance_between_foci_of_hyperbola :
  (∀ x y : ℝ, (y = 2 * x + 3) ∨ (y = -2 * x + 1)) →
  ∀ p : ℝ × ℝ, (p = (2, 1)) →
  ∃ d : ℝ, d = 2 * Real.sqrt 30 :=
by
  sorry

end distance_between_foci_of_hyperbola_l1949_194950


namespace find_value_of_a_b_ab_l1949_194981

variable (a b : ℝ)

theorem find_value_of_a_b_ab
  (h1 : 2 * a + 2 * b + a * b = 1)
  (h2 : a + b + 3 * a * b = -2) :
  a + b + a * b = 0 := 
sorry

end find_value_of_a_b_ab_l1949_194981


namespace find_k_value_l1949_194931

theorem find_k_value (k : ℝ) :
  (∃ x1 x2 : ℝ, (2 * x1^2 + k * x1 - 2 * k + 1 = 0) ∧ 
                (2 * x2^2 + k * x2 - 2 * k + 1 = 0) ∧ 
                (x1 ≠ x2)) ∧
  ((x1^2 + x2^2 = 29/4)) ↔ (k = 3) := 
sorry

end find_k_value_l1949_194931


namespace increasing_interval_l1949_194903

noncomputable def f (x : ℝ) : ℝ := Real.log x - 2 * x^2

theorem increasing_interval :
  ∃ a b : ℝ, (0 < a) ∧ (a < b) ∧ (b = 1/2) ∧ (∀ x : ℝ, a < x ∧ x < b → (deriv f x > 0)) :=
by
  sorry

end increasing_interval_l1949_194903


namespace complex_root_sixth_power_sum_equals_38908_l1949_194953

noncomputable def omega : ℂ :=
  -- By definition, omega should satisfy the below properties.
  -- The exact value of omega is not being defined, we will use algebraic properties in the proof.
  sorry

theorem complex_root_sixth_power_sum_equals_38908 : 
  ∀ (ω : ℂ), ω^3 = 1 ∧ ¬(ω.re = 1) → (2 - ω + 2 * ω^2)^6 + (2 + ω - 2 * ω^2)^6 = 38908 :=
by
  -- Proof will utilize given conditions:
  -- 1. ω^3 = 1
  -- 2. ω is not real (or ω.re is not 1)
  sorry

end complex_root_sixth_power_sum_equals_38908_l1949_194953


namespace remove_terms_l1949_194967

-- Define the fractions
def f1 := 1 / 3
def f2 := 1 / 6
def f3 := 1 / 9
def f4 := 1 / 12
def f5 := 1 / 15
def f6 := 1 / 18

-- Define the total sum
def total_sum := f1 + f2 + f3 + f4 + f5 + f6

-- Define the target sum after removal
def target_sum := 2 / 3

-- Define the condition to be proven
theorem remove_terms {x y : Real} (h1 : (x = f4) ∧ (y = f5)) : 
  total_sum - (x + y) = target_sum := by
  sorry

end remove_terms_l1949_194967


namespace find_sum_of_cubes_l1949_194973

-- Define the distinct real numbers p, q, and r
variables {p q r : ℝ}

-- Conditions
-- Distinctness condition
axiom h_distinct : p ≠ q ∧ q ≠ r ∧ r ≠ p

-- Given condition
axiom h_eq : (p^3 + 7) / p = (q^3 + 7) / q ∧ (q^3 + 7) / q = (r^3 + 7) / r

-- Proof goal
theorem find_sum_of_cubes : p^3 + q^3 + r^3 = -21 :=
sorry

end find_sum_of_cubes_l1949_194973


namespace probability_one_boy_one_girl_l1949_194975

-- Define the total number of students (5), the number of boys (3), and the number of girls (2).
def total_students : Nat := 5
def boys : Nat := 3
def girls : Nat := 2

-- Define the probability calculation in Lean.
noncomputable def select_2_students_prob : ℚ :=
  let total_combinations := Nat.choose total_students 2
  let favorable_combinations := Nat.choose boys 1 * Nat.choose girls 1
  favorable_combinations / total_combinations

-- The statement we need to prove is that this probability is 3/5
theorem probability_one_boy_one_girl : select_2_students_prob = 3 / 5 := sorry

end probability_one_boy_one_girl_l1949_194975


namespace remaining_seeds_l1949_194933

def initial_seeds : Nat := 54000
def seeds_per_zone : Nat := 3123
def number_of_zones : Nat := 7

theorem remaining_seeds (initial_seeds seeds_per_zone number_of_zones : Nat) : 
  initial_seeds - (seeds_per_zone * number_of_zones) = 32139 := 
by 
  sorry

end remaining_seeds_l1949_194933


namespace term_in_AP_is_zero_l1949_194916

theorem term_in_AP_is_zero (a d : ℤ) 
  (h : (a + 4 * d) + (a + 20 * d) = (a + 7 * d) + (a + 14 * d) + (a + 12 * d)) :
  a + (-9) * d = 0 :=
by
  sorry

end term_in_AP_is_zero_l1949_194916


namespace line_passes_through_parabola_vertex_l1949_194925

theorem line_passes_through_parabola_vertex :
  ∃ (a : ℝ), (∃ (b : ℝ), b = a ∧ (a = 0 ∨ a = 1)) :=
by
  sorry

end line_passes_through_parabola_vertex_l1949_194925


namespace probability_at_least_one_black_ball_l1949_194929

theorem probability_at_least_one_black_ball :
  let total_balls := 6
  let red_balls := 2
  let white_ball := 1
  let black_balls := 3
  let total_combinations := Nat.choose total_balls 2
  let non_black_combinations := Nat.choose (total_balls - black_balls) 2
  let probability := 1 - (non_black_combinations / total_combinations : ℚ)
  probability = 4 / 5 :=
by
  sorry

end probability_at_least_one_black_ball_l1949_194929


namespace vacuum_cleaner_cost_l1949_194994

-- Variables
variables (V : ℝ)

-- Conditions
def cost_of_dishwasher := 450
def coupon := 75
def total_spent := 625

-- The main theorem to prove
theorem vacuum_cleaner_cost : V + cost_of_dishwasher - coupon = total_spent → V = 250 :=
by
  -- Proof logic goes here
  sorry

end vacuum_cleaner_cost_l1949_194994


namespace larger_number_l1949_194910

/-- The difference of two numbers is 1375 and the larger divided by the smaller gives a quotient of 6 and a remainder of 15. 
Prove that the larger number is 1647. -/
theorem larger_number (L S : ℕ) 
  (h1 : L - S = 1375) 
  (h2 : L = 6 * S + 15) : 
  L = 1647 := 
sorry

end larger_number_l1949_194910


namespace intersection_A_B_l1949_194922

-- Define the set A as natural numbers greater than 1
def A : Set ℕ := {x | x > 1}

-- Define the set B as numbers less than or equal to 3
def B : Set ℕ := {x | x ≤ 3}

-- Define the intersection of A and B
def A_inter_B : Set ℕ := {x | x ∈ A ∧ x ∈ B}

-- State the theorem we want to prove
theorem intersection_A_B : A_inter_B = {2, 3} :=
  sorry

end intersection_A_B_l1949_194922


namespace probability_triangle_nonagon_l1949_194948

-- Define the total number of ways to choose 3 vertices from 9 vertices
def total_ways_to_choose_triangle : ℕ := Nat.choose 9 3

-- Define the number of favorable outcomes
def favorable_outcomes_one_side : ℕ := 9 * 5
def favorable_outcomes_two_sides : ℕ := 9

def total_favorable_outcomes : ℕ := favorable_outcomes_one_side + favorable_outcomes_two_sides

-- Define the probability as a rational number
def probability_at_least_one_side_nonagon (total: ℕ) (favorable: ℕ) : ℚ :=
  favorable / total
  
-- Theorem stating the probability
theorem probability_triangle_nonagon :
  probability_at_least_one_side_nonagon total_ways_to_choose_triangle total_favorable_outcomes = 9 / 14 :=
by
  sorry

end probability_triangle_nonagon_l1949_194948


namespace tangent_line_at_1_f_positive_iff_a_leq_2_l1949_194943

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (x + 1) * Real.log x - a * (x - 1)

theorem tangent_line_at_1 (a : ℝ) (h : a = 4) : 
  ∃ k b : ℝ, (k = -2) ∧ (b = 2) ∧ (∀ x : ℝ, f x a = k * (x - 1) + b) :=
sorry

theorem f_positive_iff_a_leq_2 : 
  (∀ x : ℝ, 1 < x → f x a > 0) ↔ a ≤ 2 :=
sorry

end tangent_line_at_1_f_positive_iff_a_leq_2_l1949_194943


namespace tristan_study_hours_l1949_194972

theorem tristan_study_hours :
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  saturday_hours = 2 := by
{
  let monday_hours := 4
  let tuesday_hours := 2 * monday_hours
  let wednesday_hours := 3
  let thursday_hours := 3
  let friday_hours := 3
  let total_hours := monday_hours + tuesday_hours + wednesday_hours + thursday_hours + friday_hours
  let remaining_hours := 25 - total_hours
  let saturday_hours := remaining_hours / 2
  sorry
}

end tristan_study_hours_l1949_194972


namespace rainfall_second_week_l1949_194964

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) (first_week_rainfall : ℝ) (second_week_rainfall : ℝ) :
  total_rainfall = 35 →
  ratio = 1.5 →
  total_rainfall = first_week_rainfall + second_week_rainfall →
  second_week_rainfall = ratio * first_week_rainfall →
  second_week_rainfall = 21 :=
by
  intros
  sorry

end rainfall_second_week_l1949_194964


namespace min_value_of_a1_plus_a7_l1949_194921

variable {a : ℕ → ℝ}
variable {a3 a5 : ℝ}

-- Conditions
def is_positive_geometric_sequence (a : ℕ → ℝ) := 
  ∀ n, a n > 0 ∧ (∃ r, ∀ i, a (i + 1) = a i * r)

def condition (a : ℕ → ℝ) (a3 a5 : ℝ) :=
  a 3 = a3 ∧ a 5 = a5 ∧ a3 * a5 = 64

-- Prove that the minimum value of a1 + a7 is 16
theorem min_value_of_a1_plus_a7
  (h1 : is_positive_geometric_sequence a)
  (h2 : condition a a3 a5) :
  ∃ a1 a7, a 1 = a1 ∧ a 7 = a7 ∧ (∃ (min_sum : ℝ), min_sum = 16 ∧ ∀ sum, sum = a1 + a7 → sum ≥ min_sum) :=
sorry

end min_value_of_a1_plus_a7_l1949_194921


namespace polynomial_third_and_fourth_equal_l1949_194937

theorem polynomial_third_and_fourth_equal (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p + q = 1)
  (h_eq : 45 * p^8 * q^2 = 120 * p^7 * q^3) : p = (8 : ℝ) / 11 :=
by
  sorry

end polynomial_third_and_fourth_equal_l1949_194937


namespace fraction_of_usual_speed_l1949_194902

-- Definitions based on conditions
variable (S R : ℝ)
variable (h1 : S * 60 = R * 72)

-- Goal statement
theorem fraction_of_usual_speed (h1 : S * 60 = R * 72) : R / S = 5 / 6 :=
by
  sorry

end fraction_of_usual_speed_l1949_194902


namespace correct_product_l1949_194992

theorem correct_product : 
  (0.0063 * 3.85 = 0.024255) :=
sorry

end correct_product_l1949_194992


namespace projection_of_difference_eq_l1949_194904

noncomputable def vec_magnitude (v : ℝ × ℝ) : ℝ :=
Real.sqrt (v.1^2 + v.2^2)

noncomputable def vec_dot (v w : ℝ × ℝ) : ℝ :=
v.1 * w.1 + v.2 * w.2

noncomputable def vec_projection (v w : ℝ × ℝ) : ℝ :=
vec_dot (v - w) v / vec_magnitude v

variables (a b : ℝ × ℝ)
  (congruence_cond : vec_magnitude a / vec_magnitude b = Real.cos θ)

theorem projection_of_difference_eq (h : vec_magnitude a / vec_magnitude b = Real.cos θ) :
  vec_projection (a - b) a = (vec_dot a a - vec_dot b b) / vec_magnitude a :=
sorry

end projection_of_difference_eq_l1949_194904


namespace equal_after_operations_l1949_194984

theorem equal_after_operations :
  let initial_first_number := 365
  let initial_second_number := 24
  let first_number_after_n_operations := initial_first_number - 19 * 11
  let second_number_after_n_operations := initial_second_number + 12 * 11
  first_number_after_n_operations = second_number_after_n_operations := sorry

end equal_after_operations_l1949_194984


namespace artist_paints_33_square_meters_l1949_194962

/-
Conditions:
1. The artist has 14 cubes.
2. Each cube has an edge of 1 meter.
3. The cubes are arranged in a pyramid-like structure with three layers.
4. The top layer has 1 cube, the middle layer has 4 cubes, and the bottom layer has 9 cubes.
-/

def exposed_surface_area (num_cubes : Nat) (layer1 : Nat) (layer2 : Nat) (layer3 : Nat) : Nat :=
  let layer1_area := 5 -- Each top layer cube has 5 faces exposed
  let layer2_edge_cubes := 4 -- Count of cubes on the edge in middle layer
  let layer2_area := layer2_edge_cubes * 3 -- Each middle layer edge cube has 3 faces exposed
  let layer3_area := 9 -- Each bottom layer cube has 1 face exposed
  let top_faces := layer1 + layer2 + layer3 -- All top faces exposed
  layer1_area + layer2_area + layer3_area + top_faces

theorem artist_paints_33_square_meters :
  exposed_surface_area 14 1 4 9 = 33 := 
sorry

end artist_paints_33_square_meters_l1949_194962


namespace fraction_simplification_l1949_194987

theorem fraction_simplification (a : ℝ) (h1 : a > 1) (h2 : a ≠ 2 / Real.sqrt 3) : 
  (a^3 - 3 * a^2 + 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) / 
  (a^3 + 3 * a^2 - 4 + (a^2 - 4) * Real.sqrt (a^2 - 1)) = 
  ((a - 2) * Real.sqrt (a + 1)) / ((a + 2) * Real.sqrt (a - 1)) :=
by
  sorry

end fraction_simplification_l1949_194987


namespace average_number_of_fish_is_75_l1949_194958

-- Define the conditions
def BoastPool_fish := 75
def OnumLake_fish := BoastPool_fish + 25
def RiddlePond_fish := OnumLake_fish / 2

-- Prove the average number of fish
theorem average_number_of_fish_is_75 :
  (BoastPool_fish + OnumLake_fish + RiddlePond_fish) / 3 = 75 :=
by
  sorry

end average_number_of_fish_is_75_l1949_194958


namespace product_of_two_numbers_l1949_194969

theorem product_of_two_numbers (x y : ℝ) (h1 : x + y = 60) (h2 : x - y = 10) : x * y = 875 := by
  sorry

end product_of_two_numbers_l1949_194969


namespace symmetric_points_origin_l1949_194940

theorem symmetric_points_origin (a b : ℝ) (h : (1, 2) = (-a, -b)) : a = -1 ∧ b = -2 :=
sorry

end symmetric_points_origin_l1949_194940


namespace parallelepiped_analogy_l1949_194949

-- Define plane figures and the concept of analogy for a parallelepiped 
-- (specifically here as a parallelogram) in space
inductive PlaneFigure where
  | triangle
  | parallelogram
  | trapezoid
  | rectangle

open PlaneFigure

/-- 
  Given the properties and definitions of a parallelepiped and plane figures,
  we want to show that the appropriate analogy for a parallelepiped in space
  is a parallelogram.
-/
theorem parallelepiped_analogy : 
  (analogy : PlaneFigure) = parallelogram :=
sorry

end parallelepiped_analogy_l1949_194949


namespace percentage_shoes_polished_l1949_194942

theorem percentage_shoes_polished (total_pairs : ℕ) (shoes_to_polish : ℕ)
  (total_individual_shoes : ℕ := total_pairs * 2)
  (shoes_polished : ℕ := total_individual_shoes - shoes_to_polish)
  (percentage_polished : ℚ := (shoes_polished : ℚ) / total_individual_shoes * 100) :
  total_pairs = 10 → shoes_to_polish = 11 → percentage_polished = 45 :=
by
  intros hpairs hleft
  sorry

end percentage_shoes_polished_l1949_194942


namespace sledding_small_hills_l1949_194924

theorem sledding_small_hills (total_sleds tall_hills_sleds sleds_per_tall_hill sleds_per_small_hill small_hills : ℕ) 
  (h1 : total_sleds = 14)
  (h2 : tall_hills_sleds = 2)
  (h3 : sleds_per_tall_hill = 4)
  (h4 : sleds_per_small_hill = sleds_per_tall_hill / 2)
  (h5 : total_sleds = tall_hills_sleds * sleds_per_tall_hill + small_hills * sleds_per_small_hill)
  : small_hills = 3 := 
sorry

end sledding_small_hills_l1949_194924


namespace range_of_a_if_in_first_quadrant_l1949_194985

noncomputable def is_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

theorem range_of_a_if_in_first_quadrant (a : ℝ) :
  is_first_quadrant ((1 + a * Complex.I) / (2 - Complex.I)) ↔ (-1/2 : ℝ) < a ∧ a < 2 := 
sorry

end range_of_a_if_in_first_quadrant_l1949_194985


namespace molecular_weight_of_7_moles_AlPO4_is_correct_l1949_194932

def atomic_weight_Al : Float := 26.98
def atomic_weight_P : Float := 30.97
def atomic_weight_O : Float := 16.00

def molecular_weight_AlPO4 : Float :=
  (1 * atomic_weight_Al) + (1 * atomic_weight_P) + (4 * atomic_weight_O)

noncomputable def weight_of_7_moles_AlPO4 : Float :=
  7 * molecular_weight_AlPO4

theorem molecular_weight_of_7_moles_AlPO4_is_correct :
  weight_of_7_moles_AlPO4 = 853.65 := by
  -- computation goes here
  sorry

end molecular_weight_of_7_moles_AlPO4_is_correct_l1949_194932


namespace ratio_of_150_to_10_l1949_194986

theorem ratio_of_150_to_10 : 150 / 10 = 15 := by 
  sorry

end ratio_of_150_to_10_l1949_194986


namespace fraction_inequality_l1949_194957

open Real

theorem fraction_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h : a < b + c) :
  a / (1 + a) < b / (1 + b) + c / (1 + c) :=
  sorry

end fraction_inequality_l1949_194957


namespace contrapositive_example_l1949_194915

theorem contrapositive_example (x : ℝ) : (x > 2 → x > 0) ↔ (x ≤ 2 → x ≤ 0) :=
by
  sorry

end contrapositive_example_l1949_194915


namespace other_acute_angle_right_triangle_l1949_194988

theorem other_acute_angle_right_triangle (A : ℝ) (B : ℝ) (C : ℝ) (h₁ : A + B = 90) (h₂ : B = 54) : A = 36 :=
by
  sorry

end other_acute_angle_right_triangle_l1949_194988


namespace exists_b_c_with_integral_roots_l1949_194941

theorem exists_b_c_with_integral_roots :
  ∃ (b c : ℝ), (∃ (p q : ℤ), (x^2 + b * x + c = 0) ∧ (x^2 + (b + 1) * x + (c + 1) = 0) ∧ 
               ((x - p) * (x - q) = x^2 - (p + q) * x + p*q)) ∧
              (∃ (r s : ℤ), (x^2 + (b+1) * x + (c+1) = 0) ∧ 
              ((x - r) * (x - s) = x^2 - (r + s) * x + r*s)) :=
by
  sorry

end exists_b_c_with_integral_roots_l1949_194941


namespace complement_of_angle_l1949_194917

theorem complement_of_angle (x : ℝ) (h1 : 3 * x + 10 = 90 - x) : 3 * x + 10 = 70 :=
by
  sorry

end complement_of_angle_l1949_194917


namespace digits_C_not_make_1C34_divisible_by_4_l1949_194963

theorem digits_C_not_make_1C34_divisible_by_4 :
  ∀ (C : ℕ), (C ≥ 0) ∧ (C ≤ 9) → ¬ (1034 + 100 * C) % 4 = 0 :=
by sorry

end digits_C_not_make_1C34_divisible_by_4_l1949_194963


namespace find_b_l1949_194938

theorem find_b (a b : ℝ) (h₁ : 2 * a + 3 = 5) (h₂ : b - a = 2) : b = 3 :=
by 
  sorry

end find_b_l1949_194938


namespace markup_percentage_l1949_194914

theorem markup_percentage 
  (CP : ℝ) (x : ℝ) (MP : ℝ) (SP : ℝ) 
  (h1 : CP = 100)
  (h2 : MP = CP + (x / 100) * CP)
  (h3 : SP = MP - (10 / 100) * MP)
  (h4 : SP = CP + (35 / 100) * CP) :
  x = 50 :=
by sorry

end markup_percentage_l1949_194914


namespace regular_polygon_sides_l1949_194907

theorem regular_polygon_sides (ratio : ℕ) (interior exterior : ℕ) (sum_angles : ℕ) 
  (h1 : ratio = 5)
  (h2 : interior = 5 * exterior)
  (h3 : interior + exterior = sum_angles)
  (h4 : sum_angles = 180) : 

∃ (n : ℕ), n = 12 := 
by 
  sorry

end regular_polygon_sides_l1949_194907


namespace trig_inequality_sin_cos_l1949_194935

theorem trig_inequality_sin_cos :
  Real.sin 2 + Real.cos 2 + 2 * (Real.sin 1 - Real.cos 1) ≥ 1 :=
by
  sorry

end trig_inequality_sin_cos_l1949_194935


namespace solve_system_unique_solution_l1949_194971

theorem solve_system_unique_solution:
  ∃! (x y : ℚ), 3 * x - 4 * y = -7 ∧ 4 * x + 5 * y = 23 ∧ x = 57 / 31 ∧ y = 97 / 31 := by
  sorry

end solve_system_unique_solution_l1949_194971


namespace total_amount_received_l1949_194955

def initial_price_tv : ℕ := 500
def tv_increase_rate : ℚ := 2 / 5
def initial_price_phone : ℕ := 400
def phone_increase_rate : ℚ := 0.40

theorem total_amount_received :
  initial_price_tv + initial_price_tv * tv_increase_rate + initial_price_phone + initial_price_phone * phone_increase_rate = 1260 :=
by
  sorry

end total_amount_received_l1949_194955


namespace ratio_proof_l1949_194997

variables (x y m n : ℝ)

def ratio_equation1 (x y m n : ℝ) : Prop :=
  (5 * x + 7 * y) / (3 * x + 2 * y) = m / n

def target_equation (x y m n : ℝ) : Prop :=
  (13 * x + 16 * y) / (2 * x + 5 * y) = (2 * m + n) / (m - n)

theorem ratio_proof (x y m n : ℝ) (h: ratio_equation1 x y m n) :
  target_equation x y m n :=
by
  sorry

end ratio_proof_l1949_194997


namespace taxi_fare_80_miles_l1949_194952

theorem taxi_fare_80_miles (fare_60 : ℝ) (flat_rate : ℝ) (proportional_rate : ℝ) (d : ℝ) (charge_60 : ℝ) 
  (h1 : fare_60 = 150) (h2 : flat_rate = 20) (h3 : proportional_rate * 60 = charge_60) (h4 : charge_60 = (fare_60 - flat_rate)) 
  (h5 : proportional_rate * 80 = d - flat_rate) : d = 193 := 
by
  sorry

end taxi_fare_80_miles_l1949_194952


namespace find_f_2010_l1949_194905

def f (x : ℝ) : ℝ := sorry

theorem find_f_2010 (h₁ : ∀ x, f (x + 1) = - f x) (h₂ : f 1 = 4) : f 2010 = -4 :=
by 
  sorry

end find_f_2010_l1949_194905


namespace integer_solutions_count_l1949_194927

theorem integer_solutions_count (B : ℤ) (C : ℤ) (h : B = 3) : C = 4 :=
by
  sorry

end integer_solutions_count_l1949_194927


namespace percentage_of_l1949_194991

theorem percentage_of (part whole : ℕ) (h_part : part = 120) (h_whole : whole = 80) : 
  ((part : ℚ) / (whole : ℚ)) * 100 = 150 := 
by
  sorry

end percentage_of_l1949_194991


namespace radius_of_circle_nearest_integer_l1949_194995

theorem radius_of_circle_nearest_integer (θ L : ℝ) (hθ : θ = 300) (hL : L = 2000) : 
  abs ((1200 / (Real.pi)) - 382) < 1 := 
by {
  sorry
}

end radius_of_circle_nearest_integer_l1949_194995


namespace negation_of_universal_proposition_l1949_194959
open Classical

theorem negation_of_universal_proposition :
  (¬ (∀ x : ℝ, x > 1 → x^2 ≥ 3)) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := 
by
  sorry

end negation_of_universal_proposition_l1949_194959


namespace no_valid_rectangles_l1949_194966

theorem no_valid_rectangles 
  (a b x y : ℝ) (h_ab_lt : a < b) (h_xa_lt : x < a) (h_ya_lt : y < a) 
  (h_perimeter : 2 * (x + y) = (2 * (a + b)) / 3) 
  (h_area : x * y = (a * b) / 3) : false := 
sorry

end no_valid_rectangles_l1949_194966


namespace solve_for_a_l1949_194977

theorem solve_for_a {a x : ℝ} (H : (x - 2) * (a * x^2 - x + 1) = a * x^3 + (-1 - 2 * a) * x^2 + 3 * x - 2 ∧ (-1 - 2 * a) = 0) : a = -1/2 := sorry

end solve_for_a_l1949_194977


namespace parabola_directrix_l1949_194982

theorem parabola_directrix (x y : ℝ) (h : y = 4 * x^2) : y = -1 / 16 :=
sorry

end parabola_directrix_l1949_194982


namespace original_cost_price_l1949_194978

theorem original_cost_price (C : ℝ) (h : C + 0.15 * C + 0.05 * C + 0.10 * C = 6400) : C = 4923 :=
by
  sorry

end original_cost_price_l1949_194978


namespace systematic_sampling_fourth_group_number_l1949_194993

theorem systematic_sampling_fourth_group_number (n : ℕ) (step_size : ℕ) (first_number : ℕ) : 
  n = 4 → step_size = 6 → first_number = 4 → (first_number + step_size * 3) = 22 :=
by
  intros h_n h_step_size h_first_number
  sorry

end systematic_sampling_fourth_group_number_l1949_194993


namespace simplify_and_evaluate_div_fraction_l1949_194923

theorem simplify_and_evaluate_div_fraction (a : ℤ) (h : a = -3) : 
  (a - 2) / (1 + 2 * a + a^2) / (a - 3 * a / (a + 1)) = 1 / 6 := by
  sorry

end simplify_and_evaluate_div_fraction_l1949_194923


namespace square_area_from_conditions_l1949_194976

theorem square_area_from_conditions :
  ∀ (r s l b : ℝ), 
  l = r / 4 →
  r = s →
  l * b = 35 →
  b = 5 →
  s^2 = 784 := 
by 
  intros r s l b h1 h2 h3 h4
  sorry

end square_area_from_conditions_l1949_194976


namespace least_two_multiples_of_15_gt_450_l1949_194979

-- Define a constant for the base multiple
def is_multiple_of_15 (n : ℕ) : Prop :=
  ∃ k : ℕ, n = 15 * k

-- Define a constant for being greater than 450
def is_greater_than_450 (n : ℕ) : Prop :=
  n > 450

-- Two least positive multiples of 15 greater than 450
theorem least_two_multiples_of_15_gt_450 :
  (is_multiple_of_15 465 ∧ is_greater_than_450 465 ∧
   is_multiple_of_15 480 ∧ is_greater_than_450 480) :=
by
  sorry

end least_two_multiples_of_15_gt_450_l1949_194979


namespace MF1_dot_MF2_range_proof_l1949_194945

noncomputable def MF1_dot_MF2_range : Set ℝ :=
  Set.Icc (24 - 16 * Real.sqrt 3) (24 + 16 * Real.sqrt 3)

theorem MF1_dot_MF2_range_proof :
  ∀ (M : ℝ × ℝ), (Prod.snd M + 4) ^ 2 + (Prod.fst M) ^ 2 = 12 →
    (Prod.fst M) ^ 2 + (Prod.snd M) ^ 2 - 4 ∈ MF1_dot_MF2_range :=
by
  sorry

end MF1_dot_MF2_range_proof_l1949_194945


namespace fraction_of_positive_number_l1949_194908

theorem fraction_of_positive_number (x : ℝ) (f : ℝ) (h : x = 0.4166666666666667 ∧ f * x = (25/216) * (1/x)) : f = 2/3 :=
sorry

end fraction_of_positive_number_l1949_194908


namespace time_per_potato_l1949_194974

-- Definitions from the conditions
def total_potatoes : ℕ := 12
def cooked_potatoes : ℕ := 6
def remaining_potatoes : ℕ := total_potatoes - cooked_potatoes
def total_time : ℕ := 36
def remaining_time_per_potato : ℕ := total_time / remaining_potatoes

-- Theorem to be proved
theorem time_per_potato : remaining_time_per_potato = 6 := by
  sorry

end time_per_potato_l1949_194974


namespace emma_total_investment_l1949_194912

theorem emma_total_investment (X : ℝ) (h : 0.09 * 6000 + 0.11 * (X - 6000) = 980) : X = 10000 :=
sorry

end emma_total_investment_l1949_194912


namespace tetrahedron_inequality_l1949_194936

variables (S A B C : Point)
variables (SA SB SC : Real)
variables (ABC : Plane)
variables (z : Real)
variable (h1 : angle B S C = π / 2)
variable (h2 : Project (point S) ABC = Orthocenter triangle ABC)
variable (h3 : RadiusInscribedCircle triangle ABC = z)

theorem tetrahedron_inequality :
  SA^2 + SB^2 + SC^2 >= 18 * z^2 :=
sorry

end tetrahedron_inequality_l1949_194936


namespace return_trip_time_l1949_194996

variable (d p w_1 w_2 : ℝ)
variable (t t' : ℝ)
variable (h1 : d / (p - w_1) = 120)
variable (h2 : d / (p + w_2) = t - 10)
variable (h3 : t = d / p)

theorem return_trip_time :
  t' = 72 :=
by
  sorry

end return_trip_time_l1949_194996


namespace no_integer_solutions_l1949_194980

theorem no_integer_solutions : ¬∃ (x y : ℤ), 15 * x^2 - 7 * y^2 = 9 := 
by
  sorry

end no_integer_solutions_l1949_194980


namespace smallest_possible_value_l1949_194944

theorem smallest_possible_value (x : ℕ) (m : ℕ) :
  (x > 0) →
  (Nat.gcd 36 m = x + 3) →
  (Nat.lcm 36 m = x * (x + 3)) →
  m = 12 :=
by
  sorry

end smallest_possible_value_l1949_194944


namespace sum_n_k_eq_eight_l1949_194961

-- Define the binomial coefficient function
def binom (n k : ℕ) : ℕ := Nat.choose n k

-- Define the theorem to prove that n + k = 8 given the conditions
theorem sum_n_k_eq_eight {n k : ℕ} 
  (h1 : binom n k * 3 = binom n (k + 1))
  (h2 : binom n (k + 1) * 5 = binom n (k + 2) * 3) : n + k = 8 := by
  sorry

end sum_n_k_eq_eight_l1949_194961


namespace quadratic_is_perfect_square_l1949_194926

theorem quadratic_is_perfect_square (a b c x : ℝ) (h : b^2 - 4 * a * c = 0) :
  a * x^2 + b * x + c = 0 ↔ (2 * a * x + b)^2 = 0 := 
by
  sorry

end quadratic_is_perfect_square_l1949_194926


namespace part1_part2_l1949_194919

/-- Given a triangle ABC with sides opposite to angles A, B, C being a, b, c respectively,
and a sin A sin B + b cos^2 A = 5/3 a,
prove that (1) b / a = 5/3. -/
theorem part1 (a b : ℝ) (A B : ℝ) (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a) :
  b / a = 5 / 3 :=
sorry

/-- Given the previous result b / a = 5/3 and the condition c^2 = a^2 + 8/5 b^2,
prove that (2) angle C = 2π / 3. -/
theorem part2 (a b c : ℝ) (A B C : ℝ)
  (h₁ : a * (Real.sin A) * (Real.sin B) + b * (Real.cos A) ^ 2 = (5 / 3) * a)
  (h₂ : c^2 = a^2 + (8 / 5) * b^2)
  (h₃ : b / a = 5 / 3) :
  C = 2 * Real.pi / 3 :=
sorry

end part1_part2_l1949_194919


namespace slope_tangent_line_l1949_194939

variable {f : ℝ → ℝ}

-- Assumption: f is differentiable
def differentiable_at (f : ℝ → ℝ) (x : ℝ) := ∃ f', ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ → |(f (x + h) - f x) / h - f'| < ε

-- Hypothesis: limit condition
axiom limit_condition : (∀ x, differentiable_at f (1 - x)) → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε)

-- Theorem: the slope of the tangent line to the curve y = f(x) at (1, f(1)) is -2
theorem slope_tangent_line : differentiable_at f 1 → (∀ ε > 0, ∃ δ > 0, ∀ Δx > 0, |Δx| < δ → |(f 1 - f (1 - Δx)) / (2 * Δx) + 1| < ε) → deriv f 1 = -2 :=
by
    intro h_diff h_lim
    sorry

end slope_tangent_line_l1949_194939


namespace prove_total_weekly_allowance_l1949_194947

noncomputable def total_weekly_allowance : ℕ :=
  let students := 200
  let group1 := students * 45 / 100
  let group2 := students * 30 / 100
  let group3 := students * 15 / 100
  let group4 := students - group1 - group2 - group3  -- Remaining students
  let daily_allowance := group1 * 6 + group2 * 4 + group3 * 7 + group4 * 10
  daily_allowance * 7

theorem prove_total_weekly_allowance :
  total_weekly_allowance = 8330 := by
  sorry

end prove_total_weekly_allowance_l1949_194947


namespace math_problem_l1949_194911

theorem math_problem (a b c m n : ℝ)
  (h1 : a = -b)
  (h2 : c = -1)
  (h3 : m * n = 1) : 
  (a + b) / 3 + c^2 - 4 * m * n = -3 := 
by 
  -- Proof steps would be here
  sorry

end math_problem_l1949_194911


namespace probability_divisible_by_8_l1949_194990

-- Define the problem conditions
def is_8_sided_die (n : ℕ) : Prop := n = 6
def roll_dice (m : ℕ) : Prop := m = 8

-- Define the main proof statement
theorem probability_divisible_by_8 (n m : ℕ) (hn : is_8_sided_die n) (hm : roll_dice m) :  
  (35 : ℚ) / 36 = 
  (1 - ((1/2) ^ m + 28 * ((1/n) ^ 2 * ((1/2) ^ 6))) : ℚ) :=
by
  sorry

end probability_divisible_by_8_l1949_194990


namespace unique_solution_of_functional_eqn_l1949_194913

theorem unique_solution_of_functional_eqn (f : ℝ → ℝ) :
  (∀ x y : ℝ, f (x * y - 1) + f x * f y = 2 * x * y - 1) → (∀ x : ℝ, f x = x) :=
by
  intros h
  sorry

end unique_solution_of_functional_eqn_l1949_194913


namespace tan_of_acute_angle_l1949_194900

open Real

theorem tan_of_acute_angle (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : 2 * sin (α - 15 * π / 180) - 1 = 0) : tan α = 1 :=
by
  sorry

end tan_of_acute_angle_l1949_194900


namespace bacon_sold_l1949_194946

variable (B : ℕ) -- Declare the variable for the number of slices of bacon sold

-- Define the given conditions as Lean definitions
def pancake_price := 4
def bacon_price := 2
def stacks_sold := 60
def total_raised := 420

-- The revenue from pancake sales alone
def pancake_revenue := stacks_sold * pancake_price
-- The revenue from bacon sales
def bacon_revenue := total_raised - pancake_revenue

-- Statement of the theorem
theorem bacon_sold :
  B = bacon_revenue / bacon_price :=
sorry

end bacon_sold_l1949_194946


namespace venus_hall_meal_cost_l1949_194999

theorem venus_hall_meal_cost (V : ℕ) :
  let caesars_total_cost := 800 + 30 * 60;
  let venus_hall_total_cost := 500 + V * 60;
  caesars_total_cost = venus_hall_total_cost → V = 35 :=
by
  let caesars_total_cost := 800 + 30 * 60
  let venus_hall_total_cost := 500 + V * 60
  intros h
  sorry

end venus_hall_meal_cost_l1949_194999


namespace point_in_quadrant_l1949_194934

theorem point_in_quadrant (m n : ℝ) (h₁ : 2 * (m - 1)^2 - 7 = -5) (h₂ : n > 3) :
  (m = 0 → 2*m - 3 < 0 ∧ (3*n - m)/2 > 0) ∧ 
  (m = 2 → 2*m - 3 > 0 ∧ (3*n - m)/2 > 0) :=
by 
  sorry

end point_in_quadrant_l1949_194934


namespace class_5_matches_l1949_194906

theorem class_5_matches (matches_c1 matches_c2 matches_c3 matches_c4 matches_c5 : ℕ)
  (C1 : matches_c1 = 2)
  (C2 : matches_c2 = 4)
  (C3 : matches_c3 = 4)
  (C4 : matches_c4 = 3) :
  matches_c5 = 3 :=
sorry

end class_5_matches_l1949_194906


namespace hyperbola_eccentricity_range_l1949_194956

theorem hyperbola_eccentricity_range 
(a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
(hyperbola_eq : ∀ x y, x^2 / a^2 - y^2 / b^2 = 1)
(parabola_eq : ∀ y x, y^2 = 8 * a * x)
(right_vertex : A = (a, 0))
(focus : F = (2 * a, 0))
(P : ℝ × ℝ)
(asymptote_eq : P = (x0, b / a * x0))
(perpendicular_condition : (x0 ^ 2 - (3 * a - b^2 / a^2) * x0 + 2 * a^2 = 0))
(hyperbola_properties: c^2 = a^2 + b^2) :
1 < c / a ∧ c / a <= 3 * Real.sqrt 2 / 4 :=
sorry

end hyperbola_eccentricity_range_l1949_194956


namespace problem1_problem2a_problem2b_problem2c_l1949_194970

theorem problem1 {x : ℝ} : 3 * x ^ 2 - 5 * x - 2 < 0 → -1 / 3 < x ∧ x < 2 :=
sorry

theorem problem2a {x a : ℝ} (ha : -1 / 2 < a ∧ a < 0) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x < 2 ∨ x > -1 / a :=
sorry

theorem problem2b {x a : ℝ} (ha : a = -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x ≠ 2 :=
sorry

theorem problem2c {x a : ℝ} (ha : a < -1 / 2) : 
  ax * x^2 + (1 - 2 * a) * x - 2 < 0 → x > 2 ∨ x < -1 / a :=
sorry

end problem1_problem2a_problem2b_problem2c_l1949_194970
